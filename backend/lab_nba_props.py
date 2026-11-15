"""
lab_nba_props.py — NBA Lab: player prop projections vs market odds.

Runs the NBAMatchupEngine (B3) + GameSimulator (B5) live for today's games,
joins with captured market odds from historical.db:player_props, and computes
edge = sim P(over) - devigged market P(over).

Does NOT read from scouted_props (bets.db may be down). All projections are
computed on-demand from the engine.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from math import erf, sqrt
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# ── Team abbreviation maps ────────────────────────────────────────────────────

_FULL_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

_ABBR_TO_FULL = {v: k for k, v in _FULL_TO_ABBR.items()}

# Market key → stat attribute on PlayerProjection
_MARKET_STAT = {
    "player_points": ("points", "pts_std"),
    "player_rebounds": ("rebounds", "reb_std"),
    "player_assists": ("assists", "ast_std"),
    "player_threes": ("threes", "thr_std"),
    "player_steals": ("steals", None),
    "player_blocks": ("blocks", None),
    "player_points_rebounds_assists": ("pra", "pra_std"),
}

_MARKET_LABELS = {
    "player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST",
    "player_threes": "3PM", "player_steals": "STL", "player_blocks": "BLK",
    "player_points_rebounds_assists": "PRA",
    "player_points_alternate": "PTS alt", "player_rebounds_alternate": "REB alt",
    "player_assists_alternate": "AST alt",
}


# ── Normal CDF for P(over) ───────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _p_over(mu: float, sigma: float, threshold: float) -> float:
    """P(X > threshold) assuming Normal(mu, sigma)."""
    if sigma <= 0:
        return 1.0 if mu > threshold else 0.0
    z = (threshold - mu) / sigma
    return 1.0 - _norm_cdf(z)


# ── Historical DB helpers ─────────────────────────────────────────────────────

def _hist_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_HIST_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _get_todays_games(game_date: str | None = None) -> list[dict]:
    """Get today's NBA games from player_props capture."""
    conn = _hist_conn()
    if not game_date:
        # Use most recent game_date with NBA data
        row = conn.execute("""
            SELECT MAX(game_date) FROM player_props
            WHERE sport_key = 'basketball_nba'
        """).fetchone()
        game_date = row[0] if row else None
    if not game_date:
        conn.close()
        return []

    rows = conn.execute("""
        SELECT DISTINCT event_id, home_team, away_team, game_date
        FROM player_props
        WHERE sport_key = 'basketball_nba' AND game_date = ?
    """, (game_date,)).fetchall()

    # Look up tip times from injury forward log (game_time_et per team per date)
    tip_times: dict[str, str] = {}  # team_full -> "HH:MM (ET)"
    try:
        tip_rows = conn.execute("""
            SELECT DISTINCT team, game_time_et
            FROM nba_injury_forward_log
            WHERE game_date = ? AND game_time_et IS NOT NULL
        """, (game_date,)).fetchall()
        for tr in tip_rows:
            tip_times[tr["team"]] = tr["game_time_et"]
    except Exception:
        pass
    conn.close()

    games = []
    for r in rows:
        home_full = r["home_team"] or ""
        away_full = r["away_team"] or ""
        games.append({
            "event_id": r["event_id"],
            "home_team": home_full,
            "away_team": away_full,
            "home_abbr": _FULL_TO_ABBR.get(home_full, home_full[:3].upper()),
            "away_abbr": _FULL_TO_ABBR.get(away_full, away_full[:3].upper()),
            "game_date": r["game_date"],
            "game_time_et": tip_times.get(home_full) or tip_times.get(away_full),
        })
    return games


def _get_market_odds(event_id: str) -> list[dict]:
    """Get latest devigged market odds for an event."""
    conn = _hist_conn()
    # Get all prop lines for this event (most recent fetch per combo)
    prop_rows = conn.execute("""
        SELECT p1.*
        FROM   player_props p1
        JOIN   (SELECT bookmaker, market_key, player_name, over_under, point,
                       MAX(fetched_at) AS max_fa
                FROM   player_props
                WHERE  event_id = ?
                GROUP  BY bookmaker, market_key, player_name, over_under, point
               ) p2
          ON   p1.bookmaker   = p2.bookmaker
         AND   p1.market_key  = p2.market_key
         AND   p1.player_name = p2.player_name
         AND   p1.over_under  = p2.over_under
         AND   COALESCE(p1.point, -999) = COALESCE(p2.point, -999)
         AND   p1.fetched_at  = p2.max_fa
        WHERE  p1.event_id = ?
        ORDER  BY p1.player_name, p1.market_key, p1.point
    """, (event_id, event_id)).fetchall()
    conn.close()

    if not prop_rows:
        return []

    # Group by (player, market, point, book) for devig
    pairs: dict[tuple, dict] = defaultdict(dict)
    for r in prop_rows:
        key = (r["player_name"], r["market_key"], r["point"], r["bookmaker"])
        pairs[key][r["over_under"]] = {
            "price": r["price"],
            "fetched_at": r["fetched_at"],
        }

    # Find best price per (player, market, point, over_under) across books
    best: dict[tuple, dict] = {}
    for (player, mkt, point, book), sides in pairs.items():
        for ou, info in sides.items():
            pkey = (player, mkt, point, ou)
            price = info.get("price") or 0
            if pkey not in best or price > (best[pkey].get("best_price") or 0):
                best[pkey] = {
                    "player_name": player, "market_key": mkt, "point": point,
                    "over_under": ou, "best_price": price,
                    "best_book": book, "fetched_at": info.get("fetched_at"),
                }

    # Two-sided devig (over_under is lowercase in DB: "over"/"under")
    devigged: dict[tuple, dict] = {}
    for (player, mkt, point, book), sides in pairs.items():
        if "over" in sides and "under" in sides:
            p_o = sides["over"]["price"]
            p_u = sides["under"]["price"]
            if p_o and p_u and p_o > 1.0 and p_u > 1.0:
                ip_o, ip_u = 1.0 / p_o, 1.0 / p_u
                total = ip_o + ip_u
                for ou, dv in [("over", ip_o / total), ("under", ip_u / total)]:
                    dkey = (player, mkt, point, ou, book)
                    devigged[dkey] = {"devig_prob": dv, "method": "two_sided"}

    # Build output — best price with devigged prob
    results = []
    for pkey, binfo in best.items():
        player, mkt, point, ou = pkey
        # Find best devig across books
        best_dv = None
        for (_, _, _, book), _ in pairs.items():
            dkey = (player, mkt, point, ou, book)
            if dkey in devigged:
                dv = devigged[dkey]["devig_prob"]
                best_dv = dv if best_dv is None else (best_dv + dv) / 2

        bp = binfo.get("best_price")
        if best_dv is None and bp and bp > 1.0:
            best_dv = 1.0 / bp  # raw implied

        row = dict(binfo)
        row["devig_prob"] = round(best_dv, 6) if best_dv is not None else None
        results.append(row)

    return results


def _get_market_lines(game_date: str | None = None) -> dict:
    """Get spread + total for today's NBA games from historical.db betting_lines or fixtures."""
    conn = _hist_conn()
    if not game_date:
        row = conn.execute("""
            SELECT MAX(game_date) FROM player_props
            WHERE sport_key = 'basketball_nba'
        """).fetchone()
        game_date = row[0] if row else None

    # Try betting_lines (ESPN close lines for historical, live for current)
    lines = {}
    if game_date:
        rows = conn.execute("""
            SELECT home_team, away_team, close_spread, close_total
            FROM betting_lines
            WHERE sport = 'NBA' AND game_date LIKE ?
            LIMIT 50
        """, (game_date + "%",)).fetchall()
        for r in rows:
            key = (r["home_team"] or "", r["away_team"] or "")
            lines[key] = {
                "spread": r["close_spread"],
                "total": r["close_total"],
            }
    conn.close()
    return lines


# ── Engine interface ──────────────────────────────────────────────────────────

_engine = None          # None = not started, False = failed, object = ready
_engine_loading = False  # True while background thread is running
import threading
_engine_lock = threading.Lock()


def _load_engine_bg():
    """Background loader — runs once at import time."""
    global _engine, _engine_loading
    try:
        from nba_matchup_engine import NBAMatchupEngine
        eng = NBAMatchupEngine(cache=True)
        eng.load()
        with _engine_lock:
            _engine = eng
            _engine_loading = False
        logger.info(f"[nba_lab] Engine loaded: {len(eng.player_profiles)} players")
    except Exception as e:
        with _engine_lock:
            _engine = False
            _engine_loading = False
        logger.warning(f"[nba_lab] Engine unavailable: {e}")


def _get_engine():
    """Return engine if ready, None if still loading or failed."""
    global _engine, _engine_loading
    with _engine_lock:
        if _engine is None and not _engine_loading:
            # Start background load
            _engine_loading = True
            t = threading.Thread(target=_load_engine_bg, daemon=True)
            t.start()
        return _engine if _engine not in (None, False) else None


def _engine_status() -> str:
    """Return human-readable engine status."""
    with _engine_lock:
        if _engine_loading:
            return "loading"
        if _engine is False:
            return "failed"
        if _engine is None:
            return "not_started"
        return "ready"


def _run_projection(home_abbr: str, away_abbr: str,
                    market_total: float = 0.0,
                    market_spread: float = 0.0) -> dict | None:
    """Run engine projection for a game. Returns {player_name: proj_dict}."""
    engine = _get_engine()
    if not engine:
        return None
    try:
        # Fetch injury exclusions
        exclude = set()
        if not hasattr(engine, "_injury_report"):
            engine.fetch_injury_report()
        exclude = engine.get_excluded_players()
        # Filter to this game's teams
        game_exclude = {n for n in exclude
                        if engine._injury_report.get(n, {}).get("team")
                        in (home_abbr, away_abbr)}

        game_proj = engine.project_game(
            home_abbr, away_abbr,
            market_total=market_total,
            market_spread=market_spread,
            exclude_players=game_exclude or None,
        )
        result = {}
        for name, pp in game_proj.players.items():
            result[name] = {
                "points": pp.points, "rebounds": pp.rebounds,
                "assists": pp.assists, "steals": pp.steals,
                "blocks": pp.blocks, "threes": pp.threes,
                "pra": pp.pra, "minutes": pp.minutes,
                "pts_std": pp.pts_std, "reb_std": pp.reb_std,
                "ast_std": pp.ast_std, "thr_std": pp.thr_std,
                "pra_std": pp.pra_std,
                "team": pp.team if hasattr(pp, "team") else "",
                "role": pp.role if hasattr(pp, "role") else "",
            }
        return result
    except Exception as e:
        logger.warning(f"[nba_lab] Projection failed {away_abbr}@{home_abbr}: {e}")
        return None


def _run_simulation(home_abbr: str, away_abbr: str,
                    market_total: float = 0.0,
                    market_spread: float = 0.0,
                    hours_to_tip: float | None = None):
    """Run B5 joint-draw simulation. Returns SimulationResult or None."""
    engine = _get_engine()
    if not engine:
        return None
    try:
        exclude = set()
        if hasattr(engine, "_injury_report"):
            exclude = engine.get_excluded_players()
            exclude = {n for n in exclude
                       if engine._injury_report.get(n, {}).get("team")
                       in (home_abbr, away_abbr)}

        injury_statuses = {}
        if hasattr(engine, "_injury_report"):
            for pname, info in engine._injury_report.items():
                if info.get("team") in (home_abbr, away_abbr):
                    status = info.get("status")
                    if status and pname not in exclude:
                        injury_statuses[pname] = status

        game_proj = engine.project_game(
            home_abbr, away_abbr,
            market_total=market_total,
            market_spread=market_spread,
            exclude_players=exclude or None,
        )

        absence_impacts = None
        try:
            absence_impacts = engine.get_absence_impacts_for_game(
                game_proj, home_abbr, away_abbr
            ) or None
        except Exception:
            pass

        from nba_game_sim import GameSimulator
        sim = GameSimulator(
            game_proj,
            injury_statuses=injury_statuses or None,
            absence_impacts=absence_impacts,
            hours_to_tip=hours_to_tip,
        )
        return sim.run(n_draws=10_000)
    except Exception as e:
        logger.warning(f"[nba_lab] Sim failed {away_abbr}@{home_abbr}: {e}")
        return None


# ── Main builder ──────────────────────────────────────────────────────────────

def build_nba_lab_props(game_date: str | None = None) -> dict:
    """Build NBA Lab props view for a given date.

    Returns:
        {
            "game_date": "2026-11-01",
            "games": [{home, away, home_abbr, away_abbr, spread, total}],
            "players": [{player_name, team, role, minutes, stats: [...]}],
            "injuries": {out: [], questionable: []},
            "updated_at": "...",
            "engine_loaded": bool,
            "n_games": int,
        }
    """
    games = _get_todays_games(game_date)
    if not games:
        return {"error": "No NBA games found", "game_date": game_date,
                "games": [], "players": []}

    actual_date = games[0]["game_date"] if games else game_date

    # Get market lines (spread/total) for anchoring
    market_lines = _get_market_lines(actual_date)

    engine = _get_engine()
    all_players = []
    all_injuries_out = []
    all_injuries_q = []
    game_summaries = []

    for game in games:
        home_abbr = game["home_abbr"]
        away_abbr = game["away_abbr"]
        home_full = game["home_team"]
        away_full = game["away_team"]

        # Compute hours_to_tip from game_time_et
        hours_to_tip = None
        gt = game.get("game_time_et")  # e.g. "07:30 (ET)"
        if gt and actual_date:
            try:
                from datetime import datetime, timezone, timedelta
                import re as _re
                m = _re.match(r"(\d{1,2}):(\d{2})", gt)
                if m:
                    h, mn = int(m.group(1)), int(m.group(2))
                    tip_et = datetime.strptime(actual_date, "%Y-%m-%d").replace(
                        hour=h, minute=mn, tzinfo=timezone(timedelta(hours=-5)))
                    now_utc = datetime.now(timezone.utc)
                    hours_to_tip = max((tip_et - now_utc).total_seconds() / 3600, 0.0)
            except Exception:
                pass

        # Find spread/total — try multiple key formats
        spread, total = 0.0, 0.0
        for key in [(home_full, away_full), (home_abbr, away_abbr)]:
            if key in market_lines:
                spread = float(market_lines[key].get("spread") or 0)
                total = float(market_lines[key].get("total") or 0)
                break

        # If no betting_lines, try to infer total from player_props
        # (not critical — engine still runs with total=0, just no market anchor)

        game_summaries.append({
            "home": home_abbr, "away": away_abbr,
            "home_full": home_full, "away_full": away_full,
            "spread": spread, "total": total,
            "event_id": game["event_id"],
            "hours_to_tip": hours_to_tip,
        })

        # Run engine projection
        projs = _run_projection(
            home_abbr, away_abbr,
            market_total=total, market_spread=spread,
        )

        # Get captured market odds for this event
        market_odds = _get_market_odds(game["event_id"])

        # Index market odds by (player_name, market_key, point, over_under)
        mkt_by_key: dict[tuple, dict] = {}
        for mo in market_odds:
            k = (mo["player_name"], mo["market_key"],
                 mo.get("point"), mo.get("over_under"))
            mkt_by_key[k] = mo

        # Also build set of all (player, market, point) combos with market lines
        mkt_lines: dict[tuple, list] = defaultdict(list)
        for mo in market_odds:
            if (mo.get("over_under") or "").lower() == "over":
                mkt_lines[(mo["player_name"], mo["market_key"])].append({
                    "point": mo.get("point"),
                    "best_price": mo.get("best_price"),
                    "best_book": mo.get("best_book"),
                    "devig_prob": mo.get("devig_prob"),
                })

        # Build player rows
        if projs:
            for pname, proj in projs.items():
                stats_rows = []
                team = proj.get("team", "")

                for mkt_key, (stat_attr, std_attr) in _MARKET_STAT.items():
                    mu = proj.get(stat_attr, 0)
                    sigma = proj.get(std_attr, 0) if std_attr else 0
                    if mu <= 0:
                        continue
                    # If no std available, estimate from CV
                    if sigma <= 0:
                        cv_map = {"steals": 0.8, "blocks": 0.9}
                        cv = cv_map.get(stat_attr, 0.3)
                        sigma = mu * cv

                    # Find market lines for this player/stat
                    lines = mkt_lines.get((pname, mkt_key), [])
                    # Also check alternate market
                    alt_key = mkt_key + "_alternate"
                    alt_lines = mkt_lines.get((pname, alt_key), [])

                    if lines:
                        # Main lines — take the one closest to the projection
                        for line in lines:
                            threshold = line.get("point")
                            if threshold is None:
                                continue
                            model_p_over = _p_over(mu, sigma, threshold)
                            mkt_p_over = line.get("devig_prob")
                            edge = None
                            if mkt_p_over and mkt_p_over > 0:
                                edge = model_p_over - mkt_p_over

                            stats_rows.append({
                                "market_key": mkt_key,
                                "label": _MARKET_LABELS.get(mkt_key, mkt_key),
                                "threshold": threshold,
                                "model_mean": round(mu, 1),
                                "model_std": round(sigma, 2),
                                "model_p_over": round(model_p_over, 4),
                                "market_p_over": round(mkt_p_over, 4) if mkt_p_over else None,
                                "edge_pp": round(edge * 100, 1) if edge is not None else None,
                                "best_price": line.get("best_price"),
                                "best_book": line.get("best_book"),
                                "gap": round(mu - threshold, 1),
                                "line_depth": "main",
                            })

                    if alt_lines:
                        for line in alt_lines:
                            threshold = line.get("point")
                            if threshold is None:
                                continue
                            model_p_over = _p_over(mu, sigma, threshold)
                            mkt_p_over = line.get("devig_prob")
                            edge = None
                            if mkt_p_over and mkt_p_over > 0:
                                edge = model_p_over - mkt_p_over

                            stats_rows.append({
                                "market_key": alt_key,
                                "label": _MARKET_LABELS.get(alt_key, mkt_key + " alt"),
                                "threshold": threshold,
                                "model_mean": round(mu, 1),
                                "model_std": round(sigma, 2),
                                "model_p_over": round(model_p_over, 4),
                                "market_p_over": round(mkt_p_over, 4) if mkt_p_over else None,
                                "edge_pp": round(edge * 100, 1) if edge is not None else None,
                                "best_price": line.get("best_price"),
                                "best_book": line.get("best_book"),
                                "gap": round(mu - threshold, 1),
                                "line_depth": "alt",
                            })

                    if not lines and not alt_lines:
                        # No market lines — still show projection
                        stats_rows.append({
                            "market_key": mkt_key,
                            "label": _MARKET_LABELS.get(mkt_key, mkt_key),
                            "threshold": None,
                            "model_mean": round(mu, 1),
                            "model_std": round(sigma, 2),
                            "model_p_over": None,
                            "market_p_over": None,
                            "edge_pp": None,
                            "best_price": None,
                            "best_book": None,
                            "gap": None,
                            "line_depth": "no_line",
                        })

                if stats_rows:
                    all_players.append({
                        "player_name": pname,
                        "team": team or home_abbr,
                        "role": proj.get("role", ""),
                        "minutes": round(proj.get("minutes", 0), 1),
                        "game": f"{away_abbr}@{home_abbr}",
                        "stats": stats_rows,
                    })

    # Collect injury info
    if engine and hasattr(engine, "_injury_report"):
        for pname, info in engine._injury_report.items():
            status = info.get("status", "")
            entry = {
                "name": pname,
                "team": info.get("team", ""),
                "status": status,
                "body_part": info.get("body_part", ""),
                "source": info.get("source", ""),
            }
            if status in ("Out", "OFS", "IR"):
                all_injuries_out.append(entry)
            elif status in ("Questionable", "Doubtful", "GTD", "DTD", "Probable"):
                all_injuries_q.append(entry)

    # Sort players by minutes (descending) within each game
    all_players.sort(key=lambda p: (-p.get("minutes", 0),))

    return {
        "game_date": actual_date,
        "games": game_summaries,
        "players": all_players,
        "injuries": {
            "out": all_injuries_out,
            "questionable": all_injuries_q,
        },
        "n_games": len(games),
        "n_players": len(all_players),
        "engine_loaded": engine is not None and engine is not False,
        "engine_status": _engine_status(),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_version": "B5_sim_v1",
        "note": "UNCALIBRATED — forward test started 2026-10-30. "
                "P(over) from Normal CDF on engine projections.",
    }
