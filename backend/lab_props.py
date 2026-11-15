"""
lab_props.py — NFL Lab: player props with v1.2 projections, injury adjustments, weather.

Reads stored v1.2 predictions from fantasy_forward_preds, enriches with:
  - Injury status from Sleeper (via injury_watcher)
  - Redistributed projections when key players are Out
  - Weather badges for outdoor stadiums
  - Both base and adjusted edges vs market

Does NOT modify stored v1.2 predictions or the pre-registered tracker.
Lab-adjusted projections are stored separately with model_version='v1.2_lab'.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from math import exp, log, sqrt
from typing import Any

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")
_NFL_DB = os.path.join(_DATA_DIR, "nfl.db")

# ── Team abbreviation ↔ full name ────────────────────────────────────────────

_ABBREV_TO_FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# Stats we project
_STAT_LABELS = {
    "rec_yds": "Rec Yards", "rush_yds": "Rush Yards", "pass_yds": "Pass Yards",
    "receptions": "Receptions", "pass_tds": "Pass TDs",
    "rush_att": "Rush Att", "pass_att": "Pass Att", "completions": "Completions",
}

# Reverse mapping: Odds API market_key → our canonical stat_type
_MARKET_TO_STAT = {
    "player_pass_yds": "pass_yds",
    "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yds",
    "player_rush_attempts": "rush_attempts",
    "player_reception_yds": "rec_yds",
    "player_receptions": "receptions",
    # Alternates map to same stat
    "player_pass_yds_alternate": "pass_yds",
    "player_rush_yds_alternate": "rush_yds",
    "player_reception_yds_alternate": "rec_yds",
    "player_receptions_alternate": "receptions",
}

# Stats where calibration is unverified (count stats — v1.2 trained on yardage)
_CALIBRATION_UNVERIFIED = {"receptions", "pass_tds"}

# ── Injury redistribution helpers ────────────────────────────────────────────


def _load_sleeper_injuries() -> dict[str, dict]:
    """Load current Sleeper injury statuses for NFL skill players.

    Returns: {player_name: {status, team, position, body_part, news_updated_ms}}
    """
    try:
        from injury_watcher import _fetch_sleeper_players, _SKILL_POSITIONS
        players = _fetch_sleeper_players()
        if not players:
            return {}
        result = {}
        for pid, pinfo in players.items():
            pos = pinfo.get("position", "")
            if pos not in _SKILL_POSITIONS:
                continue
            status = pinfo.get("injury_status")
            if not status:
                continue
            name = pinfo.get("full_name", "?")
            team = pinfo.get("team") or "?"
            result[name] = {
                "status": status,
                "team": team,
                "position": pos,
                "body_part": pinfo.get("injury_body_part"),
                "news_updated_ms": pinfo.get("news_updated") or 0,
            }
        return result
    except Exception as e:
        logger.warning("[LabProps] Sleeper fetch failed: %s", e)
        return {}


# Position-aware absorption weights: when a player at position X goes out,
# what fraction of their freed receiving volume goes to each absorbing position?
# Derived from NFL target redistribution research (Sharp Football, PFF).
# rush_yds handled separately — almost all goes to other RBs + QB scrambles.
_RECEIVING_ABSORPTION = {
    # absent_pos → {absorber_pos: weight}
    "WR": {"WR": 0.55, "TE": 0.25, "RB": 0.20},
    "TE": {"WR": 0.40, "TE": 0.40, "RB": 0.20},
    "RB": {"WR": 0.30, "TE": 0.25, "RB": 0.45},
    "QB": {"WR": 0.40, "TE": 0.30, "RB": 0.30},
}
_RUSH_ABSORPTION = {
    # absent_pos → {absorber_pos: weight}
    "RB": {"RB": 0.75, "QB": 0.20, "WR": 0.05},
    "QB": {"RB": 0.80, "QB": 0.00, "WR": 0.20},
    "WR": {"RB": 0.80, "QB": 0.15, "WR": 0.05},
    "TE": {"RB": 0.85, "QB": 0.10, "WR": 0.05},
}


def _redistribute_projections(
    preds_by_team: dict[str, list[dict]],
    inactive: dict[str, dict],
) -> tuple[dict[str, list[dict]], dict]:
    """Apply injury redistribution to projections.

    Only redistributes when the injury announcement (news_updated) is NEWER
    than the prediction's created_at — otherwise the consensus already
    reflects the absence and redistributing would double-count.

    Uses position-aware absorption weights instead of uniform proportional
    redistribution: WR targets go mostly to other WRs, rush yards go
    mostly to other RBs, etc.

    Returns (adjusted_preds_by_team, redistribution_report).
    """
    _REDIST_STATS = {"rec_yds", "receptions", "rush_yds"}
    adjusted = {}
    report = {"redistributed": [], "skipped_stale": [], "skipped_no_preds": []}

    for team, preds in preds_by_team.items():
        team_out = {n: info for n, info in inactive.items()
                    if info["team"] == team
                    and info["status"] in ("Out", "IR")}

        if not team_out:
            adjusted[team] = preds
            continue

        # Group predictions by player
        by_player = defaultdict(list)
        for p in preds:
            by_player[p["player_name"]].append(p)

        # Determine which Out/IR players should trigger redistribution:
        # Only if (a) they have predictions AND (b) injury news is newer
        # than the prediction timestamp.
        eligible_inactive = {}  # {name: info} — only those triggering redist
        for out_name, out_info in team_out.items():
            if out_name not in by_player:
                report["skipped_no_preds"].append(
                    f"{out_name} ({team} {out_info['status']})")
                continue

            # Compare injury news timestamp vs prediction created_at
            news_ms = out_info.get("news_updated_ms", 0)
            player_preds = by_player[out_name]
            # Use the latest prediction timestamp for this player
            latest_pred_ts = None
            for pp in player_preds:
                ca = pp.get("created_at")
                if ca:
                    try:
                        dt = datetime.fromisoformat(ca.replace("Z", "+00:00"))
                        ts_ms = int(dt.timestamp() * 1000)
                        if latest_pred_ts is None or ts_ms > latest_pred_ts:
                            latest_pred_ts = ts_ms
                    except (ValueError, TypeError):
                        pass

            if latest_pred_ts and news_ms <= latest_pred_ts:
                # Injury was known BEFORE the prediction was created.
                # Consensus should already reflect the absence.
                report["skipped_stale"].append(
                    f"{out_name} ({team} {out_info['status']}): "
                    f"injury news before consensus pull")
                continue

            eligible_inactive[out_name] = out_info

        if not eligible_inactive:
            # No eligible absences — pass through unchanged
            # But still remove Out players who have predictions (they can't play)
            new_preds = []
            for pname, ppreds in by_player.items():
                if pname in team_out:
                    continue  # still remove Out players from display
                new_preds.extend(ppreds)
            adjusted[team] = new_preds
            continue

        # Compute freed volume per (stat, absent_position)
        # freed_by_stat_pos[stat] = [(freed_mean, absent_position, player_name)]
        freed_by_stat_pos = defaultdict(list)
        # Active players: {stat: {player: (mean, position)}}
        active_by_stat = defaultdict(dict)

        # Get position for each player in by_player
        player_positions = {}
        for pname, ppreds in by_player.items():
            pos = ppreds[0].get("position", "")
            if not pos:
                # Try from inactive info
                info = eligible_inactive.get(pname) or team_out.get(pname)
                pos = info.get("position", "") if info else ""
            player_positions[pname] = pos

        for pname, ppreds in by_player.items():
            # Deduplicate: one main-line entry per (player, stat)
            seen = set()
            for p in ppreds:
                st = p["stat_type"]
                if st not in _REDIST_STATS:
                    continue
                key = (pname, st)
                if key in seen:
                    continue
                if p.get("line_depth") != "main":
                    continue
                seen.add(key)

                mean = p["consensus_mean"]
                pos = player_positions.get(pname, "")
                if pname in eligible_inactive:
                    freed_by_stat_pos[st].append((mean, pos, pname))
                elif pname not in team_out:
                    active_by_stat[st][pname] = (mean, pos)

        # Log what got redistributed
        inactive_by_stat = defaultdict(list)
        for st, freed_list in freed_by_stat_pos.items():
            names = [f[2] for f in freed_list]
            total = sum(f[0] for f in freed_list)
            inactive_by_stat[st] = names
            report["redistributed"].append(
                f"{team} {st}: {total:.1f} freed from "
                f"{', '.join(names)}")

        # Pre-compute per-active-player boost for each stat.
        # For each freed entry, distribute using position absorption weights,
        # then within each position bucket distribute proportionally by mean.
        player_boost = defaultdict(float)  # {(player, stat): total_boost}

        for st, freed_list in freed_by_stat_pos.items():
            actives = active_by_stat.get(st, {})
            if not actives:
                continue

            # Group active players by position
            pos_groups = defaultdict(dict)  # {pos: {player: mean}}
            for aplayer, (amean, apos) in actives.items():
                pos_groups[apos][aplayer] = amean

            for freed_mean, absent_pos, absent_name in freed_list:
                # Pick absorption weights based on stat type and absent position
                if st == "rush_yds":
                    weights = _RUSH_ABSORPTION.get(
                        absent_pos, {"RB": 0.80, "QB": 0.15, "WR": 0.05})
                else:
                    weights = _RECEIVING_ABSORPTION.get(
                        absent_pos, {"WR": 0.45, "TE": 0.30, "RB": 0.25})

                for absorb_pos, weight in weights.items():
                    bucket = pos_groups.get(absorb_pos, {})
                    bucket_total = sum(bucket.values())
                    if bucket_total <= 0:
                        continue
                    alloc = freed_mean * weight
                    for aplayer, amean in bucket.items():
                        share = amean / bucket_total
                        player_boost[(aplayer, st)] += alloc * share

        # Build adjusted predictions
        new_preds = []
        for pname, ppreds in by_player.items():
            if pname in team_out:
                continue  # remove all Out players from display

            for p in ppreds.copy():
                p = dict(p)
                st = p["stat_type"]
                boost = player_boost.get((pname, st), 0)

                if boost > 0:
                    p["adjusted_mean"] = round(p["consensus_mean"] + boost, 1)
                    p["adjusted_p_over"] = _recalc_p_over(
                        p["adjusted_mean"], p.get("consensus_sd", 0),
                        p.get("prop_line", 0), st,
                        position=p.get("position", ""))
                    contributors = inactive_by_stat.get(st, [])
                    p["adjustment_source"] = ", ".join(
                        f"{n} ({eligible_inactive[n]['status']})"
                        for n in contributors)
                    p["adjustment_targets"] = round(boost, 1)
                else:
                    p["adjusted_mean"] = p["consensus_mean"]
                    # Decision layer: prefer gamma for yardage
                    p["adjusted_p_over"] = (
                        p.get("p_over_v12_gamma") or p.get("p_over")
                    )

                new_preds.append(p)

        adjusted[team] = new_preds

    return adjusted, report


def _compute_gtd_scenarios(
    preds_by_team: dict[str, list[dict]],
    questionable: dict[str, dict],
) -> dict[str, dict]:
    """Compute hypothetical 'if out' boosts for each GTD player.

    For each GTD player who has predictions, computes what redistribution
    boosts teammates would receive if the GTD player is ruled Out.

    Returns:
    {
      gtd_player_name: {
        "team": str,
        "position": str,
        "status": str,
        "freed": {stat_type: freed_mean},
        "boosts": {teammate_name: {stat_type: boost_amount}},
      }
    }
    """
    _REDIST_STATS = {"rec_yds", "receptions", "rush_yds"}
    scenarios = {}

    for team, preds in preds_by_team.items():
        team_gtd = {n: info for n, info in questionable.items()
                    if info["team"] == team}
        if not team_gtd:
            continue

        # Group preds by player
        by_player = defaultdict(list)
        for p in preds:
            by_player[p["player_name"]].append(p)

        # Get position for each player
        player_positions = {}
        for pname, ppreds in by_player.items():
            pos = ppreds[0].get("position", "")
            if not pos:
                info = team_gtd.get(pname)
                pos = info.get("position", "") if info else ""
            player_positions[pname] = pos

        for gtd_name, gtd_info in team_gtd.items():
            if gtd_name not in by_player:
                continue  # No predictions for this GTD player

            gtd_pos = player_positions.get(gtd_name, gtd_info.get("position", ""))

            # Compute freed volume for this GTD player
            freed = {}
            for pp in by_player[gtd_name]:
                st = pp["stat_type"]
                if st not in _REDIST_STATS:
                    continue
                if pp.get("line_depth") != "main":
                    continue
                if st in freed:
                    continue
                freed[st] = pp.get("consensus_mean", 0)

            if not freed:
                continue

            # Compute active teammates' means per stat
            active_by_stat = defaultdict(dict)
            for pname, ppreds in by_player.items():
                if pname == gtd_name:
                    continue
                seen = set()
                for pp in ppreds:
                    st = pp["stat_type"]
                    if st not in _REDIST_STATS or st in seen:
                        continue
                    if pp.get("line_depth") != "main":
                        continue
                    seen.add(st)
                    active_by_stat[st][pname] = (
                        pp.get("consensus_mean", 0),
                        player_positions.get(pname, ""),
                    )

            # Compute boosts using position-aware absorption
            boosts = defaultdict(dict)  # {teammate: {stat: boost}}
            for st, freed_mean in freed.items():
                actives = active_by_stat.get(st, {})
                if not actives:
                    continue

                pos_groups = defaultdict(dict)
                for aplayer, (amean, apos) in actives.items():
                    pos_groups[apos][aplayer] = amean

                if st == "rush_yds":
                    weights = _RUSH_ABSORPTION.get(
                        gtd_pos, {"RB": 0.80, "QB": 0.15, "WR": 0.05})
                else:
                    weights = _RECEIVING_ABSORPTION.get(
                        gtd_pos, {"WR": 0.45, "TE": 0.30, "RB": 0.25})

                for absorb_pos, weight in weights.items():
                    bucket = pos_groups.get(absorb_pos, {})
                    bucket_total = sum(bucket.values())
                    if bucket_total <= 0:
                        continue
                    alloc = freed_mean * weight
                    for aplayer, amean in bucket.items():
                        share = amean / bucket_total
                        boosts[aplayer][st] = round(alloc * share, 1)

            scenarios[gtd_name] = {
                "team": team,
                "position": gtd_pos,
                "status": gtd_info["status"],
                "freed": {st: round(v, 1) for st, v in freed.items()},
                "boosts": dict(boosts),
            }

    return scenarios


def _recalc_p_over(mean: float, sd: float, line: float,
                   stat_type: str, position: str = "") -> float | None:
    """Recalculate P(over) — gamma for yardage (decision layer), NegBin for counts.

    Yardage uses shifted-gamma CDF (better calibrated, Brier 0.167 vs 0.209).
    Stored p_over in DB stays lognormal (frozen for FW-2 prereg); this function
    is for the Lab/display/decision layer only.
    """
    if mean <= 0 or line <= 0:
        return None
    try:
        if stat_type in ("receptions", "pass_tds"):
            # NegBin approximation
            if sd <= 0:
                sd = mean * 0.5
            var = sd ** 2
            if var <= mean:
                var = mean * 1.1
            r = (mean ** 2) / (var - mean)
            p_nb = r / (r + mean)
            # P(X > line) via normal approximation to NegBin
            z = (line + 0.5 - mean) / sqrt(var)
            return round(max(0, min(1, 1 - _normal_cdf(z))), 4)
        else:
            # Yardage: shifted gamma (decision layer)
            from nfl_interim_dispersion import (
                get_interim_cv, get_v12_base_rate,
                p_over_line_gamma_shifted,
                load_cached_params, load_v12_base_rates,
            )
            load_cached_params()
            load_v12_base_rates()
            cv = get_interim_cv(stat_type, position, mean, use_v12=True)
            br = get_v12_base_rate(stat_type, position)
            return round(p_over_line_gamma_shifted(line, mean, cv, br), 4)
    except Exception:
        return None


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    from math import erf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ── Weather ──────────────────────────────────────────────────────────────────


def _fetch_game_weather(games: list[dict]) -> dict[str, dict]:
    """Fetch weather for each game.

    Returns: {game_id: weather_dict}
    """
    from weather import get_game_weather
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    results = {}

    for g in games:
        game_id = g["game_id"]
        home = g["home_team"]
        # Build commence_time from game_date + gametime (ET)
        try:
            kick_naive = datetime.strptime(
                f"{g['game_date']} {g['gametime']}", "%Y-%m-%d %H:%M")
            kick_et = kick_naive.replace(tzinfo=et)
            kick_utc = kick_et.astimezone(timezone.utc)
            commence = kick_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            full_name = _ABBREV_TO_FULL.get(home, home)
            weather = get_game_weather(full_name, commence, sport="nfl")
            results[game_id] = weather
        except Exception as e:
            logger.warning("[LabProps] Weather fetch failed for %s: %s",
                           game_id, e)
            results[game_id] = {"error": str(e)}

    return results


# ── Raw prop line loader ─────────────────────────────────────────────────────


def _load_raw_nfl_props(game_dates: list[str]) -> dict[str, list[dict]]:
    """Load all NFL player props from historical.db for given game dates.

    Returns: {player_name: [{stat_type, prop_line, over_price, under_price,
                             market_key, line_depth, home_team, away_team}]}
    Only includes main-line props (non-alternate markets), deduplicated
    to best book per (player, market).
    """
    if not game_dates or not os.path.exists(os.path.abspath(_HIST_DB)):
        return {}

    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(game_dates))
    try:
        rows = conn.execute(f"""
            SELECT player_name, market_key, point, price, over_under,
                   bookmaker, home_team, away_team, game_date, fetched_at
            FROM player_props
            WHERE sport_key = 'americanfootball_nfl'
              AND game_date IN ({placeholders})
            ORDER BY fetched_at DESC
        """, game_dates).fetchall()
    except Exception as exc:
        logger.warning("[LabProps] Raw prop query failed: %s", exc)
        return {}
    finally:
        conn.close()

    # Load alias table for name resolution
    from fantasy_projections import _resolve_nfl_name, _normalize_player_name

    # Deduplicate: most recent per (player, market, over_under) for main lines
    seen = set()
    # Key by canonical (prediction-side) name so _has_pred matching works
    by_player: dict[str, dict[str, dict]] = {}  # {canonical_name: {stat_type: info}}
    for row in rows:
        pname_raw = row["player_name"]
        mkey = row["market_key"]
        ou = row["over_under"]

        # Skip alternates for the "no projection" display — too noisy
        if mkey.endswith("_alternate"):
            continue

        stat_type = _MARKET_TO_STAT.get(mkey)
        if not stat_type:
            continue

        dedup_key = (pname_raw, mkey, ou)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Resolve to canonical name (matches prediction-side names)
        canonical = _resolve_nfl_name(pname_raw)
        # Use display name = the raw Odds API name (more recognizable)
        # but key by the prediction-stored name for matching
        # Try to find the prediction-stored form: check if any prediction
        # uses a name that normalizes to this canonical
        pname = pname_raw  # default display name

        if canonical not in by_player:
            by_player[canonical] = {}
        if stat_type not in by_player[canonical]:
            by_player[canonical][stat_type] = {
                "stat_type": stat_type,
                "market_key": mkey,
                "prop_line": row["point"],
                "line_depth": "main",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "display_name": pname_raw,
            }
        entry = by_player[canonical][stat_type]
        if ou == "over":
            entry["over_price"] = row["price"]
        else:
            entry["under_price"] = row["price"]

    # Convert to list format, keyed by display name
    result: dict[str, list[dict]] = {}
    for canonical, stats in by_player.items():
        # Use the first entry's display_name as the player key
        display = next(iter(stats.values())).get("display_name", canonical)
        result[display] = list(stats.values())
    return result


# ── Matchup Context Loader ──────────────────────────────────────────────────


def _load_matchup_context(season: int, week: int, games: list[dict]) -> dict:
    """Load persisted matchup context for all games in this week.

    Returns per-game dict with team verified stats, QB-vs-blitz summary,
    separation estimates, CB-on-WR proxy, and script/volume.
    Display only — never modifies v1.2 predictions.
    """
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    result = {}
    game_ids = [g["game_id"] for g in games]
    if not game_ids:
        conn.close()
        return result

    placeholders = ",".join("?" for _ in game_ids)

    # Team context
    team_rows = conn.execute(f"""
        SELECT game_id, team, side,
               pass_epa_play, rush_epa_play, pass_success_rate, rush_success_rate,
               blitz_rate, heavy_rush_rate, sack_rate, pressure_rate, rz_td_rate,
               pass_epa_rank, rush_epa_rank, blitz_rate_rank, sack_rate_rank,
               pressure_rank, rz_td_rank, n_games
        FROM nfl_matchup_team_context
        WHERE game_id IN ({placeholders})
    """, game_ids).fetchall()

    # Player context
    player_rows = conn.execute(f"""
        SELECT game_id, player_name, team, position,
               qb_blitz_epa, qb_clean_epa, qb_blitz_comp_pct, qb_clean_comp_pct,
               qb_blitz_adot, qb_clean_adot, qb_blitz_sack_rate,
               ngs_avg_separation, ngs_avg_cushion, expected_separation,
               expected_sep_ci_lo, expected_sep_ci_hi,
               primary_cb_name, primary_cb_prob, cb_evidence_n,
               shadow_risk_flag
        FROM nfl_matchup_player_context
        WHERE game_id IN ({placeholders})
    """, game_ids).fetchall()

    conn.close()

    # Organize by game
    for gid in game_ids:
        game_ctx = {"teams": {}, "players": []}

        for row in team_rows:
            if row["game_id"] != gid:
                continue
            team = row["team"]
            side = row["side"]
            if team not in game_ctx["teams"]:
                game_ctx["teams"][team] = {}
            game_ctx["teams"][team][side] = dict(row)

        for row in player_rows:
            if row["game_id"] != gid:
                continue
            game_ctx["players"].append(dict(row))

        if game_ctx["teams"] or game_ctx["players"]:
            result[gid] = game_ctx

    return result


# ── Main Lab Props Builder ───────────────────────────────────────────────────


def build_lab_props(season: int | None = None,
                    week: int | None = None,
                    include_weather: bool = True) -> dict:
    """Build the full Lab props view for a given week.

    Returns:
    {
        "season": int,
        "week": int,
        "games": [{game_id, home, away, kickoff_et, weather, ...}],
        "players": [{
            player_name, position, team, opponent, game_id,
            stats: [{
                stat_type, label, prop_line, over_price, under_price,
                consensus_mean, p_over, devigged_implied, edge_vs_market,
                adjusted_mean, adjusted_p_over, adjusted_edge,
                line_depth, model_version,
            }],
            injury_status, injury_body_part, is_gtd,
            weather_badge, flags,
        }],
        "coverage": {n_games, n_players_with_props, n_players_with_preds, gaps},
        "injuries": {out: [...], questionable: [...]},
        "updated_at": str,
    }
    """
    from fantasy_projections import get_current_nfl_week
    if season is None or week is None:
        season, week = get_current_nfl_week()

    # ── 1. Load game schedule ──
    conn_nfl = sqlite3.connect(os.path.abspath(_NFL_DB))
    conn_nfl.row_factory = sqlite3.Row
    games = conn_nfl.execute("""
        SELECT game_id, game_date, gametime, weekday, home_team, away_team,
               spread_line, total_line, roof
        FROM nfl_games
        WHERE season = ? AND week = ? AND game_type = 'REG'
        ORDER BY game_date, gametime
    """, (season, week)).fetchall()
    conn_nfl.close()

    games = [dict(g) for g in games]
    if not games:
        return {"season": season, "week": week, "games": [],
                "players": [], "error": "No games found"}

    # Build team → game lookup
    team_game = {}
    for g in games:
        team_game[g["home_team"]] = g
        team_game[g["away_team"]] = g

    # ── 2. Load stored v1.2 predictions ──
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    v12_tag = "v1.2_shifted_gamma"
    # Only load distributional method (skip mean_direction duplicates)
    _METHOD_FILTER = "AND method LIKE '%distributional%'"
    # Load latest predictions: v1.2 OR NULL model_version (post-R3 pipeline
    # output doesn't set model_version yet; these are the most recent rows).
    # Prefer the most recent created_at per (player, stat, line).
    preds_rows = conn.execute(f"""
        SELECT player_name, position, stat_type, stat_category,
               consensus_mean, consensus_sd, prop_line,
               prop_over_price, prop_under_price,
               p_over, p_over_v12_gamma, devigged_implied, edge_vs_market, edge_gamma, dist_direction_gamma,
               direction, z_score, line_depth, var_deviation,
               game_id, model_version, created_at
        FROM fantasy_forward_preds
        WHERE season = ? AND week = ?
              AND (model_version = ? OR model_version IS NULL)
              {_METHOD_FILTER}
        ORDER BY player_name, stat_type
    """, (season, week, v12_tag)).fetchall()

    # If nothing found, fall back to v0
    fallback_version = None
    if not preds_rows:
        fallback_version = "v0"
        preds_rows = conn.execute(f"""
            SELECT player_name, position, stat_type, stat_category,
                   consensus_mean, consensus_sd, prop_line,
                   prop_over_price, prop_under_price,
                   p_over, p_over_v12_gamma, devigged_implied, edge_vs_market, edge_gamma, dist_direction_gamma,
                   direction, z_score, line_depth, var_deviation,
                   game_id, model_version, created_at
            FROM fantasy_forward_preds
            WHERE season = ? AND week = ? AND model_version = ?
                  {_METHOD_FILTER}
            ORDER BY player_name, stat_type
        """, (season, week, fallback_version)).fetchall()

    conn.close()

    # Convert to dicts and dedup main lines
    preds = [dict(r) for r in preds_rows]

    # Dedup: when multiple rows are tagged 'main' for the same (player, stat),
    # keep the one whose prop_line is closest to the consensus mean (the model's
    # actual projection). This anchors on the most relevant line — the one
    # closest to what the model projects — rather than an arbitrary median of
    # lines from different books. Relabel the rest as 'book_alt'.
    from statistics import median as _median
    _main_groups = defaultdict(list)  # (player, stat) → [indices]
    for i, p in enumerate(preds):
        if p.get("line_depth") == "main":
            _main_groups[(p["player_name"], p["stat_type"])].append(i)
    for key, indices in _main_groups.items():
        if len(indices) <= 1:
            continue
        # Use median of consensus means as anchor (robust to outlier books)
        means = [preds[i].get("consensus_mean") or 0 for i in indices]
        anchor = _median(means)
        # Keep the line closest to the consensus anchor
        best_idx = min(indices,
                       key=lambda i: (abs((preds[i].get("prop_line") or 0) - anchor),
                                      preds[i].get("prop_line") or 0))
        for i in indices:
            if i != best_idx:
                preds[i]["line_depth"] = "book_alt"

    preds_by_team = defaultdict(list)

    # Need to figure out each player's team from game_id or from the predictions
    # game_id format: '2026_06_SEA_DEN' → away=SEA, home=DEN
    player_team = {}
    for p in preds:
        gid = p.get("game_id") or ""
        # Try to extract team from game_id
        parts = gid.split("_")
        if len(parts) >= 4:
            away, home = parts[2], parts[3]
            # Check which team this player belongs to by looking at team_game
            # We'll need Sleeper data or store team in predictions
            # For now, use the stored position + game to figure it out
            pass

    # Build player → team and player → position from Sleeper (most reliable)
    _team_lookup = {}
    _pos_lookup = {}
    try:
        from injury_watcher import _fetch_sleeper_players
        sleeper = _fetch_sleeper_players()
        for pid, pinfo in sleeper.items():
            name = pinfo.get("full_name", "")
            team = pinfo.get("team", "")
            pos = pinfo.get("position", "")
            if name and team:
                _team_lookup[name] = team
            if name and pos:
                _pos_lookup[name] = pos
    except Exception:
        pass

    # Fallback: extract team from game_id if possible
    if not _team_lookup:
        for p in preds:
            gid = p.get("game_id") or ""
            parts = gid.split("_")
            if len(parts) >= 4:
                # Can't determine which team without more info
                pass

    for p in preds:
        team = _team_lookup.get(p["player_name"], "")
        p["team"] = team
        if team:
            preds_by_team[team].append(p)

    # ── 2b. Load raw prop lines (for prop-only players) ──
    game_dates = sorted({g["game_date"] for g in games})
    raw_props = _load_raw_nfl_props(game_dates)

    # Build set of normalized player×stat that already have predictions
    from fantasy_projections import _resolve_nfl_name, _normalize_player_name
    _has_pred_norm = set()
    for p in preds:
        _has_pred_norm.add((_resolve_nfl_name(p["player_name"]), p["stat_type"]))

    # Identify prop-only entries: players/stats with a posted line but no prediction
    prop_only_by_team: dict[str, list[dict]] = defaultdict(list)
    all_prop_players = set()
    for pname, prop_list in raw_props.items():
        all_prop_players.add(pname)
        pname_resolved = _resolve_nfl_name(pname)
        for prop in prop_list:
            if (pname_resolved, prop["stat_type"]) in _has_pred_norm:
                continue  # already covered by predictions
            team = _team_lookup.get(pname, "")
            if not team:
                # Try alias-resolved name for team lookup
                for tname, tteam in _team_lookup.items():
                    if _normalize_player_name(tname) == pname_resolved:
                        team = tteam
                        break
            prop["player_name"] = pname
            prop["team"] = team
            prop["no_projection"] = True
            if team:
                prop_only_by_team[team].append(prop)

    # ── 3. Load injury data ──
    injuries_raw = _load_sleeper_injuries()

    out_players = {n: info for n, info in injuries_raw.items()
                   if info["status"] in ("Out", "IR")}
    questionable_players = {n: info for n, info in injuries_raw.items()
                            if info["status"] in ("Questionable", "Doubtful")}

    # ── 4. Apply redistribution (timestamp-gated) ──
    adjusted_by_team, redist_report = _redistribute_projections(
        preds_by_team, out_players)

    # ── 4b. Compute GTD scenarios ──
    gtd_scenarios = _compute_gtd_scenarios(adjusted_by_team, questionable_players)

    # ── 5. Weather ──
    weather_data = {}
    if include_weather:
        weather_data = _fetch_game_weather(games)

    # ── 6. Build output ──
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")

    game_output = []
    for g in games:
        kick_naive = datetime.strptime(
            f"{g['game_date']} {g['gametime']}", "%Y-%m-%d %H:%M")
        kick_et = kick_naive.replace(tzinfo=et)

        w = weather_data.get(g["game_id"], {})
        weather_badge = None
        if w and not w.get("is_dome") and not w.get("error"):
            parts = []
            wind = w.get("wind_speed_mph")
            precip = w.get("precipitation_prob_pct")
            temp = w.get("temperature_f")
            if wind is not None and wind >= 10:
                parts.append(f"Wind {wind:.0f} mph")
            if precip is not None and precip >= 30:
                parts.append(f"Rain {precip:.0f}%")
            if temp is not None and temp <= 35:
                parts.append(f"{temp:.0f}°F")
            weather_badge = " | ".join(parts) if parts else None

        # Item 5: full roof type and forecast status
        # Static lookup for stadiums with retractable roofs
        _RETRACTABLE = {"ARI", "ATL", "DAL", "HOU", "IND"}
        _DOME = {"LA", "LAC", "LV", "MIN", "NO"}

        roof_raw = g.get("roof") or ""
        home = g["home_team"]
        if roof_raw == "dome" or (not roof_raw and home in _DOME):
            roof_label = "dome"
        elif roof_raw == "closed":
            roof_label = "retractable (closed)"
        elif roof_raw == "open" and home in _RETRACTABLE:
            roof_label = "retractable (open)"
        elif roof_raw == "open" or (not roof_raw and home not in _DOME
                                     and home not in _RETRACTABLE):
            roof_label = "open"
        elif not roof_raw and home in _RETRACTABLE:
            roof_label = "retractable"
        else:
            roof_label = roof_raw or "open"

        forecast_status = "fetched" if (w and not w.get("error")) else (
            "dome (skipped)" if w and w.get("is_dome") else
            f"error: {w.get('error', 'unknown')}" if w else "not fetched")

        game_output.append({
            "game_id": g["game_id"],
            "home": g["home_team"],
            "away": g["away_team"],
            "kickoff_et": kick_et.strftime("%a %b %d %I:%M %p"),
            "kickoff_sort": kick_et.strftime("%Y-%m-%d %H:%M"),
            "spread": g.get("spread_line"),
            "total": g.get("total_line"),
            "roof": roof_label,
            "weather": w if w and not w.get("is_dome") else None,
            "weather_badge": weather_badge,
            "is_dome": w.get("is_dome", False) if w else None,
            "forecast_status": forecast_status,
            "weather_multiplier_applied": False,  # Stage 1: flag only
        })

    # Build player output
    players_output = []
    players_with_props = set()
    players_with_preds = set()
    out_with_preds = []

    for team, team_preds in adjusted_by_team.items():
        game = team_game.get(team)
        if not game:
            continue

        opponent = (game["away_team"] if game["home_team"] == team
                    else game["home_team"])

        # Group by player
        by_player = defaultdict(list)
        for p in team_preds:
            by_player[p["player_name"]].append(p)
            players_with_preds.add(p["player_name"])
            if p.get("prop_line"):
                players_with_props.add(p["player_name"])

        for pname, ppreds in by_player.items():
            inj = injuries_raw.get(pname)
            inj_status = inj["status"] if inj else None
            inj_body = inj["body_part"] if inj else None
            is_gtd = inj_status in ("Questionable", "Doubtful")

            # Check if this is an Out player who shouldn't have preds
            if inj_status in ("Out", "IR"):
                out_with_preds.append(pname)
                continue

            stats = []
            for p in ppreds:
                base_edge = p.get("edge_vs_market")
                # Decision layer: prefer gamma for yardage, then lognormal
                adj_p_over = p.get("adjusted_p_over",
                                   p.get("p_over_v12_gamma") or p.get("p_over"))
                adj_edge = None
                if adj_p_over is not None and p.get("devigged_implied") is not None:
                    adj_edge = round(adj_p_over - p["devigged_implied"], 4)

                flags = []
                if p.get("adjustment_source"):
                    flags.append(f"Injury redistribution: {p['adjustment_source']} "
                                 f"(+{p.get('adjustment_targets', 0):.1f})")
                if p.get("location_uncertain"):
                    flags.append("Location uncertain")

                # Item 3: Badge v0 predictions
                mv = p.get("model_version") or fallback_version or v12_tag
                is_v0 = mv == "v0"
                if is_v0:
                    flags.append("v0 – overconfident")

                # Item 4: Flag |edge| > 15pp as likely model error (skip v0)
                if not is_v0 and adj_edge is not None and abs(adj_edge) > 0.15:
                    flags.append("large disagreement – likely model error")

                # Calibration unverified badge for count stats
                if p["stat_type"] in _CALIBRATION_UNVERIFIED:
                    flags.append("calibration unverified")

                # Item 4: Badge pre-registered rule matches
                # rec_yds OVER, v1.2, edge >= 5pp, odds > -150 (price >= 1.667)
                is_prereg = (
                    p["stat_type"] == "rec_yds"
                    and mv.startswith("v1.2")
                    and (adj_edge or 0) >= 0.05
                    and (p.get("prop_over_price") or 0) >= 1.667
                )
                if is_prereg:
                    flags.append("tracked rule")

                # Item 7: Compute median from fitted distribution
                c_mean = p.get("consensus_mean") or 0
                c_sd = p.get("consensus_sd") or 0
                median = None
                if c_mean > 0:
                    if p["stat_type"] in ("receptions", "pass_tds"):
                        # NegBin: median ≈ mean (slightly below for skewed)
                        median = round(c_mean - 0.33, 1) if c_mean > 1 else round(c_mean, 1)
                    else:
                        # Lognormal: median = exp(mu) where mu = log(mean) - sigma²/2
                        if c_sd > 0:
                            cv = c_sd / c_mean
                            sigma2 = log(1 + cv ** 2)
                            median = round(c_mean * exp(-sigma2 / 2), 1)
                        else:
                            median = round(c_mean, 1)

                stats.append({
                    "stat_type": p["stat_type"],
                    "label": _STAT_LABELS.get(p["stat_type"], p["stat_type"]),
                    "prop_line": p.get("prop_line"),
                    "over_price": p.get("prop_over_price"),
                    "under_price": p.get("prop_under_price"),
                    "consensus_mean": p.get("consensus_mean"),
                    "consensus_sd": p.get("consensus_sd"),
                    "median": median,
                    "p_over": adj_p_over,  # decision-layer P(over): gamma for yardage
                    "p_over_lognormal": p.get("p_over"),  # frozen lognormal (FW-2)
                    "p_over_gamma": p.get("p_over_v12_gamma"),  # shifted-gamma
                    "devigged_implied": p.get("devigged_implied"),
                    "edge_base": None if is_v0 else base_edge,  # lognormal edge (frozen)
                    "edge_gamma": None if is_v0 else p.get("edge_gamma"),  # gamma edge (decision layer)
                    "adjusted_mean": p.get("adjusted_mean"),
                    "adjusted_p_over": adj_p_over,
                    "edge_adjusted": None if is_v0 else adj_edge,
                    "direction": p.get("direction"),
                    "z_score": p.get("z_score"),
                    "line_depth": p.get("line_depth"),
                    "model_version": mv,
                    "is_v0": is_v0,
                    "created_at": p.get("created_at"),
                    "flags": flags,
                })

            kick_naive = datetime.strptime(
                f"{game['game_date']} {game['gametime']}", "%Y-%m-%d %H:%M")
            kick_et = kick_naive.replace(tzinfo=et)

            w = weather_data.get(game["game_id"], {})
            w_badge = None
            if w and not w.get("is_dome") and not w.get("error"):
                wind = w.get("wind_speed_mph", 0) or 0
                precip = w.get("precipitation_prob_pct", 0) or 0
                if wind >= 10 or precip >= 30:
                    parts = []
                    if wind >= 10:
                        parts.append(f"Wind {wind:.0f}")
                    if precip >= 30:
                        parts.append(f"Rain {precip:.0f}%")
                    w_badge = " | ".join(parts)

            pos = ppreds[0].get("position", "")

            # Item 6: GTD scenario data
            gtd_info_out = None  # For GTD player: what they free
            gtd_if_out = {}     # For teammates: boosts if GTD is out
            if is_gtd and pname in gtd_scenarios:
                sc = gtd_scenarios[pname]
                gtd_info_out = {
                    "freed": sc["freed"],
                    "n_teammates_affected": len(sc["boosts"]),
                }
            # Check if any GTD player on this team affects this player
            for gtd_name, sc in gtd_scenarios.items():
                if sc["team"] != team:
                    continue
                if pname in sc["boosts"]:
                    boosts = sc["boosts"][pname]
                    gtd_if_out[gtd_name] = {
                        "status": sc["status"],
                        "position": sc["position"],
                        "boosts": boosts,
                    }

            players_output.append({
                "player_name": pname,
                "position": pos,
                "team": team,
                "opponent": opponent,
                "game_id": game["game_id"],
                "kickoff_et": kick_et.strftime("%a %I:%M %p"),
                "kickoff_sort": kick_et.strftime("%Y-%m-%d %H:%M"),
                "stats": stats,
                "injury_status": inj_status,
                "injury_body_part": inj_body,
                "is_gtd": is_gtd,
                "weather_badge": w_badge,
                "gtd_if_out": gtd_if_out if gtd_if_out else None,
                "gtd_self_out": gtd_info_out,
            })

    # ── 6b. Add prop-only players (posted line, no prediction) ──
    _displayed_players = {p["player_name"] for p in players_output}
    for team, prop_entries in prop_only_by_team.items():
        game = team_game.get(team)
        if not game:
            continue
        opponent = (game["away_team"] if game["home_team"] == team
                    else game["home_team"])

        # Group by player
        by_player_po: dict[str, list[dict]] = defaultdict(list)
        for pe in prop_entries:
            by_player_po[pe["player_name"]].append(pe)

        for pname, props_list in by_player_po.items():
            if pname in _displayed_players:
                continue  # already shown via predictions

            inj = injuries_raw.get(pname)
            inj_status = inj["status"] if inj else None
            inj_body = inj["body_part"] if inj else None
            is_gtd = inj_status in ("Questionable", "Doubtful")

            if inj_status in ("Out", "IR"):
                continue  # don't show Out players

            stats = []
            for prop in props_list:
                prop_flags = ["no projection"]
                if prop["stat_type"] in _CALIBRATION_UNVERIFIED:
                    prop_flags.append("calibration unverified")
                stats.append({
                    "stat_type": prop["stat_type"],
                    "label": _STAT_LABELS.get(prop["stat_type"],
                                              prop["stat_type"]),
                    "prop_line": prop.get("prop_line"),
                    "over_price": prop.get("over_price"),
                    "under_price": prop.get("under_price"),
                    "consensus_mean": None,
                    "consensus_sd": None,
                    "median": None,
                    "p_over": None,
                    "devigged_implied": None,
                    "edge_base": None,
                    "adjusted_mean": None,
                    "adjusted_p_over": None,
                    "edge_adjusted": None,
                    "direction": None,
                    "z_score": None,
                    "line_depth": prop.get("line_depth", "main"),
                    "model_version": None,
                    "is_v0": False,
                    "created_at": None,
                    "flags": prop_flags,
                })

            if not stats:
                continue

            # Infer position from Sleeper lookup (already cached)
            pos = _pos_lookup.get(pname, "")

            kick_naive = datetime.strptime(
                f"{game['game_date']} {game['gametime']}", "%Y-%m-%d %H:%M")
            kick_et = kick_naive.replace(tzinfo=et)

            w = weather_data.get(game["game_id"], {})
            w_badge = None
            if w and not w.get("is_dome") and not w.get("error"):
                wind = w.get("wind_speed_mph", 0) or 0
                precip = w.get("precipitation_prob_pct", 0) or 0
                if wind >= 10 or precip >= 30:
                    badge_parts = []
                    if wind >= 10:
                        badge_parts.append(f"Wind {wind:.0f}")
                    if precip >= 30:
                        badge_parts.append(f"Rain {precip:.0f}%")
                    w_badge = " | ".join(badge_parts)

            players_with_props.add(pname)
            players_output.append({
                "player_name": pname,
                "position": pos,
                "team": team,
                "opponent": opponent,
                "game_id": game["game_id"],
                "kickoff_et": kick_et.strftime("%a %I:%M %p"),
                "kickoff_sort": kick_et.strftime("%Y-%m-%d %H:%M"),
                "stats": stats,
                "injury_status": inj_status,
                "injury_body_part": inj_body,
                "is_gtd": is_gtd,
                "weather_badge": w_badge,
                "gtd_if_out": None,
                "gtd_self_out": None,
                "no_projection": True,
            })

    # Default sort by kickoff time, then team, then position
    pos_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    players_output.sort(key=lambda p: (
        p["kickoff_sort"], p["team"],
        pos_order.get(p["position"], 9), p["player_name"]))

    # ── 7. QA checks ──
    qa_issues = []

    # Check: no Out player has projections
    if out_with_preds:
        qa_issues.append({
            "type": "out_player_has_preds",
            "detail": f"{len(out_with_preds)} Out/IR players had predictions "
                      f"(removed): {', '.join(out_with_preds[:5])}",
        })

    # Check: team rec yds vs QB pass yds
    # Ratio is normally 0.5-0.8 since not all receivers have prop lines.
    # Flag only extreme cases: ratio > 1.10 (over-projection) or < 0.25
    for team, team_preds in adjusted_by_team.items():
        # Use only main-line entries to avoid double-counting alt lines
        qb_pass = max(
            (p.get("adjusted_mean", p.get("consensus_mean", 0))
             for p in team_preds
             if p["stat_type"] == "pass_yds"
             and p.get("position") == "QB"
             and p.get("line_depth") == "main"),
            default=0,
        )
        # For rec_yds, take the main-line projection per player (max to dedup)
        rec_by_player = {}
        for p in team_preds:
            if p["stat_type"] != "rec_yds":
                continue
            nm = p["player_name"]
            mean = p.get("adjusted_mean", p.get("consensus_mean", 0))
            if p.get("line_depth") == "main":
                rec_by_player[nm] = mean
            elif nm not in rec_by_player:
                rec_by_player[nm] = mean
        rec_total = sum(rec_by_player.values())
        if qb_pass > 0 and rec_total > 0:
            ratio = rec_total / qb_pass
            if ratio > 1.10:
                qa_issues.append({
                    "type": "rec_exceeds_pass",
                    "team": team,
                    "detail": f"Σ rec_yds ({rec_total:.0f}) EXCEEDS QB pass_yds "
                              f"({qb_pass:.0f}) = {ratio:.2f}x",
                    "severity": "high",
                })
            elif ratio < 0.25:
                qa_issues.append({
                    "type": "rec_vs_pass_low",
                    "team": team,
                    "detail": f"Σ rec_yds ({rec_total:.0f}) very low vs QB pass_yds "
                              f"({qb_pass:.0f}) = {ratio:.2f}x — few receivers have props",
                    "severity": "info",
                })

    # Check: no negative values
    neg_count = 0
    for p in players_output:
        for s in p["stats"]:
            if (s.get("adjusted_mean") or 0) < 0:
                neg_count += 1
                if neg_count <= 3:  # limit detail
                    qa_issues.append({
                        "type": "negative_projection",
                        "player": p["player_name"],
                        "stat": s["stat_type"],
                        "value": s["adjusted_mean"],
                    })
    if neg_count > 3:
        qa_issues.append({
            "type": "negative_projection",
            "detail": f"{neg_count} total negative projections",
        })

    # Check: receptions use NegBin (integer distribution)
    # Flag if any receptions P(over) doesn't match NegBin recalculation
    rec_count_stats = 0
    rec_flagged = 0
    for p in players_output:
        for s in p["stats"]:
            if s["stat_type"] != "receptions":
                continue
            rec_count_stats += 1
            # Verify P(over) is plausible for integer distribution
            mean = s.get("adjusted_mean") or s.get("consensus_mean") or 0
            line = s.get("prop_line") or 0
            if mean > 0 and line > 0:
                recalc = _recalc_p_over(mean, s.get("consensus_sd", 0), line,
                                        "receptions")
                stored = s.get("adjusted_p_over") or s.get("p_over")
                if recalc is not None and stored is not None:
                    diff = abs(recalc - stored)
                    if diff > 0.05 and rec_flagged < 3:
                        rec_flagged += 1
                        qa_issues.append({
                            "type": "receptions_dist_mismatch",
                            "player": p["player_name"],
                            "detail": f"Stored P(over)={stored:.3f} vs "
                                      f"NegBin recalc={recalc:.3f} (Δ={diff:.3f})",
                        })
    if rec_count_stats > 0:
        # Informational: mark receptions calibration status
        qa_issues.append({
            "type": "receptions_calibration_note",
            "detail": f"{rec_count_stats} reception lines — NegBin P(over), "
                      f"calibration unverified (N=41)",
            "severity": "info",
        })

    # ── 7b. Per-game×market coverage QA ──
    # Compare posted prop count vs Lab-covered count per game per stat
    _COVERAGE_ALERT_THRESHOLD = 0.95
    _game_market_coverage = []
    # Build game_id for each team from team_game
    _team_to_game_id = {t: g["game_id"] for t, g in team_game.items()}
    # Count raw props per (game_id, stat_type) from raw_props
    raw_per_game_stat: dict[tuple[str, str], int] = defaultdict(int)
    for pname, prop_list in raw_props.items():
        team = _team_lookup.get(pname, "")
        game_id = _team_to_game_id.get(team, "")
        if not game_id:
            continue
        for prop in prop_list:
            raw_per_game_stat[(game_id, prop["stat_type"])] += 1

    # Count Lab-covered (has projection) per (game_id, stat_type)
    lab_per_game_stat: dict[tuple[str, str], int] = defaultdict(int)
    for p in players_output:
        gid = p.get("game_id", "")
        for s in p["stats"]:
            if "no projection" not in s.get("flags", []):
                lab_per_game_stat[(gid, s["stat_type"])] += 1

    for (game_id, stat), n_posted in raw_per_game_stat.items():
        n_covered = lab_per_game_stat.get((game_id, stat), 0)
        pct = n_covered / n_posted if n_posted > 0 else 1.0
        entry = {
            "game_id": game_id,
            "stat_type": stat,
            "n_posted": n_posted,
            "n_covered": n_covered,
            "coverage_pct": round(pct, 3),
        }
        _game_market_coverage.append(entry)
        if pct < _COVERAGE_ALERT_THRESHOLD:
            qa_issues.append({
                "type": "low_coverage",
                "game_id": game_id,
                "stat_type": stat,
                "detail": f"{stat} coverage {pct:.0%} ({n_covered}/{n_posted}) "
                          f"— below {_COVERAGE_ALERT_THRESHOLD:.0%} threshold",
                "severity": "warning",
            })

    # Coverage gaps — players with posted lines but no projections
    # and players with projections but no posted lines
    props_no_preds = players_with_props - players_with_preds
    preds_no_props = players_with_preds - players_with_props
    # Count v0 vs v1.2 rows and no-projection rows
    v0_count = sum(1 for p in players_output
                   for s in p["stats"] if s.get("is_v0"))
    v12_count = sum(1 for p in players_output
                    for s in p["stats"]
                    if not s.get("is_v0") and s.get("model_version"))
    no_proj_count = sum(1 for p in players_output
                        for s in p["stats"]
                        if "no projection" in s.get("flags", []))
    no_proj_players = sum(1 for p in players_output
                          if p.get("no_projection"))

    # ── Matchup context overlay (display only, no model changes) ──
    matchup_context = _load_matchup_context(season, week, games)

    return {
        "season": season,
        "week": week,
        "model_version": fallback_version or v12_tag,
        "games": game_output,
        "players": players_output,
        "matchup_context": matchup_context,
        "coverage": {
            "n_games": len(games),
            "n_players_with_props": len(players_with_props),
            "n_players_with_preds": len(players_with_preds),
            "n_players_displayed": len(players_output),
            "n_no_projection_players": no_proj_players,
            "n_no_projection_lines": no_proj_count,
            "v0_lines": v0_count,
            "v12_lines": v12_count,
            "props_no_preds": sorted(props_no_preds)[:20],
            "preds_no_props": sorted(preds_no_props)[:20],
            "per_game_market": sorted(
                _game_market_coverage,
                key=lambda x: x["coverage_pct"]),
        },
        "redistribution": redist_report,
        "gtd_scenarios": {name: {
            "team": sc["team"], "position": sc["position"],
            "status": sc["status"], "freed": sc["freed"],
            "n_teammates": len(sc["boosts"]),
        } for name, sc in gtd_scenarios.items()},
        "injuries": {
            "out": [{"name": n, **info} for n, info in out_players.items()
                    if info["team"] in team_game],
            "questionable": [{"name": n, **info}
                             for n, info in questionable_players.items()
                             if info["team"] in team_game],
        },
        "qa_issues": qa_issues,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
