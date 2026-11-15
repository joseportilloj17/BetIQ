"""
lab.py — NFL Lab: interactive feature exposure + user override + measurement.

Independent of negbin/live-sim. Surfaces model-vs-market divergences,
lets the user override probabilities, and measures everything by
REALIZED ROI at adequate N (not CLV — CLV misleads on soft markets).

Tables: lab_adjustments (bets.db via SQLAlchemy)
Data:   fixtures (bets.db), nfl_features (nfl.db)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from math import erf, exp, sqrt
from typing import Any, Dict, List, Optional

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_NFL_DB = os.path.join(_DB_DIR, "nfl.db")

# ─── NFL Baseline Model Constants ─────────────────────────────────────────────
# Simple power-rating model: EPA differential → margin → probability.
# Not meant to beat the market on average — meant to DIVERGE per-game
# so the user can inspect divergences and override where they have an edge.

HFA_EPA = 0.03           # Home field advantage in EPA/play units (~2.5 pts)
EPA_TO_MARGIN = 82.0     # Net EPA diff → expected margin (66 plays × ~1.24 pts)
NFL_MARGIN_STD = 13.5    # Std dev of NFL game margins
NFL_TOTAL_STD = 10.0     # Std dev of NFL game totals

# Measurement thresholds
MIN_N_SHOW = 10          # Show results but flag as insufficient
MIN_N_MEANINGFUL = 30    # Minimum to show ROI conclusion
MIN_N_CONFIDENT = 100    # "Reliable" verdict threshold

# ─── Math Utilities ───────────────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _american_to_decimal(odds: int) -> Optional[float]:
    if odds >= 100:
        return 1.0 + odds / 100.0
    elif odds <= -100:
        return 1.0 + 100.0 / abs(odds)
    return None


def _decimal_to_implied(dec: float) -> Optional[float]:
    return 1.0 / dec if dec and dec > 0 else None


def _devig(imp1: float, imp2: float) -> tuple[float, float]:
    """Remove vig from a two-way market."""
    total = imp1 + imp2
    if total <= 0:
        return (0.5, 0.5)
    return (imp1 / total, imp2 / total)


# ─── OddsAPI ↔ nfl_features Team Name Mapping ────────────────────────────────

_FULL_TO_ABBREV = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
_ABBREV_TO_FULL = {v: k for k, v in _FULL_TO_ABBREV.items()}


# ─── Feature Definitions (what to expose in the UI) ──────────────────────────

FEATURE_GROUPS = {
    "Efficiency": [
        ("s_epa_per_play",     "EPA/Play (Off)",    "higher = better offense"),
        ("s_def_epa_per_play", "EPA/Play (Def)",    "lower (more neg) = better defense"),
        ("s_success_rate",     "Success Rate (Off)", "% plays with positive EPA"),
        ("s_def_success_rate", "Def Success Rate",   "lower = better defense"),
    ],
    "Phase Splits": [
        ("s_pass_epa_off", "Pass EPA (Off)", "passing efficiency"),
        ("s_rush_epa_off", "Rush EPA (Off)", "rushing efficiency"),
        ("s_pass_epa_def", "Pass EPA (Def)", "pass defense (neg = good)"),
        ("s_rush_epa_def", "Rush EPA (Def)", "rush defense (neg = good)"),
    ],
    "QB Quality": [
        ("s_qb_epa",  "QB EPA",  "quarterback efficiency"),
        ("s_qb_cpoe", "QB CPOE", "completion % over expected"),
    ],
    "Scoring": [
        ("s_points_per_game", "Points/Game",  "scoring average"),
        ("s_points_allowed",  "Pts Allowed",  "defensive scoring avg"),
        ("s_point_diff",      "Point Diff",   "net scoring margin"),
    ],
    "Recent Form (4G)": [
        ("r4_epa_per_play", "R4 EPA/Play",  "last 4 games efficiency"),
        ("r4_point_diff",   "R4 Pt Diff",   "last 4 games margin"),
        ("r4_win_pct",      "R4 Win%",      "last 4 games win rate"),
    ],
    "Matchup vs Opponent": [
        ("mu_overall",      "Overall Matchup", "this team O vs opponent D"),
        ("mu_pass_o_vs_d",  "Pass Matchup",    "pass O vs pass D"),
        ("mu_rush_o_vs_d",  "Rush Matchup",    "rush O vs rush D"),
    ],
    "Schedule": [
        ("rest_days",      "Rest Days",    "days since last game"),
        ("is_short_week",  "Short Week",   "1 if < 6 rest days"),
        ("is_divisional",  "Divisional",   "1 if division game"),
        ("is_primetime",   "Primetime",    "1 if primetime slot"),
    ],
    "Environment": [
        ("is_dome", "Dome",     "1 if indoor"),
        ("temp",    "Temp (°F)", "game-time temperature"),
        ("wind",    "Wind (mph)", "wind speed"),
    ],
    "Pressure (PBP)": [
        ("adv_s_pressure_rate_allowed",   "Pressure% Allowed",   "sacks + QB hits per dropback (lower = better OL)"),
        ("adv_s_pressure_rate_generated", "Pressure% Generated", "sacks + QB hits per dropback (higher = better pass rush)"),
        ("adv_s_sack_rate_allowed",       "Sack% Allowed",       "sack rate per dropback"),
        ("adv_s_sack_rate_generated",     "Sack% Generated",     "sack rate forced"),
    ],
    "Separation / YAC (PBP)": [
        ("adv_s_avg_yac",       "Avg YAC",      "yards after catch per completion"),
        ("adv_s_yac_oe",        "YAC Over Exp",  "actual YAC minus expected — true separation + RAC ability"),
        ("adv_s_avg_air_yards", "Avg Air Yards", "average depth of target (aDOT proxy)"),
        ("adv_s_yac_allowed",   "YAC Allowed",   "yards after catch allowed by defense"),
    ],
    "Scheme Tendencies (PBP)": [
        ("adv_s_shotgun_rate",    "Shotgun%",       "plays from shotgun formation"),
        ("adv_s_no_huddle_rate",  "No-Huddle%",     "plays from no-huddle"),
        ("adv_s_deep_pass_rate",  "Deep Ball%",     "deep pass rate (20+ air yards)"),
        ("adv_s_middle_pass_rate","Slot Target%",   "passes to middle of field (slot proxy)"),
        ("adv_s_run_outside_rate","Outside Run%",   "outside zone / sweep rate"),
    ],
    "Situational (PBP)": [
        ("adv_s_third_down_conv_rate", "3rd Down Conv%",  "3rd down conversion rate"),
        ("adv_s_redzone_td_rate",      "Red Zone TD%",    "TD rate when inside the 20"),
        ("adv_s_explosive_rate",       "Explosive Play%", "plays gaining 20+ yards"),
        ("adv_s_third_down_stop_rate", "3rd Down Stop%",  "3rd down defense stop rate"),
        ("adv_s_explosive_rate_allowed","Explosive Alwd%", "20+ yard plays allowed"),
    ],
    "Turnovers (PBP)": [
        ("adv_s_turnover_rate",        "TO Rate",        "turnovers per play (INTs + fumbles)"),
        ("adv_s_turnover_forced_rate", "TO Forced Rate", "turnovers forced per play"),
        ("adv_s_int_rate",             "INT Rate",       "interceptions per pass attempt"),
        ("adv_s_int_forced_rate",      "INT Forced Rate","interceptions forced per pass attempt"),
    ],
    "Secondary / Coverage (PBP)": [
        ("adv_s_completion_pct_allowed",    "Completion% Alwd",    "overall completion rate allowed"),
        ("adv_s_deep_completion_allowed",   "Deep Comp% Alwd",     "deep pass completion rate allowed (CB quality)"),
        ("adv_s_short_completion_allowed",  "Short Comp% Alwd",    "short pass completion rate allowed"),
        ("adv_s_outside_completion_allowed","Outside Comp% Alwd",  "outside completion rate allowed (CB proxy)"),
        ("adv_s_middle_completion_allowed", "Middle Comp% Alwd",   "middle completion rate allowed (LB coverage)"),
        ("adv_s_avg_air_yards_allowed",     "Air Yds Allowed",     "average depth of target allowed"),
    ],
    "Run Game Detail (PBP)": [
        ("adv_s_run_epa_inside",           "Inside Run EPA",       "EPA/play on inside runs (guard/tackle)"),
        ("adv_s_run_epa_outside",          "Outside Run EPA",      "EPA/play on outside runs (end)"),
        ("adv_s_run_epa_inside_allowed",   "Inside Run EPA Alwd",  "inside run EPA allowed (DL interior)"),
        ("adv_s_run_epa_outside_allowed",  "Outside Run EPA Alwd", "outside run EPA allowed (edge defense)"),
        ("adv_s_run_epa_allowed",          "Run EPA Alwd",         "overall run EPA allowed"),
        ("adv_s_tfl_rate",                 "TFL Rate",             "tackles for loss rate (DL dominance)"),
        ("adv_s_inside_run_rate",          "Inside Run%",          "inside run tendency"),
    ],
    "QB Accuracy (PBP)": [
        ("adv_s_cpoe_avg",                 "CPOE",                 "completion % over expected (accuracy)"),
        ("adv_s_deep_pass_epa",            "Deep Ball EPA",        "EPA on deep passes"),
        ("adv_s_short_pass_epa",           "Short Pass EPA",       "EPA on short passes"),
        ("adv_s_deep_completion_rate",     "Deep Comp%",           "completion rate on deep throws"),
    ],
    "Special Teams (PBP)": [
        ("adv_s_punt_inside_20_rate",      "Punt I20%",            "punt inside 20 rate"),
        ("adv_s_fg_make_rate",             "FG Make%",             "field goal make rate"),
        ("adv_s_kickoff_touchback_rate",   "Kickoff TB%",          "kickoff touchback rate"),
        ("adv_s_punt_return_avg",          "Punt Ret Avg",         "average punt return yards"),
        ("adv_s_kick_return_avg",          "Kick Ret Avg",         "average kick return yards"),
    ],
}


# ─── NFL Feature Fetching ─────────────────────────────────────────────────────

def _get_nfl_features(team_abbrev: str, before_date: str) -> Optional[dict]:
    """
    Fetch most recent point-in-time NFL features for a team.
    Merges base features (nfl_features) with advanced PBP stats
    (nfl_advanced_rolling) into a single dict.
    """
    if not os.path.exists(_NFL_DB):
        return None
    try:
        conn = sqlite3.connect(_NFL_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM nfl_features WHERE team = ? AND game_date <= ? "
            "ORDER BY game_date DESC LIMIT 1",
            (team_abbrev, before_date),
        ).fetchone()
        result = dict(row) if row else None
        if not result:
            conn.close()
            return None

        # Merge advanced rolling stats (pressure, YAC, scheme, etc.)
        adv = conn.execute(
            "SELECT * FROM nfl_advanced_rolling WHERE team = ? AND game_date <= ? "
            "ORDER BY game_date DESC LIMIT 1",
            (team_abbrev, before_date),
        ).fetchone()
        if adv:
            adv_dict = dict(adv)
            # Prefix with 'adv_' to avoid key collisions
            for k, v in adv_dict.items():
                if k.startswith("s_") or k.startswith("r4_"):
                    result[f"adv_{k}"] = v

        conn.close()
        return result
    except Exception as e:
        log.warning("NFL feature fetch failed for %s: %s", team_abbrev, e)
        return None


def _get_game_week(home_abbr: str, away_abbr: str, game_date: str) -> Optional[int]:
    """Look up the NFL week for a game from nfl_games."""
    if not os.path.exists(_NFL_DB):
        return None
    try:
        conn = sqlite3.connect(_NFL_DB)
        row = conn.execute(
            "SELECT week FROM nfl_games "
            "WHERE home_team = ? AND away_team = ? AND game_date = ? LIMIT 1",
            (home_abbr, away_abbr, game_date),
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        log.warning("Week lookup failed: %s", e)
        return None


def _get_raw_stats(team_abbrev: str, season: int = 2025) -> Optional[dict]:
    """
    Fetch aggregated raw counting stats for a team's season.
    Returns totals and per-game averages from nfl_team_game_stats.
    """
    if not os.path.exists(_NFL_DB):
        return None
    try:
        conn = sqlite3.connect(_NFL_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM nfl_team_game_stats WHERE team = ? AND season = ? "
            "ORDER BY game_date",
            (team_abbrev, season),
        ).fetchall()
        conn.close()
        if not rows:
            return None

        n = len(rows)
        # Aggregate counting stats
        total_score = sum(r["score"] or 0 for r in rows)
        total_opp = sum(r["opp_score"] or 0 for r in rows)
        total_off_plays = sum(r["off_plays"] or 0 for r in rows)
        total_pass_plays = sum(r["pass_plays"] or 0 for r in rows)
        total_rush_plays = sum(r["rush_plays"] or 0 for r in rows)
        total_def_plays = sum(r["def_plays"] or 0 for r in rows)
        wins = sum(1 for r in rows if (r["score"] or 0) > (r["opp_score"] or 0))

        # EPA totals (sum EPA/play × plays for total EPA)
        total_off_epa = sum((r["off_epa_per_play"] or 0) * (r["off_plays"] or 0) for r in rows)
        total_def_epa = sum((r["def_epa_per_play"] or 0) * (r["def_plays"] or 0) for r in rows)
        total_pass_epa = sum((r["pass_epa_per_play"] or 0) * (r["pass_plays"] or 0) for r in rows)
        total_rush_epa = sum((r["rush_epa_per_play"] or 0) * (r["rush_plays"] or 0) for r in rows)
        total_def_pass_epa = sum((r["def_pass_epa"] or 0) * (r["def_pass_plays"] or 0) for r in rows)
        total_def_rush_epa = sum((r["def_rush_epa"] or 0) * (r["def_rush_plays"] or 0) for r in rows)

        # Last QB
        last_qb = rows[-1]["qb_name"] if rows[-1]["qb_name"] else "Unknown"

        return {
            "games": n,
            "wins": wins,
            "losses": n - wins,
            "record": f"{wins}-{n - wins}",
            "qb": last_qb,
            # Totals
            "total_points_for": total_score,
            "total_points_against": total_opp,
            "total_point_diff": total_score - total_opp,
            "total_off_plays": total_off_plays,
            "total_pass_plays": total_pass_plays,
            "total_rush_plays": total_rush_plays,
            "total_def_plays": total_def_plays,
            "total_off_epa": round(total_off_epa, 1),
            "total_def_epa": round(total_def_epa, 1),
            "total_pass_epa": round(total_pass_epa, 1),
            "total_rush_epa": round(total_rush_epa, 1),
            "total_def_pass_epa": round(total_def_pass_epa, 1),
            "total_def_rush_epa": round(total_def_rush_epa, 1),
            # Per-game averages
            "ppg": round(total_score / n, 1),
            "opp_ppg": round(total_opp / n, 1),
            "avg_off_plays": round(total_off_plays / n, 1),
            "avg_pass_plays": round(total_pass_plays / n, 1),
            "avg_rush_plays": round(total_rush_plays / n, 1),
            "avg_def_plays": round(total_def_plays / n, 1),
        }
    except Exception as e:
        log.warning("Raw stats fetch failed for %s: %s", team_abbrev, e)
        return None


def _spread_to_win_prob(spread: float) -> float:
    """Convert a point spread to implied win probability.
    spread is from the team's perspective (negative = favored).
    Uses NFL margin std dev of 13.5."""
    return _normal_cdf(-spread / NFL_MARGIN_STD)


def get_futures(db_session=None) -> dict:
    """
    Build 2026 season win projections from fixtures in bets.db.

    Uses MARKET SPREADS as the primary win probability source
    (this is where "8.5 wins" style numbers come from — the market's view).
    EPA model projection shown as secondary comparison.

    Each game's market-implied win prob = spread → normal CDF.
    Sum across 17 games = projected season wins.
    """
    from database import Fixture

    if db_session is None:
        from database import get_db
        db_session = next(get_db())

    # Fetch 2026 NFL fixtures
    fixtures = (
        db_session.query(Fixture)
        .filter(
            Fixture.sport_key == "americanfootball_nfl",
            Fixture.game_date_ct >= "2026-09-01",
            Fixture.game_date_ct <= "2027-02-15",
        )
        .order_by(Fixture.commence_time)
        .all()
    )

    if not fixtures:
        return {"error": "No 2026 fixtures found", "teams": []}

    # Build per-team game list with market-implied win probs
    team_games: Dict[str, list] = {}
    week_counter: Dict[str, int] = {}  # track week number per team

    for fix in fixtures:
        home_full = fix.home_team
        away_full = fix.away_team
        home_abbr = _FULL_TO_ABBREV.get(home_full, home_full)
        away_abbr = _FULL_TO_ABBREV.get(away_full, away_full)
        gdate = fix.game_date_ct or ""

        # Extract market lines
        bookmakers = fix.bookmakers if isinstance(fix.bookmakers, list) else []
        if isinstance(fix.bookmakers, str):
            try:
                bookmakers = json.loads(fix.bookmakers)
            except Exception:
                bookmakers = []
        market = _extract_market_lines(bookmakers, home_full, away_full)

        # Market-implied win prob from spread
        home_spread = market.get("home_spread")
        if home_spread is not None:
            home_mkt_wp = _spread_to_win_prob(home_spread)
            away_mkt_wp = 1.0 - home_mkt_wp
        else:
            # Fallback: use ML implied probs if available
            home_mkt_wp = market.get("home_ml_prob", 0.5)
            away_mkt_wp = market.get("away_ml_prob", 0.5)

        # EPA model win prob (using latest features)
        home_feats = _get_nfl_features(home_abbr, gdate) if home_abbr else None
        away_feats = _get_nfl_features(away_abbr, gdate) if away_abbr else None
        if home_feats and away_feats:
            bl = compute_baseline(
                home_feats, away_feats, regress_to_mean=True,
                home_abbr=home_abbr, away_abbr=away_abbr,
            )
            home_model_wp = bl.get("home_win_prob", 0.5)
        else:
            home_model_wp = 0.5

        # Assign week numbers per team
        for t in [home_abbr, away_abbr]:
            week_counter[t] = week_counter.get(t, 0) + 1

        game_info = {
            "fixture_id": fix.id,
            "game_date": gdate,
            "commence_time": fix.commence_time.isoformat() if fix.commence_time else None,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_full": home_full,
            "away_full": away_full,
            "spread": home_spread,
            "total_line": market.get("total_line"),
            "home_mkt_wp": round(home_mkt_wp, 3),
            "away_mkt_wp": round(away_mkt_wp, 3),
            "home_model_wp": round(home_model_wp, 3),
            "away_model_wp": round(1.0 - home_model_wp, 3),
        }

        # Add to each team's game list
        for t_abbr, is_home in [(home_abbr, True), (away_abbr, False)]:
            team_games.setdefault(t_abbr, [])
            opp = away_abbr if is_home else home_abbr
            team_spread = home_spread if is_home else (-home_spread if home_spread is not None else None)
            mkt_wp = home_mkt_wp if is_home else away_mkt_wp
            model_wp = home_model_wp if is_home else (1.0 - home_model_wp)

            team_games[t_abbr].append({
                "week": week_counter[t_abbr],
                "game_date": gdate,
                "opponent": opp,
                "is_home": is_home,
                "location": "vs" if is_home else "@",
                "spread": round(team_spread, 1) if team_spread is not None else None,
                "total_line": market.get("total_line"),
                "market_wp": round(mkt_wp, 3),
                "model_wp": round(model_wp, 3),
            })

    # Load roster adjustments for display
    radj = _get_roster_adjustments()

    # Build team summaries
    team_data = []
    for team in sorted(team_games.keys()):
        games = team_games[team]
        mkt_wins = sum(g["market_wp"] for g in games)
        model_wins = sum(g["model_wp"] for g in games)
        full_name = _ABBREV_TO_FULL.get(team, team)

        team_entry = {
            "team": team,
            "full_name": full_name,
            "games_scheduled": len(games),
            "market_wins": round(mkt_wins, 1),
            "model_wins": round(model_wins, 1),
            "edge": round(model_wins - mkt_wins, 1),
            "games": games,
        }

        # Add roster adjustment summary
        if team in radj:
            r = radj[team]
            team_entry["roster_adj"] = {
                "net_epa": r.get("net_epa", 0),
                "net_off_epa": r.get("net_off_epa", 0),
                "net_def_epa": r.get("net_def_epa", 0),
                "epa_per_play_adj": r.get("epa_per_play_adj", 0),
                "departed_epa": r.get("departed_epa", 0),
                "departed_count": r.get("departed_count", 0),
                "incoming_epa": r.get("incoming_epa", 0),
                "incoming_count": r.get("incoming_count", 0),
                "rookie_epa": r.get("rookie_epa", 0),
                "rookie_count": r.get("rookie_count", 0),
            }

        team_data.append(team_entry)

    # Sort by market wins descending
    team_data.sort(key=lambda t: -t["market_wins"])

    return {
        "season": 2026,
        "team_count": len(team_data),
        "teams": team_data,
    }


# ─── Roster Adjustment Loading ───────────────────────────────────────────────

_ROSTER_ADJ_CACHE: Optional[Dict[str, dict]] = None


def _get_roster_adjustments() -> Dict[str, dict]:
    """Load roster EPA adjustments from nfl.db (cached after first call)."""
    global _ROSTER_ADJ_CACHE
    if _ROSTER_ADJ_CACHE is not None:
        return _ROSTER_ADJ_CACHE

    if not os.path.exists(_NFL_DB):
        _ROSTER_ADJ_CACHE = {}
        return _ROSTER_ADJ_CACHE

    try:
        conn = sqlite3.connect(_NFL_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM nfl_roster_adjustments"
        ).fetchall()
        conn.close()
        _ROSTER_ADJ_CACHE = {row["team"]: dict(row) for row in rows}
        log.info("Loaded roster adjustments for %d teams", len(_ROSTER_ADJ_CACHE))
    except Exception as e:
        log.warning("Could not load roster adjustments: %s", e)
        _ROSTER_ADJ_CACHE = {}

    return _ROSTER_ADJ_CACHE


# ─── QB Continuity ───────────────────────────────────────────────────────────

_QB_CONTINUITY_CACHE: Optional[Dict[str, dict]] = None


def _get_qb_continuity() -> Dict[str, dict]:
    """
    Detect which teams retained their starting QB from last season.

    Returns {team_abbr: {"same_qb": bool, "qb_name": str, "qb_epa": float}}

    Uses nfl_player_epa (2025 QB with most snaps per team) cross-referenced
    against roster_2026 to detect movement.
    """
    global _QB_CONTINUITY_CACHE
    if _QB_CONTINUITY_CACHE is not None:
        return _QB_CONTINUITY_CACHE

    _QB_CONTINUITY_CACHE = {}

    if not os.path.exists(_NFL_DB):
        return _QB_CONTINUITY_CACHE

    try:
        conn = sqlite3.connect(_NFL_DB)
        conn.row_factory = sqlite3.Row

        # Get each team's starting QB from 2025 (QB with most snaps)
        qbs = conn.execute("""
            SELECT team, player_name, gsis_id, total_epa, snap_count
            FROM nfl_player_epa
            WHERE season = 2025 AND position_group = 'QB' AND snap_count > 200
            ORDER BY team, snap_count DESC
        """).fetchall()
        conn.close()

        # Group by team — take the QB with the most snaps
        team_qb: Dict[str, dict] = {}
        for qb in qbs:
            t = qb["team"]
            if t not in team_qb:
                team_qb[t] = dict(qb)

        # Load 2026 roster to check if QB is still on team
        import pandas as pd
        roster_path = "/tmp/nfl_roster_cache/roster_2026.parquet"
        if os.path.exists(roster_path):
            r26 = pd.read_parquet(roster_path)
            r26_teams = dict(zip(r26["gsis_id"].dropna(), r26["team"]))

            for team, qb_info in team_qb.items():
                gsis = qb_info["gsis_id"]
                new_team = r26_teams.get(gsis)
                same = new_team == team if new_team else False
                _QB_CONTINUITY_CACHE[team] = {
                    "same_qb": same,
                    "qb_name": qb_info["player_name"],
                    "qb_epa": qb_info["total_epa"],
                    "qb_snaps": qb_info["snap_count"],
                    "new_team": new_team,
                }

            log.info(
                "QB continuity: %d/%d teams retained starting QB",
                sum(1 for v in _QB_CONTINUITY_CACHE.values() if v["same_qb"]),
                len(_QB_CONTINUITY_CACHE),
            )
        else:
            log.warning("No 2026 roster file found for QB continuity check")

    except Exception as e:
        log.warning("QB continuity check failed: %s", e)

    return _QB_CONTINUITY_CACHE


# ─── Win/Loss Record ────────────────────────────────────────────────────────

_TEAM_RECORDS_CACHE: Optional[Dict[str, float]] = None


def _get_team_records(season: int = 2025) -> Dict[str, float]:
    """
    Get each team's W/L record from nfl_games, returned as win percentage.
    Used to blend results with process metrics for more accurate projections.
    """
    global _TEAM_RECORDS_CACHE
    if _TEAM_RECORDS_CACHE is not None:
        return _TEAM_RECORDS_CACHE

    _TEAM_RECORDS_CACHE = {}
    if not os.path.exists(_NFL_DB):
        return _TEAM_RECORDS_CACHE

    try:
        conn = sqlite3.connect(_NFL_DB)
        conn.row_factory = sqlite3.Row
        games = conn.execute(
            "SELECT home_team, away_team, home_score, away_score "
            "FROM nfl_games WHERE season = ? AND home_score IS NOT NULL",
            (season,),
        ).fetchall()
        conn.close()

        wins: Dict[str, int] = {}
        total: Dict[str, int] = {}
        for g in games:
            ht, at = g["home_team"], g["away_team"]
            hs, as_ = g["home_score"] or 0, g["away_score"] or 0
            for t in [ht, at]:
                total[t] = total.get(t, 0) + 1
            if hs > as_:
                wins[ht] = wins.get(ht, 0) + 1
            elif as_ > hs:
                wins[at] = wins.get(at, 0) + 1

        for t in total:
            _TEAM_RECORDS_CACHE[t] = wins.get(t, 0) / total[t] if total[t] > 0 else 0.5

        log.info("Loaded %d team records for season %d", len(_TEAM_RECORDS_CACHE), season)
    except Exception as e:
        log.warning("Could not load team records: %s", e)

    return _TEAM_RECORDS_CACHE


# ─── Player EPA + Yardage Estimates ──────────────────────────────────────────

# League-average yards-per-play for EPA↔yards conversion.
# Source: 2020-2025 NFL averages.
_AVG_PASS_YPA = 7.0    # yards per pass attempt (NFL avg)
_AVG_RUSH_YPC = 4.3    # yards per rush carry (NFL avg)
_EPA_TO_YPA_SCALE = 10.0  # 0.10 EPA/play ≈ 1.0 extra yard/play


def _get_team_players(team_abbrev: str, season: int = 2025) -> Optional[List[dict]]:
    """
    Fetch top offensive players for a team from nfl_player_epa.
    Returns list sorted by |total_epa|, capped at top 12 skill players.
    """
    if not os.path.exists(_NFL_DB):
        return None
    try:
        conn = sqlite3.connect(_NFL_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT player_name, position_group, side, total_epa, "
            "epa_per_play, snap_count, games_played "
            "FROM nfl_player_epa "
            "WHERE team = ? AND season = ? AND side = 'OFF' "
            "AND snap_count >= 20 "
            "ORDER BY total_epa DESC",
            (team_abbrev, season),
        ).fetchall()
        conn.close()
        if not rows:
            return None
        return [dict(r) for r in rows[:12]]
    except Exception as e:
        log.warning("Player EPA fetch failed for %s: %s", team_abbrev, e)
        return None


def _estimate_team_yards(raw_stats: Optional[dict]) -> Optional[dict]:
    """
    Estimate team pass/rush yards per game from raw stats we have.
    Uses play counts × league-average yards/play as baseline,
    then adjusts for EPA differential vs average.
    """
    if not raw_stats:
        return None
    n = raw_stats.get("games", 0)
    if n == 0:
        return None

    pass_plays_pg = raw_stats.get("avg_pass_plays", 33.0)
    rush_plays_pg = raw_stats.get("avg_rush_plays", 27.0)

    # EPA per play for this team (from raw per-game EPA totals ÷ plays)
    total_pass_epa = raw_stats.get("total_pass_epa", 0)
    total_rush_epa = raw_stats.get("total_rush_epa", 0)
    total_pass_plays = raw_stats.get("total_pass_plays", 1)
    total_rush_plays = raw_stats.get("total_rush_plays", 1)

    pass_epa_pp = total_pass_epa / total_pass_plays if total_pass_plays > 0 else 0
    rush_epa_pp = total_rush_epa / total_rush_plays if total_rush_plays > 0 else 0

    # Estimated yards/play = league avg + EPA-based adjustment
    est_pass_ypa = _AVG_PASS_YPA + pass_epa_pp * _EPA_TO_YPA_SCALE
    est_rush_ypc = _AVG_RUSH_YPC + rush_epa_pp * _EPA_TO_YPA_SCALE

    est_pass_ypg = round(pass_plays_pg * est_pass_ypa, 1)
    est_rush_ypg = round(rush_plays_pg * est_rush_ypc, 1)

    return {
        "pass_ypg": est_pass_ypg,
        "rush_ypg": est_rush_ypg,
        "total_ypg": round(est_pass_ypg + est_rush_ypg, 1),
        "pass_ypa": round(est_pass_ypa, 2),
        "rush_ypc": round(est_rush_ypc, 2),
        "pass_plays_pg": round(pass_plays_pg, 1),
        "rush_plays_pg": round(rush_plays_pg, 1),
    }


# ─── NFL Baseline Model ──────────────────────────────────────────────────────

# Blend weight: how much actual W/L record influences the base rating
# vs pure EPA process metrics. Record captures luck, clutch, special teams,
# coaching — things EPA misses. But record is noisier than EPA.
RECORD_BLEND = 0.50  # 50% record, 50% EPA process (record is most predictive for futures)

# Scale factor to convert win percentage to EPA-equivalent rating.
# EPA rating range: roughly ±0.15. Win pct range: 0.18 to 0.85.
# Scale of 0.50: 100% → +0.25, 0% → -0.25. Wider than EPA range to properly
# anchor tail teams (3-14 ARI, 17-4 NE) whose records carry stronger signal.
RECORD_TO_EPA_SCALE = 0.60  # (win_pct - 0.5) × 0.60 → EPA-scale rating


def _team_rating(feats: dict) -> float:
    """
    Multi-dimensional team power rating (EPA-equivalent units).

    Blends six process signals across offense, defense, and special teams:
      30% — season EPA net (offense - defense, core signal)
      15% — recent 4-game EPA (late-season trajectory)
      15% — QB EPA (most predictive single offensive feature)
      15% — defensive quality composite (pressure + coverage + run defense)
      15% — success rate net (process quality, less noisy than EPA)
      10% — special teams composite (punt, kick, FG, return)

    This separates offensive and defensive quality so roster changes on
    either side are properly captured, and adds special teams which
    accounts for ~3-5 wins/season variance league-wide.
    """
    # --- Offense ---
    off_epa = feats.get("s_epa_per_play") or 0
    qb_epa = feats.get("s_qb_epa") or 0
    off_sr = feats.get("s_success_rate") or 0.43

    # --- Defense ---
    def_epa = feats.get("s_def_epa_per_play") or 0  # positive = offense scored on you
    def_sr = feats.get("s_def_success_rate") or 0.43

    # Defensive quality composite from PBP advanced stats:
    # Pressure generated (higher = better defense), completion% allowed (lower = better),
    # run defense EPA (lower = better), TFL rate (higher = better)
    pressure_gen = feats.get("adv_s_pressure_rate_generated") or 0.22
    comp_allowed = feats.get("adv_s_completion_pct_allowed") or 0.65
    run_epa_allowed = feats.get("adv_s_run_epa_allowed") or -0.04
    tfl_rate = feats.get("adv_s_tfl_rate") or 0.038
    deep_comp_allowed = feats.get("adv_s_deep_completion_allowed") or 0.40

    # Scale each defensive dimension to ~EPA range (±0.15)
    # Above-average pressure generates ~0.02 EPA/play advantage
    def_composite = (
        (pressure_gen - 0.22) * 0.3       # pressure edge
        + (0.65 - comp_allowed) * 0.3      # coverage quality
        + (-0.04 - run_epa_allowed) * 0.2  # run defense
        + (tfl_rate - 0.038) * 0.5         # DL dominance
        + (0.40 - deep_comp_allowed) * 0.15  # deep ball defense
    )

    # --- Special Teams ---
    # Punt quality, FG reliability, kick/punt return, field position
    punt_i20 = feats.get("adv_s_punt_inside_20_rate") or 0.40
    fg_rate = feats.get("adv_s_fg_make_rate") or 0.85
    kick_ret = feats.get("adv_s_kick_return_avg") or 22.0
    punt_ret = feats.get("adv_s_punt_return_avg") or 4.0

    # Scale ST to EPA units: good ST worth ~0.01-0.03 EPA/play
    st_composite = (
        (punt_i20 - 0.40) * 0.05   # punting edge
        + (fg_rate - 0.85) * 0.1   # FG reliability
        + (kick_ret - 22.0) * 0.001  # kick return advantage
        + (punt_ret - 4.0) * 0.001   # punt return advantage
    )

    # --- Net ratings ---
    epa_net = off_epa - def_epa
    r4_epa = feats.get("r4_epa_per_play") or 0
    sr_net = off_sr - def_sr

    # Weighted composite
    return (
        0.30 * epa_net
        + 0.15 * r4_epa
        + 0.15 * qb_epa * 0.6       # QB scaled to EPA range
        + 0.15 * def_composite
        + 0.15 * sr_net * 2.0        # SR scaled to EPA range
        + 0.10 * st_composite
    )


# Year-over-year NFL quality mean-reverts aggressively (optimal via grid search).
# QB-adjusted: teams retaining their starting QB are more stable year-over-year
# because QB explains ~60% of offensive variance. New-QB teams regress harder.
REGRESSION_SAME_QB = 0.70   # Same starting QB: keep 70% of raw rating
REGRESSION_NEW_QB = 0.50    # New/lost starting QB: keep 50%
REGRESSION_DEFAULT = 0.60   # Fallback when QB status unknown


def _matchup_adjustment(home_feats: dict, away_feats: dict) -> float:
    """
    Compute matchup-specific margin adjustment from advanced PBP features.

    Seven matchup dimensions measuring style clashes:
      1. Pass rush vs OL (pressure generated vs allowed)
      2. Deep ball offense vs secondary (aDOT, deep completion%)
      3. Run style vs run defense (inside/outside gap matchup)
      4. QB accuracy vs coverage quality (CPOE vs completion% allowed)
      5. Receiver quality vs secondary (YAC over expected)
      6. Explosive play creation vs prevention
      7. Turnover tendency matchup

    Returns adjustment in POINTS (added to estimated margin, home perspective).
    Capped at ±3 points.
    """
    adj = 0.0

    # --- 1. Pass rush vs OL (DL vs OL position group matchup) ---
    h_press_gen = home_feats.get("adv_s_pressure_rate_generated") or 0.22
    a_press_alw = away_feats.get("adv_s_pressure_rate_allowed") or 0.22
    a_press_gen = away_feats.get("adv_s_pressure_rate_generated") or 0.22
    h_press_alw = home_feats.get("adv_s_pressure_rate_allowed") or 0.22
    h_tfl = home_feats.get("adv_s_tfl_rate") or 0.038
    a_tfl = away_feats.get("adv_s_tfl_rate") or 0.038
    # Interaction: home pass rush vs away OL weakness, and vice versa
    home_press_edge = (h_press_gen - 0.22) + (a_press_alw - 0.22) + (h_tfl - 0.038) * 2.0
    away_press_edge = (a_press_gen - 0.22) + (h_press_alw - 0.22) + (a_tfl - 0.038) * 2.0
    adj += (home_press_edge - away_press_edge) * 3.0

    # --- 2. Deep ball offense vs secondary (CB matchup proxy) ---
    # Team that throws deep a lot vs defense that allows deep completions
    h_deep_rate = home_feats.get("adv_s_deep_pass_rate") or 0.15
    h_deep_epa = home_feats.get("adv_s_deep_pass_epa") or 0
    a_deep_comp_alw = away_feats.get("adv_s_deep_completion_allowed") or 0.40
    a_deep_rate = away_feats.get("adv_s_deep_pass_rate") or 0.15
    a_deep_epa = away_feats.get("adv_s_deep_pass_epa") or 0
    h_deep_comp_alw = home_feats.get("adv_s_deep_completion_allowed") or 0.40
    # Teams that are good deep AND face a weak deep defense get a bump
    home_deep_edge = h_deep_epa * h_deep_rate + (a_deep_comp_alw - 0.40) * 0.3
    away_deep_edge = a_deep_epa * a_deep_rate + (h_deep_comp_alw - 0.40) * 0.3
    adj += (home_deep_edge - away_deep_edge) * 3.0

    # --- 3. Run style vs run defense (inside/outside gap matchup) ---
    # Team that runs inside a lot vs defense's inside run defense, etc.
    h_inside_rate = home_feats.get("adv_s_inside_run_rate") or 0.60
    h_run_epa_in = home_feats.get("adv_s_run_epa_inside") or -0.06
    h_run_epa_out = home_feats.get("adv_s_run_epa_outside") or 0.08
    a_run_def_in = away_feats.get("adv_s_run_epa_inside_allowed") or -0.06
    a_run_def_out = away_feats.get("adv_s_run_epa_outside_allowed") or 0.08

    a_inside_rate = away_feats.get("adv_s_inside_run_rate") or 0.60
    a_run_epa_in = away_feats.get("adv_s_run_epa_inside") or -0.06
    a_run_epa_out = away_feats.get("adv_s_run_epa_outside") or 0.08
    h_run_def_in = home_feats.get("adv_s_run_epa_inside_allowed") or -0.06
    h_run_def_out = home_feats.get("adv_s_run_epa_outside_allowed") or 0.08

    # Weighted by team's run tendency: inside-heavy team vs weak inside D
    home_run_edge = (
        h_inside_rate * (h_run_epa_in - a_run_def_in)
        + (1 - h_inside_rate) * (h_run_epa_out - a_run_def_out)
    )
    away_run_edge = (
        a_inside_rate * (a_run_epa_in - h_run_def_in)
        + (1 - a_inside_rate) * (a_run_epa_out - h_run_def_out)
    )
    adj += (home_run_edge - away_run_edge) * 2.0

    # --- 4. QB accuracy vs coverage (CPOE vs completion% allowed) ---
    # High-accuracy QB vs tight coverage = still performs
    # Low-accuracy QB vs tight coverage = collapses
    h_cpoe = home_feats.get("adv_s_cpoe_avg") or 0
    a_cpoe = away_feats.get("adv_s_cpoe_avg") or 0
    a_comp_alw = away_feats.get("adv_s_completion_pct_allowed") or 0.65
    h_comp_alw = home_feats.get("adv_s_completion_pct_allowed") or 0.65
    # Interaction: QB accuracy advantage amplified by opponent coverage tightness
    # vs weak coverage (comp_alw > 0.65), CPOE matters less (everyone completes)
    # vs tight coverage (comp_alw < 0.60), CPOE matters MORE (accuracy separates)
    home_acc_edge = h_cpoe * (0.65 - a_comp_alw + 0.5)
    away_acc_edge = a_cpoe * (0.65 - h_comp_alw + 0.5)
    adj += (home_acc_edge - away_acc_edge) * 0.5

    # --- 5. Receiver quality vs secondary (YAC + pass location matchup) ---
    h_yac_oe = home_feats.get("adv_s_yac_oe") or 0
    a_yac_oe = away_feats.get("adv_s_yac_oe") or 0
    a_yac_alw = away_feats.get("adv_s_yac_allowed") or 5.3
    h_yac_alw = home_feats.get("adv_s_yac_allowed") or 5.3
    # Good YAC team vs defense that gives up YAC = amplified advantage
    home_yac_edge = h_yac_oe + (a_yac_alw - 5.3) * 0.3
    away_yac_edge = a_yac_oe + (h_yac_alw - 5.3) * 0.3
    adj += (home_yac_edge - away_yac_edge) * 0.15

    # --- 6. Explosive plays ---
    h_expl = home_feats.get("adv_s_explosive_rate") or 0.059
    a_expl_alw = away_feats.get("adv_s_explosive_rate_allowed") or 0.059
    a_expl = away_feats.get("adv_s_explosive_rate") or 0.059
    h_expl_alw = home_feats.get("adv_s_explosive_rate_allowed") or 0.059
    home_expl = (h_expl - 0.059) + (a_expl_alw - 0.059)
    away_expl = (a_expl - 0.059) + (h_expl_alw - 0.059)
    adj += (home_expl - away_expl) * 4.0

    # --- 7. Turnover tendency ---
    h_to_rate = home_feats.get("adv_s_turnover_rate") or 0.028
    a_to_rate = away_feats.get("adv_s_turnover_rate") or 0.028
    h_to_forced = home_feats.get("adv_s_turnover_forced_rate") or 0.028
    a_to_forced = away_feats.get("adv_s_turnover_forced_rate") or 0.028
    # Net: home forces more TOs than it gives up, relative to opponent
    home_to_net = (h_to_forced - h_to_rate) - (a_to_forced - a_to_rate)
    adj += home_to_net * 20.0  # ±0.01 → ±0.2 pts

    # Cap adjustment
    return max(-3.0, min(3.0, adj))


def compute_baseline(
    home_feats: Optional[dict],
    away_feats: Optional[dict],
    market_spread: Optional[float] = None,
    market_total: Optional[float] = None,
    regress_to_mean: bool = False,
    home_abbr: Optional[str] = None,
    away_abbr: Optional[str] = None,
) -> dict:
    """
    Compute baseline probabilities from NFL features.

    Three layers:
      1. Base power rating (EPA + recent form + QB + success rate)
      2. Roster adjustments (offseason trades/FA/draft EPA impact)
      3. Matchup adjustments (pressure, YAC/separation, explosive plays)

    When regress_to_mean=True (for year-ahead projections), shrinks base
    ratings by REGRESSION_FACTOR and applies roster adjustments.
    Matchup adjustments are NOT applied for futures (wash out over 17 games).

    Returns dict with home_win_prob, estimated_margin, estimated_total,
    spread_cover_prob, total_over_prob (when market lines provided).
    """
    if not home_feats or not away_feats:
        return {"available": False, "reason": "features_missing"}

    # Layer 1: Multi-feature team power ratings (EPA process)
    home_power = _team_rating(home_feats)
    away_power = _team_rating(away_feats)

    # Apply regression to mean for year-ahead projections
    roster_adj_home = 0.0
    roster_adj_away = 0.0
    home_regr = REGRESSION_DEFAULT
    away_regr = REGRESSION_DEFAULT
    if regress_to_mean:
        # Blend in actual W/L record (captures luck, clutch, ST, coaching)
        records = _get_team_records()
        if home_abbr and home_abbr in records:
            rec_rating = (records[home_abbr] - 0.5) * RECORD_TO_EPA_SCALE
            home_power = (1 - RECORD_BLEND) * home_power + RECORD_BLEND * rec_rating
        if away_abbr and away_abbr in records:
            rec_rating = (records[away_abbr] - 0.5) * RECORD_TO_EPA_SCALE
            away_power = (1 - RECORD_BLEND) * away_power + RECORD_BLEND * rec_rating

        # QB-adjusted regression: teams with same starting QB are more stable
        qb_cont = _get_qb_continuity()
        if home_abbr and home_abbr in qb_cont:
            home_regr = REGRESSION_SAME_QB if qb_cont[home_abbr]["same_qb"] else REGRESSION_NEW_QB
        if away_abbr and away_abbr in qb_cont:
            away_regr = REGRESSION_SAME_QB if qb_cont[away_abbr]["same_qb"] else REGRESSION_NEW_QB

        home_power *= home_regr
        away_power *= away_regr

        # Layer 2: Roster adjustments (trades, FA, draft)
        # Applied AFTER regression — these are NEW information, not regressed
        # Splits into offensive and defensive EPA to properly weight both sides.
        #
        # DAMPENING: Raw roster EPA overstates impact because:
        #   1. Defensive EPA partially overlaps with offensive EPA (same plays)
        #   2. Scheme fit, coaching, and role changes mean talent ≠ production
        #   3. Roster depth absorbs some of the impact
        # Dampen to 40% of raw, and cap at ±0.04 EPA/play (~±3.3 pts/game)
        PLAYS_PER_SEASON = 1050
        ROSTER_DAMPEN = 0.40
        ROSTER_CAP = 0.04  # ±0.04 EPA/play max
        if home_abbr or away_abbr:
            radj = _get_roster_adjustments()
            if home_abbr and home_abbr in radj:
                r = radj[home_abbr]
                raw_adj = (r.get("net_epa") or 0.0) / PLAYS_PER_SEASON
                roster_adj_home = max(-ROSTER_CAP, min(ROSTER_CAP, raw_adj * ROSTER_DAMPEN))
                home_power += roster_adj_home
            if away_abbr and away_abbr in radj:
                r = radj[away_abbr]
                raw_adj = (r.get("net_epa") or 0.0) / PLAYS_PER_SEASON
                roster_adj_away = max(-ROSTER_CAP, min(ROSTER_CAP, raw_adj * ROSTER_DAMPEN))
                away_power += roster_adj_away

    # Layer 3: Matchup-specific adjustments (pressure, YAC, explosives)
    # Skip for year-ahead projections — adjustments wash out over 17 games
    matchup_adj = 0.0 if regress_to_mean else _matchup_adjustment(home_feats, away_feats)

    # Expected margin (home perspective)
    power_diff = home_power - away_power + HFA_EPA
    estimated_margin = power_diff * EPA_TO_MARGIN + matchup_adj

    # Win probability
    home_wp = _normal_cdf(estimated_margin / NFL_MARGIN_STD)

    # Total estimate: blend each team's offense with opponent's defense
    home_ppg = home_feats.get("s_points_per_game") or 22.0
    away_ppg = away_feats.get("s_points_per_game") or 22.0
    home_allowed = home_feats.get("s_points_allowed") or 22.0
    away_allowed = away_feats.get("s_points_allowed") or 22.0
    home_expected = (home_ppg + away_allowed) / 2.0
    away_expected = (away_ppg + home_allowed) / 2.0
    estimated_total = home_expected + away_expected

    result = {
        "available": True,
        "home_win_prob": round(home_wp, 4),
        "away_win_prob": round(1.0 - home_wp, 4),
        "estimated_margin": round(estimated_margin, 1),
        "estimated_total": round(estimated_total, 1),
        "home_power": round(home_power, 4),
        "away_power": round(away_power, 4),
        "matchup_adj": round(matchup_adj, 2),
        "roster_adj_home": round(roster_adj_home, 4),
        "roster_adj_away": round(roster_adj_away, 4),
    }

    if market_spread is not None:
        # market_spread is home spread (e.g., -3.5 means home favored)
        # P(home covers) = P(home_margin > -spread) for negative spread
        # Standard: spread_cover = P(margin + spread > 0)
        result["spread_cover_prob"] = round(
            _normal_cdf((estimated_margin + market_spread) / NFL_MARGIN_STD), 4
        )

    if market_total is not None:
        result["total_over_prob"] = round(
            _normal_cdf((estimated_total - market_total) / NFL_TOTAL_STD), 4
        )

    return result


# ─── Market Line Extraction ──────────────────────────────────────────────────

def _extract_market_lines(bookmakers_json: list, home_team: str, away_team: str) -> dict:
    """
    Extract best market lines from fixture bookmakers JSON.
    Prefers FanDuel, falls back to first available book.
    Returns spread, total, ML with devigged implied probs.
    """
    if not bookmakers_json:
        return {}

    # Prefer FanDuel
    book = None
    for b in bookmakers_json:
        if b.get("key") == "fanduel":
            book = b
            break
    if not book:
        book = bookmakers_json[0]

    result = {"book": book.get("key", "unknown")}
    markets = {m["key"]: m for m in book.get("markets", [])}

    # Spreads
    sp = markets.get("spreads")
    if sp:
        for out in sp.get("outcomes", []):
            if out["name"] == home_team:
                result["home_spread"] = out.get("point")
                result["home_spread_odds"] = out.get("price")
            elif out["name"] == away_team:
                result["away_spread"] = out.get("point")
                result["away_spread_odds"] = out.get("price")
        if result.get("home_spread_odds") and result.get("away_spread_odds"):
            h_imp = _decimal_to_implied(result["home_spread_odds"])
            a_imp = _decimal_to_implied(result["away_spread_odds"])
            if h_imp and a_imp:
                h_fair, a_fair = _devig(h_imp, a_imp)
                result["home_spread_prob"] = round(h_fair, 4)
                result["away_spread_prob"] = round(a_fair, 4)

    # Totals
    tot = markets.get("totals")
    if tot:
        for out in tot.get("outcomes", []):
            if out["name"] == "Over":
                result["total_line"] = out.get("point")
                result["over_odds"] = out.get("price")
            elif out["name"] == "Under":
                result["under_odds"] = out.get("price")
        if result.get("over_odds") and result.get("under_odds"):
            o_imp = _decimal_to_implied(result["over_odds"])
            u_imp = _decimal_to_implied(result["under_odds"])
            if o_imp and u_imp:
                o_fair, u_fair = _devig(o_imp, u_imp)
                result["over_prob"] = round(o_fair, 4)
                result["under_prob"] = round(u_fair, 4)

    # Moneyline
    ml = markets.get("h2h")
    if ml:
        for out in ml.get("outcomes", []):
            if out["name"] == home_team:
                result["home_ml_odds"] = out.get("price")
            elif out["name"] == away_team:
                result["away_ml_odds"] = out.get("price")
        if result.get("home_ml_odds") and result.get("away_ml_odds"):
            h_imp = _decimal_to_implied(result["home_ml_odds"])
            a_imp = _decimal_to_implied(result["away_ml_odds"])
            if h_imp and a_imp:
                h_fair, a_fair = _devig(h_imp, a_imp)
                result["home_ml_prob"] = round(h_fair, 4)
                result["away_ml_prob"] = round(a_fair, 4)

    return result


# ─── Game Assembly ────────────────────────────────────────────────────────────

def get_lab_games(
    db: Session,
    sport_key: str = "americanfootball_nfl",
    game_date: Optional[str] = None,
) -> dict:
    """
    Assemble full Lab view for upcoming games.

    Returns list of game dicts with:
      - fixture info (teams, date, time)
      - feature tables (home & away, grouped)
      - baseline model probs
      - market lines + implied probs
      - model-vs-market divergences (ALL flagged, ranked)
    """
    from database import Fixture

    now_utc = datetime.utcnow()
    if game_date:
        # Fetch games for a specific date
        fixtures = (
            db.query(Fixture)
            .filter(
                Fixture.sport_key == sport_key,
                Fixture.game_date_ct == game_date,
            )
            .order_by(Fixture.commence_time)
            .all()
        )
    else:
        # Fetch upcoming games (next 7 days)
        fixtures = (
            db.query(Fixture)
            .filter(
                Fixture.sport_key == sport_key,
                Fixture.commence_time >= now_utc - timedelta(hours=3),
            )
            .order_by(Fixture.commence_time)
            .limit(30)
            .all()
        )

    games = []
    all_edges = []

    for fix in fixtures:
        home = fix.home_team
        away = fix.away_team
        gdate = fix.game_date_ct or (fix.commence_time.strftime("%Y-%m-%d") if fix.commence_time else "")

        # Map to abbreviations for feature lookup
        home_abbr = _FULL_TO_ABBREV.get(home)
        away_abbr = _FULL_TO_ABBREV.get(away)

        # Fetch features + raw stats + week + players + yards
        home_feats = _get_nfl_features(home_abbr, gdate) if home_abbr else None
        away_feats = _get_nfl_features(away_abbr, gdate) if away_abbr else None
        home_raw = _get_raw_stats(home_abbr) if home_abbr else None
        away_raw = _get_raw_stats(away_abbr) if away_abbr else None
        week = _get_game_week(home_abbr, away_abbr, gdate) if (home_abbr and away_abbr) else None
        home_players = _get_team_players(home_abbr) if home_abbr else None
        away_players = _get_team_players(away_abbr) if away_abbr else None
        home_yards = _estimate_team_yards(home_raw)
        away_yards = _estimate_team_yards(away_raw)

        # Market lines
        bookmakers = fix.bookmakers if isinstance(fix.bookmakers, list) else []
        if isinstance(fix.bookmakers, str):
            try:
                bookmakers = json.loads(fix.bookmakers)
            except Exception:
                bookmakers = []
        market = _extract_market_lines(bookmakers, home, away)

        # Baseline model
        baseline = compute_baseline(
            home_feats,
            away_feats,
            market_spread=market.get("home_spread"),
            market_total=market.get("total_line"),
        )

        # Build exposed feature table
        features_exposed = {}
        for group_name, feat_list in FEATURE_GROUPS.items():
            group_data = []
            for col, label, hint in feat_list:
                h_val = home_feats.get(col) if home_feats else None
                a_val = away_feats.get(col) if away_feats else None
                group_data.append({
                    "key": col,
                    "label": label,
                    "hint": hint,
                    "home": round(h_val, 4) if h_val is not None else None,
                    "away": round(a_val, 4) if a_val is not None else None,
                })
            features_exposed[group_name] = group_data

        # Build outcome rows with divergences
        outcomes = []

        # Spread cover
        if baseline.get("spread_cover_prob") and market.get("home_spread_prob"):
            div_pp = round((baseline["spread_cover_prob"] - market["home_spread_prob"]) * 100, 1)
            outcomes.append({
                "type": "spread_cover",
                "label": f"{home} {market.get('home_spread', '')}",
                "line": market.get("home_spread"),
                "baseline_prob": baseline["spread_cover_prob"],
                "market_prob": market["home_spread_prob"],
                "divergence_pp": div_pp,
                "market_odds": market.get("home_spread_odds"),
            })

        # Total over
        if baseline.get("total_over_prob") and market.get("over_prob"):
            div_pp = round((baseline["total_over_prob"] - market["over_prob"]) * 100, 1)
            outcomes.append({
                "type": "total_over",
                "label": f"Over {market.get('total_line', '')}",
                "line": market.get("total_line"),
                "baseline_prob": baseline["total_over_prob"],
                "market_prob": market["over_prob"],
                "divergence_pp": div_pp,
                "market_odds": market.get("over_odds"),
            })

        # Total under
        if baseline.get("total_over_prob") and market.get("under_prob"):
            under_base = round(1.0 - baseline["total_over_prob"], 4)
            div_pp = round((under_base - market["under_prob"]) * 100, 1)
            outcomes.append({
                "type": "total_under",
                "label": f"Under {market.get('total_line', '')}",
                "line": market.get("total_line"),
                "baseline_prob": under_base,
                "market_prob": market["under_prob"],
                "divergence_pp": div_pp,
                "market_odds": market.get("under_odds"),
            })

        # Moneyline
        if baseline.get("home_win_prob") and market.get("home_ml_prob"):
            div_pp = round((baseline["home_win_prob"] - market["home_ml_prob"]) * 100, 1)
            outcomes.append({
                "type": "moneyline",
                "label": f"{home} ML",
                "line": None,
                "baseline_prob": baseline["home_win_prob"],
                "market_prob": market["home_ml_prob"],
                "divergence_pp": div_pp,
                "market_odds": market.get("home_ml_odds"),
            })
        if baseline.get("away_win_prob") and market.get("away_ml_prob"):
            div_pp = round((baseline["away_win_prob"] - market["away_ml_prob"]) * 100, 1)
            outcomes.append({
                "type": "moneyline",
                "label": f"{away} ML",
                "line": None,
                "baseline_prob": baseline["away_win_prob"],
                "market_prob": market["away_ml_prob"],
                "divergence_pp": div_pp,
                "market_odds": market.get("away_ml_odds"),
            })

        # Collect edges for the Edge Board
        for o in outcomes:
            if abs(o["divergence_pp"]) >= 1.0:  # flag any 1+pp divergence
                all_edges.append({
                    "fixture_id": fix.id,
                    "game": f"{away} @ {home}",
                    "game_date": gdate,
                    "week": week,
                    **o,
                })

        game = {
            "fixture_id": fix.id,
            "home_team": home,
            "away_team": away,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "game_date": gdate,
            "week": week,
            "commence_time": fix.commence_time.isoformat() if fix.commence_time else None,
            "features": features_exposed,
            "features_available": bool(home_feats and away_feats),
            "baseline": baseline,
            "market": market,
            "outcomes": outcomes,
            "home_raw": home_raw,
            "away_raw": away_raw,
            "home_players": home_players,
            "away_players": away_players,
            "home_yards": home_yards,
            "away_yards": away_yards,
        }
        games.append(game)

    # Sort edges by absolute divergence (largest first)
    all_edges.sort(key=lambda e: abs(e["divergence_pp"]), reverse=True)

    return {
        "sport": sport_key,
        "games": games,
        "game_count": len(games),
        "edges": all_edges,
        "edge_count": len(all_edges),
        # Model constants for JS-side recomputation
        "model_constants": {
            "HFA_EPA": HFA_EPA,
            "EPA_TO_MARGIN": EPA_TO_MARGIN,
            "NFL_MARGIN_STD": NFL_MARGIN_STD,
            "NFL_TOTAL_STD": NFL_TOTAL_STD,
            "AVG_PASS_YPA": _AVG_PASS_YPA,
            "AVG_RUSH_YPC": _AVG_RUSH_YPC,
            "EPA_TO_YPA_SCALE": _EPA_TO_YPA_SCALE,
        },
    }


# ─── Adjustment Logging ──────────────────────────────────────────────────────

def save_adjustment(db: Session, data: dict) -> dict:
    """
    Log a user adjustment (or model-flagged edge acceptance).

    Required: fixture_id, game_date, sport, home_team, away_team,
              outcome_type, outcome_line, baseline_prob, market_prob
    Optional: adjusted_prob, feature_changes, user_notes,
              bet_placed, bet_side, bet_odds_decimal, bet_amount
    """
    from database import LabAdjustment

    adj = LabAdjustment(
        fixture_id=data.get("fixture_id"),
        game_date=data["game_date"],
        sport=data["sport"],
        home_team=data["home_team"],
        away_team=data["away_team"],
        outcome_type=data["outcome_type"],
        outcome_line=data.get("outcome_line"),
        baseline_prob=data.get("baseline_prob"),
        market_prob=data.get("market_prob"),
        adjusted_prob=data.get("adjusted_prob"),
        has_override=data.get("adjusted_prob") is not None,
        model_edge_pp=round((data.get("baseline_prob", 0) - data.get("market_prob", 0)) * 100, 1)
            if data.get("baseline_prob") is not None and data.get("market_prob") is not None
            else None,
        user_edge_pp=round((data["adjusted_prob"] - data.get("market_prob", 0)) * 100, 1)
            if data.get("adjusted_prob") is not None and data.get("market_prob") is not None
            else None,
        adjustment_type=data.get("adjustment_type", "outcome_level" if data.get("adjusted_prob") else "none"),
        feature_changes=json.dumps(data["feature_changes"]) if data.get("feature_changes") else None,
        user_notes=data.get("user_notes"),
        bet_placed=data.get("bet_placed", False),
        bet_side=data.get("bet_side"),
        bet_odds_decimal=data.get("bet_odds_decimal"),
        bet_amount=data.get("bet_amount"),
    )
    db.add(adj)
    db.commit()
    db.refresh(adj)
    return {"id": adj.id, "saved": True}


# ─── Settlement ───────────────────────────────────────────────────────────────

def settle_lab_entries(db: Session) -> dict:
    """
    Settle unsettled lab entries using game scores from fixtures/bets tables.
    Matches fixture_id to find final scores, determines WIN/LOSS/PUSH.
    """
    from database import LabAdjustment

    unsettled = (
        db.query(LabAdjustment)
        .filter(LabAdjustment.settled == False, LabAdjustment.fixture_id.isnot(None))
        .all()
    )
    if not unsettled:
        return {"settled": 0, "message": "No unsettled lab entries"}

    # Fetch game results — look in mock_bets settlement data or scores API
    # For now, check if the fixture's game has completed by looking for
    # resolved scores in mock_bet_legs (which get settled via OddsAPI scores)
    settled_count = 0
    for adj in unsettled:
        result = _resolve_lab_outcome(db, adj)
        if result:
            adj.outcome = result["outcome"]
            if adj.bet_placed and adj.bet_odds_decimal and adj.bet_amount:
                if result["outcome"] == "WIN":
                    adj.actual_profit = round(adj.bet_amount * (adj.bet_odds_decimal - 1), 2)
                elif result["outcome"] == "LOSS":
                    adj.actual_profit = round(-adj.bet_amount, 2)
                else:  # PUSH
                    adj.actual_profit = 0.0
            adj.settled = True
            adj.settled_at = datetime.utcnow()
            settled_count += 1

    db.commit()
    return {"settled": settled_count, "remaining": len(unsettled) - settled_count}


def _resolve_lab_outcome(db: Session, adj) -> Optional[dict]:
    """
    Try to resolve a lab entry's outcome from available game data.
    Uses mock_bet_legs with matching fixture_id that have been settled.
    """
    # Look for settled legs from this fixture
    row = db.execute(
        sa_text("""
            SELECT resolved_home_score, resolved_away_score
            FROM mock_bet_legs
            WHERE fixture_id = :fid AND resolved_home_score IS NOT NULL
            LIMIT 1
        """),
        {"fid": adj.fixture_id},
    ).fetchone()

    if not row or row[0] is None:
        return None

    home_score, away_score = float(row[0]), float(row[1])
    margin = home_score - away_score  # positive = home won

    if adj.outcome_type == "spread_cover" and adj.outcome_line is not None:
        # home spread cover: margin + spread > 0
        adjusted = margin + adj.outcome_line
        if adjusted > 0:
            return {"outcome": "WIN"}
        elif adjusted < 0:
            return {"outcome": "LOSS"}
        return {"outcome": "PUSH"}

    elif adj.outcome_type in ("total_over", "total_under") and adj.outcome_line is not None:
        total = home_score + away_score
        if adj.outcome_type == "total_over":
            if total > adj.outcome_line:
                return {"outcome": "WIN"}
            elif total < adj.outcome_line:
                return {"outcome": "LOSS"}
            return {"outcome": "PUSH"}
        else:
            if total < adj.outcome_line:
                return {"outcome": "WIN"}
            elif total > adj.outcome_line:
                return {"outcome": "LOSS"}
            return {"outcome": "PUSH"}

    elif adj.outcome_type == "moneyline":
        # Determine which team was bet on from bet_side or label context
        # Check if home or away team is in the adjustment's label context
        if margin > 0 and adj.home_team and adj.bet_side and adj.home_team in (adj.bet_side or ""):
            return {"outcome": "WIN"}
        elif margin < 0 and adj.away_team and adj.bet_side and adj.away_team in (adj.bet_side or ""):
            return {"outcome": "WIN"}
        elif margin == 0:
            return {"outcome": "PUSH"}
        # Can't determine side — skip
        return None

    return None


# ─── Track Record / Measurement ──────────────────────────────────────────────

def get_track_record(db: Session, sport: Optional[str] = None) -> dict:
    """
    Compute the Lab measurement dashboard.

    Judges EVERYTHING by realized ROI at adequate N.
    CLV is NOT used as the arbiter. Adequate sample size is the
    anti-variance guard.

    Segments:
      - Model edges (baseline diverged from market, user agreed)
      - User overrides (user changed the probability)
      - By outcome type, by direction (bullish/bearish), by sport
    """
    from database import LabAdjustment

    q = db.query(LabAdjustment).filter(LabAdjustment.settled == True)
    if sport:
        q = q.filter(LabAdjustment.sport == sport)
    settled = q.all()

    # Always fetch recent adjustments (settled or not)
    recent = (
        db.query(LabAdjustment)
        .order_by(LabAdjustment.created_at.desc())
        .limit(20)
        .all()
    )
    recent_list = [
        {
            "id": r.id,
            "game_date": r.game_date,
            "matchup": f"{r.away_team} @ {r.home_team}",
            "outcome_type": r.outcome_type,
            "line": r.outcome_line,
            "baseline_prob": r.baseline_prob,
            "market_prob": r.market_prob,
            "adjusted_prob": r.adjusted_prob,
            "has_override": r.has_override,
            "model_edge_pp": r.model_edge_pp,
            "user_edge_pp": r.user_edge_pp,
            "bet_placed": r.bet_placed,
            "outcome": r.outcome,
            "profit": r.actual_profit,
            "settled": r.settled,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent
    ]

    if not settled:
        return {
            "total_settled": 0,
            "message": "No settled lab entries yet. Results accumulate as games complete.",
            "overall": {},
            "model_edges": {},
            "user_overrides": {},
            "segments": [],
            "recent": recent_list,
            "measurement_note": (
                "All results judged by REALIZED ROI at adequate N. "
                "CLV is not used as arbiter — it misleads on soft markets. "
                f"Minimum N for conclusions: {MIN_N_MEANINGFUL} (meaningful), {MIN_N_CONFIDENT} (confident)."
            ),
        }

    # ── Overall stats ──
    def _compute_roi(entries):
        """Compute realized ROI for a set of entries that had bets placed."""
        bet_entries = [e for e in entries if e.bet_placed and e.actual_profit is not None]
        if not bet_entries:
            return {"n": 0, "n_bets": 0, "roi": None, "pnl": None, "adequacy": "no_bets"}
        total_staked = sum(e.bet_amount or 0 for e in bet_entries)
        total_pnl = sum(e.actual_profit for e in bet_entries)
        n = len(bet_entries)
        roi = round(total_pnl / total_staked * 100, 1) if total_staked > 0 else 0
        wins = sum(1 for e in bet_entries if e.outcome == "WIN")
        wr = round(wins / n * 100, 1) if n > 0 else 0
        if n >= MIN_N_CONFIDENT:
            adequacy = "confident"
        elif n >= MIN_N_MEANINGFUL:
            adequacy = "meaningful"
        elif n >= MIN_N_SHOW:
            adequacy = "preliminary"
        else:
            adequacy = "insufficient"
        return {
            "n": n,
            "n_bets": n,
            "wins": wins,
            "win_rate": wr,
            "roi": roi,
            "pnl": round(total_pnl, 2),
            "total_staked": round(total_staked, 2),
            "adequacy": adequacy,
        }

    # All settled
    overall = _compute_roi(settled)
    overall["total_settled"] = len(settled)

    # Model edges: entries where model diverged from market (model_edge_pp != 0)
    # and user did NOT override
    model_only = [e for e in settled if not e.has_override and abs(e.model_edge_pp or 0) > 0]
    model_edges = _compute_roi(model_only)

    # User overrides: entries where user changed the probability
    overrides = [e for e in settled if e.has_override]
    user_overrides = _compute_roi(overrides)

    # ── Segmented analysis ──
    segments = []

    # By outcome type
    for otype in ["spread_cover", "total_over", "total_under", "moneyline"]:
        subset = [e for e in settled if e.outcome_type == otype]
        if subset:
            seg = _compute_roi(subset)
            seg["segment"] = "outcome_type"
            seg["value"] = otype
            seg["total_entries"] = len(subset)
            segments.append(seg)

    # By direction (bullish = model says higher than market, bearish = lower)
    bullish = [e for e in settled if (e.model_edge_pp or 0) > 0]
    bearish = [e for e in settled if (e.model_edge_pp or 0) < 0]
    if bullish:
        seg = _compute_roi(bullish)
        seg["segment"] = "direction"
        seg["value"] = "bullish (model > market)"
        seg["total_entries"] = len(bullish)
        segments.append(seg)
    if bearish:
        seg = _compute_roi(bearish)
        seg["segment"] = "direction"
        seg["value"] = "bearish (model < market)"
        seg["total_entries"] = len(bearish)
        segments.append(seg)

    # By override vs no-override
    if model_only:
        seg = _compute_roi(model_only)
        seg["segment"] = "source"
        seg["value"] = "model edge (no override)"
        seg["total_entries"] = len(model_only)
        segments.append(seg)
    if overrides:
        seg = _compute_roi(overrides)
        seg["segment"] = "source"
        seg["value"] = "user override"
        seg["total_entries"] = len(overrides)
        segments.append(seg)

    # ── User override track record (the key question) ──
    # "Do your overrides BEAT the baseline?"
    override_verdict = None
    if overrides:
        bet_overrides = [e for e in overrides if e.bet_placed and e.actual_profit is not None]
        if len(bet_overrides) >= MIN_N_MEANINGFUL:
            override_roi = user_overrides["roi"]
            baseline_roi = model_edges.get("roi")
            if baseline_roi is not None and override_roi is not None:
                if override_roi > baseline_roi + 3:
                    override_verdict = "Your overrides are adding edge (+{:.1f}pp vs baseline). Your judgment is real here.".format(
                        override_roi - baseline_roi
                    )
                elif override_roi < baseline_roi - 3:
                    override_verdict = "Your overrides are costing edge ({:.1f}pp vs baseline). Consider trusting the model more.".format(
                        override_roi - baseline_roi
                    )
                else:
                    override_verdict = "Your overrides perform similarly to the baseline (within 3pp). Keep accumulating data."
        elif len(bet_overrides) >= MIN_N_SHOW:
            override_verdict = f"Only {len(bet_overrides)} override bets settled — need {MIN_N_MEANINGFUL} for a meaningful read."

    # recent_list already fetched above (before the settled check)

    return {
        "total_settled": len(settled),
        "overall": overall,
        "model_edges": model_edges,
        "user_overrides": user_overrides,
        "override_verdict": override_verdict,
        "segments": segments,
        "recent": recent_list,
        "measurement_note": (
            "All results judged by REALIZED ROI at adequate N. "
            "CLV is not used as arbiter — it misleads on soft markets. "
            f"Minimum N for conclusions: {MIN_N_MEANINGFUL} (meaningful), {MIN_N_CONFIDENT} (confident)."
        ),
    }
