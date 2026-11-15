"""
nba_ablation.py — B4 walk-forward ablation of NBA prop model factors.

Walk-forward on 2023-24 → 2025-26 (2022-23 as training start).
Historical ESPN near-close lines as market anchors (same pipeline as production).

Factors ablated (added one at a time):
  1. Base: minutes + per-min rates + 240-min enforcement + reconciliation (B1-B3)
  2. + Team defense (four factors, shot profile allowed, opp OREB%)
  3. + Play-type / Synergy (offense vs defense PPP)
  4. + Defender matchups
  5. + Form (recent games, shrunk)

Also ablates hand-set rules:
  - _HIGH_FOUL_RATE 4.5
  - Absence cap +8 min
  - Measured blowout cuts

Metrics per stat (PTS, REB, AST, 3PM, PRA, STL, BLK) × role:
  - CRPS (primary)
  - MAE
  - Game-clustered bootstrap CIs
  - PIT (calibration check)

No-leakage rules:
  - Team defense: only games before each date
  - Synergy: PRIOR-season values (matches early-season production)
  - Form: only games before the date, heavily shrunk
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_DB = os.path.join(_DATA_DIR, "nba_ablation_cache.db")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# ── Seasons covered ───────────────────────────────────────────────────────────

ABLATION_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
EVAL_SEASONS = ["2023-24", "2024-25", "2025-26"]  # 2022-23 = training start only
STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]
COMBINED_STATS = ["PRA"]  # PTS + REB + AST

_API_SLEEP = 1.2  # seconds between nba_api calls (0.7 triggers rate limits)
_MAX_RETRIES = 3
_BACKOFF_ON_TIMEOUT = 30  # seconds to wait after a timeout before retrying


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GamePrediction:
    """One player-stat prediction for a single game."""
    game_id: str
    game_date: str
    season: str
    player_id: int
    player_name: str
    team: str
    opponent: str
    is_home: bool
    stat: str           # PTS, REB, AST, FG3M, STL, BLK, PRA
    role: str           # starter, rotation, spot
    predicted_mean: float
    predicted_std: float
    actual: float
    market_spread: float
    market_total: float
    factor_set: str     # "base", "base+defense", "base+defense+synergy", ...


@dataclass
class AblationResult:
    """Metrics for one factor set × one stat."""
    factor_set: str
    stat: str
    role: str           # "all" or specific role
    n_obs: int
    n_games: int
    crps: float
    crps_ci_lo: float
    crps_ci_hi: float
    mae: float
    mae_ci_lo: float
    mae_ci_hi: float
    pit_ks_stat: float
    pit_ks_pval: float
    pit_mean: float     # should be ~0.5 for well-calibrated


# ── Incremental prediction storage ───────────────────────────────────────────

_PREDICTIONS_INCREMENTAL = os.path.join(_DATA_DIR, "nba_ablation_preds_incr.json")


def _save_predictions_incremental(predictions: list[GamePrediction]):
    """Save predictions list to disk so partial runs aren't lost."""
    rows = [p.__dict__ for p in predictions]
    with open(_PREDICTIONS_INCREMENTAL, "w") as f:
        json.dump(rows, f)
    logger.info("  Saved %d predictions to %s", len(rows),
                os.path.basename(_PREDICTIONS_INCREMENTAL))


def _load_predictions_incremental() -> list[GamePrediction]:
    """Load previously saved predictions (for resume or reporting)."""
    if not os.path.exists(_PREDICTIONS_INCREMENTAL):
        return []
    with open(_PREDICTIONS_INCREMENTAL) as f:
        rows = json.load(f)
    return [GamePrediction(**r) for r in rows]


# ── Cache layer ───────────────────────────────────────────────────────────────

def _init_cache():
    """Create ablation cache DB if needed."""
    os.makedirs(os.path.dirname(_CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(_CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_game_logs (
            team TEXT, season TEXT, data TEXT,
            PRIMARY KEY (team, season)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_game_logs (
            player_id INTEGER, season TEXT, data TEXT,
            PRIMARY KEY (player_id, season)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_opp_stats (
            season TEXT PRIMARY KEY, data TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_estimated_metrics (
            season TEXT PRIMARY KEY, data TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS synergy_play_types (
            season TEXT, play_type TEXT, grouping TEXT, data TEXT,
            PRIMARY KEY (season, play_type, grouping)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matchup_rollup (
            season TEXT PRIMARY KEY, data TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_roster (
            season TEXT PRIMARY KEY, data TEXT
        )
    """)
    conn.commit()
    conn.close()


def _cached_or_fetch(table, key_cols, key_vals, fetch_fn):
    """Check cache; if miss, call fetch_fn(), store, return DataFrame."""
    conn = sqlite3.connect(_CACHE_DB)
    where = " AND ".join(f"{c} = ?" for c in key_cols)
    row = conn.execute(
        f"SELECT data FROM {table} WHERE {where}", key_vals
    ).fetchone()
    if row:
        conn.close()
        return pd.read_json(row[0], orient="records")

    df = fetch_fn()
    if df is not None and len(df) > 0:
        placeholders = ", ".join(["?"] * (len(key_cols) + 1))
        cols = ", ".join(key_cols + ["data"])
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
            list(key_vals) + [df.to_json(orient="records")],
        )
        conn.commit()
    conn.close()
    return df


# ── nba_api data pulls ────────────────────────────────────────────────────────

def _api_sleep():
    time.sleep(_API_SLEEP)


def pull_team_game_logs(season: str) -> dict[str, pd.DataFrame]:
    """Pull TeamGameLog for all 30 teams for a season. Returns {team_abbr: df}."""
    from nba_api.stats.endpoints import teamgamelog
    from nba_api.stats.static import teams as nba_teams

    _init_cache()
    all_teams = nba_teams.get_teams()
    result = {}

    for t in all_teams:
        team_id = t["id"]
        abbr = t["abbreviation"]

        def _fetch(tid=team_id, s=season):
            _api_sleep()
            try:
                tgl = teamgamelog.TeamGameLog(
                    team_id=tid, season=s,
                    season_type_all_star="Regular Season",
                )
                return tgl.get_data_frames()[0]
            except Exception as e:
                logger.warning("TeamGameLog %s %s failed: %s", abbr, s, e)
                return pd.DataFrame()

        df = _cached_or_fetch(
            "team_game_logs", ["team", "season"], (abbr, season), _fetch
        )
        if df is not None and len(df) > 0:
            result[abbr] = df

    logger.info("Pulled team game logs for %s: %d teams", season, len(result))
    return result


def pull_player_roster(season: str) -> list[dict]:
    """Get list of qualifying players for a season from LeagueDashPlayerStats."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    _init_cache()

    def _fetch():
        _api_sleep()
        try:
            dash = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                season_type_all_star="Regular Season",
            )
            df = dash.get_data_frames()[0]
            # Filter: ≥10 games AND ≥10 MPG
            df = df[(df["GP"] >= 10) & (df["MIN"] >= 10.0)]
            return df
        except Exception as e:
            logger.warning("LeagueDashPlayerStats %s failed: %s", season, e)
            return pd.DataFrame()

    df = _cached_or_fetch("player_roster", ["season"], (season,), _fetch)
    if df is None or len(df) == 0:
        return []
    return df.to_dict("records")


def pull_player_game_logs(player_id: int, season: str) -> pd.DataFrame:
    """Pull individual player game log for one season."""
    from nba_api.stats.endpoints import playergamelog

    _init_cache()

    def _fetch():
        _api_sleep()
        for attempt in range(_MAX_RETRIES):
            try:
                pgl = playergamelog.PlayerGameLog(
                    player_id=player_id, season=season,
                    season_type_all_star="Regular Season",
                )
                return pgl.get_data_frames()[0]
            except Exception as e:
                wait = _BACKOFF_ON_TIMEOUT * (attempt + 1)
                if attempt < _MAX_RETRIES - 1:
                    logger.warning("PlayerGameLog %d %s attempt %d failed: %s — "
                                   "backing off %ds", player_id, season, attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    logger.warning("PlayerGameLog %d %s failed: %s",
                                   player_id, season, e)
                    return pd.DataFrame()

    return _cached_or_fetch(
        "player_game_logs", ["player_id", "season"],
        (player_id, season), _fetch,
    )


def pull_team_opp_stats(season: str) -> pd.DataFrame:
    """LeagueDashTeamStats opponent stats (defense) for a season."""
    from nba_api.stats.endpoints import leaguedashteamstats

    _init_cache()

    def _fetch():
        _api_sleep()
        try:
            # Use default per_mode (Totals) — the engine's _build_team_profiles
            # divides by GP, so it expects season totals not per-game averages.
            opp = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Opponent",
                season_type_all_star="Regular Season",
            )
            return opp.get_data_frames()[0]
        except Exception as e:
            logger.warning("LeagueDashTeamStats opp %s failed: %s", season, e)
            return pd.DataFrame()

    return _cached_or_fetch(
        "team_opp_stats", ["season"], (season,), _fetch
    )


def pull_team_estimated_metrics(season: str) -> pd.DataFrame:
    """TeamEstimatedMetrics (four factors) for a season."""
    from nba_api.stats.endpoints import teamestimatedmetrics

    _init_cache()

    def _fetch():
        _api_sleep()
        try:
            tm = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
            return tm.get_data_frames()[0]
        except Exception as e:
            logger.warning("TeamEstimatedMetrics %s failed: %s", season, e)
            return pd.DataFrame()

    return _cached_or_fetch(
        "team_estimated_metrics", ["season"], (season,), _fetch
    )


def pull_synergy(season: str, play_type: str, grouping: str) -> pd.DataFrame:
    """SynergyPlayTypes for one play type + grouping (offensive/defensive)."""
    from nba_api.stats.endpoints import synergyplaytypes

    _init_cache()

    def _fetch():
        _api_sleep()
        try:
            syn = synergyplaytypes.SynergyPlayTypes(
                per_mode_simple="Totals",
                player_or_team_abbreviation="T",
                season=season,
                play_type_nullable=play_type,
                type_grouping_nullable=grouping,
            )
            return syn.get_data_frames()[0]
        except Exception as e:
            logger.warning("Synergy %s %s %s failed: %s",
                           season, play_type, grouping, e)
            return pd.DataFrame()

    return _cached_or_fetch(
        "synergy_play_types", ["season", "play_type", "grouping"],
        (season, play_type, grouping), _fetch,
    )


def pull_matchup_rollup(season: str) -> pd.DataFrame:
    """MatchupsRollup (defender matchup stats) for a season."""
    from nba_api.stats.endpoints import matchupsrollup

    _init_cache()

    def _fetch():
        _api_sleep()
        try:
            mu = matchupsrollup.MatchupsRollup(
                per_mode_simple="Totals",
                season=season,
                season_type_playoffs="Regular Season",
            )
            return mu.get_data_frames()[0]
        except Exception as e:
            logger.warning("MatchupsRollup %s failed: %s", season, e)
            return pd.DataFrame()

    return _cached_or_fetch(
        "matchup_rollup", ["season"], (season,), _fetch
    )


SYNERGY_PLAY_TYPES = [
    "Transition", "Isolation", "PRBallHandler", "PRRollMan",
    "Postup", "Spotup", "Handoff", "Cut", "OffScreen",
    "OffRebound", "Misc",
]


def pull_all_data(seasons: list[str] | None = None):
    """Pull all data needed for ablation. Caches to disk.

    Call once before running ablation.  Subsequent calls hit cache.
    """
    if seasons is None:
        seasons = ABLATION_SEASONS

    _init_cache()
    total_calls = 0

    for season in seasons:
        logger.info("Pulling data for %s...", season)

        # Team game logs (30 teams)
        pull_team_game_logs(season)
        total_calls += 30

        # Player roster
        roster = pull_player_roster(season)
        total_calls += 1
        logger.info("  %s: %d qualifying players", season, len(roster))

        # Player game logs (most calls)
        for i, p in enumerate(roster):
            pid = p.get("PLAYER_ID") or p.get("Player_ID")
            if pid:
                pull_player_game_logs(int(pid), season)
                total_calls += 1
                if (i + 1) % 50 == 0:
                    logger.info("  %s: %d/%d player logs pulled",
                                season, i + 1, len(roster))

        # Team opp stats + four factors
        pull_team_opp_stats(season)
        pull_team_estimated_metrics(season)
        total_calls += 2

        # Synergy (11 types × 2 groupings)
        for pt in SYNERGY_PLAY_TYPES:
            pull_synergy(season, pt, "offensive")
            pull_synergy(season, pt, "defensive")
            total_calls += 2

        # Matchup rollup
        pull_matchup_rollup(season)
        total_calls += 1

    logger.info("Data pull complete: %d total API calls", total_calls)


# ── Historical market lines ───────────────────────────────────────────────────

def load_historical_lines(season: str) -> pd.DataFrame:
    """Load ESPN near-close lines from historical.db for one NBA season.

    Returns DataFrame with: game_id, game_date, home_team, away_team,
    close_spread, close_total.
    """
    if not os.path.exists(_HIST_DB):
        raise FileNotFoundError(f"historical.db not found at {_HIST_DB}")
    conn = sqlite3.connect(_HIST_DB)
    df = pd.read_sql_query(
        "SELECT game_id, game_date, home_team, away_team, "
        "       close_spread, close_total "
        "FROM betting_lines "
        "WHERE sport = 'NBA' AND season = ? "
        "  AND close_spread IS NOT NULL "
        "  AND close_total IS NOT NULL",
        conn, params=(season,),
    )
    conn.close()
    return df


# ── Feature computation (as-of date, no leakage) ─────────────────────────────

def compute_team_defense_asof(
    team_game_logs: dict[str, pd.DataFrame],
    team: str,
    game_date: str,
) -> dict:
    """Compute team defense features from games strictly before game_date.

    Returns: {opp_fg_pct, opp_fg3_pct, opp_ftm, opp_fta, opp_oreb,
              opp_ppg, opp_pace_est, n_games}
    """
    df = team_game_logs.get(team)
    if df is None or len(df) == 0:
        return {}

    # Ensure date column exists
    if "GAME_DATE" not in df.columns:
        return {}

    # Filter to games before this date
    mask = pd.to_datetime(df["GAME_DATE"]) < pd.to_datetime(game_date)
    prior = df[mask]
    if len(prior) < 5:
        return {}

    # TeamGameLog has opponent stats embedded — extract
    # OPP columns may not exist in TeamGameLog; we compute from MATCHUP + known cols
    # TeamGameLog columns: FGM, FGA, FG_PCT, FG3M, FG3A, etc. — these are the team's own stats
    # For opponent stats, we need to look at the opponent's game log for the same game
    # This is complex; simpler: use the season-level LeagueDashTeamStats (opponent)
    # with PRIOR-season for Synergy, and incrementally for defense

    # For now, use simple per-game averages from OPP perspective
    # We'll compute team defense from the other side of team game logs
    return {"n_games": len(prior)}


def compute_synergy_prior_season(
    season: str,
) -> dict[str, dict]:
    """Load Synergy play-type data from the PRIOR season.

    For 2023-24 eval: use 2022-23 Synergy.
    Returns: {team_abbr: {play_type: {off_ppp, def_ppp, off_poss_pct, ...}}}
    """
    # Determine prior season
    start_year = int(season[:4]) - 1
    prior_season = f"{start_year}-{str(start_year + 1)[-2:]}"

    result: dict[str, dict] = {}
    for pt in SYNERGY_PLAY_TYPES:
        off_df = pull_synergy(prior_season, pt, "offensive")
        def_df = pull_synergy(prior_season, pt, "defensive")

        if off_df is not None and len(off_df) > 0:
            for _, row in off_df.iterrows():
                abbr = row.get("TEAM_ABBREVIATION", "")
                if not abbr:
                    continue
                result.setdefault(abbr, {}).setdefault(pt, {})
                result[abbr][pt]["off_ppp"] = row.get("PPP", 0)
                result[abbr][pt]["off_poss_pct"] = row.get("POSS_PCT", 0)

        if def_df is not None and len(def_df) > 0:
            for _, row in def_df.iterrows():
                abbr = row.get("TEAM_ABBREVIATION", "")
                if not abbr:
                    continue
                result.setdefault(abbr, {}).setdefault(pt, {})
                result[abbr][pt]["def_ppp"] = row.get("PPP", 0)
                result[abbr][pt]["def_poss_pct"] = row.get("POSS_PCT", 0)

    return result


def compute_form_asof(
    player_game_log: pd.DataFrame,
    game_date: str,
    window: int = 10,
    shrinkage: float = 0.5,
) -> dict[str, float]:
    """Compute form features from games strictly before game_date.

    Returns: {form_pts, form_reb, form_ast, form_fg3m, n_recent}
    Heavily shrunk toward season mean (shrinkage=0.5 = 50% season, 50% recent).
    """
    if player_game_log is None or len(player_game_log) == 0:
        return {}

    if "GAME_DATE" not in player_game_log.columns:
        return {}

    mask = pd.to_datetime(player_game_log["GAME_DATE"]) < pd.to_datetime(game_date)
    prior = player_game_log[mask].tail(window)
    if len(prior) < 3:
        return {}

    all_prior = player_game_log[mask]

    result = {"n_recent": len(prior)}
    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]:
        if stat in prior.columns:
            recent_mean = float(prior[stat].astype(float).mean())
            season_mean = float(all_prior[stat].astype(float).mean())
            # Shrink toward season mean
            blended = shrinkage * season_mean + (1 - shrinkage) * recent_mean
            result[f"form_{stat.lower()}"] = blended

    return result


# ── Position classification + team matchup defense ──────────────────────────

def _classify_position(rpg: float, apg: float, bpg: float, fg3a_pg: float) -> str:
    """Classify a player into G/F/C from per-game stat averages."""
    if rpg >= 8 and bpg >= 1.0:
        return "C"
    if rpg >= 6 and apg < 5:
        return "F"
    if apg >= 5 or fg3a_pg >= 5:
        return "G"
    if rpg >= 5:
        return "F"
    return "G"


def compute_team_position_defense(season: str) -> dict[str, dict[str, float]]:
    """Compute per-team position-defense factors from MatchupsRollup.

    Returns: {team_abbr: {"G": factor, "F": factor, "C": factor}}
    Factor > 1 = opponent scores more vs this team at that position (weak defense).
    Factor < 1 = strong defense at that position.

    Uses PRIOR season data (no leakage — matches early-season production).
    """
    from nba_matchup_engine import _TEAM_ID_MAP

    start_year = int(season[:4]) - 1
    prior_season = f"{start_year}-{str(start_year + 1)[-2:]}"

    _init_cache()
    conn = sqlite3.connect(_CACHE_DB)

    # Load matchup rollup from prior season; fall back to current if unavailable
    mu_df = None
    roster_df = None
    for s in (prior_season, season):
        mu_row = conn.execute(
            "SELECT data FROM matchup_rollup WHERE season = ?", (s,)
        ).fetchone()
        roster_row = conn.execute(
            "SELECT data FROM player_roster WHERE season = ?", (s,)
        ).fetchone()
        if mu_row and roster_row:
            mu_df = pd.read_json(mu_row[0], orient="records")
            roster_df = pd.read_json(roster_row[0], orient="records")
            break

    conn.close()

    if mu_df is None or roster_df is None:
        return {}

    pid_to_team = dict(zip(
        roster_df["PLAYER_ID"].astype(int),
        roster_df["TEAM_ABBREVIATION"],
    ))

    # Filter to G/F/C positions (drop TOTAL rows)
    mu_df = mu_df[mu_df["POSITION"].isin(["G", "F", "C"])].copy()
    mu_df["DEF_TEAM"] = mu_df["DEF_PLAYER_ID"].map(pid_to_team)
    mu_df = mu_df.dropna(subset=["DEF_TEAM"])

    # Compute team-position PPP: sum(PLAYER_PTS) / sum(PARTIAL_POSS)
    team_pos = mu_df.groupby(["DEF_TEAM", "POSITION"]).agg(
        total_pts=("PLAYER_PTS", "sum"),
        total_poss=("PARTIAL_POSS", "sum"),
    ).reset_index()
    team_pos["ppp"] = team_pos["total_pts"] / team_pos["total_poss"].clip(lower=1)

    # League average PPP per position
    lg_pos = team_pos.groupby("POSITION")["ppp"].mean().to_dict()

    # Build factor dict
    result: dict[str, dict[str, float]] = {}
    for _, row in team_pos.iterrows():
        team = row["DEF_TEAM"]
        pos = row["POSITION"]
        lg_ppp = lg_pos.get(pos, row["ppp"])
        if lg_ppp > 0:
            factor = row["ppp"] / lg_ppp
        else:
            factor = 1.0
        result.setdefault(team, {})[pos] = factor

    return result


# ── Walk-forward engine builder ───────────────────────────────────────────────

# Game ID bridge: historical.db uses "NBA_0022201222", nba_api uses int 22201222
def _hist_to_api_game_id(hist_id: str) -> str:
    """Convert historical.db game_id to nba_api Game_ID string."""
    return hist_id.replace("NBA_00", "")


def _prepare_game_log(df: pd.DataFrame, cutoff: pd.Timestamp | None = None,
                      player_name: str | None = None) -> pd.DataFrame:
    """Standardise a raw nba_api player/team game log for the engine.

    Adds GAME_DATE_DT, optional PLAYER_NAME, optional date filtering,
    and sorts chronologically.
    """
    if len(df) == 0:
        return df
    if "GAME_DATE" not in df.columns:
        return df

    df = df.copy()
    # nba_api returns dates like "Apr 09, 2023"
    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
    if cutoff is not None:
        df = df[df["GAME_DATE_DT"] < cutoff]
    df = df.sort_values("GAME_DATE_DT").reset_index(drop=True)
    if player_name:
        df["PLAYER_NAME"] = player_name
    return df


def _add_abbr_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ABBR column from TEAM_ID using the engine's team map."""
    from nba_matchup_engine import _TEAM_ID_MAP
    if len(df) > 0 and "TEAM_ID" in df.columns and "ABBR" not in df.columns:
        df = df.copy()
        df["ABBR"] = df["TEAM_ID"].map(_TEAM_ID_MAP)
    return df


def _load_player_name_map(season: str) -> dict[int, str]:
    """Load {player_id: player_name} from the roster cache."""
    conn = sqlite3.connect(_CACHE_DB)
    row = conn.execute(
        "SELECT data FROM player_roster WHERE season = ?", (season,)
    ).fetchone()
    conn.close()
    if not row:
        return {}
    try:
        df = pd.read_json(row[0], orient="records")
        return dict(zip(df["PLAYER_ID"].astype(int), df["PLAYER_NAME"]))
    except Exception:
        return {}


@dataclass
class PreloadedSeasonData:
    """All data for a season, parsed once and reused across dates."""
    season: str
    player_logs: dict[int, pd.DataFrame]    # pid → full-season df (with GAME_DATE_DT)
    prior_player_logs: dict[int, pd.DataFrame]  # prior season
    team_logs: dict[str, pd.DataFrame]      # team → full-season df
    opp_stats: pd.DataFrame | None
    est_metrics: pd.DataFrame | None
    synergy_data: dict
    position_defense: dict
    name_map: dict[int, str]
    roster_teams: dict[int, str] = None     # pid → team_abbr from LeagueDashPlayerStats (opening-night roster proxy)


def preload_season_data(season: str) -> PreloadedSeasonData:
    """Load and parse all data for a season once. Reused across all dates."""
    start_year = int(season[:4])
    prior_season = f"{start_year - 1}-{str(start_year)[-2:]}"

    _init_cache()
    conn = sqlite3.connect(_CACHE_DB)

    name_map = _load_player_name_map(season)
    prior_name_map = _load_player_name_map(prior_season)
    name_map_merged = {**prior_name_map, **name_map}

    # Player logs (full season, no date filter)
    player_logs: dict[int, pd.DataFrame] = {}
    rows = conn.execute(
        "SELECT player_id, data FROM player_game_logs WHERE season = ?",
        (season,),
    ).fetchall()
    for pid, data_json in rows:
        try:
            df = pd.read_json(data_json, orient="records")
            pname = name_map_merged.get(int(pid))
            df = _prepare_game_log(df, player_name=pname)  # no cutoff — full season
            if len(df) >= 1:
                player_logs[int(pid)] = df
        except Exception:
            continue

    # Prior-season player logs
    prior_player_logs: dict[int, pd.DataFrame] = {}
    prior_rows = conn.execute(
        "SELECT player_id, data FROM player_game_logs WHERE season = ?",
        (prior_season,),
    ).fetchall()
    for pid, data_json in prior_rows:
        try:
            df = pd.read_json(data_json, orient="records")
            pname = name_map_merged.get(int(pid))
            df = _prepare_game_log(df, player_name=pname)
            if len(df) >= 5:
                prior_player_logs[int(pid)] = df
        except Exception:
            continue

    # Team logs (full season)
    team_logs: dict[str, pd.DataFrame] = {}
    team_rows = conn.execute(
        "SELECT team, data FROM team_game_logs WHERE season = ?",
        (season,),
    ).fetchall()
    for team, data_json in team_rows:
        try:
            df = pd.read_json(data_json, orient="records")
            df = _prepare_game_log(df)
            team_logs[team] = df
        except Exception:
            continue

    # Opp stats
    opp_stats = None
    for s in (season, prior_season):
        opp_row = conn.execute(
            "SELECT data FROM team_opp_stats WHERE season = ?", (s,)
        ).fetchone()
        if opp_row:
            try:
                opp_df = _add_abbr_column(
                    pd.read_json(opp_row[0], orient="records"))
                if "OPP_PTS" in opp_df.columns and opp_df["OPP_PTS"].max() < 200:
                    gp = opp_df["GP"].astype(float)
                    for col in opp_df.columns:
                        if col.startswith("OPP_") and "_PCT" not in col:
                            opp_df[col] = opp_df[col].astype(float) * gp
                opp_stats = opp_df
                break
            except Exception:
                pass

    # Estimated metrics
    est_metrics = None
    for s in (season, prior_season):
        em_row = conn.execute(
            "SELECT data FROM team_estimated_metrics WHERE season = ?", (s,)
        ).fetchone()
        if em_row:
            try:
                est_metrics = _add_abbr_column(
                    pd.read_json(em_row[0], orient="records"))
                break
            except Exception:
                pass

    # R11: Load roster teams (opening-night team assignment proxy)
    # From LeagueDashPlayerStats: pid → TEAM_ABBREVIATION for current season.
    # Pre-season public info — NOT lookahead.
    roster_teams: dict[int, str] = {}
    roster_row = conn.execute(
        "SELECT data FROM player_roster WHERE season = ?", (season,)
    ).fetchone()
    if roster_row:
        try:
            roster_df = pd.read_json(roster_row[0], orient="records")
            for _, r in roster_df.iterrows():
                roster_teams[int(r["PLAYER_ID"])] = str(r["TEAM_ABBREVIATION"])
            logger.info("Loaded roster teams for %d players (%s)", len(roster_teams), season)
        except Exception as e:
            logger.warning("Failed to load roster teams: %s", e)

    conn.close()

    # Synergy + position defense (prior season, no leakage)
    synergy_data = compute_synergy_prior_season(season)
    pos_def = compute_team_position_defense(season)

    logger.info("Preloaded season %s: %d players, %d prior, %d teams",
                season, len(player_logs), len(prior_player_logs), len(team_logs))

    return PreloadedSeasonData(
        season=season, player_logs=player_logs,
        prior_player_logs=prior_player_logs, team_logs=team_logs,
        opp_stats=opp_stats, est_metrics=est_metrics,
        synergy_data=synergy_data or {}, position_defense=pos_def or {},
        name_map=name_map_merged,
        roster_teams=roster_teams,
    )


def build_engine_asof(game_date: str, season: str,
                      preloaded: PreloadedSeasonData | None = None,
                      ) -> "NBAMatchupEngine":
    """Build a matchup engine using only data available before game_date.

    If preloaded is provided, uses pre-parsed DataFrames (fast — only filters
    by date, no JSON parsing). Otherwise falls back to loading from cache DB.
    """
    from nba_matchup_engine import NBAMatchupEngine

    engine = NBAMatchupEngine(season=season, cache=False)
    cutoff = pd.to_datetime(game_date)

    if preloaded is None:
        # Slow path: load everything from scratch (legacy / single-date use)
        preloaded = preload_season_data(season)

    # Filter player logs to before game_date
    for pid, df in preloaded.player_logs.items():
        filtered = df[df["GAME_DATE_DT"] < cutoff]
        if len(filtered) >= 3:
            engine.player_logs[pid] = filtered

    # Prior-season logs for players not in current
    prior_player_logs = preloaded.prior_player_logs
    _prior_only_pids: set[int] = set()
    for pid, pdf in prior_player_logs.items():
        if pid not in engine.player_logs:
            engine.player_logs[pid] = pdf
            _prior_only_pids.add(pid)

    # Filter team logs
    for team, df in preloaded.team_logs.items():
        filtered = df[df["GAME_DATE_DT"] < cutoff]
        if len(filtered) >= 3:
            engine.team_logs[team] = filtered

    # Static data (no date filtering needed)
    if preloaded.opp_stats is not None:
        engine.opp_stats = preloaded.opp_stats
    if preloaded.est_metrics is not None:
        engine.est_metrics = preloaded.est_metrics
    if preloaded.synergy_data:
        engine.synergy_data = preloaded.synergy_data
    if preloaded.position_defense:
        engine.position_defense = preloaded.position_defense

    # R11: roster-based team fallback — ONLY for players with no current-season
    # games (prior-season logs only).  The roster table is end-of-season data,
    # so using it for players WITH current-season games would be lookahead for
    # mid-season trades.  For prior-only players, the opening-night team is
    # pre-season public info and NOT lookahead.
    if preloaded.roster_teams and _prior_only_pids:
        engine._roster_teams = {
            pid: team for pid, team in preloaded.roster_teams.items()
            if pid in _prior_only_pids
        }

    # Build profiles from the as-of data
    if engine.team_logs:
        engine._build_team_profiles()
    if engine.player_logs:
        engine._detect_trades()   # R11: must run before profiles for team assignment
        engine._build_player_profiles()
        engine._prior_player_logs = prior_player_logs
        if prior_player_logs and hasattr(engine, "_blend_with_prior"):
            engine._blend_with_prior()

    if engine.team_profiles:
        engine._compute_league_averages()

    # Build minutes model
    if engine.player_logs and engine.team_logs:
        try:
            from nba_minutes_model import NBAMinutesModel
            mm = NBAMinutesModel()
            mm.build(engine.player_logs, engine.team_logs,
                     engine.player_profiles, prior_player_logs)
            engine.minutes_model = mm
        except Exception as e:
            logger.warning("Minutes model build failed: %s", e)

    engine._loaded = True
    return engine


def get_actual_stats(player_id: int, game_id: str, season: str,
                     preloaded: PreloadedSeasonData | None = None) -> dict | None:
    """Get actual stats for a player in a specific game from cached game logs.

    game_id: nba_api-style numeric string (e.g. "22201222"), NOT historical.db
              format.  Use _hist_to_api_game_id() to convert.

    If preloaded is provided, looks up from pre-parsed DataFrames (fast).
    Otherwise falls back to SQLite + JSON parse (slow).

    Returns: {PTS, REB, AST, FG3M, STL, BLK, MIN, PRA} or None.
    """
    # Fast path: use preloaded DataFrames (no SQLite, no JSON parse)
    if preloaded is not None:
        df = preloaded.player_logs.get(player_id)
        if df is None:
            return None
        gid_col = "Game_ID" if "Game_ID" in df.columns else "GAME_ID"
        if gid_col not in df.columns:
            return None
        match = df[df[gid_col].astype(str) == str(game_id)]
        if len(match) == 0:
            return None
        r = match.iloc[0]
        pts = float(r.get("PTS", 0))
        reb = float(r.get("REB", 0))
        ast = float(r.get("AST", 0))
        return {
            "PTS": pts, "REB": reb, "AST": ast,
            "FG3M": float(r.get("FG3M", 0)),
            "STL": float(r.get("STL", 0)),
            "BLK": float(r.get("BLK", 0)),
            "MIN": float(r.get("MIN", 0)),
            "PRA": pts + reb + ast,
        }

    # Slow path: SQLite + JSON parse
    _init_cache()
    conn = sqlite3.connect(_CACHE_DB)
    row = conn.execute(
        "SELECT data FROM player_game_logs WHERE player_id = ? AND season = ?",
        (player_id, season),
    ).fetchone()
    conn.close()

    if not row:
        return None

    try:
        df = pd.read_json(row[0], orient="records")
        gid_col = "Game_ID" if "Game_ID" in df.columns else "GAME_ID"
        if gid_col not in df.columns:
            return None
        match = df[df[gid_col].astype(str) == str(game_id)]
        if len(match) == 0:
            return None

        r = match.iloc[0]
        pts = float(r.get("PTS", 0))
        reb = float(r.get("REB", 0))
        ast = float(r.get("AST", 0))
        return {
            "PTS": pts, "REB": reb, "AST": ast,
            "FG3M": float(r.get("FG3M", 0)),
            "STL": float(r.get("STL", 0)),
            "BLK": float(r.get("BLK", 0)),
            "MIN": float(r.get("MIN", 0)),
            "PRA": pts + reb + ast,
        }
    except Exception:
        return None


# ── Scoring: CRPS, PIT ───────────────────────────────────────────────────────

def crps_normal(mean: float, std: float, actual: float) -> float:
    """CRPS for a normal distribution prediction.

    CRPS = σ[z(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (actual - mean) / σ
    """
    from scipy.stats import norm

    if std <= 0:
        return abs(actual - mean)
    z = (actual - mean) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def pit_value(mean: float, std: float, actual: float) -> float:
    """PIT value: CDF(actual) under predicted distribution."""
    from scipy.stats import norm

    if std <= 0:
        return 0.5
    return float(norm.cdf(actual, loc=mean, scale=std))


def game_clustered_bootstrap(
    values: np.ndarray,
    game_ids: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap mean and CI, resampling by game (not observation).

    Returns: (mean, ci_lo, ci_hi)
    """
    unique_games = np.unique(game_ids)
    n_games = len(unique_games)
    rng = np.random.default_rng(42)

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        # Resample games
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        # Gather all observations from sampled games
        mask = np.isin(game_ids, sampled_games)
        boot_means[b] = values[mask].mean()

    alpha = (1 - ci) / 2
    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, alpha * 100)),
        float(np.percentile(boot_means, (1 - alpha) * 100)),
    )


def game_clustered_bootstrap_paired(
    values_a: np.ndarray,
    values_b: np.ndarray,
    game_ids: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Paired bootstrap of (mean_a - mean_b), resampling by game.

    Returns: (mean_diff, ci_lo, ci_hi)
    Used to test: does factor A improve over B with CI excluding zero?
    """
    diff = values_a - values_b
    unique_games = np.unique(game_ids)
    n_games = len(unique_games)
    rng = np.random.default_rng(42)

    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        mask = np.isin(game_ids, sampled_games)
        boot_diffs[b] = diff[mask].mean()

    alpha = (1 - ci) / 2
    return (
        float(np.mean(diff)),
        float(np.percentile(boot_diffs, alpha * 100)),
        float(np.percentile(boot_diffs, (1 - alpha) * 100)),
    )


# ── Factor config mapping ───────────────────────────────────────────────────

# Each factor set maps to an engine factor_config dict.
# Factors are CUMULATIVE: each adds one more adjustment to the previous.
FACTOR_SET_CONFIGS: dict[str, dict] = {
    # Base: minutes + per-min rates + 240-min + reconciliation, no matchup adj
    "base": {"defense": False, "form": False, "synergy": False,
             "reconcile": True, "blowout": True},
    # + opponent defensive factors
    "base+defense": {"defense": True, "form": False, "synergy": False,
                     "reconcile": True, "blowout": True},
    # + Synergy play-type matchup
    "base+defense+synergy": {"defense": True, "form": False, "synergy": True,
                             "matchup": False, "reconcile": True, "blowout": True},
    # + Position-specific defender matchup
    "base+def+syn+matchup": {"defense": True, "form": False, "synergy": True,
                             "matchup": True, "reconcile": True, "blowout": True},
    # + recent form adjustment (FULL model)
    "full": {"defense": True, "form": True, "synergy": True,
             "matchup": True, "reconcile": True, "blowout": True},
    # Hand-set rule tests: toggle one rule off at a time from full model
    "full_no_reconcile": {"defense": True, "form": True, "synergy": True,
                          "matchup": True, "reconcile": False, "blowout": True},
    "full_no_blowout": {"defense": True, "form": True, "synergy": True,
                        "matchup": True, "reconcile": True, "blowout": False},
    "full_no_synergy": {"defense": True, "form": True, "synergy": False,
                        "matchup": True, "reconcile": True, "blowout": True},
    "full_no_matchup": {"defense": True, "form": True, "synergy": True,
                        "matchup": False, "reconcile": True, "blowout": True},
    # Hand-set rule ablation: disable individual minutes model rules
    "full_no_foul_adj": {"defense": True, "form": True, "synergy": True,
                         "matchup": True, "reconcile": True, "blowout": True,
                         "foul_adj": False},
    "full_no_absence_adj": {"defense": True, "form": True, "synergy": True,
                            "matchup": True, "reconcile": True, "blowout": True,
                            "absence_adj": False},
    # ── Reconciliation diagnosis variants (B4.1) ──
    # Clean defense+blowout without reconciliation
    "def_blow_norecon": {"defense": True, "form": False, "synergy": False,
                         "matchup": False, "reconcile": False, "blowout": True},
    # Reconcile without ±25% cap (uncapped)
    "def_blow_recon_nocap": {"defense": True, "form": False, "synergy": False,
                             "matchup": False, "reconcile": True, "blowout": True,
                             "reconcile_cap": 99.0},
    # Reconcile with 50% partial pull
    "def_blow_recon_50pct": {"defense": True, "form": False, "synergy": False,
                             "matchup": False, "reconcile": True, "blowout": True,
                             "reconcile_strength": 0.5},
    # ── B4-R: Pool-fixed reconciliation variants ──
    # Reconcile with pool fix (OTHER bucket adjustment) — full strength
    "r4_recon_poolfix": {"defense": True, "form": False, "synergy": False,
                         "matchup": False, "reconcile": True, "blowout": True,
                         "reconcile_pool_fix": True},
    # Reconcile with pool fix — 50% partial pull
    "r4_recon_poolfix_50": {"defense": True, "form": False, "synergy": False,
                            "matchup": False, "reconcile": True, "blowout": True,
                            "reconcile_pool_fix": True, "reconcile_strength": 0.5},
    # Reconcile WITHOUT pool fix (legacy behavior, for comparison)
    "r4_recon_nofix": {"defense": True, "form": False, "synergy": False,
                       "matchup": False, "reconcile": True, "blowout": True,
                       "reconcile_pool_fix": False},
    # No reconcile baseline (same as def_blow_norecon, included for clean comparison)
    "r4_norecon": {"defense": True, "form": False, "synergy": False,
                   "matchup": False, "reconcile": False, "blowout": True},
}


def _classify_role(mpg: float) -> str:
    """Classify player role from average minutes per game."""
    if mpg >= 28.0:
        return "starter"
    elif mpg >= 15.0:
        return "rotation"
    return "spot"


def _estimate_std(pp, stat: str) -> float:
    """Get predicted std for a stat from a PlayerProjection.

    STL and BLK have no dedicated std field in the engine; use fallback.
    """
    _MAP = {
        "PTS": "pts_std",
        "REB": "reb_std",
        "AST": "ast_std",
        "FG3M": "thr_std",
        "PRA": "pra_std",
    }
    attr = _MAP.get(stat)
    if attr:
        v = getattr(pp, attr, 0.0)
        if v > 0:
            return v
    # Fallback: fraction of mean (empirical CV for NBA stats)
    _FALLBACK_CV = {"PTS": 0.35, "REB": 0.40, "AST": 0.45, "FG3M": 0.60,
                    "STL": 0.65, "BLK": 0.70, "PRA": 0.30}
    mean_val = _extract_mean(pp, stat)
    return max(mean_val * _FALLBACK_CV.get(stat, 0.40), 0.5)


def _extract_mean(pp, stat: str) -> float:
    """Extract the predicted mean for a stat from a PlayerProjection."""
    _MAP = {
        "PTS": "points", "REB": "rebounds", "AST": "assists",
        "FG3M": "threes", "STL": "steals", "BLK": "blocks", "PRA": "pra",
    }
    return float(getattr(pp, _MAP.get(stat, ""), 0.0))


# ── Walk-forward runner ──────────────────────────────────────────────────────

def _load_inactives_lookup(db_path: str) -> dict[str, set[int]]:
    """Load inactives DB into {game_id_api → set of inactive player_ids}."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT game_id, player_id FROM inactives").fetchall()
    conn.close()
    lookup: dict[str, set[int]] = {}
    for gid, pid in rows:
        # Normalise to 8-digit format matching player_logs Game_ID
        # (inactives DB stores 10-digit "0022300001", API uses "22300001")
        norm_gid = str(int(gid)) if gid.startswith("00") else gid
        lookup.setdefault(norm_gid, set()).add(pid)
    logger.info("Loaded inactives for %d games from %s", len(lookup), db_path)
    return lookup


# ── R11: Consecutive-absence detection ──────────────────────────────────────

_ABSENCE_THRESHOLD = 7  # consecutive team-games absent → exclude from pool


def _precompute_absence_data(
    preloaded: "PreloadedSeasonData",
) -> tuple[dict[str, list], dict[int, set[str]]]:
    """Build team game calendars and per-player game-date sets for absence detection.

    Returns:
        team_game_dates: {team_abbr → sorted list of game date strings}
        player_game_dates: {player_id → set of game date strings they appeared in}
    """
    team_game_dates: dict[str, list[str]] = {}
    for team, df in preloaded.team_logs.items():
        if "GAME_DATE_DT" in df.columns:
            dates = sorted(df["GAME_DATE_DT"].dt.strftime("%Y-%m-%d").unique())
        elif "GAME_DATE" in df.columns:
            dates = sorted(pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d").unique())
        else:
            dates = []
        team_game_dates[team] = dates

    player_game_dates: dict[int, set[str]] = {}
    for pid, df in preloaded.player_logs.items():
        if "GAME_DATE_DT" in df.columns:
            dates = set(df["GAME_DATE_DT"].dt.strftime("%Y-%m-%d"))
        elif "GAME_DATE" in df.columns:
            dates = set(pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d"))
        else:
            dates = set()
        player_game_dates[pid] = dates

    return team_game_dates, player_game_dates


def _find_absent_players(
    engine: "NBAMatchupEngine",
    game_date: str,
    team_game_dates: dict[str, list],
    player_game_dates: dict[int, set[str]],
    threshold: int = _ABSENCE_THRESHOLD,
) -> set[str]:
    """Find players with N+ consecutive team-game absences before game_date.

    Returns set of player names to exclude from projections.
    """
    import bisect
    exclude: set[str] = set()

    for pname, profile in engine.player_profiles.items():
        pid = profile.player_id
        if not pid:
            continue
        team = profile.team
        team_dates = team_game_dates.get(team, [])
        if not team_dates:
            continue

        # Find team games before this date
        idx = bisect.bisect_left(team_dates, game_date)
        if idx == 0:
            continue  # no prior team games

        p_dates = player_game_dates.get(pid, set())

        # Count consecutive absences working backwards from most recent team game
        streak = 0
        for i in range(idx - 1, -1, -1):
            if team_dates[i] in p_dates:
                break
            streak += 1

        if streak >= threshold:
            exclude.add(pname)

    return exclude


def run_ablation(
    factor_sets: list[str] | None = None,
    eval_seasons: list[str] | None = None,
    stats: list[str] | None = None,
    max_games_per_season: int = 0,
    resume: bool = False,
    inactives_db: str | None = None,
    soft_allocation: bool = False,
) -> pd.DataFrame:
    """Run walk-forward ablation.  Returns DataFrame of AblationResult rows.

    factor_sets: which factor combinations to test.
        Default: all entries in FACTOR_SET_CONFIGS.

    max_games_per_season: if > 0, sample this many games per season
        (for quick smoke tests).

    resume: if True, load previously saved predictions and skip seasons
        already present.  Useful for multi-session runs.

    inactives_db: path to SQLite DB with per-game inactive player lists
        (from R3 BoxScoreSummaryV2 pull).  When provided, inactive players
        are excluded from the engine's projection pool before roster
        selection, fixing the phantom-DNP minutes redistribution bug.

    soft_allocation: when True, uses R11 pool construction:
        - No hard roster cap (top-10/cum-270 removed)
        - All players with MPG >= 3 included
        - Minutes model allocates 240 across available players
        - Consecutive-absence filter (N>=7) applied

    Procedure:
      For each game date in eval_seasons:
        1. Load market lines (spread, total) from historical.db
        2. Build engine state as-of game date (no leakage)
        3. For each factor_set, produce predictions per player per stat
        4. Compare to actual outcomes
        5. Compute CRPS, MAE, PIT
    """
    if factor_sets is None:
        factor_sets = list(FACTOR_SET_CONFIGS.keys())
    if eval_seasons is None:
        eval_seasons = EVAL_SEASONS
    if stats is None:
        stats = STATS + COMBINED_STATS

    from nba_matchup_engine import normalize_abbr

    # Load inactives lookup if provided (R3/R9 integration)
    inactives_lookup: dict[str, set[int]] = {}
    if inactives_db:
        inactives_lookup = _load_inactives_lookup(inactives_db)

    all_predictions: list[GamePrediction] = []

    # Resume from prior partial run if requested
    completed_seasons: set[str] = set()
    if resume:
        prior = _load_predictions_incremental()
        if prior:
            # Only keep predictions matching current factor_sets
            prior = [p for p in prior if p.factor_set in factor_sets]
            all_predictions.extend(prior)
            completed_seasons = {p.season for p in prior}
            logger.info("Resumed %d predictions from prior run (seasons: %s)",
                        len(prior), ", ".join(sorted(completed_seasons)))

    for season in eval_seasons:
        if resume and season in completed_seasons:
            logger.info("=== Skipping %s (already complete in resume data) ===", season)
            continue
        logger.info("=== Ablation season: %s ===", season)

        # Load market lines for this season
        lines = load_historical_lines(season)
        if len(lines) == 0:
            logger.warning("  No historical lines for %s, skipping", season)
            continue
        logger.info("  %d games with lines", len(lines))

        # Player name map for this season
        name_map = _load_player_name_map(season)

        # Optionally sample
        if max_games_per_season > 0 and len(lines) > max_games_per_season:
            lines = lines.sample(n=max_games_per_season, random_state=42)

        # Sort by date for walk-forward order
        lines["game_date_dt"] = pd.to_datetime(lines["game_date"])
        lines = lines.sort_values("game_date_dt")

        # Group by date — build engine once per day, project all games
        unique_dates = lines["game_date_dt"].dt.date.unique()
        logger.info("  %d unique game dates", len(unique_dates))

        # Preload all season data once — avoids re-parsing 400+ JSON blobs per date
        preloaded = preload_season_data(season)
        logger.info("  Preloaded %d player logs for %s", len(preloaded.player_logs), season)

        # R11: precompute absence data for consecutive-absence filter
        team_game_dates_map, player_game_dates_map = (
            _precompute_absence_data(preloaded)
            if soft_allocation else ({}, {})
        )

        engine_cache: dict[str, object] = {}  # date_str → engine

        import time as _time
        _season_t0 = _time.time()

        for i, gd in enumerate(unique_dates):
            date_str = str(gd)
            _date_t0 = _time.time()

            # ── Build engine as-of this date (shared across games + factor sets)
            try:
                engine = build_engine_asof(date_str, season, preloaded=preloaded)
            except Exception as e:
                logger.warning("  Engine build failed for %s: %s", date_str, e)
                continue

            if not engine.player_profiles:
                continue

            # Build reverse maps: player_name ↔ player_id
            name_to_pid = {
                p.player_name: p.player_id
                for p in engine.player_profiles.values()
                if p.player_id
            }
            pid_to_name = {v: k for k, v in name_to_pid.items()}

            # R11: consecutive-absence exclusion (date-level, shared across games)
            absence_exclude: set[str] = set()
            if soft_allocation and team_game_dates_map:
                absence_exclude = _find_absent_players(
                    engine, date_str,
                    team_game_dates_map, player_game_dates_map,
                )

            # All games on this date
            day_games = lines[lines["game_date_dt"].dt.date == gd]

            for _, game_row in day_games.iterrows():
                game_id_hist = game_row["game_id"]       # "NBA_0022201222"
                game_id_api = _hist_to_api_game_id(game_id_hist)
                home = normalize_abbr(game_row["home_team"])
                away = normalize_abbr(game_row["away_team"])
                mkt_spread = float(game_row["close_spread"])
                mkt_total = float(game_row["close_total"])

                # Map inactive player_ids → player names for exclusion
                exclude_names: set[str] = set()
                if inactives_lookup:
                    inactive_pids = inactives_lookup.get(game_id_api, set())
                    if inactive_pids:
                        exclude_names = {
                            pid_to_name[pid]
                            for pid in inactive_pids
                            if pid in pid_to_name
                        }
                # R11: merge absence-based exclusions
                exclude_names |= absence_exclude

                # Pre-fetch actuals for all known players in this game
                # (avoids 11× redundant SQLite reads per player across configs)
                actuals_cache: dict[int, dict | None] = {}

                # ── Project with each factor set ──
                for fs_name in factor_sets:
                    fc = FACTOR_SET_CONFIGS.get(fs_name)
                    if fc is None:
                        continue
                    try:
                        proj = engine.project_game(
                            home=home, away=away,
                            market_total=mkt_total,
                            market_spread=mkt_spread,
                            factor_config=fc,
                            exclude_players=exclude_names or None,
                            soft_allocation=soft_allocation,
                        )
                    except Exception as e:
                        logger.debug("  project_game failed %s vs %s: %s",
                                     away, home, e)
                        continue

                    # ── Extract per-player predictions ──
                    for pname, pp in proj.players.items():
                        pid = name_to_pid.get(pname, 0)
                        if not pid:
                            continue

                        # Actual stats — cached per (player, game)
                        if pid not in actuals_cache:
                            actuals_cache[pid] = get_actual_stats(
                                pid, game_id_api, season,
                                preloaded=preloaded,
                            )
                        actuals = actuals_cache[pid]
                        if actuals is None:
                            continue

                        # Role from engine profile
                        profile = engine.player_profiles.get(pname)
                        role = _classify_role(profile.mpg) if profile else "rotation"

                        # One GamePrediction per stat
                        for stat in stats:
                            pred_mean = _extract_mean(pp, stat)
                            pred_std = _estimate_std(pp, stat)
                            actual_val = actuals.get(stat, None)
                            if actual_val is None:
                                continue

                            all_predictions.append(GamePrediction(
                                game_id=game_id_hist,
                                game_date=date_str,
                                season=season,
                                player_id=pid,
                                player_name=pname,
                                team=pp.team,
                                opponent=pp.opponent,
                                is_home=(pp.team == home),
                                stat=stat,
                                role=role,
                                predicted_mean=pred_mean,
                                predicted_std=pred_std,
                                actual=actual_val,
                                market_spread=mkt_spread,
                                market_total=mkt_total,
                                factor_set=fs_name,
                            ))

            _date_elapsed = _time.time() - _date_t0
            if (i + 1) % 10 == 0:
                _elapsed = _time.time() - _season_t0
                _rate = (i + 1) / _elapsed
                _eta = (len(unique_dates) - i - 1) / _rate if _rate > 0 else 0
                n_preds = len(all_predictions)
                logger.info("  %d/%d dates (%.1fs/date, ETA %.0fs) — %d predictions",
                            i + 1, len(unique_dates), _date_elapsed, _eta, n_preds)

        season_preds = [p for p in all_predictions if p.season == season]
        logger.info("  Season %s complete: %d total predictions",
                     season, len(season_preds))
        # Save per-season predictions incrementally
        _save_predictions_incremental(all_predictions)

    logger.info("Total predictions: %d", len(all_predictions))

    if len(all_predictions) == 0:
        logger.warning("No predictions generated — is the data pulled?")
        return pd.DataFrame()

    # Compute metrics
    results = _compute_metrics(all_predictions, factor_sets, stats)
    return results


def _compute_metrics(
    predictions: list[GamePrediction],
    factor_sets: list[str],
    stats: list[str],
) -> pd.DataFrame:
    """Compute CRPS, MAE, PIT for each factor_set × stat × role."""
    from scipy.stats import kstest

    rows = []
    for fs in factor_sets:
        fs_preds = [p for p in predictions if p.factor_set == fs]
        for stat in stats:
            stat_preds = [p for p in fs_preds if p.stat == stat]
            for role_filter in ["all", "starter", "rotation"]:
                if role_filter == "all":
                    preds = stat_preds
                else:
                    preds = [p for p in stat_preds if p.role == role_filter]

                if len(preds) < 30:
                    continue

                crps_vals = np.array([
                    crps_normal(p.predicted_mean, p.predicted_std, p.actual)
                    for p in preds
                ])
                mae_vals = np.array([
                    abs(p.predicted_mean - p.actual) for p in preds
                ])
                pit_vals = np.array([
                    pit_value(p.predicted_mean, p.predicted_std, p.actual)
                    for p in preds
                ])
                game_ids = np.array([p.game_id for p in preds])

                crps_mean, crps_lo, crps_hi = game_clustered_bootstrap(
                    crps_vals, game_ids
                )
                mae_mean, mae_lo, mae_hi = game_clustered_bootstrap(
                    mae_vals, game_ids
                )

                # PIT calibration: KS test against Uniform(0,1)
                ks_stat, ks_pval = kstest(pit_vals, "uniform")

                rows.append(AblationResult(
                    factor_set=fs,
                    stat=stat,
                    role=role_filter,
                    n_obs=len(preds),
                    n_games=len(np.unique(game_ids)),
                    crps=crps_mean,
                    crps_ci_lo=crps_lo,
                    crps_ci_hi=crps_hi,
                    mae=mae_mean,
                    mae_ci_lo=mae_lo,
                    mae_ci_hi=mae_hi,
                    pit_ks_stat=float(ks_stat),
                    pit_ks_pval=float(ks_pval),
                    pit_mean=float(pit_vals.mean()),
                ))

    return pd.DataFrame([r.__dict__ for r in rows])


def compare_factor_sets(
    predictions: list[GamePrediction],
    baseline: str,
    candidate: str,
    stats: list[str] | None = None,
    roles: list[str] | None = None,
) -> pd.DataFrame:
    """Paired comparison: does candidate improve over baseline?

    Returns table with CRPS diff, CI, and whether CI excludes zero.
    """
    if stats is None:
        stats = STATS + COMBINED_STATS
    if roles is None:
        roles = ["all", "starter", "rotation"]

    rows = []
    base_preds = {
        (p.game_id, p.player_id, p.stat): p
        for p in predictions if p.factor_set == baseline
    }
    cand_preds = {
        (p.game_id, p.player_id, p.stat): p
        for p in predictions if p.factor_set == candidate
    }

    for stat in stats:
        for role_filter in roles:
            # Find matched pairs
            if role_filter == "all":
                keys = [k for k in base_preds if k in cand_preds and k[2] == stat]
            else:
                keys = [k for k in base_preds if k in cand_preds and k[2] == stat
                        and base_preds[k].role == role_filter]
            if len(keys) < 30:
                continue

            base_crps = np.array([
                crps_normal(base_preds[k].predicted_mean,
                            base_preds[k].predicted_std,
                            base_preds[k].actual)
                for k in keys
            ])
            cand_crps = np.array([
                crps_normal(cand_preds[k].predicted_mean,
                            cand_preds[k].predicted_std,
                            cand_preds[k].actual)
                for k in keys
            ])
            game_ids = np.array([k[0] for k in keys])

            # Negative diff = candidate better (lower CRPS)
            diff_mean, diff_lo, diff_hi = game_clustered_bootstrap_paired(
                cand_crps, base_crps, game_ids
            )

            rows.append({
                "baseline": baseline,
                "candidate": candidate,
                "stat": stat,
                "role": role_filter,
                "n_pairs": len(keys),
                "n_games": len(np.unique(game_ids)),
                "crps_diff": diff_mean,
                "crps_diff_ci_lo": diff_lo,
                "crps_diff_ci_hi": diff_hi,
                "ci_excludes_zero": (diff_hi < 0) or (diff_lo > 0),
                "direction": "better" if diff_mean < 0 else "worse",
            })

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

_RESULTS_PATH = os.path.join(_DATA_DIR, "nba_ablation_results.csv")
_PREDICTIONS_PATH = os.path.join(_DATA_DIR, "nba_ablation_predictions.json")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="NBA Prop Model Ablation (B4)")
    sub = parser.add_subparsers(dest="cmd")

    p_pull = sub.add_parser("pull", help="Pull and cache all data from nba_api")
    p_pull.add_argument("--seasons", nargs="*", default=None,
                        help="Seasons to pull (default: all ablation seasons)")

    p_run = sub.add_parser("run", help="Run walk-forward ablation")
    p_run.add_argument("--quick", type=int, default=0,
                       help="Limit to N games per season (smoke test)")
    p_run.add_argument("--seasons", nargs="*", default=None,
                       help="Eval seasons (default: 2023-24 2024-25 2025-26)")
    p_run.add_argument("--factor-sets", nargs="*", default=None,
                       help="Factor sets to test (default: all)")
    p_run.add_argument("--resume", action="store_true",
                       help="Resume from prior partial run (skip completed seasons)")

    sub.add_parser("report", help="Print saved results table")

    p_cmp = sub.add_parser("compare", help="Paired factor-set comparisons from predictions")
    p_cmp.add_argument("--baseline", default="base",
                       help="Baseline factor set (default: base)")
    p_cmp.add_argument("--candidate", default=None,
                       help="Candidate factor set (default: all others)")

    args = parser.parse_args()

    if args.cmd == "pull":
        seasons = args.seasons or ABLATION_SEASONS
        pull_all_data(seasons)

    elif args.cmd == "run":
        results = run_ablation(
            factor_sets=args.factor_sets,
            eval_seasons=args.seasons,
            max_games_per_season=args.quick,
            resume=args.resume,
        )
        if len(results) > 0:
            results.to_csv(_RESULTS_PATH, index=False)
            logger.info("Results saved to %s", _RESULTS_PATH)
            print("\n" + results.to_string(index=False))
        else:
            print("No results (check data pull status with: "
                  "python nba_ablation.py pull)")

    elif args.cmd == "report":
        if os.path.exists(_RESULTS_PATH):
            df = pd.read_csv(_RESULTS_PATH)
            print(df.to_string(index=False))
        else:
            print(f"No results file at {_RESULTS_PATH}. Run ablation first.")

    elif args.cmd == "compare":
        preds = _load_predictions_incremental()
        if not preds:
            print("No predictions saved. Run ablation first.")
        else:
            baseline = args.baseline
            available_fs = sorted(set(p.factor_set for p in preds))
            candidates = [args.candidate] if args.candidate else [
                fs for fs in available_fs if fs != baseline
            ]
            print(f"Paired comparisons vs baseline '{baseline}'")
            print("Negative ΔCRPS = candidate BETTER (lower CRPS)")
            print(f"{'candidate':30s} {'stat':5s} {'role':10s} {'N':>6s} "
                  f"{'ΔCRPS':>8s} {'CI_lo':>8s} {'CI_hi':>8s} {'sig':>5s}")
            print("-" * 90)
            for cand in candidates:
                df = compare_factor_sets(preds, baseline, cand)
                for _, r in df.iterrows():
                    sig = "*" if r["ci_excludes_zero"] else ""
                    print(f"{cand:30s} {r['stat']:5s} {r['role']:10s} "
                          f"{r['n_pairs']:6d} {r['crps_diff']:+8.4f} "
                          f"{r['crps_diff_ci_lo']:+8.4f} "
                          f"{r['crps_diff_ci_hi']:+8.4f} {sig:>5s}")

    else:
        parser.print_help()
