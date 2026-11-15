"""
nfl_matchup_context.py — Per-game matchup context module for NFL player props.

Computes and persists:
  1. Team defense/offense verified stats (EPA, blitz, pressure, RZ)
  2. QB-vs-blitz splits (targets, aDOT, comp%, EPA under pressure)
  3. Receiver separation estimates (NGS avg_separation + MIN defensive effect)
  4. CB-on-WR proxy from PBP credited defenders (Bayesian, side-overlap prior)
  5. Script/volume (pass rate, PROE, attempt distributions)

DISPLAY ONLY — does NOT modify the v1.2 model or prereg pool.
Candidate features are logged with predicted directions for later ablation.

Data sources (all public / open):
  - nflverse PBP (nfl_data_py): play-level EPA, defenders credited, pass location
  - nflverse FTN charting: n_blitzers, n_pass_rushers, play-action, screen, etc.
  - nflverse NGS receiving: avg_separation, avg_cushion per receiver per week
  - nflverse rosters: gsis_id, position, team assignment
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")
_NFL_DB = os.path.join(_DATA_DIR, "nfl.db")
_PBP_CACHE = os.path.join(_DATA_DIR, "pbp_cache")

_LOW_CONFIDENCE_N = 100  # splits with N < this are labeled low-confidence


def _ci_rate(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion p with sample size n."""
    if n == 0 or p is None:
        return (None, None)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _ci_mean(values: pd.Series, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a mean from a Series (mean ± z * SE)."""
    n = values.notna().sum()
    if n < 2:
        return (None, None)
    m = values.mean()
    se = values.std() / np.sqrt(n)
    return (m - z * se, m + z * se)


# ══════════════════════════════════════════════════════════════════════════════
# Storage DDL
# ══════════════════════════════════════════════════════════════════════════════

_DDL_TEAM_CONTEXT = """
CREATE TABLE IF NOT EXISTS nfl_matchup_team_context (
    game_id            TEXT    NOT NULL,
    team               TEXT    NOT NULL,
    season             INTEGER NOT NULL,
    week               INTEGER NOT NULL,
    side               TEXT    NOT NULL,  -- 'offense' or 'defense'
    -- EPA / success
    pass_epa_play      REAL,
    rush_epa_play      REAL,
    pass_success_rate  REAL,
    rush_success_rate  REAL,
    -- Pressure / blitz
    blitz_rate         REAL,   -- FTN n_blitzers >= 1
    heavy_rush_rate    REAL,   -- n_pass_rushers >= 5
    sack_rate          REAL,
    pressure_rate      REAL,
    -- Red zone (per trip, not per play)
    rz_td_rate         REAL,
    rz_trips           INTEGER,
    -- Blitz / pressure metadata
    blitz_n_charted    INTEGER,
    pressure_source    TEXT,    -- 'was_pressure' or 'sack_hit_proxy'
    -- Ranks (1 = best in league for the stat direction that helps)
    pass_epa_rank      INTEGER,
    rush_epa_rank      INTEGER,
    blitz_rate_rank    INTEGER,
    sack_rate_rank     INTEGER,
    pressure_rank      INTEGER,
    rz_td_rank         INTEGER,
    -- Opponent adjustment
    opp_adj_pass_epa   REAL,
    opp_adj_rush_epa   REAL,
    -- Meta
    n_games            INTEGER,
    data_available     TEXT,   -- CSV of available sources: 'pbp,ftn,ngs'
    created_at         TEXT    NOT NULL,
    PRIMARY KEY (game_id, team, side)
)
"""

_DDL_PLAYER_CONTEXT = """
CREATE TABLE IF NOT EXISTS nfl_matchup_player_context (
    game_id             TEXT    NOT NULL,
    player_gsis_id      TEXT,
    player_name         TEXT    NOT NULL,
    team                TEXT    NOT NULL,
    position            TEXT,
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    -- QB vs blitz (QB rows only)
    qb_blitz_epa        REAL,
    qb_clean_epa        REAL,
    qb_blitz_comp_pct   REAL,
    qb_clean_comp_pct   REAL,
    qb_blitz_adot       REAL,
    qb_clean_adot       REAL,
    qb_blitz_sack_rate  REAL,
    qb_screen_rate_blitz REAL,
    qb_pa_rate_blitz    REAL,
    qb_throwaway_rate_blitz REAL,
    -- Target share under blitz (WR/TE/RB rows)
    tgt_share_blitz     REAL,
    tgt_share_clean     REAL,
    tgt_shift_blitz     REAL,   -- blitz - clean
    adot_blitz          REAL,
    adot_clean          REAL,
    -- Separation (WR/TE rows)
    ngs_avg_separation  REAL,
    ngs_avg_cushion     REAL,
    ngs_season          INTEGER,  -- which season the NGS data is from
    expected_separation REAL,     -- baseline + opp def effect, shrunk
    expected_sep_ci_lo  REAL,
    expected_sep_ci_hi  REAL,
    -- CB-on-WR proxy (WR rows)
    primary_cb_name     TEXT,
    primary_cb_gsis_id  TEXT,
    primary_cb_prob     REAL,     -- P(primary CB covers this WR)
    cb_evidence_n       INTEGER,  -- number of direct evidence plays
    cb_side_overlap     REAL,     -- side-overlap prior probability
    shadow_risk_flag    INTEGER DEFAULT 0,
    -- Script / volume
    expected_targets    REAL,
    expected_rush_att   REAL,
    -- Implied direction vs market
    implied_direction   TEXT,     -- 'over' / 'under' / null
    -- Meta
    data_available      TEXT,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (game_id, player_name, team)
)
"""

_DDL_CB_EVIDENCE = """
CREATE TABLE IF NOT EXISTS nfl_cb_wr_evidence (
    game_id             TEXT    NOT NULL,
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    receiver_gsis_id    TEXT,
    receiver_name       TEXT    NOT NULL,
    receiver_team       TEXT    NOT NULL,
    defender_gsis_id    TEXT,
    defender_name       TEXT,
    defender_team       TEXT    NOT NULL,
    evidence_type       TEXT    NOT NULL,  -- 'pbu', 'int', 'tackle_short_yac', 'tackle_long_yac'
    evidence_weight     REAL    NOT NULL,  -- 1.0 strong, 0.5 moderate, 0.2 weak
    pass_location       TEXT,              -- left/middle/right
    air_yards           REAL,
    yards_after_catch   REAL,
    play_id             TEXT,
    created_at          TEXT    NOT NULL,
    PRIMARY KEY (game_id, play_id, defender_name, receiver_name)
)
"""

_DDL_CANDIDATE_FEATURES = """
CREATE TABLE IF NOT EXISTS nfl_candidate_features (
    feature_name        TEXT    NOT NULL,
    hypothesis          TEXT    NOT NULL,
    predicted_direction TEXT    NOT NULL,  -- '+' or '-' relative to prop outcome
    data_source         TEXT    NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'candidate',
    ablation_result     TEXT,
    created_at          TEXT    NOT NULL,
    PRIMARY KEY (feature_name)
)
"""

_DDL_CB_SIDE_TENDENCY = """
CREATE TABLE IF NOT EXISTS nfl_cb_side_tendency (
    player_gsis_id  TEXT    NOT NULL,
    player_name     TEXT    NOT NULL,
    team            TEXT    NOT NULL,
    season          INTEGER NOT NULL,
    n_targets_near  INTEGER NOT NULL,
    left_share      REAL,
    right_share     REAL,
    slot_share      REAL,
    created_at      TEXT    NOT NULL,
    PRIMARY KEY (player_gsis_id, season)
)
"""

_DDL_WR_SIDE_TENDENCY = """
CREATE TABLE IF NOT EXISTS nfl_wr_side_tendency (
    player_gsis_id  TEXT    NOT NULL,
    player_name     TEXT    NOT NULL,
    team            TEXT    NOT NULL,
    season          INTEGER NOT NULL,
    n_targets       INTEGER NOT NULL,
    left_share      REAL,
    right_share     REAL,
    slot_share      REAL,
    created_at      TEXT    NOT NULL,
    PRIMARY KEY (player_gsis_id, season)
)
"""


_DDL_ELITE_CB_LIST = """
CREATE TABLE IF NOT EXISTS nfl_elite_cb_list (
    cb_name         TEXT    NOT NULL,
    team_2023       TEXT    NOT NULL,       -- team during BDB 2023 season
    epa_per_target  REAL    NOT NULL,       -- EPA/target allowed (lower = better)
    comp_pct        REAL,
    yds_per_target  REAL,
    coverage_snaps  INTEGER NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'BDB_2023',
    frozen_at       TEXT    NOT NULL,       -- timestamp list was frozen
    PRIMARY KEY (cb_name, team_2023)
)
"""

_DDL_ELITE_CB_SHADOW_RESULTS = """
CREATE TABLE IF NOT EXISTS nfl_elite_cb_shadow_results (
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    game_id         TEXT,
    def_team        TEXT    NOT NULL,
    cb_name         TEXT    NOT NULL,
    wr_name         TEXT    NOT NULL,
    off_team        TEXT    NOT NULL,
    wr_rec_yds      REAL,                  -- actual WR1 receiving yards
    market_line     REAL,                  -- devigged market rec_yds line
    v12_projection  REAL,                  -- v1.2 model projection
    wr_targets      INTEGER,
    wr_yds_per_tgt  REAL,
    shadow_confirmed TEXT,                 -- 'confirmed' | 'not_confirmed' | 'reported'
    evaluated_at    TEXT    NOT NULL,
    PRIMARY KEY (season, week, def_team, cb_name, wr_name)
)
"""

_DDL_SHADOW_LOG = """
CREATE TABLE IF NOT EXISTS nfl_shadow_log (
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    game_id         TEXT,
    def_team        TEXT    NOT NULL,
    cb_name         TEXT    NOT NULL,
    wr_name         TEXT    NOT NULL,
    off_team        TEXT    NOT NULL,
    source          TEXT    NOT NULL,   -- e.g. 'PFF', 'RotoBaller', 'ESPN'
    source_date     TEXT,               -- date of the report
    confidence      TEXT    DEFAULT 'reported',  -- 'reported' | 'confirmed'
    notes           TEXT,
    entered_at      TEXT    NOT NULL,
    PRIMARY KEY (season, week, def_team, cb_name, wr_name)
)
"""


def _ensure_tables():
    """Create matchup context tables if not present."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    for ddl in [_DDL_TEAM_CONTEXT, _DDL_PLAYER_CONTEXT, _DDL_CB_EVIDENCE,
                _DDL_CANDIDATE_FEATURES, _DDL_CB_SIDE_TENDENCY,
                _DDL_WR_SIDE_TENDENCY, _DDL_SHADOW_LOG,
                _DDL_ELITE_CB_LIST, _DDL_ELITE_CB_SHADOW_RESULTS]:
        conn.execute(ddl)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    """Load nflverse PBP for given seasons via nfl_data_py."""
    import nfl_data_py as nfl
    logger.info("[matchup] Loading PBP for seasons %s...", seasons)
    df = nfl.import_pbp_data(seasons, downcast=False)
    logger.info("[matchup] PBP: %d plays loaded", len(df))
    return df


def _load_ftn(seasons: list[int]) -> pd.DataFrame:
    """Load FTN charting data (2022+)."""
    import nfl_data_py as nfl
    valid = [s for s in seasons if s >= 2022]
    if not valid:
        return pd.DataFrame()
    logger.info("[matchup] Loading FTN for seasons %s...", valid)
    ftn = nfl.import_ftn_data(valid)
    logger.info("[matchup] FTN: %d plays loaded", len(ftn))
    return ftn


def _load_ngs_receiving(seasons: list[int]) -> pd.DataFrame:
    """Load NGS receiving data (avg_separation, avg_cushion)."""
    import nfl_data_py as nfl
    logger.info("[matchup] Loading NGS receiving for seasons %s...", seasons)
    ngs = nfl.import_ngs_data("receiving", seasons)
    logger.info("[matchup] NGS receiving: %d rows", len(ngs))
    return ngs


def _load_rosters(seasons: list[int]) -> pd.DataFrame:
    """Load nflverse rosters for player_id joins.

    PBP uses player_id format '00-XXXXXXX' in receiver_player_id,
    passer_player_id, etc.  Seasonal rosters have the same field as
    'player_id'.  We rename it to 'gsis_id' for consistency with the
    rest of this module.
    """
    import nfl_data_py as nfl
    logger.info("[matchup] Loading rosters for seasons %s...", seasons)
    rosters = nfl.import_seasonal_rosters(seasons)
    if "player_id" in rosters.columns:
        rosters = rosters.rename(columns={"player_id": "gsis_id"})
    logger.info("[matchup] Rosters: %d rows", len(rosters))
    return rosters


def _merge_ftn(pbp: pd.DataFrame, ftn: pd.DataFrame) -> pd.DataFrame:
    """Merge FTN charting onto PBP by nflverse_game_id + nflverse_play_id."""
    if ftn.empty:
        return pbp
    ftn_cols = ["nflverse_game_id", "nflverse_play_id",
                "n_blitzers", "n_pass_rushers", "is_play_action",
                "is_screen_pass", "is_throw_away", "is_qb_out_of_pocket",
                "is_catchable_ball", "is_contested_ball"]
    ftn_cols = [c for c in ftn_cols if c in ftn.columns]
    ftn_sub = ftn[ftn_cols].drop_duplicates(
        subset=["nflverse_game_id", "nflverse_play_id"])
    merged = pbp.merge(
        ftn_sub,
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "nflverse_play_id"],
        how="left", suffixes=("", "_ftn"))
    logger.info("[matchup] FTN merge: %d/%d plays matched",
                merged["n_blitzers"].notna().sum(), len(merged))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Team Defense / Offense Verified Stats
# ══════════════════════════════════════════════════════════════════════════════

def compute_team_context(pbp: pd.DataFrame, ftn: pd.DataFrame,
                         season: int, through_week: int,
                         team: str) -> dict:
    """
    Compute team defense and offense stats with league ranks.

    Returns dict with 'offense' and 'defense' sub-dicts, each containing
    EPA/play, success rate, blitz rate, sack rate, pressure rate, RZ TD%.
    All stats are season-to-date through `through_week` (no lookahead).
    """
    # Filter to season, weeks before the target game
    mask = (pbp["season"] == season) & (pbp["week"] <= through_week)
    real = pbp[mask & pbp["play_type"].isin(["pass", "run"])].copy()
    passes = real[real["play_type"] == "pass"].copy()

    if real.empty:
        return {}

    # Merge FTN for blitz data (skip if already merged globally)
    if not ftn.empty and "n_blitzers" not in real.columns:
        ftn_season = ftn[(ftn["season"] == season) & (ftn["week"] <= through_week)]
        real = _merge_ftn(real, ftn_season)
        passes = real[real["play_type"] == "pass"]

    # --- Compute per-team season aggregates ---
    all_teams = real["posteam"].unique()
    league = {}
    drive_col = "fixed_drive" if "fixed_drive" in real.columns else "drive"

    for t in all_teams:
        off = real[real["posteam"] == t]
        off_pass = passes[passes["posteam"] == t]
        defe = real[real["defteam"] == t]
        def_pass = passes[passes["defteam"] == t]

        # Offense
        o_pass_epa = off_pass["epa"].mean() if len(off_pass) > 0 else None
        o_rush_epa = off[off["play_type"] == "run"]["epa"].mean() if len(off[off["play_type"] == "run"]) > 0 else None
        o_pass_success = (off_pass["epa"] > 0).mean() if len(off_pass) > 0 else None
        o_rush_success = (off[off["play_type"] == "run"]["epa"] > 0).mean() if len(off[off["play_type"] == "run"]) > 0 else None

        # Pressure allowed
        dropbacks = off_pass[off_pass["qb_dropback"] == 1] if "qb_dropback" in off_pass.columns else off_pass
        o_sack_rate = dropbacks["sack"].mean() if len(dropbacks) > 0 else None
        o_pressure = None
        o_pressure_source = None
        if "was_pressure" in dropbacks.columns and len(dropbacks) > 0 and dropbacks["was_pressure"].notna().any():
            o_pressure = dropbacks["was_pressure"].mean()
            o_pressure_source = "was_pressure"
        elif len(dropbacks) > 0:
            o_pressure = (dropbacks["sack"].fillna(0) + dropbacks["qb_hit"].fillna(0)).clip(upper=1).mean()
            o_pressure_source = "sack_hit_proxy"

        # RZ offense (per trip: TDs / drives entering RZ)
        rz_plays_off = off[off["yardline_100"] <= 20]
        if len(rz_plays_off) > 0:
            o_rz_trips = rz_plays_off.groupby(["game_id", drive_col]).ngroups
            o_rz_tds = (rz_plays_off[rz_plays_off["touchdown"] == 1]
                        .groupby(["game_id", drive_col]).ngroups
                        if (rz_plays_off["touchdown"] == 1).any() else 0)
            o_rz_td = o_rz_tds / o_rz_trips if o_rz_trips > 0 else None
        else:
            o_rz_trips = 0
            o_rz_td = None

        # Defense
        d_pass_epa = def_pass["epa"].mean() if len(def_pass) > 0 else None
        d_rush_epa = defe[defe["play_type"] == "run"]["epa"].mean() if len(defe[defe["play_type"] == "run"]) > 0 else None
        d_pass_success = (def_pass["epa"] > 0).mean() if len(def_pass) > 0 else None
        d_rush_success = (defe[defe["play_type"] == "run"]["epa"] > 0).mean() if len(defe[defe["play_type"] == "run"]) > 0 else None

        # Blitz rate (FTN-based) with denominators
        d_blitz_rate = None
        d_heavy_rush = None
        d_blitz_n_charted = 0
        if "n_blitzers" in def_pass.columns:
            blitz_valid = def_pass["n_blitzers"].notna()
            d_blitz_n_charted = int(blitz_valid.sum())
            if d_blitz_n_charted > 0:
                d_blitz_rate = (def_pass.loc[blitz_valid, "n_blitzers"] >= 1).mean()
        if "n_pass_rushers" in def_pass.columns:
            rush_valid = def_pass["n_pass_rushers"].notna()
            if rush_valid.sum() > 0:
                d_heavy_rush = (def_pass.loc[rush_valid, "n_pass_rushers"] >= 5).mean()
        # Fallback to PBP number_of_pass_rushers
        if d_heavy_rush is None and "number_of_pass_rushers" in def_pass.columns:
            pr_valid = def_pass["number_of_pass_rushers"].notna()
            if pr_valid.sum() > 0:
                d_heavy_rush = (def_pass.loc[pr_valid, "number_of_pass_rushers"] >= 5).mean()

        # Sack / pressure generated
        def_dropbacks = def_pass
        d_sack_rate = def_dropbacks["sack"].mean() if len(def_dropbacks) > 0 else None
        d_pressure = None
        d_pressure_source = None
        if "was_pressure" in def_dropbacks.columns and len(def_dropbacks) > 0 and def_dropbacks["was_pressure"].notna().any():
            d_pressure = def_dropbacks["was_pressure"].mean()
            d_pressure_source = "was_pressure"
        elif len(def_dropbacks) > 0:
            d_pressure = (def_dropbacks["sack"].fillna(0) + def_dropbacks["qb_hit"].fillna(0)).clip(upper=1).mean()
            d_pressure_source = "sack_hit_proxy"

        # RZ defense (per trip)
        rz_plays_def = defe[defe["yardline_100"] <= 20]
        if len(rz_plays_def) > 0:
            d_rz_trips = rz_plays_def.groupby(["game_id", drive_col]).ngroups
            d_rz_tds = (rz_plays_def[rz_plays_def["touchdown"] == 1]
                        .groupby(["game_id", drive_col]).ngroups
                        if (rz_plays_def["touchdown"] == 1).any() else 0)
            d_rz_td = d_rz_tds / d_rz_trips if d_rz_trips > 0 else None
        else:
            d_rz_trips = 0
            d_rz_td = None

        n_off_games = off["game_id"].nunique()
        n_def_games = defe["game_id"].nunique()

        # Play counts (N) for CI computation
        n_off_pass = len(off_pass)
        n_off_rush = len(off[off["play_type"] == "run"])
        n_off_dropbacks = len(dropbacks)
        n_def_pass = len(def_pass)
        n_def_rush = len(defe[defe["play_type"] == "run"])
        n_def_dropbacks = len(def_dropbacks)

        league[t] = {
            "offense": {
                "pass_epa_play": o_pass_epa,
                "pass_epa_ci": _ci_mean(off_pass["epa"]) if n_off_pass > 1 else (None, None),
                "n_pass_plays": n_off_pass,
                "rush_epa_play": o_rush_epa,
                "rush_epa_ci": _ci_mean(off[off["play_type"] == "run"]["epa"]) if n_off_rush > 1 else (None, None),
                "n_rush_plays": n_off_rush,
                "pass_success_rate": o_pass_success,
                "pass_success_ci": _ci_rate(o_pass_success, n_off_pass),
                "rush_success_rate": o_rush_success,
                "rush_success_ci": _ci_rate(o_rush_success, n_off_rush),
                "sack_rate": o_sack_rate,
                "sack_rate_ci": _ci_rate(o_sack_rate, n_off_dropbacks),
                "pressure_rate": o_pressure,
                "pressure_ci": _ci_rate(o_pressure, n_off_dropbacks),
                "pressure_source": o_pressure_source,
                "n_dropbacks": n_off_dropbacks,
                "rz_td_rate": o_rz_td,
                "rz_td_ci": _ci_rate(o_rz_td, o_rz_trips),
                "rz_trips": o_rz_trips,
                "n_games": n_off_games,
            },
            "defense": {
                "pass_epa_play": d_pass_epa,
                "pass_epa_ci": _ci_mean(def_pass["epa"]) if n_def_pass > 1 else (None, None),
                "n_pass_plays": n_def_pass,
                "rush_epa_play": d_rush_epa,
                "rush_epa_ci": _ci_mean(defe[defe["play_type"] == "run"]["epa"]) if n_def_rush > 1 else (None, None),
                "n_rush_plays": n_def_rush,
                "pass_success_rate": d_pass_success,
                "pass_success_ci": _ci_rate(d_pass_success, n_def_pass),
                "rush_success_rate": d_rush_success,
                "rush_success_ci": _ci_rate(d_rush_success, n_def_rush),
                "blitz_rate": d_blitz_rate,
                "blitz_rate_ci": _ci_rate(d_blitz_rate, d_blitz_n_charted),
                "heavy_rush_rate": d_heavy_rush,
                "heavy_rush_ci": _ci_rate(d_heavy_rush, d_blitz_n_charted),
                "blitz_n_charted": d_blitz_n_charted,
                "sack_rate": d_sack_rate,
                "sack_rate_ci": _ci_rate(d_sack_rate, n_def_dropbacks),
                "pressure_rate": d_pressure,
                "pressure_ci": _ci_rate(d_pressure, n_def_dropbacks),
                "pressure_source": d_pressure_source,
                "n_dropbacks": n_def_dropbacks,
                "rz_td_rate": d_rz_td,
                "rz_td_ci": _ci_rate(d_rz_td, d_rz_trips),
                "rz_trips": d_rz_trips,
                "n_games": n_def_games,
            },
        }

    # --- Rank each team (1 = best) ---
    def _rank(stat_key: str, side: str, lower_is_better: bool = False):
        vals = {t: league[t][side].get(stat_key) for t in league}
        vals = {t: v for t, v in vals.items() if v is not None}
        sorted_teams = sorted(vals, key=lambda t: vals[t],
                              reverse=not lower_is_better)
        return {t: i + 1 for i, t in enumerate(sorted_teams)}

    # Defense: lower EPA allowed = better
    def_pass_ranks = _rank("pass_epa_play", "defense", lower_is_better=True)
    def_rush_ranks = _rank("rush_epa_play", "defense", lower_is_better=True)
    def_blitz_ranks = _rank("blitz_rate", "defense", lower_is_better=False)  # higher = more aggressive
    def_sack_ranks = _rank("sack_rate", "defense", lower_is_better=False)
    def_pressure_ranks = _rank("pressure_rate", "defense", lower_is_better=False)
    def_rz_ranks = _rank("rz_td_rate", "defense", lower_is_better=True)

    # Offense: higher EPA = better
    off_pass_ranks = _rank("pass_epa_play", "offense", lower_is_better=False)
    off_rush_ranks = _rank("rush_epa_play", "offense", lower_is_better=False)

    # --- League averages ---
    def _league_mean(stat_key: str, side: str):
        vals = [league[t][side][stat_key] for t in league
                if league[t][side].get(stat_key) is not None]
        return np.mean(vals) if vals else None

    league_avgs = {
        "offense": {
            "pass_epa_play": _league_mean("pass_epa_play", "offense"),
            "rush_epa_play": _league_mean("rush_epa_play", "offense"),
            "pass_success_rate": _league_mean("pass_success_rate", "offense"),
            "rush_success_rate": _league_mean("rush_success_rate", "offense"),
            "sack_rate": _league_mean("sack_rate", "offense"),
            "pressure_rate": _league_mean("pressure_rate", "offense"),
            "rz_td_rate": _league_mean("rz_td_rate", "offense"),
        },
        "defense": {
            "pass_epa_play": _league_mean("pass_epa_play", "defense"),
            "rush_epa_play": _league_mean("rush_epa_play", "defense"),
            "pass_success_rate": _league_mean("pass_success_rate", "defense"),
            "rush_success_rate": _league_mean("rush_success_rate", "defense"),
            "blitz_rate": _league_mean("blitz_rate", "defense"),
            "heavy_rush_rate": _league_mean("heavy_rush_rate", "defense"),
            "sack_rate": _league_mean("sack_rate", "defense"),
            "pressure_rate": _league_mean("pressure_rate", "defense"),
            "rz_td_rate": _league_mean("rz_td_rate", "defense"),
        },
    }

    # --- Opponent adjustment (mean-regressed) ---
    league_pass_epa = league_avgs["offense"]["pass_epa_play"] or 0
    league_rush_epa = league_avgs["offense"]["rush_epa_play"] or 0

    def _opp_adj(team_val, opp_val, league_avg):
        if team_val is None or opp_val is None:
            return None
        return team_val - (opp_val - league_avg)

    result = {}
    if team in league:
        for side in ["offense", "defense"]:
            d = league[team][side].copy()
            d["league_avg"] = league_avgs[side]
            if side == "offense":
                d["pass_epa_rank"] = off_pass_ranks.get(team)
                d["rush_epa_rank"] = off_rush_ranks.get(team)
            else:
                d["pass_epa_rank"] = def_pass_ranks.get(team)
                d["rush_epa_rank"] = def_rush_ranks.get(team)
                d["blitz_rate_rank"] = def_blitz_ranks.get(team)
                d["sack_rate_rank"] = def_sack_ranks.get(team)
                d["pressure_rank"] = def_pressure_ranks.get(team)
                d["rz_td_rank"] = def_rz_ranks.get(team)
            result[side] = d

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: QB vs Blitz
# ══════════════════════════════════════════════════════════════════════════════

def compute_qb_vs_blitz(pbp: pd.DataFrame, ftn: pd.DataFrame,
                        qb_name: str, team: str,
                        seasons: list[int],
                        shrink_weight: float = 0.3) -> dict:
    """
    QB splits vs blitz / no-blitz across given seasons.

    Uses FTN n_blitzers >= 1 as blitz indicator (primary),
    falls back to PBP number_of_pass_rushers >= 5.

    Returns dict with 'blitz' and 'clean' sub-dicts plus target distribution.
    `shrink_weight` controls how much to shrink multi-season toward recent season.
    """
    # Filter to QB dropbacks
    mask = (pbp["season"].isin(seasons)) & (pbp["passer_player_name"] == qb_name)
    qb_plays = pbp[mask & (pbp["play_type"] == "pass")].copy()

    if qb_plays.empty:
        logger.warning("[matchup] No plays found for QB %s", qb_name)
        return {}

    # Merge FTN for blitz data (skip if already merged globally)
    if not ftn.empty and "n_blitzers" not in qb_plays.columns:
        ftn_sub = ftn[ftn["season"].isin(seasons)]
        qb_plays = _merge_ftn(qb_plays, ftn_sub)

    # Classify blitz
    if "n_blitzers" in qb_plays.columns and qb_plays["n_blitzers"].notna().sum() > 10:
        qb_plays["is_blitz"] = qb_plays["n_blitzers"] >= 1
        blitz_source = "ftn_n_blitzers"
    elif "number_of_pass_rushers" in qb_plays.columns:
        qb_plays["is_blitz"] = qb_plays["number_of_pass_rushers"] >= 5
        blitz_source = "pbp_5plus_rushers"
    else:
        logger.warning("[matchup] No blitz indicator available for %s", qb_name)
        return {}

    blitz = qb_plays[qb_plays["is_blitz"] == True]
    clean = qb_plays[qb_plays["is_blitz"] == False]

    def _split_stats(subset):
        if len(subset) == 0:
            return {}
        n = len(subset)
        comp = subset["complete_pass"].mean() if "complete_pass" in subset.columns else None
        sack = subset["sack"].mean()
        return {
            "n_dropbacks": n,
            "low_confidence": n < _LOW_CONFIDENCE_N,
            "epa_per_dropback": subset["epa"].mean(),
            "epa_ci": _ci_mean(subset["epa"]),
            "comp_pct": comp,
            "comp_pct_ci": _ci_rate(comp, n) if comp is not None else (None, None),
            "adot": subset["air_yards"].mean() if "air_yards" in subset.columns else None,
            "sack_rate": sack,
            "sack_rate_ci": _ci_rate(sack, n),
            "screen_rate": subset["is_screen_pass"].mean() if "is_screen_pass" in subset.columns and subset["is_screen_pass"].notna().any() else None,
            "pa_rate": subset["is_play_action"].mean() if "is_play_action" in subset.columns and subset["is_play_action"].notna().any() else None,
            "throwaway_rate": subset["is_throw_away"].mean() if "is_throw_away" in subset.columns and subset["is_throw_away"].notna().any() else None,
            "oop_rate": subset["is_qb_out_of_pocket"].mean() if "is_qb_out_of_pocket" in subset.columns and subset["is_qb_out_of_pocket"].notna().any() else None,
        }

    result = {
        "blitz_source": blitz_source,
        "blitz": _split_stats(blitz),
        "clean": _split_stats(clean),
        "blitz_rate_faced": len(blitz) / len(qb_plays) if len(qb_plays) > 0 else None,
    }

    # --- Target distribution under blitz ---
    target_dist = {}
    for is_b, label in [(True, "blitz"), (False, "clean")]:
        subset = qb_plays[qb_plays["is_blitz"] == is_b]
        if len(subset) == 0:
            continue
        targets = subset[subset["receiver_player_name"].notna()]
        if len(targets) == 0:
            continue
        total = len(targets)
        by_rec = targets.groupby("receiver_player_name").agg(
            tgt_count=("receiver_player_name", "count"),
            adot=("air_yards", "mean"),
            epa=("epa", "mean"),
            comp_pct=("complete_pass", "mean"),
        ).reset_index()
        by_rec["tgt_share"] = by_rec["tgt_count"] / total
        by_rec = by_rec.sort_values("tgt_share", ascending=False)
        target_dist[label] = by_rec.to_dict("records")

    result["target_distribution"] = target_dist

    # --- Formation vs blitz ---
    formation_vs_blitz = {}
    if "shotgun" in qb_plays.columns:
        for is_b, label in [(True, "blitz"), (False, "clean")]:
            subset = qb_plays[qb_plays["is_blitz"] == is_b]
            if len(subset) == 0:
                continue
            formation_vs_blitz[label] = {
                "shotgun_rate": subset["shotgun"].mean(),
                "n_plays": len(subset),
            }
    result["formation_vs_blitz"] = formation_vs_blitz

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Separation Estimates
# ══════════════════════════════════════════════════════════════════════════════

def compute_separation_estimates(
    ngs: pd.DataFrame, pbp: pd.DataFrame,
    team: str, opp_team: str,
    season: int, through_week: int,
    shrink_factor: float = 0.5,
) -> list[dict]:
    """
    Expected separation per receiver vs opponent, opponent-adjusted.

    Uses NGS avg_separation by receiver-week directly:
      receiver_baseline = season avg_separation (all weeks)
      opp_def_effect = avg(receiver_sep_vs_opp - receiver_baseline) by position
      expected = baseline + shrink_factor * opp_effect

    Returns list of dicts per receiver with CIs.
    """
    if ngs.empty:
        return []

    # Season-to-date NGS for team's receivers
    team_ngs = ngs[
        (ngs["team_abbr"] == team) &
        (ngs["season"] == season) &
        (ngs["week"] <= through_week) &
        (ngs["season_type"] == "REG")
    ].copy()

    if team_ngs.empty:
        # Try prior season
        team_ngs = ngs[
            (ngs["team_abbr"] == team) &
            (ngs["season"] == season - 1) &
            (ngs["season_type"] == "REG")
        ].copy()
        if team_ngs.empty:
            return []

    # Receiver baselines (mean + std of weekly separation)
    receiver_stats = team_ngs.groupby(["player_display_name", "player_position"]).agg(
        avg_sep=("avg_separation", "mean"),
        sep_std=("avg_separation", "std"),
        avg_cush=("avg_cushion", "mean"),
        n_weeks=("week", "count"),
        total_targets=("targets", "sum"),
    ).reset_index()

    # League-wide avg separation by position
    league_ngs = ngs[
        (ngs["season"] == season) &
        (ngs["week"] <= through_week) &
        (ngs["season_type"] == "REG")
    ]
    league_avg_by_pos = league_ngs.groupby("player_position")["avg_separation"].mean()

    # --- Opponent defensive effect from NGS receiver-week data ---
    # Find which offensive teams faced opp_team as defense, by week
    opp_games = pbp[
        (pbp["defteam"] == opp_team) &
        (pbp["season"] == season) &
        (pbp["week"] <= through_week)
    ][["week", "posteam"]].drop_duplicates()

    # For each (week, off_team), compare receiver separation vs opp_team
    # to that receiver's own season baseline (excluding that week)
    effects_by_pos = defaultdict(list)
    for _, game in opp_games.iterrows():
        w = int(game["week"])
        off_team = game["posteam"]

        # Receivers' separation in the week they faced opp_team
        week_sep = ngs[
            (ngs["team_abbr"] == off_team) &
            (ngs["season"] == season) &
            (ngs["week"] == w) &
            (ngs["season_type"] == "REG")
        ]

        for _, rec in week_sep.iterrows():
            player = rec["player_display_name"]
            pos = rec["player_position"]
            sep_vs_opp = rec["avg_separation"]
            if sep_vs_opp is None or np.isnan(sep_vs_opp):
                continue

            # Baseline: that receiver's avg over all other weeks
            baseline_weeks = ngs[
                (ngs["player_display_name"] == player) &
                (ngs["team_abbr"] == off_team) &
                (ngs["season"] == season) &
                (ngs["week"] <= through_week) &
                (ngs["week"] != w) &
                (ngs["season_type"] == "REG")
            ]["avg_separation"]

            if len(baseline_weeks) >= 2:
                bl = baseline_weeks.mean()
                effects_by_pos[pos].append(sep_vs_opp - bl)

    # Aggregate by position with CIs
    opp_def_effect_by_pos = {}
    for pos, effects in effects_by_pos.items():
        if len(effects) >= 3:
            arr = np.array(effects)
            opp_def_effect_by_pos[pos] = {
                "mean_effect": float(arr.mean()),
                "n": len(effects),
                "se": float(arr.std() / np.sqrt(len(effects))),
                "ci_lo": float(arr.mean() - 1.96 * arr.std() / np.sqrt(len(effects))),
                "ci_hi": float(arr.mean() + 1.96 * arr.std() / np.sqrt(len(effects))),
            }

    # Build expected separation per receiver
    results = []
    for _, rec in receiver_stats.iterrows():
        baseline = rec["avg_sep"]
        if baseline is None or np.isnan(baseline):
            continue
        pos = rec["player_position"]
        league_pos_avg = league_avg_by_pos.get(pos, baseline)

        # Opponent effect from NGS (direct measurement, not YAC proxy)
        opp_info = opp_def_effect_by_pos.get(pos, {})
        opp_effect = opp_info.get("mean_effect", 0.0)
        opp_effect_n = opp_info.get("n", 0)

        # Shrink toward baseline
        expected = baseline + shrink_factor * opp_effect

        # CI from weekly std
        sep_std = rec["sep_std"] if rec["sep_std"] and not np.isnan(rec["sep_std"]) else 0.3
        n = max(rec["n_weeks"], 1)
        se = sep_std / np.sqrt(n)

        results.append({
            "player_name": rec["player_display_name"],
            "position": pos,
            "avg_separation": round(baseline, 2),
            "avg_cushion": round(rec["avg_cush"], 2) if rec["avg_cush"] and not np.isnan(rec["avg_cush"]) else None,
            "n_weeks": int(rec["n_weeks"]),
            "low_confidence": int(rec["n_weeks"]) < 4,
            "total_targets": int(rec["total_targets"]),
            "opp_effect": round(opp_effect, 3),
            "opp_effect_n": opp_effect_n,
            "opp_effect_ci": (round(opp_info.get("ci_lo", 0), 3),
                              round(opp_info.get("ci_hi", 0), 3)) if opp_effect_n >= 3 else (None, None),
            "league_pos_avg": round(league_pos_avg, 2),
            "expected_separation": round(expected, 2),
            "expected_sep_ci_lo": round(expected - 1.96 * se, 2),
            "expected_sep_ci_hi": round(expected + 1.96 * se, 2),
            "ngs_season": int(season) if len(team_ngs[team_ngs["season"] == season]) > 0 else int(season - 1),
        })

    return sorted(results, key=lambda x: x.get("total_targets", 0), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: CB-on-WR Proxy
# ══════════════════════════════════════════════════════════════════════════════

def _extract_cb_wr_evidence(pbp: pd.DataFrame, def_team: str,
                            season: int,
                            max_week: Optional[int] = None) -> pd.DataFrame:
    """
    Extract CB-on-WR evidence from PBP credited defenders on pass plays.

    Evidence types and weights:
      - PBU (pass_defense_1): weight 1.0 (strong — CB was in coverage)
      - INT (interception_player): weight 1.0 (strong — CB was in coverage)
      - Tackle on completion, YAC < 5: weight 0.5 (moderate — likely in coverage)
      - Tackle on completion, YAC >= 5: weight 0.2 (weak — could be pursuit angle)

    Returns DataFrame with one row per evidence instance.
    """
    mask = (pbp["defteam"] == def_team) & (pbp["season"] == season)
    if max_week is not None:
        mask = mask & (pbp["week"] <= max_week)
    targets = pbp[mask & (pbp["play_type"] == "pass") &
                  (pbp["receiver_player_name"].notna())].copy()

    if targets.empty:
        return pd.DataFrame()

    evidence_rows = []

    for _, play in targets.iterrows():
        receiver = play["receiver_player_name"]
        receiver_id = play.get("receiver_player_id")
        rec_team = play["posteam"]
        gid = play["game_id"]
        pid = play.get("play_id", "")
        pass_loc = play.get("pass_location")
        air_yds = play.get("air_yards")
        yac = play.get("yards_after_catch")
        week = play["week"]

        base = {
            "game_id": gid,
            "season": season,
            "week": int(week),
            "receiver_gsis_id": receiver_id,
            "receiver_name": receiver,
            "receiver_team": rec_team,
            "pass_location": pass_loc,
            "air_yards": air_yds,
            "yards_after_catch": yac,
            "play_id": str(pid),
        }

        # PBU
        pd1_id = play.get("pass_defense_1_player_id")
        pd1_name = play.get("pass_defense_1_player_name")
        if pd1_id and pd.notna(pd1_id):
            evidence_rows.append({
                **base,
                "defender_gsis_id": pd1_id,
                "defender_name": pd1_name,
                "defender_team": def_team,
                "evidence_type": "pbu",
                "evidence_weight": 1.0,
            })

        # INT
        int_id = play.get("interception_player_id")
        int_name = play.get("interception_player_name")
        if int_id and pd.notna(int_id) and int_id != pd1_id:
            evidence_rows.append({
                **base,
                "defender_gsis_id": int_id,
                "defender_name": int_name,
                "defender_team": def_team,
                "evidence_type": "int",
                "evidence_weight": 1.0,
            })

        # Tackle on completion
        if play.get("complete_pass") == 1:
            for tackle_col_id, tackle_col_name in [
                ("solo_tackle_1_player_id", "solo_tackle_1_player_name"),
                ("assist_tackle_1_player_id", "assist_tackle_1_player_name"),
            ]:
                t_id = play.get(tackle_col_id)
                t_name = play.get(tackle_col_name)
                t_team = play.get(tackle_col_id.replace("_player_id", "_team"), def_team)
                if t_id and pd.notna(t_id) and t_team == def_team:
                    yac_val = yac if yac and pd.notna(yac) else 99
                    if yac_val < 5:
                        etype = "tackle_short_yac"
                        weight = 0.5
                    else:
                        etype = "tackle_long_yac"
                        weight = 0.2
                    evidence_rows.append({
                        **base,
                        "defender_gsis_id": t_id,
                        "defender_name": t_name,
                        "defender_team": def_team,
                        "evidence_type": etype,
                        "evidence_weight": weight,
                    })

    return pd.DataFrame(evidence_rows)


def _compute_side_tendencies(pbp: pd.DataFrame, team: str, season: int,
                             max_week: Optional[int] = None,
                             role: str = "defender") -> dict:
    """
    Compute left/right/slot share for CBs (defenders) or WRs (receivers).

    Slot ≈ middle pass_location with short air_yards (< 15).
    Left/right from pass_location.
    """
    mask = (pbp["season"] == season) & (pbp["play_type"] == "pass")
    if max_week:
        mask = mask & (pbp["week"] <= max_week)
    if role == "defender":
        mask = mask & (pbp["defteam"] == team)
    else:
        mask = mask & (pbp["posteam"] == team)

    plays = pbp[mask & pbp["pass_location"].notna()].copy()
    if plays.empty:
        return {}

    # Approximate slot: middle location OR short air_yards (< 10)
    plays["is_slot"] = (plays["pass_location"] == "middle") | (
        plays["air_yards"].fillna(20) < 10)
    plays["side"] = np.where(
        plays["is_slot"], "slot",
        np.where(plays["pass_location"] == "left", "left", "right"))

    if role == "defender":
        # For each defender credited on pass plays, compute their side distribution
        result = {}
        for defender_col_id, defender_col_name in [
            ("pass_defense_1_player_id", "pass_defense_1_player_name"),
            ("solo_tackle_1_player_id", "solo_tackle_1_player_name"),
        ]:
            nearby = plays[plays[defender_col_id].notna()]
            if nearby.empty:
                continue
            for pid, grp in nearby.groupby(defender_col_id):
                if pid in result:
                    # Merge evidence
                    for s in ["left", "right", "slot"]:
                        result[pid]["counts"][s] += (grp["side"] == s).sum()
                else:
                    result[pid] = {
                        "name": grp[defender_col_name].iloc[0],
                        "counts": {s: int((grp["side"] == s).sum())
                                   for s in ["left", "right", "slot"]},
                    }
        # Convert to shares
        tendencies = {}
        for pid, info in result.items():
            total = sum(info["counts"].values())
            if total < 3:
                continue
            tendencies[pid] = {
                "name": info["name"],
                "n_targets_near": total,
                "left_share": info["counts"]["left"] / total,
                "right_share": info["counts"]["right"] / total,
                "slot_share": info["counts"]["slot"] / total,
            }
        return tendencies
    else:
        # WR side tendencies from their targets
        result = {}
        receivers = plays[plays["receiver_player_id"].notna()]
        for pid, grp in receivers.groupby("receiver_player_id"):
            total = len(grp)
            if total < 3:
                continue
            result[pid] = {
                "name": grp["receiver_player_name"].iloc[0],
                "n_targets": total,
                "left_share": (grp["side"] == "left").mean(),
                "right_share": (grp["side"] == "right").mean(),
                "slot_share": (grp["side"] == "slot").mean(),
            }
        return result


def compute_cb_on_wr_proxy(
    pbp: pd.DataFrame,
    rosters: pd.DataFrame,
    off_team: str, def_team: str,
    season: int, max_week: Optional[int] = None,
    prior_weight: float = 2.0,
) -> list[dict]:
    """
    Bayesian CB-on-WR matchup estimate.

    P(CB j covers WR i) = side-overlap prior + direct evidence, shrunk.

    prior: P_prior(j, i) ∝ overlap of CB j's side tendency with WR i's side tendency
    evidence: weighted count of CB j credited on WR i's targets
    posterior: (prior_weight * prior + evidence_weight_sum) / (prior_weight + total_evidence)

    Returns list of dicts per WR with top CB matchups and probabilities.
    """
    # Get CB roster for def_team
    cb_roster = rosters[
        (rosters["team"] == def_team) &
        (rosters["season"] == season) &
        (rosters["position"].isin(["CB", "DB"]))
    ][["gsis_id", "player_name", "position"]].drop_duplicates("gsis_id")

    # Get WR roster for off_team
    wr_roster = rosters[
        (rosters["team"] == off_team) &
        (rosters["season"] == season) &
        (rosters["position"].isin(["WR", "TE"]))
    ][["gsis_id", "player_name", "position"]].drop_duplicates("gsis_id")

    if cb_roster.empty or wr_roster.empty:
        return []

    # Extract evidence
    evidence_df = _extract_cb_wr_evidence(pbp, def_team, season, max_week)

    # Side tendencies
    cb_sides = _compute_side_tendencies(pbp, def_team, season, max_week, role="defender")
    wr_sides = _compute_side_tendencies(pbp, off_team, season, max_week, role="receiver")

    # For each WR, compute P(CB j | WR i)
    results = []
    for _, wr in wr_roster.iterrows():
        wr_id = wr["gsis_id"]
        wr_name = wr["player_name"]
        wr_pos = wr["position"]
        wr_side = wr_sides.get(wr_id, {})

        if not wr_side:
            continue

        cb_probs = {}
        for _, cb in cb_roster.iterrows():
            cb_id = cb["gsis_id"]
            cb_name = cb["player_name"]
            cb_side = cb_sides.get(cb_id, {})

            # Side-overlap prior
            if wr_side and cb_side:
                overlap = (
                    wr_side.get("left_share", 0.33) * cb_side.get("left_share", 0.33) +
                    wr_side.get("right_share", 0.33) * cb_side.get("right_share", 0.33) +
                    wr_side.get("slot_share", 0.33) * cb_side.get("slot_share", 0.33)
                )
            else:
                overlap = 1.0 / max(len(cb_roster), 1)

            # Direct evidence: weighted sum of CB credited on this WR's targets
            if not evidence_df.empty:
                wr_evidence = evidence_df[
                    (evidence_df["receiver_name"] == wr_name) &
                    (evidence_df["defender_gsis_id"] == cb_id)
                ]
                evidence_weight_sum = wr_evidence["evidence_weight"].sum()
                evidence_n = len(wr_evidence)
            else:
                evidence_weight_sum = 0
                evidence_n = 0

            # Bayesian update
            posterior = (prior_weight * overlap + evidence_weight_sum) / (
                prior_weight + evidence_weight_sum + 0.001)

            cb_probs[cb_id] = {
                "cb_name": cb_name,
                "cb_gsis_id": cb_id,
                "prior_overlap": round(overlap, 3),
                "evidence_n": evidence_n,
                "evidence_weight": round(evidence_weight_sum, 2),
                "probability": round(posterior, 3),
            }

        if not cb_probs:
            continue

        # Normalize probabilities
        total_p = sum(v["probability"] for v in cb_probs.values())
        if total_p > 0:
            for cb_id in cb_probs:
                cb_probs[cb_id]["probability"] = round(
                    cb_probs[cb_id]["probability"] / total_p, 3)

        # Sort by probability
        sorted_cbs = sorted(cb_probs.values(),
                            key=lambda x: x["probability"], reverse=True)

        primary = sorted_cbs[0] if sorted_cbs else {}
        total_evidence = sum(c["evidence_n"] for c in sorted_cbs)

        # Shadow risk: if primary CB has >= 40% share with >= 5 evidence plays
        shadow_risk = (primary.get("probability", 0) >= 0.4 and
                       primary.get("evidence_n", 0) >= 5)

        results.append({
            "wr_name": wr_name,
            "wr_gsis_id": wr_id,
            "wr_position": wr_pos,
            "wr_side": wr_side,
            "primary_cb": primary.get("cb_name"),
            "primary_cb_gsis_id": primary.get("cb_gsis_id"),
            "primary_cb_prob": primary.get("probability"),
            "primary_cb_evidence_n": primary.get("evidence_n", 0),
            "primary_cb_overlap": primary.get("prior_overlap"),
            "shadow_risk": shadow_risk,
            "total_evidence": total_evidence,
            "all_cbs": sorted_cbs[:5],  # top 5 CB matchups
        })

    return sorted(results, key=lambda x: x.get("total_evidence", 0), reverse=True)


def validate_cb_proxy_accuracy(pbp: pd.DataFrame, rosters: pd.DataFrame,
                               seasons: list[int]) -> dict:
    """
    Validate CB-on-WR proxy: how often does the proxy's top CB match
    the PBU/INT credited defender on a target?

    Uses leave-one-out: for each game, build proxy from prior games,
    check if top CB matches the credited defender on that game's targets.
    """
    results = {"total_targets_with_evidence": 0,
               "top_cb_match": 0,
               "top2_cb_match": 0,
               "by_evidence_volume": defaultdict(lambda: {"total": 0, "match": 0})}

    for season in seasons:
        season_pbp = pbp[pbp["season"] == season]
        teams = season_pbp["defteam"].unique()

        for def_team in teams:
            games = sorted(season_pbp[season_pbp["defteam"] == def_team]["week"].unique())
            if len(games) < 4:
                continue

            for week in games[3:]:  # need at least 3 prior games
                # Build proxy from prior weeks
                proxy = compute_cb_on_wr_proxy(
                    pbp, rosters,
                    off_team="",  # all teams
                    def_team=def_team,
                    season=season,
                    max_week=week - 1)

                # Check this game's strong evidence (PBU/INT only)
                game_evidence = _extract_cb_wr_evidence(
                    pbp, def_team, season, max_week=week)
                game_evidence = game_evidence[
                    (game_evidence["week"] == week) &
                    (game_evidence["evidence_type"].isin(["pbu", "int"]))]

                if game_evidence.empty:
                    continue

                # Build lookup: WR name -> proxy's top CB
                proxy_lookup = {p["wr_name"]: p for p in proxy}

                for _, ev in game_evidence.iterrows():
                    wr = ev["receiver_name"]
                    if wr not in proxy_lookup:
                        continue

                    results["total_targets_with_evidence"] += 1
                    pred = proxy_lookup[wr]
                    actual_cb = ev["defender_gsis_id"]

                    if pred.get("primary_cb_gsis_id") == actual_cb:
                        results["top_cb_match"] += 1
                    top2_ids = [c["cb_gsis_id"] for c in pred.get("all_cbs", [])[:2]]
                    if actual_cb in top2_ids:
                        results["top2_cb_match"] += 1

                    # By evidence volume
                    vol = pred.get("total_evidence", 0)
                    bucket = "0-4" if vol < 5 else ("5-14" if vol < 15 else "15+")
                    results["by_evidence_volume"][bucket]["total"] += 1
                    if pred.get("primary_cb_gsis_id") == actual_cb:
                        results["by_evidence_volume"][bucket]["match"] += 1

    total = results["total_targets_with_evidence"]
    if total > 0:
        results["top1_accuracy"] = round(results["top_cb_match"] / total, 3)
        results["top2_accuracy"] = round(results["top2_cb_match"] / total, 3)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Section 4b: Shadow Log (reported projections from free sources)
# ══════════════════════════════════════════════════════════════════════════════

def add_shadow_report(season: int, week: int, def_team: str, cb_name: str,
                      wr_name: str, off_team: str, source: str,
                      source_date: Optional[str] = None,
                      game_id: Optional[str] = None,
                      notes: Optional[str] = None):
    """Add a reported shadow matchup from a free source (PFF preview, RotoBaller, ESPN)."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute("""
        INSERT OR REPLACE INTO nfl_shadow_log
        (season, week, game_id, def_team, cb_name, wr_name, off_team,
         source, source_date, confidence, notes, entered_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reported', ?, ?)
    """, (season, week, game_id, def_team, cb_name, wr_name, off_team,
          source, source_date, notes,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    logger.info("Shadow logged: %s %s shadows %s %s (W%d, source=%s)",
                def_team, cb_name, off_team, wr_name, week, source)


def confirm_shadow(season: int, week: int, def_team: str, cb_name: str,
                    wr_name: str, confirmed: bool, notes: Optional[str] = None):
    """Post-game: mark a shadow report as confirmed or not confirmed."""
    _ensure_tables()
    status = "confirmed" if confirmed else "not_confirmed"
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute("""
        UPDATE nfl_shadow_log SET confidence = ?, notes = COALESCE(?, notes)
        WHERE season = ? AND week = ? AND def_team = ? AND cb_name = ? AND wr_name = ?
    """, (status, notes, season, week, def_team, cb_name, wr_name))
    if conn.total_changes == 0:
        logger.warning("No shadow found to confirm: %s %s→%s W%d",
                        def_team, cb_name, wr_name, week)
    conn.commit()
    conn.close()
    logger.info("Shadow %s: %s %s→%s W%d", status, def_team, cb_name, wr_name, week)


def get_shadow_reports(season: int, week: int,
                       def_team: Optional[str] = None) -> list[dict]:
    """Retrieve reported shadow matchups for a given week."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM nfl_shadow_log WHERE season = ? AND week = ?"
    params: list = [season, week]
    if def_team:
        query += " AND def_team = ?"
        params.append(def_team)
    query += " ORDER BY def_team, cb_name"
    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()
    return rows


def shadow_source_accuracy(season: int) -> dict:
    """Compute projected-vs-actual shadow accuracy by source for a season."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT source, confidence, COUNT(*) as n
        FROM nfl_shadow_log
        WHERE season = ? AND confidence != 'reported'
        GROUP BY source, confidence
    """, (season,)).fetchall()
    conn.close()

    by_source: dict[str, dict] = {}
    for r in rows:
        src = r['source']
        if src not in by_source:
            by_source[src] = {'confirmed': 0, 'not_confirmed': 0, 'total': 0}
        by_source[src][r['confidence']] = r['n']
        by_source[src]['total'] += r['n']

    for src in by_source:
        total = by_source[src]['total']
        if total > 0:
            by_source[src]['accuracy'] = round(
                by_source[src]['confirmed'] / total, 3)
    return by_source


# ── Elite CB list (frozen for forward test) ─────────────────────────────────

def freeze_elite_cb_list():
    """Freeze the elite CB list from BDB 2023 tracking-data metrics.

    Elite = bottom 25% of EPA/target allowed among CBs with >= 50 coverage snaps.
    Source: bdb_closest_cb.csv + supplementary_data.csv (BDB 2026-analytics competition).
    This list is FIXED once written and must not change mid-test.
    """
    import os as _os
    bdb_dir = _os.path.join(_HERE, "..", "data", "bdb")
    cb_file = _os.path.join(bdb_dir, "bdb_closest_cb.csv")
    supp_file = _os.path.join(bdb_dir, "supplementary_data.csv")

    if not _os.path.exists(cb_file) or not _os.path.exists(supp_file):
        raise FileNotFoundError(
            f"BDB data not found at {bdb_dir}. "
            "Run BDB 2023 processing first.")

    cb = pd.read_csv(cb_file)
    supp = pd.read_csv(supp_file, low_memory=False)

    cb_plays = cb.merge(
        supp[['game_id', 'play_id', 'yards_gained',
              'expected_points_added', 'pass_result']],
        on=['game_id', 'play_id'], how='left')
    cb_plays['completion'] = (cb_plays.pass_result == 'C').astype(int)

    cb_metrics = (cb_plays
                  .groupby(['defensive_team', 'closest_cb_name'])
                  .agg(snaps=('play_id', 'count'),
                       epa_allowed=('expected_points_added', 'mean'),
                       comp_pct=('completion', 'mean'),
                       yds_allowed=('yards_gained', 'mean'))
                  .reset_index())
    cb_metrics = cb_metrics[cb_metrics.snaps >= 50]

    q25 = cb_metrics.epa_allowed.quantile(0.25)
    elite = cb_metrics[cb_metrics.epa_allowed <= q25].copy()

    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    frozen_at = datetime.now(timezone.utc).isoformat()

    # Clear and re-freeze (idempotent)
    conn.execute("DELETE FROM nfl_elite_cb_list WHERE source = 'BDB_2023'")
    for _, row in elite.iterrows():
        conn.execute("""
            INSERT INTO nfl_elite_cb_list
            (cb_name, team_2023, epa_per_target, comp_pct, yds_per_target,
             coverage_snaps, source, frozen_at)
            VALUES (?, ?, ?, ?, ?, ?, 'BDB_2023', ?)
        """, (row.closest_cb_name, row.defensive_team,
              round(row.epa_allowed, 4), round(row.comp_pct, 4),
              round(row.yds_allowed, 2), int(row.snaps), frozen_at))
    conn.commit()
    conn.close()

    logger.info("Froze %d elite CBs (EPA/tgt <= %.3f, min 50 snaps, BDB 2023)",
                len(elite), q25)
    return {
        'n_elite': len(elite),
        'threshold': round(q25, 4),
        'frozen_at': frozen_at,
        'cbs': elite[['closest_cb_name', 'defensive_team', 'epa_allowed',
                       'snaps']].to_dict('records')
    }


def get_elite_cb_list() -> list[dict]:
    """Return the frozen elite CB list."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM nfl_elite_cb_list ORDER BY epa_per_target").fetchall()]
    conn.close()
    return rows


def is_elite_cb(cb_name: str) -> bool:
    """Check if a CB is on the frozen elite list (case-insensitive match)."""
    elite = get_elite_cb_list()
    return any(e['cb_name'].lower() == cb_name.lower() for e in elite)


def record_elite_cb_shadow_result(
        season: int, week: int, game_id: str, def_team: str,
        cb_name: str, wr_name: str, off_team: str,
        wr_rec_yds: float, market_line: float, v12_projection: float,
        wr_targets: int, wr_yds_per_tgt: float,
        shadow_confirmed: str):
    """Record the outcome of an elite CB shadow for the forward test."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute("""
        INSERT OR REPLACE INTO nfl_elite_cb_shadow_results
        (season, week, game_id, def_team, cb_name, wr_name, off_team,
         wr_rec_yds, market_line, v12_projection, wr_targets, wr_yds_per_tgt,
         shadow_confirmed, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (season, week, game_id, def_team, cb_name, wr_name, off_team,
          wr_rec_yds, market_line, v12_projection, wr_targets, wr_yds_per_tgt,
          shadow_confirmed, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def evaluate_elite_cb_shadow_test() -> dict:
    """Evaluate the forward test: do confirmed elite-CB shadows suppress WR1 rec yds?

    Returns test statistics: N, mean diff vs market, mean diff vs v1.2,
    game-clustered bootstrap CIs, and whether the market already prices shadows.
    """
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("""
        SELECT * FROM nfl_elite_cb_shadow_results
        WHERE wr_rec_yds IS NOT NULL AND market_line IS NOT NULL
    """).fetchall()]
    conn.close()

    if not rows:
        return {'n': 0, 'status': 'NO DATA — waiting for shadow entries + settlements'}

    df = pd.DataFrame(rows)
    df['diff_vs_market'] = df.wr_rec_yds - df.market_line
    df['diff_vs_v12'] = df.wr_rec_yds - df.v12_projection

    result = {
        'n': len(df),
        'n_confirmed': int((df.shadow_confirmed == 'confirmed').sum()),
        'n_not_confirmed': int((df.shadow_confirmed == 'not_confirmed').sum()),
        'n_reported': int((df.shadow_confirmed == 'reported').sum()),
        'mean_rec_yds': round(float(df.wr_rec_yds.mean()), 1),
        'mean_market_line': round(float(df.market_line.mean()), 1),
        'mean_diff_vs_market': round(float(df.diff_vs_market.mean()), 1),
        'mean_diff_vs_v12': round(float(df.diff_vs_v12.mean()), 1),
        'mean_targets': round(float(df.wr_targets.mean()), 1),
        'mean_yds_per_tgt': round(float(df.wr_yds_per_tgt.mean()), 1),
    }

    # Game-clustered bootstrap CI on diff_vs_market
    if len(df) >= 5:
        rng = np.random.RandomState(42)
        games = df.game_id.unique()
        diffs_mkt = []
        diffs_v12 = []
        for _ in range(5000):
            bg = rng.choice(games, size=len(games), replace=True)
            boot = df[df.game_id.isin(set(bg))]
            diffs_mkt.append(boot.diff_vs_market.mean())
            diffs_v12.append(boot.diff_vs_v12.mean())
        result['ci_vs_market'] = [round(x, 1)
                                  for x in np.percentile(diffs_mkt, [2.5, 50, 97.5])]
        result['ci_vs_v12'] = [round(x, 1)
                               for x in np.percentile(diffs_v12, [2.5, 50, 97.5])]
        result['market_prices_shadow'] = (
            "YES — market already adjusts" if result['ci_vs_market'][0] > -5
            else "NO — market does not fully price" if result['ci_vs_market'][2] < -5
            else "INCONCLUSIVE"
        )
    else:
        result['ci_vs_market'] = None
        result['ci_vs_v12'] = None
        result['market_prices_shadow'] = 'INSUFFICIENT N'

    return result


def _format_shadow_log(home_team: str, away_team: str,
                       season: int, week: int) -> str:
    """Format shadow log entries for Lab panel display."""
    reports = get_shadow_reports(season, week)
    if not reports:
        return "\n  No reported shadows for this week. Enter via --add-shadow."

    lines = [f"\n  Reported Shadow Matchups (W{week}) [REPORTED, NOT MODELED]:"]
    lines.append(f"    {'Defense':<6} {'CB':<22} {'shadows':<8} {'WR':<22} "
                 f"{'Source':<14} {'Date':<12} {'Status':<14}")
    lines.append(f"    {'─'*6} {'─'*22} {'─'*8} {'─'*22} {'─'*14} {'─'*12} {'─'*14}")

    for r in reports:
        # Only show reports relevant to this game
        if r['def_team'] not in (home_team, away_team):
            continue
        status = r.get('confidence', 'reported')
        lines.append(f"    {r['def_team']:<6} {r['cb_name']:<22} {'shadows':<8} "
                     f"{r['wr_name']:<22} {r['source']:<14} "
                     f"{r.get('source_date', '—'):<12} {status:<14}")
        if r.get('notes'):
            lines.append(f"           ↳ {r['notes']}")

    if len(lines) <= 3:
        return "\n  No reported shadows for this game."
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Script and Volume
# ══════════════════════════════════════════════════════════════════════════════

def compute_script_volume(pbp: pd.DataFrame, team: str, opp_team: str,
                          season: int, through_week: int) -> dict:
    """
    Pass/run tendencies, PROE, and attempt distributions for game sim.
    """
    team_plays = pbp[
        (pbp["posteam"] == team) &
        (pbp["season"] == season) &
        (pbp["week"] <= through_week) &
        (pbp["play_type"].isin(["pass", "run"]))
    ].copy()

    opp_def_plays = pbp[
        (pbp["defteam"] == opp_team) &
        (pbp["season"] == season) &
        (pbp["week"] <= through_week) &
        (pbp["play_type"].isin(["pass", "run"]))
    ].copy()

    if team_plays.empty:
        return {}

    # Neutral pass rate (1st/2nd down, score within 10, 1st-3rd quarter)
    neutral = team_plays[
        (team_plays["down"].isin([1, 2])) &
        (team_plays["score_differential"].abs() <= 10) &
        (team_plays["qtr"].isin([1, 2, 3]))
    ]
    n_neutral = len(neutral)
    neutral_pass_rate = (neutral["play_type"] == "pass").mean() if n_neutral > 0 else None

    # PROE (pass rate over expected)
    proe = None
    proe_n = 0
    if "xpass" in team_plays.columns:
        valid = team_plays["xpass"].notna()
        proe_n = int(valid.sum())
        if proe_n > 10:
            actual_pass = (team_plays.loc[valid, "play_type"] == "pass").astype(float)
            proe = (actual_pass - team_plays.loc[valid, "xpass"]).mean()

    # Opponent pass rate allowed over expected
    opp_proe = None
    opp_proe_n = 0
    if "xpass" in opp_def_plays.columns:
        valid = opp_def_plays["xpass"].notna()
        opp_proe_n = int(valid.sum())
        if opp_proe_n > 10:
            actual_pass = (opp_def_plays.loc[valid, "play_type"] == "pass").astype(float)
            opp_proe = (actual_pass - opp_def_plays.loc[valid, "xpass"]).mean()

    # Attempt distributions by game
    game_attempts = team_plays.groupby("game_id").agg(
        total_plays=("play_type", "count"),
        pass_att=("play_type", lambda x: (x == "pass").sum()),
        rush_att=("play_type", lambda x: (x == "run").sum()),
    )

    return {
        "neutral_pass_rate": round(neutral_pass_rate, 3) if neutral_pass_rate else None,
        "neutral_pass_rate_ci": _ci_rate(neutral_pass_rate, n_neutral),
        "n_neutral_plays": n_neutral,
        "low_confidence_neutral": n_neutral < _LOW_CONFIDENCE_N,
        "proe": round(proe, 4) if proe is not None else None,
        "proe_n": proe_n,
        "opp_proe_allowed": round(opp_proe, 4) if opp_proe is not None else None,
        "opp_proe_n": opp_proe_n,
        "avg_pass_att": round(game_attempts["pass_att"].mean(), 1),
        "avg_rush_att": round(game_attempts["rush_att"].mean(), 1),
        "avg_total_plays": round(game_attempts["total_plays"].mean(), 1),
        "pass_att_std": round(game_attempts["pass_att"].std(), 1),
        "rush_att_std": round(game_attempts["rush_att"].std(), 1),
        "n_games": len(game_attempts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Points
# ══════════════════════════════════════════════════════════════════════════════

def build_game_matchup_context(
    game_id: str,
    home_team: str, away_team: str,
    season: int, week: int,
    seasons_for_qb: Optional[list[int]] = None,
    persist: bool = True,
    retrospective: bool = False,
) -> dict:
    """
    Build full matchup context for a single game.

    Loads PBP/FTN/NGS, computes all sections, persists to DB.
    Returns dict with all context for Lab display.

    All inputs are pre-kickoff only (through_week = week - 1).
    retrospective=True changes display labels only, not the data window.
    """
    _ensure_tables()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if seasons_for_qb is None:
        seasons_for_qb = [season - 2, season - 1, season]
    seasons_for_qb = [s for s in seasons_for_qb if s >= 2023]

    all_seasons = sorted(set(seasons_for_qb + [season]))

    logger.info("[matchup] Building context for %s (week %d): %s @ %s",
                game_id, week, away_team, home_team)

    # Load data
    pbp = _load_pbp(all_seasons)
    ftn = _load_ftn(all_seasons)
    ngs = _load_ngs_receiving(all_seasons)
    rosters = _load_rosters([season])

    # Merge FTN onto PBP globally
    if not ftn.empty:
        pbp = _merge_ftn(pbp, ftn)

    # ALL inputs must be pre-kickoff: through_week = week - 1 always.
    # retrospective flag only changes display labels, not the data window.
    through_week = week - 1

    result = {
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "season": season,
        "week": week,
        "created_at": now,
    }

    # --- Section 1: Team context ---
    for team_label, team in [("home", home_team), ("away", away_team)]:
        ctx = compute_team_context(pbp, ftn, season, through_week, team)
        result[f"{team_label}_team_context"] = ctx

    # --- Section 2: QB vs blitz ---
    # Identify starting QBs from PBP
    for team_label, team in [("home", home_team), ("away", away_team)]:
        team_passes = pbp[
            (pbp["posteam"] == team) &
            (pbp["season"] == season) &
            (pbp["week"] <= through_week) &
            (pbp["play_type"] == "pass") &
            (pbp["passer_player_name"].notna())
        ]
        if team_passes.empty:
            continue
        qb_name = team_passes["passer_player_name"].value_counts().index[0]
        qb_splits = compute_qb_vs_blitz(pbp, ftn, qb_name, team, seasons_for_qb)
        result[f"{team_label}_qb_vs_blitz"] = qb_splits
        result[f"{team_label}_qb_name"] = qb_name

    # --- Section 3: Separation ---
    for team_label, team, opp in [("home", home_team, away_team),
                                   ("away", away_team, home_team)]:
        sep = compute_separation_estimates(ngs, pbp, team, opp, season, through_week)
        result[f"{team_label}_separation"] = sep

    # --- Section 4: CB-on-WR proxy ---
    for team_label, off_team, def_team in [("home", home_team, away_team),
                                            ("away", away_team, home_team)]:
        proxy = compute_cb_on_wr_proxy(pbp, rosters, off_team, def_team,
                                       season, through_week)
        result[f"{team_label}_cb_proxy"] = proxy

    # --- Section 5: Script/volume ---
    for team_label, team, opp in [("home", home_team, away_team),
                                   ("away", away_team, home_team)]:
        script = compute_script_volume(pbp, team, opp, season, through_week)
        result[f"{team_label}_script"] = script

    # --- Persist ---
    if persist:
        _persist_context(result, now)

    return result


def _persist_context(ctx: dict, now: str):
    """Persist matchup context to historical.db."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    game_id = ctx["game_id"]
    season = ctx["season"]
    week = ctx["week"]

    # Team context
    for side_label in ["home", "away"]:
        team = ctx[f"{side_label}_team"]
        team_ctx = ctx.get(f"{side_label}_team_context", {})
        for side in ["offense", "defense"]:
            d = team_ctx.get(side, {})
            if not d:
                continue
            conn.execute("""
                INSERT OR REPLACE INTO nfl_matchup_team_context
                (game_id, team, season, week, side,
                 pass_epa_play, rush_epa_play, pass_success_rate, rush_success_rate,
                 blitz_rate, heavy_rush_rate, sack_rate, pressure_rate,
                 rz_td_rate, rz_trips, blitz_n_charted, pressure_source,
                 pass_epa_rank, rush_epa_rank,
                 blitz_rate_rank, sack_rate_rank, pressure_rank, rz_td_rank,
                 n_games, data_available, created_at)
                VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?, ?,?,?)
            """, (game_id, team, season, week, side,
                  d.get("pass_epa_play"), d.get("rush_epa_play"),
                  d.get("pass_success_rate"), d.get("rush_success_rate"),
                  d.get("blitz_rate"), d.get("heavy_rush_rate"),
                  d.get("sack_rate"), d.get("pressure_rate"),
                  d.get("rz_td_rate"), d.get("rz_trips"),
                  d.get("blitz_n_charted"), d.get("pressure_source"),
                  d.get("pass_epa_rank"), d.get("rush_epa_rank"),
                  d.get("blitz_rate_rank"), d.get("sack_rate_rank"),
                  d.get("pressure_rank"), d.get("rz_td_rank"),
                  d.get("n_games"), "pbp,ftn", now))

    # Player context (QB, WR separation, CB proxy)
    for side_label in ["home", "away"]:
        team = ctx[f"{side_label}_team"]

        # QB vs blitz
        qb_name = ctx.get(f"{side_label}_qb_name")
        qb_data = ctx.get(f"{side_label}_qb_vs_blitz", {})
        if qb_name and qb_data:
            blitz = qb_data.get("blitz", {})
            clean = qb_data.get("clean", {})
            conn.execute("""
                INSERT OR REPLACE INTO nfl_matchup_player_context
                (game_id, player_name, team, position, season, week,
                 qb_blitz_epa, qb_clean_epa, qb_blitz_comp_pct, qb_clean_comp_pct,
                 qb_blitz_adot, qb_clean_adot, qb_blitz_sack_rate,
                 qb_screen_rate_blitz, qb_pa_rate_blitz, qb_throwaway_rate_blitz,
                 data_available, created_at)
                VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,?)
            """, (game_id, qb_name, team, "QB", season, week,
                  blitz.get("epa_per_dropback"), clean.get("epa_per_dropback"),
                  blitz.get("comp_pct"), clean.get("comp_pct"),
                  blitz.get("adot"), clean.get("adot"),
                  blitz.get("sack_rate"),
                  blitz.get("screen_rate"), blitz.get("pa_rate"),
                  blitz.get("throwaway_rate"),
                  "pbp,ftn", now))

        # Separation
        for rec in ctx.get(f"{side_label}_separation", []):
            conn.execute("""
                INSERT OR REPLACE INTO nfl_matchup_player_context
                (game_id, player_name, team, position, season, week,
                 ngs_avg_separation, ngs_avg_cushion, ngs_season,
                 expected_separation, expected_sep_ci_lo, expected_sep_ci_hi,
                 data_available, created_at)
                VALUES (?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?)
                ON CONFLICT(game_id, player_name, team) DO UPDATE SET
                    ngs_avg_separation = excluded.ngs_avg_separation,
                    ngs_avg_cushion = excluded.ngs_avg_cushion,
                    ngs_season = excluded.ngs_season,
                    expected_separation = excluded.expected_separation,
                    expected_sep_ci_lo = excluded.expected_sep_ci_lo,
                    expected_sep_ci_hi = excluded.expected_sep_ci_hi
            """, (game_id, rec["player_name"], team, rec["position"],
                  season, week,
                  rec.get("avg_separation"), rec.get("avg_cushion"),
                  rec.get("ngs_season"),
                  rec.get("expected_separation"),
                  rec.get("expected_sep_ci_lo"),
                  rec.get("expected_sep_ci_hi"),
                  "ngs", now))

        # CB proxy
        for wr in ctx.get(f"{side_label}_cb_proxy", []):
            conn.execute("""
                INSERT OR REPLACE INTO nfl_matchup_player_context
                (game_id, player_name, team, position, season, week,
                 primary_cb_name, primary_cb_gsis_id, primary_cb_prob,
                 cb_evidence_n, cb_side_overlap, shadow_risk_flag,
                 data_available, created_at)
                VALUES (?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?)
                ON CONFLICT(game_id, player_name, team) DO UPDATE SET
                    primary_cb_name = excluded.primary_cb_name,
                    primary_cb_gsis_id = excluded.primary_cb_gsis_id,
                    primary_cb_prob = excluded.primary_cb_prob,
                    cb_evidence_n = excluded.cb_evidence_n,
                    cb_side_overlap = excluded.cb_side_overlap,
                    shadow_risk_flag = excluded.shadow_risk_flag
            """, (game_id, wr["wr_name"], team, wr.get("wr_position"),
                  season, week,
                  wr.get("primary_cb"), wr.get("primary_cb_gsis_id"),
                  wr.get("primary_cb_prob"),
                  wr.get("total_evidence"), wr.get("primary_cb_overlap"),
                  1 if wr.get("shadow_risk") else 0,
                  "pbp", now))

    conn.commit()
    conn.close()
    logger.info("[matchup] Persisted context for %s", game_id)


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Candidate Features + Hypotheses
# ══════════════════════════════════════════════════════════════════════════════

_CANDIDATE_FEATURES = [
    {
        "feature_name": "opp_blitz_rate",
        "hypothesis": "Higher opponent blitz rate increases QB pressure, reducing passing EPA and increasing sack rate. Receivers see more hot routes / checkdowns under blitz.",
        "predicted_direction": "- for pass_yds, + for receptions (checkdowns)",
        "data_source": "FTN n_blitzers, PBP number_of_pass_rushers",
    },
    {
        "feature_name": "qb_blitz_epa_delta",
        "hypothesis": "QBs with larger EPA drop under blitz face higher risk in blitz-heavy matchups. Interaction: QB blitz sensitivity × opponent blitz rate.",
        "predicted_direction": "- for pass_yds when QB blitz-sensitive AND opp blitzes often",
        "data_source": "PBP EPA + FTN blitz",
    },
    {
        "feature_name": "target_share_shift_blitz",
        "hypothesis": "Receivers who gain target share under blitz (checkdown backs, slot receivers) outperform their projection vs blitz-heavy defenses.",
        "predicted_direction": "+ for receptions/rec_yds when WR gains share under blitz",
        "data_source": "PBP receiver + FTN blitz",
    },
    {
        "feature_name": "expected_separation_vs_opp",
        "hypothesis": "Receivers with higher expected separation vs opponent (NGS baseline + opponent defensive effect) outperform their line.",
        "predicted_direction": "+ for rec_yds and receptions",
        "data_source": "NGS avg_separation by receiver-week, opponent-adjusted",
    },
    {
        "feature_name": "shadow_cb_flag",
        "hypothesis": "WRs facing a shadow or top-graded CB underperform their projection and market line more than WRs not facing one.",
        "predicted_direction": "- for rec_yds (actual − projection < 0 when shadowed)",
        "data_source": "CB-on-WR proxy (PBP credited defenders, side overlap) — VALIDATED DEAD: 38.4% top-1 accuracy vs 37.5% random baseline (BDB 2023, 14K plays)",
    },
    {
        "feature_name": "primary_cb_prob",
        "hypothesis": "Higher certainty of primary CB matchup (more evidence, higher P(CB|WR)) produces more predictable outcomes — tighter distributions.",
        "predicted_direction": "reduces variance (σ) of rec_yds distribution",
        "data_source": "CB-on-WR proxy posterior probability — DEAD (proxy at chance, see shadow_cb_flag)",
    },
    {
        "feature_name": "neutral_pass_rate",
        "hypothesis": "Teams with higher neutral-situation pass rate produce more passing volume, increasing WR/TE target counts.",
        "predicted_direction": "+ for receptions, rec_yds, pass_yds",
        "data_source": "PBP down/distance/score filters",
    },
    {
        "feature_name": "opp_proe_allowed",
        "hypothesis": "Defenses that force opponents to pass more (higher PROE allowed) face more passing volume, inflating WR/TE props.",
        "predicted_direction": "+ for receptions, rec_yds",
        "data_source": "PBP xpass vs actual pass rate",
    },
]


def register_candidate_features():
    """Persist candidate feature hypotheses to DB."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for feat in _CANDIDATE_FEATURES:
        conn.execute("""
            INSERT OR REPLACE INTO nfl_candidate_features
            (feature_name, hypothesis, predicted_direction, data_source, status, created_at)
            VALUES (?, ?, ?, ?, 'candidate', ?)
        """, (feat["feature_name"], feat["hypothesis"],
              feat["predicted_direction"], feat["data_source"], now))
    conn.commit()
    conn.close()
    logger.info("[matchup] Registered %d candidate features", len(_CANDIDATE_FEATURES))


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: CV Pipeline Scope (no build)
# ══════════════════════════════════════════════════════════════════════════════

CV_SCOPE = """
## Computer Vision Matchup Pipeline — Scope Document

### Video Source Options
1. **NFL+** (subscription): Game replays, All-22, Skycam. Terms of Service
   Section 7.2 prohibits automated scraping or systematic downloading.
   Manual frame capture for personal analysis appears permissible under
   fair use but has not been tested. NOT suitable for automated pipeline.

2. **YouTube highlights** (free): 10-15 min condensed game highlights.
   Low coverage (not every play), inconsistent camera angles. Usable for
   proof-of-concept alignment classification only. Not all plays visible.

3. **NFL Game Pass International** (subscription, varies by region):
   Similar restrictions to NFL+. No automated access permitted.

4. **Big Data Bowl tracking data** (free, annual): Player x/y coordinates
   at 10Hz for ~120 games/year. No video needed — tracking data IS the
   ground truth for alignment. Available 2022-2024 via Kaggle.
   BEST source for validation.

### Pipeline Outline
1. **Snap-frame detection**: Identify pre-snap frame (ball set, players aligned).
   Input: video frame. Output: binary snap/no-snap classifier.
2. **Field registration**: Detect yard-line markings and hash marks to establish
   coordinate system. Homography from 4+ points → field coordinates.
3. **Jersey-number identity**: OCR on jersey numbers + team-color segmentation.
   Map to roster. Accuracy depends heavily on camera angle and resolution.
4. **Alignment classification**: For each WR-CB pair, classify:
   LCB/RCB/slot alignment. Based on field-registered x/y positions.
5. **Post-snap matchup** (optional): Track CB following WR through route.
   Much harder — requires multi-frame tracking with occlusion handling.

### Validation Plan
Compare CV alignment calls to Big Data Bowl tracking data on the same
2022-2024 plays. Metrics: alignment classification accuracy (%), positional
error in yards, matchup assignment accuracy vs tracking-derived truth.

### Effort Estimate
- Snap detection + field registration: 2-3 weeks (well-studied problem)
- Jersey OCR: 2-4 weeks (hardest part — low resolution, motion blur)
- Alignment classification: 1 week (straightforward from coordinates)
- Full pipeline with validation: 6-10 weeks total
- Expected accuracy: 75-85% alignment, 60-70% jersey ID (from literature)

### Recommendation
Skip video CV entirely. Big Data Bowl tracking data (2022-2024) is for
**training and validating** a CV-based alignment model (Phase 3a inventory),
not an in-season substitute for the PBP-based CB-on-WR proxy.

The PBP proxy runs in-season with live data. BDB tracking is the ground-truth
dataset for measuring the proxy's accuracy on historical games (2022-2024).
If proxy accuracy is insufficient after BDB validation, evaluate PFF matchup
data subscription before building a full CV pipeline.
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLI — Run for specific game or current week
# ══════════════════════════════════════════════════════════════════════════════

def _format_team_context(team: str, ctx: dict):
    """Format team context for terminal display."""
    lines = []

    def _fmt_ci(ci_tuple, fmt="pct"):
        """Format a CI tuple as string."""
        lo, hi = ci_tuple if ci_tuple else (None, None)
        if lo is None or hi is None:
            return ""
        if fmt == "pct":
            return f" CI [{lo:.1%}, {hi:.1%}]"
        return f" CI [{lo:+.3f}, {hi:+.3f}]"

    def _low_conf(n):
        return " *** LOW-CONFIDENCE ***" if n < _LOW_CONFIDENCE_N else ""

    for side in ["offense", "defense"]:
        d = ctx.get(side, {})
        if not d:
            continue
        lg = d.get("league_avg", {})

        def _lg_cmp(key):
            v = lg.get(key)
            if v is None:
                return ""
            if abs(v) < 1:
                if key.endswith("_rate"):
                    return f" [lg avg: {v:.1%}]"
                return f" [lg avg: {v:+.3f}]"
            return f" [lg avg: {v:.2f}]"

        n_pass = d.get("n_pass_plays", 0)
        n_rush = d.get("n_rush_plays", 0)
        n_drop = d.get("n_dropbacks", 0)

        lines.append(f"\n  {team} {side.upper()}:")
        lines.append(f"    Pass EPA/play: {d.get('pass_epa_play', 'N/A'):+.3f}" +
                     (f" (rank #{d.get('pass_epa_rank', '?')})" if d.get('pass_epa_rank') else "") +
                     f" N={n_pass}" +
                     _fmt_ci(d.get("pass_epa_ci"), "epa") +
                     _lg_cmp("pass_epa_play") +
                     _low_conf(n_pass))
        lines.append(f"    Rush EPA/play: {d.get('rush_epa_play', 'N/A'):+.3f}" +
                     (f" (rank #{d.get('rush_epa_rank', '?')})" if d.get('rush_epa_rank') else "") +
                     f" N={n_rush}" +
                     _fmt_ci(d.get("rush_epa_ci"), "epa") +
                     _lg_cmp("rush_epa_play") +
                     _low_conf(n_rush))
        lines.append(f"    Pass success:  {d.get('pass_success_rate', 0):.1%}" +
                     f" N={n_pass}" +
                     _fmt_ci(d.get("pass_success_ci")) +
                     _lg_cmp("pass_success_rate") +
                     _low_conf(n_pass))
        lines.append(f"    Rush success:  {d.get('rush_success_rate', 0):.1%}" +
                     f" N={n_rush}" +
                     _fmt_ci(d.get("rush_success_ci")) +
                     _lg_cmp("rush_success_rate") +
                     _low_conf(n_rush))
        if side == "defense":
            n_charted = d.get("blitz_n_charted", 0)
            if d.get("blitz_rate") is not None:
                lines.append(f"    Blitz rate:    {d['blitz_rate']:.1%}" +
                             (f" (rank #{d.get('blitz_rate_rank', '?')})" if d.get('blitz_rate_rank') else "") +
                             f" [FTN n_blitzers>=1, N={n_charted}]" +
                             _fmt_ci(d.get("blitz_rate_ci")) +
                             _lg_cmp("blitz_rate") +
                             _low_conf(n_charted))
            if d.get("heavy_rush_rate") is not None:
                lines.append(f"    5+ rushers:    {d['heavy_rush_rate']:.1%}" +
                             f" [n_pass_rushers>=5, N={n_charted}]" +
                             _fmt_ci(d.get("heavy_rush_ci")) +
                             _lg_cmp("heavy_rush_rate"))
            if d.get("sack_rate") is not None:
                lines.append(f"    Sack rate:     {d['sack_rate']:.1%}" +
                             (f" (rank #{d.get('sack_rate_rank', '?')})" if d.get('sack_rate_rank') else "") +
                             f" N={n_drop}" +
                             _fmt_ci(d.get("sack_rate_ci")) +
                             _lg_cmp("sack_rate"))
            if d.get("pressure_rate") is not None and not (isinstance(d["pressure_rate"], float) and np.isnan(d["pressure_rate"])):
                src = d.get("pressure_source", "unknown")
                src_label = "sack+hit proxy" if src == "sack_hit_proxy" else src
                lines.append(f"    Pressure rate: {d['pressure_rate']:.1%}" +
                             (f" (rank #{d.get('pressure_rank', '?')})" if d.get('pressure_rank') else "") +
                             f" N={n_drop} [source: {src_label}]" +
                             _fmt_ci(d.get("pressure_ci")) +
                             _lg_cmp("pressure_rate"))
        rz_trips = d.get("rz_trips", 0)
        if d.get("rz_td_rate") is not None:
            lines.append(f"    RZ TD/trip:    {d['rz_td_rate']:.1%}" +
                         (f" (rank #{d.get('rz_td_rank', '?')})" if side == "defense" and d.get('rz_td_rank') else "") +
                         f" [N={rz_trips} trips]" +
                         _fmt_ci(d.get("rz_td_ci")) +
                         _lg_cmp("rz_td_rate") +
                         _low_conf(rz_trips))
        lines.append(f"    Games:         {d.get('n_games', 0)}")
    return "\n".join(lines)


def _format_qb_blitz(qb_name: str, data: dict):
    """Format QB vs blitz splits for terminal display."""
    if not data:
        return f"\n  {qb_name}: no blitz data"

    lines = [f"\n  {qb_name} vs Blitz (source: {data.get('blitz_source', '?')}):",
             f"    Blitz rate faced: {data.get('blitz_rate_faced', 0):.1%}"]

    for label in ["blitz", "clean"]:
        d = data.get(label, {})
        if not d:
            continue
        n = d.get("n_dropbacks", 0)
        lc = " *** LOW-CONFIDENCE ***" if d.get("low_confidence") else ""
        epa_ci = d.get("epa_ci", (None, None))
        epa_ci_str = f" CI [{epa_ci[0]:+.3f}, {epa_ci[1]:+.3f}]" if epa_ci[0] is not None else ""
        comp_ci = d.get("comp_pct_ci", (None, None))
        comp_ci_str = f" CI [{comp_ci[0]:.1%}, {comp_ci[1]:.1%}]" if comp_ci[0] is not None else ""
        sack_ci = d.get("sack_rate_ci", (None, None))
        sack_ci_str = f" CI [{sack_ci[0]:.1%}, {sack_ci[1]:.1%}]" if sack_ci[0] is not None else ""
        lines.append(f"    {'BLITZ' if label == 'blitz' else 'CLEAN'} "
                     f"(N={n}):{lc}")
        lines.append(f"      EPA/drop:  {d.get('epa_per_dropback', 0):+.3f}{epa_ci_str}")
        lines.append(f"      Comp%:     {d.get('comp_pct', 0):.1%}{comp_ci_str}")
        lines.append(f"      aDOT:      {d.get('adot', 0):.1f}")
        lines.append(f"      Sack rate: {d.get('sack_rate', 0):.1%}{sack_ci_str}")
        if d.get("screen_rate") is not None:
            lines.append(f"      Screen:    {d['screen_rate']:.1%}")
        if d.get("pa_rate") is not None:
            lines.append(f"      Play-action: {d['pa_rate']:.1%}")
        if d.get("throwaway_rate") is not None:
            lines.append(f"      Throwaway: {d['throwaway_rate']:.1%}")

    # Target distribution
    td = data.get("target_distribution", {})
    for label in ["blitz", "clean"]:
        recs = td.get(label, [])
        if recs:
            lines.append(f"    Target dist ({label}):")
            for r in recs[:5]:
                lines.append(f"      {r['receiver_player_name']:<22} "
                             f"share={r['tgt_share']:.1%} "
                             f"aDOT={r.get('adot', 0):.1f} "
                             f"EPA={r.get('epa', 0):+.3f}")
    return "\n".join(lines)


def _format_separation(team: str, opp_team: str, sep_list: list):
    """Format separation estimates for terminal display."""
    if not sep_list:
        return f"\n  {team}: no NGS separation data"
    lines = [f"\n  {team} Receiver Separation vs {opp_team} (NGS, opponent-adjusted):"]
    lines.append(f"    {'Player':<22} {'Pos':>4} {'AvgSep':>7} {'OppEff':>7} "
                 f"{'ExpSep':>7} {'CI':>14} {'N_wk':>5} {'Opp_N':>5}")
    for r in sep_list[:10]:
        ci = f"[{r.get('expected_sep_ci_lo', 0):.2f}, {r.get('expected_sep_ci_hi', 0):.2f}]"
        lc = " *LC*" if r.get("low_confidence") else ""
        opp_ci = r.get("opp_effect_ci", (None, None))
        opp_ci_str = f" CI [{opp_ci[0]:+.3f}, {opp_ci[1]:+.3f}]" if opp_ci[0] is not None else ""
        lines.append(f"    {r['player_name']:<22} {r.get('position', ''):>4} "
                     f"{r.get('avg_separation', 0):>7.2f} "
                     f"{r.get('opp_effect', 0):>+7.3f} "
                     f"{r.get('expected_separation', 0):>7.2f} "
                     f"{ci:>14} {r.get('n_weeks', 0):>5} "
                     f"{r.get('opp_effect_n', 0):>5}{lc}")
        if opp_ci_str:
            lines.append(f"      opp effect for {r.get('position','?')}: "
                         f"{r.get('opp_effect', 0):+.3f}{opp_ci_str} "
                         f"[lg pos avg: {r.get('league_pos_avg', 0):.2f}]")
    return "\n".join(lines)


def _format_cb_proxy(team: str, proxy_list: list):
    """Format CB-on-WR proxy for terminal display."""
    if not proxy_list:
        return f"\n  {team}: no CB-on-WR evidence"
    lines = [f"\n  {team} CB-on-WR Matchup Proxy [INFORMATIONAL ONLY — 38.4% top-1 vs 37.5% random, BDB 2023]:"]
    total_ev = sum(w.get("total_evidence", 0) for w in proxy_list)
    if total_ev == 0:
        lines.append(f"    (side-overlap prior only — no direct h2h evidence)")
    lines.append(f"    {'WR':<22} {'Primary CB':<22} {'P(CB)':>6} "
                 f"{'Evid':>5} {'Overlap':>8} {'Shadow':>14}")
    for wr in proxy_list[:10]:
        if wr.get("shadow_risk"):
            shadow = "NOT PREDICTIVE"
        else:
            shadow = "  no"
        lines.append(f"    {wr['wr_name']:<22} "
                     f"{wr.get('primary_cb', 'unknown'):<22} "
                     f"{wr.get('primary_cb_prob', 0):>6.1%} "
                     f"{wr.get('total_evidence', 0):>5} "
                     f"{wr.get('primary_cb_overlap', 0):>8.3f} "
                     f"{shadow:>14}")
    return "\n".join(lines)


def _format_script(team: str, script: dict):
    """Format script/volume for terminal display."""
    if not script:
        return f"\n  {team}: no script data"
    lines = [f"\n  {team} Script/Volume:"]
    if script.get("neutral_pass_rate") is not None:
        n_neut = script.get("n_neutral_plays", 0)
        ci = script.get("neutral_pass_rate_ci", (None, None))
        ci_str = f" CI [{ci[0]:.1%}, {ci[1]:.1%}]" if ci[0] is not None else ""
        lc = " *** LOW-CONFIDENCE ***" if script.get("low_confidence_neutral") else ""
        lines.append(f"    Neutral pass rate: {script['neutral_pass_rate']:.1%} N={n_neut}{ci_str}{lc}")
    if script.get("proe") is not None:
        lines.append(f"    PROE:              {script['proe']:+.3f} N={script.get('proe_n', '?')}")
    if script.get("opp_proe_allowed") is not None:
        lines.append(f"    Opp PROE allowed:  {script['opp_proe_allowed']:+.3f} N={script.get('opp_proe_n', '?')}")
    lines.append(f"    Avg pass att:      {script.get('avg_pass_att', 0):.1f} "
                 f"(±{script.get('pass_att_std', 0):.1f})")
    lines.append(f"    Avg rush att:      {script.get('avg_rush_att', 0):.1f} "
                 f"(±{script.get('rush_att_std', 0):.1f})")
    lines.append(f"    Avg total plays:   {script.get('avg_total_plays', 0):.1f}")
    lines.append(f"    Games:             {script.get('n_games', 0)}")
    return "\n".join(lines)


def run_game(game_id: str, home_team: str, away_team: str,
             season: int, week: int,
             retrospective: bool = False):
    """Run matchup context for a single game with formatted output."""
    label = "RETROSPECTIVE" if retrospective else "PRE-GAME"
    print(f"\n{'='*80}")
    print(f"MATCHUP CONTEXT [{label}]: {away_team} @ {home_team} (Week {week})")
    print(f"{'='*80}")

    ctx = build_game_matchup_context(game_id, home_team, away_team,
                                     season, week,
                                     retrospective=retrospective)

    # Section 1
    print("\n--- SECTION 1: TEAM VERIFIED STATS ---")
    for side_label, team in [("home", home_team), ("away", away_team)]:
        team_ctx = ctx.get(f"{side_label}_team_context", {})
        print(_format_team_context(team, team_ctx))

    # Section 2
    print("\n--- SECTION 2: QB vs BLITZ ---")
    for side_label in ["home", "away"]:
        qb = ctx.get(f"{side_label}_qb_name", "Unknown")
        data = ctx.get(f"{side_label}_qb_vs_blitz", {})
        print(_format_qb_blitz(qb, data))

    # Section 3
    print("\n--- SECTION 3: SEPARATION ESTIMATES ---")
    for side_label, team, opp in [("home", home_team, away_team),
                                    ("away", away_team, home_team)]:
        sep = ctx.get(f"{side_label}_separation", [])
        print(_format_separation(team, opp, sep))

    # Section 4
    print("\n--- SECTION 4: CB-ON-WR PROXY ---")
    for side_label, team in [("home", home_team), ("away", away_team)]:
        proxy = ctx.get(f"{side_label}_cb_proxy", [])
        print(_format_cb_proxy(team, proxy))

    # Section 4b: Shadow log
    print("\n--- SECTION 4b: REPORTED SHADOWS ---")
    print(_format_shadow_log(home_team, away_team, season, week))

    # Section 5
    print("\n--- SECTION 5: SCRIPT / VOLUME ---")
    for side_label, team in [("home", home_team), ("away", away_team)]:
        script = ctx.get(f"{side_label}_script", {})
        print(_format_script(team, script))

    return ctx


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="NFL Matchup Context Module")
    parser.add_argument("--game", type=str, help="Game ID (e.g. 2026_08_MIN_DET)")
    parser.add_argument("--home", type=str, help="Home team")
    parser.add_argument("--away", type=str, help="Away team")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, help="Week number")
    parser.add_argument("--retrospective", action="store_true",
                        help="Run as retrospective analysis (game already played)")
    parser.add_argument("--register-features", action="store_true",
                        help="Register candidate features to DB")
    parser.add_argument("--validate-proxy", action="store_true",
                        help="Validate CB proxy accuracy on historical data")
    parser.add_argument("--cv-scope", action="store_true",
                        help="Print CV pipeline scope document")
    parser.add_argument("--add-shadow", nargs=6,
                        metavar=("DEF_TEAM", "CB_NAME", "WR_NAME", "OFF_TEAM",
                                 "SOURCE", "SOURCE_DATE"),
                        help="Add a reported shadow: DEF CB WR OFF SOURCE DATE")
    parser.add_argument("--confirm-shadow", nargs=3,
                        metavar=("DEF_TEAM", "CB_NAME", "WR_NAME"),
                        help="Confirm a shadow was observed post-game")
    parser.add_argument("--deny-shadow", nargs=3,
                        metavar=("DEF_TEAM", "CB_NAME", "WR_NAME"),
                        help="Mark a shadow as NOT confirmed post-game")
    parser.add_argument("--show-shadows", action="store_true",
                        help="Show all reported shadows for the given week")
    parser.add_argument("--source-accuracy", action="store_true",
                        help="Show shadow prediction accuracy by source")
    parser.add_argument("--freeze-elite-cbs", action="store_true",
                        help="Freeze the elite CB list from BDB 2023 metrics")
    parser.add_argument("--show-elite-cbs", action="store_true",
                        help="Show the frozen elite CB list")
    parser.add_argument("--elite-shadow-test", action="store_true",
                        help="Evaluate the elite-CB shadow forward test")

    args = parser.parse_args()

    if args.add_shadow:
        def_team, cb, wr, off_team, source, date = args.add_shadow
        if not args.week:
            parser.error("--week is required for --add-shadow")
        add_shadow_report(args.season, args.week, def_team, cb, wr, off_team,
                          source, date)
        print(f"Logged: {def_team} {cb} shadows {off_team} {wr} "
              f"(W{args.week}, {source} {date})")
        sys.exit(0)

    if args.confirm_shadow:
        def_team, cb, wr = args.confirm_shadow
        if not args.week:
            parser.error("--week is required for --confirm-shadow")
        confirm_shadow(args.season, args.week, def_team, cb, wr, confirmed=True)
        print(f"Confirmed: {def_team} {cb} did shadow {wr} (W{args.week})")
        sys.exit(0)

    if args.deny_shadow:
        def_team, cb, wr = args.deny_shadow
        if not args.week:
            parser.error("--week is required for --deny-shadow")
        confirm_shadow(args.season, args.week, def_team, cb, wr, confirmed=False)
        print(f"Not confirmed: {def_team} {cb} did NOT shadow {wr} (W{args.week})")
        sys.exit(0)

    if args.show_shadows:
        if not args.week:
            parser.error("--week is required for --show-shadows")
        reports = get_shadow_reports(args.season, args.week)
        if not reports:
            print(f"No reported shadows for {args.season} W{args.week}")
        else:
            print(f"\nReported Shadows — {args.season} W{args.week}:")
            print(f"  {'Defense':<6} {'CB':<22} {'shadows':<8} {'WR':<22} "
                  f"{'Source':<14} {'Date':<12} {'Status':<14}")
            for r in reports:
                status = r.get('confidence', 'reported')
                print(f"  {r['def_team']:<6} {r['cb_name']:<22} {'shadows':<8} "
                      f"{r['wr_name']:<22} {r['source']:<14} "
                      f"{r.get('source_date', '—'):<12} {status:<14}")
                if r.get('notes'):
                    print(f"         ↳ {r['notes']}")
        sys.exit(0)

    if args.source_accuracy:
        acc = shadow_source_accuracy(args.season)
        if not acc:
            print(f"No confirmed/denied shadows for {args.season}")
        else:
            print(f"\nShadow Source Accuracy — {args.season}:")
            print(f"  {'Source':<20} {'Confirmed':>10} {'Denied':>10} "
                  f"{'Total':>10} {'Accuracy':>10}")
            for src, data in sorted(acc.items()):
                print(f"  {src:<20} {data['confirmed']:>10} "
                      f"{data['not_confirmed']:>10} {data['total']:>10} "
                      f"{data.get('accuracy', 0):>10.1%}")
        sys.exit(0)

    if args.freeze_elite_cbs:
        result = freeze_elite_cb_list()
        print(f"\nFroze {result['n_elite']} elite CBs "
              f"(EPA/tgt <= {result['threshold']}, BDB 2023)")
        print(f"Frozen at: {result['frozen_at']}")
        print(f"\n{'CB Name':<24} {'Team':<6} {'EPA/tgt':>8} {'Snaps':>6}")
        print(f"{'─'*24} {'─'*6} {'─'*8} {'─'*6}")
        for cb in result['cbs']:
            print(f"{cb['closest_cb_name']:<24} {cb['defensive_team']:<6} "
                  f"{cb['epa_allowed']:>+8.3f} {cb['snaps']:>6}")
        sys.exit(0)

    if args.show_elite_cbs:
        cbs = get_elite_cb_list()
        if not cbs:
            print("No elite CB list frozen. Run --freeze-elite-cbs first.")
        else:
            print(f"\nFrozen Elite CB List ({len(cbs)} CBs):")
            print(f"  Source: {cbs[0].get('source', '?')}, "
                  f"Frozen: {cbs[0].get('frozen_at', '?')}")
            print(f"\n  {'CB Name':<24} {'Team':<6} {'EPA/tgt':>8} "
                  f"{'Comp%':>6} {'Yds/tgt':>8} {'Snaps':>6}")
            print(f"  {'─'*24} {'─'*6} {'─'*8} {'─'*6} {'─'*8} {'─'*6}")
            for cb in cbs:
                print(f"  {cb['cb_name']:<24} {cb['team_2023']:<6} "
                      f"{cb['epa_per_target']:>+8.3f} "
                      f"{cb.get('comp_pct', 0):>6.1%} "
                      f"{cb.get('yds_per_target', 0):>8.1f} "
                      f"{cb['coverage_snaps']:>6}")
        sys.exit(0)

    if args.elite_shadow_test:
        result = evaluate_elite_cb_shadow_test()
        print(f"\n{'='*60}")
        print("ELITE-CB SHADOW FORWARD TEST")
        print(f"{'='*60}")
        print(f"N observations: {result['n']}")
        if result['n'] == 0:
            print(result['status'])
        else:
            print(f"  Confirmed: {result['n_confirmed']}, "
                  f"Not confirmed: {result['n_not_confirmed']}, "
                  f"Pending: {result['n_reported']}")
            print(f"\nWR1 rec yds:     {result['mean_rec_yds']:.1f}")
            print(f"Market line:     {result['mean_market_line']:.1f}")
            print(f"v1.2 projection: {result.get('mean_diff_vs_v12', 0) + result['mean_rec_yds']:.1f}")
            print(f"\nDiff vs market:  {result['mean_diff_vs_market']:+.1f}")
            print(f"Diff vs v1.2:    {result['mean_diff_vs_v12']:+.1f}")
            print(f"Targets:         {result['mean_targets']:.1f}")
            print(f"Yds/target:      {result['mean_yds_per_tgt']:.1f}")
            if result.get('ci_vs_market'):
                ci = result['ci_vs_market']
                print(f"\n95% CI vs market: [{ci[0]:+.1f}, {ci[2]:+.1f}]")
                print(f"Market prices shadow: {result['market_prices_shadow']}")
            else:
                print(f"\nInsufficient N for bootstrap CI (need >= 5)")
        sys.exit(0)

    if args.register_features:
        register_candidate_features()
        print(f"Registered {len(_CANDIDATE_FEATURES)} candidate features")
        sys.exit(0)

    if args.cv_scope:
        print(CV_SCOPE)
        sys.exit(0)

    if args.validate_proxy:
        print("Validating CB-on-WR proxy on 2023-2025 data...")
        pbp = _load_pbp([2023, 2024, 2025])
        rosters = _load_rosters([2023, 2024, 2025])
        result = validate_cb_proxy_accuracy(pbp, rosters, [2023, 2024, 2025])
        print(f"\nValidation Results:")
        print(f"  Total targets with PBU/INT evidence: {result['total_targets_with_evidence']}")
        print(f"  Top-1 CB accuracy: {result.get('top1_accuracy', 0):.1%}")
        print(f"  Top-2 CB accuracy: {result.get('top2_accuracy', 0):.1%}")
        print(f"  By evidence volume:")
        for bucket, data in sorted(result.get("by_evidence_volume", {}).items()):
            acc = data["match"] / data["total"] if data["total"] > 0 else 0
            print(f"    {bucket}: {data['match']}/{data['total']} = {acc:.1%}")
        sys.exit(0)

    if not args.game or not args.home or not args.away or not args.week:
        parser.error("--game, --home, --away, and --week are required")

    run_game(args.game, args.home, args.away, args.season, args.week,
             retrospective=args.retrospective)
