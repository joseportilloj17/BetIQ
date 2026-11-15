"""
nfl_game_sim.py — Phase 2b-3: Drive-level NFL game simulation with pace.

Simulates games play-by-play:
  - Possessions alternate on a shared 3600-second clock
  - Play call (pass/rush) conditional on situation (score×time×down/distance)
  - Pace varies by situation (neutral, trailing late, leading late, two-minute)
  - Team-specific profiles: pace, pass rate, efficiency from rolling PBP history
  - Backup QB modeled via efficiency (YPA, comp%, sack%, scramble%), not volume
  - Player allocations from the corrected share model (2b-2)
  - Per-player joint draws with Σ rec_yds = QB pass_yds identity

Known limitations:
  - No overtime: regulation ends at 0:00.  OT stats count toward player
    props (~5-6% of NFL games reach OT); this slightly under-projects
    yardage/TD totals for games near pick'em.
  - No penalty model: sim pts/yard is ~93% of actual (penalties extend
    drives via automatic first downs).  Market anchoring compensates for
    scoring but pushes yardage ~1.4-1.8% above actual as a side effect.
  - No special-teams scoring (blocked punts, kick returns for TD).
  - Defensive TDs modeled as post-hoc probability on turnovers, not as
    return-yardage drives — correct for team scoring, but no return
    yardage in box-score stats.

Usage:
    python nfl_game_sim.py build-profiles [--seasons 2023,2024,2025]
    python nfl_game_sim.py validate [--seasons 2023,2024,2025]
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from math import sqrt

import numpy as np

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_NFL_DB = os.path.join(_DATA_DIR, "nfl.db")
_OUTCOME_TABLES_PATH = os.path.join(_DATA_DIR, "sim_outcome_tables.json")

# ── Load empirical outcome tables (down × distance × field-position) ──
import json as _json
_OUTCOME_TABLES = {}
_FOURTH_DOWN_RATES = {}
try:
    with open(_OUTCOME_TABLES_PATH) as _f:
        _raw = _json.load(_f)
        _OUTCOME_TABLES = _raw.get("outcome_tables", {})
        _FOURTH_DOWN_RATES = _raw.get("fourth_down_rates", {})
except FileNotFoundError:
    logger.warning("sim_outcome_tables.json not found; using flat constants")

# ══════════════════════════════════════════════════════════════════════════════
# Constants — league-wide empirical baselines (2023-2025 PBP)
# ══════════════════════════════════════════════════════════════════════════════

_GAME_SECONDS = 3600

# Situation definitions
# neutral: Q1-Q3, |score_diff| <= 8
# trailing_late: Q4, score_diff <= -4
# leading_late: Q4, score_diff >= 4
# two_min: game_seconds_remaining <= 120
# big_trail: Q1-Q3, score_diff < -8
# big_lead: Q1-Q3, score_diff > 8

# League-average dropback rate by situation (play_type='pass' + scrambles).
# In the sim, a "pass" decision leads to sack/scramble/pass; scrambles are
# play_type='run' in PBP, so we add them to get the correct sim dropback rate.
_LG_PASS_RATE = {
    "neutral": 0.596,
    "trailing_late": 0.766,
    "leading_late": 0.399,
    "two_min": 0.723,
    "big_trail": 0.646,
    "big_lead": 0.571,
}

# League-average pace: seconds consumed per running-clock play by situation.
# Measured from PBP 2023-2024 snap_gap on running-clock plays only
# (rush, completion, sack, scramble).  Clock-stop events (incomplete,
# OOB, timeout) are handled separately in the sim.
_LG_PACE = {
    "neutral": 35.5,
    "trailing_late": 31.0,
    "leading_late": 35.4,
    "two_min": 17.4,
    "big_trail": 33.8,
    "big_lead": 35.0,
}

# League-average no-huddle rate by situation
_LG_NO_HUDDLE = {
    "neutral": 0.084,
    "trailing_late": 0.238,
    "leading_late": 0.034,
    "two_min": 0.173,
    "big_trail": 0.150,
    "big_lead": 0.040,
}

# Per-dropback outcome rates (league average, 2023-2024 PBP).
# Comp rate set slightly above league mean to compensate for
# RZ compression and 3rd-down adjustments on passing downs.
_LG_COMP_RATE = 0.660       # pre-adjustment; league mean = 0.652
_LG_SACK_RATE = 0.066       # sacks / dropbacks
_LG_SCRAMBLE_RATE = 0.054   # scrambles / dropbacks
_LG_INT_RATE = 0.021        # interceptions / non-sack non-scramble pass attempts

# Yards gained distributions (empirical)
_LG_YPA_COMP = 11.6         # yards per completion (pre-RZ-adjustment; gross Y/C ~11.1)
_LG_YPA_COMP_SD = 10.5
_LG_YPA_INC = 0.0           # incomplete = 0 yards
_LG_SACK_YARDS = -6.5       # mean sack yardage
_LG_SACK_YARDS_SD = 3.0
_LG_SCRAMBLE_YARDS = 7.4    # mean scramble yards
_LG_SCRAMBLE_YARDS_SD = 6.5
_LG_YPC = 4.24              # yards per designed carry (excl scrambles; 2023-2025 PBP)
_LG_YPC_SD = 6.14
_LG_FUMBLE_RATE = 0.015     # fumbles lost per play

# Drive structure
_LG_DRIVES_PER_GAME = 10.4  # per team
_LG_PLAYS_PER_DRIVE = 5.9

# Change-of-possession dead time.
# The _PACE_TABLE values are PBP snap-to-next-snap for running-clock
# plays; the last play of each drive has no "next snap" within the
# drive, but its pace already covers play execution (~5s), and the
# next drive's first-play snap_gap covers the transition.  COP adds
# only the small residual not captured by either snap_gap:
# kickoff/punt return + ball spotting.
_COP_AFTER_TD   = 2
_COP_AFTER_FG   = 2
_COP_AFTER_PUNT = 2
_COP_AFTER_TO   = 2
_COP_AFTER_TOD  = 2
_COP_AFTER_FGMISS = 2

# ── Penalty model (PBP 2023-2024) ──────────────────────────────────
# Auto-first-down defensive penalties per play (DPI, def holding,
# roughing the passer, illegal contact).
# PBP: 1668 accepted / 70650 plays = 0.0236, minus ~15% declined ≈ 0.020.
_DEF_AUTO_FD_RATE = 0.020
# Defensive 5-yard penalties per play (offsides, encroachment, NZI).
# NOT automatic first downs — but convert when penalty yds ≥ distance.
# PBP: 594 / 70650 = 0.0084, ~5% declined ≈ 0.008.
_DEF_5YD_PENALTY_RATE = 0.008
# Offensive penalties per play (holding, false start, delay of game).
# PBP: ~0.211 per drive / 7.58 plays per drive.
_OFF_PENALTY_RATE = 0.028

# Starting field position after kickoff
_KICKOFF_MEAN_YL = 73       # yardline_100 (0=opponent end zone)
_KICKOFF_SD_YL = 8

# Punt
_PUNT_DISTANCE_MEAN = 43
_PUNT_DISTANCE_SD = 8
_PUNT_RETURN_MEAN = 8
_PUNT_RETURN_SD = 6

# Field goal success rate by distance
_FG_SUCCESS = {
    20: 0.98, 25: 0.97, 30: 0.95, 35: 0.92, 40: 0.88,
    45: 0.82, 50: 0.72, 55: 0.58, 60: 0.35,
}

# Extra point / two-point
_XP_RATE = 0.94
_TWO_PT_RATE = 0.48

# TD rate: per play when inside opponent's 10
_RED_ZONE_TD_PER_PLAY = 0.22  # approximate

# Game-level random effects: per-draw efficiency variance.
# Represents QB form, game-plan fit, defensive matchup luck — everything
# that makes one team's offense consistently better/worse IN THIS GAME
# beyond play-level noise.  Estimated from game-to-game residual variance
# of team EPA/play after controlling for team talent.
_PASS_EFF_SD = 0.12   # total pass-eff game-level SD (empirical: 0.119).
                       # Decomposed below into _SHARED_GAME_EFF_SD + _INDEP_PASS_EFF_SD.
_RUSH_EFF_SD = 0.25   # SD of rush-efficiency game-level shift (independent of pass)
_RUSH_ANCHOR_FRAC = 0.15  # fraction of symmetric anchor applied to rush (rest is noise)
_PASS_VOL_SD = 0.07   # SD of game-level pass rate shift (additive on pass rate).
                       # Captures opponent run defense, game-plan tendencies, injuries.
                       # Empirical: 0.073 (residual after team identity + game state).

# Shared game-environment factors.
# Replace the old _VOL_CROSS Cholesky coupling with two principled
# components that create cross-team correlation through real mechanisms:
#
# 1. Pass efficiency: weather, surface, referee tendencies affect both
#    teams' Y/A in the same game.  Shared draw + independent draw.
#    Variance preserved: 0.035² + 0.1148² ≈ 0.12² (total pass eff var).
#    Measured: r(A_pass_yds, B_pass_yds) ≈ +0.17 (NFL PBP).
_SHARED_GAME_EFF_SD = 0.035
_INDEP_PASS_EFF_SD = 0.1148   # sqrt(0.12² - 0.035²) preserves total variance
#
# 2. Game pace: weather delays, referee style, broadcast cadence,
#    overall game flow make both teams play faster or slower together.
#    Multiplicative on seconds/play; 5.5% SD on pace shifts total
#    plays by ~3.4 per team.
#    Measured: r(A_plays, B_plays) ≈ -0.51 (NFL PBP, vs -0.74 without).
_SHARED_PACE_SD = 0.055

# ── Anchoring dampening ─────────────────────────────────────────────
# Set by _calibrate(); initialized to last known values so the module
# works without calling calibrate at import time.
_SYM_DAMP = 0.779     # symmetric (total-driven) dampening
_ASYM_DAMP = 0.324    # asymmetric (spread-driven) dampening

# Shrinkage for team profiles
_PROFILE_SHRINK_K = 4  # games worth of league prior
_PROFILE_ROLLING_GAMES = 8

# ── Fine-grained play-call tables (2023-2025 PBP) ────────────────────
# Pass rate and pace by (score_bucket, time_bucket).
# Score buckets: trail17/trail10/trail4/trail1/even/lead1/lead4/lead10/lead17
# Time buckets: q1_q3 (>900s) / q4_gt5 (301-900s) / q4_lt5 (0-300s)

_PASS_RATE_TABLE = {
    ("trail17", "q1_q3"): 0.681, ("trail17", "q4_gt5"): 0.806,
    ("trail17", "q4_lt5"): 0.741,
    ("trail10", "q1_q3"): 0.638, ("trail10", "q4_gt5"): 0.779,
    ("trail10", "q4_lt5"): 0.946,
    ("trail4",  "q1_q3"): 0.608, ("trail4",  "q4_gt5"): 0.624,
    ("trail4",  "q4_lt5"): 0.839,
    ("trail1",  "q1_q3"): 0.597, ("trail1",  "q4_gt5"): 0.603,
    ("trail1",  "q4_lt5"): 0.783,
    ("even",    "q1_q3"): 0.582, ("even",    "q4_gt5"): 0.588,
    ("even",    "q4_lt5"): 0.653,
    ("lead1",   "q1_q3"): 0.587, ("lead1",   "q4_gt5"): 0.542,
    ("lead1",   "q4_lt5"): 0.357,
    ("lead4",   "q1_q3"): 0.605, ("lead4",   "q4_gt5"): 0.532,
    ("lead4",   "q4_lt5"): 0.293,
    ("lead10",  "q1_q3"): 0.586, ("lead10",  "q4_gt5"): 0.448,
    ("lead10",  "q4_lt5"): 0.204,
    ("lead17",  "q1_q3"): 0.532, ("lead17",  "q4_gt5"): 0.356,
    ("lead17",  "q4_lt5"): 0.162,
}

# Per-situation pace table: seconds consumed per running-clock play.
# Measured from PBP 2023-2024 snap_gap on running-clock plays only
# (rush, completion, sack, scramble).  Previous values included
# incomplete passes in the denominator, deflating every cell by 4-9s.
_PACE_TABLE = {
    ("trail17", "q1_q3"): 32.8, ("trail17", "q4_gt5"): 31.5,
    ("trail17", "q4_lt5"): 24.5,
    ("trail10", "q1_q3"): 34.4, ("trail10", "q4_gt5"): 33.1,
    ("trail10", "q4_lt5"): 19.3,
    ("trail4",  "q1_q3"): 34.9, ("trail4",  "q4_gt5"): 36.8,
    ("trail4",  "q4_lt5"): 20.3,
    ("trail1",  "q1_q3"): 35.4, ("trail1",  "q4_gt5"): 36.6,
    ("trail1",  "q4_lt5"): 21.3,
    ("even",    "q1_q3"): 36.8, ("even",    "q4_gt5"): 38.0,
    ("even",    "q4_lt5"): 23.4,
    ("lead1",   "q1_q3"): 34.7, ("lead1",   "q4_gt5"): 38.6,
    ("lead1",   "q4_lt5"): 22.7,
    ("lead4",   "q1_q3"): 34.9, ("lead4",   "q4_gt5"): 38.3,
    ("lead4",   "q4_lt5"): 20.6,
    ("lead10",  "q1_q3"): 34.4, ("lead10",  "q4_gt5"): 39.1,
    ("lead10",  "q4_lt5"): 23.2,
    ("lead17",  "q1_q3"): 36.3, ("lead17",  "q4_gt5"): 40.0,
    ("lead17",  "q4_lt5"): 33.6,
}


def _score_bucket(score_diff: float) -> str:
    """Classify score differential into pass-rate table bucket."""
    if score_diff <= -17:
        return "trail17"
    if score_diff <= -10:
        return "trail10"
    if score_diff <= -4:
        return "trail4"
    if score_diff <= -1:
        return "trail1"
    if score_diff == 0:
        return "even"
    if score_diff <= 3:
        return "lead1"
    if score_diff <= 9:
        return "lead4"
    if score_diff <= 16:
        return "lead10"
    return "lead17"


def _time_bucket(game_secs: float) -> str:
    """Classify game seconds remaining into time bucket."""
    if game_secs > 900:
        return "q1_q3"
    if game_secs > 300:
        return "q4_gt5"
    return "q4_lt5"

# QB tier priors for backup efficiency
_BACKUP_PRIORS = {
    "comp_rate": 0.60,
    "ypa_comp": 10.0,
    "sack_rate": 0.08,
    "scramble_rate": 0.04,
    "int_rate": 0.035,
}


# ══════════════════════════════════════════════════════════════════════════════
# Team profile extraction from PBP
# ══════════════════════════════════════════════════════════════════════════════

def _classify_situation(score_diff: float, qtr: int,
                        game_secs: float) -> str:
    """Classify game situation for play-calling/pace lookup."""
    if game_secs <= 120:
        return "two_min"
    if qtr <= 3:
        if abs(score_diff) <= 8:
            return "neutral"
        return "big_lead" if score_diff > 0 else "big_trail"
    # Q4
    if score_diff <= -4:
        return "trailing_late"
    if score_diff >= 4:
        return "leading_late"
    return "neutral"


def build_team_profiles(seasons: list[int] | None = None,
                        as_of_week: int | None = None) -> dict:
    """Build per-team simulation profiles from PBP.

    Returns: {team: {
        "pace": {situation: seconds_per_play},
        "pass_rate": {situation: rate},
        "no_huddle": {situation: rate},
        "comp_rate": float,
        "ypa_comp": float,  # yards per completion
        "ypa_comp_sd": float,
        "sack_rate": float,
        "scramble_rate": float,
        "int_rate": float,
        "ypc": float,
        "ypc_sd": float,
        "fumble_rate": float,
        "n_games": int,
    }}
    """
    import nfl_data_py as nfl

    if seasons is None:
        seasons = [2023, 2024, 2025]

    pbp = nfl.import_pbp_data(seasons)
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # Classify situations
    plays["situation"] = plays.apply(
        lambda r: _classify_situation(
            r.get("score_differential", 0) or 0,
            int(r.get("qtr", 1) or 1),
            r.get("game_seconds_remaining", 1800) or 1800),
        axis=1)

    # Snap gap (pace) within drives
    plays = plays.sort_values(["game_id", "game_seconds_remaining"],
                              ascending=[True, False])
    plays["next_gsr"] = plays.groupby(
        ["game_id", "posteam", "fixed_drive"])["game_seconds_remaining"].shift(-1)
    plays["snap_gap"] = plays["game_seconds_remaining"] - plays["next_gsr"]
    plays.loc[(plays["snap_gap"] <= 0) | (plays["snap_gap"] > 55),
              "snap_gap"] = np.nan

    # Dropback-level fields
    plays["is_dropback"] = ((plays["pass_attempt"] == 1) |
                            (plays["qb_scramble"] == 1) |
                            (plays["sack"] == 1)).astype(float)
    plays["is_completion"] = (
        (plays["pass_attempt"] == 1) & (plays["complete_pass"] == 1)
    ).astype(float)
    plays["comp_yards"] = np.where(
        plays["is_completion"] == 1, plays["yards_gained"], np.nan)

    teams = sorted(plays["posteam"].dropna().unique())
    profiles = {}

    for team in teams:
        tp = plays[plays["posteam"] == team].copy()
        n_games = tp["game_id"].nunique()

        # If as_of_week specified, filter to rolling window
        if as_of_week is not None:
            # Take most recent _PROFILE_ROLLING_GAMES games
            game_ids = tp.drop_duplicates("game_id").sort_values(
                ["game_id"], ascending=False)["game_id"].values
            recent_ids = set(game_ids[:_PROFILE_ROLLING_GAMES])
            tp = tp[tp["game_id"].isin(recent_ids)]
            n_games = len(recent_ids)

        # Pace by situation
        pace = {}
        for sit in _LG_PACE:
            sit_plays = tp[tp["situation"] == sit]
            gaps = sit_plays["snap_gap"].dropna()
            if len(gaps) >= 20:
                raw = gaps.mean()
                pace[sit] = _shrink(raw, _LG_PACE[sit],
                                    len(gaps), _PROFILE_SHRINK_K * 20)
            else:
                pace[sit] = _LG_PACE[sit]

        # Pass rate by situation: use is_dropback (pass_attempt + scrambles)
        # to match sim where scrambles emerge from "pass" decisions.
        pass_rate = {}
        for sit in _LG_PASS_RATE:
            sit_plays = tp[tp["situation"] == sit]
            if len(sit_plays) >= 20:
                raw = sit_plays["is_dropback"].mean()
                pass_rate[sit] = _shrink(raw, _LG_PASS_RATE[sit],
                                         len(sit_plays),
                                         _PROFILE_SHRINK_K * 20)
            else:
                pass_rate[sit] = _LG_PASS_RATE[sit]

        # No-huddle rate by situation
        no_huddle = {}
        for sit in _LG_NO_HUDDLE:
            sit_plays = tp[tp["situation"] == sit]
            if len(sit_plays) >= 20:
                raw = sit_plays["no_huddle"].mean()
                no_huddle[sit] = _shrink(raw, _LG_NO_HUDDLE[sit],
                                          len(sit_plays),
                                          _PROFILE_SHRINK_K * 20)
            else:
                no_huddle[sit] = _LG_NO_HUDDLE[sit]

        # Efficiency metrics
        dropbacks = tp[tp["is_dropback"] == 1]
        pass_atts = tp[tp["pass_attempt"] == 1]
        non_sack_pass = pass_atts[pass_atts["sack"] != 1]
        completions = tp[tp["is_completion"] == 1]
        rushes = tp[tp["rush_attempt"] == 1]

        n_db = len(dropbacks)
        n_pa = len(pass_atts)
        n_ns = len(non_sack_pass)
        n_rush = len(rushes)

        comp_rate = _shrink(
            non_sack_pass["complete_pass"].mean() if n_ns > 0 else _LG_COMP_RATE,
            _LG_COMP_RATE, n_ns, _PROFILE_SHRINK_K * 30)

        ypa_comp_vals = completions["yards_gained"].dropna()
        ypa_comp = _shrink(
            ypa_comp_vals.mean() if len(ypa_comp_vals) > 0 else _LG_YPA_COMP,
            _LG_YPA_COMP, len(ypa_comp_vals), _PROFILE_SHRINK_K * 30)
        ypa_comp_sd = (ypa_comp_vals.std()
                       if len(ypa_comp_vals) > 10 else _LG_YPA_COMP_SD)

        sack_rate = _shrink(
            dropbacks["sack"].mean() if n_db > 0 else _LG_SACK_RATE,
            _LG_SACK_RATE, n_db, _PROFILE_SHRINK_K * 30)

        scramble_rate = _shrink(
            dropbacks["qb_scramble"].mean() if n_db > 0 else _LG_SCRAMBLE_RATE,
            _LG_SCRAMBLE_RATE, n_db, _PROFILE_SHRINK_K * 30)

        int_rate = _shrink(
            pass_atts["interception"].mean() if n_pa > 0 else _LG_INT_RATE,
            _LG_INT_RATE, n_pa, _PROFILE_SHRINK_K * 30)

        ypc_vals = rushes["yards_gained"].dropna()
        ypc = _shrink(
            ypc_vals.mean() if len(ypc_vals) > 0 else _LG_YPC,
            _LG_YPC, len(ypc_vals), _PROFILE_SHRINK_K * 30)
        ypc_sd = ypc_vals.std() if len(ypc_vals) > 10 else _LG_YPC_SD

        fumble_rate = _shrink(
            tp["fumble_lost"].mean() if len(tp) > 0 else _LG_FUMBLE_RATE,
            _LG_FUMBLE_RATE, len(tp), _PROFILE_SHRINK_K * 50)

        # PROE: Pass Rate Over Expected — team's neutral-state pass tendency
        # relative to league average.  Stable trait (~0.04 SD across teams),
        # used as additive team adjustment across all game states.
        proe = (pass_rate.get("neutral", _LG_PASS_RATE.get("neutral", 0.58))
                - _LG_PASS_RATE.get("neutral", 0.58))

        profiles[team] = {
            "pace": pace,
            "pass_rate": pass_rate,
            "no_huddle": no_huddle,
            "comp_rate": comp_rate,
            "ypa_comp": ypa_comp,
            "ypa_comp_sd": ypa_comp_sd,
            "sack_rate": sack_rate,
            "scramble_rate": scramble_rate,
            "int_rate": int_rate,
            "ypc": ypc,
            "ypc_sd": ypc_sd,
            "fumble_rate": fumble_rate,
            "proe": proe,
            "n_games": n_games,
        }

    return profiles


def _shrink(raw: float, prior: float, n: int, k: float) -> float:
    """Bayesian shrinkage: (n * raw + k * prior) / (n + k)."""
    return (n * raw + k * prior) / (n + k)


def _build_profiles_from_plays(plays_df, teams: list | None = None) -> dict:
    """Build team profiles from a pre-processed plays DataFrame.

    The DataFrame must already have columns: situation, snap_gap, is_dropback,
    is_completion, comp_yards, no_huddle. Used by validate_sim for efficient
    walk-forward profile building.
    """
    if teams is None:
        teams = sorted(plays_df["posteam"].dropna().unique())

    profiles = {}
    for team in teams:
        tp = plays_df[plays_df["posteam"] == team].copy()
        n_games = tp["game_id"].nunique()
        if n_games < 2:
            continue

        # Take most recent _PROFILE_ROLLING_GAMES games
        game_ids = tp.drop_duplicates("game_id").sort_values(
            ["game_id"], ascending=False)["game_id"].values
        recent_ids = set(game_ids[:_PROFILE_ROLLING_GAMES])
        tp = tp[tp["game_id"].isin(recent_ids)]
        n_games = len(recent_ids)

        # Pace by situation
        pace = {}
        for sit in _LG_PACE:
            sit_plays = tp[tp["situation"] == sit]
            gaps = sit_plays["snap_gap"].dropna()
            if len(gaps) >= 20:
                pace[sit] = _shrink(gaps.mean(), _LG_PACE[sit],
                                    len(gaps), _PROFILE_SHRINK_K * 20)
            else:
                pace[sit] = _LG_PACE[sit]

        # Pass rate by situation (using is_dropback, includes scrambles)
        pass_rate = {}
        for sit in _LG_PASS_RATE:
            sit_plays = tp[tp["situation"] == sit]
            if len(sit_plays) >= 20:
                raw = sit_plays["is_dropback"].mean()
                pass_rate[sit] = _shrink(raw, _LG_PASS_RATE[sit],
                                         len(sit_plays),
                                         _PROFILE_SHRINK_K * 20)
            else:
                pass_rate[sit] = _LG_PASS_RATE[sit]

        # No-huddle rate
        no_huddle = {}
        for sit in _LG_NO_HUDDLE:
            sit_plays = tp[tp["situation"] == sit]
            if len(sit_plays) >= 20:
                raw = sit_plays["no_huddle"].mean()
                no_huddle[sit] = _shrink(raw, _LG_NO_HUDDLE[sit],
                                          len(sit_plays),
                                          _PROFILE_SHRINK_K * 20)
            else:
                no_huddle[sit] = _LG_NO_HUDDLE[sit]

        # Efficiency
        dropbacks = tp[tp["is_dropback"] == 1]
        pass_atts = tp[tp["pass_attempt"] == 1]
        non_sack = pass_atts[pass_atts["sack"] != 1]
        completions = tp[tp["is_completion"] == 1]
        rushes = tp[tp["rush_attempt"] == 1]

        n_db = len(dropbacks)
        n_ns = len(non_sack)
        n_rush = len(rushes)

        comp_rate = _shrink(
            non_sack["complete_pass"].mean() if n_ns > 0 else _LG_COMP_RATE,
            _LG_COMP_RATE, n_ns, _PROFILE_SHRINK_K * 30)

        ypa_vals = completions["yards_gained"].dropna()
        ypa_comp = _shrink(
            ypa_vals.mean() if len(ypa_vals) > 0 else _LG_YPA_COMP,
            _LG_YPA_COMP, len(ypa_vals), _PROFILE_SHRINK_K * 30)
        ypa_comp_sd = ypa_vals.std() if len(ypa_vals) > 10 else _LG_YPA_COMP_SD

        sack_rate = _shrink(
            dropbacks["sack"].mean() if n_db > 0 else _LG_SACK_RATE,
            _LG_SACK_RATE, n_db, _PROFILE_SHRINK_K * 30)

        scramble_rate = _shrink(
            dropbacks["qb_scramble"].mean() if n_db > 0 else _LG_SCRAMBLE_RATE,
            _LG_SCRAMBLE_RATE, n_db, _PROFILE_SHRINK_K * 30)

        int_rate = _shrink(
            pass_atts["interception"].mean() if len(pass_atts) > 0
            else _LG_INT_RATE,
            _LG_INT_RATE, len(pass_atts), _PROFILE_SHRINK_K * 30)

        ypc_vals = rushes["yards_gained"].dropna()
        ypc = _shrink(
            ypc_vals.mean() if len(ypc_vals) > 0 else _LG_YPC,
            _LG_YPC, len(ypc_vals), _PROFILE_SHRINK_K * 30)
        ypc_sd = ypc_vals.std() if len(ypc_vals) > 10 else _LG_YPC_SD

        fumble_rate = _shrink(
            tp["fumble_lost"].mean() if len(tp) > 0 else _LG_FUMBLE_RATE,
            _LG_FUMBLE_RATE, len(tp), _PROFILE_SHRINK_K * 50)

        proe = (pass_rate.get("neutral", _LG_PASS_RATE.get("neutral", 0.58))
                - _LG_PASS_RATE.get("neutral", 0.58))

        profiles[team] = {
            "pace": pace, "pass_rate": pass_rate, "no_huddle": no_huddle,
            "comp_rate": comp_rate, "ypa_comp": ypa_comp,
            "ypa_comp_sd": ypa_comp_sd, "sack_rate": sack_rate,
            "scramble_rate": scramble_rate, "int_rate": int_rate,
            "ypc": ypc, "ypc_sd": ypc_sd, "fumble_rate": fumble_rate,
            "proe": proe, "n_games": n_games,
        }

    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# QB efficiency profiles
# ══════════════════════════════════════════════════════════════════════════════

def build_qb_profiles(seasons: list[int] | None = None) -> dict:
    """Build per-QB efficiency profiles from PBP.

    Returns: {qb_name: {comp_rate, ypa_comp, sack_rate, scramble_rate,
                         int_rate, n_dropbacks, is_backup: bool}}
    """
    import nfl_data_py as nfl

    if seasons is None:
        seasons = [2023, 2024, 2025]

    pbp = nfl.import_pbp_data(seasons)
    # dropbacks: pass_attempt=1 OR sack=1 OR qb_scramble=1
    dbk = pbp[((pbp["pass_attempt"] == 1) |
               (pbp["sack"] == 1) |
               (pbp["qb_scramble"] == 1))].copy()
    dbk = dbk[dbk["passer_player_name"].notna()].copy()

    qb_profiles = {}
    for qb, qb_plays in dbk.groupby("passer_player_name"):
        n_db = len(qb_plays)
        if n_db < 30:
            continue

        pass_atts = qb_plays[qb_plays["pass_attempt"] == 1]
        non_sack = pass_atts[pass_atts["sack"] != 1]
        comps = non_sack[non_sack["complete_pass"] == 1]

        # Determine starter vs backup: starters have >= 8 games with 15+ dropbacks
        games_started = qb_plays.groupby("game_id").size()
        n_start_games = (games_started >= 15).sum()
        is_backup = n_start_games < 8

        # Shrink toward backup priors if backup, else league average
        prior = _BACKUP_PRIORS if is_backup else {
            "comp_rate": _LG_COMP_RATE,
            "ypa_comp": _LG_YPA_COMP,
            "sack_rate": _LG_SACK_RATE,
            "scramble_rate": _LG_SCRAMBLE_RATE,
            "int_rate": _LG_INT_RATE,
        }
        k = 60 if is_backup else 120  # more shrinkage for backups

        comp_rate = _shrink(
            non_sack["complete_pass"].mean() if len(non_sack) > 0
            else prior["comp_rate"],
            prior["comp_rate"], len(non_sack), k)

        ypa_comp = _shrink(
            comps["yards_gained"].mean() if len(comps) > 0
            else prior["ypa_comp"],
            prior["ypa_comp"], len(comps), k)

        sack_rate = _shrink(
            qb_plays["sack"].mean(), prior["sack_rate"], n_db, k)

        scramble_rate = _shrink(
            qb_plays["qb_scramble"].mean(), prior["scramble_rate"], n_db, k)

        int_rate = _shrink(
            pass_atts["interception"].mean() if len(pass_atts) > 0
            else prior["int_rate"],
            prior["int_rate"], len(pass_atts), k)

        qb_profiles[qb] = {
            "comp_rate": comp_rate,
            "ypa_comp": ypa_comp,
            "sack_rate": sack_rate,
            "scramble_rate": scramble_rate,
            "int_rate": int_rate,
            "n_dropbacks": n_db,
            "is_backup": is_backup,
        }

    return qb_profiles


# ══════════════════════════════════════════════════════════════════════════════
# Empirical outcome table lookup
# ══════════════════════════════════════════════════════════════════════════════

def _dist_bucket(distance: int) -> str:
    """Classify yards-to-go into distance bucket."""
    if distance <= 3:
        return "short"
    if distance <= 6:
        return "med"
    if distance <= 10:
        return "long"
    return "xlong"


def _yl_zone(yardline_100: int) -> str:
    """Classify field position into zone."""
    if yardline_100 <= 20:
        return "rz"
    if yardline_100 <= 40:
        return "opp40"
    if yardline_100 <= 79:
        return "mid"
    return "own20"


def _lookup_outcome(down: int, distance: int, yardline_100: int) -> dict | None:
    """Look up empirical outcome rates for this situation.

    Returns dict with comp_rate, sack_rate, int_rate, scramble_rate,
    ypc_comp, ypc_comp_sd, ypc_rush, ypc_rush_sd, fumble_rate, pass_rate.
    Returns None if no data for this bucket.
    """
    key = f"{down}|{_dist_bucket(distance)}|{_yl_zone(yardline_100)}"
    return _OUTCOME_TABLES.get(key)


def _lookup_4th_down(distance: int, yardline_100: int) -> dict | None:
    """Look up empirical 4th-down decision rates.

    Returns dict with go_for_it, punt, fg rates.
    """
    key = f"{_dist_bucket(distance)}|{_yl_zone(yardline_100)}"
    return _FOURTH_DOWN_RATES.get(key)


# ══════════════════════════════════════════════════════════════════════════════
# Play simulation
# ══════════════════════════════════════════════════════════════════════════════

def _sim_play(rng: np.random.Generator,
              is_pass: bool,
              profile: dict,
              qb: dict | None,
              yardline_100: int,
              down: int = 1,
              distance: int = 10) -> dict:
    """Simulate a single play outcome.

    Args:
        is_pass: True for pass play (dropback), False for designed run
        profile: team profile dict
        qb: QB profile dict (overrides team efficiency if provided)
        yardline_100: yards to opponent end zone (1-99)
        down: current down (1-4)
        distance: yards to first down

    Returns: {
        "yards": int, "play_result": str,
        "turnover": bool, "touchdown": bool,
        "sack": bool, "scramble": bool,
        "completion": bool, "interception": bool,
        "fumble": bool,
    }
    """
    # ── Base rates from team/QB profile ──
    team_comp = qb["comp_rate"] if qb else profile["comp_rate"]
    team_ypa = qb["ypa_comp"] if qb else profile["ypa_comp"]
    team_sack = qb["sack_rate"] if qb else profile["sack_rate"]
    team_scramble = qb["scramble_rate"] if qb else profile["scramble_rate"]
    team_int = qb["int_rate"] if qb else profile["int_rate"]
    team_ypc = profile["ypc"]
    team_ypc_sd = profile["ypc_sd"]
    team_ypa_sd = profile.get("ypa_comp_sd", _LG_YPA_COMP_SD)
    fumble_rate = profile["fumble_rate"]

    # ── Empirical outcome table lookup ──
    # Use the table to set situational base rates, then apply team-specific
    # multiplier: team_value / league_avg to shift above/below the base.
    emp = _lookup_outcome(down, distance, yardline_100)

    if emp and emp.get("comp_rate") is not None:
        # Situational rates from PBP data
        comp_rate = emp["comp_rate"] * (team_comp / _LG_COMP_RATE)
        comp_rate = max(0.15, min(0.90, comp_rate))
    else:
        comp_rate = team_comp

    if emp and emp.get("sack_rate") is not None:
        sack_rate = emp["sack_rate"] * (team_sack / _LG_SACK_RATE)
        sack_rate = max(0.01, min(0.25, sack_rate))
    else:
        sack_rate = team_sack

    if emp and emp.get("scramble_rate") is not None:
        scramble_rate = emp["scramble_rate"] * (team_scramble / _LG_SCRAMBLE_RATE)
        scramble_rate = max(0.01, min(0.20, scramble_rate))
    else:
        scramble_rate = team_scramble

    if emp and emp.get("int_rate") is not None:
        int_rate = emp["int_rate"] * (team_int / _LG_INT_RATE)
        int_rate = max(0.005, min(0.10, int_rate))
    else:
        int_rate = team_int

    if emp and emp.get("ypc_comp") is not None:
        ypa_comp = emp["ypc_comp"] * (team_ypa / _LG_YPA_COMP)
        ypa_comp = max(2.0, ypa_comp)
    else:
        ypa_comp = team_ypa

    if emp and emp.get("ypc_comp_sd") is not None:
        ypa_comp_sd = emp["ypc_comp_sd"]
    else:
        ypa_comp_sd = team_ypa_sd

    if emp and emp.get("ypc_rush") is not None:
        ypc = emp["ypc_rush"] * (team_ypc / _LG_YPC)
        ypc = max(0.5, ypc)
    else:
        ypc = team_ypc

    if emp and emp.get("ypc_rush_sd") is not None:
        ypc_sd = emp["ypc_rush_sd"]
    else:
        ypc_sd = team_ypc_sd

    # Cap rush CV at 1.0 to prevent explosive-run SD inflation.
    # PBP SDs include 50-80 yd breakaway runs that push CV to 1.5-1.9
    # in some buckets, creating gamma shape < 0.7 with 30%+ stuffed
    # rushes (NFL actual ~22%).  CV cap of 1.0 → shape ≥ ~1.3-1.6,
    # matching NFL short-yardage conversion rates and TFL rates.
    _max_sd = max(ypc, 1.5) * 1.0
    if ypc_sd > _max_sd:
        ypc_sd = _max_sd

    if emp and emp.get("fumble_rate") is not None:
        fumble_rate = emp["fumble_rate"]

    result = {
        "yards": 0, "play_result": "incomplete",
        "turnover": False, "touchdown": False,
        "sack": False, "scramble": False,
        "completion": False, "interception": False,
        "fumble": False,
    }

    if is_pass:
        # Dropback outcomes: sack, scramble, pass
        r = rng.random()
        if r < sack_rate:
            # Sack
            yards = round(rng.normal(_LG_SACK_YARDS, _LG_SACK_YARDS_SD))
            yards = max(yards, yardline_100 - 99)  # can't go past own 1-yard line
            result["yards"] = yards
            result["play_result"] = "sack"
            result["sack"] = True
            # Sack-fumble (rare)
            if rng.random() < fumble_rate * 1.5:
                result["fumble"] = True
                result["turnover"] = True
        elif r < sack_rate + scramble_rate:
            # Scramble
            yards = max(0, round(rng.normal(_LG_SCRAMBLE_YARDS,
                                           _LG_SCRAMBLE_YARDS_SD)))
            yards = min(yards, yardline_100)
            result["yards"] = yards
            result["play_result"] = "scramble"
            result["scramble"] = True
            if yards >= yardline_100:
                result["touchdown"] = True
            if rng.random() < fumble_rate:
                result["fumble"] = True
                result["turnover"] = True
                result["touchdown"] = False
        else:
            # Pass attempt — single draw for INT/completion/incomplete.
            # NFL comp% = completions / attempts (where attempts include INTs).
            # P(INT) = int_rate, P(completion) = comp_rate,
            # P(incomplete) = 1 - comp_rate - int_rate
            r_pass = rng.random()
            if r_pass < int_rate:
                result["play_result"] = "interception"
                result["interception"] = True
                result["turnover"] = True
                result["yards"] = 0
            elif r_pass < int_rate + comp_rate:
                # Completion — gamma distribution with shape derived from
                # empirical Y/C variance per situation bucket.
                # shape = mean² / variance; scale = mean / shape.
                # Lower shape (≈1.2 midfield) = more short completions +
                # thinner deep tail = correct 1st-down conversion rates.
                if ypa_comp_sd > 0 and ypa_comp > 0:
                    _comp_shape = max(0.8, (ypa_comp / ypa_comp_sd) ** 2)
                else:
                    _comp_shape = 1.5
                _comp_scale = max(0.5, ypa_comp / _comp_shape)
                yards = max(0, round(
                    rng.standard_gamma(_comp_shape) * _comp_scale))
                yards = min(yards, yardline_100)
                result["yards"] = yards
                result["play_result"] = "completion"
                result["completion"] = True
                if yards >= yardline_100:
                    result["touchdown"] = True
                if rng.random() < fumble_rate * 0.5:
                    result["fumble"] = True
                    result["turnover"] = True
                    result["touchdown"] = False
            else:
                # Incomplete
                result["play_result"] = "incomplete"
                result["yards"] = 0
    else:
        # Rush — shifted gamma: Gamma(shape, scale) - 1.
        # shape = (mean_unshifted / sd)² where sd is from empirical PBP.
        # Floor shape at 1.0: the raw PBP SD includes 50-80 yd breakaway
        # runs that inflate CV → shape < 0.6 → 30%+ of rushes at 0 yd.
        # NFL actual ≤0 rate is ~22-25%.  Shape ≥ 1.0 matches that and
        # produces correct 3rd-and-short conversion rates (~62% vs 63%).
        if ypc_sd > 0 and ypc > 0:
            _rush_shift = 1.0  # shifted gamma: gamma - shift
            _rush_mean_unshifted = ypc + _rush_shift
            _rush_shape = max(1.0, (_rush_mean_unshifted / ypc_sd) ** 2)
        else:
            _rush_shift = 1.0
            _rush_mean_unshifted = ypc + _rush_shift
            _rush_shape = 1.5
        _rush_scale = _rush_mean_unshifted / _rush_shape
        yards = round(rng.standard_gamma(_rush_shape) * _rush_scale
                      - _rush_shift)
        yards = max(-5, yards)  # floor on rush loss
        yards = min(yards, yardline_100)
        result["yards"] = yards
        result["play_result"] = "rush"
        if yards >= yardline_100:
            result["touchdown"] = True
        if rng.random() < fumble_rate:
            result["fumble"] = True
            result["turnover"] = True
            result["touchdown"] = False

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Drive simulation
# ══════════════════════════════════════════════════════════════════════════════

def _fg_success_rate(yardline_100: int) -> float:
    """Field goal success probability given yardline_100."""
    fg_dist = yardline_100 + 17  # snap + hold
    if fg_dist > 63:
        return 0.0
    # Interpolate from empirical table
    for d in sorted(_FG_SUCCESS.keys()):
        if fg_dist <= d:
            return _FG_SUCCESS[d]
    return 0.20  # very long


def _sim_drive(rng: np.random.Generator,
               profile: dict,
               qb: dict | None,
               start_yl: int,
               game_secs: float,
               score_diff: int,
               qtr: int,
               is_second_half_start: bool = False,
               def_timeouts: int = 3,
               ) -> dict:
    """Simulate a single offensive drive.

    Args:
        profile: team profile
        qb: QB profile (or None)
        start_yl: starting yardline_100 (1-99)
        game_secs: game_seconds_remaining at drive start
        score_diff: offense's score - defense's score
        qtr: current quarter (1-4)
        is_second_half_start: True if this is the second-half kickoff
        def_timeouts: defensive team's remaining timeouts this half

    Returns: {
        "outcome": "td" | "fg" | "fg_miss" | "punt" | "turnover" |
                   "turnover_on_downs" | "end_of_half" | "kneel",
        "points": int,
        "plays": list[dict],  # play-by-play
        "seconds_used": float,
        "end_yl": int,  # where opponent gets ball (for turnover/punt)
        "def_timeouts": int,  # remaining after this drive
    }
    """
    yl = start_yl
    down = 1
    distance = 10
    first_down_yl = max(yl - 10, 0)
    plays = []
    total_secs = 0.0

    while game_secs - total_secs > 0:
        remaining = game_secs - total_secs
        current_qtr = _quarter_from_remaining(remaining)

        sit = _classify_situation(score_diff, current_qtr, remaining)

        # Kneel-down logic: leading with < 120 seconds and have the ball
        if (sit in ("leading_late", "two_min") and
                score_diff > 0 and remaining < 120 and down <= 4):
            # Kneel: lose 1 yard, run ~40 seconds
            plays.append({"play_result": "kneel", "yards": -1,
                          "is_pass": False, "seconds": 40})
            total_secs += 40
            yl = min(yl + 1, 99)
            down += 1
            if down > 4:
                break  # punt or game over
            continue

        # ── 4th-down decision: decide BEFORE the play ──────────────
        # Uses empirical NFL decision rates from PBP tables.
        if down == 4:
            fg_dist = yl + 17
            fg_prob = _fg_success_rate(yl)
            can_fg = fg_dist <= 55 and fg_prob > 0.35

            # Look up empirical go-for-it rate from PBP tables.
            # PBP rates reflect average game-state (trailing teams go for
            # it more often, inflating the average). Apply score-context
            # scaling: dampen when leading/neutral, amplify when trailing.
            fd_rates = _lookup_4th_down(distance, yl)
            if fd_rates:
                go_rate = fd_rates["go_for_it"]
                fg_rate = fd_rates["fg"]

                # Score-context scaling.  PBP rates average over all game
                # states; trailing teams inflate go-for-it, so neutral/
                # leading need slight dampening.  But PBP 2023-2025 data
                # shows league-wide go-for-it rates have risen sharply,
                # so dampening should be modest.
                is_rz = yl <= 20
                if is_rz and distance <= 3:
                    go_mult = 1.0   # 4th-and-goal: always go for it
                elif score_diff < -10 and remaining < 600:
                    go_mult = 1.3   # desperate, go for it more
                elif score_diff < -3:
                    go_mult = 1.0   # trailing, use PBP rate as-is
                elif score_diff > 10:
                    go_mult = 0.35  # big lead, conservative
                elif is_rz:
                    go_mult = 0.90  # RZ: minimal dampening
                else:
                    go_mult = 0.80  # neutral: modest dampening
                go_rate = min(0.95, go_rate * go_mult)

                # Decision: FG first if in range, then go/punt
                r = rng.random()
                if can_fg and r < fg_rate:
                    decision = "fg"
                elif r < fg_rate + go_rate:
                    decision = "go"
                else:
                    decision = "punt"
            else:
                # Fallback: simple rules if no table data
                if yl <= 2 and distance <= 2:
                    decision = "go"
                elif can_fg:
                    decision = "fg"
                elif yl <= 40 and score_diff < -10 and remaining < 480:
                    decision = "go"
                else:
                    decision = "punt"

            if decision == "fg":
                total_secs += 8
                if rng.random() < fg_prob:
                    return {
                        "outcome": "fg",
                        "points": 3,
                        "plays": plays,
                        "seconds_used": total_secs,
                        "end_yl": _kickoff_yl(rng),
                        "def_timeouts": def_timeouts,
                    }
                else:
                    return {
                        "outcome": "fg_miss",
                        "points": 0,
                        "plays": plays,
                        "seconds_used": total_secs,
                        "end_yl": min(99, 100 - yl + 7),
                        "def_timeouts": def_timeouts,
                    }

            if decision == "punt":
                punt_dist = max(20, round(rng.normal(
                    _PUNT_DISTANCE_MEAN, _PUNT_DISTANCE_SD)))
                punt_land = yl - punt_dist
                if punt_land < 0:
                    punt_land = 0
                punt_return = max(0, round(rng.normal(
                    _PUNT_RETURN_MEAN, _PUNT_RETURN_SD)))
                opp_yl = max(1, min(99, 100 - punt_land - punt_return))
                if punt_land <= 0:
                    opp_yl = 75
                total_secs += 8
                return {
                    "outcome": "punt",
                    "points": 0,
                    "plays": plays,
                    "seconds_used": total_secs,
                    "end_yl": opp_yl,
                    "def_timeouts": def_timeouts,
                }
            # else: decision == "go" — fall through to simulate the play

        # ── Penalty check (before simulating the play) ────────────
        # Defensive auto-first-down penalties (DPI, def holding,
        # roughing, illegal contact) extend drives by granting a
        # free first down + yards.  Offensive penalties (holding,
        # false start) replay the down with yardage loss.
        # These are "no-play" events: they don't count as plays in
        # the stat line, matching PBP play_type='no_play' filtering.
        _pen_roll = rng.random()
        if _pen_roll < _DEF_AUTO_FD_RATE:
            # Defensive auto-first-down penalty
            _pen_type = rng.random()
            if _pen_type < 0.32:
                # DPI: spot foul, average ~15 yards
                _pen_yds = max(5, min(yl - 1, round(rng.normal(15, 7))))
            elif _pen_type < 0.51:
                _pen_yds = min(5, yl - 1) or 1    # Defensive holding: 5
            elif _pen_type < 0.71:
                _pen_yds = min(15, yl - 1) or 1   # Roughing/UR: 15
            else:
                _pen_yds = min(5, yl - 1) or 1    # Offside/illegal contact: 5
            yl = max(1, yl - _pen_yds)
            down = 1
            distance = min(10, yl)
            first_down_yl = max(yl - distance, 0)
            # Penalty dead time: 0s — the pace table's snap-to-snap
            # values already include any delay caused by penalties
            # between plays (PBP snap_gap naturally captures it).
            continue

        if _pen_roll < _DEF_AUTO_FD_RATE + _DEF_5YD_PENALTY_RATE:
            # Defensive 5-yard penalty (offsides, encroachment, NZI).
            # 5 yards toward opponent, same down.  If penalty yardage
            # exceeds remaining distance → first down.
            _pen_yds = min(5, yl - 1) or 1
            yl = max(1, yl - _pen_yds)
            distance -= _pen_yds
            if distance <= 0:
                # Penalty yards exceeded distance — first down
                down = 1
                distance = min(10, yl)
                first_down_yl = max(yl - distance, 0)
            # Dead ball foul: no play, no clock.
            continue

        if _pen_roll < (_DEF_AUTO_FD_RATE + _DEF_5YD_PENALTY_RATE
                        + _OFF_PENALTY_RATE):
            # Offensive penalty: replay down with yard loss
            _pen_type = rng.random()
            if _pen_type < 0.48:
                _pen_yds = 10   # Holding
            elif _pen_type < 0.72:
                _pen_yds = 5    # False start
            else:
                _pen_yds = 5    # Delay, illegal formation
            # Half the distance if penalty would push past own goal
            if yl + _pen_yds > 99:
                _pen_yds = max(1, (100 - yl) // 2)
            yl = min(99, yl + _pen_yds)
            distance = distance + _pen_yds
            # Don't increment down (replay)
            continue

        # ── Determine play call ───────────────────────────────────
        # Fine-grained pass rate from PBP table: (score_bucket × time_bucket)
        # Use `remaining` (not drive-start `game_secs`) so the time bucket
        # updates correctly within a drive — a drive starting at 360s should
        # shift from q4_gt5 to q4_lt5 once the clock crosses 300s.
        sb = _score_bucket(score_diff)
        tb = _time_bucket(remaining)
        table_pr = _PASS_RATE_TABLE.get((sb, tb), 0.58)

        # Team adjustment: PROE (Pass Rate Over Expected) — stable
        # team trait from neutral-state play-calling, applied across
        # all game states.  Game-state conditioning is already captured
        # by the fine-grained _PASS_RATE_TABLE.
        proe = profile.get("proe", 0.0)
        pr = table_pr + proe + profile.get("pass_vol_adj", 0.0)

        # Down/distance adjustment (multiplicative, preserves extreme cells).
        # The table is unconditional on down — 3rd-and-long warrants higher
        # pass rate.  Multiplicative scaling keeps adjustments proportional
        # to the game-state baseline: a +15% relative shift on table_pr=0.16
        # adds +0.024, vs +0.087 on table_pr=0.58.  This prevents the
        # additive formulation from compressing extreme cells toward center.
        emp_play = _lookup_outcome(down, distance, yl)
        if emp_play and emp_play.get("pass_rate") is not None:
            emp_pr = emp_play["pass_rate"]
            dd_mult = 1.0 + 0.15 * (emp_pr / 0.58 - 1.0)
            pr *= dd_mult
        else:
            if down >= 3 and distance >= 7:
                pr *= 1.15   # 15% relative increase for obvious passing downs
            elif down == 1:
                pr *= 0.97   # 3% relative decrease for 1st down

        pr = max(0.08, min(0.98, pr))
        is_pass = rng.random() < pr

        # ── Simulate play ─────────────────────────────────────────
        play = _sim_play(rng, is_pass, profile, qb, yl, down, distance)
        play["is_pass"] = is_pass
        play["down"] = down
        play["qtr"] = current_qtr
        play["score_diff"] = score_diff
        play["distance"] = distance
        play["yl_before"] = yl

        # ── Clock management ──────────────────────────────────────
        # Base pace from fine-grained PBP table: (score_bucket × time_bucket)
        pace = _PACE_TABLE.get((sb, tb), 30.0)
        # Team adjustment: some teams play faster/slower than average
        team_pace = profile["pace"].get(sit, _LG_PACE.get(sit, 30.0))
        lg_pace = _LG_PACE.get(sit, 30.0)
        pace += (team_pace - lg_pace)
        pace = max(8.0, pace)

        # ── Clock-stop mechanics (PBP-calibrated) ────────────────
        # The pace table values represent running-clock snap-to-snap
        # times.  Clock-stop events (incompletions, OOB, timeouts)
        # override to shorter paces.  No compensation needed: the table
        # already measures running-clock plays, not the overall average.
        clock_stopped = False

        # Incomplete pass always stops the clock.
        if play["play_result"] == "incomplete":
            pace = 5.0
            clock_stopped = True

        # Out of bounds stops clock (last 5:00 of Q4 / last 2:00 of Q2).
        # Rates from PBP 2023-2025 audit by score state and time bucket.
        if not clock_stopped and not is_pass:
            oob_rate = 0.0
            if tb == "q4_lt5":
                if score_diff < -3:
                    oob_rate = 0.102   # trailing, under 5 min Q4
                elif score_diff > 3:
                    oob_rate = 0.022   # leading, under 5 min Q4
            elif tb == "q4_gt5" and score_diff < -3:
                oob_rate = 0.094       # trailing, Q4 > 5 min
            elif (remaining <= 1920 and remaining > 1800
                  and score_diff < -3):
                oob_rate = 0.08        # trailing, Q2 two-minute drill

            if oob_rate > 0 and rng.random() < oob_rate:
                pace = 5.0
                clock_stopped = True

        # Two-minute warning: automatic clock stop at 2:00 remaining
        # in each half (game_secs 120 for Q4, 1920 for Q2).
        # Fires once when a play's clock runoff crosses the boundary.
        if not clock_stopped:
            for boundary in (120.0, 1920.0):
                if remaining > boundary >= remaining - pace:
                    # Cap pace to stop at boundary + TV timeout dead time.
                    pace = (remaining - boundary) + 5.0
                    clock_stopped = True
                    break

        # Defensive timeout: trailing defense calls TO to stop the clock.
        # PBP 2023-2025: trail 10+ uses 1.68 def TOs/Q4, trail 4-9 uses
        # 2.38.  Per running-clock play rates: ~0.11 and ~0.16.
        # Only fires on running-clock plays when defense is trailing 4+.
        if (not clock_stopped and def_timeouts > 0
                and score_diff > 3 and remaining <= 900):
            # score_diff > 0 = offense leading, defense trailing
            if score_diff >= 10:
                to_prob = 0.11   # 1.68 TOs / ~15 running-clock plays
            else:
                to_prob = 0.16   # 2.38 TOs / ~15 running-clock plays
            if rng.random() < to_prob:
                pace = 8.0       # play + TO dead time only
                def_timeouts -= 1
                clock_stopped = True

        # Note: penalty events (auto-FD, offensive) are modeled
        # explicitly above as no-play events with zero clock cost.
        # The pace table's snap-to-snap values already capture any
        # penalty delay (PBP snap_gap naturally includes it).

        play["seconds"] = pace
        total_secs += pace
        plays.append(play)

        # Handle turnover
        if play["turnover"]:
            end_yl = max(1, yl - play["yards"])
            return {
                "outcome": "turnover",
                "points": 0,
                "plays": plays,
                "seconds_used": total_secs,
                "end_yl": 100 - end_yl,
                "def_timeouts": def_timeouts,
            }

        # Handle touchdown
        if play["touchdown"]:
            return {
                "outcome": "td",
                "points": 6,  # PAT handled by caller
                "plays": plays,
                "seconds_used": total_secs,
                "end_yl": _kickoff_yl(rng),
                "def_timeouts": def_timeouts,
            }

        # Update field position
        yl = max(1, yl - play["yards"])

        # Check first down
        if play["yards"] >= distance:
            down = 1
            distance = min(10, yl)
            first_down_yl = max(yl - distance, 0)
        else:
            distance -= play["yards"]
            if distance <= 0:
                distance = min(10, yl)
                down = 1
            else:
                down += 1

            # Failed 4th-down attempt → turnover on downs
            if down > 4:
                return {
                    "outcome": "turnover_on_downs",
                    "points": 0,
                    "plays": plays,
                    "seconds_used": total_secs,
                    "end_yl": 100 - yl,
                    "def_timeouts": def_timeouts,
                }

    # Ran out of time
    return {
        "outcome": "end_of_half",
        "points": 0,
        "plays": plays,
        "seconds_used": total_secs,
        "end_yl": 0,
        "def_timeouts": def_timeouts,
    }


def _kickoff_yl(rng: np.random.Generator) -> int:
    """Starting field position after kickoff."""
    yl = int(rng.normal(_KICKOFF_MEAN_YL, _KICKOFF_SD_YL))
    return max(20, min(99, yl))


def _quarter_from_remaining(game_secs: float) -> int:
    """Determine quarter from game_seconds_remaining."""
    if game_secs > 2700:
        return 1
    if game_secs > 1800:
        return 2
    if game_secs > 900:
        return 3
    return 4


# ══════════════════════════════════════════════════════════════════════════════
# Market anchoring + game-level efficiency perturbation
# ══════════════════════════════════════════════════════════════════════════════

# Sim version tag: bump whenever constants, play model, or EFF_SD change.
# _calibrate() uses this to decide whether to re-measure.
_SIM_VERSION = "2026-10-10e"

# Will be set by _calibrate(); initialized to last known value so the
# module works without calling calibrate at import time.
_SIM_LG_MEAN_SCORE = 19.7


def _estimate_rel_eff(profile: dict) -> float:
    """Estimate a team's relative offensive efficiency vs league average.

    Returns a multiplier (>1 = above average, <1 = below average) that
    approximates how much the sim's mean score for this profile differs
    from the league-average-profile baseline.

    Note: inverse-rate terms (sack, INT) are excluded because they
    inflate the estimate for real profiles (shrinkage toward league
    average makes these ratios ≈ 1.0 with noise, biasing upward via
    Jensen's inequality on 1/x).  Only positive-outcome rates are used.
    """
    comp_rel = profile["comp_rate"] / _LG_COMP_RATE
    ypa_rel  = profile["ypa_comp"]  / _LG_YPA_COMP
    ypc_rel  = profile["ypc"]       / _LG_YPC
    # Weighted by approximate contribution to scoring.
    # Passing dominates (~65% of yards), rushing ~35%.
    return 0.35 * comp_rel + 0.40 * ypa_rel + 0.25 * ypc_rel


def _apply_game_eff(profile: dict, pass_mult: float,
                    rush_mult: float) -> dict:
    """Return a copy of *profile* with efficiency scaled by split multipliers.

    pass_mult scales passing stats (comp%, Y/A, sack%, INT%).
    rush_mult scales rushing stats (YPC only).
    The two are drawn independently — empirical game-level correlation ≈ 0.
    Pace, pass_rate, no_huddle are untouched (play-call, not efficiency).
    """
    p = dict(profile)
    pass_inv = 1.0 / pass_mult

    # Pass efficiency — comp_rate is NOT scaled by the game-level
    # effect.  Completion % is a stable per-QB skill; game-to-game
    # variance in passing efficiency comes from Y/C and sack/INT
    # rates.  Scaling comp_rate by pass_mult causes a Jensen's-
    # inequality artifact: high-efficiency games have more first
    # downs → more plays, so the high comp-rate games contribute
    # disproportionately to the play-weighted average, inflating
    # it ~4-5pp above the true league mean.
    p["comp_rate"]     = profile["comp_rate"]
    p["ypa_comp"]      = max(4.0,  profile["ypa_comp"] * pass_mult)
    p["ypa_comp_sd"]   = profile.get("ypa_comp_sd", _LG_YPA_COMP_SD)
    p["sack_rate"]     = max(0.01, min(0.20, profile["sack_rate"] * pass_inv))
    p["int_rate"]      = max(0.005, min(0.08, profile["int_rate"] * pass_inv))
    p["scramble_rate"] = profile["scramble_rate"]

    # Rush efficiency (independent)
    p["ypc"]           = max(1.5,  profile["ypc"] * rush_mult)
    p["ypc_sd"]        = profile.get("ypc_sd", _LG_YPC_SD)
    p["fumble_rate"]   = profile["fumble_rate"]
    return p


def _apply_game_eff_qb(qb: dict, pass_mult: float) -> dict:
    """Scale a QB efficiency profile by the pass-level multiplier only."""
    if qb is None:
        return None
    inv = 1.0 / pass_mult
    q = dict(qb)
    q["comp_rate"]  = qb["comp_rate"]  # stable skill, not scaled by game effect
    q["ypa_comp"]   = max(4.0,  qb["ypa_comp"] * pass_mult)
    q["sack_rate"]  = max(0.01, min(0.20, qb["sack_rate"] * inv))
    q["int_rate"]   = max(0.005, min(0.08, qb["int_rate"] * inv))
    return q


def _calibrate(n_sims: int = 1000, seed: int = 42) -> dict:
    """Measure unanchored mean score and auto-calibrate anchoring dampening.

    Steps:
      1. Measure unanchored mean score → ``_SIM_LG_MEAN_SCORE``
      2. Measure scoring elasticity at a test dampening → compute ``_SYM_DAMP``
      3. Measure spread elasticity at a test dampening → compute ``_ASYM_DAMP``
      4. Verify: anchored mean at total=45 within ±0.3 of 22.5

    Called automatically at the start of ``validate`` and can be called
    manually after changing constants.
    """
    global _SIM_LG_MEAN_SCORE, _SYM_DAMP, _ASYM_DAMP

    lg = {
        "pace": _LG_PACE, "pass_rate": _LG_PASS_RATE,
        "no_huddle": _LG_NO_HUDDLE,
        "comp_rate": _LG_COMP_RATE, "ypa_comp": _LG_YPA_COMP,
        "ypa_comp_sd": _LG_YPA_COMP_SD, "sack_rate": _LG_SACK_RATE,
        "scramble_rate": _LG_SCRAMBLE_RATE, "int_rate": _LG_INT_RATE,
        "ypc": _LG_YPC, "ypc_sd": _LG_YPC_SD,
        "fumble_rate": _LG_FUMBLE_RATE, "n_games": 100,
    }

    # ── Step 1: Measure unanchored mean ─────────────────────────────
    rng1 = np.random.default_rng(seed)
    scores = []
    for _ in range(n_sims):
        res = simulate_game(rng1, lg, lg, total=0)
        scores.append(res["home_score"])
        scores.append(res["away_score"])
    unanchored = float(np.mean(scores))
    _SIM_LG_MEAN_SCORE = round(unanchored, 1)
    logger.info("Calibrated _SIM_LG_MEAN_SCORE = %.1f (N=%d, version=%s)",
                _SIM_LG_MEAN_SCORE, n_sims, _SIM_VERSION)

    # ── Step 2: Symmetric dampening (total-driven) ──────────────────
    # Measure anchored mean at a test dampening, compute scoring
    # elasticity, solve for optimal dampening.
    test_damp = 0.50
    _SYM_DAMP = test_damp
    _ASYM_DAMP = 0.0  # disable asymmetric for this measurement

    rng2 = np.random.default_rng(seed + 1)
    test_scores = []
    for _ in range(n_sims):
        res = simulate_game(rng2, lg, lg, total=45.0, spread=0.0)
        test_scores.append(res["home_score"])
        test_scores.append(res["away_score"])
    test_mean = float(np.mean(test_scores))

    # Elasticity: how much score changes per unit change in anchor.
    # anchor = 1.0 + damp * (22.5/unanchored - 1.0)
    # delta_anchor = damp * (22.5/unanchored - 1.0)
    # elasticity = (test_mean - unanchored) / delta_anchor
    raw_ratio = 22.5 / unanchored
    delta_anchor = test_damp * (raw_ratio - 1.0)
    if abs(delta_anchor) < 1e-6:
        _SYM_DAMP = 0.45  # near-zero anchor shift → keep default
    else:
        elasticity = (test_mean - unanchored) / delta_anchor
        # Want: 22.5 = unanchored + elasticity * damp_opt * (raw_ratio - 1)
        _SYM_DAMP = (22.5 - unanchored) / (elasticity * (raw_ratio - 1.0))
        _SYM_DAMP = max(0.15, min(0.85, _SYM_DAMP))
    logger.info("Symmetric: test_mean=%.2f at damp=%.2f → elasticity=%.1f → "
                "_SYM_DAMP=%.3f", test_mean, test_damp,
                elasticity if abs(delta_anchor) >= 1e-6 else 0, _SYM_DAMP)

    # ── Step 3: Asymmetric dampening (spread-driven) ────────────────
    # Measure score diff at a test spread, compute spread elasticity.
    test_asym = 0.50
    _ASYM_DAMP = test_asym

    rng3 = np.random.default_rng(seed + 2)
    test_diffs = []
    for _ in range(n_sims):
        res = simulate_game(rng3, lg, lg, total=45.0, spread=-7.0)
        test_diffs.append(res["home_score"] - res["away_score"])
    test_diff = float(np.mean(test_diffs))

    # Expected diff for spread=-7: implied_diff = 7/2 = 3.5
    implied_diff = 3.5
    # raw_diff = implied_diff / (unanchored * 1.0) [avg_rel_eff=1 for LG]
    raw_diff = implied_diff / unanchored
    test_delta_asym = test_asym * raw_diff
    if abs(test_delta_asym) < 1e-6:
        _ASYM_DAMP = 0.32
    else:
        spread_elasticity = test_diff / test_delta_asym
        _ASYM_DAMP = implied_diff / (spread_elasticity * raw_diff)
        _ASYM_DAMP = max(0.10, min(0.80, _ASYM_DAMP))
    logger.info("Asymmetric: test_diff=%.2f at asym=%.2f, spread=-7 → "
                "spread_elast=%.1f → _ASYM_DAMP=%.3f", test_diff, test_asym,
                spread_elasticity if abs(test_delta_asym) >= 1e-6 else 0,
                _ASYM_DAMP)

    # ── Step 4: Verify + correction ─────────────────────────────────
    # Run a verify batch; if drift exceeds ±0.3, apply a proportional
    # correction to _SYM_DAMP (the verify error is mostly sym noise).
    for attempt in range(2):
        rng4 = np.random.default_rng(seed + 3 + attempt * 10)
        verify = []
        for _ in range(n_sims):
            res = simulate_game(rng4, lg, lg, total=45.0, spread=0.0)
            verify.append(res["home_score"])
            verify.append(res["away_score"])
        verify_mean = float(np.mean(verify))
        gap = abs(verify_mean - 22.5)
        logger.info("Anchored verify (attempt %d): mean=%.2f (gap=%.2f)",
                     attempt + 1, verify_mean, gap)
        if gap < 0.3:
            break
        # Correction: scale _SYM_DAMP proportionally
        correction = (22.5 - unanchored) / (verify_mean - unanchored)
        _SYM_DAMP *= correction
        _SYM_DAMP = max(0.15, min(0.85, _SYM_DAMP))
        logger.info("Correction: _SYM_DAMP → %.3f", _SYM_DAMP)
    else:
        assert gap < 0.5, (
            f"Anchored mean {verify_mean:.2f} deviates from implied "
            f"22.5 by {gap:.2f} (>0.5) after correction."
        )

    return {
        "unanchored_mean": unanchored,
        "sym_damp": _SYM_DAMP,
        "asym_damp": _ASYM_DAMP,
        "verify_mean": verify_mean,
        "verify_gap": gap,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full game simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_game(rng: np.random.Generator,
                  home_profile: dict,
                  away_profile: dict,
                  home_qb: dict | None = None,
                  away_qb: dict | None = None,
                  spread: float = 0.0,
                  total: float = 45.0,
                  ) -> dict:
    """Simulate a full NFL game.

    Returns: {
        "home_score": int, "away_score": int,
        "home_plays": list[dict], "away_plays": list[dict],
        "home_stats": dict, "away_stats": dict,
    }
    """
    home_score = 0
    away_score = 0
    game_secs = _GAME_SECONDS
    home_plays = []
    away_plays = []

    # ── Market anchoring + game-level random effects ─────────────
    if total > 0:
        # Split anchoring into symmetric (total-driven) and asymmetric
        # (spread-driven) components.  Dampening constants (_SYM_DAMP,
        # _ASYM_DAMP) are auto-calibrated by _calibrate().
        implied_avg = total / 2.0
        implied_diff = -spread / 2.0  # positive = home advantage

        # Symmetric: calibrate total scoring level
        avg_rel_eff = 0.5 * (_estimate_rel_eff(home_profile)
                             + _estimate_rel_eff(away_profile))
        raw_avg_ratio = implied_avg / (_SIM_LG_MEAN_SCORE * avg_rel_eff)
        sym_anchor = 1.0 + _SYM_DAMP * (raw_avg_ratio - 1.0)
        sym_anchor = max(0.75, min(1.30, sym_anchor))

        # Asymmetric: create home/away differential
        raw_diff = implied_diff / (_SIM_LG_MEAN_SCORE * avg_rel_eff)
        asym_adj = _ASYM_DAMP * raw_diff

        home_anchor = sym_anchor + asym_adj
        away_anchor = sym_anchor - asym_adj
    else:
        # No market data: draw efficiency centered on 1.0
        home_anchor = 1.0
        away_anchor = 1.0

    # Shared game-environment efficiency: captures weather, field surface,
    # referee style, and overall game tempo.  Both teams' passing efficiency
    # receives the same shared draw, creating the observed positive cross-team
    # correlation (A_pass_yds vs B_pass_yds ≈ +0.17 in PBP).
    shared_eff = rng.normal(0.0, _SHARED_GAME_EFF_SD)

    # Pass efficiency: shared component + independent per-team component.
    # Total variance preserved: _SHARED²  + _INDEP² = 0.05² + 0.109² ≈ 0.12².
    home_pass_eff = max(0.65, rng.normal(
        home_anchor + shared_eff, _INDEP_PASS_EFF_SD))
    away_pass_eff = max(0.65, rng.normal(
        away_anchor + shared_eff, _INDEP_PASS_EFF_SD))

    # Rush efficiency: weakly anchored, independent draws.
    # Empirical correlation(pass, rush) ≈ 0.001 — draw independently.
    home_rush_anchor = 1.0 + _RUSH_ANCHOR_FRAC * (home_anchor - 1.0)
    away_rush_anchor = 1.0 + _RUSH_ANCHOR_FRAC * (away_anchor - 1.0)
    home_rush_eff = max(0.65, rng.normal(home_rush_anchor, _RUSH_EFF_SD))
    away_rush_eff = max(0.65, rng.normal(away_rush_anchor, _RUSH_EFF_SD))

    # Game-level pass volume shift: opponent run defense, game-plan.
    # Independent per team — no ad-hoc cross-team coupling.
    home_pass_vol_adj = rng.normal(0.0, _PASS_VOL_SD)
    away_pass_vol_adj = rng.normal(0.0, _PASS_VOL_SD)

    home_gp = _apply_game_eff(home_profile, home_pass_eff, home_rush_eff)
    away_gp = _apply_game_eff(away_profile, away_pass_eff, away_rush_eff)
    home_gp["pass_vol_adj"] = home_pass_vol_adj
    away_gp["pass_vol_adj"] = away_pass_vol_adj

    # Shared game-pace factor: multiplicative shift on seconds/play for
    # BOTH teams.  Fast-game factor (<1.0) = more plays for everyone;
    # slow-game factor (>1.0) = fewer.  Adds positive cross-team play
    # correlation, reducing the structural negative from the shared clock.
    shared_pace_mult = max(0.80, rng.normal(1.0, _SHARED_PACE_SD))
    for gp in (home_gp, away_gp):
        gp["pace"] = {sit: secs * shared_pace_mult
                      for sit, secs in gp["pace"].items()}
    home_gq = _apply_game_eff_qb(home_qb, home_pass_eff)
    away_gq = _apply_game_eff_qb(away_qb, away_pass_eff)

    # Coin flip: 50/50 who receives first
    home_receives_first = rng.random() < 0.5

    # Track possession and timeouts (3 per half per team)
    home_has_ball = home_receives_first
    current_yl = _kickoff_yl(rng)
    is_second_half = False
    home_tos = 3
    away_tos = 3

    while game_secs > 0:
        profile = home_gp if home_has_ball else away_gp
        qb = home_gq if home_has_ball else away_gq
        score_diff = ((home_score - away_score) if home_has_ball
                      else (away_score - home_score))
        qtr = _quarter_from_remaining(game_secs)

        # Half break
        if game_secs <= 1800 and not is_second_half:
            is_second_half = True
            # Second half: other team receives
            home_has_ball = not home_receives_first
            current_yl = _kickoff_yl(rng)
            game_secs = 1800
            # Reset timeouts for second half
            home_tos = 3
            away_tos = 3
            continue

        # Defensive timeouts: the opposing team's remaining TOs this half.
        def_tos = away_tos if home_has_ball else home_tos

        drive = _sim_drive(
            rng, profile, qb,
            start_yl=current_yl,
            game_secs=game_secs,
            score_diff=score_diff,
            qtr=qtr,
            def_timeouts=def_tos,
        )

        # Update defensive timeouts from drive result
        if home_has_ball:
            away_tos = drive.get("def_timeouts", away_tos)
        else:
            home_tos = drive.get("def_timeouts", home_tos)

        # Record plays
        for p in drive["plays"]:
            p["team"] = "home" if home_has_ball else "away"
        if home_has_ball:
            home_plays.extend(drive["plays"])
        else:
            away_plays.extend(drive["plays"])

        game_secs -= drive["seconds_used"]

        # Score handling
        if drive["outcome"] == "td":
            points = 6
            # PAT: ~94% XP, ~5% go for 2, ~1% miss
            if rng.random() < 0.05:
                points += 2 if rng.random() < _TWO_PT_RATE else 0
            else:
                points += 1 if rng.random() < _XP_RATE else 0

            if home_has_ball:
                home_score += points
            else:
                away_score += points

        elif drive["outcome"] == "fg":
            if home_has_ball:
                home_score += 3
            else:
                away_score += 3

        # Defensive/special teams TD: pick-six or fumble return TD.
        # Empirical rates (2023-2025 PBP): INT return TD 8.9%, fumble 6.4%.
        if drive["outcome"] == "turnover":
            last_play = drive["plays"][-1] if drive["plays"] else {}
            _def_td_rate = (0.089 if last_play.get("interception")
                           else 0.064)
            if rng.random() < _def_td_rate:
                def_td_pts = 6
                if rng.random() < 0.05:
                    def_td_pts += 2 if rng.random() < _TWO_PT_RATE else 0
                else:
                    def_td_pts += 1 if rng.random() < _XP_RATE else 0
                # Defense scores: points go to the OTHER team
                if home_has_ball:
                    away_score += def_td_pts
                else:
                    home_score += def_td_pts

        # Next possession + change-of-possession dead time
        if drive["outcome"] == "td":
            current_yl = _kickoff_yl(rng)
            game_secs -= _COP_AFTER_TD
        elif drive["outcome"] == "fg":
            current_yl = _kickoff_yl(rng)
            game_secs -= _COP_AFTER_FG
        elif drive["outcome"] == "fg_miss":
            current_yl = _kickoff_yl(rng)
            game_secs -= _COP_AFTER_FGMISS
        elif drive["outcome"] == "punt":
            current_yl = drive["end_yl"]
            game_secs -= _COP_AFTER_PUNT
        elif drive["outcome"] == "turnover":
            current_yl = drive["end_yl"]
            game_secs -= _COP_AFTER_TO
        elif drive["outcome"] == "turnover_on_downs":
            current_yl = drive["end_yl"]
            game_secs -= _COP_AFTER_TOD
        elif drive["outcome"] == "end_of_half":
            if is_second_half:
                break  # game over
            # end of first half, handled by half-break logic above
            game_secs = 1800.01  # trigger half break
            continue

        home_has_ball = not home_has_ball

    # Compile stats
    home_stats = _compile_play_stats(home_plays)
    away_stats = _compile_play_stats(away_plays)

    return {
        "home_score": home_score,
        "away_score": away_score,
        "home_plays": home_plays,
        "away_plays": away_plays,
        "home_stats": home_stats,
        "away_stats": away_stats,
    }


def _compile_play_stats(plays: list[dict]) -> dict:
    """Aggregate play list into summary stats (sim-internal definitions)."""
    n_plays = len(plays)
    pass_plays = [p for p in plays if p.get("is_pass")]
    rush_plays = [p for p in plays if not p.get("is_pass")
                  and p.get("play_result") != "kneel"]
    kneel_plays = [p for p in plays if p.get("play_result") == "kneel"]

    pass_att = len(pass_plays)
    completions = sum(1 for p in pass_plays if p.get("completion"))
    sacks = sum(1 for p in pass_plays if p.get("sack"))
    scrambles = sum(1 for p in pass_plays if p.get("scramble"))
    interceptions = sum(1 for p in pass_plays if p.get("interception"))
    pass_yards = sum(p["yards"] for p in pass_plays
                     if p.get("completion"))
    sack_yards = sum(p["yards"] for p in pass_plays if p.get("sack"))
    scramble_yards = sum(p["yards"] for p in pass_plays
                         if p.get("scramble"))

    rush_att = len(rush_plays)
    rush_yards = sum(p["yards"] for p in rush_plays)
    rush_tds = sum(1 for p in rush_plays if p.get("touchdown"))

    kneels = len(kneel_plays)
    kneel_yards = sum(p["yards"] for p in kneel_plays)

    pass_tds = sum(1 for p in pass_plays if p.get("touchdown"))
    fumbles = sum(1 for p in plays if p.get("fumble"))

    return {
        "total_plays": n_plays,
        "pass_att": pass_att,
        "completions": completions,
        "sacks": sacks,
        "scrambles": scrambles,
        "interceptions": interceptions,
        "pass_yards": pass_yards,
        "sack_yards": sack_yards,
        "scramble_yards": scramble_yards,
        "rush_att": rush_att,
        "rush_yards": rush_yards,
        "rush_tds": rush_tds,
        "kneels": kneels,
        "kneel_yards": kneel_yards,
        "pass_tds": pass_tds,
        "fumbles": fumbles,
        "pass_rate": pass_att / n_plays if n_plays > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo: N simulations
# ══════════════════════════════════════════════════════════════════════════════

def simulate_game_n(n_sims: int,
                    home_profile: dict,
                    away_profile: dict,
                    home_qb: dict | None = None,
                    away_qb: dict | None = None,
                    spread: float = 0.0,
                    total: float = 45.0,
                    seed: int = 42,
                    ) -> dict:
    """Run N game simulations, return distributions.

    Returns: {
        "home_scores": array, "away_scores": array,
        "home_stats_agg": {stat: {mean, sd, p25, p50, p75}},
        "away_stats_agg": {stat: {mean, sd, p25, p50, p75}},
        "n_sims": int,
    }
    """
    rng = np.random.default_rng(seed)

    home_scores = []
    away_scores = []
    home_stats_all = defaultdict(list)
    away_stats_all = defaultdict(list)

    for _ in range(n_sims):
        result = simulate_game(rng, home_profile, away_profile,
                               home_qb, away_qb, spread, total)
        home_scores.append(result["home_score"])
        away_scores.append(result["away_score"])
        for k, v in result["home_stats"].items():
            home_stats_all[k].append(v)
        for k, v in result["away_stats"].items():
            away_stats_all[k].append(v)

    def _agg(vals):
        a = np.array(vals, dtype=float)
        return {
            "mean": float(a.mean()),
            "sd": float(a.std()),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
        }

    return {
        "home_scores": np.array(home_scores),
        "away_scores": np.array(away_scores),
        "home_stats_agg": {k: _agg(v) for k, v in home_stats_all.items()},
        "away_stats_agg": {k: _agg(v) for k, v in away_stats_all.items()},
        "n_sims": n_sims,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Player allocation from share model
# ══════════════════════════════════════════════════════════════════════════════

def allocate_to_players(team_stats: dict,
                        player_shares: dict,
                        rng: np.random.Generator,
                        ) -> dict:
    """Allocate team-level sim stats to individual players.

    Uses the corrected share model output (from compute_player_shares).
    All teammates draw from the SAME team-level stat draw.

    Args:
        team_stats: single sim's team stats (from simulate_game)
        player_shares: output of compute_player_shares()
        rng: random generator for per-target/per-carry variability

    Returns: {player_name: {
        "targets": int,
        "receptions": int,
        "rec_yards": float,
        "rush_att": int,
        "rush_yards": float,
        "rec_tds": int,
        "rush_tds": int,
    }}
    """
    total_pass_att = team_stats["pass_att"]
    total_completions = team_stats["completions"]
    total_pass_yards = team_stats["pass_yards"]
    total_rush_att = team_stats["rush_att"]
    total_rush_yards = team_stats["rush_yards"]

    comp_rate = total_completions / total_pass_att if total_pass_att > 0 else 0.65
    ypc_actual = total_rush_yards / total_rush_att if total_rush_att > 0 else 4.5

    result = {}
    for pname, pdata in player_shares.items():
        if pname == "_OTHER":
            continue

        ts = pdata["pred_target_share"]
        cs = pdata["pred_carry_share"]
        pos = pdata["position"]

        # Targets: binomial draw from total pass attempts
        targets = rng.binomial(max(0, total_pass_att), max(0, min(1, ts)))

        # Receptions: each target caught with team comp rate + player noise
        receptions = rng.binomial(targets, min(1, comp_rate * 1.05))

        # Rec yards: per-reception yards with player-level noise
        if receptions > 0:
            # Mean Y/R from team pass yards / completions, with noise
            team_ypr = total_pass_yards / max(1, total_completions)
            per_rec = rng.normal(team_ypr, team_ypr * 0.3, size=receptions)
            per_rec = np.maximum(per_rec, -5)
            rec_yards = float(per_rec.sum())
        else:
            rec_yards = 0.0

        # Rush attempts: binomial from team rush attempts
        rush_att = 0
        rush_yards = 0.0
        if pos == "RB" and cs > 0:
            rush_att = rng.binomial(max(0, total_rush_att),
                                    max(0, min(1, cs)))
            if rush_att > 0:
                per_carry = rng.normal(ypc_actual, ypc_actual * 0.4,
                                       size=rush_att)
                per_carry = np.maximum(per_carry, -5)
                rush_yards = float(per_carry.sum())

        # TDs: rough allocation proportional to share × team TDs
        team_pass_tds = team_stats.get("pass_tds", 0)
        team_rush_tds = team_stats.get("rush_tds", 0)
        rec_tds = rng.binomial(max(0, team_pass_tds),
                               max(0, min(1, ts * 1.2)))
        rush_tds = 0
        if pos == "RB" and cs > 0:
            rush_tds = rng.binomial(max(0, team_rush_tds),
                                    max(0, min(1, cs * 1.2)))

        result[pname] = {
            "targets": targets,
            "receptions": receptions,
            "rec_yards": rec_yards,
            "rush_att": rush_att,
            "rush_yards": rush_yards,
            "rec_tds": rec_tds,
            "rush_tds": rush_tds,
        }

    # ── Enforce Σ rec_yds ≈ QB pass_yds identity ──
    total_player_rec = sum(r["rec_yards"] for r in result.values())
    if total_player_rec > 0 and total_pass_yards > 0:
        scale = total_pass_yards / total_player_rec
        for r in result.values():
            r["rec_yards"] *= scale

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_sim(seasons: list[int] | None = None,
                 n_sims: int = 300,
                 verbose: bool = True) -> dict:
    """Walk-forward validate game simulation against actuals.

    For each game, builds team profiles from ONLY prior weeks (no lookahead),
    simulates N times, compares distributions to actual outcomes from PBP.

    Reports:
    - Team plays, pass att, pass yds, rush yds, score: MAE, corr, bias
    - Simulated SD vs actual SD (variance calibration)
    - CRPS for continuous metrics
    - Score calibration vs closing spread/total (with optimism note)
    """
    import nfl_data_py as nfl

    if seasons is None:
        seasons = [2023, 2024, 2025]

    # Recalibrate _SIM_LG_MEAN_SCORE from unanchored sims and assert
    # anchored output tracks implied totals.
    print("Calibrating _SIM_LG_MEAN_SCORE from unanchored sims...")
    _calibrate(n_sims=1000)

    print(f"Loading PBP for seasons {seasons}...")
    pbp = nfl.import_pbp_data(seasons)
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # Pre-process (same as build_team_profiles)
    plays["situation"] = plays.apply(
        lambda r: _classify_situation(
            r.get("score_differential", 0) or 0,
            int(r.get("qtr", 1) or 1),
            r.get("game_seconds_remaining", 1800) or 1800),
        axis=1)
    plays.sort_values(["game_id", "game_seconds_remaining"],
                      ascending=[True, False], inplace=True)
    plays["next_gsr"] = plays.groupby(
        ["game_id", "posteam", "fixed_drive"]
    )["game_seconds_remaining"].shift(-1)
    plays["snap_gap"] = plays["game_seconds_remaining"] - plays["next_gsr"]
    plays.loc[(plays["snap_gap"] <= 0) | (plays["snap_gap"] > 55),
              "snap_gap"] = np.nan
    plays["is_dropback"] = ((plays["pass_attempt"] == 1) |
                            (plays["qb_scramble"] == 1) |
                            (plays["sack"] == 1)).astype(float)
    plays["is_completion"] = (
        (plays["pass_attempt"] == 1) & (plays["complete_pass"] == 1)
    ).astype(float)

    if "no_huddle" not in plays.columns:
        plays["no_huddle"] = 0.0

    # ── Compute per-team-game actuals from PBP (prop-settlement defs) ──
    from nfl_stat_definitions import pbp_group_to_prop, sim_stats_to_prop

    print("Computing per-game actuals from PBP (prop-settlement defs)...")
    # Include qb_kneel in actuals (kneels count as rushes for props)
    actuals_plays = pbp[
        pbp["play_type"].isin(["pass", "run", "qb_kneel"]) &
        (pbp["two_point_attempt"] != 1)
    ]
    game_actuals = {}  # (game_id, team) → prop-settlement stats
    for (gid, team), gp in actuals_plays.groupby(["game_id", "posteam"]):
        game_actuals[(gid, team)] = pbp_group_to_prop(gp)

    # ── Get game metadata from nfl.db ──
    db = sqlite3.connect(_NFL_DB)
    db.row_factory = sqlite3.Row
    game_rows = db.execute("""
        SELECT game_id, season, week, home_team, away_team,
               home_score, away_score, spread_line, total_line
        FROM nfl_games
        WHERE season IN ({})
        ORDER BY season, week
    """.format(",".join(str(s) for s in seasons))).fetchall()
    db.close()

    game_meta = {}
    for r in game_rows:
        game_meta[r["game_id"]] = dict(r)

    # ── Walk-forward: build profiles per week, simulate each game ──
    # Group plays by (season, week) for walk-forward cutoff
    plays["_season"] = plays["season"].astype(int)
    plays["_week"] = plays["week"].astype(int)

    # Get unique (season, week) pairs sorted
    sw_pairs = (plays[["_season", "_week"]]
                .drop_duplicates()
                .sort_values(["_season", "_week"])
                .values.tolist())

    records = []
    n_games_done = 0
    n_games_total = len([g for g in game_meta.values()
                        if g["season"] in seasons])

    print(f"Walk-forward validation: {n_games_total} games, "
          f"{n_sims} sims each...")

    for s_idx, (season, week) in enumerate(sw_pairs):
        if week <= 1:
            continue  # need at least 1 prior week

        # Filter PBP to PRIOR games only (walk-forward)
        prior_mask = ((plays["_season"] < season) |
                      ((plays["_season"] == season) & (plays["_week"] < week)))
        prior_plays = plays[prior_mask]

        if len(prior_plays) < 100:
            continue

        # Build profiles from prior data
        profiles = _build_profiles_from_plays(prior_plays)

        # Get games in this week
        week_games = [g for g in game_meta.values()
                      if g["season"] == season and g["week"] == week]

        for game in week_games:
            gid = game["game_id"]
            home = game["home_team"]
            away = game["away_team"]

            hp = profiles.get(home)
            ap = profiles.get(away)
            if not hp or not ap:
                continue

            spread = game.get("spread_line", 0) or 0
            total = game.get("total_line", 45) or 45

            result = simulate_game_n(
                n_sims, hp, ap, spread=spread, total=total,
                seed=42 + n_games_done)

            # Sim draw rate for this game
            sim_draw_rate = float(
                (result["home_scores"] == result["away_scores"]).mean())

            # Compare to actuals for each team
            for side, team in [("home", home), ("away", away)]:
                actual = game_actuals.get((gid, team))
                if not actual:
                    continue

                stats_agg = result[f"{side}_stats_agg"]
                scores = result[f"{side}_scores"]

                actual_score = (game["home_score"] if side == "home"
                                else game["away_score"])
                if actual_score is None:
                    continue

                # Convert sim aggregate stats to prop-settlement defs.
                # sim_stats_to_prop expects raw sim fields; compute
                # prop means from the per-sim aggregates.
                sim_prop_means = {
                    "pass_att":   (stats_agg["pass_att"]["mean"]
                                   - stats_agg["sacks"]["mean"]
                                   - stats_agg["scrambles"]["mean"]),
                    "completions": stats_agg["completions"]["mean"],
                    "pass_yds":   stats_agg["pass_yards"]["mean"],
                    "rush_att":   (stats_agg["rush_att"]["mean"]
                                   + stats_agg["scrambles"]["mean"]
                                   + stats_agg.get("kneels", {}).get(
                                       "mean", 0)),
                    "rush_yds":   (stats_agg["rush_yards"]["mean"]
                                   + stats_agg["scramble_yards"]["mean"]
                                   + stats_agg.get("kneel_yards", {}).get(
                                       "mean", 0)),
                    "pass_tds":   stats_agg["pass_tds"]["mean"],
                    "rush_tds":   stats_agg["rush_tds"]["mean"],
                    "sacks":      stats_agg["sacks"]["mean"],
                    "interceptions": stats_agg["interceptions"]["mean"],
                }

                records.append({
                    "game_id": gid,
                    "team": team,
                    "season": season,
                    "week": week,
                    "side": side,
                    "spread": spread,
                    "total": total,
                    # Actuals (prop-settlement defs via pbp_group_to_prop)
                    "a_plays": actual["total_plays"],
                    "a_pass_att": actual["pass_att"],
                    "a_completions": actual["completions"],
                    "a_pass_yds": actual["pass_yds"],
                    "a_rush_att": actual["rush_att"],
                    "a_rush_yds": actual["rush_yds"],
                    "a_pass_tds": actual["pass_tds"],
                    "a_rush_tds": actual["rush_tds"],
                    "a_sacks": actual["sacks"],
                    "a_interceptions": actual["interceptions"],
                    "a_score": actual_score,
                    # Sim means (prop-settlement defs)
                    "s_plays": stats_agg["total_plays"]["mean"],
                    "s_pass_att": sim_prop_means["pass_att"],
                    "s_completions": sim_prop_means["completions"],
                    "s_pass_yds": sim_prop_means["pass_yds"],
                    "s_rush_att": sim_prop_means["rush_att"],
                    "s_rush_yds": sim_prop_means["rush_yds"],
                    "s_pass_tds": sim_prop_means["pass_tds"],
                    "s_rush_tds": sim_prop_means["rush_tds"],
                    "s_sacks": sim_prop_means["sacks"],
                    "s_interceptions": sim_prop_means["interceptions"],
                    "s_score": scores.mean(),
                    # Sim SDs
                    "s_plays_sd": stats_agg["total_plays"]["sd"],
                    "s_pass_att_sd": stats_agg["pass_att"]["sd"],
                    "s_pass_yds_sd": stats_agg["pass_yards"]["sd"],
                    "s_rush_yds_sd": stats_agg["rush_yards"]["sd"],
                    "s_score_sd": scores.std(),
                    "sim_draw_rate": sim_draw_rate,
                })

            n_games_done += 1
            if n_games_done % 50 == 0:
                print(f"  {n_games_done}/{n_games_total} games done "
                      f"(season {season} week {week})")

    print(f"\nProcessed {len(records)} team-game records "
          f"from {n_games_done} games")

    if not records or not verbose:
        return {"records": records}

    # ── Report ──
    _print_validation_report(records, seasons, n_sims)
    return {"records": records}


def _print_validation_report(records: list[dict], seasons: list[int],
                             n_sims: int):
    """Print formatted validation report."""
    n = len(records)
    n_games = n // 2  # each game produces 2 team-game records

    print(f"\n{'='*75}")
    print("GAME SIMULATION — WALK-FORWARD VALIDATION (prop-settlement defs)")
    print(f"{'='*75}")
    print(f"Seasons: {seasons}, N = {n} team-games ({n_games} games), "
          f"{n_sims} sims/game")
    print("Stat definitions: pass_att=throws only (no sacks), "
          "rush att/yds incl scrambles+kneels")

    # MC SE: standard error of the bias estimate.
    # Per-game sim mean has SE = sim_sd / sqrt(n_sims).
    # Bias SE across games = sd(sim - actual) / sqrt(N).
    score_diffs = np.array([r["s_score"] - r["a_score"]
                            for r in records], dtype=float)
    mc_se_per_game = np.mean([r.get("s_score_sd", 10.0)
                              for r in records]) / sqrt(n_sims)
    bias_se = score_diffs.std() / sqrt(n)
    print(f"MC SE per game: {mc_se_per_game:.2f} pts "
          f"(score SD / sqrt({n_sims}))")
    print(f"Bias SE: {bias_se:.2f} pts (cross-game SD / sqrt({n}))")

    metrics = [
        ("Total plays", "a_plays", "s_plays", "s_plays_sd"),
        ("Pass att (no sacks)", "a_pass_att", "s_pass_att", "s_pass_att_sd"),
        ("Completions", "a_completions", "s_completions", None),
        ("Pass yards (gross)", "a_pass_yds", "s_pass_yds", "s_pass_yds_sd"),
        ("Rush att (+scr+kneel)", "a_rush_att", "s_rush_att", None),
        ("Rush yds (+scr+kneel)", "a_rush_yds", "s_rush_yds", "s_rush_yds_sd"),
        ("Pass TDs", "a_pass_tds", "s_pass_tds", None),
        ("Rush TDs", "a_rush_tds", "s_rush_tds", None),
        ("Sacks", "a_sacks", "s_sacks", None),
        ("Interceptions", "a_interceptions", "s_interceptions", None),
        ("Score", "a_score", "s_score", "s_score_sd"),
    ]

    print(f"\n  {'Stat':<22} {'Sim':>8} {'Actual':>8} {'Bias':>8} "
          f"{'Bias%':>7} {'MAE':>7} {'Corr':>7} {'Sim SD':>8} {'Act SD':>8}")
    print("  " + "-" * 88)

    for label, a_key, s_key, sd_key in metrics:
        a = np.array([r[a_key] for r in records], dtype=float)
        s = np.array([r[s_key] for r in records], dtype=float)
        bias = float(np.mean(s - a))
        bias_pct = bias / a.mean() * 100 if a.mean() != 0 else 0
        mae = float(np.mean(np.abs(s - a)))
        corr = float(np.corrcoef(s, a)[0, 1]) if len(a) > 2 else 0
        sim_sd_str = "—"
        if sd_key:
            sim_sd = np.array([r[sd_key] for r in records]).mean()
            sim_sd_str = f"{sim_sd:8.1f}"
        print(f"  {label:<22} {s.mean():>8.1f} {a.mean():>8.1f} "
              f"{bias:>+8.1f} {bias_pct:>+6.1f}% {mae:>7.1f} {corr:>7.3f} "
              f"{sim_sd_str:>8s} {a.std():>8.1f}")

    # ── Variance calibration ──
    print(f"\n  Variance calibration (sim SD / actual SD):")
    for label, a_key, sd_key in [
        ("Pass att", "a_pass_att", "s_pass_att_sd"),
        ("Pass yards", "a_pass_yds", "s_pass_yds_sd"),
        ("Rush yards", "a_rush_yds", "s_rush_yds_sd"),
        ("Score", "a_score", "s_score_sd"),
    ]:
        a = np.array([r[a_key] for r in records], dtype=float)
        sim_sd_mean = np.array([r[sd_key] for r in records]).mean()
        ratio = sim_sd_mean / a.std() if a.std() > 0 else 0
        target = "✓" if 0.85 <= ratio <= 1.15 else "⚠"
        print(f"    {label:<14} sim={sim_sd_mean:6.1f}  actual={a.std():6.1f}"
              f"  ratio={ratio:.2f} {target}")

    # ── Score vs implied totals ──
    print(f"\n  Score vs implied totals:")
    a_score = np.array([r["a_score"] for r in records], dtype=float)
    s_score = np.array([r["s_score"] for r in records], dtype=float)
    spread = np.array([r["spread"] for r in records], dtype=float)
    total = np.array([r["total"] for r in records], dtype=float)

    implied = np.where(
        [r["side"] == "home" for r in records],
        (total - spread) / 2,
        (total + spread) / 2,
    )
    mae_sim_vs_actual = float(np.mean(np.abs(s_score - a_score)))
    mae_line_vs_actual = float(np.mean(np.abs(implied - a_score)))

    print(f"    Sim mean:       {s_score.mean():.1f}")
    print(f"    Implied mean:   {implied.mean():.1f}")
    print(f"    Actual mean:    {a_score.mean():.1f}")
    print(f"    Sim vs implied: {s_score.mean() - implied.mean():+.1f} "
          f"(sim's own shortfall)")
    print(f"    Impl vs actual: {implied.mean() - a_score.mean():+.1f} "
          f"(closing line runs below actual scoring)")
    print(f"    Sim MAE:        {mae_sim_vs_actual:.1f}")
    print(f"    Line MAE:       {mae_line_vs_actual:.1f}  "
          f"(sim MAE ~ line MAE is mechanical under anchoring)")
    print(f"    Sim-line corr:  "
          f"{float(np.corrcoef(s_score, implied)[0, 1]):.3f}")

    # ── Draws ──
    games = {}
    for r in records:
        gid = r["game_id"]
        if gid not in games:
            games[gid] = {}
        games[gid][r["side"]] = r
    if n_games > 0:
        actual_draws = sum(
            1 for g in games.values()
            if "home" in g and "away" in g
            and g["home"]["a_score"] == g["away"]["a_score"]
        )
        sim_draw_rates = [r["sim_draw_rate"] for r in records
                          if r["side"] == "home" and "sim_draw_rate" in r]
        sim_draw_pct = (np.mean(sim_draw_rates) * 100
                        if sim_draw_rates else 0)
        print(f"\n  Draws: actual {actual_draws}/{n_games} "
              f"({actual_draws/n_games*100:.1f}%), "
              f"sim {sim_draw_pct:.1f}%  (NFL avg ~1.7%)")

    # ── Cross-team correlations (sim means, per game) ──
    # Pair home/away for each game to check structural correlations.
    # SDs 0.035 (eff) and 0.055 (pace) were fit to plays×plays and
    # pass_yds×pass_yds — the other 4 are held-out checks.
    paired = [(g["home"], g["away"]) for g in games.values()
              if "home" in g and "away" in g]
    if len(paired) > 50:
        print(f"\n  Cross-team correlations (sim means, N={len(paired)} games):")
        cross_stats = [
            ("plays × plays",      "s_plays",    "s_plays",    "FITTED"),
            ("pass_att × pass_att", "s_pass_att", "s_pass_att", "held-out"),
            ("pass_yds × pass_yds", "s_pass_yds", "s_pass_yds", "FITTED"),
            ("rush_att × rush_att", "s_rush_att", "s_rush_att", "held-out"),
            ("rush_yds × rush_yds", "s_rush_yds", "s_rush_yds", "held-out"),
            ("score × score",       "s_score",    "s_score",    "held-out"),
        ]
        for label, h_key, a_key, note in cross_stats:
            h_vals = np.array([p[0][h_key] for p in paired])
            a_vals = np.array([p[1][a_key] for p in paired])
            sim_corr = float(np.corrcoef(h_vals, a_vals)[0, 1])
            # Also compute actual cross-team correlation
            h_act_key = "a_" + h_key[2:]
            a_act_key = "a_" + a_key[2:]
            h_act = np.array([p[0][h_act_key] for p in paired])
            a_act = np.array([p[1][a_act_key] for p in paired])
            act_corr = float(np.corrcoef(h_act, a_act)[0, 1])
            print(f"    {label:<22}  sim={sim_corr:+.3f}  "
                  f"actual={act_corr:+.3f}  ({note})")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"

    if cmd == "build-profiles":
        profiles = build_team_profiles()
        print(f"Built profiles for {len(profiles)} teams")
        # Show a sample
        sample = list(profiles.items())[:3]
        for team, p in sample:
            print(f"\n{team} ({p['n_games']} games):")
            print(f"  pace: {p['pace']}")
            print(f"  pass_rate: {p['pass_rate']}")
            print(f"  comp_rate={p['comp_rate']:.3f}, "
                  f"ypa={p['ypa_comp']:.1f}, "
                  f"sack%={p['sack_rate']:.3f}, "
                  f"int%={p['int_rate']:.3f}")

    elif cmd == "quick-test":
        # Quick sanity check: simulate one game with league-average profiles
        lg = {
            "pace": _LG_PACE,
            "pass_rate": _LG_PASS_RATE,
            "no_huddle": _LG_NO_HUDDLE,
            "comp_rate": _LG_COMP_RATE,
            "ypa_comp": _LG_YPA_COMP,
            "ypa_comp_sd": _LG_YPA_COMP_SD,
            "sack_rate": _LG_SACK_RATE,
            "scramble_rate": _LG_SCRAMBLE_RATE,
            "int_rate": _LG_INT_RATE,
            "ypc": _LG_YPC,
            "ypc_sd": _LG_YPC_SD,
            "fumble_rate": _LG_FUMBLE_RATE,
            "n_games": 100,
        }
        # Run 200 sims with detailed drive tracking
        rng = np.random.default_rng(42)
        drive_outcomes = defaultdict(int)
        drive_counts = []
        plays_per_drive_all = []
        all_scores = []
        all_stats = defaultdict(list)
        for _ in range(200):
            res = simulate_game(rng, lg, lg)
            all_scores.append(res["home_score"])
            for k, v in res["home_stats"].items():
                all_stats[k].append(v)
            # Count drives from plays
            n_drives = 0
            cur_drive_plays = 0
            for p in res["home_plays"]:
                if p.get("play_result") == "kneel":
                    continue
                cur_drive_plays += 1
            # Count drives by outcome from the sim itself
            # We need to track outcomes — let's just count TDs and FGs
        # Also run full N sim for stats
        result = simulate_game_n(1000, lg, lg)
        hs = result["home_stats_agg"]
        print(f"1000 sims with league-average profiles:")
        print(f"  Home score: mean={result['home_scores'].mean():.1f}, "
              f"sd={result['home_scores'].std():.1f}")
        tds_per_game = (hs['pass_tds']['mean'] + hs['rush_tds']['mean'])
        fgs_est = (result['home_scores'].mean() - tds_per_game * 7) / 3
        print(f"  TDs/game: {tds_per_game:.1f} (pass={hs['pass_tds']['mean']:.1f},"
              f" rush={hs['rush_tds']['mean']:.1f}), est FGs: {fgs_est:.1f}")
        print(f"  Total plays: {hs['total_plays']['mean']:.1f} "
              f"(sd={hs['total_plays']['sd']:.1f})")
        print(f"  Pass att: {hs['pass_att']['mean']:.1f} "
              f"(sd={hs['pass_att']['sd']:.1f})")
        print(f"  Rush att: {hs['rush_att']['mean']:.1f}")
        print(f"  Pass rate: {hs['pass_rate']['mean']:.3f}")
        print(f"  Completions: {hs['completions']['mean']:.1f}")
        print(f"  Pass yards: {hs['pass_yards']['mean']:.1f}")
        print(f"  Rush yards: {hs['rush_yards']['mean']:.1f}")
        print(f"  Pass TDs: {hs['pass_tds']['mean']:.1f}")
        print(f"  Rush TDs: {hs['rush_tds']['mean']:.1f}")
        print(f"  Sacks: {hs['sacks']['mean']:.1f}")
        print(f"  Scrambles: {hs['scrambles']['mean']:.1f}")
        print(f"  Scramble yds: {hs['scramble_yards']['mean']:.1f}")
        print(f"  INTs: {hs['interceptions']['mean']:.1f}")
        non_sack_att = (hs['pass_att']['mean'] - hs['sacks']['mean']
                        - hs['scrambles']['mean'])
        comp_rate_eff = hs['completions']['mean'] / non_sack_att if non_sack_att > 0 else 0
        print(f"  Non-sack att: {non_sack_att:.1f}, "
              f"eff comp rate: {comp_rate_eff:.3f}")
        combined_rush = hs['rush_yards']['mean'] + hs['scramble_yards']['mean']
        print(f"  Rush+scramble yds: {combined_rush:.1f}")
        # Targets
        print(f"\n  Targets:  score=22.6, plays=62, non-sack-att=32.7, "
              f"comp=21.3, pass_yds=220, rush+scram_yds=122, "
              f"pass_td=1.8, rush_td=0.9, comp_rate=0.652")

    elif cmd == "validate":
        validate_sim()

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: build-profiles, quick-test, validate")
