"""
nfl_tendency_features.py — Pre-game matchup tendency features for NFL player props.

Computes historical profiles from nflverse PBP + FTN + NGS data.
These feed the distributional prop model as pre-game features (mean AND variance).

FEATURE GROUPS:
  1. RB vs Coverage — rushing splits by box count / personnel;
                       receiving splits by man/zone coverage (separate mechanisms)
  2. Blitz Rate     — opponent blitz rate as a VARIANCE driver (sigma effect,
                       not just mean shift); feeds distributional model
  3. MLB Coverage   — MLB allowed production to TE/RB receiving;
                       defensive-personnel feature for TE/RB receiving props
  4. QB Mobility    — scramble rate → drive extension → TEAM-LEVEL volume multiplier
                       (not zero-sum); QB mobility × opponent man-rate interaction

All features are HISTORICAL / PRE-GAME (no live data).
"""
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_CACHE_DB = os.path.join(_DATA_DIR, "nfl_tendencies.db")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading (nflverse via nfl_data_py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    """Load play-by-play data for given seasons."""
    import nfl_data_py as nfl
    logger.info("[tendency] Loading PBP for seasons %s...", seasons)
    pbp = nfl.import_pbp_data(seasons)
    logger.info("[tendency] PBP: %d plays loaded", len(pbp))
    return pbp


def _load_ftn(seasons: list[int]) -> pd.DataFrame:
    """Load FTN charting data (2022+ only)."""
    import nfl_data_py as nfl
    valid = [s for s in seasons if s >= 2022]
    if not valid:
        return pd.DataFrame()
    logger.info("[tendency] Loading FTN for seasons %s...", valid)
    ftn = nfl.import_ftn_data(valid)
    logger.info("[tendency] FTN: %d plays loaded", len(ftn))
    return ftn


def _load_ngs_receiving(seasons: list[int]) -> pd.DataFrame:
    """Load Next Gen Stats receiving data."""
    import nfl_data_py as nfl
    ngs = nfl.import_ngs_data("receiving", seasons)
    logger.info("[tendency] NGS receiving: %d rows", len(ngs))
    return ngs


def _load_ngs_rushing(seasons: list[int]) -> pd.DataFrame:
    """Load Next Gen Stats rushing data."""
    import nfl_data_py as nfl
    ngs = nfl.import_ngs_data("rushing", seasons)
    logger.info("[tendency] NGS rushing: %d rows", len(ngs))
    return ngs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. RB vs COVERAGE — two distinct splits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_rb_rushing_by_box(pbp: pd.DataFrame) -> pd.DataFrame:
    """RB rushing splits by box count / defensive personnel.

    Mechanism: rushing production driven by BOX COUNT — light box
    (nickel/dime, two-high shells) = better lanes; heavy box = worse.

    Returns per-(team, game, box_bucket) rushing production.
    """
    rush = pbp[
        (pbp["rush_attempt"] == 1) &
        (pbp["defenders_in_box"].notna()) &
        (pbp["rushing_yards"].notna())
    ].copy()

    if rush.empty:
        return pd.DataFrame()

    # Box count buckets: light (<=5), standard (6), heavy (7+)
    rush["box_bucket"] = pd.cut(
        rush["defenders_in_box"],
        bins=[-1, 5, 6, 20],
        labels=["light_box", "standard_box", "heavy_box"]
    )

    # Count DBs in defense_personnel for nickel/dime detection
    rush["n_dbs"] = rush["defense_personnel"].apply(_count_dbs)
    rush["personnel_type"] = rush["n_dbs"].apply(
        lambda x: "base" if x <= 4 else ("nickel" if x == 5 else "dime_plus")
    )

    # Per-team defensive profile (how teams DEFEND the run)
    def_profile = rush.groupby(["defteam", "game_id"]).agg(
        avg_box_count=("defenders_in_box", "mean"),
        light_box_rate=("box_bucket", lambda x: (x == "light_box").mean()),
        heavy_box_rate=("box_bucket", lambda x: (x == "heavy_box").mean()),
        nickel_rate=("personnel_type", lambda x: (x == "nickel").mean()),
        dime_rate=("personnel_type", lambda x: (x == "dime_plus").mean()),
        rush_plays_against=("rushing_yards", "count"),
        rush_yds_allowed=("rushing_yards", "sum"),
        rush_ypa_allowed=("rushing_yards", "mean"),
    ).reset_index()

    # Per-team rushing production BY box bucket
    off_by_box = rush.groupby(["posteam", "game_id", "box_bucket"]).agg(
        rush_yds=("rushing_yards", "sum"),
        rush_att=("rushing_yards", "count"),
        rush_ypa=("rushing_yards", "mean"),
        rush_yds_std=("rushing_yards", "std"),
    ).reset_index()

    return def_profile, off_by_box


def compute_rb_receiving_by_coverage(pbp: pd.DataFrame) -> pd.DataFrame:
    """RB receiving splits by coverage type (man vs zone).

    Mechanism: zone coverage produces more checkdowns and RB targets
    than man coverage. Split RB receiving by coverage type.

    Returns per-(defteam, game, coverage) RB receiving production.
    """
    # RB targets/receptions
    rec = pbp[
        (pbp["pass_attempt"] == 1) &
        (pbp["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"])) &
        (pbp["receiver_player_name"].notna())
    ].copy()

    if rec.empty:
        return pd.DataFrame()

    # We need receiver position — join from roster or infer from position
    # PBP doesn't have receiver position directly, but we can use
    # rusher_player_id vs receiver_player_id + weekly rosters
    # Simpler: check if yards_after_catch, air_yards patterns indicate RB targets
    # For now: tag based on air_yards (RBs typically short targets)
    # Better approach: use receiver_player_id to join with rosters
    rec["is_rb_target"] = rec["air_yards"].fillna(0) <= 0  # Screens/checkdowns

    # Defense profile: checkdown rate by coverage type
    def_cov = rec.groupby(["defteam", "game_id", "defense_man_zone_type"]).agg(
        targets=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        yards=("yards_gained", "sum"),
        rb_targets=("is_rb_target", "sum"),
        rb_target_rate=("is_rb_target", "mean"),
    ).reset_index()

    return def_cov


def _count_dbs(personnel_str) -> int:
    """Count DBs (CB + FS + SS + S + DB) in defense_personnel string."""
    if not personnel_str or not isinstance(personnel_str, str):
        return 4  # default base
    count = 0
    for part in personnel_str.split(","):
        part = part.strip()
        if any(pos in part for pos in ["CB", "FS", "SS", " S ", "DB"]):
            try:
                n = int(part.split()[0])
                count += n
            except (ValueError, IndexError):
                pass
    return count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BLITZ RATE — Variance driver (distributional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_blitz_profiles(pbp: pd.DataFrame, ftn: pd.DataFrame = None) -> pd.DataFrame:
    """Compute opponent blitz rate as a VARIANCE effect.

    Key insight: high-blitz defenses cause more pressure (mean down)
    BUT more explosive plays when beaten (tails fatten → higher sigma).
    This is exactly what generic book alt-ladder formulas miss.

    Returns per-(defteam, game) blitz profile with mean AND variance effects.
    """
    pass_plays = pbp[
        (pbp["pass_attempt"] == 1) &
        (pbp["number_of_pass_rushers"].notna())
    ].copy()

    if pass_plays.empty:
        return pd.DataFrame()

    # Blitz = 5+ pass rushers
    pass_plays["is_blitz"] = pass_plays["number_of_pass_rushers"] >= 5

    # Join FTN blitz data if available (more accurate charted blitz)
    if ftn is not None and not ftn.empty and "n_blitzers" in ftn.columns:
        ftn_blitz = ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers"]].copy()
        ftn_blitz = ftn_blitz.rename(columns={
            "nflverse_game_id": "game_id",
            "nflverse_play_id": "play_id",
        })
        pass_plays = pass_plays.merge(
            ftn_blitz, on=["game_id", "play_id"], how="left"
        )
        # Use FTN blitz when available, fallback to PBP
        has_ftn = pass_plays["n_blitzers"].notna()
        pass_plays.loc[has_ftn, "is_blitz"] = pass_plays.loc[has_ftn, "n_blitzers"] >= 1

    # Per-team defensive blitz profile
    blitz_profile = pass_plays.groupby(["defteam", "game_id"]).agg(
        blitz_rate=("is_blitz", "mean"),
        avg_pass_rushers=("number_of_pass_rushers", "mean"),
        pass_plays_n=("pass_attempt", "sum"),
    ).reset_index()

    # Outcome VARIANCE on blitz vs non-blitz (the key distributional insight)
    for label, mask_val in [("blitz", True), ("no_blitz", False)]:
        subset = pass_plays[pass_plays["is_blitz"] == mask_val]
        outcome = subset.groupby(["defteam", "game_id"]).agg(
            **{
                f"{label}_yds_mean": ("yards_gained", "mean"),
                f"{label}_yds_std": ("yards_gained", "std"),
                f"{label}_explosive": ("yards_gained", lambda x: (x >= 20).mean()),
                f"{label}_sack_rate": ("sack", "mean"),
                f"{label}_n": ("yards_gained", "count"),
            }
        ).reset_index()
        blitz_profile = blitz_profile.merge(outcome, on=["defteam", "game_id"], how="left")

    # Compute variance ratio: std_blitz / std_no_blitz
    # > 1 means blitz widens outcomes (the distributional signal)
    blitz_profile["variance_ratio"] = (
        blitz_profile["blitz_yds_std"].fillna(0) /
        blitz_profile["no_blitz_yds_std"].replace(0, np.nan)
    )

    # Sigma multiplier: how much to inflate variance when facing high-blitz D
    # Smoothed team-season average
    blitz_profile["sigma_multiplier"] = blitz_profile["variance_ratio"].clip(0.5, 2.0)

    return blitz_profile


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. MLB COVERAGE TENDENCIES — TE/RB receiving feature
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_mlb_coverage_allowed(pbp: pd.DataFrame, rosters: pd.DataFrame = None) -> pd.DataFrame:
    """MLB coverage tendencies: how much TE/RB receiving production each defense allows.

    Mechanism: poor-coverage MLBs create mismatches for receiving TEs and pass-catching RBs.
    This is a nameable mismatch that generic projections miss.

    Approximation: since we can't identify specific MLB coverage snaps without tracking data,
    we proxy via defense_personnel (ILB/MLB count) × coverage type × TE/RB receiving production.

    Returns per-(defteam, game) TE/RB receiving allowed profiles.
    """
    rec = pbp[
        (pbp["pass_attempt"] == 1) &
        (pbp["receiver_player_name"].notna()) &
        (pbp["yards_gained"].notna())
    ].copy()

    if rec.empty:
        return pd.DataFrame()

    # Identify TE/RB targets by air yards and target depth patterns
    # RB: typically air_yards <= 2 (screens, checkdowns)
    # TE: typically air_yards 3-15 (seams, crossing routes)
    # This is approximate — better with roster position joins
    rec["target_depth"] = rec["air_yards"].fillna(0)
    rec["is_short_target"] = rec["target_depth"] <= 5  # Likely covered by LB/MLB

    # Count ILBs/MLBs in defense personnel
    rec["n_lbs"] = rec["defense_personnel"].apply(_count_lbs)

    # MLB coverage allowed: short/middle target production by defense
    mlb_profile = rec[rec["is_short_target"]].groupby(["defteam", "game_id"]).agg(
        short_targets_allowed=("pass_attempt", "sum"),
        short_completions=("complete_pass", "sum"),
        short_yards_allowed=("yards_gained", "sum"),
        short_yds_per_target=("yards_gained", "mean"),
        short_yac=("yards_after_catch", "mean"),
        avg_n_lbs=("n_lbs", "mean"),
    ).reset_index()

    # Zone vs man coverage split for short targets
    cov_rec = rec[
        (rec["is_short_target"]) &
        (rec["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"]))
    ]
    if not cov_rec.empty:
        cov_profile = cov_rec.groupby(["defteam", "game_id", "defense_man_zone_type"]).agg(
            targets=("pass_attempt", "sum"),
            yards=("yards_gained", "sum"),
            yds_per_target=("yards_gained", "mean"),
        ).reset_index()
        # Pivot for zone_short_yds vs man_short_yds
        cov_pivot = cov_profile.pivot_table(
            index=["defteam", "game_id"],
            columns="defense_man_zone_type",
            values="yds_per_target",
            aggfunc="mean"
        ).reset_index()
        cov_pivot.columns = ["defteam", "game_id", "man_short_ypt", "zone_short_ypt"]
        mlb_profile = mlb_profile.merge(cov_pivot, on=["defteam", "game_id"], how="left")

    return mlb_profile


def _count_lbs(personnel_str) -> int:
    """Count LBs (ILB + MLB + OLB + LB) in defense_personnel string."""
    if not personnel_str or not isinstance(personnel_str, str):
        return 2
    count = 0
    for part in personnel_str.split(","):
        part = part.strip()
        if any(pos in part for pos in ["ILB", "MLB", "OLB", " LB"]):
            try:
                n = int(part.split()[0])
                count += n
            except (ValueError, IndexError):
                pass
    return count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. QB MOBILITY → Drive Extension → TEAM-WIDE Volume Multiplier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_qb_mobility_multiplier(pbp: pd.DataFrame) -> pd.DataFrame:
    """QB mobility as a team-wide volume multiplier.

    THE INSIGHT: Most models treat mobile QB as ZERO-SUM (QB rush up,
    RB share down). Wrong. Scrambles CONVERT 3rd-and-long into first
    downs → drives extend → MORE TOTAL PLAYS → more volume for EVERY
    skill player on that team.

    So QB mobility / scramble rate feeds a DRIVE-SUSTAIN estimate that
    acts as a MULTIPLIER on every teammate's prop projection.

    Also: QB mobility × opponent MAN-COVERAGE RATE interaction.
    Against man, defenders have backs turned → scramble lanes open.
    Mobile QB vs man-heavy defense = extended drives + broken coverage.

    Returns per-(team, game) mobility profile with volume multiplier.
    """
    if pbp.empty:
        return pd.DataFrame()

    # QB scramble rate by team
    pass_plays = pbp[
        (pbp["pass_attempt"] == 1) | (pbp["qb_scramble"] == 1)
    ].copy()

    team_games = pass_plays.groupby(["posteam", "game_id"]).agg(
        scramble_rate=("qb_scramble", "mean"),
        scramble_count=("qb_scramble", "sum"),
        total_dropbacks=("qb_scramble", "count"),
    ).reset_index()

    # 3rd down conversion rate (drive-sustain proxy)
    third_downs = pbp[pbp["down"] == 3].copy()
    td_conv = third_downs.groupby(["posteam", "game_id"]).agg(
        third_down_att=("third_down_converted", "count"),
        third_down_conv=("third_down_converted", "sum"),
        third_down_rate=("third_down_converted", "mean"),
    ).reset_index()
    team_games = team_games.merge(td_conv, on=["posteam", "game_id"], how="left")

    # Total plays per game (the volume outcome we're predicting)
    plays_per_game = pbp[
        pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])
    ].groupby(["posteam", "game_id"]).size().reset_index(name="total_plays")
    team_games = team_games.merge(plays_per_game, on=["posteam", "game_id"], how="left")

    # Drive length (plays per drive)
    drives = pbp.groupby(["posteam", "game_id", "drive"]).agg(
        drive_plays=("play_id", "count"),
    ).reset_index()
    avg_drive = drives.groupby(["posteam", "game_id"]).agg(
        avg_drive_length=("drive_plays", "mean"),
        n_drives=("drive", "count"),
    ).reset_index()
    team_games = team_games.merge(avg_drive, on=["posteam", "game_id"], how="left")

    # QB MOBILITY × OPPONENT MAN-COVERAGE RATE interaction
    # Get opponent man-coverage rate
    opp_man_rate = pbp[
        (pbp["pass_attempt"] == 1) &
        (pbp["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"]))
    ].groupby(["defteam", "game_id"]).agg(
        opp_man_rate=("defense_man_zone_type", lambda x: (x == "MAN_COVERAGE").mean()),
    ).reset_index().rename(columns={"defteam": "opponent"})

    # To join, we need the opponent for each team in each game
    game_teams = pbp[["game_id", "home_team", "away_team"]].drop_duplicates()
    # Create opponent mapping
    home = game_teams[["game_id", "home_team", "away_team"]].rename(
        columns={"home_team": "posteam", "away_team": "opponent"})
    away = game_teams[["game_id", "home_team", "away_team"]].rename(
        columns={"away_team": "posteam", "home_team": "opponent"})
    team_opp = pd.concat([home, away]).drop_duplicates()

    team_games = team_games.merge(team_opp, on=["posteam", "game_id"], how="left")
    team_games = team_games.merge(opp_man_rate, on=["opponent", "game_id"], how="left")

    # Interaction: scramble_rate × opp_man_rate
    team_games["mobility_x_man"] = (
        team_games["scramble_rate"].fillna(0) * team_games["opp_man_rate"].fillna(0.5)
    )

    # Volume multiplier: normalize total_plays to league average
    league_avg_plays = team_games["total_plays"].mean()
    team_games["volume_multiplier"] = team_games["total_plays"] / league_avg_plays

    return team_games


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. PRESSURE CHAIN — QB performance under pressure, matchup-specific
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_ngs_passing(seasons: list[int]) -> pd.DataFrame:
    """Load Next Gen Stats passing data (avg_time_to_throw, aggressiveness)."""
    import nfl_data_py as nfl
    ngs = nfl.import_ngs_data("passing", seasons)
    logger.info("[tendency] NGS passing: %d rows", len(ngs))
    return ngs


def compute_pressure_chain(pbp: pd.DataFrame,
                           ngs_passing: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pressure/time-to-throw feature chain for QB props.

    Returns (qb_pressure_profile, def_pressure_profile):
      - qb_pressure_profile: per-(passer, game) clean/pressure splits
      - def_pressure_profile: per-(defteam, game) pressure generation rates

    Mechanism: Pressure degrades completion %, inflates INT rate, suppresses
    TD rate.  The split magnitudes are QB-specific (some handle pressure well).
    Matchup-specific pressure rate = opponent generation × team OL allowed rate.
    """
    dropbacks = pbp[
        (pbp["qb_dropback"] == 1) &
        (pbp["passer_player_name"].notna())
    ].copy()

    if dropbacks.empty:
        return pd.DataFrame(), pd.DataFrame()

    dropbacks["was_pressure"] = dropbacks["was_pressure"].fillna(False).astype(bool)

    # ── Per-QB offensive pressure profile ──
    def _qb_split(grp):
        n = len(grp)
        pressured = grp[grp["was_pressure"]]
        clean = grp[~grp["was_pressure"]]
        n_p, n_c = len(pressured), len(clean)
        return pd.Series({
            "dropbacks": n,
            "pressure_rate": n_p / n if n else 0,
            "sack_rate": grp["sack"].sum() / n if n else 0,
            "comp_pct_clean": clean["complete_pass"].mean() if n_c else 0,
            "comp_pct_pressure": pressured["complete_pass"].mean() if n_p else 0,
            "ya_clean": clean["yards_gained"].mean() if n_c else 0,
            "ya_pressure": pressured["yards_gained"].mean() if n_p else 0,
            "int_rate_clean": clean["interception"].mean() if n_c else 0,
            "int_rate_pressure": pressured["interception"].mean() if n_p else 0,
            "td_rate_clean": clean["pass_touchdown"].mean() if n_c > 0 and "pass_touchdown" in clean.columns else 0,
            "td_rate_pressure": pressured["pass_touchdown"].mean() if n_p > 0 and "pass_touchdown" in pressured.columns else 0,
        })

    qb_profile = dropbacks.groupby(["posteam", "game_id", "passer_player_name"]).apply(
        _qb_split
    ).reset_index()

    # Pressure INT uplift: how much worse each QB's INT rate gets under pressure
    qb_profile["pressure_int_uplift"] = np.where(
        qb_profile["int_rate_clean"] > 0,
        qb_profile["int_rate_pressure"] / qb_profile["int_rate_clean"],
        1.0,
    )
    qb_profile["pressure_int_uplift"] = qb_profile["pressure_int_uplift"].clip(0.5, 5.0)

    # Merge NGS avg_time_to_throw if available
    if ngs_passing is not None and not ngs_passing.empty:
        ngs_ttt = ngs_passing.groupby("team_abbr").agg(
            avg_time_to_throw=("avg_time_to_throw", "mean"),
            aggressiveness=("aggressiveness", "mean"),
        ).reset_index()
        qb_profile = qb_profile.merge(
            ngs_ttt, left_on="posteam", right_on="team_abbr", how="left"
        ).drop(columns=["team_abbr"], errors="ignore")

    # ── Per-team defensive pressure generation profile ──
    def_profile = dropbacks.groupby(["defteam", "game_id"]).apply(
        lambda g: pd.Series({
            "opp_dropbacks": len(g),
            "pressure_rate_generated": g["was_pressure"].mean(),
            "sack_rate_generated": g["sack"].mean(),
            "qb_hit_rate": g["qb_hit"].mean() if "qb_hit" in g.columns else 0,
        })
    ).reset_index()

    return qb_profile, def_profile


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. COVERAGE SPLITS — Per-QB man vs zone performance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_coverage_splits(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-QB man/zone performance splits + per-team defensive coverage mix.

    Returns (qb_coverage_splits, def_coverage_mix):
      - qb_coverage_splits: per-(passer, game) comp%/Y-A/INT/TD rate by man/zone
      - def_coverage_mix: per-(defteam, game) fraction man vs zone

    Matchup weighting: QB's coverage-weighted INT/TD rate uses opponent's
    man/zone mix to produce matchup-specific expectations.
    """
    passes = pbp[
        (pbp["pass_attempt"] == 1) &
        (pbp["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"])) &
        (pbp["passer_player_name"].notna())
    ].copy()

    if passes.empty:
        return pd.DataFrame(), pd.DataFrame()

    passes["is_man"] = passes["defense_man_zone_type"] == "MAN_COVERAGE"

    # ── Per-QB splits ──
    def _cov_split(grp):
        man = grp[grp["is_man"]]
        zone = grp[~grp["is_man"]]
        n_m, n_z = len(man), len(zone)
        out = {
            "pass_n": len(grp),
            "man_n": n_m, "zone_n": n_z,
        }
        for label, sub, n in [("man", man, n_m), ("zone", zone, n_z)]:
            out[f"comp_pct_{label}"] = sub["complete_pass"].mean() if n else 0
            out[f"ya_{label}"] = sub["yards_gained"].mean() if n else 0
            out[f"int_rate_{label}"] = sub["interception"].mean() if n else 0
            td_col = "pass_touchdown" if "pass_touchdown" in sub.columns else "touchdown"
            out[f"td_rate_{label}"] = sub[td_col].mean() if n and td_col in sub.columns else 0
        return pd.Series(out)

    qb_splits = passes.groupby(["posteam", "game_id", "passer_player_name"]).apply(
        _cov_split
    ).reset_index()

    # ── Per-team defensive coverage mix ──
    def_mix = passes.groupby(["defteam", "game_id"]).agg(
        cov_plays=("is_man", "count"),
        man_rate=("is_man", "mean"),
    ).reset_index()
    def_mix["zone_rate"] = 1.0 - def_mix["man_rate"]

    return qb_splits, def_mix


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. RED ZONE FEATURES — Trip-level RZ offense and defense
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_rz_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """Red-zone trip-level features for both offense and defense.

    Returns per-(team, game) with:
      - rz_trips, rz_tds, rz_td_rate (offensive)
      - rz_trips_allowed, rz_td_rate_allowed (defensive)
      - goalline_rate: fraction of RZ trips reaching ≤5 yardline

    These mediate the yards→TDs relationship: high-RZ-trip teams convert
    yards into TDs at higher rates.  Opponent RZ defense directly adjusts
    the pass_tds lambda.
    """
    # Red zone = yardline_100 <= 20
    rz_plays = pbp[
        (pbp["yardline_100"].notna()) &
        (pbp["yardline_100"] <= 20) &
        (pbp["play_type"].isin(["pass", "run"]))
    ].copy()

    if rz_plays.empty:
        return pd.DataFrame()

    # Identify distinct RZ trips by (team, game, drive)
    rz_drives = rz_plays.groupby(["posteam", "game_id", "drive"]).agg(
        rz_plays=("play_id", "count"),
        scored_td=("touchdown", "max"),
        min_yardline=("yardline_100", "min"),
    ).reset_index()
    rz_drives["reached_goalline"] = rz_drives["min_yardline"] <= 5

    # Offensive RZ profile per game
    off_rz = rz_drives.groupby(["posteam", "game_id"]).agg(
        rz_trips=("drive", "count"),
        rz_tds=("scored_td", "sum"),
        goalline_trips=("reached_goalline", "sum"),
    ).reset_index()
    off_rz["rz_td_rate"] = off_rz["rz_tds"] / off_rz["rz_trips"].replace(0, 1)
    off_rz["goalline_rate"] = off_rz["goalline_trips"] / off_rz["rz_trips"].replace(0, 1)

    # Defensive RZ profile: opponent's drives into YOUR red zone
    # posteam is the OFFENSE — so for defense, we look at who ALLOWED those trips
    game_teams = pbp[["game_id", "home_team", "away_team"]].drop_duplicates()
    home = game_teams.rename(columns={"home_team": "posteam_x", "away_team": "defteam_x"})
    away = game_teams.rename(columns={"away_team": "posteam_x", "home_team": "defteam_x"})
    opp_map = pd.concat([
        home[["game_id", "posteam_x", "defteam_x"]],
        away[["game_id", "posteam_x", "defteam_x"]],
    ]).drop_duplicates()

    def_rz = off_rz.merge(
        opp_map, left_on=["game_id", "posteam"], right_on=["game_id", "posteam_x"], how="left"
    )
    def_rz_agg = def_rz.groupby(["defteam_x", "game_id"]).agg(
        rz_trips_allowed=("rz_trips", "sum"),
        rz_tds_allowed=("rz_tds", "sum"),
    ).reset_index().rename(columns={"defteam_x": "defteam"})
    def_rz_agg["rz_td_rate_allowed"] = (
        def_rz_agg["rz_tds_allowed"] / def_rz_agg["rz_trips_allowed"].replace(0, 1)
    )

    return off_rz, def_rz_agg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. SEPARATION FEATURES — NGS receiver separation by team
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_separation_features(ngs_receiving: pd.DataFrame) -> pd.DataFrame:
    """Team-average receiver separation from NGS data.

    Returns per-team avg_separation — a proxy for how much cushion
    receivers create, which feeds completion probability and YAC.
    """
    if ngs_receiving is None or ngs_receiving.empty:
        return pd.DataFrame()

    if "avg_separation" not in ngs_receiving.columns:
        logger.warning("[tendency] NGS receiving missing avg_separation")
        return pd.DataFrame()

    team_sep = ngs_receiving.groupby("team_abbr").agg(
        avg_separation=("avg_separation", "mean"),
    ).reset_index()

    logger.info("[tendency] Separation features: %d teams", len(team_sep))
    return team_sep


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. FG ATTEMPT FEATURES — Kicker model inputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_fg_attempt_features(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Field goal attempt features for kicker prop model.

    Returns (off_fg_profile, def_fg_profile):
      - off_fg_profile: per-(posteam, game) FG attempts, distance, accuracy by band
      - def_fg_profile: per-(defteam, game) FG attempts forced, bend-don't-break score

    Stage 1 of the kicker model: expected FG attempts driven by opponent
    red-zone profile (bend-don't-break D = lots of drives stall in FG range).
    """
    fg_plays = pbp[
        (pbp["field_goal_attempt"] == 1)
    ].copy()

    all_drives = pbp[
        pbp["play_type"].isin(["pass", "run", "field_goal", "punt", "no_play"])
    ].copy()

    if fg_plays.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Distance bands
    fg_plays["kick_distance"] = fg_plays["kick_distance"].fillna(
        fg_plays["yardline_100"].fillna(30) + 17  # estimate from yardline
    )
    fg_plays["fg_made"] = (fg_plays["field_goal_result"] == "made").astype(int)
    fg_plays["dist_band"] = pd.cut(
        fg_plays["kick_distance"],
        bins=[0, 39, 49, 70],
        labels=["under_40", "40_49", "50_plus"],
    )

    # ── Offensive FG profile ──
    off_fg = fg_plays.groupby(["posteam", "game_id"]).agg(
        fg_att=("field_goal_attempt", "sum"),
        fg_made=("fg_made", "sum"),
        avg_fg_distance=("kick_distance", "mean"),
    ).reset_index()
    off_fg["fg_pct"] = off_fg["fg_made"] / off_fg["fg_att"].replace(0, 1)

    # Accuracy by distance band
    for band in ["under_40", "40_49", "50_plus"]:
        band_data = fg_plays[fg_plays["dist_band"] == band]
        band_agg = band_data.groupby(["posteam", "game_id"]).agg(
            **{f"fg_att_{band}": ("field_goal_attempt", "sum"),
               f"fg_pct_{band}": ("fg_made", "mean")},
        ).reset_index()
        off_fg = off_fg.merge(band_agg, on=["posteam", "game_id"], how="left")

    # ── Defensive FG profile (opponent's kicks against this D) ──
    def_fg = fg_plays.groupby(["defteam", "game_id"]).agg(
        fg_att_forced=("field_goal_attempt", "sum"),
        fg_avg_dist_forced=("kick_distance", "mean"),
    ).reset_index()

    return off_fg, def_fg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregation: Build defensive tendency profiles for upcoming matchups
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_team_tendency_profiles(seasons: list[int] = None,
                                  n_recent_games: int = 8) -> dict:
    """Build rolling tendency profiles for all NFL teams.

    Returns dict of DataFrames keyed by feature group:
        "rb_box": per-team defensive box count profile
        "blitz": per-team blitz profile with variance effects
        "mlb_coverage": per-team MLB coverage allowed profile
        "qb_mobility": per-team QB mobility + volume multiplier

    Each profile is a ROLLING AVERAGE over the last n_recent_games,
    so it can be used as a pre-game feature for upcoming matchups.
    """
    if seasons is None:
        current_year = datetime.now().year
        seasons = [current_year - 1, current_year]

    logger.info("[tendency] Building profiles for seasons %s (last %d games)",
                seasons, n_recent_games)

    # Load data
    pbp = _load_pbp(seasons)
    ftn = _load_ftn(seasons)
    try:
        ngs_rec = _load_ngs_receiving(seasons)
    except Exception as exc:
        logger.warning("[tendency] NGS receiving load failed: %s", exc)
        ngs_rec = pd.DataFrame()
    try:
        ngs_pass = _load_ngs_passing(seasons)
    except Exception as exc:
        logger.warning("[tendency] NGS passing load failed: %s", exc)
        ngs_pass = pd.DataFrame()

    # Extract week from game_id for ordering
    pbp["season"] = pbp["season"].astype(int)
    pbp["week"] = pbp["week"].astype(int)

    profiles = {}

    # 1. RB Box Count
    try:
        def_box, off_box = compute_rb_rushing_by_box(pbp)
        # Rolling average for defensive tendencies
        if not def_box.empty:
            def_box = def_box.sort_values(["defteam", "game_id"])
            rolling = def_box.groupby("defteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "avg_box_count": "mean",
                    "light_box_rate": "mean",
                    "heavy_box_rate": "mean",
                    "nickel_rate": "mean",
                    "rush_ypa_allowed": "mean",
                    "rush_plays_against": "mean",
                })
            ).reset_index()
            profiles["rb_box"] = rolling
            logger.info("[tendency] RB box profiles: %d teams", len(rolling))
    except Exception as exc:
        logger.warning("[tendency] RB box failed: %s", exc)

    # 2. Blitz Profiles
    try:
        blitz = compute_blitz_profiles(pbp, ftn)
        if not blitz.empty:
            blitz = blitz.sort_values(["defteam", "game_id"])
            rolling = blitz.groupby("defteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "blitz_rate": "mean",
                    "avg_pass_rushers": "mean",
                    "variance_ratio": "mean",
                    "sigma_multiplier": "mean",
                    "blitz_explosive": "mean",
                    "no_blitz_explosive": "mean",
                })
            ).reset_index()
            profiles["blitz"] = rolling
            logger.info("[tendency] Blitz profiles: %d teams", len(rolling))
    except Exception as exc:
        logger.warning("[tendency] Blitz failed: %s", exc)

    # 3. MLB Coverage
    try:
        mlb = compute_mlb_coverage_allowed(pbp)
        if not mlb.empty:
            mlb = mlb.sort_values(["defteam", "game_id"])
            rolling = mlb.groupby("defteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "short_targets_allowed": "mean",
                    "short_yards_allowed": "mean",
                    "short_yds_per_target": "mean",
                    "short_yac": "mean",
                    "avg_n_lbs": "mean",
                })
            ).reset_index()
            profiles["mlb_coverage"] = rolling
            logger.info("[tendency] MLB coverage profiles: %d teams", len(rolling))
    except Exception as exc:
        logger.warning("[tendency] MLB coverage failed: %s", exc)

    # 4. QB Mobility
    try:
        qb_mob = compute_qb_mobility_multiplier(pbp)
        if not qb_mob.empty:
            qb_mob = qb_mob.sort_values(["posteam", "game_id"])
            rolling = qb_mob.groupby("posteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "scramble_rate": "mean",
                    "third_down_rate": "mean",
                    "total_plays": "mean",
                    "avg_drive_length": "mean",
                    "volume_multiplier": "mean",
                    "opp_man_rate": "mean",
                    "mobility_x_man": "mean",
                })
            ).reset_index()
            profiles["qb_mobility"] = rolling
            logger.info("[tendency] QB mobility profiles: %d teams", len(rolling))
    except Exception as exc:
        logger.warning("[tendency] QB mobility failed: %s", exc)

    # 5. Pressure Chain
    try:
        qb_press, def_press = compute_pressure_chain(pbp, ngs_pass)
        if not qb_press.empty:
            qb_press = qb_press.sort_values(["posteam", "game_id"])
            rolling_qb = qb_press.groupby("posteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "pressure_rate": "mean",
                    "sack_rate": "mean",
                    "comp_pct_clean": "mean",
                    "comp_pct_pressure": "mean",
                    "ya_clean": "mean",
                    "ya_pressure": "mean",
                    "int_rate_clean": "mean",
                    "int_rate_pressure": "mean",
                    "td_rate_clean": "mean",
                    "td_rate_pressure": "mean",
                    "pressure_int_uplift": "mean",
                })
            ).reset_index()
            profiles["pressure_qb"] = rolling_qb
            logger.info("[tendency] Pressure QB profiles: %d teams", len(rolling_qb))
        if not def_press.empty:
            def_press = def_press.sort_values(["defteam", "game_id"])
            rolling_def = def_press.groupby("defteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "pressure_rate_generated": "mean",
                    "sack_rate_generated": "mean",
                })
            ).reset_index()
            profiles["pressure_def"] = rolling_def
            logger.info("[tendency] Pressure DEF profiles: %d teams", len(rolling_def))
    except Exception as exc:
        logger.warning("[tendency] Pressure chain failed: %s", exc)

    # 6. Coverage Splits
    try:
        qb_cov, def_cov = compute_coverage_splits(pbp)
        if not qb_cov.empty:
            qb_cov = qb_cov.sort_values(["posteam", "game_id"])
            rolling_cov = qb_cov.groupby("posteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "comp_pct_man": "mean",
                    "comp_pct_zone": "mean",
                    "ya_man": "mean",
                    "ya_zone": "mean",
                    "int_rate_man": "mean",
                    "int_rate_zone": "mean",
                    "td_rate_man": "mean",
                    "td_rate_zone": "mean",
                })
            ).reset_index()
            profiles["coverage_qb"] = rolling_cov
            logger.info("[tendency] Coverage QB profiles: %d teams", len(rolling_cov))
        if not def_cov.empty:
            def_cov = def_cov.sort_values(["defteam", "game_id"])
            rolling_def_cov = def_cov.groupby("defteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "man_rate": "mean",
                    "zone_rate": "mean",
                })
            ).reset_index()
            profiles["coverage_def"] = rolling_def_cov
            logger.info("[tendency] Coverage DEF profiles: %d teams", len(rolling_def_cov))
    except Exception as exc:
        logger.warning("[tendency] Coverage splits failed: %s", exc)

    # 7. Red Zone Features
    try:
        off_rz, def_rz = compute_rz_features(pbp)
        if not isinstance(off_rz, pd.DataFrame) or off_rz.empty:
            pass
        else:
            off_rz = off_rz.sort_values(["posteam", "game_id"])
            rolling_off_rz = off_rz.groupby("posteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "rz_trips": "mean",
                    "rz_td_rate": "mean",
                    "goalline_rate": "mean",
                })
            ).reset_index()
            profiles["redzone_off"] = rolling_off_rz
            logger.info("[tendency] RedZone OFF profiles: %d teams", len(rolling_off_rz))
        if not isinstance(def_rz, pd.DataFrame) or def_rz.empty:
            pass
        else:
            def_rz = def_rz.sort_values(["defteam", "game_id"])
            rolling_def_rz = def_rz.groupby("defteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "rz_trips_allowed": "mean",
                    "rz_td_rate_allowed": "mean",
                })
            ).reset_index()
            profiles["redzone_def"] = rolling_def_rz
            logger.info("[tendency] RedZone DEF profiles: %d teams", len(rolling_def_rz))
    except Exception as exc:
        logger.warning("[tendency] Red zone features failed: %s", exc)

    # 8. Separation Features
    try:
        sep = compute_separation_features(ngs_rec)
        if not sep.empty:
            profiles["separation"] = sep
            logger.info("[tendency] Separation profiles: %d teams", len(sep))
    except Exception as exc:
        logger.warning("[tendency] Separation features failed: %s", exc)

    # 9. FG Attempt Features
    try:
        off_fg, def_fg = compute_fg_attempt_features(pbp)
        if not isinstance(off_fg, pd.DataFrame) or off_fg.empty:
            pass
        else:
            off_fg = off_fg.sort_values(["posteam", "game_id"])
            rolling_off_fg = off_fg.groupby("posteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "fg_att": "mean",
                    "fg_pct": "mean",
                    "avg_fg_distance": "mean",
                })
            ).reset_index()
            profiles["fg_off"] = rolling_off_fg
            logger.info("[tendency] FG OFF profiles: %d teams", len(rolling_off_fg))
        if not isinstance(def_fg, pd.DataFrame) or def_fg.empty:
            pass
        else:
            def_fg = def_fg.sort_values(["defteam", "game_id"])
            rolling_def_fg = def_fg.groupby("defteam").apply(
                lambda g: g.tail(n_recent_games).agg({
                    "fg_att_forced": "mean",
                })
            ).reset_index()
            profiles["fg_def"] = rolling_def_fg
            logger.info("[tendency] FG DEF profiles: %d teams", len(rolling_def_fg))
    except Exception as exc:
        logger.warning("[tendency] FG features failed: %s", exc)

    return profiles


def get_matchup_features(home_team: str, away_team: str,
                          profiles: dict = None) -> dict:
    """Get pre-game tendency features for a specific matchup.

    Returns a dict of features that feed the prop model for every
    player on both teams. The QB mobility volume_multiplier applies
    TEAM-WIDE, not just to the QB.
    """
    if profiles is None:
        profiles = build_team_tendency_profiles()

    features = {"home_team": home_team, "away_team": away_team}

    # Defensive tendency of opponent (what does facing this D look like?)
    for team_label, team, opp in [("home", home_team, away_team),
                                    ("away", away_team, home_team)]:
        # RB box: opponent's defensive box tendencies
        if "rb_box" in profiles:
            rb = profiles["rb_box"]
            opp_box = rb[rb["defteam"] == opp]
            if not opp_box.empty:
                row = opp_box.iloc[0]
                features[f"{team_label}_opp_avg_box"] = row.get("avg_box_count")
                features[f"{team_label}_opp_light_box_rate"] = row.get("light_box_rate")
                features[f"{team_label}_opp_heavy_box_rate"] = row.get("heavy_box_rate")
                features[f"{team_label}_opp_rush_ypa_allowed"] = row.get("rush_ypa_allowed")

        # Blitz: opponent's blitz profile (VARIANCE effect)
        if "blitz" in profiles:
            bl = profiles["blitz"]
            opp_blitz = bl[bl["defteam"] == opp]
            if not opp_blitz.empty:
                row = opp_blitz.iloc[0]
                features[f"{team_label}_opp_blitz_rate"] = row.get("blitz_rate")
                features[f"{team_label}_opp_sigma_multiplier"] = row.get("sigma_multiplier")
                features[f"{team_label}_opp_blitz_explosive"] = row.get("blitz_explosive")

        # MLB coverage: opponent's TE/RB receiving allowed
        if "mlb_coverage" in profiles:
            mc = profiles["mlb_coverage"]
            opp_mc = mc[mc["defteam"] == opp]
            if not opp_mc.empty:
                row = opp_mc.iloc[0]
                features[f"{team_label}_opp_short_ypt"] = row.get("short_yds_per_target")
                features[f"{team_label}_opp_short_yac"] = row.get("short_yac")

        # QB mobility: OUR team's volume multiplier (team-wide!)
        if "qb_mobility" in profiles:
            qm = profiles["qb_mobility"]
            team_mob = qm[qm["posteam"] == team]
            if not team_mob.empty:
                row = team_mob.iloc[0]
                features[f"{team_label}_scramble_rate"] = row.get("scramble_rate")
                features[f"{team_label}_volume_multiplier"] = row.get("volume_multiplier")
                features[f"{team_label}_3rd_down_rate"] = row.get("third_down_rate")
                features[f"{team_label}_avg_drive_length"] = row.get("avg_drive_length")

            # Mobility × man-rate interaction for this specific matchup
            opp_blitz = profiles.get("blitz", pd.DataFrame())
            if not opp_blitz.empty:
                opp_row = opp_blitz[opp_blitz["defteam"] == opp]
                if not opp_row.empty and not team_mob.empty:
                    mob_rate = team_mob.iloc[0].get("scramble_rate", 0)
                    features[f"{team_label}_mobility_x_opp_man"] = mob_rate

        # Pressure chain: OUR team's QB pressure profile + opponent's pressure generation
        if "pressure_qb" in profiles:
            pq = profiles["pressure_qb"]
            team_pq = pq[pq["posteam"] == team]
            if not team_pq.empty:
                row = team_pq.iloc[0]
                features[f"{team_label}_pressure_rate_allowed"] = row.get("pressure_rate")
                features[f"{team_label}_int_rate_clean"] = row.get("int_rate_clean")
                features[f"{team_label}_int_rate_pressure"] = row.get("int_rate_pressure")
                features[f"{team_label}_pressure_int_uplift"] = row.get("pressure_int_uplift")
                features[f"{team_label}_td_rate_clean"] = row.get("td_rate_clean")
                features[f"{team_label}_td_rate_pressure"] = row.get("td_rate_pressure")
        if "pressure_def" in profiles:
            pd_prof = profiles["pressure_def"]
            opp_pd = pd_prof[pd_prof["defteam"] == opp]
            if not opp_pd.empty:
                row = opp_pd.iloc[0]
                features[f"{team_label}_opp_pressure_rate_gen"] = row.get("pressure_rate_generated")
                features[f"{team_label}_opp_sack_rate_gen"] = row.get("sack_rate_generated")

        # Coverage splits: opponent's coverage mix
        if "coverage_qb" in profiles:
            cq = profiles["coverage_qb"]
            team_cq = cq[cq["posteam"] == team]
            if not team_cq.empty:
                row = team_cq.iloc[0]
                features[f"{team_label}_int_rate_man"] = row.get("int_rate_man")
                features[f"{team_label}_int_rate_zone"] = row.get("int_rate_zone")
                features[f"{team_label}_td_rate_man"] = row.get("td_rate_man")
                features[f"{team_label}_td_rate_zone"] = row.get("td_rate_zone")
        if "coverage_def" in profiles:
            cd = profiles["coverage_def"]
            opp_cd = cd[cd["defteam"] == opp]
            if not opp_cd.empty:
                row = opp_cd.iloc[0]
                features[f"{team_label}_opp_man_rate"] = row.get("man_rate")
                features[f"{team_label}_opp_zone_rate"] = row.get("zone_rate")

        # Red zone: OUR offense + opponent's RZ defense
        if "redzone_off" in profiles:
            rzo = profiles["redzone_off"]
            team_rz = rzo[rzo["posteam"] == team]
            if not team_rz.empty:
                row = team_rz.iloc[0]
                features[f"{team_label}_rz_trips"] = row.get("rz_trips")
                features[f"{team_label}_rz_td_rate"] = row.get("rz_td_rate")
        if "redzone_def" in profiles:
            rzd = profiles["redzone_def"]
            opp_rzd = rzd[rzd["defteam"] == opp]
            if not opp_rzd.empty:
                row = opp_rzd.iloc[0]
                features[f"{team_label}_opp_rz_td_rate_allowed"] = row.get("rz_td_rate_allowed")
                features[f"{team_label}_opp_rz_trips_allowed"] = row.get("rz_trips_allowed")

        # Separation
        if "separation" in profiles:
            sp = profiles["separation"]
            team_sp = sp[sp["team_abbr"] == team]
            if not team_sp.empty:
                features[f"{team_label}_avg_separation"] = team_sp.iloc[0].get("avg_separation")

        # FG profiles: OUR kicker + opponent's FG-forcing tendency
        if "fg_off" in profiles:
            fgo = profiles["fg_off"]
            team_fg = fgo[fgo["posteam"] == team]
            if not team_fg.empty:
                row = team_fg.iloc[0]
                features[f"{team_label}_fg_att_pg"] = row.get("fg_att")
                features[f"{team_label}_fg_pct"] = row.get("fg_pct")
                features[f"{team_label}_avg_fg_distance"] = row.get("avg_fg_distance")
        if "fg_def" in profiles:
            fgd = profiles["fg_def"]
            opp_fgd = fgd[fgd["defteam"] == opp]
            if not opp_fgd.empty:
                features[f"{team_label}_opp_fg_att_forced"] = opp_fgd.iloc[0].get("fg_att_forced")

    return features


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    seasons = [2024, 2025]
    if len(sys.argv) > 1:
        seasons = [int(s) for s in sys.argv[1:]]

    print(f"Building tendency profiles for seasons {seasons}...")
    profiles = build_team_tendency_profiles(seasons)

    for name, df in profiles.items():
        print(f"\n{'='*60}")
        print(f"{name.upper()} (last 8 games)")
        print(f"{'='*60}")
        print(df.to_string(index=False))

    # Example matchup
    if "qb_mobility" in profiles:
        print(f"\n{'='*60}")
        print("EXAMPLE MATCHUP: KC vs BAL")
        print(f"{'='*60}")
        feats = get_matchup_features("BAL", "KC", profiles)
        for k, v in sorted(feats.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
