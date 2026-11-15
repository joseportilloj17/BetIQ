"""
Pull nflfastR play-by-play data and compute granular per-team per-game features.

Sources: https://github.com/nflverse/nflverse-data/releases/download/pbp/
Covers 2023-2025 seasons (~150K plays).

Computes features the aggregate EPA model misses:
  - Air yards / YAC (separation + RAC proxy)
  - Scheme tendencies (shotgun%, no-huddle%, pass over expected)
  - Run game detail (inside vs outside zone)
  - Pass targeting (short/deep, slot/outside)
  - Pressure generated/allowed (sack + QB hit rate)
  - Situational (3rd down, red zone, explosive plays)
  - Turnovers (INT rate, fumble rate)
  - All above computed for BOTH offense and defense
"""

import os
import sys
import sqlite3
import urllib.request
import logging

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NFL_DB = os.path.join(DATA_DIR, "nfl.db")
PBP_CACHE = os.path.join(DATA_DIR, "pbp_cache")
SEASONS = [2023, 2024, 2025, 2026]

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"


def _download_pbp(year: int) -> pd.DataFrame:
    """Download and cache nflfastR play-by-play parquet."""
    os.makedirs(PBP_CACHE, exist_ok=True)
    path = os.path.join(PBP_CACHE, f"pbp_{year}.parquet")
    if not os.path.exists(path):
        url = PBP_URL.format(year=year)
        log.info(f"Downloading {year} PBP from nflverse...")
        urllib.request.urlretrieve(url, path)
        log.info(f"  Saved to {path}")
    else:
        log.info(f"Using cached {year} PBP")
    return pd.read_parquet(path)


def _compute_team_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team per-game advanced features from play-by-play.

    Returns DataFrame with one row per team per game.
    """
    # Filter to real plays (exclude penalties, timeouts, etc.)
    plays = df[df["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])].copy()
    # Further filter to actual pass/run
    real = plays[plays["play_type"].isin(["pass", "run"])].copy()

    if real.empty:
        return pd.DataFrame()

    # Tag offensive/defensive teams
    real["off_team"] = real["posteam"]
    real["def_team"] = real["defteam"]

    # --- OFFENSIVE features (grouped by posteam + game_id) ---
    off_groups = real.groupby(["game_id", "posteam", "season"])

    # Pass plays only
    passes = real[real["play_type"] == "pass"].copy()
    pass_groups = passes.groupby(["game_id", "posteam", "season"])

    # Run plays only
    runs = real[real["play_type"] == "run"].copy()
    run_groups = runs.groupby(["game_id", "posteam", "season"])

    # Build offensive stats
    off_stats = pd.DataFrame()
    off_stats["total_plays"] = off_groups.size()
    off_stats["pass_plays"] = pass_groups.size().reindex(off_stats.index, fill_value=0)
    off_stats["run_plays"] = run_groups.size().reindex(off_stats.index, fill_value=0)

    # Scheme tendencies
    off_stats["shotgun_rate"] = off_groups["shotgun"].mean()
    off_stats["no_huddle_rate"] = off_groups["no_huddle"].mean()
    # Pass rate over expected (how much team deviates from expected play calls)
    if "xpass" in real.columns:
        off_stats["pass_oe"] = off_groups.apply(
            lambda g: (g["play_type"] == "pass").mean() - g["xpass"].mean()
            if g["xpass"].notna().any() else np.nan
        )
    else:
        off_stats["pass_oe"] = np.nan

    # Air yards / YAC (pass plays only)
    off_stats["avg_air_yards"] = pass_groups["air_yards"].mean()
    off_stats["avg_yac"] = pass_groups["yards_after_catch"].mean()
    # YAC over expected (true separation + RAC ability, not just scheme)
    if "xyac_mean_yardage" in passes.columns:
        def _yac_oe(g):
            mask = g["yards_after_catch"].notna() & g["xyac_mean_yardage"].notna()
            if mask.sum() < 3:
                return np.nan
            return (g.loc[mask, "yards_after_catch"] - g.loc[mask, "xyac_mean_yardage"]).mean()
        off_stats["yac_oe"] = pass_groups.apply(_yac_oe)
    else:
        off_stats["yac_oe"] = np.nan

    # Pass depth / location splits
    if "pass_length" in passes.columns:
        off_stats["deep_pass_rate"] = pass_groups.apply(
            lambda g: (g["pass_length"] == "deep").mean() if g["pass_length"].notna().any() else np.nan
        )
    else:
        off_stats["deep_pass_rate"] = np.nan

    if "pass_location" in passes.columns:
        off_stats["middle_pass_rate"] = pass_groups.apply(
            lambda g: (g["pass_location"] == "middle").mean() if g["pass_location"].notna().any() else np.nan
        )
    else:
        off_stats["middle_pass_rate"] = np.nan

    # Run game detail
    if "run_gap" in runs.columns and "run_location" in runs.columns:
        off_stats["run_outside_rate"] = run_groups.apply(
            lambda g: ((g["run_gap"] == "end") | (g["run_location"].isin(["left", "right"]) & (g["run_gap"] == "end"))).mean()
            if g["run_gap"].notna().any() else np.nan
        )
    else:
        off_stats["run_outside_rate"] = np.nan

    # Pressure allowed (sacks + QB hits per dropback)
    dropbacks = passes[passes["qb_dropback"] == 1] if "qb_dropback" in passes.columns else passes
    db_groups = dropbacks.groupby(["game_id", "posteam", "season"])
    off_stats["sack_rate_allowed"] = db_groups["sack"].mean().reindex(off_stats.index)
    off_stats["qb_hit_rate_allowed"] = db_groups["qb_hit"].mean().reindex(off_stats.index)
    off_stats["pressure_rate_allowed"] = off_stats["sack_rate_allowed"].fillna(0) + off_stats["qb_hit_rate_allowed"].fillna(0)

    # Situational
    third = real[real["down"] == 3]
    third_groups = third.groupby(["game_id", "posteam", "season"])
    off_stats["third_down_conv_rate"] = third_groups["first_down"].mean().reindex(off_stats.index)

    rz = real[real["yardline_100"] <= 20]
    rz_groups = rz.groupby(["game_id", "posteam", "season"])
    off_stats["redzone_td_rate"] = rz_groups["touchdown"].mean().reindex(off_stats.index)

    # Explosive plays (20+ yards)
    off_stats["explosive_rate"] = off_groups.apply(
        lambda g: (g["yards_gained"] >= 20).mean() if g["yards_gained"].notna().any() else np.nan
    )

    # Turnovers
    off_stats["int_rate"] = pass_groups["interception"].mean().reindex(off_stats.index)
    off_stats["fumble_rate"] = off_groups["fumble_lost"].mean()
    off_stats["turnover_rate"] = off_stats["int_rate"].fillna(0) + off_stats["fumble_rate"].fillna(0)

    # QB accuracy (CPOE — completion probability over expected)
    if "cpoe" in passes.columns:
        off_stats["cpoe_avg"] = pass_groups["cpoe"].mean().reindex(off_stats.index)
    else:
        off_stats["cpoe_avg"] = np.nan

    # Run EPA splits: inside (guard/tackle gaps) vs outside (end gaps)
    if "run_gap" in runs.columns:
        inside_runs = runs[runs["run_gap"].isin(["guard", "tackle"])]
        outside_runs = runs[runs["run_gap"] == "end"]
        inside_groups = inside_runs.groupby(["game_id", "posteam", "season"])
        outside_groups = outside_runs.groupby(["game_id", "posteam", "season"])
        off_stats["run_epa_inside"] = inside_groups["epa"].mean().reindex(off_stats.index)
        off_stats["run_epa_outside"] = outside_groups["epa"].mean().reindex(off_stats.index)
        off_stats["inside_run_rate"] = run_groups.apply(
            lambda g: g["run_gap"].isin(["guard", "tackle"]).mean() if g["run_gap"].notna().any() else np.nan
        )
    else:
        off_stats["run_epa_inside"] = np.nan
        off_stats["run_epa_outside"] = np.nan
        off_stats["inside_run_rate"] = np.nan

    # Pass depth EPA splits
    if "pass_length" in passes.columns:
        deep_passes = passes[passes["pass_length"] == "deep"]
        short_passes = passes[passes["pass_length"] == "short"]
        deep_groups = deep_passes.groupby(["game_id", "posteam", "season"])
        short_groups = short_passes.groupby(["game_id", "posteam", "season"])
        off_stats["deep_pass_epa"] = deep_groups["epa"].mean().reindex(off_stats.index)
        off_stats["short_pass_epa"] = short_groups["epa"].mean().reindex(off_stats.index)
        if "complete_pass" in passes.columns:
            off_stats["deep_completion_rate"] = deep_groups["complete_pass"].mean().reindex(off_stats.index)
    else:
        off_stats["deep_pass_epa"] = np.nan
        off_stats["short_pass_epa"] = np.nan
        off_stats["deep_completion_rate"] = np.nan

    # Pass location EPA splits (left/middle/right — proxy for outside vs slot)
    if "pass_location" in passes.columns:
        outside_passes = passes[passes["pass_location"].isin(["left", "right"])]
        middle_passes = passes[passes["pass_location"] == "middle"]
        out_pass_groups = outside_passes.groupby(["game_id", "posteam", "season"])
        mid_pass_groups = middle_passes.groupby(["game_id", "posteam", "season"])
        off_stats["outside_pass_epa"] = out_pass_groups["epa"].mean().reindex(off_stats.index)
        off_stats["middle_pass_epa"] = mid_pass_groups["epa"].mean().reindex(off_stats.index)
    else:
        off_stats["outside_pass_epa"] = np.nan
        off_stats["middle_pass_epa"] = np.nan

    # --- DEFENSIVE features (grouped by defteam + game_id) ---
    def_groups = real.groupby(["game_id", "defteam", "season"])
    def_pass_groups = passes.groupby(["game_id", "defteam", "season"])

    def_stats = pd.DataFrame()
    def_stats["def_total_plays"] = def_groups.size()

    # Pressure generated
    def_db = dropbacks.groupby(["game_id", "defteam", "season"]) if not dropbacks.empty else None
    if def_db is not None:
        def_stats["sack_rate_generated"] = def_db["sack"].mean().reindex(def_stats.index)
        def_stats["qb_hit_rate_generated"] = def_db["qb_hit"].mean().reindex(def_stats.index)
        def_stats["pressure_rate_generated"] = def_stats["sack_rate_generated"].fillna(0) + def_stats["qb_hit_rate_generated"].fillna(0)
    else:
        def_stats["sack_rate_generated"] = np.nan
        def_stats["qb_hit_rate_generated"] = np.nan
        def_stats["pressure_rate_generated"] = np.nan

    # Yards after catch allowed (defense separation proxy)
    def_stats["yac_allowed"] = def_pass_groups["yards_after_catch"].mean().reindex(def_stats.index)

    # Explosive plays allowed
    def_stats["explosive_rate_allowed"] = def_groups.apply(
        lambda g: (g["yards_gained"] >= 20).mean() if g["yards_gained"].notna().any() else np.nan
    )

    # Third down defense
    def_third = third.groupby(["game_id", "defteam", "season"])
    def_stats["third_down_stop_rate"] = def_third.apply(
        lambda g: 1.0 - g["first_down"].mean() if g["first_down"].notna().any() else np.nan
    )

    # Red zone defense
    def_rz = rz.groupby(["game_id", "defteam", "season"])
    def_stats["redzone_td_rate_allowed"] = def_rz.apply(
        lambda g: g["touchdown"].mean() if g["touchdown"].notna().any() else np.nan
    )

    # Turnovers forced
    def_stats["int_forced_rate"] = def_pass_groups["interception"].mean().reindex(def_stats.index)
    def_stats["fumble_forced_rate"] = def_groups["fumble_lost"].mean()
    def_stats["turnover_forced_rate"] = def_stats["int_forced_rate"].fillna(0) + def_stats["fumble_forced_rate"].fillna(0)

    # --- Defensive pass splits by depth (secondary quality proxy) ---
    if "pass_length" in passes.columns:
        def_deep = passes[passes["pass_length"] == "deep"].groupby(["game_id", "defteam", "season"])
        def_short = passes[passes["pass_length"] == "short"].groupby(["game_id", "defteam", "season"])
        def_stats["deep_pass_rate_faced"] = def_pass_groups.apply(
            lambda g: (g["pass_length"] == "deep").mean() if g["pass_length"].notna().any() else np.nan
        )
        if "complete_pass" in passes.columns:
            def_stats["deep_completion_allowed"] = def_deep["complete_pass"].mean().reindex(def_stats.index)
            def_stats["short_completion_allowed"] = def_short["complete_pass"].mean().reindex(def_stats.index)
            def_stats["completion_pct_allowed"] = def_pass_groups["complete_pass"].mean().reindex(def_stats.index)
        def_stats["deep_pass_epa_allowed"] = def_deep["epa"].mean().reindex(def_stats.index)
        def_stats["short_pass_epa_allowed"] = def_short["epa"].mean().reindex(def_stats.index)
        def_stats["avg_air_yards_allowed"] = def_pass_groups["air_yards"].mean().reindex(def_stats.index)
    else:
        def_stats["deep_pass_rate_faced"] = np.nan
        def_stats["deep_completion_allowed"] = np.nan
        def_stats["short_completion_allowed"] = np.nan
        def_stats["completion_pct_allowed"] = np.nan
        def_stats["deep_pass_epa_allowed"] = np.nan
        def_stats["short_pass_epa_allowed"] = np.nan
        def_stats["avg_air_yards_allowed"] = np.nan

    # --- Defensive pass splits by location (CB vs LB coverage proxy) ---
    if "pass_location" in passes.columns:
        def_outside = passes[passes["pass_location"].isin(["left", "right"])].groupby(["game_id", "defteam", "season"])
        def_middle = passes[passes["pass_location"] == "middle"].groupby(["game_id", "defteam", "season"])
        def_stats["outside_pass_epa_allowed"] = def_outside["epa"].mean().reindex(def_stats.index)
        def_stats["middle_pass_epa_allowed"] = def_middle["epa"].mean().reindex(def_stats.index)
        if "complete_pass" in passes.columns:
            def_stats["outside_completion_allowed"] = def_outside["complete_pass"].mean().reindex(def_stats.index)
            def_stats["middle_completion_allowed"] = def_middle["complete_pass"].mean().reindex(def_stats.index)
    else:
        def_stats["outside_pass_epa_allowed"] = np.nan
        def_stats["middle_pass_epa_allowed"] = np.nan
        def_stats["outside_completion_allowed"] = np.nan
        def_stats["middle_completion_allowed"] = np.nan

    # --- Run defense by gap (OL vs DL matchup data) ---
    def_runs = runs.groupby(["game_id", "defteam", "season"])
    if "run_gap" in runs.columns:
        def_inside_runs = runs[runs["run_gap"].isin(["guard", "tackle"])].groupby(["game_id", "defteam", "season"])
        def_outside_runs = runs[runs["run_gap"] == "end"].groupby(["game_id", "defteam", "season"])
        def_stats["run_epa_inside_allowed"] = def_inside_runs["epa"].mean().reindex(def_stats.index)
        def_stats["run_epa_outside_allowed"] = def_outside_runs["epa"].mean().reindex(def_stats.index)
        def_stats["run_success_inside_allowed"] = def_inside_runs.apply(
            lambda g: (g["epa"] > 0).mean() if len(g) > 0 else np.nan
        ).reindex(def_stats.index)
        def_stats["run_success_outside_allowed"] = def_outside_runs.apply(
            lambda g: (g["epa"] > 0).mean() if len(g) > 0 else np.nan
        ).reindex(def_stats.index)
    else:
        def_stats["run_epa_inside_allowed"] = np.nan
        def_stats["run_epa_outside_allowed"] = np.nan
        def_stats["run_success_inside_allowed"] = np.nan
        def_stats["run_success_outside_allowed"] = np.nan

    # Overall run defense EPA
    def_stats["run_epa_allowed"] = def_runs["epa"].mean().reindex(def_stats.index)

    # --- Tackle for loss rate (DL dominance proxy) ---
    if "tackled_for_loss" in real.columns:
        def_stats["tfl_rate"] = def_groups["tackled_for_loss"].mean()
    else:
        def_stats["tfl_rate"] = np.nan

    # --- SPECIAL TEAMS features (per team per game) ---
    # Punt quality
    punt_plays = df[df["play_type"] == "punt"].copy()
    if not punt_plays.empty and "punt_attempt" in punt_plays.columns:
        # Punting team stats (posteam punts)
        punt_team = punt_plays.groupby(["game_id", "posteam", "season"])
        punt_by_team = pd.DataFrame()
        punt_by_team["punt_count"] = punt_team.size()
        if "punt_inside_twenty" in punt_plays.columns:
            punt_by_team["punt_inside_20_rate"] = punt_team["punt_inside_twenty"].mean()
        if "kick_distance" in punt_plays.columns:
            punt_by_team["avg_punt_distance"] = punt_team["kick_distance"].mean()
        punt_by_team["punt_epa"] = punt_team["epa"].mean()
        punt_by_team = punt_by_team.reset_index().rename(columns={"posteam": "team"})

        # Punt return stats (return team = defteam on punt plays)
        if "return_yards" in punt_plays.columns:
            punt_ret = punt_plays[punt_plays["return_yards"].notna()].groupby(["game_id", "defteam", "season"])
            punt_ret_stats = pd.DataFrame()
            punt_ret_stats["punt_return_avg"] = punt_ret["return_yards"].mean()
            punt_ret_stats = punt_ret_stats.reset_index().rename(columns={"defteam": "team"})
        else:
            punt_ret_stats = None
    else:
        punt_by_team = None
        punt_ret_stats = None

    # Kickoff stats
    kick_plays = df[df["play_type"] == "kickoff"].copy()
    if not kick_plays.empty:
        kick_team = kick_plays.groupby(["game_id", "posteam", "season"])
        kick_stats = pd.DataFrame()
        if "touchback" in kick_plays.columns:
            kick_stats["kickoff_touchback_rate"] = kick_team["touchback"].mean()
        kick_stats["kickoff_epa"] = kick_team["epa"].mean()
        kick_stats = kick_stats.reset_index().rename(columns={"posteam": "team"})

        # Kick return stats
        if "return_yards" in kick_plays.columns:
            kick_ret = kick_plays[kick_plays["return_yards"].notna()].groupby(["game_id", "defteam", "season"])
            kick_ret_stats = pd.DataFrame()
            kick_ret_stats["kick_return_avg"] = kick_ret["return_yards"].mean()
            kick_ret_stats = kick_ret_stats.reset_index().rename(columns={"defteam": "team"})
        else:
            kick_ret_stats = None
    else:
        kick_stats = None
        kick_ret_stats = None

    # Field goal stats
    fg_plays = df[df["play_type"] == "field_goal"].copy()
    if not fg_plays.empty and "field_goal_result" in fg_plays.columns:
        fg_team = fg_plays.groupby(["game_id", "posteam", "season"])
        fg_stats = pd.DataFrame()
        fg_stats["fg_attempts"] = fg_team.size()
        fg_stats["fg_make_rate"] = fg_team.apply(
            lambda g: (g["field_goal_result"] == "made").mean()
        )
        fg_stats["fg_epa"] = fg_team["epa"].mean()
        fg_stats = fg_stats.reset_index().rename(columns={"posteam": "team"})
    else:
        fg_stats = None

    # --- Merge offensive and defensive ---
    off_stats = off_stats.reset_index()
    off_stats.rename(columns={"posteam": "team"}, inplace=True)

    def_stats = def_stats.reset_index()
    def_stats.rename(columns={"defteam": "team"}, inplace=True)

    merged = off_stats.merge(def_stats, on=["game_id", "team", "season"], how="outer")

    # Merge special teams data
    for st_df in [punt_by_team, punt_ret_stats, kick_stats, kick_ret_stats, fg_stats]:
        if st_df is not None and not st_df.empty:
            merged = merged.merge(st_df, on=["game_id", "team", "season"], how="left")

    # Add game metadata
    game_meta = df.groupby("game_id").agg(
        game_date=("game_date", "first"),
        week=("week", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
    ).reset_index()

    merged = merged.merge(game_meta, on="game_id", how="left")
    merged["is_home"] = (merged["team"] == merged["home_team"]).astype(int)
    merged["opp_team"] = np.where(merged["is_home"] == 1, merged["away_team"], merged["home_team"])

    return merged


def _store_features(features_df: pd.DataFrame, conn: sqlite3.Connection):
    """Store computed features in nfl_advanced_stats table."""
    # Select columns to store
    store_cols = [
        "game_id", "team", "season", "week", "game_date", "is_home", "opp_team",
        # Scheme
        "shotgun_rate", "no_huddle_rate", "pass_oe",
        # Air yards / separation
        "avg_air_yards", "avg_yac", "yac_oe", "deep_pass_rate", "middle_pass_rate",
        # QB accuracy
        "cpoe_avg",
        # Run detail
        "run_outside_rate", "inside_run_rate",
        "run_epa_inside", "run_epa_outside",
        # Pass depth / location EPA
        "deep_pass_epa", "short_pass_epa", "deep_completion_rate",
        "outside_pass_epa", "middle_pass_epa",
        # Pressure
        "sack_rate_allowed", "qb_hit_rate_allowed", "pressure_rate_allowed",
        "sack_rate_generated", "qb_hit_rate_generated", "pressure_rate_generated",
        # Situational
        "third_down_conv_rate", "redzone_td_rate", "explosive_rate",
        # Defense - general
        "yac_allowed", "explosive_rate_allowed", "third_down_stop_rate",
        "redzone_td_rate_allowed",
        # Defense - pass splits (secondary quality)
        "deep_pass_rate_faced", "deep_completion_allowed", "short_completion_allowed",
        "completion_pct_allowed", "deep_pass_epa_allowed", "short_pass_epa_allowed",
        "avg_air_yards_allowed",
        # Defense - pass location (CB vs LB coverage proxy)
        "outside_pass_epa_allowed", "middle_pass_epa_allowed",
        "outside_completion_allowed", "middle_completion_allowed",
        # Defense - run by gap (OL vs DL)
        "run_epa_inside_allowed", "run_epa_outside_allowed",
        "run_success_inside_allowed", "run_success_outside_allowed",
        "run_epa_allowed", "tfl_rate",
        # Turnovers
        "int_rate", "fumble_rate", "turnover_rate",
        "int_forced_rate", "fumble_forced_rate", "turnover_forced_rate",
        # Special teams
        "punt_count", "punt_inside_20_rate", "avg_punt_distance", "punt_epa",
        "punt_return_avg",
        "kickoff_touchback_rate", "kickoff_epa",
        "kick_return_avg",
        "fg_attempts", "fg_make_rate", "fg_epa",
        # Counts
        "total_plays", "pass_plays", "run_plays",
    ]

    available = [c for c in store_cols if c in features_df.columns]
    store_df = features_df[available].copy()

    # Create table
    conn.execute("DROP TABLE IF EXISTS nfl_advanced_stats")
    # Build CREATE TABLE dynamically from available columns
    col_defs = ["game_id TEXT", "team TEXT", "season INTEGER", "week INTEGER",
                "game_date TEXT", "is_home INTEGER", "opp_team TEXT"]
    int_cols = {"season", "week", "is_home", "total_plays", "pass_plays", "run_plays",
                "punt_count", "fg_attempts"}
    for c in available:
        if c in ("game_id", "team", "season", "week", "game_date", "is_home", "opp_team"):
            continue
        dtype = "INTEGER" if c in int_cols else "REAL"
        col_defs.append(f"{c} {dtype}")
    col_defs.append("PRIMARY KEY (game_id, team)")
    create_sql = "CREATE TABLE nfl_advanced_stats (\n" + ",\n".join(col_defs) + "\n)"
    conn.execute(create_sql)

    store_df.to_sql("nfl_advanced_stats", conn, if_exists="append", index=False)
    log.info(f"  Stored {len(store_df)} rows in nfl_advanced_stats")


def _compute_rolling_averages(conn: sqlite3.Connection):
    """
    Compute season-to-date rolling averages for each feature.
    Stores in nfl_advanced_rolling with one row per team per game.
    """
    df = pd.read_sql("SELECT * FROM nfl_advanced_stats ORDER BY team, season, week", conn)

    numeric_cols = [c for c in df.columns if c not in
                    ["game_id", "team", "season", "week", "game_date", "is_home", "opp_team",
                     "total_plays", "pass_plays", "run_plays"]]

    # Group by team + season, compute expanding mean (season-to-date average)
    rolling_rows = []
    for (team, season), group in df.groupby(["team", "season"]):
        group = group.sort_values("week")
        for i in range(len(group)):
            row = group.iloc[i].to_dict()
            # Use all games BEFORE this one (point-in-time, no lookahead)
            prior = group.iloc[:i]
            for col in numeric_cols:
                if col in prior.columns and len(prior) > 0:
                    row[f"s_{col}"] = prior[col].mean()
                else:
                    row[f"s_{col}"] = None
            # Also compute last-4-game averages
            recent = group.iloc[max(0, i-4):i]
            for col in numeric_cols:
                if col in recent.columns and len(recent) > 0:
                    row[f"r4_{col}"] = recent[col].mean()
                else:
                    row[f"r4_{col}"] = None
            rolling_rows.append(row)

    rolling_df = pd.DataFrame(rolling_rows)

    # Store
    conn.execute("DROP TABLE IF EXISTS nfl_advanced_rolling")
    rolling_cols = [c for c in rolling_df.columns if c.startswith("s_") or c.startswith("r4_")]
    meta_cols = ["game_id", "team", "season", "week", "game_date", "is_home", "opp_team"]
    store_cols = meta_cols + sorted(rolling_cols)
    available = [c for c in store_cols if c in rolling_df.columns]
    store = rolling_df[available]

    store.to_sql("nfl_advanced_rolling", conn, if_exists="replace", index=False)
    log.info(f"  Stored {len(store)} rows in nfl_advanced_rolling with {len(rolling_cols)} rolling features")

    return rolling_df


def main():
    log.info("=== NFL Play-by-Play Feature Pipeline ===")
    log.info(f"Seasons: {SEASONS}")
    log.info(f"Database: {NFL_DB}")

    all_features = []

    for year in SEASONS:
        log.info(f"\n--- Season {year} ---")
        try:
            pbp = _download_pbp(year)
            log.info(f"  {len(pbp)} plays loaded")

            features = _compute_team_game_features(pbp)
            log.info(f"  {len(features)} team-game rows computed")

            all_features.append(features)
        except Exception as e:
            log.error(f"  Failed for {year}: {e}")
            import traceback
            traceback.print_exc()

    if not all_features:
        log.error("No features computed!")
        return

    combined = pd.concat(all_features, ignore_index=True)
    log.info(f"\nTotal: {len(combined)} team-game rows across {len(SEASONS)} seasons")

    # Store
    conn = sqlite3.connect(NFL_DB)
    _store_features(combined, conn)

    # Compute rolling averages
    log.info("\nComputing season-to-date rolling averages...")
    _compute_rolling_averages(conn)

    # Summary stats
    log.info("\n=== Feature Summary ===")
    numeric = combined.select_dtypes(include=[np.number])
    for col in sorted(numeric.columns):
        if col in ["season", "week", "is_home", "total_plays", "pass_plays", "run_plays"]:
            continue
        vals = numeric[col].dropna()
        if len(vals) > 0:
            log.info(f"  {col:30s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                     f"min={vals.min():.4f}  max={vals.max():.4f}  pop={len(vals)/len(combined)*100:.0f}%")

    conn.close()
    log.info("\nDone!")


if __name__ == "__main__":
    main()
