#!/usr/bin/env python3
"""
totals_pit_lookahead_test.py
============================
Critical lookahead audit + recency vintage test for MLB totals style-mismatch signals.

Two signals appeared to beat the closing line:
  - ix_both_hrp (both SPs homer-prone -> OVER): +3.9pp vs breakeven
  - ix_combined_scoring (combined offensive power - SP dominance -> OVER): +3.3pp vs breakeven

BUT pitcher_stats contains FULL-SEASON cumulative stats (lookahead). This script
recomputes using strictly point-in-time data (only prior starts) and tests whether
the edge survives.

Database: ../data/historical.db
"""

import sqlite3
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "historical.db"
BE_110 = 1 / 1.9091  # 52.38% breakeven at -110

def get_conn():
    return sqlite3.connect(str(DB_PATH))


# ═══════════════════════════════════════════════════════════════════
# PART 0: SCHEMA DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def part0_schema():
    print("=" * 80)
    print("PART 0: SCHEMA DISCOVERY")
    print("=" * 80)

    conn = get_conn()
    cur = conn.cursor()

    print("\n--- pitcher_game_logs columns ---")
    cur.execute("PRAGMA table_info(pitcher_game_logs)")
    pgl_cols = cur.fetchall()
    for c in pgl_cols:
        print(f"  {c[1]:20s}  {c[2]:10s}  nullable={not c[3]}")

    print("\n--- team_game_batting columns ---")
    cur.execute("PRAGMA table_info(team_game_batting)")
    tgb_cols = cur.fetchall()
    for c in tgb_cols:
        print(f"  {c[1]:20s}  {c[2]:10s}  nullable={not c[3]}")

    print("\n--- Linking strategy ---")
    print("  pitcher_game_logs has: pitcher_name, player_id")
    print("  team_stats.stats_json has: home_probable_pitcher, away_probable_pitcher (NAMES)")
    print("  pitcher_stats has: pitcher_name, player_id (bridge table)")
    print("  -> Build name->player_id mapping from pitcher_stats for linking")

    # Build and validate the name->id map
    cur.execute("SELECT pitcher_name, player_id, season FROM pitcher_stats")
    rows = cur.fetchall()
    name_to_id = {}
    for name, pid, season in rows:
        name_to_id[name] = pid
    print(f"\n  Name->player_id map: {len(name_to_id)} unique pitcher names")

    # Check coverage: how many stats_json pitcher names can be linked?
    cur.execute("""
        SELECT stats_json FROM team_stats
        WHERE sport='MLB' AND is_home=1 AND stats_json LIKE '%probable_pitcher%'
    """)
    total, linked = 0, 0
    for (sj,) in cur.fetchall():
        d = json.loads(sj)
        for key in ["home_probable_pitcher", "away_probable_pitcher"]:
            name = d.get(key)
            if name:
                total += 1
                if name in name_to_id:
                    linked += 1
    print(f"  Probable pitcher names found: {total}")
    print(f"  Successfully linked to player_id: {linked} ({100*linked/total:.1f}%)")

    print("\n--- team_game_batting note ---")
    print("  No barrel_pct or hard_hit_pct per game (Statcast not available per-game)")
    print("  Available per-game: SLG, OPS, OBP, HR, TB, AB, etc.")
    print("  Will use season-level barrel_pct/hard_hit_pct for batting (since batting")
    print("  was never the primary signal -- SP HR/9 is the focus).")

    conn.close()
    return name_to_id


# ═══════════════════════════════════════════════════════════════════
# HELPER: Build Game Spine
# ═══════════════════════════════════════════════════════════════════

def build_game_spine():
    """Build game spine: MLB games with close_total, scores, SP names."""
    conn = get_conn()

    # betting_lines for close_total
    bl = pd.read_sql_query("""
        SELECT game_id, game_date, home_team, away_team, close_total, season
        FROM betting_lines
        WHERE sport='MLB' AND close_total IS NOT NULL
    """, conn)

    # team_stats for scores and probable pitchers
    ts = pd.read_sql_query("""
        SELECT game_id, team, is_home, score, opp_score, stats_json
        FROM team_stats
        WHERE sport='MLB'
    """, conn)

    conn.close()

    # Extract probable pitchers from stats_json
    def extract_sp(row):
        try:
            d = json.loads(row["stats_json"])
            if row["is_home"] == 1:
                return d.get("home_probable_pitcher"), d.get("away_probable_pitcher")
            else:
                return None, None
        except:
            return None, None

    ts_home = ts[ts["is_home"] == 1].copy()
    sp_info = ts_home.apply(extract_sp, axis=1, result_type="expand")
    sp_info.columns = ["home_sp", "away_sp"]
    ts_home = pd.concat([ts_home.reset_index(drop=True), sp_info.reset_index(drop=True)], axis=1)

    # Get away score from away row
    ts_away = ts[ts["is_home"] == 0][["game_id", "score"]].rename(columns={"score": "away_score"})

    # Merge
    spine = bl.merge(ts_home[["game_id", "score", "home_sp", "away_sp"]],
                     on="game_id", how="inner")
    spine = spine.rename(columns={"score": "home_score"})
    spine = spine.merge(ts_away, on="game_id", how="inner")

    spine["actual_total"] = spine["home_score"] + spine["away_score"]

    # CRITICAL: Filter out unsettled games (0-0 scores from future betting lines)
    # These are games where betting_lines exist but team_stats has no real scores yet
    unsettled = (spine["home_score"] == 0) & (spine["away_score"] == 0)
    n_unsettled = unsettled.sum()
    spine = spine[~unsettled].copy()

    spine["went_over"] = (spine["actual_total"] > spine["close_total"]).astype(int)
    # Exclude pushes
    spine = spine[spine["actual_total"] != spine["close_total"]].copy()
    spine["game_date"] = pd.to_datetime(spine["game_date"])
    spine["month"] = spine["game_date"].dt.month

    print(f"  Game spine: {len(spine)} games with totals (pushes excluded, {n_unsettled} unsettled dropped)")
    print(f"  Date range: {spine['game_date'].min().date()} to {spine['game_date'].max().date()}")
    print(f"  Overall Over WR: {spine['went_over'].mean():.4f} (BE={BE_110:.4f})")

    return spine


# ═══════════════════════════════════════════════════════════════════
# PART 1: REPLICATE FULL-SEASON (LOOKAHEAD) RESULT
# ═══════════════════════════════════════════════════════════════════

def part1_fullseason(spine, name_to_id):
    print("\n" + "=" * 80)
    print("PART 1: FULL-SEASON (LOOKAHEAD) REPLICATION")
    print("=" * 80)

    conn = get_conn()

    # Get pitcher_stats (full-season)
    ps = pd.read_sql_query("""
        SELECT pitcher_name, player_id, season, hr9, k9, bb9, fip, era, ip
        FROM pitcher_stats
    """, conn)

    # Get team_batting_stats (full-season)
    tbs = pd.read_sql_query("""
        SELECT team_name, season, barrel_pct, hard_hit_pct, slg, ops
        FROM team_batting_stats
    """, conn)

    conn.close()

    # NOTE: k9=0 and fip=99 for ALL pitchers in 2022-2025 (placeholder/missing).
    # Only hr9 and era have real data across all seasons.
    # For dominance, use ERA instead of FIP (lower ERA = more dominant).
    print("\n  DATA NOTE: k9=0 and fip=99 for all pitchers in 2022-2025 (placeholders).")
    print("  Using ERA for SP dominance instead of K/9-FIP composite.")

    # Z-score pitcher stats within season
    for col in ["hr9", "era"]:
        ps[f"z_{col}"] = ps.groupby("season")[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    ps["homer_prone"] = ps["z_hr9"]
    ps["dominance"] = -ps["z_era"]  # negative z_era: lower ERA = more dominant

    # Z-score batting stats within season
    for col in ["barrel_pct", "hard_hit_pct", "slg"]:
        tbs[f"z_{col}"] = tbs.groupby("season")[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    tbs["offensive_power"] = tbs["z_barrel_pct"] + tbs["z_hard_hit_pct"] + tbs["z_slg"]

    # Merge SP stats onto spine (use clean column selection to avoid suffix issues)
    df = spine.copy()

    # Home SP: select only needed columns before merge
    home_ps = ps[["pitcher_name", "season", "homer_prone", "dominance"]].rename(columns={
        "pitcher_name": "home_sp",
        "homer_prone": "home_sp_homer_prone",
        "dominance": "home_sp_dominance",
    })
    df = df.merge(home_ps, on=["home_sp", "season"], how="left")

    # Away SP
    away_ps = ps[["pitcher_name", "season", "homer_prone", "dominance"]].rename(columns={
        "pitcher_name": "away_sp",
        "homer_prone": "away_sp_homer_prone",
        "dominance": "away_sp_dominance",
    })
    df = df.merge(away_ps, on=["away_sp", "season"], how="left")

    # Home team batting
    home_bat = tbs[["team_name", "season", "offensive_power"]].rename(columns={
        "team_name": "home_team",
        "offensive_power": "home_offensive_power",
    })
    df = df.merge(home_bat, on=["home_team", "season"], how="left")

    # Away team batting
    away_bat = tbs[["team_name", "season", "offensive_power"]].rename(columns={
        "team_name": "away_team",
        "offensive_power": "away_offensive_power",
    })
    df = df.merge(away_bat, on=["away_team", "season"], how="left")

    # Filter to games where both SPs have stats
    has_both = df["home_sp_homer_prone"].notna() & df["away_sp_homer_prone"].notna()
    print(f"\n  Games with both SP stats: {has_both.sum()} / {len(df)}")
    df = df[has_both].copy()

    # Pool base rate for reference
    pool_over_wr = df["went_over"].mean()
    print(f"  Pool Over WR (with SP stats): {pool_over_wr:.4f}")

    # Compute interactions
    # ix_both_hrp: both SPs above-median homer_prone
    median_hp = 0  # z-scores centered at 0, so median ~ 0
    df["ix_both_hrp"] = ((df["home_sp_homer_prone"] > median_hp) &
                         (df["away_sp_homer_prone"] > median_hp)).astype(int)

    # Continuous version
    df["ix_hrp_product"] = df["home_sp_homer_prone"] * df["away_sp_homer_prone"]

    # ix_combined_scoring
    has_bat = df["home_offensive_power"].notna() & df["away_offensive_power"].notna()
    df_bat = df[has_bat].copy()
    print(f"  Games with both SP + batting stats: {len(df_bat)}")
    df_bat["ix_combined_scoring"] = (
        (df_bat["home_offensive_power"] + df_bat["away_offensive_power"]) -
        (df_bat["home_sp_dominance"] + df_bat["away_sp_dominance"])
    )

    # === Results ===
    print(f"\n  NOTE: Overall pool Over WR = {pool_over_wr:.4f}, BE at -110 = {BE_110:.4f}")
    print(f"  Edges shown vs BOTH pool base rate and -110 BE")

    print("\n--- ix_both_hrp (binary: both SPs above-median HR/9) ---")
    for val in [0, 1]:
        sub = df[df["ix_both_hrp"] == val]
        wr = sub["went_over"].mean()
        n = len(sub)
        edge_be = (wr - BE_110) * 100
        edge_pool = (wr - pool_over_wr) * 100
        print(f"  ix_both_hrp={val}: N={n:5d}  Over WR={wr:.4f}  "
              f"vs_BE={edge_be:+.2f}pp  vs_pool={edge_pool:+.2f}pp")

    print("\n--- ix_hrp_product (continuous: home_z_hr9 * away_z_hr9) ---")
    for label, lo, hi in [("Bottom Q", 0, 25), ("Q2", 25, 50), ("Q3", 50, 75), ("Top Q", 75, 100)]:
        lo_v = np.percentile(df["ix_hrp_product"].dropna(), lo)
        hi_v = np.percentile(df["ix_hrp_product"].dropna(), hi)
        if lo == 0:
            sub = df[df["ix_hrp_product"] <= hi_v]
        elif hi == 100:
            sub = df[df["ix_hrp_product"] > lo_v]
        else:
            sub = df[(df["ix_hrp_product"] > lo_v) & (df["ix_hrp_product"] <= hi_v)]
        wr = sub["went_over"].mean()
        edge_pool = (wr - pool_over_wr) * 100
        print(f"  {label:10s}: N={len(sub):5d}  Over WR={wr:.4f}  vs_pool={edge_pool:+.2f}pp")

    print("\n--- ix_combined_scoring (top Q vs bottom Q) ---")
    if len(df_bat) > 100:
        q25 = df_bat["ix_combined_scoring"].quantile(0.25)
        q75 = df_bat["ix_combined_scoring"].quantile(0.75)
        bat_pool = df_bat["went_over"].mean()
        for label, mask in [("Bottom Q", df_bat["ix_combined_scoring"] <= q25),
                            ("Top Q", df_bat["ix_combined_scoring"] > q75)]:
            sub = df_bat[mask]
            wr = sub["went_over"].mean()
            edge_pool = (wr - bat_pool) * 100
            edge_be = (wr - BE_110) * 100
            print(f"  {label:10s}: N={len(sub):5d}  Over WR={wr:.4f}  "
                  f"vs_BE={edge_be:+.2f}pp  vs_pool={edge_pool:+.2f}pp")
    else:
        print("  Insufficient batting data for ix_combined_scoring")

    # Excess runs
    print("\n--- Excess runs vs close_total ---")
    for val in [0, 1]:
        sub = df[df["ix_both_hrp"] == val]
        excess = (sub["actual_total"] - sub["close_total"]).mean()
        print(f"  ix_both_hrp={val}: excess_runs={excess:+.3f}")

    return df


# ═══════════════════════════════════════════════════════════════════
# PART 2: POINT-IN-TIME RECOMPUTATION
# ═══════════════════════════════════════════════════════════════════

def build_pit_pitcher_stats():
    """Build point-in-time pitcher stats from pitcher_game_logs."""
    conn = get_conn()
    pgl = pd.read_sql_query("""
        SELECT pitcher_name, player_id, season, game_date, ip, er, hits,
               walks, strikeouts, home_runs
        FROM pitcher_game_logs
        ORDER BY player_id, game_date
    """, conn)
    conn.close()

    pgl["game_date"] = pd.to_datetime(pgl["game_date"])

    # Sort and compute cumulative prior stats (shift(1) pattern)
    pgl = pgl.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Group by player_id and compute cumulative sums SHIFTED by 1
    for col in ["ip", "home_runs", "strikeouts", "walks", "er", "hits"]:
        pgl[f"cum_{col}"] = pgl.groupby("player_id")[col].cumsum().shift(1)
        # But shift(1) shifts across the entire df; we need per-player shift
        # Redo properly:
        pgl[f"cum_{col}"] = pgl.groupby("player_id")[col].transform(
            lambda x: x.cumsum().shift(1)
        )

    # Count prior starts
    pgl["n_prior_starts"] = pgl.groupby("player_id").cumcount()  # 0 for first start

    # Filter: need at least 3 prior starts
    pgl_valid = pgl[pgl["n_prior_starts"] >= 3].copy()

    # Compute rate stats
    pgl_valid["pit_hr9"] = (pgl_valid["cum_home_runs"] / pgl_valid["cum_ip"]) * 9
    pgl_valid["pit_k9"] = (pgl_valid["cum_strikeouts"] / pgl_valid["cum_ip"]) * 9
    pgl_valid["pit_bb9"] = (pgl_valid["cum_walks"] / pgl_valid["cum_ip"]) * 9
    pgl_valid["pit_fip"] = (
        (13 * pgl_valid["cum_home_runs"] + 3 * pgl_valid["cum_walks"] -
         2 * pgl_valid["cum_strikeouts"]) / pgl_valid["cum_ip"] + 3.2
    )
    pgl_valid["pit_era"] = (pgl_valid["cum_er"] / pgl_valid["cum_ip"]) * 9

    # Handle inf/nan from zero IP
    for col in ["pit_hr9", "pit_k9", "pit_bb9", "pit_fip", "pit_era"]:
        pgl_valid[col] = pgl_valid[col].replace([np.inf, -np.inf], np.nan)

    print(f"  PIT pitcher stats: {len(pgl_valid)} start-rows with >=3 prior starts")
    print(f"  (from {len(pgl)} total start-rows, {pgl['player_id'].nunique()} pitchers)")

    return pgl_valid


def part2_pit(spine, name_to_id):
    print("\n" + "=" * 80)
    print("PART 2: POINT-IN-TIME RECOMPUTATION (THE DECISIVE TEST)")
    print("=" * 80)

    pit = build_pit_pitcher_stats()

    # Build name->player_id for linking
    # spine has home_sp / away_sp (names), we need player_id to join to PIT stats
    df = spine.copy()

    # Map SP names to player_ids
    df["home_sp_pid"] = df["home_sp"].map(name_to_id)
    df["away_sp_pid"] = df["away_sp"].map(name_to_id)

    # Prepare PIT stats for joining (we need game_date + player_id)
    pit_join = pit[["player_id", "game_date", "pit_hr9", "pit_k9", "pit_bb9",
                    "pit_fip", "pit_era", "n_prior_starts"]].copy()

    # Join home SP PIT stats
    home_pit = pit_join.rename(columns={
        c: f"home_sp_{c}" if c not in ["player_id", "game_date"] else c
        for c in pit_join.columns
    })
    df = df.merge(home_pit, left_on=["home_sp_pid", "game_date"],
                  right_on=["player_id", "game_date"], how="left",
                  suffixes=("", "_home_pit"))
    df = df.drop(columns=["player_id"], errors="ignore")

    # Join away SP PIT stats
    away_pit = pit_join.rename(columns={
        c: f"away_sp_{c}" if c not in ["player_id", "game_date"] else c
        for c in pit_join.columns
    })
    df = df.merge(away_pit, left_on=["away_sp_pid", "game_date"],
                  right_on=["player_id", "game_date"], how="left",
                  suffixes=("", "_away_pit"))
    df = df.drop(columns=["player_id"], errors="ignore")

    # Filter to games with both SPs having PIT stats
    has_both = df["home_sp_pit_hr9"].notna() & df["away_sp_pit_hr9"].notna()
    print(f"\n  Games with both SP PIT stats: {has_both.sum()} / {len(df)}")
    df_pit = df[has_both].copy()

    if len(df_pit) < 100:
        print("  INSUFFICIENT DATA for PIT analysis!")
        return df_pit

    # Z-score PIT stats: use season-wide distribution approach
    # For each game, z-score the pitcher's cumulative stat against the
    # distribution of ALL pitchers' current cumulative stats in that season
    # Approximation: z-score within season
    for prefix in ["home_sp", "away_sp"]:
        for stat in ["pit_hr9", "pit_k9", "pit_bb9", "pit_fip", "pit_era"]:
            col = f"{prefix}_{stat}"
            df_pit[f"{col}_z"] = df_pit.groupby("season")[col].transform(
                lambda x: (x - x.mean()) / x.std()
            )

    # Compute PIT archetypes
    df_pit["home_sp_homer_prone_pit"] = df_pit["home_sp_pit_hr9_z"]
    df_pit["away_sp_homer_prone_pit"] = df_pit["away_sp_pit_hr9_z"]
    # Use -z(ERA) for dominance to match Part 1 (lower ERA = more dominant)
    # PIT ERA computed from actual game logs (real data, not placeholder)
    df_pit["home_sp_dominance_pit"] = -df_pit["home_sp_pit_era_z"]
    df_pit["away_sp_dominance_pit"] = -df_pit["away_sp_pit_era_z"]

    # ix_both_hrp PIT: both SPs above-median homer_prone
    median_hp = 0  # z-scores centered at 0
    df_pit["ix_both_hrp_pit"] = (
        (df_pit["home_sp_homer_prone_pit"] > median_hp) &
        (df_pit["away_sp_homer_prone_pit"] > median_hp)
    ).astype(int)

    # ix_combined_scoring PIT (use season batting since no per-game Statcast)
    conn = get_conn()
    tbs = pd.read_sql_query("""
        SELECT team_name, season, barrel_pct, hard_hit_pct, slg
        FROM team_batting_stats
    """, conn)
    conn.close()

    for col in ["barrel_pct", "hard_hit_pct", "slg"]:
        tbs[f"z_{col}"] = tbs.groupby("season")[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    tbs["offensive_power"] = tbs["z_barrel_pct"] + tbs["z_hard_hit_pct"] + tbs["z_slg"]

    # Merge batting onto PIT df
    home_bat = tbs[["team_name", "season", "offensive_power"]].rename(
        columns={"team_name": "ht", "offensive_power": "home_off_power"})
    df_pit = df_pit.merge(home_bat, left_on=["home_team", "season"],
                          right_on=["ht", "season"], how="left")
    df_pit.drop(columns=["ht"], errors="ignore", inplace=True)

    away_bat = tbs[["team_name", "season", "offensive_power"]].rename(
        columns={"team_name": "at", "offensive_power": "away_off_power"})
    df_pit = df_pit.merge(away_bat, left_on=["away_team", "season"],
                          right_on=["at", "season"], how="left")
    df_pit.drop(columns=["at"], errors="ignore", inplace=True)

    has_bat = df_pit["home_off_power"].notna() & df_pit["away_off_power"].notna()
    df_bat = df_pit[has_bat].copy()
    df_bat["ix_combined_scoring_pit"] = (
        (df_bat["home_off_power"] + df_bat["away_off_power"]) -
        (df_bat["home_sp_dominance_pit"] + df_bat["away_sp_dominance_pit"])
    )

    # === Results ===
    pool_pit = df_pit["went_over"].mean()
    print(f"\n  PIT pool Over WR: {pool_pit:.4f}")

    print("\n--- ix_both_hrp PIT (both SPs above-median PIT HR/9) ---")
    for val in [0, 1]:
        sub = df_pit[df_pit["ix_both_hrp_pit"] == val]
        wr = sub["went_over"].mean()
        n = len(sub)
        edge_be = (wr - BE_110) * 100
        edge_pool = (wr - pool_pit) * 100
        print(f"  ix_both_hrp_pit={val}: N={n:5d}  Over WR={wr:.4f}  "
              f"vs_BE={edge_be:+.2f}pp  vs_pool={edge_pool:+.2f}pp")

    # Continuous version
    df_pit["ix_hrp_product_pit"] = (
        df_pit["home_sp_homer_prone_pit"] * df_pit["away_sp_homer_prone_pit"]
    )
    print("\n--- ix_hrp_product PIT (continuous) ---")
    for label, lo, hi in [("Bottom Q", 0, 25), ("Q2", 25, 50), ("Q3", 50, 75), ("Top Q", 75, 100)]:
        lo_v = np.percentile(df_pit["ix_hrp_product_pit"].dropna(), lo)
        hi_v = np.percentile(df_pit["ix_hrp_product_pit"].dropna(), hi)
        if lo == 0:
            sub = df_pit[df_pit["ix_hrp_product_pit"] <= hi_v]
        elif hi == 100:
            sub = df_pit[df_pit["ix_hrp_product_pit"] > lo_v]
        else:
            sub = df_pit[(df_pit["ix_hrp_product_pit"] > lo_v) &
                         (df_pit["ix_hrp_product_pit"] <= hi_v)]
        wr = sub["went_over"].mean()
        edge_pool = (wr - pool_pit) * 100
        print(f"  {label:10s}: N={len(sub):5d}  Over WR={wr:.4f}  vs_pool={edge_pool:+.2f}pp")

    print("\n--- ix_combined_scoring PIT (top Q vs bottom Q) ---")
    if len(df_bat) > 100:
        q25 = df_bat["ix_combined_scoring_pit"].quantile(0.25)
        q75 = df_bat["ix_combined_scoring_pit"].quantile(0.75)
        bat_pool = df_bat["went_over"].mean()
        for label, mask in [("Bottom Q", df_bat["ix_combined_scoring_pit"] <= q25),
                            ("Top Q", df_bat["ix_combined_scoring_pit"] > q75)]:
            sub = df_bat[mask]
            wr = sub["went_over"].mean()
            edge_be = (wr - BE_110) * 100
            edge_pool = (wr - bat_pool) * 100
            print(f"  {label:10s}: N={len(sub):5d}  Over WR={wr:.4f}  "
                  f"vs_BE={edge_be:+.2f}pp  vs_pool={edge_pool:+.2f}pp")
    else:
        print("  Insufficient data")

    # Excess runs
    print("\n--- Excess runs vs close_total (PIT) ---")
    for val in [0, 1]:
        sub = df_pit[df_pit["ix_both_hrp_pit"] == val]
        excess = (sub["actual_total"] - sub["close_total"]).mean()
        print(f"  ix_both_hrp_pit={val}: excess_runs={excess:+.3f}")

    # Split-half stability
    print("\n--- Split-half stability (PIT ix_both_hrp) ---")
    mid = df_pit["game_date"].median()
    for label, mask in [("First half", df_pit["game_date"] <= mid),
                        ("Second half", df_pit["game_date"] > mid)]:
        sub = df_pit[mask]
        for val in [0, 1]:
            s = sub[sub["ix_both_hrp_pit"] == val]
            wr = s["went_over"].mean()
            edge_pool = (wr - pool_pit) * 100
            print(f"  {label:12s} ix_both_hrp_pit={val}: N={len(s):5d}  "
                  f"Over WR={wr:.4f}  vs_pool={edge_pool:+.2f}pp")

    # Comparison summary
    print("\n" + "-" * 60)
    print("COMPARISON: Full-Season vs Point-in-Time")
    print("-" * 60)

    return df_pit


# ═══════════════════════════════════════════════════════════════════
# PART 3: MONTHLY FINGERPRINT TEST
# ═══════════════════════════════════════════════════════════════════

def part3_monthly(df_fs, df_pit):
    print("\n" + "=" * 80)
    print("PART 3: MONTHLY FINGERPRINT TEST")
    print("=" * 80)
    print("  Lookahead fingerprint: FLAT edge across months (April ~ September)")
    print("  Real signal: WEAKER in April (less data), STRONGER later in season")

    months = [3, 4, 5, 6, 7, 8, 9, 10]
    month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}

    # Determine which columns exist
    fs_hrp_col = "ix_both_hrp" if "ix_both_hrp" in df_fs.columns else None
    pit_hrp_col = "ix_both_hrp_pit" if "ix_both_hrp_pit" in df_pit.columns else None

    print(f"\n  {'Month':>5s}  {'--- Full-Season (Lookahead) ---':^35s}  {'--- Point-in-Time ---':^35s}")
    print(f"  {'':>5s}  {'N':>5s}  {'OverWR':>8s}  {'edge_pp':>8s}     {'N':>5s}  {'OverWR':>8s}  {'edge_pp':>8s}")
    print("  " + "-" * 75)

    for m in months:
        # Full-season
        if fs_hrp_col:
            fs_m = df_fs[(df_fs["month"] == m) & (df_fs[fs_hrp_col] == 1)]
            if len(fs_m) > 5:
                fs_wr = fs_m["went_over"].mean()
                fs_edge = (fs_wr - BE_110) * 100
                fs_str = f"{len(fs_m):5d}  {fs_wr:8.4f}  {fs_edge:+8.2f}pp"
            else:
                fs_str = f"{'N<5':>5s}  {'---':>8s}  {'---':>8s}  "
        else:
            fs_str = "N/A"

        # PIT
        if pit_hrp_col:
            pit_m = df_pit[(df_pit["month"] == m) & (df_pit[pit_hrp_col] == 1)]
            if len(pit_m) > 5:
                pit_wr = pit_m["went_over"].mean()
                pit_edge = (pit_wr - BE_110) * 100
                pit_str = f"{len(pit_m):5d}  {pit_wr:8.4f}  {pit_edge:+8.2f}pp"
            else:
                pit_str = f"{'N<5':>5s}  {'---':>8s}  {'---':>8s}  "
        else:
            pit_str = "N/A"

        mn = month_names.get(m, str(m))
        print(f"  {mn:>5s}  {fs_str}     {pit_str}")

    # Also print overall when ix_both_hrp=0 for reference
    print(f"\n  --- When ix_both_hrp=0 (reference) ---")
    print(f"  {'Month':>5s}  {'--- Full-Season ---':^25s}  {'--- Point-in-Time ---':^25s}")
    print(f"  {'':>5s}  {'N':>5s}  {'OverWR':>8s}  {'edge':>6s}  {'N':>5s}  {'OverWR':>8s}  {'edge':>6s}")
    for m in months:
        fs_m0 = df_fs[(df_fs["month"] == m) & (df_fs.get(fs_hrp_col, pd.Series(dtype=float)) == 0)] if fs_hrp_col else pd.DataFrame()
        pit_m0 = df_pit[(df_pit["month"] == m) & (df_pit.get(pit_hrp_col, pd.Series(dtype=float)) == 0)] if pit_hrp_col else pd.DataFrame()

        mn = month_names.get(m, str(m))
        parts = [f"  {mn:>5s}"]
        for sub in [fs_m0, pit_m0]:
            if len(sub) > 5:
                wr = sub["went_over"].mean()
                edge = (wr - BE_110) * 100
                parts.append(f"  {len(sub):5d}  {wr:8.4f}  {edge:+.1f}pp")
            else:
                parts.append(f"  {'N<5':>5s}  {'---':>8s}  {'---':>6s}")
        print("".join(parts))


# ═══════════════════════════════════════════════════════════════════
# PART 4: STAT VINTAGE COMPARISON
# ═══════════════════════════════════════════════════════════════════

def part4_vintage(spine, name_to_id):
    print("\n" + "=" * 80)
    print("PART 4: STAT VINTAGE COMPARISON")
    print("=" * 80)

    conn = get_conn()
    pgl = pd.read_sql_query("""
        SELECT pitcher_name, player_id, season, game_date, ip, home_runs,
               strikeouts, walks, er, hits
        FROM pitcher_game_logs
        ORDER BY player_id, game_date
    """, conn)
    conn.close()

    pgl["game_date"] = pd.to_datetime(pgl["game_date"])
    pgl = pgl.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # For each start, compute multiple HR/9 vintages (all point-in-time, prior starts only)
    results = []

    for pid, grp in pgl.groupby("player_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        n = len(grp)

        cum_ip = grp["ip"].cumsum().shift(1)
        cum_hr = grp["home_runs"].cumsum().shift(1)
        cum_k = grp["strikeouts"].cumsum().shift(1)
        cum_bb = grp["walks"].cumsum().shift(1)
        n_prior = np.arange(n)  # 0 for first start

        for i in range(n):
            if n_prior[i] < 3:
                continue
            if cum_ip.iloc[i] == 0 or pd.isna(cum_ip.iloc[i]):
                continue

            row = {
                "player_id": pid,
                "pitcher_name": grp.iloc[i]["pitcher_name"],
                "game_date": grp.iloc[i]["game_date"],
                "season": grp.iloc[i]["season"],
                "n_prior": n_prior[i],
            }

            # 1. Full cumulative (all prior starts, any season)
            row["hr9_full_cum"] = (cum_hr.iloc[i] / cum_ip.iloc[i]) * 9

            # 2. This-season-only
            season_mask = grp["season"].iloc[:i] == grp.iloc[i]["season"]
            s_ip = grp["ip"].iloc[:i][season_mask].sum()
            s_hr = grp["home_runs"].iloc[:i][season_mask].sum()
            if s_ip >= 9:  # at least 1 full game equivalent
                row["hr9_season_only"] = (s_hr / s_ip) * 9
            else:
                row["hr9_season_only"] = np.nan

            # 3. Rolling-10 (last 10 starts before this one)
            recent10 = grp.iloc[max(0, i-10):i]
            r10_ip = recent10["ip"].sum()
            r10_hr = recent10["home_runs"].sum()
            if r10_ip > 0 and len(recent10) >= 3:
                row["hr9_roll10"] = (r10_hr / r10_ip) * 9
            else:
                row["hr9_roll10"] = np.nan

            # 4. Rolling-5 (last 5 starts)
            recent5 = grp.iloc[max(0, i-5):i]
            r5_ip = recent5["ip"].sum()
            r5_hr = recent5["home_runs"].sum()
            if r5_ip > 0 and len(recent5) >= 3:
                row["hr9_roll5"] = (r5_hr / r5_ip) * 9
            else:
                row["hr9_roll5"] = np.nan

            # 5. Shrinkage blend (recent 5 weighted, regressed to season mean)
            if not np.isnan(row.get("hr9_season_only", np.nan)) and r5_ip > 0 and len(recent5) >= 3:
                n_recent = len(recent5)
                recent_hr9 = (r5_hr / r5_ip) * 9
                prior_weight = 10
                row["hr9_shrinkage"] = (
                    (n_recent * recent_hr9 + prior_weight * row["hr9_season_only"]) /
                    (n_recent + prior_weight)
                )
            else:
                row["hr9_shrinkage"] = np.nan

            # Also save PIT stats for dominance
            row["pit_k9"] = (cum_k.iloc[i] / cum_ip.iloc[i]) * 9
            row["pit_fip"] = (13 * cum_hr.iloc[i] + 3 * cum_bb.iloc[i] -
                              2 * cum_k.iloc[i]) / cum_ip.iloc[i] + 3.2

            results.append(row)

    pit_all = pd.DataFrame(results)
    print(f"\n  Built {len(pit_all)} start-rows with vintage stats")

    # Map onto game spine
    df = spine.copy()
    df["home_sp_pid"] = df["home_sp"].map(name_to_id)
    df["away_sp_pid"] = df["away_sp"].map(name_to_id)

    vintages = ["hr9_full_cum", "hr9_season_only", "hr9_roll10", "hr9_roll5", "hr9_shrinkage"]
    vintage_labels = ["Full cumulative", "Season-only", "Rolling-10", "Rolling-5", "Shrinkage"]

    print(f"\n  {'Vintage':20s}  {'N':>5s}  {'OverWR_1':>8s}  {'edge_pp':>8s}  {'OverWR_0':>8s}  "
          f"{'edge0_pp':>8s}  {'exRuns_1':>8s}  {'exRuns_0':>8s}  {'SH1_edge':>8s}  {'SH2_edge':>8s}")
    print("  " + "-" * 110)

    for vintage, label in zip(vintages, vintage_labels):
        # Join this vintage's HR/9 onto the game spine
        cols_needed = ["player_id", "game_date", vintage]
        pit_v = pit_all[cols_needed].dropna(subset=[vintage])

        # Home SP
        home_v = pit_v.rename(columns={vintage: f"home_{vintage}"})
        dfv = df.merge(home_v, left_on=["home_sp_pid", "game_date"],
                       right_on=["player_id", "game_date"], how="left")
        dfv.drop(columns=["player_id"], errors="ignore", inplace=True)

        # Away SP
        away_v = pit_v.rename(columns={vintage: f"away_{vintage}"})
        dfv = dfv.merge(away_v, left_on=["away_sp_pid", "game_date"],
                        right_on=["player_id", "game_date"], how="left")
        dfv.drop(columns=["player_id"], errors="ignore", inplace=True)

        # Filter to games with both SPs
        has_both = dfv[f"home_{vintage}"].notna() & dfv[f"away_{vintage}"].notna()
        dfv = dfv[has_both].copy()

        if len(dfv) < 100:
            print(f"  {label:20s}  N<100, skipping")
            continue

        # Define "homer-prone" as above-median for this vintage
        all_hr9 = pd.concat([dfv[f"home_{vintage}"], dfv[f"away_{vintage}"]])
        median_hr9 = all_hr9.median()

        dfv["ix_both_hrp_v"] = (
            (dfv[f"home_{vintage}"] > median_hr9) &
            (dfv[f"away_{vintage}"] > median_hr9)
        ).astype(int)

        # Overall results
        n1 = dfv["ix_both_hrp_v"].sum()
        n0 = len(dfv) - n1
        sub1 = dfv[dfv["ix_both_hrp_v"] == 1]
        sub0 = dfv[dfv["ix_both_hrp_v"] == 0]
        wr1 = sub1["went_over"].mean() if len(sub1) > 0 else np.nan
        wr0 = sub0["went_over"].mean() if len(sub0) > 0 else np.nan
        edge1 = (wr1 - BE_110) * 100 if not np.isnan(wr1) else np.nan
        edge0 = (wr0 - BE_110) * 100 if not np.isnan(wr0) else np.nan

        # Excess runs
        ex1 = (sub1["actual_total"] - sub1["close_total"]).mean() if len(sub1) > 0 else np.nan
        ex0 = (sub0["actual_total"] - sub0["close_total"]).mean() if len(sub0) > 0 else np.nan

        # Split-half
        mid = dfv["game_date"].median()
        sh1 = dfv[(dfv["game_date"] <= mid) & (dfv["ix_both_hrp_v"] == 1)]
        sh2 = dfv[(dfv["game_date"] > mid) & (dfv["ix_both_hrp_v"] == 1)]
        sh1_edge = (sh1["went_over"].mean() - BE_110) * 100 if len(sh1) > 20 else np.nan
        sh2_edge = (sh2["went_over"].mean() - BE_110) * 100 if len(sh2) > 20 else np.nan

        def fmt(v):
            return f"{v:+8.2f}" if not np.isnan(v) else f"{'---':>8s}"

        print(f"  {label:20s}  {n1:5d}  {wr1:8.4f}  {fmt(edge1)}  {wr0:8.4f}  "
              f"{fmt(edge0)}  {fmt(ex1)}  {fmt(ex0)}  {fmt(sh1_edge)}  {fmt(sh2_edge)}")

    return pit_all


# ═══════════════════════════════════════════════════════════════════
# PART 5: FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════

def part5_verdict(df_fs, df_pit):
    print("\n" + "=" * 80)
    print("PART 5: FINAL VERDICT")
    print("=" * 80)

    # Full-season edge
    if "ix_both_hrp" in df_fs.columns:
        fs1 = df_fs[df_fs["ix_both_hrp"] == 1]
        fs_edge = (fs1["went_over"].mean() - BE_110) * 100
        print(f"\n  1. Full-season (lookahead) edge: {fs_edge:+.2f}pp  (N={len(fs1)})")
    else:
        fs_edge = np.nan
        print("\n  1. Full-season edge: N/A")

    # PIT edge
    if "ix_both_hrp_pit" in df_pit.columns:
        pit1 = df_pit[df_pit["ix_both_hrp_pit"] == 1]
        pit_edge = (pit1["went_over"].mean() - BE_110) * 100
        n_pit = len(pit1)

        # Split-half
        mid = df_pit["game_date"].median()
        sh1 = df_pit[(df_pit["game_date"] <= mid) & (df_pit["ix_both_hrp_pit"] == 1)]
        sh2 = df_pit[(df_pit["game_date"] > mid) & (df_pit["ix_both_hrp_pit"] == 1)]
        sh1_e = (sh1["went_over"].mean() - BE_110) * 100 if len(sh1) > 20 else np.nan
        sh2_e = (sh2["went_over"].mean() - BE_110) * 100 if len(sh2) > 20 else np.nan

        survives = pit_edge > 1.5 and not np.isnan(sh1_e) and not np.isnan(sh2_e) and sh1_e > 0 and sh2_e > 0
        status = "SURVIVES" if survives else "COLLAPSES"
        print(f"  2. Point-in-time edge: {pit_edge:+.2f}pp  (N={n_pit})  --> {status}")
        if not np.isnan(sh1_e):
            print(f"     Split-half: first={sh1_e:+.2f}pp (N={len(sh1)})  "
                  f"second={sh2_e:+.2f}pp (N={len(sh2)})")
    else:
        pit_edge = np.nan
        status = "N/A"
        print("  2. Point-in-time edge: N/A")

    # Monthly fingerprint
    print(f"  3. Monthly fingerprint: check Part 3 table above")
    if "ix_both_hrp" in df_fs.columns:
        apr_fs = df_fs[(df_fs["month"] == 4) & (df_fs["ix_both_hrp"] == 1)]
        sep_fs = df_fs[(df_fs["month"] == 9) & (df_fs["ix_both_hrp"] == 1)]
        if len(apr_fs) > 5 and len(sep_fs) > 5:
            apr_e = (apr_fs["went_over"].mean() - BE_110) * 100
            sep_e = (sep_fs["went_over"].mean() - BE_110) * 100
            diff = abs(apr_e - sep_e)
            pattern = "FLAT (lookahead)" if diff < 3.0 else "GRADIENT (real signal)"
            print(f"     Full-season Apr={apr_e:+.1f}pp  Sep={sep_e:+.1f}pp  "
                  f"diff={diff:.1f}pp  --> {pattern}")

    if "ix_both_hrp_pit" in df_pit.columns:
        apr_pit = df_pit[(df_pit["month"] == 4) & (df_pit["ix_both_hrp_pit"] == 1)]
        sep_pit = df_pit[(df_pit["month"] == 9) & (df_pit["ix_both_hrp_pit"] == 1)]
        if len(apr_pit) > 5 and len(sep_pit) > 5:
            apr_e = (apr_pit["went_over"].mean() - BE_110) * 100
            sep_e = (sep_pit["went_over"].mean() - BE_110) * 100
            diff = abs(apr_e - sep_e)
            pattern = "FLAT" if diff < 3.0 else "GRADIENT"
            print(f"     PIT          Apr={apr_e:+.1f}pp  Sep={sep_e:+.1f}pp  "
                  f"diff={diff:.1f}pp  --> {pattern}")

    # Verdict
    print(f"\n  4. Best vintage: see Part 4 table above")

    print("\n" + "=" * 80)
    if status == "COLLAPSES":
        print("  VERDICT: The totals style-mismatch signals (ix_both_hrp, ix_combined_scoring)")
        print("  were LOOKAHEAD ARTIFACTS. Using only prior-start data, the edge collapses.")
        print("  These signals should NOT be deployed.")
    elif status == "SURVIVES":
        print("  VERDICT: The totals style-mismatch signal SURVIVES point-in-time recomputation.")
        print("  Check Part 4 for the best vintage and Part 3 for monthly consistency.")
        print("  Deploy only if split-half stable AND monthly gradient is present.")
    else:
        print("  VERDICT: Insufficient data to make a determination.")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("TOTALS PIT LOOKAHEAD TEST")
    print("MLB Style-Mismatch Signals: ix_both_hrp + ix_combined_scoring")
    print(f"Database: {DB_PATH}")
    print("=" * 80)

    # Part 0
    name_to_id = part0_schema()

    # Build game spine
    print("\n--- Building game spine ---")
    spine = build_game_spine()

    # Part 1: Full-season (lookahead) replication
    df_fs = part1_fullseason(spine, name_to_id)

    # Part 2: Point-in-time
    df_pit = part2_pit(spine, name_to_id)

    # Part 3: Monthly fingerprint
    part3_monthly(df_fs, df_pit)

    # Part 4: Vintage comparison
    part4_vintage(spine, name_to_id)

    # Part 5: Verdict
    part5_verdict(df_fs, df_pit)
