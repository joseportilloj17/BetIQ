"""
Pull nflverse roster data, compute player-level EPA from PBP,
detect offseason movements (trades, FA, draft), and calculate
per-team EPA adjustments for futures projections.

Sources:
  - Rosters: https://github.com/nflverse/nflverse-data/releases/download/rosters/
  - PBP: already cached in /tmp/nfl_pbp_cache/ from pull_nfl_pbp.py
  - Players master: https://github.com/nflverse/nflverse-data/releases/download/players/

Output: nfl_roster_adjustments table in nfl.db
  One row per team with EPA delta from incoming/departing players.
"""

import os
import sqlite3
import urllib.request
import logging

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NFL_DB = os.path.join(DATA_DIR, "nfl.db")
CACHE_DIR = "/tmp/nfl_roster_cache"

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
ROSTER_URL = BASE_URL + "/rosters/roster_{year}.parquet"
PBP_CACHE = "/tmp/nfl_pbp_cache"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"

# Season we're projecting FROM (last completed season)
FROM_SEASON = 2025
# Season we're projecting INTO
TO_SEASON = 2026


def _download(url: str, path: str) -> str:
    """Download file if not cached."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        log.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
        log.info(f"  Saved to {path}")
    else:
        log.info(f"Using cached {path}")
    return path


def _load_rosters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load rosters for FROM and TO seasons."""
    r_from = pd.read_parquet(
        _download(
            ROSTER_URL.format(year=FROM_SEASON),
            os.path.join(CACHE_DIR, f"roster_{FROM_SEASON}.parquet"),
        )
    )
    r_to = pd.read_parquet(
        _download(
            ROSTER_URL.format(year=TO_SEASON),
            os.path.join(CACHE_DIR, f"roster_{TO_SEASON}.parquet"),
        )
    )
    log.info(f"  {FROM_SEASON} roster: {len(r_from)} players")
    log.info(f"  {TO_SEASON} roster: {len(r_to)} players")
    return r_from, r_to


def _compute_player_epa(seasons: list[int] = None) -> pd.DataFrame:
    """
    Compute per-player EPA totals from play-by-play data.
    Covers BOTH offensive and defensive players.

    Returns DataFrame with columns:
      gsis_id, player_name, team, position_group, season,
      total_epa, epa_per_play, snap_count, games_played, side
    """
    if seasons is None:
        seasons = [FROM_SEASON]

    all_player_stats = []

    for year in seasons:
        pbp_path = os.path.join(PBP_CACHE, f"pbp_{year}.parquet")
        if not os.path.exists(pbp_path):
            pbp_path = _download(PBP_URL.format(year=year), pbp_path)

        log.info(f"Computing player EPA for {year}...")
        pbp = pd.read_parquet(pbp_path)

        # Filter to real plays
        plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

        # ===================== OFFENSIVE PLAYERS =====================

        # --- Passers ---
        passer_plays = plays[plays["passer_player_id"].notna()].copy()
        passer_epa = (
            passer_plays.groupby(["passer_player_id", "passer_player_name", "posteam"])
            .agg(
                total_epa=("epa", "sum"),
                snap_count=("epa", "count"),
                games_played=("game_id", "nunique"),
                cpoe=("cpoe", "mean"),
            )
            .reset_index()
        )
        passer_epa.rename(
            columns={
                "passer_player_id": "gsis_id",
                "passer_player_name": "player_name",
                "posteam": "team",
            },
            inplace=True,
        )
        passer_epa["position_group"] = "QB"
        passer_epa["season"] = year
        passer_epa["side"] = "OFF"

        # --- Rushers ---
        rusher_plays = plays[
            (plays["rusher_player_id"].notna()) & (plays["play_type"] == "run")
        ].copy()
        rusher_epa = (
            rusher_plays.groupby(["rusher_player_id", "rusher_player_name", "posteam"])
            .agg(
                total_epa=("epa", "sum"),
                snap_count=("epa", "count"),
                games_played=("game_id", "nunique"),
            )
            .reset_index()
        )
        rusher_epa.rename(
            columns={
                "rusher_player_id": "gsis_id",
                "rusher_player_name": "player_name",
                "posteam": "team",
            },
            inplace=True,
        )
        rusher_epa["position_group"] = "RB"
        rusher_epa["cpoe"] = None
        rusher_epa["season"] = year
        rusher_epa["side"] = "OFF"

        # --- Receivers ---
        receiver_plays = plays[plays["receiver_player_id"].notna()].copy()
        receiver_epa = (
            receiver_plays.groupby(
                ["receiver_player_id", "receiver_player_name", "posteam"]
            )
            .agg(
                total_epa=("epa", "sum"),
                snap_count=("epa", "count"),
                games_played=("game_id", "nunique"),
                avg_yac=("yards_after_catch", "mean"),
            )
            .reset_index()
        )
        receiver_epa.rename(
            columns={
                "receiver_player_id": "gsis_id",
                "receiver_player_name": "player_name",
                "posteam": "team",
            },
            inplace=True,
        )
        receiver_epa["position_group"] = "WR/TE"
        receiver_epa["cpoe"] = None
        receiver_epa["season"] = year
        receiver_epa["side"] = "OFF"

        # ===================== DEFENSIVE PLAYERS =====================
        # Defensive EPA: attribute play EPA to every defender who participated.
        #
        # TWO-TIER attribution by position:
        #   DL/LB: tackles + impact events (sacks, TFLs, QB hits, FFs).
        #     Tackles at/near the LOS are informative of individual contribution.
        #   DB: impact events ONLY (INTs, PDs, FFs — NO tackles).
        #     DB tackles happen AFTER the damage (post-completion cleanup).
        #     A safety tackling after a 30-yard catch gets negative EPA but
        #     didn't cause the completion. Tackles would penalize cleanup defenders.
        #
        # CRITICAL: A single play can credit the same player for multiple events
        # (e.g., a sack is also a TFL, a QB hit, AND a tackle). We DEDUPLICATE
        # by (play_id, player_id) so each play's EPA counts once per defender.

        # Load roster to identify DB positions — they only get playmaking events
        roster_path = os.path.join(CACHE_DIR, f"roster_{year}.parquet")
        if not os.path.exists(roster_path):
            roster_path = _download(ROSTER_URL.format(year=year), roster_path)
        roster_pos = pd.read_parquet(roster_path, columns=["gsis_id", "position"])
        db_ids = set(
            roster_pos.loc[roster_pos["position"] == "DB", "gsis_id"].dropna()
        )
        log.info(f"  {len(db_ids)} DB players identified (tackle-excluded)")

        # --- Impact events: all positions ---
        impact_cols = [
            ("sack_player_id", "sack_player_name", "defteam", "DL"),
            ("interception_player_id", "interception_player_name", "defteam", "DB"),
            ("pass_defense_1_player_id", "pass_defense_1_player_name", "defteam", "DB"),
            ("qb_hit_1_player_id", "qb_hit_1_player_name", "defteam", "DL"),
            ("tackle_for_loss_1_player_id", "tackle_for_loss_1_player_name", "defteam", "DL/LB"),
        ]
        ff_cols = ("forced_fumble_player_1_player_id", "forced_fumble_player_1_player_name",
                   "forced_fumble_player_1_team", "DEF")

        # --- Tackle events: DL/LB only (DBs excluded below) ---
        tackle_cols = [
            ("solo_tackle_1_player_id", "solo_tackle_1_player_name", "solo_tackle_1_team", "DEF"),
            ("solo_tackle_2_player_id", "solo_tackle_2_player_name", "solo_tackle_2_team", "DEF"),
            ("assist_tackle_1_player_id", "assist_tackle_1_player_name", "assist_tackle_1_team", "DEF"),
            ("assist_tackle_2_player_id", "assist_tackle_2_player_name", "assist_tackle_2_team", "DEF"),
        ]

        def_play_rows = []
        # Add impact events (all positions)
        for id_col, name_col, team_col, pos_group in impact_cols + [ff_cols]:
            if id_col not in plays.columns:
                continue
            mask = plays[id_col].notna()
            if not mask.any():
                continue
            keep_cols = ["game_id", "play_id", "epa", id_col, name_col, team_col]
            keep_cols = [c for c in keep_cols if c in plays.columns]
            subset = plays.loc[mask, keep_cols].copy()
            subset = subset.rename(columns={
                id_col: "gsis_id", name_col: "player_name", team_col: "team"
            })
            subset["position_group"] = pos_group
            def_play_rows.append(subset[["game_id", "play_id", "epa", "gsis_id", "player_name", "team", "position_group"]])

        # Add tackle events (DL/LB only — filter out DBs)
        for id_col, name_col, team_col, pos_group in tackle_cols:
            if id_col not in plays.columns:
                continue
            mask = plays[id_col].notna()
            if not mask.any():
                continue
            keep_cols = ["game_id", "play_id", "epa", id_col, name_col, team_col]
            keep_cols = [c for c in keep_cols if c in plays.columns]
            subset = plays.loc[mask, keep_cols].copy()
            subset = subset.rename(columns={
                id_col: "gsis_id", name_col: "player_name", team_col: "team"
            })
            # Exclude DB tackles — their tackles are post-damage cleanup
            subset = subset[~subset["gsis_id"].isin(db_ids)]
            subset["position_group"] = pos_group
            def_play_rows.append(subset[["game_id", "play_id", "epa", "gsis_id", "player_name", "team", "position_group"]])

        if def_play_rows:
            all_def_plays = pd.concat(def_play_rows, ignore_index=True)
            # DEDUPLICATE: same play credited to same player only counts once.
            # A sack play is also a TFL and a QB hit — without dedup, EPA is
            # counted 3x for the same play. Use play_id + gsis_id as the key.
            before = len(all_def_plays)
            all_def_plays = all_def_plays.drop_duplicates(
                subset=["play_id", "gsis_id"], keep="first"
            )
            log.info(f"  Defensive plays: {before} raw → {len(all_def_plays)} after dedup "
                     f"({before - len(all_def_plays)} duplicates removed)")

            # Aggregate per player-team (negating EPA: positive = good for defense)
            def_player_stats = (
                all_def_plays.groupby(["gsis_id", "player_name", "team"])
                .agg(
                    total_epa=("epa", lambda x: -x.sum()),
                    snap_count=("epa", "count"),
                    games_played=("game_id", "nunique"),
                    position_group=("position_group", "first"),
                )
                .reset_index()
            )
            def_player_stats["season"] = year
            def_player_stats["side"] = "DEF"
            def_player_stats["cpoe"] = None
            log.info(f"  {len(def_player_stats)} defensive players (deduplicated)")
        else:
            def_player_stats = pd.DataFrame()

        # ===================== COMBINE OFF + DEF =====================
        off_combined = pd.concat(
            [passer_epa, rusher_epa, receiver_epa], ignore_index=True
        )

        # For offensive players who appear in multiple roles, aggregate
        # But keep QB tag if they have any QB snaps (QBs rush too)
        off_player_stats = (
            off_combined.groupby(["gsis_id", "player_name", "team", "season", "side"])
            .agg(
                total_epa=("total_epa", "sum"),
                snap_count=("snap_count", "sum"),
                games_played=("games_played", "max"),
                position_group=("position_group", "first"),  # QB takes priority since it's first
            )
            .reset_index()
        )
        off_player_stats["cpoe"] = None

        # Merge offensive + defensive
        all_stats = [off_player_stats]
        if not def_player_stats.empty:
            all_stats.append(def_player_stats)
        player_stats = pd.concat(all_stats, ignore_index=True)

        player_stats["epa_per_play"] = (
            player_stats["total_epa"] / player_stats["snap_count"]
        )

        all_player_stats.append(player_stats)

    result = pd.concat(all_player_stats, ignore_index=True)
    off_count = (result["side"] == "OFF").sum()
    def_count = (result["side"] == "DEF").sum()
    log.info(
        f"  {len(result)} player-team-season rows ({off_count} OFF, {def_count} DEF), "
        f"total EPA range: [{result['total_epa'].min():.1f}, {result['total_epa'].max():.1f}]"
    )
    return result


def _detect_movements(
    roster_from: pd.DataFrame, roster_to: pd.DataFrame
) -> pd.DataFrame:
    """
    Detect player movements between seasons via roster diff.

    Returns DataFrame with:
      gsis_id, player_name, position, from_team, to_team, move_type
    """
    # Normalize columns
    r_from = roster_from[["gsis_id", "full_name", "position", "team", "status"]].copy()
    r_from = r_from.rename(columns={"team": "from_team", "full_name": "player_name"})
    r_from = r_from[r_from["gsis_id"].notna()].drop_duplicates(subset=["gsis_id"])

    r_to = roster_to[["gsis_id", "full_name", "position", "team", "status"]].copy()
    r_to = r_to.rename(
        columns={"team": "to_team", "full_name": "player_name", "status": "to_status"}
    )
    r_to = r_to[r_to["gsis_id"].notna()].drop_duplicates(subset=["gsis_id"])

    # Outer merge to find all movements
    merged = r_from.merge(r_to[["gsis_id", "to_team", "to_status"]], on="gsis_id", how="outer")

    # Fill in names/positions from whichever side has them
    if "player_name" not in merged.columns:
        merged["player_name"] = None
    name_fill = r_to.set_index("gsis_id")["player_name"]
    merged["player_name"] = merged["player_name"].fillna(merged["gsis_id"].map(name_fill))

    pos_fill = r_to.set_index("gsis_id")["position"]
    merged["position"] = merged["position"].fillna(merged["gsis_id"].map(pos_fill))

    # Classify movements
    def _classify(row):
        if pd.isna(row.get("from_team")) and pd.notna(row.get("to_team")):
            return "ROOKIE/SIGNED"  # New to league or unsigned in prior year
        elif pd.notna(row.get("from_team")) and pd.isna(row.get("to_team")):
            return "DEPARTED"  # Retired, cut, unsigned
        elif row.get("from_team") != row.get("to_team"):
            return "CHANGED_TEAM"  # Trade or FA signing
        else:
            return "RETAINED"

    merged["move_type"] = merged.apply(_classify, axis=1)

    # Filter to actual movements (not retained)
    moves = merged[merged["move_type"] != "RETAINED"].copy()

    log.info(f"\nRoster movements {FROM_SEASON} → {TO_SEASON}:")
    log.info(f"  CHANGED_TEAM: {(moves['move_type'] == 'CHANGED_TEAM').sum()}")
    log.info(f"  DEPARTED:     {(moves['move_type'] == 'DEPARTED').sum()}")
    log.info(f"  ROOKIE/SIGNED:{(moves['move_type'] == 'ROOKIE/SIGNED').sum()}")

    return moves


def _compute_team_adjustments(
    movements: pd.DataFrame,
    player_epa: pd.DataFrame,
    roster_to: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute EPA adjustment per team based on player movements.

    Uses REPLACEMENT-LEVEL baseline: only above-replacement players
    (EPA/play > 0) matter in the calculation. Bad players leaving
    doesn't help — they get replaced by equally mediocre players.

    Snap-weighted: a QB who played 700 snaps at +0.20 EPA/play matters
    more than a WR who played 60 snaps at +0.50 EPA/play.

    For each team:
      departed_value = sum of above-replacement EPA from departed players
      incoming_value = sum of above-replacement EPA from acquired players
      rookie_value   = estimated above-replacement EPA for draft picks
      net_epa_delta  = incoming_value - departed_value + rookie_value
    """
    # Get 2025 player EPA (most recent season)
    # Some players played for multiple teams — aggregate across all teams
    # Combine offensive + defensive EPA for each player into total impact
    epa_2025 = player_epa[player_epa["season"] == FROM_SEASON].copy()
    epa_agg = (
        epa_2025.groupby("gsis_id")
        .agg(
            total_epa=("total_epa", "sum"),
            epa_per_play=("epa_per_play", "mean"),
            snap_count=("snap_count", "sum"),
            games_played=("games_played", "max"),
            position_group=("position_group", "first"),
            player_name=("player_name", "first"),
            side=("side", "first"),
        )
    )
    epa_lookup = epa_agg.to_dict("index")

    # Replacement level: ~0.0 EPA (a player who neither helps nor hurts).
    # With tackle attribution for DL/LB, defensive EPA properly includes
    # negative values, so 0.0 works for both sides.
    REPLACEMENT_LEVEL = 0.0

    # --- Minimum thresholds ---
    # OFF + DEF DL/LB: snap_count is play participations (tackles + events).
    #   30 snaps filters out fringe/garbage-time players.
    # DEF DB: snap_count is impact events ONLY (INTs, PDs, FFs) — much lower
    #   volume. A starting DB might have 10-20 impact events all season.
    #   Use games_played threshold instead.
    MIN_SNAPS = 30
    MIN_GAMES_DB = 10  # DBs must have played 10+ games

    # Historical rookie EPA ABOVE REPLACEMENT by draft round
    # Round 1 picks produce ~15 EPA above replacement in year 1
    # Later rounds are at or below replacement
    ROOKIE_EPA_BY_ROUND = {
        1: 12.0,   # 1st rounders: meaningful contributor
        2: 4.0,    # 2nd rounders: slight positive
        3: 0.0,    # 3rd rounders: replacement level
        4: 0.0,    # Day 3: replacement level
        5: 0.0,
        6: 0.0,
        7: 0.0,
    }

    # QB premium: 1st-round QBs expected to start and contribute
    ROOKIE_QB_EPA = {
        1: 20.0,   # Top QB picks (expected starters)
        2: 3.0,
        3: 0.0,
    }

    # All 32 teams
    all_teams = sorted(roster_to["team"].dropna().unique())

    team_adjustments = []

    for team in all_teams:
        # --- Players who LEFT this team ---
        departed = movements[
            (movements["from_team"] == team)
            & (movements["move_type"].isin(["CHANGED_TEAM", "DEPARTED"]))
        ]

        departed_epa = 0.0
        departed_value = 0.0  # Only above-replacement portion
        departed_off_value = 0.0
        departed_def_value = 0.0
        departed_snaps = 0
        departed_details = []
        for _, row in departed.iterrows():
            pid = row["gsis_id"]
            if pid in epa_lookup:
                info = epa_lookup[pid]
                # Position-aware threshold: DBs use games, others use snaps
                is_db = info.get("position_group") == "DB"
                if is_db:
                    if info["games_played"] < MIN_GAMES_DB:
                        continue
                elif info["snap_count"] < MIN_SNAPS:
                    continue
                departed_epa += info["total_epa"]
                departed_snaps += info["snap_count"]
                # Only count above-replacement value as a loss
                above_repl = max(0, info["total_epa"] - REPLACEMENT_LEVEL)
                departed_value += above_repl
                if info.get("side") == "DEF":
                    departed_def_value += above_repl
                else:
                    departed_off_value += above_repl
                departed_details.append({
                    "player": info["player_name"],
                    "position": info["position_group"],
                    "side": info.get("side", "OFF"),
                    "epa": round(info["total_epa"], 1),
                    "value": round(above_repl, 1),
                    "snaps": info["snap_count"],
                    "destination": row.get("to_team", "OUT"),
                })

        # --- Players who JOINED this team ---
        incoming = movements[
            (movements["to_team"] == team)
            & (movements["move_type"].isin(["CHANGED_TEAM"]))
        ]

        incoming_epa = 0.0
        incoming_value = 0.0
        incoming_off_value = 0.0
        incoming_def_value = 0.0
        incoming_snaps = 0
        incoming_details = []
        for _, row in incoming.iterrows():
            pid = row["gsis_id"]
            if pid in epa_lookup:
                info = epa_lookup[pid]
                # Position-aware threshold: DBs use games, others use snaps
                is_db = info.get("position_group") == "DB"
                if is_db:
                    if info["games_played"] < MIN_GAMES_DB:
                        continue
                elif info["snap_count"] < MIN_SNAPS:
                    continue
                incoming_epa += info["total_epa"]
                incoming_snaps += info["snap_count"]
                # Only count above-replacement value as a gain
                above_repl = max(0, info["total_epa"] - REPLACEMENT_LEVEL)
                incoming_value += above_repl
                if info.get("side") == "DEF":
                    incoming_def_value += above_repl
                else:
                    incoming_off_value += above_repl
                incoming_details.append({
                    "player": info["player_name"],
                    "position": info["position_group"],
                    "side": info.get("side", "OFF"),
                    "epa": round(info["total_epa"], 1),
                    "value": round(above_repl, 1),
                    "snaps": info["snap_count"],
                    "source": row.get("from_team", "?"),
                })

        # --- Rookies (new to league, on this team's roster) ---
        rookies = movements[
            (movements["to_team"] == team)
            & (movements["move_type"] == "ROOKIE/SIGNED")
        ]

        # Cross-reference with roster_to for draft info
        rookie_epa = 0.0
        rookie_details = []
        r_to_info = roster_to.set_index("gsis_id")
        for _, row in rookies.iterrows():
            pid = row["gsis_id"]
            if pid in r_to_info.index:
                r_info = r_to_info.loc[pid]
                draft_num = r_info.get("draft_number") if hasattr(r_info, "get") else None
                if pd.notna(draft_num):
                    # Map pick number to round
                    pick = int(draft_num)
                    if pick <= 32:
                        rd = 1
                    elif pick <= 64:
                        rd = 2
                    elif pick <= 100:
                        rd = 3
                    elif pick <= 135:
                        rd = 4
                    elif pick <= 176:
                        rd = 5
                    elif pick <= 220:
                        rd = 6
                    else:
                        rd = 7

                    pos = row.get("position", "")
                    is_qb = pos == "QB"

                    if is_qb and rd <= 3:
                        est_epa = ROOKIE_QB_EPA.get(rd, 0.0)
                    else:
                        est_epa = ROOKIE_EPA_BY_ROUND.get(rd, 0.0)

                    if est_epa > 0:
                        rookie_epa += est_epa
                        rookie_details.append({
                            "player": row.get("player_name", "Unknown"),
                            "position": pos,
                            "draft_pick": pick,
                            "round": rd,
                            "est_epa": round(est_epa, 1),
                        })

        # --- Net adjustment (replacement-level based) ---
        # Lost value = above-replacement EPA that walked out the door
        # Gained value = above-replacement EPA that walked in + rookie value
        net_epa = incoming_value - departed_value + rookie_epa

        # Convert total EPA delta to EPA/play adjustment
        # Average team runs ~1050 plays/season (65 per game × 17)
        PLAYS_PER_SEASON = 1050
        epa_per_play_adj = net_epa / PLAYS_PER_SEASON

        # Offensive vs defensive net
        net_off_epa = incoming_off_value - departed_off_value + rookie_epa
        net_def_epa = incoming_def_value - departed_def_value

        team_adjustments.append({
            "team": team,
            "departed_epa": round(departed_value, 1),
            "departed_count": len(departed_details),
            "departed_snaps": departed_snaps,
            "incoming_epa": round(incoming_value, 1),
            "incoming_count": len(incoming_details),
            "incoming_snaps": incoming_snaps,
            "rookie_epa": round(rookie_epa, 1),
            "rookie_count": len(rookie_details),
            "net_epa": round(net_epa, 1),
            "net_off_epa": round(net_off_epa, 1),
            "net_def_epa": round(net_def_epa, 1),
            "epa_per_play_adj": round(epa_per_play_adj, 4),
            "departed_details": str(departed_details),
            "incoming_details": str(incoming_details),
            "rookie_details": str(rookie_details),
        })

    result = pd.DataFrame(team_adjustments)
    log.info(f"\n=== Team EPA Adjustments ===")
    for _, r in result.sort_values("net_epa").iterrows():
        direction = "▲" if r["net_epa"] > 0 else "▼" if r["net_epa"] < 0 else "─"
        log.info(
            f"  {r['team']:3s}  {direction} net={r['net_epa']:+7.1f}  "
            f"in={r['incoming_epa']:+7.1f}({r['incoming_count']})  "
            f"out={r['departed_epa']:+7.1f}({r['departed_count']})  "
            f"rk={r['rookie_epa']:+5.1f}({r['rookie_count']})  "
            f"adj={r['epa_per_play_adj']:+.4f}/play"
        )

    return result


def _store_adjustments(adj_df: pd.DataFrame, conn: sqlite3.Connection):
    """Store team roster adjustments in nfl.db."""
    conn.execute("DROP TABLE IF EXISTS nfl_roster_adjustments")
    conn.execute("""
        CREATE TABLE nfl_roster_adjustments (
            team TEXT PRIMARY KEY,
            departed_epa REAL,
            departed_count INTEGER,
            departed_snaps INTEGER,
            incoming_epa REAL,
            incoming_count INTEGER,
            incoming_snaps INTEGER,
            rookie_epa REAL,
            rookie_count INTEGER,
            net_epa REAL,
            net_off_epa REAL,
            net_def_epa REAL,
            epa_per_play_adj REAL,
            departed_details TEXT,
            incoming_details TEXT,
            rookie_details TEXT
        )
    """)
    store_cols = [
        "team", "departed_epa", "departed_count", "departed_snaps",
        "incoming_epa", "incoming_count", "incoming_snaps",
        "rookie_epa", "rookie_count", "net_epa", "net_off_epa", "net_def_epa",
        "epa_per_play_adj",
        "departed_details", "incoming_details", "rookie_details",
    ]
    adj_df[store_cols].to_sql(
        "nfl_roster_adjustments", conn, if_exists="append", index=False
    )
    log.info(f"\nStored {len(adj_df)} team adjustments in nfl_roster_adjustments")


def _store_player_epa(player_epa: pd.DataFrame, conn: sqlite3.Connection):
    """Store player-level EPA for reference/display (offensive + defensive)."""
    conn.execute("DROP TABLE IF EXISTS nfl_player_epa")
    conn.execute("""
        CREATE TABLE nfl_player_epa (
            gsis_id TEXT,
            player_name TEXT,
            team TEXT,
            season INTEGER,
            position_group TEXT,
            side TEXT,
            total_epa REAL,
            epa_per_play REAL,
            snap_count INTEGER,
            games_played INTEGER,
            PRIMARY KEY (gsis_id, team, season, side)
        )
    """)
    store_cols = [
        "gsis_id", "player_name", "team", "season", "position_group",
        "side", "total_epa", "epa_per_play", "snap_count", "games_played",
    ]
    # Deduplicate: keep the row with highest snap_count per (gsis_id, team, season, side)
    deduped = player_epa[store_cols].sort_values("snap_count", ascending=False)
    deduped = deduped.drop_duplicates(subset=["gsis_id", "team", "season", "side"], keep="first")
    deduped.to_sql("nfl_player_epa", conn, if_exists="append", index=False)
    off_count = (deduped["side"] == "OFF").sum()
    def_count = (deduped["side"] == "DEF").sum()
    log.info(f"Stored {len(deduped)} player EPA rows ({off_count} OFF, {def_count} DEF)")


def main():
    log.info("=== NFL Roster Movement & EPA Adjustment Pipeline ===")
    log.info(f"Projecting from {FROM_SEASON} → {TO_SEASON}")

    # 1. Load rosters
    log.info("\n--- Loading Rosters ---")
    roster_from, roster_to = _load_rosters()

    # 2. Compute player-level EPA from PBP
    log.info("\n--- Computing Player EPA ---")
    player_epa = _compute_player_epa([FROM_SEASON])

    # 3. Detect movements
    log.info("\n--- Detecting Roster Movements ---")
    movements = _detect_movements(roster_from, roster_to)

    # 4. Compute team adjustments
    log.info("\n--- Computing Team EPA Adjustments ---")
    adjustments = _compute_team_adjustments(movements, player_epa, roster_to)

    # 5. Store
    log.info("\n--- Storing Results ---")
    conn = sqlite3.connect(NFL_DB)
    _store_player_epa(player_epa, conn)
    _store_adjustments(adjustments, conn)

    # 6. Show biggest movers
    log.info("\n=== Biggest Gainers ===")
    top = adjustments.nlargest(5, "net_epa")
    for _, r in top.iterrows():
        log.info(f"  {r['team']:3s}: {r['net_epa']:+.1f} EPA ({r['incoming_count']} in, {r['departed_count']} out)")

    log.info("\n=== Biggest Losers ===")
    bottom = adjustments.nsmallest(5, "net_epa")
    for _, r in bottom.iterrows():
        log.info(f"  {r['team']:3s}: {r['net_epa']:+.1f} EPA ({r['incoming_count']} in, {r['departed_count']} out)")

    conn.close()
    log.info("\nDone!")


if __name__ == "__main__":
    main()
