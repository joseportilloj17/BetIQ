#!/usr/bin/env python3
"""
backfill_handedness.py — Pitcher handedness + team batting platoon splits pipeline.

Three parts:
  A. Pitcher handedness backfill   (adds 'throws' column to pitcher_stats)
  B. Team batting splits backfill  (new 'team_batting_splits' table)
  C. Platoon feature export        (CSV for model testing — standalone, no features.py changes)

Data sources:
  - MLB Stats API /api/v1/people/{id}           → pitchHand.code (L/R)
  - MLB Stats API /teams/{id}/stats?stats=statSplits&sitCodes=vl/vr  → team batting vs LHP/RHP

Run:
  python3 backfill_handedness.py              # full pipeline
  python3 backfill_handedness.py --part A     # pitcher handedness only
  python3 backfill_handedness.py --part B     # team batting splits only
  python3 backfill_handedness.py --part C     # platoon feature export only
  python3 backfill_handedness.py --dry-run    # preview API calls without writing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict

import requests
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_handedness")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "historical.db")

# MLB team full name → Stats API team ID
TEAM_NAME_TO_ID: dict[str, int] = {
    "Los Angeles Angels":       108,
    "Arizona Diamondbacks":     109,
    "Baltimore Orioles":        110,
    "Boston Red Sox":           111,
    "Chicago Cubs":             112,
    "Cincinnati Reds":          113,
    "Cleveland Guardians":      114,
    "Colorado Rockies":         115,
    "Detroit Tigers":           116,
    "Houston Astros":           117,
    "Kansas City Royals":       118,
    "Los Angeles Dodgers":      119,
    "Washington Nationals":     120,
    "New York Mets":            121,
    "Athletics":                133,
    "Oakland Athletics":        133,
    "Pittsburgh Pirates":       134,
    "San Diego Padres":         135,
    "Seattle Mariners":         136,
    "San Francisco Giants":     137,
    "St. Louis Cardinals":      138,
    "Tampa Bay Rays":           139,
    "Texas Rangers":            140,
    "Toronto Blue Jays":        141,
    "Minnesota Twins":          142,
    "Philadelphia Phillies":    143,
    "Atlanta Braves":           144,
    "Chicago White Sox":        145,
    "Miami Marlins":            146,
    "New York Yankees":         147,
    "Milwaukee Brewers":        158,
}

SEASONS = ["2022", "2023", "2024", "2025", "2026"]


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: Pitcher Handedness Backfill
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_throws_column(conn: sqlite3.Connection) -> None:
    """Add 'throws' TEXT column to pitcher_stats if it doesn't exist."""
    cur = conn.execute("PRAGMA table_info(pitcher_stats)")
    cols = {row[1] for row in cur.fetchall()}
    if "throws" not in cols:
        conn.execute("ALTER TABLE pitcher_stats ADD COLUMN throws TEXT")
        conn.commit()
        log.info("Added 'throws' column to pitcher_stats")
    else:
        log.info("'throws' column already exists in pitcher_stats")


def _fetch_pitcher_handedness(player_id: int) -> str | None:
    """Fetch pitcher handedness from MLB Stats API. Returns 'L', 'R', or None."""
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        people = r.json().get("people", [])
        if people:
            return people[0].get("pitchHand", {}).get("code")
    except Exception as exc:
        log.debug("Failed to fetch handedness for player_id=%d: %s", player_id, exc)
    return None


def backfill_pitcher_handedness(conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """
    Backfill pitcher handedness for all rows in pitcher_stats.

    Returns summary dict with counts.
    """
    _ensure_throws_column(conn)

    # Get all unique player_ids that need handedness
    cur = conn.execute("""
        SELECT DISTINCT player_id, pitcher_name
        FROM pitcher_stats
        WHERE player_id IS NOT NULL
          AND (throws IS NULL OR throws = '')
    """)
    pending = cur.fetchall()

    # Also check already-filled ones for reporting
    cur2 = conn.execute("""
        SELECT COUNT(DISTINCT player_id)
        FROM pitcher_stats
        WHERE throws IS NOT NULL AND throws != ''
    """)
    already_done = cur2.fetchone()[0]

    log.info("Pitcher handedness: %d already done, %d pending", already_done, len(pending))

    if not pending:
        return {"already_done": already_done, "fetched": 0, "failed": 0}

    if dry_run:
        log.info("[DRY RUN] Would fetch handedness for %d pitchers", len(pending))
        return {"already_done": already_done, "pending": len(pending), "dry_run": True}

    fetched = 0
    failed = 0
    batch_size = 50  # commit every N pitchers

    for i, (player_id, name) in enumerate(pending):
        hand = _fetch_pitcher_handedness(player_id)
        if hand:
            conn.execute(
                "UPDATE pitcher_stats SET throws = ? WHERE player_id = ?",
                (hand, player_id),
            )
            fetched += 1
            if fetched % batch_size == 0:
                conn.commit()
                log.info("  Progress: %d/%d fetched (current: %s = %s)",
                         fetched, len(pending), name, hand)
        else:
            failed += 1
            log.warning("  Could not determine handedness for %s (id=%d)", name, player_id)

        # Rate limit: ~100ms between calls (MLB API is generous but be respectful)
        time.sleep(0.1)

    conn.commit()

    # Report coverage
    cur3 = conn.execute("""
        SELECT throws, COUNT(*) FROM pitcher_stats
        WHERE throws IS NOT NULL
        GROUP BY throws
    """)
    dist = dict(cur3.fetchall())
    log.info("Handedness distribution: %s", dist)

    summary = {
        "already_done": already_done,
        "fetched": fetched,
        "failed": failed,
        "distribution": dist,
    }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: Team Batting Splits (vs LHP / vs RHP)
# ═══════════════════════════════════════════════════════════════════════════════

def _create_splits_table(conn: sqlite3.Connection) -> None:
    """Create team_batting_splits table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_batting_splits (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name   TEXT    NOT NULL,
            season      TEXT    NOT NULL,
            vs_hand     TEXT    NOT NULL,   -- 'L' or 'R'
            ops         REAL,
            slg         REAL,
            obp         REAL,
            avg         REAL,
            k_pct       REAL,              -- strikeouts / plate appearances
            bb_pct      REAL,              -- walks / plate appearances
            hr          INTEGER,
            plate_appearances INTEGER,
            at_bats     INTEGER,
            babip       REAL,
            iso         REAL,              -- SLG - AVG (isolated power)
            UNIQUE(team_name, season, vs_hand)
        )
    """)
    conn.commit()
    log.info("team_batting_splits table ready")


def _safe_float(val, default: float = None) -> float | None:
    """Convert stat string like '.231' to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _fetch_team_split(team_id: int, season: str, vs_hand: str) -> dict | None:
    """
    Fetch team batting split from MLB Stats API.

    vs_hand: 'L' or 'R' — team's batting performance vs left/right-handed pitchers.
    sitCodes: 'vl' = vs left, 'vr' = vs right.
    """
    sit_code = "vl" if vs_hand == "L" else "vr"
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    params = {
        "stats": "statSplits",
        "group": "hitting",
        "season": season,
        "sitCodes": sit_code,
        "gameType": "R",
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.debug("Failed to fetch split for team %d, season %s, vs %s: %s",
                  team_id, season, vs_hand, exc)
        return None

    stats_list = data.get("stats", [])
    if not stats_list:
        return None

    splits = stats_list[0].get("splits", [])
    if not splits:
        return None

    s = splits[0].get("stat", {})
    pa = s.get("plateAppearances", 0)
    ab = s.get("atBats", 0)
    k = s.get("strikeOuts", 0)
    bb = s.get("baseOnBalls", 0)

    ops_val = _safe_float(s.get("ops"))
    slg_val = _safe_float(s.get("slg"))
    obp_val = _safe_float(s.get("obp"))
    avg_val = _safe_float(s.get("avg"))

    # Compute derived stats
    k_pct = k / pa if pa > 0 else None
    bb_pct = bb / pa if pa > 0 else None
    iso = (slg_val - avg_val) if slg_val is not None and avg_val is not None else None

    return {
        "ops": ops_val,
        "slg": slg_val,
        "obp": obp_val,
        "avg": avg_val,
        "k_pct": round(k_pct, 4) if k_pct is not None else None,
        "bb_pct": round(bb_pct, 4) if bb_pct is not None else None,
        "hr": s.get("homeRuns"),
        "plate_appearances": pa,
        "at_bats": ab,
        "babip": _safe_float(s.get("babip")),
        "iso": round(iso, 3) if iso is not None else None,
    }


def backfill_team_batting_splits(conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """
    Backfill team batting splits vs LHP and vs RHP for all teams x seasons.

    Returns summary dict.
    """
    _create_splits_table(conn)

    # Check what already exists
    cur = conn.execute("SELECT team_name, season, vs_hand FROM team_batting_splits")
    existing = {(r[0], r[1], r[2]) for r in cur.fetchall()}
    log.info("team_batting_splits: %d existing rows", len(existing))

    # Build list of (team_name, team_id, season, vs_hand) to fetch
    to_fetch = []
    for team_name, team_id in TEAM_NAME_TO_ID.items():
        if team_name == "Oakland Athletics":
            continue  # skip duplicate — "Athletics" covers it
        for season in SEASONS:
            for vs_hand in ["L", "R"]:
                if (team_name, season, vs_hand) not in existing:
                    to_fetch.append((team_name, team_id, season, vs_hand))

    log.info("team_batting_splits: %d new splits to fetch", len(to_fetch))

    if not to_fetch:
        return {"existing": len(existing), "fetched": 0}

    if dry_run:
        log.info("[DRY RUN] Would fetch %d team batting splits", len(to_fetch))
        return {"existing": len(existing), "pending": len(to_fetch), "dry_run": True}

    fetched = 0
    failed = 0

    for i, (team_name, team_id, season, vs_hand) in enumerate(to_fetch):
        stats = _fetch_team_split(team_id, season, vs_hand)
        if stats and stats.get("plate_appearances", 0) > 0:
            conn.execute("""
                INSERT OR REPLACE INTO team_batting_splits
                    (team_name, season, vs_hand, ops, slg, obp, avg,
                     k_pct, bb_pct, hr, plate_appearances, at_bats, babip, iso)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team_name, season, vs_hand,
                stats["ops"], stats["slg"], stats["obp"], stats["avg"],
                stats["k_pct"], stats["bb_pct"], stats["hr"],
                stats["plate_appearances"], stats["at_bats"],
                stats["babip"], stats["iso"],
            ))
            fetched += 1

            if fetched % 20 == 0:
                conn.commit()
                log.info("  Progress: %d/%d splits fetched (current: %s %s vs %sHP)",
                         fetched, len(to_fetch), team_name, season, vs_hand)
        else:
            failed += 1
            log.warning("  No split data for %s %s vs %sHP (team_id=%d)",
                        team_name, season, vs_hand, team_id)

        # Rate limit: ~150ms between calls
        time.sleep(0.15)

    conn.commit()

    # Report
    cur2 = conn.execute("""
        SELECT season, vs_hand, COUNT(*), ROUND(AVG(ops), 3)
        FROM team_batting_splits
        GROUP BY season, vs_hand
        ORDER BY season, vs_hand
    """)
    log.info("Splits summary:")
    for row in cur2.fetchall():
        log.info("  Season %s vs %sHP: %d teams, avg OPS=%.3f", row[0], row[1], row[2], row[3])

    return {"existing": len(existing), "fetched": fetched, "failed": failed}


# ═══════════════════════════════════════════════════════════════════════════════
# Part C: Platoon Feature Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_platoon_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Build and export platoon mismatch features per game.

    For each MLB game:
      1. Identify home/away starting pitcher from team_stats.stats_json
      2. Look up SP handedness from pitcher_stats.throws
      3. Look up team batting split matching SP handedness
      4. Compute platoon features

    Returns DataFrame and writes CSV to data/platoon_features.csv.
    """
    log.info("Building platoon feature matrix...")

    # 1. Load pitcher handedness lookup: pitcher_name → throws
    pitcher_hand = pd.read_sql_query(
        "SELECT pitcher_name, season, throws FROM pitcher_stats WHERE throws IS NOT NULL",
        conn,
    )
    pitcher_hand["season"] = pitcher_hand["season"].astype(str)
    # Some pitchers appear multiple seasons — throws doesn't change
    # Deduplicate: take any non-null throws per name (career-level)
    pitcher_hand_career = (
        pitcher_hand.dropna(subset=["throws"])
        .drop_duplicates(subset=["pitcher_name"], keep="first")
        .set_index("pitcher_name")["throws"]
        .to_dict()
    )
    log.info("  Pitcher handedness lookup: %d pitchers", len(pitcher_hand_career))

    # Season-specific lookup: (name, season) → throws
    pitcher_hand_season = dict(zip(
        zip(pitcher_hand["pitcher_name"], pitcher_hand["season"]),
        pitcher_hand["throws"],
    ))

    # 2. Load team batting splits: (team_name, season, vs_hand) → stats
    splits_df = pd.read_sql_query(
        "SELECT team_name, season, vs_hand, ops, slg, obp, avg, k_pct, bb_pct, iso FROM team_batting_splits",
        conn,
    )
    splits_df["season"] = splits_df["season"].astype(str)
    splits_idx = splits_df.set_index(["team_name", "season", "vs_hand"])

    # 3. Load team overall batting: (team_name, season) → stats
    overall_df = pd.read_sql_query(
        "SELECT team_name, season, ops, slg, obp, avg, k_pct, bb_pct FROM team_batting_stats",
        conn,
    )
    overall_df["season"] = overall_df["season"].astype(str)
    overall_idx = overall_df.set_index(["team_name", "season"])

    # 4. Load games + SP names
    sp_query = """
        SELECT
            g.game_id, g.season, g.game_date,
            g.home_team, g.away_team,
            MAX(CASE WHEN ts.is_home=1
                THEN json_extract(ts.stats_json, '$.home_probable_pitcher')
                END) AS h_sp_name,
            MAX(CASE WHEN ts.is_home=0
                THEN json_extract(ts.stats_json, '$.away_probable_pitcher')
                END) AS a_sp_name
        FROM games g
        JOIN team_stats ts ON g.game_id = ts.game_id
        WHERE g.sport = 'MLB'
          AND g.status = 'Final'
        GROUP BY g.game_id
        ORDER BY g.game_date
    """
    games = pd.read_sql_query(sp_query, conn)
    games["season"] = games["season"].astype(str)
    log.info("  Games with SP info: %d", len(games))

    # 5. Build features
    rows = []
    for _, g in games.iterrows():
        game_id = g["game_id"]
        season = g["season"]
        h_team = g["home_team"]
        a_team = g["away_team"]
        h_sp = g["h_sp_name"]
        a_sp = g["a_sp_name"]

        # Look up SP handedness (prefer season-specific, fall back to career)
        h_sp_throws = None
        a_sp_throws = None
        if h_sp and not pd.isna(h_sp):
            h_sp_throws = pitcher_hand_season.get((h_sp, season)) or pitcher_hand_career.get(h_sp)
        if a_sp and not pd.isna(a_sp):
            a_sp_throws = pitcher_hand_season.get((a_sp, season)) or pitcher_hand_career.get(a_sp)

        row = {
            "game_id": game_id,
            "game_date": g["game_date"],
            "season": season,
            "home_team": h_team,
            "away_team": a_team,
            "h_sp_name": h_sp,
            "a_sp_name": a_sp,
            "h_sp_throws": h_sp_throws,   # home starter's throwing hand
            "a_sp_throws": a_sp_throws,   # away starter's throwing hand
        }

        # Home team's batting vs away SP's hand
        # (if away SP throws R, home team needs their vs-RHP split)
        if a_sp_throws:
            try:
                split = splits_idx.loc[(h_team, season, a_sp_throws)]
                row["h_ops_vs_sp_hand"] = split["ops"]
                row["h_slg_vs_sp_hand"] = split["slg"]
                row["h_obp_vs_sp_hand"] = split["obp"]
                row["h_avg_vs_sp_hand"] = split["avg"]
                row["h_kpct_vs_sp_hand"] = split["k_pct"]
                row["h_bbpct_vs_sp_hand"] = split["bb_pct"]
                row["h_iso_vs_sp_hand"] = split["iso"]
            except KeyError:
                pass

        # Away team's batting vs home SP's hand
        if h_sp_throws:
            try:
                split = splits_idx.loc[(a_team, season, h_sp_throws)]
                row["a_ops_vs_sp_hand"] = split["ops"]
                row["a_slg_vs_sp_hand"] = split["slg"]
                row["a_obp_vs_sp_hand"] = split["obp"]
                row["a_avg_vs_sp_hand"] = split["avg"]
                row["a_kpct_vs_sp_hand"] = split["k_pct"]
                row["a_bbpct_vs_sp_hand"] = split["bb_pct"]
                row["a_iso_vs_sp_hand"] = split["iso"]
            except KeyError:
                pass

        # Overall team batting for computing gaps
        try:
            h_overall = overall_idx.loc[(h_team, season)]
            row["h_ops_overall"] = h_overall["ops"]
        except KeyError:
            pass

        try:
            a_overall = overall_idx.loc[(a_team, season)]
            row["a_ops_overall"] = a_overall["ops"]
        except KeyError:
            pass

        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("  Raw platoon rows: %d", len(df))

    # Compute derived features
    # platoon_diff: whose batting lineup has the better matchup vs opposing SP?
    df["platoon_ops_diff"] = df.get("h_ops_vs_sp_hand", np.nan) - df.get("a_ops_vs_sp_hand", np.nan)
    df["platoon_slg_diff"] = df.get("h_slg_vs_sp_hand", np.nan) - df.get("a_slg_vs_sp_hand", np.nan)

    # platoon_gap: how much better/worse is this team vs THIS hand compared to overall?
    df["h_platoon_gap"] = df.get("h_ops_vs_sp_hand", np.nan) - df.get("h_ops_overall", np.nan)
    df["a_platoon_gap"] = df.get("a_ops_vs_sp_hand", np.nan) - df.get("a_ops_overall", np.nan)
    df["platoon_gap_diff"] = df.get("h_platoon_gap", np.nan) - df.get("a_platoon_gap", np.nan)

    # Binary: is SP a lefty? (lefty starters are ~28% of starts — distinct advantage)
    df["h_sp_is_lhp"] = (df["h_sp_throws"] == "L").astype(float)
    df["a_sp_is_lhp"] = (df["a_sp_throws"] == "L").astype(float)

    # Fill rate reporting
    total = len(df)
    for col in ["h_sp_throws", "a_sp_throws", "h_ops_vs_sp_hand", "a_ops_vs_sp_hand",
                "h_platoon_gap", "a_platoon_gap"]:
        if col in df.columns:
            fill = df[col].notna().mean()
            log.info("  Fill rate %-25s: %5.1f%% (%d/%d)", col, fill * 100,
                     df[col].notna().sum(), total)

    # Handedness distribution
    if "h_sp_throws" in df.columns:
        h_dist = df["h_sp_throws"].value_counts(dropna=False)
        log.info("  Home SP handedness: %s", dict(h_dist))
    if "a_sp_throws" in df.columns:
        a_dist = df["a_sp_throws"].value_counts(dropna=False)
        log.info("  Away SP handedness: %s", dict(a_dist))

    # Write CSV
    out_path = os.path.join(os.path.dirname(__file__), "data", "platoon_features.csv")
    # Select only the model-ready columns for the CSV
    export_cols = [
        "game_id", "game_date", "season", "home_team", "away_team",
        "h_sp_name", "a_sp_name", "h_sp_throws", "a_sp_throws",
        "h_ops_vs_sp_hand", "a_ops_vs_sp_hand",
        "h_slg_vs_sp_hand", "a_slg_vs_sp_hand",
        "h_obp_vs_sp_hand", "a_obp_vs_sp_hand",
        "h_avg_vs_sp_hand", "a_avg_vs_sp_hand",
        "h_kpct_vs_sp_hand", "a_kpct_vs_sp_hand",
        "h_bbpct_vs_sp_hand", "a_bbpct_vs_sp_hand",
        "h_iso_vs_sp_hand", "a_iso_vs_sp_hand",
        "h_ops_overall", "a_ops_overall",
        "platoon_ops_diff", "platoon_slg_diff",
        "h_platoon_gap", "a_platoon_gap", "platoon_gap_diff",
        "h_sp_is_lhp", "a_sp_is_lhp",
    ]
    existing_cols = [c for c in export_cols if c in df.columns]
    df[existing_cols].to_csv(out_path, index=False)
    log.info("Wrote platoon features to %s (%d rows, %d cols)", out_path, len(df), len(existing_cols))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Verification & Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def verify_data(conn: sqlite3.Connection) -> None:
    """Run diagnostic checks on the backfilled data."""
    log.info("\n" + "=" * 60)
    log.info("VERIFICATION")
    log.info("=" * 60)

    # A. Pitcher handedness
    cur = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN throws IS NOT NULL THEN 1 ELSE 0 END) as filled,
            SUM(CASE WHEN throws = 'R' THEN 1 ELSE 0 END) as right_hand,
            SUM(CASE WHEN throws = 'L' THEN 1 ELSE 0 END) as left_hand
        FROM pitcher_stats
    """)
    row = cur.fetchone()
    log.info("Pitcher handedness:")
    log.info("  Total rows:    %d", row[0])
    log.info("  Filled:        %d (%.1f%%)", row[1], row[1] / max(row[0], 1) * 100)
    log.info("  Right-handed:  %d", row[2])
    log.info("  Left-handed:   %d", row[3])

    # B. Team batting splits
    try:
        cur2 = conn.execute("""
            SELECT season, vs_hand, COUNT(*), ROUND(AVG(ops), 3), ROUND(MIN(ops), 3), ROUND(MAX(ops), 3)
            FROM team_batting_splits
            GROUP BY season, vs_hand
            ORDER BY season, vs_hand
        """)
        log.info("\nTeam batting splits:")
        for row in cur2.fetchall():
            log.info("  %s vs %sHP: %2d teams, OPS avg=%.3f [%.3f-%.3f]",
                     row[0], row[1], row[2], row[3], row[4], row[5])
    except sqlite3.OperationalError:
        log.info("\nTeam batting splits: table not yet created")

    # C. Cross-reference: how many games can we build platoon features for?
    try:
        cur3 = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT g.game_id
                FROM games g
                JOIN team_stats ts ON g.game_id = ts.game_id AND ts.is_home = 1
                WHERE g.sport = 'MLB' AND g.status = 'Final'
                  AND ts.stats_json IS NOT NULL
                  AND ts.stats_json != ''
                  AND json_valid(ts.stats_json)
                  AND json_extract(ts.stats_json, '$.home_probable_pitcher') IS NOT NULL
            )
        """)
        games_with_sp = cur3.fetchone()[0]

        cur4 = conn.execute("SELECT COUNT(*) FROM games WHERE sport='MLB' AND status='Final'")
        total_games = cur4.fetchone()[0]

        log.info("\nPlatoon feature coverage:")
        log.info("  Total Final MLB games: %d", total_games)
        log.info("  Games with SP name:    %d (%.1f%%)",
                 games_with_sp, games_with_sp / max(total_games, 1) * 100)

        # Check coverage of handedness for SPs in games
        cur5 = conn.execute("""
            SELECT
                COUNT(*) as total_sp_refs,
                SUM(CASE WHEN ps.throws IS NOT NULL THEN 1 ELSE 0 END) as with_hand
            FROM (
                SELECT json_extract(ts.stats_json, '$.home_probable_pitcher') as sp_name,
                       g.season
                FROM team_stats ts
                JOIN games g ON ts.game_id = g.game_id
                WHERE g.sport = 'MLB' AND ts.is_home = 1 AND g.status = 'Final'
                  AND ts.stats_json IS NOT NULL
                  AND ts.stats_json != ''
                  AND json_valid(ts.stats_json)
                  AND json_extract(ts.stats_json, '$.home_probable_pitcher') IS NOT NULL
            ) sp
            LEFT JOIN pitcher_stats ps ON sp.sp_name = ps.pitcher_name AND sp.season = ps.season
        """)
        row = cur5.fetchone()
        if row and row[0] > 0:
            log.info("  SP names matching pitcher_stats with handedness: %d/%d (%.1f%%)",
                     row[1], row[0], row[1] / row[0] * 100)
    except Exception as exc:
        log.warning("Cross-reference check failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Backfill pitcher handedness and platoon splits")
    parser.add_argument("--part", choices=["A", "B", "C", "verify"], default=None,
                        help="Run only a specific part (A=handedness, B=splits, C=export, verify)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview API calls without writing to database")
    parser.add_argument("--db", default=DB_PATH,
                        help=f"Path to historical.db (default: {DB_PATH})")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        log.error("Database not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        parts = [args.part] if args.part else ["A", "B", "C", "verify"]

        if "A" in parts:
            log.info("\n" + "=" * 60)
            log.info("PART A: Pitcher Handedness Backfill")
            log.info("=" * 60)
            result = backfill_pitcher_handedness(conn, dry_run=args.dry_run)
            log.info("Part A result: %s", result)

        if "B" in parts:
            log.info("\n" + "=" * 60)
            log.info("PART B: Team Batting Splits Backfill")
            log.info("=" * 60)
            result = backfill_team_batting_splits(conn, dry_run=args.dry_run)
            log.info("Part B result: %s", result)

        if "C" in parts:
            log.info("\n" + "=" * 60)
            log.info("PART C: Platoon Feature Export")
            log.info("=" * 60)
            if args.dry_run:
                log.info("[DRY RUN] Would export platoon features")
            else:
                df = export_platoon_features(conn)
                log.info("Part C: exported %d rows", len(df))

        if "verify" in parts:
            verify_data(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
