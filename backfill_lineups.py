#!/usr/bin/env python3
"""
backfill_lineups.py — Fetch historical lineups + individual batter game stats
from MLB API boxscores. Store in historical.db for lineup composition features.

Two tables created:
  game_lineups:       game_pk, team_name, is_home, player_id, batting_order, position
  batter_game_stats:  game_pk, player_id, player_name, team_name, is_home,
                      ab, hits, doubles, triples, hr, bb, hbp, sf, k, tb

Usage:
  python3 backfill_lineups.py              # full backfill
  python3 backfill_lineups.py --resume     # resume from last fetched game_pk
  python3 backfill_lineups.py --check      # check coverage
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_lineups")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "historical.db")


def create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_lineups (
            game_pk     INTEGER NOT NULL,
            team_name   TEXT    NOT NULL,
            is_home     INTEGER NOT NULL,
            player_id   INTEGER NOT NULL,
            batting_order INTEGER NOT NULL,
            position    TEXT,
            PRIMARY KEY (game_pk, player_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batter_game_stats (
            game_pk     INTEGER NOT NULL,
            player_id   INTEGER NOT NULL,
            player_name TEXT,
            team_name   TEXT    NOT NULL,
            is_home     INTEGER NOT NULL,
            ab          INTEGER DEFAULT 0,
            hits        INTEGER DEFAULT 0,
            doubles     INTEGER DEFAULT 0,
            triples     INTEGER DEFAULT 0,
            hr          INTEGER DEFAULT 0,
            bb          INTEGER DEFAULT 0,
            hbp         INTEGER DEFAULT 0,
            sf          INTEGER DEFAULT 0,
            k           INTEGER DEFAULT 0,
            tb          INTEGER DEFAULT 0,
            PRIMARY KEY (game_pk, player_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bgs_player ON batter_game_stats(player_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gl_gamepk ON game_lineups(game_pk)")
    conn.commit()
    log.info("Tables ready")


def get_game_pks(conn: sqlite3.Connection) -> list[int]:
    """Get all game_pks from team_game_batting."""
    cur = conn.execute("SELECT DISTINCT game_pk FROM team_game_batting ORDER BY game_pk")
    return [r[0] for r in cur.fetchall()]


def get_fetched_game_pks(conn: sqlite3.Connection) -> set[int]:
    """Get game_pks already in batter_game_stats."""
    try:
        cur = conn.execute("SELECT DISTINCT game_pk FROM batter_game_stats")
        return {r[0] for r in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()


def fetch_boxscore(game_pk: int) -> dict | None:
    """Fetch boxscore from MLB API."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception:
        return None


def parse_boxscore(game_pk: int, data: dict) -> tuple[list[dict], list[dict]]:
    """
    Parse boxscore into lineup rows and batter stat rows.

    Returns (lineup_rows, batter_rows).
    """
    lineup_rows = []
    batter_rows = []

    teams = data.get("teams", {})
    for side, is_home in [("home", 1), ("away", 0)]:
        team_data = teams.get(side, {})
        team_name = team_data.get("team", {}).get("name", "Unknown")
        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        for pid in batting_order:
            pkey = f"ID{pid}"
            pdata = players.get(pkey, {})
            person = pdata.get("person", {})
            name = person.get("fullName", "")
            bo = pdata.get("battingOrder", 0)
            pos = pdata.get("position", {}).get("abbreviation", "")

            batting = pdata.get("stats", {}).get("batting", {})
            ab = batting.get("atBats", 0)
            hits = batting.get("hits", 0)
            doubles = batting.get("doubles", 0)
            triples = batting.get("triples", 0)
            hr = batting.get("homeRuns", 0)
            bb = batting.get("baseOnBalls", 0)
            hbp = batting.get("hitByPitch", 0)
            sf = batting.get("sacFlies", 0)
            k = batting.get("strikeOuts", 0)
            # Total bases: 1B + 2*2B + 3*3B + 4*HR
            singles = hits - doubles - triples - hr
            tb = singles + 2 * doubles + 3 * triples + 4 * hr

            lineup_rows.append({
                "game_pk": game_pk,
                "team_name": team_name,
                "is_home": is_home,
                "player_id": pid,
                "batting_order": bo,
                "position": pos,
            })

            batter_rows.append({
                "game_pk": game_pk,
                "player_id": pid,
                "player_name": name,
                "team_name": team_name,
                "is_home": is_home,
                "ab": ab,
                "hits": hits,
                "doubles": doubles,
                "triples": triples,
                "hr": hr,
                "bb": bb,
                "hbp": hbp,
                "sf": sf,
                "k": k,
                "tb": tb,
            })

    return lineup_rows, batter_rows


def insert_rows(conn: sqlite3.Connection, lineup_rows: list[dict],
                batter_rows: list[dict]) -> None:
    """Insert parsed rows into DB."""
    for lr in lineup_rows:
        conn.execute("""
            INSERT OR IGNORE INTO game_lineups
                (game_pk, team_name, is_home, player_id, batting_order, position)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (lr["game_pk"], lr["team_name"], lr["is_home"],
              lr["player_id"], lr["batting_order"], lr["position"]))

    for br in batter_rows:
        conn.execute("""
            INSERT OR IGNORE INTO batter_game_stats
                (game_pk, player_id, player_name, team_name, is_home,
                 ab, hits, doubles, triples, hr, bb, hbp, sf, k, tb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (br["game_pk"], br["player_id"], br["player_name"],
              br["team_name"], br["is_home"],
              br["ab"], br["hits"], br["doubles"], br["triples"],
              br["hr"], br["bb"], br["hbp"], br["sf"], br["k"], br["tb"]))


def backfill(resume: bool = False) -> None:
    """Main backfill loop."""
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    all_pks = get_game_pks(conn)
    fetched = get_fetched_game_pks(conn) if resume else set()

    to_fetch = [pk for pk in all_pks if pk not in fetched]
    log.info("Total games: %d, already fetched: %d, remaining: %d",
             len(all_pks), len(fetched), len(to_fetch))

    if not to_fetch:
        log.info("Nothing to fetch — all games already backfilled")
        return

    success = 0
    failed = 0
    batch_size = 50

    for i, game_pk in enumerate(to_fetch):
        data = fetch_boxscore(game_pk)
        if data:
            lineup_rows, batter_rows = parse_boxscore(game_pk, data)
            if lineup_rows:
                insert_rows(conn, lineup_rows, batter_rows)
                success += 1
            else:
                failed += 1
        else:
            failed += 1

        # Commit in batches
        if (i + 1) % batch_size == 0:
            conn.commit()
            elapsed_pct = (i + 1) / len(to_fetch) * 100
            log.info("  Progress: %d/%d (%.1f%%) — %d success, %d failed",
                     i + 1, len(to_fetch), elapsed_pct, success, failed)

        # Rate limit
        time.sleep(0.12)

    conn.commit()
    conn.close()

    log.info("Backfill complete: %d success, %d failed out of %d",
             success, failed, len(to_fetch))


def check_coverage() -> None:
    """Report coverage stats."""
    conn = sqlite3.connect(DB_PATH)

    try:
        cur = conn.execute("SELECT COUNT(DISTINCT game_pk) FROM batter_game_stats")
        games_with_stats = cur.fetchone()[0]
    except sqlite3.OperationalError:
        games_with_stats = 0

    cur = conn.execute("SELECT COUNT(DISTINCT game_pk) FROM team_game_batting")
    total_games = cur.fetchone()[0]

    print(f"Games with batter stats: {games_with_stats}/{total_games} "
          f"({games_with_stats/total_games*100:.1f}%)")

    if games_with_stats > 0:
        cur = conn.execute("SELECT COUNT(DISTINCT player_id) FROM batter_game_stats")
        print(f"Unique batters: {cur.fetchone()[0]}")

        cur = conn.execute("SELECT COUNT(*) FROM batter_game_stats")
        print(f"Total batter-game rows: {cur.fetchone()[0]}")

        cur = conn.execute("SELECT COUNT(DISTINCT game_pk) FROM game_lineups")
        print(f"Games with lineups: {cur.fetchone()[0]}")

        # Check lineup sizes
        cur = conn.execute("""
            SELECT game_pk, team_name, COUNT(*) as n
            FROM game_lineups
            GROUP BY game_pk, team_name
            HAVING n != 9
            LIMIT 5
        """)
        odd = cur.fetchall()
        if odd:
            print(f"Games with non-9 lineups: {len(odd)} (sample: {odd[:3]})")
        else:
            print("All lineups have exactly 9 players")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last fetched game")
    parser.add_argument("--check", action="store_true",
                        help="Check coverage only")
    args = parser.parse_args()

    if args.check:
        check_coverage()
    else:
        backfill(resume=args.resume)
