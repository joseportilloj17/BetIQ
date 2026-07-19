"""
backfill_umpires.py — Backfill home plate umpire assignments + tendency features.

Uses MLB Stats API schedule endpoint with hydrate=officials.
Bulk-fetches by month (~30 API calls for 5 seasons).

Features computed:
  - ump_career_rpg_above_avg: career runs-per-game relative to league average
  - ump_season_rpg_above_avg: season-to-date runs-per-game above average

Point-in-time: ump assignments are ANNOUNCED pre-game, so this is clean PIT.
Tendency stats use only PRIOR games (rolling lookback, no lookahead).

Usage:
    .venv/bin/python backend/backfill_umpires.py
"""
import os
import sys
import time
import sqlite3
import requests
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "historical.db")
_MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def _fetch_umpires_range(start_date: str, end_date: str) -> list[dict]:
    """Fetch all game umpire assignments for a date range via schedule+hydrate."""
    try:
        r = requests.get(_MLB_SCHEDULE_URL, params={
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "hydrate": "officials",
            "gameType": "R",  # regular season only
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("  API fail %s to %s: %s", start_date, end_date, exc)
        return []

    results = []
    for date_entry in data.get("dates", []):
        game_date = date_entry["date"]
        for game in date_entry.get("games", []):
            game_pk = game.get("gamePk")
            status = game.get("status", {}).get("abstractGameState", "")

            # Get teams
            home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")

            # Get home plate umpire
            officials = game.get("officials", [])
            hp_ump = None
            for off in officials:
                if off.get("officialType") == "Home Plate":
                    hp_ump = off.get("official", {})
                    break

            if hp_ump:
                results.append({
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "hp_ump_id": hp_ump.get("id"),
                    "hp_ump_name": hp_ump.get("fullName", ""),
                    "status": status,
                })

    return results


def _fetch_game_scores(conn) -> dict:
    """Get (game_date, home_team) → total_runs from games table."""
    rows = conn.execute("""
        SELECT game_date, home_team, home_score, away_score
        FROM games
        WHERE sport = 'MLB' AND status = 'Final'
          AND home_score IS NOT NULL AND away_score IS NOT NULL
    """).fetchall()
    scores = {}
    for gd, ht, hs, aws in rows:
        scores[(gd, ht)] = (hs or 0) + (aws or 0)
    return scores


def backfill():
    conn = sqlite3.connect(DB_PATH)

    # Create umpire table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_umpires (
            game_pk INTEGER,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT,
            hp_ump_id INTEGER,
            hp_ump_name TEXT,
            UNIQUE(game_date, home_team)
        )
    """)
    conn.commit()

    # Check existing coverage
    existing = set(conn.execute(
        "SELECT game_date, home_team FROM game_umpires"
    ).fetchall())
    log.info("Existing umpire rows: %d", len(existing))

    # Determine date range from games table
    min_date, max_date = conn.execute("""
        SELECT MIN(game_date), MAX(game_date) FROM games
        WHERE sport = 'MLB' AND status = 'Final'
    """).fetchone()
    log.info("Games range: %s to %s", min_date, max_date)

    # Fetch in monthly chunks
    start = datetime.strptime(min_date, "%Y-%m-%d")
    end = datetime.strptime(max_date, "%Y-%m-%d")

    total_written = 0
    total_skipped = 0
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=30), end)
        s_str = chunk_start.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")

        games = _fetch_umpires_range(s_str, e_str)

        new_rows = []
        for g in games:
            key = (g["game_date"], g["home_team"])
            if key in existing:
                total_skipped += 1
                continue
            new_rows.append((
                g["game_pk"], g["game_date"], g["home_team"],
                g["away_team"], g["hp_ump_id"], g["hp_ump_name"],
            ))
            existing.add(key)

        if new_rows:
            conn.executemany("""
                INSERT OR IGNORE INTO game_umpires
                    (game_pk, game_date, home_team, away_team, hp_ump_id, hp_ump_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, new_rows)
            conn.commit()
            total_written += len(new_rows)

        log.info("  %s to %s: %d fetched, %d new", s_str, e_str, len(games), len(new_rows))
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    # Summary
    total_rows = conn.execute("SELECT COUNT(*) FROM game_umpires").fetchone()[0]
    n_umps = conn.execute("SELECT COUNT(DISTINCT hp_ump_id) FROM game_umpires").fetchone()[0]
    log.info("Umpire backfill complete: %d written, %d skipped, %d total rows, %d unique umps",
             total_written, total_skipped, total_rows, n_umps)

    # ── Compute tendency features ────────────────────────────────────────────
    log.info("Computing umpire tendency features...")
    _compute_tendencies(conn)

    conn.close()


def _compute_tendencies(conn):
    """
    Compute point-in-time umpire tendency: runs-per-game above league average.

    For each game, we compute the HP umpire's PRIOR career and season RPG
    deviation from league average. This is strictly PIT — only uses games
    that occurred BEFORE the current game.
    """
    # Get all games with scores and ump assignments
    rows = conn.execute("""
        SELECT gu.game_date, gu.home_team, gu.hp_ump_id,
               g.home_score, g.away_score
        FROM game_umpires gu
        JOIN games g ON g.game_date = gu.game_date
                    AND g.home_team = gu.home_team
                    AND g.sport = 'MLB' AND g.status = 'Final'
        WHERE gu.hp_ump_id IS NOT NULL
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ORDER BY gu.game_date
    """).fetchall()

    log.info("  Computing tendencies for %d games with ump + scores", len(rows))

    # Add column if needed
    cols = [c[1] for c in conn.execute("PRAGMA table_info(game_umpires)").fetchall()]
    for col in ["ump_career_rpg_above_avg", "ump_season_rpg_above_avg", "ump_career_games"]:
        if col not in cols:
            conn.execute(f"ALTER TABLE game_umpires ADD COLUMN {col} REAL")
    conn.commit()

    # Process chronologically — strictly PIT
    # Track per-ump: list of total_runs in games they've worked
    ump_career_games = defaultdict(list)  # ump_id → [(game_date, total_runs), ...]
    league_all_runs = []  # all total_runs for league average

    updates = []
    for game_date, home_team, hp_ump_id, home_score, away_score in rows:
        total_runs = (home_score or 0) + (away_score or 0)
        season = game_date[:4]

        # PRIOR stats (strictly before this game)
        prior_career = ump_career_games[hp_ump_id]

        # Career tendency
        if len(prior_career) >= 10:
            ump_career_rpg = sum(r for _, r in prior_career) / len(prior_career)
            league_avg_rpg = sum(league_all_runs) / len(league_all_runs) if league_all_runs else ump_career_rpg
            career_above = ump_career_rpg - league_avg_rpg
        else:
            career_above = None  # insufficient data

        # Season tendency
        prior_season = [(d, r) for d, r in prior_career if d[:4] == season]
        if len(prior_season) >= 5:
            ump_season_rpg = sum(r for _, r in prior_season) / len(prior_season)
            # Season league average (all games this season before this date)
            season_league_runs = [r for i, r in enumerate(league_all_runs)
                                  if i < len(league_all_runs)]  # all prior
            # More precise: track season separately
            league_season_rpg = ump_season_rpg  # fallback
            season_league = [(d, r) for d, r in zip(
                [row[0] for row in rows[:len(league_all_runs)]],
                league_all_runs
            ) if d[:4] == season]
            if season_league:
                league_season_rpg = sum(r for _, r in season_league) / len(season_league)
            season_above = ump_season_rpg - league_season_rpg
        else:
            season_above = None

        n_career = len(prior_career)

        updates.append((career_above, season_above, float(n_career) if n_career else None,
                        game_date, home_team))

        # Update running state AFTER computing (PIT)
        ump_career_games[hp_ump_id].append((game_date, total_runs))
        league_all_runs.append(total_runs)

    # Batch update
    conn.executemany("""
        UPDATE game_umpires
        SET ump_career_rpg_above_avg = ?,
            ump_season_rpg_above_avg = ?,
            ump_career_games = ?
        WHERE game_date = ? AND home_team = ?
    """, updates)
    conn.commit()

    # Stats
    filled_career = conn.execute(
        "SELECT COUNT(*) FROM game_umpires WHERE ump_career_rpg_above_avg IS NOT NULL"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM game_umpires").fetchone()[0]
    log.info("  Tendency fill: career=%d/%d (%.1f%%), season computed",
             filled_career, total, 100 * filled_career / total if total else 0)

    # Show top/bottom umpires
    top = conn.execute("""
        SELECT hp_ump_name, AVG(ump_career_rpg_above_avg) as avg_above, COUNT(*) as n
        FROM game_umpires
        WHERE ump_career_rpg_above_avg IS NOT NULL
        GROUP BY hp_ump_id
        HAVING n >= 50
        ORDER BY avg_above DESC
        LIMIT 5
    """).fetchall()
    log.info("  Top run-friendly umps (career):")
    for name, above, n in top:
        log.info("    %s: +%.2f RPG above avg (%d games)", name, above, n)

    bottom = conn.execute("""
        SELECT hp_ump_name, AVG(ump_career_rpg_above_avg) as avg_above, COUNT(*) as n
        FROM game_umpires
        WHERE ump_career_rpg_above_avg IS NOT NULL
        GROUP BY hp_ump_id
        HAVING n >= 50
        ORDER BY avg_above ASC
        LIMIT 5
    """).fetchall()
    log.info("  Top run-suppressing umps (career):")
    for name, above, n in bottom:
        log.info("    %s: %.2f RPG above avg (%d games)", name, above, n)


if __name__ == "__main__":
    backfill()
