"""
pi_ratings.py — Margin-based Pi rating engine for MLB (home/away split).

Pi ratings are a team-strength signal that captures opponent-quality-adjusted
performance.  Unlike rolling win_pct (which treats all opponents equally),
Pi ratings update proportionally to how much a team over- or under-performed
against their opponent's strength.

Key design: separate home_pi and away_pi per team.
  - home_pi captures a team's strength when playing at home.
  - away_pi captures a team's strength on the road.
  - Expected margin for a game = home_team.home_pi - away_team.away_pi
  - This naturally captures home-field advantage (e.g. Coors effect:
    Rockies may have high home_pi but low away_pi).

All ratings start at 0.0 (zero-centered; positive = above average, negative = below).

Storage
-------
  pi_ratings table in historical.db:
      team, sport, home_pi, away_pi, games_home, games_away, last_updated
      PRIMARY KEY (team, sport)

  pi_rating_history table in historical.db:
      team, sport, game_date, home_pi, away_pi, games_home, games_away
      PRIMARY KEY (team, sport, game_date)
      One row per team per game_date — snapshot AFTER that date's games.

Public API
----------
  backfill_pi_ratings(sport, season_start, conn) -> dict
      Process all games, store ratings + history. Returns summary.

  get_team_pi_for_date(team, game_date, sport, conn) -> dict | None
      Look up Pi rating snapshot BEFORE game_date (no lookahead).

  get_all_pi_for_date(game_date, sport, conn) -> dict[str, dict]
      All teams' Pi ratings as of game_date.

  update_pi_after_settlement(games, sport, conn) -> int
      Incremental update after settling today's games.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

# ── Hyperparameters ──────────────────────────────────────────────────────────

LEARNING_RATE = 0.10    # MLB has 162 games per team — moderate rate
MARGIN_CAP    = 8.0     # Cap blowouts (10-0 counts as 8-0)


def _clamp(margin: float) -> float:
    return max(-MARGIN_CAP, min(MARGIN_CAP, margin))


# ── Schema setup ─────────────────────────────────────────────────────────────

def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create Pi rating tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pi_ratings (
            team         TEXT NOT NULL,
            sport        TEXT NOT NULL,
            home_pi      REAL DEFAULT 0.0,
            away_pi      REAL DEFAULT 0.0,
            games_home   INTEGER DEFAULT 0,
            games_away   INTEGER DEFAULT 0,
            last_updated TEXT,
            PRIMARY KEY (team, sport)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pi_rating_history (
            team         TEXT NOT NULL,
            sport        TEXT NOT NULL,
            game_date    TEXT NOT NULL,
            home_pi      REAL DEFAULT 0.0,
            away_pi      REAL DEFAULT 0.0,
            games_home   INTEGER DEFAULT 0,
            games_away   INTEGER DEFAULT 0,
            PRIMARY KEY (team, sport, game_date)
        )
    """)
    conn.commit()


# ── Core computation ─────────────────────────────────────────────────────────

def backfill_pi_ratings(
    sport: str = "MLB",
    season_start: str = "2026-03-27",
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """
    Process all completed games from season_start onward.
    Stores current ratings in pi_ratings and history in pi_rating_history.

    Returns summary dict with team count, game count, top/bottom teams.
    """
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(os.path.abspath(DB_PATH))

    try:
        _ensure_tables(conn)

        # Load completed games
        games = pd.read_sql_query("""
            SELECT game_id, game_date, home_team, away_team,
                   home_score, away_score
            FROM games
            WHERE sport = ?
              AND game_date >= ?
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY game_date, game_id
        """, conn, params=[sport, season_start])

        games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
        games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
        games = games.dropna(subset=["home_score", "away_score"])
        # Filter unplayed (both 0)
        games = games[~((games["home_score"] == 0) & (games["away_score"] == 0))]

        log.info("Pi backfill: %d %s games from %s", len(games), sport, season_start)
        if games.empty:
            return {"error": "No completed games found", "games": 0}

        # Initialize all teams at 0.0
        all_teams = set(games["home_team"].unique()) | set(games["away_team"].unique())
        ratings = {t: {"home_pi": 0.0, "away_pi": 0.0,
                       "games_home": 0, "games_away": 0} for t in all_teams}

        # Clear existing data for fresh backfill
        conn.execute("DELETE FROM pi_ratings WHERE sport = ?", [sport])
        conn.execute("DELETE FROM pi_rating_history WHERE sport = ?", [sport])
        conn.commit()

        history_rows: list[tuple] = []

        # Track the last snapshot date per team to avoid dups from doubleheaders
        last_snap: dict[str, str] = {}

        # Process chronologically
        for _, g in games.iterrows():
            ht = g["home_team"]
            at = g["away_team"]
            h_score = float(g["home_score"])
            a_score = float(g["away_score"])
            gd = str(g["game_date"])[:10]

            # Ensure teams exist (mid-season additions)
            for t in [ht, at]:
                if t not in ratings:
                    ratings[t] = {"home_pi": 0.0, "away_pi": 0.0,
                                  "games_home": 0, "games_away": 0}

            r_home = ratings[ht]
            r_away = ratings[at]

            # Expected margin = home_team.home_pi - away_team.away_pi
            expected_margin = r_home["home_pi"] - r_away["away_pi"]

            # Actual margin (clamped)
            actual_margin = _clamp(h_score - a_score)

            # Error
            error = actual_margin - expected_margin

            # Update: home team's home_pi, away team's away_pi
            r_home["home_pi"] += LEARNING_RATE * error
            r_away["away_pi"] -= LEARNING_RATE * error

            # Increment game counts
            r_home["games_home"] += 1
            r_away["games_away"] += 1

            # Snapshot history (one per team per date — last game wins for doubleheaders)
            for team in [ht, at]:
                r = ratings[team]
                history_rows.append((
                    team, sport, gd,
                    round(r["home_pi"], 4),
                    round(r["away_pi"], 4),
                    r["games_home"],
                    r["games_away"],
                ))
                last_snap[team] = gd

        # Deduplicate: keep last entry per (team, sport, game_date)
        # Since we process in order, we can INSERT OR REPLACE
        conn.executemany(
            "INSERT OR REPLACE INTO pi_rating_history "
            "(team, sport, game_date, home_pi, away_pi, games_home, games_away) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            history_rows,
        )

        # Write current ratings
        for team, r in ratings.items():
            conn.execute(
                "INSERT OR REPLACE INTO pi_ratings "
                "(team, sport, home_pi, away_pi, games_home, games_away, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [team, sport,
                 round(r["home_pi"], 4), round(r["away_pi"], 4),
                 r["games_home"], r["games_away"],
                 last_snap.get(team, season_start)],
            )
        conn.commit()

        # Build summary
        sorted_by_home = sorted(ratings.items(), key=lambda x: -x[1]["home_pi"])
        sorted_by_away = sorted(ratings.items(), key=lambda x: -x[1]["away_pi"])
        sorted_by_combined = sorted(
            ratings.items(),
            key=lambda x: -(x[1]["home_pi"] + x[1]["away_pi"]) / 2,
        )

        summary = {
            "sport": sport,
            "games_processed": len(games),
            "teams": len(all_teams),
            "history_rows": len(history_rows),
            "top5_home_pi": [(t, round(r["home_pi"], 3)) for t, r in sorted_by_home[:5]],
            "bottom5_home_pi": [(t, round(r["home_pi"], 3)) for t, r in sorted_by_home[-5:]],
            "top5_away_pi": [(t, round(r["away_pi"], 3)) for t, r in sorted_by_away[:5]],
            "bottom5_away_pi": [(t, round(r["away_pi"], 3)) for t, r in sorted_by_away[-5:]],
            "top5_combined": [(t, round((r["home_pi"]+r["away_pi"])/2, 3)) for t, r in sorted_by_combined[:5]],
            "bottom5_combined": [(t, round((r["home_pi"]+r["away_pi"])/2, 3)) for t, r in sorted_by_combined[-5:]],
        }

        log.info("Pi ratings backfilled: %d games, %d teams", len(games), len(all_teams))
        log.info("  Top 5 (combined): %s",
                 ", ".join(f"{t}={v:.3f}" for t, v in summary["top5_combined"]))
        log.info("  Bottom 5 (combined): %s",
                 ", ".join(f"{t}={v:.3f}" for t, v in summary["bottom5_combined"]))

        return summary

    finally:
        if close_conn:
            conn.close()


def get_team_pi_for_date(
    team: str,
    game_date: str,
    sport: str = "MLB",
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[dict]:
    """
    Look up the Pi rating snapshot for a team BEFORE game_date (no lookahead).
    Returns {"home_pi": float, "away_pi": float, "games_home": int, "games_away": int}
    or None if no history yet.
    """
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(os.path.abspath(DB_PATH))

    try:
        row = conn.execute(
            "SELECT home_pi, away_pi, games_home, games_away "
            "FROM pi_rating_history "
            "WHERE team = ? AND sport = ? AND game_date < ? "
            "ORDER BY game_date DESC LIMIT 1",
            [team, sport, game_date],
        ).fetchone()
        if row is None:
            return None
        return {
            "home_pi": float(row[0]),
            "away_pi": float(row[1]),
            "games_home": int(row[2]),
            "games_away": int(row[3]),
        }
    finally:
        if close_conn:
            conn.close()


def get_all_pi_for_date(
    game_date: str,
    sport: str = "MLB",
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, dict]:
    """
    Get the most recent Pi rating for ALL teams as of game_date (no lookahead).
    Returns {team: {"home_pi": float, "away_pi": float, ...}} dict.
    """
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(os.path.abspath(DB_PATH))

    try:
        # For each team, get the row with the latest game_date < given date
        rows = conn.execute("""
            SELECT h.team, h.home_pi, h.away_pi, h.games_home, h.games_away
            FROM pi_rating_history h
            INNER JOIN (
                SELECT team, MAX(game_date) AS max_date
                FROM pi_rating_history
                WHERE sport = ? AND game_date < ?
                GROUP BY team
            ) latest ON h.team = latest.team AND h.game_date = latest.max_date
            WHERE h.sport = ?
        """, [sport, game_date, sport]).fetchall()

        return {
            r[0]: {"home_pi": float(r[1]), "away_pi": float(r[2]),
                   "games_home": int(r[3]), "games_away": int(r[4])}
            for r in rows
        }
    finally:
        if close_conn:
            conn.close()


def update_pi_after_settlement(
    game_results: list[dict],
    sport: str = "MLB",
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Incremental Pi update after settling today's games.

    game_results: list of {"home_team": str, "away_team": str,
                           "home_score": int, "away_score": int,
                           "game_date": str}

    Returns number of games processed.
    """
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(os.path.abspath(DB_PATH))

    try:
        _ensure_tables(conn)
        processed = 0

        for g in game_results:
            ht = g["home_team"]
            at = g["away_team"]
            gd = str(g["game_date"])[:10]

            # Get current ratings (or initialize)
            def _get_or_init(team):
                row = conn.execute(
                    "SELECT home_pi, away_pi, games_home, games_away "
                    "FROM pi_ratings WHERE team = ? AND sport = ?",
                    [team, sport],
                ).fetchone()
                if row:
                    return {"home_pi": float(row[0]), "away_pi": float(row[1]),
                            "games_home": int(row[2]), "games_away": int(row[3])}
                return {"home_pi": 0.0, "away_pi": 0.0,
                        "games_home": 0, "games_away": 0}

            r_home = _get_or_init(ht)
            r_away = _get_or_init(at)

            expected = r_home["home_pi"] - r_away["away_pi"]
            actual = _clamp(float(g["home_score"]) - float(g["away_score"]))
            error = actual - expected

            r_home["home_pi"] += LEARNING_RATE * error
            r_away["away_pi"] -= LEARNING_RATE * error
            r_home["games_home"] += 1
            r_away["games_away"] += 1

            # Update current
            for team, r in [(ht, r_home), (at, r_away)]:
                conn.execute(
                    "INSERT OR REPLACE INTO pi_ratings "
                    "(team, sport, home_pi, away_pi, games_home, games_away, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [team, sport,
                     round(r["home_pi"], 4), round(r["away_pi"], 4),
                     r["games_home"], r["games_away"], gd],
                )
                conn.execute(
                    "INSERT OR REPLACE INTO pi_rating_history "
                    "(team, sport, game_date, home_pi, away_pi, games_home, games_away) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [team, sport, gd,
                     round(r["home_pi"], 4), round(r["away_pi"], 4),
                     r["games_home"], r["games_away"]],
                )

            processed += 1

        conn.commit()
        log.info("update_pi_after_settlement: processed %d games for %s", processed, sport)
        return processed

    finally:
        if close_conn:
            conn.close()


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sport = sys.argv[1] if len(sys.argv) > 1 else "MLB"
    start = sys.argv[2] if len(sys.argv) > 2 else "2026-03-27"

    result = backfill_pi_ratings(sport, start)
    print(f"\n{'='*60}")
    print(f"Pi Rating Backfill Complete — {sport}")
    print(f"{'='*60}")
    print(f"Games processed: {result['games_processed']}")
    print(f"Teams:           {result['teams']}")
    print(f"History rows:    {result['history_rows']}")

    print(f"\nTop 5 (combined):")
    for t, v in result["top5_combined"]:
        print(f"  {t:<30} {v:+.3f}")

    print(f"\nBottom 5 (combined):")
    for t, v in result["bottom5_combined"]:
        print(f"  {t:<30} {v:+.3f}")

    print(f"\nTop 5 home_pi:")
    for t, v in result["top5_home_pi"]:
        print(f"  {t:<30} {v:+.3f}")

    print(f"\nBottom 5 away_pi:")
    for t, v in result["bottom5_away_pi"]:
        print(f"  {t:<30} {v:+.3f}")
