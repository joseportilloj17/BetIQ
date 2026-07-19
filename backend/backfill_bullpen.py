"""
backfill_bullpen.py — Backfill bullpen state features (team pitching workload).

Uses MLB Stats API team pitching game logs (30 teams × N seasons = ~150 API calls).
Computes per-game bullpen fatigue proxy and stores differential features.

Features:
  - bp_pitches_3d: team's total pitches thrown in prior 3 games
  - bp_ip_relief_3d: estimated bullpen IP in prior 3 games (total_ip - min(sp_ip, 6))
  - bp_load_diff_3d: home_bp_pitches_3d - away_bp_pitches_3d (DIFFERENTIAL)

Point-in-time: uses only prior games (strict lookback, no lookahead).

Usage:
    .venv/bin/python backend/backfill_bullpen.py
"""
import os
import sys
import time
import sqlite3
import requests
import logging
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "historical.db")
_MLB_STATS_URL = "https://statsapi.mlb.com/api/v1"


def _get_team_ids(season: int) -> dict:
    """Get {team_name: team_id} for a season."""
    r = requests.get(f"{_MLB_STATS_URL}/teams", params={"sportId": 1, "season": season})
    teams = r.json().get("teams", [])
    return {t["name"]: t["id"] for t in teams}


def _fetch_team_pitching_logs(team_id: int, season: int) -> list[dict]:
    """Fetch per-game team pitching stats for a season."""
    try:
        r = requests.get(f"{_MLB_STATS_URL}/teams/{team_id}/stats", params={
            "stats": "gameLog",
            "group": "pitching",
            "season": season,
            "gameType": "R",
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("  API fail team=%d season=%d: %s", team_id, season, exc)
        return []

    splits = data.get("stats", [{}])[0].get("splits", [])
    results = []
    for s in splits:
        stat = s.get("stat", {})
        game_info = s.get("game", {})
        results.append({
            "game_date": s.get("date"),
            "game_pk": game_info.get("gamePk"),
            "is_home": s.get("isHome", False),
            "team_name": s.get("team", {}).get("name", ""),
            "opponent": s.get("opponent", {}).get("name", ""),
            "innings_pitched": float(stat.get("inningsPitched", 0) or 0),
            "pitches": int(stat.get("numberOfPitches", 0) or 0),
            "batters_faced": int(stat.get("battersFaced", 0) or 0),
            "strikeouts": int(stat.get("strikeOuts", 0) or 0),
            "walks": int(stat.get("baseOnBalls", 0) or 0),
            "earned_runs": int(stat.get("earnedRuns", 0) or 0),
        })
    return results


def backfill():
    conn = sqlite3.connect(DB_PATH)

    # Create team pitching game log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_pitching_games (
            game_date TEXT NOT NULL,
            team_name TEXT NOT NULL,
            game_pk INTEGER,
            is_home INTEGER,
            innings_pitched REAL,
            pitches INTEGER,
            batters_faced INTEGER,
            strikeouts INTEGER,
            walks INTEGER,
            earned_runs INTEGER,
            UNIQUE(game_date, team_name)
        )
    """)
    conn.commit()

    # Check existing
    existing = set(conn.execute(
        "SELECT game_date, team_name FROM team_pitching_games"
    ).fetchall())
    log.info("Existing team pitching rows: %d", len(existing))

    # Determine seasons to fetch
    seasons_needed = conn.execute("""
        SELECT DISTINCT CAST(SUBSTR(game_date, 1, 4) AS INTEGER) as season
        FROM games WHERE sport = 'MLB' AND status = 'Final'
        ORDER BY season
    """).fetchall()
    seasons = [s[0] for s in seasons_needed]
    log.info("Seasons to process: %s", seasons)

    total_written = 0
    api_calls = 0

    for season in seasons:
        team_map = _get_team_ids(season)
        log.info("Season %d: %d teams", season, len(team_map))

        for team_name, team_id in sorted(team_map.items()):
            logs = _fetch_team_pitching_logs(team_id, season)
            api_calls += 1

            new_rows = []
            for g in logs:
                key = (g["game_date"], g["team_name"])
                if key in existing:
                    continue
                new_rows.append((
                    g["game_date"], g["team_name"], g["game_pk"],
                    1 if g["is_home"] else 0,
                    g["innings_pitched"], g["pitches"], g["batters_faced"],
                    g["strikeouts"], g["walks"], g["earned_runs"],
                ))
                existing.add(key)

            if new_rows:
                conn.executemany("""
                    INSERT OR IGNORE INTO team_pitching_games
                        (game_date, team_name, game_pk, is_home,
                         innings_pitched, pitches, batters_faced,
                         strikeouts, walks, earned_runs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, new_rows)
                conn.commit()
                total_written += len(new_rows)

            if api_calls % 20 == 0:
                log.info("  API calls: %d, rows written: %d (current: %s %d)",
                         api_calls, total_written, team_name, season)
            time.sleep(0.3)

    total_rows = conn.execute("SELECT COUNT(*) FROM team_pitching_games").fetchone()[0]
    log.info("Team pitching backfill: %d written, %d total, %d API calls",
             total_written, total_rows, api_calls)

    # ── Compute bullpen fatigue features ─────────────────────────────────────
    log.info("Computing bullpen fatigue features...")
    _compute_bp_features(conn)

    conn.close()


def _compute_bp_features(conn):
    """
    Compute per-game bullpen workload differential.

    For each game in `games`, compute:
      - home_bp_pitches_3d: home team's total pitches in prior 3 games
      - away_bp_pitches_3d: away team's total pitches in prior 3 games
      - bp_load_diff_3d: home - away (positive = home more fatigued)

    Strictly PIT: only uses games BEFORE the current one.
    """
    # Add columns
    cols = [c[1] for c in conn.execute("PRAGMA table_info(games)").fetchall()]
    for col in ["home_bp_pitches_3d", "away_bp_pitches_3d", "bp_load_diff_3d"]:
        if col not in cols:
            conn.execute(f"ALTER TABLE games ADD COLUMN {col} REAL")
    conn.commit()

    # Load all team pitching data, sorted by date per team
    team_logs = conn.execute("""
        SELECT team_name, game_date, pitches, innings_pitched
        FROM team_pitching_games
        ORDER BY team_name, game_date
    """).fetchall()

    # Build per-team chronological list
    team_history = defaultdict(list)  # team_name → [(game_date, pitches, ip), ...]
    for team_name, game_date, pitches, ip in team_logs:
        team_history[team_name].append((game_date, pitches or 0, ip or 0.0))

    # For each game, look up prior 3 games for home and away team
    games = conn.execute("""
        SELECT game_date, home_team, away_team
        FROM games
        WHERE sport = 'MLB' AND status = 'Final'
        ORDER BY game_date
    """).fetchall()

    log.info("  Processing %d games for BP features", len(games))

    # Build index: team → sorted list with binary search capability
    # Pre-compute: for each (team, game_date), what's the sum of pitches in prior 3 games?
    team_date_idx = {}
    for team_name, history in team_history.items():
        for i, (gdate, pitches, ip) in enumerate(history):
            # Prior 3 games = history[max(0, i-3):i]
            prior_3 = history[max(0, i - 3):i]
            total_pitches = sum(p for _, p, _ in prior_3)
            team_date_idx[(team_name, gdate)] = total_pitches

    updates = []
    filled = 0
    for game_date, home_team, away_team in games:
        home_pitches = team_date_idx.get((home_team, game_date))
        away_pitches = team_date_idx.get((away_team, game_date))

        if home_pitches is not None and away_pitches is not None:
            diff = home_pitches - away_pitches
            updates.append((home_pitches, away_pitches, diff, game_date, home_team))
            filled += 1
        else:
            updates.append((home_pitches, away_pitches, None, game_date, home_team))

    conn.executemany("""
        UPDATE games
        SET home_bp_pitches_3d = ?,
            away_bp_pitches_3d = ?,
            bp_load_diff_3d = ?
        WHERE game_date = ? AND home_team = ? AND sport = 'MLB'
    """, updates)
    conn.commit()

    log.info("  BP features computed: %d/%d games have differential (%.1f%%)",
             filled, len(games), 100 * filled / len(games) if games else 0)

    # Show distribution
    stats = conn.execute("""
        SELECT AVG(bp_load_diff_3d), MIN(bp_load_diff_3d), MAX(bp_load_diff_3d),
               AVG(ABS(bp_load_diff_3d))
        FROM games
        WHERE bp_load_diff_3d IS NOT NULL AND sport = 'MLB'
    """).fetchone()
    if stats[0] is not None:
        log.info("  bp_load_diff_3d: mean=%.1f, range=[%.0f, %.0f], MAD=%.1f",
                 stats[0], stats[1], stats[2], stats[3])


if __name__ == "__main__":
    backfill()
