"""
One-time backfill: populate team and opponent columns in fantasy_forward_preds.

team: from Sleeper raw projections (JOIN on player_name, season, week).
opponent: derived from game_id format ({season}_{week}_{away}_{home}).

consensus_median: skipped — with n_sources ≤ 2, median == mean for all current data.
"""
import logging
import os
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")


def backfill_team():
    """Populate team from Sleeper projections via JOIN."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))

    updated = conn.execute("""
        UPDATE fantasy_forward_preds
        SET team = (
            SELECT fp.team FROM fantasy_projections fp
            WHERE fp.source = 'sleeper'
              AND fp.season = fantasy_forward_preds.season
              AND fp.week = fantasy_forward_preds.week
              AND fp.player_name = fantasy_forward_preds.player_name
              AND fp.team IS NOT NULL AND fp.team != ''
            LIMIT 1
        )
        WHERE team IS NULL
    """).rowcount

    conn.commit()

    # Verify
    total = conn.execute(
        "SELECT COUNT(*) FROM fantasy_forward_preds WHERE season=2026 AND actual_value IS NOT NULL"
    ).fetchone()[0]
    with_team = conn.execute(
        "SELECT COUNT(*) FROM fantasy_forward_preds WHERE season=2026 AND actual_value IS NOT NULL AND team IS NOT NULL"
    ).fetchone()[0]

    logger.info("Team backfill: %d rows updated. Coverage: %d/%d (%.0f%%)",
                updated, with_team, total, with_team / total * 100 if total else 0)
    conn.close()
    return updated


def backfill_opponent():
    """Derive opponent from game_id format: {season}_{week}_{away}_{home}."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))

    # Get all rows with game_id but no opponent
    rows = conn.execute("""
        SELECT id, team, game_id FROM fantasy_forward_preds
        WHERE game_id IS NOT NULL AND opponent IS NULL AND team IS NOT NULL
    """).fetchall()

    updated = 0
    for row_id, team, game_id in rows:
        parts = game_id.split("_")
        if len(parts) < 4:
            continue
        away, home = parts[2], parts[3]
        opp = home if team == away else away
        conn.execute("UPDATE fantasy_forward_preds SET opponent = ? WHERE id = ?",
                     (opp, row_id))
        updated += 1

    conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM fantasy_forward_preds WHERE season=2026 AND actual_value IS NOT NULL"
    ).fetchone()[0]
    with_opp = conn.execute(
        "SELECT COUNT(*) FROM fantasy_forward_preds WHERE season=2026 AND actual_value IS NOT NULL AND opponent IS NOT NULL"
    ).fetchone()[0]
    logger.info("Opponent backfill: %d rows updated. Coverage: %d/%d (%.0f%%)",
                updated, with_opp, total, with_opp / total * 100 if total else 0)
    conn.close()
    return updated


if __name__ == "__main__":
    print("Backfilling team column...")
    backfill_team()
    print("Backfilling opponent column...")
    backfill_opponent()
    print("Done.")
