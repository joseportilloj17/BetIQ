"""
soccer_etl_backfill.py — Phase 1a: Fix existing soccer bet_legs in bets.db.

Updates team, opponent, market_type, subtype for all legs where sport='Soccer'
using the improved Pikkit soccer description parser from etl.py.

Usage:
    cd /Users/joseportillo/Downloads/BetIQ
    python backend/soccer_etl_backfill.py          # dry-run (no writes)
    python backend/soccer_etl_backfill.py --write  # commit changes
"""
from __future__ import annotations

import argparse
import os
import sys
import sqlite3
from collections import Counter

# Make sure backend dir is on path for etl imports
sys.path.insert(0, os.path.dirname(__file__))
from etl import parse_soccer_description

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")


def backfill(write: bool = False) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "SELECT id, description, market_type, subtype, team, opponent "
        "FROM bet_legs WHERE sport='Soccer'"
    )
    rows = cur.fetchall()
    print(f"[backfill] Found {len(rows)} soccer bet_legs")

    updates = []
    skipped_no_v = 0
    subtype_counts: Counter = Counter()
    unresolved: list[str] = []

    for leg_id, desc, cur_market, cur_subtype, cur_team, cur_opponent in rows:
        parsed = parse_soccer_description(desc)

        if not parsed:
            # No ' v ' separator — non-soccer leg in mixed parlay; leave unchanged
            skipped_no_v += 1
            continue

        new_subtype = parsed.get('subtype')
        new_market  = parsed.get('market_type') or cur_market

        # Determine team: pick_team for ML/DC/qualify, player for props, home for totals
        new_team = (
            parsed.get('pick_team') or
            parsed.get('player') or
            parsed.get('home_team')
        )
        new_opponent = parsed.get('away_team')

        subtype_counts[new_subtype or 'none'] += 1

        if new_subtype is None:
            unresolved.append(desc)

        updates.append((new_market, new_subtype, new_team, new_opponent, leg_id))

    print(f"\n[backfill] Update plan: {len(updates)} rows, {skipped_no_v} skipped (no v separator)")
    print("[backfill] Subtype breakdown:")
    for st, cnt in subtype_counts.most_common():
        print(f"  {st:30s} {cnt}")

    if unresolved:
        print(f"\n[backfill] {len(unresolved)} rows with no subtype (kept as-is):")
        for d in unresolved[:10]:
            print(f"  {d}")

    if not write:
        print("\n[backfill] DRY RUN — pass --write to commit changes.")
        con.close()
        return

    cur.executemany(
        "UPDATE bet_legs SET market_type=?, subtype=?, team=?, opponent=? WHERE id=?",
        updates,
    )
    con.commit()
    con.close()
    print(f"\n[backfill] Committed {len(updates)} updates.")

    # Verify
    con2 = sqlite3.connect(DB_PATH)
    c2   = con2.cursor()
    c2.execute("SELECT COUNT(*) FROM bet_legs WHERE sport='Soccer' AND subtype IS NOT NULL")
    with_subtype = c2.fetchone()[0]
    c2.execute("SELECT COUNT(*) FROM bet_legs WHERE sport='Soccer'")
    total = c2.fetchone()[0]
    c2.execute("SELECT DISTINCT market_type, COUNT(*) FROM bet_legs WHERE sport='Soccer' GROUP BY market_type")
    markets = c2.fetchall()
    con2.close()

    print(f"\n[backfill] Verification: {with_subtype}/{total} soccer legs have subtype")
    print("[backfill] Market type breakdown:", markets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Commit changes (default: dry run)")
    args = parser.parse_args()
    backfill(write=args.write)
