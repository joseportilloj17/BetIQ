#!/usr/bin/env python3
"""
researcher_report.py — Keyless standalone CLI for daily MLB brief.

No API key needed. Fetches news from ESPN/MLB/FanGraphs, stores to
SQLite, writes organized raw brief to reports/.

Usage:
    python researcher_report.py                # today's date
    python researcher_report.py 2026-08-15     # specific date

Output:
    reports/brief_raw_YYYY-MM-DD.md   — organized raw items (readable as-is)
    reports/brief_raw_YYYY-MM-DD.json — structured data for programmatic use

For the synthesized brief (with soft-window flagging, cross-item connections,
decision-relevance), ask Claude Code to read the raw brief and write
reports/brief_YYYY-MM-DD.md.
"""
import logging
import os
import sys
from datetime import datetime, timezone

from researcher.sources import fetch_all
from researcher.storage import ingest_items, get_items_for_date
from researcher.synthesizer import organize_raw
from researcher.file_output import write_brief_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) > 1:
        game_date = sys.argv[1]
    else:
        game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"BetIQ Researcher — {game_date}")
    print(f"  Mode: keyless (raw organized brief)")
    print()

    # Step 1: Fetch
    print("Fetching from all sources...")
    raw_items = fetch_all(date_str=game_date)
    print(f"  Fetched {len(raw_items)} items")

    if not raw_items:
        print("  No items fetched — sources may be down. Exiting.")
        sys.exit(1)

    # Step 2: Store (dedup)
    from database import init_db
    init_db()
    new_count = ingest_items(raw_items, game_date)
    print(f"  Stored {new_count} new items (deduped)")

    # Step 3: Retrieve all stored items for this date
    all_items = get_items_for_date(game_date)
    print(f"  Total items for {game_date}: {len(all_items)}")

    # Source breakdown
    sources = {}
    for item in all_items:
        sources[item["source"]] = sources.get(item["source"], 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    # Step 4: Organize (pure Python, no LLM)
    organized = organize_raw(all_items, game_date)

    # Step 5: Write files
    print("\nWriting reports...")
    result = write_brief_files(game_date, organized, all_items)
    print(f"  {result['md_path']}")
    print(f"  {result['json_path']}")

    md_size = os.path.getsize(result["md_path"])
    json_size = os.path.getsize(result["json_path"])
    print(f"\n  .md:   {md_size:,} bytes")
    print(f"  .json: {json_size:,} bytes")
    print(f"\nDone. Raw brief: reports/brief_raw_{game_date}.md")
    print(f"For synthesized brief, ask Claude Code to read the raw and write brief_{game_date}.md")


if __name__ == "__main__":
    main()
