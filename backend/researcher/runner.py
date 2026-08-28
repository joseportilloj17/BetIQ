"""
runner.py — Main entry point for the BetIQ Researcher daily cycle.

Fetches → stores → organizes raw brief to disk.
Synthesis (Tier 2) is done by Claude Code (the agent) as a separate task —
reading the raw items file and writing reports/brief_YYYY-MM-DD.md.
No API key needed; the agent IS the LLM.

Called by scheduler.py once daily, or manually via researcher_report.py.
"""
import logging
from datetime import datetime, timezone

from researcher.sources import fetch_all
from researcher.storage import ingest_items, get_items_for_date, save_brief
from researcher.synthesizer import organize_raw
from researcher.file_output import write_brief_files

logger = logging.getLogger(__name__)


def run_daily_research(game_date: str = None) -> dict:
    """Execute the daily research cycle (keyless).

    Steps:
      1. Fetch from all news sources (ESPN, MLB, FanGraphs)
      2. Deduplicate and store in researcher_items table
      3. Organize items by category (pure Python, no LLM)
      4. Write reports/brief_raw_YYYY-MM-DD.md + .json to disk
      5. Save to researcher_briefs table for dashboard

    Returns summary dict with counts and file paths.
    """
    if game_date is None:
        game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("[researcher] Starting daily research for %s", game_date)

    # Step 1: Fetch
    raw_items = fetch_all(date_str=game_date)
    logger.info("[researcher] Fetched %d raw items", len(raw_items))

    if not raw_items:
        logger.warning("[researcher] No items fetched — sources may be down")
        return {"game_date": game_date, "fetched": 0, "ingested": 0, "brief": False}

    # Step 2: Store (dedup)
    new_count = ingest_items(raw_items, game_date)

    # Step 3: Retrieve stored items (includes any from earlier runs today)
    all_items = get_items_for_date(game_date)

    # Step 4: Organize (pure Python — no API key, no LLM)
    organized = organize_raw(all_items, game_date)

    # Step 5: Write to disk
    file_result = write_brief_files(game_date, organized, all_items)

    # Step 6: Save raw brief to DB for dashboard
    raw_md = open(file_result["md_path"]).read()
    save_brief(
        game_date=game_date,
        brief_type="raw",
        content=raw_md,
        item_count=len(all_items),
    )

    # Step 7: Check if a synthesized brief exists on disk (written by agent)
    import os
    _synth_path = os.path.join(os.path.dirname(file_result["md_path"]),
                               f"brief_{game_date}.md")
    if os.path.exists(_synth_path):
        synth_md = open(_synth_path).read()
        save_brief(
            game_date=game_date,
            brief_type="daily",
            content=synth_md,
            item_count=len(all_items),
        )
        logger.info("[researcher] Loaded synthesized brief from %s", _synth_path)

    summary = {
        "game_date": game_date,
        "fetched": len(raw_items),
        "ingested": new_count,
        "total_stored": len(all_items),
        "md_path": file_result.get("md_path"),
        "json_path": file_result.get("json_path"),
    }

    logger.info("[researcher] Daily research complete: %s", summary)
    return summary
