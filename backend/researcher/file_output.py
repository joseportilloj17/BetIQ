"""
file_output.py — Write researcher briefs to disk as .md + .json files.

Two tiers (both keyless):
  brief_raw_YYYY-MM-DD.md  — automated, organized raw items (no LLM)
  brief_YYYY-MM-DD.md      — synthesized by Claude Code (the agent)

This module writes the raw brief. The synthesized brief is written
directly by Claude Code when it reads the raw items.
"""
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")


def _ensure_reports_dir():
    os.makedirs(_REPORTS_DIR, exist_ok=True)


def get_reports_dir() -> str:
    """Return the reports directory path (creating it if needed)."""
    _ensure_reports_dir()
    return _REPORTS_DIR


def write_brief_files(game_date: str, organized: dict, items: list[dict]) -> dict:
    """Write brief_raw_YYYY-MM-DD.md and brief_raw_YYYY-MM-DD.json to reports/.

    Args:
        game_date: YYYY-MM-DD string
        organized: output from organize_raw() — category-sorted items
        items: list of raw news item dicts

    Returns:
        dict with md_path and json_path.
    """
    _ensure_reports_dir()

    md_path = os.path.join(_REPORTS_DIR, f"brief_raw_{game_date}.md")
    json_path = os.path.join(_REPORTS_DIR, f"brief_raw_{game_date}.json")

    # ── Build the markdown file ──────────────────────────────────────────────
    md = _build_markdown(game_date, items)
    with open(md_path, "w") as f:
        f.write(md)
    logger.info("[researcher] Wrote %s (%d bytes)", md_path, len(md))

    # ── Build the JSON file ──────────────────────────────────────────────────
    json_data = _build_json(game_date, items)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info("[researcher] Wrote %s", json_path)

    return {"md_path": md_path, "json_path": json_path}


def _build_markdown(game_date: str, items: list[dict]) -> str:
    """Build a well-organized raw brief — readable without any LLM synthesis."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    by_cat = {}
    for item in items:
        by_cat.setdefault(item.get("category", "other"), []).append(item)

    parts = []

    # ── Header ───────────────────────────────────────────────────────────────
    parts.append(f"# BetIQ MLB Raw Brief — {game_date}")
    parts.append(f"*Generated {now} · {len(items)} items ingested*")
    parts.append(f"*Synthesized brief: see brief_{game_date}.md (written by Claude Code)*\n")
    parts.append("---\n")

    # ── Today's Slate ────────────────────────────────────────────────────────
    lineup_items = by_cat.get("lineup", [])
    if lineup_items:
        parts.append(f"## Today's Slate ({len(lineup_items)} games)\n")
        for item in lineup_items:
            parts.append(f"- {item['headline']}")
        parts.append("")

    # ── Injuries ─────────────────────────────────────────────────────────────
    injury_items = by_cat.get("injury", [])
    if injury_items:
        parts.append(f"## Injury Updates ({len(injury_items)})\n")
        for item in injury_items:
            body_snip = ""
            if item.get("body"):
                body_snip = f" — {item['body'][:120]}"
            parts.append(f"- **{item['headline']}**{body_snip}")
        parts.append("")

    # ── Roster Moves ─────────────────────────────────────────────────────────
    roster_items = by_cat.get("roster", [])
    if roster_items:
        parts.append(f"## Roster Moves ({len(roster_items)})\n")
        for item in roster_items:
            parts.append(f"- **{item['headline']}**")
            if item.get("body"):
                parts.append(f"  - {item['body'][:150]}")
        parts.append("")

    # ── News & Analysis ──────────────────────────────────────────────────────
    analysis_items = by_cat.get("analysis", [])
    news_items = by_cat.get("news", [])
    combined = analysis_items + news_items
    if combined:
        parts.append(f"## News & Analysis ({len(combined)})\n")
        for item in combined[:20]:
            src = item.get("source", "?")
            url = item.get("url", "")
            headline = item.get("headline", "")
            link = f"[{headline}]({url})" if url else headline
            parts.append(f"- [{src}] {link}")
        if len(combined) > 20:
            parts.append(f"- *... and {len(combined) - 20} more*")
        parts.append("")

    # ── Player Mentions ──────────────────────────────────────────────────────
    player_counts = {}
    for item in items:
        for p in item.get("players", []):
            player_counts[p] = player_counts.get(p, 0) + 1
    if player_counts:
        sorted_players = sorted(player_counts.items(), key=lambda x: -x[1])
        parts.append("## Players in Today's News\n")
        for name, count in sorted_players[:20]:
            # Gather headlines for this player
            player_headlines = [
                i["headline"] for i in items
                if name in i.get("players", [])
            ][:3]
            parts.append(f"### {name} ({count} mention{'s' if count > 1 else ''})\n")
            for h in player_headlines:
                parts.append(f"- {h}")
            parts.append("")

    # ── Full Item Log (collapsible) ──────────────────────────────────────────
    parts.append("---\n")
    parts.append(f"<details>\n<summary>Full item log ({len(items)} items)</summary>\n")
    cat_order = ["injury", "lineup", "roster", "analysis", "news"]
    for cat in cat_order:
        cat_items = by_cat.get(cat, [])
        if not cat_items:
            continue
        parts.append(f"\n### {cat.upper()} ({len(cat_items)})\n")
        for item in cat_items:
            ts = item.get("timestamp", "")
            if ts:
                ts = ts[:16].replace("T", " ")
            src = item.get("source", "?")
            headline = item.get("headline", "")
            url = item.get("url", "")
            link = f"[{headline}]({url})" if url else headline
            parts.append(f"- `{ts}` [{src}] {link}")

    parts.append("\n</details>")

    return "\n".join(parts)


def _build_json(game_date: str, items: list[dict]) -> dict:
    """Build the structured JSON output for programmatic use."""
    return {
        "game_date": game_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "item_count": len(items),
        "items_by_category": {
            cat: [
                {
                    "headline": i["headline"],
                    "source": i["source"],
                    "url": i.get("url", ""),
                    "timestamp": i.get("timestamp", ""),
                    "body": i.get("body", ""),
                    "players": i.get("players", []),
                    "teams": i.get("teams", []),
                }
                for i in items
                if i.get("category") == cat
            ]
            for cat in sorted(set(i.get("category", "other") for i in items))
        },
        "entity_tags": {
            "players": sorted(set(
                p for i in items for p in i.get("players", [])
            )),
            "teams": sorted(set(
                t for i in items for t in i.get("teams", [])
            )),
        },
    }
