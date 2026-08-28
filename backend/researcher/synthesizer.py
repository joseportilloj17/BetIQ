"""
synthesizer.py — Synthesis instructions for BetIQ Researcher.

Two-tier design (both keyless):
  Tier 1 (automated): Pure Python organizes raw items into a structured
      brief_raw_YYYY-MM-DD.md — matchups, injuries, facts, news by category.
      No LLM, no API key. This is what the fetch script writes.

  Tier 2 (Claude Code): The agent reads the raw brief and writes a
      synthesized brief_YYYY-MM-DD.md with decision-relevance, cross-item
      connections, and soft-window flagging. The agent IS the LLM — no API call.

This module provides the synthesis INSTRUCTIONS (used by Claude Code when
it reads the raw items and writes the synthesized brief) and the raw
organizer (used by the automated pipeline).
"""
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Synthesis instructions — used by Claude Code (the agent)
# when reading raw items and writing the synthesized brief.
# Embedded here as the single source of truth for prompt quality.
# ─────────────────────────────────────────────────────────────
SYNTHESIS_INSTRUCTIONS = """\
You are a sharp MLB analyst writing a morning brief for a sports bettor.

The user already has the raw headline feed. Your job is to SYNTHESIZE —
connect items, surface implications, and flag soft windows. If you're just
restating headlines, you're adding zero value.

Write a scannable markdown brief with these sections:

## ⚡ Soft Windows & Action Items
Lead with this. 2-4 situations where news creates a pricing lag the market
may not have fully absorbed: returns from IL, late scratches, callup debuts,
status upgrades, lineup changes announced after lines opened. For each,
state: (a) the situation, (b) WHY the market may be slow to adjust,
(c) what to watch. If nothing qualifies, say "No clear soft windows today."

## Key Situations
3-5 compound insights (not restated headlines). CONNECT related items:
- A pitcher's matchup + the opposing lineup context + the park = one insight
- An injury return + the replacement's context + the team state = one insight
Every bullet must end with "→ implication" (what it means, not what happened).
Prioritize star players and high-profile matchups.

## Injury & Roster Watch
Notable status changes with IMPLICATION (who replaces them, how it changes
the team's profile today, whether the market has priced it). Skip injuries
with no game-day relevance.

## Matchups to Watch
2-3 pitching matchups with an angle beyond "good pitcher vs bad team."
State WHY each matchup is interesting, not just WHO is pitching.

Rules:
- DECISION-RELEVANCE: every item ends with what it means, not what happened.
- CROSS-ITEM: connect related facts into compound signals.
- SCANNABLE: bold player/team names. Use → for implications. No filler.
- Do NOT make explicit betting recommendations. Present situations and
  implications — let the user decide.
- Keep it under 500 words. Density over length.
"""

PLAYER_NARRATIVE_INSTRUCTIONS = """\
For each player, write 2-3 sentences covering:
1. THE SITUATION (one sentence — what changed)
2. THE IMPLICATION (what this means for lines, props, or matchups)
3. SOFT WINDOW (optional — if the market likely hasn't priced this yet)

CROSS-ITEM: combine related items into compound signals.
BAD: "Judge is on a rehab assignment and played catch today."
GOOD: "Judge rehab ahead of timeline — if activated this week, his props
likely open soft before the market recalibrates. Watch the opener."
"""


# ─────────────────────────────────────────────────────────────
# Raw organizer — pure Python, no LLM, no API key
# ─────────────────────────────────────────────────────────────
def organize_raw(items: list[dict], game_date: str) -> dict:
    """Organize raw items into structured categories.

    Returns a dict ready for file_output to render as the raw brief.
    No API calls, no LLM — pure Python organization.
    """
    by_category = {}
    for item in items:
        by_category.setdefault(item.get("category", "other"), []).append(item)

    # Extract player mention counts
    player_counts = {}
    for item in items:
        for p in item.get("players", []):
            player_counts[p] = player_counts.get(p, 0) + 1
    top_players = sorted(player_counts.items(), key=lambda x: -x[1])[:15]

    # Extract team mention counts
    team_counts = {}
    for item in items:
        for t in item.get("teams", []):
            team_counts[t] = team_counts.get(t, 0) + 1

    return {
        "game_date": game_date,
        "item_count": len(items),
        "by_category": by_category,
        "top_players": top_players,
        "team_counts": team_counts,
        # No LLM synthesis — these are None (raw fallback handles display)
        "daily_brief": None,
        "player_narratives": None,
        "hard_facts": None,
        "model_feed": None,
    }
