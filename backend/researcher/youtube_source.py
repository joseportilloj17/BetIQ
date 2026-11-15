"""
youtube_source.py — Ingest injury analysis from YouTube channels via transcripts.

Uses youtube-transcript-api (free, no API key) to pull captions from
configured channels. Extracts player/injury mentions and surfaces them
as NewsItems for the researcher brief.

Channels are configurable via ../data/youtube_channels.json:
[
    {
        "channel_id": "UC...",
        "name": "ProFootballDoc",
        "sport": "NFL",
        "category": "injury_analysis"
    }
]

FRAMING:
- These are ANALYST OPINIONS, not official injury reports.
- Tagged as source="youtube:<channel>" and category="injury_analysis"
  to distinguish from official ESPN/team injury reports.
- Value is SPEED and SYNTHESIS of public information.
"""
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_CHANNELS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "youtube_channels.json"
)

# Common NFL injury terms for extraction
_INJURY_TERMS = [
    "acl", "mcl", "pcl", "lcl", "meniscus", "torn", "sprain", "strain",
    "concussion", "hamstring", "quad", "ankle", "knee", "shoulder",
    "elbow", "wrist", "calf", "groin", "hip", "rib", "back",
    "fracture", "broken", "dislocated", "surgery", "ir ", "injured reserve",
    "questionable", "doubtful", "out ", "week-to-week", "day-to-day",
    "limited", "dnp", "did not practice", "full participant",
    "game-time decision", "expected to play", "expected to miss",
    "return", "comeback", "rehab", "recovery", "timetable",
    "mri", "x-ray", "ct scan", "scope",
]

# NFL team names for extraction
_NFL_TEAMS = [
    "cardinals", "falcons", "ravens", "bills", "panthers", "bears",
    "bengals", "browns", "cowboys", "broncos", "lions", "packers",
    "texans", "colts", "jaguars", "chiefs", "chargers", "rams",
    "raiders", "dolphins", "vikings", "patriots", "saints", "giants",
    "jets", "eagles", "steelers", "seahawks", "49ers", "niners",
    "buccaneers", "bucs", "titans", "commanders",
]


def _load_channels() -> list[dict]:
    """Load configured YouTube channels."""
    if not os.path.exists(_CHANNELS_PATH):
        return []
    with open(_CHANNELS_PATH) as f:
        return json.load(f)


def save_channels(channels: list[dict]):
    """Save channel config to disk."""
    os.makedirs(os.path.dirname(_CHANNELS_PATH), exist_ok=True)
    with open(_CHANNELS_PATH, "w") as f:
        json.dump(channels, f, indent=2)


def get_channels() -> list[dict]:
    """Return current channel list."""
    return _load_channels()


def _get_recent_video_ids(channel_id: str, max_results: int = 5) -> list[dict]:
    """Get recent video IDs from a channel via RSS feed (no API key needed)."""
    import requests
    import xml.etree.ElementTree as ET

    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "BetIQ-Researcher/1.0"
        })
        if resp.status_code != 200:
            logger.warning("[youtube] RSS feed failed for %s: %d", channel_id, resp.status_code)
            return []

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

        videos = []
        for entry in root.findall("atom:entry", ns)[:max_results]:
            video_id = entry.find("yt:videoId", ns)
            title = entry.find("atom:title", ns)
            published = entry.find("atom:published", ns)

            if video_id is not None:
                pub_dt = None
                if published is not None and published.text:
                    try:
                        pub_dt = datetime.fromisoformat(published.text.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                videos.append({
                    "video_id": video_id.text,
                    "title": title.text if title is not None else "",
                    "published": pub_dt,
                })

        return videos
    except Exception as exc:
        logger.warning("[youtube] RSS fetch error for %s: %s", channel_id, exc)
        return []


def _fetch_transcript(video_id: str) -> Optional[str]:
    """Fetch video transcript using youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        # Combine all snippets into one text
        full_text = " ".join(snippet.text for snippet in transcript)
        return full_text
    except Exception as exc:
        logger.debug("[youtube] No transcript for %s: %s", video_id, exc)
        return None


def _extract_injury_mentions(text: str, title: str = "") -> list[dict]:
    """Extract player/injury mentions from transcript text.

    Returns list of dicts with:
        player: str (name or context)
        injury: str (injury description)
        sentiment: str (positive/negative/neutral)
        quote: str (relevant excerpt)
    """
    mentions = []
    text_lower = text.lower()
    title_lower = title.lower()

    # Split into sentences (rough)
    sentences = re.split(r'[.!?]+', text)

    for sent in sentences:
        sent_lower = sent.lower().strip()
        if len(sent_lower) < 10:
            continue

        # Check if sentence mentions an injury term
        has_injury = any(term in sent_lower for term in _INJURY_TERMS)
        if not has_injury:
            continue

        # Check if it mentions a team
        teams_mentioned = [t for t in _NFL_TEAMS if t in sent_lower]

        # Extract injury context
        injury_terms_found = [t for t in _INJURY_TERMS if t in sent_lower]

        mentions.append({
            "quote": sent.strip()[:200],
            "injury_terms": injury_terms_found[:3],
            "teams": teams_mentioned,
        })

    return mentions


def fetch_youtube_items(channels: list[dict] = None,
                        max_age_hours: int = 48,
                        max_videos_per_channel: int = 3) -> list[dict]:
    """Fetch injury analysis items from configured YouTube channels.

    Returns list of NewsItem-compatible dicts.
    """
    if channels is None:
        channels = _load_channels()

    if not channels:
        logger.info("[youtube] No channels configured")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    items = []

    for ch in channels:
        channel_id = ch.get("channel_id", "")
        channel_name = ch.get("name", channel_id)
        sport = ch.get("sport", "NFL")

        if not channel_id:
            continue

        logger.info("[youtube] Fetching from %s (%s)", channel_name, channel_id)
        videos = _get_recent_video_ids(channel_id, max_results=max_videos_per_channel)

        for video in videos:
            # Skip old videos
            if video.get("published") and video["published"] < cutoff:
                continue

            vid_id = video["video_id"]
            title = video.get("title", "")

            # Check if title suggests injury content
            title_lower = title.lower()
            has_injury_title = any(
                term in title_lower
                for term in ["injury", "injured", "out", "update", "return",
                             "questionable", "status", "news", "report"]
            )

            # Fetch transcript
            transcript = _fetch_transcript(vid_id)
            if not transcript:
                # Still include title-only if injury-relevant
                if has_injury_title:
                    items.append({
                        "headline": f"[Analyst] {title}",
                        "source": f"youtube:{channel_name}",
                        "url": f"https://youtube.com/watch?v={vid_id}",
                        "timestamp": video.get("published", datetime.now(timezone.utc)),
                        "body": "(transcript unavailable)",
                        "players": [],
                        "teams": [],
                        "category": "injury_analysis",
                        "source_type": "analyst_opinion",
                    })
                continue

            # Extract injury mentions
            mentions = _extract_injury_mentions(transcript, title)

            if not mentions and not has_injury_title:
                continue

            # Build summary body from mentions
            body_parts = []
            all_teams = set()
            for m in mentions[:10]:
                body_parts.append(m["quote"])
                all_teams.update(m.get("teams", []))

            body = " | ".join(body_parts) if body_parts else transcript[:500]

            items.append({
                "headline": f"[Analyst] {title}",
                "source": f"youtube:{channel_name}",
                "url": f"https://youtube.com/watch?v={vid_id}",
                "timestamp": video.get("published", datetime.now(timezone.utc)),
                "body": body[:2000],
                "players": [],  # Could enhance with NER
                "teams": list(all_teams),
                "category": "injury_analysis",
                "source_type": "analyst_opinion",
                "mention_count": len(mentions),
            })

    logger.info("[youtube] Fetched %d injury analysis items from %d channels",
                len(items), len(channels))
    return items
