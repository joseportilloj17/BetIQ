"""
sources.py — News ingestion from MLB sources.

Sources (by reliability):
  ESPN API:     injuries endpoint (JSON, free, no key) — PROVEN in injury_fetcher.py
  MLB Stats:    schedule/probablePitcher (JSON, free) — PROVEN in scheduler.py
  ESPN RSS:     headlines feed (XML/RSS) — free, public
  FanGraphs:    blog RSS feed (XML/RSS) — free, public
  Rotowire:     player news (HTML scrape, rate-limited)
  MLB.com:      news feed (HTML/JSON)

Each source returns a list of NewsItem dicts.
"""

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_UA = "BetIQ-Researcher/1.0 (sports analytics research)"
_TIMEOUT = 15


class NewsItem:
    """A single news item from any source."""
    __slots__ = ("headline", "source", "url", "timestamp", "body",
                 "players", "teams", "category", "raw")

    def __init__(self, headline="", source="", url="", timestamp=None,
                 body="", players=None, teams=None, category="news", raw=""):
        self.headline = headline
        self.source = source
        self.url = url
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.body = body
        self.players = players or []
        self.teams = teams or []
        self.category = category  # news | injury | lineup | analysis | roster
        self.raw = raw

    def to_dict(self):
        return {
            "headline": self.headline,
            "source": self.source,
            "url": self.url,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "body": self.body[:2000],
            "players": self.players,
            "teams": self.teams,
            "category": self.category,
        }


# ─────────────────────────────────────────────────────────────
# ESPN Injuries API (JSON — most reliable source)
# ─────────────────────────────────────────────────────────────
def fetch_espn_injuries() -> list[NewsItem]:
    """Fetch current MLB injury report from ESPN API."""
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries"
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("[researcher] ESPN injuries fetch failed: %s", e)
        return []

    items = []
    for team_block in data.get("items", []):
        team_name = team_block.get("team", {}).get("displayName", "")
        team_abbr = team_block.get("team", {}).get("abbreviation", "")
        for injury in team_block.get("injuries", []):
            athlete = injury.get("athlete", {})
            name = athlete.get("displayName", "")
            position = athlete.get("position", {}).get("abbreviation", "")
            status = injury.get("status", "")
            detail = injury.get("longComment", "") or injury.get("shortComment", "")
            injury_date = injury.get("date", "")

            headline = f"{name} ({team_abbr}, {position}) — {status}"
            items.append(NewsItem(
                headline=headline,
                source="espn_injuries",
                url=f"https://www.espn.com/mlb/team/injuries/_/name/{team_abbr.lower()}",
                timestamp=_parse_date(injury_date),
                body=detail,
                players=[name],
                teams=[team_name],
                category="injury",
            ))

    logger.info("[researcher] ESPN injuries: %d items", len(items))
    return items


# ─────────────────────────────────────────────────────────────
# MLB Stats API — Probable Pitchers + Schedule
# ─────────────────────────────────────────────────────────────
def fetch_mlb_probables(date_str: str = None) -> list[NewsItem]:
    """Fetch probable pitchers for a given date from MLB Stats API."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team"}
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("[researcher] MLB probables fetch failed: %s", e)
        return []

    items = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_team = home.get("team", {}).get("name", "")
            away_team = away.get("team", {}).get("name", "")
            home_pitcher = home.get("probablePitcher", {}).get("fullName", "TBD")
            away_pitcher = away.get("probablePitcher", {}).get("fullName", "TBD")
            game_time = game.get("gameDate", "")

            headline = f"{away_team} ({away_pitcher}) @ {home_team} ({home_pitcher})"
            players = [p for p in [home_pitcher, away_pitcher] if p != "TBD"]
            items.append(NewsItem(
                headline=headline,
                source="mlb_api",
                url=f"https://www.mlb.com/gameday/{game.get('gamePk', '')}",
                timestamp=_parse_date(game_time),
                body=f"Game time: {game_time}",
                players=players,
                teams=[home_team, away_team],
                category="lineup",
            ))

    logger.info("[researcher] MLB probables: %d games", len(items))
    return items


# ─────────────────────────────────────────────────────────────
# ESPN Headlines RSS
# ─────────────────────────────────────────────────────────────
def fetch_espn_headlines() -> list[NewsItem]:
    """Fetch ESPN MLB headlines via RSS feed."""
    urls_to_try = [
        "https://www.espn.com/espn/rss/mlb/news",
        "https://www.espn.com/espn/rss/news?sport=mlb",
    ]
    items = []
    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item_el in root.findall(".//item"):
                title = _el_text(item_el, "title")
                link = _el_text(item_el, "link")
                desc = _el_text(item_el, "description")
                pub_date = _el_text(item_el, "pubDate")
                if title:
                    items.append(NewsItem(
                        headline=title,
                        source="espn_rss",
                        url=link,
                        timestamp=_parse_rss_date(pub_date),
                        body=_strip_html(desc),
                        category="news",
                    ))
            if items:
                break  # Got items from first working URL
        except Exception as e:
            logger.debug("[researcher] ESPN RSS %s failed: %s", url, e)
            continue

    logger.info("[researcher] ESPN RSS: %d headlines", len(items))
    return items


# ─────────────────────────────────────────────────────────────
# FanGraphs Blog RSS (analysis/opinion)
# ─────────────────────────────────────────────────────────────
def fetch_fangraphs_rss() -> list[NewsItem]:
    """Fetch FanGraphs blog posts via RSS."""
    url = "https://blogs.fangraphs.com/feed/"
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        logger.warning("[researcher] FanGraphs RSS failed: %s", e)
        return []

    items = []
    for item_el in root.findall(".//item"):
        title = _el_text(item_el, "title")
        link = _el_text(item_el, "link")
        desc = _el_text(item_el, "description")
        pub_date = _el_text(item_el, "pubDate")

        # Filter to MLB-relevant content
        content = (title or "") + " " + (desc or "")
        if not any(kw in content.lower() for kw in ["mlb", "baseball", "pitcher", "hitter",
                                                       "batting", "pitching", "trade", "roster"]):
            continue

        items.append(NewsItem(
            headline=title or "",
            source="fangraphs",
            url=link,
            timestamp=_parse_rss_date(pub_date),
            body=_strip_html(desc)[:1500],
            category="analysis",
        ))

    logger.info("[researcher] FanGraphs: %d items", len(items))
    return items


# ─────────────────────────────────────────────────────────────
# ESPN Transactions / Roster Moves
# ─────────────────────────────────────────────────────────────
def fetch_espn_transactions() -> list[NewsItem]:
    """Fetch recent MLB transactions from ESPN API."""
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/transactions"
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("[researcher] ESPN transactions failed: %s", e)
        return []

    items = []
    for txn in data.get("transactions", []):
        team = txn.get("team", {}).get("displayName", "")
        desc = txn.get("description", "")
        txn_date = txn.get("date", "")

        # Extract player names (heuristic: capitalized words before verbs)
        players = _extract_player_names(desc)

        items.append(NewsItem(
            headline=f"{team}: {desc[:100]}",
            source="espn_transactions",
            url="https://www.espn.com/mlb/transactions",
            timestamp=_parse_date(txn_date),
            body=desc,
            players=players,
            teams=[team] if team else [],
            category="roster",
        ))

    logger.info("[researcher] ESPN transactions: %d items", len(items))
    return items


# ─────────────────────────────────────────────────────────────
# MLB.com News (RSS feed)
# ─────────────────────────────────────────────────────────────
def fetch_mlb_news() -> list[NewsItem]:
    """Fetch latest MLB.com editorial content via RSS."""
    url = "https://www.mlb.com/feeds/news/rss.xml"
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        logger.warning("[researcher] MLB.com RSS failed: %s", e)
        return []

    items = []
    for item_el in root.findall(".//item"):
        title = _el_text(item_el, "title")
        link = _el_text(item_el, "link")
        pub_date = _el_text(item_el, "pubDate")
        creator = _el_text(item_el, "{http://purl.org/dc/elements/1.1/}creator")

        body = f"By {creator}" if creator else ""
        items.append(NewsItem(
            headline=title or "",
            source="mlb_com",
            url=link,
            timestamp=_parse_rss_date(pub_date),
            body=body,
            category="news",
        ))

    logger.info("[researcher] MLB.com RSS: %d items", len(items))
    return items


# ─────────────────────────────────────────────────────────────
# MASTER FETCH — all sources, rate-limited
# ─────────────────────────────────────────────────────────────
def fetch_all(date_str: str = None) -> list[NewsItem]:
    """Fetch from all available sources. Returns combined list sorted by timestamp."""
    all_items = []

    fetchers = [
        ("ESPN Injuries", fetch_espn_injuries),
        ("MLB Probables", lambda: fetch_mlb_probables(date_str)),
        ("ESPN Headlines", fetch_espn_headlines),
        ("ESPN Transactions", fetch_espn_transactions),
        ("MLB.com News", fetch_mlb_news),
        ("FanGraphs", fetch_fangraphs_rss),
    ]

    for name, fetcher in fetchers:
        try:
            items = fetcher()
            all_items.extend(items)
        except Exception as e:
            logger.warning("[researcher] %s failed: %s", name, e)
        time.sleep(0.5)  # Rate limit between sources

    # Sort by timestamp (newest first)
    all_items.sort(key=lambda x: x.timestamp or datetime.min.replace(tzinfo=timezone.utc),
                   reverse=True)

    logger.info("[researcher] Total: %d items from %d sources",
                len(all_items), len(fetchers))
    return all_items


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _el_text(parent, tag):
    el = parent.find(tag)
    return el.text.strip() if el is not None and el.text else ""


def _strip_html(text):
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', text).strip()


def _parse_date(date_str):
    if not date_str:
        return datetime.now(timezone.utc)
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _parse_rss_date(date_str):
    if not date_str:
        return datetime.now(timezone.utc)
    # RFC 822 format: "Mon, 01 Jan 2026 12:00:00 GMT"
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return _parse_date(date_str)


def _extract_player_names(text):
    """Heuristic extraction of player names from transaction text."""
    # Pattern: "Placed RHP John Smith on..."  "Recalled SS Mike Jones from..."
    pattern = r'(?:Placed|Recalled|Activated|Designated|Signed|Traded|Released|Optioned)\s+\w+\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    matches = re.findall(pattern, text)
    return matches[:5]  # Cap at 5 names per transaction
