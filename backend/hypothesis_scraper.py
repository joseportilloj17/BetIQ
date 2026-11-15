"""
hypothesis_scraper.py — Scrape methodology & feature ideas from betting analytics sources.

NOT picks. NOT systems. NOT tout records.
Only METHODS and FEATURES — things we can test in our gauntlet.

Sources:
  - Reddit (r/algobetting, r/sportsbook) — method discussion threads
  - Academic papers (Semantic Scholar API — free, no key)
  - GitHub (open-source betting models — actual feature code)
  - Blogs/RSS (FanGraphs methodology, analytics substacks)

Process: scrape → extract hypothesis → store in inbox → test in gauntlet.
A scraped idea gets NO special trust — it's a hypothesis like any we generate.

Usage:
  python hypothesis_scraper.py                   # full scrape, all sources
  python hypothesis_scraper.py --source reddit   # single source
  python hypothesis_scraper.py --list            # show inbox
  python hypothesis_scraper.py --list --status new  # filter by status
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
INBOX_PATH = os.path.join(_DIR, "..", "data", "hypothesis_inbox.json")
_UA = "BetIQ-HypothesisScraper/1.0 (sports analytics research)"
_TIMEOUT = 20

# ── Keyword filters ──────────────────────────────────────────
# Posts matching METHOD_KEYWORDS and NOT matching PICK_KEYWORDS are candidates
METHOD_KEYWORDS = {
    "model", "feature", "methodology", "algorithm", "regression", "machine learning",
    "gradient boost", "random forest", "logistic", "poisson", "negative binomial",
    "expected value", "ev", "kelly", "bankroll", "clv", "closing line",
    "line movement", "sharp", "steam", "reverse line", "rlm",
    "park factor", "dips", "fip", "xfip", "siera", "aging curve",
    "statcast", "barrel", "exit velocity", "launch angle", "xba", "xslg",
    "expected stats", "woba", "wrc+", "elo", "power rating",
    "bayesian", "prior", "shrinkage", "regulariz", "calibrat",
    "feature engineering", "feature importance", "feature selection",
    "cross validation", "walk forward", "backtest", "out of sample",
    "devig", "vig removal", "no-vig", "true probability",
    "market efficiency", "weak form", "semi-strong", "information",
    "teaser", "alt spread", "alt line", "key number", "wong",
    "correlation", "covariance", "same-game", "sgp",
    "platoon", "handedness", "splits", "lefty", "righty",
    "weather", "wind", "temperature", "humidity", "altitude",
    "umpire", "strike zone", "bullpen", "leverage",
    "pace", "rest days", "back to back", "schedule",
    "totals", "over under", "run line", "puck line",
    "prop", "player prop", "strikeout", "home run", "hits",
    "odds screen", "pinnacle", "sharp book", "soft book",
    "stale line", "opening line", "steam move",
    "neural net", "lstm", "transformer", "xgboost", "lightgbm",
    "monte carlo", "simulation", "bootstrap",
}

PICK_KEYWORDS = {
    "lock", "potd", "pick of the day", "parlay of the day",
    "free pick", "guaranteed", "units won", "record:",
    "tail", "fade", "my pick", "going with",
    "best bet", "max bet", "slam", "hammer",
    "subscribe", "premium", "vip", "paid picks",
    "tout", "handicapper", "tipster",
}

# Sports keyword maps for auto-tagging
SPORT_KEYWORDS = {
    "mlb": {"mlb", "baseball", "pitcher", "batter", "innings", "runs", "era",
            "fip", "statcast", "sabermetric", "run line", "fangraphs"},
    "nfl": {"nfl", "football", "quarterback", "touchdown", "spread", "teaser",
            "key number", "wong", "yards"},
    "nba": {"nba", "basketball", "points", "rebounds", "assists", "pace"},
    "nhl": {"nhl", "hockey", "puck line", "goalie", "save percentage"},
}


def _content_hash(text: str) -> str:
    """Short hash for dedup."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def _is_method_post(text: str) -> bool:
    """Return True if text discusses methods/features, not picks."""
    lower = text.lower()
    # Must contain at least one method keyword
    has_method = any(kw in lower for kw in METHOD_KEYWORDS)
    # Must NOT be primarily a pick/tout post
    pick_count = sum(1 for kw in PICK_KEYWORDS if kw in lower)
    return has_method and pick_count < 2


def _detect_sport(text: str) -> str:
    """Auto-detect sport from text content."""
    lower = text.lower()
    scores = {}
    for sport, keywords in SPORT_KEYWORDS.items():
        scores[sport] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def _extract_tags(text: str) -> list[str]:
    """Extract relevant methodology tags from text."""
    lower = text.lower()
    found = []
    tag_groups = {
        "CLV": ["clv", "closing line"],
        "feature_engineering": ["feature engineering", "feature importance", "feature selection"],
        "expected_stats": ["xba", "xslg", "xwoba", "expected stats", "statcast"],
        "park_factors": ["park factor", "park adjust"],
        "DIPS": ["dips", "fip", "xfip", "siera"],
        "aging_curve": ["aging curve", "age adjust"],
        "Bayesian": ["bayesian", "prior", "shrinkage"],
        "Kelly": ["kelly", "bankroll", "bet sizing"],
        "line_movement": ["line movement", "steam", "rlm", "reverse line"],
        "key_numbers": ["key number", "teaser", "wong"],
        "weather": ["weather", "wind", "temperature"],
        "platoon": ["platoon", "handedness", "splits"],
        "ML_model": ["xgboost", "lightgbm", "random forest", "neural net", "gradient boost"],
        "calibration": ["calibrat", "brier", "log loss"],
        "devig": ["devig", "vig removal", "no-vig", "true prob"],
        "market_efficiency": ["market efficien", "weak form", "semi-strong"],
        "simulation": ["monte carlo", "simulation", "bootstrap"],
        "props": ["player prop", "prop", "strikeout", "home run prop"],
        "alt_lines": ["alt spread", "alt line", "alternate"],
        "sharp_books": ["pinnacle", "sharp book", "soft book"],
    }
    for tag, keywords in tag_groups.items():
        if any(kw in lower for kw in keywords):
            found.append(tag)
    return found


# ══════════════════════════════════════════════════════════════
# SOURCE SCRAPERS
# ══════════════════════════════════════════════════════════════

def scrape_reddit(subreddits: list[str] = None, limit: int = 50) -> list[dict]:
    """Scrape method-discussion posts from Reddit using public JSON API."""
    if subreddits is None:
        subreddits = ["algobetting", "sportsbook"]

    hypotheses = []
    search_terms = [
        "model features",
        "feature engineering",
        "methodology",
        "closing line value",
        "expected value model",
        "sharp methods",
        "backtest",
        "machine learning betting",
        "park factors",
        "Bayesian sports",
        "Kelly criterion",
        "statcast betting",
        "line movement analysis",
    ]

    for sub in subreddits:
        for term in search_terms:
            url = f"https://old.reddit.com/r/{sub}/search.json"
            params = {
                "q": term,
                "restrict_sr": "1",
                "sort": "relevance",
                "t": "year",
                "limit": min(limit, 25),
            }
            try:
                r = requests.get(url, params=params, timeout=_TIMEOUT,
                                 headers={"User-Agent": _UA})
                if r.status_code == 429:
                    logger.warning("[hyp] Reddit rate-limited on r/%s, sleeping", sub)
                    time.sleep(10)
                    continue
                if r.status_code != 200:
                    logger.debug("[hyp] Reddit r/%s search=%s status=%d", sub, term, r.status_code)
                    continue
                data = r.json()
            except Exception as e:
                logger.debug("[hyp] Reddit r/%s search=%s error: %s", sub, term, e)
                continue

            for post in data.get("data", {}).get("children", []):
                pdata = post.get("data", {})
                title = pdata.get("title", "")
                selftext = pdata.get("selftext", "")[:2000]
                permalink = pdata.get("permalink", "")
                score = pdata.get("score", 0)
                num_comments = pdata.get("num_comments", 0)

                # Quality filter: skip low-engagement posts
                if score < 3 and num_comments < 3:
                    continue

                combined = f"{title} {selftext}"
                if not _is_method_post(combined):
                    continue

                h = {
                    "source_type": "reddit",
                    "source_url": f"https://reddit.com{permalink}" if permalink else "",
                    "source_title": title,
                    "hypothesis": "",  # filled by extraction
                    "detail": selftext[:500] if selftext else "",
                    "category": "methodology",
                    "sport": _detect_sport(combined),
                    "tags": _extract_tags(combined),
                    "score": score,
                    "comments": num_comments,
                    "content_hash": _content_hash(combined),
                }
                hypotheses.append(h)

            time.sleep(1.5)  # Rate limit

    # Dedup by content hash
    seen = set()
    deduped = []
    for h in hypotheses:
        if h["content_hash"] not in seen:
            seen.add(h["content_hash"])
            deduped.append(h)

    logger.info("[hyp] Reddit: %d candidates (from %d raw)", len(deduped), len(hypotheses))
    return deduped


def scrape_semantic_scholar(limit: int = 30) -> list[dict]:
    """Search Semantic Scholar for sports betting methodology papers."""
    queries = [
        "sports betting market efficiency closing line",
        "MLB prediction model features sabermetrics",
        "NFL point spread prediction methodology",
        "sports wagering market efficiency",
        "baseball expected statistics predictive model",
        "betting market microstructure line movement",
        "Poisson model sports scoring prediction",
        "Kelly criterion sports betting optimal",
    ]

    hypotheses = []
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    for query in queries:
        params = {
            "query": query,
            "limit": min(limit, 10),
            "fields": "title,abstract,url,year,citationCount,externalIds",
            "year": "2018-2026",
        }
        try:
            r = requests.get(base_url, params=params, timeout=_TIMEOUT,
                             headers={"User-Agent": _UA})
            if r.status_code == 429:
                logger.warning("[hyp] Semantic Scholar rate-limited, sleeping")
                time.sleep(30)
                continue
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception as e:
            logger.debug("[hyp] Semantic Scholar query=%s error: %s", query, e)
            continue

        for paper in data.get("data", []):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "") or ""
            url = paper.get("url", "")
            year = paper.get("year")
            citations = paper.get("citationCount", 0)

            combined = f"{title} {abstract}"
            if not _is_method_post(combined):
                continue

            # Skip very low-impact papers
            if citations is not None and citations < 2 and year and year < 2024:
                continue

            h = {
                "source_type": "academic",
                "source_url": url,
                "source_title": title,
                "hypothesis": "",
                "detail": abstract[:500] if abstract else "",
                "category": "methodology",
                "sport": _detect_sport(combined),
                "tags": _extract_tags(combined),
                "year": year,
                "citations": citations,
                "content_hash": _content_hash(combined),
            }
            hypotheses.append(h)

        time.sleep(1.0)

    seen = set()
    deduped = []
    for h in hypotheses:
        if h["content_hash"] not in seen:
            seen.add(h["content_hash"])
            deduped.append(h)

    logger.info("[hyp] Semantic Scholar: %d candidates", len(deduped))
    return deduped


def scrape_github(limit: int = 20) -> list[dict]:
    """Search GitHub for open-source betting model repos with feature code."""
    queries = [
        "sports betting model python",
        "MLB prediction features",
        "NFL spread prediction model",
        "sports analytics betting features",
        "baseball sabermetrics prediction",
        "closing line value tracker",
        "sports betting expected value",
    ]

    hypotheses = []
    base_url = "https://api.github.com/search/repositories"

    for query in queries:
        params = {
            "q": f"{query} language:python",
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, 10),
        }
        try:
            r = requests.get(base_url, params=params, timeout=_TIMEOUT,
                             headers={"User-Agent": _UA, "Accept": "application/vnd.github.v3+json"})
            if r.status_code == 403:
                logger.warning("[hyp] GitHub rate-limited, sleeping")
                time.sleep(60)
                continue
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception as e:
            logger.debug("[hyp] GitHub query=%s error: %s", query, e)
            continue

        for repo in data.get("items", []):
            name = repo.get("full_name", "")
            desc = repo.get("description", "") or ""
            url = repo.get("html_url", "")
            stars = repo.get("stargazers_count", 0)
            updated = repo.get("updated_at", "")
            topics = repo.get("topics", []) or []

            # Skip low-quality repos
            if stars < 5:
                continue

            combined = f"{name} {desc} {' '.join(topics)}"
            if not _is_method_post(combined):
                continue

            h = {
                "source_type": "github",
                "source_url": url,
                "source_title": f"{name} — {desc[:100]}",
                "hypothesis": "",
                "detail": desc,
                "category": "methodology",
                "sport": _detect_sport(combined),
                "tags": _extract_tags(combined) + [t for t in topics[:5]],
                "stars": stars,
                "content_hash": _content_hash(combined),
            }
            hypotheses.append(h)

        time.sleep(1.0)

    seen = set()
    deduped = []
    for h in hypotheses:
        if h["content_hash"] not in seen:
            seen.add(h["content_hash"])
            deduped.append(h)

    logger.info("[hyp] GitHub: %d candidates", len(deduped))
    return deduped


def scrape_blogs_rss() -> list[dict]:
    """Scrape analytics blogs/RSS for methodology posts."""
    feeds = [
        ("https://blogs.fangraphs.com/feed/", "fangraphs"),
        ("https://www.baseballprospectus.com/feed/", "baseball_prospectus"),
        ("https://community.fangraphs.com/feed/", "fangraphs_community"),
    ]

    hypotheses = []
    import xml.etree.ElementTree as ET

    for feed_url, source_name in feeds:
        try:
            r = requests.get(feed_url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
        except Exception as e:
            logger.debug("[hyp] RSS %s error: %s", source_name, e)
            continue

        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
            desc = re.sub(r'<[^>]+>', '', desc)  # strip HTML

            combined = f"{title} {desc}"
            if not _is_method_post(combined):
                continue

            h = {
                "source_type": "blog",
                "source_url": link,
                "source_title": title,
                "hypothesis": "",
                "detail": desc[:500],
                "category": "feature",
                "sport": _detect_sport(combined),
                "tags": _extract_tags(combined),
                "content_hash": _content_hash(combined),
            }
            hypotheses.append(h)

        time.sleep(0.5)

    seen = set()
    deduped = []
    for h in hypotheses:
        if h["content_hash"] not in seen:
            seen.add(h["content_hash"])
            deduped.append(h)

    logger.info("[hyp] Blogs/RSS: %d candidates", len(deduped))
    return deduped


# ══════════════════════════════════════════════════════════════
# INBOX MANAGEMENT
# ══════════════════════════════════════════════════════════════

def _load_inbox() -> list[dict]:
    """Load the hypothesis inbox from JSON file."""
    if not os.path.exists(INBOX_PATH):
        return []
    try:
        with open(INBOX_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_inbox(items: list[dict]) -> None:
    """Save the hypothesis inbox to JSON file."""
    os.makedirs(os.path.dirname(INBOX_PATH), exist_ok=True)
    with open(INBOX_PATH, "w") as f:
        json.dump(items, f, indent=2, default=str)


def _next_id(inbox: list[dict]) -> str:
    """Generate next hypothesis ID."""
    existing = [int(h["id"].split("_")[1]) for h in inbox
                if h.get("id", "").startswith("h_")]
    next_num = max(existing, default=0) + 1
    return f"h_{next_num:04d}"


def add_hypotheses(new_items: list[dict]) -> int:
    """Add scraped hypotheses to inbox, deduplicating on content_hash.

    Returns count of new items added.
    """
    inbox = _load_inbox()
    existing_hashes = {h.get("content_hash") for h in inbox}
    added = 0

    for item in new_items:
        ch = item.get("content_hash", "")
        if ch in existing_hashes:
            continue

        item["id"] = _next_id(inbox)
        item["status"] = "new"
        item["scraped_at"] = datetime.now(timezone.utc).isoformat()
        item["test_result"] = None
        item["tested_at"] = None
        item["notes"] = None
        item.setdefault("priority", 0)

        inbox.append(item)
        existing_hashes.add(ch)
        added += 1

    if added:
        _save_inbox(inbox)

    return added


def add_manual_hypothesis(
    hypothesis: str,
    detail: str = "",
    category: str = "feature",
    sport: str = "general",
    source_url: str = "",
    source_title: str = "",
    tags: list[str] = None,
    priority: int = 0,
) -> str:
    """Add a manually-identified hypothesis to the inbox. Returns the new ID."""
    inbox = _load_inbox()
    new_id = _next_id(inbox)

    item = {
        "id": new_id,
        "source_type": "manual",
        "source_url": source_url,
        "source_title": source_title,
        "hypothesis": hypothesis,
        "detail": detail,
        "category": category,
        "sport": sport,
        "status": "new",
        "tags": tags or [],
        "priority": priority,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": _content_hash(hypothesis),
        "test_result": None,
        "tested_at": None,
        "notes": None,
    }
    inbox.append(item)
    _save_inbox(inbox)
    return new_id


def update_status(hypothesis_id: str, status: str,
                  test_result: str = None, notes: str = None) -> bool:
    """Update the status of a hypothesis after testing."""
    valid_statuses = {"new", "queued", "testing", "confirmed", "dead", "duplicate"}
    if status not in valid_statuses:
        logger.warning("[hyp] Invalid status '%s', valid: %s", status, valid_statuses)
        return False

    inbox = _load_inbox()
    for h in inbox:
        if h.get("id") == hypothesis_id:
            h["status"] = status
            if test_result is not None:
                h["test_result"] = test_result
            if notes is not None:
                h["notes"] = notes
            if status in ("confirmed", "dead"):
                h["tested_at"] = datetime.now(timezone.utc).isoformat()
            _save_inbox(inbox)
            return True
    return False


def list_inbox(status: str = None, sport: str = None,
               category: str = None) -> list[dict]:
    """List inbox items, optionally filtered."""
    inbox = _load_inbox()
    if status:
        inbox = [h for h in inbox if h.get("status") == status]
    if sport:
        inbox = [h for h in inbox if h.get("sport") == sport]
    if category:
        inbox = [h for h in inbox if h.get("category") == category]
    return inbox


def inbox_summary() -> dict:
    """Return summary counts by status and source_type."""
    inbox = _load_inbox()
    by_status = {}
    by_source = {}
    by_sport = {}
    for h in inbox:
        s = h.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
        src = h.get("source_type", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        sp = h.get("sport", "unknown")
        by_sport[sp] = by_sport.get(sp, 0) + 1
    return {
        "total": len(inbox),
        "by_status": by_status,
        "by_source": by_source,
        "by_sport": by_sport,
    }


# ══════════════════════════════════════════════════════════════
# FULL SCRAPE
# ══════════════════════════════════════════════════════════════

def run_full_scrape(sources: list[str] = None) -> dict:
    """Run a full scrape across all (or specified) sources.

    Returns summary dict with counts per source and total added.
    """
    if sources is None:
        sources = ["reddit", "academic", "github", "blogs"]

    results = {}
    all_items = []

    if "reddit" in sources:
        items = scrape_reddit()
        results["reddit"] = len(items)
        all_items.extend(items)

    if "academic" in sources:
        items = scrape_semantic_scholar()
        results["academic"] = len(items)
        all_items.extend(items)

    if "github" in sources:
        items = scrape_github()
        results["github"] = len(items)
        all_items.extend(items)

    if "blogs" in sources:
        items = scrape_blogs_rss()
        results["blogs"] = len(items)
        all_items.extend(items)

    # Add to inbox (dedup)
    added = add_hypotheses(all_items)

    summary = {
        "scraped": results,
        "total_candidates": len(all_items),
        "new_added": added,
        "inbox_total": len(_load_inbox()),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("[hyp] Full scrape complete: %s", summary)
    return summary


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def _print_inbox(items: list[dict]) -> None:
    """Pretty-print inbox items to console."""
    if not items:
        print("  (empty)")
        return
    for h in items:
        status_icon = {
            "new": "[ ]", "queued": "[Q]", "testing": "[~]",
            "confirmed": "[+]", "dead": "[X]", "duplicate": "[D]",
        }.get(h.get("status"), "[?]")
        sport = h.get("sport", "?").upper()[:3]
        tags = ", ".join(h.get("tags", [])[:3])
        hyp = h.get("hypothesis") or h.get("source_title", "")[:80]
        print(f"  {status_icon} {h.get('id', '?'):>6}  [{sport}]  {hyp}")
        if tags:
            print(f"         tags: {tags}")
        if h.get("source_url"):
            print(f"         src:  {h['source_url'][:80]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="BetIQ Hypothesis Scraper")
    parser.add_argument("--source", choices=["reddit", "academic", "github", "blogs"],
                        help="Scrape a single source type")
    parser.add_argument("--list", action="store_true", help="List inbox items")
    parser.add_argument("--status", help="Filter by status (new/queued/testing/confirmed/dead)")
    parser.add_argument("--sport", help="Filter by sport (mlb/nfl/nba/general)")
    parser.add_argument("--summary", action="store_true", help="Show inbox summary")
    args = parser.parse_args()

    if args.summary:
        s = inbox_summary()
        print(f"\nHypothesis Inbox Summary ({s['total']} total)")
        print(f"  By status: {s['by_status']}")
        print(f"  By source: {s['by_source']}")
        print(f"  By sport:  {s['by_sport']}")

    elif args.list:
        items = list_inbox(status=args.status, sport=args.sport)
        print(f"\nHypothesis Inbox ({len(items)} items):")
        _print_inbox(items)

    else:
        sources = [args.source] if args.source else None
        print("Running hypothesis scrape...")
        result = run_full_scrape(sources=sources)
        print(f"\nScrape Results:")
        for src, count in result.get("scraped", {}).items():
            print(f"  {src}: {count} candidates")
        print(f"  New added: {result['new_added']}")
        print(f"  Inbox total: {result['inbox_total']}")
