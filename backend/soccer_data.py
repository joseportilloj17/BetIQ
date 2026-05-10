"""
soccer_data.py — Soccer match result fetching and storage.

Two data sources:
  - API-Football v3 (recent only — free plan: yesterday / today / tomorrow)
  - football-data.org v4 (historical — free plan: 2023+ seasons, 10 req/min)

Primary:  fetch_soccer_results()       — tries API-Football first, then fdorg
Fallback: fetch_soccer_results_fdorg() — football-data.org only

Usage:
  results = fetch_soccer_results("2026-04-23")   # auto-selects source
  store_soccer_results(results, db)
  game = get_soccer_result("Barcelona", "Real Madrid", "2026-04-23", db)
  corners = fetch_corner_stats(fixture_id)  # API-Football only
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, date as _date
from typing import Optional

import requests
from sqlalchemy.orm import Session

from config import (
    API_FOOTBALL_KEY, API_FOOTBALL_BASE, LEAGUE_IDS,
    FOOTBALL_DATA_KEY, FOOTBALL_DATA_BASE, FOOTBALL_DATA_COMPS,
)
from database import SoccerResult

logger = logging.getLogger(__name__)

_HEADERS = {
    "x-rapidapi-key":  API_FOOTBALL_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io",
}

_LEAGUE_ID_SET = set(LEAGUE_IDS.values())   # {39, 140, 78, 61, 135, 2, 3}

# Reverse map for label lookup
_ID_TO_LEAGUE_NAME: dict[int, str] = {
    39:  "EPL",
    140: "La Liga",
    78:  "Bundesliga",
    61:  "Ligue 1",
    135: "Serie A",
    2:   "UCL",
    3:   "Europa League",
}


# ──────────────────────────────────────────────────────────────────────────────
# Fetch helpers
# ──────────────────────────────────────────────────────────────────────────────

# ── football-data.org competition code → league name mapping ─────────────────
_FDORG_CODE_TO_NAME: dict[str, str] = {v: k for k, v in FOOTBALL_DATA_COMPS.items()}
# e.g. "PL" → "EPL", "CL" → "UCL", "PD" → "LaLiga", etc.
_FDORG_COMP_CODES = ",".join(FOOTBALL_DATA_COMPS.values())   # "PL,CL,PD,BL1,FL1,SA"

_FDORG_HEADERS = {"X-Auth-Token": FOOTBALL_DATA_KEY}


def fetch_soccer_results_fdorg(date_str: str) -> list[dict]:
    """
    Fetch finished matches for *date_str* from football-data.org v4.

    Covers: EPL, UCL, La Liga, Bundesliga, Ligue 1, Serie A.
    Free plan supports 2023+ seasons; 10 requests/minute rate limit.

    Returns same dict shape as fetch_soccer_results_apifootball().
    fixture_id uses football-data.org match id (negative to avoid collisions).
    """
    # football-data.org dateTo is exclusive — add 1 day for inclusive single-day query
    from datetime import timedelta
    date_obj = _date.fromisoformat(date_str)
    date_to  = (date_obj + timedelta(days=1)).isoformat()

    url = f"{FOOTBALL_DATA_BASE}/matches"
    params = {
        "competitions": _FDORG_COMP_CODES,
        "dateFrom":     date_str,
        "dateTo":       date_to,
        "status":       "FINISHED",
    }

    print(f"[soccer/fdorg] Fetching date {date_str}")
    print(f"[soccer/fdorg] Key prefix: {FOOTBALL_DATA_KEY[:4] if FOOTBALL_DATA_KEY else 'MISSING'}")

    try:
        resp = requests.get(url, headers=_FDORG_HEADERS, params=params, timeout=15)
        print(f"[soccer/fdorg] Response status: {resp.status_code}")
        if resp.status_code == 429:
            print("[soccer/fdorg] Rate limited (10 req/min) — skipping")
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("[soccer_data] fdorg error: %s", exc)
        print(f"[soccer/fdorg] Request error: {exc}")
        return []

    matches = data.get("matches", [])
    print(f"[soccer/fdorg] Returned {len(matches)} FINISHED matches")

    results: list[dict] = []
    for m in matches:
        # Filter to exact date (dateTo is exclusive so window includes next day's UTC midnight games)
        utc_date = (m.get("utcDate") or "")[:10]
        if utc_date and utc_date != date_str:
            continue

        comp_code   = m.get("competition", {}).get("code", "")
        league_name = _FDORG_CODE_TO_NAME.get(comp_code, comp_code)
        match_id    = m.get("id")
        if not match_id:
            continue

        score = m.get("score", {}).get("fullTime", {})
        home_goals = score.get("home")
        away_goals = score.get("away")

        results.append({
            "fixture_id":   match_id,          # fdorg match id (int, no collision risk)
            "date":         date_str,
            "league_id":    None,              # fdorg doesn't use API-Football league IDs
            "league_name":  league_name,
            "home_team":    m.get("homeTeam", {}).get("name", ""),
            "away_team":    m.get("awayTeam", {}).get("name", ""),
            "home_goals":   home_goals,
            "away_goals":   away_goals,
            "home_corners": None,
            "away_corners": None,
        })

    print(f"[soccer/fdorg] Stored {len(results)} results for {date_str}")
    logger.info("[soccer_data] fetch_soccer_results_fdorg(%s): %d fixtures", date_str, len(results))
    return results


def fetch_soccer_results(date_str: str) -> list[dict]:
    """
    Auto-select data source:
      - API-Football v3 for recent dates (yesterday / today / tomorrow)
      - football-data.org for all other dates (historical, 2023+)

    This is the main entry point for the scheduler and backfill endpoint.
    """
    today = _date.today()
    try:
        target = _date.fromisoformat(date_str)
    except ValueError:
        logger.warning("[soccer_data] Invalid date_str: %s", date_str)
        return []

    delta = (today - target).days   # positive = past, negative = future

    if delta <= 1:
        # Recent date — try API-Football first
        results = _fetch_apifootball(date_str)
        if results:
            return results
        # Fallback to fdorg if API-Football returned nothing
        print(f"[soccer] API-Football returned 0 for {date_str}, trying fdorg fallback")
        return fetch_soccer_results_fdorg(date_str)
    else:
        # Historical — go straight to football-data.org
        return fetch_soccer_results_fdorg(date_str)


# Rename the original raw fetch to an internal helper
def _fetch_apifootball(date_str: str) -> list[dict]:
    """Internal: fetch from API-Football v3 (recent dates only)."""
    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"date": date_str, "status": "FT"}

    print(f"[soccer/apifootball] Fetching date {date_str}")
    print(f"[soccer/apifootball] Key prefix: {API_FOOTBALL_KEY[:4] if API_FOOTBALL_KEY else 'MISSING'}")

    try:
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=15)
        print(f"[soccer/apifootball] Response status: {resp.status_code}")
        data = resp.json()
    except Exception as exc:
        logger.error("[soccer_data] apifootball error: %s", exc)
        print(f"[soccer/apifootball] Request error: {exc}")
        return []

    api_errors = data.get("errors")
    if api_errors:
        print(f"[soccer/apifootball] API errors: {api_errors}")
        return []

    fixtures = data.get("response", [])
    print(f"[soccer/apifootball] Total FT fixtures from API: {len(fixtures)}")

    results: list[dict] = []
    for fix in fixtures:
        league_id = fix.get("league", {}).get("id")
        if league_id not in _LEAGUE_ID_SET:
            continue
        fixture_id = fix.get("fixture", {}).get("id")
        if not fixture_id:
            continue
        goals = fix.get("goals", {})
        teams = fix.get("teams", {})
        results.append({
            "fixture_id":   fixture_id,
            "date":         date_str,
            "league_id":    league_id,
            "league_name":  _ID_TO_LEAGUE_NAME.get(league_id, "Soccer"),
            "home_team":    teams.get("home", {}).get("name", ""),
            "away_team":    teams.get("away", {}).get("name", ""),
            "home_goals":   goals.get("home"),
            "away_goals":   goals.get("away"),
            "home_corners": None,
            "away_corners": None,
        })

    print(f"[soccer/apifootball] Matched {len(results)} fixtures in tracked leagues")
    return results


def fetch_corner_stats(fixture_id: int) -> tuple[Optional[int], Optional[int]]:
    """
    Fetch corner counts for a single fixture.
    Returns (home_corners, away_corners) or (None, None) on failure.
    """
    url = f"{API_FOOTBALL_BASE}/fixtures/statistics"
    params = {"fixture": fixture_id}

    try:
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("[soccer_data] fetch_corner_stats(%s) error: %s", fixture_id, exc)
        return None, None

    stats = data.get("response", [])
    home_corners = away_corners = None

    for team_stats in stats:
        team_type = (team_stats.get("team", {}).get("name") or "").lower()
        is_home   = team_stats.get("statistics", [{}])[0].get("value") is not None  # positional heuristic

        for stat in team_stats.get("statistics", []):
            if stat.get("type", "").lower() == "corner kicks":
                val = stat.get("value")
                try:
                    val = int(val) if val is not None else None
                except (ValueError, TypeError):
                    val = None

                # API returns home team first, away team second
                if home_corners is None:
                    home_corners = val
                else:
                    away_corners = val
                break

    return home_corners, away_corners


# ──────────────────────────────────────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────────────────────────────────────

def store_soccer_results(results: list[dict], db: Session) -> int:
    """
    Upsert soccer results into the soccer_results table.
    Returns number of rows written.
    """
    written = 0
    for r in results:
        existing = db.query(SoccerResult).filter_by(fixture_id=r["fixture_id"]).first()
        if existing:
            # Update scores if we now have them
            existing.home_goals   = r.get("home_goals",   existing.home_goals)
            existing.away_goals   = r.get("away_goals",   existing.away_goals)
            existing.home_corners = r.get("home_corners", existing.home_corners)
            existing.away_corners = r.get("away_corners", existing.away_corners)
            existing.fetched_at   = datetime.utcnow()
        else:
            db.add(SoccerResult(
                fixture_id   = r["fixture_id"],
                date         = r["date"],
                league_id    = r.get("league_id"),
                league_name  = r.get("league_name"),
                home_team    = r["home_team"],
                away_team    = r["away_team"],
                home_goals   = r.get("home_goals"),
                away_goals   = r.get("away_goals"),
                home_corners = r.get("home_corners"),
                away_corners = r.get("away_corners"),
                fetched_at   = datetime.utcnow(),
            ))
        written += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("[soccer_data] store_soccer_results commit error: %s", exc)

    return written


# ──────────────────────────────────────────────────────────────────────────────
# Lookup
# ──────────────────────────────────────────────────────────────────────────────

_COMMON_SUFFIXES = re.compile(
    # "real", "athletic", "city", "united" intentionally excluded:
    #   - "real" / "athletic" are distinctive Spanish team words
    #   - "city" / "united" distinguish Manchester City vs Manchester United;
    #     aliases ("man city" → "manchester city") preserve them post-lookup.
    r"\b(fc|sc|cf|ac|as|rc|ud|fk|sk|bk|sporting|club|de|del|la|el)\b",
    re.IGNORECASE,
)

# Characters that NFKD + combining-char-strip doesn't handle (stroke letters, etc.)
_STROKE_TRANS = str.maketrans({
    'ø': 'o', 'Ø': 'o',
    'ł': 'l', 'Ł': 'l',
    'ð': 'd', 'Ð': 'd',
    'þ': 'th',
    'æ': 'ae', 'Æ': 'ae',
    'œ': 'oe', 'Œ': 'oe',
    'ß': 'ss',
})

# Pre-normalization alias map: maps lowercase full name → canonical form
# Applied before suffix stripping so abbreviations resolve correctly.
_TEAM_ALIASES: dict[str, str] = {
    # Manchester clubs (city/united stripped → "man" alone is ambiguous)
    "man city":          "manchester city",
    "man utd":           "manchester united",
    "man united":        "manchester united",
    # Italian club abbreviations
    "inter":             "internazionale",
    "inter milan":       "internazionale milan",
    # French clubs (English names vs official French names)
    "lyon":              "olympique lyonnais",
    "paris st-g":        "paris saint-germain",
    "paris sg":          "paris saint-germain",
    "psg":               "paris saint-germain",
    # Portuguese clubs
    "sporting lisbon":   "sporting clube portugal",
    "sporting cp":       "sporting clube portugal",
    "benfica":           "sl benfica",
    # Danish clubs
    "fc copenhagen":     "fc kobenhavn",
    "copenhagen":        "kobenhavn",
    # Serbian clubs
    "crvena zvezda":     "crvena zvezda",   # normalize after FK strip
    "red star":          "crvena zvezda",
    "red star belgrade": "crvena zvezda",
    # Spanish clubs (full official names)
    "oviedo":            "real oviedo",
}


def _normalize_team(name: str) -> str:
    """Strip accents, common suffixes/articles for fuzzy matching."""
    # Remove combining diacritics: "Atlético" → "Atletico", "Mönchengladbach" → "Monchengladbach"
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Handle stroke/non-decomposable chars: "ø" → "o", "ł" → "l", etc.
    s = s.translate(_STROKE_TRANS)
    s = s.lower().strip()
    # Apply alias before stripping suffixes (handles nicknames/abbreviations)
    s = _TEAM_ALIASES.get(s, s)
    s = _COMMON_SUFFIXES.sub("", s)
    # Normalize slash to space: "FK Bodø/Glimt" → "bodo glimt"
    s = s.replace("/", " ")
    # German city name: NFKD maps "ü" → "u" giving "munchen" ≠ English "munich"
    s = s.replace("munchen", "munich")
    return re.sub(r"\s+", " ", s).strip()


def _team_match(needle: str, candidate: str) -> bool:
    """
    True if the normalized team names refer to the same club.

    Uses word-level matching (not substring) to prevent "madrid" from
    matching both "atletico madrid" and "real madrid".

    Rules (checked in order):
      1. Exact normalized string equality
      2. Single-word shorter: that word must appear in the longer name's word set
      3. Multi-word: all words of the SHORTER name appear in the LONGER name
         (partial names like "Rayo Vallecano" match "Rayo Vallecano de Madrid")
    """
    n = _normalize_team(needle)
    c = _normalize_team(candidate)
    if n == c:
        return True
    nw = set(n.split())
    cw = set(c.split())
    if not nw or not cw:
        return False
    shorter, longer = (nw, cw) if len(nw) <= len(cw) else (cw, nw)
    if len(shorter) == 1:
        # Single distinctive word (e.g. "Nice", "Juventus", "Barcelona"):
        # it just needs to appear in the other team's word set.
        return next(iter(shorter)) in longer
    else:
        # Multi-word: all words of the SHORTER name must appear in the LONGER
        # name. Prevents "atletico madrid" matching "real madrid" (atletico ∉
        # {"real","madrid"}), while allowing "Rayo Vallecano" to match
        # "Rayo Vallecano de Madrid" ({"rayo","vallecano"} ⊆ longer set).
        return shorter.issubset(longer)


def get_soccer_result(
    home_team: str,
    away_team: str,
    date_str: str,
    db: Session,
) -> Optional[SoccerResult]:
    """
    Look up a completed soccer fixture from the local cache.

    Tries exact match first, then falls back to fuzzy name matching
    (handles "FC Barcelona" vs "Barcelona", "Man City" vs "Manchester City").
    """
    # Exact match
    exact = (
        db.query(SoccerResult)
        .filter(
            SoccerResult.date == date_str,
            SoccerResult.home_team == home_team,
            SoccerResult.away_team == away_team,
        )
        .first()
    )
    if exact:
        return exact

    # Fuzzy match — scan same-date rows
    rows = db.query(SoccerResult).filter(SoccerResult.date == date_str).all()
    for row in rows:
        if _team_match(home_team, row.home_team) and _team_match(away_team, row.away_team):
            return row

    return None
