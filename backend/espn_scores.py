"""
espn_scores.py — Free-tier score fetcher for settlement.

Replaces OddsAPI /scores endpoint with ESPN Scoreboard API and MLB Stats API.
These are completely free, require no authentication, and have no rate limits
relevant to our settlement volume.

ESPN endpoints:
  MLB:  site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=YYYYMMDD
  NBA:  site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
  NHL:  site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates=YYYYMMDD
  Soccer: site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates=YYYYMMDD

Output format mirrors OddsAPI scores format so existing settlement logic
(_settle_mock_leg, _resolve_leg) works unchanged:
  [
    {
      "id":         "<espn_event_id>",
      "sport_key":  "baseball_mlb",
      "completed":  True,
      "home_team":  "New York Yankees",
      "away_team":  "Boston Red Sox",
      "scores": [
        {"name": "New York Yankees", "score": "5"},
        {"name": "Boston Red Sox",   "score": "3"},
      ],
      "commence_time": "2026-05-17T19:05:00Z",
      "source": "espn",
    },
    ...
  ]
"""
from __future__ import annotations

import re
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import urllib.request
import urllib.error
import json

logger = logging.getLogger(__name__)

# ─── ESPN league slugs ────────────────────────────────────────────────────────

# Maps OddsAPI sport_key → ESPN scoreboard path
_SPORT_KEY_TO_ESPN: dict[str, str] = {
    "baseball_mlb":              "baseball/mlb",
    "basketball_nba":            "basketball/nba",
    "icehockey_nhl":             "hockey/nhl",
    "americanfootball_nfl":      "football/nfl",
    "americanfootball_ncaaf":    "football/college-football",
    "basketball_ncaab":          "basketball/mens-college-basketball",
    "soccer_epl":                "soccer/eng.1",
    "soccer_spain_la_liga":      "soccer/esp.1",
    "soccer_france_ligue_one":   "soccer/fra.1",
    "soccer_italy_serie_a":      "soccer/ita.1",
    "soccer_germany_bundesliga": "soccer/ger.1",
    "soccer_uefa_champs_league": "soccer/uefa.champions",
    "soccer_usa_mls":            "soccer/usa.1",
    "soccer_uefa_europa_league": "soccer/uefa.europa",
    "soccer_fifa_world_cup":     "soccer/fifa.world",
}

# Maps BetIQ sport label → OddsAPI sport_key (same as SPORT_TO_KEYS in auto_settle)
SPORT_LABEL_TO_KEY: dict[str, str] = {
    "MLB":        "baseball_mlb",
    "NBA":        "basketball_nba",
    "NHL":        "icehockey_nhl",
    "NFL":        "americanfootball_nfl",
    "EPL":        "soccer_epl",
    "La Liga":    "soccer_spain_la_liga",
    "Ligue 1":    "soccer_france_ligue_one",
    "Serie A":    "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "UCL":        "soccer_uefa_champs_league",
    "MLS":        "soccer_usa_mls",
    "Europa League": "soccer_uefa_europa_league",
    "World Cup":     "soccer_fifa_world_cup",
}

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# ─── TheSportsDB league IDs ───────────────────────────────────────────────────
# Free API (key=3, no auth).  Used as last-resort fallback when ESPN and OddsAPI
# both return nothing — e.g. end-of-season playoff/barrage games after the
# regular season calendar closes.  eventspastleague.php returns the ~5 most
# recent completed events per league, which is sufficient for these edge cases.

_TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

_SPORT_KEY_TO_TSDB_LEAGUE: dict[str, int] = {
    "baseball_mlb":              4424,
    "basketball_nba":            4387,
    "icehockey_nhl":             4380,
    "americanfootball_nfl":      4391,
    "soccer_epl":                4328,
    "soccer_spain_la_liga":      4335,
    "soccer_france_ligue_one":   4334,
    "soccer_italy_serie_a":      4332,
    "soccer_germany_bundesliga": 4331,
    "soccer_usa_mls":            4346,
    "soccer_uefa_champs_league": 4480,
    "soccer_uefa_europa_league": 4481,
    "soccer_fifa_world_cup":     4429,
}

_tsdb_cache: dict[str, tuple[list, datetime]] = {}
_TSDB_TTL = 900  # 15 minutes

# ─── Cache ────────────────────────────────────────────────────────────────────

_score_cache: dict[str, tuple[list, datetime]] = {}
_CACHE_TTL = 900  # 15 minutes


def _http_get(url: str, timeout: int = 10) -> Optional[dict]:
    """Simple urllib JSON GET — no external dependencies."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BetIQ/1.0 (settlement scoring; educational project)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        logger.warning("ESPN scores fetch failed: %s — %s", url, exc)
        return None


# ─── ESPN parser ─────────────────────────────────────────────────────────────

def _parse_espn_scoreboard(data: dict, sport_key: str) -> list[dict]:
    """
    Parse ESPN scoreboard API response → OddsAPI-compatible score dicts.
    Handles all sports (MLB, NBA, NHL, Soccer, NFL).
    """
    results: list[dict] = []
    events = data.get("events") or []

    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]

        # Status
        status_obj = comp.get("status") or {}
        status_type = (status_obj.get("type") or {})
        completed = status_type.get("completed", False)
        # Also accept "STATUS_FINAL" strings
        if not completed:
            desc = status_type.get("description", "")
            completed = "Final" in desc or "Completed" in desc

        # Teams and scores
        competitors = comp.get("competitors") or []
        home_name = away_name = ""
        home_score = away_score = None

        for c in competitors:
            team = (c.get("team") or {}).get("displayName", "")
            score_str = c.get("score", "")
            try:
                score_val = int(score_str) if str(score_str).isdigit() else int(float(score_str))
            except (ValueError, TypeError):
                score_val = None
            if c.get("homeAway") == "home":
                home_name = team
                home_score = score_val
            else:
                away_name = team
                away_score = score_val

        if not home_name or not away_name:
            continue

        scores_list = []
        if home_score is not None:
            scores_list.append({"name": home_name, "score": str(home_score)})
        if away_score is not None:
            scores_list.append({"name": away_name, "score": str(away_score)})

        # Commence time
        commence = event.get("date") or comp.get("date") or ""

        results.append({
            "id":            event.get("id", ""),
            "sport_key":     sport_key,
            "completed":     completed,
            "home_team":     home_name,
            "away_team":     away_name,
            "scores":        scores_list,
            "commence_time": commence,
            "source":        "espn",
        })

    return results


# ─── Public API ──────────────────────────────────────────────────────────────

def fetch_espn_scores(sport_key: str, days_from: int = 7) -> list[dict]:
    """
    Fetch completed game scores for a sport_key over the last `days_from` days.

    Args:
        sport_key: OddsAPI sport key (e.g. "baseball_mlb", "icehockey_nhl")
        days_from: Number of past calendar days to fetch (inclusive of today)

    Returns:
        List of OddsAPI-compatible score dicts (completed games only).
        Empty list if sport_key not supported or request fails.
    """
    espn_path = _SPORT_KEY_TO_ESPN.get(sport_key)
    if not espn_path:
        logger.debug("No ESPN mapping for sport_key=%s", sport_key)
        return []

    cache_key = f"{sport_key}:{days_from}"
    now = datetime.utcnow()
    cached = _score_cache.get(cache_key)
    if cached and (now - cached[1]).total_seconds() < _CACHE_TTL:
        return cached[0]

    all_results: list[dict] = []
    seen_ids: set[str] = set()

    # Fetch each date in the window (today + past days_from-1 days)
    for delta in range(days_from):
        date = (now - timedelta(days=delta)).strftime("%Y%m%d")
        url = f"{_ESPN_BASE}/{espn_path}/scoreboard?dates={date}&limit=50"
        data = _http_get(url)
        if not data:
            continue
        for game in _parse_espn_scoreboard(data, sport_key):
            gid = game.get("id", "")
            if gid and gid in seen_ids:
                continue
            seen_ids.add(gid)
            all_results.append(game)

    _score_cache[cache_key] = (all_results, now)
    return all_results


def fetch_espn_scores_by_label(sport_label: str, days_from: int = 7) -> list[dict]:
    """
    Convenience wrapper: accepts BetIQ sport label ("MLB", "NBA", "NHL", "EPL", etc.)
    and returns completed game scores.
    """
    sport_key = SPORT_LABEL_TO_KEY.get(sport_label)
    if not sport_key:
        # Try pipe-separated labels ("EPL | La Liga")
        results: list[dict] = []
        seen: set[str] = set()
        for part in sport_label.split("|"):
            sk = SPORT_LABEL_TO_KEY.get(part.strip())
            if sk:
                for g in fetch_espn_scores(sk, days_from):
                    gid = g.get("id", "")
                    if gid and gid in seen:
                        continue
                    seen.add(gid)
                    results.append(g)
        return results
    return fetch_espn_scores(sport_key, days_from)


def fetch_mlb_scores_statsapi(game_date: str) -> list[dict]:
    """
    Fetch MLB scores from MLB Stats API (free, no auth) for a specific date.
    Falls back to ESPN if Stats API is unavailable.

    Args:
        game_date: "YYYY-MM-DD" date string

    Returns:
        OddsAPI-compatible score dicts for completed MLB games on that date.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date}&hydrate=linescore"
    )
    data = _http_get(url)
    if not data:
        return fetch_espn_scores("baseball_mlb", days_from=7)

    results: list[dict] = []
    for date_entry in (data.get("dates") or []):
        for game in (date_entry.get("games") or []):
            status = game.get("status", {}).get("abstractGameState", "")
            completed = status in ("Final", "Completed Early")

            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_name = (home.get("team") or {}).get("name", "")
            away_name = (away.get("team") or {}).get("name", "")

            # Linescore scores
            linescore = game.get("linescore") or {}
            home_score = linescore.get("teams", {}).get("home", {}).get("runs")
            away_score = linescore.get("teams", {}).get("away", {}).get("runs")

            # Fall back to teams.*.score
            if home_score is None:
                home_score = home.get("score")
            if away_score is None:
                away_score = away.get("score")

            scores_list = []
            if home_score is not None:
                scores_list.append({"name": home_name, "score": str(home_score)})
            if away_score is not None:
                scores_list.append({"name": away_name, "score": str(away_score)})

            game_pk = str(game.get("gamePk", ""))
            commence = game.get("gameDate", game_date + "T00:00:00Z")

            results.append({
                "id":            f"statsapi_{game_pk}",
                "sport_key":     "baseball_mlb",
                "completed":     completed,
                "home_team":     home_name,
                "away_team":     away_name,
                "scores":        scores_list,
                "commence_time": commence,
                "source":        "statsapi",
            })

    return results


# ─── TheSportsDB fallback ─────────────────────────────────────────────────────

def fetch_thesportsdb_scores(sport_key: str, days_from: int = 7) -> list[dict]:
    """
    Fetch recent completed game scores from TheSportsDB's free public API.

    Last-resort fallback when ESPN and OddsAPI both return nothing — specifically
    for end-of-season playoff/barrage games that the main APIs stop serving after
    the regular season calendar closes.

    Uses eventspastleague.php which returns the most recent ~5 completed events
    per league.  That is sufficient for post-season edge cases but does NOT
    replace ESPN for full regular-season coverage.
    """
    league_id = _SPORT_KEY_TO_TSDB_LEAGUE.get(sport_key)
    if not league_id:
        return []

    cache_key = f"{sport_key}:tsdb:{days_from}"
    now = datetime.utcnow()
    cached = _tsdb_cache.get(cache_key)
    if cached and (now - cached[1]).total_seconds() < _TSDB_TTL:
        return cached[0]

    cutoff_date = (now - timedelta(days=days_from)).strftime("%Y-%m-%d")
    url = f"{_TSDB_BASE}/eventspastleague.php?id={league_id}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BetIQ/1.0 (settlement scoring; educational project)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("TheSportsDB fetch failed for %s: %s", sport_key, exc)
        _tsdb_cache[cache_key] = ([], now)
        return []

    results: list[dict] = []
    for e in data.get("events") or []:
        event_date = e.get("dateEvent") or ""
        if event_date < cutoff_date:
            continue

        home = e.get("strHomeTeam") or ""
        away = e.get("strAwayTeam") or ""
        home_sc_raw = e.get("intHomeScore")
        away_sc_raw = e.get("intAwayScore")

        if not home or not away or home_sc_raw is None or away_sc_raw is None:
            continue
        try:
            home_sc = int(home_sc_raw)
            away_sc = int(away_sc_raw)
        except (ValueError, TypeError):
            continue

        event_time = e.get("strTime") or "00:00:00"
        commence   = f"{event_date}T{event_time}Z"

        results.append({
            "id":            f"tsdb_{e.get('idEvent', '')}",
            "sport_key":     sport_key,
            "completed":     True,
            "home_team":     home,
            "away_team":     away,
            "scores": [
                {"name": home, "score": str(home_sc)},
                {"name": away, "score": str(away_sc)},
            ],
            "commence_time": commence,
            "source":        "thesportsdb",
        })

    _tsdb_cache[cache_key] = (results, now)
    if results:
        logger.info("TheSportsDB fallback: %d completed events for %s (league=%d)",
                    len(results), sport_key, league_id)
    return results


# ─── Drop-in replacement for auto_settle._fetch_scores_cached ────────────────

def fetch_scores_free(sport_key: str, days_from: int = 7) -> list[dict]:
    """
    Drop-in replacement for auto_settle._fetch_scores_cached().

    Tries ESPN first (covers all sports). For MLB, also augments with MLB
    Stats API results to maximise coverage.

    Returns OddsAPI-compatible score dicts — existing settlement code works
    unchanged.

    Credit cost: $0 (no OddsAPI calls).
    """
    espn_results = fetch_espn_scores(sport_key, days_from)

    # For MLB, also pull from MLB Stats API for each date in the window
    if sport_key == "baseball_mlb":
        now = datetime.utcnow()
        seen_ids = {g["id"] for g in espn_results}
        for delta in range(days_from):
            date_str = (now - timedelta(days=delta)).strftime("%Y-%m-%d")
            for game in fetch_mlb_scores_statsapi(date_str):
                gid = game.get("id", "")
                if gid not in seen_ids:
                    espn_results.append(game)
                    seen_ids.add(gid)

    return espn_results
