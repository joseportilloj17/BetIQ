"""
backfill_lines.py — Backfill close_spread / close_total / close_ml into betting_lines.

NBA (all seasons)
-----------------
Source: ESPN Core API (free, no key required)
  1. For each game date, fetch ESPN NBA scoreboard → build (date, home_espn_abbr) → event_id map
  2. For each event, fetch competition odds → ESPN BET provider (priority 0)
  3. Extract: spread (home perspective), overUnder, homeTeamOdds.moneyLine, awayTeamOdds.moneyLine
  4. Write to betting_lines.close_spread / close_total / close_ml_home / close_ml_away
  ESPN has historical odds going back to at least 2022-23.

MLB
---
Source: computed from stored scores (home_score - away_score vs 1.5 run line)
  - close_spread = -1.5 for all completed MLB games (home team treated as -1.5)
  - close_total  = NULL (no free source available)

Usage
-----
    python backfill_lines.py --sport NBA --dry-run
    python backfill_lines.py --sport MLB
    python backfill_lines.py --sport NBA
    python backfill_lines.py --sport NBA --seasons 2022-23 2023-24   # specific seasons
    python backfill_lines.py          # both sports, all seasons
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

# ── ESPN abbreviation → BallDontLie/nba-api abbreviation ──────────────────────
ESPN_TO_BDL: dict[str, str] = {
    "GS":   "GSW",
    "NO":   "NOP",
    "NY":   "NYK",
    "SA":   "SAS",
    "UTAH": "UTA",
    "WSH":  "WAS",
}

# ESPN Core API base
ESPN_CORE_NBA  = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ESPN_SITE_NBA  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_CORE_MLB  = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
ESPN_SITE_MLB  = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

# Backwards-compat aliases used by existing NBA functions
ESPN_CORE = ESPN_CORE_NBA
ESPN_SITE = ESPN_SITE_NBA

# Request delay to stay well within ESPN's rate limits
_DELAY_SCOREBOARD = 0.15   # seconds between scoreboard requests
_DELAY_ODDS       = 0.20   # seconds between odds requests


# ══════════════════════════════════════════════════════════════════════════════
# NBA — ESPN odds backfill
# ══════════════════════════════════════════════════════════════════════════════

def _espn_get(url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[dict]:
    """GET with retry/backoff. Returns None on persistent failure."""
    from http_retry import get_with_retry
    r = get_with_retry(url, params=params, timeout=15, max_attempts=retries, label="ESPN")
    if r is None:
        return None
    return r.json()


def _build_date_event_map(game_dates: list[str]) -> dict[tuple[str, str], str]:
    """
    For each date, fetch ESPN scoreboard and return:
        {(date_str, home_bdl_abbr): espn_event_id}
    """
    event_map: dict[tuple[str, str], str] = {}
    for i, date_str in enumerate(game_dates):
        date_esc = date_str.replace("-", "")
        data = _espn_get(f"{ESPN_SITE}/scoreboard", params={"dates": date_esc, "limit": 30})
        if not data:
            log.warning("scoreboard: no data for %s", date_str)
            time.sleep(_DELAY_SCOREBOARD)
            continue

        for event in data.get("events", []):
            eid = event.get("id")
            comp = (event.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            espn_home = home.get("team", {}).get("abbreviation", "")
            bdl_home = ESPN_TO_BDL.get(espn_home, espn_home)   # translate if needed
            if eid and bdl_home:
                event_map[(date_str, bdl_home)] = eid

        if (i + 1) % 25 == 0:
            log.info("  scoreboard: %d/%d dates fetched, %d events mapped",
                     i + 1, len(game_dates), len(event_map))
        time.sleep(_DELAY_SCOREBOARD)

    return event_map


def _fetch_espn_odds(event_id: str) -> Optional[dict]:
    """
    Fetch odds for *event_id* from ESPN Core API.
    Uses the first (highest-priority) provider — ESPN BET.

    Returns dict with keys: spread, over_under, ml_home, ml_away
    or None if odds unavailable.
    """
    odds_list_url = (
        f"{ESPN_CORE}/events/{event_id}/competitions/{event_id}/odds"
    )
    list_data = _espn_get(odds_list_url)
    if not list_data or not list_data.get("items"):
        return None

    # Fetch first provider (priority 0 = ESPN BET)
    first_ref = list_data["items"][0].get("$ref")
    if not first_ref:
        return None
    odds_data = _espn_get(first_ref)
    if not odds_data:
        return None

    spread     = odds_data.get("spread")          # home team's spread, e.g. -5.5
    over_under = odds_data.get("overUnder")        # O/U total, e.g. 221.5
    home_odds  = odds_data.get("homeTeamOdds", {})
    away_odds  = odds_data.get("awayTeamOdds", {})

    # Prefer close ML; fall back to top-level moneyLine
    ml_home = (
        _parse_american(home_odds.get("close", {}).get("moneyLine", {}).get("american"))
        or home_odds.get("moneyLine")
    )
    ml_away = (
        _parse_american(away_odds.get("close", {}).get("moneyLine", {}).get("american"))
        or away_odds.get("moneyLine")
    )

    if spread is None and over_under is None:
        return None

    return {
        "spread":     float(spread)     if spread     is not None else None,
        "over_under": float(over_under) if over_under is not None else None,
        "ml_home":    float(ml_home)    if ml_home    is not None else None,
        "ml_away":    float(ml_away)    if ml_away    is not None else None,
    }


def _parse_american(value) -> Optional[float]:
    """Parse an American odds string/number like '+160' or '-190' → float."""
    if value is None:
        return None
    try:
        return float(str(value).replace("+", ""))
    except (ValueError, TypeError):
        return None


def backfill_nba_espn(
    dry_run: bool = False,
    seasons: list[str] | None = None,
) -> dict:
    """
    Backfill close_spread / close_total / close_ml for NBA games from ESPN.

    seasons : optional list of season strings (e.g. ["2022-23", "2023-24"]).
              Defaults to ALL seasons (no filter) — fills any row with NULL spread.
    Returns summary dict with counts.
    """
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    cur  = conn.cursor()

    # Build season filter clause if requested
    if seasons:
        placeholders = ",".join("?" * len(seasons))
        season_clause = f"AND g.season IN ({placeholders})"
        season_params = list(seasons)
    else:
        season_clause = ""
        season_params = []

    # Fetch NBA games that still have NULL close_spread
    cur.execute(f"""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team
        FROM games g
        JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.sport   = 'NBA'
          {season_clause}
          AND g.status  = 'Final'
          AND bl.close_spread IS NULL
        ORDER BY g.game_date
    """, season_params)
    games = cur.fetchall()
    log.info("NBA: %d games need odds backfill", len(games))

    if not games:
        conn.close()
        return {"sport": "NBA", "attempted": 0, "filled": 0, "missing": 0}

    # Build date → event_id map
    unique_dates = sorted(set(row[1] for row in games))
    log.info("Fetching ESPN scoreboard for %d unique dates…", len(unique_dates))
    event_map = _build_date_event_map(unique_dates)
    log.info("Event map built: %d entries", len(event_map))

    filled = missing = 0
    for idx, (game_id, game_date, home_team, away_team) in enumerate(games):
        event_id = event_map.get((game_date, home_team))
        if not event_id:
            log.debug("No ESPN event for game_id=%s (%s %s @ %s)",
                      game_id, game_date, away_team, home_team)
            missing += 1
            continue

        time.sleep(_DELAY_ODDS)
        odds = _fetch_espn_odds(event_id)
        if not odds:
            log.debug("No ESPN odds for event_id=%s (game_id=%s)", event_id, game_id)
            missing += 1
            continue

        if dry_run:
            log.info("[DRY RUN] %s %s %s@%s → spread=%.1f total=%s ml=%s/%s",
                     game_id, game_date, away_team, home_team,
                     odds["spread"] or 0,
                     odds["over_under"],
                     odds["ml_home"], odds["ml_away"])
        else:
            cur.execute("""
                UPDATE betting_lines
                SET close_spread   = ?,
                    close_total    = ?,
                    close_ml_home  = ?,
                    close_ml_away  = ?
                WHERE game_id = ?
            """, (odds["spread"], odds["over_under"],
                  odds["ml_home"], odds["ml_away"],
                  game_id))

        filled += 1

        if (idx + 1) % 100 == 0:
            if not dry_run:
                conn.commit()
            log.info("  NBA odds: %d/%d processed (filled=%d, missing=%d)",
                     idx + 1, len(games), filled, missing)

    if not dry_run:
        conn.commit()

    conn.close()
    log.info("NBA backfill done: filled=%d  missing=%d  dry_run=%s",
             filled, missing, dry_run)
    return {"sport": "NBA", "attempted": len(games), "filled": filled, "missing": missing}


# ══════════════════════════════════════════════════════════════════════════════
# MLB — ESPN moneyline-based run-line backfill
# ══════════════════════════════════════════════════════════════════════════════

def _build_mlb_date_event_map(game_dates: list[str]) -> dict[tuple[str, str], str]:
    """
    For each date, fetch ESPN MLB scoreboard and return:
        {(date_str, home_team_displayName): espn_event_id}

    MLB: ESPN team.displayName matches our DB home_team exactly (e.g. "Boston Red Sox").
    """
    event_map: dict[tuple[str, str], str] = {}
    for i, date_str in enumerate(game_dates):
        date_esc = date_str.replace("-", "")
        data = _espn_get(f"{ESPN_SITE_MLB}/scoreboard", params={"dates": date_esc, "limit": 20})
        if not data:
            log.warning("MLB scoreboard: no data for %s", date_str)
            time.sleep(_DELAY_SCOREBOARD)
            continue

        for event in data.get("events", []):
            eid = event.get("id")
            comp = (event.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            display_name = home.get("team", {}).get("displayName", "")
            if eid and display_name:
                event_map[(date_str, display_name)] = eid

        if (i + 1) % 25 == 0:
            log.info("  MLB scoreboard: %d/%d dates, %d events mapped",
                     i + 1, len(game_dates), len(event_map))
        time.sleep(_DELAY_SCOREBOARD)

    return event_map


def _fetch_mlb_runline_odds(event_id: str) -> Optional[dict]:
    """
    Fetch run-line odds for an MLB event from ESPN Core API.

    MLB: The competition odds endpoint returns data INLINE in items[0] —
    no secondary $ref fetch needed (unlike NBA).

    - spread field = 1.5 (always absolute for MLB)
    - homeTeamOdds.favorite boolean determines sign:
        home favorite  → close_spread = -1.5
        away favorite  → close_spread = +1.5
    - Closing ML: homeTeamOdds.close.moneyLine.american (fall back to moneyLine)
    - Closing total: close.total.alternateDisplayValue (fall back to overUnder)

    Returns dict with keys:
        home_is_favorite (int 0/1), close_spread (float),
        ml_home (float|None), ml_away (float|None), over_under (float|None)
    or None if odds unavailable.
    """
    odds_list_url = (
        f"{ESPN_CORE_MLB}/events/{event_id}/competitions/{event_id}/odds"
    )
    list_data = _espn_get(odds_list_url)
    if not list_data or not list_data.get("items"):
        return None

    # MLB odds are inline — no $ref to follow
    odds_data  = list_data["items"][0]
    home_odds  = odds_data.get("homeTeamOdds", {})
    away_odds  = odds_data.get("awayTeamOdds", {})

    # home_is_favorite from ESPN boolean
    home_is_fav  = bool(home_odds.get("favorite", False))
    close_spread = -1.5 if home_is_fav else 1.5

    # Closing ML: prefer homeTeamOdds.close.moneyLine.american, fall back to moneyLine
    ml_home = (
        _parse_american(home_odds.get("close", {}).get("moneyLine", {}).get("american"))
        or home_odds.get("moneyLine")
    )
    ml_away = (
        _parse_american(away_odds.get("close", {}).get("moneyLine", {}).get("american"))
        or away_odds.get("moneyLine")
    )

    # Closing total: prefer close.total.alternateDisplayValue, fall back to overUnder
    close_sect = odds_data.get("close", {})
    over_under = (
        _parse_american(close_sect.get("total", {}).get("alternateDisplayValue"))
        or odds_data.get("overUnder")
    )

    return {
        "home_is_favorite": int(home_is_fav),
        "close_spread":     close_spread,
        "over_under":       float(over_under) if over_under is not None else None,
        "ml_home":          float(ml_home)    if ml_home    is not None else None,
        "ml_away":          float(ml_away)    if ml_away    is not None else None,
    }


def backfill_mlb_espn(dry_run: bool = False) -> dict:
    """
    Backfill MLB run-line data using ESPN Core API.

    - Fetches moneylines to determine which team is the ML favorite.
    - Sets close_spread = -1.5 (home favored) or +1.5 (away favored).
    - Stores home_is_favorite, close_ml_home, close_ml_away, close_total.

    Replaces the old synthetic-always-home-favorite approach.
    """
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    cur  = conn.cursor()

    cur.execute("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team
        FROM games g
        JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.sport        = 'MLB'
          AND g.status       = 'Final'
          AND bl.close_spread IS NULL
        ORDER BY g.game_date
    """)
    games = cur.fetchall()
    log.info("MLB ESPN: %d games need run-line backfill", len(games))

    if not games:
        conn.close()
        return {"sport": "MLB", "attempted": 0, "filled": 0, "missing": 0}

    unique_dates = sorted(set(row[1] for row in games))
    log.info("Fetching MLB scoreboard for %d unique dates…", len(unique_dates))
    event_map = _build_mlb_date_event_map(unique_dates)
    log.info("MLB event map: %d entries", len(event_map))

    filled = missing = 0
    for idx, (game_id, game_date, home_team, away_team) in enumerate(games):
        event_id = event_map.get((game_date, home_team))
        if not event_id:
            log.debug("No ESPN event for game_id=%s (%s %s @ %s)",
                      game_id, game_date, away_team, home_team)
            missing += 1
            continue

        time.sleep(_DELAY_ODDS)
        odds = _fetch_mlb_runline_odds(event_id)
        if not odds:
            # Fallback: if ESPN has no odds, apply synthetic home -1.5
            # (marks home_is_favorite as NULL so we know it's synthetic)
            if not dry_run:
                cur.execute("""
                    UPDATE betting_lines SET close_spread = -1.5 WHERE game_id = ?
                """, (game_id,))
            log.debug("No ESPN odds for event_id=%s — synthetic -1.5 applied", event_id)
            missing += 1
            continue

        if dry_run:
            log.info("[DRY RUN] %s %s %s@%s → fav=%s spread=%.1f ml=%s/%s total=%s",
                     game_id, game_date, away_team, home_team,
                     "HOME" if odds["home_is_favorite"] else "AWAY",
                     odds["close_spread"],
                     odds["ml_home"], odds["ml_away"], odds["over_under"])
        else:
            cur.execute("""
                UPDATE betting_lines
                SET close_spread      = ?,
                    close_total       = ?,
                    close_ml_home     = ?,
                    close_ml_away     = ?,
                    home_is_favorite  = ?
                WHERE game_id = ?
            """, (odds["close_spread"], odds["over_under"],
                  odds["ml_home"], odds["ml_away"],
                  odds["home_is_favorite"],
                  game_id))

        filled += 1

        if (idx + 1) % 100 == 0:
            if not dry_run:
                conn.commit()
            log.info("  MLB: %d/%d processed (filled=%d missing=%d)",
                     idx + 1, len(games), filled, missing)

    if not dry_run:
        conn.commit()

    conn.close()
    log.info("MLB ESPN backfill done: filled=%d  missing=%d  dry_run=%s",
             filled, missing, dry_run)
    return {"sport": "MLB", "attempted": len(games), "filled": filled, "missing": missing}


# ══════════════════════════════════════════════════════════════════════════════
# NBA — nightly updater for ongoing season (2025-26 +)
# ══════════════════════════════════════════════════════════════════════════════

def backfill_nba_nightly(date_str: Optional[str] = None, dry_run: bool = False) -> dict:
    """
    Nightly odds update for NBA games.  Fetches ESPN odds for a single date
    (default: yesterday) and updates any rows in betting_lines that still have
    NULL close_spread.

    Intended to be called from a scheduler (e.g. scheduler.py) each morning
    after the previous night's games have settled.

    Parameters
    ----------
    date_str : YYYY-MM-DD  (default: yesterday UTC)
    dry_run  : if True, print without writing to DB
    """
    import datetime
    if date_str is None:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

    log.info("NBA nightly update: date=%s  dry_run=%s", date_str, dry_run)

    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    cur  = conn.cursor()

    cur.execute("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team
        FROM games g
        JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.sport      = 'NBA'
          AND g.game_date  = ?
          AND bl.close_spread IS NULL
        ORDER BY g.game_date
    """, (date_str,))
    games = cur.fetchall()
    log.info("NBA nightly: %d games on %s need odds", len(games), date_str)

    if not games:
        conn.close()
        return {"sport": "NBA", "date": date_str, "attempted": 0, "filled": 0}

    event_map = _build_date_event_map([date_str])
    log.info("Event map: %d entries for %s", len(event_map), date_str)

    filled = missing = 0
    for game_id, game_date, home_team, away_team in games:
        event_id = event_map.get((game_date, home_team))
        if not event_id:
            log.debug("No ESPN event for game_id=%s (%s %s @ %s)",
                      game_id, game_date, away_team, home_team)
            missing += 1
            continue

        time.sleep(_DELAY_ODDS)
        odds = _fetch_espn_odds(event_id)
        if not odds:
            log.debug("No ESPN odds for event_id=%s", event_id)
            missing += 1
            continue

        if dry_run:
            log.info("[DRY RUN] %s %s@%s → spread=%.1f total=%s ml=%s/%s",
                     game_date, away_team, home_team,
                     odds["spread"] or 0,
                     odds["over_under"], odds["ml_home"], odds["ml_away"])
        else:
            cur.execute("""
                UPDATE betting_lines
                SET close_spread  = ?,
                    close_total   = ?,
                    close_ml_home = ?,
                    close_ml_away = ?
                WHERE game_id = ?
            """, (odds["spread"], odds["over_under"],
                  odds["ml_home"], odds["ml_away"],
                  game_id))

        filled += 1

    if not dry_run:
        conn.commit()

    conn.close()
    log.info("NBA nightly done: date=%s  filled=%d  missing=%d",
             date_str, filled, missing)
    return {"sport": "NBA", "date": date_str, "attempted": len(games),
            "filled": filled, "missing": missing}


# ══════════════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify(sport: Optional[str] = None) -> None:
    """Print coverage stats for betting_lines after backfill."""
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    cur  = conn.cursor()

    sports = [sport] if sport else ["NBA", "MLB", "NFL"]
    for s in sports:
        cur.execute("""
            SELECT
                COUNT(*)                                              AS total,
                SUM(CASE WHEN bl.close_spread IS NOT NULL THEN 1 ELSE 0 END) AS has_spread,
                SUM(CASE WHEN bl.close_total  IS NOT NULL THEN 1 ELSE 0 END) AS has_total,
                SUM(CASE WHEN bl.close_ml_home IS NOT NULL THEN 1 ELSE 0 END) AS has_ml
            FROM betting_lines bl
            JOIN games g ON bl.game_id = g.game_id
            WHERE g.sport = ?
        """, (s,))
        row = cur.fetchone()
        total, spread, total_l, ml = row
        pct_s = f"{spread/total*100:.1f}%" if total else "—"
        pct_t = f"{total_l/total*100:.1f}%" if total else "—"
        pct_m = f"{ml/total*100:.1f}%" if total else "—"
        log.info("%s: %d games | spread=%d (%s) | total=%d (%s) | ml=%d (%s)",
                 s, total, spread, pct_s, total_l, pct_t, ml, pct_m)

    # ATS label coverage (what features.py will see)
    for s in sports:
        cur.execute("""
            SELECT COUNT(*)
            FROM games g
            JOIN betting_lines bl ON g.game_id = bl.game_id
            WHERE g.sport        = ?
              AND g.home_score   IS NOT NULL
              AND g.away_score   IS NOT NULL
              AND bl.close_spread IS NOT NULL
        """, (s,))
        ats_rows = cur.fetchone()[0]
        log.info("%s: %d ATS-labelable rows (score + spread both present)", s, ats_rows)

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# NHL — ESPN moneyline-based puck-line backfill
# ══════════════════════════════════════════════════════════════════════════════

ESPN_CORE_NHL = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl"
ESPN_SITE_NHL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"

# ESPN → our DB abbreviation normalization for NHL teams
_ESPN_NHL_ABBREV_MAP: dict[str, str] = {
    "TB":  "TBL",
    "SJ":  "SJS",
    "NJ":  "NJD",
    "LA":  "LAK",
    "CLB": "CBJ",
    "VGS": "VGK",
    "ARI": "ARI",   # legacy Arizona
    "UTAH": "UTA",
}


def _build_nhl_date_event_map(game_dates: list[str]) -> dict[tuple[str, str], str]:
    """
    For each date, fetch ESPN NHL scoreboard and return:
        {(date_str, home_abbrev): espn_event_id}

    Tries both raw ESPN abbreviation and the normalized DB abbreviation.
    """
    event_map: dict[tuple[str, str], str] = {}
    for i, date_str in enumerate(game_dates):
        date_esc = date_str.replace("-", "")
        data = _espn_get(f"{ESPN_SITE_NHL}/scoreboard", params={"dates": date_esc, "limit": 20})
        if not data:
            time.sleep(_DELAY_SCOREBOARD)
            continue

        for event in data.get("events", []):
            eid  = event.get("id")
            comp = (event.get("competitions") or [{}])[0]
            home = next((c for c in comp.get("competitors", [])
                         if c.get("homeAway") == "home"), {})
            abbr_raw = (home.get("team") or {}).get("abbreviation", "")
            if eid and abbr_raw:
                # Store both ESPN abbrev and our mapped abbrev for broader matching
                event_map[(date_str, abbr_raw)]  = eid
                mapped = _ESPN_NHL_ABBREV_MAP.get(abbr_raw, abbr_raw)
                event_map[(date_str, mapped)] = eid

        if (i + 1) % 25 == 0:
            log.info("  NHL scoreboard: %d/%d dates, %d entries", i + 1, len(game_dates), len(event_map))
        time.sleep(_DELAY_SCOREBOARD)

    return event_map


def _fetch_nhl_puckline_odds(event_id: str) -> Optional[dict]:
    """
    Fetch puck-line moneyline odds for an NHL event from ESPN Core API.
    Same structure as MLB odds endpoint — odds inline in items[0].

    Returns dict: {home_is_favorite, close_spread, ml_home, ml_away, over_under}
    or None if unavailable.
    """
    url = f"{ESPN_CORE_NHL}/events/{event_id}/competitions/{event_id}/odds"
    list_data = _espn_get(url)
    if not list_data or not list_data.get("items"):
        return None

    odds_data = list_data["items"][0]
    home_odds = odds_data.get("homeTeamOdds", {})
    away_odds = odds_data.get("awayTeamOdds", {})

    home_is_fav  = bool(home_odds.get("favorite", False))
    close_spread = -1.5 if home_is_fav else 1.5

    ml_home = (
        _parse_american(home_odds.get("close", {}).get("moneyLine", {}).get("american"))
        or _parse_american(home_odds.get("moneyLine"))
    )
    ml_away = (
        _parse_american(away_odds.get("close", {}).get("moneyLine", {}).get("american"))
        or _parse_american(away_odds.get("moneyLine"))
    )
    close_sect = odds_data.get("close", {})
    over_under = (
        _parse_american(close_sect.get("total", {}).get("alternateDisplayValue"))
        or odds_data.get("overUnder")
    )

    return {
        "home_is_favorite": int(home_is_fav),
        "close_spread":     close_spread,
        "over_under":       float(over_under) if over_under is not None else None,
        "ml_home":          float(ml_home)    if ml_home    is not None else None,
        "ml_away":          float(ml_away)    if ml_away    is not None else None,
    }


def backfill_nhl_espn(dry_run: bool = False) -> dict:
    """
    Backfill NHL puck-line data using ESPN Core API.

    - Fetches moneylines to determine which team is the ML favorite.
    - Sets close_spread = -1.5 (home favored) or +1.5 (away favored).
    - Stores home_is_favorite, close_ml_home, close_ml_away, close_total.
    - Fallback: synthetic close_spread = -1.5 (home default) when ESPN has no odds.
    """
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    cur  = conn.cursor()

    cur.execute("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team
        FROM games g
        JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.sport        = 'NHL'
          AND g.status       = 'Final'
          AND bl.close_spread IS NULL
        ORDER BY g.game_date
    """)
    games = cur.fetchall()
    log.info("NHL ESPN: %d games need puck-line backfill", len(games))

    if not games:
        conn.close()
        return {"sport": "NHL", "attempted": 0, "filled": 0, "missing": 0}

    unique_dates = sorted(set(row[1] for row in games))
    log.info("Fetching NHL scoreboard for %d unique dates…", len(unique_dates))
    event_map = _build_nhl_date_event_map(unique_dates)
    log.info("NHL event map: %d entries", len(event_map))

    filled = missing = 0
    for idx, (game_id, game_date, home_team, away_team) in enumerate(games):
        event_id = event_map.get((game_date, home_team))
        if not event_id:
            # Apply synthetic home -1.5 fallback
            if not dry_run:
                cur.execute("""
                    UPDATE betting_lines SET close_spread = -1.5 WHERE game_id = ?
                """, (game_id,))
            log.debug("No ESPN event for game_id=%s (%s %s @ %s) — synthetic -1.5",
                      game_id, game_date, away_team, home_team)
            missing += 1
            continue

        time.sleep(_DELAY_ODDS)
        odds = _fetch_nhl_puckline_odds(event_id)
        if not odds:
            if not dry_run:
                cur.execute("""
                    UPDATE betting_lines SET close_spread = -1.5 WHERE game_id = ?
                """, (game_id,))
            log.debug("No ESPN odds for event_id=%s — synthetic -1.5", event_id)
            missing += 1
            continue

        if dry_run:
            log.info("[DRY RUN] %s %s %s@%s → fav=%s spread=%.1f ml=%s/%s total=%s",
                     game_id, game_date, away_team, home_team,
                     "HOME" if odds["home_is_favorite"] else "AWAY",
                     odds["close_spread"],
                     odds["ml_home"], odds["ml_away"], odds["over_under"])
        else:
            cur.execute("""
                UPDATE betting_lines
                SET close_spread      = ?,
                    close_total       = ?,
                    close_ml_home     = ?,
                    close_ml_away     = ?,
                    home_is_favorite  = ?
                WHERE game_id = ?
            """, (odds["close_spread"], odds["over_under"],
                  odds["ml_home"], odds["ml_away"],
                  odds["home_is_favorite"],
                  game_id))

        filled += 1

        if (idx + 1) % 100 == 0:
            if not dry_run:
                conn.commit()
            log.info("  NHL: %d/%d processed (filled=%d missing=%d)",
                     idx + 1, len(games), filled, missing)

    if not dry_run:
        conn.commit()

    conn.close()
    log.info("NHL ESPN backfill done: filled=%d  missing=%d  dry_run=%s",
             filled, missing, dry_run)
    return {"sport": "NHL", "attempted": len(games), "filled": filled, "missing": missing}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill betting lines into historical.db")
    p.add_argument("--sport", choices=["NBA", "MLB", "NHL"], default=None,
                   help="Sport to backfill (default: all three)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without touching the DB")
    p.add_argument("--nightly", action="store_true",
                   help="NBA nightly mode: update yesterday's games only")
    p.add_argument("--date", default=None,
                   help="YYYY-MM-DD date for --nightly mode (default: yesterday)")
    p.add_argument("--seasons", nargs="+", default=None, metavar="SEASON",
                   help="NBA seasons to backfill, e.g. --seasons 2022-23 2023-24 (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.nightly:
        log.info("━━━  NBA nightly update  ━━━")
        backfill_nba_nightly(date_str=args.date, dry_run=args.dry_run)
        verify(sport="NBA")
    else:
        sports = [args.sport] if args.sport else ["MLB", "NHL", "NBA"]

        results = []
        for sport in sports:
            log.info("━━━  %s  ━━━", sport)
            if sport == "NBA":
                r = backfill_nba_espn(dry_run=args.dry_run, seasons=args.seasons)
            elif sport == "MLB":
                r = backfill_mlb_espn(dry_run=args.dry_run)
            elif sport == "NHL":
                r = backfill_nhl_espn(dry_run=args.dry_run)
            else:
                continue
            results.append(r)

        log.info("━━━  Verification  ━━━")
        verify()
