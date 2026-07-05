"""
injury_fetcher.py — Game-day injury data integration (Phase B)

Data sources (all free):
  • ESPN injuries endpoint: site.api.espn.com/apis/site/v2/sports/{sport}/injuries
  • MLB Stats API probable pitchers: statsapi.mlb.com/api/v1/schedule?hydrate=probablePitcher
  • NHL injury reports via ESPN NHL endpoint

Refresh schedule (called by cron in main.py):
  7:00 AM CT  — morning sweep
  1:00 PM CT  — midday update (catches late scratches)
  4:00 PM CT  — pre-evening game update
  Per-game: 60 min before commence_time (targeted update)

Impact scores (initial calibration, refine from outcomes over time):
  NBA: superstar=0.35, starter=0.15, rotation=0.05
  NFL: starting QB=0.50, starting skill=0.10, other=0.03
  MLB: starting pitcher=0.40, top hitter=0.08, regular=0.03
  NHL: starting goalie=0.30, top forward=0.10
"""
from __future__ import annotations

import logging
import requests
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── ESPN sport path segments ──────────────────────────────────────────────────
_ESPN_SPORT_PATHS = {
    "NBA": "basketball/nba",
    "MLB": "baseball/mlb",
    "NHL": "hockey/nhl",
    "NFL": "americanfootball/nfl",
}

_ESPN_INJURIES_BASE = "https://site.api.espn.com/apis/site/v2/sports/{sport_path}/injuries"
_MLB_SCHEDULE_URL   = "https://statsapi.mlb.com/api/v1/schedule"

# ── Position-based impact scores ──────────────────────────────────────────────
_IMPACT_SCORES: dict[str, dict[str, float]] = {
    "NBA": {
        "PG": 0.20, "SG": 0.20, "SF": 0.20, "PF": 0.15, "C": 0.15,
        # All-Star / superstar override applied by name-based logic
        "DEFAULT": 0.10,
    },
    "MLB": {
        "SP": 0.40,  # Starting pitcher — single most important injury signal
        "RP": 0.05,
        "C": 0.08, "1B": 0.06, "2B": 0.06, "3B": 0.06, "SS": 0.07,
        "LF": 0.05, "CF": 0.07, "RF": 0.05,
        "DH": 0.05,
        "DEFAULT": 0.03,
    },
    "NHL": {
        "G": 0.30,   # Goalie — largest impact in any position-sport combo
        "D": 0.08,
        "LW": 0.10, "RW": 0.10, "C": 0.10,
        "DEFAULT": 0.05,
    },
    "NFL": {
        "QB": 0.50,  # Starting QB — most impactful single player in team sports
        "RB": 0.08, "WR": 0.08, "TE": 0.06,
        "OL": 0.04, "DL": 0.04, "LB": 0.05, "CB": 0.06, "S": 0.05,
        "DEFAULT": 0.03,
    },
}

_ACTIVE_STATUSES = frozenset({"OUT", "DOUBTFUL", "QUESTIONABLE", "DAY_TO_DAY", "PROBABLE"})
_SIGNIFICANT_STATUSES = frozenset({"OUT", "DOUBTFUL"})

_REQUEST_TIMEOUT = 8  # seconds


def _impact_for(sport: str, position: str | None) -> float:
    """Return base impact score for a player given sport + position."""
    table = _IMPACT_SCORES.get(sport, {})
    if not position:
        return table.get("DEFAULT", 0.05)
    pos = position.upper().strip()
    return table.get(pos, table.get("DEFAULT", 0.05))


def _fetch_espn_injuries(sport: str) -> list[dict]:
    """Fetch injury reports from ESPN API for a given sport.

    Returns list of dicts with keys:
        player_name, player_id, team, position, status, injury_type
    """
    path = _ESPN_SPORT_PATHS.get(sport)
    if not path:
        return []

    url = _ESPN_INJURIES_BASE.format(sport_path=path)
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("[injury] ESPN fetch failed for %s: %s", sport, exc)
        return []

    injuries: list[dict] = []
    # ESPN structure: {"injuries": [{"team": {...}, "injuries": [...]}]}
    _logged_structure = False
    for team_block in data.get("injuries", []):
        team_raw = team_block.get("team") or {}
        if isinstance(team_raw, str):
            team_name = team_raw
        else:
            team_name = (
                team_raw.get("displayName")
                or team_raw.get("name")
                or team_raw.get("shortDisplayName")
                or team_raw.get("abbreviation")
                or ""
            )
        # Log the raw team block once so we can diagnose parsing failures
        if not team_name and not _logged_structure:
            logger.warning("[injury] team name empty for %s — team_block keys: %s, team_raw: %s",
                           sport, list(team_block.keys()), team_raw)
            _logged_structure = True
        for inj in team_block.get("injuries", []):
            athlete   = inj.get("athlete") or {}
            pos_block = athlete.get("position") or {}
            status    = inj.get("status", "")
            injuries.append({
                "player_name":  athlete.get("displayName", ""),
                "player_id":    str(athlete.get("id", "")),
                "team":         team_name,
                "position":     pos_block.get("abbreviation", ""),
                "status":       status.upper(),
                "injury_type":  inj.get("longComment") or inj.get("shortComment") or "",
            })

    logger.info("[injury] ESPN %s: fetched %d injury records", sport, len(injuries))
    return injuries


def _fetch_mlb_probable_pitchers(game_date: str) -> list[dict]:
    """Fetch MLB probable starters from MLB Stats API.

    game_date: YYYY-MM-DD
    Returns list of {team, player_name, player_id, position, status, game_id}
    """
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "probablePitcher,team",
    }
    try:
        resp = requests.get(_MLB_SCHEDULE_URL, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("[injury] MLB probable pitchers fetch failed: %s", exc)
        return []

    pitchers: list[dict] = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            game_id = str(game.get("gamePk", ""))
            for side in ("home", "away"):
                team_data   = game.get(f"{side}Team", {}) or {}
                team_name   = (team_data.get("team") or {}).get("name", "")
                probable    = team_data.get("probablePitcher") or {}
                if not probable:
                    continue
                pitchers.append({
                    "player_name": probable.get("fullName", ""),
                    "player_id":   str(probable.get("id", "")),
                    "team":        team_name,
                    "position":    "SP",
                    "status":      "PROBABLE",  # confirmed starter
                    "injury_type": "",
                    "game_id":     game_id,
                })

    logger.info("[injury] MLB probable pitchers: %d starters for %s", len(pitchers), game_date)
    return pitchers


def refresh_injuries(db, game_date: Optional[str] = None) -> dict:
    """
    Main entry point — fetch and store all injury data for today's games.

    Called by:
      • Cron at 7 AM, 1 PM, 4 PM CT
      • Pre-game trigger 60 min before commence_time

    Returns summary dict with counts per sport.
    """
    from database import InjuryReport

    if game_date is None:
        import pytz
        ct = pytz.timezone("America/Chicago")
        game_date = datetime.now(ct).strftime("%Y-%m-%d")

    now_utc = datetime.now(timezone.utc)
    summary: dict[str, int] = {}

    for sport in ("NBA", "MLB", "NHL", "NFL"):
        records: list[dict] = []

        # ESPN injuries (all sports)
        espn_injuries = _fetch_espn_injuries(sport)
        records.extend(espn_injuries)

        # MLB: also fetch probable pitchers from MLB Stats API (more reliable than ESPN)
        if sport == "MLB":
            prob_pitchers = _fetch_mlb_probable_pitchers(game_date)
            # Deduplicate by player_id — prefer ESPN injury record if available
            espn_ids = {r["player_id"] for r in espn_injuries}
            for p in prob_pitchers:
                if p["player_id"] not in espn_ids:
                    records.append(p)

        saved = 0
        for rec in records:
            if not rec.get("player_name"):
                continue
            status = rec.get("status", "").upper()
            # Only store active/reportable statuses
            if status not in _ACTIVE_STATUSES and status != "PROBABLE":
                continue

            impact = _impact_for(sport, rec.get("position"))
            try:
                inj = InjuryReport(
                    sport        = sport,
                    team         = rec.get("team"),
                    player_name  = rec.get("player_name"),
                    player_id    = rec.get("player_id"),
                    position     = rec.get("position"),
                    status       = status,
                    injury_type  = rec.get("injury_type"),
                    reported_at  = now_utc,
                    game_date    = game_date,
                    impact_score = impact,
                    source       = "espn" if not rec.get("game_id") else "mlb_stats_api",
                )
                db.add(inj)
                saved += 1
            except Exception as exc:
                logger.warning("[injury] Failed to store record: %s", exc)

        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error("[injury] DB commit failed for %s: %s", sport, exc)

        summary[sport] = saved
        logger.info("[injury] %s: saved %d records for %s", sport, saved, game_date)

    return {"game_date": game_date, "records_saved": summary}


def get_team_injury_impact(db, team: str, sport: str, game_date: str) -> float:
    """
    Compute cumulative injury impact for a team on a given game date.

    Returns a value in [0, 1] representing total team strength reduction.
    Multiple injuries compound: impact = 1 - product(1 - each_impact).
    Capped at 0.70 (a team can't be more than 70% weakened by injuries).
    """
    from database import InjuryReport

    significant = (
        db.query(InjuryReport)
        .filter(
            InjuryReport.team == team,
            InjuryReport.sport == sport,
            InjuryReport.game_date == game_date,
            InjuryReport.status.in_(["OUT", "DOUBTFUL"]),
        )
        .all()
    )
    if not significant:
        return 0.0

    # Compound: total_impact = 1 - product(1 - individual_impacts)
    survival = 1.0
    for inj in significant:
        impact = inj.impact_score or 0.0
        survival *= (1.0 - impact)

    total_impact = round(min(1.0 - survival, 0.70), 4)
    return total_impact


def attach_injury_signals(leg: dict, db, game_date: str) -> dict:
    """
    Attach injury impact signals to a leg dict for composite scoring.

    Sets:
        leg["home_team_injury_impact"]  — home team strength reduction [0, 1]
        leg["away_team_injury_impact"]  — away team strength reduction [0, 1]
        leg["injury_net_advantage"]     — signed: positive = home advantaged by injuries
    """
    home = leg.get("home_team") or ""
    away = leg.get("away_team") or ""
    sport = leg.get("sport") or ""

    if not home or not away or not sport:
        return leg

    try:
        home_impact = get_team_injury_impact(db, home, sport, game_date)
        away_impact = get_team_injury_impact(db, away, sport, game_date)
        leg["home_team_injury_impact"] = home_impact
        leg["away_team_injury_impact"] = away_impact
        # net_advantage > 0 means home team benefits (away is more injured)
        leg["injury_net_advantage"] = round(away_impact - home_impact, 4)
    except Exception as exc:
        logger.debug("[injury] attach_injury_signals failed: %s", exc)

    return leg
