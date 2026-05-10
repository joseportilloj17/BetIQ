"""
scout/data_sources.py — ESPN, MLB Stats API, and NHL API wrappers for scout pipeline.

All functions return None / empty on failure (never raise).
Uses http_retry.get_with_retry for all external calls.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from http_retry import get_with_retry
from scout.base import GameInfo

log = logging.getLogger(__name__)

# ── ESPN base URLs (reuse backfill_lines constants) ───────────────────────────
ESPN_SITE_NBA  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_CORE_NBA  = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ESPN_SITE_MLB  = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
ESPN_CORE_MLB  = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
ESPN_SITE_NHL  = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
ESPN_CORE_NHL  = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl"
NHL_API        = "https://api-web.nhle.com/v1"
MLB_STATS_API  = "https://statsapi.mlb.com/api/v1"

_DELAY = 0.15   # seconds between ESPN requests


def _espn(url: str, params: dict | None = None) -> dict | None:
    r = get_with_retry(url, params=params, timeout=12, max_attempts=3, label="Scout-ESPN")
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _nhl(url: str) -> dict | None:
    r = get_with_retry(
        url, timeout=12, max_attempts=3, label="Scout-NHL",
        headers={"User-Agent": "BetIQ/1.0"},
    )
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _mlb(path: str, params: dict | None = None) -> dict | None:
    r = get_with_retry(f"{MLB_STATS_API}{path}", params=params,
                       timeout=12, max_attempts=3, label="Scout-MLB")
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _today() -> str:
    return date.today().isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# NBA
# ═══════════════════════════════════════════════════════════════════════════════

def get_nba_games_today(date_str: str | None = None) -> list[GameInfo]:
    """Return today's NBA games as GameInfo objects."""
    date_str = date_str or _today()
    date_esc = date_str.replace("-", "")
    data = _espn(f"{ESPN_SITE_NBA}/scoreboard", {"dates": date_esc, "limit": 20})
    if not data:
        return []
    games = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        if not home or not away:
            continue
        games.append(GameInfo(
            game_id       = event.get("id", ""),
            sport         = "NBA",
            home_team     = home.get("team", {}).get("displayName", ""),
            away_team     = away.get("team", {}).get("displayName", ""),
            home_team_id  = home.get("team", {}).get("id", ""),
            away_team_id  = away.get("team", {}).get("id", ""),
            commence_time = event.get("date", ""),
            venue         = (comp.get("venue") or {}).get("fullName"),
            extra         = {
                "home_abbr": home.get("team", {}).get("abbreviation", ""),
                "away_abbr": away.get("team", {}).get("abbreviation", ""),
                "status":    (event.get("status") or {}).get("type", {}).get("name", ""),
            }
        ))
    return games


def get_nba_team_roster(team_id: str) -> list[dict]:
    """Return list of {id, displayName, position} for a team."""
    data = _espn(f"{ESPN_SITE_NBA}/teams/{team_id}/roster")
    if not data:
        return []
    players = []
    for group in data.get("athletes", []):
        for a in (group if isinstance(group, list) else group.get("items", [])):
            players.append({
                "id":   a.get("id", ""),
                "name": a.get("displayName") or a.get("fullName", ""),
                "pos":  (a.get("position") or {}).get("abbreviation", ""),
            })
    return players


def get_nba_player_season_stats(athlete_id: str) -> dict:
    """
    Fetch season averages for an NBA player from ESPN Core API.
    Returns dict with keys: points, rebounds, assists, steals, blocks,
    threePointFieldGoalsMade, minutesPerGame.
    """
    url = f"{ESPN_CORE_NBA}/athletes/{athlete_id}/statistics/0"
    data = _espn(url)
    if not data:
        return {}
    stats = {}
    for cat in data.get("splits", {}).get("categories", []):
        for stat in cat.get("stats", []):
            name = stat.get("name", "")
            val  = stat.get("value")
            if val is not None:
                try:
                    stats[name] = float(val)
                except (TypeError, ValueError):
                    pass
    return stats


def get_nba_player_recent_games(athlete_id: str, n: int = 10) -> list[dict]:
    """
    Fetch last n NBA game logs for a player via ESPN Core API.
    Returns list of stat dicts.
    """
    url = f"{ESPN_CORE_NBA}/athletes/{athlete_id}/gamelog"
    data = _espn(url)
    if not data:
        return []
    rows = []
    events = data.get("events", {})
    items  = events.get("items", []) if isinstance(events, dict) else []
    for item in items[-n:]:
        stats = {}
        for cat in item.get("categories", []):
            for s in cat.get("stats", []):
                try:
                    stats[s["name"]] = float(s["value"])
                except Exception:
                    pass
        if stats:
            rows.append(stats)
    return rows[-n:]


def get_nba_team_season_stats(team_id: str) -> dict:
    """Return team season stats dict (offensive + defensive ratings, pace, record)."""
    data = _espn(f"{ESPN_SITE_NBA}/teams/{team_id}")
    if not data:
        return {}
    team = data.get("team", {})
    record = {}
    for r in (team.get("record", {}).get("items") or []):
        for s in r.get("stats", []):
            record[s.get("name", "")] = s.get("value")
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# MLB
# ═══════════════════════════════════════════════════════════════════════════════

def get_mlb_games_today(date_str: str | None = None) -> list[GameInfo]:
    """Return today's MLB games via statsapi."""
    date_str = date_str or _today()
    data = _mlb("/schedule", {"sportId": "1", "date": date_str, "hydrate": "probablePitcher,linescore"})
    if not data:
        return []
    games = []
    for date_obj in data.get("dates", []):
        for g in date_obj.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_pitcher = (home.get("probablePitcher") or {}).get("fullName") or ""
            away_pitcher = (away.get("probablePitcher") or {}).get("fullName") or ""
            home_pitcher_id = str((home.get("probablePitcher") or {}).get("id") or "")
            away_pitcher_id = str((away.get("probablePitcher") or {}).get("id") or "")
            games.append(GameInfo(
                game_id       = str(g.get("gamePk", "")),
                sport         = "MLB",
                home_team     = home.get("team", {}).get("name", ""),
                away_team     = away.get("team", {}).get("name", ""),
                home_team_id  = str(home.get("team", {}).get("id", "")),
                away_team_id  = str(away.get("team", {}).get("id", "")),
                commence_time = g.get("gameDate", ""),
                venue         = (g.get("venue") or {}).get("name"),
                extra         = {
                    "home_pitcher":    home_pitcher,
                    "away_pitcher":    away_pitcher,
                    "home_pitcher_id": home_pitcher_id,
                    "away_pitcher_id": away_pitcher_id,
                    "game_status":     g.get("status", {}).get("detailedState", ""),
                }
            ))
    return games


def get_mlb_player_hitting_stats(player_id: str) -> dict:
    """
    Return season + last 15 game hitting stats for an MLB batter.
    Keys: avg, hits_per_g, total_bases_per_g, rbi_per_g, hr_per_g, runs_per_g,
          games, recent_avg, recent_hits_per_g, etc.
    """
    data = _mlb(f"/people/{player_id}/stats", {
        "stats": "season,gameLog",
        "group": "hitting",
        "season": str(date.today().year),
        "limit": "15",
    })
    if not data:
        return {}
    result = {}
    for split_group in data.get("stats", []):
        stype = split_group.get("type", {}).get("displayName", "")
        splits = split_group.get("splits", [])
        if not splits:
            continue
        if stype == "season" or "Season" in stype:
            s = splits[0].get("stat", {})
            g = max(int(s.get("gamesPlayed") or 1), 1)
            result.update({
                "games":              g,
                "avg":                float(s.get("avg") or 0),
                "hits_per_g":         int(s.get("hits") or 0) / g,
                "total_bases_per_g":  int(s.get("totalBases") or 0) / g,
                "rbi_per_g":          int(s.get("rbi") or 0) / g,
                "hr_per_g":           int(s.get("homeRuns") or 0) / g,
                "runs_per_g":         int(s.get("runs") or 0) / g,
                "at_bats_per_g":      int(s.get("atBats") or 0) / g,
                "strikeouts_per_g":   int(s.get("strikeOuts") or 0) / g,
            })
        elif "Log" in stype or "game" in stype.lower():
            recent = splits[:10]
            if recent:
                ng = len(recent)
                result.update({
                    "recent_hits_per_g":        sum(int(x.get("stat", {}).get("hits") or 0) for x in recent) / ng,
                    "recent_total_bases_per_g":  sum(int(x.get("stat", {}).get("totalBases") or 0) for x in recent) / ng,
                    "recent_rbi_per_g":          sum(int(x.get("stat", {}).get("rbi") or 0) for x in recent) / ng,
                    "recent_hr_per_g":           sum(int(x.get("stat", {}).get("homeRuns") or 0) for x in recent) / ng,
                    "recent_runs_per_g":         sum(int(x.get("stat", {}).get("runs") or 0) for x in recent) / ng,
                    "recent_games":              ng,
                })
    return result


def get_mlb_player_pitching_stats(player_id: str) -> dict:
    """Return season + recent pitching stats for an MLB pitcher."""
    data = _mlb(f"/people/{player_id}/stats", {
        "stats": "season,gameLog",
        "group": "pitching",
        "season": str(date.today().year),
        "limit": "10",
    })
    if not data:
        return {}
    result = {}
    for split_group in data.get("stats", []):
        stype = split_group.get("type", {}).get("displayName", "")
        splits = split_group.get("splits", [])
        if not splits:
            continue
        if stype == "season" or "Season" in stype:
            s = splits[0].get("stat", {})
            gs = max(int(s.get("gamesStarted") or 1), 1)
            result.update({
                "era":                float(s.get("era") or 0),
                "games_started":      gs,
                "so_per_g":           int(s.get("strikeOuts") or 0) / gs,
                "hits_per_g":         int(s.get("hits") or 0) / gs,
                "er_per_g":           int(s.get("earnedRuns") or 0) / gs,
                "ip_per_g":           float(s.get("inningsPitched") or 0) / gs,
                "whip":               float(s.get("whip") or 0),
            })
        elif "Log" in stype or "game" in stype.lower():
            recent = splits[:5]
            if recent:
                ng = len(recent)
                result.update({
                    "recent_so_per_g":    sum(int(x.get("stat", {}).get("strikeOuts") or 0) for x in recent) / ng,
                    "recent_hits_per_g":  sum(int(x.get("stat", {}).get("hits") or 0) for x in recent) / ng,
                    "recent_er_per_g":    sum(int(x.get("stat", {}).get("earnedRuns") or 0) for x in recent) / ng,
                    "recent_ip_per_g":    sum(float(x.get("stat", {}).get("inningsPitched") or 0) for x in recent) / ng,
                    "recent_games":       ng,
                })
    return result


def get_mlb_team_roster(team_id: str) -> list[dict]:
    """Return active roster for an MLB team (full raw roster dicts)."""
    data = _mlb(f"/teams/{team_id}/roster", {"rosterType": "active"})
    if not data:
        return []
    return data.get("roster", [])


def get_mlb_player_recent_games(player_id: str, n: int = 7) -> list[dict]:
    """
    Return last n game-log entries for an MLB hitter via MLB Stats API.
    Each entry is a stat dict with hits, rbi, homeRuns, totalBases, etc.
    """
    try:
        import statsapi
        logs = statsapi.player_stat_data(
            int(player_id),
            type="gameLog",
            group="hitting",
        )
        splits = logs.get("stats", [{}])[0].get("splits", [])
        results = []
        for s in splits[-n:]:
            stat = s.get("stat", {})
            stat["date"] = s.get("date", "")
            results.append(stat)
        return results
    except Exception:
        return []


def get_mlb_team_hitting_stats(team_id: str) -> dict:
    """Return team season hitting stats."""
    data = _mlb(f"/teams/{team_id}/stats", {
        "stats": "season",
        "group": "hitting",
        "season": str(date.today().year),
    })
    if not data:
        return {}
    for sg in data.get("stats", []):
        splits = sg.get("splits", [])
        if splits:
            return splits[0].get("stat", {})
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# NHL
# ═══════════════════════════════════════════════════════════════════════════════

def get_nhl_games_today(date_str: str | None = None) -> list[GameInfo]:
    """Return today's NHL games via NHL Web API."""
    date_str = date_str or _today()
    data = _nhl(f"{NHL_API}/schedule/{date_str}")
    if not data:
        return []
    games = []
    for week in data.get("gameWeek", []):
        for g in week.get("games", []):
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            games.append(GameInfo(
                game_id       = str(g.get("id", "")),
                sport         = "NHL",
                home_team     = home.get("placeName", {}).get("default", "") + " " + home.get("commonName", {}).get("default", ""),
                away_team     = away.get("placeName", {}).get("default", "") + " " + away.get("commonName", {}).get("default", ""),
                home_team_id  = str(home.get("id", "")),
                away_team_id  = str(away.get("id", "")),
                commence_time = g.get("startTimeUTC", ""),
                venue         = g.get("venue", {}).get("default", ""),
                extra         = {
                    "home_abbr": home.get("abbrev", ""),
                    "away_abbr": away.get("abbrev", ""),
                    "game_type": g.get("gameType", ""),
                }
            ))
    return games


def get_nhl_team_roster(team_abbr: str, season: str = "20252026") -> list[dict]:
    """Return current roster for an NHL team."""
    data = _nhl(f"{NHL_API}/roster/{team_abbr}/current")
    if not data:
        return []
    players = []
    for group in ["forwards", "defensemen", "goalies"]:
        for p in data.get(group, []):
            players.append({
                "id":   str(p.get("id", "")),
                "name": f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}".strip(),
                "pos":  p.get("positionCode", ""),
            })
    return players


def get_nhl_player_recent_games(player_id: str, n: int = 10) -> list[dict]:
    """Return last n game logs for an NHL player."""
    data = _nhl(f"{NHL_API}/player/{player_id}/game-log/now")
    if not data:
        return []
    logs = data.get("gameLog", [])[:n]
    return [
        {
            "goals":   int(g.get("goals") or 0),
            "assists": int(g.get("assists") or 0),
            "points":  int(g.get("points") or 0),
            "shots":   int(g.get("shots") or 0),
            "toi":     g.get("toi", ""),
            "pim":     int(g.get("pim") or 0),
        }
        for g in logs
    ]


def get_nhl_player_season_stats(player_id: str) -> dict:
    """Return season stats for an NHL player."""
    data = _nhl(f"{NHL_API}/player/{player_id}/landing")
    if not data:
        return {}
    season_totals = data.get("seasonTotals", [])
    # Find most recent NHL regular season
    for s in reversed(season_totals):
        if s.get("leagueAbbrev") == "NHL" and s.get("gameTypeId") == 2:
            gp = max(int(s.get("gamesPlayed") or 1), 1)
            return {
                "games":          gp,
                "goals_per_g":    int(s.get("goals") or 0) / gp,
                "assists_per_g":  int(s.get("assists") or 0) / gp,
                "points_per_g":   int(s.get("points") or 0) / gp,
                "shots_per_g":    int(s.get("shots") or 0) / gp,
                "pim_per_g":      int(s.get("pim") or 0) / gp,
                "toi_avg":        s.get("avgToi", ""),
            }
    return {}


def get_nhl_team_standings(team_abbr: str) -> dict:
    """Return current season record/pts% for an NHL team."""
    data = _nhl(f"{NHL_API}/standings/now")
    if not data:
        return {}
    for team in data.get("standings", []):
        if team.get("teamAbbrev", {}).get("default", "") == team_abbr:
            gp = max(int(team.get("gamesPlayed") or 1), 1)
            wins = int(team.get("wins") or 0)
            return {
                "win_pct":    wins / gp,
                "gp":         gp,
                "wins":       wins,
                "losses":     int(team.get("losses") or 0),
                "otl":        int(team.get("otLosses") or 0),
                "pts":        int(team.get("points") or 0),
                "goals_for_per_g":     float(team.get("goalFor") or 0) / gp,
                "goals_against_per_g": float(team.get("goalAgainst") or 0) / gp,
            }
    return {}
