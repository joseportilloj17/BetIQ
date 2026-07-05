"""
player_projections.py — Free-data player projection pipeline.

Builds next-game projections for MLB, NBA, and NHL players using only
free public APIs. Designed to be ready when OddsAPI player props are enabled.

Architecture:
  1. Fetch player season stats + recent game logs (ESPN / MLB Stats / NHL API)
  2. Compute base projection = recency-weighted season average
  3. Apply matchup adjustment (opposing pitcher quality vs league avg for MLB;
     opposing team defensive rank for NBA/NHL)
  4. Apply venue adjustment (park factors for MLB; home/away splits)
  5. Store in player_projections table

When OddsAPI player props are enabled (future):
  - Compare projection to FanDuel/DraftKings line
  - Compute edge = P(projection_dist exceeds line) - implied_prob_from_odds
  - Recommend prop bet if edge > threshold

Runs daily as part of the scout pipeline after game schedule is known.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Optional
import urllib.request
import urllib.error

from sqlalchemy import text as _text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_MLB_BASE  = "https://statsapi.mlb.com/api/v1"
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_NHL_BASE  = "https://api-web.nhle.com/v1"

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 12) -> Optional[dict]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BetIQ/1.0 (player projections; educational)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Projection fetch failed: %s — %s", url, exc)
        return None


# ─── MLB projection builder ──────────────────────────────────────────────────

# MLB park factors — single source of truth in features.py
from features import MLB_PARK_FACTORS as _MLB_PARK_FACTORS
_DEFAULT_PARK_FACTOR = 1.00


def _mlb_player_stats(player_id: int) -> Optional[dict]:
    """Fetch MLB player season hitting stats from MLB Stats API (free)."""
    url = (
        f"{_MLB_BASE}/people/{player_id}/stats"
        f"?stats=season&group=hitting&season=2025"
    )
    data = _get(url)
    if not data:
        return None
    stats_list = (data.get("stats") or [{}])[0].get("splits") or []
    if not stats_list:
        return None
    return stats_list[0].get("stat", {})


def _mlb_recent_game_logs(player_id: int, n: int = 15) -> list[dict]:
    """Fetch last N games from MLB Stats API player game log."""
    url = (
        f"{_MLB_BASE}/people/{player_id}/stats"
        f"?stats=gameLog&group=hitting&season=2025&limit={n}"
    )
    data = _get(url)
    if not data:
        return []
    splits = (data.get("stats") or [{}])[0].get("splits") or []
    return [s.get("stat", {}) for s in splits[:n]]


def _mlb_pitcher_stats(pitcher_id: int) -> Optional[dict]:
    """Fetch starting pitcher season ERA/WHIP from MLB Stats API."""
    url = (
        f"{_MLB_BASE}/people/{pitcher_id}/stats"
        f"?stats=season&group=pitching&season=2025"
    )
    data = _get(url)
    if not data:
        return None
    splits = (data.get("stats") or [{}])[0].get("splits") or []
    if not splits:
        return None
    return splits[0].get("stat", {})


def _recency_weighted_avg(game_logs: list[dict], stat_key: str,
                           recent_n: int = 7, weights: tuple = (2.0, 1.0)) -> Optional[float]:
    """
    Compute recency-weighted average for a stat across game logs.
    recent_n most recent games get weight[0]; older games get weight[1].
    Returns None if no games have the stat.
    """
    values = []
    for g in game_logs:
        v = g.get(stat_key)
        if v is not None:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                pass
    if not values:
        return None
    recent = values[:recent_n]
    older  = values[recent_n:]
    total_w = len(recent) * weights[0] + len(older) * weights[1]
    if total_w == 0:
        return None
    wsum = sum(v * weights[0] for v in recent) + sum(v * weights[1] for v in older)
    return wsum / total_w


def build_mlb_projections(
    db: Session,
    game_date: str,
    games_today: list[dict],  # [{home_team, away_team, home_pitcher_id, away_pitcher_id}]
) -> list[dict]:
    """
    Build MLB player hitting projections for all players with active game today.

    For each team with a game:
      1. Fetch roster from MLB Stats API
      2. For each player in batting lineup positions:
         - Season stats + recent game log
         - Recency-weighted projection per stat
         - Matchup adjustment vs opposing starter ERA
         - Park factor adjustment

    Returns list of projection dicts ready for DB upsert.
    """
    league_avg_era = 4.20  # 2025 MLB average (update seasonally)

    projections: list[dict] = []

    for game in games_today:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        home_pitcher_id = game.get("home_pitcher_id")
        away_pitcher_id = game.get("away_pitcher_id")

        park_factor = _MLB_PARK_FACTORS.get(home_team, _DEFAULT_PARK_FACTOR)

        # Fetch pitcher stats for matchup adjustment
        home_pitcher_era = league_avg_era
        away_pitcher_era = league_avg_era
        if home_pitcher_id:
            ps = _mlb_pitcher_stats(int(home_pitcher_id))
            if ps:
                try:
                    home_pitcher_era = float(ps.get("era", league_avg_era))
                except (TypeError, ValueError):
                    pass
        if away_pitcher_id:
            ps = _mlb_pitcher_stats(int(away_pitcher_id))
            if ps:
                try:
                    away_pitcher_era = float(ps.get("era", league_avg_era))
                except (TypeError, ValueError):
                    pass

        # Matchup adjustment: better pitcher (lower ERA) → lower hitting projection
        # Formula: home batters face away pitcher (away_pitcher_era)
        # adj = 1 + (opponent_era - league_avg) / league_avg * 0.15
        def matchup_adj(pitcher_era: float) -> float:
            return 1.0 + (pitcher_era - league_avg_era) / league_avg_era * 0.15

        home_adj = matchup_adj(away_pitcher_era) * park_factor
        away_adj = matchup_adj(home_pitcher_era) * (1.0 / park_factor if park_factor > 0 else 1.0)

        # Fetch rosters + project each batter
        for team_name, opp_adj, is_home in [
            (home_team, home_adj, True),
            (away_team, away_adj, False),
        ]:
            roster_url = (
                f"{_MLB_BASE}/teams/roster"
                f"?rosterType=active&season=2025"
                f"&teamName={urllib.request.quote(team_name)}"
            )
            # Use team search to get team_id first
            search_url = f"{_MLB_BASE}/teams?sportId=1&season=2025"
            teams_data = _get(search_url)
            if not teams_data:
                continue

            team_id = None
            for t in (teams_data.get("teams") or []):
                if t.get("name", "").lower() == team_name.lower():
                    team_id = t.get("id")
                    break
            if not team_id:
                continue

            roster_data = _get(f"{_MLB_BASE}/teams/{team_id}/roster?rosterType=active&season=2025")
            if not roster_data:
                continue

            for entry in (roster_data.get("roster") or []):
                person = entry.get("person", {})
                player_id   = person.get("id")
                player_name = person.get("fullName", "")
                position    = (entry.get("position") or {}).get("abbreviation", "")

                # Only project hitters (skip pitchers in batting positions)
                if not player_id or position in ("P", "TWP"):
                    continue

                season_stats = _mlb_player_stats(int(player_id))
                if not season_stats:
                    continue

                games_played = int(season_stats.get("gamesPlayed") or 0)
                if games_played < 10:
                    continue  # insufficient sample

                game_logs = _mlb_recent_game_logs(int(player_id), n=15)

                # Per-game averages from season
                hits_pg       = _safe_ratio(season_stats.get("hits"),      games_played)
                tb_pg         = _safe_ratio(season_stats.get("totalBases"), games_played)
                hr_pg         = _safe_ratio(season_stats.get("homeRuns"),   games_played)
                rbi_pg        = _safe_ratio(season_stats.get("rbi"),        games_played)
                runs_pg       = _safe_ratio(season_stats.get("runs"),       games_played)
                strikeouts_pg = _safe_ratio(season_stats.get("strikeOuts"), games_played)

                # Recency-weighted from game logs
                hits_rw  = _recency_weighted_avg(game_logs, "hits")
                tb_rw    = _recency_weighted_avg(game_logs, "totalBases")
                hr_rw    = _recency_weighted_avg(game_logs, "homeRuns")

                # Blend season avg (60%) + recent (40%)
                def blend(season_v, recent_v):
                    if season_v is None:
                        return None
                    if recent_v is None:
                        return season_v
                    return 0.60 * season_v + 0.40 * recent_v

                for stat_type, base_proj in [
                    ("hits",        blend(hits_pg,       hits_rw)),
                    ("total_bases", blend(tb_pg,         tb_rw)),
                    ("home_runs",   blend(hr_pg,         hr_rw)),
                    ("rbi",         rbi_pg),
                    ("runs",        runs_pg),
                    ("strikeouts",  strikeouts_pg),
                ]:
                    if base_proj is None:
                        continue

                    adjusted = base_proj * opp_adj
                    confidence = _confidence(games_played, game_logs, stat_type)

                    projections.append({
                        "player_id":            str(player_id),
                        "player_name":          player_name,
                        "sport":                "MLB",
                        "team":                 team_name,
                        "opponent":             away_team if is_home else home_team,
                        "stat_type":            stat_type,
                        "projection":           round(adjusted, 4),
                        "confidence":           round(confidence, 3),
                        "matchup_adjustment":   round(opp_adj - 1.0, 4),
                        "venue_adjustment":     round(park_factor - 1.0 if is_home else (1.0 / park_factor) - 1.0, 4),
                        "injury_adjustment":    0.0,
                        "game_date":            game_date,
                        "projected_at":         datetime.utcnow().isoformat() + "Z",
                    })

    return projections


# ─── NBA projection builder ───────────────────────────────────────────────────

def _nba_player_season_stats(team_id: str) -> list[dict]:
    """Fetch NBA team roster + per-player season averages via ESPN."""
    url = f"{_ESPN_BASE}/basketball/nba/teams/{team_id}/roster"
    data = _get(url)
    if not data:
        return []
    athletes = data.get("athletes") or []
    result = []
    for a in athletes:
        stats = a.get("statistics") or {}
        # ESPN returns season averages in athlete stats
        result.append({
            "player_id":   a.get("id", ""),
            "player_name": a.get("displayName", ""),
            "stats":       stats,
        })
    return result


def build_nba_projections(
    db: Session,
    game_date: str,
    games_today: list[dict],
) -> list[dict]:
    """
    Build NBA player projections from ESPN team stats.
    Uses season averages as base; adjusts for opponent defensive rank.
    """
    projections: list[dict] = []
    # League averages for normalization
    league_avgs = {"points": 11.0, "rebounds": 4.2, "assists": 2.7, "steals": 0.8,
                   "blocks": 0.5, "threePointFieldGoalsMade": 1.1}

    for game in games_today:
        home_id = game.get("home_espn_id", "")
        away_id = game.get("away_espn_id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        if not home_id or not away_id:
            continue

        for team_id, team_name, opp_team in [
            (home_id, home_team, away_team),
            (away_id, away_team, home_team),
        ]:
            players = _nba_player_season_stats(team_id)
            for p in players:
                stats = p.get("stats") or {}
                # ESPN athlete stats keys vary; use common stat names
                pts   = _float(stats.get("points") or stats.get("avgPoints"))
                reb   = _float(stats.get("rebounds") or stats.get("avgRebounds"))
                ast   = _float(stats.get("assists") or stats.get("avgAssists"))
                stl   = _float(stats.get("steals") or stats.get("avgSteals"))
                blk   = _float(stats.get("blocks") or stats.get("avgBlocks"))
                threes = _float(stats.get("threePointFieldGoalsMade") or stats.get("avgThreePointFieldGoalsMade"))

                gp    = int(_float(stats.get("gamesPlayed")) or 0)
                if gp < 10 or pts is None:
                    continue

                for stat_type, val in [
                    ("points",   pts),
                    ("rebounds", reb),
                    ("assists",  ast),
                    ("steals",   stl),
                    ("blocks",   blk),
                    ("threes",   threes),
                ]:
                    if val is None:
                        continue
                    confidence = min(0.95, gp / 60.0)
                    projections.append({
                        "player_id":            p["player_id"],
                        "player_name":          p["player_name"],
                        "sport":                "NBA",
                        "team":                 team_name,
                        "opponent":             opp_team,
                        "stat_type":            stat_type,
                        "projection":           round(val, 3),
                        "confidence":           round(confidence, 3),
                        "matchup_adjustment":   0.0,
                        "venue_adjustment":     0.0,
                        "injury_adjustment":    0.0,
                        "game_date":            game_date,
                        "projected_at":         datetime.utcnow().isoformat() + "Z",
                    })

    return projections


# ─── NHL projection builder ───────────────────────────────────────────────────

def _nhl_player_season_stats(team_abbr: str) -> list[dict]:
    """Fetch NHL roster + season stats from NHL Stats API (free)."""
    url = f"{_NHL_BASE}/roster/{team_abbr}/current"
    data = _get(url)
    if not data:
        return []

    forwards   = data.get("forwards") or []
    defensemen = data.get("defensemen") or []
    players    = forwards + defensemen

    result = []
    for p in players:
        pid = p.get("id", "")
        name = f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}".strip()
        result.append({"player_id": str(pid), "player_name": name})

    return result


def build_nhl_projections(
    db: Session,
    game_date: str,
    games_today: list[dict],
) -> list[dict]:
    """
    Build NHL player projections (goals, assists, SOG) from NHL Stats API.
    """
    projections: list[dict] = []

    for game in games_today:
        home_abbr = game.get("home_abbr", "")
        away_abbr = game.get("away_abbr", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        for team_abbr, team_name, opp_team in [
            (home_abbr, home_team, away_team),
            (away_abbr, away_team, home_team),
        ]:
            if not team_abbr:
                continue

            # Fetch player landing page (includes season stats)
            for player in _nhl_player_season_stats(team_abbr):
                pid = player["player_id"]
                player_url = f"{_NHL_BASE}/player/{pid}/landing"
                p_data = _get(player_url)
                if not p_data:
                    continue

                season_totals = p_data.get("seasonTotals") or []
                # Find most recent NHL season
                nhl_season = None
                for s in reversed(season_totals):
                    if s.get("leagueAbbrev") == "NHL":
                        nhl_season = s
                        break
                if not nhl_season:
                    continue

                gp    = int(nhl_season.get("gamesPlayed") or 0)
                goals = _float(nhl_season.get("goals"))
                asst  = _float(nhl_season.get("assists"))
                sog   = _float(nhl_season.get("shots"))

                if gp < 10:
                    continue

                for stat_type, season_total in [
                    ("goals",            goals),
                    ("assists",          asst),
                    ("shots_on_goal",    sog),
                ]:
                    if season_total is None:
                        continue
                    per_game = season_total / gp
                    confidence = min(0.90, gp / 70.0)
                    projections.append({
                        "player_id":            pid,
                        "player_name":          player["player_name"],
                        "sport":                "NHL",
                        "team":                 team_name,
                        "opponent":             opp_team,
                        "stat_type":            stat_type,
                        "projection":           round(per_game, 4),
                        "confidence":           round(confidence, 3),
                        "matchup_adjustment":   0.0,
                        "venue_adjustment":     0.0,
                        "injury_adjustment":    0.0,
                        "game_date":            game_date,
                        "projected_at":         datetime.utcnow().isoformat() + "Z",
                    })

    return projections


# ─── DB upsert ────────────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO player_projections (
    player_id, player_name, sport, team, opponent,
    stat_type, projection, confidence,
    matchup_adjustment, venue_adjustment, injury_adjustment,
    game_date, projected_at
) VALUES (
    :player_id, :player_name, :sport, :team, :opponent,
    :stat_type, :projection, :confidence,
    :matchup_adjustment, :venue_adjustment, :injury_adjustment,
    :game_date, :projected_at
)
ON CONFLICT(player_id, stat_type, game_date) DO UPDATE SET
    projection          = excluded.projection,
    confidence          = excluded.confidence,
    matchup_adjustment  = excluded.matchup_adjustment,
    venue_adjustment    = excluded.venue_adjustment,
    injury_adjustment   = excluded.injury_adjustment,
    player_name         = excluded.player_name,
    team                = excluded.team,
    opponent            = excluded.opponent,
    projected_at        = excluded.projected_at;
"""


def upsert_projections(db: Session, projections: list[dict]) -> int:
    """
    Upsert projection rows into player_projections table.

    Batches inserts for efficiency and forces a WAL checkpoint after commit
    so data survives a server restart without waiting for auto-checkpoint.
    Returns count written.
    """
    if not projections:
        return 0
    written = 0
    # Batch in groups of 100 to keep transactions small
    for i in range(0, len(projections), 100):
        batch = projections[i:i + 100]
        for row in batch:
            try:
                db.execute(_text(_UPSERT_SQL), row)
                written += 1
            except Exception as exc:
                logger.warning("Projection upsert failed for %s/%s: %s",
                               row.get("player_name"), row.get("stat_type"), exc)
        db.commit()

    # Force WAL checkpoint so data survives a server restart
    try:
        db.execute(_text("PRAGMA wal_checkpoint(PASSIVE)"))
    except Exception:
        pass

    return written


# ─── Daily runner ─────────────────────────────────────────────────────────────

def run_daily_projections(db: Session, game_date: str | None = None) -> dict:
    """
    Main entry point: build projections for all sports with games today.

    Called daily from the scheduler after fixture_refresh has populated today's
    game schedule. Returns summary dict.
    """
    from datetime import timezone as _tz
    from zoneinfo import ZoneInfo

    _CT = ZoneInfo("America/Chicago")
    today = game_date or datetime.now(_CT).strftime("%Y-%m-%d")

    summary = {
        "game_date":         today,
        "mlb_projections":   0,
        "nba_projections":   0,
        "nhl_projections":   0,
        "total_projections": 0,
        "errors":            [],
    }

    # ── Fetch today's MLB games + pitcher data ─────────────────────────────────
    try:
        import urllib.request as _req
        mlb_url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={today}&hydrate=probablePitcher"
        )
        data = _get(mlb_url)
        mlb_games = []
        if data:
            for date_entry in (data.get("dates") or []):
                for game in (date_entry.get("games") or []):
                    teams = game.get("teams", {})
                    home_sp = (teams.get("home", {}).get("probablePitcher") or {}).get("id")
                    away_sp = (teams.get("away", {}).get("probablePitcher") or {}).get("id")
                    mlb_games.append({
                        "home_team":       (teams.get("home", {}).get("team") or {}).get("name", ""),
                        "away_team":       (teams.get("away", {}).get("team") or {}).get("name", ""),
                        "home_pitcher_id": home_sp,
                        "away_pitcher_id": away_sp,
                    })
        mlb_projs = build_mlb_projections(db, today, mlb_games)
        written = upsert_projections(db, mlb_projs)
        summary["mlb_projections"] = written
    except Exception as exc:
        summary["errors"].append(f"MLB: {exc}")
        logger.warning("MLB projection error: %s", exc)

    # ── Fetch today's NBA games ────────────────────────────────────────────────
    try:
        nba_url = f"{_ESPN_BASE}/basketball/nba/scoreboard?dates={today.replace('-','')}"
        data = _get(nba_url)
        nba_games = []
        if data:
            for event in (data.get("events") or []):
                comps = event.get("competitions") or [{}]
                comp = comps[0]
                home = away = {}
                for c in (comp.get("competitors") or []):
                    if c.get("homeAway") == "home":
                        home = c
                    else:
                        away = c
                nba_games.append({
                    "home_team":    (home.get("team") or {}).get("displayName", ""),
                    "away_team":    (away.get("team") or {}).get("displayName", ""),
                    "home_espn_id": (home.get("team") or {}).get("id", ""),
                    "away_espn_id": (away.get("team") or {}).get("id", ""),
                })
        nba_projs = build_nba_projections(db, today, nba_games)
        written = upsert_projections(db, nba_projs)
        summary["nba_projections"] = written
    except Exception as exc:
        summary["errors"].append(f"NBA: {exc}")
        logger.warning("NBA projection error: %s", exc)

    # ── Fetch today's NHL games ────────────────────────────────────────────────
    try:
        nhl_schedule_url = f"{_NHL_BASE}/schedule/{today}"
        data = _get(nhl_schedule_url)
        nhl_games = []
        if data:
            for game_week in (data.get("gameWeek") or []):
                for game in (game_week.get("games") or []):
                    home = game.get("homeTeam") or {}
                    away = game.get("awayTeam") or {}
                    nhl_games.append({
                        "home_team":  home.get("placeName", {}).get("default", "") + " " + home.get("commonName", {}).get("default", ""),
                        "away_team":  away.get("placeName", {}).get("default", "") + " " + away.get("commonName", {}).get("default", ""),
                        "home_abbr":  home.get("abbrev", ""),
                        "away_abbr":  away.get("abbrev", ""),
                    })
        nhl_projs = build_nhl_projections(db, today, nhl_games)
        written = upsert_projections(db, nhl_projs)
        summary["nhl_projections"] = written
    except Exception as exc:
        summary["errors"].append(f"NHL: {exc}")
        logger.warning("NHL projection error: %s", exc)

    summary["total_projections"] = (
        summary["mlb_projections"] +
        summary["nba_projections"] +
        summary["nhl_projections"]
    )

    logger.info(
        "[PlayerProjections] %s: total=%d (MLB=%d, NBA=%d, NHL=%d)",
        today,
        summary["total_projections"],
        summary["mlb_projections"],
        summary["nba_projections"],
        summary["nhl_projections"],
    )
    return summary


# ─── Lookup helper (for prop bet generation when OddsAPI props enabled) ───────

def get_projection(db: Session, player_name: str, stat_type: str,
                   game_date: str) -> Optional[dict]:
    """
    Look up stored projection for a player × stat × date.

    Returns {projection, confidence, matchup_adjustment, ...} or None.
    Used by prop bet recommender to compare projection against FanDuel line.
    """
    row = db.execute(
        _text("""
            SELECT player_id, player_name, sport, team, opponent,
                   stat_type, projection, confidence,
                   matchup_adjustment, venue_adjustment, injury_adjustment,
                   projected_at
            FROM player_projections
            WHERE LOWER(player_name) = LOWER(:name)
              AND stat_type = :stat_type
              AND game_date = :game_date
            LIMIT 1
        """),
        {"name": player_name, "stat_type": stat_type, "game_date": game_date},
    ).fetchone()

    if not row:
        return None
    keys = ["player_id", "player_name", "sport", "team", "opponent",
            "stat_type", "projection", "confidence",
            "matchup_adjustment", "venue_adjustment", "injury_adjustment",
            "projected_at"]
    return dict(zip(keys, row))


def edge_vs_line(projection: float, line: float, confidence: float,
                 variance_scale: float = 0.3) -> float:
    """
    Estimate P(stat >= line) from projection using a normal approximation.
    Used to compute edge vs sportsbook implied probability.

    Args:
        projection:    Model's point estimate (e.g., 1.42 hits)
        line:          Sportsbook over/under line (e.g., 1.5)
        confidence:    0-1 confidence weight (scales variance)
        variance_scale: Fractional std dev of projection (default 0.3 = 30% of mean)

    Returns:
        P(actual >= line), e.g. 0.42 for "42% chance of over"
    """
    if projection <= 0:
        return 0.0
    std = max(0.1, projection * variance_scale / max(0.1, confidence))
    # Standard normal CDF approximation
    z = (line - projection) / std
    return _norm_cdf(-z)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun approximation."""
    if z < -6:
        return 0.0
    if z > 6:
        return 1.0
    # Abramowitz & Stegun 26.2.17
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    p = 1.0 - phi * poly
    return p if z >= 0 else 1.0 - p


# ─── Utilities ────────────────────────────────────────────────────────────────

def _safe_ratio(numerator, denominator) -> Optional[float]:
    try:
        n = float(numerator)
        d = float(denominator)
        return n / d if d > 0 else None
    except (TypeError, ValueError):
        return None


def _float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _confidence(games_played: int, game_logs: list[dict], stat_type: str) -> float:
    """
    Estimate projection confidence from sample size and recent variance.
    Returns 0-1 float.
    """
    if games_played < 10:
        return 0.2
    base = min(0.95, games_played / 100.0)
    # Penalise high variance in recent games
    if game_logs:
        stat_key_map = {
            "hits": "hits", "total_bases": "totalBases",
            "home_runs": "homeRuns", "rbi": "rbi",
            "runs": "runs", "strikeouts": "strikeOuts",
        }
        sk = stat_key_map.get(stat_type, stat_type)
        vals = []
        for g in game_logs[:10]:
            v = _float(g.get(sk))
            if v is not None:
                vals.append(v)
        if len(vals) >= 5:
            mean = sum(vals) / len(vals)
            if mean > 0:
                cv = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / mean
                base = base * max(0.4, 1.0 - cv * 0.5)
    return round(base, 3)
