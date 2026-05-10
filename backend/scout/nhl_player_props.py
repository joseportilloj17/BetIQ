"""
scout/nhl_player_props.py — NHL player prop projections.

Markets: player_shots, player_points (goals+assists),
         player_goals, player_assists.

Data: NHL Web API (api-web.nhle.com) via data_sources.py.

Methodology:
  - Season stats from NHL API (/skater-stats-leaders or /player/{id}/landing)
  - Recent form from last 5-7 games
  - Home/road note
  - Normal CDF projection vs. common lines
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from scout.base import GameInfo, ScoutedProp, PROJECTION_VERSION
from scout.projection_engine import (
    blend_season_recent,
    clamp_probability,
    compute_ci_95,
    grade_from_probability,
    project_over_under,
    std_from_average,
    validate_projection,
)
import scout.data_sources as ds

# ── Market definitions ─────────────────────────────────────────────────────────

_STAT_MAP = {
    # market_type          season_key      recent_key   sport_stat
    "player_shots":      ("shots_pg",     "shots",     "shots"),
    "player_points":     ("points_pg",    "points",    "points"),
    "player_goals":      ("goals_pg",     "goals",     "goals"),
    "player_assists":    ("assists_pg",   "assists",   "assists"),
}

_COMMON_THRESHOLDS: dict[str, list[float]] = {
    "player_shots":   [1.5, 2.5, 3.5, 4.5, 5.5],
    "player_points":  [0.5, 1.5, 2.5],
    "player_goals":   [0.5, 1.5],
    "player_assists": [0.5, 1.5],
}

_MIN_AVG: dict[str, float] = {
    "player_shots":   1.0,
    "player_points":  0.20,
    "player_goals":   0.10,
    "player_assists": 0.10,
}

_MIN_EMIT_PROB = 0.45

# Positions that take shots (exclude goalies, scratch forwards)
_SKATER_POSITIONS = {"C", "LW", "RW", "D", "F", "W"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _season_per_game(season_stats: dict, games_key: str = "gamesPlayed") -> dict:
    """Convert cumulative NHL season stats to per-game rates."""
    gp = max(1, int(_safe_float(season_stats.get(games_key, 1))))
    return {
        "shots_pg":   _safe_float(season_stats.get("shots",   0)) / gp,
        "points_pg":  _safe_float(season_stats.get("points",  0)) / gp,
        "goals_pg":   _safe_float(season_stats.get("goals",   0)) / gp,
        "assists_pg": _safe_float(season_stats.get("assists", 0)) / gp,
        "gp":         gp,
    }


def _recent_avg(recent_games: list[dict], stat_key: str, n: int = 7) -> Optional[float]:
    vals = []
    for g in recent_games[:n]:
        v = _safe_float(g.get(stat_key))
        if v >= 0:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pick_best_threshold(
    market: str,
    projected_mean: float,
    std: float,
) -> tuple[float, float]:
    lines = _COMMON_THRESHOLDS.get(market, [projected_mean])
    best_t    = lines[0]
    best_prob = project_over_under(projected_mean, std, best_t)
    best_edge = abs(best_prob - 0.50)
    for t in lines[1:]:
        p    = project_over_under(projected_mean, std, t)
        edge = abs(p - 0.50)
        if edge > best_edge:
            best_edge, best_prob, best_t = edge, p, t
    return best_t, best_prob


# ── Per-player projection ──────────────────────────────────────────────────────

def _scout_player(
    player: dict,
    game: GameInfo,
    team: str,
    scout_date: str,
) -> List[ScoutedProp]:
    results: List[ScoutedProp] = []

    # NHL API returns player IDs as integers
    player_id   = str(player.get("id", player.get("playerId", "")))
    player_name = (player.get("fullName")
                   or player.get("firstName", {}).get("default", "")
                   + " " + player.get("lastName", {}).get("default", "")).strip()
    if not player_name:
        player_name = "Unknown"

    pos = str(player.get("positionCode", player.get("position", ""))).upper()
    if pos == "G":
        return results   # skip goalies for skater props

    season_stats_raw = ds.get_nhl_player_season_stats(player_id)
    if not season_stats_raw:
        return results

    season_pg = _season_per_game(season_stats_raw)
    if season_pg.get("gp", 0) < 5:
        return results   # too few games

    recent_games = ds.get_nhl_player_recent_games(player_id, n=7)

    is_home   = (team == game.home_team)
    home_road = "home" if is_home else "away"

    for market_type, (season_key, recent_key, sport_stat) in _STAT_MAP.items():
        season_avg = _safe_float(season_pg.get(season_key))
        if season_avg < _MIN_AVG.get(market_type, 0.05):
            continue

        recent_avg_val = _recent_avg(recent_games, recent_key, n=7) if recent_games else None
        projected_mean = blend_season_recent(season_avg, recent_avg_val)
        std            = std_from_average(projected_mean, "NHL", sport_stat)

        confidence_factors = [
            f"Season ({season_pg.get('gp', '?')}GP): {season_avg:.2f}/g",
            home_road,
        ]
        risk_factors = []
        if recent_avg_val is not None:
            confidence_factors.append(f"L7 avg: {recent_avg_val:.2f}")
        else:
            risk_factors.append("No recent game log")

        risk_factors = validate_projection(projected_mean, season_avg, std, risk_factors)

        threshold, hit_prob = _pick_best_threshold(market_type, projected_mean, std)
        if hit_prob < _MIN_EMIT_PROB:
            continue

        grade = grade_from_probability(hit_prob)
        side  = "over" if hit_prob >= 0.50 else "under"
        if side == "under":
            hit_prob = clamp_probability(1.0 - hit_prob)

        lo95, hi95 = compute_ci_95(projected_mean, std)

        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "NHL",
            game_id           = game.game_id,
            home_team         = game.home_team,
            away_team         = game.away_team,
            commence_time     = game.commence_time,
            market_type       = market_type,
            player_name       = player_name,
            player_id         = player_id,
            team              = team,
            side              = side,
            threshold         = threshold,
            projected_value   = round(projected_mean, 3),
            projected_low_95  = round(lo95, 3),
            projected_high_95 = round(hi95, 3),
            projected_std_dev = round(std, 3),
            hit_probability   = round(hit_prob, 4),
            quality_grade     = grade,
            confidence_factors= confidence_factors,
            risk_factors      = risk_factors,
            data_source       = "nhl_api",
            projection_version= PROJECTION_VERSION,
        ))

    return results


# ── Game-level entry point ─────────────────────────────────────────────────────

def scout_game(game: GameInfo) -> List[ScoutedProp]:
    """Generate all NHL player prop ScoutedProps for one game."""
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    for team_id, team_name in [
        (game.home_team_id, game.home_team),
        (game.away_team_id, game.away_team),
    ]:
        roster = ds.get_nhl_team_roster(team_id)
        if not roster:
            continue

        for player in roster:
            try:
                props = _scout_player(player, game, team_name, scout_date)
                results.extend(props)
            except Exception as exc:
                name = (player.get("fullName")
                        or player.get("firstName", {}).get("default", "?"))
                print(f"[nhl_player_props] skip {name}: {exc}")

    return results
