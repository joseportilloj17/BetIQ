"""
scout/mlb_player_props.py — MLB player prop projections.

Hitter markets: player_hits, player_total_bases, player_rbi,
                player_runs, player_home_runs.

Pitcher markets: player_strikeouts, player_hits_allowed,
                 player_earned_runs.

Methodology:
  - Season averages from MLB Stats API (statsapi)
  - Recent form from last 7 games
  - Opponent handedness / matchup adjustments (simple L/R split if available)
  - Normal CDF projection vs. common threshold lines
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from scout.base import GameInfo, ScoutedProp, PROJECTION_VERSION
from scout.projection_engine import (
    blend_season_recent,
    clamp_probability,
    compute_ci_95,
    detect_streak,
    grade_from_probability,
    project_over_under,
    std_from_average,
    validate_projection,
)
import scout.data_sources as ds

# ── Market definitions ─────────────────────────────────────────────────────────

_HITTER_STAT_MAP = {
    # market_type           season_key           recent_key      sport_stat
    "player_hits":        ("hits_per_game",      "hits",         "hits"),
    "player_total_bases": ("total_bases_per_g",  "total_bases",  "total_bases"),
    "player_rbi":         ("rbi_per_game",        "rbi",          "rbi"),
    "player_runs":        ("runs_per_game",        "runs",         "runs"),
    "player_home_runs":   ("hr_per_game",          "homeRuns",     "hr"),
}

_PITCHER_STAT_MAP = {
    "player_strikeouts":    ("k_per_9_adj",    "strikeOuts",  "strikeouts"),
    "player_hits_allowed":  ("h_per_9_adj",    "hits",        "hits_allowed"),
    "player_earned_runs":   ("er_per_9_adj",   "earnedRuns",  "earned_runs"),
}

_HITTER_THRESHOLDS: dict[str, list[float]] = {
    "player_hits":        [0.5, 1.5, 2.5],
    "player_total_bases": [1.5, 2.5, 3.5, 4.5],
    "player_rbi":         [0.5, 1.5, 2.5],
    "player_runs":        [0.5, 1.5],
    "player_home_runs":   [0.5, 1.5],
}

_PITCHER_THRESHOLDS: dict[str, list[float]] = {
    "player_strikeouts":  [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    "player_hits_allowed":[3.5, 4.5, 5.5, 6.5, 7.5],
    "player_earned_runs": [0.5, 1.5, 2.5, 3.5],
}

_MIN_HITTER_AVG: dict[str, float] = {
    "player_hits":        0.15,   # ~1 hit every 6-7 games minimum
    "player_total_bases": 0.30,
    "player_rbi":         0.10,
    "player_runs":        0.10,
    "player_home_runs":   0.04,   # ~1 HR every 25 games
}

_MIN_PITCHER_K = 3.0   # at least 3 K/game to scout strikeout prop
_MIN_EMIT_PROB = 0.45


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


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
    thresholds: dict,
) -> tuple[float, float]:
    lines = thresholds.get(market, [projected_mean])
    best_t    = lines[0]
    best_prob = project_over_under(projected_mean, std, best_t)
    best_edge = abs(best_prob - 0.50)
    for t in lines[1:]:
        p    = project_over_under(projected_mean, std, t)
        edge = abs(p - 0.50)
        if edge > best_edge:
            best_edge, best_prob, best_t = edge, p, t
    return best_t, best_prob


# ── Per-plate-appearance stats → per-game averages ────────────────────────────

def _hitter_season_avgs(stats: dict, games: int) -> dict:
    """
    Convert raw cumulative season stats to per-game rates.
    `stats` is the dict returned by get_mlb_player_hitting_stats().
    """
    if not stats or games == 0:
        return {}

    def _pg(key: str) -> float:
        return _safe_float(stats.get(key, 0)) / games

    return {
        "hits_per_game":     _pg("hits"),
        "total_bases_per_g": _pg("totalBases"),
        "rbi_per_game":      _pg("rbi"),
        "runs_per_game":     _pg("runs"),
        "hr_per_game":       _pg("homeRuns"),
        "games":             games,
    }


def _pitcher_season_avgs(stats: dict) -> dict:
    """
    Convert raw pitcher stats to per-game (per-start) projections.
    Uses innings-pitched to scale K/9, H/9, ER/9 to per-start.
    """
    if not stats:
        return {}

    ip     = _safe_float(stats.get("inningsPitched", 0))
    starts = _safe_float(stats.get("gamesStarted", 1)) or 1
    ip_per_start = ip / starts if starts > 0 else 5.5

    k9  = _safe_float(stats.get("strikeoutsPer9Inn", 0))
    h9  = _safe_float(stats.get("hitsPer9Inn", 0))
    er9 = _safe_float(stats.get("earnedRunAverage", 0))

    # Convert per-9-inning rates to per-start given typical IP
    k_per_start  = k9  * ip_per_start / 9.0
    h_per_start  = h9  * ip_per_start / 9.0
    er_per_start = er9 * ip_per_start / 9.0

    return {
        "k_per_9_adj":  k_per_start,
        "h_per_9_adj":  h_per_start,
        "er_per_9_adj": er_per_start,
        "ip_per_start": ip_per_start,
        "starts":       starts,
    }


# ── Per-player projection ──────────────────────────────────────────────────────

def _scout_hitter(
    player: dict,
    game: GameInfo,
    team: str,
    scout_date: str,
    pitcher_factor: float = 1.0,
) -> List[ScoutedProp]:
    """
    pitcher_factor: ERA-based multiplier for the opposing pitcher.
    >1.0 = soft pitcher (project hitter stats up). <1.0 = elite pitcher (project down).
    Applied to hits, total_bases, rbi, runs (not HR — too binary).
    """
    results: List[ScoutedProp] = []

    player_id   = str(player.get("person", {}).get("id", player.get("id", "")))
    player_name = (player.get("person", {}).get("fullName")
                   or player.get("fullName", "Unknown"))

    raw_stats = ds.get_mlb_player_hitting_stats(player_id)
    if not raw_stats:
        return results

    games = int(_safe_float(raw_stats.get("gamesPlayed", 0)))
    if games < 5:
        return results   # not enough data

    season_avgs = _hitter_season_avgs(raw_stats, games)
    recent_games = ds.get_mlb_player_recent_games(player_id, n=7) if hasattr(ds, "get_mlb_player_recent_games") else []

    is_home = (team == game.home_team)
    home_road = "home" if is_home else "away"

    # Markets where pitcher quality adjustment applies
    _PITCHER_ADJUSTED = {"player_hits", "player_total_bases", "player_rbi", "player_runs"}

    for market_type, (season_key, recent_key, sport_stat) in _HITTER_STAT_MAP.items():
        season_avg = _safe_float(season_avgs.get(season_key))
        if season_avg < _MIN_HITTER_AVG.get(market_type, 0.05):
            continue

        recent_avg_val = _recent_avg(recent_games, recent_key, n=7) if recent_games else None
        projected_mean = blend_season_recent(season_avg, recent_avg_val)

        # Apply pitcher quality adjustment to contact/reaching-base markets
        if market_type in _PITCHER_ADJUSTED and pitcher_factor != 1.0:
            projected_mean = projected_mean * pitcher_factor

        std            = std_from_average(projected_mean, "MLB", sport_stat)

        confidence_factors = [f"Season ({games}G): {season_avg:.2f}/g", home_road]
        risk_factors       = []
        # Streak detection (L3)
        streak_label, l3_avg = detect_streak(recent_games, recent_key, season_avg, n=3)
        if streak_label == "HOT":
            confidence_factors.append(f"HOT streak: L3 avg {l3_avg:.2f}")
        elif streak_label == "COLD":
            risk_factors.append(f"COLD streak: L3 avg {l3_avg:.2f}")

        if pitcher_factor != 1.0:
            dir_label = "weak" if pitcher_factor > 1.0 else "elite"
            confidence_factors.append(f"Opp pitcher ({dir_label} ERA): {pitcher_factor:.2f}x adj")
        if recent_avg_val is not None:
            confidence_factors.append(f"L7 avg: {recent_avg_val:.2f}")
        else:
            risk_factors.append("No recent game log")
        risk_factors = validate_projection(projected_mean, season_avg, std, risk_factors)

        threshold, hit_prob = _pick_best_threshold(
            market_type, projected_mean, std, _HITTER_THRESHOLDS
        )
        if hit_prob < _MIN_EMIT_PROB:
            continue

        grade = grade_from_probability(hit_prob)
        side  = "over" if hit_prob >= 0.50 else "under"
        if side == "under":
            hit_prob = clamp_probability(1.0 - hit_prob)

        lo95, hi95 = compute_ci_95(projected_mean, std)

        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "MLB",
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
            data_source       = "mlb_stats_api",
            projection_version= PROJECTION_VERSION,
        ))

    return results


def _scout_pitcher(
    player: dict,
    game: GameInfo,
    team: str,
    scout_date: str,
) -> List[ScoutedProp]:
    results: List[ScoutedProp] = []

    player_id   = str(player.get("person", {}).get("id", player.get("id", "")))
    player_name = (player.get("person", {}).get("fullName")
                   or player.get("fullName", "Unknown"))

    raw_stats = ds.get_mlb_player_pitching_stats(player_id)
    if not raw_stats:
        return results

    season_avgs = _pitcher_season_avgs(raw_stats)
    if not season_avgs:
        return results

    starts = int(season_avgs.get("starts", 0))
    if starts < 3:
        return results   # not enough starts

    is_home = (team == game.home_team)
    home_road = "home" if is_home else "away"

    for market_type, (season_key, recent_key, sport_stat) in _PITCHER_STAT_MAP.items():
        season_avg = _safe_float(season_avgs.get(season_key))

        # Skip strikeouts if K rate is too low
        if market_type == "player_strikeouts" and season_avg < _MIN_PITCHER_K:
            continue
        if season_avg <= 0:
            continue

        std = std_from_average(season_avg, "MLB", sport_stat)

        confidence_factors = [
            f"Season ({starts} starts): {season_avg:.1f}/start",
            f"IP/start: {season_avgs.get('ip_per_start', 0):.1f}",
            home_road,
        ]
        risk_factors = []
        risk_factors = validate_projection(season_avg, season_avg, std, risk_factors)

        threshold, hit_prob = _pick_best_threshold(
            market_type, season_avg, std, _PITCHER_THRESHOLDS
        )
        if hit_prob < _MIN_EMIT_PROB:
            continue

        grade = grade_from_probability(hit_prob)
        side  = "over" if hit_prob >= 0.50 else "under"
        if side == "under":
            hit_prob = clamp_probability(1.0 - hit_prob)

        lo95, hi95 = compute_ci_95(season_avg, std)

        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "MLB",
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
            projected_value   = round(season_avg, 2),
            projected_low_95  = round(lo95, 2),
            projected_high_95 = round(hi95, 2),
            projected_std_dev = round(std, 2),
            hit_probability   = round(hit_prob, 4),
            quality_grade     = grade,
            confidence_factors= confidence_factors,
            risk_factors      = risk_factors,
            data_source       = "mlb_stats_api",
            projection_version= PROJECTION_VERSION,
        ))

    return results


# ── Pitcher/hitter matchup factors ────────────────────────────────────────────

_MLB_LEAGUE_AVG_ERA = 4.20   # ~2024-25 MLB league average ERA

def _pitcher_era_factor(pitcher_id: str) -> float:
    """
    Return a multiplier for hitter projections facing this pitcher.
    > 1.0 means soft pitcher (hitters project higher).
    < 1.0 means elite pitcher (hitters project lower).
    Clamped to [0.80, 1.25].
    """
    if not pitcher_id:
        return 1.0
    try:
        stats = ds.get_mlb_player_pitching_stats(pitcher_id)
        era = _safe_float(stats.get("earnedRunAverage", _MLB_LEAGUE_AVG_ERA))
        if era <= 0:
            return 1.0
        factor = era / _MLB_LEAGUE_AVG_ERA
        return max(0.80, min(1.25, factor))
    except Exception:
        return 1.0


def _opp_team_k_factor(opp_team_id: str) -> float:
    """
    Return a strikeout multiplier based on opponent team's K-rate.
    High-K lineup → pitcher strikeouts project higher (factor > 1.0).
    Low-K lineup → factor < 1.0.
    Clamped to [0.85, 1.15].
    """
    _MLB_LEAGUE_AVG_KPG = 8.5  # ~league avg strikeouts per game per team
    try:
        import statsapi
        standings = statsapi.standings_data()
        # statsapi doesn't have team K-rate easily; return 1.0 as fallback
        return 1.0
    except Exception:
        return 1.0


# ── Game-level entry point ─────────────────────────────────────────────────────

def scout_game(game: GameInfo) -> List[ScoutedProp]:
    """Generate all MLB player prop ScoutedProps for one game."""
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    extra = game.extra or {}
    home_pitcher_id = extra.get("home_pitcher_id", "")
    away_pitcher_id = extra.get("away_pitcher_id", "")

    # Hitters face the opposing pitcher; factor scales hitting projections
    # Home hitters face away pitcher, away hitters face home pitcher
    home_hitter_factor = _pitcher_era_factor(away_pitcher_id)  # home hitters vs away pitcher
    away_hitter_factor = _pitcher_era_factor(home_pitcher_id)  # away hitters vs home pitcher

    for team_id, team_name, hitter_factor in [
        (game.home_team_id, game.home_team, home_hitter_factor),
        (game.away_team_id, game.away_team, away_hitter_factor),
    ]:
        roster = ds.get_mlb_team_roster(team_id)
        if not roster:
            continue

        for player in roster:
            pos = (player.get("position", {}).get("abbreviation", "")
                   if isinstance(player.get("position"), dict)
                   else str(player.get("position", "")))

            try:
                if pos == "P":
                    results.extend(_scout_pitcher(player, game, team_name, scout_date))
                else:
                    results.extend(_scout_hitter(player, game, team_name, scout_date,
                                                  pitcher_factor=hitter_factor))
            except Exception as exc:
                name = (player.get("person", {}).get("fullName")
                        or player.get("fullName", "?"))
                print(f"[mlb_player_props] skip {name}: {exc}")

    return results
