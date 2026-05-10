"""
scout/mlb_team_markets.py — MLB team market projections.

Markets: h2h (moneyline), spread (run line ±1.5), totals.

MLB specifics:
  - Run line is almost always ±1.5 (not variable like NBA spread)
  - Totals are 7.5–11.5 range
  - Win probability uses log5 with home boost of +4% (larger than NBA)
  - Expected runs derived from team wRC+ / ERA proxied by wins/losses
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from scout.base import GameInfo, ScoutedProp, PROJECTION_VERSION
from scout.projection_engine import (
    clamp_probability,
    compute_ci_95,
    grade_from_probability,
    project_spread_cover,
    project_total_over,
    project_win_probability,
)
import scout.data_sources as ds

# ── Constants ──────────────────────────────────────────────────────────────────

_HOME_ADVANTAGE = 0.040   # MLB home field ~4pp
_MARGIN_STD     = 3.2     # run-differential std dev
_TOTAL_STD      = 3.0     # combined runs std dev
_DEFAULT_RPG    = 4.4     # league-average runs per game

_TOTAL_LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
_RUN_LINE    = 1.5        # MLB run line is always ±1.5

_MIN_EMIT_PROB = 0.45


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _win_pct(stats: dict) -> float:
    w = _safe_float(stats.get("wins"))
    l = _safe_float(stats.get("losses"))
    if w + l == 0:
        return 0.500
    return w / (w + l)


def _runs_per_game(stats: dict) -> float:
    rpg = _safe_float(stats.get("runs_per_game") or stats.get("avg_runs"))
    return rpg if rpg > 1 else _DEFAULT_RPG


def _best_total_threshold(expected_total: float) -> tuple[float, float]:
    best_t, best_prob, best_edge = _TOTAL_LINES[0], 0.5, 0.0
    for t in _TOTAL_LINES:
        p    = project_total_over(expected_total, _TOTAL_STD, t)
        edge = abs(p - 0.50)
        if edge > best_edge:
            best_edge, best_prob, best_t = edge, p, t
    return best_t, best_prob


# ── Market builders ────────────────────────────────────────────────────────────

def _build_h2h(
    game: GameInfo,
    home_wp: float,
    away_wp: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    home_prob = project_win_probability(
        home_wp, away_wp,
        home_advantage=_HOME_ADVANTAGE,
        is_home=True,
    )
    away_prob = clamp_probability(1.0 - home_prob)
    results = []

    for team, prob, side in [
        (game.home_team, home_prob, "home"),
        (game.away_team, away_prob, "away"),
    ]:
        if prob < _MIN_EMIT_PROB:
            continue
        grade = grade_from_probability(prob)
        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "MLB",
            game_id           = game.game_id,
            home_team         = game.home_team,
            away_team         = game.away_team,
            commence_time     = game.commence_time,
            market_type       = "h2h",
            player_name       = None,
            player_id         = None,
            team              = team,
            side              = side,
            threshold         = None,
            projected_value   = round(prob, 4),
            projected_low_95  = round(max(0.0, prob - 0.10), 4),
            projected_high_95 = round(min(1.0, prob + 0.10), 4),
            projected_std_dev = 0.10,
            hit_probability   = round(prob, 4),
            quality_grade     = grade,
            confidence_factors= confidence_factors,
            risk_factors      = risk_factors,
            data_source       = "espn",
            projection_version= PROJECTION_VERSION,
        ))

    return results


def _build_run_line(
    game: GameInfo,
    expected_margin: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    """MLB run line is always ±1.5."""
    # Home covering -1.5 (favored) or +1.5 (underdog)
    home_cover_prob = project_spread_cover(expected_margin, _MARGIN_STD, -_RUN_LINE)
    away_cover_prob = clamp_probability(1.0 - home_cover_prob)

    results = []
    for team, prob, side, line in [
        (game.home_team, home_cover_prob, "home", -_RUN_LINE),
        (game.away_team, away_cover_prob, "away", +_RUN_LINE),
    ]:
        if prob < _MIN_EMIT_PROB:
            continue
        grade = grade_from_probability(prob)
        lo, hi = compute_ci_95(expected_margin, _MARGIN_STD)
        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "MLB",
            game_id           = game.game_id,
            home_team         = game.home_team,
            away_team         = game.away_team,
            commence_time     = game.commence_time,
            market_type       = "spread",
            player_name       = None,
            player_id         = None,
            team              = team,
            side              = side,
            threshold         = line,
            projected_value   = round(expected_margin, 2),
            projected_low_95  = round(lo, 2),
            projected_high_95 = round(hi, 2),
            projected_std_dev = _MARGIN_STD,
            hit_probability   = round(prob, 4),
            quality_grade     = grade,
            confidence_factors= confidence_factors,
            risk_factors      = risk_factors,
            data_source       = "espn",
            projection_version= PROJECTION_VERSION,
        ))

    return results


def _build_total(
    game: GameInfo,
    expected_total: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    total_line, hit_prob = _best_total_threshold(expected_total)

    side = "over"
    if hit_prob < 0.50:
        hit_prob = clamp_probability(1.0 - hit_prob)
        side     = "under"

    if hit_prob < _MIN_EMIT_PROB:
        return []

    grade  = grade_from_probability(hit_prob)
    lo, hi = compute_ci_95(expected_total, _TOTAL_STD)

    return [ScoutedProp(
        scout_date        = scout_date,
        sport             = "MLB",
        game_id           = game.game_id,
        home_team         = game.home_team,
        away_team         = game.away_team,
        commence_time     = game.commence_time,
        market_type       = "totals",
        player_name       = None,
        player_id         = None,
        team              = None,
        side              = side,
        threshold         = total_line,
        projected_value   = round(expected_total, 2),
        projected_low_95  = round(lo, 2),
        projected_high_95 = round(hi, 2),
        projected_std_dev = _TOTAL_STD,
        hit_probability   = round(hit_prob, 4),
        quality_grade     = grade,
        confidence_factors= confidence_factors,
        risk_factors      = risk_factors,
        data_source       = "espn",
        projection_version= PROJECTION_VERSION,
    )]


# ── Game-level entry point ─────────────────────────────────────────────────────

def scout_game(game: GameInfo) -> List[ScoutedProp]:
    """Generate MLB team market ScoutedProps for one game."""
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    home_stats = ds.get_mlb_team_hitting_stats(game.home_team_id)
    away_stats = ds.get_mlb_team_hitting_stats(game.away_team_id)

    confidence_factors = []
    risk_factors       = []

    if not home_stats:
        risk_factors.append(f"No stats for {game.home_team}")
        home_wp, home_rpg = 0.500, _DEFAULT_RPG
    else:
        home_wp  = _win_pct(home_stats)
        home_rpg = _runs_per_game(home_stats)
        confidence_factors.append(f"{game.home_team} W%: {home_wp:.3f}, RPG: {home_rpg:.1f}")

    if not away_stats:
        risk_factors.append(f"No stats for {game.away_team}")
        away_wp, away_rpg = 0.500, _DEFAULT_RPG
    else:
        away_wp  = _win_pct(away_stats)
        away_rpg = _runs_per_game(away_stats)
        confidence_factors.append(f"{game.away_team} W%: {away_wp:.3f}, RPG: {away_rpg:.1f}")

    # Expected margin (positive = home wins)
    expected_margin = (home_rpg - away_rpg) + 0.25   # tiny home advantage in runs
    expected_total  = home_rpg + away_rpg

    results.extend(_build_h2h(game, home_wp, away_wp, scout_date, confidence_factors, risk_factors))
    results.extend(_build_run_line(game, expected_margin, scout_date, confidence_factors, risk_factors))
    results.extend(_build_total(game, expected_total, scout_date, confidence_factors, risk_factors))

    return results
