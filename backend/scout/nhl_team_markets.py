"""
scout/nhl_team_markets.py — NHL team market projections.

Markets: h2h (moneyline), spread (puck line ±1.5), totals.

NHL specifics:
  - Puck line is always ±1.5 (like MLB run line)
  - Totals typically 5.0–7.5
  - Home advantage ~3-4%
  - Win probability from standings (points%, not W-L-OT)
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

_HOME_ADVANTAGE  = 0.035
_MARGIN_STD      = 1.8     # NHL goal differential std (~1.8 goals)
_TOTAL_STD       = 1.5     # combined goals std
_DEFAULT_GF_PG   = 3.1     # league avg goals for per game
_PUCK_LINE       = 1.5

_TOTAL_LINES = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
_MIN_EMIT_PROB = 0.45


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _win_pct_from_standings(standings: dict, team_id: str) -> float:
    """
    Extract win% from NHL standings data.
    Prefers points% (pts / maxPts) which accounts for OT losses.
    """
    if not standings:
        return 0.500

    # Try direct lookup if standings is a flat dict for this team
    pct = _safe_float(standings.get("pointPct") or standings.get("win_pct"))
    if 0 < pct <= 1:
        return pct

    w = _safe_float(standings.get("wins"))
    gp = _safe_float(standings.get("gamesPlayed"))
    if gp > 0:
        return w / gp

    return 0.500


def _gf_per_game(standings: dict) -> float:
    gf = _safe_float(standings.get("goalFor") or standings.get("goals_for"))
    gp = _safe_float(standings.get("gamesPlayed", 1)) or 1
    val = gf / gp
    return val if val > 1 else _DEFAULT_GF_PG


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
            sport             = "NHL",
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
            data_source       = "nhl_api",
            projection_version= PROJECTION_VERSION,
        ))

    return results


def _build_puck_line(
    game: GameInfo,
    expected_margin: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    """Puck line ±1.5 for both sides."""
    home_cover = project_spread_cover(expected_margin, _MARGIN_STD, -_PUCK_LINE)
    away_cover = clamp_probability(1.0 - home_cover)
    results = []
    lo, hi = compute_ci_95(expected_margin, _MARGIN_STD)

    for team, prob, side, line in [
        (game.home_team, home_cover, "home", -_PUCK_LINE),
        (game.away_team, away_cover, "away", +_PUCK_LINE),
    ]:
        if prob < _MIN_EMIT_PROB:
            continue
        grade = grade_from_probability(prob)
        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "NHL",
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
            data_source       = "nhl_api",
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
        sport             = "NHL",
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
        data_source       = "nhl_api",
        projection_version= PROJECTION_VERSION,
    )]


# ── Game-level entry point ─────────────────────────────────────────────────────

def scout_game(game: GameInfo) -> List[ScoutedProp]:
    """Generate NHL team market ScoutedProps for one game."""
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    home_standings = ds.get_nhl_team_standings(game.home_team_id)
    away_standings = ds.get_nhl_team_standings(game.away_team_id)

    confidence_factors = []
    risk_factors       = []

    if not home_standings:
        risk_factors.append(f"No standings for {game.home_team}")
        home_wp, home_gf = 0.500, _DEFAULT_GF_PG
    else:
        home_wp  = _win_pct_from_standings(home_standings, game.home_team_id)
        home_gf  = _gf_per_game(home_standings)
        confidence_factors.append(f"{game.home_team} W%: {home_wp:.3f}, GF/G: {home_gf:.2f}")

    if not away_standings:
        risk_factors.append(f"No standings for {game.away_team}")
        away_wp, away_gf = 0.500, _DEFAULT_GF_PG
    else:
        away_wp  = _win_pct_from_standings(away_standings, game.away_team_id)
        away_gf  = _gf_per_game(away_standings)
        confidence_factors.append(f"{game.away_team} W%: {away_wp:.3f}, GF/G: {away_gf:.2f}")

    # Home court in goals: +0.1 goal boost at home
    expected_margin = (home_gf - away_gf) + 0.1
    expected_total  = home_gf + away_gf

    results.extend(_build_h2h(game, home_wp, away_wp, scout_date, confidence_factors, risk_factors))
    results.extend(_build_puck_line(game, expected_margin, scout_date, confidence_factors, risk_factors))
    results.extend(_build_total(game, expected_total, scout_date, confidence_factors, risk_factors))

    return results
