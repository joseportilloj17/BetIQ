"""
scout/nba_team_markets.py — NBA team market projections.

Markets: h2h (moneyline), spread, totals.

Methodology:
  - Win probability: Log5 formula with home-field adjustment (+3.5pp)
  - Spread: Expected margin from win pcts and team offensive/defensive pace
  - Total: Sum of projected team scores using offensive pace ratings
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

# NBA game-to-game score std dev (used for total and spread projections)
_TEAM_SCORE_STD   = 11.0   # single-team pts std
_MARGIN_STD       = 13.5   # point-differential std
_TOTAL_STD        = 15.0   # combined score std

# Typical NBA common total lines (used to find best edge)
_TOTAL_THRESHOLDS = [
    205.5, 208.5, 210.5, 212.5, 215.5, 218.5, 220.5, 222.5,
    225.5, 228.5, 230.5, 233.5, 235.5, 238.5, 240.5,
]

# Typical spread lines scanned for edge
_SPREAD_LINES = [
    -12.5, -10.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5,
    -2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
]

_MIN_EMIT_PROB = 0.45


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _win_pct_from_stats(stats: dict) -> float:
    """Derive season win % from team stats dict. Falls back to 0.50."""
    w = _safe_float(stats.get("wins"))
    l = _safe_float(stats.get("losses"))
    if w + l == 0:
        return 0.50
    return w / (w + l)


def _expected_team_score(stats: dict, default: float = 112.0) -> float:
    """Pull offensive PPG from team stats."""
    ppg = _safe_float(stats.get("avg_points"))
    return ppg if ppg > 80 else default


def _best_total_threshold(expected_total: float) -> tuple[float, float]:
    """Find the total line with the greatest edge vs expected total."""
    from scout.projection_engine import project_total_over
    best_t    = _TOTAL_THRESHOLDS[0]
    best_prob = project_total_over(expected_total, _TOTAL_STD, best_t)
    best_edge = abs(best_prob - 0.50)
    for t in _TOTAL_THRESHOLDS[1:]:
        p    = project_total_over(expected_total, _TOTAL_STD, t)
        edge = abs(p - 0.50)
        if edge > best_edge:
            best_edge, best_prob, best_t = edge, p, t
    return best_t, best_prob


def _best_spread_line(expected_margin: float) -> tuple[float, float]:
    """Find the spread line (for home team) with the greatest edge."""
    best_s    = _SPREAD_LINES[0]
    best_prob = project_spread_cover(expected_margin, _MARGIN_STD, best_s)
    best_edge = abs(best_prob - 0.50)
    for s in _SPREAD_LINES[1:]:
        p    = project_spread_cover(expected_margin, _MARGIN_STD, s)
        edge = abs(p - 0.50)
        if edge > best_edge:
            best_edge, best_prob, best_s = edge, p, s
    return best_s, best_prob


# ── Market builders ────────────────────────────────────────────────────────────

def _build_h2h(
    game: GameInfo,
    home_wp: float,
    away_wp: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    """Moneyline props for both sides."""
    results = []

    home_prob = project_win_probability(home_wp, away_wp, is_home=True)
    away_prob = clamp_probability(1.0 - home_prob)

    for team, prob, side in [
        (game.home_team, home_prob, "home"),
        (game.away_team, away_prob, "away"),
    ]:
        if prob < _MIN_EMIT_PROB:
            continue
        grade = grade_from_probability(prob)
        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "NBA",
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


def _build_spread(
    game: GameInfo,
    expected_margin: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    """Best-edge spread prop for the home team."""
    spread_line, hit_prob = _best_spread_line(expected_margin)

    side = "home"
    if hit_prob < 0.50:
        # Home underdog against this spread — flip to away cover perspective
        hit_prob   = clamp_probability(1.0 - hit_prob)
        side       = "away"
        spread_line = -spread_line

    if hit_prob < _MIN_EMIT_PROB:
        return []

    grade = grade_from_probability(hit_prob)
    lo, hi = compute_ci_95(expected_margin, _MARGIN_STD)

    return [ScoutedProp(
        scout_date        = scout_date,
        sport             = "NBA",
        game_id           = game.game_id,
        home_team         = game.home_team,
        away_team         = game.away_team,
        commence_time     = game.commence_time,
        market_type       = "spread",
        player_name       = None,
        player_id         = None,
        team              = game.home_team if side == "home" else game.away_team,
        side              = side,
        threshold         = spread_line,
        projected_value   = round(expected_margin, 2),
        projected_low_95  = round(lo, 2),
        projected_high_95 = round(hi, 2),
        projected_std_dev = _MARGIN_STD,
        hit_probability   = round(hit_prob, 4),
        quality_grade     = grade,
        confidence_factors= confidence_factors,
        risk_factors      = risk_factors,
        data_source       = "espn",
        projection_version= PROJECTION_VERSION,
    )]


def _build_total(
    game: GameInfo,
    expected_total: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    """Over/under total prop."""
    total_line, hit_prob = _best_total_threshold(expected_total)

    side = "over"
    if hit_prob < 0.50:
        hit_prob = clamp_probability(1.0 - hit_prob)
        side     = "under"

    if hit_prob < _MIN_EMIT_PROB:
        return []

    grade = grade_from_probability(hit_prob)
    lo, hi = compute_ci_95(expected_total, _TOTAL_STD)

    return [ScoutedProp(
        scout_date        = scout_date,
        sport             = "NBA",
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
    """Generate NBA team market ScoutedProps (h2h, spread, totals) for one game."""
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    # Fetch team stats
    home_stats = ds.get_nba_team_season_stats(game.home_team_id)
    away_stats = ds.get_nba_team_season_stats(game.away_team_id)

    confidence_factors = []
    risk_factors       = []

    if not home_stats:
        risk_factors.append(f"No season stats for {game.home_team}")
        home_wp = 0.50
        home_ppg = 112.0
    else:
        home_wp  = _win_pct_from_stats(home_stats)
        home_ppg = _expected_team_score(home_stats)
        confidence_factors.append(f"{game.home_team} win%: {home_wp:.3f}")

    if not away_stats:
        risk_factors.append(f"No season stats for {game.away_team}")
        away_wp  = 0.50
        away_ppg = 112.0
    else:
        away_wp  = _win_pct_from_stats(away_stats)
        away_ppg = _expected_team_score(away_stats)
        confidence_factors.append(f"{game.away_team} win%: {away_wp:.3f}")

    # Adjusted expected scores using opponent points-allowed if available
    # Score estimate = avg(team_off_ppg, opp_def_ppg_allowed)
    home_def_allowed = _safe_float(away_stats.get("avg_points_allowed", 0))
    away_def_allowed = _safe_float(home_stats.get("avg_points_allowed", 0))

    if home_def_allowed > 80:  # sanity check
        home_score_est = (home_ppg + home_def_allowed) / 2.0
    else:
        home_score_est = home_ppg

    if away_def_allowed > 80:
        away_score_est = (away_ppg + away_def_allowed) / 2.0
    else:
        away_score_est = away_ppg

    expected_margin = (home_score_est - away_score_est) + 3.0  # +3 home court
    expected_total  = home_score_est + away_score_est
    confidence_factors.append(f"Exp total: {expected_total:.1f} ({game.home_team} {home_score_est:.0f} / {game.away_team} {away_score_est:.0f})")

    # Build markets
    results.extend(_build_h2h(game, home_wp, away_wp, scout_date, confidence_factors, risk_factors))
    results.extend(_build_spread(game, expected_margin, scout_date, confidence_factors, risk_factors))
    results.extend(_build_total(game, expected_total, scout_date, confidence_factors, risk_factors))

    return results
