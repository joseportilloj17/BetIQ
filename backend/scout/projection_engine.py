"""
scout/projection_engine.py — Shared projection math for all scout modules.

Uses scipy.stats.norm (already a dependency via scikit-learn) for CDF.
Falls back to a manual approximation if scipy is unavailable.
"""
from __future__ import annotations
import math
from typing import Optional

from scout.base import (
    GRADE_THRESHOLDS, PROB_MIN, PROB_MAX, RECENT_FORM_WEIGHT
)


# ── Normal CDF ─────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def project_over_under(mean: float, std: float, threshold: float) -> float:
    """
    P(stat >= threshold) assuming normally distributed stat with given mean/std.
    Returns probability in [PROB_MIN, PROB_MAX].
    """
    if std <= 0:
        return clamp_probability(1.0 if mean >= threshold else 0.0)
    z = (threshold - mean) / std
    p = 1.0 - _norm_cdf(z)
    return clamp_probability(p)


def project_under(mean: float, std: float, threshold: float) -> float:
    """P(stat < threshold)."""
    return clamp_probability(1.0 - project_over_under(mean, std, threshold))


def project_win_probability(
    team_win_pct: float,
    opponent_win_pct: float,
    home_advantage: float = 0.035,
    is_home: bool = True,
) -> float:
    """
    Simple log5 win probability with home-field adjustment.
    home_advantage: additional win-prob boost for home team (default 3.5pp).
    """
    if team_win_pct <= 0:
        return clamp_probability(0.3)
    if opponent_win_pct <= 0:
        return clamp_probability(0.7)
    # Log5 formula
    p   = team_win_pct
    q   = opponent_win_pct
    log5 = (p * (1 - q)) / (p * (1 - q) + (1 - p) * q)
    adj = home_advantage if is_home else -home_advantage
    return clamp_probability(log5 + adj)


def project_spread_cover(
    expected_margin: float,
    std: float,
    spread: float,
) -> float:
    """
    P(team covers spread) = P(actual_margin - spread > 0).
    expected_margin: positive means home team wins by that margin.
    spread: negative means home team favored (e.g. -3.5).
    """
    # Cover if actual_margin > -spread (for home team covering -spread)
    threshold = -spread
    return project_over_under(expected_margin, std, threshold)


def project_total_over(
    expected_total: float,
    std: float,
    total_line: float,
) -> float:
    """P(combined score > total_line)."""
    return project_over_under(expected_total, std, total_line)


# ── Blending helpers ───────────────────────────────────────────────────────────

def blend_season_recent(
    season_avg: float,
    recent_avg: Optional[float],
    recent_weight: float = RECENT_FORM_WEIGHT,
) -> float:
    """Weighted blend: 60% recent / 40% season (or 100% season if no recent)."""
    if recent_avg is None or recent_avg <= 0:
        return season_avg
    return recent_weight * recent_avg + (1 - recent_weight) * season_avg


def compute_ci_95(mean: float, std: float) -> tuple[float, float]:
    """Return (lower, upper) 95% confidence interval. Lower floored at 0."""
    margin = 1.96 * std
    return max(0.0, mean - margin), mean + margin


# ── Grading ───────────────────────────────────────────────────────────────────

def grade_from_probability(p: float) -> str:
    """Map hit probability to A/B/C/D grade."""
    if p >= GRADE_THRESHOLDS["A"]:
        return "A"
    if p >= GRADE_THRESHOLDS["B"]:
        return "B"
    if p >= GRADE_THRESHOLDS["C"]:
        return "C"
    return "D"


def clamp_probability(p: float) -> float:
    """Clamp to [PROB_MIN, PROB_MAX] to avoid degenerate outputs."""
    return max(PROB_MIN, min(PROB_MAX, p))


# ── Sanity checks ─────────────────────────────────────────────────────────────

def detect_streak(
    recent_games: list[dict],
    stat_key: str,
    season_avg: float,
    n: int = 3,
) -> tuple[str | None, float]:
    """
    Detect hot/cold streak from last N games.
    Returns (streak_label, recent_n_avg) where:
      streak_label = 'HOT' | 'COLD' | None
    Threshold: if L{n} avg >= 130% of season avg → HOT
               if L{n} avg <= 70% of season avg → COLD
    """
    if not recent_games or season_avg <= 0:
        return None, 0.0

    vals = []
    for g in recent_games[:n]:
        v = g.get(stat_key)
        try:
            f = float(v)
            if f >= 0:
                vals.append(f)
        except (TypeError, ValueError):
            pass

    if not vals:
        return None, 0.0

    recent_avg = sum(vals) / len(vals)
    ratio = recent_avg / season_avg

    if ratio >= 1.30:
        return "HOT", recent_avg
    if ratio <= 0.70:
        return "COLD", recent_avg
    return None, recent_avg


def validate_projection(
    projected_value: float,
    season_avg: float,
    std: float,
    risk_factors: list,
) -> list:
    """
    Check projection sanity; append risk factors for anomalies.
    Returns updated risk_factors list.
    """
    rf = list(risk_factors)

    # Projection more than 2x outside historical range
    if season_avg > 0 and projected_value > season_avg * 2:
        rf.append(f"Projection ({projected_value:.1f}) > 2x season avg ({season_avg:.1f}) — flagged")

    if season_avg > 0 and projected_value < season_avg * 0.3:
        rf.append(f"Projection ({projected_value:.1f}) < 30% of season avg ({season_avg:.1f}) — flagged")

    # Confidence interval narrower than expected
    if std < season_avg * 0.1 and season_avg > 2:
        rf.append("Confidence interval suspiciously narrow — limited data")

    return rf


def std_from_average(avg: float, sport: str, stat: str) -> float:
    """
    Estimate standard deviation from average using sport/stat-specific
    coefficients of variation (CV).  Used when only season avg is available.
    """
    # CV table: typical std/mean ratios by sport+stat
    _CV = {
        # NBA
        ("NBA", "points"):   0.40,
        ("NBA", "rebounds"):  0.50,
        ("NBA", "assists"):   0.55,
        ("NBA", "threes"):    0.75,
        ("NBA", "steals"):    0.80,
        ("NBA", "blocks"):    0.90,
        # MLB hitting
        ("MLB", "hits"):      0.80,
        ("MLB", "total_bases"): 0.85,
        ("MLB", "rbi"):       0.90,
        ("MLB", "hr"):        1.00,
        ("MLB", "runs"):      0.90,
        # MLB pitching
        ("MLB", "strikeouts"): 0.40,
        ("MLB", "hits_allowed"): 0.50,
        ("MLB", "earned_runs"): 0.70,
        # NHL
        ("NHL", "shots"):     0.60,
        ("NHL", "points"):    0.90,
        ("NHL", "goals"):     1.00,
        ("NHL", "assists"):   0.95,
    }
    cv = _CV.get((sport, stat), 0.60)  # default 60% CV
    return max(0.5, avg * cv)
