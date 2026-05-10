"""
scout/base.py — Shared dataclasses, constants, and grade thresholds.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

# ── Grade thresholds (hit_probability) ────────────────────────────────────────
GRADE_THRESHOLDS = {
    "A": 0.65,   # strong play
    "B": 0.55,   # solid
    "C": 0.50,   # marginal
    # below 0.50 → Grade D (skip / bet opposite direction)
}

# Scout-grade composite multipliers (applied to recommender composite score)
GRADE_MULTIPLIERS = {
    "A": 1.5,
    "B": 1.2,
    "C": 1.0,
    "D": 0.5,
}

# Probability clamp — avoid extremes from sparse data
PROB_MIN = 0.05
PROB_MAX = 0.95

# Recent form weight in blend (remaining weight goes to season average)
RECENT_FORM_WEIGHT = 0.60

# Projection version — bump when methodology changes
PROJECTION_VERSION = "1.0"

# Markets where the scout projects a binary direction (not over/under a threshold)
DIRECTION_MARKETS = {"h2h", "spread", "alt_spreads"}
TOTAL_MARKETS     = {"totals", "alt_totals"}


@dataclass
class GameInfo:
    """Minimal game context passed into every scout function."""
    game_id:      str
    sport:        str                    # NBA | MLB | NHL
    home_team:    str
    away_team:    str
    home_team_id: str
    away_team_id: str
    commence_time: str                   # ISO string UTC
    venue:        Optional[str] = None
    extra:        dict          = field(default_factory=dict)  # sport-specific context


@dataclass
class ScoutedProp:
    """
    One scouted market/player combination.
    Persisted to scouted_props table; linked to mock_bet_legs and user_pick_legs.
    """
    scout_date:      str             # YYYY-MM-DD CT
    sport:           str             # NBA | MLB | NHL
    game_id:         str
    home_team:       str
    away_team:       str
    commence_time:   str             # ISO UTC

    market_type:     str             # player_points | h2h | spread | totals | …
    player_name:     Optional[str]   # None for team markets
    player_id:       Optional[str]
    team:            Optional[str]
    side:            Optional[str]   # over | under | home | away

    threshold:       Optional[float] # line for props/totals; spread for spreads
    projected_value: float           # model projection (mean)
    projected_low_95:  float         # lower bound of 95% CI
    projected_high_95: float         # upper bound of 95% CI
    projected_std_dev: float
    hit_probability: float           # P(outcome covers | projection)
    quality_grade:   str             # A | B | C | D

    confidence_factors: List[str]    = field(default_factory=list)
    risk_factors:       List[str]    = field(default_factory=list)

    actual_outcome_value: Optional[float] = None
    actual_hit:           Optional[bool]  = None
    scout_accuracy:       Optional[float] = None

    data_source:         str = "espn"
    projection_version:  str = PROJECTION_VERSION
