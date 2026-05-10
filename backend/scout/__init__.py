"""
scout — Game-level scouting + projection layer for BetIQ.

Exports the main entry points used by the scheduler and placement layer.
"""
from scout.base import ScoutedProp, GameInfo, GRADE_THRESHOLDS
from scout.projection_engine import grade_from_probability, project_over_under
from scout.runner import run_daily_scout

__all__ = [
    "ScoutedProp", "GameInfo", "GRADE_THRESHOLDS",
    "grade_from_probability", "project_over_under",
    "run_daily_scout",
]
