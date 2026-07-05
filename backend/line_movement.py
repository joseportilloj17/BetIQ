"""
line_movement.py — Line movement tracking and sharp money signal (Phase C)

Every odds fetch appends a snapshot to line_movement_log.
Deltas vs the first fetch today reveal direction of movement.

Signal interpretation:
    If line moves toward the public-favored side → books adjusting to retail money (public chasing)
    If line moves against the public-favored side → sharp/professional money on the other side

line_movement_signal ∈ [-1.0, +1.0]:
    +1.0 = strong sharp signal toward pick side
    -1.0 = public/retail money on pick side (potential fade)
     0.0 = stable, no movement signal

Used in composite scoring:
    composite += line_movement_signal * 0.10  (when signal is available)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def record_odds_snapshot(
    db,
    fixture_id: str,
    sport: str,
    home_team: str,
    away_team: str,
    market_type: str,
    point: Optional[float],
    side: str,
    american_odds: Optional[int],
    decimal_odds: Optional[float],
    book: str = "fanduel",
) -> None:
    """
    Append an odds snapshot to line_movement_log.
    Call this each time odds are fetched for a fixture.
    """
    from database import LineMovementLog
    import pytz
    from sqlalchemy import text as _text

    now_utc = datetime.now(timezone.utc)

    # Find first snapshot today for this fixture+market+side to compute delta
    ct  = pytz.timezone("America/Chicago")
    today_ct = datetime.now(ct).strftime("%Y-%m-%d")

    try:
        first = (
            db.query(LineMovementLog)
            .filter(
                LineMovementLog.fixture_id  == fixture_id,
                LineMovementLog.market_type == market_type,
                LineMovementLog.side        == side,
                LineMovementLog.book        == book,
            )
            .filter(LineMovementLog.fetched_at >= f"{today_ct}T00:00:00")
            .order_by(LineMovementLog.fetched_at.asc())
            .first()
        )
    except Exception:
        first = None

    line_opened = first.decimal_odds if first else decimal_odds
    if line_opened and decimal_odds:
        line_delta       = round(decimal_odds - line_opened, 4)
        hours_since_open = (
            (now_utc - first.fetched_at).total_seconds() / 3600
            if first and first.fetched_at else 0.0
        )
    else:
        line_delta       = 0.0
        hours_since_open = 0.0

    # Classify movement direction (simple heuristic without public %):
    # Odds shortening (decimal goes down) = money coming in on this side
    # Odds lengthening (decimal goes up) = money leaving this side
    if abs(line_delta) < 0.02:
        direction = "stable"
    elif line_delta < 0:
        # Odds shortened → money piling in → could be sharp OR public
        # Without public % data we label as "steamed" (unknown sharp/public split)
        direction = f"steamed_{side}"
    else:
        direction = f"drifted_{side}"

    try:
        row = LineMovementLog(
            fixture_id         = fixture_id,
            sport              = sport,
            home_team          = home_team,
            away_team          = away_team,
            market_type        = market_type,
            point              = point,
            side               = side,
            american_odds      = american_odds,
            decimal_odds       = decimal_odds,
            fetched_at         = now_utc,
            book               = book,
            line_opened        = line_opened,
            line_delta         = line_delta,
            line_delta_hours   = round(hours_since_open, 2),
            movement_direction = direction,
        )
        db.add(row)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("[line_movement] Failed to record snapshot: %s", exc)


def get_line_movement_signal(
    db,
    fixture_id: str,
    market_type: str,
    side: str,
    pick_side: str,
    book: str = "fanduel",
    lookback_hours: float = 4.0,
) -> float:
    """
    Compute a line_movement_signal for a pick leg in [-1.0, +1.0].

    Logic:
      - If the pick side's odds shortened (steamed) → positive signal (money backing this side)
      - If the pick side's odds lengthened (drifted) → negative signal (money leaving)
      - Scale by magnitude of movement and recency (recent moves weighted 2×)

    Returns 0.0 if insufficient data (< 2 snapshots today).
    """
    from database import LineMovementLog
    import pytz

    ct       = pytz.timezone("America/Chicago")
    today_ct = datetime.now(ct).strftime("%Y-%m-%d")

    try:
        snapshots = (
            db.query(LineMovementLog)
            .filter(
                LineMovementLog.fixture_id  == fixture_id,
                LineMovementLog.market_type == market_type,
                LineMovementLog.side        == side,
                LineMovementLog.book        == book,
            )
            .filter(LineMovementLog.fetched_at >= f"{today_ct}T00:00:00")
            .order_by(LineMovementLog.fetched_at.asc())
            .all()
        )
    except Exception:
        return 0.0

    if len(snapshots) < 2:
        return 0.0

    first_odds = snapshots[0].decimal_odds or 0.0
    last_odds  = snapshots[-1].decimal_odds or 0.0
    if not first_odds or not last_odds:
        return 0.0

    total_delta = last_odds - first_odds   # negative = odds shortened = money in

    # Recent movement (last `lookback_hours`): weighted 2×
    now_utc    = datetime.now(timezone.utc)
    recent = [
        s for s in snapshots
        if s.fetched_at and (now_utc - s.fetched_at).total_seconds() <= lookback_hours * 3600
    ]
    recent_delta = 0.0
    if len(recent) >= 2:
        recent_delta = (recent[-1].decimal_odds or 0.0) - (recent[0].decimal_odds or 0.0)

    # Weighted delta: total + 2× recent
    weighted_delta = total_delta + 2.0 * recent_delta

    # Normalise: a 0.20 decimal odds move (~20 cents) = strong signal
    # Cap at ±1.0
    signal = max(-1.0, min(1.0, -weighted_delta / 0.20))  # negative delta = positive signal (odds shortened)

    # If pick_side matches the side being tracked, signal is aligned; otherwise invert
    if pick_side.lower() != side.lower():
        signal = -signal

    return round(signal, 3)


def get_movement_summary(db, fixture_id: str) -> dict:
    """
    Return a summary of today's line movement for a fixture.
    Used in health endpoint and diagnostic logging.
    """
    from database import LineMovementLog
    import pytz

    ct       = pytz.timezone("America/Chicago")
    today_ct = datetime.now(ct).strftime("%Y-%m-%d")

    try:
        rows = (
            db.query(LineMovementLog)
            .filter(
                LineMovementLog.fixture_id == fixture_id,
            )
            .filter(LineMovementLog.fetched_at >= f"{today_ct}T00:00:00")
            .all()
        )
    except Exception:
        return {}

    if not rows:
        return {"snapshots": 0}

    markets: dict = {}
    for r in rows:
        key = f"{r.market_type}:{r.side}"
        if key not in markets:
            markets[key] = {"first": r.decimal_odds, "last": r.decimal_odds,
                            "direction": r.movement_direction, "n": 0}
        markets[key]["last"] = r.decimal_odds
        markets[key]["n"] += 1
        markets[key]["delta"] = round((r.decimal_odds or 0) - (markets[key]["first"] or 0), 4)

    return {"snapshots": len(rows), "markets": markets}
