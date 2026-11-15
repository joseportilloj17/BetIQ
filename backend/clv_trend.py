"""
clv_trend.py — Rolling CLV trend monitor for deployed arms.

Tracks whether betting arms are losing their edge over time by computing
rolling-window CLV (Closing Line Value) trends per arm.

CLV measures:
  1. Devig CLV = devig_entry_prob - devig_close_prob  (positive = beat the close)
     Available only when both devig probs are present AND different.
  2. American CLV = clv_cents (from mock_bet_legs)
     Convention: clv_cents < 0 means positive CLV (beat the close).
     We INVERT the sign here so positive = good in the output.

Arms are identified by mock_bets.source (e.g. 'prospective', 'shadow_negbin',
'shadow_altline', etc.).

Usage:
    # From scheduler or CLI
    from clv_trend import clv_trend_detector
    result = clv_trend_detector(window_days=30, alert_threshold=0.0)

    # From API
    GET /api/clv-trends?window_days=30&alert_threshold=0.0
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from database import SessionLocal

logger = logging.getLogger(__name__)

# Arms worth monitoring — everything else is diagnostic/one-off.
# If an arm isn't in this set, it still appears in results but won't trigger alerts.
_MONITORED_ARMS = {
    "prospective",
    "prospective_pm",
    "top_picks_page",
    "shadow_negbin",
    "shadow_negbin_fullstack",
    "shadow_negbin_distwp_gate",
    "shadow_negbin_bothgates",
    "shadow_negbin_edgegate",
    "shadow_negbin_composite",
    "shadow_negbin_recal",
    "shadow_negbin_recal_fh",
    "shadow_negbin_fh",
    "shadow_negbin_atscap",
    "shadow_altline",
    "shadow_lineup",
    "shadow_prop",
    "shadow_teaser",
    "shadow_underdog_lag",
}

# Minimum settled legs in the window to produce a trend reading
_MIN_LEGS_FOR_TREND = 10

# How many sub-windows to compute for trend direction (slope)
_TREND_HALVES = 2


def clv_trend_detector(
    window_days: int = 30,
    alert_threshold: float = 0.0,
    db: Optional[Session] = None,
) -> dict:
    """
    Rolling CLV trend monitor for deployed arms.

    Parameters
    ----------
    window_days : int
        Look-back window in days (default 30).
    alert_threshold : float
        CLV level (in probability points for devig, or cents for American)
        below which an alert is raised. Default 0.0 = alert when CLV goes
        negative (losing edge).
    db : Session, optional
        SQLAlchemy session. If None, creates one internally.

    Returns
    -------
    dict with keys:
        window_days      — the window used
        alert_threshold  — the threshold used
        generated_at     — ISO timestamp
        arms             — list of per-arm dicts:
            arm               — source name
            model_used        — primary model for this arm
            clv_type          — 'devig' or 'american' (which measure was used)
            settled_legs      — number of settled legs with CLV in window
            current_clv       — rolling average CLV (positive = beating close)
            first_half_clv    — CLV in first half of window
            second_half_clv   — CLV in second half of window
            trend             — 'improving' / 'declining' / 'flat'
            alert             — True if current_clv < alert_threshold AND arm is monitored
            beat_close_pct    — % of legs that beat the closing line
            message           — human-readable summary
        alerts           — list of arm names with active alerts
        summary          — overall summary string
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()

    try:
        return _run_detector(db, window_days, alert_threshold)
    except Exception as e:
        logger.error(f"[CLV-Trend] Error: {e}", exc_info=True)
        return {
            "window_days": window_days,
            "alert_threshold": alert_threshold,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "arms": [],
            "alerts": [],
            "summary": f"Error running CLV trend detector: {e}",
            "error": str(e),
        }
    finally:
        if own_session:
            db.close()


def _run_detector(db: Session, window_days: int, alert_threshold: float) -> dict:
    """Core logic, separated for clean error handling."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    midpoint = datetime.now(timezone.utc) - timedelta(days=window_days / 2)
    midpoint_str = midpoint.strftime("%Y-%m-%d %H:%M:%S")

    # ── Query all settled legs with CLV data in the window ────────────────
    # We pull both CLV measures and decide per-arm which to use.
    query = text("""
        SELECT
            b.source,
            b.model_used,
            b.settled_at,
            l.devig_entry_prob,
            l.devig_close_prob,
            l.clv_cents,
            l.clv_available,
            l.leg_result
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE b.settled_at >= :cutoff
          AND l.leg_result IS NOT NULL
          AND b.source IS NOT NULL
          AND (
              (l.devig_entry_prob IS NOT NULL AND l.devig_close_prob IS NOT NULL)
              OR l.clv_available = 1
          )
        ORDER BY b.source, b.settled_at
    """)

    rows = db.execute(query, {"cutoff": cutoff_str}).fetchall()

    if not rows:
        return {
            "window_days": window_days,
            "alert_threshold": alert_threshold,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "arms": [],
            "alerts": [],
            "summary": f"No settled legs with CLV data in the last {window_days} days.",
        }

    # ── Group by arm ──────────────────────────────────────────────────────
    arm_data: dict[str, list[dict]] = defaultdict(list)
    arm_models: dict[str, str] = {}

    for row in rows:
        source = row[0] or "unknown"
        model_used = row[1] or "unknown"
        settled_at = str(row[2]) if row[2] else ""
        devig_entry = row[3]
        devig_close = row[4]
        clv_cents = row[5]
        clv_available = row[6]

        arm_data[source].append({
            "settled_at": settled_at,
            "devig_entry": devig_entry,
            "devig_close": devig_close,
            "clv_cents": clv_cents,
            "clv_available": clv_available,
        })
        # Track most common model_used per arm
        if source not in arm_models:
            arm_models[source] = model_used

    # ── Compute per-arm stats ─────────────────────────────────────────────
    arm_results = []
    alert_arms = []

    for arm, legs in sorted(arm_data.items()):
        # Decide which CLV measure to use for this arm.
        # ALWAYS prefer devig CLV (probability edge) over American CLV.
        # American CLV on heavy favorites picks up vig timing (bookmaker
        # tightening overround e.g. -5000 → -10000), not real probability
        # edge.  If devig data exists but entry == close for all legs,
        # report devig CLV = 0.000 — do NOT fall back to American.
        devig_legs_diff = [
            lg for lg in legs
            if lg["devig_entry"] is not None
            and lg["devig_close"] is not None
            and lg["devig_entry"] != lg["devig_close"]
        ]
        devig_legs_all = [
            lg for lg in legs
            if lg["devig_entry"] is not None
            and lg["devig_close"] is not None
        ]
        american_legs = [
            lg for lg in legs
            if lg["clv_available"] == 1 and lg["clv_cents"] is not None
        ]

        if len(devig_legs_diff) >= _MIN_LEGS_FOR_TREND:
            # Real devig CLV data with actual differences
            clv_type = "devig"
            active_legs = devig_legs_diff
        elif len(devig_legs_all) >= _MIN_LEGS_FOR_TREND:
            # Devig data exists but entry == close for (nearly) all legs.
            # This IS the answer: zero probability CLV.  Do NOT fall back
            # to American CLV which is vig-timing noise on favorites.
            clv_type = "devig"
            active_legs = devig_legs_all
        elif len(american_legs) >= _MIN_LEGS_FOR_TREND:
            # No devig data at all — American is the only option.
            # Flag it as vig-timing-contaminated on heavy favorites.
            clv_type = "american"
            active_legs = american_legs
        else:
            # Not enough data for either measure
            total_available = max(len(devig_legs), len(american_legs))
            arm_results.append({
                "arm": arm,
                "model_used": arm_models.get(arm, "unknown"),
                "clv_type": "insufficient",
                "settled_legs": total_available,
                "current_clv": None,
                "first_half_clv": None,
                "second_half_clv": None,
                "trend": "insufficient_data",
                "alert": False,
                "beat_close_pct": None,
                "message": (
                    f"Insufficient CLV data: {total_available} legs "
                    f"(need {_MIN_LEGS_FOR_TREND})"
                ),
            })
            continue

        # Compute CLV values
        if clv_type == "devig":
            clv_values = [lg["devig_entry"] - lg["devig_close"] for lg in active_legs]
            beat_close = [v for v in clv_values if v > 0]
        else:
            # American: clv_cents < 0 means we beat the close, so invert
            clv_values = [-lg["clv_cents"] for lg in active_legs]
            beat_close = [v for v in clv_values if v > 0]

        n = len(clv_values)
        current_clv = sum(clv_values) / n
        beat_close_pct = round(100.0 * len(beat_close) / n, 1) if n > 0 else 0.0

        # Split into halves for trend direction
        # Use settled_at timestamp to split, not index
        first_half = []
        second_half = []
        for i, lg in enumerate(active_legs):
            settled_str = lg["settled_at"]
            if clv_type == "devig":
                val = lg["devig_entry"] - lg["devig_close"]
            else:
                val = -lg["clv_cents"]

            if settled_str <= midpoint_str:
                first_half.append(val)
            else:
                second_half.append(val)

        first_half_clv = (
            sum(first_half) / len(first_half) if first_half else None
        )
        second_half_clv = (
            sum(second_half) / len(second_half) if second_half else None
        )

        # Determine trend
        if first_half_clv is not None and second_half_clv is not None:
            delta = second_half_clv - first_half_clv
            # Use a meaningful threshold: 5% of absolute first-half CLV or 0.5 cents
            noise_band = max(abs(first_half_clv) * 0.05, 0.5) if clv_type == "american" else max(abs(first_half_clv) * 0.05, 0.001)
            if delta > noise_band:
                trend = "improving"
            elif delta < -noise_band:
                trend = "declining"
            else:
                trend = "flat"
        else:
            trend = "insufficient_data"

        # Alert check
        is_monitored = arm in _MONITORED_ARMS
        has_alert = is_monitored and current_clv < alert_threshold
        if has_alert:
            alert_arms.append(arm)

        # Build message
        clv_unit = "pp" if clv_type == "devig" else "cents"
        # Flag American CLV as vig-timing-contaminated
        clv_reliability = "probability_edge"
        if clv_type == "american":
            clv_reliability = "vig_timing_contaminated"
        msg_parts = [
            f"{arm}: rolling {window_days}d CLV = {current_clv:+.2f} {clv_unit}",
            f"({n} legs, {beat_close_pct}% beat close)",
        ]
        if clv_type == "american":
            msg_parts.append("[AMERICAN — vig-timing only, NOT probability edge]")
        if trend == "declining":
            msg_parts.append("-- DECLINING")
        elif trend == "improving":
            msg_parts.append("-- improving")
        if has_alert:
            msg_parts.append("** ALERT: below threshold **")

        arm_results.append({
            "arm": arm,
            "model_used": arm_models.get(arm, "unknown"),
            "clv_type": clv_type,
            "clv_reliability": clv_reliability,
            "settled_legs": n,
            "current_clv": round(current_clv, 4),
            "first_half_clv": round(first_half_clv, 4) if first_half_clv is not None else None,
            "second_half_clv": round(second_half_clv, 4) if second_half_clv is not None else None,
            "trend": trend,
            "alert": has_alert,
            "beat_close_pct": beat_close_pct,
            "message": " ".join(msg_parts),
        })

    # ── Build summary ─────────────────────────────────────────────────────
    n_arms = len([a for a in arm_results if a["trend"] != "insufficient_data"])
    n_declining = len([a for a in arm_results if a["trend"] == "declining"])
    n_alerts = len(alert_arms)

    if n_alerts > 0:
        summary = (
            f"ALERT: {n_alerts} arm(s) below CLV threshold: "
            f"{', '.join(alert_arms)}. "
            f"{n_declining}/{n_arms} arms declining."
        )
    elif n_declining > 0:
        summary = (
            f"{n_declining}/{n_arms} arms showing declining CLV trend. "
            f"No alerts triggered."
        )
    else:
        summary = f"All {n_arms} arms with sufficient data are stable or improving."

    return {
        "window_days": window_days,
        "alert_threshold": alert_threshold,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": arm_results,
        "alerts": alert_arms,
        "summary": summary,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    from database import init_db
    init_db()

    window = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    result = clv_trend_detector(window_days=window, alert_threshold=threshold)
    print(json.dumps(result, indent=2, default=str))

    # Print concise arm-level summary to stderr for quick scanning
    for arm in result.get("arms", []):
        if arm["trend"] != "insufficient_data":
            print(arm["message"], file=sys.stderr)
