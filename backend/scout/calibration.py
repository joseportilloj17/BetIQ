"""
scout/calibration.py — Scout-specific calibration tracking.

Tracks actual vs expected hit rates for scouted props, per grade and sport.
Writes results to `scout_calibration_log` table.
Separate from the main calibration_tracker.py which covers mock_bet_legs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# ── Alert thresholds ───────────────────────────────────────────────────────────

_ALERT_PP   = 10.0    # drift >= 10pp → alert
_CRITICAL_PP = 20.0   # drift >= 20pp → critical
_MIN_SAMPLE  = 20     # minimum resolved props to run calibration


# ── Schema ─────────────────────────────────────────────────────────────────────

CREATE_SCOUT_CALIBRATION_LOG = """
CREATE TABLE IF NOT EXISTS scout_calibration_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT NOT NULL,
    sport           TEXT,
    quality_grade   TEXT,
    sample_size     INTEGER,
    expected_hit_pct REAL,
    actual_hit_pct  REAL,
    drift_pp        REAL,
    alert_level     TEXT
)
"""


# ── Grade expected rates ───────────────────────────────────────────────────────

_EXPECTED_HIT_PCT = {
    "A": 0.70,   # A-grade props should hit ~70%
    "B": 0.60,
    "C": 0.52,
    "D": 0.40,
}


# ── Main functions ─────────────────────────────────────────────────────────────

def initialize(engine) -> None:
    """Create scout_calibration_log table if it doesn't exist."""
    try:
        with engine.connect() as conn:
            conn.execute(_sa_text(CREATE_SCOUT_CALIBRATION_LOG))
            conn.commit()
    except Exception as exc:
        print(f"[scout.calibration] init error: {exc}")


def _sa_text(sql: str):
    """Lazy import of sqlalchemy.text."""
    from sqlalchemy import text
    return text(sql)


def run_scout_calibration(engine, db) -> dict:
    """
    Compute hit-rate drift for scouted props by grade and sport.
    Reads from scouted_props where actual_hit IS NOT NULL.
    Returns summary dict with alert_level.
    """
    try:
        from sqlalchemy import text

        slices = []
        worst_drift = 0.0
        overall_level = "ok"

        # Per-grade calibration
        grade_rows = db.execute(text("""
            SELECT quality_grade,
                   COUNT(*) AS n,
                   AVG(CAST(actual_hit AS REAL)) AS actual_rate,
                   AVG(hit_probability) AS expected_rate
            FROM   scouted_props
            WHERE  actual_hit IS NOT NULL
            GROUP  BY quality_grade
        """)).fetchall()

        for row in grade_rows:
            grade, n, actual_rate, expected_rate = row
            if n < _MIN_SAMPLE:
                continue
            if actual_rate is None or expected_rate is None:
                continue
            drift = abs(actual_rate - expected_rate) * 100
            level = _classify(drift)
            slices.append({
                "dimension": "grade",
                "value": grade,
                "sample_size": n,
                "expected_hit_pct": round(expected_rate * 100, 1),
                "actual_hit_pct": round(actual_rate * 100, 1),
                "drift_pp": round(drift, 1),
                "alert_level": level,
            })
            if drift > worst_drift:
                worst_drift = drift
            if level == "critical":
                overall_level = "critical"
            elif level == "alert" and overall_level != "critical":
                overall_level = "alert"

        # Per-sport calibration
        sport_rows = db.execute(text("""
            SELECT sport,
                   COUNT(*) AS n,
                   AVG(CAST(actual_hit AS REAL)) AS actual_rate,
                   AVG(hit_probability) AS expected_rate
            FROM   scouted_props
            WHERE  actual_hit IS NOT NULL
            GROUP  BY sport
        """)).fetchall()

        for row in sport_rows:
            sport, n, actual_rate, expected_rate = row
            if n < _MIN_SAMPLE:
                continue
            if actual_rate is None or expected_rate is None:
                continue
            drift = abs(actual_rate - expected_rate) * 100
            level = _classify(drift)
            slices.append({
                "dimension": "sport",
                "value": sport,
                "sample_size": n,
                "expected_hit_pct": round(expected_rate * 100, 1),
                "actual_hit_pct": round(actual_rate * 100, 1),
                "drift_pp": round(drift, 1),
                "alert_level": level,
            })
            if drift > worst_drift:
                worst_drift = drift
            if level == "critical":
                overall_level = "critical"
            elif level == "alert" and overall_level != "critical":
                overall_level = "alert"

        # Persist to scout_calibration_log
        run_at = datetime.now(timezone.utc).isoformat()
        for s in slices:
            db.execute(text("""
                INSERT INTO scout_calibration_log
                    (run_at, sport, quality_grade, sample_size,
                     expected_hit_pct, actual_hit_pct, drift_pp, alert_level)
                VALUES
                    (:run_at, :sport, :grade, :n,
                     :exp, :act, :drift, :level)
            """), {
                "run_at": run_at,
                "sport":  s["value"] if s["dimension"] == "sport" else None,
                "grade":  s["value"] if s["dimension"] == "grade" else None,
                "n":      s["sample_size"],
                "exp":    s["expected_hit_pct"],
                "act":    s["actual_hit_pct"],
                "drift":  s["drift_pp"],
                "level":  s["alert_level"],
            })
        db.commit()

        return {
            "alert_level":     overall_level,
            "worst_drift_pp":  round(worst_drift, 1),
            "slices_checked":  len(slices),
            "slices":          slices,
            "run_at":          run_at,
        }

    except Exception as exc:
        print(f"[scout.calibration] run error: {exc}")
        return {"error": str(exc), "alert_level": "error"}


def _classify(drift_pp: float) -> str:
    if drift_pp >= _CRITICAL_PP:
        return "critical"
    if drift_pp >= _ALERT_PP:
        return "alert"
    return "ok"


def latest_scout_calibration_summary(engine) -> Optional[dict]:
    """Return most recent scout calibration result from DB."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT run_at, alert_level, drift_pp, sample_size
                FROM   scout_calibration_log
                ORDER  BY id DESC
                LIMIT  1
            """)).fetchone()
        if not row:
            return None
        return {
            "run_at":       row[0],
            "alert_level":  row[1],
            "worst_drift_pp": row[2],
            "sample_size":  row[3],
        }
    except Exception as exc:
        print(f"[scout.calibration] summary error: {exc}")
        return None


def settle_scouted_props(db, date: Optional[str] = None) -> dict:
    """
    Placeholder: match scouted_props to actual outcomes and set actual_hit.
    In production this would join to a results feed or scores API.
    For now returns a count of unsettled props.
    """
    try:
        from sqlalchemy import text
        date_filter = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = db.execute(text("""
            SELECT COUNT(*) FROM scouted_props
            WHERE  actual_hit IS NULL
            AND    scout_date <= :d
        """), {"d": date_filter}).fetchone()
        unsettled = row[0] if row else 0
        return {"unsettled_props": unsettled, "date": date_filter}
    except Exception as exc:
        return {"error": str(exc)}
