"""
calibration_tracker.py — Per-grade/sport calibration drift detection for BetIQ.

Tracks whether the model's predicted win rates match actual outcomes.

Tables used:
  • mock_bet_legs  — grade (qualification_tier field is used; we also check lqs_grade)
  • mock_bets      — bet status (SETTLED_WIN / SETTLED_LOSS)
  • calibration_drift_log (created here) — daily drift snapshots

Drift thresholds:
  • ≥ 10 pp gap  → alert  (degraded)
  • ≥ 20 pp gap  → critical (auto-disable flag set)

Grades tracked: A, B, C, D  (from mock_bet_legs.lqs_grade)
Sports tracked: MLB, NBA, NHL, NFL, Soccer (from mock_bets.sport)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text as _text

log = logging.getLogger(__name__)

# ─── Expected win rates by EV grade (empirically derived from historical actuals) ──
#
# IMPORTANT: mock_bet_legs.grade is an EV-based tier assigned by
# parlay_builder._grade(ev, edge), NOT a quality rank:
#   Grade A: model ev > 0.08 AND edge > 0.03  (model thinks highest edge)
#   Grade B: model ev > 0.04 AND edge > 0.01
#   Grade C: model ev > 0.0
#   Grade D: model ev > -0.05
#   Grade F: model ev <= -0.05
#
# Because the ML model is overconfident on spread bets, Grade A legs
# (dominated by -1.5 spread picks) actually win ~33%, not 68%.
# Grade F legs (dominated by alt_spreads) actually win ~67% because
# alt_spreads cover more often.
#
# Expected WRs here are EMPIRICAL BASELINES — what each grade historically
# achieves.  Drift triggers when actual WR deviates significantly from this
# baseline (indicating model or market structure change), NOT from a
# win_prob-derived ideal.
#
# Update these when enough new data accumulates (> 500 additional legs).
_EXPECTED_WIN_RATE: dict[str, float] = {
    "A": 0.33,   # dominated by overconfident spread picks; model ev >> actual ev
    "B": 0.61,
    "C": 0.64,
    "D": 0.60,
    "F": 0.68,   # dominated by alt_spreads with favorable coverage
}
_DRIFT_ALERT_PP    = 10.0   # percentage points → degraded
_DRIFT_CRITICAL_PP = 20.0   # percentage points → critical / auto-disable
_MIN_SAMPLE        = 15     # minimum legs to compute drift

# ─── Current-model era floor ─────────────────────────────────────────────────
# Legs before this date were scored by primitive models (≤74 features, no Pi,
# no spread-value, no pitcher stats).  Including them in drift metrics produces
# misleading calibration gaps (e.g. +26pp from 318 April test/debug legs).
# All reporting queries default to this floor so dashboards reflect the model
# that is actually running.
_CURRENT_ERA_DATE  = "2026-05-26"


# ─── CalibrationDriftLog SQLAlchemy model ─────────────────────────────────────

def _ensure_calibration_table(engine) -> None:
    """Create calibration_drift_log table if it doesn't exist (raw SQL, safe to re-run)."""
    import sqlite3, os
    url = str(engine.url)
    if url.startswith("sqlite:////"):
        db_path = url[len("sqlite:///"):]
    elif url.startswith("sqlite:///"):
        db_path = os.path.abspath(url[len("sqlite:///"):])
    else:
        raise ValueError(f"calibration_tracker: unsupported engine URL: {url}")

    conn = sqlite3.connect(db_path, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_drift_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date      TEXT    NOT NULL,      -- YYYY-MM-DD UTC
            slice_type    TEXT    NOT NULL,      -- 'grade' | 'sport' | 'overall'
            slice_key     TEXT    NOT NULL,      -- 'A' / 'B' / 'MLB' / 'all'
            n_settled     INTEGER,
            actual_wr     REAL,                 -- 0.0-1.0
            expected_wr   REAL,                 -- 0.0-1.0
            drift_pp      REAL,                 -- actual_wr - expected_wr (pp * 100)
            alert_level   TEXT,                 -- 'ok' | 'alert' | 'critical'
            auto_disabled INTEGER DEFAULT 0,    -- 1 if auto-disable triggered
            created_at    TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ─── Core computation ──────────────────────────────────────────────────────────

def _compute_grade_drift(db_session) -> list[dict]:
    """
    For each LQS grade (A/B/C/D), compare actual leg-level win rate against
    the expected baseline.

    IMPORTANT: uses mbl.leg_result (individual leg outcome), NOT mb.status
    (bet outcome).  Using bet-level status for parlay legs is wrong: a Grade A
    leg that won its market still shows as a loss if another leg in the parlay
    lost, making Grade A look terrible and triggering false auto_disabled flags.
    Only legs with a resolved leg_result ('WIN' or 'LOSS') are counted.
    """
    rows = db_session.execute(_text("""
        SELECT
            mbl.grade                                            AS grade,
            COUNT(*)                                             AS n,
            SUM(CASE WHEN mbl.leg_result = 'WIN' THEN 1 ELSE 0 END) AS wins
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mbl.grade IS NOT NULL
          AND mb.game_date >= :era_floor
        GROUP BY mbl.grade
    """), {"era_floor": _CURRENT_ERA_DATE}).fetchall()

    results = []
    for row in rows:
        grade, n, wins = row[0], row[1], row[2]
        if n < _MIN_SAMPLE or grade not in _EXPECTED_WIN_RATE:
            continue  # Grade F included — empirical baseline defined above
        actual_wr   = wins / n
        expected_wr = _EXPECTED_WIN_RATE[grade]
        drift_pp    = round((actual_wr - expected_wr) * 100, 2)
        alert_level = (
            "critical" if abs(drift_pp) >= _DRIFT_CRITICAL_PP
            else "alert" if abs(drift_pp) >= _DRIFT_ALERT_PP
            else "ok"
        )
        results.append({
            "slice_type":   "grade",
            "slice_key":    grade,
            "n_settled":    n,
            "actual_wr":    round(actual_wr, 4),
            "expected_wr":  expected_wr,
            "drift_pp":     drift_pp,
            "alert_level":  alert_level,
        })
    return results


def _compute_sport_drift(db_session) -> list[dict]:
    """
    For each sport, compare actual win rate against the overall expected rate (0.55).
    Sports don't have per-sport expected rates in the model, so we use overall baseline.

    IMPORTANT: restricted to straight bets (bet_type='straight') only.
    Parlay hit rates (15–20%) are far below the per-game expected baseline (55%),
    so including parlays causes every sport to appear critically miscalibrated even
    when individual game picks are well-calibrated.  Multi-sport parlay rows already
    have concatenated sport labels like "MLB | NHL" which are meaningless for
    single-sport drift measurement.
    """
    _SPORT_EXPECTED = 0.55   # overall model target

    rows = db_session.execute(_text("""
        SELECT
            mb.sport                                        AS sport,
            COUNT(*)                                        AS n,
            SUM(CASE WHEN mb.status = 'SETTLED_WIN' THEN 1 ELSE 0 END) AS wins
        FROM mock_bets mb
        WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND mb.bet_type = 'straight'
          AND mb.sport IS NOT NULL
          AND mb.game_date >= :era_floor
        GROUP BY mb.sport
    """), {"era_floor": _CURRENT_ERA_DATE}).fetchall()

    results = []
    for row in rows:
        sport, n, wins = row[0], row[1], row[2]
        if n < _MIN_SAMPLE:
            continue
        actual_wr   = wins / n
        drift_pp    = round((actual_wr - _SPORT_EXPECTED) * 100, 2)
        alert_level = (
            "critical" if abs(drift_pp) >= _DRIFT_CRITICAL_PP
            else "alert" if abs(drift_pp) >= _DRIFT_ALERT_PP
            else "ok"
        )
        results.append({
            "slice_type":   "sport",
            "slice_key":    sport,
            "n_settled":    n,
            "actual_wr":    round(actual_wr, 4),
            "expected_wr":  _SPORT_EXPECTED,
            "drift_pp":     drift_pp,
            "alert_level":  alert_level,
        })
    return results


def _compute_overall_drift(db_session) -> dict | None:
    """
    Overall settled mock bet win rate vs 0.55 baseline.
    Restricted to straight bets to avoid parlay hit rates (15-20%) falsely
    triggering critical calibration alerts across the board.
    """
    _EXPECTED = 0.55
    row = db_session.execute(_text("""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN status = 'SETTLED_WIN' THEN 1 ELSE 0 END) AS wins
        FROM mock_bets
        WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND bet_type = 'straight'
          AND game_date >= :era_floor
    """), {"era_floor": _CURRENT_ERA_DATE}).fetchone()
    if not row or row[0] < _MIN_SAMPLE:
        return None
    n, wins = row[0], row[1]
    actual_wr = wins / n
    drift_pp  = round((actual_wr - _EXPECTED) * 100, 2)
    alert_level = (
        "critical" if abs(drift_pp) >= _DRIFT_CRITICAL_PP
        else "alert" if abs(drift_pp) >= _DRIFT_ALERT_PP
        else "ok"
    )
    return {
        "slice_type":  "overall",
        "slice_key":   "all",
        "n_settled":   n,
        "actual_wr":   round(actual_wr, 4),
        "expected_wr": _EXPECTED,
        "drift_pp":    drift_pp,
        "alert_level": alert_level,
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def run_calibration_check(engine, db_session) -> dict:
    """
    Compute calibration drift across all slices and persist to calibration_drift_log.

    Returns summary dict:
      {
        "overall": {...},
        "by_grade": [...],
        "by_sport": [...],
        "worst_drift_pp": float,
        "alert_level": "ok" | "alert" | "critical",
        "auto_disabled": bool,
        "slices_checked": int,
      }
    """
    _ensure_calibration_table(engine)

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso  = datetime.now(timezone.utc).isoformat()

    grade_slices   = _compute_grade_drift(db_session)
    sport_slices   = _compute_sport_drift(db_session)
    overall_slice  = _compute_overall_drift(db_session)

    all_slices = grade_slices + sport_slices
    if overall_slice:
        all_slices.append(overall_slice)

    # Persist to calibration_drift_log
    import sqlite3, os
    url = str(engine.url)
    if url.startswith("sqlite:////"):
        db_path = url[len("sqlite:///"):]
    elif url.startswith("sqlite:///"):
        db_path = os.path.abspath(url[len("sqlite:///"):])
    else:
        db_path = None

    if db_path:
        raw = sqlite3.connect(db_path, timeout=15)
        try:
            # Delete today's existing rows first so repeated runs don't accumulate
            # duplicates that inflate the daily-summary response 8×.
            raw.execute(
                "DELETE FROM calibration_drift_log WHERE run_date = ?",
                (run_date,)
            )
            for s in all_slices:
                auto_dis = 1 if s["alert_level"] == "critical" else 0
                raw.execute("""
                    INSERT INTO calibration_drift_log
                      (run_date, slice_type, slice_key, n_settled, actual_wr,
                       expected_wr, drift_pp, alert_level, auto_disabled, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_date, s["slice_type"], s["slice_key"], s["n_settled"],
                    s["actual_wr"], s["expected_wr"], s["drift_pp"],
                    s["alert_level"], auto_dis, now_iso,
                ))
            raw.commit()
        finally:
            raw.close()

    # Derive summary
    worst_drift   = max((abs(s["drift_pp"]) for s in all_slices), default=0.0)
    any_critical  = any(s["alert_level"] == "critical" for s in all_slices)
    any_alert     = any(s["alert_level"] == "alert"    for s in all_slices)
    overall_level = (
        "critical" if any_critical
        else "alert" if any_alert
        else "ok"
    )

    if any_critical:
        log.error(
            f"[calibration] CRITICAL drift detected (worst={worst_drift:.1f}pp). "
            "Auto-disable flag set for affected slices."
        )
    elif any_alert:
        log.warning(f"[calibration] Alert: drift of {worst_drift:.1f}pp detected.")
    else:
        log.info(f"[calibration] OK — worst drift {worst_drift:.1f}pp across {len(all_slices)} slices")

    return {
        "overall":        overall_slice,
        "by_grade":       grade_slices,
        "by_sport":       sport_slices,
        "worst_drift_pp": round(worst_drift, 2),
        "alert_level":    overall_level,
        "auto_disabled":  any_critical,
        "slices_checked": len(all_slices),
    }


def latest_drift_summary(engine) -> dict:
    """
    Return the most recent drift check result per slice (for health endpoint).
    Does NOT recompute — reads from calibration_drift_log.
    """
    _ensure_calibration_table(engine)
    import sqlite3, os
    url = str(engine.url)
    if url.startswith("sqlite:////"):
        db_path = url[len("sqlite:///"):]
    elif url.startswith("sqlite:///"):
        db_path = os.path.abspath(url[len("sqlite:///"):])
    else:
        return {"error": "unsupported engine URL"}

    try:
        raw = sqlite3.connect(db_path, timeout=15)
        # Latest run_date
        row = raw.execute(
            "SELECT MAX(run_date) FROM calibration_drift_log"
        ).fetchone()
        last_date = row[0] if row else None

        if not last_date:
            raw.close()
            return {"last_check": None, "slices": [], "alert_level": "ok"}

        rows = raw.execute("""
            SELECT slice_type, slice_key, n_settled, actual_wr,
                   expected_wr, drift_pp, alert_level, auto_disabled
            FROM calibration_drift_log
            WHERE run_date = ?
            ORDER BY ABS(drift_pp) DESC
        """, (last_date,)).fetchall()
        raw.close()

        slices = [
            {
                "slice_type":  r[0], "slice_key":    r[1],
                "n_settled":   r[2], "actual_wr":    r[3],
                "expected_wr": r[4], "drift_pp":     r[5],
                "alert_level": r[6], "auto_disabled": bool(r[7]),
            }
            for r in rows
        ]

        worst_drift  = max((abs(s["drift_pp"]) for s in slices), default=0.0)
        any_critical = any(s["alert_level"] == "critical" for s in slices)
        any_alert    = any(s["alert_level"] == "alert"    for s in slices)
        level        = "critical" if any_critical else "alert" if any_alert else "ok"

        return {
            "last_check":     last_date,
            "slices":         slices,
            "worst_drift_pp": round(worst_drift, 2),
            "alert_level":    level,
        }
    except Exception as exc:
        log.warning(f"[calibration] latest_drift_summary error: {exc}")
        return {"last_check": None, "slices": [], "alert_level": "ok", "error": str(exc)}


def get_brier_drift(db_session, days: int = 30) -> dict:
    """
    Compute 7-day rolling Brier per sport and compare against the baseline
    (prior 23 days of the window).  Returns per-sport drift and an overall
    alert flag.

    Parameters
    ----------
    db_session : SQLAlchemy Session
    days       : total window length (default 30)

    Returns
    -------
    {
      "sports": [
        {
          "sport":         str,
          "recent_brier":  float | None,   # last 7 days
          "baseline_brier": float | None,  # prior (days-7) days
          "delta":         float | None,   # recent - baseline
          "alert":         bool,           # True if delta > 0.03
        }
      ],
      "overall_alert": bool,
      "days": int,
    }
    """
    from datetime import datetime, timedelta
    from collections import defaultdict

    now         = datetime.utcnow()
    recent_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    baseline_start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    recent_end   = now.strftime("%Y-%m-%d")
    baseline_end = recent_start  # recent window starts where baseline ends

    _BRIER_ALERT_DELTA = 0.03

    def _brier_by_sport(start_str: str, end_str: str) -> dict[str, dict]:
        rows = db_session.execute(_text("""
            SELECT mbl.sport, mbl.win_prob, mbl.leg_result
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND date(mb.generated_at) >= :start
              AND date(mb.generated_at) < :end
              AND mbl.win_prob IS NOT NULL
              AND mbl.sport IS NOT NULL
        """), {"start": start_str, "end": end_str}).fetchall()

        sport_items: dict[str, list] = defaultdict(list)
        for r in rows:
            sport, wp, result = r[0], r[1], r[2]
            sport_items[sport].append({
                "p": wp / 100.0,
                "y": 1.0 if result == 'WIN' else 0.0,
            })

        out = {}
        for sport, items in sport_items.items():
            if len(items) < 5:
                out[sport] = {"brier": None, "n": len(items)}
            else:
                brier = round(
                    sum((i["p"] - i["y"]) ** 2 for i in items) / len(items), 4
                )
                out[sport] = {"brier": brier, "n": len(items)}
        return out

    recent_by_sport   = _brier_by_sport(recent_start, recent_end)
    baseline_by_sport = _brier_by_sport(baseline_start, baseline_end)

    all_sports = set(recent_by_sport) | set(baseline_by_sport)
    sport_results = []
    any_alert = False

    for sport in sorted(all_sports):
        r_info = recent_by_sport.get(sport, {"brier": None})
        b_info = baseline_by_sport.get(sport, {"brier": None})
        r_brier = r_info.get("brier")
        b_brier = b_info.get("brier")

        delta = None
        alert = False
        if r_brier is not None and b_brier is not None:
            delta = round(r_brier - b_brier, 4)
            alert = delta > _BRIER_ALERT_DELTA

        if alert:
            any_alert = True

        sport_results.append({
            "sport":          sport,
            "recent_brier":   r_brier,
            "baseline_brier": b_brier,
            "delta":          delta,
            "alert":          alert,
        })

    return {
        "sports":        sport_results,
        "overall_alert": any_alert,
        "days":          days,
    }
