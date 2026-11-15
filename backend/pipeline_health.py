"""
pipeline_health.py — Per-pipeline error tracking, zero-row alerts, and health reporting.

Every scheduled pipeline registers its run result via record_run().
Morning email / API status endpoint reads get_health_summary().

Alert rules:
  - errors > 0 in any run → DEGRADED
  - rows_written == 0 AND games_exist → DOWN (alert fires)
  - no run in 36 hours → STALE
"""
from __future__ import annotations

import logging
import os
import sqlite3
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# ── Known pipelines ──────────────────────────────────────────────────────────
# name → expected_interval_hours (how often it should run on a game day)
PIPELINES = {
    "nfl_fantasy_pipeline":    {"interval_h": 36,  "table": "fantasy_forward_preds"},   # Wed-Sat daily
    "nfl_forward_preds":       {"interval_h": 36,  "table": "fantasy_forward_preds"},   # outcome: rows in table
    "nfl_grading":             {"interval_h": 192, "table": "fantasy_forward_preds"},   # Tuesdays only
    "nba_scout":               {"interval_h": 28,  "table": "nba_forward_preds"},       # daily game-day
    "nba_forward_preds":       {"interval_h": 28,  "table": "nba_forward_preds"},       # outcome: rows in table
    "nba_pretip":              {"interval_h": 28,  "table": "nba_forward_preds"},       # daily game-day
    "nba_grading":             {"interval_h": 28,  "table": "nba_forward_preds"},       # daily after scores
    "nfl_scout":               {"interval_h": 28,  "table": "scouted_props"},           # daily AM
    "mock_bet_generate":       {"interval_h": 28,  "table": "mock_bets"},               # daily AM+PM
    "mock_bet_settle":         {"interval_h": 28,  "table": "mock_bets"},               # every 30-min cycle
    "injury_watcher":          {"interval_h": 28,  "table": "injury_edge_snapshots"},   # 3x/day game-day
    "model_scorecard":         {"interval_h": 192, "table": "weekly_model_scorecard"},  # Tuesdays
    "nightly_maintenance":     {"interval_h": 28,  "table": None},                      # daily overnight
    "db_backup":               {"interval_h": 28,  "table": None},                      # daily 3 AM
}


def _ensure_table():
    """Create pipeline_health_log table if missing."""
    con = sqlite3.connect(os.path.abspath(_HIST_DB), timeout=10)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_health_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline TEXT NOT NULL,
            run_ts TEXT NOT NULL,
            rows_written INTEGER DEFAULT 0,
            errors INTEGER DEFAULT 0,
            error_details TEXT,
            games_exist INTEGER,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_phl_pipeline
        ON pipeline_health_log(pipeline, run_ts)
    """)
    con.commit()
    con.close()


def record_run(
    pipeline: str,
    rows_written: int = 0,
    errors: int = 0,
    error_details: str | None = None,
    games_exist: bool | None = None,
) -> str:
    """Record a pipeline run and return status string.

    Returns one of: 'OK', 'DEGRADED', 'DOWN'.
    Fires macOS alert on DOWN.
    """
    _ensure_table()

    if errors > 0 and rows_written > 0:
        status = "DEGRADED"
    elif rows_written == 0 and games_exist:
        status = "DOWN"
    elif errors > 0:
        status = "DEGRADED"
    else:
        status = "OK"

    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(os.path.abspath(_HIST_DB), timeout=10)
    con.execute("""
        INSERT INTO pipeline_health_log
        (pipeline, run_ts, rows_written, errors, error_details, games_exist, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (pipeline, now, rows_written, errors,
          error_details[:2000] if error_details else None,
          1 if games_exist else (0 if games_exist is not None else None),
          status, now))
    con.commit()
    con.close()

    if status in ("DOWN", "DEGRADED"):
        _fire_alert(pipeline, rows_written, errors, error_details, status=status)

    return status


def _fire_alert(pipeline: str, rows: int, errors: int, details: str | None,
                status: str = "DOWN"):
    """macOS notification + email + console log for DOWN/DEGRADED status."""
    msg = f"{pipeline}: {rows} rows written, {errors} errors"
    if details:
        msg += f" — {details[:200]}"
    logger.error("[PIPELINE %s] %s", status, msg)
    print(f"[PIPELINE {status}] {msg}")
    try:
        os.system(
            f'osascript -e \'display notification "{msg}" '
            f'with title "BetIQ Pipeline {status}" sound name "Basso"\''
        )
    except Exception:
        pass
    # Email alert — subject starts with "BETIQ DOWN" or "BETIQ DEGRADED"
    try:
        from researcher.delivery import send_email
        subj = f"BETIQ {status}: {pipeline}"
        body = (f"Pipeline: {pipeline}\n"
                f"Status: {status}\n"
                f"Rows written: {rows}\n"
                f"Errors: {errors}\n")
        if details:
            body += f"Details: {details[:500]}\n"
        body += f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}"
        send_email(subj, body)
    except Exception as exc:
        logger.warning("[pipeline_health] Email alert failed: %s", exc)


# Day-of-week names for "Not due" labels
_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Weekly pipeline schedule: pipeline → (run_weekdays, label)
_WEEKLY_SCHEDULE = {
    "nfl_fantasy_pipeline": ([2, 3, 4, 5], "Wed-Sat"),   # Wed-Sat
    "nfl_forward_preds":    ([2, 3, 4, 5], "Wed-Sat"),   # Wed-Sat (outcome check)
    "nfl_grading":          ([1], "Tue"),                  # Tuesdays
    "model_scorecard":      ([1], "Tue"),                  # Tuesdays
}


def get_health_summary(hours: int = 24) -> list[dict]:
    """Return per-pipeline health for the last N hours.

    Each entry: {pipeline, last_run, rows_24h, errors_24h, status}.
    Status reflects the *latest* run, with "recovered" notation when earlier
    runs in the window had errors.  Weekly pipelines show "Not due" instead
    of STALE when they simply haven't run yet this window.
    """
    _ensure_table()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    con = sqlite3.connect(os.path.abspath(_HIST_DB), timeout=10)
    con.row_factory = sqlite3.Row

    # Aggregate stats
    agg_rows = con.execute("""
        SELECT pipeline,
               MAX(run_ts) as last_run,
               SUM(rows_written) as rows_24h,
               SUM(errors) as errors_24h
        FROM pipeline_health_log
        WHERE run_ts >= ?
        GROUP BY pipeline
    """, (cutoff,)).fetchall()

    # Latest run status per pipeline
    latest_rows = con.execute("""
        SELECT pipeline, status FROM pipeline_health_log
        WHERE (pipeline, run_ts) IN (
            SELECT pipeline, MAX(run_ts)
            FROM pipeline_health_log
            WHERE run_ts >= ?
            GROUP BY pipeline
        )
    """, (cutoff,)).fetchall()

    # Count of error runs per pipeline (for "recovered" notation)
    error_counts = con.execute("""
        SELECT pipeline,
               SUM(CASE WHEN status IN ('DOWN','DEGRADED') THEN 1 ELSE 0 END) as err_runs
        FROM pipeline_health_log
        WHERE run_ts >= ?
        GROUP BY pipeline
    """, (cutoff,)).fetchall()
    con.close()

    latest_map = {r["pipeline"]: r["status"] for r in latest_rows}
    error_map = {r["pipeline"]: r["err_runs"] for r in error_counts}

    results = []
    for r in agg_rows:
        name = r["pipeline"]
        latest_status = latest_map.get(name, "UNKNOWN")
        err_runs = error_map.get(name, 0)

        # "recovered" if latest is OK but earlier runs had errors
        if latest_status == "OK" and err_runs > 0:
            latest_status = f"OK (recovered, {err_runs} error{'s' if err_runs > 1 else ''} earlier)"

        results.append({
            "pipeline": name,
            "last_run": r["last_run"],
            "rows_24h": r["rows_24h"] or 0,
            "errors_24h": r["errors_24h"] or 0,
            "status": latest_status,
        })

    # Handle pipelines that haven't reported in this window
    reported = {r["pipeline"] for r in results}
    now_utc = datetime.now(timezone.utc)
    today_dow = now_utc.weekday()  # 0=Mon

    for name, cfg in PIPELINES.items():
        if name not in reported:
            sched = _WEEKLY_SCHEDULE.get(name)
            if sched:
                run_days, day_label = sched
                if today_dow not in run_days:
                    # Find next run day
                    for offset in range(1, 8):
                        next_dow = (today_dow + offset) % 7
                        if next_dow in run_days:
                            next_label = _DOW_NAMES[next_dow]
                            break
                    else:
                        next_label = day_label
                    status = f"Not due (next {next_label})"
                else:
                    status = "STALE"
            else:
                status = "STALE"
            results.append({
                "pipeline": name,
                "last_run": None,
                "rows_24h": 0,
                "errors_24h": 0,
                "status": status,
            })

    def _sort_key(x):
        s = x["status"]
        if "DOWN" in s:
            return (0, x["pipeline"])
        if "DEGRADED" in s:
            return (1, x["pipeline"])
        if s == "STALE":
            return (2, x["pipeline"])
        if "Not due" in s:
            return (4, x["pipeline"])
        return (3, x["pipeline"])

    results.sort(key=_sort_key)
    return results


def _read_git_commit_local() -> str:
    """Read short git commit from .git/HEAD + refs (no subprocess)."""
    repo = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    git_dir = os.path.join(repo, ".git")
    try:
        head = open(os.path.join(git_dir, "HEAD")).read().strip()
        if head.startswith("ref: "):
            ref_path = os.path.join(git_dir, head[5:])
            return open(ref_path).read().strip()[:7]
        return head[:7]
    except Exception:
        pass
    try:
        import subprocess as _sp
        return _sp.check_output(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__), stderr=_sp.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _check_working_tree_dirty() -> bool:
    """Return True if the git working tree has uncommitted changes."""
    repo = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["/usr/bin/git", "status", "--porcelain"],
            cwd=repo, stderr=_sp.DEVNULL, timeout=5,
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return False  # can't tell — don't false-alarm


def _get_version_info() -> dict:
    """Get running vs disk git commit + dirty-tree flag for version guard."""
    info = {"running": "unknown", "disk": "unknown", "match": False,
            "dirty": False}
    try:
        import requests
        resp = requests.get("http://localhost:8000/api/health", timeout=5)
        if resp.ok:
            data = resp.json()
            info["running"] = data.get("git_commit_running", "unknown")
            info["disk"] = data.get("git_commit_disk", "unknown")
            info["match"] = data.get("version_match", False)
            info["dirty"] = data.get("working_tree_dirty", False)
    except Exception:
        info["disk"] = _read_git_commit_local()
        info["dirty"] = _check_working_tree_dirty()
    # "unknown" is always a mismatch
    if info["running"] == "unknown" or info["disk"] == "unknown":
        info["match"] = False
    return info


def format_health_email_block(hours: int = 24) -> str:
    """Format health summary as a text block for the morning email."""
    summary = get_health_summary(hours)
    if not summary:
        return "No pipeline health data in last 24h.\n"

    lines = ["Pipeline Health (last 24h):", ""]

    # Version guard
    ver = _get_version_info()
    if not ver["match"]:
        lines.append(f"  ⚠️  VERSION MISMATCH: running={ver['running']} disk={ver['disk']}")
        lines.append(f"     Server needs restart to pick up code changes!")
        lines.append("")
    else:
        lines.append(f"  Git: {ver['running']} (running = disk)")
        lines.append("")
    if ver.get("dirty"):
        lines.append("  ⚠️  UNCOMMITTED CHANGES: running code includes uncommitted changes")
        lines.append("     Commit and push before relying on version hash!")
        lines.append("")

    lines.append(f"  {'Pipeline':<28s} {'Status':<40s} {'Rows':>6s} {'Errors':>6s}  Last Run")
    lines.append("  " + "-" * 100)
    for s in summary:
        lr = s["last_run"][:16] if s["last_run"] else "NEVER"
        flag = ""
        st = s["status"]
        if "DOWN" in st or "DEGRADED" in st:
            flag = " ⚠️"
        elif st == "STALE":
            flag = " ?"
        lines.append(
            f"  {s['pipeline']:<28s} {st:<40s} {s['rows_24h']:>6d} "
            f"{s['errors_24h']:>6d}  {lr}{flag}"
        )
    lines.append("")
    return "\n".join(lines)


def capture_exception(pipeline: str) -> str:
    """Capture current exception traceback as a string for error_details."""
    return traceback.format_exc()
