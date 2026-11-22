"""
nfl_odds_history.py — Backfill historical NFL game odds snapshots (TheOddsAPI historical).

Purpose: GM4 needs EARLY-WEEK lines (openers) so the model can be judged on beating
the opener / predicting open→close movement, not only on the close (nflverse has close only).

Runs INSIDE the API server (single-writer rule): credits are logged through
creator_tier._get → credits_log, so the credit tripwire sees them.  Raw snapshots
are written to data/odds_history/nfl/<snapshot_ts>.json (durable, gitignored);
nothing is written to bets.db.  Resumable: existing snapshot files are skipped.

Snapshots per regular-season week: Tuesday 16:00Z and Friday 16:00Z before the
week's Sunday slate (Tuesday ≈ opener after look-ahead settles; Friday ≈ mid-week).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "odds_history", "nfl"))
_BOOKS = "pinnacle,fanduel,draftkings,betmgm,williamhill_us,espnbet,betrivers,bovada"
STATUS: dict = {"running": False}
_LOCK = threading.Lock()


def snapshot_targets(seasons):
    import nfl_data_py as nfl
    s = nfl.import_schedules(list(seasons))
    s = s[s.game_type == "REG"]
    out = []
    for (season, week), g in s.groupby(["season", "week"]):
        days = sorted(datetime.fromisoformat(str(d)[:10]) for d in g.gameday)
        sunday = next((d for d in days if d.weekday() == 6), days[-1])
        tue = sunday - timedelta(days=5)
        fri = sunday - timedelta(days=2)
        for lab, d in (("tue", tue), ("fri", fri)):
            out.append({"season": int(season), "week": int(week), "label": lab,
                        "ts": d.replace(hour=16, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return out


def path_for(ts):
    return os.path.join(_DIR, ts.replace(":", "") + ".json")


def fetch_one(t):
    import creator_tier as ct
    data = ct._get("/historical/sports/americanfootball_nfl/odds", {
        "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "decimal",
        "date": t["ts"], "bookmakers": _BOOKS})
    if not data or not isinstance(data, dict) or "data" not in data:
        logger.warning("[OddsHistory] empty/failed snapshot %s (%s W%s %s)", t["ts"], t["season"], t["week"], t["label"])
        return False
    os.makedirs(_DIR, exist_ok=True)
    with open(path_for(t["ts"]), "w") as f:
        json.dump({**t, "timestamp": data.get("timestamp"), "data": data["data"]}, f)
    return True


def run_backfill(seasons=(2023, 2024, 2025, 2026), dry_run=False, limit=None):
    with _LOCK:
        if STATUS.get("running"):
            return {"error": "backfill already running", **STATUS}
        STATUS.clear()
        STATUS.update(running=True, started=datetime.now(timezone.utc).isoformat(), done=0, failed=0, skipped=0)
    try:
        now = datetime.now(timezone.utc)
        targets = [t for t in snapshot_targets(seasons)
                   if datetime.fromisoformat(t["ts"].replace("Z", "+00:00")) < now]
        if limit:
            targets = targets[:limit]
        STATUS["total"] = len(targets)
        for t in targets:
            if os.path.exists(path_for(t["ts"])):
                STATUS["skipped"] += 1
                continue
            if dry_run:
                STATUS["skipped"] += 1
                continue
            ok = fetch_one(t)
            STATUS["done" if ok else "failed"] += 1
            time.sleep(0.4)
        if STATUS["failed"]:
            logger.warning("[OddsHistory] backfill finished with %d failed snapshots", STATUS["failed"])
        return dict(STATUS)
    finally:
        STATUS["running"] = False
        STATUS["finished"] = datetime.now(timezone.utc).isoformat()


def start_background(seasons, dry_run=False, limit=None):
    th = threading.Thread(target=run_backfill, args=(seasons, dry_run, limit), daemon=True, name="NFLOddsHistory")
    th.start()
    return {"started": True, "seasons": list(seasons), "dry_run": dry_run, "limit": limit}
