"""
injury_watcher.py — Event-triggered injury monitoring for NFL prop edge detection.

Polls Sleeper player status every ~15 min on game days (Thu-Mon).
On status change for a player with props:
  1. Fire one targeted prop fetch for that game
  2. Log the share model's redistributed projections
  3. Track: at the first post-news snapshot, is model-vs-line edge on
     teammates' props larger than baseline? Does it close by next snapshot?

Sleeper player DB fields used:
  injury_status  — Questionable / Doubtful / Out / IR / NA / None
  injury_body_part — body part affected
  news_updated   — millisecond timestamp of last news update

Credit usage: ~1 targeted prop fetch per status change (6 credits per
              fetch for 3 market batches). Sleeper API is free (no auth).

Run:
    python injury_watcher.py poll          # single poll cycle
    python injury_watcher.py watch         # continuous ~15min polling
    python injury_watcher.py report        # edge-at-news report
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

_SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

# Injury statuses that matter for prop redistribution
_ACTIONABLE_STATUSES = {"Out", "Doubtful", "IR"}
# Status changes that trigger a prop fetch
_TRIGGER_TRANSITIONS = {
    # From → To transitions that move lines
    (None, "Out"), (None, "Doubtful"), (None, "IR"),
    ("Questionable", "Out"), ("Questionable", "Doubtful"),
    ("Questionable", "IR"),
    ("Doubtful", "Out"), ("Doubtful", "IR"),
    ("Out", None), ("Out", "Questionable"),  # return from injury
    ("IR", None), ("IR", "Questionable"),
    ("Doubtful", None), ("Doubtful", "Questionable"),
}

# Only track skill positions with props
_SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# Polling interval
_POLL_INTERVAL_SEC = 15 * 60  # 15 minutes

# Game-day schedule: Thu=3, Fri=4, Sat=5, Sun=6, Mon=0
_GAME_DAYS = {0, 3, 4, 5, 6}  # Mon, Thu, Fri, Sat, Sun


_NFL_DB = os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db")


def _get_team_kickoff(team: str, game_date: str | None = None) -> dict | None:
    """Look up kickoff time for a team's next game from nfl_games.

    Returns: {game_id, game_date, kickoff_utc, inactive_release_utc} or None.
    NFL inactive lists are released 90 min before kickoff.
    """
    try:
        conn = sqlite3.connect(os.path.abspath(_NFL_DB))
        conn.row_factory = sqlite3.Row
        if game_date:
            row = conn.execute("""
                SELECT game_id, game_date, gametime, home_team, away_team
                FROM nfl_games
                WHERE (home_team = ? OR away_team = ?) AND game_date = ?
                LIMIT 1
            """, (team, team, game_date)).fetchone()
        else:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            row = conn.execute("""
                SELECT game_id, game_date, gametime, home_team, away_team
                FROM nfl_games
                WHERE (home_team = ? OR away_team = ?) AND game_date >= ?
                ORDER BY game_date LIMIT 1
            """, (team, team, today)).fetchone()
        conn.close()
        if not row:
            return None

        # gametime is ET (Eastern), e.g. "13:00" or "20:15"
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        kick_naive = datetime.strptime(
            f"{row['game_date']} {row['gametime']}", "%Y-%m-%d %H:%M"
        )
        kick_et = kick_naive.replace(tzinfo=et)
        kick_utc = kick_et.astimezone(timezone.utc)
        inactive_utc = kick_utc - timedelta(minutes=90)
        return {
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "kickoff_utc": kick_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inactive_release_utc": inactive_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as e:
        logger.warning("[InjuryWatcher] Kickoff lookup failed for %s: %s", team, e)
        return None


def _ensure_tables():
    """Create injury_watch tables if they don't exist."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injury_status_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            player_id TEXT,
            position TEXT,
            team TEXT,
            prev_status TEXT,
            new_status TEXT,
            injury_body_part TEXT,
            news_updated_ms INTEGER,
            detected_at TEXT NOT NULL,
            is_game_day INTEGER,
            game_id TEXT,
            kickoff_utc TEXT,
            inactive_release_utc TEXT,
            sleeper_lag_minutes REAL,
            prop_fetch_triggered INTEGER DEFAULT 0,
            credits_used INTEGER DEFAULT 0
        )
    """)
    # Migrate: add columns if table already existed without them
    for col, ctype in [
        ("game_id", "TEXT"),
        ("kickoff_utc", "TEXT"),
        ("inactive_release_utc", "TEXT"),
        ("sleeper_lag_minutes", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE injury_status_log ADD COLUMN {col} {ctype}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.execute("""
        CREATE TABLE IF NOT EXISTS injury_edge_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            change_id INTEGER REFERENCES injury_status_log(id),
            player_name TEXT NOT NULL,
            teammate_name TEXT,
            teammate_stat TEXT,
            teammate_line REAL,
            model_p_over REAL,
            market_p_over REAL,
            edge_vs_market REAL,
            snapshot_type TEXT,  -- 'pre_news' / 'post_news' / 'later'
            snapshot_at TEXT NOT NULL,
            hours_after_news REAL
        )
    """)
    conn.commit()
    conn.close()


def _fetch_sleeper_players() -> dict:
    """Fetch current Sleeper player DB. Returns {player_id: player_dict}."""
    try:
        resp = requests.get(_SLEEPER_PLAYERS_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        import traceback as _tb
        logger.error("[InjuryWatcher] Sleeper fetch failed: %s\n%s", e, _tb.format_exc())
        return {}


def _load_last_statuses() -> dict:
    """Load the most recent injury status per player from the log.

    Returns: {player_id: {"status": str, "news_updated_ms": int}}
    """
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT player_id, new_status, news_updated_ms
        FROM injury_status_log
        WHERE id IN (
            SELECT MAX(id) FROM injury_status_log
            GROUP BY player_id
        )
    """).fetchall()
    conn.close()
    return {
        r["player_id"]: {
            "status": r["new_status"],
            "news_updated_ms": r["news_updated_ms"],
        }
        for r in rows
    }


def poll_status_changes() -> list[dict]:
    """Poll Sleeper for injury status changes. Returns list of changes.

    Each change: {player_name, player_id, position, team,
                  prev_status, new_status, injury_body_part,
                  news_updated_ms, is_trigger}
    """
    _ensure_tables()

    players = _fetch_sleeper_players()
    if not players:
        return []

    last_statuses = _load_last_statuses()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Use CT weekday so it matches the scheduler's gate
    try:
        from zoneinfo import ZoneInfo
        ct_wd = datetime.now(ZoneInfo("America/Chicago")).weekday()
    except Exception:
        ct_wd = datetime.now(timezone.utc).weekday()
    is_game_day = ct_wd in _GAME_DAYS

    changes = []
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))

    for pid, pinfo in players.items():
        pos = pinfo.get("position", "")
        if pos not in _SKILL_POSITIONS:
            continue
        team = pinfo.get("team")
        if not team:
            continue

        new_status = pinfo.get("injury_status") or None
        news_ms = pinfo.get("news_updated") or 0
        name = pinfo.get("full_name", "?")
        body_part = pinfo.get("injury_body_part")

        prev = last_statuses.get(str(pid))
        prev_status = prev["status"] if prev else None
        prev_news_ms = prev["news_updated_ms"] if prev else 0

        # Detect change: either status changed or news_updated changed
        status_changed = new_status != prev_status
        news_changed = news_ms > prev_news_ms and prev_news_ms > 0

        if not status_changed and not news_changed:
            continue

        is_trigger = (prev_status, new_status) in _TRIGGER_TRANSITIONS

        # Look up kickoff + inactive release time for Sleeper lag
        kickoff_info = _get_team_kickoff(team)
        game_id = kickoff_info["game_id"] if kickoff_info else None
        kickoff_utc = kickoff_info["kickoff_utc"] if kickoff_info else None
        inactive_utc = kickoff_info["inactive_release_utc"] if kickoff_info else None

        # Compute Sleeper lag: how many minutes after the official inactive
        # release did we first detect this change?
        sleeper_lag = None
        if inactive_utc and is_game_day:
            try:
                inact_dt = datetime.strptime(inactive_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc)
                det_dt = datetime.strptime(now_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc)
                sleeper_lag = round((det_dt - inact_dt).total_seconds() / 60, 1)
            except Exception:
                pass

        change = {
            "player_name": name,
            "player_id": str(pid),
            "position": pos,
            "team": team,
            "prev_status": prev_status,
            "new_status": new_status,
            "injury_body_part": body_part,
            "news_updated_ms": news_ms,
            "is_trigger": is_trigger,
            "is_game_day": is_game_day,
            "game_id": game_id,
            "kickoff_utc": kickoff_utc,
            "sleeper_lag_minutes": sleeper_lag,
        }
        changes.append(change)

        # Log to DB
        conn.execute("""
            INSERT INTO injury_status_log
            (player_name, player_id, position, team, prev_status,
             new_status, injury_body_part, news_updated_ms,
             detected_at, is_game_day,
             game_id, kickoff_utc, inactive_release_utc, sleeper_lag_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, str(pid), pos, team, prev_status, new_status,
              body_part, news_ms, now_str, 1 if is_game_day else 0,
              game_id, kickoff_utc, inactive_utc, sleeper_lag))

    conn.commit()
    conn.close()

    return changes


def trigger_prop_fetch(change: dict) -> int:
    """Fire a targeted prop fetch for the game containing this player.

    Returns credits used (0 if skipped).
    """
    team = change.get("team")
    if not team:
        return 0

    try:
        from creator_tier import fetch_player_props
    except ImportError:
        logger.warning("[InjuryWatcher] creator_tier not available")
        return 0

    logger.info(
        "[InjuryWatcher] Triggering prop fetch for %s (%s) — "
        "%s → %s",
        change["player_name"], team,
        change["prev_status"], change["new_status"],
    )

    # Fetch props for NFL (will get all imminent games, but that's the
    # minimum granularity for The Odds API)
    result = fetch_player_props(
        "americanfootball_nfl",
        hours_ahead=72,  # catch Thu-Mon games
    )

    credits = 0
    if result and not result.get("skipped_reason"):
        # Estimate credits from result
        credits = result.get("props_upserted", 0) // 100  # rough estimate
        credits = max(credits, 6)  # minimum ~6 credits per fetch

    # Update the most recent log row for this player
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute("""
        UPDATE injury_status_log
        SET prop_fetch_triggered = 1, credits_used = ?
        WHERE id = (
            SELECT id FROM injury_status_log
            WHERE player_id = ?
            ORDER BY id DESC LIMIT 1
        )
    """, (credits, change["player_id"]))
    conn.commit()
    conn.close()

    # Write edge snapshots for teammates' props
    try:
        _write_edge_snapshots(change)
    except Exception as exc:
        logger.warning("[InjuryWatcher] snapshot write failed: %s", exc)

    return credits


def _write_edge_snapshots(change: dict) -> int:
    """After a trigger, snapshot teammate prop edges into injury_edge_snapshots.

    Queries fantasy_forward_preds for the injured player's teammates,
    recording model vs market edge at the time of the injury news.
    Returns number of rows inserted.
    """
    team = change.get("team")
    player_name = change.get("player_name", "?")
    if not team:
        return 0

    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    # Get the log row id for this change
    log_row = conn.execute("""
        SELECT id FROM injury_status_log
        WHERE player_id = ?
        ORDER BY id DESC LIMIT 1
    """, (change["player_id"],)).fetchone()
    if not log_row:
        conn.close()
        return 0
    change_id = log_row["id"]

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Find teammate props created in the last 48h with distributional model
    teammates = conn.execute("""
        SELECT player_name, stat_type, prop_line,
               p_over, devigged_implied, edge_vs_market,
               created_at
        FROM fantasy_forward_preds
        WHERE game_id LIKE ?
          AND player_name != ?
          AND method LIKE '%distributional%'
          AND created_at >= datetime('now', '-48 hours')
        ORDER BY created_at DESC
    """, (f"%{team}%", player_name)).fetchall()

    if not teammates:
        conn.close()
        return 0

    inserted = 0
    for t in teammates:
        conn.execute("""
            INSERT INTO injury_edge_snapshots
            (change_id, player_name, teammate_name, teammate_stat,
             teammate_line, model_p_over, market_p_over, edge_vs_market,
             snapshot_type, snapshot_at, hours_after_news)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'post_news', ?, NULL)
        """, (
            change_id, player_name, t["player_name"], t["stat_type"],
            t["prop_line"], t["p_over"], t["devigged_implied"],
            t["edge_vs_market"], now_utc,
        ))
        inserted += 1

    conn.commit()
    conn.close()

    if inserted:
        logger.info(
            "[InjuryWatcher] Wrote %d edge snapshots for %s (%s)",
            inserted, player_name, team,
        )
    return inserted


def run_poll_cycle() -> dict:
    """Run one poll cycle: detect changes, trigger fetches for actionable ones.

    Returns summary dict.
    """
    changes = poll_status_changes()
    if not changes:
        return {"changes": 0, "triggers": 0, "credits": 0}

    triggers = 0
    total_credits = 0
    triggered_teams = set()

    for change in changes:
        team = change["team"]
        is_trigger = change["is_trigger"]
        is_gd = change["is_game_day"]

        lag = change.get("sleeper_lag_minutes")
        lag_str = f"  lag={lag:+.0f}min" if lag is not None else ""
        print(f"  {change['player_name']:25s} {change['position']:3s} "
              f"{change['team']:4s} "
              f"{change['prev_status'] or '—':>12s} → "
              f"{change['new_status'] or '—':>12s} "
              f"{'[TRIGGER]' if is_trigger else ''}{lag_str}")

        # Only trigger prop fetch on game days for actionable transitions
        if is_trigger and is_gd and team not in triggered_teams:
            try:
                credits = trigger_prop_fetch(change)
                total_credits += credits
                triggers += 1
                triggered_teams.add(team)
            except Exception as exc:
                logger.error(
                    "[InjuryWatcher] trigger_prop_fetch FAILED for %s (%s): %s",
                    change["player_name"], team, exc,
                )
                print(f"[InjuryWatcher] ERROR trigger_prop_fetch "
                      f"{change['player_name']} ({team}): {exc}")

    return {
        "changes": len(changes),
        "triggers": triggers,
        "credits": total_credits,
        "triggered_teams": list(triggered_teams),
    }


def run_watch(interval_sec: int = _POLL_INTERVAL_SEC) -> None:
    """Continuous polling loop for game days."""
    print(f"[InjuryWatcher] Starting watch loop ({interval_sec}s interval)")
    print(f"  Active on game days: Thu-Mon (weekdays {sorted(_GAME_DAYS)})")

    while True:
        now = datetime.now(timezone.utc)
        weekday = now.weekday()

        if weekday not in _GAME_DAYS:
            print(f"[{now.strftime('%H:%M')}] Not a game day "
                  f"(weekday={weekday}). Sleeping 1h.")
            time.sleep(3600)
            continue

        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')} UTC] Polling...")
        result = run_poll_cycle()
        print(f"  → {result['changes']} changes, "
              f"{result['triggers']} triggers, "
              f"{result['credits']} credits used")

        time.sleep(interval_sec)


def report_edge_at_news() -> dict:
    """Report: at first post-news snapshot, was model edge on teammates' props
    larger than baseline? Did it close by next snapshot?

    Joins injury_status_log with fantasy_forward_preds to find teammate
    props stored around the time of injury news.
    """
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    # Get all triggered status changes
    changes = conn.execute("""
        SELECT id, player_name, player_id, position, team,
               prev_status, new_status, detected_at, credits_used
        FROM injury_status_log
        WHERE prop_fetch_triggered = 1
        ORDER BY detected_at
    """).fetchall()

    if not changes:
        conn.close()
        return {"n_triggers": 0, "message": "No triggered changes yet"}

    results = []
    for ch in changes:
        team = ch["team"]
        detected = ch["detected_at"]

        # Find teammate props in the same week's predictions
        # Look for predictions created within 48h of the detection
        teammates = conn.execute("""
            SELECT player_name, stat_type, prop_line,
                   p_over, devigged_implied, edge_vs_market,
                   direction, created_at, model_version
            FROM fantasy_forward_preds
            WHERE game_id LIKE ?
              AND player_name != ?
              AND method LIKE '%distributional%'
              AND created_at BETWEEN datetime(?, '-48 hours')
                  AND datetime(?, '+48 hours')
            ORDER BY created_at
        """, (f"%{team}%", ch["player_name"], detected, detected)).fetchall()

        if teammates:
            edges = [abs(t["edge_vs_market"]) for t in teammates
                     if t["edge_vs_market"] is not None]
            avg_edge = sum(edges) / len(edges) if edges else 0

            results.append({
                "injured_player": ch["player_name"],
                "team": team,
                "transition": f"{ch['prev_status']} → {ch['new_status']}",
                "detected_at": detected,
                "n_teammate_props": len(teammates),
                "avg_abs_edge": round(avg_edge, 4),
                "credits_used": ch["credits_used"],
            })

    conn.close()

    # Baseline edge (from all predictions, for comparison)
    conn2 = sqlite3.connect(os.path.abspath(_HIST_DB))
    baseline = conn2.execute("""
        SELECT AVG(ABS(edge_vs_market)) as avg_edge
        FROM fantasy_forward_preds
        WHERE edge_vs_market IS NOT NULL
          AND method LIKE '%distributional%'
          AND data_error IS NULL
    """).fetchone()
    conn2.close()

    baseline_edge = float(baseline[0]) if baseline and baseline[0] else 0

    return {
        "n_triggers": len(changes),
        "results": results,
        "baseline_avg_abs_edge": round(baseline_edge, 4),
        "credit_total": sum(r.get("credits_used", 0) for r in results),
    }


def print_edge_report(report: dict) -> None:
    """Pretty-print edge-at-news report."""
    if report.get("n_triggers", 0) == 0:
        print("No triggered injury changes yet.")
        return

    print("=" * 80)
    print(f"INJURY EDGE REPORT — {report['n_triggers']} triggered changes")
    print("=" * 80)
    print(f"  Baseline avg |edge|: {report['baseline_avg_abs_edge']:.4f}")
    print(f"  Total credits used:  {report['credit_total']}")

    print(f"\n  {'Player':25s} {'Team':5s} {'Transition':25s} "
          f"{'#Props':>6s} {'Avg |edge|':>10s}")
    print("  " + "─" * 75)

    for r in report.get("results", []):
        print(f"  {r['injured_player']:25s} {r['team']:5s} "
              f"{r['transition']:25s} {r['n_teammate_props']:6d} "
              f"{r['avg_abs_edge']:10.4f}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "poll":
        print("Running single poll cycle...")
        result = run_poll_cycle()
        print(f"\nResult: {result}")
    elif cmd == "watch":
        run_watch()
    elif cmd == "report":
        report = report_edge_at_news()
        print_edge_report(report)
    else:
        print("Usage: python injury_watcher.py <command>")
        print()
        print("Commands:")
        print("  poll   — run one poll cycle (detect + trigger)")
        print("  watch  — continuous ~15min polling on game days")
        print("  report — edge-at-news analysis")
