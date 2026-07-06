"""
scheduler.py — Phase 2B: Background polling scheduler.

Runs auto-settle in a background thread so the FastAPI server
doesn't need an external cron job. Configurable interval.

Usage (standalone):
    python scheduler.py                   # every 30 min
    python scheduler.py --interval 60     # every 60 min

The scheduler is also started automatically when FastAPI starts
(via startup hook in main.py) in daemon mode.
"""
from __future__ import annotations
import os
import time
import threading
import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from database import SessionLocal, init_db, DB_WRITE_LOCK
from auto_settle import run_auto_settle, RETRAIN_STATE, check_retrain_status

_CT = ZoneInfo("America/Chicago")

# ── State (read by API for status endpoint) ────────────────────────────────────
_scheduler_state = {
    "running":              False,
    "interval_mins":        30,
    "last_run":             None,
    "last_result":          None,
    "next_run":             None,
    "total_cycles":         0,
    "total_settled":        0,
    "last_odds_update":     None,   # ISO date string of last nightly NBA odds run
    "last_odds_result":     None,
    "last_ats_update":      None,   # ISO date string of last nightly NBA ATS run
    "last_ats_result":      None,
    "last_mlb_update":      None,   # ISO date string of last nightly MLB scores run
    "last_mlb_result":      None,
    "last_nhl_update":      None,   # ISO date string of last nightly NHL scores run
    "last_nhl_result":      None,
    "last_nhl_etl_update":  None,   # ISO date string of last daily NHL feature-matrix refresh
    "last_nhl_etl_result":  None,
    "last_mlb_etl_update":  None,   # ISO date string of last daily MLB feature-matrix refresh
    "last_mlb_etl_result":  None,
    # System 1 — FanDuel weekly sync (Sunday 8 AM UTC)
    "last_fanduel_sync":    None,
    "last_fanduel_result":  None,
    # Morning pre-generation sequence
    "last_fixture_refresh":        None,
    "last_fixture_refresh_result": None,
    "last_mlb_pitcher_fetch":        None,
    "last_mlb_pitcher_fetch_result": None,
    "last_mlb_pitcher_fetch_2":        None,   # 10:30 AM CT second fetch for late lineup confirms
    "last_mlb_pitcher_fetch_2_result": None,
    # System 3 — Mock bet daily jobs
    "last_mock_generate":   None,
    "last_mock_generate_result": None,
    "last_mock_settle":     None,
    "last_mock_settle_result":   None,
    # Afternoon top-up (3 PM CT) — evening game parlays only
    "last_mock_generate_pm":        None,
    "last_mock_generate_pm_result": None,
    # Daily regime classification (8:30 AM CT)
    "last_regime_classify":        None,
    "last_regime_classify_result": None,
    # Creator tier — targeted fetch schedule
    "last_imminent_fetch":        None,   # ISO timestamp of last 45-min imminent fetch
    "last_imminent_fetch_result": None,
    "last_props_11am":            None,   # date string (YYYY-MM-DD) of last 11 AM props run
    "last_props_11am_result":     None,
    "last_props_4pm":             None,   # date string of last 4 PM props run
    "last_props_4pm_result":      None,
    "last_best_lines":            None,   # date string of last daily best-lines run
    "last_best_lines_result":     None,
    # API-Football soccer results fetch (runs once daily at 8 AM UTC)
    "last_soccer_fetch":         None,
    "last_soccer_fetch_result":  None,
    # World Cup odds refresh (daily during tournament: June 11 – July 19 2026)
    "last_wc_odds_refresh":      None,
    "last_wc_odds_refresh_result": None,
    # Alternate lines batch fetch (twice daily via TheOddsAPI sport-level endpoint)
    "last_alt_lines_fetch_morning":           None,   # date string (YYYY-MM-DD)
    "last_alt_lines_fetch_morning_result":    None,
    "last_alt_lines_fetch_getaway":           None,   # 10:45 AM CT day-game near-close
    "last_alt_lines_fetch_getaway_result":    None,
    "last_alt_lines_fetch_afternoon":         None,
    "last_alt_lines_fetch_afternoon_result":  None,
    "last_alt_lines_fetch_clv":              None,   # 5 PM CT near-close capture
    "last_alt_lines_fetch_clv_result":       None,
    # NegBin shadow AM — standalone, runs AFTER alt_lines_morning completes
    "last_negbin_shadow_am":                  None,
    "last_negbin_shadow_am_result":           None,
    # NegBin Shadow B (recalibrated) — parallel A/B, runs AFTER Shadow A
    "last_negbin_recal_shadow_am":            None,
    "last_negbin_recal_shadow_am_result":     None,
    # ATS ungated shadow — scores ALL MLB spread legs (no gates) for discrimination analysis
    "last_ats_ungated_shadow":                None,
    "last_ats_ungated_shadow_result":         None,
    # Weekly personal edge profile refresh (Sunday 9 AM CT)
    "last_personal_edge_refresh":        None,
    "last_personal_edge_refresh_result": None,
    # Weekly signal analysis (Sunday 10 AM CT — read-only, appends to analysis_log.json)
    "last_signal_analysis":      None,
    "last_signal_analysis_result": None,
    # Daily calibration drift check (10:30 AM CT — after overnight scores settle + mock settle)
    "last_calibration_check":        None,
    "last_calibration_check_result": None,
    # Daily scout run (7:00 AM CT — before fixture refresh + pick generation)
    "last_scout_run":        None,
    "last_scout_run_result": None,
    # Injury data refresh (3x daily: 7 AM, 1 PM, 4 PM CT)
    "last_injury_refresh_am":           None,   # YYYY-MM-DD
    "last_injury_refresh_am_result":    None,
    "last_injury_refresh_pm1":          None,   # YYYY-MM-DD (1 PM refresh)
    "last_injury_refresh_pm1_result":   None,
    "last_injury_refresh_pm2":          None,   # YYYY-MM-DD (4 PM refresh)
    "last_injury_refresh_pm2_result":   None,
    # Nightly DB maintenance (3 AM CT — prune old rows, checkpoint WAL)
    "last_db_maintenance":        None,
    "last_db_maintenance_result": None,
    # ── Watchdog verification flags (reset daily at midnight CT) ────────────────
    # True once the watchdog confirms the job actually produced results, not just ran.
    "watchdog_fixture_verified": False,   # fixtures loaded for today
    "watchdog_pitcher_verified": False,   # pitcher data for today's MLB starters
    "watchdog_picks_verified":   False,   # picks cache populated for today
    "watchdog_mocks_verified":       False,   # >= 10 mock bets generated today
    "watchdog_midmorning_verified":  False,   # pick count adequate at 11 AM CT
    "watchdog_last_reset_date":  None,    # CT date string of last midnight reset
    # Settlement CT timestamp (separate from cooldown key — used by watchdog + health check)
    "settle_last_ran_ct":        None,    # datetime in CT, updated after every settle run
    # Settlement cron state (replaces SettlementWatcher thread)
    "settle_cron_running":           False,
    "settle_cron_last_trigger_ct":   None,   # "YYYY-MM-DD HH:MM CT" of last cron fire
    "settle_cron_last_trigger_type": None,   # "primary" | "watchdog"
    # Cashout live-odds polling (every 15 min during game windows when open LIVE bets exist)
    "last_cashout_poll":             None,   # ISO timestamp of last poll
    "last_cashout_poll_result":      None,
}
_stop_event  = threading.Event()
_thread: threading.Thread | None = None

# Hour (CT) at which the nightly NBA odds update runs (3 AM CT)
_NIGHTLY_ODDS_HOUR_CT = 3
# Hour (CT) at which yesterday's NBA ATS results are pulled from BDL (1 AM CT)
_NIGHTLY_ATS_HOUR_CT  = 1
# Hour (CT) for MLB and NHL nightly score pulls (2 AM CT — after overnight games complete)
_NIGHTLY_MLB_HOUR_CT  = 2
_NIGHTLY_NHL_HOUR_CT  = 2
# Hour (CT) for soccer results fetch via API-Football (9 AM CT — after European leagues finish)
_NIGHTLY_SOCCER_HOUR_CT = 9


def get_state() -> dict:
    return dict(_scheduler_state)


def _run_cycle(auto_retrain: bool = True) -> dict:
    db = SessionLocal()
    try:
        result = run_auto_settle(db, days_back=3, auto_retrain=auto_retrain)

        # Live bet resolution: auto-settle PLACED bets and log cashout snapshots
        try:
            import live_monitor as lm
            live_results = lm.resolve_placed_bets(db, auto_settle=True)
            settled_now  = [r for r in live_results if r.get("auto_settled")]
            for item in live_results:
                if item.get("bet_outcome") == "IN_PROGRESS":
                    bet_row = db.query(__import__("database").Bet).filter(
                        __import__("database").Bet.id == item["bet_id"]
                    ).first()
                    if bet_row:
                        cashout = lm.evaluate_cashout(bet_row, item["resolved_legs"], db)
                        lm.log_cashout_recommendation(bet_row, cashout, db)
                        if cashout.get("reasons") and item.get("resolved_legs"):
                            changed = (
                                cashout.get("reasons")  # any recommendation is notable
                            )
                            if changed:
                                print(
                                    f"[LiveMonitor] {item['bet_id'][:8]}… "
                                    f"outcome={item['bet_outcome']} "
                                    f"rec={cashout['recommendation']} "
                                    f"won={item['legs_won']} "
                                    f"rem={item['legs_remaining']}"
                                )
            if settled_now:
                print(f"[LiveMonitor] Auto-settled {len(settled_now)} PLACED bet(s): "
                      + ", ".join(r["auto_settled"] for r in settled_now))
                # Trigger retrain if we settled new personal bets
                if auto_retrain and settled_now:
                    try:
                        from auto_settle import run_auto_settle as _ras
                        _ras(db, days_back=1, auto_retrain=True)
                    except Exception:
                        pass
            result["live_bets_checked"]  = len(live_results)
            result["live_bets_settled"]  = len(settled_now)
        except Exception as exc:
            print(f"[LiveMonitor] Error in live bet resolution: {exc}")
            result["live_monitor_error"] = str(exc)

        return result
    finally:
        db.close()


def _run_nightly_odds_update() -> dict:
    """
    Fetch closing odds for yesterday's games from ESPN and write to DB.
    Covers NBA, MLB, and NHL — safe to call multiple times (only fills NULL rows).
    """
    results = {}
    for sport, func_name in [("NBA", "backfill_nba_nightly"),
                              ("MLB", "backfill_mlb_nightly"),
                              ("NHL", "backfill_nhl_nightly")]:
        try:
            from backfill_lines import backfill_nba_nightly, backfill_mlb_nightly, backfill_nhl_nightly
            func = {"backfill_nba_nightly": backfill_nba_nightly,
                    "backfill_mlb_nightly": backfill_mlb_nightly,
                    "backfill_nhl_nightly": backfill_nhl_nightly}[func_name]
            result = func()
            filled = result.get("filled", 0)
            missing = result.get("missing", 0)
            print(f"[Scheduler] Nightly {sport} odds: filled={filled} "
                  f"missing={missing} date={result.get('date', '?')}")
            results[sport] = result
        except Exception as exc:
            print(f"[Scheduler] Nightly {sport} odds error: {exc}")
            results[sport] = {"error": str(exc)}
    return results


def _now_ct() -> datetime:
    """Current time in America/Chicago."""
    return datetime.now(_CT)


def _should_run_nightly_odds(now: datetime) -> bool:
    """True if it's past _NIGHTLY_ODDS_HOUR_CT today (CT) and we haven't run yet today."""
    ct = _now_ct()
    if ct.hour < _NIGHTLY_ODDS_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_odds_update") != today_str


def _run_nightly_ats_update() -> dict:
    """
    Fetch yesterday's NBA game scores from BallDontLie and update historical.db.
    Runs once daily after _NIGHTLY_ATS_HOUR_UTC UTC.
    """
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        from historical_etl import update_nba_scores_for_date
        result = update_nba_scores_for_date(yesterday)
        print(f"[Scheduler] Nightly NBA ATS: date={yesterday}  "
              f"bdl_fetched={result.get('bdl_fetched', 0)}  "
              f"updated={result.get('updated', 0)}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Nightly ATS update error: {exc}")
        return {"date": yesterday, "error": str(exc)}


def _should_run_nightly_ats(now: datetime) -> bool:
    """True if it's past _NIGHTLY_ATS_HOUR_CT today (CT) and we haven't run yet today."""
    ct = _now_ct()
    if ct.hour < _NIGHTLY_ATS_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_ats_update") != today_str


def _run_nightly_mlb_update() -> dict:
    """
    Fetch yesterday's MLB scores from MLB Stats API and update historical.db.
    Only fills rows with NULL home_score — safe to call multiple times.
    Runs once daily after _NIGHTLY_MLB_HOUR_UTC UTC.
    """
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        from historical_etl import update_mlb_scores_for_date
        result = update_mlb_scores_for_date(yesterday)
        print(f"[Scheduler] Nightly MLB scores: date={yesterday}  "
              f"fetched={result.get('fetched', 0)}  updated={result.get('updated', 0)}")
        # Update Pi ratings after scores are backfilled
        try:
            from pi_ratings import backfill_pi_ratings
            pi_result = backfill_pi_ratings("MLB", "2026-03-27")
            print(f"[Scheduler] Pi ratings updated: {pi_result.get('games_processed', 0)} games")
        except Exception as _pi_exc:
            print(f"[Scheduler] Pi rating update warning: {_pi_exc}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Nightly MLB update error: {exc}")
        return {"date": yesterday, "error": str(exc)}


def _should_run_nightly_mlb(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _NIGHTLY_MLB_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_mlb_update") != today_str


def _run_nightly_nhl_update() -> dict:
    """
    Fetch yesterday's NHL scores from NHL Web API and update historical.db.
    Only fills rows with NULL home_score — safe to call multiple times.
    Also updates ot_so_game flag in betting_lines for OT/SO games.
    Runs once daily after _NIGHTLY_NHL_HOUR_UTC UTC.
    """
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        from historical_etl import update_nhl_scores_for_date
        result = update_nhl_scores_for_date(yesterday)
        print(f"[Scheduler] Nightly NHL scores: date={yesterday}  "
              f"fetched={result.get('fetched', 0)}  updated={result.get('updated', 0)}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Nightly NHL update error: {exc}")
        return {"date": yesterday, "error": str(exc)}


def _should_run_nightly_nhl(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _NIGHTLY_NHL_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_nhl_update") != today_str


# System 1 — FanDuel weekly sync: Sunday (weekday 6) at 8 AM CT
_WEEKLY_SYNC_DOW_CT  = 6   # Sunday
_WEEKLY_SYNC_HOUR_CT = 8

# Morning pre-generation sequence (CT times — DST-safe via ZoneInfo)
_DAILY_FIXTURE_REFRESH_HOUR_CT   = 7   # 7:15 AM CT — refresh fixture odds (minute set in _should_run)
_DAILY_FIXTURE_REFRESH_MINUTE_CT = 15
_DAILY_MLB_PITCHER_HOUR_CT       = 7   # 7:15 AM CT — pull today's MLB probable pitchers
_DAILY_MLB_PITCHER_MINUTE_CT     = 15

# System 3 — Mock bet daily schedule (CT times)
_DAILY_MOCK_GENERATE_HOUR_CT   = 7    # 7:30 AM CT — after fixture refresh
_DAILY_MOCK_GENERATE_MINUTE_CT = 30
_DAILY_MOCK_SETTLE_HOUR_CT     = 9    # 9:00 AM CT — after scores have settled
_AFTERNOON_MOCK_GENERATE_HOUR_CT   = 15   # 3:00 PM CT — evening game top-up
_REGIME_CLASSIFY_HOUR_CT           = 8    # 8:00 AM CT — after fixture refresh + mock generate
_REGIME_CLASSIFY_MINUTE_CT         = 0

# Props and soccer (CT times)
_PROPS_MORNING_HOUR_CT  = 9    # 9:00 AM CT — replaces old 4 AM UTC
_SOCCER_FETCH_HOUR_CT   = 9    # 9:00 AM CT

# Alt lines batch fetch — four daily captures (CT times)
# AM (open) → Getaway (pre-day-game close) → PM (afternoon refresh) → CLV (near-close)
_ALT_LINES_MORNING_HOUR_CT     = 7    # 7:15 AM CT — alongside fixture refresh
_ALT_LINES_MORNING_MINUTE_CT   = 15
_ALT_LINES_GETAWAY_HOUR_CT     = 10   # 10:45 AM CT — near-close for day/getaway games (11:11+ CT starts)
_ALT_LINES_GETAWAY_MINUTE_CT   = 45
_ALT_LINES_AFTERNOON_HOUR_CT   = 14   # 2:30 PM CT — 30 min before PM pick run
_ALT_LINES_AFTERNOON_MINUTE_CT = 30
_ALT_LINES_CLV_HOUR_CT         = 17   # 5:00 PM CT — near-close for early-evening games (6:10+ ET starts)
_ALT_LINES_CLV_MINUTE_CT       = 0

# Hour (CT) for daily feature-matrix ETL refresh (10 AM CT — after overnight scores settle)
_DAILY_ETL_HOUR_CT = 10

# Hour (CT) for daily calibration drift check (10:30 AM CT — after mock settle + ETL)
_CALIBRATION_HOUR_CT   = 10
_CALIBRATION_MINUTE_CT = 30

# Hour (CT) for daily scout run (7:00 AM CT — BEFORE fixture refresh + pick generation)
# Scout uses ESPN APIs independently (not fixtures), so it can run first.
# Grade A at 73% WR is the strongest validated signal — must be available at scoring time.
_SCOUT_HOUR_CT = 7

# Creator tier — targeted fetch schedule (credit-optimized)
# Imminent-games snapshot: every 90 min, 8 AM–10 PM CT (overnight skip already in place)
# 60-min → 90-min saves ~80 credits/day = ~2,080 over remaining billing cycle
_IMMINENT_FETCH_INTERVAL_MINS = 90
# Props: twice daily at these local hours (11 AM + 4 PM)
_PROPS_LOCAL_HOURS = (11, 16)
# Best lines: once daily at 10 AM local
_BEST_LINES_LOCAL_HOUR = 10


def _run_daily_nhl_etl() -> dict:
    """
    Run a quick NHL ETL refresh to keep rolling-stat feature matrix current.
    Fetches current-season schedule + team stats (skips slow pitcher logs).
    Also invalidates the ml_model feature-matrix cache so next prediction rebuilds.

    Runs in a background thread — ETL can take 2+ hours and must not block the
    dispatch cycle (which needs to reach fixture_refresh before 7:15 AM).
    """
    import threading as _threading

    def _bg():
        try:
            from historical_etl import run as etl_run
            etl_run(sports=["NHL"], reset=False, dry_run=False, quick=False)
            try:
                import ml_model
                ml_model._FM_CACHE.pop("NHL", None)
            except Exception:
                pass
            print("[Scheduler] Daily NHL ETL refresh complete — cache invalidated")
        except Exception as exc:
            print(f"[Scheduler] Daily NHL ETL error: {exc}")

    _threading.Thread(target=_bg, daemon=True, name="nhl-etl-bg").start()
    return {"status": "started", "sport": "NHL"}


def _run_daily_mlb_etl() -> dict:
    """
    Run a quick MLB ETL refresh (--quick skips 7-min pitcher game log re-fetch).
    Keeps team batting stats, team stats, and scheduled games current for sub-model.
    Also invalidates the ml_model feature-matrix cache.

    Runs in a background thread — ETL can take 2+ hours and must not block the
    dispatch cycle (which needs to reach fixture_refresh before 7:15 AM).
    """
    import threading as _threading

    def _bg():
        try:
            from historical_etl import run as etl_run
            etl_run(sports=["MLB"], reset=False, dry_run=False, quick=True)
            try:
                import ml_model
                ml_model._FM_CACHE.pop("MLB", None)
            except Exception:
                pass
            print("[Scheduler] Daily MLB ETL refresh complete — cache invalidated")
        except Exception as exc:
            print(f"[Scheduler] Daily MLB ETL error: {exc}")

    _threading.Thread(target=_bg, daemon=True, name="mlb-etl-bg").start()
    return {"status": "started", "sport": "MLB"}


def _should_run_daily_nhl_etl(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _DAILY_ETL_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_nhl_etl_update") != today_str


def _should_run_daily_mlb_etl(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _DAILY_ETL_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_mlb_etl_update") != today_str


# ── System 1: FanDuel weekly sync ─────────────────────────────────────────────

def _run_weekly_fanduel_sync() -> dict:
    """
    Run the FanDuel weekly CSV sync: finds latest fanduel_*.csv, imports
    new settled bets, and backfills cash-out decisions.
    """
    try:
        from fanduel_importer import run_weekly_sync
        result = run_weekly_sync()
        print(f"[Scheduler] FanDuel weekly sync: {result.get('status')}  "
              f"new_bets={result.get('new_bets', 0)}")
        return result
    except Exception as exc:
        print(f"[Scheduler] FanDuel sync error: {exc}")
        return {"status": "error", "message": str(exc)}


def _should_run_weekly_fanduel_sync(now: datetime) -> bool:
    """True on Sunday after _WEEKLY_SYNC_HOUR_CT (CT) if not yet run this week."""
    ct = _now_ct()
    if ct.weekday() != _WEEKLY_SYNC_DOW_CT:
        return False
    if ct.hour < _WEEKLY_SYNC_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_fanduel_sync") != today_str


# ── Weekly personal edge profile refresh (Sunday 9 AM CT) ────────────────────
_PERSONAL_EDGE_DOW_CT  = 6   # Sunday
_PERSONAL_EDGE_HOUR_CT = 9   # 9 AM CT — after mock settle, before signal analysis

def _run_personal_edge_refresh() -> dict:
    """
    Rebuild personal_edge_profile from all resolved leg data.
    Runs every Sunday so new weekly sim results feed Component A.
    """
    try:
        import personal_edge_profile as _pep
        result = _pep.refresh_personal_edge_profiles()
        print(f"[Scheduler] personal_edge refresh → {result}")
        return {**result, "status": "ok"}
    except Exception as exc:
        print(f"[Scheduler] personal_edge refresh error: {exc}")
        return {"status": "error", "message": str(exc)}


def _should_run_personal_edge_refresh(now: datetime) -> bool:
    """True on Sunday after _PERSONAL_EDGE_HOUR_CT (CT) if not yet run this week."""
    ct = _now_ct()
    if ct.weekday() != _PERSONAL_EDGE_DOW_CT:
        return False
    if ct.hour < _PERSONAL_EDGE_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_personal_edge_refresh") != today_str


# ── Weekly automated model retraining (Sunday 3 AM CT) ───────────────────────
_WEEKLY_RETRAIN_DOW_CT  = 6   # Sunday
_WEEKLY_RETRAIN_HOUR_CT = 3   # 3 AM CT — low activity; full week of settled data

def _run_weekly_retrain() -> dict:
    """
    Train a new candidate model on all settled mock_bet_legs, validate on the
    last 7 days of stored predictions, and deploy only if acceptance criteria pass.

    Runs in a background thread so the 30-min dispatch loop is not blocked
    (training can take 60–90 seconds on large datasets).
    """
    import threading as _threading

    def _bg():
        try:
            from database import SessionLocal as _SL
            import weekly_retrain as _wr
            _db = _SL()
            try:
                result = _wr.run_weekly_retrain(_db)
                status = result.get("status", "unknown")
                version = result.get("version", "?")
                auc    = result.get("auc", result.get("candidate_auc"))
                brier  = result.get("brier", result.get("candidate_brier"))
                if status == "deployed":
                    print(f"[Scheduler] Weekly retrain — deployed {version}: "
                          f"AUC {auc:.4f} Brier {brier:.4f}")
                elif status == "rejected":
                    reasons = "; ".join(result.get("rejection_reasons", []))
                    print(f"[Scheduler] Weekly retrain — rejected {version}: {reasons}")
                else:
                    print(f"[Scheduler] Weekly retrain — {status}: {result}")
                _scheduler_state["last_weekly_retrain_result"] = result
            finally:
                _db.close()
        except Exception as exc:
            print(f"[Scheduler] Weekly retrain background error: {exc}")
            _scheduler_state["last_weekly_retrain_result"] = {"error": str(exc)}

    _threading.Thread(target=_bg, daemon=True, name="weekly-retrain-bg").start()
    return {"status": "started", "note": "running in background thread"}


def _should_run_weekly_retrain(now: datetime) -> bool:
    """True on Sunday after 3 AM CT if not yet run this week.

    Checks BOTH in-memory state AND persisted state file to prevent
    double-runs after server restarts.
    """
    ct = _now_ct()
    if ct.weekday() != _WEEKLY_RETRAIN_DOW_CT:
        return False
    if ct.hour < _WEEKLY_RETRAIN_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    # Check in-memory state
    if _scheduler_state.get("last_weekly_retrain") == today_str:
        return False
    # Check persisted state (survives restarts)
    try:
        import weekly_retrain as _wr
        persisted = _wr._load_retrain_state()
        if persisted.get("last_weekly_retrain") == today_str:
            # Sync in-memory to prevent repeated disk reads
            _scheduler_state["last_weekly_retrain"] = today_str
            return False
    except Exception:
        pass
    return True


# ── Weekly signal analysis (Sunday 10 AM CT) ─────────────────────────────────
_SIGNAL_ANALYSIS_DOW_CT  = 6   # Sunday
_SIGNAL_ANALYSIS_HOUR_CT = 10  # 10 AM CT

def _run_weekly_signal_analysis() -> dict:
    """
    Run backend/analysis/signal_analysis.py as a subprocess — read-only.
    Appends one JSON entry to data/analysis_log.json.
    Never imports production modules; safe to run alongside the live server.
    """
    import subprocess as _sp
    import os as _os2
    script = _os2.path.abspath(
        _os2.path.join(_os2.path.dirname(__file__), "analysis", "signal_analysis.py")
    )
    log_path = _os2.path.abspath(
        _os2.path.join(_os2.path.dirname(__file__), "..", "data", "analysis_log.json")
    )
    try:
        result = _sp.run(
            ["python3", script, "--output", log_path],
            capture_output=True, text=True, timeout=120,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        status = "ok" if result.returncode == 0 else "error"
        if status == "ok":
            print(f"[Scheduler] signal_analysis complete → {log_path}")
        else:
            print(f"[Scheduler] signal_analysis error (rc={result.returncode}): {stderr[:200]}")
        return {"status": status, "returncode": result.returncode,
                "stdout": stdout[-500:], "stderr": stderr[-200:]}
    except Exception as exc:
        print(f"[Scheduler] signal_analysis exception: {exc}")
        return {"status": "error", "message": str(exc)}


def _should_run_weekly_signal_analysis(now: datetime) -> bool:
    """True on Sunday after _SIGNAL_ANALYSIS_HOUR_CT (CT) if not yet run this week."""
    ct = _now_ct()
    if ct.weekday() != _SIGNAL_ANALYSIS_DOW_CT:
        return False
    if ct.hour < _SIGNAL_ANALYSIS_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_signal_analysis") != today_str


# ── Daily calibration drift check (10:30 AM CT) ───────────────────────────────

def _run_calibration_check() -> dict:
    """
    Compute per-grade and per-sport calibration drift, then refresh the
    win_prob_calibration table so recommender.calibrated_wp() uses current data.
    Writes to calibration_drift_log.  Never raises — failure is non-fatal.
    """
    result: dict = {}
    try:
        import calibration_tracker as _ct
        from database import engine as _engine, SessionLocal as _SL
        db = _SL()
        try:
            result = _ct.run_calibration_check(_engine, db)
        finally:
            db.close()
        level = result.get("alert_level", "ok")
        worst = result.get("worst_drift_pp", 0.0)
        slices = result.get("slices_checked", 0)
        print(
            f"[Scheduler] Calibration check: level={level} worst={worst:.1f}pp "
            f"slices={slices} auto_disabled={result.get('auto_disabled', False)}"
        )
    except Exception as exc:
        print(f"[Scheduler] Calibration check error: {exc}")
        result["error"] = str(exc)

    # Always refresh win_prob_calibration table regardless of drift check outcome.
    # This keeps calibrated_wp() current as new leg outcomes accumulate.
    _db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
    try:
        from win_prob_calibration import refresh_calibration_table as _refresh_cal
        cal_result = _refresh_cal(_db_path)
        print(f"[Scheduler] win_prob_calibration refreshed: {cal_result}")
        result["wp_calibration_refresh"] = cal_result
    except Exception as exc:
        print(f"[Scheduler] win_prob_calibration refresh error: {exc}")

    # Also rebuild parlay calibration table so get_calibrated_parlay_wp() stays current.
    try:
        from win_prob_calibration import build_parlay_calibration_table as _build_pcal
        pcal_result = _build_pcal(_db_path)
        print(f"[Scheduler] parlay_calibration refreshed: {pcal_result}")
        result["parlay_calibration_refresh"] = pcal_result
    except Exception as exc:
        print(f"[Scheduler] parlay_calibration refresh error: {exc}")

    return result


def _should_run_calibration_check(now: datetime) -> bool:
    """True once daily at or after 10:30 AM CT."""
    ct = _now_ct()
    if ct.hour < _CALIBRATION_HOUR_CT:
        return False
    if ct.hour == _CALIBRATION_HOUR_CT and ct.minute < _CALIBRATION_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_calibration_check") == today_str:
        return False
    # DB evidence: calibration_drift_log rows for today survive a restart
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM calibration_drift_log WHERE run_date = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_calibration_check"] = today_str
            print(f"[Scheduler] Restart detected — calibration_check already ran today ({_n} slices)")
            return False
    except Exception:
        pass
    return True


# ── Brier daily snapshot (fires after each settlement) ──────────────────────

def _snapshot_brier_daily():
    """
    Insert/update today's row in brier_daily_log after settlement.
    Production sources only (prospective, prospective_pm, top_picks_page).
    """
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _con.execute("""
            CREATE TABLE IF NOT EXISTS brier_daily_log (
                log_date TEXT PRIMARY KEY,
                production_brier REAL,
                production_reliability REAL,
                production_resolution REAL,
                production_legs INTEGER,
                production_wr REAL,
                computed_at DATETIME
            )
        """)
        today = _now_ct().strftime("%Y-%m-%d")
        row = _con.execute("""
            SELECT
                ROUND(AVG(POWER(CAST(mbl.win_prob AS REAL)/100.0 -
                    CASE WHEN mbl.leg_result='WIN' THEN 1.0 ELSE 0.0 END, 2)), 6) as brier,
                COUNT(*) as legs,
                ROUND(AVG(CASE WHEN mbl.leg_result='WIN' THEN 100.0 ELSE 0.0 END), 2) as wr
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source IN ('prospective','prospective_pm','top_picks_page')
            AND mbl.leg_result IN ('WIN','LOSS')
            AND COALESCE(mbl.excluded_from_metrics, 0) = 0
            AND mbl.win_prob IS NOT NULL
            AND mb.game_date = ?
        """, (today,)).fetchone()
        if row and row[1] and row[1] > 0:
            with DB_WRITE_LOCK:
                _con.execute("""
                    INSERT OR REPLACE INTO brier_daily_log
                        (log_date, production_brier, production_legs, production_wr, computed_at)
                    VALUES (?, ?, ?, ?, datetime('now'))
                """, (today, row[0], row[1], row[2]))
                _con.commit()
        _con.close()
    except Exception as exc:
        print(f"[BrierSnapshot] Error: {exc}")


# ── Daily scout run (7:00 AM CT — before pick generation) ────────────────────

def _run_daily_scout() -> dict:
    """
    Run the scouting pipeline for all sports with games today.
    Persists ScoutedProp rows to scouted_props table.
    Returns summary dict with counts by sport and grade.
    """
    try:
        import scout.runner as _runner
        from database import SessionLocal as _SL, engine as _engine
        db = _SL()
        try:
            result = _runner.run_daily_scout(_engine, db)
        finally:
            db.close()
        total   = result.get("total_props", 0)
        by_sport = result.get("by_sport", {})
        print(f"[Scheduler] Scout run complete: {total} props — {by_sport}")

        # Backfill scout grades on legs that were generated before scout data existed
        try:
            from placement import backfill_scout_grades_for_today
            _bf = backfill_scout_grades_for_today()
            if _bf:
                result["scout_backfill_updated"] = _bf
                # Rescore composites now that scout grades are attached
                from placement import rescore_composites_for_today
                _rc = rescore_composites_for_today()
                result["composites_rescored"] = _rc.get("updated", 0)
        except Exception as _bf_exc:
            print(f"[Scheduler] Scout backfill error (non-fatal): {_bf_exc}")

        return result
    except Exception as exc:
        print(f"[Scheduler] Scout run error: {exc}")
        return {"error": str(exc)}


def _should_run_daily_scout(now: datetime) -> bool:
    """True once daily at or after 7:00 AM CT (before pick generation)."""
    ct = _now_ct()
    if ct.hour < _SCOUT_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_scout_run") == today_str:
        return False
    # DB evidence: scouted_props rows for today survive a restart.
    # Require ≥ 30 props to guard against partial runs (e.g. only NBA committed
    # before a restart mid-pipeline).
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM scouted_props WHERE scout_date = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n >= 30:
            _scheduler_state["last_scout_run"] = today_str
            print(f"[Scheduler] Restart detected — daily_scout already ran today ({_n} props)")
            return False
    except Exception:
        pass
    return True


# ── Injury data refresh (3x daily) ───────────────────────────────────────────

def _run_injury_refresh() -> dict:
    """Fetch and store game-day injury reports from ESPN + MLB Stats API."""
    try:
        import injury_fetcher as _if
        from database import SessionLocal as _SL
        db = _SL()
        try:
            result = _if.refresh_injuries(db)
        finally:
            db.close()
        total = sum(result.get("records_saved", {}).values())
        print(f"[Scheduler] Injury refresh: {total} records saved for {result.get('game_date')}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Injury refresh error: {exc}")
        return {"error": str(exc)}


def _should_run_injury_refresh(slot: str) -> bool:
    """
    Return True if the given time slot's refresh hasn't run today.
    slot: 'am' (7 AM CT), 'pm1' (1 PM CT), 'pm2' (4 PM CT)
    """
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    _slot_hours = {"am": 7, "pm1": 13, "pm2": 16}
    required_hour = _slot_hours.get(slot, 7)
    if ct.hour < required_hour:
        return False
    state_key = f"last_injury_refresh_{slot}"
    if _scheduler_state.get(state_key) == today_str:
        return False
    # DB evidence: injury_reports with game_date = today and reported_at >= slot start (CT)
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM injury_reports "
            "WHERE game_date = ? "
            "AND CAST(strftime('%H', datetime(reported_at, '-5 hours')) AS INTEGER) >= ?",
            (today_str, required_hour)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state[state_key] = today_str
            print(f"[Scheduler] Restart detected — injury_refresh_{slot} already ran today ({_n} reports)")
            return False
    except Exception:
        pass
    return True


# ── Morning pre-generation sequence ──────────────────────────────────────────

def _run_daily_fixture_refresh() -> dict:
    """
    Refresh today's fixture odds from TheOddsAPI for all active sports.
    Runs at 6 AM UTC — one hour before mock bet generation — so picks are
    built from fresh odds rather than yesterday's stale lines.

    On API failure, the DB fixtures remain as a stale fallback.
    Staleness age is logged so health checks can surface it.
    """
    try:
        from database import SessionLocal, init_db
        from odds_api import fetch_all_fixtures
        init_db()
        db = SessionLocal()
        try:
            result = fetch_all_fixtures(db)
        finally:
            db.close()
        new     = result.get("new", 0)
        updated = result.get("updated", 0)
        print(f"[Scheduler] Fixture refresh: new={new} updated={updated} "
              f"sports={result.get('sports_fetched', 0)}")
        if new == 0 and updated == 0:
            age = fixture_staleness_hours()
            result["using_cached"] = True
            result["cached_age_hours"] = age
            print(f"[Scheduler] WARNING: Fixture refresh returned 0 events — "
                  f"using cached fixtures ({age:.1f}h old)" if age else
                  "[Scheduler] WARNING: Fixture refresh returned 0 events and no cached fixtures found")
        return result
    except Exception as exc:
        age = fixture_staleness_hours()
        print(f"[Scheduler] Fixture refresh error: {exc}. "
              f"Falling back to cached fixtures ({age:.1f}h old)." if age else
              f"[Scheduler] Fixture refresh error: {exc}. No cached fixtures available.")
        return {"error": str(exc), "using_cached": True, "cached_age_hours": age}


def _run_daily_mlb_pitcher_fetch() -> dict:
    """
    Pull today's MLB probable starters from the free MLB Stats API and upsert
    into historical.db so the MLB ATS sub-model has pitcher context for today's
    games when mock bets are generated at 7 AM UTC.
    """
    from datetime import date
    today = date.today().isoformat()
    try:
        from historical_etl import upsert_mlb_today_pitchers
        result = upsert_mlb_today_pitchers(today)
        print(f"[Scheduler] MLB pitcher fetch: date={today} "
              f"games_found={result.get('games_found',0)} "
              f"stats_written={result.get('stats_written',0)}")
        return result
    except Exception as exc:
        print(f"[Scheduler] MLB pitcher fetch error: {exc}")
        return {"date": today, "error": str(exc)}


def _should_run_daily_fixture_refresh(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _DAILY_FIXTURE_REFRESH_HOUR_CT:
        return False
    if ct.hour == _DAILY_FIXTURE_REFRESH_HOUR_CT and ct.minute < _DAILY_FIXTURE_REFRESH_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if (_scheduler_state.get("last_fixture_refresh") or "")[:10] == today_str:
        return False
    # DB evidence: fixtures fetched today survive a restart
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM fixtures WHERE date(fetched_at, '-5 hours') = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_fixture_refresh"] = today_str
            print(f"[Scheduler] Restart detected — fixture_refresh already ran today ({_n} fixtures)")
            return False
    except Exception:
        pass
    return True


def _should_run_daily_mlb_pitcher_fetch(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _DAILY_MLB_PITCHER_HOUR_CT:
        return False
    if ct.hour == _DAILY_MLB_PITCHER_HOUR_CT and ct.minute < _DAILY_MLB_PITCHER_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_mlb_pitcher_fetch") != today_str


def _should_run_mlb_pitcher_fetch_2(now: datetime) -> bool:
    """Second pitcher fetch at 10:30 AM CT — catches late day-of lineup confirms.

    Most evening starts (5–7 PM CT) don't lock probable starters until ~10 AM.
    This second pass runs once per day after the first and picks up whatever
    the 7:15 AM pass missed.
    """
    ct = _now_ct()
    if ct.hour < 10:
        return False
    if ct.hour == 10 and ct.minute < 30:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_mlb_pitcher_fetch_2") != today_str


# ── System 3: Daily mock bet jobs ─────────────────────────────────────────────

def _run_daily_mock_generate() -> dict:
    """
    7:30 AM job: pre-generate today's picks into the in-memory cache,
    then generate paper mock bets.  Doing picks first means the UI
    is instantaneous (<1 s) for the rest of the day.
    """
    picks_result: dict = {}

    # ── Step 1: pre-generate and cache today's picks ─────────────────────────
    try:
        from database import SessionLocal, init_db
        import recommender as rec
        from pydantic import BaseModel

        class _DefaultReq:
            n_picks      = 5
            stake        = 10.0
            max_legs     = 4
            min_legs     = 2
            min_odds     = 2.0
            sort_by      = "win_prob"
            sport_filter = None
            refresh      = True   # always fresh at 7:30 AM

        init_db()
        db = SessionLocal()
        try:
            import main as _main
            resp = _main._build_todays_picks(_DefaultReq(), db)
            _main._set_cached_picks(resp)
            n_picks = len(resp.get("picks") or [])
            n_power = len(resp.get("power_picks") or [])
            picks_result = {"picks_cached": n_picks, "power_picks_cached": n_power}
            print(f"[Scheduler] Picks pre-generated: picks={n_picks} power={n_power}")
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] Picks pre-generate error: {exc}")
        picks_result = {"picks_cache_error": str(exc)}

    # ── Step 2: generate paper mock bets ─────────────────────────────────────
    try:
        from database import SessionLocal, init_db
        import mock_bets as mb
        init_db()
        db = SessionLocal()
        try:
            result = mb.generate_mock_bets(db)
            # Step 3a: legacy adjacent-line exploration from just-generated picks
            try:
                expl = mb.generate_exploration_bets(db, run_id=result.get("run_id"))
                result["exploration_legacy"] = expl
                print(f"[Scheduler] Exploration (legacy): created={expl.get('created', 0)} cushion_legs={expl.get('cushion_legs', 0)}")
            except Exception as _expl_exc:
                result["exploration_legacy"] = {"error": str(_expl_exc)}
                print(f"[Scheduler] Exploration legacy error: {_expl_exc}")
            # Step 3b: tiered exploration (validation / expansion / boundary_testing)
            try:
                expl_tiered = mb.generate_tiered_exploration_bets(db, run_id=result.get("run_id"))
                result["exploration_tiered"] = expl_tiered
                print(
                    f"[Scheduler] Exploration (tiered): created={expl_tiered.get('created', 0)} "
                    f"by_tier={expl_tiered.get('created_by_tier', {})}"
                )
            except Exception as _expl_t_exc:
                result["exploration_tiered"] = {"error": str(_expl_t_exc)}
                print(f"[Scheduler] Exploration tiered error: {_expl_t_exc}")
            # Step 3c: daily sims — full-slate coverage for every game today
            try:
                daily_sims = mb.generate_daily_sims(db)
                result["daily_sims"] = daily_sims
                print(
                    f"[Scheduler] Daily sims: created={daily_sims.get('created', 0)} "
                    f"by_sport={daily_sims.get('by_sport', {})}"
                )
            except Exception as _sims_exc:
                result["daily_sims"] = {"error": str(_sims_exc)}
                print(f"[Scheduler] Daily sims error: {_sims_exc}")
            # NegBin shadow AM removed from here — now runs as standalone
            # dispatch AFTER alt_lines_morning to guarantee full-slate alt-spread
            # odds are available (see _run_negbin_shadow_am / _should_run_negbin_shadow_am).
        finally:
            db.close()
        print(f"[Scheduler] Mock generate: generated={result.get('generated', 0)}")
        result.update(picks_result)
        return result
    except Exception as exc:
        print(f"[Scheduler] Mock generate error: {exc}")
        return {"error": str(exc), **picks_result}


def _run_top_picks_seed() -> dict:
    """
    Seed top_picks_page mock bets from the cached picks.

    Runs as the final step of morning and afternoon chains so that
    top_picks are generated server-side on a schedule — the frontend
    _autoFeedMockBets is kept as a fallback only.  Uses the same picks
    cache that _run_daily_mock_generate populates (Step 1).
    """
    try:
        from database import SessionLocal, init_db
        import mock_bets as mb
        init_db()
        db = SessionLocal()
        try:
            import main as _main
            cached = _main._get_cached_picks()
            cache_picks = cached.get("picks", []) if cached else []
            if not cache_picks:
                print("[Scheduler] top_picks_seed: no cached picks — skipping")
                return {"skipped": True, "reason": "no_cached_picks"}

            result = mb.seed_top_picks_mock_bets(
                cache_picks, db,
                kelly_fraction=0.25,
                bankroll=500.0,
            )
            n_gen = result.get("generated", 0)
            print(f"[Scheduler] top_picks_seed: generated={n_gen}")

            # Exploration from the same run (mirrors the frontend path)
            if n_gen > 0:
                try:
                    expl = mb.generate_exploration_bets(db, run_id=result.get("run_id"))
                    result["exploration"] = expl
                except Exception as _e:
                    result["exploration"] = {"error": str(_e)}
                try:
                    tiered = mb.generate_tiered_exploration_bets(db, run_id=result.get("run_id"))
                    result["exploration_tiered"] = tiered
                except Exception as _e:
                    result["exploration_tiered"] = {"error": str(_e)}

            return result
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] top_picks_seed error: {exc}")
        return {"error": str(exc)}


def _run_daily_mock_settle() -> dict:
    """Settle pending mock bets against available game results."""
    try:
        from database import SessionLocal, init_db
        import mock_bets as mb
        init_db()
        db = SessionLocal()
        try:
            result = mb.settle_mock_bets(db)
        finally:
            db.close()
        print(f"[Scheduler] Mock settle: settled={result.get('settled', 0)}")
        # Record settle timestamp BEFORE regime update so a regime failure
        # never prevents the watchdog from seeing settlement as recent.
        _scheduler_state["settle_last_ran_ct"] = _now_ct()
        # Update today's regime log row with actual win rate + CLV (best-effort)
        try:
            _run_update_regime_settlement()
        except Exception as _re:
            print(f"[Scheduler] Regime settlement update error (non-fatal): {_re}")
        # Update regime A/B log with standard win_rate from settled mock bets (best-effort)
        try:
            db2 = SessionLocal()
            try:
                _run_update_regime_ab_settlement(db2)
            finally:
                db2.close()
        except Exception as _rae:
            print(f"[Scheduler] Regime A/B settlement update error (non-fatal): {_rae}")
        # Settle shadow predictions + refresh shadow calibration (best-effort)
        try:
            _run_shadow_settle()
        except Exception as _se:
            print(f"[Scheduler] Shadow settle error (non-fatal): {_se}")
        # Backfill devig on newly-settled legs (Group F — measure-only, best-effort)
        try:
            from devig import backfill_devig
            dv = backfill_devig(force=False)
            print(f"[Scheduler] Devig backfill: {dv.get('entry_devig', 0)} entry, {dv.get('close_devig', 0)} close")
        except Exception as _dve:
            print(f"[Scheduler] Devig backfill error (non-fatal): {_dve}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Mock settle error: {exc}")
        return {"error": str(exc)}


def _run_shadow_scoring() -> dict:
    """Score today's slate with all registered shadow models + configs (non-blocking, non-live)."""
    # Ensure D-group config shadows are registered (idempotent)
    try:
        from shadow_configs import register_all_configs
        register_all_configs()
    except Exception as _reg_exc:
        print(f"[Scheduler] Shadow config registration warning: {_reg_exc}")

    combined = {}
    # Model shadows
    try:
        from shadow_registry import run_shadow_scoring
        result = run_shadow_scoring()
        scored = result.get("scored", 0)
        print(f"[Scheduler] Shadow model scoring: scored={scored} slots={list(result.get('slots', {}).keys())}")
        combined["model_shadows"] = result
    except Exception as exc:
        print(f"[Scheduler] Shadow model scoring error (non-fatal): {exc}")
        combined["model_shadows"] = {"error": str(exc)}
    # Config shadows
    try:
        from shadow_registry import run_shadow_config_scoring
        cfg_result = run_shadow_config_scoring()
        cfg_scored = cfg_result.get("scored", 0)
        print(f"[Scheduler] Shadow config scoring: scored={cfg_scored} configs={list(cfg_result.get('configs', {}).keys())}")
        combined["config_shadows"] = cfg_result
    except Exception as exc:
        print(f"[Scheduler] Shadow config scoring error (non-fatal): {exc}")
        combined["config_shadows"] = {"error": str(exc)}
    # Selection pass — run ALE-style selection on config shadow predictions
    try:
        from shadow_registry import run_shadow_selection
        sel_result = run_shadow_selection()
        sel_total = sel_result.get("total_selected", 0)
        print(f"[Scheduler] Shadow selection pass: {sel_total} legs selected across configs")
        combined["selection"] = sel_result
    except Exception as exc:
        print(f"[Scheduler] Shadow selection error (non-fatal): {exc}")
        combined["selection"] = {"error": str(exc)}
    return combined


def _run_shadow_settle() -> dict:
    """Settle shadow predictions + refresh per-shadow calibration (leak-safe)."""
    try:
        from shadow_registry import settle_shadow_predictions, refresh_shadow_calibration
        s_result = settle_shadow_predictions()
        c_result = refresh_shadow_calibration()
        settled = s_result.get("settled", 0)
        calibrated = c_result.get("updated", 0)
        print(f"[Scheduler] Shadow settle: settled={settled}, calibration buckets updated={calibrated}")
        return {"settled": settled, "calibration_updated": calibrated}
    except Exception as exc:
        print(f"[Scheduler] Shadow settle error (non-fatal): {exc}")
        return {"error": str(exc)}


def _run_daily_soccer_fetch() -> dict:
    """
    Fetch yesterday's API-Football results for all supported leagues and store
    in the soccer_results table (1 API call).  Runs once daily at 9 AM CT.
    After storing results, refreshes team_soccer_form for today's picks.
    """
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        from soccer_data import fetch_soccer_results, store_soccer_results
        from database import SessionLocal, init_db
        init_db()
        db = SessionLocal()
        try:
            results = fetch_soccer_results(yesterday)
            n = store_soccer_results(results, db)
        finally:
            db.close()
        print(f"[Scheduler] Soccer fetch: date={yesterday}  fixtures={n}")
        # Refresh team form for upcoming fixtures so scoring is current
        _run_soccer_form_refresh()
        return {"date": yesterday, "fixtures_stored": n}
    except Exception as exc:
        print(f"[Scheduler] Soccer fetch error: {exc}")
        return {"date": yesterday, "error": str(exc)}


def _run_soccer_form_refresh() -> None:
    """
    Refresh team_soccer_form for teams with upcoming fixtures today.
    Called after soccer_fetch to keep form data current for pick scoring.
    Runs in under 1 second — queries bets.db only, no API calls.
    """
    try:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(__file__))
        from soccer_backfill import build_team_form
        build_team_form(dry_run=False)
        print("[Scheduler] Soccer form refresh complete")
    except Exception as exc:
        print(f"[Scheduler] Soccer form refresh error: {exc}")


def _should_run_daily_soccer_fetch(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _SOCCER_FETCH_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_soccer_fetch") != today_str


# ── World Cup odds refresh ───────────────────────────────────────────────────
# Runs daily at 8 AM CT during the tournament window (June 11 – July 19 2026).
# Fetches fresh h2h/spreads/totals odds from TheOddsAPI for soccer_fifa_world_cup,
# overwrites data/wc2026_odds_snapshot.json, and invalidates the in-memory cache.
_WC_ODDS_HOUR_CT = 8
_WC_START = "2026-06-09"  # start fetching 2 days before tournament
_WC_END   = "2026-07-19"  # final day


def _run_wc_odds_refresh() -> dict:
    """Fetch latest WC odds from TheOddsAPI and save snapshot."""
    import json as _json
    try:
        from odds_api import fetch_odds
        events = fetch_odds(
            "soccer_fifa_world_cup",
            regions="us,us2,eu",
            markets="h2h,spreads,totals",
        )
        if not events:
            return {"events": 0, "message": "No WC events returned"}

        # Preserve venue data from existing snapshot
        snap_path = os.path.join(os.path.dirname(__file__), "..", "data", "wc2026_odds_snapshot.json")
        venue_data = {}
        try:
            with open(snap_path) as _f:
                old_snap = _json.load(_f)
            for m in old_snap:
                key = (m.get("home_team", ""), m.get("away_team", ""))
                if m.get("venue"):
                    venue_data[key] = {
                        k: m.get(k) for k in ("venue", "venue_city", "venue_state",
                                               "venue_alt_ft", "venue_temp_f")
                    }
        except Exception:
            pass

        # Merge venue data into fresh events
        for ev in events:
            key = (ev.get("home_team", ""), ev.get("away_team", ""))
            if key in venue_data:
                ev.update(venue_data[key])

        with open(snap_path, "w") as _f:
            _json.dump(events, _f, indent=2, default=str)

        # Invalidate WC data cache in ml_model
        try:
            from ml_model import _WC_DATA_CACHE
            import ml_model
            ml_model._WC_DATA_CACHE = None
            ml_model._WC_CACHE_TS = 0.0
        except Exception:
            pass

        bk_count = max(len(ev.get("bookmakers", [])) for ev in events) if events else 0
        print(f"[Scheduler] WC odds refresh: {len(events)} events, up to {bk_count} bookmakers")
        return {"events": len(events), "max_bookmakers": bk_count}
    except Exception as exc:
        print(f"[Scheduler] WC odds refresh error: {exc}")
        return {"error": str(exc)}


def _should_run_wc_odds_refresh(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _WC_ODDS_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    # Only run during tournament window
    if today_str < _WC_START or today_str > _WC_END:
        return False
    return _scheduler_state.get("last_wc_odds_refresh") != today_str


def _should_run_daily_mock_generate(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _DAILY_MOCK_GENERATE_HOUR_CT:
        return False
    if ct.hour == _DAILY_MOCK_GENERATE_HOUR_CT and ct.minute < _DAILY_MOCK_GENERATE_MINUTE_CT:
        return False
    today_ct = ct.date().isoformat()   # "YYYY-MM-DD" in CT
    if (_scheduler_state.get("last_mock_generate", "") or "")[:10] == today_ct:
        return False
    # DB evidence: prospective mock bets generated today survive a restart
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM mock_bets WHERE source = 'prospective' AND game_date = ?",
            (today_ct,)
        ).fetchone()[0]
        _con.close()
        if _n >= 10:
            _scheduler_state["last_mock_generate"] = today_ct
            print(f"[Scheduler] Restart detected — mock_generate already ran today ({_n} bets)")
            return False
    except Exception:
        pass
    # Require fixture_refresh to have run today — use [:10] slice so this
    # stays robust whether the stored value is a date string or ISO timestamp.
    # Pitcher fetch is best-effort: if it hasn't run but fixtures are ready,
    # proceed anyway after 9 AM so mocks aren't blocked all day by a failed
    # pitcher ingest.
    fixture_ran_today  = (_scheduler_state.get("last_fixture_refresh",  "") or "")[:10] == today_ct
    pitcher_ran_today  = (_scheduler_state.get("last_mlb_pitcher_fetch","") or "")[:10] == today_ct
    if not fixture_ran_today:
        return False
    if not pitcher_ran_today and ct.hour < 9:
        # Before 9 AM, wait for pitcher data.  After 9 AM, proceed without it.
        return False
    return True


_SETTLE_COOLDOWN_SECONDS = 25 * 60  # 25 minutes between settlement runs


def _should_run_daily_mock_settle(now: datetime) -> bool:
    """
    Settlement runs every 30-min cycle (independent of other jobs).
    Guards:
      1. Not before 9 AM CT — overnight scores may not be available yet.
      2. 25-minute cooldown — prevents double-runs on back-to-back ticks.
    No dependency on fixture_refresh or pitcher_fetch; pending bets should
    always be resolved as soon as scores are available.
    """
    from datetime import timezone as _tz
    ct = _now_ct()
    if ct.hour < _DAILY_MOCK_SETTLE_HOUR_CT:
        return False
    last = _scheduler_state.get("last_mock_settle")
    if last is None:
        return True
    # Stored as full UTC ISO timestamp (new format).
    # Always compare timezone-aware → timezone-aware to avoid TypeError.
    if isinstance(last, str) and len(last) > 10:
        try:
            last_str = last.rstrip("Z")                      # strip trailing Z
            last_dt  = datetime.fromisoformat(last_str)
            if last_dt.tzinfo is None:                       # naive → assume UTC
                last_dt = last_dt.replace(tzinfo=_tz.utc)
            elapsed  = (datetime.now(_tz.utc) - last_dt).total_seconds()
            return elapsed >= _SETTLE_COOLDOWN_SECONDS
        except Exception:
            return True
    # Legacy date-string stored by old code — eligible immediately
    return True


# ── Afternoon mock bet top-up (3 PM CT) — evening games only ─────────────────

def _run_afternoon_mock_generate() -> dict:
    """
    3 PM CT top-up: generate prospective_pm bets from already-fetched data.

    PM alt_lines fetch runs at 2:30 PM CT (re-enabled with Creator tier upgrade),
    so PM generation at 3 PM CT scores against fresh alt-lines.  The
    'prospective_pm' source tag keeps it separate from the AM 'prospective' run
    in signal_analysis and CLV tracking.

    Exploration bets are NOT regenerated here — they are generated once per day
    in the AM cycle (_run_daily_mock_generate) to avoid duplicating probe data.
    Daily sims ARE re-run here to pick up any late-starting games added since AM.
    """
    try:
        from database import SessionLocal, init_db
        import mock_bets as mb
        init_db()
        db = SessionLocal()
        try:
            result = mb.generate_mock_bets(
                db,
                n_picks = 10,
                source  = "prospective_pm",
            )
            # Daily sims top-up: cover any games not yet simulated (deduped automatically)
            try:
                daily_sims = mb.generate_daily_sims(db)
                result["daily_sims"] = daily_sims
                print(
                    f"[Scheduler] PM daily sims: created={daily_sims.get('created', 0)} "
                    f"by_sport={daily_sims.get('by_sport', {})}"
                )
            except Exception as _sims_exc:
                result["daily_sims"] = {"error": str(_sims_exc)}
                print(f"[Scheduler] PM daily sims error: {_sims_exc}")
            # NegBin shadow PM refresh — A + B on same fixture snapshot
            try:
                from negbin_shadow import (
                    generate_negbin_shadow_picks as _nb_shadow,
                    generate_negbin_recal_shadow_picks as _nb_recal,
                    _get_todays_fixtures as _nb_fixtures,
                )
                _pm_fixtures = _nb_fixtures(today_str)
                _nb_result = _nb_shadow(db, fixtures=_pm_fixtures)
                result["negbin_shadow_pm"] = {"generated": len(_nb_result)}
                print(f"[Scheduler] PM NegBin shadow A: {len(_nb_result)} parlays")
                _nb_recal_result = _nb_recal(db, fixtures=_pm_fixtures)
                result["negbin_recal_shadow_pm"] = {"generated": len(_nb_recal_result)}
                print(f"[Scheduler] PM NegBin shadow B: {len(_nb_recal_result)} parlays")
            except Exception as _nb_exc:
                result["negbin_shadow_pm"] = {"error": str(_nb_exc)}
                print(f"[Scheduler] PM NegBin shadow error: {_nb_exc}")
            # ATS ungated shadow PM refresh
            try:
                from ats_ungated_shadow import generate_ungated_shadow as _ats_ug
                _ats_result = _ats_ug(db)
                result["ats_ungated_pm"] = {"legs_scored": len(_ats_result) if _ats_result else 0}
                print(f"[Scheduler] PM ATS ungated: {len(_ats_result) if _ats_result else 0} legs")
            except Exception as _ats_exc:
                result["ats_ungated_pm"] = {"error": str(_ats_exc)}
                print(f"[Scheduler] PM ATS ungated error: {_ats_exc}")
        finally:
            db.close()
        generated = result.get("generated", 0)
        print(f"[Scheduler] PM mock generate: generated={generated}")

        # Backfill scout grades on legs created during PM generation
        try:
            from placement import backfill_scout_grades_for_today
            _bf = backfill_scout_grades_for_today()
            if _bf:
                result["scout_backfill_updated"] = _bf
                print(f"[Scheduler] PM scout backfill: updated {_bf} legs")
                # Rescore composites now that scout grades are attached
                from placement import rescore_composites_for_today
                _rc = rescore_composites_for_today()
                result["composites_rescored"] = _rc.get("updated", 0)
        except Exception as _bf_exc:
            print(f"[Scheduler] PM scout backfill error: {_bf_exc}")

        return result
    except Exception as exc:
        print(f"[Scheduler] PM mock generate error: {exc}")
        return {"error": str(exc)}


def _should_run_afternoon_mock_generate(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _AFTERNOON_MOCK_GENERATE_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_mock_generate_pm") == today_str:
        return False
    # DB evidence: prospective_pm bets today survive a restart
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM mock_bets WHERE source = 'prospective_pm' AND game_date = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n >= 3:
            _scheduler_state["last_mock_generate_pm"] = today_str
            print(f"[Scheduler] Restart detected — mock_generate_pm already ran today ({_n} bets)")
            return False
    except Exception:
        pass
    return True


# ── Daily player projections (7:10 AM CT — before pick generation) ────────────

_PLAYER_PROJ_HOUR_CT   = 7    # 7:10 AM CT — after pitcher fetch, before picks
_PLAYER_PROJ_MINUTE_CT = 10
_PLAYER_PROJ_PM_HOUR_CT = 14  # 2:00 PM CT — refresh with lineup data before PM picks


def _should_run_player_projections(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _PLAYER_PROJ_HOUR_CT:
        return False
    if ct.hour == _PLAYER_PROJ_HOUR_CT and ct.minute < _PLAYER_PROJ_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_player_projections") == today_str:
        return False
    # DB evidence: player_projections for today survive a restart
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM player_projections WHERE game_date = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n >= 50:
            _scheduler_state["last_player_projections"] = today_str
            print(f"[Scheduler] Restart detected — player_projections already ran today ({_n} rows)")
            return False
    except Exception:
        pass
    return True


def _should_run_player_projections_pm(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _PLAYER_PROJ_PM_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_player_projections_pm") == today_str:
        return False
    return True


def _run_player_projections() -> dict:
    """
    Build daily player projections from free APIs (ESPN, MLB Stats, NHL).
    Stores results in player_projections table for prop bet pipeline.
    """
    try:
        from database import SessionLocal, init_db
        import player_projections as _pp
        init_db()
        db = SessionLocal()
        try:
            result = _pp.run_daily_projections(db)
        finally:
            db.close()
        total = result.get("total_projections", 0)
        print(f"[Scheduler] Player projections: total={total} "
              f"(MLB={result.get('mlb_projections',0)}, "
              f"NBA={result.get('nba_projections',0)}, "
              f"NHL={result.get('nhl_projections',0)})")
        return result
    except Exception as exc:
        print(f"[Scheduler] Player projections error: {exc}")
        return {"error": str(exc)}


# ── Daily regime classification (8:30 AM CT) ─────────────────────────────────

def _run_regime_classification() -> dict:
    """
    Classify today's market regime and write it to market_regime_log.
    Runs regime_classifier.py as a subprocess — never imports production modules.
    """
    import subprocess as _sp
    import os as _os2
    script = _os2.path.abspath(
        _os2.path.join(_os2.path.dirname(__file__), "analysis", "regime_classifier.py")
    )
    try:
        result = _sp.run(
            ["python3", script, "--write-db"],
            capture_output=True, text=True, timeout=60,
        )
        stdout  = result.stdout.strip()
        status  = "ok" if result.returncode == 0 else "error"
        # Extract regime from stdout for state logging
        regime  = "unknown"
        for line in stdout.splitlines():
            if "Regime:" in line:
                regime = line.split("Regime:")[-1].strip().lower()
                break
        if status == "ok":
            print(f"[Scheduler] Regime classified: {regime}")
        else:
            print(f"[Scheduler] Regime error: {result.stderr[:200]}")
        return {"status": status, "regime": regime, "stdout": stdout[-300:]}
    except Exception as exc:
        print(f"[Scheduler] Regime exception: {exc}")
        return {"status": "error", "message": str(exc)}


def _run_update_regime_settlement() -> dict:
    """
    After a settlement cycle, update today's market_regime_log row with
    actual win rate, P&L, and CLV stats.
    """
    import subprocess as _sp
    import os as _os2
    script = _os2.path.abspath(
        _os2.path.join(_os2.path.dirname(__file__), "analysis", "regime_classifier.py")
    )
    try:
        # Use Python to call update_regime_settlement directly
        import sys as _sys
        _sys.path.insert(0, _os2.path.dirname(__file__))
        # Import safely — analysis file, not production
        import importlib
        rc = importlib.import_module("analysis.regime_classifier")
        result = rc.update_regime_settlement()
        n = result.get("n", 0)
        wr = result.get("win_rate")
        print(f"[Scheduler] Regime settlement updated: n={n} wr={wr}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Regime settlement update failed: {exc}")
        return {"error": str(exc)}


def _run_update_regime_ab_settlement(db) -> None:
    """
    After settlement, fill in standard_win_rate/standard_pnl on regime_ab_log rows
    whose outcomes are now known.

    Loops through regime_ab_log rows where standard_win_rate IS NULL and
    standard_pick_ids is set.  For each, looks up the mock_bets rows in that
    JSON array and computes the actual win_rate + pnl from settled status.
    """
    import json as _json
    from sqlalchemy import text as _sqt
    from database import RegimeAbLog

    rows = db.execute(_sqt("""
        SELECT date, standard_pick_ids, regime_n
        FROM regime_ab_log
        WHERE standard_win_rate IS NULL
          AND standard_pick_ids IS NOT NULL
    """)).fetchall()

    for row in rows:
        date_str, pick_ids_json, regime_n = row
        try:
            pick_ids = _json.loads(pick_ids_json or "[]")
        except Exception:
            continue

        real_ids = [pid for pid in pick_ids if pid]
        if not real_ids:
            continue

        # Fetch settled outcomes for these mock_bets
        placeholders = ",".join(f"'{_id}'" for _id in real_ids)
        settled = db.execute(_sqt(f"""
            SELECT id, status, actual_profit, amount
            FROM mock_bets
            WHERE id IN ({placeholders})
              AND status IN ('SETTLED_WIN', 'SETTLED_LOSS')
        """)).fetchall()

        if not settled:
            continue   # not yet settled — check next cycle

        wins  = sum(1 for r in settled if r[1] == "SETTLED_WIN")
        n     = len(settled)
        pnl   = round(sum((r[2] or 0) for r in settled), 2)
        wr    = round(wins / n, 4) if n else None

        db.execute(_sqt("""
            UPDATE regime_ab_log
            SET standard_win_rate = :wr,
                standard_pnl      = :pnl,
                standard_n        = :n
            WHERE date = :d
        """), {"wr": wr, "pnl": pnl, "n": n, "d": date_str})

    with DB_WRITE_LOCK:
        db.commit()


def _should_run_regime_classification(now: datetime) -> bool:
    ct = _now_ct()
    if ct.hour < _REGIME_CLASSIFY_HOUR_CT:
        return False
    if ct.hour == _REGIME_CLASSIFY_HOUR_CT and ct.minute < _REGIME_CLASSIFY_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_regime_classify") != today_str


# ── Alt lines batch fetch (7:45 AM CT + 2:30 PM CT) ─────────────────────────

def _run_alt_lines_fetch() -> dict:
    """
    Batch-fetch alternate_spreads + alternate_totals for 10 sport keys via
    the sport-level OddsAPI endpoint.  Stores results in alt_lines (historical.db).
    Shared worker for both the morning and afternoon schedule slots.
    Skipped automatically if credits_remaining < 500 (THRESHOLD_EMERGENCY).
    """
    try:
        from creator_tier import fetch_alt_lines_batch
        result = fetch_alt_lines_batch()
        skipped = result.get("skipped_reason")
        if skipped:
            print(f"[Scheduler] Alt lines fetch skipped: {skipped}")
        else:
            print(
                f"[Scheduler] Alt lines fetch: "
                f"events={result.get('events_found', 0)} "
                f"rows={result.get('rows_inserted', 0)} "
                f"budget={result.get('credits_budget_status')}"
            )
        return result
    except Exception as exc:
        print(f"[Scheduler] Alt lines fetch error: {exc}")
        return {"error": str(exc)}


def _run_alt_lines_fetch_pm() -> dict:
    """Chain-compatible wrapper: fetch PM alt-lines if not already done today."""
    now = _now_ct()
    if not _should_run_alt_lines_afternoon(now):
        print("[Chain] alt_lines_afternoon already ran today — skipping")
        return {"skipped": True, "reason": "already_ran_today"}
    result = _run_alt_lines_fetch()
    _scheduler_state["last_alt_lines_fetch_afternoon"] = now.strftime("%Y-%m-%d")
    _scheduler_state["last_alt_lines_fetch_afternoon_result"] = result
    return result


def _should_run_alt_lines_morning(now: datetime) -> bool:
    """True if it's past 7:15 AM CT today and the morning slot hasn't run yet."""
    ct = _now_ct()
    if ct.hour < _ALT_LINES_MORNING_HOUR_CT:
        return False
    if ct.hour == _ALT_LINES_MORNING_HOUR_CT and ct.minute < _ALT_LINES_MORNING_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_alt_lines_fetch_morning") == today_str:
        return False
    # DB evidence: alt_lines fetched today (by fetched_at CT) survive a restart.
    # Must check fetched_at, NOT game_date — soccer alt_lines are pre-fetched days
    # in advance with future game_date values, which would cause false positives.
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM alt_lines WHERE date(fetched_at, '-5 hours') = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_alt_lines_fetch_morning"] = today_str
            print(f"[Scheduler] Restart detected — alt_lines_morning already ran today ({_n} rows)")
            return False
    except Exception:
        pass
    return True


def _should_run_alt_lines_getaway(now: datetime) -> bool:
    """True if it's past 10:45 AM CT today and the getaway slot hasn't run yet.

    Near-close capture for day/getaway games (starts ~11:11 AM CT / 12:11 ET earliest).
    At 10:45 AM CT, lands ~26 min before the earliest day-game first pitches and
    ~75 min before the noon-start bulk. Closes the getaway-game CLV gap where
    these games otherwise fall back to the 7:15 AM (open) snapshot.
    """
    ct = _now_ct()
    if ct.hour < _ALT_LINES_GETAWAY_HOUR_CT:
        return False
    if ct.hour == _ALT_LINES_GETAWAY_HOUR_CT and ct.minute < _ALT_LINES_GETAWAY_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_alt_lines_fetch_getaway") == today_str:
        return False
    # DB evidence: check if getaway alt_lines already fetched today (survives restart)
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        _con = _sq.connect(_db)
        # Getaway fetch = rows fetched between 10:30 AM and 12:00 PM CT today
        _n = _con.execute(
            "SELECT COUNT(*) FROM alt_lines"
            " WHERE date(fetched_at, '-5 hours') = ?"
            "   AND (CAST(strftime('%%H', fetched_at, '-5 hours') AS INTEGER) * 60"
            "      + CAST(strftime('%%M', fetched_at, '-5 hours') AS INTEGER)) BETWEEN 630 AND 720",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_alt_lines_fetch_getaway"] = today_str
            print(f"[Scheduler] Restart detected — alt_lines_getaway already ran today ({_n} rows)")
            return False
    except Exception:
        pass
    return True


def _should_run_alt_lines_afternoon(now: datetime) -> bool:
    """True if the afternoon alt-line slot hasn't run yet today.

    Re-enabled after Creator tier upgrade (100k credits/month).
    Cost: ~70 credits/day — negligible against 100k budget.
    Ensures PM mock generation scores against fresh alt-lines instead of
    7-hour-stale AM snapshot, allowing novel (non-deduped) PM picks.

    NOTE: Time gating is handled by the afternoon chain (fires at 2 PM CT).
    This function only checks "already ran today" — no hour/minute gate needed.
    """
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_alt_lines_fetch_afternoon") == today_str:
        return False
    # DB evidence: check if afternoon alt_lines already fetched today (survives restart)
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        _con = _sq.connect(_db)
        # Afternoon fetch = rows fetched after noon CT today
        _n = _con.execute(
            "SELECT COUNT(*) FROM alt_lines"
            " WHERE date(fetched_at, '-5 hours') = ?"
            "   AND CAST(strftime('%%H', fetched_at, '-5 hours') AS INTEGER) >= 12",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_alt_lines_fetch_afternoon"] = today_str
            print(f"[Scheduler] Restart detected — alt_lines_afternoon already ran today ({_n} rows)")
            return False
    except Exception:
        pass
    return True


def _should_run_alt_lines_clv(now: datetime) -> bool:
    """True if it's past 5:00 PM CT today and the CLV slot hasn't run yet.

    Near-close alt-line capture for CLV computation. At 5:00 PM CT (6:00 PM ET),
    this lands ~40 min before the earliest MLB first pitches (typically 6:40 PM ET)
    and ~10 min before early-start games (6:10 PM ET). Provides a genuine
    near-close snapshot for afternoon/early-evening games.

    Night games (8-10 PM ET starts) still gap — see notes in code.
    """
    ct = _now_ct()
    if ct.hour < _ALT_LINES_CLV_HOUR_CT:
        return False
    if ct.hour == _ALT_LINES_CLV_HOUR_CT and ct.minute < _ALT_LINES_CLV_MINUTE_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_alt_lines_fetch_clv") == today_str:
        return False
    # DB evidence: check if CLV alt_lines already fetched today (survives restart)
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        _con = _sq.connect(_db)
        # CLV fetch = rows fetched after 4:30 PM CT today
        _n = _con.execute(
            "SELECT COUNT(*) FROM alt_lines"
            " WHERE date(fetched_at, '-5 hours') = ?"
            "   AND (CAST(strftime('%%H', fetched_at, '-5 hours') AS INTEGER) * 60"
            "      + CAST(strftime('%%M', fetched_at, '-5 hours') AS INTEGER)) >= 990",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_alt_lines_fetch_clv"] = today_str
            print(f"[Scheduler] Restart detected — alt_lines_clv already ran today ({_n} rows)")
            return False
    except Exception:
        pass
    return True


# ── NegBin shadow AM — runs AFTER alt_lines_morning guarantees full-slate odds ──

def _should_run_negbin_shadow_am() -> bool:
    """True if alt_lines_morning has completed today and shadow hasn't run yet.

    Dependency: alt_lines_morning must have completed (full-slate alt-spread
    odds landed in historical.db) before shadow generation. This ensures the
    shadow sees ~all MLB games, not just the 3-5 day-ahead west-coast games.
    """
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_negbin_shadow_am") == today_str:
        return False
    # Gate: alt_lines_morning must have completed today
    if _scheduler_state.get("last_alt_lines_fetch_morning") != today_str:
        return False
    return True


def _run_negbin_shadow_am() -> dict:
    """Generate NegBin shadow picks for today (AM pass, post-alt-lines).

    Runs BOTH Shadow A (uncalibrated) and Shadow B (recalibrated) on the
    exact same fixture snapshot — guarantees identical slates for A/B comparison.
    """
    try:
        from database import SessionLocal, init_db
        from negbin_shadow import (
            generate_negbin_shadow_picks,
            generate_negbin_recal_shadow_picks,
            _get_todays_fixtures,
        )
        from datetime import datetime as _dt
        init_db()
        db = SessionLocal()
        try:
            today_str = _dt.now(_CT).strftime("%Y-%m-%d")

            # Snapshot fixtures ONCE — both shadows see the exact same slate
            fixtures = _get_todays_fixtures(today_str)

            # Shadow A (uncalibrated control)
            result_a = generate_negbin_shadow_picks(db, today_str=today_str,
                                                     fixtures=fixtures)
            count_a = len(result_a) if result_a else 0
            print(f"[Scheduler] NegBin shadow A AM: {count_a} parlays")

            # Shadow B (recalibrated) — same fixtures, same slate
            result_b = generate_negbin_recal_shadow_picks(db, today_str=today_str,
                                                           fixtures=fixtures)
            count_b = len(result_b) if result_b else 0
            print(f"[Scheduler] NegBin shadow B AM: {count_b} parlays")

            # Stamp Shadow B as done too
            _scheduler_state["last_negbin_recal_shadow_am"] = today_str

            return {"shadow_a": count_a, "shadow_b": count_b}
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] NegBin shadow AM error: {exc}")
        return {"error": str(exc)}


# ── ATS ungated shadow — scores ALL MLB spread legs (no gates) ──

def _should_run_ats_ungated_shadow() -> bool:
    """True if alt_lines_morning completed today and ungated shadow hasn't run."""
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_ats_ungated_shadow") == today_str:
        return False
    if _scheduler_state.get("last_alt_lines_fetch_morning") != today_str:
        return False
    return True


def _run_ats_ungated_shadow() -> dict:
    """Score all MLB spread legs through ATS model with no gates."""
    try:
        from database import SessionLocal, init_db
        from ats_ungated_shadow import generate_ungated_shadow
        init_db()
        db = SessionLocal()
        try:
            legs = generate_ungated_shadow(db)
            count = len(legs) if legs else 0
            print(f"[Scheduler] ATS ungated shadow: {count} legs scored")
            return {"legs_scored": count}
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] ATS ungated shadow error: {exc}")
        return {"error": str(exc)}


# ── Creator tier: targeted imminent-game fetch (every 60 min, 8 AM–11 PM local) ──

def _run_imminent_fetch() -> dict:
    """Fetch line snapshots for games starting within 6 hours."""
    try:
        from creator_tier import fetch_imminent_games_odds
        result = fetch_imminent_games_odds()
        skipped = result.get("skipped_reason")
        if skipped:
            print(f"[Scheduler] Imminent fetch skipped: {skipped}")
        else:
            print(f"[Scheduler] Imminent fetch: sports={result.get('imminent_sports',[])} "
                  f"events={result.get('events_captured',0)} "
                  f"rows={result.get('rows_inserted',0)}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Imminent fetch error: {exc}")
        return {"error": str(exc)}


def _should_run_imminent_fetch(now: datetime) -> bool:
    """True at 3 daily credit-budget windows: 7:15 AM, 2:30 PM, and 5 PM CT.

    Credit-save mode (3x/day):
      AM slot:  7:00–7:29 CT — runs before 7:30 AM mock generation.
      PM slot:  14:00–14:59 CT — runs before 3:00 PM picks regen.
      CLV slot: 17:00–17:29 CT — closing-line snapshot ~30 min before
                earliest MLB game start (typically 6:10 PM CT).
                Cost: ~3-6 credits/day. Used as the true closing line
                for CLV computation at settlement time.

    Each window fires at most once per calendar day (CT) using per-slot flags.
    Wide windows ensure a 30-min scheduler cycle always lands inside them.
    """
    ct     = _now_ct()
    today  = ct.strftime("%Y-%m-%d")
    hour   = ct.hour
    minute = ct.minute

    # AM slot: 7:00–7:29 CT (widened from 7:15–7:19 so scheduler always catches it)
    if hour == 7 and minute < 30:
        if _scheduler_state.get("last_imminent_fetch_am_date") != today:
            return True

    # PM slot: 14:00–14:59 CT (widened from 14:30–14:34)
    if hour == 14:
        if _scheduler_state.get("last_imminent_fetch_pm_date") != today:
            return True

    # CLV slot: 17:00–17:29 CT — closing-line capture for CLV
    if hour == 17 and minute < 30:
        if _scheduler_state.get("last_imminent_fetch_clv_date") != today:
            return True

    return False


def _run_props_fetch(slot: str) -> dict:
    """
    Fetch player props for same-day games only.
    slot: "11am" or "4pm" — used for state-key tracking.
    """
    try:
        from creator_tier import fetch_player_props
        results = {}
        for sk in ["baseball_mlb", "basketball_nba", "icehockey_nhl"]:
            results[sk] = fetch_player_props(sk, hours_ahead=12)
        total = sum(r.get("props_upserted", 0) for r in results.values())
        print(f"[Scheduler] Props fetch ({slot}): total_props={total}")
        return {"status": "ok", "slot": slot, "total_props": total, "by_sport": results}
    except Exception as exc:
        print(f"[Scheduler] Props fetch ({slot}) error: {exc}")
        return {"error": str(exc)}


def _run_best_lines_daily() -> dict:
    """Fetch best lines across all active sports once per day."""
    try:
        from creator_tier import fetch_best_lines_daily
        result = fetch_best_lines_daily()
        print(f"[Scheduler] Best lines daily: sports={result.get('sport_keys_fetched',[])} "
              f"lines={result.get('total_lines',0)}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Best lines daily error: {exc}")
        return {"error": str(exc)}


def _should_run_props(slot: str) -> bool:
    """Slot 11am (hour 11) or 4pm (hour 16). Run once per day per slot (CT)."""
    target = 11 if slot == "11am" else 16
    ct = _now_ct()
    if ct.hour < target:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get(f"last_props_{slot}") != today_str


def _should_run_best_lines() -> bool:
    ct = _now_ct()
    if ct.hour < _BEST_LINES_LOCAL_HOUR:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get("last_best_lines") != today_str


# ── Watchdog verification helpers ────────────────────────────────────────────
# Each returns (ok: bool, detail: str) so callers can log + expose via health API.

def _verify_fixtures_today() -> tuple[bool, str]:
    """Fixtures loaded for today CT in bets.db."""
    try:
        import sqlite3 as _sq
        _BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        con = _sq.connect(_BETS_DB)
        today_str = _now_ct().strftime("%Y-%m-%d")
        n = con.execute(
            "SELECT COUNT(*) FROM fixtures WHERE game_date_ct = ?",
            (today_str,)
        ).fetchone()[0]
        con.close()
        ok = n >= 5
        return ok, f"{n}_fixtures"
    except Exception as e:
        return False, f"error:{e}"


def fixture_staleness_hours() -> float | None:
    """
    Return how many hours ago fixtures were last fetched, or None on error.
    Used by the health endpoint and recommender to detect stale odds.
    """
    try:
        import sqlite3 as _sq
        from datetime import datetime as _dt2, timezone as _tz
        _BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        con = _sq.connect(_BETS_DB)
        row = con.execute("SELECT MAX(fetched_at) FROM fixtures").fetchone()
        con.close()
        if not row or not row[0]:
            return None
        last = _dt2.fromisoformat(str(row[0])).replace(tzinfo=_tz.utc)
        age  = (_dt2.now(_tz.utc) - last).total_seconds() / 3600
        return round(age, 2)
    except Exception:
        return None


def _verify_pitcher_today() -> tuple[bool, str]:
    """Probable starters written for today's MLB games (cross-checked with fixtures)."""
    try:
        import sqlite3 as _sq, json as _json
        _HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        _BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        today_str = _now_ct().strftime("%Y-%m-%d")

        # Get fixture IDs for today's MLB games from bets.db
        bcon = _sq.connect(_BETS_DB)
        today_fixture_ids = {r[0] for r in bcon.execute(
            """SELECT id FROM fixtures
               WHERE sport_key LIKE '%mlb%'
               AND game_date_ct = ?""",
            (today_str,)
        ).fetchall()}
        bcon.close()

        if not today_fixture_ids:
            # Fallback: just verify pitcher fetch ran today via scheduler state
            ran_today = (_scheduler_state.get("last_mlb_pitcher_fetch") == today_str)
            return ran_today, "from_scheduler_state"

        # Count team_stats rows with non-empty probable pitcher for today's game IDs
        hcon = _sq.connect(_HIST_DB)
        rows = hcon.execute(
            "SELECT game_id, stats_json FROM team_stats WHERE sport='MLB'"
        ).fetchall()
        hcon.close()

        count = 0
        for game_id, sj in rows:
            try:
                s = _json.loads(sj) if sj else {}
                hp = s.get("home_probable_pitcher") or ""
                ap = s.get("away_probable_pitcher") or ""
                if (hp.strip() or ap.strip()):
                    count += 1
            except Exception:
                pass

        # Fallback to scheduler state if DB count is implausible
        ran_today = (_scheduler_state.get("last_mlb_pitcher_fetch") == today_str)
        ok = ran_today or count >= 2
        return ok, f"{min(count, 99)}_starters_in_db"
    except Exception as e:
        return False, f"error:{e}"


def _verify_picks_published() -> tuple[bool, str]:
    """Picks cache has valid content for today CT."""
    try:
        import json as _json
        _CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "picks_cache.json")
        today_str = _now_ct().strftime("%Y-%m-%d")
        if not os.path.exists(_CACHE):
            return False, "cache_missing"
        with open(_CACHE) as _f:
            payload = _json.load(_f)
        if str(payload.get("date", ""))[:10] != today_str:
            return False, "cache_stale"
        picks = payload.get("picks", {})
        total = sum(
            len(picks.get(k, []))
            for k in ("anchor", "core", "mixed", "tier_b", "tier_c", "tier_d")
        )
        if total == 0:
            return False, "zero_picks"
        return True, f"{total}_picks"
    except Exception as e:
        return False, f"error:{e}"


def _verify_mock_generated() -> tuple[bool, str]:
    """At least 10 prospective mock bets generated for today CT."""
    try:
        import sqlite3 as _sq
        _BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        con = _sq.connect(_BETS_DB)
        today_str = _now_ct().strftime("%Y-%m-%d")
        n = con.execute(
            """SELECT COUNT(*) FROM mock_bets
               WHERE game_date = ?
               AND source IN ('prospective','prospective_pm','top_picks_page','forced_generation')""",
            (today_str,)
        ).fetchone()[0]
        con.close()
        return n >= 10, f"{n}_bets"
    except Exception as e:
        return False, f"error:{e}"


def _midmorning_picks_adequate() -> tuple[bool, str]:
    """
    11 AM CT safety check: verify pick count is adequate relative to today's fixture slate.
    Returns (ok, detail). Fires once per day at/after 11 AM CT.

    Expected picks = max(3, fixture_count // 4).
    If actual < 50% of expected, regenerate now — before first pitches.
    """
    try:
        import json as _json
        from database import SessionLocal, Fixture
        from datetime import timezone as _tz

        today_str = _now_ct().strftime("%Y-%m-%d")
        now_utc   = datetime.now(_tz.utc)

        # Count today's fixtures using CT game date (DST-correct)
        db = SessionLocal()
        try:
            today_fixtures = [
                f for f in db.query(Fixture).filter(
                    Fixture.game_date_ct == today_str
                ).all()
            ]
            fixtures_today = len(today_fixtures)
        finally:
            db.close()

        expected_picks = max(3, fixtures_today // 4)

        # Read cached picks
        _CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "picks_cache.json")
        if not os.path.exists(_CACHE):
            return False, f"cache_missing(fixtures={fixtures_today},expected={expected_picks})"
        with open(_CACHE) as _f:
            payload = _json.load(_f)
        if str(payload.get("date", ""))[:10] != today_str:
            return False, f"cache_stale(fixtures={fixtures_today},expected={expected_picks})"
        picks = payload.get("picks", {})
        total = sum(len(picks.get(k, [])) for k in ("anchor", "core", "mixed", "tier_b", "tier_c", "tier_d"))

        ok = total >= expected_picks * 0.5
        return ok, f"picks={total},fixtures={fixtures_today},expected={expected_picks}"
    except Exception as e:
        return False, f"error:{e}"


def _verify_settle_recent() -> tuple[bool, str]:
    """Settlement ran within the last 35 minutes."""
    last = _scheduler_state.get("settle_last_ran_ct")
    if last is None:
        return False, "never_ran"
    try:
        elapsed = (_now_ct() - last).total_seconds()
        mins = int(elapsed / 60)
        ok = elapsed < 35 * 60
        return ok, f"{mins}min_ago"
    except Exception as e:
        return False, f"error:{e}"


def _run_db_maintenance() -> dict:
    """Nightly DB maintenance: prune old rows, checkpoint WAL.

    Retention tiers:
      - Operational logs (settle_log, scheduler_runs, injury_reports): 14 days
      - Feature data (player_projections): 30 days
      - Analytical / CLV data (line_movement_log): 90 days
      - NEVER pruned: signal_conflict_outcomes, shadow_predictions,
        shadow_calibration, mock_bets, mock_bet_legs, fixtures
    """
    import sqlite3
    result = {}

    # ── bets.db ──────────────────────────────────────────────────────────────
    try:
        bets_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        bets_path = os.path.normpath(bets_path)
        con = sqlite3.connect(bets_path, timeout=30)
        con.execute("PRAGMA busy_timeout = 10000")
        total_pruned = 0

        # (table, column, retention_days, is_date_only)
        _PRUNE_RULES = [
            ("settle_log",          "settled_at",  14, False),
            ("scheduler_runs",      "run_at",      14, False),
            ("injury_reports",      "game_date",   14, True),
            ("player_projections",  "game_date",   30, True),
            ("line_movement_log",   "fetched_at",  90, False),
            # signal_conflict_outcomes: EXCLUDED — analytical, needs 200+ pairs
            # shadow_predictions:       EXCLUDED — tripwire accumulation in progress
            # shadow_calibration:       EXCLUDED — tripwire accumulation in progress
            # mock_bets / mock_bet_legs: EXCLUDED — permanent record
        ]

        for tbl, col, days, date_only in _PRUNE_RULES:
            try:
                if date_only:
                    con.execute(f"DELETE FROM {tbl} WHERE {col} < date('now', '-{days} days')")
                else:
                    con.execute(f"DELETE FROM {tbl} WHERE {col} < datetime('now', '-{days} days')")
                pruned = con.execute("SELECT changes()").fetchone()[0]
                if pruned:
                    total_pruned += pruned
                    result[f"pruned_{tbl}"] = pruned
            except Exception as exc:
                result[f"error_{tbl}"] = str(exc)

        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        result["bets_pruned_total"] = total_pruned
    except Exception as e:
        result["bets_error"] = str(e)

    # ── historical.db ────────────────────────────────────────────────────────
    try:
        hist_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        hist_path = os.path.normpath(hist_path)
        con = sqlite3.connect(hist_path, timeout=30)
        con.execute("PRAGMA busy_timeout = 10000")
        con.execute("DELETE FROM player_props WHERE game_date < date('now', '-14 days')")
        pp_pruned = con.execute("SELECT changes()").fetchone()[0]
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        result["pruned_player_props"] = pp_pruned
    except Exception as e:
        result["historical_error"] = str(e)

    print(f"[DB-Maintenance] Pruned: {result}")
    return result


def _run_watchdog() -> None:
    """
    Post-cycle watchdog: verify critical morning jobs completed and auto-recover
    if they didn't. Only active during daytime hours (7 AM – 11 PM CT).

    Checks (and recovers):
      1. Fixtures loaded for today        (required by 7:30 AM CT)
      2. Pitcher starters available       (required by 7:30 AM CT)
      3. Picks cache published            (required by 8 AM CT — verifies 7:30 AM mock gen)
      4. Mock bets generated              (required by 9 AM CT)
      5. Pick count adequate for slate    (11 AM CT safety — before first pitches)
      6. Settlement health                (runs every cycle; overdue > 35 min)

    Verification flags are reset once per day at midnight CT.
    """
    ct      = _now_ct()
    hour_ct = ct.hour
    today_str = ct.strftime("%Y-%m-%d")

    # Only run watchdog during active hours (7 AM – 11 PM CT)
    if hour_ct < 7 or hour_ct >= 23:
        return

    # ── Daily reset at midnight CT ──────────────────────────────────────────
    if _scheduler_state.get("watchdog_last_reset_date") != today_str:
        _scheduler_state["watchdog_fixture_verified"] = False
        _scheduler_state["watchdog_pitcher_verified"] = False
        _scheduler_state["watchdog_picks_verified"]   = False
        _scheduler_state["watchdog_mocks_verified"]     = False
        _scheduler_state["watchdog_midmorning_verified"] = False
        _scheduler_state["watchdog_last_reset_date"]  = today_str
        # Reset second pitcher fetch so it re-runs each calendar day
        _scheduler_state["last_mlb_pitcher_fetch_2"]  = None
        print("[Watchdog] Daily verification flags reset")

    # ── CHECK 1: Fixtures ──────────────────────────────────────────────────
    # Fixture refresh fires at 7:15 AM; verify on the 7:30 AM tick onward.
    if (hour_ct > 7 or (hour_ct == 7 and ct.minute >= 30)) and not _scheduler_state.get("watchdog_fixture_verified"):
        ok, detail = _verify_fixtures_today()
        if ok:
            _scheduler_state["watchdog_fixture_verified"] = True
            print(f"[Watchdog] ✅ Fixtures verified ({detail})")
        else:
            print(f"[Watchdog] ⚠️  Fixtures not verified ({detail}) — re-running fixture refresh")
            try:
                _run_daily_fixture_refresh()
                _scheduler_state["last_fixture_refresh"] = today_str
            except Exception as _e:
                print(f"[Watchdog] Fixture re-run failed: {_e}")

    # ── CHECK 2: Pitcher data ──────────────────────────────────────────────
    # Pitcher fetch fires at 7:15 AM; verify on the 7:30 AM tick onward.
    if (hour_ct > 7 or (hour_ct == 7 and ct.minute >= 30)) and not _scheduler_state.get("watchdog_pitcher_verified"):
        ok, detail = _verify_pitcher_today()
        if ok:
            _scheduler_state["watchdog_pitcher_verified"] = True
            print(f"[Watchdog] ✅ Pitcher data verified ({detail})")
        else:
            print(f"[Watchdog] ⚠️  Pitcher data not verified ({detail}) — re-running pitcher fetch")
            try:
                _run_daily_mlb_pitcher_fetch()
                _scheduler_state["last_mlb_pitcher_fetch"] = today_str
            except Exception as _e:
                print(f"[Watchdog] Pitcher re-run failed: {_e}")

    # ── CHECK 3: Picks cache ───────────────────────────────────────────────
    if hour_ct >= 8 and not _scheduler_state.get("watchdog_picks_verified"):
        ok, detail = _verify_picks_published()
        if ok:
            _scheduler_state["watchdog_picks_verified"] = True
            print(f"[Watchdog] ✅ Picks cache verified ({detail})")
        else:
            print(f"[Watchdog] ⚠️  Picks not published ({detail}) — triggering background regen")
            try:
                import threading as _threading
                import main as _main
                _t = _threading.Thread(
                    target=_main._background_regen,
                    args=({"n_picks": 5, "stake": 10.0, "max_legs": 4,
                           "min_legs": 2, "min_odds": 2.0, "refresh": True},),
                    daemon=True,
                )
                _t.start()
            except Exception as _e:
                print(f"[Watchdog] Picks regen failed: {_e}")

    # ── CHECK 4: Mock bets generated ───────────────────────────────────────
    if hour_ct >= 9 and not _scheduler_state.get("watchdog_mocks_verified"):
        ok, detail = _verify_mock_generated()
        if ok:
            _scheduler_state["watchdog_mocks_verified"] = True
            print(f"[Watchdog] ✅ Mock bets verified ({detail})")
        else:
            print(f"[Watchdog] ⚠️  Mock bets low ({detail}) — re-running mock generate")
            try:
                result = _run_daily_mock_generate()
                _scheduler_state["last_mock_generate"] = today_str
                print(f"[Watchdog] Mock regen result: {result.get('generated', 0)} generated")
            except Exception as _e:
                print(f"[Watchdog] Mock regen failed: {_e}")

    # ── CHECK 5: 11 AM pick-count safety ──────────────────────────────────
    # Before first pitches: if pick count is abnormally low for the day's slate,
    # regenerate now so users have qualified picks before games start.
    # Runs once per day (guarded by watchdog_midmorning_verified flag).
    if hour_ct >= 11 and not _scheduler_state.get("watchdog_midmorning_verified"):
        ok, detail = _midmorning_picks_adequate()
        if ok:
            _scheduler_state["watchdog_midmorning_verified"] = True
            print(f"[Watchdog] ✅ Midmorning pick count adequate ({detail})")
        else:
            print(f"[Watchdog] ⚠️  Low pick count at 11 AM ({detail}) — triggering regeneration")
            try:
                import threading as _threading
                import main as _main
                _t = _threading.Thread(
                    target=_main._background_regen,
                    args=({"n_picks": 5, "stake": 10.0, "max_legs": 4,
                           "min_legs": 2, "min_odds": 2.0, "refresh": True},),
                    daemon=True,
                )
                _t.start()
                _scheduler_state["watchdog_midmorning_verified"] = True
                print(f"[Watchdog] Midmorning picks regen started")
            except Exception as _e:
                print(f"[Watchdog] Midmorning picks regen failed: {_e}")

    # ── CHECK 6: Settlement health ─────────────────────────────────────────
    # Runs every cycle regardless of verification state (not a once-per-day check).
    if hour_ct >= 9:
        ok, detail = _verify_settle_recent()
        if not ok:
            print(f"[Watchdog] ⚠️  Settlement overdue ({detail}) — triggering now")
            try:
                _run_daily_mock_settle()
                _scheduler_state["last_mock_settle"] = datetime.now(timezone.utc).isoformat()
            except Exception as _e:
                print(f"[Watchdog] Settlement re-run failed: {_e}")

    # ── CHECK 6: Retrain subprocess status ─────────────────────────────────
    try:
        check_retrain_status()
    except Exception as _e:
        print(f"[Watchdog] Retrain status check error: {_e}")


# ── Dependency-based execution chains ─────────────────────────────────────────
#
# Morning and afternoon job sequences run as ordered chains instead of
# independent clock-gated dispatches.  Each job waits for its dependencies
# to complete; slow jobs delay subsequent ones rather than being skipped.
# Non-critical failures are logged and skipped; critical failures stop the chain.
#
# Safety ceilings prevent indefinite hangs (API down, network issue).
# Critical jobs get one retry after a 30-second pause.

import json as _json

_MORNING_CHAIN_HOUR_CT   = 7    # Trigger at 7:00 AM CT
_AFTERNOON_CHAIN_HOUR_CT = 14   # Trigger at 2:00 PM CT
_SAFETY_CHECK_HOUR_CT    = 9    # 9:00 AM CT safety net

# Timeout ceilings (seconds) — generous, not aggressive.
_CEILING_NON_CRITICAL = 300     # 5 min for optional jobs
_CEILING_CRITICAL     = 600     # 10 min for critical jobs (retried once)
_CEILING_GENERATE     = 900     # 15 min for pick generation
_CRITICAL_RETRY_PAUSE = 30      # seconds between retry attempts


class _JobTimeout(Exception):
    """Raised when a chain job exceeds its safety ceiling."""
    pass


def _run_with_ceiling(name: str, func, ceiling_sec: int):
    """Run *func* in a daemon thread; raise _JobTimeout if it exceeds *ceiling_sec*."""
    result_box: list = []
    error_box:  list = []

    def _worker():
        try:
            result_box.append(func())
        except Exception as _e:
            error_box.append(_e)

    t = threading.Thread(target=_worker, daemon=True, name=f"Chain-{name}")
    t.start()
    t.join(timeout=ceiling_sec)
    if t.is_alive():
        raise _JobTimeout(f"{name} exceeded {ceiling_sec}s ceiling")
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else {}


# name:        human label + state key suffix
# run_fn:      callable that returns a result dict
# critical:    if True, failure stops the chain (with one retry on timeout)
# depends_on:  list of job names that must have completed (not failed) first
# state_key:   key in _scheduler_state to stamp when done
# ceiling:     max seconds before the job is considered hung

MORNING_CHAIN: list[dict] = [
    # Phase 1: Free API data (no dependencies, run in sequence)
    {"name": "scout",              "run_fn": "_run_daily_scout",
     "critical": False,            "depends_on": [],
     "state_key": "last_scout_run",
     "ceiling": _CEILING_NON_CRITICAL},

    {"name": "injury_refresh_am",  "run_fn": "_run_injury_refresh",
     "critical": False,            "depends_on": [],
     "state_key": "last_injury_refresh_am",
     "ceiling": _CEILING_NON_CRITICAL},

    {"name": "player_projections", "run_fn": "_run_player_projections",
     "critical": False,            "depends_on": [],
     "state_key": "last_player_projections",
     "ceiling": _CEILING_NON_CRITICAL},

    # Phase 2: Paid API data
    {"name": "fixture_refresh",    "run_fn": "_run_daily_fixture_refresh",
     "critical": True,             "depends_on": [],
     "state_key": "last_fixture_refresh",
     "ceiling": _CEILING_CRITICAL},

    {"name": "mlb_pitcher_fetch",  "run_fn": "_run_daily_mlb_pitcher_fetch",
     "critical": False,            "depends_on": ["fixture_refresh"],
     "state_key": "last_mlb_pitcher_fetch",
     "ceiling": _CEILING_NON_CRITICAL},

    # Phase 3: Generation (after all data is ready)
    {"name": "mock_generate",      "run_fn": "_run_daily_mock_generate",
     "critical": True,             "depends_on": ["fixture_refresh", "scout"],
     "state_key": "last_mock_generate",
     "ceiling": _CEILING_GENERATE},

    # Phase 4: Top picks (server-side, after mock_generate populates the cache)
    {"name": "top_picks_seed",     "run_fn": "_run_top_picks_seed",
     "critical": False,            "depends_on": ["mock_generate"],
     "state_key": "last_top_picks_seed",
     "ceiling": _CEILING_NON_CRITICAL},

    # Phase 5: Shadow scoring (after active picks, never blocks the live chain)
    {"name": "shadow_scoring_am",  "run_fn": "_run_shadow_scoring",
     "critical": False,            "depends_on": ["mock_generate"],
     "state_key": "last_shadow_scoring_am",
     "ceiling": _CEILING_GENERATE},
]

AFTERNOON_CHAIN: list[dict] = [
    {"name": "injury_refresh_pm",  "run_fn": "_run_injury_refresh",
     "critical": False,            "depends_on": [],
     "state_key": "last_injury_refresh_pm1",
     "ceiling": _CEILING_NON_CRITICAL},

    {"name": "player_projections_pm", "run_fn": "_run_player_projections",
     "critical": False,            "depends_on": [],
     "state_key": "last_player_projections_pm",
     "ceiling": _CEILING_NON_CRITICAL},

    {"name": "alt_lines_afternoon", "run_fn": "_run_alt_lines_fetch_pm",
     "critical": False,            "depends_on": [],
     "state_key": "last_alt_lines_fetch_afternoon",
     "ceiling": _CEILING_CRITICAL},

    {"name": "afternoon_odds_fetch", "run_fn": "_run_daily_fixture_refresh",
     "critical": True,             "depends_on": [],
     "state_key": "last_fixture_refresh_pm",
     "ceiling": _CEILING_CRITICAL},

    {"name": "pm_mock_generate",   "run_fn": "_run_afternoon_mock_generate",
     "critical": True,             "depends_on": ["afternoon_odds_fetch", "alt_lines_afternoon"],
     "state_key": "last_mock_generate_pm",
     "ceiling": _CEILING_GENERATE},

    # Top picks refresh (server-side, after PM mock_generate refreshes the cache)
    {"name": "top_picks_seed_pm",  "run_fn": "_run_top_picks_seed",
     "critical": False,            "depends_on": ["pm_mock_generate"],
     "state_key": "last_top_picks_seed_pm",
     "ceiling": _CEILING_NON_CRITICAL},

    # Shadow scoring (after active picks, never blocks the live chain)
    {"name": "shadow_scoring_pm",  "run_fn": "_run_shadow_scoring",
     "critical": False,            "depends_on": ["pm_mock_generate"],
     "state_key": "last_shadow_scoring_pm",
     "ceiling": _CEILING_GENERATE},
]


def _run_chain(chain: list[dict], chain_name: str) -> dict:
    """Execute a dependency chain with safety ceilings.  Returns summary dict."""
    start = _now_ct()
    today_ct = start.strftime("%Y-%m-%d")
    completed: dict[str, dict] = {}
    failed: list[str]   = []
    skipped: list[str]  = []
    warnings: list[str] = []
    timeouts: list[str] = []
    retries: list[str]  = []

    _ns = globals()

    print(f"[Chain] {chain_name} chain started at {start.strftime('%H:%M:%S')} CT")

    for job in chain:
        name      = job["name"]
        critical  = job["critical"]
        deps      = job.get("depends_on", [])
        fn_name   = job["run_fn"]
        state_key = job["state_key"]
        ceiling   = job.get("ceiling", _CEILING_NON_CRITICAL)

        # ── Check dependencies ────────────────────────────────────────────
        dep_failures = [d for d in deps if d in failed]
        if dep_failures:
            if critical:
                print(f"[Chain] SKIPPING {name} — critical dependency failed: {dep_failures}")
                skipped.append(name)
                failed.append(name)
                print(f"[Chain] Critical job skipped due to dependency — stopping chain")
                break
            else:
                print(f"[Chain] SKIPPING {name} — dependency failed: {dep_failures}")
                skipped.append(name)
                continue

        # ── Resolve function ──────────────────────────────────────────────
        run_fn = _ns.get(fn_name)
        if run_fn is None:
            print(f"[Chain] ERROR: no function named {fn_name}")
            failed.append(name)
            if critical:
                print(f"[Chain] Critical job missing — stopping chain")
                break
            continue

        # ── Execute with ceiling ──────────────────────────────────────────
        job_start = _now_ct()
        try:
            result = _run_with_ceiling(name, run_fn, ceiling)
            duration = (_now_ct() - job_start).total_seconds()
            completed[name] = result or {}

            _scheduler_state[state_key]              = today_ct
            _scheduler_state[f"{state_key}_result"]  = result

            print(f"[Chain] {name} completed in {duration:.0f}s")

            if duration > 120:
                warn = f"{name} took {duration:.0f}s (>120s threshold)"
                warnings.append(warn)
                print(f"[Chain] WARNING: {warn}")

        except _JobTimeout:
            duration = (_now_ct() - job_start).total_seconds()
            timeouts.append(f"{name} (hit {ceiling}s ceiling)")
            print(f"[Chain] {name} TIMED OUT after {duration:.0f}s (ceiling={ceiling}s)")

            if critical:
                # ── Retry once for critical jobs ──────────────────────────
                print(f"[Chain] Critical job — retrying in {_CRITICAL_RETRY_PAUSE}s")
                retries.append(name)
                time.sleep(_CRITICAL_RETRY_PAUSE)

                retry_start = _now_ct()
                try:
                    result = _run_with_ceiling(name, run_fn, ceiling)
                    retry_dur = (_now_ct() - retry_start).total_seconds()
                    completed[name] = result or {}

                    _scheduler_state[state_key]              = today_ct
                    _scheduler_state[f"{state_key}_result"]  = result

                    info = f"{name} retry succeeded in {retry_dur:.0f}s"
                    retries[-1] = info   # replace bare name with outcome
                    print(f"[Chain] {info}")

                except _JobTimeout:
                    retry_dur = (_now_ct() - retry_start).total_seconds()
                    failed.append(name)
                    _scheduler_state[f"{state_key}_result"] = {
                        "error": f"timed out twice ({ceiling}s + {ceiling}s)"
                    }
                    retries[-1] = f"{name} retry also timed out ({retry_dur:.0f}s)"
                    print(f"[Chain] {name} FAILED on retry — critical failure, stopping chain")
                    break

                except Exception as exc2:
                    retry_dur = (_now_ct() - retry_start).total_seconds()
                    failed.append(name)
                    _scheduler_state[f"{state_key}_result"] = {"error": str(exc2)}
                    retries[-1] = f"{name} retry failed: {exc2}"
                    print(f"[Chain] {name} retry FAILED after {retry_dur:.0f}s: {exc2} — stopping chain")
                    break
            else:
                # Non-critical timeout — mark failed, continue
                failed.append(name)
                _scheduler_state[f"{state_key}_result"] = {
                    "error": f"timed out ({ceiling}s ceiling)"
                }
                print(f"[Chain] Non-critical timeout — continuing chain")
                continue

        except Exception as exc:
            duration = (_now_ct() - job_start).total_seconds()
            failed.append(name)
            _scheduler_state[f"{state_key}_result"] = {"error": str(exc)}
            print(f"[Chain] {name} FAILED after {duration:.0f}s: {exc}")

            if critical:
                print(f"[Chain] Critical job failed — stopping chain")
                break
            else:
                print(f"[Chain] Non-critical — continuing chain")
                continue

    total = (_now_ct() - start).total_seconds()
    print(f"[Chain] {chain_name} chain finished in {total:.0f}s "
          f"({len(completed)} completed, {len(failed)} failed, {len(skipped)} skipped)")

    summary = {
        "chain":      chain_name,
        "started":    start.strftime("%H:%M:%S CT"),
        "finished":   _now_ct().strftime("%H:%M:%S CT"),
        "duration_sec": round(total, 1),
        "completed":  list(completed.keys()),
        "failed":     failed,
        "skipped":    skipped,
        "warnings":   warnings,
        "timeouts":   timeouts,
        "retries":    retries,
    }

    # ── Persist to chain_execution_log ────────────────────────────────────
    try:
        import sqlite3 as _sq3
        _db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _conn = _sq3.connect(_db_path)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_execution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_name TEXT,
                started_at DATETIME,
                finished_at DATETIME,
                total_duration_sec REAL,
                jobs_completed TEXT,
                jobs_failed TEXT,
                jobs_skipped TEXT,
                warnings TEXT,
                timeouts TEXT,
                retries TEXT
            )
        """)
        # Safe-add columns if table already exists without them
        for _col in ("timeouts TEXT", "retries TEXT"):
            try:
                _conn.execute(f"ALTER TABLE chain_execution_log ADD COLUMN {_col}")
            except Exception:
                pass  # column already exists
        with DB_WRITE_LOCK:
            _conn.execute(
                "INSERT INTO chain_execution_log "
                "(chain_name, started_at, finished_at, total_duration_sec, "
                " jobs_completed, jobs_failed, jobs_skipped, warnings, timeouts, retries) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    chain_name,
                    start.isoformat(),
                    _now_ct().isoformat(),
                    round(total, 1),
                    _json.dumps(list(completed.keys())),
                    _json.dumps(failed),
                    _json.dumps(skipped),
                    _json.dumps(warnings),
                    _json.dumps(timeouts),
                    _json.dumps(retries),
                ),
            )
            _conn.commit()
        _conn.close()
    except Exception as _log_exc:
        print(f"[Chain] WARNING: failed to log chain execution: {_log_exc}")

    # Store latest chain result in scheduler state for health endpoint
    _scheduler_state[f"last_{chain_name}_chain"]        = today_ct
    _scheduler_state[f"last_{chain_name}_chain_result"] = summary

    return summary


def _should_run_morning_chain() -> bool:
    """True once daily at or after 7:00 AM CT, if not already run today."""
    ct = _now_ct()
    if ct.hour < _MORNING_CHAIN_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_morning_chain") == today_str:
        return False
    # DB evidence: check chain_execution_log for today's morning run
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM chain_execution_log "
            "WHERE chain_name = 'morning' AND DATE(started_at) = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_morning_chain"] = today_str
            print(f"[Scheduler] Restart detected — morning chain already ran today")
            return False
    except Exception:
        pass  # table may not exist yet on first run
    return True


def _should_run_afternoon_chain() -> bool:
    """True once daily at or after 2:00 PM CT, if not already run today."""
    ct = _now_ct()
    if ct.hour < _AFTERNOON_CHAIN_HOUR_CT:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_afternoon_chain") == today_str:
        return False
    # DB evidence
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM chain_execution_log "
            "WHERE chain_name = 'afternoon' AND DATE(started_at) = ?",
            (today_str,)
        ).fetchone()[0]
        _con.close()
        if _n > 0:
            _scheduler_state["last_afternoon_chain"] = today_str
            print(f"[Scheduler] Restart detected — afternoon chain already ran today")
            return False
    except Exception:
        pass
    return True


# ── 9:00 AM CT safety check ──────────────────────────────────────────────────

def _morning_chain_safety_check():
    """Verify morning chain ran today. If not, retrigger it.

    Called once at ~9:00 AM CT.  Two checks:
      1. chain_execution_log has a 'morning' row for today
      2. Secondary: mock bets exist for today (picks were generated)

    If both are missing, re-run the morning chain.
    """
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")

    chain_ran = False
    picks_exist = False
    _picks = 0

    # ── Check 1: chain_execution_log ──────────────────────────────────────
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db)
        _n = _con.execute(
            "SELECT COUNT(*) FROM chain_execution_log "
            "WHERE chain_name = 'morning' AND DATE(started_at) = ?",
            (today_str,)
        ).fetchone()[0]
        chain_ran = _n > 0

        # ── Check 2: mock bets generated today ───────────────────────────
        try:
            _picks = _con.execute(
                "SELECT COUNT(*) FROM mock_bets WHERE DATE(created_at) = ?",
                (today_str,)
            ).fetchone()[0]
            picks_exist = _picks >= 5
        except Exception:
            pass  # table may not exist

        _con.close()
    except Exception as _e:
        print(f"[SafetyCheck] DB check failed: {_e}")

    if chain_ran and picks_exist:
        print(f"[SafetyCheck] 9 AM verification passed — morning chain ran, "
              f"{_picks if picks_exist else '?'} picks exist")
        return

    if chain_ran and not picks_exist:
        print(f"[SafetyCheck] WARNING: Morning chain ran but no picks found — "
              f"retriggering mock_generate only")
        try:
            _ns = globals()
            gen_fn = _ns.get("_run_daily_mock_generate")
            if gen_fn:
                result = _run_with_ceiling("safety_mock_generate", gen_fn, _CEILING_GENERATE)
                _scheduler_state["last_mock_generate"] = today_str
                _scheduler_state["last_mock_generate_result"] = result
                print(f"[SafetyCheck] Recovery mock_generate completed: {result}")
        except Exception as _exc:
            print(f"[SafetyCheck] Recovery mock_generate failed: {_exc}")
        return

    # Neither chain ran nor picks exist — full retrigger
    print(f"[SafetyCheck] ALERT: Morning chain did NOT run today — retriggering full chain")
    try:
        summary = _run_chain(MORNING_CHAIN, "morning")
        print(f"[SafetyCheck] Recovery chain completed: "
              f"{len(summary.get('completed', []))} completed, "
              f"{len(summary.get('failed', []))} failed")
    except Exception as _exc:
        print(f"[SafetyCheck] Recovery chain FAILED: {_exc}")


def _loop(interval_mins: int, auto_retrain: bool):
    global _scheduler_state
    print(f"[Scheduler] Started — interval: {interval_mins}m, auto_retrain: {auto_retrain}")

    # Wait for startup threads (Pi backfill, soccer form, feature matrix) to
    # finish their initial DB writes before the first cycle.  Without this the
    # PRAGMA journal_mode=WAL on the first SessionLocal() connection races with
    # concurrent SHM creation from other threads, causing "disk I/O error" on macOS.
    import time as _time
    _time.sleep(40)

    while not _stop_event.is_set():
        now_utc = datetime.now(timezone.utc)
        now_ct  = _now_ct()
        today_ct = now_ct.strftime("%Y-%m-%d")

        _scheduler_state["last_run"]    = now_utc.isoformat()
        _scheduler_state["total_cycles"] += 1

        print(f"[Scheduler] Running cycle #{_scheduler_state['total_cycles']} at {now_utc.isoformat()} "
              f"({now_ct.strftime('%H:%M')} CT)")
        try:
            result = _run_cycle(auto_retrain=auto_retrain)
            _scheduler_state["last_result"]    = result
            _scheduler_state["total_settled"] += result.get("bets_settled", 0)
            print(f"[Scheduler] Done — {result.get('message', '')}")
        except Exception as e:
            import traceback as _tb
            print(f"[Scheduler] Error in cycle: {e}")
            _tb.print_exc()
            _scheduler_state["last_result"] = {"error": str(e)}

        # ── Daily jobs — each wrapped so one failure can't kill the loop ─────────
        def _dispatch(name: str, should_fn, run_fn, state_key: str,
                      ts_arg=None, use_now_utc: bool = True, store_ts: bool = False,
                      timeout: int | None = None):
            """Run a scheduled job if its guard passes; log any exception.

            store_ts=True: stores full UTC ISO timestamp in state_key instead of
            today's date string — used for jobs that run multiple times per day
            (e.g. mock_settle) so the guard can enforce a cooldown window rather
            than a once-per-day limit.

            timeout: max seconds to wait for the job. If exceeded, the job
            continues in its daemon thread but _dispatch returns immediately
            so the scheduler loop can proceed to subsequent jobs (e.g. PM gen).
            """
            try:
                arg = now_utc if use_now_utc else None
                if not (should_fn(arg) if arg is not None else should_fn()):
                    return

                if timeout is not None:
                    # Run in a daemon thread with a time limit
                    _result_box: list = []
                    _err_box:    list = []

                    def _worker():
                        try:
                            r = run_fn() if ts_arg is None else run_fn(ts_arg)
                            _result_box.append(r)
                        except Exception as _e:
                            _err_box.append(_e)

                    _t = threading.Thread(target=_worker, daemon=True, name=f"Dispatch-{name}")
                    _t.start()
                    _t.join(timeout=timeout)
                    if _t.is_alive():
                        print(f"[Scheduler] WARNING: {name} exceeded {timeout}s timeout — "
                              f"continuing in background (scheduler proceeds)")
                        _scheduler_state[state_key + "_result"] = {"warning": f"timeout after {timeout}s, still running"}
                        return
                    if _err_box:
                        raise _err_box[0]
                    result = _result_box[0] if _result_box else {}
                else:
                    result = run_fn() if ts_arg is None else run_fn(ts_arg)

                _scheduler_state[state_key]             = now_utc.isoformat() if store_ts else today_ct
                _scheduler_state[state_key + "_result"] = result
            except Exception as _exc:
                print(f"[Scheduler] ERROR in {name}: {_exc}")
                _scheduler_state[state_key + "_result"] = {"error": str(_exc)}

        _dispatch("nightly_odds",        _should_run_nightly_odds,          _run_nightly_odds_update,       "last_odds_update")
        _dispatch("nightly_ats",         _should_run_nightly_ats,           _run_nightly_ats_update,        "last_ats_update")
        _dispatch("nightly_mlb",         _should_run_nightly_mlb,           _run_nightly_mlb_update,        "last_mlb_update")
        _dispatch("nightly_nhl",         _should_run_nightly_nhl,           _run_nightly_nhl_update,        "last_nhl_update")
        _dispatch("nhl_etl",             _should_run_daily_nhl_etl,         _run_daily_nhl_etl,             "last_nhl_etl_update")
        _dispatch("mlb_etl",             _should_run_daily_mlb_etl,         _run_daily_mlb_etl,             "last_mlb_etl_update")
        _dispatch("fanduel_sync",        _should_run_weekly_fanduel_sync,   _run_weekly_fanduel_sync,       "last_fanduel_sync")
        _dispatch("personal_edge_refresh", _should_run_personal_edge_refresh, _run_personal_edge_refresh,      "last_personal_edge_refresh")
        _dispatch("signal_analysis",     _should_run_weekly_signal_analysis,_run_weekly_signal_analysis,    "last_signal_analysis")
        _dispatch("weekly_retrain",      _should_run_weekly_retrain,        _run_weekly_retrain,            "last_weekly_retrain")
        _dispatch("soccer_fetch",        _should_run_daily_soccer_fetch,    _run_daily_soccer_fetch,        "last_soccer_fetch")
        _dispatch("wc_odds_refresh",    _should_run_wc_odds_refresh,       _run_wc_odds_refresh,           "last_wc_odds_refresh")

        # ── Morning chain (7:00 AM CT) ────────────────────────────────────────
        # Dependency-based with safety ceilings: scout → injury → projections
        # → fixture_refresh → mlb_pitcher_fetch → mock_generate.
        if _should_run_morning_chain():
            try:
                _run_chain(MORNING_CHAIN, "morning")
            except Exception as _chain_exc:
                print(f"[Scheduler] ERROR in morning chain: {_chain_exc}")

        # ── Afternoon chain (2:00 PM CT) ──────────────────────────────────────
        # injury_refresh_pm → player_projections_pm → afternoon_odds_fetch →
        # pm_mock_generate.  PM generation waits for fresh odds.
        if _should_run_afternoon_chain():
            try:
                _run_chain(AFTERNOON_CHAIN, "afternoon")
            except Exception as _chain_exc:
                print(f"[Scheduler] ERROR in afternoon chain: {_chain_exc}")

        # ── 9 AM CT safety check — verify morning chain ran ──────────────────
        hour_ct  = now_ct.hour
        min_ct   = now_ct.minute
        if (hour_ct >= _SAFETY_CHECK_HOUR_CT
                and _scheduler_state.get("last_safety_check") != today_ct):
            _scheduler_state["last_safety_check"] = today_ct
            try:
                _morning_chain_safety_check()
            except Exception as _sc_exc:
                print(f"[Scheduler] ERROR in morning safety check: {_sc_exc}")

        # ── Standalone clock-based jobs (no chain dependencies) ───────────────
        # mock_settle is handled by the clock-aligned _settlement_cron_loop thread
        # (fires at :00 and :30 CT) — do NOT dispatch here to avoid double-settlement.
        _dispatch("regime_classify",     _should_run_regime_classification, _run_regime_classification,     "last_regime_classify")
        _dispatch("alt_lines_morning",   _should_run_alt_lines_morning,     _run_alt_lines_fetch,           "last_alt_lines_fetch_morning")
        # NegBin shadow AM — MUST run after alt_lines_morning completes (full-slate odds).
        # _dispatch is synchronous: alt_lines_morning blocks until done, so this line
        # only executes after the fetch has returned and state is stamped.
        _dispatch("negbin_shadow_am",    _should_run_negbin_shadow_am,      _run_negbin_shadow_am,          "last_negbin_shadow_am", use_now_utc=False)
        # Shadow B (recalibrated) stamped inside _run_negbin_shadow_am — same fixture snapshot
        # ATS ungated shadow — scores ALL MLB spread legs (no gates) for discrimination analysis
        _dispatch("ats_ungated_shadow", _should_run_ats_ungated_shadow, _run_ats_ungated_shadow, "last_ats_ungated_shadow", use_now_utc=False)
        # 10:45 AM CT near-close for day/getaway games (standalone — no downstream chain)
        _dispatch("alt_lines_getaway",   _should_run_alt_lines_getaway,     _run_alt_lines_fetch,           "last_alt_lines_fetch_getaway")
        # alt_lines_afternoon moved into AFTERNOON_CHAIN (runs before pm_mock_generate)
        # 5 PM CT near-close alt-line capture for CLV (standalone — no downstream chain)
        _dispatch("alt_lines_clv",       _should_run_alt_lines_clv,         _run_alt_lines_fetch,           "last_alt_lines_fetch_clv")
        _dispatch("calibration_check",   _should_run_calibration_check,    _run_calibration_check,         "last_calibration_check")
        # 4 PM injury refresh (standalone — not part of afternoon chain)
        try:
            if _should_run_injury_refresh("pm2"):
                _inj_result = _run_injury_refresh()
                _scheduler_state["last_injury_refresh_pm2"]           = today_ct
                _scheduler_state["last_injury_refresh_pm2_result"]    = _inj_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in injury_refresh_pm2: {_exc}")

        # Imminent fetch — 2x/day at 7:15 AM and 2:30 PM CT
        try:
            if _should_run_imminent_fetch(now_utc):
                imm_result = _run_imminent_fetch()
                _scheduler_state["last_imminent_fetch"]        = now_utc.isoformat()
                _scheduler_state["last_imminent_fetch_result"] = imm_result
                # Stamp per-slot flag so each window only fires once per day
                _ct_now = _now_ct()
                _today_ct_str = _ct_now.strftime("%Y-%m-%d")
                if _ct_now.hour == 7:
                    _scheduler_state["last_imminent_fetch_am_date"] = _today_ct_str
                elif _ct_now.hour == 14:
                    _scheduler_state["last_imminent_fetch_pm_date"] = _today_ct_str
                elif _ct_now.hour == 17:
                    _scheduler_state["last_imminent_fetch_clv_date"] = _today_ct_str
                    print("[Scheduler] CLV closing-line snapshot captured")
        except Exception as _exc:
            print(f"[Scheduler] ERROR in imminent_fetch: {_exc}")

        # Props and best-lines (no now_utc arg)
        try:
            if _should_run_props("11am"):
                p11_result = _run_props_fetch("11am")
                _scheduler_state["last_props_11am"]        = today_ct
                _scheduler_state["last_props_11am_result"] = p11_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in props_11am: {_exc}")

        try:
            if _should_run_props("4pm"):
                p4_result = _run_props_fetch("4pm")
                _scheduler_state["last_props_4pm"]        = today_ct
                _scheduler_state["last_props_4pm_result"] = p4_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in props_4pm: {_exc}")

        try:
            if _should_run_best_lines():
                bl_result = _run_best_lines_daily()
                _scheduler_state["last_best_lines"]        = today_ct
                _scheduler_state["last_best_lines_result"] = bl_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in best_lines: {_exc}")

        # ── Cashout live-odds poll (every 15 min during game windows) ────────
        try:
            from cash_out_engine import should_poll_live_odds, poll_live_odds as _poll_live
            if should_poll_live_odds():
                # Rate limit: skip if we polled less than 14 min ago
                _last_poll = _scheduler_state.get("last_cashout_poll")
                _poll_ok = True
                if _last_poll:
                    try:
                        _lp_dt = datetime.fromisoformat(_last_poll)
                        if _lp_dt.tzinfo is None:
                            _lp_dt = _lp_dt.replace(tzinfo=timezone.utc)
                        _poll_ok = (datetime.now(timezone.utc) - _lp_dt).total_seconds() >= 840
                    except Exception:
                        pass
                if _poll_ok:
                    _poll_result = _poll_live()
                    _scheduler_state["last_cashout_poll"]        = datetime.now(timezone.utc).isoformat()
                    _scheduler_state["last_cashout_poll_result"] = _poll_result
        except Exception as _co_exc:
            print(f"[Scheduler] ERROR in cashout_poll: {_co_exc}")

        # ── Nightly DB maintenance (3 AM CT): prune old rows, checkpoint WAL ──
        try:
            if hour_ct == 3 and min_ct < 30 and _scheduler_state.get("last_db_maintenance") != today_ct:
                _scheduler_state["last_db_maintenance"] = today_ct
                _db_maint_result = _run_db_maintenance()
                _scheduler_state["last_db_maintenance_result"] = _db_maint_result
                print(f"[Scheduler] DB maintenance: {_db_maint_result}")
        except Exception as _maint_exc:
            print(f"[Scheduler] ERROR in db_maintenance: {_maint_exc}")

        # ── Watchdog: verify critical jobs completed; auto-recover if not ───────
        try:
            _run_watchdog()
        except Exception as _wde:
            print(f"[Scheduler] ERROR in watchdog: {_wde}")

        # ── SettlementCron self-heal: restart the thread if it died ─────────────
        try:
            _cron_alive = any(
                t.name == "SettlementCron" and t.is_alive()
                for t in threading.enumerate()
            )
            if not _cron_alive and not _stop_event.is_set():
                print("[Scheduler] ⚠️  SettlementCron thread died — restarting")
                _scheduler_state["settle_cron_running"] = False
                _heal_cron = threading.Thread(
                    target=_settlement_cron_loop,
                    daemon=True,
                    name="SettlementCron",
                )
                _heal_cron.start()
                try:
                    next_p, next_w = _next_settle_marks_str()
                    print(f"[Scheduler] SettlementCron restarted — next primary: {next_p}")
                except Exception:
                    print("[Scheduler] SettlementCron restarted")
        except Exception as _cron_exc:
            print(f"[Scheduler] ERROR in SettlementCron self-heal: {_cron_exc}")

        next_run = datetime.now(timezone.utc).timestamp() + interval_mins * 60
        _scheduler_state["next_run"] = datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat()

        # Sleep in short chunks so stop_event is checked frequently
        for _ in range(interval_mins * 60 // 5):
            if _stop_event.is_set():
                break
            time.sleep(5)

    print("[Scheduler] Stopped.")
    _scheduler_state["running"] = False


# CT minute marks for the settlement cron:
#   :00 / :30 → primary settlement run
#   :05 / :35 → watchdog (triggers if primary missed)
_SETTLE_CRON_PRIMARY  = frozenset({0, 30})
_SETTLE_CRON_WATCHDOG = frozenset({5, 35})
_SETTLE_CRON_ALL      = _SETTLE_CRON_PRIMARY | _SETTLE_CRON_WATCHDOG

# Credit-save mode: settlement fires once daily at 11:30 PM CT only.
# Previously ran every :00 and :30 from 9 AM–2 AM = ~34 runs/day.
# 11:30 PM covers the end of the last MLB/NBA/NHL games on the East Coast.
# Late West Coast finishes (past midnight CT) settle the following night.
# Savings: ~33 fewer score API calls/day vs full schedule.
# Re-enable full schedule by setting _SETTLE_ALLOWED_HOURS_CT to None when budget allows.
_SETTLE_ALLOWED_HOURS_CT: frozenset[int] | None = frozenset({23})
_SETTLE_ALLOWED_MINUTE_CT: int | None = 30  # must be :30 mark (23:30 = 11:30 PM CT)


def _sleep_until_next_settle_mark() -> None:
    """
    Sleep (in 15-second chunks) until the clock reaches the next :00, :05,
    :30, or :35 past the hour (CT).  Minimum 5-second sleep to avoid a hot
    loop if we're called exactly on a mark.
    """
    now = _now_ct()
    current_secs = now.minute * 60 + now.second

    # Seconds until each mark.
    # Only add 3600 when the mark has already passed (delta <= 0) — never skip
    # an upcoming mark just because it's a few seconds away.  The max(5, best)
    # floor handles the "woke exactly on a mark" case without accidentally
    # bumping a mark that is 1-4 seconds in the future to next hour.
    best: float | None = None
    for m in _SETTLE_CRON_ALL:
        delta = m * 60 - current_secs
        if delta <= 0:
            delta += 3600
        if best is None or delta < best:
            best = delta

    sleep_total = max(5, best)  # type: ignore[arg-type]
    # Use time.time() (wall-clock) not time.monotonic() — on macOS, monotonic
    # pauses during system sleep, so the cron would keep sleeping toward a mark
    # that already passed hours ago after the Mac wakes up.  Wall-clock time
    # advances through sleep, so the loop exits immediately on wake.
    end = time.time() + sleep_total
    while time.time() < end:
        if _stop_event.is_set():
            return
        remaining = end - time.time()
        time.sleep(min(15, max(0, remaining)))


def _next_settle_marks_str() -> tuple[str, str]:
    """
    Return (next_primary_ct, next_watchdog_ct) as human-readable strings.
    Used by the health endpoint.
    """
    from datetime import timedelta

    def _fmt(marks: frozenset) -> str:
        now = _now_ct()
        cur_min = now.minute
        cur_sec = now.second
        for m in sorted(marks):
            delta_min = m - cur_min
            if delta_min > 0 or (delta_min == 0 and cur_sec == 0):
                t = now.replace(minute=m, second=0, microsecond=0)
                return t.strftime("%-I:%M %p CT")
        # All marks already passed this hour — wrap to next hour
        next_hour = (now + timedelta(hours=1)).replace(
            minute=min(marks), second=0, microsecond=0
        )
        return next_hour.strftime("%-I:%M %p CT")

    return _fmt(_SETTLE_CRON_PRIMARY), _fmt(_SETTLE_CRON_WATCHDOG)


def _settlement_cron_loop() -> None:
    """
    Clock-aligned settlement cron — replaces the old SettlementWatcher sleep loop.

    Wakes at the next :00, :05, :30, or :35 mark (CT):
      :00 / :30 → primary: run settlement unconditionally (active hours only)
      :05 / :35 → watchdog: fire only if settlement hasn't run in 10 min

    Driven by wall-clock time, not relative sleep intervals, so there is no
    drift from long-running blocking calls.
    """
    print("[SettleCron] Settlement cron started — fires at :00, :05, :30, :35 CT")
    _scheduler_state["settle_cron_running"] = True

    # Restart-resilient: populate settle_last_ran_ct from scheduler_runs so the
    # watchdog (:05/:35) doesn't see "never_ran" and double-fire on restart.
    if _scheduler_state.get("settle_last_ran_ct") is None:
        try:
            import sqlite3 as _sq
            _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
            _con = _sq.connect(_db)
            _row = _con.execute("SELECT MAX(run_at) FROM scheduler_runs").fetchone()
            _con.close()
            if _row and _row[0]:
                _last_dt = datetime.fromisoformat(_row[0].rstrip("Z"))
                if _last_dt.tzinfo is None:
                    _last_dt = _last_dt.replace(tzinfo=timezone.utc)
                _scheduler_state["settle_last_ran_ct"] = _last_dt.astimezone(_CT)
                print(f"[SettleCron] Restart detected — backfilled settle_last_ran_ct ({_row[0]})")
        except Exception:
            pass

    while True:
        if _stop_event.is_set():
            print("[SettleCron] Stop event — exiting")
            _scheduler_state["settle_cron_running"] = False
            return

        _sleep_until_next_settle_mark()

        if _stop_event.is_set():
            _scheduler_state["settle_cron_running"] = False
            return

        now_ct = _now_ct()
        minute = now_ct.minute
        now_str = now_ct.strftime("%Y-%m-%d %H:%M CT")

        # ── Catch-up: if the Mac slept through the 11:30 PM window, fire ────────
        # Condition: last settle > 16 hours ago AND it's past 9 AM CT (games from
        # the previous night have had time to finalize in the scores API).
        # This bypasses the credit-save hour restriction because catch-up runs
        # against already-cached scores (no new API credits spent).
        _catchup_fired = False
        _last_ran = _scheduler_state.get("settle_last_ran_ct")
        if _last_ran is not None and now_ct.hour >= 9:
            try:
                _elapsed_h = (now_ct - _last_ran).total_seconds() / 3600
                if _elapsed_h >= 16:
                    print(f"[SettleCron] Catch-up: last settle {_elapsed_h:.1f}h ago — running now")
                    try:
                        result = _run_daily_mock_settle()
                        _scheduler_state["settle_last_ran_ct"]      = _now_ct()
                        _scheduler_state["last_mock_settle"]        = datetime.now(timezone.utc).isoformat()
                        _scheduler_state["last_mock_settle_result"] = result
                        print(f"[SettleCron] Catch-up done — settled={result.get('settled', 0)}")
                        _snapshot_brier_daily()
                    except Exception as _ce:
                        print(f"[SettleCron] Catch-up ERROR: {_ce}")
                    _catchup_fired = True
            except Exception:
                pass
        if _catchup_fired:
            continue  # go back to sleep until the next :xx mark

        # Credit-save mode: only fire at 11:30 PM CT (hour=23, minute=30)
        if _SETTLE_ALLOWED_HOURS_CT is not None:
            if now_ct.hour not in _SETTLE_ALLOWED_HOURS_CT:
                continue  # skip quietly — not in allowed hour
            if _SETTLE_ALLOWED_MINUTE_CT is not None and minute != _SETTLE_ALLOWED_MINUTE_CT:
                continue  # right hour but wrong minute mark (:00 instead of :30)

        # Only fire during active hours (9 AM – 2:59 AM CT next day)
        # Late-night window (0–2 AM) covers NBA/NHL games finishing past 11 PM
        active = now_ct.hour >= 9 or now_ct.hour < 3
        if not active:
            print(f"[SettleCron] Skipping mark :{minute:02d} — outside active hours")
            continue

        # If the sleep overshot past a mark (e.g. targeted :05, woke at :08),
        # log the miss and loop back — do NOT update trigger state for a no-op.
        if minute not in _SETTLE_CRON_ALL:
            print(f"[SettleCron] Overshot mark — woke at :{minute:02d}, skipping")
            continue

        # Only update trigger state when settlement is actually about to run.
        _scheduler_state["settle_cron_last_trigger_ct"] = now_str

        if minute in _SETTLE_CRON_PRIMARY:
            # ── Primary: run settlement regardless of last run time ──────────
            _scheduler_state["settle_cron_last_trigger_type"] = "primary"
            print(f"[SettleCron] Primary settlement at {now_str}")
            try:
                result = _run_daily_mock_settle()
                _scheduler_state["settle_last_ran_ct"]       = _now_ct()
                _scheduler_state["last_mock_settle"]         = datetime.now(timezone.utc).isoformat()
                _scheduler_state["last_mock_settle_result"]  = result
                print(f"[SettleCron] Primary done — settled={result.get('settled', 0)}")
                _snapshot_brier_daily()
            except Exception as _e:
                print(f"[SettleCron] ERROR in primary settlement: {_e}")

        elif minute in _SETTLE_CRON_WATCHDOG:
            # ── Watchdog: fire only if primary missed ────────────────────────
            _scheduler_state["settle_cron_last_trigger_type"] = "watchdog"
            last = _scheduler_state.get("settle_last_ran_ct")
            elapsed_min: float | None = None
            if last is not None:
                try:
                    elapsed_min = (_now_ct() - last).total_seconds() / 60
                except Exception:
                    pass

            if elapsed_min is not None and elapsed_min < 10:
                print(f"[SettleCron] Watchdog ✅ settlement current "
                      f"({elapsed_min:.0f} min ago)")
            else:
                detail = f"{elapsed_min:.0f}min_ago" if elapsed_min else "never_ran"
                print(f"[SettleCron] Watchdog ⚠️ overdue ({detail}) — triggering")
                try:
                    result = _run_daily_mock_settle()
                    _scheduler_state["settle_last_ran_ct"]      = _now_ct()
                    _scheduler_state["last_mock_settle"]        = datetime.now(timezone.utc).isoformat()
                    _scheduler_state["last_mock_settle_result"] = result
                    print(f"[SettleCron] Watchdog done — settled={result.get('settled', 0)}")
                    _snapshot_brier_daily()
                except Exception as _e:
                    print(f"[SettleCron] ERROR in watchdog settlement: {_e}")


def start(interval_mins: int = 30, auto_retrain: bool = True, daemon: bool = True):
    """Start the background scheduler thread."""
    global _thread, _scheduler_state
    if _scheduler_state["running"]:
        print("[Scheduler] Already running.")
        return

    _stop_event.clear()
    _scheduler_state["running"]       = True
    _scheduler_state["interval_mins"] = interval_mins

    _thread = threading.Thread(
        target=_loop,
        args=(interval_mins, auto_retrain),
        daemon=daemon,
        name="AutoSettleScheduler",
    )
    _thread.start()

    # Clock-aligned settlement cron — fires at :00, :05, :30, :35 CT.
    # Replaces the old sleep-based SettlementWatcher thread.
    _settle_cron = threading.Thread(
        target=_settlement_cron_loop,
        daemon=True,
        name="SettlementCron",
    )
    _settle_cron.start()
    next_primary, next_watchdog = _next_settle_marks_str()
    print(f"[Scheduler] SettlementCron started — next primary: {next_primary}, "
          f"next watchdog: {next_watchdog}")

    # Startup settlement: run once after server is ready to catch any bets that
    # completed while the Mac was asleep / server was down.  30-second delay
    # lets the DB and all modules finish initialising before the settle call.
    # Staggered after scout (0s) to avoid concurrent DB writes.
    def _startup_settle():
        time.sleep(30)  # settle waits 30s (scout runs first at 0s)
        if _stop_event.is_set():
            return
        print("[StartupSettle] Running post-startup settlement check")
        try:
            result = _run_daily_mock_settle()
            settled = result.get("settled", 0)
            skipped = result.get("skipped", 0)
            print(f"[StartupSettle] Done — settled={settled}, skipped={skipped}")
            _scheduler_state["settle_last_ran_ct"]      = _now_ct()
            _scheduler_state["last_mock_settle"]        = datetime.now(timezone.utc).isoformat()
            _scheduler_state["last_mock_settle_result"] = result
        except Exception as _e:
            print(f"[StartupSettle] ERROR: {_e}")

    _startup_thread = threading.Thread(
        target=_startup_settle,
        daemon=True,
        name="StartupSettle",
    )
    _startup_thread.start()
    print("[Scheduler] StartupSettle thread launched (fires in 30s, after scout)")

    # Startup scout: run after 7 AM CT if scout hasn't run today.
    # Fires immediately (0s delay) — scout runs first in the stagger order.
    def _startup_scout():
        time.sleep(5)  # minimal delay to let DB init finish
        if _stop_event.is_set():
            return
        ct = _now_ct()
        if ct.hour < _SCOUT_HOUR_CT:
            print(f"[StartupScout] Before {_SCOUT_HOUR_CT}:00 CT — skipping until scheduled time")
            return
        if not _should_run_daily_scout(ct):
            print("[StartupScout] Scout already ran today — skipping")
            return
        print("[StartupScout] Running post-startup scout (not yet run today)")
        try:
            result = _run_daily_scout()
            total = result.get("total_props", 0)
            print(f"[StartupScout] Done — {total} props generated")
            _scheduler_state["last_scout_run"]        = _now_ct().strftime("%Y-%m-%d")
            _scheduler_state["last_scout_run_result"] = result
        except Exception as _e:
            print(f"[StartupScout] ERROR: {_e}")

    _startup_scout_thread = threading.Thread(
        target=_startup_scout,
        daemon=True,
        name="StartupScout",
    )
    _startup_scout_thread.start()
    print("[Scheduler] StartupScout thread launched (fires in ~5s)")


def stop():
    """Signal the scheduler to stop after the current sleep."""
    global _scheduler_state
    _stop_event.set()
    _scheduler_state["running"] = False
    print("[Scheduler] Stop requested.")


def run_once(auto_retrain: bool = True) -> dict:
    """Run a single settle cycle immediately (blocking)."""
    return _run_cycle(auto_retrain=auto_retrain)


def run_nightly_odds_now() -> dict:
    """Force the nightly NBA/MLB/NHL odds update regardless of time/date (blocking)."""
    result = _run_nightly_odds_update()
    _scheduler_state["last_odds_update"] = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_odds_result"] = result
    return result


def run_nightly_ats_now() -> dict:
    """Force the nightly NBA ATS scores update regardless of time/date (blocking)."""
    result = _run_nightly_ats_update()
    _scheduler_state["last_ats_update"] = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_ats_result"] = result
    return result


def run_nightly_mlb_now() -> dict:
    """Force the nightly MLB scores update regardless of time/date (blocking)."""
    result = _run_nightly_mlb_update()
    _scheduler_state["last_mlb_update"] = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_mlb_result"] = result
    return result


def run_nightly_nhl_now() -> dict:
    """Force the nightly NHL scores update regardless of time/date (blocking)."""
    result = _run_nightly_nhl_update()
    _scheduler_state["last_nhl_update"] = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_nhl_result"] = result
    return result


def run_daily_nhl_etl_now() -> dict:
    """Force the daily NHL feature-matrix ETL refresh (blocking)."""
    result = _run_daily_nhl_etl()
    _scheduler_state["last_nhl_etl_update"] = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_nhl_etl_result"] = result
    return result


def run_daily_mlb_etl_now() -> dict:
    """Force the daily MLB feature-matrix ETL refresh (blocking, quick mode)."""
    result = _run_daily_mlb_etl()
    _scheduler_state["last_mlb_etl_update"] = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_mlb_etl_result"] = result
    return result


def run_fanduel_sync_now() -> dict:
    """Force the FanDuel weekly CSV sync regardless of day/time (blocking)."""
    result = _run_weekly_fanduel_sync()
    _scheduler_state["last_fanduel_sync"]   = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_fanduel_result"] = result
    return result


def run_fixture_refresh_now() -> dict:
    """Force today's fixture odds refresh regardless of time (blocking)."""
    result = _run_daily_fixture_refresh()
    _scheduler_state["last_fixture_refresh"]        = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_fixture_refresh_result"] = result
    return result


def run_mlb_pitcher_fetch_now() -> dict:
    """Force today's MLB probable pitcher fetch regardless of time (blocking)."""
    result = _run_daily_mlb_pitcher_fetch()
    _scheduler_state["last_mlb_pitcher_fetch"]        = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_mlb_pitcher_fetch_result"] = result
    return result


def run_mock_generate_now() -> dict:
    """Force the daily mock bet generation regardless of time (blocking)."""
    result = _run_daily_mock_generate()
    _scheduler_state["last_mock_generate"]        = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_mock_generate_result"] = result
    return result


def run_mock_settle_now() -> dict:
    """Force the daily mock bet settlement regardless of time (blocking)."""
    result = _run_daily_mock_settle()
    _scheduler_state["last_mock_settle"]        = datetime.utcnow().isoformat() + "Z"
    _scheduler_state["last_mock_settle_result"] = result
    return result


def run_imminent_fetch_now() -> dict:
    """Force an imminent-game snapshot regardless of schedule (blocking)."""
    result = _run_imminent_fetch()
    _scheduler_state["last_imminent_fetch"]        = datetime.utcnow().isoformat()
    _scheduler_state["last_imminent_fetch_result"] = result
    return result


def run_props_fetch_now(slot: str = "manual") -> dict:
    """Force a props fetch for same-day games regardless of time (blocking)."""
    result = _run_props_fetch(slot)
    return result


def run_best_lines_now() -> dict:
    """Force a daily best-lines fetch regardless of time (blocking)."""
    result = _run_best_lines_daily()
    _scheduler_state["last_best_lines"]        = datetime.utcnow().strftime("%Y-%m-%d")
    _scheduler_state["last_best_lines_result"] = result
    return result


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BetIQ Auto-Settle Scheduler")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in minutes")
    parser.add_argument("--no-retrain", action="store_true", help="Disable auto-retraining")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    args = parser.parse_args()

    init_db()

    if args.once:
        print("[Scheduler] Running single cycle…")
        result = run_once(auto_retrain=not args.no_retrain)
        print(result)
    else:
        start(interval_mins=args.interval, auto_retrain=not args.no_retrain, daemon=False)
        # Keep main thread alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            stop()
