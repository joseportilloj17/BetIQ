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


def _notify(title: str, msg: str, sound: str = "Sosumi") -> None:
    """Send macOS notification + print.  Non-fatal on failure."""
    print(f"[NOTIFY] {title}: {msg}")
    try:
        import subprocess
        subprocess.Popen([
            "osascript", "-e",
            f'display notification "{msg}" with title "{title}" sound name "{sound}"'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


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
    # Lineup shadow — ATS + lineup_gap_resid (orthogonal lineup composition feature)
    "last_lineup_shadow":                     None,
    "last_lineup_shadow_result":              None,
    # F5 shadow — First-5-Innings spread predictions (pitcher-dominated market)
    "last_f5_shadow":                         None,
    "last_f5_shadow_result":                  None,
    # Underdog lag shadow — forward paper test for cross-book lag edge (runs after PM alt_lines)
    "last_underdog_lag_shadow":               None,
    "last_underdog_lag_shadow_result":        None,
    # Prop shadow — batter HR + TB forward paper test (Statcast-augmented model)
    "last_prop_shadow":                       None,
    "last_prop_shadow_result":               None,
    # Alt-line shadow — forward validation of mlb_ats_v1 alt-spread edge (real book prices)
    "last_altline_shadow":                    None,
    "last_altline_shadow_result":            None,
    # Researcher — daily news ingestion + Claude synthesis
    "last_researcher":                        None,
    "last_researcher_result":                None,
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
    "watchdog_db_write_verified":    False,   # bets.db can accept writes (corruption detector)
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
    # Injury watcher — Sleeper status polling (every 15 min on NFL game days Thu-Mon)
    "last_injury_watch_poll":        None,   # ISO timestamp of last poll
    "last_injury_watch_poll_result": None,
    # Monthly NegBin mu model retrain (1st of month, 4 AM CT)
    "last_negbin_retrain":           None,   # YYYY-MM month string
    "last_negbin_retrain_result":    None,
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
                    except Exception as _ras_exc:
                        print(f"[Scheduler] auto_settle failed silently: {_ras_exc}")
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


def _is_mlb_season() -> bool:
    """True during the MLB active window (late March through early November)."""
    ct = _now_ct()
    # Spring training starts ~mid-Feb, regular season ends ~late Oct,
    # World Series can run into early Nov.  Gate: Mar 20 – Nov 10.
    if ct.month == 3 and ct.day >= 20:
        return True
    if 4 <= ct.month <= 10:
        return True
    if ct.month == 11 and ct.day <= 10:
        return True
    return False


def _should_run_nightly_mlb(now: datetime) -> bool:
    if not _is_mlb_season():
        return False
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
            except Exception as _cache_exc:
                print(f"[Scheduler] NHL cache invalidation failed: {_cache_exc}")
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
            except Exception as _cache_exc:
                print(f"[Scheduler] MLB cache invalidation failed: {_cache_exc}")
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
    if not _is_mlb_season():
        return False
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


# ── Monthly NegBin mu model retrain (1st of month, 4 AM CT) ────────────────
_NEGBIN_RETRAIN_DAY_CT  = 1   # 1st of month
_NEGBIN_RETRAIN_HOUR_CT = 4   # 4 AM CT — after nightly scores settle, before morning chain

def _should_run_monthly_negbin_retrain() -> bool:
    """True on the 1st of the month after 4 AM CT if not yet run this month."""
    ct = _now_ct()
    if ct.day != _NEGBIN_RETRAIN_DAY_CT:
        return False
    if ct.hour < _NEGBIN_RETRAIN_HOUR_CT:
        return False
    month_str = ct.strftime("%Y-%m")
    # _dispatch stores today_ct ("YYYY-MM-DD"), so compare prefix
    last = _scheduler_state.get("last_negbin_retrain") or ""
    if last.startswith(month_str):
        return False
    # Persisted check: retrain log in historical.db survives restarts
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        _con = _sq.connect(_db)
        try:
            row = _con.execute(
                "SELECT 1 FROM negbin_retrain_log WHERE month = ?", (month_str,)
            ).fetchone()
            if row:
                _scheduler_state["last_negbin_retrain"] = ct.strftime("%Y-%m-%d")
                print(f"[Scheduler] Restart detected — negbin_retrain already ran for {month_str}")
                _con.close()
                return False
        except Exception:
            pass  # table may not exist yet
        _con.close()
    except Exception:
        pass
    return True


def _run_monthly_negbin_retrain() -> dict:
    """Retrain NegBin mu model, validate, deploy if safe, alert loudly on failure.

    Post-retrain validation gates:
      1. Global bias |mean_resid| < 0.3
      2. MAE not worse than previous + 0.05
      3. NegBin LL not dropped > 10
      4. Discrimination ρ > 0.15
    Logs result to negbin_retrain_log table and emits macOS notification on failure.
    """
    import threading as _threading

    def _bg():
        import sqlite3 as _sq

        ct = _now_ct()
        month_str = ct.strftime("%Y-%m")
        retrain_date = ct.strftime("%Y-%m-%d")
        _HIST_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data", "historical.db")

        try:
            import negbin_retrain as _nr
            result = _nr.retrain()

            # ── Log to persistent table ──
            try:
                con = _sq.connect(_HIST_DB)
                con.execute("""
                    CREATE TABLE IF NOT EXISTS negbin_retrain_log (
                        month TEXT PRIMARY KEY,
                        retrain_date TEXT,
                        saved INTEGER,
                        bias_new REAL,
                        rho_new REAL,
                        mae_home REAL,
                        mae_away REAL,
                        ll_new REAL,
                        ll_old REAL
                    )
                """)
                con.execute("""
                    INSERT OR REPLACE INTO negbin_retrain_log
                        (month, retrain_date, saved, bias_new, rho_new,
                         mae_home, mae_away, ll_new, ll_old)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (month_str, retrain_date,
                      1 if result.get("saved") else 0,
                      result.get("bias_new"),
                      result.get("rho_new"),
                      result.get("mae_home_new"),
                      result.get("mae_away_new"),
                      result.get("ll_new"),
                      result.get("ll_old")))
                con.commit()
                con.close()
            except Exception as log_exc:
                print(f"[NegBinRetrain] Failed to persist log: {log_exc}")

            saved = result.get("saved", False)
            bias = result.get("bias_new", 999)
            rho = result.get("rho_new", 0)

            if saved:
                print(f"[Scheduler] NegBin retrain — deployed: "
                      f"bias={bias:+.4f} ρ={rho:.4f}")
            else:
                # ── LOUD FAILURE ALERT ──
                _alert_msg = (f"[NEGBIN RETRAIN ALERT] {retrain_date}: "
                              f"REJECTED — bias={bias:+.4f} ρ={rho:.4f}")
                print(f"[Scheduler] ERROR {_alert_msg}")
                try:
                    os.system(
                        f'osascript -e \'display notification "{_alert_msg}" '
                        f'with title "BetIQ NegBin Retrain FAIL" sound name "Basso"\''
                    )
                except Exception:
                    pass

            _scheduler_state["last_negbin_retrain_result"] = result

        except Exception as exc:
            _alert_msg = f"[NEGBIN RETRAIN ALERT] {retrain_date}: EXCEPTION — {exc}"
            print(f"[Scheduler] ERROR {_alert_msg}")
            try:
                os.system(
                    f'osascript -e \'display notification "{_alert_msg}" '
                    f'with title "BetIQ NegBin Retrain FAIL" sound name "Basso"\''
                )
            except Exception:
                pass
            _scheduler_state["last_negbin_retrain_result"] = {"error": str(exc)}

    _threading.Thread(target=_bg, daemon=True, name="negbin-retrain-bg").start()
    return {"status": "started", "note": "running in background thread"}


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

    # SGP haircut monitor — track FanDuel's adjustment factor per archetype.
    # Edge rests on a software choice; if they tighten, edge dies.
    try:
        from parlay_builder import sgp_haircut_report as _sgp_report
        sgp_result = _sgp_report(_db_path)
        alerts = sgp_result.get("alerts", [])
        n_total = sgp_result.get("total_sgm_bets", 0)
        n_archetypes = len(sgp_result.get("archetypes", {}))
        if alerts:
            for a in alerts:
                print(f"[Scheduler] SGP ALERT: {a}")
        else:
            print(
                f"[Scheduler] SGP haircut check: {n_total} SGM bets, "
                f"{n_archetypes} archetypes tracked, no alerts"
            )
        result["sgp_haircut"] = sgp_result
    except Exception as exc:
        print(f"[Scheduler] SGP haircut check error: {exc}")

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
        # Alert if zero props for any sport that should have games
        errors = result.get("errors", [])
        if errors:
            _notify("BetIQ Scout", f"{len(errors)} sport error(s): {errors[0]}")
        if total == 0:
            _notify("BetIQ Scout", "ZERO props captured across all sports")

        return result
    except Exception as exc:
        _notify("BetIQ Scout", f"Scout run FAILED: {exc}")
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
        # NBA forward log: fetch official PDF report and log snapshot
        try:
            _log_nba_injury_forward()
        except Exception as _nba_exc:
            print(f"[Scheduler] NBA injury forward-log error (non-fatal): {_nba_exc}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Injury refresh error: {exc}")
        return {"error": str(exc)}


def _log_nba_injury_forward() -> None:
    """Fetch the official NBA injury report PDF and append to the forward log.

    Uses the nbainjuries package to parse the PDF.  Each row is tagged with
    the PDF's own ``Game Date`` so settlement matches against box-score data
    for the correct date.  Called from every injury_refresh slot (am/pm1/pm2).
    """
    from datetime import datetime, timezone
    import nbainjuries.injury as _nba_inj
    import sqlite3 as _sq

    now_utc = datetime.now(timezone.utc)
    # NBA PDFs are keyed by ET timestamp — use current time
    try:
        from zoneinfo import ZoneInfo
        now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        now_et = now_utc  # fallback — timestamp still fine
    ts = now_et.replace(tzinfo=None)  # nbainjuries expects naive dt

    if not _nba_inj.check_reportvalid(ts):
        return  # no report available yet

    df = _nba_inj.get_reportdata(ts, return_df=True)
    if df is None or len(df) == 0:
        return

    snapshot_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
    conn = _sq.connect(os.path.abspath(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nba_injury_forward_log (
            snapshot_utc   TEXT,
            game_date      TEXT,
            player_name    TEXT,
            player_id      INTEGER,
            team           TEXT,
            status         TEXT,
            source         TEXT,
            body_part      TEXT,
            role           TEXT,
            mpg            REAL,
            played         INTEGER,
            actual_minutes REAL,
            game_time_et   TEXT,
            PRIMARY KEY (snapshot_utc, player_name)
        )
    """)

    rows = []
    for _, row in df.iterrows():
        pname = str(row.get("Player Name", ""))
        status = str(row.get("Current Status", ""))
        team = str(row.get("Team", ""))
        reason = str(row.get("Reason", ""))
        raw_gd = str(row.get("Game Date", ""))
        game_time = str(row.get("Game Time", ""))
        if not pname or pname == "nan":
            continue
        try:
            pdf_date = datetime.strptime(raw_gd, "%m/%d/%Y").strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue  # skip rows with unparseable dates
        rows.append((
            snapshot_utc, pdf_date, pname, 0, team,
            status, "nba_pdf", reason, "", 0.0, game_time,
        ))

    if rows:
        conn.executemany("""
            INSERT OR IGNORE INTO nba_injury_forward_log
            (snapshot_utc, game_date, player_name, player_id, team,
             status, source, body_part, role, mpg, game_time_et)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        print(f"[Scheduler] NBA forward log: {len(rows)} rows for "
              f"{len(set(r[1] for r in rows))} game date(s)")
    conn.close()


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
        _notify("BetIQ Fixtures",
                f"Refresh failed: {exc}" + (f" (cached {age:.0f}h)" if age else " NO CACHE"))
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
        # Settle Wong teaser shadow legs (NFL — uses nfl.db scores, not generic settle)
        try:
            from shadow_teaser import settle_teaser_legs
            db_t = SessionLocal()
            try:
                t_settled = settle_teaser_legs(db_t)
                if t_settled and t_settled > 0:
                    print(f"[Scheduler] Teaser settle: {t_settled} legs settled")
            finally:
                db_t.close()
        except Exception as _te:
            print(f"[Scheduler] Teaser settle error (non-fatal): {_te}")
        # Backfill devig on newly-settled legs (Group F — measure-only, best-effort)
        try:
            from devig import backfill_devig
            dv = backfill_devig(force=False)
            print(f"[Scheduler] Devig backfill: {dv.get('entry_devig', 0)} entry, {dv.get('close_devig', 0)} close")
        except Exception as _dve:
            print(f"[Scheduler] Devig backfill error (non-fatal): {_dve}")
        # Settle NBA injury forward log against box scores (best-effort)
        try:
            from nba_matchup_engine import NBAMatchupEngine
            _nba_settled = NBAMatchupEngine.settle_injury_log()
            if _nba_settled:
                print(f"[Scheduler] NBA injury settle: {_nba_settled} rows")
        except Exception as _nis:
            print(f"[Scheduler] NBA injury settle error (non-fatal): {_nis}")
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


# ── Weather refresh — runs BEFORE NegBin shadow to provide forecast wind ────────

def _should_run_weather_refresh() -> bool:
    """True if weather hasn't been refreshed today."""
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_weather_refresh") == today_str:
        return False
    # Gate: alt_lines_morning must have completed (we need the fixture list)
    if _scheduler_state.get("last_alt_lines_fetch_morning") != today_str:
        return False
    return True


def _run_weather_refresh() -> dict:
    """Fetch forecast wind for today's games and store in game_weather (historical.db).

    Uses Open-Meteo Forecast API (same as live serving) — the returned wind IS
    the forecast, so wind_speed_forecast_mph = wind_speed_mph for today's fetch.
    """
    import sqlite3
    import requests

    _HIST_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "historical.db")
    _FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    try:
        from weather import MLB_STADIUMS, _find_stadium, _is_dome
    except ImportError:
        from backend.weather import MLB_STADIUMS, _find_stadium, _is_dome

    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")

    conn = sqlite3.connect(_HIST_DB)

    # Ensure column exists
    cols = [c[1] for c in conn.execute("PRAGMA table_info(game_weather)").fetchall()]
    if "wind_speed_forecast_mph" not in cols:
        conn.execute("ALTER TABLE game_weather ADD COLUMN wind_speed_forecast_mph REAL")
        conn.commit()

    # Get today's games that need weather
    games_needing_wx = conn.execute("""
        SELECT DISTINCT g.home_team
        FROM games g
        WHERE g.sport = 'MLB' AND g.game_date = ?
          AND NOT EXISTS (
              SELECT 1 FROM game_weather gw
              WHERE gw.home_team = g.home_team AND gw.game_date = ?
                AND gw.wind_speed_forecast_mph IS NOT NULL
          )
    """, (today_str, today_str)).fetchall()

    if not games_needing_wx:
        conn.close()
        return {"status": "ok", "games": 0, "note": "all games already have forecast wind"}

    written = 0
    failed = 0
    for (home_team,) in games_needing_wx:
        stadium_info = _find_stadium(home_team, "mlb")
        if not stadium_info:
            failed += 1
            continue
        lat, lon, stadium_name = stadium_info

        if _is_dome(stadium_name):
            # Dome: wind=0, forecast wind=0
            conn.execute("""
                INSERT OR REPLACE INTO game_weather
                    (home_team, game_date, temperature_f, wind_speed_mph,
                     wind_speed_forecast_mph, wind_direction_deg, is_dome, stadium)
                VALUES (?, ?, 72.0, 0.0, 0.0, NULL, 1, ?)
            """, (home_team, today_str, stadium_name))
            written += 1
            continue

        try:
            r = requests.get(_FORECAST_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "timezone": "America/New_York",
                "forecast_days": 2,
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[Weather] FAILED for {home_team}: {exc}")
            failed += 1
            continue

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        dirs_ = hourly.get("wind_direction_10m", [])

        # Find 7 PM hour on today
        idx = None
        for i, t in enumerate(times):
            if t.startswith(today_str) and len(t) >= 13 and int(t[11:13]) == 19:
                idx = i
                break
        if idx is None:
            # Fallback: first hour matching today
            for i, t in enumerate(times):
                if t.startswith(today_str):
                    idx = i
                    break

        if idx is None:
            failed += 1
            continue

        temp_c = temps[idx] if idx < len(temps) and temps[idx] is not None else None
        wind_ms = winds[idx] if idx < len(winds) and winds[idx] is not None else None
        wind_dir = dirs_[idx] if idx < len(dirs_) and dirs_[idx] is not None else None

        temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
        wind_mph = round(wind_ms * 2.237, 1) if wind_ms is not None else None

        # For today's fetch: the Forecast API IS the forecast, so
        # wind_speed_forecast_mph = wind_speed_mph (both are forecast values)
        conn.execute("""
            INSERT OR REPLACE INTO game_weather
                (home_team, game_date, temperature_f, wind_speed_mph,
                 wind_speed_forecast_mph, wind_direction_deg, is_dome, stadium)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
        """, (home_team, today_str, temp_f, wind_mph, wind_mph, wind_dir, stadium_name))
        written += 1
        import time as _time
        _time.sleep(0.1)

    conn.commit()
    conn.close()

    result = {"status": "ok", "games_written": written, "games_failed": failed}

    # ── LOUD FAILURE ALERT ──
    if failed > 0 or written == 0:
        _alert_msg = (f"[WEATHER REFRESH ALERT] {today_str}: "
                      f"written={written}, failed={failed}")
        print(f"[Scheduler] ERROR {_alert_msg}")
        # macOS notification
        try:
            os.system(
                f'osascript -e \'display notification "{_alert_msg}" '
                f'with title "BetIQ Weather FAIL" sound name "Basso"\''
            )
        except Exception:
            pass
        result["alert"] = _alert_msg

    print(f"[Scheduler] Weather refresh: {written} written, {failed} failed")
    return result


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
            generate_atscap_counterfactual,
            generate_edgegate_counterfactual,
            generate_fullstack_counterfactual,
            generate_composite_counterfactual,
            generate_distwp_gated_counterfactual,
            generate_bothgates_counterfactual,
            generate_highconv_counterfactual,
            generate_neg15_straight,
            generate_plus35_straight,
            generate_neg15_epp_filtered,
            generate_evaware_parlay,
            generate_straight_stack,
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

            # ATS Asymmetric Cap — counterfactual on A's parlays (separate shadow arm)
            result_cap = generate_atscap_counterfactual(db, today_str=today_str)
            count_cap = len(result_cap) if result_cap else 0
            print(f"[Scheduler] NegBin ATS-cap counterfactual: {count_cap} parlays "
                  f"(from {count_a} A parlays)")

            # Edge gate — counterfactual on A's parlays (separate shadow arm)
            result_edge = generate_edgegate_counterfactual(db, today_str=today_str)
            count_edge = len(result_edge) if result_edge else 0
            print(f"[Scheduler] NegBin edge-gate counterfactual: {count_edge} parlays "
                  f"(from {count_a} A parlays, gate={12}pp)")

            # Full-stack: NegBin select → edge gate → dist_wp quarter-Kelly
            result_fs = generate_fullstack_counterfactual(db, today_str=today_str)
            count_fs = len(result_fs) if result_fs else 0
            print(f"[Scheduler] NegBin full-stack counterfactual: {count_fs} parlays "
                  f"(select→gate→dist_wp Kelly)")

            # Composite: ridge_composite leg selection + bayes floor
            result_comp = generate_composite_counterfactual(db, today_str=today_str)
            count_comp = len(result_comp) if result_comp else 0
            print(f"[Scheduler] NegBin composite counterfactual: {count_comp} parlays "
                  f"(ridge>=0.80, bayes>=0.65)")

            # dist_wp gate: keep legs only where dist_wp >= market_implied
            result_dwg = generate_distwp_gated_counterfactual(db, today_str=today_str)
            count_dwg = len(result_dwg) if result_dwg else 0
            print(f"[Scheduler] NegBin dist_wp-gate counterfactual: {count_dwg} parlays "
                  f"(dist_wp>=market_implied per leg)")

            # Both gates: dist_wp >= market_implied AND edge < 12pp
            result_bg = generate_bothgates_counterfactual(db, today_str=today_str)
            count_bg = len(result_bg) if result_bg else 0
            print(f"[Scheduler] NegBin both-gates counterfactual: {count_bg} parlays "
                  f"(dist_wp>=mkt_implied + edge<12pp)")

            # High-conviction: dist_wp >= 0.85 absolute threshold
            count_hc = 0
            _arm_errors = []
            try:
                result_hc = generate_highconv_counterfactual(db, today_str=today_str)
                count_hc = len(result_hc) if result_hc else 0
                print(f"[Scheduler] NegBin high-conviction counterfactual: {count_hc} parlays "
                      f"(dist_wp>={0.85} absolute)")
            except Exception as _hc_err:
                import traceback as _tb
                print(f"[Scheduler] *** HIGHCONV ARM FAILED: {_hc_err}")
                _tb.print_exc()
                _arm_errors.append(f"highconv: {_hc_err}")

            # Straight -1.5: PAUSED 2026-08-04
            # Rung -1.5 recent rpb=0.011 (pure noise), edge=-53.7%.
            # mu_away never discriminative (broken from start, not fixable by retrain).
            # Ex-hot-window backtest is breakeven. Gate uses wrong rung's dist_wp.
            count_n15 = 0
            print("[Scheduler] NegBin straight -1.5: PAUSED (zero discrimination on -1.5 recently)")

            # Straight +3.5: one bet per fixture where dist_wp >= 0.90
            count_p35 = 0
            try:
                result_p35 = generate_plus35_straight(db, today_str=today_str)
                count_p35 = len(result_p35) if result_p35 else 0
                print(f"[Scheduler] NegBin straight +3.5: {count_p35} bets "
                      f"(dist_wp >= 0.90, $10 flat)")
            except Exception as _p35_err:
                import traceback as _tb
                print(f"[Scheduler] *** PLUS35 ARM FAILED: {_p35_err}")
                _tb.print_exc()
                _arm_errors.append(f"plus35: {_p35_err}")

            # Neg15 + edge_pp filter: PAUSED 2026-08-04 (same as neg15 — -1.5 is dead)
            count_n15_epp = 0
            print("[Scheduler] NegBin neg15+epp: PAUSED (parent arm paused)")

            # EV-aware rung selection variant
            count_eva = 0
            try:
                result_eva = generate_evaware_parlay(db, today_str=today_str)
                count_eva = len(result_eva) if result_eva else 0
                print(f"[Scheduler] NegBin EV-aware rung: {count_eva} parlays "
                      f"(min-rung selection, dist_wp>=0.70)")
            except Exception as _eva_err:
                import traceback as _tb
                print(f"[Scheduler] *** EVAWARE ARM FAILED: {_eva_err}")
                _tb.print_exc()
                _arm_errors.append(f"evaware: {_eva_err}")

            # Straight-bet stack: NegBin select → discriminator rank → straight
            count_stack = 0
            try:
                result_stack = generate_straight_stack(db, today_str=today_str)
                count_stack = len(result_stack) if result_stack else 0
                print(f"[Scheduler] Straight-bet stack: {count_stack} bets "
                      f"(NegBin select → epp<12 → ridge+dist_wp rank → best-edge rung)")
            except Exception as _stack_err:
                import traceback as _tb
                print(f"[Scheduler] *** STRAIGHT-STACK ARM FAILED: {_stack_err}")
                _tb.print_exc()
                _arm_errors.append(f"straight_stack: {_stack_err}")

            if _arm_errors:
                print(f"[Scheduler] *** ARM ERRORS TODAY: {'; '.join(_arm_errors)}")

            return {"shadow_a": count_a, "shadow_b": count_b,
                    "atscap": count_cap, "edgegate": count_edge,
                    "fullstack": count_fs, "composite": count_comp,
                    "distwp_gate": count_dwg, "bothgates": count_bg,
                    "highconv": count_hc, "neg15": count_n15,
                    "plus35": count_p35,
                    "neg15_epp": count_n15_epp, "evaware": count_eva,
                    "straight_stack": count_stack}
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


# ── Lineup shadow — ATS + lineup_gap_resid (orthogonal lineup composition) ──

def _should_run_lineup_shadow() -> bool:
    """True if alt_lines_morning completed today, lineups are available (~2 PM CT), and hasn't run."""
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_lineup_shadow") == today_str:
        return False
    if _scheduler_state.get("last_alt_lines_fetch_morning") != today_str:
        return False
    # MLB confirmed lineups are typically posted 2-3 hours before game time.
    # Most games start 6-7 PM CT, so lineups are available ~3-4 PM.
    # Running at 7:30 AM (when alt_lines_morning fires) gets 0 lineups.
    if ct.hour < 14:
        return False
    return True


def _run_lineup_shadow() -> dict:
    """Score today's MLB spread legs with the lineup-augmented ATS model."""
    try:
        from database import SessionLocal, init_db
        from shadow_lineup import generate_lineup_shadow
        init_db()
        db = SessionLocal()
        try:
            legs = generate_lineup_shadow(db)
            count = len(legs) if legs else 0
            print(f"[Scheduler] Lineup shadow: {count} legs scored")
            return {"legs_scored": count}
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] Lineup shadow error: {exc}")
        return {"error": str(exc)}


# ── Alt-line shadow — forward validation of mlb_ats_v1 alt-spread edge ────────

def _should_run_altline_shadow() -> bool:
    """True if alt_lines_morning completed today and altline shadow hasn't run."""
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_altline_shadow") == today_str:
        return False
    if _scheduler_state.get("last_alt_lines_fetch_morning") != today_str:
        return False
    return True


def _run_altline_shadow() -> dict:
    """Score MLB alt-spread legs with ATS model and log real book prices."""
    try:
        from database import SessionLocal, init_db
        from shadow_altline import generate_altline_shadow
        init_db()
        db = SessionLocal()
        try:
            legs = generate_altline_shadow(db)
            count = len(legs) if legs else 0
            print(f"[Scheduler] Alt-line shadow: {count} legs stored")
            if count == 0:
                print("[Scheduler] Alt-line shadow: 0 legs — check alt_lines table and fixtures for today's alt-spread data")
            return {"legs_stored": count}
        finally:
            db.close()
    except Exception as exc:
        import traceback
        print(f"[Scheduler] Alt-line shadow error: {exc}")
        traceback.print_exc()
        return {"error": str(exc)}


# ── F5 shadow — First-5-Innings spread predictions (pitcher-dominated market) ──

def _should_run_f5_shadow() -> bool:
    """True if alt_lines_morning completed today and F5 shadow hasn't run."""
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_f5_shadow") == today_str:
        return False
    if _scheduler_state.get("last_alt_lines_fetch_morning") != today_str:
        return False
    return True


def _run_f5_shadow() -> dict:
    """Fetch F5 odds, score with ATS model, store as shadow_f5 bets."""
    try:
        from database import SessionLocal, init_db
        from f5_shadow import generate_f5_shadow, settle_f5_bets
        init_db()
        db = SessionLocal()
        try:
            legs = generate_f5_shadow(db)
            count = len(legs) if legs else 0
            print(f"[Scheduler] F5 shadow: {count} legs scored")
            # Also settle any pending F5 bets
            settle_result = settle_f5_bets()
            return {"legs_scored": count, "settlement": settle_result}
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] F5 shadow error: {exc}")
        return {"error": str(exc)}


# ── Underdog lag shadow — forward paper test for cross-book lag edge ──

def _should_run_underdog_lag_shadow() -> bool:
    """True if alt_lines_afternoon completed today and underdog lag hasn't run.

    Needs AM + PM snapshots to detect intra-day staleness. Runs after
    the afternoon alt_lines fetch (2 PM CT chain).
    """
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_underdog_lag_shadow") == today_str:
        return False
    # Must have BOTH morning AND afternoon alt_lines for stale detection
    if _scheduler_state.get("last_alt_lines_fetch_morning") != today_str:
        return False
    if _scheduler_state.get("last_alt_lines_fetch_afternoon") != today_str:
        # DB evidence: recover afternoon stamp after restart (same pattern as morning)
        try:
            import sqlite3 as _sq
            _db = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
            _con = _sq.connect(_db)
            # Afternoon rows: fetched after noon CT (17:00 UTC)
            _n = _con.execute(
                "SELECT COUNT(*) FROM alt_lines WHERE date(fetched_at, '-5 hours') = ? "
                "AND CAST(strftime('%%H', fetched_at) AS INTEGER) >= 17",
                (today_str,)
            ).fetchone()[0]
            _con.close()
            if _n > 0:
                _scheduler_state["last_alt_lines_fetch_afternoon"] = today_str
                print(f"[Scheduler] Restart detected — alt_lines_afternoon already ran today ({_n} PM rows)")
            else:
                return False
        except Exception:
            return False
    return True


def _run_underdog_lag_shadow() -> dict:
    """Detect stale underdog lines and log paper bets."""
    try:
        from database import SessionLocal, init_db
        from shadow_underdog_lag import generate_underdog_lag_shadow
        init_db()
        db = SessionLocal()
        try:
            opps = generate_underdog_lag_shadow(db)
            count = len(opps) if opps else 0
            print(f"[Scheduler] Underdog lag shadow: {count} opportunities logged")
            return {"opportunities": count}
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] Underdog lag shadow error: {exc}")
        return {"error": str(exc)}


# ── Prop shadow — batter HR + TB forward paper test (Statcast model) ──

def _should_run_prop_shadow() -> bool:
    """True if 4 PM props have been fetched today and prop shadow hasn't run."""
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_prop_shadow") == today_str:
        return False
    # Gate: need props fetched today (4 PM slot preferred, 11 AM as fallback)
    if _scheduler_state.get("last_props_4pm") == today_str:
        return True
    if _scheduler_state.get("last_props_11am") == today_str:
        return ct.hour >= 16  # Only run on 11am data if it's past 4pm
    return False


def _run_prop_shadow() -> dict:
    """Generate shadow prop bets (HR + TB) using Statcast-augmented model."""
    try:
        from database import SessionLocal, init_db
        from shadow_prop import generate_prop_shadow
        init_db()
        db = SessionLocal()
        try:
            bets = generate_prop_shadow(db)
            count = len(bets) if bets else 0
            print(f"[Scheduler] Prop shadow: {count} bets logged")
            return {"bets": count}
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] Prop shadow error: {exc}")
        return {"error": str(exc)}


# ── Wong teaser shadow — NFL structural edge forward validation ───────────────

def _should_run_teaser_shadow() -> bool:
    """True on NFL game days (Sun/Mon/Thu) after 10 AM CT, once per day.

    During NFL season (Sep-Feb), identifies qualifying Wong teaser legs
    from today's slate. The gate is simpler than MLB shadows: no upstream
    alt_lines dependency, just NFL schedule awareness.
    """
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_teaser_shadow") == today_str:
        return False
    # Only run during NFL season months (Sep-Feb) and after 10 AM CT
    if ct.month not in (1, 2, 9, 10, 11, 12):
        return False
    if ct.hour < 10:
        return False
    # NFL games are primarily Sun/Mon/Thu (weekday: Mon=0, Thu=3, Sun=6)
    # But also Sat in late season. Run daily during season to catch all.
    return True


def _run_teaser_shadow() -> dict:
    """Generate Wong teaser shadow legs for today's NFL games."""
    try:
        from database import SessionLocal, init_db
        from shadow_teaser import generate_teaser_shadow
        init_db()
        db = SessionLocal()
        try:
            legs = generate_teaser_shadow(db)
            count = len(legs) if legs else 0
            print(f"[Scheduler] Teaser shadow: {count} Wong legs logged")
            return {"legs": count}
        finally:
            db.close()
    except Exception as exc:
        print(f"[Scheduler] Teaser shadow error: {exc}")
        return {"error": str(exc)}


# ── Researcher: daily news ingestion + synthesis ──────────────────────────────

def _should_run_researcher() -> bool:
    """True once daily after 9 AM CT, before researcher has run today."""
    ct = _now_ct()
    today_str = ct.strftime("%Y-%m-%d")
    if _scheduler_state.get("last_researcher") == today_str:
        return False
    return ct.hour >= 9


def _run_researcher() -> dict:
    """Run the daily researcher cycle: fetch → store → synthesize → brief."""
    try:
        from researcher.runner import run_daily_research
        from database import init_db
        init_db()
        result = run_daily_research()
        print(f"[Scheduler] Researcher: {result.get('fetched', 0)} items, brief={result.get('brief', False)}")
        return result
    except Exception as exc:
        print(f"[Scheduler] Researcher error: {exc}")
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
        _active_sports = ["basketball_nba", "icehockey_nhl", "americanfootball_nfl"]
        if _is_mlb_season():
            _active_sports.insert(0, "baseball_mlb")
        for sk in _active_sports:
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
    """Slot 7am (hour 7), 11am (hour 11), or 4pm (hour 16). Once per day per slot (CT)."""
    target = {"7am": 7, "11am": 11, "4pm": 16}.get(slot, 11)
    ct = _now_ct()
    if ct.hour < target:
        return False
    today_str = ct.strftime("%Y-%m-%d")
    return _scheduler_state.get(f"last_props_{slot}") != today_str


def _run_nba_props_morning() -> dict:
    """Morning NBA prop snapshot — runs before scout at 7 AM CT.

    Captures opening prop odds for comparison with model projections.
    NBA-only to save credits (~18 credits/day for a 6-game slate).
    The 11 AM and 4 PM slots cover all sports.
    """
    try:
        from creator_tier import fetch_player_props
        result = fetch_player_props("basketball_nba", hours_ahead=16)
        total = result.get("props_upserted", 0)
        print(f"[Scheduler] NBA morning props: {total} lines captured")
        return {"status": "ok", "slot": "7am", "nba_props": total, "detail": result}
    except Exception as exc:
        print(f"[Scheduler] NBA morning props error: {exc}")
        return {"error": str(exc)}


def _should_run_pretip_props() -> bool:
    """Run pre-tip NBA prop snapshot every scheduler cycle (10 AM - 11 PM CT).

    Weekend/holiday NBA games can tip as early as ~11 AM CT (12 PM ET).
    The fetch itself only hits games in the 30-60 min pre-tip window,
    so running the check earlier just means it finds nothing on weekdays.
    """
    ct = _now_ct()
    return 10 <= ct.hour <= 23


def _run_pretip_props() -> dict:
    """Fetch NBA props for games tipping in 30-60 min.
    Skips games that already had a pre-tip snapshot this cycle."""
    try:
        from creator_tier import fetch_pretip_nba_props
        result = fetch_pretip_nba_props(min_before_low=30, min_before_high=60)
        n = result.get("events_pretip", 0)
        if n > 0:
            print(f"[Scheduler] Pre-tip props: {n} games, "
                  f"{result.get('props_upserted', 0)} lines")
        return result
    except Exception as exc:
        print(f"[Scheduler] Pre-tip props error: {exc}")
        return {"error": str(exc)}


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


def _verify_db_write_health() -> tuple[bool, str]:
    """Verify bets.db can accept writes — catches corruption before 23 silent-fail days.

    Writes a sentinel row to a lightweight table, reads it back, deletes it.
    If any step fails, the DB is likely corrupted (header destroyed, WAL issue, etc.).

    On corruption: attempts .recover FIRST (salvaged 91.2% of pages in the Aug 6-28
    incident). Only falls back to checkpoint restore as last resort. NEVER overwrites
    a corrupted file without recovering first.
    """
    import sqlite3 as _sq
    _db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
    try:
        conn = _sq.connect(_db_path, timeout=5)
        # Verify header is valid SQLite
        conn.execute("PRAGMA integrity_check(1)")
        # Attempt a write + read cycle
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _health_check (
                id INTEGER PRIMARY KEY, ts TEXT
            )
        """)
        sentinel = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT OR REPLACE INTO _health_check (id, ts) VALUES (1, ?)",
                      (sentinel,))
        conn.commit()
        row = conn.execute("SELECT ts FROM _health_check WHERE id = 1").fetchone()
        conn.execute("DELETE FROM _health_check WHERE id = 1")
        conn.commit()
        conn.close()
        if row and row[0] == sentinel:
            return True, "write+read OK"
        return False, "read-back mismatch"
    except _sq.DatabaseError as e:
        # ── Corruption detected — attempt auto-recovery ──────────────────
        msg = f"DatabaseError: {e}"
        print(f"[DB-Health] CORRUPTION DETECTED: {msg}")
        recovery_result = _attempt_db_recovery(_db_path)
        if recovery_result:
            return False, f"{msg} | RECOVERY: {recovery_result}"
        return False, msg
    except Exception as e:
        return False, f"error: {e}"
    finally:
        # Log open file handles on bets.db — detects stale connections
        # from crashed processes that could hold WAL locks open
        try:
            import subprocess as _sub
            _lsof = _sub.run(
                ["lsof", _db_path],
                capture_output=True, text=True, timeout=5,
            )
            _lines = [l for l in _lsof.stdout.strip().splitlines() if l]
            if len(_lines) > 1:  # first line is header
                _pids = set()
                for _l in _lines[1:]:
                    parts = _l.split()
                    if len(parts) >= 2:
                        _pids.add(parts[1])
                if _pids:
                    print(f"[DB-Health] Open handles on bets.db: "
                          f"{len(_pids)} process(es) — PIDs: {sorted(_pids)}")
        except Exception:
            pass  # lsof unavailable or timed out


def _attempt_db_recovery(db_path: str) -> str:
    """Attempt to recover a corrupted DB using .recover BEFORE any checkpoint restore.

    Protocol (learned from Aug 6-28 incident where 22 days were unnecessarily lost):
      1. Back up the corrupted file (NEVER overwrite it)
      2. Try sqlite3 .recover to extract data from intact pages
      3. Compare row counts: recovered vs latest checkpoint
      4. Alert with results — do NOT auto-overwrite
      5. Only restore from checkpoint if .recover yields zero data

    Returns a status string describing what was found.
    """
    import shutil
    import subprocess
    import sqlite3 as _sq

    data_dir = os.path.dirname(db_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 1. Preserve the corrupted file
    corrupt_bak = os.path.join(data_dir, f"bets.db.corrupted_{ts}.bak")
    try:
        shutil.copy2(db_path, corrupt_bak)
        print(f"[DB-Recovery] Corrupted file backed up: {corrupt_bak}")
    except Exception as exc:
        return f"backup_failed: {exc}"

    # 2. Try .recover
    recovered_path = os.path.join(data_dir, f"bets.db.recovered_{ts}")
    try:
        proc = subprocess.run(
            ["sqlite3", db_path, ".recover"],
            capture_output=True, text=True, timeout=300
        )
        if proc.stdout.strip():
            # Pipe recovered SQL into a new DB
            proc2 = subprocess.run(
                ["sqlite3", recovered_path],
                input=proc.stdout, capture_output=True, text=True, timeout=300
            )
            if proc2.returncode == 0 and os.path.exists(recovered_path):
                # Count recovered rows
                try:
                    rconn = _sq.connect(recovered_path)
                    tables = [r[0] for r in rconn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()]
                    counts = {}
                    for tbl in tables:
                        try:
                            cnt = rconn.execute(f"SELECT COUNT(*) FROM [{tbl}]").fetchone()[0]
                            if cnt > 0:
                                counts[tbl] = cnt
                        except Exception:
                            pass
                    rconn.close()

                    total = sum(counts.values())
                    mock_bets = counts.get("mock_bets", counts.get("lost_and_found", 0))
                    msg = (
                        f"RECOVERED {total} rows across {len(counts)} tables "
                        f"(mock_bets-like: {mock_bets}). "
                        f"Recovered file: {recovered_path}. "
                        f"Corrupted backup: {corrupt_bak}. "
                        f"ACTION REQUIRED: merge recovered data before restoring "
                        f"from checkpoint. Do NOT overwrite."
                    )
                    print(f"[DB-Recovery] {msg}")
                    return msg
                except Exception as exc:
                    return f"recover_count_failed: {exc}, file at {recovered_path}"
            else:
                return f"recover_import_failed: {proc2.stderr[:200] if proc2.stderr else 'no output'}"
        else:
            return "recover_empty: .recover produced no output"
    except subprocess.TimeoutExpired:
        return "recover_timeout: .recover took >5min"
    except Exception as exc:
        return f"recover_error: {exc}"


_ICLOUD_BACKUP_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/BetIQ-backups"
)
# Per-DB generation limits: bets.db is small (~40 MB gz), historical.db is large (~140 MB gz)
_ICLOUD_MAX_GENS = {"bets_": 7, "historical_": 3}


def _icloud_backup(daily_dir: str, result: dict) -> None:
    """Copy daily backup DBs to iCloud Drive (gzipped).

    Only copies files whose integrity_check passed.
    Keep 7 generations of bets.db, 3 of historical.db.
    """
    import gzip as _gzip
    import shutil as _shutil

    if not os.path.isdir(daily_dir):
        return

    os.makedirs(_ICLOUD_BACKUP_DIR, exist_ok=True)
    today_str = os.path.basename(daily_dir).replace("daily_", "")

    copied = []
    skipped = []

    # ── bets.db: uncompressed in daily_dir → gzip to iCloud ──
    if result.get("backup_integrity") != "ok":
        skipped.append(
            f"bets.db (integrity={result.get('backup_integrity', 'missing')})")
    else:
        src = os.path.join(daily_dir, "bets.db")
        dst = os.path.join(_ICLOUD_BACKUP_DIR, f"bets_{today_str}.db.gz")
        if os.path.exists(dst):
            copied.append(f"{os.path.basename(dst)} (exists)")
        elif os.path.exists(src):
            with open(src, "rb") as f_in, _gzip.open(dst, "wb") as f_out:
                while True:
                    chunk = f_in.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    f_out.write(chunk)
            sz = os.path.getsize(dst)
            copied.append(f"{os.path.basename(dst)} ({sz / 1048576:.1f} MB)")

    # ── historical.db: already gzipped locally → copy to iCloud ──
    if result.get("hist_integrity") != "ok":
        skipped.append(
            f"historical.db (integrity="
            f"{result.get('hist_integrity', 'missing')})")
    else:
        src_gz = os.path.join(daily_dir, "historical.db.gz")
        dst = os.path.join(_ICLOUD_BACKUP_DIR,
                           f"historical_{today_str}.db.gz")
        if os.path.exists(dst):
            copied.append(f"{os.path.basename(dst)} (exists)")
        elif os.path.exists(src_gz):
            _shutil.copy2(src_gz, dst)
            sz = os.path.getsize(dst)
            copied.append(f"{os.path.basename(dst)} ({sz / 1048576:.1f} MB)")

    result["icloud_copied"] = copied
    if skipped:
        result["icloud_skipped"] = skipped
        print(f"[DB-Maintenance] iCloud skipped (integrity): "
              f"{', '.join(skipped)}")
    if copied:
        print(f"[DB-Maintenance] iCloud backup: {', '.join(copied)}")

    # Prune old generations — per-DB limits
    for prefix, max_gens in _ICLOUD_MAX_GENS.items():
        files = sorted(
            [f for f in os.listdir(_ICLOUD_BACKUP_DIR)
             if f.startswith(prefix) and f.endswith(".db.gz")],
            reverse=True,
        )
        for old in files[max_gens:]:
            try:
                os.remove(os.path.join(_ICLOUD_BACKUP_DIR, old))
                result.setdefault("icloud_pruned", []).append(old)
                print(f"[DB-Maintenance] iCloud pruned: {old}")
            except OSError:
                pass


def _run_restore_test(result: dict) -> None:
    """Decompress newest iCloud backup of each DB, integrity-check + row counts.

    Restores to data/restore_test/, validates, then deletes test copies.
    Alerts on any failure.
    """
    import gzip as _gzip
    import sqlite3

    restore_dir = os.path.join(os.path.dirname(__file__), "..", "data",
                                "restore_test")
    restore_dir = os.path.normpath(restore_dir)
    os.makedirs(restore_dir, exist_ok=True)

    test_results = {}

    _KEY_TABLES = {
        "bets": ["mock_bets", "mock_bet_legs", "fixtures",
                  "shadow_predictions"],
        "historical": ["games", "alt_lines", "player_props"],
    }

    for prefix in ("bets", "historical"):
        if not os.path.isdir(_ICLOUD_BACKUP_DIR):
            test_results[prefix] = "no_icloud_dir"
            continue
        candidates = sorted(
            [f for f in os.listdir(_ICLOUD_BACKUP_DIR)
             if f.startswith(f"{prefix}_") and f.endswith(".db.gz")],
            reverse=True,
        )
        if not candidates:
            test_results[prefix] = "no_icloud_backup"
            continue

        src_gz = os.path.join(_ICLOUD_BACKUP_DIR, candidates[0])
        dst_db = os.path.join(restore_dir, f"{prefix}_test.db")

        try:
            t0 = time.monotonic()
            # Decompress
            with _gzip.open(src_gz, "rb") as f_in, \
                 open(dst_db, "wb") as f_out:
                while True:
                    chunk = f_in.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    f_out.write(chunk)
            decompress_sz = os.path.getsize(dst_db)

            # Integrity check
            conn = sqlite3.connect(dst_db, timeout=30)
            ic = conn.execute("PRAGMA integrity_check").fetchone()
            ic_ok = ic and ic[0] == "ok"

            # Row counts on key tables
            counts = {}
            for tbl in _KEY_TABLES.get(prefix, []):
                try:
                    n = conn.execute(
                        f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    counts[tbl] = n
                except Exception:
                    counts[tbl] = "missing"

            conn.close()
            elapsed = time.monotonic() - t0

            test_results[prefix] = {
                "source": candidates[0],
                "size_mb": round(decompress_sz / 1048576, 1),
                "integrity": "ok" if ic_ok else f"FAILED: {ic}",
                "counts": counts,
                "duration_s": round(elapsed, 1),
            }

            if not ic_ok:
                print(f"[Restore-Test] ALERT: {prefix} integrity FAILED: "
                      f"{ic}")
            else:
                print(f"[Restore-Test] {prefix}: ok, "
                      f"{decompress_sz / 1048576:.1f} MB, "
                      f"{elapsed:.1f}s, counts={counts}")

        except Exception as exc:
            test_results[prefix] = f"error: {exc}"
            print(f"[Restore-Test] ALERT: {prefix} restore FAILED: {exc}")
        finally:
            if os.path.exists(dst_db):
                os.remove(dst_db)

    # Clean up restore_test dir if empty
    try:
        os.rmdir(restore_dir)
    except OSError:
        pass

    result["restore_test"] = test_results


def _run_db_maintenance() -> dict:
    """Nightly DB maintenance: daily backup, prune old rows, checkpoint WAL.

    Retention tiers:
      - Operational logs (settle_log, scheduler_runs, injury_reports): 14 days
      - Feature data (player_projections): 30 days
      - Analytical / CLV data (line_movement_log): 90 days
      - NEVER pruned: signal_conflict_outcomes, shadow_predictions,
        shadow_calibration, mock_bets, mock_bet_legs, fixtures
    """
    import shutil
    import sqlite3
    result = {}

    bets_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
    bets_path = os.path.normpath(bets_path)

    # ── Daily backup (worst-case loss = hours, not weeks) ────────────────────
    try:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        daily_dir = os.path.join(checkpoint_dir, f"daily_{today_str}")
        if not os.path.exists(daily_dir):
            os.makedirs(daily_dir, exist_ok=True)
            # Use sqlite3 backup API for a consistent snapshot (safe even during writes)
            src_conn = sqlite3.connect(bets_path, timeout=10)
            dst_path = os.path.join(daily_dir, "bets.db")
            dst_conn = sqlite3.connect(dst_path)
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
            result["daily_backup"] = dst_path
            result["daily_backup_size"] = os.path.getsize(dst_path)
            print(f"[DB-Maintenance] Daily backup: {dst_path} "
                  f"({result['daily_backup_size'] / 1024 / 1024:.1f} MB)")

            # Integrity check on backup — alert immediately on failure
            try:
                check_conn = sqlite3.connect(dst_path, timeout=10)
                ic = check_conn.execute("PRAGMA integrity_check").fetchone()
                check_conn.close()
                if ic and ic[0] == "ok":
                    result["backup_integrity"] = "ok"
                else:
                    result["backup_integrity"] = f"FAILED: {ic}"
                    print(f"[DB-Maintenance] ALERT: backup integrity check "
                          f"FAILED: {ic}")
            except Exception as ic_exc:
                result["backup_integrity"] = f"ERROR: {ic_exc}"
                print(f"[DB-Maintenance] ALERT: backup integrity check "
                      f"error: {ic_exc}")

            # Integrity check on live DB
            try:
                live_conn = sqlite3.connect(bets_path, timeout=10)
                ic_live = live_conn.execute("PRAGMA integrity_check").fetchone()
                live_conn.close()
                if ic_live and ic_live[0] == "ok":
                    result["live_integrity"] = "ok"
                else:
                    result["live_integrity"] = f"FAILED: {ic_live}"
                    print(f"[DB-Maintenance] ALERT: live bets.db integrity "
                          f"check FAILED: {ic_live}")
            except Exception as ic_exc:
                result["live_integrity"] = f"ERROR: {ic_exc}"
                print(f"[DB-Maintenance] ALERT: live bets.db integrity "
                      f"error: {ic_exc}")

            # Prune backups older than 14 days (keep ≥2 weeks of daily snapshots)
            for entry in os.listdir(checkpoint_dir):
                if entry.startswith("daily_") and entry != f"daily_{today_str}":
                    entry_date = entry.replace("daily_", "")
                    try:
                        age = (datetime.now(timezone.utc) -
                               datetime.strptime(entry_date, "%Y%m%d").replace(
                                   tzinfo=timezone.utc))
                        if age.days > 14:
                            shutil.rmtree(os.path.join(checkpoint_dir, entry),
                                          ignore_errors=True)
                            result.setdefault("pruned_backups", []).append(entry)
                    except ValueError:
                        pass  # Not a daily backup dir
        else:
            result["daily_backup"] = "already_exists"
    except Exception as e:
        result["backup_error"] = str(e)
        print(f"[DB-Maintenance] Daily backup FAILED: {e}")

    # ── Daily backup: historical.db (VACUUM INTO — compacted, no freelist) ──
    import gzip as _gzip
    hist_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             "historical.db")
    hist_path = os.path.normpath(hist_path)
    try:
        hist_tmp = os.path.join(daily_dir, "historical.db")
        hist_gz = os.path.join(daily_dir, "historical.db.gz")
        already = os.path.exists(hist_gz) or os.path.exists(hist_tmp)
        if os.path.isdir(daily_dir) and not already:
            t0 = time.monotonic()
            # VACUUM INTO creates a consistent, compacted copy (no freelist)
            vac = sqlite3.connect(hist_path, timeout=30)
            vac.execute("PRAGMA busy_timeout = 30000")
            vac.execute(f"VACUUM INTO '{hist_tmp}'")
            vac.close()
            vacuum_sz = os.path.getsize(hist_tmp)
            result["hist_vacuum_size"] = vacuum_sz
            print(f"[DB-Maintenance] historical.db VACUUM INTO: "
                  f"{vacuum_sz / 1048576:.1f} MB "
                  f"({time.monotonic() - t0:.1f}s)")

            # Integrity check on the compacted copy
            ic_ok = False
            try:
                c = sqlite3.connect(hist_tmp, timeout=10)
                ic = c.execute("PRAGMA integrity_check").fetchone()
                c.close()
                if ic and ic[0] == "ok":
                    result["hist_integrity"] = "ok"
                    ic_ok = True
                else:
                    result["hist_integrity"] = f"FAILED: {ic}"
                    print(f"[DB-Maintenance] ALERT: historical.db VACUUM "
                          f"integrity FAILED: {ic}")
            except Exception as _ic_e:
                result["hist_integrity"] = f"ERROR: {_ic_e}"
                print(f"[DB-Maintenance] ALERT: historical.db integrity "
                      f"error: {_ic_e}")

            if ic_ok:
                # Gzip the compacted copy, then delete uncompressed
                with open(hist_tmp, "rb") as f_in, \
                     _gzip.open(hist_gz, "wb") as f_out:
                    while True:
                        chunk = f_in.read(1 << 20)  # 1 MB
                        if not chunk:
                            break
                        f_out.write(chunk)
                gz_sz = os.path.getsize(hist_gz)
                os.remove(hist_tmp)
                result["hist_backup"] = hist_gz
                result["hist_backup_size"] = gz_sz
                elapsed = time.monotonic() - t0
                result["hist_backup_duration_s"] = round(elapsed, 1)
                print(f"[DB-Maintenance] historical.db compressed: "
                      f"{gz_sz / 1048576:.1f} MB (total {elapsed:.1f}s)")

                # Keep only 1 local compressed generation
                for entry in os.listdir(checkpoint_dir):
                    if (entry.startswith("daily_")
                            and entry != os.path.basename(daily_dir)):
                        old_gz = os.path.join(checkpoint_dir, entry,
                                              "historical.db.gz")
                        if os.path.exists(old_gz):
                            os.remove(old_gz)
                            result.setdefault(
                                "hist_pruned_local", []).append(
                                os.path.join(entry, "historical.db.gz"))
            else:
                # Integrity failed — keep uncompressed for diagnosis
                result["hist_backup"] = f"integrity_failed:{hist_tmp}"
                print(f"[DB-Maintenance] historical.db kept uncompressed "
                      f"(integrity failed): {hist_tmp}")
        else:
            result["hist_backup"] = "already_exists"
    except Exception as e:
        result["hist_backup_error"] = str(e)
        print(f"[DB-Maintenance] historical.db backup FAILED: {e}")
        # Clean up partial VACUUM INTO on error
        _htmp = os.path.join(daily_dir, "historical.db")
        if os.path.exists(_htmp):
            try:
                os.remove(_htmp)
            except OSError:
                pass

    # ── Off-machine copy to iCloud Drive ──────────────────────────────────────
    try:
        _icloud_backup(daily_dir, result)
    except Exception as e:
        result["icloud_error"] = str(e)
        print(f"[DB-Maintenance] iCloud backup FAILED: {e}")

    # ── Monthly restore test (1st of each month) ────────────────────────────
    if datetime.now(timezone.utc).day == 1:
        try:
            _run_restore_test(result)
        except Exception as e:
            result["restore_test_error"] = str(e)
            print(f"[DB-Maintenance] Restore test FAILED: {e}")

    # ── bets.db: prune old rows ──────────────────────────────────────────────
    # Acquire DB_WRITE_LOCK to serialize with other bets.db writers
    # (mock_bets, auto_settle, scout, etc.) and avoid SQLITE_LOCKED.
    try:
        with DB_WRITE_LOCK:
            con = sqlite3.connect(bets_path, timeout=60)
            con.execute("PRAGMA busy_timeout = 60000")
            con.execute("PRAGMA journal_size_limit=67108864")
            total_pruned = 0

            # (table, column, retention_days, is_date_only)
            _PRUNE_RULES = [
                ("settle_log",          "settled_at",  14, False),
                ("scheduler_runs",      "run_at",      14, False),
                # injury_reports:           EXCLUDED — needed by injury watcher replay
                # player_projections:       EXCLUDED — needed by NBA S5-2 forward test
                # line_movement_log:        EXCLUDED — odds data needed by forward tests
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
                    print(f"[DB-Maintenance] ALERT: prune {tbl} failed: {exc}")

            # WAL checkpoint — PASSIVE only. NEVER use TRUNCATE or RESTART:
            # TRUNCATE rewrites page 1 and corrupts the database if interrupted
            # (confirmed root cause of recurring bets.db corruption Oct 2026).
            # PASSIVE is always safe — it checkpoints whatever it can without
            # blocking readers. If WAL grows large, it's harmless.
            try:
                wal_result = con.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                wal_pages = wal_result[1] if wal_result else 0
                checkpointed = wal_result[2] if wal_result else 0
                remaining = wal_pages - checkpointed if wal_pages and checkpointed else 0
                result["wal_passive"] = {"total": wal_pages, "checkpointed": checkpointed}
                if remaining > 100:
                    print(f"[DB-Maintenance] WAL: PASSIVE left {remaining} pages "
                          f"(will checkpoint on next cycle, NOT escalating)")
                else:
                    print(f"[DB-Maintenance] bets.db WAL checkpoint: ok "
                          f"({checkpointed}/{wal_pages} pages)")
            except Exception as wal_exc:
                result["wal_error"] = str(wal_exc)
                print(f"[DB-Maintenance] ALERT: bets.db WAL checkpoint "
                      f"failed: {wal_exc}")

            con.commit()
            con.close()
            result["bets_pruned_total"] = total_pruned
            print(f"[DB-Maintenance] bets.db prune: ok "
                  f"({total_pruned} rows removed)")
    except Exception as e:
        result["bets_error"] = str(e)
        print(f"[DB-Maintenance] ALERT: bets.db prune/checkpoint failed: {e}")

    # ── historical.db: WAL checkpoint (no row pruning — props/lines preserved) ─
    try:
        hist_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        hist_path = os.path.normpath(hist_path)
        con = sqlite3.connect(hist_path, timeout=60)
        con.execute("PRAGMA busy_timeout = 60000")
        con.execute("PRAGMA journal_size_limit=67108864")
        # player_props / alt_lines: NEVER pruned — forward tests need full history
        wal_h = con.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        h_pages = wal_h[1] if wal_h else 0
        h_done = wal_h[2] if wal_h else 0
        result["hist_wal"] = {"total": h_pages, "checkpointed": h_done}
        print(f"[DB-Maintenance] historical.db WAL checkpoint: ok "
              f"({h_done}/{h_pages} pages)")
        con.commit()
        con.close()
    except Exception as e:
        result["historical_error"] = str(e)
        print(f"[DB-Maintenance] ALERT: historical.db WAL checkpoint "
              f"failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    # Check for any failures across all steps
    failures = [k for k in result if "error" in k.lower() or
                (isinstance(result[k], str) and "FAILED" in result[k])]
    if failures:
        _notify("BetIQ Maintenance",
                f"{len(failures)} step(s) failed: {', '.join(failures)}")
    else:
        print(f"[DB-Maintenance] All steps completed successfully")

    # INFRA-2: Persist maintenance results to bets.db for 7-day audit
    try:
        import json as _j
        _mcon = sqlite3.connect(bets_path, timeout=30)
        _mcon.execute("PRAGMA busy_timeout = 30000")
        _mcon.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                run_utc TEXT NOT NULL,
                result_json TEXT NOT NULL,
                has_errors INTEGER NOT NULL DEFAULT 0
            )
        """)
        _mcon.execute("CREATE INDEX IF NOT EXISTS idx_maint_date "
                      "ON maintenance_log(run_date)")
        _now = datetime.now(timezone.utc)
        _mcon.execute(
            "INSERT INTO maintenance_log (run_date, run_utc, result_json, has_errors) "
            "VALUES (?, ?, ?, ?)",
            (_now.strftime("%Y-%m-%d"), _now.isoformat(),
             _j.dumps(result, default=str), 1 if failures else 0),
        )
        # Prune entries older than 30 days
        _mcon.execute(
            "DELETE FROM maintenance_log WHERE run_date < date('now', '-30 days')")
        _mcon.commit()
        _mcon.close()
        print(f"[DB-Maintenance] INFRA-2: result persisted to maintenance_log")
    except Exception as _pe:
        print(f"[DB-Maintenance] INFRA-2: persist error (non-fatal): {_pe}")

    return result


def _run_prop_health_check() -> dict:
    """Daily health check: verify captured prop rows match expectations.

    Checks historical.db:player_props for today's data. Alerts if:
    - Zero rows for any game that had events available
    - Any market has zero rows (e.g. alternate lines missing)
    - Any expected book has zero rows

    Returns summary dict; prints ALERT on any failure.
    """
    import sqlite3
    result = {}
    try:
        hist_path = os.path.join(os.path.dirname(__file__), "..", "data",
                                 "historical.db")
        hist_path = os.path.normpath(hist_path)
        con = sqlite3.connect(hist_path, timeout=10)
        today = _now_ct().strftime("%Y-%m-%d")

        # Rows per game (event_id) today
        rows = con.execute(
            "SELECT event_id, COUNT(*) as cnt, "
            "COUNT(DISTINCT market_key) as markets, "
            "COUNT(DISTINCT bookmaker) as books "
            "FROM player_props "
            "WHERE sport_key = 'basketball_nba' "
            "AND DATE(fetched_at) = ? "
            "GROUP BY event_id", (today,)
        ).fetchall()
        con.close()

        result["date"] = today
        result["games_with_data"] = len(rows)

        if not rows:
            result["alert"] = "ZERO prop rows captured today"
            _notify("BetIQ PropHealth", result["alert"])
        else:
            total_rows = sum(r[1] for r in rows)
            min_markets = min(r[2] for r in rows) if rows else 0
            min_books = min(r[3] for r in rows) if rows else 0
            result["total_rows"] = total_rows
            result["min_markets_per_game"] = min_markets
            result["min_books_per_game"] = min_books

            alerts = []
            for event_id, cnt, mkts, bks in rows:
                if cnt < 50:
                    alerts.append(f"{event_id}: only {cnt} rows")
                if mkts < 3:
                    alerts.append(f"{event_id}: only {mkts} markets")
            if alerts:
                result["alerts"] = alerts
                _notify("BetIQ PropHealth",
                        f"{len(alerts)} game(s) with low coverage")
            else:
                result["status"] = "healthy"
                print(f"[PropHealth] OK: {len(rows)} games, "
                      f"{total_rows} rows, "
                      f"min {min_markets}/game markets, "
                      f"min {min_books}/game books")
    except Exception as exc:
        result["error"] = str(exc)
        print(f"[PropHealth] ERROR: {exc}")
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
        _scheduler_state["watchdog_db_write_verified"] = False
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

    # ── CHECK 7: DB write-health (detect corruption before 23 days of silent failure)
    if not _scheduler_state.get("watchdog_db_write_verified"):
        try:
            _db_ok, _db_detail = _verify_db_write_health()
            if _db_ok:
                _scheduler_state["watchdog_db_write_verified"] = True
            else:
                print(f"[Watchdog] 🚨 DB WRITE HEALTH FAILED: {_db_detail}")
                print(f"[Watchdog] bets.db may be corrupted — all shadow arms are non-functional!")
        except Exception as _e:
            print(f"[Watchdog] DB write-health check error: {_e}")

    # ── CHECK 8: Retrain subprocess status ─────────────────────────────────
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

    _HEARTBEAT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".betiq_heartbeat"
    )

    while not _stop_event.is_set():
        now_utc = datetime.now(timezone.utc)
        now_ct  = _now_ct()
        today_ct = now_ct.strftime("%Y-%m-%d")

        _scheduler_state["last_run"]    = now_utc.isoformat()
        _scheduler_state["total_cycles"] += 1

        # ── Heartbeat file — external watchdog reads this ──────────────────
        try:
            import time as _hb_time
            with open(_HEARTBEAT_FILE, "w") as _hb:
                _hb.write(f"{int(_hb_time.time())}\n")
        except Exception:
            pass

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

        # S5-2: Grade NBA forward preds after game logs are updated
        try:
            if _scheduler_state.get("last_ats_update") == today_ct:
                from scout.runner import grade_nba_forward_preds
                _grade = grade_nba_forward_preds()
                if _grade.get("graded", 0) > 0:
                    print(f"[Scheduler] S5-2 grading: {_grade}")
        except Exception as _exc:
            _notify("BetIQ S5-2", f"Grading error: {_exc}", sound="Basso")

        _dispatch("nightly_mlb",         _should_run_nightly_mlb,           _run_nightly_mlb_update,        "last_mlb_update")
        _dispatch("nightly_nhl",         _should_run_nightly_nhl,           _run_nightly_nhl_update,        "last_nhl_update")
        _dispatch("nhl_etl",             _should_run_daily_nhl_etl,         _run_daily_nhl_etl,             "last_nhl_etl_update")
        _dispatch("mlb_etl",             _should_run_daily_mlb_etl,         _run_daily_mlb_etl,             "last_mlb_etl_update")
        _dispatch("fanduel_sync",        _should_run_weekly_fanduel_sync,   _run_weekly_fanduel_sync,       "last_fanduel_sync")
        _dispatch("personal_edge_refresh", _should_run_personal_edge_refresh, _run_personal_edge_refresh,      "last_personal_edge_refresh")
        _dispatch("signal_analysis",     _should_run_weekly_signal_analysis,_run_weekly_signal_analysis,    "last_signal_analysis")
        _dispatch("weekly_retrain",      _should_run_weekly_retrain,        _run_weekly_retrain,            "last_weekly_retrain")
        _dispatch("negbin_retrain",      _should_run_monthly_negbin_retrain, _run_monthly_negbin_retrain,    "last_negbin_retrain", use_now_utc=False)
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
        # Weather refresh — DISABLED 2026-07-23: all 4 weather features removed from
        # NegBin MU_FEATURE_COLS (harmful/noise under fixed multi-seed harness).
        # No model reads game_weather. Keep code + table for potential future use.
        # To re-enable: uncomment the _dispatch line below.
        # _dispatch("weather_refresh",     _should_run_weather_refresh,       _run_weather_refresh,           "last_weather_refresh", use_now_utc=False)
        # NegBin shadow AM — MUST run after alt_lines_morning complete.
        _dispatch("negbin_shadow_am",    _should_run_negbin_shadow_am,      _run_negbin_shadow_am,          "last_negbin_shadow_am", use_now_utc=False)
        # Shadow B (recalibrated) stamped inside _run_negbin_shadow_am — same fixture snapshot
        # ATS ungated shadow — scores ALL MLB spread legs (no gates) for discrimination analysis
        _dispatch("ats_ungated_shadow", _should_run_ats_ungated_shadow, _run_ats_ungated_shadow, "last_ats_ungated_shadow", use_now_utc=False)
        # Lineup shadow — ATS + lineup_gap_resid, parallel to baseline for forward comparison
        _dispatch("lineup_shadow", _should_run_lineup_shadow, _run_lineup_shadow, "last_lineup_shadow", use_now_utc=False)
        # Alt-line shadow — forward validation of mlb_ats_v1 alt-spread edge (real book prices)
        _dispatch("altline_shadow", _should_run_altline_shadow, _run_altline_shadow, "last_altline_shadow", use_now_utc=False)
        # F5 shadow — First-5-Innings spread predictions + settlement
        _dispatch("f5_shadow", _should_run_f5_shadow, _run_f5_shadow, "last_f5_shadow", use_now_utc=False)
        # Underdog lag shadow — cross-book lag edge forward test (needs AM + PM alt_lines)
        _dispatch("underdog_lag_shadow", _should_run_underdog_lag_shadow, _run_underdog_lag_shadow, "last_underdog_lag_shadow", use_now_utc=False)
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
        # 7 AM CT: NBA-only morning prop snapshot (before scout runs at 9 AM)
        # Captures opening prop odds for S5-2 model-vs-market comparison.
        # ~18 credits/day for a 6-game slate. Stored in historical.db:player_props.
        try:
            if _should_run_props("7am"):
                p7_result = _run_nba_props_morning()
                # Only mark done if no error — allows retry on next cycle
                if "error" not in p7_result:
                    _scheduler_state["last_props_7am"]        = today_ct
                else:
                    print(f"[Scheduler] props_7am FAILED, will retry: {p7_result['error']}")
                _scheduler_state["last_props_7am_result"] = p7_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in props_7am: {_exc}")

        try:
            if _should_run_props("11am"):
                p11_result = _run_props_fetch("11am")
                if "error" not in p11_result:
                    _scheduler_state["last_props_11am"]        = today_ct
                else:
                    print(f"[Scheduler] props_11am FAILED, will retry: {p11_result['error']}")
                _scheduler_state["last_props_11am_result"] = p11_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in props_11am: {_exc}")

        try:
            if _should_run_props("4pm"):
                p4_result = _run_props_fetch("4pm")
                if "error" not in p4_result:
                    _scheduler_state["last_props_4pm"]        = today_ct
                else:
                    print(f"[Scheduler] props_4pm FAILED, will retry: {p4_result['error']}")
                _scheduler_state["last_props_4pm_result"] = p4_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in props_4pm: {_exc}")

        # Pre-tip NBA injury refresh — capture latest report before prop snapshot
        try:
            if _should_run_pretip_props():
                _log_nba_injury_forward()
        except Exception as _exc:
            print(f"[Scheduler] Pre-tip injury refresh error (non-fatal): {_exc}")

        # Pre-tip NBA props — every cycle 10 AM-11 PM CT, targets games tipping in 30-60 min
        try:
            if _should_run_pretip_props():
                pretip_result = _run_pretip_props()
                pretip_events = pretip_result.get("event_ids", [])
                if pretip_events:
                    # Track which events got a pre-tip snapshot today
                    done = _scheduler_state.get("pretip_done_events", set())
                    done.update(pretip_events)
                    _scheduler_state["pretip_done_events"] = done
                    _scheduler_state["last_pretip_result"] = pretip_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in pretip_props: {_exc}")

        # S5-2: Pre-tip forward predictions — re-run sim with latest injuries
        try:
            if _should_run_pretip_props():
                from scout.runner import run_pretip_nba_forward_preds
                _fp_result = run_pretip_nba_forward_preds()
                if _fp_result.get("rows", 0) > 0:
                    print(f"[Scheduler] S5-2 pre-tip forward preds: "
                          f"{_fp_result['games']} games, {_fp_result['rows']} rows")
        except Exception as _exc:
            _notify("BetIQ S5-2", f"Pre-tip forward preds error: {_exc}", sound="Basso")

        try:
            if _should_run_best_lines():
                bl_result = _run_best_lines_daily()
                _scheduler_state["last_best_lines"]        = today_ct
                _scheduler_state["last_best_lines_result"] = bl_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in best_lines: {_exc}")

        # Prop capture health check — once daily after 4 PM slot
        try:
            if (hour_ct >= 17
                    and _scheduler_state.get("last_prop_health") != today_ct):
                _prop_health = _run_prop_health_check()
                _scheduler_state["last_prop_health"] = today_ct
                _scheduler_state["last_prop_health_result"] = _prop_health
                # Record first day of complete capture
                if (_prop_health.get("status") == "healthy"
                        and not _scheduler_state.get("first_complete_capture_day")):
                    _scheduler_state["first_complete_capture_day"] = today_ct
                    print(f"[PropHealth] Forward test effective day 1: {today_ct}")
        except Exception as _exc:
            print(f"[Scheduler] ERROR in prop_health_check: {_exc}")

        # Wong teaser shadow — NFL structural edge forward validation
        _dispatch("teaser_shadow", _should_run_teaser_shadow, _run_teaser_shadow, "last_teaser_shadow", use_now_utc=False)
        # Prop shadow — batter HR + TB (runs after 4 PM props fetch)
        _dispatch("prop_shadow", _should_run_prop_shadow, _run_prop_shadow, "last_prop_shadow", use_now_utc=False)

        # Fantasy projections — NFL weekly consensus + prop gap analysis
        # Runs once per day on Wednesdays-Saturdays during NFL season (Sep-Jan)
        try:
            ct_now = _now_ct()
            _is_nfl_season = ct_now.month >= 9 or ct_now.month <= 1
            _is_proj_day = ct_now.weekday() in (2, 3, 4, 5)  # Wed-Sat
            _proj_ran_today = _scheduler_state.get("last_fantasy_proj") == today_ct
            if _is_nfl_season and _is_proj_day and not _proj_ran_today and ct_now.hour >= 10:
                from fantasy_projections import run_weekly_pipeline as _run_fp
                fp_result = _run_fp(fetch_props=True)
                _scheduler_state["last_fantasy_proj"]        = today_ct
                _scheduler_state["last_fantasy_proj_result"] = {
                    "n_gaps": fp_result.get("n_gaps", 0),
                    "sources": fp_result.get("sources", []),
                    "n_matched": fp_result.get("n_matched", 0),
                }
                print(f"[Scheduler] Fantasy projections: {fp_result.get('n_gaps', 0)} gaps found")
        except Exception as _exc:
            print(f"[Scheduler] ERROR in fantasy_projections: {_exc}")

        # Fantasy forward grading — grade last week's predictions on Tuesdays (after MNF settles)
        try:
            ct_now = _now_ct()
            _is_nfl_season_g = ct_now.month >= 9 or ct_now.month <= 1
            _grade_ran_today = _scheduler_state.get("last_fantasy_grade") == today_ct
            if _is_nfl_season_g and ct_now.weekday() == 1 and not _grade_ran_today and ct_now.hour >= 12:
                from fantasy_projections import grade_forward_predictions as _grade_fp, get_current_nfl_week as _get_week
                _s, _w = _get_week()
                _grade_week = max(1, _w - 1)  # grade last week
                _grade_result = _grade_fp(_s, _grade_week)
                _scheduler_state["last_fantasy_grade"] = today_ct
                _graded = _grade_result.get("graded", 0)
                print(f"[Scheduler] Fantasy forward grade: week {_grade_week}, {_graded} predictions graded")
        except Exception as _exc:
            print(f"[Scheduler] ERROR in fantasy forward grading: {_exc}")

        # v1.2 prediction coverage check — once/day Wed-Sun 11 AM CT during NFL season
        try:
            ct_now_pc = _now_ct()
            _is_nfl_pc = ct_now_pc.month >= 9 or ct_now_pc.month <= 1
            _is_check_day_pc = ct_now_pc.weekday() in (2, 3, 4, 5, 6)  # Wed-Sun
            _pc_ran_today = _scheduler_state.get("last_pred_coverage_check") == today_ct
            if _is_nfl_pc and _is_check_day_pc and not _pc_ran_today and ct_now_pc.hour >= 11:
                from nfl_calibration_diagnosis import check_prediction_coverage
                _pc_result = check_prediction_coverage()
                _scheduler_state["last_pred_coverage_check"] = today_ct
                _scheduler_state["last_pred_coverage_result"] = {
                    "ok": _pc_result.get("ok"),
                    "total_preds": _pc_result.get("total_preds", 0),
                    "missing": _pc_result.get("missing", []),
                }
                if _pc_result.get("missing"):
                    print(f"[Scheduler] ⚠ v1.2 coverage gap: "
                          f"{len(_pc_result['missing'])} games missing predictions")
        except Exception as _exc:
            print(f"[Scheduler] ERROR in pred_coverage_check: {_exc}")

        # Researcher — daily news + synthesis (runs once after 9 AM CT)
        try:
            if _should_run_researcher():
                res_result = _run_researcher()
                _scheduler_state["last_researcher"]        = today_ct
                _scheduler_state["last_researcher_result"] = res_result
        except Exception as _exc:
            print(f"[Scheduler] ERROR in researcher: {_exc}")

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

        # ── Injury watcher — Sleeper status poll (every 15 min, NFL game days) ──
        try:
            ct_now_iw = _now_ct()
            _is_nfl_season_iw = ct_now_iw.month >= 9 or ct_now_iw.month <= 1
            _is_game_day_iw = ct_now_iw.weekday() in (0, 3, 4, 5, 6)  # Mon,Thu,Fri,Sat,Sun
            if _is_nfl_season_iw and _is_game_day_iw and ct_now_iw.hour >= 10:
                _last_iw = _scheduler_state.get("last_injury_watch_poll")
                _iw_ok = True
                if _last_iw:
                    try:
                        _iw_dt = datetime.fromisoformat(_last_iw)
                        if _iw_dt.tzinfo is None:
                            _iw_dt = _iw_dt.replace(tzinfo=timezone.utc)
                        _iw_ok = (datetime.now(timezone.utc) - _iw_dt).total_seconds() >= 840
                    except Exception:
                        pass
                if _iw_ok:
                    from injury_watcher import run_poll_cycle as _iw_poll
                    _iw_ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    _iw_result = _iw_poll()
                    _scheduler_state["last_injury_watch_poll"]        = datetime.now(timezone.utc).isoformat()
                    _scheduler_state["last_injury_watch_poll_result"] = _iw_result
                    _iw_ch = _iw_result.get("changes", 0)
                    _iw_tr = _iw_result.get("triggers", 0)
                    print(f"[Scheduler] InjuryWatcher heartbeat {_iw_ts} UTC — "
                          f"{_iw_ch} changes, {_iw_tr} triggers"
                          + (f" ({_iw_result['credits']} credits)" if _iw_tr > 0 else ""))
            # End-of-day alert: if it's 11 PM CT on a game day and we had
            # zero triggered changes all day, something is wrong
            if ct_now_iw.hour == 23 and _scheduler_state.get("iw_alerted_today") != today_ct:
                _iw_day_triggers = _scheduler_state.get("iw_triggers_today", 0)
                if _iw_ch > 0:
                    _scheduler_state["iw_triggers_today"] = _iw_day_triggers + _iw_tr
                if _scheduler_state.get("iw_triggers_today", 0) == 0:
                    print(f"[InjuryWatcher] ALERT: 0 triggered changes on game day "
                          f"{today_ct}. Check trigger_prop_fetch() and Sleeper API.")
                _scheduler_state["iw_alerted_today"] = today_ct
            elif ct_now_iw.hour < 23:
                # Accumulate daily trigger count
                if _iw_tr > 0:
                    _scheduler_state["iw_triggers_today"] = (
                        _scheduler_state.get("iw_triggers_today", 0) + _iw_tr)
                # Reset counter on new day
                if _scheduler_state.get("iw_count_date") != today_ct:
                    _scheduler_state["iw_count_date"] = today_ct
                    _scheduler_state["iw_triggers_today"] = _iw_tr
        except Exception as _iw_exc:
            print(f"[Scheduler] ERROR in injury_watch_poll: {_iw_exc}")

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

    # ── SQLite version check (WAL-Reset bug) ──
    import sqlite3 as _sq_ver
    _sv = _sq_ver.sqlite_version
    _svt = tuple(int(x) for x in _sv.split("."))
    if _svt < (3, 44, 6):
        print(f"[Scheduler] WARNING: SQLite {_sv} is vulnerable to WAL-Reset "
              f"bug (silent lost writes). Upgrade to >= 3.51.3.")
    else:
        print(f"[Scheduler] SQLite {_sv} — WAL-Reset safe")

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
    """Signal the scheduler to stop after the current sleep.

    Performs a clean WAL checkpoint on shutdown to avoid leaving
    a dirty WAL that could corrupt page 1 if the process is killed
    during an auto-checkpoint on next startup.
    """
    global _scheduler_state
    _stop_event.set()
    _scheduler_state["running"] = False
    print("[Scheduler] Stop requested.")

    # Clean WAL checkpoint on shutdown — PASSIVE is always safe
    try:
        import sqlite3 as _sq
        _db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq.connect(_db, timeout=5)
        _wal = _con.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        _con.close()
        print(f"[Scheduler] Shutdown WAL checkpoint: {_wal}")
    except Exception as _e:
        print(f"[Scheduler] Shutdown WAL checkpoint failed (non-fatal): {_e}")


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
