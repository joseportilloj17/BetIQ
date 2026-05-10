"""
creator_tier.py — TheOddsAPI Creator Tier feature set.

Credit budget: 20,000 / month on Creator tier.
Target operational spend: < 7,000 / month.
Reserve 3,000 for CLV backfill + playoffs + manual queries.
Sustainable daily burn rate: 233 credits / day.

Fetch strategy (replaces 30-min universal snapshot):
  - fetch_imminent_games_odds(): games starting within 6 hours,
    every 45 min, 8 AM – 11 PM local time only.
  - fetch_player_props(): same-day games only, twice daily.
  - Best lines: once daily at 10 AM local time.
  - check_credit_budget(): guard before every scheduled fetch.

Tables (auto-created on import)
--------------------------------
  player_props   — historical.db
  line_snapshots — historical.db
  credits_log    — historical.db  (every API call logged)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
import calendar

import requests

# ─── Config ───────────────────────────────────────────────────────────────────

_ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
_BASE_URL = "https://api.the-odds-api.com/v4"
_HIST_DB  = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

# Sports that have close_spread / ML data we want to backfill via Creator tier
_CLV_SPORTS: dict[str, str] = {
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
}

# Player-prop markets to fetch per sport (TheOddsAPI market keys)
_PROP_MARKETS: dict[str, list[str]] = {
    "baseball_mlb": [
        "pitcher_strikeouts",
        "batter_hits",
        "batter_home_runs",
        "batter_rbis",
        "batter_runs_scored",
        "batter_total_bases",
        "pitcher_earned_runs",
        "pitcher_hits_allowed",
        "pitcher_walks",
    ],
    "basketball_nba": [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_blocks",
        "player_steals",
        "player_points_rebounds_assists",
    ],
    "icehockey_nhl": [
        "player_points",
        "player_goals",
        "player_assists",
        "player_shots_on_goal",
        "player_saves",
    ],
    "americanfootball_nfl": [
        "player_pass_tds",
        "player_pass_yds",
        "player_rush_yds",
        "player_receptions",
        "player_reception_yds",
        "player_anytime_td",
    ],
}

# Bookmakers to pull when shopping lines
_SHOP_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pinnacle",
               "betrivers", "pointsbetus", "espnbet"]

# Bookmakers to pull for player props
_PROP_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars",
               "pinnacle", "betrivers", "pointsbetus", "fanatics", "espnbet"]

# Alternate line market keys per sport
_ALT_MARKETS: dict[str, list[str]] = {
    "baseball_mlb":              ["alternate_spreads", "alternate_totals"],
    "icehockey_nhl":             ["alternate_spreads", "alternate_totals"],
    "basketball_nba":            ["alternate_spreads", "alternate_totals"],
    "americanfootball_nfl":      ["alternate_spreads", "alternate_totals"],
    "soccer_epl":                ["alternate_spreads", "alternate_totals"],
    "soccer_spain_la_liga":      ["alternate_spreads", "alternate_totals"],
    "soccer_germany_bundesliga": ["alternate_spreads", "alternate_totals"],
    "soccer_france_ligue_one":   ["alternate_spreads", "alternate_totals"],
    "soccer_italy_serie_a":      ["alternate_spreads", "alternate_totals"],
    "soccer_uefa_champs_league": ["alternate_spreads", "alternate_totals"],
    "soccer_uefa_europa_league": ["alternate_spreads", "alternate_totals"],
}

# Sport keys for the scheduled twice-daily batch alt-lines fetch
_ALT_BATCH_SPORT_KEYS: list[str] = [
    "baseball_mlb",
    "icehockey_nhl",
    "basketball_nba",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_italy_serie_a",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
]

# Per-unit line shift → win-prob adjustment (percentage points per spread point)
_SPREAD_ADJ_PP: float = 3.0

# Alt line scoring thresholds
_ANCHOR_PROB: float = 72.0   # our_prob >= this → ANCHOR candidate
_ANCHOR_EDGE: float = 2.0    # edge_pp >= this → ANCHOR
_VALUE_EDGE:  float = 3.0    # edge_pp >= this → VALUE

# ─── Credit budget constants ───────────────────────────────────────────────────
MONTHLY_BUDGET       = 20_000   # Creator tier allocation
OPERATIONAL_TARGET   = 7_000    # target spend for recurring fetches
RESERVE_CREDITS      = 3_000    # reserved for CLV backfill + playoffs + manual

# Thresholds trigger progressively restrictive modes
THRESHOLD_SKIP       = 1_500    # remaining < this → skip non-essential fetches
THRESHOLD_EMERGENCY  = 200      # remaining < this → skip alt lines, use cached only
THRESHOLD_SUSPENDED  = 100      # remaining < this → all fetches suspended

SUSTAINABLE_DAILY    = 233      # operational_target / 30 days
MAX_DAILY_WARN       = 333      # monthly_budget / 30; warn if exceeded

# ─── DB helpers ───────────────────────────────────────────────────────────────

def _hist_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables():
    """Create player_props, line_snapshots, and credits_log tables if they don't exist."""
    conn = _hist_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS credits_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at         TEXT NOT NULL,
            endpoint          TEXT NOT NULL,
            credits_used      INTEGER,
            credits_remaining INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_credits_log_at ON credits_log(logged_at);

        CREATE TABLE IF NOT EXISTS player_props (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at     TEXT NOT NULL,
            sport_key      TEXT NOT NULL,
            event_id       TEXT NOT NULL,
            game_date      TEXT,
            home_team      TEXT,
            away_team      TEXT,
            bookmaker      TEXT NOT NULL,
            market_key     TEXT NOT NULL,
            player_name    TEXT NOT NULL,
            description    TEXT,
            price          REAL,
            point          REAL,
            over_under     TEXT,
            UNIQUE(event_id, bookmaker, market_key, player_name, over_under, fetched_at)
        );

        CREATE TABLE IF NOT EXISTS line_snapshots (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at    TEXT NOT NULL,
            sport_key      TEXT NOT NULL,
            event_id       TEXT NOT NULL,
            game_date      TEXT,
            home_team      TEXT,
            away_team      TEXT,
            bookmaker      TEXT NOT NULL,
            market_key     TEXT NOT NULL,
            outcome_name   TEXT NOT NULL,
            price          REAL,
            point          REAL
        );

        CREATE INDEX IF NOT EXISTS idx_line_snap_event
            ON line_snapshots(event_id, captured_at);
        CREATE INDEX IF NOT EXISTS idx_player_props_event
            ON player_props(event_id, market_key, fetched_at);

        CREATE TABLE IF NOT EXISTS alt_lines (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at     TEXT NOT NULL,
            sport_key      TEXT NOT NULL,
            event_id       TEXT NOT NULL,
            game_date      TEXT,
            home_team      TEXT,
            away_team      TEXT,
            bookmaker      TEXT NOT NULL,
            market_key     TEXT NOT NULL,
            team           TEXT,
            line           REAL,
            over_under     TEXT,
            odds           REAL NOT NULL,
            is_main_market INTEGER DEFAULT 0,
            UNIQUE(event_id, bookmaker, market_key, team, over_under, fetched_at)
        );
        CREATE INDEX IF NOT EXISTS idx_alt_lines_event
            ON alt_lines(event_id, fetched_at);
    """)
    conn.commit()
    conn.close()


# ─── TheOddsAPI helpers ───────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, retries: int = 3) -> dict | list | None:
    from http_retry import get_with_retry
    params = dict(params)
    params["apiKey"] = _ODDS_API_KEY
    url = f"{_BASE_URL}{endpoint}"
    r = get_with_retry(url, params=params, timeout=20, max_attempts=retries, label="CreatorTier")
    if r is None:
        return None
    if r.status_code == 422:
        print(f"[CreatorTier] 422 Unprocessable: {r.text[:200]}")
        return None
    remaining_str = r.headers.get("x-requests-remaining", "")
    used_str      = r.headers.get("x-requests-last", "")
    remaining = int(remaining_str) if remaining_str.isdigit() else None
    used      = int(used_str)      if used_str.isdigit()      else None
    print(f"[CreatorTier] {endpoint.split('?')[0][-60:]}  cost={used} remaining={remaining}")
    _log_credits(endpoint, used, remaining)
    return r.json()


def _log_credits(endpoint: str, used: int | None, remaining: int | None):
    """Write one row to credits_log. Fire-and-forget — never raises."""
    try:
        _ensure_tables()
        conn = _hist_conn()
        conn.execute(
            "INSERT INTO credits_log (logged_at, endpoint, credits_used, credits_remaining)"
            " VALUES (?,?,?,?)",
            (datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
             endpoint[:200], used, remaining),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Credit budget management
# ══════════════════════════════════════════════════════════════════════════════

def get_credit_status() -> dict:
    """
    Read credits_log and return budget dashboard data.

    Returns
    -------
    {
      credits_remaining,   # last known value from API headers
      credits_used_today,  # sum of credits_used since midnight UTC
      credits_used_month,  # sum since start of current calendar month
      burn_rate_daily,     # 7-day rolling average (credits/day)
      days_until_reset,    # calendar days left in current month
      est_month_total,     # projected total if burn rate continues
      status,              # "ok" / "warn" / "skip" / "emergency" / "suspended"
      budget,              # MONTHLY_BUDGET constant
      target,              # OPERATIONAL_TARGET constant
    }
    """
    _ensure_tables()
    conn = _hist_conn()
    now_utc   = datetime.utcnow()
    today_str = now_utc.strftime("%Y-%m-%d")
    month_str = now_utc.strftime("%Y-%m")

    # Latest credits_remaining from any call
    row = conn.execute(
        "SELECT credits_remaining FROM credits_log"
        " WHERE credits_remaining IS NOT NULL"
        " ORDER BY logged_at DESC LIMIT 1"
    ).fetchone()
    remaining = row[0] if row else None

    # Credits used today
    used_today = conn.execute(
        "SELECT COALESCE(SUM(credits_used),0) FROM credits_log"
        " WHERE logged_at >= ? AND credits_used IS NOT NULL",
        (f"{today_str}T00:00:00Z",),
    ).fetchone()[0]

    # Credits used this calendar month
    used_month = conn.execute(
        "SELECT COALESCE(SUM(credits_used),0) FROM credits_log"
        " WHERE logged_at >= ? AND credits_used IS NOT NULL",
        (f"{month_str}-01T00:00:00Z",),
    ).fetchone()[0]

    # 7-day rolling burn rate (credits/day)
    week_ago  = (now_utc - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    used_7d   = conn.execute(
        "SELECT COALESCE(SUM(credits_used),0) FROM credits_log"
        " WHERE logged_at >= ? AND credits_used IS NOT NULL",
        (week_ago,),
    ).fetchone()[0]
    burn_daily = round(used_7d / 7, 1)

    # Days left in month
    days_in_month = calendar.monthrange(now_utc.year, now_utc.month)[1]
    days_left     = days_in_month - now_utc.day + 1

    # Projected month-end total
    est_total = round(used_month + burn_daily * days_left)

    conn.close()

    # Status classification
    if remaining is not None and remaining < THRESHOLD_SUSPENDED:
        status = "suspended"
    elif remaining is not None and remaining < THRESHOLD_EMERGENCY:
        status = "emergency"
    elif remaining is not None and remaining < THRESHOLD_SKIP:
        status = "skip"
    elif burn_daily > MAX_DAILY_WARN:
        status = "warn"
    else:
        status = "ok"

    return {
        "credits_remaining":  remaining,
        "credits_used_today": int(used_today),
        "credits_used_month": int(used_month),
        "burn_rate_daily":    burn_daily,
        "days_until_reset":   days_left,
        "est_month_total":    est_total,
        "status":             status,
        "budget":             MONTHLY_BUDGET,
        "target":             OPERATIONAL_TARGET,
        "sustainable_daily":  SUSTAINABLE_DAILY,
        "max_daily_warn":     MAX_DAILY_WARN,
    }


def check_credit_budget() -> str:
    """
    Quick pre-flight check before any scheduled fetch.

    Returns one of:
      "ok"         — all clear, proceed
      "warn"       — above sustainable rate but proceed (just a flag)
      "skip"       — remaining < 1500, skip non-essential (snapshots/best-lines)
      "emergency"  — remaining < 500, props only, no snapshots
      "suspended"  — remaining < 100, no fetches at all

    Reads from credits_log — free, no API call.
    """
    cs = get_credit_status()
    return cs["status"]


# ══════════════════════════════════════════════════════════════════════════════
# Targeted imminent-game fetcher (replaces universal 30-min snapshot)
# ══════════════════════════════════════════════════════════════════════════════

def _is_in_fetch_window() -> bool:
    """True if current local time is between 8 AM and 11 PM (inclusive)."""
    local_hour = datetime.now().hour
    return 8 <= local_hour <= 23


def _get_active_sports_today() -> list[str]:
    """
    Return sport_keys that have at least one unstarted fixture today in the
    main bets.db fixtures table.  Used to pre-filter the imminent-fetch loop
    so we only call TheOddsAPI for sports with actual games.

    Falls back to an empty list on any error (caller will use full key list).
    """
    try:
        _main_db = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        conn = sqlite3.connect(os.path.abspath(_main_db), timeout=10)
        conn.row_factory = sqlite3.Row
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        now_str   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        rows = conn.execute(
            """
            SELECT DISTINCT sport_key FROM fixtures
            WHERE DATE(commence_time) = ?
              AND commence_time > ?
            """,
            (today_str, now_str),
        ).fetchall()
        conn.close()
        return [r["sport_key"] for r in rows]
    except Exception as _exc:
        print(f"[CreatorTier] _get_active_sports_today error: {_exc}")
        return []


def fetch_imminent_games_odds(
    hours_ahead: int = 6,
    sport_keys: list[str] | None = None,
    bookmakers: list[str] | None = None,
) -> dict:
    """
    Fetch odds only for games starting within *hours_ahead* hours.
    Captures line snapshots for those games.

    Replaces the 30-min universal capture_line_snapshot().
    Runs every 45 min, 8 AM–11 PM local time.

    Credit cost: ~1-3 per sport per call, but only for sports with
    imminent games — on a typical 6-game MLB slate that's 1-2 credits
    vs 10+ for a full sport-by-sport sweep.

    Returns: {captured_at, imminent_sports, events_captured, rows_inserted,
              credits_budget_status, skipped_reason}
    """
    _ensure_tables()

    budget = check_credit_budget()
    if budget == "suspended":
        return {"skipped_reason": "credits_suspended", "credits_budget_status": budget,
                "events_captured": 0, "rows_inserted": 0}

    if not _is_in_fetch_window():
        return {"skipped_reason": "outside_fetch_window_8am_11pm",
                "credits_budget_status": budget, "events_captured": 0, "rows_inserted": 0}

    from odds_api import ALL_SPORT_KEYS, get_active_sport_keys, fetch_odds

    # Pre-filter: only iterate sports that have unstarted fixtures in our DB today.
    # This avoids paying 1 credit per sport for sports with zero games scheduled.
    # On a day with only MLB + NHL + NBA, saves ~9 credits per cycle × 15 cycles = 135/day.
    _today_sports = _get_active_sports_today()
    _requested    = sport_keys or ALL_SPORT_KEYS
    if _today_sports:
        _requested = [k for k in _requested if k in _today_sports]
        if not _requested:
            return {"skipped_reason": "no_active_sports_today",
                    "credits_budget_status": budget, "events_captured": 0, "rows_inserted": 0}
        print(f"[ImmFetch] Active sports today: {_today_sports} → fetching {_requested}")
    else:
        # Fallback: _get_active_sports_today returned nothing (empty DB or error) — fetch all
        print("[ImmFetch] No fixture pre-filter available; fetching all sport keys")

    keys      = get_active_sport_keys(_requested)
    books     = bookmakers or _SHOP_BOOKS
    cutoff    = datetime.utcnow() + timedelta(hours=hours_ahead)
    now_utc   = datetime.utcnow()

    captured_at     = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    imminent_sports: list[str] = []
    total_rows       = 0
    total_events     = 0

    conn = _hist_conn()

    for sk in keys:
        # Fetch all events for this sport — 1-3 credits
        events = fetch_odds(sk, markets="h2h,spreads,totals", odds_format="decimal")
        if not events:
            continue

        # Filter to games starting within the window
        imminent = []
        for event in events:
            ct = event.get("commence_time", "")
            try:
                game_dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue
            if now_utc <= game_dt <= cutoff:
                imminent.append(event)

        if not imminent:
            continue

        imminent_sports.append(sk)
        rows_to_insert: list[tuple] = []

        for event in imminent:
            total_events += 1
            event_id  = event.get("id", "")
            game_date = (event.get("commence_time") or "")[:10]
            home      = event.get("home_team", "")
            away      = event.get("away_team", "")

            for bk in event.get("bookmakers", []):
                bk_key = bk.get("key", "")
                if bk_key not in books:
                    continue
                for mkt in bk.get("markets", []):
                    for oc in mkt.get("outcomes", []):
                        rows_to_insert.append((
                            captured_at, sk, event_id, game_date,
                            home, away, bk_key, mkt.get("key", ""),
                            oc.get("name", ""),
                            oc.get("price"),
                            oc.get("point"),
                        ))

        conn.executemany("""
            INSERT INTO line_snapshots
              (captured_at, sport_key, event_id, game_date, home_team, away_team,
               bookmaker, market_key, outcome_name, price, point)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, rows_to_insert)
        conn.commit()
        total_rows += len(rows_to_insert)

    conn.close()
    result = {
        "captured_at":         captured_at,
        "hours_ahead":         hours_ahead,
        "imminent_sports":     imminent_sports,
        "events_captured":     total_events,
        "rows_inserted":       total_rows,
        "credits_budget_status": budget,
        "skipped_reason":      None,
    }
    print(f"[ImminentOdds] {result}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLV Backfill
# ══════════════════════════════════════════════════════════════════════════════

def backfill_clv(sport: str = "NHL", days_back: int = 180,
                 dry_run: bool = False) -> dict:
    """
    Fill missing close_spread / close_ml_home / close_ml_away in betting_lines
    for the given sport using TheOddsAPI historical odds endpoint.

    Each API call costs 20 credits and returns one snapshot of all games at
    that moment. We request one snapshot per game date (noon UTC on game day)
    which gives us the closing line.

    Returns: {filled, skipped, errors, credits_used}
    """
    _ensure_tables()
    sport_key = _CLV_SPORTS.get(sport)
    if not sport_key:
        return {"error": f"Unknown sport '{sport}'. Use NHL or MLB."}

    conn = _hist_conn()

    # Find game dates where close_ml_home is NULL (not yet filled)
    rows = conn.execute("""
        SELECT DISTINCT game_date
        FROM   betting_lines
        WHERE  sport = ?
          AND  close_ml_home IS NULL
          AND  game_date >= date('now', ? || ' days')
          AND  game_date <  date('now')      -- exclude future scheduled games
        ORDER  BY game_date
    """, (sport, f"-{days_back}")).fetchall()

    game_dates = [r["game_date"] for r in rows]
    print(f"[CLV-Backfill] {sport} — {len(game_dates)} dates to fill "
          f"(dry_run={dry_run})")

    filled = skipped = errors = credits_used = 0

    for game_date in game_dates:
        # Request a historical snapshot at 17:00 UTC on game day (noon ET — pregame,
        # before most MLB/NHL games start, so bookmakers still have active lines)
        snapshot_ts = f"{game_date}T17:00:00Z"

        if dry_run:
            print(f"  [DRY] Would fetch {sport_key} snapshot for {game_date}")
            skipped += 1
            continue

        data = _get(
            f"/historical/sports/{sport_key}/odds",
            {
                "regions":    "us",
                "markets":    "h2h,spreads",
                "oddsFormat": "american",
                "date":       snapshot_ts,
                "bookmakers": "fanduel,draftkings,betmgm,caesars,pinnacle",
            },
        )
        credits_used += 20  # historical odds cost 20 credits per call
        time.sleep(0.5)     # be kind to rate limits

        if not data or not isinstance(data, dict):
            print(f"  [CLV] No data for {game_date}")
            errors += 1
            continue

        events = data.get("data", [])
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            event_date = (event.get("commence_time") or "")[:10]

            if event_date != game_date:
                continue

            # Average the ml/spread across bookmakers for a consensus line
            ml_home_list: list[float] = []
            ml_away_list: list[float] = []
            spread_home_list: list[float] = []

            for bk in event.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt["key"] == "h2h":
                        for oc in mkt.get("outcomes", []):
                            if oc["name"] == home:
                                ml_home_list.append(float(oc["price"]))
                            elif oc["name"] == away:
                                ml_away_list.append(float(oc["price"]))
                    elif mkt["key"] == "spreads":
                        for oc in mkt.get("outcomes", []):
                            if oc["name"] == home:
                                try:
                                    spread_home_list.append(float(oc["point"]))
                                except (TypeError, ValueError):
                                    pass

            if not ml_home_list:
                continue

            avg_ml_home   = round(sum(ml_home_list) / len(ml_home_list), 1)
            avg_ml_away   = round(sum(ml_away_list) / len(ml_away_list), 1) if ml_away_list else None
            avg_spread    = round(sum(spread_home_list) / len(spread_home_list), 1) if spread_home_list else None
            covered_pl_val = None

            conn.execute("""
                UPDATE betting_lines
                SET    close_ml_home = ?,
                       close_ml_away = ?,
                       close_spread  = COALESCE(close_spread, ?),
                       source        = CASE WHEN source IS NULL THEN 'theoddsapi_clv'
                                            ELSE source END
                WHERE  sport      = ?
                  AND  game_date  = ?
                  AND  home_team  = ?
                  AND  close_ml_home IS NULL
            """, (avg_ml_home, avg_ml_away, avg_spread, sport, game_date, home))
            affected = conn.execute("SELECT changes()").fetchone()[0]
            if affected:
                filled += affected

        conn.commit()

    conn.close()
    result = {
        "sport":        sport,
        "dates_checked": len(game_dates),
        "filled":       filled,
        "skipped":      skipped,
        "errors":       errors,
        "credits_used": credits_used,
    }
    print(f"[CLV-Backfill] Done: {result}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. Player Props
# ══════════════════════════════════════════════════════════════════════════════

def fetch_player_props(sport_key: str, bookmakers: list[str] | None = None,
                       hours_ahead: int = 12) -> dict:
    """
    Fetch player prop odds for events starting within *hours_ahead* hours.
    Runs twice daily (11 AM + 4 PM local time) via scheduler.

    Budget: ~6 games × 3 market batches × 2 fetches/day ≈ 36 credits/day
    vs previous all-games × 3 fetches = ~135 credits/day.

    Returns: {sport_key, events_found, events_scoped, props_upserted, skipped}
    """
    _ensure_tables()

    budget = check_credit_budget()
    if budget == "suspended":
        return {"skipped_reason": "credits_suspended", "credits_budget_status": budget,
                "sport_key": sport_key, "events_found": 0, "props_upserted": 0}

    markets = _PROP_MARKETS.get(sport_key)
    if not markets:
        return {"error": f"No prop markets configured for {sport_key}"}

    books      = ",".join(bookmakers or _PROP_BOOKS)
    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    now_utc    = datetime.utcnow()
    cutoff     = now_utc + timedelta(hours=hours_ahead)

    # Emergency mode: only 1 market batch instead of all
    if budget == "emergency":
        markets = markets[:4]

    total_upserted = 0
    events_found   = 0
    events_scoped  = 0
    skipped        = 0

    conn = _hist_conn()

    # First get the event list to know which games are imminent
    event_list_data = _get(
        f"/sports/{sport_key}/odds",
        {"regions": "us", "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"},
    )
    time.sleep(0.2)

    # Build set of imminent event IDs
    imminent_ids: set[str] = set()
    if event_list_data and isinstance(event_list_data, list):
        for ev in event_list_data:
            ct = ev.get("commence_time", "")
            try:
                game_dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue
            if now_utc <= game_dt <= cutoff:
                imminent_ids.add(ev.get("id", ""))
        events_found = len(event_list_data)

    if not imminent_ids:
        conn.close()
        return {"sport_key": sport_key, "fetched_at": fetched_at,
                "events_found": events_found, "events_scoped": 0,
                "props_upserted": 0, "skipped_batches": 0,
                "note": f"No games starting within {hours_ahead}h"}

    # Fetch props in batches of 4 markets, filtered to imminent event IDs
    batch_size = 4
    for i in range(0, len(markets), batch_size):
        market_batch = ",".join(markets[i : i + batch_size])

        data = _get(
            f"/sports/{sport_key}/odds",
            {
                "regions":    "us",
                "markets":    market_batch,
                "oddsFormat": "decimal",
                "dateFormat":  "iso",
                "bookmakers": books,
            },
        )
        time.sleep(0.3)

        if not data or not isinstance(data, list):
            skipped += 1
            continue

        for event in data:
            event_id = event.get("id", "")
            if event_id not in imminent_ids:
                continue   # only process imminent games

            events_scoped += 1
            game_date = (event.get("commence_time") or "")[:10]
            home      = event.get("home_team", "")
            away      = event.get("away_team", "")

            for bk in event.get("bookmakers", []):
                bk_key = bk.get("key", "")
                for mkt in bk.get("markets", []):
                    mkt_key = mkt.get("key", "")
                    for oc in mkt.get("outcomes", []):
                        player = oc.get("description") or oc.get("name", "")
                        desc   = oc.get("name", "")
                        price  = oc.get("price")
                        point  = oc.get("point")
                        over_under = "over" if "over" in desc.lower() else (
                                     "under" if "under" in desc.lower() else desc.lower()
                        )
                        try:
                            conn.execute("""
                                INSERT OR REPLACE INTO player_props
                                  (fetched_at, sport_key, event_id, game_date,
                                   home_team, away_team, bookmaker, market_key,
                                   player_name, description, price, point, over_under)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """, (fetched_at, sport_key, event_id, game_date,
                                  home, away, bk_key, mkt_key,
                                  player, desc, price, point, over_under))
                            total_upserted += 1
                        except sqlite3.IntegrityError:
                            pass

        conn.commit()

    conn.close()
    result = {
        "sport_key":            sport_key,
        "fetched_at":           fetched_at,
        "hours_ahead":          hours_ahead,
        "events_found":         events_found,
        "events_scoped":        events_scoped,
        "props_upserted":       total_upserted,
        "skipped_batches":      skipped,
        "credits_budget_status": budget,
    }
    print(f"[PlayerProps] {result}")
    return result


def get_props_for_event(event_id: str, market_key: str | None = None) -> list[dict]:
    """
    Return the most recent props for an event (all markets or a specific one).
    """
    _ensure_tables()
    conn = _hist_conn()
    # Most recent fetch per (bookmaker, market_key, player_name, over_under)
    if market_key:
        rows = conn.execute("""
            SELECT p1.*
            FROM   player_props p1
            JOIN   (SELECT bookmaker, market_key, player_name, over_under,
                           MAX(fetched_at) AS max_fa
                    FROM   player_props
                    WHERE  event_id = ? AND market_key = ?
                    GROUP  BY bookmaker, market_key, player_name, over_under
                   ) p2
              ON   p1.bookmaker   = p2.bookmaker
             AND   p1.market_key  = p2.market_key
             AND   p1.player_name = p2.player_name
             AND   p1.over_under  = p2.over_under
             AND   p1.fetched_at  = p2.max_fa
            WHERE  p1.event_id   = ?
            ORDER  BY p1.player_name, p1.market_key, p1.over_under, p1.bookmaker
        """, (event_id, market_key, event_id)).fetchall()
    else:
        rows = conn.execute("""
            SELECT p1.*
            FROM   player_props p1
            JOIN   (SELECT bookmaker, market_key, player_name, over_under,
                           MAX(fetched_at) AS max_fa
                    FROM   player_props
                    WHERE  event_id = ?
                    GROUP  BY bookmaker, market_key, player_name, over_under
                   ) p2
              ON   p1.bookmaker   = p2.bookmaker
             AND   p1.market_key  = p2.market_key
             AND   p1.player_name = p2.player_name
             AND   p1.over_under  = p2.over_under
             AND   p1.fetched_at  = p2.max_fa
            WHERE  p1.event_id   = ?
            ORDER  BY p1.player_name, p1.market_key, p1.over_under, p1.bookmaker
        """, (event_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_best_props(event_id: str, market_key: str) -> list[dict]:
    """
    Return the best available price for each player/over_under combination
    across all bookmakers, plus the FanDuel price for comparison.
    """
    rows = get_props_for_event(event_id, market_key)
    grouped: dict[tuple, dict] = {}
    fd_prices: dict[tuple, float] = {}

    for r in rows:
        key = (r["player_name"], r["over_under"])
        price = r["price"] or 0.0
        if r["bookmaker"] == "fanduel":
            fd_prices[key] = price
        if key not in grouped or price > grouped[key]["best_price"]:
            grouped[key] = {
                "player_name":    r["player_name"],
                "market_key":     market_key,
                "over_under":     r["over_under"],
                "point":          r["point"],
                "best_price":     price,
                "best_book":      r["bookmaker"],
            }

    results = []
    for key, item in grouped.items():
        item["fanduel_price"] = fd_prices.get(key)
        fd = item["fanduel_price"]
        best = item["best_price"]
        if fd and best and fd > 0:
            item["fanduel_vs_best_pct"] = round((fd - best) / best * 100, 1)
        else:
            item["fanduel_vs_best_pct"] = None
        results.append(item)

    results.sort(key=lambda x: x["player_name"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Line Movement (capture snapshots)
# ══════════════════════════════════════════════════════════════════════════════

def capture_line_snapshot(sport_key: str | None = None,
                          bookmakers: list[str] | None = None) -> dict:
    """
    Fetch current spread/total/h2h for all active events and store
    one row per bookmaker × market × outcome in line_snapshots.

    NOTE: The scheduler no longer calls this on every 30-min cycle.
    Use fetch_imminent_games_odds() for scheduled snapshots.
    This function is kept for manual/API-triggered full snapshots.

    Returns: {captured_at, sport_keys, events, rows_inserted}
    """
    _ensure_tables()
    budget = check_credit_budget()
    if budget in ("suspended", "skip", "emergency"):
        return {"skipped_reason": f"credits_{budget}", "credits_budget_status": budget,
                "events": 0, "rows_inserted": 0}

    from odds_api import ALL_SPORT_KEYS, get_active_sport_keys, fetch_odds

    keys = get_active_sport_keys([sport_key] if sport_key else ALL_SPORT_KEYS)
    if not keys:
        return {"captured_at": datetime.utcnow().isoformat(),
                "sport_keys": [], "events": 0, "rows_inserted": 0}

    books = bookmakers or _SHOP_BOOKS
    captured_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    total_rows = 0
    total_events = 0

    conn = _hist_conn()

    for sk in keys:
        events = fetch_odds(sk, markets="h2h,spreads,totals", odds_format="decimal")
        if not events:
            continue

        rows_to_insert: list[tuple] = []
        for event in events:
            total_events += 1
            event_id  = event.get("id", "")
            game_date = (event.get("commence_time") or "")[:10]
            home      = event.get("home_team", "")
            away      = event.get("away_team", "")

            for bk in event.get("bookmakers", []):
                bk_key = bk.get("key", "")
                if bk_key not in books:
                    continue
                for mkt in bk.get("markets", []):
                    mkt_key = mkt.get("key", "")
                    for oc in mkt.get("outcomes", []):
                        rows_to_insert.append((
                            captured_at, sk, event_id, game_date,
                            home, away, bk_key, mkt_key,
                            oc.get("name", ""),
                            oc.get("price"),
                            oc.get("point"),
                        ))

        conn.executemany("""
            INSERT INTO line_snapshots
              (captured_at, sport_key, event_id, game_date, home_team, away_team,
               bookmaker, market_key, outcome_name, price, point)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, rows_to_insert)
        conn.commit()
        total_rows += len(rows_to_insert)

    conn.close()
    result = {
        "captured_at":  captured_at,
        "sport_keys":   keys,
        "events":       total_events,
        "rows_inserted": total_rows,
    }
    print(f"[LineSnapshot] {result}")
    return result


def get_line_movement(event_id: str, market_key: str = "spreads",
                      outcome_name: str | None = None) -> list[dict]:
    """
    Return all snapshots for an event/market ordered by time.
    Aggregates across bookmakers (returns average per timestamp).
    """
    _ensure_tables()
    conn = _hist_conn()

    if outcome_name:
        rows = conn.execute("""
            SELECT   captured_at,
                     outcome_name,
                     bookmaker,
                     price,
                     point
            FROM     line_snapshots
            WHERE    event_id     = ?
              AND    market_key   = ?
              AND    outcome_name = ?
            ORDER BY captured_at ASC
        """, (event_id, market_key, outcome_name)).fetchall()
    else:
        rows = conn.execute("""
            SELECT   captured_at,
                     outcome_name,
                     bookmaker,
                     price,
                     point
            FROM     line_snapshots
            WHERE    event_id   = ?
              AND    market_key = ?
            ORDER BY captured_at ASC
        """, (event_id, market_key)).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def detect_line_movement(event_id: str, market_key: str = "spreads",
                         threshold: float = 0.5) -> dict:
    """
    Analyze line movement for an event.
    Returns opening line, current line, total move, direction, and alert flag.
    threshold: minimum point-move to flag as significant (default 0.5).
    """
    snapshots = get_line_movement(event_id, market_key)
    if not snapshots:
        return {"event_id": event_id, "market_key": market_key,
                "data_points": 0, "alert": False}

    # Group by (outcome_name, bookmaker) and get first/last
    timeline: dict[tuple, list[dict]] = {}
    for s in snapshots:
        k = (s["outcome_name"], s["bookmaker"])
        timeline.setdefault(k, []).append(s)

    moves = []
    for (outcome, book), snaps in timeline.items():
        if len(snaps) < 2:
            continue
        opening = snaps[0]["point"] or snaps[0]["price"]
        closing = snaps[-1]["point"] or snaps[-1]["price"]
        if opening is None or closing is None:
            continue
        move = closing - opening
        moves.append({
            "outcome":     outcome,
            "bookmaker":   book,
            "opening":     opening,
            "current":     closing,
            "move":        round(move, 2),
            "first_seen":  snaps[0]["captured_at"],
            "last_seen":   snaps[-1]["captured_at"],
            "snapshots":   len(snaps),
        })

    if not moves:
        return {"event_id": event_id, "market_key": market_key,
                "data_points": len(snapshots), "alert": False}

    max_move = max(moves, key=lambda x: abs(x["move"]))
    return {
        "event_id":    event_id,
        "market_key":  market_key,
        "data_points": len(snapshots),
        "moves":       sorted(moves, key=lambda x: abs(x["move"]), reverse=True),
        "max_move":    max_move["move"],
        "alert":       abs(max_move["move"]) >= threshold,
        "direction":   "steam" if max_move["move"] < 0 else "reverse",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Best Line Shopping
# ══════════════════════════════════════════════════════════════════════════════

def get_best_lines(sport_key: str, market: str = "h2h") -> list[dict]:
    """
    Fetch current odds for a sport and return the best available line
    for each game outcome, along with FanDuel's line for comparison.

    Returns list of dicts, one per game × outcome, sorted by EV gap.
    """
    from odds_api import fetch_odds

    events = fetch_odds(sport_key, markets=market, odds_format="decimal")
    results = []

    for event in events:
        event_id  = event.get("id", "")
        home      = event.get("home_team", "")
        away      = event.get("away_team", "")
        commence  = event.get("commence_time", "")

        # Collect all prices per outcome
        prices_per_outcome: dict[str, list[tuple[str, float]]] = {}
        fd_prices: dict[str, float] = {}

        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key", "")
            for mkt in bk.get("markets", []):
                if mkt.get("key") != market:
                    continue
                for oc in mkt.get("outcomes", []):
                    name  = oc.get("name", "")
                    price = float(oc.get("price", 0))
                    prices_per_outcome.setdefault(name, []).append((bk_key, price))
                    if bk_key == "fanduel":
                        fd_prices[name] = price

        for outcome, book_prices in prices_per_outcome.items():
            if not book_prices:
                continue
            best_book, best_price = max(book_prices, key=lambda x: x[1])
            fd_price = fd_prices.get(outcome)
            if fd_price and best_price > 0:
                gap_pct = round((best_price - fd_price) / fd_price * 100, 1)
            else:
                gap_pct = None

            results.append({
                "event_id":      event_id,
                "sport_key":     sport_key,
                "home_team":     home,
                "away_team":     away,
                "commence_time": commence,
                "outcome":       outcome,
                "market":        market,
                "best_price":    best_price,
                "best_book":     best_book,
                "fanduel_price": fd_price,
                "gap_pct":       gap_pct,
                "alert":         gap_pct is not None and gap_pct >= 5.0,
                "all_books": sorted(
                    [{"book": b, "price": p} for b, p in book_prices],
                    key=lambda x: x["price"],
                    reverse=True,
                ),
            })

    # Sort by gap descending (worst FD value first)
    results.sort(
        key=lambda x: x["gap_pct"] if x["gap_pct"] is not None else -999,
        reverse=True,
    )
    return results


def fetch_best_lines_daily(sport_keys: list[str] | None = None) -> dict:
    """
    Fetch best lines for all (or specified) active sports once per day.
    Scheduled at 10 AM local time. Results power the pick-card display.

    Credit cost: 1-3 credits per sport per call.
    Typical daily use: ~6 active sports × 2 credits = ~12 credits/day.

    Returns: {sport_keys_fetched, total_lines, status}
    """
    budget = check_credit_budget()
    if budget == "suspended":
        return {"skipped_reason": "credits_suspended", "sport_keys_fetched": [],
                "total_lines": 0, "credits_budget_status": budget}

    from odds_api import ALL_SPORT_KEYS, get_active_sport_keys

    keys       = get_active_sport_keys(sport_keys or ALL_SPORT_KEYS)
    all_lines: dict[str, list] = {}
    total      = 0

    for sk in keys:
        lines = get_best_lines(sk, market="h2h")
        if lines:
            all_lines[sk] = lines
            total += len(lines)

    result = {
        "fetched_at":          datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sport_keys_fetched":  list(all_lines.keys()),
        "total_lines":         total,
        "credits_budget_status": budget,
    }
    print(f"[BestLinesDaily] {result}")
    return result


def get_line_shopping_summary(sport_keys: list[str] | None = None) -> dict:
    """
    Quick summary: for each sport, count how many outcomes have FanDuel
    ≥5% worse than market best.

    Returns {sport_key: {total_outcomes, alerts, worst_gap_pct, worst_game}}
    """
    from odds_api import ALL_SPORT_KEYS, get_active_sport_keys

    keys = get_active_sport_keys(sport_keys or ALL_SPORT_KEYS)
    summary = {}

    for sk in keys:
        lines = get_best_lines(sk, market="h2h")
        total   = len(lines)
        alerts  = [l for l in lines if l["alert"]]
        worst   = max(lines, key=lambda x: x["gap_pct"] or 0) if lines else None
        summary[sk] = {
            "total_outcomes": total,
            "alerts":         len(alerts),
            "worst_gap_pct":  worst["gap_pct"] if worst else None,
            "worst_game":     (f"{worst['home_team']} vs {worst['away_team']}"
                               if worst else None),
        }

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 5. Alternate Lines
# ══════════════════════════════════════════════════════════════════════════════

def fetch_alt_lines(sport_key: str, event_id: str,
                    bookmakers: list[str] | None = None) -> dict:
    """
    Fetch alternate spreads / totals / puck-line variants for a single event.

    Stores results in alt_lines table (historical.db).
    Returns: {event_id, sport_key, rows_inserted, markets_fetched, error?}
    """
    _ensure_tables()

    markets = _ALT_MARKETS.get(sport_key)
    if not markets:
        return {"error": f"No alt markets configured for {sport_key}",
                "event_id": event_id, "sport_key": sport_key}

    budget = check_credit_budget()
    if budget == "suspended":
        return {"skipped_reason": "credits_suspended", "event_id": event_id,
                "rows_inserted": 0}

    books      = ",".join(bookmakers or _SHOP_BOOKS)
    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    data = _get(
        f"/sports/{sport_key}/events/{event_id}/odds",
        {
            "regions":    "us",
            "markets":    ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "bookmakers": books,
        },
    )

    if not data or not isinstance(data, dict):
        return {"error": "No data returned", "event_id": event_id,
                "sport_key": sport_key, "rows_inserted": 0}

    home_team  = data.get("home_team", "")
    away_team  = data.get("away_team", "")
    game_date  = (data.get("commence_time") or "")[:10]
    rows_inserted = 0

    conn = _hist_conn()
    for bk in data.get("bookmakers", []):
        bk_key = bk.get("key", "")
        for mkt in bk.get("markets", []):
            mkt_key = mkt.get("key", "")
            for oc in mkt.get("outcomes", []):
                team      = oc.get("name", "")
                line      = oc.get("point")
                odds      = oc.get("price")
                over_under = None
                if "over" in team.lower():
                    over_under, team = "over", team
                elif "under" in team.lower():
                    over_under, team = "under", team
                if not odds:
                    continue
                if not _is_valid_spread_line(line, sport_key, mkt_key):
                    continue
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO alt_lines
                          (fetched_at, sport_key, event_id, game_date,
                           home_team, away_team, bookmaker, market_key,
                           market_type,
                           team, line, over_under, odds, is_main_market)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                    """, (fetched_at, sport_key, event_id, game_date,
                          home_team, away_team, bk_key, mkt_key,
                          mkt_key,
                          team, line, over_under, odds))
                    rows_inserted += 1
                except sqlite3.IntegrityError:
                    pass

    conn.commit()
    conn.close()

    return {
        "event_id":       event_id,
        "sport_key":      sport_key,
        "home_team":      home_team,
        "away_team":      away_team,
        "game_date":      game_date,
        "markets_fetched": markets,
        "rows_inserted":  rows_inserted,
    }


def fetch_alt_lines_batch(
    sport_keys: list[str] | None = None,
    bookmakers: list[str] | None = None,
    hours_ahead: int = 24,
) -> dict:
    """
    Batch-fetch alternate_spreads + alternate_totals for all active sports.

    Note: TheOddsAPI's sport-level /odds endpoint does NOT support alternate
    market keys.  Alternate lines require the per-event endpoint:
      GET /sports/{sport_key}/events/{event_id}/odds?markets=alternate_spreads,...
    This function uses a two-step approach:
      1. GET /sports/{sport_key}/events  — 0 credits; returns upcoming event IDs
      2. Per-event /events/{id}/odds     — 1 credit per event
    Only events starting within hours_ahead hours are fetched to control cost.

    Scheduled twice daily: 7:45 AM CT (alongside fixture refresh) and 2:30 PM CT
    (30 min before the PM pick run).
    Typical cost: 0 (event lists) + N events × 1 credit/event.
    For today's typical slate (MLB ~10, NBA ~4, soccer ~5 total):
      ~20–40 credits/run, ≤ 80 credits/day.

    Credit guard: skips if credits_remaining < THRESHOLD_EMERGENCY (200).

    Stores results in the alt_lines table (historical.db).
    Returns: {fetched_at, sport_keys, events_found, rows_inserted, by_sport,
              credits_budget_status, credits_used, skipped_reason?}
    """
    _ensure_tables()

    credit_status = get_credit_status()
    remaining     = credit_status.get("credits_remaining")
    budget        = credit_status.get("status", "ok")

    if remaining is not None and remaining < THRESHOLD_EMERGENCY:
        print(f"[Credits] Emergency mode — using cached only (credits_remaining={remaining} < {THRESHOLD_EMERGENCY})")
        return {
            "skipped_reason":        f"credits_low_{remaining}",
            "credits_budget_status": "emergency",
            "rows_inserted":         0,
            "events_found":          0,
        }

    keys       = sport_keys or _ALT_BATCH_SPORT_KEYS
    books      = bookmakers or _SHOP_BOOKS
    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    now_utc    = datetime.utcnow()
    # Use 4 AM CT betting-day cutoff (same boundary as parlay_builder.get_available_legs).
    # hours_ahead=24 would pull 5/5 and 5/6 games when run on 5/4 morning — wrong.
    # We only want alt lines for games on today's CT betting slate (until next 4 AM CT).
    from zoneinfo import ZoneInfo as _ZI_alt
    _ct_tz_alt  = _ZI_alt("America/Chicago")
    _now_ct_alt = datetime.now(_ct_tz_alt)
    _4am_alt    = _now_ct_alt.replace(hour=4, minute=0, second=0, microsecond=0)
    if _now_ct_alt >= _4am_alt:
        _slate_end_ct_alt = _4am_alt + timedelta(days=1)
    else:
        _slate_end_ct_alt = _4am_alt
    # Convert to naive UTC (game_dt comparison below is also naive UTC)
    cutoff = _slate_end_ct_alt.astimezone(timezone.utc).replace(tzinfo=None)

    conn          = _hist_conn()
    total_rows    = 0
    total_events  = 0
    credits_start = remaining or 0
    by_sport: dict[str, dict] = {}

    for sk in keys:
        markets = _ALT_MARKETS.get(sk, ["alternate_spreads", "alternate_totals"])

        # Step 1: free event-list fetch to get IDs for upcoming games
        params = {"apiKey": _ODDS_API_KEY, "dateFormat": "iso"}
        try:
            import requests as _req
            r = _req.get(f"{_BASE_URL}/sports/{sk}/events", params=params, timeout=15)
            if r.status_code != 200:
                by_sport[sk] = {"events": 0, "rows": 0, "error": f"events_http_{r.status_code}"}
                continue
            events_list = r.json()
        except Exception as _e:
            by_sport[sk] = {"events": 0, "rows": 0, "error": str(_e)}
            continue

        # Filter to games within the fetch window
        upcoming = []
        for ev in events_list:
            ct_str = ev.get("commence_time", "")
            try:
                game_dt = datetime.fromisoformat(ct_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue
            if now_utc <= game_dt <= cutoff:
                upcoming.append(ev)

        if not upcoming:
            by_sport[sk] = {"events": 0, "rows": 0}
            continue

        # Step 2: per-event alt lines fetch (1 credit each)
        sport_rows   = 0
        sport_events = 0

        for ev in upcoming:
            event_id  = ev.get("id", "")
            game_date = (ev.get("commence_time") or "")[:10]
            home      = ev.get("home_team", "")
            away      = ev.get("away_team", "")

            data = _get(
                f"/sports/{sk}/events/{event_id}/odds",
                {
                    "regions":    "us",
                    "markets":    ",".join(markets),
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                    "bookmakers": ",".join(books),
                },
            )
            if not data or not isinstance(data, dict):
                continue

            sport_events += 1
            rows_to_insert: list[tuple] = []

            for bk in data.get("bookmakers", []):
                bk_key = bk.get("key", "")
                if bk_key not in books:
                    continue
                for mkt in bk.get("markets", []):
                    mkt_key = mkt.get("key", "")
                    for oc in mkt.get("outcomes", []):
                        team       = oc.get("name", "")
                        line       = oc.get("point")
                        odds       = oc.get("price")
                        over_under = None
                        tl         = (team or "").lower()
                        if "over" in tl:
                            over_under = "over"
                        elif "under" in tl:
                            over_under = "under"
                        if not odds:
                            continue
                        if not _is_valid_spread_line(line, sk, mkt_key):
                            continue
                        rows_to_insert.append((
                            fetched_at, sk, event_id, game_date,
                            home, away, bk_key, mkt_key,
                            mkt_key,   # market_type = market_key
                            team, line, over_under, odds, 0,
                        ))

            try:
                conn.executemany("""
                    INSERT OR REPLACE INTO alt_lines
                      (fetched_at, sport_key, event_id, game_date,
                       home_team, away_team, bookmaker, market_key,
                       market_type,
                       team, line, over_under, odds, is_main_market)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, rows_to_insert)
                conn.commit()
                sport_rows += len(rows_to_insert)
            except Exception as _e:
                conn.rollback()
                print(f"[AltLinesBatch] {sk}/{event_id}: insert error — {_e}")

        total_rows   += sport_rows
        total_events += sport_events
        by_sport[sk]  = {"events": sport_events, "rows": sport_rows}
        print(f"[AltLinesBatch] {sk}: {sport_events} events, {sport_rows} rows")

    conn.close()

    # Estimate credits used this run
    final_remaining = get_credit_status().get("credits_remaining") or 0
    credits_used    = max(0, (credits_start or 0) - final_remaining)

    result = {
        "fetched_at":            fetched_at,
        "sport_keys":            keys,
        "events_found":          total_events,
        "rows_inserted":         total_rows,
        "by_sport":              by_sport,
        "credits_budget_status": budget,
        "credits_used_estimate": credits_used,
        "skipped_reason":        None,
    }
    print(f"[AltLinesBatch] TOTAL: {total_events} events, {total_rows} rows "
          f"across {len([k for k in by_sport if by_sport[k].get('events', 0) > 0])} sports "
          f"| credits_used≈{credits_used}")
    return result


def _is_valid_spread_line(point, sport_key: str, market_key: str) -> bool:
    """
    Return False for lines that should not be stored in the alt_lines table.

    1. Spread rules: MLB/NBA/NHL must use .5-increment lines only.
    2. Total rules:  Soccer alternate_totals must use 0.5-increment lines only
       (2.5, 3.5 …).  Quarter-goal Asian lines (2.25, 2.75, 4.25 …) confuse
       parlay building and are excluded at ingest time.
    """
    _SOCCER_SPORT_KEYS = frozenset([
        "soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga",
        "soccer_france_ligue_one", "soccer_italy_serie_a",
        "soccer_uefa_champs_league", "soccer_uefa_europa_league",
        "soccer_usa_mls",
    ])

    if point is None:
        return True
    try:
        val = float(point)
    except (TypeError, ValueError):
        return True

    # ── Totals: soccer integer-line filter ───────────────────────────────────
    # FanDuel offers only x.5 soccer totals (2.5, 3.5 …).
    # Block integers (2, 3) and quarter-goal lines (2.25, 2.75).
    if "total" in (market_key or "") and sport_key in _SOCCER_SPORT_KEYS:
        return abs(val % 1.0 - 0.5) < 0.01   # must be exactly x.5

    # ── Spreads: soccer integer-line filter ──────────────────────────────────
    # FanDuel offers only x.5 Asian handicap lines (-0.5, +1.5 …).
    # Block integers (-1, -2) and quarter-goal lines (+1.25, +2.75).
    if "spread" in (market_key or "") and sport_key in _SOCCER_SPORT_KEYS:
        return abs(val % 1.0 - 0.5) < 0.01   # must be exactly x.5

    # ── Spreads: .5-only enforcement for MLB/NBA/NHL ──────────────────────────
    if market_key not in ("alternate_spreads", "spreads"):
        return True
    half = abs(val) % 1
    _MLB = {"baseball_mlb", "baseball_mlb_preseason"}
    _NBA = {"basketball_nba", "basketball_nba_preseason"}
    _NHL = {"icehockey_nhl", "icehockey_nhl_preseason"}
    if sport_key in _MLB or sport_key in _NBA or sport_key in _NHL:
        return abs(half - 0.5) < 1e-9   # must be exactly .5
    # NFL etc. — allow anything
    return True


def score_alt_line(alt_row: dict, model_prob: float,
                   main_line: float | None = None) -> dict:
    """
    Score one alt line row.

    Parameters
    ----------
    alt_row    : dict from alt_lines table (team, line, odds, market_key, …)
    model_prob : 0-1 model win probability for the home/favorite team
    main_line  : main market spread for this team (used to compute shift);
                 None → no adjustment, use model_prob directly

    Returns
    -------
    alt_row augmented with: implied_prob, our_prob, edge_pp, ev, value_label
    """
    odds = float(alt_row.get("odds") or 2.0)
    line = alt_row.get("line")

    implied_prob = round((1.0 / odds) * 100.0, 1)

    # Shift model probability based on line difference from main spread
    our_prob = model_prob * 100.0
    if main_line is not None and line is not None:
        try:
            # Positive shift = alt line is easier (smaller number for spread fav)
            shift = float(main_line) - float(line)
            our_prob += shift * _SPREAD_ADJ_PP
        except (TypeError, ValueError):
            pass

    our_prob = max(5.0, min(95.0, our_prob))
    edge_pp  = round(our_prob - implied_prob, 2)
    ev       = round((our_prob / 100.0) * (odds - 1) * 10.0
                     - (1.0 - our_prob / 100.0) * 10.0, 2)   # $10 stake

    if our_prob >= _ANCHOR_PROB and edge_pp >= _ANCHOR_EDGE:
        label = "ANCHOR"
    elif edge_pp >= _VALUE_EDGE:
        label = "VALUE"
    else:
        label = "SKIP"

    return {
        **alt_row,
        "implied_prob": implied_prob,
        "our_prob":     round(our_prob, 1),
        "edge_pp":      edge_pp,
        "ev":           ev,
        "value_label":  label,
    }


def get_scored_alt_lines(event_id: str, model_prob: float = 0.52,
                         main_line: float | None = None,
                         min_label: str = "VALUE") -> list[dict]:
    """
    Read the most recent alt line fetch for event_id, score every line,
    deduplicate to best-odds per (team, market, line) combo, return sorted
    by EV descending.

    min_label: "ANCHOR" returns only anchors; "VALUE" returns VALUE+ANCHOR;
               "ALL" returns everything including SKIP lines.
    """
    _ensure_tables()
    conn = _hist_conn()

    # Most recent fetch timestamp for this event
    row = conn.execute(
        "SELECT MAX(fetched_at) FROM alt_lines WHERE event_id=?",
        (event_id,)
    ).fetchone()
    latest = row[0] if row and row[0] else None

    if not latest:
        conn.close()
        return []

    rows = conn.execute(
        "SELECT * FROM alt_lines WHERE event_id=? AND fetched_at=?",
        (event_id, latest)
    ).fetchall()
    conn.close()

    # Score each row
    scored: list[dict] = []
    seen: dict[tuple, dict] = {}  # best-odds per (team, market, line)
    for r in rows:
        d = dict(r)
        if not _is_valid_spread_line(d.get("line"), d.get("sport_key", ""), d.get("market_key", "")):
            continue
        s = score_alt_line(d, model_prob, main_line)
        key = (s.get("team",""), s.get("market_key",""), s.get("line"))
        # keep the bookmaker with the best (highest) odds for this combo
        if key not in seen or (s.get("odds") or 0) > (seen[key].get("odds") or 0):
            seen[key] = s

    scored = list(seen.values())

    label_rank = {"ANCHOR": 0, "VALUE": 1, "SKIP": 2}
    if min_label == "ANCHOR":
        scored = [s for s in scored if s["value_label"] == "ANCHOR"]
    elif min_label == "VALUE":
        scored = [s for s in scored if s["value_label"] != "SKIP"]

    scored.sort(key=lambda s: -s["ev"])
    return scored


def get_todays_alt_lines(sport_key: str = "",
                          market_key: str = "",
                          filter_label: str = "ALL") -> list[dict]:
    """
    Return all alt lines stored today for all events (or a specific sport_key /
    market_key), scored and sorted by EV descending.

    No odds-range filtering is applied here — that belongs in evaluate_alt_lines()
    at pick-generation time.  _is_valid_spread_line is also skipped so the API
    returns every row that is actually stored in the DB.
    """
    _ensure_tables()
    conn = _hist_conn()
    today   = datetime.utcnow().strftime("%Y-%m-%d")
    cutoff  = f"{today}T00:00:00Z"

    conditions = ["fetched_at >= ?"]
    params: list = [cutoff]
    if sport_key:
        conditions.append("sport_key = ?")
        params.append(sport_key)
    if market_key:
        conditions.append("market_key = ?")
        params.append(market_key)

    sql  = "SELECT * FROM alt_lines WHERE " + " AND ".join(conditions)
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    # Best odds per (event_id, team, market_key, line) across bookmakers
    seen: dict[tuple, dict] = {}
    for r in rows:
        d = dict(r)
        s = score_alt_line(d, 0.52)  # no model prob available globally; scores for EV sort
        key = (s.get("event_id"), s.get("team", ""), s.get("market_key", ""), s.get("line"))
        if key not in seen or (s.get("odds") or 0) > (seen[key].get("odds") or 0):
            seen[key] = s

    result = list(seen.values())
    if filter_label == "ANCHOR":
        result = [r for r in result if r["value_label"] == "ANCHOR"]
    elif filter_label == "VALUE":
        result = [r for r in result if r["value_label"] != "SKIP"]

    result.sort(key=lambda r: -r["ev"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json

    parser = argparse.ArgumentParser(description="BetIQ Creator Tier tools")
    sub = parser.add_subparsers(dest="cmd")

    p_clv = sub.add_parser("clv", help="CLV backfill for NHL/MLB")
    p_clv.add_argument("--sport",    default="NHL", choices=list(_CLV_SPORTS))
    p_clv.add_argument("--days",     type=int, default=180)
    p_clv.add_argument("--dry-run",  action="store_true")

    p_props = sub.add_parser("props", help="Fetch player props")
    p_props.add_argument("--sport", default="baseball_mlb",
                         choices=list(_PROP_MARKETS))

    p_snap = sub.add_parser("snapshot", help="Capture line snapshot")
    p_snap.add_argument("--sport", default=None, help="Specific sport key (default: all active)")

    p_best = sub.add_parser("best-lines", help="Best line shopping")
    p_best.add_argument("--sport", default="basketball_nba")
    p_best.add_argument("--market", default="h2h")

    p_move = sub.add_parser("movement", help="Line movement for an event")
    p_move.add_argument("event_id")
    p_move.add_argument("--market", default="spreads")

    args = parser.parse_args()

    if args.cmd == "clv":
        result = backfill_clv(args.sport, args.days, args.dry_run)
        print(_json.dumps(result, indent=2))
    elif args.cmd == "props":
        result = fetch_player_props(args.sport)
        print(_json.dumps(result, indent=2))
    elif args.cmd == "snapshot":
        result = capture_line_snapshot(args.sport)
        print(_json.dumps(result, indent=2))
    elif args.cmd == "best-lines":
        lines = get_best_lines(args.sport, args.market)
        print(_json.dumps(lines[:20], indent=2))
    elif args.cmd == "movement":
        result = detect_line_movement(args.event_id, args.market)
        print(_json.dumps(result, indent=2))
    else:
        parser.print_help()
