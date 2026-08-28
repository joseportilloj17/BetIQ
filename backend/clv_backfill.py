"""
clv_backfill.py — Retroactive CLV (Closing Line Value) scoring.

Pulls closing-line odds from TheOddsAPI historical endpoint and computes
per-leg CLV for every settled mock bet in cohort=clean.

CLV% = (entry_decimal / closing_decimal) - 1
  Positive → we got a better price than the close (beat the market).

Usage:
    python clv_backfill.py                    # dry-run: show budget only
    python clv_backfill.py --run              # pull closing odds + compute CLV
    python clv_backfill.py --run --budget 5000  # cap at 5000 credits
    python clv_backfill.py --report           # aggregate CLV by arm (no API calls)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ── Config ───────────────────────────────────────────────────────────────────

API_KEY = "REDACTED_ODDS_API_KEY"
BASE_URL = "https://api.the-odds-api.com"
BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
CLV_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "clv_api_log.jsonl")
CREDITS_PER_GAME = 10
DEFAULT_BUDGET = 5000
RATE_LIMIT_SLEEP = 0.5  # seconds between API calls


# ── Helpers ──────────────────────────────────────────────────────────────────

def _american_to_decimal(american: float) -> float:
    if american >= 100:
        return 1 + american / 100
    elif american <= -100:
        return 1 + 100 / abs(american)
    return 2.0


def _decimal_to_implied(decimal_odds: float) -> float:
    """Implied probability from decimal odds (with vig)."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def _devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vig from a two-way market (multiplicative method)."""
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


def _parse_leg_description(desc: str) -> tuple[str, float]:
    """
    Parse 'Team Name +X.5' or 'Team Name +X.5 (Alt Spread)'
    → (team_name, spread_value)
    """
    m = re.match(r'(.+?)\s+([+-]\d+\.?\d*)\s*(?:\(Alt Spread\))?$', desc.strip())
    if m:
        return m.group(1).strip(), float(m.group(2))
    raise ValueError(f"Cannot parse leg description: {desc}")


def _log_api_call(event_id: str, status: int, credits_used: int,
                  remaining: Optional[int], detail: str):
    """Append one line to the API log."""
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "event_id": event_id,
        "status": status,
        "credits_used": credits_used,
        "credits_remaining": remaining,
        "detail": detail,
    }
    with open(CLV_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Step 1: Extract bet universe ─────────────────────────────────────────────

def extract_bet_universe(conn_bets: sqlite3.Connection) -> list[dict]:
    """
    Get all settled spread legs from cohort=clean mock bets.
    Returns list of dicts with leg info + bet-level arm identification.
    """
    cur = conn_bets.cursor()
    cur.execute("""
        SELECT
            l.id            AS leg_id,
            l.mock_bet_id,
            l.description,
            l.fixture_id,
            l.market_type,
            l.leg_result,
            m.model_used    AS arm,
            m.generated_at,
            m.odds          AS parlay_odds,
            m.status,
            m.actual_profit
        FROM mock_bet_legs l
        JOIN mock_bets m ON l.mock_bet_id = m.id
        WHERE m.cohort = 'clean'
          AND m.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND l.fixture_id IS NOT NULL
          AND l.market_type = 'alternate_spreads'
        ORDER BY m.generated_at, l.mock_bet_id, l.leg_index
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    # Parse team + spread from description
    for row in rows:
        try:
            team, spread = _parse_leg_description(row["description"])
            row["team"] = team
            row["spread"] = spread
        except ValueError:
            row["team"] = None
            row["spread"] = None

    return rows


def get_unique_games(legs: list[dict],
                     conn_bets: sqlite3.Connection) -> dict[str, dict]:
    """
    De-duplicate to unique games. Returns {fixture_id: {commence_time, ...}}.
    """
    fixture_ids = list({l["fixture_id"] for l in legs if l["fixture_id"]})
    cur = conn_bets.cursor()
    placeholders = ",".join(["?"] * len(fixture_ids))
    cur.execute(f"""
        SELECT id, commence_time, home_team, away_team
        FROM fixtures
        WHERE id IN ({placeholders})
    """, fixture_ids)

    games = {}
    for fid, ct, home, away in cur.fetchall():
        games[fid] = {
            "fixture_id": fid,
            "commence_time": ct,
            "home_team": home,
            "away_team": away,
        }
    return games


# ── Step 2: Get entry odds from local alt_lines ─────────────────────────────

def get_entry_odds(legs: list[dict],
                   conn_hist: sqlite3.Connection) -> dict:
    """
    For each leg, find the alt_lines snapshot closest to (but <= ) generated_at.
    Returns {leg_id: entry_decimal_odds} using median across books.
    """
    cur = conn_hist.cursor()
    entry_odds = {}
    miss_count = 0

    for leg in legs:
        if not leg["team"] or leg["spread"] is None or not leg.get("generated_at"):
            miss_count += 1
            continue

        gen_at = leg["generated_at"].replace(" ", "T")

        # Find the closest snapshot <= generation time
        cur.execute("""
            SELECT DISTINCT fetched_at
            FROM alt_lines
            WHERE event_id = ?
              AND market_key = 'alternate_spreads'
              AND team = ?
              AND line = ?
              AND fetched_at <= ?
            ORDER BY fetched_at DESC
            LIMIT 1
        """, (leg["fixture_id"], leg["team"], leg["spread"], gen_at))
        snap_row = cur.fetchone()

        if not snap_row:
            # Try without the time constraint (earliest available)
            cur.execute("""
                SELECT DISTINCT fetched_at
                FROM alt_lines
                WHERE event_id = ?
                  AND market_key = 'alternate_spreads'
                  AND team = ?
                  AND line = ?
                ORDER BY fetched_at ASC
                LIMIT 1
            """, (leg["fixture_id"], leg["team"], leg["spread"]))
            snap_row = cur.fetchone()

        if not snap_row:
            miss_count += 1
            continue

        snap_time = snap_row[0]

        # Get all book prices at this snapshot
        cur.execute("""
            SELECT odds
            FROM alt_lines
            WHERE event_id = ?
              AND market_key = 'alternate_spreads'
              AND team = ?
              AND line = ?
              AND fetched_at = ?
        """, (leg["fixture_id"], leg["team"], leg["spread"], snap_time))
        prices = [r[0] for r in cur.fetchall() if r[0] and r[0] > 1.0]

        if prices:
            entry_odds[leg["leg_id"]] = {
                "decimal": round(statistics.median(prices), 4),
                "n_books": len(prices),
                "snapshot": snap_time,
            }
        else:
            miss_count += 1

    print(f"  Entry odds: matched {len(entry_odds)}/{len(legs)} legs "
          f"({miss_count} misses)")
    return entry_odds


# ── Step 3: Pull closing odds from API ───────────────────────────────────────

def _create_clv_table(conn_bets: sqlite3.Connection):
    """Create the bet_clv table if it doesn't exist."""
    conn_bets.execute("""
        CREATE TABLE IF NOT EXISTS bet_clv (
            leg_id              INTEGER PRIMARY KEY,
            mock_bet_id         TEXT,
            fixture_id          TEXT,
            arm                 TEXT,
            team                TEXT,
            spread              REAL,
            leg_result          TEXT,

            -- Entry (our bet)
            entry_decimal       REAL,
            entry_n_books       INTEGER,
            entry_snapshot      TEXT,

            -- Closing (market at game time)
            close_decimal       REAL,
            close_n_books       INTEGER,
            close_snapshot      TEXT,

            -- Opposite side closing (for devig)
            close_opp_decimal   REAL,
            close_opp_n_books   INTEGER,

            -- CLV metrics
            clv_raw_pct         REAL,   -- (entry/close - 1) × 100
            beat_close           INTEGER,  -- 1 if entry > close (got better price)

            -- No-vig CLV
            close_fair_prob      REAL,   -- devigged closing prob for our side
            entry_implied_prob   REAL,   -- 1/entry_decimal
            clv_novig_pp         REAL,   -- close_fair_prob - entry_implied_prob (pp)

            -- Per-book closing odds (JSON)
            close_per_book      TEXT,

            computed_at         TEXT
        )
    """)
    conn_bets.execute("""
        CREATE INDEX IF NOT EXISTS idx_clv_arm ON bet_clv(arm)
    """)
    conn_bets.execute("""
        CREATE INDEX IF NOT EXISTS idx_clv_fixture ON bet_clv(fixture_id)
    """)
    conn_bets.commit()


def _games_already_pulled(conn_bets: sqlite3.Connection) -> set[str]:
    """Return fixture_ids we've already pulled closing odds for (for resume)."""
    try:
        cur = conn_bets.execute(
            "SELECT DISTINCT fixture_id FROM bet_clv WHERE close_decimal IS NOT NULL"
        )
        return {r[0] for r in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()


def pull_closing_odds(games: dict[str, dict],
                      budget: int,
                      conn_bets: sqlite3.Connection) -> dict[str, dict]:
    """
    Pull closing alt-spread odds from TheOddsAPI historical endpoint.
    Returns {fixture_id: {rung: {team: {decimal, per_book}}}}
    """
    already_done = _games_already_pulled(conn_bets)
    to_pull = {fid: g for fid, g in games.items() if fid not in already_done}

    if already_done:
        print(f"  Resume: {len(already_done)} games already done, "
              f"{len(to_pull)} remaining")

    cost_estimate = len(to_pull) * CREDITS_PER_GAME
    if cost_estimate > budget:
        print(f"  WARNING: estimated cost {cost_estimate} exceeds budget {budget}")
        print(f"  Will pull first {budget // CREDITS_PER_GAME} games")
        # Trim to budget
        to_pull = dict(list(to_pull.items())[:budget // CREDITS_PER_GAME])

    print(f"  Pulling closing odds for {len(to_pull)} games "
          f"(~{len(to_pull) * CREDITS_PER_GAME} credits)")

    closing_data = {}
    credits_spent = 0
    errors = 0

    for i, (fid, game) in enumerate(to_pull.items()):
        ct = game["commence_time"]
        if not ct:
            _log_api_call(fid, 0, 0, None, "no commence_time")
            errors += 1
            continue

        # Format commence_time as ISO 8601 for the API
        # The API wants: YYYY-MM-DDTHH:MM:SSZ
        ct_str = str(ct).replace(" ", "T")
        if not ct_str.endswith("Z"):
            ct_str = ct_str.split(".")[0] + "Z"

        params = {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "alternate_spreads",
            "oddsFormat": "decimal",
            "date": ct_str,
        }
        url = (f"{BASE_URL}/v4/historical/sports/baseball_mlb/"
               f"events/{fid}/odds?{urlencode(params)}")

        try:
            req = Request(url)
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                status = resp.status
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")

            # Parse the response
            game_odds = _parse_closing_response(data)
            closing_data[fid] = game_odds
            credits_spent += CREDITS_PER_GAME

            _log_api_call(fid, status, CREDITS_PER_GAME,
                          int(remaining) if remaining else None,
                          f"ok, {len(game_odds)} rungs")

            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(to_pull)}] {credits_spent} credits spent"
                      f" (remaining: {remaining})")

        except HTTPError as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            _log_api_call(fid, e.code, 0, None, f"HTTP {e.code}: {body[:200]}")
            errors += 1
            if e.code == 429:
                print(f"    Rate limited at game {i+1}, sleeping 60s...")
                time.sleep(60)
            elif e.code == 422:
                # Unprocessable — usually date out of range
                pass
            else:
                print(f"    HTTP error {e.code} for {fid}: {body[:100]}")
        except (URLError, TimeoutError) as e:
            _log_api_call(fid, 0, 0, None, f"Network error: {str(e)[:200]}")
            errors += 1
            time.sleep(2)

        # Rate limit
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"  Done: {len(closing_data)} games pulled, {credits_spent} credits, "
          f"{errors} errors")
    return closing_data


def _parse_closing_response(data: dict) -> dict:
    """
    Parse API response into: {(team, rung): {decimal: median, per_book: [...]}}
    """
    result = {}
    timestamp = data.get("timestamp", "")

    bookmakers = data.get("bookmakers", [])
    if not bookmakers:
        # Try nested data structure
        inner = data.get("data", {})
        bookmakers = inner.get("bookmakers", [])
        if not timestamp:
            timestamp = inner.get("timestamp", "")

    for bk in bookmakers:
        book_key = bk.get("key", "unknown")
        for market in bk.get("markets", []):
            if market.get("key") != "alternate_spreads":
                continue
            for outcome in market.get("outcomes", []):
                team = outcome.get("name", "")
                point = outcome.get("point")
                price = outcome.get("price")
                if team and point is not None and price:
                    key = (team, float(point))
                    if key not in result:
                        result[key] = {"prices": [], "books": [], "timestamp": timestamp}
                    result[key]["prices"].append(price)
                    result[key]["books"].append(book_key)

    # Compute medians
    for key in result:
        prices = result[key]["prices"]
        result[key]["median"] = round(statistics.median(prices), 4)
        result[key]["n_books"] = len(prices)
        result[key]["per_book"] = dict(zip(result[key]["books"], prices))

    return result


# ── Step 3 (local): Get closing odds from local alt_lines ────────────────────

def get_closing_odds_local(games: dict[str, dict],
                           conn_hist: sqlite3.Connection) -> dict:
    """
    Use the latest alt_lines snapshot before commence_time as closing proxy.
    Returns same format as pull_closing_odds: {fixture_id: {(team, rung): {...}}}
    """
    cur = conn_hist.cursor()
    closing_data = {}
    matched = 0
    missed = 0

    for fid, game in games.items():
        ct = game.get("commence_time")
        if not ct:
            missed += 1
            continue

        ct_str = str(ct).replace(" ", "T")
        if not ct_str.endswith("Z"):
            ct_str = ct_str.split(".")[0] + "Z"

        # Find the latest snapshot <= commence_time
        cur.execute("""
            SELECT DISTINCT fetched_at
            FROM alt_lines
            WHERE event_id = ?
              AND market_key = 'alternate_spreads'
              AND fetched_at <= ?
            ORDER BY fetched_at DESC
            LIMIT 1
        """, (fid, ct_str))
        snap_row = cur.fetchone()

        if not snap_row:
            # Fallback: use the latest available snapshot regardless of timing
            cur.execute("""
                SELECT DISTINCT fetched_at
                FROM alt_lines
                WHERE event_id = ?
                  AND market_key = 'alternate_spreads'
                ORDER BY fetched_at DESC
                LIMIT 1
            """, (fid,))
            snap_row = cur.fetchone()

        if not snap_row:
            missed += 1
            continue

        snap_time = snap_row[0]

        # Get all odds at this snapshot
        cur.execute("""
            SELECT team, line, odds, bookmaker
            FROM alt_lines
            WHERE event_id = ?
              AND market_key = 'alternate_spreads'
              AND fetched_at = ?
        """, (fid, snap_time))

        game_odds = {}
        for team, line, odds, book in cur.fetchall():
            if odds and odds > 1.0:
                key = (team, float(line))
                if key not in game_odds:
                    game_odds[key] = {
                        "prices": [], "books": [], "timestamp": snap_time
                    }
                game_odds[key]["prices"].append(odds)
                game_odds[key]["books"].append(book)

        # Compute medians
        for key in game_odds:
            prices = game_odds[key]["prices"]
            game_odds[key]["median"] = round(statistics.median(prices), 4)
            game_odds[key]["n_books"] = len(prices)
            game_odds[key]["per_book"] = dict(
                zip(game_odds[key]["books"], prices))

        if game_odds:
            closing_data[fid] = game_odds
            matched += 1
        else:
            missed += 1

    print(f"  Local closing odds: {matched}/{len(games)} games matched "
          f"({missed} missed)")
    return closing_data


# ── Step 4: Compute CLV per leg ──────────────────────────────────────────────

def compute_and_store_clv(legs: list[dict],
                          entry_odds: dict,
                          closing_data: dict,
                          games: dict,
                          conn_bets: sqlite3.Connection):
    """Compute CLV for each leg and store in bet_clv table."""
    _create_clv_table(conn_bets)

    already_done = _games_already_pulled(conn_bets)
    scored = 0
    skipped_no_entry = 0
    skipped_no_close = 0

    for leg in legs:
        lid = leg["leg_id"]

        # Check if already computed
        cur = conn_bets.execute("SELECT 1 FROM bet_clv WHERE leg_id = ?", (lid,))
        if cur.fetchone():
            continue

        fid = leg["fixture_id"]
        team = leg["team"]
        spread = leg["spread"]
        if not team or spread is None:
            skipped_no_entry += 1
            continue

        # Entry odds
        entry = entry_odds.get(lid)
        if not entry:
            skipped_no_entry += 1
            continue

        entry_dec = entry["decimal"]

        # Closing odds
        game_close = closing_data.get(fid, {})
        close_key = (team, spread)
        close_info = game_close.get(close_key)

        if not close_info:
            skipped_no_close += 1
            continue

        close_dec = close_info["median"]
        close_ts = close_info.get("timestamp", "")

        # Find opposite side for devig
        game = games.get(fid, {})
        opp_team = (game.get("away_team") if team == game.get("home_team")
                    else game.get("home_team"))
        opp_spread = -spread
        opp_key = (opp_team, opp_spread)
        opp_info = game_close.get(opp_key)

        opp_dec = opp_info["median"] if opp_info else None
        opp_n = opp_info["n_books"] if opp_info else None

        # Raw CLV: positive = we got a better price
        clv_raw = (entry_dec / close_dec - 1) * 100
        beat_close = 1 if entry_dec > close_dec else 0

        # No-vig CLV
        close_fair_prob = None
        entry_implied = 1.0 / entry_dec
        clv_novig = None

        if opp_dec and close_dec > 1.0 and opp_dec > 1.0:
            our_implied = 1.0 / close_dec
            opp_implied = 1.0 / opp_dec
            fair_our, _ = _devig_two_way(our_implied, opp_implied)
            close_fair_prob = round(fair_our, 6)
            clv_novig = round((close_fair_prob - entry_implied) * 100, 4)

        conn_bets.execute("""
            INSERT OR REPLACE INTO bet_clv (
                leg_id, mock_bet_id, fixture_id, arm, team, spread, leg_result,
                entry_decimal, entry_n_books, entry_snapshot,
                close_decimal, close_n_books, close_snapshot,
                close_opp_decimal, close_opp_n_books,
                clv_raw_pct, beat_close,
                close_fair_prob, entry_implied_prob, clv_novig_pp,
                close_per_book, computed_at
            ) VALUES (?,?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?, ?,?, ?,?,?, ?,?)
        """, (
            lid, leg["mock_bet_id"], fid, leg["arm"], team, spread, leg["leg_result"],
            round(entry_dec, 4), entry["n_books"], entry["snapshot"],
            round(close_dec, 4), close_info["n_books"], close_ts,
            round(opp_dec, 4) if opp_dec else None, opp_n,
            round(clv_raw, 4), beat_close,
            close_fair_prob, round(entry_implied, 6),
            clv_novig, json.dumps(close_info.get("per_book", {})),
            datetime.utcnow().isoformat(),
        ))
        scored += 1

    conn_bets.commit()
    print(f"  CLV scored: {scored} legs")
    print(f"  Skipped (no entry odds): {skipped_no_entry}")
    print(f"  Skipped (no close odds): {skipped_no_close}")


# ── Step 5: Aggregate & report ───────────────────────────────────────────────

def report_clv_by_arm(conn_bets: sqlite3.Connection):
    """Print CLV% aggregation by arm — the payoff."""
    cur = conn_bets.cursor()

    # Check we have data
    try:
        cur.execute("SELECT count(*) FROM bet_clv WHERE clv_raw_pct IS NOT NULL")
        total = cur.fetchone()[0]
    except sqlite3.OperationalError:
        print("No bet_clv table found. Run with --run first.")
        return
    if total == 0:
        print("No CLV data computed yet.")
        return

    print(f"\n{'='*80}")
    print(f"CLV REPORT — {total} legs scored")
    print(f"{'='*80}")

    # Arm-level CLV
    cur.execute("""
        SELECT
            arm,
            count(*)                              AS n_legs,
            round(avg(clv_raw_pct), 3)            AS mean_clv,
            round(avg(CASE WHEN beat_close = 1 THEN 100.0 ELSE 0.0 END), 1)
                                                  AS pct_beat,
            round(avg(clv_novig_pp), 3)           AS mean_novig,
            round(avg(CASE WHEN leg_result = 'WIN' THEN 100.0 ELSE 0.0 END), 1)
                                                  AS leg_wr
        FROM bet_clv
        WHERE clv_raw_pct IS NOT NULL
        GROUP BY arm
        ORDER BY n_legs DESC
    """)
    rows = cur.fetchall()

    print(f"\n{'Arm':45s} {'N':>6s} {'CLV%':>8s} {'Beat%':>7s} "
          f"{'NoVig':>8s} {'LegWR':>7s}")
    print("-" * 85)
    for arm, n, clv, beat, novig, wr in rows:
        arm_str = arm if arm else "(none)"
        novig_str = f"{novig:>+7.3f}%" if novig is not None else "    N/A"
        print(f"{arm_str:45s} {n:>6d} {clv:>+7.3f}% {beat:>6.1f}% "
              f"{novig_str} {wr:>6.1f}%")

    # By rung
    print(f"\n{'─'*60}")
    print("CLV by RUNG:")
    cur.execute("""
        SELECT
            spread,
            count(*)                              AS n,
            round(avg(clv_raw_pct), 3)            AS mean_clv,
            round(avg(CASE WHEN beat_close = 1 THEN 100.0 ELSE 0.0 END), 1)
                                                  AS pct_beat
        FROM bet_clv
        WHERE clv_raw_pct IS NOT NULL
        GROUP BY spread
        ORDER BY spread
    """)
    for sp, n, clv, beat in cur.fetchall():
        print(f"  {sp:>+5.1f}  N={n:>5d}  CLV={clv:>+7.3f}%  Beat={beat:>5.1f}%")

    # PnL cross-check: arm-level PnL from mock_bets
    print(f"\n{'─'*60}")
    print("CROSS-CHECK: CLV vs PnL by arm")
    print(f"{'Arm':45s} {'CLV%':>8s} {'PnL/leg':>9s} {'Verdict':>12s}")
    print("-" * 78)

    cur.execute("""
        SELECT
            c.arm,
            round(avg(c.clv_raw_pct), 3)  AS mean_clv,
            count(c.leg_id)               AS n_clv_legs
        FROM bet_clv c
        WHERE c.clv_raw_pct IS NOT NULL
        GROUP BY c.arm
        ORDER BY n_clv_legs DESC
    """)
    clv_by_arm = {r[0]: r[1] for r in cur.fetchall()}

    # Get PnL per parlay-leg for each arm
    cur.execute("""
        SELECT
            m.model_used AS arm,
            round(avg(m.actual_profit / m.legs), 3) AS pnl_per_leg
        FROM mock_bets m
        WHERE m.cohort = 'clean'
          AND m.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND m.model_used IS NOT NULL
          AND m.legs > 0
        GROUP BY m.model_used
    """)
    pnl_by_arm = {r[0]: r[1] for r in cur.fetchall()}

    for arm in sorted(clv_by_arm, key=lambda a: clv_by_arm.get(a, 0), reverse=True):
        clv = clv_by_arm.get(arm, 0)
        pnl = pnl_by_arm.get(arm)
        pnl_str = f"${pnl:>+7.3f}" if pnl is not None else "     N/A"

        if clv > 0 and pnl is not None and pnl > 0:
            verdict = "REAL EDGE"
        elif clv > 0 and pnl is not None and pnl <= 0:
            verdict = "CLV+ PnL-"
        elif clv <= 0 and pnl is not None and pnl > 0:
            verdict = "VARIANCE"
        elif clv <= 0:
            verdict = "no edge"
        else:
            verdict = "?"

        print(f"{str(arm):45s} {clv:>+7.3f}% {pnl_str} {verdict:>12s}")

    # Edge_gate deep dive
    print(f"\n{'─'*60}")
    print("EDGE_GATE deep dive (CLV by rung):")
    cur.execute("""
        SELECT
            spread,
            count(*) AS n,
            round(avg(clv_raw_pct), 3) AS mean_clv,
            round(avg(CASE WHEN beat_close = 1 THEN 100.0 ELSE 0.0 END), 1) AS pct_beat,
            round(avg(CASE WHEN leg_result = 'WIN' THEN 100.0 ELSE 0.0 END), 1) AS wr
        FROM bet_clv
        WHERE arm LIKE '%edge_gate%'
          AND clv_raw_pct IS NOT NULL
        GROUP BY spread
        ORDER BY spread
    """)
    eg_rows = cur.fetchall()
    if eg_rows:
        for sp, n, clv, beat, wr in eg_rows:
            print(f"  {sp:>+5.1f}  N={n:>4d}  CLV={clv:>+7.3f}%  Beat={beat:>5.1f}%  WR={wr:.1f}%")
    else:
        print("  (no edge_gate legs in CLV data)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retroactive CLV scoring")
    parser.add_argument("--run", action="store_true",
                        help="Pull closing odds and compute CLV")
    parser.add_argument("--report", action="store_true",
                        help="Show CLV report (no API calls)")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help=f"Max credits to spend (default {DEFAULT_BUDGET})")
    parser.add_argument("--local", action="store_true",
                        help="Use local alt_lines for closing odds instead of API")
    args = parser.parse_args()

    conn_bets = sqlite3.connect(BETS_DB)
    conn_hist = sqlite3.connect(HIST_DB)

    # Step 1: Extract bet universe
    print("Step 1: Extracting bet universe...")
    legs = extract_bet_universe(conn_bets)
    print(f"  {len(legs)} spread legs from cohort=clean settled bets")

    games = get_unique_games(legs, conn_bets)
    print(f"  {len(games)} unique games")

    credit_estimate = len(games) * CREDITS_PER_GAME
    print(f"\n  Budget estimate: {len(games)} games × {CREDITS_PER_GAME} = "
          f"{credit_estimate} credits")
    print(f"  Budget cap: {args.budget} credits")

    if args.report:
        report_clv_by_arm(conn_bets)
        conn_bets.close()
        conn_hist.close()
        return

    if not args.run:
        print("\n  Dry run — pass --run to execute.")
        conn_bets.close()
        conn_hist.close()
        return

    # Step 2: Get entry odds from local alt_lines
    print("\nStep 2: Getting entry odds from local alt_lines...")
    entry_odds = get_entry_odds(legs, conn_hist)

    # Step 3: Get closing odds
    if args.local:
        print("\nStep 3: Getting closing odds from LOCAL alt_lines "
              "(latest snapshot before commence_time)...")
        closing_data = get_closing_odds_local(games, conn_hist)
    else:
        print("\nStep 3: Pulling closing odds from TheOddsAPI...")
        closing_data = pull_closing_odds(games, args.budget, conn_bets)

    # Step 4: Compute CLV
    print("\nStep 4: Computing CLV per leg...")
    compute_and_store_clv(legs, entry_odds, closing_data, games, conn_bets)

    # Step 5: Report
    report_clv_by_arm(conn_bets)

    conn_bets.close()
    conn_hist.close()
    print(f"\nAPI log: {CLV_LOG}")


if __name__ == "__main__":
    main()
