"""
props_clv_backfill.py — Historical prop closing-line backfill + de-vigged CLV.

Pulls closing prop odds from OddsAPI historical endpoint, matches to our
entry-time props, computes de-vigged CLV per leg.

Usage:
    python props_clv_backfill.py --test 5          # dry-run 5 events, show costs
    python props_clv_backfill.py --run --budget 10000  # run with credit cap
    python props_clv_backfill.py --report          # report on stored data (no API)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ODDS_API_KEY", "")
if not API_KEY:
    try:
        from config import ODDS_API_KEY as _cfg_key
        API_KEY = _cfg_key
    except Exception:
        pass
if not API_KEY:
    raise RuntimeError("ODDS_API_KEY not set — export it or add to backend/.env")
BASE_URL = "https://api.the-odds-api.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HIST_DB = os.path.join(DATA_DIR, "historical.db")
BETS_DB = os.path.join(DATA_DIR, "bets.db")
RATE_LIMIT = 0.5  # seconds between API calls

PROP_MARKETS = "pitcher_strikeouts,batter_hits,batter_total_bases,batter_rbis"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Multiplicative de-vig for a two-way market."""
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


def _get_credits_remaining(db_path: str = HIST_DB) -> int:
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT credits_remaining FROM credits_log ORDER BY logged_at DESC LIMIT 1"
    ).fetchone()
    con.close()
    return row[0] if row else 0


# ── Step 1: Get events and commence times ────────────────────────────────────

def _get_events_needing_close(hist_db: str = HIST_DB) -> list[dict]:
    """
    Get all MLB events that have entry-time props but no closing props stored.
    Returns list of {event_id, game_date, n_entry_props}.
    """
    con = sqlite3.connect(hist_db)

    # Events with entry props
    rows = con.execute("""
        SELECT event_id, game_date,
               COUNT(*) as n_props,
               COUNT(DISTINCT market_key) as n_markets,
               COUNT(DISTINCT player_name) as n_players,
               MAX(fetched_at) as last_fetch
        FROM player_props
        WHERE sport_key = 'baseball_mlb'
          AND game_date < date('now')
        GROUP BY event_id
        ORDER BY game_date DESC
    """).fetchall()

    # Check which events already have closing props stored
    existing = set()
    try:
        existing_rows = con.execute(
            "SELECT DISTINCT event_id FROM props_closing_odds"
        ).fetchall()
        existing = {r[0] for r in existing_rows}
    except sqlite3.OperationalError:
        pass  # table doesn't exist yet

    con.close()

    events = []
    for r in rows:
        if r[0] not in existing:
            events.append({
                "event_id": r[0],
                "game_date": r[1],
                "n_props": r[2],
                "n_markets": r[3],
                "n_players": r[4],
                "last_fetch": r[5],
            })

    return events


def _get_commence_times(game_dates: list[str]) -> dict[str, str]:
    """
    Call historical events list for each unique game_date to get commence_times.
    Returns {event_id: commence_time}.
    """
    unique_dates = sorted(set(game_dates))
    event_ct = {}

    print(f"  Fetching commence times for {len(unique_dates)} game dates...")
    for i, gd in enumerate(unique_dates):
        params = {
            "apiKey": API_KEY,
            "date": f"{gd}T23:59:59Z",
        }
        url = f"{BASE_URL}/v4/historical/sports/baseball_mlb/events?{urlencode(params)}"

        try:
            req = Request(url)
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                remaining = resp.headers.get("x-requests-remaining")

            events_data = data.get("data", [])
            for ev in events_data:
                eid = ev.get("id", "")
                ct = ev.get("commence_time", "")
                if eid and ct:
                    event_ct[eid] = ct

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(unique_dates)}] {len(event_ct)} events mapped"
                      f" (credits: {remaining})")

        except HTTPError as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            print(f"    WARN: {gd} HTTP {e.code}: {body[:100]}")
        except Exception as e:
            print(f"    WARN: {gd} {e}")

        time.sleep(RATE_LIMIT)

    print(f"  Mapped {len(event_ct)} events to commence times")
    return event_ct


# ── Step 2: Fetch closing props ─────────────────────────────────────────────

def _ensure_closing_table(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS props_closing_odds (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT NOT NULL,
            game_date       TEXT,
            fetched_at      TEXT,
            api_timestamp   TEXT,
            bookmaker       TEXT NOT NULL,
            market_key      TEXT NOT NULL,
            player_name     TEXT NOT NULL,
            over_under      TEXT,
            point           REAL,
            price           REAL,
            UNIQUE(event_id, bookmaker, market_key, player_name, over_under, point)
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_props_close_event
            ON props_closing_odds(event_id, market_key)
    """)
    con.commit()


def _fetch_closing_props(events: list[dict], commence_times: dict[str, str],
                         budget: int, hist_db: str = HIST_DB) -> dict:
    """
    Fetch closing prop odds for each event from historical endpoint.
    Stores in props_closing_odds table.
    Returns stats dict.
    """
    con = sqlite3.connect(hist_db)
    _ensure_closing_table(con)

    credits_spent = 0
    events_fetched = 0
    events_skipped_no_ct = 0
    events_404 = 0
    total_outcomes = 0
    errors = 0

    for i, ev in enumerate(events):
        eid = ev["event_id"]
        ct = commence_times.get(eid)

        if not ct:
            events_skipped_no_ct += 1
            continue

        if credits_spent + 50 > budget:
            print(f"\n  BUDGET LIMIT: spent {credits_spent}, limit {budget}, stopping")
            break

        params = {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": PROP_MARKETS,
            "oddsFormat": "decimal",
            "date": ct,
        }
        url = (f"{BASE_URL}/v4/historical/sports/baseball_mlb/"
               f"events/{eid}/odds?{urlencode(params)}")

        try:
            req = Request(url)
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                remaining = resp.headers.get("x-requests-remaining")

            api_ts = data.get("timestamp", "")
            odds_data = data.get("data", data)
            bookmakers = odds_data.get("bookmakers", [])

            n_outcomes = 0
            now = datetime.utcnow().isoformat()

            for bk in bookmakers:
                bk_key = bk.get("key", "")
                for mkt in bk.get("markets", []):
                    mkt_key = mkt.get("key", "")
                    for out in mkt.get("outcomes", []):
                        player = out.get("description", "")
                        side = out.get("name", "")  # Over/Under
                        point = out.get("point")
                        price = out.get("price")

                        con.execute("""
                            INSERT OR REPLACE INTO props_closing_odds
                            (event_id, game_date, fetched_at, api_timestamp,
                             bookmaker, market_key, player_name,
                             over_under, point, price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (eid, ev["game_date"], now, api_ts,
                              bk_key, mkt_key, player,
                              side, point, price))
                        n_outcomes += 1

            con.commit()
            events_fetched += 1
            total_outcomes += n_outcomes
            credits_spent += 40  # approximate

            if (events_fetched) % 25 == 0:
                print(f"    [{events_fetched}/{len(events)}] "
                      f"~{credits_spent} credits spent, "
                      f"{total_outcomes} outcomes stored "
                      f"(API remaining: {remaining})")

        except HTTPError as e:
            if e.code == 404:
                events_404 += 1
            else:
                errors += 1
                body = e.read().decode() if hasattr(e, 'read') else str(e)
                print(f"    ERROR {e.code} for {eid}: {body[:100]}")
            if e.code == 429:
                print("    Rate limited, sleeping 60s...")
                time.sleep(60)
        except Exception as e:
            errors += 1
            print(f"    ERROR for {eid}: {e}")

        time.sleep(RATE_LIMIT)

    con.close()

    return {
        "events_fetched": events_fetched,
        "events_skipped_no_ct": events_skipped_no_ct,
        "events_404": events_404,
        "errors": errors,
        "total_outcomes": total_outcomes,
        "credits_spent_approx": credits_spent,
    }


# ── Step 3: Compute de-vigged CLV ──────────────────────────────────────────

def _compute_prop_clv(hist_db: str = HIST_DB) -> dict:
    """
    Match entry props to closing props, compute de-vigged CLV.
    Returns summary stats.
    """
    con = sqlite3.connect(hist_db)

    # Get all closing props with both sides for de-vigging
    # Group by event/bookmaker/market/player/point — need Over AND Under
    close_pairs = con.execute("""
        SELECT c1.event_id, c1.bookmaker, c1.market_key,
               c1.player_name, c1.point,
               c1.price as over_price, c2.price as under_price,
               c1.game_date
        FROM props_closing_odds c1
        JOIN props_closing_odds c2
          ON c1.event_id = c2.event_id
         AND c1.bookmaker = c2.bookmaker
         AND c1.market_key = c2.market_key
         AND c1.player_name = c2.player_name
         AND c1.point = c2.point
        WHERE c1.over_under = 'Over'
          AND c2.over_under = 'Under'
          AND c1.price > 1.0
          AND c2.price > 1.0
    """).fetchall()

    print(f"\n  Closing pairs (both sides): {len(close_pairs):,}")

    # Build lookup: (event, market, player, point) → best close data
    # Use median across bookmakers for robustness
    close_by_leg = defaultdict(list)
    for row in close_pairs:
        eid, book, mkt, player, point, over_p, under_p = row[:7]
        key = (eid, mkt, player, point)
        over_impl = 1.0 / over_p
        under_impl = 1.0 / under_p
        fair_over, fair_under = _devig(over_impl, under_impl)
        overround = over_impl + under_impl
        close_by_leg[key].append({
            "bookmaker": book,
            "over_price": over_p,
            "under_price": under_p,
            "over_implied": over_impl,
            "under_implied": under_impl,
            "fair_over": fair_over,
            "fair_under": fair_under,
            "overround": overround,
        })

    print(f"  Unique close legs: {len(close_by_leg):,}")

    # Get entry props — latest fetch per event
    entry_rows = con.execute("""
        SELECT p.event_id, p.market_key, p.player_name, p.point,
               p.over_under, p.price, p.bookmaker, p.fetched_at
        FROM player_props p
        INNER JOIN (
            SELECT event_id, market_key, player_name, point, over_under,
                   MAX(fetched_at) as last_fetch
            FROM player_props
            WHERE sport_key = 'baseball_mlb'
            GROUP BY event_id, market_key, player_name, point, over_under
        ) latest
          ON p.event_id = latest.event_id
         AND p.market_key = latest.market_key
         AND p.player_name = latest.player_name
         AND p.point = latest.point
         AND p.over_under = latest.over_under
         AND p.fetched_at = latest.last_fetch
        WHERE p.sport_key = 'baseball_mlb'
          AND p.price > 1.0
    """).fetchall()

    print(f"  Entry prop rows: {len(entry_rows):,}")

    # Match and compute CLV
    results = []
    matched = 0
    unmatched = 0

    # Map market keys: entry uses different names than close sometimes
    mkt_map = {
        "pitcher_strikeouts": "pitcher_strikeouts",
        "batter_hits": "batter_hits",
        "batter_total_bases": "batter_total_bases",
        "batter_rbis": "batter_rbis",
        "pitcher_strikeouts_alternate": "pitcher_strikeouts",
        "batter_hits_alternate": "batter_hits",
        "batter_total_bases_alternate": "batter_total_bases",
    }

    for row in entry_rows:
        eid, mkt, player, point, side, entry_price, book, fetched = row
        close_mkt = mkt_map.get(mkt, mkt)
        key = (eid, close_mkt, player, point)

        if key not in close_by_leg:
            unmatched += 1
            continue

        closes = close_by_leg[key]
        # Median fair probability across bookmakers
        if side == "over" or side == "Over":
            fair_probs = [c["fair_over"] for c in closes]
        else:
            fair_probs = [c["fair_under"] for c in closes]

        if not fair_probs:
            unmatched += 1
            continue

        median_fair = sorted(fair_probs)[len(fair_probs) // 2]
        entry_implied = 1.0 / entry_price  # includes vig

        # Raw CLV: how much better is our entry price vs close?
        # Positive = we got a better (higher) price than close
        close_prices = [c["over_price"] if side in ("over", "Over")
                        else c["under_price"] for c in closes]
        median_close_price = sorted(close_prices)[len(close_prices) // 2]
        clv_raw_pct = (entry_price / median_close_price - 1) * 100

        # De-vigged CLV: entry implied vs fair close prob
        # Positive = market moved toward us (we had the right side)
        clv_devigged_pp = (median_fair - entry_implied) * 100

        # Entry de-vigged too (need both sides of entry)
        # For now, compare raw entry implied vs fair close
        # This is CONSERVATIVE — entry vig inflates entry_implied,
        # making clv_devigged more negative than reality

        avg_overround = sum(c["overround"] for c in closes) / len(closes)

        results.append({
            "event_id": eid,
            "market_key": mkt,
            "player_name": player,
            "point": point,
            "side": side,
            "entry_price": entry_price,
            "entry_implied": entry_implied,
            "entry_book": book,
            "close_price_median": median_close_price,
            "close_fair_prob": median_fair,
            "close_overround": avg_overround,
            "close_n_books": len(closes),
            "clv_raw_pct": clv_raw_pct,
            "clv_devigged_pp": clv_devigged_pp,
        })
        matched += 1

    con.close()

    print(f"  Matched: {matched:,}, Unmatched: {unmatched:,}")
    return results


# ── Step 4: Report ──────────────────────────────────────────────────────────

def _report(results: list[dict]):
    """Print comprehensive CLV report."""
    import numpy as np
    from scipy import stats as sp_stats

    if not results:
        print("\n  NO RESULTS — nothing to report")
        return

    raw_clv = np.array([r["clv_raw_pct"] for r in results])
    devig_clv = np.array([r["clv_devigged_pp"] for r in results])
    overrounds = np.array([r["close_overround"] for r in results])
    n = len(results)
    n_events = len(set(r["event_id"] for r in results))

    print("\n" + "=" * 70)
    print("PROPS CLV REPORT — DE-VIGGED")
    print("=" * 70)

    print(f"\n  Legs matched: {n:,}")
    print(f"  Unique events: {n_events}")
    print(f"  Markets: {sorted(set(r['market_key'] for r in results))}")

    # ── Overall CLV ──
    print(f"\n  ── OVERALL ──")
    print(f"  Raw CLV%:      mean={np.mean(raw_clv):+.3f}%, "
          f"median={np.median(raw_clv):+.3f}%")
    print(f"  De-vigged CLV: mean={np.mean(devig_clv):+.3f}pp, "
          f"median={np.median(devig_clv):+.3f}pp")
    print(f"  Close overround: mean={np.mean(overrounds):.4f} "
          f"(vig={((np.mean(overrounds)-1)*100):.1f}%)")
    print(f"  Beat close (raw): {sum(1 for c in raw_clv if c > 0)}/{n} "
          f"= {sum(1 for c in raw_clv if c > 0)/n*100:.1f}%")

    # ── CI ──
    for label, vals in [("Raw CLV%", raw_clv), ("De-vigged CLV pp", devig_clv)]:
        se = np.std(vals, ddof=1) / np.sqrt(n)
        t_crit = sp_stats.t.ppf(0.975, df=n-1)
        ci_lo = np.mean(vals) - t_crit * se
        ci_hi = np.mean(vals) + t_crit * se
        t_stat = np.mean(vals) / se if se > 0 else 0
        p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n-1)) if se > 0 else 1
        print(f"\n  {label}:")
        print(f"    Mean: {np.mean(vals):+.3f}, Std: {np.std(vals, ddof=1):.3f}")
        print(f"    95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
        print(f"    Includes zero: {'YES' if ci_lo <= 0 <= ci_hi else 'NO'}")
        print(f"    t={t_stat:.2f}, p={p_val:.4f}")

    # ── By market ──
    print(f"\n  ── BY MARKET ──")
    by_mkt = defaultdict(list)
    for r in results:
        by_mkt[r["market_key"]].append(r)

    for mkt in sorted(by_mkt.keys()):
        legs = by_mkt[mkt]
        raw = np.array([r["clv_raw_pct"] for r in legs])
        dev = np.array([r["clv_devigged_pp"] for r in legs])
        print(f"  {mkt}: N={len(legs)}, raw={np.mean(raw):+.2f}%, "
              f"devig={np.mean(dev):+.2f}pp, "
              f"beat_close={sum(1 for c in raw if c > 0)/len(raw)*100:.0f}%")

    # ── Split-half stability ──
    print(f"\n  ── SPLIT-HALF STABILITY ──")
    # Sort by event date, split in half
    sorted_results = sorted(results, key=lambda r: r["event_id"])
    half = len(sorted_results) // 2
    first_half = sorted_results[:half]
    second_half = sorted_results[half:]

    for label, subset in [("First half", first_half), ("Second half", second_half)]:
        raw_s = np.array([r["clv_raw_pct"] for r in subset])
        dev_s = np.array([r["clv_devigged_pp"] for r in subset])
        n_ev = len(set(r["event_id"] for r in subset))
        print(f"  {label}: N={len(subset)} legs, {n_ev} events, "
              f"raw={np.mean(raw_s):+.2f}%, devig={np.mean(dev_s):+.2f}pp")

    # ── De-vig impact ──
    print(f"\n  ── VIG IMPACT ──")
    print(f"  Raw CLV mean:     {np.mean(raw_clv):+.3f}%")
    print(f"  De-vigged mean:   {np.mean(devig_clv):+.3f}pp")
    print(f"  Vig deflation:    {np.mean(devig_clv) - np.mean(raw_clv):+.3f}pp")
    print(f"  Close vig (avg):  {((np.mean(overrounds)-1)*100):.1f}%")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Props CLV backfill")
    parser.add_argument("--test", type=int, default=0,
                        help="Dry-run: fetch N events then stop")
    parser.add_argument("--run", action="store_true",
                        help="Execute the full backfill")
    parser.add_argument("--budget", type=int, default=10000,
                        help="Max credits to spend (default 10000)")
    parser.add_argument("--report", action="store_true",
                        help="Report on existing data (no API calls)")
    args = parser.parse_args()

    print("=" * 70)
    print("PROPS CLV BACKFILL")
    print("=" * 70)

    if args.report:
        results = _compute_prop_clv()
        _report(results)
        return

    # Get events needing close
    events = _get_events_needing_close()
    print(f"\nEvents with entry props, no closing data: {len(events)}")

    if not events:
        print("Nothing to backfill. Run --report to analyze existing data.")
        return

    credits_remaining = _get_credits_remaining()
    print(f"Credits remaining: {credits_remaining:,}")
    print(f"Budget cap: {args.budget:,}")

    if args.test:
        events = events[:args.test]
        print(f"TEST MODE: processing {len(events)} events only")

    n_events = len(events)
    est_cost_events_list = len(set(e["game_date"] for e in events)) * 50
    est_cost_per_event = 40
    est_total = est_cost_events_list + n_events * est_cost_per_event
    print(f"\nEstimated cost:")
    print(f"  Events list calls: ~{est_cost_events_list:,} credits")
    print(f"  Per-event props:   ~{n_events * est_cost_per_event:,} credits")
    print(f"  Total:             ~{est_total:,} credits")

    if not args.run and not args.test:
        print("\nDry run — add --run to execute, or --test N to test N events")
        return

    # Step 1: Get commence times
    print("\n── Step 1: Get commence times ──")
    game_dates = [e["game_date"] for e in events]
    commence_times = _get_commence_times(game_dates)

    # Step 2: Fetch closing props
    print("\n── Step 2: Fetch closing props ──")
    stats = _fetch_closing_props(events, commence_times, args.budget)

    print(f"\n  Results:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # Step 3: Compute CLV
    print("\n── Step 3: Compute de-vigged CLV ──")
    results = _compute_prop_clv()

    # Step 4: Report
    _report(results)


if __name__ == "__main__":
    main()
