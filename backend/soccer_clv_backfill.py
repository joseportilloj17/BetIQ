"""
soccer_clv_backfill.py — CLV (Closing Line Value) backfill for FanDuel soccer bets.

Fetches historical h2h odds from TheOddsAPI to compute CLV on the user's
43 real FanDuel soccer Moneyline/Double Chance legs.

Two-pass approach:
  1. Discovery: fetch events + odds at bet-placement time (team name matching)
  2. Closing:   fetch per-event odds at commence_time (closing line)

CLV = devigged_close_prob - entry_implied_prob (pp)
  Positive → entry price beat the closing line (evidence of real edge).

Usage:
    python soccer_clv_backfill.py                   # dry-run: show legs + cost
    python soccer_clv_backfill.py --run              # run the backfill
    python soccer_clv_backfill.py --run --budget 800 # cap credits
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY = "REDACTED_ODDS_API_KEY"
BASE_URL = "https://api.the-odds-api.com"
BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "soccer_clv_log.jsonl")
CREDITS_PER_CALL = 10
DEFAULT_BUDGET = 800
RATE_LIMIT_SLEEP = 0.6  # seconds between API calls

# ── FanDuel competition → OddsAPI sport key ─────────────────────────────────
_FD_COMP_TO_SPORT_KEY: dict[str, str] = {
    "Spanish La Liga":          "soccer_spain_la_liga",
    "UEFA Champions League":    "soccer_uefa_champs_league",
    "English Premier League":   "soccer_epl",
    "Italian Serie A":          "soccer_italy_serie_a",
    "French Ligue 1":           "soccer_france_ligue_one",
    "German Bundesliga":        "soccer_germany_bundesliga",
    # Copa Del Rey: not available in OddsAPI standard sport keys
}

# ── Team name normalization for fuzzy matching ──────────────────────────────
_TEAM_ALIASES: dict[str, list[str]] = {
    "Barcelona":        ["Barcelona", "FC Barcelona"],
    "Atletico Madrid":  ["Atlético Madrid", "Atletico Madrid", "Atletico", "Atlético"],
    "Man Utd":          ["Manchester United", "Man Utd", "Man United"],
    "Man City":         ["Manchester City", "Man City"],
    "Bayern Munich":    ["Bayern Munich", "Bayern München", "FC Bayern"],
    "Paris St-G":       ["Paris Saint Germain", "PSG", "Paris St-G", "Paris Saint-Germain"],
    "Bodo Glimt":       ["Bodø/Glimt", "Bodo/Glimt", "Bodo Glimt"],
    "Como":             ["Como", "Como 1907"],
    "Monaco":           ["Monaco", "AS Monaco"],
}


def _team_matches(fd_team: str, api_team: str) -> bool:
    """Check if a FanDuel team name matches an OddsAPI team name."""
    fd_norm = fd_team.strip().lower()
    api_norm = api_team.strip().lower()

    # Direct match
    if fd_norm == api_norm or fd_norm in api_norm or api_norm in fd_norm:
        return True

    # Alias match
    for canonical, aliases in _TEAM_ALIASES.items():
        fd_matches = any(a.lower() in fd_norm or fd_norm in a.lower()
                         for a in aliases + [canonical])
        api_matches = any(a.lower() in api_norm or api_norm in a.lower()
                          for a in aliases + [canonical])
        if fd_matches and api_matches:
            return True

    return False


# ── Devig ────────────────────────────────────────────────────────────────────

def _devig_3way(probs: list[float]) -> list[float]:
    """Multiplicative devig for 3-way market. Returns fair probabilities."""
    total = sum(probs)
    if total <= 0:
        return probs
    return [p / total for p in probs]


# ── API helpers ──────────────────────────────────────────────────────────────

def _log(event_id: str, status: int, detail: str):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "event_id": event_id,
        "status": status,
        "detail": detail,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _api_call(url: str, label: str = "") -> tuple[Optional[dict], Optional[int]]:
    """Make an API call. Returns (data, credits_remaining) or (None, None)."""
    try:
        req = Request(url)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            remaining = resp.headers.get("x-requests-remaining")
            _log(label, resp.status, f"ok, remaining={remaining}")
            return data, int(remaining) if remaining else None
    except HTTPError as e:
        body = e.read().decode()[:200] if hasattr(e, "read") else str(e)
        _log(label, e.code, f"HTTP {e.code}: {body}")
        if e.code == 429:
            print(f"  Rate limited, sleeping 60s...")
            time.sleep(60)
        return None, None
    except (URLError, TimeoutError) as e:
        _log(label, 0, f"Network error: {str(e)[:200]}")
        return None, None


# ── Step 1: Extract FanDuel soccer ML legs ──────────────────────────────────

def extract_fd_soccer_legs(conn: sqlite3.Connection) -> list[dict]:
    """Get all FanDuel soccer Moneyline/DC legs with entry odds."""
    rows = conn.execute("""
        SELECT fl.rowid, fl.selection, fl.market_type, fl.competition,
               fl.price, fl.original_price, fl.result,
               fb.placed_date, fb.bet_receipt_id
        FROM fanduel_bet_legs fl
        JOIN fanduel_bets fb ON fl.bet_receipt_id = fb.bet_receipt_id
        WHERE fl.result IN ('WON', 'LOST')
          AND fl.original_price IS NOT NULL
          AND fl.competition IN (
              'Spanish La Liga', 'UEFA Champions League', 'English Premier League',
              'Italian Serie A', 'French Ligue 1', 'German Bundesliga',
              'Spanish Copa Del Rey'
          )
          AND fl.market_type IN ('WIN-DRAW-WIN', 'DOUBLE_CHANCE', 'MONEY_LINE')
        ORDER BY fb.placed_date
    """).fetchall()

    legs = []
    for r in rows:
        selection = r[1] or ""
        is_dc = r[2] == "DOUBLE_CHANCE"
        # Extract team name (remove "And Draw" for DC bets)
        team = selection.replace(" And Draw", "").strip()

        legs.append({
            "rowid": r[0],
            "selection": selection,
            "team": team,
            "is_dc": is_dc,
            "market_type": r[2],
            "competition": r[3],
            "sport_key": _FD_COMP_TO_SPORT_KEY.get(r[3]),
            "entry_price": float(r[4]),  # decimal odds at bet time
            "original_price": float(r[5]),
            "result": r[6],
            "placed_date": r[7],
            "bet_id": r[8],
            # To be filled by matching
            "event_id": None,
            "commence_time": None,
            "home_team": None,
            "away_team": None,
            "close_odds": None,  # {home: x, draw: y, away: z}
        })
    return legs


# ── Step 2: Discovery — match legs to OddsAPI events ───────────────────────

def discover_events(legs: list[dict], budget: int) -> tuple[int, int]:
    """
    Fetch event listings from OddsAPI to match FD legs to event IDs.
    Modifies legs in-place (sets event_id, commence_time, home_team, away_team).
    Returns (credits_spent, credits_remaining).
    """
    # Group unmatched legs by (sport_key, placed_week)
    # Use placed_date to determine which week to query
    queries: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for leg in legs:
        if not leg["sport_key"]:
            continue
        # Query at placed_date + 1 day noon UTC (captures games on/after bet placement)
        placed = leg["placed_date"][:10] if leg["placed_date"] else None
        if placed:
            queries[(leg["sport_key"], placed)].append(leg)

    # De-duplicate query dates: if two dates are within 3 days, use the earlier one
    # (the API response spans ~2 weeks of events)
    unique_queries: dict[str, list[str]] = defaultdict(list)
    for (sport_key, date), _ in sorted(queries.items()):
        unique_queries[sport_key].append(date)

    # Merge dates within 7 days
    merged_queries: list[tuple[str, str]] = []
    for sport_key, dates in unique_queries.items():
        sorted_dates = sorted(set(dates))
        used = set()
        for d in sorted_dates:
            if d in used:
                continue
            merged_queries.append((sport_key, d))
            # Mark nearby dates as covered
            d_dt = datetime.fromisoformat(d)
            for d2 in sorted_dates:
                d2_dt = datetime.fromisoformat(d2)
                if abs((d2_dt - d_dt).days) <= 7:
                    used.add(d2)

    cost = len(merged_queries) * CREDITS_PER_CALL
    print(f"  Discovery: {len(merged_queries)} API calls needed (~{cost} credits)")
    if cost > budget:
        print(f"  WARNING: exceeds budget {budget}, will truncate")
        merged_queries = merged_queries[:budget // CREDITS_PER_CALL]

    # Cache all discovered events
    all_events: dict[str, dict] = {}  # event_id → event data
    credits_spent = 0
    credits_remaining = None

    for i, (sport_key, date) in enumerate(merged_queries):
        params = {
            "apiKey": API_KEY,
            "regions": "us,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "date": f"{date}T12:00:00Z",
        }
        url = f"{BASE_URL}/v4/historical/sports/{sport_key}/odds?{urlencode(params)}"
        data, remaining = _api_call(url, f"discover_{sport_key}_{date}")
        credits_spent += CREDITS_PER_CALL
        if remaining is not None:
            credits_remaining = remaining

        if data:
            events = data.get("data", [])
            for ev in events:
                eid = ev.get("id")
                if eid:
                    all_events[eid] = {
                        "id": eid,
                        "sport_key": sport_key,
                        "home_team": ev.get("home_team", ""),
                        "away_team": ev.get("away_team", ""),
                        "commence_time": ev.get("commence_time", ""),
                    }

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(merged_queries)}] {credits_spent} credits, "
                  f"{len(all_events)} events found, remaining={remaining}")
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"  Discovery done: {len(all_events)} total events, "
          f"{credits_spent} credits spent")

    # Match legs to events by team name + sport_key
    matched = 0
    for leg in legs:
        if leg["event_id"]:
            continue
        if not leg["sport_key"]:
            continue

        team = leg["team"]
        placed_dt = datetime.fromisoformat(leg["placed_date"][:19])

        best_match = None
        best_distance = None

        for eid, ev in all_events.items():
            if ev["sport_key"] != leg["sport_key"]:
                continue
            # Check team name match
            if (_team_matches(team, ev["home_team"]) or
                    _team_matches(team, ev["away_team"])):
                # Check timing: event should be within 7 days of bet placement
                try:
                    ct = datetime.fromisoformat(
                        ev["commence_time"].replace("Z", "+00:00").replace("+00:00", ""))
                    distance = abs((ct - placed_dt).total_seconds())
                    if distance < 7 * 86400:  # within 7 days
                        if best_distance is None or distance < best_distance:
                            best_match = ev
                            best_distance = distance
                except (ValueError, TypeError):
                    pass

        if best_match:
            leg["event_id"] = best_match["id"]
            leg["commence_time"] = best_match["commence_time"]
            leg["home_team"] = best_match["home_team"]
            leg["away_team"] = best_match["away_team"]
            matched += 1

    print(f"  Matched {matched}/{len(legs)} legs to events")
    return credits_spent, credits_remaining or 0


# ── Step 3: Fetch closing odds per event ────────────────────────────────────

def fetch_closing_odds(legs: list[dict], budget: int,
                       credits_spent: int) -> int:
    """
    For each matched leg, fetch h2h odds at commence_time (closing line).
    Modifies legs in-place (sets close_odds).
    Returns additional credits spent.
    """
    # Group by event_id to avoid duplicate API calls
    events_to_fetch: dict[str, dict] = {}
    for leg in legs:
        eid = leg.get("event_id")
        if eid and eid not in events_to_fetch:
            events_to_fetch[eid] = {
                "event_id": eid,
                "sport_key": leg["sport_key"],
                "commence_time": leg["commence_time"],
            }

    remaining_budget = budget - credits_spent
    cost = len(events_to_fetch) * CREDITS_PER_CALL
    print(f"  Closing: {len(events_to_fetch)} unique events to fetch "
          f"(~{cost} credits, budget remaining={remaining_budget})")

    if cost > remaining_budget:
        print(f"  WARNING: exceeds remaining budget, will truncate")
        items = list(events_to_fetch.items())[:remaining_budget // CREDITS_PER_CALL]
        events_to_fetch = dict(items)

    closing_cache: dict[str, dict] = {}  # event_id → {home: x, draw: y, away: z}
    additional_credits = 0

    for i, (eid, ev) in enumerate(events_to_fetch.items()):
        ct = ev["commence_time"]
        ct_str = ct.replace(" ", "T")
        if not ct_str.endswith("Z"):
            ct_str = ct_str.split(".")[0] + "Z"

        sport_key = ev["sport_key"]
        params = {
            "apiKey": API_KEY,
            "regions": "us,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "date": ct_str,
        }
        url = (f"{BASE_URL}/v4/historical/sports/{sport_key}/"
               f"events/{eid}/odds?{urlencode(params)}")

        data, remaining = _api_call(url, f"close_{eid[:20]}")
        additional_credits += CREDITS_PER_CALL

        if data:
            odds = _extract_h2h_odds(data)
            if odds:
                closing_cache[eid] = odds

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(events_to_fetch)}] "
                  f"{additional_credits} credits, remaining={remaining}")
        time.sleep(RATE_LIMIT_SLEEP)

    # Apply closing odds to legs
    for leg in legs:
        eid = leg.get("event_id")
        if eid and eid in closing_cache:
            leg["close_odds"] = closing_cache[eid]

    matched = sum(1 for l in legs if l.get("close_odds"))
    print(f"  Closing odds fetched for {matched}/{len(legs)} legs")
    return additional_credits


def _extract_h2h_odds(data: dict) -> Optional[dict]:
    """
    Extract median h2h odds from API response.
    Returns {home_team: price, away_team: price, 'Draw': price, '_teams': (home, away)}.
    """
    # Handle nested data structure
    inner = data.get("data", data)
    bookmakers = inner.get("bookmakers", [])
    home_team = inner.get("home_team", "")
    away_team = inner.get("away_team", "")

    if not bookmakers:
        return None

    # Collect all h2h prices per outcome
    prices: dict[str, list[float]] = defaultdict(list)
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            for outcome in mkt.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if name and price and price > 1.0:
                    prices[name].append(price)

    if not prices:
        return None

    # Use median across books for each outcome
    import statistics
    result = {"_teams": (home_team, away_team)}
    for name, plist in prices.items():
        result[name] = round(statistics.median(plist), 4)
        result[f"_n_{name}"] = len(plist)

    return result


# ── Step 4: Compute CLV ─────────────────────────────────────────────────────

def compute_clv(legs: list[dict]) -> list[dict]:
    """
    Compute CLV for each matched leg.
    Returns list of result dicts for reporting.
    """
    results = []

    for leg in legs:
        close = leg.get("close_odds")
        if not close:
            continue

        entry_price = leg["original_price"]
        entry_implied = 1.0 / entry_price
        team = leg["team"]
        is_dc = leg["is_dc"]

        home_team, away_team = close.get("_teams", ("", ""))
        draw_price = close.get("Draw")

        # Determine which side we bet on
        if _team_matches(team, home_team):
            our_price = close.get(home_team)
            our_side = "home"
        elif _team_matches(team, away_team):
            our_price = close.get(away_team)
            our_side = "away"
        else:
            continue

        if not our_price or not draw_price:
            continue

        opp_team = away_team if our_side == "home" else home_team
        opp_price = close.get(opp_team)
        if not opp_price:
            continue

        # 3-way devig
        implied = [1.0 / our_price, 1.0 / draw_price, 1.0 / opp_price]
        fair = _devig_3way(implied)
        fair_our = fair[0]
        fair_draw = fair[1]

        if is_dc:
            # Double chance: "Team And Draw" → P(team_win) + P(draw)
            close_fair_prob = fair_our + fair_draw
        else:
            # Straight win (WIN-DRAW-WIN)
            close_fair_prob = fair_our

        # CLV = close_fair_prob - entry_implied (in pp)
        clv_pp = (close_fair_prob - entry_implied) * 100

        results.append({
            "team": team,
            "competition": leg["competition"],
            "market_type": leg["market_type"],
            "placed_date": leg["placed_date"][:10],
            "entry_price": entry_price,
            "entry_implied": round(entry_implied * 100, 1),
            "close_fair_prob": round(close_fair_prob * 100, 1),
            "clv_pp": round(clv_pp, 2),
            "beat_close": 1 if clv_pp > 0 else 0,
            "result": leg["result"],
            "home_team": home_team,
            "away_team": away_team,
            "close_our_price": our_price,
            "close_draw_price": draw_price,
            "is_dc": is_dc,
            "event_id": leg["event_id"],
        })

    return results


# ── Step 5: Report ──────────────────────────────────────────────────────────

def report(results: list[dict]):
    """Print CLV summary and per-leg detail."""
    if not results:
        print("\nNo CLV results to report.")
        return

    print(f"\n{'='*90}")
    print(f"SOCCER CLV REPORT — {len(results)} legs scored")
    print(f"{'='*90}")

    # Per-leg detail
    print(f"\n{'Date':>10s} {'Team':>20s} {'Mkt':>6s} {'Entry':>6s} "
          f"{'EntryP':>7s} {'CloseP':>7s} {'CLV pp':>7s} {'Beat':>4s} {'Result':>6s}")
    print("-" * 82)

    for r in results:
        mkt = "DC" if r["is_dc"] else "W"
        print(f"{r['placed_date']:>10s} {r['team']:>20s} {mkt:>6s} "
              f"{r['entry_price']:>6.3f} {r['entry_implied']:>6.1f}% "
              f"{r['close_fair_prob']:>6.1f}% {r['clv_pp']:>+6.2f} "
              f"{'Y' if r['beat_close'] else 'N':>4s} {r['result']:>6s}")

    # Summary
    avg_clv = sum(r["clv_pp"] for r in results) / len(results)
    beat_pct = sum(r["beat_close"] for r in results) / len(results) * 100
    win_pct = sum(1 for r in results if r["result"] == "WON") / len(results) * 100

    print(f"\n{'─'*60}")
    print(f"SUMMARY:")
    print(f"  Legs scored:      {len(results)}")
    print(f"  Mean CLV:         {avg_clv:+.2f} pp")
    print(f"  Beat close rate:  {beat_pct:.1f}%")
    print(f"  Win rate:         {win_pct:.1f}%")

    # Split by market type
    for mkt_label, is_dc_val in [("WIN-DRAW-WIN (straight)", False),
                                  ("DOUBLE_CHANCE", True)]:
        subset = [r for r in results if r["is_dc"] == is_dc_val]
        if subset:
            avg = sum(r["clv_pp"] for r in subset) / len(subset)
            beat = sum(r["beat_close"] for r in subset) / len(subset) * 100
            wr = sum(1 for r in subset if r["result"] == "WON") / len(subset) * 100
            print(f"\n  {mkt_label} (N={len(subset)}):")
            print(f"    Mean CLV: {avg:+.2f} pp  |  Beat: {beat:.1f}%  |  WR: {wr:.1f}%")

    # By competition
    print(f"\n  By competition:")
    comps = defaultdict(list)
    for r in results:
        comps[r["competition"]].append(r)
    for comp, subset in sorted(comps.items(), key=lambda x: -len(x[1])):
        avg = sum(r["clv_pp"] for r in subset) / len(subset)
        beat = sum(r["beat_close"] for r in subset) / len(subset) * 100
        print(f"    {comp:30s} N={len(subset):>3d}  CLV={avg:>+6.2f}pp  Beat={beat:>5.1f}%")


# ── Store results ────────────────────────────────────────────────────────────

def store_results(results: list[dict], conn: sqlite3.Connection):
    """Store CLV results in a table for future reference."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fanduel_soccer_clv (
            event_id         TEXT,
            team             TEXT,
            competition      TEXT,
            market_type      TEXT,
            placed_date      TEXT,
            entry_price      REAL,
            entry_implied    REAL,
            close_fair_prob  REAL,
            clv_pp           REAL,
            beat_close       INTEGER,
            result           TEXT,
            home_team        TEXT,
            away_team        TEXT,
            is_dc            INTEGER,
            computed_at      TEXT,
            UNIQUE(event_id, team, market_type, placed_date)
        )
    """)

    now = datetime.utcnow().isoformat()
    for r in results:
        conn.execute("""
            INSERT OR REPLACE INTO fanduel_soccer_clv
                (event_id, team, competition, market_type, placed_date,
                 entry_price, entry_implied, close_fair_prob, clv_pp,
                 beat_close, result, home_team, away_team, is_dc, computed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (r["event_id"], r["team"], r["competition"], r["market_type"],
              r["placed_date"], r["entry_price"], r["entry_implied"],
              r["close_fair_prob"], r["clv_pp"], r["beat_close"],
              r["result"], r["home_team"], r["away_team"],
              1 if r["is_dc"] else 0, now))
    conn.commit()
    print(f"\n  Stored {len(results)} CLV records in fanduel_soccer_clv table")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Soccer CLV backfill")
    parser.add_argument("--run", action="store_true",
                        help="Execute the backfill (API calls)")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help=f"Max credits (default {DEFAULT_BUDGET})")
    args = parser.parse_args()

    conn = sqlite3.connect(BETS_DB)

    # Step 1: Extract legs
    print("Step 1: Extracting FanDuel soccer ML/DC legs...")
    legs = extract_fd_soccer_legs(conn)
    print(f"  {len(legs)} legs found")

    # Show what we're working with
    no_key = sum(1 for l in legs if not l["sport_key"])
    print(f"  {len(legs) - no_key} matchable (have sport_key), "
          f"{no_key} unmatchable (Copa Del Rey)")

    # Estimate cost
    sport_dates = set()
    for leg in legs:
        if leg["sport_key"] and leg["placed_date"]:
            sport_dates.add((leg["sport_key"], leg["placed_date"][:10]))
    print(f"  ~{len(sport_dates)} unique (sport, date) pairs")
    print(f"  Estimated cost: ~{len(sport_dates) * CREDITS_PER_CALL} discovery + "
          f"~300 closing = ~{len(sport_dates) * CREDITS_PER_CALL + 300} credits")

    if not args.run:
        print("\n  Dry run — pass --run to execute.")
        conn.close()
        return

    # Step 2: Discover events
    print(f"\nStep 2: Discovering events (budget={args.budget})...")
    credits_spent, credits_remaining = discover_events(legs, args.budget)

    # Step 3: Fetch closing odds
    print(f"\nStep 3: Fetching closing odds...")
    closing_credits = fetch_closing_odds(legs, args.budget, credits_spent)
    credits_spent += closing_credits

    # Step 4: Compute CLV
    print(f"\nStep 4: Computing CLV...")
    results = compute_clv(legs)

    # Step 5: Report
    report(results)

    # Store
    if results:
        store_results(results, conn)

    print(f"\nTotal credits spent: {credits_spent}")
    print(f"API log: {LOG_FILE}")

    conn.close()


if __name__ == "__main__":
    main()
