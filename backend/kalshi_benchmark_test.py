#!/usr/bin/env python3
"""
Kalshi as Sharp Benchmark — THE RESOLUTION (v3 — fixed)
=========================================================
v2 BUG: normalized home_neg15 + away_neg15 to sum to 1.0.
But these are NOT complementary! Both can lose (game within 1 run).
FIX: Use Kalshi yes_price DIRECTLY as P(team covers -1.5).
     No normalization needed — exchange price IS the probability.

Approach:
- For each game, use Kalshi "Home wins by >1.5 runs" market
- yes_price = P(home -1.5 covers), no_price = P(home does NOT cover)
- The NO side = P(away +1.5 covers) [they're complements within one market]
- Compare directly against soft book de-vigged P(home -1.5)
"""

import json
import sqlite3
import statistics
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta

DB_PATH = "data/historical.db"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

KALSHI_TO_FULL = {
    "AZ": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "ATH": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}
FULL_TO_KALSHI = {v: k for k, v in KALSHI_TO_FULL.items()}


def kalshi_get(path, params=None, retries=3):
    url = f"{KALSHI_BASE}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{qs}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            print(f"    API error: {e}")
            return None
    return None


def parse_market_ticker(ticker):
    """Parse KXMLBSPREAD-26AUGDDHHMMteamAteamB-TEAM2"""
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    raw = parts[1]
    suffix = parts[2]

    if not suffix.endswith("2"):
        return None

    team_code = suffix[:-1]
    if team_code not in KALSHI_TO_FULL:
        return None

    try:
        year = 2000 + int(raw[:2])
        month_str = raw[2:5]
        day = int(raw[5:7])
        hour = int(raw[7:9])
        minute = int(raw[9:11])
        teams_str = raw[11:]

        months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,
                  "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,
                  "NOV": 11, "DEC": 12}
        month = months.get(month_str)
        if not month:
            return None

        game_dt_et = datetime(year, month, day, hour, minute)
        game_dt_utc = game_dt_et + timedelta(hours=4)

        team_a = team_b = None
        for code in sorted(KALSHI_TO_FULL.keys(), key=len, reverse=True):
            if teams_str.startswith(code):
                team_a = code
                remainder = teams_str[len(code):]
                if remainder in KALSHI_TO_FULL:
                    team_b = remainder
                    break
                team_a = None

        if not team_a or not team_b:
            return None

        game_date = f"{year}-{month:02d}-{day:02d}"
        event_key = f"{parts[0]}-{parts[1]}"  # e.g. KXMLBSPREAD-26AUG091410CLECWS
        return {
            "game_date": game_date,
            "game_time_utc": game_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "team_a_code": team_a,
            "team_b_code": team_b,
            "spread_team_code": team_code,
            "spread_team_full": KALSHI_TO_FULL[team_code],
            "event_key": event_key,
        }
    except (ValueError, IndexError):
        return None


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print("=" * 70)
    print("KALSHI AS SHARP BENCHMARK — THE RESOLUTION (v3 fixed)")
    print("Using Kalshi yes_price directly — NO normalization")
    print("=" * 70)

    # ── Step 1: Bulk-pull ALL settled -1.5 spread markets ──
    print("\n── Step 1: Bulk-pulling settled spread markets... ──")
    all_markets = []
    cursor = ""
    page = 0
    while True:
        params = {"series_ticker": "KXMLBSPREAD", "status": "settled", "limit": "200"}
        if cursor:
            params["cursor"] = cursor
        data = kalshi_get("/markets", params)
        if not data:
            break
        batch = data.get("markets", [])
        if not batch:
            break
        all_markets.extend(batch)
        cursor = data.get("cursor", "")
        page += 1
        if page % 10 == 0:
            print(f"    Page {page}: total {len(all_markets)}")
        if not cursor or page > 50:
            break
        time.sleep(0.3)

    print(f"  Total settled markets: {len(all_markets)}")

    # Filter to -1.5 and parse
    m15 = []
    for m in all_markets:
        parsed = parse_market_ticker(m["ticker"])
        if parsed:
            m["_parsed"] = parsed
            m15.append(m)

    print(f"  -1.5 spread markets: {len(m15)}")

    # ── Step 2: Load DB data ──
    print("\n── Step 2: Loading DB data... ──")

    outcomes = {}
    for r in db.execute("""
        SELECT home_team, away_team, game_date, home_score, away_score
        FROM games WHERE sport='MLB' AND home_score IS NOT NULL
    """).fetchall():
        outcomes[(r["home_team"], r["away_team"], r["game_date"])] = (
            r["home_score"], r["away_score"])

    alt_rows = db.execute("""
        SELECT fetched_at, bookmaker, team, odds, line,
               home_team, away_team, game_date_ct
        FROM alt_lines
        WHERE sport_key='baseball_mlb' AND market_key='alternate_spreads'
          AND line IN (-1.5, 1.5)
          AND bookmaker IN ('pinnacle', 'draftkings', 'fanduel', 'betmgm')
        ORDER BY fetched_at DESC
    """).fetchall()

    # Keep latest per (game, book, team, line)
    alt_latest = {}
    for r in alt_rows:
        key = (r["home_team"], r["away_team"], r["game_date_ct"],
               r["bookmaker"], r["team"], r["line"])
        if key not in alt_latest:
            alt_latest[key] = r["odds"]

    alt_by_game = defaultdict(lambda: defaultdict(dict))
    for (home, away, gdate, book, team, line), odds in alt_latest.items():
        alt_by_game[(home, away, gdate)][book][(team, line)] = odds

    print(f"  Outcomes: {len(outcomes)}, Alt_lines games: {len(alt_by_game)}")

    # ── Step 3: Match Kalshi markets to DB ──
    # KEY FIX: each market stands alone. "Home wins by >1.5" gives P(home -1.5)
    # directly. No pairing needed.
    print("\n── Step 3: Matching Kalshi → DB (per-market, not per-pair)... ──")

    # For each Kalshi -1.5 market, find the matching game in our DB
    # and determine if that team is home or away
    matched_markets = []

    for m in m15:
        p = m["_parsed"]
        spread_team = p["spread_team_full"]
        team_a_full = KALSHI_TO_FULL.get(p["team_a_code"])
        team_b_full = KALSHI_TO_FULL.get(p["team_b_code"])
        if not team_a_full or not team_b_full:
            continue

        gdate = p["game_date"]

        # Find outcome — try both orderings
        # Kalshi ticker format: AWAYCODE + HOMECODE
        # But we don't know for sure, so try both
        outcome = None
        home_full = away_full = None
        for h, a in [(team_b_full, team_a_full), (team_a_full, team_b_full)]:
            outcome = outcomes.get((h, a, gdate))
            if outcome:
                home_full, away_full = h, a
                break

        if not outcome:
            continue

        game_key = (home_full, away_full, gdate)
        if game_key not in alt_by_game:
            continue

        home_score, away_score = outcome
        # This market is "spread_team wins by >1.5"
        # outcome: did spread_team actually cover -1.5?
        if spread_team == home_full:
            covers = (home_score - away_score) >= 2
            is_home_side = True
        elif spread_team == away_full:
            covers = (away_score - home_score) >= 2
            is_home_side = False
        else:
            continue

        matched_markets.append({
            "market": m,
            "parsed": p,
            "spread_team": spread_team,
            "home_full": home_full,
            "away_full": away_full,
            "gdate": gdate,
            "covers": covers,  # did this team cover -1.5?
            "is_home_side": is_home_side,
            "alt_data": alt_by_game[game_key],
            "game_key": game_key,
        })

    print(f"  Matched -1.5 markets: {len(matched_markets)}")

    # Deduplicate: keep one market per (game, team) — prefer home side
    # Actually, keep ALL markets (each is an independent data point)
    # But for the game-level Brier comparison, we want one per game
    # Group by game to get home-side market per game
    by_game = defaultdict(list)
    for mm in matched_markets:
        by_game[mm["game_key"]].append(mm)

    print(f"  Unique games: {len(by_game)}")

    # ── Step 4: Fetch pre-game trades ──
    # v3 bug: only fetched 1 page (200 trades) — selection bias toward
    # low-volume (losing) markets. Fix: paginate up to 10 pages per market.
    print(f"\n── Step 4: Fetching pre-game trades for {len(matched_markets)} markets... ──")
    print(f"    (paginating up to 10 pages per market to find pre-game trades)")

    results = []
    no_pregame = 0
    has_pregame = 0

    for i, mm in enumerate(matched_markets):
        ticker = mm["market"]["ticker"]
        game_time = mm["parsed"]["game_time_utc"]

        # Paginate through trades until we find pre-game ones
        pregame = []
        cursor = ""
        for page in range(10):  # up to 10 pages = 2000 trades
            params = {"ticker": ticker, "limit": "200"}
            if cursor:
                params["cursor"] = cursor
            trades_data = kalshi_get("/markets/trades", params)
            time.sleep(0.3)

            if not trades_data or not trades_data.get("trades"):
                break

            trades = trades_data["trades"]
            page_pregame = [(float(t["yes_price_dollars"]), float(t["count_fp"]))
                            for t in trades if t["created_time"] < game_time]
            pregame.extend(page_pregame)

            # If we found pre-game trades on this page, we have enough
            if page_pregame:
                break

            # If oldest trade on this page is still after game time, keep paging
            oldest = trades[-1]["created_time"]
            if oldest >= game_time:
                cursor = trades_data.get("cursor", "")
                if not cursor:
                    break
            else:
                break

        if not pregame:
            no_pregame += 1
            continue

        has_pregame += 1

        # VWAP of pre-game trades = Kalshi's fair P(team covers -1.5)
        total_vol = sum(p[1] for p in pregame)
        if total_vol > 0:
            kalshi_prob = sum(p[0] * p[1] for p in pregame) / total_vol
        else:
            kalshi_prob = statistics.median([p[0] for p in pregame])

        # Get soft book de-vigged P for the SAME team covering -1.5
        soft_probs = {}
        spread_team = mm["spread_team"]
        home = mm["home_full"]
        away = mm["away_full"]

        for book in ["draftkings", "fanduel", "betmgm", "pinnacle"]:
            if book not in mm["alt_data"]:
                continue
            bd = mm["alt_data"][book]

            if mm["is_home_side"]:
                # Market is "home -1.5". De-vig pair: home -1.5 vs away +1.5
                odds_neg = bd.get((home, -1.5))
                odds_pos = bd.get((away, 1.5))
            else:
                # Market is "away -1.5". De-vig pair: away -1.5 vs home +1.5
                odds_neg = bd.get((away, -1.5))
                odds_pos = bd.get((home, 1.5))

            if not odds_neg or not odds_pos or odds_neg <= 1 or odds_pos <= 1:
                continue

            i_neg = 1.0 / odds_neg
            i_pos = 1.0 / odds_pos
            t = i_neg + i_pos
            soft_probs[book] = i_neg / t  # P(team covers -1.5)

        if not any(b in soft_probs for b in ["draftkings", "fanduel", "betmgm"]):
            continue

        results.append({
            "kalshi_prob": kalshi_prob,  # P(team covers -1.5)
            "soft_probs": soft_probs,
            "covers": mm["covers"],
            "is_home_side": mm["is_home_side"],
            "spread_team": spread_team,
            "n_pregame": len(pregame),
            "pregame_vol": total_vol,
        })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(matched_markets)} processed, "
                  f"{len(results)} with data, {no_pregame} no pregame...")

    print(f"\n  Results: {len(results)} markets with pre-game Kalshi + soft book data")
    print(f"  Had pre-game trades: {has_pregame}")
    print(f"  No pre-game trades (skipped): {no_pregame}")

    if len(results) < 10:
        print("  INSUFFICIENT DATA.")
        db.close()
        return

    # Sanity checks
    avg_kalshi = statistics.mean([r["kalshi_prob"] for r in results])
    base_rate = sum(r["covers"] for r in results) / len(results)
    avg_pregame_n = statistics.mean([r["n_pregame"] for r in results])
    avg_vol = statistics.mean([r["pregame_vol"] for r in results])
    print(f"\n  Sanity checks:")
    print(f"    Avg Kalshi prob: {avg_kalshi:.3f} (should be ~0.35)")
    print(f"    Actual cover rate: {base_rate:.3f} (should be ~0.35)")
    print(f"    Avg pre-game trades per market: {avg_pregame_n:.0f}")
    print(f"    Avg pre-game volume: ${avg_vol:.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: BRIER — Is Kalshi sharper?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: OUTCOME PREDICTION (Brier Score)")
    print("Who predicts team covering -1.5 best? Lower = better.")
    print("=" * 70)

    brier = defaultdict(list)
    for r in results:
        actual = 1 if r["covers"] else 0
        brier["kalshi"].append((r["kalshi_prob"] - actual) ** 2)
        for book, prob in r["soft_probs"].items():
            brier[book].append((prob - actual) ** 2)

    print(f"\n  {'Source':>15s} {'Brier':>10s} {'N':>6s} {'Avg Prob':>9s}")
    print("  " + "-" * 45)
    brier_means = {}
    for source in ["kalshi", "pinnacle", "draftkings", "fanduel", "betmgm"]:
        if source not in brier or not brier[source]:
            continue
        b = statistics.mean(brier[source])
        vals = ([r["kalshi_prob"] for r in results] if source == "kalshi"
                else [r["soft_probs"][source] for r in results
                      if source in r["soft_probs"]])
        avg_p = statistics.mean(vals) if vals else 0
        brier_means[source] = b
        print(f"  {source:>15s} {b:>10.6f} {len(brier[source]):>6d} {avg_p:>9.4f}")

    # Consensus
    cons_brier_list = []
    for r in results:
        sf = [r["soft_probs"][b] for b in ["draftkings", "fanduel", "betmgm"]
              if b in r["soft_probs"]]
        if sf:
            cons = statistics.mean(sf)
            actual = 1 if r["covers"] else 0
            cons_brier_list.append((cons - actual) ** 2)
    if cons_brier_list:
        cb = statistics.mean(cons_brier_list)
        brier_means["consensus"] = cb
        print(f"  {'consensus':>15s} {cb:>10.6f} {len(cons_brier_list):>6d}")

    ranked = sorted(brier_means.items(), key=lambda x: x[1])
    print(f"\n  Ranking:")
    for i, (src, b) in enumerate(ranked, 1):
        marker = " ← SHARPEST" if i == 1 else ""
        print(f"    {i}. {src:15s} {b:.6f}{marker}")

    kalshi_b = brier_means.get("kalshi", 999)
    best_soft_b = min((brier_means.get(b, 999)
                       for b in ["draftkings", "fanduel", "betmgm"]
                       if b in brier_means), default=999)
    best_soft_name = min(
        [b for b in ["draftkings", "fanduel", "betmgm"] if b in brier_means],
        key=lambda b: brier_means[b], default="?")
    pin_b = brier_means.get("pinnacle", float("nan"))

    print(f"\n  Kalshi vs best soft ({best_soft_name}): "
          f"{kalshi_b - best_soft_b:+.6f} ({'SHARPER' if kalshi_b < best_soft_b else 'WORSE'})")
    if "pinnacle" in brier_means:
        print(f"  Kalshi vs Pinnacle: "
              f"{kalshi_b - pin_b:+.6f} ({'SHARPER' if kalshi_b < pin_b else 'WORSE'})")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: SIDE SPLIT — +1.5 UNDERDOG vs -1.5 FAVORITE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: SIDE SPLIT — +1.5 vs -1.5 EDGE vs KALSHI")
    print("When soft books disagree with Kalshi, which side has edge?")
    print("=" * 70)

    # For each result, if soft_prob > kalshi_prob: soft thinks team MORE likely
    # to cover -1.5 → soft is "stale high" → bet the OTHER side (+1.5)
    # If soft_prob < kalshi_prob: soft underprices -1.5 → bet -1.5

    dog_bets = []   # Bet +1.5 (against covering -1.5)
    fav_bets = []   # Bet -1.5 (for covering)

    for r in results:
        kp = r["kalshi_prob"]
        for book in ["draftkings", "fanduel", "betmgm"]:
            if book not in r["soft_probs"]:
                continue
            sp = r["soft_probs"][book]
            diff = sp - kp  # >0 = soft overprices -1.5

            if abs(diff) < 0.01:  # < 1pp disagreement
                continue

            if diff > 0:
                # Soft overprices -1.5 → underprices +1.5
                # Bet +1.5: wins when team does NOT cover -1.5
                dog_bets.append({
                    "book": book, "diff_pp": diff * 100,
                    "won": not r["covers"],
                    "kalshi_fair": 1 - kp,
                    "is_home_side": r["is_home_side"],
                })
            else:
                # Soft underprices -1.5 → bet -1.5
                fav_bets.append({
                    "book": book, "diff_pp": abs(diff) * 100,
                    "won": r["covers"],
                    "kalshi_fair": kp,
                    "is_home_side": r["is_home_side"],
                })

    def report_side(label, bets):
        if not bets:
            print(f"\n  {label}: No bets")
            return None
        wins = sum(b["won"] for b in bets)
        wr = wins / len(bets) * 100
        exp = statistics.mean([b["kalshi_fair"] for b in bets]) * 100
        edge = wr - exp
        avg_diff = statistics.mean([b["diff_pp"] for b in bets])
        print(f"\n  {label}:")
        print(f"    N={len(bets)}, WR={wr:.1f}%, Kalshi exp={exp:.1f}%, "
              f"edge={edge:+.1f}pp")
        print(f"    Avg disagreement: {avg_diff:.1f}pp")

        for book in ["draftkings", "fanduel", "betmgm"]:
            sub = [b for b in bets if b["book"] == book]
            if sub:
                sw = sum(b["won"] for b in sub) / len(sub) * 100
                se = statistics.mean([b["kalshi_fair"] for b in sub]) * 100
                print(f"      {book:15s}: WR={sw:.1f}% vs exp={se:.1f}% "
                      f"({sw-se:+.1f}pp), N={len(sub)}")
        return edge

    dog_edge = report_side(
        "A. +1.5 UNDERDOG (soft overprices -1.5 → bet +1.5 vs Kalshi)", dog_bets)
    fav_edge = report_side(
        "B. -1.5 FAVORITE (soft underprices -1.5 → bet -1.5 vs Kalshi)", fav_bets)

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: SPLIT-HALF
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: SPLIT-HALF STABILITY")
    print("=" * 70)

    for label, bets in [("+1.5 dog", dog_bets), ("-1.5 fav", fav_bets)]:
        if len(bets) < 20:
            print(f"  {label}: N={len(bets)} too small for split-half")
            continue
        half = len(bets) // 2
        h1, h2 = bets[:half], bets[half:]
        wr1 = sum(b["won"] for b in h1) / len(h1) * 100
        wr2 = sum(b["won"] for b in h2) / len(h2) * 100
        e1 = statistics.mean([b["kalshi_fair"] for b in h1]) * 100
        e2 = statistics.mean([b["kalshi_fair"] for b in h2]) * 100
        print(f"  {label}: H1={wr1-e1:+.1f}pp (N={len(h1)}), "
              f"H2={wr2-e2:+.1f}pp (N={len(h2)}), "
              f"gap={abs((wr1-e1)-(wr2-e2)):.1f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: KALSHI vs PINNACLE HEAD-TO-HEAD
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 4: KALSHI vs PINNACLE")
    print("=" * 70)

    both = [r for r in results if "pinnacle" in r["soft_probs"]]
    if len(both) >= 10:
        k_br = statistics.mean(
            [(r["kalshi_prob"] - (1 if r["covers"] else 0)) ** 2 for r in both])
        p_br = statistics.mean(
            [(r["soft_probs"]["pinnacle"] - (1 if r["covers"] else 0)) ** 2
             for r in both])
        print(f"  On {len(both)} shared markets:")
        print(f"    Kalshi Brier:    {k_br:.6f}")
        print(f"    Pinnacle Brier:  {p_br:.6f}")
        print(f"    Gap: {k_br - p_br:+.6f} "
              f"({'Kalshi SHARPER' if k_br < p_br else 'Pinnacle SHARPER'})")
        mae = statistics.mean(
            [abs(r["kalshi_prob"] - r["soft_probs"]["pinnacle"]) * 100
             for r in both])
        print(f"    MAE between them: {mae:.2f}pp")
    else:
        print(f"  Only {len(both)} shared markets — too few")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    print(f"""
  SHARPNESS (Brier, lower = better):
    Kalshi:           {kalshi_b:.6f}
    Best soft book:   {best_soft_name} = {best_soft_b:.6f}
    Pinnacle:         {pin_b:.6f}
    → Kalshi {'IS' if kalshi_b < best_soft_b else 'is NOT'} sharper than soft books
    → Kalshi {'IS' if kalshi_b < pin_b else 'is NOT'} sharper than Pinnacle

  SIDE SPLIT vs KALSHI:
    +1.5 underdog: {f'{dog_edge:+.1f}pp' if dog_edge is not None else 'N/A'} (N={len(dog_bets)})
    -1.5 favorite: {f'{fav_edge:+.1f}pp' if fav_edge is not None else 'N/A'} (N={len(fav_bets)})

  +1.5 VERDICT: {'SURVIVES — edge real against sharp benchmark' if dog_edge and dog_edge > 2 else 'DEAD — no edge vs sharp benchmark' if dog_edge and dog_edge < -2 else 'MARGINAL (±2pp)' if dog_edge is not None else 'N/A'}
  -1.5 VERDICT: {'NEGATIVE confirmed (fast propagation)' if fav_edge and fav_edge < -2 else 'NOT negative' if fav_edge and fav_edge > 2 else 'MARGINAL' if fav_edge is not None else 'N/A'}

  N={len(results)} market-observations
  Avg pre-game trades: {avg_pregame_n:.0f}, avg volume: ${avg_vol:.0f}
  Base rate: {base_rate:.3f}, Kalshi avg: {avg_kalshi:.3f}
""")

    db.close()


if __name__ == "__main__":
    main()
