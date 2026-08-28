#!/usr/bin/env python3
"""
Kalshi Four-Cell Gate — THE FOUNDATION TEST
=============================================
Validates the underdog cross-book edge against Kalshi (exchange).

Step 0: Coverage check — how many games have liquid Kalshi markets?
Step 1: Four-cell split (fav/dog × rung) against Kalshi benchmark
Step 2: Verdict — real edge / proxy artifact / inconclusive

Caches Kalshi trade data to JSON to avoid re-fetching (~1hr).
"""

import json
import os
import sqlite3
import statistics
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta

DB_PATH = "data/historical.db"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
CACHE_FILE = "data/kalshi_spread_cache.json"

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
            return None
    return None


def parse_market_ticker(ticker):
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
        months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        month = months.get(month_str)
        if not month:
            return None
        game_dt_utc = datetime(year, month, day, hour, minute) + timedelta(hours=4)
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
        return {
            "game_date": f"{year}-{month:02d}-{day:02d}",
            "game_time_utc": game_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "team_a_code": team_a, "team_b_code": team_b,
            "spread_team_code": team_code,
            "spread_team_full": KALSHI_TO_FULL[team_code],
        }
    except (ValueError, IndexError):
        return None


def fetch_kalshi_data():
    """Fetch all settled -1.5 spread markets + pre-game trades. Cache to disk."""
    if os.path.exists(CACHE_FILE):
        print(f"  Loading from cache: {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            return json.load(f)

    print("  Fetching from Kalshi API (this takes ~1 hour)...")

    # Step 1: Bulk-pull all settled markets
    print("  Pulling settled spread markets...")
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
            print(f"    Page {page}: {len(all_markets)} markets")
        if not cursor or page > 50:
            break
        time.sleep(0.3)

    print(f"  Total markets: {len(all_markets)}")

    # Filter to -1.5 and parse
    m15 = []
    for m in all_markets:
        parsed = parse_market_ticker(m["ticker"])
        if parsed:
            m15.append({"ticker": m["ticker"], "parsed": parsed,
                        "volume": m.get("volume_fp", "0"),
                        "result": m.get("result", "")})

    print(f"  -1.5 spread markets: {len(m15)}")

    # Step 2: Fetch pre-game trades for each
    print(f"  Fetching pre-game trades for {len(m15)} markets...")
    results = []

    for i, m in enumerate(m15):
        ticker = m["ticker"]
        game_time = m["parsed"]["game_time_utc"]

        pregame = []
        cursor = ""
        for page in range(10):
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
            if page_pregame:
                break
            oldest = trades[-1]["created_time"]
            if oldest >= game_time:
                cursor = trades_data.get("cursor", "")
                if not cursor:
                    break
            else:
                break

        if pregame:
            total_vol = sum(p[1] for p in pregame)
            if total_vol > 0:
                vwap = sum(p[0] * p[1] for p in pregame) / total_vol
            else:
                vwap = statistics.median([p[0] for p in pregame])

            results.append({
                "ticker": ticker,
                "parsed": m["parsed"],
                "kalshi_prob": vwap,
                "n_pregame": len(pregame),
                "pregame_vol": total_vol,
                "volume": m["volume"],
                "result": m["result"],
            })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(m15)}, {len(results)} with pregame data")

    print(f"  Total with pre-game data: {len(results)}")

    # Save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached to {CACHE_FILE}")

    return results


def _implied(odds):
    return 1.0 / odds if odds > 1.0 else 1.0


def _devig(odds_a, odds_b):
    ia, ib = _implied(odds_a), _implied(odds_b)
    t = ia + ib
    return (ia / t, ib / t) if t > 0 else (0.5, 0.5)


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print("=" * 70)
    print("KALSHI FOUR-CELL GATE — THE FOUNDATION TEST")
    print("Underdog edge vs genuinely sharp benchmark")
    print("=" * 70)

    # ── Fetch/load Kalshi data ──
    print("\n── Kalshi data ──")
    kalshi_data = fetch_kalshi_data()
    print(f"  Kalshi markets with pre-game prices: {len(kalshi_data)}")

    # ── Load DB data ──
    print("\n── Loading DB data ──")
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

    alt_latest = {}
    for r in alt_rows:
        key = (r["home_team"], r["away_team"], r["game_date_ct"],
               r["bookmaker"], r["team"], r["line"])
        if key not in alt_latest:
            alt_latest[key] = r["odds"]

    alt_by_game = defaultdict(lambda: defaultdict(dict))
    for (home, away, gdate, book, team, line), odds in alt_latest.items():
        alt_by_game[(home, away, gdate)][book][(team, line)] = odds

    # ── Match Kalshi → DB ──
    print("\n── Matching Kalshi → DB ──")

    matched = []
    for kd in kalshi_data:
        p = kd["parsed"]
        spread_team = p["spread_team_full"]
        ta_full = KALSHI_TO_FULL.get(p["team_a_code"])
        tb_full = KALSHI_TO_FULL.get(p["team_b_code"])
        if not ta_full or not tb_full:
            continue

        gdate = p["game_date"]
        outcome = None
        home_full = away_full = None
        for h, a in [(tb_full, ta_full), (ta_full, tb_full)]:
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
        if spread_team == home_full:
            covers = (home_score - away_score) >= 2
        elif spread_team == away_full:
            covers = (away_score - home_score) >= 2
        else:
            continue

        # Determine favorite using Pinnacle or soft consensus
        books = alt_by_game[game_key]
        ref = books.get("pinnacle", {})
        if not ref:
            for b in ["draftkings", "fanduel", "betmgm"]:
                if b in books:
                    ref = books[b]
                    break
        if not ref:
            continue

        home_neg15_odds = ref.get((home_full, -1.5))
        away_neg15_odds = ref.get((away_full, -1.5))
        if not home_neg15_odds or not away_neg15_odds:
            continue

        # Favorite = lower -1.5 odds (higher implied prob)
        if home_neg15_odds <= away_neg15_odds:
            fav, dog = home_full, away_full
        else:
            fav, dog = away_full, home_full

        # What cell is this market?
        # Market is "spread_team wins by >1.5" = spread_team -1.5
        if spread_team == fav:
            cell_neg15 = "fav_neg15"  # YES = fav covers -1.5
            cell_pos15 = "dog_pos15"  # NO = dog covers +1.5
        else:
            cell_neg15 = "dog_neg15"  # YES = dog covers -1.5
            cell_pos15 = "fav_pos15"  # NO = fav covers +1.5

        # Compute soft book de-vigged probs for comparison
        # The de-vig pair: spread_team -1.5 vs other_team +1.5
        other_team = away_full if spread_team == home_full else home_full

        soft_probs = {}
        for book in ["draftkings", "fanduel", "betmgm"]:
            if book not in books:
                continue
            bd = books[book]
            odds_neg = bd.get((spread_team, -1.5))
            odds_pos = bd.get((other_team, 1.5))
            if not odds_neg or not odds_pos or odds_neg <= 1 or odds_pos <= 1:
                continue
            fair_neg, fair_pos = _devig(odds_neg, odds_pos)
            vig = (_implied(odds_neg) + _implied(odds_pos) - 1) * 100
            soft_probs[book] = {
                "fair_neg15": fair_neg,  # P(spread_team covers -1.5)
                "fair_pos15": fair_pos,  # P(other_team covers +1.5)
                "vig": vig,
                "odds_neg": odds_neg,
                "odds_pos": odds_pos,
            }

        if not soft_probs:
            continue

        # Also get Pinnacle de-vigged for comparison
        pin_prob = None
        if "pinnacle" in books:
            bd = books["pinnacle"]
            odds_neg = bd.get((spread_team, -1.5))
            odds_pos = bd.get((other_team, 1.5))
            if odds_neg and odds_pos and odds_neg > 1 and odds_pos > 1:
                fair_neg, fair_pos = _devig(odds_neg, odds_pos)
                pin_prob = fair_neg

        matched.append({
            "kalshi_prob": kd["kalshi_prob"],  # P(spread_team covers -1.5)
            "covers": covers,
            "cell_neg15": cell_neg15,
            "cell_pos15": cell_pos15,
            "spread_team": spread_team,
            "other_team": other_team,
            "fav": fav, "dog": dog,
            "soft_probs": soft_probs,
            "pin_prob": pin_prob,
            "n_pregame": kd["n_pregame"],
            "pregame_vol": kd["pregame_vol"],
        })

    print(f"  Matched: {len(matched)}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 0: COVERAGE CHECK
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 0: KALSHI COVERAGE CHECK")
    print("=" * 70)

    total_kalshi = len(kalshi_data)
    matched_n = len(matched)
    avg_pregame_n = statistics.mean([m["n_pregame"] for m in matched]) if matched else 0
    avg_vol = statistics.mean([m["pregame_vol"] for m in matched]) if matched else 0

    # Liquidity tiers
    thin = sum(1 for m in matched if m["pregame_vol"] < 100)
    moderate = sum(1 for m in matched if 100 <= m["pregame_vol"] < 1000)
    liquid = sum(1 for m in matched if m["pregame_vol"] >= 1000)

    # Unique games
    unique_games = len(set(
        (m["fav"], m["dog"]) for m in matched))

    print(f"""
  Kalshi settled -1.5 markets: {total_kalshi}
  Matched to our DB (alt_lines + outcomes): {matched_n}
  Unique games: {unique_games}
  Avg pre-game trades per market: {avg_pregame_n:.0f}
  Avg pre-game volume: ${avg_vol:.0f}

  Liquidity tiers:
    Thin (<$100 vol):     {thin} ({thin/max(matched_n,1)*100:.0f}%)
    Moderate ($100-1K):   {moderate} ({moderate/max(matched_n,1)*100:.0f}%)
    Liquid (>$1K):        {liquid} ({liquid/max(matched_n,1)*100:.0f}%)

  Coverage verdict: {'SUFFICIENT' if liquid >= 100 else 'MARGINAL' if liquid >= 30 else 'THIN — treat results as directional only'}
""")

    # Sanity
    base_rate = sum(m["covers"] for m in matched) / max(len(matched), 1)
    avg_kalshi = statistics.mean([m["kalshi_prob"] for m in matched]) if matched else 0
    print(f"  Sanity: cover rate={base_rate:.3f} (expect ~0.35), "
          f"avg Kalshi prob={avg_kalshi:.3f}")

    if len(matched) < 20:
        print("\n  INSUFFICIENT DATA — cannot validate.")
        db.close()
        return

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: FOUR-CELL SPLIT vs KALSHI
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: FOUR-CELL SPLIT vs KALSHI BENCHMARK")
    print("=" * 70)

    # Build the 4 cells
    # Each matched entry has a -1.5 side and a +1.5 side (complements)
    cell_data = {
        "fav_neg15": [], "fav_pos15": [],
        "dog_pos15": [], "dog_neg15": [],
    }

    for m in matched:
        kp = m["kalshi_prob"]  # P(spread_team covers -1.5)

        for book, sp_data in m["soft_probs"].items():
            soft_neg = sp_data["fair_neg15"]  # soft's P(spread_team -1.5)
            soft_pos = sp_data["fair_pos15"]  # soft's P(other_team +1.5)
            vig = sp_data["vig"]

            # The -1.5 side of this market
            diff_neg = (soft_neg - kp) * 100  # >0 = soft overprices -1.5
            cell_data[m["cell_neg15"]].append({
                "book": book, "diff_pp": diff_neg,
                "won": m["covers"],  # did spread_team cover -1.5?
                "kalshi_fair": kp,
                "soft_fair": soft_neg,
                "vig": vig,
                "pregame_vol": m["pregame_vol"],
            })

            # The +1.5 side (complement)
            kalshi_pos = 1 - kp
            diff_pos = (soft_pos - kalshi_pos) * 100
            cell_data[m["cell_pos15"]].append({
                "book": book, "diff_pp": diff_pos,
                "won": not m["covers"],  # other team covers +1.5
                "kalshi_fair": kalshi_pos,
                "soft_fair": soft_pos,
                "vig": vig,
                "pregame_vol": m["pregame_vol"],
            })

    labels = {
        "fav_neg15": "Cell 1: FAV -1.5 (fav wins by 2+)",
        "fav_pos15": "Cell 2: FAV +1.5 (ultra-safe chalk)",
        "dog_pos15": "Cell 3: DOG +1.5 (take the points)",
        "dog_neg15": "Cell 4: DOG -1.5 (longshot)",
    }

    print("\n  A. BRIER BY CELL (who predicts better: Kalshi or soft books?)")
    print("  " + "-" * 60)

    for cell_name in ["fav_neg15", "fav_pos15", "dog_pos15", "dog_neg15"]:
        data = cell_data[cell_name]
        if not data:
            continue
        kalshi_brier = statistics.mean(
            [(d["kalshi_fair"] - (1 if d["won"] else 0)) ** 2 for d in data])
        soft_brier = statistics.mean(
            [(d["soft_fair"] - (1 if d["won"] else 0)) ** 2 for d in data])
        wr = sum(d["won"] for d in data) / len(data) * 100
        avg_k = statistics.mean([d["kalshi_fair"] for d in data]) * 100
        better = "KALSHI" if kalshi_brier < soft_brier else "SOFT"
        print(f"\n  {labels[cell_name]}: N={len(data)}")
        print(f"    WR={wr:.1f}%, Kalshi avg={avg_k:.1f}%")
        print(f"    Kalshi Brier={kalshi_brier:.6f}, Soft Brier={soft_brier:.6f}")
        print(f"    → {better} is sharper (gap={kalshi_brier-soft_brier:+.6f})")

    # B. THE KEY TEST: underdog edge vs Kalshi
    print("\n\n  B. UNDERDOG EDGE vs KALSHI")
    print("  When soft overprices underdog line vs Kalshi (>1pp), bet it — win?")
    print("  " + "-" * 60)

    for cell_name in ["fav_neg15", "fav_pos15", "dog_pos15", "dog_neg15"]:
        data = cell_data[cell_name]
        label = labels[cell_name]

        # Soft overprices this bet vs Kalshi (soft > kalshi by > 1pp)
        over = [d for d in data if d["diff_pp"] > 1.0]
        under = [d for d in data if d["diff_pp"] < -1.0]

        print(f"\n  {label}:")

        if len(over) >= 10:
            wr = sum(d["won"] for d in over) / len(over) * 100
            kexp = statistics.mean([d["kalshi_fair"] for d in over]) * 100
            sexp = statistics.mean([d["soft_fair"] for d in over]) * 100
            avg_diff = statistics.mean([d["diff_pp"] for d in over])
            avg_vig = statistics.mean([d["vig"] for d in over])
            edge_vs_kalshi = wr - kexp
            edge_net_vig = edge_vs_kalshi - avg_vig / 2  # approx vig cost

            print(f"    SOFT OVERPRICES (avg +{avg_diff:.1f}pp vs Kalshi):")
            print(f"      N={len(over)}, WR={wr:.1f}%, Kalshi exp={kexp:.1f}%")
            print(f"      Edge vs Kalshi: {edge_vs_kalshi:+.1f}pp")
            print(f"      Avg vig: {avg_vig:.1f}%, edge net vig: ~{edge_net_vig:+.1f}pp")

            # By book
            for book in ["draftkings", "fanduel", "betmgm"]:
                sub = [d for d in over if d["book"] == book]
                if sub:
                    sw = sum(d["won"] for d in sub) / len(sub) * 100
                    se = statistics.mean([d["kalshi_fair"] for d in sub]) * 100
                    print(f"        {book:15s}: WR={sw:.1f}% vs {se:.1f}% "
                          f"({sw-se:+.1f}pp, N={len(sub)})")
        else:
            print(f"    SOFT OVERPRICES: N={len(over)} (too few)")

        if len(under) >= 10:
            wr = sum(d["won"] for d in under) / len(under) * 100
            kexp = statistics.mean([d["kalshi_fair"] for d in under]) * 100
            edge = wr - kexp
            print(f"    SOFT UNDERPRICES (Kalshi higher): N={len(under)}, "
                  f"WR={wr:.1f}% vs {kexp:.1f}% ({edge:+.1f}pp)")

    # ═══════════════════════════════════════════════════════════════════
    # SPLIT-HALF STABILITY for underdog cells
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n  C. SPLIT-HALF STABILITY (underdog cells)")
    print("  " + "-" * 60)

    for cell_name in ["dog_pos15", "dog_neg15"]:
        data = cell_data[cell_name]
        over = [d for d in data if d["diff_pp"] > 1.0]
        if len(over) < 20:
            print(f"  {labels[cell_name]}: N={len(over)} too small")
            continue
        half = len(over) // 2
        h1, h2 = over[:half], over[half:]
        wr1 = sum(d["won"] for d in h1) / len(h1) * 100
        wr2 = sum(d["won"] for d in h2) / len(h2) * 100
        e1 = statistics.mean([d["kalshi_fair"] for d in h1]) * 100
        e2 = statistics.mean([d["kalshi_fair"] for d in h2]) * 100
        print(f"  {labels[cell_name]}:")
        print(f"    H1: {wr1-e1:+.1f}pp (N={len(h1)}), "
              f"H2: {wr2-e2:+.1f}pp (N={len(h2)}), "
              f"gap={abs((wr1-e1)-(wr2-e2)):.1f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # AXIS TEST: RUNG vs TEAM-ROLE vs KALSHI
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("AXIS TEST: RUNG vs TEAM-ROLE (against Kalshi)")
    print("=" * 70)

    for axis_label, cell_names in [
        ("ALL UNDERDOG (dog+1.5 + dog-1.5)", ["dog_pos15", "dog_neg15"]),
        ("ALL FAVORITE (fav+1.5 + fav-1.5)", ["fav_pos15", "fav_neg15"]),
        ("ALL +1.5 (dog+1.5 + fav+1.5)", ["dog_pos15", "fav_pos15"]),
        ("ALL -1.5 (fav-1.5 + dog-1.5)", ["fav_neg15", "dog_neg15"]),
    ]:
        pooled = []
        for cn in cell_names:
            pooled.extend(cell_data[cn])
        over = [d for d in pooled if d["diff_pp"] > 1.0]
        if len(over) >= 20:
            wr = sum(d["won"] for d in over) / len(over) * 100
            kexp = statistics.mean([d["kalshi_fair"] for d in over]) * 100
            edge = wr - kexp
            print(f"\n  {axis_label}: N={len(over)}")
            print(f"    WR={wr:.1f}%, Kalshi exp={kexp:.1f}%, edge={edge:+.1f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: VERDICT")
    print("=" * 70)

    # Compute per-cell edges
    cell_edges = {}
    for cn in ["fav_neg15", "fav_pos15", "dog_pos15", "dog_neg15"]:
        over = [d for d in cell_data[cn] if d["diff_pp"] > 1.0]
        if len(over) >= 20:
            wr = sum(d["won"] for d in over) / len(over) * 100
            kexp = statistics.mean([d["kalshi_fair"] for d in over]) * 100
            cell_edges[cn] = (wr - kexp, len(over))
        else:
            cell_edges[cn] = (None, len(over))

    # Underdog pooled
    dog_over = [d for d in cell_data["dog_pos15"] + cell_data["dog_neg15"]
                if d["diff_pp"] > 1.0]
    fav_over = [d for d in cell_data["fav_pos15"] + cell_data["fav_neg15"]
                if d["diff_pp"] > 1.0]

    dog_edge = None
    if len(dog_over) >= 20:
        dwr = sum(d["won"] for d in dog_over) / len(dog_over) * 100
        dexp = statistics.mean([d["kalshi_fair"] for d in dog_over]) * 100
        dog_edge = dwr - dexp

    fav_edge = None
    if len(fav_over) >= 20:
        fwr = sum(d["won"] for d in fav_over) / len(fav_over) * 100
        fexp = statistics.mean([d["kalshi_fair"] for d in fav_over]) * 100
        fav_edge = fwr - fexp

    coverage = "SUFFICIENT" if liquid >= 100 else "MARGINAL" if liquid >= 30 else "THIN"

    print(f"""
  COVERAGE: {coverage} ({liquid} liquid markets, {matched_n} total)

  PER-CELL EDGE vs KALSHI (soft overprices → bet it):
    Cell 1 FAV -1.5:  {f'{cell_edges["fav_neg15"][0]:+.1f}pp (N={cell_edges["fav_neg15"][1]})' if cell_edges["fav_neg15"][0] is not None else f'N={cell_edges["fav_neg15"][1]} (too few)'}
    Cell 2 FAV +1.5:  {f'{cell_edges["fav_pos15"][0]:+.1f}pp (N={cell_edges["fav_pos15"][1]})' if cell_edges["fav_pos15"][0] is not None else f'N={cell_edges["fav_pos15"][1]} (too few)'}
    Cell 3 DOG +1.5:  {f'{cell_edges["dog_pos15"][0]:+.1f}pp (N={cell_edges["dog_pos15"][1]})' if cell_edges["dog_pos15"][0] is not None else f'N={cell_edges["dog_pos15"][1]} (too few)'}
    Cell 4 DOG -1.5:  {f'{cell_edges["dog_neg15"][0]:+.1f}pp (N={cell_edges["dog_neg15"][1]})' if cell_edges["dog_neg15"][0] is not None else f'N={cell_edges["dog_neg15"][1]} (too few)'}

  UNDERDOG pooled: {f'{dog_edge:+.1f}pp (N={len(dog_over)})' if dog_edge is not None else 'insufficient N'}
  FAVORITE pooled: {f'{fav_edge:+.1f}pp (N={len(fav_over)})' if fav_edge is not None else 'insufficient N'}

  VERDICT:
  {'REAL EDGE — underdog edge survives Kalshi (sharp benchmark)' if dog_edge and dog_edge > 2 and coverage != 'THIN' else 'SOFT-PROXY ARTIFACT — edge does not survive sharp benchmark' if dog_edge is not None and dog_edge < -2 else 'INCONCLUSIVE — ' + ('thin coverage' if coverage == 'THIN' else f'edge {dog_edge:+.1f}pp within noise band (±2pp)') if dog_edge is not None else 'INSUFFICIENT DATA'}

  {'STEP 3 BONUS: Kalshi is an exchange (no limiting). If edge > vig, Kalshi itself is a venue.' if dog_edge and dog_edge > 3 else ''}
""")

    db.close()


if __name__ == "__main__":
    main()
