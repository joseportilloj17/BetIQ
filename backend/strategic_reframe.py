#!/usr/bin/env python3
"""
strategic_reframe.py — Test "run faster / different target" instead of "predict better."

The prior conclusion ("market is efficient") was over-generalized. The model hit
parity vs the CLOSE — the hardest target. This script tests the axes where the
market is actually beatable:

  1. BEAT THE OPENER: does the model beat the entry line (soft window)?
  2. OPEN->CLOSE CLV: does betting early capture positive line movement?
  3. ALT vs MAIN: are alt-lines systematically softer (more model edge)?
  4. BEST-PRICE EXECUTION: how much does cross-book best-price add?
  5. NO-VIG CLV by ARM: which shadow arms beat the close after devigging?

Data sources:
  - mock_bet_legs: open_odds, close_odds, clv_cents, devig probs (63K settled legs)
  - bet_clv: no-vig CLV with per-book closing prices (20K legs, 6 books)
  - alt_lines (historical.db): 728K rows across 6 bookmakers
"""
import json
import os
import sqlite3
from collections import defaultdict

import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bets.db")
HIST_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "historical.db")


def american_to_implied(american):
    """Convert American odds to implied probability."""
    if american is None:
        return None
    if american >= 100:
        return 100.0 / (american + 100)
    elif american <= -100:
        return abs(american) / (abs(american) + 100)
    return None


def american_to_decimal(american):
    """Convert American odds to decimal odds."""
    if american is None:
        return None
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    return None


# ---------------------------------------------------------------------------
# Load settled legs with direct open/close columns
# ---------------------------------------------------------------------------
def load_legs(con):
    """Load settled single-leg bets with model prob, open/close American odds, CLV."""
    cur = con.execute("""
        SELECT mb.id AS bet_id, mb.odds AS parlay_odds, mb.status, mb.game_date,
               mb.source, mb.notes,
               mbl.win_prob, mbl.predicted_win_prob,
               mbl.open_odds, mbl.close_odds, mbl.clv_cents,
               mbl.devig_entry_prob, mbl.devig_close_prob,
               mbl.market_type, mbl.fixture_id, mbl.is_alt_line,
               mbl.leg_result, mbl.description
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND mb.legs = 1
          AND mbl.fixture_id IS NOT NULL
    """)

    legs = []
    for row in cur:
        # Normalize model probability to 0-1 scale
        wp = row["win_prob"]
        pwp = row["predicted_win_prob"]
        if pwp is not None and 0 < pwp < 1:
            model_prob = pwp
        elif wp is not None:
            model_prob = wp / 100.0 if wp > 1.0 else wp
        else:
            model_prob = None

        legs.append({
            "bet_id": row["bet_id"],
            "parlay_odds": row["parlay_odds"],
            "won": row["leg_result"] == "WIN",
            "game_date": row["game_date"],
            "source": row["source"],
            "model_prob": model_prob,
            "open_odds": row["open_odds"],      # American
            "close_odds": row["close_odds"],    # American
            "clv_cents": row["clv_cents"],
            "devig_entry": row["devig_entry_prob"],
            "devig_close": row["devig_close_prob"],
            "market_type": row["market_type"],
            "fixture_id": row["fixture_id"],
            "is_alt": bool(row["is_alt_line"]),
        })

    print(f"  {len(legs):,} settled single-leg legs loaded")
    print(f"  With open_odds: {sum(1 for l in legs if l['open_odds'] is not None):,}")
    print(f"  With close_odds: {sum(1 for l in legs if l['close_odds'] is not None):,}")
    print(f"  With model_prob: {sum(1 for l in legs if l['model_prob'] is not None):,}")
    return legs


# ---------------------------------------------------------------------------
# Analysis 1: BEAT THE OPENER
# ---------------------------------------------------------------------------
def analysis_opener(legs):
    """Does the model beat the entry price even where it can't beat the close?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: MODEL EDGE vs ENTRY PRICE vs CLOSING PRICE")
    print("=" * 70)

    edge_vs_entry = []
    edge_vs_close = []
    beat_entry_not_close = 0
    beat_close_not_entry = 0
    beat_both = 0
    beat_neither = 0
    n = 0

    for leg in legs:
        if leg["model_prob"] is None or leg["open_odds"] is None or leg["close_odds"] is None:
            continue

        entry_imp = american_to_implied(leg["open_odds"])
        close_imp = american_to_implied(leg["close_odds"])
        if entry_imp is None or close_imp is None:
            continue

        model_p = leg["model_prob"]
        n += 1

        e_entry = model_p - entry_imp
        e_close = model_p - close_imp

        edge_vs_entry.append(e_entry)
        edge_vs_close.append(e_close)

        be = e_entry > 0
        bc = e_close > 0
        if be and not bc:
            beat_entry_not_close += 1
        elif bc and not be:
            beat_close_not_entry += 1
        elif be and bc:
            beat_both += 1
        else:
            beat_neither += 1

    if not edge_vs_entry:
        print("  No data — skipping")
        return

    eo = np.array(edge_vs_entry)
    ec = np.array(edge_vs_close)

    print(f"\n  Legs with entry + close odds + model prob: {n:,}")
    print(f"\n  Model edge vs ENTRY:  mean = {eo.mean()*100:+.2f}pp, median = {np.median(eo)*100:+.2f}pp")
    print(f"  Model edge vs CLOSE:  mean = {ec.mean()*100:+.2f}pp, median = {np.median(ec)*100:+.2f}pp")
    print(f"  Difference (entry - close): {(eo.mean() - ec.mean())*100:+.2f}pp")

    print(f"\n  Beats ENTRY only (not close): {beat_entry_not_close:>6,d} ({beat_entry_not_close/n*100:.1f}%)")
    print(f"  Beats CLOSE only (not entry): {beat_close_not_entry:>6,d} ({beat_close_not_entry/n*100:.1f}%)")
    print(f"  Beats BOTH:                   {beat_both:>6,d} ({beat_both/n*100:.1f}%)")
    print(f"  Beats NEITHER:                {beat_neither:>6,d} ({beat_neither/n*100:.1f}%)")

    # PnL simulation at entry vs close price
    print(f"\n  -- PnL simulation (flat $10) --")
    for label, odds_key in [("ENTRY", "open_odds"), ("CLOSE", "close_odds")]:
        stake = 10.0
        pnl = 0.0
        cnt = 0
        for leg in legs:
            am = leg[odds_key]
            if am is None or leg["model_prob"] is None:
                continue
            dec = american_to_decimal(am)
            if dec is None or dec <= 1.0:
                continue
            cnt += 1
            if leg["won"]:
                pnl += stake * (dec - 1)
            else:
                pnl -= stake
        if cnt:
            print(f"  At {label} price: N={cnt:,}, PnL=${pnl:+,.0f}, ROI={pnl/(cnt*stake)*100:+.1f}%")

    # Break by alt vs main
    print(f"\n  -- Entry vs close edge by market type --")
    for label, filt in [("MAIN", lambda l: not l["is_alt"]), ("ALT", lambda l: l["is_alt"])]:
        ee, ec2 = [], []
        for leg in legs:
            if not filt(leg) or leg["model_prob"] is None or leg["open_odds"] is None or leg["close_odds"] is None:
                continue
            ei = american_to_implied(leg["open_odds"])
            ci = american_to_implied(leg["close_odds"])
            if ei and ci:
                ee.append(leg["model_prob"] - ei)
                ec2.append(leg["model_prob"] - ci)
        if ee:
            ee, ec2 = np.array(ee), np.array(ec2)
            print(f"  {label:>5s}: N={len(ee):,}, edge_vs_entry={ee.mean()*100:+.2f}pp, "
                  f"edge_vs_close={ec2.mean()*100:+.2f}pp, gap={((ee-ec2).mean())*100:+.2f}pp")


# ---------------------------------------------------------------------------
# Analysis 2: OPEN->CLOSE CLV (direct from clv_cents)
# ---------------------------------------------------------------------------
def analysis_clv(legs):
    """Does betting early capture positive line movement?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: OPEN->CLOSE CLV (LINE MOVEMENT IN OUR FAVOR)")
    print("=" * 70)

    # clv_cents = close_odds - open_odds in American cents
    # Positive clv_cents on underdog side (+odds) = close is higher = we got worse price
    # On favorite side (-odds), sign can be confusing.
    # Better: compute implied prob shift directly.

    clv_implied = []  # close_implied - entry_implied (positive = line moved our way)
    clv_by_segment = defaultdict(list)

    for leg in legs:
        if leg["open_odds"] is None or leg["close_odds"] is None:
            continue
        entry_imp = american_to_implied(leg["open_odds"])
        close_imp = american_to_implied(leg["close_odds"])
        if entry_imp is None or close_imp is None:
            continue

        # Positive CLV = close implied > entry implied = market moved toward our side
        clv = close_imp - entry_imp
        clv_implied.append(clv)

        seg = "alt" if leg["is_alt"] else "main"
        clv_by_segment[seg].append(clv)

    if not clv_implied:
        print("  No CLV data — skipping")
        return

    c = np.array(clv_implied)
    pos = sum(1 for x in c if x > 0.001)
    neg = sum(1 for x in c if x < -0.001)
    flat = len(c) - pos - neg

    print(f"\n  Legs with open+close odds: {len(c):,}")
    print(f"  Mean CLV (implied shift): {c.mean()*100:+.3f}pp")
    print(f"  Median CLV: {np.median(c)*100:+.3f}pp")
    print(f"  Line moved our way (>0.1pp):   {pos:,} ({pos/len(c)*100:.1f}%)")
    print(f"  Line moved against (<-0.1pp):  {neg:,} ({neg/len(c)*100:.1f}%)")
    print(f"  Flat (within 0.1pp):           {flat:,} ({flat/len(c)*100:.1f}%)")

    print(f"\n  -- CLV by market type --")
    for seg in ["main", "alt"]:
        if seg not in clv_by_segment:
            continue
        v = np.array(clv_by_segment[seg])
        pos_pct = sum(1 for x in v if x > 0.001) / len(v) * 100
        print(f"  {seg:>5s}: N={len(v):,}, mean={v.mean()*100:+.3f}pp, "
              f"median={np.median(v)*100:+.3f}pp, moved_our_way={pos_pct:.0f}%")

    # CLV vs win rate: do high-CLV legs win more?
    print(f"\n  -- CLV quintiles vs win rate --")
    clv_with_result = [(c_val, leg["won"]) for c_val, leg in zip(clv_implied,
        [l for l in legs if l["open_odds"] is not None and l["close_odds"] is not None
         and american_to_implied(l["open_odds"]) is not None and american_to_implied(l["close_odds"]) is not None])]

    clv_with_result.sort(key=lambda x: x[0])
    q_size = len(clv_with_result) // 5
    for i in range(5):
        chunk = clv_with_result[i * q_size : (i + 1) * q_size] if i < 4 else clv_with_result[i * q_size:]
        if not chunk:
            continue
        clv_vals = [x[0] for x in chunk]
        wr = sum(1 for x in chunk if x[1]) / len(chunk) * 100
        print(f"  Q{i+1} (CLV {min(clv_vals)*100:+.1f} to {max(clv_vals)*100:+.1f}pp): "
              f"N={len(chunk):,}, WR={wr:.1f}%")


# ---------------------------------------------------------------------------
# Analysis 3: ALT vs MAIN softness
# ---------------------------------------------------------------------------
def analysis_alt_vs_main(legs):
    """Are alt-lines systematically softer?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: ALT-LINES vs MAIN LINES -- SOFTNESS COMPARISON")
    print("=" * 70)

    segments = {"main": [], "alt": []}
    for leg in legs:
        if leg["model_prob"] is None or leg["open_odds"] is None:
            continue
        seg = "alt" if leg["is_alt"] else "main"
        entry_dec = american_to_decimal(leg["open_odds"])
        entry_imp = american_to_implied(leg["open_odds"])
        if entry_dec and entry_imp:
            segments[seg].append({
                "edge": leg["model_prob"] - entry_imp,
                "won": leg["won"],
                "dec_odds": entry_dec,
                "clv_cents": leg["clv_cents"],
            })

    for seg_name in ["main", "alt"]:
        seg = segments[seg_name]
        if not seg:
            continue

        edges = np.array([s["edge"] for s in seg])
        wins = sum(1 for s in seg if s["won"])
        n = len(seg)
        wr = wins / n * 100

        avg_odds = np.mean([s["dec_odds"] for s in seg])
        be = 1 / avg_odds * 100

        stake = 10.0
        pnl = sum(stake * (s["dec_odds"] - 1) if s["won"] else -stake for s in seg)
        roi = pnl / (n * stake) * 100

        pos_edge = sum(1 for e in edges if e > 0)

        # CLV for those with data
        clv_vals = [s["clv_cents"] for s in seg if s["clv_cents"] is not None]
        clv_str = f"avg_clv={np.mean(clv_vals):+.1f}c" if clv_vals else "no CLV"

        print(f"\n  -- {seg_name.upper()} LINES --")
        print(f"  N={n:,}, WR={wr:.1f}%, BE={be:.1f}%, ROI={roi:+.1f}%")
        print(f"  Mean model edge vs entry: {edges.mean()*100:+.2f}pp")
        print(f"  Positive model edge: {pos_edge}/{n} ({pos_edge/n*100:.0f}%)")
        print(f"  PnL: ${pnl:+,.0f}, avg_odds={avg_odds:.3f}")
        print(f"  CLV (American cents): {clv_str} (N={len(clv_vals):,})")


# ---------------------------------------------------------------------------
# Analysis 4: BEST-PRICE EXECUTION (using bet_clv close_per_book)
# ---------------------------------------------------------------------------
def analysis_best_price(con):
    """Quantify the gain from best-price execution using actual multi-book data."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: BEST-PRICE EXECUTION ACROSS BOOKS")
    print("=" * 70)

    cur = con.execute("""
        SELECT leg_id, arm, entry_decimal, close_decimal, close_per_book,
               leg_result, close_fair_prob, clv_novig_pp, beat_close
        FROM bet_clv
        WHERE close_per_book IS NOT NULL
    """)

    total_fd_pnl = 0.0
    total_best_pnl = 0.0
    gains = []
    gains_by_book = defaultdict(list)
    n = 0
    book_win_counts = defaultdict(int)
    stake = 10.0

    for row in cur:
        try:
            cpb = json.loads(row["close_per_book"])
        except (json.JSONDecodeError, TypeError):
            continue

        if len(cpb) < 2:
            continue

        fd_price = cpb.get("fanduel")
        if fd_price is None:
            continue

        best_price = max(cpb.values())
        best_book = max(cpb, key=cpb.get)
        gain = best_price - fd_price

        won = row["leg_result"] == "WIN"
        n += 1

        # PnL at FanDuel
        if won:
            total_fd_pnl += stake * (fd_price - 1)
        else:
            total_fd_pnl -= stake

        # PnL at best price
        if won:
            total_best_pnl += stake * (best_price - 1)
        else:
            total_best_pnl -= stake

        gains.append(gain)
        if gain > 0:
            book_win_counts[best_book] += 1

        for book, price in cpb.items():
            gains_by_book[book].append(price)

    if not gains:
        print("  No multi-book data — skipping")
        return

    g = np.array(gains)
    print(f"\n  Legs with multi-book closing prices: {n:,}")
    print(f"  Books available: {list(gains_by_book.keys())}")
    print(f"\n  Mean gain (best - FanDuel): {g.mean():.4f} decimal ({g.mean()*100:.2f}%)")
    print(f"  Median gain: {np.median(g):.4f}")
    print(f"  Legs where best != FanDuel: {sum(1 for x in g if x > 0.001):,} ({sum(1 for x in g if x > 0.001)/n*100:.0f}%)")

    print(f"\n  -- PnL comparison (flat ${stake:.0f}) --")
    print(f"  FanDuel only:   PnL=${total_fd_pnl:+,.0f}, ROI={total_fd_pnl/(n*stake)*100:+.1f}%")
    print(f"  Best available: PnL=${total_best_pnl:+,.0f}, ROI={total_best_pnl/(n*stake)*100:+.1f}%")
    print(f"  Execution gain: ${total_best_pnl - total_fd_pnl:+,.0f} ({(total_best_pnl - total_fd_pnl)/(n*stake)*100:+.2f}% ROI improvement)")

    # Which book is cheapest most often?
    print(f"\n  -- Best-price book frequency --")
    for book, count in sorted(book_win_counts.items(), key=lambda x: -x[1]):
        print(f"  {book:<20s}: best price {count:,} times ({count/n*100:.0f}%)")

    # Average closing price by book
    print(f"\n  -- Average closing decimal odds by book --")
    for book in sorted(gains_by_book.keys()):
        prices = gains_by_book[book]
        print(f"  {book:<20s}: avg={np.mean(prices):.4f}, N={len(prices):,}")


# ---------------------------------------------------------------------------
# Analysis 5: No-vig CLV by arm (from bet_clv)
# ---------------------------------------------------------------------------
def analysis_novig_clv_by_arm(con):
    """Which shadow arms beat the close after devigging?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: NO-VIG CLV BY SHADOW ARM (bet_clv table)")
    print("=" * 70)

    cur = con.execute("""
        SELECT arm, COUNT(*) as n,
               AVG(clv_novig_pp) as avg_novig,
               AVG(clv_raw_pct) as avg_raw,
               AVG(CAST(beat_close AS FLOAT)) as beat_pct,
               AVG(entry_decimal) as avg_entry,
               AVG(close_decimal) as avg_close
        FROM bet_clv
        WHERE arm IS NOT NULL
        GROUP BY arm
        ORDER BY COUNT(*) DESC
    """)

    print(f"\n  {'Arm':<40s} {'N':>6s} {'NoVig':>8s} {'Raw%':>7s} {'BeatCl':>7s} {'AvgEnt':>7s} {'AvgCl':>7s}")
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for row in cur:
        print(f"  {row['arm']:<40s} {row['n']:>6,d} {row['avg_novig']:>+7.2f}pp "
              f"{row['avg_raw']:>+6.2f}% {row['beat_pct']*100:>6.1f}% "
              f"{row['avg_entry']:>7.3f} {row['avg_close']:>7.3f}")


# ---------------------------------------------------------------------------
# Analysis 6: Edge by source
# ---------------------------------------------------------------------------
def analysis_by_source(legs):
    """Break down edge and ROI by source."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: EDGE BY SOURCE / SHADOW ARM")
    print("=" * 70)

    by_source = defaultdict(list)
    for leg in legs:
        if leg["model_prob"] is None:
            continue
        by_source[leg["source"]].append(leg)

    print(f"\n  {'Source':<22s} {'N':>6s} {'WR%':>6s} {'BE%':>6s} {'ROI%':>7s} {'PnL':>9s} {'EdgeVsEntry':>12s}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*12}")

    for source in sorted(by_source.keys(), key=lambda s: -len(by_source[s])):
        seg = by_source[source]
        n = len(seg)
        if n < 10:
            continue

        wins = sum(1 for s in seg if s["won"])
        wr = wins / n * 100

        # Use entry (open_odds) for PnL
        pnl = 0.0
        cnt_odds = 0
        edge_vals = []
        for s in seg:
            if s["open_odds"] is not None:
                dec = american_to_decimal(s["open_odds"])
                imp = american_to_implied(s["open_odds"])
                if dec and imp:
                    cnt_odds += 1
                    pnl += 10 * (dec - 1) if s["won"] else -10
                    edge_vals.append(s["model_prob"] - imp)

        be_str = ""
        roi_str = ""
        pnl_str = ""
        edge_str = ""
        if cnt_odds > 0:
            avg_dec = pnl / cnt_odds  # just for ROI
            roi = pnl / (cnt_odds * 10) * 100
            roi_str = f"{roi:>+6.1f}%"
            pnl_str = f"${pnl:>+8,.0f}"
        if edge_vals:
            edge_str = f"{np.mean(edge_vals)*100:>+10.2f}pp"

        print(f"  {source:<22s} {n:>6,d} {wr:>5.1f}% {'':>6s} {roi_str:>7s} {pnl_str:>9s} {edge_str:>12s}")


# ---------------------------------------------------------------------------
# Analysis 7: Cross-book spread comparison (alt_lines)
# ---------------------------------------------------------------------------
def analysis_cross_book_spreads():
    """Compare odds across books for the same spread line."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: CROSS-BOOK SPREAD COMPARISON (alt_lines)")
    print("=" * 70)

    if not os.path.exists(HIST_DB):
        print("  historical.db not found — skipping")
        return

    hist = sqlite3.connect(HIST_DB)
    hist.row_factory = sqlite3.Row

    # For MLB alt_lines, compare same (event_id, team, line) across bookmakers
    cur = hist.execute("""
        SELECT event_id, team, line, bookmaker, odds, fetched_at, game_date_ct
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_spreads'
          AND odds > 1.0
        ORDER BY event_id, team, line, bookmaker
    """)

    # Group by (event_id, team, line) — use earliest fetch per bookmaker
    groups = defaultdict(dict)
    for row in cur:
        key = (row["event_id"], row["team"], row["line"])
        book = row["bookmaker"]
        if book not in groups[key]:
            groups[key][book] = row["odds"]

    # For each group with 3+ books, compute spread
    spreads = []
    fd_vs_best = []
    for key, books in groups.items():
        if len(books) < 3:
            continue
        prices = list(books.values())
        best = max(prices)
        worst = min(prices)
        spread = best - worst
        spreads.append(spread)

        if "fanduel" in books:
            fd = books["fanduel"]
            fd_vs_best.append(best - fd)

    if not spreads:
        print("  No multi-book MLB alt-spread data — skipping")
        hist.close()
        return

    s = np.array(spreads)
    print(f"\n  Unique (event, team, line) groups with 3+ books: {len(s):,}")
    print(f"  Mean price spread (best - worst): {s.mean():.4f} decimal")
    print(f"  Median spread: {np.median(s):.4f}")
    print(f"  >5% spread: {sum(1 for x in s if x > 0.05)/len(s)*100:.0f}%")
    print(f"  >10% spread: {sum(1 for x in s if x > 0.10)/len(s)*100:.0f}%")

    if fd_vs_best:
        fb = np.array(fd_vs_best)
        print(f"\n  FanDuel vs best available:")
        print(f"  Groups with FanDuel: {len(fb):,}")
        print(f"  Mean gain (best - FD): {fb.mean():.4f} ({fb.mean()*100:.2f}%)")
        print(f"  FD is best price: {sum(1 for x in fb if x < 0.001)/len(fb)*100:.0f}%")
        print(f"  FD worse by >5%: {sum(1 for x in fb if x > 0.05)/len(fb)*100:.0f}%")

    hist.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("STRATEGIC REFRAME: Run Faster / Different Target / Better Execution")
    print("=" * 70)
    print(f"Database: {DB_PATH}")
    print(f"Historical: {HIST_DB}\n")

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    legs = load_legs(con)
    if not legs:
        print("No settled legs found — exiting")
        return

    analysis_opener(legs)
    analysis_clv(legs)
    analysis_alt_vs_main(legs)
    analysis_best_price(con)
    analysis_novig_clv_by_arm(con)
    analysis_by_source(legs)
    analysis_cross_book_spreads()

    # Summary
    print("\n" + "=" * 70)
    print("STRATEGIC SUMMARY")
    print("=" * 70)
    print("""
  The prior conclusion was: "GBC on public data hits parity vs the close,
  therefore the market is efficient." This OVER-GENERALIZED.

  What was actually proven: the model can't beat the CLOSE on FanDuel
  alt-spreads -- the sharpest, hardest target on one book.

  What wasn't tested (until now):
    1. ENTRY vs CLOSE: is there more model edge at the entry/opener?
    2. CLV: do our picks capture favorable line movement?
    3. ALT vs MAIN: which market type is softer to exploit?
    4. BEST-PRICE: how much does cross-book execution add? (MEASURED)
    5. BY ARM: which construction methods beat the close after devigging?

  The honest frame: compete on SPEED (bet the opener), TARGET (softer
  markets), and EXECUTION (best price across books) -- not on prediction
  from public data.
    """)

    con.close()


if __name__ == "__main__":
    main()
