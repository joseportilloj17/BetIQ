#!/usr/bin/env python3
"""
h_0053 BLIND TEST: Top-decile model-edge threshold selection
============================================================
Protocol: Train threshold on pre-July data, apply blind to Jul-Aug.
Split-half validation on blind period (H1=July, H2=August).

Tests whether D1 edge filter generates real profit OOS,
and whether the CLV paradox (negative CLV yet profitable) holds.
"""

import sqlite3
import numpy as np
from collections import defaultdict

DB = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"

TRAIN_CUTOFF = "2026-07-01"
SPLIT_HALF_CUTOFF = "2026-08-01"

def fetch_data():
    """Fetch all settled MLB legs with model + market probabilities."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT
            l.id,
            l.predicted_win_prob,
            l.devig_entry_prob,
            l.devig_close_prob,
            l.leg_result,
            l.market_type,
            l.spread_value,
            l.is_favorite_spread,
            l.fixture_id,
            b.game_date
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE l.sport = 'MLB'
          AND l.leg_result IN ('WIN', 'LOSS')
          AND l.predicted_win_prob IS NOT NULL
          AND l.devig_entry_prob IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def normalize_prob(val):
    """Normalize predicted_win_prob to 0-1 scale."""
    if val is None:
        return None
    return val / 100.0 if val > 1.0 else val


def compute_metrics(legs, label=""):
    """Compute full metric set for a group of legs."""
    if not legs:
        return None

    n_raw = len(legs)
    fixture_ids = set(r["fixture_id"] for r in legs)
    n_deduped = len(fixture_ids)

    wins = sum(1 for r in legs if r["leg_result"] == "WIN")
    wr = wins / n_raw

    avg_entry = np.mean([r["devig_entry_prob"] for r in legs])
    be = avg_entry  # breakeven = average devig entry prob

    edge_pp = (wr - be) * 100

    # ROI at fair odds: profit if bet at devig_entry_prob implied odds
    # Fair odds = 1 / devig_entry_prob
    # Profit per bet = (result * fair_odds) - 1
    fair_profits = []
    for r in legs:
        fair_odds = 1.0 / r["devig_entry_prob"] if r["devig_entry_prob"] > 0 else 0
        if r["leg_result"] == "WIN":
            fair_profits.append(fair_odds - 1.0)
        else:
            fair_profits.append(-1.0)
    roi_fair = np.mean(fair_profits) * 100 if fair_profits else 0

    # ROI at 5% vig: odds = (1 / devig_entry_prob) * 0.95
    vig_profits = []
    for r in legs:
        vig_odds = (1.0 / r["devig_entry_prob"] * 0.95) if r["devig_entry_prob"] > 0 else 0
        if r["leg_result"] == "WIN":
            vig_profits.append(vig_odds - 1.0)
        else:
            vig_profits.append(-1.0)
    roi_vig = np.mean(vig_profits) * 100 if vig_profits else 0

    # CLV: entry - close (positive = we got a better price)
    clv_legs = [r for r in legs if r["devig_close_prob"] is not None]
    if clv_legs:
        clv_vals = [r["devig_entry_prob"] - r["devig_close_prob"] for r in clv_legs]
        avg_clv = np.mean(clv_vals) * 100  # in percentage points
        clv_n = len(clv_legs)
        # Beat-close: entry prob < close prob means we got a better line
        beat_close = sum(1 for v in clv_vals if v < 0) / len(clv_vals) * 100
    else:
        avg_clv = None
        clv_n = 0
        beat_close = None

    # Model edge stats
    edges = [normalize_prob(r["predicted_win_prob"]) - r["devig_entry_prob"] for r in legs]
    avg_model_edge = np.mean(edges) * 100

    return {
        "label": label,
        "n_raw": n_raw,
        "n_deduped": n_deduped,
        "wins": wins,
        "wr": wr,
        "be": be,
        "edge_pp": edge_pp,
        "roi_fair": roi_fair,
        "roi_vig": roi_vig,
        "avg_clv_pp": avg_clv,
        "clv_n": clv_n,
        "beat_close_pct": beat_close,
        "avg_model_edge_pp": avg_model_edge,
    }


def print_metrics(m, indent=0):
    """Pretty print metrics dict."""
    if m is None:
        print(f"{'  ' * indent}  (no data)")
        return
    pfx = "  " * indent
    print(f"{pfx}  N legs (raw):       {m['n_raw']}")
    print(f"{pfx}  N games (deduped):  {m['n_deduped']}")
    print(f"{pfx}  Wins:               {m['wins']}")
    print(f"{pfx}  Win Rate:           {m['wr']:.4f}  ({m['wr']*100:.2f}%)")
    print(f"{pfx}  Breakeven (avg):    {m['be']:.4f}  ({m['be']*100:.2f}%)")
    print(f"{pfx}  Edge (WR - BE):     {m['edge_pp']:+.2f} pp")
    print(f"{pfx}  ROI (fair odds):    {m['roi_fair']:+.2f}%")
    print(f"{pfx}  ROI (5% vig):       {m['roi_vig']:+.2f}%")
    print(f"{pfx}  Avg model edge:     {m['avg_model_edge_pp']:+.2f} pp")
    if m["avg_clv_pp"] is not None:
        print(f"{pfx}  CLV:                {m['avg_clv_pp']:+.3f} pp  (N={m['clv_n']})")
        print(f"{pfx}  Beat-close %:       {m['beat_close_pct']:.1f}%")
    else:
        print(f"{pfx}  CLV:                N/A (no close probs)")


def main():
    print("=" * 80)
    print("h_0053 BLIND TEST: Top-Decile Model-Edge Threshold Selection")
    print("=" * 80)

    rows = fetch_data()
    print(f"\nTotal settled MLB legs with model + market probs: {len(rows)}")

    # ─── Normalize and split ───
    train = [r for r in rows if r["game_date"] < TRAIN_CUTOFF]
    blind = [r for r in rows if r["game_date"] >= TRAIN_CUTOFF]

    print(f"\nTRAIN period: < {TRAIN_CUTOFF}  (N={len(train)}, "
          f"games={len(set(r['fixture_id'] for r in train))})")
    print(f"BLIND period: >= {TRAIN_CUTOFF} (N={len(blind)}, "
          f"games={len(set(r['fixture_id'] for r in blind))})")

    # ─── STEP 1: Find D1 threshold on TRAIN ───
    print("\n" + "=" * 80)
    print("STEP 1: TRAIN — Compute D1 (90th percentile) threshold")
    print("=" * 80)

    train_edges = [normalize_prob(r["predicted_win_prob"]) - r["devig_entry_prob"]
                   for r in train]
    threshold = np.percentile(train_edges, 90)
    print(f"\n  Train edge distribution:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(train_edges, pct)
        print(f"    P{pct:02d}: {val*100:+.2f} pp")
    print(f"\n  >>> D1 THRESHOLD (P90): {threshold*100:+.4f} pp <<<")
    print(f"  (Only carry this single number forward)")

    # Train D1 metrics (for reference, but NOT used for decisions)
    train_d1 = [r for r, e in zip(train, train_edges) if e >= threshold]
    train_rest = [r for r, e in zip(train, train_edges) if e < threshold]
    print(f"\n  Train D1: {len(train_d1)} legs")
    print_metrics(compute_metrics(train_d1, "Train D1"))
    print(f"\n  Train Rest (D2-D10): {len(train_rest)} legs")
    print_metrics(compute_metrics(train_rest, "Train Rest"))

    # ─── STEP 2: Apply to BLIND ───
    print("\n" + "=" * 80)
    print("STEP 2: BLIND — Apply train-derived threshold")
    print("=" * 80)

    blind_edges = [normalize_prob(r["predicted_win_prob"]) - r["devig_entry_prob"]
                   for r in blind]
    blind_d1 = [r for r, e in zip(blind, blind_edges) if e >= threshold]
    blind_rest = [r for r, e in zip(blind, blind_edges) if e < threshold]

    blind_d1_pct = len(blind_d1) / len(blind) * 100 if blind else 0
    print(f"\n  Blind D1: {len(blind_d1)} legs ({blind_d1_pct:.1f}% of blind)")
    d1_metrics = compute_metrics(blind_d1, "Blind D1")
    print_metrics(d1_metrics)

    print(f"\n  Blind Rest (below threshold): {len(blind_rest)} legs")
    rest_metrics = compute_metrics(blind_rest, "Blind Rest")
    print_metrics(rest_metrics)

    if d1_metrics and rest_metrics:
        print(f"\n  >>> D1 vs Rest DELTA:")
        print(f"      Edge:  {d1_metrics['edge_pp'] - rest_metrics['edge_pp']:+.2f} pp")
        print(f"      ROI fair: {d1_metrics['roi_fair'] - rest_metrics['roi_fair']:+.2f}%")
        print(f"      ROI vig:  {d1_metrics['roi_vig'] - rest_metrics['roi_vig']:+.2f}%")

    # ─── STEP 3: SPLIT-HALF on blind ───
    print("\n" + "=" * 80)
    print("STEP 3: SPLIT-HALF — Blind H1 (July) vs H2 (August)")
    print("=" * 80)

    blind_h1 = [r for r in blind if r["game_date"] < SPLIT_HALF_CUTOFF]
    blind_h2 = [r for r in blind if r["game_date"] >= SPLIT_HALF_CUTOFF]

    for half_label, half_data in [("H1 (July)", blind_h1), ("H2 (August)", blind_h2)]:
        half_edges = [normalize_prob(r["predicted_win_prob"]) - r["devig_entry_prob"]
                      for r in half_data]
        h_d1 = [r for r, e in zip(half_data, half_edges) if e >= threshold]
        h_rest = [r for r, e in zip(half_data, half_edges) if e < threshold]

        print(f"\n  {half_label}:  total={len(half_data)}, D1={len(h_d1)}, Rest={len(h_rest)}")
        print(f"  D1:")
        print_metrics(compute_metrics(h_d1, f"D1 {half_label}"), indent=1)
        print(f"  Rest:")
        print_metrics(compute_metrics(h_rest, f"Rest {half_label}"), indent=1)

    # ─── STEP 4: SPREAD UNDERDOG DEEP DIVE ───
    print("\n" + "=" * 80)
    print("STEP 4: SPREAD UNDERDOG DEEP DIVE (Blind D1)")
    print("=" * 80)

    # Split D1 by market_type
    d1_by_market = defaultdict(list)
    for r in blind_d1:
        d1_by_market[r["market_type"] or "unknown"].append(r)

    print("\n  D1 by market_type:")
    for mkt in sorted(d1_by_market.keys()):
        legs = d1_by_market[mkt]
        m = compute_metrics(legs, f"D1 {mkt}")
        print(f"\n  --- {mkt} (N={len(legs)}) ---")
        print_metrics(m, indent=1)

    # Spread favorite vs underdog
    for mkt in ["spreads", "alt_spreads"]:
        if mkt not in d1_by_market:
            continue
        spread_legs = d1_by_market[mkt]
        # Underdog: devig_entry_prob < 0.50 (less than even chance)
        dogs = [r for r in spread_legs if r["devig_entry_prob"] < 0.50]
        favs = [r for r in spread_legs if r["devig_entry_prob"] >= 0.50]

        print(f"\n  === {mkt.upper()} FAV/DOG SPLIT ===")
        print(f"\n  Underdogs (entry_prob < 0.50): N={len(dogs)}")
        print_metrics(compute_metrics(dogs, f"{mkt} Dogs"), indent=1)
        print(f"\n  Favorites (entry_prob >= 0.50): N={len(favs)}")
        print_metrics(compute_metrics(favs, f"{mkt} Favs"), indent=1)

    # Also do the FULL blind (not just D1) spread dog vs fav for comparison
    print(f"\n  === FULL BLIND (all legs) SPREAD FAV/DOG COMPARISON ===")
    for mkt in ["spreads", "alt_spreads"]:
        all_mkt = [r for r in blind if (r["market_type"] or "") == mkt]
        if not all_mkt:
            continue
        all_dogs = [r for r in all_mkt if r["devig_entry_prob"] < 0.50]
        all_favs = [r for r in all_mkt if r["devig_entry_prob"] >= 0.50]
        print(f"\n  {mkt} ALL Dogs: N={len(all_dogs)}")
        print_metrics(compute_metrics(all_dogs, f"All {mkt} Dogs"), indent=1)
        print(f"  {mkt} ALL Favs: N={len(all_favs)}")
        print_metrics(compute_metrics(all_favs, f"All {mkt} Favs"), indent=1)

    # ─── STEP 5: CLV PARADOX TEST ───
    print("\n" + "=" * 80)
    print("STEP 5: CLV PARADOX — Negative CLV yet profitable?")
    print("=" * 80)

    d1_clv = [r for r in blind_d1 if r["devig_close_prob"] is not None]
    rest_clv = [r for r in blind_rest if r["devig_close_prob"] is not None]

    if d1_clv:
        d1_clv_vals = [(r["devig_entry_prob"] - r["devig_close_prob"]) * 100 for r in d1_clv]
        d1_wins = sum(1 for r in d1_clv if r["leg_result"] == "WIN")
        d1_wr_clv = d1_wins / len(d1_clv) if d1_clv else 0
        d1_be_clv = np.mean([r["devig_entry_prob"] for r in d1_clv])
        print(f"\n  D1 legs with close prob: {len(d1_clv)}")
        print(f"  D1 avg CLV:              {np.mean(d1_clv_vals):+.3f} pp")
        print(f"  D1 CLV < 0 (negative):   {sum(1 for v in d1_clv_vals if v < 0)} / {len(d1_clv_vals)} "
              f"({sum(1 for v in d1_clv_vals if v < 0)/len(d1_clv_vals)*100:.1f}%)")
        print(f"  D1 WR (CLV subset):      {d1_wr_clv:.4f} ({d1_wr_clv*100:.2f}%)")
        print(f"  D1 BE (CLV subset):      {d1_be_clv:.4f} ({d1_be_clv*100:.2f}%)")
        print(f"  D1 Edge (CLV subset):    {(d1_wr_clv - d1_be_clv)*100:+.2f} pp")
        negative_clv = np.mean(d1_clv_vals) < 0
        profitable = d1_wr_clv > d1_be_clv
        print(f"\n  >>> PARADOX CHECK: CLV negative={negative_clv}, Profitable={profitable}")
        if negative_clv and profitable:
            print(f"  >>> CLV PARADOX HOLDS in blind period!")
        elif not negative_clv:
            print(f"  >>> No paradox: CLV is positive (model is getting better lines)")
        else:
            print(f"  >>> No paradox: not profitable despite negative CLV")
    else:
        print("\n  No close prob data available for D1 legs")

    # Split-half CLV paradox
    print(f"\n  --- CLV Paradox Split-Half ---")
    for half_label, half_data in [("H1 (July)", blind_h1), ("H2 (August)", blind_h2)]:
        half_edges_sh = [normalize_prob(r["predicted_win_prob"]) - r["devig_entry_prob"]
                         for r in half_data]
        h_d1_sh = [r for r, e in zip(half_data, half_edges_sh) if e >= threshold]
        h_d1_clv = [r for r in h_d1_sh if r["devig_close_prob"] is not None]
        if h_d1_clv:
            clv_vals_h = [(r["devig_entry_prob"] - r["devig_close_prob"]) * 100 for r in h_d1_clv]
            wins_h = sum(1 for r in h_d1_clv if r["leg_result"] == "WIN")
            wr_h = wins_h / len(h_d1_clv)
            be_h = np.mean([r["devig_entry_prob"] for r in h_d1_clv])
            neg = np.mean(clv_vals_h) < 0
            prof = wr_h > be_h
            print(f"  {half_label}: CLV={np.mean(clv_vals_h):+.3f}pp, "
                  f"WR={wr_h*100:.2f}%, BE={be_h*100:.2f}%, "
                  f"Edge={((wr_h-be_h)*100):+.2f}pp, "
                  f"Paradox={'YES' if neg and prof else 'NO'}")
        else:
            print(f"  {half_label}: no CLV data")

    # ─── STEP 6: UNDERDOG CONNECTION — Wong Teaser Overlap ───
    print("\n" + "=" * 80)
    print("STEP 6: UNDERDOG CONNECTION — Spread values + Wong teaser overlap")
    print("=" * 80)

    d1_spreads = [r for r in blind_d1
                  if r["market_type"] in ("spreads", "alt_spreads")
                  and r["devig_entry_prob"] < 0.50]

    if d1_spreads:
        spread_vals = [r["spread_value"] for r in d1_spreads if r["spread_value"] is not None]
        print(f"\n  D1 spread underdog legs: {len(d1_spreads)}")
        print(f"  With spread_value: {len(spread_vals)}")

        if spread_vals:
            # Distribution of spread values
            from collections import Counter
            sv_counts = Counter(spread_vals)
            print(f"\n  Spread value distribution:")
            for sv in sorted(sv_counts.keys()):
                pct = sv_counts[sv] / len(spread_vals) * 100
                print(f"    {sv:+.1f}: {sv_counts[sv]:4d} ({pct:.1f}%)")

            # Wong teaser key numbers: +1.5, +2.5
            wong_keys = [v for v in spread_vals if v in (1.5, 2.5)]
            print(f"\n  Wong key numbers (+1.5, +2.5): {len(wong_keys)} / {len(spread_vals)} "
                  f"({len(wong_keys)/len(spread_vals)*100:.1f}%)")

            # Check if these are truly underdogs (positive spread = receiving points)
            pos_spreads = [v for v in spread_vals if v > 0]
            neg_spreads = [v for v in spread_vals if v < 0]
            print(f"  Positive spread (true dogs): {len(pos_spreads)}")
            print(f"  Negative spread (favs at alt): {len(neg_spreads)}")
    else:
        print("\n  No D1 spread underdog legs found")

    # ─── STEP 7: STATISTICAL SIGNIFICANCE ───
    print("\n" + "=" * 80)
    print("STEP 7: STATISTICAL SIGNIFICANCE")
    print("=" * 80)

    if d1_metrics:
        # Binomial test: observed WR vs BE
        from scipy import stats as scipy_stats
        n = d1_metrics["n_raw"]
        k = d1_metrics["wins"]
        p0 = d1_metrics["be"]

        # One-sided binomial test: P(X >= k | p = p0)
        p_value = 1 - scipy_stats.binom.cdf(k - 1, n, p0)
        print(f"\n  Binomial test: {k} wins in {n} trials vs BE={p0:.4f}")
        print(f"  P-value (one-sided): {p_value:.6f}")
        print(f"  Significant at 5%:   {'YES' if p_value < 0.05 else 'NO'}")
        print(f"  Significant at 1%:   {'YES' if p_value < 0.01 else 'NO'}")

        # Z-score
        se = np.sqrt(p0 * (1 - p0) / n)
        z = (d1_metrics["wr"] - p0) / se
        print(f"  Z-score:             {z:.3f}")

    # ─── STEP 8: DECILE TABLE (BLIND ONLY) ───
    print("\n" + "=" * 80)
    print("STEP 8: FULL DECILE TABLE (BLIND PERIOD)")
    print("=" * 80)

    # Sort blind legs by model edge, split into deciles
    blind_with_edges = list(zip(blind, blind_edges))
    blind_with_edges.sort(key=lambda x: x[1], reverse=True)

    decile_size = len(blind_with_edges) // 10
    print(f"\n  {'Decile':<8} {'N':>6} {'AvgEdge':>10} {'WR':>8} {'BE':>8} {'WR-BE':>8} {'ROI_vig':>10}")
    print("  " + "-" * 62)

    for d in range(10):
        start = d * decile_size
        end = start + decile_size if d < 9 else len(blind_with_edges)
        decile_data = blind_with_edges[start:end]
        legs_d = [x[0] for x in decile_data]
        edges_d = [x[1] for x in decile_data]

        m = compute_metrics(legs_d, f"D{d+1}")
        avg_e = np.mean(edges_d) * 100
        print(f"  D{d+1:<6} {m['n_raw']:>6} {avg_e:>+10.2f} {m['wr']*100:>7.2f}% {m['be']*100:>7.2f}% "
              f"{m['edge_pp']:>+7.2f} {m['roi_vig']:>+9.2f}%")

    # ─── SUMMARY VERDICT ───
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if d1_metrics:
        print(f"\n  BLIND D1 Performance:")
        print(f"    N={d1_metrics['n_raw']} legs, {d1_metrics['n_deduped']} unique games")
        print(f"    WR={d1_metrics['wr']*100:.2f}% vs BE={d1_metrics['be']*100:.2f}%")
        print(f"    Edge={d1_metrics['edge_pp']:+.2f}pp")
        print(f"    ROI (after 5% vig)={d1_metrics['roi_vig']:+.2f}%")

        # Check split-half
        for half_label, half_data in [("H1 (July)", blind_h1), ("H2 (August)", blind_h2)]:
            half_edges_v = [normalize_prob(r["predicted_win_prob"]) - r["devig_entry_prob"]
                            for r in half_data]
            h_d1_v = [r for r, e in zip(half_data, half_edges_v) if e >= threshold]
            if h_d1_v:
                mv = compute_metrics(h_d1_v, half_label)
                print(f"    {half_label}: Edge={mv['edge_pp']:+.2f}pp, ROI_vig={mv['roi_vig']:+.2f}%")

        # Decision criteria
        positive_blind = d1_metrics["edge_pp"] > 0
        positive_vig = d1_metrics["roi_vig"] > 0

        # Split half check
        split_ok = True
        for half_label, half_data in [("H1", blind_h1), ("H2", blind_h2)]:
            half_edges_c = [normalize_prob(r["predicted_win_prob"]) - r["devig_entry_prob"]
                            for r in half_data]
            h_d1_c = [r for r, e in zip(half_data, half_edges_c) if e >= threshold]
            if h_d1_c:
                mc = compute_metrics(h_d1_c, half_label)
                if mc["edge_pp"] <= 0:
                    split_ok = False

        print(f"\n  BLIND POSITIVE:        {'PASS' if positive_blind else 'FAIL'}")
        print(f"  PROFITABLE AFTER VIG:  {'PASS' if positive_vig else 'FAIL'}")
        print(f"  SPLIT-HALF STABLE:     {'PASS' if split_ok else 'FAIL'}")
        print(f"  STAT SIGNIFICANT:      {'PASS' if p_value < 0.05 else 'FAIL'} (p={p_value:.4f})")

        if positive_blind and positive_vig and split_ok and p_value < 0.05:
            print(f"\n  *** h_0053 CONFIRMED — Deployable edge ***")
        elif positive_blind and positive_vig:
            print(f"\n  ** h_0053 MARGINAL — Positive but fails stability/significance **")
        else:
            print(f"\n  *** h_0053 REJECTED — Does not survive blind test ***")

    print("\n" + "=" * 80)
    print("END OF BLIND TEST")
    print("=" * 80)


if __name__ == "__main__":
    main()
