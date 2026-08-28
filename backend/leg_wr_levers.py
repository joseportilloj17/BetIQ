"""
leg_wr_levers.py — Phase E: Raise leg WR through selection levers.

Tests:
  1. Baseline leg WR stability (split-half) vs odds-implied breakeven
  2. Tighter gating: negbin_wp floor, short-odds exclusion
  3. Sub-populations: by negbin_wp band, negbin_cell (rung), parlay odds band
  4. Optimal parlay size (2-6 legs)
  5. Cross-arm agreement filter (legs appearing in multiple arms)

All tests require split-half stability and above-noise-floor effects.
"""
from __future__ import annotations
import sqlite3, os, math
import numpy as np

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bets.db")

# Noise floor for proportions: 2*sqrt(p*(1-p)/n) — depends on n and p
def prop_noise_floor(p, n):
    """2-sigma noise floor for a proportion."""
    return 2.0 * math.sqrt(p * (1.0 - p) / n) if n > 0 else float('inf')


def binomial_test_two_sided(wins, n, p0):
    """Two-sided binomial test: is observed win rate significantly different from p0?"""
    from scipy.stats import binom_test
    try:
        return binom_test(wins, n, p0)
    except Exception:
        from scipy.stats import binomtest
        return binomtest(wins, n, p0).pvalue


def load_edgegate_legs(conn):
    """Load all resolved edgegate legs with bet-level info."""
    rows = conn.execute("""
        SELECT
            l.id, l.mock_bet_id, l.win_prob, l.negbin_wp, l.negbin_cell,
            l.leg_result, l.fixture_id,
            b.legs as parlay_legs, b.odds as parlay_odds, b.game_date,
            b.actual_profit, b.amount, b.predicted_win_prob as bet_pred_wp
        FROM mock_bet_legs l
        JOIN mock_bets b ON b.id = l.mock_bet_id
        WHERE b.source = 'shadow_negbin_edgegate'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
        ORDER BY b.game_date, l.id
    """).fetchall()
    cols = ['leg_id','bet_id','win_prob','negbin_wp','negbin_cell',
            'leg_result','fixture_id','parlay_legs','parlay_odds','game_date',
            'actual_profit','amount','bet_pred_wp']
    return [dict(zip(cols, r)) for r in rows]


def load_cross_arm_fixture_counts(conn):
    """Count how many shadow arms each fixture_id appears in (settled legs only)."""
    rows = conn.execute("""
        SELECT l.fixture_id, COUNT(DISTINCT b.source) as n_arms
        FROM mock_bet_legs l
        JOIN mock_bets b ON b.id = l.mock_bet_id
        WHERE b.source LIKE 'shadow_negbin%'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
        GROUP BY l.fixture_id
    """).fetchall()
    return {r[0]: r[1] for r in rows}


def load_dist_wp_by_fixture(conn):
    """Get ats_dist_wp from the ungated shadow arm, keyed by fixture_id."""
    rows = conn.execute("""
        SELECT l.fixture_id, l.ats_dist_wp
        FROM mock_bet_legs l
        JOIN mock_bets b ON b.id = l.mock_bet_id
        WHERE b.source = 'shadow_negbin'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL
        GROUP BY l.fixture_id
    """).fetchall()
    return {r[0]: r[1] for r in rows}


def split_half(legs):
    """Split legs into chronological halves by game_date."""
    dates = sorted(set(l['game_date'] for l in legs))
    mid = len(dates) // 2
    h1_dates = set(dates[:mid])
    h1 = [l for l in legs if l['game_date'] in h1_dates]
    h2 = [l for l in legs if l['game_date'] not in h1_dates]
    return h1, h2


def wr_stats(legs, label=""):
    """Compute win rate and breakeven stats for a set of legs."""
    if not legs:
        return None
    wins = sum(1 for l in legs if l['leg_result'] == 'WIN')
    n = len(legs)
    wr = wins / n
    # Odds-implied breakeven: average win_prob (negbin model's prediction)
    avg_implied = np.mean([l['win_prob'] for l in legs])
    # Noise floor
    nf = prop_noise_floor(wr, n)
    return {
        'n': n, 'wins': wins, 'wr': wr,
        'avg_implied': avg_implied,
        'edge_pp': (wr - avg_implied) * 100,
        'noise_floor': nf,
        'label': label,
    }


def print_wr(stats, indent=4):
    """Pretty-print win rate stats."""
    if stats is None:
        print(f"{' '*indent}(no data)")
        return
    s = stats
    edge_sig = "ABOVE FLOOR" if abs(s['edge_pp']/100) > s['noise_floor'] else "in noise"
    print(f"{' '*indent}{s['label']}: WR={s['wr']:.1%} ({s['wins']}/{s['n']})  "
          f"implied={s['avg_implied']:.1%}  edge={s['edge_pp']:+.1f}pp  "
          f"floor=±{s['noise_floor']*100:.1f}pp  [{edge_sig}]")


def section_1_baseline(legs):
    """Section 1: Baseline leg WR stability vs odds-implied breakeven."""
    print("\n" + "="*100)
    print("  SECTION 1: BASELINE LEG WR — SPLIT-HALF STABILITY vs ODDS-IMPLIED BREAKEVEN")
    print("="*100)

    full = wr_stats(legs, "Full sample")
    print_wr(full)

    h1, h2 = split_half(legs)
    s1 = wr_stats(h1, "H1 (early)")
    s2 = wr_stats(h2, "H2 (late)")
    print_wr(s1)
    print_wr(s2)

    # Stability: are both halves above breakeven?
    stable = (s1['wr'] > s1['avg_implied']) and (s2['wr'] > s2['avg_implied'])
    print(f"\n    Split-half stable above breakeven: {'YES' if stable else 'NO'}")
    print(f"    H1 edge: {s1['edge_pp']:+.1f}pp, H2 edge: {s2['edge_pp']:+.1f}pp")

    # Binomial test: is full WR significantly above implied?
    from scipy.stats import binomtest
    result = binomtest(full['wins'], full['n'], full['avg_implied'], alternative='greater')
    print(f"    Binomial test (WR > implied): p = {result.pvalue:.4f}")

    # By-day breakdown
    print(f"\n    By-day leg WR:")
    dates = sorted(set(l['game_date'] for l in legs))
    for d in dates:
        day_legs = [l for l in legs if l['game_date'] == d]
        day_wins = sum(1 for l in day_legs if l['leg_result'] == 'WIN')
        day_implied = np.mean([l['win_prob'] for l in day_legs])
        day_wr = day_wins / len(day_legs)
        above = "+" if day_wr > day_implied else "-"
        print(f"      {d}: {day_wr:.1%} ({day_wins}/{len(day_legs)}) implied={day_implied:.1%} {above}")


def section_2_tighter_gating(legs):
    """Section 2: Tighter gating — negbin_wp floor, short-odds exclusion."""
    print("\n" + "="*100)
    print("  SECTION 2: TIGHTER GATING — WP FLOOR & SHORT-ODDS EXCLUSION")
    print("="*100)

    full = wr_stats(legs, "Full baseline")
    print_wr(full)

    # Test negbin_wp floor thresholds
    print(f"\n    --- negbin_wp FLOOR thresholds ---")
    floors = [0.70, 0.75, 0.80, 0.85, 0.90]
    for floor in floors:
        subset = [l for l in legs if l['win_prob'] >= floor]
        stats = wr_stats(subset, f"wp≥{floor:.0%}")
        if stats and stats['n'] >= 30:
            h1, h2 = split_half(subset)
            s1 = wr_stats(h1, "H1")
            s2 = wr_stats(h2, "H2")
            edge_stable = (s1['wr'] > s1['avg_implied']) and (s2['wr'] > s2['avg_implied'])
            print_wr(stats)
            print(f"          H1: {s1['wr']:.1%} ({s1['n']}), H2: {s2['wr']:.1%} ({s2['n']})  stable={edge_stable}")
        elif stats:
            print(f"    wp≥{floor:.0%}: n={stats['n']} (too small)")

    # Test short-odds EXCLUSION (parlay odds < threshold)
    # Legs in short-odds parlays
    print(f"\n    --- Short-odds EXCLUSION (by parlay odds) ---")
    # Get unique bet-level info
    bet_odds = {}
    for l in legs:
        if l['bet_id'] not in bet_odds:
            bet_odds[l['bet_id']] = l['parlay_odds']

    thresholds = [1.50, 1.60, 1.70, 1.80]
    for thresh in thresholds:
        # Exclude legs from parlays with odds below threshold
        short_bets = {bid for bid, odds in bet_odds.items() if odds and odds < thresh}
        remaining = [l for l in legs if l['bet_id'] not in short_bets]
        excluded = [l for l in legs if l['bet_id'] in short_bets]

        if len(remaining) >= 30:
            rem_s = wr_stats(remaining, f"excl odds<{thresh}")
            exc_s = wr_stats(excluded, f"excluded legs") if excluded else None
            print_wr(rem_s)
            if exc_s:
                print(f"          Excluded: WR={exc_s['wr']:.1%} ({exc_s['n']} legs)")
            h1, h2 = split_half(remaining)
            s1 = wr_stats(h1, "H1")
            s2 = wr_stats(h2, "H2")
            stable = (s1 and s2 and s1['wr'] > s1['avg_implied'] and s2['wr'] > s2['avg_implied'])
            print(f"          Split-half: H1={s1['wr']:.1%} ({s1['n']}), H2={s2['wr']:.1%} ({s2['n']})  stable={stable}")


def section_3_subpopulations(legs, dist_wp_map):
    """Section 3: Sub-populations by negbin_wp band, rung (cell), odds band."""
    print("\n" + "="*100)
    print("  SECTION 3: SUB-POPULATION LEG WR — BY WP BAND, RUNG, FIXTURE TYPE")
    print("="*100)

    # --- By negbin_wp band ---
    print(f"\n    --- By negbin_wp BAND ---")
    bands = [(0.60, 0.70), (0.70, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.00)]
    for lo, hi in bands:
        subset = [l for l in legs if lo <= l['win_prob'] < hi]
        if len(subset) < 20:
            print(f"    [{lo:.0%},{hi:.0%}): n={len(subset)} (skip)")
            continue
        stats = wr_stats(subset, f"[{lo:.0%},{hi:.0%})")
        h1, h2 = split_half(subset)
        s1 = wr_stats(h1, "H1")
        s2 = wr_stats(h2, "H2")
        print_wr(stats)
        # Edge relative to band's implied
        if s1 and s2 and s1['n'] >= 10 and s2['n'] >= 10:
            stable = (s1['wr'] > s1['avg_implied']) and (s2['wr'] > s2['avg_implied'])
            print(f"          H1: {s1['wr']:.1%} ({s1['n']}), H2: {s2['wr']:.1%} ({s2['n']})  stable_edge={stable}")
        else:
            print(f"          (half too small for stability)")

    # --- By negbin_cell (rung) ---
    print(f"\n    --- By negbin_cell (RUNG) ---")
    cells = {}
    for l in legs:
        cell = l.get('negbin_cell', 'unknown')
        if cell is None:
            cell = 'unknown'
        cells.setdefault(cell, []).append(l)

    for cell in sorted(cells.keys()):
        subset = cells[cell]
        if len(subset) < 20:
            continue
        stats = wr_stats(subset, f"cell={cell}")
        h1, h2 = split_half(subset)
        s1 = wr_stats(h1, "H1")
        s2 = wr_stats(h2, "H2")
        print_wr(stats)
        if s1 and s2 and s1['n'] >= 10 and s2['n'] >= 10:
            stable = (s1['wr'] > s1['avg_implied']) and (s2['wr'] > s2['avg_implied'])
            print(f"          H1: {s1['wr']:.1%} ({s1['n']}), H2: {s2['wr']:.1%} ({s2['n']})  stable_edge={stable}")

    # --- By ats_dist_wp band (from ungated arm, matched by fixture) ---
    print(f"\n    --- By ats_dist_wp BAND (matched from ungated arm) ---")
    legs_with_dist = []
    for l in legs:
        fid = l.get('fixture_id')
        if fid and fid in dist_wp_map:
            l2 = dict(l)
            l2['ats_dist_wp'] = dist_wp_map[fid]
            legs_with_dist.append(l2)

    print(f"    Matched: {len(legs_with_dist)}/{len(legs)} legs have ats_dist_wp")
    if legs_with_dist:
        dist_bands = [(0.40, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
        for lo, hi in dist_bands:
            subset = [l for l in legs_with_dist if lo <= l['ats_dist_wp'] < hi]
            if len(subset) < 20:
                print(f"    dist_wp [{lo:.0%},{hi:.0%}): n={len(subset)} (skip)")
                continue
            stats = wr_stats(subset, f"dist[{lo:.0%},{hi:.0%})")
            h1, h2 = split_half(subset)
            s1 = wr_stats(h1, "H1")
            s2 = wr_stats(h2, "H2")
            print_wr(stats)
            if s1 and s2 and s1['n'] >= 10 and s2['n'] >= 10:
                stable = (s1['wr'] > s1['avg_implied']) and (s2['wr'] > s2['avg_implied'])
                print(f"          H1: {s1['wr']:.1%} ({s1['n']}), H2: {s2['wr']:.1%} ({s2['n']})  stable_edge={stable}")


def section_4_parlay_size(legs):
    """Section 4: Optimal parlay size — edge by leg count."""
    print("\n" + "="*100)
    print("  SECTION 4: PARLAY SIZE — OPTIMAL LEG COUNT FOR EDGE")
    print("="*100)

    # Group legs by bet, then analyze by parlay size
    bets = {}
    for l in legs:
        bid = l['bet_id']
        if bid not in bets:
            bets[bid] = {
                'legs': [], 'parlay_legs': l['parlay_legs'],
                'parlay_odds': l['parlay_odds'], 'game_date': l['game_date'],
                'actual_profit': l['actual_profit'], 'amount': l['amount'],
            }
        bets[bid]['legs'].append(l)

    sizes = sorted(set(b['parlay_legs'] for b in bets.values()))
    print(f"\n    Available parlay sizes: {sizes}")

    for size in sizes:
        size_bets = {bid: b for bid, b in bets.items() if b['parlay_legs'] == size}
        size_legs = [l for b in size_bets.values() for l in b['legs']]

        if len(size_legs) < 30:
            print(f"\n    {size}-leg: n_legs={len(size_legs)} (skip)")
            continue

        stats = wr_stats(size_legs, f"{size}-leg")
        n_bets = len(size_bets)
        bet_wins = sum(1 for b in size_bets.values() if b['actual_profit'] > 0)
        bet_wr = bet_wins / n_bets if n_bets > 0 else 0
        avg_odds = np.mean([b['parlay_odds'] for b in size_bets.values() if b['parlay_odds']])
        breakeven_bet_wr = 1.0 / avg_odds if avg_odds > 0 else 0
        # Margin of leg WR over what's needed for breakeven
        # For k-leg parlay at leg WR p: bet WR = p^k, breakeven = 1/odds
        # Needed leg WR = (1/odds)^(1/k)
        needed_leg_wr = breakeven_bet_wr ** (1.0 / size) if size > 0 else 0
        margin_pp = (stats['wr'] - needed_leg_wr) * 100

        total_profit = sum(b['actual_profit'] for b in size_bets.values())
        roi = total_profit / sum(b['amount'] for b in size_bets.values()) * 100

        print(f"\n    {size}-leg parlays:")
        print(f"      Legs: WR={stats['wr']:.1%} ({stats['wins']}/{stats['n']})  "
              f"implied={stats['avg_implied']:.1%}  edge={stats['edge_pp']:+.1f}pp")
        print(f"      Bets: {bet_wins}/{n_bets} = {bet_wr:.1%}  "
              f"avg_odds={avg_odds:.3f}  breakeven={breakeven_bet_wr:.1%}")
        print(f"      Needed leg WR for breakeven: {needed_leg_wr:.1%}  "
              f"margin: {margin_pp:+.1f}pp")
        print(f"      PnL: ${total_profit:+.0f}  ROI: {roi:+.1f}%")

        # Split-half
        h1, h2 = split_half(size_legs)
        s1 = wr_stats(h1, "H1")
        s2 = wr_stats(h2, "H2")
        if s1 and s2 and s1['n'] >= 10 and s2['n'] >= 10:
            needed1 = breakeven_bet_wr ** (1.0 / size)
            stable = (s1['wr'] > needed1) and (s2['wr'] > needed1)
            print(f"      H1: WR={s1['wr']:.1%} ({s1['n']}), H2: WR={s2['wr']:.1%} ({s2['n']})  "
                  f"stable_above_breakeven={stable}")

    # Theoretical edge by size at observed leg WR
    print(f"\n    --- Theoretical edge at observed leg WR ---")
    overall_wr = sum(1 for l in legs if l['leg_result'] == 'WIN') / len(legs)
    print(f"    Observed overall leg WR: {overall_wr:.4f}")
    for k in range(2, 7):
        bet_wr = overall_wr ** k
        # Typical parlay odds for k legs at this WR
        fair_odds = 1.0 / bet_wr
        # Standard vig: ~10% haircut on parlay odds
        typical_odds = fair_odds * 0.90  # rough estimate
        typical_be = 1.0 / typical_odds if typical_odds > 0 else 0
        margin = (bet_wr - typical_be) * 100
        ev_per_10 = (bet_wr * typical_odds - 1) * 10
        print(f"    {k}-leg: bet_WR={bet_wr:.1%}  fair_odds={fair_odds:.1f}  "
              f"~payout_odds={typical_odds:.1f}  margin={margin:+.1f}pp  "
              f"EV/10=${ev_per_10:+.2f}")


def section_5_cross_arm(legs, fixture_arm_counts):
    """Section 5: Cross-arm agreement filter."""
    print("\n" + "="*100)
    print("  SECTION 5: CROSS-ARM AGREEMENT — LEGS IN MULTIPLE SHADOW ARMS")
    print("="*100)

    # Add arm count to each leg
    legs_w_arms = []
    for l in legs:
        l2 = dict(l)
        l2['n_arms'] = fixture_arm_counts.get(l['fixture_id'], 1)
        legs_w_arms.append(l2)

    full = wr_stats(legs_w_arms, "All edgegate legs")
    print_wr(full)

    # By arm count
    print(f"\n    --- By number of shadow arms the fixture appears in ---")
    max_arms = max(l['n_arms'] for l in legs_w_arms)
    for threshold in range(1, min(max_arms + 1, 12)):
        subset = [l for l in legs_w_arms if l['n_arms'] >= threshold]
        if len(subset) < 20:
            print(f"    ≥{threshold} arms: n={len(subset)} (skip)")
            continue
        stats = wr_stats(subset, f"≥{threshold} arms")
        h1, h2 = split_half(subset)
        s1 = wr_stats(h1, "H1")
        s2 = wr_stats(h2, "H2")
        print_wr(stats)
        if s1 and s2 and s1['n'] >= 10 and s2['n'] >= 10:
            stable = (s1['wr'] > s1['avg_implied']) and (s2['wr'] > s2['avg_implied'])
            print(f"          H1: {s1['wr']:.1%} ({s1['n']}), H2: {s2['wr']:.1%} ({s2['n']})  stable_edge={stable}")

    # Exact arm count
    print(f"\n    --- By EXACT arm count ---")
    for count in range(1, min(max_arms + 1, 12)):
        subset = [l for l in legs_w_arms if l['n_arms'] == count]
        if len(subset) < 20:
            continue
        stats = wr_stats(subset, f"exactly {count} arms")
        print_wr(stats)


def section_6_combined_filters(legs, fixture_arm_counts):
    """Section 6: Combined filters — best candidates stacked."""
    print("\n" + "="*100)
    print("  SECTION 6: COMBINED FILTERS — STACKING BEST LEVERS")
    print("="*100)

    legs_w_arms = []
    for l in legs:
        l2 = dict(l)
        l2['n_arms'] = fixture_arm_counts.get(l['fixture_id'], 1)
        legs_w_arms.append(l2)

    full = wr_stats(legs, "Full baseline")
    print_wr(full)

    # Combinations to test
    filters = [
        ("wp≥80%", lambda l: l['win_prob'] >= 0.80),
        ("wp≥85%", lambda l: l['win_prob'] >= 0.85),
        ("≥3 arms", lambda l: l['n_arms'] >= 3),
        ("≥4 arms", lambda l: l['n_arms'] >= 4),
        ("wp≥80% + ≥3 arms", lambda l: l['win_prob'] >= 0.80 and l['n_arms'] >= 3),
        ("wp≥85% + ≥3 arms", lambda l: l['win_prob'] >= 0.85 and l['n_arms'] >= 3),
        ("wp≥80% + ≥4 arms", lambda l: l['win_prob'] >= 0.80 and l['n_arms'] >= 4),
        ("wp≥85% + ≥4 arms", lambda l: l['win_prob'] >= 0.85 and l['n_arms'] >= 4),
    ]

    print(f"\n    {'Filter':<30} {'WR':>6} {'N':>6} {'Edge':>8} {'Floor':>8} {'Sig':>6} {'H1':>6} {'H2':>6} {'Stable':>7}")
    print(f"    {'─'*30} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")

    for name, filt in filters:
        subset = [l for l in legs_w_arms if filt(l)]
        if len(subset) < 20:
            print(f"    {name:<30} n={len(subset)} (skip)")
            continue
        wins = sum(1 for l in subset if l['leg_result'] == 'WIN')
        wr = wins / len(subset)
        implied = np.mean([l['win_prob'] for l in subset])
        edge = (wr - implied) * 100
        nf = prop_noise_floor(wr, len(subset)) * 100

        h1, h2 = split_half(subset)
        if h1 and h2:
            w1 = sum(1 for l in h1 if l['leg_result'] == 'WIN')
            w2 = sum(1 for l in h2 if l['leg_result'] == 'WIN')
            wr1 = w1 / len(h1) if h1 else 0
            wr2 = w2 / len(h2) if h2 else 0
            imp1 = np.mean([l['win_prob'] for l in h1])
            imp2 = np.mean([l['win_prob'] for l in h2])
            stable = (wr1 > imp1) and (wr2 > imp2)
        else:
            wr1 = wr2 = 0
            stable = False

        sig = "YES" if abs(edge) > nf else "no"
        print(f"    {name:<30} {wr:>5.1%} {len(subset):>6} {edge:>+7.1f}pp {nf:>7.1f}pp {sig:>6} "
              f"{wr1:>5.1%} {wr2:>5.1%} {'YES' if stable else 'no':>7}")


def main():
    conn = sqlite3.connect(DB)
    legs = load_edgegate_legs(conn)
    fixture_arm_counts = load_cross_arm_fixture_counts(conn)
    dist_wp_map = load_dist_wp_by_fixture(conn)

    print(f"\n  Loaded {len(legs)} resolved edgegate legs")
    print(f"  Fixture-arm map: {len(fixture_arm_counts)} fixtures")
    print(f"  Dist_wp map: {len(dist_wp_map)} fixtures")

    section_1_baseline(legs)
    section_2_tighter_gating(legs)
    section_3_subpopulations(legs, dist_wp_map)
    section_4_parlay_size(legs)
    section_5_cross_arm(legs, fixture_arm_counts)
    section_6_combined_filters(legs, fixture_arm_counts)

    conn.close()
    print(f"\n{'='*100}")
    print(f"  DONE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
