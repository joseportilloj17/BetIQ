"""
neg15_durability.py — Verify -1.5 straight edge durability and honest parlay comparison.

1. H2 margin over breakeven: is it comfortable or trending toward BE?
2. Non-overlapping parlay comparison (each fixture used once per day)
3. Straight -1.5 vs +4.5 on identical fixtures (the real finding)
"""
from __future__ import annotations
import sqlite3, os, math, random
import numpy as np

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bets.db")

RUNG_ODDS = {-1.5: 2.929, 1.5: 1.507, 2.5: 1.372, 3.5: 1.293, 4.5: 1.203}
RUNG_BE = {r: 1.0 / o for r, o in RUNG_ODDS.items()}
STAKE = 10.0


def load_fixtures(conn):
    rows = conn.execute("""
        SELECT l.fixture_id, MAX(l.ats_dist_wp), l.resolved_margin,
               b.game_date, l.negbin_cell
        FROM mock_bet_legs l
        JOIN mock_bets b ON b.id = l.mock_bet_id
        WHERE b.source = 'shadow_negbin'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL
          AND l.resolved_margin IS NOT NULL
        GROUP BY l.fixture_id
        ORDER BY b.game_date
    """).fetchall()
    return [{'fixture_id': r[0], 'max_dist_wp': r[1], 'margin': r[2],
             'game_date': r[3], 'cell': r[4]} for r in rows]


def section_1_h2_durability(fixtures):
    """Deep dive on H2 margin over breakeven."""
    print("\n" + "=" * 105)
    print("  SECTION 1: H2 DURABILITY — Is the -1.5 edge durable or hot-started?")
    print("=" * 105)

    band = [f for f in fixtures if 0.85 <= f['max_dist_wp'] < 0.90]
    n = len(band)
    dates = sorted(set(f['game_date'] for f in band))
    n_days = len(dates)
    be = RUNG_BE[-1.5]
    odds = RUNG_ODDS[-1.5]

    # Full sample
    full_wins = sum(1 for f in band if f['margin'] > 1.5)
    full_wr = full_wins / n
    print(f"\n  Full sample: {full_wr:.1%} ({full_wins}/{n}), {n_days} days, BE={be:.1%}")

    # Split-half by dates
    mid = n_days // 2
    h1_dates = set(dates[:mid])
    h2_dates = set(dates[mid:])
    h1 = [f for f in band if f['game_date'] in h1_dates]
    h2 = [f for f in band if f['game_date'] in h2_dates]

    h1_wins = sum(1 for f in h1 if f['margin'] > 1.5)
    h2_wins = sum(1 for f in h2 if f['margin'] > 1.5)
    h1_wr = h1_wins / len(h1)
    h2_wr = h2_wins / len(h2)

    h1_pnl = h1_wins * STAKE * (odds - 1) - (len(h1) - h1_wins) * STAKE
    h2_pnl = h2_wins * STAKE * (odds - 1) - (len(h2) - h2_wins) * STAKE

    print(f"\n  H1 ({dates[0]} to {dates[mid-1]}): {len(h1)} fixtures, {mid} days")
    print(f"    WR: {h1_wr:.1%} ({h1_wins}/{len(h1)}) vs BE {be:.1%}")
    print(f"    Margin over BE: {(h1_wr - be)*100:+.1f}pp")
    print(f"    PnL: ${h1_pnl:+.0f}")

    print(f"\n  H2 ({dates[mid]} to {dates[-1]}): {len(h2)} fixtures, {n_days - mid} days")
    print(f"    WR: {h2_wr:.1%} ({h2_wins}/{len(h2)}) vs BE {be:.1%}")
    print(f"    Margin over BE: {(h2_wr - be)*100:+.1f}pp")
    print(f"    PnL: ${h2_pnl:+.0f}")

    # Binomial CI for H2
    from scipy.stats import binom
    h2_n = len(h2)
    # One-sided: P(WR >= observed | p=BE)
    p_above_be = 1 - binom.cdf(h2_wins - 1, h2_n, be)
    # 95% CI for H2 WR
    ci_lo = h2_wr - 1.96 * math.sqrt(h2_wr * (1 - h2_wr) / h2_n)
    ci_hi = h2_wr + 1.96 * math.sqrt(h2_wr * (1 - h2_wr) / h2_n)

    print(f"\n  H2 statistical assessment:")
    print(f"    H2 WR 95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"    CI lower bound vs BE: {ci_lo:.1%} vs {be:.1%} → "
          f"{'CI excludes BE → SIGNIFICANT' if ci_lo > be else 'CI includes BE → NOT YET significant'}")
    print(f"    P(WR ≥ {h2_wr:.1%} | true_p = BE={be:.1%}): {p_above_be:.4f}")

    # Rolling window: 5-day trailing WR
    print(f"\n  --- Rolling 5-day -1.5 WR (trend check) ---")
    print(f"  {'Window':<25} {'N':>4} {'Wins':>5} {'WR':>6} {'vs BE':>8}")
    print(f"  {'─'*25} {'─'*4} {'─'*5} {'─'*6} {'─'*8}")

    for i in range(0, n_days - 4):
        window_dates = set(dates[i:i+5])
        window = [f for f in band if f['game_date'] in window_dates]
        if not window:
            continue
        w = sum(1 for f in window if f['margin'] > 1.5)
        wr = w / len(window)
        label = f"{dates[i]} → {dates[min(i+4, n_days-1)]}"
        marker = "▓" if wr > be else "░"
        print(f"  {label:<25} {len(window):>4} {w:>5} {wr:>5.0%} {(wr-be)*100:>+7.1f}pp {marker}")

    # Thirds split (more granular than halves)
    print(f"\n  --- Thirds split ---")
    t1_end = n_days // 3
    t2_end = 2 * n_days // 3
    thirds = [
        ("T1 (early)", set(dates[:t1_end])),
        ("T2 (middle)", set(dates[t1_end:t2_end])),
        ("T3 (late)", set(dates[t2_end:])),
    ]
    for label, td in thirds:
        t = [f for f in band if f['game_date'] in td]
        if not t:
            continue
        tw = sum(1 for f in t if f['margin'] > 1.5)
        twr = tw / len(t)
        t_pnl = tw * STAKE * (odds - 1) - (len(t) - tw) * STAKE
        above = "ABOVE" if twr > be else "BELOW"
        print(f"    {label}: {twr:.1%} ({tw}/{len(t)}) vs BE {be:.1%} → {above} | PnL: ${t_pnl:+.0f}")

    # What fraction of total PnL came from H1?
    total_pnl = h1_pnl + h2_pnl
    print(f"\n  PnL weighting:")
    print(f"    H1 contributed: ${h1_pnl:+.0f} ({h1_pnl/total_pnl*100:.0f}% of total)")
    print(f"    H2 contributed: ${h2_pnl:+.0f} ({h2_pnl/total_pnl*100:.0f}% of total)")
    print(f"    Total: ${total_pnl:+.0f}")

    # Compare to the broad edgegate H1/H2 split that died
    print(f"\n  --- Comparison: -1.5 85-90% vs broad edgegate (which died) ---")
    print(f"    Broad edgegate +4.5 (all dist_wp, leg-level):")
    print(f"      H1: 86.2% (+2.7pp over implied) — hot")
    print(f"      H2: 80.9% (-4.4pp below implied) — UNDERWATER")
    print(f"    -1.5 straight 85-90% (fixture-level):")
    print(f"      H1: {h1_wr:.1%} ({(h1_wr-be)*100:+.1f}pp over BE)")
    print(f"      H2: {h2_wr:.1%} ({(h2_wr-be)*100:+.1f}pp over BE)")
    print(f"      Key difference: H2 is {(h2_wr-be)*100:+.1f}pp ABOVE BE, not below.")
    print(f"      Edgegate H2 was -4.4pp below. This H2 is +{(h2_wr-be)*100:.1f}pp above.")


def section_2_honest_parlays(fixtures):
    """Non-overlapping parlay comparison — each fixture used at most once per day."""
    print("\n" + "=" * 105)
    print("  SECTION 2: HONEST PARLAY COMPARISON — Non-overlapping legs only")
    print("=" * 105)

    band = [f for f in fixtures if 0.85 <= f['max_dist_wp'] < 0.90]
    dates = sorted(set(f['game_date'] for f in band))
    odds_neg15 = RUNG_ODDS[-1.5]
    be_neg15 = RUNG_BE[-1.5]

    # For each day: how many fixtures qualify?
    # Non-overlapping: each fixture used in at most ONE parlay per day
    # Strategy: greedily form parlays from the day's fixtures (random order, 1000 sims)

    print(f"\n  --- Daily fixture availability ---")
    day_fixtures = {}
    for d in dates:
        day_fixtures[d] = [f for f in band if f['game_date'] == d]

    fixture_counts = [len(v) for v in day_fixtures.values()]
    print(f"  Days: {len(dates)}")
    print(f"  Fixtures/day: min={min(fixture_counts)}, median={np.median(fixture_counts):.0f}, "
          f"max={max(fixture_counts)}, mean={np.mean(fixture_counts):.1f}")

    # Straight -1.5 (baseline)
    straight_pnl = 0
    straight_bets = 0
    for d in dates:
        for f in day_fixtures[d]:
            straight_bets += 1
            if f['margin'] > 1.5:
                straight_pnl += STAKE * (odds_neg15 - 1)
            else:
                straight_pnl -= STAKE
    straight_roi = straight_pnl / (straight_bets * STAKE) * 100

    # Non-overlapping 2-leg parlays: pair fixtures greedily (deterministic: sorted order)
    parlay2_pnl = 0
    parlay2_bets = 0
    parlay2_wins = 0
    parlay2_odds = odds_neg15 ** 2 * 0.90  # 10% vig

    for d in dates:
        fxs = day_fixtures[d]
        # Pair in order (no overlap): fixtures 0+1, 2+3, etc.
        for i in range(0, len(fxs) - 1, 2):
            parlay2_bets += 1
            if fxs[i]['margin'] > 1.5 and fxs[i+1]['margin'] > 1.5:
                parlay2_pnl += STAKE * (parlay2_odds - 1)
                parlay2_wins += 1
            else:
                parlay2_pnl -= STAKE

    parlay2_staked = parlay2_bets * STAKE
    parlay2_roi = parlay2_pnl / parlay2_staked * 100 if parlay2_staked > 0 else 0
    parlay2_wr = parlay2_wins / parlay2_bets if parlay2_bets > 0 else 0
    parlay2_be = 1 / parlay2_odds

    # Non-overlapping 3-leg parlays: triple fixtures greedily
    parlay3_pnl = 0
    parlay3_bets = 0
    parlay3_wins = 0
    parlay3_odds = odds_neg15 ** 3 * 0.85  # 15% vig

    for d in dates:
        fxs = day_fixtures[d]
        for i in range(0, len(fxs) - 2, 3):
            parlay3_bets += 1
            if all(fxs[i+k]['margin'] > 1.5 for k in range(3)):
                parlay3_pnl += STAKE * (parlay3_odds - 1)
                parlay3_wins += 1
            else:
                parlay3_pnl -= STAKE

    parlay3_staked = parlay3_bets * STAKE
    parlay3_roi = parlay3_pnl / parlay3_staked * 100 if parlay3_staked > 0 else 0
    parlay3_wr = parlay3_wins / parlay3_bets if parlay3_bets > 0 else 0
    parlay3_be = 1 / parlay3_odds

    print(f"\n  === NON-OVERLAPPING COMPARISON ===")
    print(f"  {'Strategy':<25} {'Bets':>6} {'Wins':>5} {'WR':>6} {'BE':>6} "
          f"{'Staked':>8} {'PnL':>9} {'ROI':>7} {'EV/bet':>8}")
    print(f"  {'─'*25} {'─'*6} {'─'*5} {'─'*6} {'─'*6} "
          f"{'─'*8} {'─'*9} {'─'*7} {'─'*8}")
    print(f"  {'Straight -1.5':<25} {straight_bets:>6} "
          f"{sum(1 for f in band if f['margin']>1.5):>5} "
          f"{sum(1 for f in band if f['margin']>1.5)/straight_bets:>5.1%} {be_neg15:>5.1%} "
          f"${straight_bets*STAKE:>7.0f} ${straight_pnl:>+8.0f} {straight_roi:>+6.1f}% "
          f"${straight_pnl/straight_bets:>+7.2f}")
    print(f"  {'2-leg -1.5 (no overlap)':<25} {parlay2_bets:>6} "
          f"{parlay2_wins:>5} {parlay2_wr:>5.1%} {parlay2_be:>5.1%} "
          f"${parlay2_staked:>7.0f} ${parlay2_pnl:>+8.0f} {parlay2_roi:>+6.1f}% "
          f"${parlay2_pnl/max(parlay2_bets,1):>+7.2f}")
    print(f"  {'3-leg -1.5 (no overlap)':<25} {parlay3_bets:>6} "
          f"{parlay3_wins:>5} {parlay3_wr:>5.1%} {parlay3_be:>5.1%} "
          f"${parlay3_staked:>7.0f} ${parlay3_pnl:>+8.0f} {parlay3_roi:>+6.1f}% "
          f"${parlay3_pnl/max(parlay3_bets,1):>+7.2f}")

    # Monte Carlo: randomize fixture ordering 1000 times for robust non-overlapping estimate
    print(f"\n  --- Monte Carlo (1000 shuffles) for non-overlapping parlay robustness ---")
    mc_2leg_pnls = []
    mc_3leg_pnls = []
    rng = random.Random(42)

    for _ in range(1000):
        p2_pnl = 0
        p3_pnl = 0
        for d in dates:
            fxs = list(day_fixtures[d])
            rng.shuffle(fxs)
            # 2-leg
            for i in range(0, len(fxs) - 1, 2):
                if fxs[i]['margin'] > 1.5 and fxs[i+1]['margin'] > 1.5:
                    p2_pnl += STAKE * (parlay2_odds - 1)
                else:
                    p2_pnl -= STAKE
            # 3-leg
            for i in range(0, len(fxs) - 2, 3):
                if all(fxs[i+k]['margin'] > 1.5 for k in range(3)):
                    p3_pnl += STAKE * (parlay3_odds - 1)
                else:
                    p3_pnl -= STAKE
        mc_2leg_pnls.append(p2_pnl)
        mc_3leg_pnls.append(p3_pnl)

    print(f"  2-leg non-overlap (1000 shuffles):")
    print(f"    PnL: mean=${np.mean(mc_2leg_pnls):+.0f}, "
          f"P5=${np.percentile(mc_2leg_pnls,5):+.0f}, "
          f"P50=${np.median(mc_2leg_pnls):+.0f}, "
          f"P95=${np.percentile(mc_2leg_pnls,95):+.0f}")
    print(f"    ROI: mean={np.mean(mc_2leg_pnls)/(parlay2_bets*STAKE)*100:+.1f}%, "
          f"P5={np.percentile(mc_2leg_pnls,5)/(parlay2_bets*STAKE)*100:+.1f}%, "
          f"P95={np.percentile(mc_2leg_pnls,95)/(parlay2_bets*STAKE)*100:+.1f}%")
    print(f"    % profitable: {sum(1 for p in mc_2leg_pnls if p > 0)/10:.1f}%")

    print(f"  3-leg non-overlap (1000 shuffles):")
    print(f"    PnL: mean=${np.mean(mc_3leg_pnls):+.0f}, "
          f"P5=${np.percentile(mc_3leg_pnls,5):+.0f}, "
          f"P50=${np.median(mc_3leg_pnls):+.0f}, "
          f"P95=${np.percentile(mc_3leg_pnls,95):+.0f}")
    print(f"    ROI: mean={np.mean(mc_3leg_pnls)/(max(parlay3_bets,1)*STAKE)*100:+.1f}%, "
          f"P5={np.percentile(mc_3leg_pnls,5)/(max(parlay3_bets,1)*STAKE)*100:+.1f}%, "
          f"P95={np.percentile(mc_3leg_pnls,95)/(max(parlay3_bets,1)*STAKE)*100:+.1f}%")
    print(f"    % profitable: {sum(1 for p in mc_3leg_pnls if p > 0)/10:.1f}%")

    # Key comparison: straight vs non-overlapping
    print(f"\n  === THE COMPARISON THAT MATTERS ===")
    print(f"  Straight -1.5: ${straight_pnl:+.0f} PnL, {straight_roi:+.1f}% ROI "
          f"(CERTAIN — no shuffle dependence)")
    print(f"  2-leg parlay:  ${np.mean(mc_2leg_pnls):+.0f} PnL mean, "
          f"{np.mean(mc_2leg_pnls)/(parlay2_bets*STAKE)*100:+.1f}% ROI mean "
          f"(P5-P95: ${np.percentile(mc_2leg_pnls,5):+.0f} to ${np.percentile(mc_2leg_pnls,95):+.0f})")
    print(f"  3-leg parlay:  ${np.mean(mc_3leg_pnls):+.0f} PnL mean, "
          f"{np.mean(mc_3leg_pnls)/(max(parlay3_bets,1)*STAKE)*100:+.1f}% ROI mean "
          f"(P5-P95: ${np.percentile(mc_3leg_pnls,5):+.0f} to ${np.percentile(mc_3leg_pnls,95):+.0f})")


def section_3_neg15_vs_pos45(fixtures):
    """The real comparison: -1.5 vs +4.5 on identical fixtures."""
    print("\n" + "=" * 105)
    print("  SECTION 3: THE REAL FINDING — -1.5 vs +4.5 on IDENTICAL fixtures")
    print("=" * 105)

    band = [f for f in fixtures if 0.85 <= f['max_dist_wp'] < 0.90]
    n = len(band)
    dates = sorted(set(f['game_date'] for f in band))

    odds_neg15 = RUNG_ODDS[-1.5]
    odds_pos45 = RUNG_ODDS[4.5]
    be_neg15 = RUNG_BE[-1.5]
    be_pos45 = RUNG_BE[4.5]

    # Per-fixture PnL at each rung
    neg15_pnl = 0
    pos45_pnl = 0
    neg15_wins = 0
    pos45_wins = 0

    for f in band:
        if f['margin'] > 1.5:  # covers -1.5
            neg15_pnl += STAKE * (odds_neg15 - 1)
            neg15_wins += 1
        else:
            neg15_pnl -= STAKE

        if f['margin'] > -4.5:  # covers +4.5
            pos45_pnl += STAKE * (odds_pos45 - 1)
            pos45_wins += 1
        else:
            pos45_pnl -= STAKE

    neg15_wr = neg15_wins / n
    pos45_wr = pos45_wins / n
    neg15_roi = neg15_pnl / (n * STAKE) * 100
    pos45_roi = pos45_pnl / (n * STAKE) * 100

    print(f"\n  Same {n} fixtures, same $10 stake, different rung:")
    print(f"\n  {'Rung':<8} {'WR':>6} {'BE':>6} {'Margin':>8} {'PnL':>9} {'ROI':>7} {'EV/bet':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*9} {'─'*7} {'─'*8}")
    print(f"  {'-1.5':<8} {neg15_wr:>5.1%} {be_neg15:>5.1%} {(neg15_wr-be_neg15)*100:>+7.1f}pp "
          f"${neg15_pnl:>+8.0f} {neg15_roi:>+6.1f}% ${neg15_pnl/n:>+7.2f}")
    print(f"  {'+4.5':<8} {pos45_wr:>5.1%} {be_pos45:>5.1%} {(pos45_wr-be_pos45)*100:>+7.1f}pp "
          f"${pos45_pnl:>+8.0f} {pos45_roi:>+6.1f}% ${pos45_pnl/n:>+7.2f}")
    print(f"\n  -1.5 ADVANTAGE: ${neg15_pnl - pos45_pnl:+.0f} "
          f"({neg15_roi - pos45_roi:+.1f}pp ROI)")
    print(f"  Same fixtures, same stake, same model — just different rung.")

    # Per-day comparison
    print(f"\n  --- Per-day: -1.5 vs +4.5 ---")
    print(f"  {'Date':<12} {'N':>3} {'-1.5 W':>6} {'-1.5 PnL':>9} "
          f"{'+4.5 W':>6} {'+4.5 PnL':>9} {'Delta':>8}")
    print(f"  {'─'*12} {'─'*3} {'─'*6} {'─'*9} {'─'*6} {'─'*9} {'─'*8}")

    cum_neg15 = 0
    cum_pos45 = 0
    neg15_better_days = 0
    for d in dates:
        day = [f for f in band if f['game_date'] == d]
        nd = len(day)
        w15 = sum(1 for f in day if f['margin'] > 1.5)
        w45 = sum(1 for f in day if f['margin'] > -4.5)
        pnl15 = w15 * STAKE * (odds_neg15 - 1) - (nd - w15) * STAKE
        pnl45 = w45 * STAKE * (odds_pos45 - 1) - (nd - w45) * STAKE
        cum_neg15 += pnl15
        cum_pos45 += pnl45
        delta = pnl15 - pnl45
        if delta > 0:
            neg15_better_days += 1
        print(f"  {d:<12} {nd:>3} {w15:>6} ${pnl15:>+8.0f} "
              f"{w45:>6} ${pnl45:>+8.0f} ${delta:>+7.0f}")

    print(f"\n  -1.5 beats +4.5 on {neg15_better_days}/{len(dates)} days "
          f"({neg15_better_days/len(dates):.0%})")
    print(f"  Cumulative: -1.5 = ${cum_neg15:+.0f}, +4.5 = ${cum_pos45:+.0f}, "
          f"delta = ${cum_neg15 - cum_pos45:+.0f}")

    # Split-half for both
    mid = len(dates) // 2
    h1_dates = set(dates[:mid])

    for rung_label, cover_fn, rung_odds in [("-1.5", lambda f: f['margin'] > 1.5, odds_neg15),
                                              ("+4.5", lambda f: f['margin'] > -4.5, odds_pos45)]:
        h1 = [f for f in band if f['game_date'] in h1_dates]
        h2 = [f for f in band if f['game_date'] not in h1_dates]
        h1w = sum(1 for f in h1 if cover_fn(f))
        h2w = sum(1 for f in h2 if cover_fn(f))
        h1wr = h1w / len(h1)
        h2wr = h2w / len(h2)
        rung_be = 1 / rung_odds
        h1_pnl = h1w * STAKE * (rung_odds - 1) - (len(h1) - h1w) * STAKE
        h2_pnl = h2w * STAKE * (rung_odds - 1) - (len(h2) - h2w) * STAKE
        print(f"\n  {rung_label} split-half:")
        print(f"    H1: {h1wr:.1%} ({h1w}/{len(h1)}) vs BE {rung_be:.1%} → "
              f"{(h1wr-rung_be)*100:+.1f}pp | PnL ${h1_pnl:+.0f}")
        print(f"    H2: {h2wr:.1%} ({h2w}/{len(h2)}) vs BE {rung_be:.1%} → "
              f"{(h2wr-rung_be)*100:+.1f}pp | PnL ${h2_pnl:+.0f}")


def main():
    conn = sqlite3.connect(DB)
    fixtures = load_fixtures(conn)
    print(f"\n  Loaded {len(fixtures)} unique fixtures")

    section_1_h2_durability(fixtures)
    section_2_honest_parlays(fixtures)
    section_3_neg15_vs_pos45(fixtures)

    conn.close()
    print(f"\n{'='*105}")
    print(f"  DONE")
    print(f"{'='*105}")


if __name__ == "__main__":
    main()
