#!/usr/bin/env python3
"""
Targeted test: NegBin discrete margin distribution vs Normal CDF at each line.
Shows integer-run lumpiness the Normal approximation misses.
Also: does min(dist_wp) beat dist_wp product OOS at parlay level.
"""

import sqlite3
import numpy as np
import re
from collections import defaultdict
from scipy import stats
from scipy.stats import poisson, norm
from datetime import datetime, timedelta

DB_PATH = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT l.description, l.ats_dist_wp, l.bayes_team_wp, l.ridge_composite,
               l.predicted_edge_pp, l.negbin_wp, l.negbin_mu_home, l.negbin_mu_away,
               l.negbin_cell, l.leg_result, b.game_date, b.odds, b.status,
               l.mock_bet_id, l.id as leg_id, l.fixture_id
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE b.source='shadow_negbin' AND b.cohort='clean'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL
          AND l.negbin_mu_home IS NOT NULL AND l.negbin_mu_away IS NOT NULL
        ORDER BY b.game_date
    """).fetchall()
    conn.close()

    data = []
    for row in rows:
        desc = row[0]
        m = re.search(r'([+-]?\d+\.?\d*)\s*$', desc)
        spread = float(m.group(1)) if m else None
        if spread is None:
            continue

        mu_home, mu_away = row[6], row[7]
        cell = row[8]

        if cell in ('fav_away', 'dog_away'):
            pick_mu, opp_mu = mu_away, mu_home
        elif cell in ('fav_home', 'dog_home'):
            pick_mu, opp_mu = mu_home, mu_away
        else:
            continue

        expected_margin = pick_mu - opp_mu
        margin_cushion = expected_margin + spread
        poisson_sigma = np.sqrt(mu_home + mu_away)
        sigma_dist = margin_cushion / poisson_sigma if poisson_sigma > 0 else 0

        data.append({
            'description': desc, 'dist_wp': row[1], 'bayes_wp': row[2],
            'composite': row[3], 'edge_pp': row[4], 'negbin_wp': row[5],
            'mu_home': mu_home, 'mu_away': mu_away, 'cell': cell,
            'result': 1 if row[9] == 'WIN' else 0,
            'game_date': row[10], 'odds': row[11], 'parlay_status': row[12],
            'parlay_id': row[13], 'leg_id': row[14], 'fixture_id': row[15],
            'spread': spread, 'pick_mu': pick_mu, 'opp_mu': opp_mu,
            'expected_margin': expected_margin, 'margin_cushion': margin_cushion,
            'poisson_sigma': poisson_sigma, 'sigma_dist': sigma_dist,
        })
    return data


def temporal_purged_cv(dates_arr, n_folds=5, purge_days=3):
    unique_dates = sorted(set(dates_arr))
    n_dates = len(unique_dates)
    fold_size = max(n_dates // (n_folds + 1), 1)
    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_dates)
        if test_start >= n_dates:
            break
        train_dates = set(unique_dates[:train_end])
        test_dates_list = unique_dates[test_start:test_end]
        test_dt = [datetime.strptime(d, '%Y-%m-%d') for d in test_dates_list]
        purged = set()
        for td in train_dates:
            td_dt = datetime.strptime(td, '%Y-%m-%d')
            if not any(abs((td_dt - t).days) <= purge_days for t in test_dt):
                purged.add(td)
        if len(purged) < 2 or len(test_dates_list) < 1:
            continue
        test_dates = set(test_dates_list)
        tr = [i for i, d in enumerate(dates_arr) if d in purged]
        te = [i for i, d in enumerate(dates_arr) if d in test_dates]
        if len(tr) > 10 and len(te) > 10:
            folds.append((np.array(tr), np.array(te)))
    return folds


# ═══════════════════════════════════════════════════════════════
# TEST A: INTEGER MARGIN LUMPINESS
# ═══════════════════════════════════════════════════════════════

def test_integer_lumpiness(data):
    """Compute Poisson-predicted probability mass at each integer margin,
    compare with Normal CDF at each spread line."""
    print("=" * 70)
    print("INTEGER MARGIN LUMPINESS: Poisson discrete vs Normal continuous")
    print("=" * 70)

    # For each spread, compute the Poisson cover probability and Normal cover probability
    # using the actual mu_home/mu_away from each leg

    for spread_val in [1.5, 2.5, 3.5, 4.5]:
        rung_data = [d for d in data if d['spread'] == spread_val]
        if not rung_data:
            continue

        print(f"\n  ── Rung +{spread_val} (N={len(rung_data)}) ──")

        # For a representative fixture, show the margin distribution
        # Use the median-mu fixture
        medians = sorted(rung_data, key=lambda d: d['pick_mu'])
        rep = medians[len(medians) // 2]

        pick_mu, opp_mu = rep['pick_mu'], rep['opp_mu']
        print(f"  Representative fixture: pick_mu={pick_mu:.2f}, opp_mu={opp_mu:.2f}")

        # Compute Poisson margin distribution: P(pick_score - opp_score = k)
        # via convolution of Poisson(pick_mu) and Poisson(opp_mu)
        max_score = 20
        pick_pmf = np.array([poisson.pmf(k, pick_mu) for k in range(max_score + 1)])
        opp_pmf = np.array([poisson.pmf(k, opp_mu) for k in range(max_score + 1)])

        # P(margin = k) for k in range(-max_score, max_score+1)
        margin_range = range(-max_score, max_score + 1)
        margin_pmf = {}
        for k in margin_range:
            prob = 0.0
            for pick_score in range(max_score + 1):
                opp_score = pick_score - k
                if 0 <= opp_score <= max_score:
                    prob += pick_pmf[pick_score] * opp_pmf[opp_score]
            margin_pmf[k] = prob

        # Cover probability for +spread: margin > -spread
        # e.g., +1.5: cover if margin > -1.5, i.e., margin >= -1
        # For +1.5: covers if score diff >= -1 (lose by 1 or less, or win)
        # margin > -1.5 means margin >= -1 (integer)
        cover_cutoff = int(np.ceil(-spread_val + 0.001))  # margin >= this covers

        poisson_cover = sum(margin_pmf[k] for k in margin_range if k >= cover_cutoff)
        normal_cover = 1 - norm.cdf(-spread_val, loc=(pick_mu - opp_mu),
                                     scale=np.sqrt(pick_mu + opp_mu))

        # Show probability mass at each integer margin near the line
        print(f"  Cover cutoff: margin >= {cover_cutoff}")
        print(f"  Poisson cover prob: {poisson_cover:.4f}")
        print(f"  Normal cover prob:  {normal_cover:.4f}")
        print(f"  Difference:         {poisson_cover - normal_cover:+.4f}")

        print(f"\n  Probability mass near the line:")
        print(f"  {'Margin':<10} {'Poisson':<12} {'Normal':<12} {'Diff':<12} {'Covers?'}")
        for k in range(cover_cutoff - 3, cover_cutoff + 5):
            if k < -max_score or k > max_score:
                continue
            p_poisson = margin_pmf.get(k, 0)
            # Normal density at integer point ≈ CDF(k+0.5) - CDF(k-0.5)
            mu = pick_mu - opp_mu
            sig = np.sqrt(pick_mu + opp_mu)
            p_normal = norm.cdf(k + 0.5, mu, sig) - norm.cdf(k - 0.5, mu, sig)
            covers = "YES" if k >= cover_cutoff else "no"
            print(f"  {k:+3d}       {p_poisson:.4f}       {p_normal:.4f}       "
                  f"{p_poisson - p_normal:+.4f}       {covers}")

        # Now aggregate across ALL fixtures at this rung
        print(f"\n  Aggregate across all {len(rung_data)} legs:")
        all_poisson_covers = []
        all_normal_covers = []
        all_diffs = []
        mass_at_critical = []  # probability mass at the exact margin that decides cover/bust

        for d in rung_data:
            pm, om = d['pick_mu'], d['opp_mu']
            pp = np.array([poisson.pmf(k, pm) for k in range(max_score + 1)])
            op = np.array([poisson.pmf(k, om) for k in range(max_score + 1)])

            m_pmf = {}
            for k in range(-max_score, max_score + 1):
                prob = 0.0
                for ps in range(max_score + 1):
                    os_val = ps - k
                    if 0 <= os_val <= max_score:
                        prob += pp[ps] * op[os_val]
                m_pmf[k] = prob

            p_cover = sum(m_pmf[k] for k in range(-max_score, max_score + 1) if k >= cover_cutoff)
            mu = pm - om
            sig = np.sqrt(pm + om)
            n_cover = 1 - norm.cdf(-spread_val, loc=mu, scale=sig)

            all_poisson_covers.append(p_cover)
            all_normal_covers.append(n_cover)
            all_diffs.append(p_cover - n_cover)

            # Mass at the critical margin (the integer that decides cover/bust)
            # For +1.5: margin=-1 covers, margin=-2 busts. Critical = -1 and -2
            critical_cover = cover_cutoff  # just covers
            critical_bust = cover_cutoff - 1  # just busts
            mass_cover = m_pmf.get(critical_cover, 0)
            mass_bust = m_pmf.get(critical_bust, 0)
            mass_at_critical.append(mass_cover + mass_bust)

        print(f"  Poisson cover: mean={np.mean(all_poisson_covers):.4f}, "
              f"std={np.std(all_poisson_covers):.4f}")
        print(f"  Normal cover:  mean={np.mean(all_normal_covers):.4f}, "
              f"std={np.std(all_normal_covers):.4f}")
        print(f"  Diff (P - N):  mean={np.mean(all_diffs):+.4f}, "
              f"std={np.std(all_diffs):.4f}")
        print(f"  Mass at critical margins ({cover_cutoff-1}/{cover_cutoff}): "
              f"mean={np.mean(mass_at_critical):.4f}, "
              f"max={np.max(mass_at_critical):.4f}")

        # How much TOTAL probability sits within ±1 of the cover threshold?
        near_line = []
        for d in rung_data:
            pm, om = d['pick_mu'], d['opp_mu']
            pp = np.array([poisson.pmf(k, pm) for k in range(max_score + 1)])
            op = np.array([poisson.pmf(k, om) for k in range(max_score + 1)])
            m_pmf = {}
            for k in range(-max_score, max_score + 1):
                prob = 0.0
                for ps in range(max_score + 1):
                    os_val = ps - k
                    if 0 <= os_val <= max_score:
                        prob += pp[ps] * op[os_val]
                m_pmf[k] = prob
            # Mass within ±1 of cutoff
            mass = sum(m_pmf.get(k, 0) for k in [cover_cutoff - 1, cover_cutoff, cover_cutoff + 1])
            near_line.append(mass)

        print(f"  Mass within ±1 of line: mean={np.mean(near_line):.4f} "
              f"({np.mean(near_line)*100:.1f}% of all probability)")


# ═══════════════════════════════════════════════════════════════
# TEST B: MIN(DIST_WP) vs DIST_WP PRODUCT — OOS PARLAY TEST
# ═══════════════════════════════════════════════════════════════

def test_min_dist_wp_oos(data):
    """OOS test: does min(dist_wp) predict parlay outcome better than product?"""
    print("\n" + "=" * 70)
    print("MIN(DIST_WP) vs PRODUCT: OOS PARLAY-LEVEL TEST")
    print("=" * 70)

    # Group legs by parlay
    parlay_legs = defaultdict(list)
    for d in data:
        parlay_legs[d['parlay_id']].append(d)

    conn = sqlite3.connect(DB_PATH)
    parlay_info = conn.execute("""
        SELECT id, odds, status, game_date
        FROM mock_bets
        WHERE source='shadow_negbin' AND cohort='clean'
          AND status IN ('SETTLED_WIN','SETTLED_LOSS')
    """).fetchall()
    conn.close()

    parlay_map = {r[0]: {'odds': r[1], 'status': r[2], 'date': r[3]} for r in parlay_info}

    parlays = []
    for pid, legs in parlay_legs.items():
        if pid not in parlay_map or len(legs) < 2:
            continue
        pinfo = parlay_map[pid]
        parlays.append({
            'pid': pid,
            'dist_product': np.prod([l['dist_wp'] for l in legs]),
            'min_dist_wp': min(l['dist_wp'] for l in legs),
            'max_dist_wp': max(l['dist_wp'] for l in legs),
            'min_sigma': min(l['sigma_dist'] for l in legs),
            'min_cushion': min(l['margin_cushion'] for l in legs),
            'n_legs': len(legs),
            'won': 1 if pinfo['status'] == 'SETTLED_WIN' else 0,
            'odds': pinfo['odds'],
            'date': pinfo['date'],
            'legs': legs,
        })

    y = np.array([p['won'] for p in parlays])
    dates_arr = np.array([p['date'] for p in parlays])

    print(f"\n  {len(parlays)} parlays, WR={np.mean(y):.4f}")

    # ── In-sample discrimination comparison ──
    print(f"\n  ── In-sample discrimination ──")
    sigs = {
        'dist_wp product': np.array([p['dist_product'] for p in parlays]),
        'min dist_wp': np.array([p['min_dist_wp'] for p in parlays]),
        'max dist_wp': np.array([p['max_dist_wp'] for p in parlays]),
        'min sigma-dist': np.array([p['min_sigma'] for p in parlays]),
        'min margin cushion': np.array([p['min_cushion'] for p in parlays]),
    }

    for name, arr in sigs.items():
        rho, p = stats.spearmanr(arr, y)
        print(f"  {name:<25} ρ={rho:+.4f} (p={p:.6f})")

    # ── OOS test: rank parlays by product vs min_dist, compare PnL ──
    folds = temporal_purged_cv(dates_arr, n_folds=5, purge_days=3)
    print(f"\n  {len(folds)} OOS folds")

    # For each fold: rank by product, rank by min_dist. Top-half PnL vs bottom-half.
    oos_product_top_pnl = []
    oos_product_bot_pnl = []
    oos_min_top_pnl = []
    oos_min_bot_pnl = []

    for train_idx, test_idx in folds:
        test_parlays = [parlays[i] for i in test_idx]
        test_y = y[test_idx]

        # Rank by product
        sorted_by_prod = sorted(range(len(test_parlays)),
                                key=lambda i: test_parlays[i]['dist_product'], reverse=True)
        half = len(sorted_by_prod) // 2

        top_prod = sorted_by_prod[:half]
        bot_prod = sorted_by_prod[half:]

        for i in top_prod:
            p = test_parlays[i]
            pnl = (p['odds'] - 1) if p['won'] else -1
            oos_product_top_pnl.append(pnl)
        for i in bot_prod:
            p = test_parlays[i]
            pnl = (p['odds'] - 1) if p['won'] else -1
            oos_product_bot_pnl.append(pnl)

        # Rank by min_dist_wp
        sorted_by_min = sorted(range(len(test_parlays)),
                               key=lambda i: test_parlays[i]['min_dist_wp'], reverse=True)

        top_min = sorted_by_min[:half]
        bot_min = sorted_by_min[half:]

        for i in top_min:
            p = test_parlays[i]
            pnl = (p['odds'] - 1) if p['won'] else -1
            oos_min_top_pnl.append(pnl)
        for i in bot_min:
            p = test_parlays[i]
            pnl = (p['odds'] - 1) if p['won'] else -1
            oos_min_bot_pnl.append(pnl)

    print(f"\n  ── OOS top-half vs bottom-half PnL ──")
    print(f"  Ranked by dist_wp product:")
    print(f"    Top half: N={len(oos_product_top_pnl)}, "
          f"WR={sum(1 for p in oos_product_top_pnl if p > 0)/len(oos_product_top_pnl):.3f}, "
          f"PnL={sum(oos_product_top_pnl):+.1f}")
    print(f"    Bot half: N={len(oos_product_bot_pnl)}, "
          f"WR={sum(1 for p in oos_product_bot_pnl if p > 0)/len(oos_product_bot_pnl):.3f}, "
          f"PnL={sum(oos_product_bot_pnl):+.1f}")
    print(f"    Lift (top-bot): {sum(oos_product_top_pnl) - sum(oos_product_bot_pnl):+.1f}")

    print(f"\n  Ranked by min dist_wp:")
    print(f"    Top half: N={len(oos_min_top_pnl)}, "
          f"WR={sum(1 for p in oos_min_top_pnl if p > 0)/len(oos_min_top_pnl):.3f}, "
          f"PnL={sum(oos_min_top_pnl):+.1f}")
    print(f"    Bot half: N={len(oos_min_bot_pnl)}, "
          f"WR={sum(1 for p in oos_min_bot_pnl if p > 0)/len(oos_min_bot_pnl):.3f}, "
          f"PnL={sum(oos_min_bot_pnl):+.1f}")
    print(f"    Lift (top-bot): {sum(oos_min_top_pnl) - sum(oos_min_bot_pnl):+.1f}")

    # ── OOS: sizing by min_dist_wp vs product ──
    print(f"\n  ── OOS sizing: weight by signal rank ──")
    daily_pnl_equal = defaultdict(float)
    daily_pnl_product = defaultdict(float)
    daily_pnl_min = defaultdict(float)
    daily_pnl_combined = defaultdict(float)  # product × min

    for train_idx, test_idx in folds:
        test_parlays_fold = [parlays[i] for i in test_idx]

        # Rank-based weights
        n = len(test_parlays_fold)
        if n < 4:
            continue

        by_prod = sorted(range(n), key=lambda i: test_parlays_fold[i]['dist_product'])
        by_min = sorted(range(n), key=lambda i: test_parlays_fold[i]['min_dist_wp'])

        prod_rank = [0] * n
        min_rank = [0] * n
        for rank, idx in enumerate(by_prod):
            prod_rank[idx] = (rank + 1) / n  # 0→1
        for rank, idx in enumerate(by_min):
            min_rank[idx] = (rank + 1) / n

        for i, p in enumerate(test_parlays_fold):
            pnl = (p['odds'] - 1) if p['won'] else -1
            date = p['date']

            daily_pnl_equal[date] += pnl
            daily_pnl_product[date] += pnl * prod_rank[i] * 2  # normalize
            daily_pnl_min[date] += pnl * min_rank[i] * 2
            daily_pnl_combined[date] += pnl * (prod_rank[i] * min_rank[i]) * 3

    all_dates = sorted(set(daily_pnl_equal.keys()))

    print(f"\n  {'Date':<12} {'Equal':<10} {'Product':<10} {'Min_dist':<10} {'Combined':<10}")
    for d in all_dates:
        print(f"  {d:<12} {daily_pnl_equal[d]:+.1f}    "
              f"{daily_pnl_product[d]:+.1f}    "
              f"{daily_pnl_min[d]:+.1f}    "
              f"{daily_pnl_combined[d]:+.1f}")

    cum_e = sum(daily_pnl_equal[d] for d in all_dates)
    cum_p = sum(daily_pnl_product[d] for d in all_dates)
    cum_m = sum(daily_pnl_min[d] for d in all_dates)
    cum_c = sum(daily_pnl_combined[d] for d in all_dates)
    print(f"\n  Cumulative:")
    print(f"    Equal:    {cum_e:+.1f}")
    print(f"    Product:  {cum_p:+.1f}")
    print(f"    Min dist: {cum_m:+.1f}")
    print(f"    Combined: {cum_c:+.1f}")

    e_d = [daily_pnl_equal[d] for d in all_dates]
    p_d = [daily_pnl_product[d] for d in all_dates]
    m_d = [daily_pnl_min[d] for d in all_dates]
    c_d = [daily_pnl_combined[d] for d in all_dates]

    if len(e_d) > 2:
        t_pe, p_pe = stats.ttest_rel(p_d, e_d)
        t_me, p_me = stats.ttest_rel(m_d, e_d)
        t_mp, p_mp = stats.ttest_rel(m_d, p_d)
        t_ce, p_ce = stats.ttest_rel(c_d, e_d)
        print(f"\n  Paired t-tests:")
        print(f"    Product vs Equal:  t={t_pe:.3f}, p={p_pe:.4f}")
        print(f"    Min vs Equal:      t={t_me:.3f}, p={p_me:.4f}")
        print(f"    Min vs Product:    t={t_mp:.3f}, p={p_mp:.4f}")
        print(f"    Combined vs Equal: t={t_ce:.3f}, p={p_ce:.4f}")

    # ── What does the weakest leg look like in winning vs losing parlays? ──
    print(f"\n  ── Weakest leg profile: winners vs losers ──")
    won_parlays = [p for p in parlays if p['won']]
    lost_parlays = [p for p in parlays if not p['won']]

    print(f"  {'Metric':<25} {'Winners (N={len(won_parlays)})':<25} {'Losers (N={len(lost_parlays)})':<25}")

    for metric in ['min_dist_wp', 'dist_product', 'min_cushion', 'min_sigma']:
        w_vals = [p[metric] for p in won_parlays]
        l_vals = [p[metric] for p in lost_parlays]
        t, tp = stats.ttest_ind(w_vals, l_vals)
        print(f"  {metric:<25} {np.mean(w_vals):.4f} ± {np.std(w_vals):.4f}      "
              f"{np.mean(l_vals):.4f} ± {np.std(l_vals):.4f}      t={t:.2f}, p={tp:.4f}")

    # ── Breakeven analysis: at what min_dist_wp threshold does WR cover the odds? ──
    print(f"\n  ── Min dist_wp threshold analysis ──")
    print(f"  {'Threshold':<12} {'Kept':<8} {'WR':<8} {'PnL':<10} {'ROI'}")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        kept = [p for p in parlays if p['min_dist_wp'] >= thresh]
        if len(kept) < 20:
            continue
        wr = np.mean([p['won'] for p in kept])
        pnl = sum((p['odds'] - 1) if p['won'] else -1 for p in kept)
        roi = pnl / len(kept) * 100
        print(f"  ≥{thresh:.2f}       {len(kept):<8} {wr:.3f}  {pnl:+.1f}     {roi:+.1f}%")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} settled legs, {len(set(d['game_date'] for d in data))} days")

    test_integer_lumpiness(data)
    test_min_dist_wp_oos(data)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
