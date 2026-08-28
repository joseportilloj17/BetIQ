#!/usr/bin/env python3
"""
Min vs Max dist_wp: is the weakest-leg mechanism genuine, or is it
"overall leg quality" (min/max move together)?
"""

import sqlite3
import numpy as np
import re
from collections import defaultdict
from scipy import stats
from datetime import datetime

DB_PATH = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"


def load_parlays():
    conn = sqlite3.connect(DB_PATH)
    legs = conn.execute("""
        SELECT l.description, l.ats_dist_wp, l.mock_bet_id, l.negbin_cell,
               l.negbin_mu_home, l.negbin_mu_away, l.leg_result
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE b.source='shadow_negbin' AND b.cohort='clean'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL
        ORDER BY b.game_date
    """).fetchall()

    parlays_raw = conn.execute("""
        SELECT id, odds, status, game_date
        FROM mock_bets
        WHERE source='shadow_negbin' AND cohort='clean'
          AND status IN ('SETTLED_WIN','SETTLED_LOSS')
    """).fetchall()
    conn.close()

    parlay_map = {r[0]: {'odds': r[1], 'status': r[2], 'date': r[3]} for r in parlays_raw}

    leg_groups = defaultdict(list)
    for row in legs:
        leg_groups[row[2]].append({
            'desc': row[0], 'dist_wp': row[1], 'result': row[6],
        })

    parlays = []
    for pid, pinfo in parlay_map.items():
        pleg = leg_groups.get(pid, [])
        if len(pleg) < 2:
            continue
        dists = [l['dist_wp'] for l in pleg]
        parlays.append({
            'pid': pid,
            'n_legs': len(pleg),
            'min_dist': min(dists),
            'max_dist': max(dists),
            'mean_dist': np.mean(dists),
            'product': np.prod(dists),
            'spread_dist': max(dists) - min(dists),
            'won': 1 if pinfo['status'] == 'SETTLED_WIN' else 0,
            'odds': pinfo['odds'],
            'date': pinfo['date'],
        })
    return parlays


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


def main():
    parlays = load_parlays()
    print(f"Loaded {len(parlays)} parlays, WR={np.mean([p['won'] for p in parlays]):.4f}")

    y = np.array([p['won'] for p in parlays])
    dates = np.array([p['date'] for p in parlays])
    min_d = np.array([p['min_dist'] for p in parlays])
    max_d = np.array([p['max_dist'] for p in parlays])
    mean_d = np.array([p['mean_dist'] for p in parlays])
    prod = np.array([p['product'] for p in parlays])
    spread_d = np.array([p['spread_dist'] for p in parlays])
    n_legs = np.array([p['n_legs'] for p in parlays])

    # ═══════════════════════════════════════════════════════════
    print("=" * 70)
    print("1. MIN vs MAX: DO THEY MOVE TOGETHER?")
    print("=" * 70)

    r_mm, _ = stats.pearsonr(min_d, max_d)
    print(f"\n  Pearson r(min, max) = {r_mm:.4f}")
    print(f"  Pearson r(min, mean) = {stats.pearsonr(min_d, mean_d)[0]:.4f}")
    print(f"  Pearson r(max, mean) = {stats.pearsonr(max_d, mean_d)[0]:.4f}")
    print(f"  Pearson r(min, product) = {stats.pearsonr(min_d, prod)[0]:.4f}")
    print(f"  Pearson r(min, n_legs) = {stats.pearsonr(min_d, n_legs)[0]:.4f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. SOLO DISCRIMINATION")
    print("=" * 70)

    sigs = {
        'min dist_wp': min_d, 'max dist_wp': max_d, 'mean dist_wp': mean_d,
        'product': prod, 'spread (max-min)': spread_d, 'n_legs': n_legs,
    }
    for name, arr in sigs.items():
        rho, p = stats.spearmanr(arr, y)
        print(f"  {name:<20} ρ={rho:+.4f} (p={p:.6f})")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. PARTIAL CORRELATIONS — IS MIN ADDITIVE BEYOND MAX?")
    print("=" * 70)

    def partial_rho(x, y_arr, control):
        sl, ic, _, _, _ = stats.linregress(control, x)
        x_r = x - (sl * control + ic)
        sl2, ic2, _, _, _ = stats.linregress(control, y_arr)
        y_r = y_arr - (sl2 * control + ic2)
        return stats.spearmanr(x_r, y_r)

    tests = [
        ("min | max", min_d, max_d),
        ("max | min", max_d, min_d),
        ("min | mean", min_d, mean_d),
        ("max | mean", max_d, mean_d),
        ("min | product", min_d, prod),
        ("max | product", max_d, prod),
        ("min | n_legs", min_d, n_legs.astype(float)),
        ("max | n_legs", max_d, n_legs.astype(float)),
    ]

    print(f"\n  {'Test':<20} {'Partial ρ':<12} {'p-value'}")
    for name, x, ctrl in tests:
        pr, pp = partial_rho(x, y, ctrl)
        flag = " ***" if abs(pr) > 0.05 and pp < 0.01 else ""
        print(f"  {name:<20} {pr:+.4f}      {pp:.6f}{flag}")

    # Control for BOTH max and n_legs simultaneously
    print(f"\n  ── Min controlling for max AND n_legs (double residual) ──")
    from sklearn.linear_model import LinearRegression
    X_ctrl = np.column_stack([max_d, n_legs.astype(float)])
    lr = LinearRegression().fit(X_ctrl, min_d)
    min_resid = min_d - lr.predict(X_ctrl)
    lr2 = LinearRegression().fit(X_ctrl, y)
    y_resid = y - lr2.predict(X_ctrl)
    pr, pp = stats.spearmanr(min_resid, y_resid)
    print(f"  min | (max + n_legs): partial ρ={pr:+.4f}, p={pp:.6f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. OOS HEAD-TO-HEAD: MIN vs MAX vs MEAN SIZING")
    print("=" * 70)

    folds = temporal_purged_cv(dates, n_folds=5, purge_days=3)
    print(f"  {len(folds)} OOS folds")

    daily = {s: defaultdict(float) for s in ['equal', 'min', 'max', 'mean', 'product', 'min_floor']}

    for train_idx, test_idx in folds:
        test_p = [parlays[i] for i in test_idx]
        n = len(test_p)
        if n < 4:
            continue

        # Rank each signal within the fold
        for sig_name, sig_arr in [('min', min_d), ('max', max_d), ('mean', mean_d), ('product', prod)]:
            vals = sig_arr[test_idx]
            ranks = stats.rankdata(vals) / len(vals)  # 0→1

            for i, p_idx in enumerate(test_idx):
                p = parlays[p_idx]
                pnl = (p['odds'] - 1) if p['won'] else -1
                wt = ranks[i] * 2  # normalize so mean weight ≈ 1
                daily[sig_name][p['date']] += pnl * wt

        # Equal weight
        for p_idx in test_idx:
            p = parlays[p_idx]
            pnl = (p['odds'] - 1) if p['won'] else -1
            daily['equal'][p['date']] += pnl

        # Min-floor strategy: reject if min < threshold (in-fold calibrated)
        train_min = min_d[train_idx]
        train_y = y[train_idx]
        # Find threshold that maximizes WR lift (simple: median of train mins)
        threshold = np.percentile(train_min, 30)
        for p_idx in test_idx:
            p = parlays[p_idx]
            pnl = (p['odds'] - 1) if p['won'] else -1
            if p['min_dist'] >= threshold:
                daily['min_floor'][p['date']] += pnl

    all_dates = sorted(set(daily['equal'].keys()))
    print(f"\n  {'Date':<12} {'Equal':<9} {'Min':<9} {'Max':<9} {'Mean':<9} {'Product':<9} {'MinFloor'}")
    for d in all_dates:
        print(f"  {d:<12} {daily['equal'][d]:+.1f}   {daily['min'][d]:+.1f}   "
              f"{daily['max'][d]:+.1f}   {daily['mean'][d]:+.1f}   "
              f"{daily['product'][d]:+.1f}   {daily['min_floor'][d]:+.1f}")

    cums = {s: sum(daily[s][d] for d in all_dates) for s in daily}
    print(f"\n  Cumulative PnL:")
    for s in ['equal', 'min', 'max', 'mean', 'product', 'min_floor']:
        print(f"    {s:<12} {cums[s]:+.1f}")

    # Paired t-tests
    print(f"\n  Paired t-tests (daily PnL):")
    for a, b in [('min', 'equal'), ('max', 'equal'), ('min', 'max'),
                 ('min', 'product'), ('mean', 'equal'), ('min_floor', 'equal')]:
        a_d = [daily[a][d] for d in all_dates]
        b_d = [daily[b][d] for d in all_dates]
        t, p = stats.ttest_rel(a_d, b_d)
        print(f"    {a} vs {b}: t={t:+.3f}, p={p:.4f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. MECHANISM TEST: CONCENTRATED vs UNIFORM PARLAYS")
    print("=" * 70)

    # If it's "overall quality" (min≈max), then parlays where min≈max (uniform)
    # should behave the same as parlays where min<<max (concentrated risk).
    # If it's "weakest leg," concentrated-risk parlays should be MORE sensitive
    # to their min value.

    median_spread = np.median(spread_d)
    uniform = [p for p in parlays if p['spread_dist'] <= median_spread]
    concentrated = [p for p in parlays if p['spread_dist'] > median_spread]

    print(f"\n  Uniform parlays (spread ≤ {median_spread:.3f}): N={len(uniform)}")
    print(f"  Concentrated parlays (spread > {median_spread:.3f}): N={len(concentrated)}")

    for label, subset in [("Uniform", uniform), ("Concentrated", concentrated)]:
        sub_y = np.array([p['won'] for p in subset])
        sub_min = np.array([p['min_dist'] for p in subset])
        sub_max = np.array([p['max_dist'] for p in subset])
        sub_prod = np.array([p['product'] for p in subset])

        rho_min, p_min = stats.spearmanr(sub_min, sub_y)
        rho_max, p_max = stats.spearmanr(sub_max, sub_y)
        rho_prod, p_prod = stats.spearmanr(sub_prod, sub_y)

        # Partial: min|max within this subset
        sl, ic, _, _, _ = stats.linregress(sub_max, sub_min)
        min_r = sub_min - (sl * sub_max + ic)
        sl2, ic2, _, _, _ = stats.linregress(sub_max, sub_y)
        y_r = sub_y - (sl2 * sub_max + ic2)
        pr, pp = stats.spearmanr(min_r, y_r)

        print(f"\n  {label} (N={len(subset)}, WR={np.mean(sub_y):.3f}):")
        print(f"    min ρ={rho_min:+.4f}, max ρ={rho_max:+.4f}, product ρ={rho_prod:+.4f}")
        print(f"    min|max partial ρ={pr:+.4f} (p={pp:.4f})")
        print(f"    r(min,max)={stats.pearsonr(sub_min, sub_max)[0]:.4f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("6. DOES LEG COUNT CONFOUND MIN?")
    print("=" * 70)

    for nl in sorted(set(n_legs)):
        mask = n_legs == nl
        if mask.sum() < 30:
            continue
        sub_y = y[mask]
        sub_min = min_d[mask]
        sub_max = max_d[mask]
        rho_min, _ = stats.spearmanr(sub_min, sub_y)
        rho_max, _ = stats.spearmanr(sub_max, sub_y)
        print(f"  {nl}-leg parlays (N={mask.sum()}, WR={np.mean(sub_y):.3f}): "
              f"min ρ={rho_min:+.4f}, max ρ={rho_max:+.4f}")


if __name__ == "__main__":
    main()
