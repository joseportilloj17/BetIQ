#!/usr/bin/env python3
"""
Margin-of-cover analysis: sigma-distance from line, weakest-leg cushion,
empirical margin lumpiness at each spread.

Uses NegBin model parameters (mu_home, mu_away) to compute predicted margin
distributions. Tests whether sigma-distance discriminates better than dist_wp.
"""

import sqlite3
import numpy as np
import re
from collections import defaultdict
from scipy import stats
from scipy.special import comb as scipy_comb
from sklearn.metrics import brier_score_loss

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

        mu_home = row[6]
        mu_away = row[7]
        cell = row[8]

        # Determine pick team's expected runs.
        # cell encoding: "fav_away" means favorite is away team.
        # pick_side from code: home picks get fav_home/dog_home cells,
        #                      away picks get fav_away/dog_away cells.
        # - fav_away: pick_side=away, pick is the favorite (away team)
        # - dog_away: pick_side=away, pick is the dog (away team)
        # - fav_home: pick_side=home, pick is the favorite (home team)
        # - dog_home: pick_side=home, pick is the dog (home team)

        if cell in ('fav_away', 'dog_away'):
            pick_mu = mu_away
            opp_mu = mu_home
        elif cell in ('fav_home', 'dog_home'):
            pick_mu = mu_home
            opp_mu = mu_away
        else:
            # Unknown — try to infer from description. Skip for safety.
            continue

        # Expected margin from pick team's perspective: pick_runs - opp_runs
        expected_margin = pick_mu - opp_mu

        # For alt runlines like +1.5: the team covers if their actual margin > -spread
        # i.e., they can lose by up to (spread - 0.5) runs
        # Margin cushion = expected_margin - (-spread) = expected_margin + spread
        # This is how far ABOVE the cover threshold the expected outcome sits.
        margin_cushion = expected_margin + spread

        # Sigma for NegBin margin distribution.
        # For independent Poisson: Var(H - A) = mu_h + mu_a, so sigma = sqrt(mu_h + mu_a)
        # NegBin has higher variance: Var = mu + mu^2/r. Without r, use Poisson as lower bound.
        # Also compute using the user's global sigma=4.217 for comparison.
        poisson_sigma = np.sqrt(mu_home + mu_away)
        global_sigma = 4.217

        # Sigma-distance from line: how many sigma is the cushion?
        sigma_dist_poisson = margin_cushion / poisson_sigma if poisson_sigma > 0 else 0
        sigma_dist_global = margin_cushion / global_sigma

        data.append({
            'description': desc,
            'dist_wp': row[1],
            'bayes_wp': row[2],
            'composite': row[3],
            'edge_pp': row[4],
            'negbin_wp': row[5],
            'mu_home': mu_home,
            'mu_away': mu_away,
            'cell': cell,
            'result': 1 if row[9] == 'WIN' else 0,
            'game_date': row[10],
            'odds': row[11],
            'parlay_status': row[12],
            'parlay_id': row[13],
            'leg_id': row[14],
            'fixture_id': row[15],
            'spread': spread,
            'pick_mu': pick_mu,
            'opp_mu': opp_mu,
            'expected_margin': expected_margin,
            'margin_cushion': margin_cushion,
            'poisson_sigma': poisson_sigma,
            'sigma_dist_poisson': sigma_dist_poisson,
            'sigma_dist_global': sigma_dist_global,
            'game_total': mu_home + mu_away,
        })

    return data


def temporal_purged_cv(dates_arr, n_folds=5, purge_days=3):
    from datetime import datetime, timedelta
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
# TEST 1: MARGIN CUSHION vs DIST_WP AS DISCRIMINATOR
# ═══════════════════════════════════════════════════════════════

def test_margin_discrimination(data):
    print("=" * 70)
    print("TEST 1: MARGIN CUSHION vs DIST_WP DISCRIMINATION")
    print("=" * 70)

    y = np.array([d['result'] for d in data])
    dist_wp = np.array([d['dist_wp'] for d in data])
    sigma_d_p = np.array([d['sigma_dist_poisson'] for d in data])
    sigma_d_g = np.array([d['sigma_dist_global'] for d in data])
    margin_c = np.array([d['margin_cushion'] for d in data])
    negbin_wp = np.array([d['negbin_wp'] for d in data])

    signals = {
        'dist_wp': dist_wp,
        'negbin_wp': negbin_wp,
        'margin_cushion': margin_c,
        'sigma_dist (Poisson)': sigma_d_p,
        'sigma_dist (global σ=4.217)': sigma_d_g,
    }

    print(f"\n  ── Solo discrimination (Spearman ρ with outcome) ──")
    print(f"  {'Signal':<30} {'ρ':<10} {'p':<12} {'Quintile ρ'}")
    for name, arr in signals.items():
        rho, p = stats.spearmanr(arr, y)
        # Quintile WR
        qs = np.percentile(arr, [20, 40, 60, 80])
        q_wrs = []
        for qi in range(5):
            if qi == 0: mask = arr <= qs[0]
            elif qi == 4: mask = arr > qs[3]
            else: mask = (arr > qs[qi-1]) & (arr <= qs[qi])
            q_wrs.append(np.mean(y[mask]) if mask.sum() > 0 else 0.5)
        q_rho, _ = stats.spearmanr(range(5), q_wrs)
        print(f"  {name:<30} {rho:+.4f}    {p:.6f}  {q_rho:+.4f}  "
              f"Q0={q_wrs[0]:.3f}→Q4={q_wrs[4]:.3f}")

    # ── Does sigma-distance add value BEYOND dist_wp? ──
    print(f"\n  ── Partial correlation (controlling for dist_wp) ──")
    for name, arr in signals.items():
        if name == 'dist_wp':
            continue
        # Residualize
        sl, ic, _, _, _ = stats.linregress(dist_wp, arr)
        resid_feat = arr - (sl * dist_wp + ic)
        sl2, ic2, _, _, _ = stats.linregress(dist_wp, y)
        resid_y = y - (sl2 * dist_wp + ic2)
        pr, pp = stats.spearmanr(resid_feat, resid_y)
        print(f"  {name:<30} partial ρ={pr:+.4f}, p={pp:.6f}")

    # ── Per-rung: does sigma-distance work better at narrow rungs? ──
    print(f"\n  ── Per-rung sigma-distance discrimination ──")
    print(f"  {'Rung':<8} {'N':<6} {'dist_wp ρ':<12} {'σ-dist ρ':<12} {'Δρ':<10} {'cushion ρ'}")
    for spread_val in [1.5, 2.5, 3.5, 4.5]:
        rung_mask = np.array([d['spread'] == spread_val for d in data])
        if rung_mask.sum() < 40:
            continue
        rung_y = y[rung_mask]
        rung_dist = dist_wp[rung_mask]
        rung_sig = sigma_d_p[rung_mask]
        rung_mc = margin_c[rung_mask]

        rho_d, _ = stats.spearmanr(rung_dist, rung_y)
        rho_s, _ = stats.spearmanr(rung_sig, rung_y)
        rho_c, _ = stats.spearmanr(rung_mc, rung_y)
        print(f"  +{spread_val:<7} {rung_mask.sum():<6} {rho_d:+.4f}      "
              f"{rho_s:+.4f}      {rho_s - rho_d:+.4f}    {rho_c:+.4f}")

    # ── Does dist_wp collapse "covers by 8" and "covers by 1"? ──
    print(f"\n  ── Dist_wp vs margin cushion correlation ──")
    r, p = stats.pearsonr(dist_wp, margin_c)
    print(f"  Pearson r(dist_wp, margin_cushion) = {r:.4f} (p={p:.6f})")
    print(f"  → {'HIGH' if abs(r) > 0.7 else 'MODERATE' if abs(r) > 0.4 else 'LOW'} correlation")
    print(f"  If high: dist_wp already encodes margin distance.")
    print(f"  If low: sigma-distance captures something dist_wp misses.")

    # Check: among legs with similar dist_wp, do sigma-distance extremes differ in WR?
    # Split dist_wp into deciles, then within each decile compare high vs low sigma-dist
    print(f"\n  ── Within dist_wp decile: sigma-distance discrimination ──")
    decile_edges = np.percentile(dist_wp, np.arange(10, 100, 10))
    for i in range(10):
        if i == 0:
            mask = dist_wp <= decile_edges[0]
        elif i == 9:
            mask = dist_wp > decile_edges[8]
        else:
            mask = (dist_wp > decile_edges[i-1]) & (dist_wp <= decile_edges[i])
        if mask.sum() < 30:
            continue
        sub_sig = sigma_d_p[mask]
        sub_y = y[mask]
        median_sig = np.median(sub_sig)
        high = sub_sig >= median_sig
        low = sub_sig < median_sig
        wr_high = np.mean(sub_y[high]) if high.sum() > 0 else 0
        wr_low = np.mean(sub_y[low]) if low.sum() > 0 else 0
        print(f"  Decile {i}: dist_wp=[{np.min(dist_wp[mask]):.3f}-{np.max(dist_wp[mask]):.3f}] "
              f"N={mask.sum():4d}  σ-high WR={wr_high:.3f}  σ-low WR={wr_low:.3f}  "
              f"Δ={wr_high - wr_low:+.3f}")


# ═══════════════════════════════════════════════════════════════
# TEST 2: CONFIDENCE OF COVER — sigma as sizing signal
# ═══════════════════════════════════════════════════════════════

def test_sigma_sizing(data):
    print("\n" + "=" * 70)
    print("TEST 2: SIGMA-DISTANCE AS SIZING SIGNAL")
    print("=" * 70)

    y = np.array([d['result'] for d in data])
    dates_arr = np.array([d['game_date'] for d in data])
    dist_wp = np.array([d['dist_wp'] for d in data])
    sigma_d = np.array([d['sigma_dist_poisson'] for d in data])

    # Build sigma quintiles and check WR
    print(f"\n  ── Sigma-distance quintile cover rates ──")
    qs = np.percentile(sigma_d, [20, 40, 60, 80])
    print(f"  {'Quintile':<10} {'σ-range':<20} {'N':<6} {'WR':<8} {'Avg dist_wp'}")
    for qi in range(5):
        if qi == 0: mask = sigma_d <= qs[0]
        elif qi == 4: mask = sigma_d > qs[3]
        else: mask = (sigma_d > qs[qi-1]) & (sigma_d <= qs[qi])
        n = mask.sum()
        wr = np.mean(y[mask])
        avg_d = np.mean(dist_wp[mask])
        lo = np.min(sigma_d[mask])
        hi = np.max(sigma_d[mask])
        print(f"  Q{qi}        [{lo:.2f}, {hi:.2f}]    {n:<6} {wr:.3f}  {avg_d:.3f}")

    # OOS test: does sigma-distance-weighted strategy beat equal?
    folds = temporal_purged_cv(dates_arr, n_folds=5, purge_days=3)

    daily_equal = defaultdict(float)
    daily_sigma_wt = defaultdict(float)
    daily_dist_wt = defaultdict(float)

    for train_idx, test_idx in folds:
        # In-fold: compute sigma quintile WRs
        train_sig = sigma_d[train_idx]
        train_y = y[train_idx]
        train_qs = np.percentile(train_sig, [20, 40, 60, 80])

        # Map sigma to weight based on in-fold quintile WR
        q_wr_map = []
        for qi in range(5):
            if qi == 0: m = train_sig <= train_qs[0]
            elif qi == 4: m = train_sig > train_qs[3]
            else: m = (train_sig > train_qs[qi-1]) & (train_sig <= train_qs[qi])
            q_wr_map.append(np.mean(train_y[m]) if m.sum() > 0 else 0.5)

        avg_qw = np.mean(q_wr_map)

        for i in test_idx:
            d = data[i]
            date = d['game_date']
            won = d['result']
            pnl = 1.0 if won else -1.0

            daily_equal[date] += pnl

            # Sigma weight
            sig = d['sigma_dist_poisson']
            if sig <= train_qs[0]: qi = 0
            elif sig > train_qs[3]: qi = 4
            elif sig > train_qs[2]: qi = 3
            elif sig > train_qs[1]: qi = 2
            else: qi = 1
            wt = q_wr_map[qi] / avg_qw
            daily_sigma_wt[date] += pnl * wt

            # dist_wp weight (same approach)
            dwp = d['dist_wp']
            daily_dist_wt[date] += pnl * dwp / np.mean(dist_wp)

    all_dates = sorted(set(daily_equal.keys()))
    cum_e = sum(daily_equal[d] for d in all_dates)
    cum_s = sum(daily_sigma_wt[d] for d in all_dates)
    cum_d = sum(daily_dist_wt[d] for d in all_dates)

    print(f"\n  OOS cumulative PnL (leg units):")
    print(f"    Equal weight:    {cum_e:+.1f}")
    print(f"    Sigma-weighted:  {cum_s:+.1f}")
    print(f"    Dist_wp-weighted:{cum_d:+.1f}")

    e_daily = [daily_equal[d] for d in all_dates]
    s_daily = [daily_sigma_wt[d] for d in all_dates]
    d_daily = [daily_dist_wt[d] for d in all_dates]
    if len(e_daily) > 2:
        t_se, p_se = stats.ttest_rel(s_daily, e_daily)
        t_sd, p_sd = stats.ttest_rel(s_daily, d_daily)
        print(f"    Paired t (σ vs equal): t={t_se:.3f}, p={p_se:.4f}")
        print(f"    Paired t (σ vs dist):  t={t_sd:.3f}, p={p_sd:.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST 3: PARLAY-LEVEL — WEAKEST LEG CUSHION
# ═══════════════════════════════════════════════════════════════

def test_weakest_leg(data):
    print("\n" + "=" * 70)
    print("TEST 3: WEAKEST-LEG CUSHION (parlay-level)")
    print("=" * 70)

    # Group legs by parlay_id
    parlay_legs = defaultdict(list)
    for d in data:
        parlay_legs[d['parlay_id']].append(d)

    # For each parlay, compute:
    # - dist_wp product
    # - min sigma-distance (weakest leg)
    # - avg sigma-distance
    # - parlay outcome

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
        if pid not in parlay_map:
            continue
        pinfo = parlay_map[pid]
        if len(legs) < 2:
            continue

        dist_product = np.prod([l['dist_wp'] for l in legs])
        min_sigma = min(l['sigma_dist_poisson'] for l in legs)
        avg_sigma = np.mean([l['sigma_dist_poisson'] for l in legs])
        min_dist_wp = min(l['dist_wp'] for l in legs)
        min_cushion = min(l['margin_cushion'] for l in legs)
        won = 1 if pinfo['status'] == 'SETTLED_WIN' else 0

        parlays.append({
            'pid': pid,
            'dist_product': dist_product,
            'min_sigma': min_sigma,
            'avg_sigma': avg_sigma,
            'min_dist_wp': min_dist_wp,
            'min_cushion': min_cushion,
            'n_legs': len(legs),
            'won': won,
            'odds': pinfo['odds'],
            'date': pinfo['date'],
        })

    y = np.array([p['won'] for p in parlays])

    signals = {
        'dist_wp product': np.array([p['dist_product'] for p in parlays]),
        'min dist_wp': np.array([p['min_dist_wp'] for p in parlays]),
        'min σ-distance (weakest leg)': np.array([p['min_sigma'] for p in parlays]),
        'avg σ-distance': np.array([p['avg_sigma'] for p in parlays]),
        'min margin cushion': np.array([p['min_cushion'] for p in parlays]),
    }

    print(f"\n  {len(parlays)} parlays analyzed")
    print(f"  Parlay WR: {np.mean(y):.4f}")

    print(f"\n  ── Solo discrimination at parlay level ──")
    print(f"  {'Signal':<35} {'ρ':<10} {'p':<12} {'Q0→Q4 WR'}")
    for name, arr in signals.items():
        rho, p = stats.spearmanr(arr, y)
        qs = np.percentile(arr, [20, 40, 60, 80])
        q_wrs = []
        for qi in range(5):
            if qi == 0: mask = arr <= qs[0]
            elif qi == 4: mask = arr > qs[3]
            else: mask = (arr > qs[qi-1]) & (arr <= qs[qi])
            q_wrs.append(np.mean(y[mask]) if mask.sum() > 0 else 0.5)
        print(f"  {name:<35} {rho:+.4f}    {p:.6f}  {q_wrs[0]:.3f}→{q_wrs[4]:.3f}")

    # ── Partial: min sigma controlling for dist_product ──
    print(f"\n  ── Partial correlation (controlling for dist_wp product) ──")
    dp = signals['dist_wp product']
    for name, arr in signals.items():
        if name == 'dist_wp product':
            continue
        sl, ic, _, _, _ = stats.linregress(dp, arr)
        resid_f = arr - (sl * dp + ic)
        sl2, ic2, _, _, _ = stats.linregress(dp, y)
        resid_y = y - (sl2 * dp + ic2)
        pr, pp = stats.spearmanr(resid_f, resid_y)
        print(f"  {name:<35} partial ρ={pr:+.4f}, p={pp:.6f}")

    # ── Quintile comparison: dist_product vs min_sigma ──
    print(f"\n  ── Quintile P&L: dist_product rank vs min_sigma rank ──")
    for name, arr in [('dist_wp product', signals['dist_wp product']),
                      ('min σ-distance', signals['min σ-distance (weakest leg)'])]:
        qs = np.percentile(arr, [20, 40, 60, 80])
        print(f"  {name}:")
        for qi in range(5):
            if qi == 0: mask = arr <= qs[0]
            elif qi == 4: mask = arr > qs[3]
            else: mask = (arr > qs[qi-1]) & (arr <= qs[qi])
            sub_y = y[np.where(mask)[0]]
            sub_odds = np.array([parlays[i]['odds'] for i in np.where(mask)[0]])
            wr = np.mean(sub_y)
            pnl = sum(sub_odds[j] - 1 if sub_y[j] else -1 for j in range(len(sub_y)))
            print(f"    Q{qi}: WR={wr:.3f}  N={mask.sum():4d}  PnL={pnl:+.1f}")

    # ── Does weakest-leg selection improve PnL? ──
    print(f"\n  ── Strategy: reject parlays with min_σ < threshold ──")
    for threshold in [0.5, 0.75, 1.0, 1.25, 1.5]:
        kept = [p for p in parlays if p['min_sigma'] >= threshold]
        rej = [p for p in parlays if p['min_sigma'] < threshold]
        if not kept or not rej:
            continue
        wr_k = np.mean([p['won'] for p in kept])
        wr_r = np.mean([p['won'] for p in rej])
        pnl_k = sum((p['odds'] - 1) if p['won'] else -1 for p in kept)
        pnl_r = sum((p['odds'] - 1) if p['won'] else -1 for p in rej)
        print(f"  σ≥{threshold:.2f}: kept={len(kept)}, WR={wr_k:.3f}, PnL={pnl_k:+.1f} | "
              f"rejected={len(rej)}, WR={wr_r:.3f}, PnL={pnl_r:+.1f}")


# ═══════════════════════════════════════════════════════════════
# TEST 4: EMPIRICAL MARGIN LUMPINESS — Normal assumption check
# ═══════════════════════════════════════════════════════════════

def test_margin_lumpiness(data):
    print("\n" + "=" * 70)
    print("TEST 4: EMPIRICAL MARGIN DISTRIBUTION vs NORMAL ASSUMPTION")
    print("=" * 70)

    print("""
  Note: We don't have actual realized margins for the shadow period.
  Instead we test whether the NegBin MODEL's predicted cover probability
  (negbin_wp) deviates from the Normal-CDF approximation at each spread.

  If the NegBin discrete distribution (which IS lumpy by construction)
  gives materially different cover probs than a Normal CDF with the
  same mean and sigma, the Normal approximation is losing information.
    """)

    # Compare negbin_wp (from discrete NegBin simulation) vs Normal CDF approximation
    for spread_val in [1.5, 2.5, 3.5, 4.5]:
        rung_data = [d for d in data if d['spread'] == spread_val]
        if not rung_data:
            continue

        negbin_probs = np.array([d['negbin_wp'] for d in rung_data])
        normal_probs = np.array([
            stats.norm.cdf(d['margin_cushion'] / d['poisson_sigma'])
            for d in rung_data
        ])
        dist_probs = np.array([d['dist_wp'] for d in rung_data])
        y = np.array([d['result'] for d in rung_data])

        # How different are NegBin vs Normal CDF?
        diff = negbin_probs - normal_probs
        corr, _ = stats.pearsonr(negbin_probs, normal_probs)

        # Which is better calibrated?
        brier_negbin = brier_score_loss(y, negbin_probs)
        brier_normal = brier_score_loss(y, normal_probs)
        brier_dist = brier_score_loss(y, dist_probs)

        # Discrimination
        rho_negbin, _ = stats.spearmanr(negbin_probs, y)
        rho_normal, _ = stats.spearmanr(normal_probs, y)
        rho_dist, _ = stats.spearmanr(dist_probs, y)

        print(f"  ── Rung +{spread_val} (N={len(rung_data)}) ──")
        print(f"  NegBin vs Normal CDF correlation: r={corr:.4f}")
        print(f"  NegBin - Normal diff: mean={np.mean(diff):+.4f}, "
              f"std={np.std(diff):.4f}, max|diff|={np.max(np.abs(diff)):.4f}")
        print(f"  Calibration (Brier): NegBin={brier_negbin:.4f}, "
              f"Normal={brier_normal:.4f}, dist_wp={brier_dist:.4f}")
        print(f"  Discrimination (ρ):  NegBin={rho_negbin:+.4f}, "
              f"Normal={rho_normal:+.4f}, dist_wp={rho_dist:+.4f}")
        print(f"  Normal CDF vs NegBin ΔBrier: {brier_normal - brier_negbin:+.4f}")
        print()

    # ── Margin distribution near each line: where does the model place probability? ──
    print(f"  ── Model probability mass near each line ──")
    print(f"  (Fraction of legs with σ-distance < 1.0 — the 'clutch zone')")
    for spread_val in [1.5, 2.5, 3.5, 4.5]:
        rung_data = [d for d in data if d['spread'] == spread_val]
        clutch = [d for d in rung_data if abs(d['sigma_dist_poisson']) < 1.0]
        if not rung_data:
            continue
        pct = len(clutch) / len(rung_data) * 100
        clutch_wr = np.mean([d['result'] for d in clutch]) if clutch else 0
        non_clutch = [d for d in rung_data if abs(d['sigma_dist_poisson']) >= 1.0]
        nc_wr = np.mean([d['result'] for d in non_clutch]) if non_clutch else 0
        print(f"  +{spread_val}: {pct:.1f}% in clutch zone ({len(clutch)}/{len(rung_data)}), "
              f"clutch WR={clutch_wr:.3f}, non-clutch WR={nc_wr:.3f}, Δ={clutch_wr - nc_wr:+.3f}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} settled legs, {len(set(d['game_date'] for d in data))} days")
    print(f"WR: {np.mean([d['result'] for d in data]):.4f}")

    # Distribution of sigma-distance
    sigs = [d['sigma_dist_poisson'] for d in data]
    print(f"\nSigma-distance distribution:")
    print(f"  mean={np.mean(sigs):.3f}, median={np.median(sigs):.3f}, "
          f"std={np.std(sigs):.3f}")
    print(f"  min={np.min(sigs):.3f}, max={np.max(sigs):.3f}")
    print(f"  <0.5σ: {sum(1 for s in sigs if s < 0.5)} legs ({sum(1 for s in sigs if s < 0.5)/len(sigs)*100:.1f}%)")
    print(f"  0.5-1σ: {sum(1 for s in sigs if 0.5 <= s < 1.0)} legs")
    print(f"  1-2σ: {sum(1 for s in sigs if 1.0 <= s < 2.0)} legs")
    print(f"  >2σ: {sum(1 for s in sigs if s >= 2.0)} legs")

    test_margin_discrimination(data)
    test_sigma_sizing(data)
    test_weakest_leg(data)
    test_margin_lumpiness(data)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
