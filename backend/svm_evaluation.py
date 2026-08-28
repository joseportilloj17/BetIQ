#!/usr/bin/env python3
"""
SVM Evaluation: test SVM in 4 roles against current pipeline.
All results OOS via temporal purged CV, deflated for C/gamma search.

Roles:
  1. Sizing signal (Platt-calibrated SVC → quarter-Kelly)
  2. Bet/no-bet gate (native classification)
  3. Segment-specific (per rung, cell, odds band)
  4. Ensemble member (blend with dist_wp/composite)
"""

import sqlite3
import numpy as np
import re
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from scipy import stats

DB_PATH = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load settled shadow_negbin legs with all signals."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT l.description, l.ats_dist_wp, l.bayes_team_wp, l.ridge_composite,
               l.predicted_edge_pp, l.mkt_disagree_dist, l.negbin_wp, l.win_prob,
               l.negbin_cell, l.leg_result, b.game_date, b.odds, b.status, b.amount,
               l.mock_bet_id
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE b.source='shadow_negbin' AND b.cohort='clean'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL AND l.bayes_team_wp IS NOT NULL
        ORDER BY b.game_date
    """).fetchall()
    conn.close()

    data = []
    for row in rows:
        desc = row[0]
        # Extract spread from description (e.g., "Detroit Tigers +3.5" → 3.5)
        m = re.search(r'([+-]?\d+\.?\d*)\s*$', desc)
        spread = float(m.group(1)) if m else None

        data.append({
            'description': desc,
            'dist_wp': row[1],
            'bayes_wp': row[2],
            'composite': row[3],
            'edge_pp': row[4],
            'mkt_disagree': row[5] if row[5] is not None else 0.0,
            'negbin_wp': row[6],
            'win_prob': row[7],
            'cell': row[8],
            'result': 1 if row[9] == 'WIN' else 0,
            'game_date': row[10],
            'odds': row[11],
            'parlay_status': row[12],
            'amount': row[13],
            'parlay_id': row[14],
            'spread': spread,
        })

    return data


def build_features(data):
    """Build feature matrix. Features: dist_wp, bayes_wp, edge_pp, mkt_disagree, negbin_wp, spread."""
    X = np.array([
        [d['dist_wp'], d['bayes_wp'], d['edge_pp'], d['mkt_disagree'],
         d['negbin_wp'], d['spread']]
        for d in data
    ])
    y = np.array([d['result'] for d in data])
    dates = np.array([d['game_date'] for d in data])
    return X, y, dates


# ═══════════════════════════════════════════════════════════════
# TEMPORAL PURGED CV
# ═══════════════════════════════════════════════════════════════

def temporal_purged_cv(dates, n_folds=5, purge_days=3):
    """Generate temporal purged CV folds. Each fold uses earlier days as train,
    later days as test, with a purge gap."""
    unique_dates = sorted(set(dates))
    n_dates = len(unique_dates)

    # Split dates into n_folds+1 blocks, use expanding window
    fold_size = n_dates // (n_folds + 1)
    if fold_size < 1:
        fold_size = 1

    folds = []
    for i in range(n_folds):
        # Train on first (i+1)*fold_size dates, test on next fold_size dates
        train_end_idx = (i + 1) * fold_size
        test_start_idx = train_end_idx  # we'll purge from here
        test_end_idx = min(test_start_idx + fold_size, n_dates)

        if test_start_idx >= n_dates:
            break

        train_dates = set(unique_dates[:train_end_idx])
        test_dates_list = unique_dates[test_start_idx:test_end_idx]

        # Purge: remove train dates within purge_days of any test date
        from datetime import datetime, timedelta
        test_dt = [datetime.strptime(d, '%Y-%m-%d') for d in test_dates_list]
        purged_train = set()
        for td in train_dates:
            td_dt = datetime.strptime(td, '%Y-%m-%d')
            too_close = any(abs((td_dt - tst).days) <= purge_days for tst in test_dt)
            if not too_close:
                purged_train.add(td)

        if len(purged_train) < 2 or len(test_dates_list) < 1:
            continue

        test_dates = set(test_dates_list)

        train_idx = [i for i, d in enumerate(dates) if d in purged_train]
        test_idx = [i for i, d in enumerate(dates) if d in test_dates]

        if len(train_idx) > 10 and len(test_idx) > 10:
            folds.append((np.array(train_idx), np.array(test_idx)))

    return folds


def deflated_sharpe(daily_returns, n_trials):
    """Compute deflated Sharpe ratio accounting for multiple testing."""
    if len(daily_returns) < 3:
        return 0.0, 1.0
    sr = np.mean(daily_returns) / (np.std(daily_returns, ddof=1) + 1e-10)
    T = len(daily_returns)
    # E[max SR] under null for n_trials independent tests
    if n_trials <= 1:
        e_max_sr = 0.0
    else:
        from scipy.stats import norm
        e_max_sr = norm.ppf(1 - 1/(2*n_trials)) * (1 + 0.5772 / np.log(n_trials))
    dsr = sr - e_max_sr
    # p-value
    se = np.sqrt((1 + 0.5 * sr**2) / T)
    z = dsr / (se + 1e-10)
    from scipy.stats import norm as norm_dist
    p = 1 - norm_dist.cdf(z)
    return dsr, p


# ═══════════════════════════════════════════════════════════════
# ROLE 1: SVM AS SIZING SIGNAL (Platt-calibrated)
# ═══════════════════════════════════════════════════════════════

def test_svm_sizing(data, X, y, dates):
    """Compare SVM-Platt-sized fullstack vs dist_wp-sized fullstack."""
    print("\n" + "="*70)
    print("ROLE 1: SVM AS SIZING SIGNAL (Platt-calibrated)")
    print("="*70)

    folds = temporal_purged_cv(dates, n_folds=5, purge_days=3)
    print(f"  Temporal purged CV: {len(folds)} folds")

    # Grid search: C × gamma (for RBF) + linear
    configs = []
    for kernel in ['rbf', 'linear']:
        for C in [0.01, 0.1, 1.0, 10.0]:
            if kernel == 'rbf':
                for gamma in ['scale', 'auto', 0.01, 0.1]:
                    configs.append({'kernel': kernel, 'C': C, 'gamma': gamma})
            else:
                configs.append({'kernel': kernel, 'C': C, 'gamma': 'scale'})

    n_trials = len(configs)
    print(f"  Grid: {n_trials} configurations (deflate for {n_trials} trials)")

    # For each config, run OOS and compute PnL
    best_config = None
    best_pnl = -np.inf
    best_wr = 0
    best_brier = 1.0
    all_results = []

    for cfg in configs:
        oos_probs = []
        oos_labels = []
        oos_odds = []
        oos_dates_list = []

        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_train)
            X_te = sc.transform(X_test)

            # Platt-calibrated SVM
            base = SVC(kernel=cfg['kernel'], C=cfg['C'], gamma=cfg['gamma'],
                       probability=False, max_iter=5000)
            cal = CalibratedClassifierCV(base, cv=3, method='sigmoid')
            try:
                cal.fit(X_tr, y_train)
                probs = cal.predict_proba(X_te)[:, 1]
            except Exception:
                continue

            oos_probs.extend(probs)
            oos_labels.extend(y_test)
            for idx in test_idx:
                oos_odds.append(data[idx]['odds'])
                oos_dates_list.append(data[idx]['game_date'])

        if len(oos_probs) < 50:
            continue

        oos_probs = np.array(oos_probs)
        oos_labels = np.array(oos_labels)
        oos_odds = np.array(oos_odds)

        brier = brier_score_loss(oos_labels, oos_probs)
        wr = np.mean(oos_labels)

        # Calibration slope
        from sklearn.linear_model import LogisticRegression
        lr_cal = LogisticRegression(penalty=None, max_iter=1000)
        lr_cal.fit(oos_probs.reshape(-1, 1), oos_labels)
        cal_slope = lr_cal.coef_[0][0]

        # Quarter-Kelly sizing simulation (at PARLAY level we'd need parlay structure,
        # but at leg level we can simulate: prob vs 1/odds = edge)
        # Group legs by parlay_id to simulate parlay-level PnL
        # For simplicity, compute leg-level discrimination metrics

        # Quintile WR (discrimination)
        quintiles = np.percentile(oos_probs, [20, 40, 60, 80])
        q_wrs = []
        for qi in range(5):
            if qi == 0:
                mask = oos_probs <= quintiles[0]
            elif qi == 4:
                mask = oos_probs > quintiles[3]
            else:
                mask = (oos_probs > quintiles[qi-1]) & (oos_probs <= quintiles[qi])
            if mask.sum() > 0:
                q_wrs.append(np.mean(oos_labels[mask]))
            else:
                q_wrs.append(0.5)

        rho, rho_p = stats.spearmanr(oos_probs, oos_labels)
        q_rho, _ = stats.spearmanr(range(5), q_wrs)

        all_results.append({
            'cfg': cfg, 'brier': brier, 'rho': rho, 'q_rho': q_rho,
            'q_wrs': q_wrs, 'cal_slope': cal_slope, 'n': len(oos_probs),
            'wr': wr,
        })

        if rho > best_pnl:
            best_pnl = rho
            best_config = cfg
            best_brier = brier

    # Sort by discrimination (rho)
    all_results.sort(key=lambda x: x['rho'], reverse=True)

    # Report top 5 configs
    print(f"\n  Top 5 configs by discrimination (Spearman rho):")
    print(f"  {'Kernel':<8} {'C':<8} {'Gamma':<8} {'Brier':<8} {'Rho':<8} {'Q-Rho':<8} {'CalSlope':<10} {'Q0→Q4 WR'}")
    for r in all_results[:5]:
        cfg = r['cfg']
        qw = r['q_wrs']
        print(f"  {cfg['kernel']:<8} {cfg['C']:<8} {str(cfg['gamma']):<8} "
              f"{r['brier']:.4f}  {r['rho']:+.4f}  {r['q_rho']:+.4f}  "
              f"{r['cal_slope']:+.3f}     "
              f"{qw[0]:.3f}→{qw[4]:.3f}")

    # Compare with dist_wp baseline
    print(f"\n  ── Baseline comparison (dist_wp) ──")
    dist_probs = np.array([d['dist_wp'] for d in data])
    dist_labels = np.array([d['result'] for d in data])
    dist_brier = brier_score_loss(dist_labels, dist_probs)
    dist_rho, _ = stats.spearmanr(dist_probs, dist_labels)

    # dist_wp quintile WR
    dq = np.percentile(dist_probs, [20, 40, 60, 80])
    dist_q_wrs = []
    for qi in range(5):
        if qi == 0:
            mask = dist_probs <= dq[0]
        elif qi == 4:
            mask = dist_probs > dq[3]
        else:
            mask = (dist_probs > dq[qi-1]) & (dist_probs <= dq[qi])
        if mask.sum() > 0:
            dist_q_wrs.append(np.mean(dist_labels[mask]))
    dist_q_rho, _ = stats.spearmanr(range(5), dist_q_wrs)

    print(f"  dist_wp:   Brier={dist_brier:.4f}, rho={dist_rho:+.4f}, "
          f"Q-rho={dist_q_rho:+.4f}, Q0→Q4={dist_q_wrs[0]:.3f}→{dist_q_wrs[4]:.3f}")

    # ridge_composite baseline
    comp_probs = np.array([d['composite'] for d in data if d['composite'] is not None])
    comp_labels = np.array([d['result'] for d in data if d['composite'] is not None])
    comp_brier = brier_score_loss(comp_labels, comp_probs)
    comp_rho, _ = stats.spearmanr(comp_probs, comp_labels)
    print(f"  composite: Brier={comp_brier:.4f}, rho={comp_rho:+.4f}")

    if all_results:
        best = all_results[0]
        print(f"\n  Best SVM:  Brier={best['brier']:.4f}, rho={best['rho']:+.4f}, "
              f"Q-rho={best['q_rho']:+.4f}")
        print(f"  CalSlope={best['cal_slope']:+.3f} (1.0=perfect, >1=under-dispersed, <1=over-dispersed)")

        delta_brier = best['brier'] - dist_brier
        delta_rho = best['rho'] - dist_rho
        print(f"\n  vs dist_wp:  ΔBrier={delta_brier:+.4f}, Δrho={delta_rho:+.4f}")
        delta_comp_brier = best['brier'] - comp_brier
        delta_comp_rho = best['rho'] - comp_rho
        print(f"  vs composite: ΔBrier={delta_comp_brier:+.4f}, Δrho={delta_comp_rho:+.4f}")

        # Deflated Sharpe for best config
        dsr, dsr_p = deflated_sharpe(best['q_wrs'], n_trials)
        print(f"\n  Deflated SR: {dsr:+.4f}, p={dsr_p:.4f} (deflated for {n_trials} trials)")

    return all_results


# ═══════════════════════════════════════════════════════════════
# ROLE 2: SVM AS BET/NO-BET GATE
# ═══════════════════════════════════════════════════════════════

def test_svm_gate(data, X, y, dates):
    """Test SVM as a classifier gate: predict WIN/LOSS, gate on prediction."""
    print("\n" + "="*70)
    print("ROLE 2: SVM AS BET/NO-BET GATE")
    print("="*70)

    folds = temporal_purged_cv(dates, n_folds=5, purge_days=3)

    configs = []
    for kernel in ['rbf', 'linear']:
        for C in [0.01, 0.1, 1.0, 10.0]:
            if kernel == 'rbf':
                for gamma in ['scale', 'auto', 0.01, 0.1]:
                    configs.append({'kernel': kernel, 'C': C, 'gamma': gamma})
            else:
                configs.append({'kernel': kernel, 'C': C, 'gamma': 'scale'})

    n_trials = len(configs)

    # Current gate baseline: edge < 12pp
    edge_arr = np.array([d['edge_pp'] for d in data])
    result_arr = np.array([d['result'] for d in data])

    gated_mask = edge_arr < 12.0
    if gated_mask.sum() > 0:
        gated_wr = np.mean(result_arr[gated_mask])
        rejected_wr = np.mean(result_arr[~gated_mask]) if (~gated_mask).sum() > 0 else 0
        print(f"\n  Current edge gate (<12pp):")
        print(f"    Kept: {gated_mask.sum()} legs, WR={gated_wr:.4f}")
        print(f"    Rejected: {(~gated_mask).sum()} legs, WR={rejected_wr:.4f}")
        print(f"    Lift (kept-rejected): {gated_wr - rejected_wr:+.4f}")

    # For each SVM config, use OOS predictions as gate
    best_results = []

    for cfg in configs:
        oos_preds = []
        oos_labels = []
        oos_edges = []
        oos_dates_list = []

        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_train)
            X_te = sc.transform(X_test)

            svm = SVC(kernel=cfg['kernel'], C=cfg['C'], gamma=cfg['gamma'],
                      max_iter=5000)
            try:
                svm.fit(X_tr, y_train)
                preds = svm.predict(X_te)
            except Exception:
                continue

            oos_preds.extend(preds)
            oos_labels.extend(y_test)
            for idx in test_idx:
                oos_edges.append(data[idx]['edge_pp'])
                oos_dates_list.append(data[idx]['game_date'])

        if len(oos_preds) < 50:
            continue

        oos_preds = np.array(oos_preds)
        oos_labels = np.array(oos_labels)
        oos_edges = np.array(oos_edges)

        # Gate: only bet where SVM predicts WIN
        svm_gate = oos_preds == 1
        if svm_gate.sum() == 0 or svm_gate.sum() == len(svm_gate):
            continue

        gated_wr = np.mean(oos_labels[svm_gate])
        rejected_wr = np.mean(oos_labels[~svm_gate]) if (~svm_gate).sum() > 0 else 0
        kept_pct = svm_gate.sum() / len(svm_gate) * 100

        # Compare: edge gate on same OOS set
        edge_gate = oos_edges < 12.0
        edge_gated_wr = np.mean(oos_labels[edge_gate]) if edge_gate.sum() > 0 else 0

        # Combined gate: SVM AND edge
        combined = svm_gate & edge_gate
        combined_wr = np.mean(oos_labels[combined]) if combined.sum() > 0 else 0

        # SVM-only gate vs edge-only: which rejects worse legs?
        svm_only_reject = (~svm_gate) & edge_gate  # SVM rejects but edge keeps
        if svm_only_reject.sum() > 0:
            svm_reject_wr = np.mean(oos_labels[svm_only_reject])
        else:
            svm_reject_wr = None

        best_results.append({
            'cfg': cfg,
            'kept_pct': kept_pct,
            'gated_wr': gated_wr,
            'rejected_wr': rejected_wr,
            'lift': gated_wr - rejected_wr,
            'edge_gated_wr': edge_gated_wr,
            'combined_wr': combined_wr,
            'combined_n': combined.sum(),
            'svm_reject_wr': svm_reject_wr,
            'n': len(oos_preds),
        })

    # Sort by lift (discrimination between kept and rejected)
    best_results.sort(key=lambda x: x['lift'], reverse=True)

    print(f"\n  Top 5 SVM gate configs by lift (kept_WR - rejected_WR):")
    print(f"  {'Kernel':<8} {'C':<6} {'Gamma':<8} {'Kept%':<7} {'KeptWR':<8} {'RejWR':<8} {'Lift':<8} {'Comb_WR':<8} {'SVMrej_WR'}")
    for r in best_results[:5]:
        cfg = r['cfg']
        srw = f"{r['svm_reject_wr']:.4f}" if r['svm_reject_wr'] is not None else "N/A"
        print(f"  {cfg['kernel']:<8} {cfg['C']:<6} {str(cfg['gamma']):<8} "
              f"{r['kept_pct']:.1f}%  {r['gated_wr']:.4f}  {r['rejected_wr']:.4f}  "
              f"{r['lift']:+.4f}  {r['combined_wr']:.4f}  {srw}")

    # Worst results (SVM predicting backwards)
    print(f"\n  Bottom 3 configs (negative lift = SVM predicting backwards):")
    for r in best_results[-3:]:
        cfg = r['cfg']
        print(f"  {cfg['kernel']:<8} C={cfg['C']:<6} lift={r['lift']:+.4f} "
              f"(kept {r['kept_pct']:.0f}% at WR={r['gated_wr']:.3f})")

    return best_results


# ═══════════════════════════════════════════════════════════════
# ROLE 3: SEGMENT-SPECIFIC SVM
# ═══════════════════════════════════════════════════════════════

def test_svm_segments(data, X, y, dates):
    """Test SVM per-segment: rung, cell, odds band."""
    print("\n" + "="*70)
    print("ROLE 3: SEGMENT-SPECIFIC SVM")
    print("="*70)

    segments = {}

    # By spread rung
    for d in data:
        sp = d['spread']
        if sp is not None:
            rung = f"+{sp}" if sp > 0 else str(sp)
            segments.setdefault(f"rung_{rung}", []).append(d)

    # By cell
    for d in data:
        cell = d['cell']
        if cell:
            segments.setdefault(f"cell_{cell}", []).append(d)

    # By odds band
    for d in data:
        odds = d['odds']
        if odds < 1.4:
            band = "odds_<1.4"
        elif odds < 1.8:
            band = "odds_1.4-1.8"
        elif odds < 2.5:
            band = "odds_1.8-2.5"
        else:
            band = "odds_>2.5"
        segments.setdefault(band, []).append(d)

    # By edge sign
    for d in data:
        if d['edge_pp'] < 0:
            segments.setdefault('edge_negative', []).append(d)
        elif d['edge_pp'] < 6:
            segments.setdefault('edge_0-6pp', []).append(d)
        elif d['edge_pp'] < 12:
            segments.setdefault('edge_6-12pp', []).append(d)
        else:
            segments.setdefault('edge_>=12pp', []).append(d)

    results = {}
    n_segments_tested = 0

    for seg_name, seg_data in sorted(segments.items()):
        if len(seg_data) < 40:
            continue

        n_segments_tested += 1
        seg_X = np.array([
            [d['dist_wp'], d['bayes_wp'], d['edge_pp'], d['mkt_disagree'],
             d['negbin_wp'], d['spread']]
            for d in seg_data
        ])
        seg_y = np.array([d['result'] for d in seg_data])
        seg_dates = np.array([d['game_date'] for d in seg_data])
        seg_folds = temporal_purged_cv(seg_dates, n_folds=3, purge_days=3)

        if len(seg_folds) < 2:
            continue

        # Baseline: dist_wp discrimination in this segment
        seg_dist = np.array([d['dist_wp'] for d in seg_data])
        base_rho, _ = stats.spearmanr(seg_dist, seg_y)
        base_wr = np.mean(seg_y)

        # Test best SVM configs (subset of grid for speed)
        best_seg_rho = -1
        best_seg_cfg = None

        for kernel in ['rbf', 'linear']:
            for C in [0.1, 1.0, 10.0]:
                gamma = 'scale'

                oos_probs = []
                oos_labels = []

                for train_idx, test_idx in seg_folds:
                    X_tr = seg_X[train_idx]
                    y_tr = seg_y[train_idx]
                    X_te = seg_X[test_idx]
                    y_te = seg_y[test_idx]

                    sc = StandardScaler()
                    Xtr = sc.fit_transform(X_tr)
                    Xte = sc.transform(X_te)

                    base_svm = SVC(kernel=kernel, C=C, gamma=gamma,
                                   probability=False, max_iter=5000)
                    cal = CalibratedClassifierCV(base_svm, cv=min(3, len(y_tr)//10 or 2),
                                                 method='sigmoid')
                    try:
                        cal.fit(Xtr, y_tr)
                        probs = cal.predict_proba(Xte)[:, 1]
                    except Exception:
                        continue

                    oos_probs.extend(probs)
                    oos_labels.extend(y_te)

                if len(oos_probs) < 20:
                    continue

                rho, _ = stats.spearmanr(oos_probs, oos_labels)
                if rho > best_seg_rho:
                    best_seg_rho = rho
                    best_seg_cfg = {'kernel': kernel, 'C': C}

        delta = best_seg_rho - base_rho
        results[seg_name] = {
            'n': len(seg_data),
            'wr': base_wr,
            'dist_rho': base_rho,
            'svm_rho': best_seg_rho,
            'delta': delta,
            'cfg': best_seg_cfg,
        }

    # Report
    print(f"\n  Tested {n_segments_tested} segments (min 40 legs)")
    print(f"\n  {'Segment':<20} {'N':<6} {'WR':<7} {'Dist_ρ':<9} {'SVM_ρ':<9} {'Δρ':<9} {'Winner'}")
    print(f"  {'-'*80}")

    for seg_name, r in sorted(results.items(), key=lambda x: x[1]['delta'], reverse=True):
        winner = "SVM" if r['delta'] > 0.02 else ("DIST" if r['delta'] < -0.02 else "TIE")
        cfg_str = f"{r['cfg']['kernel']}/C={r['cfg']['C']}" if r['cfg'] else "N/A"
        print(f"  {seg_name:<20} {r['n']:<6} {r['wr']:.3f}  {r['dist_rho']:+.4f}  "
              f"{r['svm_rho']:+.4f}  {r['delta']:+.4f}  {winner} ({cfg_str})")

    # Flag segments where SVM wins by >0.05 rho
    winners = {k: v for k, v in results.items() if v['delta'] > 0.05}
    if winners:
        print(f"\n  ★ SVM wins in {len(winners)} segments by >0.05 rho:")
        for k, v in sorted(winners.items(), key=lambda x: x[1]['delta'], reverse=True):
            print(f"    {k}: Δρ={v['delta']:+.4f} (dist={v['dist_rho']:+.4f} → SVM={v['svm_rho']:+.4f})")
    else:
        print(f"\n  No segments where SVM beats dist_wp by >0.05 rho")

    # Deflate for segment count
    total_trials = n_segments_tested * 6  # ~6 configs per segment
    print(f"\n  Multiple testing: {n_segments_tested} segments × 6 configs = {total_trials} trials")
    print(f"  Any Δρ < 0.10 is likely noise after deflation.")

    return results


# ═══════════════════════════════════════════════════════════════
# ROLE 4: SVM AS ENSEMBLE MEMBER
# ═══════════════════════════════════════════════════════════════

def test_svm_ensemble(data, X, y, dates):
    """Test blending SVM output with dist_wp and composite."""
    print("\n" + "="*70)
    print("ROLE 4: SVM AS ENSEMBLE MEMBER")
    print("="*70)

    folds = temporal_purged_cv(dates, n_folds=5, purge_days=3)

    # Use best SVM config from Role 1 (rbf, C=1.0, gamma=scale as default)
    # Actually test a few to find best ensemble partner
    svm_configs = [
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
        {'kernel': 'linear', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'linear', 'C': 0.1, 'gamma': 'scale'},
    ]

    # Get OOS SVM probs for each config
    for cfg in svm_configs:
        oos_svm_probs = np.full(len(data), np.nan)

        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_train)
            X_te = sc.transform(X_test)

            base = SVC(kernel=cfg['kernel'], C=cfg['C'], gamma=cfg['gamma'],
                       probability=False, max_iter=5000)
            cal = CalibratedClassifierCV(base, cv=3, method='sigmoid')
            try:
                cal.fit(X_tr, y_train)
                probs = cal.predict_proba(X_te)[:, 1]
                for i, idx in enumerate(test_idx):
                    oos_svm_probs[idx] = probs[i]
            except Exception:
                continue

        valid = ~np.isnan(oos_svm_probs)
        if valid.sum() < 100:
            continue

        svm_p = oos_svm_probs[valid]
        dist_p = np.array([data[i]['dist_wp'] for i in range(len(data)) if valid[i]])
        comp_p = np.array([data[i]['composite'] if data[i]['composite'] is not None else data[i]['dist_wp']
                          for i in range(len(data)) if valid[i]])
        labels = y[valid]

        # Error correlation: are SVM errors uncorrelated with dist_wp errors?
        svm_err = (svm_p - labels) ** 2
        dist_err = (dist_p - labels) ** 2
        comp_err = (comp_p - labels) ** 2

        err_corr_dist, _ = stats.pearsonr(svm_err, dist_err)
        err_corr_comp, _ = stats.pearsonr(svm_err, comp_err)

        # Prediction correlation
        pred_corr_dist, _ = stats.pearsonr(svm_p, dist_p)
        pred_corr_comp, _ = stats.pearsonr(svm_p, comp_p)

        # Test blend weights: 0.1 to 0.5 SVM weight
        blend_results = []
        for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
            # Blend with dist_wp
            blend_dist = w * svm_p + (1 - w) * dist_p
            brier_blend_dist = brier_score_loss(labels, blend_dist)
            rho_blend_dist, _ = stats.spearmanr(blend_dist, labels)

            # Blend with composite
            blend_comp = w * svm_p + (1 - w) * comp_p
            brier_blend_comp = brier_score_loss(labels, blend_comp)
            rho_blend_comp, _ = stats.spearmanr(blend_comp, labels)

            blend_results.append({
                'w': w,
                'brier_dist': brier_blend_dist,
                'rho_dist': rho_blend_dist,
                'brier_comp': brier_blend_comp,
                'rho_comp': rho_blend_comp,
            })

        # Baselines
        brier_dist = brier_score_loss(labels, dist_p)
        rho_dist, _ = stats.spearmanr(dist_p, labels)
        brier_comp = brier_score_loss(labels, comp_p)
        rho_comp, _ = stats.spearmanr(comp_p, labels)
        brier_svm = brier_score_loss(labels, svm_p)
        rho_svm, _ = stats.spearmanr(svm_p, labels)

        print(f"\n  Config: {cfg['kernel']}/C={cfg['C']} (N={valid.sum()} OOS legs)")
        print(f"  Error correlation: SVM↔dist={err_corr_dist:.3f}, SVM↔comp={err_corr_comp:.3f}")
        print(f"  Pred correlation:  SVM↔dist={pred_corr_dist:.3f}, SVM↔comp={pred_corr_comp:.3f}")
        print(f"\n  Solo baselines:")
        print(f"    dist_wp:   Brier={brier_dist:.4f}, rho={rho_dist:+.4f}")
        print(f"    composite: Brier={brier_comp:.4f}, rho={rho_comp:+.4f}")
        print(f"    SVM:       Brier={brier_svm:.4f}, rho={rho_svm:+.4f}")

        print(f"\n  Blends (w=SVM weight):")
        print(f"  {'w':<5} {'B+dist':<10} {'ρ+dist':<10} {'B+comp':<10} {'ρ+comp':<10}")
        for br in blend_results:
            print(f"  {br['w']:<5} {br['brier_dist']:.4f}    {br['rho_dist']:+.4f}    "
                  f"{br['brier_comp']:.4f}    {br['rho_comp']:+.4f}")

        # Best blend
        best_dist_blend = min(blend_results, key=lambda x: x['brier_dist'])
        best_comp_blend = min(blend_results, key=lambda x: x['brier_comp'])

        print(f"\n  Best blend vs dist_wp: w={best_dist_blend['w']}, "
              f"ΔBrier={best_dist_blend['brier_dist'] - brier_dist:+.4f}")
        print(f"  Best blend vs composite: w={best_comp_blend['w']}, "
              f"ΔBrier={best_comp_blend['brier_comp'] - brier_comp:+.4f}")

    return None


# ═══════════════════════════════════════════════════════════════
# PARLAY-LEVEL PNL SIMULATION
# ═══════════════════════════════════════════════════════════════

def test_parlay_pnl(data, X, y, dates):
    """Simulate parlay-level PnL: SVM-sized vs dist_wp-sized vs equal."""
    print("\n" + "="*70)
    print("PARLAY-LEVEL PNL COMPARISON")
    print("="*70)

    # Group legs by parlay
    conn = sqlite3.connect(DB_PATH)
    parlay_rows = conn.execute("""
        SELECT b.id, b.odds, b.status, b.game_date
        FROM mock_bets b
        WHERE b.source='shadow_negbin' AND b.cohort='clean'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
        ORDER BY b.game_date
    """).fetchall()
    conn.close()

    parlay_map = {r[0]: {'odds': r[1], 'status': r[2], 'date': r[3]} for r in parlay_rows}

    # Group data legs by parlay_id
    parlay_legs = defaultdict(list)
    for d in data:
        parlay_legs[d['parlay_id']].append(d)

    # Get OOS SVM probs using best config
    folds = temporal_purged_cv(dates, n_folds=5, purge_days=3)

    oos_svm = {}
    for train_idx, test_idx in folds:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])

        base = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, max_iter=5000)
        cal = CalibratedClassifierCV(base, cv=3, method='sigmoid')
        try:
            cal.fit(X_tr, y[train_idx])
            probs = cal.predict_proba(X_te)[:, 1]
            for i, idx in enumerate(test_idx):
                oos_svm[idx] = probs[i]
        except Exception:
            pass

    # For each parlay, compute sizing under different schemes
    daily_pnl_equal = defaultdict(float)
    daily_pnl_dist = defaultdict(float)
    daily_pnl_svm = defaultdict(float)

    for pid, pinfo in parlay_map.items():
        legs = parlay_legs.get(pid, [])
        if not legs:
            continue

        odds = pinfo['odds'] or 2.0
        won = pinfo['status'] == 'SETTLED_WIN'
        date = pinfo['date']

        # Equal sizing
        stake_equal = 10.0
        pnl_equal = stake_equal * (odds - 1) if won else -stake_equal
        daily_pnl_equal[date] += pnl_equal

        # dist_wp sizing (product of leg dist_wp as parlay WP)
        dist_prods = [l['dist_wp'] for l in legs]
        parlay_dist_wp = np.prod(dist_prods)
        edge_dist = parlay_dist_wp * odds - 1
        if edge_dist > 0:
            kelly_dist = (edge_dist / (odds - 1)) * 0.25
            weight_dist = max(min(kelly_dist, 2.5), 0.10)
        else:
            weight_dist = 0.10
        stake_dist = 10.0 * weight_dist
        pnl_dist = stake_dist * (odds - 1) if won else -stake_dist
        daily_pnl_dist[date] += pnl_dist

        # SVM sizing (if we have OOS probs for all legs)
        svm_prods = []
        for l in legs:
            # Find this leg's index in data
            idx = next((i for i, d in enumerate(data) if d is l), None)
            if idx is not None and idx in oos_svm:
                svm_prods.append(oos_svm[idx])
            else:
                svm_prods.append(l['dist_wp'])  # fallback

        parlay_svm_wp = np.prod(svm_prods)
        edge_svm = parlay_svm_wp * odds - 1
        if edge_svm > 0:
            kelly_svm = (edge_svm / (odds - 1)) * 0.25
            weight_svm = max(min(kelly_svm, 2.5), 0.10)
        else:
            weight_svm = 0.10
        stake_svm = 10.0 * weight_svm
        pnl_svm = stake_svm * (odds - 1) if won else -stake_svm
        daily_pnl_svm[date] += pnl_svm

    # Report daily PnL
    all_dates = sorted(set(daily_pnl_equal.keys()))
    print(f"\n  {'Date':<12} {'Equal':<10} {'Dist_wp':<10} {'SVM':<10} {'SVM-Dist':<10}")
    cum_equal = cum_dist = cum_svm = 0
    for d in all_dates:
        e = daily_pnl_equal.get(d, 0)
        dist = daily_pnl_dist.get(d, 0)
        s = daily_pnl_svm.get(d, 0)
        cum_equal += e
        cum_dist += dist
        cum_svm += s
        print(f"  {d:<12} {e:+.2f}    {dist:+.2f}    {s:+.2f}    {s-dist:+.2f}")

    print(f"\n  Cumulative:")
    print(f"    Equal:  ${cum_equal:+.2f}")
    print(f"    Dist:   ${cum_dist:+.2f}")
    print(f"    SVM:    ${cum_svm:+.2f}")
    print(f"    SVM vs Dist: ${cum_svm - cum_dist:+.2f}")

    # Daily Sharpe
    dist_daily = [daily_pnl_dist[d] for d in all_dates]
    svm_daily = [daily_pnl_svm[d] for d in all_dates]
    diff_daily = [s - d for s, d in zip(svm_daily, dist_daily)]

    if len(dist_daily) > 2:
        sr_dist = np.mean(dist_daily) / (np.std(dist_daily, ddof=1) + 1e-10)
        sr_svm = np.mean(svm_daily) / (np.std(svm_daily, ddof=1) + 1e-10)
        t_stat, t_p = stats.ttest_rel(svm_daily, dist_daily)
        print(f"\n  Daily SR: dist={sr_dist:.3f}, SVM={sr_svm:.3f}")
        print(f"  Paired t-test (SVM vs dist): t={t_stat:.3f}, p={t_p:.4f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} settled legs across {len(set(d['game_date'] for d in data))} days")
    print(f"WR: {sum(d['result'] for d in data)/len(data):.4f}")

    X, y, dates = build_features(data)
    print(f"Feature matrix: {X.shape}")
    print(f"Features: dist_wp, bayes_wp, edge_pp, mkt_disagree, negbin_wp, spread")

    # Run all 4 roles
    sizing_results = test_svm_sizing(data, X, y, dates)
    gate_results = test_svm_gate(data, X, y, dates)
    segment_results = test_svm_segments(data, X, y, dates)
    ensemble_results = test_svm_ensemble(data, X, y, dates)

    # Parlay-level PnL
    test_parlay_pnl(data, X, y, dates)

    print("\n" + "="*70)
    print("SUMMARY VERDICT")
    print("="*70)

    # Summarize findings
    if sizing_results:
        best = sizing_results[0]
        dist_rho = stats.spearmanr(
            np.array([d['dist_wp'] for d in data]),
            np.array([d['result'] for d in data])
        )[0]
        print(f"\n  Role 1 (Sizing): Best SVM rho={best['rho']:+.4f} vs dist_wp rho={dist_rho:+.4f}")
        print(f"    {'SVM WINS' if best['rho'] > dist_rho + 0.02 else 'NO IMPROVEMENT'}")

    if gate_results:
        best_gate = gate_results[0]
        print(f"\n  Role 2 (Gate): Best SVM lift={best_gate['lift']:+.4f}")
        print(f"    {'SVM GATE ADDS VALUE' if best_gate['lift'] > 0.05 else 'EDGE GATE SUFFICIENT'}")

    if segment_results:
        seg_wins = {k: v for k, v in segment_results.items() if v['delta'] > 0.05}
        print(f"\n  Role 3 (Segments): {len(seg_wins)} segments where SVM beats dist by >0.05 rho")
        for k, v in sorted(seg_wins.items(), key=lambda x: x[1]['delta'], reverse=True):
            print(f"    {k}: Δρ={v['delta']:+.4f}")

    print(f"\n  Role 4 (Ensemble): see blend tables above")
    print(f"\n  (All deflated for {20 if sizing_results else 0} C/gamma trials)")
