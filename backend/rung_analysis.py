#!/usr/bin/env python3
"""
Rung granularity analysis: inter-rung edge concentration + intra-rung conditional signal.
OOS temporal purged CV, deflated for segments tested.
"""

import sqlite3
import numpy as np
import re
from collections import defaultdict
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

DB_PATH = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT l.description, l.ats_dist_wp, l.bayes_team_wp, l.ridge_composite,
               l.predicted_edge_pp, l.mkt_disagree_dist, l.negbin_wp, l.win_prob,
               l.negbin_cell, l.negbin_mu_home, l.negbin_mu_away,
               l.leg_result, b.game_date, b.odds, b.status, b.amount, b.id as parlay_id
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
        m = re.search(r'([+-]?\d+\.?\d*)\s*$', desc)
        spread = float(m.group(1)) if m else None
        if spread is None:
            continue

        mu_home = row[9] if row[9] is not None else 0
        mu_away = row[10] if row[10] is not None else 0
        game_total = mu_home + mu_away
        # Blowout propensity: how far apart are the mus?
        mu_diff = abs(mu_home - mu_away)
        # Favorite strength proxy: the higher mu
        fav_strength = max(mu_home, mu_away)

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
            'mu_home': mu_home,
            'mu_away': mu_away,
            'game_total': game_total,
            'mu_diff': mu_diff,
            'fav_strength': fav_strength,
            'result': 1 if row[11] == 'WIN' else 0,
            'game_date': row[12],
            'odds': row[13],
            'parlay_status': row[14],
            'amount': row[15],
            'parlay_id': row[16],
            'spread': spread,
            'rung': f"+{spread}" if spread > 0 else str(spread),
        })

    return data


def temporal_purged_cv(dates_arr, n_folds=5, purge_days=3):
    unique_dates = sorted(set(dates_arr))
    n_dates = len(unique_dates)
    fold_size = max(n_dates // (n_folds + 1), 1)

    folds = []
    for i in range(n_folds):
        train_end_idx = (i + 1) * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_dates)
        if test_start_idx >= n_dates:
            break

        train_dates = set(unique_dates[:train_end_idx])
        test_dates_list = unique_dates[test_start_idx:test_end_idx]
        test_dt = [datetime.strptime(d, '%Y-%m-%d') for d in test_dates_list]

        purged_train = set()
        for td in train_dates:
            td_dt = datetime.strptime(td, '%Y-%m-%d')
            if not any(abs((td_dt - tst).days) <= purge_days for tst in test_dt):
                purged_train.add(td)

        if len(purged_train) < 2 or len(test_dates_list) < 1:
            continue

        test_dates = set(test_dates_list)
        train_idx = [i for i, d in enumerate(dates_arr) if d in purged_train]
        test_idx = [i for i, d in enumerate(dates_arr) if d in test_dates]

        if len(train_idx) > 10 and len(test_idx) > 10:
            folds.append((np.array(train_idx), np.array(test_idx)))

    return folds


# ═══════════════════════════════════════════════════════════════
# PART 1: INTER-RUNG — Edge concentration across rungs
# ═══════════════════════════════════════════════════════════════

def inter_rung_analysis(data):
    print("=" * 70)
    print("PART 1: INTER-RUNG EDGE CONCENTRATION")
    print("=" * 70)

    # Group by rung
    by_rung = defaultdict(list)
    for d in data:
        by_rung[d['rung']].append(d)

    print(f"\n  {'Rung':<8} {'N':<6} {'WR':<7} {'AvgEdge':<9} {'GateRej%':<10} "
          f"{'AvgOdds':<9} {'AvgDist':<9} {'AvgBayes':<9}")
    print(f"  {'-'*75}")

    rung_stats = {}
    for rung in sorted(by_rung.keys(), key=lambda x: float(x)):
        legs = by_rung[rung]
        n = len(legs)
        wr = np.mean([l['result'] for l in legs])
        avg_edge = np.mean([l['edge_pp'] for l in legs])
        gate_rej = np.mean([1 if l['edge_pp'] >= 12.0 else 0 for l in legs]) * 100
        avg_odds = np.mean([l['odds'] for l in legs])
        avg_dist = np.mean([l['dist_wp'] for l in legs])
        avg_bayes = np.mean([l['bayes_wp'] for l in legs])

        print(f"  {rung:<8} {n:<6} {wr:.3f}  {avg_edge:+.2f}pp  {gate_rej:.1f}%     "
              f"{avg_odds:.3f}    {avg_dist:.3f}    {avg_bayes:.3f}")

        rung_stats[rung] = {
            'n': n, 'wr': wr, 'avg_edge': avg_edge, 'gate_rej_pct': gate_rej,
            'avg_odds': avg_odds, 'avg_dist': avg_dist
        }

    # ── Per-rung realized ROI (at parlay level, need to compute leg-contribution) ──
    # Since legs are in parlays, compute per-rung PnL contribution as:
    # leg_roi = (parlay_odds - 1) if parlay won, else -1, attributed equally to each leg
    # But we don't have parlay-level win/loss per leg easily. Use leg WR as proxy.
    print(f"\n  ── Rung-level WR after edge gate (<12pp) ──")
    print(f"  {'Rung':<8} {'N_kept':<8} {'WR_kept':<10} {'N_rej':<8} {'WR_rej':<10} {'Lift':<8}")
    for rung in sorted(by_rung.keys(), key=lambda x: float(x)):
        legs = by_rung[rung]
        kept = [l for l in legs if l['edge_pp'] < 12.0]
        rej = [l for l in legs if l['edge_pp'] >= 12.0]
        wr_k = np.mean([l['result'] for l in kept]) if kept else 0
        wr_r = np.mean([l['result'] for l in rej]) if rej else 0
        lift = wr_k - wr_r if rej else 0
        print(f"  {rung:<8} {len(kept):<8} {wr_k:.3f}     {len(rej):<8} "
              f"{wr_r:.3f}     {lift:+.3f}")

    # ── Is edge rung-dependent after controlling for odds? ──
    print(f"\n  ── Rung effect controlling for odds (logistic regression) ──")
    # Encode rung as numeric, fit logistic: result ~ rung + odds + dist_wp
    rung_numeric = np.array([d['spread'] for d in data])
    odds_arr = np.array([d['odds'] for d in data])
    dist_arr = np.array([d['dist_wp'] for d in data])
    y = np.array([d['result'] for d in data])

    # Model 1: result ~ dist_wp + odds (no rung)
    X1 = np.column_stack([dist_arr, odds_arr])
    lr1 = LogisticRegression(penalty=None, max_iter=5000)
    lr1.fit(X1, y)
    pred1 = lr1.predict_proba(X1)[:, 1]
    brier1 = brier_score_loss(y, pred1)

    # Model 2: result ~ dist_wp + odds + rung
    X2 = np.column_stack([dist_arr, odds_arr, rung_numeric])
    lr2 = LogisticRegression(penalty=None, max_iter=5000)
    lr2.fit(X2, y)
    pred2 = lr2.predict_proba(X2)[:, 1]
    brier2 = brier_score_loss(y, pred2)

    # Model 3: result ~ dist_wp + odds + rung + rung^2 (nonlinear rung effect)
    X3 = np.column_stack([dist_arr, odds_arr, rung_numeric, rung_numeric**2])
    lr3 = LogisticRegression(penalty=None, max_iter=5000)
    lr3.fit(X3, y)
    pred3 = lr3.predict_proba(X3)[:, 1]
    brier3 = brier_score_loss(y, pred3)

    print(f"  Model (dist+odds):          Brier={brier1:.4f}")
    print(f"  Model (dist+odds+rung):     Brier={brier2:.4f}  ΔBrier={brier2-brier1:+.4f}")
    print(f"  Model (dist+odds+rung+rg²): Brier={brier3:.4f}  ΔBrier={brier3-brier1:+.4f}")
    print(f"  Rung coef (linear): {lr2.coef_[0][2]:+.4f}")
    print(f"  Rung coef (in quadratic): linear={lr3.coef_[0][2]:+.4f}, quad={lr3.coef_[0][3]:+.4f}")

    # ── OOS test: rung-weighted vs uniform ──
    print(f"\n  ── OOS: rung-weighted vs uniform sizing ──")
    dates_arr = np.array([d['game_date'] for d in data])
    folds = temporal_purged_cv(dates_arr, n_folds=5, purge_days=3)

    # Strategy 1: equal weight all legs
    # Strategy 2: weight by rung (proportional to in-sample WR per rung)
    # Strategy 3: exclude +1.5 entirely (gate-reject-heavy rung)

    daily_pnl_uniform = defaultdict(float)
    daily_pnl_rung_wt = defaultdict(float)
    daily_pnl_no15 = defaultdict(float)

    for train_idx, test_idx in folds:
        # Learn rung weights from train
        train_rung_wr = defaultdict(lambda: {'wins': 0, 'total': 0})
        for i in train_idx:
            r = data[i]['rung']
            train_rung_wr[r]['total'] += 1
            train_rung_wr[r]['wins'] += data[i]['result']

        rung_weights = {}
        for r, v in train_rung_wr.items():
            rung_weights[r] = v['wins'] / v['total'] if v['total'] > 0 else 0.5

        # Normalize weights so mean = 1
        avg_w = np.mean(list(rung_weights.values()))
        for r in rung_weights:
            rung_weights[r] /= avg_w

        # Apply to test
        for i in test_idx:
            d = data[i]
            date = d['game_date']
            won = d['result']
            pnl = 1.0 if won else -1.0

            daily_pnl_uniform[date] += pnl
            rw = rung_weights.get(d['rung'], 1.0)
            daily_pnl_rung_wt[date] += pnl * rw
            if d['spread'] != 1.5:
                daily_pnl_no15[date] += pnl

    all_dates = sorted(set(daily_pnl_uniform.keys()))
    cum_u = cum_r = cum_n = 0
    for d in all_dates:
        cum_u += daily_pnl_uniform.get(d, 0)
        cum_r += daily_pnl_rung_wt.get(d, 0)
        cum_n += daily_pnl_no15.get(d, 0)

    print(f"  OOS cumulative PnL (leg units, not dollars):")
    print(f"    Uniform:       {cum_u:+.1f}")
    print(f"    Rung-weighted: {cum_r:+.1f}")
    print(f"    Exclude +1.5:  {cum_n:+.1f}")

    u_daily = [daily_pnl_uniform.get(d, 0) for d in all_dates]
    r_daily = [daily_pnl_rung_wt.get(d, 0) for d in all_dates]
    if len(u_daily) > 2:
        t_stat, t_p = stats.ttest_rel(r_daily, u_daily)
        print(f"    Paired t (rung-wt vs uniform): t={t_stat:.3f}, p={t_p:.4f}")

    return rung_stats


# ═══════════════════════════════════════════════════════════════
# PART 2: INTRA-RUNG — Feature discrimination WITHIN each rung
# ═══════════════════════════════════════════════════════════════

def intra_rung_analysis(data):
    print("\n" + "=" * 70)
    print("PART 2: INTRA-RUNG CONDITIONAL SIGNAL")
    print("=" * 70)

    by_rung = defaultdict(list)
    for d in data:
        by_rung[d['rung']].append(d)

    # ── Data sufficiency ──
    print(f"\n  ── Data sufficiency per rung ──")
    print(f"  {'Rung':<8} {'N':<6} {'Days':<6} {'Status'}")
    for rung in sorted(by_rung.keys(), key=lambda x: float(x)):
        legs = by_rung[rung]
        n = len(legs)
        n_days = len(set(l['game_date'] for l in legs))
        if n >= 200:
            status = "MODELABLE"
        elif n >= 80:
            status = "MARGINAL"
        else:
            status = "TOO THIN"
        print(f"  {rung:<8} {n:<6} {n_days:<6} {status}")

    # ── Features to test within each rung ──
    feature_names = [
        ('dist_wp', 'Distributional WP'),
        ('bayes_wp', 'Bayesian team WP'),
        ('edge_pp', 'NegBin edge (pp)'),
        ('mkt_disagree', 'Market disagreement'),
        ('game_total', 'Expected game total (mu_h + mu_a)'),
        ('mu_diff', 'Blowout propensity (|mu_h - mu_a|)'),
        ('fav_strength', 'Favorite strength (max mu)'),
        ('composite', 'Ridge composite'),
    ]

    # ── For each rung with enough data, test each feature ──
    n_tests = 0
    results = {}

    for rung in sorted(by_rung.keys(), key=lambda x: float(x)):
        legs = by_rung[rung]
        if len(legs) < 80:
            continue

        print(f"\n  ── Rung {rung} (N={len(legs)}) ──")
        print(f"  {'Feature':<30} {'Solo_ρ':<10} {'Partial_ρ':<12} {'Part_p':<10} {'Direction'}")

        rung_results = {}

        y = np.array([l['result'] for l in legs])
        dist_arr = np.array([l['dist_wp'] for l in legs])

        for feat_key, feat_name in feature_names:
            feat_arr = np.array([l[feat_key] if l[feat_key] is not None else 0 for l in legs])

            if np.std(feat_arr) < 1e-10:
                continue

            # Solo discrimination
            solo_rho, solo_p = stats.spearmanr(feat_arr, y)

            # Partial correlation controlling for dist_wp
            # residualize feat on dist_wp, then correlate with y
            if feat_key != 'dist_wp':
                # Regress feat on dist_wp
                slope, intercept, _, _, _ = stats.linregress(dist_arr, feat_arr)
                feat_resid = feat_arr - (slope * dist_arr + intercept)
                # Regress y on dist_wp
                slope_y, intercept_y, _, _, _ = stats.linregress(dist_arr, y)
                y_resid = y - (slope_y * dist_arr + intercept_y)
                partial_rho, partial_p = stats.spearmanr(feat_resid, y_resid)
            else:
                partial_rho, partial_p = solo_rho, solo_p

            direction = "+" if solo_rho > 0 else "-"
            flag = " ***" if abs(partial_rho) > 0.10 and partial_p < 0.05 else ""

            print(f"  {feat_name:<30} {solo_rho:+.4f}    {partial_rho:+.4f}      "
                  f"{partial_p:.4f}    {direction}{flag}")

            rung_results[feat_key] = {
                'solo_rho': solo_rho, 'partial_rho': partial_rho,
                'partial_p': partial_p, 'direction': direction,
            }
            n_tests += 1

        results[rung] = rung_results

    # ── Cross-rung sign comparison: opposite signs across rungs? ──
    print(f"\n  {'='*70}")
    print(f"  CROSS-RUNG SIGN COMPARISON (opposite signs = rung-conditional signal)")
    print(f"  {'='*70}")

    modeled_rungs = [r for r in sorted(results.keys(), key=lambda x: float(x))]
    if len(modeled_rungs) >= 2:
        print(f"\n  {'Feature':<30}", end="")
        for r in modeled_rungs:
            print(f"  {r:<12}", end="")
        print("  Sign flip?")

        for feat_key, feat_name in feature_names:
            if feat_key == 'dist_wp':
                continue
            print(f"  {feat_name:<30}", end="")
            signs = []
            for r in modeled_rungs:
                if feat_key in results[r]:
                    rho = results[r][feat_key]['partial_rho']
                    p = results[r][feat_key]['partial_p']
                    sig = "*" if p < 0.05 else " "
                    print(f"  {rho:+.3f}{sig}     ", end="")
                    signs.append(rho)
                else:
                    print(f"  {'N/A':<12}", end="")

            # Check for sign flip between extreme rungs
            if len(signs) >= 2:
                if signs[0] * signs[-1] < 0:
                    print("  YES ←")
                else:
                    print("  no")
            else:
                print()

    # ── OOS validation of any promising rung-conditional features ──
    print(f"\n  {'='*70}")
    print(f"  OOS VALIDATION: rung-conditional features")
    print(f"  {'='*70}")

    # Test: for each rung, does adding features beyond dist_wp improve OOS Brier?
    for rung in modeled_rungs:
        legs = by_rung[rung]
        if len(legs) < 100:
            continue

        y = np.array([l['result'] for l in legs])
        dates_arr = np.array([l['game_date'] for l in legs])
        folds = temporal_purged_cv(dates_arr, n_folds=4, purge_days=3)

        if len(folds) < 2:
            print(f"\n  Rung {rung}: insufficient folds for OOS validation")
            continue

        # Model A: dist_wp only
        # Model B: dist_wp + game_total + mu_diff + fav_strength
        oos_brier_a = []
        oos_brier_b = []
        oos_rho_a = []
        oos_rho_b = []

        for train_idx, test_idx in folds:
            y_train, y_test = y[train_idx], y[test_idx]

            Xa_tr = np.array([[legs[i]['dist_wp']] for i in train_idx])
            Xa_te = np.array([[legs[i]['dist_wp']] for i in test_idx])

            Xb_tr = np.array([[legs[i]['dist_wp'], legs[i]['game_total'],
                               legs[i]['mu_diff'], legs[i]['fav_strength'],
                               legs[i]['bayes_wp'], legs[i]['edge_pp']]
                              for i in train_idx])
            Xb_te = np.array([[legs[i]['dist_wp'], legs[i]['game_total'],
                               legs[i]['mu_diff'], legs[i]['fav_strength'],
                               legs[i]['bayes_wp'], legs[i]['edge_pp']]
                              for i in test_idx])

            sc_a = StandardScaler()
            sc_b = StandardScaler()

            try:
                lr_a = LogisticRegression(C=0.01, penalty='l2', max_iter=5000)
                lr_a.fit(sc_a.fit_transform(Xa_tr), y_train)
                pred_a = lr_a.predict_proba(sc_a.transform(Xa_te))[:, 1]
                oos_brier_a.extend([(p, t) for p, t in zip(pred_a, y_test)])
                oos_rho_a.append(stats.spearmanr(pred_a, y_test)[0])

                lr_b = LogisticRegression(C=0.01, penalty='l2', max_iter=5000)
                lr_b.fit(sc_b.fit_transform(Xb_tr), y_train)
                pred_b = lr_b.predict_proba(sc_b.transform(Xb_te))[:, 1]
                oos_brier_b.extend([(p, t) for p, t in zip(pred_b, y_test)])
                oos_rho_b.append(stats.spearmanr(pred_b, y_test)[0])
            except Exception as e:
                continue

        if oos_brier_a and oos_brier_b:
            ba = brier_score_loss([t for _, t in oos_brier_a], [p for p, _ in oos_brier_a])
            bb = brier_score_loss([t for _, t in oos_brier_b], [p for p, _ in oos_brier_b])
            ra = np.mean(oos_rho_a) if oos_rho_a else 0
            rb = np.mean(oos_rho_b) if oos_rho_b else 0

            print(f"\n  Rung {rung} (N={len(legs)}):")
            print(f"    dist_wp only:      OOS Brier={ba:.4f}, avg ρ={ra:+.4f}")
            print(f"    dist+total+mu+bay: OOS Brier={bb:.4f}, avg ρ={rb:+.4f}")
            print(f"    ΔBrier={bb-ba:+.4f}, Δρ={rb-ra:+.4f}")
            if bb < ba - 0.005:
                print(f"    → RUNG-SPECIFIC MODEL IMPROVES OOS")
            elif bb > ba + 0.005:
                print(f"    → RUNG-SPECIFIC MODEL OVERFITS (worse OOS)")
            else:
                print(f"    → NO MEANINGFUL DIFFERENCE")

    print(f"\n  Total tests conducted: {n_tests}")
    print(f"  Deflation note: {n_tests} partial correlations tested across "
          f"{len(modeled_rungs)} rungs — Bonferroni threshold p < {0.05/n_tests:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# PART 3: RUNG-CONDITIONAL INTERACTION TEST
# ═══════════════════════════════════════════════════════════════

def interaction_test(data):
    """Test whether feature×rung interactions improve OOS prediction."""
    print("\n" + "=" * 70)
    print("PART 3: RUNG×FEATURE INTERACTION (pooled OOS test)")
    print("=" * 70)

    y = np.array([d['result'] for d in data])
    dates_arr = np.array([d['game_date'] for d in data])
    folds = temporal_purged_cv(dates_arr, n_folds=5, purge_days=3)

    # Features
    dist = np.array([d['dist_wp'] for d in data])
    bayes = np.array([d['bayes_wp'] for d in data])
    edge = np.array([d['edge_pp'] for d in data])
    spread = np.array([d['spread'] for d in data])
    total = np.array([d['game_total'] for d in data])
    mu_diff = np.array([d['mu_diff'] for d in data])

    # Model A: dist + bayes + edge (current composite features, no rung)
    Xa = np.column_stack([dist, bayes, edge])

    # Model B: same + rung
    Xb = np.column_stack([dist, bayes, edge, spread])

    # Model C: same + rung + interactions (dist×rung, bayes×rung, total×rung)
    Xc = np.column_stack([dist, bayes, edge, spread,
                          dist * spread, bayes * spread, total * spread, mu_diff * spread])

    # Model D: same + rung + interactions + game_total + mu_diff
    Xd = np.column_stack([dist, bayes, edge, spread, total, mu_diff,
                          dist * spread, bayes * spread, total * spread, mu_diff * spread])

    models = {
        'A: dist+bayes+edge': Xa,
        'B: A + rung': Xb,
        'C: B + interactions': Xc,
        'D: C + total+mu_diff': Xd,
    }

    results = {}
    for name, X_full in models.items():
        oos_preds = []
        oos_labels = []

        for train_idx, test_idx in folds:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_full[train_idx])
            X_te = sc.transform(X_full[test_idx])
            y_tr = y[train_idx]
            y_te = y[test_idx]

            lr = LogisticRegression(C=0.01, penalty='l2', max_iter=5000)
            try:
                lr.fit(X_tr, y_tr)
                pred = lr.predict_proba(X_te)[:, 1]
                oos_preds.extend(pred)
                oos_labels.extend(y_te)
            except Exception:
                continue

        if oos_preds:
            brier = brier_score_loss(oos_labels, oos_preds)
            rho, _ = stats.spearmanr(oos_preds, oos_labels)
            results[name] = {'brier': brier, 'rho': rho, 'n': len(oos_preds)}

    print(f"\n  {'Model':<35} {'OOS Brier':<12} {'OOS ρ':<10} {'N'}")
    base_brier = None
    for name, r in results.items():
        delta = f"  (Δ={r['brier'] - base_brier:+.4f})" if base_brier is not None else ""
        print(f"  {name:<35} {r['brier']:.4f}       {r['rho']:+.4f}    {r['n']}{delta}")
        if base_brier is None:
            base_brier = r['brier']

    # ── Fit full model on all data to read interaction coefficients ──
    print(f"\n  ── Interaction coefficients (in-sample, for sign inspection) ──")
    sc = StandardScaler()
    X_full = sc.fit_transform(Xd)
    lr = LogisticRegression(C=0.01, penalty='l2', max_iter=5000)
    lr.fit(X_full, y)
    feat_names = ['dist_wp', 'bayes_wp', 'edge_pp', 'spread', 'game_total', 'mu_diff',
                  'dist×spread', 'bayes×spread', 'total×spread', 'mu_diff×spread']
    print(f"  {'Feature':<20} {'Coef':<10}")
    for fn, c in zip(feat_names, lr.coef_[0]):
        flag = " ←" if 'spread' in fn.lower() and '×' in fn and abs(c) > 0.05 else ""
        print(f"  {fn:<20} {c:+.4f}{flag}")

    return results


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} settled legs, {len(set(d['game_date'] for d in data))} days")
    print(f"Overall WR: {np.mean([d['result'] for d in data]):.4f}")
    print(f"Rungs: {sorted(set(d['rung'] for d in data), key=lambda x: float(x))}")

    rung_stats = inter_rung_analysis(data)
    intra_results = intra_rung_analysis(data)
    interaction_results = interaction_test(data)

    # ── FINAL VERDICT ──
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    print("""
  1. INTER-RUNG: Is edge rung-concentrated?
     → See per-rung WR/edge/gate-rejection table above.
     → See rung-weighted vs uniform OOS PnL comparison.

  2. INTRA-RUNG: Does any feature discriminate within a rung beyond dist_wp?
     → See partial correlations per rung + cross-rung sign comparison.
     → Sign flips between rungs would indicate rung-conditional signal.

  3. INTERACTION: Does rung×feature interaction improve OOS prediction?
     → See Model A→D progression. ΔBrier < -0.005 = real improvement.
     → Interaction coefficients show which features work differently per rung.

  4. DATA SUFFICIENCY: Can we model per-rung without overfitting?
     → See per-rung N and modelability status.
     → OOS Brier comparison (dist-only vs dist+features) per rung.
    """)
