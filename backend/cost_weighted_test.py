"""
Cost-Weighted & Fraud-Detection-Inspired Tests for MLB Alt-Spread Pipeline
============================================================================
4 tests: threshold optimization, cost-sensitive training, anomaly detection, top-K selection
Validation: temporal purged 5-fold CV, 3-day purge, date-level blocks
"""

import sqlite3
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")
np.random.seed(42)

DB_PATH = "data/bets.db"

# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT l.*, b.game_date, b.status as parlay_status, b.odds as parlay_odds,
               b.id as bet_id, b.legs as n_legs
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE b.source='shadow_negbin' AND b.cohort='clean'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Binary outcome
    df["y"] = (df["leg_result"] == "WIN").astype(int)

    # PnL per leg: WIN = +(parlay_odds-1), LOSS = -1
    df["leg_pnl"] = np.where(df["y"] == 1, df["parlay_odds"] - 1, -1.0)

    # Extract rung from description
    df["rung"] = df["description"].apply(
        lambda d: float(re.search(r"[+-](\d+\.5)", str(d)).group(1))
        if re.search(r"[+-](\d+\.5)", str(d)) else np.nan
    )

    # Cell encoding
    cell_map = {"fav_home": 0, "fav_away": 1, "dog_home": 2, "dog_away": 3}
    df["cell_num"] = df["negbin_cell"].map(cell_map).fillna(0).astype(int)

    return df


FEATURES = ["ats_dist_wp", "bayes_team_wp", "negbin_edge_pp", "mkt_disagree_dist", "ridge_composite"]
FEATURES_FULL = FEATURES + ["cell_num"]


# ── Temporal Purged CV ─────────────────────────────────────────────────────────
def make_cv_folds(df, n_folds=5, purge_days=3):
    """Date-level blocks, 3-day purge between train/test."""
    dates = sorted(df["game_date"].unique())
    n = len(dates)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n
        test_dates = set(dates[test_start:test_end])

        # Purge: remove train dates within purge_days of any test date
        test_dates_dt = [pd.Timestamp(d) for d in test_dates]
        train_dates = []
        for d in dates:
            if d in test_dates:
                continue
            d_dt = pd.Timestamp(d)
            too_close = any(abs((d_dt - td).days) <= purge_days for td in test_dates_dt)
            if not too_close:
                train_dates.append(d)

        train_idx = df[df["game_date"].isin(train_dates)].index
        test_idx = df[df["game_date"].isin(test_dates)].index
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


def impute_and_scale(df_train, df_test, features):
    """Mean-impute from train, then StandardScaler."""
    means = df_train[features].mean()
    X_train = df_train[features].fillna(means).values
    X_test = df_test[features].fillna(means).values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def deflated_sharpe(sharpe_obs, n_trials):
    """Bailey-Lopez de Prado deflated Sharpe."""
    if n_trials <= 1:
        return sharpe_obs
    e_max_sr = np.sqrt(2 * np.log(n_trials))  # expected max SR under null
    return stats.norm.cdf((sharpe_obs - e_max_sr) / 1.0)


# ── TEST 1: Cost-Weighted Threshold Optimization ──────────────────────────────
def test1_threshold_optimization(df, folds):
    print("\n" + "=" * 80)
    print("TEST 1: COST-WEIGHTED THRESHOLD OPTIMIZATION")
    print("=" * 80)

    thresholds = np.arange(0.50, 0.96, 0.01)
    baseline_pnls = []
    optimized_pnls = []
    opt_thresholds = []
    fold_details = []

    for fi, (train_idx, test_idx) in enumerate(folds):
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        # Use dist_wp as the scoring column
        score_col = "ats_dist_wp"
        train_scores = train[score_col].fillna(0)
        test_scores = test[score_col].fillna(0)

        # Sweep thresholds on training data
        best_t, best_profit_per_bet = 0.50, -999
        train_curve = []
        for t in thresholds:
            mask = train_scores >= t
            if mask.sum() < 5:
                continue
            sub = train.loc[mask]
            wins = sub[sub["y"] == 1]
            losses = sub[sub["y"] == 0]
            prec = len(wins) / len(sub)
            avg_win_pay = wins["leg_pnl"].mean() if len(wins) > 0 else 0
            profit_per_bet = prec * avg_win_pay - (1 - prec) * 1.0
            total_pnl = sub["leg_pnl"].sum()
            train_curve.append((t, prec, len(sub), total_pnl, profit_per_bet))
            if profit_per_bet > best_profit_per_bet:
                best_profit_per_bet = profit_per_bet
                best_t = t

        opt_thresholds.append(best_t)

        # Apply to test data
        test_all_pnl = test["leg_pnl"].sum()
        test_all_n = len(test)

        test_mask = test_scores >= best_t
        test_sel = test.loc[test_mask]
        test_sel_pnl = test_sel["leg_pnl"].sum() if len(test_sel) > 0 else 0
        test_sel_n = len(test_sel)
        test_sel_wr = test_sel["y"].mean() if len(test_sel) > 0 else 0

        baseline_pnls.append(test_all_pnl)
        optimized_pnls.append(test_sel_pnl)

        # Confusion matrix at optimal threshold
        tp = ((test_scores >= best_t) & (test["y"] == 1)).sum()
        fp = ((test_scores >= best_t) & (test["y"] == 0)).sum()
        tn = ((test_scores < best_t) & (test["y"] == 0)).sum()
        fn = ((test_scores < best_t) & (test["y"] == 1)).sum()

        fold_details.append({
            "fold": fi, "threshold": best_t,
            "baseline_n": test_all_n, "baseline_pnl": test_all_pnl,
            "selected_n": test_sel_n, "selected_pnl": test_sel_pnl,
            "selected_wr": test_sel_wr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "baseline_pnl_per_bet": test_all_pnl / test_all_n if test_all_n > 0 else 0,
            "selected_pnl_per_bet": test_sel_pnl / test_sel_n if test_sel_n > 0 else 0,
        })

    # Print fold details
    print(f"\n{'Fold':>4} {'Thr':>6} {'Base N':>7} {'Base PnL':>9} {'Sel N':>6} {'Sel PnL':>9} "
          f"{'Sel WR':>7} {'PnL/bet(B)':>11} {'PnL/bet(S)':>11} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print("-" * 105)
    for d in fold_details:
        print(f"{d['fold']:>4} {d['threshold']:>6.2f} {d['baseline_n']:>7} {d['baseline_pnl']:>9.2f} "
              f"{d['selected_n']:>6} {d['selected_pnl']:>9.2f} {d['selected_wr']:>7.3f} "
              f"{d['baseline_pnl_per_bet']:>11.4f} {d['selected_pnl_per_bet']:>11.4f} "
              f"{d['tp']:>4} {d['fp']:>4} {d['tn']:>4} {d['fn']:>4}")

    total_base = sum(baseline_pnls)
    total_opt = sum(optimized_pnls)
    base_n = sum(d["baseline_n"] for d in fold_details)
    sel_n = sum(d["selected_n"] for d in fold_details)

    print(f"\nAGGREGATE:")
    print(f"  Baseline: {base_n} legs, PnL = {total_base:.2f}, PnL/bet = {total_base/base_n:.4f}")
    print(f"  Optimized: {sel_n} legs, PnL = {total_opt:.2f}, PnL/bet = {total_opt/sel_n:.4f}" if sel_n > 0 else "  Optimized: 0 legs")
    print(f"  Avg optimal threshold: {np.mean(opt_thresholds):.3f}")

    # Paired t-test on fold-level PnL
    if len(baseline_pnls) > 1:
        t_stat, p_val = stats.ttest_rel(optimized_pnls, baseline_pnls)
        print(f"  Paired t-test (opt vs base total PnL): t={t_stat:.3f}, p={p_val:.3f}")

    # Precision-recall-PnL curve on full data (informational)
    print("\n  Full-data dist_wp threshold curve (in-sample, informational only):")
    print(f"  {'Threshold':>10} {'N':>6} {'WR':>7} {'Total PnL':>10} {'PnL/bet':>9}")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        mask = df["ats_dist_wp"].fillna(0) >= t
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        wr = sub["y"].mean()
        pnl = sub["leg_pnl"].sum()
        print(f"  {t:>10.2f} {len(sub):>6} {wr:>7.3f} {pnl:>10.2f} {pnl/len(sub):>9.4f}")

    verdict = "ADDITIVE" if total_opt > total_base else "DEAD"
    print(f"\n  VERDICT: {verdict} (threshold optimization {'improves' if verdict == 'ADDITIVE' else 'does not improve'} OOS PnL)")
    return total_base, total_opt, verdict


# ── TEST 2: Cost-Sensitive Training ───────────────────────────────────────────
def test2_cost_sensitive(df, folds):
    print("\n" + "=" * 80)
    print("TEST 2: COST-SENSITIVE LOGISTIC REGRESSION")
    print("=" * 80)

    results = {"standard": [], "cost_w1": [], "cost_w2": []}
    rank_corrs = []

    for fi, (train_idx, test_idx) in enumerate(folds):
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        X_train, X_test = impute_and_scale(train, test, FEATURES_FULL)
        y_train = train["y"].values
        y_test = test["y"].values
        odds_train = train["parlay_odds"].values
        odds_test = test["parlay_odds"].values
        pnl_test = test["leg_pnl"].values

        # Standard logistic regression
        lr_std = LogisticRegression(max_iter=1000, C=1.0)
        lr_std.fit(X_train, y_train)
        p_std = lr_std.predict_proba(X_test)[:, 1]

        # Cost-weighted (sample_weight = parlay_odds)
        lr_cw1 = LogisticRegression(max_iter=1000, C=1.0)
        lr_cw1.fit(X_train, y_train, sample_weight=odds_train)
        p_cw1 = lr_cw1.predict_proba(X_test)[:, 1]

        # Cost-weighted^2 (sample_weight = parlay_odds^2)
        lr_cw2 = LogisticRegression(max_iter=1000, C=1.0)
        lr_cw2.fit(X_train, y_train, sample_weight=odds_train ** 2)
        p_cw2 = lr_cw2.predict_proba(X_test)[:, 1]

        # PnL: bet on all legs, weight by predicted probability (Kelly-style sizing proxy)
        # More practical: compare at a fixed threshold (0.5 prob → bet)
        # Since WR is ~78%, all legs pass 0.5. Use expected profit ranking to select top half.
        for name, preds in [("standard", p_std), ("cost_w1", p_cw1), ("cost_w2", p_cw2)]:
            # Simple: bet all legs, PnL weighted by prediction confidence
            # Expected profit per leg = pred * (odds-1) - (1-pred) * 1
            exp_profit = preds * (odds_test - 1) - (1 - preds)
            # Only bet legs with positive expected profit
            mask = exp_profit > 0
            sel_pnl = pnl_test[mask].sum() if mask.any() else 0
            all_pnl = pnl_test.sum()
            results[name].append({
                "fold": fi, "all_pnl": all_pnl,
                "sel_n": mask.sum(), "sel_pnl": sel_pnl,
                "brier": brier_score_loss(y_test, preds),
            })

        # Rank correlation between standard and cost-weighted predictions
        rho, _ = stats.spearmanr(p_std, p_cw1)
        rank_corrs.append(rho)

    # Print results
    print(f"\n{'Model':>12} {'Folds':>6} {'Total Sel N':>12} {'Total Sel PnL':>14} {'Total All PnL':>14} {'Avg Brier':>10}")
    print("-" * 72)
    for name in ["standard", "cost_w1", "cost_w2"]:
        data = results[name]
        total_sel_n = sum(d["sel_n"] for d in data)
        total_sel_pnl = sum(d["sel_pnl"] for d in data)
        total_all_pnl = sum(d["all_pnl"] for d in data)
        avg_brier = np.mean([d["brier"] for d in data])
        label = {"standard": "Equal Wt", "cost_w1": "Odds Wt", "cost_w2": "Odds^2 Wt"}[name]
        print(f"{label:>12} {len(data):>6} {total_sel_n:>12} {total_sel_pnl:>14.2f} "
              f"{total_all_pnl:>14.2f} {avg_brier:>10.4f}")

    print(f"\n  Rank correlation (standard vs cost-weighted preds): {np.mean(rank_corrs):.4f} (avg across folds)")

    # Per-fold detail
    print(f"\n  Per-fold PnL comparison (selected legs with positive expected profit):")
    print(f"  {'Fold':>4} {'Std PnL':>9} {'CW1 PnL':>9} {'CW2 PnL':>9} {'All PnL':>9}")
    for fi in range(len(folds)):
        print(f"  {fi:>4} {results['standard'][fi]['sel_pnl']:>9.2f} "
              f"{results['cost_w1'][fi]['sel_pnl']:>9.2f} "
              f"{results['cost_w2'][fi]['sel_pnl']:>9.2f} "
              f"{results['standard'][fi]['all_pnl']:>9.2f}")

    # Significance
    std_pnls = [d["sel_pnl"] for d in results["standard"]]
    cw1_pnls = [d["sel_pnl"] for d in results["cost_w1"]]
    cw2_pnls = [d["sel_pnl"] for d in results["cost_w2"]]
    base_pnls = [d["all_pnl"] for d in results["standard"]]

    best_name = "standard"
    best_pnls = std_pnls
    if sum(cw1_pnls) > sum(best_pnls):
        best_name, best_pnls = "cost_w1", cw1_pnls
    if sum(cw2_pnls) > sum(best_pnls):
        best_name, best_pnls = "cost_w2", cw2_pnls

    t_stat, p_val = stats.ttest_rel(best_pnls, base_pnls)
    print(f"\n  Best model: {best_name}, paired t-test vs baseline (all legs): t={t_stat:.3f}, p={p_val:.3f}")

    verdict = "ADDITIVE" if sum(best_pnls) > sum(base_pnls) else "DEAD"
    print(f"  VERDICT: {verdict}")
    return sum(base_pnls), sum(best_pnls), verdict


# ── TEST 3: Anomaly / One-Class Detection ─────────────────────────────────────
def test3_anomaly(df, folds):
    print("\n" + "=" * 80)
    print("TEST 3: ANOMALY / ONE-CLASS DETECTION")
    print("=" * 80)

    configs = []
    # One-class SVM
    for nu in [0.1, 0.2, 0.3]:
        configs.append(("OCSVM", {"nu": nu}, nu))
    # Isolation Forest
    for cont in [0.1, 0.2, 0.3]:
        configs.append(("IsoForest", {"contamination": cont, "n_estimators": 100, "random_state": 42}, cont))
    # LOF
    for cont in [0.1, 0.2]:
        configs.append(("LOF", {"novelty": True, "n_neighbors": 20, "contamination": cont}, cont))

    all_results = []

    for model_name, params, param_val in configs:
        fold_results = []
        for fi, (train_idx, test_idx) in enumerate(folds):
            train = df.loc[train_idx]
            test = df.loc[test_idx]

            X_train, X_test = impute_and_scale(train, test, FEATURES)
            y_train = train["y"].values
            y_test = test["y"].values
            pnl_test = test["leg_pnl"].values

            # Fit on WINNING training legs only
            win_mask = y_train == 1
            X_train_wins = X_train[win_mask]

            if len(X_train_wins) < 10:
                continue

            try:
                if model_name == "OCSVM":
                    model = OneClassSVM(kernel="rbf", nu=params["nu"])
                    model.fit(X_train_wins)
                    scores = model.decision_function(X_test)
                    preds = model.predict(X_test)  # +1 normal, -1 anomaly
                elif model_name == "IsoForest":
                    model = IsolationForest(**params)
                    model.fit(X_train_wins)
                    scores = model.decision_function(X_test)
                    preds = model.predict(X_test)
                elif model_name == "LOF":
                    model = LocalOutlierFactor(**params)
                    model.fit(X_train_wins)
                    scores = model.decision_function(X_test)
                    preds = model.predict(X_test)
            except Exception as e:
                continue

            normal_mask = preds == 1
            anomaly_mask = preds == -1

            wr_normal = y_test[normal_mask].mean() if normal_mask.sum() > 0 else np.nan
            wr_anomaly = y_test[anomaly_mask].mean() if anomaly_mask.sum() > 0 else np.nan

            # Spearman: does score predict outcome?
            rho, p = stats.spearmanr(scores, y_test)

            # PnL: filter anomalies (only bet normal legs)
            pnl_all = pnl_test.sum()
            pnl_normal = pnl_test[normal_mask].sum() if normal_mask.sum() > 0 else 0
            n_normal = normal_mask.sum()
            n_anomaly = anomaly_mask.sum()

            # Brier with anomaly score added to logistic
            # Train logistic with anomaly score as extra feature
            train_scores_anom = None
            try:
                if model_name == "OCSVM":
                    train_scores_anom = model.decision_function(X_train)
                elif model_name == "IsoForest":
                    train_scores_anom = model.decision_function(X_train)
                elif model_name == "LOF":
                    train_scores_anom = model.decision_function(X_train)

                X_train_aug = np.column_stack([X_train, train_scores_anom])
                X_test_aug = np.column_stack([X_test, scores])
                lr_aug = LogisticRegression(max_iter=1000, C=1.0)
                lr_aug.fit(X_train_aug, y_train)
                p_aug = lr_aug.predict_proba(X_test_aug)[:, 1]
                brier_aug = brier_score_loss(y_test, p_aug)

                lr_base = LogisticRegression(max_iter=1000, C=1.0)
                lr_base.fit(X_train, y_train)
                p_base = lr_base.predict_proba(X_test)[:, 1]
                brier_base = brier_score_loss(y_test, p_base)
            except:
                brier_aug = np.nan
                brier_base = np.nan

            fold_results.append({
                "fold": fi, "wr_normal": wr_normal, "wr_anomaly": wr_anomaly,
                "rho": rho, "rho_p": p, "pnl_all": pnl_all, "pnl_normal": pnl_normal,
                "n_normal": n_normal, "n_anomaly": n_anomaly,
                "brier_base": brier_base, "brier_aug": brier_aug,
                "pnl_per_bet_all": pnl_all / len(test) if len(test) > 0 else 0,
                "pnl_per_bet_normal": pnl_normal / n_normal if n_normal > 0 else 0,
            })

        if not fold_results:
            continue

        avg_wr_normal = np.nanmean([r["wr_normal"] for r in fold_results])
        avg_wr_anomaly = np.nanmean([r["wr_anomaly"] for r in fold_results])
        avg_rho = np.mean([r["rho"] for r in fold_results])
        total_pnl_all = sum(r["pnl_all"] for r in fold_results)
        total_pnl_normal = sum(r["pnl_normal"] for r in fold_results)
        total_n_normal = sum(r["n_normal"] for r in fold_results)
        total_n_anomaly = sum(r["n_anomaly"] for r in fold_results)
        avg_brier_base = np.nanmean([r["brier_base"] for r in fold_results])
        avg_brier_aug = np.nanmean([r["brier_aug"] for r in fold_results])

        all_results.append({
            "model": model_name, "param": param_val,
            "wr_normal": avg_wr_normal, "wr_anomaly": avg_wr_anomaly,
            "rho": avg_rho, "pnl_all": total_pnl_all, "pnl_normal": total_pnl_normal,
            "n_normal": total_n_normal, "n_anomaly": total_n_anomaly,
            "brier_base": avg_brier_base, "brier_aug": avg_brier_aug,
            "fold_data": fold_results,
        })

    # Print summary table
    print(f"\n{'Model':>12} {'Param':>6} {'N Norm':>7} {'N Anom':>7} {'WR Norm':>8} {'WR Anom':>8} "
          f"{'Rho':>7} {'PnL All':>9} {'PnL Norm':>9} {'Brier B':>8} {'Brier+A':>8}")
    print("-" * 105)
    for r in all_results:
        print(f"{r['model']:>12} {r['param']:>6.1f} {r['n_normal']:>7} {r['n_anomaly']:>7} "
              f"{r['wr_normal']:>8.3f} {r['wr_anomaly']:>8.3f} {r['rho']:>7.3f} "
              f"{r['pnl_all']:>9.2f} {r['pnl_normal']:>9.2f} "
              f"{r['brier_base']:>8.4f} {r['brier_aug']:>8.4f}")

    # Find best anomaly filter
    best = max(all_results, key=lambda r: r["pnl_normal"])
    baseline_pnl = best["pnl_all"]

    # Per-bet comparison for best model
    print(f"\n  Best anomaly model: {best['model']} (param={best['param']})")
    print(f"  Baseline PnL: {baseline_pnl:.2f} ({sum(r['n_normal'] + r['n_anomaly'] for r in best['fold_data'])} legs)")
    print(f"  Filtered PnL: {best['pnl_normal']:.2f} ({best['n_normal']} normal legs)")
    total_legs = sum(len(df.loc[folds[r['fold']][1]]) for r in best['fold_data'])
    print(f"  PnL/bet baseline: {baseline_pnl / total_legs:.4f}")
    print(f"  PnL/bet filtered: {best['pnl_normal'] / best['n_normal']:.4f}" if best['n_normal'] > 0 else "")

    # Significance
    base_fold_pnls = [r["pnl_all"] for r in best["fold_data"]]
    norm_fold_pnls = [r["pnl_normal"] for r in best["fold_data"]]
    if len(base_fold_pnls) > 1:
        t_stat, p_val = stats.ttest_rel(norm_fold_pnls, base_fold_pnls)
        print(f"  Paired t-test (filtered vs all): t={t_stat:.3f}, p={p_val:.3f}")

    verdict = "ADDITIVE" if best["pnl_normal"] > baseline_pnl else "DEAD"
    # Also check PnL per bet
    pnl_per_bet_all = baseline_pnl / total_legs if total_legs > 0 else 0
    pnl_per_bet_norm = best["pnl_normal"] / best["n_normal"] if best["n_normal"] > 0 else 0
    verdict_per_bet = "ADDITIVE" if pnl_per_bet_norm > pnl_per_bet_all else "DEAD"
    print(f"  VERDICT (total PnL): {verdict}")
    print(f"  VERDICT (PnL/bet): {verdict_per_bet}")
    return baseline_pnl, best["pnl_normal"], verdict


# ── TEST 4: Top-K Selection ──────────────────────────────────────────────────
def test4_topk(df, folds):
    print("\n" + "=" * 80)
    print("TEST 4: PRECISION-AT-K / TOP-K SELECTION")
    print("=" * 80)

    K_values = [3, 5, 10, 15, 20, 25, "ALL"]

    # Two ranking methods: (A) by raw dist_wp, (B) by expected profit
    results_prob = {k: {"n": 0, "wins": 0, "pnl": 0} for k in K_values}
    results_eprofit = {k: {"n": 0, "wins": 0, "pnl": 0} for k in K_values}

    fold_details_prob = []
    fold_details_eprofit = []

    for fi, (train_idx, test_idx) in enumerate(folds):
        test = df.loc[test_idx].copy()

        dist_wp = test["ats_dist_wp"].fillna(0)
        odds = test["parlay_odds"].values
        y = test["y"].values
        pnl = test["leg_pnl"].values

        # Expected profit: p * (odds - 1) - (1-p) * 1
        exp_profit = dist_wp.values * (odds - 1) - (1 - dist_wp.values)

        # Rank by probability
        rank_prob = np.argsort(-dist_wp.values)
        # Rank by expected profit
        rank_eprofit = np.argsort(-exp_profit)

        for k in K_values:
            if k == "ALL":
                sel_prob = np.arange(len(test))
                sel_eprof = np.arange(len(test))
            else:
                sel_prob = rank_prob[:min(k, len(test))]
                sel_eprof = rank_eprofit[:min(k, len(test))]

            n_p = len(sel_prob)
            n_e = len(sel_eprof)

            results_prob[k]["n"] += n_p
            results_prob[k]["wins"] += y[sel_prob].sum()
            results_prob[k]["pnl"] += pnl[sel_prob].sum()

            results_eprofit[k]["n"] += n_e
            results_eprofit[k]["wins"] += y[sel_eprof].sum()
            results_eprofit[k]["pnl"] += pnl[sel_eprof].sum()

    # Print results
    print("\n  Ranking by PROBABILITY (dist_wp):")
    print(f"  {'K':>6} {'N':>6} {'Wins':>6} {'WR':>7} {'Total PnL':>10} {'PnL/bet':>9}")
    print("  " + "-" * 50)
    for k in K_values:
        r = results_prob[k]
        wr = r["wins"] / r["n"] if r["n"] > 0 else 0
        ppb = r["pnl"] / r["n"] if r["n"] > 0 else 0
        print(f"  {str(k):>6} {r['n']:>6} {r['wins']:>6} {wr:>7.3f} {r['pnl']:>10.2f} {ppb:>9.4f}")

    print("\n  Ranking by EXPECTED PROFIT (p*(odds-1) - (1-p)):")
    print(f"  {'K':>6} {'N':>6} {'Wins':>6} {'WR':>7} {'Total PnL':>10} {'PnL/bet':>9}")
    print("  " + "-" * 50)
    for k in K_values:
        r = results_eprofit[k]
        wr = r["wins"] / r["n"] if r["n"] > 0 else 0
        ppb = r["pnl"] / r["n"] if r["n"] > 0 else 0
        print(f"  {str(k):>6} {r['n']:>6} {r['wins']:>6} {wr:>7.3f} {r['pnl']:>10.2f} {ppb:>9.4f}")

    # Compare: does any K beat ALL on PnL/bet?
    all_ppb_prob = results_prob["ALL"]["pnl"] / results_prob["ALL"]["n"] if results_prob["ALL"]["n"] > 0 else 0
    all_ppb_eprof = results_eprofit["ALL"]["pnl"] / results_eprofit["ALL"]["n"] if results_eprofit["ALL"]["n"] > 0 else 0
    all_pnl = results_prob["ALL"]["pnl"]

    print(f"\n  Baseline (ALL legs): PnL = {all_pnl:.2f}, PnL/bet = {all_ppb_prob:.4f}")

    best_k = None
    best_ppb = all_ppb_prob
    best_method = None
    best_total_pnl = all_pnl
    for k in [3, 5, 10, 15, 20, 25]:
        for method, res in [("prob", results_prob), ("eprofit", results_eprofit)]:
            ppb = res[k]["pnl"] / res[k]["n"] if res[k]["n"] > 0 else 0
            if ppb > best_ppb:
                best_ppb = ppb
                best_k = k
                best_method = method
                best_total_pnl = res[k]["pnl"]

    if best_k is not None:
        print(f"  Best top-K: K={best_k}, method={best_method}, PnL/bet={best_ppb:.4f}, total PnL={best_total_pnl:.2f}")
    else:
        print(f"  No K value beats baseline PnL/bet")

    # Does expected-profit ranking beat probability ranking at same K?
    print("\n  Expected-profit vs probability ranking (PnL/bet difference at each K):")
    for k in [3, 5, 10, 15, 20, 25]:
        ppb_p = results_prob[k]["pnl"] / results_prob[k]["n"] if results_prob[k]["n"] > 0 else 0
        ppb_e = results_eprofit[k]["pnl"] / results_eprofit[k]["n"] if results_eprofit[k]["n"] > 0 else 0
        diff = ppb_e - ppb_p
        winner = "eprofit" if diff > 0 else "prob"
        print(f"    K={k:>3}: prob={ppb_p:.4f}, eprofit={ppb_e:.4f}, diff={diff:+.4f} ({winner})")

    verdict = "ADDITIVE" if best_k is not None else "DEAD"
    print(f"\n  VERDICT (PnL/bet): {verdict}")
    # Total PnL verdict: top-K always has less total PnL (fewer bets) unless PnL/bet is much better
    total_verdict = "DEAD"
    for k in [3, 5, 10, 15, 20, 25]:
        for res in [results_prob, results_eprofit]:
            if res[k]["pnl"] > all_pnl:
                total_verdict = "ADDITIVE"
    print(f"  VERDICT (total PnL): {total_verdict}")
    return all_pnl, best_total_pnl if best_k else all_pnl, verdict


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} legs, {df['y'].sum()} wins ({df['y'].mean():.1%} WR)")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique dates: {df['game_date'].nunique()}")
    print(f"Feature NaN rates:")
    for f in FEATURES:
        print(f"  {f}: {df[f].isna().mean():.1%}")

    # Build CV folds
    folds = make_cv_folds(df, n_folds=5, purge_days=3)
    print(f"\nCV folds: {len(folds)}")
    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i}: train={len(tr)}, test={len(te)}, "
              f"train dates={df.loc[tr, 'game_date'].nunique()}, "
              f"test dates={df.loc[te, 'game_date'].nunique()}")

    baseline_pnl = df["leg_pnl"].sum()
    baseline_n = len(df)
    print(f"\nBaseline (all legs): N={baseline_n}, PnL={baseline_pnl:.2f}, "
          f"PnL/bet={baseline_pnl / baseline_n:.4f}, WR={df['y'].mean():.3f}")

    # Run tests
    b1, o1, v1 = test1_threshold_optimization(df, folds)
    b2, o2, v2 = test2_cost_sensitive(df, folds)
    b3, o3, v3 = test3_anomaly(df, folds)
    b4, o4, v4 = test4_topk(df, folds)

    # GRAND SUMMARY
    print("\n" + "=" * 80)
    print("GRAND SUMMARY")
    print("=" * 80)
    print(f"\n{'Test':>35} {'Baseline PnL':>13} {'Method PnL':>11} {'Verdict':>10}")
    print("-" * 75)
    print(f"{'1. Threshold Optimization':>35} {b1:>13.2f} {o1:>11.2f} {v1:>10}")
    print(f"{'2. Cost-Sensitive Training':>35} {b2:>13.2f} {o2:>11.2f} {v2:>10}")
    print(f"{'3. Anomaly Detection':>35} {b3:>13.2f} {o3:>11.2f} {v3:>10}")
    print(f"{'4. Top-K Selection':>35} {b4:>13.2f} {o4:>11.2f} {v4:>10}")

    n_additive = sum(1 for v in [v1, v2, v3, v4] if v == "ADDITIVE")
    print(f"\n  {n_additive}/4 tests show ADDITIVE signal.")
    print(f"  Deflated Sharpe note: threshold sweep has {len(np.arange(0.50, 0.96, 0.01))} trials/fold,")
    print(f"  anomaly detection has {8} configs tested — interpret accordingly.")

    if n_additive == 0:
        print("\n  CONCLUSION: None of the fraud-detection-inspired techniques improve OOS PnL.")
        print("  The current 'bet all qualifying legs' approach is not leaving money on the table")
        print("  through suboptimal selection. Edge (if any) comes from the upstream model, not post-hoc filtering.")
    else:
        additive_tests = []
        if v1 == "ADDITIVE": additive_tests.append("threshold optimization")
        if v2 == "ADDITIVE": additive_tests.append("cost-sensitive training")
        if v3 == "ADDITIVE": additive_tests.append("anomaly filtering")
        if v4 == "ADDITIVE": additive_tests.append("top-K selection")
        print(f"\n  CONCLUSION: {', '.join(additive_tests)} show promise. Validate on forward data before deploying.")


if __name__ == "__main__":
    main()
