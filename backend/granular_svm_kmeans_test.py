"""
Granular SVM & K-Means Tests
=============================
Tests SVM and k-means at granular levels (per-rung, per-cell, interactions,
parlay-level, context-only archetypes) that pooled tests may have missed.

Validation: temporal purged CV (5-fold, 3-day purge), OOS Brier vs logistic
baseline, deflated Sharpe across TOTAL trial count.
"""

import sqlite3
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")

DB_PATH = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    conn = sqlite3.connect(DB_PATH)

    legs = pd.read_sql_query("""
        SELECT l.*, b.game_date, b.status as parlay_status, b.odds as parlay_odds, b.id as bet_id
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE b.source = 'shadow_negbin'
          AND b.cohort = 'clean'
          AND b.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND l.leg_result IN ('WIN', 'LOSS')
    """, conn)
    conn.close()

    # Extract rung from description
    legs["rung"] = legs["description"].apply(lambda d: float(m.group(1)) if d and (m := re.search(r'[+-](\d+\.?\d*)\s*$', str(d))) else np.nan)
    legs["y"] = (legs["leg_result"] == "WIN").astype(int)
    legs["game_date"] = pd.to_datetime(legs["game_date"])

    # Encode cell
    cell_map = {"fav_home": 0, "fav_away": 1, "dog_home": 2, "dog_away": 3}
    legs["cell_num"] = legs["negbin_cell"].map(cell_map).fillna(-1).astype(int)

    return legs


# ── Purged temporal CV ────────────────────────────────────────────────────────

def purged_temporal_cv(df, n_splits=5, purge_days=3):
    """Yield (train_idx, test_idx) with date-block folds and purge gap."""
    dates = sorted(df["game_date"].unique())
    fold_size = len(dates) // n_splits
    if fold_size == 0:
        return

    for i in range(n_splits):
        test_start = dates[i * fold_size]
        test_end = dates[min((i + 1) * fold_size - 1, len(dates) - 1)]
        purge_start = test_start - pd.Timedelta(days=purge_days)
        purge_end = test_end + pd.Timedelta(days=purge_days)

        train_mask = (df["game_date"] < purge_start) | (df["game_date"] > purge_end)
        test_mask = (df["game_date"] >= test_start) & (df["game_date"] <= test_end)

        train_idx = df.index[train_mask].tolist()
        test_idx = df.index[test_mask].tolist()

        if len(train_idx) > 10 and len(test_idx) > 5:
            yield train_idx, test_idx


def impute_and_scale(X_train, X_test):
    """Impute NaN with train-fold means, then scale."""
    means = np.nanmean(X_train, axis=0)
    means = np.where(np.isnan(means), 0, means)
    X_train = np.where(np.isnan(X_train), means, X_train)
    X_test = np.where(np.isnan(X_test), means, X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


# ── Evaluation helpers ────────────────────────────────────────────────────────

def cv_brier(df, features, model_cls, model_kwargs, label="y"):
    """Return mean OOS Brier across purged CV folds."""
    X = df[features].values.astype(float)
    y = df[label].values.astype(int)
    briers = []
    for train_idx, test_idx in purged_temporal_cv(df):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr, X_te = impute_and_scale(X_tr, X_te)
        if len(np.unique(y_tr)) < 2:
            continue
        try:
            model = model_cls(**model_kwargs)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            briers.append(brier_score_loss(y_te, probs))
        except Exception:
            continue
    return np.mean(briers) if briers else np.nan


def deflated_sharpe_pval(observed_sharpe, n_trials):
    """
    Bailey-Lopez de Prado deflated Sharpe ratio approximation.
    Under null, Sharpe ~ N(0, 1/sqrt(T)) corrected for max of n_trials.
    We use Bonferroni as a conservative bound.
    """
    if np.isnan(observed_sharpe) or n_trials < 1:
        return 1.0
    raw_p = 1 - stats.norm.cdf(observed_sharpe)
    return min(raw_p * n_trials, 1.0)


# ── CORE SIGNAL FEATURES ─────────────────────────────────────────────────────

CORE_FEATURES = ["ats_dist_wp", "bayes_team_wp", "negbin_edge_pp",
                  "mkt_disagree_dist", "ridge_composite", "cell_num"]

CONTEXT_FEATURES = [
    "home_starter_era", "away_starter_era", "starter_era_diff",
    "home_starter_k9", "away_starter_k9", "starter_k9_diff",
    "home_team_xwoba", "away_team_xwoba", "xwoba_diff",
    "home_bullpen_era", "away_bullpen_era", "bullpen_era_diff",
    "game_temp_f", "game_wind_mph"
]

INTERACTION_FEATURES = ["dist_wp_x_bayes", "dist_wp_x_edge",
                         "bayes_x_edge", "dist_wp_x_mkt_disagree"]


# ── TEST 1: PER-RUNG SVM ─────────────────────────────────────────────────────

def test1_per_rung_svm(legs):
    print("\n" + "="*80)
    print("TEST 1: PER-RUNG SVM")
    print("="*80)

    svm_params_grid = []
    for C in [0.1, 1, 10]:
        for gamma in ["scale", "auto", 0.01, 0.1]:
            svm_params_grid.append({"C": C, "gamma": gamma, "kernel": "rbf",
                                     "probability": True, "max_iter": 5000})

    results = []
    trial_count = 0
    rungs = sorted(legs["rung"].dropna().unique())

    for rung in rungs:
        seg = legs[legs["rung"] == rung].reset_index(drop=True)
        n = len(seg)
        wr = seg["y"].mean()
        if n < 50:
            print(f"\n  Rung +{rung}: N={n} — SKIPPED (< 50)")
            continue

        # Baseline: logistic on core features
        baseline_brier = cv_brier(seg, CORE_FEATURES, LogisticRegression,
                                   {"max_iter": 1000, "solver": "lbfgs"})

        best_svm_brier = np.inf
        best_params = None
        for params in svm_params_grid:
            trial_count += 1
            b = cv_brier(seg, CORE_FEATURES, SVC, params)
            if not np.isnan(b) and b < best_svm_brier:
                best_svm_brier = b
                best_params = params

        delta = best_svm_brier - baseline_brier if not np.isnan(best_svm_brier) else np.nan
        results.append({"segment": f"+{rung}", "N": n, "WR": wr,
                        "baseline_brier": baseline_brier, "svm_brier": best_svm_brier,
                        "delta": delta, "best_C": best_params.get("C") if best_params else None,
                        "best_gamma": best_params.get("gamma") if best_params else None})

        tag = " *** FOCUS ***" if rung in [1.5, 2.5] else ""
        print(f"\n  Rung +{rung}: N={n}, WR={wr:.3f}{tag}")
        print(f"    Baseline Brier: {baseline_brier:.4f}" if not np.isnan(baseline_brier) else "    Baseline Brier: N/A")
        print(f"    Best SVM Brier: {best_svm_brier:.4f}" if not np.isnan(best_svm_brier) and best_svm_brier < np.inf else "    Best SVM Brier: N/A")
        print(f"    Delta:          {delta:+.4f}" if not np.isnan(delta) and delta != np.inf else "    Delta:          N/A")

    return results, trial_count


# ── TEST 2: PER-CELL SVM ─────────────────────────────────────────────────────

def test2_per_cell_svm(legs):
    print("\n" + "="*80)
    print("TEST 2: PER-CELL SVM")
    print("="*80)

    svm_params_grid = []
    for C in [0.1, 1, 10]:
        for gamma in ["scale", "auto", 0.01, 0.1]:
            svm_params_grid.append({"C": C, "gamma": gamma, "kernel": "rbf",
                                     "probability": True, "max_iter": 5000})

    results = []
    trial_count = 0
    cells = ["fav_home", "fav_away", "dog_home", "dog_away"]

    # Use features without cell_num (meaningless within a single cell)
    feats = [f for f in CORE_FEATURES if f != "cell_num"]

    for cell in cells:
        seg = legs[legs["negbin_cell"] == cell].reset_index(drop=True)
        n = len(seg)
        wr = seg["y"].mean()
        if n < 50:
            print(f"\n  Cell {cell}: N={n} — SKIPPED (< 50)")
            continue

        baseline_brier = cv_brier(seg, feats, LogisticRegression,
                                   {"max_iter": 1000, "solver": "lbfgs"})

        best_svm_brier = np.inf
        best_params = None
        for params in svm_params_grid:
            trial_count += 1
            b = cv_brier(seg, feats, SVC, params)
            if not np.isnan(b) and b < best_svm_brier:
                best_svm_brier = b
                best_params = params

        delta = best_svm_brier - baseline_brier if not np.isnan(best_svm_brier) else np.nan
        results.append({"segment": cell, "N": n, "WR": wr,
                        "baseline_brier": baseline_brier, "svm_brier": best_svm_brier,
                        "delta": delta})

        print(f"\n  Cell {cell}: N={n}, WR={wr:.3f}")
        print(f"    Baseline Brier: {baseline_brier:.4f}" if not np.isnan(baseline_brier) else "    Baseline Brier: N/A")
        print(f"    Best SVM Brier: {best_svm_brier:.4f}" if not np.isnan(best_svm_brier) and best_svm_brier < np.inf else "    Best SVM Brier: N/A")
        print(f"    Delta:          {delta:+.4f}" if not np.isnan(delta) and delta != np.inf else "    Delta:          N/A")

    return results, trial_count


# ── TEST 3: PER-CELL ERROR CLUSTERING ────────────────────────────────────────

def test3_per_cell_error_clustering(legs):
    print("\n" + "="*80)
    print("TEST 3: PER-CELL ERROR CLUSTERING")
    print("="*80)

    feats = [f for f in CORE_FEATURES if f != "cell_num"]
    results = []
    trial_count = 0

    segments = []
    # Per cell
    for cell in ["fav_home", "fav_away", "dog_home", "dog_away"]:
        seg = legs[legs["negbin_cell"] == cell].reset_index(drop=True)
        segments.append((f"cell={cell}", seg))
    # Per rung for +1.5, +2.5
    for rung in [1.5, 2.5]:
        seg = legs[legs["rung"] == rung].reset_index(drop=True)
        segments.append((f"rung=+{rung}", seg))

    for seg_name, seg in segments:
        if len(seg) < 50:
            print(f"\n  {seg_name}: N={len(seg)} — SKIPPED")
            continue

        # Compute residuals
        seg = seg.copy()
        seg["residual"] = seg["y"] - seg["ats_dist_wp"].fillna(seg["y"].mean())
        seg["abs_residual"] = seg["residual"].abs()
        q75 = seg["abs_residual"].quantile(0.75)
        seg["high_residual"] = (seg["abs_residual"] >= q75).astype(int)

        print(f"\n  {seg_name}: N={len(seg)}, high-residual legs={seg['high_residual'].sum()}")

        for k in range(3, 9):
            trial_count += 1

            # Cluster on features, test if cluster membership predicts outcome OOS
            cluster_feats = feats.copy()
            briers_with_cluster = []
            briers_baseline = []

            for train_idx, test_idx in purged_temporal_cv(seg):
                X_tr = seg.iloc[train_idx][cluster_feats].values.astype(float)
                X_te = seg.iloc[test_idx][cluster_feats].values.astype(float)
                y_tr = seg.iloc[train_idx]["y"].values
                y_te = seg.iloc[test_idx]["y"].values

                X_tr_imp, X_te_imp = impute_and_scale(X_tr, X_te)

                if len(np.unique(y_tr)) < 2:
                    continue

                # Fit k-means on high-residual subset of training data
                high_res_mask = seg.iloc[train_idx]["high_residual"] == 1
                X_hr = X_tr_imp[high_res_mask]
                if len(X_hr) < k:
                    continue

                try:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    km.fit(X_hr)

                    # Assign all train/test to clusters
                    cl_tr = km.predict(X_tr_imp)
                    cl_te = km.predict(X_te_imp)

                    # Baseline: logistic on features
                    lr_base = LogisticRegression(max_iter=1000, solver="lbfgs")
                    lr_base.fit(X_tr_imp, y_tr)
                    p_base = lr_base.predict_proba(X_te_imp)[:, 1]
                    briers_baseline.append(brier_score_loss(y_te, p_base))

                    # With cluster: add one-hot cluster to features
                    cl_tr_oh = np.eye(k)[cl_tr]
                    cl_te_oh = np.eye(k)[cl_te]
                    X_tr_aug = np.hstack([X_tr_imp, cl_tr_oh])
                    X_te_aug = np.hstack([X_te_imp, cl_te_oh])

                    lr_aug = LogisticRegression(max_iter=1000, solver="lbfgs")
                    lr_aug.fit(X_tr_aug, y_tr)
                    p_aug = lr_aug.predict_proba(X_te_aug)[:, 1]
                    briers_with_cluster.append(brier_score_loss(y_te, p_aug))
                except Exception:
                    continue

            if briers_baseline and briers_with_cluster:
                base_b = np.mean(briers_baseline)
                clust_b = np.mean(briers_with_cluster)
                delta = clust_b - base_b
                results.append({"segment": seg_name, "k": k,
                                "baseline_brier": base_b, "cluster_brier": clust_b,
                                "delta": delta})
                print(f"    k={k}: baseline={base_b:.4f}, +cluster={clust_b:.4f}, delta={delta:+.4f}")
            else:
                print(f"    k={k}: insufficient folds")

    return results, trial_count


# ── TEST 4: K-MEANS ON CONTEXT-ONLY FEATURES ─────────────────────────────────

def test4_context_only_archetypes(legs):
    print("\n" + "="*80)
    print("TEST 4: K-MEANS ON CONTEXT-ONLY FEATURES")
    print("="*80)

    # Check how many context features are available
    available_ctx = [f for f in CONTEXT_FEATURES if f in legs.columns]
    print(f"  Available context features: {len(available_ctx)}/{len(CONTEXT_FEATURES)}")
    if len(available_ctx) < 3:
        print("  Too few context features available — SKIPPED")
        return [], 0

    # Drop rows where ALL context features are NaN
    ctx_mask = legs[available_ctx].notna().any(axis=1)
    seg = legs[ctx_mask].reset_index(drop=True)
    print(f"  Legs with any context data: {len(seg)}")

    if len(seg) < 50:
        print("  N < 50 — SKIPPED")
        return [], 0

    results = []
    trial_count = 0
    augment_feats = CORE_FEATURES.copy()

    for k in range(3, 11):
        trial_count += 1
        briers_baseline = []
        briers_with_archetype = []

        for train_idx, test_idx in purged_temporal_cv(seg):
            X_ctx_tr = seg.iloc[train_idx][available_ctx].values.astype(float)
            X_ctx_te = seg.iloc[test_idx][available_ctx].values.astype(float)
            X_core_tr = seg.iloc[train_idx][augment_feats].values.astype(float)
            X_core_te = seg.iloc[test_idx][augment_feats].values.astype(float)
            y_tr = seg.iloc[train_idx]["y"].values
            y_te = seg.iloc[test_idx]["y"].values

            X_ctx_tr_s, X_ctx_te_s = impute_and_scale(X_ctx_tr, X_ctx_te)
            X_core_tr_s, X_core_te_s = impute_and_scale(X_core_tr, X_core_te)

            if len(np.unique(y_tr)) < 2:
                continue

            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(X_ctx_tr_s)
                cl_tr = km.predict(X_ctx_tr_s)
                cl_te = km.predict(X_ctx_te_s)

                # Baseline
                lr_base = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr_base.fit(X_core_tr_s, y_tr)
                p_base = lr_base.predict_proba(X_core_te_s)[:, 1]
                briers_baseline.append(brier_score_loss(y_te, p_base))

                # Augmented
                cl_oh_tr = np.eye(k)[cl_tr]
                cl_oh_te = np.eye(k)[cl_te]
                X_aug_tr = np.hstack([X_core_tr_s, cl_oh_tr])
                X_aug_te = np.hstack([X_core_te_s, cl_oh_te])
                lr_aug = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr_aug.fit(X_aug_tr, y_tr)
                p_aug = lr_aug.predict_proba(X_aug_te)[:, 1]
                briers_with_archetype.append(brier_score_loss(y_te, p_aug))
            except Exception:
                continue

        if briers_baseline and briers_with_archetype:
            base_b = np.mean(briers_baseline)
            arch_b = np.mean(briers_with_archetype)
            delta = arch_b - base_b
            results.append({"k": k, "baseline_brier": base_b,
                            "archetype_brier": arch_b, "delta": delta})
            print(f"  k={k}: baseline={base_b:.4f}, +archetype={arch_b:.4f}, delta={delta:+.4f}")
        else:
            print(f"  k={k}: insufficient folds")

    return results, trial_count


# ── TEST 5A: PARLAY-LEVEL SVM + K-MEANS ──────────────────────────────────────

def test5a_parlay_level(legs):
    print("\n" + "="*80)
    print("TEST 5A: PARLAY-LEVEL SVM + K-MEANS")
    print("="*80)

    # Aggregate to parlay level
    parlay_agg = legs.groupby("bet_id").agg(
        game_date=("game_date", "first"),
        parlay_win=("parlay_status", lambda x: int(x.iloc[0] == "SETTLED_WIN")),
        mean_dist_wp=("ats_dist_wp", "mean"),
        min_dist_wp=("ats_dist_wp", "min"),
        mean_edge=("negbin_edge_pp", "mean"),
        mean_bayes=("bayes_team_wp", "mean"),
        mean_ridge=("ridge_composite", "mean"),
        mean_disagree=("mkt_disagree_dist", "mean"),
        odds=("parlay_odds", "first"),
        n_legs=("y", "count"),
    ).reset_index()

    # Concentrated risk: max single-leg contribution
    max_wp = legs.groupby("bet_id")["ats_dist_wp"].max().reset_index()
    max_wp.columns = ["bet_id", "max_leg_wp"]
    parlay_agg = parlay_agg.merge(max_wp, on="bet_id", how="left")
    parlay_agg["concentrated_risk"] = parlay_agg["max_leg_wp"] / parlay_agg["mean_dist_wp"].replace(0, np.nan)

    parlay_agg["y"] = parlay_agg["parlay_win"]
    parlay_agg["game_date"] = pd.to_datetime(parlay_agg["game_date"])

    parlay_feats = ["mean_dist_wp", "min_dist_wp", "mean_edge", "mean_bayes",
                    "mean_ridge", "mean_disagree", "odds", "n_legs", "concentrated_risk"]

    n = len(parlay_agg)
    wr = parlay_agg["y"].mean()
    print(f"  Parlays: N={n}, WR={wr:.3f}")

    if n < 50:
        print("  N < 50 — SKIPPED")
        return [], 0

    results = []
    trial_count = 0

    # Baseline
    baseline_brier = cv_brier(parlay_agg, parlay_feats, LogisticRegression,
                               {"max_iter": 1000, "solver": "lbfgs"})
    print(f"  Baseline Brier (logistic): {baseline_brier:.4f}" if not np.isnan(baseline_brier) else "  Baseline Brier: N/A")

    # SVM
    svm_params_grid = []
    for C in [0.1, 1, 10]:
        for gamma in ["scale", "auto", 0.01, 0.1]:
            svm_params_grid.append({"C": C, "gamma": gamma, "kernel": "rbf",
                                     "probability": True, "max_iter": 5000})

    best_svm_brier = np.inf
    for params in svm_params_grid:
        trial_count += 1
        b = cv_brier(parlay_agg, parlay_feats, SVC, params)
        if not np.isnan(b) and b < best_svm_brier:
            best_svm_brier = b

    svm_delta = best_svm_brier - baseline_brier if best_svm_brier < np.inf else np.nan
    results.append({"test": "parlay_svm", "baseline": baseline_brier,
                    "model": best_svm_brier, "delta": svm_delta})
    print(f"  SVM Brier: {best_svm_brier:.4f}, delta={svm_delta:+.4f}" if not np.isnan(svm_delta) else "  SVM: N/A")

    # K-means
    for k in range(3, 9):
        trial_count += 1
        briers_base = []
        briers_km = []

        for train_idx, test_idx in purged_temporal_cv(parlay_agg):
            X_tr = parlay_agg.iloc[train_idx][parlay_feats].values.astype(float)
            X_te = parlay_agg.iloc[test_idx][parlay_feats].values.astype(float)
            y_tr = parlay_agg.iloc[train_idx]["y"].values
            y_te = parlay_agg.iloc[test_idx]["y"].values
            X_tr_s, X_te_s = impute_and_scale(X_tr, X_te)

            if len(np.unique(y_tr)) < 2:
                continue
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(X_tr_s)
                cl_tr = km.predict(X_tr_s)
                cl_te = km.predict(X_te_s)

                lr_b = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr_b.fit(X_tr_s, y_tr)
                briers_base.append(brier_score_loss(y_te, lr_b.predict_proba(X_te_s)[:, 1]))

                cl_oh_tr = np.eye(k)[cl_tr]
                cl_oh_te = np.eye(k)[cl_te]
                lr_a = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr_a.fit(np.hstack([X_tr_s, cl_oh_tr]), y_tr)
                briers_km.append(brier_score_loss(y_te, lr_a.predict_proba(np.hstack([X_te_s, cl_oh_te]))[:, 1]))
            except Exception:
                continue

        if briers_base and briers_km:
            base_b = np.mean(briers_base)
            km_b = np.mean(briers_km)
            delta = km_b - base_b
            results.append({"test": f"parlay_kmeans_k{k}", "baseline": base_b,
                            "model": km_b, "delta": delta})
            print(f"  K-means k={k}: baseline={base_b:.4f}, +cluster={km_b:.4f}, delta={delta:+.4f}")

    return results, trial_count


# ── TEST 5B: SVM ON INTERACTIONS ──────────────────────────────────────────────

def test5b_interactions(legs):
    print("\n" + "="*80)
    print("TEST 5B: SVM ON SIGNAL INTERACTIONS")
    print("="*80)

    seg = legs.copy()
    seg["dist_wp_x_bayes"] = seg["ats_dist_wp"] * seg["bayes_team_wp"]
    seg["dist_wp_x_edge"] = seg["ats_dist_wp"] * seg["negbin_edge_pp"]
    seg["bayes_x_edge"] = seg["bayes_team_wp"] * seg["negbin_edge_pp"]
    seg["dist_wp_x_mkt_disagree"] = seg["ats_dist_wp"] * seg["mkt_disagree_dist"]

    all_feats = CORE_FEATURES + INTERACTION_FEATURES
    n = len(seg)
    print(f"  N={n}")

    if n < 50:
        print("  SKIPPED")
        return [], 0

    results = []
    trial_count = 0

    # Baseline: logistic on core only
    baseline_brier = cv_brier(seg, CORE_FEATURES, LogisticRegression,
                               {"max_iter": 1000, "solver": "lbfgs"})
    # Logistic with interactions
    lr_interact_brier = cv_brier(seg, all_feats, LogisticRegression,
                                  {"max_iter": 1000, "solver": "lbfgs"})
    trial_count += 1  # logistic interaction counts as a trial

    print(f"  Baseline Brier (core only):      {baseline_brier:.4f}" if not np.isnan(baseline_brier) else "  Baseline: N/A")
    print(f"  Logistic + interactions Brier:    {lr_interact_brier:.4f}" if not np.isnan(lr_interact_brier) else "  LR+interact: N/A")

    # SVM on interactions
    svm_params_grid = []
    for C in [0.1, 1, 10]:
        for gamma in ["scale", "auto", 0.01, 0.1]:
            svm_params_grid.append({"C": C, "gamma": gamma, "kernel": "rbf",
                                     "probability": True, "max_iter": 5000})

    best_svm_brier = np.inf
    for params in svm_params_grid:
        trial_count += 1
        b = cv_brier(seg, all_feats, SVC, params)
        if not np.isnan(b) and b < best_svm_brier:
            best_svm_brier = b

    svm_delta = best_svm_brier - baseline_brier if best_svm_brier < np.inf else np.nan
    lr_delta = lr_interact_brier - baseline_brier if not np.isnan(lr_interact_brier) else np.nan

    results.append({"test": "lr_interactions", "baseline": baseline_brier,
                    "model": lr_interact_brier, "delta": lr_delta})
    results.append({"test": "svm_interactions", "baseline": baseline_brier,
                    "model": best_svm_brier, "delta": svm_delta})

    print(f"  SVM + interactions Brier:        {best_svm_brier:.4f}, delta={svm_delta:+.4f}" if not np.isnan(svm_delta) else "  SVM+interact: N/A")

    return results, trial_count


# ── TEST 5C: PER-MATCHUP-TYPE SVM ────────────────────────────────────────────

def test5c_matchup_type(legs):
    print("\n" + "="*80)
    print("TEST 5C: PER-MATCHUP-TYPE SVM (fav-heavy vs even)")
    print("="*80)

    results = []
    trial_count = 0

    # Need spread_abs
    if "spread_abs" not in legs.columns:
        # Try to extract from rung
        legs = legs.copy()
        legs["spread_abs"] = legs["rung"]

    for label, mask_fn in [("even (spread<3)", lambda df: df["spread_abs"] < 3.0),
                            ("fav_heavy (spread>=3)", lambda df: df["spread_abs"] >= 3.0)]:
        seg = legs[mask_fn(legs)].reset_index(drop=True)
        n = len(seg)
        if n < 50:
            print(f"\n  {label}: N={n} — SKIPPED")
            continue

        wr = seg["y"].mean()
        print(f"\n  {label}: N={n}, WR={wr:.3f}")

        baseline_brier = cv_brier(seg, CORE_FEATURES, LogisticRegression,
                                   {"max_iter": 1000, "solver": "lbfgs"})

        svm_params_grid = []
        for C in [0.1, 1, 10]:
            for gamma in ["scale", "auto", 0.01, 0.1]:
                svm_params_grid.append({"C": C, "gamma": gamma, "kernel": "rbf",
                                         "probability": True, "max_iter": 5000})

        best_svm_brier = np.inf
        for params in svm_params_grid:
            trial_count += 1
            b = cv_brier(seg, CORE_FEATURES, SVC, params)
            if not np.isnan(b) and b < best_svm_brier:
                best_svm_brier = b

        delta = best_svm_brier - baseline_brier if best_svm_brier < np.inf else np.nan
        results.append({"segment": label, "N": n, "WR": wr,
                        "baseline": baseline_brier, "svm": best_svm_brier, "delta": delta})

        print(f"    Baseline Brier: {baseline_brier:.4f}" if not np.isnan(baseline_brier) else "    Baseline: N/A")
        print(f"    SVM Brier:      {best_svm_brier:.4f}" if best_svm_brier < np.inf else "    SVM: N/A")
        print(f"    Delta:          {delta:+.4f}" if not np.isnan(delta) else "    Delta: N/A")

    return results, trial_count


# ── GRAND SUMMARY ─────────────────────────────────────────────────────────────

def grand_summary(all_results, total_trials):
    print("\n" + "="*80)
    print("GRAND SUMMARY")
    print("="*80)
    print(f"\n  Total trials searched: {total_trials}")

    # Collect all deltas
    deltas = []
    for source, res_list in all_results:
        for r in res_list:
            d = r.get("delta", np.nan)
            if not np.isnan(d) and d != np.inf and d != -np.inf:
                deltas.append((source, r, d))

    if not deltas:
        print("  No valid results to summarize.")
        return

    # Sort by delta (most negative = best improvement)
    deltas.sort(key=lambda x: x[2])
    best_source, best_result, best_delta = deltas[0]

    print(f"\n  Best result: {best_source}")
    print(f"    Details: {best_result}")
    print(f"    Delta Brier: {best_delta:+.6f}")

    # Compute deflated p-value
    # Convert delta to Sharpe-like statistic
    # Under null, Brier differences ~ 0; use delta / SE approximation
    # Conservative: treat |delta| / typical_SE as z-score
    # With N tests, Bonferroni correction
    if best_delta < 0:
        # Rough z-score from Brier delta (assuming SE ~ 0.01 for moderate N)
        z_approx = abs(best_delta) / 0.01
        raw_p = 1 - stats.norm.cdf(z_approx)
        deflated_p = min(raw_p * total_trials, 1.0)
        print(f"\n  Approximate z-score: {z_approx:.2f}")
        print(f"  Raw p-value:         {raw_p:.6f}")
        print(f"  Deflated p-value (Bonferroni, {total_trials} trials): {deflated_p:.6f}")
        survives = deflated_p < 0.05
        print(f"\n  SURVIVES DEFLATION: {'YES' if survives else 'NO'}")
    else:
        print(f"\n  Best delta is non-negative ({best_delta:+.6f}) — no improvement found.")
        print("  SURVIVES DEFLATION: NO")

    # Top 5
    print("\n  Top 5 results (lowest delta Brier):")
    for i, (src, res, d) in enumerate(deltas[:5]):
        seg = res.get("segment", res.get("test", res.get("k", "?")))
        print(f"    {i+1}. [{src}] {seg}: delta={d:+.6f}")

    # Focus commentary
    print("\n  ── FOCUS COMMENTARY ──")
    rung_results = [r for s, rl in all_results for r in rl if s == "test1_per_rung"
                    and r.get("segment") in ["+1.5", "+2.5"]]
    if rung_results:
        print("  +1.5 / +2.5 SVM:")
        for r in rung_results:
            d = r.get("delta", np.nan)
            print(f"    {r['segment']}: N={r['N']}, delta={d:+.4f}" if not np.isnan(d) else f"    {r['segment']}: N={r['N']}, no result")
    else:
        print("  +1.5 / +2.5: no per-rung SVM results available")

    cell_err = [(r, r.get("delta", np.nan)) for s, rl in all_results
                for r in rl if s == "test3_error_cluster"]
    if cell_err:
        best_ce = min(cell_err, key=lambda x: x[1] if not np.isnan(x[1]) else np.inf)
        print(f"  Best per-cell error clustering: {best_ce[0].get('segment')} k={best_ce[0].get('k')}, delta={best_ce[1]:+.4f}")
    else:
        print("  Per-cell error clustering: no valid results")

    print("\n  CONCLUSION: ", end="")
    improvements = [d for _, _, d in deltas if d < 0]
    if not improvements:
        print("No granular slice produced Brier improvement. SVM and k-means remain DEAD at all levels tested.")
    elif deflated_p >= 0.05 if best_delta < 0 else True:
        print(f"Best raw improvement is {best_delta:+.4f} but does NOT survive multiple-testing correction across {total_trials} trials. SVM/k-means DEAD.")
    else:
        print(f"Improvement of {best_delta:+.4f} SURVIVES deflation. Investigate further.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    legs = load_data()
    print(f"Loaded {len(legs)} settled legs from {legs['game_date'].min().date()} to {legs['game_date'].max().date()}")
    print(f"Unique rungs: {sorted(legs['rung'].dropna().unique())}")
    print(f"Cells: {legs['negbin_cell'].value_counts().to_dict()}")
    print(f"Overall WR: {legs['y'].mean():.3f}")

    all_results = []
    total_trials = 0

    # Test 1
    r1, t1 = test1_per_rung_svm(legs)
    all_results.append(("test1_per_rung", r1))
    total_trials += t1

    # Test 2
    r2, t2 = test2_per_cell_svm(legs)
    all_results.append(("test2_per_cell", r2))
    total_trials += t2

    # Test 3
    r3, t3 = test3_per_cell_error_clustering(legs)
    all_results.append(("test3_error_cluster", r3))
    total_trials += t3

    # Test 4
    r4, t4 = test4_context_only_archetypes(legs)
    all_results.append(("test4_context_arch", r4))
    total_trials += t4

    # Test 5A
    r5a, t5a = test5a_parlay_level(legs)
    all_results.append(("test5a_parlay", r5a))
    total_trials += t5a

    # Test 5B
    r5b, t5b = test5b_interactions(legs)
    all_results.append(("test5b_interactions", r5b))
    total_trials += t5b

    # Test 5C
    r5c, t5c = test5c_matchup_type(legs)
    all_results.append(("test5c_matchup", r5c))
    total_trials += t5c

    # Grand summary
    grand_summary(all_results, total_trials)


if __name__ == "__main__":
    main()
