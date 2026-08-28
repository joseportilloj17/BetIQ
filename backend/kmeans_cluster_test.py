#!/usr/bin/env python3
"""
K-Means Clustering Test for MLB ATS Shadow Pipeline
====================================================
Three tests with temporal purged CV (5-fold, 3-day purge, date-level blocks).

Test 1: Latent Game Archetypes — do game-level features cluster into types that
        predict outcome beyond existing signals?
Test 2: Error/Residual Clustering — do large-residual legs cluster into
        systematic blind spots?
Test 3: Disagreement-Pattern Clustering — do types of model disagreement
        predict outcome beyond raw signal values?

Validation: OOS Brier improvement over baseline logistic regression on existing
signals, with deflated Sharpe for k-search multiple testing.
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, silhouette_score
from scipy import stats

warnings.filterwarnings("ignore")

DB_PATH = "data/bets.db"
PURGE_DAYS = 3
N_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load settled clean shadow_negbin legs with game-level features via
    fixture_id join."""
    conn = sqlite3.connect(DB_PATH)

    # Main query: shadow legs with signal columns
    legs_df = pd.read_sql_query("""
        SELECT l.id as leg_id, l.fixture_id, l.ats_dist_wp, l.bayes_team_wp,
               l.negbin_edge_pp, l.mkt_disagree_dist, l.ridge_composite,
               l.negbin_cell, l.negbin_mu_home, l.negbin_mu_away,
               l.leg_result, l.description,
               b.game_date
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE b.source = 'shadow_negbin'
          AND b.cohort = 'clean'
          AND b.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND l.leg_result IN ('WIN', 'LOSS')
          AND l.ats_dist_wp IS NOT NULL
          AND l.bayes_team_wp IS NOT NULL
    """, conn)

    # Game-level features via fixture_id join (take first match per fixture)
    features_df = pd.read_sql_query("""
        SELECT fixture_id,
               home_starter_era, home_starter_whip, home_starter_k9, home_starter_hr9,
               away_starter_era, away_starter_whip, away_starter_k9, away_starter_hr9,
               starter_era_diff, starter_whip_diff, starter_k9_diff,
               game_temp_f, game_wind_mph, wind_helps_scoring,
               home_team_xwoba, away_team_xwoba, xwoba_diff,
               home_team_barrel_pct, away_team_barrel_pct, barrel_diff,
               home_bullpen_era, away_bullpen_era, bullpen_era_diff,
               home_team_hard_hit_pct, away_team_hard_hit_pct, hard_hit_diff,
               spread_abs
        FROM mock_bet_legs
        WHERE home_starter_era IS NOT NULL
        GROUP BY fixture_id
    """, conn)
    conn.close()

    # Merge
    df = legs_df.merge(features_df, on="fixture_id", how="inner")
    df["y"] = (df["leg_result"] == "WIN").astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Encode negbin_cell as numeric
    cell_map = {"fav_home": 0, "fav_away": 1, "dog_home": 2, "dog_away": 3}
    df["cell_num"] = df["negbin_cell"].map(cell_map).fillna(-1)

    print(f"Loaded {len(df)} settled legs, date range: "
          f"{df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"Win rate: {df['y'].mean():.4f}")
    return df


# ---------------------------------------------------------------------------
# Temporal Purged CV
# ---------------------------------------------------------------------------

def temporal_purged_cv(df, n_folds=N_FOLDS, purge_days=PURGE_DAYS):
    """Generate temporal purged CV folds. Date-level blocks, 3-day purge."""
    dates = sorted(df["game_date"].unique())
    n_dates = len(dates)
    fold_size = n_dates // n_folds

    folds = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n_dates
        test_dates = set(dates[test_start:test_end])

        # Purge: remove dates within purge_days of any test date
        purge_td = pd.Timedelta(days=purge_days)
        train_dates = []
        for d in dates:
            if d in test_dates:
                continue
            too_close = False
            for td in test_dates:
                if abs(d - td) <= purge_td:
                    too_close = True
                    break
            if not too_close:
                train_dates.append(d)

        train_mask = df["game_date"].isin(train_dates)
        test_mask = df["game_date"].isin(test_dates)

        if train_mask.sum() > 20 and test_mask.sum() > 10:
            folds.append((train_mask, test_mask))

    return folds


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

SIGNAL_COLS = ["ats_dist_wp", "bayes_team_wp", "negbin_edge_pp",
               "mkt_disagree_dist", "ridge_composite", "cell_num"]


def baseline_brier(df, folds, signal_cols=SIGNAL_COLS):
    """Logistic regression baseline on existing signals, return per-fold OOS
    Brier scores."""
    briers = []
    for train_mask, test_mask in folds:
        X_tr = df.loc[train_mask, signal_cols].values
        y_tr = df.loc[train_mask, "y"].values
        X_te = df.loc[test_mask, signal_cols].values
        y_te = df.loc[test_mask, "y"].values

        # Handle NaNs: impute with column mean from training
        col_means = np.nanmean(X_tr, axis=0)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = col_means[c]
            X_te[np.isnan(X_te[:, c]), c] = col_means[c]

        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_tr, y_tr)
        p = lr.predict_proba(X_te)[:, 1]
        briers.append(brier_score_loss(y_te, p))
    return briers


# ---------------------------------------------------------------------------
# Deflated Sharpe for k-search
# ---------------------------------------------------------------------------

def deflated_sharpe(observed_sharpe, n_trials, n_obs, skew=0.0, kurt=3.0):
    """Bailey & Lopez de Prado deflated Sharpe ratio.
    Returns p-value: probability that observed Sharpe is a fluke given
    n_trials strategies tested."""
    # Expected max Sharpe under null
    e_max_sharpe = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + \
                   np.euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))

    # Standard error of Sharpe
    se = np.sqrt((1 - skew * observed_sharpe +
                  (kurt - 1) / 4.0 * observed_sharpe ** 2) / n_obs)

    # z-test: is observed Sharpe > expected max under null?
    if se < 1e-10:
        return 1.0
    z = (observed_sharpe - e_max_sharpe) / se
    return 1.0 - stats.norm.cdf(z)


# ---------------------------------------------------------------------------
# Test 1: Latent Game Archetypes
# ---------------------------------------------------------------------------

GAME_FEATURE_COLS = [
    "home_starter_era", "home_starter_whip", "home_starter_k9", "home_starter_hr9",
    "away_starter_era", "away_starter_whip", "away_starter_k9", "away_starter_hr9",
    "starter_era_diff", "starter_whip_diff", "starter_k9_diff",
    "home_team_xwoba", "away_team_xwoba", "xwoba_diff",
    "home_team_barrel_pct", "away_team_barrel_pct", "barrel_diff",
    "home_bullpen_era", "away_bullpen_era", "bullpen_era_diff",
    "negbin_mu_home", "negbin_mu_away", "spread_abs",
]


def test1_game_archetypes(df, folds):
    """Cluster game-level features; test if cluster membership adds OOS lift."""
    print("\n" + "=" * 70)
    print("TEST 1: LATENT GAME ARCHETYPES")
    print("=" * 70)

    # Drop rows with too many NaN game features
    feat_df = df[GAME_FEATURE_COLS].copy()
    valid_mask = feat_df.notna().sum(axis=1) >= len(GAME_FEATURE_COLS) * 0.6
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    print(f"Legs with >=60% game features: {len(df_valid)}/{len(df)}")

    if len(df_valid) < 100:
        print("INSUFFICIENT DATA for Test 1. SKIP.")
        return

    folds_valid = temporal_purged_cv(df_valid)
    base_briers = baseline_brier(df_valid, folds_valid)
    print(f"Baseline OOS Brier: {np.mean(base_briers):.6f} "
          f"(+/- {np.std(base_briers):.6f})")

    k_range = range(3, 13)
    best_k_per_fold = []
    all_briers = {k: [] for k in k_range}

    for fold_idx, (train_mask, test_mask) in enumerate(folds_valid):
        X_game_tr = df_valid.loc[train_mask, GAME_FEATURE_COLS].values
        X_game_te = df_valid.loc[test_mask, GAME_FEATURE_COLS].values

        # Impute NaN with training mean
        col_means = np.nanmean(X_game_tr, axis=0)
        for c in range(X_game_tr.shape[1]):
            X_game_tr[np.isnan(X_game_tr[:, c]), c] = col_means[c]
            X_game_te[np.isnan(X_game_te[:, c]), c] = col_means[c]

        scaler = StandardScaler()
        X_game_tr_s = scaler.fit_transform(X_game_tr)
        X_game_te_s = scaler.transform(X_game_te)

        # Pick k by silhouette on TRAINING fold only
        best_sil, best_k = -1, 3
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            labels = km.fit_predict(X_game_tr_s)
            sil = silhouette_score(X_game_tr_s, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k

        best_k_per_fold.append(best_k)

        # For each k, fit clusters on train, predict on test, add as feature
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            km.fit(X_game_tr_s)
            train_labels = km.predict(X_game_tr_s)
            test_labels = km.predict(X_game_te_s)

            # One-hot encode clusters
            train_dummies = np.eye(k)[train_labels]
            test_dummies = np.eye(k)[test_labels]

            # Augment signal features with cluster dummies
            X_sig_tr = df_valid.loc[train_mask, SIGNAL_COLS].values.copy()
            X_sig_te = df_valid.loc[test_mask, SIGNAL_COLS].values.copy()
            sig_means = np.nanmean(X_sig_tr, axis=0)
            for c in range(X_sig_tr.shape[1]):
                X_sig_tr[np.isnan(X_sig_tr[:, c]), c] = sig_means[c]
                X_sig_te[np.isnan(X_sig_te[:, c]), c] = sig_means[c]

            X_aug_tr = np.hstack([X_sig_tr, train_dummies])
            X_aug_te = np.hstack([X_sig_te, test_dummies])

            y_tr = df_valid.loc[train_mask, "y"].values
            y_te = df_valid.loc[test_mask, "y"].values

            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_aug_tr, y_tr)
            p = lr.predict_proba(X_aug_te)[:, 1]
            all_briers[k].append(brier_score_loss(y_te, p))

    # Results
    print(f"\nSilhouette-selected k per fold: {best_k_per_fold}")
    print(f"\nk  | Mean OOS Brier | Delta vs baseline")
    print("-" * 50)
    base_mean = np.mean(base_briers)
    best_delta, best_k_overall = 0, None
    for k in k_range:
        mean_b = np.mean(all_briers[k])
        delta = mean_b - base_mean
        marker = " <-- best" if delta == min(np.mean(all_briers[kk]) - base_mean
                                              for kk in k_range) else ""
        print(f"k={k:2d} | {mean_b:.6f}     | {delta:+.6f}{marker}")
        if delta < best_delta:
            best_delta = delta
            best_k_overall = k

    # Deflated Sharpe
    n_trials = len(k_range)
    if best_delta < 0:
        # Treat Brier improvement as a "return" series
        improvements = [base_briers[i] - all_briers[best_k_overall][i]
                        for i in range(len(folds_valid))]
        if np.std(improvements) > 0:
            sharpe = np.mean(improvements) / np.std(improvements)
        else:
            sharpe = 0
        p_val = deflated_sharpe(sharpe, n_trials, len(df_valid))
        print(f"\nBest k={best_k_overall}: delta Brier = {best_delta:+.6f}")
        print(f"Deflated Sharpe: {sharpe:.3f}, p-value (deflated): {p_val:.4f}")
        print(f"k values searched: {n_trials}")
        if p_val < 0.05:
            print("VERDICT: ADDITIVE — cluster membership adds OOS lift beyond "
                  "existing signals (survives deflation)")
        else:
            print("VERDICT: DEAD — improvement does not survive multiple-testing "
                  "deflation (re-encodes known structure)")
    else:
        print(f"\nNo k improved over baseline.")
        print("VERDICT: DEAD — game archetypes re-encode existing signal structure")


# ---------------------------------------------------------------------------
# Test 2: Error/Residual Clustering
# ---------------------------------------------------------------------------

def test2_residual_clustering(df, folds):
    """Cluster features of high-residual legs; test if cluster predicts
    future residual magnitude or sign OOS."""
    print("\n" + "=" * 70)
    print("TEST 2: ERROR/RESIDUAL CLUSTERING")
    print("=" * 70)

    df = df.copy()
    df["residual"] = df["y"] - df["ats_dist_wp"]
    df["abs_residual"] = df["residual"].abs()

    # Features for clustering: signal cols + available game cols
    cluster_features = SIGNAL_COLS + [c for c in GAME_FEATURE_COLS
                                       if c in df.columns]
    # Drop features with >50% NaN
    valid_feats = [c for c in cluster_features
                   if df[c].notna().mean() > 0.5]
    print(f"Clustering features ({len(valid_feats)}): {valid_feats}")

    folds_local = temporal_purged_cv(df)

    k_range = range(3, 11)
    # Two tests: (a) predict |residual| magnitude, (b) predict residual sign

    # Test A: does cluster membership predict whether a leg will be in the
    # high-residual quartile?
    print("\n--- Test 2A: Predicting high-|residual| OOS ---")
    base_briers_2a = []
    cluster_briers_2a = {k: [] for k in k_range}

    for fold_idx, (train_mask, test_mask) in enumerate(folds_local):
        y_tr_raw = df.loc[train_mask, "abs_residual"].values
        y_te_raw = df.loc[test_mask, "abs_residual"].values

        # High-residual = top quartile of training residuals
        q75 = np.percentile(y_tr_raw, 75)
        y_tr_high = (y_tr_raw >= q75).astype(int)
        y_te_high = (y_te_raw >= q75).astype(int)

        # Baseline: predict high-residual from signal cols
        X_sig_tr = df.loc[train_mask, SIGNAL_COLS].values.copy()
        X_sig_te = df.loc[test_mask, SIGNAL_COLS].values.copy()
        sig_means = np.nanmean(X_sig_tr, axis=0)
        for c in range(X_sig_tr.shape[1]):
            X_sig_tr[np.isnan(X_sig_tr[:, c]), c] = sig_means[c]
            X_sig_te[np.isnan(X_sig_te[:, c]), c] = sig_means[c]

        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_sig_tr, y_tr_high)
        p_base = lr.predict_proba(X_sig_te)[:, 1]
        base_briers_2a.append(brier_score_loss(y_te_high, p_base))

        # Cluster the HIGH-residual training legs, assign all legs to clusters
        high_mask_tr = y_tr_raw >= q75
        X_feat_tr = df.loc[train_mask, valid_feats].values.copy()
        X_feat_te = df.loc[test_mask, valid_feats].values.copy()
        feat_means = np.nanmean(X_feat_tr, axis=0)
        for c in range(X_feat_tr.shape[1]):
            X_feat_tr[np.isnan(X_feat_tr[:, c]), c] = feat_means[c]
            X_feat_te[np.isnan(X_feat_te[:, c]), c] = feat_means[c]

        scaler = StandardScaler()
        X_feat_tr_s = scaler.fit_transform(X_feat_tr)
        X_feat_te_s = scaler.transform(X_feat_te)

        for k in k_range:
            # Fit on high-residual training legs only
            if high_mask_tr.sum() < k:
                cluster_briers_2a[k].append(base_briers_2a[-1])
                continue

            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            km.fit(X_feat_tr_s[high_mask_tr])

            # Assign ALL legs to nearest cluster
            train_labels = km.predict(X_feat_tr_s)
            test_labels = km.predict(X_feat_te_s)

            train_dummies = np.eye(k)[train_labels]
            test_dummies = np.eye(k)[test_labels]

            X_aug_tr = np.hstack([X_sig_tr, train_dummies])
            X_aug_te = np.hstack([X_sig_te, test_dummies])

            lr2 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr2.fit(X_aug_tr, y_tr_high)
            p_aug = lr2.predict_proba(X_aug_te)[:, 1]
            cluster_briers_2a[k].append(brier_score_loss(y_te_high, p_aug))

    base_mean_2a = np.mean(base_briers_2a)
    print(f"Baseline OOS Brier (predict high-|residual|): {base_mean_2a:.6f}")
    print(f"\nk  | Mean OOS Brier | Delta vs baseline")
    print("-" * 50)
    for k in k_range:
        mean_b = np.mean(cluster_briers_2a[k])
        delta = mean_b - base_mean_2a
        print(f"k={k:2d} | {mean_b:.6f}     | {delta:+.6f}")

    # Test B: does error cluster predict OUTCOME (direct Brier improvement)?
    print("\n--- Test 2B: Error clusters predicting outcome directly ---")
    base_briers_2b = baseline_brier(df, folds_local)
    cluster_briers_2b = {k: [] for k in k_range}

    for fold_idx, (train_mask, test_mask) in enumerate(folds_local):
        y_tr = df.loc[train_mask, "y"].values
        y_te = df.loc[test_mask, "y"].values

        # Cluster on residual-feature space
        X_feat_tr = df.loc[train_mask, valid_feats].values.copy()
        X_feat_te = df.loc[test_mask, valid_feats].values.copy()
        feat_means = np.nanmean(X_feat_tr, axis=0)
        for c in range(X_feat_tr.shape[1]):
            X_feat_tr[np.isnan(X_feat_tr[:, c]), c] = feat_means[c]
            X_feat_te[np.isnan(X_feat_te[:, c]), c] = feat_means[c]

        # Include residual as clustering feature (training only)
        residuals_tr = df.loc[train_mask, "residual"].values.reshape(-1, 1)
        X_cluster_tr = np.hstack([X_feat_tr, residuals_tr])

        scaler = StandardScaler()
        X_cluster_tr_s = scaler.fit_transform(X_cluster_tr)
        # For test: residual unknown, use 0 (mean after standardization)
        X_cluster_te = np.hstack([X_feat_te, np.zeros((X_feat_te.shape[0], 1))])
        X_cluster_te_s = scaler.transform(X_cluster_te)

        X_sig_tr = df.loc[train_mask, SIGNAL_COLS].values.copy()
        X_sig_te = df.loc[test_mask, SIGNAL_COLS].values.copy()
        sig_means = np.nanmean(X_sig_tr, axis=0)
        for c in range(X_sig_tr.shape[1]):
            X_sig_tr[np.isnan(X_sig_tr[:, c]), c] = sig_means[c]
            X_sig_te[np.isnan(X_sig_te[:, c]), c] = sig_means[c]

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            km.fit(X_cluster_tr_s)
            train_labels = km.predict(X_cluster_tr_s)
            test_labels = km.predict(X_cluster_te_s)

            train_dummies = np.eye(k)[train_labels]
            test_dummies = np.eye(k)[test_labels]

            X_aug_tr = np.hstack([X_sig_tr, train_dummies])
            X_aug_te = np.hstack([X_sig_te, test_dummies])

            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_aug_tr, y_tr)
            p = lr.predict_proba(X_aug_te)[:, 1]
            cluster_briers_2b[k].append(brier_score_loss(y_te, p))

    base_mean_2b = np.mean(base_briers_2b)
    print(f"Baseline OOS Brier (outcome): {base_mean_2b:.6f}")
    print(f"\nk  | Mean OOS Brier | Delta vs baseline")
    print("-" * 50)
    best_delta_2b = 0
    best_k_2b = None
    for k in k_range:
        mean_b = np.mean(cluster_briers_2b[k])
        delta = mean_b - base_mean_2b
        print(f"k={k:2d} | {mean_b:.6f}     | {delta:+.6f}")
        if delta < best_delta_2b:
            best_delta_2b = delta
            best_k_2b = k

    n_trials = len(k_range) * 2  # searched k over 2 sub-tests
    if best_delta_2b < 0 and best_k_2b is not None:
        improvements = [base_briers_2b[i] - cluster_briers_2b[best_k_2b][i]
                        for i in range(len(folds_local))]
        sharpe = np.mean(improvements) / (np.std(improvements) + 1e-10)
        p_val = deflated_sharpe(sharpe, n_trials, len(df))
        print(f"\nBest k={best_k_2b}: delta Brier = {best_delta_2b:+.6f}")
        print(f"Deflated Sharpe: {sharpe:.3f}, p-value: {p_val:.4f}")
        print(f"k values searched: {n_trials}")
        if p_val < 0.05:
            print("VERDICT: ADDITIVE — systematic blind spot detected, "
                  "error clusters predict outcome OOS")
        else:
            print("VERDICT: DEAD — errors are random, no systematic blind spot "
                  "survives deflation")
    else:
        print("\nNo k improved over baseline.")
        print("VERDICT: DEAD — errors are random, no systematic blind spot")


# ---------------------------------------------------------------------------
# Test 3: Disagreement-Pattern Clustering
# ---------------------------------------------------------------------------

def test3_disagreement_patterns(df, folds):
    """Cluster disagreement vectors; test if disagreement types predict
    outcome beyond raw signal values."""
    print("\n" + "=" * 70)
    print("TEST 3: DISAGREEMENT-PATTERN CLUSTERING")
    print("=" * 70)

    disagree_cols = ["ats_dist_wp", "bayes_team_wp", "negbin_edge_pp",
                     "mkt_disagree_dist"]

    # Check completeness
    valid_mask = df[disagree_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    print(f"Legs with all disagreement signals: {len(df_valid)}/{len(df)}")

    if len(df_valid) < 100:
        print("INSUFFICIENT DATA for Test 3. SKIP.")
        return

    folds_valid = temporal_purged_cv(df_valid)
    base_briers = baseline_brier(df_valid, folds_valid)
    base_mean = np.mean(base_briers)
    print(f"Baseline OOS Brier: {base_mean:.6f} (+/- {np.std(base_briers):.6f})")

    k_range = range(3, 11)
    all_briers = {k: [] for k in k_range}
    # Collect per-cluster stats for the silhouette-best k
    cluster_stats_best = {}

    for fold_idx, (train_mask, test_mask) in enumerate(folds_valid):
        X_dis_tr = df_valid.loc[train_mask, disagree_cols].values.copy()
        X_dis_te = df_valid.loc[test_mask, disagree_cols].values.copy()

        scaler = StandardScaler()
        X_dis_tr_s = scaler.fit_transform(X_dis_tr)
        X_dis_te_s = scaler.transform(X_dis_te)

        # Pick k by silhouette on training
        best_sil, best_k_fold = -1, 3
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            labels = km.fit_predict(X_dis_tr_s)
            sil = silhouette_score(X_dis_tr_s, labels)
            if sil > best_sil:
                best_sil = sil
                best_k_fold = k

        X_sig_tr = df_valid.loc[train_mask, SIGNAL_COLS].values.copy()
        X_sig_te = df_valid.loc[test_mask, SIGNAL_COLS].values.copy()
        sig_means = np.nanmean(X_sig_tr, axis=0)
        for c in range(X_sig_tr.shape[1]):
            X_sig_tr[np.isnan(X_sig_tr[:, c]), c] = sig_means[c]
            X_sig_te[np.isnan(X_sig_te[:, c]), c] = sig_means[c]

        y_tr = df_valid.loc[train_mask, "y"].values
        y_te = df_valid.loc[test_mask, "y"].values

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            km.fit(X_dis_tr_s)
            train_labels = km.predict(X_dis_tr_s)
            test_labels = km.predict(X_dis_te_s)

            # Collect per-cluster stats for last fold
            if fold_idx == len(folds_valid) - 1 and k == best_k_fold:
                for cl in range(k):
                    cl_mask = test_labels == cl
                    if cl_mask.sum() > 0:
                        cl_wr = y_te[cl_mask].mean()
                        cl_n = cl_mask.sum()
                        # Get mean disagree values for this cluster
                        cl_centers = km.cluster_centers_[cl]
                        cl_centers_orig = scaler.inverse_transform(
                            cl_centers.reshape(1, -1))[0]
                        cluster_stats_best[cl] = {
                            "wr": cl_wr, "n": cl_n,
                            "dist_wp": cl_centers_orig[0],
                            "bayes": cl_centers_orig[1],
                            "negbin_edge": cl_centers_orig[2],
                            "mkt_disagree": cl_centers_orig[3],
                        }

            train_dummies = np.eye(k)[train_labels]
            test_dummies = np.eye(k)[test_labels]

            X_aug_tr = np.hstack([X_sig_tr, train_dummies])
            X_aug_te = np.hstack([X_sig_te, test_dummies])

            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_aug_tr, y_tr)
            p = lr.predict_proba(X_aug_te)[:, 1]
            all_briers[k].append(brier_score_loss(y_te, p))

    # Results
    print(f"\nk  | Mean OOS Brier | Delta vs baseline")
    print("-" * 50)
    best_delta, best_k_overall = 0, None
    for k in k_range:
        mean_b = np.mean(all_briers[k])
        delta = mean_b - base_mean
        print(f"k={k:2d} | {mean_b:.6f}     | {delta:+.6f}")
        if delta < best_delta:
            best_delta = delta
            best_k_overall = k

    # Per-cluster stats
    if cluster_stats_best:
        print(f"\nPer-cluster OOS stats (last fold, silhouette-best k):")
        print(f"{'Cluster':>8} | {'N':>4} | {'WR':>6} | {'dist_wp':>8} | "
              f"{'bayes':>7} | {'neg_edge':>8} | {'mkt_dis':>7}")
        print("-" * 70)
        for cl in sorted(cluster_stats_best.keys()):
            s = cluster_stats_best[cl]
            print(f"   {cl:>5} | {s['n']:>4} | {s['wr']:.3f} | "
                  f"{s['dist_wp']:>8.4f} | {s['bayes']:>7.4f} | "
                  f"{s['negbin_edge']:>8.2f} | {s['mkt_disagree']:>7.4f}")

    # Deflated Sharpe
    n_trials = len(k_range)
    if best_delta < 0 and best_k_overall is not None:
        improvements = [base_briers[i] - all_briers[best_k_overall][i]
                        for i in range(len(folds_valid))]
        sharpe = np.mean(improvements) / (np.std(improvements) + 1e-10)
        p_val = deflated_sharpe(sharpe, n_trials, len(df_valid))
        print(f"\nBest k={best_k_overall}: delta Brier = {best_delta:+.6f}")
        print(f"Deflated Sharpe: {sharpe:.3f}, p-value (deflated): {p_val:.4f}")
        print(f"k values searched: {n_trials}")
        if p_val < 0.05:
            print("VERDICT: ADDITIVE — disagreement pattern types predict "
                  "outcome beyond raw signals (survives deflation)")
        else:
            print("VERDICT: DEAD — disagreement clusters do not survive "
                  "multiple-testing deflation")
    else:
        print(f"\nNo k improved over baseline.")
        print("VERDICT: DEAD — disagreement types do not predict outcome "
              "beyond raw signal values")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("K-MEANS CLUSTERING TEST — MLB ATS SHADOW PIPELINE")
    print("=" * 70)
    print(f"CV: {N_FOLDS}-fold temporal purged, {PURGE_DAYS}-day purge, "
          "date-level blocks")
    print(f"Deflation: Bailey-Lopez de Prado deflated Sharpe for k-search")
    print()

    df = load_data()
    folds = temporal_purged_cv(df)
    print(f"CV folds: {len(folds)}")
    for i, (tr, te) in enumerate(folds):
        tr_dates = df.loc[tr, "game_date"]
        te_dates = df.loc[te, "game_date"]
        print(f"  Fold {i}: train={tr.sum()} ({tr_dates.min().date()}-"
              f"{tr_dates.max().date()}), "
              f"test={te.sum()} ({te_dates.min().date()}-"
              f"{te_dates.max().date()})")

    test1_game_archetypes(df, folds)
    test2_residual_clustering(df, folds)
    test3_disagreement_patterns(df, folds)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
