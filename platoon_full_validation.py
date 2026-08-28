"""
platoon_full_validation.py — Full validation gauntlet for platoon gap ATS feature.

Tests:
  1. D vs D+E combined (10 seeds, same protocol as platoon_test.py)
  2. TimeSeriesSplit CV (5 folds on all 2026 data)
  3. Random 50/50 split (5 seeds, train on pre-5/26, test halves post-5/26)
  4. Bootstrap CI on WR difference (D minus baseline)
  5. Game-level dedup verification

Context:
  - Model D (+platoon gap: h_platoon_gap, a_platoon_gap, platoon_gap_diff)
    showed Late AUC +0.014, Late Brier -0.0010, Late WR +4.3pp (p=0.0015)
  - Model E (+SP handedness: h_sp_is_lhp, a_sp_is_lhp) showed minimal lift
  - This script tests whether D+E combined is better than D alone,
    validates with multiple CV strategies, and computes honest CIs.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))

from features import build_feature_matrix, MODEL_FEATURE_COLS_V1_PRUNED, TARGET_ATS

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

N_SEEDS = 10
TRAIN_CUTOFF = "2026-05-26"
EARLY_START = "2026-05-26"
EARLY_END = "2026-06-15"
LATE_START = "2026-07-01"
LATE_END = "2026-08-05"

GBC_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    min_samples_leaf=5,
)

BASELINE = list(MODEL_FEATURE_COLS_V1_PRUNED)  # 21 features
MODEL_D_FEATS = BASELINE + ["h_platoon_gap", "a_platoon_gap", "platoon_gap_diff"]
MODEL_DE_FEATS = BASELINE + [
    "h_platoon_gap", "a_platoon_gap", "platoon_gap_diff",
    "h_sp_is_lhp", "a_sp_is_lhp",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ci(n, w, z=1.96):
    """Wilson score interval for win rate."""
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (round((centre - margin) * 100, 2), round((centre + margin) * 100, 2))


def train_calibrated_gbc(X_train, y_train, seed):
    """Train GBC + isotonic calibration. Returns calibrated model."""
    gbc = GradientBoostingClassifier(**GBC_PARAMS, random_state=seed)
    cal = CalibratedClassifierCV(gbc, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


def prep_features(train_df, test_df, feature_cols):
    """Impute + scale features. Returns X_train, X_test."""
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(
        train_df[feature_cols].values.astype(float)
    ))
    X_te = scaler.transform(imp.transform(
        test_df[feature_cols].values.astype(float)
    ))
    return X_tr, X_te


def evaluate(model, X_test, y_test):
    """Return dict of AUC, Brier, accuracy, WR (all games, threshold=0.5)."""
    proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = np.nan
    brier = brier_score_loss(y_test, proba)
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    wr = y_test[preds == 1].mean() if preds.sum() > 0 else np.nan
    n_picks = int(preds.sum())
    return {
        "auc": auc, "brier": brier, "acc": acc,
        "wr": wr, "n_picks": n_picks, "proba": proba, "preds": preds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load feature matrix and join platoon features."""
    print("=" * 110)
    print("  PLATOON FULL VALIDATION GAUNTLET")
    print("=" * 110)

    print("\n  Loading feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"  Feature matrix: {len(df)} games")

    print("  Loading platoon features...")
    platoon = pd.read_csv(os.path.join(_HERE, "data", "platoon_features.csv"))
    print(f"  Platoon CSV: {len(platoon)} games")

    platoon_cols = [
        "game_id",
        "h_sp_is_lhp", "a_sp_is_lhp",
        "h_platoon_gap", "a_platoon_gap", "platoon_gap_diff",
    ]
    # Keep only columns that exist in platoon CSV
    platoon_cols = [c for c in platoon_cols if c in platoon.columns]
    df = df.merge(platoon[platoon_cols], on="game_id", how="left")

    # Report fill rates
    print("\n  Platoon feature fill rates:")
    for col in platoon_cols[1:]:
        fill = df[col].notna().mean()
        print(f"    {col:25s}: {fill * 100:.1f}%")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df[TARGET_ATS].notna()].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"\n  Games with ATS labels: {len(df)}")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: D vs D+E COMBINED (10 seeds)
# ═══════════════════════════════════════════════════════════════════════════════

def test_d_vs_de(df):
    """Train baseline, D, and D+E on pre-5/26, test on early and late."""
    print("\n" + "=" * 110)
    print("  TEST 1: D vs D+E COMBINED (10 seeds)")
    print("=" * 110)

    train = df[df["game_date"] < TRAIN_CUTOFF].copy()
    early = df[(df["game_date"] >= EARLY_START) & (df["game_date"] <= EARLY_END)].copy()
    late = df[(df["game_date"] >= LATE_START) & (df["game_date"] <= LATE_END)].copy()

    print(f"  Train: {len(train)} games")
    print(f"  Early test: {len(early)} games ({EARLY_START} to {EARLY_END})")
    print(f"  Late test:  {len(late)} games ({LATE_START} to {LATE_END})")

    # Dedup check
    print(f"\n  [Dedup check]")
    if "game_id" in late.columns:
        print(f"    Late: {len(late)} rows, {late['game_id'].nunique()} unique game_ids")
    if "game_id" in early.columns:
        print(f"    Early: {len(early)} rows, {early['game_id'].nunique()} unique game_ids")

    y_train = train[TARGET_ATS].values
    y_early = early[TARGET_ATS].values
    y_late = late[TARGET_ATS].values

    specs = {
        "Baseline (21)": BASELINE,
        "D: +Gap (24)": MODEL_D_FEATS,
        "DE: +Gap+Hand (26)": MODEL_DE_FEATS,
    }

    results = {}
    # Also store per-seed late probas for bootstrap CI later
    per_seed_late = {}

    t0 = time.time()
    for name, feats in specs.items():
        print(f"\n  Training: {name}...")
        results[name] = {
            "early_auc": [], "early_brier": [], "early_wr": [],
            "late_auc": [], "late_brier": [], "late_wr": [],
            "late_n_picks": [],
        }
        per_seed_late[name] = {"proba": [], "preds": []}

        for seed in range(N_SEEDS):
            X_tr, X_early = prep_features(train, early, feats)
            _, X_late = prep_features(train, late, feats)
            model = train_calibrated_gbc(X_tr, y_train, seed=seed * 7 + 42)

            e_res = evaluate(model, X_early, y_early)
            l_res = evaluate(model, X_late, y_late)

            results[name]["early_auc"].append(e_res["auc"])
            results[name]["early_brier"].append(e_res["brier"])
            results[name]["early_wr"].append(e_res["wr"])
            results[name]["late_auc"].append(l_res["auc"])
            results[name]["late_brier"].append(l_res["brier"])
            results[name]["late_wr"].append(l_res["wr"])
            results[name]["late_n_picks"].append(l_res["n_picks"])
            per_seed_late[name]["proba"].append(l_res["proba"])
            per_seed_late[name]["preds"].append(l_res["preds"])

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")

    # Print results table
    print(f"\n  {'Model':>22s}  {'Early AUC':>10}  {'Late AUC':>10}  {'dAUC':>7}  "
          f"{'Early Brier':>12}  {'Late Brier':>12}  {'dBrier':>8}  "
          f"{'Early WR':>9}  {'Late WR':>9}  {'dWR':>7}")
    print("  " + "-" * 120)

    bl_late_auc = np.nanmean(results["Baseline (21)"]["late_auc"])
    bl_late_brier = np.nanmean(results["Baseline (21)"]["late_brier"])
    bl_late_wr = np.nanmean(results["Baseline (21)"]["late_wr"])

    for name, r in results.items():
        ea = np.nanmean(r["early_auc"])
        la = np.nanmean(r["late_auc"])
        eb = np.nanmean(r["early_brier"])
        lb = np.nanmean(r["late_brier"])
        ewr = np.nanmean(r["early_wr"])
        lwr = np.nanmean(r["late_wr"])
        d_la = la - bl_late_auc
        d_lb = lb - bl_late_brier
        d_lwr = (lwr - bl_late_wr) * 100

        print(f"  {name:>22s}  {ea:.4f}      {la:.4f}      {d_la:>+.4f}  "
              f"{eb:.6f}      {lb:.6f}      {d_lb:>+.6f}  "
              f"{ewr * 100:>6.1f}%    {lwr * 100:>6.1f}%    {d_lwr:>+.1f}pp")

    # Paired t-tests
    print("\n  Paired t-tests (10-seed late Brier) vs baseline:")
    bl_brier_arr = results["Baseline (21)"]["late_brier"]
    for name in ["D: +Gap (24)", "DE: +Gap+Hand (26)"]:
        t_stat, p_val = stats.ttest_rel(bl_brier_arr, results[name]["late_brier"])
        print(f"    {name}: t={t_stat:+.3f}, p={p_val:.4f} "
              f"{'** SIGNIFICANT' if p_val < 0.05 else '* marginal' if p_val < 0.10 else 'not significant'}")

    print("\n  Paired t-tests (10-seed late WR) vs baseline:")
    bl_wr_arr = results["Baseline (21)"]["late_wr"]
    for name in ["D: +Gap (24)", "DE: +Gap+Hand (26)"]:
        t_stat, p_val = stats.ttest_rel(bl_wr_arr, results[name]["late_wr"])
        print(f"    {name}: t={t_stat:+.3f}, p={p_val:.4f} "
              f"{'** SIGNIFICANT' if p_val < 0.05 else '* marginal' if p_val < 0.10 else 'not significant'}")

    return results, per_seed_late, y_late, late


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: TimeSeriesSplit CV (5 folds)
# ═══════════════════════════════════════════════════════════════════════════════

def test_timeseries_cv(df):
    """TimeSeriesSplit with 5 folds on all 2026 data."""
    print("\n" + "=" * 110)
    print("  TEST 2: TimeSeriesSplit CV (5 folds, all 2026 data)")
    print("=" * 110)

    # Use all 2026 data sorted chronologically
    df_2026 = df[df["game_date"].dt.year == 2026].copy()
    df_2026 = df_2026.sort_values("game_date").reset_index(drop=True)
    print(f"  2026 games: {len(df_2026)}")
    print(f"  Date range: {df_2026['game_date'].min().date()} to {df_2026['game_date'].max().date()}")

    # Determine which model to test (D is the candidate from Test 1)
    specs = {
        "Baseline (21)": BASELINE,
        "D: +Gap (24)": MODEL_D_FEATS,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_2026)):
        fold_train = df_2026.iloc[train_idx]
        fold_test = df_2026.iloc[test_idx]
        y_tr = fold_train[TARGET_ATS].values
        y_te = fold_test[TARGET_ATS].values

        train_dates = f"{fold_train['game_date'].min().date()} to {fold_train['game_date'].max().date()}"
        test_dates = f"{fold_test['game_date'].min().date()} to {fold_test['game_date'].max().date()}"
        print(f"\n  Fold {fold_idx + 1}: train={len(fold_train)} ({train_dates}), "
              f"test={len(fold_test)} ({test_dates})")

        fold_row = {"fold": fold_idx + 1, "n_train": len(fold_train), "n_test": len(fold_test)}

        for name, feats in specs.items():
            # Single seed (median of 3 to reduce noise but keep it tractable)
            aucs, briers = [], []
            for seed in [42, 49, 56]:
                X_tr, X_te = prep_features(fold_train, fold_test, feats)
                model = train_calibrated_gbc(X_tr, y_tr, seed=seed)
                res = evaluate(model, X_te, y_te)
                aucs.append(res["auc"])
                briers.append(res["brier"])

            fold_row[f"{name}_auc"] = np.nanmean(aucs)
            fold_row[f"{name}_brier"] = np.nanmean(briers)

        fold_results.append(fold_row)

    # Print fold results
    print(f"\n  {'Fold':>5}  {'N_test':>6}  {'BL AUC':>8}  {'D AUC':>8}  {'dAUC':>7}  "
          f"{'BL Brier':>10}  {'D Brier':>10}  {'dBrier':>8}")
    print("  " + "-" * 80)

    for row in fold_results:
        bl_a = row["Baseline (21)_auc"]
        d_a = row["D: +Gap (24)_auc"]
        bl_b = row["Baseline (21)_brier"]
        d_b = row["D: +Gap (24)_brier"]
        print(f"  {row['fold']:>5d}  {row['n_test']:>6d}  {bl_a:>8.4f}  {d_a:>8.4f}  {d_a - bl_a:>+.4f}  "
              f"{bl_b:>10.6f}  {d_b:>10.6f}  {d_b - bl_b:>+.6f}")

    # Means
    mean_bl_auc = np.mean([r["Baseline (21)_auc"] for r in fold_results])
    mean_d_auc = np.mean([r["D: +Gap (24)_auc"] for r in fold_results])
    mean_bl_brier = np.mean([r["Baseline (21)_brier"] for r in fold_results])
    mean_d_brier = np.mean([r["D: +Gap (24)_brier"] for r in fold_results])

    print(f"\n  {'MEAN':>5}  {'':>6}  {mean_bl_auc:>8.4f}  {mean_d_auc:>8.4f}  "
          f"{mean_d_auc - mean_bl_auc:>+.4f}  "
          f"{mean_bl_brier:>10.6f}  {mean_d_brier:>10.6f}  "
          f"{mean_d_brier - mean_bl_brier:>+.6f}")

    # Is the delta consistently positive across folds?
    auc_deltas = [r["D: +Gap (24)_auc"] - r["Baseline (21)_auc"] for r in fold_results]
    brier_deltas = [r["D: +Gap (24)_brier"] - r["Baseline (21)_brier"] for r in fold_results]
    n_auc_pos = sum(1 for d in auc_deltas if d > 0)
    n_brier_neg = sum(1 for d in brier_deltas if d < 0)
    print(f"\n  AUC delta positive in {n_auc_pos}/5 folds")
    print(f"  Brier delta negative (better) in {n_brier_neg}/5 folds")

    return fold_results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Random 50/50 Split (5 seeds)
# ═══════════════════════════════════════════════════════════════════════════════

def test_random_splits(df):
    """Train on pre-5/26, randomly split post-5/26 into halves, test each."""
    print("\n" + "=" * 110)
    print("  TEST 3: Random 50/50 Split (5 seeds)")
    print("=" * 110)

    train = df[df["game_date"] < TRAIN_CUTOFF].copy()
    test_pool = df[df["game_date"] >= TRAIN_CUTOFF].copy()
    print(f"  Train: {len(train)} games")
    print(f"  Test pool (post {TRAIN_CUTOFF}): {len(test_pool)} games")

    y_train = train[TARGET_ATS].values

    specs = {
        "Baseline (21)": BASELINE,
        "D: +Gap (24)": MODEL_D_FEATS,
    }

    split_results = []

    for split_seed in range(5):
        rng = np.random.RandomState(split_seed * 13 + 7)
        idx = rng.permutation(len(test_pool))
        mid = len(idx) // 2
        half_a = test_pool.iloc[idx[:mid]]
        half_b = test_pool.iloc[idx[mid:]]

        row = {"split_seed": split_seed}

        for name, feats in specs.items():
            # Train once (single seed 42)
            X_tr, X_a = prep_features(train, half_a, feats)
            _, X_b = prep_features(train, half_b, feats)
            model = train_calibrated_gbc(X_tr, y_train, seed=42)

            res_a = evaluate(model, X_a, half_a[TARGET_ATS].values)
            res_b = evaluate(model, X_b, half_b[TARGET_ATS].values)

            row[f"{name}_a_auc"] = res_a["auc"]
            row[f"{name}_b_auc"] = res_b["auc"]
            row[f"{name}_a_brier"] = res_a["brier"]
            row[f"{name}_b_brier"] = res_b["brier"]
            row[f"{name}_a_wr"] = res_a["wr"]
            row[f"{name}_b_wr"] = res_b["wr"]

        split_results.append(row)

    # Print results
    print(f"\n  {'Seed':>5}  {'Half-A dAUC':>12}  {'Half-B dAUC':>12}  "
          f"{'Half-A dBrier':>14}  {'Half-B dBrier':>14}  "
          f"{'Half-A dWR':>11}  {'Half-B dWR':>11}")
    print("  " + "-" * 95)

    for row in split_results:
        d_a_auc = row["D: +Gap (24)_a_auc"] - row["Baseline (21)_a_auc"]
        d_b_auc = row["D: +Gap (24)_b_auc"] - row["Baseline (21)_b_auc"]
        d_a_brier = row["D: +Gap (24)_a_brier"] - row["Baseline (21)_a_brier"]
        d_b_brier = row["D: +Gap (24)_b_brier"] - row["Baseline (21)_b_brier"]
        d_a_wr = (row["D: +Gap (24)_a_wr"] - row["Baseline (21)_a_wr"]) * 100
        d_b_wr = (row["D: +Gap (24)_b_wr"] - row["Baseline (21)_b_wr"]) * 100
        print(f"  {row['split_seed']:>5d}  {d_a_auc:>+12.4f}  {d_b_auc:>+12.4f}  "
              f"{d_a_brier:>+14.6f}  {d_b_brier:>+14.6f}  "
              f"{d_a_wr:>+9.1f}pp  {d_b_wr:>+9.1f}pp")

    # Summary: how often does D help in BOTH halves?
    both_auc = 0
    both_brier = 0
    for row in split_results:
        d_a = row["D: +Gap (24)_a_auc"] - row["Baseline (21)_a_auc"]
        d_b = row["D: +Gap (24)_b_auc"] - row["Baseline (21)_b_auc"]
        if d_a > 0 and d_b > 0:
            both_auc += 1
        d_a_b = row["D: +Gap (24)_a_brier"] - row["Baseline (21)_a_brier"]
        d_b_b = row["D: +Gap (24)_b_brier"] - row["Baseline (21)_b_brier"]
        if d_a_b < 0 and d_b_b < 0:
            both_brier += 1

    print(f"\n  AUC positive in BOTH halves: {both_auc}/5 splits")
    print(f"  Brier negative in BOTH halves: {both_brier}/5 splits")

    return split_results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Confidence Interval on WR Difference
# ═══════════════════════════════════════════════════════════════════════════════

def test_confidence_intervals(per_seed_late, y_late, late_df):
    """Bootstrap CI on WR difference (D minus baseline)."""
    print("\n" + "=" * 110)
    print("  TEST 4: Confidence Intervals on WR Improvement")
    print("=" * 110)

    # --- Method A: Per-seed WR difference ---
    bl_wrs = []
    d_wrs = []
    for seed in range(N_SEEDS):
        bl_preds = per_seed_late["Baseline (21)"]["preds"][seed]
        d_preds = per_seed_late["D: +Gap (24)"]["preds"][seed]

        bl_picks = bl_preds == 1
        d_picks = d_preds == 1

        bl_wr = y_late[bl_picks].mean() if bl_picks.sum() > 0 else np.nan
        d_wr = y_late[d_picks].mean() if d_picks.sum() > 0 else np.nan
        bl_wrs.append(bl_wr)
        d_wrs.append(d_wr)

    diffs_seeds = [d - b for d, b in zip(d_wrs, bl_wrs) if not np.isnan(d) and not np.isnan(b)]
    mean_diff = np.mean(diffs_seeds) * 100
    se_diff = np.std(diffs_seeds, ddof=1) / np.sqrt(len(diffs_seeds)) * 100

    print(f"\n  Method A: Per-seed WR difference (10 seeds)")
    print(f"    Mean WR (Baseline): {np.nanmean(bl_wrs) * 100:.2f}%")
    print(f"    Mean WR (D):        {np.nanmean(d_wrs) * 100:.2f}%")
    print(f"    Mean difference:    {mean_diff:+.2f}pp")
    print(f"    SE:                 {se_diff:.2f}pp")
    print(f"    95% CI:             [{mean_diff - 1.96 * se_diff:+.2f}pp, {mean_diff + 1.96 * se_diff:+.2f}pp]")

    # t-test on differences
    if len(diffs_seeds) > 1:
        t_stat, p_val = stats.ttest_1samp(diffs_seeds, 0)
        print(f"    t-test vs 0: t={t_stat:.3f}, p={p_val:.4f}")

    # --- Method B: Bootstrap on averaged probas ---
    # Average probas across seeds for stability
    bl_proba_avg = np.mean(per_seed_late["Baseline (21)"]["proba"], axis=0)
    d_proba_avg = np.mean(per_seed_late["D: +Gap (24)"]["proba"], axis=0)

    bl_pred_avg = (bl_proba_avg >= 0.5).astype(int)
    d_pred_avg = (d_proba_avg >= 0.5).astype(int)

    # Point estimates
    bl_wr_avg = y_late[bl_pred_avg == 1].mean() if bl_pred_avg.sum() > 0 else np.nan
    d_wr_avg = y_late[d_pred_avg == 1].mean() if d_pred_avg.sum() > 0 else np.nan
    n_bl = int(bl_pred_avg.sum())
    n_d = int(d_pred_avg.sum())

    print(f"\n  Method B: Bootstrap on seed-averaged probas")
    print(f"    Baseline: WR={bl_wr_avg * 100:.2f}% (N={n_bl} picks out of {len(y_late)} games)")
    print(f"    Model D:  WR={d_wr_avg * 100:.2f}% (N={n_d} picks out of {len(y_late)} games)")

    # Wilson CI on each WR
    bl_wilson = wilson_ci(n_bl, int(y_late[bl_pred_avg == 1].sum()))
    d_wilson = wilson_ci(n_d, int(y_late[d_pred_avg == 1].sum()))
    print(f"    Baseline Wilson 95% CI: [{bl_wilson[0]:.2f}%, {bl_wilson[1]:.2f}%]")
    print(f"    Model D  Wilson 95% CI: [{d_wilson[0]:.2f}%, {d_wilson[1]:.2f}%]")

    # Bootstrap the WR difference
    np.random.seed(42)
    n_boot = 10000
    boot_diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y_late), len(y_late), replace=True)
        y_b = y_late[idx]
        bl_p = bl_pred_avg[idx]
        d_p = d_pred_avg[idx]

        bl_picks = bl_p == 1
        d_picks = d_p == 1

        if bl_picks.sum() > 0 and d_picks.sum() > 0:
            wr_bl = y_b[bl_picks].mean()
            wr_d = y_b[d_picks].mean()
            boot_diffs.append(wr_d - wr_bl)

    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"\n    Bootstrap WR difference (D - Baseline): "
          f"{np.mean(boot_diffs) * 100:+.2f}pp "
          f"[{ci_lo * 100:+.2f}pp, {ci_hi * 100:+.2f}pp]")
    print(f"    Bootstrap samples: {len(boot_diffs)}/{n_boot}")
    print(f"    P(D > Baseline): {(boot_diffs > 0).mean() * 100:.1f}%")

    # Is lower bound clearly above 0?
    if ci_lo > 0:
        print(f"\n    CONCLUSION: Lower bound {ci_lo * 100:+.2f}pp > 0 => "
              f"WR improvement is STATISTICALLY SIGNIFICANT at 95%")
    else:
        print(f"\n    CONCLUSION: Lower bound {ci_lo * 100:+.2f}pp <= 0 => "
              f"CI includes zero, improvement NOT statistically significant at 95%")
        # Check 90% CI
        ci_lo_90, ci_hi_90 = np.percentile(boot_diffs, [5, 95])
        if ci_lo_90 > 0:
            print(f"    90% CI: [{ci_lo_90 * 100:+.2f}pp, {ci_hi_90 * 100:+.2f}pp] => "
                  f"significant at 90%")
        else:
            print(f"    90% CI: [{ci_lo_90 * 100:+.2f}pp, {ci_hi_90 * 100:+.2f}pp] => "
                  f"NOT significant at 90% either")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Game-Level Dedup Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_dedup(df):
    """Verify no rung duplication in the test sets."""
    print("\n" + "=" * 110)
    print("  TEST 5: Game-Level Dedup Verification")
    print("=" * 110)

    if "game_id" not in df.columns:
        print("  WARNING: no game_id column found, cannot verify dedup")
        return

    train = df[df["game_date"] < TRAIN_CUTOFF]
    early = df[(df["game_date"] >= EARLY_START) & (df["game_date"] <= EARLY_END)]
    late = df[(df["game_date"] >= LATE_START) & (df["game_date"] <= LATE_END)]
    all_2026 = df[df["game_date"].dt.year == 2026]

    for name, subset in [("Train", train), ("Early test", early),
                         ("Late test", late), ("All 2026", all_2026)]:
        n_rows = len(subset)
        n_unique = subset["game_id"].nunique()
        dupes = n_rows - n_unique
        status = "CLEAN" if dupes == 0 else f"WARNING: {dupes} duplicates!"
        print(f"    {name:>12s}: {n_rows} rows, {n_unique} unique game_ids => {status}")

    # Check train/test overlap
    train_ids = set(train["game_id"])
    early_ids = set(early["game_id"])
    late_ids = set(late["game_id"])

    te_overlap = train_ids & early_ids
    tl_overlap = train_ids & late_ids
    el_overlap = early_ids & late_ids

    print(f"\n    Train-Early overlap: {len(te_overlap)} games")
    print(f"    Train-Late overlap:  {len(tl_overlap)} games")
    print(f"    Early-Late overlap:  {len(el_overlap)} games")

    if len(te_overlap) + len(tl_overlap) + len(el_overlap) == 0:
        print("    ALL CLEAN: No overlap between splits")
    else:
        print("    WARNING: Overlapping game_ids detected!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_data()

    # Test 1: D vs D+E
    results_1, per_seed_late, y_late, late_df = test_d_vs_de(df)

    # Test 2: TimeSeriesSplit CV
    fold_results = test_timeseries_cv(df)

    # Test 3: Random 50/50 splits
    split_results = test_random_splits(df)

    # Test 4: Confidence intervals
    test_confidence_intervals(per_seed_late, y_late, late_df)

    # Test 5: Dedup verification
    verify_dedup(df)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  FINAL VERDICT")
    print("=" * 110)

    # D vs DE
    d_late_brier = np.nanmean(results_1["D: +Gap (24)"]["late_brier"])
    de_late_brier = np.nanmean(results_1["DE: +Gap+Hand (26)"]["late_brier"])
    bl_late_brier = np.nanmean(results_1["Baseline (21)"]["late_brier"])

    if d_late_brier <= de_late_brier:
        winner = "D: +Gap (24)"
        print(f"  D vs DE: Model D WINS (Brier {d_late_brier:.6f} vs DE {de_late_brier:.6f})")
        print(f"           SP handedness adds noise, not signal.")
    else:
        winner = "DE: +Gap+Hand (26)"
        print(f"  D vs DE: Model DE WINS (Brier {de_late_brier:.6f} vs D {d_late_brier:.6f})")

    # TSCV consistency
    auc_deltas = [r["D: +Gap (24)_auc"] - r["Baseline (21)_auc"] for r in fold_results]
    brier_deltas = [r["D: +Gap (24)_brier"] - r["Baseline (21)_brier"] for r in fold_results]
    n_auc_pos = sum(1 for d in auc_deltas if d > 0)
    n_brier_neg = sum(1 for d in brier_deltas if d < 0)
    print(f"\n  TimeSeriesSplit: AUC+ in {n_auc_pos}/5, Brier- in {n_brier_neg}/5 folds")

    # Random split consistency
    both_brier_count = 0
    for row in split_results:
        d_a_b = row["D: +Gap (24)_a_brier"] - row["Baseline (21)_a_brier"]
        d_b_b = row["D: +Gap (24)_b_brier"] - row["Baseline (21)_b_brier"]
        if d_a_b < 0 and d_b_b < 0:
            both_brier_count += 1
    print(f"  Random splits: Brier- in BOTH halves for {both_brier_count}/5 seeds")

    # Overall assessment
    total_evidence = n_brier_neg + both_brier_count
    print(f"\n  Total positive evidence: {total_evidence}/10 tests (TSCV + random)")
    if total_evidence >= 7:
        print("  ASSESSMENT: STRONG — platoon gap signal is robust across splits")
    elif total_evidence >= 5:
        print("  ASSESSMENT: MODERATE — platoon gap helps in majority of tests")
    elif total_evidence >= 3:
        print("  ASSESSMENT: WEAK — platoon gap inconsistent, may be overfit to specific window")
    else:
        print("  ASSESSMENT: DEAD — platoon gap does not generalize")


if __name__ == "__main__":
    main()
