"""
component_mismatch_test.py — Test component-mismatch INTERACTION features
for ATS spread-cover prediction.

Hypothesis: specific-strength-vs-specific-weakness matchups (e.g., high-walk
team facing wild pitcher) predict outcomes better than aggregate differentials
(ops_diff, sp_era_diff) alone.

Protocol:
  - Train: pre-2026-05-26
  - Early test: 2026-05-26 to 2026-06-15
  - Late test: 2026-07-01 to 2026-08-05
  - Game-level dedup (one row per game_id per spread value)
  - GBC: n_est=500, d=4, lr=0.03, sub=0.8, leaf=5
  - Isotonic calibration (cv=3)
  - 3 random seeds, report mean
  - Spread-aware augmentation: [-1.5, 1.5, 2.5, 3.5, 4.5]

Models tested:
  A. Baseline (V1_PRUNED 21 features + spread cols)
  B. + Discipline mismatches (disc_mismatch_diff, k_mismatch_diff)
  C. + Batted-ball x park (power_park_diff, hard_hit_park)
  D. + Component x component (ops_vs_era_diff, slug_vs_whip_diff)
  E. + Spread-conditional (ops_diff x spread, era_diff x spread)
  F. + ALL above interactions (kitchen sink)
  G. + Best subset (whichever individual groups helped)

Plus: INVERSION TEST — when aggregate and component signals disagree,
      which one gets it right?
"""
from __future__ import annotations

import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from features import (
    build_feature_matrix,
    MODEL_FEATURE_COLS,
    MODEL_FEATURE_COLS_V1_PRUNED,
    TARGET_ATS,
)

# ── Constants ──────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 777]
_AUGMENT_SPREADS = [-1.5, 1.5, 2.5, 3.5, 4.5]

GBC_PARAMS_BASE = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
)

TRAIN_CUTOFF = "2026-05-26"
EARLY_START = "2026-05-26"
EARLY_END = "2026-06-15"
LATE_START = "2026-07-01"
LATE_END = "2026-08-05"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load feature matrix, inspect columns, build augmented dataset."""
    print("=" * 110)
    print("  COMPONENT MISMATCH INTERACTION TEST")
    print("=" * 110)

    print("\n  Loading feature matrix...")
    matrix = build_feature_matrix("MLB")
    matrix = matrix[matrix["home_score"].notna() & matrix["away_score"].notna()].copy()
    matrix = matrix[matrix["home_score"] < 50].copy()
    matrix["game_date"] = pd.to_datetime(matrix["game_date"], errors="coerce")
    matrix = matrix.sort_values("game_date").reset_index(drop=True)
    print(f"  Total games: {len(matrix)}")

    # ── Check available columns ────────────────────────────────────────────
    interaction_inputs = [
        "h_sp_k9", "a_sp_k9", "h_sp_bb9", "a_sp_bb9",
        "h_k_pct", "a_k_pct", "h_bb_pct", "a_bb_pct",
        "h_barrel_pct", "a_barrel_pct", "h_hard_hit_pct", "a_hard_hit_pct",
        "hard_hit_diff",
        "h_park_factor",
        "h_ops", "a_ops", "ops_diff",
        "h_slg", "a_slg",
        "h_sp_era", "a_sp_era", "sp_era_diff",
        "h_sp_whip", "a_sp_whip",
        "close_spread",
    ]
    print("\n  COLUMN AVAILABILITY CHECK:")
    for col in interaction_inputs:
        if col in matrix.columns:
            fill = matrix[col].notna().mean()
            print(f"    {col:25s}  present  fill={fill:.1%}")
        else:
            print(f"    {col:25s}  MISSING")

    return matrix


def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Build all component-mismatch interaction features."""
    df = df.copy()

    # ── Plate Discipline Mismatches ────────────────────────────────────────
    # disc_mismatch_h: away team walk rate x home pitcher walk rate
    #   (high-walk team vs wild pitcher = compounding walk tendency)
    df["disc_mismatch_h"] = df["a_bb_pct"] * df["h_sp_bb9"]
    df["disc_mismatch_a"] = df["h_bb_pct"] * df["a_sp_bb9"]
    df["disc_mismatch_diff"] = df["disc_mismatch_h"] - df["disc_mismatch_a"]

    # k_mismatch_h: away team K rate x home pitcher K/9
    #   (high-K team vs high-K pitcher = compounding K tendency)
    df["k_mismatch_h"] = df["a_k_pct"] * df["h_sp_k9"]
    df["k_mismatch_a"] = df["h_k_pct"] * df["a_sp_k9"]
    df["k_mismatch_diff"] = df["k_mismatch_h"] - df["k_mismatch_a"]

    # ── Batted-Ball x Park ─────────────────────────────────────────────────
    # power_park: team barrel rate x park factor
    df["power_park_h"] = df["h_barrel_pct"] * df["h_park_factor"]
    df["power_park_a"] = df["a_barrel_pct"] * df["h_park_factor"]
    df["power_park_diff"] = df["power_park_h"] - df["power_park_a"]
    df["hard_hit_park"] = df["hard_hit_diff"] * df["h_park_factor"]

    # ── Component vs Component (offense x opposing pitcher) ────────────────
    # ops_vs_era: home offense vs away pitcher quality
    #   Invert ERA so higher = better pitcher, then interaction captures
    #   "good offense facing bad pitcher" vs "bad offense facing good pitcher"
    df["ops_vs_era_h"] = df["h_ops"] * (5.0 - df["a_sp_era"])
    df["ops_vs_era_a"] = df["a_ops"] * (5.0 - df["h_sp_era"])
    df["ops_vs_era_diff"] = df["ops_vs_era_h"] - df["ops_vs_era_a"]

    # slug_vs_whip: team power vs opposing pitcher control
    df["slug_vs_whip_h"] = df["h_slg"] * df["a_sp_whip"]
    df["slug_vs_whip_a"] = df["a_slg"] * df["h_sp_whip"]
    df["slug_vs_whip_diff"] = df["slug_vs_whip_h"] - df["slug_vs_whip_a"]

    # ── Spread-Conditional ─────────────────────────────────────────────────
    # After augmentation, _spread_value is the spread; before, use close_spread
    if "_spread_value" in df.columns:
        spread_col = "_spread_value"
    elif "close_spread" in df.columns:
        spread_col = "close_spread"
    else:
        spread_col = None

    if spread_col is not None:
        df["ops_diff_x_spread"] = df["ops_diff"] * pd.to_numeric(df[spread_col], errors="coerce")
        df["era_diff_x_spread"] = df["sp_era_diff"] * pd.to_numeric(df[spread_col], errors="coerce")
    else:
        df["ops_diff_x_spread"] = np.nan
        df["era_diff_x_spread"] = np.nan

    return df


def augment_spreads(matrix: pd.DataFrame) -> pd.DataFrame:
    """Spread-aware augmentation: one row per game per spread value."""
    margin = (
        pd.to_numeric(matrix["home_score"], errors="coerce")
        - pd.to_numeric(matrix["away_score"], errors="coerce")
    )

    aug_rows = []
    for spread_val in _AUGMENT_SPREADS:
        chunk = matrix.copy()
        chunk["_spread_value"] = float(spread_val)
        chunk["_spread_abs"] = abs(spread_val)
        chunk["_is_fav_spread"] = 1.0 if spread_val < 0 else 0.0
        ats_val = margin + spread_val
        chunk[TARGET_ATS] = np.where(
            ats_val > 0, 1.0,
            np.where(ats_val < 0, 0.0, np.nan),
        )
        aug_rows.append(chunk)

    augmented = pd.concat(aug_rows, ignore_index=True)
    augmented = augmented[augmented[TARGET_ATS].notna()].copy()
    augmented = augmented.sort_values("game_date").reset_index(drop=True)
    return augmented


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_eval_model(
    augmented: pd.DataFrame,
    feat_cols: list[str],
    label: str,
    seed: int,
) -> dict:
    """
    Train on pre-cutoff, evaluate on Early and Late windows.
    Returns dict of metrics.
    """
    # ── Split ──────────────────────────────────────────────────────────────
    train_mask = augmented["game_date"] < TRAIN_CUTOFF
    early_mask = (
        (augmented["game_date"] >= EARLY_START)
        & (augmented["game_date"] <= EARLY_END)
    )
    late_mask = (
        (augmented["game_date"] >= LATE_START)
        & (augmented["game_date"] <= LATE_END)
    )

    # Ensure all feature cols exist
    available_cols = [c for c in feat_cols if c in augmented.columns]
    if len(available_cols) < len(feat_cols):
        missing = set(feat_cols) - set(available_cols)
        print(f"    WARNING: Missing columns for {label}: {missing}")
    feat_cols = available_cols

    X_tr = augmented.loc[train_mask, feat_cols].values.astype(float)
    y_tr = augmented.loc[train_mask, TARGET_ATS].values.astype(float)
    X_early = augmented.loc[early_mask, feat_cols].values.astype(float)
    y_early = augmented.loc[early_mask, TARGET_ATS].values.astype(float)
    X_late = augmented.loc[late_mask, feat_cols].values.astype(float)
    y_late = augmented.loc[late_mask, TARGET_ATS].values.astype(float)

    if len(X_tr) == 0 or len(X_early) == 0 or len(X_late) == 0:
        return {"error": "Insufficient data"}

    # ── Impute + Scale ─────────────────────────────────────────────────────
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_early_imp = imp.transform(X_early)
    X_late_imp = imp.transform(X_late)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_early_s = sc.transform(X_early_imp)
    X_late_s = sc.transform(X_late_imp)

    # ── Train ──────────────────────────────────────────────────────────────
    params = {**GBC_PARAMS_BASE, "random_state": seed}
    base_clf = GradientBoostingClassifier(**params)
    clf = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    # ── Evaluate ───────────────────────────────────────────────────────────
    results = {
        "label": label,
        "seed": seed,
        "n_features": len(feat_cols),
        "n_train": len(X_tr),
    }

    for period_name, X_s, y_true in [
        ("early", X_early_s, y_early),
        ("late", X_late_s, y_late),
    ]:
        y_prob = clf.predict_proba(X_s)[:, 1]
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan
        brier = brier_score_loss(y_true, y_prob)
        acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

        results[f"{period_name}_n"] = len(y_true)
        results[f"{period_name}_auc"] = auc
        results[f"{period_name}_brier"] = brier
        results[f"{period_name}_acc"] = acc

    # ── Feature importances (average across calibrated classifiers) ─────
    all_imp = [
        cal.estimator.feature_importances_
        for cal in clf.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    if all_imp:
        mean_imp = np.mean(all_imp, axis=0)
        results["importances"] = dict(zip(feat_cols, mean_imp))

    # Store classifier for inversion test
    results["clf"] = clf
    results["imp"] = imp
    results["sc"] = sc
    results["feat_cols"] = feat_cols

    return results


def game_level_dedup(augmented: pd.DataFrame) -> pd.DataFrame:
    """
    Dedup to one row per (game_id, _spread_value).
    If game_id is not available, use (game_date, home_team, _spread_value).
    """
    if "game_id" in augmented.columns:
        key_cols = ["game_id", "_spread_value"]
    else:
        key_cols = ["game_date", "home_team", "_spread_value"]

    before = len(augmented)
    augmented = augmented.drop_duplicates(subset=key_cols, keep="first")
    after = len(augmented)
    if before != after:
        print(f"  Game-level dedup: {before} -> {after} rows")
    return augmented


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INVERSION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_inversion_test(
    augmented: pd.DataFrame,
    baseline_feat_cols: list[str],
    seed: int = 42,
):
    """
    Test games where aggregate model and component mismatch disagree.

    1. Train baseline model (aggregate features only)
    2. Predict test games
    3. Compute component mismatch scores
    4. Find "inversion" games: baseline favors side X but component favors side Y
    5. Report which signal gets it right
    """
    print("\n" + "=" * 110)
    print("  INVERSION TEST: Do component mismatches override aggregate when they disagree?")
    print("=" * 110)

    # ── Split and train baseline ───────────────────────────────────────────
    test_mask = augmented["game_date"] >= EARLY_START

    available_cols = [c for c in baseline_feat_cols if c in augmented.columns]

    X_tr = augmented.loc[~test_mask, available_cols].values.astype(float)
    y_tr = augmented.loc[~test_mask, TARGET_ATS].values.astype(float)
    X_te = augmented.loc[test_mask, available_cols].values.astype(float)
    y_te = augmented.loc[test_mask, TARGET_ATS].values.astype(float)

    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_te_s = sc.transform(X_te_imp)

    params = {**GBC_PARAMS_BASE, "random_state": seed}
    base_clf = GradientBoostingClassifier(**params)
    clf = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    y_prob = clf.predict_proba(X_te_s)[:, 1]

    # ── Get test data with mismatch features ───────────────────────────────
    test_df = augmented.loc[test_mask].copy()
    test_df["y_prob_agg"] = y_prob
    test_df["y_true"] = y_te

    # ── Mismatch signals: use differential features ────────────────────────
    # The key mismatch features (differentials, positive = home team advantage)
    mismatch_cols = [
        "disc_mismatch_diff",
        "k_mismatch_diff",
        "ops_vs_era_diff",
        "slug_vs_whip_diff",
        "power_park_diff",
    ]

    avail_mismatch = [c for c in mismatch_cols if c in test_df.columns and test_df[c].notna().sum() > 0]
    if not avail_mismatch:
        print("  No mismatch columns available for inversion test!")
        return

    # Create composite mismatch signal (average of available z-scored mismatches)
    mismatch_data = test_df[avail_mismatch].copy()
    # Z-score each mismatch relative to train distribution
    train_df = augmented.loc[~test_mask]
    for col in avail_mismatch:
        mu = train_df[col].mean()
        sd = train_df[col].std()
        if sd > 0:
            mismatch_data[col] = (mismatch_data[col] - mu) / sd
        else:
            mismatch_data[col] = 0.0

    test_df["mismatch_composite"] = mismatch_data[avail_mismatch].mean(axis=1)

    print(f"\n  Test games: {len(test_df)}")
    print(f"  Mismatch columns used: {avail_mismatch}")
    print(f"  Mismatch composite: mean={test_df['mismatch_composite'].mean():.4f}, "
          f"std={test_df['mismatch_composite'].std():.4f}")

    # ── Define inversions ──────────────────────────────────────────────────
    # Aggregate favors home (P > 0.55) BUT mismatch favors away (composite < 0)
    # OR aggregate favors away (P < 0.45) BUT mismatch favors home (composite > 0)
    has_both = test_df["mismatch_composite"].notna() & test_df["y_prob_agg"].notna()
    test_valid = test_df[has_both].copy()

    inv_home_agg = (
        (test_valid["y_prob_agg"] > 0.55)
        & (test_valid["mismatch_composite"] < -0.3)
    )
    inv_away_agg = (
        (test_valid["y_prob_agg"] < 0.45)
        & (test_valid["mismatch_composite"] > 0.3)
    )
    inversions = test_valid[inv_home_agg | inv_away_agg].copy()

    print(f"\n  Inversion games (aggregate and component disagree):")
    print(f"    Total: {len(inversions)}")

    if len(inversions) == 0:
        print("    No inversions found at these thresholds.")
        # Try relaxed thresholds
        for p_thresh, m_thresh in [(0.53, 0.2), (0.52, 0.1), (0.51, 0.0)]:
            inv_h = (test_valid["y_prob_agg"] > p_thresh) & (test_valid["mismatch_composite"] < -m_thresh)
            inv_a = (test_valid["y_prob_agg"] < 1 - p_thresh) & (test_valid["mismatch_composite"] > m_thresh)
            n = (inv_h | inv_a).sum()
            print(f"    At p>{p_thresh}/mismatch>{m_thresh}: {n} inversions")
        return

    # ── In inversion games: who gets it right? ─────────────────────────────
    # "Aggregate says cover": y_prob_agg > 0.5 → predict y=1
    # "Component says don't cover": mismatch suggests opposite side
    inv_agg_says_home = inversions["y_prob_agg"] > 0.5
    inv_actual = inversions["y_true"]

    # Aggregate correctness: did the aggregate prediction match?
    agg_correct = (inv_agg_says_home == inv_actual.astype(bool)).mean()

    # Component correctness: component signal direction
    # mismatch_composite > 0 means home advantage, < 0 means away advantage
    comp_says_home = inversions["mismatch_composite"] > 0
    comp_correct = (comp_says_home == inv_actual.astype(bool)).mean()

    print(f"    Aggregate WR on inversions:  {agg_correct:.1%}")
    print(f"    Component WR on inversions:  {comp_correct:.1%}")
    print(f"    (Higher = that signal is right when they disagree)")

    # ── Breakdown by inversion type ────────────────────────────────────────
    inv_home = inversions[inv_home_agg.loc[inversions.index]]
    inv_away = inversions[inv_away_agg.loc[inversions.index]]

    if len(inv_home) > 0:
        print(f"\n    Agg says HOME covers, component disagrees: N={len(inv_home)}")
        print(f"      Actual cover rate: {inv_home['y_true'].mean():.1%}")
        print(f"      (If component is right, cover rate should be LOW)")

    if len(inv_away) > 0:
        print(f"\n    Agg says AWAY covers, component disagrees: N={len(inv_away)}")
        print(f"      Actual cover rate: {inv_away['y_true'].mean():.1%}")
        print(f"      (If component is right, cover rate should be HIGH)")

    # ── Non-inversion (agreement) games as control ─────────────────────────
    agreement = test_valid[~(inv_home_agg | inv_away_agg)]
    if len(agreement) > 0:
        agg_acc_agree = accuracy_score(
            agreement["y_true"],
            (agreement["y_prob_agg"] >= 0.5).astype(int),
        )
        print(f"\n    Agreement games (both signals same side): N={len(agreement)}")
        print(f"      Aggregate accuracy: {agg_acc_agree:.1%}")

    # ── Sweep inversion thresholds ─────────────────────────────────────────
    print(f"\n  Inversion threshold sweep:")
    print(f"  {'P thresh':>10}  {'M thresh':>10}  {'N inv':>6}  {'Agg WR':>8}  {'Comp WR':>8}  {'Delta':>8}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")

    for p_thresh in [0.51, 0.52, 0.53, 0.54, 0.55]:
        for m_thresh in [0.0, 0.1, 0.2, 0.3, 0.5]:
            inv_h = (test_valid["y_prob_agg"] > p_thresh) & (test_valid["mismatch_composite"] < -m_thresh)
            inv_a = (test_valid["y_prob_agg"] < 1 - p_thresh) & (test_valid["mismatch_composite"] > m_thresh)
            inv_mask = inv_h | inv_a
            n = inv_mask.sum()
            if n < 15:
                continue

            inv_sub = test_valid[inv_mask]
            a_says_h = inv_sub["y_prob_agg"] > 0.5
            a_correct = (a_says_h == inv_sub["y_true"].astype(bool)).mean()
            c_says_h = inv_sub["mismatch_composite"] > 0
            c_correct = (c_says_h == inv_sub["y_true"].astype(bool)).mean()
            delta = c_correct - a_correct

            marker = " ***" if delta > 0.05 else ""
            print(f"  {p_thresh:>10.2f}  {m_thresh:>10.1f}  {n:>6}  "
                  f"{a_correct:>7.1%}  {c_correct:>7.1%}  {delta:>+7.1%}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Step 1: Load data ──────────────────────────────────────────────────
    matrix = load_data()

    # ── Step 2: Build interaction features BEFORE augmentation ─────────────
    # (interactions from raw game data, not spread-augmented duplicates)
    matrix = build_interactions(matrix)

    # Print interaction feature fill rates
    interaction_cols = [
        "disc_mismatch_diff", "k_mismatch_diff",
        "power_park_diff", "hard_hit_park",
        "ops_vs_era_diff", "slug_vs_whip_diff",
    ]
    print("\n  INTERACTION FEATURE FILL RATES:")
    for col in interaction_cols:
        if col in matrix.columns:
            fill = matrix[col].notna().mean()
            print(f"    {col:25s}  fill={fill:.1%}")
        else:
            print(f"    {col:25s}  MISSING")

    # ── Step 3: Augment spreads ────────────────────────────────────────────
    print("\n  Augmenting spreads...")
    augmented = augment_spreads(matrix)

    # Now build the spread-conditional interactions (need _spread_value)
    augmented["ops_diff_x_spread"] = (
        augmented["ops_diff"]
        * pd.to_numeric(augmented["_spread_value"], errors="coerce")
    )
    augmented["era_diff_x_spread"] = (
        augmented["sp_era_diff"]
        * pd.to_numeric(augmented["_spread_value"], errors="coerce")
    )

    # ── Game-level dedup ───────────────────────────────────────────────────
    augmented = game_level_dedup(augmented)

    # ── Date stats ─────────────────────────────────────────────────────────
    print(f"\n  Augmented dataset: {len(augmented)} rows")
    print(f"  Date range: {augmented['game_date'].min().date()} to {augmented['game_date'].max().date()}")

    n_train = (augmented["game_date"] < TRAIN_CUTOFF).sum()
    n_early = ((augmented["game_date"] >= EARLY_START) & (augmented["game_date"] <= EARLY_END)).sum()
    n_late = ((augmented["game_date"] >= LATE_START) & (augmented["game_date"] <= LATE_END)).sum()
    print(f"  Train: {n_train}  |  Early test: {n_early}  |  Late test: {n_late}")

    for sv in _AUGMENT_SPREADS:
        n = (augmented["_spread_value"] == sv).sum()
        cr = augmented.loc[augmented["_spread_value"] == sv, TARGET_ATS].mean()
        print(f"    spread={sv:+.1f}: {n} rows, cover rate = {cr:.1%}")

    # ── Step 4: Define feature sets ────────────────────────────────────────
    baseline_cols = [c for c in MODEL_FEATURE_COLS_V1_PRUNED if c in augmented.columns]
    spread_cols = ["_spread_value", "_spread_abs", "_is_fav_spread"]

    # Full 87-feature set columns needed for interactions
    extra_cols_for_interactions = [
        "h_k_pct", "a_k_pct",
        "h_barrel_pct", "a_barrel_pct",
        "h_hard_hit_pct", "a_hard_hit_pct",
        "h_park_factor",
    ]

    models = {
        "A_baseline": baseline_cols + spread_cols,
        "B_discipline": baseline_cols + spread_cols + [
            "disc_mismatch_diff", "k_mismatch_diff",
        ],
        "C_batted_park": baseline_cols + spread_cols + [
            "power_park_diff", "hard_hit_park",
        ],
        "D_component": baseline_cols + spread_cols + [
            "ops_vs_era_diff", "slug_vs_whip_diff",
        ],
        "E_spread_cond": baseline_cols + spread_cols + [
            "ops_diff_x_spread", "era_diff_x_spread",
        ],
        "F_kitchen_sink": baseline_cols + spread_cols + [
            "disc_mismatch_diff", "k_mismatch_diff",
            "power_park_diff", "hard_hit_park",
            "ops_vs_era_diff", "slug_vs_whip_diff",
            "ops_diff_x_spread", "era_diff_x_spread",
        ],
    }

    # ── Step 5: Run multi-seed evaluation ──────────────────────────────────
    print("\n" + "=" * 110)
    print("  MULTI-SEED MODEL COMPARISON (3 seeds)")
    print("=" * 110)

    all_results = []
    for model_name, feat_cols in models.items():
        # Deduplicate the feature list
        feat_cols = list(dict.fromkeys(feat_cols))
        print(f"\n  {model_name}: {len(feat_cols)} features")
        seed_results = []
        for seed in SEEDS:
            res = train_eval_model(augmented, feat_cols, model_name, seed)
            if "error" in res:
                print(f"    Seed {seed}: ERROR — {res['error']}")
                continue
            seed_results.append(res)
            print(f"    Seed {seed}: Early AUC={res['early_auc']:.4f} Brier={res['early_brier']:.4f} | "
                  f"Late AUC={res['late_auc']:.4f} Brier={res['late_brier']:.4f}")

        if seed_results:
            mean_res = {
                "model": model_name,
                "n_features": seed_results[0]["n_features"],
                "n_train": seed_results[0]["n_train"],
                "early_n": seed_results[0]["early_n"],
                "late_n": seed_results[0]["late_n"],
            }
            for metric in ["early_auc", "early_brier", "early_acc", "late_auc", "late_brier", "late_acc"]:
                vals = [r[metric] for r in seed_results if not np.isnan(r[metric])]
                mean_res[metric] = np.mean(vals) if vals else np.nan
                mean_res[f"{metric}_std"] = np.std(vals) if vals else np.nan
            all_results.append(mean_res)

            # Report feature importances for last seed
            if "importances" in seed_results[-1]:
                imps = seed_results[-1]["importances"]
                sorted_imps = sorted(imps.items(), key=lambda x: -x[1])
                # Show interaction features if they're in top 15
                interaction_feats = {
                    "disc_mismatch_diff", "k_mismatch_diff",
                    "power_park_diff", "hard_hit_park",
                    "ops_vs_era_diff", "slug_vs_whip_diff",
                    "ops_diff_x_spread", "era_diff_x_spread",
                }
                for fname, fval in sorted_imps[:15]:
                    marker = " <<< INTERACTION" if fname in interaction_feats else ""
                    print(f"      imp: {fname:30s} = {fval:.4f}{marker}")

    # ── Summary table ──────────────────────────────────────────────────────
    if all_results:
        print("\n\n" + "=" * 110)
        print("  SUMMARY TABLE")
        print("=" * 110)

        # Get baseline values for delta computation
        baseline = next((r for r in all_results if r["model"] == "A_baseline"), None)

        header = (
            f"  {'Model':<20} {'Feats':>5}  "
            f"{'Early AUC':>10} {'dAUC':>7}  {'Early Brier':>12} {'dBrier':>8}  "
            f"{'Late AUC':>10} {'dAUC':>7}  {'Late Brier':>12} {'dBrier':>8}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for r in all_results:
            d_early_auc = (r["early_auc"] - baseline["early_auc"]) if baseline else 0
            d_early_brier = (r["early_brier"] - baseline["early_brier"]) if baseline else 0
            d_late_auc = (r["late_auc"] - baseline["late_auc"]) if baseline else 0
            d_late_brier = (r["late_brier"] - baseline["late_brier"]) if baseline else 0

            print(
                f"  {r['model']:<20} {r['n_features']:>5}  "
                f"{r['early_auc']:>10.4f} {d_early_auc:>+7.4f}  "
                f"{r['early_brier']:>12.4f} {d_early_brier:>+8.4f}  "
                f"{r['late_auc']:>10.4f} {d_late_auc:>+7.4f}  "
                f"{r['late_brier']:>12.4f} {d_late_brier:>+8.4f}"
            )

        # ── Early vs Late stability ────────────────────────────────────────
        print("\n  EARLY vs LATE STABILITY:")
        for r in all_results:
            early_auc = r["early_auc"]
            late_auc = r["late_auc"]
            decay = late_auc - early_auc
            print(f"    {r['model']:<20}: Early AUC={early_auc:.4f} -> Late AUC={late_auc:.4f}  "
                  f"Decay={decay:+.4f}")

        # ── Determine best subset (Model G) ────────────────────────────────
        # Check which individual groups (B, C, D, E) improved over baseline
        if baseline:
            helpers = []
            group_feats = {
                "B": ["disc_mismatch_diff", "k_mismatch_diff"],
                "C": ["power_park_diff", "hard_hit_park"],
                "D": ["ops_vs_era_diff", "slug_vs_whip_diff"],
                "E": ["ops_diff_x_spread", "era_diff_x_spread"],
            }

            print("\n  INDIVIDUAL GROUP CONTRIBUTION:")
            for group_letter, group_model in [("B", "B_discipline"), ("C", "C_batted_park"),
                                               ("D", "D_component"), ("E", "E_spread_cond")]:
                r = next((x for x in all_results if x["model"] == group_model), None)
                if r:
                    d_late_auc = r["late_auc"] - baseline["late_auc"]
                    d_late_brier = r["late_brier"] - baseline["late_brier"]
                    verdict = "HELPS" if d_late_auc > 0.002 and d_late_brier < -0.001 else (
                        "HURTS" if d_late_auc < -0.002 or d_late_brier > 0.001 else "NEUTRAL"
                    )
                    print(f"    Group {group_letter} ({group_model}): "
                          f"Late dAUC={d_late_auc:+.4f} dBrier={d_late_brier:+.4f}  -> {verdict}")
                    if verdict == "HELPS":
                        helpers.append((group_letter, group_feats[group_letter]))

            # Build Model G if any helpers
            if helpers:
                g_feats = baseline_cols + spread_cols
                for _, feats in helpers:
                    g_feats.extend(feats)
                g_feats = list(dict.fromkeys(g_feats))  # dedup

                print(f"\n  MODEL G (best subset): groups {[h[0] for h in helpers]}")
                print(f"    Features: {len(g_feats)}")

                g_seed_results = []
                for seed in SEEDS:
                    res = train_eval_model(augmented, g_feats, "G_best_subset", seed)
                    if "error" not in res:
                        g_seed_results.append(res)
                        print(f"    Seed {seed}: Early AUC={res['early_auc']:.4f} | "
                              f"Late AUC={res['late_auc']:.4f}")

                if g_seed_results:
                    g_mean = {
                        "model": "G_best_subset",
                        "n_features": g_seed_results[0]["n_features"],
                        "n_train": g_seed_results[0]["n_train"],
                        "early_n": g_seed_results[0]["early_n"],
                        "late_n": g_seed_results[0]["late_n"],
                    }
                    for metric in ["early_auc", "early_brier", "early_acc",
                                   "late_auc", "late_brier", "late_acc"]:
                        vals = [r[metric] for r in g_seed_results if not np.isnan(r[metric])]
                        g_mean[metric] = np.mean(vals) if vals else np.nan
                    all_results.append(g_mean)

                    print(f"\n  G_best_subset: "
                          f"Early AUC={g_mean['early_auc']:.4f} "
                          f"(d={g_mean['early_auc']-baseline['early_auc']:+.4f}) | "
                          f"Late AUC={g_mean['late_auc']:.4f} "
                          f"(d={g_mean['late_auc']-baseline['late_auc']:+.4f})")
            else:
                print("\n  MODEL G: No individual groups helped. Skipping best-subset.")

    # ── Step 6: Inversion test ─────────────────────────────────────────────
    run_inversion_test(augmented, baseline_cols + spread_cols, seed=42)

    # ── Step 7: Late-season hold ───────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  LATE-SEASON STRUCTURAL HYPOTHESIS")
    print("=" * 110)
    print("""
  Hypothesis: Component mismatches are structural/physical (park factor,
  plate discipline, batted-ball quality) and should NOT decay like
  aggregate form features (10-game rolling, momentum).

  Evidence from results above:
""")
    if all_results and baseline:
        for r in all_results:
            if r["model"] == "A_baseline":
                continue
            early_delta = r["early_auc"] - baseline["early_auc"]
            late_delta = r["late_auc"] - baseline["late_auc"]
            # Does the feature help MORE in late than early? (less decay)
            structural_signal = "STRUCTURAL" if late_delta > early_delta else "DECAYS"
            print(f"    {r['model']:<20}: Early dAUC={early_delta:+.4f}  "
                  f"Late dAUC={late_delta:+.4f}  -> {structural_signal}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
