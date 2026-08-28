"""
pi_strength_test.py — Test whether adding Pi ratings improves ATS prediction.

Protocol:
  - GBC: n_estimators=500, max_depth=4, lr=0.03, subsample=0.8, min_samples_leaf=5
  - CalibratedClassifierCV(isotonic, cv=3)
  - Game-level dedup (one row per game, keeping first rung)
  - 3 random seeds, report mean
  - Train: all data before 2026-05-26
  - Early test: 2026-05-26 to 2026-06-15
  - Late test:  2026-07-01 to 2026-08-05

Models:
  1. Baseline:     V1_PRUNED (21 features)
  2. + Pi diff:    21 + pi_diff (1 feature)
  3. + Pi full:    21 + home_pi, away_pi, pi_diff (3 features)
  4. + Pi + trend:  21 + home_pi, away_pi, pi_diff + home_pi_trend_10g, away_pi_trend_10g (5 features)
  5. + Pi x rung:  21 + pi_diff + pi_diff_x_spread (2 features)
"""
from __future__ import annotations

import os
import sys
import warnings
import sqlite3

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))

from features import (
    build_feature_matrix,
    MODEL_FEATURE_COLS_V1_PRUNED,
    TARGET_ATS,
)

# ── Config ──────────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = "2026-05-26"
EARLY_START = "2026-05-26"
EARLY_END = "2026-06-15"
LATE_START = "2026-07-01"
LATE_END = "2026-08-05"
SEEDS = [42, 123, 777]

GBC_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    min_samples_leaf=5,
)

BASELINE_FEATURES = list(MODEL_FEATURE_COLS_V1_PRUNED)

PI_DIFF_FEATURES = BASELINE_FEATURES + ["pi_diff"]
PI_FULL_FEATURES = BASELINE_FEATURES + ["home_pi", "away_pi", "pi_diff"]
PI_TREND_FEATURES = BASELINE_FEATURES + [
    "home_pi", "away_pi", "pi_diff",
    "home_pi_trend_10g", "away_pi_trend_10g",
]
PI_RUNG_FEATURES = BASELINE_FEATURES + ["pi_diff", "pi_diff_x_spread"]


def _compute_pi_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute the team's Pi change over the last 10 games.
    Uses pi_rating_history to compute the trend.
    """
    conn = sqlite3.connect(os.path.join(_HERE, "data", "historical.db"))
    try:
        pi_hist = pd.read_sql_query(
            "SELECT team, game_date, home_pi, away_pi "
            "FROM pi_rating_history WHERE sport = 'MLB' ORDER BY team, game_date",
            conn,
        )
    finally:
        conn.close()

    if pi_hist.empty:
        df["home_pi_trend_10g"] = np.nan
        df["away_pi_trend_10g"] = np.nan
        return df

    pi_hist["game_date"] = pd.to_datetime(pi_hist["game_date"])
    pi_hist = pi_hist.sort_values(["team", "game_date"])

    # For each team, compute the Pi change over the last 10 snapshots
    # Pi trend = current Pi - Pi from 10 games ago (shift(10) within team)
    grp = pi_hist.groupby("team")
    pi_hist["home_pi_10ago"] = grp["home_pi"].shift(10)
    pi_hist["away_pi_10ago"] = grp["away_pi"].shift(10)
    pi_hist["home_pi_trend"] = pi_hist["home_pi"] - pi_hist["home_pi_10ago"]
    pi_hist["away_pi_trend"] = pi_hist["away_pi"] - pi_hist["away_pi_10ago"]

    # Merge: for each game in df, find the latest pi_hist entry BEFORE game_date
    # Home team trend
    df = df.sort_values("game_date").copy()

    home_lookup = pi_hist[["team", "game_date", "home_pi_trend"]].dropna(
        subset=["home_pi_trend"]
    ).rename(columns={"team": "home_team", "home_pi_trend": "home_pi_trend_10g"})
    home_lookup = home_lookup.sort_values("game_date")

    df = pd.merge_asof(
        df,
        home_lookup,
        on="game_date",
        by="home_team",
        direction="backward",
        allow_exact_matches=False,
    )

    away_lookup = pi_hist[["team", "game_date", "away_pi_trend"]].dropna(
        subset=["away_pi_trend"]
    ).rename(columns={"team": "away_team", "away_pi_trend": "away_pi_trend_10g"})
    away_lookup = away_lookup.sort_values("game_date")

    df = pd.merge_asof(
        df,
        away_lookup,
        on="game_date",
        by="away_team",
        direction="backward",
        allow_exact_matches=False,
    )

    return df


def _compute_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Add pi_diff * close_spread interaction term."""
    if "pi_diff" in df.columns and "close_spread" in df.columns:
        spread = pd.to_numeric(df["close_spread"], errors="coerce")
        df["pi_diff_x_spread"] = df["pi_diff"] * spread
    else:
        df["pi_diff_x_spread"] = np.nan
    return df


def _dedup_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per game (first occurrence by game_id)."""
    if "game_id" in df.columns:
        return df.drop_duplicates(subset=["game_id"], keep="first")
    return df


def _train_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    seed: int,
) -> dict:
    """Train GBC + isotonic calibration, evaluate on test set."""
    # Filter to available columns
    cols = [c for c in feature_cols if c in X_train.columns and c in X_test.columns]
    Xtr = X_train[cols].copy()
    Xte = X_test[cols].copy()

    # Drop rows with NaN in selected features
    train_mask = Xtr.notna().all(axis=1)
    test_mask = Xte.notna().all(axis=1)

    Xtr = Xtr[train_mask]
    ytr = y_train[train_mask]
    Xte = Xte[test_mask]
    yte = y_test[test_mask]

    if len(ytr) < 50 or len(yte) < 10:
        return {"auc": np.nan, "wr": np.nan, "brier": np.nan, "n_train": len(ytr),
                "n_test": len(yte), "n_feat": len(cols)}

    gbc = GradientBoostingClassifier(**{**GBC_PARAMS, "random_state": seed})
    cal = CalibratedClassifierCV(gbc, method="isotonic", cv=3)
    cal.fit(Xtr, ytr)

    proba = cal.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(yte, proba) if len(yte.unique()) > 1 else np.nan
    wr = accuracy_score(yte, preds)
    brier = brier_score_loss(yte, proba)

    return {
        "auc": auc,
        "wr": wr,
        "brier": brier,
        "n_train": len(ytr),
        "n_test": len(yte),
        "n_feat": len(cols),
    }


def run_pi_conditional_analysis(matrix: pd.DataFrame) -> None:
    """Bin games by Pi gap and report WR vs breakeven per bin."""
    test_df = matrix[
        (matrix["game_date"] >= EARLY_START) & (matrix["game_date"] <= LATE_END)
    ].copy()
    test_df = _dedup_game_level(test_df)
    test_df = test_df[test_df[TARGET_ATS].notna() & test_df["pi_diff"].notna()].copy()

    if len(test_df) < 30:
        print("\nPi-conditional analysis: insufficient data")
        return

    # Bin by absolute pi_diff
    abs_pi = test_df["pi_diff"].abs()
    test_df["pi_gap_bin"] = pd.cut(
        abs_pi,
        bins=[0, 0.5, 1.0, 2.0, float("inf")],
        labels=["Small (<0.5)", "Medium (0.5-1.0)", "Large (1.0-2.0)", "XL (>2.0)"],
    )

    # Also compute breakeven from close_ml_home (if available)
    has_ml = "close_ml_home" in test_df.columns

    print("\n" + "=" * 80)
    print("PI-CONDITIONAL ANALYSIS: WR by Pi gap bin (all test dates)")
    print("=" * 80)
    print(f"{'Bin':<20} {'N':>5} {'WR':>8} {'Cover%':>8}")
    print("-" * 50)

    for bin_label in ["Small (<0.5)", "Medium (0.5-1.0)", "Large (1.0-2.0)", "XL (>2.0)"]:
        sub = test_df[test_df["pi_gap_bin"] == bin_label]
        if len(sub) < 5:
            print(f"{bin_label:<20} {len(sub):>5}   (too few)")
            continue
        wr = sub[TARGET_ATS].mean()
        print(f"{bin_label:<20} {len(sub):>5} {wr:>7.1%} {'---':>8}")

    # Also check: does the stronger Pi team cover more often?
    test_df["pi_fav_covers"] = (
        ((test_df["pi_diff"] > 0) & (test_df[TARGET_ATS] == 1)) |
        ((test_df["pi_diff"] < 0) & (test_df[TARGET_ATS] == 0))
    ).astype(int)

    n_valid = test_df["pi_fav_covers"].notna().sum()
    if n_valid > 0:
        fav_wr = test_df["pi_fav_covers"].mean()
        print(f"\nPi-stronger team covers: {fav_wr:.1%} ({n_valid} games)")


def main():
    print("=" * 80)
    print("Pi STRENGTH TEST — ATS Model Enhancement")
    print("=" * 80)

    # ── 1. Build feature matrix ─────────────────────────────────────────────
    print("\n[1/4] Building feature matrix...")
    matrix = build_feature_matrix("MLB")
    print(f"  Matrix shape: {matrix.shape}")

    # Check Pi fill
    pi_fill = matrix["home_pi"].notna().mean()
    print(f"  Pi fill rate: {pi_fill:.1%}")

    # ── 2. Add derived features ─────────────────────────────────────────────
    print("\n[2/4] Computing Pi trends and interaction features...")
    matrix = _compute_pi_trends(matrix)
    matrix = _compute_interaction(matrix)

    trend_fill = matrix["home_pi_trend_10g"].notna().mean()
    interaction_fill = matrix["pi_diff_x_spread"].notna().mean()
    print(f"  Pi trend fill: {trend_fill:.1%}")
    print(f"  Pi x spread interaction fill: {interaction_fill:.1%}")

    # ── 3. Prepare train/test splits ────────────────────────────────────────
    print("\n[3/4] Preparing train/test splits...")
    # Dedup to game level
    matrix = _dedup_game_level(matrix)
    matrix = matrix[matrix[TARGET_ATS].notna()].copy()
    print(f"  After game-level dedup: {len(matrix)} games")

    # Train set: before cutoff
    train_df = matrix[matrix["game_date"] < TRAIN_CUTOFF].copy()
    early_df = matrix[
        (matrix["game_date"] >= EARLY_START) & (matrix["game_date"] <= EARLY_END)
    ].copy()
    late_df = matrix[
        (matrix["game_date"] >= LATE_START) & (matrix["game_date"] <= LATE_END)
    ].copy()

    print(f"  Train: {len(train_df)} games (before {TRAIN_CUTOFF})")
    print(f"  Early test: {len(early_df)} games ({EARLY_START} to {EARLY_END})")
    print(f"  Late test:  {len(late_df)} games ({LATE_START} to {LATE_END})")

    # ── 4. Run experiments ──────────────────────────────────────────────────
    print("\n[4/4] Running model comparisons (3 seeds x 5 models x 2 windows)...")

    models = {
        "Baseline (21)": BASELINE_FEATURES,
        "+ Pi diff (22)": PI_DIFF_FEATURES,
        "+ Pi full (24)": PI_FULL_FEATURES,
        "+ Pi+trend (26)": PI_TREND_FEATURES,
        "+ Pi x rung (23)": PI_RUNG_FEATURES,
    }

    y_train = train_df[TARGET_ATS].astype(float)
    y_early = early_df[TARGET_ATS].astype(float)
    y_late = late_df[TARGET_ATS].astype(float)

    results = []

    for model_name, feat_list in models.items():
        seed_results_early = []
        seed_results_late = []

        for seed in SEEDS:
            res_early = _train_eval(
                train_df, y_train, early_df, y_early, feat_list, seed
            )
            res_late = _train_eval(
                train_df, y_train, late_df, y_late, feat_list, seed
            )
            seed_results_early.append(res_early)
            seed_results_late.append(res_late)

        # Average across seeds
        def _avg(results_list, key):
            vals = [r[key] for r in results_list if not np.isnan(r[key])]
            return np.mean(vals) if vals else np.nan

        results.append({
            "model": model_name,
            "n_feat": seed_results_early[0]["n_feat"],
            "n_train": seed_results_early[0]["n_train"],
            "early_auc": _avg(seed_results_early, "auc"),
            "early_wr": _avg(seed_results_early, "wr"),
            "early_brier": _avg(seed_results_early, "brier"),
            "early_n": seed_results_early[0]["n_test"],
            "late_auc": _avg(seed_results_late, "auc"),
            "late_wr": _avg(seed_results_late, "wr"),
            "late_brier": _avg(seed_results_late, "brier"),
            "late_n": seed_results_late[0]["n_test"],
        })

        print(f"  {model_name}: Early AUC={results[-1]['early_auc']:.4f}, "
              f"Late AUC={results[-1]['late_auc']:.4f}")

    # ── 5. Print results table ──────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RESULTS TABLE (3-seed mean)")
    print("=" * 110)
    baseline_early_auc = results[0]["early_auc"]
    baseline_late_auc = results[0]["late_auc"]
    baseline_early_brier = results[0]["early_brier"]
    baseline_late_brier = results[0]["late_brier"]

    header = (
        f"{'Model':<20} {'N_feat':>6} {'N_trn':>6} "
        f"{'Early AUC':>10} {'Late AUC':>10} "
        f"{'Early WR':>9} {'Late WR':>9} "
        f"{'Early Brier':>12} {'Late Brier':>12} "
        f"{'Late AUC D':>10} {'Late Brier D':>13}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        late_auc_delta = r["late_auc"] - baseline_late_auc
        late_brier_delta = r["late_brier"] - baseline_late_brier
        # AUC delta: positive = better; Brier delta: negative = better
        auc_d_str = f"{late_auc_delta:+.4f}" if not np.isnan(late_auc_delta) else "---"
        brier_d_str = f"{late_brier_delta:+.6f}" if not np.isnan(late_brier_delta) else "---"
        if r["model"].startswith("Baseline"):
            auc_d_str = "---"
            brier_d_str = "---"

        print(
            f"{r['model']:<20} {r['n_feat']:>6} {r['n_train']:>6} "
            f"{r['early_auc']:>10.4f} {r['late_auc']:>10.4f} "
            f"{r['early_wr']:>8.1%} {r['late_wr']:>8.1%} "
            f"{r['early_brier']:>12.6f} {r['late_brier']:>12.6f} "
            f"{auc_d_str:>10} {brier_d_str:>13}"
        )

    print(f"\nEarly N = {results[0]['early_n']}  |  Late N = {results[0]['late_n']}")

    # ── 6. Feature importance check ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE — Pi full model (seed=42)")
    print("=" * 80)

    cols_pi = [c for c in PI_FULL_FEATURES if c in train_df.columns]
    Xtr = train_df[cols_pi].copy()
    ytr = y_train.copy()
    mask = Xtr.notna().all(axis=1)
    Xtr = Xtr[mask]
    ytr = ytr[mask]

    gbc = GradientBoostingClassifier(**{**GBC_PARAMS, "random_state": 42})
    gbc.fit(Xtr, ytr)

    importances = pd.Series(gbc.feature_importances_, index=cols_pi)
    importances = importances.sort_values(ascending=False)
    print(f"\nTop 15 features (importance):")
    for feat, imp in importances.head(15).items():
        pi_marker = " <-- PI" if "pi" in feat.lower() else ""
        print(f"  {feat:<30} {imp:.4f}{pi_marker}")

    # Pi-specific
    print(f"\nPi feature importances:")
    for feat in ["home_pi", "away_pi", "pi_diff"]:
        if feat in importances.index:
            rank = list(importances.index).index(feat) + 1
            print(f"  {feat:<15} imp={importances[feat]:.4f}  rank={rank}/{len(importances)}")

    # ── 7. Pi-conditional analysis ──────────────────────────────────────────
    run_pi_conditional_analysis(matrix)

    # ── 8. Redundancy check — correlation with existing features ────────────
    print("\n" + "=" * 80)
    print("REDUNDANCY CHECK — Pi vs existing features (train set)")
    print("=" * 80)

    check_df = train_df[
        BASELINE_FEATURES + ["home_pi", "away_pi", "pi_diff"]
    ].dropna()

    if len(check_df) > 100:
        for pi_col in ["home_pi", "away_pi", "pi_diff"]:
            corrs = check_df[BASELINE_FEATURES].corrwith(check_df[pi_col]).abs()
            top3 = corrs.nlargest(3)
            print(f"\n  {pi_col} — top 3 absolute correlations:")
            for feat, corr in top3.items():
                print(f"    {feat:<30} r={corr:.3f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
