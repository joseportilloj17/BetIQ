"""
f5_recency_diagnostic.py — Stale data vs market efficiency

Two hypotheses for F5's year-over-year decline (2023 +0.15pp → 2026 -2.86pp):
  A) Stale patterns — model's learned relationships going stale → AUC declines
  B) Market efficiency — market got sharper at pricing F5 → AUC holds, edge declines

Test 1 (cheap): Year-level AUC decomposition. AUC decline = stale. AUC stable = market.
Test 2: Recency-trained F5 model (2025-26 only) vs all-data (2023-26) on held-out 2026.

Context: MLB's big rule changes (pitch clock, shift ban, bigger bases) all hit 2023 —
the START of our data window. All 2023-26 data is new-rules-era. The decline is
WITHIN new-rules baseball, so it's either stale learned coefficients or market sharpening.

Usage:
  python f5_recency_diagnostic.py
"""
from __future__ import annotations

import os
import sys
import sqlite3
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_BASE = os.path.dirname(os.path.dirname(__file__))
_DATA = os.path.join(_BASE, "data")
_HIST_DB = os.path.join(_DATA, "historical.db")

sys.path.insert(0, os.path.join(_BASE, "backend"))

F5_ODDS = 1.909   # -110 decimal
F5_BE = 1.0 / F5_ODDS  # 52.38%


def _load_f5_data() -> pd.DataFrame:
    """Load feature matrix + F5 outcomes + year column."""
    from features import build_feature_matrix, TARGET_ATS, MODEL_FEATURE_COLS_V1_PRUNED

    fm = build_feature_matrix("MLB")
    fm["game_pk"] = fm["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")

    conn = sqlite3.connect(_HIST_DB)
    f5 = pd.read_sql_query(
        "SELECT game_pk, f5_home, f5_away FROM game_innings "
        "WHERE f5_home IS NOT NULL AND f5_away IS NOT NULL",
        conn,
    )
    conn.close()

    f5["f5_target"] = ((f5["f5_home"] - f5["f5_away"] + 0.5) > 0).astype(int)
    merged = fm.merge(f5[["game_pk", "f5_target"]], on="game_pk", how="inner")
    merged = merged.dropna(subset=[TARGET_ATS, "f5_target"])

    # Year column
    merged["year"] = pd.to_datetime(merged["game_date"]).dt.year

    # Sort chronologically
    merged = merged.sort_values("game_date").reset_index(drop=True)

    print(f"  Loaded {len(merged)} games with features + F5 target")
    for yr, cnt in merged.groupby("year").size().items():
        print(f"    {yr}: {cnt} games")

    return merged


def _score_with_existing_model(merged: pd.DataFrame) -> pd.DataFrame:
    """Score all games with the existing ATS model."""
    from ml_model import _load_ats_submodel

    artifacts = _load_ats_submodel("MLB")
    clf, scaler, imputer, feat_cols = artifacts

    available = [c for c in feat_cols if c in merged.columns and not c.startswith("_")]
    X = merged[available].copy()
    if "_spread_value" in feat_cols:
        X["_spread_value"] = -1.5
    if "_spread_abs" in feat_cols:
        X["_spread_abs"] = 1.5
    if "_is_fav_spread" in feat_cols:
        X["_is_fav_spread"] = 1
    X = X.reindex(columns=feat_cols, fill_value=0).fillna(X.median())

    X_s = scaler.transform(X) if scaler else X.values
    X_i = imputer.transform(X_s) if imputer else X_s

    merged = merged.copy()
    merged["prob_existing"] = clf.predict_proba(X_i)[:, 1]
    return merged


def _wr_from_probs(probs: np.ndarray, y: np.ndarray) -> float:
    """WR when picking home if prob >= 0.5, else away."""
    picks_home = probs >= 0.5
    correct = np.where(picks_home, y == 1, y == 0)
    return correct.mean()


def _confidence_interval(wr, n, z=1.96):
    p = wr
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: AUC-vs-Edge decomposition by year
# ═══════════════════════════════════════════════════════════════════════════════

def test1_auc_decomposition(merged: pd.DataFrame):
    """
    Per-year AUC + WR-vs-BE on F5 outcomes using the existing model.
    If AUC holds but WR-BE declines → market got sharper (efficiency).
    If AUC declines → stale patterns (recency might help).
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    print("\n" + "=" * 70)
    print("TEST 1: AUC-vs-EDGE DECOMPOSITION BY YEAR")
    print("Does F5 discrimination (AUC) decline, or only the edge (WR-BE)?")
    print("=" * 70)

    probs = merged["prob_existing"].values
    y = merged["f5_target"].values

    print(f"\n  {'Year':<6} {'N':<6} {'AUC':<8} {'Brier':<8} {'WR%':<8} "
          f"{'WR-BE':<8} {'F5 base%':<10}")

    year_results = []
    for yr in sorted(merged["year"].unique()):
        mask = merged["year"] == yr
        p_yr = probs[mask.values]
        y_yr = y[mask.values]
        n_yr = mask.sum()

        if n_yr < 50:
            continue

        auc = roc_auc_score(y_yr, p_yr)
        brier = brier_score_loss(y_yr, p_yr)
        wr = _wr_from_probs(p_yr, y_yr)
        base_rate = y_yr.mean()  # F5 home cover base rate

        year_results.append({
            "year": yr, "n": n_yr, "auc": auc, "brier": brier,
            "wr": wr, "wr_be": wr - F5_BE, "base_rate": base_rate,
        })

        print(f"  {yr:<6} {n_yr:<6} {auc:.4f}  {brier:.4f}  "
              f"{wr*100:.2f}%  {(wr-F5_BE)*100:+.2f}pp {base_rate*100:.1f}%")

    # Half-sample: first half vs second half
    mid = len(merged) // 2
    p1, y1 = probs[:mid], y[:mid]
    p2, y2 = probs[mid:], y[mid:]
    auc1, auc2 = roc_auc_score(y1, p1), roc_auc_score(y2, p2)
    wr1, wr2 = _wr_from_probs(p1, y1), _wr_from_probs(p2, y2)

    print(f"\n  Half-sample split:")
    print(f"    First half  (N={mid}):  AUC={auc1:.4f}, WR={wr1*100:.2f}%, "
          f"WR-BE={((wr1-F5_BE)*100):+.2f}pp")
    print(f"    Second half (N={len(merged)-mid}): AUC={auc2:.4f}, WR={wr2*100:.2f}%, "
          f"WR-BE={((wr2-F5_BE)*100):+.2f}pp")
    print(f"    ΔAUC = {auc2-auc1:+.4f}")

    # Diagnosis
    aucs = [r["auc"] for r in year_results]
    wrbes = [r["wr_be"] for r in year_results]

    auc_trend = np.polyfit(range(len(aucs)), aucs, 1)[0]
    wrbe_trend = np.polyfit(range(len(wrbes)), wrbes, 1)[0]

    print(f"\n  ── Trend analysis ──")
    print(f"    AUC slope:   {auc_trend:+.4f}/year")
    print(f"    WR-BE slope: {wrbe_trend*100:+.2f}pp/year")

    if abs(auc_trend) < 0.01 and wrbe_trend < -0.005:
        diagnosis = "MARKET EFFICIENCY — AUC stable, edge declining"
        implication = "Recency training WON'T help. Market prices the signal."
    elif auc_trend < -0.01:
        diagnosis = "STALE PATTERNS — AUC itself declining"
        implication = "Recency training MIGHT help. Model relationships going stale."
    elif auc_trend > 0.01:
        diagnosis = "MODEL IMPROVING — AUC increasing but edge still declining"
        implication = "Market catching up FASTER than model improves."
    else:
        diagnosis = "AMBIGUOUS — small trends, need recency test to disambiguate"
        implication = "Run Test 2 to check directly."

    print(f"\n  DIAGNOSIS: {diagnosis}")
    print(f"  IMPLICATION: {implication}")

    return year_results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Recency-trained F5 model vs all-data model
# ═══════════════════════════════════════════════════════════════════════════════

def test2_recency_training(merged: pd.DataFrame):
    """
    Train F5-specific classifier on:
      A) All data (2023-2026 train, hold out late 2026)
      B) Recent only (2025-2026 train, hold out late 2026)
    Test both on the SAME held-out 2026 games.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from features import MODEL_FEATURE_COLS_V1_PRUNED

    print("\n" + "=" * 70)
    print("TEST 2: RECENCY-TRAINED F5 MODEL vs ALL-DATA MODEL")
    print("Train on recent-only (2025-26) vs all-data (2023-26), test on 2026 OOS")
    print("=" * 70)

    feat_cols = MODEL_FEATURE_COLS_V1_PRUNED
    available = [c for c in feat_cols if c in merged.columns]
    missing = [c for c in feat_cols if c not in merged.columns]
    if missing:
        print(f"  Missing features (dropped): {missing}")

    # ── Define test set: 2026 games (chronological OOS) ──
    test_mask = merged["year"] == 2026
    n_test = test_mask.sum()

    if n_test < 50:
        print(f"  Only {n_test} 2026 games — insufficient for test")
        return

    # Further split 2026 into train/test: use last 40% of 2026 as test
    idx_2026 = merged[test_mask].index
    split_2026 = int(len(idx_2026) * 0.6)
    test_final_idx = idx_2026[split_2026:]
    train_2026_idx = idx_2026[:split_2026]

    test_df = merged.loc[test_final_idx]
    n_test_final = len(test_df)

    print(f"\n  Test set: last 40% of 2026 = {n_test_final} games")
    print(f"  Test date range: {test_df['game_date'].min()} to {test_df['game_date'].max()}")

    y_test = test_df["f5_target"].values
    X_test = test_df[available].fillna(test_df[available].median())

    # ── Model A: ALL data (2023 through first 60% of 2026) ──
    train_all_mask = merged.index < test_final_idx[0]
    train_all = merged[train_all_mask]

    # ── Model B: Recent only (2025 through first 60% of 2026) ──
    train_recent_mask = train_all_mask & (merged["year"] >= 2025)
    train_recent = merged[train_recent_mask]

    # ── Model C: 2025 only (pure recent, no 2026 leakage concern) ──
    train_2025_mask = merged["year"] == 2025
    train_2025 = merged[train_2025_mask]

    configs = [
        ("All data (2023-26)", train_all),
        ("Recent (2025-26)", train_recent),
        ("2025 only", train_2025),
    ]

    print(f"\n  {'Config':<25} {'Train N':<10} {'Years':<15}")
    for name, df in configs:
        yrs = sorted(df["year"].unique())
        print(f"  {name:<25} {len(df):<10} {yrs}")

    # ── Train and evaluate each config ──
    print(f"\n  {'Config':<25} {'AUC':<8} {'Brier':<8} {'WR%':<8} "
          f"{'WR-BE':<8} {'ROI%':<8}")

    model_results = []
    for name, train_df in configs:
        X_train = train_df[available].fillna(train_df[available].median())
        y_train = train_df["f5_target"].values

        # Scale
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)

        # Impute
        imp = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr_s)
        X_te_i = imp.transform(X_te_s)

        # Train GBC (same hyperparams as ATS model)
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_tr_i, y_train)

        probs = clf.predict_proba(X_te_i)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        wr = _wr_from_probs(probs, y_test)
        ci_lo, ci_hi = _confidence_interval(wr, n_test_final)

        model_results.append({
            "name": name, "auc": auc, "brier": brier,
            "wr": wr, "wr_be": wr - F5_BE, "roi": (wr * F5_ODDS - 1) * 100,
            "n_train": len(train_df), "n_test": n_test_final,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })

        print(f"  {name:<25} {auc:.4f}  {brier:.4f}  "
              f"{wr*100:.2f}%  {(wr-F5_BE)*100:+.2f}pp {(wr*F5_ODDS-1)*100:+.2f}%")

    # ── Also test the EXISTING pre-trained model on same test set ──
    existing_probs = test_df["prob_existing"].values
    existing_auc = roc_auc_score(y_test, existing_probs)
    existing_brier = brier_score_loss(y_test, existing_probs)
    existing_wr = _wr_from_probs(existing_probs, y_test)

    print(f"  {'Existing ATS (pre-trained)':<25} {existing_auc:.4f}  "
          f"{existing_brier:.4f}  {existing_wr*100:.2f}%  "
          f"{(existing_wr-F5_BE)*100:+.2f}pp {(existing_wr*F5_ODDS-1)*100:+.2f}%")

    # ── Multi-seed robustness (5 seeds, GBC) ──
    print(f"\n  ── Multi-seed robustness (5 seeds) ──")
    print(f"  {'Config':<25} {'Mean WR%':<10} {'Mean WR-BE':<12} {'Std':<8} {'Min':<8} {'Max'}")

    for name, train_df in configs:
        X_train = train_df[available].fillna(train_df[available].median())
        y_train = train_df["f5_target"].values
        seed_wrs = []

        for seed in range(5):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_train)
            X_te_s = sc.transform(X_test)
            imp = SimpleImputer(strategy="median")
            X_tr_i = imp.fit_transform(X_tr_s)
            X_te_i = imp.transform(X_te_s)

            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=seed,
            )
            clf.fit(X_tr_i, y_train)
            probs = clf.predict_proba(X_te_i)[:, 1]
            wr = _wr_from_probs(probs, y_test)
            seed_wrs.append(wr)

        mean_wr = np.mean(seed_wrs)
        std_wr = np.std(seed_wrs)
        print(f"  {name:<25} {mean_wr*100:.2f}%    "
              f"{(mean_wr-F5_BE)*100:+.2f}pp     {std_wr*100:.2f}pp "
              f"  {min(seed_wrs)*100:.2f}% {max(seed_wrs)*100:.2f}%")

    # ── Interpretation ──
    best = max(model_results, key=lambda r: r["wr_be"])
    all_data = [r for r in model_results if "All" in r["name"]][0]
    recent = [r for r in model_results if "Recent" in r["name"]][0]

    print(f"\n  ── Interpretation ──")
    delta_wr = recent["wr"] - all_data["wr"]
    delta_auc = recent["auc"] - all_data["auc"]
    print(f"    Recent vs All-data: ΔAUC={delta_auc:+.4f}, ΔWR={delta_wr*100:+.2f}pp")

    if recent["wr_be"] > 0 and recent["wr_be"] > all_data["wr_be"] + 0.02:
        print(f"    → RECENCY HELPS: recent-only beats all-data by "
              f"{(recent['wr']-all_data['wr'])*100:+.2f}pp")
        print(f"    → Old data was hurting. Worth retraining on recent.")
        print(f"    → CHECK FULL-GAME SPREADS with recency too.")
    elif recent["wr_be"] > all_data["wr_be"]:
        print(f"    → RECENCY SLIGHTLY BETTER but both below/near BE")
        print(f"    → Marginal gain, not enough to resurrect F5")
    else:
        print(f"    → RECENCY DOESN'T HELP: all-data ≥ recent-only")
        print(f"    → Stale data is NOT the problem. Market is efficient.")

    return model_results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Full-game spread recency (only if F5 recency helps)
# ═══════════════════════════════════════════════════════════════════════════════

def test3_fullgame_recency(merged: pd.DataFrame):
    """
    If F5 recency helped, test whether full-game spread model also benefits
    from recency training. Uses covered_spread as target.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from features import MODEL_FEATURE_COLS_V1_PRUNED, TARGET_ATS

    print("\n" + "=" * 70)
    print("TEST 3: FULL-GAME SPREAD RECENCY CHECK")
    print("Does recency-training help full-game ATS too?")
    print("=" * 70)

    feat_cols = MODEL_FEATURE_COLS_V1_PRUNED
    available = [c for c in feat_cols if c in merged.columns]

    # Same test set as Test 2
    test_mask = merged["year"] == 2026
    idx_2026 = merged[test_mask].index
    split_2026 = int(len(idx_2026) * 0.6)
    test_final_idx = idx_2026[split_2026:]
    test_df = merged.loc[test_final_idx]

    y_test = test_df[TARGET_ATS].values.astype(int)
    X_test = test_df[available].fillna(test_df[available].median())

    train_all_mask = merged.index < test_final_idx[0]
    train_recent_mask = train_all_mask & (merged["year"] >= 2025)

    configs = [
        ("All data (2023-26)", merged[train_all_mask]),
        ("Recent (2025-26)", merged[train_recent_mask]),
    ]

    print(f"\n  Target: {TARGET_ATS} (full-game spread cover)")
    print(f"  Test set: {len(test_df)} games (late 2026)")
    print(f"\n  {'Config':<25} {'AUC':<8} {'WR%':<8} {'WR-BE':<8}")

    full_game_be = 0.5238  # -110 spread breakeven

    for name, train_df in configs:
        X_train = train_df[available].fillna(train_df[available].median())
        y_train = train_df[TARGET_ATS].values.astype(int)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
        imp = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr_s)
        X_te_i = imp.transform(X_te_s)

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_tr_i, y_train)
        probs = clf.predict_proba(X_te_i)[:, 1]
        auc = roc_auc_score(y_test, probs)
        wr = _wr_from_probs(probs, y_test)

        print(f"  {name:<25} {auc:.4f}  {wr*100:.2f}%  {(wr-full_game_be)*100:+.2f}pp")


def main():
    print("=" * 70)
    print("F5 RECENCY DIAGNOSTIC")
    print("Stale data vs market efficiency — which explains the decline?")
    print("=" * 70)
    print("\n  Context: MLB rule changes (pitch clock, shift ban, bigger bases)")
    print("  all hit in 2023 — the START of our window. All data is new-rules-era.")
    print("  The year-degradation is WITHIN new-rules baseball.")

    # Load and score
    print("\n── Loading data + scoring with existing model ──")
    merged = _load_f5_data()
    merged = _score_with_existing_model(merged)

    # Test 1: AUC decomposition (cheap, runs first)
    year_results = test1_auc_decomposition(merged)

    # Test 2: Recency training
    model_results = test2_recency_training(merged)

    # Test 3: Full-game recency (always run for completeness)
    test3_fullgame_recency(merged)

    # ═══ FINAL VERDICT ═══
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Summarize AUC trend
    aucs = [r["auc"] for r in year_results]
    auc_trend = np.polyfit(range(len(aucs)), aucs, 1)[0]
    wrbes = [r["wr_be"] for r in year_results]
    wrbe_trend = np.polyfit(range(len(wrbes)), wrbes, 1)[0]

    print(f"\n  AUC trend:   {auc_trend:+.4f}/year "
          f"({'DECLINING' if auc_trend < -0.01 else 'STABLE' if abs(auc_trend) < 0.01 else 'IMPROVING'})")
    print(f"  WR-BE trend: {wrbe_trend*100:+.2f}pp/year "
          f"({'DECLINING' if wrbe_trend < -0.005 else 'STABLE'})")

    # Recency comparison
    all_data = [r for r in model_results if "All" in r["name"]][0]
    recent = [r for r in model_results if "Recent" in r["name"]][0]
    print(f"\n  Recency vs all-data on 2026 OOS:")
    print(f"    All-data:    WR={all_data['wr']*100:.2f}%, WR-BE={all_data['wr_be']*100:+.2f}pp")
    print(f"    Recent-only: WR={recent['wr']*100:.2f}%, WR-BE={recent['wr_be']*100:+.2f}pp")

    if abs(auc_trend) < 0.01 and wrbe_trend < -0.005:
        print(f"\n  → MARKET EFFICIENCY: AUC stable, edge declining")
        print(f"    The model ranks F5 outcomes just as well as before,")
        print(f"    but the market prices it too. Recency can't fix this.")
    elif auc_trend < -0.01 and recent["wr_be"] > all_data["wr_be"] + 0.01:
        print(f"\n  → STALE DATA confirmed + RECENCY HELPS")
        print(f"    Model patterns going stale AND recent training improves edge.")
        print(f"    Worth retraining on recent data and re-testing.")
    elif auc_trend < -0.01:
        print(f"\n  → STALE PATTERNS but recency doesn't rescue edge")
        print(f"    Discrimination declining but even fresh model can't clear vig.")
    else:
        print(f"\n  → NEITHER pure stale nor pure efficiency")
        print(f"    Mixed signal. F5 edge was never robust enough to survive vig.")


if __name__ == "__main__":
    main()
