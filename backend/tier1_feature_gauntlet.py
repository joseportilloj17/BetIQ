#!/usr/bin/env python3
"""
Tier 1 feature gauntlet — combined test for:
  1a. Feature×rung explicit interaction terms
  1b. Statcast process stats (barrel_diff, hard_hit_diff, xwoba_diff)

Protocol:
  - 5-fold TSCV with 3-day purge gap
  - GBC(300, d3, lr=0.05) + isotonic cv=3  (production params)
  - ATS model is DETERMINISTIC at fixed random_state — single-run exact
  - Report AUC and Brier vs V1_PRUNED baseline
  - ΔBrier < -0.002 required to pass first-look

Baseline: V1_PRUNED (21 feat) + spread augmentation (3 feat) = 24 features
"""
from __future__ import annotations

import os
import sys
import sqlite3
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from features import (
    build_feature_matrix,
    MODEL_FEATURE_COLS_V1_PRUNED,
    TARGET_ATS,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

# Production GBC params
GBC_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)

# Spread augmentation (matches production)
SPREAD_COLS = ["_spread_value", "_spread_abs", "_is_fav_spread"]


# ═══════════════════════════════════════════════════════════════
# TSCV ENGINE
# ═══════════════════════════════════════════════════════════════

def temporal_cv_splits(dates: pd.Series, n_folds: int = 5, purge_days: int = 3):
    """
    Expanding-window temporal CV with purge gap.
    Returns list of (train_idx, test_idx) arrays.
    """
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end_idx = (i + 1) * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_dates)
        if test_start_idx >= n_dates:
            break

        train_end_date = unique_dates[train_end_idx - 1]
        test_start_date = unique_dates[test_start_idx]

        # Purge: remove train dates within purge_days of test start
        purge_cutoff = pd.Timestamp(test_start_date) - timedelta(days=purge_days)
        train_mask = (dates <= str(purge_cutoff))
        test_mask = dates.isin(unique_dates[test_start_idx:test_end_idx])

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 50 and len(test_idx) > 50:
            folds.append((train_idx, test_idx))

    return folds


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    feat_cols: list[str],
    label: str,
    n_folds: int = 5,
) -> dict:
    """
    Run TSCV on the given feature set, return pooled AUC + Brier.
    """
    folds = temporal_cv_splits(dates, n_folds=n_folds)
    all_preds = []
    all_labels = []

    for train_idx, test_idx in folds:
        X_train = X.iloc[train_idx][feat_cols].values
        X_test = X.iloc[test_idx][feat_cols].values
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values

        # Impute NaN
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        # GBC + isotonic
        gbc = GradientBoostingClassifier(**GBC_PARAMS)
        cal = CalibratedClassifierCV(gbc, method="isotonic", cv=3)
        cal.fit(X_train, y_train)
        preds = cal.predict_proba(X_test)[:, 1]

        all_preds.extend(preds)
        all_labels.extend(y_test)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    brier = brier_score_loss(all_labels, all_preds)

    return {"label": label, "auc": auc, "brier": brier, "n": len(all_preds),
            "n_feat": len(feat_cols)}


# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix with all candidate features, return (X, y, dates).
    """
    print("Building feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"  Raw: {len(df)} games")

    # Filter to rows with ATS target
    df = df.dropna(subset=[TARGET_ATS])
    y = df[TARGET_ATS].astype(int)

    # Add spread augmentation (matches production inference)
    if "close_spread" in df.columns:
        df["_spread_value"] = df["close_spread"].abs()
        df["_spread_abs"] = df["close_spread"].abs()
        df["_is_fav_spread"] = (df["close_spread"] < 0).astype(float)
    else:
        df["_spread_value"] = np.nan
        df["_spread_abs"] = np.nan
        df["_is_fav_spread"] = np.nan

    # Add xwoba_diff from team_batting_stats
    print("  Joining xwoba from team_batting_stats...")
    conn = sqlite3.connect(DB_PATH)
    xwoba_rows = conn.execute(
        "SELECT team_name, season, xwoba FROM team_batting_stats WHERE xwoba IS NOT NULL"
    ).fetchall()
    conn.close()

    xwoba_map = {}
    for team, season, xw in xwoba_rows:
        xwoba_map[(team, str(season))] = xw

    # Extract season from game_date
    if "season" in df.columns:
        df["_season"] = df["season"].astype(str)
    elif "game_date" in df.columns:
        df["_season"] = pd.to_datetime(df["game_date"]).dt.year.astype(str)
    else:
        df["_season"] = "2026"

    # Match xwoba to home/away teams (vectorized)
    xwoba_series = pd.Series(xwoba_map)
    if "home_team" in df.columns:
        h_keys = list(zip(df["home_team"], df["_season"]))
        a_keys = list(zip(df["away_team"], df["_season"]))
        df["h_xwoba"] = [xwoba_map.get(k, np.nan) for k in h_keys]
        df["a_xwoba"] = [xwoba_map.get(k, np.nan) for k in a_keys]
    else:
        df["h_xwoba"] = np.nan
        df["a_xwoba"] = np.nan
    df["xwoba_diff"] = df["h_xwoba"] - df["a_xwoba"]
    xwoba_fill = df["xwoba_diff"].notna().sum()
    print(f"  xwoba_diff fill: {xwoba_fill}/{len(df)} ({xwoba_fill / len(df) * 100:.1f}%)")

    # Build explicit interaction terms (feature × spread_value)
    for feat in MODEL_FEATURE_COLS_V1_PRUNED:
        if feat in df.columns:
            df[f"{feat}__x_spread"] = df[feat] * df["_spread_value"]

    # Process-vs-outcome gap features
    # barrel_gap: barrel_diff residualized against ops_diff (process divergence from outcome)
    if "barrel_diff" in df.columns and "ops_diff" in df.columns:
        mask = df["barrel_diff"].notna() & df["ops_diff"].notna()
        if mask.sum() > 100:
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(
                df.loc[mask, "ops_diff"], df.loc[mask, "barrel_diff"]
            )
            df["barrel_gap"] = np.nan
            df.loc[mask, "barrel_gap"] = (
                df.loc[mask, "barrel_diff"] - (slope * df.loc[mask, "ops_diff"] + intercept)
            )
            print(f"  barrel_gap (resid vs ops_diff): r={np.corrcoef(df.loc[mask,'barrel_gap'], df.loc[mask,'ops_diff'])[0,1]:.4f}")
        else:
            df["barrel_gap"] = np.nan

    # hard_hit_gap: hard_hit_diff residualized against ops_diff
    if "hard_hit_diff" in df.columns and "ops_diff" in df.columns:
        mask = df["hard_hit_diff"].notna() & df["ops_diff"].notna()
        if mask.sum() > 100:
            slope, intercept, _, _, _ = linregress(
                df.loc[mask, "ops_diff"], df.loc[mask, "hard_hit_diff"]
            )
            df["hard_hit_gap"] = np.nan
            df.loc[mask, "hard_hit_gap"] = (
                df.loc[mask, "hard_hit_diff"] - (slope * df.loc[mask, "ops_diff"] + intercept)
            )
            print(f"  hard_hit_gap (resid vs ops_diff): r={np.corrcoef(df.loc[mask,'hard_hit_gap'], df.loc[mask,'ops_diff'])[0,1]:.4f}")
        else:
            df["hard_hit_gap"] = np.nan

    # xwoba_gap: xwoba_diff residualized against ops_diff
    if "xwoba_diff" in df.columns and "ops_diff" in df.columns:
        mask = df["xwoba_diff"].notna() & df["ops_diff"].notna()
        if mask.sum() > 100:
            slope, intercept, _, _, _ = linregress(
                df.loc[mask, "ops_diff"], df.loc[mask, "xwoba_diff"]
            )
            df["xwoba_gap"] = np.nan
            df.loc[mask, "xwoba_gap"] = (
                df.loc[mask, "xwoba_diff"] - (slope * df.loc[mask, "ops_diff"] + intercept)
            )
            print(f"  xwoba_gap (resid vs ops_diff): r={np.corrcoef(df.loc[mask,'xwoba_gap'], df.loc[mask,'ops_diff'])[0,1]:.4f}")
        else:
            df["xwoba_gap"] = np.nan

    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    else:
        dates = pd.Series(["2020-01-01"] * len(df), index=df.index)

    return df, y, dates


# ═══════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════

def run_gauntlet():
    df, y, dates = prepare_data()

    # ── Baseline: V1_PRUNED + spread augmentation ──
    baseline_cols = [c for c in MODEL_FEATURE_COLS_V1_PRUNED if c in df.columns] + \
                    [c for c in SPREAD_COLS if c in df.columns]

    results = []

    print(f"\n{'='*75}")
    print("TIER 1a: FEATURE×RUNG INTERACTION TERMS")
    print(f"{'='*75}")
    print("Q: Do explicit feature×spread interactions improve beyond GBC(d3)?")
    print(f"Baseline: V1_PRUNED ({len(MODEL_FEATURE_COLS_V1_PRUNED)}) + spread aug (3) = {len(baseline_cols)} features\n")

    # A0: Baseline
    r = evaluate_feature_set(df, y, dates, baseline_cols, "A0: Baseline (V1+spread)")
    results.append(r)
    print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  N={r['n']}  feat={r['n_feat']}")

    # A1: Top-5 feature×spread interactions
    top5_interact = [
        "sp_era_diff__x_spread",
        "ops_diff__x_spread",
        "opp_strength_diff__x_spread",
        "sp_whip_diff__x_spread",
        "sp_era5_diff__x_spread",
    ]
    top5_avail = [c for c in top5_interact if c in df.columns]
    if top5_avail:
        r = evaluate_feature_set(df, y, dates, baseline_cols + top5_avail,
                                 f"A1: + top-5 interactions ({len(top5_avail)})")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  N={r['n']}  feat={r['n_feat']}")

    # A2: All 21 V1 feature×spread interactions
    all_interact = [f"{c}__x_spread" for c in MODEL_FEATURE_COLS_V1_PRUNED]
    all_interact_avail = [c for c in all_interact if c in df.columns]
    if all_interact_avail:
        r = evaluate_feature_set(df, y, dates, baseline_cols + all_interact_avail,
                                 f"A2: + all-21 interactions ({len(all_interact_avail)})")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  N={r['n']}  feat={r['n_feat']}")

    # A3: FAST SCREEN — single chrono split to rank individual interactions by importance
    # (Full TSCV on all 21 individually would take ~21 min; screen first, validate winners)
    print(f"\n  ── Individual interaction importance (single chrono split screen) ──")
    base_brier = results[0]["brier"]
    base_auc = results[0]["auc"]
    individual = []

    # Fit one GBC with ALL interactions, read importance
    all_int_cols = [f"{c}__x_spread" for c in MODEL_FEATURE_COLS_V1_PRUNED if f"{c}__x_spread" in df.columns]
    screen_cols = baseline_cols + all_int_cols
    n_total = len(df)
    train_end = int(n_total * 0.7)

    X_screen_tr = df.iloc[:train_end][screen_cols].values
    X_screen_te = df.iloc[train_end:][screen_cols].values
    y_screen_tr = y.iloc[:train_end].values
    y_screen_te = y.iloc[train_end:].values

    imp_screen = SimpleImputer(strategy="median")
    X_screen_tr = imp_screen.fit_transform(X_screen_tr)
    X_screen_te = imp_screen.transform(X_screen_te)

    gbc_screen = GradientBoostingClassifier(**GBC_PARAMS)
    gbc_screen.fit(X_screen_tr, y_screen_tr)

    importances = gbc_screen.feature_importances_
    n_base = len(baseline_cols)
    print(f"    {'Interaction':<35} {'Importance':<12} {'Rank'}")
    int_imps = []
    for i, col in enumerate(all_int_cols):
        imp_val = importances[n_base + i]
        int_imps.append((col, imp_val))
    int_imps.sort(key=lambda x: -x[1])
    for rank, (col, imp_val) in enumerate(int_imps, 1):
        flag = " ←" if imp_val > 0.01 else ""
        print(f"    {col:<35} {imp_val:.4f}       #{rank}{flag}")

    # Full TSCV on top-3 individual interactions only
    top3_int = [x[0] for x in int_imps[:3]]
    print(f"\n  ── Full TSCV on top-3 individual interactions ──")
    for col in top3_int:
        r = evaluate_feature_set(df, y, dates, baseline_cols + [col], f"  + {col}")
        delta_b = r["brier"] - base_brier
        delta_a = r["auc"] - base_auc
        flag = " ***" if delta_b < -0.002 else (" *" if delta_b < -0.001 else "")
        print(f"    {col:<35} ΔBrier={delta_b:+.4f}  ΔAUC={delta_a:+.4f}{flag}")
        individual.append((col.replace("__x_spread", ""), delta_b, delta_a, r))
    individual.sort(key=lambda x: x[1])

    print(f"\n{'='*75}")
    print("TIER 1b: STATCAST PROCESS STATS")
    print(f"{'='*75}")
    print("Q: Do barrel_diff / hard_hit_diff / xwoba_diff add to V1_PRUNED?")
    print("   (They exist in feature matrix but were pruned — orthogonal residuals too)\n")

    # B1: + barrel_diff
    if "barrel_diff" in df.columns:
        r = evaluate_feature_set(df, y, dates, baseline_cols + ["barrel_diff"],
                                 "B1: + barrel_diff")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B2: + hard_hit_diff
    if "hard_hit_diff" in df.columns:
        r = evaluate_feature_set(df, y, dates, baseline_cols + ["hard_hit_diff"],
                                 "B2: + hard_hit_diff")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B3: + barrel_diff + hard_hit_diff
    both_statcast = [c for c in ["barrel_diff", "hard_hit_diff"] if c in df.columns]
    if both_statcast:
        r = evaluate_feature_set(df, y, dates, baseline_cols + both_statcast,
                                 "B3: + barrel + hard_hit")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B4: + xwoba_diff (raw)
    if "xwoba_diff" in df.columns and df["xwoba_diff"].notna().sum() > 1000:
        r = evaluate_feature_set(df, y, dates, baseline_cols + ["xwoba_diff"],
                                 "B4: + xwoba_diff (raw)")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B5: + barrel_gap (residualized vs ops_diff)
    if "barrel_gap" in df.columns and df["barrel_gap"].notna().sum() > 1000:
        r = evaluate_feature_set(df, y, dates, baseline_cols + ["barrel_gap"],
                                 "B5: + barrel_gap (resid)")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B6: + hard_hit_gap (residualized vs ops_diff)
    if "hard_hit_gap" in df.columns and df["hard_hit_gap"].notna().sum() > 1000:
        r = evaluate_feature_set(df, y, dates, baseline_cols + ["hard_hit_gap"],
                                 "B6: + hard_hit_gap (resid)")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B7: + xwoba_gap (residualized vs ops_diff)
    if "xwoba_gap" in df.columns and df["xwoba_gap"].notna().sum() > 1000:
        r = evaluate_feature_set(df, y, dates, baseline_cols + ["xwoba_gap"],
                                 "B7: + xwoba_gap (resid)")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B8: All Statcast (barrel+hard_hit+xwoba raw)
    all_statcast = [c for c in ["barrel_diff", "hard_hit_diff", "xwoba_diff"] if c in df.columns and df[c].notna().sum() > 1000]
    if len(all_statcast) >= 2:
        r = evaluate_feature_set(df, y, dates, baseline_cols + all_statcast,
                                 f"B8: + all Statcast raw ({len(all_statcast)})")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # B9: All Statcast gaps (residualized)
    all_gaps = [c for c in ["barrel_gap", "hard_hit_gap", "xwoba_gap"] if c in df.columns and df[c].notna().sum() > 1000]
    if all_gaps:
        r = evaluate_feature_set(df, y, dates, baseline_cols + all_gaps,
                                 f"B9: + all gaps resid ({len(all_gaps)})")
        results.append(r)
        print(f"  {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # ── COMBINED BEST ──
    print(f"\n{'='*75}")
    print("COMBINED: BEST SURVIVORS FROM TIER 1a + 1b")
    print(f"{'='*75}")

    # Collect all candidates that improved baseline
    survivors = []
    for r in results[1:]:  # skip baseline
        if r["brier"] < base_brier - 0.001:
            survivors.append(r)

    if survivors:
        print(f"\n  Candidates with ΔBrier < -0.001:")
        for s in sorted(survivors, key=lambda x: x["brier"]):
            print(f"    {s['label']:<45} ΔBrier={s['brier']-base_brier:+.4f}  ΔAUC={s['auc']-base_auc:+.4f}")
    else:
        print(f"\n  NO candidates passed ΔBrier < -0.001 threshold")

    # Best individual interactions
    if individual:
        best_interactions = [x for x in individual if x[1] < -0.001]
        if best_interactions:
            print(f"\n  Best individual interactions (ΔBrier < -0.001):")
            for feat, db, da, _ in best_interactions:
                print(f"    {feat}×spread: ΔBrier={db:+.4f}, ΔAUC={da:+.4f}")

            # Test combining best interaction survivors
            best_int_cols = [f"{x[0]}__x_spread" for x in best_interactions if f"{x[0]}__x_spread" in df.columns]
            if len(best_int_cols) >= 2:
                r = evaluate_feature_set(df, y, dates, baseline_cols + best_int_cols,
                                         f"COMBO: best interactions ({len(best_int_cols)})")
                results.append(r)
                print(f"\n    {r['label']:<45} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ΔBrier={r['brier']-base_brier:+.4f}")

    # ── CORRELATION CHECK: are Statcast features redundant with existing? ──
    print(f"\n{'='*75}")
    print("REDUNDANCY CHECK: Statcast vs existing V1 features")
    print(f"{'='*75}")

    check_feats = ["barrel_diff", "hard_hit_diff", "xwoba_diff", "barrel_gap", "hard_hit_gap", "xwoba_gap"]
    v1_corr_targets = ["ops_diff", "sp_era_diff", "a_ops", "h_bb_pct"]

    for feat in check_feats:
        if feat not in df.columns or df[feat].notna().sum() < 1000:
            continue
        corrs = []
        for v1f in v1_corr_targets:
            if v1f in df.columns:
                mask = df[feat].notna() & df[v1f].notna()
                if mask.sum() > 100:
                    r_val = np.corrcoef(df.loc[mask, feat], df.loc[mask, v1f])[0, 1]
                    corrs.append(f"{v1f}={r_val:+.3f}")
        print(f"  {feat:<20} → {', '.join(corrs)}")

    # ── FINAL VERDICT ──
    print(f"\n{'='*75}")
    print("FINAL VERDICT")
    print(f"{'='*75}")

    any_survivor = False
    for r in results[1:]:
        if r["brier"] < base_brier - 0.002:
            print(f"  PASS (ΔBrier < -0.002): {r['label']}")
            any_survivor = True

    if not any_survivor:
        print("  NO SURVIVORS at ΔBrier < -0.002 threshold")
        print("  → Neither feature×rung interactions nor Statcast process stats")
        print("    improve the V1_PRUNED+spread baseline via GBC(d3)")

    # Check if any individual interaction passed
    if individual:
        int_survivors = [x for x in individual if x[1] < -0.002]
        if int_survivors:
            print(f"\n  Individual interaction survivors (ΔBrier < -0.002):")
            for feat, db, da, _ in int_survivors:
                print(f"    {feat}×spread: ΔBrier={db:+.4f}")
            any_survivor = True

    if not any_survivor:
        print("\n  TIER 1 CLOSED: model is feature-saturated at 24 features")
        print("  Next edge lives in bet construction, info timing, or new markets — not more features")
    else:
        print("\n  SURVIVORS FOUND — proceed to validation gauntlet (multi-cut chrono + random)")

    return results


if __name__ == "__main__":
    results = run_gauntlet()
