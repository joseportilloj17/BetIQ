"""
ats_feature_audit.py — Leave-one-out audit of ATS distributional model features.

For each of 21 features, run 10-seed × 5-fold (TimeSeriesSplit) comparison:
  full model (21 features) vs model with that feature removed (20 features).

Protocol: append-only ordering, 10-seed averaging, ΔBrier vs 2σ/√k noise floor.
Positive ΔBrier(full − removed) = feature HURTS (removing it improves Brier).
Negative ΔBrier = feature HELPS.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

N_SEEDS = 10
N_FOLDS = 5
NOISE_FLOOR_K10 = 0.000846  # 2σ/√k at k=10 (from measured null)
SIGMA = 4.217  # ATS distributional sigma

# Production ATS features (from pkl)
ATS_FEATURES = [
    "sp_era_diff", "ops_diff", "opp_strength_diff", "h_sp_era",
    "sp_whip_diff", "sp_era5_diff", "a_ops", "a_sp_era",
    "h_sp_era_5", "h_bb_pct", "a_opp_impl_prob_5g", "h_opp_impl_prob_5g",
    "h_sp_era_3", "a_bb_pct", "a_sp_era_5", "a_sp_whip_3",
    "a_margin_10g", "margin_diff_10g", "def_diff_10g", "a_slg",
    "sp_kbb_diff",
]

# Production ATS model params (from pkl)
GBM_BASE_PARAMS = dict(
    loss="squared_error",
    max_iter=200,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=50,
    max_features=1.0,
    n_iter_no_change=10,
    validation_fraction=0.1,
)


def _brier_margin(y_actual_margin, y_pred_margin, sigma=SIGMA, rung=1.5):
    """Spread-cover Brier score at +1.5 rung."""
    brier = 0.0
    n = 0
    for i in range(len(y_actual_margin)):
        actual_cover = 1.0 if y_actual_margin[i] > -rung else (
            0.0 if y_actual_margin[i] < -rung else np.nan)
        if np.isnan(actual_cover):
            continue
        p_cover = float(1.0 - norm.cdf((-rung - y_pred_margin[i]) / sigma))
        brier += (p_cover - actual_cover) ** 2
        n += 1
    return brier / n if n > 0 else np.nan


def run_audit():
    from features import build_feature_matrix

    print("=" * 110)
    print("  ATS DISTRIBUTIONAL — Leave-One-Out Feature Audit (10 seeds × 5 folds)")
    print("=" * 110)

    # ── Load data ──
    df = build_feature_matrix("MLB")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    df["margin"] = df["home_score"] - df["away_score"]
    print(f"\n  Games: {len(df)}")

    avail = [c for c in ATS_FEATURES if c in df.columns]
    missing = [c for c in ATS_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Available: {len(avail)}/{len(ATS_FEATURES)} features")

    y = df["margin"].values.astype(float)

    # Precompute fold indices
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    X_full = df[avail].values.astype(float)
    fold_indices = list(tscv.split(X_full))

    # ── First: baseline (full model) Brier across all seeds × folds ──
    print(f"\n  Computing BASELINE (full {len(avail)}-feature model)...")
    baseline_briers = {}  # (seed, fold) → brier
    t0 = time.time()

    for seed in range(N_SEEDS):
        for fold_idx, (train_ix, test_ix) in enumerate(fold_indices):
            params = {**GBM_BASE_PARAMS, "random_state": seed * 1000 + fold_idx}
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_full[train_ix])
            X_te = imp.transform(X_full[test_ix])

            model = HistGradientBoostingRegressor(**params)
            model.fit(X_tr, y[train_ix])
            mu_pred = model.predict(X_te)

            brier = _brier_margin(y[test_ix], mu_pred)
            baseline_briers[(seed, fold_idx)] = brier

        seed_mean = np.mean([baseline_briers[(seed, f)] for f in range(N_FOLDS)])
        print(f"    Seed {seed:2d}: mean Brier = {seed_mean:.6f}")

    baseline_elapsed = time.time() - t0
    overall_baseline = np.mean(list(baseline_briers.values()))
    print(f"  Baseline mean Brier: {overall_baseline:.6f} ({baseline_elapsed:.0f}s)")

    # ── Leave-one-out for each feature ──
    audit_results = []

    for feat_idx, feat_name in enumerate(avail):
        print(f"\n  [{feat_idx+1}/{len(avail)}] Testing removal of: {feat_name}")
        reduced_cols = [c for c in avail if c != feat_name]
        X_reduced = df[reduced_cols].values.astype(float)

        deltas = []  # per (seed, fold) ΔBrier = baseline − reduced (positive = removal helps)
        reduced_briers = {}

        for seed in range(N_SEEDS):
            for fold_idx, (train_ix, test_ix) in enumerate(fold_indices):
                params = {**GBM_BASE_PARAMS, "random_state": seed * 1000 + fold_idx}
                imp = SimpleImputer(strategy="median")
                X_tr = imp.fit_transform(X_reduced[train_ix])
                X_te = imp.transform(X_reduced[test_ix])

                model = HistGradientBoostingRegressor(**params)
                model.fit(X_tr, y[train_ix])
                mu_pred = model.predict(X_te)

                brier_reduced = _brier_margin(y[test_ix], mu_pred)
                reduced_briers[(seed, fold_idx)] = brier_reduced
                delta = baseline_briers[(seed, fold_idx)] - brier_reduced
                deltas.append(delta)

        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        # Positive mean_delta = baseline worse than reduced = feature HURTS
        if mean_delta > 0:
            verdict = "HARMFUL" if abs(mean_delta) > NOISE_FLOOR_K10 else "noise (slight hurt)"
        else:
            verdict = "HELPFUL" if abs(mean_delta) > NOISE_FLOOR_K10 else "noise (slight help)"

        audit_results.append({
            "feature": feat_name,
            "mean_delta": mean_delta,  # positive = removal improves
            "std_delta": std_delta,
            "abs_delta": abs(mean_delta),
            "above_floor": abs(mean_delta) > NOISE_FLOOR_K10,
            "verdict": verdict,
        })

        print(f"    ΔBrier = {mean_delta:+.6f} ± {std_delta:.6f}  |  "
              f"|Δ|={abs(mean_delta):.6f} vs floor={NOISE_FLOOR_K10:.6f}  |  {verdict}")

    # ── Summary table ──
    print(f"\n\n{'='*110}")
    print(f"  LEAVE-ONE-OUT AUDIT SUMMARY")
    print(f"{'='*110}")
    print(f"\n  {'Feature':<25} {'ΔBrier':>10} {'±SD':>10} {'|Δ|':>10} {'Floor':>8} {'Verdict':<20}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*20}")

    # Sort by mean_delta descending (most harmful first)
    sorted_results = sorted(audit_results, key=lambda x: x["mean_delta"], reverse=True)
    for r in sorted_results:
        flag = "***" if r["above_floor"] and r["mean_delta"] > 0 else ""
        print(f"  {r['feature']:<25} {r['mean_delta']:>+10.6f} {r['std_delta']:>10.6f} "
              f"{r['abs_delta']:>10.6f} {NOISE_FLOOR_K10:>8.6f} {r['verdict']:<20} {flag}")

    # Identify features to remove
    harmful = [r for r in audit_results if r["above_floor"] and r["mean_delta"] > 0]
    noise_hurt = [r for r in audit_results if not r["above_floor"] and r["mean_delta"] > 0]
    helpful = [r for r in audit_results if r["mean_delta"] < 0]

    print(f"\n  HARMFUL (above floor, removal improves): {len(harmful)}")
    for r in harmful:
        print(f"    - {r['feature']}: {r['mean_delta']:+.6f}")

    print(f"\n  Noise (slight hurt, below floor): {len(noise_hurt)}")
    for r in noise_hurt:
        print(f"    - {r['feature']}: {r['mean_delta']:+.6f}")

    print(f"\n  Helpful (removal worsens): {len(helpful)}")
    for r in sorted(helpful, key=lambda x: x["mean_delta"]):
        flag = " (ABOVE FLOOR)" if r["above_floor"] else ""
        print(f"    - {r['feature']}: {r['mean_delta']:+.6f}{flag}")

    return audit_results


if __name__ == "__main__":
    results = run_audit()
