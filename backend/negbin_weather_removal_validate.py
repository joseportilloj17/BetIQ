"""
negbin_weather_removal_validate.py — Validate NegBin weather feature removal

Protocol: append-only ordering, 10 seeds × 5 folds (TimeSeriesSplit).
Compare 84-feature (no weather) vs 88-feature (with weather) model.
Report mean ΔBrier ± seed-SD against noise floor (0.000846 at k=10).

Then retrain production model via negbin_retrain.retrain().
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

R_GLOBAL = 3.26
N_SEEDS = 10
N_FOLDS = 5
NOISE_FLOOR_K10 = 0.000846  # 2σ/√k at k=10

WEATHER_COLS = ["temperature_f", "wind_speed_mph", "wind_direction_deg", "is_dome"]

# 84-feature list (current MU_FEATURE_COLS, no weather)
from negbin_model import MU_FEATURE_COLS as COLS_NO_WEATHER

# 88-feature list = current + weather APPENDED at end (append-only protocol)
COLS_WITH_WEATHER = list(COLS_NO_WEATHER) + WEATHER_COLS


def _brier_margin(y_actual_margin, y_pred_margin, sigma=4.217, rung=1.5):
    """Spread-cover Brier score for a given rung."""
    from scipy.stats import norm
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


def run_multiseed_validation():
    from features import build_feature_matrix

    print("=" * 110)
    print("  WEATHER REMOVAL — Multi-Seed Validation (10 seeds × 5 folds)")
    print("=" * 110)

    # ── Load data ──
    df = build_feature_matrix("MLB")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"\n  Games: {len(df)}")
    print(f"  Date range: {str(df['game_date'].iloc[0])[:10]} → {str(df['game_date'].iloc[-1])[:10]}")

    # ── Check feature availability ──
    avail_no_wx = [c for c in COLS_NO_WEATHER if c in df.columns]
    avail_with_wx = [c for c in COLS_WITH_WEATHER if c in df.columns]
    print(f"  No-weather features: {len(avail_no_wx)}")
    print(f"  With-weather features: {len(avail_with_wx)}")
    wx_present = [c for c in WEATHER_COLS if c in df.columns]
    print(f"  Weather columns present: {wx_present}")

    y_home = df["home_score"].values.astype(float)
    y_away = df["away_score"].values.astype(float)

    X_no_wx = df[avail_no_wx].values.astype(float)
    X_with_wx = df[avail_with_wx].values.astype(float)

    # ── Multi-seed × multi-fold ──
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    fold_indices = list(tscv.split(X_no_wx))

    results = []  # list of {seed, fold, brier_no_wx, brier_with_wx, delta}

    for seed in range(N_SEEDS):
        for fold_idx, (train_ix, test_ix) in enumerate(fold_indices):
            gbm_params = dict(
                loss="poisson", max_iter=300, max_depth=4,
                learning_rate=0.04, min_samples_leaf=10,
                max_features=0.8, random_state=seed * 1000 + fold_idx,
                early_stopping=False,
            )

            # ── No-weather model (84 features) ──
            gbm_h_nw = HistGradientBoostingRegressor(**gbm_params)
            gbm_a_nw = HistGradientBoostingRegressor(**gbm_params)
            gbm_h_nw.fit(X_no_wx[train_ix], y_home[train_ix])
            gbm_a_nw.fit(X_no_wx[train_ix], y_away[train_ix])

            mu_h_nw = np.clip(gbm_h_nw.predict(X_no_wx[test_ix]), 0.5, 15)
            mu_a_nw = np.clip(gbm_a_nw.predict(X_no_wx[test_ix]), 0.5, 15)
            margin_nw = mu_h_nw - mu_a_nw

            # ── With-weather model (88 features, weather APPENDED) ──
            gbm_h_wx = HistGradientBoostingRegressor(**gbm_params)
            gbm_a_wx = HistGradientBoostingRegressor(**gbm_params)
            gbm_h_wx.fit(X_with_wx[train_ix], y_home[train_ix])
            gbm_a_wx.fit(X_with_wx[train_ix], y_away[train_ix])

            mu_h_wx = np.clip(gbm_h_wx.predict(X_with_wx[test_ix]), 0.5, 15)
            mu_a_wx = np.clip(gbm_a_wx.predict(X_with_wx[test_ix]), 0.5, 15)
            margin_wx = mu_h_wx - mu_a_wx

            # ── Brier at +1.5 rung ──
            margin_actual = y_home[test_ix] - y_away[test_ix]
            brier_nw = _brier_margin(margin_actual, margin_nw)
            brier_wx = _brier_margin(margin_actual, margin_wx)
            delta = brier_nw - brier_wx  # negative = no-weather better

            results.append({
                "seed": seed, "fold": fold_idx,
                "brier_no_wx": brier_nw, "brier_with_wx": brier_wx,
                "delta": delta,
            })

        # Progress
        seed_deltas = [r["delta"] for r in results if r["seed"] == seed]
        print(f"  Seed {seed:2d}: mean ΔBrier = {np.mean(seed_deltas):+.6f} "
              f"(folds: {', '.join(f'{d:+.4f}' for d in seed_deltas)})")

    # ── Aggregate ──
    rdf = pd.DataFrame(results)

    print(f"\n{'='*110}")
    print(f"  AGGREGATE RESULTS ({N_SEEDS} seeds × {N_FOLDS} folds = {len(rdf)} trials)")
    print(f"{'='*110}")

    mean_delta = rdf["delta"].mean()
    std_delta = rdf["delta"].std()
    se_delta = std_delta / np.sqrt(len(rdf))

    print(f"\n  Mean ΔBrier (no_wx − with_wx): {mean_delta:+.6f}")
    print(f"  SD: {std_delta:.6f}")
    print(f"  SE: {se_delta:.6f}")
    print(f"  |Mean ΔBrier|: {abs(mean_delta):.6f}")
    print(f"  Noise floor (k={N_SEEDS}): {NOISE_FLOOR_K10:.6f}")

    if mean_delta < 0:
        print(f"\n  → REMOVING weather IMPROVES Brier by {abs(mean_delta):.6f}")
    else:
        print(f"\n  → REMOVING weather WORSENS Brier by {mean_delta:.6f}")

    if abs(mean_delta) > NOISE_FLOOR_K10:
        print(f"  → Effect ABOVE noise floor — detectable")
    else:
        print(f"  → Effect BELOW noise floor — indistinguishable from zero")

    # Per-fold averages
    print(f"\n  Per-fold mean ΔBrier (averaged over {N_SEEDS} seeds):")
    for fold in range(N_FOLDS):
        fold_data = rdf[rdf["fold"] == fold]
        print(f"    Fold {fold}: {fold_data['delta'].mean():+.6f} ± {fold_data['delta'].std():.6f}")

    # Discrimination check (Spearman ρ on last seed, full holdout-like split)
    print(f"\n  Discrimination check (seed=0, last fold):")
    last_train, last_test = fold_indices[-1]
    gbm_params_0 = dict(
        loss="poisson", max_iter=300, max_depth=4,
        learning_rate=0.04, min_samples_leaf=10,
        max_features=0.8, random_state=42, early_stopping=False,
    )
    gbm_h = HistGradientBoostingRegressor(**gbm_params_0)
    gbm_a = HistGradientBoostingRegressor(**gbm_params_0)
    gbm_h.fit(X_no_wx[last_train], y_home[last_train])
    gbm_a.fit(X_no_wx[last_train], y_away[last_train])
    m_pred = np.clip(gbm_h.predict(X_no_wx[last_test]), 0.5, 15) - \
             np.clip(gbm_a.predict(X_no_wx[last_test]), 0.5, 15)
    m_actual = y_home[last_test] - y_away[last_test]
    rho, p_rho = sp_stats.spearmanr(m_pred, m_actual)
    print(f"    ρ = {rho:.4f} (p = {p_rho:.4f})")

    return rdf


def retrain_production():
    """Retrain the production model with weather removed."""
    from negbin_retrain import retrain
    print(f"\n\n{'='*110}")
    print(f"  PRODUCTION RETRAIN (84 features, no weather)")
    print(f"{'='*110}\n")
    return retrain()


if __name__ == "__main__":
    rdf = run_multiseed_validation()
    result = retrain_production()

    print(f"\n{'='*110}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*110}")
    print(f"  Retrain saved: {result['saved']}")
    print(f"  Bias: {result['bias_new']:+.4f}")
    print(f"  Discrimination ρ: {result['rho_new']:.4f}")
    print(f"  MAE home: {result['mae_home_new']:.4f} (old: {result['mae_home_old']:.4f})")
    print(f"  MAE away: {result['mae_away_new']:.4f} (old: {result['mae_away_old']:.4f})")
    print(f"  LL: {result['ll_new']:.1f} (old: {result['ll_old']:.1f})")
