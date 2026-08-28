"""
negbin_retrain.py — Retrain NegBin mu model including 2026 data

Problem: model trained 80/20 on 11,223 games → train ends mid-2025.
         2026 data has +0.544 systematic bias (under-predicts home margin).

Fix: retrain with all available data through 2026, small forward holdout
     for validation. Back up old model, compare pre/post metrics.

Validation checklist:
  1. Global bias on holdout ≈ 0 (currently +0.544)
  2. MAE doesn't degrade (currently 2.37/2.50)
  3. NegBin log-likelihood doesn't degrade
  4. Spread cover discrimination preserved
"""
from __future__ import annotations
import os, sys, shutil, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
DATA_DIR = os.path.join(_HERE, "..", "data")
SUBMODEL_DIR = os.path.join(DATA_DIR, "submodels")
MODEL_PATH = os.path.join(SUBMODEL_DIR, "mlb_negbin_model.pkl")
BACKUP_PATH = os.path.join(SUBMODEL_DIR, "mlb_negbin_model_prev.pkl")

R_GLOBAL = 3.26
TIE_HOME_P = 0.60
N_SIMS = 10_000


def retrain():
    from features import build_feature_matrix
    from negbin_model import MU_FEATURE_COLS, _simulate_margins

    print("=" * 110)
    print("  NEGBIN MODEL RETRAIN — Monthly refresh with all available data")
    print("=" * 110)

    # ── Back up current model ──
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"\n  Backed up current model to {BACKUP_PATH}")

    # ── Load old model for comparison ──
    with open(BACKUP_PATH, "rb") as f:
        old_art = pickle.load(f)
    print(f"  Old model: train_size={old_art['train_size']}, test_size={old_art['test_size']}")
    print(f"  Old model MAE: home={old_art['mae_home']:.4f}, away={old_art['mae_away']:.4f}")

    # ── Load feature matrix ──
    print("\n  Loading feature matrix...")
    df = build_feature_matrix("MLB")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {str(df['game_date'].iloc[0])[:10]} → {str(df['game_date'].iloc[-1])[:10]}")

    # ── Feature prep ──
    available = [c for c in MU_FEATURE_COLS if c in df.columns]
    missing = [c for c in MU_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Missing features ({len(missing)}): {missing[:5]}...")
    print(f"  Using {len(available)} features")

    X = df[available].values.astype(float)
    y_home = df["home_score"].values.astype(float)
    y_away = df["away_score"].values.astype(float)
    dates = df["game_date"].values

    # ── Split: train on 93%, holdout last 7% (~2 months) for validation ──
    # This ensures 2026 data is in the TRAINING set while keeping
    # the most recent games for honest validation.
    holdout_frac = 0.07
    split_idx = int(len(df) * (1.0 - holdout_frac))
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_home_tr, y_home_te = y_home[:split_idx], y_home[split_idx:]
    y_away_tr, y_away_te = y_away[:split_idx], y_away[split_idx:]
    dates_tr, dates_te = dates[:split_idx], dates[split_idx:]

    print(f"\n  Train: {len(X_tr)} games ({str(dates_tr[0])[:10]} → {str(dates_tr[-1])[:10]})")
    print(f"  Holdout: {len(X_te)} games ({str(dates_te[0])[:10]} → {str(dates_te[-1])[:10]})")

    # ── Preprocessing (for compatibility) ──
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(X_tr)
    scaler.fit(imputer.transform(X_tr))

    # ── Train ──
    gbm_params = dict(
        loss="poisson",
        max_iter=300,
        max_depth=4,
        learning_rate=0.04,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=42,
        early_stopping=False,
    )

    print(f"\n  Training home μ model...")
    gbm_home = HistGradientBoostingRegressor(**gbm_params)
    gbm_home.fit(X_tr, y_home_tr)

    print(f"  Training away μ model...")
    gbm_away = HistGradientBoostingRegressor(**gbm_params)
    gbm_away.fit(X_tr, y_away_tr)

    # ── Evaluate on holdout ──
    mu_home_new = np.clip(gbm_home.predict(X_te), 0.5, 15)
    mu_away_new = np.clip(gbm_away.predict(X_te), 0.5, 15)

    mae_home = np.mean(np.abs(mu_home_new - y_home_te))
    mae_away = np.mean(np.abs(mu_away_new - y_away_te))
    margin_pred = mu_home_new - mu_away_new
    margin_actual = y_home_te - y_away_te
    rmse_margin = np.sqrt(np.mean((margin_pred - margin_actual) ** 2))

    # Residual bias
    resid = margin_actual - margin_pred
    mean_resid = resid.mean()
    t_stat, p_val = sp_stats.ttest_1samp(resid, 0)

    print(f"\n{'='*110}")
    print(f"  HOLDOUT RESULTS (new model)")
    print(f"{'='*110}")
    print(f"  MAE home: {mae_home:.4f}  (old: {old_art['mae_home']:.4f})")
    print(f"  MAE away: {mae_away:.4f}  (old: {old_art['mae_away']:.4f})")
    print(f"  RMSE margin: {rmse_margin:.4f}")
    print(f"  Residual bias: {mean_resid:+.4f}  (t={t_stat:+.3f}, p={p_val:.4f})")
    print(f"  → {'BIAS CLOSED ✓' if abs(mean_resid) < 0.2 and p_val > 0.05 else 'BIAS REMAINS ✗'}")

    # ── Compare old model on same holdout ──
    old_feat_cols = old_art["feature_cols"]
    old_avail = [c for c in old_feat_cols if c in df.columns]
    X_te_old = df.iloc[split_idx:][old_avail].values.astype(float)
    mu_home_old = np.clip(old_art["gbm_home"].predict(X_te_old), 0.5, 15)
    mu_away_old = np.clip(old_art["gbm_away"].predict(X_te_old), 0.5, 15)
    margin_pred_old = mu_home_old - mu_away_old
    resid_old = margin_actual - margin_pred_old
    mae_home_old = np.mean(np.abs(mu_home_old - y_home_te))
    mae_away_old = np.mean(np.abs(mu_away_old - y_away_te))

    print(f"\n  OLD MODEL on same holdout:")
    print(f"  MAE home: {mae_home_old:.4f}")
    print(f"  MAE away: {mae_away_old:.4f}")
    print(f"  Residual bias: {resid_old.mean():+.4f}")

    # ── NegBin log-likelihood comparison ──
    r = R_GLOBAL
    def _ll(y, mu_arr):
        total = 0
        for yi, mi in zip(y, mu_arr):
            mi = max(mi, 0.1)
            p = r / (r + mi)
            total += sp_stats.nbinom.logpmf(int(yi), r, p)
        return total

    ll_new_h = _ll(y_home_te, mu_home_new)
    ll_new_a = _ll(y_away_te, mu_away_new)
    ll_old_h = _ll(y_home_te, mu_home_old)
    ll_old_a = _ll(y_away_te, mu_away_old)

    print(f"\n  NegBin LL (higher = better):")
    print(f"    New: home={ll_new_h:.1f}, away={ll_new_a:.1f}  (total={ll_new_h+ll_new_a:.1f})")
    print(f"    Old: home={ll_old_h:.1f}, away={ll_old_a:.1f}  (total={ll_old_h+ll_old_a:.1f})")
    print(f"    Δ = {(ll_new_h+ll_new_a)-(ll_old_h+ll_old_a):+.1f}")

    # ── Spread cover Brier ──
    from scipy.stats import norm
    sigma = 4.217

    print(f"\n  Spread Cover Brier (holdout):")
    for rung in [1.5, 2.5, 3.5]:
        brier_new = 0
        brier_old = 0
        n = 0
        for i in range(len(margin_actual)):
            actual_cover = 1.0 if margin_actual[i] > -rung else (
                0.0 if margin_actual[i] < -rung else np.nan)
            if np.isnan(actual_cover):
                continue
            p_new = float(1.0 - norm.cdf((-rung - margin_pred[i]) / sigma))
            p_old = float(1.0 - norm.cdf((-rung - margin_pred_old[i]) / sigma))
            brier_new += (p_new - actual_cover) ** 2
            brier_old += (p_old - actual_cover) ** 2
            n += 1
        if n > 0:
            print(f"    +{rung}: new={brier_new/n:.6f}  old={brier_old/n:.6f}  "
                  f"Δ={(brier_new-brier_old)/n:+.6f}  "
                  f"({'NEW better' if brier_new < brier_old else 'OLD better'})")

    # ── Discrimination (Spearman ρ: predicted margin → actual margin) ──
    rho_new, p_rho_new = sp_stats.spearmanr(margin_pred, margin_actual)
    rho_old, p_rho_old = sp_stats.spearmanr(margin_pred_old, margin_actual)
    print(f"\n  Discrimination (Spearman ρ):")
    print(f"    New: ρ={rho_new:.4f}  (p={p_rho_new:.4f})")
    print(f"    Old: ρ={rho_old:.4f}  (p={p_rho_old:.4f})")
    print(f"    Δρ = {rho_new - rho_old:+.4f}")

    # ── Per-month bias check (the critical test) ──
    holdout_df = df.iloc[split_idx:].copy()
    holdout_df["resid_new"] = margin_actual - margin_pred
    holdout_df["resid_old"] = margin_actual - margin_pred_old
    holdout_df["month"] = pd.to_datetime(holdout_df["game_date"]).dt.month
    holdout_df["year"] = pd.to_datetime(holdout_df["game_date"]).dt.year

    print(f"\n  Per-month bias comparison:")
    print(f"  {'Month':>8} {'N':>5} {'Old bias':>10} {'New bias':>10}")
    print(f"  {'─'*8} {'─'*5} {'─'*10} {'─'*10}")
    for (yr, mo), grp in holdout_df.groupby(["year", "month"]):
        print(f"  {yr}-{mo:02d} {len(grp):>5} {grp['resid_old'].mean():>+10.4f} "
              f"{grp['resid_new'].mean():>+10.4f}")

    # ── Save new model ──
    print(f"\n{'='*110}")
    print(f"  SAVE DECISION")
    print(f"{'='*110}")

    # Safety checks
    bias_ok = abs(mean_resid) < 0.3
    mae_ok = mae_home <= mae_home_old + 0.05 and mae_away <= mae_away_old + 0.05
    ll_ok = (ll_new_h + ll_new_a) >= (ll_old_h + ll_old_a) - 10
    discrim_ok = rho_new > 0.15  # discrimination must remain meaningful

    if bias_ok and mae_ok and ll_ok and discrim_ok:
        artifacts = {
            "gbm_home": gbm_home,
            "gbm_away": gbm_away,
            "imputer": imputer,
            "scaler": scaler,
            "feature_cols": available,
            "r_global": R_GLOBAL,
            "n_sims": N_SIMS,
            "tie_home_p": TIE_HOME_P,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "mae_home": mae_home,
            "mae_away": mae_away,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"  ✓ All checks passed. Saved new model to {MODEL_PATH}")
        print(f"    Backup at: {BACKUP_PATH}")
        print(f"    Train size: {len(X_tr)} (was {old_art['train_size']})")
        print(f"    Bias closed: {mean_resid:+.4f} (was {resid_old.mean():+.4f})")
    else:
        print(f"  ✗ Safety check FAILED — NOT saving.")
        if not bias_ok:
            print(f"    bias_ok=False: |mean_resid|={abs(mean_resid):.4f} > 0.3")
        if not mae_ok:
            print(f"    mae_ok=False: new MAE worse than old + 0.05")
        if not ll_ok:
            print(f"    ll_ok=False: LL dropped by > 10")
        if not discrim_ok:
            print(f"    discrim_ok=False: ρ={rho_new:.4f} < 0.15")

    return {
        "bias_new": mean_resid,
        "bias_old": resid_old.mean(),
        "mae_home_new": mae_home,
        "mae_away_new": mae_away,
        "mae_home_old": mae_home_old,
        "mae_away_old": mae_away_old,
        "ll_new": ll_new_h + ll_new_a,
        "ll_old": ll_old_h + ll_old_a,
        "rho_new": rho_new,
        "rho_old": rho_old,
        "saved": bias_ok and mae_ok and ll_ok and discrim_ok,
    }


if __name__ == "__main__":
    retrain()
