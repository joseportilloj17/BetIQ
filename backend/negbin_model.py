"""
negbin_model.py — Thread 2 Step B: NegBin distribution model for MLB

Architecture:
  - GBM (sklearn GradientBoostingRegressor) predicts μ_home, μ_away per game
  - Global dispersion r = 3.26 (empirically fitted, stable across teams/months)
  - Simulate 10K margin draws per game: NegBin(r, p=r/(r+μ)) for each side
  - Ties resolved ±1 with 60/40 home bias (irrelevant for half-point rungs)
  - Derive P(cover spread X.5) for all rungs, P(total ≥ X.5) for all lines

Validation:
  1. Per-game μ must beat quadrant-average μ on per-cell cover accuracy
  2. Model must beat current ATS on backtest (parlay ROI + per-rung WR)
"""
from __future__ import annotations
import os, sys, pickle, math, json, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

DATA_DIR = os.path.join(_HERE, "..", "data")
SUBMODEL_DIR = os.path.join(DATA_DIR, "submodels")

# ── Global dispersion parameter (from poisson_negbin_precheck.py) ──
R_GLOBAL = 3.26

# ── Tie-resolution: ±1 with 60/40 home bias ──
TIE_HOME_P = 0.60

# ── Simulation settings ──
N_SIMS = 10_000

# ── Features for the μ prediction model ──
# Use the historical features from the feature matrix (not personal-bet features)
# These are the columns available at inference time for a given game
MU_FEATURE_COLS = [
    # Team quality (rolling)
    "h_pts_10g", "h_pts_allowed_10g", "h_win_pct_10g", "h_margin_10g",
    "a_pts_10g", "a_pts_allowed_10g", "a_win_pct_10g", "a_margin_10g",
    "h_pts_3g", "h_pts_allowed_3g", "h_win_pct_3g",
    "a_pts_3g", "a_pts_allowed_3g", "a_win_pct_3g",
    # Differentials
    "margin_diff_10g", "win_pct_diff_10g", "form_diff_3g",
    "pts_diff_10g", "def_diff_10g",
    # Schedule
    "h_rest_days", "a_rest_days", "rest_advantage",
    "h_is_btb", "a_is_btb", "back_to_back_road",
    "h_games_7d", "a_games_7d",
    # Momentum
    "h_days_since_win", "a_days_since_win", "days_since_win_diff",
    # Strength of schedule
    "h_opp_impl_prob_5g", "a_opp_impl_prob_5g", "opp_strength_diff",
    # Pi ratings
    "home_pi", "away_pi", "pi_diff", "pi_confidence",
    # Starting pitcher
    "h_sp_era", "a_sp_era", "sp_era_diff",
    "h_sp_whip", "a_sp_whip", "sp_whip_diff",
    "h_sp_k9", "a_sp_k9", "sp_k9_diff",
    "h_sp_hr9", "a_sp_hr9", "sp_hr9_diff",
    # Pitcher rolling windows
    "h_sp_era_3", "a_sp_era_3", "sp_era3_diff",
    "h_sp_era_5", "a_sp_era_5", "sp_era5_diff",
    "h_sp_whip_3", "a_sp_whip_3", "sp_whip3_diff",
    "h_sp_whip_5", "a_sp_whip_5", "sp_whip5_diff",
    "h_sp_hr9_3", "a_sp_hr9_3", "sp_hr9_3_diff",
    "h_sp_hr9_5", "a_sp_hr9_5", "sp_hr9_5_diff",
    # Batting quality
    "h_ops", "a_ops", "ops_diff",
    "h_slg", "a_slg",
    "h_k_pct", "a_k_pct",
    "h_bb_pct", "a_bb_pct",
    # Park + environment
    "h_park_factor",
    "temperature_f", "wind_speed_mph", "wind_direction_deg",
    "is_dome", "neutral_site",
    # Injury
    "h_star_out", "a_star_out", "star_out_diff",
    "h_stars_missing", "a_stars_missing",
    # Season phase
    "season_phase",
]


def _negbin_deviance_loss(y_true, y_pred):
    """NegBin deviance for monitoring (not used as GBM objective)."""
    y_pred = np.clip(y_pred, 0.1, 30)
    r = R_GLOBAL
    # NegBin log-likelihood: sum of log P(Y=y | r, p=r/(r+mu))
    ll = np.mean([stats.nbinom.logpmf(int(y), r, r / (r + mu))
                   for y, mu in zip(y_true, y_pred)])
    return -ll  # deviance = -LL


def _simulate_margins(mu_home, mu_away, n_sims=N_SIMS, seed=None):
    """Simulate game margins from NegBin(r, mu) for each side.

    Returns array of n_sims margins (home - away), ties resolved ±1 60/40.
    """
    rng = np.random.RandomState(seed)
    r = R_GLOBAL
    p_h = r / (r + mu_home)
    p_a = r / (r + mu_away)

    sim_h = rng.negative_binomial(r, p_h, n_sims)
    sim_a = rng.negative_binomial(r, p_a, n_sims)
    margins = sim_h.astype(np.int32) - sim_a.astype(np.int32)

    # Resolve ties: ±1 with 60/40 home bias
    tie_mask = margins == 0
    n_ties = tie_mask.sum()
    if n_ties > 0:
        coin = rng.random(n_ties)
        margins[tie_mask] = np.where(coin < TIE_HOME_P, 1, -1)

    return margins


def _simulate_totals(mu_home, mu_away, n_sims=N_SIMS, seed=None):
    """Simulate game totals from NegBin."""
    rng = np.random.RandomState(seed)
    r = R_GLOBAL
    p_h = r / (r + mu_home)
    p_a = r / (r + mu_away)
    sim_h = rng.negative_binomial(r, p_h, n_sims)
    sim_a = rng.negative_binomial(r, p_a, n_sims)
    return sim_h + sim_a


def derive_spread_probs(mu_home, mu_away, n_sims=N_SIMS, seed=42):
    """Derive cover probabilities for all spread rungs.

    Returns dict: {spread_value: cover_prob} where spread is from home's perspective.
    e.g. {+1.5: 0.63, -1.5: 0.42, +2.5: 0.72, ...}

    Cover +1.5 means home wins or loses by exactly 1. P(margin > -1.5).
    Cover -1.5 means home wins by 2+. P(margin > 1.5).
    """
    margins = _simulate_margins(mu_home, mu_away, n_sims, seed)
    result = {}
    for rung in [1.5, 2.5, 3.5, 4.5, 5.5, -1.5, -2.5, -3.5, -4.5, -5.5]:
        # Home covers spread X: margin > -X (home perspective)
        # So cover +1.5: margin > -1.5 = margin >= -1
        # Cover -1.5: margin > 1.5 = margin >= 2
        result[rung] = float(np.mean(margins > -rung))
    return result


def derive_total_probs(mu_home, mu_away, n_sims=N_SIMS, seed=42):
    """Derive over probabilities for all total lines.

    Returns dict: {total_line: over_prob}
    e.g. {7.5: 0.52, 8.5: 0.38, ...}
    """
    totals = _simulate_totals(mu_home, mu_away, n_sims, seed)
    result = {}
    for line in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
        result[line] = float(np.mean(totals > line))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_negbin_model(dry_run=False):
    """Train the NegBin μ prediction model.

    Two regressors: one for μ_home, one for μ_away.
    Target: actual runs scored (home_score, away_score).
    Loss: Poisson deviance (sklearn's 'poisson' loss — closest to NegBin
          for mean prediction; dispersion is handled by global r).

    Uses temporal 80/20 split (no shuffle).
    """
    from features import build_feature_matrix

    print("=" * 100)
    print("  NEGBIN MODEL: TRAINING")
    print("=" * 100)

    # ── Load data ──
    print("\n  Loading feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"  Raw matrix: {len(df)} games")

    # Filter to usable rows
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()  # drop aberrant
    print(f"  After score filter: {len(df)} games")

    # ── Feature selection ──
    # Use only columns that exist in the matrix
    available = [c for c in MU_FEATURE_COLS if c in df.columns]
    missing = [c for c in MU_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Missing features ({len(missing)}): {missing[:10]}...")
    print(f"  Using {len(available)} features")

    X = df[available].values.astype(float)
    y_home = df["home_score"].values.astype(float)
    y_away = df["away_score"].values.astype(float)
    dates = df["game_date"].values

    # ── Temporal split ──
    split_idx = int(len(df) * 0.80)
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_home_tr, y_home_te = y_home[:split_idx], y_home[split_idx:]
    y_away_tr, y_away_te = y_away[:split_idx], y_away[split_idx:]
    dates_te = dates[split_idx:]

    print(f"  Train: {len(X_tr)} games, Test: {len(X_te)} games")
    print(f"  Test period: {str(dates_te[0])[:10]} to {str(dates_te[-1])[:10]}")

    # ── Preprocessing ──
    # HistGradientBoostingRegressor handles NaN natively — no imputer needed
    # Still fit imputer/scaler for the prediction path (other code may need them)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(X_tr)
    scaler.fit(imputer.transform(X_tr))

    # For HistGBR, pass raw X (with NaN) — it handles missing internally
    X_tr_s = X_tr
    X_te_s = X_te

    # ── Train regressors ──
    # HistGradientBoostingRegressor with Poisson deviance loss
    # Poisson loss = best approximation for count data mean prediction
    # Dispersion is handled separately by global r
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

    print(f"\n  Training home μ model (HistGBR, Poisson loss)...")
    gbm_home = HistGradientBoostingRegressor(**gbm_params)
    gbm_home.fit(X_tr_s, y_home_tr)

    print(f"  Training away μ model (HistGBR, Poisson loss)...")
    gbm_away = HistGradientBoostingRegressor(**gbm_params)
    gbm_away.fit(X_tr_s, y_away_tr)

    # ── Evaluate on test set ──
    mu_home_pred = gbm_home.predict(X_te_s)
    mu_away_pred = gbm_away.predict(X_te_s)

    # Clip to reasonable range
    mu_home_pred = np.clip(mu_home_pred, 0.5, 15)
    mu_away_pred = np.clip(mu_away_pred, 0.5, 15)

    print(f"\n{'='*100}")
    print(f"  TEST SET EVALUATION")
    print(f"{'='*100}")

    # A. μ prediction quality
    mae_home = np.mean(np.abs(mu_home_pred - y_home_te))
    mae_away = np.mean(np.abs(mu_away_pred - y_away_te))
    rmse_home = np.sqrt(np.mean((mu_home_pred - y_home_te) ** 2))
    rmse_away = np.sqrt(np.mean((mu_away_pred - y_away_te) ** 2))

    # Baseline: quadrant-average μ
    avg_home = y_home_tr.mean()
    avg_away = y_away_tr.mean()
    mae_baseline_h = np.mean(np.abs(avg_home - y_home_te))
    mae_baseline_a = np.mean(np.abs(avg_away - y_away_te))

    print(f"\n  A. μ PREDICTION QUALITY:")
    print(f"    {'':>20} {'GBM MAE':>10} {'Baseline MAE':>14} {'Improvement':>12}")
    print(f"    {'Home μ':<20} {mae_home:>10.3f} {mae_baseline_h:>14.3f} {(1-mae_home/mae_baseline_h)*100:>+11.1f}%")
    print(f"    {'Away μ':<20} {mae_away:>10.3f} {mae_baseline_a:>14.3f} {(1-mae_away/mae_baseline_a)*100:>+11.1f}%")
    print(f"\n    RMSE: home={rmse_home:.3f}, away={rmse_away:.3f}")
    print(f"    Pred range: home=[{mu_home_pred.min():.2f}, {mu_home_pred.max():.2f}], "
          f"away=[{mu_away_pred.min():.2f}, {mu_away_pred.max():.2f}]")

    # B. NegBin log-likelihood: GBM μ vs constant μ
    r = R_GLOBAL

    def _negbin_ll(y, mu_arr):
        """Total NegBin log-likelihood."""
        total = 0
        for yi, mi in zip(y, mu_arr):
            mi = max(mi, 0.1)
            p = r / (r + mi)
            total += stats.nbinom.logpmf(int(yi), r, p)
        return total

    ll_gbm_h = _negbin_ll(y_home_te, mu_home_pred)
    ll_gbm_a = _negbin_ll(y_away_te, mu_away_pred)
    ll_base_h = _negbin_ll(y_home_te, np.full_like(y_home_te, avg_home))
    ll_base_a = _negbin_ll(y_away_te, np.full_like(y_away_te, avg_away))

    print(f"\n  B. NEGBIN LOG-LIKELIHOOD (higher = better):")
    print(f"    Home: GBM={ll_gbm_h:.1f}  Baseline={ll_base_h:.1f}  Δ={ll_gbm_h-ll_base_h:+.1f}")
    print(f"    Away: GBM={ll_gbm_a:.1f}  Baseline={ll_base_a:.1f}  Δ={ll_gbm_a-ll_base_a:+.1f}")

    # C. Spread cover accuracy: simulate per game, check actual cover
    print(f"\n  C. SPREAD COVER ACCURACY (per-game μ vs constant μ):")

    # Load h2h prices for fav/dog classification
    import sqlite3
    conn = sqlite3.connect(os.path.join(DATA_DIR, "bets.db"))
    fixtures = conn.execute("""
        SELECT id, home_team, away_team, bookmakers
        FROM fixtures WHERE sport_key='baseball_mlb' AND bookmakers IS NOT NULL
    """).fetchall()
    conn.close()

    fav_map = {}
    for fid, ht, at, bk_json in fixtures:
        if not bk_json:
            continue
        try:
            bks = json.loads(bk_json)
        except:
            continue
        home_price = None
        for pref in ["fanduel", "draftkings", "pinnacle", None]:
            for bk in bks:
                if pref is not None and bk.get("key") != pref:
                    continue
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    for oc in mkt.get("outcomes", []):
                        if oc.get("name", "").lower().strip() == ht.lower().strip():
                            home_price = float(oc.get("price", 0))
                            break
                    if home_price:
                        break
                if home_price:
                    break
            if home_price:
                break
        if home_price and home_price > 0:
            fav_map[(ht.lower().strip(), at.lower().strip())] = home_price < 2.0

    # Build per-game predictions and check cover accuracy
    test_df = df.iloc[split_idx:].copy()
    test_df["mu_home_gbm"] = mu_home_pred
    test_df["mu_away_gbm"] = mu_away_pred
    test_df["mu_home_base"] = avg_home
    test_df["mu_away_base"] = avg_away
    test_df["margin"] = test_df["home_score"] - test_df["away_score"]

    # Classify fav/dog
    test_df["home_is_fav"] = test_df.apply(
        lambda r: fav_map.get((str(r.get("home_team","")).lower().strip(),
                               str(r.get("away_team","")).lower().strip())), axis=1)

    # Per-rung cover accuracy
    rungs = [1.5, 2.5, 3.5, 4.5, -1.5, -2.5, -3.5]

    print(f"\n    {'Rung':>8} {'N':>5} {'GBM Brier':>10} {'Base Brier':>11} {'Δ':>8}")
    print(f"    {'─'*45}")

    for rung in rungs:
        # Actual cover: home covers spread X means margin > -X
        actual_cover = (test_df["margin"] > -rung).astype(float).values

        gbm_probs = []
        base_probs = []
        for i in range(len(test_df)):
            mh_g = mu_home_pred[i]
            ma_g = mu_away_pred[i]
            mh_b = avg_home
            ma_b = avg_away

            # Simulate cover prob
            margins_g = _simulate_margins(mh_g, ma_g, n_sims=2000, seed=i)
            margins_b = _simulate_margins(mh_b, ma_b, n_sims=2000, seed=i)

            gbm_probs.append(np.mean(margins_g > -rung))
            base_probs.append(np.mean(margins_b > -rung))

        gbm_probs = np.array(gbm_probs)
        base_probs = np.array(base_probs)

        brier_gbm = np.mean((gbm_probs - actual_cover) ** 2)
        brier_base = np.mean((base_probs - actual_cover) ** 2)

        print(f"    {rung:>+8.1f} {len(actual_cover):>5} {brier_gbm:>10.4f} {brier_base:>11.4f} "
              f"{brier_gbm-brier_base:>+7.4f}")

    # D. Per-cell accuracy (the make-or-break test)
    print(f"\n  D. PER-CELL ACCURACY (Fav/Dog × Home/Away):")

    cells = {
        "Fav@Home": test_df[test_df["home_is_fav"] == True],
        "Dog@Home": test_df[test_df["home_is_fav"] == False],
    }

    for cell_label, cell_df in cells.items():
        if len(cell_df) < 20:
            print(f"\n    {cell_label}: n={len(cell_df)} — too few")
            continue

        idx = cell_df.index - split_idx  # offset into test arrays
        idx = idx[idx >= 0]
        if len(idx) == 0:
            continue

        print(f"\n    {cell_label} (n={len(idx)}):")
        print(f"      {'Rung':>8} {'GBM Brier':>10} {'Base Brier':>11} {'Δ':>8} {'GBM wins?':>10}")
        print(f"      {'─'*50}")

        for rung in [1.5, 2.5, 3.5, -1.5]:
            actual = (test_df.iloc[idx]["margin"] > -rung).astype(float).values
            gp, bp = [], []
            for j, i in enumerate(idx):
                mg = _simulate_margins(mu_home_pred[i], mu_away_pred[i], 2000, seed=i)
                mb = _simulate_margins(avg_home, avg_away, 2000, seed=i)
                gp.append(np.mean(mg > -rung))
                bp.append(np.mean(mb > -rung))

            bg = np.mean((np.array(gp) - actual) ** 2)
            bb = np.mean((np.array(bp) - actual) ** 2)
            win = "YES" if bg < bb else "no"
            print(f"      {rung:>+8.1f} {bg:>10.4f} {bb:>11.4f} {bg-bb:>+7.4f} {win:>10}")

    # E. Feature importance (permutation-based, since HistGBR lacks .feature_importances_)
    from sklearn.inspection import permutation_importance
    print(f"\n  E. TOP 15 FEATURES (permutation importance, home model):")
    perm_h = permutation_importance(gbm_home, X_te_s, y_home_te,
                                     n_repeats=5, random_state=42,
                                     scoring="neg_mean_absolute_error")
    feat_imp = sorted(zip(available, perm_h.importances_mean), key=lambda x: -x[1])
    for feat, imp in feat_imp[:15]:
        print(f"    {feat:<30} {imp:.4f}")

    print(f"\n  TOP 15 FEATURES (away model):")
    perm_a = permutation_importance(gbm_away, X_te_s, y_away_te,
                                     n_repeats=5, random_state=42,
                                     scoring="neg_mean_absolute_error")
    feat_imp_a = sorted(zip(available, perm_a.importances_mean), key=lambda x: -x[1])
    for feat, imp in feat_imp_a[:15]:
        print(f"    {feat:<30} {imp:.4f}")

    # ── Save artifacts ──
    if not dry_run:
        os.makedirs(SUBMODEL_DIR, exist_ok=True)
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
        path = os.path.join(SUBMODEL_DIR, "mlb_negbin_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"\n  Saved to {path}")
    else:
        print(f"\n  [dry_run] Not saving.")

    return {
        "mae_home": mae_home, "mae_away": mae_away,
        "ll_gbm_h": ll_gbm_h, "ll_base_h": ll_base_h,
        "ll_gbm_a": ll_gbm_a, "ll_base_a": ll_base_a,
        "n_train": len(X_tr), "n_test": len(X_te),
        "features": available,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING + PREDICTION (for integration with mock_bets)
# ═══════════════════════════════════════════════════════════════════════════════

_CACHED_MODEL = None


def load_negbin_model():
    """Load the trained NegBin model artifacts."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    path = os.path.join(SUBMODEL_DIR, "mlb_negbin_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        _CACHED_MODEL = pickle.load(f)
    return _CACHED_MODEL


def predict_game_negbin(
    home_team: str,
    away_team: str,
    game_date: str | None = None,
    spread_value: float | None = None,
    total_line: float | None = None,
) -> dict | None:
    """Predict spread cover and total probabilities for a game.

    Returns:
        {
            "mu_home": float,
            "mu_away": float,
            "spread_probs": {1.5: 0.63, -1.5: 0.42, ...},
            "total_probs": {7.5: 0.52, ...},
            "cover_prob": float (for the specific spread_value if given),
            "over_prob": float (for the specific total_line if given),
        }
    """
    model = load_negbin_model()
    if model is None:
        return None

    from features import build_feature_matrix
    from ml_model import _get_feature_matrix

    matrix = _get_feature_matrix("MLB")
    if matrix.empty:
        return None

    ht = home_team.lower().strip()
    at = away_team.lower().strip()
    mask = (
        matrix["home_team"].str.lower().str.strip() == ht
    ) & (
        matrix["away_team"].str.lower().str.strip() == at
    )
    if game_date:
        mask = mask & (matrix["game_date"].astype(str).str[:10] == str(game_date)[:10])

    matched = matrix[mask]
    if matched.empty:
        return None

    row = matched.sort_values("game_date", ascending=False).iloc[0]

    feat_cols = model["feature_cols"]
    X = np.array([[row[c] if c in row.index and not pd.isna(row[c]) else np.nan
                    for c in feat_cols]], dtype=float)

    # HistGBR handles NaN natively — pass raw features
    mu_home = float(np.clip(model["gbm_home"].predict(X)[0], 0.5, 15))
    mu_away = float(np.clip(model["gbm_away"].predict(X)[0], 0.5, 15))

    spread_probs = derive_spread_probs(mu_home, mu_away)
    total_probs = derive_total_probs(mu_home, mu_away)

    result = {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "spread_probs": spread_probs,
        "total_probs": total_probs,
    }

    if spread_value is not None:
        # Find the closest half-point rung
        closest = min(spread_probs.keys(), key=lambda x: abs(x - spread_value))
        result["cover_prob"] = spread_probs[closest]

    if total_line is not None:
        closest = min(total_probs.keys(), key=lambda x: abs(x - total_line))
        result["over_prob"] = total_probs[closest]

    return result


if __name__ == "__main__":
    train_negbin_model(dry_run=False)
