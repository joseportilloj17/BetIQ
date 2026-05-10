"""
soccer_model.py — Phase 3: Train soccer sub-models.

Standalone script. Trains per-market classifiers using TimeSeriesSplit
(same methodology as MLB ATS model) to prevent data leakage.

Markets:
  total_goals — Over/Under form-based model (119 training rows)
  moneyline   — ML + Double Chance form-based model (97 training rows)
  (shots_on_target deferred — needs player-level stats)

Accepts AUC >= 0.54 (same gate as MLB model).
Saves to: data/submodels/soccer_{market}_clf.pkl (+ scaler + imputer + feat_cols)

Usage:
    cd /Users/joseportillo/Downloads/BetIQ
    python backend/analysis/soccer_model.py              # train all + report
    python backend/analysis/soccer_model.py --market total_goals
    python backend/analysis/soccer_model.py --market moneyline
    python backend/analysis/soccer_model.py --eval-only  # report saved model AUC
"""
from __future__ import annotations

import argparse
import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SUBMODELS_DIR = os.path.join(DATA_DIR, "submodels")
os.makedirs(SUBMODELS_DIR, exist_ok=True)

AUC_GATE = 0.54

# ─────────────────────────────────────────────────────────────────────────────
# Feature selection by market
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_GOALS_FEATURES = [
    "line",
    "direction",
    "home_gf_avg_5",
    "home_ga_avg_5",
    "home_gf_avg_10",
    "home_ga_avg_10",
    "home_over25_r10",
    "away_gf_avg_5",
    "away_ga_avg_5",
    "away_gf_avg_10",
    "away_ga_avg_10",
    "away_over25_r10",
    "combined_goals_exp_5",
    "avg_over25_rate",
    "home_form_wr_5",
    "away_form_wr_5",
]

MONEYLINE_FEATURES = [
    "is_home",
    "is_double_chance",
    "pick_wr_5",
    "pick_wr_10",
    "pick_unbeaten_5",
    "pick_gf_avg_5",
    "pick_ga_avg_5",
    "pick_gf_avg_10",
    "pick_ga_avg_10",
    "pick_home_wr_5",
    "pick_away_wr_5",
    "opp_wr_5",
    "opp_gf_avg_5",
    "opp_ga_avg_5",
    "opp_unbeaten_5",
    "wr_diff_5",
    "league_home_wr",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(market: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"soccer_{market}_features.csv")
    if not os.path.exists(path):
        print(f"[soccer_model] Feature CSV not found: {path}")
        print("  Run: python backend/analysis/soccer_features.py --export")
        return None
    df = pd.read_csv(path)
    # Sort by date to preserve temporal order
    if "bet_date" in df.columns:
        df = df.sort_values("bet_date").reset_index(drop=True)
    return df


def _build_Xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix and label vector, keeping only available columns."""
    avail = [c for c in feature_cols if c in df.columns]
    X = df[avail].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y, avail


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_market(
    market: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Train a GradientBoostingClassifier for one market using TimeSeriesSplit.

    Returns result dict with AUC, model, and save status.
    """
    if len(df) < 50:
        return {
            "market": market, "n": len(df),
            "auc": None, "saved": False,
            "reason": f"Insufficient data: {len(df)} < 50 minimum",
        }

    X, y, used_cols = _build_Xy(df, feature_cols)
    n = len(y)

    if verbose:
        win_rate = y.mean()
        print(f"\n[{market}] Training on {n} rows | "
              f"WIN rate: {win_rate:.1%} | Features: {len(used_cols)}")
        print(f"  Columns: {used_cols}")

    # ── TimeSeriesSplit cross-validation ─────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    # Try multiple classifiers; keep best AUC
    candidates = [
        ("GBM", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
        ("RF", RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42,
        )),
        ("LR", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ]

    best_auc = 0.0
    best_name = None
    best_clf  = None
    cv_details: dict[str, list[float]] = {}

    for clf_name, clf in candidates:
        fold_aucs = []
        for train_idx, test_idx in tscv.split(X_imp):
            X_tr, X_te = X_imp[train_idx], X_imp[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue  # Skip folds without both classes

            try:
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_te)[:, 1]
                fold_aucs.append(roc_auc_score(y_te, probs))
            except Exception:
                continue

        if fold_aucs:
            mean_auc = float(np.mean(fold_aucs))
            cv_details[clf_name] = fold_aucs
            if verbose:
                print(f"  {clf_name:4s}: fold AUCs={[f'{a:.4f}' for a in fold_aucs]} "
                      f"→ mean={mean_auc:.4f}")
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_name = clf_name
                best_clf  = type(clf)(**clf.get_params())  # fresh instance

    if best_clf is None:
        return {
            "market": market, "n": n, "auc": None, "saved": False,
            "reason": "All CV folds failed (likely class imbalance in small dataset)",
        }

    if verbose:
        print(f"  Best: {best_name} AUC={best_auc:.4f}")

    # ── Decision gate ─────────────────────────────────────────────────────────
    if best_auc >= AUC_GATE:
        decision = f"WORTH BUILDING: soccer {market} sub-model justified (AUC >= {AUC_GATE})"
    elif best_auc >= 0.50:
        decision = f"WAIT: soccer {market} model won't help yet (0.50 <= AUC < {AUC_GATE})"
    else:
        decision = f"NO SIGNAL: soccer {market} model is worse than random (AUC < 0.50)"

    if verbose:
        print(f"\n  Decision: {decision}")

    # ── Refit on all data ─────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_final = scaler.fit_transform(X_imp)
    best_clf.fit(X_final, y)

    # ── Feature importance (GBM / RF only) ────────────────────────────────────
    if hasattr(best_clf, "feature_importances_"):
        importances = sorted(
            zip(used_cols, best_clf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        if verbose:
            print("\n  Feature importances:")
            for feat, imp in importances[:8]:
                print(f"    {feat:30s} {imp:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save = best_auc >= AUC_GATE
    if save:
        prefix = os.path.join(SUBMODELS_DIR, f"soccer_{market}")
        pickle.dump(best_clf, open(f"{prefix}_clf.pkl",       "wb"))
        pickle.dump(imputer,  open(f"{prefix}_imputer.pkl",   "wb"))
        pickle.dump(scaler,   open(f"{prefix}_scaler.pkl",    "wb"))
        pickle.dump(used_cols, open(f"{prefix}_feat_cols.pkl", "wb"))
        if verbose:
            print(f"\n  ✅ Saved to data/submodels/soccer_{market}_*.pkl")
    else:
        if verbose:
            print(f"\n  ⏳ Not saved — AUC {best_auc:.4f} < gate {AUC_GATE}")

    return {
        "market":   market,
        "n":        n,
        "auc":      round(best_auc, 4),
        "saved":    save,
        "clf_name": best_name,
        "decision": decision,
        "cv_details": cv_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation of saved models
# ─────────────────────────────────────────────────────────────────────────────

def _eval_saved(market: str) -> None:
    prefix = os.path.join(SUBMODELS_DIR, f"soccer_{market}")
    clf_path = f"{prefix}_clf.pkl"
    if not os.path.exists(clf_path):
        print(f"  [eval] soccer_{market}: no saved model")
        return

    clf       = pickle.load(open(clf_path,                    "rb"))
    imputer   = pickle.load(open(f"{prefix}_imputer.pkl",     "rb"))
    scaler    = pickle.load(open(f"{prefix}_scaler.pkl",      "rb"))
    feat_cols = pickle.load(open(f"{prefix}_feat_cols.pkl",   "rb"))

    print(f"\n[eval] soccer_{market}:")
    print(f"  Model type: {type(clf).__name__}")
    print(f"  Features: {feat_cols}")

    market_map = {"total_goals": TOTAL_GOALS_FEATURES, "moneyline": MONEYLINE_FEATURES}
    feat_config = market_map.get(market, feat_cols)
    df = _load_csv(market)
    if df is None or df.empty:
        print("  No data to evaluate against")
        return

    X, y, _ = _build_Xy(df, feat_config)
    X_imp = imputer.transform(X)
    X_sc  = scaler.transform(X_imp)
    probs = clf.predict_proba(X_sc)[:, 1]
    auc   = roc_auc_score(y, probs)
    print(f"  Train-set AUC (whole dataset): {auc:.4f}")
    print(f"  N={len(y)}, WIN%={y.mean():.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train_all(markets: list[str] | None = None, verbose: bool = True) -> list[dict]:
    if markets is None:
        markets = ["total_goals", "moneyline"]

    results = []
    for market in markets:
        feat_map = {
            "total_goals": TOTAL_GOALS_FEATURES,
            "moneyline":   MONEYLINE_FEATURES,
        }
        if market not in feat_map:
            print(f"[soccer_model] Unknown market: {market}")
            continue

        df = _load_csv(market)
        if df is None:
            continue

        result = _train_market(market, df, feat_map[market], verbose=verbose)
        results.append(result)

    # Summary
    print("\n━━━ Soccer Model Training Summary ━━━")
    for r in results:
        saved = "✅ SAVED" if r.get("saved") else "⏳ NOT SAVED"
        auc   = f"AUC={r['auc']:.4f}" if r.get("auc") else "AUC=N/A"
        print(f"  soccer_{r['market']:15s}  n={r.get('n',0):4d}  "
              f"{auc}  {saved}")
        if r.get("decision"):
            print(f"    → {r['decision']}")

    saved_any = any(r.get("saved") for r in results)
    if saved_any:
        print("\n  Next step: run integration (Phase 4)")
        print("  python backend/analysis/soccer_model.py --eval-only")
    else:
        print("\n  Models below AUC gate. Collect more bet data before retraining.")
        print("  Expected improvement path:")
        print("    +data from ongoing bets → retrain every 50 new resolved legs")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market",    type=str, help="total_goals | moneyline (default: all)")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate saved models only")
    parser.add_argument("--quiet",     action="store_true", help="Suppress per-fold output")
    args = parser.parse_args()

    if args.eval_only:
        for m in ["total_goals", "moneyline"]:
            _eval_saved(m)
    else:
        markets = [args.market] if args.market else None
        train_all(markets=markets, verbose=not args.quiet)
