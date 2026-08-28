#!/usr/bin/env python3
"""
Staging-only retrain: test whether devigging implied_prob improves the model.

Approach:
- Load the same training data as production (personal + historical + mock legs)
- Replace vigged implied_prob with devigged version using proportional method
- Retrain with same hyperparameters (gradient_boost, 300 trees)
- Compare AUC, Brier, accuracy to current v18 (run #86)
- Also re-derive log_odds and odds_bracket from devigged values
- Save artifacts to staging directory, never touch production

Devigging method:
  For two-way markets (moneyline, spread): average overround is ~107.9%
    devigged = raw_implied / 1.079
  For totals: similar overround, same approach
  For player props: slightly higher vig (~112%), use 1.12
  This is the proportional (multiplicative) method — simple but effective
  since we don't have the opposing side's odds in the training data.
"""

import os
import sys
import math
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ── Load training data the same way production does ─────────────────────────
from database import get_db, SessionLocal
from ml_model import (
    _build_personal_df, _load_historical_df, _build_mock_leg_df,
    add_clv_features, FEATURE_COLS, Bet, _build_clf,
)

db = SessionLocal()
now = datetime.utcnow()

print("=" * 60)
print("STAGING DEVIG RETRAIN — implied_prob devigging test")
print("=" * 60)

# ── 1. Build training data (same as production) ────────────────────────────
bets = db.query(Bet).filter(
    Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
).order_by(Bet.time_placed).all()
personal_df = _build_personal_df(bets, cutoff=now)
historical_df = _load_historical_df()
mock_leg_df = _build_mock_leg_df(db)

personal_df = add_clv_features(personal_df)
if not historical_df.empty:
    historical_df = add_clv_features(historical_df)

frames = [personal_df]
if not historical_df.empty:
    frames.append(historical_df[personal_df.columns])
if not mock_leg_df.empty:
    frames.append(mock_leg_df[personal_df.columns])
combined = pd.concat(frames, ignore_index=True)
combined = combined.sort_values("_date").reset_index(drop=True)

print(f"\nTotal training rows: {len(combined)}")
print(f"  Personal: {len(personal_df)}")
print(f"  Historical: {len(historical_df)}")
print(f"  Mock legs: {len(mock_leg_df)}")

# ── 2. Create vigged (baseline) and devigged variants ──────────────────────

# Overround constants (measured from actual data)
OVERROUND_2WAY = 1.079   # moneyline, spread — pairs sum to ~107.9%
OVERROUND_TOTAL = 1.079  # totals — similar
OVERROUND_PROP = 1.12    # player props — slightly higher

def devig_implied_prob(ip, market_pct=None):
    """Proportional devigging: divide by overround factor."""
    if ip is None or np.isnan(ip):
        return ip
    # Use 2-way overround as default (most of our data is ML/spread)
    overround = OVERROUND_2WAY
    # Could refine by market type, but implied_prob doesn't carry market info
    # in the personal/historical rows. The difference is small (1.079 vs 1.12).
    return ip / overround

# Make two copies of the combined df
vigged_df = combined.copy()
devigged_df = combined.copy()

# Apply devigging to the devigged copy
if "implied_prob" in devigged_df.columns:
    orig_ip = devigged_df["implied_prob"].copy()
    devigged_df["implied_prob"] = devigged_df["implied_prob"].apply(devig_implied_prob)

    # Also update log_odds (derived from odds = 1/implied_prob)
    # If implied_prob was vigged odds-derived, devigged odds = 1/devigged_ip
    # log_odds = log(1/devigged_ip) = -log(devigged_ip)
    devigged_ip = devigged_df["implied_prob"]
    valid = devigged_ip.notna() & (devigged_ip > 0) & (devigged_ip < 1)
    devigged_df.loc[valid, "log_odds"] = -np.log(devigged_df.loc[valid, "implied_prob"])
    devigged_df.loc[valid, "odds"] = 1.0 / devigged_df.loc[valid, "implied_prob"]

    # Update odds_bracket based on devigged decimal odds
    devigged_odds = devigged_df["odds"]
    devigged_df.loc[valid, "odds_bracket"] = np.select(
        [devigged_odds < 2, devigged_odds < 5, devigged_odds < 15, devigged_odds < 50],
        [0, 1, 2, 3],
        default=4
    )[valid]

    # Report the shift
    mask = orig_ip.notna() & devigged_ip.notna()
    shift = (orig_ip[mask] - devigged_ip[mask]).mean()
    print(f"\n--- Implied Prob Shift ---")
    print(f"  Avg vigged implied_prob:   {orig_ip[mask].mean():.4f} ({orig_ip[mask].mean()*100:.1f}%)")
    print(f"  Avg devigged implied_prob: {devigged_ip[mask].mean():.4f} ({devigged_ip[mask].mean()*100:.1f}%)")
    print(f"  Avg shift:                 {shift:.4f} ({shift*100:.1f}pp)")

# ── 3. Train both models with identical hyperparams ────────────────────────

def train_variant(df, label, feature_cols):
    """Train and evaluate a model variant. Returns metrics dict."""
    _goal_cols = list(feature_cols)
    for c in _goal_cols:
        if c not in df.columns:
            df[c] = np.nan

    split_idx = int(len(df) * 0.80)
    train_portion = df.iloc[:split_idx]

    nan_mask = train_portion[_goal_cols].isna().all()
    dropped = [c for c in _goal_cols if nan_mask.get(c, False)]
    train_cols = [c for c in _goal_cols if not nan_mask.get(c, False)]

    X_all = df[train_cols].values.astype(float)
    y_all = df["_label"].values.astype(float)
    sw_all = df["_weight"].values.astype(float)

    X_tr, X_te = X_all[:split_idx], X_all[split_idx:]
    y_tr, y_te = y_all[:split_idx], y_all[split_idx:]
    sw_tr = sw_all[:split_idx]

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_te_s = scaler.transform(X_te_imp)

    clf = _build_clf("gradient_boost")
    clf.fit(X_tr_s, y_tr, sample_weight=sw_tr)

    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    acc = float(accuracy_score(y_te, y_pred))
    auc = float(roc_auc_score(y_te, y_prob)) if len(set(y_te)) > 1 else 0.5
    brier = float(brier_score_loss(y_te, y_prob))

    # Feature importances
    imps = {}
    if hasattr(clf, "calibrated_classifiers_"):
        for cc in clf.calibrated_classifiers_:
            base = cc.estimator
            if hasattr(base, "feature_importances_"):
                for i, c in enumerate(train_cols):
                    imps[c] = imps.get(c, 0) + base.feature_importances_[i]
        n_cc = len(clf.calibrated_classifiers_)
        imps = {k: v / n_cc for k, v in imps.items()}

    top5 = sorted(imps.items(), key=lambda x: -x[1])[:5]

    print(f"\n--- {label} ---")
    print(f"  N train: {split_idx}  N test: {len(X_te)}")
    print(f"  Features: {len(train_cols)} (dropped {len(dropped)} all-NaN)")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Brier:    {brier:.4f}")
    print(f"  Top features: {', '.join(f'{n}={v:.4f}' for n, v in top5)}")

    return {
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "n_train": split_idx,
        "n_test": len(X_te),
        "n_features": len(train_cols),
        "top_features": dict(top5),
        "clf": clf,
        "imputer": imputer,
        "scaler": scaler,
        "train_cols": train_cols,
    }

print("\n" + "=" * 60)
print("Training both variants...")
print("=" * 60)

vigged_result = train_variant(vigged_df, "v18-vigged (baseline reproduction)", FEATURE_COLS)
devigged_result = train_variant(devigged_df, "v18-devig (devigged implied_prob)", FEATURE_COLS)

# ── 4. Head-to-head comparison ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("HEAD-TO-HEAD COMPARISON")
print("=" * 60)

metrics = ["accuracy", "auc", "brier"]
labels = {"accuracy": "Accuracy", "auc": "AUC (↑ better)", "brier": "Brier (↓ better)"}
for m in metrics:
    v = vigged_result[m]
    d = devigged_result[m]
    diff = d - v
    better = "DEVIG" if (diff > 0 and m != "brier") or (diff < 0 and m == "brier") else "VIGGED"
    sym = "+" if diff > 0 else ""
    print(f"  {labels[m]:20s}  Vigged={v:.4f}  Devig={d:.4f}  Δ={sym}{diff:.4f}  ← {better}")

# ── 5. Save devigged model to staging ──────────────────────────────────────

staging_dir = os.path.join("data", "staging_devig")
os.makedirs(staging_dir, exist_ok=True)

with open(os.path.join(staging_dir, "model.pkl"), "wb") as f:
    pickle.dump(devigged_result["clf"], f)
with open(os.path.join(staging_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(devigged_result["scaler"], f)
with open(os.path.join(staging_dir, "imputer.pkl"), "wb") as f:
    pickle.dump(devigged_result["imputer"], f)
with open(os.path.join(staging_dir, "active_feature_cols.pkl"), "wb") as f:
    pickle.dump(devigged_result["train_cols"], f)
with open(os.path.join(staging_dir, "comparison.json"), "w") as f:
    json.dump({
        "vigged": {k: v for k, v in vigged_result.items() if k not in ("clf", "imputer", "scaler", "train_cols")},
        "devigged": {k: v for k, v in devigged_result.items() if k not in ("clf", "imputer", "scaler", "train_cols")},
        "timestamp": now.isoformat(),
    }, f, indent=2, default=str)

print(f"\nStaging artifacts saved to {staging_dir}/")

# ── 6. Edge classification analysis (period 3: 5/27+) ─────────────────────

print("\n" + "=" * 60)
print("EDGE CLASSIFICATION IMPACT (Period 3: 5/27+)")
print("=" * 60)

import sqlite3
con = sqlite3.connect("data/bets.db")

# Get settled mock legs from period 3 with both vigged and devigged implied probs
rows = con.execute("""
    SELECT
        mbl.open_odds,
        mbl.predicted_win_prob,
        mbl.leg_result,
        mbl.description
    FROM mock_bet_legs mbl
    JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
    WHERE mbl.leg_result IN ('WIN', 'LOSS')
      AND COALESCE(mbl.excluded_from_metrics, 0) = 0
      AND mbl.open_odds IS NOT NULL
      AND mb.game_date >= '2026-05-27'
""").fetchall()
con.close()

if rows:
    vigged_high = 0; vigged_high_win = 0
    devig_high = 0; devig_high_win = 0
    total = len(rows)

    for open_odds_am, pred_wp, result, desc in rows:
        # Convert American odds to decimal
        if open_odds_am is None:
            continue
        oo = float(open_odds_am)
        if oo > 0:
            dec = oo / 100 + 1
        elif oo < 0:
            dec = 100 / abs(oo) + 1
        else:
            dec = 1.909

        vigged_ip = (1.0 / dec) * 100  # vigged implied prob (%)
        devigged_ip = vigged_ip / 1.079  # devigged (%)

        model_wp = float(pred_wp or 50)
        is_win = result == "WIN"

        # Vigged edge classification
        if model_wp > vigged_ip:
            vigged_high += 1
            if is_win:
                vigged_high_win += 1

        # Devigged edge classification
        if model_wp > devigged_ip:
            devig_high += 1
            if is_win:
                devig_high_win += 1

    vigged_wr = (vigged_high_win / vigged_high * 100) if vigged_high else 0
    devig_wr = (devig_high_win / devig_high * 100) if devig_high else 0

    print(f"  Total settled legs (5/27+): {total}")
    print(f"")
    print(f"  Vigged 'model high':   {vigged_high}/{total} legs ({vigged_high/total*100:.0f}%)")
    print(f"    Win rate: {vigged_wr:.1f}%")
    print(f"")
    print(f"  Devigged 'model high': {devig_high}/{total} legs ({devig_high/total*100:.0f}%)")
    print(f"    Win rate: {devig_wr:.1f}%")
    print(f"")
    print(f"  ΔN model_high: +{devig_high - vigged_high} legs")
    print(f"  Δ WR:          {devig_wr - vigged_wr:+.1f}pp")

    # The key insight: if devig makes nearly all legs "model high",
    # the signal is diluted — WR should drop toward base rate
    if devig_high > total * 0.9:
        print(f"\n  ⚠ Devigging makes {devig_high/total*100:.0f}% of legs 'model high'")
        print(f"    → Signal is diluted; model was calibrated against vigged values")

print("\n" + "=" * 60)
print("DONE — Staging artifacts in data/staging_devig/")
print("=" * 60)
