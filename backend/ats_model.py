"""
ats_model.py — Train and evaluate ATS / O-U models for BetIQ.

Uses features.py to build the feature matrix from historical.db, then fits
a GradientBoostingClassifier with isotonic calibration (probability output).

Usage
-----
    python ats_model.py --sport NFL
    python ats_model.py --sport NFL --target ou
    python ats_model.py --sport NFL --include-lines   # adds close_spread etc. to X
    python ats_model.py --sport NBA --save            # saves model to data/

Output
------
    - Temporal train/test split stats
    - Accuracy, ROC-AUC, Brier score on test set
    - Top N feature importances (ranked)
    - Optionally saves model to data/ats_model_{SPORT}.pkl
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from features import (
    TARGET_ATS,
    TARGET_OU,
    build_feature_matrix,
    get_model_ready,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Model hyper-parameters ─────────────────────────────────────────────────────
GBC_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)

TRAIN_FRAC = 0.80   # chronological split: first 80 % train, last 20 % test
TOP_N      = 10     # feature importances to display


# ══════════════════════════════════════════════════════════════════════════════
# Core train / evaluate
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    sport: str,
    target: str = TARGET_ATS,
    include_lines: bool = False,
    save: bool = False,
) -> dict:
    """
    Train a calibrated GBC on historical data for *sport* and return metrics.

    Parameters
    ----------
    sport        : e.g. "NFL", "NBA"
    target       : TARGET_ATS or TARGET_OU
    include_lines: whether to add close_spread / close_total to X
    save         : if True, persist model to data/ats_model_{sport}.pkl

    Returns
    -------
    dict with keys: sport, target, n_train, n_test, accuracy, roc_auc,
                    brier, feature_importances (pd.Series)
    """
    # ── 1. Build feature matrix ─────────────────────────────────────────────
    log.info("Building feature matrix: sport=%s  target=%s", sport, target)
    X, y, feature_cols = get_model_ready(
        sport=sport,
        target=target,
        include_lines=include_lines,
        dropna=True,
    )

    n_total = len(y)
    if n_total < 50:
        raise ValueError(
            f"{sport}: only {n_total} labelled rows after dropna — "
            "not enough data to train a model."
        )

    # ── 2. Temporal train / test split (no shuffle) ─────────────────────────
    split_idx = int(n_total * TRAIN_FRAC)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log.info(
        "Split: %d train  /  %d test  (%.0f%% / %.0f%%)",
        len(y_train), len(y_test), 100 * TRAIN_FRAC, 100 * (1 - TRAIN_FRAC),
    )

    # ── 3. Train ────────────────────────────────────────────────────────────
    base_clf = GradientBoostingClassifier(**GBC_PARAMS)
    # Isotonic calibration on the test fold; cv=5 uses cross-val on train set
    model = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    # ── 4. Evaluate ─────────────────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_prob)
    brier    = brier_score_loss(y_test, y_prob)

    # ── 5. Feature importances (average over calibrated estimators) ──────────
    importances = _aggregate_importances(model, feature_cols)

    # ── 6. Print report ──────────────────────────────────────────────────────
    _print_report(
        sport=sport,
        target=target,
        n_train=len(y_train),
        n_test=len(y_test),
        acc=acc,
        roc_auc=roc_auc,
        brier=brier,
        importances=importances,
    )

    # ── 7. Optionally save ───────────────────────────────────────────────────
    if save:
        target_tag = "ats" if target == TARGET_ATS else "ou"
        model_path = DATA_DIR / f"{target_tag}_model_{sport}.pkl"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as fh:
            pickle.dump({"model": model, "feature_cols": feature_cols}, fh)
        log.info("Model saved → %s", model_path)

    return {
        "sport":               sport,
        "target":              target,
        "n_train":             len(y_train),
        "n_test":              len(y_test),
        "accuracy":            acc,
        "roc_auc":             roc_auc,
        "brier":               brier,
        "feature_importances": importances,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_importances(
    calibrated_model: CalibratedClassifierCV,
    feature_cols: list[str],
) -> pd.Series:
    """
    CalibratedClassifierCV wraps multiple base estimators (one per CV fold).
    Average feature_importances_ across all of them.
    """
    all_imp = []
    for estimator in calibrated_model.calibrated_classifiers_:
        # estimator.estimator is the fitted GBC
        base = estimator.estimator
        if hasattr(base, "feature_importances_"):
            all_imp.append(base.feature_importances_)

    if not all_imp:
        return pd.Series(dtype=float)

    mean_imp = np.mean(all_imp, axis=0)
    return (
        pd.Series(mean_imp, index=feature_cols)
        .sort_values(ascending=False)
    )


def _print_report(
    sport: str,
    target: str,
    n_train: int,
    n_test: int,
    acc: float,
    roc_auc: float,
    brier: float,
    importances: pd.Series,
) -> None:
    target_label = "ATS (covered_spread)" if target == TARGET_ATS else "O-U (over)"
    line = "─" * 54

    print(f"\n{line}")
    print(f"  {sport}  |  {target_label}")
    print(line)
    print(f"  Train rows : {n_train:,}")
    print(f"  Test rows  : {n_test:,}")
    print(line)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  Brier score: {brier:.4f}  (lower = better; 0.25 = random)")
    print(line)
    print(f"  Top {TOP_N} feature importances:")
    for i, (feat, imp) in enumerate(importances.head(TOP_N).items(), 1):
        bar = "█" * int(imp * 200)
        print(f"    {i:2d}. {feat:<28s}  {imp:.4f}  {bar}")
    print(line)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BetIQ ATS/O-U model")
    p.add_argument(
        "--sport",
        default="NFL",
        help="Sport key (NFL, NBA, MLB, NHL, CFB, etc.) — default: NFL",
    )
    p.add_argument(
        "--target",
        choices=["ats", "ou"],
        default="ats",
        help="Target variable: ats (covered_spread) or ou (total_result)",
    )
    p.add_argument(
        "--include-lines",
        action="store_true",
        help="Include close_spread / close_total in X (post-line prediction only)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save trained model to data/ats_model_{SPORT}.pkl",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help=f"Number of top features to display (default {TOP_N})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    TOP_N = args.top_n

    target = TARGET_ATS if args.target == "ats" else TARGET_OU
    train_and_evaluate(
        sport=args.sport.upper(),
        target=target,
        include_lines=args.include_lines,
        save=args.save,
    )
