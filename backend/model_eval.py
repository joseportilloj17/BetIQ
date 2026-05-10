"""
model_eval.py — Calibration check + profit simulation for BetIQ.

Reconstructs the exact same train/test split used by ml_model.train(),
runs calibration diagnostics, and simulates a flat-bet backtest on the
held-out test set.

Usage
-----
    python model_eval.py                        # saves plots to data/
    python model_eval.py --plot-dir /tmp/
    python model_eval.py --threshold 0.55       # P&L threshold (default 0.55)
    python model_eval.py --no-plots             # text output only

Output files
------------
    data/calibration_curve.png    — reliability diagram
    data/profit_simulation.png    — cumulative P&L curve on test set
    data/eval_results.json        — machine-readable summary
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
DATA_DIR    = BACKEND_DIR.parent / "data"
sys.path.insert(0, str(BACKEND_DIR))

# ── Imports (after path setup) ─────────────────────────────────────────────────
from database import SessionLocal, Bet
from ml_model import (
    FEATURE_COLS, WEIGHT_CONFIG,
    _build_personal_df, _load_historical_df,
    load_model, add_clv_features,
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild_dataset() -> pd.DataFrame:
    """Rebuild the full training+test dataset identical to train()."""
    now = datetime.utcnow()
    db  = SessionLocal()
    try:
        bets = (
            db.query(Bet)
            .filter(Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]))
            .order_by(Bet.time_placed)
            .all()
        )
        personal_df = _build_personal_df(bets, cutoff=now)
    finally:
        db.close()

    historical_df = _load_historical_df()

    personal_df   = add_clv_features(personal_df)
    if not historical_df.empty:
        historical_df = add_clv_features(historical_df)

    frames = [personal_df]
    if not historical_df.empty:
        frames.append(historical_df[personal_df.columns])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("_date").reset_index(drop=True)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Calibration check
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> dict:
    """
    Compute reliability diagram and calibration statistics.

    ECE (Expected Calibration Error): weighted mean |predicted - actual|.
    MCE (Maximum Calibration Error):  worst single bin.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    mce = float(np.max(np.abs(frac_pos - mean_pred)))

    print("\n── Calibration Reliability Table ──────────────────────────────")
    print(f"  {'Predicted':>12}  {'Actual':>8}  {'Delta':>8}  {'Bars'}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*20}")
    for p, a in zip(mean_pred, frac_pos):
        delta = a - p
        bar   = "█" * int(abs(delta) * 100)
        sign  = "+" if delta >= 0 else ""
        print(f"  {p:>11.1%}  {a:>7.1%}  {sign}{delta:>6.1%}  {bar}")
    print(f"  ECE: {ece:.4f}  |  MCE: {mce:.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  Brier:    {brier_score_loss(y_true, y_prob):.4f}")
    print(f"  Pred range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    print()

    if save_path or show:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Reliability diagram
            ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1.5)
            ax1.plot(mean_pred, frac_pos, "o-", color="#2196F3", lw=2,
                     markersize=8, label="Model")
            ax1.fill_between(mean_pred, mean_pred, frac_pos,
                             alpha=0.15, color="#2196F3")
            ax1.set_xlabel("Mean predicted probability", fontsize=12)
            ax1.set_ylabel("Fraction of positives (actual win rate)", fontsize=12)
            ax1.set_title(f"Calibration Curve  |  ECE={ece:.3f}  MCE={mce:.3f}",
                          fontsize=13)
            ax1.legend(fontsize=10)
            ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
            ax1.grid(alpha=0.3)

            # Histogram of predictions
            ax2.hist(y_prob, bins=40, color="#4CAF50", edgecolor="white",
                     linewidth=0.5)
            ax2.axvline(0.5, color="k", linestyle="--", lw=1.5, label="50%")
            ax2.axvline(0.55, color="#FF5722", linestyle="--", lw=1.5,
                        label="55% threshold")
            ax2.set_xlabel("Predicted probability", fontsize=12)
            ax2.set_ylabel("Count", fontsize=12)
            ax2.set_title("Distribution of Predicted Probabilities", fontsize=13)
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"  Calibration plot saved → {save_path}")
            if show:
                plt.show()
            plt.close()
        except ImportError:
            print("  matplotlib not available — skipping plot")

    return {
        "ece":   ece,
        "mce":   mce,
        "auc":   float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "pred_min": float(y_prob.min()),
        "pred_max": float(y_prob.max()),
        "pred_mean": float(y_prob.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Profit simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_profit_sim(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    dates:     pd.Series,
    weights:   np.ndarray,
    threshold: float = 0.55,
    flat_bet:  float = 10.0,
    juice:     float = -110,     # standard ATS juice (positive EV threshold)
    save_path: str | None = None,
    show: bool = False,
) -> dict:
    """
    Flat-bet backtest on test rows where predicted prob > threshold.

    Payout convention:
      Historical rows (weight = 1×, ATS bets): standard -110 juice.
        Win: +$9.09  |  Loss: -$10.00
      Personal rows (weight > 1×): use implied odds from probability.
        Win: +flat_bet × (1/implied_prob - 1)  |  Loss: -flat_bet

    Returns summary dict with ROI, Sharpe, and per-bet P&L series.
    """
    # juice → decimal payout on win for ATS bets
    juice_dec   = (100 / abs(juice)) if juice < 0 else (juice / 100)
    hist_payout = flat_bet * juice_dec   # e.g. 10 × (100/110) ≈ 9.09

    mask = y_prob > threshold
    n_total   = int(mask.sum())
    n_pos     = int(((y_true == 1) & mask).sum())
    is_hist   = weights == WEIGHT_CONFIG["historical_base"]

    pnl_list  = []
    dates_list = []

    for i in np.where(mask)[0]:
        is_win = y_true[i] == 1
        is_h   = is_hist[i]

        if is_h:
            pnl = hist_payout if is_win else -flat_bet
        else:
            payout = flat_bet * (1 / max(y_prob[i], 0.01) - 1)
            pnl    = payout if is_win else -flat_bet

        pnl_list.append(float(pnl))
        dates_list.append(dates.iloc[i] if hasattr(dates, "iloc") else dates[i])

    pnl_arr   = np.array(pnl_list)
    cum_pnl   = np.cumsum(pnl_arr)
    total_pnl = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0
    total_inv = float(n_total * flat_bet)
    roi       = total_pnl / total_inv if total_inv > 0 else 0.0
    win_rate  = n_pos / n_total if n_total > 0 else 0.0

    # Sharpe: mean P&L / std P&L per bet
    sharpe = (float(pnl_arr.mean()) / float(pnl_arr.std(ddof=1))
               if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0.0)

    # Break-even win rate at -110
    breakeven = abs(juice) / (abs(juice) + 100)

    print(f"── Profit Simulation (threshold={threshold:.0%}, flat_bet=${flat_bet:.0f}) ─────")
    print(f"  Bets placed   : {n_total}")
    print(f"  Win rate      : {win_rate:.1%}  (break-even at {breakeven:.1%} for -110)")
    print(f"  Total P&L     : ${total_pnl:+.2f}  (invested ${total_inv:.0f})")
    print(f"  ROI           : {roi:+.1%}")
    print(f"  Sharpe (bet)  : {sharpe:+.3f}")
    if n_total == 0:
        print("  No bets passed the threshold — try lowering --threshold.")
        print()
        return {"n_bets": 0, "win_rate": 0, "total_pnl": 0, "roi": 0}
    print()

    if save_path or show:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                            gridspec_kw={"height_ratios": [3, 1]})

            xs = range(len(cum_pnl))
            ax1.plot(xs, cum_pnl, lw=2, color="#2196F3", label="Cumulative P&L")
            ax1.fill_between(xs, 0, cum_pnl,
                             where=cum_pnl >= 0, color="#4CAF50", alpha=0.2,
                             label="Profit zone")
            ax1.fill_between(xs, 0, cum_pnl,
                             where=cum_pnl < 0, color="#F44336", alpha=0.2,
                             label="Loss zone")
            ax1.axhline(0, color="k", lw=1.2, linestyle="--")
            ax1.set_ylabel("Cumulative P&L ($)", fontsize=12)
            ax1.set_title(
                f"Profit Simulation — prob>{threshold:.0%} | "
                f"{n_total} bets | ROI={roi:+.1%} | Sharpe={sharpe:+.3f}",
                fontsize=13,
            )
            ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

            # Per-bet bars
            colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnl_arr]
            ax2.bar(xs, pnl_arr, color=colors, width=0.8, alpha=0.7)
            ax2.axhline(0, color="k", lw=0.8)
            ax2.set_xlabel("Bet # (chronological)", fontsize=12)
            ax2.set_ylabel("P&L ($)", fontsize=12)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"  Profit simulation plot saved → {save_path}")
            if show:
                plt.show()
            plt.close()
        except ImportError:
            print("  matplotlib not available — skipping plot")

    return {
        "n_bets":    n_total,
        "win_rate":  round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "roi":       round(roi, 4),
        "sharpe":    round(sharpe, 4),
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(threshold: float = 0.55, plot_dir: str | None = None, no_plots: bool = False):
    plot_dir = plot_dir or str(DATA_DIR)
    os.makedirs(plot_dir, exist_ok=True)

    cal_path  = os.path.join(plot_dir, "calibration_curve.png")  if not no_plots else None
    prof_path = os.path.join(plot_dir, "profit_simulation.png")  if not no_plots else None

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("Loading trained model…")
    clf, scaler, imputer = load_model()

    # ── 2. Rebuild dataset ─────────────────────────────────────────────────
    print("Rebuilding dataset…")
    combined = _rebuild_dataset()
    n = len(combined)
    print(f"  {n} total rows  ({int((combined['_weight']>1).sum())} personal "
          f"+ {int((combined['_weight']==1).sum())} historical)")

    # ── 3. Reproduce 80/20 temporal split ─────────────────────────────────
    split = int(n * 0.80)
    test  = combined.iloc[split:].reset_index(drop=True)

    X_te  = test[FEATURE_COLS].values.astype(float)
    y_te  = test["_label"].values.astype(float)
    sw_te = test["_weight"].values.astype(float)

    print(f"  Test set: {len(test)} rows "
          f"({int((sw_te>1).sum())} personal + {int((sw_te==1).sum())} historical)")

    if imputer is not None:
        X_te = imputer.transform(X_te)
    X_te_s = scaler.transform(X_te)

    y_prob = clf.predict_proba(X_te_s)[:, 1]

    # ── 4. Calibration check ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  CALIBRATION CHECK")
    print("═" * 60)
    cal_stats = run_calibration(
        y_true    = y_te,
        y_prob    = y_prob,
        n_bins    = 10,
        save_path = cal_path,
    )

    # ── 5. Profit simulation ───────────────────────────────────────────────
    print("═" * 60)
    print("  PROFIT SIMULATION")
    print("═" * 60)
    # Run at multiple thresholds for context
    sim_results = {}
    for thr in [0.52, 0.55, 0.60]:
        r = run_profit_sim(
            y_true    = y_te,
            y_prob    = y_prob,
            dates     = test["_date"],
            weights   = sw_te,
            threshold = thr,
            flat_bet  = 10.0,
            save_path = prof_path if thr == threshold else None,
        )
        sim_results[str(thr)] = r

    # ── 6. Calibration quality judgement ──────────────────────────────────
    print("═" * 60)
    print("  VERDICT")
    print("═" * 60)
    ece = cal_stats["ece"]
    if ece < 0.03:
        cal_grade = "EXCELLENT — probabilities are trustworthy for Kelly sizing"
    elif ece < 0.06:
        cal_grade = "GOOD — minor miscalibration; Kelly sizing usable with caution"
    elif ece < 0.10:
        cal_grade = "FAIR — meaningful miscalibration; scale Kelly fractions by 0.5×"
    else:
        cal_grade = "POOR — significant miscalibration; do not use raw probs for sizing"
    print(f"  Calibration (ECE={ece:.3f}): {cal_grade}")

    primary_sim = sim_results.get(str(threshold), {})
    if primary_sim.get("roi", 0) > 0.03:
        print(f"  P&L ({threshold:.0%} cutoff): POSITIVE ROI={primary_sim['roi']:+.1%} — model has edge")
    elif primary_sim.get("roi", 0) > -0.02:
        print(f"  P&L ({threshold:.0%} cutoff): BREAKEVEN ROI={primary_sim['roi']:+.1%} — insufficient sample")
    else:
        print(f"  P&L ({threshold:.0%} cutoff): NEGATIVE ROI={primary_sim['roi']:+.1%} — below threshold or overfitting")
    print()

    # ── 7. Save JSON summary ───────────────────────────────────────────────
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_test":        len(test),
        "calibration":   cal_stats,
        "profit_sim":    sim_results,
    }
    result_path = os.path.join(plot_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {result_path}")

    return summary


def _parse_args():
    p = argparse.ArgumentParser(description="BetIQ model evaluation")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Primary P&L simulation threshold (default 0.55)")
    p.add_argument("--plot-dir", default=None,
                   help="Directory to save plots (default: data/)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(threshold=args.threshold, plot_dir=args.plot_dir, no_plots=args.no_plots)
