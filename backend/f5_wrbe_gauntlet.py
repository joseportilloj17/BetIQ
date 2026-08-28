"""
f5_wrbe_gauntlet.py — F5 WR-vs-Breakeven Gauntlet

Does the 0.69 AUC on F5 cover (+0.5) become MONEY?

Same gauntlet that killed everything else:
  - 10 random seeds × 3 CV schemes (chrono, random 5-fold, TimeSeriesSplit)
  - Game-level dedup (one pick per game, home-side convention)
  - WR vs breakeven at -110/-110 standard F5 spread vig (BE = 52.38%)
  - Noise floor: shuffle labels, same pipeline → what WR does chance give?

Key constraint: we do NOT have historical F5 odds.
  - F5 spreads are almost always ±0.5 at -110/-110
  - Breakeven at -110 = 1 / (1 + 1/1.909) ≈ 52.38%
  - If our model picks the right SIDE >52.38%, it's printing money

Usage:
  python f5_wrbe_gauntlet.py
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

# F5 spread at -110 both sides → breakeven
F5_ODDS = 1.909  # decimal odds for -110
F5_BE = 1.0 / F5_ODDS  # 0.5238


def _load_data() -> pd.DataFrame:
    """
    Load feature matrix merged with F5 outcomes.
    Returns one row per game with features + f5_home_covers_05 target.
    """
    from features import build_feature_matrix, TARGET_ATS

    fm = build_feature_matrix("MLB")
    if "game_id" not in fm.columns:
        raise RuntimeError("No game_id in feature matrix")

    fm["game_pk"] = fm["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")

    # Load F5 outcomes
    conn = sqlite3.connect(_HIST_DB)
    f5 = pd.read_sql_query(
        "SELECT game_pk, f5_home, f5_away FROM game_innings "
        "WHERE f5_home IS NOT NULL AND f5_away IS NOT NULL",
        conn,
    )
    conn.close()

    # F5 cover: home covers +0.5 (i.e., home leads or ties after 5)
    f5["f5_home_covers_05"] = ((f5["f5_home"] - f5["f5_away"] + 0.5) > 0).astype(int)

    # Merge
    merged = fm.merge(f5[["game_pk", "f5_home_covers_05"]], on="game_pk", how="inner")

    # Require ATS target too (sanity check)
    merged = merged.dropna(subset=[TARGET_ATS, "f5_home_covers_05"])
    merged["f5_target"] = merged["f5_home_covers_05"].astype(int)

    print(f"  Loaded {len(merged)} games with features + F5 outcomes")
    return merged


def _score(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Score all games with ATS model → P(home covers).
    """
    from ml_model import _load_ats_submodel

    artifacts = _load_ats_submodel("MLB")
    if artifacts is None:
        raise RuntimeError("Can't load MLB ATS model")
    clf, scaler, imputer, feat_cols = artifacts

    # Build X
    X = merged[feat_cols].copy() if all(c in merged.columns for c in feat_cols) else None
    if X is None:
        available = [c for c in feat_cols if c in merged.columns and not c.startswith("_")]
        X = merged[available].copy()
        if "_spread_value" in feat_cols:
            X["_spread_value"] = -1.5
        if "_spread_abs" in feat_cols:
            X["_spread_abs"] = 1.5
        if "_is_fav_spread" in feat_cols:
            X["_is_fav_spread"] = 1
        X = X.reindex(columns=feat_cols, fill_value=0)

    X = X.fillna(X.median())
    X_s = scaler.transform(X) if scaler else X.values
    X_i = imputer.transform(X_s) if imputer else X_s

    merged = merged.copy()
    merged["prob_home_covers"] = clf.predict_proba(X_i)[:, 1]

    return merged


def _pick_side(prob: float) -> str:
    """Given P(home covers), pick home or away."""
    return "home" if prob >= 0.5 else "away"


def _wr_from_picks(probs: np.ndarray, y_f5: np.ndarray) -> float:
    """
    Win rate if we bet the model's favored side at -110.
    Pick home if prob >= 0.5, else away.
    y_f5 = 1 if home covers +0.5.
    """
    picks_home = probs >= 0.5
    correct = np.where(picks_home, y_f5 == 1, y_f5 == 0)
    return correct.mean()


def _roi_from_wr(wr: float, odds: float = F5_ODDS) -> float:
    """Flat-bet ROI given WR and decimal odds."""
    return wr * odds - 1.0


def _run_cv(merged: pd.DataFrame, seed: int, scheme: str) -> dict:
    """
    Run one CV fold scheme. Returns {wr, roi, n_bets, n_correct, scheme, seed}.
    """
    from sklearn.model_selection import KFold, TimeSeriesSplit

    n = len(merged)
    probs = merged["prob_home_covers"].values
    y = merged["f5_target"].values

    if scheme == "chrono":
        # Chronological 80/20 split
        split = int(n * 0.8)
        test_idx = np.arange(split, n)
    elif scheme == "random":
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        # Collect all test indices
        test_idx = np.concatenate([test for _, test in kf.split(probs)])
    elif scheme == "tscv":
        tscv = TimeSeriesSplit(n_splits=5)
        test_idx = np.concatenate([test for _, test in tscv.split(probs)])
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    p_test = probs[test_idx]
    y_test = y[test_idx]

    wr = _wr_from_picks(p_test, y_test)
    roi = _roi_from_wr(wr)

    return {
        "scheme": scheme,
        "seed": seed,
        "wr": wr,
        "wr_minus_be": wr - F5_BE,
        "roi_pct": roi * 100,
        "n_bets": len(test_idx),
        "n_correct": int((p_test >= 0.5).sum() * wr + (p_test < 0.5).sum() * wr),
    }


def _noise_floor(merged: pd.DataFrame, n_shuffles: int = 100) -> float:
    """
    Shuffle F5 target labels, run same pipeline → noise WR.
    Returns mean WR under shuffled labels.
    """
    rng = np.random.default_rng(42)
    probs = merged["prob_home_covers"].values
    y = merged["f5_target"].values.copy()

    wrs = []
    for _ in range(n_shuffles):
        y_shuf = rng.permutation(y)
        wr = _wr_from_picks(probs, y_shuf)
        wrs.append(wr)

    return np.mean(wrs)


def _confidence_interval(wr: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for win rate."""
    p = wr
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def main():
    print("=" * 70)
    print("F5 WR-vs-BREAKEVEN GAUNTLET")
    print("Does the 0.69 AUC on F5 cover (+0.5) become MONEY?")
    print("=" * 70)
    print(f"\n  F5 spread odds: -110 / -110 (decimal {F5_ODDS:.3f})")
    print(f"  Breakeven WR: {F5_BE:.4f} ({F5_BE*100:.2f}%)")

    # ── Step 1: Load data ──
    print("\n── Loading data ──")
    merged = _load_data()

    # Sort chronologically for TSCV/chrono schemes
    if "game_date" in merged.columns:
        merged = merged.sort_values("game_date").reset_index(drop=True)

    # ── Step 2: Score with ATS model ──
    print("\n── Scoring with ATS model ──")
    merged = _score(merged)

    # ── Step 3: Overall WR (full sample, no CV — upper bound) ──
    probs = merged["prob_home_covers"].values
    y_f5 = merged["f5_target"].values
    wr_full = _wr_from_picks(probs, y_f5)
    roi_full = _roi_from_wr(wr_full)
    ci_lo, ci_hi = _confidence_interval(wr_full, len(merged))

    print(f"\n── Full-sample (in-sample upper bound) ──")
    print(f"  N games:  {len(merged)}")
    print(f"  WR:       {wr_full:.4f} ({wr_full*100:.2f}%)")
    print(f"  BE:       {F5_BE:.4f} ({F5_BE*100:.2f}%)")
    print(f"  WR - BE:  {(wr_full - F5_BE)*100:+.2f}pp")
    print(f"  ROI:      {roi_full*100:+.2f}%")
    print(f"  95% CI:   [{ci_lo:.4f}, {ci_hi:.4f}]")
    be_in_ci = "YES — BE inside CI (not significant)" if ci_lo <= F5_BE <= ci_hi else "NO — BE outside CI (significant)"
    print(f"  BE in CI: {be_in_ci}")

    # ── Step 4: Side-pick distribution ──
    n_home = (probs >= 0.5).sum()
    n_away = (probs < 0.5).sum()
    print(f"\n  Pick distribution: {n_home} home ({n_home/len(merged)*100:.1f}%), "
          f"{n_away} away ({n_away/len(merged)*100:.1f}%)")

    # ── Step 5: Multi-scheme CV gauntlet ──
    print(f"\n── 10-seed × 3-scheme CV gauntlet ──")
    print(f"  {'Scheme':<10} {'Seed':<6} {'WR%':<8} {'WR-BE':<8} {'ROI%':<8} {'N'}")

    results = []
    for seed in range(10):
        for scheme in ["chrono", "random", "tscv"]:
            r = _run_cv(merged, seed, scheme)
            results.append(r)
            print(f"  {r['scheme']:<10} {r['seed']:<6} "
                  f"{r['wr']*100:.2f}%  {r['wr_minus_be']*100:+.2f}pp "
                  f"{r['roi_pct']:+.2f}%  {r['n_bets']}")

    # ── Step 6: Aggregate by scheme ──
    print(f"\n── Aggregate by CV scheme ──")
    print(f"  {'Scheme':<10} {'Mean WR%':<10} {'Mean WR-BE':<12} {'Mean ROI%':<10} "
          f"{'Min WR-BE':<10} {'Max WR-BE':<10}")
    for scheme in ["chrono", "random", "tscv"]:
        sub = [r for r in results if r["scheme"] == scheme]
        wrs = [r["wr"] for r in sub]
        wrbes = [r["wr_minus_be"] for r in sub]
        rois = [r["roi_pct"] for r in sub]
        print(f"  {scheme:<10} {np.mean(wrs)*100:.2f}%    "
              f"{np.mean(wrbes)*100:+.2f}pp     {np.mean(rois):+.2f}%     "
              f"{np.min(wrbes)*100:+.2f}pp    {np.max(wrbes)*100:+.2f}pp")

    # Grand mean
    all_wrbes = [r["wr_minus_be"] for r in results]
    all_wrs = [r["wr"] for r in results]
    all_rois = [r["roi_pct"] for r in results]
    print(f"\n  GRAND MEAN: WR={np.mean(all_wrs)*100:.2f}%, "
          f"WR-BE={np.mean(all_wrbes)*100:+.2f}pp, "
          f"ROI={np.mean(all_rois):+.2f}%")

    # How many of 30 trials are above BE?
    n_above = sum(1 for r in results if r["wr_minus_be"] > 0)
    print(f"  Trials above BE: {n_above}/30 ({n_above/30*100:.0f}%)")

    # ── Step 7: Noise floor ──
    print(f"\n── Noise floor (100 label shuffles) ──")
    noise_wr = _noise_floor(merged)
    print(f"  Noise WR:  {noise_wr:.4f} ({noise_wr*100:.2f}%)")
    print(f"  Actual WR: {wr_full:.4f} ({wr_full*100:.2f}%)")
    print(f"  Signal above noise: {(wr_full - noise_wr)*100:+.2f}pp")

    # ── Step 8: Quintile WR (does discrimination → money?) ──
    print(f"\n── Quintile WR (strongest → weakest signal) ──")
    print(f"  {'Q':<6} {'Prob':<10} {'F5 WR':<10} {'N':<6} {'WR-BE'}")
    sorted_idx = np.argsort(-probs)  # strongest signal first
    q_size = len(sorted_idx) // 5
    for i in range(5):
        start = i * q_size
        end = start + q_size if i < 4 else len(sorted_idx)
        idx = sorted_idx[start:end]
        avg_p = np.mean(probs[idx])
        wr = _wr_from_picks(probs[idx], y_f5[idx])
        print(f"  Q{i+1:<5} {avg_p:.4f}    {wr*100:.1f}%     {len(idx):<6} {(wr-F5_BE)*100:+.1f}pp")

    # ── Step 9: Confidence threshold sweep ──
    print(f"\n── Confidence threshold sweep (pick only when model confident) ──")
    print(f"  {'Thresh':<10} {'N picks':<10} {'WR%':<8} {'WR-BE':<8} {'ROI%'}")
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        mask = np.maximum(probs, 1 - probs) >= thresh
        if mask.sum() < 50:
            print(f"  {thresh:.2f}      {mask.sum():<10} — too few")
            continue
        p_sub = probs[mask]
        y_sub = y_f5[mask]
        wr = _wr_from_picks(p_sub, y_sub)
        roi = _roi_from_wr(wr)
        print(f"  {thresh:.2f}      {mask.sum():<10} {wr*100:.2f}%  "
              f"{(wr-F5_BE)*100:+.2f}pp {roi*100:+.2f}%")

    # ── Step 10: Year-level WR stability ──
    if "game_date" in merged.columns:
        print(f"\n── Year-level WR stability ──")
        merged_tmp = merged.copy()
        merged_tmp["year"] = pd.to_datetime(merged_tmp["game_date"]).dt.year.astype(str)
        print(f"  {'Year':<6} {'WR%':<8} {'WR-BE':<8} {'N'}")
        for yr, grp in merged_tmp.groupby("year"):
            p = grp["prob_home_covers"].values
            y = grp["f5_target"].values
            wr = _wr_from_picks(p, y)
            print(f"  {yr:<6} {wr*100:.2f}%  {(wr-F5_BE)*100:+.2f}pp {len(grp)}")

    # ═══ VERDICT ═══
    mean_wrbe = np.mean(all_wrbes) * 100
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    if mean_wrbe > 3.0 and n_above >= 25:
        print(f"  ✓ F5 PASSES — mean WR-BE = {mean_wrbe:+.2f}pp, "
              f"{n_above}/30 above BE")
        print(f"    The 0.69 AUC IS becoming money.")
    elif mean_wrbe > 0 and n_above >= 20:
        print(f"  ~ F5 MARGINAL — mean WR-BE = {mean_wrbe:+.2f}pp, "
              f"{n_above}/30 above BE")
        print(f"    Signal exists but thin. Needs CLV confirmation.")
    else:
        print(f"  ✗ F5 FAILS — mean WR-BE = {mean_wrbe:+.2f}pp, "
              f"{n_above}/30 above BE")
        print(f"    AUC does NOT become money. Efficient market wall.")


if __name__ == "__main__":
    main()
