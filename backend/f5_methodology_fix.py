"""
f5_methodology_fix.py — Fix the leaky F5 methodology

PROBLEM FOUND: F5 home covers +0.5 has a 60% BASE RATE.
Picking HOME every single game gives 60% WR. The "purpose-built F5 model"
getting 61% WR is adding ~1pp above the naive baseline, NOT +8pp above
a 52.38% breakeven.

The 52.38% breakeven (-110/-110) is WRONG for F5 +0.5:
  - F5 +0.5 is not a balanced 50/50 market like full-game ATS
  - It's essentially a moneyline bet with ties eliminated
  - Home cover base rate = 60% → fair price ≈ -150 (not -110)
  - With vig (~4.5%), actual BE ≈ 62-63%

CALIBRATION CHECK: full-game spreads are KNOWN negative-ROI forward
(ats_ungated: 705W/713L = 49.72% WR). Any honest test must reproduce
this. If it shows +9pp, the test is broken.

This script:
  1. Shows the base rate problem explicitly
  2. Computes model value ABOVE naive baseline (the real signal)
  3. Uses realistic breakeven for F5 (not -110/-110)
  4. Calibrates against full-game spread ground truth
  5. Re-runs walk-forward with corrected methodology

Usage:
  python f5_methodology_fix.py
"""
from __future__ import annotations

import os
import sys
import sqlite3
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_BASE, "backend"))


def _load_all():
    """Load feature matrix with both F5 and ATS targets."""
    from features import build_feature_matrix, TARGET_ATS, MODEL_FEATURE_COLS_V1_PRUNED

    fm = build_feature_matrix("MLB")
    fm["game_pk"] = fm["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")

    conn = sqlite3.connect(os.path.join(_BASE, "data", "historical.db"))
    f5 = pd.read_sql_query(
        "SELECT game_pk, f5_home, f5_away FROM game_innings "
        "WHERE f5_home IS NOT NULL AND f5_away IS NOT NULL", conn)
    conn.close()

    f5["f5_target"] = ((f5["f5_home"] - f5["f5_away"] + 0.5) > 0).astype(int)
    merged = fm.merge(f5[["game_pk", "f5_target"]], on="game_pk", how="inner")
    merged = merged.dropna(subset=[TARGET_ATS, "f5_target"])
    merged["year"] = pd.to_datetime(merged["game_date"]).dt.year
    merged = merged.sort_values("game_date").reset_index(drop=True)

    return merged, MODEL_FEATURE_COLS_V1_PRUNED, TARGET_ATS


def _wr(probs, y):
    """WR: pick home if prob >= 0.5, else away."""
    picks_home = probs >= 0.5
    correct = np.where(picks_home, y == 1, y == 0)
    return correct.mean()


def _train_gbc(train_df, test_df, feat_cols, target_col, seed=42):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    available = [c for c in feat_cols if c in train_df.columns]
    X_tr = train_df[available].fillna(train_df[available].median())
    y_tr = train_df[target_col].values.astype(int)
    X_te = test_df[available].fillna(test_df[available].median())

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr_s)
    X_te_i = imp.transform(X_te_s)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    clf.fit(X_tr_i, y_tr)
    return clf.predict_proba(X_te_i)[:, 1]


def main():
    from sklearn.metrics import roc_auc_score

    print("=" * 70)
    print("F5 METHODOLOGY FIX — Corrected Walk-Forward")
    print("=" * 70)

    merged, feat_cols, target_ats = _load_all()
    print(f"  Total: {len(merged)} games")

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: THE BASE RATE PROBLEM
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"PART 1: BASE RATE REALITY CHECK")
    print(f"{'='*70}")

    ats_base = merged[target_ats].mean()
    f5_base = merged["f5_target"].mean()

    print(f"\n  Full-game covered_spread base rate: {ats_base:.4f} ({ats_base*100:.2f}%)")
    print(f"  F5 home covers +0.5 base rate:      {f5_base:.4f} ({f5_base*100:.2f}%)")
    print(f"\n  'Always pick home' WR:")
    print(f"    Full-game: {ats_base*100:.2f}% (≈50% — balanced by spread design)")
    print(f"    F5 +0.5:   {f5_base*100:.2f}% (≈60% — NOT balanced, home advantage + tie elimination)")

    # What does this mean for breakeven?
    # Standard vig is ~4.5% total (each side at -110)
    # For a balanced market (base=50%): BE = 52.38% at -110/-110
    # For an unbalanced market (base=60%): market prices home side higher
    # Typical F5 home +0.5 odds: around -150 (1.667), away -0.5: +130 (2.30)
    # BE for home side at -150: 1/1.667 = 60.0% (already at base rate!)
    # With standard vig: home at -155 → BE = 60.8%
    #                    away at +135 → BE = 42.6%

    print(f"\n  ── F5 +0.5 realistic pricing ──")
    print(f"  Market knows home covers 60% → prices home +0.5 around -150 to -160")
    print(f"  Home +0.5 at -155: BE = {100/155*100:.1f}%")
    print(f"  Away -0.5 at +135: BE = {100/235*100:.1f}%")
    print(f"\n  → A model picking home with 60% accuracy is AT breakeven, not +8pp above it")
    print(f"  → The correct WR-vs-BE comparison uses ~60-61% BE, not 52.38%")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: MODEL VALUE ABOVE NAIVE BASELINE
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"PART 2: MODEL VALUE ABOVE NAIVE ('always pick home') BASELINE")
    print(f"{'='*70}")

    print(f"\n  The only question that matters: does the model beat 'always pick home'?")
    print(f"  If not, the model adds ZERO value — you could pick home without a model.")

    # Walk-forward: train on prior years, test on next year
    folds = [
        ("2023→2024", [2023], 2024),
        ("2023-24→2025", [2023, 2024], 2025),
        ("2023-25→2026", [2023, 2024, 2025], 2026),
    ]

    print(f"\n  ── Walk-forward: purpose-built F5 vs naive baselines ──")
    print(f"  {'Fold':<20} {'N test':<8} {'Model WR':<10} {'Home WR':<10} "
          f"{'Base rate':<10} {'Model-Home':<12} {'AUC'}")

    all_model_wrs = []
    all_home_wrs = []
    all_ns = []

    for name, train_years, test_year in folds:
        train = merged[merged["year"].isin(train_years)]
        test = merged[merged["year"] == test_year]

        if len(test) < 30:
            continue

        y_test = test["f5_target"].values
        base = y_test.mean()  # Actual base rate in test window
        home_wr = base  # "Always pick home" WR = base rate

        # Multi-seed model
        seed_wrs = []
        seed_aucs = []
        for seed in range(5):
            probs = _train_gbc(train, test, feat_cols, "f5_target", seed)
            model_wr = _wr(probs, y_test)
            auc = roc_auc_score(y_test, probs)
            seed_wrs.append(model_wr)
            seed_aucs.append(auc)

        mean_wr = np.mean(seed_wrs)
        mean_auc = np.mean(seed_aucs)
        delta = mean_wr - home_wr

        all_model_wrs.append(mean_wr)
        all_home_wrs.append(home_wr)
        all_ns.append(len(test))

        print(f"  {name:<20} {len(test):<8} {mean_wr*100:.2f}%    "
              f"{home_wr*100:.2f}%    {base*100:.1f}%     "
              f"{delta*100:+.2f}pp      {mean_auc:.4f}")

    # Pooled
    pooled_model = sum(w * n for w, n in zip(all_model_wrs, all_ns)) / sum(all_ns)
    pooled_home = sum(w * n for w, n in zip(all_home_wrs, all_ns)) / sum(all_ns)
    print(f"\n  {'POOLED':<20} {sum(all_ns):<8} {pooled_model*100:.2f}%    "
          f"{pooled_home*100:.2f}%    {'—':<10} {(pooled_model-pooled_home)*100:+.2f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: CALIBRATION CHECK — FULL-GAME SPREADS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"PART 3: CALIBRATION CHECK — Full-game spreads (known negative forward)")
    print(f"{'='*70}")
    print(f"\n  Ground truth: ats_ungated = 705W/713L = 49.72% WR (NEGATIVE)")
    print(f"  If our test shows spreads positive, the test is still broken.")

    print(f"\n  ── Walk-forward: full-game spread model ──")
    print(f"  {'Fold':<20} {'N test':<8} {'Model WR':<10} {'Home WR':<10} "
          f"{'Base rate':<10} {'Model-Home':<12} {'WR vs 52.38% BE'}")

    for name, train_years, test_year in folds:
        train = merged[merged["year"].isin(train_years)]
        test = merged[merged["year"] == test_year]

        if len(test) < 30:
            continue

        y_test_ats = test[target_ats].values.astype(int)
        base_ats = y_test_ats.mean()
        home_wr_ats = base_ats

        seed_wrs = []
        for seed in range(5):
            probs = _train_gbc(train, test, feat_cols, target_ats, seed)
            wr = _wr(probs, y_test_ats)
            seed_wrs.append(wr)

        mean_wr = np.mean(seed_wrs)
        delta = mean_wr - home_wr_ats
        wr_vs_be = mean_wr - 0.5238

        print(f"  {name:<20} {len(test):<8} {mean_wr*100:.2f}%    "
              f"{home_wr_ats*100:.2f}%    {base_ats*100:.1f}%     "
              f"{delta*100:+.2f}pp      {wr_vs_be*100:+.2f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: THE REAL F5 QUESTION — Does model beat the market?
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"PART 4: DOES F5 MODEL BEAT THE MARKET?")
    print(f"{'='*70}")

    print(f"""
  The market prices F5 +0.5 at roughly -150 to -160 (home) / +130 to +135 (away).
  These prices ALREADY reflect the 60% home cover base rate.

  For the model to make money, it must:
  1. Identify games where home covers MORE than ~62% → bet home (at -155 BE≈61%)
  2. Identify games where home covers LESS than ~42% → bet away (at +135 BE≈43%)
  3. The MODEL's contribution = games correctly re-ranked vs the base rate

  We can't test actual profitability without F5 odds (unavailable historically).
  But we CAN test: does the model RE-RANK games above the base rate?""")

    # Test: conditional WR by model confidence
    print(f"\n  ── Walk-forward: model conditional accuracy vs base rate ──")
    print(f"  Can the model identify WHEN home covers more/less than average?")

    for name, train_years, test_year in folds:
        train = merged[merged["year"].isin(train_years)]
        test = merged[merged["year"] == test_year]

        if len(test) < 100:
            continue

        y_test = test["f5_target"].values
        base = y_test.mean()

        probs = _train_gbc(train, test, feat_cols, "f5_target", seed=42)
        auc = roc_auc_score(y_test, probs)

        # Quintile analysis: does the model discriminate?
        sorted_idx = np.argsort(probs)
        q_size = len(sorted_idx) // 5

        print(f"\n  {name} (N={len(test)}, base={base*100:.1f}%, AUC={auc:.4f}):")
        print(f"    {'Q':<4} {'Prob':<8} {'Actual%':<10} {'vs base':<10} {'N'}")

        q_rates = []
        for i in range(5):
            start = i * q_size
            end = start + q_size if i < 4 else len(sorted_idx)
            idx = sorted_idx[start:end]
            avg_p = np.mean(probs[idx])
            actual = np.mean(y_test[idx])
            q_rates.append(actual)
            print(f"    Q{i+1:<3} {avg_p:.4f}  {actual*100:.1f}%     "
                  f"{(actual-base)*100:+.1f}pp    {len(idx)}")

        # Monotonicity check
        diffs = [q_rates[i+1] - q_rates[i] for i in range(len(q_rates)-1)]
        monotonic = all(d >= 0 for d in diffs)
        spread = q_rates[-1] - q_rates[0]
        print(f"    Monotonic: {'YES' if monotonic else 'NO'}, "
              f"Q5-Q1 spread: {spread*100:.1f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: RECENCY (only for pitcher-driven targets)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"PART 5: RECENCY — F5 (pitcher-driven) vs SPREADS (team-level)")
    print(f"{'='*70}")
    print(f"  Recency helps fast-changing units (pitchers) not slow-changing (teams)")

    # F5 recency: train 2025→test 2026 vs train 2023-25→test 2026
    test26 = merged[merged["year"] == 2026]
    if len(test26) >= 50:
        y26 = test26["f5_target"].values
        base26 = y26.mean()

        configs = [
            ("F5: all (2023-25)", merged[merged["year"].isin([2023,2024,2025])], "f5_target"),
            ("F5: recent (2025)", merged[merged["year"] == 2025], "f5_target"),
            ("ATS: all (2023-25)", merged[merged["year"].isin([2023,2024,2025])], target_ats),
            ("ATS: recent (2025)", merged[merged["year"] == 2025], target_ats),
        ]

        print(f"\n  Test: 2026 (N={len(test26)})")
        print(f"  F5 base rate: {base26*100:.1f}%")
        print(f"  ATS base rate: {test26[target_ats].mean()*100:.1f}%")
        print(f"\n  {'Config':<25} {'Train N':<8} {'Model WR':<10} {'Naive WR':<10} {'Δ model-naive'}")

        for config_name, train_df, target in configs:
            y_eval = test26["f5_target"].values if "F5" in config_name else test26[target_ats].values.astype(int)
            naive = y_eval.mean() if "F5" in config_name else y_eval.mean()

            seed_wrs = []
            for seed in range(5):
                probs = _train_gbc(train_df, test26, feat_cols, target, seed)
                wr = _wr(probs, y_eval)
                seed_wrs.append(wr)

            mean_wr = np.mean(seed_wrs)
            print(f"  {config_name:<25} {len(train_df):<8} {mean_wr*100:.2f}%    "
                  f"{naive*100:.2f}%    {(mean_wr-naive)*100:+.2f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    print(f"""
  THE LEAK: F5 home covers +0.5 has a 60% base rate.
  "Always pick home" gives 60% WR — the previous test compared this against
  a 52.38% breakeven (-110/-110), producing a phantom +8pp edge.

  THE REALITY: F5 +0.5 is priced at ~-150/-155 (home), not -110. The actual
  breakeven is ~60-61%, not 52.38%. The model's 61% WR is AT breakeven.

  MODEL VALUE: the real question is Model WR minus Naive WR (always-pick-home).
  Pooled model WR = {pooled_model*100:.2f}%, pooled naive WR = {pooled_home*100:.2f}%
  Model adds: {(pooled_model-pooled_home)*100:+.2f}pp above naive.

  This {(pooled_model-pooled_home)*100:+.2f}pp is the REAL signal — and it has to clear the
  ~4-5% vig at realistic F5 odds, not the 4.76% vig at -110.

  CALIBRATION: full-game spreads should show negative-ROI forward
  (ground truth: 49.72% WR). See Part 3 above — does our test reproduce this?""")


if __name__ == "__main__":
    main()
