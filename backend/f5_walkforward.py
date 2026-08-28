"""
f5_walkforward.py — Walk-forward validation of purpose-built F5 model

The original F5 gauntlet tested the WRONG model (ATS transferred to F5).
A purpose-built F5 model showed +8-12pp WR-BE on N=250 (one 20-day window).

This script validates whether that edge holds across MULTIPLE independent
test windows via walk-forward:

  Fold 1: Train 2023      → Test 2024 (full year, ~2400 games)
  Fold 2: Train 2023-24   → Test 2025 (full year, ~2400 games)
  Fold 3: Train 2023-25   → Test 2026 (partial year, ~600 games)

  + Recency variants (trailing 1 year only):
  Fold 2R: Train 2024     → Test 2025
  Fold 3R: Train 2025     → Test 2026

Each fold: train GBC on F5 target, test on held-out year, report WR-vs-BE.
Multi-seed (5 seeds) for each fold.

This is the make-or-break: +12pp on 250 games is the same N as money-map
cells that died. Walk-forward across years with N=2400 per window is real.

Usage:
  python f5_walkforward.py
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
_DATA = os.path.join(_BASE, "data")
_HIST_DB = os.path.join(_DATA, "historical.db")
sys.path.insert(0, os.path.join(_BASE, "backend"))

F5_ODDS = 1.909
F5_BE = 1.0 / F5_ODDS  # 52.38%
N_SEEDS = 5


def _load_data():
    from features import build_feature_matrix, TARGET_ATS, MODEL_FEATURE_COLS_V1_PRUNED

    fm = build_feature_matrix("MLB")
    fm["game_pk"] = fm["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")

    conn = sqlite3.connect(_HIST_DB)
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
    picks_home = probs >= 0.5
    correct = np.where(picks_home, y == 1, y == 0)
    return correct.mean()


def _ci(wr, n, z=1.96):
    p = wr
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    s = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return max(0, c-s), min(1, c+s)


def _train_and_test(train_df, test_df, feat_cols, target_col, seed=42):
    """Train GBC on train_df[target_col], test on test_df. Return probs."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    available = [c for c in feat_cols if c in train_df.columns]

    X_train = train_df[available].fillna(train_df[available].median())
    y_train = train_df[target_col].values.astype(int)
    X_test = test_df[available].fillna(test_df[available].median())

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]


def main():
    from sklearn.metrics import roc_auc_score, brier_score_loss

    print("=" * 70)
    print("F5 WALK-FORWARD VALIDATION")
    print("Does purpose-built F5 model hold across multiple year windows?")
    print("=" * 70)

    merged, feat_cols, target_ats = _load_data()
    print(f"\n  Total: {len(merged)} games")
    for yr, cnt in merged.groupby("year").size().items():
        print(f"    {yr}: {cnt}")

    # ═══════════════════════════════════════════════════════════════════
    # WALK-FORWARD FOLDS
    # ═══════════════════════════════════════════════════════════════════

    folds = [
        # (name, train_years, test_year, target_col, description)
        ("F5: 2023→2024",       [2023],       2024, "f5_target",  "purpose-built F5"),
        ("F5: 2023-24→2025",    [2023, 2024], 2025, "f5_target",  "purpose-built F5"),
        ("F5: 2023-25→2026",    [2023, 2024, 2025], 2026, "f5_target", "purpose-built F5"),
        # Recency variants
        ("F5-recent: 2024→2025", [2024],      2025, "f5_target",  "recency F5"),
        ("F5-recent: 2025→2026", [2025],      2026, "f5_target",  "recency F5"),
        # ATS transfer baseline (what the original gauntlet tested)
        ("ATS→F5: 2023-24→2025", [2023, 2024], 2025, target_ats,  "ATS transfer to F5"),
        ("ATS→F5: 2023-25→2026", [2023, 2024, 2025], 2026, target_ats, "ATS transfer to F5"),
    ]

    print(f"\n{'='*70}")
    print(f"WALK-FORWARD RESULTS (multi-seed, {N_SEEDS} seeds each)")
    print(f"{'='*70}")
    print(f"\n  {'Fold':<25} {'Train N':<8} {'Test N':<8} "
          f"{'AUC':<8} {'WR%':<8} {'WR-BE':<9} {'ROI%':<8} {'95% CI'}")

    all_results = []

    for name, train_years, test_year, target_col, desc in folds:
        train_mask = merged["year"].isin(train_years)
        test_mask = merged["year"] == test_year
        train_df = merged[train_mask]
        test_df = merged[test_mask]

        if len(test_df) < 30 or len(train_df) < 100:
            print(f"  {name:<25} SKIPPED (insufficient data)")
            continue

        y_test = test_df["f5_target"].values  # ALWAYS evaluate on F5 target

        seed_results = []
        for seed in range(N_SEEDS):
            probs = _train_and_test(train_df, test_df, feat_cols, target_col, seed)
            auc = roc_auc_score(y_test, probs)
            brier = brier_score_loss(y_test, probs)
            wr_val = _wr(probs, y_test)
            seed_results.append({
                "auc": auc, "brier": brier, "wr": wr_val,
                "wr_be": wr_val - F5_BE,
                "roi": (wr_val * F5_ODDS - 1) * 100,
            })

        mean_auc = np.mean([r["auc"] for r in seed_results])
        mean_wr = np.mean([r["wr"] for r in seed_results])
        mean_wrbe = np.mean([r["wr_be"] for r in seed_results])
        mean_roi = np.mean([r["roi"] for r in seed_results])
        ci_lo, ci_hi = _ci(mean_wr, len(test_df))

        all_results.append({
            "name": name, "desc": desc,
            "train_n": len(train_df), "test_n": len(test_df),
            "test_year": test_year, "target": target_col,
            "mean_auc": mean_auc, "mean_wr": mean_wr,
            "mean_wrbe": mean_wrbe, "mean_roi": mean_roi,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "min_wr": min(r["wr"] for r in seed_results),
            "max_wr": max(r["wr"] for r in seed_results),
            "seeds": seed_results,
        })

        be_marker = " ***" if ci_lo > F5_BE else ""
        print(f"  {name:<25} {len(train_df):<8} {len(test_df):<8} "
              f"{mean_auc:.4f}  {mean_wr*100:.2f}%  "
              f"{mean_wrbe*100:+.2f}pp  {mean_roi:+.2f}%  "
              f"[{ci_lo:.3f},{ci_hi:.3f}]{be_marker}")

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON: PURPOSE-BUILT vs ATS TRANSFER on same test windows
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"PURPOSE-BUILT F5 vs ATS TRANSFER (same test windows)")
    print(f"{'='*70}")

    for test_yr in [2025, 2026]:
        pb = [r for r in all_results
              if r["test_year"] == test_yr and "purpose" in r["desc"]
              and "recent" not in r["desc"]]
        tr = [r for r in all_results
              if r["test_year"] == test_yr and "transfer" in r["desc"]]
        if pb and tr:
            pb, tr = pb[0], tr[0]
            delta_wr = pb["mean_wr"] - tr["mean_wr"]
            print(f"\n  Test year {test_yr} (N={pb['test_n']}):")
            print(f"    Purpose-built F5: WR={pb['mean_wr']*100:.2f}%, "
                  f"WR-BE={pb['mean_wrbe']*100:+.2f}pp, AUC={pb['mean_auc']:.4f}")
            print(f"    ATS transfer:     WR={tr['mean_wr']*100:.2f}%, "
                  f"WR-BE={tr['mean_wrbe']*100:+.2f}pp, AUC={tr['mean_auc']:.4f}")
            print(f"    ΔWR (purpose - transfer) = {delta_wr*100:+.2f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # RECENCY COMPARISON on same test windows
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"RECENCY vs ALL-DATA (purpose-built F5, same test windows)")
    print(f"{'='*70}")

    for test_yr in [2025, 2026]:
        alldata = [r for r in all_results
                   if r["test_year"] == test_yr and "purpose" in r["desc"]]
        recent = [r for r in all_results
                  if r["test_year"] == test_yr and "recency" in r["desc"]]
        if alldata and recent:
            a, rc = alldata[0], recent[0]
            print(f"\n  Test year {test_yr}:")
            print(f"    All-data:     WR={a['mean_wr']*100:.2f}%, "
                  f"WR-BE={a['mean_wrbe']*100:+.2f}pp (train N={a['train_n']})")
            print(f"    Recent-only:  WR={rc['mean_wr']*100:.2f}%, "
                  f"WR-BE={rc['mean_wrbe']*100:+.2f}pp (train N={rc['train_n']})")
            print(f"    ΔWR (recent - all) = {(rc['mean_wr']-a['mean_wr'])*100:+.2f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # POOLED OOS RESULT (all walk-forward test windows combined)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"POOLED WALK-FORWARD (all purpose-built F5 test windows combined)")
    print(f"{'='*70}")

    # Combine all purpose-built F5 fold results
    pb_folds = [r for r in all_results if "purpose" in r["desc"]]
    if pb_folds:
        total_correct = sum(r["mean_wr"] * r["test_n"] for r in pb_folds)
        total_n = sum(r["test_n"] for r in pb_folds)
        pooled_wr = total_correct / total_n
        pooled_ci = _ci(pooled_wr, total_n)

        print(f"\n  Pooled N = {total_n} (across {len(pb_folds)} folds)")
        print(f"  Pooled WR = {pooled_wr*100:.2f}%")
        print(f"  Pooled WR-BE = {(pooled_wr-F5_BE)*100:+.2f}pp")
        print(f"  Pooled ROI = {(pooled_wr*F5_ODDS-1)*100:+.2f}%")
        print(f"  95% CI = [{pooled_ci[0]:.4f}, {pooled_ci[1]:.4f}]")
        print(f"  BE ({F5_BE:.4f}) {'OUTSIDE CI → significant' if pooled_ci[0] > F5_BE else 'INSIDE CI → not significant'}")

    # Also pool ATS transfer for comparison
    tr_folds = [r for r in all_results if "transfer" in r["desc"]]
    if tr_folds:
        total_correct_tr = sum(r["mean_wr"] * r["test_n"] for r in tr_folds)
        total_n_tr = sum(r["test_n"] for r in tr_folds)
        pooled_wr_tr = total_correct_tr / total_n_tr

        print(f"\n  ATS transfer pooled: WR={pooled_wr_tr*100:.2f}%, "
              f"WR-BE={(pooled_wr_tr-F5_BE)*100:+.2f}pp (N={total_n_tr})")
        print(f"  Purpose-built advantage: {(pooled_wr-pooled_wr_tr)*100:+.2f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # PER-SEED DETAIL (worst case analysis)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"WORST-CASE ANALYSIS (minimum WR across all seeds)")
    print(f"{'='*70}")

    for r in all_results:
        if "purpose" in r["desc"] or "recency" in r["desc"]:
            wrs = [s["wr"] for s in r["seeds"]]
            print(f"  {r['name']:<25} min={min(wrs)*100:.2f}% "
                  f"max={max(wrs)*100:.2f}% "
                  f"above BE: {sum(1 for w in wrs if w > F5_BE)}/{len(wrs)}")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if pb_folds:
        n_folds_above = sum(1 for r in pb_folds if r["mean_wrbe"] > 0)
        all_above = n_folds_above == len(pb_folds)

        if all_above and pooled_wr > F5_BE + 0.03:
            print(f"\n  PURPOSE-BUILT F5 VALIDATED")
            print(f"  Pooled WR={pooled_wr*100:.2f}% ({(pooled_wr-F5_BE)*100:+.2f}pp above BE)")
            print(f"  {n_folds_above}/{len(pb_folds)} folds above BE across {total_n} games")
            print(f"  The +8-12pp edge from purpose-building is REAL, not lucky N=250.")
        elif all_above:
            print(f"\n  PURPOSE-BUILT F5 MARGINAL BUT CONSISTENT")
            print(f"  Pooled WR={pooled_wr*100:.2f}% ({(pooled_wr-F5_BE)*100:+.2f}pp)")
            print(f"  {n_folds_above}/{len(pb_folds)} folds above BE but narrow margin")
            print(f"  Forward CLV is the tiebreaker.")
        elif n_folds_above >= len(pb_folds) // 2:
            print(f"\n  PURPOSE-BUILT F5 INCONSISTENT")
            print(f"  Pooled WR={pooled_wr*100:.2f}% ({(pooled_wr-F5_BE)*100:+.2f}pp)")
            print(f"  Only {n_folds_above}/{len(pb_folds)} folds above BE")
            print(f"  Edge exists in some windows, not others. Not deployable.")
        else:
            print(f"\n  PURPOSE-BUILT F5 FAILS WALK-FORWARD")
            print(f"  Pooled WR={pooled_wr*100:.2f}% ({(pooled_wr-F5_BE)*100:+.2f}pp)")
            print(f"  Only {n_folds_above}/{len(pb_folds)} folds above BE")
            print(f"  N=250 was lucky. Purpose-building helps ranking but not enough for profit.")


if __name__ == "__main__":
    main()
