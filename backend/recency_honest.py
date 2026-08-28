"""
recency_honest.py — Honest recency test with equal-N + forward validation

Prior recency conclusions ("recency helps F5, not spreads") used walk-forward
(now known artifact-prone) AND didn't control for sample size.

This script:
  1. Equal-N: recent-N vs random-N vs oldest-N (same N), kills sample-size confound
  2. Walk-forward (acknowledged artifact-prone, but apples-to-apples within the artifact)
  3. Forward reconciliation: does production model's anti-predictiveness (-2pp on
     model-favored +1.5 legs) vary by time — indirect signal for recency
  4. Naive baseline + actual-odds BE throughout
  5. Honest verdict on whether recency is a real effect or sample-size/artifact

For each target: covered_spread, f5_target, total_result

Usage:
  python recency_honest.py
"""
from __future__ import annotations

import os, sys, sqlite3, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_BASE, "backend"))

F5_BE_REALISTIC = 0.615  # home +0.5 at ~-160
SPREAD_BE = 0.5238  # -110
TOTAL_BE = 0.5238   # -110


def _load_all():
    from features import build_feature_matrix, TARGET_ATS, TARGET_OU, MODEL_FEATURE_COLS_V1_PRUNED

    fm = build_feature_matrix("MLB")
    fm["game_pk"] = fm["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")

    conn = sqlite3.connect(os.path.join(_BASE, "data", "historical.db"))
    f5 = pd.read_sql_query(
        "SELECT game_pk, f5_home, f5_away FROM game_innings "
        "WHERE f5_home IS NOT NULL AND f5_away IS NOT NULL", conn)
    conn.close()

    f5["f5_target"] = ((f5["f5_home"] - f5["f5_away"] + 0.5) > 0).astype(int)
    merged = fm.merge(f5[["game_pk", "f5_target"]], on="game_pk", how="inner")
    merged["year"] = pd.to_datetime(merged["game_date"]).dt.year
    merged = merged.sort_values("game_date").reset_index(drop=True)

    return merged, MODEL_FEATURE_COLS_V1_PRUNED, TARGET_ATS, TARGET_OU


def _wr(probs, y):
    picks_home = probs >= 0.5
    return np.where(picks_home, y == 1, y == 0).mean()


def _train_gbc(X_train, y_train, X_test, seed=42):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=seed)
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]


def _run_equal_n(merged, feat_cols, target_col, target_name, test_year):
    """Equal-N recency test for one target and one test year."""
    from sklearn.metrics import roc_auc_score

    available = [c for c in feat_cols if c in merged.columns]

    test_mask = merged["year"] == test_year
    train_pool = merged[~test_mask & (merged["year"] < test_year)]
    test_df = merged[test_mask].dropna(subset=[target_col])

    if len(test_df) < 50 or len(train_pool) < 100:
        return None

    y_test = test_df[target_col].values.astype(int)
    X_test = test_df[available].fillna(test_df[available].median())
    base_rate = y_test.mean()

    # Equal N = size of one year (~2400)
    n_equal = min(2400, len(train_pool))

    configs = {}

    # Recent-N: last N games from training pool (chronologically nearest to test)
    recent = train_pool.tail(n_equal)
    configs["Recent-N"] = recent

    # Oldest-N: first N games from training pool (chronologically farthest)
    oldest = train_pool.head(n_equal)
    configs["Oldest-N"] = oldest

    # Random-N: random N games from training pool (5 random draws, average)
    configs["Random-N"] = "RANDOM"  # handled specially

    # All data
    configs["All-data"] = train_pool

    results = []

    for name, train_df in configs.items():
        if name == "Random-N":
            # Average across 5 random draws
            seed_wrs = []
            seed_aucs = []
            for rseed in range(5):
                rng = np.random.default_rng(rseed)
                idx = rng.choice(len(train_pool), size=n_equal, replace=False)
                rand_df = train_pool.iloc[idx]
                X_tr = rand_df[available].fillna(rand_df[available].median())
                y_tr = rand_df[target_col].values.astype(int)

                # Average across 3 GBC seeds
                fold_wrs = []
                fold_aucs = []
                for gseed in range(3):
                    probs = _train_gbc(X_tr, y_tr, X_test, seed=gseed)
                    fold_wrs.append(_wr(probs, y_test))
                    fold_aucs.append(roc_auc_score(y_test, probs))
                seed_wrs.append(np.mean(fold_wrs))
                seed_aucs.append(np.mean(fold_aucs))

            results.append({
                "name": name, "n_train": n_equal,
                "mean_wr": np.mean(seed_wrs), "std_wr": np.std(seed_wrs),
                "mean_auc": np.mean(seed_aucs),
            })
        else:
            X_tr = train_df[available].fillna(train_df[available].median())
            y_tr = train_df[target_col].values.astype(int)

            # 3 GBC seeds
            fold_wrs = []
            fold_aucs = []
            for gseed in range(3):
                probs = _train_gbc(X_tr, y_tr, X_test, seed=gseed)
                fold_wrs.append(_wr(probs, y_test))
                fold_aucs.append(roc_auc_score(y_test, probs))

            results.append({
                "name": name, "n_train": len(train_df),
                "mean_wr": np.mean(fold_wrs), "std_wr": np.std(fold_wrs),
                "mean_auc": np.mean(fold_aucs),
            })

    return {
        "target": target_name,
        "test_year": test_year,
        "n_test": len(test_df),
        "base_rate": base_rate,
        "n_equal": n_equal,
        "results": results,
    }


def _forward_recency_check():
    """
    Check if the production model's forward anti-predictiveness varies by time.
    If model predictions are better on recent-data-similar games, that's indirect
    evidence for recency. If uniformly anti-predictive, recency won't help.
    """
    import sqlite3, re
    from collections import defaultdict

    conn = sqlite3.connect(os.path.join(_BASE, "data", "bets.db"))

    rows = conn.execute('''
        SELECT game_date, bet_info, predicted_win_prob, odds, status
        FROM mock_bets
        WHERE source = 'ats_ungated'
          AND status IN ('SETTLED_WIN', 'SETTLED_LOSS')
        ORDER BY game_date
    ''').fetchall()
    conn.close()

    legs = []
    for date, info, wp, odds_val, status in rows:
        m = re.match(r'(.+?)\s+([+-]\d+\.\d+)\s+\(Spread\)', info)
        if not m:
            continue
        spread = float(m.group(2))
        won = 1 if status == 'SETTLED_WIN' else 0
        legs.append({"date": date, "spread": spread, "wp": wp, "odds": odds_val, "won": won})

    # Split forward data into early vs late halves
    dates = sorted(set(l["date"] for l in legs))
    if len(dates) < 10:
        return None

    mid = dates[len(dates) // 2]
    early = [l for l in legs if l["date"] <= mid]
    late = [l for l in legs if l["date"] > mid]

    result = {}
    for label, subset in [("Early forward", early), ("Late forward", late), ("All forward", legs)]:
        plus15 = [l for l in subset if l["spread"] == 1.5]
        if len(plus15) < 30:
            continue

        wps = np.array([l["wp"] for l in plus15])
        wins = np.array([l["won"] for l in plus15])
        median_wp = np.median(wps)

        fav_mask = wps >= median_wp
        fav_wr = np.mean(wins[fav_mask]) if fav_mask.sum() > 0 else 0
        unfav_wr = np.mean(wins[~fav_mask]) if (~fav_mask).sum() > 0 else 0
        disc = fav_wr - unfav_wr

        # Per-bet odds BE
        all_odds = np.array([l["odds"] for l in plus15])
        avg_be = np.mean(1.0 / all_odds)
        overall_wr = np.mean(wins)
        pnl = sum(10 * (l["odds"] - 1) if l["won"] else -10 for l in plus15)
        roi = pnl / (10 * len(plus15)) * 100

        result[label] = {
            "n": len(plus15), "wr": overall_wr, "be": avg_be,
            "disc": disc, "fav_wr": fav_wr, "unfav_wr": unfav_wr,
            "roi": roi, "date_range": (subset[0]["date"], subset[-1]["date"]),
        }

    return result


def main():
    print("=" * 70)
    print("HONEST RECENCY TEST")
    print("Equal-N + forward validation, killing sample-size confound")
    print("=" * 70)

    merged, feat_cols, target_ats, target_ou = _load_all()

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: EQUAL-N WALK-FORWARD (artifact-prone but apples-to-apples)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 1: EQUAL-N WALK-FORWARD")
    print("Same N trains, same test window — isolates recency from sample size")
    print("(Walk-forward is artifact-prone per audit, but controls sample size)")
    print(f"{'='*70}")

    targets = [
        (target_ats, "Full-game spreads", "covered_spread"),
        ("f5_target", "F5 cover +0.5", "f5_target"),
    ]
    # Add totals if available
    if target_ou in merged.columns and merged[target_ou].notna().sum() > 1000:
        targets.append((target_ou, "Totals O/U", "total_result"))

    for target_col, target_name, _ in targets:
        if target_col not in merged.columns:
            continue
        valid = merged.dropna(subset=[target_col])
        if len(valid) < 2000:
            continue

        print(f"\n  ── {target_name} ──")

        for test_year in [2025, 2026]:
            result = _run_equal_n(merged, feat_cols, target_col, target_name, test_year)
            if result is None:
                continue

            base = result["base_rate"]
            naive_wr = max(base, 1 - base)  # always-majority-class WR

            print(f"\n  Test: {test_year} (N={result['n_test']}, "
                  f"base rate={base*100:.1f}%, naive WR={naive_wr*100:.1f}%)")
            print(f"  Equal N = {result['n_equal']} per config")
            print(f"  {'Config':<15} {'Train N':<10} {'AUC':<8} {'WR%':<8} "
                  f"{'WR-naive':<10} {'Std'}")

            for r in result["results"]:
                delta_naive = r["mean_wr"] - naive_wr
                print(f"  {r['name']:<15} {r['n_train']:<10} "
                      f"{r['mean_auc']:.4f}  {r['mean_wr']*100:.2f}%  "
                      f"{delta_naive*100:+.2f}pp    {r['std_wr']*100:.2f}pp")

            # Recency vs random at equal N
            recent = [r for r in result["results"] if r["name"] == "Recent-N"][0]
            random_r = [r for r in result["results"] if r["name"] == "Random-N"][0]
            oldest = [r for r in result["results"] if r["name"] == "Oldest-N"][0]

            rec_vs_rand = recent["mean_wr"] - random_r["mean_wr"]
            rec_vs_old = recent["mean_wr"] - oldest["mean_wr"]
            print(f"\n  Recent vs Random (equal N): {rec_vs_rand*100:+.2f}pp")
            print(f"  Recent vs Oldest (equal N): {rec_vs_old*100:+.2f}pp")

            if abs(rec_vs_rand) < 0.5 / 100:
                print(f"  → EQUAL: recency effect < 0.5pp at equal N")
            elif rec_vs_rand > 0.5 / 100:
                print(f"  → RECENCY HELPS at equal N (+{rec_vs_rand*100:.2f}pp)")
            else:
                print(f"  → RECENCY HURTS at equal N ({rec_vs_rand*100:.2f}pp)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: FORWARD RECONCILIATION
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 2: FORWARD RECONCILIATION")
    print("Does production model's anti-predictiveness vary over time?")
    print("If early-forward is predictive but late-forward isn't → staleness")
    print("If uniformly anti-predictive → recency won't help")
    print(f"{'='*70}")

    fwd = _forward_recency_check()
    if fwd:
        print(f"\n  {'Period':<18} {'N':<6} {'WR%':<8} {'BE%':<8} "
              f"{'ROI%':<8} {'Fav WR':<8} {'Unfav WR':<8} {'Disc':<8} {'Dates'}")
        for label in ["Early forward", "Late forward", "All forward"]:
            if label not in fwd:
                continue
            r = fwd[label]
            print(f"  {label:<18} {r['n']:<6} {r['wr']*100:.1f}%   "
                  f"{r['be']*100:.1f}%   {r['roi']:+.1f}%   "
                  f"{r['fav_wr']*100:.1f}%   {r['unfav_wr']*100:.1f}%   "
                  f"{r['disc']*100:+.1f}pp  {r['date_range'][0]}–{r['date_range'][1]}")

        early = fwd.get("Early forward", {})
        late = fwd.get("Late forward", {})
        if early and late:
            print(f"\n  Early discrimination: {early.get('disc', 0)*100:+.1f}pp")
            print(f"  Late discrimination:  {late.get('disc', 0)*100:+.1f}pp")
            if early.get("disc", 0) > 0.02 and late.get("disc", 0) < -0.02:
                print(f"  → STALENESS SIGNAL: model was predictive early, anti-predictive late")
                print(f"    Recency retraining might restore early-period performance")
            elif early.get("disc", 0) < -0.02 and late.get("disc", 0) < -0.02:
                print(f"  → UNIFORMLY ANTI-PREDICTIVE: no time-based pattern")
                print(f"    Recency won't help — the model is fundamentally mispriced")
            else:
                print(f"  → NOISY: discrimination bounces around zero")
                print(f"    No clear staleness vs efficiency signal in forward data")
    else:
        print("  Insufficient forward data for time-based analysis")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: CAN WE TEST RECENCY FORWARD? (honest scoping)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 3: FORWARD RECENCY — WHAT WE CAN AND CAN'T TEST")
    print(f"{'='*70}")

    print("""
  WHAT WE CAN'T DO:
    - We have NO forward shadow data from a recency-trained model
    - The production model was trained on ALL data, not recent-only
    - Walk-forward backtests are artifact-prone (per this session's audit)
    - Any walk-forward "recency helps" finding could be the same 1-year
      persistence artifact that made spreads look +9pp

  WHAT WE CAN DO:
    - Equal-N walk-forward (Part 1) controls sample size within the artifact
    - Forward time-split (Part 2) checks if staleness shows in forward data
    - To TRULY test recency forward: deploy a recency-trained shadow arm
      alongside the production arm and accumulate 30+ settled days

  THE HONEST ANSWER:
    Recency has never been tested without confounds. Equal-N walk-forward
    (Part 1 above) is the cleanest test available today, but it's still
    within the artifact-prone walk-forward framework. True forward validation
    requires deploying a recency-trained shadow arm.""")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
