#!/usr/bin/env python3
"""
EDGE DECAY CURVE: Weekly resolution of May 26 ATS model edge
from MLB opening day (2026-03-20) through August 5 2026.

Replicates the exact May 26 model (67 features, GBC 500/0.03/d4, isotonic,
NO spread augmentation) and evaluates on weekly windows at game level.

Uses 3 random seeds and averages for robustness.
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── May 26 feature list (exact replica from 97a0d7d) ──────────────────────────
MAY26_FEATURE_COLS = [
    "h_pts_10g", "h_pts_allowed_10g", "h_win_pct_10g", "h_margin_10g",
    "h_pts_3g", "h_pts_allowed_3g", "h_win_pct_3g",
    "a_pts_10g", "a_pts_allowed_10g", "a_win_pct_10g", "a_margin_10g",
    "a_pts_3g", "a_pts_allowed_3g", "a_win_pct_3g",
    "h_rest_days", "h_is_btb", "h_games_7d",
    "a_rest_days", "a_is_btb", "a_games_7d",
    "rest_advantage", "back_to_back_road", "season_phase",
    "pts_diff_10g", "def_diff_10g", "margin_diff_10g", "win_pct_diff_10g",
    "form_diff_3g", "days_since_win_diff", "h_days_since_win", "a_days_since_win",
    "h_opp_impl_prob_5g", "a_opp_impl_prob_5g", "opp_strength_diff",
    "h_star_out", "a_star_out", "h_stars_missing", "a_stars_missing",
    "h_rotation_missing", "a_rotation_missing", "star_out_diff",
    "neutral_site", "ot_so_game",
    "h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip",
    "sp_era_diff", "sp_whip_diff", "h_park_factor",
    "h_sp_era_3", "a_sp_era_3", "h_sp_era_5", "a_sp_era_5",
    "h_sp_whip_3", "a_sp_whip_3", "sp_era3_diff", "sp_era5_diff",
    "h_ops", "a_ops", "ops_diff", "h_slg", "a_slg",
    "h_k_pct", "a_k_pct", "h_bb_pct", "a_bb_pct",
]

GBC_BASE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
)

TARGET_ATS = "covered_spread"
BREAKEVEN = 0.5238
SEEDS = [42, 123, 7]

# ── Week boundaries ────────────────────────────────────────────────────────────
def generate_weeks(start_date: str, end_date: str):
    """Generate weekly (Mon-Sun) windows covering start_date through end_date."""
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    # Align to Monday
    week_start = s - pd.Timedelta(days=s.weekday())  # Monday of the week containing s
    weeks = []
    while week_start <= e:
        week_end = week_start + pd.Timedelta(days=6)  # Sunday
        weeks.append((week_start, week_end))
        week_start += pd.Timedelta(days=7)
    return weeks


def build_matrix():
    from backend.features import build_feature_matrix
    matrix = build_feature_matrix("MLB")
    if matrix.empty:
        raise RuntimeError("No MLB data in historical.db")
    return matrix


def train_model(train_df, feat_cols, seed):
    """Train one May 26 replica model with given seed."""
    X_tr = train_df[feat_cols].values.astype(float)
    y_tr = train_df[TARGET_ATS].values.astype(float)

    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)

    params = {**GBC_BASE_PARAMS, "random_state": seed}
    base = GradientBoostingClassifier(**params)
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    return clf, imp, sc


def predict(clf, imp, sc, feat_cols, test_df):
    """Return predicted class (0/1) for test_df rows."""
    X_te = test_df[feat_cols].values.astype(float)
    X_te_imp = imp.transform(X_te)
    X_te_s = sc.transform(X_te_imp)
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    return (y_prob >= 0.5).astype(int)


def main():
    print("=" * 78)
    print("EDGE DECAY CURVE: May 26 ATS Model — Weekly Resolution")
    print("=" * 78)

    # 1. Build feature matrix
    print("\nBuilding feature matrix...")
    matrix = build_matrix()
    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    labeled["_gd"] = pd.to_datetime(labeled["game_date"])
    print(f"  Total labeled games: {len(labeled)}")
    print(f"  Date range: {labeled['_gd'].min().date()} to {labeled['_gd'].max().date()}")

    # 2. Resolve available features
    feat_cols = [c for c in MAY26_FEATURE_COLS if c in labeled.columns]
    missing = [c for c in MAY26_FEATURE_COLS if c not in labeled.columns]
    if missing:
        print(f"  Missing features (skipped): {missing}")
    print(f"  Available features: {len(feat_cols)}")

    # 3. Train data: everything before 2026-05-26
    train_df = labeled[labeled["_gd"] < "2026-05-26"].copy()
    print(f"  Training rows (pre 2026-05-26): {len(train_df)}")
    print(f"  Train cover rate: {train_df[TARGET_ATS].mean()*100:.1f}%")

    # 4. Train 3 models (different seeds)
    print(f"\nTraining {len(SEEDS)} models (seeds: {SEEDS})...")
    models = []
    for seed in SEEDS:
        clf, imp, sc = train_model(train_df, feat_cols, seed)
        models.append((clf, imp, sc))
        print(f"  Seed {seed}: trained")

    # 5. Generate weekly windows
    weeks = generate_weeks("2026-03-20", "2026-08-05")
    print(f"\nEvaluating {len(weeks)} weekly windows...\n")

    # 6. Evaluate each week
    results = []
    for week_start, week_end in weeks:
        mask = (labeled["_gd"] >= week_start) & (labeled["_gd"] <= week_end)
        week_df = labeled[mask]

        if len(week_df) == 0:
            results.append({
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_end": week_end.strftime("%Y-%m-%d"),
                "n_games": 0,
                "wr": np.nan,
                "edge_pp": np.nan,
            })
            continue

        y_true = week_df[TARGET_ATS].values.astype(float)

        # Average predictions across seeds
        seed_correct = []
        for clf, imp, sc in models:
            y_pred = predict(clf, imp, sc, feat_cols, week_df)
            correct = (y_pred == y_true).sum()
            seed_correct.append(correct)

        avg_correct = np.mean(seed_correct)
        n_games = len(week_df)
        wr = avg_correct / n_games
        edge = (wr - BREAKEVEN) * 100

        results.append({
            "week_start": week_start.strftime("%Y-%m-%d"),
            "week_end": week_end.strftime("%Y-%m-%d"),
            "n_games": n_games,
            "wr": wr,
            "edge_pp": edge,
        })

    # 7. Compute rolling 3-week average and cumulative edge
    df = pd.DataFrame(results)
    valid = df[df["n_games"] > 0].copy()

    # Rolling 3-week WR (smoothed)
    valid["roll3_wr"] = valid["wr"].rolling(3, min_periods=2, center=True).mean()
    valid["roll3_edge"] = (valid["roll3_wr"] - BREAKEVEN) * 100

    # Cumulative WR
    valid["cum_correct"] = (valid["wr"] * valid["n_games"]).cumsum()
    valid["cum_games"] = valid["n_games"].cumsum()
    valid["cum_wr"] = valid["cum_correct"] / valid["cum_games"]
    valid["cum_edge"] = (valid["cum_wr"] - BREAKEVEN) * 100

    # 8. Print the table
    print("=" * 100)
    print(f"{'Week Start':<12} {'Week End':<12} {'N':>4} {'WR%':>7} {'Edge':>8} {'Roll3 WR':>9} {'Roll3 E':>8} {'Cum WR':>8} {'Cum E':>8}")
    print("-" * 100)

    for _, row in valid.iterrows():
        wr_str = f"{row['wr']*100:.1f}%"
        edge_str = f"{row['edge_pp']:+.1f}pp"
        roll3_wr = f"{row['roll3_wr']*100:.1f}%" if not np.isnan(row.get('roll3_wr', np.nan)) else "   --"
        roll3_e = f"{row['roll3_edge']:+.1f}pp" if not np.isnan(row.get('roll3_edge', np.nan)) else "   --"
        cum_wr = f"{row['cum_wr']*100:.1f}%"
        cum_e = f"{row['cum_edge']:+.1f}pp"

        # Mark key thresholds
        marker = ""
        if row["edge_pp"] > 0:
            marker = " +"
        else:
            marker = " -"

        print(f"{row['week_start']:<12} {row['week_end']:<12} {int(row['n_games']):>4} {wr_str:>7} {edge_str:>8} {roll3_wr:>9} {roll3_e:>8} {cum_wr:>8} {cum_e:>8}{marker}")

    # 9. Find crossover point
    print("\n" + "=" * 78)
    print("CROSSOVER ANALYSIS")
    print("=" * 78)

    # Find where rolling 3-week edge crosses from positive to <=0
    roll3_edges = valid[["week_start", "roll3_edge"]].dropna()
    crossover_found = False
    for i in range(1, len(roll3_edges)):
        prev = roll3_edges.iloc[i-1]
        curr = roll3_edges.iloc[i]
        if prev["roll3_edge"] > 0 and curr["roll3_edge"] <= 0:
            print(f"\n  ROLLING 3-WEEK crossover: edge goes negative between")
            print(f"    {prev['week_start']} (edge={prev['roll3_edge']:+.1f}pp)")
            print(f"    {curr['week_start']} (edge={curr['roll3_edge']:+.1f}pp)")
            crossover_found = True
            break

    if not crossover_found:
        # Check if edge was ever positive
        pos = roll3_edges[roll3_edges["roll3_edge"] > 0]
        if len(pos) == 0:
            print("\n  Rolling 3-week edge was NEVER positive.")
        else:
            last_pos = pos.iloc[-1]
            print(f"\n  Rolling 3-week edge last positive at {last_pos['week_start']} ({last_pos['roll3_edge']:+.1f}pp)")
            print(f"  Edge fluctuates rather than monotonically decaying.")

    # Also find raw weekly crossover
    print("\n  RAW WEEKLY edge crossover points (positive -> negative):")
    for i in range(1, len(valid)):
        prev = valid.iloc[i-1]
        curr = valid.iloc[i]
        if prev["edge_pp"] > 0 and curr["edge_pp"] <= 0:
            print(f"    {prev['week_start']} ({prev['edge_pp']:+.1f}pp) -> {curr['week_start']} ({curr['edge_pp']:+.1f}pp)")

    # Cumulative edge crossover
    cum = valid[["week_start", "cum_edge"]].copy()
    print("\n  CUMULATIVE edge crossover (all positive -> first <=0):")
    for i in range(1, len(cum)):
        prev = cum.iloc[i-1]
        curr = cum.iloc[i]
        if prev["cum_edge"] > 0 and curr["cum_edge"] <= 0:
            print(f"    Cumulative edge crosses zero at {curr['week_start']} ({curr['cum_edge']:+.1f}pp)")
            break
    else:
        last = cum.iloc[-1]
        print(f"    Cumulative edge at final week: {last['cum_edge']:+.1f}pp (week of {last['week_start']})")

    # Monthly summary
    print("\n" + "=" * 78)
    print("MONTHLY SUMMARY")
    print("=" * 78)

    valid["_month"] = pd.to_datetime(valid["week_start"]).dt.to_period("M")
    for month, grp in valid.groupby("_month"):
        total_games = int(grp["n_games"].sum())
        weighted_correct = (grp["wr"] * grp["n_games"]).sum()
        month_wr = weighted_correct / total_games if total_games > 0 else np.nan
        month_edge = (month_wr - BREAKEVEN) * 100
        print(f"  {month}: {total_games:>4} games, WR={month_wr*100:.1f}%, edge={month_edge:+.1f}pp")

    # Per-seed variance check
    print("\n" + "=" * 78)
    print("PER-SEED FULL-PERIOD WR (robustness check)")
    print("=" * 78)
    all_test = labeled[labeled["_gd"] >= "2026-03-20"].copy()
    y_true_all = all_test[TARGET_ATS].values.astype(float)
    for seed, (clf, imp, sc) in zip(SEEDS, models):
        y_pred = predict(clf, imp, sc, feat_cols, all_test)
        wr = (y_pred == y_true_all).mean()
        print(f"  Seed {seed:>3}: WR={wr*100:.1f}%, edge={((wr-BREAKEVEN)*100):+.1f}pp (N={len(all_test)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
