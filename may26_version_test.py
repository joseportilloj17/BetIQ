#!/usr/bin/env python3
"""
DECISIVE TEST: Reconstruct the exact May 26 ATS model and test it on
both the winning-stretch period (May 26 - Jun 9) AND current data (Jul 1 - Aug 5).

The May 26 model was:
  - train_diagnostic_sport_model("MLB") from the May 10 commit (97a0d7d)
  - 67 features (MODEL_FEATURE_COLS from that era): rolling + schedule +
    differentials + pitcher ERA/WHIP + batting OPS/SLG/K%/BB%
  - NO spread augmentation (added July 5)
  - NO V1_PRUNED feature selection (added later)
  - GBC: n_estimators=500, lr=0.03, max_depth=4, subsample=0.8, min_samples_leaf=5
  - Isotonic calibration cv=3
  - SimpleImputer(median) + StandardScaler
  - 80/20 chronological split
  - Trained on all MLB games in historical.db with game_date < cutoff

This script:
  1. Builds the feature matrix from historical.db using the SAME feature list
  2. Trains the May 26 model on data available before 2026-05-26
  3. Predicts on two test windows:
     a) May 26 - Jun 9 (the winning stretch)
     b) Jul 1 - Aug 5 (the current/harder period)
  4. Evaluates at GAME LEVEL (dedup: each game-team counts once)
  5. Reports WR vs breakeven, AUC, and Brier for both windows
"""
from __future__ import annotations
import os
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
HIST_DB = os.path.join(BASE, "data", "historical.db")
BETS_DB = os.path.join(BASE, "data", "bets.db")

# ── May 26 feature list (exact replica from 97a0d7d commit) ──────────────────
MAY26_FEATURE_COLS = [
    # Rolling 10g and 3g
    "h_pts_10g", "h_pts_allowed_10g", "h_win_pct_10g", "h_margin_10g",
    "h_pts_3g", "h_pts_allowed_3g", "h_win_pct_3g",
    "a_pts_10g", "a_pts_allowed_10g", "a_win_pct_10g", "a_margin_10g",
    "a_pts_3g", "a_pts_allowed_3g", "a_win_pct_3g",
    # Schedule
    "h_rest_days", "h_is_btb", "h_games_7d",
    "a_rest_days", "a_is_btb", "a_games_7d",
    "rest_advantage", "back_to_back_road", "season_phase",
    # Differentials
    "pts_diff_10g", "def_diff_10g", "margin_diff_10g", "win_pct_diff_10g",
    "form_diff_3g", "days_since_win_diff", "h_days_since_win", "a_days_since_win",
    "h_opp_impl_prob_5g", "a_opp_impl_prob_5g", "opp_strength_diff",
    "h_star_out", "a_star_out", "h_stars_missing", "a_stars_missing",
    "h_rotation_missing", "a_rotation_missing", "star_out_diff",
    # Game-level
    "neutral_site", "ot_so_game",
    "h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip",
    "sp_era_diff", "sp_whip_diff", "h_park_factor",
    "h_sp_era_3", "a_sp_era_3", "h_sp_era_5", "a_sp_era_5",
    "h_sp_whip_3", "a_sp_whip_3", "sp_era3_diff", "sp_era5_diff",
    "h_ops", "a_ops", "ops_diff", "h_slg", "a_slg",
    "h_k_pct", "a_k_pct", "h_bb_pct", "a_bb_pct",
]

# ── May 26 GBC hyperparams (exact replica) ──────────────────────────────────
GBC_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)

TARGET_ATS = "covered_spread"


def build_matrix():
    """Build the feature matrix using the current build_feature_matrix (same logic)."""
    from backend.features import build_feature_matrix
    matrix = build_feature_matrix("MLB")
    if matrix.empty:
        raise RuntimeError("No MLB data in historical.db")
    return matrix


def train_may26_model(matrix: pd.DataFrame, cutoff: str = "2026-05-26"):
    """
    Train the May 26 model exactly as it was trained in May:
    - Use MAY26_FEATURE_COLS (67 features)
    - No spread augmentation
    - Train on all games before cutoff
    - GBC + isotonic calibration
    """
    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    gd = pd.to_datetime(labeled["game_date"])

    # Split: train on everything before cutoff
    train_mask = gd < cutoff
    train_df = labeled[train_mask].copy()

    print(f"\n{'='*70}")
    print(f"TRAINING May 26 model replica")
    print(f"  Cutoff:     {cutoff}")
    print(f"  Train rows: {len(train_df)}")
    print(f"  Features:   {len(MAY26_FEATURE_COLS)} (May MODEL_FEATURE_COLS)")
    print(f"  Augment:    NO (augmentation was added July 5)")
    print(f"{'='*70}")

    # Get features that exist in the matrix
    feat_cols = [c for c in MAY26_FEATURE_COLS if c in train_df.columns]
    missing = [c for c in MAY26_FEATURE_COLS if c not in train_df.columns]
    if missing:
        print(f"  Missing features (will be skipped): {missing}")
    print(f"  Available features: {len(feat_cols)}")

    X_tr = train_df[feat_cols].values.astype(float)
    y_tr = train_df[TARGET_ATS].values.astype(float)

    print(f"  Train cover rate: {y_tr.mean()*100:.1f}%")

    # Impute + scale (exactly as May code did)
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)

    # Train GBC + isotonic calibration (cv=3, exactly as May code)
    base = GradientBoostingClassifier(**GBC_PARAMS)
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    return clf, imp, sc, feat_cols


def predict_window(
    clf, imp, sc, feat_cols, matrix, window_start, window_end, window_name
):
    """
    Predict on a time window and evaluate at GAME LEVEL.
    """
    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    gd = pd.to_datetime(labeled["game_date"])

    test_mask = (gd >= window_start) & (gd <= window_end)
    test_df = labeled[test_mask].copy()

    if len(test_df) == 0:
        print(f"\n  {window_name}: NO DATA in [{window_start}, {window_end}]")
        return None

    X_te = test_df[feat_cols].values.astype(float)
    y_te = test_df[TARGET_ATS].values.astype(float)

    # Transform
    X_te_imp = imp.transform(X_te)
    X_te_s = sc.transform(X_te_imp)

    # Predict
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # --- GAME LEVEL evaluation ---
    # Each row in the matrix IS one game (build_feature_matrix returns one row per game)
    # The model predicts: will HOME TEAM cover the CLOSE SPREAD?
    # For game-level dedup, each game already IS one row.

    n_games = len(test_df)
    n_correct = int((y_pred == y_te).sum())
    wr = n_correct / n_games

    # AUC and Brier
    try:
        auc = roc_auc_score(y_te, y_prob)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_te, y_prob)

    # Breakeven for a typical -110 spread bet = 52.38%
    breakeven = 0.5238

    print(f"\n{'─'*70}")
    print(f"  {window_name}")
    print(f"  Period:     {window_start} to {window_end}")
    print(f"  Games:      {n_games}")
    print(f"  Cover rate: {y_te.mean()*100:.1f}%  (base rate)")
    print(f"  Model WR:   {wr*100:.1f}%  ({n_correct}/{n_games})")
    print(f"  Breakeven:  {breakeven*100:.1f}%")
    print(f"  Edge:       {(wr - breakeven)*100:+.1f}pp {'BEATS BE' if wr > breakeven else 'BELOW BE'}")
    print(f"  AUC:        {auc:.4f}")
    print(f"  Brier:      {brier:.4f}")

    # Calibration buckets
    print(f"\n  Calibration buckets:")
    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred_p = y_prob[mask].mean()
        act_r = y_te[mask].mean()
        gap = (act_r - pred_p) * 100
        print(f"    {int(lo*100):>3d}-{int(hi*100):<3d}%: n={n:>4d}  pred={pred_p*100:.1f}%  actual={act_r*100:.1f}%  gap={gap:+.1f}pp")

    return {
        "window": window_name,
        "n_games": n_games,
        "cover_rate": y_te.mean(),
        "model_wr": wr,
        "breakeven": breakeven,
        "edge_pp": (wr - breakeven) * 100,
        "auc": auc,
        "brier": brier,
    }


def test_with_augmentation(
    clf, imp, sc, feat_cols, matrix, window_start, window_end, window_name
):
    """
    Simulate what the model ACTUALLY did during the winning stretch:
    For each game, augment across spread values and predict cover prob for each.
    This is what the mock_bets pipeline would have done.

    But wait -- the May model had NO augmentation. It predicted:
    "will home team cover the CLOSE SPREAD?" -- one prediction per game.

    So the game-level test IS the right test.
    """
    pass  # Not needed -- see above


def compare_to_actual_bets():
    """
    Cross-reference with actual mock_bet_legs for MLB spread bets.
    """
    conn = sqlite3.connect(BETS_DB)

    print(f"\n{'='*70}")
    print("ACTUAL MOCK BET LEGS — MLB Spread")
    print(f"{'='*70}")

    for period, start, end in [
        ("May 26 - Jun 9", "2026-05-26", "2026-06-09"),
        ("Jul 1 - Aug 5", "2026-07-01", "2026-08-05"),
    ]:
        q = """
        SELECT
            mbl.leg_result,
            mbl.spread_value,
            mbl.description,
            mb.game_date,
            mbl.win_prob,
            mb.model_used,
            -- For game-level dedup we need a game identifier
            -- Extract team name from description to create game-team key
            mbl.fixture_id
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.sport = 'MLB'
          AND mbl.market_type IN ('spreads', 'alternate_spreads', 'Spread')
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND mb.game_date BETWEEN ? AND ?
        """
        df = pd.read_sql_query(q, conn, params=[start, end])

        if df.empty:
            print(f"\n  {period}: NO DATA")
            continue

        # --- LEG LEVEL ---
        leg_wins = (df["leg_result"] == "WIN").sum()
        leg_total = len(df)
        leg_wr = leg_wins / leg_total

        # --- GAME LEVEL dedup ---
        # Create a game-team key: game_date + first part of description (team name)
        # Then take the FIRST leg per game-team (any leg, since same game = same outcome direction)
        df["_team"] = df["description"].str.extract(r'^(.*?)\s+[+-]')[0]
        df["_game_key"] = df["game_date"].astype(str) + "|" + df["_team"].astype(str)

        game_df = df.drop_duplicates(subset=["_game_key"], keep="first")
        game_wins = (game_df["leg_result"] == "WIN").sum()
        game_total = len(game_df)
        game_wr = game_wins / game_total if game_total > 0 else 0

        # Breakeven depends on odds mix -- use average implied prob from spread_value
        # +1.5 typical odds ~= -155 to -180 (implied ~60-64%)
        # For simplicity, use the stored win_prob as breakeven proxy
        avg_stored_wp = df["win_prob"].mean() / 100.0 if df["win_prob"].notna().any() else 0.5238

        # Standard -110 breakeven
        be = 0.5238

        print(f"\n  {period}:")
        print(f"    Leg level:  {leg_wins}W / {leg_total-leg_wins}L = {leg_wr*100:.1f}% WR")
        print(f"    Game level: {game_wins}W / {game_total-game_wins}L = {game_wr*100:.1f}% WR (deduped)")
        print(f"    vs BE (52.4%): {(game_wr-be)*100:+.1f}pp")
        print(f"    Avg stored win_prob: {avg_stored_wp*100:.1f}%")

        # By spread value (rung)
        print(f"\n    By spread rung (game-level dedup):")
        for sv, grp in game_df.groupby("spread_value"):
            w = (grp["leg_result"] == "WIN").sum()
            t = len(grp)
            print(f"      {sv:+5.1f}: {w}W/{t-w}L = {w/t*100:.1f}% (n={t})")

    conn.close()


def main():
    print("=" * 70)
    print("DECISIVE TEST: May 26 ATS Model Version on Current Data")
    print("=" * 70)

    # 1. Build feature matrix
    print("\nStep 1: Building feature matrix from historical.db...")
    matrix = build_matrix()
    gd = pd.to_datetime(matrix["game_date"])
    print(f"  Total MLB games: {len(matrix)}")
    print(f"  Date range: {gd.min().date()} to {gd.max().date()}")
    print(f"  Games with ATS label: {matrix[TARGET_ATS].notna().sum()}")

    # 2. Train the May 26 model
    clf, imp, sc, feat_cols = train_may26_model(matrix, cutoff="2026-05-26")

    # 3. Test on both windows
    results = []

    r1 = predict_window(
        clf, imp, sc, feat_cols, matrix,
        "2026-05-26", "2026-06-09",
        "WINNING STRETCH (May 26 - Jun 9)"
    )
    if r1:
        results.append(r1)

    r2 = predict_window(
        clf, imp, sc, feat_cols, matrix,
        "2026-07-01", "2026-08-05",
        "CURRENT PERIOD (Jul 1 - Aug 5)"
    )
    if r2:
        results.append(r2)

    # Also test the gap period
    r3 = predict_window(
        clf, imp, sc, feat_cols, matrix,
        "2026-06-10", "2026-06-30",
        "GAP PERIOD (Jun 10 - Jun 30)"
    )
    if r3:
        results.append(r3)

    # 4. Also train the CURRENT pruned+augmented model for comparison
    print(f"\n{'='*70}")
    print("COMPARISON: Current V1_PRUNED + Augmented model")
    print(f"{'='*70}")

    from backend.features import MODEL_FEATURE_COLS_V1_PRUNED

    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    gd_l = pd.to_datetime(labeled["game_date"])
    train_mask = gd_l < "2026-05-26"
    train_df = labeled[train_mask].copy()

    # Spread augmentation (exactly as current code does)
    AUGMENT_SPREADS = [-1.5, 1.5, 2.5, 3.5, 4.5]
    margin = (
        pd.to_numeric(train_df["home_score"], errors="coerce")
        - pd.to_numeric(train_df["away_score"], errors="coerce")
    )
    aug_rows = []
    for spread_val in AUGMENT_SPREADS:
        chunk = train_df.copy()
        chunk["_spread_value"] = float(spread_val)
        chunk["_spread_abs"] = abs(spread_val)
        chunk["_is_fav_spread"] = 1.0 if spread_val < 0 else 0.0
        ats_val = margin + spread_val
        chunk[TARGET_ATS] = np.where(ats_val > 0, 1.0,
                            np.where(ats_val < 0, 0.0, np.nan))
        aug_rows.append(chunk)
    train_aug = pd.concat(aug_rows, ignore_index=True)
    train_aug = train_aug[train_aug[TARGET_ATS].notna()].copy()
    train_aug = train_aug.sort_values("game_date").reset_index(drop=True)

    pruned_cols = [c for c in MODEL_FEATURE_COLS_V1_PRUNED if c in train_aug.columns]
    pruned_cols_full = pruned_cols + ["_spread_value", "_spread_abs", "_is_fav_spread"]

    X_aug = train_aug[pruned_cols_full].values.astype(float)
    y_aug = train_aug[TARGET_ATS].values.astype(float)

    imp2 = SimpleImputer(strategy="median")
    X_aug_imp = imp2.fit_transform(X_aug)
    sc2 = StandardScaler()
    X_aug_s = sc2.fit_transform(X_aug_imp)

    base2 = GradientBoostingClassifier(**GBC_PARAMS)
    clf2 = CalibratedClassifierCV(base2, cv=3, method="isotonic")
    clf2.fit(X_aug_s, y_aug)

    print(f"  Trained pruned+augmented model: {len(train_aug)} rows, {len(pruned_cols_full)} features")

    # Test the augmented model on both windows
    # For each test game, we predict at the +1.5 rung (the dominant rung)
    for window_name, ws, we in [
        ("WINNING STRETCH (Pruned+Aug)", "2026-05-26", "2026-06-09"),
        ("CURRENT PERIOD (Pruned+Aug)", "2026-07-01", "2026-08-05"),
    ]:
        test_df = labeled[(gd_l >= ws) & (gd_l <= we)].copy()
        if len(test_df) == 0:
            continue

        # Test at the +1.5 rung (most common)
        test_margin = (
            pd.to_numeric(test_df["home_score"], errors="coerce")
            - pd.to_numeric(test_df["away_score"], errors="coerce")
        )
        test_df["_spread_value"] = 1.5
        test_df["_spread_abs"] = 1.5
        test_df["_is_fav_spread"] = 0.0

        # Ground truth at +1.5: away team +1.5
        ats_15 = test_margin + 1.5
        y_te = np.where(ats_15 > 0, 1.0,
                np.where(ats_15 < 0, 0.0, np.nan))
        valid = ~np.isnan(y_te)
        test_df = test_df[valid].copy()
        y_te = y_te[valid]

        X_te = test_df[pruned_cols_full].values.astype(float)
        X_te_imp = imp2.transform(X_te)
        X_te_s = sc2.transform(X_te_imp)
        y_prob = clf2.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        wr = (y_pred == y_te).mean()
        try:
            auc = roc_auc_score(y_te, y_prob)
        except:
            auc = float("nan")
        brier = brier_score_loss(y_te, y_prob)
        be = 0.5238

        print(f"\n  {window_name}:")
        print(f"    Games (at +1.5): {len(test_df)}")
        print(f"    Cover rate:      {y_te.mean()*100:.1f}%")
        print(f"    Model WR:        {wr*100:.1f}%")
        print(f"    vs BE (52.4%):   {(wr-be)*100:+.1f}pp")
        print(f"    AUC:             {auc:.4f}")
        print(f"    Brier:           {brier:.4f}")

    # 5. Cross-reference with actual bets
    compare_to_actual_bets()

    # 6. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if results:
        for r in results:
            beat = "YES" if r["edge_pp"] > 0 else "NO"
            print(f"  {r['window']}:")
            print(f"    WR={r['model_wr']*100:.1f}%  BE=52.4%  Edge={r['edge_pp']:+.1f}pp  AUC={r['auc']:.4f}  BEATS BE? {beat}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if len(results) >= 2:
        may_edge = results[0]["edge_pp"]
        jul_edge = results[1]["edge_pp"]

        if may_edge > 0 and jul_edge > 0:
            print("  The May 26 model beats breakeven on BOTH periods.")
            print("  This would suggest real edge (not just an easier market window).")
        elif may_edge > 0 and jul_edge <= 0:
            print("  The May 26 model beats breakeven on May/Jun but NOT on Jul/Aug.")
            print("  This was an easier-market window, not model edge.")
        elif may_edge <= 0 and jul_edge <= 0:
            print("  The May 26 model does NOT beat breakeven on EITHER period.")
            print("  The winning streak was variance, not model edge.")
        else:
            print("  Unexpected pattern. Investigate further.")

    print()


if __name__ == "__main__":
    main()
