#!/usr/bin/env python3
"""
lineup_diagnostic.py — Resolve the chrono-vs-random disagreement, diagnose
the extreme-subset collapse, test alternative constructions, and check
WR-vs-breakeven on the deployable set.

Four investigations:
  1. CHRONO-vs-RANDOM: many temporal cuts + 20-seed random
  2. EXTREME-SUBSET AUTOPSY: why Q90 collapses
  3. ALTERNATIVE CONSTRUCTIONS: better gap framings
  4. WR-vs-BREAKEVEN: does AUC translate to real money?
"""
from __future__ import annotations

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DB_HIST = os.path.join(os.path.dirname(__file__), "data", "historical.db")

V1_PRUNED = [
    "sp_era_diff", "ops_diff", "opp_strength_diff",
    "h_opp_impl_prob_5g", "a_opp_impl_prob_5g",
    "h_sp_era", "sp_whip_diff", "sp_kbb_diff",
    "a_sp_whip_3", "a_sp_era_5", "sp_era5_diff",
    "a_sp_era", "h_sp_era_3", "h_sp_era_5",
    "h_bb_pct", "margin_diff_10g", "def_diff_10g",
    "a_margin_10g", "a_bb_pct", "a_ops", "a_slg",
]


def train_and_eval(X_train, y_train, X_test, y_test, seed=42):
    gbc = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=10, random_state=seed,
    )
    gbc.fit(X_train, y_train)
    probs = gbc.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return {"auc": auc, "probs": probs, "model": gbc}


def get_Xy(df, features, target_col):
    X = df[features].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — reuse build_lineup_features from the main test
# ═══════════════════════════════════════════════════════════════════════════════

def load_merged_data():
    """Load and merge lineup features with ATS feature matrix."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
    from features import build_feature_matrix

    # Import build_lineup_features from lineup_composition_test
    from lineup_composition_test import build_lineup_features

    print("[DATA] Building lineup features...")
    lineup = build_lineup_features()

    print("[DATA] Loading ATS feature matrix...")
    fm = build_feature_matrix("MLB")

    lineup_cols = ["game_id", "h_lineup_gap", "a_lineup_gap",
                   "lineup_gap_diff", "h_lineup_ops", "a_lineup_ops",
                   "lineup_ops_diff", "h_lineup_std", "a_lineup_std",
                   "h_mean_cum_pa", "a_mean_cum_pa"]
    # game_pk may or may not be present
    if "game_pk" in lineup.columns:
        lineup_cols.append("game_pk")
    merged = fm.merge(lineup[lineup_cols], on="game_id", how="inner")
    target_col = "covered_spread"
    merged = merged[merged[target_col].notna()].copy()
    base_features = [f for f in V1_PRUNED if f in merged.columns]

    # Sort by date, dedup
    merged = merged.sort_values("game_date").reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["game_id"], keep="first")

    # Fill NaN in features
    all_cols = base_features + ["lineup_gap_diff"]
    for f in all_cols:
        if f in merged.columns:
            merged[f] = merged[f].fillna(merged[f].median())

    print(f"[DATA] Ready: {len(merged)} games, {len(base_features)} base features")
    return merged, base_features, target_col


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 1: CHRONO-vs-RANDOM DISAGREEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def investigate_chrono_vs_random(valid, base_features, target_col):
    print("\n" + "=" * 70)
    print("INVESTIGATION 1: CHRONO-vs-RANDOM DISAGREEMENT")
    print("=" * 70)

    lineup_feats = ["lineup_gap_diff"]
    all_features = base_features + lineup_feats
    n = len(valid)

    # ── A. Many chrono cut points ─────────────────────────────────────────────
    print("\n--- A. Multiple chrono cut points (10 seeds each) ---")
    cut_fractions = [0.30, 0.40, 0.50, 0.60, 0.70]
    chrono_results = []

    for frac in cut_fractions:
        cut = int(n * frac)
        train_df = valid.iloc[:cut]
        test_df = valid.iloc[cut:]

        deltas = []
        for seed in range(10):
            X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
            X_tr_l, _ = get_Xy(train_df, all_features, target_col)
            X_te_b, y_te = get_Xy(test_df, base_features, target_col)
            X_te_l, _ = get_Xy(test_df, all_features, target_col)

            rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
            rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)
            deltas.append(rl["auc"] - rb["auc"])

        mean_d = np.mean(deltas)
        wins = sum(d > 0 for d in deltas)
        chrono_results.append((frac, mean_d, wins, len(test_df)))
        print(f"  {int(frac*100)}/{int((1-frac)*100)} split "
              f"(train={cut}, test={n-cut}): "
              f"delta={mean_d:+.4f}, wins={wins}/10")

    # Summary
    positive_cuts = sum(1 for _, d, _, _ in chrono_results if d > 0)
    print(f"\n  Positive at {positive_cuts}/{len(chrono_results)} cut points")

    # ── B. Quarterly chrono windows ───────────────────────────────────────────
    print("\n--- B. Quarterly expanding-window chrono ---")
    # Split into ~6 blocks, train on expanding window, test on next block
    block_size = n // 6
    for i in range(1, 6):
        train_end = block_size * i
        test_end = min(block_size * (i + 1), n)
        train_df = valid.iloc[:train_end]
        test_df = valid.iloc[train_end:test_end]

        if len(test_df) < 200:
            continue

        deltas = []
        for seed in range(10):
            X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
            X_tr_l, _ = get_Xy(train_df, all_features, target_col)
            X_te_b, y_te = get_Xy(test_df, base_features, target_col)
            X_te_l, _ = get_Xy(test_df, all_features, target_col)

            rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
            rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)
            deltas.append(rl["auc"] - rb["auc"])

        date_range = f"{valid.iloc[train_end]['game_date']} to {valid.iloc[test_end-1]['game_date']}"
        print(f"  Block {i} ({date_range}, N={len(test_df)}): "
              f"delta={np.mean(deltas):+.4f}, wins={sum(d > 0 for d in deltas)}/10")

    # ── C. 20-seed random splits ──────────────────────────────────────────────
    print("\n--- C. 20-seed random 50/50 splits ---")
    random_deltas = []
    for seed in range(20):
        rng = np.random.RandomState(seed + 100)
        gids = valid["game_id"].unique()
        rng.shuffle(gids)
        mid_g = len(gids) // 2
        tr_r = valid[valid["game_id"].isin(set(gids[:mid_g]))]
        te_r = valid[valid["game_id"].isin(set(gids[mid_g:]))]

        X_tr_b, y_tr = get_Xy(tr_r, base_features, target_col)
        X_tr_l, _ = get_Xy(tr_r, all_features, target_col)
        X_te_b, y_te = get_Xy(te_r, base_features, target_col)
        X_te_l, _ = get_Xy(te_r, all_features, target_col)

        rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
        rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)
        d = rl["auc"] - rb["auc"]
        random_deltas.append(d)
        print(f"  Seed {seed:2d}: delta={d:+.4f}")

    random_deltas = np.array(random_deltas)
    print(f"\n  20-seed random mean delta: {random_deltas.mean():+.4f}")
    print(f"  Positive: {(random_deltas > 0).sum()}/20")
    print(f"  Std: {random_deltas.std():.4f}")
    print(f"  Range: [{random_deltas.min():+.4f}, {random_deltas.max():+.4f}]")

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    print("\n--- DIAGNOSIS ---")
    chrono_positive = positive_cuts >= 4
    random_positive = random_deltas.mean() > 0
    random_majority = (random_deltas > 0).sum() > 10

    if chrono_positive and random_positive:
        print("  CONSISTENT POSITIVE: both chrono + random agree → REAL signal")
    elif chrono_positive and not random_positive:
        if random_majority:
            print("  CHRONO FAVORED: chrono robust across cuts, random split "
                  "on sign but majority positive → likely time-structured signal")
        else:
            print("  AMBIGUOUS: chrono positive but random clearly negative → "
                  "could be temporal overfitting or season-specific")
    elif not chrono_positive:
        print("  NOT ROBUST: fails even across multiple chrono cuts → NOT real")

    return chrono_results, random_deltas


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 2: WHY DOES EXTREME SUBSET COLLAPSE?
# ═══════════════════════════════════════════════════════════════════════════════

def investigate_extreme_collapse(valid, base_features, target_col):
    print("\n\n" + "=" * 70)
    print("INVESTIGATION 2: WHY DOES EXTREME SUBSET COLLAPSE?")
    print("=" * 70)

    gap_abs = valid["lineup_gap_diff"].abs()
    q75 = gap_abs.quantile(0.75)
    q90 = gap_abs.quantile(0.90)

    q90_games = valid[gap_abs >= q90].copy()
    normal_games = valid[gap_abs < q75].copy()

    print(f"\n  Q90 extreme games: {len(q90_games)}")
    print(f"  Normal games (< Q75): {len(normal_games)}")

    # ── A. What's different about Q90 games? ─────────────────────────────────
    print("\n--- A. Profile of Q90 extreme-gap games vs normal ---")

    profile_cols = ["ops_diff", "sp_era_diff", "margin_diff_10g",
                    "opp_strength_diff", "h_opp_impl_prob_5g",
                    "lineup_gap_diff", "h_lineup_gap", "a_lineup_gap"]
    available_cols = [c for c in profile_cols if c in valid.columns]

    for col in available_cols:
        q90_mean = q90_games[col].mean()
        norm_mean = normal_games[col].mean()
        q90_std = q90_games[col].std()
        norm_std = normal_games[col].std()
        print(f"  {col:25s}: Q90={q90_mean:+.4f} (std={q90_std:.4f})"
              f"  Normal={norm_mean:+.4f} (std={norm_std:.4f})")

    # ── B. Is Q90 team-concentrated? ──────────────────────────────────────────
    print("\n--- B. Team concentration in Q90 ---")
    if "home_team" in valid.columns:
        q90_home_teams = q90_games["home_team"].value_counts().head(10)
        norm_home_teams = normal_games["home_team"].value_counts().head(10)
        print("  Top 10 home teams in Q90:")
        for team, count in q90_home_teams.items():
            pct = count / len(q90_games) * 100
            print(f"    {team:20s}: {count} ({pct:.1f}%)")
        print(f"\n  Q90 unique home teams: {q90_games['home_team'].nunique()}")
        print(f"  Normal unique home teams: {normal_games['home_team'].nunique()}")

    # ── C. Is Q90 seasonal? ───────────────────────────────────────────────────
    print("\n--- C. Seasonal distribution ---")
    if "season" in valid.columns:
        for label, df in [("Q90", q90_games), ("Normal", normal_games)]:
            dist = df["season"].value_counts().sort_index()
            pcts = (dist / len(df) * 100).round(1)
            print(f"  {label}: {dict(pcts)}")

    # ── D. Is Q90 an early-season artifact? ──────────────────────────────────
    print("\n--- D. Mean batter PA in Q90 vs normal ---")
    for col in ["h_mean_cum_pa", "a_mean_cum_pa"]:
        if col in valid.columns:
            q90_m = q90_games[col].mean()
            norm_m = normal_games[col].mean()
            print(f"  {col}: Q90={q90_m:.1f}  Normal={norm_m:.1f}")

    # ── E. Base rate of covering in Q90 ──────────────────────────────────────
    print("\n--- E. Outcome base rates ---")
    q90_wr = q90_games[target_col].mean()
    norm_wr = normal_games[target_col].mean()
    all_wr = valid[target_col].mean()
    print(f"  Q90  covered_spread rate: {q90_wr:.4f}")
    print(f"  Normal covered_spread rate: {norm_wr:.4f}")
    print(f"  All  covered_spread rate: {all_wr:.4f}")

    # ── F. Does lineup_gap_diff PREDICT correctly in Q90? ─────────────────────
    print("\n--- F. Lineup direction accuracy in Q90 ---")
    # When lineup_gap_diff > 0 (home lineup advantage), does home cover more?
    q90_pos = q90_games[q90_games["lineup_gap_diff"] > 0]
    q90_neg = q90_games[q90_games["lineup_gap_diff"] < 0]
    if len(q90_pos) > 0 and len(q90_neg) > 0:
        print(f"  lineup_gap_diff > 0 (home advantage): "
              f"WR={q90_pos[target_col].mean():.4f} ({len(q90_pos)} games)")
        print(f"  lineup_gap_diff < 0 (away advantage): "
              f"WR={q90_neg[target_col].mean():.4f} ({len(q90_neg)} games)")
        print(f"  Split: {q90_pos[target_col].mean() - q90_neg[target_col].mean():+.4f}")

    # Same for normal games
    norm_pos = normal_games[normal_games["lineup_gap_diff"] > 0]
    norm_neg = normal_games[normal_games["lineup_gap_diff"] < 0]
    if len(norm_pos) > 0 and len(norm_neg) > 0:
        print(f"\n  Normal: gap>0 WR={norm_pos[target_col].mean():.4f} "
              f"({len(norm_pos)}), gap<0 WR={norm_neg[target_col].mean():.4f} "
              f"({len(norm_neg)})")
        print(f"  Normal split: {norm_pos[target_col].mean() - norm_neg[target_col].mean():+.4f}")

    # ── G. Small-N bootstrap on Q90 delta ─────────────────────────────────────
    print("\n--- G. Bootstrap on Q90 AUC delta (is -0.0059 significant?) ---")
    n = len(valid)
    mid = n // 2
    train_df = valid.iloc[:mid]

    # Get Q90 test-half only
    test_half = valid.iloc[mid:]
    test_q90 = test_half[test_half["lineup_gap_diff"].abs() >= q90]
    print(f"  Q90 games in test half: {len(test_q90)}")

    if len(test_q90) > 50:
        all_features = base_features + ["lineup_gap_diff"]
        X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_l, _ = get_Xy(train_df, all_features, target_col)
        X_te_b, y_te = get_Xy(test_q90, base_features, target_col)
        X_te_l, _ = get_Xy(test_q90, all_features, target_col)

        rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, 0)
        rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, 0)

        boot_deltas = []
        rng = np.random.RandomState(42)
        nt = len(y_te)
        for _ in range(1000):
            idx = rng.choice(nt, nt, replace=True)
            try:
                ab = roc_auc_score(y_te[idx], rb["probs"][idx])
                al = roc_auc_score(y_te[idx], rl["probs"][idx])
                boot_deltas.append(al - ab)
            except ValueError:
                pass

        boot_deltas = np.array(boot_deltas)
        print(f"  Q90 bootstrap delta: {boot_deltas.mean():+.4f}")
        print(f"  95% CI: [{np.percentile(boot_deltas, 2.5):+.4f}, "
              f"{np.percentile(boot_deltas, 97.5):+.4f}]")
        print(f"  P(delta > 0): {(boot_deltas > 0).mean()*100:.1f}%")
        ci_width = np.percentile(boot_deltas, 97.5) - np.percentile(boot_deltas, 2.5)
        print(f"  CI width: {ci_width:.4f} (vs full-sample CI ~0.012)")
        if ci_width > 0.04:
            print("  → CI is WIDE — the -0.006 is noise, not a real collapse")
        else:
            print("  → CI is tight enough to trust the sign")


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 3: ALTERNATIVE CONSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def investigate_constructions(valid, base_features, target_col):
    print("\n\n" + "=" * 70)
    print("INVESTIGATION 3: ALTERNATIVE FEATURE CONSTRUCTIONS")
    print("=" * 70)

    n = len(valid)
    mid = n // 2
    train_df = valid.iloc[:mid]
    test_df = valid.iloc[mid:]

    constructions = {}

    # ── A. Current: lineup_gap_diff (gap from team OPS) ───────────────────────
    print("\n--- Testing feature constructions (10-seed chrono each) ---")

    # Construction 1: current lineup_gap_diff
    feat_name = "lineup_gap_diff"
    deltas = _run_construction(train_df, test_df, base_features, target_col, feat_name)
    constructions["gap_diff (current)"] = deltas

    # Construction 2: raw lineup_ops_diff (no gap, just lineup OPS diff)
    feat_name = "lineup_ops_diff"
    deltas = _run_construction(train_df, test_df, base_features, target_col, feat_name)
    constructions["ops_diff (raw)"] = deltas

    # Construction 3: lineup_std spread (dispersion within lineup)
    # Create spread feature: h_lineup_std - a_lineup_std
    if "h_lineup_std" in valid.columns and "a_lineup_std" in valid.columns:
        valid["lineup_std_diff"] = valid["h_lineup_std"] - valid["a_lineup_std"]
        valid["lineup_std_diff"] = valid["lineup_std_diff"].fillna(0)
        train_df = valid.iloc[:mid]
        test_df = valid.iloc[mid:]
        deltas = _run_construction(train_df, test_df, base_features, target_col,
                                   "lineup_std_diff")
        constructions["std_diff (dispersion)"] = deltas

    # Construction 4: gap_diff RESIDUALIZED on ops_diff
    # Regress lineup_gap_diff ~ ops_diff, use residual as feature
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(valid[["ops_diff"]].values, valid["lineup_gap_diff"].values)
    valid["lineup_gap_resid"] = valid["lineup_gap_diff"] - lr.predict(
        valid[["ops_diff"]].values)
    train_df = valid.iloc[:mid]
    test_df = valid.iloc[mid:]
    deltas = _run_construction(train_df, test_df, base_features, target_col,
                               "lineup_gap_resid")
    constructions["gap_resid (ortho to ops_diff)"] = deltas
    corr_resid = valid[["lineup_gap_resid", "ops_diff"]].corr().iloc[0, 1]
    print(f"    corr(lineup_gap_resid, ops_diff) = {corr_resid:.4f}")

    # Construction 5: PA-weighted lineup OPS (not simple mean)
    # Weight each batter's contribution by their cumulative PA
    if "h_mean_cum_pa" in valid.columns:
        # We don't have per-batter weights here; test using cum_pa as a
        # confidence weight on the gap
        valid["lineup_gap_conf"] = valid["lineup_gap_diff"] * np.minimum(
            valid[["h_mean_cum_pa", "a_mean_cum_pa"]].mean(axis=1) / 300, 1.0)
        valid["lineup_gap_conf"] = valid["lineup_gap_conf"].fillna(0)
        train_df = valid.iloc[:mid]
        test_df = valid.iloc[mid:]
        deltas = _run_construction(train_df, test_df, base_features, target_col,
                                   "lineup_gap_conf")
        constructions["gap_conf (PA-weighted)"] = deltas

    # Construction 6: both gap_diff + lineup_std_diff together
    if "lineup_std_diff" in valid.columns:
        combo_feats = ["lineup_gap_diff", "lineup_std_diff"]
        deltas_combo = []
        for seed in range(10):
            bf_combo = base_features + combo_feats
            X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
            X_tr_l, _ = get_Xy(train_df, bf_combo, target_col)
            X_te_b, y_te = get_Xy(test_df, base_features, target_col)
            X_te_l, _ = get_Xy(test_df, bf_combo, target_col)

            rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
            rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)
            deltas_combo.append(rl["auc"] - rb["auc"])
        constructions["gap + std (combo)"] = deltas_combo

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n--- Construction comparison (10-seed chrono, sorted) ---")
    ranked = sorted(constructions.items(), key=lambda x: -np.mean(x[1]))
    for name, ds in ranked:
        m = np.mean(ds)
        wins = sum(d > 0 for d in ds)
        print(f"  {name:30s}: delta={m:+.5f}, wins={wins}/10")

    best_name, best_deltas = ranked[0]
    print(f"\n  Best construction: {best_name} ({np.mean(best_deltas):+.5f})")

    # ── Orthogonality of best construction ────────────────────────────────────
    # Check correlation of best with existing features
    print("\n--- Orthogonality of best vs current ---")
    corr_targets = ["ops_diff", "sp_era_diff", "margin_diff_10g",
                    "opp_strength_diff"]
    for candidate in ["lineup_gap_diff", "lineup_gap_resid", "lineup_ops_diff"]:
        if candidate in valid.columns:
            print(f"\n  {candidate}:")
            for ct in corr_targets:
                if ct in valid.columns:
                    r = valid[[candidate, ct]].dropna().corr().iloc[0, 1]
                    print(f"    corr({ct}) = {r:.3f}")

    return constructions


def _run_construction(train_df, test_df, base_features, target_col, feat_name):
    """Run 10-seed chrono test for a single feature construction."""
    all_features = base_features + [feat_name]
    deltas = []
    for seed in range(10):
        X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_l, _ = get_Xy(train_df, all_features, target_col)
        X_te_b, y_te = get_Xy(test_df, base_features, target_col)
        X_te_l, _ = get_Xy(test_df, all_features, target_col)

        rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
        rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)
        deltas.append(rl["auc"] - rb["auc"])

    m = np.mean(deltas)
    wins = sum(d > 0 for d in deltas)
    print(f"  {feat_name:30s}: delta={m:+.5f}, wins={wins}/10")
    return deltas


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 4: WR-vs-BREAKEVEN DEPLOY TEST
# ═══════════════════════════════════════════════════════════════════════════════

def investigate_wr_vs_be(valid, base_features, target_col):
    print("\n\n" + "=" * 70)
    print("INVESTIGATION 4: WR-vs-BREAKEVEN — DOES AUC MAKE MONEY?")
    print("=" * 70)

    lineup_feats = ["lineup_gap_diff"]
    all_features = base_features + lineup_feats
    n = len(valid)
    mid = n // 2
    train_df = valid.iloc[:mid]
    test_df = valid.iloc[mid:].copy()

    # Standard ATS breakeven at -110 juice
    BE = 1 / 1.91  # ≈ 0.5236

    # ── A. Baseline model WR on test half ─────────────────────────────────────
    print("\n--- A. Flat-bet WR comparison (test half, chrono) ---")

    base_wrs = []
    lineup_wrs = []
    for seed in range(10):
        X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_l, _ = get_Xy(train_df, all_features, target_col)
        X_te_b, y_te = get_Xy(test_df, base_features, target_col)
        X_te_l, _ = get_Xy(test_df, all_features, target_col)

        rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
        rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)

        # "Bet home" when model prob > 0.5, "bet away" when < 0.5
        # WR = fraction where model picks correctly
        base_picks = rb["probs"] > 0.5
        base_correct = (base_picks == y_te.astype(bool))
        base_wr = base_correct.mean()

        lineup_picks = rl["probs"] > 0.5
        lineup_correct = (lineup_picks == y_te.astype(bool))
        lineup_wr = lineup_correct.mean()

        base_wrs.append(base_wr)
        lineup_wrs.append(lineup_wr)

    print(f"  Breakeven: {BE:.4f}")
    print(f"  Base WR:   {np.mean(base_wrs):.4f} +/- {np.std(base_wrs):.4f} "
          f"(edge: {np.mean(base_wrs)-BE:+.4f})")
    print(f"  +Lineup WR: {np.mean(lineup_wrs):.4f} +/- {np.std(lineup_wrs):.4f} "
          f"(edge: {np.mean(lineup_wrs)-BE:+.4f})")
    print(f"  WR delta:  {np.mean(lineup_wrs) - np.mean(base_wrs):+.4f}")

    # ── B. Edge-gated WR (bet only when model is confident) ───────────────────
    print("\n--- B. Edge-gated WR (prob > threshold) ---")
    thresholds = [0.52, 0.54, 0.56, 0.58, 0.60]

    for thr in thresholds:
        base_gated_wrs = []
        lineup_gated_wrs = []
        base_ns = []
        lineup_ns = []

        for seed in range(10):
            X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
            X_tr_l, _ = get_Xy(train_df, all_features, target_col)
            X_te_b, y_te = get_Xy(test_df, base_features, target_col)
            X_te_l, _ = get_Xy(test_df, all_features, target_col)

            rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
            rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)

            # Gate: bet only when max(prob, 1-prob) > threshold
            base_conf = np.maximum(rb["probs"], 1 - rb["probs"])
            lineup_conf = np.maximum(rl["probs"], 1 - rl["probs"])

            base_mask = base_conf >= thr
            lineup_mask = lineup_conf >= thr

            if base_mask.sum() > 0:
                base_picks = rb["probs"][base_mask] > 0.5
                base_correct = (base_picks == y_te[base_mask].astype(bool))
                base_gated_wrs.append(base_correct.mean())
                base_ns.append(base_mask.sum())

            if lineup_mask.sum() > 0:
                lineup_picks = rl["probs"][lineup_mask] > 0.5
                lineup_correct = (lineup_picks == y_te[lineup_mask].astype(bool))
                lineup_gated_wrs.append(lineup_correct.mean())
                lineup_ns.append(lineup_mask.sum())

        if base_gated_wrs and lineup_gated_wrs:
            base_n = int(np.mean(base_ns))
            lineup_n = int(np.mean(lineup_ns))
            print(f"  Threshold {thr:.2f}: "
                  f"Base WR={np.mean(base_gated_wrs):.4f} (N~{base_n}), "
                  f"+Lineup WR={np.mean(lineup_gated_wrs):.4f} (N~{lineup_n}), "
                  f"delta={np.mean(lineup_gated_wrs) - np.mean(base_gated_wrs):+.4f}")

    # ── C. Flat-$10 PnL simulation ───────────────────────────────────────────
    print("\n--- C. Flat-$10 PnL at -110 (test half) ---")
    pnl_base = []
    pnl_lineup = []
    for seed in range(10):
        X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_l, _ = get_Xy(train_df, all_features, target_col)
        X_te_b, y_te = get_Xy(test_df, base_features, target_col)
        X_te_l, _ = get_Xy(test_df, all_features, target_col)

        rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
        rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)

        for probs_arr, label, pnl_list in [
            (rb["probs"], "base", pnl_base),
            (rl["probs"], "lineup", pnl_lineup),
        ]:
            picks = probs_arr > 0.5
            correct = (picks == y_te.astype(bool))
            # At -110: win $9.09, lose $10
            wins_amt = correct.sum() * 9.09
            losses_amt = (~correct).sum() * 10.0
            net = wins_amt - losses_amt
            pnl_list.append(net)

    print(f"  Base PnL:   ${np.mean(pnl_base):+.0f} +/- ${np.std(pnl_base):.0f}")
    print(f"  +Lineup PnL: ${np.mean(pnl_lineup):+.0f} +/- ${np.std(pnl_lineup):.0f}")
    print(f"  PnL delta:  ${np.mean(pnl_lineup) - np.mean(pnl_base):+.0f}")

    # ── D. Per-game expected value calculation ────────────────────────────────
    print("\n--- D. Per-game EV ---")
    n_test = len(test_df)
    print(f"  Test games: {n_test}")
    base_ev = np.mean(pnl_base) / n_test
    lineup_ev = np.mean(pnl_lineup) / n_test
    print(f"  Base EV/game:   ${base_ev:+.2f}")
    print(f"  +Lineup EV/game: ${lineup_ev:+.2f}")
    print(f"  Marginal EV/game from lineup: ${lineup_ev - base_ev:+.2f}")

    return base_wrs, lineup_wrs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("LINEUP COMPOSITION DIAGNOSTIC — Resolving mixed signal")
    print("=" * 70)

    valid, base_features, target_col = load_merged_data()

    # Investigation 1
    chrono_results, random_deltas = investigate_chrono_vs_random(
        valid, base_features, target_col)

    # Investigation 2
    investigate_extreme_collapse(valid, base_features, target_col)

    # Investigation 3
    constructions = investigate_constructions(valid, base_features, target_col)

    # Investigation 4
    base_wrs, lineup_wrs = investigate_wr_vs_be(valid, base_features, target_col)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SYNTHESIS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("FINAL SYNTHESIS — DEPLOY or PARK?")
    print("=" * 70)

    # 1. Chrono robustness
    positive_chrono = sum(1 for _, d, _, _ in chrono_results if d > 0)
    print(f"\n  1. Chrono robustness: {positive_chrono}/{len(chrono_results)} cuts positive")

    # 2. Random verdict
    rand_mean = random_deltas.mean()
    rand_pos = (random_deltas > 0).sum()
    print(f"  2. 20-seed random: mean={rand_mean:+.5f}, {rand_pos}/20 positive")

    # 3. Best construction
    best = max(constructions.items(), key=lambda x: np.mean(x[1]))
    print(f"  3. Best construction: {best[0]} ({np.mean(best[1]):+.5f})")

    # 4. WR edge
    wr_delta = np.mean(lineup_wrs) - np.mean(base_wrs)
    print(f"  4. WR delta: {wr_delta:+.4f}")

    # Decision logic
    chrono_robust = positive_chrono >= 4
    random_positive = rand_mean > 0 and rand_pos >= 10
    wr_positive = wr_delta > 0.003

    print("\n  " + "-" * 60)
    if chrono_robust and random_positive and wr_positive:
        print("  VERDICT: DEPLOY — robust chrono + positive random + WR edge")
    elif chrono_robust and random_positive:
        print("  VERDICT: DEPLOY (cautious) — robust chrono + positive random, "
              "WR gain marginal")
    elif chrono_robust and not random_positive:
        print("  VERDICT: PARK — chrono positive but random doesn't confirm; "
              "could be temporal artifact")
    else:
        print("  VERDICT: KILL — doesn't survive multiple chrono cuts")
    print("  " + "-" * 60)


if __name__ == "__main__":
    main()
