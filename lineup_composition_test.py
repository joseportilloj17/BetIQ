#!/usr/bin/env python3
"""
lineup_composition_test.py — Test whether individual lineup composition
(who's actually batting tonight) adds predictive value beyond team-aggregate
features already in the ATS model.

HYPOTHESIS: Team-level OPS doesn't know that Judge is resting or that the
manager started platoon subs. Lineup composition captures game-specific
batting quality that the model currently misses.

WHY THIS IS ORTHOGONAL (unlike team platoon splits):
- Team platoon died because GBC already had SP ERA/WHIP → implicitly captured
  "how batters do against this pitcher"
- Lineup composition is genuinely NEW: model has team-avg OPS but not
  "tonight's specific 9 guys vs season-average 9 guys"
- This is the COARSENESS fix: individual-level, not roster-average

Features built:
  lineup_ops_gap_h: home lineup quality - home team season OPS (point-in-time, shrunk)
  lineup_ops_gap_a: away lineup quality - away team season OPS
  lineup_ops_gap_diff: home gap - away gap (positive = home lineup upgrade relative to team)

Data requirements:
  - game_lineups table (from backfill_lineups.py)
  - batter_game_stats table (from backfill_lineups.py)
  - ATS feature matrix (from features.py)
"""
from __future__ import annotations

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

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


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Build point-in-time lineup composition features
# ═══════════════════════════════════════════════════════════════════════════════

def build_lineup_features() -> pd.DataFrame:
    """
    Build point-in-time lineup composition features from batter_game_stats.

    For each game:
      1. Identify the 9 starters from game_lineups
      2. For each starter, compute their cumulative season-to-date OPS
         (PRIOR games only, Bayesian shrinkage)
      3. Lineup quality = mean of starter cumulative OPS values
      4. Gap = lineup quality - team cumulative OPS (from team_game_batting)

    Returns DataFrame with game_pk, lineup features.
    """
    conn = sqlite3.connect(DB_HIST)

    # ── 1. Load batter game stats ─────────────────────────────────────────────
    bgs = pd.read_sql_query("""
        SELECT bgs.game_pk, bgs.player_id, bgs.player_name, bgs.team_name,
               bgs.is_home, bgs.ab, bgs.hits, bgs.doubles, bgs.triples,
               bgs.hr, bgs.bb, bgs.hbp, bgs.sf, bgs.k, bgs.tb,
               tgb.game_date, tgb.season
        FROM batter_game_stats bgs
        JOIN team_game_batting tgb ON bgs.game_pk = tgb.game_pk
                                  AND bgs.team_name = tgb.team_name
        ORDER BY bgs.player_id, tgb.season, tgb.game_date
    """, conn)

    print(f"  Batter game stats: {len(bgs)} rows, {bgs['player_id'].nunique()} unique batters")
    print(f"  Games covered: {bgs['game_pk'].nunique()}")

    # ── 2. Load game_lineups to identify the 9 starters ──────────────────────
    lineups = pd.read_sql_query("""
        SELECT gl.game_pk, gl.player_id, gl.team_name, gl.is_home, gl.batting_order
        FROM game_lineups gl
    """, conn)
    print(f"  Lineup rows: {len(lineups)}")

    # Mark which batter-game rows are starters (in the batting order)
    starter_keys = lineups[["game_pk", "player_id"]].drop_duplicates()
    starter_keys["is_starter"] = True
    bgs = bgs.merge(starter_keys, on=["game_pk", "player_id"], how="left")
    bgs["is_starter"] = bgs["is_starter"].fillna(False)
    starters_only = bgs[bgs["is_starter"]].copy()
    print(f"  Starter batter-game rows: {len(starters_only)}")

    # ── 3. Compute per-batter cumulative OPS (point-in-time, shrinkage) ──────
    # For each batter-season, sort by game_date, compute cumulative stats
    # PRIOR to each game (shift by 1)

    # Shrinkage: league average OPS by season
    league_ops = (
        bgs.groupby("season")
        .agg(tot_h=("hits", "sum"), tot_ab=("ab", "sum"),
             tot_tb=("tb", "sum"), tot_bb=("bb", "sum"),
             tot_hbp=("hbp", "sum"), tot_sf=("sf", "sum"))
        .reset_index()
    )
    league_ops["lg_ops"] = np.where(
        league_ops["tot_ab"] > 0,
        (league_ops["tot_h"] + league_ops["tot_bb"] + league_ops["tot_hbp"]) /
        (league_ops["tot_ab"] + league_ops["tot_bb"] + league_ops["tot_hbp"] + league_ops["tot_sf"])
        + league_ops["tot_tb"] / league_ops["tot_ab"],
        0.720
    )
    lg_ops_map = dict(zip(league_ops["season"], league_ops["lg_ops"]))
    print(f"\n  League-avg OPS by season: {dict((s, round(v, 3)) for s, v in lg_ops_map.items())}")

    # PRIOR_PA for shrinkage: 150 PA ≈ ~37 games
    PRIOR_PA = 150

    # Sort and compute cumulative stats per batter-season
    bgs_sorted = bgs.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)

    # Cumulative stats shifted by 1 game (point-in-time: sees only prior games)
    cum_cols = ["ab", "hits", "tb", "bb", "hbp", "sf"]
    for col in cum_cols:
        bgs_sorted[f"cum_{col}"] = bgs_sorted.groupby(["player_id", "season"])[col].transform(
            lambda x: x.cumsum().shift(1)
        )

    # Compute cumulative OPS with shrinkage
    bgs_sorted["cum_pa"] = (
        bgs_sorted["cum_ab"].fillna(0) + bgs_sorted["cum_bb"].fillna(0) +
        bgs_sorted["cum_hbp"].fillna(0) + bgs_sorted["cum_sf"].fillna(0)
    )

    # Raw OPS components
    denom = bgs_sorted["cum_ab"].fillna(0) + bgs_sorted["cum_bb"].fillna(0) + \
            bgs_sorted["cum_hbp"].fillna(0) + bgs_sorted["cum_sf"].fillna(0)
    raw_obp = np.where(
        denom > 0,
        (bgs_sorted["cum_hits"].fillna(0) + bgs_sorted["cum_bb"].fillna(0) +
         bgs_sorted["cum_hbp"].fillna(0)) / denom,
        0
    )
    raw_slg = np.where(
        bgs_sorted["cum_ab"].fillna(0) > 0,
        bgs_sorted["cum_tb"].fillna(0) / bgs_sorted["cum_ab"].fillna(1),
        0
    )
    raw_ops = raw_obp + raw_slg

    # Bayesian shrinkage toward league average
    lg_prior = bgs_sorted["season"].map(lg_ops_map).fillna(0.720)
    cum_pa = bgs_sorted["cum_pa"].fillna(0)
    bgs_sorted["batter_ops_pit"] = np.where(
        cum_pa > 0,
        (cum_pa * raw_ops + PRIOR_PA * lg_prior) / (cum_pa + PRIOR_PA),
        lg_prior  # no prior data → pure league average
    )

    # ── 4. For each game, compute lineup-averaged OPS ─────────────────────────
    # Filter to starters only
    starter_stats = bgs_sorted[bgs_sorted["is_starter"] == True].copy()

    lineup_quality = (
        starter_stats.groupby(["game_pk", "team_name", "is_home"])
        .agg(
            lineup_ops=("batter_ops_pit", "mean"),
            lineup_ops_std=("batter_ops_pit", "std"),
            n_starters=("player_id", "count"),
            min_batter_ops=("batter_ops_pit", "min"),
            max_batter_ops=("batter_ops_pit", "max"),
            mean_cum_pa=("cum_pa", "mean"),
        )
        .reset_index()
    )
    print(f"\n  Lineup quality rows: {len(lineup_quality)}")
    print(f"  n_starters distribution: {lineup_quality['n_starters'].value_counts().to_dict()}")

    # ── 5. Compute team cumulative OPS (for the gap) ──────────────────────────
    tgb = pd.read_sql_query("""
        SELECT team_name, season, game_date, game_pk, is_home,
               at_bats, hits, total_bases, walks, hit_by_pitch, sac_flies
        FROM team_game_batting
        ORDER BY team_name, season, game_date
    """, conn)
    conn.close()

    # Cumulative team OPS (point-in-time, shifted)
    team_cum_cols = {"at_bats": "t_ab", "hits": "t_h", "total_bases": "t_tb",
                     "walks": "t_bb", "hit_by_pitch": "t_hbp", "sac_flies": "t_sf"}
    for src, dst in team_cum_cols.items():
        tgb[dst] = tgb.groupby(["team_name", "season"])[src].transform(
            lambda x: x.cumsum().shift(1)
        )

    t_denom = tgb["t_ab"].fillna(0) + tgb["t_bb"].fillna(0) + \
              tgb["t_hbp"].fillna(0) + tgb["t_sf"].fillna(0)
    t_obp = np.where(
        t_denom > 0,
        (tgb["t_h"].fillna(0) + tgb["t_bb"].fillna(0) + tgb["t_hbp"].fillna(0)) / t_denom,
        0
    )
    t_slg = np.where(
        tgb["t_ab"].fillna(0) > 0,
        tgb["t_tb"].fillna(0) / tgb["t_ab"].fillna(1),
        0
    )
    t_pa = t_denom
    lg_prior_t = tgb["season"].map(lg_ops_map).fillna(0.720)
    tgb["team_ops_pit"] = np.where(
        t_pa > 0,
        (t_pa * (t_obp + t_slg) + PRIOR_PA * lg_prior_t) / (t_pa + PRIOR_PA),
        lg_prior_t
    )

    # ── 6. Merge and compute gap ──────────────────────────────────────────────
    tgb_slim = tgb[["game_pk", "team_name", "is_home", "team_ops_pit"]].copy()
    merged = lineup_quality.merge(tgb_slim, on=["game_pk", "team_name", "is_home"], how="inner")

    # Gap = lineup OPS - team cumulative OPS
    # Positive gap = tonight's lineup is BETTER than the team's season average
    merged["lineup_gap"] = merged["lineup_ops"] - merged["team_ops_pit"]

    print(f"\n  Merged rows: {len(merged)}")
    print(f"  lineup_gap stats: mean={merged['lineup_gap'].mean():.4f}, "
          f"std={merged['lineup_gap'].std():.4f}, "
          f"min={merged['lineup_gap'].min():.4f}, max={merged['lineup_gap'].max():.4f}")

    # ── 7. Pivot to game level ────────────────────────────────────────────────
    home = merged[merged["is_home"] == 1][
        ["game_pk", "lineup_ops", "lineup_gap", "lineup_ops_std",
         "min_batter_ops", "max_batter_ops", "mean_cum_pa"]
    ].rename(columns={
        "lineup_ops": "h_lineup_ops",
        "lineup_gap": "h_lineup_gap",
        "lineup_ops_std": "h_lineup_std",
        "min_batter_ops": "h_min_batter_ops",
        "max_batter_ops": "h_max_batter_ops",
        "mean_cum_pa": "h_mean_cum_pa",
    })

    away = merged[merged["is_home"] == 0][
        ["game_pk", "lineup_ops", "lineup_gap", "lineup_ops_std",
         "min_batter_ops", "max_batter_ops", "mean_cum_pa"]
    ].rename(columns={
        "lineup_ops": "a_lineup_ops",
        "lineup_gap": "a_lineup_gap",
        "lineup_ops_std": "a_lineup_std",
        "min_batter_ops": "a_min_batter_ops",
        "max_batter_ops": "a_max_batter_ops",
        "mean_cum_pa": "a_mean_cum_pa",
    })

    game_level = home.merge(away, on="game_pk", how="inner")
    game_level["lineup_gap_diff"] = game_level["h_lineup_gap"] - game_level["a_lineup_gap"]
    game_level["lineup_ops_diff"] = game_level["h_lineup_ops"] - game_level["a_lineup_ops"]
    game_level["game_id"] = "MLB_" + game_level["game_pk"].astype(str)

    print(f"\n  Game-level features: {len(game_level)} games")
    fill = game_level["lineup_gap_diff"].notna().sum()
    print(f"  lineup_gap_diff fill: {fill}/{len(game_level)} ({fill/len(game_level)*100:.1f}%)")

    # ── 8. Verify point-in-time: same team should have different values ───────
    # Check NYY 2025
    nyy_home = merged[(merged["game_pk"].isin(
        tgb[(tgb["team_name"] == "New York Yankees") & (tgb["season"] == "2025")]["game_pk"]
    )) & (merged["is_home"] == 1)]
    if len(nyy_home) > 5:
        vals = nyy_home["lineup_ops"].values[:10]
        print(f"\n  LEAKAGE CHECK — NYY home lineup OPS first 10 games:")
        print(f"    {[round(v, 4) for v in vals]}")
        uniq = len(np.unique(np.round(vals, 4)))
        print(f"    Unique values: {uniq}/{len(vals)}")
        if uniq > 1:
            print("    PASS — values vary (different lineups + cumulative update)")
        else:
            print("    FAIL — constant (LEAKAGE)")

    # ── 9. Variance check: does lineup_gap actually vary meaningfully? ────────
    print(f"\n  lineup_gap_diff std: {game_level['lineup_gap_diff'].std():.4f}")
    print(f"  lineup_gap_diff IQR: [{game_level['lineup_gap_diff'].quantile(0.25):.4f}, "
          f"{game_level['lineup_gap_diff'].quantile(0.75):.4f}]")

    # How much does lineup composition vary from team average?
    big_gap = (game_level["h_lineup_gap"].abs() > 0.020) | (game_level["a_lineup_gap"].abs() > 0.020)
    print(f"  Games with |lineup_gap| > 0.020: {big_gap.sum()}/{len(game_level)} "
          f"({big_gap.mean()*100:.1f}%)")

    return game_level


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Model training and evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_eval(X_train, y_train, X_test, y_test, seed=42):
    gbc = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=10, random_state=seed,
    )
    gbc.fit(X_train, y_train)
    probs = gbc.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    return {"auc": auc, "brier": brier, "probs": probs, "model": gbc}


def get_Xy(df, features, target_col):
    X = df[features].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Full gauntlet + orthogonality test
# ═══════════════════════════════════════════════════════════════════════════════

def run_gauntlet(merged, base_features, target_col, lineup_feats):
    all_features = base_features + lineup_feats

    merged = merged.sort_values("game_date").reset_index(drop=True)
    valid = merged.dropna(subset=lineup_feats, how="all").copy()
    for f in all_features:
        if f in valid.columns:
            valid[f] = valid[f].fillna(valid[f].median())
    valid = valid.drop_duplicates(subset=["game_id"], keep="first")
    print(f"\n  Valid rows: {len(valid)}")

    n = len(valid)
    mid = n // 2
    results = {"chrono": {"base": [], "plat": []},
               "tscv": {"base": [], "plat": []},
               "random": {"base": [], "plat": []}}

    # ── TEST 1: 10-seed chrono 50/50 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: 10-seed chrono 50/50 split")
    print("=" * 70)
    train_df = valid.iloc[:mid]
    test_df = valid.iloc[mid:]

    for seed in range(10):
        X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_l, _ = get_Xy(train_df, all_features, target_col)
        X_te_b, y_te = get_Xy(test_df, base_features, target_col)
        X_te_l, _ = get_Xy(test_df, all_features, target_col)

        rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
        rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)
        results["chrono"]["base"].append(rb["auc"])
        results["chrono"]["plat"].append(rl["auc"])

    bm = np.mean(results["chrono"]["base"])
    lm = np.mean(results["chrono"]["plat"])
    print(f"  Baseline AUC: {bm:.4f} +/- {np.std(results['chrono']['base']):.4f}")
    print(f"  +Lineup  AUC: {lm:.4f} +/- {np.std(results['chrono']['plat']):.4f}")
    print(f"  Delta: {lm - bm:+.4f}")
    wins = sum(l > b for l, b in zip(results["chrono"]["plat"], results["chrono"]["base"]))
    print(f"  Lineup wins: {wins}/10 seeds")

    # ── TEST 2: TimeSeriesSplit 5-fold CV ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: TimeSeriesSplit 5-fold CV")
    print("=" * 70)
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(valid)):
        tr_cv = valid.iloc[tr_idx]
        te_cv = valid.iloc[te_idx]
        X_tr_b, y_tr = get_Xy(tr_cv, base_features, target_col)
        X_tr_l, _ = get_Xy(tr_cv, all_features, target_col)
        X_te_b, y_te = get_Xy(te_cv, base_features, target_col)
        X_te_l, _ = get_Xy(te_cv, all_features, target_col)

        rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, 42)
        rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, 42)
        results["tscv"]["base"].append(rb["auc"])
        results["tscv"]["plat"].append(rl["auc"])
        print(f"  Fold {fold}: base={rb['auc']:.4f}, +lineup={rl['auc']:.4f}, "
              f"delta={rl['auc'] - rb['auc']:+.4f}")

    print(f"  Mean: base={np.mean(results['tscv']['base']):.4f}, "
          f"+lineup={np.mean(results['tscv']['plat']):.4f}, "
          f"delta={np.mean(results['tscv']['plat']) - np.mean(results['tscv']['base']):+.4f}")

    # ── TEST 3: 5-seed random 50/50 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: 5-seed random 50/50 splits")
    print("=" * 70)
    for seed in range(5):
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
        results["random"]["base"].append(rb["auc"])
        results["random"]["plat"].append(rl["auc"])
        print(f"  Seed {seed}: base={rb['auc']:.4f}, +lineup={rl['auc']:.4f}, "
              f"delta={rl['auc'] - rb['auc']:+.4f}")

    print(f"  Mean: base={np.mean(results['random']['base']):.4f}, "
          f"+lineup={np.mean(results['random']['plat']):.4f}, "
          f"delta={np.mean(results['random']['plat']) - np.mean(results['random']['base']):+.4f}")

    # ── TEST 4: Bootstrap CI ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: Bootstrap CI (1000 resamples)")
    print("=" * 70)
    X_tr_b, y_tr = get_Xy(train_df, base_features, target_col)
    X_tr_l, _ = get_Xy(train_df, all_features, target_col)
    X_te_b, y_te = get_Xy(test_df, base_features, target_col)
    X_te_l, _ = get_Xy(test_df, all_features, target_col)

    rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, 0)
    rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, 0)

    deltas = []
    rng = np.random.RandomState(999)
    nt = len(y_te)
    for _ in range(1000):
        idx = rng.choice(nt, nt, replace=True)
        try:
            ab = roc_auc_score(y_te[idx], rb["probs"][idx])
            al = roc_auc_score(y_te[idx], rl["probs"][idx])
            deltas.append(al - ab)
        except ValueError:
            pass

    deltas = np.array(deltas)
    print(f"  AUC delta: {np.mean(deltas):+.4f}")
    print(f"  95% CI: [{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]")
    print(f"  P(delta > 0): {(deltas > 0).mean() * 100:.1f}%")

    # ── TEST 5: ORTHOGONALITY — Feature importance ────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5: ORTHOGONALITY — Feature importance in combined model")
    print("=" * 70)
    # Train the combined model and check if lineup features get non-trivial importance
    X_tr_l, y_tr = get_Xy(train_df, all_features, target_col)
    rl_full = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, 42)

    importances = rl_full["model"].feature_importances_
    feat_imp = sorted(zip(all_features, importances), key=lambda x: -x[1])

    print("  Top 10 features by importance:")
    for fname, imp in feat_imp[:10]:
        marker = " *** LINEUP" if fname in lineup_feats else ""
        print(f"    {fname:30s}: {imp:.4f}{marker}")

    # Where do lineup features rank?
    print("\n  Lineup feature rankings:")
    for fname in lineup_feats:
        rank = [f for f, _ in feat_imp].index(fname) + 1
        imp = dict(feat_imp)[fname]
        print(f"    {fname:30s}: rank {rank}/{len(all_features)}, importance={imp:.4f}")

    # ── TEST 6: Correlation with existing features ────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 6: Correlation of lineup features with existing features")
    print("=" * 70)
    # Check if lineup_gap_diff is redundant with ops_diff
    corr_pairs = [
        ("lineup_gap_diff", "ops_diff"),
        ("lineup_ops_diff", "ops_diff"),
        ("lineup_gap_diff", "sp_era_diff"),
        ("lineup_gap_diff", "margin_diff_10g"),
    ]
    for f1, f2 in corr_pairs:
        if f1 in valid.columns and f2 in valid.columns:
            r = valid[[f1, f2]].dropna().corr().iloc[0, 1]
            print(f"  corr({f1}, {f2}) = {r:.3f}")

    return results, deltas


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Extreme lineup deviation subset
# ═══════════════════════════════════════════════════════════════════════════════

def run_extreme_subset(merged, base_features, target_col, lineup_feats):
    """Test where lineup deviation is largest — where the feature should matter most."""
    all_features = base_features + lineup_feats

    valid = merged.dropna(subset=lineup_feats, how="all").copy()
    for f in all_features:
        if f in valid.columns:
            valid[f] = valid[f].fillna(valid[f].median())
    valid = valid.drop_duplicates(subset=["game_id"], keep="first")
    valid = valid.sort_values("game_date").reset_index(drop=True)

    # Identify games with large lineup deviations
    gap_abs = valid["lineup_gap_diff"].abs()
    q75 = gap_abs.quantile(0.75)
    q90 = gap_abs.quantile(0.90)

    print(f"\n  lineup_gap_diff percentiles: 75th={q75:.4f}, 90th={q90:.4f}")

    for threshold, label in [(q75, "Q75"), (q90, "Q90")]:
        subset = valid[gap_abs >= threshold].copy()
        print(f"\n" + "=" * 70)
        print(f"EXTREME SUBSET ({label}): |lineup_gap_diff| >= {threshold:.4f} ({len(subset)} games)")
        print("=" * 70)

        mid = len(valid) // 2
        train_all = valid.iloc[:mid]
        test_sub = valid.iloc[mid:]
        test_sub = test_sub[test_sub["lineup_gap_diff"].abs() >= threshold]

        if len(test_sub) < 50:
            print(f"  Only {len(test_sub)} games in test subset — skipping")
            continue

        base_aucs = []
        plat_aucs = []
        for seed in range(10):
            X_tr_b, y_tr = get_Xy(train_all, base_features, target_col)
            X_tr_l, _ = get_Xy(train_all, all_features, target_col)
            X_te_b, y_te = get_Xy(test_sub, base_features, target_col)
            X_te_l, _ = get_Xy(test_sub, all_features, target_col)

            rb = train_and_eval(X_tr_b, y_tr, X_te_b, y_te, seed)
            rl = train_and_eval(X_tr_l, y_tr, X_te_l, y_te, seed)
            base_aucs.append(rb["auc"])
            plat_aucs.append(rl["auc"])

        print(f"  Baseline AUC: {np.mean(base_aucs):.4f}")
        print(f"  +Lineup  AUC: {np.mean(plat_aucs):.4f}")
        print(f"  Delta: {np.mean(plat_aucs) - np.mean(base_aucs):+.4f}")
        wins = sum(l > b for l, b in zip(plat_aucs, base_aucs))
        print(f"  Lineup wins: {wins}/10")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("LINEUP COMPOSITION TEST — Individual batter-level, point-in-time")
    print("=" * 70)

    # Step 1: Check data availability
    conn = sqlite3.connect(DB_HIST)
    try:
        cur = conn.execute("SELECT COUNT(DISTINCT game_pk) FROM batter_game_stats")
        n_games = cur.fetchone()[0]
    except sqlite3.OperationalError:
        n_games = 0
    conn.close()

    if n_games < 1000:
        print(f"\nERROR: Only {n_games} games with batter stats. "
              f"Run backfill_lineups.py first.")
        return

    print(f"\n  Games with batter stats: {n_games}")

    # Step 2: Build features
    print("\n[1/4] Building lineup composition features...")
    lineup = build_lineup_features()

    # Step 3: Load feature matrix
    print("\n[2/4] Loading ATS feature matrix...")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
    from features import build_feature_matrix
    fm = build_feature_matrix("MLB")
    print(f"  Feature matrix: {len(fm)} rows")

    # Step 4: Merge
    print("\n[3/4] Merging...")
    merged = fm.merge(
        lineup[["game_id", "h_lineup_gap", "a_lineup_gap", "lineup_gap_diff",
                "h_lineup_ops", "a_lineup_ops", "lineup_ops_diff",
                "h_lineup_std", "a_lineup_std",
                "h_mean_cum_pa", "a_mean_cum_pa"]],
        on="game_id",
        how="inner",
    )
    target_col = "covered_spread"
    merged = merged[merged[target_col].notna()].copy()
    base_features = [f for f in V1_PRUNED if f in merged.columns]
    print(f"  Merged: {len(merged)} rows, {len(base_features)} base features")

    # The key feature: lineup_gap_diff (lineup composition deviation from team avg)
    lineup_feats = ["lineup_gap_diff"]

    # Step 5: Gauntlet
    print("\n[4/4] Running gauntlet...")
    results, boot_deltas = run_gauntlet(merged, base_features, target_col, lineup_feats)

    # Step 6: Extreme subset
    print("\n\n[BONUS] Extreme lineup deviation subset...")
    run_extreme_subset(merged, base_features, target_col, lineup_feats)

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL VERDICT — LINEUP COMPOSITION TEST")
    print("=" * 70)

    chrono_d = np.mean(results["chrono"]["plat"]) - np.mean(results["chrono"]["base"])
    tscv_d = np.mean(results["tscv"]["plat"]) - np.mean(results["tscv"]["base"])
    random_d = np.mean(results["random"]["plat"]) - np.mean(results["random"]["base"])
    boot_mean = np.mean(boot_deltas)
    boot_ci = (np.percentile(boot_deltas, 2.5), np.percentile(boot_deltas, 97.5))
    boot_pct = (boot_deltas > 0).mean() * 100

    print(f"\n  Chrono 50/50 (10 seeds):  delta = {chrono_d:+.4f}")
    print(f"  TimeSeriesSplit (5-fold): delta = {tscv_d:+.4f}")
    print(f"  Random 50/50 (5 seeds):  delta = {random_d:+.4f}")
    print(f"  Bootstrap CI (1000):     delta = {boot_mean:+.4f} [{boot_ci[0]:+.4f}, {boot_ci[1]:+.4f}]")
    print(f"  P(delta > 0):            {boot_pct:.1f}%")

    all_positive = chrono_d > 0 and tscv_d > 0 and random_d > 0
    ci_excludes_zero = boot_ci[0] > 0

    if all_positive and ci_excludes_zero:
        print("\n  VERDICT: LINEUP COMPOSITION SURVIVES — genuinely orthogonal signal")
        print("           Wire lineup fetch into forward pipeline")
    elif boot_pct > 75:
        print("\n  VERDICT: MARGINAL — some signal but not robust across all tests")
    else:
        print("\n  VERDICT: LINEUP COMPOSITION DOES NOT SURVIVE")
        print("           Individual batter quality is already captured by team-level features")


if __name__ == "__main__":
    main()
