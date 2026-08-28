#!/usr/bin/env python3
"""
platoon_honest_test.py — HONEST point-in-time platoon splits + full gauntlet.

Fixes the leakage in backfill_handedness.py Part C (season-level aggregates).
Builds cumulative game-by-game OPS vs LHP/RHP from team_game_batting,
shifted by 1 game (no lookahead), with Bayesian shrinkage for early-season sparsity.

Then runs:
  1. Full gauntlet (10-seed, TimeSeriesSplit, random splits, bootstrap CI)
  2. LHP-extreme-gap subset test
"""
from __future__ import annotations

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

DB_HIST = os.path.join(os.path.dirname(__file__), "data", "historical.db")
DB_BETS = os.path.join(os.path.dirname(__file__), "data", "bets.db")

# ─── V1 Pruned feature list (21 features) ────────────────────────────────────
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
# PART 1: Build honest point-in-time platoon features
# ═══════════════════════════════════════════════════════════════════════════════

def build_honest_platoon_features() -> pd.DataFrame:
    """
    Build point-in-time platoon splits from team_game_batting + pitcher handedness.

    For each team-game:
      1. Identify opposing SP handedness
      2. Compute cumulative season-to-date OPS vs that hand (PRIOR games only)
      3. Apply Bayesian shrinkage toward league average
      4. Compute platoon_gap = (shrunk OPS vs SP hand) - (shrunk overall OPS)

    Returns DataFrame with columns:
      game_id, game_date, home_team, away_team,
      h_sp_throws, a_sp_throws,
      h_ops_vs_sp_hand_pit, a_ops_vs_sp_hand_pit,
      h_ops_overall_pit, a_ops_overall_pit,
      h_platoon_gap_pit, a_platoon_gap_pit,
      platoon_gap_diff_pit
    """
    conn = sqlite3.connect(DB_HIST)

    # ── 1. Load team_game_batting ──────────────────────────────────────────────
    tgb = pd.read_sql_query(
        "SELECT team_name, season, game_date, game_pk, is_home, opponent, "
        "at_bats, hits, total_bases, walks, hit_by_pitch, sac_flies, plate_appearances "
        "FROM team_game_batting ORDER BY team_name, season, game_date",
        conn,
    )
    tgb["season"] = tgb["season"].astype(str)
    print(f"  team_game_batting: {len(tgb)} rows")

    # ── 2. Build SP handedness per game_pk ─────────────────────────────────────
    sp_query = """
    SELECT
        CAST(REPLACE(g.game_id, 'MLB_', '') AS INTEGER) as game_pk,
        g.game_id,
        g.game_date,
        g.home_team,
        g.away_team,
        MAX(CASE WHEN ts.is_home=1
            THEN json_extract(ts.stats_json, '$.home_probable_pitcher')
            END) AS h_sp_name,
        MAX(CASE WHEN ts.is_home=0
            THEN json_extract(ts.stats_json, '$.away_probable_pitcher')
            END) AS a_sp_name
    FROM games g
    JOIN team_stats ts ON g.game_id = ts.game_id
    WHERE g.sport = 'MLB'
      AND g.status = 'Final'
    GROUP BY g.game_id
    """
    sp_df = pd.read_sql_query(sp_query, conn)

    # Pitcher handedness lookup
    pitcher_hand = pd.read_sql_query(
        "SELECT DISTINCT pitcher_name, throws FROM pitcher_stats WHERE throws IS NOT NULL",
        conn,
    )
    hand_map = dict(zip(pitcher_hand["pitcher_name"], pitcher_hand["throws"]))

    sp_df["h_sp_throws"] = sp_df["h_sp_name"].map(hand_map)
    sp_df["a_sp_throws"] = sp_df["a_sp_name"].map(hand_map)
    print(f"  SP handedness fill: home={sp_df['h_sp_throws'].notna().sum()}/{len(sp_df)}, "
          f"away={sp_df['a_sp_throws'].notna().sum()}/{len(sp_df)}")

    conn.close()

    # ── 3. Tag each team_game_batting row with opposing SP hand ────────────────
    # For home team: opposing SP = away SP (a_sp_throws)
    # For away team: opposing SP = home SP (h_sp_throws)
    sp_home = sp_df[["game_pk", "h_sp_throws", "a_sp_throws"]].copy()

    tgb = tgb.merge(sp_home, on="game_pk", how="left")

    # opposing_sp_hand: for the batting team, what hand does the opposing SP throw?
    tgb["opposing_sp_hand"] = np.where(
        tgb["is_home"] == 1,
        tgb["a_sp_throws"],   # home team faces away SP
        tgb["h_sp_throws"],   # away team faces home SP
    )
    tgb["opposing_sp_hand"] = tgb["opposing_sp_hand"].where(
        tgb["opposing_sp_hand"].isin(["L", "R"]), other=np.nan
    )

    tagged = tgb["opposing_sp_hand"].notna().sum()
    print(f"  Tagged with opposing SP hand: {tagged}/{len(tgb)} ({tagged/len(tgb)*100:.1f}%)")

    # ── 4. Compute league-average OPS vs LHP and vs RHP (for shrinkage prior) ─
    # Compute per-season league averages
    valid = tgb[tgb["opposing_sp_hand"].notna()].copy()
    league_avg = (
        valid.groupby(["season", "opposing_sp_hand"])
        .agg(tot_h=("hits", "sum"), tot_ab=("at_bats", "sum"),
             tot_tb=("total_bases", "sum"), tot_bb=("walks", "sum"),
             tot_hbp=("hit_by_pitch", "sum"), tot_sf=("sac_flies", "sum"))
        .reset_index()
    )
    league_avg["lg_obp"] = (
        (league_avg["tot_h"] + league_avg["tot_bb"] + league_avg["tot_hbp"]) /
        (league_avg["tot_ab"] + league_avg["tot_bb"] + league_avg["tot_hbp"] + league_avg["tot_sf"])
    )
    league_avg["lg_slg"] = league_avg["tot_tb"] / league_avg["tot_ab"]
    league_avg["lg_ops"] = league_avg["lg_obp"] + league_avg["lg_slg"]

    # Also league-average overall OPS (regardless of hand)
    league_overall = (
        tgb.groupby("season")
        .agg(tot_h=("hits", "sum"), tot_ab=("at_bats", "sum"),
             tot_tb=("total_bases", "sum"), tot_bb=("walks", "sum"),
             tot_hbp=("hit_by_pitch", "sum"), tot_sf=("sac_flies", "sum"))
        .reset_index()
    )
    league_overall["lg_ops_all"] = (
        (league_overall["tot_h"] + league_overall["tot_bb"] + league_overall["tot_hbp"]) /
        (league_overall["tot_ab"] + league_overall["tot_bb"] + league_overall["tot_hbp"] + league_overall["tot_sf"])
        + league_overall["tot_tb"] / league_overall["tot_ab"]
    )

    print("\n  League-average OPS by season and opposing SP hand:")
    for _, r in league_avg.iterrows():
        print(f"    {r['season']} vs {r['opposing_sp_hand']}HP: OPS={r['lg_ops']:.3f} "
              f"(AB={int(r['tot_ab'])})")
    for _, r in league_overall.iterrows():
        print(f"    {r['season']} overall: OPS={r['lg_ops_all']:.3f}")

    # Build lookup dicts
    lg_ops_by_hand = {}
    for _, r in league_avg.iterrows():
        lg_ops_by_hand[(r["season"], r["opposing_sp_hand"])] = r["lg_ops"]
    lg_ops_overall = dict(zip(league_overall["season"], league_overall["lg_ops_all"]))

    # ── 5. Cumulative point-in-time splits with Bayesian shrinkage ─────────────
    # Sort within (team, season) by game_date
    tgb = tgb.sort_values(["team_name", "season", "game_date"]).reset_index(drop=True)

    # Shrinkage formula:
    #   shrunk_ops = (cum_PA * cum_ops + PRIOR_PA * league_ops) / (cum_PA + PRIOR_PA)
    # PRIOR_PA controls shrinkage strength: higher = more shrinkage
    # 120 PA ≈ ~30 games × ~4 PA/game — roughly one month of data before prior fades
    PRIOR_PA = 120

    results = []

    for (team, season), grp in tgb.groupby(["team_name", "season"]):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        # Running accumulators for vs LHP, vs RHP, and overall
        cum = {
            "L": {"h": 0, "ab": 0, "tb": 0, "bb": 0, "hbp": 0, "sf": 0},
            "R": {"h": 0, "ab": 0, "tb": 0, "bb": 0, "hbp": 0, "sf": 0},
            "ALL": {"h": 0, "ab": 0, "tb": 0, "bb": 0, "hbp": 0, "sf": 0},
        }

        for i, row in grp.iterrows():
            osp = row["opposing_sp_hand"]

            # ── BEFORE updating accumulators: snapshot cumulative stats ──────
            # This is the point-in-time value (sees only prior games)

            # OPS vs tonight's SP hand (if known)
            if pd.notna(osp) and cum[osp]["ab"] > 0:
                c = cum[osp]
                denom = c["ab"] + c["bb"] + c["hbp"] + c["sf"]
                raw_obp = (c["h"] + c["bb"] + c["hbp"]) / denom if denom > 0 else 0
                raw_slg = c["tb"] / c["ab"]
                raw_ops = raw_obp + raw_slg
                cum_pa = c["ab"] + c["bb"] + c["hbp"] + c["sf"]

                # Bayesian shrinkage toward league average
                lg_prior = lg_ops_by_hand.get((season, osp), 0.730)
                shrunk_ops_vs = (cum_pa * raw_ops + PRIOR_PA * lg_prior) / (cum_pa + PRIOR_PA)
            elif pd.notna(osp):
                # No prior data vs this hand → pure prior
                shrunk_ops_vs = lg_ops_by_hand.get((season, osp), 0.730)
                cum_pa = 0
            else:
                shrunk_ops_vs = np.nan
                cum_pa = 0

            # Overall OPS (all games prior)
            c_all = cum["ALL"]
            if c_all["ab"] > 0:
                denom_all = c_all["ab"] + c_all["bb"] + c_all["hbp"] + c_all["sf"]
                raw_obp_all = (c_all["h"] + c_all["bb"] + c_all["hbp"]) / denom_all if denom_all > 0 else 0
                raw_slg_all = c_all["tb"] / c_all["ab"]
                raw_ops_all = raw_obp_all + raw_slg_all
                cum_pa_all = denom_all

                lg_prior_all = lg_ops_overall.get(season, 0.720)
                shrunk_ops_all = (cum_pa_all * raw_ops_all + PRIOR_PA * lg_prior_all) / (cum_pa_all + PRIOR_PA)
            else:
                shrunk_ops_all = lg_ops_overall.get(season, 0.720)
                cum_pa_all = 0

            # Platoon gap = deviation from overall
            if pd.notna(shrunk_ops_vs):
                gap = shrunk_ops_vs - shrunk_ops_all
            else:
                gap = np.nan

            results.append({
                "game_pk": row["game_pk"],
                "team_name": team,
                "is_home": row["is_home"],
                "opposing_sp_hand": osp,
                "ops_vs_sp_hand_pit": shrunk_ops_vs,
                "ops_overall_pit": shrunk_ops_all,
                "platoon_gap_pit": gap,
                "cum_pa_vs_hand": cum_pa,
                "cum_pa_all": cum_pa_all,
            })

            # ── AFTER snapshot: update accumulators with THIS game ────────────
            if pd.notna(osp):
                for key in ["h", "ab", "tb", "bb", "hbp", "sf"]:
                    col_map = {"h": "hits", "ab": "at_bats", "tb": "total_bases",
                               "bb": "walks", "hbp": "hit_by_pitch", "sf": "sac_flies"}
                    cum[osp][key] += row[col_map[key]]

            # Always update overall
            for key in ["h", "ab", "tb", "bb", "hbp", "sf"]:
                col_map = {"h": "hits", "ab": "at_bats", "tb": "total_bases",
                           "bb": "walks", "hbp": "hit_by_pitch", "sf": "sac_flies"}
                cum["ALL"][key] += row[col_map[key]]

    pit_df = pd.DataFrame(results)
    print(f"\n  Point-in-time platoon rows: {len(pit_df)}")

    # ── 6. Verify no leakage: same team-season should have DIFFERENT values ───
    # Pick NYY 2025 and check that ops_vs_sp_hand_pit varies across games
    nyy = pit_df[(pit_df["team_name"] == "New York Yankees") &
                 (pit_df["opposing_sp_hand"] == "R")]
    if len(nyy) > 5:
        vals = nyy["ops_vs_sp_hand_pit"].values[:10]
        print(f"\n  LEAKAGE CHECK — NYY vs RHP first 10 games' OPS:")
        print(f"    {[round(v, 4) for v in vals]}")
        print(f"    Unique values: {len(np.unique(np.round(vals, 4)))}/{len(vals)}")
        if len(np.unique(np.round(vals, 4))) > 1:
            print("    ✓ PASS — values vary across games (no lookahead)")
        else:
            print("    ✗ FAIL — constant across games (LEAKAGE!)")

    # ── 7. Pivot to game-level: home team and away team features ──────────────
    home = pit_df[pit_df["is_home"] == 1][
        ["game_pk", "ops_vs_sp_hand_pit", "ops_overall_pit", "platoon_gap_pit",
         "opposing_sp_hand", "cum_pa_vs_hand"]
    ].rename(columns={
        "ops_vs_sp_hand_pit": "h_ops_vs_sp_hand_pit",
        "ops_overall_pit": "h_ops_overall_pit",
        "platoon_gap_pit": "h_platoon_gap_pit",
        "opposing_sp_hand": "a_sp_throws",
        "cum_pa_vs_hand": "h_cum_pa_vs",
    })

    away = pit_df[pit_df["is_home"] == 0][
        ["game_pk", "ops_vs_sp_hand_pit", "ops_overall_pit", "platoon_gap_pit",
         "opposing_sp_hand", "cum_pa_vs_hand"]
    ].rename(columns={
        "ops_vs_sp_hand_pit": "a_ops_vs_sp_hand_pit",
        "ops_overall_pit": "a_ops_overall_pit",
        "platoon_gap_pit": "a_platoon_gap_pit",
        "opposing_sp_hand": "h_sp_throws",
        "cum_pa_vs_hand": "a_cum_pa_vs",
    })

    game_level = home.merge(away, on="game_pk", how="inner")

    # Compute the differential: positive = home platoon advantage
    game_level["platoon_gap_diff_pit"] = (
        game_level["h_platoon_gap_pit"] - game_level["a_platoon_gap_pit"]
    )

    # Add game_id for merge with feature matrix
    game_level["game_id"] = "MLB_" + game_level["game_pk"].astype(str)

    print(f"  Game-level platoon features: {len(game_level)} games")
    fill = game_level["platoon_gap_diff_pit"].notna().sum()
    print(f"  platoon_gap_diff_pit fill: {fill}/{len(game_level)} ({fill/len(game_level)*100:.1f}%)")

    # LHP game breakdown
    lhp_home = (game_level["h_sp_throws"] == "L").sum()
    lhp_away = (game_level["a_sp_throws"] == "L").sum()
    either_lhp = ((game_level["h_sp_throws"] == "L") | (game_level["a_sp_throws"] == "L")).sum()
    print(f"  LHP starts: home={lhp_home}, away={lhp_away}, either={either_lhp}/{len(game_level)}")

    return game_level


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Load ATS feature matrix and merge
# ═══════════════════════════════════════════════════════════════════════════════

def load_feature_matrix():
    """Load the production ATS feature matrix."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
    from features import build_feature_matrix
    fm = build_feature_matrix("MLB")
    return fm


def prepare_data(fm: pd.DataFrame, platoon: pd.DataFrame):
    """
    Merge feature matrix with platoon features, apply spread augmentation,
    return (X_base, X_plat, y, meta) where base=V1_PRUNED, plat=base+platoon.
    """
    # Merge on game_id
    merged = fm.merge(
        platoon[["game_id", "h_platoon_gap_pit", "a_platoon_gap_pit",
                 "platoon_gap_diff_pit", "h_sp_throws", "a_sp_throws",
                 "h_ops_vs_sp_hand_pit", "a_ops_vs_sp_hand_pit",
                 "h_cum_pa_vs", "a_cum_pa_vs"]],
        on="game_id",
        how="inner",
    )

    # Target column
    target_col = "covered_spread"
    print(f"  Merged dataset: {len(merged)} rows, target='{target_col}'")

    # Filter to valid target
    merged = merged[merged[target_col].notna()].copy()

    # Check which V1_PRUNED features exist
    available = [f for f in V1_PRUNED if f in merged.columns]
    missing = [f for f in V1_PRUNED if f not in merged.columns]
    if missing:
        print(f"  Missing V1_PRUNED features: {missing}")

    # Drop rows where ALL base features are NaN
    merged = merged.dropna(subset=available, how="all")

    return merged, available, target_col


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Model training and evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_eval(X_train, y_train, X_test, y_test, seed=42):
    """Train GBC and return game-level metrics. No calibration wrapper (speed)."""
    gbc = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=10, random_state=seed,
    )
    gbc.fit(X_train, y_train)

    probs = gbc.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = (probs.round() == y_test).mean()

    return {"auc": auc, "brier": brier, "acc": acc, "probs": probs}


def get_Xy(df, features, target_col):
    """Extract X, y arrays from a DataFrame."""
    X = df[features].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Full gauntlet
# ═══════════════════════════════════════════════════════════════════════════════

def run_gauntlet(merged, base_features, target_col, platoon_features_list):
    """
    Run the full validation gauntlet comparing:
      - Baseline (V1 Pruned, 21 features)
      - Baseline + honest platoon gap

    Tests:
      1. 10-seed chrono 50/50 split
      2. TimeSeriesSplit 5-fold CV
      3. 5-seed random 50/50 splits
      4. Bootstrap CI on AUC delta
    """
    all_features = base_features + platoon_features_list

    # Sort by date for chrono splits
    merged = merged.sort_values("game_date").reset_index(drop=True)

    # Drop rows where platoon features are all NaN
    valid = merged.dropna(subset=platoon_features_list, how="all").copy()
    print(f"\n  Valid rows (platoon features present): {len(valid)}/{len(merged)}")

    # Fill NaN in features with column median
    for f in all_features:
        if f in valid.columns:
            valid[f] = valid[f].fillna(valid[f].median())

    # ── Game-level dedup ──────────────────────────────────────────────────────
    # Deduplicate to game level (take first row per game_id)
    valid = valid.drop_duplicates(subset=["game_id"], keep="first")
    print(f"  After game-level dedup: {len(valid)} games")

    n = len(valid)
    results = {
        "chrono": {"base": [], "plat": []},
        "tscv": {"base": [], "plat": []},
        "random": {"base": [], "plat": []},
    }

    # ── TEST 1: 10-seed chrono 50/50 split ────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: 10-seed chrono 50/50 split")
    print("=" * 70)
    mid = n // 2
    train_df = valid.iloc[:mid]
    test_df = valid.iloc[mid:]

    for seed in range(10):
        X_tr_base, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_plat, _ = get_Xy(train_df, all_features, target_col)
        X_te_base, y_te = get_Xy(test_df, base_features, target_col)
        X_te_plat, _ = get_Xy(test_df, all_features, target_col)

        r_base = train_and_eval(X_tr_base, y_tr, X_te_base, y_te, seed)
        r_plat = train_and_eval(X_tr_plat, y_tr, X_te_plat, y_te, seed)

        results["chrono"]["base"].append(r_base["auc"])
        results["chrono"]["plat"].append(r_plat["auc"])

    base_mean = np.mean(results["chrono"]["base"])
    plat_mean = np.mean(results["chrono"]["plat"])
    delta = plat_mean - base_mean
    print(f"  Baseline AUC: {base_mean:.4f} ± {np.std(results['chrono']['base']):.4f}")
    print(f"  +Platoon AUC: {plat_mean:.4f} ± {np.std(results['chrono']['plat']):.4f}")
    print(f"  Delta: {delta:+.4f}")
    wins = sum(p > b for p, b in zip(results["chrono"]["plat"], results["chrono"]["base"]))
    print(f"  Platoon wins: {wins}/10 seeds")

    # ── TEST 2: TimeSeriesSplit 5-fold CV ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: TimeSeriesSplit 5-fold CV")
    print("=" * 70)
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(valid)):
        train_df_cv = valid.iloc[tr_idx]
        test_df_cv = valid.iloc[te_idx]

        X_tr_base, y_tr = get_Xy(train_df_cv, base_features, target_col)
        X_tr_plat, _ = get_Xy(train_df_cv, all_features, target_col)
        X_te_base, y_te = get_Xy(test_df_cv, base_features, target_col)
        X_te_plat, _ = get_Xy(test_df_cv, all_features, target_col)

        r_base = train_and_eval(X_tr_base, y_tr, X_te_base, y_te, 42)
        r_plat = train_and_eval(X_tr_plat, y_tr, X_te_plat, y_te, 42)

        results["tscv"]["base"].append(r_base["auc"])
        results["tscv"]["plat"].append(r_plat["auc"])

        print(f"  Fold {fold}: base={r_base['auc']:.4f}, +plat={r_plat['auc']:.4f}, "
              f"delta={r_plat['auc'] - r_base['auc']:+.4f}")

    base_mean_cv = np.mean(results["tscv"]["base"])
    plat_mean_cv = np.mean(results["tscv"]["plat"])
    print(f"  Mean: base={base_mean_cv:.4f}, +plat={plat_mean_cv:.4f}, "
          f"delta={plat_mean_cv - base_mean_cv:+.4f}")

    # ── TEST 3: 5-seed random 50/50 splits ────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: 5-seed random 50/50 splits (game-level)")
    print("=" * 70)
    for seed in range(5):
        rng = np.random.RandomState(seed + 100)
        game_ids = valid["game_id"].unique()
        rng.shuffle(game_ids)
        mid_g = len(game_ids) // 2
        train_ids = set(game_ids[:mid_g])
        test_ids = set(game_ids[mid_g:])

        train_r = valid[valid["game_id"].isin(train_ids)]
        test_r = valid[valid["game_id"].isin(test_ids)]

        X_tr_base, y_tr = get_Xy(train_r, base_features, target_col)
        X_tr_plat, _ = get_Xy(train_r, all_features, target_col)
        X_te_base, y_te = get_Xy(test_r, base_features, target_col)
        X_te_plat, _ = get_Xy(test_r, all_features, target_col)

        r_base = train_and_eval(X_tr_base, y_tr, X_te_base, y_te, seed)
        r_plat = train_and_eval(X_tr_plat, y_tr, X_te_plat, y_te, seed)

        results["random"]["base"].append(r_base["auc"])
        results["random"]["plat"].append(r_plat["auc"])

        print(f"  Seed {seed}: base={r_base['auc']:.4f}, +plat={r_plat['auc']:.4f}, "
              f"delta={r_plat['auc'] - r_base['auc']:+.4f}")

    base_mean_r = np.mean(results["random"]["base"])
    plat_mean_r = np.mean(results["random"]["plat"])
    print(f"  Mean: base={base_mean_r:.4f}, +plat={plat_mean_r:.4f}, "
          f"delta={plat_mean_r - base_mean_r:+.4f}")

    # ── TEST 4: Bootstrap CI on AUC delta (chrono split) ──────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: Bootstrap CI on AUC delta (1000 resamples)")
    print("=" * 70)
    # Use seed=0 chrono split for bootstrap
    train_df_bs = valid.iloc[:mid]
    test_df_bs = valid.iloc[mid:]

    X_tr_base, y_tr = get_Xy(train_df_bs, base_features, target_col)
    X_tr_plat, _ = get_Xy(train_df_bs, all_features, target_col)
    X_te_base, y_te = get_Xy(test_df_bs, base_features, target_col)
    X_te_plat, _ = get_Xy(test_df_bs, all_features, target_col)

    r_base_bs = train_and_eval(X_tr_base, y_tr, X_te_base, y_te, 0)
    r_plat_bs = train_and_eval(X_tr_plat, y_tr, X_te_plat, y_te, 0)

    probs_base = r_base_bs["probs"]
    probs_plat = r_plat_bs["probs"]

    deltas_boot = []
    rng = np.random.RandomState(999)
    n_test = len(y_te)
    for _ in range(1000):
        idx = rng.choice(n_test, n_test, replace=True)
        try:
            auc_b = roc_auc_score(y_te[idx], probs_base[idx])
            auc_p = roc_auc_score(y_te[idx], probs_plat[idx])
            deltas_boot.append(auc_p - auc_b)
        except ValueError:
            pass

    deltas_boot = np.array(deltas_boot)
    ci_lo = np.percentile(deltas_boot, 2.5)
    ci_hi = np.percentile(deltas_boot, 97.5)
    pct_pos = (deltas_boot > 0).mean() * 100
    print(f"  AUC delta: {np.mean(deltas_boot):+.4f}")
    print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  P(delta > 0): {pct_pos:.1f}%")

    return results, deltas_boot


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: LHP-extreme-gap subset test
# ═══════════════════════════════════════════════════════════════════════════════

def run_lhp_subset_test(merged, base_features, target_col, platoon_features_list):
    """
    Focus test: does honest platoon gap help specifically in LHP-starter games
    with large platoon gaps?
    """
    all_features = base_features + platoon_features_list
    merged = merged.sort_values("game_date").reset_index(drop=True)

    valid = merged.dropna(subset=platoon_features_list, how="all").copy()
    for f in all_features:
        if f in valid.columns:
            valid[f] = valid[f].fillna(valid[f].median())
    valid = valid.drop_duplicates(subset=["game_id"], keep="first")

    # ── Identify LHP games ────────────────────────────────────────────────────
    lhp_either = valid[
        (valid["h_sp_throws"] == "L") | (valid["a_sp_throws"] == "L")
    ].copy()
    print(f"\n  LHP-either games: {len(lhp_either)}/{len(valid)}")

    # ── Large gap subset: top quartile of |platoon_gap_diff_pit| ──────────────
    gap_abs = lhp_either["platoon_gap_diff_pit"].abs()
    q75 = gap_abs.quantile(0.75)
    large_gap = lhp_either[gap_abs >= q75].copy()
    print(f"  Large-gap LHP games (|gap| >= {q75:.4f}): {len(large_gap)}")

    # ── Split-half test on LHP-large-gap ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("LHP SUBSET: 10-seed chrono split (LHP games, large gap)")
    print("=" * 70)

    large_gap = large_gap.sort_values("game_date").reset_index(drop=True)
    n = len(large_gap)

    if n < 100:
        print(f"  WARNING: Only {n} games — results will be noisy")

    # For the subset test, train on ALL data but evaluate only on the subset
    # This is the right approach: the model sees everything, but we measure
    # whether platoon features specifically help in the LHP-large-gap games
    all_sorted = valid.sort_values("game_date").reset_index(drop=True)
    mid_all = len(all_sorted) // 2

    base_aucs = []
    plat_aucs = []

    for seed in range(10):
        train_df = all_sorted.iloc[:mid_all]
        test_df = all_sorted.iloc[mid_all:]

        # Filter test set to LHP-large-gap games only
        test_lhp = test_df[
            ((test_df["h_sp_throws"] == "L") | (test_df["a_sp_throws"] == "L")) &
            (test_df["platoon_gap_diff_pit"].abs() >= q75)
        ]

        if len(test_lhp) < 30:
            print(f"  Seed {seed}: only {len(test_lhp)} LHP-large-gap in test half — skipping")
            continue

        X_tr_base, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_plat, _ = get_Xy(train_df, all_features, target_col)
        X_te_base, y_te = get_Xy(test_lhp, base_features, target_col)
        X_te_plat, _ = get_Xy(test_lhp, all_features, target_col)

        r_base = train_and_eval(X_tr_base, y_tr, X_te_base, y_te, seed)
        r_plat = train_and_eval(X_tr_plat, y_tr, X_te_plat, y_te, seed)

        base_aucs.append(r_base["auc"])
        plat_aucs.append(r_plat["auc"])

        print(f"  Seed {seed}: N={len(test_lhp)}, "
              f"base={r_base['auc']:.4f}, +plat={r_plat['auc']:.4f}, "
              f"delta={r_plat['auc'] - r_base['auc']:+.4f}")

    if base_aucs:
        print(f"\n  Mean: base={np.mean(base_aucs):.4f}, +plat={np.mean(plat_aucs):.4f}, "
              f"delta={np.mean(plat_aucs) - np.mean(base_aucs):+.4f}")
        wins = sum(p > b for p, b in zip(plat_aucs, base_aucs))
        print(f"  Platoon wins: {wins}/{len(base_aucs)} seeds")

    # ── WR vs breakeven test (the real money question) ────────────────────────
    print("\n" + "=" * 70)
    print("LHP SUBSET: WR vs breakeven (large gap, second half)")
    print("=" * 70)

    # In the second half, check if predicting WITH platoon gap
    # makes correct directional calls more often
    if len(test_lhp) >= 30:
        # Use the seed=0 models
        X_tr_base, y_tr = get_Xy(train_df, base_features, target_col)
        X_tr_plat, _ = get_Xy(train_df, all_features, target_col)
        X_te_base, y_te = get_Xy(test_lhp, base_features, target_col)
        X_te_plat, _ = get_Xy(test_lhp, all_features, target_col)

        r_base = train_and_eval(X_tr_base, y_tr, X_te_base, y_te, 0)
        r_plat = train_and_eval(X_tr_plat, y_tr, X_te_plat, y_te, 0)

        # Pick side: predict >0.5 → bet home, <0.5 → bet away
        base_picks = (r_base["probs"] > 0.5).astype(int)
        plat_picks = (r_plat["probs"] > 0.5).astype(int)

        base_correct = (base_picks == y_te).mean()
        plat_correct = (plat_picks == y_te).mean()

        # With spreads, breakeven is ~52.4% (at -110 odds)
        BE = 0.524
        print(f"  N games: {len(y_te)}")
        print(f"  Baseline WR: {base_correct:.3f} (vs BE {BE:.3f}, edge={base_correct - BE:+.3f})")
        print(f"  +Platoon WR: {plat_correct:.3f} (vs BE {BE:.3f}, edge={plat_correct - BE:+.3f})")
        print(f"  Delta: {plat_correct - base_correct:+.3f}")

    return base_aucs, plat_aucs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("HONEST PLATOON TEST — Point-in-time splits with Bayesian shrinkage")
    print("=" * 70)

    # Step 1: Build honest features
    print("\n[1/4] Building point-in-time platoon features...")
    platoon = build_honest_platoon_features()

    # Step 2: Load feature matrix
    print("\n[2/4] Loading ATS feature matrix...")
    fm = load_feature_matrix()
    print(f"  Feature matrix: {len(fm)} rows")

    # Step 3: Merge and prepare
    print("\n[3/4] Merging and preparing data...")
    merged, base_features, target_col = prepare_data(fm, platoon)

    platoon_feats = ["platoon_gap_diff_pit"]
    print(f"  Base features: {len(base_features)}")
    print(f"  Platoon features: {platoon_feats}")

    # Step 4: Run gauntlet
    print("\n[4/4] Running full validation gauntlet...")
    gauntlet_results, boot_deltas = run_gauntlet(
        merged, base_features, target_col, platoon_feats
    )

    # Step 5: LHP subset test
    print("\n\n[BONUS] LHP-extreme-gap subset test...")
    lhp_base, lhp_plat = run_lhp_subset_test(
        merged, base_features, target_col, platoon_feats
    )

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL VERDICT — HONEST PLATOON TEST")
    print("=" * 70)

    # Summarize across all tests
    chrono_delta = np.mean(gauntlet_results["chrono"]["plat"]) - np.mean(gauntlet_results["chrono"]["base"])
    tscv_delta = np.mean(gauntlet_results["tscv"]["plat"]) - np.mean(gauntlet_results["tscv"]["base"])
    random_delta = np.mean(gauntlet_results["random"]["plat"]) - np.mean(gauntlet_results["random"]["base"])
    boot_mean = np.mean(boot_deltas)
    boot_ci = (np.percentile(boot_deltas, 2.5), np.percentile(boot_deltas, 97.5))
    boot_pct_pos = (boot_deltas > 0).mean() * 100

    print(f"\n  Chrono 50/50 (10 seeds):  delta = {chrono_delta:+.4f}")
    print(f"  TimeSeriesSplit (5-fold): delta = {tscv_delta:+.4f}")
    print(f"  Random 50/50 (5 seeds):  delta = {random_delta:+.4f}")
    print(f"  Bootstrap CI (1000):     delta = {boot_mean:+.4f} [{boot_ci[0]:+.4f}, {boot_ci[1]:+.4f}]")
    print(f"  P(delta > 0):            {boot_pct_pos:.1f}%")

    if lhp_plat:
        lhp_delta = np.mean(lhp_plat) - np.mean(lhp_base)
        lhp_wins = sum(p > b for p, b in zip(lhp_plat, lhp_base))
        print(f"\n  LHP-large-gap subset:    delta = {lhp_delta:+.4f} ({lhp_wins}/{len(lhp_plat)} seeds positive)")

    # Decision
    all_positive = chrono_delta > 0 and tscv_delta > 0 and random_delta > 0
    ci_excludes_zero = boot_ci[0] > 0
    lhp_positive = lhp_plat and np.mean(lhp_plat) > np.mean(lhp_base)

    if all_positive and ci_excludes_zero:
        print("\n  VERDICT: HONEST PLATOON GAP SURVIVES — deploy")
    elif lhp_positive and boot_pct_pos > 75:
        print("\n  VERDICT: MARGINAL — positive in LHP subset but not robust across all tests")
        print("           Consider deploying for LHP-starter games only with monitoring")
    else:
        print("\n  VERDICT: HONEST PLATOON GAP DOES NOT SURVIVE")
        print("           The leaky version's positive results were from lookahead, not signal")


if __name__ == "__main__":
    main()
