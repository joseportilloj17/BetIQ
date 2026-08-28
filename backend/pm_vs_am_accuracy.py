#!/usr/bin/env python3
"""
PM-vs-AM accuracy diagnostic — does the model predict better with confirmed
(PM) lineups vs projected (AM) lineups?

Historical proxy approach:
  - PM (confirmed): actual boxscore starters from batter_game_stats
  - AM (projected): simulated by randomly swapping ~14.4% of batter slots
    with same-team/same-season alternates (mimics observed AM→PM mismatch rate)

The lineup-augmented ATS model (25 features: 21 V1-pruned + lineup_gap_resid +
3 spread augmentation) is scored with both PM and AM features.

Metrics: AUC, Brier, WR, paired accuracy on changed-lineup games.
"""
from __future__ import annotations

import json
import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")
_SUBMODEL_DIR = os.path.join(_DATA_DIR, "submodels")

PRIOR_PA = 150
SWAP_RATE = 0.144  # observed AM→PM mismatch rate
N_SEEDS = 10


def _league_ops_by_season(conn: sqlite3.Connection) -> dict:
    """Compute league-average OPS per season from batter_game_stats."""
    bgs = pd.read_sql_query(
        "SELECT tgb.season, SUM(bgs.hits) AS h, SUM(bgs.ab) AS ab, SUM(bgs.tb) AS tb, "
        "SUM(bgs.bb) AS bb, SUM(bgs.hbp) AS hbp, SUM(bgs.sf) AS sf "
        "FROM batter_game_stats bgs "
        "JOIN team_game_batting tgb ON bgs.game_pk = tgb.game_pk AND bgs.team_name = tgb.team_name "
        "GROUP BY tgb.season",
        conn,
    )
    result = {}
    for _, r in bgs.iterrows():
        denom = r["ab"] + r["bb"] + r["hbp"] + r["sf"]
        if denom > 0 and r["ab"] > 0:
            obp = (r["h"] + r["bb"] + r["hbp"]) / denom
            slg = r["tb"] / r["ab"]
            result[r["season"]] = round(obp + slg, 4)
        else:
            result[r["season"]] = 0.720
    return result


def build_lineup_features_pm_am():
    """
    Build game-level lineup_gap_resid for PM (actual) and AM (simulated) lineups.

    Returns DataFrame with game_id, lineup_gap_resid_pm, lineup_gap_resid_am_{seed},
    plus merge keys for the feature matrix.
    """
    conn = sqlite3.connect(_HIST_DB)
    lg_ops = _league_ops_by_season(conn)

    print("  Loading batter game stats...")
    bgs = pd.read_sql_query("""
        SELECT bgs.game_pk, bgs.player_id, bgs.team_name,
               bgs.is_home, bgs.ab, bgs.hits AS hits, bgs.tb,
               bgs.bb, bgs.hbp, bgs.sf,
               tgb.game_date, tgb.season
        FROM batter_game_stats bgs
        JOIN team_game_batting tgb ON bgs.game_pk = tgb.game_pk
                                  AND bgs.team_name = tgb.team_name
        ORDER BY bgs.player_id, tgb.season, tgb.game_date
    """, conn)

    # Compute per-batter cumulative OPS (point-in-time, shifted)
    bgs = bgs.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    cum_cols = ["ab", "hits", "tb", "bb", "hbp", "sf"]
    for col in cum_cols:
        bgs[f"cum_{col}"] = bgs.groupby(["player_id", "season"])[col].transform(
            lambda x: x.cumsum().shift(1)
        )

    cum_pa = (bgs["cum_ab"].fillna(0) + bgs["cum_bb"].fillna(0) +
              bgs["cum_hbp"].fillna(0) + bgs["cum_sf"].fillna(0))

    raw_obp = np.where(cum_pa > 0,
        (bgs["cum_hits"].fillna(0) + bgs["cum_bb"].fillna(0) + bgs["cum_hbp"].fillna(0)) / cum_pa, 0)
    raw_slg = np.where(bgs["cum_ab"].fillna(0) > 0,
        bgs["cum_tb"].fillna(0) / bgs["cum_ab"].fillna(1), 0)
    raw_ops = raw_obp + raw_slg

    lg_prior = bgs["season"].map(lg_ops).fillna(0.720)
    bgs["batter_ops_pit"] = np.where(
        cum_pa > 0,
        (cum_pa * raw_ops + PRIOR_PA * lg_prior) / (cum_pa + PRIOR_PA),
        lg_prior
    )

    # Build team rosters: {(team, season) → [player_ids with cumulative OPS]}
    # For AM simulation, we swap actual starters with roster alternates
    print("  Building team rosters for AM simulation...")
    roster_ops = bgs.groupby(["team_name", "season", "player_id"])["batter_ops_pit"].last()
    team_rosters: dict[tuple, list[int]] = {}
    for (team, season, pid), _ in roster_ops.items():
        key = (team, season)
        if key not in team_rosters:
            team_rosters[key] = []
        team_rosters[key].append(pid)

    # PM lineup quality = mean of 9 actual starters' cumulative OPS
    print("  Computing PM (actual) lineup quality...")
    lineup_q = (
        bgs.groupby(["game_pk", "team_name", "is_home"])
        .agg(
            lineup_ops=("batter_ops_pit", "mean"),
            n_batters=("player_id", "count"),
            player_ids=("player_id", lambda x: list(x)),
            batter_ops_list=("batter_ops_pit", lambda x: list(x)),
        )
        .reset_index()
    )
    # Add season info
    game_seasons = bgs.groupby("game_pk")["season"].first()
    lineup_q["season"] = lineup_q["game_pk"].map(game_seasons)

    # Team cumulative OPS (point-in-time)
    tgb = pd.read_sql_query("""
        SELECT team_name, season, game_date, game_pk, is_home,
               at_bats, hits, total_bases, walks, hit_by_pitch, sac_flies
        FROM team_game_batting
        ORDER BY team_name, season, game_date
    """, conn)
    conn.close()

    for src, dst in [("at_bats", "t_ab"), ("hits", "t_h"), ("total_bases", "t_tb"),
                     ("walks", "t_bb"), ("hit_by_pitch", "t_hbp"), ("sac_flies", "t_sf")]:
        tgb[dst] = tgb.groupby(["team_name", "season"])[src].transform(
            lambda x: x.cumsum().shift(1)
        )

    t_denom = tgb["t_ab"].fillna(0) + tgb["t_bb"].fillna(0) + \
              tgb["t_hbp"].fillna(0) + tgb["t_sf"].fillna(0)
    t_obp = np.where(t_denom > 0,
        (tgb["t_h"].fillna(0) + tgb["t_bb"].fillna(0) + tgb["t_hbp"].fillna(0)) / t_denom, 0)
    t_slg = np.where(tgb["t_ab"].fillna(0) > 0,
        tgb["t_tb"].fillna(0) / tgb["t_ab"].fillna(1), 0)
    lg_prior_t = tgb["season"].map(lg_ops).fillna(0.720)
    tgb["team_ops_pit"] = np.where(
        t_denom > 0,
        (t_denom * (t_obp + t_slg) + PRIOR_PA * lg_prior_t) / (t_denom + PRIOR_PA),
        lg_prior_t
    )

    # Merge and compute gap
    tgb_slim = tgb[["game_pk", "team_name", "is_home", "team_ops_pit"]].copy()
    merged = lineup_q.merge(tgb_slim, on=["game_pk", "team_name", "is_home"], how="inner")
    merged["lineup_gap_pm"] = merged["lineup_ops"] - merged["team_ops_pit"]

    # ── AM simulation: swap ~14.4% of batters per game per seed ──
    print(f"  Simulating AM lineups ({N_SEEDS} seeds, {SWAP_RATE:.1%} swap rate)...")

    # Build a lookup: (player_id, game_pk) → batter_ops_pit
    batter_lookup = dict(zip(
        zip(bgs["player_id"], bgs["game_pk"]),
        bgs["batter_ops_pit"]
    ))
    # Build roster OPS lookup: player_id → mean batter_ops_pit across career
    player_mean_ops = bgs.groupby("player_id")["batter_ops_pit"].mean().to_dict()

    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        am_ops_list = []

        for _, row in merged.iterrows():
            pids = row["player_ids"]
            ops_vals = row["batter_ops_list"]
            team = row["team_name"]
            season = row["season"]

            roster_key = (team, season)
            roster = team_rosters.get(roster_key, [])
            # Exclude current starters from replacement pool
            bench = [p for p in roster if p not in pids]

            if not bench:
                am_ops_list.append(row["lineup_ops"])
                continue

            # Swap each slot with SWAP_RATE probability
            new_ops = list(ops_vals)
            for i in range(len(new_ops)):
                if rng.random() < SWAP_RATE:
                    replacement = rng.choice(bench)
                    # Use the replacement's mean OPS
                    new_ops[i] = player_mean_ops.get(replacement, lg_ops.get(season, 0.720))

            am_ops_list.append(np.mean(new_ops))

        merged[f"lineup_ops_am_{seed}"] = am_ops_list
        merged[f"lineup_gap_am_{seed}"] = merged[f"lineup_ops_am_{seed}"] - merged["team_ops_pit"]

    # Pivot to game level
    home = merged[merged["is_home"] == 1].copy()
    away = merged[merged["is_home"] == 0].copy()

    game_level = home[["game_pk"]].copy()
    game_level["h_gap_pm"] = home["lineup_gap_pm"].values
    for seed in range(N_SEEDS):
        game_level[f"h_gap_am_{seed}"] = home[f"lineup_gap_am_{seed}"].values

    away_data = away[["game_pk"]].copy()
    away_data["a_gap_pm"] = away["lineup_gap_pm"].values
    for seed in range(N_SEEDS):
        away_data[f"a_gap_am_{seed}"] = away[f"lineup_gap_am_{seed}"].values

    game_level = game_level.merge(away_data, on="game_pk", how="inner")

    # Compute lineup_gap_diff (PM and AM per seed)
    game_level["lgd_pm"] = game_level["h_gap_pm"] - game_level["a_gap_pm"]
    for seed in range(N_SEEDS):
        game_level[f"lgd_am_{seed}"] = (
            game_level[f"h_gap_am_{seed}"] - game_level[f"a_gap_am_{seed}"]
        )

    # Residualize using frozen params
    params_path = os.path.join(_SUBMODEL_DIR, "lineup_resid_params.json")
    with open(params_path) as f:
        resid_params = json.load(f)
    coef = resid_params["coef"]
    intercept = resid_params["intercept"]

    game_level["game_id"] = "MLB_" + game_level["game_pk"].astype(str)

    # Get ops_diff from feature matrix
    from features import build_feature_matrix
    fm = build_feature_matrix("MLB")
    fm_slim = fm[["game_id", "ops_diff"]].copy()
    game_level = game_level.merge(fm_slim, on="game_id", how="inner")

    valid = game_level[game_level["ops_diff"].notna()].copy()

    # Residualize PM
    valid["lineup_gap_resid_pm"] = valid["lgd_pm"] - (coef * valid["ops_diff"] + intercept)
    # Residualize AM (per seed)
    for seed in range(N_SEEDS):
        valid[f"lineup_gap_resid_am_{seed}"] = (
            valid[f"lgd_am_{seed}"] - (coef * valid["ops_diff"] + intercept)
        )

    print(f"  {len(valid)} games with PM + AM lineup features")
    return valid, fm


def score_and_compare():
    """Load model, score PM vs AM, compare accuracy."""
    print("=" * 70)
    print("PM-vs-AM ACCURACY: Does confirmed lineup improve predictions?")
    print("=" * 70)

    # Build features
    print("\n  Building features...")
    valid, fm = build_lineup_features_pm_am()

    # Load lineup model
    print("  Loading lineup-augmented ATS model...")
    clf_path = os.path.join(_SUBMODEL_DIR, "mlb_ats_lineup_clf.pkl")
    scaler_path = os.path.join(_SUBMODEL_DIR, "mlb_ats_lineup_scaler.pkl")
    imputer_path = os.path.join(_SUBMODEL_DIR, "mlb_ats_lineup_imputer.pkl")
    feat_path = os.path.join(_SUBMODEL_DIR, "mlb_ats_lineup_feat_cols.pkl")

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    print(f"  Model features: {feat_cols}")

    # Get the feature matrix with all base features
    from features import TARGET_ATS, MODEL_FEATURE_COLS_V1_PRUNED
    base_feats = list(MODEL_FEATURE_COLS_V1_PRUNED)

    # Merge valid with full feature matrix for base features
    # valid already has ops_diff (from residualization step) — drop it before merge
    # to avoid column collision with the feature matrix's ops_diff
    valid_for_merge = valid.drop(columns=["ops_diff"], errors="ignore")
    fm_base = fm[["game_id"] + base_feats + [TARGET_ATS]].copy()
    combined = valid_for_merge.merge(fm_base, on="game_id", how="inner")

    # Filter to games with target
    combined = combined[combined[TARGET_ATS].notna()].copy()
    y = combined[TARGET_ATS].values.astype(int)
    print(f"  {len(combined)} games with features + target")

    if len(combined) < 200:
        print("  INSUFFICIENT DATA")
        return

    # Build feature matrices for PM and AM
    # Features: base_feats + lineup_gap_resid + spread_value + spread_abs + is_fav_spread
    # For this test, use spread_value = -1.5 (most common rung)
    spread_val = -1.5

    def _build_X(lineup_resid_col):
        X = combined[base_feats].copy()
        X["lineup_gap_resid"] = combined[lineup_resid_col].values
        X["_spread_value"] = spread_val
        X["_spread_abs"] = abs(spread_val)
        X["_is_fav_spread"] = 1 if spread_val < 0 else 0
        # Reorder to match model's expected features
        X = X[feat_cols]
        X = X.fillna(X.median())
        X_s = scaler.transform(X)
        X_i = imputer.transform(X_s)
        return clf.predict_proba(X_i)[:, 1]

    # Score PM
    probs_pm = _build_X("lineup_gap_resid_pm")

    # Score AM (per seed) and average
    am_probs_list = []
    for seed in range(N_SEEDS):
        probs_am_seed = _build_X(f"lineup_gap_resid_am_{seed}")
        am_probs_list.append(probs_am_seed)
    probs_am = np.mean(am_probs_list, axis=0)

    # ── Accuracy comparison ──
    print(f"\n{'='*70}")
    print("RESULTS: PM (confirmed lineup) vs AM (projected lineup)")
    print(f"{'='*70}")

    auc_pm = roc_auc_score(y, probs_pm)
    auc_am = roc_auc_score(y, probs_am)
    brier_pm = brier_score_loss(y, probs_pm)
    brier_am = brier_score_loss(y, probs_am)

    # WR at 50% threshold
    wr_pm = np.mean((probs_pm >= 0.5) == y)
    wr_am = np.mean((probs_am >= 0.5) == y)

    print(f"\n  {'Metric':<15} {'PM (confirmed)':<18} {'AM (projected)':<18} {'Δ':<12}")
    print(f"  {'AUC':<15} {auc_pm:.6f}          {auc_am:.6f}          {auc_pm - auc_am:+.6f}")
    print(f"  {'Brier':<15} {brier_pm:.6f}          {brier_am:.6f}          {brier_pm - brier_am:+.6f}")
    print(f"  {'WR':<15} {wr_pm:.4f}            {wr_am:.4f}            {wr_pm - wr_am:+.4f}")

    # ── AM seed variability ──
    print(f"\n  ── AM seed variability ({N_SEEDS} seeds) ──")
    seed_aucs = [roc_auc_score(y, p) for p in am_probs_list]
    seed_briers = [brier_score_loss(y, p) for p in am_probs_list]
    print(f"  AUC range:   {min(seed_aucs):.6f} – {max(seed_aucs):.6f} "
          f"(mean={np.mean(seed_aucs):.6f})")
    print(f"  Brier range: {min(seed_briers):.6f} – {max(seed_briers):.6f} "
          f"(mean={np.mean(seed_briers):.6f})")

    # ── Changed-lineup subsample ──
    # Games where AM simulation actually changed the lineup (any seed)
    pm_resid = combined["lineup_gap_resid_pm"].values
    am_resid_mean = np.mean(
        [combined[f"lineup_gap_resid_am_{s}"].values for s in range(N_SEEDS)], axis=0
    )
    resid_diff = np.abs(pm_resid - am_resid_mean)
    changed_mask = resid_diff > 0.001  # non-trivial change

    n_changed = changed_mask.sum()
    print(f"\n  ── Changed-lineup games (resid diff > 0.001) ──")
    print(f"  N={n_changed} / {len(combined)} ({n_changed/len(combined)*100:.1f}%)")

    if n_changed > 50:
        auc_pm_c = roc_auc_score(y[changed_mask], probs_pm[changed_mask])
        auc_am_c = roc_auc_score(y[changed_mask], probs_am[changed_mask])
        brier_pm_c = brier_score_loss(y[changed_mask], probs_pm[changed_mask])
        brier_am_c = brier_score_loss(y[changed_mask], probs_am[changed_mask])
        print(f"  PM AUC:   {auc_pm_c:.6f}   (vs AM: {auc_am_c:.6f}, Δ={auc_pm_c-auc_am_c:+.6f})")
        print(f"  PM Brier: {brier_pm_c:.6f}   (vs AM: {brier_am_c:.6f}, Δ={brier_pm_c-brier_am_c:+.6f})")

    # ── Disagreement analysis ──
    # Games where PM and AM predict different sides (one > 0.5, other < 0.5)
    pm_pick = probs_pm >= 0.5
    am_pick = probs_am >= 0.5
    disagree_mask = pm_pick != am_pick
    n_disagree = disagree_mask.sum()

    print(f"\n  ── Disagreement analysis ──")
    print(f"  Games where PM and AM pick different sides: {n_disagree} "
          f"({n_disagree/len(combined)*100:.1f}%)")

    if n_disagree > 10:
        pm_correct = np.mean(pm_pick[disagree_mask] == y[disagree_mask])
        am_correct = np.mean(am_pick[disagree_mask] == y[disagree_mask])
        print(f"  PM correct on disagreements: {pm_correct:.3f}")
        print(f"  AM correct on disagreements: {am_correct:.3f}")
        print(f"  PM wins {pm_correct*100:.1f}% of disagreement games "
              f"(N={n_disagree})")

    # ── Feature magnitude comparison ──
    print(f"\n  ── Feature magnitude ──")
    print(f"  PM resid: mean={pm_resid.mean():.6f}, std={pm_resid.std():.6f}")
    print(f"  AM resid: mean={am_resid_mean.mean():.6f}, std={am_resid_mean.std():.6f}")
    print(f"  |PM - AM|: mean={resid_diff.mean():.6f}, max={resid_diff.max():.6f}")

    # ── Verdict ──
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    delta_auc = auc_pm - auc_am
    delta_brier = brier_pm - brier_am

    if abs(delta_auc) < 0.001:
        print("""
  PM and AM produce INDISTINGUISHABLE predictions.
  ΔAUC < 0.001 — lineup confirmation adds NO measurable accuracy.
  The model's signal comes from season-level stats (ERA, OPS, WHIP),
  not from knowing which specific 9 batters are in today's lineup.

  CONCLUSION: A PM re-run cycle is NOT worth the infrastructure cost.
  The lineup feature's marginal contribution is too small for AM→PM
  input freshness to matter.""")
    elif delta_auc > 0.001:
        print(f"""
  PM (confirmed lineup) is MORE accurate than AM (projected).
  ΔAUC = {delta_auc:+.4f}, ΔBrier = {delta_brier:+.6f}
  {f'PM wins {pm_correct*100:.0f}% of disagreement games (N={n_disagree})' if n_disagree > 10 else ''}

  CONCLUSION: A PM re-run cycle WOULD improve accuracy.
  Recommended: add PM scoring after lineups confirmed (~1-2 hrs pre-game).
  Use existing re-run infrastructure (scheduler_run_pm endpoint).""")
    else:
        print(f"""
  AM (projected lineup) is slightly MORE accurate than PM (confirmed).
  ΔAUC = {delta_auc:+.4f} — likely noise from the AM simulation.

  CONCLUSION: No benefit from PM re-run. The lineup feature is too weak
  for input freshness to move the needle.""")


if __name__ == "__main__":
    score_and_compare()
