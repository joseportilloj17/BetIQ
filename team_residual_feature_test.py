#!/usr/bin/env python3
"""
team_residual_feature_test.py — Test whether RESIDUAL-BASED team features
improve MLB ATS prediction where K-means clustering failed.

Rationale
---------
K-means clustering on profile features added +0.000 AUC because it re-encoded
information the model already has.  Team-level residual variance is 2.8-5.3x
random (p<0.001), proving a team-specific signal EXISTS — but it lives in the
RESIDUAL (what's left after profile-matching), not in the profiles themselves.

This script injects each team's historical mean residual (actual cover rate
minus model prediction) as a leak-free expanding-window feature.

Protocol
--------
  - Train on data before 2026-05-26
  - Test on Early (May 26 - Jun 15) and Late (Jul 1 - Aug 5)
  - Game-level dedup (one row per game_id)
  - 3 random seeds
  - GBC: n_estimators=500, max_depth=4, lr=0.03, subsample=0.8, min_samples_leaf=5
  - Compare: Baseline (21 feats) vs enhanced variants with residual features

Variants
--------
  A. Baseline (V1_Pruned, 21 features)
  B. + Expanding mean residual (home_team_mean_resid, away_team_mean_resid)
  C. + EWM residual (span=20, recent games weighted more)
  D. + Context-split residual (separate home-as-home / away-as-away residuals)
  E. + Residual interaction (home_resid * away_resid)
  F. Kitchen sink (B + C + D + E combined)
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, "/Users/joseportillo/Downloads/BetIQ")
sys.path.insert(0, "/Users/joseportillo/Downloads/BetIQ/backend")

from backend.features import (
    build_feature_matrix,
    MODEL_FEATURE_COLS_V1_PRUNED,
    TARGET_ATS,
)

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_CUTOFF = "2026-05-26"
EARLY_START  = "2026-05-26"
EARLY_END    = "2026-06-15"
LATE_START   = "2026-07-01"
LATE_END     = "2026-08-05"
SEEDS        = [42, 123, 777]
BE_THRESHOLD = 0.524  # breakeven WR at typical -110 odds

GBC_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    min_samples_leaf=5,
)


# ═══════════════════════════════════════════════════════════════════════════════
# RESIDUAL COMPUTATION (leak-free)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_baseline_predictions(df: pd.DataFrame, feat_cols: list[str],
                                  seed: int) -> pd.Series:
    """
    Train a baseline GBC on pre-cutoff data and return predicted P(cover)
    for ALL rows.  These predictions are used to compute residuals.

    Leak-free because:
    - Model is trained ONLY on pre-cutoff data
    - Predictions on post-cutoff rows are genuine OOS
    - For pre-cutoff rows, predictions are in-sample but residuals from these
      are only used as features for LATER games (expanding window + shift)
    """
    df = df.copy().sort_values("game_date").reset_index(drop=True)
    train_mask = df["game_date"] < pd.Timestamp(TRAIN_CUTOFF)
    train_sub = df[train_mask].copy()

    avail = [c for c in feat_cols if c in df.columns]
    X_tr = train_sub[avail].values.astype(float)
    y_tr = train_sub[TARGET_ATS].values.astype(float)

    valid = ~np.isnan(y_tr)
    X_tr, y_tr = X_tr[valid], y_tr[valid]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr_imp = sc.fit_transform(imp.fit_transform(X_tr))

    gbc = GradientBoostingClassifier(**GBC_PARAMS, random_state=seed)
    gbc.fit(X_tr_imp, y_tr)

    X_all = df[avail].values.astype(float)
    X_all_imp = sc.transform(imp.transform(X_all))
    preds = gbc.predict_proba(X_all_imp)[:, 1]

    return pd.Series(preds, index=df.index)


def add_expanding_mean_residual(df: pd.DataFrame, feat_cols: list[str],
                                 seed: int) -> pd.DataFrame:
    """
    Variant B: Expanding mean residual per team.

    For each game, each team's residual feature = mean(actual - predicted)
    from ALL prior games involving that team, shifted by 1 to exclude
    the current game.  First game for each team gets NaN (no prior data).
    """
    df = df.copy().sort_values("game_date").reset_index(drop=True)
    preds = compute_baseline_predictions(df, feat_cols, seed)

    # Raw residuals: actual cover - predicted cover
    df["_resid_home"] = df[TARGET_ATS] - preds  # from home team perspective
    df["_resid_away"] = (1 - df[TARGET_ATS]) - (1 - preds)  # equivalent: preds - df[TARGET_ATS] ... wait
    # Actually from away team perspective: away covers when home does NOT cover
    # So away_actual = 1 - home_actual, away_pred = 1 - home_pred
    # away_resid = (1 - home_actual) - (1 - home_pred) = home_pred - home_actual = -home_resid
    # More useful: track each team's residual across ALL their games (home + away)

    # Build a long-form: one row per team-game
    # For home team: residual = actual - predicted (positive = team covers more than expected)
    # For away team: residual = (1 - actual) - (1 - predicted) = -(actual - predicted)
    home_records = df[["game_date", "home_team", "_resid_home"]].rename(
        columns={"home_team": "team", "_resid_home": "resid"})
    away_records = df[["game_date", "away_team"]].copy()
    away_records["resid"] = -df["_resid_home"]  # away team's residual is negated
    away_records = away_records.rename(columns={"away_team": "team"})

    team_long = pd.concat([home_records, away_records], ignore_index=True)
    team_long = team_long.sort_values("game_date").reset_index(drop=True)

    # For each team, expanding mean with shift(1)
    team_mean_resid = {}
    for team, grp in team_long.groupby("team"):
        grp = grp.sort_values("game_date")
        expanding = grp["resid"].expanding().mean().shift(1)
        # Store as dict: (team, row_index_in_grp) -> value
        # We need to map back to the original df indices
        team_mean_resid[team] = expanding.values

    # Now map back: for each game in df, look up home_team's and away_team's
    # expanding mean residual at that point in time
    # We need to track how many games each team has played up to each game
    team_game_counter = defaultdict(int)
    home_resid_vals = np.full(len(df), np.nan)
    away_resid_vals = np.full(len(df), np.nan)

    # Rebuild per-team expanding mean more carefully using original df order
    # Build per-team game lists in chronological order
    team_home_games = defaultdict(list)  # team -> [(df_idx, resid)]
    team_away_games = defaultdict(list)

    for idx in df.index:
        ht = df.loc[idx, "home_team"]
        at = df.loc[idx, "away_team"]
        resid_h = df.loc[idx, "_resid_home"]
        team_home_games[ht].append((idx, resid_h))
        team_away_games[at].append((idx, -resid_h))

    # For each team, merge home+away games chronologically, compute expanding mean
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    team_expanding = {}  # team -> {df_idx: expanding_mean_value}

    for team in all_teams:
        # Collect all games for this team
        games = []
        for idx, resid in team_home_games.get(team, []):
            games.append((df.loc[idx, "game_date"], idx, resid, "home"))
        for idx, resid in team_away_games.get(team, []):
            games.append((df.loc[idx, "game_date"], idx, resid, "away"))

        # Sort chronologically
        games.sort(key=lambda x: (x[0], x[1]))

        # Expanding mean with shift(1)
        resid_so_far = []
        idx_to_val = {}
        for date, idx, resid, side in games:
            if len(resid_so_far) > 0 and not np.isnan(np.nanmean(resid_so_far)):
                idx_to_val[idx] = np.nanmean(resid_so_far)
            else:
                idx_to_val[idx] = np.nan
            if not np.isnan(resid):
                resid_so_far.append(resid)

        team_expanding[team] = idx_to_val

    # Map back to df
    for idx in df.index:
        ht = df.loc[idx, "home_team"]
        at = df.loc[idx, "away_team"]
        if ht in team_expanding and idx in team_expanding[ht]:
            home_resid_vals[df.index.get_loc(idx)] = team_expanding[ht][idx]
        if at in team_expanding and idx in team_expanding[at]:
            away_resid_vals[df.index.get_loc(idx)] = team_expanding[at][idx]

    df["home_team_mean_resid"] = home_resid_vals
    df["away_team_mean_resid"] = away_resid_vals

    # Clean up
    df.drop(columns=["_resid_home", "_resid_away"], inplace=True, errors="ignore")

    return df


def add_ewm_residual(df: pd.DataFrame, feat_cols: list[str],
                      seed: int, span: int = 20) -> pd.DataFrame:
    """
    Variant C: Exponentially weighted mean residual.

    Recent games weighted more heavily (span=20 ~ half-life of ~10 games).
    """
    df = df.copy().sort_values("game_date").reset_index(drop=True)
    preds = compute_baseline_predictions(df, feat_cols, seed)
    df["_resid_home"] = df[TARGET_ATS] - preds

    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

    # Build per-team game lists
    team_home_games = defaultdict(list)
    team_away_games = defaultdict(list)
    for idx in df.index:
        ht = df.loc[idx, "home_team"]
        at = df.loc[idx, "away_team"]
        resid_h = df.loc[idx, "_resid_home"]
        team_home_games[ht].append((idx, resid_h))
        team_away_games[at].append((idx, -resid_h))

    home_ewm_vals = np.full(len(df), np.nan)
    away_ewm_vals = np.full(len(df), np.nan)
    alpha = 2.0 / (span + 1)

    for team in all_teams:
        games = []
        for idx, resid in team_home_games.get(team, []):
            games.append((df.loc[idx, "game_date"], idx, resid, "home"))
        for idx, resid in team_away_games.get(team, []):
            games.append((df.loc[idx, "game_date"], idx, resid, "away"))
        games.sort(key=lambda x: (x[0], x[1]))

        # EWM with shift(1)
        ewm_val = np.nan
        for date, idx, resid, side in games:
            pos = df.index.get_loc(idx)
            if side == "home":
                home_ewm_vals[pos] = ewm_val
            else:
                away_ewm_vals[pos] = ewm_val

            if not np.isnan(resid):
                if np.isnan(ewm_val):
                    ewm_val = resid
                else:
                    ewm_val = alpha * resid + (1 - alpha) * ewm_val

    df["home_team_ewm_resid"] = home_ewm_vals
    df["away_team_ewm_resid"] = away_ewm_vals
    df.drop(columns=["_resid_home"], inplace=True, errors="ignore")

    return df


def add_context_residual(df: pd.DataFrame, feat_cols: list[str],
                          seed: int) -> pd.DataFrame:
    """
    Variant D: Context-split residuals.

    Separate residuals for when team plays at HOME vs AWAY.
    A team might systematically over/under-perform its profile specifically
    at home or on the road.
    """
    df = df.copy().sort_values("game_date").reset_index(drop=True)
    preds = compute_baseline_predictions(df, feat_cols, seed)
    df["_resid_home"] = df[TARGET_ATS] - preds

    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

    home_ctx_vals = np.full(len(df), np.nan)  # home team's home-context residual
    away_ctx_vals = np.full(len(df), np.nan)  # away team's away-context residual

    for team in all_teams:
        # Home context: only games where this team was HOME
        h_indices = df.index[df["home_team"] == team].tolist()
        h_resids = []
        for idx in h_indices:
            pos = df.index.get_loc(idx)
            if len(h_resids) > 0:
                home_ctx_vals[pos] = np.nanmean(h_resids)
            resid = df.loc[idx, "_resid_home"]
            if not np.isnan(resid):
                h_resids.append(resid)

        # Away context: only games where this team was AWAY
        a_indices = df.index[df["away_team"] == team].tolist()
        a_resids = []
        for idx in a_indices:
            pos = df.index.get_loc(idx)
            # Away team covers when home does NOT: resid for away = -(home resid)
            if len(a_resids) > 0:
                away_ctx_vals[pos] = np.nanmean(a_resids)
            resid = -df.loc[idx, "_resid_home"]
            if not np.isnan(resid):
                a_resids.append(resid)

    df["home_team_home_ctx_resid"] = home_ctx_vals
    df["away_team_away_ctx_resid"] = away_ctx_vals
    df.drop(columns=["_resid_home"], inplace=True, errors="ignore")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(df: pd.DataFrame, feat_cols: list[str], seed: int,
                       label: str) -> dict:
    """Train GBC on pre-cutoff data, evaluate on early and late windows."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df[TARGET_ATS].notna()].copy()
    feat_cols = [c for c in feat_cols if c in df.columns]

    train = df[df["game_date"] < pd.Timestamp(TRAIN_CUTOFF)].copy()
    early = df[(df["game_date"] >= pd.Timestamp(EARLY_START)) &
               (df["game_date"] <= pd.Timestamp(EARLY_END))].copy()
    late  = df[(df["game_date"] >= pd.Timestamp(LATE_START)) &
               (df["game_date"] <= pd.Timestamp(LATE_END))].copy()

    if len(train) < 50 or len(early) < 20 or len(late) < 20:
        return {"label": label, "seed": seed, "error": "insufficient data",
                "n_train": len(train), "n_early": len(early), "n_late": len(late)}

    X_tr = train[feat_cols].values.astype(float)
    y_tr = train[TARGET_ATS].values.astype(float)

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr_imp = sc.fit_transform(imp.fit_transform(X_tr))

    gbc = GradientBoostingClassifier(**GBC_PARAMS, random_state=seed)
    gbc.fit(X_tr_imp, y_tr)

    results = {"label": label, "seed": seed, "n_train": len(train),
               "n_features": len(feat_cols)}

    for period_name, period_df in [("early", early), ("late", late)]:
        X_te = period_df[feat_cols].values.astype(float)
        y_te = period_df[TARGET_ATS].values.astype(float)

        X_te_imp = sc.transform(imp.transform(X_te))
        y_prob = gbc.predict_proba(X_te_imp)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        wr = float((y_pred == y_te).mean())

        try:
            auc = float(roc_auc_score(y_te, y_prob))
        except ValueError:
            auc = float("nan")
        brier = float(brier_score_loss(y_te, y_prob))
        edge_pp = (wr - BE_THRESHOLD) * 100

        results[f"{period_name}_n"] = len(y_te)
        results[f"{period_name}_auc"] = round(auc, 4)
        results[f"{period_name}_brier"] = round(brier, 4)
        results[f"{period_name}_wr"] = round(wr * 100, 1)
        results[f"{period_name}_edge_pp"] = round(edge_pp, 1)

    # Feature importances (top 10)
    imps = sorted(zip(feat_cols, gbc.feature_importances_), key=lambda x: -x[1])[:10]
    results["top_features"] = [(name, round(float(v), 4)) for name, v in imps]

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(all_results: list[dict]):
    """Print summary table across all models and seeds."""
    grouped = defaultdict(list)
    for r in all_results:
        if "error" in r:
            print(f"  ERROR: {r['label']} seed={r['seed']}: {r['error']} "
                  f"(train={r.get('n_train',0)}, early={r.get('n_early',0)}, "
                  f"late={r.get('n_late',0)})")
            continue
        grouped[r["label"]].append(r)

    if not grouped:
        print("No valid results to display.")
        return

    print("\n" + "=" * 130)
    print(f"{'Model':<35} {'N_ft':>4} {'E_N':>5} {'E_AUC':>7} {'E_Brier':>8} "
          f"{'E_WR%':>6} {'E_Edge':>7} {'L_N':>5} {'L_AUC':>7} {'L_Brier':>8} "
          f"{'L_WR%':>6} {'L_Edge':>7} {'Decay':>7}")
    print("-" * 130)

    def _avg(runs, key):
        vals = [r[key] for r in runs if key in r and not np.isnan(r[key])]
        return np.mean(vals) if vals else float("nan")

    for label, runs in sorted(grouped.items()):
        n_feat = runs[0]["n_features"]
        e_n = int(_avg(runs, "early_n"))
        l_n = int(_avg(runs, "late_n"))
        e_auc = _avg(runs, "early_auc")
        l_auc = _avg(runs, "late_auc")
        decay = l_auc - e_auc

        print(f"{label:<35} {n_feat:>4} {e_n:>5} "
              f"{e_auc:>7.4f} {_avg(runs, 'early_brier'):>8.4f} "
              f"{_avg(runs, 'early_wr'):>5.1f}% {_avg(runs, 'early_edge_pp'):>+6.1f}pp "
              f"{l_n:>5} "
              f"{l_auc:>7.4f} {_avg(runs, 'late_brier'):>8.4f} "
              f"{_avg(runs, 'late_wr'):>5.1f}% {_avg(runs, 'late_edge_pp'):>+6.1f}pp "
              f"{decay:>+7.4f}")

    print("=" * 130)

    # Per-seed detail
    print("\n--- Per-seed detail ---")
    for label, runs in sorted(grouped.items()):
        print(f"\n  {label}:")
        for r in sorted(runs, key=lambda x: x["seed"]):
            decay = r["late_auc"] - r["early_auc"]
            print(f"    seed={r['seed']:>3}: "
                  f"Early AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}% | "
                  f"Decay={decay:+.4f}")
        if runs[0].get("top_features"):
            print(f"    Top features (seed {runs[0]['seed']}): "
                  f"{', '.join(f'{n}={v:.3f}' for n, v in runs[0]['top_features'][:5])}")

    # Delta vs baseline
    if "A_Baseline (V1_Pruned)" in grouped:
        base = grouped["A_Baseline (V1_Pruned)"]
        base_e_auc = _avg(base, "early_auc")
        base_l_auc = _avg(base, "late_auc")
        base_e_wr  = _avg(base, "early_wr")
        base_l_wr  = _avg(base, "late_wr")
        base_decay = base_l_auc - base_e_auc

        print("\n--- Delta vs Baseline (averaged across seeds) ---")
        for label, runs in sorted(grouped.items()):
            if label == "A_Baseline (V1_Pruned)":
                continue
            e_auc = _avg(runs, "early_auc")
            l_auc = _avg(runs, "late_auc")
            e_wr  = _avg(runs, "early_wr")
            l_wr  = _avg(runs, "late_wr")
            decay = l_auc - e_auc
            print(f"\n  {label}:")
            print(f"    Early AUC: {e_auc - base_e_auc:+.4f}  |  Late AUC: {l_auc - base_l_auc:+.4f}")
            print(f"    Early WR:  {e_wr - base_e_wr:+.1f}pp  |  Late WR:  {l_wr - base_l_wr:+.1f}pp")
            print(f"    Decay: baseline={base_decay:+.4f}  this={decay:+.4f}  "
                  f"improvement={decay - base_decay:+.4f}")


def print_residual_diagnostics(df: pd.DataFrame):
    """Print diagnostics about the residual features to verify sanity."""
    print("\n--- Residual Feature Diagnostics ---")

    for col in ["home_team_mean_resid", "away_team_mean_resid",
                "home_team_ewm_resid", "away_team_ewm_resid",
                "home_team_home_ctx_resid", "away_team_away_ctx_resid"]:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {col}: N={len(vals)}, mean={vals.mean():.4f}, "
                  f"std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

    # Check for leakage: residual features should be NaN for first game of each team
    if "home_team_mean_resid" in df.columns:
        train_df = df[df["game_date"] < pd.Timestamp(TRAIN_CUTOFF)]
        test_df  = df[df["game_date"] >= pd.Timestamp(TRAIN_CUTOFF)]
        train_fill = train_df["home_team_mean_resid"].notna().mean()
        test_fill  = test_df["home_team_mean_resid"].notna().mean()
        print(f"\n  home_team_mean_resid fill rate: train={train_fill:.1%}, test={test_fill:.1%}")

    # Top/bottom teams by mean residual (training period)
    if "home_team_mean_resid" in df.columns:
        train_df = df[df["game_date"] < pd.Timestamp(TRAIN_CUTOFF)]
        last_resid = {}
        for team in train_df["home_team"].unique():
            mask = train_df["home_team"] == team
            vals = train_df.loc[mask, "home_team_mean_resid"].dropna()
            if len(vals) > 0:
                last_resid[team] = vals.iloc[-1]

        if last_resid:
            sorted_teams = sorted(last_resid.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Top 5 teams by end-of-training residual (over-performers):")
            for team, val in sorted_teams[:5]:
                print(f"    {team}: {val:+.4f}")
            print(f"  Bottom 5 teams (under-performers):")
            for team, val in sorted_teams[-5:]:
                print(f"    {team}: {val:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("RESIDUAL-BASED TEAM FEATURE TEST — MLB ATS Late-Season Decay")
    print("=" * 80)
    print("\nRationale: K-means on profile features = +0.000 AUC (redundant).")
    print("Team residual variance is 2.8-5.3x random (p<0.001).")
    print("Signal is in the RESIDUAL, not the profile. Test: inject it directly.")

    # ── 1. Build feature matrix ───────────────────────────────────────────────
    print("\n[1] Building MLB feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"    Raw matrix: {len(df)} rows, {len(df.columns)} columns")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df[TARGET_ATS].notna()].copy()
    print(f"    With ATS labels: {len(df)} rows")

    # Game-level dedup
    df = df.drop_duplicates(subset=["game_id"], keep="first").copy()
    print(f"    After game-level dedup: {len(df)} rows")

    print(f"    Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"    Teams: {df['home_team'].nunique()}")

    train_n = len(df[df["game_date"] < pd.Timestamp(TRAIN_CUTOFF)])
    early_n = len(df[(df["game_date"] >= pd.Timestamp(EARLY_START)) &
                     (df["game_date"] <= pd.Timestamp(EARLY_END))])
    late_n  = len(df[(df["game_date"] >= pd.Timestamp(LATE_START)) &
                     (df["game_date"] <= pd.Timestamp(LATE_END))])
    print(f"    Train (<{TRAIN_CUTOFF}): {train_n} games")
    print(f"    Early ({EARLY_START} to {EARLY_END}): {early_n} games")
    print(f"    Late  ({LATE_START} to {LATE_END}):  {late_n} games")

    # ── 2. Baseline features ─────────────────────────────────────────────────
    baseline_feats = [c for c in MODEL_FEATURE_COLS_V1_PRUNED if c in df.columns]
    print(f"\n[2] Baseline features (V1_Pruned): {len(baseline_feats)}")
    for f in baseline_feats:
        print(f"      {f}")

    all_results = []

    # ── 3. Run experiments ────────────────────────────────────────────────────
    for seed in SEEDS:
        print(f"\n{'=' * 70}")
        print(f"  SEED {seed}")
        print(f"{'=' * 70}")

        # ── A. Baseline ──────────────────────────────────────────────────────
        print(f"\n  [A] Baseline (V1_Pruned, {len(baseline_feats)} features)...")
        r = train_and_evaluate(df, baseline_feats, seed, "A_Baseline (V1_Pruned)")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── B. + Expanding mean residual ─────────────────────────────────────
        print(f"\n  [B] + Expanding mean residual...")
        df_b = add_expanding_mean_residual(df, baseline_feats, seed)
        b_feats = baseline_feats + ["home_team_mean_resid", "away_team_mean_resid"]
        r = train_and_evaluate(df_b, b_feats, seed, "B_+Expanding Mean Resid")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # Print diagnostics on first seed only
        if seed == SEEDS[0]:
            print_residual_diagnostics(df_b)

        # ── C. + EWM residual (span=20) ─────────────────────────────────────
        print(f"\n  [C] + EWM residual (span=20)...")
        df_c = add_ewm_residual(df, baseline_feats, seed, span=20)
        c_feats = baseline_feats + ["home_team_ewm_resid", "away_team_ewm_resid"]
        r = train_and_evaluate(df_c, c_feats, seed, "C_+EWM Resid (span=20)")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── D. + Context-split residual ─────────────────────────────────────
        print(f"\n  [D] + Context-split residual (home-as-home, away-as-away)...")
        df_d = add_context_residual(df, baseline_feats, seed)
        d_feats = baseline_feats + ["home_team_home_ctx_resid", "away_team_away_ctx_resid"]
        r = train_and_evaluate(df_d, d_feats, seed, "D_+Context Resid")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── E. + Residual interaction ────────────────────────────────────────
        print(f"\n  [E] + Residual interaction (mean_resid + interaction)...")
        df_e = add_expanding_mean_residual(df, baseline_feats, seed)
        df_e["resid_interaction"] = (df_e["home_team_mean_resid"].fillna(0) *
                                     df_e["away_team_mean_resid"].fillna(0))
        df_e["resid_diff"] = (df_e["home_team_mean_resid"].fillna(0) -
                              df_e["away_team_mean_resid"].fillna(0))
        e_feats = baseline_feats + ["home_team_mean_resid", "away_team_mean_resid",
                                     "resid_interaction", "resid_diff"]
        r = train_and_evaluate(df_e, e_feats, seed, "E_+Resid Interaction")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── F. Kitchen sink ──────────────────────────────────────────────────
        print(f"\n  [F] Kitchen sink (all residual variants)...")
        df_f = add_expanding_mean_residual(df, baseline_feats, seed)
        # Add EWM
        preds_f = compute_baseline_predictions(df, baseline_feats, seed)
        df_tmp_c = add_ewm_residual(df, baseline_feats, seed, span=20)
        df_f["home_team_ewm_resid"] = df_tmp_c["home_team_ewm_resid"].values
        df_f["away_team_ewm_resid"] = df_tmp_c["away_team_ewm_resid"].values
        # Add context
        df_tmp_d = add_context_residual(df, baseline_feats, seed)
        df_f["home_team_home_ctx_resid"] = df_tmp_d["home_team_home_ctx_resid"].values
        df_f["away_team_away_ctx_resid"] = df_tmp_d["away_team_away_ctx_resid"].values
        # Add interaction
        df_f["resid_interaction"] = (df_f["home_team_mean_resid"].fillna(0) *
                                     df_f["away_team_mean_resid"].fillna(0))
        df_f["resid_diff"] = (df_f["home_team_mean_resid"].fillna(0) -
                              df_f["away_team_mean_resid"].fillna(0))

        f_feats = baseline_feats + [
            "home_team_mean_resid", "away_team_mean_resid",
            "home_team_ewm_resid", "away_team_ewm_resid",
            "home_team_home_ctx_resid", "away_team_away_ctx_resid",
            "resid_interaction", "resid_diff",
        ]
        r = train_and_evaluate(df_f, f_feats, seed, "F_Kitchen Sink")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n\n")
    print_results_table(all_results)

    # ── 5. Verdict ────────────────────────────────────────────────────────────
    grouped = defaultdict(list)
    for r in all_results:
        if "error" not in r:
            grouped[r["label"]].append(r)

    def _avg(runs, key):
        vals = [r[key] for r in runs if key in r and not np.isnan(r[key])]
        return np.mean(vals) if vals else float("nan")

    if "A_Baseline (V1_Pruned)" in grouped:
        base = grouped["A_Baseline (V1_Pruned)"]
        base_l_auc = _avg(base, "late_auc")
        base_e_auc = _avg(base, "early_auc")
        base_l_wr  = _avg(base, "late_wr")
        base_decay = base_l_auc - base_e_auc

        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        print(f"\nBaseline decay (late AUC - early AUC): {base_decay:+.4f}")
        print(f"Baseline late AUC: {base_l_auc:.4f}, late WR: {base_l_wr:.1f}%")

        any_improvement = False
        best_label = None
        best_late_delta = -999

        for label, runs in sorted(grouped.items()):
            if label == "A_Baseline (V1_Pruned)":
                continue
            e_auc = _avg(runs, "early_auc")
            l_auc = _avg(runs, "late_auc")
            l_wr  = _avg(runs, "late_wr")
            decay = l_auc - e_auc
            late_delta = l_auc - base_l_auc

            if late_delta > best_late_delta:
                best_late_delta = late_delta
                best_label = label

            if late_delta > 0.005:
                any_improvement = True
                print(f"\n  {label}: Late AUC {late_delta:+.4f} (decay {decay:+.4f} vs baseline {base_decay:+.4f})")
                print(f"    Late WR: {l_wr:.1f}% (delta {l_wr - base_l_wr:+.1f}pp)")
                print(f"    -> Residual feature HELPS late-season prediction")
            else:
                print(f"\n  {label}: Late AUC delta {late_delta:+.4f} (not meaningful)")

        print(f"\n{'─' * 60}")
        if any_improvement:
            print(f"CONCLUSION: Residual-based features improve late-season AUC.")
            print(f"Best variant: {best_label} (late AUC {best_late_delta:+.4f})")
            print(f"The team-specific signal in the residual IS capturable as a feature.")
        else:
            print(f"CONCLUSION: Residual-based features do NOT improve late-season AUC.")
            print(f"Best variant: {best_label} (late AUC {best_late_delta:+.4f})")
            print(f"\nDespite 2.8-5.3x non-random team residual variance, injecting")
            print(f"historical mean residuals as features does not help the GBC.")
            print(f"Possible explanations:")
            print(f"  1. Signal is real but too noisy per-team (30 teams, ~50 games each)")
            print(f"  2. Residual variance is NON-STATIONARY — a team's bias shifts")
            print(f"     over the season, so historical mean is a lagging indicator")
            print(f"  3. The GBC already captures team identity IMPLICITLY through")
            print(f"     unique feature-value combinations (SP ERA + park + SOS)")
            print(f"  4. The residual variance is real but UNPREDICTABLE — driven by")
            print(f"     unobservable factors (injuries, clubhouse, umpires)")
        print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
