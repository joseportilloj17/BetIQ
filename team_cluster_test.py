"""
team_cluster_test.py — Test whether team-specific features improve late-season
MLB ATS prediction accuracy.

Hypothesis: the baseline ATS model (V1_PRUNED, 21 features) shows +7pp edge
in late May / early June but decays to breakeven by July.  Adding team-specific
representation (clusters, form residuals, identity encoding) might restore
late-season accuracy by capturing team tendencies that emerge as the season
progresses.

Protocol:
  - Train on data before 2026-05-26 (same as prior baseline tests)
  - Game-level dedup: one row per game (use +1.5 spread or close_spread)
  - No spread augmentation
  - GBC hyperparams: n_estimators=500, max_depth=4, lr=0.03, subsample=0.8, min_samples_leaf=5
  - 3 random seeds
  - Evaluate on Early (May 26 – Jun 15) and Late (Jul 1 – Aug 5) windows
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
TRAIN_CUTOFF    = "2026-05-26"
EARLY_START     = "2026-05-26"
EARLY_END       = "2026-06-15"
LATE_START      = "2026-07-01"
LATE_END        = "2026-08-05"
SEEDS           = [42, 123, 777]
BE_THRESHOLD    = 0.524   # breakeven WR at typical -110 odds

# GBC hyperparams (match production diagnostic)
GBC_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    min_samples_leaf=5,
)

# Clustering features — team-season averages of key stats
CLUSTER_PROFILE_COLS = [
    "h_sp_era", "h_sp_whip", "h_ops", "h_slg",
    "h_k_pct", "h_bb_pct", "h_opp_impl_prob_5g",
    "h_margin_10g", "h_win_pct_10g",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def dedup_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per game (dedup by game_id)."""
    return df.drop_duplicates(subset=["game_id"], keep="first").copy()


def compute_team_clusters(df_train: pd.DataFrame, df_full: pd.DataFrame,
                          n_clusters: int = 5) -> pd.DataFrame:
    """
    Option A: K-means cluster teams by their average feature profiles.

    Fit clusters on training data only, then predict cluster for all teams.
    Returns df_full with home_cluster, away_cluster, cluster_interaction added.
    """
    # Build team-season profile from training data
    # For each home team, compute average of profile columns
    home_profiles = (
        df_train.groupby("home_team")[CLUSTER_PROFILE_COLS]
        .mean()
        .rename(columns={c: c.replace("h_", "") for c in CLUSTER_PROFILE_COLS})
    )
    # For each away team, compute similarly
    away_profile_cols = [c.replace("h_", "a_") for c in CLUSTER_PROFILE_COLS]
    away_profile_cols_present = [c for c in away_profile_cols if c in df_train.columns]
    away_profiles = (
        df_train.groupby("away_team")[away_profile_cols_present]
        .mean()
        .rename(columns={c: c.replace("a_", "") for c in away_profile_cols_present})
    )

    # Combine home + away profiles per team
    all_profiles = pd.concat([home_profiles, away_profiles]).groupby(level=0).mean()

    # Impute and scale
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    profile_vals = sc.fit_transform(imp.fit_transform(all_profiles.values))

    # Fit K-means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(profile_vals)

    # Map team -> cluster
    team_cluster = dict(zip(all_profiles.index, km.labels_))

    # Apply to full dataframe
    df_out = df_full.copy()
    df_out["home_cluster"] = df_out["home_team"].map(team_cluster).fillna(-1).astype(float)
    df_out["away_cluster"] = df_out["away_team"].map(team_cluster).fillna(-1).astype(float)
    df_out["cluster_interaction"] = df_out["home_cluster"] * 10 + df_out["away_cluster"]

    return df_out


def compute_form_residuals(df: pd.DataFrame, baseline_feat_cols: list[str],
                           seed: int, window: int = 15) -> pd.DataFrame:
    """
    Option B: Team-specific form residuals.

    For each team, compute rolling residual: actual cover rate minus
    model-predicted cover rate over last N games.
    This captures "team X is currently overperforming its profile."

    Must be computed sequentially to avoid lookahead.
    """
    df_out = df.copy()
    df_out = df_out.sort_values("game_date").reset_index(drop=True)

    # Train a quick baseline model on early data to get predictions
    train_mask = df_out["game_date"] < pd.Timestamp(TRAIN_CUTOFF)
    train_sub = df_out[train_mask].copy()

    feat_cols = [c for c in baseline_feat_cols if c in train_sub.columns]
    X_tr = train_sub[feat_cols].values.astype(float)
    y_tr = train_sub[TARGET_ATS].values.astype(float)

    valid = ~np.isnan(y_tr)
    X_tr = X_tr[valid]
    y_tr = y_tr[valid]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr_imp = sc.fit_transform(imp.fit_transform(X_tr))

    gbc = GradientBoostingClassifier(**GBC_PARAMS, random_state=seed)
    gbc.fit(X_tr_imp, y_tr)

    # Predict on ALL rows
    X_all = df_out[feat_cols].values.astype(float)
    X_all_imp = sc.transform(imp.transform(X_all))
    df_out["_pred_prob"] = gbc.predict_proba(X_all_imp)[:, 1]

    # Compute rolling residual per team (home perspective and away perspective)
    # For home team: actual cover - predicted cover
    df_out["_home_residual_raw"] = df_out[TARGET_ATS] - df_out["_pred_prob"]
    # For away team: (1 - actual cover) - (1 - predicted cover) = predicted - actual
    # But from the away team's perspective, "covering" is the opposite
    df_out["_away_residual_raw"] = (1 - df_out[TARGET_ATS]) - (1 - df_out["_pred_prob"])

    # Rolling residual per team with shift to avoid lookahead
    home_resid = []
    away_resid = []

    # Build team-level rolling residuals
    for team in df_out["home_team"].unique():
        # Games where this team was HOME
        h_mask = df_out["home_team"] == team
        h_idx = df_out.index[h_mask]
        if len(h_idx) > 0:
            vals = df_out.loc[h_idx, "_home_residual_raw"]
            rolled = vals.shift(1).rolling(window, min_periods=3).mean()
            home_resid.append(pd.Series(rolled.values, index=h_idx))

    for team in df_out["away_team"].unique():
        a_mask = df_out["away_team"] == team
        a_idx = df_out.index[a_mask]
        if len(a_idx) > 0:
            vals = df_out.loc[a_idx, "_away_residual_raw"]
            rolled = vals.shift(1).rolling(window, min_periods=3).mean()
            away_resid.append(pd.Series(rolled.values, index=a_idx))

    df_out["home_form_residual"] = pd.concat(home_resid).reindex(df_out.index)
    df_out["away_form_residual"] = pd.concat(away_resid).reindex(df_out.index)

    # Clean up temp columns
    df_out.drop(columns=["_pred_prob", "_home_residual_raw", "_away_residual_raw"],
                inplace=True)

    return df_out


def add_team_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Option C: Label-encode team names.
    Let GBC learn team-specific effects directly.
    """
    df_out = df.copy()
    le = LabelEncoder()
    all_teams = pd.concat([df_out["home_team"], df_out["away_team"]]).unique()
    le.fit(all_teams)
    df_out["home_team_id"] = le.transform(df_out["home_team"]).astype(float)
    df_out["away_team_id"] = le.transform(df_out["away_team"]).astype(float)
    return df_out


def train_and_evaluate(df: pd.DataFrame, feat_cols: list[str], seed: int,
                       label: str) -> dict:
    """
    Train GBC on pre-cutoff data, evaluate on early and late windows.
    Returns dict with metrics.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Filter to rows with valid ATS labels
    df = df[df[TARGET_ATS].notna()].copy()

    # Only use features that exist
    feat_cols = [c for c in feat_cols if c in df.columns]

    # Split
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

    # Impute + scale
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr_imp = sc.fit_transform(imp.fit_transform(X_tr))

    # Train
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

        n = len(y_te)
        wr = float(y_pred[y_pred == y_te].shape[0]) / n if n > 0 else 0
        # More precise: accuracy
        wr = float((y_pred == y_te).mean())

        try:
            auc = float(roc_auc_score(y_te, y_prob))
        except ValueError:
            auc = float("nan")
        brier = float(brier_score_loss(y_te, y_prob))
        edge_pp = (wr - BE_THRESHOLD) * 100

        results[f"{period_name}_n"] = n
        results[f"{period_name}_auc"] = round(auc, 4)
        results[f"{period_name}_brier"] = round(brier, 4)
        results[f"{period_name}_wr"] = round(wr * 100, 1)
        results[f"{period_name}_edge_pp"] = round(edge_pp, 1)

    # Feature importances (top 10)
    imps = sorted(zip(feat_cols, gbc.feature_importances_), key=lambda x: -x[1])[:10]
    results["top_features"] = [(name, round(float(imp_val), 4)) for name, imp_val in imps]

    return results


def print_results_table(all_results: list[dict]):
    """Print a summary table across all models and seeds."""
    # Group by label, average across seeds
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        if "error" in r:
            print(f"  ERROR: {r['label']} seed={r['seed']}: {r['error']} "
                  f"(train={r.get('n_train',0)}, early={r.get('n_early',0)}, late={r.get('n_late',0)})")
            continue
        grouped[r["label"]].append(r)

    if not grouped:
        print("No valid results to display.")
        return

    print("\n" + "=" * 120)
    print(f"{'Model':<30} {'N_feat':>6} {'Early_N':>7} {'Early_AUC':>10} {'Early_Brier':>12} "
          f"{'Early_WR%':>9} {'Early_Edge':>10} {'Late_N':>7} {'Late_AUC':>10} {'Late_Brier':>12} "
          f"{'Late_WR%':>9} {'Late_Edge':>10}")
    print("-" * 120)

    for label, runs in sorted(grouped.items()):
        n_feat = runs[0]["n_features"]

        def _avg(key):
            vals = [r[key] for r in runs if key in r and not np.isnan(r[key])]
            return np.mean(vals) if vals else float("nan")

        def _std(key):
            vals = [r[key] for r in runs if key in r and not np.isnan(r[key])]
            return np.std(vals) if len(vals) > 1 else 0.0

        early_n = int(_avg("early_n"))
        late_n  = int(_avg("late_n"))

        print(f"{label:<30} {n_feat:>6} {early_n:>7} "
              f"{_avg('early_auc'):>10.4f} {_avg('early_brier'):>12.4f} "
              f"{_avg('early_wr'):>8.1f}% {_avg('early_edge_pp'):>+9.1f}pp "
              f"{late_n:>7} "
              f"{_avg('late_auc'):>10.4f} {_avg('late_brier'):>12.4f} "
              f"{_avg('late_wr'):>8.1f}% {_avg('late_edge_pp'):>+9.1f}pp")

    print("=" * 120)

    # Print per-seed detail
    print("\n--- Per-seed detail ---")
    for label, runs in sorted(grouped.items()):
        print(f"\n  {label}:")
        for r in sorted(runs, key=lambda x: x["seed"]):
            print(f"    seed={r['seed']:>3}: "
                  f"Early AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")
        if runs[0].get("top_features"):
            print(f"    Top features (seed {runs[0]['seed']}): "
                  f"{', '.join(f'{n}={v:.3f}' for n, v in runs[0]['top_features'][:5])}")

    # Delta analysis
    print("\n--- Delta: Enhanced vs Baseline (averaged across seeds) ---")
    if "A_Baseline (V1_Pruned)" in grouped:
        base = grouped["A_Baseline (V1_Pruned)"]
        base_early_auc = np.mean([r["early_auc"] for r in base])
        base_late_auc  = np.mean([r["late_auc"]  for r in base])
        base_early_wr  = np.mean([r["early_wr"]  for r in base])
        base_late_wr   = np.mean([r["late_wr"]   for r in base])

        for label, runs in sorted(grouped.items()):
            if label == "A_Baseline (V1_Pruned)":
                continue
            e_auc = np.mean([r["early_auc"] for r in runs])
            l_auc = np.mean([r["late_auc"]  for r in runs])
            e_wr  = np.mean([r["early_wr"]  for r in runs])
            l_wr  = np.mean([r["late_wr"]   for r in runs])
            print(f"  {label}:")
            print(f"    Early AUC delta: {e_auc - base_early_auc:+.4f}  |  Late AUC delta: {l_auc - base_late_auc:+.4f}")
            print(f"    Early WR delta:  {e_wr - base_early_wr:+.1f}pp  |  Late WR delta:  {l_wr - base_late_wr:+.1f}pp")
            decay_base = base_late_auc - base_early_auc
            decay_enh  = l_auc - e_auc
            print(f"    Decay (late-early AUC): baseline={decay_base:+.4f}  this={decay_enh:+.4f}  "
                  f"improvement={decay_enh - decay_base:+.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Team Cluster / Identity Test — MLB ATS Late-Season Decay")
    print("=" * 80)

    # ── 1. Build feature matrix ───────────────────────────────────────────────
    print("\n[1] Building MLB feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"    Raw matrix: {len(df)} rows, {len(df.columns)} columns")

    df["game_date"] = pd.to_datetime(df["game_date"])

    # Filter to rows with ATS labels
    df = df[df[TARGET_ATS].notna()].copy()
    print(f"    With ATS labels: {len(df)} rows")

    # Game-level dedup
    df = dedup_game_level(df)
    print(f"    After game-level dedup: {len(df)} rows")

    # Report date range
    print(f"    Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    # Report team counts
    teams = sorted(df["home_team"].unique())
    print(f"    Teams: {len(teams)}")

    # Report period sizes
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

    all_results = []

    # ── 3. Run experiments ────────────────────────────────────────────────────
    for seed in SEEDS:
        print(f"\n{'─' * 60}")
        print(f"  Seed {seed}")
        print(f"{'─' * 60}")

        # ── A. Baseline ──────────────────────────────────────────────────────
        print(f"  [A] Baseline (V1_Pruned, {len(baseline_feats)} features)...")
        r = train_and_evaluate(df, baseline_feats, seed, "A_Baseline (V1_Pruned)")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── B. Baseline + Team Clusters ──────────────────────────────────────
        print(f"  [B] + Team Clusters (K=5)...")
        train_mask = df["game_date"] < pd.Timestamp(TRAIN_CUTOFF)
        df_clust = compute_team_clusters(df[train_mask], df, n_clusters=5)
        cluster_feats = baseline_feats + ["home_cluster", "away_cluster", "cluster_interaction"]
        r = train_and_evaluate(df_clust, cluster_feats, seed, "B_+Clusters (K=5)")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── B2. Clusters K=8 ─────────────────────────────────────────────────
        print(f"  [B2] + Team Clusters (K=8)...")
        df_clust8 = compute_team_clusters(df[train_mask], df, n_clusters=8)
        cluster_feats8 = baseline_feats + ["home_cluster", "away_cluster", "cluster_interaction"]
        r = train_and_evaluate(df_clust8, cluster_feats8, seed, "B2_+Clusters (K=8)")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── C. Baseline + Form Residuals ─────────────────────────────────────
        print(f"  [C] + Form Residuals (window=15)...")
        df_resid = compute_form_residuals(df, baseline_feats, seed, window=15)
        resid_feats = baseline_feats + ["home_form_residual", "away_form_residual"]
        r = train_and_evaluate(df_resid, resid_feats, seed, "C_+Form Residuals")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── D. Baseline + Team Identity ──────────────────────────────────────
        print(f"  [D] + Team Identity (label-encoded)...")
        df_ident = add_team_identity(df)
        ident_feats = baseline_feats + ["home_team_id", "away_team_id"]
        r = train_and_evaluate(df_ident, ident_feats, seed, "D_+Team Identity")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

        # ── E. Kitchen sink: clusters + residuals + identity ─────────────────
        print(f"  [E] Kitchen sink (clusters + residuals + identity)...")
        df_sink = compute_team_clusters(df[train_mask], df_resid, n_clusters=5)
        df_sink = add_team_identity(df_sink)
        sink_feats = baseline_feats + [
            "home_cluster", "away_cluster", "cluster_interaction",
            "home_form_residual", "away_form_residual",
            "home_team_id", "away_team_id",
        ]
        r = train_and_evaluate(df_sink, sink_feats, seed, "E_Kitchen Sink")
        all_results.append(r)
        if "error" not in r:
            print(f"      Early: AUC={r['early_auc']:.4f} WR={r['early_wr']:.1f}% | "
                  f"Late: AUC={r['late_auc']:.4f} WR={r['late_wr']:.1f}%")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n")
    print_results_table(all_results)

    # ── 5. Verdict ────────────────────────────────────────────────────────────
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        if "error" not in r:
            grouped[r["label"]].append(r)

    if "A_Baseline (V1_Pruned)" in grouped:
        base_runs = grouped["A_Baseline (V1_Pruned)"]
        base_late_auc = np.mean([r["late_auc"] for r in base_runs])
        base_early_auc = np.mean([r["early_auc"] for r in base_runs])
        base_decay = base_late_auc - base_early_auc

        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        print(f"\nBaseline decay (late AUC - early AUC): {base_decay:+.4f}")

        any_improvement = False
        for label, runs in sorted(grouped.items()):
            if label == "A_Baseline (V1_Pruned)":
                continue
            e_auc = np.mean([r["early_auc"] for r in runs])
            l_auc = np.mean([r["late_auc"]  for r in runs])
            decay = l_auc - e_auc
            late_delta = l_auc - base_late_auc

            if late_delta > 0.005:
                any_improvement = True
                print(f"\n  {label}: Late AUC improved by {late_delta:+.4f} (decay {decay:+.4f} vs baseline {base_decay:+.4f})")
                print(f"    -> Team-specific features HELP late-season prediction")
            else:
                print(f"\n  {label}: Late AUC delta {late_delta:+.4f} (not meaningful)")

        if not any_improvement:
            print(f"\n  CONCLUSION: No team-specific feature variant improves late-season AUC by >0.5pp.")
            print(f"  The late-season decay is NOT caused by missing team identity information.")
            print(f"  The decay is likely driven by market adaptation (lines become more efficient)")
            print(f"  or by the model's features becoming stale (season-level stats saturate).")
        else:
            print(f"\n  CONCLUSION: At least one team-specific feature improves late-season prediction.")
            print(f"  The decay IS partially structural — team identity carries useful signal.")


if __name__ == "__main__":
    main()
