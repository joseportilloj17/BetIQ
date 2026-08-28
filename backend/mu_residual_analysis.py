"""
mu_residual_analysis.py — Find systematic mu biases by segment

Compute residuals (actual margin − predicted mu) on clean OOS data.
Bin by every available dimension. Biased pockets = missing information
the model should capture. Random residuals = mu is extracted.

Temporal 80/20 split (same boundary as model training).
Only report on TEST set (OOS).
"""
from __future__ import annotations
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
DATA_DIR = os.path.join(_HERE, "..", "data")
SUBMODEL_DIR = os.path.join(DATA_DIR, "submodels")

R_GLOBAL = 3.26


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_oos_residuals():
    """Load feature matrix, predict mu, compute residuals on OOS split."""
    from features import build_feature_matrix

    print("=" * 110)
    print("  MU RESIDUAL ANALYSIS — WHERE IS THE MODEL SYSTEMATICALLY BIASED?")
    print("=" * 110)

    df = build_feature_matrix("MLB")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    # NegBin model
    with open(os.path.join(SUBMODEL_DIR, "mlb_negbin_model.pkl"), "rb") as f:
        art = pickle.load(f)

    feat_cols = art["feature_cols"]
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].values.astype(float)

    mu_home = np.clip(art["gbm_home"].predict(X), 0.5, 15)
    mu_away = np.clip(art["gbm_away"].predict(X), 0.5, 15)

    df["pred_mu_home"] = mu_home
    df["pred_mu_away"] = mu_away
    df["pred_margin"] = mu_home - mu_away
    df["pred_total"] = mu_home + mu_away
    df["actual_margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["actual_total"] = df["home_score"].astype(float) + df["away_score"].astype(float)

    # Residuals
    df["resid_margin"] = df["actual_margin"] - df["pred_margin"]
    df["resid_home"] = df["home_score"].astype(float) - df["pred_mu_home"]
    df["resid_away"] = df["away_score"].astype(float) - df["pred_mu_away"]
    df["resid_total"] = df["actual_total"] - df["pred_total"]

    # Temporal split — same as training
    split = int(len(df) * 0.80)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    print(f"\n  Total games:  {len(df)}")
    print(f"  Train:        {len(train)} ({str(train['game_date'].iloc[0])[:10]} → "
          f"{str(train['game_date'].iloc[-1])[:10]})")
    print(f"  Test (OOS):   {len(test)} ({str(test['game_date'].iloc[0])[:10]} → "
          f"{str(test['game_date'].iloc[-1])[:10]})")

    # Global OOS stats
    print(f"\n  OOS residual stats (margin):")
    r = test["resid_margin"]
    print(f"    Mean: {r.mean():+.4f}  Median: {r.median():+.4f}  "
          f"SD: {r.std():.3f}  Skew: {r.skew():+.3f}  Kurt: {r.kurtosis():+.3f}")
    t_stat, p_val = sp_stats.ttest_1samp(r, 0)
    print(f"    t-test vs 0: t={t_stat:+.3f}, p={p_val:.4f}")

    return df, train, test


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def add_segment_columns(df):
    """Add all segmentation columns."""
    # Month
    df["month"] = pd.to_datetime(df["game_date"]).dt.month

    # Game total band (from close_total)
    ct = pd.to_numeric(df.get("close_total"), errors="coerce")
    df["total_band"] = pd.cut(ct, bins=[0, 7.5, 8.5, 9.5, 20],
                               labels=["≤7.5", "8–8.5", "9–9.5", "≥10"])

    # Predicted total band (terciles)
    df["pred_total_band"] = pd.qcut(df["pred_total"], q=3,
                                     labels=["pred_low", "pred_mid", "pred_high"],
                                     duplicates="drop")

    # Park factor band
    pf = pd.to_numeric(df.get("h_park_factor"), errors="coerce")
    df["park_band"] = pd.cut(pf, bins=[0, 0.95, 1.00, 1.05, 2.0],
                              labels=["pitcher (<0.95)", "neutral (0.95-1.0)",
                                      "slight hit (1.0-1.05)", "hitter (>1.05)"])

    # Rest advantage
    h_rest = pd.to_numeric(df.get("h_rest_days"), errors="coerce")
    a_rest = pd.to_numeric(df.get("a_rest_days"), errors="coerce")
    rest_adv = h_rest - a_rest
    df["rest_seg"] = "even"
    df.loc[rest_adv > 0.5, "rest_seg"] = "home_rested"
    df.loc[rest_adv < -0.5, "rest_seg"] = "away_rested"

    # Back-to-back
    h_btb = pd.to_numeric(df.get("h_is_btb"), errors="coerce")
    a_btb = pd.to_numeric(df.get("a_is_btb"), errors="coerce")
    df["btb_seg"] = "neither_btb"
    df.loc[h_btb == 1, "btb_seg"] = "home_btb"
    df.loc[a_btb == 1, "btb_seg"] = "away_btb"
    df.loc[(h_btb == 1) & (a_btb == 1), "btb_seg"] = "both_btb"

    # Season phase
    sp = pd.to_numeric(df.get("season_phase"), errors="coerce")
    df["season_seg"] = sp.map({0: "early", 1: "mid", 2: "late"})

    # |Predicted margin| band (favorite strength from model's perspective)
    df["fav_strength"] = pd.cut(df["pred_margin"].abs(),
                                 bins=[-0.1, 0.3, 0.8, 1.5, 20],
                                 labels=["toss-up (≤0.3)", "lean (0.3-0.8)",
                                         "clear (0.8-1.5)", "heavy (>1.5)"])

    # Pitcher ERA differential band
    era_diff = pd.to_numeric(df.get("sp_era_diff"), errors="coerce")
    df["era_diff_band"] = pd.cut(era_diff,
                                  bins=[-20, -1.5, -0.5, 0.5, 1.5, 20],
                                  labels=["home_ace (≤-1.5)", "home_better (-1.5 to -0.5)",
                                          "even (-0.5 to 0.5)",
                                          "away_better (0.5 to 1.5)", "away_ace (>1.5)"])

    # Season (year)
    df["season_year"] = pd.to_datetime(df["game_date"]).dt.year

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BIAS SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def scan_for_bias(test_df, segment_col, resid_col="resid_margin",
                  min_n=30, alpha=0.05, label=None):
    """
    For each bin in segment_col, test whether mean residual ≠ 0.
    Returns list of (segment_value, n, mean_resid, se, t, p, significant).
    """
    results = []
    for val, grp in test_df.groupby(segment_col, observed=True):
        r = grp[resid_col].dropna()
        n = len(r)
        if n < min_n:
            continue
        mean_r = r.mean()
        se = r.std() / np.sqrt(n)
        t_stat, p_val = sp_stats.ttest_1samp(r, 0)
        results.append({
            "segment": str(val),
            "n": n,
            "mean_resid": mean_r,
            "se": se,
            "t": t_stat,
            "p": p_val,
            "significant": p_val < alpha,
            "effect_runs": mean_r,  # in runs (margin perspective)
        })
    return results


def run_full_scan(test_df):
    """Scan all segment dimensions for bias."""
    test_df = add_segment_columns(test_df)

    segments = {
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Month": "month",
        "Season Phase": "season_seg",
        "Season Year": "season_year",
        "Total Band (close)": "total_band",
        "Pred Total Band": "pred_total_band",
        "Park Factor Band": "park_band",
        "Rest Advantage": "rest_seg",
        "Back-to-Back": "btb_seg",
        "Favorite Strength": "fav_strength",
        "ERA Diff Band": "era_diff_band",
    }

    all_results = []

    print(f"\n{'='*110}")
    print(f"  OOS RESIDUAL BIAS SCAN (margin residuals, test set only)")
    print(f"{'='*110}")

    for label, col in segments.items():
        if col not in test_df.columns:
            print(f"\n  ── {label} ── [SKIP: column missing]")
            continue

        results = scan_for_bias(test_df, col, "resid_margin")
        if not results:
            print(f"\n  ── {label} ── [SKIP: no bins with N≥30]")
            continue

        sig_count = sum(1 for r in results if r["significant"])
        n_bins = len(results)
        # Bonferroni-corrected threshold for this dimension
        bonf_alpha = 0.05 / n_bins

        print(f"\n  ── {label} ({n_bins} bins, {sig_count} raw-significant, "
              f"Bonferroni α={bonf_alpha:.4f}) ──")
        print(f"  {'Segment':<30} {'N':>5} {'Mean Resid':>11} {'SE':>8} "
              f"{'t':>8} {'p':>10} {'Sig':>5} {'Bonf':>5}")
        print(f"  {'─'*30} {'─'*5} {'─'*11} {'─'*8} {'─'*8} {'─'*10} {'─'*5} {'─'*5}")

        for r in sorted(results, key=lambda x: x["p"]):
            bonf_sig = "***" if r["p"] < bonf_alpha else ""
            raw_sig = "*" if r["significant"] else ""
            print(f"  {r['segment']:<30} {r['n']:>5} {r['mean_resid']:>+11.4f} "
                  f"{r['se']:>8.4f} {r['t']:>+8.3f} {r['p']:>10.4f} "
                  f"{raw_sig:>5} {bonf_sig:>5}")

            r["dimension"] = label
            r["bonferroni_sig"] = r["p"] < bonf_alpha
            all_results.append(r)

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME/AWAY SIDE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def side_decomposition(test_df):
    """Check if bias is in home prediction, away prediction, or both."""
    test_df = add_segment_columns(test_df)

    print(f"\n{'='*110}")
    print(f"  SIDE DECOMPOSITION: Is bias in home μ, away μ, or both?")
    print(f"{'='*110}")

    # Global
    for resid_name, col in [("margin", "resid_margin"),
                             ("home_runs", "resid_home"),
                             ("away_runs", "resid_away"),
                             ("total", "resid_total")]:
        r = test_df[col].dropna()
        t_stat, p_val = sp_stats.ttest_1samp(r, 0)
        print(f"  {resid_name:<12}: mean={r.mean():+.4f}  SD={r.std():.3f}  "
              f"t={t_stat:+.3f}  p={p_val:.4f}")

    # Per significant dimension, decompose
    key_segs = ["total_band", "park_band", "era_diff_band"]
    for seg_col in key_segs:
        if seg_col not in test_df.columns:
            continue
        print(f"\n  Side decomposition by {seg_col}:")
        print(f"  {'Segment':<30} {'Margin':>10} {'Home_μ':>10} {'Away_μ':>10} {'Total':>10} {'N':>5}")
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*5}")
        for val, grp in test_df.groupby(seg_col, observed=True):
            if len(grp) < 30:
                continue
            print(f"  {str(val):<30} {grp['resid_margin'].mean():>+10.4f} "
                  f"{grp['resid_home'].mean():>+10.4f} "
                  f"{grp['resid_away'].mean():>+10.4f} "
                  f"{grp['resid_total'].mean():>+10.4f} {len(grp):>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM-LEVEL DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════

def team_residual_deep_dive(test_df):
    """Per-team residual analysis — are some teams systematically mis-predicted?"""
    test_df = add_segment_columns(test_df)

    print(f"\n{'='*110}")
    print(f"  TEAM-LEVEL RESIDUAL DEEP DIVE (home games only, OOS)")
    print(f"{'='*110}")

    # For each team as home, check residuals
    team_results = []
    for team, grp in test_df.groupby("home_team"):
        n = len(grp)
        if n < 15:
            continue
        r = grp["resid_margin"]
        rh = grp["resid_home"]
        ra = grp["resid_away"]
        t_stat, p_val = sp_stats.ttest_1samp(r, 0)
        team_results.append({
            "team": team,
            "n": n,
            "mean_margin_resid": r.mean(),
            "mean_home_resid": rh.mean(),
            "mean_away_resid": ra.mean(),
            "t": t_stat,
            "p": p_val,
        })

    tdf = pd.DataFrame(team_results).sort_values("p")
    bonf = 0.05 / len(tdf)

    print(f"\n  Bonferroni α = {bonf:.5f} (30 teams)")
    print(f"  {'Team':<28} {'N':>4} {'Margin':>9} {'Home_μ':>9} {'Away_μ':>9} "
          f"{'t':>8} {'p':>10} {'Flag':>6}")
    print(f"  {'─'*28} {'─'*4} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*10} {'─'*6}")

    for _, r in tdf.iterrows():
        flag = "***" if r["p"] < bonf else ("*" if r["p"] < 0.05 else "")
        print(f"  {r['team']:<28} {r['n']:>4} {r['mean_margin_resid']:>+9.3f} "
              f"{r['mean_home_resid']:>+9.3f} {r['mean_away_resid']:>+9.3f} "
              f"{r['t']:>+8.3f} {r['p']:>10.5f} {flag:>6}")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def interaction_scan(test_df):
    """Check 2-way interactions: park × total, rest × era_diff, etc."""
    test_df = add_segment_columns(test_df)

    interactions = [
        ("park_band", "total_band", "Park × Total"),
        ("rest_seg", "era_diff_band", "Rest × ERA Diff"),
        ("btb_seg", "fav_strength", "B2B × Fav Strength"),
        ("season_seg", "total_band", "Season Phase × Total"),
    ]

    print(f"\n{'='*110}")
    print(f"  2-WAY INTERACTION SCAN (OOS)")
    print(f"{'='*110}")

    for col_a, col_b, label in interactions:
        if col_a not in test_df.columns or col_b not in test_df.columns:
            continue

        print(f"\n  ── {label} ──")
        print(f"  {'Combo':<50} {'N':>5} {'Mean Resid':>11} {'p':>10} {'Flag':>5}")
        print(f"  {'─'*50} {'─'*5} {'─'*11} {'─'*10} {'─'*5}")

        combos = []
        for (va, vb), grp in test_df.groupby([col_a, col_b], observed=True):
            r = grp["resid_margin"].dropna()
            if len(r) < 20:
                continue
            t_stat, p_val = sp_stats.ttest_1samp(r, 0)
            combos.append({
                "combo": f"{va} × {vb}",
                "n": len(r),
                "mean_resid": r.mean(),
                "p": p_val,
            })

        n_combos = len(combos)
        bonf = 0.05 / max(n_combos, 1)
        for c in sorted(combos, key=lambda x: x["p"]):
            flag = "**" if c["p"] < bonf else ("*" if c["p"] < 0.05 else "")
            print(f"  {c['combo']:<50} {c['n']:>5} {c['mean_resid']:>+11.4f} "
                  f"{c['p']:>10.4f} {flag:>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(results_df):
    """Print summary of findings."""
    print(f"\n{'='*110}")
    print(f"  SUMMARY: SYSTEMATIC MU BIASES")
    print(f"{'='*110}")

    # Bonferroni-significant findings
    bonf_sig = results_df[results_df["bonferroni_sig"] == True]
    raw_sig = results_df[results_df["significant"] == True]

    print(f"\n  Total bins tested: {len(results_df)}")
    print(f"  Raw significant (p<0.05): {len(raw_sig)} "
          f"({len(raw_sig)/len(results_df)*100:.1f}%)")
    print(f"  Bonferroni significant:   {len(bonf_sig)}")

    if len(bonf_sig) > 0:
        print(f"\n  BONFERRONI-SIGNIFICANT BIASES:")
        for _, r in bonf_sig.sort_values("p").iterrows():
            direction = "OVER-predicts home margin" if r["mean_resid"] < 0 else "UNDER-predicts home margin"
            print(f"    [{r['dimension']}] {r['segment']}: "
                  f"mean resid = {r['mean_resid']:+.3f} runs ({direction}), "
                  f"N={r['n']}, p={r['p']:.6f}")
    else:
        print(f"\n  NO Bonferroni-significant biases found.")
        print(f"  → Residuals are random across all tested segments.")
        print(f"  → Model mu extraction is complete within current feature set.")

    # Expected false positive rate
    expected_fp = len(results_df) * 0.05
    print(f"\n  Expected false positives at α=0.05: {expected_fp:.1f}")
    print(f"  Observed raw significant: {len(raw_sig)}")
    ratio = len(raw_sig) / max(expected_fp, 1)
    if ratio < 1.5:
        print(f"  Ratio {ratio:.2f} ≈ 1.0 → consistent with random noise, no real signal")
    elif ratio < 2.5:
        print(f"  Ratio {ratio:.2f} → mild excess, check largest effects")
    else:
        print(f"  Ratio {ratio:.2f} → excess suggests real biases present")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, train, test = load_oos_residuals()
    results_df = run_full_scan(test)
    side_decomposition(test)
    team_residual_deep_dive(test)
    interaction_scan(test)
    summarize(results_df)
