"""
kl_margin_analysis.py — Per-segment KL divergence on full margin distribution

Tests whether the fixed-dispersion assumption (sigma=4.217 or r=3.26) breaks
by segment.  High KL in a segment → conditional dispersion could help.

Segments:
  - close_total band: low (<7.5), mid (7.5–9.0), high (>9.0)
  - |close_spread| band: pick (≤1.5), moderate (1.5–3.0), strong (>3.0)
  - predicted total (mu_h + mu_a) band: low/mid/high terciles

Two distributional assumptions tested:
  1. Normal(mu, sigma=4.217) — the ATS distributional model
  2. NegBin(r=3.26) simulation — the NegBin model

OOS conditional-sigma test:
  - Temporal 80/20 split (same as model training)
  - Train: fit per-segment sigma from residuals
  - Test: compare Brier (fixed sigma vs conditional sigma) on spread cover probs
"""
from __future__ import annotations
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr  # KL components

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
DATA_DIR = os.path.join(_HERE, "..", "data")
SUBMODEL_DIR = os.path.join(DATA_DIR, "submodels")

# ── Constants ──
R_GLOBAL = 3.26
SIGMA_FIXED = 4.217
MARGIN_BINS = np.arange(-15, 16)  # integer margins -15 to +15
N_BINS = len(MARGIN_BINS)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load feature matrix with predictions and actual scores."""
    from features import build_feature_matrix

    print("=" * 100)
    print("  KL DIVERGENCE: FULL MARGIN DISTRIBUTION ANALYSIS")
    print("=" * 100)

    # Load feature matrix
    print("\n  Loading MLB feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"  Raw matrix: {len(df)} games")

    # Filter to scored games
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    print(f"  After score filter: {len(df)} games")

    # Actual margin
    df["actual_margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)

    # Load NegBin model and predict mu_home, mu_away
    negbin_path = os.path.join(SUBMODEL_DIR, "mlb_negbin_model.pkl")
    with open(negbin_path, "rb") as f:
        negbin_art = pickle.load(f)

    feat_cols = negbin_art["feature_cols"]
    available = [c for c in feat_cols if c in df.columns]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing {len(missing)} features: {missing[:5]}...")
    print(f"  Using {len(available)} / {len(feat_cols)} NegBin features")

    X = df[available].values.astype(float)
    mu_home = np.clip(negbin_art["gbm_home"].predict(X), 0.5, 15)
    mu_away = np.clip(negbin_art["gbm_away"].predict(X), 0.5, 15)

    df["pred_mu_home"] = mu_home
    df["pred_mu_away"] = mu_away
    df["pred_margin"] = mu_home - mu_away
    df["pred_total"] = mu_home + mu_away
    df["residual"] = df["actual_margin"] - df["pred_margin"]

    # Segmentation columns
    df["close_spread_num"] = pd.to_numeric(df.get("close_spread"), errors="coerce")
    df["close_total_num"] = pd.to_numeric(df.get("close_total"), errors="coerce")
    df["abs_spread"] = df["close_spread_num"].abs()

    n_with_spread = df["close_spread_num"].notna().sum()
    n_with_total = df["close_total_num"].notna().sum()
    print(f"  Games with close_spread: {n_with_spread}")
    print(f"  Games with close_total:  {n_with_total}")

    # Load ATS distributional sigma for comparison
    ats_path = os.path.join(SUBMODEL_DIR, "mlb_ats_distributional.pkl")
    sigma = SIGMA_FIXED
    if os.path.exists(ats_path):
        with open(ats_path, "rb") as f:
            ats_art = pickle.load(f)
        sigma = ats_art.get("sigma", SIGMA_FIXED)
    print(f"  ATS sigma: {sigma:.4f}")
    print(f"  NegBin r_global: {R_GLOBAL}")

    return df, sigma


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MARGIN DISTRIBUTION COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def normal_margin_pmf(mu, sigma, bins=MARGIN_BINS):
    """
    Discretised Normal PMF for margin = home - away.
    P(margin = k) = Phi((k+0.5-mu)/sigma) - Phi((k-0.5-mu)/sigma)
    """
    cdf_hi = stats.norm.cdf((bins + 0.5 - mu) / sigma)
    cdf_lo = stats.norm.cdf((bins - 0.5 - mu) / sigma)
    pmf = cdf_hi - cdf_lo
    # Accumulate tails
    pmf[0] += stats.norm.cdf((bins[0] - 0.5 - mu) / sigma)
    pmf[-1] += 1.0 - stats.norm.cdf((bins[-1] + 0.5 - mu) / sigma)
    return np.maximum(pmf, 1e-12)


def negbin_margin_pmf(mu_h, mu_a, r=R_GLOBAL, bins=MARGIN_BINS):
    """
    Exact PMF of margin = NegBin(r, p_h) - NegBin(r, p_a) via convolution.
    Computed over integer margins in bins.
    """
    p_h = r / (r + mu_h)
    p_a = r / (r + mu_a)

    # PMF of each team's runs over 0..30
    max_runs = 30
    runs_range = np.arange(0, max_runs + 1)
    pmf_h = stats.nbinom.pmf(runs_range, r, p_h)
    pmf_a = stats.nbinom.pmf(runs_range, r, p_a)

    # Normalise to sum to 1 (truncation correction)
    pmf_h = pmf_h / pmf_h.sum()
    pmf_a = pmf_a / pmf_a.sum()

    # Full convolution: margin = h - a, range -(max_runs)..(max_runs)
    full_margin_pmf = np.convolve(pmf_h, pmf_a[::-1], mode='full')
    # full_margin_pmf[i] = P(margin = i - max_runs)
    full_margin_range = np.arange(-max_runs, max_runs + 1)

    # Map to our bins
    result = np.zeros(len(bins))
    for i, b in enumerate(bins):
        idx = b + max_runs  # index into full_margin_range
        if 0 <= idx < len(full_margin_pmf):
            result[i] = full_margin_pmf[idx]

    # Accumulate tails
    lo_idx = bins[0] + max_runs
    hi_idx = bins[-1] + max_runs
    result[0] += full_margin_pmf[:max(lo_idx, 0)].sum()
    result[-1] += full_margin_pmf[min(hi_idx + 1, len(full_margin_pmf)):].sum()

    return np.maximum(result, 1e-12)


def mixture_pmf(pmf_func, games_df, sigma=None, bins=MARGIN_BINS):
    """
    Compute mixture PMF: average per-game PMFs across all games in segment.
    pmf_func: 'normal' or 'negbin'
    """
    n = len(games_df)
    mixture = np.zeros(len(bins))

    if pmf_func == 'normal':
        for _, row in games_df.iterrows():
            mu = row["pred_margin"]
            s = sigma if sigma is not None else SIGMA_FIXED
            mixture += normal_margin_pmf(mu, s, bins)
    elif pmf_func == 'negbin':
        for _, row in games_df.iterrows():
            mixture += negbin_margin_pmf(row["pred_mu_home"], row["pred_mu_away"],
                                          R_GLOBAL, bins)

    mixture /= n
    return np.maximum(mixture, 1e-12)


def empirical_pmf(margins, bins=MARGIN_BINS):
    """Empirical PMF of actual margins, binned at integers."""
    # Clip to bin range (tails go to edge bins)
    clipped = np.clip(margins, bins[0], bins[-1]).astype(int)
    counts = np.zeros(len(bins))
    for m in clipped:
        idx = m - bins[0]
        counts[idx] += 1
    counts /= counts.sum()
    return np.maximum(counts, 1e-12)


def kl_divergence(p, q):
    """KL(P || Q) — bits lost using Q to approximate P."""
    return float(np.sum(rel_entr(p, q)))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def assign_segments(df):
    """Add segment columns."""
    # -- Total band (from close_total) --
    df["seg_total"] = pd.cut(
        df["close_total_num"],
        bins=[0, 7.5, 9.0, 20],
        labels=["low (<7.5)", "mid (7.5–9)", "high (>9)"],
    )

    # -- Spread band (from |close_spread|) --
    df["seg_spread"] = pd.cut(
        df["abs_spread"],
        bins=[-0.1, 1.5, 3.0, 20],
        labels=["pick (≤1.5)", "moderate (1.5–3)", "strong (>3)"],
    )

    # -- Predicted total terciles --
    df["seg_pred_total"] = pd.qcut(
        df["pred_total"], q=3,
        labels=["pred_low", "pred_mid", "pred_high"],
    )

    # -- |Predicted margin| bands (proxy for game closeness) --
    df["seg_pred_margin"] = pd.cut(
        df["pred_margin"].abs(),
        bins=[-0.1, 0.5, 1.5, 20],
        labels=["close (|μ|≤0.5)", "lean (0.5<|μ|≤1.5)", "mismatch (|μ|>1.5)"],
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. KL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_kl_analysis(df, sigma):
    """Compute per-segment KL divergence for both distributional families."""
    df = assign_segments(df)

    segment_cols = ["seg_total", "seg_spread", "seg_pred_total", "seg_pred_margin"]
    segment_labels = {
        "seg_total": "Close Total Band",
        "seg_spread": "|Close Spread| Band",
        "seg_pred_total": "Predicted Total Tercile",
        "seg_pred_margin": "|Predicted Margin| Band",
    }

    results = []
    print(f"\n{'='*100}")
    print(f"  PER-SEGMENT KL DIVERGENCE")
    print(f"{'='*100}")

    for seg_col in segment_cols:
        seg_name = segment_labels[seg_col]
        print(f"\n  ── {seg_name} ──")
        print(f"  {'Segment':<25} {'N':>6} {'KL_Normal':>12} {'KL_NegBin':>12} "
              f"{'σ_resid':>10} {'σ_fixed':>10} {'σ_ratio':>10}")
        print(f"  {'─'*25} {'─'*6} {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

        for seg_val, seg_df in df.groupby(seg_col, observed=True):
            if len(seg_df) < 30:
                print(f"  {str(seg_val):<25} {len(seg_df):>6}   (too few games)")
                continue

            # Empirical distribution
            emp = empirical_pmf(seg_df["actual_margin"].values)

            # Predicted distributions
            pred_normal = mixture_pmf('normal', seg_df, sigma=sigma)
            pred_negbin = mixture_pmf('negbin', seg_df)

            kl_norm = kl_divergence(emp, pred_normal)
            kl_nb = kl_divergence(emp, pred_negbin)

            # Segment-specific residual sigma
            resid = seg_df["residual"].values
            sigma_seg = float(np.std(resid, ddof=1))
            sigma_ratio = sigma_seg / sigma

            results.append({
                "segment_type": seg_name,
                "segment": str(seg_val),
                "n_games": len(seg_df),
                "kl_normal": kl_norm,
                "kl_negbin": kl_nb,
                "sigma_residual": sigma_seg,
                "sigma_fixed": sigma,
                "sigma_ratio": sigma_ratio,
                "mean_pred_margin": float(seg_df["pred_margin"].mean()),
                "mean_actual_margin": float(seg_df["actual_margin"].mean()),
            })

            print(f"  {str(seg_val):<25} {len(seg_df):>6} {kl_norm:>12.5f} "
                  f"{kl_nb:>12.5f} {sigma_seg:>10.3f} {sigma:>10.3f} {sigma_ratio:>10.3f}")

    # -- Global baseline --
    print(f"\n  ── Global (all games) ──")
    emp_all = empirical_pmf(df["actual_margin"].values)
    pred_norm_all = mixture_pmf('normal', df, sigma=sigma)
    pred_nb_all = mixture_pmf('negbin', df)
    kl_norm_all = kl_divergence(emp_all, pred_norm_all)
    kl_nb_all = kl_divergence(emp_all, pred_nb_all)
    sigma_all = float(np.std(df["residual"].values, ddof=1))
    print(f"  {'ALL':<25} {len(df):>6} {kl_norm_all:>12.5f} "
          f"{kl_nb_all:>12.5f} {sigma_all:>10.3f} {sigma:>10.3f} {sigma_all/sigma:>10.3f}")

    results.append({
        "segment_type": "Global",
        "segment": "ALL",
        "n_games": len(df),
        "kl_normal": kl_norm_all,
        "kl_negbin": kl_nb_all,
        "sigma_residual": sigma_all,
        "sigma_fixed": sigma,
        "sigma_ratio": sigma_all / sigma,
    })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RESIDUAL SHAPE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def residual_shape_analysis(df, sigma):
    """Check normality of residuals per segment — heavy tails?"""
    df = assign_segments(df)

    print(f"\n{'='*100}")
    print(f"  RESIDUAL SHAPE ANALYSIS (Normality tests)")
    print(f"{'='*100}")
    print(f"  {'Segment':<30} {'N':>6} {'Skew':>8} {'Kurt':>8} {'JB_p':>10} {'σ':>8}")
    print(f"  {'─'*30} {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")

    for seg_col in ["seg_total", "seg_spread", "seg_pred_margin"]:
        for seg_val, seg_df in df.groupby(seg_col, observed=True):
            if len(seg_df) < 50:
                continue
            r = seg_df["residual"].values
            sk = float(stats.skew(r))
            ku = float(stats.kurtosis(r))  # excess kurtosis
            jb_stat, jb_p = stats.jarque_bera(r)
            sig = float(np.std(r, ddof=1))
            label = f"{seg_col}={seg_val}"
            print(f"  {label:<30} {len(seg_df):>6} {sk:>+8.3f} {ku:>+8.3f} "
                  f"{jb_p:>10.4f} {sig:>8.3f}")

    # Global
    r_all = df["residual"].values
    sk = float(stats.skew(r_all))
    ku = float(stats.kurtosis(r_all))
    jb_stat, jb_p = stats.jarque_bera(r_all)
    sig = float(np.std(r_all, ddof=1))
    print(f"  {'GLOBAL':<30} {len(df):>6} {sk:>+8.3f} {ku:>+8.3f} "
          f"{jb_p:>10.4f} {sig:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OOS CONDITIONAL SIGMA TEST
# ═══════════════════════════════════════════════════════════════════════════════

def conditional_sigma_oos_test(df, sigma):
    """
    Temporal 80/20 split.
    Train: fit per-segment sigma from residuals.
    Test: compare Brier scores (fixed sigma vs conditional sigma) on spread cover.
    """
    from scipy.stats import norm

    df = assign_segments(df)

    # Sort by date, temporal split
    df = df.sort_values("game_date").reset_index(drop=True)
    split = int(len(df) * 0.80)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    print(f"\n{'='*100}")
    print(f"  OOS CONDITIONAL SIGMA TEST")
    print(f"{'='*100}")
    print(f"  Train: {len(train)} games  ({str(train['game_date'].iloc[0])[:10]} → "
          f"{str(train['game_date'].iloc[-1])[:10]})")
    print(f"  Test:  {len(test)} games  ({str(test['game_date'].iloc[0])[:10]} → "
          f"{str(test['game_date'].iloc[-1])[:10]})")

    # -- Fit conditional sigma from train residuals --
    # Use seg_total and seg_spread (market-observable at prediction time)
    seg_sigmas = {}
    for seg_col in ["seg_total", "seg_spread", "seg_pred_margin"]:
        for seg_val, seg_df in train.groupby(seg_col, observed=True):
            if len(seg_df) >= 30:
                seg_sigmas[(seg_col, str(seg_val))] = float(np.std(seg_df["residual"].values, ddof=1))

    print(f"\n  Fitted conditional sigmas (from train):")
    for (col, val), s in sorted(seg_sigmas.items()):
        print(f"    {col}={val}: σ={s:.3f}  (vs fixed {sigma:.3f}, ratio={s/sigma:.3f})")

    # -- Test: Brier on spread cover for each rung --
    # For each test game: compute P(home covers +1.5) with fixed vs conditional sigma
    # Then check against actual outcome
    rungs = [1.5, 2.5, 3.5]

    print(f"\n  OOS Brier Scores (lower = better):")
    print(f"  {'Rung':<8} {'Seg_col':<25} {'Brier_fixed':>14} {'Brier_cond':>14} "
          f"{'Δ_Brier':>10} {'N':>6}")
    print(f"  {'─'*8} {'─'*25} {'─'*14} {'─'*14} {'─'*10} {'─'*6}")

    for seg_col in ["seg_total", "seg_spread", "seg_pred_margin"]:
        for rung in rungs:
            brier_fixed_sum = 0
            brier_cond_sum = 0
            n = 0

            for _, row in test.iterrows():
                margin = row["actual_margin"]
                mu = row["pred_margin"]

                # Actual cover: home covers +rung if margin > -rung
                actual_cover = 1.0 if margin > -rung else (
                    0.0 if margin < -rung else np.nan)
                if np.isnan(actual_cover):
                    continue  # push

                # Fixed sigma P(cover)
                p_fixed = float(1.0 - norm.cdf((-rung - mu) / sigma))

                # Conditional sigma
                seg_val = str(row[seg_col])
                cond_sigma = seg_sigmas.get((seg_col, seg_val), sigma)
                p_cond = float(1.0 - norm.cdf((-rung - mu) / cond_sigma))

                brier_fixed_sum += (p_fixed - actual_cover) ** 2
                brier_cond_sum += (p_cond - actual_cover) ** 2
                n += 1

            if n > 0:
                bf = brier_fixed_sum / n
                bc = brier_cond_sum / n
                delta = bc - bf
                print(f"  +{rung:<7.1f} {seg_col:<25} {bf:>14.6f} {bc:>14.6f} "
                      f"{delta:>+10.6f} {n:>6}")

    # -- Also test via NegBin KL on test set --
    print(f"\n  OOS KL (test set only):")
    emp_test = empirical_pmf(test["actual_margin"].values)
    pred_norm_test = mixture_pmf('normal', test, sigma=sigma)
    pred_nb_test = mixture_pmf('negbin', test)
    kl_norm = kl_divergence(emp_test, pred_norm_test)
    kl_nb = kl_divergence(emp_test, pred_nb_test)
    print(f"    KL(emp || Normal_fixed):  {kl_norm:.5f}")
    print(f"    KL(emp || NegBin_fixed):  {kl_nb:.5f}")

    # Conditional-sigma normal on test
    # For each test game, use segment-specific sigma
    for seg_col in ["seg_total", "seg_spread"]:
        pred_cond = np.zeros(N_BINS)
        for _, row in test.iterrows():
            seg_val = str(row[seg_col])
            cond_sigma = seg_sigmas.get((seg_col, seg_val), sigma)
            pred_cond += normal_margin_pmf(row["pred_margin"], cond_sigma)
        pred_cond /= len(test)
        pred_cond = np.maximum(pred_cond, 1e-12)
        kl_cond = kl_divergence(emp_test, pred_cond)
        print(f"    KL(emp || Normal_cond[{seg_col}]): {kl_cond:.5f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DATA VOLUME ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def data_volume_report(df):
    """Report games per segment — is there enough for reliable KL?"""
    df = assign_segments(df)

    print(f"\n{'='*100}")
    print(f"  DATA VOLUME PER SEGMENT")
    print(f"{'='*100}")

    for seg_col, label in [
        ("seg_total", "Close Total Band"),
        ("seg_spread", "|Close Spread| Band"),
        ("seg_pred_total", "Predicted Total Tercile"),
        ("seg_pred_margin", "|Predicted Margin| Band"),
    ]:
        print(f"\n  {label}:")
        vc = df[seg_col].value_counts(dropna=False).sort_index()
        for val, cnt in vc.items():
            flag = "  ⚠ LOW" if (cnt < 100 and pd.notna(val)) else ""
            print(f"    {str(val):<25} {cnt:>6} games{flag}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, sigma = load_data()
    data_volume_report(df)
    kl_results = run_kl_analysis(df, sigma)
    residual_shape_analysis(df, sigma)
    conditional_sigma_oos_test(df, sigma)

    # Summary
    print(f"\n{'='*100}")
    print(f"  SUMMARY")
    print(f"{'='*100}")

    # Highest KL segments
    seg_results = kl_results[kl_results["segment"] != "ALL"].copy()
    seg_results = seg_results.sort_values("kl_normal", ascending=False)

    print(f"\n  Top 5 highest-KL segments (Normal model):")
    for _, r in seg_results.head(5).iterrows():
        print(f"    {r['segment_type']}: {r['segment']:<25} "
              f"KL={r['kl_normal']:.5f}  σ_resid={r['sigma_residual']:.3f}  "
              f"σ_ratio={r['sigma_ratio']:.3f}  N={r['n_games']}")

    # Sigma variation
    sigma_range = seg_results["sigma_residual"]
    print(f"\n  Residual σ range across segments: {sigma_range.min():.3f} – {sigma_range.max():.3f}")
    print(f"  Fixed σ: {sigma:.3f}")
    print(f"  Max deviation from fixed: {abs(sigma_range.max() - sigma):.3f} "
          f"({abs(sigma_range.max()/sigma - 1)*100:.1f}%)")

    global_kl = kl_results[kl_results["segment"] == "ALL"].iloc[0]
    print(f"\n  Global KL — Normal: {global_kl['kl_normal']:.5f}, "
          f"NegBin: {global_kl['kl_negbin']:.5f}")
    print(f"  (Lower KL = better fit to empirical margin distribution)")
