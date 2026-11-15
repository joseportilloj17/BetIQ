#!/usr/bin/env python3
"""
PIT check for Gamma-Poisson sim draws on STL/BLK/FG3M.

Uses the ablation predictions (predicted_mean, predicted_std for per-minute rates)
and computes PIT using Gamma-Poisson (NegBin) CDF instead of Normal CDF.
Compares KS statistics: Normal vs Gamma-Poisson.
"""
import gzip, json, os, sys
import numpy as np
from scipy import stats as sp_stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PREDS_PATH = os.path.join(DATA_DIR, "nba_ablation_preds_incr.json.gz")

# Stats: PTS=0, REB=1, AST=2, STL=3, BLK=4, FG3M=5, PRA=6
STAT_NAMES = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "PRA"]
DISCRETE_INDICES = {3, 4, 5}  # STL, BLK, FG3M


def negbin_cdf(x, mu, var):
    """CDF of NegBin(r, p) where r = mu^2/(var-mu), p = r/(r+mu)."""
    if mu <= 0 or var <= mu:
        # Degenerate: Poisson
        return float(sp_stats.poisson.cdf(int(x), mu=max(mu, 1e-6)))
    r = mu ** 2 / (var - mu)
    r = max(r, 0.1)
    p = r / (r + mu)
    return float(sp_stats.nbinom.cdf(int(x), n=r, p=p))


def normal_cdf(x, mu, sigma):
    """CDF of Normal(mu, sigma)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return float(sp_stats.norm.cdf(x, loc=mu, scale=sigma))


def main():
    if not os.path.exists(PREDS_PATH):
        print(f"ERROR: {PREDS_PATH} not found")
        return

    print(f"Loading predictions from {PREDS_PATH}...")
    with gzip.open(PREDS_PATH, "rt") as f:
        data = json.load(f)

    # data structure: list of dicts or nested
    if isinstance(data, dict):
        preds = data.get("predictions", [])
    else:
        preds = data

    print(f"Total predictions: {len(preds)}")

    # Check structure of first pred
    if preds:
        print(f"Keys: {list(preds[0].keys())[:15]}")

    # Filter to factor_set = 'full' or the best available
    factor_sets = set(p.get("factor_set", "full") for p in preds)
    print(f"Factor sets: {factor_sets}")

    # Use the default/full factor set
    target_fs = "def_blow_recon_50pct" if "def_blow_recon_50pct" in factor_sets else "full"
    if target_fs not in factor_sets:
        target_fs = list(factor_sets)[0]
    print(f"Using factor_set: {target_fs}")

    preds = [p for p in preds if p.get("factor_set") == target_fs]
    print(f"Predictions with factor_set={target_fs}: {len(preds)}")

    # Map stat names to indices for discrete check
    DISCRETE_STATS = {"STL", "BLK", "FG3M"}

    # Compute PIT for each stat
    stat_names_in_data = sorted(set(p.get("stat", "") for p in preds))
    print(f"Stats in data: {stat_names_in_data}")

    for stat_name in stat_names_in_data:
        if stat_name == "PRA":
            continue  # composite, skip

        is_discrete = stat_name in DISCRETE_STATS
        pit_normal = []
        pit_negbin = []

        for p in preds:
            if p.get("stat") != stat_name:
                continue
            actual = p.get("actual")
            mu = p.get("predicted_mean")
            sigma = p.get("predicted_std")

            if actual is None or mu is None or sigma is None:
                continue
            if mu <= 0 or sigma <= 0:
                continue

            # Normal PIT
            pit_n = normal_cdf(actual, mu, sigma)
            pit_normal.append(pit_n)

            # For discrete stats: Gamma-Poisson = NegBin marginal
            # Normal var = sigma^2 = (minutes * rate_std)^2
            # NegBin var = mean + sigma^2 (Poisson variance + rate uncertainty)
            # This is the correct marginal variance for Gamma(rate) → Poisson(minutes*rate)
            normal_var = sigma ** 2
            if is_discrete:
                negbin_var = mu + normal_var  # E[Poisson var] + Var[Poisson mean]
                pit_gp = negbin_cdf(actual, mu, negbin_var)
                pit_negbin.append(pit_gp)
            else:
                pit_negbin.append(pit_n)  # same as Normal for continuous

        if not pit_normal:
            continue

        pit_normal = np.array(pit_normal)
        pit_negbin = np.array(pit_negbin)

        # KS test against Uniform(0,1)
        ks_n, p_n = sp_stats.kstest(pit_normal, "uniform")
        ks_gp, p_gp = sp_stats.kstest(pit_negbin, "uniform")

        dist_label = "Gamma-Poisson" if is_discrete else "Normal (same)"
        improved = ks_gp < ks_n

        print(f"\n{stat_name} (N={len(pit_normal)}, {'DISCRETE' if is_discrete else 'continuous'}):")
        print(f"  Normal CDF PIT:        KS={ks_n:.4f}, p={p_n:.4e}, mean={pit_normal.mean():.4f}")
        print(f"  {dist_label} PIT: KS={ks_gp:.4f}, p={p_gp:.4e}, mean={pit_negbin.mean():.4f}")
        if is_discrete:
            print(f"  Δ KS: {ks_gp - ks_n:+.4f} {'✓ IMPROVED' if improved else '✗ WORSE'}")
            if ks_gp < 0.10:
                print(f"  → KS < 0.10: ACCEPTABLE calibration")
            elif ks_gp < 0.15:
                print(f"  → KS 0.10-0.15: MARGINAL calibration")
            else:
                print(f"  → KS ≥ 0.15: POOR calibration")

    print("\n✓ PIT check complete")


if __name__ == "__main__":
    main()
