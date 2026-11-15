"""
nfl_calibration_diagnosis.py — Phase 2a: Diagnose why probability tails
are overconfident.

Analyses:
1. Reliability curve: binned P(over) vs actual hit rate
2. PIT histograms: should be uniform; U-shaped = distributions too tight
3. CRPS: proper scoring rule for distributional forecasts
4. Variance decomposition: model-assumed CV vs actual game-to-game CV
5. Per stat×position breakdown

Uses stored forward_preds (2026 W2-W5 graded) + nflverse PBP for
historical variance benchmarks.

Run:
    python nfl_calibration_diagnosis.py
"""
from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from math import log, sqrt, exp, floor

import numpy as np

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")
_NFL_DB = os.path.join(_DATA_DIR, "nfl.db")

# Same norm CVs used in find_prop_gaps
_TYPICAL_CV = {
    "pass_yds": 0.25, "rush_yds": 0.45, "rec_yds": 0.45,
    "pass_tds": 0.80, "pass_ints": 1.00, "receptions": 0.40,
}


# ══════════════════════════════════════════════════════════════════════════════
# Distribution helpers (mirror fantasy_projections.py)
# ══════════════════════════════════════════════════════════════════════════════

def _lognormal_cdf(x: float, mean: float, var: float) -> float:
    """CDF of lognormal at x, parameterized by (mean, variance)."""
    from scipy.stats import lognorm
    if mean <= 0 or var <= 0 or x <= 0:
        return 0.0
    sigma_sq = log(1 + var / (mean ** 2))
    sigma = sqrt(sigma_sq)
    mu = log(mean) - sigma_sq / 2
    return float(lognorm.cdf(x, s=sigma, scale=exp(mu)))


def _negbin_cdf(x: float, mean: float, var: float) -> float:
    """CDF of negative binomial at x, parameterized by (mean, variance)."""
    from scipy.stats import nbinom, poisson
    if mean <= 0:
        return 0.0
    if var <= mean:
        return float(poisson.cdf(floor(x), mu=mean))
    r = (mean ** 2) / (var - mean)
    r = max(r, 0.1)
    if r >= 500:
        return float(poisson.cdf(floor(x), mu=mean))
    p = r / (r + mean)
    return float(nbinom.cdf(floor(x), n=r, p=p))


def _lognormal_crps(actual: float, mean: float, var: float) -> float:
    """CRPS for lognormal distribution via numerical integration."""
    from scipy.stats import lognorm
    from scipy.integrate import quad
    if mean <= 0 or var <= 0:
        return abs(actual - mean)
    sigma_sq = log(1 + var / (mean ** 2))
    sigma = sqrt(sigma_sq)
    mu = log(mean) - sigma_sq / 2
    scale = exp(mu)

    # CRPS = E|X - y| = integral_0^inf |F(x) - 1(x >= y)| dx (squared form)
    # Use closed-form: CRPS = y*(2F(y)-1) - 2*mean*(2*Phi(z-sigma)-1)
    # where z = (log(y) - mu) / sigma, Phi = standard normal CDF
    # But simpler to use numerical integration bounded
    def integrand(x):
        cdf_val = lognorm.cdf(x, s=sigma, scale=scale)
        indicator = 1.0 if x >= actual else 0.0
        return (cdf_val - indicator) ** 2

    upper = max(actual * 3, mean * 3, 10)
    result, _ = quad(integrand, 0, upper, limit=100)
    return float(result)


def _negbin_crps(actual: float, mean: float, var: float) -> float:
    """CRPS for negative binomial via discrete sum."""
    from scipy.stats import nbinom, poisson
    if mean <= 0:
        return abs(actual)
    actual_int = int(round(actual))
    if var <= mean:
        # Poisson
        from scipy.stats import poisson
        upper = max(actual_int * 3, int(mean * 3), 20)
        crps = 0.0
        for k in range(upper + 1):
            cdf_val = poisson.cdf(k, mu=mean)
            indicator = 1.0 if k >= actual_int else 0.0
            crps += (cdf_val - indicator) ** 2
        return crps

    r = max((mean ** 2) / (var - mean), 0.1)
    p = r / (r + mean)
    upper = max(actual_int * 3, int(mean * 3), 20)
    crps = 0.0
    for k in range(upper + 1):
        cdf_val = nbinom.cdf(k, n=r, p=p)
        indicator = 1.0 if k >= actual_int else 0.0
        crps += (cdf_val - indicator) ** 2
    return crps


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_graded_predictions() -> list[dict]:
    """Load all graded distributional predictions."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT player_name, position, stat_type, stat_category,
               consensus_mean, consensus_sd, var_deviation,
               prop_line, p_over, devigged_implied, edge_vs_market,
               direction, actual_value, is_win, season, week,
               conviction_bucket, line_depth,
               prop_over_price, prop_under_price, game_id
        FROM fantasy_forward_preds
        WHERE actual_value IS NOT NULL
          AND method LIKE '%distributional%'
          AND method NOT LIKE 'v0_capped%'
          AND stat_category IN ('yardage', 'count')
          AND data_error IS NULL
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _reconstruct_variance(mean: float, stat_type: str,
                          var_deviation: float | None) -> float:
    """Reconstruct the variance the model used for this prediction."""
    norm_cv = _TYPICAL_CV.get(stat_type, 0.40)
    if var_deviation is not None and var_deviation != 0:
        player_cv = norm_cv * (1 + var_deviation)
    else:
        player_cv = norm_cv
    player_cv = max(player_cv, 0.05)  # floor
    return (mean * player_cv) ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1: Reliability curve
# ══════════════════════════════════════════════════════════════════════════════

def reliability_curve(preds: list[dict], n_bins: int = 10) -> list[dict]:
    """Bin predictions by P(over), compute actual over-hit rate."""
    bins = defaultdict(lambda: {"sum_p": 0.0, "n": 0, "over_hits": 0})

    for p in preds:
        if p["stat_category"] != "yardage":
            continue
        p_over = p["p_over"]
        actual = p["actual_value"]
        line = p["prop_line"]
        if p_over is None or actual is None or line is None:
            continue

        # Bin index
        b = min(int(p_over * n_bins), n_bins - 1)
        bins[b]["sum_p"] += p_over
        bins[b]["n"] += 1
        if actual > line:
            bins[b]["over_hits"] += 1
        elif actual == line:
            bins[b]["over_hits"] += 0.5

    results = []
    for i in range(n_bins):
        d = bins[i]
        if d["n"] == 0:
            continue
        mean_p = d["sum_p"] / d["n"]
        actual_rate = d["over_hits"] / d["n"]
        results.append({
            "bin": f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}",
            "n": d["n"],
            "mean_predicted": round(mean_p, 4),
            "actual_rate": round(actual_rate, 4),
            "gap": round(actual_rate - mean_p, 4),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2: PIT histogram
# ══════════════════════════════════════════════════════════════════════════════

def pit_values(preds: list[dict]) -> dict[str, list[float]]:
    """Compute PIT = F(actual) for each prediction, grouped by stat_type."""
    result = defaultdict(list)

    for p in preds:
        mean = p["consensus_mean"]
        actual = p["actual_value"]
        stat_type = p["stat_type"]
        cat = p["stat_category"]
        if mean is None or actual is None or mean <= 0:
            continue

        var = _reconstruct_variance(mean, stat_type, p.get("var_deviation"))

        if cat == "yardage":
            pit = _lognormal_cdf(actual, mean, var)
        else:
            pit = _negbin_cdf(actual, mean, var)

        result[stat_type].append(pit)

    return dict(result)


def pit_histogram(pit_vals: list[float], n_bins: int = 10) -> list[dict]:
    """Bin PIT values into uniform bins. Uniform PIT = well-calibrated."""
    counts = [0] * n_bins
    for v in pit_vals:
        b = min(int(v * n_bins), n_bins - 1)
        counts[b] += 1

    total = len(pit_vals)
    expected = total / n_bins

    results = []
    for i in range(n_bins):
        results.append({
            "bin": f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}",
            "count": counts[i],
            "expected": round(expected, 1),
            "ratio": round(counts[i] / expected, 3) if expected > 0 else 0,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3: CRPS
# ══════════════════════════════════════════════════════════════════════════════

def compute_crps(preds: list[dict]) -> dict[str, dict]:
    """Compute CRPS per stat_type. Lower = better calibrated."""
    crps_by_stat = defaultdict(list)

    for p in preds:
        mean = p["consensus_mean"]
        actual = p["actual_value"]
        stat_type = p["stat_type"]
        cat = p["stat_category"]
        if mean is None or actual is None or mean <= 0:
            continue

        var = _reconstruct_variance(mean, stat_type, p.get("var_deviation"))

        if cat == "yardage":
            crps = _lognormal_crps(actual, mean, var)
        else:
            crps = _negbin_crps(actual, mean, var)

        crps_by_stat[stat_type].append(crps)

    results = {}
    for stat, vals in crps_by_stat.items():
        arr = np.array(vals)
        results[stat] = {
            "mean_crps": round(float(np.mean(arr)), 3),
            "median_crps": round(float(np.median(arr)), 3),
            "n": len(vals),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4: Variance decomposition — model vs actual
# ══════════════════════════════════════════════════════════════════════════════

def _load_nflverse_player_stats(seasons: list[int] = None) -> dict:
    """
    Load per-player per-game stats from nflverse PBP.
    Returns {player_name: {stat: [game_values]}} for variance computation.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    try:
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data(seasons)
    except Exception as exc:
        print(f"  Could not load PBP: {exc}")
        return {}

    # Receiving stats per player per game
    recv = pbp[pbp["receiver_player_name"].notna() & (pbp["pass_attempt"] == 1)]
    recv_stats = (recv.groupby(["game_id", "receiver_player_name", "posteam"])
                  .agg(
                      rec_yds=("yards_gained", "sum"),
                      receptions=("complete_pass", "sum"),
                      targets=("pass_attempt", "sum"),
                  ).reset_index())

    # Rushing stats per player per game
    rush = pbp[pbp["rusher_player_name"].notna() & (pbp["rush_attempt"] == 1)]
    rush_stats = (rush.groupby(["game_id", "rusher_player_name", "posteam"])
                  .agg(rush_yds=("yards_gained", "sum"))
                  .reset_index()
                  .rename(columns={"rusher_player_name": "player_name"}))

    # Passing stats per player per game
    passing = pbp[pbp["passer_player_name"].notna() & (pbp["pass_attempt"] == 1)]
    pass_stats = (passing.groupby(["game_id", "passer_player_name", "posteam"])
                  .agg(
                      pass_yds=("yards_gained", "sum"),
                      pass_tds=("pass_touchdown", "sum"),
                      pass_ints=("interception", "sum"),
                      completions=("complete_pass", "sum"),
                      attempts=("pass_attempt", "sum"),
                  ).reset_index()
                  .rename(columns={"passer_player_name": "player_name"}))

    # Build lookup
    result = defaultdict(lambda: defaultdict(list))

    for _, row in recv_stats.iterrows():
        name = row["receiver_player_name"]
        result[name]["rec_yds"].append(float(row["rec_yds"]))
        result[name]["receptions"].append(float(row["receptions"]))

    for _, row in rush_stats.iterrows():
        name = row["player_name"]
        result[name]["rush_yds"].append(float(row["rush_yds"]))

    for _, row in pass_stats.iterrows():
        name = row["player_name"]
        result[name]["pass_yds"].append(float(row["pass_yds"]))
        result[name]["pass_tds"].append(float(row["pass_tds"]))

    return dict(result)


def variance_decomposition(preds: list[dict],
                           nflverse_stats: dict | None = None) -> dict:
    """
    Compare model-assumed variance vs actual game-to-game variance.

    For each stat_type:
    - model_cv: what the model assumes (from var_deviation + norm CV)
    - actual_cv: empirical game-to-game CV from nflverse data
    - ratio: model_cv / actual_cv (< 1 means model is too tight)
    """
    # Group predictions by (stat_type, position) and compute model CVs
    model_cvs = defaultdict(list)
    for p in preds:
        stat = p["stat_type"]
        mean = p["consensus_mean"]
        if mean <= 0:
            continue
        var = _reconstruct_variance(mean, stat, p.get("var_deviation"))
        cv = sqrt(var) / mean
        model_cvs[stat].append(cv)

    # Compute actual CVs from nflverse data
    actual_cvs = {}
    if nflverse_stats:
        for stat in model_cvs:
            all_player_cvs = []
            for player, stats in nflverse_stats.items():
                vals = stats.get(stat, [])
                if len(vals) >= 4:
                    m = np.mean(vals)
                    if m > 5:  # only meaningful for players with real volume
                        s = np.std(vals, ddof=1)
                        all_player_cvs.append(s / m)
            if all_player_cvs:
                actual_cvs[stat] = {
                    "mean_cv": round(float(np.mean(all_player_cvs)), 3),
                    "median_cv": round(float(np.median(all_player_cvs)), 3),
                    "p25": round(float(np.percentile(all_player_cvs, 25)), 3),
                    "p75": round(float(np.percentile(all_player_cvs, 75)), 3),
                    "n_players": len(all_player_cvs),
                }

    # Also compute "prediction error CV" — how far actuals deviate from
    # the model mean, relative to the model mean
    pred_error_cv = defaultdict(list)
    for p in preds:
        stat = p["stat_type"]
        mean = p["consensus_mean"]
        actual = p["actual_value"]
        if mean > 5 and actual is not None:
            pred_error_cv[stat].append(abs(actual - mean) / mean)

    results = {}
    for stat in model_cvs:
        model_arr = np.array(model_cvs[stat])
        results[stat] = {
            "model_cv_mean": round(float(np.mean(model_arr)), 3),
            "model_cv_median": round(float(np.median(model_arr)), 3),
            "norm_cv": _TYPICAL_CV.get(stat, 0.40),
        }
        if stat in actual_cvs:
            results[stat]["actual_cv"] = actual_cvs[stat]
            results[stat]["ratio"] = round(
                results[stat]["model_cv_mean"] /
                actual_cvs[stat]["mean_cv"], 3)

        if stat in pred_error_cv:
            err_arr = np.array(pred_error_cv[stat])
            results[stat]["pred_error_mad"] = round(float(np.mean(err_arr)), 3)
            # What CV SHOULD the model use so that 68% of actuals fall within
            # 1 sigma? (assuming lognormal)
            results[stat]["implied_cv_from_errors"] = round(
                float(np.percentile(err_arr, 68)), 3)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5: Per-position reliability
# ══════════════════════════════════════════════════════════════════════════════

def reliability_by_stat_position(preds: list[dict]) -> dict:
    """Reliability curves broken down by stat_type and position."""
    groups = defaultdict(list)
    for p in preds:
        if p["stat_category"] != "yardage":
            continue
        key = (p["stat_type"], p["position"])
        groups[key].append(p)

    results = {}
    for (stat, pos), group_preds in groups.items():
        if len(group_preds) < 30:
            continue
        curve = reliability_curve(group_preds, n_bins=5)
        results[f"{stat}_{pos}"] = {
            "n": len(group_preds),
            "curve": curve,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 6: Overconfidence by prediction distance from 0.5
# ══════════════════════════════════════════════════════════════════════════════

def overconfidence_by_extremity(preds: list[dict]) -> list[dict]:
    """
    How much does calibration degrade as P(over) moves away from 0.5?

    Groups predictions by |P(over) - 0.5| and checks actual hit rate.
    """
    bins = [
        (0.00, 0.10, "near_50pct"),
        (0.10, 0.20, "moderate"),
        (0.20, 0.30, "strong"),
        (0.30, 0.40, "very_strong"),
        (0.40, 0.50, "extreme"),
    ]

    results = []
    for lo, hi, label in bins:
        subset = [p for p in preds
                  if p["stat_category"] == "yardage"
                  and p["p_over"] is not None
                  and lo <= abs(p["p_over"] - 0.5) < hi]
        if len(subset) < 10:
            continue

        # Compute mean predicted and actual for this extremity band
        mean_p = np.mean([p["p_over"] for p in subset])
        actual_rates = []
        for p in subset:
            if p["actual_value"] > p["prop_line"]:
                actual_rates.append(1.0)
            elif p["actual_value"] == p["prop_line"]:
                actual_rates.append(0.5)
            else:
                actual_rates.append(0.0)
        actual_rate = np.mean(actual_rates)

        # For symmetric analysis: expected = mean_p for overs, (1-mean_p) for unders
        # Better: compute absolute calibration error
        # For each pred, error = |predicted - actual|
        errors = []
        for p in subset:
            pred_p = p["p_over"]
            hit = 1.0 if p["actual_value"] > p["prop_line"] else (
                0.5 if p["actual_value"] == p["prop_line"] else 0.0)
            errors.append(abs(pred_p - hit))

        results.append({
            "extremity": label,
            "abs_distance": f"{lo:.1f}-{hi:.1f}",
            "n": len(subset),
            "mean_abs_calib_error": round(float(np.mean(errors)), 4),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main report
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration_report(load_nflverse: bool = True):
    """Full Phase 2a calibration diagnostic report."""
    print("=" * 80)
    print("PHASE 2a: CALIBRATION DIAGNOSIS")
    print("=" * 80)

    # Load data
    preds = _load_graded_predictions()
    yardage_preds = [p for p in preds if p["stat_category"] == "yardage"]
    print(f"\nLoaded {len(preds)} graded predictions "
          f"({len(yardage_preds)} yardage)")

    # ── 1. Reliability curve ──
    print("\n" + "─" * 60)
    print("1. RELIABILITY CURVE (yardage stats only)")
    print("─" * 60)
    curve = reliability_curve(preds, n_bins=10)
    print(f"  {'Bin':12s} {'N':>5s} {'Predicted':>10s} {'Actual':>8s} {'Gap':>8s}")
    for row in curve:
        print(f"  {row['bin']:12s} {row['n']:5d} {row['mean_predicted']:10.4f} "
              f"{row['actual_rate']:8.4f} {row['gap']:+8.4f}")

    # Overall Brier score
    brier_terms = []
    for p in yardage_preds:
        if p["p_over"] is not None and p["actual_value"] is not None:
            hit = 1.0 if p["actual_value"] > p["prop_line"] else (
                0.5 if p["actual_value"] == p["prop_line"] else 0.0)
            brier_terms.append((p["p_over"] - hit) ** 2)
    if brier_terms:
        brier = np.mean(brier_terms)
        print(f"\n  Brier score: {brier:.4f}  (lower is better; 0.25 = no skill)")

    # ── 2. PIT histogram ──
    print("\n" + "─" * 60)
    print("2. PIT HISTOGRAMS (should be uniform; U-shaped = too tight)")
    print("─" * 60)
    pit_vals = pit_values(preds)
    for stat_type in ["rec_yds", "rush_yds", "pass_yds"]:
        vals = pit_vals.get(stat_type, [])
        if len(vals) < 30:
            continue
        hist = pit_histogram(vals, n_bins=10)
        print(f"\n  {stat_type} (N={len(vals)}):")
        print(f"  {'Bin':12s} {'Count':>6s} {'Expected':>9s} {'Ratio':>7s}  Bar")
        for row in hist:
            bar = "█" * int(row["ratio"] * 10)
            print(f"  {row['bin']:12s} {row['count']:6d} {row['expected']:9.1f} "
                  f"{row['ratio']:7.3f}  {bar}")

        # U-shape metric: ratio of extreme bins to middle bins
        if len(hist) >= 4:
            extreme = (hist[0]["ratio"] + hist[-1]["ratio"]) / 2
            middle = np.mean([h["ratio"] for h in hist[3:7]])
            print(f"  U-shape metric: extreme/middle = {extreme/middle:.2f} "
                  f"(1.00 = uniform, >1.3 = too tight)")

    # ── 3. CRPS ──
    print("\n" + "─" * 60)
    print("3. CRPS (Continuous Ranked Probability Score)")
    print("─" * 60)
    crps_results = compute_crps(preds)
    print(f"  {'Stat':12s} {'Mean CRPS':>10s} {'Median':>8s} {'N':>5s}")
    for stat, d in sorted(crps_results.items()):
        print(f"  {stat:12s} {d['mean_crps']:10.3f} {d['median_crps']:8.3f} {d['n']:5d}")

    # ── 4. Variance decomposition ──
    print("\n" + "─" * 60)
    print("4. VARIANCE DECOMPOSITION")
    print("─" * 60)

    nfl_stats = None
    if load_nflverse:
        print("  Loading nflverse PBP for 2023-2025 (this may take a minute)...")
        nfl_stats = _load_nflverse_player_stats([2023, 2024, 2025])
        print(f"  Loaded stats for {len(nfl_stats)} players")

    var_results = variance_decomposition(preds, nfl_stats)
    print(f"\n  {'Stat':12s} {'Model CV':>9s} {'Norm CV':>8s} "
          f"{'Actual CV':>10s} {'Ratio':>7s} {'PredErr MAD':>12s} {'Implied CV':>11s}")
    for stat in ["pass_yds", "rush_yds", "rec_yds", "receptions", "pass_tds"]:
        d = var_results.get(stat, {})
        if not d:
            continue
        actual_cv = d.get("actual_cv", {}).get("mean_cv", "N/A")
        ratio = d.get("ratio", "N/A")
        pred_err = d.get("pred_error_mad", "N/A")
        implied = d.get("implied_cv_from_errors", "N/A")
        actual_s = f"{actual_cv:10.3f}" if isinstance(actual_cv, float) else f"{actual_cv:>10s}"
        ratio_s = f"{ratio:7.3f}" if isinstance(ratio, float) else f"{ratio:>7s}"
        pred_s = f"{pred_err:12.3f}" if isinstance(pred_err, float) else f"{pred_err:>12s}"
        impl_s = f"{implied:11.3f}" if isinstance(implied, float) else f"{implied:>11s}"
        print(f"  {stat:12s} {d['model_cv_mean']:9.3f} {d['norm_cv']:8.3f} "
              f"{actual_s} {ratio_s} {pred_s} {impl_s}")

    # ── 5. Per stat×position ──
    print("\n" + "─" * 60)
    print("5. RELIABILITY BY STAT × POSITION (5 bins, yardage only)")
    print("─" * 60)
    sp_results = reliability_by_stat_position(preds)
    for key, data in sorted(sp_results.items()):
        print(f"\n  {key} (N={data['n']}):")
        print(f"  {'Bin':12s} {'N':>5s} {'Pred':>7s} {'Actual':>7s} {'Gap':>7s}")
        for row in data["curve"]:
            print(f"  {row['bin']:12s} {row['n']:5d} {row['mean_predicted']:7.3f} "
                  f"{row['actual_rate']:7.3f} {row['gap']:+7.3f}")

    # ── 6. Overconfidence by extremity ──
    print("\n" + "─" * 60)
    print("6. OVERCONFIDENCE BY PREDICTION EXTREMITY")
    print("─" * 60)
    ext_results = overconfidence_by_extremity(preds)
    print(f"  {'Distance':15s} {'N':>5s} {'Mean Abs Calib Error':>22s}")
    for row in ext_results:
        print(f"  {row['abs_distance']:15s} {row['n']:5d} "
              f"{row['mean_abs_calib_error']:22.4f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    # Quantify the overconfidence
    tail_low = [r for r in curve if r["mean_predicted"] < 0.15]
    tail_high = [r for r in curve if r["mean_predicted"] > 0.75]
    if tail_low:
        avg_low_gap = np.mean([r["gap"] for r in tail_low])
        print(f"\n  Under tail (P<0.15): model says {np.mean([r['mean_predicted'] for r in tail_low]):.1%}, "
              f"actual {np.mean([r['actual_rate'] for r in tail_low]):.1%} "
              f"({avg_low_gap:+.1%}pp)")
    if tail_high:
        avg_high_gap = np.mean([r["gap"] for r in tail_high])
        print(f"  Over tail (P>0.75): model says {np.mean([r['mean_predicted'] for r in tail_high]):.1%}, "
              f"actual {np.mean([r['actual_rate'] for r in tail_high]):.1%} "
              f"({avg_high_gap:+.1%}pp)")

    # Root cause
    for stat in ["rec_yds", "rush_yds", "pass_yds"]:
        d = var_results.get(stat, {})
        if d and "actual_cv" in d and "ratio" in d:
            ratio = d["ratio"]
            if ratio < 0.8:
                print(f"\n  ROOT CAUSE ({stat}): Model CV ({d['model_cv_mean']:.3f}) is "
                      f"{1/ratio:.1f}x BELOW actual CV ({d['actual_cv']['mean_cv']:.3f})")
                print(f"    → Distributions are {1/ratio:.1f}x too tight → extreme probabilities")
                print(f"    → Fix: increase norm_cv from {d['norm_cv']:.2f} "
                      f"to ~{d['actual_cv']['mean_cv']:.2f}")

    return {
        "reliability": curve,
        "pit": pit_vals,
        "crps": crps_results,
        "variance": var_results,
        "brier": float(brier) if brier_terms else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# V0 vs V1 comparison
# ══════════════════════════════════════════════════════════════════════════════

def _rescore_v1(preds: list[dict], use_v11: bool = False,
                use_v12: bool = False) -> list[dict]:
    """Re-score graded predictions through v1, v1.1, or v1.2.

    use_v11: PIT-calibrated flat CVs, gamma parameterized by mean.
    use_v12: Same CVs as v1.1, but shifted gamma so
             F(projection) = 1 - P_empirical(actual > projection).
    """
    from nfl_interim_dispersion import (
        load_cached_params, get_interim_cv, p_over_line_gamma,
        p_over_line_negbin, DISPERSION_PARAMS, _YARDAGE_STATS,
        gamma_cdf, gamma_crps_single,
    )
    if not DISPERSION_PARAMS:
        load_cached_params()

    # v1.2: load base rates
    if use_v12:
        from nfl_interim_dispersion import (
            load_v12_base_rates, get_v12_base_rate,
            gamma_cdf_shifted, gamma_crps_shifted,
            p_over_line_gamma_shifted,
        )
        load_v12_base_rates()

    v1_preds = []
    for p in preds:
        mean = p["consensus_mean"]
        stat = p["stat_type"]
        pos = p["position"]
        line = p["prop_line"]
        cat = p["stat_category"]

        if mean is None or mean <= 0 or line is None:
            continue

        cv = get_interim_cv(stat, pos, mean,
                            use_v11=use_v11, use_v12=use_v12)

        if cat == "yardage" or stat in _YARDAGE_STATS:
            if use_v12:
                p_over_at_proj = get_v12_base_rate(stat, pos)
                p_ov = p_over_line_gamma_shifted(line, mean, cv,
                                                 p_over_at_proj)
                pit_val = (gamma_cdf_shifted(p["actual_value"], mean,
                                            cv, p_over_at_proj)
                           if p["actual_value"] is not None else None)
                crps_val = (gamma_crps_shifted(p["actual_value"], mean,
                                              cv, p_over_at_proj)
                            if p["actual_value"] is not None else None)
            else:
                p_ov = p_over_line_gamma(line, mean, cv)
                pit_val = (gamma_cdf(p["actual_value"], mean, cv)
                           if p["actual_value"] is not None else None)
                crps_val = (gamma_crps_single(p["actual_value"], mean, cv)
                            if p["actual_value"] is not None else None)
        else:
            var = (mean * cv) ** 2
            p_ov = p_over_line_negbin(line, mean, var)
            from nfl_interim_dispersion import negbin_cdf, negbin_crps_single
            pit_val = (negbin_cdf(p["actual_value"], mean, var)
                       if p["actual_value"] is not None else None)
            crps_val = (negbin_crps_single(p["actual_value"], mean, var)
                        if p["actual_value"] is not None else None)

        v1_pred = dict(p)
        v1_pred["p_over"] = round(p_ov, 4)
        v1_pred["v1_cv"] = round(cv, 4)
        v1_pred["v1_pit"] = pit_val
        v1_pred["v1_crps"] = crps_val
        # Also compute devigged edge under v1
        devigged = p.get("devigged_implied")
        if devigged is not None:
            v1_pred["edge_vs_market"] = round(p_ov - devigged, 4)
        v1_preds.append(v1_pred)

    return v1_preds


def run_v0_v1_comparison(use_v11: bool = False, use_v12: bool = False,
                         three_way: bool = False):
    """Full comparison of v0 vs v1/v1.1/v1.2 on 2026 W2-W5.

    three_way=True: show v0 vs v1.1 vs v1.2 side-by-side.
    """
    if three_way:
        _run_three_way_comparison()
        return

    v_label = "V1.2" if use_v12 else ("V1.1" if use_v11 else "V1")
    if use_v12:
        v_desc = "gamma + PIT-CV + shifted location"
    elif use_v11:
        v_desc = "gamma + PIT-calibrated flat CV"
    else:
        v_desc = "gamma + CRPS-fitted power-law CV"
    print("=" * 80)
    print(f"V0 vs {v_label} CALIBRATION COMPARISON")
    print("Evaluated on 2026 W2-W5 graded predictions")
    print(f"v0 = lognormal + flat CV | {v_label.lower()} = {v_desc}")
    print("=" * 80)

    v0_preds = _load_graded_predictions()
    v1_preds = _rescore_v1(v0_preds, use_v11=use_v11, use_v12=use_v12)
    v0_yard = [p for p in v0_preds if p["stat_category"] == "yardage"]
    v1_yard = [p for p in v1_preds if p["stat_category"] == "yardage"]

    print(f"\nN = {len(v0_yard)} yardage predictions")

    # ── 1. Reliability curves side-by-side ──
    print("\n" + "─" * 70)
    print("1. RELIABILITY CURVES (10 bins, yardage)")
    print("─" * 70)
    v0_curve = reliability_curve(v0_preds, n_bins=10)
    v1_curve = reliability_curve(v1_preds, n_bins=10)

    # Align bins
    v0_map = {r["bin"]: r for r in v0_curve}
    v1_map = {r["bin"]: r for r in v1_curve}
    all_bins = sorted(set(v0_map) | set(v1_map))

    print(f"  {'Bin':12s} {'N':>5s} {'v0 Pred':>8s} {'v1 Pred':>8s} {'Actual':>7s} {'v0 Gap':>7s} {'v1 Gap':>7s}")
    for b in all_bins:
        v0r = v0_map.get(b, {})
        v1r = v1_map.get(b, {})
        n = v0r.get("n", v1r.get("n", 0))
        v0p = v0r.get("mean_predicted", 0)
        v1p = v1r.get("mean_predicted", 0)
        actual = v0r.get("actual_rate", v1r.get("actual_rate", 0))
        v0g = v0r.get("gap", 0)
        v1g = v1r.get("gap", 0)
        print(f"  {b:12s} {n:5d} {v0p:8.4f} {v1p:8.4f} {actual:7.4f} "
              f"{v0g:+7.4f} {v1g:+7.4f}")

    # Brier scores
    def brier(preds_list):
        terms = []
        for p in preds_list:
            if p.get("stat_category") != "yardage":
                continue
            if p["p_over"] is not None and p["actual_value"] is not None:
                hit = 1.0 if p["actual_value"] > p["prop_line"] else (
                    0.5 if p["actual_value"] == p["prop_line"] else 0.0)
                terms.append((p["p_over"] - hit) ** 2)
        return np.mean(terms) if terms else None

    b0 = brier(v0_preds)
    b1 = brier(v1_preds)
    print(f"\n  Brier: v0={b0:.4f}  v1={b1:.4f}  Δ={b1-b0:+.4f}")

    # ── 2. PIT histograms ──
    print("\n" + "─" * 70)
    print("2. PIT HISTOGRAMS")
    print("─" * 70)
    v0_pit = pit_values(v0_preds)

    # v1 PIT uses gamma CDF
    v1_pit_vals = defaultdict(list)
    for p in v1_preds:
        if p.get("v1_pit") is not None:
            v1_pit_vals[p["stat_type"]].append(p["v1_pit"])

    for stat in ["rec_yds", "rush_yds", "pass_yds"]:
        v0_vals = v0_pit.get(stat, [])
        v1_vals = v1_pit_vals.get(stat, [])
        if len(v0_vals) < 30:
            continue

        v0_hist = pit_histogram(v0_vals, n_bins=10)
        v1_hist = pit_histogram(v1_vals, n_bins=10)

        print(f"\n  {stat} (N={len(v0_vals)}):")
        print(f"  {'Bin':12s} {'v0 ratio':>9s} {'v1 ratio':>9s}")
        for h0, h1 in zip(v0_hist, v1_hist):
            bar0 = "▓" * min(int(h0["ratio"] * 5), 20)
            bar1 = "░" * min(int(h1["ratio"] * 5), 20)
            print(f"  {h0['bin']:12s} {h0['ratio']:9.3f} {h1['ratio']:9.3f}  "
                  f"v0:{bar0} v1:{bar1}")

        # U-shape metrics
        if len(v0_hist) >= 4:
            v0_extreme = (v0_hist[0]["ratio"] + v0_hist[-1]["ratio"]) / 2
            v0_middle = np.mean([h["ratio"] for h in v0_hist[3:7]])
            v1_extreme = (v1_hist[0]["ratio"] + v1_hist[-1]["ratio"]) / 2
            v1_middle = np.mean([h["ratio"] for h in v1_hist[3:7]])
            print(f"  U-shape: v0={v0_extreme/v0_middle:.2f}  "
                  f"v1={v1_extreme/v1_middle:.2f}")

    # ── 3. CRPS ──
    print("\n" + "─" * 70)
    print("3. CRPS COMPARISON")
    print("─" * 70)
    v0_crps = compute_crps(v0_preds)
    # v1 CRPS from pre-computed values
    v1_crps_by_stat = defaultdict(list)
    for p in v1_preds:
        if p.get("v1_crps") is not None:
            v1_crps_by_stat[p["stat_type"]].append(p["v1_crps"])

    print(f"  {'Stat':12s} {'v0 CRPS':>9s} {'v1 CRPS':>9s} {'Δ':>8s} {'Δ%':>7s} {'N':>5s}")
    for stat in sorted(v0_crps):
        v0c = v0_crps[stat]["mean_crps"]
        v1_vals = v1_crps_by_stat.get(stat, [])
        if v1_vals:
            v1c = round(float(np.mean(v1_vals)), 3)
            delta = v1c - v0c
            pct = (delta / v0c * 100) if v0c > 0 else 0
            print(f"  {stat:12s} {v0c:9.3f} {v1c:9.3f} {delta:+8.3f} {pct:+6.1f}% "
                  f"{v0_crps[stat]['n']:5d}")
        else:
            print(f"  {stat:12s} {v0c:9.3f}       N/A      N/A     N/A "
                  f"{v0_crps[stat]['n']:5d}")

    # ── 4. Reliability by stat×position ──
    print("\n" + "─" * 70)
    print("4. RELIABILITY BY STAT × POSITION (v1)")
    print("─" * 70)
    v1_sp = reliability_by_stat_position(v1_preds)
    for key, data in sorted(v1_sp.items()):
        print(f"\n  {key} (N={data['n']}):")
        print(f"  {'Bin':12s} {'N':>5s} {'Pred':>7s} {'Actual':>7s} {'Gap':>7s}")
        for row in data["curve"]:
            print(f"  {row['bin']:12s} {row['n']:5d} {row['mean_predicted']:7.3f} "
                  f"{row['actual_rate']:7.3f} {row['gap']:+7.3f}")

    # ── 5. Edge and volume distribution changes ──
    print("\n" + "─" * 70)
    print("5. EDGE & VOLUME DISTRIBUTION CHANGES")
    print("─" * 70)

    # Edge distribution
    v0_edges = [p["edge_vs_market"] for p in v0_preds
                if p.get("edge_vs_market") is not None and p["stat_category"] == "yardage"]
    v1_edges = [p["edge_vs_market"] for p in v1_preds
                if p.get("edge_vs_market") is not None and p["stat_category"] == "yardage"]

    print(f"\n  Edge distribution (|edge_vs_market|):")
    print(f"  {'':15s} {'v0':>8s} {'v1':>8s}")
    if v0_edges and v1_edges:
        v0_abs = np.abs(v0_edges)
        v1_abs = np.abs(v1_edges)
        print(f"  {'Mean':15s} {np.mean(v0_abs):8.4f} {np.mean(v1_abs):8.4f}")
        print(f"  {'Median':15s} {np.median(v0_abs):8.4f} {np.median(v1_abs):8.4f}")
        print(f"  {'P90':15s} {np.percentile(v0_abs, 90):8.4f} {np.percentile(v1_abs, 90):8.4f}")
        print(f"  {'Max':15s} {np.max(v0_abs):8.4f} {np.max(v1_abs):8.4f}")

    # Volume at various thresholds
    print(f"\n  Pick volume at edge thresholds:")
    print(f"  {'Threshold':15s} {'v0':>5s} {'v1':>5s} {'Δ':>5s}")
    for thresh in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]:
        v0_n = sum(1 for e in v0_edges if abs(e) >= thresh)
        v1_n = sum(1 for e in v1_edges if abs(e) >= thresh)
        print(f"  ≥{thresh:.0%}{'':<10s} {v0_n:5d} {v1_n:5d} {v1_n-v0_n:+5d}")

    # Direction flip count — with breakdown
    flips_over_to_under = 0
    flips_under_to_over = 0
    for v0p, v1p in zip(v0_preds, v1_preds):
        if v0p["stat_category"] != "yardage":
            continue
        v0_dir = "over" if v0p["p_over"] > 0.5 else "under"
        v1_dir = "over" if v1p["p_over"] > 0.5 else "under"
        if v0_dir != v1_dir:
            if v0_dir == "over":
                flips_over_to_under += 1
            else:
                flips_under_to_over += 1
    total_flips = flips_over_to_under + flips_under_to_over
    print(f"\n  Direction flips: {total_flips} / {len(v0_yard)}")
    print(f"    over→under: {flips_over_to_under}")
    print(f"    under→over: {flips_under_to_over}")
    if total_flips > 0:
        balance = min(flips_over_to_under, flips_under_to_over) / total_flips
        print(f"    balance ratio: {balance:.2f} (0.0=all one way, 0.5=perfect)")

    # ── 6. Flip validation ──
    if total_flips > 0:
        print("\n" + "─" * 70)
        print("6. FLIP VALIDATION — realized over-hit rate on direction flips")
        print("─" * 70)
        _validate_flips(v0_preds, v1_preds)

    # ── 7. Receptions reliability (separate per user request) ──
    rec_preds_v0 = [p for p in v0_preds if p["stat_type"] == "receptions"]
    rec_preds_v1 = [p for p in v1_preds if p["stat_type"] == "receptions"]
    if len(rec_preds_v0) >= 10:
        print("\n" + "─" * 70)
        print("7. RECEPTIONS RELIABILITY (N too small for conclusions)")
        print("─" * 70)
        print(f"  N = {len(rec_preds_v0)}")
        v0_rec_curve = reliability_curve(rec_preds_v0, n_bins=4)
        v1_rec_curve = reliability_curve(rec_preds_v1, n_bins=4)
        print(f"  {'Bin':12s} {'N':>5s} {'v0 Pred':>8s} {'v1 Pred':>8s} {'Actual':>7s}")
        for h0, h1 in zip(v0_rec_curve, v1_rec_curve):
            print(f"  {h0['bin']:12s} {h0['n']:5d} {h0['mean_predicted']:8.3f} "
                  f"{h1['mean_predicted']:8.3f} {h0['actual_rate']:7.3f}")

    # ── 8. P(over at projection) vs empirical ──
    if use_v12:
        print("\n" + "─" * 70)
        print("8. P(OVER AT PROJECTION) — model vs empirical (target ±3pp)")
        print("─" * 70)
        _check_p_over_at_projection(v1_preds)


def _validate_flips(v0_preds: list[dict], v1_preds: list[dict]):
    """For direction flips (v0→v1), report realized over-hit rate.

    If over→under flips have realized over-hit ≥ 50%, the flips were wrong
    (v0 was right to say over).
    """
    by_type = defaultdict(lambda: {"n": 0, "over_hits": 0})

    for v0p, v1p in zip(v0_preds, v1_preds):
        if v0p["stat_category"] != "yardage":
            continue
        actual = v0p["actual_value"]
        line = v0p["prop_line"]
        if actual is None or line is None:
            continue

        v0_dir = "over" if v0p["p_over"] > 0.5 else "under"
        v1_dir = "over" if v1p["p_over"] > 0.5 else "under"
        if v0_dir == v1_dir:
            continue

        flip_type = f"{v0_dir}→{v1_dir}"
        hit = 1.0 if actual > line else (0.5 if actual == line else 0.0)

        # Track by flip type and by stat
        by_type[flip_type]["n"] += 1
        by_type[flip_type]["over_hits"] += hit
        stat_key = f"{flip_type}|{v0p['stat_type']}"
        by_type[stat_key]["n"] += 1
        by_type[stat_key]["over_hits"] += hit

    print(f"\n  {'Flip type':20s} {'N':>5s} {'Over-hit%':>10s} {'SE':>7s} {'Verdict':>12s}")
    print("  " + "-" * 60)

    for key in sorted(by_type.keys()):
        d = by_type[key]
        if d["n"] < 5:
            continue
        rate = d["over_hits"] / d["n"]
        se = sqrt(rate * (1 - rate) / d["n"]) if d["n"] > 1 else 0
        if "over→under" in key:
            verdict = "WRONG" if rate >= 0.50 else "correct"
        else:
            verdict = "WRONG" if rate < 0.50 else "correct"
        print(f"  {key:20s} {d['n']:5d} {rate:10.3f} {se:7.3f} {verdict:>12s}")


def _check_p_over_at_projection(v1_preds: list[dict]):
    """Check model P(over at projection) vs empirical per stat×pos.

    Target: within ±3pp.
    """
    from nfl_interim_dispersion import get_v12_base_rate, get_interim_cv
    from nfl_interim_dispersion import gamma_params_shifted
    from scipy import stats as sp_stats

    groups = defaultdict(lambda: {"n": 0, "over_hits": 0})
    for p in v1_preds:
        stat = p["stat_type"]
        pos = p["position"]
        mean = p["consensus_mean"]
        actual = p["actual_value"]
        if mean is None or mean <= 0 or actual is None:
            continue
        if p.get("stat_category") != "yardage":
            continue

        key = (stat, pos)
        groups[key]["n"] += 1
        if actual > mean:
            groups[key]["over_hits"] += 1

    print(f"\n  {'Stat':12s} {'Pos':5s} {'N':>5s} {'Empirical':>10s} "
          f"{'Model':>7s} {'Gap_pp':>7s} {'Pass':>5s}")
    print("  " + "-" * 60)

    for (stat, pos) in sorted(groups.keys()):
        d = groups[(stat, pos)]
        if d["n"] < 10:
            continue
        empirical = d["over_hits"] / d["n"]

        # Model P(over at projection) = base rate (which is what we fit to)
        model_rate = get_v12_base_rate(stat, pos)
        gap_pp = (model_rate - empirical) * 100
        passed = "YES" if abs(gap_pp) <= 3.0 else "NO"
        print(f"  {stat:12s} {pos:5s} {d['n']:5d} {empirical:10.3f} "
              f"{model_rate:7.3f} {gap_pp:+7.1f} {passed:>5s}")


def derive_edge_thresholds(v12_preds: list[dict] = None):
    """Derive edge thresholds from v1.2 with game-clustered CIs.

    For each edge cutoff, compute:
    - N picks, N games (independent units)
    - Win rate vs devigged market
    - Game-clustered bootstrap CI on win rate
    """
    if v12_preds is None:
        v0_preds = _load_graded_predictions()
        v12_preds = _rescore_v1(v0_preds, use_v12=True)

    # Build per-game clusters for bootstrap
    # Use (season, week, player_name) as a unique key — but cluster by game
    by_game = defaultdict(list)
    for p in v12_preds:
        if p.get("stat_category") != "yardage":
            continue
        actual = p["actual_value"]
        line = p["prop_line"]
        edge = p.get("edge_vs_market")
        devigged = p.get("devigged_implied")
        if actual is None or line is None or edge is None or devigged is None:
            continue

        hit = 1.0 if actual > line else (0.5 if actual == line else 0.0)
        p_ov = p["p_over"]
        direction = "over" if p_ov > 0.5 else "under"
        # Win = correct direction
        if direction == "over":
            win = hit
        else:
            win = 1.0 - hit

        game_key = p.get("game_id") or (p["season"], p["week"])
        by_game[game_key].append({
            "edge": abs(edge),
            "win": win,
            "p_over": p_ov,
            "devigged": devigged,
            "stat": p["stat_type"],
            "pos": p["position"],
        })

    all_preds = []
    for preds_list in by_game.values():
        all_preds.extend(preds_list)

    game_keys = list(by_game.keys())
    n_games = len(game_keys)

    print(f"\n{'='*80}")
    print("EDGE THRESHOLD ANALYSIS (v1.2)")
    print(f"N = {len(all_preds)} yardage picks across {n_games} game-weeks")
    print(f"Note: ~4 weeks of data; CIs are wide")
    print(f"{'='*80}")

    thresholds = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    n_boot = 2000
    rng = np.random.default_rng(42)

    print(f"\n  {'Edge≥':>7s} {'N_picks':>8s} {'N_games':>8s} {'WR%':>7s} "
          f"{'vs_mkt':>7s} {'95% CI':>14s} {'Signif':>7s}")
    print("  " + "-" * 65)

    for thresh in thresholds:
        # Filter predictions at this threshold
        filtered = [p for p in all_preds if p["edge"] >= thresh]
        if len(filtered) < 10:
            continue

        # Filter by game for clustering
        game_filtered = {}
        for gk in game_keys:
            gf = [p for p in by_game[gk] if p["edge"] >= thresh]
            if gf:
                game_filtered[gk] = gf

        n_picks = len(filtered)
        n_gf = len(game_filtered)
        win_rate = np.mean([p["win"] for p in filtered])

        # Market baseline: devigged probability
        mkt_wr = np.mean([p["devigged"] for p in filtered])
        # For "under" picks, devigged is P(over), so market WR = 1 - devigged
        # Actually edge_vs_market = p_over - devigged, and direction is based on p_over
        # For over picks: win ~ p_over, market ~ devigged
        # For under picks: win ~ 1-p_over, market ~ 1-devigged
        # So vs_market = win_rate - market_win_rate
        mkt_wins = []
        for p in filtered:
            if p["p_over"] > 0.5:
                mkt_wins.append(p["devigged"])
            else:
                mkt_wins.append(1.0 - p["devigged"])
        mkt_wr = np.mean(mkt_wins)
        vs_mkt = win_rate - mkt_wr

        # Game-clustered bootstrap
        boot_wrs = []
        gk_list = list(game_filtered.keys())
        gk_indices = np.arange(len(gk_list))
        for _ in range(n_boot):
            sampled_idx = rng.choice(gk_indices, size=len(gk_list), replace=True)
            boot_preds = []
            for idx in sampled_idx:
                boot_preds.extend(game_filtered[gk_list[idx]])
            if boot_preds:
                boot_wrs.append(np.mean([p["win"] for p in boot_preds]))

        if boot_wrs:
            ci_lo = np.percentile(boot_wrs, 2.5)
            ci_hi = np.percentile(boot_wrs, 97.5)
            ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]"
            signif = "YES" if ci_lo > mkt_wr else "no"
        else:
            ci_str = "N/A"
            signif = "N/A"

        print(f"  ≥{thresh:5.0%}   {n_picks:8d} {n_gf:8d} {win_rate:7.3f} "
              f"{vs_mkt:+7.3f} {ci_str:>14s} {signif:>7s}")

    # Also show by stat type at ≥5% threshold
    print(f"\n  By stat type (edge ≥ 5%):")
    print(f"  {'Stat':12s} {'N':>5s} {'WR%':>7s} {'Mkt_WR':>7s} {'vs_mkt':>7s}")
    print("  " + "-" * 45)
    by_stat = defaultdict(list)
    for p in all_preds:
        if p["edge"] >= 0.05:
            by_stat[p["stat"]].append(p)

    for stat in sorted(by_stat.keys()):
        items = by_stat[stat]
        wr = np.mean([p["win"] for p in items])
        mkt = np.mean([p["devigged"] if p["p_over"] > 0.5
                       else 1 - p["devigged"] for p in items])
        print(f"  {stat:12s} {len(items):5d} {wr:7.3f} {mkt:7.3f} {wr-mkt:+7.3f}")


def derive_roi_thresholds():
    """ROI at actual offered prices by edge bucket and stat.

    Uses v1.2 re-scored edges. Only includes picks where the actual
    betting-side price is available (over→prop_over_price,
    under→prop_under_price). Reports ROI including vig.
    """
    v0_preds = _load_graded_predictions()
    v12_preds = _rescore_v1(v0_preds, use_v12=True)

    # Build records with actual prices
    records = []
    skipped_no_price = 0
    for p in v12_preds:
        if p.get("stat_category") != "yardage":
            continue
        actual = p.get("actual_value")
        line = p.get("prop_line")
        edge = p.get("edge_vs_market")
        if actual is None or line is None or edge is None:
            continue

        p_ov = p["p_over"]
        direction = "over" if p_ov > 0.5 else "under"

        # Get the price on the side we'd bet
        if direction == "over":
            price = p.get("prop_over_price")
        else:
            price = p.get("prop_under_price")

        if price is None or price <= 1.0:
            skipped_no_price += 1
            continue

        # Determine win
        if actual > line:
            hit = 1.0
        elif actual == line:
            hit = 0.5  # push
        else:
            hit = 0.0

        if direction == "over":
            win = hit
        else:
            win = 1.0 - hit

        # ROI at actual price: win pays (price-1), loss pays -1, push pays 0
        if win == 1.0:
            roi = price - 1.0
        elif win == 0.0:
            roi = -1.0
        else:
            roi = 0.0  # push

        # Breakeven for this price
        be = 1.0 / price

        records.append({
            "edge": abs(edge),
            "signed_edge": edge,
            "win": win,
            "roi": roi,
            "price": price,
            "be": be,
            "p_over": p_ov,
            "direction": direction,
            "stat": p["stat_type"],
            "pos": p["position"],
            "season": p["season"],
            "week": p["week"],
            "game_id": p.get("game_id"),
        })

    print(f"\n{'='*80}")
    print("ROI THRESHOLD ANALYSIS (v1.2, actual prices incl. vig)")
    print(f"N = {len(records)} yardage picks with prices "
          f"(skipped {skipped_no_price} without price)")
    print(f"Note: ~4 weeks of data; treat as directional only")
    print(f"{'='*80}")

    # ── 1. By edge bucket ──
    thresholds = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    print(f"\n  {'Edge≥':>7s} {'N':>6s} {'WR%':>7s} {'BE%':>7s} "
          f"{'ROI%':>8s} {'AvgOdds':>8s} {'PnL/100':>8s}")
    print("  " + "-" * 60)

    for thresh in thresholds:
        filt = [r for r in records if r["edge"] >= thresh]
        if len(filt) < 5:
            continue
        wr = np.mean([r["win"] for r in filt])
        avg_be = np.mean([r["be"] for r in filt])
        avg_roi = np.mean([r["roi"] for r in filt])
        avg_price = np.mean([r["price"] for r in filt])
        total_pnl = sum(r["roi"] for r in filt)
        n = len(filt)
        print(f"  ≥{thresh:5.0%}   {n:6d} {wr:7.1%} {avg_be:7.1%} "
              f"{avg_roi:+8.1%} {avg_price:8.3f} {total_pnl:+8.1f}")

    # ── 2. Also show 15%+ bucket explicitly ──
    big = [r for r in records if r["edge"] >= 0.15]
    if big:
        print(f"\n  15%+ edge detail (N={len(big)}):")
        print(f"    WR={np.mean([r['win'] for r in big]):.1%}  "
              f"ROI={np.mean([r['roi'] for r in big]):+.1%}  "
              f"avg_price={np.mean([r['price'] for r in big]):.3f}")
        # Per-stat
        by_stat_big = defaultdict(list)
        for r in big:
            by_stat_big[r["stat"]].append(r)
        for stat in sorted(by_stat_big):
            items = by_stat_big[stat]
            print(f"    {stat:12s} N={len(items):4d} "
                  f"WR={np.mean([r['win'] for r in items]):.1%} "
                  f"ROI={np.mean([r['roi'] for r in items]):+.1%}")
    else:
        print(f"\n  15%+ edge: N=0")

    # ── 3. By stat type (all edges) ──
    print(f"\n  By stat type:")
    print(f"  {'Stat':12s} {'N':>6s} {'WR%':>7s} {'BE%':>7s} "
          f"{'ROI%':>8s} {'AvgEdge':>8s}")
    print("  " + "-" * 55)
    by_stat = defaultdict(list)
    for r in records:
        by_stat[r["stat"]].append(r)
    for stat in sorted(by_stat):
        items = by_stat[stat]
        wr = np.mean([r["win"] for r in items])
        avg_be = np.mean([r["be"] for r in items])
        avg_roi = np.mean([r["roi"] for r in items])
        avg_edge = np.mean([r["edge"] for r in items])
        print(f"  {stat:12s} {len(items):6d} {wr:7.1%} {avg_be:7.1%} "
              f"{avg_roi:+8.1%} {avg_edge:8.1%}")

    # ── 4. By stat at ≥5% edge ──
    print(f"\n  By stat type (edge ≥ 5%):")
    print(f"  {'Stat':12s} {'N':>6s} {'WR%':>7s} {'BE%':>7s} "
          f"{'ROI%':>8s} {'AvgEdge':>8s}")
    print("  " + "-" * 55)
    for stat in sorted(by_stat):
        items = [r for r in by_stat[stat] if r["edge"] >= 0.05]
        if len(items) < 5:
            continue
        wr = np.mean([r["win"] for r in items])
        avg_be = np.mean([r["be"] for r in items])
        avg_roi = np.mean([r["roi"] for r in items])
        avg_edge = np.mean([r["edge"] for r in items])
        print(f"  {stat:12s} {len(items):6d} {wr:7.1%} {avg_be:7.1%} "
              f"{avg_roi:+8.1%} {avg_edge:8.1%}")

    # ── 5. Max-edge cap evaluation ──
    caps = [0.20, 0.15, 0.12, 0.10]
    print(f"\n  Max-edge cap evaluation:")
    print(f"  {'Cap':>7s} {'Removed':>8s} {'ROI_kept':>9s} {'ROI_removed':>12s}")
    print("  " + "-" * 42)
    for cap in caps:
        kept = [r for r in records if r["edge"] < cap]
        removed = [r for r in records if r["edge"] >= cap]
        if kept and removed:
            roi_k = np.mean([r["roi"] for r in kept])
            roi_r = np.mean([r["roi"] for r in removed])
            print(f"  ≤{cap:5.0%}   {len(removed):8d} {roi_k:+9.1%} {roi_r:+12.1%}")
        elif kept:
            roi_k = np.mean([r["roi"] for r in kept])
            print(f"  ≤{cap:5.0%}   {0:8d} {roi_k:+9.1%} {'N/A':>12s}")

    # ── 6. Game-clustered bootstrap on ROI at ≥5% edge ──
    by_game = defaultdict(list)
    for r in records:
        if r["edge"] >= 0.05:
            gk = r.get("game_id") or (r["season"], r["week"])
            by_game[gk].append(r)

    if by_game:
        n_boot = 2000
        rng = np.random.default_rng(42)
        gk_list = list(by_game.keys())
        gk_idx = np.arange(len(gk_list))
        boot_rois = []
        for _ in range(n_boot):
            sampled = rng.choice(gk_idx, size=len(gk_list), replace=True)
            boot_picks = []
            for idx in sampled:
                boot_picks.extend(by_game[gk_list[idx]])
            if boot_picks:
                boot_rois.append(np.mean([r["roi"] for r in boot_picks]))
        ci_lo = np.percentile(boot_rois, 2.5)
        ci_hi = np.percentile(boot_rois, 97.5)
        mean_roi = np.mean([r["roi"] for r in records if r["edge"] >= 0.05])
        n_picks = sum(1 for r in records if r["edge"] >= 0.05)
        n_games = len(by_game)
        print(f"\n  Game-clustered ROI CI (edge ≥ 5%):")
        print(f"    N={n_picks} picks, {n_games} game-weeks")
        print(f"    ROI = {mean_roi:+.1%}  "
              f"95% CI [{ci_lo:+.1%}, {ci_hi:+.1%}]")
        signif = "YES" if ci_lo > 0 else "no"
        print(f"    Significant: {signif}")


def flag_edge_crosstab():
    """Flag × edge-bucket ROI cross-tabulation with game-clustered CIs.

    IN-SAMPLE: uses the same graded data that informed filter design.
    Answers: does the location-uncertainty flag differentiate ROI *within*
    each edge bucket, or is it redundant with edge alone?
    """
    from nfl_interim_dispersion import (
        load_cached_params, load_v12_base_rates,
        check_location_uncertain, get_interim_cv,
    )
    load_cached_params()
    load_v12_base_rates()

    v0_preds = _load_graded_predictions()
    v12_preds = _rescore_v1(v0_preds, use_v12=True)

    _MIN_DECIMAL = 1.6667  # -150
    _YARDAGE = {"pass_yds", "rush_yds", "rec_yds"}

    records = []
    for p in v12_preds:
        if p.get("stat_category") != "yardage":
            continue
        actual = p.get("actual_value")
        line = p.get("prop_line")
        edge = p.get("edge_vs_market")
        if actual is None or line is None or edge is None:
            continue

        p_ov = p["p_over"]
        direction = "over" if p_ov > 0.5 else "under"
        price = (p.get("prop_over_price") if direction == "over"
                 else p.get("prop_under_price"))
        if price is None or price <= 1.0:
            continue
        if price < _MIN_DECIMAL:
            continue  # only kept pool

        if actual > line:
            hit = 1.0
        elif actual == line:
            hit = 0.5
        else:
            hit = 0.0
        win = hit if direction == "over" else 1.0 - hit
        roi = (price - 1.0) if win == 1.0 else (-1.0 if win == 0.0 else 0.0)

        stat = p["stat_type"]
        pos = p["position"]
        mean = p["consensus_mean"]
        devigged = p.get("devigged_implied")
        uncertain = False
        if stat in _YARDAGE and mean and mean > 0 and devigged is not None:
            cv = get_interim_cv(stat, pos, mean, use_v12=True)
            if cv:
                uncertain = check_location_uncertain(
                    line, mean, cv, stat, pos, devigged)

        records.append({
            "edge": abs(edge), "roi": roi, "win": win, "price": price,
            "uncertain": uncertain, "stat": stat, "pos": pos,
            "game_id": p.get("game_id"), "season": p["season"],
            "week": p["week"], "p_over": p_ov,
        })

    print(f"\n{'='*80}")
    print("FLAG × EDGE BUCKET ROI CROSS-TAB (v1.2, kept pool only)")
    print(f"IN-SAMPLE — same data that informed filter design")
    print(f"N = {len(records)} yardage picks at odds > -150")
    print(f"{'='*80}")

    edge_buckets = [
        (0.03, 0.06, "3–6%"),
        (0.06, 0.10, "6–10%"),
        (0.10, 0.15, "10–15%"),
        (0.15, 1.00, "15%+"),
    ]

    rng = np.random.default_rng(42)
    n_boot = 2000

    print(f"\n  {'Edge':<8} {'Flag':<10} {'N':>5} {'WR%':>7} {'ROI%':>8} "
          f"{'95% CI':>16} {'Games':>6}")
    print("  " + "-" * 68)

    for lo, hi, label in edge_buckets:
        for flag_label, flag_val in [("Unflagged", False), ("Flagged", True)]:
            pool = [r for r in records
                    if lo <= r["edge"] < hi and r["uncertain"] == flag_val]
            if len(pool) < 5:
                print(f"  {label:<8} {flag_label:<10} {len(pool):>5} "
                      f"{'—':>7} {'—':>8} {'—':>16} {'—':>6}")
                continue

            wr = np.mean([r["win"] for r in pool])
            mean_roi = np.mean([r["roi"] for r in pool])

            by_game = defaultdict(list)
            for r in pool:
                gk = r.get("game_id") or (r["season"], r["week"])
                by_game[gk].append(r)
            gk_list = list(by_game.keys())

            boot = []
            for _ in range(n_boot):
                sampled = rng.choice(len(gk_list), size=len(gk_list),
                                     replace=True)
                bpicks = []
                for idx in sampled:
                    bpicks.extend(by_game[gk_list[idx]])
                if bpicks:
                    boot.append(np.mean([r["roi"] for r in bpicks]))

            ci_lo = np.percentile(boot, 2.5) if boot else 0
            ci_hi = np.percentile(boot, 97.5) if boot else 0
            print(f"  {label:<8} {flag_label:<10} {len(pool):>5} {wr:>6.1%} "
                  f"{mean_roi:>+7.1%} [{ci_lo:+.1%},{ci_hi:+.1%}] "
                  f"{len(gk_list):>5}")

        # Blank line between buckets
        print()

    # Summary: all-edge flagged vs unflagged
    flagged = [r for r in records if r["uncertain"]]
    unflagged = [r for r in records if not r["uncertain"]]
    print(f"  ALL EDGES (at ≥3%):")
    all_ge3 = [r for r in records if r["edge"] >= 0.03]
    for flag_label, pool in [("Unflagged",
                              [r for r in all_ge3 if not r["uncertain"]]),
                             ("Flagged",
                              [r for r in all_ge3 if r["uncertain"]])]:
        if len(pool) < 5:
            continue
        wr = np.mean([r["win"] for r in pool])
        mean_roi = np.mean([r["roi"] for r in pool])
        print(f"  {'ALL':<8} {flag_label:<10} {len(pool):>5} {wr:>6.1%} "
              f"{mean_roi:>+7.1%}")


def high_confidence_concentration():
    """Game concentration of P>0.80 picks at odds > -150.

    IN-SAMPLE. Reports distinct game count, top-5 games' share of profit,
    and whether one cluster drives the headline result.
    """
    from nfl_interim_dispersion import load_cached_params, load_v12_base_rates
    load_cached_params()
    load_v12_base_rates()

    v0_preds = _load_graded_predictions()
    v12_preds = _rescore_v1(v0_preds, use_v12=True)

    _MIN_DECIMAL = 1.6667  # -150

    records = []
    for p in v12_preds:
        if p.get("stat_category") != "yardage":
            continue
        actual = p.get("actual_value")
        line = p.get("prop_line")
        if actual is None or line is None:
            continue

        p_ov = p["p_over"]
        direction = "over" if p_ov > 0.5 else "under"
        price = (p.get("prop_over_price") if direction == "over"
                 else p.get("prop_under_price"))
        if price is None or price <= 1.0:
            continue
        if price < _MIN_DECIMAL:
            continue

        conf = max(p_ov, 1.0 - p_ov)
        if conf < 0.80:
            continue

        if actual > line:
            hit = 1.0
        elif actual == line:
            hit = 0.5
        else:
            hit = 0.0
        win = hit if direction == "over" else 1.0 - hit
        roi = (price - 1.0) if win == 1.0 else (-1.0 if win == 0.0 else 0.0)

        game_key = p.get("game_id") or f"S{p['season']}W{p['week']}"
        records.append({
            "roi": roi, "win": win, "price": price, "conf": conf,
            "stat": p["stat_type"], "pos": p["position"],
            "player": p.get("player_name", ""),
            "game_key": game_key, "season": p["season"], "week": p["week"],
            "direction": direction, "p_over": p_ov,
        })

    print(f"\n{'='*80}")
    print("HIGH-CONFIDENCE GAME CONCENTRATION (P>0.80, odds > -150)")
    print(f"IN-SAMPLE")
    print(f"{'='*80}")
    print(f"\n  Total picks: {len(records)}")

    if not records:
        print("  No picks match P>0.80 at odds > -150")
        return

    total_roi = np.mean([r["roi"] for r in records])
    total_wr = np.mean([r["win"] for r in records])
    total_pnl = sum(r["roi"] for r in records)
    print(f"  WR: {total_wr:.1%}  ROI: {total_roi:+.1%}  "
          f"Total PnL: {total_pnl:+.1f} units")

    # Group by game
    by_game = defaultdict(list)
    for r in records:
        by_game[r["game_key"]].append(r)

    n_games = len(by_game)
    print(f"  Distinct games: {n_games}")
    print(f"  Picks per game: mean={len(records)/n_games:.1f}")

    # Game-level PnL
    game_pnl = []
    for gk, picks in by_game.items():
        pnl = sum(r["roi"] for r in picks)
        wr = np.mean([r["win"] for r in picks])
        game_pnl.append({
            "game_key": gk, "pnl": pnl, "wr": wr, "n": len(picks),
            "season": picks[0]["season"], "week": picks[0]["week"],
        })
    game_pnl.sort(key=lambda x: x["pnl"], reverse=True)

    # Top-5 and bottom-5 games
    print(f"\n  Top-5 games by PnL:")
    print(f"  {'Game':<22} {'N':>4} {'WR%':>7} {'PnL':>8} {'%Total':>8}")
    print("  " + "-" * 52)
    top5_pnl = 0
    for g in game_pnl[:5]:
        pct = (g["pnl"] / total_pnl * 100) if total_pnl != 0 else 0
        top5_pnl += g["pnl"]
        print(f"  {g['game_key']:<22} {g['n']:>4} {g['wr']:>6.1%} "
              f"{g['pnl']:>+7.1f} {pct:>+7.1f}%")
    top5_pct = (top5_pnl / total_pnl * 100) if total_pnl != 0 else 0
    print(f"  {'Top-5 combined:':<30} {top5_pnl:>+7.1f} {top5_pct:>+7.1f}%")

    print(f"\n  Bottom-5 games by PnL:")
    print(f"  {'Game':<22} {'N':>4} {'WR%':>7} {'PnL':>8}")
    print("  " + "-" * 42)
    for g in game_pnl[-5:]:
        print(f"  {g['game_key']:<22} {g['n']:>4} {g['wr']:>6.1%} "
              f"{g['pnl']:>+7.1f}")

    # Concentration: what % of total PnL comes from top N games?
    if total_pnl > 0:
        cumul = 0
        for i, g in enumerate(game_pnl):
            cumul += g["pnl"]
            if cumul >= total_pnl * 0.5:
                print(f"\n  50% of PnL from top {i+1}/{n_games} games "
                      f"({100*(i+1)/n_games:.0f}%)")
                break

    # Game-clustered bootstrap CI
    rng = np.random.default_rng(42)
    gk_list = list(by_game.keys())
    boot = []
    for _ in range(2000):
        sampled = rng.choice(len(gk_list), size=len(gk_list), replace=True)
        bpicks = []
        for idx in sampled:
            bpicks.extend(by_game[gk_list[idx]])
        if bpicks:
            boot.append(np.mean([r["roi"] for r in bpicks]))
    ci_lo = np.percentile(boot, 2.5)
    ci_hi = np.percentile(boot, 97.5)
    print(f"\n  Game-clustered CI:")
    print(f"    ROI = {total_roi:+.1%}  95% CI [{ci_lo:+.1%}, {ci_hi:+.1%}]")
    signif = "YES" if ci_lo > 0 else "no"
    print(f"    Significant: {signif}")

    # Stat mix
    print(f"\n  Stat breakdown:")
    print(f"  {'Stat':<12} {'N':>5} {'WR%':>7} {'ROI%':>8}")
    print("  " + "-" * 35)
    by_stat = defaultdict(list)
    for r in records:
        by_stat[r["stat"]].append(r)
    for stat in sorted(by_stat):
        items = by_stat[stat]
        if len(items) < 3:
            continue
        wr = np.mean([r["win"] for r in items])
        roi = np.mean([r["roi"] for r in items])
        print(f"  {stat:<12} {len(items):>5} {wr:>6.1%} {roi:>+7.1%}")


def _decimal_to_american(dec_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if dec_odds >= 2.0:
        return (dec_odds - 1) * 100
    elif dec_odds > 1.0:
        return -100 / (dec_odds - 1)
    return 0


def roi_by_odds_band():
    """ROI by odds band × stat with game-clustered CI.

    Bands: ≤-300, -300 to -150, -150 to +150, >+150 (American odds).
    Uses v1.2 re-scored edges and actual offered prices.
    """
    v0_preds = _load_graded_predictions()
    v12_preds = _rescore_v1(v0_preds, use_v12=True)

    # Build records with actual prices (same as derive_roi_thresholds)
    records = []
    for p in v12_preds:
        if p.get("stat_category") != "yardage":
            continue
        actual = p.get("actual_value")
        line = p.get("prop_line")
        edge = p.get("edge_vs_market")
        if actual is None or line is None or edge is None:
            continue

        p_ov = p["p_over"]
        direction = "over" if p_ov > 0.5 else "under"

        if direction == "over":
            price = p.get("prop_over_price")
        else:
            price = p.get("prop_under_price")

        if price is None or price <= 1.0:
            continue

        if actual > line:
            hit = 1.0
        elif actual == line:
            hit = 0.5
        else:
            hit = 0.0

        win = hit if direction == "over" else 1.0 - hit

        if win == 1.0:
            roi = price - 1.0
        elif win == 0.0:
            roi = -1.0
        else:
            roi = 0.0

        american = _decimal_to_american(price)

        records.append({
            "edge": abs(edge),
            "win": win,
            "roi": roi,
            "price": price,
            "american": american,
            "stat": p["stat_type"],
            "pos": p["position"],
            "game_id": p.get("game_id"),
            "season": p["season"],
            "week": p["week"],
        })

    print(f"\n{'='*80}")
    print("ROI BY ODDS BAND × STAT (v1.2, actual prices)")
    print(f"N = {len(records)} yardage picks with prices")
    print(f"{'='*80}")

    # Define bands by American odds
    bands = [
        ("≤-300",    lambda a: a <= -300),
        ("-300/-150", lambda a: -300 < a <= -150),
        ("-150/+150", lambda a: -150 < a <= 150),
        (">+150",    lambda a: a > 150),
    ]

    stats = sorted(set(r["stat"] for r in records))

    # ── 1. Overall by band ──
    print(f"\n  Overall by odds band:")
    print(f"  {'Band':12s} {'N':>6s} {'WR%':>7s} {'BE%':>7s} "
          f"{'ROI%':>8s} {'PnL':>7s}")
    print("  " + "-" * 55)

    for band_name, band_fn in bands:
        filt = [r for r in records if band_fn(r["american"])]
        if not filt:
            continue
        wr = np.mean([r["win"] for r in filt])
        be = np.mean([1/r["price"] for r in filt])
        roi = np.mean([r["roi"] for r in filt])
        pnl = sum(r["roi"] for r in filt)
        print(f"  {band_name:12s} {len(filt):6d} {wr:7.1%} {be:7.1%} "
              f"{roi:+8.1%} {pnl:+7.1f}")

    # ── 2. Band × Stat ──
    print(f"\n  By odds band × stat:")
    print(f"  {'Band':12s} {'Stat':12s} {'N':>6s} {'WR%':>7s} {'BE%':>7s} "
          f"{'ROI%':>8s}")
    print("  " + "-" * 60)

    for band_name, band_fn in bands:
        for stat in stats:
            filt = [r for r in records
                    if band_fn(r["american"]) and r["stat"] == stat]
            if len(filt) < 5:
                continue
            wr = np.mean([r["win"] for r in filt])
            be = np.mean([1/r["price"] for r in filt])
            roi = np.mean([r["roi"] for r in filt])
            print(f"  {band_name:12s} {stat:12s} {len(filt):6d} {wr:7.1%} "
                  f"{be:7.1%} {roi:+8.1%}")

    # ── 3. Game-clustered CI per band ──
    print(f"\n  Game-clustered ROI CI per band:")
    print(f"  {'Band':12s} {'N':>6s} {'Games':>6s} {'ROI%':>8s} "
          f"{'95% CI':>16s} {'Signif':>7s}")
    print("  " + "-" * 65)

    n_boot = 2000
    rng = np.random.default_rng(42)

    for band_name, band_fn in bands:
        filt = [r for r in records if band_fn(r["american"])]
        if len(filt) < 10:
            continue

        by_game = defaultdict(list)
        for r in filt:
            gk = r.get("game_id") or (r["season"], r["week"])
            by_game[gk].append(r)

        gk_list = list(by_game.keys())
        gk_idx = np.arange(len(gk_list))
        boot_rois = []
        for _ in range(n_boot):
            sampled = rng.choice(gk_idx, size=len(gk_list), replace=True)
            bpicks = []
            for idx in sampled:
                bpicks.extend(by_game[gk_list[idx]])
            if bpicks:
                boot_rois.append(np.mean([r["roi"] for r in bpicks]))

        mean_roi = np.mean([r["roi"] for r in filt])
        ci_lo = np.percentile(boot_rois, 2.5)
        ci_hi = np.percentile(boot_rois, 97.5)
        ci_str = f"[{ci_lo:+.1%},{ci_hi:+.1%}]"
        signif = "YES" if ci_lo > 0 else ("NEGATIVE" if ci_hi < 0 else "no")

        print(f"  {band_name:12s} {len(filt):6d} {len(gk_list):6d} "
              f"{mean_roi:+8.1%} {ci_str:>16s} {signif:>7s}")


def roi_price_filter_report():
    """ROI by stat for picks kept vs excluded by -150 odds filter (v1.2).

    Also reports location-uncertainty flag ROI split within the kept pool.
    """
    from nfl_interim_dispersion import (
        load_cached_params, load_v12_base_rates,
        check_location_uncertain, get_interim_cv,
    )
    load_cached_params()
    load_v12_base_rates()

    v0_preds = _load_graded_predictions()
    v12_preds = _rescore_v1(v0_preds, use_v12=True)

    _MIN_DECIMAL = 1.6667  # American -150
    _YARDAGE = {"pass_yds", "rush_yds", "rec_yds"}

    records = []
    for p in v12_preds:
        if p.get("stat_category") != "yardage":
            continue
        actual = p.get("actual_value")
        line = p.get("prop_line")
        edge = p.get("edge_vs_market")
        if actual is None or line is None or edge is None:
            continue

        p_ov = p["p_over"]
        direction = "over" if p_ov > 0.5 else "under"
        price = (p.get("prop_over_price") if direction == "over"
                 else p.get("prop_under_price"))
        if price is None or price <= 1.0:
            continue

        if actual > line:
            hit = 1.0
        elif actual == line:
            hit = 0.5
        else:
            hit = 0.0
        win = hit if direction == "over" else 1.0 - hit
        roi = (price - 1.0) if win == 1.0 else (-1.0 if win == 0.0 else 0.0)

        american = _decimal_to_american(price)
        kept = price >= _MIN_DECIMAL

        # Location uncertainty flag
        stat = p["stat_type"]
        pos = p["position"]
        mean = p["consensus_mean"]
        devigged = p.get("devigged_implied")
        uncertain = False
        if stat in _YARDAGE and mean and mean > 0 and devigged is not None:
            cv = get_interim_cv(stat, pos, mean, use_v12=True)
            if cv:
                uncertain = check_location_uncertain(
                    line, mean, cv, stat, pos, devigged)

        records.append({
            "stat": stat, "pos": pos, "roi": roi, "win": win,
            "price": price, "american": american, "kept": kept,
            "uncertain": uncertain,
            "game_id": p.get("game_id"), "season": p["season"],
            "week": p["week"], "edge": abs(edge),
        })

    print(f"\n{'='*80}")
    print("PRICE-BAND FILTER REPORT (v1.2, odds > -150 kept)")
    print(f"{'='*80}")

    # ── 1. Kept vs excluded overall and by stat ──
    kept = [r for r in records if r["kept"]]
    excl = [r for r in records if not r["kept"]]

    print(f"\n  {'Pool':<12} {'N':>5} {'WR%':>7} {'ROI%':>8} {'Avg$':>8}")
    print("  " + "-" * 45)
    for label, pool in [("KEPT", kept), ("EXCLUDED", excl), ("ALL", records)]:
        if not pool:
            continue
        wr = np.mean([r["win"] for r in pool])
        roi = np.mean([r["roi"] for r in pool])
        avg_am = np.mean([r["american"] for r in pool])
        print(f"  {label:<12} {len(pool):>5} {wr:>6.1%} {roi:>+7.1%} {avg_am:>+7.0f}")

    stats = sorted(set(r["stat"] for r in records))
    print(f"\n  Kept pool by stat:")
    print(f"  {'Stat':<12} {'N':>5} {'WR%':>7} {'ROI%':>8}")
    print("  " + "-" * 35)
    for s in stats:
        filt = [r for r in kept if r["stat"] == s]
        if len(filt) < 5:
            continue
        wr = np.mean([r["win"] for r in filt])
        roi = np.mean([r["roi"] for r in filt])
        print(f"  {s:<12} {len(filt):>5} {wr:>6.1%} {roi:>+7.1%}")

    # ── 2. Location-uncertainty flag split (kept pool only) ──
    flagged = [r for r in kept if r["uncertain"]]
    unflagged = [r for r in kept if not r["uncertain"]]

    print(f"\n  Location-uncertainty flag (kept pool only):")
    print(f"  {'Flag':<12} {'N':>5} {'WR%':>7} {'ROI%':>8} {'AvgEdge':>8}")
    print("  " + "-" * 45)
    for label, pool in [("Unflagged", unflagged), ("Flagged", flagged)]:
        if not pool:
            continue
        wr = np.mean([r["win"] for r in pool])
        roi = np.mean([r["roi"] for r in pool])
        avg_edge = np.mean([r["edge"] for r in pool])
        print(f"  {label:<12} {len(pool):>5} {wr:>6.1%} {roi:>+7.1%} "
              f"{avg_edge:>+7.1%}")

    # Game-clustered CI for flagged vs unflagged
    n_boot = 2000
    rng = np.random.default_rng(42)

    print(f"\n  Game-clustered CI:")
    print(f"  {'Flag':<12} {'N':>5} {'Games':>6} {'ROI%':>8} {'95% CI':>16}")
    print("  " + "-" * 55)
    for label, pool in [("Unflagged", unflagged), ("Flagged", flagged)]:
        if len(pool) < 10:
            continue
        by_game = defaultdict(list)
        for r in pool:
            gk = r.get("game_id") or (r["season"], r["week"])
            by_game[gk].append(r)
        gk_list = list(by_game.keys())
        gk_idx = np.arange(len(gk_list))
        boot = []
        for _ in range(n_boot):
            sampled = rng.choice(gk_idx, size=len(gk_list), replace=True)
            bpicks = []
            for idx in sampled:
                bpicks.extend(by_game[gk_list[idx]])
            if bpicks:
                boot.append(np.mean([r["roi"] for r in bpicks]))
        mean_roi = np.mean([r["roi"] for r in pool])
        ci_lo = np.percentile(boot, 2.5)
        ci_hi = np.percentile(boot, 97.5)
        print(f"  {label:<12} {len(pool):>5} {len(gk_list):>6} "
              f"{mean_roi:>+7.1%} [{ci_lo:+.1%},{ci_hi:+.1%}]")


def report_pass_yds_watchlist():
    """Report pass_yds at ≥5% edge as a separate watchlist line.

    Called from pipeline and weekly diagnostics. pass_yds is -5.5pp vs
    market at ≥5% edge (N=178 v0, N=141 with prices). Track forward
    without excluding — decision deferred to ≥8 weeks.
    """
    v0_preds = _load_graded_predictions()
    v12_preds = _rescore_v1(v0_preds, use_v12=True)

    pass_yds_5pct = []
    for p in v12_preds:
        if p.get("stat_category") != "yardage":
            continue
        if p.get("stat_type") != "pass_yds":
            continue
        edge = p.get("edge_vs_market")
        if edge is None or abs(edge) < 0.05:
            continue
        actual = p.get("actual_value")
        line = p.get("prop_line")
        if actual is None or line is None:
            continue

        p_ov = p["p_over"]
        direction = "over" if p_ov > 0.5 else "under"
        devigged = p.get("devigged_implied", 0.5)

        if actual > line:
            hit = 1.0
        elif actual == line:
            hit = 0.5
        else:
            hit = 0.0

        win = hit if direction == "over" else 1.0 - hit
        mkt_win = devigged if direction == "over" else 1.0 - devigged

        # ROI at actual price
        if direction == "over":
            price = p.get("prop_over_price")
        else:
            price = p.get("prop_under_price")

        roi = None
        if price and price > 1.0:
            roi = (price - 1.0) if win == 1.0 else (-1.0 if win == 0.0 else 0.0)

        pass_yds_5pct.append({
            "win": win, "mkt_win": mkt_win, "edge": abs(edge),
            "roi": roi, "week": p["week"],
        })

    if not pass_yds_5pct:
        print("  [pass_yds watchlist] No data yet")
        return

    n = len(pass_yds_5pct)
    wr = np.mean([p["win"] for p in pass_yds_5pct])
    mkt_wr = np.mean([p["mkt_win"] for p in pass_yds_5pct])
    vs_mkt = wr - mkt_wr
    with_roi = [p for p in pass_yds_5pct if p["roi"] is not None]
    roi_str = (f"ROI={np.mean([p['roi'] for p in with_roi]):+.1%} "
               f"(N={len(with_roi)} priced)") if with_roi else "no prices"

    weeks = sorted(set(p["week"] for p in pass_yds_5pct))

    print(f"  [pass_yds ≥5% edge] N={n} picks, weeks={weeks}")
    print(f"    WR={wr:.1%} vs mkt={mkt_wr:.1%} → {vs_mkt:+.1%}pp")
    print(f"    {roi_str}")
    if vs_mkt < -0.03:
        print(f"    ⚠ UNDERPERFORMING: {vs_mkt*100:+.1f}pp vs market")


def _run_three_way_comparison():
    """Full v0 vs v1.1 vs v1.2 comparison."""
    print("=" * 90)
    print("V0 vs V1.1 vs V1.2 CALIBRATION COMPARISON")
    print("Evaluated on 2026 W2-W5 graded predictions")
    print("v0 = lognormal + flat CV")
    print("v1.1 = gamma + PIT-calibrated flat CV")
    print("v1.2 = gamma + PIT-CV + shifted location (decoupled)")
    print("=" * 90)

    v0_preds = _load_graded_predictions()
    v11_preds = _rescore_v1(v0_preds, use_v11=True)
    v12_preds = _rescore_v1(v0_preds, use_v12=True)
    v0_yard = [p for p in v0_preds if p["stat_category"] == "yardage"]

    print(f"\nN = {len(v0_yard)} yardage predictions")

    # ── 1. Reliability curves ──
    print("\n" + "─" * 90)
    print("1. RELIABILITY CURVES (10 bins, yardage)")
    print("─" * 90)
    v0_curve = reliability_curve(v0_preds, n_bins=10)
    v11_curve = reliability_curve(v11_preds, n_bins=10)
    v12_curve = reliability_curve(v12_preds, n_bins=10)

    v0_map = {r["bin"]: r for r in v0_curve}
    v11_map = {r["bin"]: r for r in v11_curve}
    v12_map = {r["bin"]: r for r in v12_curve}
    all_bins = sorted(set(v0_map) | set(v11_map) | set(v12_map))

    # Each model bins independently — show per-model N and actual
    print(f"  {'Bin':12s} {'v0_N':>5s} {'v0 Gap':>7s}  "
          f"{'v11_N':>5s} {'v11Gap':>7s}  "
          f"{'v12_N':>5s} {'v12Gap':>7s}  {'Actual':>7s}")
    for b in all_bins:
        v0r = v0_map.get(b, {})
        v11r = v11_map.get(b, {})
        v12r = v12_map.get(b, {})
        v0n = v0r.get("n", 0)
        v11n = v11r.get("n", 0)
        v12n = v12r.get("n", 0)
        actual = v12r.get("actual_rate", v11r.get("actual_rate",
                 v0r.get("actual_rate", 0)))
        v0g = v0r.get("gap", 0)
        v11g = v11r.get("gap", 0)
        v12g = v12r.get("gap", 0)
        print(f"  {b:12s} {v0n:5d} {v0g:+7.4f}  "
              f"{v11n:5d} {v11g:+7.4f}  "
              f"{v12n:5d} {v12g:+7.4f}  {actual:7.4f}")

    # Brier scores
    def brier(preds_list):
        terms = []
        for p in preds_list:
            if p.get("stat_category") != "yardage":
                continue
            if p["p_over"] is not None and p["actual_value"] is not None:
                hit = 1.0 if p["actual_value"] > p["prop_line"] else (
                    0.5 if p["actual_value"] == p["prop_line"] else 0.0)
                terms.append((p["p_over"] - hit) ** 2)
        return np.mean(terms) if terms else None

    b0 = brier(v0_preds)
    b11 = brier(v11_preds)
    b12 = brier(v12_preds)
    print(f"\n  Brier: v0={b0:.4f}  v1.1={b11:.4f}  v1.2={b12:.4f}")
    print(f"  Δ vs v0: v1.1={b11-b0:+.4f}  v1.2={b12-b0:+.4f}")

    # ── 2. PIT histograms ──
    print("\n" + "─" * 90)
    print("2. PIT HISTOGRAMS")
    print("─" * 90)
    v0_pit = pit_values(v0_preds)

    v11_pit_vals = defaultdict(list)
    for p in v11_preds:
        if p.get("v1_pit") is not None:
            v11_pit_vals[p["stat_type"]].append(p["v1_pit"])
    v12_pit_vals = defaultdict(list)
    for p in v12_preds:
        if p.get("v1_pit") is not None:
            v12_pit_vals[p["stat_type"]].append(p["v1_pit"])

    for stat in ["rec_yds", "rush_yds", "pass_yds"]:
        v0_vals = v0_pit.get(stat, [])
        v11_vals = v11_pit_vals.get(stat, [])
        v12_vals = v12_pit_vals.get(stat, [])
        if len(v0_vals) < 30:
            continue

        v0_hist = pit_histogram(v0_vals, n_bins=10)
        v11_hist = pit_histogram(v11_vals, n_bins=10)
        v12_hist = pit_histogram(v12_vals, n_bins=10)

        print(f"\n  {stat} (N={len(v0_vals)}):")
        print(f"  {'Bin':12s} {'v0':>7s} {'v1.1':>7s} {'v1.2':>7s}")
        for h0, h11, h12 in zip(v0_hist, v11_hist, v12_hist):
            print(f"  {h0['bin']:12s} {h0['ratio']:7.3f} {h11['ratio']:7.3f} "
                  f"{h12['ratio']:7.3f}")

        if len(v0_hist) >= 4:
            def u_shape(hist):
                ext = (hist[0]["ratio"] + hist[-1]["ratio"]) / 2
                mid = np.mean([h["ratio"] for h in hist[3:7]])
                return ext / mid if mid > 0 else 99
            print(f"  U-shape: v0={u_shape(v0_hist):.2f}  "
                  f"v1.1={u_shape(v11_hist):.2f}  "
                  f"v1.2={u_shape(v12_hist):.2f}")

    # ── 3. CRPS ──
    print("\n" + "─" * 90)
    print("3. CRPS COMPARISON")
    print("─" * 90)
    v0_crps = compute_crps(v0_preds)

    v11_crps_by_stat = defaultdict(list)
    for p in v11_preds:
        if p.get("v1_crps") is not None:
            v11_crps_by_stat[p["stat_type"]].append(p["v1_crps"])
    v12_crps_by_stat = defaultdict(list)
    for p in v12_preds:
        if p.get("v1_crps") is not None:
            v12_crps_by_stat[p["stat_type"]].append(p["v1_crps"])

    print(f"  {'Stat':12s} {'v0':>9s} {'v1.1':>9s} {'v1.2':>9s} "
          f"{'Δ11%':>7s} {'Δ12%':>7s} {'N':>5s}")
    for stat in sorted(v0_crps):
        v0c = v0_crps[stat]["mean_crps"]
        v11_vals = v11_crps_by_stat.get(stat, [])
        v12_vals = v12_crps_by_stat.get(stat, [])
        v11c = round(float(np.mean(v11_vals)), 3) if v11_vals else None
        v12c = round(float(np.mean(v12_vals)), 3) if v12_vals else None
        d11 = f"{(v11c-v0c)/v0c*100:+6.1f}%" if v11c else "   N/A"
        d12 = f"{(v12c-v0c)/v0c*100:+6.1f}%" if v12c else "   N/A"
        v11s = f"{v11c:9.3f}" if v11c else "      N/A"
        v12s = f"{v12c:9.3f}" if v12c else "      N/A"
        print(f"  {stat:12s} {v0c:9.3f} {v11s} {v12s} {d11:>7s} {d12:>7s} "
              f"{v0_crps[stat]['n']:5d}")

    # ── 4. Reliability by stat×position ──
    print("\n" + "─" * 90)
    print("4. RELIABILITY BY STAT × POSITION (v1.2)")
    print("─" * 90)
    v12_sp = reliability_by_stat_position(v12_preds)
    for key, data in sorted(v12_sp.items()):
        print(f"\n  {key} (N={data['n']}):")
        print(f"  {'Bin':12s} {'N':>5s} {'Pred':>7s} {'Actual':>7s} {'Gap':>7s}")
        for row in data["curve"]:
            print(f"  {row['bin']:12s} {row['n']:5d} {row['mean_predicted']:7.3f} "
                  f"{row['actual_rate']:7.3f} {row['gap']:+7.3f}")

    # ── 5. Edge & volume ──
    print("\n" + "─" * 90)
    print("5. EDGE & VOLUME DISTRIBUTION CHANGES")
    print("─" * 90)

    v0_edges = [p["edge_vs_market"] for p in v0_preds
                if p.get("edge_vs_market") is not None and p["stat_category"] == "yardage"]
    v11_edges = [p["edge_vs_market"] for p in v11_preds
                 if p.get("edge_vs_market") is not None and p["stat_category"] == "yardage"]
    v12_edges = [p["edge_vs_market"] for p in v12_preds
                 if p.get("edge_vs_market") is not None and p["stat_category"] == "yardage"]

    print(f"\n  Edge distribution (|edge_vs_market|):")
    print(f"  {'':15s} {'v0':>8s} {'v1.1':>8s} {'v1.2':>8s}")
    if v0_edges and v11_edges and v12_edges:
        v0_abs = np.abs(v0_edges)
        v11_abs = np.abs(v11_edges)
        v12_abs = np.abs(v12_edges)
        for label, fn in [("Mean", np.mean), ("Median", np.median),
                          ("P90", lambda x: np.percentile(x, 90)),
                          ("Max", np.max)]:
            print(f"  {label:15s} {fn(v0_abs):8.4f} {fn(v11_abs):8.4f} {fn(v12_abs):8.4f}")

    print(f"\n  Pick volume at edge thresholds:")
    print(f"  {'Threshold':15s} {'v0':>5s} {'v1.1':>5s} {'v1.2':>5s}")
    for thresh in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]:
        v0_n = sum(1 for e in v0_edges if abs(e) >= thresh)
        v11_n = sum(1 for e in v11_edges if abs(e) >= thresh)
        v12_n = sum(1 for e in v12_edges if abs(e) >= thresh)
        print(f"  ≥{thresh:.0%}{'':<10s} {v0_n:5d} {v11_n:5d} {v12_n:5d}")

    # Direction flips v0→v1.2
    flips_o2u = flips_u2o = 0
    for v0p, v12p in zip(v0_preds, v12_preds):
        if v0p["stat_category"] != "yardage":
            continue
        v0d = "over" if v0p["p_over"] > 0.5 else "under"
        v12d = "over" if v12p["p_over"] > 0.5 else "under"
        if v0d != v12d:
            if v0d == "over":
                flips_o2u += 1
            else:
                flips_u2o += 1
    total = flips_o2u + flips_u2o
    print(f"\n  Direction flips (v0→v1.2): {total} / {len(v0_yard)}")
    print(f"    over→under: {flips_o2u}")
    print(f"    under→over: {flips_u2o}")
    if total > 0:
        balance = min(flips_o2u, flips_u2o) / total
        print(f"    balance ratio: {balance:.2f}")

    # ── 6. Flip validation ──
    if total > 0:
        print("\n" + "─" * 90)
        print("6. FLIP VALIDATION (v0→v1.2)")
        print("─" * 90)
        _validate_flips(v0_preds, v12_preds)

    # ── 7. P(over at projection) vs empirical ──
    print("\n" + "─" * 90)
    print("7. P(OVER AT PROJECTION) — model vs empirical (target ±3pp)")
    print("─" * 90)
    _check_p_over_at_projection(v12_preds)


# ══════════════════════════════════════════════════════════════════════════════
# Pre-registered forward tracker: rec_yds OVER
# ══════════════════════════════════════════════════════════════════════════════
#
# Rule (pre-registered 2026-10-11, before W6 grades):
#   stat_type  = rec_yds
#   direction  = OVER
#   filter     = v1.2 P(over) >= devigged_implied + 0.05
#   odds       = prop_over_price > 1.667 (American > −150)
#   scope      = W6+ only (no W2-W5 reuse)
#   decision   = at 8 forward weeks of data
#
# Metrics: N, hit rate (model vs market), ROI at offered prices,
#          game-clustered CI on hit rate and ROI.
#
# DATA REGIME (updated 2026-11-07 with precise bug inventory):
#
# Regime 1 (R1): created_at < 2026-11-05T22:03:48Z (W7–W8)
#   Known bugs active during R1:
#     a) Suppression: near_alt OVER rows silently dropped (active since
#        2026-10-02 when alt-line code deployed). ~48% of alt pool missing.
#     b) Alias bug: build_consensus used _normalize_player_name while
#        _fetch_current_nfl_prop_lines used _resolve_nfl_name. Affected
#        players (Chig Okonkwo, Mike Evans, Gabe Davis) never matched.
#        68 rows across R1+R2 flagged data_error='alias_mismatch'.
#     c) Sparse alt-line coverage: W7 avg 1.9 OVER rungs/player,
#        W8 avg 1.3/player. Most players have main line only.
#   Other:
#     - v1.2_shifted_gamma pipeline started W7. W6 has only v0 predictions.
#     - bets.db crash (Oct 23–31) may have lost some W7 predictions.
#     - Total: 207 qualifying, 304 graded rec_yds OVER.
#     - Last R1 row: 2026-11-04T21:27:45Z. No rows in overnight gap.
#
# Regime 2 (R2): 2026-11-05T22:03:48Z <= created_at < 2026-11-07T10:47:26Z
#   W9 pre-fix. Same bugs as R1 (suppression + alias), plus:
#     - Bias-corrected projections deployed (rolling prior + no-bet threshold).
#     - Alt-line rung coverage otherwise complete, but suppression still
#       dropped all near_alt/deep_alt OVER rows.
#   3,171 W9 rows. Last R2 row: 2026-11-06T16:26:12Z. No rows in gap.
#
# Regime 3 (R3): created_at >= 2026-11-07T10:47:26Z (W9 post-fix + W10+)
#   All fixes applied:
#     - All predictions stored (suppression_rule as flag, not skip).
#     - Alias bug fixed (both sides use _resolve_nfl_name).
#     - data_error column flags contaminated historical rows.
#     - UNIQUE key includes line_depth (no alt→main overwrites).
#   This is the first clean regime; primary evaluation pool.
#   1,804 W9 rows belong to R3 (first: 2026-11-07T10:47:26Z).
#
# Implications:
#   - R1 and R2 both have suppression + alias bugs. Use with caution;
#     directional context only.
#   - R3 is the clean pool.  Decision at 8 weeks should weight R3
#     most heavily, with R1/R2 as directional context only.
#   - Regime assignment uses created_at timestamp, NOT week number.
#     Some W9 rows are R2, some are R3.

_PREREG_RULE = {
    "stat_type": "rec_yds",
    "direction": "over",
    "min_edge_pp": 0.05,        # v1.2 P(over) >= devigged + 5pp
    "min_price": 1.667,         # odds > -150
    "min_week": 6,              # W6+ only
    "model_version": "v1.2_shifted_gamma",
    "decision_weeks": 8,
}


def _prereg_load_qualifying(rule: dict = _PREREG_RULE) -> list[dict]:
    """Load predictions qualifying under the pre-registered rule."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT player_name, position, season, week, game_id,
               consensus_mean, prop_line,
               p_over, devigged_implied, edge_vs_market,
               prop_over_price, prop_under_price,
               direction, actual_value, is_win, devig_method,
               model_version, created_at
        FROM fantasy_forward_preds
        WHERE stat_type = ?
          AND method LIKE '%distributional%'
          AND week >= ?
          AND data_error IS NULL
    """, (rule["stat_type"], rule["min_week"])).fetchall()
    conn.close()

    qualified = []
    for r in [dict(x) for x in rows]:
        # Must be v1.2 predictions (or recomputed v1.2)
        mv = r.get("model_version") or ""
        if rule["model_version"] and rule["model_version"] not in mv:
            continue
        # Direction = OVER
        if (r.get("direction") or "").lower() != rule["direction"]:
            continue
        # Edge >= 5pp
        p_over = r.get("p_over")
        devig = r.get("devigged_implied")
        if p_over is None or devig is None:
            continue
        if p_over - devig < rule["min_edge_pp"]:
            continue
        # Odds filter
        price = r.get("prop_over_price")
        if price is None or price <= rule["min_price"]:
            continue
        qualified.append(r)

    return qualified


def _game_clustered_ci(values: list[float], game_ids: list[str],
                       n_boot: int = 10000, alpha: float = 0.05) -> tuple:
    """Bootstrap CI resampling by game cluster."""
    if not values:
        return (float("nan"), float("nan"))

    game_map = defaultdict(list)
    for v, g in zip(values, game_ids):
        game_map[g].append(v)
    game_keys = list(game_map.keys())
    n_games = len(game_keys)
    if n_games < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(game_keys, size=n_games, replace=True)
        boot_vals = []
        for g in sampled:
            boot_vals.extend(game_map[g])
        boot_means.append(np.mean(boot_vals))
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def prereg_rec_yds_over_report():
    """Run the pre-registered rec_yds OVER forward tracker."""
    rule = _PREREG_RULE
    preds = _prereg_load_qualifying(rule)

    print("=" * 80)
    print("PRE-REGISTERED FORWARD TRACKER: rec_yds OVER")
    print("=" * 80)
    print(f"\n  Rule: stat={rule['stat_type']}, dir={rule['direction']}, "
          f"edge>={rule['min_edge_pp']:.0%}pp, "
          f"price>{rule['min_price']:.3f}, W{rule['min_week']}+")
    print(f"  Model: {rule['model_version']}")
    print(f"  Decision: after W14 graded (8 weeks: W7–W14)")
    print(f"  W6 excluded: v1.2_shifted_gamma pipeline started W7; "
          f"W6 has only v0 predictions")

    if not preds:
        print("\n  No qualifying predictions yet. Waiting for W6+ v1.2 data.")
        return {"n": 0, "status": "waiting"}

    graded = [p for p in preds if p["actual_value"] is not None]
    ungraded = [p for p in preds if p["actual_value"] is None]

    weeks_seen = sorted(set(p["week"] for p in preds))
    weeks_graded = sorted(set(p["week"] for p in graded))

    print(f"\n  Weeks with data: {weeks_seen}")
    print(f"  Weeks graded:    {weeks_graded}")
    print(f"  Total qualifying: {len(preds)} ({len(graded)} graded, "
          f"{len(ungraded)} pending)")

    if not graded:
        # Show ungraded qualifying preds
        print(f"\n  Qualifying (pending grade):")
        print(f"  {'Player':25s} {'Pos':4s} {'Wk':>3s} {'Line':>6s} "
              f"{'P(ov)':>6s} {'Mkt':>6s} {'Edge':>6s} {'Price':>6s}")
        for p in sorted(ungraded, key=lambda x: -x["p_over"]):
            print(f"  {p['player_name']:25s} {p['position']:4s} "
                  f"W{p['week']:>2d} {p['prop_line']:6.1f} "
                  f"{p['p_over']:6.3f} {p['devigged_implied']:6.3f} "
                  f"{p['p_over'] - p['devigged_implied']:+6.3f} "
                  f"{p['prop_over_price']:6.3f}")
        return {"n": len(preds), "n_graded": 0, "status": "pending"}

    # ── Compute metrics on graded predictions ──
    hits = []
    profits = []
    game_ids = []
    model_hits = []
    market_hits = []

    print(f"\n  {'Player':25s} {'Pos':4s} {'Wk':>3s} {'Line':>6s} "
          f"{'P(ov)':>6s} {'Mkt':>6s} {'Edge':>6s} {'Price':>6s} "
          f"{'Actual':>7s} {'Hit':>4s} {'Profit':>7s}")
    print("  " + "─" * 100)

    for p in sorted(graded, key=lambda x: (x["week"], x["player_name"])):
        actual = p["actual_value"]
        line = p["prop_line"]
        hit = 1.0 if actual > line else (0.5 if actual == line else 0.0)
        price = p["prop_over_price"]
        profit = (price - 1.0) if hit == 1.0 else (-1.0 if hit == 0.0 else 0.0)

        hits.append(hit)
        profits.append(profit)
        game_ids.append(p["game_id"] or f"W{p['week']}")

        # Model says OVER (by construction — all qualifying), did it hit?
        model_hits.append(hit)
        # Market would say... use devigged P(over) vs 0.5
        mkt_says_over = p["devigged_implied"] > 0.5
        if mkt_says_over:
            market_hits.append(hit)
        else:
            market_hits.append(1.0 - hit)

        tag = "WIN" if hit == 1.0 else ("PUSH" if hit == 0.5 else "LOSS")
        print(f"  {p['player_name']:25s} {p['position']:4s} "
              f"W{p['week']:>2d} {line:6.1f} "
              f"{p['p_over']:6.3f} {p['devigged_implied']:6.3f} "
              f"{p['p_over'] - p['devigged_implied']:+6.3f} "
              f"{price:6.3f} {actual:7.1f} {tag:>4s} {profit:+7.3f}")

    # ── Summary stats ──
    n = len(hits)
    hit_rate = np.mean(hits)
    roi = np.mean(profits)
    total_profit = sum(profits)

    model_acc = np.mean(model_hits)
    market_acc = np.mean(market_hits) if market_hits else float("nan")

    n_games = len(set(game_ids))
    hr_ci = _game_clustered_ci(hits, game_ids)
    roi_ci = _game_clustered_ci(profits, game_ids)

    print(f"\n  SUMMARY (N={n} picks, {n_games} games):")
    print(f"    Hit rate:       {hit_rate:.1%}  "
          f"95% CI [{hr_ci[0]:.1%}, {hr_ci[1]:.1%}]  "
          f"(game-clustered, {n_games} clusters)")
    print(f"    ROI (flat $1):  {roi:+.1%}  "
          f"95% CI [{roi_ci[0]:+.1%}, {roi_ci[1]:+.1%}]  "
          f"(game-clustered)")
    print(f"    Total profit:   {total_profit:+.2f} units on {n} bets")
    print(f"    Model accuracy: {model_acc:.1%} (OVER calls that hit)")
    print(f"    Market accuracy:{market_acc:.1%} (market-direction calls that hit)")

    # ── Breakeven check ──
    avg_price = np.mean([p["prop_over_price"] for p in graded])
    be_rate = 1.0 / avg_price
    print(f"\n    Avg offered price: {avg_price:.3f} (BE = {be_rate:.1%})")
    print(f"    Hit rate vs BE:    {hit_rate:.1%} vs {be_rate:.1%} "
          f"({hit_rate - be_rate:+.1%}pp)")

    # ── Regime split by created_at timestamp (not week number) ──
    # R1: created_at < first W9 row (W7-W8, suppression + alias bugs)
    # R2: first W9 row through last pre-fix W9 row (same bugs + bias-corrected)
    # R3: created_at >= first post-fix row (W9 post-fix + W10+, clean)
    # Exact boundaries derived from DB: no rows exist in the overnight gaps.
    _R2_START = "2026-11-05T22:03:48Z"   # first W9 row
    _R3_START = "2026-11-07T10:47:26Z"   # first post-fix row
    r1 = [p for p in graded if (p.get("created_at") or "") < _R2_START]
    r2 = [p for p in graded
          if _R2_START <= (p.get("created_at") or "") < _R3_START]
    r3 = [p for p in graded if (p.get("created_at") or "") >= _R3_START]
    if r1 or r2 or r3:
        print(f"\n  REGIME SPLIT (by created_at):")
        for label, subset in [
                ("R1 (W7-W8, suppression+alias bugs)", r1),
                ("R2 (W9 pre-fix, same bugs)", r2),
                ("R3 (post-fix, clean)", r3)]:
            if not subset:
                print(f"    {label}: no graded data yet")
                continue
            s_hits = [1.0 if p["actual_value"] > p["prop_line"]
                      else (0.5 if p["actual_value"] == p["prop_line"] else 0.0)
                      for p in subset]
            s_prof = [(p["prop_over_price"] - 1.0) if h == 1.0
                      else (-1.0 if h == 0.0 else 0.0)
                      for p, h in zip(subset, s_hits)]
            s_game_ids = [p.get("game_id") or f"W{p['week']}" for p in subset]
            s_n_games = len(set(s_game_ids))
            s_avg_price = np.mean([p["prop_over_price"] for p in subset])
            s_be = 1.0 / s_avg_price
            s_hr = np.mean(s_hits)
            s_roi = np.mean(s_prof)
            s_hr_ci = _game_clustered_ci(s_hits, s_game_ids)
            s_roi_ci = _game_clustered_ci(s_prof, s_game_ids)
            print(f"    {label}: N={len(subset)} picks, {s_n_games} games")
            print(f"      HR={s_hr:.1%} [{s_hr_ci[0]:.1%}, {s_hr_ci[1]:.1%}] "
                  f"(BE={s_be:.1%}, delta={s_hr - s_be:+.1%}pp)")
            print(f"      ROI={s_roi:+.1%} [{s_roi_ci[0]:+.1%}, {s_roi_ci[1]:+.1%}]")

    # ── Decision status ──
    # First data week is W7 (v1.2 started W7). 8 weeks = W7–W14.
    n_weeks = len(weeks_graded)
    print(f"\n  STATUS: {n_weeks}/{rule['decision_weeks']} weeks graded")
    print(f"  Decision: after W14 graded (W7–W14 = 8 weeks)")
    if n_weeks >= rule["decision_weeks"]:
        if hr_ci[0] > be_rate:
            verdict = "PASS — lower CI bound exceeds breakeven"
        elif hr_ci[1] < be_rate:
            verdict = "FAIL — upper CI bound below breakeven"
        else:
            verdict = "INCONCLUSIVE — CI spans breakeven"
        print(f"  VERDICT: {verdict}")
    else:
        remaining = rule["decision_weeks"] - n_weeks
        print(f"  {remaining} more weeks needed before decision.")

    return {
        "n": n,
        "n_graded": len(graded),
        "n_pending": len(ungraded),
        "weeks_graded": weeks_graded,
        "hit_rate": float(hit_rate),
        "hit_rate_ci": hr_ci,
        "roi": float(roi),
        "roi_ci": roi_ci,
        "model_accuracy": float(model_acc),
        "be_rate": float(be_rate),
        "status": "decision" if n_weeks >= rule["decision_weeks"] else "accumulating",
    }


def check_prediction_coverage(season: int | None = None,
                              week: int | None = None) -> dict:
    """Verify v1.2 predictions exist for every game before kickoff.

    Returns {games: [...], missing: [...], ok: bool}.
    """
    from fantasy_projections import get_current_nfl_week
    if season is None or week is None:
        season, week = get_current_nfl_week()

    # Load game schedule from nfl.db
    nfl_db = os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db")
    conn_nfl = sqlite3.connect(os.path.abspath(nfl_db))
    conn_nfl.row_factory = sqlite3.Row
    games = conn_nfl.execute("""
        SELECT game_id, game_date, gametime, home_team, away_team
        FROM nfl_games WHERE season = ? AND week = ? AND game_type = 'REG'
        ORDER BY game_date, gametime
    """, (season, week)).fetchall()
    conn_nfl.close()

    if not games:
        print(f"  No games found for {season} W{week}")
        return {"games": [], "missing": [], "ok": False, "error": "no_games"}

    # Load predictions from historical.db
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    v12_tag = _PREREG_RULE["model_version"]
    results = []
    missing = []

    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)

    print("=" * 80)
    print(f"PREDICTION COVERAGE CHECK — {season} W{week} ({v12_tag})")
    print("=" * 80)
    print(f"\n  {'Game':25s} {'Kickoff (ET)':18s} {'v1.2 preds':>10s} "
          f"{'Before kick':>12s} {'Status':>10s}")
    print("  " + "─" * 80)

    for g in games:
        game_id = g["game_id"]

        # Kickoff in UTC
        kick_naive = datetime.strptime(
            f"{g['game_date']} {g['gametime']}", "%Y-%m-%d %H:%M")
        kick_et = kick_naive.replace(tzinfo=et)
        kick_utc = kick_et.astimezone(timezone.utc)

        # Count v1.2 predictions for this game
        row = conn.execute("""
            SELECT COUNT(*) as n,
                   MIN(created_at) as first_created,
                   MAX(created_at) as last_created
            FROM fantasy_forward_preds
            WHERE game_id = ? AND model_version = ?
        """, (game_id, v12_tag)).fetchone()

        n_preds = row["n"] if row else 0

        # Check if predictions were created before kickoff
        before_kick = True
        if n_preds > 0 and row["last_created"]:
            try:
                last_dt = datetime.strptime(
                    row["last_created"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc)
                before_kick = last_dt < kick_utc
            except Exception:
                pass

        time_to_kick = (kick_utc - now_utc).total_seconds() / 3600
        if time_to_kick < 0:
            kick_status = "PAST"
        elif n_preds == 0:
            if time_to_kick > 48:
                kick_status = "PENDING"  # still time
            else:
                kick_status = "⚠ MISSING"
                missing.append(game_id)
        elif not before_kick:
            kick_status = "⚠ LATE"
            missing.append(game_id)
        else:
            kick_status = "OK"

        label = f"{g['away_team']}@{g['home_team']}"
        kick_str = kick_et.strftime("%a %I:%M %p ET")
        print(f"  {label:25s} {kick_str:18s} {n_preds:>10d} "
              f"{'yes' if before_kick and n_preds > 0 else 'n/a':>12s} "
              f"{kick_status:>10s}")

        results.append({
            "game_id": game_id,
            "label": label,
            "kickoff_utc": kick_utc.isoformat(),
            "n_preds": n_preds,
            "before_kick": before_kick,
            "status": kick_status,
        })

    conn.close()

    ok = len(missing) == 0 and any(r["n_preds"] > 0 for r in results)
    total_preds = sum(r["n_preds"] for r in results)
    print(f"\n  Total: {total_preds} predictions across {len(games)} games")
    if missing:
        print(f"  ⚠ ALERT: {len(missing)} games missing v1.2 predictions: "
              f"{', '.join(missing)}")
    elif total_preds == 0:
        print(f"  No predictions yet — pipeline runs Wed-Sat during season")
    else:
        print(f"  All games covered ✓")

    return {"games": results, "missing": missing, "ok": ok,
            "total_preds": total_preds}


def weekly_forward_report(season: int | None = None,
                          week: int | None = None) -> dict:
    """Monday-night weekly report: injury edge + prereg + v1.2 vs market Brier.

    Designed to run after MNF settles (Tuesday 12 PM CT or manually).
    If week is None, reports on the just-completed week.
    """
    from fantasy_projections import get_current_nfl_week
    if season is None or week is None:
        season, _cur_week = get_current_nfl_week()
        week = _cur_week - 1 if week is None else week

    result = {"season": season, "week": week}

    print("=" * 80)
    print(f"WEEKLY FORWARD REPORT — {season} W{week}")
    print("=" * 80)

    # ── 1. Injury edge-at-news ──
    print(f"\n{'─' * 80}")
    print("1. INJURY EDGE AT NEWS")
    print("─" * 80)
    try:
        from injury_watcher import report_edge_at_news, print_edge_report
        edge_report = report_edge_at_news()
        print_edge_report(edge_report)
        result["injury_edge"] = edge_report
    except Exception as e:
        print(f"  Error: {e}")
        result["injury_edge"] = {"error": str(e)}

    # ── 2. Pre-registered rec_yds OVER tracker ──
    print(f"\n{'─' * 80}")
    print("2. PRE-REGISTERED TRACKER: rec_yds OVER")
    print("─" * 80)
    try:
        prereg = prereg_rec_yds_over_report()
        result["prereg"] = prereg
    except Exception as e:
        print(f"  Error: {e}")
        result["prereg"] = {"error": str(e)}

    # ── 3. v1.2 vs market forward Brier ──
    print(f"\n{'─' * 80}")
    print("3. v1.2 vs MARKET FORWARD BRIER")
    print("─" * 80)
    try:
        conn = sqlite3.connect(os.path.abspath(_HIST_DB))
        conn.row_factory = sqlite3.Row
        v12_tag = _PREREG_RULE["model_version"]

        preds = conn.execute("""
            SELECT player_name, stat_type, position, week, game_id,
                   p_over, devigged_implied, prop_line, actual_value,
                   method, model_version
            FROM fantasy_forward_preds
            WHERE season = ? AND model_version = ?
              AND actual_value IS NOT NULL
              AND data_error IS NULL
            ORDER BY week, player_name
        """, (season, v12_tag)).fetchall()
        conn.close()

        if not preds:
            print(f"\n  No graded v1.2 predictions for {season} yet.")
            result["brier"] = {"n": 0, "status": "waiting"}
        else:
            # Compute Brier for v1.2 and market (devigged)
            v12_terms = []
            mkt_terms = []
            game_map_v12 = defaultdict(list)
            game_map_mkt = defaultdict(list)
            weeks_seen = set()

            for p in preds:
                if p["p_over"] is None or p["actual_value"] is None:
                    continue
                if p["devigged_implied"] is None:
                    continue
                hit = 1.0 if p["actual_value"] > p["prop_line"] else (
                    0.5 if p["actual_value"] == p["prop_line"] else 0.0)
                v12_err = (p["p_over"] - hit) ** 2
                mkt_err = (p["devigged_implied"] - hit) ** 2
                v12_terms.append(v12_err)
                mkt_terms.append(mkt_err)
                gid = p["game_id"] or f"W{p['week']}"
                game_map_v12[gid].append(v12_err)
                game_map_mkt[gid].append(mkt_err)
                weeks_seen.add(p["week"])

            n = len(v12_terms)
            if n == 0:
                print(f"\n  No yardage predictions with outcomes.")
                result["brier"] = {"n": 0, "status": "no_data"}
            else:
                b_v12 = float(np.mean(v12_terms))
                b_mkt = float(np.mean(mkt_terms))
                delta = b_v12 - b_mkt

                # Game-clustered CI on delta
                game_deltas = []
                for gid in game_map_v12:
                    if gid in game_map_mkt:
                        v12_game = np.mean(game_map_v12[gid])
                        mkt_game = np.mean(game_map_mkt[gid])
                        game_deltas.append(v12_game - mkt_game)

                ci = None
                if len(game_deltas) >= 5:
                    rng = np.random.default_rng(42)
                    boot = [float(np.mean(rng.choice(game_deltas,
                            size=len(game_deltas), replace=True)))
                            for _ in range(5000)]
                    ci = (float(np.percentile(boot, 2.5)),
                          float(np.percentile(boot, 97.5)))

                print(f"\n  Weeks: {sorted(weeks_seen)}")
                print(f"  N predictions: {n} across {len(game_map_v12)} games")
                print(f"  Brier v1.2:  {b_v12:.4f}")
                print(f"  Brier market: {b_mkt:.4f}")
                print(f"  Δ (v1.2 − mkt): {delta:+.4f}"
                      + (f"  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""))

                # Per-stat breakdown
                stat_map = defaultdict(lambda: {"v12": [], "mkt": []})
                for p in preds:
                    if p["p_over"] is None or p["actual_value"] is None:
                        continue
                    if p["devigged_implied"] is None:
                        continue
                    hit = 1.0 if p["actual_value"] > p["prop_line"] else (
                        0.5 if p["actual_value"] == p["prop_line"] else 0.0)
                    stat_map[p["stat_type"]]["v12"].append(
                        (p["p_over"] - hit) ** 2)
                    stat_map[p["stat_type"]]["mkt"].append(
                        (p["devigged_implied"] - hit) ** 2)

                print(f"\n  {'Stat':12s} {'N':>5s} {'v1.2':>8s} {'Market':>8s} {'Δ':>8s}")
                for st in sorted(stat_map):
                    sv = stat_map[st]
                    sv12 = float(np.mean(sv["v12"]))
                    smkt = float(np.mean(sv["mkt"]))
                    print(f"  {st:12s} {len(sv['v12']):5d} "
                          f"{sv12:8.4f} {smkt:8.4f} {sv12-smkt:+8.4f}")

                result["brier"] = {
                    "n": n,
                    "n_games": len(game_map_v12),
                    "weeks": sorted(weeks_seen),
                    "v12": round(b_v12, 4),
                    "market": round(b_mkt, 4),
                    "delta": round(delta, 4),
                    "ci": ci,
                }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
        result["brier"] = {"error": str(e)}

    # ── 4. Prediction coverage for next week ──
    print(f"\n{'─' * 80}")
    print(f"4. COVERAGE CHECK — W{week + 1}")
    print("─" * 80)
    try:
        cov = check_prediction_coverage(season, week + 1)
        result["next_week_coverage"] = {
            "ok": cov.get("ok"),
            "total_preds": cov.get("total_preds", 0),
            "missing": cov.get("missing", []),
        }
    except Exception as e:
        print(f"  Error: {e}")
        result["next_week_coverage"] = {"error": str(e)}

    return result


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "compare":
        run_v0_v1_comparison()
    elif cmd == "compare-v11":
        run_v0_v1_comparison(use_v11=True)
    elif cmd == "compare-v12":
        run_v0_v1_comparison(use_v12=True)
    elif cmd == "compare-3way":
        run_v0_v1_comparison(three_way=True)
    elif cmd == "fit-v12":
        from nfl_interim_dispersion import fit_v12_base_rates
        fit_v12_base_rates()
    elif cmd == "edge-thresholds":
        derive_edge_thresholds()
    elif cmd == "roi-thresholds":
        derive_roi_thresholds()
    elif cmd == "pass-yds-watch":
        report_pass_yds_watchlist()
    elif cmd == "roi-odds-band":
        roi_by_odds_band()
    elif cmd == "roi-price-filter":
        roi_price_filter_report()
    elif cmd == "flag-edge-crosstab":
        flag_edge_crosstab()
    elif cmd == "high-conf-concentration":
        high_confidence_concentration()
    elif cmd == "prereg":
        prereg_rec_yds_over_report()
    elif cmd == "coverage":
        _cov_week = int(sys.argv[2]) if len(sys.argv) > 2 else None
        _cov_season = None
        if _cov_week:
            from fantasy_projections import get_current_nfl_week
            _cov_season, _ = get_current_nfl_week()
        check_prediction_coverage(season=_cov_season, week=_cov_week)
    elif cmd == "weekly":
        _wk_week = int(sys.argv[2]) if len(sys.argv) > 2 else None
        weekly_forward_report(week=_wk_week)
    else:
        run_calibration_report(load_nflverse=True)
