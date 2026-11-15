"""
nfl_interim_dispersion.py — Interim dispersion model (v1_interim_gamma).

Replaces the flat _typical_cv per stat with:
  1.  σ = a · μ^b  power-law dispersion, fit per (stat, position)
  2.  Gamma family for yardage (shape = 1/CV², scale = μ·CV²)
  3.  NegBin for counts (receptions, TDs, INTs) — unchanged
  4.  Parameters fitted by CRPS minimisation on 2023-2025 nflverse actuals,
      walk-forward (train weeks 1..w-1, predict week w).

Usage from pipeline:
    from nfl_interim_dispersion import (
        get_interim_dispersion,     # (stat, position, projected_mean) → variance
        p_over_line_gamma,          # replaces p_over_line for yardage
        DISPERSION_PARAMS,          # fitted {(stat, pos): (a, b)} dict
        MODEL_VERSION,
        MODEL_SWITCH_DATE,
    )
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from math import exp, floor, log, sqrt

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar, minimize

logger = logging.getLogger(__name__)

MODEL_VERSION = "v1_interim_gamma"
MODEL_VERSION_V11 = "v1.1_two_stage"
MODEL_VERSION_V12 = "v1.2_shifted_gamma"
MODEL_SWITCH_DATE = "2026-10-09"   # v1.2 live shadow; v0 frozen baseline

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_CACHE_PATH = os.path.join(_DATA_DIR, "interim_dispersion_params.json")
_V11_CACHE_PATH = os.path.join(_DATA_DIR, "v11_dispersion_params.json")

# ══════════════════════════════════════════════════════════════════════════════
# Fitted dispersion parameters: σ = a · μ^b  per (stat, position)
# These are populated by fit_dispersion_model() and cached to disk.
# Keys: (stat_type, position) → {"a": float, "b": float, "cv_at_mean": float}
# ══════════════════════════════════════════════════════════════════════════════
DISPERSION_PARAMS: dict[tuple[str, str], dict] = {}

# Fallback CVs (used when no fit available for a stat×position combo)
_FALLBACK_CV = {
    "pass_yds": 0.35, "rush_yds": 0.65, "rec_yds": 0.76,
    "pass_tds": 0.80, "pass_ints": 1.00, "receptions": 0.40,
}

# v1.1 PIT-calibrated CVs (verified on 2026 W2-W5)
# PIT U-shape targets 0.8-1.25; max reliability gap <8pp at N>=30.
# Keyed by (stat, position) for per-position overrides, then stat-level fallback.
_V11_PIT_CV_BY_POS = {
    ("rec_yds", "WR"): 0.80,   # PIT U=1.24, max gap=7.1pp at N=297
}
_V11_PIT_CV = {
    "pass_yds": 0.35,   # PIT U=0.87, shape=8.16
    "rec_yds": 0.90,    # PIT U=0.94, shape=1.23 (RB/TE default)
    "rush_yds": 0.75,   # PIT U=1.03, shape=1.78
}

# ── v1.2: decoupled location ────────────────────────────────────────────────
# Shrunk P(actual > projection) per (stat, pos).  Fitted LOO on 2026 W2-W5.
# Shrinkage: rate = (n×raw + k×pooled) / (n+k), k=30, pooled=0.4275
_V12_POOLED_RATE = 0.4275
_V12_SHRINKAGE_K = 30
_V12_BASE_RATES: dict[tuple[str, str], float] = {}   # populated by fit_v12_base_rates()
_V12_BASE_RATE_CI: dict[tuple[str, str], float] = {}  # CI halfwidth per combo
_V12_BASE_RATE_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {}  # (ci_lo, ci_hi)
_V12_CACHE_PATH = os.path.join(_DATA_DIR, "v12_base_rates.json")


# ══════════════════════════════════════════════════════════════════════════════
# Gamma distribution helpers
# ══════════════════════════════════════════════════════════════════════════════

def gamma_params(mean: float, cv: float) -> tuple[float, float]:
    """Convert (mean, CV) to gamma (shape, scale) parameters.

    shape = 1 / CV²
    scale = mean · CV²
    mean  = shape · scale = mean  ✓
    var   = shape · scale² = mean² · CV²  ✓
    """
    if mean <= 0 or cv <= 0:
        return (100.0, max(mean, 0.1) / 100.0)  # near-deterministic fallback
    shape = 1.0 / (cv ** 2)
    scale = mean * (cv ** 2)
    return (shape, scale)


def gamma_cdf(x: float, mean: float, cv: float) -> float:
    """CDF of gamma distribution at x."""
    if x <= 0:
        return 0.0
    shape, scale = gamma_params(mean, cv)
    return float(sp_stats.gamma.cdf(x, a=shape, scale=scale))


def gamma_crps_single(actual: float, mean: float, cv: float) -> float:
    """CRPS for gamma distribution — closed-form.

    CRPS(Gamma(α,β), y) = y·(2F(y;α,β) - 1) - αβ·(2F(y;α+1,β) - 1)
                          - β / B(0.5, α+0.5)
    where F(y;α,β) is the gamma CDF with shape α and scale β,
    and B is the beta function.
    """
    from scipy.special import beta as beta_fn
    shape, scale = gamma_params(mean, cv)
    if shape <= 0 or scale <= 0:
        return abs(actual - mean)

    y = max(actual, 0.0)
    F_y = float(sp_stats.gamma.cdf(y, a=shape, scale=scale))
    F_y_plus1 = float(sp_stats.gamma.cdf(y, a=shape + 1, scale=scale))

    crps = (y * (2 * F_y - 1)
            - shape * scale * (2 * F_y_plus1 - 1)
            - scale / beta_fn(0.5, shape + 0.5))
    return abs(crps)


def p_over_line_gamma(line: float, mean: float, cv: float) -> float:
    """P(actual > line) from gamma distribution."""
    if mean <= 0:
        return 0.5
    return 1.0 - gamma_cdf(line, mean, cv)


def gamma_params_shifted(projection: float, cv: float,
                         p_over_at_proj: float) -> tuple[float, float]:
    """Compute gamma (shape, scale) decoupling location from dispersion.

    shape is determined by cv (dispersion).
    scale is chosen so that P(X > projection) = p_over_at_proj.

    This means gamma_CDF(projection, shape, scale) = 1 - p_over_at_proj.
    """
    if projection <= 0 or cv <= 0:
        return (100.0, max(projection, 0.1) / 100.0)

    shape = 1.0 / (cv ** 2)
    target_cdf = 1.0 - p_over_at_proj

    # Clamp target_cdf to avoid degenerate cases
    target_cdf = max(0.01, min(target_cdf, 0.99))

    # Standardized quantile: gamma_ppf(target_cdf, shape, scale=1)
    q = float(sp_stats.gamma.ppf(target_cdf, a=shape))
    if q <= 0:
        return (shape, projection * cv ** 2)  # fallback to mean-param

    scale = projection / q
    return (shape, scale)


def gamma_cdf_shifted(x: float, projection: float, cv: float,
                      p_over_at_proj: float) -> float:
    """CDF of shifted gamma at x."""
    if x <= 0:
        return 0.0
    shape, scale = gamma_params_shifted(projection, cv, p_over_at_proj)
    return float(sp_stats.gamma.cdf(x, a=shape, scale=scale))


def gamma_crps_shifted(actual: float, projection: float, cv: float,
                       p_over_at_proj: float) -> float:
    """CRPS for shifted gamma — closed-form."""
    from scipy.special import beta as beta_fn
    shape, scale = gamma_params_shifted(projection, cv, p_over_at_proj)
    if shape <= 0 or scale <= 0:
        return abs(actual - projection)
    y = max(actual, 0.0)
    F_y = float(sp_stats.gamma.cdf(y, a=shape, scale=scale))
    F_y_plus1 = float(sp_stats.gamma.cdf(y, a=shape + 1, scale=scale))
    crps = (y * (2 * F_y - 1)
            - shape * scale * (2 * F_y_plus1 - 1)
            - scale / beta_fn(0.5, shape + 0.5))
    return abs(crps)


def p_over_line_gamma_shifted(line: float, projection: float, cv: float,
                              p_over_at_proj: float) -> float:
    """P(actual > line) from shifted gamma."""
    if projection <= 0:
        return 0.5
    return 1.0 - gamma_cdf_shifted(line, projection, cv, p_over_at_proj)


def negbin_cdf(x: float, mean: float, var: float) -> float:
    """CDF of NegBin at x (same as existing)."""
    if mean <= 0:
        return 0.0
    if var <= mean:
        return float(sp_stats.poisson.cdf(floor(x), mu=mean))
    r = max((mean ** 2) / (var - mean), 0.1)
    if r >= 500:
        return float(sp_stats.poisson.cdf(floor(x), mu=mean))
    p = r / (r + mean)
    return float(sp_stats.nbinom.cdf(floor(x), n=r, p=p))


def p_over_line_negbin(line: float, mean: float, var: float) -> float:
    """P(actual > line) from NegBin."""
    if mean <= 0:
        return 0.5
    k = floor(line)
    if line == int(line):
        return 1.0 - negbin_cdf(line, mean, var)
    else:
        return 1.0 - negbin_cdf(k, mean, var)


def negbin_crps_single(actual: float, mean: float, var: float) -> float:
    """CRPS for NegBin via vectorized discrete sum."""
    if mean <= 0:
        return abs(actual)
    actual_int = int(round(actual))
    upper = max(actual_int * 3, int(mean * 3), 20)
    ks = np.arange(upper + 1)
    indicators = (ks >= actual_int).astype(float)

    if var <= mean:
        cdf_vals = sp_stats.poisson.cdf(ks, mu=mean)
    else:
        r = max((mean ** 2) / (var - mean), 0.1)
        p_param = r / (r + mean)
        cdf_vals = sp_stats.nbinom.cdf(ks, n=r, p=p_param)

    return float(np.sum((cdf_vals - indicators) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# Negative mass under lognormal at new CVs (diagnostic)
# ══════════════════════════════════════════════════════════════════════════════

def report_lognormal_negative_mass():
    """Report how much probability lognormal puts below 0 at various CVs.

    Lognormal is always ≥ 0 by construction, so the negative-mass problem
    is about the NORMAL approximation people sometimes mentally use.
    But the real issue is the left-tail shape: at high CV, lognormal has
    extreme right skew that misrepresents the left tail for player props.
    Gamma handles this better.
    """
    print("\n  Lognormal vs Gamma left-tail comparison:")
    print(f"  {'Mean':>6s} {'CV':>5s} {'P(X<0) LN':>10s} {'P(X<5) LN':>10s} "
          f"{'P(X<5) Gam':>11s} {'Skew LN':>8s} {'Skew Gam':>9s}")
    for mean_val in [20, 40, 80, 200]:
        for cv in [0.45, 0.65, 0.76, 0.94]:
            # Lognormal
            sigma_sq = log(1 + cv ** 2)
            sigma = sqrt(sigma_sq)
            mu = log(mean_val) - sigma_sq / 2
            p_neg_ln = 0.0  # lognormal is always >= 0
            p_lt5_ln = float(sp_stats.lognorm.cdf(5, s=sigma, scale=exp(mu)))
            skew_ln = float((exp(sigma_sq) + 2) * sqrt(exp(sigma_sq) - 1))

            # Gamma
            shape, scale = gamma_params(mean_val, cv)
            p_lt5_gam = float(sp_stats.gamma.cdf(5, a=shape, scale=scale))
            skew_gam = 2.0 / sqrt(shape)

            print(f"  {mean_val:6.0f} {cv:5.2f} {p_neg_ln:10.4f} {p_lt5_ln:10.4f} "
                  f"{p_lt5_gam:11.4f} {skew_ln:8.2f} {skew_gam:9.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading: player-game stats from nflverse PBP
# ══════════════════════════════════════════════════════════════════════════════

def _load_player_game_stats(seasons: list[int]) -> list[dict]:
    """Load per-player per-game stats from nflverse PBP.

    Returns list of dicts with keys:
        season, week, game_id, player_name, team, position,
        stat_type, stat_value
    """
    import nfl_data_py as nfl

    pbp = nfl.import_pbp_data(seasons)
    rosters = nfl.import_weekly_rosters(seasons)

    # Build player→position lookup (most common position across weeks)
    pos_lookup = {}
    if rosters is not None and len(rosters) > 0:
        for _, row in rosters.drop_duplicates("player_id").iterrows():
            name = row.get("player_name", "")
            pos = row.get("position", "")
            if name and pos:
                pos_lookup[name] = pos

    records = []
    roster_id_pos = {}

    # Build receiver ID → position lookup from PBP receiver_player_id + roster
    recv_pos = {}
    if rosters is not None and len(rosters) > 0:
        for _, row in rosters.drop_duplicates("player_id").iterrows():
            pid = row.get("player_id", "")
            pos = row.get("position", "")
            if pid and pos:
                roster_id_pos[pid] = pos
        # Map receiver_player_id → position
        recv_ids = pbp[pbp["receiver_player_id"].notna()][
            ["receiver_player_name", "receiver_player_id"]
        ].drop_duplicates("receiver_player_id")
        for _, row in recv_ids.iterrows():
            rid = row["receiver_player_id"]
            name = row["receiver_player_name"]
            pos = roster_id_pos.get(rid, pos_lookup.get(name, "WR"))
            recv_pos[rid] = pos
            recv_pos[name] = pos  # fallback by name

    # Receiving
    recv = pbp[pbp["receiver_player_name"].notna() & (pbp["pass_attempt"] == 1)].copy()
    recv_agg = (recv.groupby(["season", "week", "game_id",
                              "receiver_player_name", "receiver_player_id",
                              "posteam"])
                .agg(
                    rec_yds=("yards_gained", "sum"),
                    receptions=("complete_pass", "sum"),
                    rec_tds=("pass_touchdown", "sum"),
                    targets=("pass_attempt", "sum"),
                ).reset_index())

    for _, row in recv_agg.iterrows():
        name = row["receiver_player_name"]
        rid = row["receiver_player_id"]
        team = row["posteam"]
        pos = recv_pos.get(rid, recv_pos.get(name, pos_lookup.get(name, "WR")))
        base = {
            "season": int(row["season"]), "week": int(row["week"]),
            "game_id": row["game_id"], "player_name": name,
            "team": team, "position": pos, "targets": int(row["targets"]),
        }
        records.append({**base, "stat_type": "rec_yds",
                        "stat_value": float(row["rec_yds"])})
        records.append({**base, "stat_type": "receptions",
                        "stat_value": float(row["receptions"])})

    # Build rusher ID → position lookup
    rush_pos = {}
    if rosters is not None and len(rosters) > 0:
        rush_ids = pbp[pbp["rusher_player_id"].notna()][
            ["rusher_player_name", "rusher_player_id"]
        ].drop_duplicates("rusher_player_id")
        for _, row in rush_ids.iterrows():
            rid = row["rusher_player_id"]
            name = row["rusher_player_name"]
            pos = roster_id_pos.get(rid, pos_lookup.get(name, "RB"))
            rush_pos[rid] = pos
            rush_pos[name] = pos

    # Rushing
    rush = pbp[pbp["rusher_player_name"].notna() & (pbp["rush_attempt"] == 1)].copy()
    rush_agg = (rush.groupby(["season", "week", "game_id",
                              "rusher_player_name", "rusher_player_id",
                              "posteam"])
                .agg(rush_yds=("yards_gained", "sum"),
                     rush_att=("rush_attempt", "sum"))
                .reset_index())

    for _, row in rush_agg.iterrows():
        name = row["rusher_player_name"]
        rid = row["rusher_player_id"]
        pos = rush_pos.get(rid, rush_pos.get(name, pos_lookup.get(name, "RB")))
        records.append({
            "season": int(row["season"]), "week": int(row["week"]),
            "game_id": row["game_id"], "player_name": name,
            "team": row["posteam"], "position": pos,
            "stat_type": "rush_yds", "stat_value": float(row["rush_yds"]),
        })

    # Passing
    passing = pbp[pbp["passer_player_name"].notna() & (pbp["pass_attempt"] == 1)].copy()
    pass_agg = (passing.groupby(["season", "week", "game_id",
                                 "passer_player_name", "posteam"])
                .agg(
                    pass_yds=("yards_gained", "sum"),
                    pass_tds=("pass_touchdown", "sum"),
                    pass_ints=("interception", "sum"),
                    attempts=("pass_attempt", "sum"),
                ).reset_index())

    for _, row in pass_agg.iterrows():
        name = row["passer_player_name"]
        pos = "QB"
        base = {
            "season": int(row["season"]), "week": int(row["week"]),
            "game_id": row["game_id"], "player_name": name,
            "team": row["posteam"], "position": pos,
        }
        records.append({**base, "stat_type": "pass_yds",
                        "stat_value": float(row["pass_yds"])})
        records.append({**base, "stat_type": "pass_tds",
                        "stat_value": float(row["pass_tds"])})

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Dispersion fitting: σ = a · μ^b, CRPS-optimised walk-forward
# ══════════════════════════════════════════════════════════════════════════════

_YARDAGE_STATS = {"pass_yds", "rush_yds", "rec_yds"}
_COUNT_STATS = {"receptions", "pass_tds", "pass_ints"}

# Minimum games to include a player in dispersion fitting
_MIN_GAMES_FIT = 4
# Minimum projected mean to include (avoid noise from low-volume)
_MIN_MEAN_FIT = {"pass_yds": 80, "rush_yds": 10, "rec_yds": 8,
                 "receptions": 1.5, "pass_tds": 0.3, "pass_ints": 0.1}


def _compute_player_dispersion(records: list[dict]) -> list[dict]:
    """Compute per-player (mean, SD, CV) from game-level records.

    Returns list of {stat_type, position, player_name, mean, sd, cv, n_games}.
    """
    # Group: (player, stat, position, season) → [values]
    groups = defaultdict(list)
    for r in records:
        key = (r["player_name"], r["stat_type"], r["position"], r["season"])
        groups[key].append(r["stat_value"])

    results = []
    for (player, stat, pos, season), vals in groups.items():
        if len(vals) < _MIN_GAMES_FIT:
            continue
        m = np.mean(vals)
        min_mean = _MIN_MEAN_FIT.get(stat, 5)
        if m < min_mean:
            continue
        s = np.std(vals, ddof=1)
        cv = s / m if m > 0 else 0
        results.append({
            "player_name": player, "stat_type": stat, "position": pos,
            "season": season, "mean": float(m), "sd": float(s),
            "cv": float(cv), "n_games": len(vals),
        })
    return results


def _fit_power_law(dispersions: list[dict], stat: str, pos: str
                   ) -> dict | None:
    """Fit σ = a · μ^b for a given (stat, position) group.

    Returns {"a": float, "b": float, "n": int, "cv_at_50": float}
    or None if insufficient data.
    """
    subset = [d for d in dispersions
              if d["stat_type"] == stat and d["position"] == pos]
    if len(subset) < 5:
        return None

    # Log-linear regression: log(σ) = log(a) + b·log(μ)
    log_means = np.array([log(d["mean"]) for d in subset])
    log_sds = np.array([log(max(d["sd"], 0.01)) for d in subset])

    # OLS
    X = np.column_stack([np.ones(len(log_means)), log_means])
    beta = np.linalg.lstsq(X, log_sds, rcond=None)[0]
    log_a, b = beta[0], beta[1]
    a = exp(log_a)

    # CV at representative means
    ref_means = {"pass_yds": 230, "rush_yds": 60, "rec_yds": 50,
                 "receptions": 4, "pass_tds": 1.5, "pass_ints": 0.8}
    ref = ref_means.get(stat, 50)
    cv_at_ref = a * (ref ** b) / ref

    return {"a": round(a, 6), "b": round(b, 4), "n": len(subset),
            "cv_at_ref": round(cv_at_ref, 3), "ref_mean": ref}


def _crps_for_cv(cv: float, mean: float, actual: float,
                 stat_type: str) -> float:
    """CRPS for a single prediction given a CV value."""
    if stat_type in _YARDAGE_STATS:
        return gamma_crps_single(actual, mean, max(cv, 0.05))
    else:
        var = (mean * max(cv, 0.05)) ** 2
        return negbin_crps_single(actual, mean, var)


def fit_dispersion_model(seasons: list[int] = None,
                         walk_forward: bool = True) -> dict:
    """
    Fit dispersion parameters by CRPS minimisation.

    Walk-forward: for each season-week, train on all prior data,
    predict current week. Optimise the (a, b) power-law parameters
    to minimise total CRPS across the validation set.

    Returns fitted {(stat, pos): {"a", "b", "crps_before", "crps_after"}}
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    print(f"Loading nflverse PBP for {seasons}...")
    records = _load_player_game_stats(seasons)
    print(f"  {len(records)} player-game records loaded")

    # Compute per-player dispersion from ALL data (for initial power-law fit)
    dispersions = _compute_player_dispersion(records)
    print(f"  {len(dispersions)} player-season dispersion estimates")

    # Fit power law per (stat, position)
    stat_pos_combos = set((d["stat_type"], d["position"]) for d in dispersions)
    fitted = {}

    for stat, pos in sorted(stat_pos_combos):
        pl = _fit_power_law(dispersions, stat, pos)
        if pl is None:
            continue

        # Now CRPS-optimise: scale the power-law CV by a single multiplier
        # to minimise CRPS on walk-forward validation data.
        #
        # Walk-forward: for each (season, week), use player's mean from
        # prior games as projection, then score against actual.
        val_data = _build_walk_forward_set(records, stat, pos)
        if len(val_data) < 20:
            fitted[(stat, pos)] = {**pl, "crps_multiplier": 1.0}
            continue

        # Cap validation set for speed (random sample if too large)
        if len(val_data) > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(val_data), 2000, replace=False)
            val_data = [val_data[i] for i in idx]

        a_base, b_base = pl["a"], pl["b"]

        def crps_objective(mult):
            total = 0.0
            for v in val_data:
                cv = (a_base * mult) * (v["proj_mean"] ** b_base) / v["proj_mean"]
                cv = max(cv, 0.05)
                total += _crps_for_cv(cv, v["proj_mean"], v["actual"], stat)
            return total / len(val_data)

        # Baseline CRPS at mult=1.0
        crps_before = crps_objective(1.0)

        # Optimise multiplier in [0.3, 5.0]
        result = minimize_scalar(crps_objective, bounds=(0.3, 5.0),
                                 method="bounded")
        opt_mult = result.x
        crps_after = result.fun

        fitted[(stat, pos)] = {
            **pl,
            "a": round(a_base * opt_mult, 6),
            "crps_multiplier": round(opt_mult, 3),
            "crps_before": round(crps_before, 3),
            "crps_after": round(crps_after, 3),
        }

    # Store globally
    global DISPERSION_PARAMS
    DISPERSION_PARAMS = fitted

    # Cache to disk
    _save_params(fitted)

    return fitted


def _build_walk_forward_set(records: list[dict], stat: str, pos: str
                            ) -> list[dict]:
    """Build walk-forward validation set for CRPS optimisation.

    For each player-week, projection = mean of their prior games that season.
    Actual = stat_value for that week.
    """
    # Group by (player, season) → sorted list of (week, value)
    player_seasons = defaultdict(list)
    for r in records:
        if r["stat_type"] == stat and r["position"] == pos:
            player_seasons[(r["player_name"], r["season"])].append(
                (r["week"], r["stat_value"]))

    val_data = []
    for (player, season), games in player_seasons.items():
        games.sort(key=lambda x: x[0])
        # Need at least 4 prior games before making predictions
        for i in range(4, len(games)):
            prior_vals = [g[1] for g in games[:i]]
            proj_mean = np.mean(prior_vals)
            min_mean = _MIN_MEAN_FIT.get(stat, 5)
            if proj_mean < min_mean:
                continue
            val_data.append({
                "player": player, "season": season,
                "week": games[i][0],
                "proj_mean": float(proj_mean),
                "actual": float(games[i][1]),
            })

    return val_data


# ══════════════════════════════════════════════════════════════════════════════
# Two-stage dispersion fit:
#   Stage 1: shape (b exponent) from nflverse  — already fitted above
#   Stage 2: level (a scalar) from 2026 consensus predictions, LOO CRPS
# ══════════════════════════════════════════════════════════════════════════════

# Shape floor: gamma shape must be > 1 to avoid mode-at-zero / median << mean.
# shape = 1/CV²  →  shape > 1  ⟺  CV < 1.0
_CV_CAP = 1.0  # hard cap: shape ≥ 1


def _load_2026_graded() -> list[dict]:
    """Load deduplicated 2026 graded predictions from fantasy_forward_preds.

    Returns one record per (player, week, stat, position) with:
        consensus_mean, actual_value, week, stat_type, position
    """
    import sqlite3
    db_path = os.path.join(_DATA_DIR, "historical.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT player_name, week, stat_type, position,
               consensus_mean, actual_value
        FROM fantasy_forward_preds
        WHERE season = 2026
          AND method = 'distributional'
          AND actual_value IS NOT NULL
          AND data_error IS NULL
        GROUP BY player_name, week, stat_type, position
        ORDER BY stat_type, position, week
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _compute_cv_capped(a: float, b: float, proj_mean: float) -> float:
    """Compute CV from power-law, capped so gamma shape ≥ 1."""
    if proj_mean <= 0:
        return 0.50
    cv = a * (proj_mean ** (b - 1))
    return max(0.05, min(cv, _CV_CAP))


def fit_two_stage(stage1_params: dict = None) -> dict:
    """Two-stage dispersion fit.

    Stage 1 (b exponents) — from nflverse, already stored in stage1_params
    or loaded from cache.

    Stage 2 — for each (stat, pos), fit a single scalar 'a' to minimise
    LOO CRPS on 2026 W2-W5 consensus predictions. The b exponent is fixed
    from Stage 1. CV is capped at _CV_CAP (shape ≥ 1).

    Returns dict with fitted params and LOO CRPS report.
    """
    # Load Stage 1 params
    if stage1_params is None:
        load_cached_params()
        stage1_params = dict(DISPERSION_PARAMS)

    if not stage1_params:
        print("ERROR: No Stage 1 params available. Run fit_dispersion_model() first.")
        return {}

    # Load 2026 graded predictions
    graded = _load_2026_graded()
    print(f"  Loaded {len(graded)} unique graded predictions from 2026")

    # Group by (stat, pos)
    by_sp = defaultdict(list)
    for r in graded:
        by_sp[(r["stat_type"], r["position"])].append(r)

    # Get available weeks per group
    for sp, recs in by_sp.items():
        weeks = sorted(set(r["week"] for r in recs))
        ns = {w: sum(1 for r in recs if r["week"] == w) for w in weeks}

    results = {}

    print(f"\n{'='*90}")
    print("TWO-STAGE DISPERSION FIT")
    print(f"  Stage 1: b exponents from nflverse 2023-2025")
    print(f"  Stage 2: a level from 2026 consensus predictions (LOO CRPS)")
    print(f"  Shape floor: gamma shape ≥ 1 (CV ≤ {_CV_CAP:.2f})")
    print(f"{'='*90}")

    print(f"\n  {'Stat':12s} {'Pos':5s} {'b_s1':>6s} {'a_s1':>10s} {'a_s2':>10s} "
          f"{'ratio':>6s} {'CV@ref':>7s} {'shape':>6s} "
          f"{'LOO_CRPS':>9s} {'N':>4s} {'folds':>5s}")
    print("  " + "-" * 95)

    ref_means = {"pass_yds": 230, "rush_yds": 60, "rec_yds": 50,
                 "receptions": 4, "pass_tds": 1.5, "pass_ints": 0.8}

    for (stat, pos) in sorted(stage1_params.keys()):
        s1 = stage1_params[(stat, pos)]
        b_fixed = s1["b"]
        a_s1 = s1["a"]

        # Get 2026 data for this combo
        recs = by_sp.get((stat, pos), [])

        # If too few, try pooling across positions for this stat
        if len(recs) < 10:
            recs = []
            for (s, p), r_list in by_sp.items():
                if s == stat:
                    recs.extend(r_list)

        if len(recs) < 10:
            print(f"  {stat:12s} {pos:5s}  — skipped (N={len(recs)} < 10)")
            results[(stat, pos)] = {**s1, "stage": "s1_only"}
            continue

        # Available weeks with enough data
        weeks = sorted(set(r["week"] for r in recs))
        week_counts = {w: sum(1 for r in recs if r["week"] == w) for w in weeks}
        # Only include weeks with ≥5 observations
        usable_weeks = [w for w in weeks if week_counts[w] >= 5]

        if len(usable_weeks) < 2:
            # Can't do LOO with < 2 folds — fit on all data, no LOO score
            a_fitted = _fit_a_on_set(recs, b_fixed, stat)
            ref = ref_means.get(stat, 50)
            cv_at_ref = min(a_fitted * (ref ** (b_fixed - 1)), _CV_CAP)
            shape_at_ref = 1.0 / (cv_at_ref ** 2)
            ratio = a_fitted / a_s1 if a_s1 > 0 else 0
            print(f"  {stat:12s} {pos:5s} {b_fixed:6.3f} {a_s1:10.4f} "
                  f"{a_fitted:10.4f} {ratio:6.3f} {cv_at_ref:7.3f} "
                  f"{shape_at_ref:6.2f}     —    {len(recs):4d}    1f")
            results[(stat, pos)] = {
                **s1, "a": round(a_fitted, 6), "stage": "s2_no_loo",
                "a_s1": a_s1, "ratio_s2_s1": round(ratio, 4),
                "cv_at_ref": round(cv_at_ref, 3),
            }
            continue

        # LOO across usable weeks
        loo_crps_scores = []
        loo_a_values = []

        for held_out_week in usable_weeks:
            train = [r for r in recs if r["week"] != held_out_week]
            test = [r for r in recs if r["week"] == held_out_week]

            if len(train) < 5 or len(test) < 3:
                continue

            # Fit a on train set
            a_train = _fit_a_on_set(train, b_fixed, stat)
            loo_a_values.append(a_train)

            # Score on test set
            test_crps = _crps_on_set(test, a_train, b_fixed, stat)
            loo_crps_scores.append(test_crps)

        if not loo_crps_scores:
            print(f"  {stat:12s} {pos:5s}  — LOO failed")
            results[(stat, pos)] = {**s1, "stage": "s1_only"}
            continue

        loo_crps = np.mean(loo_crps_scores)

        # Final fit on all data
        a_fitted = _fit_a_on_set(recs, b_fixed, stat)
        ref = ref_means.get(stat, 50)
        cv_at_ref = min(a_fitted * (ref ** (b_fixed - 1)), _CV_CAP)
        shape_at_ref = 1.0 / (cv_at_ref ** 2)
        ratio = a_fitted / a_s1 if a_s1 > 0 else 0

        print(f"  {stat:12s} {pos:5s} {b_fixed:6.3f} {a_s1:10.4f} "
              f"{a_fitted:10.4f} {ratio:6.3f} {cv_at_ref:7.3f} "
              f"{shape_at_ref:6.2f} {loo_crps:9.3f} {len(recs):4d} "
              f"{len(usable_weeks):4d}f")

        results[(stat, pos)] = {
            **s1, "a": round(a_fitted, 6), "b": b_fixed,
            "stage": "s2_loo",
            "a_s1": a_s1,
            "ratio_s2_s1": round(ratio, 4),
            "cv_at_ref": round(cv_at_ref, 3),
            "shape_at_ref": round(shape_at_ref, 2),
            "loo_crps": round(loo_crps, 3),
            "n_predictions": len(recs),
            "n_folds": len(usable_weeks),
            "loo_a_range": [round(min(loo_a_values), 4),
                            round(max(loo_a_values), 4)],
        }

    # Compare to 0.85 shrinkage guess
    print(f"\n  {'Comparison to 0.85 shrinkage:':40s}")
    print(f"  {'Stat':12s} {'Pos':5s} {'a_s1×0.85':>10s} {'a_s2':>10s} "
          f"{'ratio/0.85':>10s} {'verdict':>10s}")
    print("  " + "-" * 65)
    for (stat, pos), r in sorted(results.items()):
        if r.get("stage") == "s1_only":
            continue
        a_085 = r.get("a_s1", r["a"]) * 0.85
        a_s2 = r["a"]
        ratio = a_s2 / a_085 if a_085 > 0 else 0
        verdict = "wider" if ratio > 1.05 else "narrower" if ratio < 0.95 else "≈0.85"
        print(f"  {stat:12s} {pos:5s} {a_085:10.4f} {a_s2:10.4f} "
              f"{ratio:10.3f}  {verdict:>10s}")

    # Save v1.1 params
    v11_params = {}
    for (stat, pos), r in results.items():
        v11_params[(stat, pos)] = {
            "a": r["a"], "b": r["b"], "n": r.get("n", 0),
            "cv_at_ref": r.get("cv_at_ref", 0),
            "ref_mean": r.get("ref_mean", 50),
            "stage": r.get("stage", "s1_only"),
        }
    _save_params(v11_params, path=_V11_CACHE_PATH, version=MODEL_VERSION_V11)

    # Also load into DISPERSION_PARAMS for immediate use
    DISPERSION_PARAMS.clear()
    DISPERSION_PARAMS.update(v11_params)

    return results


def _fit_a_on_set(recs: list[dict], b_fixed: float,
                  stat: str) -> float:
    """Fit the a parameter to minimise CRPS on a set of predictions.

    CV = a · μ^(b-1), capped at _CV_CAP (shape ≥ 1).
    """
    def objective(log_a):
        a = exp(log_a)
        total = 0.0
        for r in recs:
            mu = r["consensus_mean"]
            if mu <= 0:
                continue
            cv = min(a * (mu ** (b_fixed - 1)), _CV_CAP)
            cv = max(cv, 0.05)
            actual = r["actual_value"]
            total += _crps_for_cv(cv, mu, actual, stat)
        return total / len(recs)

    # Search in log-space for stability; a typically in [0.1, 100]
    result = minimize_scalar(objective, bounds=(log(0.01), log(500)),
                             method="bounded")
    return exp(result.x)


def _crps_on_set(recs: list[dict], a: float, b: float,
                 stat: str) -> float:
    """Compute mean CRPS on a set using given a, b parameters."""
    total = 0.0
    n = 0
    for r in recs:
        mu = r["consensus_mean"]
        if mu <= 0:
            continue
        cv = min(a * (mu ** (b - 1)), _CV_CAP)
        cv = max(cv, 0.05)
        total += _crps_for_cv(cv, mu, r["actual_value"], stat)
        n += 1
    return total / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
# v1.2: fit base rates (location decoupling) with LOO
# ══════════════════════════════════════════════════════════════════════════════

def fit_v12_base_rates() -> dict:
    """Fit P(actual > projection) per stat×pos with expanding window.

    Uses all graded data as a growing window (no LOO). Reports 95%
    Clopper-Pearson CI per combo. Updated weekly — each call replaces
    the cached rates with the latest expanding-window estimate.

    Returns {(stat, pos): {"rate": shrunk_rate, "raw": raw_rate, "n": N,
                           "ci_lo": float, "ci_hi": float, ...}}.
    """
    import sqlite3
    db_path = os.path.join(_DATA_DIR, "historical.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT player_name, week, stat_type, position,
               consensus_mean, actual_value
        FROM fantasy_forward_preds
        WHERE season = 2026
          AND method = 'distributional'
          AND actual_value IS NOT NULL
          AND data_error IS NULL
        GROUP BY player_name, week, stat_type, position
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Group by (stat, pos) → all over indicators
    by_sp: dict[tuple, list] = defaultdict(list)
    by_sp_week: dict[tuple, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        stat, pos = r["stat_type"], r["position"]
        if stat not in ("pass_yds", "rec_yds", "rush_yds"):
            continue
        if not r["consensus_mean"] or r["consensus_mean"] <= 0:
            continue
        over = 1 if r["actual_value"] > r["consensus_mean"] else 0
        by_sp[(stat, pos)].append(over)
        by_sp_week[(stat, pos)][r["week"]].append(over)

    k = _V12_SHRINKAGE_K

    results = {}

    print(f"\n{'='*80}")
    print("V1.2 BASE RATE FITTING (expanding window)")
    print(f"  Shrinkage: k={k} toward pooled rate={_V12_POOLED_RATE:.4f}")
    print(f"{'='*80}")
    print(f"\n  {'Stat':12s} {'Pos':5s} {'N':>5s} {'raw':>6s} {'shrunk':>7s} "
          f"{'95% CI':>16s} {'CI_hw':>6s} {'cv':>5s} "
          f"{'shape':>6s} {'shift':>7s}")
    print("  " + "-" * 90)

    for (stat, pos) in sorted(by_sp.keys()):
        indicators = by_sp[(stat, pos)]
        total_n = len(indicators)
        total_overs = sum(indicators)
        raw_rate = total_overs / total_n if total_n > 0 else 0.5
        shrunk_rate = (total_n * raw_rate + k * _V12_POOLED_RATE) / (total_n + k)

        # 95% Clopper-Pearson CI on raw rate
        alpha = 0.05
        if total_overs == 0 or total_overs == total_n or total_n < 5:
            # Degenerate case: CI undefined, use wide fallback
            ci_lo_shrunk = _V12_POOLED_RATE - 0.10
            ci_hi_shrunk = _V12_POOLED_RATE + 0.10
        else:
            ci_lo = float(sp_stats.beta.ppf(alpha / 2, total_overs,
                                            total_n - total_overs + 1))
            ci_hi = float(sp_stats.beta.ppf(1 - alpha / 2, total_overs + 1,
                                            total_n - total_overs))
            ci_lo_shrunk = (total_n * ci_lo + k * _V12_POOLED_RATE) / (total_n + k)
            ci_hi_shrunk = (total_n * ci_hi + k * _V12_POOLED_RATE) / (total_n + k)
        ci_halfwidth = (ci_hi_shrunk - ci_lo_shrunk) / 2

        # CV from v1.1
        cv = _V11_PIT_CV_BY_POS.get((stat, pos), _V11_PIT_CV.get(stat, 0.50))
        shape = 1.0 / (cv ** 2)
        gamma_p_ov = 1.0 - float(sp_stats.gamma.cdf(shape, a=shape))
        shift = shrunk_rate - gamma_p_ov

        ci_str = f"[{ci_lo_shrunk:.3f},{ci_hi_shrunk:.3f}]"
        print(f"  {stat:12s} {pos:5s} {total_n:5d} {raw_rate:6.3f} {shrunk_rate:7.3f} "
              f"{ci_str:>16s} {ci_halfwidth:6.3f} {cv:5.2f} "
              f"{shape:6.2f} {shift:+7.3f}")

        # Per-week expanding window trace (diagnostic)
        weeks_data = by_sp_week[(stat, pos)]
        all_weeks = sorted(weeks_data.keys())
        expanding = []
        cum_n = 0
        cum_overs = 0
        for w in all_weeks:
            cum_n += len(weeks_data[w])
            cum_overs += sum(weeks_data[w])
            exp_rate = (cum_n * (cum_overs / cum_n) + k * _V12_POOLED_RATE) / (cum_n + k)
            expanding.append({"week": w, "n": cum_n, "rate": round(exp_rate, 4)})

        results[(stat, pos)] = {
            "rate": round(shrunk_rate, 4),
            "raw": round(raw_rate, 4),
            "n": total_n,
            "ci_lo": round(ci_lo_shrunk, 4),
            "ci_hi": round(ci_hi_shrunk, 4),
            "ci_halfwidth": round(ci_halfwidth, 4),
            "expanding_trace": expanding,
            "cv": cv,
            "shift_pp": round(shift * 100, 1),
        }

    # Save to global and disk
    global _V12_BASE_RATES, _V12_BASE_RATE_CI, _V12_BASE_RATE_BOUNDS
    _V12_BASE_RATES = {k: v["rate"] for k, v in results.items()}
    _V12_BASE_RATE_CI = {k: v["ci_halfwidth"] for k, v in results.items()}
    _V12_BASE_RATE_BOUNDS = {k: (v["ci_lo"], v["ci_hi"]) for k, v in results.items()}

    serializable = {f"{s}|{p}": v for (s, p), v in results.items()}
    with open(_V12_CACHE_PATH, "w") as f:
        json.dump({"model_version": "v1.2_expanding",
                   "pooled_rate": _V12_POOLED_RATE,
                   "shrinkage_k": k,
                   "rates": serializable}, f, indent=2)
    print(f"\n  Saved to {_V12_CACHE_PATH}")

    return results


def load_v12_base_rates() -> bool:
    """Load cached v1.2 base rates. Returns True if successful."""
    global _V12_BASE_RATES, _V12_BASE_RATE_CI, _V12_BASE_RATE_BOUNDS
    if not os.path.exists(_V12_CACHE_PATH):
        return False
    try:
        with open(_V12_CACHE_PATH) as f:
            data = json.load(f)
        for key, val in data.get("rates", {}).items():
            stat, pos = key.split("|")
            _V12_BASE_RATES[(stat, pos)] = val["rate"]
            if "ci_halfwidth" in val:
                _V12_BASE_RATE_CI[(stat, pos)] = val["ci_halfwidth"]
            if "ci_lo" in val and "ci_hi" in val:
                _V12_BASE_RATE_BOUNDS[(stat, pos)] = (val["ci_lo"], val["ci_hi"])
        return True
    except Exception as exc:
        logger.warning("Failed to load v1.2 base rates: %s", exc)
        return False


def get_v12_base_rate(stat_type: str, position: str) -> float:
    """Get the shrunk P(actual > projection) for v1.2."""
    rate = _V12_BASE_RATES.get((stat_type, position))
    if rate is not None:
        return rate
    # Fallback: pooled yardage rate
    return _V12_POOLED_RATE


def get_v12_base_rate_ci(stat_type: str, position: str) -> float:
    """Get the CI halfwidth for the location parameter (shrunk rate).

    Returns 0.0 if CI not loaded (fallback: no flagging).
    """
    return _V12_BASE_RATE_CI.get((stat_type, position), 0.0)


def check_location_uncertain(line: float, projection: float, cv: float,
                             stat_type: str, position: str,
                             devigged_implied: float,
                             edge_threshold: float = 0.0) -> bool:
    """Check if a pick's edge is uncertain under base-rate CI bounds.

    Recomputes P(over line) at the CI low and high base rates. Returns
    True (uncertain) if:
      - The edge changes sign across the CI range, OR
      - The edge drops below edge_threshold at either bound.

    Returns False (not uncertain) if bounds aren't loaded.
    """
    bounds = _V12_BASE_RATE_BOUNDS.get((stat_type, position))
    if not bounds:
        return False

    ci_lo, ci_hi = bounds

    # Compute P(over) at lo and hi base rates
    p_ov_lo = p_over_line_gamma_shifted(line, projection, cv, ci_lo)
    p_ov_hi = p_over_line_gamma_shifted(line, projection, cv, ci_hi)

    # Compute edges at each bound
    edge_lo = p_ov_lo - devigged_implied
    edge_hi = p_ov_hi - devigged_implied

    # Direction at center
    rate_center = _V12_BASE_RATES.get((stat_type, position), _V12_POOLED_RATE)
    p_ov_center = p_over_line_gamma_shifted(line, projection, cv, rate_center)
    edge_center = p_ov_center - devigged_implied

    # Check: does the edge change sign?
    if edge_center > 0:
        # Currently "over" pick — uncertain if lo bound flips to negative
        if edge_lo < -edge_threshold:
            return True
    else:
        # Currently "under" pick — uncertain if hi bound flips to positive
        if edge_hi > edge_threshold:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# Public API: get dispersion for a prediction
# ══════════════════════════════════════════════════════════════════════════════

def get_interim_cv(stat_type: str, position: str,
                   projected_mean: float,
                   use_v11: bool = False,
                   use_v12: bool = False) -> float:
    """Get the CV for a prediction.

    If use_v11 or use_v12, returns PIT-calibrated flat CV per stat.
    Otherwise, uses power-law from DISPERSION_PARAMS (v1).
    """
    if use_v11 or use_v12:
        cv = _V11_PIT_CV_BY_POS.get((stat_type, position))
        if cv is None:
            cv = _V11_PIT_CV.get(stat_type)
        if cv is not None:
            return cv
        # Fall through to power-law for stats not in _V11_PIT_CV

    params = DISPERSION_PARAMS.get((stat_type, position))
    if params is None:
        # Try stat-only (aggregate across positions)
        for key, val in DISPERSION_PARAMS.items():
            if key[0] == stat_type:
                params = val
                break
    if params is None:
        return _FALLBACK_CV.get(stat_type, 0.50)

    if projected_mean <= 0:
        return _FALLBACK_CV.get(stat_type, 0.50)

    a, b = params["a"], params["b"]
    cv = a * (projected_mean ** (b - 1))
    # Clamp: floor 0.10, cap at _CV_CAP (shape ≥ 1)
    cv = max(0.10, min(cv, _CV_CAP))
    return cv


def get_interim_dispersion(stat_type: str, position: str,
                           projected_mean: float) -> float:
    """Get variance for the interim model. Returns variance (σ²)."""
    cv = get_interim_cv(stat_type, position, projected_mean)
    return (projected_mean * cv) ** 2


def p_over_interim(line: float, mean: float, stat_type: str,
                   position: str) -> float:
    """P(actual > line) using the interim dispersion model.

    Gamma for yardage, NegBin for counts.
    """
    cv = get_interim_cv(stat_type, position, mean)

    if stat_type in _YARDAGE_STATS:
        return p_over_line_gamma(line, mean, cv)
    else:
        var = (mean * cv) ** 2
        return p_over_line_negbin(line, mean, var)


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def _save_params(params: dict, path: str = None, version: str = None):
    """Cache fitted params to disk."""
    if path is None:
        path = _CACHE_PATH
    if version is None:
        version = MODEL_VERSION
    serializable = {}
    for (stat, pos), val in params.items():
        serializable[f"{stat}|{pos}"] = val
    with open(path, "w") as f:
        json.dump({"model_version": version,
                   "switch_date": MODEL_SWITCH_DATE,
                   "params": serializable}, f, indent=2)
    print(f"  Saved params to {path}")


def load_cached_params(path: str = None) -> bool:
    """Load cached params from disk. Returns True if successful."""
    global DISPERSION_PARAMS
    if path is None:
        path = _CACHE_PATH
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        for key, val in data.get("params", {}).items():
            stat, pos = key.split("|")
            DISPERSION_PARAMS[(stat, pos)] = val
        return True
    except Exception as exc:
        logger.warning("Failed to load cached dispersion params: %s", exc)
        return False


def load_v11_params() -> bool:
    """Load v1.1 two-stage params. Returns True if successful."""
    return load_cached_params(path=_V11_CACHE_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# CLI: fit + report
# ══════════════════════════════════════════════════════════════════════════════

def print_fit_report(fitted: dict):
    """Print the fitted dispersion parameters."""
    print("\n" + "=" * 80)
    print(f"INTERIM DISPERSION MODEL ({MODEL_VERSION})")
    print(f"Switch date: {MODEL_SWITCH_DATE}")
    print("=" * 80)

    print(f"\n  {'Stat':12s} {'Pos':5s} {'a':>10s} {'b':>6s} {'N':>4s} "
          f"{'CV@ref':>7s} {'ref_μ':>6s} {'mult':>5s} "
          f"{'CRPS_pre':>9s} {'CRPS_post':>10s} {'Δ':>7s}")
    print("  " + "-" * 90)

    for (stat, pos), d in sorted(fitted.items()):
        crps_pre = d.get("crps_before", "")
        crps_post = d.get("crps_after", "")
        delta = ""
        if isinstance(crps_pre, float) and isinstance(crps_post, float):
            delta = f"{crps_post - crps_pre:+7.3f}"
            crps_pre = f"{crps_pre:9.3f}"
            crps_post = f"{crps_post:10.3f}"

        cv_str = f"{d.get('cv_at_ref', 0):7.3f}"
        # Recompute CV at ref after multiplier
        ref = d.get("ref_mean", 50)
        cv_final = d["a"] * (ref ** (d["b"] - 1))
        cv_str = f"{cv_final:7.3f}"

        print(f"  {stat:12s} {pos:5s} {d['a']:10.6f} {d['b']:6.3f} "
              f"{d['n']:4d} {cv_str:>7s} {ref:6.0f} "
              f"{d.get('crps_multiplier', 1.0):5.2f} "
              f"{crps_pre:>9s} {crps_post:>10s} {delta:>7s}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "two-stage":
        # Stage 1: load nflverse-fitted b exponents from cache
        print("Stage 1: Loading nflverse-fitted b exponents from cache...")
        load_cached_params()
        s1 = dict(DISPERSION_PARAMS)
        print(f"  {len(s1)} (stat, pos) combos loaded")
        # Stage 2: fit a level from 2026 consensus predictions
        results = fit_two_stage(stage1_params=s1)
    else:
        fitted = fit_dispersion_model(seasons=[2023, 2024, 2025])
        print_fit_report(fitted)
        print("\n")
        report_lognormal_negative_mass()
