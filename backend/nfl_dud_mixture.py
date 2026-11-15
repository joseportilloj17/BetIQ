"""
nfl_dud_mixture.py — Diagnose tail overconfidence in v1.2 gamma model.

Hypothesis: gamma with shape > 1 has almost no mass near zero, but real
outcomes include a "dud" component (injury exit, blowout benching, snap
collapse, coverage shutdown).

Steps:
  1. Break 0.8-1.0 bins by stat × position × threshold distance
  2. From 2023-2025 nflverse + snap counts: dud rate by position
  3. Fit mixture: (1-π)·gamma + π·dud, π by position
  4. Re-score 0.8-1.0 bins; check PIT/central calibration preserved

Run:
    python nfl_dud_mixture.py
"""
from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Break 0.8-1.0 bins by stat × pos × threshold distance
# ══════════════════════════════════════════════════════════════════════════════

def _load_graded_v12() -> list[dict]:
    """Load graded predictions, re-score with v1.2."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT player_name, position, stat_type, stat_category,
               consensus_mean, prop_line, actual_value, season, week,
               devigged_implied
        FROM fantasy_forward_preds
        WHERE actual_value IS NOT NULL
          AND method LIKE '%distributional%'
          AND method NOT LIKE 'v0_capped%'
          AND stat_category = 'yardage'
          AND data_error IS NULL
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def break_tail_bins():
    """Break 0.8-1.0 tail into sub-bins by stat × pos × distance."""
    from nfl_interim_dispersion import (
        get_interim_cv, load_v12_base_rates, get_v12_base_rate,
        p_over_line_gamma_shifted,
    )
    load_v12_base_rates()

    preds = _load_graded_v12()

    print("=" * 80)
    print("TAIL BIN BREAKDOWN (v1.2 P(over) in [0.80, 1.0])")
    print("=" * 80)

    # Re-score each prediction with v1.2
    tail_preds = []
    for p in preds:
        mean = p["consensus_mean"]
        stat = p["stat_type"]
        pos = p["position"]
        line = p["prop_line"]
        if not mean or mean <= 0 or not line:
            continue

        cv = get_interim_cv(stat, pos, mean, use_v12=True)
        br = get_v12_base_rate(stat, pos)
        p_ov = p_over_line_gamma_shifted(line, mean, cv, br)

        if p_ov >= 0.80 or p_ov <= 0.20:
            actual = p["actual_value"]
            hit = 1.0 if actual > line else (0.5 if actual == line else 0.0)
            # Threshold distance: how far line is from projection
            dist_pct = (mean - line) / mean if mean > 0 else 0
            tail_preds.append({
                **p,
                "v12_p_over": round(p_ov, 4),
                "hit": hit,
                "dist_pct": round(dist_pct, 3),
                "is_over_side": p_ov >= 0.80,
                "is_dud": actual < mean * 0.25,
            })

    if not tail_preds:
        print("  No tail predictions found.")
        return

    # ── By stat × pos ──
    groups = defaultdict(list)
    for tp in tail_preds:
        groups[(tp["stat_type"], tp["position"])].append(tp)

    print(f"\n  {'Stat':12s} {'Pos':5s} {'N':>5s} {'Side':>6s} "
          f"{'Mean P':>7s} {'Hit%':>6s} {'Gap_pp':>7s} "
          f"{'Dud%':>6s} {'Avg dist%':>9s}")
    print("  " + "-" * 75)

    for (stat, pos) in sorted(groups.keys()):
        items = groups[(stat, pos)]
        if len(items) < 5:
            continue
        over_side = [t for t in items if t["is_over_side"]]
        under_side = [t for t in items if not t["is_over_side"]]

        for side, label in [(over_side, "over"), (under_side, "under")]:
            if len(side) < 3:
                continue
            mean_p = np.mean([t["v12_p_over"] for t in side])
            hit_rate = np.mean([t["hit"] for t in side])
            if label == "under":
                # For under-side, compare 1-p_over to actual under-hit
                mean_p_adj = 1 - mean_p
                hit_rate_adj = 1 - hit_rate
                gap = (hit_rate_adj - mean_p_adj) * 100
            else:
                gap = (hit_rate - mean_p) * 100
            dud_rate = np.mean([t["is_dud"] for t in side])
            avg_dist = np.mean([t["dist_pct"] for t in side])
            print(f"  {stat:12s} {pos:5s} {len(side):5d} {label:>6s} "
                  f"{mean_p:7.3f} {hit_rate:6.3f} {gap:+7.1f} "
                  f"{dud_rate:6.3f} {avg_dist:9.3f}")

    # ── By distance bin ──
    print(f"\n  Breakdown by threshold distance from projection:")
    print(f"  {'dist_bin':12s} {'N':>5s} {'Mean P':>7s} {'Hit%':>6s} "
          f"{'Gap_pp':>7s} {'Dud%':>6s}")
    print("  " + "-" * 50)

    dist_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
    for lo, hi in dist_bins:
        subset = [t for t in tail_preds if t["is_over_side"]
                  and lo <= abs(t["dist_pct"]) < hi]
        if len(subset) < 5:
            continue
        mean_p = np.mean([t["v12_p_over"] for t in subset])
        hit_rate = np.mean([t["hit"] for t in subset])
        gap = (hit_rate - mean_p) * 100
        dud_rate = np.mean([t["is_dud"] for t in subset])
        print(f"  {lo:.1f}-{hi:.1f}      {len(subset):5d} {mean_p:7.3f} "
              f"{hit_rate:6.3f} {gap:+7.1f} {dud_rate:6.3f}")

    return tail_preds


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dud rates from 2023-2025 nflverse + snap counts
# ══════════════════════════════════════════════════════════════════════════════

def compute_dud_rates(seasons: list[int] = None) -> dict:
    """Compute dud rates by position from nflverse PBP + snap counts.

    A "dud" is defined as:
      - stat_value < 25% of player's rolling-mean projection, OR
      - offense_pct < 50% of player's season-average snap share

    Returns {(stat, pos): {"dud_rate": float, "n": int,
                           "snap_dud_rate": float, ...}}
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    import nfl_data_py as nfl

    print(f"\n  Loading nflverse PBP for {seasons}...")
    from nfl_interim_dispersion import _load_player_game_stats
    records = _load_player_game_stats(seasons)
    print(f"  {len(records)} player-game records")

    print(f"  Loading snap counts...")
    try:
        snaps = nfl.import_snap_counts(seasons)
        print(f"  {len(snaps)} snap count records")
    except Exception as exc:
        print(f"  Snap count load failed: {exc}")
        snaps = None

    # Build pfr_id-keyed snap lookup via gsis_id↔pfr_id crosswalk
    # PBP uses abbreviated names (J.Jefferson), snaps use full names + pfr_player_id
    # Bridge: import_ids() has both gsis_id and pfr_id
    print(f"  Building ID crosswalk (gsis_id ↔ pfr_id)...")
    gsis_to_pfr = {}
    pfr_to_full = {}
    try:
        ids_df = nfl.import_ids()
        for _, row in ids_df.iterrows():
            gsis = row.get("gsis_id", "")
            pfr = row.get("pfr_id", "")
            full_name = row.get("name", "")
            if gsis and pfr:
                gsis_to_pfr[gsis] = pfr
            if pfr and full_name:
                pfr_to_full[pfr] = full_name
        print(f"  {len(gsis_to_pfr)} GSIS→PFR entries")
    except Exception as exc:
        print(f"  ID crosswalk load failed: {exc}")

    # Build (abbreviated PBP name, season) → pfr_id from PBP player IDs
    # Load PBP again briefly just for the ID columns
    # Key by (name, season) to handle abbreviated name collisions
    print(f"  Building PBP name→pfr_id lookup...")
    abbrev_to_pfr = {}  # (name, season) → pfr_id, falling back to name → pfr_id
    try:
        pbp = nfl.import_pbp_data(seasons)
        for role, name_col, id_col in [
            ("recv", "receiver_player_name", "receiver_player_id"),
            ("rush", "rusher_player_name", "rusher_player_id"),
            ("pass", "passer_player_name", "passer_player_id"),
        ]:
            subset = pbp[pbp[id_col].notna()][[name_col, id_col, "season"]].drop_duplicates()
            for _, row in subset.iterrows():
                gsis = row[id_col]
                name = row[name_col]
                season = int(row["season"])
                pfr = gsis_to_pfr.get(gsis)
                if pfr and name:
                    abbrev_to_pfr[(name, season)] = pfr
                    # Also store name-only for fallback
                    abbrev_to_pfr[name] = pfr
        del pbp  # free memory
        # Count unique names (not (name, season) pairs)
        n_names = len({k for k in abbrev_to_pfr if isinstance(k, str)})
        print(f"  {n_names} unique PBP names mapped to PFR "
              f"({len(abbrev_to_pfr)} total entries)")
    except Exception as exc:
        print(f"  PBP name→pfr lookup failed: {exc}")
        abbrev_to_pfr = {}

    # Build per-player per-season rolling stats
    # Group: (player, stat, pos, season) → [(week, value)]
    player_games = defaultdict(list)
    for r in records:
        if r["stat_type"] not in ("pass_yds", "rec_yds", "rush_yds"):
            continue
        key = (r["player_name"], r["stat_type"], r["position"], r["season"])
        player_games[key].append((r["week"], r["stat_value"]))

    # Build snap share norms keyed by pfr_player_id
    snap_norms = {}   # (pfr_id, season) → avg offense_pct
    snap_by_game = {} # (pfr_id, season, week) → offense_pct
    if snaps is not None and len(snaps) > 0:
        for _, row in snaps.iterrows():
            if row.get("game_type") != "REG":
                continue
            pfr_id = row.get("pfr_player_id", "")
            season = row.get("season", 0)
            week = row.get("week", 0)
            off_pct = row.get("offense_pct", 0) or 0
            if pfr_id and off_pct > 0:
                snap_by_game[(pfr_id, season, week)] = off_pct
                if (pfr_id, season) not in snap_norms:
                    snap_norms[(pfr_id, season)] = []
                snap_norms[(pfr_id, season)].append(off_pct)

        # Convert to averages
        snap_norms = {k: np.mean(v) for k, v in snap_norms.items()
                      if len(v) >= 4}

    # Report match rate for snap join (by position)
    snap_matches = 0
    snap_attempts = 0
    snap_match_by_pos = defaultdict(lambda: [0, 0])  # [attempts, matches]

    # Compute dud rates
    results = defaultdict(lambda: {"n_games": 0, "n_yard_duds": 0,
                                    "n_snap_duds": 0, "dud_details": []})

    for (player, stat, pos, season), games in player_games.items():
        games.sort(key=lambda x: x[0])
        if len(games) < 4:
            continue

        # Resolve abbreviated PBP name → pfr_player_id via crosswalk
        # Try (name, season) first, then name-only fallback
        pfr_id = abbrev_to_pfr.get((player, season), abbrev_to_pfr.get(player))

        # Need at least 4 games to compute rolling mean
        for i in range(4, len(games)):
            prior_vals = [g[1] for g in games[:i]]
            rolling_mean = np.mean(prior_vals)
            if rolling_mean < 5:  # skip very low-volume players
                continue

            actual = games[i][1]
            week = games[i][0]
            key = (stat, pos)

            results[key]["n_games"] += 1

            # Yard dud: < 25% of rolling mean
            if actual < rolling_mean * 0.25:
                results[key]["n_yard_duds"] += 1

            # Snap dud: offense_pct < 50% of player's season norm
            # Use pfr_id for snap lookup (gsis_id→pfr_id crosswalk)
            snap_attempts += 1
            snap_match_by_pos[pos][0] += 1
            if pfr_id:
                norm = snap_norms.get((pfr_id, season))
                game_snap = snap_by_game.get((pfr_id, season, week))
                if norm and game_snap:
                    snap_matches += 1
                    snap_match_by_pos[pos][1] += 1
                    if norm > 0.2 and game_snap < norm * 0.50:
                        results[key]["n_snap_duds"] += 1

    # Report snap join match rate (overall and by position)
    if snap_attempts > 0:
        match_rate = snap_matches / snap_attempts
        print(f"\n  Snap join match rate: {snap_matches}/{snap_attempts} "
              f"({match_rate:.1%})")
        for pos in sorted(snap_match_by_pos.keys()):
            att, mat = snap_match_by_pos[pos]
            if att > 0:
                print(f"    {pos:3s}: {mat}/{att} ({mat/att:.1%})")
    else:
        print(f"\n  Snap join: no attempts")

    # Format output
    print(f"\n{'='*80}")
    print("DUD RATES BY STAT × POSITION (2023-2025)")
    print("Yard dud: actual < 25% of rolling mean")
    print("Snap dud: offense_pct < 50% of player's season average")
    print(f"{'='*80}")
    print(f"\n  {'Stat':12s} {'Pos':5s} {'N_games':>8s} {'Yard_dud%':>10s} "
          f"{'Snap_dud%':>10s} {'Combined':>9s}")
    print("  " + "-" * 60)

    formatted = {}
    for (stat, pos) in sorted(results.keys()):
        d = results[(stat, pos)]
        n = d["n_games"]
        if n < 30:
            continue
        yd_rate = d["n_yard_duds"] / n
        snap_rate = d["n_snap_duds"] / n if d["n_snap_duds"] > 0 else 0
        # Approximate combined: P(yard_dud) is a superset in many cases
        # Use yard_dud as primary signal
        print(f"  {stat:12s} {pos:5s} {n:8d} {yd_rate:10.3f} "
              f"{snap_rate:10.3f} {yd_rate:9.3f}")
        formatted[(stat, pos)] = {
            "dud_rate": round(yd_rate, 4),
            "snap_dud_rate": round(snap_rate, 4),
            "n_games": n,
        }

    return formatted


# ══════════════════════════════════════════════════════════════════════════════
# 3. Fit (1-π)·gamma + π·dud mixture
# ══════════════════════════════════════════════════════════════════════════════

def _mixture_cdf(x, shape, scale, pi, dud_mean):
    """CDF of (1-π)·Gamma(shape, scale) + π·Exponential(dud_mean)."""
    gamma_cdf = float(sp_stats.gamma.cdf(x, a=shape, scale=scale))
    # Dud component: exponential with mean = dud_mean
    # (captures near-zero outcomes from injuries/benching)
    if dud_mean > 0:
        dud_cdf = float(sp_stats.expon.cdf(x, scale=dud_mean))
    else:
        dud_cdf = 1.0 if x >= 0 else 0.0
    return (1 - pi) * gamma_cdf + pi * dud_cdf


def _mixture_p_over(line, projection, cv, p_over_at_proj, pi, dud_mean):
    """P(actual > line) under the mixture model."""
    from nfl_interim_dispersion import gamma_params_shifted
    shape, scale = gamma_params_shifted(projection, cv, p_over_at_proj)
    return 1.0 - _mixture_cdf(line, shape, scale, pi, dud_mean)


def _mixture_crps(actual, shape, scale, pi, dud_mean):
    """CRPS for the mixture via numerical integration."""
    from scipy.integrate import quad
    y = max(actual, 0.0)
    upper = max(y * 3, shape * scale * 3, 50)

    def integrand(x):
        cdf_val = _mixture_cdf(x, shape, scale, pi, dud_mean)
        indicator = 1.0 if x >= y else 0.0
        return (cdf_val - indicator) ** 2

    result, _ = quad(integrand, 0, upper, limit=200)
    return float(result)


def _mixture_pit(actual, shape, scale, pi, dud_mean):
    """PIT value: F_mixture(actual)."""
    return _mixture_cdf(max(actual, 0.0), shape, scale, pi, dud_mean)


def fit_and_evaluate_mixture(dud_rates: dict):
    """Fit mixture per stat×pos and evaluate on 2026 graded data.

    For each combo:
      - π = empirical dud_rate from nflverse
      - dud_mean = typical dud yardage (position-dependent)
      - Re-score all predictions; compare PIT/reliability/CRPS vs pure gamma
    """
    from nfl_interim_dispersion import (
        get_interim_cv, load_v12_base_rates, get_v12_base_rate,
        gamma_params_shifted, gamma_cdf_shifted, gamma_crps_shifted,
        p_over_line_gamma_shifted,
    )
    load_v12_base_rates()

    # Typical dud yardage by position (exponential mean)
    # When a player has a dud, how many yards do they get on average?
    # RB dud ≈ 5-10 yds, WR/TE dud ≈ 5-15 yds, QB dud ≈ 30-50 yds
    _DUD_MEAN = {
        ("pass_yds", "QB"): 40.0,
        ("rec_yds", "WR"): 8.0,
        ("rec_yds", "TE"): 6.0,
        ("rec_yds", "RB"): 4.0,
        ("rush_yds", "RB"): 8.0,
        ("rush_yds", "QB"): 3.0,
    }

    preds = _load_graded_v12()

    print(f"\n{'='*90}")
    print("MIXTURE MODEL: (1-π)·Gamma + π·Exponential(dud)")
    print(f"{'='*90}")

    # Re-score all predictions under mixture and pure gamma
    results_by_sp = defaultdict(lambda: {
        "gamma_pits": [], "mix_pits": [],
        "gamma_crps": [], "mix_crps": [],
        "gamma_brier": [], "mix_brier": [],
        "tail_gamma": [], "tail_mix": [],
    })

    for p in preds:
        mean = p["consensus_mean"]
        stat = p["stat_type"]
        pos = p["position"]
        line = p["prop_line"]
        actual = p["actual_value"]
        if not mean or mean <= 0 or not line or actual is None:
            continue

        cv = get_interim_cv(stat, pos, mean, use_v12=True)
        br = get_v12_base_rate(stat, pos)
        pi = dud_rates.get((stat, pos), {}).get("dud_rate", 0.05)
        dud_mean = _DUD_MEAN.get((stat, pos), 5.0)

        # Pure gamma (v1.2)
        g_p_ov = p_over_line_gamma_shifted(line, mean, cv, br)
        g_pit = gamma_cdf_shifted(actual, mean, cv, br)
        g_crps = gamma_crps_shifted(actual, mean, cv, br)

        # Mixture
        shape, scale = gamma_params_shifted(mean, cv, br)
        m_p_ov = _mixture_p_over(line, mean, cv, br, pi, dud_mean)
        m_pit = _mixture_pit(actual, shape, scale, pi, dud_mean)
        m_crps = _mixture_crps(actual, shape, scale, pi, dud_mean)

        hit = 1.0 if actual > line else (0.5 if actual == line else 0.0)
        g_brier = (g_p_ov - hit) ** 2
        m_brier = (m_p_ov - hit) ** 2

        key = (stat, pos)
        r = results_by_sp[key]
        r["gamma_pits"].append(g_pit)
        r["mix_pits"].append(m_pit)
        r["gamma_crps"].append(g_crps)
        r["mix_crps"].append(m_crps)
        r["gamma_brier"].append(g_brier)
        r["mix_brier"].append(m_brier)

        # Tail tracking (P > 0.80 on over side)
        if g_p_ov >= 0.80:
            r["tail_gamma"].append({"p": g_p_ov, "hit": hit})
            r["tail_mix"].append({"p": m_p_ov, "hit": hit})

    # Report per stat×pos
    print(f"\n  {'Stat':12s} {'Pos':5s} {'π':>5s} {'dud_μ':>6s} "
          f"{'N':>5s} {'G_CRPS':>7s} {'M_CRPS':>7s} {'Δ%':>6s} "
          f"{'G_Brier':>8s} {'M_Brier':>8s} {'Δ%':>6s}")
    print("  " + "-" * 85)

    for (stat, pos) in sorted(results_by_sp.keys()):
        r = results_by_sp[(stat, pos)]
        pi = dud_rates.get((stat, pos), {}).get("dud_rate", 0.05)
        dud_mean = _DUD_MEAN.get((stat, pos), 5.0)
        n = len(r["gamma_crps"])
        if n < 10:
            continue

        g_crps = np.mean(r["gamma_crps"])
        m_crps = np.mean(r["mix_crps"])
        g_brier = np.mean(r["gamma_brier"])
        m_brier = np.mean(r["mix_brier"])
        crps_pct = (m_crps - g_crps) / g_crps * 100 if g_crps > 0 else 0
        brier_pct = (m_brier - g_brier) / g_brier * 100 if g_brier > 0 else 0

        print(f"  {stat:12s} {pos:5s} {pi:5.3f} {dud_mean:6.1f} "
              f"{n:5d} {g_crps:7.2f} {m_crps:7.2f} {crps_pct:+5.1f}% "
              f"{g_brier:8.4f} {m_brier:8.4f} {brier_pct:+5.1f}%")

    # PIT comparison
    print(f"\n  PIT U-SHAPE COMPARISON (gamma vs mixture):")
    print(f"  {'Stat':12s} {'Pos':5s} {'G U-shape':>10s} {'M U-shape':>10s}")
    print("  " + "-" * 45)

    for (stat, pos) in sorted(results_by_sp.keys()):
        r = results_by_sp[(stat, pos)]
        if len(r["gamma_pits"]) < 30:
            continue

        def u_shape(pit_vals):
            n_bins = 10
            counts = [0] * n_bins
            for v in pit_vals:
                b = min(int(v * n_bins), n_bins - 1)
                counts[b] += 1
            total = len(pit_vals)
            expected = total / n_bins
            ratios = [c / expected if expected > 0 else 0 for c in counts]
            if len(ratios) >= 4:
                ext = (ratios[0] + ratios[-1]) / 2
                mid = np.mean(ratios[3:7])
                return ext / mid if mid > 0 else 99
            return 99

        g_u = u_shape(r["gamma_pits"])
        m_u = u_shape(r["mix_pits"])
        print(f"  {stat:12s} {pos:5s} {g_u:10.2f} {m_u:10.2f}")

    # Tail bin analysis (0.80-1.0)
    print(f"\n  TAIL RELIABILITY (P(over) >= 0.80):")
    print(f"  {'Stat':12s} {'Pos':5s} {'N':>5s} "
          f"{'G_pred':>7s} {'G_hit':>6s} {'G_gap':>7s} "
          f"{'M_pred':>7s} {'M_hit':>6s} {'M_gap':>7s}")
    print("  " + "-" * 70)

    all_tail_g = []
    all_tail_m = []
    for (stat, pos) in sorted(results_by_sp.keys()):
        r = results_by_sp[(stat, pos)]
        tg = r["tail_gamma"]
        tm = r["tail_mix"]
        if len(tg) < 5:
            continue

        g_pred = np.mean([t["p"] for t in tg])
        g_hit = np.mean([t["hit"] for t in tg])
        m_pred = np.mean([t["p"] for t in tm])
        m_hit = np.mean([t["hit"] for t in tm])

        print(f"  {stat:12s} {pos:5s} {len(tg):5d} "
              f"{g_pred:7.3f} {g_hit:6.3f} {g_hit-g_pred:+7.3f} "
              f"{m_pred:7.3f} {m_hit:6.3f} {m_hit-m_pred:+7.3f}")

        all_tail_g.extend(tg)
        all_tail_m.extend(tm)

    if all_tail_g:
        g_pred = np.mean([t["p"] for t in all_tail_g])
        g_hit = np.mean([t["hit"] for t in all_tail_g])
        m_pred = np.mean([t["p"] for t in all_tail_m])
        m_hit = np.mean([t["hit"] for t in all_tail_m])
        print(f"\n  {'ALL':12s} {'':5s} {len(all_tail_g):5d} "
              f"{g_pred:7.3f} {g_hit:6.3f} {g_hit-g_pred:+7.3f} "
              f"{m_pred:7.3f} {m_hit:6.3f} {m_hit-m_pred:+7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tail-only":
        break_tail_bins()
    else:
        # Full analysis: tail breakdown + dud rates + mixture fit
        print("Step 1: Tail bin breakdown")
        break_tail_bins()

        print("\n\nStep 2: Dud rates from nflverse")
        dud_rates = compute_dud_rates()

        print("\n\nStep 3: Mixture model fit and evaluation")
        fit_and_evaluate_mixture(dud_rates)
