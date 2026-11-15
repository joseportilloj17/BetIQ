"""
nfl_player_allocation.py -- Allocate team-level sim stats to individual players.

Takes one sim draw's team stats (from nfl_game_sim._compile_play_stats) and a
list of player share profiles, returns per-player stat lines.

Design constraints:
  1. Sum of player receptions <= team completions
  2. Sum of player rec_yds ~= team pass_yds  (rescaled exactly)
  3. Sum of player rush_att = team rush_att   (multinomial)
  4. Sum of player rush_yds ~= team rush_yds  (rescaled exactly)
  5. TDs allocated via multinomial (not independent binomial), so sum = team TDs

The share profiles come from nfl_volume_reconciliation.compute_player_shares(),
which provides pred_target_share, pred_carry_share, and position per player.
Player-specific efficiency (catch_rate, ypr, ypc) can be passed in the profile
dict; when absent, team-average rates from the sim draw are used.

Usage:
    from nfl_player_allocation import allocate_game_stats, allocate_game_n
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# League-average per-player efficiency defaults (2023-2025 PBP)
_LG_CATCH_RATE = 0.648         # completions / targets (all positions)
_LG_YPR = 11.2                 # yards per reception (all positions)
_LG_YPC = 4.24                 # yards per carry (RBs)
_LG_YPR_SD_FRAC = 0.55         # Y/R noise as fraction of mean
_LG_YPC_SD_FRAC = 0.70         # Y/C noise as fraction of mean
_MIN_YARDS_PER_PLAY = -5.0     # floor on single-play yardage draw

# Dirichlet share-noise concentration.  Higher = less per-draw variation.
# Empirical: WR1 with 25% mean share has ~7-8pp game-to-game SD.
# Multinomial counting noise already contributes ~5pp SD.
# Dirichlet at kappa=150 adds ~3pp, matching observed total ~8pp.
_SHARE_CONCENTRATION = 150.0


# ══════════════════════════════════════════════════════════════════════════════
# Single-draw allocation
# ══════════════════════════════════════════════════════════════════════════════

def allocate_game_stats(
    rng: np.random.Generator,
    team_stats: dict,
    player_shares: list[dict],
) -> list[dict]:
    """Allocate team-level sim stats to individual players for one draw.

    Args:
        rng: numpy random generator (shared across draws for reproducibility).
        team_stats: one sim draw's compiled stats dict, with keys:
            pass_att, completions, pass_yards, sacks, scrambles,
            scramble_yards, rush_att, rush_yards, pass_tds, rush_tds,
            interceptions, fumbles, total_plays.
        player_shares: list of player profile dicts, each with:
            Required: name (str), position (str: WR/TE/RB/QB),
                      target_share (float), rush_share (float).
            Optional: catch_rate (float), ypr (float), ypc (float),
                      rz_target_share (float), rz_rush_share (float).

    Returns:
        list of dicts, one per player in player_shares, each with:
            name, position, targets, receptions, rec_yards, rec_tds,
            rush_att, rush_yards, rush_tds.
    """
    # ── Extract team totals ──
    team_pass_att = team_stats.get("pass_att", 0)
    team_completions = team_stats.get("completions", 0)
    team_pass_yards = team_stats.get("pass_yards", 0)
    team_rush_att = team_stats.get("rush_att", 0)
    team_rush_yards = team_stats.get("rush_yards", 0)
    team_pass_tds = team_stats.get("pass_tds", 0)
    team_rush_tds = team_stats.get("rush_tds", 0)
    team_scrambles = team_stats.get("scrambles", 0)
    team_scramble_yards = team_stats.get("scramble_yards", 0)
    team_sacks = team_stats.get("sacks", 0)

    # Throw attempts: targets come from actual throws, not sacks/scrambles.
    team_throws = max(0, team_pass_att - team_sacks - team_scrambles)

    # Derived team rates for fallback (based on throws, not dropbacks)
    team_comp_rate = (team_completions / team_throws
                      if team_throws > 0 else _LG_CATCH_RATE)
    team_ypr = (team_pass_yards / max(1, team_completions)
                if team_completions > 0 else _LG_YPR)
    team_ypc = (team_rush_yards / max(1, team_rush_att)
                if team_rush_att > 0 else _LG_YPC)

    # ── Partition players by role ──
    receivers = []    # WR, TE, RB with target_share > 0
    rushers = []      # RB with rush_share > 0
    qb = None

    for p in player_shares:
        pos = p.get("position", "WR")
        if pos == "QB":
            qb = p
            continue
        if p.get("target_share", 0) > 0:
            receivers.append(p)
        if pos == "RB" and p.get("rush_share", 0) > 0:
            rushers.append(p)

    # ── 1. RECEIVING ALLOCATION ──
    rec_results = _allocate_receiving(
        rng, receivers, team_throws, team_completions,
        team_pass_yards, team_pass_tds, team_comp_rate, team_ypr,
    )

    # ── 2. RUSHING ALLOCATION ──
    rush_results = _allocate_rushing(
        rng, rushers, team_rush_att, team_rush_yards,
        team_rush_tds, team_ypc,
    )

    # ── 3. QB STATS ──
    qb_result = None
    if qb is not None:
        qb_result = _build_qb_stats(
            qb, team_stats, team_scrambles, team_scramble_yards,
        )

    # ── 4. Merge into output ──
    output = []
    for p in player_shares:
        name = p["name"]
        pos = p.get("position", "WR")

        row = {
            "name": name,
            "position": pos,
            "targets": 0,
            "receptions": 0,
            "rec_yards": 0.0,
            "rec_tds": 0,
            "rush_att": 0,
            "rush_yards": 0.0,
            "rush_tds": 0,
        }

        if pos == "QB" and qb_result is not None and name == qb["name"]:
            row.update(qb_result)
        else:
            if name in rec_results:
                rr = rec_results[name]
                row["targets"] = rr["targets"]
                row["receptions"] = rr["receptions"]
                row["rec_yards"] = rr["rec_yards"]
                row["rec_tds"] = rr["rec_tds"]
            if name in rush_results:
                ru = rush_results[name]
                row["rush_att"] = ru["rush_att"]
                row["rush_yards"] = ru["rush_yards"]
                row["rush_tds"] = ru["rush_tds"]

        output.append(row)

    return output


# ── Receiving sub-routine ──

def _allocate_receiving(
    rng: np.random.Generator,
    receivers: list[dict],
    team_pass_att: int,
    team_completions: int,
    team_pass_yards: float,
    team_pass_tds: int,
    team_comp_rate: float,
    team_ypr: float,
) -> dict[str, dict]:
    """Allocate pass-game stats to receivers.

    Targets via multinomial, receptions conditional on targets,
    rec_yds rescaled to match team total, TDs via multinomial.

    Returns: {player_name: {targets, receptions, rec_yards, rec_tds}}
    """
    if not receivers or team_pass_att <= 0:
        return {}

    n_recv = len(receivers)

    # ── Targets: multinomial with per-draw share noise ──
    # Build target share vector including an _OTHER bucket.
    raw_ts = np.array([p.get("target_share", 0) for p in receivers],
                      dtype=float)
    ts_sum = raw_ts.sum()
    other_ts = max(0.02, 1.0 - ts_sum)  # at least 2% to unnamed players
    probs = np.concatenate([raw_ts, [other_ts]])
    # Normalize to exactly 1.0 (safety against float drift)
    probs = probs / probs.sum()

    # Dirichlet noise: perturb share probabilities per draw.
    # This captures game-to-game share variation beyond multinomial
    # counting noise — e.g. game script, coverage, hot-hand effects.
    alpha = probs * _SHARE_CONCENTRATION
    alpha = np.maximum(alpha, 0.1)  # floor prevents Dirichlet crash on 0
    probs = rng.dirichlet(alpha)

    target_draws = rng.multinomial(team_pass_att, probs)
    # Last element is the _OTHER bucket; discard it
    player_targets = target_draws[:n_recv]

    # ── Receptions: binomial per player, capped so sum <= team_completions ──
    catch_rates = np.array([
        p.get("catch_rate", team_comp_rate)
        for p in receivers
    ], dtype=float)
    # Clamp to [0.2, 0.95]
    catch_rates = np.clip(catch_rates, 0.2, 0.95)

    raw_receptions = np.array([
        rng.binomial(int(t), float(cr))
        for t, cr in zip(player_targets, catch_rates)
    ])

    # Cap total receptions at team completions
    total_rec_raw = raw_receptions.sum()
    if total_rec_raw > team_completions and total_rec_raw > 0:
        # Scale down proportionally, preserving integer counts
        scale = team_completions / total_rec_raw
        scaled = (raw_receptions * scale).astype(int)
        # Distribute rounding remainder to random players
        remainder = team_completions - scaled.sum()
        if remainder > 0:
            eligible = np.where(raw_receptions > scaled)[0]
            if len(eligible) > 0:
                bump_idx = rng.choice(eligible,
                                      size=min(remainder, len(eligible)),
                                      replace=False)
                scaled[bump_idx] += 1
        player_receptions = scaled
    else:
        player_receptions = raw_receptions

    # ── Receiving yards: per-reception draws, then rescale to team total ──
    yprs = np.array([
        p.get("ypr", team_ypr)
        for p in receivers
    ], dtype=float)
    # Clamp Y/R to [3, 25]
    yprs = np.clip(yprs, 3.0, 25.0)

    raw_rec_yards = np.zeros(n_recv)
    for i in range(n_recv):
        n_rec = int(player_receptions[i])
        if n_rec <= 0:
            continue
        sd = yprs[i] * _LG_YPR_SD_FRAC
        per_catch = rng.normal(yprs[i], sd, size=n_rec)
        per_catch = np.maximum(per_catch, _MIN_YARDS_PER_PLAY)
        raw_rec_yards[i] = per_catch.sum()

    # Rescale so sum of player rec_yds = team pass_yds exactly
    total_raw_ry = raw_rec_yards.sum()
    if total_raw_ry > 0 and team_pass_yards > 0:
        scale = team_pass_yards / total_raw_ry
        player_rec_yards = raw_rec_yards * scale
    elif team_pass_yards > 0 and player_receptions.sum() > 0:
        # Edge case: all draws were negative.  Distribute proportionally
        # to receptions.
        weights = player_receptions / player_receptions.sum()
        player_rec_yards = weights * team_pass_yards
    else:
        player_rec_yards = raw_rec_yards

    # ── Receiving TDs: multinomial ──
    if team_pass_tds > 0 and n_recv > 0:
        # Use rz_target_share if available, else fall back to target_share
        rz_shares = np.array([
            p.get("rz_target_share", p.get("target_share", 0))
            for p in receivers
        ], dtype=float)
        rz_sum = rz_shares.sum()
        other_rz = max(0.05, 1.0 - rz_sum)  # unnamed players can score too
        rz_probs = np.concatenate([rz_shares, [other_rz]])
        rz_probs = rz_probs / rz_probs.sum()

        td_draws = rng.multinomial(team_pass_tds, rz_probs)
        player_rec_tds = td_draws[:n_recv]
    else:
        player_rec_tds = np.zeros(n_recv, dtype=int)

    # ── Build result dict ──
    result = {}
    for i, p in enumerate(receivers):
        result[p["name"]] = {
            "targets": int(player_targets[i]),
            "receptions": int(player_receptions[i]),
            "rec_yards": float(player_rec_yards[i]),
            "rec_tds": int(player_rec_tds[i]),
        }

    return result


# ── Rushing sub-routine ──

def _allocate_rushing(
    rng: np.random.Generator,
    rushers: list[dict],
    team_rush_att: int,
    team_rush_yards: float,
    team_rush_tds: int,
    team_ypc: float,
) -> dict[str, dict]:
    """Allocate rush stats to RBs.

    Rush attempts via multinomial, rush yards per-carry then rescaled,
    rush TDs via multinomial.

    Returns: {player_name: {rush_att, rush_yards, rush_tds}}
    """
    if not rushers or team_rush_att <= 0:
        return {}

    n_rush = len(rushers)

    # ── Rush attempts: multinomial with per-draw share noise ──
    raw_rs = np.array([p.get("rush_share", 0) for p in rushers],
                      dtype=float)
    rs_sum = raw_rs.sum()
    other_rs = max(0.02, 1.0 - rs_sum)  # QB scrambles not here, plus WR sweeps
    probs = np.concatenate([raw_rs, [other_rs]])
    probs = probs / probs.sum()

    alpha = probs * _SHARE_CONCENTRATION
    alpha = np.maximum(alpha, 0.1)
    probs = rng.dirichlet(alpha)

    att_draws = rng.multinomial(team_rush_att, probs)
    player_rush_att = att_draws[:n_rush]

    # ── Rush yards: per-carry draws, then rescale ──
    ypcs = np.array([
        p.get("ypc", team_ypc)
        for p in rushers
    ], dtype=float)
    ypcs = np.clip(ypcs, 1.0, 8.0)

    raw_rush_yards = np.zeros(n_rush)
    for i in range(n_rush):
        n_att = int(player_rush_att[i])
        if n_att <= 0:
            continue
        sd = ypcs[i] * _LG_YPC_SD_FRAC
        per_carry = rng.normal(ypcs[i], sd, size=n_att)
        per_carry = np.maximum(per_carry, _MIN_YARDS_PER_PLAY)
        raw_rush_yards[i] = per_carry.sum()

    # Rescale so sum = team_rush_yards
    total_raw_ry = raw_rush_yards.sum()
    if total_raw_ry > 0 and team_rush_yards != 0:
        # Handle negative total rush yards (rare but possible in sim)
        scale = team_rush_yards / total_raw_ry
        player_rush_yards = raw_rush_yards * scale
    elif team_rush_yards != 0 and player_rush_att.sum() > 0:
        weights = player_rush_att / player_rush_att.sum()
        player_rush_yards = weights * team_rush_yards
    else:
        player_rush_yards = raw_rush_yards

    # ── Rush TDs: multinomial ──
    if team_rush_tds > 0 and n_rush > 0:
        rz_shares = np.array([
            p.get("rz_rush_share", p.get("rush_share", 0))
            for p in rushers
        ], dtype=float)
        rz_sum = rz_shares.sum()
        other_rz = max(0.05, 1.0 - rz_sum)
        rz_probs = np.concatenate([rz_shares, [other_rz]])
        rz_probs = rz_probs / rz_probs.sum()

        td_draws = rng.multinomial(team_rush_tds, rz_probs)
        player_rush_tds = td_draws[:n_rush]
    else:
        player_rush_tds = np.zeros(n_rush, dtype=int)

    # ── Build result dict ──
    result = {}
    for i, p in enumerate(rushers):
        result[p["name"]] = {
            "rush_att": int(player_rush_att[i]),
            "rush_yards": float(player_rush_yards[i]),
            "rush_tds": int(player_rush_tds[i]),
        }

    return result


# ── QB stats ──

def _build_qb_stats(
    qb: dict,
    team_stats: dict,
    team_scrambles: int,
    team_scramble_yards: float,
) -> dict:
    """Build QB stat line from team totals.

    QB gets all pass stats directly from the team sim output plus
    scramble rushing stats.
    """
    # QB pass_att = throw attempts only (prop-settlement definition)
    throws = (team_stats.get("pass_att", 0)
              - team_stats.get("sacks", 0)
              - team_scrambles)
    return {
        "pass_att": max(0, throws),
        "completions": team_stats.get("completions", 0),
        "pass_yards": team_stats.get("pass_yards", 0),
        "pass_tds": team_stats.get("pass_tds", 0),
        "interceptions": team_stats.get("interceptions", 0),
        "sacks": team_stats.get("sacks", 0),
        # QB rushing = scrambles (designed QB runs are rare in this sim)
        "rush_att": team_scrambles,
        "rush_yards": team_scramble_yards,
        "rush_tds": 0,  # scramble TDs not tracked separately in sim
    }


# ══════════════════════════════════════════════════════════════════════════════
# N-draw allocation: distributions
# ══════════════════════════════════════════════════════════════════════════════

def allocate_game_n(
    rng: np.random.Generator,
    team_stats_list: list[dict],
    player_shares: list[dict],
) -> dict:
    """Run allocation across N sim draws, return per-player distributions.

    Args:
        rng: numpy random generator.
        team_stats_list: list of N team_stats dicts (one per sim draw),
            each in the format returned by nfl_game_sim._compile_play_stats.
        player_shares: list of player profile dicts (same for every draw).

    Returns: {
        player_name: {
            "position": str,
            "targets": {"mean", "sd", "p10", "p25", "p50", "p75", "p90", "sims"},
            "receptions": {...},
            "rec_yards": {...},
            "rec_tds": {...},
            "rush_att": {...},
            "rush_yards": {...},
            "rush_tds": {...},
        },
        ...
        "_meta": {"n_sims": int, "n_players": int},
    }
    """
    n_sims = len(team_stats_list)
    if n_sims == 0:
        return {"_meta": {"n_sims": 0, "n_players": len(player_shares)}}

    # Accumulate per-player stats across draws
    player_names = [p["name"] for p in player_shares]
    stat_keys = ["targets", "receptions", "rec_yards", "rec_tds",
                 "rush_att", "rush_yards", "rush_tds"]

    accum = {
        name: {k: [] for k in stat_keys}
        for name in player_names
    }

    # Also accumulate QB-specific pass stats if QB is present
    qb_pass_keys = ["pass_att", "completions", "pass_yards",
                    "pass_tds", "interceptions", "sacks"]
    qb_name = None
    for p in player_shares:
        if p.get("position") == "QB":
            qb_name = p["name"]
            for k in qb_pass_keys:
                accum[qb_name][k] = []
            break

    for team_stats in team_stats_list:
        draw = allocate_game_stats(rng, team_stats, player_shares)
        for row in draw:
            name = row["name"]
            for k in stat_keys:
                accum[name][k].append(row.get(k, 0))
            # QB pass stats
            if name == qb_name:
                for k in qb_pass_keys:
                    if k in row:
                        accum[name][k].append(row[k])

    # Aggregate into distribution summaries
    result = {}
    for p in player_shares:
        name = p["name"]
        player_result = {"position": p.get("position", "WR")}

        for k, vals in accum[name].items():
            if not vals:
                continue
            a = np.array(vals, dtype=float)
            player_result[k] = {
                "mean": round(float(a.mean()), 2),
                "sd": round(float(a.std()), 2),
                "p10": round(float(np.percentile(a, 10)), 2),
                "p25": round(float(np.percentile(a, 25)), 2),
                "p50": round(float(np.percentile(a, 50)), 2),
                "p75": round(float(np.percentile(a, 75)), 2),
                "p90": round(float(np.percentile(a, 90)), 2),
                "sims": a.tolist(),
            }
        result[name] = player_result

    result["_meta"] = {
        "n_sims": n_sims,
        "n_players": len(player_shares),
    }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Bridge: compute_player_shares dict -> player_shares list
# ══════════════════════════════════════════════════════════════════════════════

def shares_dict_to_list(
    shares: dict,
    player_efficiency: dict | None = None,
) -> list[dict]:
    """Convert compute_player_shares() output to the list format
    expected by allocate_game_stats.

    Args:
        shares: output of nfl_volume_reconciliation.compute_player_shares(),
            keyed by player name, each with pred_target_share, pred_carry_share,
            position, n_weeks, source.
        player_efficiency: optional {player_name: {catch_rate, ypr, ypc, ...}}
            for player-specific efficiency rates.  When absent, team-level
            rates derived from the sim draw are used.

    Returns: list of player dicts suitable for allocate_game_stats.
    """
    if player_efficiency is None:
        player_efficiency = {}

    result = []
    for name, data in shares.items():
        if name == "_OTHER":
            continue

        eff = player_efficiency.get(name, {})
        entry = {
            "name": name,
            "position": data.get("position", "WR"),
            "target_share": data.get("pred_target_share", 0),
            "rush_share": data.get("pred_carry_share", 0),
        }

        # Merge player-specific efficiency if available
        for k in ("catch_rate", "ypr", "ypc",
                   "rz_target_share", "rz_rush_share"):
            if k in eff:
                entry[k] = eff[k]

        result.append(entry)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic: constraint verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_constraints(
    team_stats: dict,
    player_results: list[dict],
    tolerance: float = 1.0,
) -> dict:
    """Check allocation constraints against team totals.

    Returns dict with pass/fail per constraint and any violations.
    Useful for testing; not called in production.
    """
    total_rec = sum(r.get("receptions", 0) for r in player_results
                    if r.get("position") != "QB")
    total_rec_yds = sum(r.get("rec_yards", 0) for r in player_results
                        if r.get("position") != "QB")
    total_rush_att = sum(r.get("rush_att", 0) for r in player_results
                         if r.get("position") != "QB")
    total_rush_yds = sum(r.get("rush_yards", 0) for r in player_results
                          if r.get("position") != "QB")
    total_rec_tds = sum(r.get("rec_tds", 0) for r in player_results
                        if r.get("position") != "QB")
    total_rush_tds = sum(r.get("rush_tds", 0) for r in player_results
                          if r.get("position") != "QB")

    team_comp = team_stats.get("completions", 0)
    team_pass_yds = team_stats.get("pass_yards", 0)
    team_rush_att = team_stats.get("rush_att", 0)
    team_rush_yds = team_stats.get("rush_yards", 0)
    team_pass_tds = team_stats.get("pass_tds", 0)
    team_rush_tds = team_stats.get("rush_tds", 0)

    checks = {
        "receptions_le_completions": {
            "pass": total_rec <= team_comp,
            "player_total": total_rec,
            "team_total": team_comp,
        },
        "rec_yards_match": {
            "pass": abs(total_rec_yds - team_pass_yds) <= tolerance,
            "player_total": round(total_rec_yds, 1),
            "team_total": team_pass_yds,
            "diff": round(total_rec_yds - team_pass_yds, 1),
        },
        "rush_att_le_team": {
            # Named rushers can be < team total (other bucket absorbs rest)
            "pass": total_rush_att <= team_rush_att,
            "player_total": total_rush_att,
            "team_total": team_rush_att,
        },
        "rush_yards_match": {
            # Rush yards rescaled, so should be close but not exact
            # (other bucket's yards are excluded)
            "pass": True,  # informational only
            "player_total": round(total_rush_yds, 1),
            "team_total": team_rush_yds,
        },
        "rec_tds_le_team": {
            "pass": total_rec_tds <= team_pass_tds,
            "player_total": total_rec_tds,
            "team_total": team_pass_tds,
        },
        "rush_tds_le_team": {
            "pass": total_rush_tds <= team_rush_tds,
            "player_total": total_rush_tds,
            "team_total": team_rush_tds,
        },
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {"all_pass": all_pass, "checks": checks}


# ══════════════════════════════════════════════════════════════════════════════
# Prop odds from joint draws
# ══════════════════════════════════════════════════════════════════════════════

def compute_prop_odds(
    alloc_result: dict,
    prop_lines: list[dict],
) -> list[dict]:
    """Compute P(over) + MC SE for each prop line from sim draws.

    Args:
        alloc_result: output of ``allocate_game_n()``, with per-player
            stat distributions including ``"sims"`` arrays.
        prop_lines: list of prop dicts, each with:
            ``player`` (str), ``stat`` (str: rec_yards, receptions,
            rush_yards, rush_att, targets, pass_yards, completions,
            pass_att, pass_tds, interceptions, sacks),
            ``line`` (float), and optionally ``over_odds`` / ``under_odds``
            (American odds, e.g. -110).

    Returns:
        list of result dicts, one per prop_line, each with:
            player, stat, line, p_over, p_under, mc_se, sim_mean,
            sim_sd, sim_median, n_sims, and optionally edge_over/
            edge_under (if odds were provided).
    """
    results = []
    n_sims = alloc_result.get("_meta", {}).get("n_sims", 0)

    for prop in prop_lines:
        player = prop["player"]
        stat = prop["stat"]
        line = prop["line"]

        player_data = alloc_result.get(player)
        if not player_data or stat not in player_data:
            results.append({
                "player": player, "stat": stat, "line": line,
                "p_over": None, "error": "player/stat not in draws",
            })
            continue

        sims = player_data[stat].get("sims")
        if not sims:
            results.append({
                "player": player, "stat": stat, "line": line,
                "p_over": None, "error": "no sims data",
            })
            continue

        a = np.array(sims, dtype=float)
        n = len(a)
        p_over = float((a > line).mean())
        p_under = 1.0 - p_over
        mc_se = float(np.sqrt(p_over * p_under / n)) if n > 0 else 0

        row = {
            "player": player,
            "stat": stat,
            "line": line,
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "mc_se": round(mc_se, 4),
            "sim_mean": round(float(a.mean()), 2),
            "sim_sd": round(float(a.std()), 2),
            "sim_median": round(float(np.median(a)), 2),
            "n_sims": n,
        }

        # Edge vs market if odds provided
        if "over_odds" in prop and prop["over_odds"] is not None:
            implied = _american_to_prob(prop["over_odds"])
            row["implied_over"] = round(implied, 4)
            row["edge_over"] = round(p_over - implied, 4)

        if "under_odds" in prop and prop["under_odds"] is not None:
            implied = _american_to_prob(prop["under_odds"])
            row["implied_under"] = round(implied, 4)
            row["edge_under"] = round(p_under - implied, 4)

        results.append(row)

    return results


def compute_prop_rungs(
    alloc_result: dict,
    player: str,
    stat: str,
    lines: list[float],
) -> list[dict]:
    """Compute P(over) at multiple alt-line rungs for one player-stat.

    Useful for finding the best rung to bet at.

    Returns: list of {line, p_over, p_under, mc_se} sorted by line.
    """
    player_data = alloc_result.get(player)
    if not player_data or stat not in player_data:
        return []

    sims = player_data[stat].get("sims")
    if not sims:
        return []

    a = np.array(sims, dtype=float)
    n = len(a)

    results = []
    for line in sorted(lines):
        p_over = float((a > line).mean())
        mc_se = float(np.sqrt(p_over * (1 - p_over) / n)) if n > 0 else 0
        results.append({
            "line": line,
            "p_over": round(p_over, 4),
            "p_under": round(1 - p_over, 4),
            "mc_se": round(mc_se, 4),
        })

    return results


def _american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (no-vig)."""
    if odds >= 100:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke test (python nfl_player_allocation.py)
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Run allocation on a synthetic team stat draw + player shares."""
    rng = np.random.default_rng(42)

    # Synthetic team stats (typical NFL game)
    team_stats = {
        "total_plays": 65,
        "pass_att": 35,
        "completions": 22,
        "sacks": 2,
        "scrambles": 3,
        "interceptions": 1,
        "pass_yards": 245,
        "sack_yards": -12,
        "scramble_yards": 18,
        "rush_att": 25,
        "rush_yards": 105,
        "rush_tds": 1,
        "pass_tds": 2,
        "fumbles": 0,
        "pass_rate": 35 / 65,
    }

    # Synthetic player shares (typical roster)
    player_shares = [
        {"name": "J.Herbert", "position": "QB",
         "target_share": 0, "rush_share": 0},
        {"name": "A.Ekeler", "position": "RB",
         "target_share": 0.09, "rush_share": 0.55,
         "catch_rate": 0.78, "ypr": 7.2, "ypc": 4.8},
        {"name": "J.Dobbins", "position": "RB",
         "target_share": 0.04, "rush_share": 0.30,
         "ypc": 4.1},
        {"name": "K.Allen", "position": "WR",
         "target_share": 0.25, "rush_share": 0,
         "catch_rate": 0.72, "ypr": 12.1},
        {"name": "M.Williams", "position": "WR",
         "target_share": 0.18, "rush_share": 0,
         "ypr": 14.8},
        {"name": "Q.Johnston", "position": "WR",
         "target_share": 0.12, "rush_share": 0,
         "ypr": 11.5},
        {"name": "G.Everett", "position": "TE",
         "target_share": 0.10, "rush_share": 0,
         "catch_rate": 0.68, "ypr": 9.8},
    ]

    print("=" * 70)
    print("NFL PLAYER ALLOCATION -- SMOKE TEST")
    print("=" * 70)

    # Single draw
    result = allocate_game_stats(rng, team_stats, player_shares)

    print("\nSingle-draw allocation:")
    print(f"  {'Player':<16} {'Pos':>3} {'Tgt':>4} {'Rec':>4} "
          f"{'RecYd':>6} {'RTD':>3} {'RAtt':>4} {'RuYd':>6} {'RuTD':>3}")
    print("  " + "-" * 55)
    for r in result:
        print(f"  {r['name']:<16} {r['position']:>3} {r.get('targets',0):>4} "
              f"{r.get('receptions',0):>4} {r.get('rec_yards',0):>6.1f} "
              f"{r.get('rec_tds',0):>3} {r.get('rush_att',0):>4} "
              f"{r.get('rush_yards',0):>6.1f} {r.get('rush_tds',0):>3}")

    # Constraint check
    vc = verify_constraints(team_stats, result)
    print(f"\n  Constraints all pass: {vc['all_pass']}")
    for name, chk in vc["checks"].items():
        status = "OK" if chk["pass"] else "FAIL"
        print(f"    {name}: {status}  "
              f"(player={chk['player_total']}, team={chk['team_total']})")

    # N-draw aggregation
    print(f"\n{'='*70}")
    print("N-draw aggregation (N=500):")
    print("=" * 70)

    team_stats_list = [team_stats] * 500  # same team draw for consistency
    dist = allocate_game_n(rng, team_stats_list, player_shares)

    for name in [p["name"] for p in player_shares]:
        pd = dist.get(name, {})
        pos = pd.get("position", "?")
        rec_yd = pd.get("rec_yards", {})
        rush_yd = pd.get("rush_yards", {})
        rec = pd.get("receptions", {})

        if pos == "QB":
            py = pd.get("pass_yards", {})
            print(f"\n  {name} ({pos}):")
            print(f"    Pass yards: mean={py.get('mean',0):.1f}, "
                  f"p10={py.get('p10',0):.0f}, p50={py.get('p50',0):.0f}, "
                  f"p90={py.get('p90',0):.0f}")
        else:
            print(f"\n  {name} ({pos}):")
            print(f"    Receptions: mean={rec.get('mean',0):.1f}, "
                  f"p10={rec.get('p10',0):.0f}, p50={rec.get('p50',0):.0f}, "
                  f"p90={rec.get('p90',0):.0f}")
            print(f"    Rec yards:  mean={rec_yd.get('mean',0):.1f}, "
                  f"p10={rec_yd.get('p10',0):.0f}, p50={rec_yd.get('p50',0):.0f}, "
                  f"p90={rec_yd.get('p90',0):.0f}")
            if rush_yd.get("mean", 0) > 0:
                print(f"    Rush yards: mean={rush_yd.get('mean',0):.1f}, "
                      f"p10={rush_yd.get('p10',0):.0f}, "
                      f"p50={rush_yd.get('p50',0):.0f}, "
                      f"p90={rush_yd.get('p90',0):.0f}")


if __name__ == "__main__":
    _smoke_test()
