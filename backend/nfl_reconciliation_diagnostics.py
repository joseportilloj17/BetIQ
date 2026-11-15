"""
nfl_reconciliation_diagnostics.py — Team-level constraint violation detection.

Quantifies how often the current independent-player projection model violates
team-level accounting identities (e.g., Σ WR rec_yds > QB pass_yds).

This is a DIAGNOSTIC module — it does NOT change any predictions.
It reports violations for inspection before building the reconciliation engine.

Usage:
    python nfl_reconciliation_diagnostics.py              # current week
    python nfl_reconciliation_diagnostics.py --historical  # backtest on stored preds
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

_HERE     = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB  = os.path.join(_DATA_DIR, "historical.db")
_NFL_DB   = os.path.join(_DATA_DIR, "nfl.db")

# Map Sleeper/PBP team abbreviations that differ from nfl_data_py
_TEAM_ALIAS = {"LAR": "LA"}


def _canon_team(team: str) -> str:
    """Normalize team abbreviation to nfl_data_py convention."""
    return _TEAM_ALIAS.get(team, team)


# ══════════════════════════════════════════════════════════════════════════════
# Historical volume baselines from nfl_team_game_stats
# ══════════════════════════════════════════════════════════════════════════════

def compute_historical_volume_stats(seasons: list[int] | None = None) -> dict:
    """
    Load historical team volume stats from nfl_team_game_stats + nfl_games.

    Returns:
        {season: {team: {
            'avg_off_plays': float,
            'avg_pass_plays': float,
            'avg_rush_plays': float,
            'avg_pass_rate': float,
            'games': int,
        }}}
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    conn = sqlite3.connect(os.path.abspath(_NFL_DB))
    conn.row_factory = sqlite3.Row

    results = {}
    for season in seasons:
        rows = conn.execute("""
            SELECT t.team, t.week,
                   t.off_plays, t.pass_plays, t.rush_plays, t.pass_rate,
                   g.spread_line, g.total_line, t.is_home
            FROM nfl_team_game_stats t
            JOIN nfl_games g ON t.game_id = g.game_id
            WHERE t.season = ? AND g.game_type = 'REG'
            ORDER BY t.team, t.week
        """, (season,)).fetchall()

        team_stats = defaultdict(lambda: {
            "off_plays": [], "pass_plays": [], "rush_plays": [],
            "pass_rate": [], "spreads": [], "totals": [],
        })
        for r in rows:
            t = r["team"]
            team_stats[t]["off_plays"].append(r["off_plays"] or 0)
            team_stats[t]["pass_plays"].append(r["pass_plays"] or 0)
            team_stats[t]["rush_plays"].append(r["rush_plays"] or 0)
            team_stats[t]["pass_rate"].append(r["pass_rate"] or 0)
            # Spread from home perspective — flip for away
            spread = r["spread_line"] or 0
            if not r["is_home"]:
                spread = -spread
            team_stats[t]["spreads"].append(spread)
            team_stats[t]["totals"].append(r["total_line"] or 0)

        season_result = {}
        for team, s in team_stats.items():
            n = len(s["off_plays"])
            if n == 0:
                continue
            season_result[team] = {
                "avg_off_plays": sum(s["off_plays"]) / n,
                "avg_pass_plays": sum(s["pass_plays"]) / n,
                "avg_rush_plays": sum(s["rush_plays"]) / n,
                "avg_pass_rate": sum(s["pass_rate"]) / n,
                "avg_spread": sum(s["spreads"]) / n,
                "avg_total": sum(s["totals"]) / n,
                "games": n,
            }
        results[season] = season_result

    conn.close()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Violation detection on consensus/gaps data
# ══════════════════════════════════════════════════════════════════════════════

def _group_by_team(gaps: list[dict], consensus: dict | None = None) -> dict:
    """
    Group gaps and consensus data by team.

    Uses gap['team'] field. Falls back to consensus dict if gap doesn't have team.

    Returns:
        {team: {
            'qb_pass_yds': [(player, consensus_mean, prop_line), ...],
            'rec_yds': [(player, consensus_mean, prop_line), ...],
            'rush_yds': [(player, consensus_mean, prop_line), ...],
            'receptions': [(player, consensus_mean, prop_line), ...],
            'pass_tds': [(player, consensus_mean, prop_line), ...],
            'gaps': [gap_dicts...],
        }}
    """
    teams = defaultdict(lambda: {
        "qb_pass_yds": [], "rec_yds": [], "rush_yds": [],
        "receptions": [], "pass_tds": [], "pass_ints": [],
        "gaps": [],
    })

    # Deduplicate: one entry per (player, stat) — consensus_mean is the same
    # across bookmaker lines, so pick the first occurrence and use median line.
    seen = {}  # (team, player, stat) -> (mean, [lines])
    for g in gaps:
        team = _canon_team(g.get("team", ""))
        if not team:
            continue
        depth = g.get("line_depth", "main")
        if depth != "main":
            continue

        key = (team, g["player"], g["stat"])
        if key not in seen:
            seen[key] = {
                "player": g["player"], "stat": g["stat"],
                "position": g.get("position", ""),
                "mean": g["consensus_mean"], "lines": [],
            }
        seen[key]["lines"].append(g["prop_line"])

    for (team, player, stat), entry in seen.items():
        pos = entry["position"]
        mean = entry["mean"]
        # Use median prop line as representative
        lines_sorted = sorted(entry["lines"])
        line = lines_sorted[len(lines_sorted) // 2]

        teams[team]["gaps"].append(entry)

        if stat == "pass_yds" and pos == "QB":
            teams[team]["qb_pass_yds"].append((player, mean, line))
        elif stat == "rec_yds":
            teams[team]["rec_yds"].append((player, mean, line))
        elif stat == "rush_yds":
            teams[team]["rush_yds"].append((player, mean, line))
        elif stat == "receptions":
            teams[team]["receptions"].append((player, mean, line))
        elif stat == "pass_tds" and pos == "QB":
            teams[team]["pass_tds"].append((player, mean, line))

    return dict(teams)


def detect_team_violations(gaps: list[dict],
                           consensus: dict | None = None,
                           tolerance: float = 0.15) -> list[dict]:
    """
    Check team-level accounting violations in the current projection set.

    Violations checked:
    1. SUM_REC_YDS_VS_QB_PASS_YDS: Σ(receiver rec_yds projections) vs QB pass_yds.
       Tolerance: ±15%. Receivers should not be projected for more receiving yards
       than their QB is projected to throw.
    2. SUM_RECEPTIONS_VS_QB_COMPLETIONS: Σ(receptions) vs QB implied completions
       (pass_yds / historical yards_per_completion ~11.5).
    3. PASS_YDS_CONSISTENCY: If Σ rec_yds projections >> QB pass_yds projection,
       the model is hallucinating volume that doesn't exist at the team level.

    Args:
        gaps: list of gap dicts from find_prop_gaps() (must include 'team' field)
        consensus: optional consensus dict for supplemental team mapping
        tolerance: fractional tolerance for violations (0.15 = ±15%)

    Returns:
        list of violation dicts with type, team, details, severity
    """
    team_data = _group_by_team(gaps, consensus)
    violations = []

    for team, tdata in team_data.items():
        qb_entries = tdata["qb_pass_yds"]
        rec_entries = tdata["rec_yds"]
        rec_count_entries = tdata["receptions"]

        if not qb_entries:
            continue

        # Use the QB with the highest projected pass_yds (starter)
        qb_name, qb_pass_proj, qb_pass_line = max(qb_entries, key=lambda x: x[1])

        # ── Violation 1: Σ rec_yds vs QB pass_yds ──
        if rec_entries:
            sum_rec_proj = sum(e[1] for e in rec_entries)
            sum_rec_lines = sum(e[2] for e in rec_entries)

            # Ratio: how much total receiver yds exceeds QB passing yds
            ratio_proj = sum_rec_proj / qb_pass_proj if qb_pass_proj > 0 else 0
            ratio_line = sum_rec_lines / qb_pass_line if qb_pass_line > 0 else 0

            # Flag if Σ rec_yds > QB pass_yds × (1 + tolerance)
            # Or if Σ rec_yds < QB pass_yds × (1 - tolerance) × coverage_factor
            # Note: not all receivers have props, so Σ rec_yds < QB pass_yds is expected.
            # We only flag OVER-projection (sum exceeds QB).
            if ratio_proj > (1 + tolerance):
                severity = "critical" if ratio_proj > 1.3 else "warning"
                violations.append({
                    "type": "SUM_REC_YDS_VS_QB_PASS_YDS",
                    "team": team,
                    "severity": severity,
                    "qb": qb_name,
                    "qb_pass_proj": round(qb_pass_proj, 1),
                    "qb_pass_line": round(qb_pass_line, 1),
                    "sum_rec_proj": round(sum_rec_proj, 1),
                    "sum_rec_lines": round(sum_rec_lines, 1),
                    "ratio_proj": round(ratio_proj, 3),
                    "ratio_line": round(ratio_line, 3),
                    "n_receivers": len(rec_entries),
                    "receivers": [(e[0], round(e[1], 1), round(e[2], 1)) for e in rec_entries],
                    "detail": (f"Σ rec_yds proj ({sum_rec_proj:.0f}) = "
                               f"{ratio_proj:.0%} of QB pass_yds proj ({qb_pass_proj:.0f}). "
                               f"Market: {ratio_line:.0%} ({sum_rec_lines:.0f}/{qb_pass_line:.0f})."),
                })

            # ── Also check: model vs market divergence direction ──
            # If model says ratio_proj > ratio_line, model is inflating team receiving
            if ratio_proj > 0 and ratio_line > 0:
                divergence = ratio_proj - ratio_line
                if abs(divergence) > 0.10:
                    violations.append({
                        "type": "REC_YDS_RATIO_DIVERGENCE",
                        "team": team,
                        "severity": "info",
                        "ratio_proj": round(ratio_proj, 3),
                        "ratio_line": round(ratio_line, 3),
                        "divergence": round(divergence, 3),
                        "detail": (f"Model rec/pass ratio ({ratio_proj:.1%}) differs from "
                                   f"market ({ratio_line:.1%}) by {divergence:+.1%}."),
                    })

        # ── Violation 2: Σ receptions vs implied completions ──
        if rec_count_entries and qb_pass_proj > 0:
            sum_rec_count_proj = sum(e[1] for e in rec_count_entries)
            # Implied completions from QB pass_yds / ~11.5 Y/C (league avg)
            implied_completions = qb_pass_proj / 11.5
            rec_to_comp_ratio = sum_rec_count_proj / implied_completions if implied_completions > 0 else 0

            # Σ receptions can't exceed total completions.
            # Again, not all receivers have props, so < 1.0 is expected.
            if rec_to_comp_ratio > (1 + tolerance):
                violations.append({
                    "type": "SUM_RECEPTIONS_VS_COMPLETIONS",
                    "team": team,
                    "severity": "warning",
                    "qb": qb_name,
                    "qb_pass_proj": round(qb_pass_proj, 1),
                    "implied_completions": round(implied_completions, 1),
                    "sum_receptions_proj": round(sum_rec_count_proj, 1),
                    "ratio": round(rec_to_comp_ratio, 3),
                    "n_receivers": len(rec_count_entries),
                    "detail": (f"Σ receptions proj ({sum_rec_count_proj:.0f}) = "
                               f"{rec_to_comp_ratio:.0%} of implied completions "
                               f"({implied_completions:.0f} from {qb_pass_proj:.0f} pass_yds / 11.5 Y/C)."),
                })

    return violations


def detect_correlated_disagreement(gaps: list[dict],
                                   min_same_dir: int = 3,
                                   min_edge_pp: float = 0.03) -> list[dict]:
    """
    Flag when ≥min_same_dir players on the same team all deviate from market
    in the same direction. This suggests a team-level input error, not
    independent player-level edges.

    Args:
        gaps: list of gap dicts from find_prop_gaps()
        min_same_dir: minimum players in same direction to flag (default 3)
        min_edge_pp: minimum |edge_vs_market| to count a player (default 3pp)

    Returns:
        list of correlated disagreement flags
    """
    # Group by team, only main lines, deduplicated per (player, stat)
    team_dirs = defaultdict(lambda: {"over": [], "under": []})
    seen = set()

    for g in gaps:
        team = _canon_team(g.get("team", ""))
        if not team:
            continue
        if g.get("line_depth", "main") != "main":
            continue
        edge = g.get("edge_vs_market")
        if edge is None or abs(edge) < min_edge_pp:
            continue

        # Deduplicate: one entry per (team, player, stat)
        dedup_key = (team, g["player"], g["stat"])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        direction = "over" if edge > 0 else "under"
        team_dirs[team][direction].append({
            "player": g["player"],
            "stat": g["stat"],
            "position": g.get("position", ""),
            "edge_pp": round(edge * 100, 1),
            "consensus_mean": g["consensus_mean"],
            "prop_line": g["prop_line"],
        })

    flags = []
    for team, dirs in team_dirs.items():
        for direction, players in dirs.items():
            # Count UNIQUE players (not player×stat combos)
            unique_players = set(p["player"] for p in players)
            n_unique = len(unique_players)
            if n_unique >= min_same_dir:
                avg_edge = sum(p["edge_pp"] for p in players) / len(players)
                flags.append({
                    "team": team,
                    "direction": direction,
                    "n_unique_players": n_unique,
                    "n_player_stats": len(players),
                    "avg_edge_pp": round(avg_edge, 1),
                    "players": players,
                    "detail": (f"{team}: {n_unique} unique players all {direction.upper()} "
                               f"({len(players)} player×stat, "
                               f"avg edge {avg_edge:+.1f}pp). "
                               f"Probable team-level input error."),
                })

    # Sort by number of unique correlated players descending
    flags.sort(key=lambda x: x["n_unique_players"], reverse=True)
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# Edge capping for correlated clusters (shadow arm: capped vs uncapped)
# ══════════════════════════════════════════════════════════════════════════════

# Default cap: cluster edges get halved. This is a conservative starting point.
# If a cluster of 5 players all show +20pp edge, each gets capped to +10pp.
# Rationale: independent edges would be additive; correlated edges from a
# shared input error should be discounted toward the team-level uncertainty.
CLUSTER_EDGE_SCALE = 0.50


def cap_correlated_edges(gaps: list[dict],
                         min_same_dir: int = 3,
                         min_edge_pp: float = 0.03,
                         scale: float = CLUSTER_EDGE_SCALE) -> list[dict]:
    """
    Cap edge_vs_market for predictions inside correlated clusters.

    When ≥min_same_dir unique players on the same team deviate from market in
    the same direction, scale each player's edge by `scale` (default 0.50).

    This creates a "capped" version of the gaps for shadow comparison.
    The original gaps are not modified — returns a new list.

    Args:
        gaps: list of gap dicts from find_prop_gaps()
        min_same_dir: min unique players in same direction to trigger cap
        min_edge_pp: min |edge| to count toward cluster detection
        scale: multiplier applied to edge_vs_market for cluster members (0.50 = halve)

    Returns:
        New list of gap dicts with capped edges. Each dict gets:
        - 'edge_vs_market_uncapped': original edge
        - 'in_correlated_cluster': bool
        - 'cluster_team': team if in cluster
        - 'cluster_n_unique': number of unique players in the cluster
    """
    # Identify clusters (same logic as detect_correlated_disagreement)
    team_dirs = defaultdict(lambda: {"over": set(), "under": set()})
    for g in gaps:
        team = _canon_team(g.get("team", ""))
        if not team:
            continue
        if g.get("line_depth", "main") != "main":
            continue
        edge = g.get("edge_vs_market")
        if edge is None or abs(edge) < min_edge_pp:
            continue
        direction = "over" if edge > 0 else "under"
        team_dirs[team][direction].add(g["player"])

    # Build set of (team, direction) pairs that are clusters
    cluster_keys = set()
    cluster_sizes = {}
    for team, dirs in team_dirs.items():
        for direction, players in dirs.items():
            if len(players) >= min_same_dir:
                cluster_keys.add((team, direction))
                cluster_sizes[(team, direction)] = len(players)

    # Build capped copy
    capped = []
    n_capped = 0
    for g in gaps:
        gc = dict(g)  # shallow copy
        team = _canon_team(g.get("team", ""))
        edge = g.get("edge_vs_market")
        gc["edge_vs_market_uncapped"] = edge

        if edge is not None and team:
            direction = "over" if edge > 0 else "under"
            key = (team, direction)
            if key in cluster_keys and g.get("line_depth", "main") == "main":
                gc["edge_vs_market"] = round(edge * scale, 4)
                gc["in_correlated_cluster"] = True
                gc["cluster_team"] = team
                gc["cluster_n_unique"] = cluster_sizes[key]
                n_capped += 1
            else:
                gc["in_correlated_cluster"] = False
                gc["cluster_team"] = None
                gc["cluster_n_unique"] = 0
        else:
            gc["in_correlated_cluster"] = False
            gc["cluster_team"] = None
            gc["cluster_n_unique"] = 0

        capped.append(gc)

    logger.info("[EdgeCap] Capped %d/%d gaps (scale=%.2f, %d clusters)",
                n_capped, len(gaps), scale, len(cluster_keys))
    return capped


# ══════════════════════════════════════════════════════════════════════════════
# Historical violation analysis on stored forward predictions
# ══════════════════════════════════════════════════════════════════════════════

def _load_forward_preds_with_team(season: int, week: int) -> list[dict]:
    """
    Load forward predictions and reconstruct team from raw projections or props.

    Since fantasy_forward_preds doesn't store team, we join with
    fantasy_projections (which has team from Sleeper) to get team assignment.
    """
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    # Get forward preds (main line, distributional method only)
    preds = conn.execute("""
        SELECT player_name, position, stat_type, consensus_mean,
               prop_line, direction, p_over, edge_vs_market,
               line_depth, actual_value, is_win
        FROM fantasy_forward_preds
        WHERE season = ? AND week = ?
              AND method = 'distributional'
              AND line_depth = 'main'
    """, (season, week)).fetchall()

    # Get team mapping from raw projections (Sleeper has clean team abbrevs)
    team_map = {}
    for row in conn.execute("""
        SELECT DISTINCT player_name, team FROM fantasy_projections
        WHERE season = ? AND week = ? AND source = 'sleeper' AND team != ''
    """, (season, week)):
        team_map[row["player_name"]] = _canon_team(row["team"])

    conn.close()

    results = []
    for p in preds:
        team = team_map.get(p["player_name"], "")
        results.append({
            "player": p["player_name"],
            "position": p["position"],
            "team": team,
            "stat": p["stat_type"],
            "consensus_mean": p["consensus_mean"],
            "prop_line": p["prop_line"],
            "direction": p["direction"],
            "p_over": p["p_over"],
            "edge_vs_market": p["edge_vs_market"],
            "line_depth": p["line_depth"],
            "actual_value": p["actual_value"],
            "is_win": p["is_win"],
        })

    return results


def run_violation_report(season: int | None = None, week: int | None = None,
                         historical: bool = False) -> dict:
    """
    Orchestrate violation detection and produce a formatted report.

    Two modes:
    1. Live: run on current consensus/gaps (requires running the pipeline first)
    2. Historical: run on stored forward predictions for a given season/week

    Args:
        season, week: NFL season/week. If None, uses current.
        historical: if True, load from stored forward predictions.

    Returns:
        dict with violations, correlated_disagreements, and summary stats
    """
    if historical:
        return _run_historical_report(season, week)
    else:
        return _run_live_report(season, week)


def _run_historical_report(season: int | None, week: int | None) -> dict:
    """Run violation report on stored forward predictions."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))

    # Find available season/weeks
    available = conn.execute("""
        SELECT DISTINCT season, week FROM fantasy_forward_preds
        ORDER BY season, week
    """).fetchall()
    conn.close()

    if not available:
        print("[Diagnostics] No forward predictions found")
        return {"error": "No forward predictions available"}

    # If season/week specified, run single. Otherwise run all.
    if season is not None and week is not None:
        targets = [(season, week)]
    else:
        targets = [(r[0], r[1]) for r in available]

    all_results = []
    total_violations = 0
    total_corr_flags = 0
    total_team_weeks = 0

    for s, w in targets:
        preds = _load_forward_preds_with_team(s, w)
        if not preds:
            continue

        violations = detect_team_violations(preds)
        corr_flags = detect_correlated_disagreement(preds)

        # Count unique teams this week
        teams_this_week = len(set(p["team"] for p in preds if p["team"]))
        total_team_weeks += teams_this_week
        total_violations += len(violations)
        total_corr_flags += len(corr_flags)

        all_results.append({
            "season": s, "week": w,
            "n_preds": len(preds),
            "n_teams": teams_this_week,
            "violations": violations,
            "correlated_disagreements": corr_flags,
        })

    # ── Format report ──
    print("\n" + "=" * 80)
    print("  NFL PROP MODEL — TEAM-LEVEL VIOLATION REPORT")
    print("=" * 80)

    for r in all_results:
        print(f"\n{'─' * 60}")
        print(f"  Season {r['season']} Week {r['week']}: "
              f"{r['n_preds']} predictions across {r['n_teams']} teams")
        print(f"{'─' * 60}")

        if r["violations"]:
            print(f"\n  VIOLATIONS ({len(r['violations'])}):")
            for v in r["violations"]:
                sev_marker = "🔴" if v["severity"] == "critical" else "🟡" if v["severity"] == "warning" else "ℹ️"
                print(f"\n    {sev_marker} [{v['type']}] {v['team']}")
                print(f"       {v['detail']}")
                if "receivers" in v:
                    for name, proj, line in v["receivers"]:
                        print(f"         {name:<20} proj={proj:>6.1f}  line={line:>6.1f}")
        else:
            print("\n  No accounting violations detected.")

        if r["correlated_disagreements"]:
            print(f"\n  CORRELATED DISAGREEMENTS ({len(r['correlated_disagreements'])}):")
            for cd in r["correlated_disagreements"]:
                print(f"\n    ⚠️  {cd['detail']}")
                for p in cd["players"]:
                    print(f"         {p['player']:<20} {p['stat']:<12} edge={p['edge_pp']:+.1f}pp")
        else:
            print("\n  No correlated disagreements detected.")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Weeks analyzed:           {len(all_results)}")
    print(f"  Total team-weeks:         {total_team_weeks}")
    print(f"  Total violations:         {total_violations}")
    print(f"  Total corr. disagreements: {total_corr_flags}")
    if total_team_weeks > 0:
        # Violation rate per team-week
        viol_per_tw = total_violations / total_team_weeks
        print(f"  Violations per team-week: {viol_per_tw:.2f}")
    print()

    return {
        "weeks": all_results,
        "summary": {
            "n_weeks": len(all_results),
            "total_team_weeks": total_team_weeks,
            "total_violations": total_violations,
            "total_correlated_disagreements": total_corr_flags,
            "violations_per_team_week": (total_violations / total_team_weeks
                                         if total_team_weeks > 0 else 0),
        },
    }


def _run_live_report(season: int | None, week: int | None) -> dict:
    """Run violation report on live consensus/gaps (requires pipeline data)."""
    try:
        from fantasy_projections import run_weekly_pipeline, get_current_nfl_week
    except ImportError:
        print("[Diagnostics] Cannot import fantasy_projections — run from backend/")
        return {"error": "Cannot import fantasy_projections"}

    if season is None or week is None:
        s, w = get_current_nfl_week()
        season = season or s
        week = week or w

    print(f"[Diagnostics] Running pipeline for {season} week {week}...")
    result = run_weekly_pipeline(season, week, fetch_props=False)

    gaps = result.get("all_gaps", [])
    consensus = result.get("consensus", {})

    violations = detect_team_violations(gaps, consensus)
    corr_flags = detect_correlated_disagreement(gaps)

    # Print results
    teams_with_data = len(set(g.get("team", "") for g in gaps if g.get("team")))
    print(f"\n[Diagnostics] {len(violations)} violations, "
          f"{len(corr_flags)} correlated disagreements "
          f"across {teams_with_data} teams")

    for v in violations:
        print(f"  [{v['severity'].upper()}] {v['type']} — {v['team']}: {v['detail']}")

    for cd in corr_flags:
        print(f"  [CORR] {cd['detail']}")

    return {
        "season": season, "week": week,
        "violations": violations,
        "correlated_disagreements": corr_flags,
        "n_teams": teams_with_data,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Backtest plan (walk-forward structure, not implemented yet — design only)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_reconciliation_plan() -> str:
    """
    Return the walk-forward backtest design for the reconciliation model.

    This is NOT executable yet — it documents the validation protocol
    that will be used once the reconciliation engine (Phase 2) is built.
    """
    plan = """
    ════════════════════════════════════════════════════════════════
    RECONCILIATION BACKTEST PLAN (walk-forward, game-clustered)
    ════════════════════════════════════════════════════════════════

    Structure:
    - Training window: weeks [1..w-1] of the current season
    - Prediction window: week w
    - Expand training window each week (expanding window, not rolling)
    - Seasons: 2023-2025 (all 18 regular season weeks)

    Training components per window:
    1. Team volume regression: total_plays, pass_rate = f(spread, total)
       trained on nfl_team_game_stats + nfl_games
    2. Target shares: rolling 6-game PBP receiver shares
       (shrunk toward position-tier average for <10 targets)
    3. Completion rate, Y/A, Y/C from team rolling 8-game history

    Prediction for week w:
    - Project team volume from game lines (spread, total)
    - Compute target shares from weeks [1..w-1]
    - Run constrained allocation (QP)
    - Fit distributions with reconciled means
    - Generate P(over) for each player-prop

    Clustering:
    - Unit = game, NOT player
    - Within-game predictions are correlated (shared volume pool)
    - Paired bootstrap: per-game RMSE(reconciled) - RMSE(independent)
    - 10,000 bootstrap resamples of games
    - Report: mean difference, 95% CI, p-value

    Metrics:
    - Primary: MAE of consensus_mean vs actual_value (reconciled vs independent)
    - Secondary: calibration by P(over) bucket [0.30-0.40, 0.40-0.50, ..., 0.70-0.80]
    - Tertiary: Brier score on is_win outcome
    - Segment: backup-QB games, high-spread games (|spread| > 7)

    Controls:
    - TB/DAL Week 5 2026 = regression test case, NOT in training
    - No parameter tuning on this game
    - All parameters frozen before walk-forward begins

    Acceptance criteria:
    - Reconciled MAE must be lower than independent MAE with p < 0.05
      (paired bootstrap, game-clustered)
    - No calibration degradation in any bucket
    - If neutral or worse: do NOT deploy reconciliation
    """
    print(plan)
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="NFL Prop Model Violation Diagnostics")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--historical", action="store_true",
                        help="Run on stored forward predictions")
    parser.add_argument("--backtest-plan", action="store_true",
                        help="Print the walk-forward backtest design")
    args = parser.parse_args()

    if args.backtest_plan:
        backtest_reconciliation_plan()
    else:
        run_violation_report(args.season, args.week, historical=args.historical)
