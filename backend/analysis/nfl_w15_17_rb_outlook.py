#!/usr/bin/env python3
"""
W15–17 game-script and RB outlook for KC, TB, JAX.

Generates:
  data/reports/nfl_w15_17_kc_tb_jax_2026-11-18.html
  data/reports/nfl_w15_17_kc_tb_jax_2026-11-18.csv
  Email via researcher.delivery.send_email (if SMTP configured)

Usage:
    conda run -n betiq python backend/analysis/nfl_w15_17_rb_outlook.py
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
_DATA = os.path.join(_ROOT, "data")
_NFL_DB = os.path.join(_DATA, "nfl.db")
_HIST_DB = os.path.join(_DATA, "historical.db")
_REPORT_DIR = os.path.join(_DATA, "reports")
os.makedirs(_REPORT_DIR, exist_ok=True)

TEAMS = ["KC", "TB", "JAX"]
WEEKS = [15, 16, 17]
SEASON = 2026
N_SIMS = 10_000
SEED = 42
NOW_UTC = datetime.now(timezone.utc)

# Target RBs
TARGET_RBS = {
    "KC": ["Kenneth Walker III"],
    "TB": ["Bucky Irving"],
    "JAX": ["Bhayshul Tuten"],
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def load_schedule():
    """Load W15–17 games involving KC, TB, JAX from nfl.db."""
    con = sqlite3.connect(_NFL_DB, timeout=10)
    con.row_factory = sqlite3.Row
    placeholders = ",".join(["?"] * len(TEAMS))
    rows = con.execute(f"""
        SELECT game_id, season, week, game_date, weekday, gametime,
               home_team, away_team, spread_line, total_line,
               roof, surface, stadium, home_rest, away_rest, div_game
        FROM nfl_games
        WHERE season = ? AND week BETWEEN 15 AND 17
          AND (home_team IN ({placeholders}) OR away_team IN ({placeholders}))
        ORDER BY week, game_date, gametime
    """, [SEASON] + TEAMS + TEAMS).fetchall()
    con.close()

    # Dedup (if two target teams play each other)
    seen_ids = set()
    games = []
    for r in rows:
        if r["game_id"] in seen_ids:
            continue
        seen_ids.add(r["game_id"])
        g = dict(r)
        # Convert gametime to CT (ET - 1h for display)
        try:
            et_hour = int(g["gametime"].split(":")[0])
            et_min = int(g["gametime"].split(":")[1])
            ct_hour = et_hour - 1
            g["kickoff_ct"] = f"{ct_hour}:{et_min:02d} CT"
        except Exception:
            g["kickoff_ct"] = g["gametime"] or "TBD"
        games.append(g)
    return games


# ══════════════════════════════════════════════════════════════════════════════
# 2. INJURY / PLAYER CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def load_injury_context():
    """Load latest injury status for target RBs + teammates."""
    con = sqlite3.connect(_HIST_DB, timeout=10)
    con.row_factory = sqlite3.Row

    injuries = {}
    for team, players in TARGET_RBS.items():
        for pname in players:
            row = con.execute("""
                SELECT player_name, new_status, injury_body_part, detected_at
                FROM injury_status_log
                WHERE player_name = ?
                ORDER BY id DESC LIMIT 1
            """, (pname,)).fetchone()
            if row:
                injuries[pname] = {
                    "status": row["new_status"] or "Active",
                    "injury": row["injury_body_part"] or "",
                    "detected": row["detected_at"] or "",
                }
            else:
                injuries[pname] = {"status": "Active", "injury": "", "detected": ""}

    # Also check nfl.db injuries table for most recent week
    con2 = sqlite3.connect(_NFL_DB, timeout=10)
    con2.row_factory = sqlite3.Row
    for team in TEAMS:
        rows = con2.execute("""
            SELECT player_name, position, injury, status
            FROM nfl_injuries
            WHERE team = ? AND season = 2026
            ORDER BY week DESC
        """, (team,)).fetchall()
        # Group by player, take latest
        seen = set()
        for r in rows:
            if r["player_name"] not in seen and r["position"] == "RB":
                seen.add(r["player_name"])
                if r["player_name"] not in injuries:
                    injuries[r["player_name"]] = {
                        "status": r["status"] or "Active",
                        "injury": r["injury"] or "",
                        "detected": "",
                        "team": team,
                    }
    con.close()
    con2.close()
    return injuries


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEAM EFFICIENCY (derive spreads/totals)
# ══════════════════════════════════════════════════════════════════════════════

def load_team_efficiency():
    """Load team efficiency from nfl_team_game_stats (2025 as proxy if no 2026)."""
    con = sqlite3.connect(_NFL_DB, timeout=10)
    con.row_factory = sqlite3.Row

    # Try 2026 first, then 2025
    for szn in [2026, 2025]:
        rows = con.execute("""
            SELECT team,
                   AVG(off_epa_per_play) as off_epa,
                   AVG(def_epa_per_play) as def_epa,
                   AVG(score) as ppg,
                   AVG(opp_score) as opp_ppg,
                   AVG(pass_rate) as pass_rate,
                   AVG(rush_epa_per_play) as rush_epa,
                   AVG(pass_epa_per_play) as pass_epa,
                   AVG(def_rush_epa) as def_rush_epa,
                   AVG(def_pass_epa) as def_pass_epa,
                   COUNT(*) as games
            FROM nfl_team_game_stats
            WHERE season = ?
            GROUP BY team
            HAVING games >= 4
        """, (szn,)).fetchall()
        if rows:
            logger.info(f"Using {szn} team efficiency data ({len(rows)} teams)")
            eff = {r["team"]: dict(r) for r in rows}
            con.close()
            return eff, szn

    con.close()
    logger.warning("No team efficiency data available")
    return {}, None


def derive_spread_total(home, away, eff, hfa=2.5):
    """Derive spread and total from team efficiency.

    Returns (spread, total, source_label).
    Negative spread = home favored.
    """
    if not eff or home not in eff or away not in eff:
        return 0.0, 45.0, "league-average (no team data)"

    h = eff[home]
    a = eff[away]

    # Points expectation: team PPG adjusted for opponent defense
    lg_ppg = np.mean([t["ppg"] for t in eff.values()])
    h_pts = h["ppg"] * (a["opp_ppg"] / lg_ppg)
    a_pts = a["ppg"] * (h["opp_ppg"] / lg_ppg)

    # Home field advantage
    h_pts += hfa / 2
    a_pts -= hfa / 2

    total = h_pts + a_pts
    spread = -(h_pts - a_pts)  # negative = home favored

    return round(spread * 2) / 2, round(total * 2) / 2, "model-derived (opponent-adjusted PPG)"


# ══════════════════════════════════════════════════════════════════════════════
# 4. GAME SIMULATION (simplified quarter-by-quarter Monte Carlo)
# ══════════════════════════════════════════════════════════════════════════════

# League-average pass rates by game state (from nfl_game_sim.py empirical data)
PASS_RATE_BY_STATE = {
    "neutral": 0.596,       # Q1-Q3, |diff| <= 8
    "trailing_late": 0.766, # Q4, diff <= -4
    "leading_late": 0.399,  # Q4, diff >= 4
    "big_trail": 0.646,     # Q1-Q3, diff < -8
    "big_lead": 0.571,      # Q1-Q3, diff > 8
}

# RB rush share increase when leading (empirical from 2023-2025 PBP)
# Leading by 10+: rush rate jumps ~15pp, RB1 share increases ~8%
RB1_SHARE_BOOST_LEADING = 0.08
RUSH_RATE_BOOST_LEADING = 0.15


def simulate_games(schedule, eff, eff_season):
    """Run quarter-by-quarter game simulations for all scheduled games.

    Returns list of game result dicts.
    """
    rng = np.random.default_rng(SEED)
    results = []

    for game in schedule:
        home = game["home_team"]
        away = game["away_team"]
        spread, total, spread_src = derive_spread_total(home, away, eff)

        # Use stored spread/total if available (nflverse lookahead)
        if game["spread_line"] and abs(game["spread_line"]) > 0:
            spread = game["spread_line"]
            spread_src = f"nflverse (season={eff_season})"
        if game["total_line"] and game["total_line"] > 0:
            total = game["total_line"]

        game["spread"] = spread
        game["total"] = total
        game["spread_src"] = spread_src

        # Implied points
        home_implied = (total / 2) - (spread / 2)
        away_implied = (total / 2) + (spread / 2)

        # Per-quarter scoring: NFL scoring is roughly 23/25/24/28 pct by quarter
        qtr_pcts = np.array([0.23, 0.25, 0.24, 0.28])

        # Sim arrays
        home_q = np.zeros((N_SIMS, 4))
        away_q = np.zeros((N_SIMS, 4))

        for i in range(N_SIMS):
            # Game-level random effects
            game_eff = rng.normal(0, 0.12)  # shared scoring environment
            h_eff = rng.normal(home_implied * (1 + game_eff), home_implied * 0.08)
            a_eff = rng.normal(away_implied * (1 + game_eff), away_implied * 0.08)

            for q in range(4):
                # Quarter scoring: gamma-distributed (always >= 0)
                h_q_mean = max(0.5, h_eff * qtr_pcts[q])
                a_q_mean = max(0.5, a_eff * qtr_pcts[q])

                # NFL scoring is lumpy (0, 3, 7 point chunks), use gamma for shape
                h_shape = max(0.5, h_q_mean / 3.5)
                a_shape = max(0.5, a_q_mean / 3.5)
                home_q[i, q] = rng.gamma(h_shape, 3.5)
                away_q[i, q] = rng.gamma(a_shape, 3.5)

        # Cumulative scores at each quarter break
        home_cum = np.cumsum(home_q, axis=1)
        away_cum = np.cumsum(away_q, axis=1)
        margin = home_cum - away_cum  # positive = home leading

        # Compute requested metrics
        qtr_metrics = []
        for q in range(4):
            m = margin[:, q]
            abs_m = np.abs(m)
            qtr_metrics.append({
                "quarter": q + 1,
                "home_pts_mean": float(home_q[:, q].mean()),
                "away_pts_mean": float(away_q[:, q].mean()),
                "home_cum_mean": float(home_cum[:, q].mean()),
                "away_cum_mean": float(away_cum[:, q].mean()),
                "margin_median": float(np.median(m)),
                "margin_p10": float(np.percentile(m, 10)),
                "margin_p90": float(np.percentile(m, 90)),
                "p_one_score": float(np.mean(abs_m <= 8)),
            })

        # Additional game-level metrics
        p_lead14_half = float(np.mean(np.abs(margin[:, 1]) >= 14))
        p_lead17_q4 = float(np.mean(np.abs(margin[:, 2]) >= 17))

        # P(halftime leader loses)
        ht_leader = np.sign(margin[:, 1])
        final_leader = np.sign(margin[:, 3])
        ht_had_lead = ht_leader != 0
        p_ht_leader_loses = float(
            np.mean((ht_leader[ht_had_lead] != final_leader[ht_had_lead]))
        ) if ht_had_lead.sum() > 0 else 0.0

        # Classification
        close_all_qtrs = all(qm["p_one_score"] >= 0.55 for qm in qtr_metrics)
        p_blowout_q3 = float(np.mean(np.abs(margin[:, 2]) >= 14))
        likely_leader = ""
        if p_blowout_q3 >= 0.40:
            # Which team more likely leading?
            home_leading = float(np.mean(margin[:, 2] >= 14))
            away_leading = float(np.mean(margin[:, 2] <= -14))
            likely_leader = home if home_leading > away_leading else away

        if close_all_qtrs:
            classification = "close throughout"
        elif p_blowout_q3 >= 0.40:
            classification = f"likely pull-away ({likely_leader})"
        else:
            classification = "volatile"

        # Closeness score for ranking (lower = closer)
        closeness = np.mean([qm["p_one_score"] for qm in qtr_metrics])

        # Game-script states for RB conditioning
        # Halftime margin bins
        ht_margin = margin[:, 1]
        team_up_10 = {}
        team_within_1score = {}
        team_down_10 = {}
        for tm in [home, away]:
            if tm == home:
                tm_margin = ht_margin
            else:
                tm_margin = -ht_margin
            team_up_10[tm] = tm_margin >= 10
            team_within_1score[tm] = np.abs(tm_margin) < 10
            team_down_10[tm] = tm_margin <= -10

        result = {
            "game": game,
            "home": home,
            "away": away,
            "spread": spread,
            "total": total,
            "spread_src": spread_src,
            "qtr_metrics": qtr_metrics,
            "p_lead14_half": p_lead14_half,
            "p_lead17_q4": p_lead17_q4,
            "p_ht_leader_loses": p_ht_leader_loses,
            "classification": classification,
            "closeness": closeness,
            "likely_leader": likely_leader,
            # For RB conditioning
            "home_q": home_q,
            "away_q": away_q,
            "margin": margin,
            "team_up_10": team_up_10,
            "team_within_1score": team_within_1score,
            "team_down_10": team_down_10,
        }
        results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. RB PROJECTIONS (conditioned on game script)
# ══════════════════════════════════════════════════════════════════════════════

def load_rb_baselines():
    """Load most recent forward pred projections for target RBs."""
    con = sqlite3.connect(_HIST_DB, timeout=10)
    con.row_factory = sqlite3.Row

    baselines = {}
    for team, players in TARGET_RBS.items():
        for pname in players:
            rows = con.execute("""
                SELECT stat_type, prop_line, consensus_mean, consensus_sd,
                       p_over, p_over_v12_gamma, devigged_implied, week
                FROM fantasy_forward_preds
                WHERE player_name = ? AND season = 2026 AND line_depth = 'main'
                  AND method = 'distributional'
                ORDER BY week DESC
            """, (pname,)).fetchall()

            # Take most recent week per stat
            seen_stats = set()
            player_lines = {}
            for r in rows:
                if r["stat_type"] not in seen_stats:
                    seen_stats.add(r["stat_type"])
                    player_lines[r["stat_type"]] = {
                        "line": r["prop_line"],
                        "proj": r["consensus_mean"],
                        "sd": r["consensus_sd"] or 0,
                        "p_over": r["p_over"],
                        "p_over_gamma": r["p_over_v12_gamma"],
                        "devigged": r["devigged_implied"],
                        "week": r["week"],
                    }
            baselines[pname] = {"team": team, "lines": player_lines}

    con.close()
    return baselines


def simulate_rb_stats(sim_result, player_name, team, baselines, rng):
    """Generate RB stat distributions conditioned on game script.

    Uses the team's total scoring + game state to modulate volume.
    """
    pdata = baselines.get(player_name, {}).get("lines", {})

    # Base projections from forward preds
    rush_proj = pdata.get("rush_yds", {}).get("proj", 60.0)
    rush_sd = max(15.0, pdata.get("rush_yds", {}).get("sd", 20.0) * 3)
    rec_yds_proj = pdata.get("rec_yds", {}).get("proj", 15.0)
    rec_sd = max(8.0, pdata.get("rec_yds", {}).get("sd", 10.0) * 3)
    rec_proj = pdata.get("receptions", {}).get("proj", 2.5)

    # Rush attempts: derive from rush_yds / typical YPC
    ypc_mean = 4.5
    rush_att_mean = rush_proj / ypc_mean
    targets_mean = rec_proj / 0.75  # catch rate ~ 75% for RBs

    is_home = (team == sim_result["home"])
    margin = sim_result["margin"]  # (N_SIMS, 4), positive = home leading
    team_margin = margin if is_home else -margin

    # Overall stats (10K draws)
    n = N_SIMS
    overall = {
        "rush_att": np.zeros(n),
        "rush_yds": np.zeros(n),
        "targets": np.zeros(n),
        "receptions": np.zeros(n),
        "rec_yds": np.zeros(n),
    }

    # By-half stats
    h1 = {k: np.zeros(n) for k in overall}
    h2 = {k: np.zeros(n) for k in overall}

    for i in range(n):
        # Halftime margin for this sim
        ht_margin = team_margin[i, 1]
        # Q4 margin
        q3_margin = team_margin[i, 2]

        # Volume modulation by game script
        # Leading big: more rushes, fewer targets
        # Trailing big: fewer rushes, more targets
        if ht_margin >= 10:
            rush_mult_h2 = 1.25
            target_mult_h2 = 0.75
        elif ht_margin <= -10:
            rush_mult_h2 = 0.70
            target_mult_h2 = 1.30
        else:
            rush_mult_h2 = 1.0
            target_mult_h2 = 1.0

        # First half: neutral script assumption
        h1_rush_att = max(0, rng.poisson(rush_att_mean * 0.45))
        h1_targets = max(0, rng.poisson(targets_mean * 0.45))
        h1_rush_yds = max(0, rng.normal(h1_rush_att * ypc_mean, h1_rush_att * 2.5)) if h1_rush_att > 0 else 0
        h1_rec = rng.binomial(h1_targets, 0.75)
        h1_rec_yds = max(0, rng.normal(h1_rec * (rec_yds_proj / max(1, rec_proj)), h1_rec * 5)) if h1_rec > 0 else 0

        # Second half: script-modulated
        h2_rush_att = max(0, rng.poisson(rush_att_mean * 0.55 * rush_mult_h2))
        h2_targets = max(0, rng.poisson(targets_mean * 0.55 * target_mult_h2))
        h2_rush_yds = max(0, rng.normal(h2_rush_att * ypc_mean, h2_rush_att * 2.5)) if h2_rush_att > 0 else 0
        h2_rec = rng.binomial(h2_targets, 0.75)
        h2_rec_yds = max(0, rng.normal(h2_rec * (rec_yds_proj / max(1, rec_proj)), h2_rec * 5)) if h2_rec > 0 else 0

        h1["rush_att"][i] = h1_rush_att
        h1["rush_yds"][i] = h1_rush_yds
        h1["targets"][i] = h1_targets
        h1["receptions"][i] = h1_rec
        h1["rec_yds"][i] = h1_rec_yds

        h2["rush_att"][i] = h2_rush_att
        h2["rush_yds"][i] = h2_rush_yds
        h2["targets"][i] = h2_targets
        h2["receptions"][i] = h2_rec
        h2["rec_yds"][i] = h2_rec_yds

        for k in overall:
            overall[k][i] = h1[k][i] + h2[k][i]

    def _stats(arr):
        return {
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean()),
        }

    # Conditional on halftime state
    up10 = sim_result["team_up_10"][team]
    within = sim_result["team_within_1score"][team]
    down10 = sim_result["team_down_10"][team]

    def _cond_stats(arr_h2, mask):
        if mask.sum() < 50:
            return {"median": None, "p10": None, "p90": None, "mean": None, "n": int(mask.sum())}
        sub = arr_h2[mask]
        return {
            "median": float(np.median(sub)),
            "p10": float(np.percentile(sub, 10)),
            "p90": float(np.percentile(sub, 90)),
            "mean": float(sub.mean()),
            "n": int(mask.sum()),
        }

    # P(over) at reference lines
    p_overs = {}
    for stat, arr in [("rush_yds", overall["rush_yds"]),
                       ("receptions", overall["receptions"]),
                       ("rec_yds", overall["rec_yds"])]:
        line_data = pdata.get(stat, {})
        line = line_data.get("line")
        if line is not None:
            p_over = float(np.mean(arr > line))
            p_under = 1.0 - p_over
            p_overs[stat] = {
                "line": line,
                "p_over": p_over,
                "p_under": p_under,
                "source_week": line_data.get("week"),
            }

    return {
        "overall": {k: _stats(v) for k, v in overall.items()},
        "h1": {k: _stats(v) for k, v in h1.items()},
        "h2": {k: _stats(v) for k, v in h2.items()},
        "h2_up10": {k: _cond_stats(h2[k], up10) for k in h2},
        "h2_within": {k: _cond_stats(h2[k], within) for k in h2},
        "h2_down10": {k: _cond_stats(h2[k], down10) for k in h2},
        "p_overs": p_overs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. OPPONENT RUN DEFENSE
# ══════════════════════════════════════════════════════════════════════════════

def load_run_defense(eff):
    """Extract opponent run defense metrics."""
    if not eff:
        return {}
    defense = {}
    for team, data in eff.items():
        defense[team] = {
            "def_rush_epa": data.get("def_rush_epa", 0),
            "def_epa": data.get("def_epa", 0),
        }
    return defense


# ══════════════════════════════════════════════════════════════════════════════
# 7. REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def format_pct(v):
    return f"{v:.1%}" if v is not None else "—"

def format_f1(v):
    return f"{v:.1f}" if v is not None else "—"

def format_f0(v):
    return f"{v:.0f}" if v is not None else "—"


def build_report(schedule, sim_results, rb_stats, injuries, eff, eff_season, run_def):
    """Build the full report as markdown/text for email and HTML for file."""

    lines = []
    csv_rows = []

    # ── Executive summary ──
    # Rank games by closeness
    ranked = sorted(sim_results, key=lambda x: -x["closeness"])

    lines.append("# W15–17 KC/TB/JAX Game Script & RB Outlook")
    lines.append(f"Generated: {NOW_UTC.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Sims: {N_SIMS:,} per game | Seed: {SEED}")
    lines.append(f"Team efficiency source: {eff_season or 'league average'}")
    lines.append("")

    # Executive summary paragraph
    close_games = [r for r in ranked if r["classification"] == "close throughout"]
    pullaway_games = [r for r in ranked if "pull-away" in r["classification"]]
    volatile_games = [r for r in ranked if r["classification"] == "volatile"]

    summary_parts = []
    summary_parts.append(
        f"Across 9 team-games (W15–17), {len(close_games)} project as close throughout, "
        f"{len(pullaway_games)} as likely pull-aways, and {len(volatile_games)} as volatile."
    )

    # Top RB takeaway
    for pname in ["Bucky Irving", "Kenneth Walker III", "Bhayshul Tuten"]:
        if pname in rb_stats and rb_stats[pname]:
            best_game = max(rb_stats[pname].items(),
                           key=lambda x: x[1]["overall"]["rush_yds"]["median"])
            gid, stats = best_game
            summary_parts.append(
                f"{pname}: best rushing outlook in {gid} "
                f"(median {stats['overall']['rush_yds']['median']:.0f} rush yds)."
            )

    walker_inj = injuries.get("Kenneth Walker III", {})
    if walker_inj.get("status") == "Out":
        summary_parts.append(
            f"Kenneth Walker III remains OUT (groin) as of "
            f"{walker_inj.get('detected', 'unknown')}; "
            f"projections assume he returns for W15 (with 'remains out' scenario noted)."
        )

    lines.append("## Executive Summary")
    lines.append(" ".join(summary_parts))
    lines.append("")

    # ── Game ranking ──
    lines.append("## Game Ranking (closest to most lopsided)")
    lines.append("")
    lines.append("| Rank | Week | Game | Spread | Total | Classification | Avg P(≤8) |")
    lines.append("|------|------|------|--------|-------|----------------|-----------|")
    for i, r in enumerate(ranked, 1):
        g = r["game"]
        matchup = f"{g['away_team']}@{g['home_team']}"
        lines.append(
            f"| {i} | W{g['week']} | {matchup} | {r['spread']:+.1f} | {r['total']:.1f} | "
            f"{r['classification']} | {r['closeness']:.1%} |"
        )
        csv_rows.append({
            "section": "game_ranking", "rank": i, "week": g["week"],
            "game": matchup, "spread": r["spread"], "total": r["total"],
            "classification": r["classification"], "avg_p_one_score": r["closeness"],
        })
    lines.append("")

    # ── Per-game quarter tables ──
    lines.append("## Per-Game Quarter-by-Quarter Projections")
    for r in sorted(sim_results, key=lambda x: (x["game"]["week"], x["game"]["game_date"])):
        g = r["game"]
        matchup = f"{g['away_team']} @ {g['home_team']}"
        lines.append(f"\n### W{g['week']} — {matchup}")
        lines.append(f"Date: {g['game_date']} ({g['weekday']}) {g['kickoff_ct']}")
        lines.append(f"Stadium: {g['stadium']} | Roof: {g['roof']} | Surface: {g['surface']}")
        lines.append(f"Spread: {r['spread']:+.1f} | Total: {r['total']:.1f} ({r['spread_src']})")
        lines.append(f"Classification: **{r['classification']}**")
        lines.append("")

        lines.append(f"| Quarter | {g['home_team']} pts | {g['away_team']} pts | "
                     f"{g['home_team']} cum | {g['away_team']} cum | Margin (med) | "
                     f"Margin 10–90th | P(≤8) |")
        lines.append("|---------|----------|----------|----------|----------|-------------|"
                     "---------------|-------|")

        for qm in r["qtr_metrics"]:
            lines.append(
                f"| Q{qm['quarter']} | {qm['home_pts_mean']:.1f} | {qm['away_pts_mean']:.1f} | "
                f"{qm['home_cum_mean']:.1f} | {qm['away_cum_mean']:.1f} | "
                f"{qm['margin_median']:+.1f} | "
                f"[{qm['margin_p10']:+.1f}, {qm['margin_p90']:+.1f}] | "
                f"{qm['p_one_score']:.1%} |"
            )
            csv_rows.append({
                "section": "quarter", "week": g["week"],
                "game": f"{g['away_team']}@{g['home_team']}",
                "quarter": qm["quarter"],
                "home_pts": qm["home_pts_mean"], "away_pts": qm["away_pts_mean"],
                "home_cum": qm["home_cum_mean"], "away_cum": qm["away_cum_mean"],
                "margin_median": qm["margin_median"],
                "margin_p10": qm["margin_p10"], "margin_p90": qm["margin_p90"],
                "p_one_score": qm["p_one_score"],
            })

        lines.append("")
        lines.append(f"P(either team ≥14 at half): {r['p_lead14_half']:.1%}")
        lines.append(f"P(lead ≥17 entering Q4): {r['p_lead17_q4']:.1%}")
        lines.append(f"P(halftime leader loses): {r['p_ht_leader_loses']:.1%}")
        lines.append("")

    # ── RB Breakdowns ──
    lines.append("## RB Breakdowns")
    for pname in ["Bucky Irving", "Kenneth Walker III", "Bhayshul Tuten"]:
        inj = injuries.get(pname, {})
        team = None
        for t, players in TARGET_RBS.items():
            if pname in players:
                team = t
                break

        lines.append(f"\n### {pname} ({team})")
        lines.append(f"Injury: {inj.get('status', 'Active')}"
                     f"{' — ' + inj.get('injury', '') if inj.get('injury') else ''}")
        if pname == "Kenneth Walker III" and inj.get("status") == "Out":
            lines.append("**Note**: Walker was OUT (Groin) through W11. Projections below "
                        "assume he returns for W15. If he remains out, his carries redistribute "
                        "to Kareem Hunt / Clyde Edwards-Helaire (not modeled individually).")
        lines.append("")

        if pname not in rb_stats or not rb_stats[pname]:
            lines.append("(No simulation data available)")
            continue

        for game_key, stats in sorted(rb_stats[pname].items()):
            lines.append(f"#### {game_key}")

            # Overall
            lines.append("**Full game:**")
            lines.append("| Stat | Median | 10th | 90th |")
            lines.append("|------|--------|------|------|")
            for stat_name, label in [("rush_att", "Rush Att"), ("rush_yds", "Rush Yds"),
                                      ("targets", "Targets"), ("receptions", "Receptions"),
                                      ("rec_yds", "Rec Yds")]:
                s = stats["overall"][stat_name]
                lines.append(f"| {label} | {s['median']:.1f} | {s['p10']:.1f} | {s['p90']:.1f} |")
                csv_rows.append({
                    "section": "rb_overall", "player": pname, "game": game_key,
                    "stat": stat_name, "half": "full",
                    "median": s["median"], "p10": s["p10"], "p90": s["p90"],
                })

            # By half
            lines.append("")
            lines.append("**By half:**")
            lines.append("| Stat | H1 med | H1 10-90 | H2 med | H2 10-90 |")
            lines.append("|------|--------|----------|--------|----------|")
            for stat_name, label in [("rush_att", "Rush Att"), ("rush_yds", "Rush Yds"),
                                      ("receptions", "Rec"), ("rec_yds", "Rec Yds")]:
                h1s = stats["h1"][stat_name]
                h2s = stats["h2"][stat_name]
                lines.append(
                    f"| {label} | {h1s['median']:.1f} | [{h1s['p10']:.0f}–{h1s['p90']:.0f}] | "
                    f"{h2s['median']:.1f} | [{h2s['p10']:.0f}–{h2s['p90']:.0f}] |"
                )

            # Conditional on halftime state
            lines.append("")
            lines.append("**H2 conditioned on halftime state:**")
            lines.append("| State | N sims | H2 Rush Att | H2 Rush Yds | H2 Rec |")
            lines.append("|-------|--------|-------------|-------------|--------|")
            for state_key, state_label in [("h2_up10", "Up 10+"), ("h2_within", "Within 1 score"),
                                            ("h2_down10", "Down 10+")]:
                ra = stats[state_key]["rush_att"]
                ry = stats[state_key]["rush_yds"]
                rc = stats[state_key]["receptions"]
                n_s = ra.get("n", 0)
                lines.append(
                    f"| {state_label} | {n_s:,} | {format_f1(ra['median'])} | "
                    f"{format_f1(ry['median'])} | {format_f1(rc['median'])} |"
                )
                csv_rows.append({
                    "section": "rb_conditional", "player": pname, "game": game_key,
                    "state": state_label, "n_sims": n_s,
                    "h2_rush_att_med": ra.get("median"), "h2_rush_yds_med": ry.get("median"),
                    "h2_rec_med": rc.get("median"),
                })

            # P(over) at reference lines
            if stats["p_overs"]:
                lines.append("")
                lines.append("**P(over) at reference lines:**")
                lines.append("| Stat | Line | P(over) | P(under) | Line from |")
                lines.append("|------|------|---------|----------|-----------|")
                for stat, po in stats["p_overs"].items():
                    lines.append(
                        f"| {stat} | {po['line']:.1f} | {po['p_over']:.1%} | "
                        f"{po['p_under']:.1%} | W{po['source_week']} |"
                    )
                    csv_rows.append({
                        "section": "rb_p_over", "player": pname, "game": game_key,
                        "stat": stat, "line": po["line"],
                        "p_over": po["p_over"], "p_under": po["p_under"],
                    })

                lines.append("*Note: Rush yds OVERs have shown systematic over-projection "
                            "in our scorecard (W2-W10). UNDERs may carry better value.*")

            lines.append("")

    # ── Opponent run defense ──
    lines.append("## Opponent Run Defense")
    lines.append("| Team | Def Rush EPA/play | Def EPA/play | Note |")
    lines.append("|------|-------------------|-------------|------|")

    # Collect all opponents
    opps = set()
    for r in sim_results:
        for tm in TEAMS:
            if tm == r["home"]:
                opps.add(r["away"])
            elif tm == r["away"]:
                opps.add(r["home"])

    for opp in sorted(opps):
        d = run_def.get(opp, {})
        rush_epa = d.get("def_rush_epa", None)
        def_epa = d.get("def_epa", None)
        note = ""
        if rush_epa is not None:
            if rush_epa < -0.05:
                note = "Strong run D"
            elif rush_epa > 0.05:
                note = "Weak run D"
        lines.append(
            f"| {opp} | {format_f1(rush_epa) if rush_epa else '—'} | "
            f"{format_f1(def_epa) if def_epa else '—'} | {note} |"
        )
    lines.append(f"*Source: nfl_team_game_stats {eff_season or 'N/A'}*")
    lines.append("")

    # ── Caveats ──
    lines.append("## Caveats")
    lines.append(f"- **Data sources**: nfl.db schedule (nflverse), "
                f"historical.db forward preds (Sleeper+ESPN consensus), "
                f"nfl_team_game_stats ({eff_season or 'unavailable'}).")
    lines.append(f"- **Timestamps**: Report generated {NOW_UTC.strftime('%Y-%m-%d %H:%M UTC')}. "
                f"Forward preds are from most recent available week per player.")
    lines.append(f"- **Market lines**: {'Model-derived from opponent-adjusted PPG' if eff else 'League average (no team data)'}. "
                f"No sportsbook W15-17 lines available yet — refresh when lines post.")
    lines.append("- **Injury assumptions**: Kenneth Walker III is OUT (Groin) through W11; "
                "projections assume return by W15. Bucky Irving and Bhayshul Tuten assumed healthy.")
    lines.append("- **Weather**: Not modeled. KC (Arrowhead) is outdoor December — monitor for "
                "cold/wind. TB (Raymond James) is outdoor but warm. JAX varies by venue.")
    lines.append("- **Week 17 rest risk**: If a team has clinched seeding by W17, starters may "
                "rest. KC most likely to rest if they've secured the 1 seed. Flag for refresh.")
    lines.append("- **Sim limitations**: Simplified quarter-level Monte Carlo (not play-by-play). "
                "No OT, no penalties, no special teams scoring. Market-anchored when lines exist. "
                "RB volume modulation uses empirical pass-rate shifts, not play-by-play drives.")
    lines.append("- **Rush yds over-projection**: Our scorecard (W2-W10) shows systematic "
                "over-projection of rush yds, particularly for OVER sides. Weight UNDERs accordingly.")
    lines.append("- **Refresh**: Re-run this analysis when each week's lines post for market-anchored sims.")

    return "\n".join(lines), csv_rows


def build_html(md_text):
    """Convert markdown-ish text to basic HTML."""
    html = ["<html><head><meta charset='utf-8'>",
            "<style>",
            "body{font-family:monospace;max-width:1200px;margin:auto;padding:20px;background:#1a1a2e;color:#e0e0e0;}",
            "h1{color:#e94560;} h2{color:#0f3460;background:#16213e;padding:8px;border-radius:4px;color:#e0e0e0;}",
            "h3{color:#53e0d0;} h4{color:#a8d8ea;}",
            "table{border-collapse:collapse;width:100%;margin:10px 0;}",
            "th,td{border:1px solid #444;padding:6px 10px;text-align:right;}",
            "th{background:#16213e;color:#e0e0e0;} td{background:#1a1a2e;}",
            "td:first-child,th:first-child{text-align:left;}",
            ".close{color:#4ecca3;} .pullaway{color:#e94560;} .volatile{color:#f9a825;}",
            "</style></head><body>"]

    for line in md_text.split("\n"):
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("#### "):
            html.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("|"):
            # Table row
            if "---" in line:
                continue  # skip separator
            cells = [c.strip() for c in line.split("|")[1:-1]]
            tag = "th" if html and "<table>" in html[-1] else "td"
            # First table row after a non-table = new table with headers
            if not any("<table>" in h for h in html[-3:]):
                html.append("<table>")
                tag = "th"
            row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            html.append(f"<tr>{row}</tr>")
        elif line.startswith("*"):
            html.append(f"<p><em>{line.strip('* ')}</em></p>")
        elif line.startswith("- "):
            html.append(f"<li>{line[2:]}</li>")
        elif line.startswith("**"):
            html.append(f"<p><strong>{line.strip('*')}</strong></p>")
        else:
            if html and "<table>" in html[-5:] and not line.strip():
                html.append("</table>")
            if line.strip():
                html.append(f"<p>{line}</p>")

    html.append("</body></html>")
    return "\n".join(html)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("W15-17 KC/TB/JAX Game Script & RB Outlook")
    logger.info("=" * 70)

    errors = []

    # 1. Schedule
    logger.info("Loading schedule...")
    schedule = load_schedule()
    logger.info(f"Found {len(schedule)} games")
    for g in schedule:
        logger.info(f"  W{g['week']} {g['game_date']} {g['away_team']}@{g['home_team']} "
                    f"({g['stadium']}, {g['roof']})")

    # 2. Injuries
    logger.info("Loading injury context...")
    injuries = load_injury_context()
    for name, inj in injuries.items():
        if name in [p for ps in TARGET_RBS.values() for p in ps]:
            logger.info(f"  {name}: {inj.get('status', '?')} ({inj.get('injury', '')})")

    # 3. Team efficiency
    logger.info("Loading team efficiency...")
    eff, eff_season = load_team_efficiency()
    if eff:
        logger.info(f"Using {eff_season} data for {len(eff)} teams")
    else:
        errors.append("No team efficiency data available; using league averages")
        logger.warning(errors[-1])

    # 4. Run defense
    run_def = load_run_defense(eff)

    # 5. Game simulations
    logger.info(f"Running {N_SIMS:,} sims per game...")
    sim_results = simulate_games(schedule, eff, eff_season)
    for r in sim_results:
        logger.info(f"  W{r['game']['week']} {r['away']}@{r['home']}: "
                    f"spread={r['spread']:+.1f} total={r['total']:.1f} "
                    f"→ {r['classification']}")

    # 6. RB baselines
    logger.info("Loading RB baselines...")
    baselines = load_rb_baselines()

    # 7. RB simulations
    logger.info("Simulating RB stats...")
    rng = np.random.default_rng(SEED + 1)
    rb_stats = {}
    for pname in [p for ps in TARGET_RBS.values() for p in ps]:
        team = baselines.get(pname, {}).get("team")
        if not team:
            errors.append(f"No team found for {pname}")
            continue

        rb_stats[pname] = {}
        for r in sim_results:
            if team in (r["home"], r["away"]):
                game_key = f"W{r['game']['week']} {r['away']}@{r['home']}"
                stats = simulate_rb_stats(r, pname, team, baselines, rng)
                rb_stats[pname][game_key] = stats
                logger.info(f"  {pname} {game_key}: "
                           f"rush_yds median={stats['overall']['rush_yds']['median']:.0f}")

    # 8. Build report
    logger.info("Building report...")
    md_text, csv_rows = build_report(
        schedule, sim_results, rb_stats, injuries, eff, eff_season, run_def
    )

    # 9. Save files
    html_path = os.path.join(_REPORT_DIR, "nfl_w15_17_kc_tb_jax_2026-11-18.html")
    csv_path = os.path.join(_REPORT_DIR, "nfl_w15_17_kc_tb_jax_2026-11-18.csv")

    html_content = build_html(md_text)
    with open(html_path, "w") as f:
        f.write(html_content)
    logger.info(f"Saved HTML: {html_path}")

    # CSV
    if csv_rows:
        all_keys = set()
        for row in csv_rows:
            all_keys.update(row.keys())
        all_keys = sorted(all_keys)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Saved CSV: {csv_path}")

    # 10. Email
    logger.info("Attempting email delivery...")
    try:
        sys.path.insert(0, os.path.join(_HERE, ".."))
        from researcher.delivery import send_email

        # Add errors to email body if any
        email_body = md_text
        if errors:
            email_body += "\n\n## Errors / Missing\n"
            for e in errors:
                email_body += f"- {e}\n"

        result = send_email(
            subject="BetIQ: W15–17 KC/TB/JAX game script + RB outlook",
            body_md=email_body,
        )
        if result.get("sent"):
            logger.info(f"Email sent: {result}")
        else:
            logger.warning(f"Email not sent: {result}")
            errors.append(f"Email delivery failed: {result.get('reason') or result.get('error', 'unknown')}")
    except Exception as exc:
        logger.error(f"Email error: {exc}")
        errors.append(f"Email error: {exc}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info(f"HTML: {html_path}")
    logger.info(f"CSV:  {csv_path}")
    if errors:
        logger.info("ERRORS:")
        for e in errors:
            logger.info(f"  - {e}")
    logger.info("=" * 70)

    return errors


if __name__ == "__main__":
    main()
