"""
wc_market_anchored.py — Market-Anchored WC 2026 Prediction Model

Philosophy: instead of predicting from scratch, anchor to devigged bookmaker
implied probabilities and apply small structural adjustments where Pi ratings,
chemistry, dependency, league quality, or form data identify genuine edge.

Target: agree with the market ~80 % of the time; diverge on the 15-20 % where
structural factors create actionable edge.

Sections
--------
1. compute_adjustments(row)        — per-match structural adjustments
2. compute_anchored_probs(row)     — adjusted 3-way probabilities
3. Historical walk-forward backtest (WC 2014, 2018, 2022)
4. Full WC 2026 match predictions
5. Monte Carlo tournament simulation (10 000 sims)
6. main()                         — runs everything, saves JSON
"""
from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ─── Team-name normalisation (shared with wc_pressure_test_v2) ───────────────

_NAME_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Korea Republic": "South Korea", "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States", "Republic of Ireland": "Ireland",
    "IR Iran": "Iran", "Korea DPR": "North Korea",
    "Chinese Taipei": "Taiwan",
    "Curaçao": "Curacao",
}


def _norm(n: str) -> str:
    return _NAME_ALIASES.get(n, n)


def _lj(fn):
    with open(os.path.join(_DATA_DIR, fn)) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Structural adjustments
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adjustments(row: dict) -> dict:
    """Compute small structural adjustments for a single match row.

    Each adjustment is on the order of -0.05 to +0.05, representing how much
    to nudge the home-win probability away from the market line.
    """
    adjustments = {}

    # ── Pi adjustment — only when Pi meaningfully disagrees with market ──
    pi_delta = float(row.get("pi_delta", 0))
    pi_prob = 1.0 / (1.0 + math.exp(-0.25 * pi_delta))
    market_home = float(row.get("odds_home_implied", 0.33))
    pi_edge = pi_prob - market_home
    adjustments["pi_adj"] = pi_edge * 0.12          # directional but noisy; keep at 0.12

    # ── Chemistry — quality-weighted chemistry delta ──
    qw_chem = float(row.get("qw_chemistry_delta", 0))
    # Range ~ -322 to +325; scale to roughly ±0.04
    # 2022 backtest: +15 net directional signal (23 helps, 8 hurts) — 2nd strongest
    adjustments["chemistry_adj"] = (qw_chem / 325.0) * 0.04  # +15 net directional — keep

    # ── Dependency ratio — star-dependent teams carry more variance risk ──
    dep_delta = float(row.get("dependency_ratio_delta", 0))
    # Range ~ -0.32 to +0.29; higher = home more dependent = riskier
    # 2022 backtest: 0 net signal (16/16) — reduce weight
    adjustments["dependency_adj"] = -dep_delta * 0.005  # 0 net signal — halved from 0.01

    # ── League quality delta ──
    lq_delta = float(row.get("league_quality_delta", 0))
    # Range ~ -1.84 to +1.96
    # 2022 backtest: +22 net directional signal (34 helps, 12 hurts) — strongest
    adjustments["league_quality_adj"] = (lq_delta / 2.0) * 0.035  # +6% lift validated; bumped from 0.03

    # ── Form — recent results momentum ──
    form_wp_delta = float(row.get("form_win_pct_delta", 0))
    # 2022 backtest: +11 net signal (28 helps, 17 hurts) — solid
    adjustments["form_adj"] = form_wp_delta * 0.05  # +7% lift validated; bumped from 0.04

    # ── Altitude — high-altitude venues (Mexico) penalise non-acclimatised teams ──
    venue_alt = float(row.get("venue_alt_ft", 0) or 0)
    if venue_alt > 4000:
        # Home team is the FIFA-designated "home" (not necessarily acclimatised)
        # But CONCACAF teams (Mexico especially) train at altitude
        # Small penalty to the away team's chances (nudge home up)
        adjustments["altitude_adj"] = 0.02 * (venue_alt / 7349)  # max ~+0.02 for Mexico City
    else:
        adjustments["altitude_adj"] = 0.0

    # ── Rest delta — teams with less rest are slightly disadvantaged ──
    rest_delta = float(row.get("rest_delta", 0) or 0)
    # rest_delta = home_rest - away_rest; positive = home more rested
    # Range: -2 to +2; scale to roughly ±0.01
    adjustments["rest_adj"] = rest_delta * 0.005

    # ── xG proxy — goals scored/conceded per game in recent competitive matches ──
    xg_diff_delta = float(row.get("xg_diff_delta", 0) or 0)
    # Range: roughly -3 to +3; positive = home has better goal diff rate
    # Small weight — this is a proxy, not true xG
    adjustments["xg_adj"] = (xg_diff_delta / 3.0) * 0.01   # 0% lift — halved from 0.02

    # ── Squad depth — scoring concentration (star_share) is a fragility risk ──
    star_share_delta = float(row.get("star_share_delta", 0) or 0)
    # Positive = home more star-dependent = home more fragile
    # Range: roughly -0.5 to +0.5
    squad_goal_rate_delta = float(row.get("squad_goal_rate_delta", 0) or 0)
    # Positive = home has higher squad goal rate = better squad quality
    # Range: roughly -0.3 to +0.3
    # 2022 backtest: -1 net signal (23/24) — reduce both components
    adjustments["squad_depth_adj"] = (
        -star_share_delta * 0.005        # -4% lift — cut to minimal (was 0.01)
        + squad_goal_rate_delta * 0.005  # -4% lift — cut to minimal (was 0.015)
    )

    return adjustments


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Anchored probability computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_anchored_probs(row: dict) -> dict:
    """Anchor to market implied probs and apply structural adjustments.

    The feature-matrix odds are devigged per-bookmaker and should sum to ~1.0.
    We renormalise as a safety net, then apply structural nudges.
    """
    raw_home = float(row.get("odds_home_implied", 0.33))
    raw_draw = float(row.get("odds_draw_implied", 0.25))
    raw_away = float(row.get("odds_away_implied", 0.20))

    # Renormalise market baseline to 1.0
    raw_total = raw_home + raw_draw + raw_away
    if raw_total <= 0:
        raw_total = 1.0
    base_home = raw_home / raw_total
    base_draw = raw_draw / raw_total
    base_away = raw_away / raw_total

    adjustments = compute_adjustments(row)
    total_adj = sum(adjustments.values())

    # Apply: positive total_adj → home more likely
    adj_home = base_home + total_adj
    adj_draw = base_draw - total_adj * 0.3       # draw absorbs less
    adj_away = base_away - total_adj * 0.7       # away absorbs more

    # Clamp to sensible ranges
    adj_home = max(0.05, min(0.90, adj_home))
    adj_draw = max(0.05, min(0.40, adj_draw))
    adj_away = max(0.05, min(0.90, adj_away))

    # ── WC draw calibration ─────────────────────────────────────────────
    # Dynamic draw floor based on match lopsidedness.  Heavy favorites
    # (>75% one side) legitimately have very low draw probability (~5%),
    # while evenly matched games warrant a higher floor (~15%).
    max_side = max(adj_home, adj_away)
    if max_side > 0.75:
        _draw_floor = 0.05   # heavy favorite — minimal draw floor
    elif max_side > 0.60:
        _draw_floor = 0.10   # moderate favorite
    else:
        _draw_floor = 0.15   # evenly matched — draws more likely

    if adj_draw < _draw_floor:
        deficit = _draw_floor - adj_draw
        ha_total = adj_home + adj_away
        if ha_total > 0:
            adj_home -= deficit * (adj_home / ha_total)
            adj_away -= deficit * (adj_away / ha_total)
            adj_draw = _draw_floor

    gap = abs(adj_home - adj_away)
    if gap < 0.10:
        draw_boost = 0.03 * (1.0 - gap / 0.10)   # up to +3 pp
        ha_total = adj_home + adj_away
        if ha_total > 0:
            adj_draw += draw_boost
            adj_home -= draw_boost * (adj_home / ha_total)
            adj_away -= draw_boost * (adj_away / ha_total)

    # Renormalise to sum to 1.0
    total = adj_home + adj_draw + adj_away
    return {
        "home_win": round(adj_home / total, 4),
        "draw": round(adj_draw / total, 4),
        "away_win": round(adj_away / total, 4),
        "adjustments": adjustments,
        "total_adj": round(total_adj, 4),
        "base_home": round(base_home, 4),
        "base_draw": round(base_draw, 4),
        "base_away": round(base_away, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Historical walk-forward backtest  (WC 2014, 2018, 2022)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_pi_ratings(results_df, before_date, lookback_years=4):
    """Compute opponent-weighted Pi ratings from scratch."""
    cutoff = pd.Timestamp(before_date)
    start = cutoff - pd.DateOffset(years=lookback_years)
    df = results_df[(results_df["date"] >= start) & (results_df["date"] < cutoff)]

    ratings = {}
    K_BASE = 0.1
    for _, row in df.iterrows():
        home, away = _norm(str(row["home_team"])), _norm(str(row["away_team"]))
        margin = float(row["home_score"]) - float(row["away_score"])
        tourn = str(row.get("tournament", "")).lower()
        is_comp = any(kw in tourn for kw in
                      ["world cup", "euro", "copa america", "nations league",
                       "african cup", "asian cup", "gold cup", "confederations",
                       "qualification"])
        importance = 1.0 if is_comp else 0.5

        for t in [home, away]:
            if t not in ratings:
                ratings[t] = {"home_pi": 0.0, "away_pi": 0.0, "games": 0}

        hc = ratings[home]["home_pi"] + ratings[home]["away_pi"]
        ac = ratings[away]["home_pi"] + ratings[away]["away_pi"]
        k_h = K_BASE * importance * (1 + max(ac, 0) / 5.0)
        k_a = K_BASE * importance * (1 + max(hc, 0) / 5.0)
        ratings[home]["home_pi"] += k_h * margin
        ratings[home]["games"] += 1
        ratings[away]["away_pi"] -= k_a * margin
        ratings[away]["games"] += 1
    return ratings


def _form_features(results_df, team, before_date, n=10):
    t = _norm(team)
    cutoff = pd.Timestamp(before_date)
    df = results_df[results_df["date"] < cutoff]
    mask = (df["home_team"].apply(_norm) == t) | (df["away_team"].apply(_norm) == t)
    tdf = df[mask].sort_values("date", ascending=False).head(n)

    if len(tdf) == 0:
        return {"form_win_pct": 0.5, "form_gf_avg": 1.2, "form_ga_avg": 1.0}

    w, gf, ga = 0, 0, 0
    for _, row in tdf.iterrows():
        h = _norm(str(row["home_team"]))
        hs, as_ = float(row["home_score"]), float(row["away_score"])
        g_for, g_ag = (hs, as_) if h == t else (as_, hs)
        gf += g_for
        ga += g_ag
        if g_for > g_ag:
            w += 1

    m = len(tdf)
    return {
        "form_win_pct": round(w / m, 4),
        "form_gf_avg": round(gf / m, 4),
        "form_ga_avg": round(ga / m, 4),
    }


def _outcome_3way(hs, as_):
    """0 = away win, 1 = draw, 2 = home win."""
    if hs > as_:
        return 2
    elif hs == as_:
        return 1
    else:
        return 0


def _build_backtest_row(pi_cache, results_df, home, away, cutoff):
    """Build a pseudo-feature-matrix row suitable for compute_adjustments.

    Since we lack historical bookmaker odds, we compute a v2-style Pi-based
    3-way probability as the 'market proxy' and then let the anchored model
    nudge it with structural adjustments.
    """
    h, a = _norm(home), _norm(away)
    hp = pi_cache.get(h, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
    ap = pi_cache.get(a, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
    hc = hp["home_pi"] + hp["away_pi"]
    ac = ap["home_pi"] + ap["away_pi"]
    pi_delta = hc - ac

    # Pi-based 3-way proxy "market"
    strength = 1.0 / (1.0 + math.exp(-0.25 * pi_delta))
    draw_base = 0.27 * math.exp(-0.04 * pi_delta ** 2)
    draw_base = max(0.12, min(0.38, draw_base))
    remaining = 1.0 - draw_base
    p_hw = remaining * strength
    p_aw = remaining * (1.0 - strength)
    p_dr = draw_base

    hf = _form_features(results_df, home, cutoff)
    af = _form_features(results_df, away, cutoff)

    form_wp_delta = hf["form_win_pct"] - af["form_win_pct"]

    return {
        "pi_delta": pi_delta,
        "odds_home_implied": p_hw,
        "odds_draw_implied": p_dr,
        "odds_away_implied": p_aw,
        "qw_chemistry_delta": 0.0,      # not available historically
        "dependency_ratio_delta": 0.0,   # not available historically
        "league_quality_delta": 0.0,     # not available historically
        "form_win_pct_delta": form_wp_delta,
    }


def run_historical_backtest(results_df):
    """Walk-forward backtest for WC 2014, 2018, 2022.

    Uses the Pi-based 3-way probability as a proxy 'market', then applies
    the anchored adjustment mechanism on top.  This tests whether the
    adjustment mechanism improves predictions over the base proxy.
    """
    print("\n" + "=" * 75, flush=True)
    print("  HISTORICAL BACKTEST — MARKET-ANCHORED ADJUSTMENTS", flush=True)
    print("  (Pi proxy as 'market', adjustments on top)", flush=True)
    print("=" * 75, flush=True)

    tournaments = [
        {"year": 2014, "cutoff": "2014-06-12"},
        {"year": 2018, "cutoff": "2018-06-14"},
        {"year": 2022, "cutoff": "2022-11-20"},
    ]

    all_results = {}
    comb_base_hw, comb_adj_hw = [], []
    comb_actual_3 = []
    comb_agree_correct, comb_agree_total = 0, 0
    comb_disagree_model_correct, comb_disagree_market_correct = 0, 0
    comb_disagree_total = 0

    for t in tournaments:
        year, cutoff = t["year"], t["cutoff"]
        print(f"\n{'─' * 70}", flush=True)
        print(f"  WC {year}", flush=True)
        print(f"{'─' * 70}", flush=True)

        wc_df = results_df[results_df["tournament"] == "FIFA World Cup"]
        wc = wc_df[wc_df["date"].dt.year == year].reset_index(drop=True)
        pi_cache = _compute_pi_ratings(results_df, cutoff)

        base_correct_3, adj_correct_3, n_matches = 0, 0, 0
        base_brier, adj_brier = 0.0, 0.0

        for _, m in wc.iterrows():
            home, away = str(m["home_team"]), str(m["away_team"])
            actual = _outcome_3way(float(m["home_score"]), float(m["away_score"]))
            row = _build_backtest_row(pi_cache, results_df, home, away, cutoff)

            # Base (proxy market) probabilities
            b_hw = row["odds_home_implied"]
            b_dr = row["odds_draw_implied"]
            b_aw = row["odds_away_implied"]

            # Anchored probabilities
            anch = compute_anchored_probs(row)
            a_hw = anch["home_win"]
            a_dr = anch["draw"]
            a_aw = anch["away_win"]

            # 3-way prediction: highest probability class
            base_pred = max([(b_hw, 2), (b_dr, 1), (b_aw, 0)], key=lambda x: x[0])[1]
            adj_pred = max([(a_hw, 2), (a_dr, 1), (a_aw, 0)], key=lambda x: x[0])[1]

            if base_pred == actual:
                base_correct_3 += 1
            if adj_pred == actual:
                adj_correct_3 += 1

            # Multi-class Brier score
            for cls_val, bp, ap_ in [(2, b_hw, a_hw), (1, b_dr, a_dr), (0, b_aw, a_aw)]:
                ind = 1.0 if actual == cls_val else 0.0
                base_brier += (bp - ind) ** 2
                adj_brier += (ap_ - ind) ** 2

            n_matches += 1

            comb_base_hw.append(b_hw)
            comb_adj_hw.append(a_hw)
            comb_actual_3.append(actual)

            # Track edge performance: does model disagree with market by >5 % (home)?
            edge = a_hw - b_hw
            if abs(edge) > 0.05:
                comb_disagree_total += 1
                model_side = 2 if edge > 0 else 0  # model favours home or away
                market_side = 2 if b_hw > b_aw else 0
                if actual == model_side:
                    comb_disagree_model_correct += 1
                if actual == market_side:
                    comb_disagree_market_correct += 1
            else:
                comb_agree_total += 1
                if adj_pred == actual:
                    comb_agree_correct += 1

        base_acc = base_correct_3 / max(n_matches, 1)
        adj_acc = adj_correct_3 / max(n_matches, 1)
        base_brier /= max(n_matches, 1)
        adj_brier /= max(n_matches, 1)

        print(f"  Matches: {n_matches}", flush=True)
        print(f"  Base (proxy market) 3-way acc: {base_acc:.1%}  Brier: {base_brier:.4f}", flush=True)
        print(f"  Anchored           3-way acc: {adj_acc:.1%}  Brier: {adj_brier:.4f}", flush=True)
        delta_acc = adj_acc - base_acc
        delta_brier = adj_brier - base_brier
        print(f"  Delta:             acc {delta_acc:+.1%}  Brier {delta_brier:+.4f}", flush=True)

        all_results[year] = {
            "matches": n_matches,
            "base_acc": round(base_acc, 4),
            "adj_acc": round(adj_acc, 4),
            "base_brier": round(base_brier, 4),
            "adj_brier": round(adj_brier, 4),
        }

    # ── Combined metrics ──
    total = len(comb_actual_3)
    comb_actual_3 = np.array(comb_actual_3)
    comb_base_hw = np.array(comb_base_hw)
    comb_adj_hw = np.array(comb_adj_hw)

    corr_base_adj = np.corrcoef(comb_base_hw, comb_adj_hw)[0, 1]

    print(f"\n{'─' * 70}", flush=True)
    print(f"  COMBINED ({total} matches)", flush=True)
    print(f"{'─' * 70}", flush=True)
    print(f"  Base-vs-Anchored home-win correlation: {corr_base_adj:.4f}", flush=True)

    if comb_disagree_total > 0:
        model_edge_rate = comb_disagree_model_correct / comb_disagree_total
        market_edge_rate = comb_disagree_market_correct / comb_disagree_total
        print(f"\n  Edge performance (model disagrees with market by >5 %):", flush=True)
        print(f"    Disagreement matches: {comb_disagree_total}", flush=True)
        print(f"    Model's side correct: {comb_disagree_model_correct}/{comb_disagree_total} = {model_edge_rate:.1%}", flush=True)
        print(f"    Market's side correct: {comb_disagree_market_correct}/{comb_disagree_total} = {market_edge_rate:.1%}", flush=True)
    else:
        model_edge_rate = 0.0
        market_edge_rate = 0.0

    if comb_agree_total > 0:
        agree_acc = comb_agree_correct / comb_agree_total
        print(f"\n  Agreement matches ({comb_agree_total}): accuracy = {agree_acc:.1%}", flush=True)
    else:
        agree_acc = 0.0

    all_results["combined"] = {
        "total_matches": total,
        "base_adj_corr": round(corr_base_adj, 4),
        "disagree_count": comb_disagree_total,
        "model_edge_rate": round(model_edge_rate, 4),
        "market_edge_rate": round(market_edge_rate, 4),
        "agree_accuracy": round(agree_acc, 4),
    }

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Full WC 2026 match predictions
# ═══════════════════════════════════════════════════════════════════════════════

def run_wc2026_predictions():
    """Apply anchored model to all 72 WC 2026 group-stage matches."""
    print("\n\n" + "=" * 75, flush=True)
    print("  WC 2026 — MARKET-ANCHORED MATCH PREDICTIONS", flush=True)
    print("=" * 75, flush=True)

    feat_path = os.path.join(_DATA_DIR, "wc2026_feature_matrix.csv")
    if not os.path.exists(feat_path):
        print("  [SKIP] Feature matrix not found", flush=True)
        return [], {}

    df = pd.read_csv(feat_path)
    print(f"\n  Loaded {len(df)} matches from feature matrix", flush=True)

    match_results = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        group = str(row.get("group", "?"))
        commence = str(row.get("commence_time", ""))

        anch = compute_anchored_probs(row.to_dict())

        match_results.append({
            "home_team": home,
            "away_team": away,
            "group": group,
            "commence_time": commence,
            "market_home": anch["base_home"],
            "market_draw": anch["base_draw"],
            "market_away": anch["base_away"],
            "adj_home": anch["home_win"],
            "adj_draw": anch["draw"],
            "adj_away": anch["away_win"],
            "total_adj": anch["total_adj"],
            "adjustments": anch["adjustments"],
            "edge_home": round(anch["home_win"] - anch["base_home"], 4),
        })

    # ── Summary statistics ──
    edges = [abs(m["edge_home"]) for m in match_results]
    agree_count = sum(1 for e in edges if e < 0.05)
    agree_rate = agree_count / max(len(edges), 1)

    mkt_homes = [m["market_home"] for m in match_results]
    adj_homes = [m["adj_home"] for m in match_results]
    corr = np.corrcoef(mkt_homes, adj_homes)[0, 1]

    print(f"\n  Agreement rate (<5 % edge): {agree_count}/{len(edges)} = {agree_rate:.1%}", flush=True)
    print(f"  Market-vs-Adjusted correlation: {corr:.4f}", flush=True)
    print(f"  Average |edge|: {np.mean(edges):.4f}", flush=True)
    print(f"  Max |edge|: {max(edges):.4f}", flush=True)

    # ── Print all 72 matches ──
    print(f"\n  {'Match':<40} {'Mkt H':>6} {'Adj H':>6} {'Mkt D':>6} {'Adj D':>6} "
          f"{'Mkt A':>6} {'Adj A':>6} {'Edge':>6} {'Top Adj':>14}", flush=True)
    print(f"  {'─' * 120}", flush=True)

    for m in match_results:
        label = f"{m['home_team']} vs {m['away_team']}"
        # Identify largest adjustment factor
        adjs = m["adjustments"]
        top_adj_name = max(adjs, key=lambda k: abs(adjs[k]))
        top_adj_val = adjs[top_adj_name]
        tag = f"{top_adj_name.replace('_adj', '')[:10]}={top_adj_val:+.3f}"

        print(f"  {label:<40} {m['market_home']:>5.1%} {m['adj_home']:>5.1%} "
              f"{m['market_draw']:>5.1%} {m['adj_draw']:>5.1%} "
              f"{m['market_away']:>5.1%} {m['adj_away']:>5.1%} "
              f"{m['edge_home']:>+5.1%} {tag:>14}", flush=True)

    # ── Top 10 edge matches ──
    sorted_by_edge = sorted(match_results, key=lambda m: abs(m["edge_home"]), reverse=True)
    print(f"\n  TOP 10 — LARGEST EDGE (model disagrees with market):", flush=True)
    print(f"  {'#':>2} {'Match':<40} {'Mkt H':>6} {'Adj H':>6} {'Edge':>7} {'Adjustments':>40}", flush=True)
    print(f"  {'─' * 100}", flush=True)
    for i, m in enumerate(sorted_by_edge[:10], 1):
        label = f"{m['home_team']} vs {m['away_team']}"
        adj_str = "  ".join(f"{k.replace('_adj', '')[:6]}={v:+.3f}"
                            for k, v in m["adjustments"].items() if abs(v) > 0.001)
        print(f"  {i:>2}. {label:<38} {m['market_home']:>5.1%} {m['adj_home']:>5.1%} "
              f"{m['edge_home']:>+6.1%}  {adj_str}", flush=True)

    # ── Edge distribution ──
    print(f"\n  Edge distribution (home win):", flush=True)
    for lo, hi, label in [
        (0.05, 1.0, "Model > market (>+5 %)"),
        (0.02, 0.05, "Mild model > market (+2-5 %)"),
        (-0.02, 0.02, "Agreement (+-2 %)"),
        (-0.05, -0.02, "Mild market > model (-2-5 %)"),
        (-1.0, -0.05, "Market > model (>-5 %)"),
    ]:
        count = sum(1 for m in match_results if lo < m["edge_home"] <= hi)
        if lo == -1.0:
            count = sum(1 for m in match_results if m["edge_home"] <= hi)
        if hi == 1.0:
            count = sum(1 for m in match_results if m["edge_home"] > lo)
        pct = count / max(len(match_results), 1)
        print(f"    {label:<40}: {count:2d} ({pct:.0%})", flush=True)

    # ── Group stage predictions ──
    print(f"\n  GROUP STAGE PREDICTIONS:", flush=True)

    # Gather unique groups
    groups_seen = {}
    for m in match_results:
        g = m["group"].split("/")[0] if "/" in m["group"] else m["group"]
        if g not in groups_seen:
            groups_seen[g] = set()
        groups_seen[g].add(m["home_team"])
        groups_seen[g].add(m["away_team"])

    # For each group, compute expected points from group-internal matches
    for g in sorted(groups_seen.keys()):
        teams = sorted(groups_seen[g])
        # Only consider intra-group matches
        group_matches = [m for m in match_results
                         if m["group"] == g or m["group"].startswith(g + "/")]
        if len(group_matches) < 3:
            continue

        exp_pts = defaultdict(float)
        for m in group_matches:
            h, a = m["home_team"], m["away_team"]
            if h in teams and a in teams:
                exp_pts[h] += m["adj_home"] * 3 + m["adj_draw"] * 1
                exp_pts[a] += m["adj_away"] * 3 + m["adj_draw"] * 1

        if not exp_pts:
            continue

        ranked = sorted(exp_pts.items(), key=lambda x: -x[1])
        print(f"\n    Group {g}:", flush=True)
        for rank, (team, pts) in enumerate(ranked, 1):
            tag = " <-- advance" if rank <= 2 else ""
            print(f"      {rank}. {team:<22} {pts:.2f} exp pts{tag}", flush=True)

    summary = {
        "total_matches": len(match_results),
        "agree_rate": round(agree_rate, 4),
        "correlation": round(corr, 4),
        "avg_edge": round(float(np.mean(edges)), 4),
        "max_edge": round(float(max(edges)), 4),
    }

    return match_results, summary


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Monte Carlo tournament simulation
# ═══════════════════════════════════════════════════════════════════════════════

def _american_to_implied(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def run_monte_carlo(match_results, n_sims=10000):
    """Monte Carlo tournament simulation using anchored 3-way probabilities.

    Format: 12 groups of 4 (A-L).  Top 2 per group + 8 best 3rd-place teams
    advance to a 32-team knockout (R32 -> QF -> SF -> Final).
    """
    print(f"\n\n{'=' * 75}", flush=True)
    print(f"  MONTE CARLO — {n_sims:,} TOURNAMENT SIMULATIONS", flush=True)
    print(f"{'=' * 75}", flush=True)

    # ── Load groups ──
    try:
        raw_groups = _lj("wc2026_groups.json")
        teams_list = _lj("wc2026_teams.json")
        pi_data = _lj("wc2026_pi_ratings_v2.json")
        winner_odds = _lj("wc2026_winner_odds.json")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}", flush=True)
        return {}

    # ── Deduplicate groups: assign each team to first group it appears in ──
    team_primary = {}
    for grp in sorted(raw_groups.keys()):
        if len(grp) > 1:
            continue
        for team in raw_groups[grp]:
            t = _norm(team)
            if t not in team_primary:
                team_primary[t] = grp

    clean_groups = defaultdict(list)
    for team, grp in team_primary.items():
        clean_groups[grp].append(team)

    all_assigned = set(team_primary.keys())
    all_teams_norm = [_norm(t) for t in teams_list]
    unassigned = [t for t in all_teams_norm if t not in all_assigned]

    for grp in sorted(clean_groups.keys()):
        while len(clean_groups[grp]) < 4 and unassigned:
            clean_groups[grp].append(unassigned.pop(0))

    grp_letters = "ABCDEFGHIJKL"
    for letter in grp_letters:
        if letter not in clean_groups:
            clean_groups[letter] = []
    for letter in grp_letters:
        while len(clean_groups[letter]) < 4 and unassigned:
            clean_groups[letter].append(unassigned.pop(0))

    valid_groups = {g: t for g, t in clean_groups.items() if len(t) == 4}
    all_teams = set()
    for teams in valid_groups.values():
        all_teams.update(teams)

    # ── Build lookup for anchored match probabilities ──
    # For group-stage matches, use the anchored probabilities from section 4
    match_prob_lookup = {}
    if match_results:
        for m in match_results:
            key = (_norm(m["home_team"]), _norm(m["away_team"]))
            match_prob_lookup[key] = (m["adj_home"], m["adj_draw"], m["adj_away"])

    # ── Pi ratings for fallback ──
    def _get_pi(team):
        t = _norm(team)
        d = pi_data.get(t, {"home_pi": 0.0, "away_pi": 0.0})
        return d["home_pi"] + d["away_pi"]

    def _sim_match_anchored(h, a, rng, allow_draw=True):
        """Simulate a match using anchored probabilities if available,
        else fall back to Pi-based probabilities."""
        key = (_norm(h), _norm(a))
        rev_key = (_norm(a), _norm(h))

        if key in match_prob_lookup:
            p_hw, p_dr, p_aw = match_prob_lookup[key]
        elif rev_key in match_prob_lookup:
            p_aw_orig, p_dr_orig, p_hw_orig = match_prob_lookup[rev_key]
            p_hw, p_dr, p_aw = p_hw_orig, p_dr_orig, p_aw_orig
        else:
            # Fallback: Pi-based
            delta = _get_pi(h) - _get_pi(a)
            strength = 1.0 / (1.0 + math.exp(-0.25 * delta))
            dr = 0.27 * math.exp(-0.04 * delta ** 2)
            dr = max(0.10, min(0.40, dr))
            rem = 1.0 - dr
            p_hw = rem * strength
            p_aw = rem * (1.0 - strength)
            p_dr = dr

        if not allow_draw:
            # Redistribute draw probability
            total = p_hw + p_aw
            if total <= 0:
                total = 1.0
            p_hw = p_hw / total
            p_aw = p_aw / total
            p_dr = 0.0

        r = rng.random()
        if r < p_hw:
            return "home"
        elif r < p_hw + p_dr:
            return "draw"
        else:
            return "away"

    # ── Market futures ──
    market_futures = {}
    for entry in winner_odds:
        for bm in entry.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "outrights":
                    for o in mkt["outcomes"]:
                        t = _norm(o["name"])
                        prob = _american_to_implied(o["price"])
                        market_futures.setdefault(t, []).append(prob)
    market_futures = {t: np.mean(p) for t, p in market_futures.items()}

    print(f"\n  Groups: {len(valid_groups)}", flush=True)
    print(f"  Teams: {len(all_teams)}", flush=True)
    print(f"  Anchored match probs available: {len(match_prob_lookup)}", flush=True)
    print(f"  Market futures loaded: {len(market_futures)}", flush=True)

    # ── Counters ──
    group_wins = defaultdict(int)
    advanced = defaultdict(int)
    r32_count = defaultdict(int)
    qf_count = defaultdict(int)
    sf_count = defaultdict(int)
    final_count = defaultdict(int)
    champion = defaultdict(int)

    rng = np.random.default_rng(42)

    for sim in range(n_sims):
        if sim % 2500 == 0 and sim > 0:
            print(f"    ... {sim:,}/{n_sims:,} simulations", flush=True)

        group_ranked = {}
        third_place = []

        # ── Group stage ──
        for grp, teams in sorted(valid_groups.items()):
            pts = {t: 0 for t in teams}
            gd = {t: 0.0 for t in teams}
            gf = {t: 0.0 for t in teams}

            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    h, a = teams[i], teams[j]
                    res = _sim_match_anchored(h, a, rng, allow_draw=True)

                    if res == "home":
                        pts[h] += 3
                        gd[h] += 1
                        gd[a] -= 1
                        gf[h] += 1.5
                        gf[a] += 0.5
                    elif res == "draw":
                        pts[h] += 1
                        pts[a] += 1
                        gf[h] += 1
                        gf[a] += 1
                    else:
                        pts[a] += 3
                        gd[a] += 1
                        gd[h] -= 1
                        gf[a] += 1.5
                        gf[h] += 0.5

            ranked = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t], -_get_pi(t)))
            group_ranked[grp] = ranked
            group_wins[ranked[0]] += 1

            # Top 2 advance directly
            for t in ranked[:2]:
                advanced[t] += 1

            # 3rd place goes to best-3rd pool with points & gd
            third_place.append({
                "team": ranked[2],
                "pts": pts[ranked[2]],
                "gd": gd[ranked[2]],
                "gf": gf[ranked[2]],
                "pi": _get_pi(ranked[2]),
                "group": grp,
            })

        # ── Best 8 third-place teams ──
        third_sorted = sorted(third_place,
                              key=lambda x: (-x["pts"], -x["gd"], -x["gf"], -x["pi"]))
        best_thirds = [x["team"] for x in third_sorted[:8]]
        for t in best_thirds:
            advanced[t] += 1

        # ── Round of 32 bracket ──
        # Bracket structure: group winners vs best 3rds, runners-up cross-paired
        # Simplified: pair groups A/B, C/D, E/F, G/H, I/J, K/L
        # 1A vs 2B, 1B vs 2A, 1C vs 2D, 1D vs 2C, etc.
        # Then best-3rds fill remaining R32 slots against group winners
        grp_list = sorted(valid_groups.keys())

        # Build R32 matchups: 12 group winners + 12 runners-up + 8 best thirds = 32
        r32_matches = []

        # Group winners vs paired runners-up (12 matches)
        for i in range(0, len(grp_list) - 1, 2):
            g1, g2 = grp_list[i], grp_list[i + 1]
            if g1 in group_ranked and g2 in group_ranked:
                r32_matches.append((group_ranked[g1][0], group_ranked[g2][1]))
                r32_matches.append((group_ranked[g2][0], group_ranked[g1][1]))

        # Best thirds: pair them against each other to fill R32 to 16 matches
        # (12 cross-group matches + 4 best-3rd matches = 16 R32 matches)
        for i in range(0, len(best_thirds) - 1, 2):
            r32_matches.append((best_thirds[i], best_thirds[i + 1]))

        # R32
        r16 = []
        for h, a in r32_matches:
            r32_count[h] += 1
            r32_count[a] += 1
            w = h if _sim_match_anchored(h, a, rng, False) == "home" else a
            r16.append(w)

        # QF (Round of 16 -> QF)
        qf = []
        for i in range(0, len(r16) - 1, 2):
            h, a = r16[i], r16[i + 1]
            qf_count[h] += 1
            qf_count[a] += 1
            w = h if _sim_match_anchored(h, a, rng, False) == "home" else a
            qf.append(w)

        # SF
        sf = []
        for i in range(0, len(qf) - 1, 2):
            h, a = qf[i], qf[i + 1]
            sf_count[h] += 1
            sf_count[a] += 1
            w = h if _sim_match_anchored(h, a, rng, False) == "home" else a
            sf.append(w)

        # Final
        fin = []
        for i in range(0, len(sf) - 1, 2):
            h, a = sf[i], sf[i + 1]
            final_count[h] += 1
            final_count[a] += 1
            w = h if _sim_match_anchored(h, a, rng, False) == "home" else a
            fin.append(w)

        if len(fin) >= 2:
            h, a = fin[0], fin[1]
            final_count[h] += 1
            final_count[a] += 1
            w = h if _sim_match_anchored(h, a, rng, False) == "home" else a
            champion[w] += 1
        elif len(fin) == 1:
            champion[fin[0]] += 1

    print(f"    ... {n_sims:,}/{n_sims:,} simulations complete", flush=True)

    # ── Results ──
    model_probs = {t: champion.get(t, 0) / n_sims for t in all_teams}
    sorted_teams = sorted(model_probs.items(), key=lambda x: -x[1])

    print(f"\n  {'Rk':>3} {'Team':<22} {'Win %':>7} {'Market':>7} {'Edge':>7} "
          f"{'Adv':>5} {'R32':>5} {'QF':>5} {'SF':>5} {'Fin':>5}", flush=True)
    print(f"  {'─' * 95}", flush=True)

    edges = []
    for rank, (team, wp) in enumerate(sorted_teams, 1):
        mkt = market_futures.get(team, 0)
        adv = advanced.get(team, 0) / n_sims
        r32 = r32_count.get(team, 0) / n_sims
        qf = qf_count.get(team, 0) / n_sims
        sf_ = sf_count.get(team, 0) / n_sims
        fn = final_count.get(team, 0) / n_sims
        edge = wp - mkt
        edges.append({"team": team, "model": round(wp, 4), "market": round(mkt, 4),
                       "edge": round(edge, 4)})

        if rank <= 30 or wp > 0.005:
            print(f"  {rank:>3}. {team:<22} {wp:>6.1%} {mkt:>6.1%} {edge:>+6.1%} "
                  f"{adv:>4.0%} {r32:>4.0%} {qf:>4.0%} {sf_:>4.0%} {fn:>4.0%}", flush=True)

    # ── Value bets ──
    value = sorted(edges, key=lambda x: -x["edge"])
    print(f"\n  VALUE BETS (model > market):", flush=True)
    for v in value[:10]:
        if v["edge"] > 0.002:
            print(f"    {v['team']:<22} model={v['model']:.2%}  "
                  f"market={v['market']:.2%}  edge={v['edge']:+.2%}", flush=True)

    overvalued = sorted(edges, key=lambda x: x["edge"])
    print(f"\n  OVERVALUED BY MARKET:", flush=True)
    for v in overvalued[:8]:
        if v["edge"] < -0.002:
            print(f"    {v['team']:<22} model={v['model']:.2%}  "
                  f"market={v['market']:.2%}  edge={v['edge']:+.2%}", flush=True)

    # ── Comparison with market futures ──
    model_vals = []
    market_vals = []
    for e in edges:
        if e["market"] > 0:
            model_vals.append(e["model"])
            market_vals.append(e["market"])
    if len(model_vals) > 2:
        futures_corr = np.corrcoef(model_vals, market_vals)[0, 1]
        print(f"\n  Model-vs-Market futures correlation: {futures_corr:.4f}", flush=True)
    else:
        futures_corr = 0.0

    return {
        "n_sims": n_sims,
        "teams_simulated": len(all_teams),
        "groups": len(valid_groups),
        "champion_probs": {t: round(champion.get(t, 0) / n_sims, 4) for t in all_teams},
        "advance_probs": {t: round(advanced.get(t, 0) / n_sims, 4) for t in all_teams},
        "value_bets": [v for v in value[:15] if v["edge"] > 0.002],
        "overvalued": [v for v in overvalued[:10] if v["edge"] < -0.002],
        "futures_correlation": round(futures_corr, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 75, flush=True)
    print("  WC 2026 — MARKET-ANCHORED PREDICTION MODEL", flush=True)
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("=" * 75, flush=True)
    print("\n  Philosophy: anchor to bookmaker odds, apply small structural", flush=True)
    print("  adjustments (Pi, chemistry, dependency, league quality, form).", flush=True)
    print("  Target: agree with market ~80 %; identify 15-20 % genuine edge.", flush=True)

    # ── Load historical results ──
    results_path = os.path.join(_DATA_DIR, "international_results.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        results_df["date"] = pd.to_datetime(results_df["date"])
        print(f"\n  Loaded {len(results_df):,} international results", flush=True)
    else:
        results_df = None
        print("\n  [WARN] No international results file — skipping backtest", flush=True)

    all_output = {
        "run_date": datetime.now().isoformat(),
        "model": "market_anchored_v1",
        "philosophy": "Anchor to market odds; small structural adjustments",
    }

    # ── 3. Historical backtest ──
    if results_df is not None:
        bt_results = run_historical_backtest(results_df)
        all_output["backtest"] = bt_results
    else:
        bt_results = {}

    # ── 4. WC 2026 match predictions ──
    match_results, pred_summary = run_wc2026_predictions()
    all_output["predictions"] = {
        "summary": pred_summary,
        "matches": match_results,
    }

    # ── 5. Monte Carlo ──
    mc_results = run_monte_carlo(match_results, n_sims=10000)
    all_output["monte_carlo"] = mc_results

    # ── Final summary ──
    print(f"\n\n{'=' * 75}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'=' * 75}", flush=True)

    if pred_summary:
        print(f"\n  Match predictions:", flush=True)
        print(f"    Agreement rate: {pred_summary.get('agree_rate', 0):.1%}", flush=True)
        print(f"    Market correlation: {pred_summary.get('correlation', 0):.4f}", flush=True)
        print(f"    Avg |edge|: {pred_summary.get('avg_edge', 0):.4f}", flush=True)

    if bt_results and "combined" in bt_results:
        bc = bt_results["combined"]
        print(f"\n  Historical backtest:", flush=True)
        print(f"    Base-Adj correlation: {bc.get('base_adj_corr', 0):.4f}", flush=True)
        print(f"    Disagreement matches: {bc.get('disagree_count', 0)}", flush=True)
        print(f"    Model edge rate: {bc.get('model_edge_rate', 0):.1%}", flush=True)
        print(f"    Market edge rate: {bc.get('market_edge_rate', 0):.1%}", flush=True)

    if mc_results and "champion_probs" in mc_results:
        cp = mc_results["champion_probs"]
        top5 = sorted(cp.items(), key=lambda x: -x[1])[:5]
        print(f"\n  Monte Carlo top 5:", flush=True)
        for t, p in top5:
            mkt = mc_results.get("value_bets", [])
            print(f"    {t:<22} {p:.1%}", flush=True)
        print(f"    Futures correlation: {mc_results.get('futures_correlation', 0):.4f}", flush=True)

    # ── Save ──
    out_path = os.path.join(_DATA_DIR, "wc2026_market_anchored_results.json")
    with open(out_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
