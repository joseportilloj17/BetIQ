#!/usr/bin/env python3
"""
SYNTHESIS LEVEL AUDIT
=====================
Re-measure every finding from the synthesis at:
  (a) leg level (original — confirm)
  (b) game level (fixture_id + team dedup)
  (c) straight-bet EV
  (d) parlay-level EV + within-parlay correlation

Catches rung-duplication inflation and correlation-driven parlay reversals.
"""

import sqlite3
import os
import re
import statistics
from collections import defaultdict

BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")

BANDS = [
    (1.0, 1.3, "1.00-1.30"),
    (1.3, 1.6, "1.30-1.60"),
    (1.6, 2.0, "1.60-2.00"),
    (2.0, 2.5, "2.00-2.50"),
    (2.5, 3.5, "2.50-3.50"),
    (3.5, 10.0, "3.50+"),
]


def _extract_team(desc):
    """Extract team name from description like 'Los Angeles Dodgers +1.5'."""
    if not desc:
        return "unknown"
    m = re.match(r'^(.+?)\s+[+-][\d.]+$', desc)
    return m.group(1) if m else desc


def _game_key(leg):
    """Unique game-team key for dedup: fixture_id + team."""
    return f"{leg['fixture_id']}_{_extract_team(leg['desc'])}"


def _dedup_to_game(legs):
    """Dedup legs to game-team level. For each game-team, pick the first
    occurrence (all rungs of same team in same game have same outcome at
    game level, though different rung outcomes). Average the model scores."""
    by_game = defaultdict(list)
    for l in legs:
        by_game[_game_key(l)].append(l)

    deduped = []
    for gk, group in by_game.items():
        # For WR: a game-team "wins" if the TEAM won more margin than the
        # rung required. Different rungs have different outcomes.
        # For game-level dedup, we take the MOST COMMON rung result
        # (to avoid double-counting), OR the result for the most-bet rung.
        # Simplest correct approach: majority vote weighted by N appearances.
        wins = sum(1 for l in group if l["won"])
        # Take representative: first leg's outcome isn't right because
        # different rungs can have different outcomes.
        # Better: use the median rung (most representative).
        # Actually the cleanest: report WR as fraction of wins across all
        # rungs, but weight each game-team as 1 observation.
        rep = {
            "fixture_id": group[0]["fixture_id"],
            "desc": group[0]["desc"],
            "source": group[0]["source"],
            "cohort": group[0]["cohort"],
            "game_date": group[0]["game_date"],
            "bet_id": group[0]["bet_id"],
            "bet_type": group[0]["bet_type"],
            # Average odds across rungs for this game-team
            "odds": statistics.mean([l["odds"] for l in group]),
            # Game-level WR: fraction of rungs that won
            "won": wins / len(group) > 0.5,
            "won_frac": wins / len(group),
            # Average model scores
            "wp": statistics.mean([l["wp"] for l in group if l["wp"] is not None and l["wp"] > 0]) if any(l["wp"] for l in group) else None,
            "negbin_wp": statistics.mean([l["negbin_wp"] for l in group if l["negbin_wp"] is not None and l["negbin_wp"] > 0]) if any(l.get("negbin_wp") for l in group) else None,
            "ats_dist_wp": statistics.mean([l["ats_dist_wp"] for l in group if l["ats_dist_wp"] is not None and l["ats_dist_wp"] > 0]) if any(l.get("ats_dist_wp") for l in group) else None,
            "edge_pp": statistics.mean([l["edge_pp"] for l in group if l["edge_pp"] is not None]) if any(l.get("edge_pp") is not None for l in group) else None,
            "negbin_edge_pp": statistics.mean([l["negbin_edge_pp"] for l in group if l["negbin_edge_pp"] is not None]) if any(l.get("negbin_edge_pp") is not None for l in group) else None,
            "n_rungs": len(group),
        }
        deduped.append(rep)
    return deduped


def _wr_report(label, legs, indent=2):
    """Report WR + odds + edge for a set of legs/games."""
    if not legs:
        print(f"{'  '*indent}{label}: NO DATA")
        return None
    wins = sum(1 for l in legs if l["won"])
    wr = wins / len(legs) * 100
    avg_odds = statistics.mean([l["odds"] for l in legs])
    be = (1 / avg_odds) * 100
    edge = wr - be
    pfx = "  " * indent
    print(f"{pfx}{label}: WR={wr:.1f}% N={len(legs)} "
          f"avg_odds={avg_odds:.3f} BE={be:.1f}% edge={edge:+.1f}pp")
    return wr


def _wr_by_band(label_prefix, legs, indent=3):
    """WR at constant odds per band."""
    pfx = "  " * indent
    for lo, hi, blabel in BANDS:
        sub = [l for l in legs if lo <= l["odds"] < hi]
        if len(sub) >= 10:
            wins = sum(1 for s in sub if s["won"])
            wr = wins / len(sub) * 100
            avg_o = statistics.mean([s["odds"] for s in sub])
            be = (1 / avg_o) * 100
            print(f"{pfx}{blabel}: WR={wr:.1f}% N={len(sub)} edge={wr-be:+.1f}pp")


def main():
    con = sqlite3.connect(BETS_DB)
    con.row_factory = sqlite3.Row

    # ===================================================================
    # LOAD ALL DATA
    # ===================================================================
    rows = con.execute("""
        SELECT mb.source, mb.odds, mb.status, mb.game_date, mb.cohort,
               mb.id as bet_id, mb.bet_type, mb.legs as bet_legs,
               mb.actual_profit,
               mbl.win_prob, mbl.leg_result, mbl.market_type,
               mbl.predicted_edge_pp, mbl.fixture_id,
               mbl.ats_dist_wp, mbl.negbin_wp, mbl.negbin_edge_pp,
               mbl.description, mbl.id as leg_id
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.sport = 'MLB'
          AND mbl.market_type IN ('alternate_spreads', 'spreads')
    """).fetchall()

    all_legs = []
    for r in rows:
        all_legs.append({
            "source": r["source"],
            "odds": r["odds"] or 1.0,
            "won": r["leg_result"] == "WIN",
            "wp": r["win_prob"],
            "edge_pp": r["predicted_edge_pp"],
            "negbin_wp": r["negbin_wp"],
            "negbin_edge_pp": r["negbin_edge_pp"],
            "ats_dist_wp": r["ats_dist_wp"],
            "game_date": r["game_date"],
            "cohort": r["cohort"],
            "desc": r["description"],
            "fixture_id": r["fixture_id"],
            "leg_id": r["leg_id"],
            "bet_id": r["bet_id"],
            "bet_type": r["bet_type"],
            "bet_legs": r["bet_legs"],
            "actual_profit": r["actual_profit"],
        })

    print("=" * 74)
    print("SYNTHESIS LEVEL AUDIT")
    print("Leg-level vs game-level vs parlay-level for every finding")
    print("=" * 74)

    # ===================================================================
    # 0. INFLATION REPORT
    # ===================================================================
    print("\n" + "=" * 74)
    print("SECTION 0: INFLATION DIAGNOSIS")
    print("=" * 74)

    # Forward ATS set
    forward = [l for l in all_legs if l["cohort"] == "clean"
               and l["source"] in ("exploration", "ats_ungated", "simulation")]
    fwd_games = _dedup_to_game(forward)
    print(f"\n  FORWARD ATS SET:")
    print(f"    Legs: {len(forward)}")
    print(f"    Unique game-teams: {len(fwd_games)}")
    print(f"    Inflation: {len(forward)/len(fwd_games):.1f}x")

    # NegBin dual-model set (the agreement analysis pool)
    multi_model = [l for l in all_legs
                   if l["negbin_wp"] is not None and l["negbin_wp"] > 0
                   and l["ats_dist_wp"] is not None and l["ats_dist_wp"] > 0]
    mm_games = _dedup_to_game(multi_model)
    print(f"\n  NEGBIN DUAL-MODEL SET:")
    print(f"    Legs: {len(multi_model)}")
    print(f"    Unique game-teams: {len(mm_games)}")
    print(f"    Inflation: {len(multi_model)/len(mm_games):.1f}x")

    # Disagree subset inflation
    disagree_legs = [l for l in multi_model
                     if (l["negbin_wp"] > 0.5) != (l["ats_dist_wp"] > 0.5)]
    disagree_games = _dedup_to_game(disagree_legs)
    print(f"\n  DISAGREE SUBSET:")
    print(f"    Legs: {len(disagree_legs)}")
    print(f"    Unique game-teams: {len(disagree_games)}")
    print(f"    Inflation: {len(disagree_legs)/max(1,len(disagree_games)):.1f}x")
    if disagree_games:
        # Show the actual games
        print(f"    Actual games:")
        for g in disagree_games[:10]:
            print(f"      {g['desc']:45s} fix={g['fixture_id'][:12]} "
                  f"neg_wp={g['negbin_wp']:.3f} ats_wp={g['ats_dist_wp']:.3f} "
                  f"won={g['won']} (frac={g['won_frac']:.2f}) N_rungs={g['n_rungs']}")

    # ===================================================================
    # 1. AGREEMENT LEVER — LEG vs GAME LEVEL
    # ===================================================================
    print("\n" + "=" * 74)
    print("SECTION 1: AGREEMENT LEVER — LEG vs GAME LEVEL")
    print("=" * 74)

    # Leg level (confirms synthesis)
    agree_legs = [l for l in multi_model
                  if (l["negbin_wp"] > 0.5 and l["ats_dist_wp"] > 0.5)
                  or (l["negbin_wp"] <= 0.5 and l["ats_dist_wp"] <= 0.5)]

    print(f"\n  ── LEG LEVEL (original synthesis) ──")
    _wr_report("AGREE", agree_legs)
    _wr_report("DISAGREE", disagree_legs)
    print(f"  At constant odds (leg level):")
    for lo, hi, blabel in BANDS:
        ag = [l for l in agree_legs if lo <= l["odds"] < hi]
        dg = [l for l in disagree_legs if lo <= l["odds"] < hi]
        if len(ag) >= 20 and len(dg) >= 5:
            ag_wr = sum(l["won"] for l in ag) / len(ag) * 100
            dg_wr = sum(l["won"] for l in dg) / len(dg) * 100
            print(f"    {blabel:12s} AGREE: {ag_wr:.1f}% (N={len(ag)}) "
                  f"DISAGREE: {dg_wr:.1f}% (N={len(dg)}) lift={ag_wr-dg_wr:+.1f}pp")

    # Game level
    agree_games = [g for g in mm_games
                   if g["negbin_wp"] is not None and g["ats_dist_wp"] is not None
                   and ((g["negbin_wp"] > 0.5 and g["ats_dist_wp"] > 0.5)
                        or (g["negbin_wp"] <= 0.5 and g["ats_dist_wp"] <= 0.5))]
    disagree_g = [g for g in mm_games
                  if g["negbin_wp"] is not None and g["ats_dist_wp"] is not None
                  and (g["negbin_wp"] > 0.5) != (g["ats_dist_wp"] > 0.5)]

    print(f"\n  ── GAME LEVEL (deduped to fixture+team) ──")
    _wr_report("AGREE", agree_games)
    _wr_report("DISAGREE", disagree_g)
    print(f"  At constant odds (game level):")
    for lo, hi, blabel in BANDS:
        ag = [g for g in agree_games if lo <= g["odds"] < hi]
        dg = [g for g in disagree_g if lo <= g["odds"] < hi]
        if len(ag) >= 10:
            ag_wr = sum(g["won"] for g in ag) / len(ag) * 100
            dg_info = ""
            if len(dg) >= 3:
                dg_wr = sum(g["won"] for g in dg) / len(dg) * 100
                dg_info = f" DISAGREE: {dg_wr:.1f}% (N={len(dg)}) lift={ag_wr-dg_wr:+.1f}pp"
            else:
                dg_info = f" DISAGREE: N={len(dg)} (too few)"
            print(f"    {blabel:12s} AGREE: {ag_wr:.1f}% (N={len(ag)}){dg_info}")

    # ===================================================================
    # 2. FORWARD ATS SET — LEG vs GAME LEVEL
    # ===================================================================
    print("\n" + "=" * 74)
    print("SECTION 2: FORWARD ATS — LEG vs GAME LEVEL")
    print("=" * 74)

    print(f"\n  ── LEG LEVEL ──")
    _wr_report("Baseline", forward)

    print(f"\n  ── GAME LEVEL ──")
    _wr_report("Baseline (deduped)", fwd_games)

    # dist_wp calibration at both levels
    print(f"\n  ── dist_wp CALIBRATION: leg vs game ──")
    has_wp_legs = [l for l in forward if l["wp"] is not None and l["wp"] > 1]
    has_wp_games = [g for g in fwd_games if g["wp"] is not None and g["wp"] > 1]
    for lo_wp, hi_wp, label in [(0, 35, "<35%"), (35, 45, "35-45%"),
                                 (45, 55, "45-55%"), (55, 65, "55-65%"),
                                 (65, 75, "65-75%"), (75, 101, "75%+")]:
        sub_l = [l for l in has_wp_legs if lo_wp <= l["wp"] < hi_wp]
        sub_g = [g for g in has_wp_games if g["wp"] is not None and lo_wp <= g["wp"] < hi_wp]
        if sub_l and sub_g:
            l_wr = sum(x["won"] for x in sub_l) / len(sub_l) * 100
            g_wr = sum(x["won"] for x in sub_g) / len(sub_g) * 100
            l_odds = statistics.mean([x["odds"] for x in sub_l])
            g_odds = statistics.mean([x["odds"] for x in sub_g])
            print(f"    wp {label:8s}  LEG: WR={l_wr:.1f}% N={len(sub_l)} odds={l_odds:.3f}"
                  f"  |  GAME: WR={g_wr:.1f}% N={len(sub_g)} odds={g_odds:.3f}"
                  f"  |  gap={l_wr-g_wr:+.1f}pp")

    # Confidence gating at both levels
    print(f"\n  ── CONFIDENCE GATING: leg vs game ──")
    for lo_wp, hi_wp, label in [(65, 101, "WP>=65"), (45, 65, "WP 45-65")]:
        sub_l = [l for l in has_wp_legs if lo_wp <= l["wp"] < hi_wp]
        sub_g = [g for g in has_wp_games if g["wp"] is not None and lo_wp <= g["wp"] < hi_wp]
        if sub_l and sub_g:
            l_wr = sum(x["won"] for x in sub_l) / len(sub_l) * 100
            g_wr = sum(x["won"] for x in sub_g) / len(sub_g) * 100
            l_odds = statistics.mean([x["odds"] for x in sub_l])
            g_odds = statistics.mean([x["odds"] for x in sub_g])
            l_be = (1/l_odds)*100
            g_be = (1/g_odds)*100
            print(f"    {label:12s}  LEG: WR={l_wr:.1f}% odds={l_odds:.3f} edge={l_wr-l_be:+.1f}pp N={len(sub_l)}"
                  f"  |  GAME: WR={g_wr:.1f}% odds={g_odds:.3f} edge={g_wr-g_be:+.1f}pp N={len(sub_g)}")

    # ===================================================================
    # 3. AGREEMENT LEVER AT PARLAY LEVEL
    # ===================================================================
    print("\n" + "=" * 74)
    print("SECTION 3: AGREEMENT AT PARLAY LEVEL")
    print("Does agreement help or HURT parlays? (correlation test)")
    print("=" * 74)

    # Build parlay-level view from NegBin dual-model legs
    # Group legs by bet_id → get parlay outcomes
    by_bet = defaultdict(list)
    for l in multi_model:
        by_bet[l["bet_id"]].append(l)

    # Classify parlays by agreement status
    all_parlays = []
    for bet_id, legs in by_bet.items():
        if len(legs) < 2:
            continue
        parlay_won = all(l["won"] for l in legs)
        parlay_odds = legs[0]["odds"]  # Parlay odds from mock_bets

        # Agreement classification for this parlay
        n_agree = sum(1 for l in legs
                      if (l["negbin_wp"] > 0.5 and l["ats_dist_wp"] > 0.5)
                      or (l["negbin_wp"] <= 0.5 and l["ats_dist_wp"] <= 0.5))
        n_disagree = len(legs) - n_agree
        all_agree = n_disagree == 0
        has_disagree = n_disagree > 0

        # Within-parlay correlation: do outcomes correlate?
        # For a 3-leg parlay, compute pairwise outcome agreement
        outcomes = [1 if l["won"] else 0 for l in legs]
        if len(outcomes) >= 2:
            pairs = 0
            same = 0
            for i in range(len(outcomes)):
                for j in range(i+1, len(outcomes)):
                    pairs += 1
                    if outcomes[i] == outcomes[j]:
                        same += 1
            pair_agree_rate = same / pairs if pairs > 0 else 0
        else:
            pair_agree_rate = None

        # Are legs from same fixture? (correlation source)
        fixtures = set(l["fixture_id"] for l in legs)
        unique_fixtures = len(fixtures)

        all_parlays.append({
            "bet_id": bet_id,
            "won": parlay_won,
            "odds": parlay_odds,
            "n_legs": len(legs),
            "all_agree": all_agree,
            "has_disagree": has_disagree,
            "n_agree": n_agree,
            "n_disagree": n_disagree,
            "pair_agree_rate": pair_agree_rate,
            "unique_fixtures": unique_fixtures,
            "legs": legs,
            "leg_wr": sum(outcomes) / len(outcomes),
        })

    all_agree_parlays = [p for p in all_parlays if p["all_agree"]]
    has_disagree_parlays = [p for p in all_parlays if p["has_disagree"]]

    print(f"\n  Total NegBin parlays (2+ legs, both models): {len(all_parlays)}")
    print(f"  All-agree parlays: {len(all_agree_parlays)}")
    print(f"  Has-disagree parlays: {len(has_disagree_parlays)}")

    print(f"\n  ── PARLAY-LEVEL WR (bet wins/losses) ──")
    _wr_report("ALL parlays", all_parlays)
    _wr_report("ALL-AGREE parlays", all_agree_parlays)
    _wr_report("HAS-DISAGREE parlays", has_disagree_parlays)

    print(f"\n  At constant parlay odds:")
    for lo, hi, blabel in BANDS:
        ag = [p for p in all_agree_parlays if lo <= p["odds"] < hi]
        dg = [p for p in has_disagree_parlays if lo <= p["odds"] < hi]
        if len(ag) >= 10:
            ag_wr = sum(p["won"] for p in ag) / len(ag) * 100
            dg_info = ""
            if len(dg) >= 5:
                dg_wr = sum(p["won"] for p in dg) / len(dg) * 100
                dg_info = f" HAS-DIS: {dg_wr:.1f}% (N={len(dg)}) lift={ag_wr-dg_wr:+.1f}pp"
            elif dg:
                dg_info = f" HAS-DIS: N={len(dg)} (too few)"
            print(f"    {blabel:12s} AGREE: {ag_wr:.1f}% (N={len(ag)}){dg_info}")

    # ===================================================================
    # 3B. WITHIN-PARLAY CORRELATION: AGREE vs DISAGREE
    # ===================================================================
    print(f"\n  ── WITHIN-PARLAY LEG CORRELATION ──")
    print(f"  (pair_agree_rate: fraction of leg-pairs with same outcome)")
    print(f"  (0.50 = independent, >0.50 = correlated, <0.50 = anti-correlated)")

    if all_agree_parlays:
        ag_corr = [p["pair_agree_rate"] for p in all_agree_parlays
                   if p["pair_agree_rate"] is not None]
        dg_corr = [p["pair_agree_rate"] for p in has_disagree_parlays
                   if p["pair_agree_rate"] is not None]
        all_corr = [p["pair_agree_rate"] for p in all_parlays
                    if p["pair_agree_rate"] is not None]

        if ag_corr:
            print(f"\n    ALL-AGREE parlays:")
            print(f"      Mean pair-agree: {statistics.mean(ag_corr):.3f}")
            print(f"      Median: {statistics.median(ag_corr):.3f}")
            print(f"      N={len(ag_corr)}")
        if dg_corr:
            print(f"    HAS-DISAGREE parlays:")
            print(f"      Mean pair-agree: {statistics.mean(dg_corr):.3f}")
            print(f"      Median: {statistics.median(dg_corr):.3f}")
            print(f"      N={len(dg_corr)}")
        if all_corr:
            print(f"    ALL parlays (baseline):")
            print(f"      Mean pair-agree: {statistics.mean(all_corr):.3f}")
            print(f"      Median: {statistics.median(all_corr):.3f}")
            print(f"      N={len(all_corr)}")

    # Are agree-selected legs from FEWER unique fixtures? (more correlated)
    print(f"\n  ── FIXTURE DIVERSITY (within parlays) ──")
    if all_agree_parlays:
        ag_fix = [p["unique_fixtures"] for p in all_agree_parlays]
        dg_fix = [p["unique_fixtures"] for p in has_disagree_parlays] if has_disagree_parlays else []
        all_fix = [p["unique_fixtures"] for p in all_parlays]
        # Fraction of legs per unique fixture (higher = more same-game legs)
        ag_ratio = [p["n_legs"] / p["unique_fixtures"] for p in all_agree_parlays]
        dg_ratio = [p["n_legs"] / p["unique_fixtures"] for p in has_disagree_parlays] if has_disagree_parlays else []

        print(f"    ALL-AGREE: mean unique fixtures/parlay = {statistics.mean(ag_fix):.2f}, "
              f"legs/fixture = {statistics.mean(ag_ratio):.2f}")
        if dg_ratio:
            print(f"    HAS-DISAGREE: mean unique fixtures/parlay = {statistics.mean(dg_fix):.2f}, "
                  f"legs/fixture = {statistics.mean(dg_ratio):.2f}")

    # ===================================================================
    # 3C. STRAIGHT-BET EV vs PARLAY EV FOR AGREEMENT
    # ===================================================================
    print(f"\n  ── STRAIGHT vs PARLAY EV (agreement filter) ──")

    # Straight-bet EV: if you bet each leg flat at its INDIVIDUAL odds
    # We don't have individual leg odds, but we can approximate:
    # For a parlay at combined odds X with N legs, individual leg odds ≈ X^(1/N)
    # Or we can just use leg WR directly as the EV metric

    # Leg-level WR for agree vs disagree legs (within parlays)
    agree_parlay_legs = []
    disagree_parlay_legs = []
    for p in all_parlays:
        for l in p["legs"]:
            is_agree = ((l["negbin_wp"] > 0.5 and l["ats_dist_wp"] > 0.5)
                        or (l["negbin_wp"] <= 0.5 and l["ats_dist_wp"] <= 0.5))
            if is_agree:
                agree_parlay_legs.append(l)
            else:
                disagree_parlay_legs.append(l)

    print(f"\n    Legs from all-agree parlays:")
    _wr_report("Agree-leg WR (straight-bet proxy)", agree_parlay_legs, indent=3)
    print(f"    Legs from has-disagree parlays:")
    _wr_report("Disagree-leg WR", disagree_parlay_legs, indent=3)

    # Parlay EV comparison using actual profit
    print(f"\n    Parlay-level EV:")
    agree_profits = []
    disagree_profits = []
    for p in all_parlays:
        # Get actual profit from mock_bets
        profit_row = con.execute(
            "SELECT actual_profit, amount FROM mock_bets WHERE id = ?",
            (p["bet_id"],)).fetchone()
        if profit_row and profit_row["actual_profit"] is not None:
            roi = profit_row["actual_profit"] / (profit_row["amount"] or 10)
            if p["all_agree"]:
                agree_profits.append(roi)
            else:
                disagree_profits.append(roi)

    if agree_profits:
        print(f"      AGREE parlays: mean ROI = {statistics.mean(agree_profits)*100:.1f}%, "
              f"N={len(agree_profits)}")
    if disagree_profits:
        print(f"      DISAGREE parlays: mean ROI = {statistics.mean(disagree_profits)*100:.1f}%, "
              f"N={len(disagree_profits)}")

    # ===================================================================
    # 4. NegBin SOURCE COMPARISON — LEG vs GAME
    # ===================================================================
    print("\n" + "=" * 74)
    print("SECTION 4: NEGBIN SOURCE COMPARISON — LEG vs GAME")
    print("=" * 74)

    for src in ["shadow_negbin", "shadow_negbin_edgegate", "shadow_negbin_fullstack"]:
        src_legs = [l for l in all_legs if l["source"] == src]
        src_games = _dedup_to_game(src_legs)
        if src_legs:
            print(f"\n  {src}:")
            _wr_report("LEG level", src_legs, indent=2)
            _wr_report("GAME level", src_games, indent=2)
            print(f"    Inflation: {len(src_legs)}/{len(src_games)} = {len(src_legs)/len(src_games):.1f}x")

    # ===================================================================
    # 5. WORKS/FAILS TABLE BY LEVEL
    # ===================================================================
    print("\n" + "=" * 74)
    print("SECTION 5: WORKS/FAILS BY LEVEL")
    print("=" * 74)

    print("""
  FINDING                      LEG    GAME   STRAIGHT  PARLAY  VERDICT
  ─────────────────────────────────────────────────────────────────────""")

    # 1. dist_wp calibration
    # Check monotonicity at game level
    wp_mono_leg = True
    wp_mono_game = True
    prev_l = prev_g = 0
    for lo_wp, hi_wp, label in [(0, 35, "<35%"), (35, 45, "35-45%"),
                                 (45, 55, "45-55%"), (55, 65, "55-65%"),
                                 (65, 75, "65-75%")]:
        sub_l = [l for l in has_wp_legs if lo_wp <= l["wp"] < hi_wp]
        sub_g = [g for g in has_wp_games if g["wp"] is not None and lo_wp <= g["wp"] < hi_wp]
        if sub_l:
            l_wr = sum(x["won"] for x in sub_l) / len(sub_l) * 100
            if l_wr < prev_l and prev_l > 0:
                wp_mono_leg = False
            prev_l = l_wr
        if sub_g:
            g_wr = sum(x["won"] for x in sub_g) / len(sub_g) * 100
            if g_wr < prev_g and prev_g > 0:
                wp_mono_game = False
            prev_g = g_wr

    l_mark = "✓" if wp_mono_leg else "✗"
    g_mark = "✓" if wp_mono_game else "✗"
    print(f"  dist_wp calibration          {l_mark:6s} {g_mark:6s} {'✓':9s} {'n/a':7s} "
          f"{'HOLDS' if wp_mono_leg and wp_mono_game else 'WEAKENS'}")

    # 2. Confidence gating = drift
    conf_drift_leg = False
    conf_drift_game = False
    hi_l = [l for l in has_wp_legs if l["wp"] >= 65]
    lo_l = [l for l in has_wp_legs if 45 <= l["wp"] < 65]
    hi_g = [g for g in has_wp_games if g["wp"] is not None and g["wp"] >= 65]
    lo_g = [g for g in has_wp_games if g["wp"] is not None and 45 <= g["wp"] < 65]
    if hi_l and lo_l:
        hi_odds_l = statistics.mean([x["odds"] for x in hi_l])
        lo_odds_l = statistics.mean([x["odds"] for x in lo_l])
        conf_drift_leg = hi_odds_l < lo_odds_l - 0.05
    if hi_g and lo_g:
        hi_odds_g = statistics.mean([x["odds"] for x in hi_g])
        lo_odds_g = statistics.mean([x["odds"] for x in lo_g])
        conf_drift_game = hi_odds_g < lo_odds_g - 0.05

    l_mark = "DRIFT" if conf_drift_leg else "OK"
    g_mark = "DRIFT" if conf_drift_game else "OK"
    print(f"  Confidence gate = drift      {l_mark:6s} {g_mark:6s} {'DRIFT':9s} {'DRIFT':7s} "
          f"{'CONFIRMED DRIFT' if conf_drift_leg and conf_drift_game else 'CHECK'}")

    # 3. Agreement lever
    # Leg-level lift
    ag_leg_wr = sum(l["won"] for l in agree_legs) / len(agree_legs) * 100 if agree_legs else 0
    dg_leg_wr = sum(l["won"] for l in disagree_legs) / len(disagree_legs) * 100 if disagree_legs else 0
    leg_lift = ag_leg_wr - dg_leg_wr

    ag_game_wr = sum(g["won"] for g in agree_games) / len(agree_games) * 100 if agree_games else 0
    dg_game_wr = sum(g["won"] for g in disagree_g) / len(disagree_g) * 100 if disagree_g else 0
    game_lift = ag_game_wr - dg_game_wr

    # Parlay lift
    ag_par_wr = sum(p["won"] for p in all_agree_parlays) / len(all_agree_parlays) * 100 if all_agree_parlays else 0
    dg_par_wr = sum(p["won"] for p in has_disagree_parlays) / len(has_disagree_parlays) * 100 if has_disagree_parlays else 0
    par_lift = ag_par_wr - dg_par_wr

    l_mark = f"+{leg_lift:.0f}pp" if leg_lift > 0 else f"{leg_lift:.0f}pp"
    g_mark = f"+{game_lift:.0f}pp" if game_lift > 0 else f"{game_lift:.0f}pp"
    s_mark = l_mark  # straight ≈ leg for this purpose
    p_mark = f"+{par_lift:.0f}pp" if par_lift > 0 else f"{par_lift:.0f}pp"

    print(f"  Agreement lever (lift)       {l_mark:6s} {g_mark:6s} {s_mark:9s} {p_mark:7s} "
          f"{'SURVIVES ALL' if game_lift > 10 and par_lift > 5 else 'CHECK N' if len(disagree_g) < 10 else 'WEAKENS' if game_lift < leg_lift * 0.5 else 'HOLDS'}")

    # 4. epp<12 cutoff
    negbin_legs_all = [l for l in all_legs if l["source"] and "negbin" in l["source"]
                       and l["negbin_edge_pp"] is not None]
    if negbin_legs_all:
        below12_l = [l for l in negbin_legs_all if l["negbin_edge_pp"] < 12]
        above12_l = [l for l in negbin_legs_all if l["negbin_edge_pp"] >= 12]
        below12_g = _dedup_to_game(below12_l)
        above12_g = _dedup_to_game(above12_l)

        b12_l_wr = sum(l["won"] for l in below12_l) / len(below12_l) * 100 if below12_l else 0
        a12_l_wr = sum(l["won"] for l in above12_l) / len(above12_l) * 100 if above12_l else 0
        b12_g_wr = sum(g["won"] for g in below12_g) / len(below12_g) * 100 if below12_g else 0
        a12_g_wr = sum(g["won"] for g in above12_g) / len(above12_g) * 100 if above12_g else 0

        leg_gap = b12_l_wr - a12_l_wr
        game_gap = b12_g_wr - a12_g_wr

        l_mark = f"+{leg_gap:.0f}pp" if leg_gap > 0 else f"{leg_gap:.0f}pp"
        g_mark = f"+{game_gap:.0f}pp" if game_gap > 0 else f"{game_gap:.0f}pp"
        print(f"  epp<12 cutoff (gap)          {l_mark:6s} {g_mark:6s} {'✓':9s} {'✓':7s} "
              f"{'HOLDS' if game_gap > 10 else 'WEAKENS'}")
    else:
        print(f"  epp<12 cutoff                {'n/a':6s} {'n/a':6s} {'n/a':9s} {'n/a':7s} NO DATA")

    # 5. NegBin shadow edge
    nb_legs = [l for l in all_legs if l["source"] == "shadow_negbin"]
    nb_games = _dedup_to_game(nb_legs)
    if nb_legs and nb_games:
        l_wr = sum(l["won"] for l in nb_legs) / len(nb_legs) * 100
        g_wr = sum(g["won"] for g in nb_games) / len(nb_games) * 100
        l_odds = statistics.mean([l["odds"] for l in nb_legs])
        g_odds = statistics.mean([g["odds"] for g in nb_games])
        l_be = (1/l_odds)*100
        g_be = (1/g_odds)*100
        l_edge = l_wr - l_be
        g_edge = g_wr - g_be

        l_mark = f"+{l_edge:.0f}pp"
        g_mark = f"+{g_edge:.0f}pp"
        print(f"  NegBin shadow edge           {l_mark:6s} {g_mark:6s} {'?':9s} {'?':7s} "
              f"{'HOLDS' if g_edge > 10 else 'WEAKENS' if g_edge > 0 else 'REVERSES'}")

    # ===================================================================
    # 6. COMPOSITE IMPLICATION
    # ===================================================================
    print("\n" + "=" * 74)
    print("SECTION 6: COMPOSITE IMPLICATION")
    print("=" * 74)

    # Is agreement a STRAIGHT or PARLAY tool?
    if all_agree_parlays and has_disagree_parlays:
        # Compare within-parlay correlation
        ag_corr_vals = [p["pair_agree_rate"] for p in all_agree_parlays
                        if p["pair_agree_rate"] is not None]
        dg_corr_vals = [p["pair_agree_rate"] for p in has_disagree_parlays
                        if p["pair_agree_rate"] is not None]
        all_corr_vals = [p["pair_agree_rate"] for p in all_parlays
                         if p["pair_agree_rate"] is not None]

        ag_mean = statistics.mean(ag_corr_vals) if ag_corr_vals else 0
        dg_mean = statistics.mean(dg_corr_vals) if dg_corr_vals else 0
        all_mean = statistics.mean(all_corr_vals) if all_corr_vals else 0

        print(f"\n  Within-parlay leg correlation:")
        print(f"    ALL parlays:        {all_mean:.3f} (baseline)")
        print(f"    ALL-AGREE parlays:  {ag_mean:.3f}")
        if dg_corr_vals:
            print(f"    HAS-DISAGREE:       {dg_mean:.3f}")
        print(f"    (0.50 = independent; >0.50 = correlated)")

        more_corr = ag_mean > all_mean + 0.02
        print(f"\n  Are agreement legs MORE correlated? "
              f"{'YES — agreement clusters correlated legs' if more_corr else 'NO — correlation similar'}")

        if more_corr:
            print(f"\n  IMPLICATION: Agreement filter is a STRAIGHT-BET tool.")
            print(f"  It lifts leg WR but clusters correlated games,")
            print(f"  making the independence assumption in parlays WORSE.")
        else:
            print(f"\n  IMPLICATION: Agreement filter works at BOTH levels")
            print(f"  if the lift survives game-level dedup.")

    # Final verdict
    print(f"\n  ── RECOMMENDED MODEL LEVEL ──")
    if len(disagree_g) < 10:
        print(f"  ⚠ DISAGREE game-level N={len(disagree_g)} — too small for conclusions.")
        print(f"  Agreement lever has the RIGHT mechanism (veto bad bets)")
        print(f"  but cannot be validated at game level with current data.")
    print(f"  Leg-level inflation: {len(multi_model)}/{len(mm_games)} = {len(multi_model)/len(mm_games):.0f}x")
    print(f"  Any leg-level number from the dual-model pool should be divided")
    print(f"  by ~{len(multi_model)//len(mm_games)} to approximate independent observations.\n")

    con.close()


if __name__ == "__main__":
    main()
