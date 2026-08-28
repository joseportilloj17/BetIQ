#!/usr/bin/env python3
"""
SYNTHESIS ("Frankenstein") Analysis
====================================
Catalogue what WORKS and FAILS across all models, test model agreement
as a discrimination lever, and design a composite targeting 60%+ WR
at constant odds.

THE METRIC: WR at held-constant odds. A WR gain at constant odds = real
discrimination. A WR gain that shortens odds = drift trap. Report WR AND
odds distribution together EVERY time.
"""

import sqlite3
import os
import statistics
from collections import defaultdict

BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")


def _wr_at_odds(legs, odds_lo, odds_hi):
    """WR for legs in a specific odds band. Returns (wr, n, avg_odds, be)."""
    sub = [l for l in legs if odds_lo <= l["odds"] < odds_hi]
    if not sub:
        return None, 0, None, None
    wins = sum(1 for l in sub if l["won"])
    wr = wins / len(sub) * 100
    avg_odds = statistics.mean([l["odds"] for l in sub])
    be = (1 / avg_odds) * 100
    return wr, len(sub), avg_odds, be


def _report_wr_with_odds(label, legs, indent=2):
    """Report WR alongside odds distribution to detect drift."""
    if not legs:
        print(f"{'  '*indent}{label}: NO DATA")
        return
    wins = sum(1 for l in legs if l["won"])
    wr = wins / len(legs) * 100
    avg_odds = statistics.mean([l["odds"] for l in legs])
    med_odds = statistics.median([l["odds"] for l in legs])
    be = (1 / avg_odds) * 100
    edge = wr - be
    pfx = "  " * indent
    print(f"{pfx}{label}: WR={wr:.1f}% N={len(legs)} "
          f"avg_odds={avg_odds:.3f} med_odds={med_odds:.3f} "
          f"BE={be:.1f}% edge={edge:+.1f}pp")


def main():
    con = sqlite3.connect(BETS_DB)
    con.row_factory = sqlite3.Row

    print("=" * 74)
    print("SYNTHESIS ANALYSIS — THE FRANKENSTEIN")
    print("Target: 60%+ WR at constant odds (real discrimination, not drift)")
    print("=" * 74)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0: ESTABLISH CURRENT ODDS RANGE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("STEP 0: CURRENT ODDS RANGE (the reference)")
    print("=" * 74)

    # Load all settled MLB spread legs from exploration + ats_ungated (the
    # broad forward universe). Include negbin legs for cross-model comparison.
    all_legs = []
    rows = con.execute("""
        SELECT mb.source, mb.odds, mb.status, mb.game_date, mb.cohort,
               mbl.win_prob, mbl.leg_result, mbl.market_type,
               mbl.predicted_edge_pp, mbl.fixture_id,
               mbl.clv_cents, mbl.clv_available,
               mbl.ats_dist_wp, mbl.negbin_wp, mbl.negbin_edge_pp,
               mbl.bayes_team_wp, mbl.ridge_composite,
               mbl.description, mbl.id as leg_id
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.sport = 'MLB'
          AND mbl.market_type IN ('alternate_spreads', 'spreads')
    """).fetchall()

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
            "bayes_wp": r["bayes_team_wp"],
            "ridge": r["ridge_composite"],
            "clv_cents": r["clv_cents"],
            "clv_avail": r["clv_available"],
            "game_date": r["game_date"],
            "cohort": r["cohort"],
            "desc": r["description"],
            "fixture_id": r["fixture_id"],
            "leg_id": r["leg_id"],
        })

    print(f"\n  Total settled MLB spread legs: {len(all_legs)}")

    # The broad forward set: exploration + ats_ungated (clean cohort)
    forward = [l for l in all_legs if l["cohort"] == "clean"
               and l["source"] in ("exploration", "ats_ungated", "simulation")]
    print(f"  Forward set (clean cohort, explor+ats+sim): {len(forward)}")

    # Current odds distribution
    all_odds = [l["odds"] for l in forward if l["odds"] > 1]
    if all_odds:
        all_odds.sort()
        p10 = all_odds[int(len(all_odds) * 0.10)]
        p25 = all_odds[int(len(all_odds) * 0.25)]
        p50 = all_odds[int(len(all_odds) * 0.50)]
        p75 = all_odds[int(len(all_odds) * 0.75)]
        p90 = all_odds[int(len(all_odds) * 0.90)]
        print(f"\n  Current odds distribution (forward set):")
        print(f"    P10={p10:.3f}  P25={p25:.3f}  MEDIAN={p50:.3f}  "
              f"P75={p75:.3f}  P90={p90:.3f}")
        print(f"    Mean={statistics.mean(all_odds):.3f}")

    # WR by odds band
    print(f"\n  WR by odds band (forward set):")
    BANDS = [(1.0, 1.3, "1.00-1.30 (heavy chalk)"),
             (1.3, 1.6, "1.30-1.60 (moderate fav)"),
             (1.6, 2.0, "1.60-2.00 (slight fav)"),
             (2.0, 2.5, "2.00-2.50 (pick'em/slight dog)"),
             (2.5, 3.5, "2.50-3.50 (underdog)"),
             (3.5, 10.0, "3.50+ (longshot)")]
    for lo, hi, label in BANDS:
        wr, n, avg_o, be = _wr_at_odds(forward, lo, hi)
        if wr is not None:
            print(f"    {label:35s} WR={wr:5.1f}% N={n:5d} "
                  f"BE={be:5.1f}% edge={wr-be:+5.1f}pp")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: WHAT WORKS (validated discrimination at constant odds)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("PART 1: WHAT WORKS (validated, at constant odds)")
    print("=" * 74)

    # 1A. dist_wp calibration buckets
    print("\n  ── 1A. dist_wp CALIBRATION ──")
    has_dist = [l for l in forward if l["ats_dist_wp"] is not None and l["ats_dist_wp"] > 0]
    if has_dist:
        print(f"  Legs with ats_dist_wp: {len(has_dist)}")
        for lo_wp, hi_wp, label in [(0, 0.35, "<35%"), (0.35, 0.45, "35-45%"),
                                     (0.45, 0.55, "45-55%"), (0.55, 0.65, "55-65%"),
                                     (0.65, 0.75, "65-75%"), (0.75, 1.01, "75%+")]:
            sub = [l for l in has_dist if lo_wp <= l["ats_dist_wp"] < hi_wp]
            _report_wr_with_odds(f"dist_wp {label}", sub)
    else:
        # Try win_prob as fallback
        has_wp = [l for l in forward if l["wp"] is not None and l["wp"] > 1]
        print(f"  No ats_dist_wp in forward set. Using win_prob ({len(has_wp)} legs)")
        for lo_wp, hi_wp, label in [(0, 35, "<35%"), (35, 45, "35-45%"),
                                     (45, 55, "45-55%"), (55, 65, "55-65%"),
                                     (65, 75, "65-75%"), (75, 101, "75%+")]:
            sub = [l for l in has_wp if lo_wp <= l["wp"] < hi_wp]
            _report_wr_with_odds(f"wp {label}", sub)

    # 1B. 12+ epp cutoff
    print("\n  ── 1B. 12+ epp CUTOFF ──")
    has_epp = [l for l in forward if l["edge_pp"] is not None]
    if has_epp:
        print(f"  Legs with predicted_edge_pp: {len(has_epp)}")
        below12 = [l for l in has_epp if l["edge_pp"] < 12]
        above12 = [l for l in has_epp if l["edge_pp"] >= 12]
        _report_wr_with_odds("edge_pp < 12 (KEEP)", below12)
        _report_wr_with_odds("edge_pp >= 12 (CUT)", above12)

        # By edge_pp band
        for lo, hi, label in [(0, 3, "0-3pp"), (3, 6, "3-6pp"), (6, 9, "6-9pp"),
                               (9, 12, "9-12pp"), (12, 20, "12-20pp"), (20, 50, "20+pp")]:
            sub = [l for l in has_epp if lo <= l["edge_pp"] < hi]
            _report_wr_with_odds(f"  epp {label}", sub, indent=3)

    # Also check negbin_edge_pp on negbin legs
    negbin_legs = [l for l in all_legs if l["source"] and "negbin" in l["source"]
                   and l["negbin_edge_pp"] is not None]
    if negbin_legs:
        print(f"\n  NegBin legs with negbin_edge_pp: {len(negbin_legs)}")
        below12n = [l for l in negbin_legs if l["negbin_edge_pp"] < 12]
        above12n = [l for l in negbin_legs if l["negbin_edge_pp"] >= 12]
        _report_wr_with_odds("negbin_epp < 12 (KEEP)", below12n)
        _report_wr_with_odds("negbin_epp >= 12 (CUT)", above12n)

    # 1C. CLV analysis (does any filter beat the close?)
    print("\n  ── 1C. CLV — DOES ANYTHING BEAT THE CLOSE? ──")
    has_clv = [l for l in forward if l["clv_avail"] == 1 and l["clv_cents"] is not None]
    if has_clv:
        clv_pos = [l for l in has_clv if l["clv_cents"] < 0]  # open > close = beat line
        clv_neg = [l for l in has_clv if l["clv_cents"] > 0]
        print(f"  Legs with CLV: {len(has_clv)}")
        print(f"  CLV-positive (beat close): {len(clv_pos)} ({len(clv_pos)/len(has_clv)*100:.1f}%)")
        print(f"  Avg clv_cents: {statistics.mean([l['clv_cents'] for l in has_clv]):.1f}")
        _report_wr_with_odds("CLV-positive (beat close)", clv_pos)
        _report_wr_with_odds("CLV-negative (lost to close)", clv_neg)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2: WHAT FAILS (rules to encode)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("PART 2: WHAT FAILS (rules the composite must follow)")
    print("=" * 74)

    # 2A. Confidence-gating: high WP legs vs low WP — WR at constant odds
    print("\n  ── 2A. CONFIDENCE GATING (high WP = better?) ──")
    has_wp2 = [l for l in forward if l["wp"] is not None and l["wp"] > 1]
    if has_wp2:
        # Split into high-confidence (wp > 65) and lower (wp 45-65)
        hi_conf = [l for l in has_wp2 if l["wp"] >= 65]
        lo_conf = [l for l in has_wp2 if 45 <= l["wp"] < 65]
        _report_wr_with_odds("WP >= 65 (high confidence)", hi_conf)
        _report_wr_with_odds("WP 45-65 (moderate)", lo_conf)
        print("  → If high-conf has SHORTER odds and only marginally better WR,")
        print("    confidence-gating is drift (safer bets), not discrimination.")

    # 2B. High edge_pp = WORST WR (the inversion)
    print("\n  ── 2B. HIGH EDGE_PP INVERSION ──")
    if has_epp:
        # Already shown in 1B — the 12-20pp and 20+pp bands
        # Confirm: does edge_pp correlate NEGATIVELY with WR at constant odds?
        for odds_lo, odds_hi, band_label in [(1.3, 2.0, "odds 1.3-2.0"),
                                              (2.0, 3.0, "odds 2.0-3.0"),
                                              (3.0, 5.0, "odds 3.0-5.0")]:
            band_legs = [l for l in has_epp if odds_lo <= l["odds"] < odds_hi]
            if len(band_legs) < 50:
                continue
            lo_epp = [l for l in band_legs if l["edge_pp"] < 6]
            hi_epp = [l for l in band_legs if l["edge_pp"] >= 12]
            if lo_epp and hi_epp:
                lo_wr = sum(l["won"] for l in lo_epp) / len(lo_epp) * 100
                hi_wr = sum(l["won"] for l in hi_epp) / len(hi_epp) * 100
                print(f"    {band_label}: low_epp WR={lo_wr:.1f}% (N={len(lo_epp)}) "
                      f"vs high_epp WR={hi_wr:.1f}% (N={len(hi_epp)}) "
                      f"gap={lo_wr-hi_wr:+.1f}pp")

    # 2C. Parlay vs straight
    print("\n  ── 2C. PARLAY vs STRAIGHT ──")
    # Prospective = parlays (multi-leg), exploration = straights
    prosp = [l for l in all_legs if l["source"] == "prospective" and l["cohort"] == "clean"]
    explor = [l for l in forward if l["source"] == "exploration"]
    _report_wr_with_odds("Prospective (parlays, per-LEG WR)", prosp)
    _report_wr_with_odds("Exploration (straights, per-leg WR)", explor)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3: MODEL AGREEMENT AS DISCRIMINATION LEVER
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("PART 3: MODEL AGREEMENT / DISAGREEMENT")
    print("Does consensus lift WR at constant odds?")
    print("=" * 74)

    # Find legs that have BOTH negbin_wp AND ats_dist_wp (or win_prob)
    # These are the cross-scored legs from negbin shadow
    multi_model = [l for l in all_legs
                   if l["negbin_wp"] is not None and l["negbin_wp"] > 0
                   and l["ats_dist_wp"] is not None and l["ats_dist_wp"] > 0]
    print(f"\n  Legs with BOTH negbin_wp AND ats_dist_wp: {len(multi_model)}")

    if len(multi_model) >= 100:
        # Agreement: both models predict same direction (above/below 50%)
        # For spread bets, "agree" = both think this side covers > 50%
        # negbin_wp is stored on 0-1 scale, ats_dist_wp is on 0-1 scale
        agree = [l for l in multi_model
                 if (l["negbin_wp"] > 0.5 and l["ats_dist_wp"] > 0.5)
                 or (l["negbin_wp"] <= 0.5 and l["ats_dist_wp"] <= 0.5)]
        disagree = [l for l in multi_model
                    if (l["negbin_wp"] > 0.5) != (l["ats_dist_wp"] > 0.5)]

        print(f"  Models AGREE (both >50% or both <50%): {len(agree)}")
        print(f"  Models DISAGREE: {len(disagree)}")

        _report_wr_with_odds("ALL multi-model legs", multi_model)
        _report_wr_with_odds("AGREE (consensus)", agree)
        _report_wr_with_odds("DISAGREE (conflict)", disagree)

        # Finer agreement: both predict high (>60%)
        both_high = [l for l in multi_model
                     if l["negbin_wp"] > 0.6 and l["ats_dist_wp"] > 0.6]
        both_mod = [l for l in multi_model
                    if 0.45 <= l["negbin_wp"] <= 0.6 and 0.45 <= l["ats_dist_wp"] <= 0.6]
        if both_high:
            _report_wr_with_odds("STRONG AGREE (both >60%)", both_high)
        if both_mod:
            _report_wr_with_odds("MODERATE (both 45-60%)", both_mod)

        # WR at matched odds bands for agree vs disagree
        print("\n  Agreement vs Disagreement at CONSTANT ODDS:")
        for lo, hi, label in BANDS:
            ag = [l for l in agree if lo <= l["odds"] < hi]
            dg = [l for l in disagree if lo <= l["odds"] < hi]
            if len(ag) >= 20 and len(dg) >= 10:
                ag_wr = sum(l["won"] for l in ag) / len(ag) * 100
                dg_wr = sum(l["won"] for l in dg) / len(dg) * 100
                ag_be = (1 / statistics.mean([l["odds"] for l in ag])) * 100
                dg_be = (1 / statistics.mean([l["odds"] for l in dg])) * 100
                print(f"    {label:35s} "
                      f"AGREE: WR={ag_wr:.1f}% (N={len(ag)}) "
                      f"DISAGREE: WR={dg_wr:.1f}% (N={len(dg)}) "
                      f"lift={ag_wr-dg_wr:+.1f}pp")
    else:
        print("  Insufficient cross-model data. Trying bayes_wp + ridge...")

    # Also try: bayes_wp and ridge_composite as additional signals
    has_bayes = [l for l in forward if l["bayes_wp"] is not None and l["bayes_wp"] > 0]
    has_ridge = [l for l in forward if l["ridge"] is not None and l["ridge"] > 0]
    print(f"\n  Legs with bayes_wp: {len(has_bayes)}")
    print(f"  Legs with ridge_composite: {len(has_ridge)}")

    # Triple agreement: wp + bayes + ridge all agree on direction
    triple = [l for l in forward
              if l["wp"] is not None and l["wp"] > 1
              and l["bayes_wp"] is not None and l["bayes_wp"] > 0
              and l["ridge"] is not None and l["ridge"] > 0]
    if len(triple) >= 100:
        print(f"\n  Triple-model legs (wp + bayes + ridge): {len(triple)}")
        # All three say covers (wp>50, bayes>0.5, ridge>0.5)
        t_agree = [l for l in triple
                   if l["wp"] > 50 and l["bayes_wp"] > 0.5 and l["ridge"] > 0.5]
        t_mixed = [l for l in triple
                   if l not in t_agree]
        _report_wr_with_odds("TRIPLE AGREE (all say covers)", t_agree)
        _report_wr_with_odds("MIXED (at least one disagrees)", t_mixed)

        # At constant odds
        print("\n  Triple agreement at CONSTANT ODDS:")
        for lo, hi, label in BANDS:
            ta = [l for l in t_agree if lo <= l["odds"] < hi]
            tm = [l for l in t_mixed if lo <= l["odds"] < hi]
            if len(ta) >= 15 and len(tm) >= 15:
                ta_wr = sum(l["won"] for l in ta) / len(ta) * 100
                tm_wr = sum(l["won"] for l in tm) / len(tm) * 100
                print(f"    {label:35s} "
                      f"AGREE: WR={ta_wr:.1f}% (N={len(ta)}) "
                      f"MIXED: WR={tm_wr:.1f}% (N={len(tm)}) "
                      f"lift={ta_wr-tm_wr:+.1f}pp")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3B: MODEL DISAGREEMENT WITH MARKET
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  ── 3B. MODEL vs MARKET DISAGREEMENT ──")
    if has_epp:
        # edge_pp = model disagrees with market. We know high edge_pp = BAD.
        # Does same pattern hold for model-model disagreement?
        print("  Already shown: high edge_pp (model vs market disagree) = WORSE WR")
        print("  This is the edge_pp inversion — model wrong where it most disagrees.")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 4: THE COMPOSITE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("PART 4: COMPOSITE DESIGN — 60%+ WR AT CONSTANT ODDS")
    print("=" * 74)

    # Build composite filters on the forward set
    # Filter 1: epp < 12 (proven)
    # Filter 2: model agreement (if Part 3 shows lift)
    # Filter 3: specific odds band where edge exists
    # Filter 4: dist_wp / wp calibration bucket

    composite_candidates = forward  # start with full forward set

    print(f"\n  Baseline (full forward set):")
    _report_wr_with_odds("ALL", composite_candidates, indent=2)

    # Apply filters incrementally
    # Filter 1: epp < 12
    f1 = [l for l in composite_candidates
          if l["edge_pp"] is None or l["edge_pp"] < 12]
    print(f"\n  After Filter 1 (epp < 12):")
    _report_wr_with_odds("epp<12", f1, indent=2)

    # Filter 2: wp >= 50 (model says covers)
    f2 = [l for l in f1 if l["wp"] is not None and l["wp"] >= 50]
    print(f"\n  After Filter 2 (wp >= 50):")
    _report_wr_with_odds("wp>=50", f2, indent=2)

    # Filter 3: wp between 50 and 70 (avoid extreme confidence = drift)
    f3 = [l for l in f1 if l["wp"] is not None and 50 <= l["wp"] <= 70]
    print(f"\n  After Filter 3 (wp 50-70, avoid overconfidence):")
    _report_wr_with_odds("wp 50-70", f3, indent=2)

    # Filter 4: moderate epp only (3-9pp sweet spot)
    f4 = [l for l in f1 if l["edge_pp"] is not None and 3 <= l["edge_pp"] < 9]
    print(f"\n  After Filter 4 (epp 3-9pp sweet spot):")
    _report_wr_with_odds("epp 3-9", f4, indent=2)

    # Composite A: epp < 12 + wp 50-70
    compA = [l for l in f1 if l["wp"] is not None and 50 <= l["wp"] <= 70]
    print(f"\n  Composite A (epp<12 + wp 50-70):")
    _report_wr_with_odds("Comp A", compA, indent=2)
    for lo, hi, label in BANDS:
        sub = [l for l in compA if lo <= l["odds"] < hi]
        if sub:
            _report_wr_with_odds(f"  {label}", sub, indent=3)

    # Composite B: epp 3-9 + wp 50-65
    compB = [l for l in f1 if l["edge_pp"] is not None and 3 <= l["edge_pp"] < 9
             and l["wp"] is not None and 50 <= l["wp"] <= 65]
    print(f"\n  Composite B (epp 3-9 + wp 50-65):")
    _report_wr_with_odds("Comp B", compB, indent=2)

    # Triple-agreement composite (if available)
    if len(triple) >= 100:
        compC = [l for l in t_agree
                 if l["edge_pp"] is None or l["edge_pp"] < 12]
        print(f"\n  Composite C (triple agree + epp<12):")
        _report_wr_with_odds("Comp C", compC, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # SPLIT-HALF STABILITY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("SPLIT-HALF STABILITY (best composite)")
    print("=" * 74)

    # Use Composite A for split-half
    best_comp = compA
    if len(best_comp) >= 40:
        half = len(best_comp) // 2
        h1, h2 = best_comp[:half], best_comp[half:]
        h1_wr = sum(l["won"] for l in h1) / len(h1) * 100
        h2_wr = sum(l["won"] for l in h2) / len(h2) * 100
        h1_odds = statistics.mean([l["odds"] for l in h1])
        h2_odds = statistics.mean([l["odds"] for l in h2])
        print(f"\n  H1: WR={h1_wr:.1f}% (N={len(h1)}, avg_odds={h1_odds:.3f})")
        print(f"  H2: WR={h2_wr:.1f}% (N={len(h2)}, avg_odds={h2_odds:.3f})")
        print(f"  Gap: {abs(h1_wr - h2_wr):.1f}pp (WR), "
              f"{abs(h1_odds - h2_odds):.3f} (odds)")

    # ═══════════════════════════════════════════════════════════════════════
    # PER-SOURCE COMPARISON AT CONSTANT ODDS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("SOURCE COMPARISON AT CONSTANT ODDS")
    print("=" * 74)

    sources_of_interest = ["exploration", "ats_ungated", "simulation",
                           "shadow_negbin", "shadow_negbin_edgegate",
                           "shadow_negbin_fullstack"]
    for src in sources_of_interest:
        src_legs = [l for l in all_legs if l["source"] == src]
        if len(src_legs) >= 50:
            _report_wr_with_odds(src, src_legs, indent=1)

    # ═══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)

    baseline_wr = sum(l["won"] for l in forward) / len(forward) * 100 if forward else 0
    baseline_odds = statistics.mean([l["odds"] for l in forward]) if forward else 0
    baseline_be = (1 / baseline_odds) * 100 if baseline_odds > 0 else 0

    compA_wr = sum(l["won"] for l in compA) / len(compA) * 100 if compA else 0
    compA_odds = statistics.mean([l["odds"] for l in compA]) if compA else 0
    compA_be = (1 / compA_odds) * 100 if compA_odds > 0 else 0

    print(f"""
  BASELINE (full forward):
    WR={baseline_wr:.1f}%, avg_odds={baseline_odds:.3f}, BE={baseline_be:.1f}%
    edge={baseline_wr-baseline_be:+.1f}pp, N={len(forward)}

  COMPOSITE A (epp<12 + wp 50-70):
    WR={compA_wr:.1f}%, avg_odds={compA_odds:.3f}, BE={compA_be:.1f}%
    edge={compA_wr-compA_be:+.1f}pp, N={len(compA)}

  WR LIFT: {compA_wr-baseline_wr:+.1f}pp
  ODDS SHIFT: {compA_odds-baseline_odds:+.3f} ({'DRIFT — shorter odds' if compA_odds < baseline_odds - 0.05 else 'HELD — real discrimination' if abs(compA_odds - baseline_odds) <= 0.05 else 'LONGER odds — extra real'})

  VALIDATED ELEMENTS:
    ✓ dist_wp calibration (monotonic WR across buckets)
    ✓ 12+ epp cutoff (toxic overconfidence, proven)
    ✓ Straight > parlay (correlation invalidates joint prob)

  HYPOTHESIZED (needs forward confirmation):
    ? Model agreement as discrimination filter
    ? epp 3-9pp sweet spot
    ? Underdog cross-book lag (shadow accumulating)

  60%+ WR AT CONSTANT ODDS:
    {'ACHIEVED' if compA_wr >= 60 and abs(compA_odds - baseline_odds) <= 0.1 else 'NOT YET — current best composite does not reach 60% at held odds'}
""")

    con.close()


if __name__ == "__main__":
    main()
