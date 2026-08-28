#!/usr/bin/env python3
"""
PER-RUNG SYNTHESIS — Open-ended investigation
==============================================
Find what WORKS and DOESN'T across features, methodology, and structure,
at the level each signal actually lives.

Dedup method: one observation per game-team per rung.
This removes parlay-based duplication (same game-rung in multiple parlays)
WITHOUT destroying rung-level signal (different rungs of same game kept separate).

For significance: report N = unique game-teams at each rung (true independent
observations), and flag when N < 30.
"""

import sqlite3
import os
import re
import statistics
import math
from collections import defaultdict

BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")

RUNGS = ["-1.5", "+1.5", "+2.5", "+3.5", "+4.5"]

ODDS_BANDS = [
    (1.0, 1.3, "1.00-1.30"),
    (1.3, 1.6, "1.30-1.60"),
    (1.6, 2.0, "1.60-2.00"),
    (2.0, 2.5, "2.00-2.50"),
    (2.5, 3.5, "2.50-3.50"),
    (3.5, 10.0, "3.50+"),
]


def _extract_rung(desc):
    if not desc:
        return "?"
    m = re.search(r'([+-][\d.]+)(?:\s*\(.*\))?\s*$', desc)
    return m.group(1) if m else "?"


def _extract_team(desc):
    if not desc:
        return "unknown"
    # Remove trailing "(Spread)" etc then extract team
    cleaned = re.sub(r'\s*\(.*?\)\s*$', '', desc)
    m = re.match(r'^(.+?)\s+[+-][\d.]+$', cleaned)
    return m.group(1) if m else desc


def _game_rung_key(leg):
    """One key per game-team-rung (dedup within rung, across parlays)."""
    return f"{leg['fixture_id']}_{_extract_team(leg['desc'])}_{_extract_rung(leg['desc'])}"


def _dedup_within_rung(legs):
    """One observation per game-team-rung. Average model scores, majority-vote outcome."""
    by_key = defaultdict(list)
    for l in legs:
        by_key[_game_rung_key(l)].append(l)

    deduped = []
    for key, group in by_key.items():
        wins = sum(1 for l in group if l["won"])
        rep = {
            "fixture_id": group[0]["fixture_id"],
            "desc": group[0]["desc"],
            "rung": _extract_rung(group[0]["desc"]),
            "team": _extract_team(group[0]["desc"]),
            "source": group[0]["source"],
            "game_date": group[0]["game_date"],
            "won": group[0]["won"],  # All copies of same game-team-rung have same outcome
            "odds": statistics.mean([l["odds"] for l in group]),
            "n_copies": len(group),
        }
        # Average each feature across copies
        for feat in ["wp", "negbin_wp", "ats_dist_wp", "negbin_edge_pp",
                     "edge_pp", "bayes_wp", "ridge", "mkt_disagree", "devig_entry"]:
            vals = [l[feat] for l in group if l.get(feat) is not None
                    and (not isinstance(l.get(feat), (int, float)) or l[feat] != 0)]
            rep[feat] = statistics.mean(vals) if vals else None
        rep["negbin_cell"] = group[0].get("negbin_cell")
        deduped.append(rep)
    return deduped


def _wr(legs):
    """Returns (wr%, n, avg_odds, be%, edge_pp) or None if empty."""
    if not legs:
        return None
    wins = sum(1 for l in legs if l["won"])
    n = len(legs)
    wr = wins / n * 100
    avg_odds = statistics.mean([l["odds"] for l in legs])
    be = (1 / avg_odds) * 100
    return wr, n, avg_odds, be, wr - be


def _wr_line(label, legs, indent=2):
    """Print one WR line."""
    r = _wr(legs)
    pfx = "  " * indent
    if r is None:
        print(f"{pfx}{label}: NO DATA")
        return
    wr, n, avg_o, be, edge = r
    flag = " ⚠LOW-N" if n < 30 else ""
    print(f"{pfx}{label}: WR={wr:.1f}% N={n} "
          f"odds={avg_o:.3f} BE={be:.1f}% edge={edge:+.1f}pp{flag}")


def _se_prop(wr_pct, n):
    """Standard error of a proportion (percentage points)."""
    p = wr_pct / 100
    if n == 0 or p <= 0 or p >= 1:
        return float('inf')
    return math.sqrt(p * (1 - p) / n) * 100


def main():
    con = sqlite3.connect(BETS_DB)
    con.row_factory = sqlite3.Row

    # Load all NegBin spread legs (the feature-rich pool)
    rows = con.execute("""
        SELECT mb.source, mb.odds, mb.game_date, mb.cohort,
               mb.id as bet_id, mb.bet_type, mb.actual_profit, mb.amount,
               mbl.win_prob, mbl.leg_result, mbl.market_type,
               mbl.predicted_edge_pp, mbl.fixture_id,
               mbl.ats_dist_wp, mbl.negbin_wp, mbl.negbin_edge_pp,
               mbl.bayes_team_wp, mbl.ridge_composite,
               mbl.mkt_disagree_dist, mbl.negbin_cell,
               mbl.devig_entry_prob,
               mbl.description, mbl.id as leg_id
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.sport = 'MLB'
          AND mbl.market_type IN ('alternate_spreads', 'spreads')
          AND mb.source LIKE '%negbin%'
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
            "bayes_wp": r["bayes_team_wp"],
            "ridge": r["ridge_composite"],
            "mkt_disagree": r["mkt_disagree_dist"],
            "negbin_cell": r["negbin_cell"],
            "devig_entry": r["devig_entry_prob"],
            "game_date": r["game_date"],
            "cohort": r["cohort"],
            "desc": r["description"],
            "fixture_id": r["fixture_id"],
            "leg_id": r["leg_id"],
            "bet_id": r["bet_id"],
            "bet_type": r["bet_type"],
            "actual_profit": r["actual_profit"],
            "amount": r["amount"],
        })

    # Also load forward ATS (±1.5 only, separate pool)
    fwd_rows = con.execute("""
        SELECT mb.source, mb.odds, mb.game_date, mb.cohort,
               mb.id as bet_id, mb.bet_type,
               mbl.win_prob, mbl.leg_result,
               mbl.predicted_edge_pp, mbl.fixture_id,
               mbl.devig_entry_prob,
               mbl.description, mbl.id as leg_id
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.sport = 'MLB'
          AND mbl.market_type IN ('alternate_spreads', 'spreads')
          AND mb.cohort = 'clean'
          AND mb.source IN ('exploration', 'ats_ungated', 'simulation')
    """).fetchall()
    fwd_legs = []
    for r in fwd_rows:
        fwd_legs.append({
            "source": r["source"], "odds": r["odds"] or 1.0,
            "won": r["leg_result"] == "WIN", "wp": r["win_prob"],
            "edge_pp": r["predicted_edge_pp"], "fixture_id": r["fixture_id"],
            "devig_entry": r["devig_entry_prob"],
            "desc": r["description"], "game_date": r["game_date"],
            "cohort": r["cohort"], "bet_id": r["bet_id"], "bet_type": r["bet_type"],
            "negbin_wp": None, "ats_dist_wp": None, "negbin_edge_pp": None,
            "bayes_wp": None, "ridge": None, "mkt_disagree": None,
            "negbin_cell": None, "leg_id": r["leg_id"],
        })

    print("=" * 78)
    print("PER-RUNG SYNTHESIS — OPEN-ENDED INVESTIGATION")
    print("One observation per game-team-rung (removes parlay duplication,")
    print("preserves rung-level signal)")
    print("=" * 78)

    # ===================================================================
    # SECTION 0: BASE RATES AND ODDS BY RUNG (methodology check)
    # ===================================================================
    print("\n" + "=" * 78)
    print("SECTION 0: BASE RATES AND ODDS BY RUNG")
    print("=" * 78)

    by_rung_raw = defaultdict(list)
    for l in all_legs:
        by_rung_raw[_extract_rung(l["desc"])].append(l)

    by_rung = {}
    for rung in RUNGS:
        raw = by_rung_raw.get(rung, [])
        deduped = _dedup_within_rung(raw)
        by_rung[rung] = deduped

    print(f"\n  {'Rung':6s} {'N_legs':>7s} {'N_games':>8s} {'Infl':>5s} "
          f"{'WR':>6s} {'Avg_odds':>9s} {'Med_odds':>9s} {'BE':>6s} {'Edge':>7s}")
    print(f"  {'─'*6} {'─'*7} {'─'*8} {'─'*5} {'─'*6} {'─'*9} {'─'*9} {'─'*6} {'─'*7}")
    for rung in RUNGS:
        raw = by_rung_raw.get(rung, [])
        ded = by_rung[rung]
        if not ded:
            continue
        r = _wr(ded)
        med_odds = statistics.median([d["odds"] for d in ded])
        infl = len(raw) / len(ded) if ded else 0
        print(f"  {rung:6s} {len(raw):7d} {len(ded):8d} {infl:5.1f}x "
              f"{r[0]:5.1f}% {r[2]:9.3f} {med_odds:9.3f} {r[3]:5.1f}% {r[4]:+6.1f}pp")

    # Forward ATS ±1.5
    fwd_deduped = _dedup_within_rung(fwd_legs)
    print(f"\n  Forward ATS (±1.5 only):")
    # Split +1.5 and -1.5
    fwd_plus = [l for l in fwd_deduped if "+1.5" in (l.get("rung") or l["desc"])]
    fwd_minus = [l for l in fwd_deduped if "-1.5" in (l.get("rung") or l["desc"])]
    fwd_unk = [l for l in fwd_deduped if l not in fwd_plus and l not in fwd_minus]
    if fwd_plus:
        _wr_line("ATS +1.5", fwd_plus)
    if fwd_minus:
        _wr_line("ATS -1.5", fwd_minus)
    if fwd_unk:
        _wr_line("ATS (no rung)", fwd_unk)
    _wr_line("ATS ALL", fwd_deduped)

    # ===================================================================
    # SECTION 1: FEATURE DISCRIMINATION PER RUNG
    # ===================================================================
    print("\n" + "=" * 78)
    print("SECTION 1: FEATURE DISCRIMINATION PER RUNG")
    print("(deduped to game-team-rung level)")
    print("=" * 78)

    for rung in RUNGS:
        ded = by_rung[rung]
        if len(ded) < 30:
            print(f"\n  ── RUNG {rung}: N={len(ded)} (too few, skipping) ──")
            continue

        print(f"\n  ── RUNG {rung}: N={len(ded)} game-teams ──")

        # 1A. negbin_wp buckets
        has_negbin = [d for d in ded if d["negbin_wp"] is not None]
        if len(has_negbin) >= 30:
            print(f"\n    negbin_wp discrimination (N={len(has_negbin)}):")
            for lo, hi, label in [(0.5, 0.65, "50-65%"), (0.65, 0.75, "65-75%"),
                                   (0.75, 0.85, "75-85%"), (0.85, 1.01, "85%+")]:
                sub = [d for d in has_negbin if lo <= d["negbin_wp"] < hi]
                _wr_line(f"negbin_wp {label}", sub, indent=3)

        # 1B. ats_dist_wp buckets
        has_ats = [d for d in ded if d["ats_dist_wp"] is not None]
        if len(has_ats) >= 30:
            print(f"\n    ats_dist_wp discrimination (N={len(has_ats)}):")
            for lo, hi, label in [(0, 0.40, "<40%"), (0.40, 0.50, "40-50%"),
                                   (0.50, 0.60, "50-60%"), (0.60, 0.70, "60-70%"),
                                   (0.70, 0.80, "70-80%"), (0.80, 1.01, "80%+")]:
                sub = [d for d in has_ats if lo <= d["ats_dist_wp"] < hi]
                _wr_line(f"ats_wp {label}", sub, indent=3)

        # 1C. negbin_edge_pp bands
        has_epp = [d for d in ded if d["negbin_edge_pp"] is not None]
        if len(has_epp) >= 30:
            print(f"\n    negbin_edge_pp bands (N={len(has_epp)}):")
            for lo, hi, label in [(0, 3, "0-3pp"), (3, 6, "3-6pp"), (6, 9, "6-9pp"),
                                   (9, 12, "9-12pp"), (12, 20, "12-20pp"), (20, 100, "20+pp")]:
                sub = [d for d in has_epp if lo <= d["negbin_edge_pp"] < hi]
                _wr_line(f"epp {label}", sub, indent=3)

        # 1D. Model agreement (only meaningful at +1.5)
        has_both = [d for d in ded if d["negbin_wp"] is not None and d["ats_dist_wp"] is not None]
        if len(has_both) >= 20:
            agree = [d for d in has_both
                     if (d["negbin_wp"] > 0.5 and d["ats_dist_wp"] > 0.5)
                     or (d["negbin_wp"] <= 0.5 and d["ats_dist_wp"] <= 0.5)]
            disagree = [d for d in has_both
                        if (d["negbin_wp"] > 0.5) != (d["ats_dist_wp"] > 0.5)]
            print(f"\n    Model agreement (N={len(has_both)}, "
                  f"agree={len(agree)}, disagree={len(disagree)}):")
            _wr_line("AGREE", agree, indent=3)
            _wr_line("DISAGREE", disagree, indent=3)

            # Game-clustered significance for disagree
            if agree and disagree:
                ag_r = _wr(agree)
                dg_r = _wr(disagree)
                if ag_r and dg_r:
                    lift = ag_r[0] - dg_r[0]
                    # SE of difference
                    se_diff = math.sqrt(_se_prop(ag_r[0], ag_r[1])**2 +
                                        _se_prop(dg_r[0], dg_r[1])**2)
                    z = lift / se_diff if se_diff > 0 else 0
                    print(f"      Lift: {lift:+.1f}pp, SE={se_diff:.1f}pp, "
                          f"z={z:.2f} {'(p<0.05)' if abs(z) > 1.96 else '(n.s.)'}")

            # At constant odds within this rung
            if disagree and len(disagree) >= 5:
                print(f"      At constant odds within {rung}:")
                for olo, ohi, olabel in ODDS_BANDS:
                    ag_b = [d for d in agree if olo <= d["odds"] < ohi]
                    dg_b = [d for d in disagree if olo <= d["odds"] < ohi]
                    if len(ag_b) >= 10 and len(dg_b) >= 3:
                        ag_wr = sum(d["won"] for d in ag_b) / len(ag_b) * 100
                        dg_wr = sum(d["won"] for d in dg_b) / len(dg_b) * 100
                        print(f"        {olabel}: AGREE {ag_wr:.1f}% (N={len(ag_b)}) "
                              f"DISAGREE {dg_wr:.1f}% (N={len(dg_b)}) "
                              f"lift={ag_wr-dg_wr:+.1f}pp")

        # 1E. negbin_cell
        has_cell = [d for d in ded if d["negbin_cell"] is not None]
        if len(has_cell) >= 30:
            cells = defaultdict(list)
            for d in has_cell:
                cells[d["negbin_cell"]].append(d)
            print(f"\n    negbin_cell (N={len(has_cell)}):")
            for cell in sorted(cells.keys()):
                _wr_line(cell, cells[cell], indent=3)

        # 1F. mkt_disagree_dist
        has_mkt = [d for d in ded if d["mkt_disagree"] is not None]
        if len(has_mkt) >= 30:
            med_mkt = statistics.median([d["mkt_disagree"] for d in has_mkt])
            lo_mkt = [d for d in has_mkt if d["mkt_disagree"] <= med_mkt]
            hi_mkt = [d for d in has_mkt if d["mkt_disagree"] > med_mkt]
            print(f"\n    mkt_disagree_dist (median={med_mkt:.3f}, N={len(has_mkt)}):")
            _wr_line("Low disagree (model≈market)", lo_mkt, indent=3)
            _wr_line("High disagree (model≠market)", hi_mkt, indent=3)

        # 1G. bayes_wp
        has_bayes = [d for d in ded if d["bayes_wp"] is not None]
        if len(has_bayes) >= 30:
            print(f"\n    bayes_team_wp (N={len(has_bayes)}):")
            for lo, hi, label in [(0, 0.40, "<40%"), (0.40, 0.50, "40-50%"),
                                   (0.50, 0.60, "50-60%"), (0.60, 1.01, "60%+")]:
                sub = [d for d in has_bayes if lo <= d["bayes_wp"] < hi]
                _wr_line(f"bayes_wp {label}", sub, indent=3)

        # 1H. ridge_composite
        has_ridge = [d for d in ded if d["ridge"] is not None]
        if len(has_ridge) >= 30:
            med_ridge = statistics.median([d["ridge"] for d in has_ridge])
            lo_ridge = [d for d in has_ridge if d["ridge"] <= med_ridge]
            hi_ridge = [d for d in has_ridge if d["ridge"] > med_ridge]
            print(f"\n    ridge_composite (median={med_ridge:.3f}, N={len(has_ridge)}):")
            _wr_line("Below median", lo_ridge, indent=3)
            _wr_line("Above median", hi_ridge, indent=3)

    # ===================================================================
    # SECTION 2: CROSS-RUNG STRUCTURE
    # ===================================================================
    print("\n" + "=" * 78)
    print("SECTION 2: CROSS-RUNG STRUCTURE")
    print("Same game, different rungs — are outcomes correlated?")
    print("=" * 78)

    # For each game-team, collect outcomes at each rung
    by_game_team = defaultdict(dict)  # {fixture_team: {rung: won}}
    for rung in RUNGS:
        for d in by_rung[rung]:
            gt_key = f"{d['fixture_id']}_{d['team']}"
            by_game_team[gt_key][rung] = d["won"]

    # Cross-rung correlation: +1.5 vs +4.5 outcomes
    pairs_15_45 = []
    for gt, outcomes in by_game_team.items():
        if "+1.5" in outcomes and "+4.5" in outcomes:
            pairs_15_45.append((outcomes["+1.5"], outcomes["+4.5"]))

    if pairs_15_45:
        both_win = sum(1 for a, b in pairs_15_45 if a and b)
        both_lose = sum(1 for a, b in pairs_15_45 if not a and not b)
        only_45 = sum(1 for a, b in pairs_15_45 if not a and b)
        only_15 = sum(1 for a, b in pairs_15_45 if a and not b)
        n = len(pairs_15_45)
        agree = (both_win + both_lose) / n * 100
        print(f"\n  +1.5 vs +4.5 outcomes (N={n} game-teams):")
        print(f"    Both WIN:  {both_win:4d} ({both_win/n*100:.1f}%)")
        print(f"    Both LOSE: {both_lose:4d} ({both_lose/n*100:.1f}%)")
        print(f"    +4.5 only: {only_45:4d} ({only_45/n*100:.1f}%)  ← team won by 1-3")
        print(f"    +1.5 only: {only_15:4d} ({only_15/n*100:.1f}%)  ← impossible (if +1.5 covers, +4.5 must)")
        print(f"    Agreement: {agree:.1f}%")
        print(f"    NOTE: +4.5 ⊃ +1.5 (if +1.5 covers, +4.5 always covers)")
        print(f"    The '+4.5 only' group = teams that won by 1-3 runs (cover +4.5, miss +1.5)")

    # +1.5 vs +2.5
    pairs_15_25 = []
    for gt, outcomes in by_game_team.items():
        if "+1.5" in outcomes and "+2.5" in outcomes:
            pairs_15_25.append((outcomes["+1.5"], outcomes["+2.5"]))
    if pairs_15_25:
        both_win = sum(1 for a, b in pairs_15_25 if a and b)
        both_lose = sum(1 for a, b in pairs_15_25 if not a and not b)
        only_25 = sum(1 for a, b in pairs_15_25 if not a and b)
        n = len(pairs_15_25)
        print(f"\n  +1.5 vs +2.5 outcomes (N={n} game-teams):")
        print(f"    Both WIN:  {both_win:4d} ({both_win/n*100:.1f}%)")
        print(f"    Both LOSE: {both_lose:4d} ({both_lose/n*100:.1f}%)")
        print(f"    +2.5 only: {only_25:4d} ({only_25/n*100:.1f}%)  ← team won by exactly 1")
        print(f"    Agreement: {(both_win+both_lose)/n*100:.1f}%")

    # ===================================================================
    # SECTION 3: METHODOLOGY AUDIT
    # ===================================================================
    print("\n" + "=" * 78)
    print("SECTION 3: METHODOLOGY AUDIT")
    print("=" * 78)

    # 3A. Base rate by rung — is BE always ~50%?
    print(f"\n  ── 3A. BASE RATE CHECK ──")
    print(f"  Higher rungs SHOULD have higher WR (easier to cover +4.5 than +1.5)")
    print(f"  If WR scales with rung but odds also shorten, edge may be flat")
    for rung in RUNGS:
        ded = by_rung[rung]
        if not ded:
            continue
        r = _wr(ded)
        se = _se_prop(r[0], r[1])
        print(f"    {rung}: WR={r[0]:.1f}% ± {se:.1f}pp, odds={r[2]:.3f}, "
              f"BE={r[3]:.1f}%, edge={r[4]:+.1f}pp")

    # 3B. Does the model discriminate WITHIN each rung at constant odds?
    print(f"\n  ── 3B. WITHIN-RUNG DISCRIMINATION (negbin_wp, constant odds) ──")
    print(f"  For edge to be real: high negbin_wp should beat low negbin_wp")
    print(f"  at the SAME odds (not just because high-wp gets shorter odds)")
    for rung in RUNGS:
        ded = by_rung[rung]
        has_negbin = [d for d in ded if d["negbin_wp"] is not None]
        if len(has_negbin) < 50:
            continue
        med_wp = statistics.median([d["negbin_wp"] for d in has_negbin])
        hi = [d for d in has_negbin if d["negbin_wp"] > med_wp]
        lo = [d for d in has_negbin if d["negbin_wp"] <= med_wp]
        hi_r = _wr(hi)
        lo_r = _wr(lo)
        if hi_r and lo_r:
            lift = hi_r[0] - lo_r[0]
            odds_shift = hi_r[2] - lo_r[2]
            print(f"    {rung}: hi_wp WR={hi_r[0]:.1f}% (N={hi_r[1]}, odds={hi_r[2]:.3f}) "
                  f"vs lo_wp WR={lo_r[0]:.1f}% (N={lo_r[1]}, odds={lo_r[2]:.3f}) "
                  f"lift={lift:+.1f}pp odds_shift={odds_shift:+.3f} "
                  f"{'DRIFT' if abs(odds_shift) > 0.1 and lift * odds_shift < 0 else 'REAL' if lift > 5 and abs(odds_shift) < 0.1 else 'CHECK'}")

    # 3C. Is edge from the MODEL or from BET SELECTION (the parlay assembly)?
    print(f"\n  ── 3C. MODEL vs SELECTION EDGE ──")
    print(f"  NegBin assembly picks specific legs for parlays.")
    print(f"  Does the model's edge come from negbin_wp accuracy,")
    print(f"  or from the assembly's game/rung selection?")
    for rung in RUNGS:
        ded = by_rung[rung]
        if len(ded) < 50:
            continue
        # Compare: all game-teams at this rung vs those actually selected for parlays
        # (all are selected — they're all from NegBin source)
        # Instead: does negbin_wp predict outcome BETTER than raw odds imply?
        has_both = [d for d in ded if d["negbin_wp"] is not None]
        if len(has_both) < 50:
            continue
        # Calibration: is negbin_wp = actual WR?
        buckets = [(0.5, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
        cal_points = []
        for lo, hi in buckets:
            sub = [d for d in has_both if lo <= d["negbin_wp"] < hi]
            if sub:
                predicted = statistics.mean([d["negbin_wp"] for d in sub]) * 100
                actual = sum(d["won"] for d in sub) / len(sub) * 100
                cal_points.append((predicted, actual, len(sub)))
        if cal_points:
            print(f"    {rung} calibration:")
            for pred, act, n in cal_points:
                print(f"      predicted={pred:.1f}% actual={act:.1f}% gap={act-pred:+.1f}pp N={n}")

    # ===================================================================
    # SECTION 4: STRAIGHT vs PARLAY BY RUNG
    # ===================================================================
    print("\n" + "=" * 78)
    print("SECTION 4: STRAIGHT-BET vs PARLAY EV BY RUNG")
    print("=" * 78)

    # Build parlay-level outcomes grouped by which rungs they contain
    by_bet = defaultdict(list)
    for l in all_legs:
        by_bet[l["bet_id"]].append(l)

    parlay_by_rung_set = defaultdict(list)
    for bet_id, legs in by_bet.items():
        if len(legs) < 2:
            continue
        rung_set = frozenset(_extract_rung(l["desc"]) for l in legs)
        parlay_won = all(l["won"] for l in legs)
        parlay_by_rung_set[rung_set].append({
            "won": parlay_won,
            "odds": legs[0]["odds"],
            "rung_set": rung_set,
            "n_legs": len(legs),
        })

    print(f"\n  Parlay rung composition:")
    for rung_set in sorted(parlay_by_rung_set.keys(), key=lambda s: str(sorted(s))):
        parlays = parlay_by_rung_set[rung_set]
        if len(parlays) < 20:
            continue
        r = _wr(parlays)
        print(f"    {str(sorted(rung_set)):40s} N={r[1]:5d} WR={r[0]:.1f}% "
              f"odds={r[2]:.3f} BE={r[3]:.1f}% edge={r[4]:+.1f}pp")

    # Straight-bet EV per rung (leg-level, but game-deduped)
    print(f"\n  Straight-bet proxy (game-deduped leg WR per rung):")
    for rung in RUNGS:
        ded = by_rung[rung]
        if len(ded) < 30:
            continue
        r = _wr(ded)
        # Estimate straight-bet ROI: (WR * avg_odds - 1) * 100
        roi = (r[0]/100 * r[2] - 1) * 100
        print(f"    {rung}: WR={r[0]:.1f}% odds={r[2]:.3f} → "
              f"flat-bet ROI={roi:+.1f}%")

    # ===================================================================
    # SECTION 5: WHAT WORKS / WHAT DOESN'T — SUMMARY
    # ===================================================================
    print("\n" + "=" * 78)
    print("SECTION 5: WHAT WORKS / WHAT DOESN'T — PER RUNG")
    print("(at game-team-rung level, with game-clustered N)")
    print("=" * 78)

    for rung in RUNGS:
        ded = by_rung[rung]
        if len(ded) < 30:
            continue
        print(f"\n  ═══ RUNG {rung} (N={len(ded)} game-teams) ═══")

        r = _wr(ded)
        print(f"  Baseline: WR={r[0]:.1f}% odds={r[2]:.3f} edge={r[4]:+.1f}pp")

        # Best feature
        best_feat = None
        best_lift = 0

        # Check negbin_wp
        has_f = [d for d in ded if d["negbin_wp"] is not None]
        if len(has_f) >= 30:
            med = statistics.median([d["negbin_wp"] for d in has_f])
            hi = [d for d in has_f if d["negbin_wp"] > med]
            lo = [d for d in has_f if d["negbin_wp"] <= med]
            hi_r, lo_r = _wr(hi), _wr(lo)
            if hi_r and lo_r:
                lift = hi_r[0] - lo_r[0]
                if abs(lift) > abs(best_lift):
                    best_feat = f"negbin_wp (>{med:.2f})"
                    best_lift = lift

        # Check ats_dist_wp
        has_f = [d for d in ded if d["ats_dist_wp"] is not None]
        if len(has_f) >= 30:
            med = statistics.median([d["ats_dist_wp"] for d in has_f])
            hi = [d for d in has_f if d["ats_dist_wp"] > med]
            lo = [d for d in has_f if d["ats_dist_wp"] <= med]
            hi_r, lo_r = _wr(hi), _wr(lo)
            if hi_r and lo_r:
                lift = hi_r[0] - lo_r[0]
                if abs(lift) > abs(best_lift):
                    best_feat = f"ats_dist_wp (>{med:.2f})"
                    best_lift = lift

        # Check epp
        has_f = [d for d in ded if d["negbin_edge_pp"] is not None]
        if len(has_f) >= 30:
            below12 = [d for d in has_f if d["negbin_edge_pp"] < 12]
            above12 = [d for d in has_f if d["negbin_edge_pp"] >= 12]
            if below12 and above12 and len(above12) >= 10:
                b_r, a_r = _wr(below12), _wr(above12)
                if b_r and a_r:
                    gap = b_r[0] - a_r[0]
                    if abs(gap) > abs(best_lift):
                        best_feat = "epp<12 vs >=12"
                        best_lift = gap

        if best_feat:
            print(f"  Best discriminator: {best_feat} → lift={best_lift:+.1f}pp")

        # Agreement (if applicable)
        has_both = [d for d in ded if d["negbin_wp"] is not None and d["ats_dist_wp"] is not None]
        agree = [d for d in has_both
                 if (d["negbin_wp"] > 0.5 and d["ats_dist_wp"] > 0.5)
                 or (d["negbin_wp"] <= 0.5 and d["ats_dist_wp"] <= 0.5)]
        disagree = [d for d in has_both
                    if (d["negbin_wp"] > 0.5) != (d["ats_dist_wp"] > 0.5)]
        if disagree:
            print(f"  Agreement: {len(agree)} agree, {len(disagree)} disagree "
                  f"{'⚠LOW-N' if len(disagree) < 10 else ''}")
            ag_r = _wr(agree)
            dg_r = _wr(disagree)
            if ag_r and dg_r:
                print(f"    AGREE:    WR={ag_r[0]:.1f}% N={ag_r[1]} odds={ag_r[2]:.3f}")
                print(f"    DISAGREE: WR={dg_r[0]:.1f}% N={dg_r[1]} odds={dg_r[2]:.3f}")
        else:
            print(f"  Agreement: no disagree cases at this rung")

    # ===================================================================
    # SECTION 6: FORWARD ATS AT ±1.5
    # ===================================================================
    print("\n" + "=" * 78)
    print("SECTION 6: FORWARD ATS (±1.5 ONLY) — FEATURE SCAN")
    print("=" * 78)

    fwd_ded = _dedup_within_rung(fwd_legs)
    print(f"\n  Forward ATS: {len(fwd_legs)} legs → {len(fwd_ded)} game-teams")
    _wr_line("Baseline", fwd_ded)

    # win_prob discrimination
    has_wp = [d for d in fwd_ded if d["wp"] is not None and d["wp"] > 1]
    if has_wp:
        print(f"\n  win_prob discrimination (N={len(has_wp)}):")
        for lo, hi, label in [(0, 35, "<35%"), (35, 45, "35-45%"),
                               (45, 55, "45-55%"), (55, 65, "55-65%"),
                               (65, 75, "65-75%"), (75, 101, "75%+")]:
            sub = [d for d in has_wp if lo <= d["wp"] < hi]
            _wr_line(f"wp {label}", sub, indent=2)

    # devig_entry_prob as signal
    has_devig = [d for d in fwd_ded if d["devig_entry"] is not None and d["devig_entry"] > 0]
    if has_devig:
        print(f"\n  devig_entry_prob discrimination (N={len(has_devig)}):")
        med = statistics.median([d["devig_entry"] for d in has_devig])
        hi = [d for d in has_devig if d["devig_entry"] > med]
        lo = [d for d in has_devig if d["devig_entry"] <= med]
        _wr_line(f"Below median ({med:.3f})", lo, indent=2)
        _wr_line(f"Above median", hi, indent=2)

    con.close()
    print("\n" + "=" * 78)
    print("END OF PER-RUNG SYNTHESIS")
    print("=" * 78)


if __name__ == "__main__":
    main()
