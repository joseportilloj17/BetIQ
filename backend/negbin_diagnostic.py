"""
negbin_diagnostic.py — Deep audit of NegBin backtest results.

Answers three questions:
1. Why is per-rung WR 25-30pp above live realized?
2. What's the parlay/leg structure behind the +63.6% ROI?
3. Where does the 45% home → 23% home drop happen in the selection pipeline?
"""
from __future__ import annotations
import os, sys, math, warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

STAKE = 10.0
START = "2026-05-26"
END   = "2026-06-26"
MAX_PARLAYS_PER_DAY = 8
PARLAY_SIZES = [3, 2]


def _load_ml_favorites(start, end):
    import sqlite3
    db_path = os.path.join(_HERE, "..", "data", "bets.db")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date_ct, bookmakers
        FROM fixtures
        WHERE sport_key = 'baseball_mlb' AND bookmakers IS NOT NULL
          AND game_date_ct BETWEEN ? AND ?
    """, (start, end)).fetchall()
    conn.close()
    import json
    favs = {}
    for home, away, gd, bk_json in rows:
        try:
            bks = json.loads(bk_json)
        except Exception:
            continue
        for bk in bks:
            for mkt in bk.get("markets", []):
                if mkt.get("key") == "h2h":
                    outcomes = {o["name"]: o.get("price", 0) for o in mkt.get("outcomes", [])}
                    hp = outcomes.get(home, 0)
                    ap = outcomes.get(away, 0)
                    if hp and ap:
                        favs[(home, away, gd)] = hp < ap  # True = home is fav
                    break
            if (home, away, gd) in favs:
                break
    return favs


def _load_odds_raw(start, end):
    import sqlite3
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date, team, line, odds
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_spreads'
          AND game_date BETWEEN ? AND ?
          AND odds IS NOT NULL AND odds > 1.0
    """, (start, end)).fetchall()
    conn.close()
    odds = {}
    for home, away, gd, team, line, o in rows:
        odds[(home, away, gd, team, float(line))] = float(o)
    return odds


def main():
    from negbin_backtest import generate_negbin_candidates, assemble_parlays
    from walk_forward_backtest import _load_alt_line_odds, _load_scores

    print("Loading data...")
    ml_favs = _load_ml_favorites(START, END)
    odds_raw = _load_odds_raw(START, END)
    scores = _load_scores(START, END)
    spread_wp = _load_alt_line_odds(START, END)

    print("Generating NegBin candidates...")
    nb_daily, coherence_log = generate_negbin_candidates(START, END, ml_favs, odds_raw)

    # Flatten all candidates
    all_cands = []
    for gd, cands in nb_daily.items():
        all_cands.extend(cands)

    print(f"\nTotal NegBin candidates: {len(all_cands)}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: WR RECONCILIATION — WHY IS NegBin WR SO HIGH?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  SECTION 1: WR RECONCILIATION (candidates vs selected vs live)")
    print("=" * 100)

    # 1a. Candidate-level stats per rung
    print("\n  A. CANDIDATE-LEVEL per-rung stats:")
    print(f"    {'Rung':>6} {'N':>6} {'Home%':>7} {'Fav%':>6} {'WR':>7} "
          f"{'MeanWP':>8} {'MedWP':>8} {'MeanOdds':>10} {'MeanEdge':>10}")
    print(f"    {'─' * 80}")

    for sv in [1.5, 2.5, 3.5, 4.5]:
        rung_cands = [c for c in all_cands if c["spread_value"] == sv]
        if not rung_cands:
            continue
        n = len(rung_cands)
        home_pct = sum(1 for c in rung_cands if c["pick_side"] == "home") / n * 100
        fav_pct = sum(1 for c in rung_cands if c.get("is_pick_fav")) / n * 100
        settled = [c for c in rung_cands if c["actual_result"] in ("WIN", "LOSS")]
        wr = sum(1 for c in settled if c["actual_result"] == "WIN") / max(len(settled), 1) * 100
        wps = [c["model_wp"] for c in rung_cands]
        odds = [c["decimal_odds"] for c in rung_cands if c.get("decimal_odds") and c["decimal_odds"] > 1]
        edges = [c["edge_pp"] for c in rung_cands]
        print(f"    {sv:>+5.1f} {n:>6} {home_pct:>6.1f}% {fav_pct:>5.1f}% {wr:>6.1f}% "
              f"{np.mean(wps):>8.3f} {np.median(wps):>8.3f} {np.mean(odds):>10.3f} {np.mean(edges):>9.1f}pp")

    # 1b. WP distribution of candidates
    print("\n  B. WP DISTRIBUTION of all candidates:")
    wps = [c["model_wp"] for c in all_cands]
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        count = sum(1 for w in wps if w >= threshold)
        print(f"    WP ≥ {threshold:.2f}: {count:>6} ({count/len(wps)*100:>5.1f}%)")

    # 1c. Compare home vs away WP distributions
    print("\n  C. HOME vs AWAY candidate WP distributions:")
    home_cands = [c for c in all_cands if c["pick_side"] == "home"]
    away_cands = [c for c in all_cands if c["pick_side"] == "away"]
    home_wps = [c["model_wp"] for c in home_cands]
    away_wps = [c["model_wp"] for c in away_cands]
    home_edges = [c["edge_pp"] for c in home_cands]
    away_edges = [c["edge_pp"] for c in away_cands]
    home_odds = [c["decimal_odds"] for c in home_cands if c.get("decimal_odds") and c["decimal_odds"] > 1]
    away_odds = [c["decimal_odds"] for c in away_cands if c.get("decimal_odds") and c["decimal_odds"] > 1]

    print(f"    {'':>10} {'N':>6} {'MeanWP':>8} {'MeanEdge':>10} {'MeanOdds':>10} {'WR':>7}")
    print(f"    {'─' * 55}")
    for label, cands, wps_l, edges_l, odds_l in [
        ("Home", home_cands, home_wps, home_edges, home_odds),
        ("Away", away_cands, away_wps, away_edges, away_odds),
    ]:
        settled = [c for c in cands if c["actual_result"] in ("WIN", "LOSS")]
        wr = sum(1 for c in settled if c["actual_result"] == "WIN") / max(len(settled), 1) * 100
        print(f"    {label:>10} {len(cands):>6} {np.mean(wps_l):>8.3f} "
              f"{np.mean(edges_l):>9.1f}pp {np.mean(odds_l):>10.3f} {wr:>6.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: PARLAY/LEG STRUCTURE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  SECTION 2: PARLAY/LEG STRUCTURE")
    print("=" * 100)

    # Build parlays
    nb_parlays = []
    for gd in sorted(nb_daily.keys()):
        from negbin_backtest import assemble_parlays
        day_parlays = assemble_parlays(nb_daily[gd], MAX_PARLAYS_PER_DAY)
        for p in day_parlays:
            p["game_date"] = gd
        nb_parlays.extend(day_parlays)

    print(f"\n  Total parlays: {len(nb_parlays)}")

    # 2a. Parlay size distribution
    size_counts = Counter(len(p["legs"]) for p in nb_parlays)
    print(f"  Parlay sizes: {dict(sorted(size_counts.items()))}")

    # 2b. Odds distribution
    parlay_odds = [p["combined_odds"] for p in nb_parlays]
    print(f"\n  A. PARLAY ODDS DISTRIBUTION:")
    print(f"    Mean: {np.mean(parlay_odds):.3f}")
    print(f"    Median: {np.median(parlay_odds):.3f}")
    print(f"    Std: {np.std(parlay_odds):.3f}")
    print(f"    Min: {min(parlay_odds):.3f}")
    print(f"    Max: {max(parlay_odds):.3f}")
    for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 100)]:
        cnt = sum(1 for o in parlay_odds if lo <= o < hi)
        print(f"    [{lo:.1f}, {hi:.1f}): {cnt} ({cnt/len(parlay_odds)*100:.1f}%)")

    # 2c. Leg-level odds distribution
    all_legs = [l for p in nb_parlays for l in p["legs"]]
    leg_odds = [l["decimal_odds"] for l in all_legs if l.get("decimal_odds") and l["decimal_odds"] > 1]
    print(f"\n  B. LEG-LEVEL ODDS DISTRIBUTION ({len(leg_odds)} legs):")
    print(f"    Mean: {np.mean(leg_odds):.3f}")
    print(f"    Median: {np.median(leg_odds):.3f}")
    for lo, hi in [(1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 100)]:
        cnt = sum(1 for o in leg_odds if lo <= o < hi)
        print(f"    [{lo:.1f}, {hi:.1f}): {cnt} ({cnt/len(leg_odds)*100:.1f}%)")

    # 2d. Per-rung WR in selected legs
    print(f"\n  C. PER-RUNG WR (selected legs in parlays):")
    print(f"    {'Rung':>6} {'N':>6} {'WR':>7} {'MeanWP':>8} {'MeanOdds':>10}")
    print(f"    {'─' * 45}")
    for sv in [1.5, 2.5, 3.5, 4.5]:
        rung_legs = [l for l in all_legs if l["spread_value"] == sv]
        if not rung_legs:
            continue
        settled = [l for l in rung_legs if l["actual_result"] in ("WIN", "LOSS")]
        wr = sum(1 for l in settled if l["actual_result"] == "WIN") / max(len(settled), 1) * 100
        wps = [l["model_wp"] for l in rung_legs]
        odds = [l["decimal_odds"] for l in rung_legs if l.get("decimal_odds") and l["decimal_odds"] > 1]
        print(f"    {sv:>+5.1f} {len(rung_legs):>6} {wr:>6.1f}% {np.mean(wps):>8.3f} {np.mean(odds):>10.3f}")

    # 2e. PnL concentration
    print(f"\n  D. PnL CONCENTRATION:")
    pnls = []
    for p in nb_parlays:
        won = all(l["actual_result"] == "WIN" for l in p["legs"]
                  if l["actual_result"] in ("WIN", "LOSS"))
        pnl = STAKE * (p["combined_odds"] - 1) if won else -STAKE
        pnls.append(pnl)

    pnls_sorted = sorted(pnls, reverse=True)
    total_pnl = sum(pnls)
    top5_pnl = sum(pnls_sorted[:5])
    top10_pnl = sum(pnls_sorted[:10])
    bot5_pnl = sum(pnls_sorted[-5:])
    print(f"    Total PnL: ${total_pnl:+.0f}")
    print(f"    Top 5 parlays PnL: ${top5_pnl:+.0f} ({top5_pnl/total_pnl*100:.0f}% of total)")
    print(f"    Top 10 parlays PnL: ${top10_pnl:+.0f} ({top10_pnl/total_pnl*100:.0f}% of total)")
    print(f"    Bottom 5 parlays PnL: ${bot5_pnl:+.0f}")
    print(f"    Winning parlays: {sum(1 for p in pnls if p > 0)}/{len(pnls)} ({sum(1 for p in pnls if p > 0)/len(pnls)*100:.1f}%)")
    print(f"    Losing parlays: {sum(1 for p in pnls if p < 0)}/{len(pnls)} ({sum(1 for p in pnls if p < 0)/len(pnls)*100:.1f}%)")

    # 2f. Daily leg volume
    print(f"\n  E. DAILY LEG VOLUME:")
    daily_leg_counts = defaultdict(int)
    daily_unique_games = defaultdict(set)
    for gd, cands in nb_daily.items():
        daily_leg_counts[gd] = len(cands)
        for c in cands:
            daily_unique_games[gd].add(c["fixture_key"])

    cand_counts = list(daily_leg_counts.values())
    game_counts = [len(v) for v in daily_unique_games.values()]
    print(f"    Candidates/day: mean={np.mean(cand_counts):.1f}, "
          f"min={min(cand_counts)}, max={max(cand_counts)}")
    print(f"    Unique games/day: mean={np.mean(game_counts):.1f}, "
          f"min={min(game_counts)}, max={max(game_counts)}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: AWAY SKEW — CANDIDATE vs SELECTED MATRIX
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  SECTION 3: AWAY SKEW — CANDIDATE vs SELECTED MATRIX")
    print("=" * 100)

    # Build selected leg set
    selected_sigs = set()
    for p in nb_parlays:
        for l in p["legs"]:
            selected_sigs.add((l["fixture_key"], l["spread_value"], l["pick_side"]))

    def _matrix(cands, label):
        """Print home/away × fav/dog matrix with WR, edge, WP, odds."""
        cells = {
            "home_fav": [], "home_dog": [],
            "away_fav": [], "away_dog": [],
        }
        for c in cands:
            side = c["pick_side"]
            is_fav = c.get("is_pick_fav")
            if is_fav is None:
                continue
            key = f"{side}_{'fav' if is_fav else 'dog'}"
            cells[key].append(c)

        print(f"\n  {label}:")
        print(f"    {'Cell':>12} {'N':>6} {'%':>7} {'WR':>7} {'MeanWP':>8} "
              f"{'MeanEdge':>10} {'MeanOdds':>10} {'MeanLOS':>8}")
        print(f"    {'─' * 75}")

        total = sum(len(v) for v in cells.values())
        for key in ["home_fav", "home_dog", "away_fav", "away_dog"]:
            cands_cell = cells[key]
            if not cands_cell:
                print(f"    {key:>12}      0")
                continue
            n = len(cands_cell)
            pct = n / max(total, 1) * 100
            settled = [c for c in cands_cell if c["actual_result"] in ("WIN", "LOSS")]
            wr = sum(1 for c in settled if c["actual_result"] == "WIN") / max(len(settled), 1) * 100
            wps = [c["model_wp"] for c in cands_cell]
            edges = [c["edge_pp"] for c in cands_cell]
            odds = [c["decimal_odds"] for c in cands_cell
                    if c.get("decimal_odds") and c["decimal_odds"] > 1]
            los_vals = [c["los"] for c in cands_cell if "los" in c]
            print(f"    {key:>12} {n:>6} {pct:>6.1f}% {wr:>6.1f}% {np.mean(wps):>8.3f} "
                  f"{np.mean(edges):>9.1f}pp {np.mean(odds) if odds else 0:>10.3f} "
                  f"{np.mean(los_vals) if los_vals else 0:>8.4f}")

    _matrix(all_cands, "A. ALL CANDIDATES (pre-selection)")

    # Mark which candidates were selected
    selected_cands = [c for c in all_cands
                      if (c["fixture_key"], c["spread_value"], c["pick_side"]) in selected_sigs]
    _matrix(selected_cands, "B. SELECTED CANDIDATES (in parlays)")

    # 3c. Per-rung home/away in candidates vs selected
    print(f"\n  C. PER-RUNG HOME/AWAY COMPOSITION:")
    print(f"    {'Rung':>6} │ {'Cand Home%':>10} {'Cand Away%':>10} │ "
          f"{'Sel Home%':>10} {'Sel Away%':>10} │ {'Shift':>7}")
    print(f"    {'─' * 70}")
    for sv in [1.5, 2.5, 3.5, 4.5]:
        rc = [c for c in all_cands if c["spread_value"] == sv]
        rs = [c for c in selected_cands if c["spread_value"] == sv]
        if not rc:
            continue
        ch = sum(1 for c in rc if c["pick_side"] == "home") / len(rc) * 100
        ca = 100 - ch
        if rs:
            sh = sum(1 for c in rs if c["pick_side"] == "home") / len(rs) * 100
            sa = 100 - sh
            shift = sh - ch
        else:
            sh, sa, shift = 0, 0, 0
        print(f"    {sv:>+5.1f} │ {ch:>9.1f}% {ca:>9.1f}% │ {sh:>9.1f}% {sa:>9.1f}% │ {shift:>+6.1f}pp")

    # 3d. Composite score comparison: home vs away
    print(f"\n  D. COMPOSITE SCORE — HOME vs AWAY (candidates):")
    # Reconstruct composite scores
    for c in all_cands:
        wp = c["model_wp"]
        lqs_norm = c.get("lqs", 50) / 100.0
        los = c.get("los", 0.5)
        odds = c.get("decimal_odds", 1.5)
        combined_wp = wp  # single-leg
        norm_ev = max(0, combined_wp * (odds - 1) - (1 - combined_wp))
        c["_composite"] = wp * 0.40 + lqs_norm * 0.20 + los * 0.30 + min(norm_ev, 1.0) * 0.10

    home_comp = [c["_composite"] for c in all_cands if c["pick_side"] == "home"]
    away_comp = [c["_composite"] for c in all_cands if c["pick_side"] == "away"]
    print(f"    Home composite: mean={np.mean(home_comp):.4f}, median={np.median(home_comp):.4f}")
    print(f"    Away composite: mean={np.mean(away_comp):.4f}, median={np.median(away_comp):.4f}")

    # 3e. LOS comparison
    home_los = [c["los"] for c in all_cands if c["pick_side"] == "home"]
    away_los = [c["los"] for c in all_cands if c["pick_side"] == "away"]
    print(f"\n  E. LOS — HOME vs AWAY:")
    print(f"    Home LOS: mean={np.mean(home_los):.4f}, median={np.median(home_los):.4f}")
    print(f"    Away LOS: mean={np.mean(away_los):.4f}, median={np.median(away_los):.4f}")

    # 3f. Edge comparison
    print(f"\n  F. EDGE (pp) — HOME vs AWAY:")
    print(f"    Home edge: mean={np.mean(home_edges):.2f}pp, median={np.median(home_edges):.2f}pp")
    print(f"    Away edge: mean={np.mean(away_edges):.2f}pp, median={np.median(away_edges):.2f}pp")

    # 3g. Per-rung edge, home vs away
    print(f"\n  G. PER-RUNG EDGE — HOME vs AWAY:")
    print(f"    {'Rung':>6} │ {'Home N':>7} {'Home Edge':>10} {'Home WP':>8} │ "
          f"{'Away N':>7} {'Away Edge':>10} {'Away WP':>8}")
    print(f"    {'─' * 70}")
    for sv in [1.5, 2.5, 3.5, 4.5]:
        rh = [c for c in all_cands if c["spread_value"] == sv and c["pick_side"] == "home"]
        ra = [c for c in all_cands if c["spread_value"] == sv and c["pick_side"] == "away"]
        if not rh and not ra:
            continue
        h_n = len(rh)
        a_n = len(ra)
        h_edge = np.mean([c["edge_pp"] for c in rh]) if rh else 0
        a_edge = np.mean([c["edge_pp"] for c in ra]) if ra else 0
        h_wp = np.mean([c["model_wp"] for c in rh]) if rh else 0
        a_wp = np.mean([c["model_wp"] for c in ra]) if ra else 0
        print(f"    {sv:>+5.1f} │ {h_n:>7} {h_edge:>9.2f}pp {h_wp:>8.3f} │ "
              f"{a_n:>7} {a_edge:>9.2f}pp {a_wp:>8.3f}")

    # 3h. Check market WP for home vs away legs
    print(f"\n  H. MARKET IMPLIED WP — HOME vs AWAY:")
    home_mkt = [c["market_wp"] for c in all_cands if c["pick_side"] == "home"]
    away_mkt = [c["market_wp"] for c in all_cands if c["pick_side"] == "away"]
    print(f"    Home mkt_wp: mean={np.mean(home_mkt):.4f}, median={np.median(home_mkt):.4f}")
    print(f"    Away mkt_wp: mean={np.mean(away_mkt):.4f}, median={np.median(away_mkt):.4f}")

    # Check for unreasonably low mkt_away values (fallback bug indicator)
    low_mkt = [c for c in away_cands if c["market_wp"] < 0.35]
    print(f"    Away legs with mkt_wp < 0.35: {len(low_mkt)} ({len(low_mkt)/max(len(away_cands),1)*100:.1f}%)")

    print("\n" + "=" * 100)
    print("  END DIAGNOSTIC")
    print("=" * 100)


if __name__ == "__main__":
    main()
