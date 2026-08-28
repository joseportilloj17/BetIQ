"""
consensus_benchmark_test.py — Test whether de-vigged consensus (or Pinnacle)
is a valid sharp benchmark for +EV identification.

We have Pinnacle in alt_lines (1,242 events) — the sharpest public book.
We also have 4 soft books in line_snapshots (fanduel, draftkings, betrivers, betmgm).

Tests:
  1. Is Pinnacle sharper than the soft books? (predicts close/outcome better)
  2. Does the de-vigged 4-book consensus match or beat Pinnacle?
  3. +EV backtest: bet soft-book outliers vs Pinnacle/consensus benchmark
  4. Sharp-subset: which soft book is closest to Pinnacle?

Usage:
    python consensus_benchmark_test.py
"""
from __future__ import annotations

import sqlite3
import os
import numpy as np
from collections import defaultdict
from scipy import stats as sp_stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HIST_DB = os.path.join(DATA_DIR, "historical.db")
BETS_DB = os.path.join(DATA_DIR, "bets.db")


def _devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


def _load_spread_data() -> dict:
    """
    Load matched spread data: for each (event, team, spread=±1.5),
    get pinnacle line, soft book lines, and closing line + game outcome.

    Returns dict keyed by (event_id, team) with:
      pinnacle_fair, soft_lines{book: fair}, consensus_fair,
      close_fair, game_result
    """
    con = sqlite3.connect(HIST_DB)

    # ── First, get canonical spread sides from line_snapshots ──
    # Each team has ONE canonical side: favorite=-1.5, underdog=+1.5
    canonical_rows = con.execute("""
        SELECT DISTINCT event_id, outcome_name, point
        FROM line_snapshots
        WHERE sport_key = 'baseball_mlb' AND market_key = 'spreads'
    """).fetchall()

    canonical_side = {}  # (event_id, team) → spread_value (-1.5 or +1.5)
    for r in canonical_rows:
        canonical_side[(r[0], r[1])] = r[2]

    print(f"  Canonical spread sides: {len(canonical_side)} (event, team) pairs")

    # ── Pinnacle lines (from alt_lines) ──
    # Get the LATEST pinnacle snapshot per event, matching CANONICAL side
    pin_all_rows = con.execute("""
        SELECT a1.event_id, a1.team, a1.line, a1.odds as our_odds,
               a2.odds as opp_odds, a1.fetched_at
        FROM alt_lines a1
        INNER JOIN alt_lines a2
          ON a1.event_id = a2.event_id
         AND a1.bookmaker = a2.bookmaker
         AND a1.fetched_at = a2.fetched_at
         AND a1.market_key = a2.market_key
         AND a1.line = -a2.line
         AND a1.team != a2.team
        WHERE a1.sport_key = 'baseball_mlb'
          AND a1.bookmaker = 'pinnacle'
          AND a1.market_key = 'alternate_spreads'
          AND ABS(a1.line) = 1.5
          AND a1.odds > 1.0 AND a2.odds > 1.0
        ORDER BY a1.fetched_at DESC
    """).fetchall()

    pinnacle = {}
    for row in pin_all_rows:
        eid, team, line, our_odds, opp_odds, fetched = row
        key = (eid, team)
        # Only use the row matching this team's CANONICAL spread side
        canon = canonical_side.get(key)
        if canon is None or abs(line - canon) > 0.01:
            continue
        if key in pinnacle:
            continue  # already have latest

        our_impl = 1.0 / our_odds
        opp_impl = 1.0 / opp_odds
        fair, _ = _devig(our_impl, opp_impl)
        pinnacle[key] = {
            "fair": fair,
            "odds": our_odds,
            "line": line,
            "overround": our_impl + opp_impl,
            "fetched": fetched,
        }

    print(f"  Pinnacle lines loaded: {len(pinnacle)} (event, team) pairs")

    # ── Soft book lines (from line_snapshots) ──
    # Get the LATEST snapshot per (event, book) for spreads
    soft_rows = con.execute("""
        SELECT s1.event_id, s1.bookmaker, s1.outcome_name as team,
               s1.point as line, s1.price as our_price,
               s2.price as opp_price, s1.captured_at
        FROM line_snapshots s1
        INNER JOIN line_snapshots s2
          ON s1.event_id = s2.event_id
         AND s1.bookmaker = s2.bookmaker
         AND s1.captured_at = s2.captured_at
         AND s1.market_key = s2.market_key
         AND s1.point = -s2.point
         AND s1.outcome_name != s2.outcome_name
        INNER JOIN (
            SELECT event_id, bookmaker, outcome_name,
                   MAX(captured_at) as max_cap
            FROM line_snapshots
            WHERE sport_key = 'baseball_mlb' AND market_key = 'spreads'
            GROUP BY event_id, bookmaker, outcome_name
        ) latest
          ON s1.event_id = latest.event_id
         AND s1.bookmaker = latest.bookmaker
         AND s1.outcome_name = latest.outcome_name
         AND s1.captured_at = latest.max_cap
        WHERE s1.sport_key = 'baseball_mlb'
          AND s1.market_key = 'spreads'
          AND s1.price > 1.0 AND s2.price > 1.0
    """).fetchall()

    soft = defaultdict(dict)
    for row in soft_rows:
        eid, book, team, line, our_p, opp_p, captured = row
        our_impl = 1.0 / our_p
        opp_impl = 1.0 / opp_p
        fair, _ = _devig(our_impl, opp_impl)
        soft[(eid, team)][book] = {
            "fair": fair,
            "odds": our_p,
            "line": line,
            "overround": our_impl + opp_impl,
            "captured": captured,
        }

    print(f"  Soft book lines loaded: {len(soft)} (event, team) pairs")

    # ── Game outcomes (from games table, matched via date + teams) ──
    outcomes = {}

    # Match events to games via alt_lines metadata (has home/away/game_date_ct)
    outcome_rows = con.execute("""
        SELECT DISTINCT al.event_id, g.home_team, g.away_team,
               g.home_score, g.away_score
        FROM (
            SELECT DISTINCT event_id, home_team, away_team, game_date_ct
            FROM alt_lines WHERE sport_key = 'baseball_mlb'
        ) al
        INNER JOIN games g
          ON al.home_team = g.home_team
         AND al.away_team = g.away_team
         AND al.game_date_ct = g.game_date
        WHERE g.sport = 'MLB'
          AND g.home_score IS NOT NULL
    """).fetchall()

    for row in outcome_rows:
        eid, home, away, h_score, a_score = row
        margin = h_score - a_score
        # For each team at each spread:
        # +1.5 (underdog): covers if their margin > -1.5
        outcomes[(eid, home)] = 1 if margin + 1.5 > 0 else (
            0 if margin + 1.5 < 0 else 0.5)  # home as underdog +1.5
        outcomes[(eid, away)] = 1 if -margin + 1.5 > 0 else (
            0 if -margin + 1.5 < 0 else 0.5)  # away as underdog +1.5
        # -1.5 (favorite): covers if their margin > 1.5
        outcomes[(eid, home + "_fav")] = 1 if margin - 1.5 > 0 else (
            0 if margin - 1.5 < 0 else 0.5)
        outcomes[(eid, away + "_fav")] = 1 if -margin - 1.5 > 0 else (
            0 if -margin - 1.5 < 0 else 0.5)

    con.close()
    print(f"  Game outcomes loaded: {len(outcomes)} (event, team) pairs")

    # ── Merge ──
    # Only merge when Pinnacle and soft book have the SAME spread for the team
    merged = {}
    spread_mismatches = 0
    for key in pinnacle:
        if key not in soft:
            continue
        eid, team = key
        pin = pinnacle[key]
        s = soft[key]

        # Filter soft books to only those with the SAME spread as Pinnacle
        pin_spread = pin["line"]
        matching_soft = {b: info for b, info in s.items()
                         if abs(info["line"] - pin_spread) < 0.01}

        if len(matching_soft) < 2:
            spread_mismatches += 1
            continue

        # Determine if this team is the +1.5 or -1.5 side
        if pin_spread > 0:  # underdog +1.5
            outcome_key = key
        else:  # favorite -1.5
            outcome_key = (eid, team + "_fav")

        outcome = outcomes.get(outcome_key)

        # Consensus: average de-vigged fair prob across MATCHING soft books
        soft_fairs = [matching_soft[b]["fair"] for b in matching_soft]
        consensus_fair = np.mean(soft_fairs)

        merged[key] = {
            "event_id": eid,
            "team": team,
            "spread": pin_spread,
            "pinnacle_fair": pin["fair"],
            "pinnacle_odds": pin["odds"],
            "pinnacle_overround": pin["overround"],
            "soft_books": matching_soft,
            "consensus_fair": consensus_fair,
            "n_soft_books": len(matching_soft),
            "outcome": outcome,
        }

    print(f"  Merged (pin + soft, same spread): {len(merged)} legs")
    print(f"  Spread mismatches skipped: {spread_mismatches}")
    n_with_outcome = sum(1 for v in merged.values() if v["outcome"] is not None)
    print(f"  With known outcome: {n_with_outcome}")

    return merged


def _run_analysis(data: dict):
    """Run all 4 parts of the consensus benchmark test."""

    legs = list(data.values())
    legs_with_outcome = [l for l in legs if l["outcome"] is not None]

    print(f"\n  Total legs: {len(legs)}")
    print(f"  With outcomes: {len(legs_with_outcome)}")

    # ================================================================
    # PART 1: SHARPNESS RANKING — who predicts outcomes best?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: SHARPNESS RANKING")
    print("Who predicts game outcomes best? (Brier score, lower = better)")
    print("=" * 70)

    if not legs_with_outcome:
        print("\n  NO OUTCOMES — skipping Brier analysis")
        print("  (games table may not match event_ids)")
        # Fall through to parts that don't need outcomes
    else:
        # Brier score for each source
        brier = defaultdict(list)
        for l in legs_with_outcome:
            outcome = l["outcome"]
            # Pinnacle
            brier["pinnacle"].append((l["pinnacle_fair"] - outcome) ** 2)
            # Consensus
            brier["consensus"].append((l["consensus_fair"] - outcome) ** 2)
            # Each soft book
            for book, info in l["soft_books"].items():
                brier[book].append((info["fair"] - outcome) ** 2)

        print(f"\n  {'Source':20s}  {'Brier':>8s}  {'N':>6s}  {'Avg Fair':>8s}")
        print("  " + "-" * 50)
        for source in sorted(brier.keys(), key=lambda s: np.mean(brier[s])):
            vals = brier[source]
            fairs = []
            if source == "pinnacle":
                fairs = [l["pinnacle_fair"] for l in legs_with_outcome]
            elif source == "consensus":
                fairs = [l["consensus_fair"] for l in legs_with_outcome]
            else:
                fairs = [l["soft_books"][source]["fair"]
                         for l in legs_with_outcome if source in l["soft_books"]]
            print(f"  {source:20s}  {np.mean(vals):.6f}  {len(vals):6d}  "
                  f"{np.mean(fairs):.4f}")

    # ================================================================
    # PART 2: CONSENSUS vs PINNACLE — agreement analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: CONSENSUS vs PINNACLE AGREEMENT")
    print("How close is 4-book consensus to Pinnacle fair?")
    print("=" * 70)

    diffs = [l["consensus_fair"] - l["pinnacle_fair"] for l in legs]
    diffs = np.array(diffs)

    print(f"\n  Consensus - Pinnacle (pp):")
    print(f"    Mean: {np.mean(diffs)*100:+.2f}pp")
    print(f"    Std:  {np.std(diffs)*100:.2f}pp")
    print(f"    MAE:  {np.mean(np.abs(diffs))*100:.2f}pp")
    print(f"    Correlation: {np.corrcoef([l['pinnacle_fair'] for l in legs], [l['consensus_fair'] for l in legs])[0,1]:.4f}")

    # Per-book deviation from Pinnacle
    print(f"\n  Individual book deviation from Pinnacle:")
    print(f"  {'Book':20s}  {'Mean diff':>10s}  {'MAE':>8s}  {'Corr':>6s}  {'N':>5s}")
    print("  " + "-" * 55)

    all_books = set()
    for l in legs:
        all_books.update(l["soft_books"].keys())

    book_stats = {}
    for book in sorted(all_books):
        book_legs = [l for l in legs if book in l["soft_books"]]
        if len(book_legs) < 100:
            continue
        book_diffs = [l["soft_books"][book]["fair"] - l["pinnacle_fair"]
                      for l in book_legs]
        book_diffs = np.array(book_diffs)
        pin_fairs = [l["pinnacle_fair"] for l in book_legs]
        book_fairs = [l["soft_books"][book]["fair"] for l in book_legs]
        corr = np.corrcoef(pin_fairs, book_fairs)[0, 1]
        mae = np.mean(np.abs(book_diffs))

        print(f"  {book:20s}  {np.mean(book_diffs)*100:+8.2f}pp  "
              f"{mae*100:6.2f}pp  {corr:.4f}  {len(book_legs):5d}")
        book_stats[book] = {"mae": mae, "corr": corr, "n": len(book_legs)}

    # Consensus vs individual books
    consensus_mae = np.mean(np.abs(diffs))
    consensus_corr = np.corrcoef(
        [l["pinnacle_fair"] for l in legs],
        [l["consensus_fair"] for l in legs]
    )[0, 1]

    print(f"\n  {'CONSENSUS':20s}  {np.mean(diffs)*100:+8.2f}pp  "
          f"{consensus_mae*100:6.2f}pp  {consensus_corr:.4f}  {len(legs):5d}")

    beats_all = all(consensus_mae < book_stats[b]["mae"]
                    for b in book_stats)
    print(f"\n  Consensus MAE < all individual books: {'YES' if beats_all else 'NO'}")

    # ================================================================
    # PART 3: +EV BACKTEST — bet soft outliers vs benchmark
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: +EV BACKTEST")
    print("When a soft book is mispriced vs Pinnacle, does betting it")
    print("produce positive CLV?")
    print("=" * 70)

    # For each leg: if a soft book offers BETTER odds than Pinnacle fair implies,
    # that's a +EV opportunity. Measure CLV of those bets.
    ev_bets = []
    for l in legs:
        pin_fair = l["pinnacle_fair"]
        for book, info in l["soft_books"].items():
            soft_implied = 1.0 / info["odds"]
            # +EV if soft book implied < pinnacle fair
            # (we get a price implying lower prob than true prob)
            edge = pin_fair - soft_implied
            if edge > 0.02:  # at least 2pp edge vs Pinnacle
                # CLV: compare soft book odds to Pinnacle odds
                clv_raw = (info["odds"] / l["pinnacle_odds"] - 1) * 100
                ev_bets.append({
                    "event_id": l["event_id"],
                    "team": l["team"],
                    "spread": l["spread"],
                    "book": book,
                    "soft_odds": info["odds"],
                    "soft_implied": soft_implied,
                    "soft_fair": info["fair"],
                    "pin_fair": pin_fair,
                    "pin_odds": l["pinnacle_odds"],
                    "edge_pp": edge * 100,
                    "clv_raw_pct": clv_raw,
                    "outcome": l["outcome"],
                })

    print(f"\n  Legs where soft book offers >2pp edge vs Pinnacle: {len(ev_bets)}")

    if ev_bets:
        n_events = len(set(b["event_id"] for b in ev_bets))
        edges = np.array([b["edge_pp"] for b in ev_bets])
        clvs = np.array([b["clv_raw_pct"] for b in ev_bets])

        print(f"  Unique events: {n_events}")
        print(f"  Edge vs Pinnacle: mean={np.mean(edges):.2f}pp, "
              f"median={np.median(edges):.2f}pp")
        print(f"  CLV (soft vs Pinnacle odds): mean={np.mean(clvs):+.2f}%, "
              f"median={np.median(clvs):+.2f}%")
        print(f"  Beat Pinnacle price: {sum(1 for c in clvs if c > 0)}/{len(clvs)} "
              f"= {sum(1 for c in clvs if c > 0)/len(clvs)*100:.1f}%")

        # CI
        se = np.std(clvs, ddof=1) / np.sqrt(len(clvs))
        t_crit = sp_stats.t.ppf(0.975, df=len(clvs)-1)
        ci_lo = np.mean(clvs) - t_crit * se
        ci_hi = np.mean(clvs) + t_crit * se
        print(f"  CLV 95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")
        print(f"  Includes zero: {'YES' if ci_lo <= 0 <= ci_hi else 'NO'}")

        # By book
        print(f"\n  By book:")
        by_book = defaultdict(list)
        for b in ev_bets:
            by_book[b["book"]].append(b)
        for book in sorted(by_book.keys()):
            bk_bets = by_book[book]
            bk_clvs = np.array([b["clv_raw_pct"] for b in bk_bets])
            bk_edges = np.array([b["edge_pp"] for b in bk_bets])
            print(f"    {book:15s}: N={len(bk_bets)}, edge={np.mean(bk_edges):.1f}pp, "
                  f"CLV={np.mean(bk_clvs):+.2f}%, "
                  f"beat_pin={sum(1 for c in bk_clvs if c > 0)/len(bk_clvs)*100:.0f}%")

        # Win rate on outcomes (if available)
        bets_with_outcome = [b for b in ev_bets if b["outcome"] is not None]
        if bets_with_outcome:
            wins = sum(1 for b in bets_with_outcome if b["outcome"] == 1)
            n_out = len(bets_with_outcome)
            wr = wins / n_out * 100
            # Expected WR from pinnacle fair
            avg_pin_fair = np.mean([b["pin_fair"] for b in bets_with_outcome])
            print(f"\n  Outcome validation (N={n_out}):")
            print(f"    Win rate: {wr:.1f}% (expected from pin fair: {avg_pin_fair*100:.1f}%)")

    # Different edge thresholds
    print(f"\n  Edge threshold sweep (soft vs Pinnacle):")
    print(f"  {'Threshold':>10s}  {'N bets':>7s}  {'Avg edge':>9s}  "
          f"{'CLV%':>8s}  {'Beat pin':>9s}")
    print("  " + "-" * 50)
    for thresh in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        thresh_bets = []
        for l in legs:
            pin_fair = l["pinnacle_fair"]
            for book, info in l["soft_books"].items():
                soft_implied = 1.0 / info["odds"]
                edge = pin_fair - soft_implied
                if edge > thresh:
                    clv_raw = (info["odds"] / l["pinnacle_odds"] - 1) * 100
                    thresh_bets.append({"edge": edge * 100, "clv": clv_raw})

        if thresh_bets:
            edges = np.array([b["edge"] for b in thresh_bets])
            clvs = np.array([b["clv"] for b in thresh_bets])
            beat = sum(1 for c in clvs if c > 0) / len(clvs) * 100
            print(f"  {thresh*100:8.0f}pp  {len(thresh_bets):7d}  "
                  f"{np.mean(edges):+7.2f}pp  {np.mean(clvs):+6.2f}%  {beat:7.1f}%")

    # ================================================================
    # PART 4: SHARP-SUBSET — which book is closest to Pinnacle?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 4: SHARP-SUBSET ANALYSIS")
    print("Which soft book tracks Pinnacle most closely?")
    print("=" * 70)

    if book_stats:
        # Rank by MAE (lower = sharper)
        ranked = sorted(book_stats.items(), key=lambda x: x[1]["mae"])
        print(f"\n  Sharpness ranking (by MAE vs Pinnacle):")
        for i, (book, stats) in enumerate(ranked, 1):
            print(f"    {i}. {book:15s}: MAE={stats['mae']*100:.2f}pp, "
                  f"corr={stats['corr']:.4f}")

        # Sharp-weighted consensus: weight by inverse MAE
        sharpest = ranked[0][0] if ranked else None
        if sharpest:
            print(f"\n  Sharpest soft book: {sharpest}")
            # Compare: consensus, sharpest-only, sharp-weighted
            print(f"\n  Benchmark comparison (MAE vs Pinnacle):")
            print(f"    Flat consensus:      {consensus_mae*100:.2f}pp")

            # Sharp-weighted consensus
            weights = {b: 1.0 / s["mae"] for b, s in book_stats.items()}
            w_total = sum(weights.values())
            weighted_diffs = []
            for l in legs:
                w_fair = 0
                w_sum = 0
                for book, info in l["soft_books"].items():
                    if book in weights:
                        w = weights[book] / w_total
                        w_fair += info["fair"] * w
                        w_sum += w
                if w_sum > 0:
                    w_fair /= w_sum
                    weighted_diffs.append(abs(w_fair - l["pinnacle_fair"]))

            if weighted_diffs:
                print(f"    Sharp-weighted cons:  {np.mean(weighted_diffs)*100:.2f}pp")

            # Sharpest-only
            sharpest_diffs = [abs(l["soft_books"][sharpest]["fair"] - l["pinnacle_fair"])
                              for l in legs if sharpest in l["soft_books"]]
            if sharpest_diffs:
                print(f"    Sharpest-only ({sharpest}): {np.mean(sharpest_diffs)*100:.2f}pp")

    # ================================================================
    # PART 5: OVERROUND COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 5: VIG COMPARISON")
    print("=" * 70)

    pin_or = [l["pinnacle_overround"] for l in legs]
    print(f"\n  Pinnacle overround: {np.mean(pin_or):.4f} (vig={((np.mean(pin_or)-1)*100):.2f}%)")

    for book in sorted(all_books):
        book_legs = [l for l in legs if book in l["soft_books"]]
        if len(book_legs) < 100:
            continue
        ors = [l["soft_books"][book]["overround"] for l in book_legs]
        print(f"  {book:15s} overround: {np.mean(ors):.4f} "
              f"(vig={((np.mean(ors)-1)*100):.2f}%, N={len(book_legs)})")


def main():
    print("=" * 70)
    print("CONSENSUS BENCHMARK TEST")
    print("Is Pinnacle / de-vigged consensus a valid +EV benchmark?")
    print("=" * 70)

    print("\n── Loading data ──")
    data = _load_spread_data()

    if not data:
        print("\nNo merged data — check event matching")
        return

    print("\n── Running analysis ──")
    _run_analysis(data)


if __name__ == "__main__":
    main()
