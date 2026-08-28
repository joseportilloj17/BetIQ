#!/usr/bin/env python3
"""
Sharp Proxy Validation — THE GATE TEST
========================================
Gates BOTH the speed thesis and the +EV thesis.

Question: Is our "sharp" benchmark (Pinnacle, consensus, or any
fast-mover) ACTUALLY sharper than the soft books it supposedly leads?

Three tests:
1. OUTCOME PREDICTION (Brier): who predicts game outcomes best?
2. CLOSE PREDICTION (MAE vs close): who at time T is closest to
   the eventual closing line?
3. CLOSE LEADERSHIP: when a book moves first, do others follow?
   (Tests whether the "fast mover" is informative or just noisy.)

If Pinnacle/proxy is NOT sharper → the +5.9% CLV and +EV approach
are both measuring soft-vs-soft noise, not genuine stale lines.

Data:
- alt_lines: Pinnacle + DK/FD/BetMGM, -1.5/+1.5 spreads
- games: outcomes
- Multiple snapshots per event for close prediction
"""

import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime

DB_PATH = "data/historical.db"


def _implied(odds):
    return 1.0 / odds if odds > 1.0 else 1.0


def _devig(odds_a, odds_b):
    ia, ib = _implied(odds_a), _implied(odds_b)
    t = ia + ib
    return (ia / t, ib / t) if t > 0 else (0.5, 0.5)


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print("=" * 70)
    print("SHARP PROXY VALIDATION — THE GATE TEST")
    print("Is Pinnacle / any proxy ACTUALLY sharper than the soft books?")
    print("This gates BOTH the speed thesis and the +EV thesis.")
    print("=" * 70)

    # ── Confirm Pinnacle exists ──
    print("\n── Data inventory ──")
    for r in db.execute("""
        SELECT bookmaker, COUNT(*), COUNT(DISTINCT event_id),
               MIN(fetched_at), MAX(fetched_at)
        FROM alt_lines
        WHERE sport_key='baseball_mlb' AND market_key='alternate_spreads'
          AND line IN (-1.5, 1.5)
        GROUP BY bookmaker ORDER BY bookmaker
    """).fetchall():
        print(f"  {r[0]:15s}: {r[1]:>6d} rows, {r[2]:>4d} events, "
              f"{r[3][:10]}–{r[4][:10]}")

    # ── Load all data ──
    rows = db.execute("""
        SELECT fetched_at, event_id, bookmaker, team, odds, line,
               home_team, away_team, game_date_ct
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_spreads'
          AND line IN (-1.5, 1.5)
          AND bookmaker IN ('pinnacle', 'draftkings', 'fanduel', 'betmgm')
        ORDER BY event_id, fetched_at
    """).fetchall()

    # Organize: {event_id: {fetched_at: {bookmaker: {(team, line): odds}}}}
    events = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    event_meta = {}
    ALL_BOOKS = set()

    for r in rows:
        events[r["event_id"]][r["fetched_at"]][r["bookmaker"]][(r["team"], r["line"])] = r["odds"]
        event_meta[r["event_id"]] = (r["home_team"], r["away_team"], r["game_date_ct"])
        ALL_BOOKS.add(r["bookmaker"])

    # Load outcomes
    outcomes = {}
    for r in db.execute("""
        SELECT home_team, away_team, game_date, home_score, away_score
        FROM games WHERE sport='MLB' AND home_score IS NOT NULL
    """).fetchall():
        outcomes[(r["home_team"], r["away_team"], r["game_date"])] = (
            r["home_score"], r["away_score"]
        )

    print(f"\n  Events: {len(events)}")
    print(f"  Books: {sorted(ALL_BOOKS)}")

    # ── Build per-book fair probabilities at each snapshot ──
    # For each (event, time, book), compute fair prob of home covering -1.5
    # using de-vig pair: home -1.5 vs away +1.5

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    snapshots = []  # list of {event_id, time, book, fair_home_neg15, ...}

    for event_id, times_data in events.items():
        home, away, gdate = event_meta[event_id]
        outcome = outcomes.get((home, away, gdate))
        if outcome:
            home_covers = outcome[0] - outcome[1] >= 2
        else:
            home_covers = None

        for fetched_at, books_data in times_data.items():
            for book, prices in books_data.items():
                home_fav = prices.get((home, -1.5))
                away_dog = prices.get((away, 1.5))
                if not home_fav or not away_dog or home_fav <= 1 or away_dog <= 1:
                    continue

                fair_home, fair_away = _devig(home_fav, away_dog)
                overround = _implied(home_fav) + _implied(away_dog)

                snapshots.append({
                    "event_id": event_id,
                    "time": fetched_at,
                    "book": book,
                    "fair": fair_home,  # fair prob of home covering -1.5
                    "odds_fav": home_fav,
                    "overround": overround,
                    "home_covers": home_covers,
                    "home": home, "away": away, "gdate": gdate,
                })

    print(f"  Snapshots: {len(snapshots)}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: OUTCOME PREDICTION (Brier score)
    # Who predicts home covering -1.5 best?
    # Use ALL snapshots (each snapshot is a prediction of the outcome)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: OUTCOME PREDICTION (Brier Score)")
    print("Lower = better. Who predicts home covers -1.5 best?")
    print("=" * 70)

    # Use only snapshots with known outcomes
    with_outcome = [s for s in snapshots if s["home_covers"] is not None]
    print(f"\n  Snapshots with outcomes: {len(with_outcome)}")

    by_book = defaultdict(list)
    for s in with_outcome:
        by_book[s["book"]].append(s)

    print(f"\n  {'Book':>15s} {'Brier':>10s} {'N':>7s} {'Avg Fair':>9s} {'Vig':>6s}")
    print("  " + "-" * 52)
    brier_scores = {}
    for book in sorted(ALL_BOOKS):
        data = by_book[book]
        if not data:
            continue
        brier = statistics.mean([(s["fair"] - (1 if s["home_covers"] else 0)) ** 2
                                  for s in data])
        avg_fair = statistics.mean([s["fair"] for s in data])
        avg_vig = (statistics.mean([s["overround"] for s in data]) - 1) * 100
        brier_scores[book] = brier
        print(f"  {book:>15s} {brier:>10.6f} {len(data):>7d} {avg_fair:>9.4f} {avg_vig:>5.1f}%")

    # Also compute consensus Brier
    # Consensus = average fair prob across available soft books at each (event, time)
    snap_by_et = defaultdict(dict)
    for s in with_outcome:
        snap_by_et[(s["event_id"], s["time"])][s["book"]] = s

    consensus_briers = []
    for (eid, t), books in snap_by_et.items():
        soft_fairs = [books[b]["fair"] for b in ["draftkings", "fanduel", "betmgm"]
                      if b in books]
        if not soft_fairs:
            continue
        consensus_fair = statistics.mean(soft_fairs)
        actual = 1 if list(books.values())[0]["home_covers"] else 0
        consensus_briers.append((consensus_fair - actual) ** 2)

    if consensus_briers:
        cons_brier = statistics.mean(consensus_briers)
        brier_scores["consensus"] = cons_brier
        print(f"  {'consensus':>15s} {cons_brier:>10.6f} {len(consensus_briers):>7d}")

    # Rank
    ranked = sorted(brier_scores.items(), key=lambda x: x[1])
    print(f"\n  Ranking (best to worst):")
    for i, (book, brier) in enumerate(ranked, 1):
        marker = " ← SHARPEST" if i == 1 else ""
        print(f"    {i}. {book:15s} {brier:.6f}{marker}")

    best_book = ranked[0][0]
    is_pin_best = best_book == "pinnacle"
    print(f"\n  Is Pinnacle the sharpest? {'YES' if is_pin_best else 'NO — ' + best_book + ' is sharper'}")

    # Statistical significance: is the gap meaningful?
    if len(ranked) >= 2:
        gap = ranked[-1][1] - ranked[0][1]
        print(f"  Best-to-worst gap: {gap:.6f} ({gap/ranked[0][1]*100:.2f}% relative)")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: CLOSE PREDICTION
    # At time T, whose price is closest to the eventual close?
    # "Close" = last snapshot for that event.
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: CLOSE PREDICTION")
    print("At an earlier snapshot, whose fair prob is closest to the")
    print("eventual closing fair prob? (Close = last snapshot.)")
    print("=" * 70)

    # For each event, define "close" = consensus fair at last snapshot
    # Then for each earlier snapshot, measure each book's MAE vs close
    event_timelines = defaultdict(lambda: defaultdict(dict))
    for s in snapshots:
        event_timelines[s["event_id"]][s["time"]][s["book"]] = s["fair"]

    close_errors = defaultdict(list)  # book → list of |fair_t - close|

    for event_id, timeline in event_timelines.items():
        times = sorted(timeline.keys())
        if len(times) < 2:
            continue

        close_time = times[-1]
        close_books = timeline[close_time]

        # Close = Pinnacle fair at close (if available), else consensus
        if "pinnacle" in close_books:
            close_fair = close_books["pinnacle"]
            close_source = "pinnacle"
        else:
            soft_vals = [close_books[b] for b in ["draftkings", "fanduel", "betmgm"]
                         if b in close_books]
            if not soft_vals:
                continue
            close_fair = statistics.mean(soft_vals)
            close_source = "consensus"

        # Also compute consensus close for comparison
        soft_close = [close_books[b] for b in ["draftkings", "fanduel", "betmgm"]
                      if b in close_books]
        consensus_close = statistics.mean(soft_close) if soft_close else close_fair

        # For each earlier snapshot, measure error vs BOTH close definitions
        for t in times[:-1]:
            for book, fair_t in timeline[t].items():
                # Error vs Pinnacle close
                close_errors[(book, "vs_pin_close")].append(
                    abs(fair_t - close_fair) * 100)
                # Error vs consensus close
                close_errors[(book, "vs_cons_close")].append(
                    abs(fair_t - consensus_close) * 100)

            # Also consensus at time T
            soft_t = [timeline[t][b] for b in ["draftkings", "fanduel", "betmgm"]
                      if b in timeline[t]]
            if soft_t:
                cons_t = statistics.mean(soft_t)
                close_errors[("consensus", "vs_pin_close")].append(
                    abs(cons_t - close_fair) * 100)
                close_errors[("consensus", "vs_cons_close")].append(
                    abs(cons_t - consensus_close) * 100)

    # Report: MAE vs Pinnacle close
    print(f"\n  A. MAE vs Pinnacle close (pp):")
    print(f"  {'Book':>15s} {'MAE':>8s} {'N':>7s}")
    print("  " + "-" * 35)
    mae_pin = {}
    for book in sorted(ALL_BOOKS) + ["consensus"]:
        key = (book, "vs_pin_close")
        if key not in close_errors or not close_errors[key]:
            continue
        mae = statistics.mean(close_errors[key])
        mae_pin[book] = mae
        print(f"  {book:>15s} {mae:>7.2f}pp {len(close_errors[key]):>7d}")

    if mae_pin:
        ranked_mae = sorted(mae_pin.items(), key=lambda x: x[1])
        print(f"\n  Best predictor of Pinnacle close: {ranked_mae[0][0]} ({ranked_mae[0][1]:.2f}pp)")

    # Report: MAE vs consensus close
    print(f"\n  B. MAE vs Consensus close (pp):")
    print(f"  {'Book':>15s} {'MAE':>8s} {'N':>7s}")
    print("  " + "-" * 35)
    mae_cons = {}
    for book in sorted(ALL_BOOKS) + ["consensus"]:
        key = (book, "vs_cons_close")
        if key not in close_errors or not close_errors[key]:
            continue
        mae = statistics.mean(close_errors[key])
        mae_cons[book] = mae
        print(f"  {book:>15s} {mae:>7.2f}pp {len(close_errors[key]):>7d}")

    if mae_cons:
        ranked_mae_c = sorted(mae_cons.items(), key=lambda x: x[1])
        print(f"\n  Best predictor of consensus close: {ranked_mae_c[0][0]} ({ranked_mae_c[0][1]:.2f}pp)")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: CLOSE LEADERSHIP — WHO MOVES FIRST?
    # When a line moves, which book moves first and do others follow?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: CLOSE LEADERSHIP — WHO MOVES FIRST?")
    print("When lines move between snapshots, which book changes first?")
    print("=" * 70)

    # For each event with 3+ snapshots, check: at time T1,
    # which book's price is closest to the T2 price?
    # The "leader" at T1 already shows what others will show at T2.

    leader_counts = defaultdict(int)
    leader_events = 0

    for event_id, timeline in event_timelines.items():
        times = sorted(timeline.keys())
        if len(times) < 3:
            continue

        for i in range(len(times) - 2):
            t1, t2 = times[i], times[i + 1]
            books_t1 = timeline[t1]
            books_t2 = timeline[t2]

            # What's the "consensus direction" between t1 and t2?
            # Use all-book average at each time
            all_t1 = [books_t1[b] for b in books_t1]
            all_t2 = [books_t2[b] for b in books_t2]
            if not all_t1 or not all_t2:
                continue
            avg_t1 = statistics.mean(all_t1)
            avg_t2 = statistics.mean(all_t2)
            move = avg_t2 - avg_t1

            if abs(move) < 0.005:  # < 0.5pp move, skip
                continue

            # Which book at T1 was already closest to the T2 consensus?
            best_book = None
            best_err = float("inf")
            for book, fair_t1 in books_t1.items():
                err = abs(fair_t1 - avg_t2)
                if err < best_err:
                    best_err = err
                    best_book = book

            if best_book:
                leader_counts[best_book] += 1
                leader_events += 1

    if leader_events > 0:
        print(f"\n  Total movement events (>0.5pp): {leader_events}")
        print(f"\n  {'Book':>15s} {'Leads':>7s} {'%':>7s}")
        print("  " + "-" * 32)
        for book, count in sorted(leader_counts.items(), key=lambda x: -x[1]):
            print(f"  {book:>15s} {count:>7d} {count/leader_events*100:>6.1f}%")

        leader = max(leader_counts, key=leader_counts.get)
        print(f"\n  Most frequent leader: {leader} "
              f"({leader_counts[leader]/leader_events*100:.1f}%)")
        if leader == "pinnacle":
            print("  → Pinnacle leads the market (expected for a sharp book)")
        else:
            print(f"  → {leader} leads, NOT Pinnacle")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: PINNACLE-DERIVED CLV — IS IT REAL?
    # If Pinnacle IS sharp, betting when soft disagrees should be +CLV.
    # If Pinnacle is NOT sharp, the "CLV" is noise.
    # Test: same bets, but measure CLV against OUTCOME, not just close.
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 4: PINNACLE AS BENCHMARK — OUTCOME P&L")
    print("If we bet when soft disagrees with Pinnacle, do we WIN?")
    print("(Not just beat the close — actually win the bet.)")
    print("=" * 70)

    # For each snapshot where Pinnacle and a soft book both exist:
    # If soft_fair < pin_fair (soft underprices → higher odds → bet it),
    # did the bet WIN?
    outcome_bets = []
    for s in with_outcome:
        if s["book"] == "pinnacle":
            continue
        # Find Pinnacle at same (event, time)
        key = (s["event_id"], s["time"])
        if key not in snap_by_et or "pinnacle" not in snap_by_et[key]:
            continue
        pin = snap_by_et[key]["pinnacle"]
        diff = s["fair"] - pin["fair"]

        if abs(diff) < 0.01:  # < 1pp disagreement
            continue

        # Bet the side where soft offers better odds than Pinnacle says
        if diff < -0.01:
            # Soft underprices home → bet home -1.5 at soft odds
            won = s["home_covers"]
            bet_side = "home -1.5"
        else:
            # Soft overprices home → bet away +1.5
            won = not s["home_covers"]
            bet_side = "away +1.5"

        outcome_bets.append({
            "book": s["book"],
            "diff_pp": diff * 100,
            "won": won,
            "entry_odds": s["odds_fav"] if diff < 0 else None,
            "pin_fair": pin["fair"],
            "soft_fair": s["fair"],
            "bet_side": bet_side,
        })

    if outcome_bets:
        print(f"\n  Bets where soft disagrees with Pinnacle >1pp: {len(outcome_bets)}")
        wins = sum(b["won"] for b in outcome_bets)
        wr = wins / len(outcome_bets) * 100

        # Expected WR from Pinnacle fair
        exp_wr = statistics.mean([b["pin_fair"] if b["diff_pp"] < 0
                                   else 1 - b["pin_fair"]
                                   for b in outcome_bets]) * 100

        print(f"  Actual WR:   {wr:.1f}%")
        print(f"  Expected WR (Pinnacle fair): {exp_wr:.1f}%")
        print(f"  Edge vs Pinnacle: {wr - exp_wr:+.1f}pp")
        print(f"  → {'Pinnacle is WRONG (soft books were right)' if wr > exp_wr + 2 else 'Pinnacle is RIGHT (soft books were wrong)' if wr < exp_wr - 2 else 'NEITHER is systematically better (within noise)'}")

        # By book
        print(f"\n  By soft book:")
        for book in ["draftkings", "fanduel", "betmgm"]:
            subset = [b for b in outcome_bets if b["book"] == book]
            if subset:
                sub_wr = sum(b["won"] for b in subset) / len(subset) * 100
                sub_exp = statistics.mean([b["pin_fair"] if b["diff_pp"] < 0
                                            else 1 - b["pin_fair"]
                                            for b in subset]) * 100
                print(f"    {book:15s}: WR={sub_wr:.1f}% vs exp={sub_exp:.1f}% "
                      f"({sub_wr-sub_exp:+.1f}pp), N={len(subset)}")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICT: IS THE SHARP PROXY VALID?")
    print("=" * 70)

    pin_brier = brier_scores.get("pinnacle", 999)
    best_soft_brier = min(brier_scores.get(b, 999) for b in ["draftkings", "fanduel", "betmgm"])
    best_soft_name = min(["draftkings", "fanduel", "betmgm"],
                         key=lambda b: brier_scores.get(b, 999))
    cons_brier_val = brier_scores.get("consensus", 999)

    pin_sharper = pin_brier < best_soft_brier

    print(f"""
  OUTCOME PREDICTION (Brier):
    Pinnacle:       {pin_brier:.6f}
    Best soft book: {best_soft_name} = {best_soft_brier:.6f}
    Consensus:      {cons_brier_val:.6f}
    → Pinnacle {'IS' if pin_sharper else 'is NOT'} sharper than the best soft book

  CLOSE PREDICTION (MAE):
    Best predictor of Pinnacle close: {ranked_mae[0][0] if mae_pin else '?'} ({ranked_mae[0][1]:.2f}pp)
    Best predictor of consensus close: {ranked_mae_c[0][0] if mae_cons else '?'} ({ranked_mae_c[0][1]:.2f}pp)

  MARKET LEADERSHIP:
    Most frequent leader: {leader if leader_events else '?'} ({leader_counts.get(leader,0)/max(leader_events,1)*100:.1f}%)
    → {'Pinnacle leads (consistent with sharpness)' if leader == 'pinnacle' else leader + ' leads, NOT Pinnacle'}

  {'='*60}

  {'GATE PASSES: Pinnacle is a valid sharp proxy.' if pin_sharper and leader == 'pinnacle' else 'GATE FAILS: Pinnacle is NOT demonstrably sharper than soft books.'}

  {'Proceed to validate the +5.9% CLV against Pinnacle benchmark.' if pin_sharper else 'The +5.9% cross-book CLV and the +EV consensus approach both rest on a benchmark that is NOT proven sharper than the soft books. Both directions are UNVALIDATED.'}

  {'The speed thesis dies here: there is no sharp leader to exploit lag from.' if not pin_sharper and leader != 'pinnacle' else ''}

  WHAT THIS MEANS:
  {'- Pinnacle predicts outcomes AND leads line movements → valid benchmark' if pin_sharper and leader == 'pinnacle' else '- Pinnacle does NOT predict outcomes better than ' + best_soft_name + ' on -1.5 spreads'}
  {'- Cross-book disagreements are MUTUAL noise between equally-sharp books' if not pin_sharper else '- Cross-book disagreements reflect Pinnacle leading soft books'}
  {'- Speed thesis: DEAD (no sharp leader to detect and follow)' if not pin_sharper else '- Speed thesis: proceed to validate CLV net of vig'}
  {'- +EV thesis: DEAD (no sharp benchmark to identify soft mispricings)' if not pin_sharper else '- +EV thesis: proceed to validate consensus as benchmark'}
""")

    db.close()


if __name__ == "__main__":
    main()
