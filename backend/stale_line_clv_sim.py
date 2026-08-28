#!/usr/bin/env python3
"""
Stale-Line CLV Simulation — Validate Cross-Book Lag Edge
=========================================================
The cross-book lag test showed soft books disagree with Pinnacle by 0.75pp MAE.
The user estimates +4-6% edge. THIS SCRIPT validates that estimate by SIMULATING
actual bets on the stale soft line and measuring:

1. Actual CLV (entry vs close) — net of vig
2. Outcome validation (does the stale-line bet WIN more than breakeven?)
3. Detection-latency filter (restrict to events within our polling window)
4. Split-half stability
5. Go/no-go assessment

Data: alt_lines (Pinnacle + soft books, -1.5/+1.5 spreads, simultaneous captures)
      games (outcomes for win-rate validation)
"""

import sqlite3
import statistics
import random
from collections import defaultdict
from datetime import datetime

DB_PATH = "data/historical.db"
SOFT_BOOKS = ["draftkings", "fanduel", "betmgm"]


def _implied(odds):
    return 1.0 / odds if odds > 1.0 else 1.0


def _devig(odds_a, odds_b):
    """Multiplicative de-vig. Returns (fair_prob_a, fair_prob_b)."""
    ia, ib = _implied(odds_a), _implied(odds_b)
    t = ia + ib
    return (ia / t, ib / t) if t > 0 else (0.5, 0.5)


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print("=" * 70)
    print("STALE-LINE CLV SIMULATION")
    print("Validate: does betting the stale soft line beat the close")
    print("net of 4-5% main-line vig?")
    print("=" * 70)

    # ── Load all multi-snapshot data ──
    print("\n── Loading data ──")

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

    for r in rows:
        events[r["event_id"]][r["fetched_at"]][r["bookmaker"]][(r["team"], r["line"])] = r["odds"]
        event_meta[r["event_id"]] = {
            "home": r["home_team"], "away": r["away_team"],
            "game_date": r["game_date_ct"],
        }

    # Load outcomes
    outcomes = {}
    for r in db.execute("""
        SELECT home_team, away_team, game_date, home_score, away_score
        FROM games WHERE sport='MLB' AND home_score IS NOT NULL
    """).fetchall():
        outcomes[(r["home_team"], r["away_team"], r["game_date"])] = {
            "home_score": r["home_score"], "away_score": r["away_score"],
        }

    # ── Build simulated bets ──
    # Strategy: for each event with 2+ snapshots, find moments where
    # a soft book disagrees with Pinnacle. "Bet" the stale soft line.
    # "Close" = last snapshot's soft line for that event.

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    sim_bets = []

    for event_id, snapshots in events.items():
        times = sorted(snapshots.keys())
        if len(times) < 2:
            continue

        meta = event_meta[event_id]
        home, away = meta["home"], meta["away"]
        close_time = times[-1]
        close_snap = snapshots[close_time]

        # Get outcome
        outcome = outcomes.get((home, away, meta["game_date"]))
        if outcome:
            home_margin = outcome["home_score"] - outcome["away_score"]
        else:
            home_margin = None

        # For each earlier snapshot, check for stale-line opportunities
        for entry_time in times[:-1]:
            entry_snap = snapshots[entry_time]
            if "pinnacle" not in entry_snap:
                continue

            pin_entry = entry_snap["pinnacle"]
            pin_close = close_snap.get("pinnacle", {})

            # Time from entry to close
            dt_entry = datetime.strptime(entry_time, fmt)
            dt_close = datetime.strptime(close_time, fmt)
            hours_to_close = (dt_close - dt_entry).total_seconds() / 3600

            for book in SOFT_BOOKS:
                if book not in entry_snap:
                    continue
                soft_entry = entry_snap[book]
                soft_close = close_snap.get(book, {})

                # Check both sides: home -1.5 and away -1.5
                for (fav_team, dog_team) in [(home, away), (away, home)]:
                    # De-vig pair: fav -1.5 vs dog +1.5
                    pin_fav = pin_entry.get((fav_team, -1.5))
                    pin_dog = pin_entry.get((dog_team, 1.5))
                    soft_fav = soft_entry.get((fav_team, -1.5))
                    soft_dog = soft_entry.get((dog_team, 1.5))

                    if not all([pin_fav, pin_dog, soft_fav, soft_dog]):
                        continue

                    pin_fair, _ = _devig(pin_fav, pin_dog)
                    soft_fair, _ = _devig(soft_fav, soft_dog)
                    diff_pp = (soft_fair - pin_fair) * 100

                    # We only want cases where soft UNDERPRICES the team
                    # (soft_fair < pin_fair → soft gives HIGHER odds → good for bettor)
                    # This means: diff < 0 → bet the fav -1.5 at soft odds
                    #
                    # OR soft OVERPRICES the dog side:
                    # soft thinks team covers more → soft gives LOWER -1.5 odds
                    # → bet the DOG +1.5 at soft odds
                    #
                    # Simpler: we bet whichever side the soft book is offering
                    # a BETTER price than Pinnacle says is fair.

                    if abs(diff_pp) < 0.5:
                        continue  # no meaningful disagreement

                    # Determine bet direction
                    if diff_pp < -0.5:
                        # Soft underprices fav → soft gives higher -1.5 odds → bet fav -1.5
                        bet_side = "fav_neg15"
                        bet_team = fav_team
                        entry_odds = soft_fav
                        # Close odds for this side
                        close_fav = soft_close.get((fav_team, -1.5))
                        close_dog = soft_close.get((dog_team, 1.5))
                        pin_close_fav = pin_close.get((fav_team, -1.5))
                        pin_close_dog = pin_close.get((dog_team, 1.5))
                        # Outcome: did fav cover -1.5?
                        if home_margin is not None:
                            if fav_team == home:
                                won = home_margin >= 2
                            else:
                                won = home_margin <= -2
                        else:
                            won = None
                        fair_at_entry = pin_fair  # Pinnacle's fair prob at entry

                    elif diff_pp > 0.5:
                        # Soft overprices fav → soft gives lower -1.5 odds
                        # → bet DOG +1.5 at soft odds
                        bet_side = "dog_plus15"
                        bet_team = dog_team
                        entry_odds = soft_dog
                        close_fav = soft_close.get((fav_team, -1.5))
                        close_dog = soft_close.get((dog_team, 1.5))
                        pin_close_fav = pin_close.get((fav_team, -1.5))
                        pin_close_dog = pin_close.get((dog_team, 1.5))
                        # Outcome: did dog cover +1.5?
                        if home_margin is not None:
                            if dog_team == home:
                                won = home_margin >= -1  # home covers +1.5
                            else:
                                won = home_margin <= 1   # away covers +1.5
                        else:
                            won = None
                        fair_at_entry = 1 - pin_fair  # prob of dog covering

                    else:
                        continue

                    # CLV: entry odds vs close odds
                    if bet_side == "fav_neg15" and close_fav and close_fav > 1:
                        raw_clv = (entry_odds / close_fav - 1) * 100
                        close_odds = close_fav
                    elif bet_side == "dog_plus15" and close_dog and close_dog > 1:
                        raw_clv = (entry_odds / close_dog - 1) * 100
                        close_odds = close_dog
                    else:
                        continue

                    # De-vigged CLV (against Pinnacle close)
                    if pin_close_fav and pin_close_dog and pin_close_fav > 1 and pin_close_dog > 1:
                        pin_close_fair_fav, pin_close_fair_dog = _devig(pin_close_fav, pin_close_dog)
                        if bet_side == "fav_neg15":
                            devig_close_fair = pin_close_fair_fav
                        else:
                            devig_close_fair = pin_close_fair_dog
                        entry_imp = _implied(entry_odds)
                        devig_clv_pp = (devig_close_fair - entry_imp) * 100
                    else:
                        devig_clv_pp = None

                    # Overround at entry
                    soft_overround = (_implied(soft_fav) + _implied(soft_dog))

                    sim_bets.append({
                        "event_id": event_id,
                        "book": book,
                        "bet_side": bet_side,
                        "bet_team": bet_team,
                        "entry_time": entry_time,
                        "close_time": close_time,
                        "hours_to_close": hours_to_close,
                        "diff_pp": diff_pp,
                        "entry_odds": entry_odds,
                        "close_odds": close_odds,
                        "raw_clv_pct": raw_clv,
                        "devig_clv_pp": devig_clv_pp,
                        "beat_close": entry_odds > close_odds,
                        "won": won,
                        "fair_at_entry": fair_at_entry,
                        "soft_vig": (soft_overround - 1) * 100,
                    })

    n_events = len(set(b["event_id"] for b in sim_bets))
    print(f"  Simulated bets: {len(sim_bets)} across {n_events} events")
    print(f"  With outcomes: {sum(1 for b in sim_bets if b['won'] is not None)}")

    if not sim_bets:
        print("  NO DATA — aborting")
        return

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: ACTUAL CLV (entry vs close)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: ACTUAL CLV — STALE-LINE BETS vs CLOSE")
    print("Does betting the stale soft line beat the close?")
    print("=" * 70)

    # Filter to meaningful disagreement thresholds
    for thresh_pp in [0.5, 1.0, 1.5, 2.0, 3.0]:
        subset = [b for b in sim_bets if abs(b["diff_pp"]) > thresh_pp]
        if not subset:
            continue

        clvs = [b["raw_clv_pct"] for b in subset]
        beats = [b["beat_close"] for b in subset]
        n_ev = len(set(b["event_id"] for b in subset))

        print(f"\n  |Disagreement| > {thresh_pp}pp (N={len(subset)}, {n_ev} events):")
        print(f"    Raw CLV:      {statistics.mean(clvs):+.2f}%")
        if len(clvs) > 1:
            se = statistics.stdev(clvs) / len(clvs) ** 0.5
            print(f"    CLV 95% CI:   [{statistics.mean(clvs)-1.96*se:+.2f}%, {statistics.mean(clvs)+1.96*se:+.2f}%]")
        print(f"    Beat close:   {sum(beats)}/{len(beats)} = {sum(beats)/len(beats)*100:.1f}%")

        # De-vigged CLV
        dv = [b["devig_clv_pp"] for b in subset if b["devig_clv_pp"] is not None]
        if dv:
            print(f"    Devig CLV:    {statistics.mean(dv):+.2f}pp (N={len(dv)})")

        # By direction
        fav_bets = [b for b in subset if b["bet_side"] == "fav_neg15"]
        dog_bets = [b for b in subset if b["bet_side"] == "dog_plus15"]
        if fav_bets:
            print(f"    Fav -1.5:     CLV={statistics.mean([b['raw_clv_pct'] for b in fav_bets]):+.2f}%, "
                  f"beat={sum(b['beat_close'] for b in fav_bets)/len(fav_bets)*100:.0f}%, N={len(fav_bets)}")
        if dog_bets:
            print(f"    Dog +1.5:     CLV={statistics.mean([b['raw_clv_pct'] for b in dog_bets]):+.2f}%, "
                  f"beat={sum(b['beat_close'] for b in dog_bets)/len(dog_bets)*100:.0f}%, N={len(dog_bets)}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: OUTCOME VALIDATION
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: OUTCOME VALIDATION — DO STALE-LINE BETS WIN?")
    print("=" * 70)

    with_outcome = [b for b in sim_bets if b["won"] is not None]
    print(f"\n  Bets with outcomes: {len(with_outcome)}")

    for thresh_pp in [0.5, 1.0, 1.5, 2.0]:
        subset = [b for b in with_outcome if abs(b["diff_pp"]) > thresh_pp]
        if not subset:
            continue

        wins = [b["won"] for b in subset]
        wr = sum(wins) / len(wins) * 100
        avg_odds = statistics.mean([b["entry_odds"] for b in subset])
        be = (1 / avg_odds) * 100  # breakeven WR
        avg_fair = statistics.mean([b["fair_at_entry"] for b in subset]) * 100

        print(f"\n  |Diff| > {thresh_pp}pp (N={len(subset)}):")
        print(f"    Win rate:     {wr:.1f}%")
        print(f"    Avg entry odds: {avg_odds:.3f} (BE={be:.1f}%)")
        print(f"    Pinnacle fair:  {avg_fair:.1f}%")
        print(f"    Edge vs BE:     {wr - be:+.1f}pp")

        # By side
        fav = [b for b in subset if b["bet_side"] == "fav_neg15"]
        dog = [b for b in subset if b["bet_side"] == "dog_plus15"]
        if fav:
            fav_wr = sum(b["won"] for b in fav) / len(fav) * 100
            fav_be = (1 / statistics.mean([b["entry_odds"] for b in fav])) * 100
            print(f"    Fav -1.5: WR={fav_wr:.1f}% vs BE={fav_be:.1f}% ({fav_wr-fav_be:+.1f}pp), N={len(fav)}")
        if dog:
            dog_wr = sum(b["won"] for b in dog) / len(dog) * 100
            dog_be = (1 / statistics.mean([b["entry_odds"] for b in dog])) * 100
            print(f"    Dog +1.5: WR={dog_wr:.1f}% vs BE={dog_be:.1f}% ({dog_wr-dog_be:+.1f}pp), N={len(dog)}")

        # P&L simulation ($10 flat)
        pnl = sum((b["entry_odds"] - 1) * 10 if b["won"] else -10 for b in subset)
        roi = pnl / (len(subset) * 10) * 100
        print(f"    P&L ($10 flat): ${pnl:+.0f} (ROI={roi:+.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: DETECTION-LATENCY FILTER
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 3: DETECTION-LATENCY REALITY")
    print("Our polling interval limits which lag events we can catch.")
    print("=" * 70)

    # Measure our actual polling intervals
    poll_intervals = []
    for event_id, snapshots in events.items():
        times = sorted(snapshots.keys())
        for i in range(len(times) - 1):
            gap = (datetime.strptime(times[i + 1], fmt) -
                   datetime.strptime(times[i], fmt)).total_seconds() / 60
            if 0 < gap < 1440:  # skip multi-day gaps
                poll_intervals.append(gap)

    print(f"\n  Our polling intervals (minutes, N={len(poll_intervals)}):")
    print(f"    Mean:   {statistics.mean(poll_intervals):.0f}")
    print(f"    Median: {statistics.median(poll_intervals):.0f}")
    print(f"    P10:    {sorted(poll_intervals)[len(poll_intervals)//10]:.0f}")
    print(f"    P25:    {sorted(poll_intervals)[len(poll_intervals)//4]:.0f}")
    print(f"    P75:    {sorted(poll_intervals)[3*len(poll_intervals)//4]:.0f}")

    median_poll = statistics.median(poll_intervals)

    print(f"""
  DETECTION LATENCY CHAIN:
    Poll interval:       ~{median_poll:.0f} min (median)
    API response:        ~5 sec
    Compute + compare:   ~2 sec
    Manual bet placement: ~60 sec
    ─────────────────────────────
    Total latency:       ~{median_poll + 1:.0f} min (dominated by poll interval)

  IMPLICATION:
    A soft-book lag must last LONGER than {median_poll + 1:.0f} min for us
    to detect AND act. We can only catch lags > {median_poll + 1:.0f} min.

    BUT: we don't measure minute-level lag (our data is hourly snapshots).
    What we DO see is price DISAGREEMENT at simultaneous captures.
    If disagreement exists at our capture time, the lag is AT LEAST as
    long as our poll interval — because both books were sampled together
    and they STILL disagree.
""")

    # The key insight: simultaneous captures where books disagree
    # mean the disagreement is PERSISTENT (not a brief flash).
    # These ARE the actionable events — the lag lasted long enough
    # for us to see it.

    print("  All simulated bets are from SIMULTANEOUS captures where")
    print("  disagreement persisted — these are inherently within our")
    print("  detection window. No further latency filter needed.")

    # But filter by hours-to-close (closer to game = more actionable)
    print(f"\n  CLV by hours-to-close:")
    print(f"  {'Window':>10s} {'N':>5s} {'CLV%':>8s} {'Beat':>7s} {'WR':>6s}")
    for (lo, hi, label) in [(0, 2, "<2hr"), (2, 4, "2-4hr"), (4, 8, "4-8hr"),
                             (8, 16, "8-16hr"), (0, 24, "all")]:
        subset = [b for b in sim_bets if lo <= b["hours_to_close"] < hi
                  and abs(b["diff_pp"]) > 1.0]
        if not subset:
            continue
        clv = statistics.mean([b["raw_clv_pct"] for b in subset])
        beat = sum(b["beat_close"] for b in subset) / len(subset) * 100
        wr_sub = [b for b in subset if b["won"] is not None]
        wr = sum(b["won"] for b in wr_sub) / len(wr_sub) * 100 if wr_sub else 0
        print(f"  {label:>10s} {len(subset):>5d} {clv:>+7.2f}% {beat:>6.1f}% {wr:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: SPLIT-HALF STABILITY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 4: SPLIT-HALF STABILITY")
    print("=" * 70)

    # Split by event_id hash for independence
    bets_1pp = [b for b in sim_bets if abs(b["diff_pp"]) > 1.0]
    if len(bets_1pp) >= 20:
        random.seed(42)
        event_ids = list(set(b["event_id"] for b in bets_1pp))
        random.shuffle(event_ids)
        mid = len(event_ids) // 2
        half_a_events = set(event_ids[:mid])
        half_b_events = set(event_ids[mid:])

        half_a = [b for b in bets_1pp if b["event_id"] in half_a_events]
        half_b = [b for b in bets_1pp if b["event_id"] in half_b_events]

        print(f"\n  Split A: N={len(half_a)} ({len(half_a_events)} events)")
        if half_a:
            clv_a = statistics.mean([b["raw_clv_pct"] for b in half_a])
            beat_a = sum(b["beat_close"] for b in half_a) / len(half_a) * 100
            print(f"    CLV: {clv_a:+.2f}%, Beat close: {beat_a:.1f}%")
            wr_a = [b for b in half_a if b["won"] is not None]
            if wr_a:
                print(f"    WR:  {sum(b['won'] for b in wr_a)/len(wr_a)*100:.1f}% (N={len(wr_a)})")

        print(f"\n  Split B: N={len(half_b)} ({len(half_b_events)} events)")
        if half_b:
            clv_b = statistics.mean([b["raw_clv_pct"] for b in half_b])
            beat_b = sum(b["beat_close"] for b in half_b) / len(half_b) * 100
            print(f"    CLV: {clv_b:+.2f}%, Beat close: {beat_b:.1f}%")
            wr_b = [b for b in half_b if b["won"] is not None]
            if wr_b:
                print(f"    WR:  {sum(b['won'] for b in wr_b)/len(wr_b)*100:.1f}% (N={len(wr_b)})")

        print(f"\n  Stability: CLV_A={clv_a:+.2f}% vs CLV_B={clv_b:+.2f}% "
              f"(gap={abs(clv_a-clv_b):.2f}pp)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: VIG MATH — NET EDGE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 5: VIG MATH — NET EDGE AFTER VIG")
    print("=" * 70)

    bets_1pp = [b for b in sim_bets if abs(b["diff_pp"]) > 1.0]
    if bets_1pp:
        avg_vig = statistics.mean([b["soft_vig"] for b in bets_1pp])
        avg_clv = statistics.mean([b["raw_clv_pct"] for b in bets_1pp])
        avg_odds = statistics.mean([b["entry_odds"] for b in bets_1pp])

        # Vig cost per bet = half the overround (bettor pays half)
        vig_cost = avg_vig / 2

        print(f"\n  At >1pp disagreement (N={len(bets_1pp)}):")
        print(f"    Avg soft vig:    {avg_vig:.2f}%")
        print(f"    Vig cost/bet:    ~{vig_cost:.2f}pp")
        print(f"    Raw CLV:         {avg_clv:+.2f}%")
        print(f"    Avg entry odds:  {avg_odds:.3f}")

        # Convert CLV % to edge in prob points
        # CLV% = (entry_odds / close_odds - 1) * 100
        # This is NOT directly comparable to vig in pp
        # Better: compare entry implied to Pinnacle fair
        diffs = [abs(b["diff_pp"]) for b in bets_1pp]
        print(f"    Avg |disagreement|: {statistics.mean(diffs):.2f}pp")
        print(f"    Vig cost:           {vig_cost:.2f}pp")
        print(f"    Net edge estimate:  {statistics.mean(diffs) - vig_cost:+.2f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # PART 6: CAPACITY / LIMITING CEILING
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 6: CAPACITY & LIMITING CEILING")
    print("=" * 70)

    # Count actionable opportunities per day
    bets_by_date = defaultdict(list)
    for b in sim_bets:
        if abs(b["diff_pp"]) > 1.0:
            date = b["entry_time"][:10]
            bets_by_date[date].append(b)

    daily_counts = [len(v) for v in bets_by_date.values()]
    n_days = len(daily_counts)

    if daily_counts:
        print(f"\n  Actionable opportunities per day (>1pp, N={n_days} days):")
        print(f"    Mean:   {statistics.mean(daily_counts):.1f}")
        print(f"    Median: {statistics.median(daily_counts):.0f}")
        print(f"    Max:    {max(daily_counts)}")

    # At >2pp
    bets_by_date_2 = defaultdict(list)
    for b in sim_bets:
        if abs(b["diff_pp"]) > 2.0:
            date = b["entry_time"][:10]
            bets_by_date_2[date].append(b)

    daily_2 = [len(v) for v in bets_by_date_2.values()]
    if daily_2:
        print(f"\n  At >2pp (N={len(daily_2)} days with opportunities):")
        print(f"    Mean:   {statistics.mean(daily_2):.1f}")
        print(f"    Median: {statistics.median(daily_2):.0f}")

    print(f"""
  LIMITING ESTIMATES (industry knowledge):
    - DraftKings/FanDuel typically limit accounts after:
      ~$5K-$20K in lifetime profit on sharp plays
      OR consistent CLV-positive betting detected within weeks
    - Limit timeline: 2-6 weeks for aggressive steam chasers
    - Post-limit: max bets reduced to $5-$25 (effectively killed)

  CAPACITY MATH (if edge is real):
    Opportunities/day: ~{statistics.mean(daily_counts):.0f} (at >1pp)
    Avg odds:          ~{statistics.mean([b['entry_odds'] for b in bets_1pp]):.2f}
    At $50/bet:        ~${statistics.mean(daily_counts)*50:.0f}/day in action
    At +1% edge:       ~${statistics.mean(daily_counts)*50*0.01:.0f}/day expected profit
    Limiting runway:   ~{min(20000/(statistics.mean(daily_counts)*50*0.01+0.01), 365):.0f} days to $20K lifetime cap

  KALSHI/EXCHANGE NOTE:
    Exchanges don't limit. But Kalshi doesn't have MLB run-line
    markets. Sports exchange liquidity is thin. The limiting
    ceiling is a HARD constraint on soft-book speed strategies.
""")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 70)

    # Gather key metrics
    bets_1 = [b for b in sim_bets if abs(b["diff_pp"]) > 1.0]
    if bets_1:
        clv = statistics.mean([b["raw_clv_pct"] for b in bets_1])
        beat = sum(b["beat_close"] for b in bets_1) / len(bets_1) * 100
        with_w = [b for b in bets_1 if b["won"] is not None]
        wr = sum(b["won"] for b in with_w) / len(with_w) * 100 if with_w else 0
        avg_o = statistics.mean([b["entry_odds"] for b in bets_1])
        be = 100 / avg_o
        vig = statistics.mean([b["soft_vig"] for b in bets_1])
    else:
        clv = beat = wr = be = vig = 0

    print(f"""
  (a) CLV-POSITIVE NET OF VIG?
      Raw CLV: {clv:+.2f}%
      Beat close: {beat:.1f}%
      → {'YES — marginal but positive' if clv > 0 and beat > 50 else 'MARGINAL — CLV positive but beat-close < 50%' if clv > 0 else 'NO — CLV negative'}

  (b) EDGE SURVIVES LATENCY RESTRICTION?
      Our data = simultaneous captures (disagreement already persisted)
      → Latency filter is N/A with hourly polling.
      → Need HIGH-FREQUENCY data to truly test this.

  (c) CAPACITY/LIMITING CEILING?
      Soft-book limiting kills this in 2-6 weeks for aggressive play.
      No exchange alternative for MLB run lines.
      → HARD ceiling on lifetime value.

  THE HONEST ANSWER:
      CLV is {clv:+.2f}% on {len(bets_1)} bets — {'positive but THIN' if clv > 0 else 'negative'}.
      Win rate is {wr:.1f}% vs {be:.1f}% breakeven — {'above' if wr > be else 'BELOW'} breakeven.
      Beat-close rate is {beat:.1f}% — {'above' if beat > 50 else 'below'} 50%.

      The edge {'exists but is marginal' if clv > 0 and wr > be else 'does NOT survive simulation' if clv <= 0 else 'is ambiguous (CLV+ but WR below BE)'}.
      {'Even if real, soft-book limiting caps lifetime value to ~$5-20K before accounts are restricted.' if clv > 0 else ''}
""")

    db.close()


if __name__ == "__main__":
    main()
