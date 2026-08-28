#!/usr/bin/env python3
"""
Cross-Book Main-Line Lag Test
=============================
Tests: when sharp books (Pinnacle) move, do soft books (DK/FD/BetMGM)
lag on their MAIN line — and is that lag exploitable at 4-5% vig?

De-vig pair: Team_A -1.5 vs Team_B +1.5 (complementary events).
"""

import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime

DB_PATH = "data/historical.db"
SOFT_BOOKS = ["draftkings", "fanduel", "betmgm"]


def _implied(decimal_odds):
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def _devig_pair(odds_side_a, odds_side_b):
    """De-vig a complementary pair. Returns (fair_prob_a, fair_prob_b)."""
    imp_a = _implied(odds_side_a)
    imp_b = _implied(odds_side_b)
    total = imp_a + imp_b
    if total == 0:
        return 0.5, 0.5
    return imp_a / total, imp_b / total


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print("=" * 70)
    print("CROSS-BOOK MAIN-LINE LAG TEST")
    print("Sharp (Pinnacle) vs Soft books — simultaneous -1.5 spread captures")
    print("=" * 70)

    # ── Load all -1.5/+1.5 spread data from alt_lines ──
    print("\n── Loading data ──")
    rows = db.execute("""
        SELECT fetched_at, event_id, bookmaker, team, odds, line,
               home_team, away_team, game_date_ct
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_spreads'
          AND line IN (-1.5, 1.5)
          AND bookmaker IN ('pinnacle', 'draftkings', 'fanduel', 'betmgm')
        ORDER BY event_id, fetched_at, bookmaker, team
    """).fetchall()
    print(f"  Raw rows: {len(rows)}")

    # Organize: {(event_id, fetched_at)} → {bookmaker: {(team, line): odds}}
    snapshots = defaultdict(lambda: defaultdict(dict))
    event_teams = {}  # event_id → (home, away)

    for r in rows:
        key = (r["event_id"], r["fetched_at"])
        snapshots[key][r["bookmaker"]][(r["team"], r["line"])] = r["odds"]
        event_teams[r["event_id"]] = (r["home_team"], r["away_team"])

    # Build comparisons: for each (event, time), de-vig Pinnacle and each soft book
    # De-vig pair: home -1.5 vs away +1.5 (complementary)
    comparisons = []
    for (event_id, fetched_at), books in snapshots.items():
        if "pinnacle" not in books:
            continue
        pin = books["pinnacle"]
        home, away = event_teams[event_id]

        # Get Pinnacle pair: home -1.5 vs away +1.5
        pin_home_fav = pin.get((home, -1.5))  # home covers -1.5
        pin_away_dog = pin.get((away, 1.5))    # away covers +1.5 (complement)
        if not pin_home_fav or not pin_away_dog:
            continue

        pin_fair_home_covers, _ = _devig_pair(pin_home_fav, pin_away_dog)
        pin_vig = _implied(pin_home_fav) + _implied(pin_away_dog)

        # Also get the reverse: away -1.5 vs home +1.5
        pin_away_fav = pin.get((away, -1.5))
        pin_home_dog = pin.get((home, 1.5))
        if pin_away_fav and pin_home_dog:
            pin_fair_away_covers, _ = _devig_pair(pin_away_fav, pin_home_dog)
        else:
            pin_fair_away_covers = None

        for book in SOFT_BOOKS:
            if book not in books:
                continue
            soft = books[book]

            # Soft: home -1.5 vs away +1.5
            soft_home_fav = soft.get((home, -1.5))
            soft_away_dog = soft.get((away, 1.5))

            # Soft: away -1.5 vs home +1.5
            soft_away_fav = soft.get((away, -1.5))
            soft_home_dog = soft.get((home, 1.5))

            # Compare home -1.5 side
            if soft_home_fav and soft_away_dog:
                soft_fair_home, _ = _devig_pair(soft_home_fav, soft_away_dog)
                soft_vig = _implied(soft_home_fav) + _implied(soft_away_dog)
                comparisons.append({
                    "event_id": event_id,
                    "fetched_at": fetched_at,
                    "team": home,
                    "side": "home -1.5",
                    "book": book,
                    "pin_fair": pin_fair_home_covers,
                    "soft_fair": soft_fair_home,
                    "diff_pp": (soft_fair_home - pin_fair_home_covers) * 100,
                    "pin_odds": pin_home_fav,
                    "soft_odds": soft_home_fav,
                    "pin_vig": pin_vig,
                    "soft_vig": soft_vig,
                })

            # Compare away -1.5 side
            if soft_away_fav and soft_home_dog and pin_fair_away_covers:
                soft_fair_away, _ = _devig_pair(soft_away_fav, soft_home_dog)
                soft_vig2 = _implied(soft_away_fav) + _implied(soft_home_dog)
                comparisons.append({
                    "event_id": event_id,
                    "fetched_at": fetched_at,
                    "team": away,
                    "side": "away -1.5",
                    "book": book,
                    "pin_fair": pin_fair_away_covers,
                    "soft_fair": soft_fair_away,
                    "diff_pp": (soft_fair_away - pin_fair_away_covers) * 100,
                    "pin_odds": pin_away_fav,
                    "soft_odds": soft_away_fav,
                    "pin_vig": _implied(pin_away_fav) + _implied(pin_home_dog),
                    "soft_vig": soft_vig2,
                })

    n_events = len(set(c["event_id"] for c in comparisons))
    print(f"  Comparison pairs: {len(comparisons)} across {n_events} events")

    if not comparisons:
        print("  NO DATA — aborting")
        return

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Simultaneous cross-book disagreement
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: SIMULTANEOUS CROSS-BOOK DISAGREEMENT")
    print("At the same capture instant, how much do soft books disagree")
    print("with Pinnacle on -1.5 spread fair probability?")
    print("=" * 70)

    by_book = defaultdict(list)
    for c in comparisons:
        by_book[c["book"]].append(c)

    for book in SOFT_BOOKS:
        data = by_book[book]
        if not data:
            print(f"\n  {book}: NO DATA")
            continue
        diffs = [c["diff_pp"] for c in data]
        abs_diffs = [abs(d) for d in diffs]
        n = len(data)
        print(f"\n  {book} (N={n}):")
        print(f"    Mean diff:     {statistics.mean(diffs):+.2f}pp")
        print(f"    Std diff:      {statistics.stdev(diffs):.2f}pp")
        print(f"    MAE:           {statistics.mean(abs_diffs):.2f}pp")
        print(f"    Median |diff|: {statistics.median(abs_diffs):.2f}pp")
        for thresh in [1, 2, 3, 5]:
            ct = sum(1 for d in abs_diffs if d > thresh)
            print(f"    >{thresh}pp:          {ct}/{n} = {ct/n*100:.1f}%")

    all_diffs = [c["diff_pp"] for c in comparisons]
    all_abs = [abs(d) for d in all_diffs]
    n = len(comparisons)
    print(f"\n  ALL SOFT (N={n}):")
    print(f"    Mean diff:     {statistics.mean(all_diffs):+.2f}pp")
    print(f"    MAE:           {statistics.mean(all_abs):.2f}pp")
    for thresh in [1, 2, 3, 5]:
        ct = sum(1 for d in all_abs if d > thresh)
        print(f"    >{thresh}pp:          {ct}/{n} = {ct/n*100:.1f}%")

    # Distribution of disagreement
    buckets = [0, 0.5, 1, 1.5, 2, 3, 5, 10, 100]
    print(f"\n  Disagreement distribution (|soft_fair - pin_fair|):")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        ct = sum(1 for d in all_abs if lo <= d < hi)
        print(f"    [{lo:.1f}, {hi:.1f})pp: {ct:>5d} ({ct/n*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Vig comparison (main-line -1.5 spreads)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: VIG COMPARISON (-1.5 SPREAD)")
    print("=" * 70)

    pin_vigs = [c["pin_vig"] for c in comparisons]
    print(f"\n  Pinnacle overround: {statistics.mean(pin_vigs):.4f} "
          f"(vig = {(statistics.mean(pin_vigs)-1)*100:.2f}%)")
    for book in SOFT_BOOKS:
        vigs = [c["soft_vig"] for c in by_book[book]]
        if vigs:
            print(f"  {book:15s}: {statistics.mean(vigs):.4f} "
                  f"(vig = {(statistics.mean(vigs)-1)*100:.2f}%)")

    # Also get h2h vig from line_snapshots
    print(f"\n  H2H moneyline vig (from line_snapshots):")
    h2h_vigs = {}
    for book in SOFT_BOOKS + ["betrivers"]:
        rows = db.execute("""
            SELECT ls1.price as p1, ls2.price as p2
            FROM line_snapshots ls1
            JOIN line_snapshots ls2
              ON ls1.event_id = ls2.event_id
              AND ls1.captured_at = ls2.captured_at
              AND ls1.bookmaker = ls2.bookmaker
              AND ls1.market_key = ls2.market_key
              AND ls1.outcome_name != ls2.outcome_name
            WHERE ls1.sport_key = 'baseball_mlb'
              AND ls1.market_key = 'h2h'
              AND ls1.bookmaker = ?
              AND ls1.price > 1.0 AND ls2.price > 1.0
        """, (book,)).fetchall()
        if rows:
            overrounds = [_implied(r[0]) + _implied(r[1]) for r in rows]
            avg_vig = (statistics.mean(overrounds) - 1) * 100
            h2h_vigs[book] = avg_vig
            print(f"    {book:15s}: {avg_vig:.2f}% (N={len(rows)})")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Movement tracking between snapshots
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 3: PINNACLE MOVEMENT TRACKING")
    print("When Pinnacle moves between captures, do soft books follow?")
    print("=" * 70)

    # Group by (event, team, book) → time series
    ts = defaultdict(list)
    for c in comparisons:
        ts[(c["event_id"], c["team"], c["book"])].append(c)

    multi = {k: sorted(v, key=lambda x: x["fetched_at"])
             for k, v in ts.items() if len(v) >= 2}
    print(f"\n  Time series with 2+ observations: {len(multi)}")

    deltas = []
    for key, series in multi.items():
        for i in range(len(series) - 1):
            t0, t1 = series[i], series[i + 1]
            pin_d = (t1["pin_fair"] - t0["pin_fair"]) * 100
            soft_d = (t1["soft_fair"] - t0["soft_fair"]) * 100
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            gap = (datetime.strptime(t1["fetched_at"], fmt) -
                   datetime.strptime(t0["fetched_at"], fmt)).total_seconds() / 60
            deltas.append({
                "event_id": key[0], "team": key[1], "book": key[2],
                "pin_d": pin_d, "soft_d": soft_d, "lag": pin_d - soft_d,
                "gap_min": gap,
                "t0_diff": t0["diff_pp"], "t1_diff": t1["diff_pp"],
            })

    print(f"  Consecutive pairs: {len(deltas)}")

    # Filter to Pinnacle moved > 0.5pp
    moved = [d for d in deltas if abs(d["pin_d"]) > 0.5]
    print(f"  Pinnacle moved >0.5pp: {len(moved)}")

    if moved:
        print(f"    Avg |pin delta|:  {statistics.mean([abs(d['pin_d']) for d in moved]):.2f}pp")
        print(f"    Avg |soft delta|: {statistics.mean([abs(d['soft_d']) for d in moved]):.2f}pp")
        print(f"    Avg lag:          {statistics.mean([d['lag'] for d in moved]):+.2f}pp")
        print(f"    Avg time gap:     {statistics.mean([d['gap_min'] for d in moved]):.0f} min")

        # Soft lagged (didn't follow pin in same direction)
        lagged = [d for d in moved
                  if d["pin_d"] * d["lag"] > 0 and abs(d["lag"]) > 0.5]
        print(f"\n    Soft LAGGED (>0.5pp in pin's direction): {len(lagged)}/{len(moved)}")
        if lagged:
            print(f"    Avg lag: {statistics.mean([abs(d['lag']) for d in lagged]):.2f}pp")
            print(f"    Avg gap: {statistics.mean([d['gap_min'] for d in lagged]):.0f} min")
            lagged.sort(key=lambda x: abs(x["lag"]), reverse=True)
            print(f"\n    Top 10 lag events:")
            print(f"    {'Book':<12s} {'Pin Δ':>7s} {'Soft Δ':>7s} {'Lag':>7s} {'Gap':>6s}")
            for d in lagged[:10]:
                print(f"    {d['book']:<12s} {d['pin_d']:>+6.1f}p {d['soft_d']:>+6.1f}p "
                      f"{d['lag']:>+6.1f}p {d['gap_min']:>5.0f}m")

        # Soft over-reacted (moved MORE than pin)
        over = [d for d in moved
                if d["pin_d"] * d["lag"] < 0 and abs(d["lag"]) > 0.5]
        print(f"\n    Soft OVER-REACTED (>0.5pp past pin): {len(over)}/{len(moved)}")

    # Pin moved >1pp
    big_moves = [d for d in deltas if abs(d["pin_d"]) > 1.0]
    print(f"\n  Pinnacle moved >1pp: {len(big_moves)}")
    if big_moves:
        for d in sorted(big_moves, key=lambda x: abs(x["pin_d"]), reverse=True)[:10]:
            print(f"    {d['book']:<12s} pin={d['pin_d']:+.1f}pp soft={d['soft_d']:+.1f}pp "
                  f"lag={d['lag']:+.1f}pp gap={d['gap_min']:.0f}m")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: CLV of disagreement-based bets
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 4: CLV OF STALE-LINE BETS")
    print("When soft disagrees with Pinnacle, does betting the soft line")
    print("beat its own close?")
    print("=" * 70)

    # For each (event, team, book), use LAST capture as "close"
    # and earlier captures with >1pp disagreement as "entry"
    event_series = defaultdict(list)
    for c in comparisons:
        event_series[(c["event_id"], c["team"], c["book"])].append(c)

    clv_results = []
    for key, series in event_series.items():
        if len(series) < 2:
            continue
        series.sort(key=lambda x: x["fetched_at"])
        close = series[-1]

        for snap in series[:-1]:
            diff = snap["diff_pp"]
            if abs(diff) < 1.0:
                continue

            entry_odds = snap["soft_odds"]
            close_odds = close["soft_odds"]
            if entry_odds <= 1.01 or close_odds <= 1.01:
                continue

            # Positive CLV = entry odds were better than close odds
            # For -1.5 spread: higher odds = better for bettor
            raw_clv = (entry_odds / close_odds - 1) * 100
            # If soft underpriced (diff < 0 means soft_fair < pin_fair),
            # soft gives HIGHER odds on this side → potentially +CLV
            # because the market should correct toward Pinnacle
            #
            # Direction: if diff < 0 (soft underestimates),
            # close should move UP (fair prob increases toward pin),
            # meaning close_odds should DECREASE → positive CLV

            clv_results.append({
                "event_id": key[0], "team": key[1], "book": key[2],
                "diff_pp": diff,
                "entry_odds": entry_odds, "close_odds": close_odds,
                "clv_pct": raw_clv,
                "beat_close": entry_odds > close_odds,
                "direction": "soft_under" if diff < 0 else "soft_over",
            })

    if clv_results:
        print(f"\n  Legs with >1pp disagreement + later capture: {len(clv_results)}")
        clvs = [c["clv_pct"] for c in clv_results]
        beats = [c["beat_close"] for c in clv_results]
        print(f"  Mean CLV: {statistics.mean(clvs):+.2f}%")
        print(f"  Beat close: {sum(beats)}/{len(beats)} = {sum(beats)/len(beats)*100:.1f}%")

        # By direction
        for direction in ["soft_under", "soft_over"]:
            subset = [c for c in clv_results if c["direction"] == direction]
            if subset:
                label = ("Soft UNDERPRICES (soft_fair < pin → higher odds, bet FOR)"
                         if direction == "soft_under"
                         else "Soft OVERPRICES (soft_fair > pin → lower odds)")
                print(f"\n  {label}:")
                print(f"    N = {len(subset)}")
                print(f"    Avg |diff|: {statistics.mean([abs(c['diff_pp']) for c in subset]):.2f}pp")
                print(f"    Avg CLV: {statistics.mean([c['clv_pct'] for c in subset]):+.2f}%")
                print(f"    Beat close: {sum(c['beat_close'] for c in subset)}/{len(subset)} = "
                      f"{sum(c['beat_close'] for c in subset)/len(subset)*100:.1f}%")

        # CLV by disagreement threshold
        print(f"\n  CLV by disagreement threshold:")
        print(f"  {'Thresh':>6s} {'N':>5s} {'Avg|diff|':>10s} {'CLV%':>8s} {'Beat':>7s}")
        for thresh in [1, 1.5, 2, 3, 5]:
            subset = [c for c in clv_results if abs(c["diff_pp"]) > thresh]
            if subset:
                print(f"  >{thresh:.1f}pp {len(subset):>5d} "
                      f"{statistics.mean([abs(c['diff_pp']) for c in subset]):>9.2f}pp "
                      f"{statistics.mean([c['clv_pct'] for c in subset]):>+7.2f}% "
                      f"{sum(c['beat_close'] for c in subset)/len(subset)*100:>6.1f}%")
    else:
        print("\n  No CLV data (need 2+ captures with >1pp disagreement)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: Outcome validation — does Pinnacle predict better?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 5: OUTCOME VALIDATION")
    print("When Pinnacle and soft disagree, who's right?")
    print("=" * 70)

    # Load game outcomes
    outcomes = {}
    game_rows = db.execute("""
        SELECT g.home_team, g.away_team, g.game_date, g.home_score, g.away_score
        FROM games g
        WHERE g.sport = 'MLB' AND g.home_score IS NOT NULL
    """).fetchall()
    for r in game_rows:
        outcomes[(r["home_team"], r["away_team"], r["game_date"])] = {
            "home_score": r["home_score"],
            "away_score": r["away_score"],
            "home_covers_neg15": r["home_score"] - r["away_score"] >= 2,
            "away_covers_neg15": r["away_score"] - r["home_score"] >= 2,
        }

    # Match comparisons to outcomes
    matched = []
    for c in comparisons:
        eid = c["event_id"]
        if eid not in event_teams:
            continue
        home, away = event_teams[eid]

        # Try to find outcome
        # Need game_date_ct from the event
        gdate = None
        for (ev, ft), books in snapshots.items():
            if ev == eid:
                # Get game_date_ct from any row
                for bk, entries in books.items():
                    break
                break

        # Use alt_lines to get game_date_ct
        gd_row = db.execute("""
            SELECT game_date_ct FROM alt_lines
            WHERE event_id = ? LIMIT 1
        """, (eid,)).fetchone()
        if not gd_row:
            continue
        gdate = gd_row["game_date_ct"]

        outcome = outcomes.get((home, away, gdate))
        if not outcome:
            continue

        team = c["team"]
        if team == home:
            covers = outcome["home_covers_neg15"]
        elif team == away:
            covers = outcome["away_covers_neg15"]
        else:
            continue

        matched.append({**c, "covers": covers})

    print(f"\n  Comparisons with outcomes: {len(matched)}")

    if matched:
        # Brier score: Pinnacle vs each soft book
        print(f"\n  Brier scores (lower = better):")
        for source in ["pinnacle"] + SOFT_BOOKS:
            if source == "pinnacle":
                data = matched
                key = "pin_fair"
            else:
                data = [m for m in matched if m["book"] == source]
                key = "soft_fair"
            if not data:
                continue
            brier = statistics.mean([(m[key] - (1 if m["covers"] else 0)) ** 2
                                     for m in data])
            print(f"    {source:15s}: {brier:.6f} (N={len(data)})")

        # When they disagree >1pp, who's right?
        disagree = [m for m in matched if abs(m["diff_pp"]) > 1]
        if disagree:
            print(f"\n  When they disagree >1pp (N={len(disagree)}):")

            # Pin says higher prob → did it cover more?
            pin_higher = [m for m in disagree if m["diff_pp"] < 0]  # soft < pin
            pin_lower = [m for m in disagree if m["diff_pp"] > 0]   # soft > pin

            if pin_higher:
                wr = sum(m["covers"] for m in pin_higher) / len(pin_higher)
                avg_pin = statistics.mean([m["pin_fair"] for m in pin_higher])
                avg_soft = statistics.mean([m["soft_fair"] for m in pin_higher])
                print(f"    Pin says HIGHER prob (N={len(pin_higher)}): "
                      f"covers {wr*100:.1f}% (pin_fair={avg_pin*100:.1f}%, soft_fair={avg_soft*100:.1f}%)")

            if pin_lower:
                wr = sum(m["covers"] for m in pin_lower) / len(pin_lower)
                avg_pin = statistics.mean([m["pin_fair"] for m in pin_lower])
                avg_soft = statistics.mean([m["soft_fair"] for m in pin_lower])
                print(f"    Pin says LOWER prob  (N={len(pin_lower)}): "
                      f"covers {wr*100:.1f}% (pin_fair={avg_pin*100:.1f}%, soft_fair={avg_soft*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 6: Polling resolution & honest assessment
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 6: DATA RESOLUTION & HONEST ASSESSMENT")
    print("=" * 70)

    # Measure polling intervals
    event_capture_times = defaultdict(set)
    for c in comparisons:
        event_capture_times[c["event_id"]].add(c["fetched_at"])

    intervals = []
    for eid, times in event_capture_times.items():
        times = sorted(times)
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        for i in range(len(times) - 1):
            gap = (datetime.strptime(times[i + 1], fmt) -
                   datetime.strptime(times[i], fmt)).total_seconds() / 60
            if gap > 0:
                intervals.append(gap)

    if intervals:
        print(f"\n  Polling intervals (same event, minutes):")
        print(f"    Mean:    {statistics.mean(intervals):.0f}")
        print(f"    Median:  {statistics.median(intervals):.0f}")
        print(f"    P10:     {sorted(intervals)[len(intervals)//10]:.0f}")
        print(f"    P90:     {sorted(intervals)[9*len(intervals)//10]:.0f}")
        print(f"    <5 min:  {sum(1 for i in intervals if i < 5)}/{len(intervals)}")
        print(f"    <30 min: {sum(1 for i in intervals if i < 30)}/{len(intervals)}")
        print(f"    <60 min: {sum(1 for i in intervals if i < 60)}/{len(intervals)}")

    # H2H vig for the final verdict
    avg_h2h_vig = statistics.mean(list(h2h_vigs.values())) if h2h_vigs else 4.5
    avg_spread_vig_pin = (statistics.mean(pin_vigs) - 1) * 100
    soft_spread_vigs = [(statistics.mean([c["soft_vig"] for c in by_book[b]]) - 1) * 100
                        for b in SOFT_BOOKS if by_book[b]]
    avg_spread_vig_soft = statistics.mean(soft_spread_vigs) if soft_spread_vigs else 5.0

    print(f"""
  ════════════════════════════════════════════════════════════════
  VERDICT
  ════════════════════════════════════════════════════════════════

  VIG MATH (the user's core thesis):
    Prop vig:        6.9%  → breakeven edge ~3.5pp → KILLED (CLV = -3.23pp)
    -1.5 spread vig: Pinnacle {avg_spread_vig_pin:.1f}% / Soft {avg_spread_vig_soft:.1f}%
    H2H ML vig:      Soft avg {avg_h2h_vig:.1f}%
    → Main-line breakeven edge: ~{avg_h2h_vig/2:.1f}pp

  DISAGREMENT SIZE:
    Soft books disagree with Pinnacle by MAE = {statistics.mean(all_abs):.2f}pp
    Cases >2pp: {sum(1 for d in all_abs if d > 2)/len(all_abs)*100:.1f}% of captures
    Cases >3pp: {sum(1 for d in all_abs if d > 3)/len(all_abs)*100:.1f}% of captures

  DATA RESOLUTION (the hard constraint):
    Our polling is ~hourly, NOT per-minute.
    We CANNOT measure minute-level lag timing with this data.
    We CAN measure price disagreement at simultaneous captures.

  WHAT'S NEEDED FOR A TRUE TEST:
    High-frequency capture: poll every 1-2 minutes for 50+ events
    on game day (~3hr window each), correlating line moves with
    lineup/scratch announcements. Cost: ~30K OddsAPI credits.
    This would directly measure: how many minutes do soft books
    lag behind Pinnacle after a sharp move?
""")

    db.close()


if __name__ == "__main__":
    main()
