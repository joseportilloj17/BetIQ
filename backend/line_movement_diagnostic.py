#!/usr/bin/env python3
"""
Line movement structure diagnostic — selection-free analysis.

Q: Is line movement directionally predictable INDEPENDENT of our model?
   If yes → selection-free signal (reverse-line-movement / steam).
   If no → info timing is dead for spreads.

Data: bets.db line_movement_log (219K entries, 776 MLB fixtures, FanDuel)
      + link to mock_bets for settlement (865 legs with fixture_id)

Tests:
  1. CHARACTERIZE: when do lines move? Is there a time-of-day softness pattern?
  2. PREDICTABILITY: does early movement predict close? (direction, magnitude)
  3. OUTCOME PREDICTION: does line movement predict who covers, UNCONDITIONALLY?
     (Not "do our picks win" — does the LINE MOVE predict the outcome)
  4. REVERSE-LINE-MOVEMENT: when odds shift one way but the line doesn't follow,
     does the non-moving side win? (sharp money tell)
"""
import sqlite3
import numpy as np
from collections import defaultdict
from datetime import datetime

DB_PATH = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"


def load_line_trajectories():
    """
    Load MLB line trajectories from line_movement_log.
    Group by fixture_id + market_type, order by fetched_at.
    """
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT fixture_id, market_type, point, side, decimal_odds, fetched_at
        FROM line_movement_log
        WHERE sport = 'MLB'
        ORDER BY fixture_id, market_type, side, fetched_at
    """).fetchall()

    # Group: (fixture_id, market_type, side) → [(odds, timestamp)]
    trajectories = defaultdict(list)
    for fix_id, mkt, point, side, odds, fetched_at in rows:
        key = (fix_id, mkt, side, point)
        trajectories[key].append((odds, fetched_at))

    conn.close()
    return trajectories


def load_settled_outcomes():
    """
    Load settled bet outcomes linked to fixture_ids.
    Returns {fixture_id: {market, result, odds, description, spread}}.
    """
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT l.fixture_id, l.market_type, l.leg_result, b.odds, l.description,
               l.win_prob, b.game_date
        FROM mock_bet_legs l
        JOIN mock_bets b ON l.mock_bet_id = b.id
        WHERE l.fixture_id IS NOT NULL
          AND b.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND l.leg_result IN ('WIN', 'LOSS')
          AND b.source IN ('shadow_negbin', 'shadow_negbin_edgegate',
                           'shadow_negbin_fullstack', 'ats_ungated')
          AND b.cohort = 'clean'
    """).fetchall()

    outcomes = {}
    for fix_id, mkt, result, odds, desc, wp, gdate in rows:
        if fix_id not in outcomes:
            outcomes[fix_id] = {
                "market": mkt, "won": result == "WIN", "odds": odds,
                "desc": desc, "wp": wp, "game_date": gdate,
            }

    conn.close()
    return outcomes


def analyze_movement_structure():
    """Part 1: Characterize when and how lines move."""
    print("="*70)
    print("PART 1: LINE MOVEMENT STRUCTURE")
    print("="*70)

    trajs = load_line_trajectories()

    # Filter to spreads market, home side (canonical)
    spread_trajs = {}
    for (fix_id, mkt, side, point), snaps in trajs.items():
        if mkt == "spreads" and side == "home" and len(snaps) >= 2:
            spread_trajs[fix_id] = snaps

    print(f"\n  MLB spread trajectories (home side, 2+ snapshots): {len(spread_trajs)}")

    if not spread_trajs:
        print("  NO TRAJECTORY DATA — cannot analyze")
        return {}

    # Compute movement metrics per fixture
    movements = []
    for fix_id, snaps in spread_trajs.items():
        first_odds = snaps[0][0]
        last_odds = snaps[-1][0]
        delta = last_odds - first_odds  # positive = home odds drifted up (less favored)
        n_snaps = len(snaps)

        # When did most movement happen? Parse timestamps
        try:
            times = [datetime.strptime(s[1][:19], "%Y-%m-%d %H:%M:%S") for s in snaps]
            hours = [t.hour for t in times]
            span_hours = (times[-1] - times[0]).total_seconds() / 3600
        except Exception:
            hours = []
            span_hours = 0

        # Intra-trajectory volatility
        odds_arr = [s[0] for s in snaps]
        vol = np.std(odds_arr) if len(odds_arr) > 1 else 0

        movements.append({
            "fix_id": fix_id,
            "first": first_odds,
            "last": last_odds,
            "delta": delta,
            "abs_delta": abs(delta),
            "n_snaps": n_snaps,
            "vol": vol,
            "span_hours": span_hours,
            "hours": hours,
        })

    deltas = [m["delta"] for m in movements]
    abs_deltas = [m["abs_delta"] for m in movements]

    print(f"  Fixtures with movement data: {len(movements)}")
    print(f"  Odds delta (last - first):")
    print(f"    mean:  {np.mean(deltas):+.4f}")
    print(f"    std:   {np.std(deltas):.4f}")
    print(f"    moved (|Δ|>0.02): {sum(1 for d in abs_deltas if d > 0.02)}/{len(movements)} "
          f"({sum(1 for d in abs_deltas if d > 0.02)/len(movements)*100:.1f}%)")
    print(f"    big move (|Δ|>0.10): {sum(1 for d in abs_deltas if d > 0.10)}/{len(movements)} "
          f"({sum(1 for d in abs_deltas if d > 0.10)/len(movements)*100:.1f}%)")

    # Time-of-day analysis: when do lines move most?
    all_hours = []
    for m in movements:
        all_hours.extend(m["hours"])

    if all_hours:
        print(f"\n  Time-of-day snapshot distribution:")
        hour_counts = defaultdict(int)
        for h in all_hours:
            hour_counts[h] += 1
        for h in sorted(hour_counts.keys()):
            bar = "█" * (hour_counts[h] // 20)
            print(f"    {h:02d}:00  {hour_counts[h]:>5}  {bar}")

    return {m["fix_id"]: m for m in movements}


def analyze_movement_predicts_outcome(movements: dict):
    """Part 2: Does line movement predict who covers? (SELECTION-FREE)"""
    print(f"\n{'='*70}")
    print("PART 2: DOES LINE MOVEMENT PREDICT OUTCOMES? (SELECTION-FREE)")
    print("="*70)

    outcomes = load_settled_outcomes()

    # Match movements to outcomes
    matched = []
    for fix_id, m in movements.items():
        if fix_id in outcomes:
            o = outcomes[fix_id]
            matched.append({**m, **o})

    print(f"\n  Fixtures with movement + outcome: {len(matched)}")

    if len(matched) < 30:
        print("  INSUFFICIENT DATA for meaningful analysis")
        return

    # SELECTION-FREE test: does delta predict outcome?
    # Positive delta = home odds went UP (home became less favored / away became more favored)
    # If market is efficient, movement should predict nothing after close
    # If movement overshoots, betting AGAINST the movement predicts outcomes

    # Split by movement direction
    moved_up = [m for m in matched if m["delta"] > 0.02]    # home odds went up
    moved_down = [m for m in matched if m["delta"] < -0.02]  # home odds went down
    stable = [m for m in matched if abs(m["delta"]) <= 0.02]

    print(f"\n  Movement groups (|Δ| > 0.02 threshold):")
    print(f"    Moved up (home less favored):   N={len(moved_up)}")
    print(f"    Moved down (home more favored): N={len(moved_down)}")
    print(f"    Stable:                         N={len(stable)}")

    # For each group, what's the cover rate?
    for label, group in [("Moved up", moved_up), ("Moved down", moved_down), ("Stable", stable)]:
        if len(group) < 5:
            continue
        wr = np.mean([g["won"] for g in group])
        n = len(group)
        se = np.sqrt(wr * (1 - wr) / n)
        print(f"    {label:<30}: WR={wr:.3f} ± {se:.3f} (N={n})")

    # Continuous test: correlation between delta and outcome
    deltas = np.array([m["delta"] for m in matched])
    outcomes_arr = np.array([int(m["won"]) for m in matched])
    corr = np.corrcoef(deltas, outcomes_arr)[0, 1]
    print(f"\n  r(odds_delta, outcome) = {corr:+.4f}")

    # Quintile analysis: sort by delta magnitude, check WR by quintile
    if len(matched) >= 25:
        sorted_by_delta = sorted(matched, key=lambda m: m["delta"])
        q_size = len(sorted_by_delta) // 5
        print(f"\n  ── Quintile analysis (sorted by odds delta) ──")
        print(f"  {'Quintile':<12} {'Avg Δ':<10} {'WR':<8} {'N':<6} {'Direction'}")
        for i in range(5):
            start = i * q_size
            end = start + q_size if i < 4 else len(sorted_by_delta)
            q = sorted_by_delta[start:end]
            avg_d = np.mean([m["delta"] for m in q])
            wr = np.mean([m["won"] for m in q])
            direction = "home fav↑" if avg_d < 0 else "away fav↑"
            print(f"  Q{i+1:<11} {avg_d:+.4f}    {wr:.3f}  {len(q):<6} {direction}")

    # Reverse-line-movement proxy: big odds move + outcome opposite to movement
    # If odds went UP (home less favored) but home WINS → RLM signal
    if len(moved_up) >= 10 and len(moved_down) >= 10:
        print(f"\n  ── Reverse-line-movement test ──")
        print(f"  If odds move AWAY from a team, does that team still cover?")
        # moved_up = home became underdog / less favored → if home covers, that's RLM
        rlm_home = np.mean([m["won"] for m in moved_up])  # our bets may not be home-specific
        rlm_away = np.mean([m["won"] for m in moved_down])
        print(f"    Home odds↑ (less favored): WR={rlm_home:.3f} (N={len(moved_up)})")
        print(f"    Home odds↓ (more favored): WR={rlm_away:.3f} (N={len(moved_down)})")
        print(f"    If RLM works, home-odds↑ should have LOWER WR (market was right to move)")
        print(f"    If RLM is exploitable, home-odds↑ should have HIGHER WR (overreaction)")

    # H2H moneyline movement → spread outcomes correlation
    print(f"\n  ── Moneyline movement → spread outcome ──")
    trajs = load_line_trajectories()
    h2h_moves = {}
    for (fix_id, mkt, side, point), snaps in trajs.items():
        if mkt == "h2h" and side == "home" and len(snaps) >= 2:
            first = snaps[0][0]
            last = snaps[-1][0]
            h2h_moves[fix_id] = last - first

    h2h_matched = []
    for fix_id, delta in h2h_moves.items():
        if fix_id in outcomes:
            h2h_matched.append({"delta": delta, "won": outcomes[fix_id]["won"]})

    if len(h2h_matched) >= 30:
        h2h_deltas = np.array([m["delta"] for m in h2h_matched])
        h2h_outcomes = np.array([int(m["won"]) for m in h2h_matched])
        h2h_corr = np.corrcoef(h2h_deltas, h2h_outcomes)[0, 1]
        print(f"  r(h2h_delta, spread_outcome) = {h2h_corr:+.4f} (N={len(h2h_matched)})")

        big_h2h = [m for m in h2h_matched if abs(m["delta"]) > 0.05]
        if big_h2h:
            print(f"  Big h2h moves (|Δ|>0.05): N={len(big_h2h)}, "
                  f"WR={np.mean([m['won'] for m in big_h2h]):.3f}")


def main():
    movements = analyze_movement_structure()
    if movements:
        analyze_movement_predicts_outcome(movements)

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print("="*70)
    print("""
  The question was: is line movement a SELECTION-FREE signal?

  - If r(delta, outcome) ≈ 0: movement is noise / already priced at close.
    → Info timing is dead. The close is efficient.

  - If r(delta, outcome) > 0: movement direction predicts outcomes.
    → Market overreacts early; betting against movement works.
    → This is exploitable and selection-free.

  - If r(delta, outcome) < 0: movement is informative.
    → Market corrects toward truth; betting WITH movement works.
    → Line-shopping (get best early number) is the play.
    """)


if __name__ == "__main__":
    main()
