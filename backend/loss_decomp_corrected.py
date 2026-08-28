"""
loss_decomp_corrected.py — Corrected loss decomposition under TRUE-winner definition.

The original Piece 4 used "+side right" = "would cover a safer + rung."
For -1.5, "right side" must mean "picked the actual winner."

At + rungs: ALL losses = picked the loser (margin < 0 by definition).
The "32% right-side-wrong-rung" was really "picked a team that lost by 1-4"
— NOT right-side under any winner-based definition.

This script:
  1. Side-by-side decomposition: +side def vs true-winner def
  2. Confirm -1.5 WR = P(team wins by 2+), not P(covers some rung)
  3. Reconcile pool size: fixture-level vs leg-level counting
"""
from __future__ import annotations
import sqlite3, os, math
import numpy as np

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bets.db")

RUNG_ODDS = {
    -1.5: 2.929, 1.5: 1.507, 2.5: 1.372, 3.5: 1.293, 4.5: 1.203,
}
RUNG_BREAKEVEN = {r: 1.0 / o for r, o in RUNG_ODDS.items()}


def parse_rung(desc):
    if not desc:
        return None
    for r in ['+4.5', '+3.5', '+2.5', '+1.5', '-1.5']:
        if r in desc:
            return float(r)
    return None


def load_deduped_legs(conn):
    """De-duplicated (fixture × rung) settled legs from shadow_negbin."""
    rows = conn.execute("""
        SELECT l.fixture_id, l.ats_dist_wp, l.resolved_margin,
               l.leg_result, l.description, l.negbin_cell, b.game_date
        FROM mock_bet_legs l
        JOIN mock_bets b ON b.id = l.mock_bet_id
        WHERE b.source = 'shadow_negbin'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL
          AND l.resolved_margin IS NOT NULL
        ORDER BY b.game_date, l.fixture_id
    """).fetchall()
    seen = set()
    legs = []
    for r in rows:
        rung = parse_rung(r[4])
        key = (r[0], rung)
        if key in seen:
            continue
        seen.add(key)
        legs.append({
            'fixture_id': r[0], 'dist_wp': r[1], 'margin': r[2],
            'result': r[3], 'description': r[4], 'cell': r[5],
            'game_date': r[6], 'rung': rung,
        })
    return legs


def section_1_side_by_side(legs):
    """Side-by-side loss decomposition: +side def vs true-winner def."""
    print("\n" + "=" * 105)
    print("  CORRECTED LOSS DECOMPOSITION — +side def vs TRUE-WINNER def")
    print("=" * 105)
    print("  '+side right': team lost by ≤4 (would cover a safer + rung)")
    print("  'TRUE right':  team WON the game (margin > 0)")
    print("  For + rungs these are NEVER the same — every + rung loss = team lost the game.\n")

    losses = [l for l in legs if l['result'] == 'LOSS']

    # By rung
    print(f"  {'Rung':<8} {'Losses':>6}  "
          f"{'|':>2} {'+Side Wrong':>12} {'+Side Right':>12}  "
          f"{'|':>2} {'TRUE Wrong':>11} {'TRUE Right':>11}")
    print(f"  {'':8} {'':6}  "
          f"{'|':>2} {'(lost by 5+)':>12} {'(lost by 1-4)':>12}  "
          f"{'|':>2} {'(team lost)':>11} {'(team won)':>11}")
    print(f"  {'─'*8} {'─'*6}  {'─'*2} {'─'*12} {'─'*12}  {'─'*2} {'─'*11} {'─'*11}")

    for rung in [1.5, 2.5, 3.5, 4.5]:
        rung_losses = [l for l in losses if l['rung'] == rung]
        if not rung_losses:
            continue
        n = len(rung_losses)

        # +side definition
        plus_wrong = sum(1 for l in rung_losses if l['margin'] <= -4.5)
        plus_right = n - plus_wrong  # lost by < 4.5 (would cover +4.5)

        # TRUE-winner definition
        true_wrong = sum(1 for l in rung_losses if l['margin'] <= 0)
        true_right = sum(1 for l in rung_losses if l['margin'] > 0)

        print(f"  +{rung:<7} {n:>6}  "
              f"| {plus_wrong:>6} ({plus_wrong/n:>4.0%}) {plus_right:>6} ({plus_right/n:>4.0%})  "
              f"| {true_wrong:>5} ({true_wrong/n:>4.0%}) {true_right:>5} ({true_right/n:>4.0%})")

    # Overall
    n_total = len(losses)
    plus_wrong_all = sum(1 for l in losses if l['margin'] <= -4.5)
    plus_right_all = n_total - plus_wrong_all
    true_wrong_all = sum(1 for l in losses if l['margin'] <= 0)
    true_right_all = sum(1 for l in losses if l['margin'] > 0)

    print(f"  {'─'*8} {'─'*6}  {'─'*2} {'─'*12} {'─'*12}  {'─'*2} {'─'*11} {'─'*11}")
    print(f"  {'ALL':<8} {n_total:>6}  "
          f"| {plus_wrong_all:>6} ({plus_wrong_all/n_total:>4.0%}) {plus_right_all:>6} ({plus_right_all/n_total:>4.0%})  "
          f"| {true_wrong_all:>5} ({true_wrong_all/n_total:>4.0%}) {true_right_all:>5} ({true_right_all/n_total:>4.0%})")

    print(f"\n  ORIGINAL CLAIM: 32% 'right-side-wrong-rung'")
    print(f"  CORRECTED:      {true_right_all/n_total:.0%} true right-side "
          f"({true_right_all}/{n_total})")
    print(f"  The '32%' was teams that LOST the game by 1-4 runs —")
    print(f"  repackaged as 'right side' under a + rung definition.")
    print(f"  Under true-winner: 100% of + rung losses = picked the loser.")

    # By dist_wp band — true-winner definition
    print(f"\n  --- By dist_wp band (TRUE-winner definition) ---")
    dist_bands = [(0.40, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    band_labels = ["40-60%", "60-70%", "70-80%", "80-90%", "90%+"]

    print(f"  {'dist_wp':<10} {'Losses':>6} {'Picked loser':>13} {'Picked winner':>14}")
    print(f"  {'─'*10} {'─'*6} {'─'*13} {'─'*14}")
    for i, (lo, hi) in enumerate(dist_bands):
        band = [l for l in losses if lo <= l['dist_wp'] < hi]
        if not band:
            continue
        n = len(band)
        loser = sum(1 for l in band if l['margin'] <= 0)
        winner = sum(1 for l in band if l['margin'] > 0)
        print(f"  {band_labels[i]:<10} {n:6d} {loser:>7} ({loser/n:>4.0%}) {winner:>8} ({winner/n:>4.0%})")

    # What about hypothetical -1.5 losses?
    print(f"\n  --- Hypothetical -1.5 loss decomposition ---")
    print(f"  If we BET -1.5 on every fixture, losses would be margin ≤ 1.5:")
    print(f"  'TRUE wrong side': team lost (margin ≤ 0)")
    print(f"  'Right side, not decisive enough': team won by 1 (0 < margin ≤ 1.5)")

    # Use ALL fixtures (wins + losses) to compute -1.5 hypothetical losses
    fixture_data = {}
    for l in legs:
        fid = l['fixture_id']
        if fid not in fixture_data:
            fixture_data[fid] = {'margin': l['margin'], 'game_date': l['game_date']}
        if 'max_dist_wp' not in fixture_data[fid]:
            fixture_data[fid]['max_dist_wp'] = l['dist_wp']
        fixture_data[fid]['max_dist_wp'] = max(fixture_data[fid]['max_dist_wp'], l['dist_wp'])

    for thresh_label, thresh in [("All fixtures", 0.0), ("dist_wp≥0.85", 0.85), ("dist_wp≥0.90", 0.90)]:
        fxs = {fid: fd for fid, fd in fixture_data.items() if fd['max_dist_wp'] >= thresh}
        n = len(fxs)
        if n < 10:
            continue

        neg15_wins = sum(1 for fd in fxs.values() if fd['margin'] > 1.5)     # covers -1.5
        neg15_losses = n - neg15_wins

        # Among -1.5 losses: team won by 1, drew, lost by 1-4, lost by 5+
        team_won_by_1 = sum(1 for fd in fxs.values() if 0 < fd['margin'] <= 1.5)
        team_drew = sum(1 for fd in fxs.values() if fd['margin'] == 0)
        team_lost_close = sum(1 for fd in fxs.values() if -4.5 < fd['margin'] < 0)
        team_lost_blowout = sum(1 for fd in fxs.values() if fd['margin'] <= -4.5)

        print(f"\n    {thresh_label} (N={n}):")
        print(f"      -1.5 WR: {neg15_wins}/{n} = {neg15_wins/n:.1%} "
              f"(BE={RUNG_BREAKEVEN[-1.5]:.1%}, margin={((neg15_wins/n)-RUNG_BREAKEVEN[-1.5])*100:+.1f}pp)")
        print(f"      -1.5 losses breakdown ({neg15_losses} losses):")
        if neg15_losses > 0:
            print(f"        Team won by exactly 1:  {team_won_by_1:4d} ({team_won_by_1/neg15_losses:>4.0%}) ← TRUE right side, not decisive")
            print(f"        Team drew:               {team_drew:4d} ({team_drew/neg15_losses:>4.0%})")
            print(f"        Team lost by 1-4:        {team_lost_close:4d} ({team_lost_close/neg15_losses:>4.0%}) ← wrong side")
            print(f"        Team lost by 5+:         {team_lost_blowout:4d} ({team_lost_blowout/neg15_losses:>4.0%}) ← wrong side (blowout)")
            true_right_neg15 = team_won_by_1
            true_wrong_neg15 = neg15_losses - team_won_by_1
            print(f"      TRUE right-side (team won but margin ≤ 1.5): {true_right_neg15/neg15_losses:.0%}")
            print(f"      TRUE wrong-side (team lost or drew):          {true_wrong_neg15/neg15_losses:.0%}")


def section_2_confirm_neg15(legs):
    """Confirm -1.5 WR = P(team wins by 2+), not P(covers some rung)."""
    print("\n" + "=" * 105)
    print("  CONFIRMING: -1.5 WR = P(team wins by 2+)")
    print("=" * 105)

    fixture_data = {}
    for l in legs:
        fid = l['fixture_id']
        if fid not in fixture_data:
            fixture_data[fid] = {'margin': l['margin'], 'game_date': l['game_date']}
        if 'max_dist_wp' not in fixture_data[fid]:
            fixture_data[fid]['max_dist_wp'] = l['dist_wp']
        fixture_data[fid]['max_dist_wp'] = max(fixture_data[fid]['max_dist_wp'], l['dist_wp'])

    print(f"\n  Definition: -1.5 cover = team margin > 1.5 = team wins by 2+")
    print(f"  This is EQUIVALENT to: team won AND won by ≥ 2 runs")
    print(f"  It is NOT: 'team covered some + rung' (which is a much weaker condition)")

    thresholds = [("All (≥0)", 0.0), ("≥0.80", 0.80), ("≥0.85", 0.85),
                  ("≥0.88", 0.88), ("≥0.90", 0.90)]

    be = RUNG_BREAKEVEN[-1.5]
    odds = RUNG_ODDS[-1.5]

    print(f"\n  {'Slice':<12} {'N':>5} "
          f"{'Team won':>9} {'Won by 2+':>10} {'-1.5 WR':>8} {'BE':>6} {'Margin':>8} {'EV/$10':>8}")
    print(f"  {'─'*12} {'─'*5} {'─'*9} {'─'*10} {'─'*8} {'─'*6} {'─'*8} {'─'*8}")

    for label, thresh in thresholds:
        fxs = {fid: fd for fid, fd in fixture_data.items() if fd['max_dist_wp'] >= thresh}
        n = len(fxs)
        if n < 10:
            continue

        team_won = sum(1 for fd in fxs.values() if fd['margin'] > 0)
        won_by_2 = sum(1 for fd in fxs.values() if fd['margin'] > 1.5)
        wr = won_by_2 / n
        margin = (wr - be) * 100
        ev = (wr * odds - 1) * 10

        print(f"  {label:<12} {n:>5} "
              f"{team_won:>5} ({team_won/n:>4.0%}) {won_by_2:>6} ({won_by_2/n:>4.0%}) "
              f"{wr:>7.1%} {be:>5.1%} {margin:>+7.1f}pp ${ev:>+7.2f}")

    # Break down the 85-90% band specifically (the best -1.5 cell)
    print(f"\n  --- 85-90% band detail (best -1.5 cell) ---")
    band_fxs = {fid: fd for fid, fd in fixture_data.items()
                if 0.85 <= fd['max_dist_wp'] < 0.90}
    n = len(band_fxs)
    if n >= 10:
        margins = sorted([fd['margin'] for fd in band_fxs.values()])

        print(f"  N = {n} fixtures")
        print(f"  Margin histogram (team perspective):")
        for lo in range(-12, 14, 2):
            hi = lo + 2
            count = sum(1 for m in margins if lo <= m < hi)
            if count > 0:
                bar = "█" * count
                label = ""
                if lo >= 2:
                    label = " ← wins by 2+ (covers -1.5)"
                elif lo >= 0:
                    label = " ← wins by 0-1 (covers +1.5 but NOT -1.5)"
                print(f"    [{lo:+3d},{hi:+3d}): {count:3d} {bar}{label}")

        won_game = sum(1 for m in margins if m > 0)
        won_by_2 = sum(1 for m in margins if m > 1.5)
        lost_game = sum(1 for m in margins if m <= 0)

        print(f"\n  Outcome breakdown:")
        print(f"    Team won the game:     {won_game:3d} ({won_game/n:.0%})")
        print(f"      Won by 2+ (-1.5):    {won_by_2:3d} ({won_by_2/n:.0%})")
        print(f"      Won by exactly 1:    {won_game - won_by_2:3d} ({(won_game-won_by_2)/n:.0%})")
        print(f"    Team lost/drew:        {lost_game:3d} ({lost_game/n:.0%})")

        # Split-half
        dates = sorted(set(fd['game_date'] for fd in band_fxs.values()))
        mid = len(dates) // 2
        h1_dates = set(dates[:mid])
        h1 = [fd for fd in band_fxs.values() if fd['game_date'] in h1_dates]
        h2 = [fd for fd in band_fxs.values() if fd['game_date'] not in h1_dates]

        if h1 and h2:
            wr1 = sum(1 for fd in h1 if fd['margin'] > 1.5) / len(h1)
            wr2 = sum(1 for fd in h2 if fd['margin'] > 1.5) / len(h2)
            print(f"\n  Split-half -1.5 WR:")
            print(f"    H1: {wr1:.1%} ({len(h1)} fixtures), H2: {wr2:.1%} ({len(h2)} fixtures)")
            print(f"    Both above BE ({be:.1%}): {'YES' if (wr1 > be and wr2 > be) else 'NO'}")


def section_3_pool_reconciliation(conn, legs):
    """Reconcile: fixture-level vs leg-level counting."""
    print("\n" + "=" * 105)
    print("  POOL SIZE RECONCILIATION — Fixture-level vs Leg-level")
    print("=" * 105)

    # Fixture-level: unique game-sides
    fixture_data = {}
    for l in legs:
        fid = l['fixture_id']
        if fid not in fixture_data:
            fixture_data[fid] = {'max_dist_wp': l['dist_wp']}
        fixture_data[fid]['max_dist_wp'] = max(fixture_data[fid]['max_dist_wp'], l['dist_wp'])

    total_fixtures = len(fixture_data)

    # Leg-level counting from EDGEGATE (where the "7%" number came from)
    edgegate_legs = conn.execute("""
        SELECT l.fixture_id, l.win_prob, l.leg_result
        FROM mock_bet_legs l
        JOIN mock_bets b ON b.id = l.mock_bet_id
        WHERE b.source = 'shadow_negbin_edgegate'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
    """).fetchall()

    total_eg_legs = len(edgegate_legs)
    eg_fixtures = len(set(r[0] for r in edgegate_legs))

    # The "7%" was: 122 legs out of 1801 edgegate legs had dist_wp >= 0.90
    # But those 122 legs came from how many unique fixtures?
    # Look up dist_wp for edgegate fixtures from the ungated arm
    dist_wp_map = {}
    for l in legs:
        if l['fixture_id'] not in dist_wp_map:
            dist_wp_map[l['fixture_id']] = {}
        if l['dist_wp'] is not None:
            dist_wp_map[l['fixture_id']][l.get('rung', 0)] = l['dist_wp']

    # For each edgegate fixture, what's its max dist_wp?
    eg_fixture_max_dwp = {}
    for r in edgegate_legs:
        fid = r[0]
        if fid in dist_wp_map and fid not in eg_fixture_max_dwp:
            eg_fixture_max_dwp[fid] = max(dist_wp_map[fid].values())

    print(f"\n  --- Counting units ---")
    print(f"  Shadow_negbin (ungated): {total_fixtures} unique fixtures, {len(legs)} (fixture×rung) legs")
    print(f"  Edgegate: {eg_fixtures} unique fixtures, {total_eg_legs} total legs (in parlays)")
    print(f"  Legs/fixture in edgegate: {total_eg_legs/eg_fixtures:.1f}x "
          f"(same fixture appears in ~{total_eg_legs/eg_fixtures:.0f} parlays)")

    print(f"\n  --- dist_wp threshold at FIXTURE level ---")
    for thresh in [0.80, 0.85, 0.88, 0.90, 0.92]:
        n_fix = sum(1 for fd in fixture_data.values() if fd['max_dist_wp'] >= thresh)
        pct = n_fix / total_fixtures * 100
        print(f"    dist_wp≥{thresh:.2f}: {n_fix}/{total_fixtures} fixtures ({pct:.0f}%)")

    print(f"\n  --- dist_wp threshold at EDGEGATE LEG level ---")
    # Each edgegate leg was matched to its fixture's dist_wp
    # A fixture's legs ALL get the same dist_wp (fixture-level property)
    for thresh in [0.80, 0.85, 0.88, 0.90, 0.92]:
        n_legs = sum(1 for r in edgegate_legs
                    if r[0] in eg_fixture_max_dwp
                    and eg_fixture_max_dwp[r[0]] >= thresh)
        pct = n_legs / total_eg_legs * 100
        print(f"    dist_wp≥{thresh:.2f}: {n_legs}/{total_eg_legs} legs ({pct:.0f}%)")

    print(f"\n  --- Reconciliation ---")
    n_90_fix = sum(1 for fd in fixture_data.values() if fd['max_dist_wp'] >= 0.90)
    n_85_fix = sum(1 for fd in fixture_data.values() if fd['max_dist_wp'] >= 0.85)

    n_90_legs = sum(1 for r in edgegate_legs
                   if r[0] in eg_fixture_max_dwp
                   and eg_fixture_max_dwp[r[0]] >= 0.90)

    print(f"  The '7% of legs' (from earlier): {n_90_legs} legs with dist_wp≥0.90 "
          f"out of {total_eg_legs} = {n_90_legs/total_eg_legs:.0%}")
    print(f"  Those legs come from {n_90_fix} unique fixtures out of {total_fixtures} "
          f"= {n_90_fix/total_fixtures:.0%} of fixtures")
    print(f"  The '60% of fixtures' (0.85+): {n_85_fix}/{total_fixtures} "
          f"= {n_85_fix/total_fixtures:.0%}")

    print(f"\n  WHY THEY DIFFER:")
    print(f"    - Each fixture appears in ~{total_eg_legs/eg_fixtures:.0f} parlay legs (multiple parlays reuse same fixture)")
    print(f"    - 'dist_wp' is the RUNG-SPECIFIC model confidence, not fixture-level")
    print(f"    - A fixture can have dist_wp=0.92 at +4.5 but dist_wp=0.65 at +1.5")
    print(f"    - '60% of fixtures' means 60% have ≥1 rung with dist_wp≥0.85")
    print(f"    - The ARM gates at the LEG level (all legs in a parlay must pass)")

    # What matters for the arm: how many LEGS per day pass?
    print(f"\n  --- What the arm actually filters ---")
    print(f"  The highconv arm gates at LEG level: every leg in a parlay must have dist_wp≥0.85.")
    print(f"  A 3-leg parlay passes only if ALL 3 legs are ≥0.85.")
    print(f"  Fixture-level 60% is an upper bound — actual pass rate depends on")
    print(f"  how many parlays have all legs from high-dist_wp fixtures.")

    # Check: from the shadow_negbin arm, how many parlays have ALL legs ≥ 0.85?
    parlay_data = conn.execute("""
        SELECT b.id, b.game_date,
               MIN(l.ats_dist_wp) as min_dist_wp,
               COUNT(l.id) as n_legs
        FROM mock_bets b
        JOIN mock_bet_legs l ON l.mock_bet_id = b.id
        WHERE b.source = 'shadow_negbin'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL
          AND b.cohort = 'clean'
        GROUP BY b.id
    """).fetchall()

    total_parlays = len(parlay_data)
    dates = sorted(set(r[1] for r in parlay_data))
    n_days = len(dates)

    for thresh in [0.80, 0.85, 0.88, 0.90]:
        passing = [r for r in parlay_data if r[2] is not None and r[2] >= thresh]
        n = len(passing)
        ppd = n / n_days if n_days > 0 else 0
        pass_legs = sum(r[3] for r in passing)
        print(f"    All-legs ≥{thresh:.2f}: {n}/{total_parlays} parlays "
              f"({n/total_parlays:.0%}), {ppd:.1f}/day, {pass_legs} total legs")


def main():
    conn = sqlite3.connect(DB)
    legs = load_deduped_legs(conn)
    print(f"\n  Loaded {len(legs)} de-duplicated (fixture×rung) legs, "
          f"{len(set(l['fixture_id'] for l in legs))} unique fixtures")

    section_1_side_by_side(legs)
    section_2_confirm_neg15(legs)
    section_3_pool_reconciliation(conn, legs)

    conn.close()
    print(f"\n{'='*105}")
    print(f"  DONE")
    print(f"{'='*105}")


if __name__ == "__main__":
    main()
