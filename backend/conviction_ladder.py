"""
conviction_ladder.py — Phase E continued: dist_wp conviction ladder & -1.5 analysis.

PIECE 1: Volume frontier (dist_wp threshold 0.85/0.88/0.90 stability)
PIECE 2: Full dist_wp × rung EV map (the conviction ladder)
PIECE 3: Does the top slice reach -1.5 already?
PIECE 4: Right-side-wrong-rung decomposition of losses

Uses main shadow_negbin arm (has ats_dist_wp + resolved scores).
De-duplicates by (fixture_id, rung) to avoid counting same game-side multiple times.
"""
from __future__ import annotations
import sqlite3, os, math
import numpy as np

DB_BETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bets.db")
DB_HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "historical.db")

# Breakeven at each rung (from actual alt_lines avg decimal odds, trimmed)
RUNG_ODDS = {
    -4.5: 5.621, -3.5: 4.294, -2.5: 3.651, -1.5: 2.929,
    1.5: 1.507, 2.5: 1.372, 3.5: 1.293, 4.5: 1.203,
}
RUNG_BREAKEVEN = {r: 1.0 / o for r, o in RUNG_ODDS.items()}

ALL_RUNGS = [-1.5, 1.5, 2.5, 3.5, 4.5]


def prop_noise_floor(p, n):
    return 2.0 * math.sqrt(p * (1.0 - p) / n) if n > 0 else float('inf')


def covers_rung(margin, rung):
    """Does a team with this margin cover this rung?
    margin = team_score - opponent_score (from team perspective).
    Rung +4.5: team gets 4.5 → covers if margin > -4.5
    Rung -1.5: team gives 1.5 → covers if margin > 1.5
    """
    return margin > -rung


def parse_rung(description):
    """Extract the spread rung from leg description like 'Team Name +4.5'."""
    if not description:
        return None
    for r in ['+4.5', '+3.5', '+2.5', '+1.5', '-1.5', '-2.5', '-3.5', '-4.5']:
        if r in description:
            return float(r)
    return None


def load_legs(conn):
    """Load de-duplicated settled legs from shadow_negbin with dist_wp and scores."""
    rows = conn.execute("""
        SELECT
            l.fixture_id,
            l.ats_dist_wp,
            l.negbin_wp,
            l.resolved_margin,
            l.negbin_cell,
            l.description,
            l.leg_result,
            b.game_date
        FROM mock_bet_legs l
        JOIN mock_bets b ON b.id = l.mock_bet_id
        WHERE b.source = 'shadow_negbin'
          AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND l.leg_result IN ('WIN','LOSS')
          AND l.ats_dist_wp IS NOT NULL
          AND l.resolved_margin IS NOT NULL
        ORDER BY b.game_date, l.fixture_id
    """).fetchall()

    # De-duplicate by (fixture_id, rung)
    seen = set()
    legs = []
    for r in rows:
        rung = parse_rung(r[5])
        key = (r[0], rung)
        if key in seen:
            continue
        seen.add(key)
        legs.append({
            'fixture_id': r[0], 'dist_wp': r[1], 'negbin_wp': r[2],
            'margin': r[3], 'cell': r[4], 'description': r[5],
            'result': r[6], 'game_date': r[7], 'rung': rung,
        })
    return legs


def split_half(legs):
    dates = sorted(set(l['game_date'] for l in legs))
    mid = len(dates) // 2
    h1_dates = set(dates[:mid])
    return ([l for l in legs if l['game_date'] in h1_dates],
            [l for l in legs if l['game_date'] not in h1_dates])


def piece_1_volume_frontier(legs):
    """Test dist_wp thresholds 0.85/0.88/0.90 for split-half stability."""
    print("\n" + "=" * 105)
    print("  PIECE 1: VOLUME FRONTIER — dist_wp threshold stability")
    print("=" * 105)

    # De-dup by fixture (across rungs) for unique game-sides
    fixture_best = {}  # fixture_id → leg with highest dist_wp
    for l in legs:
        fid = l['fixture_id']
        if fid not in fixture_best or l['dist_wp'] > fixture_best[fid]['dist_wp']:
            fixture_best[fid] = l

    unique_legs = list(fixture_best.values())
    print(f"\n  Unique game-sides: {len(unique_legs)}")
    dates = sorted(set(l['game_date'] for l in unique_legs))
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    thresholds = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
    print(f"\n  {'Threshold':<12} {'N':>5} {'Legs/day':>8} {'WR':>6} "
          f"{'Implied':>8} {'Edge':>8} {'Floor':>8} {'Sig':>5} "
          f"{'H1_WR':>6} {'H2_WR':>6} {'H1_N':>5} {'H2_N':>5} {'Stable':>7}")
    print(f"  {'─'*12} {'─'*5} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*5} "
          f"{'─'*6} {'─'*6} {'─'*5} {'─'*5} {'─'*7}")

    for thresh in thresholds:
        subset = [l for l in unique_legs if l['dist_wp'] >= thresh]
        if len(subset) < 15:
            print(f"  dist_wp≥{thresh:.2f}  n={len(subset):3d} (skip)")
            continue

        n = len(subset)
        wins = sum(1 for l in subset if l['result'] == 'WIN')
        wr = wins / n
        # For implied, use the rung-specific breakeven for each leg's actual rung
        implied = np.mean([l['dist_wp'] for l in subset])  # dist_wp IS the model's cover prob
        edge = (wr - implied) * 100
        nf = prop_noise_floor(wr, n) * 100
        sig = abs(edge) > nf

        days_in = len(set(l['game_date'] for l in subset))
        lpd = n / days_in if days_in > 0 else 0

        h1, h2 = split_half(subset)
        if h1 and h2 and len(h1) >= 5 and len(h2) >= 5:
            w1 = sum(1 for l in h1 if l['result'] == 'WIN') / len(h1)
            w2 = sum(1 for l in h2 if l['result'] == 'WIN') / len(h2)
            imp1 = np.mean([l['dist_wp'] for l in h1])
            imp2 = np.mean([l['dist_wp'] for l in h2])
            stable = (w1 > imp1) and (w2 > imp2)
        else:
            w1 = w2 = 0
            stable = False

        print(f"  dist_wp≥{thresh:.2f}  {n:5d} {lpd:7.1f} {wr:5.1%} "
              f"{implied:7.1%} {edge:>+7.1f}pp {nf:>7.1f}pp {'YES' if sig else ' no':>5} "
              f"{w1:5.1%} {w2:5.1%} {len(h1):5d} {len(h2):5d} {'YES' if stable else ' no':>7}")


def piece_2_conviction_ladder(legs):
    """Full dist_wp × rung EV map."""
    print("\n" + "=" * 105)
    print("  PIECE 2: CONVICTION LADDER — dist_wp × rung realized EV map")
    print("=" * 105)
    print("  For each dist_wp band × each rung: realized cover rate vs rung breakeven")
    print("  Positive margin = +EV at that rung. This IS the strategy map.\n")

    # Group legs by their actual rung (as bet)
    # For each, also compute counterfactual cover at ALL rungs
    dist_bands = [(0.40, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.85),
                  (0.85, 0.90), (0.90, 1.01)]
    band_labels = ["40-60%", "60-70%", "70-80%", "80-85%", "85-90%", "90%+"]

    # De-dup by fixture for counterfactual analysis
    # For each unique fixture, take the one entry (any rung) to get the margin
    fixture_data = {}
    for l in legs:
        fid = l['fixture_id']
        if fid not in fixture_data:
            fixture_data[fid] = {
                'margin': l['margin'], 'game_date': l['game_date'],
                'cell': l['cell'],
            }
        # Collect dist_wp by rung for this fixture
        if 'dist_wp_by_rung' not in fixture_data[fid]:
            fixture_data[fid]['dist_wp_by_rung'] = {}
        if l['rung'] is not None:
            fixture_data[fid]['dist_wp_by_rung'][l['rung']] = l['dist_wp']

    print(f"  Unique fixtures with scores: {len(fixture_data)}")

    # For each fixture, compute the MAX dist_wp (best rung's confidence)
    for fid, fd in fixture_data.items():
        if fd['dist_wp_by_rung']:
            fd['max_dist_wp'] = max(fd['dist_wp_by_rung'].values())
        else:
            fd['max_dist_wp'] = 0

    # THE MAP: for each dist_wp band of the fixture's MAX dist_wp,
    # show realized cover rate at EVERY rung
    print(f"\n  {'dist_wp band':<12} {'N':>5}  ", end="")
    for rung in ALL_RUNGS:
        r_label = f"{rung:+.1f}"
        print(f"{'WR':>5} {'BE':>5} {'Margin':>7}  ", end="")
    print(f"  {'Best rung':<12}")

    print(f"  {'─'*12} {'─'*5}  ", end="")
    for _ in ALL_RUNGS:
        print(f"{'─'*5} {'─'*5} {'─'*7}  ", end="")
    print(f"  {'─'*12}")

    # Header with rung names
    print(f"  {'':12} {'':5}  ", end="")
    for rung in ALL_RUNGS:
        print(f"  {rung:+.1f}             ", end="")
    print()

    for band_idx, (lo, hi) in enumerate(dist_bands):
        band_fixtures = {fid: fd for fid, fd in fixture_data.items()
                        if lo <= fd['max_dist_wp'] < hi}
        n = len(band_fixtures)
        if n < 10:
            print(f"  {band_labels[band_idx]:<12} {n:5d}  (skip)")
            continue

        print(f"  {band_labels[band_idx]:<12} {n:5d}  ", end="")

        best_margin = -999
        best_rung = None
        for rung in ALL_RUNGS:
            be = RUNG_BREAKEVEN.get(rung, 0.5)
            covers = sum(1 for fd in band_fixtures.values() if covers_rung(fd['margin'], rung))
            wr = covers / n
            margin_pp = (wr - be) * 100
            print(f"{wr:4.0%} {be:4.0%} {margin_pp:>+6.1f}pp  ", end="")
            if margin_pp > best_margin:
                best_margin = margin_pp
                best_rung = rung
        print(f"  {best_rung:+.1f} ({best_margin:+.1f}pp)")

    # Split-half stability for the best cells
    print(f"\n  --- Split-half stability for promising cells ---")
    for band_idx, (lo, hi) in enumerate(dist_bands):
        band_fixtures = {fid: fd for fid, fd in fixture_data.items()
                        if lo <= fd['max_dist_wp'] < hi}
        n = len(band_fixtures)
        if n < 20:
            continue

        for rung in ALL_RUNGS:
            be = RUNG_BREAKEVEN.get(rung, 0.5)
            covers = sum(1 for fd in band_fixtures.values() if covers_rung(fd['margin'], rung))
            wr = covers / n
            margin_pp = (wr - be) * 100
            nf = prop_noise_floor(wr, n) * 100

            if margin_pp > 0 and abs(margin_pp) > nf:
                # Check split-half
                all_legs = [{'game_date': fd['game_date'],
                             'covers': covers_rung(fd['margin'], rung)}
                            for fd in band_fixtures.values()]
                dates = sorted(set(l['game_date'] for l in all_legs))
                mid = len(dates) // 2
                h1_dates = set(dates[:mid])
                h1 = [l for l in all_legs if l['game_date'] in h1_dates]
                h2 = [l for l in all_legs if l['game_date'] not in h1_dates]

                if h1 and h2 and len(h1) >= 5 and len(h2) >= 5:
                    wr1 = sum(1 for l in h1 if l['covers']) / len(h1)
                    wr2 = sum(1 for l in h2 if l['covers']) / len(h2)
                    stable = (wr1 > be) and (wr2 > be)
                    flag = "STABLE" if stable else "unstable"
                    print(f"    {band_labels[band_idx]:>8} × {rung:+.1f}: "
                          f"WR={wr:.1%} vs BE={be:.1%} = {margin_pp:+.1f}pp (floor={nf:.1f}pp) "
                          f"H1={wr1:.1%}({len(h1)}) H2={wr2:.1%}({len(h2)}) → {flag}")

    # EV map: compute $ EV per $10 bet at each cell
    print(f"\n  --- EV per $10 bet at each cell ---")
    print(f"  {'dist_wp band':<12} {'N':>5}  ", end="")
    for rung in ALL_RUNGS:
        print(f"  {rung:+.1f}     ", end="")
    print()
    print(f"  {'─'*12} {'─'*5}  ", end="")
    for _ in ALL_RUNGS:
        print(f"{'─'*10} ", end="")
    print()

    for band_idx, (lo, hi) in enumerate(dist_bands):
        band_fixtures = {fid: fd for fid, fd in fixture_data.items()
                        if lo <= fd['max_dist_wp'] < hi}
        n = len(band_fixtures)
        if n < 10:
            continue

        print(f"  {band_labels[band_idx]:<12} {n:5d}  ", end="")
        for rung in ALL_RUNGS:
            odds = RUNG_ODDS.get(rung, 2.0)
            covers = sum(1 for fd in band_fixtures.values() if covers_rung(fd['margin'], rung))
            wr = covers / n
            ev_per_10 = (wr * odds - 1) * 10
            color = "+" if ev_per_10 > 0 else ""
            print(f"  ${ev_per_10:>+6.2f}   ", end="")
        print()


def piece_3_neg15_analysis(legs):
    """Does the top conviction slice already reach -1.5?"""
    print("\n" + "=" * 105)
    print("  PIECE 3: DOES THE TOP SLICE REACH -1.5?")
    print("=" * 105)

    # De-dup by fixture
    fixture_data = {}
    for l in legs:
        fid = l['fixture_id']
        if fid not in fixture_data:
            fixture_data[fid] = {
                'margin': l['margin'], 'game_date': l['game_date'],
                'cell': l['cell'], 'description': l['description'],
            }
        if 'dist_wp_by_rung' not in fixture_data[fid]:
            fixture_data[fid]['dist_wp_by_rung'] = {}
        if l['rung'] is not None:
            fixture_data[fid]['dist_wp_by_rung'][l['rung']] = l['dist_wp']
        fixture_data[fid]['max_dist_wp'] = max(
            fixture_data[fid].get('max_dist_wp', 0), l['dist_wp'])

    be_neg15 = RUNG_BREAKEVEN[-1.5]  # ~34.1%
    odds_neg15 = RUNG_ODDS[-1.5]  # ~2.929

    print(f"\n  -1.5 rung: team must win by 2+ (margin > 1.5)")
    print(f"  Breakeven: {be_neg15:.1%} (at avg odds {odds_neg15:.2f})")

    thresholds = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    print(f"\n  {'dist_wp≥':<12} {'N':>5} {'Win by 2+':>10} {'WR_-1.5':>8} "
          f"{'BE':>6} {'Margin':>8} {'Floor':>8} {'Sig':>5} "
          f"{'EV/$10':>8} {'H1':>6} {'H2':>6} {'Stable':>7}")
    print(f"  {'─'*12} {'─'*5} {'─'*10} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*5} "
          f"{'─'*8} {'─'*6} {'─'*6} {'─'*7}")

    for thresh in thresholds:
        fixtures = {fid: fd for fid, fd in fixture_data.items()
                   if fd['max_dist_wp'] >= thresh}
        n = len(fixtures)
        if n < 10:
            print(f"  ≥{thresh:.2f}       {n:5d} (skip)")
            continue

        win_by_2 = sum(1 for fd in fixtures.values() if fd['margin'] > 1.5)
        wr = win_by_2 / n
        margin = (wr - be_neg15) * 100
        nf = prop_noise_floor(wr, n) * 100
        sig = margin > 0 and abs(margin) > nf
        ev = (wr * odds_neg15 - 1) * 10

        # Split-half
        all_items = [{'game_date': fd['game_date'], 'win2': fd['margin'] > 1.5}
                    for fd in fixtures.values()]
        dates = sorted(set(i['game_date'] for i in all_items))
        mid = len(dates) // 2
        h1_dates = set(dates[:mid])
        h1 = [i for i in all_items if i['game_date'] in h1_dates]
        h2 = [i for i in all_items if i['game_date'] not in h1_dates]

        if h1 and h2 and len(h1) >= 5 and len(h2) >= 5:
            wr1 = sum(1 for i in h1 if i['win2']) / len(h1)
            wr2 = sum(1 for i in h2 if i['win2']) / len(h2)
            stable = (wr1 > be_neg15) and (wr2 > be_neg15)
        else:
            wr1 = wr2 = 0
            stable = False

        print(f"  ≥{thresh:.2f}       {n:5d} {win_by_2:>10} {wr:>7.1%} "
              f"{be_neg15:>5.1%} {margin:>+7.1f}pp {nf:>7.1f}pp {'YES' if sig else ' no':>5} "
              f"${ev:>+7.2f} {wr1:>5.1%} {wr2:>5.1%} {'YES' if stable else ' no':>7}")

    # Detailed look at top slice
    print(f"\n  --- Detailed: dist_wp ≥ 0.90 fixtures ---")
    top = {fid: fd for fid, fd in fixture_data.items() if fd['max_dist_wp'] >= 0.90}
    print(f"  N = {len(top)} unique game-sides")

    # Margin distribution
    margins = sorted([fd['margin'] for fd in top.values()])
    print(f"  Margin distribution (team perspective):")
    print(f"    min={min(margins):.0f}  P25={np.percentile(margins,25):.0f}  "
          f"median={np.median(margins):.0f}  P75={np.percentile(margins,75):.0f}  "
          f"max={max(margins):.0f}")

    # By margin bucket
    print(f"\n  Margin histogram:")
    for lo_m in range(-8, 12, 2):
        hi_m = lo_m + 2
        count = sum(1 for m in margins if lo_m <= m < hi_m)
        bar = "█" * count
        covers_neg15 = "← covers -1.5" if lo_m >= 2 else ""
        print(f"    [{lo_m:+3d},{hi_m:+3d}): {count:3d} {bar} {covers_neg15}")

    # What if we looked at the RUNG-SPECIFIC dist_wp at +1.5 for these fixtures?
    # (i.e., not their max dist_wp, but their dist_wp at the closest-to--1.5 rung)
    print(f"\n  --- These fixtures' dist_wp at each rung ---")
    for rung in [4.5, 3.5, 2.5, 1.5]:
        has_rung = [fd for fd in top.values() if rung in fd.get('dist_wp_by_rung', {})]
        if has_rung:
            avg_dwp = np.mean([fd['dist_wp_by_rung'][rung] for fd in has_rung])
            print(f"    Rung {rung:+.1f}: {len(has_rung)} fixtures, avg dist_wp = {avg_dwp:.3f}")


def piece_4_loss_decomposition(legs):
    """Right-side-wrong-rung decomposition of losses."""
    print("\n" + "=" * 105)
    print("  PIECE 4: LOSS DECOMPOSITION — Right-side-wrong-rung vs Wrong-side")
    print("=" * 105)
    print("  For LOSING legs: was the team pick correct (just wrong rung) or wrong entirely?")
    print("  'Right side, rung too thin': same team would've won at a safer rung")
    print("  'Wrong side entirely': team lost even at +4.5 (discrimination miss)")

    # De-dup by fixture, keeping rung info
    fixture_data = {}
    for l in legs:
        fid = l['fixture_id']
        if fid not in fixture_data:
            fixture_data[fid] = {
                'margin': l['margin'], 'game_date': l['game_date'],
                'cell': l['cell'], 'description': l['description'],
            }
        if 'legs_by_rung' not in fixture_data[fid]:
            fixture_data[fid]['legs_by_rung'] = {}
        if l['rung'] is not None:
            fixture_data[fid]['legs_by_rung'][l['rung']] = l
        fixture_data[fid]['max_dist_wp'] = max(
            fixture_data[fid].get('max_dist_wp', 0), l['dist_wp'])

    # Analyze by rung: for each leg that LOST at its bet rung
    print(f"\n  --- By actual bet rung ---")
    for rung in [1.5, 2.5, 3.5, 4.5]:
        rung_legs = [(fid, fd) for fid, fd in fixture_data.items()
                     if rung in fd.get('legs_by_rung', {})
                     and fd['legs_by_rung'][rung]['result'] == 'LOSS']
        if not rung_legs:
            continue

        n_loss = len(rung_legs)
        # Categories:
        # 1. Wrong side entirely: margin < -4.5 (lost by 5+, wouldn't cover even +4.5)
        wrong_side = sum(1 for _, fd in rung_legs if fd['margin'] <= -4.5)
        # 2. Right side, rung too thin: would have covered at a safer rung
        #    margin is between -rung and -4.5 exclusive
        #    i.e., didn't cover this rung but would cover +4.5
        right_side_wrong_rung = sum(1 for _, fd in rung_legs
                                    if fd['margin'] > -4.5 and not covers_rung(fd['margin'], rung))
        # 3. Edge case: margin exactly at boundary
        other = n_loss - wrong_side - right_side_wrong_rung

        print(f"\n    Rung +{rung}: {n_loss} losses")
        print(f"      Wrong side entirely (margin ≤ -4.5):     {wrong_side:4d} ({wrong_side/n_loss:.0%})")
        print(f"      Right side, rung too thin:                {right_side_wrong_rung:4d} ({right_side_wrong_rung/n_loss:.0%})")
        if other:
            print(f"      Other (boundary):                         {other:4d} ({other/n_loss:.0%})")

        # For the "right side, wrong rung" losses: what rung would they have needed?
        if right_side_wrong_rung > 0:
            needed_rungs = []
            for _, fd in rung_legs:
                if fd['margin'] > -4.5 and not covers_rung(fd['margin'], rung):
                    for safer in [4.5, 3.5, 2.5]:
                        if covers_rung(fd['margin'], safer):
                            needed_rungs.append(safer)
                            break
            if needed_rungs:
                from collections import Counter
                dist = Counter(needed_rungs)
                print(f"      Would have needed rung: {dict(sorted(dist.items()))}")

    # Overall decomposition across ALL rungs
    print(f"\n  --- OVERALL decomposition (all rungs combined) ---")
    all_losses = []
    for fid, fd in fixture_data.items():
        for rung, leg in fd.get('legs_by_rung', {}).items():
            if leg['result'] == 'LOSS':
                all_losses.append({
                    'fid': fid, 'rung': rung, 'margin': fd['margin'],
                    'dist_wp': leg['dist_wp'], 'game_date': fd['game_date'],
                })

    # De-dup losses by (fixture, rung)
    seen = set()
    deduped_losses = []
    for loss in all_losses:
        key = (loss['fid'], loss['rung'])
        if key not in seen:
            seen.add(key)
            deduped_losses.append(loss)

    n_total = len(deduped_losses)
    wrong_side = [l for l in deduped_losses if l['margin'] <= -4.5]
    right_side = [l for l in deduped_losses if l['margin'] > -4.5]

    print(f"  Total unique losses (fixture × rung): {n_total}")
    print(f"  Wrong side entirely (margin ≤ -4.5):  {len(wrong_side):4d} ({len(wrong_side)/n_total:.0%})")
    print(f"  Right side, rung too thin:             {len(right_side):4d} ({len(right_side)/n_total:.0%})")

    # For right-side losses: margin distribution
    if right_side:
        rs_margins = [l['margin'] for l in right_side]
        print(f"\n  Right-side losses margin distribution:")
        for m in range(-4, 1):
            count = sum(1 for x in rs_margins if m <= x < m + 1)
            if count > 0:
                print(f"    margin [{m:+d},{m+1:+d}): {count} losses (would need rung ≥ +{abs(m)+0.5:.1f})")

    # By dist_wp band
    print(f"\n  --- Loss type by dist_wp band ---")
    dist_bands = [(0.40, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    band_labels = ["40-60%", "60-70%", "70-80%", "80-90%", "90%+"]

    print(f"  {'dist_wp':<10} {'Losses':>6} {'Wrong side':>11} {'Right side':>11} {'WS%':>5}")
    print(f"  {'─'*10} {'─'*6} {'─'*11} {'─'*11} {'─'*5}")
    for i, (lo, hi) in enumerate(dist_bands):
        band_loss = [l for l in deduped_losses if lo <= l['dist_wp'] < hi]
        if not band_loss:
            continue
        ws = sum(1 for l in band_loss if l['margin'] <= -4.5)
        rs = len(band_loss) - ws
        print(f"  {band_labels[i]:<10} {len(band_loss):6d} {ws:11d} {rs:11d} {ws/len(band_loss):4.0%}")


def main():
    conn = sqlite3.connect(DB_BETS)
    legs = load_legs(conn)
    print(f"\n  Loaded {len(legs)} de-duplicated (fixture × rung) settled legs")
    print(f"  Unique fixtures: {len(set(l['fixture_id'] for l in legs))}")

    piece_1_volume_frontier(legs)
    piece_2_conviction_ladder(legs)
    piece_3_neg15_analysis(legs)
    piece_4_loss_decomposition(legs)

    conn.close()
    print(f"\n{'='*105}")
    print(f"  DONE")
    print(f"{'='*105}")


if __name__ == "__main__":
    main()
