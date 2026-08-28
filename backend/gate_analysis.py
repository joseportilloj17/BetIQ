"""
Gate 1: CLV test — does NegBin high-conviction +1.5 beat the CLOSING LINE?
Gate 2: Mechanism — WHAT does NegBin high-conviction select on?

CORRECTED approach: CLV measured directly from alt_lines AM vs PM snapshots.
Back-derivation from negbin_wp-negbin_edge_pp is unreliable (mismatch detected).
Control group: all +1.5 lines AM→PM movement (not just NegBin-selected).
"""

import sqlite3
import re
import numpy as np
from collections import defaultdict


def _parse_spread_and_team(desc):
    m = re.match(r"(.+?)\s+([+-]\d+\.5)", desc)
    if not m:
        return None, None
    return m.group(1), float(m.group(2))


def run():
    bets_db = sqlite3.connect("data/bets.db")
    bets_db.row_factory = sqlite3.Row
    hist_db = sqlite3.connect("data/historical.db")
    hist_db.row_factory = sqlite3.Row

    # ═══════════════════════════════════════════════
    # STEP 0: Identify NegBin +1.5 game-teams (deduped)
    # ═══════════════════════════════════════════════
    raw = bets_db.execute("""
        SELECT mbl.description, mbl.fixture_id,
               mbl.negbin_wp, mbl.negbin_edge_pp,
               mbl.negbin_mu_home, mbl.negbin_mu_away, mbl.negbin_cell,
               mbl.leg_result,
               mbl.resolved_home_team, mbl.resolved_away_team,
               mbl.resolved_home_score, mbl.resolved_away_score,
               mbl.ridge_composite, mbl.ats_dist_wp,
               mb.game_date
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mb.source = 'shadow_negbin'
          AND mb.cohort = 'clean'
          AND mbl.leg_result IS NOT NULL
    """).fetchall()

    game_team_map = {}
    for r in raw:
        team, spread = _parse_spread_and_team(r["description"])
        if spread != 1.5 or team is None:
            continue
        key = (r["fixture_id"], team)
        if key in game_team_map:
            continue
        game_team_map[key] = dict(r) | {"team": team, "spread": spread}

    legs = list(game_team_map.values())
    print(f"NegBin shadow +1.5 game-teams (deduped): {len(legs)}")

    # ═══════════════════════════════════════════════
    # GATE 1: CLV from alt_lines AM vs PM (direct)
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"=== GATE 1: CLV TEST (alt_lines AM vs PM) ===")
    print(f"{'='*60}")

    # For each NegBin leg, get AM (earliest) and PM (latest) odds from alt_lines
    # Use pinnacle as the reference bookmaker (sharpest) for consistency
    # If no pinnacle, fall back to any bookmaker

    REFERENCE_BK = "pinnacle"  # sharpest book, least vig

    clv_data = []
    no_data = 0
    single_snap = 0

    for leg in legs:
        fixture_id = leg["fixture_id"]
        team = leg["team"]
        game_date = leg["game_date"]

        # Get ALL snapshots for this game-team at +1.5
        snaps = hist_db.execute("""
            SELECT odds, bookmaker, fetched_at FROM alt_lines
            WHERE event_id = ?
              AND team = ?
              AND ABS(line - 1.5) < 0.26
              AND market_key = 'alternate_spreads'
              AND odds IS NOT NULL AND odds > 1.0
            ORDER BY fetched_at
        """, (fixture_id, team)).fetchall()

        if not snaps:
            # Try with game_date_ct + team matching
            snaps = hist_db.execute("""
                SELECT odds, bookmaker, fetched_at FROM alt_lines
                WHERE game_date_ct = ?
                  AND team = ?
                  AND ABS(line - 1.5) < 0.26
                  AND market_key = 'alternate_spreads'
                  AND odds IS NOT NULL AND odds > 1.0
                ORDER BY fetched_at
            """, (game_date, team)).fetchall()

        if not snaps:
            no_data += 1
            continue

        # Group by bookmaker, get earliest and latest for each
        bk_snaps = defaultdict(list)
        for s in snaps:
            bk_snaps[s["bookmaker"]].append({"odds": s["odds"], "fetched_at": s["fetched_at"]})

        # Prefer pinnacle
        bk = REFERENCE_BK if REFERENCE_BK in bk_snaps else list(bk_snaps.keys())[0]
        bk_data = bk_snaps[bk]

        if len(bk_data) < 2:
            # Check if ANY bookmaker has 2+ snapshots
            found_multi = False
            for b, d in bk_snaps.items():
                if len(d) >= 2:
                    bk = b
                    bk_data = d
                    found_multi = True
                    break
            if not found_multi:
                single_snap += 1
                continue

        am_odds = bk_data[0]["odds"]  # earliest
        pm_odds = bk_data[-1]["odds"]  # latest
        am_fetched = bk_data[0]["fetched_at"]
        pm_fetched = bk_data[-1]["fetched_at"]

        am_implied = 1.0 / am_odds
        pm_implied = 1.0 / pm_odds

        clv_pp = (pm_implied - am_implied) * 100  # positive = line tightened
        won = 1 if leg["leg_result"] == "WIN" else 0

        clv_data.append({
            "team": team,
            "game_date": game_date,
            "am_odds": am_odds,
            "pm_odds": pm_odds,
            "am_implied": am_implied,
            "pm_implied": pm_implied,
            "am_fetched": am_fetched,
            "pm_fetched": pm_fetched,
            "clv_pp": clv_pp,
            "won": won,
            "bookmaker": bk,
            "fixture_id": fixture_id,
            **{k: leg[k] for k in ["negbin_wp", "negbin_edge_pp", "negbin_mu_home",
                                     "negbin_mu_away", "negbin_cell",
                                     "resolved_home_team", "resolved_away_team",
                                     "resolved_home_score", "resolved_away_score"]},
        })

    print(f"Games with AM+PM snapshots: {len(clv_data)}")
    print(f"Games with no alt_lines:    {no_data}")
    print(f"Games with single snapshot: {single_snap}")

    if not clv_data:
        print("NO DATA — cannot compute CLV.")
        return

    # ── NegBin CLV Results ──
    clv_values = [d["clv_pp"] for d in clv_data]
    wins = [d["won"] for d in clv_data]
    am_implieds = [d["am_implied"] for d in clv_data]
    pm_implieds = [d["pm_implied"] for d in clv_data]

    mean_clv = np.mean(clv_values)
    median_clv = np.median(clv_values)
    pct_positive = np.mean([1 if c > 0 else 0 for c in clv_values]) * 100
    std_clv = np.std(clv_values, ddof=1)
    se_clv = std_clv / np.sqrt(len(clv_values))
    z_clv = mean_clv / se_clv if se_clv > 0 else 0

    wr = np.mean(wins)
    avg_am = np.mean(am_implieds)
    avg_pm = np.mean(pm_implieds)

    print(f"\n--- NegBin +1.5 CLV (AM→PM, same bookmaker) ---")
    print(f"  Mean CLV:     {mean_clv:+.2f} pp")
    print(f"  Median CLV:   {median_clv:+.2f} pp")
    print(f"  CLV-positive: {pct_positive:.1f}%")
    print(f"  Std dev:      {std_clv:.2f} pp")
    print(f"  SE:           {se_clv:.2f} pp")
    print(f"  z-stat:       {z_clv:.2f}")
    print(f"  N:            {len(clv_values)}")
    print(f"\n  Avg AM odds: {np.mean([d['am_odds'] for d in clv_data]):.3f} (implied {avg_am*100:.1f}%)")
    print(f"  Avg PM odds: {np.mean([d['pm_odds'] for d in clv_data]):.3f} (implied {avg_pm*100:.1f}%)")
    print(f"\n  WR:           {wr*100:.1f}%")
    print(f"  AM BE:        {avg_am*100:.1f}%")
    print(f"  PM BE:        {avg_pm*100:.1f}%")
    print(f"  Edge vs AM:   {(wr - avg_am)*100:+.1f} pp")
    print(f"  Edge vs PM:   {(wr - avg_pm)*100:+.1f} pp")

    # ── Split-half ──
    np.random.seed(42)
    idx = np.random.permutation(len(clv_values))
    h = len(idx) // 2
    h1 = [clv_values[i] for i in idx[:h]]
    h2 = [clv_values[i] for i in idx[h:]]
    print(f"\n--- Split-Half CLV ---")
    print(f"  Half 1: {np.mean(h1):+.2f} pp (N={len(h1)})")
    print(f"  Half 2: {np.mean(h2):+.2f} pp (N={len(h2)})")

    # ── Temporal split ──
    by_date = defaultdict(list)
    for d in clv_data:
        by_date[d["game_date"]].append(d["clv_pp"])
    dates_sorted = sorted(by_date.keys())
    mid = len(dates_sorted) // 2
    early_clv = [v for dt in dates_sorted[:mid] for v in by_date[dt]]
    late_clv = [v for dt in dates_sorted[mid:] for v in by_date[dt]]
    if early_clv and late_clv:
        print(f"\n--- Temporal Split ---")
        print(f"  Early ({dates_sorted[0]} to {dates_sorted[mid-1]}): {np.mean(early_clv):+.2f} pp (N={len(early_clv)})")
        print(f"  Late  ({dates_sorted[mid]} to {dates_sorted[-1]}): {np.mean(late_clv):+.2f} pp (N={len(late_clv)})")

    # ── Distribution ──
    print(f"\n--- CLV Distribution ---")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct}: {np.percentile(clv_values, pct):+.2f} pp")

    # ── Example legs ──
    print(f"\n--- Example Legs (first 8 with AM≠PM) ---")
    for d in clv_data[:8]:
        print(f"  {d['game_date']} {d['team']}: AM={d['am_odds']:.3f} ({d['am_implied']*100:.1f}%) "
              f"→ PM={d['pm_odds']:.3f} ({d['pm_implied']*100:.1f}%) "
              f"CLV={d['clv_pp']:+.1f}pp {'W' if d['won'] else 'L'} [{d['bookmaker']}]")

    # ═══════════════════════════════════════════════
    # CONTROL GROUP: All +1.5 lines AM→PM movement
    # ═══════════════════════════════════════════════
    print(f"\n--- CONTROL: ALL +1.5 AM→PM Movement ---")

    # Get all game_date_ct values that NegBin has bets on
    negbin_dates = set(d["game_date"] for d in clv_data)

    control_clvs = []
    for game_date in sorted(negbin_dates):
        # Get all teams with +1.5 data on this date
        all_teams = hist_db.execute("""
            SELECT DISTINCT team, event_id FROM alt_lines
            WHERE game_date_ct = ?
              AND ABS(line - 1.5) < 0.26
              AND market_key = 'alternate_spreads'
              AND odds IS NOT NULL AND odds > 1.0
        """, (game_date,)).fetchall()

        for row in all_teams:
            t, eid = row["team"], row["event_id"]
            snaps = hist_db.execute("""
                SELECT odds, fetched_at FROM alt_lines
                WHERE event_id = ? AND team = ?
                  AND ABS(line - 1.5) < 0.26
                  AND market_key = 'alternate_spreads'
                  AND bookmaker = ?
                  AND odds IS NOT NULL AND odds > 1.0
                ORDER BY fetched_at
            """, (eid, t, REFERENCE_BK)).fetchall()

            if len(snaps) < 2:
                # Try any bookmaker
                snaps = hist_db.execute("""
                    SELECT odds, bookmaker, fetched_at FROM alt_lines
                    WHERE event_id = ? AND team = ?
                      AND ABS(line - 1.5) < 0.26
                      AND market_key = 'alternate_spreads'
                      AND odds IS NOT NULL AND odds > 1.0
                    ORDER BY fetched_at
                """, (eid, t)).fetchall()
                if len(snaps) < 2:
                    continue
                # Use first bookmaker that has 2+ snapshots
                bk_data = defaultdict(list)
                for s in snaps:
                    bk_data[s["bookmaker"]].append(s)
                found = False
                for b, d in bk_data.items():
                    if len(d) >= 2:
                        snaps = d
                        found = True
                        break
                if not found:
                    continue

            am = 1.0 / snaps[0]["odds"]
            pm = 1.0 / snaps[-1]["odds"]
            control_clvs.append((pm - am) * 100)

    if control_clvs:
        print(f"  All +1.5 AM→PM: N={len(control_clvs)}, mean={np.mean(control_clvs):+.2f} pp, "
              f"median={np.median(control_clvs):+.2f} pp")
        print(f"  NegBin +1.5:    N={len(clv_values)}, mean={mean_clv:+.2f} pp")
        diff = mean_clv - np.mean(control_clvs)
        print(f"  Excess CLV (NegBin - control): {diff:+.2f} pp")
        if diff > 0:
            # Is the excess significant?
            pooled_se = np.sqrt(np.var(clv_values, ddof=1)/len(clv_values) +
                                np.var(control_clvs, ddof=1)/len(control_clvs))
            z_excess = diff / pooled_se if pooled_se > 0 else 0
            print(f"  z-stat (excess): {z_excess:.2f}")

    # ═══════════════════════════════════════════════
    # GATE 2: MECHANISM
    # ═══════════════════════════════════════════════
    print(f"\n\n{'='*60}")
    print(f"=== GATE 2: MECHANISM ===")
    print(f"{'='*60}")

    # Use full +1.5 deduped set (all 331 game-teams)
    print(f"\nAnalyzing {len(legs)} +1.5 game-team legs")

    margins = []
    cells = defaultdict(list)
    for leg in legs:
        mu_h = leg["negbin_mu_home"]
        mu_a = leg["negbin_mu_away"]
        team = leg["team"]
        home = leg["resolved_home_team"]
        away = leg["resolved_away_team"]

        if team == home:
            pred_margin = mu_h - mu_a
            actual_margin = (leg["resolved_home_score"] or 0) - (leg["resolved_away_score"] or 0)
        elif team == away:
            pred_margin = mu_a - mu_h
            actual_margin = (leg["resolved_away_score"] or 0) - (leg["resolved_home_score"] or 0)
        else:
            continue

        won = 1 if leg["leg_result"] == "WIN" else 0
        margins.append({
            "predicted_margin": pred_margin,
            "actual_margin": actual_margin,
            "won": won,
            "cell": leg["negbin_cell"],
            "mu_h": mu_h, "mu_a": mu_a,
            "negbin_wp": leg["negbin_wp"],
            "team": team,
            "game_date": leg["game_date"],
        })
        cells[leg["negbin_cell"]].append(won)

    if not margins:
        print("No margin data — team names may not match.")
        # Try fuzzy match
        matched = 0
        unmatched = []
        for leg in legs:
            team = leg["team"]
            home = leg["resolved_home_team"] or ""
            away = leg["resolved_away_team"] or ""
            if team not in (home, away):
                unmatched.append((team, home, away))
        if unmatched:
            print(f"  Unmatched examples: {unmatched[:5]}")
        return

    pred_margins = [m["predicted_margin"] for m in margins]
    pct_fav = np.mean([1 if m > 0 else 0 for m in pred_margins]) * 100

    print(f"\n--- NegBin Predicted Margin (team-opponent) ---")
    print(f"  Mean:   {np.mean(pred_margins):+.2f} runs")
    print(f"  Median: {np.median(pred_margins):+.2f} runs")
    print(f"  Range:  [{np.min(pred_margins):+.2f}, {np.max(pred_margins):+.2f}]")
    print(f"  % predicted favorites: {pct_fav:.0f}%")

    # Win rate by margin bucket
    print(f"\n--- WR by Predicted Margin ---")
    buckets = [(-99, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 99)]
    for lo, hi in buckets:
        b = [m for m in margins if lo <= m["predicted_margin"] < hi]
        if b:
            wr = np.mean([m["won"] for m in b])
            avg_pm = np.mean([m["predicted_margin"] for m in b])
            print(f"  [{lo:+d}, {hi:+d}): N={len(b):3d}, WR={wr*100:.1f}%, avg_margin={avg_pm:+.2f}")

    # Cell distribution
    print(f"\n--- NegBin Cell ---")
    for cell in sorted(cells):
        outcomes = cells[cell]
        print(f"  {cell:10s}: N={len(outcomes):3d}, WR={np.mean(outcomes)*100:.1f}%")

    # How +1.5 wins happen
    total_won = sum(1 for m in margins if m["won"])
    won_outright = sum(1 for m in margins if m["won"] and m["actual_margin"] > 0)
    lost_by_1 = sum(1 for m in margins if m["won"] and m["actual_margin"] == -1)
    won_by_2plus = sum(1 for m in margins if m["won"] and m["actual_margin"] >= 2)
    won_by_1 = sum(1 for m in margins if m["won"] and m["actual_margin"] == 1)

    print(f"\n--- How +1.5 Wins Happen (N={total_won}) ---")
    if total_won:
        print(f"  Won outright:      {won_outright} ({won_outright/total_won*100:.1f}%)")
        print(f"    Won by 2+:       {won_by_2plus} ({won_by_2plus/total_won*100:.1f}%)")
        print(f"    Won by 1:        {won_by_1} ({won_by_1/total_won*100:.1f}%)")
        print(f"  Lost by 1 (spread): {lost_by_1} ({lost_by_1/total_won*100:.1f}%)")

    # Direction accuracy
    correct = sum(1 for m in margins if m["predicted_margin"] > 0 and m["actual_margin"] > 0)
    correct += sum(1 for m in margins if m["predicted_margin"] < 0 and m["actual_margin"] < 0)
    valid = sum(1 for m in margins if m["predicted_margin"] != 0 and m["actual_margin"] != 0)
    print(f"\n--- Direction Accuracy ---")
    print(f"  Correct winner prediction: {correct}/{valid} = {correct/valid*100:.1f}%" if valid else "  N/A")

    # NegBin confidence
    wps = [m["negbin_wp"] for m in margins]
    print(f"\n--- NegBin Confidence ---")
    print(f"  Mean WP:   {np.mean(wps)*100:.1f}%")
    print(f"  Median WP: {np.median(wps)*100:.1f}%")
    print(f"  Actual WR: {np.mean([m['won'] for m in margins])*100:.1f}%")
    print(f"  Overconfidence: {(np.mean(wps) - np.mean([m['won'] for m in margins]))*100:+.1f}pp")

    # Key insight: NegBin says 75% but actual is 67% → 8pp overconfident
    # But edge = WR - BE = 67% - 65% ≈ +2pp → real edge exists

    # Expected runs
    mu_team = []
    mu_opp = []
    for m in margins:
        if m["predicted_margin"] >= 0:
            if m["mu_h"] - m["mu_a"] >= 0:
                mu_team.append(m["mu_h"])
                mu_opp.append(m["mu_a"])
            else:
                mu_team.append(m["mu_a"])
                mu_opp.append(m["mu_h"])
        else:
            if m["mu_a"] - m["mu_h"] >= 0:
                mu_team.append(m["mu_a"])
                mu_opp.append(m["mu_h"])
            else:
                mu_team.append(m["mu_h"])
                mu_opp.append(m["mu_a"])

    print(f"\n--- Expected Run Totals ---")
    print(f"  Team μ:     {np.mean(mu_team):.2f}")
    print(f"  Opponent μ: {np.mean(mu_opp):.2f}")
    print(f"  Predicted margin: {np.mean(pred_margins):+.2f}")
    print(f"  Actual margin:    {np.mean([m['actual_margin'] for m in margins]):+.2f}")

    # ═══════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════
    print(f"\n\n{'='*60}")
    print(f"=== VERDICT ===")
    print(f"{'='*60}")

    print(f"\nGATE 1 (CLV):")
    print(f"  NegBin +1.5 AM→PM CLV: {mean_clv:+.2f}pp (z={z_clv:.2f}, N={len(clv_values)})")
    if control_clvs:
        ctrl_mean = np.mean(control_clvs)
        excess = mean_clv - ctrl_mean
        print(f"  Control (all +1.5):    {ctrl_mean:+.2f}pp (N={len(control_clvs)})")
        print(f"  Excess CLV:            {excess:+.2f}pp")
    print(f"  WR={wr*100:.1f}% vs AM BE={avg_am*100:.1f}% → edge vs entry: {(wr-avg_am)*100:+.1f}pp")
    print(f"  WR={wr*100:.1f}% vs PM BE={avg_pm*100:.1f}% → edge vs close: {(wr-avg_pm)*100:+.1f}pp")

    if mean_clv > 0 and z_clv >= 1.96:
        print(f"  → CLV is POSITIVE and significant")
    elif mean_clv > 0:
        print(f"  → CLV is positive but NOT significant (z < 1.96)")
    else:
        print(f"  → CLV is NEGATIVE — edge vs breakeven only, not vs close")

    print(f"\nGATE 2 (Mechanism):")
    print(f"  {pct_fav:.0f}% predicted favorites")
    print(f"  Mean predicted margin: {np.mean(pred_margins):+.2f} runs")
    if valid:
        print(f"  Direction accuracy: {correct/valid*100:.1f}%")
    print(f"  Overconfidence: {(np.mean(wps) - np.mean([m['won'] for m in margins]))*100:+.1f}pp")
    print(f"  fav cells WR: {np.mean(cells.get('fav_home', [0]) + cells.get('fav_away', [0]))*100:.1f}%")
    print(f"  dog cells WR: {np.mean(cells.get('dog_home', [0]) + cells.get('dog_away', [0]))*100:.1f}%")

    bets_db.close()
    hist_db.close()


if __name__ == "__main__":
    run()
