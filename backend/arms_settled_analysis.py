"""
arms_settled_analysis.py — Comprehensive settled-day analysis across NegBin arms.

All analysis on REAL PnL (dollars, real odds). No cover-rate proxies.
Tests WP thresholds, arm agreement, losing-day characterization.
Split-half stability on all findings.
"""
from __future__ import annotations
import sqlite3, os
import numpy as np
import pandas as pd
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "data", "bets.db")

# Arms with real PnL (non-zero stakes)
REAL_PNL_ARMS = [
    "shadow_negbin_edgegate",
    "shadow_negbin_fullstack",
    "shadow_negbin_atscap",
    "shadow_negbin_distwp_gate",
    "shadow_negbin_composite",
    "shadow_negbin_bothgates",
]

SHORT_NAMES = {
    "shadow_negbin_edgegate": "edgegate",
    "shadow_negbin_fullstack": "fullstack",
    "shadow_negbin_atscap": "atscap",
    "shadow_negbin_distwp_gate": "distwp",
    "shadow_negbin_composite": "composite",
    "shadow_negbin_bothgates": "bothgates",
}


def load_data():
    conn = sqlite3.connect(DB_PATH)

    # Load all settled bets from real-PnL arms
    bets = pd.read_sql_query("""
        SELECT id, source, game_date, odds, amount, actual_profit,
               predicted_win_prob, legs, status, mean_leg_dist_wp,
               min_leg_dist_wp, kelly_edge, concentrated_risk
        FROM mock_bets
        WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS', 'SETTLED_PUSH')
          AND actual_profit IS NOT NULL
          AND amount > 0
          AND source IN ({})
    """.format(",".join(f"'{a}'" for a in REAL_PNL_ARMS)), conn)

    # Load legs for agreement analysis
    legs = pd.read_sql_query("""
        SELECT ml.mock_bet_id, ml.leg_index, ml.description, ml.fixture_id,
               ml.leg_result, ml.spread_value, ml.negbin_wp, ml.ats_dist_wp,
               ml.win_prob, ml.negbin_edge_pp, ml.market_type
        FROM mock_bet_legs ml
        JOIN mock_bets mb ON mb.id = ml.mock_bet_id
        WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS', 'SETTLED_PUSH')
          AND mb.actual_profit IS NOT NULL
          AND mb.amount > 0
          AND mb.source IN ({})
    """.format(",".join(f"'{a}'" for a in REAL_PNL_ARMS)), conn)

    conn.close()
    bets["arm"] = bets["source"].map(SHORT_NAMES)
    return bets, legs


def section_1_wp_threshold(bets):
    """Verify WP>60% claim across multiple thresholds."""
    print("\n" + "=" * 110)
    print("  SECTION 1: WP THRESHOLD VERIFICATION (real PnL, dollars)")
    print("=" * 110)

    # Focus on arms with enough data
    main_arms = ["edgegate", "fullstack", "atscap"]

    for arm_name in main_arms:
        arm = bets[bets["arm"] == arm_name].copy()
        if len(arm) == 0:
            continue

        print(f"\n  ── {arm_name.upper()} ({len(arm)} bets, {arm['game_date'].nunique()} days) ──")
        print(f"  {'Threshold':>10} {'N bets':>8} {'N days':>8} {'Wins':>6} {'Losses':>8} "
              f"{'WR%':>8} {'Total PnL':>12} {'PnL/bet':>10} {'ROI%':>8}")
        print(f"  {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 8} "
              f"{'─' * 8} {'─' * 12} {'─' * 10} {'─' * 8}")

        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            subset = arm[arm["predicted_win_prob"] >= thresh]
            if len(subset) == 0:
                continue
            n = len(subset)
            n_days = subset["game_date"].nunique()
            wins = (subset["actual_profit"] > 0).sum()
            losses = (subset["actual_profit"] < 0).sum()
            total_pnl = subset["actual_profit"].sum()
            total_staked = subset["amount"].sum()
            wr = wins / n * 100 if n > 0 else 0
            roi = total_pnl / total_staked * 100 if total_staked > 0 else 0
            pnl_per = total_pnl / n

            print(f"  {'≥' + f'{thresh:.0%}':>10} {n:>8} {n_days:>8} {wins:>6} {losses:>8} "
                  f"{wr:>7.1f}% {total_pnl:>+12.2f} {pnl_per:>+10.3f} {roi:>+7.1f}%")

    # Daily aggregation: per-day WP (average of all bets that day) vs per-day PnL
    print(f"\n\n  ── DAILY WP vs DAILY PnL (per-day average WP across all bets) ──")
    for arm_name in main_arms:
        arm = bets[bets["arm"] == arm_name].copy()
        if len(arm) == 0:
            continue

        daily = arm.groupby("game_date").agg(
            n_bets=("id", "count"),
            total_pnl=("actual_profit", "sum"),
            total_staked=("amount", "sum"),
            avg_wp=("predicted_win_prob", "mean"),
            min_wp=("predicted_win_prob", "min"),
            max_wp=("predicted_win_prob", "max"),
        ).reset_index()

        print(f"\n  {arm_name.upper()} — {len(daily)} days")
        print(f"  {'WP Bucket':>12} {'N days':>8} {'Profit days':>12} {'Loss days':>11} "
              f"{'Total PnL':>12} {'Avg daily PnL':>14} {'Avg staked':>11}")
        print(f"  {'─' * 12} {'─' * 8} {'─' * 12} {'─' * 11} "
              f"{'─' * 12} {'─' * 14} {'─' * 11}")

        for lo, hi, label in [(0.50, 0.55, "50-55%"), (0.55, 0.60, "55-60%"),
                               (0.60, 0.65, "60-65%"), (0.65, 0.70, "65-70%"),
                               (0.70, 0.75, "70-75%"), (0.75, 1.0, "75%+")]:
            bucket = daily[(daily["avg_wp"] >= lo) & (daily["avg_wp"] < hi)]
            if len(bucket) == 0:
                continue
            n_d = len(bucket)
            profit_d = (bucket["total_pnl"] > 0).sum()
            loss_d = (bucket["total_pnl"] < 0).sum()
            total = bucket["total_pnl"].sum()
            avg_daily = total / n_d
            avg_stk = bucket["total_staked"].mean()
            print(f"  {label:>12} {n_d:>8} {profit_d:>12} {loss_d:>11} "
                  f"{total:>+12.2f} {avg_daily:>+14.2f} {avg_stk:>11.1f}")

    # Split-half stability
    print(f"\n\n  ── SPLIT-HALF STABILITY (first half vs second half of days) ──")
    for arm_name in main_arms:
        arm = bets[bets["arm"] == arm_name].copy()
        if len(arm) == 0:
            continue

        dates = sorted(arm["game_date"].unique())
        mid = len(dates) // 2
        h1_dates = set(dates[:mid])
        h2_dates = set(dates[mid:])

        h1 = arm[arm["game_date"].isin(h1_dates)]
        h2 = arm[arm["game_date"].isin(h2_dates)]

        print(f"\n  {arm_name.upper()}")
        for thresh in [0.55, 0.60, 0.65, 0.70]:
            h1_sub = h1[h1["predicted_win_prob"] >= thresh]
            h2_sub = h2[h2["predicted_win_prob"] >= thresh]
            h1_pnl = h1_sub["actual_profit"].sum() if len(h1_sub) > 0 else 0
            h2_pnl = h2_sub["actual_profit"].sum() if len(h2_sub) > 0 else 0
            h1_roi = h1_pnl / h1_sub["amount"].sum() * 100 if len(h1_sub) > 0 and h1_sub["amount"].sum() > 0 else 0
            h2_roi = h2_pnl / h2_sub["amount"].sum() * 100 if len(h2_sub) > 0 and h2_sub["amount"].sum() > 0 else 0
            print(f"    ≥{thresh:.0%}: H1 n={len(h1_sub):>4} PnL={h1_pnl:>+8.1f} ROI={h1_roi:>+6.1f}%  |  "
                  f"H2 n={len(h2_sub):>4} PnL={h2_pnl:>+8.1f} ROI={h2_roi:>+6.1f}%  |  "
                  f"{'SAME SIGN' if h1_pnl * h2_pnl > 0 else 'OPPOSITE'}")


def section_2_arm_comparison(bets, legs):
    """Compare arms on real PnL and test agreement/intersection."""
    print("\n\n" + "=" * 110)
    print("  SECTION 2: ARM COMPARISON AND AGREEMENT (real PnL)")
    print("=" * 110)

    # ── Per-arm summary ──
    print(f"\n  {'Arm':<12} {'N bets':>8} {'N days':>8} {'Total PnL':>12} {'PnL/bet':>10} "
          f"{'ROI%':>8} {'WR%':>8} {'Avg odds':>10} {'Avg WP':>8}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 12} {'─' * 10} "
          f"{'─' * 8} {'─' * 8} {'─' * 10} {'─' * 8}")

    for arm_name in sorted(bets["arm"].unique()):
        arm = bets[bets["arm"] == arm_name]
        n = len(arm)
        n_days = arm["game_date"].nunique()
        total_pnl = arm["actual_profit"].sum()
        total_staked = arm["amount"].sum()
        roi = total_pnl / total_staked * 100 if total_staked > 0 else 0
        wr = (arm["actual_profit"] > 0).sum() / n * 100
        print(f"  {arm_name:<12} {n:>8} {n_days:>8} {total_pnl:>+12.2f} "
              f"{total_pnl / n:>+10.3f} {roi:>+7.1f}% {wr:>7.1f}% "
              f"{arm['odds'].mean():>10.3f} {arm['predicted_win_prob'].mean():>7.1f}%")

    # ── Per-day PnL by arm ──
    print(f"\n\n  ── PER-DAY PnL (each row = one day) ──")
    daily_all = {}
    for arm_name in ["edgegate", "fullstack", "atscap"]:
        arm = bets[bets["arm"] == arm_name]
        daily = arm.groupby("game_date")["actual_profit"].sum()
        daily_all[arm_name] = daily

    # Common days
    common_dates = sorted(set.intersection(*[set(d.index) for d in daily_all.values()]))
    print(f"\n  Common days across edgegate/fullstack/atscap: {len(common_dates)}")
    print(f"  {'Date':<12} {'edgegate':>12} {'fullstack':>12} {'atscap':>12} {'All agree':>12}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")

    agree_profit = 0
    agree_loss = 0
    disagree_days = 0
    for d in common_dates:
        eg = daily_all["edgegate"].get(d, 0)
        fs = daily_all["fullstack"].get(d, 0)
        ac = daily_all["atscap"].get(d, 0)
        signs = [np.sign(eg), np.sign(fs), np.sign(ac)]
        all_agree = len(set(signs)) == 1 and 0 not in signs
        if all_agree and signs[0] > 0:
            agree_profit += 1
        elif all_agree and signs[0] < 0:
            agree_loss += 1
        else:
            disagree_days += 1
        print(f"  {d:<12} {eg:>+12.2f} {fs:>+12.2f} {ac:>+12.2f} "
              f"{'ALL+' if all_agree and signs[0] > 0 else 'ALL-' if all_agree and signs[0] < 0 else 'MIXED':>12}")

    print(f"\n  Agreement summary: ALL+ = {agree_profit}, ALL- = {agree_loss}, MIXED = {disagree_days}")

    # ── Leg-level agreement ──
    print(f"\n\n  ── LEG-LEVEL AGREEMENT (do arms pick the same legs?) ──")

    # Build fixture-date-spread-pick tuples per arm
    leg_picks = defaultdict(lambda: defaultdict(set))  # arm -> fixture_spread_key -> set of bet_ids
    for _, row in legs.iterrows():
        bet = bets[bets["id"] == row["mock_bet_id"]]
        if len(bet) == 0:
            continue
        arm = bet.iloc[0]["arm"]
        # Key: fixture_id + spread_value as identifier for "same pick"
        key = (row["fixture_id"], row.get("spread_value"), row.get("description", "")[:30])
        leg_picks[arm][key].add(row["mock_bet_id"])

    # Find legs that appear in multiple arms
    all_legs_by_key = defaultdict(set)  # key -> set of arms
    leg_results_by_key = {}
    for _, row in legs.iterrows():
        key = (row["fixture_id"], row.get("spread_value"), row.get("description", "")[:30])
        bet = bets[bets["id"] == row["mock_bet_id"]]
        if len(bet) > 0:
            all_legs_by_key[key].add(bet.iloc[0]["arm"])
            leg_results_by_key[key] = row["leg_result"]

    # For each overlap count, what's the win rate?
    for min_arms in [1, 2, 3]:
        keys_at_level = [k for k, arms in all_legs_by_key.items() if len(arms) >= min_arms]
        if not keys_at_level:
            continue
        wins = sum(1 for k in keys_at_level if leg_results_by_key.get(k) == "WIN")
        losses = sum(1 for k in keys_at_level if leg_results_by_key.get(k) == "LOSS")
        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        print(f"  Legs in ≥{min_arms} arms: {total} resolved, WR = {wr:.1f}% ({wins}W/{losses}L)")

    # ── Intersection analysis: edge_gate AND fullstack both include ──
    print(f"\n\n  ── INTERSECTION ANALYSIS (bets in BOTH arms) ──")

    # Find bets that share the same legs across arms
    # Group bets by game_date, then check which fixture+spread combos overlap
    for d in common_dates:
        eg_bets = bets[(bets["arm"] == "edgegate") & (bets["game_date"] == d)]
        fs_bets = bets[(bets["arm"] == "fullstack") & (bets["game_date"] == d)]

    # Better approach: per bet, get its leg keys; find bets across arms with same leg set
    bet_leg_keys = {}
    for bet_id in bets["id"].unique():
        bet_legs = legs[legs["mock_bet_id"] == bet_id]
        keys = frozenset(
            (r["fixture_id"], r.get("spread_value"))
            for _, r in bet_legs.iterrows()
        )
        bet_leg_keys[bet_id] = keys

    # For each arm pair, find bets with identical leg sets
    arm_pairs = [("edgegate", "fullstack"), ("edgegate", "atscap"),
                 ("fullstack", "atscap"), ("edgegate", "bothgates")]

    for a1, a2 in arm_pairs:
        bets_a1 = bets[bets["arm"] == a1]
        bets_a2 = bets[bets["arm"] == a2]
        if len(bets_a1) == 0 or len(bets_a2) == 0:
            continue

        # Find matching parlays (same leg set)
        a1_keys = {bet_leg_keys.get(bid, frozenset()): bid for bid in bets_a1["id"]}
        a2_keys = {bet_leg_keys.get(bid, frozenset()): bid for bid in bets_a2["id"]}
        shared = set(a1_keys.keys()) & set(a2_keys.keys()) - {frozenset()}

        if shared:
            shared_a1_ids = [a1_keys[k] for k in shared]
            shared_bets = bets_a1[bets_a1["id"].isin(shared_a1_ids)]
            pnl = shared_bets["actual_profit"].sum()
            staked = shared_bets["amount"].sum()
            roi = pnl / staked * 100 if staked > 0 else 0
            wr = (shared_bets["actual_profit"] > 0).sum() / len(shared_bets) * 100
            print(f"  {a1} ∩ {a2}: {len(shared)} identical parlays, "
                  f"PnL={pnl:+.2f}, ROI={roi:+.1f}%, WR={wr:.1f}%")
        else:
            print(f"  {a1} ∩ {a2}: 0 identical parlays (arms construct differently)")


def section_3_losing_days(bets, legs):
    """Characterize winning vs losing days."""
    print("\n\n" + "=" * 110)
    print("  SECTION 3: LOSING DAY CHARACTERIZATION")
    print("=" * 110)

    # Focus on edgegate (most data, flat stakes = clean PnL)
    for arm_name in ["edgegate", "fullstack", "atscap"]:
        arm = bets[bets["arm"] == arm_name].copy()
        if len(arm) == 0:
            continue

        daily = arm.groupby("game_date").agg(
            n_bets=("id", "count"),
            total_pnl=("actual_profit", "sum"),
            total_staked=("amount", "sum"),
            avg_wp=("predicted_win_prob", "mean"),
            avg_odds=("odds", "mean"),
            n_legs_avg=("legs", "mean"),
            wins=("actual_profit", lambda x: (x > 0).sum()),
            losses=("actual_profit", lambda x: (x < 0).sum()),
        ).reset_index()
        daily["wr"] = daily["wins"] / daily["n_bets"]
        daily["roi"] = daily["total_pnl"] / daily["total_staked"] * 100

        win_days = daily[daily["total_pnl"] > 0]
        loss_days = daily[daily["total_pnl"] < 0]
        even_days = daily[daily["total_pnl"] == 0]

        print(f"\n  ── {arm_name.upper()} ──")
        print(f"  Total days: {len(daily)} (profit: {len(win_days)}, loss: {len(loss_days)}, even: {len(even_days)})")

        for label, subset in [("PROFIT days", win_days), ("LOSS days", loss_days)]:
            if len(subset) == 0:
                continue
            print(f"\n  {label} ({len(subset)}):")
            print(f"    Avg bets/day:     {subset['n_bets'].mean():.1f}")
            print(f"    Avg WP:           {subset['avg_wp'].mean():.3f}")
            print(f"    Avg odds:         {subset['avg_odds'].mean():.3f}")
            print(f"    Avg parlay WR:    {subset['wr'].mean():.1%}")
            print(f"    Avg daily PnL:    {subset['total_pnl'].mean():+.2f}")
            print(f"    Avg ROI:          {subset['roi'].mean():+.1f}%")
            print(f"    Avg legs/parlay:  {subset['n_legs_avg'].mean():.2f}")

        # Spread-value distribution on winning vs losing bets
        arm_legs = legs[legs["mock_bet_id"].isin(arm["id"])]
        arm_with_pnl = arm[["id", "actual_profit", "game_date"]].copy()
        arm_legs_merged = arm_legs.merge(arm_with_pnl, left_on="mock_bet_id", right_on="id", how="left")

        # Rung distribution
        if "spread_value" in arm_legs_merged.columns and arm_legs_merged["spread_value"].notna().any():
            print(f"\n  Rung distribution (winning vs losing bets):")
            win_legs = arm_legs_merged[arm_legs_merged["actual_profit"] > 0]
            loss_legs = arm_legs_merged[arm_legs_merged["actual_profit"] < 0]

            all_rungs = sorted(arm_legs_merged["spread_value"].dropna().unique())
            print(f"    {'Rung':>8} {'Win-bet legs':>14} {'Loss-bet legs':>15} {'Leg WR (all)':>14}")
            for rung in all_rungs:
                wl = len(win_legs[win_legs["spread_value"] == rung])
                ll = len(loss_legs[loss_legs["spread_value"] == rung])
                all_at_rung = arm_legs_merged[arm_legs_merged["spread_value"] == rung]
                leg_wins = (all_at_rung["leg_result"] == "WIN").sum()
                leg_total = all_at_rung["leg_result"].isin(["WIN", "LOSS"]).sum()
                leg_wr = leg_wins / leg_total * 100 if leg_total > 0 else 0
                print(f"    {rung:>8.1f} {wl:>14} {ll:>15} {leg_wr:>13.1f}%")

    # ── What distinguishes the WORST days? ──
    print(f"\n\n  ── WORST DAYS DEEP DIVE (edgegate) ──")
    eg = bets[bets["arm"] == "edgegate"].copy()
    daily_eg = eg.groupby("game_date").agg(
        n_bets=("id", "count"),
        total_pnl=("actual_profit", "sum"),
        avg_wp=("predicted_win_prob", "mean"),
        avg_odds=("odds", "mean"),
        max_wp=("predicted_win_prob", "max"),
        min_wp=("predicted_win_prob", "min"),
    ).reset_index().sort_values("total_pnl")

    print(f"\n  Bottom 5 days (worst PnL):")
    print(f"  {'Date':<12} {'PnL':>10} {'N bets':>8} {'Avg WP':>8} {'Min WP':>8} {'Max WP':>8} {'Avg odds':>10}")
    for _, row in daily_eg.head(5).iterrows():
        print(f"  {row['game_date']:<12} {row['total_pnl']:>+10.2f} {row['n_bets']:>8} "
              f"{row['avg_wp']:>8.3f} {row['min_wp']:>8.3f} {row['max_wp']:>8.3f} "
              f"{row['avg_odds']:>10.3f}")

    print(f"\n  Top 5 days (best PnL):")
    for _, row in daily_eg.tail(5).iterrows():
        print(f"  {row['game_date']:<12} {row['total_pnl']:>+10.2f} {row['n_bets']:>8} "
              f"{row['avg_wp']:>8.3f} {row['min_wp']:>8.3f} {row['max_wp']:>8.3f} "
              f"{row['avg_odds']:>10.3f}")

    # ── Correlation analysis: pre-bet observables vs daily PnL ──
    print(f"\n\n  ── PRE-BET OBSERVABLES vs DAILY PnL (edgegate) ──")
    from scipy.stats import spearmanr

    daily_features = eg.groupby("game_date").agg(
        total_pnl=("actual_profit", "sum"),
        avg_wp=("predicted_win_prob", "mean"),
        min_wp=("predicted_win_prob", "min"),
        max_wp=("predicted_win_prob", "max"),
        wp_range=("predicted_win_prob", lambda x: x.max() - x.min()),
        avg_odds=("odds", "mean"),
        n_bets=("id", "count"),
        avg_legs=("legs", "mean"),
        avg_kelly=("kelly_edge", "mean"),
    ).reset_index()

    print(f"\n  Spearman ρ(observable, daily_pnl):")
    for col in ["avg_wp", "min_wp", "max_wp", "wp_range", "avg_odds", "n_bets", "avg_legs", "avg_kelly"]:
        valid = daily_features[[col, "total_pnl"]].dropna()
        if len(valid) < 5:
            continue
        rho, p = spearmanr(valid[col], valid["total_pnl"])
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"    {col:<20} ρ = {rho:+.3f}  (p = {p:.3f}) {sig}")

    # ── Leg correlation on losing days ──
    print(f"\n\n  ── LEG CORRELATION ON LOSING DAYS ──")
    # On losing days, how many legs lost? (measure of correlation — all-lose vs mixed)
    eg_daily = eg.groupby("game_date")["actual_profit"].sum().reset_index()
    loss_dates = set(eg_daily[eg_daily["actual_profit"] < 0]["game_date"])
    win_dates = set(eg_daily[eg_daily["actual_profit"] > 0]["game_date"])

    for label, date_set in [("LOSS days", loss_dates), ("PROFIT days", win_dates)]:
        day_bets = eg[eg["game_date"].isin(date_set)]
        day_legs = legs[legs["mock_bet_id"].isin(day_bets["id"])]
        if len(day_legs) == 0:
            continue

        # Unique fixtures per day
        fixtures_per_day = day_legs.groupby(
            day_bets.set_index("id").loc[day_legs["mock_bet_id"].values, "game_date"].values
        )["fixture_id"].nunique()

        # Leg results
        leg_wins = (day_legs["leg_result"] == "WIN").sum()
        leg_losses = (day_legs["leg_result"] == "LOSS").sum()
        leg_total = leg_wins + leg_losses
        leg_wr = leg_wins / leg_total * 100 if leg_total > 0 else 0

        # How many unique fixtures across all bets that day?
        avg_unique_fix = fixtures_per_day.mean() if len(fixtures_per_day) > 0 else 0

        print(f"  {label}: leg WR = {leg_wr:.1f}% ({leg_wins}W/{leg_losses}L), "
              f"avg unique fixtures/day = {avg_unique_fix:.1f}")


def section_4_pre_bet_condition(bets):
    """Search for a pre-bet-observable condition that separates good from bad days."""
    print("\n\n" + "=" * 110)
    print("  SECTION 4: PRE-BET CONDITION SEARCH (actionable signals)")
    print("=" * 110)

    eg = bets[bets["arm"] == "edgegate"].copy()

    # Test: is "avg daily WP ≥ X" a profitable-day filter?
    daily = eg.groupby("game_date").agg(
        n_bets=("id", "count"),
        total_pnl=("actual_profit", "sum"),
        total_staked=("amount", "sum"),
        avg_wp=("predicted_win_prob", "mean"),
        min_wp=("predicted_win_prob", "min"),
        n_high_wp=("predicted_win_prob", lambda x: (x >= 0.70).sum()),
        n_low_wp=("predicted_win_prob", lambda x: (x < 0.55).sum()),
        avg_kelly=("kelly_edge", "mean"),
    ).reset_index()

    daily["pct_high_wp"] = daily["n_high_wp"] / daily["n_bets"]
    daily["pct_low_wp"] = daily["n_low_wp"] / daily["n_bets"]

    # Test candidate conditions
    print(f"\n  EDGEGATE — Candidate pre-bet conditions on daily PnL:")
    print(f"  {'Condition':<40} {'N days':>8} {'Profit':>8} {'Loss':>8} "
          f"{'PnL':>12} {'Avg PnL':>10} {'ROI%':>8} {'H1':>8} {'H2':>8}")
    print(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 8} "
          f"{'─' * 12} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8}")

    dates = sorted(daily["game_date"].unique())
    mid = len(dates) // 2
    h1_set = set(dates[:mid])
    h2_set = set(dates[mid:])

    conditions = [
        ("avg_wp ≥ 0.55", lambda d: d["avg_wp"] >= 0.55),
        ("avg_wp ≥ 0.60", lambda d: d["avg_wp"] >= 0.60),
        ("avg_wp ≥ 0.65", lambda d: d["avg_wp"] >= 0.65),
        ("avg_wp ≥ 0.70", lambda d: d["avg_wp"] >= 0.70),
        ("min_wp ≥ 0.55", lambda d: d["min_wp"] >= 0.55),
        ("min_wp ≥ 0.60", lambda d: d["min_wp"] >= 0.60),
        ("n_bets ≥ 20", lambda d: d["n_bets"] >= 20),
        ("n_bets ≥ 30", lambda d: d["n_bets"] >= 30),
        ("n_bets ≤ 20", lambda d: d["n_bets"] <= 20),
        ("n_bets ≤ 30 AND avg_wp ≥ 0.60", lambda d: (d["n_bets"] <= 30) & (d["avg_wp"] >= 0.60)),
        ("pct_high_wp (≥70%) > 30%", lambda d: d["pct_high_wp"] > 0.30),
        ("pct_low_wp (<55%) < 20%", lambda d: d["pct_low_wp"] < 0.20),
        ("no low-wp bets (<55%)", lambda d: d["n_low_wp"] == 0),
        ("avg_kelly ≥ 0.03", lambda d: d["avg_kelly"] >= 0.03),
        ("avg_kelly ≥ 0.05", lambda d: d["avg_kelly"] >= 0.05),
    ]

    for label, cond_fn in conditions:
        try:
            mask = cond_fn(daily)
            subset = daily[mask]
        except Exception:
            continue
        if len(subset) == 0:
            continue

        n_d = len(subset)
        profit = (subset["total_pnl"] > 0).sum()
        loss = (subset["total_pnl"] < 0).sum()
        total_pnl = subset["total_pnl"].sum()
        total_staked = subset["total_staked"].sum()
        roi = total_pnl / total_staked * 100 if total_staked > 0 else 0

        # Split-half
        h1_sub = subset[subset["game_date"].isin(h1_set)]
        h2_sub = subset[subset["game_date"].isin(h2_set)]
        h1_pnl = h1_sub["total_pnl"].sum()
        h2_pnl = h2_sub["total_pnl"].sum()
        h1_stk = h1_sub["total_staked"].sum()
        h2_stk = h2_sub["total_staked"].sum()
        h1_roi = h1_pnl / h1_stk * 100 if h1_stk > 0 else 0
        h2_roi = h2_pnl / h2_stk * 100 if h2_stk > 0 else 0

        print(f"  {label:<40} {n_d:>8} {profit:>8} {loss:>8} "
              f"{total_pnl:>+12.2f} {total_pnl / n_d:>+10.2f} {roi:>+7.1f}% "
              f"{h1_roi:>+7.1f}% {h2_roi:>+7.1f}%")

    # Now do the same for FULLSTACK
    print(f"\n  FULLSTACK — same conditions:")
    fs = bets[bets["arm"] == "fullstack"].copy()
    daily_fs = fs.groupby("game_date").agg(
        n_bets=("id", "count"),
        total_pnl=("actual_profit", "sum"),
        total_staked=("amount", "sum"),
        avg_wp=("predicted_win_prob", "mean"),
        min_wp=("predicted_win_prob", "min"),
        n_high_wp=("predicted_win_prob", lambda x: (x >= 0.70).sum()),
        n_low_wp=("predicted_win_prob", lambda x: (x < 0.55).sum()),
        avg_kelly=("kelly_edge", "mean"),
    ).reset_index()

    daily_fs["pct_high_wp"] = daily_fs["n_high_wp"] / daily_fs["n_bets"]
    daily_fs["pct_low_wp"] = daily_fs["n_low_wp"] / daily_fs["n_bets"]

    dates_fs = sorted(daily_fs["game_date"].unique())
    mid_fs = len(dates_fs) // 2
    h1_set_fs = set(dates_fs[:mid_fs])
    h2_set_fs = set(dates_fs[mid_fs:])

    print(f"  {'Condition':<40} {'N days':>8} {'Profit':>8} {'Loss':>8} "
          f"{'PnL':>12} {'Avg PnL':>10} {'ROI%':>8} {'H1':>8} {'H2':>8}")
    print(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 8} "
          f"{'─' * 12} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8}")

    for label, cond_fn in conditions:
        try:
            mask = cond_fn(daily_fs)
            subset = daily_fs[mask]
        except Exception:
            continue
        if len(subset) == 0:
            continue

        n_d = len(subset)
        profit = (subset["total_pnl"] > 0).sum()
        loss = (subset["total_pnl"] < 0).sum()
        total_pnl = subset["total_pnl"].sum()
        total_staked = subset["total_staked"].sum()
        roi = total_pnl / total_staked * 100 if total_staked > 0 else 0

        h1_sub = subset[subset["game_date"].isin(h1_set_fs)]
        h2_sub = subset[subset["game_date"].isin(h2_set_fs)]
        h1_pnl = h1_sub["total_pnl"].sum()
        h2_pnl = h2_sub["total_pnl"].sum()
        h1_stk = h1_sub["total_staked"].sum()
        h2_stk = h2_sub["total_staked"].sum()
        h1_roi = h1_pnl / h1_stk * 100 if h1_stk > 0 else 0
        h2_roi = h2_pnl / h2_stk * 100 if h2_stk > 0 else 0

        print(f"  {label:<40} {n_d:>8} {profit:>8} {loss:>8} "
              f"{total_pnl:>+12.2f} {total_pnl / n_d:>+10.2f} {roi:>+7.1f}% "
              f"{h1_roi:>+7.1f}% {h2_roi:>+7.1f}%")


if __name__ == "__main__":
    bets, legs = load_data()
    section_1_wp_threshold(bets)
    section_2_arm_comparison(bets, legs)
    section_3_losing_days(bets, legs)
    section_4_pre_bet_condition(bets)
