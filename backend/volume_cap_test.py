#!/usr/bin/env python3
"""
Volume-cap / exposure-limit test on existing bet history.

The disaster-day asymmetry: losing days are 2-3× bigger than winning days,
clustering on high-volume days. Does a volume cap improve the PnL distribution
without touching the signal?

Tests:
  1. Daily bet-count caps (max N bets per day): 5, 10, 15, 20, 30
  2. Daily exposure caps (total $ at risk): $50, $100, $150, $200
  3. Cherry-pick ordering: when capped, which bets to keep?
     a. Random (shuffled) — no model advantage
     b. By predicted edge (keep highest-edge bets) — model-guided
     c. By rung (keep certain spreads only)

Metrics: cumulative PnL, max drawdown, Sharpe, winning-day%, disaster-day count
"""
import sqlite3
import numpy as np
from collections import defaultdict

DB_PATH = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"


def load_daily_bets():
    """Load settled mock_bets grouped by date, for the edge-gated production arm."""
    conn = sqlite3.connect(DB_PATH)

    # Use shadow_negbin_edgegate (the production-like arm with edge gating)
    # Also test on shadow_negbin_fullstack
    arms = ["shadow_negbin_edgegate", "shadow_negbin_fullstack", "shadow_negbin"]

    results = {}
    for arm in arms:
        rows = conn.execute("""
            SELECT b.game_date, b.odds, b.status, b.amount,
                   l.win_prob, l.predicted_edge_pp, l.description
            FROM mock_bets b
            JOIN mock_bet_legs l ON l.mock_bet_id = b.id
            WHERE b.source = ? AND b.cohort = 'clean'
              AND b.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
            ORDER BY b.game_date, l.predicted_edge_pp DESC
        """, (arm,)).fetchall()

        if not rows:
            continue

        daily = defaultdict(list)
        for gdate, odds, status, amount, wp, edge, desc in rows:
            daily[gdate].append({
                "odds": odds,
                "won": status == "SETTLED_WIN",
                "amount": amount if amount and amount > 0 else 10.0,  # default $10 flat
                "wp": wp or 0.5,
                "edge_pp": edge or 0,
                "desc": desc or "",
            })

        results[arm] = dict(daily)

    conn.close()
    return results


def simulate_cap(daily_bets: dict, cap_type: str, cap_value: float,
                 selection: str = "edge") -> dict:
    """
    Simulate a volume cap on daily bets.

    cap_type: "count" or "exposure"
    cap_value: max bets per day (count) or max $ at risk (exposure)
    selection: "edge" (keep highest edge), "random", "worst_edge" (keep lowest)
    """
    rng = np.random.RandomState(42)
    dates = sorted(daily_bets.keys())
    daily_pnl = []
    daily_n = []
    total_bets = 0
    total_won = 0

    for date in dates:
        bets = daily_bets[date].copy()

        # Sort by selection criterion
        if selection == "edge":
            bets.sort(key=lambda b: b["edge_pp"], reverse=True)
        elif selection == "random":
            rng.shuffle(bets)
        elif selection == "worst_edge":
            bets.sort(key=lambda b: b["edge_pp"])

        # Apply cap
        if cap_type == "count":
            bets = bets[:int(cap_value)]
        elif cap_type == "exposure":
            kept = []
            total_risk = 0
            for b in bets:
                if total_risk + b["amount"] <= cap_value:
                    kept.append(b)
                    total_risk += b["amount"]
            bets = kept

        # Compute PnL (flat $10 per bet)
        day_pnl = 0
        for b in bets:
            stake = 10.0
            if b["won"]:
                day_pnl += stake * (b["odds"] - 1)
                total_won += 1
            else:
                day_pnl -= stake
            total_bets += 1

        daily_pnl.append(day_pnl)
        daily_n.append(len(bets))

    cum_pnl = np.cumsum(daily_pnl)
    wr = total_won / total_bets if total_bets > 0 else 0

    # Metrics
    max_dd = 0
    peak = 0
    for p in cum_pnl:
        if p > peak:
            peak = p
        dd = peak - p
        if dd > max_dd:
            max_dd = dd

    daily_arr = np.array(daily_pnl)
    sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252) if daily_arr.std() > 0 else 0
    win_days = sum(1 for p in daily_pnl if p > 0)
    loss_days = sum(1 for p in daily_pnl if p < 0)
    disaster_days = sum(1 for p in daily_pnl if p < -50)  # > $50 loss in a day
    big_loss_days = sum(1 for p in daily_pnl if p < -30)

    return {
        "total_bets": total_bets,
        "total_days": len(dates),
        "avg_daily_n": np.mean(daily_n),
        "cum_pnl": float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0,
        "wr": wr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_days": win_days,
        "loss_days": loss_days,
        "win_day_pct": win_days / len(dates) * 100 if dates else 0,
        "disaster_days": disaster_days,
        "big_loss_days": big_loss_days,
        "avg_daily_pnl": float(daily_arr.mean()),
        "worst_day": float(daily_arr.min()),
        "best_day": float(daily_arr.max()),
    }


def run_volume_cap_analysis():
    all_daily = load_daily_bets()

    for arm, daily_bets in all_daily.items():
        n_days = len(daily_bets)
        n_bets = sum(len(v) for v in daily_bets.values())
        daily_counts = [len(v) for v in daily_bets.values()]

        print(f"\n{'='*80}")
        print(f"ARM: {arm}")
        print(f"{'='*80}")
        print(f"  {n_bets} bets across {n_days} days")
        print(f"  Daily count: mean={np.mean(daily_counts):.1f}, "
              f"median={np.median(daily_counts):.0f}, "
              f"max={max(daily_counts)}, "
              f"p90={np.percentile(daily_counts, 90):.0f}")

        # Baseline: no cap
        base = simulate_cap(daily_bets, "count", 9999, "edge")
        print(f"\n  BASELINE (no cap):")
        print(f"    PnL=${base['cum_pnl']:+.0f}, WR={base['wr']:.3f}, "
              f"Sharpe={base['sharpe']:.2f}, MaxDD=${base['max_dd']:.0f}")
        print(f"    Win days={base['win_days']}/{base['total_days']} ({base['win_day_pct']:.1f}%), "
              f"disaster(>$50)={base['disaster_days']}, big_loss(>$30)={base['big_loss_days']}")
        print(f"    Worst day=${base['worst_day']:.0f}, Best day=${base['best_day']:.0f}")

        # Count caps
        print(f"\n  ── BET-COUNT CAPS (keep highest-edge bets) ──")
        print(f"  {'Cap':<8} {'Bets':<8} {'PnL':<10} {'WR':<8} {'Sharpe':<8} "
              f"{'MaxDD':<8} {'Win%':<8} {'Disaster':<10} {'Worst':<8}")

        for cap in [5, 10, 15, 20, 30, 50]:
            r = simulate_cap(daily_bets, "count", cap, "edge")
            delta_pnl = r["cum_pnl"] - base["cum_pnl"]
            flag = " ←" if r["cum_pnl"] > base["cum_pnl"] and r["disaster_days"] < base["disaster_days"] else ""
            print(f"  {cap:<8} {r['total_bets']:<8} ${r['cum_pnl']:+7.0f}   "
                  f"{r['wr']:.3f}  {r['sharpe']:+5.2f}   "
                  f"${r['max_dd']:5.0f}  {r['win_day_pct']:5.1f}%  "
                  f"{r['disaster_days']:<10} ${r['worst_day']:6.0f}{flag}")

        # Does SELECTION matter? Compare edge-sorted vs random vs worst-edge
        print(f"\n  ── SELECTION MATTERS? (cap=15) ──")
        print(f"  {'Selection':<15} {'PnL':<10} {'WR':<8} {'Sharpe':<8} {'Disaster'}")
        for sel in ["edge", "random", "worst_edge"]:
            r = simulate_cap(daily_bets, "count", 15, sel)
            print(f"  {sel:<15} ${r['cum_pnl']:+7.0f}   {r['wr']:.3f}  "
                  f"{r['sharpe']:+5.2f}   {r['disaster_days']}")

        # Exposure caps
        print(f"\n  ── EXPOSURE CAPS (total $ at risk per day) ──")
        print(f"  {'Cap':<10} {'Bets':<8} {'PnL':<10} {'WR':<8} {'Sharpe':<8} "
              f"{'MaxDD':<8} {'Disaster'}")
        for cap in [50, 100, 150, 200, 300]:
            r = simulate_cap(daily_bets, "exposure", cap, "edge")
            print(f"  ${cap:<9} {r['total_bets']:<8} ${r['cum_pnl']:+7.0f}   "
                  f"{r['wr']:.3f}  {r['sharpe']:+5.2f}   "
                  f"${r['max_dd']:5.0f}  {r['disaster_days']}")

        # Daily PnL histogram (shows the asymmetry)
        print(f"\n  ── DAILY PnL DISTRIBUTION (baseline) ──")
        base_full = simulate_cap(daily_bets, "count", 9999, "edge")
        daily_pnls = []
        for date in sorted(daily_bets.keys()):
            day_pnl = 0
            for b in daily_bets[date]:
                if b["won"]:
                    day_pnl += 10.0 * (b["odds"] - 1)
                else:
                    day_pnl -= 10.0
            daily_pnls.append((date, day_pnl, len(daily_bets[date])))

        daily_pnls.sort(key=lambda x: x[1])
        print(f"  Worst 5 days:")
        for date, pnl, n in daily_pnls[:5]:
            print(f"    {date}: PnL=${pnl:+.0f} ({n} bets)")
        print(f"  Best 5 days:")
        for date, pnl, n in daily_pnls[-5:]:
            print(f"    {date}: PnL=${pnl:+.0f} ({n} bets)")

        # Correlation: daily bet count vs daily PnL
        counts = [x[2] for x in daily_pnls]
        pnls = [x[1] for x in daily_pnls]
        corr = np.corrcoef(counts, pnls)[0, 1]
        print(f"\n  r(daily_count, daily_pnl) = {corr:+.3f}")

        # High-volume vs low-volume performance
        median_n = np.median(counts)
        high_vol = [p for c, p in zip(counts, pnls) if c > median_n]
        low_vol = [p for c, p in zip(counts, pnls) if c <= median_n]
        print(f"  High-volume days (>{median_n:.0f} bets): avg PnL=${np.mean(high_vol):+.1f}, "
              f"N={len(high_vol)}")
        print(f"  Low-volume days (≤{median_n:.0f} bets):  avg PnL=${np.mean(low_vol):+.1f}, "
              f"N={len(low_vol)}")


if __name__ == "__main__":
    run_volume_cap_analysis()
