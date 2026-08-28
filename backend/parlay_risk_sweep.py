"""
parlay_risk_sweep.py — Full risk-metric table across ×0.82-0.92.

For EACH penalty strength: ROI, WR, Sharpe, max drawdown, daily std, neg days.
Question: is there a value that wins across-the-board on risk metrics,
or do they bounce non-monotonically (→ pick mid-range by default)?
"""
from __future__ import annotations
import os, sys, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parlay_assembly_backtest import (
    _load_ml_favorites, _load_alt_odds_raw, generate_daily_candidates,
    wilson_ci, STAKE,
)
from parlay_penalty_sweep import (
    assemble_with_rule, make_comp_only_rule, _pnl_row,
)

MAX_PARLAYS_PER_DAY = 8
FULL_START = "2026-04-22"
FULL_END   = "2026-06-26"
POST_START = "2026-05-26"


def _run_config(daily_cands, start, end, rule_fn):
    all_parlays = []
    for gd in sorted(daily_cands.keys()):
        if gd < start or gd > end:
            continue
        ps = assemble_with_rule(daily_cands[gd], MAX_PARLAYS_PER_DAY, rule_fn)
        for p in ps:
            p["game_date"] = gd
        all_parlays.extend(ps)
    return all_parlays


def _daily_pnl(parlays):
    by_day = defaultdict(list)
    for p in parlays:
        by_day[p["game_date"]].append(p)
    result = []
    for gd in sorted(by_day.keys()):
        pnl = sum(p["profit"] for p in by_day[gd])
        result.append((gd, pnl, len(by_day[gd])))
    return result


def _risk_stats(daily):
    pnls = [d[1] for d in daily]
    n = len(pnls)
    if n < 2:
        return {"mean": 0, "std": 0, "sharpe": 0, "max_dd": 0,
                "worst_day": 0, "neg_days": 0, "n_days": n}
    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / (n - 1)
    std = math.sqrt(var)
    sharpe = mean / std if std > 0 else 0

    cum = peak = 0
    max_dd = 0
    for _, pnl, _ in daily:
        cum += pnl
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "mean": round(mean, 2), "std": round(std, 2),
        "sharpe": round(sharpe, 3), "max_dd": round(max_dd, 2),
        "worst_day": round(min(pnls), 2),
        "neg_days": sum(1 for x in pnls if x < 0),
        "n_days": n,
    }


def main():
    print("=" * 120)
    print("  FULL RISK-METRIC TABLE: ×0.82 → ×0.92 (+ blind baseline)")
    print("=" * 120)

    print("\nLoading data...")
    ml_favs = _load_ml_favorites(FULL_START, FULL_END)
    odds_raw = _load_alt_odds_raw(FULL_START, FULL_END)
    daily_cands = generate_daily_candidates(FULL_START, FULL_END, ml_favs, odds_raw)
    total = sum(len(v) for v in daily_cands.values())
    print(f"  {len(daily_cands)} days, {total} candidates")

    fine_mults = [1.0, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]

    for window_label, start, end in [("Full Period", FULL_START, FULL_END),
                                      ("Post-5/26", POST_START, FULL_END)]:
        print(f"\n{'='*120}")
        print(f"  {window_label}: {start} → {end}")
        print(f"{'='*120}")

        print(f"\n  {'Penalty':>8} {'N':>5} {'WR':>7} {'ROI':>8} {'DailyMean':>10} "
              f"{'DailyStd':>10} {'Sharpe':>8} {'MaxDD':>8} {'WorstDay':>10} "
              f"{'NegDays':>8} {'NegDay%':>8}")
        print(f"  {'─'*105}")

        results = []
        for em in fine_mults:
            rule = make_comp_only_rule(em)
            parlays = _run_config(daily_cands, start, end, rule)
            s = _pnl_row(parlays)
            daily = _daily_pnl(parlays)
            r = _risk_stats(daily)

            label = "blind" if em == 1.0 else f"×{em:.2f}"
            neg_pct = r["neg_days"] / r["n_days"] * 100 if r["n_days"] > 0 else 0

            print(f"  {label:>8} {s['n']:>5} {s['wr']:>5.1f}% {s['roi']:>+7.1f}% "
                  f"${r['mean']:>+8.2f} ${r['std']:>8.2f} {r['sharpe']:>8.3f} "
                  f"${r['max_dd']:>7.2f} ${r['worst_day']:>+8.2f} "
                  f"{r['neg_days']:>5}/{r['n_days']:<3} {neg_pct:>6.0f}%")

            results.append({"em": em, "label": label, **s, **r})

        # ── Rank each metric ──────────────────────────────────────
        # Higher is better: ROI, WR, Sharpe, mean
        # Lower is better: std, max_dd, neg_days, worst_day (more negative = worse)
        print(f"\n  ── RANK TABLE (1 = best) ──")
        # Exclude blind from ranking
        ranked = [r for r in results if r["em"] != 1.0]

        # Sort for each metric and assign ranks
        metrics_higher = ["roi", "wr", "sharpe", "mean"]
        metrics_lower = ["std", "max_dd", "neg_days"]

        # Build rank dict: em → {metric: rank}
        ranks = {r["em"]: {} for r in ranked}

        for m in metrics_higher:
            sorted_r = sorted(ranked, key=lambda x: -x[m])
            for i, r in enumerate(sorted_r):
                ranks[r["em"]][m] = i + 1

        for m in metrics_lower:
            sorted_r = sorted(ranked, key=lambda x: x[m])
            for i, r in enumerate(sorted_r):
                ranks[r["em"]][m] = i + 1

        all_metrics = metrics_higher + metrics_lower
        print(f"\n  {'Penalty':>8}", end="")
        for m in all_metrics:
            print(f" {m:>8}", end="")
        print(f" {'AvgRank':>8} {'WinsTop3':>9}")
        print(f"  {'─'*90}")

        for em in sorted(ranks.keys()):
            label = f"×{em:.2f}"
            r = ranks[em]
            avg_rank = sum(r[m] for m in all_metrics) / len(all_metrics)
            top3 = sum(1 for m in all_metrics if r[m] <= 3)
            print(f"  {label:>8}", end="")
            for m in all_metrics:
                rank = r[m]
                flag = "◆" if rank <= 3 else " "
                print(f" {rank:>6}{flag} ", end="")
            print(f" {avg_rank:>7.1f} {top3:>9}")

        # ── Best by each metric ──────────────────────────────────
        print(f"\n  ── BEST VALUE PER METRIC ──")
        for m in all_metrics:
            if m in metrics_higher:
                best = max(ranked, key=lambda x: x[m])
            else:
                best = min(ranked, key=lambda x: x[m])
            print(f"    {m:>8}: ×{best['em']:.2f} ({best[m]})")

        # ── Monotonicity check ────────────────────────────────────
        print(f"\n  ── MONOTONICITY CHECK (are risk metrics stable or bouncing?) ──")
        for m in all_metrics:
            vals = [r[m] for r in sorted(ranked, key=lambda x: x["em"])]
            # Count direction changes
            changes = 0
            for i in range(1, len(vals)):
                if i >= 2:
                    d1 = vals[i-1] - vals[i-2]
                    d2 = vals[i] - vals[i-1]
                    if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
                        changes += 1
            max_changes = len(vals) - 2
            bounce_pct = changes / max_changes * 100 if max_changes > 0 else 0
            stability = "STABLE" if bounce_pct < 40 else "MODERATE" if bounce_pct < 60 else "BOUNCY"
            print(f"    {m:>8}: {changes}/{max_changes} reversals ({bounce_pct:.0f}%) — {stability}")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  VERDICT")
    print(f"{'='*120}")
    print(f"\n  See tables above. Selection criterion:")
    print(f"  - If one value wins across-the-board on risk → pick it on principle")
    print(f"  - If risk metrics bounce → pick mid-range (×0.87) by default")


if __name__ == "__main__":
    main()
