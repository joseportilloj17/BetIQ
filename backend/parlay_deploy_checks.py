"""
parlay_deploy_checks.py — Pre-deploy checks for comp-only ×0.85 rule.

1. Variance/concentration: daily P&L variance + drawdown vs blind
2. Robustness: sweep ×0.82–×0.92 at fine increments to confirm stability
3. Full backtest: fav-count breakdown, within-band, daily P&L
"""
from __future__ import annotations
import os, sys, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parlay_assembly_backtest import (
    _load_ml_favorites, _load_alt_odds_raw, generate_daily_candidates,
    wilson_ci, STAKE, _classify_composition,
)
from parlay_penalty_sweep import (
    assemble_with_rule, make_comp_only_rule, _pnl_row, _thin,
    _is_extreme, _is_mixed,
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
    """Returns sorted list of (date, pnl, n_tickets)."""
    by_day = defaultdict(list)
    for p in parlays:
        by_day[p["game_date"]].append(p)
    result = []
    for gd in sorted(by_day.keys()):
        day_p = by_day[gd]
        pnl = sum(p["profit"] for p in day_p)
        result.append((gd, pnl, len(day_p)))
    return result


def _variance_stats(daily):
    """Compute variance, std, max drawdown, Sharpe-like ratio from daily P&L."""
    pnls = [d[1] for d in daily]
    n = len(pnls)
    if n < 2:
        return {"mean": 0, "std": 0, "sharpe": 0, "max_dd": 0, "worst_day": 0,
                "best_day": 0, "neg_days": 0, "pos_days": 0, "n_days": n}
    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / (n - 1)
    std = math.sqrt(var)
    sharpe = mean / std if std > 0 else 0

    # Max drawdown on cumulative P&L
    cum = 0
    peak = 0
    max_dd = 0
    for _, pnl, _ in daily:
        cum += pnl
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "mean": mean, "std": std, "sharpe": sharpe, "max_dd": max_dd,
        "worst_day": min(pnls), "best_day": max(pnls),
        "neg_days": sum(1 for x in pnls if x < 0),
        "pos_days": sum(1 for x in pnls if x > 0),
        "n_days": n,
    }


def main():
    print("=" * 110)
    print("  PRE-DEPLOY CHECKS: comp-only ×0.85")
    print("=" * 110)

    print("\nLoading data...")
    ml_favs = _load_ml_favorites(FULL_START, FULL_END)
    odds_raw = _load_alt_odds_raw(FULL_START, FULL_END)
    daily_cands = generate_daily_candidates(FULL_START, FULL_END, ml_favs, odds_raw)
    total = sum(len(v) for v in daily_cands.values())
    print(f"  {len(daily_cands)} days, {total} candidates")

    blind_rule = make_comp_only_rule(1.0)
    rule_85 = make_comp_only_rule(0.85)

    # ═══════════════════════════════════════════════════════════
    # CHECK 1: VARIANCE / CONCENTRATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  CHECK 1: VARIANCE / CONCENTRATION / DRAWDOWN")
    print(f"{'='*110}")

    for window_label, start, end in [("Full Period", FULL_START, FULL_END),
                                      ("Post-5/26", POST_START, FULL_END)]:
        blind_p = _run_config(daily_cands, start, end, blind_rule)
        rule_p  = _run_config(daily_cands, start, end, rule_85)

        blind_daily = _daily_pnl(blind_p)
        rule_daily  = _daily_pnl(rule_p)

        bv = _variance_stats(blind_daily)
        rv = _variance_stats(rule_daily)

        bs = _pnl_row(blind_p)
        rs = _pnl_row(rule_p)

        print(f"\n  ── {window_label} ──")
        print(f"  {'Metric':<25} {'Blind':>12} {'×0.85':>12} {'Delta':>12}")
        print(f"  {'─'*55}")
        print(f"  {'Total ROI':<25} {bs['roi']:>+11.1f}% {rs['roi']:>+11.1f}% {rs['roi']-bs['roi']:>+11.1f}pp")
        print(f"  {'Daily mean P&L':<25} ${bv['mean']:>+10.2f} ${rv['mean']:>+10.2f} ${rv['mean']-bv['mean']:>+10.2f}")
        print(f"  {'Daily std P&L':<25} ${bv['std']:>10.2f} ${rv['std']:>10.2f} ${rv['std']-bv['std']:>+10.2f}")
        print(f"  {'Sharpe (daily)':<25} {bv['sharpe']:>12.3f} {rv['sharpe']:>12.3f} {rv['sharpe']-bv['sharpe']:>+12.3f}")
        print(f"  {'Max drawdown':<25} ${bv['max_dd']:>10.2f} ${rv['max_dd']:>10.2f} ${rv['max_dd']-bv['max_dd']:>+10.2f}")
        print(f"  {'Worst day':<25} ${bv['worst_day']:>+10.2f} ${rv['worst_day']:>+10.2f}")
        print(f"  {'Best day':<25} ${bv['best_day']:>+10.2f} ${rv['best_day']:>+10.2f}")
        print(f"  {'Neg days / total':<25} {bv['neg_days']:>5}/{bv['n_days']:<5}    {rv['neg_days']:>5}/{rv['n_days']:<5}")
        print(f"  {'Pos days / total':<25} {bv['pos_days']:>5}/{bv['n_days']:<5}    {rv['pos_days']:>5}/{rv['n_days']:<5}")

    # Daily P&L side-by-side (post-5/26 only for readability)
    blind_post = _run_config(daily_cands, POST_START, FULL_END, blind_rule)
    rule_post  = _run_config(daily_cands, POST_START, FULL_END, rule_85)
    bd = _daily_pnl(blind_post)
    rd = _daily_pnl(rule_post)

    # Merge by date
    bd_dict = {d[0]: d for d in bd}
    rd_dict = {d[0]: d for d in rd}
    all_dates = sorted(set(d[0] for d in bd) | set(d[0] for d in rd))

    print(f"\n  ── Daily P&L: Post-5/26 ──")
    print(f"  {'Date':<12} {'Blind':>8} {'×0.85':>8} {'Winner':>8} {'B cum':>8} {'R cum':>8}")
    print(f"  {'─'*55}")
    b_cum = r_cum = 0
    b_wins = r_wins = ties = 0
    for gd in all_dates:
        bp = bd_dict.get(gd, (gd, 0, 0))[1]
        rp = rd_dict.get(gd, (gd, 0, 0))[1]
        b_cum += bp
        r_cum += rp
        if rp > bp: winner = "×0.85"; r_wins += 1
        elif bp > rp: winner = "BLIND"; b_wins += 1
        else: winner = "TIE"; ties += 1
        print(f"  {gd:<12} ${bp:>+7.2f} ${rp:>+7.2f} {winner:>8} ${b_cum:>+7.0f} ${r_cum:>+7.0f}")

    print(f"\n  Daily wins: Blind={b_wins}  ×0.85={r_wins}  Ties={ties}")

    # ═══════════════════════════════════════════════════════════
    # CHECK 2: ROBUSTNESS — fine sweep ×0.82 to ×0.92
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  CHECK 2: ROBUSTNESS — does ROI swing wildly near ×0.85?")
    print(f"{'='*110}")

    fine_mults = [0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]

    print(f"\n  {'Extreme×':>10} {'── Full Period ──':>25} {'── Post-5/26 ──':>25}")
    print(f"  {'':>10} {'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6}  "
          f"{'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6}")
    print(f"  {'─'*70}")

    full_rois = []
    post_rois = []

    for em in fine_mults:
        rule = make_comp_only_rule(em)
        full_p = _run_config(daily_cands, FULL_START, FULL_END, rule)
        post_p = _run_config(daily_cands, POST_START, FULL_END, rule)
        sf = _pnl_row(full_p)
        sp = _pnl_row(post_p)
        ext_f = sum(1 for p in full_p if _is_extreme(p["legs"], p["n_legs"]))
        ext_p = sum(1 for p in post_p if _is_extreme(p["legs"], p["n_legs"]))
        ef_pct = ext_f / sf["n"] * 100 if sf["n"] > 0 else 0
        ep_pct = ext_p / sp["n"] * 100 if sp["n"] > 0 else 0

        full_rois.append(sf["roi"])
        post_rois.append(sp["roi"])

        print(f"  ×{em:.2f}     {sf['n']:>5} {sf['wr']:>5.1f}% {sf['roi']:>+7.1f}% {ef_pct:>4.0f}%  "
              f"{sp['n']:>5} {sp['wr']:>5.1f}% {sp['roi']:>+7.1f}% {ep_pct:>4.0f}%")

    # Stability summary
    full_mean = sum(full_rois) / len(full_rois)
    full_std = math.sqrt(sum((x - full_mean)**2 for x in full_rois) / (len(full_rois) - 1))
    post_mean = sum(post_rois) / len(post_rois)
    post_std = math.sqrt(sum((x - post_mean)**2 for x in post_rois) / (len(post_rois) - 1))

    print(f"\n  Stability across ×0.82–×0.92:")
    print(f"    Full period: mean ROI={full_mean:+.1f}%, std={full_std:.1f}pp, "
          f"range=[{min(full_rois):+.1f}%, {max(full_rois):+.1f}%]")
    print(f"    Post-5/26:   mean ROI={post_mean:+.1f}%, std={post_std:.1f}pp, "
          f"range=[{min(post_rois):+.1f}%, {max(post_rois):+.1f}%]")

    knife_edge = post_std > 10
    if knife_edge:
        print(f"    ⚠ Post-5/26 std={post_std:.1f}pp > 10pp — KNIFE-EDGE sensitivity, overfitting risk")
    else:
        print(f"    ✓ Std < 10pp — rule is robust across this range")

    # ═══════════════════════════════════════════════════════════
    # CHECK 3: FULL BACKTEST — ×0.85 detail
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  CHECK 3: FULL BACKTEST DETAIL — comp-only ×0.85")
    print(f"{'='*110}")

    for window_label, start, end in [("Full Period", FULL_START, FULL_END),
                                      ("Post-5/26", POST_START, FULL_END)]:
        rule_p = _run_config(daily_cands, start, end, rule_85)
        s_all = _pnl_row(rule_p)

        print(f"\n  ── {window_label}: {s_all['n']} tickets, WR={s_all['wr']:.1f}%, "
              f"ROI={s_all['roi']:+.1f}% ──")

        # Fav-count breakdown by leg count
        for n_legs in [3, 2]:
            subset = [p for p in rule_p if p["n_legs"] == n_legs]
            if not subset:
                continue
            print(f"\n  {n_legs}-leg ({len(subset)} tickets):")
            print(f"  {'Fav':>4} {'Dog':>4} {'N':>5} {'WR':>7} {'CI 95%':>18} "
                  f"{'Avg Odds':>10} {'ROI':>9}")
            print(f"  {'─'*65}")
            for fc in range(n_legs + 1):
                dc = n_legs - fc
                g = [p for p in subset if p["fav_count"] == fc]
                s = _pnl_row(g)
                if s["n"] == 0:
                    print(f"  {fc:>4} {dc:>4}     0       —                  —          —       —")
                    continue
                lo, hi = wilson_ci(s["n"], s["wins"])
                print(f"  {fc:>4} {dc:>4} {s['n']:>5} {s['wr']:>5.1f}% "
                      f"[{lo:>5.1f}%,{hi:>5.1f}%] {s['avg_odds']:>10.3f} "
                      f"{s['roi']:>+8.1f}%{_thin(s['n'])}")

        # Within-quality-band
        if len(rule_p) >= 9:
            los_vals = sorted(p["avg_los"] for p in rule_p)
            n = len(los_vals)
            t1 = los_vals[n // 3]
            t2 = los_vals[2 * n // 3]
            bands = [
                ("Low LOS",  lambda p, _t1=t1: p["avg_los"] <= _t1),
                ("Mid LOS",  lambda p, _t1=t1, _t2=t2: _t1 < p["avg_los"] <= _t2),
                ("High LOS", lambda p, _t2=t2: p["avg_los"] > _t2),
            ]
            print(f"\n  Within-band (LOS cuts: ≤{t1:.4f} | ≤{t2:.4f} | >{t2:.4f}):")
            for bname, bfn in bands:
                bp = [p for p in rule_p if bfn(p)]
                print(f"\n    {bname} ({len(bp)} parlays):")
                print(f"    {'Fav#':>5} {'N':>5} {'WR':>7} {'ROI':>9} {'AvgOdds':>9} {'AvgLOS':>8}")
                print(f"    {'─'*50}")
                for fc in sorted(set(p["fav_count"] for p in bp)):
                    g = [p for p in bp if p["fav_count"] == fc]
                    s = _pnl_row(g)
                    avg_los_g = sum(p["avg_los"] for p in g) / len(g)
                    print(f"    {fc:>5} {s['n']:>5} {s['wr']:>5.1f}% {s['roi']:>+8.1f}% "
                          f"{s['avg_odds']:>9.3f} {avg_los_g:>7.4f}{_thin(s['n'])}")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    blind_full = _run_config(daily_cands, FULL_START, FULL_END, blind_rule)
    blind_post = _run_config(daily_cands, POST_START, FULL_END, blind_rule)
    rule_full  = _run_config(daily_cands, FULL_START, FULL_END, rule_85)
    rule_post  = _run_config(daily_cands, POST_START, FULL_END, rule_85)

    bf = _pnl_row(blind_full); bp = _pnl_row(blind_post)
    rf = _pnl_row(rule_full);  rp = _pnl_row(rule_post)

    bfv = _variance_stats(_daily_pnl(blind_post))
    rfv = _variance_stats(_daily_pnl(rule_post))

    print(f"\n  {'':20} {'Blind':>12} {'×0.85':>12} {'Delta':>12}")
    print(f"  {'─'*55}")
    print(f"  {'Full ROI':<20} {bf['roi']:>+11.1f}% {rf['roi']:>+11.1f}% {rf['roi']-bf['roi']:>+11.1f}pp")
    print(f"  {'Post-5/26 ROI':<20} {bp['roi']:>+11.1f}% {rp['roi']:>+11.1f}% {rp['roi']-bp['roi']:>+11.1f}pp")
    print(f"  {'Post Sharpe':<20} {bfv['sharpe']:>12.3f} {rfv['sharpe']:>12.3f} {rfv['sharpe']-bfv['sharpe']:>+12.3f}")
    print(f"  {'Post MaxDD':<20} ${bfv['max_dd']:>10.2f} ${rfv['max_dd']:>10.2f} ${rfv['max_dd']-bfv['max_dd']:>+10.2f}")
    print(f"  {'Robust (×0.82-0.92)':<20} {'std=' + f'{post_std:.1f}pp':>12}")

    if rfv["sharpe"] >= bfv["sharpe"] and not knife_edge:
        print(f"\n  ✓ DEPLOY: ×0.85 beats blind on ROI AND risk-adjusted. Rule is robust. Ready for Shadow B.")
    elif rfv["sharpe"] < bfv["sharpe"]:
        print(f"\n  ⚠ CAUTION: ×0.85 has better ROI but worse risk-adjusted return. Review drawdown before deploy.")
    elif knife_edge:
        print(f"\n  ⚠ CAUTION: rule is knife-edge sensitive to penalty value. Overfitting risk.")
    else:
        print(f"\n  ✓ DEPLOY: ready for Shadow B forward confirmation.")


if __name__ == "__main__":
    main()
