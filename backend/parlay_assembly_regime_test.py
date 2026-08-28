"""
parlay_assembly_regime_test.py — Regime stability test for mixed>extremes finding.

Runs blind assembly across:
  - FULL period (Apr 22 – Jun 26)
  - Pre-5/26 only (Apr 22 – May 25)
  - Post-5/26 only (May 26 – Jun 26)

For each window:
  1. Explicit fav-count breakdown (0/1/2/3) with N, WR, CI, ROI
  2. Within-quality-band composition test (LOS terciles × fav count)

Then pools full-period data for hardened within-band CIs.
"""
from __future__ import annotations
import os, sys, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parlay_assembly_backtest import (
    _load_ml_favorites, _load_alt_odds_raw, generate_daily_candidates,
    assemble_parlays, wilson_ci, STAKE,
)

MAX_PARLAYS_PER_DAY = 8
REGIME_CUT = "2026-05-26"


def _pnl_row(parlays):
    n = len(parlays)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0, "roi": 0, "avg_odds": 0,
                "pnl": 0, "avg_los": 0, "avg_lqs": 0}
    wins = sum(1 for p in parlays if p["won"])
    pnl = sum(p["profit"] for p in parlays)
    staked = n * STAKE
    all_lqs = [l["lqs"] for p in parlays for l in p["legs"]]
    return {
        "n": n, "wins": wins, "wr": wins / n * 100,
        "roi": pnl / staked * 100,
        "avg_odds": sum(p["combined_odds"] for p in parlays) / n,
        "pnl": pnl, "avg_los": sum(p["avg_los"] for p in parlays) / n,
        "avg_lqs": sum(all_lqs) / len(all_lqs) if all_lqs else 0,
    }


def _thin(n):
    if n < 10: return " ⚠THIN"
    if n < 20: return " ⚠thin"
    return ""


def _build_blind(daily_cands, start, end):
    """Assemble blind parlays for a date range."""
    parlays = []
    for gd in sorted(daily_cands.keys()):
        if gd < start or gd > end:
            continue
        bp = assemble_parlays(daily_cands[gd], "blind", MAX_PARLAYS_PER_DAY)
        for p in bp:
            p["game_date"] = gd
        parlays.extend(bp)
    return parlays


def _print_fav_breakdown(parlays, label):
    """Print fav-count breakdown for 3-leg and 2-leg parlays."""
    for n_legs in [3, 2]:
        subset = [p for p in parlays if p["n_legs"] == n_legs]
        if not subset:
            continue
        print(f"\n  {label} — {n_legs}-leg ({len(subset)} tickets):")
        print(f"  {'Fav':>4} {'Dog':>4} {'N':>5} {'WR':>7} {'CI 95%':>18} "
              f"{'Avg Odds':>10} {'ROI':>9} {'P&L':>10}")
        print(f"  {'─'*80}")
        for fav_n in range(n_legs + 1):
            dog_n = n_legs - fav_n
            g = [p for p in subset if p["fav_count"] == fav_n]
            s = _pnl_row(g)
            if s["n"] == 0:
                print(f"  {fav_n:>4} {dog_n:>4} {'0':>5}     —                  —"
                      f"          —       —          —")
                continue
            lo, hi = wilson_ci(s["n"], s["wins"])
            print(f"  {fav_n:>4} {dog_n:>4} {s['n']:>5} {s['wr']:>5.1f}% "
                  f"[{lo:>5.1f}%,{hi:>5.1f}%] {s['avg_odds']:>10.3f} "
                  f"{s['roi']:>+8.1f}% ${s['pnl']:>+9.2f}{_thin(s['n'])}")


def _print_within_band(parlays, label):
    """Within-quality-band test: LOS terciles × fav count."""
    if len(parlays) < 9:
        print(f"\n  {label}: too few parlays ({len(parlays)}) for within-band test")
        return

    los_vals = sorted(p["avg_los"] for p in parlays)
    n = len(los_vals)
    t1 = los_vals[n // 3]
    t2 = los_vals[2 * n // 3]

    bands = [
        ("Low LOS",  lambda p: p["avg_los"] <= t1),
        ("Mid LOS",  lambda p: t1 < p["avg_los"] <= t2),
        ("High LOS", lambda p: p["avg_los"] > t2),
    ]

    print(f"\n  {label} — Within-band test (LOS cuts: ≤{t1:.4f} | ≤{t2:.4f} | >{t2:.4f}):")
    for band_name, band_fn in bands:
        bp = [p for p in parlays if band_fn(p)]
        if not bp:
            continue
        print(f"\n    {band_name} ({len(bp)} parlays):")
        print(f"    {'Fav#':>5} {'N':>5} {'WR':>7} {'CI 95%':>18} {'ROI':>9} "
              f"{'Avg Odds':>10} {'Avg LOS':>8}")
        print(f"    {'─'*72}")
        for fc in sorted(set(p["fav_count"] for p in bp)):
            g = [p for p in bp if p["fav_count"] == fc]
            s = _pnl_row(g)
            lo, hi = wilson_ci(s["n"], s["wins"])
            print(f"    {fc:>5} {s['n']:>5} {s['wr']:>5.1f}% "
                  f"[{lo:>5.1f}%,{hi:>5.1f}%] {s['roi']:>+8.1f}% "
                  f"{s['avg_odds']:>10.3f} {s['avg_los']:>7.4f}{_thin(s['n'])}")


def main():
    FULL_START = "2026-04-22"
    FULL_END   = "2026-06-26"

    print("=" * 100)
    print("  REGIME STABILITY TEST: mixed > extremes?")
    print(f"  Full: {FULL_START} → {FULL_END}")
    print(f"  Pre-5/26: {FULL_START} → 2026-05-25")
    print(f"  Post-5/26: {REGIME_CUT} → {FULL_END}")
    print("=" * 100)

    print("\nLoading data for full period...")
    ml_favs = _load_ml_favorites(FULL_START, FULL_END)
    odds_raw = _load_alt_odds_raw(FULL_START, FULL_END)
    print(f"  {len(ml_favs)} fixtures, {len(odds_raw)} odds entries")

    print("Generating daily candidates (full period)...")
    daily_cands = generate_daily_candidates(FULL_START, FULL_END, ml_favs, odds_raw)
    total = sum(len(v) for v in daily_cands.values())
    print(f"  {len(daily_cands)} days, {total} total candidates")

    # Build blind parlays for each window
    full_blind  = _build_blind(daily_cands, FULL_START, FULL_END)
    pre_blind   = _build_blind(daily_cands, FULL_START, "2026-05-25")
    post_blind  = _build_blind(daily_cands, REGIME_CUT, FULL_END)

    print(f"\n  Full:     {len(full_blind)} tickets")
    print(f"  Pre-5/26: {len(pre_blind)} tickets")
    print(f"  Post-5/26:{len(post_blind)} tickets")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: Fav-count breakdown for each regime
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  1. FAV-COUNT BREAKDOWN BY REGIME")
    print(f"{'='*100}")

    for label, group in [("FULL PERIOD", full_blind),
                         ("PRE-5/26", pre_blind),
                         ("POST-5/26", post_blind)]:
        s = _pnl_row(group)
        print(f"\n  {'─'*90}")
        print(f"  {label}: {s['n']} tickets, WR={s['wr']:.1f}%, ROI={s['roi']:+.1f}%")
        print(f"  {'─'*90}")
        _print_fav_breakdown(group, label)

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: Regime comparison — does mixed>extremes hold?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  2. REGIME COMPARISON: mixed > extremes in both?")
    print(f"{'='*100}")

    for label, group in [("PRE-5/26", pre_blind), ("POST-5/26", post_blind)]:
        three = [p for p in group if p["n_legs"] == 3]
        if not three:
            continue
        extremes = [p for p in three if p["fav_count"] in (0, len(p["legs"]))]
        mixed = [p for p in three if 0 < p["fav_count"] < len(p["legs"])]
        se = _pnl_row(extremes)
        sm = _pnl_row(mixed)
        print(f"\n  {label} 3-leg:")
        print(f"    Extremes (0/3, 3/3): N={se['n']}, WR={se['wr']:.1f}%, ROI={se['roi']:+.1f}%")
        print(f"    Mixed (1/3, 2/3):    N={sm['n']}, WR={sm['wr']:.1f}%, ROI={sm['roi']:+.1f}%")
        if se['n'] > 0 and sm['n'] > 0:
            delta = sm['roi'] - se['roi']
            print(f"    Delta: mixed is {delta:+.1f}pp vs extremes")
            if delta > 0:
                print(f"    → mixed > extremes ✓")
            else:
                print(f"    → extremes > mixed ✗ (finding FLIPS in this regime)")

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: Within-quality-band — per regime
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  3. WITHIN-QUALITY-BAND BY REGIME")
    print(f"{'='*100}")

    for label, group in [("PRE-5/26", pre_blind), ("POST-5/26", post_blind)]:
        _print_within_band(group, label)

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: HARDENED within-band — full period (more N)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  4. HARDENED WITHIN-BAND: full period pooled ({len(full_blind)} tickets)")
    print(f"     Question: with ~2x the N, does mixed>extremes survive in each band?")
    print(f"{'='*100}")

    _print_within_band(full_blind, "FULL PERIOD")

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: Summary table
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  5. SUMMARY: mixed>extremes stability check")
    print(f"{'='*100}")

    print(f"\n  {'Window':<14} {'Mixed ROI':>10} {'Extreme ROI':>12} {'Delta':>8} {'Mixed N':>8} {'Extr N':>7} {'Verdict':>10}")
    print(f"  {'─'*75}")

    for label, group in [("Full", full_blind), ("Pre-5/26", pre_blind), ("Post-5/26", post_blind)]:
        three = [p for p in group if p["n_legs"] == 3]
        extremes = [p for p in three if p["fav_count"] in (0, len(p["legs"]))]
        mixed = [p for p in three if 0 < p["fav_count"] < len(p["legs"])]
        se = _pnl_row(extremes)
        sm = _pnl_row(mixed)
        delta = sm["roi"] - se["roi"] if se["n"] > 0 and sm["n"] > 0 else 0
        verdict = "mixed>ext ✓" if delta > 0 else "ext>mixed ✗"
        print(f"  {label:<14} {sm['roi']:>+9.1f}% {se['roi']:>+11.1f}% {delta:>+7.1f}pp "
              f"{sm['n']:>8} {se['n']:>7}   {verdict}")

    # 2-leg summary
    print(f"\n  2-leg parlays:")
    print(f"  {'Window':<14} {'Mixed(1/2)':>12} {'Ext(0/2+2/2)':>14} {'Delta':>8} {'M N':>5} {'E N':>5}")
    print(f"  {'─'*60}")
    for label, group in [("Full", full_blind), ("Pre-5/26", pre_blind), ("Post-5/26", post_blind)]:
        two = [p for p in group if p["n_legs"] == 2]
        ext2 = [p for p in two if p["fav_count"] in (0, 2)]
        mix2 = [p for p in two if p["fav_count"] == 1]
        se2 = _pnl_row(ext2)
        sm2 = _pnl_row(mix2)
        delta2 = sm2["roi"] - se2["roi"] if se2["n"] > 0 and sm2["n"] > 0 else 0
        print(f"  {label:<14} {sm2['roi']:>+11.1f}% {se2['roi']:>+13.1f}% {delta2:>+7.1f}pp "
              f"{sm2['n']:>5} {se2['n']:>5}")


if __name__ == "__main__":
    main()
