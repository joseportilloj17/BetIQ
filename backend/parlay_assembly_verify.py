"""
parlay_assembly_verify.py — Verify the blind-vs-polarized backtest result.

1. Confirm timeframe is post-5/26
2. Explicit fav-leg-count breakdown (0/1/2/3) with N + WR + ROI
3. Within-quality-band composition test: does fav count predict ROI
   when avg LOS is held roughly constant?

Reuses generate_daily_candidates / assemble_parlays from parlay_assembly_backtest.
"""
from __future__ import annotations
import os, sys, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parlay_assembly_backtest import (
    _load_ml_favorites, _load_alt_odds_raw, generate_daily_candidates,
    assemble_parlays, STAKE, START, END, MAX_PARLAYS_PER_DAY,
    wilson_ci,
)


def _pnl_row(parlays: list) -> dict:
    n = len(parlays)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0, "roi": 0, "avg_odds": 0, "pnl": 0,
                "avg_los": 0, "avg_lqs": 0}
    wins = sum(1 for p in parlays if p["won"])
    pnl = sum(p["profit"] for p in parlays)
    staked = n * STAKE
    avg_odds = sum(p["combined_odds"] for p in parlays) / n
    avg_los = sum(p["avg_los"] for p in parlays) / n
    # avg LQS across all legs
    all_lqs = [l["lqs"] for p in parlays for l in p["legs"]]
    avg_lqs = sum(all_lqs) / len(all_lqs) if all_lqs else 0
    return {
        "n": n, "wins": wins, "wr": wins / n * 100,
        "roi": pnl / staked * 100, "avg_odds": avg_odds,
        "pnl": pnl, "avg_los": avg_los, "avg_lqs": avg_lqs,
    }


def _thin(n):
    if n < 10: return " ⚠THIN"
    if n < 20: return " ⚠thin"
    return ""


def main():
    print("=" * 100)
    print("  PARLAY ASSEMBLY VERIFICATION")
    print(f"  Window: {START} → {END}")
    print("=" * 100)

    # ── 1. TIMEFRAME ────────────────────────────────────────────
    print(f"\n  1. TIMEFRAME CHECK")
    print(f"     START = {START}, END = {END}")
    if START >= "2026-05-26":
        print(f"     ✓ Post-5/26 regime only. NOT contaminated by pre-5/26 dog data.")
    else:
        print(f"     ✗ FULL PERIOD — includes pre-5/26 broken regime. Results may be contaminated.")
        print(f"       Need to re-run with START='2026-05-26'.")

    print("\n  Loading data...")
    ml_favs = _load_ml_favorites(START, END)
    odds_raw = _load_alt_odds_raw(START, END)
    daily_cands = generate_daily_candidates(START, END, ml_favs, odds_raw)
    total_cands = sum(len(v) for v in daily_cands.values())
    print(f"  {len(daily_cands)} days, {total_cands} candidates")

    # Assemble under both strategies
    all_blind = []
    all_polar = []
    for gd in sorted(daily_cands.keys()):
        cands = daily_cands[gd]
        bp = assemble_parlays(cands, "blind", MAX_PARLAYS_PER_DAY)
        pp = assemble_parlays(cands, "polarized", MAX_PARLAYS_PER_DAY)
        for p in bp: p["game_date"] = gd
        for p in pp: p["game_date"] = gd
        all_blind.extend(bp)
        all_polar.extend(pp)

    print(f"  Blind: {len(all_blind)} tickets, Polarized: {len(all_polar)} tickets")

    # ── 2. EXPLICIT FAV-LEG-COUNT BREAKDOWN ────────────────────
    # 3-leg parlays: fav count 0,1,2,3
    # 2-leg parlays: fav count 0,1,2
    print(f"\n{'='*100}")
    print(f"  2. EXPLICIT FAV-LEG-COUNT BREAKDOWN  (N / WR / ROI)")
    print(f"{'='*100}")

    for label, group in [("BLIND", all_blind), ("POLARIZED", all_polar)]:
        for n_legs in [3, 2]:
            subset = [p for p in group if p["n_legs"] == n_legs]
            if not subset:
                continue
            print(f"\n  {label} — {n_legs}-leg parlays ({len(subset)} tickets):")
            print(f"  {'Fav Legs':<12} {'Dog Legs':<10} {'N':>5} {'WR':>7} {'CI 95%':>18} "
                  f"{'Avg Odds':>10} {'ROI':>8} {'P&L':>9} {'Avg LOS':>8}")
            print(f"  {'─'*90}")

            for fav_n in range(n_legs + 1):
                dog_n = n_legs - fav_n
                g = [p for p in subset if p["fav_count"] == fav_n]
                if not g:
                    print(f"  {fav_n} fav       {dog_n} dog       {'0':>5}   {'—':>5}   {'—':>16}   "
                          f"{'—':>8}   {'—':>6}   {'—':>7}   {'—':>6}")
                    continue
                s = _pnl_row(g)
                lo, hi = wilson_ci(s["n"], s["wins"])
                print(f"  {fav_n} fav       {dog_n} dog    {s['n']:>5} {s['wr']:>5.1f}% "
                      f"[{lo:>5.1f}%,{hi:>5.1f}%] {s['avg_odds']:>10.3f} "
                      f"{s['roi']:>+7.1f}% ${s['pnl']:>+8.2f} {s['avg_los']:>7.4f}{_thin(s['n'])}")

    # ── 3. WITHIN-QUALITY-BAND COMPOSITION TEST ────────────────
    # Bucket blind parlays by avg LOS tercile, then check fav count within band
    print(f"\n{'='*100}")
    print(f"  3. WITHIN-QUALITY-BAND COMPOSITION TEST")
    print(f"     Question: holding avg LOS constant, does fav count still predict ROI?")
    print(f"{'='*100}")

    # Use blind only (polarized has distorted composition mix)
    # Compute avg LOS per parlay
    los_values = [p["avg_los"] for p in all_blind]
    los_values.sort()

    if len(los_values) < 6:
        print("\n  Not enough data for quality-band analysis.")
        return

    # Split into terciles
    n_total = len(los_values)
    t1_cut = los_values[n_total // 3]
    t2_cut = los_values[2 * n_total // 3]

    bands = [
        ("Low LOS",  lambda p: p["avg_los"] <= t1_cut),
        ("Mid LOS",  lambda p: t1_cut < p["avg_los"] <= t2_cut),
        ("High LOS", lambda p: p["avg_los"] > t2_cut),
    ]

    print(f"\n  LOS tercile cuts: ≤{t1_cut:.4f} | ≤{t2_cut:.4f} | >{t2_cut:.4f}")
    print(f"  (Using BLIND parlays only — polarized has distorted composition mix)")

    for band_name, band_fn in bands:
        band_parlays = [p for p in all_blind if band_fn(p)]
        print(f"\n  ── {band_name} band ({len(band_parlays)} parlays, "
              f"avg LOS range {min(p['avg_los'] for p in band_parlays):.4f} – "
              f"{max(p['avg_los'] for p in band_parlays):.4f}) ──")

        # Group by fav-majority vs dog-majority vs balanced
        groups = {
            "fav-majority": [p for p in band_parlays if p["fav_count"] > p["n_legs"] / 2],
            "balanced":     [p for p in band_parlays if p["fav_count"] == p["n_legs"] / 2],
            "dog-majority": [p for p in band_parlays if p["fav_count"] < p["n_legs"] / 2],
        }

        print(f"  {'Group':<15} {'N':>5} {'WR':>7} {'ROI':>8} {'Avg Odds':>10} {'Avg LOS':>8} {'Avg LQS':>8}")
        print(f"  {'─'*65}")
        for gname, gdata in groups.items():
            if not gdata:
                print(f"  {gname:<15} {'0':>5}   {'—':>5}   {'—':>6}   {'—':>8}   {'—':>6}   {'—':>6}")
                continue
            s = _pnl_row(gdata)
            print(f"  {gname:<15} {s['n']:>5} {s['wr']:>5.1f}% {s['roi']:>+7.1f}% "
                  f"{s['avg_odds']:>10.3f} {s['avg_los']:>7.4f} {s['avg_lqs']:>7.1f}{_thin(s['n'])}")

    # ── 3b. Finer: by explicit fav count within each band ──────
    print(f"\n  ── 3b. Finer breakdown: explicit fav count within each LOS band ──")
    for band_name, band_fn in bands:
        band_parlays = [p for p in all_blind if band_fn(p)]
        print(f"\n  {band_name} ({len(band_parlays)} parlays):")
        print(f"  {'Fav#':>5} {'N':>5} {'WR':>7} {'ROI':>8} {'Avg Odds':>10} {'Avg LOS':>8}")
        print(f"  {'─'*50}")
        for fc in sorted(set(p["fav_count"] for p in band_parlays)):
            g = [p for p in band_parlays if p["fav_count"] == fc]
            s = _pnl_row(g)
            print(f"  {fc:>5} {s['n']:>5} {s['wr']:>5.1f}% {s['roi']:>+7.1f}% "
                  f"{s['avg_odds']:>10.3f} {s['avg_los']:>7.4f}{_thin(s['n'])}")

    # ── 4. QUALITY CORRELATION CHECK ───────────────────────────
    # Is avg LOS correlated with fav count? (confirms/denies confound)
    print(f"\n{'='*100}")
    print(f"  4. QUALITY-COMPOSITION CORRELATION")
    print(f"     Is avg LOS correlated with fav count? (confound check)")
    print(f"{'='*100}")

    # Group blind parlays by fav count, show avg LOS
    fav_counts = sorted(set(p["fav_count"] for p in all_blind))
    print(f"\n  {'Fav#':>5} {'N':>5} {'Avg LOS':>10} {'Avg LQS':>10} {'Avg WP':>10} {'Avg Composite':>15}")
    print(f"  {'─'*60}")
    for fc in fav_counts:
        g = [p for p in all_blind if p["fav_count"] == fc]
        avg_los = sum(p["avg_los"] for p in g) / len(g)
        all_lqs = [l["lqs"] for p in g for l in p["legs"]]
        avg_lqs = sum(all_lqs) / len(all_lqs) if all_lqs else 0
        avg_wp = sum(p["combined_wp"] for p in g) / len(g)
        avg_comp = sum(p["composite"] for p in g) / len(g)
        print(f"  {fc:>5} {len(g):>5} {avg_los:>10.4f} {avg_lqs:>10.1f} {avg_wp:>10.4f} {avg_comp:>15.4f}")

    # ── VERDICT ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  VERDICT")
    print(f"{'='*100}")
    print(f"\n  Window: {START} → {END} (post-5/26 ✓)")
    sb = _pnl_row(all_blind)
    sp = _pnl_row(all_polar)
    print(f"  Blind:     {sb['n']} tickets, WR={sb['wr']:.1f}%, ROI={sb['roi']:+.1f}%")
    print(f"  Polarized: {sp['n']} tickets, WR={sp['wr']:.1f}%, ROI={sp['roi']:+.1f}%")
    print(f"  Delta:     {sp['roi'] - sb['roi']:+.1f}pp")

    # Summarize within-band findings
    print(f"\n  Within-band composition test:")
    for band_name, band_fn in bands:
        bp = [p for p in all_blind if band_fn(p)]
        fav_maj = [p for p in bp if p["fav_count"] > p["n_legs"] / 2]
        dog_maj = [p for p in bp if p["fav_count"] < p["n_legs"] / 2]
        sf = _pnl_row(fav_maj)
        sd = _pnl_row(dog_maj)
        fav_roi = f"{sf['roi']:+.1f}%" if sf['n'] > 0 else "—"
        dog_roi = f"{sd['roi']:+.1f}%" if sd['n'] > 0 else "—"
        print(f"    {band_name:>10}: fav-maj ROI={fav_roi} (n={sf['n']}), "
              f"dog-maj ROI={dog_roi} (n={sd['n']})")


if __name__ == "__main__":
    main()
