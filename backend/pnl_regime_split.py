"""
pnl_regime_split.py — Regime-split parlay P&L: pre-5/26 vs post-5/26.

Tests whether dog-majority -17.6% ROI on full period is contaminated by
the worse pre-5/26 system, or is a true property of dog picks.
"""
from __future__ import annotations
import os, sys, math

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pnl_parlay_audit import (
    _load_ml_favorites, load_parlays, _pnl_stats, wilson_ci,
    _american_to_decimal, STAKE,
)

FULL_START = "2026-04-22"
FULL_END   = "2026-06-26"
REGIME_CUT = "2026-05-26"  # post = >= this date


def _thin_flag(n: int) -> str:
    if n < 10:
        return " ⚠THIN"
    elif n < 20:
        return " ⚠thin"
    return ""


def _print_comp_table(parlays: list, label: str):
    """Print composition ROI table for a set of parlays."""
    comps = ["all-fav", "fav-heavy", "even-split", "dog-heavy", "all-dog"]
    print(f"\n  {label}")
    print(f"  {'Comp':<14} {'N':>5} {'WR':>7} {'CI 95%':>18} {'Avg Odds':>10} "
          f"{'P&L':>9} {'ROI':>8}")
    print(f"  {'─'*75}")
    for comp in comps:
        group = [p for p in parlays if p["composition"] == comp]
        if not group:
            continue
        s = _pnl_stats(group)
        lo, hi = wilson_ci(s["n"], s["wins"])
        flag = _thin_flag(s["n"])
        print(f"  {comp:<14} {s['n']:>5} {s['wr']:>5.1f}% [{lo:>5.1f}%,{hi:>5.1f}%] "
              f"{s['avg_odds']:>10.3f} ${s['pnl']:>+8.2f} {s['roi']:>+7.1f}%{flag}")

    fav_maj = [p for p in parlays if p["fav_fraction"] > 0.5]
    dog_maj = [p for p in parlays if p["fav_fraction"] < 0.5]
    sf = _pnl_stats(fav_maj)
    sd = _pnl_stats(dog_maj)
    fl, fh = wilson_ci(sf["n"], sf["wins"])
    dl, dh = wilson_ci(sd["n"], sd["wins"])
    print(f"\n  Fav-majority: n={sf['n']:>4}  WR={sf['wr']:>5.1f}% [{fl}%,{fh}%]  "
          f"odds={sf['avg_odds']:.3f}  ROI={sf['roi']:>+6.1f}%{_thin_flag(sf['n'])}")
    print(f"  Dog-majority: n={sd['n']:>4}  WR={sd['wr']:>5.1f}% [{dl}%,{dh}%]  "
          f"odds={sd['avg_odds']:.3f}  ROI={sd['roi']:>+6.1f}%{_thin_flag(sd['n'])}")
    return sf, sd


def _print_fav_count_table(parlays: list, n_legs: int, label: str):
    """Print fav-count ROI table for parlays of a specific leg count."""
    subset = [p for p in parlays if p["n_legs"] == n_legs]
    if not subset:
        return
    print(f"\n  {n_legs}-leg parlays: {len(subset)} tickets ({label})")
    print(f"  {'Fav Legs':<10} {'N':>5} {'WR':>7} {'CI 95%':>18} {'Avg Odds':>10} "
          f"{'P&L':>9} {'ROI':>8}")
    print(f"  {'─'*72}")
    for fav_n in range(n_legs + 1):
        group = [p for p in subset if p["fav_count"] == fav_n]
        if not group:
            continue
        s = _pnl_stats(group)
        lo, hi = wilson_ci(s["n"], s["wins"])
        flag = _thin_flag(s["n"])
        print(f"  {fav_n} of {n_legs}     {s['n']:>5} {s['wr']:>5.1f}% [{lo:>5.1f}%,{hi:>5.1f}%] "
              f"{s['avg_odds']:>10.3f} ${s['pnl']:>+8.2f} {s['roi']:>+7.1f}%{flag}")


def _bust_rate_analysis(parlays: list, label: str):
    """Print bust rate by fav/dog and by quadrant."""
    all_legs = [l for p in parlays for l in p["legs"]]
    if not all_legs:
        print(f"\n  Bust rate ({label}): no legs")
        return

    fav_legs = [l for l in all_legs if l["is_pick_fav"]]
    dog_legs = [l for l in all_legs if not l["is_pick_fav"]]

    fav_wins = sum(1 for l in fav_legs if l["leg_result"] == "WIN")
    dog_wins = sum(1 for l in dog_legs if l["leg_result"] == "WIN")

    fav_bust = (len(fav_legs) - fav_wins) / len(fav_legs) * 100 if fav_legs else 0
    dog_bust = (len(dog_legs) - dog_wins) / len(dog_legs) * 100 if dog_legs else 0

    fav_bl, fav_bh = wilson_ci(len(fav_legs), len(fav_legs) - fav_wins)
    dog_bl, dog_bh = wilson_ci(len(dog_legs), len(dog_legs) - dog_wins)

    print(f"\n  Bust rate ({label}):")
    print(f"    Fav legs: {fav_bust:>5.1f}% bust ({len(fav_legs):>3} legs)  CI [{fav_bl}%,{fav_bh}%]")
    print(f"    Dog legs: {dog_bust:>5.1f}% bust ({len(dog_legs):>3} legs)  CI [{dog_bl}%,{dog_bh}%]")
    gap = dog_bust - fav_bust
    print(f"    Gap: {gap:+.1f}pp", end="")
    if fav_bh < dog_bl:
        print(f"  → SIGNIFICANT (CIs don't overlap)")
    elif dog_bh < fav_bl:
        print(f"  → REVERSED & SIGNIFICANT")
    else:
        print(f"  → CIs overlap (not significant)")

    # Per-quadrant
    print(f"\n    Per-quadrant bust rates ({label}):")
    print(f"    {'Quadrant':<12} {'N':>5} {'Bust%':>7} {'CI':>18}")
    print(f"    {'─'*46}")
    for quad in ["Fav:Home", "Fav:Away", "Dog:Home", "Dog:Away"]:
        qlegs = [l for l in all_legs if l["quadrant"] == quad]
        if not qlegs:
            continue
        wins = sum(1 for l in qlegs if l["leg_result"] == "WIN")
        bust = len(qlegs) - wins
        bl, bh = wilson_ci(len(qlegs), bust)
        print(f"    {quad:<12} {len(qlegs):>5} {bust/len(qlegs)*100:>5.1f}% "
              f"[{bl:>5.1f}%,{bh:>5.1f}%]{_thin_flag(len(qlegs))}")

    return fav_bust, dog_bust, gap


def main():
    print("=" * 95)
    print("  REGIME-SPLIT PARLAY P&L: PRE vs POST 5/26")
    print("  Question: is dog-majority -17.6% full-period ROI contaminated")
    print("  by the worse pre-5/26 system, or a true property of dog picks?")
    print("=" * 95)

    ml_favs = _load_ml_favorites(FULL_START, FULL_END)
    print(f"\n  ML favorites: {len(ml_favs)} fixtures")

    all_parlays = load_parlays(ml_favs, FULL_START, FULL_END)
    print(f"  Total classified parlays: {len(all_parlays)}")

    pre = [p for p in all_parlays if p["game_date"] < REGIME_CUT]
    post = [p for p in all_parlays if p["game_date"] >= REGIME_CUT]
    print(f"\n  Pre-5/26:  {len(pre)} parlays ({min(p['game_date'] for p in pre) if pre else '?'} "
          f"to {max(p['game_date'] for p in pre) if pre else '?'})")
    print(f"  Post-5/26: {len(post)} parlays ({min(p['game_date'] for p in post) if post else '?'} "
          f"to {max(p['game_date'] for p in post) if post else '?'})")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Composition ROI by regime
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  1. COMPOSITION ROI BY REGIME")
    print(f"{'='*95}")

    sf_pre, sd_pre = _print_comp_table(pre, "PRE-5/26 (old system)")
    sf_post, sd_post = _print_comp_table(post, "POST-5/26 (current system)")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: 3-leg and 2-leg by fav count, per regime
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  2. FAV-COUNT BREAKDOWN BY REGIME")
    print(f"{'='*95}")

    _print_fav_count_table(pre, 3, "pre-5/26")
    _print_fav_count_table(pre, 2, "pre-5/26")
    _print_fav_count_table(post, 3, "post-5/26")
    _print_fav_count_table(post, 2, "post-5/26")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: Bust rate by regime
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  3. BUST RATE BY REGIME")
    print(f"{'='*95}")

    fb_pre, db_pre, gap_pre = _bust_rate_analysis(pre, "pre-5/26")
    fb_post, db_post, gap_post = _bust_rate_analysis(post, "post-5/26")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: Side-by-side comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  4. REGIME COMPARISON — SIDE BY SIDE")
    print(f"{'='*95}")

    print(f"\n  {'Metric':<35} {'Pre-5/26':>12} {'Post-5/26':>12} {'Δ':>10}")
    print(f"  {'─'*72}")

    metrics = [
        ("Fav-majority N", sf_pre["n"], sf_post["n"], ""),
        ("Fav-majority WR", f"{sf_pre['wr']:.1f}%", f"{sf_post['wr']:.1f}%",
         f"{sf_post['wr']-sf_pre['wr']:+.1f}pp"),
        ("Fav-majority ROI", f"{sf_pre['roi']:+.1f}%", f"{sf_post['roi']:+.1f}%",
         f"{sf_post['roi']-sf_pre['roi']:+.1f}pp"),
        ("Fav-majority avg odds", f"{sf_pre['avg_odds']:.3f}", f"{sf_post['avg_odds']:.3f}",
         f"{sf_post['avg_odds']-sf_pre['avg_odds']:+.3f}"),
        ("", "", "", ""),
        ("Dog-majority N", sd_pre["n"], sd_post["n"], ""),
        ("Dog-majority WR", f"{sd_pre['wr']:.1f}%", f"{sd_post['wr']:.1f}%",
         f"{sd_post['wr']-sd_pre['wr']:+.1f}pp"),
        ("Dog-majority ROI", f"{sd_pre['roi']:+.1f}%", f"{sd_post['roi']:+.1f}%",
         f"{sd_post['roi']-sd_pre['roi']:+.1f}pp"),
        ("Dog-majority avg odds", f"{sd_pre['avg_odds']:.3f}", f"{sd_post['avg_odds']:.3f}",
         f"{sd_post['avg_odds']-sd_pre['avg_odds']:+.3f}"),
        ("", "", "", ""),
        ("Fav leg bust rate", f"{fb_pre:.1f}%", f"{fb_post:.1f}%",
         f"{fb_post-fb_pre:+.1f}pp"),
        ("Dog leg bust rate", f"{db_pre:.1f}%", f"{db_post:.1f}%",
         f"{db_post-db_pre:+.1f}pp"),
        ("Bust gap (dog-fav)", f"{gap_pre:+.1f}pp", f"{gap_post:+.1f}pp",
         f"{gap_post-gap_pre:+.1f}pp"),
    ]

    for name, pre_v, post_v, delta in metrics:
        if not name:
            print()
            continue
        print(f"  {name:<35} {str(pre_v):>12} {str(post_v):>12} {delta:>10}")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: All-dog 2-leg deep dive
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  5. ALL-DOG 2-LEG PARLAYS: THE TOXIC CELL")
    print(f"{'='*95}")

    for label, group in [("Pre-5/26", pre), ("Post-5/26", post)]:
        two_dog = [p for p in group if p["n_legs"] == 2 and p["fav_count"] == 0]
        s = _pnl_stats(two_dog)
        if s["n"] > 0:
            lo, hi = wilson_ci(s["n"], s["wins"])
            print(f"\n  {label}: n={s['n']:>4}  WR={s['wr']:>5.1f}% [{lo}%,{hi}%]  "
                  f"odds={s['avg_odds']:.3f}  ROI={s['roi']:>+6.1f}%{_thin_flag(s['n'])}")
        else:
            print(f"\n  {label}: no all-dog 2-leg parlays")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  VERDICT")
    print(f"{'='*95}")

    print(f"\n  Q1: Is dog-majority -17.6% full-period ROI contaminated by pre-5/26?")
    if sd_pre["roi"] < sd_post["roi"]:
        print(f"    → YES. Pre-5/26 dog-maj ROI = {sd_pre['roi']:+.1f}%, "
              f"post-5/26 = {sd_post['roi']:+.1f}%")
        print(f"    → The {sd_pre['n']} pre-5/26 dog tickets dragged the average down")
        if sd_post["roi"] > 0:
            print(f"    → Post-5/26 dogs are PROFITABLE ({sd_post['roi']:+.1f}%) — "
                  f"the old system was the problem")
        else:
            print(f"    → Post-5/26 dogs are still negative — it's partially regime, "
                  f"partially structural")
    else:
        print(f"    → NO. Post-5/26 dog ROI ({sd_post['roi']:+.1f}%) is not better "
              f"than pre ({sd_pre['roi']:+.1f}%)")

    print(f"\n  Q2: Is the bust-rate gap regime-dependent?")
    print(f"    Pre-5/26:  {gap_pre:+.1f}pp  (fav busts less)")
    print(f"    Post-5/26: {gap_post:+.1f}pp  (fav busts less)")
    if abs(gap_pre - gap_post) < 3:
        print(f"    → STABLE across regimes — bust advantage is structural, not model-dependent")
    elif gap_post < gap_pre:
        print(f"    → NARROWED post-5/26 — system got better at dog legs specifically")
    else:
        print(f"    → WIDENED post-5/26 — fav advantage grew with better model")

    print(f"\n  Q3: Is fav-heavy ROI stable across regimes?")
    print(f"    Pre-5/26:  {sf_pre['roi']:+.1f}%")
    print(f"    Post-5/26: {sf_post['roi']:+.1f}%")
    if sf_pre["roi"] > 0 and sf_post["roi"] > 0:
        print(f"    → Profitable in BOTH regimes — fav thesis is regime-independent")
    elif sf_post["roi"] > sf_pre["roi"]:
        print(f"    → Improved with new system")
    else:
        print(f"    → Declined with new system — investigate")

    print(f"\n  Q4: Which regime should the distribution model target?")
    if sd_post["roi"] > 20 and gap_post < 5:
        print(f"    → Post-5/26 dogs are viable ({sd_post['roi']:+.1f}% ROI, "
              f"bust gap only {gap_post:+.1f}pp)")
        print(f"    → Target: BOTH fav-heavy and selective dog parlays (not all-dog 2-leg)")
    elif sf_post["roi"] > sd_post["roi"] and gap_post > 3:
        print(f"    → Fav-heavy still dominates post-5/26 (ROI {sf_post['roi']:+.1f}% vs "
              f"{sd_post['roi']:+.1f}%, bust gap {gap_post:+.1f}pp)")
        print(f"    → Target: fav-heavy composition, but monitor dog improvement")
    else:
        print(f"    → Inconclusive — small post-5/26 sample")


if __name__ == "__main__":
    main()
