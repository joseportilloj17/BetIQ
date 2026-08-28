"""
parlay_penalty_sweep.py — Penalty strength sweep for inverted composition rule.

Tests:
  A) Composition-only: penalize extremes (0-fav, 3-fav in 3-leg; 0-fav, 2-fav in 2-leg)
     at multiple strengths (×0.95, ×0.90, ×0.85, ×0.80, ×0.70)
  B) Composition×quality: penalize extremes only in low/mid LOS bands,
     leave high-LOS extremes unpenalized (since 0-fav high-LOS = +45.6%)
  C) Mixed boost variants: neutral vs mild boost (×1.05, ×1.10) on mixed

Runs each config on full period + post-5/26. Reports ROI/WR/N per config.
"""
from __future__ import annotations
import os, sys, itertools, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parlay_assembly_backtest import (
    _load_ml_favorites, _load_alt_odds_raw, generate_daily_candidates,
    wilson_ci, STAKE, _classify_composition,
)

MAX_PARLAYS_PER_DAY = 8
PARLAY_SIZES = [3, 2]

FULL_START = "2026-04-22"
FULL_END   = "2026-06-26"
POST_START = "2026-05-26"


def assemble_with_rule(candidates, max_parlays, rule_fn):
    """Like assemble_parlays but accepts an arbitrary rule_fn(composite, legs, size, avg_los) → adjusted composite."""
    if len(candidates) < 2:
        return []

    pool = [c for c in candidates
            if c.get("is_pick_fav") is not None
            and c.get("decimal_odds") is not None
            and c["decimal_odds"] > 1.0
            and c.get("actual_result") in ("WIN", "LOSS")]

    if len(pool) < 2:
        return []

    parlays = []
    used_sigs = set()

    for size in PARLAY_SIZES:
        if len(pool) < size:
            continue

        combos = list(itertools.combinations(range(len(pool)), size))
        if len(combos) > 3000:
            import random
            combos = random.sample(combos, 3000)

        for combo in combos:
            legs = [pool[i] for i in combo]

            fixture_keys = [l["fixture_key"] for l in legs]
            if len(fixture_keys) != len(set(fixture_keys)):
                continue

            combined_wp = 1.0
            combined_odds = 1.0
            for l in legs:
                combined_wp *= l["model_wp"]
                combined_odds *= l["decimal_odds"]

            wp_floor = {3: 0.30, 2: 0.40}.get(size, 0.30)
            if combined_wp < wp_floor:
                continue
            min_odds = {3: 2.0, 2: 1.5}.get(size, 1.5)
            if combined_odds < min_odds:
                continue

            sig = frozenset(id(l) for l in legs)
            if sig in used_sigs:
                continue
            used_sigs.add(sig)

            avg_los = sum(l["los"] for l in legs) / len(legs)
            avg_lqs = sum(l["lqs"] for l in legs) / len(legs)
            avg_wp = sum(l["model_wp"] for l in legs) / len(legs)

            norm_ev = max(0, combined_wp * (combined_odds - 1) - (1 - combined_wp))
            composite = avg_wp * 0.40 + (avg_lqs / 100) * 0.20 + avg_los * 0.30 + min(norm_ev, 1.0) * 0.10

            # Apply the rule
            composite = rule_fn(composite, legs, size, avg_los)

            parlay_won = all(l["actual_result"] == "WIN" for l in legs)

            parlays.append({
                "legs": legs,
                "n_legs": size,
                "combined_odds": combined_odds,
                "combined_wp": combined_wp,
                "composite": composite,
                "avg_los": avg_los,
                "won": parlay_won,
                "profit": (combined_odds - 1) * STAKE if parlay_won else -STAKE,
                "fav_count": sum(1 for l in legs if l["is_pick_fav"]),
                "composition": _classify_composition(legs),
            })

    parlays.sort(key=lambda p: -p["composite"])
    return parlays[:max_parlays]


def _pnl_row(parlays):
    n = len(parlays)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0, "roi": 0, "pnl": 0, "avg_odds": 0}
    wins = sum(1 for p in parlays if p["won"])
    pnl = sum(p["profit"] for p in parlays)
    staked = n * STAKE
    return {
        "n": n, "wins": wins, "wr": wins / n * 100,
        "roi": pnl / staked * 100, "pnl": pnl,
        "avg_odds": sum(p["combined_odds"] for p in parlays) / n,
    }


def _thin(n):
    if n < 10: return " ⚠THIN"
    if n < 20: return " ⚠thin"
    return ""


def _run_config(daily_cands, start, end, rule_fn):
    """Run a single config over a date range."""
    all_parlays = []
    for gd in sorted(daily_cands.keys()):
        if gd < start or gd > end:
            continue
        ps = assemble_with_rule(daily_cands[gd], MAX_PARLAYS_PER_DAY, rule_fn)
        for p in ps:
            p["game_date"] = gd
        all_parlays.extend(ps)
    return all_parlays


def _is_extreme(legs, size):
    """Is this parlay an extreme (all-fav or all-dog)?"""
    fav_count = sum(1 for l in legs if l["is_pick_fav"])
    return fav_count == 0 or fav_count == size


def _is_mixed(legs, size):
    """Is this parlay mixed (1 or 2 favs in a 3-leg, or 1 fav in 2-leg)?"""
    fav_count = sum(1 for l in legs if l["is_pick_fav"])
    return 0 < fav_count < size


# ── Build rule functions ─────────────────────────────────────────

def make_comp_only_rule(extreme_mult, mixed_mult=1.0):
    """Composition-only: penalize extremes, optionally boost mixed."""
    def rule(composite, legs, size, avg_los):
        if _is_extreme(legs, size):
            composite *= extreme_mult
        elif _is_mixed(legs, size) and mixed_mult != 1.0:
            composite *= mixed_mult
        return composite
    return rule


def make_comp_x_quality_rule(extreme_mult, los_threshold, mixed_mult=1.0):
    """Composition×quality: penalize extremes ONLY when avg_los < threshold."""
    def rule(composite, legs, size, avg_los):
        if _is_extreme(legs, size) and avg_los < los_threshold:
            composite *= extreme_mult
        elif _is_mixed(legs, size) and mixed_mult != 1.0:
            composite *= mixed_mult
        return composite
    return rule


def main():
    print("=" * 110)
    print("  PENALTY STRENGTH SWEEP: inverted composition rule")
    print(f"  Full: {FULL_START} → {FULL_END}  |  Post-5/26: {POST_START} → {FULL_END}")
    print("=" * 110)

    print("\nLoading data...")
    ml_favs = _load_ml_favorites(FULL_START, FULL_END)
    odds_raw = _load_alt_odds_raw(FULL_START, FULL_END)
    daily_cands = generate_daily_candidates(FULL_START, FULL_END, ml_favs, odds_raw)
    total = sum(len(v) for v in daily_cands.values())
    print(f"  {len(daily_cands)} days, {total} candidates")

    # ── Compute LOS threshold for comp×quality rules ──────────
    # Use the median of all parlays' avg_los as the quality threshold
    # (high LOS = top half, don't penalize extremes there)
    blind_rule = make_comp_only_rule(1.0, 1.0)  # no adjustment = blind
    all_blind = _run_config(daily_cands, FULL_START, FULL_END, blind_rule)
    los_vals = sorted(p["avg_los"] for p in all_blind)
    los_median = los_vals[len(los_vals) // 2]
    # Also compute tercile cut (top 1/3)
    los_t2 = los_vals[2 * len(los_vals) // 3]

    print(f"  LOS median: {los_median:.4f}, top-tercile cut: {los_t2:.4f}")

    # ═══════════════════════════════════════════════════════════
    # PART A: Composition-only sweep
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  PART A: COMPOSITION-ONLY SWEEP (penalize extremes, mixed=neutral)")
    print(f"{'='*110}")

    extreme_mults = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]

    print(f"\n  {'Extreme×':>10} {'──── Full Period ────':>30} {'──── Post-5/26 ────':>30}")
    print(f"  {'':>10} {'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6} {'MixROI':>8}  "
          f"{'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6} {'MixROI':>8}")
    print(f"  {'─'*95}")

    best_full_roi = -999
    best_full_config = ""
    best_post_roi = -999
    best_post_config = ""

    results_a = {}

    for em in extreme_mults:
        rule = make_comp_only_rule(em, 1.0)
        full_p = _run_config(daily_cands, FULL_START, FULL_END, rule)
        post_p = _run_config(daily_cands, POST_START, FULL_END, rule)

        sf = _pnl_row(full_p)
        sp = _pnl_row(post_p)

        # Composition distribution
        ext_f = sum(1 for p in full_p if _is_extreme(p["legs"], p["n_legs"]))
        ext_p = sum(1 for p in post_p if _is_extreme(p["legs"], p["n_legs"]))
        ext_pct_f = ext_f / sf["n"] * 100 if sf["n"] > 0 else 0
        ext_pct_p = ext_p / sp["n"] * 100 if sp["n"] > 0 else 0

        # Mixed-only ROI
        mix_f = [p for p in full_p if _is_mixed(p["legs"], p["n_legs"])]
        mix_p = [p for p in post_p if _is_mixed(p["legs"], p["n_legs"])]
        smf = _pnl_row(mix_f)
        smp = _pnl_row(mix_p)

        label = f"×{em:.2f}" if em != 1.0 else "blind"
        print(f"  {label:>10} {sf['n']:>5} {sf['wr']:>5.1f}% {sf['roi']:>+7.1f}% {ext_pct_f:>4.0f}% "
              f"{smf['roi']:>+7.1f}%  {sp['n']:>5} {sp['wr']:>5.1f}% {sp['roi']:>+7.1f}% "
              f"{ext_pct_p:>4.0f}% {smp['roi']:>+7.1f}%")

        results_a[em] = {"full": sf, "post": sp, "full_parlays": full_p, "post_parlays": post_p}

        if sf["roi"] > best_full_roi:
            best_full_roi = sf["roi"]
            best_full_config = f"comp-only ×{em}"
        if sp["roi"] > best_post_roi:
            best_post_roi = sp["roi"]
            best_post_config = f"comp-only ×{em}"

    # ═══════════════════════════════════════════════════════════
    # PART B: Composition-only with mixed boost
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  PART B: COMPOSITION-ONLY + MIXED BOOST")
    print(f"{'='*110}")

    mixed_boosts = [1.0, 1.05, 1.10]
    best_extreme_mults = [0.95, 0.90, 0.85]  # narrow to interesting range

    print(f"\n  {'Extreme×':>10} {'Mixed×':>8} {'── Full Period ──':>25} {'── Post-5/26 ──':>25}")
    print(f"  {'':>10} {'':>8} {'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6}  "
          f"{'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6}")
    print(f"  {'─'*80}")

    for em in best_extreme_mults:
        for mb in mixed_boosts:
            rule = make_comp_only_rule(em, mb)
            full_p = _run_config(daily_cands, FULL_START, FULL_END, rule)
            post_p = _run_config(daily_cands, POST_START, FULL_END, rule)

            sf = _pnl_row(full_p)
            sp = _pnl_row(post_p)
            ext_f = sum(1 for p in full_p if _is_extreme(p["legs"], p["n_legs"]))
            ext_p = sum(1 for p in post_p if _is_extreme(p["legs"], p["n_legs"]))
            ext_pct_f = ext_f / sf["n"] * 100 if sf["n"] > 0 else 0
            ext_pct_p = ext_p / sp["n"] * 100 if sp["n"] > 0 else 0

            print(f"  ×{em:.2f}     ×{mb:.2f}   {sf['n']:>5} {sf['wr']:>5.1f}% {sf['roi']:>+7.1f}% "
                  f"{ext_pct_f:>4.0f}%  {sp['n']:>5} {sp['wr']:>5.1f}% {sp['roi']:>+7.1f}% {ext_pct_p:>4.0f}%")

            if sf["roi"] > best_full_roi:
                best_full_roi = sf["roi"]
                best_full_config = f"comp-only ×{em} + mixed ×{mb}"
            if sp["roi"] > best_post_roi:
                best_post_roi = sp["roi"]
                best_post_config = f"comp-only ×{em} + mixed ×{mb}"

    # ═══════════════════════════════════════════════════════════
    # PART C: Composition × Quality (penalize extremes only in low/mid LOS)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  PART C: COMPOSITION × QUALITY (penalize extremes only when avg_los < threshold)")
    print(f"  Thresholds tested: median ({los_median:.4f}), top-tercile ({los_t2:.4f})")
    print(f"{'='*110}")

    thresholds = [
        ("median", los_median),
        ("top-1/3", los_t2),
    ]

    print(f"\n  {'Extreme×':>10} {'Thresh':>10} {'Mixed×':>8} {'── Full Period ──':>25} {'── Post-5/26 ──':>25}")
    print(f"  {'':>10} {'':>10} {'':>8} {'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6}  "
          f"{'N':>5} {'WR':>7} {'ROI':>8} {'Ext%':>6}")
    print(f"  {'─'*90}")

    for em in [0.95, 0.90, 0.85, 0.80]:
        for t_name, t_val in thresholds:
            for mb in [1.0, 1.05]:
                rule = make_comp_x_quality_rule(em, t_val, mb)
                full_p = _run_config(daily_cands, FULL_START, FULL_END, rule)
                post_p = _run_config(daily_cands, POST_START, FULL_END, rule)

                sf = _pnl_row(full_p)
                sp = _pnl_row(post_p)
                ext_f = sum(1 for p in full_p if _is_extreme(p["legs"], p["n_legs"]))
                ext_p = sum(1 for p in post_p if _is_extreme(p["legs"], p["n_legs"]))
                ext_pct_f = ext_f / sf["n"] * 100 if sf["n"] > 0 else 0
                ext_pct_p = ext_p / sp["n"] * 100 if sp["n"] > 0 else 0

                mb_str = f"×{mb:.2f}" if mb != 1.0 else "  —  "
                print(f"  ×{em:.2f}     {t_name:>10} {mb_str:>8}  {sf['n']:>5} {sf['wr']:>5.1f}% "
                      f"{sf['roi']:>+7.1f}% {ext_pct_f:>4.0f}%  {sp['n']:>5} {sp['wr']:>5.1f}% "
                      f"{sp['roi']:>+7.1f}% {ext_pct_p:>4.0f}%")

                config_label = f"comp×qual ×{em} thresh={t_name}" + (f" + mixed ×{mb}" if mb != 1.0 else "")
                if sf["roi"] > best_full_roi:
                    best_full_roi = sf["roi"]
                    best_full_config = config_label
                if sp["roi"] > best_post_roi:
                    best_post_roi = sp["roi"]
                    best_post_config = config_label

    # ═══════════════════════════════════════════════════════════
    # PART D: Detailed breakdown of best configs
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  PART D: BEST CONFIGS — DETAILED BREAKDOWN")
    print(f"{'='*110}")

    print(f"\n  Best full-period ROI:  {best_full_roi:+.1f}% — {best_full_config}")
    print(f"  Best post-5/26 ROI:   {best_post_roi:+.1f}% — {best_post_config}")

    # Re-run the top comp-only configs and show fav-count breakdown
    print(f"\n  ── Fav-count breakdown for top comp-only configs (full period, 3-leg) ──")

    for em in [1.0, 0.95, 0.90, 0.85]:
        rule = make_comp_only_rule(em, 1.0)
        full_p = _run_config(daily_cands, FULL_START, FULL_END, rule)
        three = [p for p in full_p if p["n_legs"] == 3]
        s_all = _pnl_row(three)

        label = f"×{em:.2f}" if em != 1.0 else "blind"
        print(f"\n  {label} ({len(three)} 3-leg tickets, ROI={s_all['roi']:+.1f}%):")
        print(f"  {'Fav#':>5} {'N':>5} {'WR':>7} {'ROI':>8} {'%ofTotal':>9}")
        print(f"  {'─'*40}")
        for fc in range(4):
            g = [p for p in three if p["fav_count"] == fc]
            s = _pnl_row(g)
            pct = s["n"] / len(three) * 100 if three else 0
            if s["n"] == 0:
                print(f"  {fc:>5}     0       —        —         —")
            else:
                print(f"  {fc:>5} {s['n']:>5} {s['wr']:>5.1f}% {s['roi']:>+7.1f}% {pct:>7.1f}%{_thin(s['n'])}")

    # ═══════════════════════════════════════════════════════════
    # PART E: Overshoot check — are extremes being zeroed out?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  PART E: OVERSHOOT CHECK — extreme cell counts at each penalty")
    print(f"{'='*110}")

    print(f"\n  {'Config':>18} {'0-fav N':>8} {'3-fav N':>8} {'0-fav ROI':>10} {'3-fav ROI':>10} {'Tot Ext':>8}")
    print(f"  {'─'*60}")

    for em in [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]:
        rule = make_comp_only_rule(em, 1.0)
        full_p = _run_config(daily_cands, FULL_START, FULL_END, rule)
        three = [p for p in full_p if p["n_legs"] == 3]
        z = [p for p in three if p["fav_count"] == 0]
        t = [p for p in three if p["fav_count"] == 3]
        sz = _pnl_row(z)
        st = _pnl_row(t)
        label = f"×{em:.2f}" if em != 1.0 else "blind"
        z_roi = f"{sz['roi']:>+8.1f}%" if sz['n'] > 0 else "       —"
        t_roi = f"{st['roi']:>+8.1f}%" if st['n'] > 0 else "       —"
        print(f"  {label:>18} {sz['n']:>8} {st['n']:>8} {z_roi:>10} {t_roi:>10} {sz['n']+st['n']:>8}")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    print(f"\n  Best full-period:  {best_full_roi:+.1f}% — {best_full_config}")
    print(f"  Best post-5/26:    {best_post_roi:+.1f}% — {best_post_config}")
    print(f"  Blind baseline:    Full={results_a[1.0]['full']['roi']:+.1f}%, "
          f"Post={results_a[1.0]['post']['roi']:+.1f}%")


if __name__ == "__main__":
    main()
