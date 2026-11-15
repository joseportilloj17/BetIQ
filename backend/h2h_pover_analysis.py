#!/usr/bin/env python3
"""
Head-to-head: lognormal vs shifted-gamma P(over) on graded W2-W9 predictions.

1. Add p_over_v12_gamma column if missing
2. Recompute shifted-gamma P(over) for all yardage graded rows from stored inputs
3. Backfill p_over_v12_gamma
4. Compare: Brier, calibration, ROI by odds band, prereg rule results under each
"""
import os, sys, sqlite3, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nfl_interim_dispersion import (
    get_interim_cv, get_v12_base_rate,
    p_over_line_gamma_shifted,
    load_cached_params, load_v12_base_rates,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

YARDAGE_STATS = {"pass_yds", "rush_yds", "rec_yds"}

def main():
    # Load params
    ok1 = load_cached_params()
    ok2 = load_v12_base_rates()
    print(f"Loaded dispersion params: {ok1}, v1.2 base rates: {ok2}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # -- Step 1: Ensure column exists --
    try:
        conn.execute("ALTER TABLE fantasy_forward_preds ADD COLUMN p_over_v12_gamma REAL")
        print("Added p_over_v12_gamma column")
    except Exception:
        print("p_over_v12_gamma column already exists")

    try:
        conn.execute("ALTER TABLE fantasy_forward_preds ADD COLUMN distributional_median REAL")
        print("Added distributional_median column")
    except Exception:
        pass

    # -- Step 2: Load all graded rows --
    rows = conn.execute("""
        SELECT id, season, week, player_name, position, stat_type, method,
               consensus_mean, prop_line, p_over, devigged_implied,
               actual_value, is_win, prop_over_price, prop_under_price,
               line_depth, data_error, model_version, p_over_v12_gamma
        FROM fantasy_forward_preds
        WHERE actual_value IS NOT NULL
          AND p_over IS NOT NULL
          AND method = 'distributional'
          AND data_error IS NULL
        ORDER BY week, player_name, stat_type
    """).fetchall()
    print(f"\nTotal clean graded distributional rows: {len(rows)}")

    # Count by stat
    stat_counts = {}
    for r in rows:
        st = r["stat_type"]
        stat_counts[st] = stat_counts.get(st, 0) + 1
    for st, c in sorted(stat_counts.items(), key=lambda x: -x[1]):
        print(f"  {st}: {c}")

    # -- Step 3: Compute gamma P(over) for yardage rows --
    updates = []
    skipped = 0
    for r in rows:
        if r["stat_type"] not in YARDAGE_STATS:
            continue
        mean_val = r["consensus_mean"]
        line_val = r["prop_line"]
        pos = r["position"]
        stat = r["stat_type"]
        if mean_val is None or line_val is None or mean_val <= 0:
            skipped += 1
            continue
        cv = get_interim_cv(stat, pos, mean_val, use_v12=True)
        br = get_v12_base_rate(stat, pos)
        try:
            p_gamma = round(p_over_line_gamma_shifted(line_val, mean_val, cv, br), 4)
        except Exception as e:
            skipped += 1
            continue
        updates.append((p_gamma, r["id"]))

    print(f"\nGamma P(over) computed: {len(updates)}, skipped: {skipped}")

    # -- Step 4: Backfill --
    conn.executemany("UPDATE fantasy_forward_preds SET p_over_v12_gamma = ? WHERE id = ?", updates)
    conn.commit()
    print(f"Backfilled {len(updates)} rows")

    # -- Step 5: Reload with gamma values --
    rows = conn.execute("""
        SELECT id, season, week, player_name, position, stat_type, method,
               consensus_mean, prop_line, p_over, p_over_v12_gamma,
               devigged_implied, actual_value, is_win,
               prop_over_price, prop_under_price,
               line_depth, model_version
        FROM fantasy_forward_preds
        WHERE actual_value IS NOT NULL
          AND p_over IS NOT NULL
          AND method = 'distributional'
          AND data_error IS NULL
        ORDER BY week, player_name, stat_type
    """).fetchall()

    # Only yardage rows have gamma P(over)
    yardage_rows = [r for r in rows if r["stat_type"] in YARDAGE_STATS
                    and r["p_over_v12_gamma"] is not None]
    print(f"\nYardage rows with both P(over) values: {len(yardage_rows)}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ════════════════════════════════════════════════════════════════

    def brier(p_over, actual, line):
        """Brier score: (p_predicted - outcome)^2 where outcome = actual > line."""
        if actual is None or p_over is None:
            return None
        outcome = 1.0 if actual > line else 0.0
        if actual == line:  # push
            return None
        return (p_over - outcome) ** 2

    # -- 5A: Overall Brier --
    print("\n" + "="*70)
    print("BRIER SCORES (lower = better)")
    print("="*70)

    def brier_set(rows_subset, label):
        ln_briers = []
        gm_briers = []
        for r in rows_subset:
            b_ln = brier(r["p_over"], r["actual_value"], r["prop_line"])
            b_gm = brier(r["p_over_v12_gamma"], r["actual_value"], r["prop_line"])
            if b_ln is not None:
                ln_briers.append(b_ln)
            if b_gm is not None:
                gm_briers.append(b_gm)
        if not ln_briers:
            return
        mean_ln = np.mean(ln_briers)
        mean_gm = np.mean(gm_briers)
        diff = mean_gm - mean_ln
        n = len(ln_briers)
        # Paired t-test
        if n > 1:
            diffs = np.array([brier(r["p_over_v12_gamma"], r["actual_value"], r["prop_line"])
                              for r in rows_subset
                              if brier(r["p_over"], r["actual_value"], r["prop_line"]) is not None
                              and brier(r["p_over_v12_gamma"], r["actual_value"], r["prop_line"]) is not None])
            lns = np.array([brier(r["p_over"], r["actual_value"], r["prop_line"])
                            for r in rows_subset
                            if brier(r["p_over"], r["actual_value"], r["prop_line"]) is not None
                            and brier(r["p_over_v12_gamma"], r["actual_value"], r["prop_line"]) is not None])
            paired_diff = diffs - lns
            se = np.std(paired_diff, ddof=1) / np.sqrt(len(paired_diff))
            t_stat = np.mean(paired_diff) / se if se > 0 else 0
        else:
            se = t_stat = 0
        winner = "GAMMA" if diff < 0 else "LOGNORMAL" if diff > 0 else "TIE"
        print(f"\n{label} (N={n}):")
        print(f"  Lognormal Brier:     {mean_ln:.6f}")
        print(f"  Shifted-Gamma Brier: {mean_gm:.6f}")
        print(f"  Δ (gamma - ln):      {diff:+.6f}  [{winner}]")
        print(f"  Paired SE: {se:.6f}, t={t_stat:.3f}")

    # Overall
    brier_set(yardage_rows, "ALL YARDAGE")

    # By stat
    for stat in sorted(YARDAGE_STATS):
        subset = [r for r in yardage_rows if r["stat_type"] == stat]
        brier_set(subset, stat.upper())

    # By week
    weeks = sorted(set(r["week"] for r in yardage_rows))
    print("\n--- By Week ---")
    for w in weeks:
        subset = [r for r in yardage_rows if r["week"] == w]
        brier_set(subset, f"Week {w}")

    # By line depth
    for ld in ["main", "alt"]:
        subset = [r for r in yardage_rows if r["line_depth"] == ld]
        if subset:
            brier_set(subset, f"line_depth={ld}")

    # -- 5B: Calibration (10 buckets) --
    print("\n" + "="*70)
    print("CALIBRATION (10 buckets)")
    print("="*70)

    def calibration(rows_subset, p_col):
        buckets = [[] for _ in range(10)]
        for r in rows_subset:
            p = r[p_col]
            if p is None:
                continue
            actual = r["actual_value"]
            line = r["prop_line"]
            if actual == line:
                continue
            outcome = 1.0 if actual > line else 0.0
            bucket = min(int(p * 10), 9)
            buckets[bucket].append(outcome)
        results = []
        for i, b in enumerate(buckets):
            lo = i * 0.1
            hi = (i + 1) * 0.1
            mid = (lo + hi) / 2
            if b:
                actual_rate = np.mean(b)
                n = len(b)
                results.append((mid, actual_rate, n))
            else:
                results.append((mid, None, 0))
        return results

    print(f"\n{'Bucket':>8} {'LN_pred':>8} {'LN_act':>8} {'LN_N':>6} {'GM_pred':>8} {'GM_act':>8} {'GM_N':>6}")
    ln_cal = calibration(yardage_rows, "p_over")
    gm_cal = calibration(yardage_rows, "p_over_v12_gamma")
    for i in range(10):
        ln_mid, ln_act, ln_n = ln_cal[i]
        gm_mid, gm_act, gm_n = gm_cal[i]
        ln_act_s = f"{ln_act:.3f}" if ln_act is not None else "---"
        gm_act_s = f"{gm_act:.3f}" if gm_act is not None else "---"
        print(f"  {ln_mid:.1f}    {ln_mid:.3f}   {ln_act_s:>7}  {ln_n:>5}   {gm_mid:.3f}   {gm_act_s:>7}  {gm_n:>5}")

    # Calibration error (mean abs deviation from diagonal)
    def cal_error(cal_results):
        errs = []
        for mid, act, n in cal_results:
            if act is not None and n >= 5:
                errs.append(abs(act - mid))
        return np.mean(errs) if errs else None

    ln_ce = cal_error(ln_cal)
    gm_ce = cal_error(gm_cal)
    print(f"\nMean abs calibration error (N>=5 buckets):")
    print(f"  Lognormal:     {ln_ce:.4f}" if ln_ce else "  Lognormal:     N/A")
    print(f"  Shifted-Gamma: {gm_ce:.4f}" if gm_ce else "  Shifted-Gamma: N/A")

    # -- 5C: P(over) difference statistics --
    print("\n" + "="*70)
    print("P(over) DIFFERENCE STATISTICS (gamma - lognormal)")
    print("="*70)

    diffs = []
    for r in yardage_rows:
        d = r["p_over_v12_gamma"] - r["p_over"]
        diffs.append(d)
    diffs = np.array(diffs)
    print(f"  N:      {len(diffs)}")
    print(f"  Mean:   {np.mean(diffs):+.4f}")
    print(f"  Median: {np.median(diffs):+.4f}")
    print(f"  Std:    {np.std(diffs):.4f}")
    print(f"  Min:    {np.min(diffs):+.4f}")
    print(f"  Max:    {np.max(diffs):+.4f}")
    print(f"  |Δ|>0.05: {np.sum(np.abs(diffs) > 0.05)} ({np.sum(np.abs(diffs) > 0.05)/len(diffs)*100:.1f}%)")
    print(f"  |Δ|>0.10: {np.sum(np.abs(diffs) > 0.10)} ({np.sum(np.abs(diffs) > 0.10)/len(diffs)*100:.1f}%)")

    # By stat
    for stat in sorted(YARDAGE_STATS):
        subset = [r["p_over_v12_gamma"] - r["p_over"] for r in yardage_rows if r["stat_type"] == stat]
        if subset:
            s = np.array(subset)
            print(f"\n  {stat}: mean={np.mean(s):+.4f}, median={np.median(s):+.4f}, std={np.std(s):.4f}, N={len(s)}")

    # -- 5D: Direction agreement --
    print("\n" + "="*70)
    print("DIRECTION AGREEMENT")
    print("="*70)

    agree = 0
    disagree = 0
    disagree_rows = []
    for r in yardage_rows:
        ln_dir = "over" if r["p_over"] > 0.5 else "under"
        gm_dir = "over" if r["p_over_v12_gamma"] > 0.5 else "under"
        if ln_dir == gm_dir:
            agree += 1
        else:
            disagree += 1
            disagree_rows.append(r)
    print(f"  Agree: {agree} ({agree/(agree+disagree)*100:.1f}%)")
    print(f"  Disagree: {disagree} ({disagree/(agree+disagree)*100:.1f}%)")

    # When they disagree, who's right?
    if disagree_rows:
        ln_right = 0
        gm_right = 0
        push = 0
        for r in disagree_rows:
            actual = r["actual_value"]
            line = r["prop_line"]
            if actual == line:
                push += 1
                continue
            outcome = 1.0 if actual > line else 0.0
            ln_dir = 1.0 if r["p_over"] > 0.5 else 0.0
            gm_dir = 1.0 if r["p_over_v12_gamma"] > 0.5 else 0.0
            if ln_dir == outcome:
                ln_right += 1
            if gm_dir == outcome:
                gm_right += 1
        print(f"  On disagreements: LN right={ln_right}, GM right={gm_right}, pushes={push}")

    # -- 5E: ROI by odds band (prereg-style) --
    print("\n" + "="*70)
    print("PREREG RULE RESULTS")
    print("="*70)

    def prereg_results(rows_subset, p_col, label):
        """Simulate the prereg rule: bet OVER when p > devigged_implied + threshold."""
        for threshold in [0.00, 0.03, 0.05]:
            bets = []
            for r in rows_subset:
                p = r[p_col]
                devig = r["devigged_implied"]
                if p is None or devig is None:
                    continue
                edge = p - devig
                if edge < threshold:
                    continue
                actual = r["actual_value"]
                line = r["prop_line"]
                if actual == line:
                    continue
                over_price = r["prop_over_price"]
                if over_price is None or over_price <= 1.0:
                    continue
                # Prices are already decimal (e.g. 1.88)
                dec_odds = over_price
                won = actual > line
                pnl = (dec_odds - 1) if won else -1
                bets.append({
                    "edge": edge,
                    "pnl": pnl,
                    "won": won,
                    "odds": dec_odds,
                    "stat": r["stat_type"],
                    "week": r["week"],
                })
            if not bets:
                print(f"  {label} @≥{threshold:.0%} edge: N=0")
                continue
            n = len(bets)
            wr = sum(1 for b in bets if b["won"]) / n
            roi = np.mean([b["pnl"] for b in bets])
            total_pnl = sum(b["pnl"] for b in bets)
            avg_odds = np.mean([b["odds"] for b in bets])
            be = 1 / avg_odds
            print(f"  {label} @≥{threshold:.0%} edge: N={n}, WR={wr:.1%}, ROI={roi:+.1%}, "
                  f"PnL={total_pnl:+.1f}u, BE={be:.1%}, avg_odds={avg_odds:.3f}")

    # Main lines only
    main_yard = [r for r in yardage_rows if r["line_depth"] == "main"]
    print(f"\nMain-line yardage only (N={len(main_yard)}):")
    prereg_results(main_yard, "p_over", "LOGNORMAL")
    prereg_results(main_yard, "p_over_v12_gamma", "GAMMA")

    # All lines
    print(f"\nAll yardage lines (N={len(yardage_rows)}):")
    prereg_results(yardage_rows, "p_over", "LOGNORMAL")
    prereg_results(yardage_rows, "p_over_v12_gamma", "GAMMA")

    # -- 5F: ROI by odds band --
    print("\n" + "="*70)
    print("ROI BY ODDS BAND (using OVER price)")
    print("="*70)

    def roi_by_band(rows_subset, p_col, label):
        bands = [(1.5, 1.8), (1.8, 2.0), (2.0, 2.2), (2.2, 2.5), (2.5, 3.0), (3.0, 5.0)]
        for lo, hi in bands:
            bets = []
            for r in rows_subset:
                p = r[p_col]
                devig = r["devigged_implied"]
                if p is None or devig is None:
                    continue
                edge = p - devig
                if edge < 0.03:
                    continue
                over_price = r["prop_over_price"]
                if over_price is None or over_price <= 1.0:
                    continue
                dec_odds = over_price
                if dec_odds < lo or dec_odds >= hi:
                    continue
                actual = r["actual_value"]
                line = r["prop_line"]
                if actual == line:
                    continue
                won = actual > line
                pnl = (dec_odds - 1) if won else -1
                bets.append(pnl)
            if bets:
                n = len(bets)
                wr = sum(1 for p in bets if p > 0) / n
                roi = np.mean(bets)
                print(f"  {label} odds [{lo:.1f},{hi:.1f}): N={n}, WR={wr:.1%}, ROI={roi:+.1%}")
            else:
                print(f"  {label} odds [{lo:.1f},{hi:.1f}): N=0")

    roi_by_band(main_yard, "p_over", "LN")
    print()
    roi_by_band(main_yard, "p_over_v12_gamma", "GM")

    # -- 5G: Regime tagging --
    print("\n" + "="*70)
    print("REGIME BREAKDOWN")
    print("="*70)

    # By model version
    for mv in sorted(set(r["model_version"] or "NULL" for r in yardage_rows)):
        subset = [r for r in yardage_rows if (r["model_version"] or "NULL") == mv]
        brier_set(subset, f"model_version={mv}")

    # By position
    positions = sorted(set(r["position"] for r in yardage_rows if r["position"]))
    for pos in positions:
        subset = [r for r in yardage_rows if r["position"] == pos]
        if len(subset) >= 30:
            brier_set(subset, f"position={pos}")

    conn.close()
    print("\n✓ Analysis complete")


if __name__ == "__main__":
    main()
