"""
SINGLE-FEATURE GATE SWEEP on spread bets.

For each feature individually, sweep thresholds.
At each threshold: N, WR, actual breakeven of those bets, edge = WR - BE.
Matched-yardstick rule: compare to THEIR OWN vig, not a global BE.
Split-half + multi-fold validation for any positive-edge gate.
"""

import sqlite3, json, re, math, sys
import numpy as np
from collections import defaultdict

DB = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"


def load_dataset():
    """Load all spread legs with features and actual odds."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.leg_id, c.fixture_id, c.team, c.spread, c.leg_result,
               c.entry_decimal, c.arm,
               l.ats_dist_wp, l.ridge_composite, l.bayes_team_wp,
               l.negbin_wp, l.negbin_edge_pp,
               l.negbin_mu_home, l.negbin_mu_away,
               l.description,
               l.resolved_home_score, l.resolved_away_score,
               l.resolved_home_team, l.resolved_away_team
        FROM bet_clv c
        JOIN mock_bet_legs l ON l.id = c.leg_id
        WHERE c.entry_decimal IS NOT NULL
          AND c.leg_result IN ('WIN', 'LOSS')
    """).fetchall()
    conn.close()

    dataset = []
    for r in rows:
        desc = r["description"] or ""
        # Derive home/away from resolved team names (description parsing unreliable)
        is_home = (r["team"] == r["resolved_home_team"]) if r["resolved_home_team"] else False
        is_away = (r["team"] == r["resolved_away_team"]) if r["resolved_away_team"] else False

        # Predicted margin: SIGNED relative to our pick (positive = model favors our team)
        margin = None
        margin_unsigned = None  # also keep unsigned (mu_home - mu_away) for comparison
        if r["negbin_mu_home"] is not None and r["negbin_mu_away"] is not None:
            margin_unsigned = r["negbin_mu_home"] - r["negbin_mu_away"]
            if is_home:
                margin = r["negbin_mu_home"] - r["negbin_mu_away"]  # positive = our team (home) favored
            elif is_away:
                margin = r["negbin_mu_away"] - r["negbin_mu_home"]  # positive = our team (away) favored
            else:
                margin = margin_unsigned  # fallback

        # Predicted total
        pred_total = None
        if r["negbin_mu_home"] is not None and r["negbin_mu_away"] is not None:
            pred_total = r["negbin_mu_home"] + r["negbin_mu_away"]

        won = r["leg_result"] == "WIN"
        be = 1.0 / r["entry_decimal"]  # actual breakeven for THIS bet

        d = {
            "leg_id": r["leg_id"],
            "fixture_id": r["fixture_id"],
            "team": r["team"],
            "rung": r["spread"],
            "won": won,
            "entry_dec": r["entry_decimal"],
            "be": be,
            "arm": r["arm"],
            # Features
            "dist_wp": r["ats_dist_wp"],
            "ridge": r["ridge_composite"],
            "bayes_wp": r["bayes_team_wp"],
            "negbin_wp": r["negbin_wp"],
            "edge_pp": r["negbin_edge_pp"],
            "mu_away": r["negbin_mu_away"],
            "mu_home": r["negbin_mu_home"],
            "margin": margin,              # signed, positive = our team favored by model
            "margin_unsigned": margin_unsigned,  # mu_home - mu_away (raw)
            "pred_total": pred_total,
            "is_home": is_home,
            "is_away": is_away,
        }
        dataset.append(d)

    print(f"Loaded {len(dataset)} spread legs with entry odds")
    return dataset


def edge_stats(subset):
    """Compute edge stats for a subset of legs."""
    if not subset:
        return {"n": 0, "wr": 0, "be": 0, "edge": 0, "z": 0, "pnl": 0}
    n = len(subset)
    wins = sum(1 for d in subset if d["won"])
    wr = wins / n
    avg_be = np.mean([d["be"] for d in subset])
    edge = wr - avg_be
    se = math.sqrt(avg_be * (1 - avg_be) / n) if n > 0 else 1
    z = edge / se if se > 0 else 0
    pnl = sum((d["entry_dec"] - 1) if d["won"] else -1.0 for d in subset) / n
    return {"n": n, "wr": wr, "be": avg_be, "edge": edge, "z": z, "pnl": pnl}


def gate_sweep(dataset, feature, thresholds, direction=">=", label=None):
    """Sweep a single feature at multiple thresholds."""
    label = label or feature
    print(f"\n{'─'*75}")
    print(f"  {label}")
    print(f"{'─'*75}")
    print(f"{'Gate':>22}  {'N':>6}  {'WR%':>6}  {'BE%':>6}  {'Edge':>8}  {'Z':>6}  {'PnL/u':>8}")
    print(f"{'─'*75}")

    # Baseline (no gate)
    valid = [d for d in dataset if d.get(feature) is not None]
    if not valid:
        print(f"  No data for {feature}")
        return []

    baseline = edge_stats(valid)
    print(f"{'(all, no gate)':>22}  {baseline['n']:>6}  {baseline['wr']*100:>5.1f}%  "
          f"{baseline['be']*100:>5.1f}%  {baseline['edge']*100:>+7.2f}pp  "
          f"{baseline['z']:>+5.2f}  ${baseline['pnl']:>+7.4f}")

    results = []
    for thresh in thresholds:
        if direction == ">=":
            subset = [d for d in valid if d[feature] >= thresh]
            gate_name = f">= {thresh}"
        elif direction == "<=":
            subset = [d for d in valid if d[feature] <= thresh]
            gate_name = f"<= {thresh}"
        elif direction == "==":
            subset = [d for d in valid if d[feature] == thresh]
            gate_name = f"== {thresh}"
        else:
            continue

        s = edge_stats(subset)
        if s["n"] < 10:
            continue

        marker = " ***" if s["edge"] > 0 and s["n"] >= 100 else ""
        print(f"{gate_name:>22}  {s['n']:>6}  {s['wr']*100:>5.1f}%  "
              f"{s['be']*100:>5.1f}%  {s['edge']*100:>+7.2f}pp  "
              f"{s['z']:>+5.2f}  ${s['pnl']:>+7.4f}{marker}")

        results.append({
            "feature": feature, "thresh": thresh, "direction": direction,
            "gate_name": gate_name, **s
        })

    return results


def split_half_full(dataset, feature, thresh, direction=">="):
    """Full split-half: chrono + 5 random + 5-fold."""
    if direction == ">=":
        subset = [d for d in dataset if d.get(feature) is not None and d[feature] >= thresh]
    elif direction == "<=":
        subset = [d for d in dataset if d.get(feature) is not None and d[feature] <= thresh]
    elif direction == "==":
        subset = [d for d in dataset if d.get(feature) is not None and d[feature] == thresh]
    else:
        return

    n = len(subset)
    if n < 40:
        print(f"    N={n} too small for split-half")
        return {"chrono": False, "random": False, "fold": False, "n": n}

    subset_sorted = sorted(subset, key=lambda d: d["fixture_id"])
    mid = n // 2

    # ── CHRONO SPLIT ──
    h1 = subset_sorted[:mid]
    h2 = subset_sorted[mid:]
    e1 = edge_stats(h1)
    e2 = edge_stats(h2)
    chrono_pass = e1["edge"] > 0 and e2["edge"] > 0
    print(f"    Chrono: H1 edge={e1['edge']*100:>+5.1f}pp (WR={e1['wr']*100:.1f}% BE={e1['be']*100:.1f}% N={e1['n']})  "
          f"H2 edge={e2['edge']*100:>+5.1f}pp (WR={e2['wr']*100:.1f}% BE={e2['be']*100:.1f}% N={e2['n']})  "
          f"{'PASS' if chrono_pass else 'FAIL'}")

    # ── RANDOM SPLITS ──
    n_seeds = 5
    pos_count = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        s1 = [subset[i] for i in idx[:mid]]
        s2 = [subset[i] for i in idx[mid:]]
        e_s1 = edge_stats(s1)
        e_s2 = edge_stats(s2)
        if e_s1["edge"] > 0 and e_s2["edge"] > 0:
            pos_count += 1
    random_pass = pos_count >= 3  # majority
    print(f"    Random: {pos_count}/{n_seeds} seeds BOTH halves positive  "
          f"{'PASS' if random_pass else 'FAIL'}")

    # ── MULTI-FOLD ──
    k = 5
    n_seeds_fold = 5
    noise_floor = math.sqrt(1.0 / (k * n))

    all_fold_edges = []
    for seed in range(n_seeds_fold):
        rng = np.random.default_rng(seed + 100)
        idx = rng.permutation(n)
        fold_edges = []
        fold_size = n // k
        for f in range(k):
            start = f * fold_size
            end = start + fold_size if f < k - 1 else n
            fold = [subset_sorted[idx[i]] for i in range(start, end)]
            fe = edge_stats(fold)
            fold_edges.append(fe["edge"])
        all_fold_edges.append(fold_edges)

    avg_edges = np.mean(all_fold_edges, axis=0)
    folds_above = sum(1 for e in avg_edges if e > noise_floor)
    mean_fold = np.mean(avg_edges)
    fold_se = np.std(avg_edges) / math.sqrt(k) if k > 1 else 999
    fold_z = mean_fold / fold_se if fold_se > 0 else 0
    fold_pass = folds_above >= 4  # 4/5 folds above noise floor
    print(f"    {k}-fold: {folds_above}/{k} above noise ({noise_floor*100:.3f}pp)  "
          f"mean={mean_fold*100:+.2f}pp  Z={fold_z:+.2f}  "
          f"{'PASS' if fold_pass else 'FAIL'}")
    print(f"    Per-fold: {['%+.1f' % (e*100) for e in avg_edges]}")

    all_pass = chrono_pass and random_pass and fold_pass
    print(f"    ──> VERDICT: {'ALL THREE PASS ✓' if all_pass else 'DOES NOT PASS all three'}")
    return {"chrono": chrono_pass, "random": random_pass, "fold": fold_pass,
            "n": n, "all_pass": all_pass}


def main():
    dataset = load_dataset()

    # ═══════════════════════════════════════════════════════════
    # 1. GATE SWEEP FOR EACH FEATURE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print(f"  PART 1: SINGLE-FEATURE GATE SWEEP (edge = WR - own BE)")
    print(f"{'='*75}")

    all_results = []

    # ── dist_wp ──
    r = gate_sweep(dataset, "dist_wp",
                   [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                   ">=", "dist_wp (distributional cover prob)")
    all_results.extend(r)

    # ── ridge_composite ──
    r = gate_sweep(dataset, "ridge",
                   [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                   ">=", "ridge_composite")
    all_results.extend(r)

    # ── bayes_team_wp ──
    r = gate_sweep(dataset, "bayes_wp",
                   [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
                   ">=", "bayes_team_wp (rung-independent win prob)")
    all_results.extend(r)

    # ── negbin_wp ──
    r = gate_sweep(dataset, "negbin_wp",
                   [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98],
                   ">=", "negbin_wp (NegBin cover prob)")
    all_results.extend(r)

    # ── edge_pp (BOTH directions: high = thin edge, low = wide edge) ──
    r = gate_sweep(dataset, "edge_pp",
                   [0, 2, 4, 6, 8, 10, 12],
                   "<=", "edge_pp <= (low edge_pp = market-aligned)")
    all_results.extend(r)
    r = gate_sweep(dataset, "edge_pp",
                   [0, 2, 4, 6, 8, 10],
                   ">=", "edge_pp >= (high edge_pp = disagrees with market)")
    all_results.extend(r)

    # ── mu_away (lower = opponent scores less → easier cover) ──
    r = gate_sweep(dataset, "mu_away",
                   [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                   "<=", "mu_away <= (weak opponent)")
    all_results.extend(r)
    r = gate_sweep(dataset, "mu_away",
                   [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                   ">=", "mu_away >= (strong opponent)")
    all_results.extend(r)

    # ── margin SIGNED (positive = our team favored by model) ──
    r = gate_sweep(dataset, "margin",
                   [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                   ">=", "signed_margin >= (model favors OUR pick)")
    all_results.extend(r)

    # ── margin UNSIGNED (mu_home - mu_away, large = lopsided game) ──
    r = gate_sweep(dataset, "margin_unsigned",
                   [-2.0, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                   ">=", "unsigned_margin >= (home-favored, lopsided)")
    all_results.extend(r)
    r = gate_sweep(dataset, "margin_unsigned",
                   [-3.0, -2.5, -2.0, -1.5, -1.0, 0.0],
                   "<=", "unsigned_margin <= (away-favored, lopsided)")
    all_results.extend(r)

    # ── predicted total ──
    r = gate_sweep(dataset, "pred_total",
                   [6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0],
                   "<=", "pred_total <= (low-scoring game)")
    all_results.extend(r)
    r = gate_sweep(dataset, "pred_total",
                   [7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                   ">=", "pred_total >= (high-scoring game)")
    all_results.extend(r)

    # ── rung ──
    r = gate_sweep(dataset, "rung",
                   [1.5, 2.5, 3.5, 4.5],
                   "==", "rung == (specific spread)")
    all_results.extend(r)
    r = gate_sweep(dataset, "rung",
                   [1.5, 2.5, 3.5],
                   "<=", "rung <= (thin rungs only)")
    all_results.extend(r)

    # ── home/away ──
    home_data = [d for d in dataset if d["is_home"]]
    away_data = [d for d in dataset if d["is_away"]]
    print(f"\n{'─'*75}")
    print(f"  home vs away pick side")
    print(f"{'─'*75}")
    eh = edge_stats(home_data)
    ea = edge_stats(away_data)
    print(f"  Home picks: N={eh['n']} WR={eh['wr']*100:.1f}% BE={eh['be']*100:.1f}% "
          f"Edge={eh['edge']*100:+.2f}pp Z={eh['z']:+.2f} PnL/u=${eh['pnl']:+.4f}")
    print(f"  Away picks: N={ea['n']} WR={ea['wr']*100:.1f}% BE={ea['be']*100:.1f}% "
          f"Edge={ea['edge']*100:+.2f}pp Z={ea['z']:+.2f} PnL/u=${ea['pnl']:+.4f}")

    # ═══════════════════════════════════════════════════════════
    # 2. IDENTIFY ALL POSITIVE-EDGE GATES WITH N >= 100
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'='*75}")
    print(f"  PART 2: SPLIT-HALF VALIDATION (all gates with edge > 0 AND N >= 100)")
    print(f"{'='*75}")

    positive_gates = [r for r in all_results if r["edge"] > 0 and r["n"] >= 100]
    positive_gates.sort(key=lambda x: -x["edge"])

    if not positive_gates:
        print("\n  NO single-feature gate produces positive edge at N >= 100.")
        print("  Every individual feature is sub-vig on spreads.")
    else:
        print(f"\n  {len(positive_gates)} gates to validate:\n")
        validated = []
        for pg in positive_gates:
            print(f"\n  {pg['feature']} {pg['gate_name']} "
                  f"(N={pg['n']}, edge={pg['edge']*100:+.2f}pp, Z={pg['z']:+.2f}):")
            result = split_half_full(dataset, pg["feature"], pg["thresh"], pg["direction"])
            result["gate"] = pg
            validated.append(result)

        # ═══════════════════════════════════════════════════════
        # 3. SUMMARY TABLE
        # ═══════════════════════════════════════════════════════
        print(f"\n\n{'='*75}")
        print(f"  PART 3: VALIDATION SUMMARY")
        print(f"{'='*75}")
        print(f"{'Feature':>15} {'Gate':>10} {'N':>6} {'Edge':>8} {'Chrono':>7} {'Random':>7} {'Fold':>7} {'ALL':>5}")
        print(f"{'─'*75}")
        for v in validated:
            g = v["gate"]
            print(f"{g['feature']:>15} {g['gate_name']:>10} {g['n']:>6} "
                  f"{g['edge']*100:>+7.2f}pp "
                  f"{'PASS' if v.get('chrono') else 'FAIL':>7} "
                  f"{'PASS' if v.get('random') else 'FAIL':>7} "
                  f"{'PASS' if v.get('fold') else 'FAIL':>7} "
                  f"{'YES' if v.get('all_pass') else 'no':>5}")

        # ═══════════════════════════════════════════════════════
        # 4. COMBINATION TEST (if any single gate passed)
        # ═══════════════════════════════════════════════════════
        passed = [v for v in validated if v.get("all_pass")]
        if passed:
            best = passed[0]
            print(f"\n\n{'='*75}")
            print(f"  PART 4: COMBINATION TEST — best gate + second feature")
            print(f"{'='*75}")
            best_feat = best["gate"]["feature"]
            best_thresh = best["gate"]["thresh"]
            best_dir = best["gate"]["direction"]

            # Apply best gate first
            if best_dir == ">=":
                gated = [d for d in dataset if d.get(best_feat) is not None
                         and d[best_feat] >= best_thresh]
            elif best_dir == "<=":
                gated = [d for d in dataset if d.get(best_feat) is not None
                         and d[best_feat] <= best_thresh]
            else:
                gated = dataset

            print(f"  Base gate: {best_feat} {best['gate']['gate_name']} → N={len(gated)}")

            # Add each other feature as AND gate
            other_features = {
                "dist_wp": [0.70, 0.75, 0.80, 0.85],
                "ridge": [0.70, 0.75, 0.80, 0.85],
                "bayes_wp": [0.60, 0.70, 0.80],
                "negbin_wp": [0.85, 0.90, 0.95],
                "edge_pp_low": [4, 6, 8, 10],
                "mu_away_low": [3.5, 4.0, 4.5],
                "margin": [0.0, 0.5, 1.0],
            }
            for feat2, thresholds2 in other_features.items():
                if feat2 == best_feat:
                    continue
                actual_feat = feat2.replace("_low", "")
                dir2 = "<=" if "_low" in feat2 else ">="

                print(f"\n  + {feat2}:")
                print(f"    {'Gate':>20} {'N':>6} {'Edge':>8} {'Z':>6}  {'Corner?':>8}")
                for t2 in thresholds2:
                    if dir2 == ">=":
                        combo = [d for d in gated if d.get(actual_feat) is not None
                                 and d[actual_feat] >= t2]
                    else:
                        combo = [d for d in gated if d.get(actual_feat) is not None
                                 and d[actual_feat] <= t2]
                    s = edge_stats(combo)
                    if s["n"] < 10:
                        continue
                    corner = "CORNER" if s["n"] < 150 else ""
                    print(f"    {dir2} {t2:>6}          {s['n']:>6} "
                          f"{s['edge']*100:>+7.2f}pp {s['z']:>+5.2f}  {corner:>8}")
        else:
            # No single gate passed all three — still do combo test on best-edge gate
            print(f"\n\n{'='*75}")
            print(f"  PART 4: COMBINATION TEST (no gate passed all three — using best edge)")
            print(f"{'='*75}")

            if positive_gates:
                best_pg = positive_gates[0]
                best_feat = best_pg["feature"]
                best_thresh = best_pg["thresh"]
                best_dir = best_pg["direction"]

                if best_dir == ">=":
                    gated = [d for d in dataset if d.get(best_feat) is not None
                             and d[best_feat] >= best_thresh]
                elif best_dir == "<=":
                    gated = [d for d in dataset if d.get(best_feat) is not None
                             and d[best_feat] <= best_thresh]
                else:
                    gated = dataset

                print(f"  Base gate: {best_feat} {best_pg['gate_name']} "
                      f"(N={len(gated)}, edge={best_pg['edge']*100:+.2f}pp)")

                other_features = {
                    "dist_wp": ([0.70, 0.75, 0.80, 0.85], ">="),
                    "ridge": ([0.70, 0.75, 0.80, 0.85], ">="),
                    "bayes_wp": ([0.60, 0.70, 0.80], ">="),
                    "negbin_wp": ([0.85, 0.90, 0.95], ">="),
                    "edge_pp": ([4, 6, 8, 10], "<="),
                    "mu_away": ([3.5, 4.0, 4.5], "<="),
                    "margin": ([0.0, 0.5, 1.0], ">="),
                    "rung": ([1.5, 2.5, 3.5], "<="),
                }
                for feat2, (thresholds2, dir2) in other_features.items():
                    if feat2 == best_feat:
                        continue
                    print(f"\n  + {feat2} ({dir2}):")
                    print(f"    {'Thresh':>10} {'N':>6} {'Edge':>8} {'Z':>6}  {'Flag':>8}")
                    for t2 in thresholds2:
                        if dir2 == ">=":
                            combo = [d for d in gated if d.get(feat2) is not None
                                     and d[feat2] >= t2]
                        else:
                            combo = [d for d in gated if d.get(feat2) is not None
                                     and d[feat2] <= t2]
                        s = edge_stats(combo)
                        if s["n"] < 10:
                            continue
                        corner = "CORNER" if s["n"] < 150 else ""
                        improved = "BETTER" if s["edge"] > best_pg["edge"] else ""
                        flag = corner or improved
                        print(f"    {t2:>10} {s['n']:>6} "
                              f"{s['edge']*100:>+7.2f}pp {s['z']:>+5.2f}  {flag:>8}")

    # ═══════════════════════════════════════════════════════════
    # 5. THE ISOLATION VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'='*75}")
    print(f"  PART 5: ISOLATION VERDICT")
    print(f"{'='*75}")

    any_positive = any(r["edge"] > 0 and r["n"] >= 100 for r in all_results)
    any_passed_all = any(v.get("all_pass") for v in (validated if positive_gates else []))

    if any_passed_all:
        winners = [v for v in validated if v.get("all_pass")]
        print(f"\n  SINGLE-FEATURE GATES THAT PASS ALL THREE:")
        for w in winners:
            g = w["gate"]
            print(f"    {g['feature']} {g['gate_name']}: "
                  f"N={g['n']} edge={g['edge']*100:+.2f}pp Z={g['z']:+.2f}")
    elif any_positive:
        print(f"\n  Some gates show positive edge but NONE pass all three validation tests.")
        print(f"  The positive edges are unstable — likely noise or hot streaks.")
        print(f"  Conclusion: no individual feature reliably beats the spread vig.")
    else:
        print(f"\n  NO single-feature gate produces positive edge-over-breakeven at N >= 100.")
        print(f"  EVERY feature is individually sub-vig on spreads.")
        print(f"  This is definitive: it's not a bundling/interaction problem —")
        print(f"  the signals themselves don't clear the vig.")


if __name__ == "__main__":
    main()
