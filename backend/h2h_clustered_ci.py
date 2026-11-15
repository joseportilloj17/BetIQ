#!/usr/bin/env python3
"""
Game-clustered bootstrap CI on gamma vs lognormal ROI and Brier.
Split in-sample (W2-W5, base rate training) vs out-of-sample (W6-W8).
"""
import os, sys, sqlite3, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
YARDAGE_STATS = {"pass_yds", "rush_yds", "rec_yds"}
N_BOOT = 5000
RNG = np.random.default_rng(42)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, season, week, player_name, position, stat_type,
               consensus_mean, prop_line, p_over, p_over_v12_gamma,
               devigged_implied, actual_value, is_win,
               prop_over_price, prop_under_price,
               line_depth, game_id
        FROM fantasy_forward_preds
        WHERE actual_value IS NOT NULL
          AND p_over IS NOT NULL
          AND p_over_v12_gamma IS NOT NULL
          AND method = 'distributional'
          AND data_error IS NULL
          AND line_depth = 'main'
          AND stat_type IN ('pass_yds', 'rush_yds', 'rec_yds')
        ORDER BY week, game_id, player_name
    """).fetchall()
    conn.close()

    rows = [dict(r) for r in rows]
    print(f"Main-line yardage rows with both P(over): {len(rows)}")

    # Compute per-row metrics
    for r in rows:
        actual = r["actual_value"]
        line = r["prop_line"]
        if actual == line:
            r["push"] = True
            continue
        r["push"] = False
        outcome = 1.0 if actual > line else 0.0
        r["brier_ln"] = (r["p_over"] - outcome) ** 2
        r["brier_gm"] = (r["p_over_v12_gamma"] - outcome) ** 2
        r["brier_diff"] = r["brier_gm"] - r["brier_ln"]

        # ROI for gamma prereg rule (edge >= 3pp)
        devig = r["devigged_implied"]
        p_gm = r["p_over_v12_gamma"]
        p_ln = r["p_over"]
        over_price = r["prop_over_price"]

        r["gm_edge"] = (p_gm - devig) if devig is not None else None
        r["ln_edge"] = (p_ln - devig) if devig is not None else None

        if over_price and over_price > 1.0:
            dec_odds = over_price
            r["pnl_if_bet"] = (dec_odds - 1) if actual > line else -1
            r["dec_odds"] = dec_odds
        else:
            r["pnl_if_bet"] = None
            r["dec_odds"] = None

    # Remove pushes
    rows = [r for r in rows if not r["push"]]
    print(f"After push removal: {len(rows)}")

    # Assign game clusters
    # Use game_id where available, else week as cluster
    for r in rows:
        r["cluster"] = r["game_id"] or f"week_{r['week']}"

    clusters = sorted(set(r["cluster"] for r in rows))
    cluster_to_rows = {}
    for r in rows:
        cluster_to_rows.setdefault(r["cluster"], []).append(r)

    print(f"Game clusters: {len(clusters)}")

    def game_clustered_bootstrap(rows_subset, metric_fn, n_boot=N_BOOT):
        """Bootstrap resampling at game-cluster level."""
        clust_map = {}
        for r in rows_subset:
            clust_map.setdefault(r["cluster"], []).append(r)
        cluster_keys = list(clust_map.keys())
        n_clusters = len(cluster_keys)
        boot_stats = []
        for _ in range(n_boot):
            sampled_keys = RNG.choice(cluster_keys, size=n_clusters, replace=True)
            sampled_rows = []
            for k in sampled_keys:
                sampled_rows.extend(clust_map[k])
            val = metric_fn(sampled_rows)
            if val is not None:
                boot_stats.append(val)
        boot_stats = np.array(boot_stats)
        return np.percentile(boot_stats, [2.5, 97.5]), np.mean(boot_stats), np.std(boot_stats)

    # ── Brier delta ──
    def brier_delta(rows_sub):
        diffs = [r["brier_diff"] for r in rows_sub if "brier_diff" in r]
        return np.mean(diffs) if diffs else None

    # ── ROI (gamma, edge >= 3pp) ──
    def roi_gm_3pp(rows_sub):
        bets = [r["pnl_if_bet"] for r in rows_sub
                if r.get("gm_edge") is not None and r["gm_edge"] >= 0.03
                and r.get("pnl_if_bet") is not None]
        return np.mean(bets) if bets else None

    # ── ROI (gamma, edge >= 5pp) ──
    def roi_gm_5pp(rows_sub):
        bets = [r["pnl_if_bet"] for r in rows_sub
                if r.get("gm_edge") is not None and r["gm_edge"] >= 0.05
                and r.get("pnl_if_bet") is not None]
        return np.mean(bets) if bets else None

    # ── ROI (lognormal, edge >= 3pp) ──
    def roi_ln_3pp(rows_sub):
        bets = [r["pnl_if_bet"] for r in rows_sub
                if r.get("ln_edge") is not None and r["ln_edge"] >= 0.03
                and r.get("pnl_if_bet") is not None]
        return np.mean(bets) if bets else None

    # ── WR (gamma, edge >= 3pp) ──
    def wr_gm_3pp(rows_sub):
        bets = [r["pnl_if_bet"] for r in rows_sub
                if r.get("gm_edge") is not None and r["gm_edge"] >= 0.03
                and r.get("pnl_if_bet") is not None]
        return np.mean([1 if p > 0 else 0 for p in bets]) if bets else None

    def report(label, metric_fn, rows_sub):
        ci, mean, se = game_clustered_bootstrap(rows_sub, metric_fn)
        point = metric_fn(rows_sub)
        n_obs = len(rows_sub)
        n_clust = len(set(r["cluster"] for r in rows_sub))
        print(f"  {label}: point={point:+.4f}, mean={mean:+.4f}, "
              f"95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}], "
              f"N={n_obs}, clusters={n_clust}")

    # ═══════════════════════════════════════════════════
    # OVERALL
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("OVERALL (main-line yardage, all weeks)")
    print("=" * 70)
    report("Brier Δ (gamma-ln)", brier_delta, rows)
    report("ROI gamma @≥3pp", roi_gm_3pp, rows)
    report("ROI gamma @≥5pp", roi_gm_5pp, rows)
    report("ROI lognormal @≥3pp", roi_ln_3pp, rows)
    report("WR gamma @≥3pp", wr_gm_3pp, rows)

    # ═══════════════════════════════════════════════════
    # IN-SAMPLE (W2-W5) vs OUT-OF-SAMPLE (W6-W8)
    # ═══════════════════════════════════════════════════
    in_sample = [r for r in rows if r["week"] <= 5]
    oos = [r for r in rows if r["week"] >= 6]

    print("\n" + "=" * 70)
    print(f"IN-SAMPLE W2-W5 (base rate training data, N={len(in_sample)})")
    print("=" * 70)
    report("Brier Δ (gamma-ln)", brier_delta, in_sample)
    report("ROI gamma @≥3pp", roi_gm_3pp, in_sample)
    report("ROI gamma @≥5pp", roi_gm_5pp, in_sample)
    report("ROI lognormal @≥3pp", roi_ln_3pp, in_sample)
    report("WR gamma @≥3pp", wr_gm_3pp, in_sample)

    print("\n" + "=" * 70)
    print(f"OUT-OF-SAMPLE W6-W8 (N={len(oos)})")
    print("=" * 70)
    report("Brier Δ (gamma-ln)", brier_delta, oos)
    report("ROI gamma @≥3pp", roi_gm_3pp, oos)
    report("ROI gamma @≥5pp", roi_gm_5pp, oos)
    report("ROI lognormal @≥3pp", roi_ln_3pp, oos)
    report("WR gamma @≥3pp", wr_gm_3pp, oos)

    # ═══════════════════════════════════════════════════
    # BY STAT (overall)
    # ═══════════════════════════════════════════════════
    for stat in sorted(YARDAGE_STATS):
        subset = [r for r in rows if r["stat_type"] == stat]
        print(f"\n--- {stat.upper()} (N={len(subset)}) ---")
        report("Brier Δ", brier_delta, subset)
        report("ROI gamma @≥3pp", roi_gm_3pp, subset)

    print("\n✓ Clustered CI analysis complete")


if __name__ == "__main__":
    main()
