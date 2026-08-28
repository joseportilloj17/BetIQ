"""
Phase 1b: Compare batter prop model accuracy vs market implied probabilities.
Overlap window: Jun 10-17 2026 (8 days where we have both outcomes and prop odds).
De-vig the prop odds (8-11% vig) to get fair market probabilities.
"""

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import warnings
warnings.filterwarnings("ignore")

HIST_DB = "data/historical.db"


def devig_odds(over_price, under_price):
    """Remove vig from 2-way market (over/under) odds. Returns devigged over prob."""
    if over_price <= 1.0 or under_price <= 1.0:
        return None
    imp_over = 1.0 / over_price
    imp_under = 1.0 / under_price
    total = imp_over + imp_under
    if total <= 0:
        return None
    return imp_over / total


def run():
    db = sqlite3.connect(HIST_DB)
    db.row_factory = sqlite3.Row

    # ── Get batter outcomes in the overlap window ──
    print("Loading batter outcomes (Jun 10-17 2026)...")
    outcomes = pd.read_sql_query("""
        SELECT b.player_name, b.ab, b.hits, b.hr, b.tb,
               g.game_date, b.game_pk, b.team_name
        FROM batter_game_stats b
        JOIN games g ON g.game_id = 'MLB_' || CAST(b.game_pk AS TEXT)
        WHERE g.game_date >= '2026-06-10' AND g.game_date <= '2026-06-17'
          AND b.ab >= 2
    """, db)
    print(f"  Batter rows: {len(outcomes)}")

    # ── Get prop odds ──
    print("Loading prop odds...")
    props = pd.read_sql_query("""
        SELECT player_name, game_date, market_key, bookmaker,
               over_under, price, point
        FROM player_props
        WHERE sport_key = 'baseball_mlb'
          AND game_date >= '2026-06-10' AND game_date <= '2026-06-17'
          AND market_key IN ('batter_home_runs', 'batter_hits', 'batter_total_bases')
    """, db)
    print(f"  Prop rows: {len(props)}")

    # ── Pair over/under and devig ──
    print("Devigging prop odds...")
    # Group by (player_name, game_date, market_key, bookmaker, point)
    # and extract over/under prices
    grouped = props.groupby(["player_name", "game_date", "market_key", "bookmaker", "point"])

    devigged = []
    for (player, date, market, bk, point), group in grouped:
        over_rows = group[group["over_under"].str.lower() == "over"]
        under_rows = group[group["over_under"].str.lower() == "under"]
        if len(over_rows) == 0 or len(under_rows) == 0:
            continue
        over_price = over_rows.iloc[0]["price"]
        under_price = under_rows.iloc[0]["price"]
        dv = devig_odds(over_price, under_price)
        if dv is None:
            continue

        vig = (1/over_price + 1/under_price - 1) * 100

        devigged.append({
            "player_name": player,
            "game_date": date,
            "market_key": market,
            "bookmaker": bk,
            "point": point,
            "over_price": over_price,
            "under_price": under_price,
            "devigged_over": dv,
            "vig_pct": vig,
        })

    dv_df = pd.DataFrame(devigged)
    print(f"  Devigged pairs: {len(dv_df)}")

    if len(dv_df) == 0:
        print("No devigged data!")
        return

    # ── Vig analysis ──
    print(f"\n--- Vig Analysis ---")
    for market in ["batter_home_runs", "batter_hits", "batter_total_bases"]:
        sub = dv_df[dv_df["market_key"] == market]
        if len(sub) > 0:
            print(f"  {market}: mean vig = {sub['vig_pct'].mean():.1f}%, "
                  f"median = {sub['vig_pct'].median():.1f}%, N={len(sub)}")

    # ── Use pinnacle (sharpest) or average across books ──
    # Prefer pinnacle for sharp pricing; fall back to mean across books
    market_map = {
        "batter_home_runs": ("hr", 0.5),       # HR ≥ 0.5 = HR ≥ 1
        "batter_hits": ("hits", 0.5),           # Hits ≥ 0.5 = Hits ≥ 1
        "batter_total_bases": ("tb", 1.5),      # TB ≥ 1.5 = TB ≥ 2
    }

    results = []
    for market_key, (stat, threshold) in market_map.items():
        print(f"\n{'='*50}")
        if stat == "hr":
            target_name = "HR ≥ 1"
        elif stat == "hits":
            target_name = "Hits ≥ 1"
        else:
            target_name = "TB ≥ 2"

        # Filter devigged odds for this market + standard point
        # HR: point=0.5 (over 0.5 = at least 1 HR)
        # Hits: point=0.5 (over 0.5 = at least 1 hit)
        # TB: point=1.5 (over 1.5 = at least 2 TB)
        market_odds = dv_df[(dv_df["market_key"] == market_key) &
                            (dv_df["point"] == threshold)].copy()

        if len(market_odds) == 0:
            print(f"  {target_name}: No odds at point={threshold}")
            # Try other points
            available = dv_df[dv_df["market_key"] == market_key]["point"].unique()
            print(f"  Available points: {sorted(available)}")
            continue

        # Average across bookmakers per (player, date)
        avg_odds = market_odds.groupby(["player_name", "game_date"]).agg(
            market_prob=("devigged_over", "mean"),
            n_books=("bookmaker", "count"),
        ).reset_index()

        # Join with outcomes
        merged = avg_odds.merge(outcomes, on=["player_name", "game_date"], how="inner")

        if len(merged) == 0:
            print(f"  {target_name}: No outcome matches!")
            continue

        # Compute actual outcome
        if stat == "hr":
            merged["actual"] = (merged["hr"] >= 1).astype(int)
        elif stat == "hits":
            merged["actual"] = (merged["hits"] >= 1).astype(int)
        else:
            merged["actual"] = (merged["tb"] >= 2).astype(int)

        y_true = merged["actual"].values
        market_pred = merged["market_prob"].values
        base_rate = y_true.mean()

        print(f"--- {target_name} ---")
        print(f"  Matched rows: {len(merged)} ({merged['player_name'].nunique()} players, "
              f"{merged['game_date'].nunique()} dates)")
        print(f"  Base rate: {base_rate*100:.1f}%")

        # Market accuracy
        market_auc = roc_auc_score(y_true, market_pred)
        market_brier = brier_score_loss(y_true, market_pred)
        market_ll = log_loss(y_true, market_pred)

        print(f"  Market (devigged): AUC={market_auc:.4f}  Brier={market_brier:.4f}  LogLoss={market_ll:.4f}")
        print(f"  Market mean pred: {market_pred.mean()*100:.1f}% vs actual: {base_rate*100:.1f}%")

        # Naive baseline: batter's season-to-date rate
        # Compute from batter_game_stats (pre-Jun 10 data)
        season_rates = {}
        for player in merged["player_name"].unique():
            pre = db.execute("""
                SELECT SUM(hr) as total_hr, SUM(hits) as total_hits,
                       SUM(tb) as total_tb, SUM(ab) as total_ab
                FROM batter_game_stats b
                JOIN games g ON g.game_id = 'MLB_' || CAST(b.game_pk AS TEXT)
                WHERE b.player_name = ? AND g.game_date < '2026-06-10'
                  AND g.game_date >= '2026-01-01'
            """, (player,)).fetchone()
            if pre and pre["total_ab"] and pre["total_ab"] > 30:
                ab = pre["total_ab"]
                if stat == "hr":
                    rate = pre["total_hr"] / ab
                elif stat == "hits":
                    rate = pre["total_hits"] / ab
                else:
                    rate = pre["total_tb"] / ab
                season_rates[player] = rate

        # Poisson approximation for event probability
        naive_preds = []
        valid_mask = []
        for _, row in merged.iterrows():
            rate = season_rates.get(row["player_name"])
            if rate is not None:
                # P(X >= threshold+1) using Poisson(lambda = rate * expected_AB)
                expected_ab = row["ab"]  # use actual AB for fair comparison
                lam = rate * expected_ab
                if stat == "hr":
                    prob = 1 - np.exp(-lam)  # P(≥1)
                elif stat == "hits":
                    prob = 1 - np.exp(-lam)  # P(≥1)
                else:
                    prob = 1 - np.exp(-lam) - lam * np.exp(-lam)  # P(≥2)
                naive_preds.append(np.clip(prob, 0.001, 0.999))
                valid_mask.append(True)
            else:
                naive_preds.append(base_rate)
                valid_mask.append(False)

        naive_preds = np.array(naive_preds)
        naive_auc = roc_auc_score(y_true, naive_preds)
        naive_brier = brier_score_loss(y_true, naive_preds)

        print(f"  Naive (season rate): AUC={naive_auc:.4f}  Brier={naive_brier:.4f}")

        # Show market calibration (quintiles)
        print(f"  Market calibration by quintile:")
        quintile_edges = np.percentile(market_pred, np.arange(0, 101, 20))
        for i in range(5):
            lo, hi = quintile_edges[i], quintile_edges[i+1]
            mask = (market_pred >= lo) & (market_pred < hi + 0.0001)
            if mask.sum() == 0:
                continue
            pred_mean = market_pred[mask].mean()
            actual_mean = y_true[mask].mean()
            n = mask.sum()
            print(f"    Q{i}: pred={pred_mean*100:.1f}% actual={actual_mean*100:.1f}% N={n}")

        results.append({
            "target": target_name,
            "base_rate": base_rate,
            "market_auc": market_auc,
            "market_brier": market_brier,
            "naive_auc": naive_auc,
            "naive_brier": naive_brier,
            "n": len(merged),
        })

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"=== MARKET COMPARISON SUMMARY ===")
    print(f"{'='*60}")
    print(f"{'Target':<10s} {'Base%':>6s} {'Market':>7s} {'Naive':>7s} {'Δ':>7s} {'N':>6s}")
    for r in results:
        delta_auc = r["market_auc"] - r["naive_auc"]
        print(f"{r['target']:<10s} {r['base_rate']*100:5.1f}% "
              f"AUC={r['market_auc']:.4f} AUC={r['naive_auc']:.4f} "
              f"{delta_auc:+.4f} {r['n']:5d}")

    print(f"\nOur GBC model AUCs from Phase 1 (full 2026 test):")
    print(f"  HR ≥ 1: 0.6175  |  Hits ≥ 1: 0.5547  |  Hits ≥ 2: 0.5762  |  TB ≥ 2: 0.5580")
    print(f"\nIf Market AUC > GBC AUC → market is more accurate → no edge.")
    print(f"If Market AUC < GBC AUC → model might have edge (BUT: 8-day sample, "
          f"need forward validation).")

    db.close()


if __name__ == "__main__":
    run()
