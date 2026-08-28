"""
Full-window star/role player miscalibration analysis.
Window: Jun 10 - Aug 13 2026 (all dates with both prop odds and batter outcomes).
Tests whether market systematically misprices HR props for stars vs role players.
Includes split-half validation and significance testing.
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

HIST_DB = "data/historical.db"


def devig_odds(over_price, under_price):
    """Remove vig from 2-way market. Returns devigged over probability."""
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

    # ── Load all HR prop odds in the full window ──
    print("Loading HR prop odds (full window)...")
    props = pd.read_sql_query("""
        SELECT player_name, game_date, bookmaker,
               over_under, price, point
        FROM player_props
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'batter_home_runs'
          AND game_date >= '2026-06-10' AND game_date <= '2026-08-13'
    """, db)
    print(f"  Raw prop rows: {len(props)}")
    print(f"  Date range: {props['game_date'].min()} to {props['game_date'].max()}")
    print(f"  Distinct dates: {props['game_date'].nunique()}")

    # ── Load batter outcomes ──
    print("\nLoading batter outcomes...")
    outcomes = pd.read_sql_query("""
        SELECT b.player_name, b.ab, b.hits, b.hr, b.tb,
               g.game_date, b.game_pk, b.team_name
        FROM batter_game_stats b
        JOIN games g ON g.game_id = 'MLB_' || CAST(b.game_pk AS TEXT)
        WHERE g.game_date >= '2026-06-10' AND g.game_date <= '2026-08-13'
          AND b.ab >= 2
    """, db)
    print(f"  Batter outcome rows: {len(outcomes)}")
    print(f"  Date range: {outcomes['game_date'].min()} to {outcomes['game_date'].max()}")

    # ── Devig HR props (point=0.5, i.e. Over 0.5 = at least 1 HR) ──
    print("\nDevigging HR props (point=0.5)...")
    hr_props = props[props["point"] == 0.5].copy()
    grouped = hr_props.groupby(["player_name", "game_date", "bookmaker"])

    devigged = []
    for (player, date, bk), group in grouped:
        over_rows = group[group["over_under"].str.lower() == "over"]
        under_rows = group[group["over_under"].str.lower() == "under"]
        if len(over_rows) == 0 or len(under_rows) == 0:
            continue
        over_price = over_rows.iloc[0]["price"]
        under_price = under_rows.iloc[0]["price"]
        dv = devig_odds(over_price, under_price)
        if dv is None:
            continue
        devigged.append({
            "player_name": player,
            "game_date": date,
            "bookmaker": bk,
            "devigged_over": dv,
        })

    dv_df = pd.DataFrame(devigged)
    print(f"  Devigged pairs: {len(dv_df)}")

    # Average across bookmakers per (player, date)
    avg_odds = dv_df.groupby(["player_name", "game_date"]).agg(
        market_prob=("devigged_over", "mean"),
        n_books=("bookmaker", "count"),
    ).reset_index()
    print(f"  Unique (player, date) pairs: {len(avg_odds)}")

    # ── Join with outcomes ──
    merged = avg_odds.merge(outcomes, on=["player_name", "game_date"], how="inner")
    merged["hit_hr"] = (merged["hr"] >= 1).astype(int)
    print(f"  Matched rows: {len(merged)} ({merged['player_name'].nunique()} players, "
          f"{merged['game_date'].nunique()} dates)")
    print(f"  HR base rate: {merged['hit_hr'].mean()*100:.1f}%")

    if len(merged) < 100:
        print("Not enough matched data!")
        return

    # ── Compute season HR rate for each player (pre-window data) ──
    print("\nComputing season HR rates (pre-Jun 10 data)...")
    season_rates = {}
    for player in merged["player_name"].unique():
        pre = db.execute("""
            SELECT SUM(hr) as total_hr, SUM(ab) as total_ab
            FROM batter_game_stats b
            JOIN games g ON g.game_id = 'MLB_' || CAST(b.game_pk AS TEXT)
            WHERE b.player_name = ? AND g.game_date < '2026-06-10'
              AND g.game_date >= '2026-01-01'
        """, (player,)).fetchone()
        if pre and pre["total_ab"] and pre["total_ab"] > 30:
            season_rates[player] = pre["total_hr"] / pre["total_ab"]

    merged["szn_hr_rate"] = merged["player_name"].map(season_rates)
    has_rate = merged.dropna(subset=["szn_hr_rate"])
    print(f"  Players with season HR rate: {has_rate['player_name'].nunique()}/{merged['player_name'].nunique()}")
    print(f"  Rows with rate: {len(has_rate)}/{len(merged)}")

    # ── Define tiers by season HR rate ──
    print("\n" + "=" * 60)
    print("=== STAR/ROLE PLAYER TIER ANALYSIS (Full Window) ===")
    print("=" * 60)

    # Tier definitions based on HR rate per AB
    tiers = [
        ("Role (<2.5%)", 0.0, 0.025),
        ("Mid (2.5-4%)", 0.025, 0.04),
        ("Star (>4%)", 0.04, 1.0),
    ]

    tier_results = []
    for tier_name, lo, hi in tiers:
        sub = has_rate[(has_rate["szn_hr_rate"] >= lo) & (has_rate["szn_hr_rate"] < hi)]
        if len(sub) < 30:
            print(f"\n{tier_name}: N={len(sub)} (too few)")
            continue

        actual = sub["hit_hr"].mean()
        market = sub["market_prob"].mean()
        gap = actual - market  # positive = market underprices (actual > predicted)
        n = len(sub)
        n_players = sub["player_name"].nunique()

        # Standard error of the gap
        se = np.sqrt(actual * (1 - actual) / n)

        # Binomial test: is actual rate significantly different from market prediction?
        # Use the market prob as the null hypothesis
        successes = sub["hit_hr"].sum()
        binom_p = stats.binomtest(successes, n, market, alternative="two-sided").pvalue

        print(f"\n--- {tier_name} ---")
        print(f"  N={n} obs, {n_players} players, {sub['game_date'].nunique()} dates")
        print(f"  Season HR rate: {sub['szn_hr_rate'].mean()*100:.2f}%")
        print(f"  Market pred (devigged): {market*100:.2f}%")
        print(f"  Actual HR rate: {actual*100:.2f}%")
        print(f"  Gap (actual - market): {gap*100:+.2f}pp")
        print(f"  SE: {se*100:.2f}pp")
        print(f"  z-score: {gap/se:.2f}")
        print(f"  Binomial p-value: {binom_p:.4f}")

        tier_results.append({
            "tier": tier_name,
            "n": n,
            "n_players": n_players,
            "market": market,
            "actual": actual,
            "gap_pp": gap * 100,
            "z": gap / se,
            "p": binom_p,
        })

    # ── Overall calibration ──
    print(f"\n{'='*60}")
    print("=== OVERALL CALIBRATION ===")
    overall_actual = has_rate["hit_hr"].mean()
    overall_market = has_rate["market_prob"].mean()
    overall_gap = overall_actual - overall_market
    overall_se = np.sqrt(overall_actual * (1 - overall_actual) / len(has_rate))
    print(f"  N={len(has_rate)}, {has_rate['player_name'].nunique()} players")
    print(f"  Market pred: {overall_market*100:.2f}%  Actual: {overall_actual*100:.2f}%  "
          f"Gap: {overall_gap*100:+.2f}pp  z={overall_gap/overall_se:.2f}")

    # ── Split-half validation ──
    print(f"\n{'='*60}")
    print("=== SPLIT-HALF VALIDATION ===")
    print("(Random 50/50 split, repeated 100 times)")

    np.random.seed(42)
    star_gaps_a, star_gaps_b = [], []
    role_gaps_a, role_gaps_b = [], []

    star_sub = has_rate[has_rate["szn_hr_rate"] >= 0.04]
    role_sub = has_rate[has_rate["szn_hr_rate"] < 0.025]

    for _ in range(100):
        # Split star
        if len(star_sub) >= 20:
            idx = np.random.permutation(len(star_sub))
            half = len(idx) // 2
            a_star = star_sub.iloc[idx[:half]]
            b_star = star_sub.iloc[idx[half:]]
            star_gaps_a.append((a_star["hit_hr"].mean() - a_star["market_prob"].mean()) * 100)
            star_gaps_b.append((b_star["hit_hr"].mean() - b_star["market_prob"].mean()) * 100)

        # Split role
        if len(role_sub) >= 20:
            idx = np.random.permutation(len(role_sub))
            half = len(idx) // 2
            a_role = role_sub.iloc[idx[:half]]
            b_role = role_sub.iloc[idx[half:]]
            role_gaps_a.append((a_role["hit_hr"].mean() - a_role["market_prob"].mean()) * 100)
            role_gaps_b.append((b_role["hit_hr"].mean() - b_role["market_prob"].mean()) * 100)

    if star_gaps_a:
        corr_star = np.corrcoef(star_gaps_a, star_gaps_b)[0, 1]
        print(f"\nStar (>4% HR rate):")
        print(f"  Split-half gap correlation: r={corr_star:.3f}")
        print(f"  Half A mean gap: {np.mean(star_gaps_a):+.2f}pp  Half B: {np.mean(star_gaps_b):+.2f}pp")
        print(f"  Both halves same sign: {sum(1 for a, b in zip(star_gaps_a, star_gaps_b) if np.sign(a) == np.sign(b))}%")

    if role_gaps_a:
        corr_role = np.corrcoef(role_gaps_a, role_gaps_b)[0, 1]
        print(f"\nRole (<2.5% HR rate):")
        print(f"  Split-half gap correlation: r={corr_role:.3f}")
        print(f"  Half A mean gap: {np.mean(role_gaps_a):+.2f}pp  Half B: {np.mean(role_gaps_b):+.2f}pp")
        print(f"  Both halves same sign: {sum(1 for a, b in zip(role_gaps_a, role_gaps_b) if np.sign(a) == np.sign(b))}%")

    # ── Time stability: first half vs second half of window ──
    print(f"\n{'='*60}")
    print("=== TIME STABILITY (First half vs second half) ===")
    dates_sorted = sorted(has_rate["game_date"].unique())
    mid_date = dates_sorted[len(dates_sorted) // 2]
    print(f"  Split date: {mid_date}")

    for tier_name, lo, hi in tiers:
        sub = has_rate[(has_rate["szn_hr_rate"] >= lo) & (has_rate["szn_hr_rate"] < hi)]
        first = sub[sub["game_date"] <= mid_date]
        second = sub[sub["game_date"] > mid_date]
        if len(first) < 20 or len(second) < 20:
            continue
        gap1 = (first["hit_hr"].mean() - first["market_prob"].mean()) * 100
        gap2 = (second["hit_hr"].mean() - second["market_prob"].mean()) * 100
        print(f"  {tier_name}: First half gap={gap1:+.2f}pp (N={len(first)})  "
              f"Second half gap={gap2:+.2f}pp (N={len(second)})  "
              f"{'CONSISTENT' if np.sign(gap1) == np.sign(gap2) else 'FLIP'}")

    # ── Market quintile calibration (all players) ──
    print(f"\n{'='*60}")
    print("=== MARKET QUINTILE CALIBRATION ===")
    quintile_edges = np.percentile(has_rate["market_prob"], np.arange(0, 101, 20))
    for i in range(5):
        lo_q, hi_q = quintile_edges[i], quintile_edges[i + 1]
        mask = (has_rate["market_prob"] >= lo_q) & (has_rate["market_prob"] < hi_q + 0.0001)
        if mask.sum() == 0:
            continue
        pred = has_rate.loc[mask, "market_prob"].mean()
        act = has_rate.loc[mask, "hit_hr"].mean()
        n = mask.sum()
        print(f"  Q{i}: pred={pred*100:.1f}% actual={act*100:.1f}% gap={((act-pred)*100):+.1f}pp N={n}")

    # ── Betting simulation: flat bet OVER on stars, UNDER on role players ──
    print(f"\n{'='*60}")
    print("=== BETTING SIMULATION (flat $10 bets) ===")

    # Get actual over/under prices for simulation
    over_prices = props[(props["point"] == 0.5) & (props["over_under"].str.lower() == "over")]
    under_prices = props[(props["point"] == 0.5) & (props["over_under"].str.lower() == "under")]

    # Best available over price per (player, date)
    best_over = over_prices.groupby(["player_name", "game_date"])["price"].max().reset_index()
    best_over.columns = ["player_name", "game_date", "best_over_price"]
    best_under = under_prices.groupby(["player_name", "game_date"])["price"].max().reset_index()
    best_under.columns = ["player_name", "game_date", "best_under_price"]

    sim = has_rate.merge(best_over, on=["player_name", "game_date"], how="left")
    sim = sim.merge(best_under, on=["player_name", "game_date"], how="left")

    # Strategy 1: OVER on stars
    stars = sim[sim["szn_hr_rate"] >= 0.04].dropna(subset=["best_over_price"])
    if len(stars) > 0:
        star_pnl = []
        for _, row in stars.iterrows():
            if row["hit_hr"] == 1:
                star_pnl.append(10 * (row["best_over_price"] - 1))
            else:
                star_pnl.append(-10)
        total_star = sum(star_pnl)
        wr_star = stars["hit_hr"].mean()
        be_star = 1.0 / stars["best_over_price"].mean()
        print(f"\n  OVER on Stars (HR rate >= 4%):")
        print(f"    N bets: {len(stars)}")
        print(f"    Win rate: {wr_star*100:.1f}%  Breakeven: {be_star*100:.1f}%")
        print(f"    PnL: ${total_star:+.2f}  ROI: {total_star/(len(stars)*10)*100:+.1f}%")

    # Strategy 2: UNDER on role players
    roles = sim[sim["szn_hr_rate"] < 0.025].dropna(subset=["best_under_price"])
    if len(roles) > 0:
        role_pnl = []
        for _, row in roles.iterrows():
            if row["hit_hr"] == 0:  # Under wins when no HR
                role_pnl.append(10 * (row["best_under_price"] - 1))
            else:
                role_pnl.append(-10)
        total_role = sum(role_pnl)
        wr_role = (1 - roles["hit_hr"]).mean()
        be_role = 1.0 / roles["best_under_price"].mean()
        print(f"\n  UNDER on Role Players (HR rate < 2.5%):")
        print(f"    N bets: {len(roles)}")
        print(f"    Win rate: {wr_role*100:.1f}%  Breakeven: {be_role*100:.1f}%")
        print(f"    PnL: ${total_role:+.2f}  ROI: {total_role/(len(roles)*10)*100:+.1f}%")

    # ── SUMMARY ──
    print(f"\n{'='*60}")
    print("=== VERDICT ===")
    print("=" * 60)
    for r in tier_results:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        direction = "UNDER-priced (actual > market)" if r["gap_pp"] > 0 else "OVER-priced (actual < market)"
        print(f"  {r['tier']}: gap={r['gap_pp']:+.2f}pp  z={r['z']:.2f}  p={r['p']:.4f} {sig}")
        print(f"    → Market {direction}")

    db.close()


if __name__ == "__main__":
    run()
