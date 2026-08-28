"""
Phase 1: Batter prop prediction models.
Targets: HR ≥ 1, Hits ≥ 1, Hits ≥ 2, TB ≥ 2.
Features: batter rolling stats, pitcher matchup, batting order, home/away.
Validation: train 2022-2025, test 2026 (chrono), compare to naive base rate.
"""

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

HIST_DB = "data/historical.db"
BETS_DB = "data/bets.db"
MIN_SEASON_AB = 30      # Minimum season AB before using rolling features
MIN_RECENT_GAMES = 7    # Recent window
RECENT_WINDOW = 14      # Last N games for recent form


def load_data():
    """Load and join batter game stats with game dates, lineups, and pitcher matchups."""
    db = sqlite3.connect(HIST_DB)

    print("Loading batter game stats...")
    batter_df = pd.read_sql_query("""
        SELECT b.game_pk, b.player_id, b.player_name, b.team_name, b.is_home,
               b.ab, b.hits, b.doubles, b.triples, b.hr, b.bb, b.hbp, b.sf, b.k, b.tb,
               g.game_date,
               SUBSTR(g.game_date, 1, 4) AS season
        FROM batter_game_stats b
        JOIN games g ON g.game_id = 'MLB_' || CAST(b.game_pk AS TEXT)
        WHERE b.ab >= 2
        ORDER BY b.player_id, g.game_date
    """, db)
    print(f"  Loaded {len(batter_df)} rows ({batter_df['player_id'].nunique()} batters)")

    print("Loading lineups...")
    lineup_df = pd.read_sql_query("""
        SELECT game_pk, player_id, team_name, is_home, batting_order, position
        FROM game_lineups
    """, db)

    print("Loading pitcher game logs...")
    pitcher_logs = pd.read_sql_query("""
        SELECT game_pk, player_id AS pitcher_id, pitcher_name, ip
        FROM pitcher_game_logs
    """, db)

    print("Loading pitcher season stats...")
    pitcher_stats = pd.read_sql_query("""
        SELECT player_id AS pitcher_id, season, era, whip, k9, hr9, fip, throws,
               gs, ip AS season_ip, xera,
               CASE WHEN batters_faced > 0
                    THEN CAST(strikeouts AS REAL)/batters_faced
                    ELSE 0 END AS k_rate,
               CASE WHEN batters_faced > 0
                    THEN CAST(home_runs_allowed AS REAL)/batters_faced
                    ELSE 0 END AS hr_allowed_rate
        FROM pitcher_stats
    """, db)

    db.close()
    return batter_df, lineup_df, pitcher_logs, pitcher_stats


def compute_rolling_features(batter_df):
    """Compute point-in-time rolling features per batter."""
    print("Computing rolling features (point-in-time)...")

    # Sort by player and date
    batter_df = batter_df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # PA = AB + BB + HBP + SF
    batter_df["pa"] = batter_df["ab"] + batter_df["bb"] + batter_df["hbp"] + batter_df["sf"]

    # Rolling features: cumulative BEFORE current game (shift by 1)
    features = []
    for pid, group in batter_df.groupby("player_id"):
        g = group.copy()
        n = len(g)

        # Cumulative sums (shifted: exclude current game)
        cum_ab = g["ab"].cumsum().shift(1, fill_value=0)
        cum_hits = g["hits"].cumsum().shift(1, fill_value=0)
        cum_hr = g["hr"].cumsum().shift(1, fill_value=0)
        cum_tb = g["tb"].cumsum().shift(1, fill_value=0)
        cum_k = g["k"].cumsum().shift(1, fill_value=0)
        cum_bb = g["bb"].cumsum().shift(1, fill_value=0)
        cum_pa = g["pa"].cumsum().shift(1, fill_value=0)
        cum_doubles = g["doubles"].cumsum().shift(1, fill_value=0)

        # Season-to-date stats (need to reset at season boundaries)
        # For simplicity, use overall cumulative (cross-season).
        # The model will see season as a feature implicitly via the split.
        # We also add a season-level reset version.
        season_starts = g["season"].ne(g["season"].shift(1))
        season_cum_ab = g.groupby((season_starts).cumsum())["ab"].cumsum().shift(1, fill_value=0)
        season_cum_hits = g.groupby((season_starts).cumsum())["hits"].cumsum().shift(1, fill_value=0)
        season_cum_hr = g.groupby((season_starts).cumsum())["hr"].cumsum().shift(1, fill_value=0)
        season_cum_tb = g.groupby((season_starts).cumsum())["tb"].cumsum().shift(1, fill_value=0)
        season_cum_k = g.groupby((season_starts).cumsum())["k"].cumsum().shift(1, fill_value=0)
        season_cum_bb = g.groupby((season_starts).cumsum())["bb"].cumsum().shift(1, fill_value=0)
        season_cum_pa = g.groupby((season_starts).cumsum())["pa"].cumsum().shift(1, fill_value=0)
        season_games = g.groupby((season_starts).cumsum()).cumcount()  # games in this season so far

        # Season-to-date rates (with Bayesian shrinkage toward population)
        # Prior: MLB average rates
        PRIOR_AVG = 0.248
        PRIOR_SLG = 0.395
        PRIOR_ISO = 0.147
        PRIOR_HR_RATE = 0.033
        PRIOR_K_RATE = 0.225
        PRIOR_BB_RATE = 0.083
        PRIOR_WEIGHT = 50  # prior equivalent AB

        safe_ab = season_cum_ab.clip(lower=1)
        safe_pa = season_cum_pa.clip(lower=1)

        # Shrunk season stats
        g["szn_avg"] = (season_cum_hits + PRIOR_WEIGHT * PRIOR_AVG) / (season_cum_ab + PRIOR_WEIGHT)
        g["szn_slg"] = (season_cum_tb + PRIOR_WEIGHT * PRIOR_SLG) / (season_cum_ab + PRIOR_WEIGHT)
        g["szn_iso"] = g["szn_slg"] - g["szn_avg"]
        g["szn_hr_rate"] = (season_cum_hr + PRIOR_WEIGHT * PRIOR_HR_RATE) / (season_cum_ab + PRIOR_WEIGHT)
        g["szn_k_rate"] = (season_cum_k + PRIOR_WEIGHT * PRIOR_K_RATE) / (season_cum_pa + PRIOR_WEIGHT)
        g["szn_bb_rate"] = (season_cum_bb + PRIOR_WEIGHT * PRIOR_BB_RATE) / (season_cum_pa + PRIOR_WEIGHT)
        g["szn_ab"] = season_cum_ab
        g["szn_games"] = season_games

        # Recent form (last RECENT_WINDOW games rolling mean)
        g["recent_avg"] = g["hits"].rolling(RECENT_WINDOW, min_periods=MIN_RECENT_GAMES).mean().shift(1)
        g["recent_hr"] = g["hr"].rolling(RECENT_WINDOW, min_periods=MIN_RECENT_GAMES).mean().shift(1)
        g["recent_tb"] = g["tb"].rolling(RECENT_WINDOW, min_periods=MIN_RECENT_GAMES).mean().shift(1)
        g["recent_k"] = g["k"].rolling(RECENT_WINDOW, min_periods=MIN_RECENT_GAMES).mean().shift(1)
        g["recent_ab"] = g["ab"].rolling(RECENT_WINDOW, min_periods=MIN_RECENT_GAMES).mean().shift(1)

        features.append(g)

    result = pd.concat(features, ignore_index=True)

    # Drop rows with insufficient history
    result = result[result["szn_ab"] >= MIN_SEASON_AB].copy()
    print(f"  After minimum AB filter: {len(result)} rows")
    return result


def add_matchup_features(batter_df, lineup_df, pitcher_logs, pitcher_stats):
    """Add batting order + opposing pitcher stats.

    Pitcher team identification: for each pitcher, find the team that appears
    in ALL (or most) of their game appearances. A pitcher's own team is always
    one of the two teams in every game they pitch.
    """
    print("Adding matchup features...")

    db = sqlite3.connect(HIST_DB)

    # ── Batting order ──
    lineup_map = {}
    for _, row in lineup_df.iterrows():
        lineup_map[(row["game_pk"], row["player_id"])] = row["batting_order"]

    batter_df["batting_order"] = batter_df.apply(
        lambda r: lineup_map.get((r["game_pk"], r["player_id"]), 5), axis=1
    )

    # ── Build game_pk → (home_team, away_team) from games table ──
    games_info = pd.read_sql_query("""
        SELECT game_id, home_team, away_team
        FROM games WHERE sport='MLB'
    """, db)
    db.close()

    # game_pk is in game_id as 'MLB_<game_pk>'
    games_info["game_pk"] = games_info["game_id"].str.replace("MLB_", "").astype(int)
    gp_to_teams = dict(zip(games_info["game_pk"],
                           zip(games_info["home_team"], games_info["away_team"])))

    # ── Identify each pitcher's team via frequency analysis ──
    # For each pitcher, across all their game_pks, count how often each team appears.
    # The pitcher's team appears in EVERY game they pitch (either home or away).
    # Other teams appear only once or twice.
    print("  Identifying pitcher teams via game frequency...")

    pitcher_game_pks = defaultdict(list)
    for _, row in pitcher_logs.iterrows():
        pitcher_game_pks[row["pitcher_id"]].append(row["game_pk"])

    pitcher_team_map = {}  # pitcher_id → team_name
    for pid, gps in pitcher_game_pks.items():
        team_counts = defaultdict(int)
        for gp in gps:
            teams = gp_to_teams.get(gp)
            if teams:
                team_counts[teams[0]] += 1  # home
                team_counts[teams[1]] += 1  # away
        if team_counts:
            # The pitcher's team appears in ALL their games (highest count)
            best_team = max(team_counts, key=team_counts.get)
            best_count = team_counts[best_team]
            total_games = len(gps)
            # Confidence: should appear in 100% of games (or close, if traded)
            if best_count >= total_games * 0.6:  # allow for mid-season trades
                pitcher_team_map[pid] = best_team

    print(f"  Identified teams for {len(pitcher_team_map)}/{len(pitcher_game_pks)} pitchers "
          f"({len(pitcher_team_map)/len(pitcher_game_pks)*100:.1f}%)")

    # ── Build opposing pitcher features ──
    game_pitchers = defaultdict(list)
    for _, row in pitcher_logs.iterrows():
        game_pitchers[row["game_pk"]].append({
            "pitcher_id": row["pitcher_id"],
            "pitcher_name": row["pitcher_name"],
            "ip": row["ip"],
        })

    game_season = batter_df.groupby("game_pk")["season"].first().to_dict()

    ps_lookup = {}
    for _, row in pitcher_stats.iterrows():
        ps_lookup[(row["pitcher_id"], row["season"])] = row

    opp_rows = []
    matched = 0
    unmatched = 0

    for gp, pitchers in game_pitchers.items():
        season = game_season.get(gp)
        if not season:
            continue
        teams = gp_to_teams.get(gp)
        if not teams:
            continue
        home_team, away_team = teams

        for p in pitchers:
            pid = p["pitcher_id"]
            team = pitcher_team_map.get(pid)
            if not team:
                unmatched += 1
                continue

            # Determine which side's batters this pitcher faces
            if team == home_team:
                faces_side = 0  # home pitcher faces away batters (is_home=0)
            elif team == away_team:
                faces_side = 1  # away pitcher faces home batters (is_home=1)
            else:
                # Pitcher's team doesn't match either side (trade/error)
                unmatched += 1
                continue

            stats = ps_lookup.get((pid, season))
            if stats is None:
                unmatched += 1
                continue

            opp_rows.append({
                "game_pk": gp,
                "batter_is_home": faces_side,
                "opp_era": stats["era"],
                "opp_whip": stats["whip"],
                "opp_k9": stats["k9"],
                "opp_hr9": stats["hr9"],
                "opp_fip": stats["fip"],
                "opp_throws_l": 1 if stats["throws"] == "L" else 0,
                "opp_k_rate": stats["k_rate"],
                "opp_hr_allowed_rate": stats["hr_allowed_rate"],
            })
            matched += 1

    print(f"  Pitcher feature rows: {matched} matched, {unmatched} unmatched")

    opp_df = pd.DataFrame(opp_rows)
    if len(opp_df) > 0:
        batter_df = batter_df.merge(
            opp_df, left_on=["game_pk", "is_home"],
            right_on=["game_pk", "batter_is_home"],
            how="left"
        )
        fill_rate = batter_df["opp_era"].notna().sum()
        print(f"  Opposing pitcher fill rate: {fill_rate}/{len(batter_df)} "
              f"({fill_rate/len(batter_df)*100:.1f}%)")
    else:
        print("  WARNING: No opposing pitcher features!")
        for col in ["opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip",
                     "opp_throws_l", "opp_k_rate", "opp_hr_allowed_rate"]:
            batter_df[col] = np.nan

    return batter_df


def build_features_and_targets(df):
    """Select feature columns and create target variables."""
    feature_cols = [
        # Batter season-to-date (shrunk)
        "szn_avg", "szn_slg", "szn_iso", "szn_hr_rate",
        "szn_k_rate", "szn_bb_rate", "szn_ab", "szn_games",
        # Recent form
        "recent_avg", "recent_hr", "recent_tb", "recent_k", "recent_ab",
        # Context
        "is_home", "batting_order",
        # Opposing pitcher
        "opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip",
        "opp_throws_l", "opp_k_rate", "opp_hr_allowed_rate",
    ]

    # Fill missing pitcher features with league median (don't drop rows)
    pitcher_cols = [c for c in feature_cols if c.startswith("opp_")]
    for col in pitcher_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Targets
    df["target_hr1"] = (df["hr"] >= 1).astype(int)
    df["target_hits1"] = (df["hits"] >= 1).astype(int)
    df["target_hits2"] = (df["hits"] >= 2).astype(int)
    df["target_tb2"] = (df["tb"] >= 2).astype(int)

    targets = ["target_hr1", "target_hits1", "target_hits2", "target_tb2"]
    target_names = ["HR ≥ 1", "Hits ≥ 1", "Hits ≥ 2", "TB ≥ 2"]

    # Drop rows with NaN features
    valid = df[feature_cols + targets + ["season", "game_date", "player_id",
                                          "player_name", "game_pk"]].dropna()
    print(f"  Valid rows after NaN drop: {len(valid)} ({len(valid)/len(df)*100:.1f}%)")

    return valid, feature_cols, targets, target_names


def train_and_evaluate(df, feature_cols, targets, target_names):
    """Train on 2022-2025, test on 2026. Compare to naive baseline."""

    train = df[df["season"].isin(["2022", "2023", "2024", "2025"])].copy()
    test = df[df["season"] == "2026"].copy()

    print(f"\n{'='*60}")
    print(f"Train: {len(train)} rows (2022-2025)")
    print(f"Test:  {len(test)} rows (2026)")
    print(f"{'='*60}")

    if len(test) == 0:
        print("ERROR: No 2026 test data!")
        return

    X_train = train[feature_cols].values
    X_test = test[feature_cols].values

    # Standardize for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for target, name in zip(targets, target_names):
        y_train = train[target].values
        y_test = test[target].values
        base_rate = y_test.mean()

        print(f"\n--- {name} (base rate: {base_rate*100:.1f}%) ---")

        # Naive baseline: batter's season-to-date rate for this specific target
        if target == "target_hr1":
            naive_pred = test["szn_hr_rate"].values
        elif target == "target_hits1":
            naive_pred = test["szn_avg"].values * test["recent_ab"].values  # approx P(≥1 hit)
            naive_pred = np.clip(naive_pred, 0.01, 0.99)
        elif target == "target_hits2":
            # Poisson approx: P(≥2) = 1 - P(0) - P(1)
            lam = test["szn_avg"].values * test["recent_ab"].values.clip(1)
            naive_pred = 1 - np.exp(-lam) - lam * np.exp(-lam)
            naive_pred = np.clip(naive_pred, 0.01, 0.99)
        else:  # target_tb2
            naive_pred = test["szn_slg"].values * test["recent_ab"].values
            naive_pred = np.clip(naive_pred, 0.01, 0.99)

        # Better naive: use the batter's personal rate from season-to-date stats
        # P(HR ≥ 1) ≈ 1 - (1 - hr_rate)^AB ≈ hr_rate * expected_AB
        # More precisely, use Poisson: P(X≥1) = 1 - exp(-lambda)
        if target == "target_hr1":
            expected_ab = test["recent_ab"].fillna(4).values
            lam = test["szn_hr_rate"].values * expected_ab
            naive_pred = 1 - np.exp(-lam)
            naive_pred = np.clip(naive_pred, 0.001, 0.999)
        elif target == "target_hits1":
            expected_ab = test["recent_ab"].fillna(4).values
            lam = test["szn_avg"].values * expected_ab
            naive_pred = 1 - np.exp(-lam)
            naive_pred = np.clip(naive_pred, 0.01, 0.99)
        elif target == "target_hits2":
            expected_ab = test["recent_ab"].fillna(4).values
            lam = test["szn_avg"].values * expected_ab
            naive_pred = 1 - np.exp(-lam) - lam * np.exp(-lam)
            naive_pred = np.clip(naive_pred, 0.01, 0.99)
        elif target == "target_tb2":
            expected_ab = test["recent_ab"].fillna(4).values
            # TB per AB is SLG; expected TB = SLG * AB
            lam = test["szn_slg"].values * expected_ab
            naive_pred = 1 - np.exp(-lam) - lam * np.exp(-lam)
            naive_pred = np.clip(naive_pred, 0.01, 0.99)

        naive_auc = roc_auc_score(y_test, naive_pred)
        naive_brier = brier_score_loss(y_test, naive_pred)
        naive_ll = log_loss(y_test, naive_pred)

        print(f"  Naive baseline (batter season rate):")
        print(f"    AUC={naive_auc:.4f}  Brier={naive_brier:.4f}  LogLoss={naive_ll:.4f}")

        # GBC model
        gbc = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=50, random_state=42
        )
        gbc.fit(X_train, y_train)
        gbc_pred = gbc.predict_proba(X_test)[:, 1]

        gbc_auc = roc_auc_score(y_test, gbc_pred)
        gbc_brier = brier_score_loss(y_test, gbc_pred)
        gbc_ll = log_loss(y_test, gbc_pred)

        print(f"  GBC model:")
        print(f"    AUC={gbc_auc:.4f}  Brier={gbc_brier:.4f}  LogLoss={gbc_ll:.4f}")
        print(f"    Δ AUC vs naive: {(gbc_auc - naive_auc):+.4f}")
        print(f"    Δ Brier vs naive: {(gbc_brier - naive_brier):+.4f} (negative = better)")

        # Logistic regression (baseline)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict_proba(X_test_scaled)[:, 1]

        lr_auc = roc_auc_score(y_test, lr_pred)
        lr_brier = brier_score_loss(y_test, lr_pred)

        print(f"  Logistic Regression:")
        print(f"    AUC={lr_auc:.4f}  Brier={lr_brier:.4f}")

        # Calibration check: mean predicted vs actual
        print(f"  Calibration:")
        print(f"    GBC mean pred={gbc_pred.mean()*100:.1f}% vs actual={base_rate*100:.1f}%")
        print(f"    Naive mean pred={naive_pred.mean()*100:.1f}% vs actual={base_rate*100:.1f}%")

        # Calibration by decile
        decile_edges = np.percentile(gbc_pred, np.arange(0, 101, 10))
        print(f"  Calibration by decile:")
        for i in range(10):
            lo, hi = decile_edges[i], decile_edges[i+1]
            mask = (gbc_pred >= lo) & (gbc_pred < hi + 0.0001)
            if mask.sum() == 0:
                continue
            pred_mean = gbc_pred[mask].mean()
            actual_mean = y_test[mask].mean()
            n = mask.sum()
            print(f"    D{i}: pred={pred_mean*100:.1f}% actual={actual_mean*100:.1f}% N={n}")

        # Split-half validation
        np.random.seed(42)
        idx = np.random.permutation(len(y_test))
        h = len(idx) // 2
        h1_idx, h2_idx = idx[:h], idx[h:]

        h1_auc = roc_auc_score(y_test[h1_idx], gbc_pred[h1_idx])
        h2_auc = roc_auc_score(y_test[h2_idx], gbc_pred[h2_idx])
        print(f"  Split-half AUC: {h1_auc:.4f} / {h2_auc:.4f}")

        # Feature importance (top 5)
        importances = gbc.feature_importances_
        top5 = np.argsort(importances)[-5:][::-1]
        print(f"  Top 5 features:")
        for i in top5:
            print(f"    {feature_cols[i]}: {importances[i]:.3f}")

        results.append({
            "target": name,
            "base_rate": base_rate,
            "naive_auc": naive_auc,
            "naive_brier": naive_brier,
            "gbc_auc": gbc_auc,
            "gbc_brier": gbc_brier,
            "lr_auc": lr_auc,
            "delta_auc": gbc_auc - naive_auc,
            "delta_brier": gbc_brier - naive_brier,
        })

    # Summary table
    print(f"\n{'='*60}")
    print(f"=== SUMMARY ===")
    print(f"{'='*60}")
    print(f"{'Target':<10s} {'Base%':>6s} {'Naive':>6s} {'GBC':>6s} {'LR':>6s} {'ΔAUC':>7s} {'ΔBrier':>7s}")
    for r in results:
        print(f"{r['target']:<10s} {r['base_rate']*100:5.1f}% "
              f"{r['naive_auc']:.4f} {r['gbc_auc']:.4f} {r['lr_auc']:.4f} "
              f"{r['delta_auc']:+.4f} {r['delta_brier']:+.4f}")

    print(f"\nΔAUC > 0 and ΔBrier < 0 = model beats naive baseline.")
    print(f"AUC > 0.55 = meaningful discrimination.")
    print(f"ΔBrier < -0.001 = meaningful calibration improvement.")

    return results


def run():
    batter_df, lineup_df, pitcher_logs, pitcher_stats = load_data()
    batter_df = compute_rolling_features(batter_df)
    batter_df = add_matchup_features(batter_df, lineup_df, pitcher_logs, pitcher_stats)
    valid, feature_cols, targets, target_names = build_features_and_targets(batter_df)
    results = train_and_evaluate(valid, feature_cols, targets, target_names)
    return results


if __name__ == "__main__":
    run()
