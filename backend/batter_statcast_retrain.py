"""
Batter prop model retrain with Statcast features.
Pulls season-level Statcast from Baseball Savant (2021-2025),
uses prior-season stats as features, retrains HR/hits/TB models.
Tests whether Statcast closes the accuracy gap vs market (0.6175 vs 0.6385 for HR).
"""

import os, sqlite3, time, requests, io
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")


def fetch_savant_statcast(year):
    """Fetch season-level Statcast batting leaderboard from Baseball Savant."""
    url = f"https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&year={year}&position=&team=&min=25&csv=true"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return None
    df = pd.read_csv(io.StringIO(r.text))
    df["year"] = year
    return df


def fetch_savant_expected(year):
    """Fetch season-level expected stats (xwOBA, xBA, xSLG) from Baseball Savant."""
    url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={year}&position=&team=&min=25&csv=true"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return None
    df = pd.read_csv(io.StringIO(r.text))
    df["year"] = year
    return df


def pull_all_statcast():
    """Pull Statcast data for 2021-2025 (prior seasons for 2022-2026 games)."""
    print("Pulling Statcast data from Baseball Savant...")
    all_sc = []
    all_exp = []
    for year in range(2021, 2026):
        sc = fetch_savant_statcast(year)
        exp = fetch_savant_expected(year)
        if sc is not None:
            print(f"  {year} statcast: {len(sc)} batters")
            all_sc.append(sc)
        if exp is not None:
            print(f"  {year} expected: {len(exp)} batters")
            all_exp.append(exp)
        time.sleep(1)

    sc_df = pd.concat(all_sc, ignore_index=True)
    exp_df = pd.concat(all_exp, ignore_index=True)

    # Merge statcast + expected on player_id + year
    merged = sc_df.merge(exp_df, on=["player_id", "year"], how="outer", suffixes=("", "_exp"))

    # Clean up name column
    if "last_name, first_name" in merged.columns:
        merged["savant_name"] = merged["last_name, first_name"]
    if "last_name, first_name_exp" in merged.columns:
        merged.drop(columns=["last_name, first_name_exp"], inplace=True, errors="ignore")

    # Select features we want
    feature_cols = {
        "avg_hit_speed": "sc_avg_ev",
        "max_hit_speed": "sc_max_ev",
        "ev95percent": "sc_hard_hit_pct",
        "brl_percent": "sc_barrel_pct",
        "brl_pa": "sc_barrel_pa",
        "avg_hit_angle": "sc_avg_la",
        "anglesweetspotpercent": "sc_sweet_spot_pct",
        "est_woba": "sc_xwoba",
        "est_ba": "sc_xba",
        "est_slg": "sc_xslg",
    }

    result = merged[["player_id", "year"]].copy()
    if "savant_name" in merged.columns:
        result["savant_name"] = merged["savant_name"]
    for old_col, new_col in feature_cols.items():
        if old_col in merged.columns:
            result[new_col] = pd.to_numeric(merged[old_col], errors="coerce")

    return result


def build_player_id_map(db):
    """Map player_name -> MLBAM player_id using batter_game_stats."""
    rows = db.execute("""
        SELECT DISTINCT player_name, player_id
        FROM batter_game_stats
        WHERE player_id IS NOT NULL
    """).fetchall()
    return {r[0]: r[1] for r in rows}


def compute_rolling_features(batter_df, games_df):
    """Compute point-in-time rolling batter features (same as Phase 1 model)."""
    merged = batter_df.merge(games_df[["game_pk", "game_date"]], on="game_pk", how="inner")
    merged.sort_values(["player_id", "game_date", "game_pk"], inplace=True)

    # Bayesian priors
    prior_ab = 50
    prior_avg = 0.248
    prior_slg = 0.395
    prior_iso = 0.147
    prior_hr_rate = 0.033

    rows = []
    for pid, grp in merged.groupby("player_id"):
        grp = grp.sort_values("game_date")
        cum_ab, cum_hits, cum_hr, cum_doubles, cum_triples = 0, 0, 0, 0, 0
        cum_bb, cum_k, cum_hbp, cum_sf, cum_tb = 0, 0, 0, 0, 0
        cum_games = 0
        prev_season = None
        recent_window = []

        for _, row in grp.iterrows():
            season = row["game_date"][:4]
            if prev_season is not None and season != prev_season:
                cum_ab, cum_hits, cum_hr, cum_doubles, cum_triples = 0, 0, 0, 0, 0
                cum_bb, cum_k, cum_hbp, cum_sf, cum_tb = 0, 0, 0, 0, 0
                cum_games = 0
                recent_window = []
            prev_season = season

            if cum_ab >= 30:
                total_ab = cum_ab + prior_ab
                szn_avg = (cum_hits + prior_ab * prior_avg) / total_ab
                szn_slg = (cum_tb + prior_ab * prior_slg) / total_ab
                szn_iso = szn_slg - szn_avg
                szn_hr_rate = (cum_hr + prior_ab * prior_hr_rate) / total_ab
                pa = cum_ab + cum_bb + cum_hbp + cum_sf
                szn_k_rate = cum_k / pa if pa > 0 else 0
                szn_bb_rate = cum_bb / pa if pa > 0 else 0

                recent_ab = sum(r["ab"] for r in recent_window)
                recent_hr = sum(r["hr"] for r in recent_window)
                recent_hits = sum(r["hits"] for r in recent_window)
                recent_tb = sum(r["tb"] for r in recent_window)
                recent_k = sum(r["k"] for r in recent_window)
                recent_avg = recent_hits / recent_ab if recent_ab > 0 else szn_avg
                recent_hr_r = recent_hr / recent_ab if recent_ab > 0 else szn_hr_rate
                recent_tb_r = recent_tb / recent_ab if recent_ab > 0 else szn_slg
                recent_k_r = recent_k / recent_ab if recent_ab > 0 else szn_k_rate

                rows.append({
                    "player_id": pid,
                    "player_name": row["player_name"],
                    "game_pk": row["game_pk"],
                    "game_date": row["game_date"],
                    "team_name": row["team_name"],
                    "is_home": row["is_home"],
                    "ab": row["ab"],
                    "hits": row["hits"],
                    "hr": row["hr"],
                    "tb": row["tb"],
                    "szn_avg": szn_avg,
                    "szn_slg": szn_slg,
                    "szn_iso": szn_iso,
                    "szn_hr_rate": szn_hr_rate,
                    "szn_k_rate": szn_k_rate,
                    "szn_bb_rate": szn_bb_rate,
                    "szn_ab": cum_ab,
                    "szn_games": cum_games,
                    "recent_avg": recent_avg,
                    "recent_hr": recent_hr_r,
                    "recent_tb": recent_tb_r,
                    "recent_k": recent_k_r,
                    "recent_ab": recent_ab,
                })

            cum_ab += row["ab"]
            cum_hits += row["hits"]
            cum_hr += row["hr"]
            cum_doubles += row["doubles"]
            cum_triples += row["triples"]
            cum_bb += row["bb"]
            cum_k += row["k"]
            cum_hbp += row["hbp"]
            cum_sf += row["sf"]
            cum_tb += row["tb"]
            cum_games += 1
            recent_window.append(row.to_dict())
            if len(recent_window) > 14:
                recent_window.pop(0)

    return pd.DataFrame(rows)


def add_matchup_features(feat_df, db):
    """Add opposing pitcher features via frequency-based team identification."""
    print("Adding matchup features...")

    # Load pitcher game logs
    pitcher_logs = pd.read_sql_query("""
        SELECT player_id as pitcher_id, game_pk, ip, er, hits as hits_allowed,
               walks as bb_allowed, strikeouts as k, home_runs as hr_allowed
        FROM pitcher_game_logs
    """, db)

    # Load pitcher season stats
    pitcher_season = pd.read_sql_query("""
        SELECT player_id as pitcher_id, season, era, whip, k9 as k_per_9, hr9 as hr_per_9, fip
        FROM pitcher_stats
    """, db)

    # Load games for team info
    games = pd.read_sql_query("""
        SELECT game_id, home_team, away_team
        FROM games WHERE sport = 'MLB'
    """, db)
    games["game_pk"] = games["game_id"].str.replace("MLB_", "").astype(int)

    # Identify pitcher teams via frequency
    pitcher_games = pitcher_logs.merge(games[["game_pk", "home_team", "away_team"]], on="game_pk")
    team_counts = {}
    for pid, grp in pitcher_games.groupby("pitcher_id"):
        counts = {}
        for _, r in grp.iterrows():
            for t in [r["home_team"], r["away_team"]]:
                counts[t] = counts.get(t, 0) + 1
        team_counts[pid] = max(counts, key=counts.get) if counts else None

    # Find opposing pitcher per game
    game_pitchers = pitcher_logs.groupby("game_pk").apply(
        lambda g: g.nlargest(2, "ip")[["pitcher_id", "ip"]].to_dict("records")
    ).to_dict()

    opp_features = []
    for _, row in feat_df.iterrows():
        gp = row["game_pk"]
        team = row["team_name"]
        season = row["game_date"][:4]

        pitchers = game_pitchers.get(gp, [])
        opp_pid = None
        for p in pitchers:
            pid = p["pitcher_id"]
            pteam = team_counts.get(pid)
            if pteam and pteam != team:
                opp_pid = pid
                break

        if opp_pid:
            ps = pitcher_season[(pitcher_season["pitcher_id"] == opp_pid) &
                                (pitcher_season["season"] == season)]
            if len(ps) > 0:
                ps = ps.iloc[0]
                opp_features.append({
                    "opp_era": ps["era"],
                    "opp_whip": ps["whip"],
                    "opp_k9": ps["k_per_9"],
                    "opp_hr9": ps["hr_per_9"],
                    "opp_fip": ps["fip"],
                })
                continue
        opp_features.append({})

    opp_df = pd.DataFrame(opp_features, index=feat_df.index)
    for col in ["opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip"]:
        if col in opp_df.columns:
            feat_df[col] = opp_df[col]
        else:
            feat_df[col] = np.nan

    # Fill missing with median
    for col in ["opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip"]:
        feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    return feat_df


def add_lineup_features(feat_df, db):
    """Add batting order from game_lineups."""
    lineups = pd.read_sql_query("""
        SELECT game_pk, player_id, batting_order
        FROM game_lineups
    """, db)
    lineups["batting_order_pos"] = (lineups["batting_order"] / 100).astype(int)
    feat_df = feat_df.merge(
        lineups[["game_pk", "player_id", "batting_order_pos"]],
        on=["game_pk", "player_id"], how="left"
    )
    feat_df.rename(columns={"batting_order_pos": "batting_order"}, inplace=True)
    feat_df["batting_order"] = feat_df["batting_order"].fillna(feat_df["batting_order"].median())
    return feat_df


def run():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # ── Step 1: Pull Statcast data ──
    statcast = pull_all_statcast()
    print(f"\nTotal Statcast rows: {len(statcast)}")

    # ── Step 2: Build player_id map ──
    pid_map = build_player_id_map(db)
    print(f"Player name->ID map: {len(pid_map)} players")

    # Shift Statcast by 1 year (use season N-1 for season N games = PIT safe)
    statcast["use_for_season"] = (statcast["year"] + 1).astype(str)

    # ── Step 3: Load batter game stats ──
    print("\nLoading batter game stats...")
    batter_df = pd.read_sql_query("""
        SELECT game_pk, player_id, player_name, team_name, is_home,
               ab, hits, doubles, triples, hr, bb, hbp, sf, k, tb
        FROM batter_game_stats
    """, db)
    print(f"  Loaded {len(batter_df)} rows ({batter_df['player_id'].nunique()} batters)")

    # Load games for dates
    games_df = pd.read_sql_query("""
        SELECT game_id, game_date
        FROM games WHERE sport = 'MLB'
    """, db)
    games_df["game_pk"] = games_df["game_id"].str.replace("MLB_", "").astype(int)

    # ── Step 4: Compute rolling features ──
    print("Computing rolling features (point-in-time)...")
    feat_df = compute_rolling_features(batter_df, games_df)
    print(f"  After minimum AB filter: {len(feat_df)} rows")

    # ── Step 5: Add matchup + lineup features ──
    feat_df = add_matchup_features(feat_df, db)
    feat_df = add_lineup_features(feat_df, db)

    # ── Step 6: Join Statcast features ──
    print("\nJoining Statcast features...")
    feat_df["season"] = feat_df["game_date"].str[:4]

    # Map player_id to Statcast player_id (same MLBAM ID system)
    sc_features = ["sc_avg_ev", "sc_max_ev", "sc_hard_hit_pct", "sc_barrel_pct",
                    "sc_barrel_pa", "sc_avg_la", "sc_sweet_spot_pct",
                    "sc_xwoba", "sc_xba", "sc_xslg"]

    # Join on player_id + season
    statcast_join = statcast[["player_id", "use_for_season"] + [c for c in sc_features if c in statcast.columns]].copy()
    statcast_join.rename(columns={"use_for_season": "season"}, inplace=True)

    feat_df = feat_df.merge(statcast_join, on=["player_id", "season"], how="left")

    sc_available = [c for c in sc_features if c in feat_df.columns]
    sc_fill_rate = feat_df[sc_available].notna().mean()
    print(f"  Statcast feature fill rates:")
    for col in sc_available:
        rate = feat_df[col].notna().mean()
        print(f"    {col}: {rate*100:.1f}%")

    # Fill missing Statcast with median
    for col in sc_available:
        feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    # ── Step 7: Train/test split ──
    feat_df["year"] = feat_df["game_date"].str[:4].astype(int)
    train = feat_df[feat_df["year"] <= 2025].copy()
    test = feat_df[feat_df["year"] >= 2026].copy()

    print(f"\n{'='*60}")
    print(f"Train: {len(train)} rows (2022-2025)")
    print(f"Test:  {len(test)} rows (2026)")
    print(f"{'='*60}")

    # ── Feature sets ──
    base_features = [
        "szn_avg", "szn_slg", "szn_iso", "szn_hr_rate", "szn_k_rate", "szn_bb_rate",
        "szn_ab", "szn_games", "recent_avg", "recent_hr", "recent_tb", "recent_k",
        "recent_ab", "batting_order", "is_home",
        "opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip",
    ]

    statcast_features = [c for c in sc_available if c in feat_df.columns]
    all_features = base_features + statcast_features

    # Drop any remaining NaN
    train_clean = train.dropna(subset=all_features)
    test_clean = test.dropna(subset=all_features)
    print(f"After NaN drop: train={len(train_clean)}, test={len(test_clean)}")

    # ── Step 8: Train and evaluate ──
    targets = [
        ("HR >= 1", "hr", 1),
        ("Hits >= 1", "hits", 1),
        ("Hits >= 2", "hits", 2),
        ("TB >= 2", "tb", 2),
    ]

    results = []
    for target_name, stat_col, threshold in targets:
        y_train = (train_clean[stat_col] >= threshold).astype(int).values
        y_test = (test_clean[stat_col] >= threshold).astype(int).values
        base_rate = y_test.mean()

        print(f"\n--- {target_name} (base rate: {base_rate*100:.1f}%) ---")

        # Model A: base features only (Phase 1 replication)
        X_train_base = train_clean[base_features].values
        X_test_base = test_clean[base_features].values

        gbc_base = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=50, random_state=42
        )
        gbc_base.fit(X_train_base, y_train)
        pred_base = gbc_base.predict_proba(X_test_base)[:, 1]
        auc_base = roc_auc_score(y_test, pred_base)
        brier_base = brier_score_loss(y_test, pred_base)

        print(f"  Base model (no Statcast): AUC={auc_base:.4f}  Brier={brier_base:.4f}")

        # Model B: base + Statcast features
        X_train_all = train_clean[all_features].values
        X_test_all = test_clean[all_features].values

        gbc_all = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=50, random_state=42
        )
        gbc_all.fit(X_train_all, y_train)
        pred_all = gbc_all.predict_proba(X_test_all)[:, 1]
        auc_all = roc_auc_score(y_test, pred_all)
        brier_all = brier_score_loss(y_test, pred_all)

        delta_auc = auc_all - auc_base
        delta_brier = brier_all - brier_base

        print(f"  + Statcast model:         AUC={auc_all:.4f}  Brier={brier_all:.4f}")
        print(f"  Δ AUC: {delta_auc:+.4f}  Δ Brier: {delta_brier:+.4f}")

        # Feature importance for Statcast features
        importances = dict(zip(all_features, gbc_all.feature_importances_))
        sc_imp = {k: v for k, v in importances.items() if k.startswith("sc_")}
        sc_imp_sorted = sorted(sc_imp.items(), key=lambda x: -x[1])
        total_sc_imp = sum(sc_imp.values())
        print(f"  Statcast total importance: {total_sc_imp:.3f}")
        print(f"  Top Statcast features:")
        for fname, imp in sc_imp_sorted[:5]:
            print(f"    {fname}: {imp:.4f}")

        # Top 5 overall features
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
        print(f"  Top 5 overall features:")
        for fname, imp in top5:
            print(f"    {fname}: {imp:.3f}")

        # Calibration by decile
        print(f"  Calibration (+ Statcast model):")
        decile_edges = np.percentile(pred_all, np.arange(0, 101, 10))
        for i in range(10):
            lo, hi = decile_edges[i], decile_edges[i + 1]
            mask = (pred_all >= lo) & (pred_all < hi + 0.0001)
            if mask.sum() == 0:
                continue
            pred_mean = pred_all[mask].mean()
            actual_mean = y_test[mask].mean()
            n = mask.sum()
            print(f"    D{i}: pred={pred_mean*100:.1f}% actual={actual_mean*100:.1f}% N={n}")

        # Split-half stability
        half = len(test_clean) // 2
        auc_a = roc_auc_score(y_test[:half], pred_all[:half])
        auc_b = roc_auc_score(y_test[half:], pred_all[half:])
        print(f"  Split-half AUC: {auc_a:.4f} / {auc_b:.4f}")

        results.append({
            "target": target_name,
            "base_rate": base_rate,
            "auc_base": auc_base,
            "auc_statcast": auc_all,
            "delta_auc": delta_auc,
            "brier_base": brier_base,
            "brier_statcast": brier_all,
            "delta_brier": delta_brier,
            "sc_importance": total_sc_imp,
            "n": len(test_clean),
        })

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"=== STATCAST RETRAIN SUMMARY ===")
    print(f"{'='*70}")
    print(f"{'Target':<10s} {'Base%':>6s} {'Base AUC':>9s} {'+SC AUC':>9s} {'ΔAUC':>8s} {'ΔBrier':>8s} {'SC Imp':>7s}")
    for r in results:
        print(f"{r['target']:<10s} {r['base_rate']*100:5.1f}% "
              f"  {r['auc_base']:.4f}   {r['auc_statcast']:.4f}  "
              f"{r['delta_auc']:+.4f}  {r['delta_brier']:+.4f}  "
              f"{r['sc_importance']:.3f}")

    print(f"\nMarket reference (8-day overlap):")
    print(f"  HR >= 1: Market AUC = 0.6385")
    print(f"\nVerdict: if +Statcast AUC approaches 0.639, gap is closing.")
    print(f"         if ΔAUC < +0.005, Statcast adds nothing at season level.")

    db.close()


if __name__ == "__main__":
    run()
