"""
DECISIVE CLV TEST: Does the Statcast-enhanced batter prop model beat
the de-vigged prop close net of vig?

THE BAR: AUC parity (0.641 vs 0.639) is a tie. A tie LOSES to ~6-7% vig.
To be +EV, model must beat de-vigged close systematically by MORE than vig.

Tests HR and TB on the full Jun 10 - Aug 13 window, with segment analysis
and split-half validation.
"""

import os, sqlite3, time, requests, io
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")


# ─────────────────────────────────────────────────────────────
# STATCAST DATA (pulled from Baseball Savant)
# ─────────────────────────────────────────────────────────────
def fetch_savant_statcast(year):
    url = f"https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&year={year}&position=&team=&min=25&csv=true"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    return pd.read_csv(io.StringIO(r.text)).assign(year=year) if r.status_code == 200 else None


def fetch_savant_expected(year):
    url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={year}&position=&team=&min=25&csv=true"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    return pd.read_csv(io.StringIO(r.text)).assign(year=year) if r.status_code == 200 else None


def pull_all_statcast():
    print("Pulling Statcast from Baseball Savant...")
    all_sc, all_exp = [], []
    for year in range(2021, 2026):
        sc = fetch_savant_statcast(year)
        exp = fetch_savant_expected(year)
        if sc is not None:
            all_sc.append(sc)
            print(f"  {year}: {len(sc)} batters (statcast)")
        if exp is not None:
            all_exp.append(exp)
        time.sleep(1)

    sc_df = pd.concat(all_sc, ignore_index=True)
    exp_df = pd.concat(all_exp, ignore_index=True)
    merged = sc_df.merge(exp_df, on=["player_id", "year"], how="outer", suffixes=("", "_exp"))

    feature_map = {
        "avg_hit_speed": "sc_avg_ev", "max_hit_speed": "sc_max_ev",
        "ev95percent": "sc_hard_hit_pct", "brl_percent": "sc_barrel_pct",
        "brl_pa": "sc_barrel_pa", "avg_hit_angle": "sc_avg_la",
        "anglesweetspotpercent": "sc_sweet_spot_pct",
        "est_woba": "sc_xwoba", "est_ba": "sc_xba", "est_slg": "sc_xslg",
    }
    result = merged[["player_id", "year"]].copy()
    for old_col, new_col in feature_map.items():
        if old_col in merged.columns:
            result[new_col] = pd.to_numeric(merged[old_col], errors="coerce")
    return result


# ─────────────────────────────────────────────────────────────
# ROLLING FEATURES (point-in-time, Bayesian shrinkage)
# ─────────────────────────────────────────────────────────────
def compute_rolling_features(batter_df, games_df):
    merged = batter_df.merge(games_df[["game_pk", "game_date"]], on="game_pk", how="inner")
    merged.sort_values(["player_id", "game_date", "game_pk"], inplace=True)

    prior_ab, prior_avg, prior_slg, prior_iso, prior_hr = 50, 0.248, 0.395, 0.147, 0.033
    rows = []
    for pid, grp in merged.groupby("player_id"):
        grp = grp.sort_values("game_date")
        cum = {"ab": 0, "hits": 0, "hr": 0, "doubles": 0, "triples": 0,
               "bb": 0, "k": 0, "hbp": 0, "sf": 0, "tb": 0, "games": 0}
        prev_season = None
        recent = []

        for _, row in grp.iterrows():
            season = row["game_date"][:4]
            if prev_season and season != prev_season:
                cum = {k: 0 for k in cum}
                recent = []
            prev_season = season

            if cum["ab"] >= 30:
                tot = cum["ab"] + prior_ab
                szn_avg = (cum["hits"] + prior_ab * prior_avg) / tot
                szn_slg = (cum["tb"] + prior_ab * prior_slg) / tot
                szn_hr_rate = (cum["hr"] + prior_ab * prior_hr) / tot
                pa = cum["ab"] + cum["bb"] + cum["hbp"] + cum["sf"]
                szn_k_rate = cum["k"] / pa if pa > 0 else 0
                szn_bb_rate = cum["bb"] / pa if pa > 0 else 0

                r_ab = sum(x["ab"] for x in recent)
                rows.append({
                    "player_id": pid, "player_name": row["player_name"],
                    "game_pk": row["game_pk"], "game_date": row["game_date"],
                    "team_name": row["team_name"], "is_home": row["is_home"],
                    "ab": row["ab"], "hits": row["hits"], "hr": row["hr"], "tb": row["tb"],
                    "szn_avg": szn_avg, "szn_slg": szn_slg,
                    "szn_iso": szn_slg - szn_avg, "szn_hr_rate": szn_hr_rate,
                    "szn_k_rate": szn_k_rate, "szn_bb_rate": szn_bb_rate,
                    "szn_ab": cum["ab"], "szn_games": cum["games"],
                    "recent_avg": sum(x["hits"] for x in recent) / r_ab if r_ab > 0 else szn_avg,
                    "recent_hr": sum(x["hr"] for x in recent) / r_ab if r_ab > 0 else szn_hr_rate,
                    "recent_tb": sum(x["tb"] for x in recent) / r_ab if r_ab > 0 else szn_slg,
                    "recent_k": sum(x["k"] for x in recent) / r_ab if r_ab > 0 else szn_k_rate,
                    "recent_ab": r_ab,
                })

            for k in ["ab", "hits", "hr", "doubles", "triples", "bb", "k", "hbp", "sf", "tb"]:
                cum[k] += row[k]
            cum["games"] += 1
            recent.append(row.to_dict())
            if len(recent) > 14:
                recent.pop(0)

    return pd.DataFrame(rows)


def add_matchup_features(feat_df, db):
    pitcher_logs = pd.read_sql_query(
        "SELECT player_id as pitcher_id, game_pk, ip FROM pitcher_game_logs", db)
    pitcher_season = pd.read_sql_query(
        "SELECT player_id as pitcher_id, season, era, whip, k9 as k_per_9, hr9 as hr_per_9, fip FROM pitcher_stats", db)
    games = pd.read_sql_query("SELECT game_id, home_team, away_team FROM games WHERE sport='MLB'", db)
    games["game_pk"] = games["game_id"].str.replace("MLB_", "").astype(int)

    pitcher_games = pitcher_logs.merge(games[["game_pk", "home_team", "away_team"]], on="game_pk")
    team_map = {}
    for pid, grp in pitcher_games.groupby("pitcher_id"):
        counts = {}
        for _, r in grp.iterrows():
            for t in [r["home_team"], r["away_team"]]:
                counts[t] = counts.get(t, 0) + 1
        team_map[pid] = max(counts, key=counts.get) if counts else None

    game_pitchers = pitcher_logs.groupby("game_pk").apply(
        lambda g: g.nlargest(2, "ip")[["pitcher_id"]].values.flatten().tolist()
    ).to_dict()

    opp_rows = []
    for _, row in feat_df.iterrows():
        pitchers = game_pitchers.get(row["game_pk"], [])
        season = row["game_date"][:4]
        found = False
        for pid in pitchers:
            if team_map.get(pid) and team_map[pid] != row["team_name"]:
                ps = pitcher_season[(pitcher_season["pitcher_id"] == pid) &
                                     (pitcher_season["season"] == season)]
                if len(ps) > 0:
                    opp_rows.append(ps.iloc[0][["era", "whip", "k_per_9", "hr_per_9", "fip"]].to_dict())
                    found = True
                    break
        if not found:
            opp_rows.append({})

    opp_df = pd.DataFrame(opp_rows, index=feat_df.index)
    for col, new in [("era", "opp_era"), ("whip", "opp_whip"), ("k_per_9", "opp_k9"),
                      ("hr_per_9", "opp_hr9"), ("fip", "opp_fip")]:
        feat_df[new] = opp_df[col] if col in opp_df.columns else np.nan
        feat_df[new] = feat_df[new].fillna(feat_df[new].median())
    return feat_df


def add_lineup_features(feat_df, db):
    lineups = pd.read_sql_query("SELECT game_pk, player_id, batting_order FROM game_lineups", db)
    lineups["bo"] = (lineups["batting_order"] / 100).astype(int)
    feat_df = feat_df.merge(lineups[["game_pk", "player_id", "bo"]], on=["game_pk", "player_id"], how="left")
    feat_df.rename(columns={"bo": "batting_order"}, inplace=True)
    feat_df["batting_order"] = feat_df["batting_order"].fillna(feat_df["batting_order"].median())
    return feat_df


def devig(over_price, under_price):
    if over_price <= 1.0 or under_price <= 1.0:
        return None
    imp_o = 1.0 / over_price
    imp_u = 1.0 / under_price
    tot = imp_o + imp_u
    return imp_o / tot if tot > 0 else None


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # ── 1. Pull Statcast ──
    statcast = pull_all_statcast()
    statcast["use_season"] = (statcast["year"] + 1).astype(str)
    sc_features = [c for c in statcast.columns if c.startswith("sc_")]

    # ── 2. Build feature matrix ──
    print("\nLoading batter game stats...")
    batter_df = pd.read_sql_query(
        "SELECT game_pk, player_id, player_name, team_name, is_home, "
        "ab, hits, doubles, triples, hr, bb, hbp, sf, k, tb FROM batter_game_stats", db)
    games_df = pd.read_sql_query("SELECT game_id, game_date FROM games WHERE sport='MLB'", db)
    games_df["game_pk"] = games_df["game_id"].str.replace("MLB_", "").astype(int)

    print("Computing rolling features...")
    feat_df = compute_rolling_features(batter_df, games_df)
    print(f"  {len(feat_df)} rows")

    feat_df = add_matchup_features(feat_df, db)
    feat_df = add_lineup_features(feat_df, db)

    # Join Statcast (prior season)
    feat_df["season"] = feat_df["game_date"].str[:4]
    sc_join = statcast[["player_id", "use_season"] + sc_features].rename(columns={"use_season": "season"})
    feat_df = feat_df.merge(sc_join, on=["player_id", "season"], how="left")
    for c in sc_features:
        if c in feat_df.columns:
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    # ── 3. Train model (2022-2025), predict 2026 ──
    base_features = [
        "szn_avg", "szn_slg", "szn_iso", "szn_hr_rate", "szn_k_rate", "szn_bb_rate",
        "szn_ab", "szn_games", "recent_avg", "recent_hr", "recent_tb", "recent_k",
        "recent_ab", "batting_order", "is_home",
        "opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip",
    ]
    all_features = base_features + [c for c in sc_features if c in feat_df.columns]

    feat_df["year"] = feat_df["game_date"].str[:4].astype(int)
    train = feat_df[feat_df["year"] <= 2025].dropna(subset=all_features)
    test = feat_df[feat_df["year"] >= 2026].dropna(subset=all_features)

    print(f"\nTrain: {len(train)}, Test: {len(test)}")

    # Train HR model
    y_train_hr = (train["hr"] >= 1).astype(int).values
    gbc_hr = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=42)
    gbc_hr.fit(train[all_features].values, y_train_hr)
    test["model_hr_prob"] = gbc_hr.predict_proba(test[all_features].values)[:, 1]

    # Train TB model
    y_train_tb = (train["tb"] >= 2).astype(int).values
    gbc_tb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=42)
    gbc_tb.fit(train[all_features].values, y_train_tb)
    test["model_tb_prob"] = gbc_tb.predict_proba(test[all_features].values)[:, 1]

    hr_auc = roc_auc_score((test["hr"] >= 1).astype(int), test["model_hr_prob"])
    tb_auc = roc_auc_score((test["tb"] >= 2).astype(int), test["model_tb_prob"])
    print(f"Model AUC — HR: {hr_auc:.4f}, TB: {tb_auc:.4f}")

    # ── 4. Load prop odds and de-vig ──
    print("\nLoading prop odds...")
    props = pd.read_sql_query("""
        SELECT player_name, game_date, bookmaker, over_under, price, point, market_key
        FROM player_props
        WHERE sport_key = 'baseball_mlb'
          AND game_date >= '2026-06-10' AND game_date <= '2026-08-13'
          AND market_key IN ('batter_home_runs', 'batter_total_bases')
    """, db)

    # For each market, devig and get (player, date) level market prob
    for market_key, point_val, label, model_col, actual_col, threshold in [
        ("batter_home_runs", 0.5, "HR >= 1", "model_hr_prob", "hr", 1),
        ("batter_total_bases", 1.5, "TB >= 2", "model_tb_prob", "tb", 2),
    ]:
        print(f"\n{'='*70}")
        print(f"=== CLV TEST: {label} ===")
        print(f"{'='*70}")

        mkt = props[(props["market_key"] == market_key) & (props["point"] == point_val)]
        grouped = mkt.groupby(["player_name", "game_date", "bookmaker"])

        devigged = []
        for (player, date, bk), group in grouped:
            overs = group[group["over_under"].str.lower() == "over"]
            unders = group[group["over_under"].str.lower() == "under"]
            if len(overs) == 0 or len(unders) == 0:
                continue
            op = overs.iloc[0]["price"]
            up = unders.iloc[0]["price"]
            dv = devig(op, up)
            if dv is None:
                continue
            vig = (1/op + 1/up - 1)
            devigged.append({
                "player_name": player, "game_date": date, "bookmaker": bk,
                "over_price": op, "under_price": up,
                "market_prob": dv, "vig": vig,
            })

        dv_df = pd.DataFrame(devigged)
        if len(dv_df) == 0:
            print(f"  No devigged odds for {label}!")
            continue

        # Best price per side per (player, date) — for betting sim
        best = dv_df.groupby(["player_name", "game_date"]).agg(
            market_prob=("market_prob", "mean"),
            best_over=("over_price", "max"),
            best_under=("under_price", "max"),
            avg_vig=("vig", "mean"),
            n_books=("bookmaker", "count"),
        ).reset_index()

        # Merge with model predictions + outcomes
        merged = best.merge(
            test[["player_name", "game_date", model_col, actual_col, "szn_hr_rate",
                  "batting_order", "szn_ab"]].drop_duplicates(["player_name", "game_date"]),
            on=["player_name", "game_date"], how="inner"
        )
        merged["actual"] = (merged[actual_col] >= threshold).astype(int)
        merged["model_prob"] = merged[model_col]
        merged["edge"] = merged["model_prob"] - merged["market_prob"]

        n = len(merged)
        n_players = merged["player_name"].nunique()
        n_dates = merged["game_date"].nunique()
        print(f"  Matched: {n} obs, {n_players} players, {n_dates} dates")
        print(f"  Base rate: {merged['actual'].mean()*100:.1f}%")
        print(f"  Avg vig: {merged['avg_vig'].mean()*100:.1f}%")
        print(f"  Mean market prob: {merged['market_prob'].mean()*100:.2f}%")
        print(f"  Mean model prob: {merged['model_prob'].mean()*100:.2f}%")

        # ── Model vs Market AUC on matched set ──
        mkt_auc = roc_auc_score(merged["actual"], merged["market_prob"])
        mdl_auc = roc_auc_score(merged["actual"], merged["model_prob"])
        mkt_brier = brier_score_loss(merged["actual"], merged["market_prob"])
        mdl_brier = brier_score_loss(merged["actual"], merged["model_prob"])
        print(f"\n  --- ACCURACY ON MATCHED SET ---")
        print(f"  Market AUC: {mkt_auc:.4f}  Brier: {mkt_brier:.4f}")
        print(f"  Model AUC:  {mdl_auc:.4f}  Brier: {mdl_brier:.4f}")
        print(f"  Δ AUC: {mdl_auc - mkt_auc:+.4f}  Δ Brier: {mdl_brier - mkt_brier:+.4f}")

        # ── OVERALL BETTING SIMULATION ──
        print(f"\n  --- BETTING SIMULATION (flat $10) ---")

        # Bet OVER when model > market, UNDER when model < market
        over_bets = merged[merged["edge"] > 0].copy()
        under_bets = merged[merged["edge"] < 0].copy()

        # Over bets PnL
        over_wins = over_bets["actual"].sum()
        over_n = len(over_bets)
        if over_n > 0:
            over_pnl = sum(
                10 * (row["best_over"] - 1) if row["actual"] == 1 else -10
                for _, row in over_bets.iterrows()
            )
            over_wr = over_bets["actual"].mean()
            over_be = 1.0 / over_bets["best_over"].mean()
            print(f"  OVER bets (model > market): N={over_n}")
            print(f"    Win rate: {over_wr*100:.1f}%  Breakeven: {over_be*100:.1f}%  "
                  f"PnL: ${over_pnl:+.0f}  ROI: {over_pnl/(over_n*10)*100:+.1f}%")

        # Under bets PnL
        under_wins = (1 - under_bets["actual"]).sum()
        under_n = len(under_bets)
        if under_n > 0:
            under_pnl = sum(
                10 * (row["best_under"] - 1) if row["actual"] == 0 else -10
                for _, row in under_bets.iterrows()
            )
            under_wr = (1 - under_bets["actual"]).mean()
            under_be = 1.0 / under_bets["best_under"].mean()
            print(f"  UNDER bets (model < market): N={under_n}")
            print(f"    Win rate: {under_wr*100:.1f}%  Breakeven: {under_be*100:.1f}%  "
                  f"PnL: ${under_pnl:+.0f}  ROI: {under_pnl/(under_n*10)*100:+.1f}%")

        # Combined
        total_pnl = (over_pnl if over_n > 0 else 0) + (under_pnl if under_n > 0 else 0)
        total_n = over_n + under_n
        print(f"  COMBINED: N={total_n}  PnL: ${total_pnl:+.0f}  "
              f"ROI: {total_pnl/(total_n*10)*100:+.1f}%")

        # ── CLV ANALYSIS ──
        # CLV = model's directional edge vs market
        # For Over bets: does actual rate exceed market prob? (actual > market = model right)
        # For Under bets: does actual rate fall below market prob? (actual < market = model right)
        print(f"\n  --- CLV ANALYSIS ---")
        print(f"  Model edge distribution:")
        print(f"    Mean: {merged['edge'].mean()*100:+.2f}pp")
        print(f"    Median: {merged['edge'].median()*100:+.2f}pp")
        print(f"    Std: {merged['edge'].std()*100:.2f}pp")

        # Binned edge analysis
        edge_bins = [(-1, -0.05), (-0.05, -0.02), (-0.02, -0.01), (-0.01, 0),
                     (0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1)]
        print(f"\n  Edge-binned performance:")
        print(f"  {'Edge bin':>15s} {'N':>6s} {'Actual%':>8s} {'Market%':>8s} {'Model%':>8s} {'PnL':>8s} {'ROI':>7s}")
        for lo, hi in edge_bins:
            if lo >= 0:
                sub = over_bets[(over_bets["edge"] >= lo) & (over_bets["edge"] < hi)]
                if len(sub) < 5:
                    continue
                pnl = sum(10*(r["best_over"]-1) if r["actual"]==1 else -10 for _, r in sub.iterrows())
            else:
                sub = under_bets[(under_bets["edge"] >= lo) & (under_bets["edge"] < hi)]
                if len(sub) < 5:
                    continue
                pnl = sum(10*(r["best_under"]-1) if r["actual"]==0 else -10 for _, r in sub.iterrows())
            act = sub["actual"].mean()
            mkt = sub["market_prob"].mean()
            mdl = sub["model_prob"].mean()
            roi = pnl / (len(sub) * 10) * 100
            label_str = f"[{lo*100:+.0f},{hi*100:+.0f})pp"
            print(f"  {label_str:>15s} {len(sub):>6d} {act*100:>7.1f}% {mkt*100:>7.1f}% "
                  f"{mdl*100:>7.1f}% ${pnl:>+7.0f} {roi:>+6.1f}%")

        # ── HIGH-CONVICTION SUBSET ──
        print(f"\n  --- HIGH-CONVICTION (|edge| > 3pp) ---")
        hc_over = over_bets[over_bets["edge"] > 0.03]
        hc_under = under_bets[under_bets["edge"] < -0.03]
        if len(hc_over) > 10:
            pnl_hc = sum(10*(r["best_over"]-1) if r["actual"]==1 else -10 for _, r in hc_over.iterrows())
            print(f"  OVER (edge > +3pp): N={len(hc_over)}  WR={hc_over['actual'].mean()*100:.1f}%  "
                  f"BE={1/hc_over['best_over'].mean()*100:.1f}%  "
                  f"PnL=${pnl_hc:+.0f}  ROI={pnl_hc/(len(hc_over)*10)*100:+.1f}%")
        if len(hc_under) > 10:
            pnl_hc = sum(10*(r["best_under"]-1) if r["actual"]==0 else -10 for _, r in hc_under.iterrows())
            wr_hc = (1-hc_under["actual"]).mean()
            print(f"  UNDER (edge < -3pp): N={len(hc_under)}  WR={wr_hc*100:.1f}%  "
                  f"BE={1/hc_under['best_under'].mean()*100:.1f}%  "
                  f"PnL=${pnl_hc:+.0f}  ROI={pnl_hc/(len(hc_under)*10)*100:+.1f}%")

        # ── SEGMENT ANALYSIS ──
        print(f"\n  --- SEGMENT ANALYSIS ---")

        # By batter tier (HR rate)
        if "szn_hr_rate" in merged.columns:
            print(f"\n  By batter HR rate tier:")
            for tier_name, lo_r, hi_r in [("Role (<2.5%)", 0, 0.025),
                                           ("Mid (2.5-4%)", 0.025, 0.04),
                                           ("Star (>4%)", 0.04, 1.0)]:
                sub = merged[(merged["szn_hr_rate"] >= lo_r) & (merged["szn_hr_rate"] < hi_r)]
                if len(sub) < 20:
                    continue
                over_sub = sub[sub["edge"] > 0]
                under_sub = sub[sub["edge"] < 0]
                pnl_o = sum(10*(r["best_over"]-1) if r["actual"]==1 else -10 for _, r in over_sub.iterrows()) if len(over_sub) > 0 else 0
                pnl_u = sum(10*(r["best_under"]-1) if r["actual"]==0 else -10 for _, r in under_sub.iterrows()) if len(under_sub) > 0 else 0
                total_pnl_seg = pnl_o + pnl_u
                total_n_seg = len(over_sub) + len(under_sub)
                mdl_auc_seg = roc_auc_score(sub["actual"], sub["model_prob"]) if sub["actual"].nunique() > 1 else 0
                mkt_auc_seg = roc_auc_score(sub["actual"], sub["market_prob"]) if sub["actual"].nunique() > 1 else 0
                print(f"    {tier_name}: N={total_n_seg}  Model AUC={mdl_auc_seg:.4f}  "
                      f"Mkt AUC={mkt_auc_seg:.4f}  PnL=${total_pnl_seg:+.0f}  "
                      f"ROI={total_pnl_seg/(total_n_seg*10)*100:+.1f}%")

        # By batting order
        print(f"\n  By batting order:")
        for bo_name, lo_bo, hi_bo in [("Top 1-3", 0, 4), ("Mid 4-6", 4, 7), ("Bottom 7-9", 7, 10)]:
            sub = merged[(merged["batting_order"] >= lo_bo) & (merged["batting_order"] < hi_bo)]
            if len(sub) < 20:
                continue
            over_sub = sub[sub["edge"] > 0]
            under_sub = sub[sub["edge"] < 0]
            pnl_o = sum(10*(r["best_over"]-1) if r["actual"]==1 else -10 for _, r in over_sub.iterrows()) if len(over_sub) > 0 else 0
            pnl_u = sum(10*(r["best_under"]-1) if r["actual"]==0 else -10 for _, r in under_sub.iterrows()) if len(under_sub) > 0 else 0
            total_seg = pnl_o + pnl_u
            n_seg = len(over_sub) + len(under_sub)
            print(f"    {bo_name}: N={n_seg}  PnL=${total_seg:+.0f}  "
                  f"ROI={total_seg/(n_seg*10)*100:+.1f}%")

        # By edge direction (does model add value when it disagrees strongly?)
        print(f"\n  Conviction ladder:")
        for min_edge in [0.0, 0.01, 0.02, 0.03, 0.05]:
            over_sub = over_bets[over_bets["edge"] > min_edge]
            under_sub = under_bets[under_bets["edge"] < -min_edge]
            if len(over_sub) + len(under_sub) < 10:
                continue
            pnl_o = sum(10*(r["best_over"]-1) if r["actual"]==1 else -10 for _, r in over_sub.iterrows()) if len(over_sub) > 0 else 0
            pnl_u = sum(10*(r["best_under"]-1) if r["actual"]==0 else -10 for _, r in under_sub.iterrows()) if len(under_sub) > 0 else 0
            n_total = len(over_sub) + len(under_sub)
            roi = (pnl_o + pnl_u) / (n_total * 10) * 100
            print(f"    |edge| > {min_edge*100:.0f}pp: N={n_total}  "
                  f"PnL=${pnl_o+pnl_u:+.0f}  ROI={roi:+.1f}%")

        # ── SPLIT-HALF VALIDATION ──
        print(f"\n  --- SPLIT-HALF ---")
        dates_sorted = sorted(merged["game_date"].unique())
        mid = dates_sorted[len(dates_sorted) // 2]

        for half_name, sub in [("First half", merged[merged["game_date"] <= mid]),
                                ("Second half", merged[merged["game_date"] > mid])]:
            if len(sub) < 50:
                continue
            over_sub = sub[sub["edge"] > 0]
            under_sub = sub[sub["edge"] < 0]
            pnl_o = sum(10*(r["best_over"]-1) if r["actual"]==1 else -10 for _, r in over_sub.iterrows()) if len(over_sub) > 0 else 0
            pnl_u = sum(10*(r["best_under"]-1) if r["actual"]==0 else -10 for _, r in under_sub.iterrows()) if len(under_sub) > 0 else 0
            n_h = len(over_sub) + len(under_sub)
            mdl_auc_h = roc_auc_score(sub["actual"], sub["model_prob"]) if sub["actual"].nunique() > 1 else 0
            mkt_auc_h = roc_auc_score(sub["actual"], sub["market_prob"]) if sub["actual"].nunique() > 1 else 0
            print(f"  {half_name} (<= {mid if half_name=='First half' else 'end'}):")
            print(f"    N={n_h}  Model AUC={mdl_auc_h:.4f}  Market AUC={mkt_auc_h:.4f}  "
                  f"PnL=${pnl_o+pnl_u:+.0f}  ROI={(pnl_o+pnl_u)/(n_h*10)*100:+.1f}%")

        # ── CALIBRATION: MODEL vs MARKET ──
        print(f"\n  --- CALIBRATION (model vs market vs actual) ---")
        for label_q, pred_col in [("Market quintile", "market_prob"), ("Model quintile", "model_prob")]:
            q_edges = np.percentile(merged[pred_col], np.arange(0, 101, 20))
            print(f"  {label_q}:")
            for i in range(5):
                mask = (merged[pred_col] >= q_edges[i]) & (merged[pred_col] < q_edges[i+1] + 0.0001)
                if mask.sum() == 0:
                    continue
                act_q = merged.loc[mask, "actual"].mean()
                mkt_q = merged.loc[mask, "market_prob"].mean()
                mdl_q = merged.loc[mask, "model_prob"].mean()
                print(f"    Q{i}: actual={act_q*100:.1f}%  market={mkt_q*100:.1f}%  model={mdl_q*100:.1f}%  N={mask.sum()}")

    # ── FINAL VERDICT ──
    print(f"\n{'='*70}")
    print(f"=== FINAL VERDICT ===")
    print(f"{'='*70}")
    print(f"THE BAR: model must beat de-vigged close by MORE than the ~6-7% vig.")
    print(f"AUC parity (0.641 vs 0.639) is a TIE. A tie LOSES to vig.")
    print(f"Only CLV-positive net of vig, at adequate N, split-half stable = REAL edge.")
    print(f"Check the numbers above. If overall ROI < 0: parity, clean stop.")
    print(f"If a segment ROI > 0 at N>100 in both halves: bettable pocket.")

    db.close()


if __name__ == "__main__":
    run()
