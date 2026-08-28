"""
shadow_prop — Forward shadow arm for batter props (HR + Total Bases).

Uses the Statcast-augmented GBC model (barrel%/xwOBA features, Bayesian
shrinkage, PIT rolling stats, opposing pitcher). Generates $0 paper bets
when model prob diverges from de-vigged market implied.

Architecture: mock_bets + mock_bet_legs (Architecture B), same as
ats_ungated_shadow. Settlement via _settle_player_prop_leg() in mock_bets.py.

Source tag: 'shadow_prop'
Bet category: 'player_prop'
Market types: 'player_home_runs', 'player_total_bases'
"""

import os
import io
import time
import uuid
import sqlite3
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_CT = ZoneInfo("America/Chicago")
_HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

# ── Statcast features from Baseball Savant (cached per session) ──
_STATCAST_CACHE = None

_SC_FEATURE_MAP = {
    "avg_hit_speed": "sc_avg_ev", "max_hit_speed": "sc_max_ev",
    "ev95percent": "sc_hard_hit_pct", "brl_percent": "sc_barrel_pct",
    "brl_pa": "sc_barrel_pa", "avg_hit_angle": "sc_avg_la",
    "anglesweetspotpercent": "sc_sweet_spot_pct",
    "est_woba": "sc_xwoba", "est_ba": "sc_xba", "est_slg": "sc_xslg",
}

BASE_FEATURES = [
    "szn_avg", "szn_slg", "szn_iso", "szn_hr_rate", "szn_k_rate", "szn_bb_rate",
    "szn_ab", "szn_games", "recent_avg", "recent_hr", "recent_tb", "recent_k",
    "recent_ab", "batting_order", "is_home",
    "opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip",
]

SC_FEATURES = list(_SC_FEATURE_MAP.values())
ALL_FEATURES = BASE_FEATURES + SC_FEATURES


def _fetch_savant(endpoint, year):
    """Fetch season-level Statcast from Baseball Savant."""
    url = f"https://baseballsavant.mlb.com/leaderboard/{endpoint}?type=batter&year={year}&position=&team=&min=25&csv=true"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text)).assign(year=year)
    except Exception as e:
        logger.warning("[shadow_prop] Savant fetch failed for %s/%s: %s", endpoint, year, e)
    return None


def _load_statcast():
    """Load and cache Statcast data for all training years."""
    global _STATCAST_CACHE
    if _STATCAST_CACHE is not None:
        return _STATCAST_CACHE

    logger.info("[shadow_prop] Loading Statcast data from Baseball Savant...")
    all_sc, all_exp = [], []
    for year in range(2021, 2026):
        sc = _fetch_savant("statcast", year)
        exp = _fetch_savant("expected_statistics", year)
        if sc is not None:
            all_sc.append(sc)
        if exp is not None:
            all_exp.append(exp)
        time.sleep(0.5)

    if not all_sc:
        logger.error("[shadow_prop] No Statcast data loaded!")
        return pd.DataFrame()

    sc_df = pd.concat(all_sc, ignore_index=True)
    exp_df = pd.concat(all_exp, ignore_index=True)
    merged = sc_df.merge(exp_df, on=["player_id", "year"], how="outer", suffixes=("", "_exp"))

    result = merged[["player_id", "year"]].copy()
    for old_col, new_col in _SC_FEATURE_MAP.items():
        if old_col in merged.columns:
            result[new_col] = pd.to_numeric(merged[old_col], errors="coerce")

    # Prior-season shift: use year N stats for year N+1 games
    result["use_season"] = (result["year"] + 1).astype(str)
    _STATCAST_CACHE = result
    logger.info("[shadow_prop] Statcast loaded: %d rows", len(result))
    return result


def _compute_rolling_features(batter_df, games_df):
    """Point-in-time rolling batter features with Bayesian shrinkage."""
    merged = batter_df.merge(games_df[["game_pk", "game_date"]], on="game_pk", how="inner")
    merged.sort_values(["player_id", "game_date", "game_pk"], inplace=True)

    prior_ab, prior_avg, prior_slg, prior_hr = 50, 0.248, 0.395, 0.033
    rows = []
    for pid, grp in merged.groupby("player_id"):
        grp = grp.sort_values("game_date")
        cum = {k: 0 for k in ["ab", "hits", "hr", "doubles", "triples",
                                "bb", "k", "hbp", "sf", "tb", "games"]}
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
                r_ab = sum(x["ab"] for x in recent)

                rows.append({
                    "player_id": pid, "player_name": row["player_name"],
                    "game_pk": row["game_pk"], "game_date": row["game_date"],
                    "team_name": row["team_name"], "is_home": row["is_home"],
                    "ab": row["ab"], "hits": row["hits"], "hr": row["hr"], "tb": row["tb"],
                    "szn_avg": szn_avg, "szn_slg": szn_slg,
                    "szn_iso": szn_slg - szn_avg, "szn_hr_rate": szn_hr_rate,
                    "szn_k_rate": cum["k"] / pa if pa > 0 else 0,
                    "szn_bb_rate": cum["bb"] / pa if pa > 0 else 0,
                    "szn_ab": cum["ab"], "szn_games": cum["games"],
                    "recent_avg": sum(x["hits"] for x in recent) / r_ab if r_ab > 0 else szn_avg,
                    "recent_hr": sum(x["hr"] for x in recent) / r_ab if r_ab > 0 else szn_hr_rate,
                    "recent_tb": sum(x["tb"] for x in recent) / r_ab if r_ab > 0 else szn_slg,
                    "recent_k": sum(x["k"] for x in recent) / r_ab if r_ab > 0 else (cum["k"] / pa if pa > 0 else 0),
                    "recent_ab": r_ab,
                })

            for k2 in ["ab", "hits", "hr", "doubles", "triples", "bb", "k", "hbp", "sf", "tb"]:
                cum[k2] += row[k2]
            cum["games"] += 1
            recent.append(row.to_dict())
            if len(recent) > 14:
                recent.pop(0)

    return pd.DataFrame(rows)


def _add_matchup_features(feat_df, hdb):
    """Add opposing pitcher season stats via frequency-based team ID."""
    pitcher_logs = pd.read_sql_query(
        "SELECT player_id as pitcher_id, game_pk, ip FROM pitcher_game_logs", hdb)
    pitcher_season = pd.read_sql_query(
        "SELECT player_id as pitcher_id, season, era, whip, k9 as k_per_9, hr9 as hr_per_9, fip FROM pitcher_stats", hdb)
    games = pd.read_sql_query("SELECT game_id, home_team, away_team FROM games WHERE sport='MLB'", hdb)
    games["game_pk"] = games["game_id"].str.replace("MLB_", "").astype(int)

    pg = pitcher_logs.merge(games[["game_pk", "home_team", "away_team"]], on="game_pk")
    tm = {}
    for pid, grp in pg.groupby("pitcher_id"):
        c = {}
        for _, r in grp.iterrows():
            for t in [r["home_team"], r["away_team"]]:
                c[t] = c.get(t, 0) + 1
        tm[pid] = max(c, key=c.get) if c else None

    gp = pitcher_logs.groupby("game_pk").apply(
        lambda g: g.nlargest(2, "ip")[["pitcher_id"]].values.flatten().tolist()
    ).to_dict()

    opp = []
    for _, row in feat_df.iterrows():
        pitchers = gp.get(row["game_pk"], [])
        season = row["game_date"][:4]
        found = False
        for pid in pitchers:
            if tm.get(pid) and tm[pid] != row["team_name"]:
                ps = pitcher_season[(pitcher_season["pitcher_id"] == pid) &
                                     (pitcher_season["season"] == season)]
                if len(ps) > 0:
                    opp.append(ps.iloc[0][["era", "whip", "k_per_9", "hr_per_9", "fip"]].to_dict())
                    found = True
                    break
        if not found:
            opp.append({})

    odf = pd.DataFrame(opp, index=feat_df.index)
    for c, n in [("era", "opp_era"), ("whip", "opp_whip"), ("k_per_9", "opp_k9"),
                  ("hr_per_9", "opp_hr9"), ("fip", "opp_fip")]:
        feat_df[n] = odf[c] if c in odf.columns else np.nan
        feat_df[n] = feat_df[n].fillna(feat_df[n].median())
    return feat_df


def _add_lineup_features(feat_df, hdb):
    """Add batting order from game_lineups."""
    lineups = pd.read_sql_query("SELECT game_pk, player_id, batting_order FROM game_lineups", hdb)
    lineups["bo"] = (lineups["batting_order"] / 100).astype(int)
    feat_df = feat_df.merge(lineups[["game_pk", "player_id", "bo"]],
                             on=["game_pk", "player_id"], how="left")
    feat_df.rename(columns={"bo": "batting_order"}, inplace=True)
    feat_df["batting_order"] = feat_df["batting_order"].fillna(feat_df["batting_order"].median())
    return feat_df


def _devig(over_price, under_price):
    """Remove vig from 2-way market. Returns devigged over probability."""
    if over_price <= 1.0 or under_price <= 1.0:
        return None
    t = 1.0 / over_price + 1.0 / under_price
    return (1.0 / over_price) / t if t > 0 else None


# ── Cached trained models ──
_MODELS = {}


def _train_models():
    """Train HR and TB GBC models on 2022-2025 data. Cached after first call."""
    if _MODELS:
        return _MODELS

    logger.info("[shadow_prop] Training HR and TB models...")
    hdb = sqlite3.connect(_HIST_DB)

    statcast = _load_statcast()

    batter_df = pd.read_sql_query(
        "SELECT game_pk, player_id, player_name, team_name, is_home, "
        "ab, hits, doubles, triples, hr, bb, hbp, sf, k, tb FROM batter_game_stats", hdb)
    games_df = pd.read_sql_query("SELECT game_id, game_date FROM games WHERE sport='MLB'", hdb)
    games_df["game_pk"] = games_df["game_id"].str.replace("MLB_", "").astype(int)

    feat_df = _compute_rolling_features(batter_df, games_df)
    feat_df = _add_matchup_features(feat_df, hdb)
    feat_df = _add_lineup_features(feat_df, hdb)

    # Join Statcast (prior season)
    feat_df["season"] = feat_df["game_date"].str[:4]
    sc_join = statcast[["player_id", "use_season"] + SC_FEATURES].rename(
        columns={"use_season": "season"})
    feat_df = feat_df.merge(sc_join, on=["player_id", "season"], how="left")
    for c in SC_FEATURES:
        if c in feat_df.columns:
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    feat_df["year"] = feat_df["game_date"].str[:4].astype(int)
    train = feat_df[feat_df["year"] <= 2025].dropna(subset=ALL_FEATURES)

    X_train = train[ALL_FEATURES].values

    # HR model
    y_hr = (train["hr"] >= 1).astype(int).values
    gbc_hr = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=42)
    gbc_hr.fit(X_train, y_hr)

    # TB model
    y_tb = (train["tb"] >= 2).astype(int).values
    gbc_tb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=42)
    gbc_tb.fit(X_train, y_tb)

    # Store feature medians for filling missing values on new data
    medians = {c: train[c].median() for c in ALL_FEATURES}

    hdb.close()
    _MODELS["hr"] = gbc_hr
    _MODELS["tb"] = gbc_tb
    _MODELS["medians"] = medians
    logger.info("[shadow_prop] Models trained: HR and TB (train N=%d)", len(train))
    return _MODELS


def _get_today_batter_features(today_str):
    """Build feature rows for today's batters using point-in-time data."""
    hdb = sqlite3.connect(_HIST_DB)
    statcast = _load_statcast()

    # Load all batter game stats (need full history for rolling features)
    batter_df = pd.read_sql_query(
        "SELECT game_pk, player_id, player_name, team_name, is_home, "
        "ab, hits, doubles, triples, hr, bb, hbp, sf, k, tb FROM batter_game_stats", hdb)
    games_df = pd.read_sql_query("SELECT game_id, game_date FROM games WHERE sport='MLB'", hdb)
    games_df["game_pk"] = games_df["game_id"].str.replace("MLB_", "").astype(int)

    # Compute rolling features — the last row per player is their current state
    feat_df = _compute_rolling_features(batter_df, games_df)

    # Get latest features per player (most recent game)
    feat_df = feat_df.sort_values("game_date")
    latest = feat_df.groupby("player_name").last().reset_index()

    # Add matchup features (placeholder — use median for now since we don't
    # have today's probable pitchers in the DB yet)
    models = _train_models()
    medians = models["medians"]
    for col in ["opp_era", "opp_whip", "opp_k9", "opp_hr9", "opp_fip"]:
        if col not in latest.columns:
            latest[col] = medians.get(col, 0)

    # Batting order — use most recent
    if "batting_order" not in latest.columns:
        latest["batting_order"] = medians.get("batting_order", 5)

    # Join Statcast
    season = today_str[:4]
    latest["season"] = season
    sc_join = statcast[["player_id", "use_season"] + SC_FEATURES].rename(
        columns={"use_season": "season"})
    latest = latest.merge(sc_join, on=["player_id", "season"], how="left")
    for c in SC_FEATURES:
        if c in latest.columns:
            latest[c] = latest[c].fillna(medians.get(c, 0))
        else:
            latest[c] = medians.get(c, 0)

    # Fill any remaining NaN with medians
    for c in ALL_FEATURES:
        if c in latest.columns:
            latest[c] = latest[c].fillna(medians.get(c, 0))
        else:
            latest[c] = medians.get(c, 0)

    hdb.close()
    return latest


def generate_prop_shadow(db=None, today_str=None):
    """Generate shadow prop bets for today's slate.

    For each player with HR/TB prop odds today:
    1. Compute model probability
    2. De-vig market probability
    3. If model disagrees with market (|edge| > 0), log paper bet

    Returns list of stored bet dicts.
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[shadow_prop] Generating prop shadow for {today_str}")

    # Train models (cached after first call)
    models = _train_models()
    gbc_hr = models["hr"]
    gbc_tb = models["tb"]
    medians = models["medians"]

    # Get today's prop odds from historical.db
    hdb = sqlite3.connect(_HIST_DB)
    props = pd.read_sql_query("""
        SELECT player_name, game_date, bookmaker, over_under, price, point, market_key
        FROM player_props
        WHERE sport_key = 'baseball_mlb'
          AND game_date = ?
          AND market_key IN ('batter_home_runs', 'batter_total_bases')
    """, hdb, params=(today_str,))

    if len(props) == 0:
        print(f"[shadow_prop] No prop odds for {today_str}")
        hdb.close()
        return []

    print(f"[shadow_prop] {len(props)} prop rows for {today_str}")

    # Get latest batter features
    batter_feats = _get_today_batter_features(today_str)
    print(f"[shadow_prop] {len(batter_feats)} batters with features")

    # Process each prop market
    stored_bets = []
    run_id = str(uuid.uuid4())

    for market_key, point_val, model, market_type_str, label in [
        ("batter_home_runs", 0.5, gbc_hr, "player_home_runs", "HR"),
        ("batter_total_bases", 1.5, gbc_tb, "player_total_bases", "TB"),
    ]:
        mkt_props = props[(props["market_key"] == market_key) & (props["point"] == point_val)]
        if len(mkt_props) == 0:
            continue

        # Group by (player, date, bookmaker) and de-vig
        grouped = mkt_props.groupby(["player_name", "game_date", "bookmaker"])
        devigged = []
        for (player, date, bk), group in grouped:
            overs = group[group["over_under"].str.lower() == "over"]
            unders = group[group["over_under"].str.lower() == "under"]
            if len(overs) == 0 or len(unders) == 0:
                continue
            op = overs.iloc[0]["price"]
            up = unders.iloc[0]["price"]
            dv = _devig(op, up)
            if dv is None:
                continue
            devigged.append({
                "player_name": player, "bookmaker": bk,
                "over_price": op, "under_price": up,
                "market_prob": dv,
            })

        if not devigged:
            continue

        dv_df = pd.DataFrame(devigged)
        # Best price per player (across bookmakers)
        best = dv_df.groupby("player_name").agg(
            market_prob=("market_prob", "mean"),
            best_over=("over_price", "max"),
            best_under=("under_price", "max"),
        ).reset_index()

        # Match with model features
        matched = best.merge(batter_feats[["player_name"] + ALL_FEATURES],
                              on="player_name", how="inner")

        if len(matched) == 0:
            continue

        # Generate predictions
        X = matched[ALL_FEATURES].values
        matched["model_prob"] = model.predict_proba(X)[:, 1]
        matched["edge"] = matched["model_prob"] - matched["market_prob"]

        print(f"[shadow_prop] {label}: {len(matched)} players matched, "
              f"mean edge={matched['edge'].mean()*100:+.2f}pp")

        # Generate bets for ALL players where edge != 0
        # (Log everything for forward measurement — filtering is analysis-side)
        for _, row in matched.iterrows():
            player = row["player_name"]
            model_prob = row["model_prob"]
            market_prob = row["market_prob"]
            edge = row["edge"]

            if edge > 0:
                # Model says OVER
                direction = "Over"
                odds = row["best_over"]
                be = 1.0 / odds
            else:
                # Model says UNDER
                direction = "Under"
                odds = row["best_under"]
                be = 1.0 / odds

            wp = model_prob * 100 if direction == "Over" else (1 - model_prob) * 100
            ev = (wp / 100) * (odds - 1) - (1 - wp / 100)

            line = point_val
            desc = f"{player} {direction} {line} {label}"

            # Dedup
            if db is not None:
                existing = db.execute(_text(
                    "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
                    "AND source='shadow_prop' AND bet_info LIKE :bi"
                ), {"gd": today_str, "bi": f"%{player}%{label}%"}).scalar()
                if existing > 0:
                    continue

            mock_id = str(uuid.uuid4())
            bet_info = (f"{desc} | model={model_prob*100:.1f}% market={market_prob*100:.1f}% "
                        f"edge={edge*100:+.1f}pp")

            if db is not None:
                mock = MockBet(
                    id=mock_id,
                    generation_run_id=run_id,
                    game_date=today_str,
                    sport="MLB",
                    bet_type="straight",
                    odds=odds,
                    amount=0.0,
                    legs=1,
                    bet_info=bet_info,
                    predicted_win_prob=wp,
                    model_used="batter_prop_statcast_v1",
                    status="PENDING",
                    source="shadow_prop",
                    cohort="clean",
                    notes=f"market_type={market_type_str} devig_prob={market_prob:.4f} "
                          f"model_prob={model_prob:.4f} edge_pp={edge*100:.1f}",
                )
                db.add(mock)

                mock_leg = MockBetLeg(
                    mock_bet_id=mock_id,
                    leg_index=0,
                    description=desc,
                    market_type=market_type_str,
                    sport="MLB",
                    win_prob=wp,
                    ev=ev,
                    is_alt_line=False,
                    bet_category="player_prop",
                )
                db.add(mock_leg)

            stored_bets.append({
                "player": player, "market": label, "direction": direction,
                "model_prob": model_prob, "market_prob": market_prob,
                "edge_pp": edge * 100, "odds": odds, "wp": wp,
            })

    if db is not None and stored_bets:
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            print(f"[shadow_prop] Commit error: {exc}")
            return []

    hdb.close()

    print(f"[shadow_prop] Stored {len(stored_bets)} shadow prop bets "
          f"(source=shadow_prop)")

    # Summary
    hr_bets = [b for b in stored_bets if b["market"] == "HR"]
    tb_bets = [b for b in stored_bets if b["market"] == "TB"]
    if hr_bets:
        over_hr = sum(1 for b in hr_bets if b["direction"] == "Over")
        print(f"[shadow_prop] HR: {len(hr_bets)} bets ({over_hr} Over, {len(hr_bets)-over_hr} Under)")
    if tb_bets:
        over_tb = sum(1 for b in tb_bets if b["direction"] == "Over")
        print(f"[shadow_prop] TB: {len(tb_bets)} bets ({over_tb} Over, {len(tb_bets)-over_tb} Under)")

    return stored_bets


if __name__ == "__main__":
    from database import SessionLocal, init_db
    init_db()
    db = SessionLocal()
    try:
        bets = generate_prop_shadow(db)
        if bets:
            print(f"\n[shadow_prop] Summary: {len(bets)} bets")
            for b in sorted(bets, key=lambda x: -abs(x["edge_pp"]))[:15]:
                print(f"  {b['direction']:>5s} {b['market']:>2s} {b['player']:<25s} "
                      f"model={b['model_prob']*100:5.1f}% mkt={b['market_prob']*100:5.1f}% "
                      f"edge={b['edge_pp']:+5.1f}pp odds={b['odds']:.2f}")
    finally:
        db.close()
