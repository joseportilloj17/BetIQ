"""
shadow_lineup.py — Shadow arm for the lineup composition feature.

Runs the ATS model WITH lineup_gap_resid (residualized lineup quality,
orthogonal to ops_diff) in parallel with the plain baseline on the same
daily slate.  The ONLY difference between shadow_lineup and the baseline
is the lineup feature — isolates the feature's forward marginal contribution.

Architecture:
  - Reuses get_available_legs() + score_leg() pattern from ats_ungated_shadow
  - Loads a separately-trained ATS model (mlb_ats_lineup_*.pkl) that has
    22 rolling features + 3 spread features = 25 total
  - For today's games, fetches lineups from MLB API and computes per-batter
    cumulative OPS → lineup quality → residualized gap
  - Stores individual legs as 1-leg mock_bets, source='shadow_lineup'
  - Settled by the same ESPN pipeline as all other shadow arms

Construction: lineup_gap_resid
  - For each team-game: mean cumulative OPS of 9 starters (Bayesian shrunk)
    minus team cumulative OPS = lineup_gap
  - lineup_gap_diff = h_lineup_gap - a_lineup_gap
  - Residualize against ops_diff (r=0.000 by construction)
  - This is the ORTHOGONAL component: what lineup composition tells you
    that team OPS doesn't already capture

Training: train_lineup_model()
  - Same pipeline as production ATS (GBC 500/0.03/d4 + isotonic cv=3)
  - Same spread augmentation (-1.5, +1.5, +2.5, +3.5, +4.5)
  - Adds lineup_gap_resid as 22nd rolling feature
  - Saves to data/submodels/mlb_ats_lineup_*.pkl

Forward read: N=40 settled days → WR-vs-BE + CLV vs plain baseline
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests

log = logging.getLogger("shadow_lineup")
_CT = ZoneInfo("America/Chicago")

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")
_SUBMODEL_DIR = os.path.join(_DATA_DIR, "submodels")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Build lineup features from historical data
# ═══════════════════════════════════════════════════════════════════════════════

# Shrinkage prior: 150 PA ≈ ~37 games
PRIOR_PA = 150


def _league_ops_by_season(conn: sqlite3.Connection) -> dict[str, float]:
    """Compute league-average OPS per season from batter_game_stats."""
    import pandas as pd
    bgs = pd.read_sql_query(
        "SELECT tgb.season, SUM(bgs.hits) AS h, SUM(bgs.ab) AS ab, SUM(bgs.tb) AS tb, "
        "SUM(bgs.bb) AS bb, SUM(bgs.hbp) AS hbp, SUM(bgs.sf) AS sf "
        "FROM batter_game_stats bgs "
        "JOIN team_game_batting tgb ON bgs.game_pk = tgb.game_pk AND bgs.team_name = tgb.team_name "
        "GROUP BY tgb.season",
        conn,
    )
    result = {}
    for _, r in bgs.iterrows():
        denom = r["ab"] + r["bb"] + r["hbp"] + r["sf"]
        if denom > 0 and r["ab"] > 0:
            obp = (r["h"] + r["bb"] + r["hbp"]) / denom
            slg = r["tb"] / r["ab"]
            result[r["season"]] = round(obp + slg, 4)
        else:
            result[r["season"]] = 0.720
    return result


def build_lineup_features_for_training() -> "pd.DataFrame":
    """
    Build game-level lineup_gap_resid for ALL historical games.

    Returns DataFrame with game_id, lineup_gap_resid, plus other lineup cols.
    Uses the same logic as lineup_composition_test.py but adds residualization.
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    conn = sqlite3.connect(_HIST_DB)

    lg_ops = _league_ops_by_season(conn)

    # Load batter game stats with game metadata
    bgs = pd.read_sql_query("""
        SELECT bgs.game_pk, bgs.player_id, bgs.team_name,
               bgs.is_home, bgs.ab, bgs.hits AS hits, bgs.tb,
               bgs.bb, bgs.hbp, bgs.sf,
               tgb.game_date, tgb.season
        FROM batter_game_stats bgs
        JOIN team_game_batting tgb ON bgs.game_pk = tgb.game_pk
                                  AND bgs.team_name = tgb.team_name
        ORDER BY bgs.player_id, tgb.season, tgb.game_date
    """, conn)

    # All rows in batter_game_stats are starters (backfill used battingOrder = 9)
    # Compute per-batter cumulative OPS (point-in-time, shifted)
    bgs = bgs.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    cum_cols = ["ab", "hits", "tb", "bb", "hbp", "sf"]
    for col in cum_cols:
        bgs[f"cum_{col}"] = bgs.groupby(["player_id", "season"])[col].transform(
            lambda x: x.cumsum().shift(1)
        )

    cum_pa = (bgs["cum_ab"].fillna(0) + bgs["cum_bb"].fillna(0) +
              bgs["cum_hbp"].fillna(0) + bgs["cum_sf"].fillna(0))

    denom = cum_pa
    raw_obp = np.where(denom > 0,
        (bgs["cum_hits"].fillna(0) + bgs["cum_bb"].fillna(0) + bgs["cum_hbp"].fillna(0)) / denom, 0)
    raw_slg = np.where(bgs["cum_ab"].fillna(0) > 0,
        bgs["cum_tb"].fillna(0) / bgs["cum_ab"].fillna(1), 0)
    raw_ops = raw_obp + raw_slg

    lg_prior = bgs["season"].map(lg_ops).fillna(0.720)
    bgs["batter_ops_pit"] = np.where(
        cum_pa > 0,
        (cum_pa * raw_ops + PRIOR_PA * lg_prior) / (cum_pa + PRIOR_PA),
        lg_prior
    )

    # Lineup quality = mean of 9 starters' cumulative OPS
    lineup_q = (
        bgs.groupby(["game_pk", "team_name", "is_home"])
        .agg(lineup_ops=("batter_ops_pit", "mean"))
        .reset_index()
    )

    # Team cumulative OPS (point-in-time)
    tgb = pd.read_sql_query("""
        SELECT team_name, season, game_date, game_pk, is_home,
               at_bats, hits, total_bases, walks, hit_by_pitch, sac_flies
        FROM team_game_batting
        ORDER BY team_name, season, game_date
    """, conn)
    conn.close()

    for src, dst in [("at_bats", "t_ab"), ("hits", "t_h"), ("total_bases", "t_tb"),
                     ("walks", "t_bb"), ("hit_by_pitch", "t_hbp"), ("sac_flies", "t_sf")]:
        tgb[dst] = tgb.groupby(["team_name", "season"])[src].transform(
            lambda x: x.cumsum().shift(1)
        )

    t_denom = tgb["t_ab"].fillna(0) + tgb["t_bb"].fillna(0) + \
              tgb["t_hbp"].fillna(0) + tgb["t_sf"].fillna(0)
    t_obp = np.where(t_denom > 0,
        (tgb["t_h"].fillna(0) + tgb["t_bb"].fillna(0) + tgb["t_hbp"].fillna(0)) / t_denom, 0)
    t_slg = np.where(tgb["t_ab"].fillna(0) > 0,
        tgb["t_tb"].fillna(0) / tgb["t_ab"].fillna(1), 0)
    lg_prior_t = tgb["season"].map(lg_ops).fillna(0.720)
    tgb["team_ops_pit"] = np.where(
        t_denom > 0,
        (t_denom * (t_obp + t_slg) + PRIOR_PA * lg_prior_t) / (t_denom + PRIOR_PA),
        lg_prior_t
    )

    # Merge and compute gap
    tgb_slim = tgb[["game_pk", "team_name", "is_home", "team_ops_pit"]].copy()
    merged = lineup_q.merge(tgb_slim, on=["game_pk", "team_name", "is_home"], how="inner")
    merged["lineup_gap"] = merged["lineup_ops"] - merged["team_ops_pit"]

    # Pivot to game level
    home = merged[merged["is_home"] == 1][["game_pk", "lineup_gap"]].rename(
        columns={"lineup_gap": "h_lineup_gap"})
    away = merged[merged["is_home"] == 0][["game_pk", "lineup_gap"]].rename(
        columns={"lineup_gap": "a_lineup_gap"})

    game_level = home.merge(away, on="game_pk", how="inner")
    game_level["lineup_gap_diff"] = game_level["h_lineup_gap"] - game_level["a_lineup_gap"]
    game_level["game_id"] = "MLB_" + game_level["game_pk"].astype(str)

    # Now merge with feature matrix to get ops_diff for residualization
    from features import build_feature_matrix
    fm = build_feature_matrix("MLB")
    fm_slim = fm[["game_id", "ops_diff"]].copy()

    game_level = game_level.merge(fm_slim, on="game_id", how="inner")

    # Residualize: regress lineup_gap_diff ~ ops_diff, keep residual
    valid_mask = game_level["ops_diff"].notna() & game_level["lineup_gap_diff"].notna()
    valid = game_level[valid_mask].copy()

    lr = LinearRegression()
    lr.fit(valid[["ops_diff"]].values, valid["lineup_gap_diff"].values)
    valid["lineup_gap_resid"] = valid["lineup_gap_diff"] - lr.predict(
        valid[["ops_diff"]].values)

    # Store the regression coefficients for use at inference time
    resid_params = {
        "coef": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
    }
    params_path = os.path.join(_SUBMODEL_DIR, "lineup_resid_params.json")
    with open(params_path, "w") as f:
        json.dump(resid_params, f)
    log.info("Saved residualization params: coef=%.6f, intercept=%.6f",
             resid_params["coef"], resid_params["intercept"])

    return valid[["game_id", "lineup_gap_resid"]], resid_params


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Train lineup-augmented ATS model
# ═══════════════════════════════════════════════════════════════════════════════

def train_lineup_model(save: bool = True) -> dict:
    """
    Train ATS model with lineup_gap_resid as 22nd feature (+ 3 spread = 25).

    Same pipeline as production:
      - GBC(500, lr=0.03, depth=4, min_leaf=5) + isotonic cv=3
      - Spread augmentation (-1.5, +1.5, +2.5, +3.5, +4.5)
      - Imputer + StandardScaler
    """
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler

    from features import build_feature_matrix, TARGET_ATS, MODEL_FEATURE_COLS_V1_PRUNED

    print("[lineup-model] Building feature matrix...")
    matrix = build_feature_matrix("MLB")

    # Build lineup features
    print("[lineup-model] Building lineup features...")
    lineup_df, resid_params = build_lineup_features_for_training()
    matrix = matrix.merge(lineup_df, on="game_id", how="left")

    # Filter to labeled rows
    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    labeled = labeled[labeled["lineup_gap_resid"].notna()].copy()
    print(f"[lineup-model] {len(labeled)} labeled games with lineup feature")

    # Feature columns: V1 pruned + lineup_gap_resid
    feat_cols = [c for c in MODEL_FEATURE_COLS_V1_PRUNED if c in labeled.columns]
    feat_cols.append("lineup_gap_resid")

    # Spread augmentation (same as production)
    _AUGMENT_SPREADS = [-1.5, 1.5, 2.5, 3.5, 4.5]
    margin = (
        pd.to_numeric(labeled["home_score"], errors="coerce")
        - pd.to_numeric(labeled["away_score"], errors="coerce")
    )
    aug_rows = []
    for spread_val in _AUGMENT_SPREADS:
        chunk = labeled.copy()
        chunk["_spread_value"] = float(spread_val)
        chunk["_spread_abs"] = abs(spread_val)
        chunk["_is_fav_spread"] = 1.0 if spread_val < 0 else 0.0
        ats_val = margin + spread_val
        chunk[TARGET_ATS] = np.where(ats_val > 0, 1.0,
                            np.where(ats_val < 0, 0.0, np.nan))
        aug_rows.append(chunk)
    labeled = pd.concat(aug_rows, ignore_index=True)
    labeled = labeled[labeled[TARGET_ATS].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    feat_cols = feat_cols + ["_spread_value", "_spread_abs", "_is_fav_spread"]
    print(f"[lineup-model] Augmented: {len(labeled)} rows, {len(feat_cols)} features")

    X = labeled[feat_cols].values.astype(float)
    y = labeled[TARGET_ATS].values.astype(float)

    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Impute → scale
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_te_s = sc.transform(X_te_imp)

    # Train
    print("[lineup-model] Training GBC(500, lr=0.03, d4) + isotonic cv=3...")
    base = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    # Evaluate
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    auc = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    print(f"[lineup-model] AUC: {auc:.4f}, Brier: {brier:.4f}")

    # Feature importances
    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])
    print("[lineup-model] Top features:")
    for fname, imp_val in imps[:10]:
        marker = " *** LINEUP" if "lineup" in fname else ""
        print(f"  {fname:30s}: {imp_val:.4f}{marker}")

    # Find lineup feature rank
    for i, (fname, _) in enumerate(imps):
        if fname == "lineup_gap_resid":
            print(f"\n[lineup-model] lineup_gap_resid rank: #{i+1}/{len(feat_cols)}")
            break

    if save:
        os.makedirs(_SUBMODEL_DIR, exist_ok=True)
        prefix = "mlb_ats_lineup"
        with open(os.path.join(_SUBMODEL_DIR, f"{prefix}_clf.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(_SUBMODEL_DIR, f"{prefix}_scaler.pkl"), "wb") as f:
            pickle.dump(sc, f)
        with open(os.path.join(_SUBMODEL_DIR, f"{prefix}_imputer.pkl"), "wb") as f:
            pickle.dump(imp, f)
        with open(os.path.join(_SUBMODEL_DIR, f"{prefix}_feat_cols.pkl"), "wb") as f:
            pickle.dump(feat_cols, f)
        meta = {
            "sport": "MLB",
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "n_features": len(feat_cols),
            "n_train": int(len(X_tr)),
            "spread_aware": True,
            "lineup_feature": "lineup_gap_resid",
            "resid_coef": resid_params["coef"],
            "resid_intercept": resid_params["intercept"],
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(_SUBMODEL_DIR, f"{prefix}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[lineup-model] Saved to {_SUBMODEL_DIR}/{prefix}_*.pkl")

    return {"auc": auc, "brier": brier, "n_features": len(feat_cols),
            "n_train": len(X_tr), "n_test": len(X_te)}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Daily lineup feature computation (for live scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_todays_lineups(today_str: str) -> list[dict]:
    """Fetch confirmed lineups from MLB API for today's games.

    Returns list of dicts: {game_pk, home_team, away_team,
                            home_lineup: [player_ids], away_lineup: [player_ids]}
    """
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?date={today_str}&sportId=1&hydrate=lineups")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            log.warning("MLB schedule API returned %d", r.status_code)
            return []
    except Exception as exc:
        log.warning("MLB schedule API error: %s", exc)
        return []

    data = r.json()
    games = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            gpk = game.get("gamePk")
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_name = home.get("team", {}).get("name", "")
            away_name = away.get("team", {}).get("name", "")

            # Extract lineup player IDs
            lineups = game.get("lineups", {})
            home_lu = [p.get("id") for p in lineups.get("homePlayers", [])]
            away_lu = [p.get("id") for p in lineups.get("awayPlayers", [])]

            if len(home_lu) >= 9 and len(away_lu) >= 9:
                games.append({
                    "game_pk": gpk,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_lineup": home_lu[:9],
                    "away_lineup": away_lu[:9],
                })

    return games


def _get_batter_cumulative_ops(player_ids: list[int], season: str) -> dict[int, float]:
    """Look up cumulative OPS for each batter from batter_game_stats.

    Returns {player_id: shrunk_cumulative_ops}.
    For batters not in DB, returns league average.
    Uses a single batch query instead of per-player queries.
    """
    conn = sqlite3.connect(_HIST_DB)
    lg_ops = _league_ops_by_season(conn)
    lg_prior = lg_ops.get(season, 0.720)

    result = {pid: lg_prior for pid in player_ids}

    if not player_ids:
        conn.close()
        return result

    placeholders = ",".join("?" * len(player_ids))
    rows = conn.execute(f"""
        SELECT bgs.player_id,
               SUM(bgs.ab) AS ab, SUM(bgs.hits) AS h, SUM(bgs.tb) AS tb,
               SUM(bgs.bb) AS bb, SUM(bgs.hbp) AS hbp, SUM(bgs.sf) AS sf
        FROM batter_game_stats bgs
        JOIN team_game_batting tgb ON bgs.game_pk = tgb.game_pk
                                  AND bgs.team_name = tgb.team_name
        WHERE bgs.player_id IN ({placeholders}) AND tgb.season = ?
        GROUP BY bgs.player_id
    """, list(player_ids) + [season]).fetchall()

    for row in rows:
        pid, ab, h, tb, bb, hbp, sf = row
        if ab and ab > 0:
            pa = ab + bb + hbp + sf
            if pa > 0:
                raw_obp = (h + bb + hbp) / pa
                raw_slg = tb / ab
                raw_ops = raw_obp + raw_slg
                shrunk = (pa * raw_ops + PRIOR_PA * lg_prior) / (pa + PRIOR_PA)
                result[pid] = shrunk

    conn.close()
    return result


def _get_team_cumulative_ops(team_name: str, season: str) -> float:
    """Get team's cumulative season-to-date OPS (point-in-time)."""
    conn = sqlite3.connect(_HIST_DB)
    lg_ops = _league_ops_by_season(conn)
    lg_prior = lg_ops.get(season, 0.720)

    cur = conn.execute("""
        SELECT SUM(at_bats) AS ab, SUM(hits) AS h, SUM(total_bases) AS tb,
               SUM(walks) AS bb, SUM(hit_by_pitch) AS hbp, SUM(sac_flies) AS sf
        FROM team_game_batting
        WHERE team_name = ? AND season = ?
    """, (team_name, season))
    row = cur.fetchone()
    conn.close()

    if row and row[0] and row[0] > 0:
        ab, h, tb, bb, hbp, sf = row
        pa = ab + bb + hbp + sf
        if pa > 0 and ab > 0:
            raw_obp = (h + bb + hbp) / pa
            raw_slg = tb / ab
            raw_ops = raw_obp + raw_slg
            shrunk = (pa * raw_ops + PRIOR_PA * lg_prior) / (pa + PRIOR_PA)
            return shrunk
    return lg_prior


def compute_lineup_gap_resid_today(today_str: str) -> dict[int, float]:
    """Compute lineup_gap_resid for today's games.

    Returns {game_pk: lineup_gap_resid}.

    Uses batched DB queries (single connection, one query for all batters)
    to avoid the N-per-player query bottleneck.
    """
    # Load residualization params
    params_path = os.path.join(_SUBMODEL_DIR, "lineup_resid_params.json")
    if not os.path.exists(params_path):
        log.warning("No lineup_resid_params.json — run train_lineup_model first")
        return {}
    with open(params_path) as f:
        resid_params = json.load(f)

    season = today_str[:4]
    games = _fetch_todays_lineups(today_str)
    if not games:
        log.info("[shadow-lineup] No confirmed lineups for %s", today_str)
        return {}

    log.info("[shadow-lineup] %d games with confirmed lineups", len(games))

    conn = sqlite3.connect(_HIST_DB)

    # 1. League average OPS (once)
    lg_ops = _league_ops_by_season(conn)
    lg_prior = lg_ops.get(season, 0.720)

    # 2. Team season OPS from team_batting_stats (for residualization)
    team_ops_rows = conn.execute(
        "SELECT team_name, ops FROM team_batting_stats WHERE season = ?",
        (season,),
    ).fetchall()
    team_ops_map = {name: ops for name, ops in team_ops_rows if ops}

    # 3. Batch all player IDs into one query
    all_pids = set()
    for game in games:
        all_pids.update(game["home_lineup"])
        all_pids.update(game["away_lineup"])

    batter_ops = {}
    if all_pids:
        placeholders = ",".join("?" * len(all_pids))
        rows = conn.execute(f"""
            SELECT bgs.player_id,
                   SUM(bgs.ab), SUM(bgs.hits), SUM(bgs.tb),
                   SUM(bgs.bb), SUM(bgs.hbp), SUM(bgs.sf)
            FROM batter_game_stats bgs
            JOIN team_game_batting tgb ON bgs.game_pk = tgb.game_pk
                                      AND bgs.team_name = tgb.team_name
            WHERE bgs.player_id IN ({placeholders}) AND tgb.season = ?
            GROUP BY bgs.player_id
        """, list(all_pids) + [season]).fetchall()
        for row in rows:
            pid, ab, h, tb, bb, hbp, sf = row
            if ab and ab > 0:
                pa = ab + bb + hbp + sf
                if pa > 0:
                    raw_ops = (h + bb + hbp) / pa + tb / ab
                    batter_ops[pid] = (pa * raw_ops + PRIOR_PA * lg_prior) / (pa + PRIOR_PA)

    # 4. Team cumulative OPS (point-in-time, from team_game_batting)
    team_cum_rows = conn.execute("""
        SELECT team_name,
               SUM(at_bats), SUM(hits), SUM(total_bases),
               SUM(walks), SUM(hit_by_pitch), SUM(sac_flies)
        FROM team_game_batting WHERE season = ?
        GROUP BY team_name
    """, (season,)).fetchall()
    team_cum_ops = {}
    for name, ab, h, tb, bb, hbp, sf in team_cum_rows:
        if ab and ab > 0:
            pa = ab + bb + hbp + sf
            if pa > 0:
                raw_ops = (h + bb + hbp) / pa + tb / ab
                team_cum_ops[name] = (pa * raw_ops + PRIOR_PA * lg_prior) / (pa + PRIOR_PA)

    conn.close()

    log.info("[shadow-lineup] %d/%d batters found in DB", len(batter_ops), len(all_pids))

    # 5. Compute lineup_gap_resid per game
    result = {}
    for game in games:
        gpk = game["game_pk"]

        # Home lineup quality
        h_ops_list = [batter_ops.get(pid, lg_prior) for pid in game["home_lineup"]]
        h_lineup_ops = np.mean(h_ops_list)
        h_team_ops = team_cum_ops.get(game["home_team"], lg_prior)
        h_gap = h_lineup_ops - h_team_ops

        # Away lineup quality
        a_ops_list = [batter_ops.get(pid, lg_prior) for pid in game["away_lineup"]]
        a_lineup_ops = np.mean(a_ops_list)
        a_team_ops = team_cum_ops.get(game["away_team"], lg_prior)
        a_gap = a_lineup_ops - a_team_ops

        lineup_gap_diff = h_gap - a_gap

        # Residualize against ops_diff (from season-level team_batting_stats)
        h_team_ops_season = team_ops_map.get(game["home_team"])
        a_team_ops_season = team_ops_map.get(game["away_team"])
        if h_team_ops_season is not None and a_team_ops_season is not None:
            ops_diff = h_team_ops_season - a_team_ops_season
        else:
            ops_diff = 0.0
            log.warning("[shadow-lineup] Missing team OPS for %s vs %s, using ops_diff=0",
                        game["home_team"], game["away_team"])
        predicted = resid_params["coef"] * ops_diff + resid_params["intercept"]
        lineup_gap_resid = lineup_gap_diff - predicted

        result[gpk] = lineup_gap_resid

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Shadow generation (daily scoring)
# ═══════════════════════════════════════════════════════════════════════════════

_LINEUP_MODEL_CACHE = None


def _load_lineup_model():
    """Load the lineup-augmented ATS model artifacts."""
    global _LINEUP_MODEL_CACHE
    if _LINEUP_MODEL_CACHE is not None:
        return _LINEUP_MODEL_CACHE

    prefix = "mlb_ats_lineup"
    paths = {
        "clf":  os.path.join(_SUBMODEL_DIR, f"{prefix}_clf.pkl"),
        "sc":   os.path.join(_SUBMODEL_DIR, f"{prefix}_scaler.pkl"),
        "imp":  os.path.join(_SUBMODEL_DIR, f"{prefix}_imputer.pkl"),
        "cols": os.path.join(_SUBMODEL_DIR, f"{prefix}_feat_cols.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        log.warning("Lineup model artifacts not found — run train_lineup_model() first")
        return None

    with open(paths["clf"], "rb") as f:  clf = pickle.load(f)
    with open(paths["sc"], "rb") as f:   sc = pickle.load(f)
    with open(paths["imp"], "rb") as f:  imp = pickle.load(f)
    with open(paths["cols"], "rb") as f: feat_cols = pickle.load(f)

    _LINEUP_MODEL_CACHE = (clf, sc, imp, feat_cols)
    return _LINEUP_MODEL_CACHE


def _predict_with_lineup_model(
    home_team: str,
    away_team: str,
    game_date: str,
    spread_value: float,
    lineup_gap_resid: float,
) -> float | None:
    """Score a single game with the lineup-augmented ATS model.

    Returns P(home covers) or None.
    """
    import pandas as pd

    arts = _load_lineup_model()
    if arts is None:
        return None
    clf, sc, imp, feat_cols = arts

    from ml_model import _get_feature_matrix, _normalize_team_name

    matrix = _get_feature_matrix("MLB")
    if matrix.empty:
        return None

    ht = _normalize_team_name("MLB", home_team).lower().strip()
    at = _normalize_team_name("MLB", away_team).lower().strip()
    mask = (
        matrix["home_team"].str.lower().str.strip() == ht
    ) & (
        matrix["away_team"].str.lower().str.strip() == at
    )
    if game_date:
        date_str = str(game_date)[:10]
        mask = mask & (matrix["game_date"].astype(str).str[:10] == date_str)

    matched = matrix[mask]
    if matched.empty and game_date:
        team_mask = (
            matrix["home_team"].str.lower().str.strip() == ht
        ) & (
            matrix["away_team"].str.lower().str.strip() == at
        )
        matched = matrix[team_mask]
    if matched.empty:
        return None

    row = matched.sort_values("game_date", ascending=False).iloc[0]

    _sv = spread_value if spread_value is not None else -1.5

    def _val(col: str) -> float:
        if col == "lineup_gap_resid":
            return float(lineup_gap_resid)
        if col == "_spread_value":
            return float(_sv)
        if col == "_spread_abs":
            return abs(float(_sv))
        if col == "_is_fav_spread":
            return 1.0 if _sv < 0 else 0.0
        return row[col] if col in row.index and not pd.isna(row[col]) else np.nan

    X = np.array([[_val(c) for c in feat_cols]], dtype=float)
    X_imp = imp.transform(X)
    X_s = sc.transform(X_imp)
    return float(clf.predict_proba(X_s)[0][1])


def generate_lineup_shadow(db=None, today_str=None):
    """Score today's MLB spread legs with the lineup-augmented model.

    Parallel to ats_ungated: stores individual legs as 1-leg mock_bets,
    source='shadow_lineup'.
    """
    from parlay_builder import get_available_legs

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[shadow-lineup] Generating lineup shadow for {today_str}")

    # 0. Ensure yesterday's batter stats are in the DB
    try:
        n_backfilled = backfill_yesterday_boxscores(today_str)
        if n_backfilled > 0:
            print(f"[shadow-lineup] Backfilled {n_backfilled} boxscores")
    except Exception as exc:
        print(f"[shadow-lineup] Backfill warning: {exc}")

    # 1. Compute lineup features for today
    lineup_resids = compute_lineup_gap_resid_today(today_str)
    if not lineup_resids:
        print("[shadow-lineup] No lineup features available")
        return []

    print(f"[shadow-lineup] {len(lineup_resids)} games with lineup features")

    # 2. Get all MLB spread legs
    all_legs = get_available_legs(
        db, markets=["h2h", "spreads", "totals", "alternate_spreads"]
    )
    mlb_spread_legs = [
        leg for leg in (all_legs or [])
        if leg.get("sport", "").upper() in ("MLB", "BASEBALL")
        and leg.get("market", "") in ("spreads", "alternate_spreads")
    ]
    print(f"[shadow-lineup] {len(mlb_spread_legs)} MLB spread legs")

    if not mlb_spread_legs:
        return []

    # 3. Build home_team → lineup_gap_resid lookup
    #    Cache today's lineup data once for all legs
    lineup_games = _fetch_todays_lineups(today_str)
    # Map: normalized home_team_name → lineup_gap_resid
    team_to_resid = {}
    for game in lineup_games:
        gpk = game["game_pk"]
        resid = lineup_resids.get(gpk)
        if resid is not None:
            team_to_resid[game["home_team"].lower().strip()] = resid

    # 4. Score each leg with lineup model
    scored = []
    for leg in mlb_spread_legs:
        home = leg.get("home_team", "")
        away = leg.get("away_team", "")
        gdate = leg.get("game_date", today_str)

        # Match fixture to lineup data by home team name (substring match)
        matched_resid = None
        home_lower = home.lower().strip()
        for team_name, resid in team_to_resid.items():
            if home_lower in team_name or team_name in home_lower:
                matched_resid = resid
                break

        if matched_resid is None:
            continue

        # Extract spread value
        desc = leg.get("description", "")
        try:
            from ml_model import _extract_spread_value
            sv = _extract_spread_value(desc, leg.get("market", ""))
        except Exception:
            sv = None

        # Flip spread for away picks (model expects home perspective)
        pick = leg.get("pick", "").lower().strip()
        away_lower = leg.get("away_team", "").lower().strip()
        if pick and away_lower and pick == away_lower and sv is not None:
            sv = -sv

        ats_prob = _predict_with_lineup_model(
            home_team=home,
            away_team=away,
            game_date=gdate,
            spread_value=sv,
            lineup_gap_resid=matched_resid,
        )
        if ats_prob is None:
            continue

        # Invert for away picks
        win_prob = (1.0 - ats_prob) if (pick and away_lower and pick == away_lower) else ats_prob

        dec_odds = leg["odds"]
        imp_prob = 1 / dec_odds
        ev = win_prob * (dec_odds - 1) - (1 - win_prob)
        edge = win_prob - imp_prob

        scored.append({
            **leg,
            "win_prob": round(win_prob * 100, 2),
            "ev": round(ev, 4),
            "edge": round(edge * 100, 2),
            "lineup_gap_resid": matched_resid,
            "model_used": "mlb_ats_lineup_v1",
        })

    print(f"[shadow-lineup] {len(scored)} legs scored with lineup model")

    # 4. Store as mock_bets
    if db is not None and scored:
        stored = _store_lineup_shadow(db, scored, today_str)
        print(f"[shadow-lineup] Stored {stored} shadow legs (source=shadow_lineup)")

    return scored


def _store_lineup_shadow(db, scored_legs, today_str):
    """Persist scored legs as 1-leg mock_bets, source='shadow_lineup'."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    run_id = str(uuid.uuid4())
    stored = 0

    for leg in scored_legs:
        mock_id = str(uuid.uuid4())
        desc = leg.get("description", "")
        wp = leg.get("win_prob", 0)
        odds = leg.get("odds", 1.0)
        mkt = leg.get("market", "")
        resid = leg.get("lineup_gap_resid", 0.0)

        bet_info = f"{desc} | wp={wp:.1f} | resid={resid:.4f}"

        # Dedup
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
            "AND source='shadow_lineup' AND bet_info LIKE :bi"
        ), {"gd": today_str, "bi": f"{desc}%"}).scalar()
        if existing > 0:
            continue

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
            model_used="mlb_ats_lineup_v1",
            status="PENDING",
            source="shadow_lineup",
            cohort="clean",
            notes=f"lineup_gap_resid={resid:.4f} mkt={mkt}",
        )
        db.add(mock)

        mock_leg = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=desc,
            market_type=mkt,
            sport="MLB",
            win_prob=wp,
            ev=leg.get("ev", 0),
            fixture_id=leg.get("fixture_id"),
            is_alt_line=(mkt == "alternate_spreads"),
        )
        db.add(mock_leg)
        stored += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[shadow-lineup] Commit error: {exc}")
        return 0

    return stored


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Daily batter stats backfill (incremental)
# ═══════════════════════════════════════════════════════════════════════════════

def backfill_yesterday_boxscores(today_str: str | None = None) -> int:
    """Fetch boxscores for yesterday's games that aren't in batter_game_stats.

    Called daily before lineup shadow scoring to keep cumulative stats current.
    Returns number of games backfilled.
    """
    from datetime import timedelta

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    yesterday = (datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(_HIST_DB)

    # Find yesterday's game_pks from team_game_batting
    cur = conn.execute(
        "SELECT DISTINCT game_pk FROM team_game_batting WHERE game_date = ?",
        (yesterday,)
    )
    yesterday_pks = {r[0] for r in cur.fetchall()}

    # Find which are already in batter_game_stats
    if yesterday_pks:
        placeholders = ",".join("?" * len(yesterday_pks))
        cur = conn.execute(
            f"SELECT DISTINCT game_pk FROM batter_game_stats WHERE game_pk IN ({placeholders})",
            list(yesterday_pks),
        )
        already_fetched = {r[0] for r in cur.fetchall()}
    else:
        already_fetched = set()

    to_fetch = yesterday_pks - already_fetched
    if not to_fetch:
        log.info("[lineup-backfill] Yesterday's %d games already in DB", len(yesterday_pks))
        conn.close()
        return 0

    log.info("[lineup-backfill] Fetching %d/%d boxscores for %s",
             len(to_fetch), len(yesterday_pks), yesterday)

    # Import parse/insert from backfill_lineups
    import sys
    sys.path.insert(0, _BASE_DIR)
    from backfill_lineups import fetch_boxscore, parse_boxscore, insert_rows

    import time
    success = 0
    for gpk in to_fetch:
        data = fetch_boxscore(gpk)
        if data:
            lineup_rows, batter_rows = parse_boxscore(gpk, data)
            if lineup_rows:
                insert_rows(conn, lineup_rows, batter_rows)
                success += 1
        time.sleep(0.12)

    conn.commit()
    conn.close()
    log.info("[lineup-backfill] Done: %d/%d boxscores fetched", success, len(to_fetch))
    return success


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save lineup model")
    parser.add_argument("--shadow", action="store_true", help="Generate today's shadow picks")
    args = parser.parse_args()

    if args.train:
        result = train_lineup_model(save=True)
        print(f"\nResult: {result}")

    elif args.shadow:
        from database import SessionLocal, init_db
        init_db()
        db = SessionLocal()
        try:
            legs = generate_lineup_shadow(db)
            if legs:
                print(f"\nSummary: {len(legs)} legs scored")
                for leg in sorted(legs, key=lambda l: l.get("win_prob", 0), reverse=True)[:10]:
                    wp = leg.get("win_prob", 0)
                    desc = leg.get("description", "")[:50]
                    resid = leg.get("lineup_gap_resid", 0)
                    print(f"  wp={wp:5.1f}% resid={resid:+.4f} {desc}")
        finally:
            db.close()
    else:
        parser.print_help()
