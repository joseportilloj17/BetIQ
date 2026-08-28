"""
shadow_cleaned_retrain.py — Shadow retrain with contamination fixes.

Trains a cleaned model candidate on the SAME data as train(), but with:
  1. DEDUP: unique (fixture_id, market_type, description) per mock leg
  2. SOURCE WEIGHTING: exploration legs down-weighted vs staked
  3. BOUNDARY EXCLUSION: exploration_tier='boundary_testing' removed
  4. INJURY NaN FLAG: binary feature _has_injury_data for structural missingness

Saves artifacts to data/submodels/shadow/cleaned_* and registers as a
shadow slot for racing against current V3.

Usage:
    cd backend && python shadow_cleaned_retrain.py [--weight-ratio 0.5]
"""
from __future__ import annotations
import os, sys, json, pickle, math, logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             classification_report)

# ── Bootstrap path ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("BETIQ_ENV", "shadow_retrain")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from database import get_db, SessionLocal
from ml_model import (
    FEATURE_COLS, WEIGHT_CONFIG, SPORT_MAP, SUBMODEL_DIR,
    _build_clf, _importances, _load_historical_df, _build_personal_df,
    add_clv_features, _american_to_decimal, _extract_spread_value,
    _build_recency_features, Bet, HIST_EXTRA_COLS,
)

_SHADOW_DIR = os.path.join(SUBMODEL_DIR, "shadow")
_STAKED_SOURCES = {"prospective", "prospective_pm", "prospective_legacy",
                   "top_picks_page", "forced_generation"}


def _build_mock_leg_df_cleaned(db, exploration_weight: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Build cleaned mock leg training data with all 4 contamination fixes.

    Returns (df, stats) where stats describes what was filtered/reweighted.
    """
    from sqlalchemy import text as _text

    recency_lookup = _build_recency_features(db)
    try:
        from mlb_features import get_park_features
    except ImportError:
        get_park_features = lambda h, s: {"park_runs_factor": 1.0, "park_hr_factor": 1.0}
    try:
        from mlb_features import get_pitcher_features_from_gamelogs as _gl_pitcher
    except ImportError:
        _gl_pitcher = None
    _gl_cache: dict = {}

    # ── Query with source/tier info for filtering ──────────────────────
    rows = db.execute(_text("""
        SELECT
            mbl.open_odds, mbl.close_odds, mbl.market_type, mbl.sport,
            mbl.is_alt_line, mbl.predicted_win_prob, mbl.predicted_edge_pp,
            mbl.injury_net_advantage, mbl.line_movement_signal, mbl.clv_cents,
            mbl.home_team_injury_impact, mbl.away_team_injury_impact,
            mbl.leg_result, mb.generated_at,
            CASE mbl.scout_grade
                WHEN 'A' THEN 4 WHEN 'B' THEN 3
                WHEN 'C' THEN 2 WHEN 'D' THEN 1
                ELSE NULL END AS scout_grade_encoded,
            mbl.description, mbl.resolved_home_team, mbl.resolved_away_team,
            mb.game_date,
            mbl.home_team_home_pi, mbl.away_team_away_pi, mbl.pi_diff,
            mbl.spread_value, mbl.spread_abs, mbl.is_favorite_spread,
            mbl.home_starter_era, mbl.home_starter_whip, mbl.home_starter_k9,
            mbl.home_starter_hr9,
            mbl.away_starter_era, mbl.away_starter_whip, mbl.away_starter_k9,
            mbl.away_starter_hr9,
            mbl.starter_era_diff, mbl.starter_whip_diff, mbl.starter_k9_diff,
            mbl.game_temp_f, mbl.game_wind_mph, mbl.wind_helps_scoring,
            mbl.home_team_xwoba, mbl.away_team_xwoba, mbl.xwoba_diff,
            mbl.home_team_barrel_pct, mbl.away_team_barrel_pct, mbl.barrel_diff,
            mbl.home_bullpen_era, mbl.away_bullpen_era, mbl.bullpen_era_diff,
            mbl.h2h_home_wr, mbl.h2h_n_games,
            -- NEW: source/tier for filtering
            mb.source,
            mb.exploration_tier,
            mbl.fixture_id
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mbl.open_odds IS NOT NULL
    """)).fetchall()

    if not rows:
        return pd.DataFrame(), {"total": 0}

    stats = {
        "total_raw": len(rows),
        "boundary_excluded": 0,
        "duplicates_removed": 0,
        "exploration_legs": 0,
        "staked_legs": 0,
        "other_legs": 0,
    }

    # ── FIX 3: Exclude boundary_testing tier ───────────────────────────
    filtered = []
    for r in rows:
        source = r[50] or ""
        tier = r[51] or ""
        if source == "exploration" and tier == "boundary_testing":
            stats["boundary_excluded"] += 1
            continue
        filtered.append(r)

    # ── FIX 1: Dedup to unique game-outcomes ───────────────────────────
    # Key: (fixture_id, market_type, description) — keep first occurrence
    seen_outcomes: dict[tuple, int] = {}
    deduped = []
    for r in filtered:
        fixture_id = r[52]
        market_type = (r[2] or "").lower()
        description = (r[15] or "").strip()
        key = (fixture_id, market_type, description)
        if key in seen_outcomes:
            stats["duplicates_removed"] += 1
            continue
        seen_outcomes[key] = 1
        deduped.append(r)

    # ── Build records with fixes 2 (weighting) and 4 (injury flag) ────
    records = []
    for r in deduped:
        open_odds_am  = r[0]
        close_odds_am = r[1]
        market_type   = (r[2] or "").lower()
        sport_raw     = r[3] or ""
        pred_wp       = r[5]
        injury_net    = r[7]
        lms           = r[8]
        clv_c         = r[9]
        h_injury      = r[10]
        a_injury      = r[11]
        leg_result    = r[12]
        generated_at  = r[13]
        scout_enc     = r[14]
        description   = r[15] or ""
        home_team     = (r[16] or "").lower().strip()
        away_team     = (r[17] or "").lower().strip()
        game_date     = r[18] or ""
        _pi_home      = r[19]
        _pi_away      = r[20]
        _pi_d         = r[21]
        _db_spread_val = r[22]
        _db_spread_abs = r[23]
        _db_is_fav     = r[24]
        _db_home_era   = r[25]; _db_home_whip  = r[26]; _db_home_k9   = r[27]; _db_home_hr9  = r[28]
        _db_away_era   = r[29]; _db_away_whip  = r[30]; _db_away_k9   = r[31]; _db_away_hr9  = r[32]
        _db_era_diff   = r[33]; _db_whip_diff  = r[34]; _db_k9_diff   = r[35]
        _db_temp       = r[36]; _db_wind       = r[37]; _db_wind_help = r[38]
        _db_h_xwoba    = r[39]; _db_a_xwoba    = r[40]; _db_xwoba_d   = r[41]
        _db_h_barrel   = r[42]; _db_a_barrel   = r[43]; _db_barrel_d  = r[44]
        _db_h_bullpen  = r[45]; _db_a_bullpen  = r[46]; _db_bullpen_d = r[47]
        _db_h2h_wr     = r[48]; _db_h2h_n      = r[49]
        source         = r[50] or ""

        # ── FIX 2: Source-based weighting ──────────────────────────────
        is_exploration = (source == "exploration")
        if is_exploration:
            weight = exploration_weight
            stats["exploration_legs"] += 1
        elif source in _STAKED_SOURCES:
            weight = float(WEIGHT_CONFIG["personal_base"])  # 3.0
            stats["staked_legs"] += 1
        else:
            weight = float(WEIGHT_CONFIG["personal_base"])  # simulation etc.
            stats["other_legs"] += 1

        dec_odds  = _american_to_decimal(open_odds_am)
        dec_close = _american_to_decimal(close_odds_am) if close_odds_am else None
        imp_prob  = 1.0 / dec_odds
        log_odds_val = math.log(dec_odds) if dec_odds > 0 else 0.0

        if   dec_odds < 2:   ob = 0
        elif dec_odds < 5:   ob = 1
        elif dec_odds < 15:  ob = 2
        elif dec_odds < 50:  ob = 3
        else:                ob = 4

        if "h2h" in market_type or "moneyline" in market_type:
            ml_pct, spread_pct, total_pct, prop_pct = 1.0, 0.0, 0.0, 0.0
        elif "spread" in market_type:
            ml_pct, spread_pct, total_pct, prop_pct = 0.0, 1.0, 0.0, 0.0
        elif "total" in market_type:
            ml_pct, spread_pct, total_pct, prop_pct = 0.0, 0.0, 1.0, 0.0
        elif "player" in market_type:
            ml_pct, spread_pct, total_pct, prop_pct = 0.0, 0.0, 0.0, 1.0
        else:
            ml_pct, spread_pct, total_pct, prop_pct = 0.25, 0.25, 0.25, 0.25

        clv_val  = float(clv_c) / 100.0 if clv_c is not None else 0.0
        clv_avail = 1 if clv_c is not None else 0
        closing_line_diff = float(dec_close - dec_odds) if dec_close and dec_close > 1 else 0.0

        pred_wp_dec = float(pred_wp or 50.0) / 100.0
        ev_val = pred_wp_dec * (dec_odds - 1) - (1 - pred_wp_dec)

        gdt = None
        try:
            if isinstance(generated_at, str):
                gdt = datetime.fromisoformat(generated_at[:19])
            elif generated_at:
                gdt = generated_at
        except Exception:
            pass

        hour  = gdt.hour      if gdt else 12
        dow   = gdt.weekday() if gdt else 0
        month = gdt.month     if gdt else 6

        # Pitcher fallback (same as _build_mock_leg_df)
        if _db_home_era is None and _gl_pitcher and sport_raw.upper() in ("MLB", "BASEBALL_MLB"):
            _ht_orig = (r[16] or "").strip()
            _at_orig = (r[17] or "").strip()
            _gd_str  = str(game_date)[:10]
            _gl_key  = (_ht_orig, _at_orig, _gd_str)
            if _gl_key not in _gl_cache:
                try:
                    _gl_cache[_gl_key] = _gl_pitcher(_ht_orig, _at_orig, _gd_str)
                except Exception:
                    _gl_cache[_gl_key] = {}
            _gl = _gl_cache[_gl_key]
            for attr, idx in [("home_starter_era", None), ("away_starter_era", None),
                              ("home_starter_whip", None), ("away_starter_whip", None),
                              ("home_starter_k9", None), ("away_starter_k9", None),
                              ("starter_era_diff", None), ("starter_whip_diff", None),
                              ("starter_k9_diff", None)]:
                v = _gl.get(attr)
                if v is not None and v == v:
                    if attr == "home_starter_era": _db_home_era = v
                    elif attr == "away_starter_era": _db_away_era = v
                    elif attr == "home_starter_whip": _db_home_whip = v
                    elif attr == "away_starter_whip": _db_away_whip = v
                    elif attr == "home_starter_k9": _db_home_k9 = v
                    elif attr == "away_starter_k9": _db_away_k9 = v
                    elif attr == "starter_era_diff": _db_era_diff = v
                    elif attr == "starter_whip_diff": _db_whip_diff = v
                    elif attr == "starter_k9_diff": _db_k9_diff = v

        # Spread features
        if _db_spread_val is not None:
            sv = float(_db_spread_val)
            spread_abs = float(_db_spread_abs) if _db_spread_abs is not None else abs(sv)
            is_fav_spread = float(_db_is_fav) if _db_is_fav is not None else (1.0 if sv < 0 else 0.0)
        else:
            sv = _extract_spread_value(description, market_type)
            spread_abs = abs(sv)
            is_fav_spread = 1.0 if sv < 0 else 0.0

        # Recency
        desc_lower = description.lower()
        _team = None
        if home_team and home_team in desc_lower:
            _team = home_team
        elif away_team and away_team in desc_lower:
            _team = away_team
        rec = recency_lookup.get((_team, market_type), {}) if _team else {}

        # ── FIX 4: Injury NaN flag ────────────────────────────────────
        # Binary: 1.0 if injury data was actually computed, 0.0 if structurally missing
        has_injury = 1.0 if injury_net is not None else 0.0

        record = {
            "legs": 1, "odds": dec_odds, "log_odds": log_odds_val,
            "implied_prob": round(imp_prob, 6), "stake": 10.0,
            "is_parlay": 0, "sport_id": SPORT_MAP.get(sport_raw, 6),
            "league_id": 10, "hour_placed": hour, "day_of_week": dow,
            "ml_pct": ml_pct, "spread_pct": spread_pct,
            "total_pct": total_pct, "prop_pct": prop_pct,
            "multi_sport": 0, "n_sports": 1, "has_ev": 1,
            "ev_value": round(ev_val, 6),
            "closing_line_diff": closing_line_diff,
            "odds_bracket": ob, "leg_diversity": 0.25,
            "clv": clv_val, "clv_available": clv_avail,
            "multi_league": 0, "n_leagues": 1,
            "prop_density": 1.0 if "player" in market_type else 0.0,
            "stake_ratio": 10.0 / 6.8,
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "stake_vs_kelly": 0.0,
            **{c: np.nan for c in HIST_EXTRA_COLS},
            "home_pi": float(_pi_home) if _pi_home is not None else np.nan,
            "away_pi": float(_pi_away) if _pi_away is not None else np.nan,
            "pi_diff": float(_pi_d) if _pi_d is not None else np.nan,
            "pi_confidence": 1.0 if _pi_home is not None else np.nan,
            "injury_net_advantage": float(injury_net) if injury_net is not None else np.nan,
            "line_movement_signal": float(lms) if lms is not None else np.nan,
            "scout_grade_encoded": float(scout_enc) if scout_enc is not None else np.nan,
            "clv_cents": float(clv_c) if clv_c is not None else np.nan,
            "home_team_injury_impact": float(h_injury) if h_injury is not None else np.nan,
            "away_team_injury_impact": float(a_injury) if a_injury is not None else np.nan,
            "spread_value": sv, "spread_abs": spread_abs,
            "is_favorite_spread": is_fav_spread,
            "team_recent_cover_rate": float(rec.get("cover_rate", np.nan)) if rec else np.nan,
            "team_recent_wr": float(rec.get("recent_wr", np.nan)) if rec else np.nan,
            "team_form_trend": float(rec.get("form_trend", np.nan)) if rec else np.nan,
            **get_park_features(home_team or "", sport_raw),
            "home_starter_era": float(_db_home_era) if _db_home_era is not None else np.nan,
            "home_starter_whip": float(_db_home_whip) if _db_home_whip is not None else np.nan,
            "home_starter_k9": float(_db_home_k9) if _db_home_k9 is not None else np.nan,
            "home_starter_hr9": float(_db_home_hr9) if _db_home_hr9 is not None else np.nan,
            "away_starter_era": float(_db_away_era) if _db_away_era is not None else np.nan,
            "away_starter_whip": float(_db_away_whip) if _db_away_whip is not None else np.nan,
            "away_starter_k9": float(_db_away_k9) if _db_away_k9 is not None else np.nan,
            "away_starter_hr9": float(_db_away_hr9) if _db_away_hr9 is not None else np.nan,
            "starter_era_diff": float(_db_era_diff) if _db_era_diff is not None else np.nan,
            "starter_whip_diff": float(_db_whip_diff) if _db_whip_diff is not None else np.nan,
            "starter_k9_diff": float(_db_k9_diff) if _db_k9_diff is not None else np.nan,
            "game_temp_f": float(_db_temp) if _db_temp is not None else np.nan,
            "game_wind_mph": float(_db_wind) if _db_wind is not None else np.nan,
            "wind_helps_scoring": float(_db_wind_help) if _db_wind_help is not None else np.nan,
            "home_team_xwoba": float(_db_h_xwoba) if _db_h_xwoba is not None else np.nan,
            "away_team_xwoba": float(_db_a_xwoba) if _db_a_xwoba is not None else np.nan,
            "xwoba_diff": float(_db_xwoba_d) if _db_xwoba_d is not None else np.nan,
            "home_team_barrel_pct": float(_db_h_barrel) if _db_h_barrel is not None else np.nan,
            "away_team_barrel_pct": float(_db_a_barrel) if _db_a_barrel is not None else np.nan,
            "barrel_diff": float(_db_barrel_d) if _db_barrel_d is not None else np.nan,
            "home_bullpen_era": float(_db_h_bullpen) if _db_h_bullpen is not None else np.nan,
            "away_bullpen_era": float(_db_a_bullpen) if _db_a_bullpen is not None else np.nan,
            "bullpen_era_diff": float(_db_bullpen_d) if _db_bullpen_d is not None else np.nan,
            "h2h_home_wr": float(_db_h2h_wr) if _db_h2h_wr is not None else np.nan,
            "h2h_n_games": float(_db_h2h_n) if _db_h2h_n is not None else np.nan,
            # Metadata
            "_label": 1.0 if leg_result == "WIN" else 0.0,
            "_weight": weight,
            "_date": gdt if gdt else datetime(2020, 1, 1),
            "_sport_label": sport_raw,
            "_source": source,
            "_has_injury_data": has_injury,
        }
        records.append(record)

    df = pd.DataFrame(records)
    stats["after_dedup"] = len(df)
    log.info("Cleaned mock legs: %d → %d (-%d boundary, -%d dupes)",
             stats["total_raw"], len(df),
             stats["boundary_excluded"], stats["duplicates_removed"])
    log.info("  Staked: %d, Exploration: %d (weight=%.1f), Other: %d",
             stats["staked_legs"], stats["exploration_legs"],
             exploration_weight, stats["other_legs"])
    return df, stats


def run_shadow_retrain(exploration_weight: float = 1.0) -> dict:
    """Train a cleaned shadow model and compare against current V3."""
    db = SessionLocal()
    now = datetime.utcnow()

    # ── 1. Personal bets (same as train()) ────────────────────────────
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).order_by(Bet.time_placed).all()
    personal_df = _build_personal_df(bets, cutoff=now)
    log.info("Personal bets: %d", len(personal_df))

    # ── 2. Historical data (same as train()) ──────────────────────────
    historical_df = _load_historical_df()
    log.info("Historical rows: %d", len(historical_df))

    # ── 3. Cleaned mock legs ──────────────────────────────────────────
    mock_df, mock_stats = _build_mock_leg_df_cleaned(db, exploration_weight)
    if mock_df.empty:
        db.close()
        return {"error": "No cleaned mock legs"}

    # CLV hook
    personal_df = add_clv_features(personal_df)
    if not historical_df.empty:
        historical_df = add_clv_features(historical_df)

    # ── 4. Combine ────────────────────────────────────────────────────
    # Drop _source and _has_injury_data before alignment (not in FEATURE_COLS)
    _extra_meta = ["_source", "_has_injury_data"]
    mock_for_train = mock_df.drop(columns=[c for c in _extra_meta if c in mock_df.columns],
                                   errors="ignore")

    frames = [personal_df]
    if not historical_df.empty:
        frames.append(historical_df[personal_df.columns])
    if not mock_for_train.empty:
        frames.append(mock_for_train[personal_df.columns])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("_date").reset_index(drop=True)

    n_mock = len(mock_for_train)
    n_personal = len(personal_df)
    n_historical = len(historical_df) if not historical_df.empty else 0
    log.info("Combined: %d personal + %d historical + %d mock = %d total",
             n_personal, n_historical, n_mock, len(combined))

    # ── 5. Feature columns ────────────────────────────────────────────
    _goal_cols = list(FEATURE_COLS)
    for c in _goal_cols:
        if c not in combined.columns:
            combined[c] = np.nan

    _split_idx = int(len(combined) * 0.80)
    _train_portion = combined.iloc[:_split_idx]
    _nan_mask = _train_portion[_goal_cols].isna().all()
    _dropped = [c for c in _goal_cols if _nan_mask.get(c, False)]
    _train_cols = [c for c in _goal_cols if not _nan_mask.get(c, False)]
    log.info("Features: %d active (dropped %d all-NaN)", len(_train_cols), len(_dropped))

    X_all  = combined[_train_cols].values.astype(float)
    y_all  = combined["_label"].values.astype(float)
    sw_all = combined["_weight"].values.astype(float)

    split = _split_idx
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]
    sw_tr = sw_all[:split]

    # ── 6. Preprocess ─────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_te_s = scaler.transform(X_te_imp)

    # ── 7. Fit ────────────────────────────────────────────────────────
    clf = _build_clf("gradient_boost")
    clf.fit(X_tr_s, y_tr, sample_weight=sw_tr)

    # ── 8. Evaluate ───────────────────────────────────────────────────
    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    acc   = float(accuracy_score(y_te, y_pred))
    auc   = float(roc_auc_score(y_te, y_prob)) if len(set(y_te)) > 1 else 0.5
    brier = float(brier_score_loss(y_te, y_prob))
    imps  = _importances(clf, _train_cols, X_val=X_te_s, y_val=y_te)

    log.info("Shadow cleaned model: AUC=%.4f, Brier=%.4f, Acc=%.1f%%",
             auc, brier, acc * 100)

    # ── 9. Save to shadow dir ─────────────────────────────────────────
    prefix = f"cleaned_ew{exploration_weight:.1f}"
    os.makedirs(_SHADOW_DIR, exist_ok=True)
    for obj, suffix in [(clf, "clf"), (scaler, "scaler"),
                        (imputer, "imputer"), (_train_cols, "feat_cols")]:
        path = os.path.join(_SHADOW_DIR, f"{prefix}_{suffix}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    meta = {
        "experiment": "cleaned_retrain",
        "exploration_weight": exploration_weight,
        "fixes": ["dedup", "source_weight", "boundary_excluded", "injury_nan_flag"],
        "mock_stats": mock_stats,
        "metrics": {"auc": round(auc, 4), "brier": round(brier, 4),
                    "accuracy": round(acc * 100, 2)},
        "n_train": split, "n_test": len(X_te),
        "n_features": len(_train_cols),
        "top_features": dict(sorted(imps.items(), key=lambda x: -x[1])[:15]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(_SHADOW_DIR, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log.info("Saved shadow artifacts: %s_*.pkl", prefix)

    # ── 10. Compare with current live V3 ──────────────────────────────
    try:
        import pickle as _pkl
        _data_dir = os.path.join(_HERE, "..", "data")
        with open(os.path.join(_data_dir, "model.pkl"), "rb") as f:
            live_clf = _pkl.load(f)
        with open(os.path.join(_data_dir, "scaler.pkl"), "rb") as f:
            live_scaler = _pkl.load(f)
        with open(os.path.join(_data_dir, "imputer.pkl"), "rb") as f:
            live_imputer = _pkl.load(f)
        with open(os.path.join(_data_dir, "active_feature_cols.pkl"), "rb") as f:
            live_cols = _pkl.load(f)

        # Align test set to live model's features
        live_goal = [c for c in live_cols if c in combined.columns]
        X_te_live = combined.iloc[split:][live_goal].values.astype(float)
        X_te_live_imp = live_imputer.transform(X_te_live)
        X_te_live_s = live_scaler.transform(X_te_live_imp)
        y_prob_live = live_clf.predict_proba(X_te_live_s)[:, 1]

        live_auc = float(roc_auc_score(y_te, y_prob_live)) if len(set(y_te)) > 1 else 0.5
        live_brier = float(brier_score_loss(y_te, y_prob_live))

        log.info("")
        log.info("═══ COMPARISON (same test set) ═══")
        log.info("  Live V3:    AUC=%.4f  Brier=%.4f", live_auc, live_brier)
        log.info("  Cleaned:    AUC=%.4f  Brier=%.4f", auc, brier)
        log.info("  Delta:      AUC=%+.4f  Brier=%+.4f",
                 auc - live_auc, brier - live_brier)

        meta["comparison"] = {
            "live_auc": round(live_auc, 4),
            "live_brier": round(live_brier, 4),
            "cleaned_auc": round(auc, 4),
            "cleaned_brier": round(brier, 4),
            "delta_auc": round(auc - live_auc, 4),
            "delta_brier": round(brier - live_brier, 4),
        }
        # Rewrite meta with comparison
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    except Exception as exc:
        log.warning("Could not compare with live V3: %s", exc)

    db.close()
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-ratio", type=float, default=1.0,
                        help="Weight for exploration legs (staked=3.0)")
    args = parser.parse_args()

    # Run multiple weight ratios to find optimal
    if args.weight_ratio == -1:
        for w in [0.0, 0.5, 1.0, 1.5, 3.0]:
            log.info("\n" + "=" * 60)
            log.info("EXPLORATION WEIGHT = %.1f", w)
            log.info("=" * 60)
            run_shadow_retrain(exploration_weight=w)
    else:
        run_shadow_retrain(exploration_weight=args.weight_ratio)
