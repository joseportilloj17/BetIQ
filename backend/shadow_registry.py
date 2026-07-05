"""
shadow_registry.py — Shadow model + config scoring and comparison harness.

Shadow models score the live slate in parallel, write to shadow_predictions,
bet nothing, and are excluded from all live metrics.  The active model and
live pipeline are NEVER touched by any code in this module.

Two shadow types:
  MODEL shadows  — swap the ML model (clf/scaler/imputer), score each game×spread
  CONFIG shadows — same V3 model, different post-processing (calibration, CLV,
                   per-rung adjustments). A config shadow applies a transform_fn
                   to the base V3 prediction: transform_fn(base_wp, game_row,
                   spread_value, ctx) → adjusted_wp.

Both types write to the SAME shadow_predictions table, settle identically,
and are evaluated by the same tripwire.  Settlement, calibration, and
comparison are slot-agnostic — they just see shadow_slot names.

Public API
──────────
  register_shadow(slot_name, ...) → register a model shadow slot
  register_shadow_config(slot_name, ...) → register a config shadow slot
  run_shadow_scoring(db) → score today's slate with all registered shadows
  run_shadow_config_scoring(db) → score today's slate with all config shadows
  settle_shadow_predictions(db) → update shadow preds with actual outcomes
  refresh_shadow_calibration(db) → per-shadow self-cal on settled only
  compare_shadows_vs_active(db) → tripwire comparator (recommendation-only)
"""
from __future__ import annotations
import json, logging, os, pickle, sqlite3
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_DIR, "..", "data")
_SHADOW_SLOTS_PATH = os.path.join(_DATA_DIR, "shadow_slots.json")
_DB_PATH = os.path.join(_DATA_DIR, "bets.db")
TRIPWIRE_MIN_LEGS = 100
TRIPWIRE_MIN_DAYS = 7

# ── Schema ───────────────────────────────────────────────────────────────
_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shadow_slot     TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    game_date       TEXT    NOT NULL,
    sport           TEXT    NOT NULL,
    home_team       TEXT    NOT NULL,
    away_team       TEXT    NOT NULL,
    market_type     TEXT    NOT NULL,
    spread_value    REAL,
    predicted_wp    REAL    NOT NULL,
    features_hash   TEXT,
    scored_at       TEXT    NOT NULL,
    actual_result   TEXT,
    settled_at      TEXT,
    UNIQUE(shadow_slot, game_date, home_team, away_team, market_type, spread_value)
)"""
_CREATE_CALIBRATION = """
CREATE TABLE IF NOT EXISTS shadow_calibration (
    shadow_slot     TEXT NOT NULL,
    sport           TEXT NOT NULL,
    market_type     TEXT NOT NULL,
    wp_bucket       TEXT NOT NULL,
    avg_predicted   REAL,
    avg_actual      REAL,
    haircut_pp      REAL,
    sample_size     INTEGER,
    last_updated    TEXT NOT NULL,
    PRIMARY KEY (shadow_slot, sport, market_type, wp_bucket)
)"""

def _ensure_shadow_tables(con):
    con.execute(_CREATE_PREDICTIONS)
    con.execute(_CREATE_CALIBRATION)
    # Migration: add 'selected' column for config-shadow selection tracking
    try:
        con.execute("ALTER TABLE shadow_predictions ADD COLUMN selected INTEGER DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # column already exists
    # Migration: add 'pick_side' column for both-sides selection tracking
    # 'home' = bet is on home team covering sv (default, legacy)
    # 'away' = bet is on away team at +|sv| (wp and result must be inverted)
    try:
        con.execute("ALTER TABLE shadow_predictions ADD COLUMN pick_side TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # column already exists
    con.commit()

def _get_con():
    con = sqlite3.connect(_DB_PATH)
    _ensure_shadow_tables(con)
    return con

# ── Slot registry (JSON) ────────────────────────────────────────────────
def _load_slots():
    if os.path.exists(_SHADOW_SLOTS_PATH):
        with open(_SHADOW_SLOTS_PATH) as f:
            return json.load(f)
    return {"slots": {}}

def _save_slots(data):
    with open(_SHADOW_SLOTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def register_shadow(slot_name, clf_path, scaler_path, imputer_path,
                    feat_cols_path, meta=None):
    for p in [clf_path, scaler_path, imputer_path, feat_cols_path]:
        full = os.path.join(_DATA_DIR, p) if not os.path.isabs(p) else p
        if not os.path.exists(full):
            raise FileNotFoundError(f"Shadow artifact not found: {full}")
    slots = _load_slots()
    slots["slots"][slot_name] = {
        "clf_path": clf_path, "scaler_path": scaler_path,
        "imputer_path": imputer_path, "feat_cols_path": feat_cols_path,
        "meta": meta or {}, "registered_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    }
    _save_slots(slots)
    log.info("[shadow] registered slot '%s'", slot_name)
    return slots["slots"][slot_name]

def list_shadows():
    return _load_slots().get("slots", {})

def _load_shadow_artifacts(slot):
    def _load(rel_path):
        full = os.path.join(_DATA_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
        with open(full, "rb") as f:
            return pickle.load(f)
    return _load(slot["clf_path"]), _load(slot["scaler_path"]), \
           _load(slot["imputer_path"]), _load(slot["feat_cols_path"])

# ── Shadow scoring pass ─────────────────────────────────────────────────
def _predict_shadow(clf, sc, imp, feat_cols, row, spread_value,
                    total_line=None):
    """Predict win/over prob for a shadow model.

    For ATS models, pass spread_value.
    For totals models, pass total_line (and spread_value is ignored).
    """
    # Pre-compute line_over_base_rate for totals (league avg over rates)
    _OVER_BASE = {6.5: 0.78, 7.5: 0.63, 8.5: 0.49, 9.5: 0.37,
                  10.5: 0.25, 11.5: 0.15}

    def _val(col):
        # ATS features
        if col == "_spread_value":
            return float(spread_value) if spread_value is not None else np.nan
        if col == "_spread_abs":
            return abs(float(spread_value)) if spread_value is not None else np.nan
        if col == "_spread_sq":
            return float(spread_value) ** 2 if spread_value is not None else np.nan
        if col == "_is_fav_spread":
            return 1.0 if (spread_value is not None and spread_value < 0) else 0.0
        # Totals features
        if col == "_total_line":
            return float(total_line) if total_line is not None else np.nan
        if col == "_total_line_abs":
            return abs(float(total_line)) if total_line is not None else np.nan
        if col == "_is_high_total":
            return 1.0 if (total_line is not None and total_line >= 9.5) else 0.0
        if col == "_line_over_base_rate":
            return _OVER_BASE.get(total_line, 0.45) if total_line is not None else np.nan
        if col == "ot_so_game":
            return 0.0
        return row[col] if col in row.index and not pd.isna(row[col]) else np.nan
    try:
        X = np.array([[_val(c) for c in feat_cols]], dtype=float)
        X_imp = imp.transform(X)
        X_s = sc.transform(X_imp)
        # PerLineCorrectedClassifier accepts total_lines kwarg
        if total_line is not None and hasattr(clf, 'corrections'):
            return float(clf.predict_proba(X_s, total_lines=np.array([total_line]))[0][1])
        return float(clf.predict_proba(X_s)[0][1])
    except Exception as exc:
        log.debug("[shadow] prediction failed: %s", exc)
        return None

def run_shadow_scoring(db=None):
    from ml_model import _get_feature_matrix, _canonical_sport
    slots = _load_slots().get("slots", {})
    active_slots = {k: v for k, v in slots.items() if v.get("active", True)}
    if not active_slots:
        return {"status": "no_slots", "scored": 0}
    today = datetime.utcnow().strftime("%Y-%m-%d")
    con = _get_con()
    total_scored = 0
    results = {}
    sport = "MLB"
    try:
        matrix = _get_feature_matrix(_canonical_sport(sport))
    except Exception as exc:
        con.close()
        return {"status": "matrix_error", "error": str(exc)}
    if matrix.empty:
        con.close()
        return {"status": "empty_matrix", "scored": 0}
    today_games = matrix[matrix["game_date"].astype(str).str[:10] == today]
    if today_games.empty:
        today_games = matrix[matrix["game_date"].astype(str).str[:10] >= today].head(50)
    if today_games.empty:
        con.close()
        return {"status": "no_games_today", "scored": 0}
    # Spread universe covers actual MLB alt-spread bet range (live ALE sees ±1.5 to ±6.5)
    spreads = [-2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    total_lines = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    for slot_name, slot_cfg in active_slots.items():
        slot_scored = 0
        try:
            clf, sc, imp, feat_cols = _load_shadow_artifacts(slot_cfg)
        except Exception as exc:
            results[slot_name] = {"error": str(exc)}
            continue
        if "_spread_sq" in feat_cols:
            results[slot_name] = {"skipped": "spread_sq_guard"}
            continue
        # Detect totals vs ATS model by feature list
        is_totals = "_total_line" in feat_cols
        model_version = slot_cfg.get("meta", {}).get("version", slot_name)
        for _, row in today_games.iterrows():
            ht = row.get("home_team", "")
            at = row.get("away_team", "")
            gdate = str(row.get("game_date", today))[:10]
            if is_totals:
                for tl in total_lines:
                    wp = _predict_shadow(clf, sc, imp, feat_cols, row,
                                         spread_value=None, total_line=tl)
                    if wp is None:
                        continue
                    fhash = "|".join(f"{c}={row.get(c)}" for c in feat_cols[:5])
                    try:
                        con.execute("""
                            INSERT INTO shadow_predictions
                                (shadow_slot, model_version, game_date, sport,
                                 home_team, away_team, market_type, spread_value,
                                 predicted_wp, features_hash, scored_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(shadow_slot, game_date, home_team, away_team,
                                        market_type, spread_value)
                            DO UPDATE SET predicted_wp=excluded.predicted_wp,
                                          features_hash=excluded.features_hash,
                                          scored_at=excluded.scored_at
                        """, (slot_name, model_version, gdate, sport, ht, at,
                              "totals", tl, round(wp, 6), fhash,
                              datetime.now(timezone.utc).isoformat()))
                        slot_scored += 1
                    except Exception as exc:
                        log.debug("[shadow] insert error: %s", exc)
            else:
                for sv in spreads:
                    wp = _predict_shadow(clf, sc, imp, feat_cols, row, sv)
                    if wp is None:
                        continue
                    fhash = "|".join(f"{c}={row.get(c)}" for c in feat_cols[:5])
                    try:
                        con.execute("""
                            INSERT INTO shadow_predictions
                                (shadow_slot, model_version, game_date, sport,
                                 home_team, away_team, market_type, spread_value,
                                 predicted_wp, features_hash, scored_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(shadow_slot, game_date, home_team, away_team,
                                        market_type, spread_value)
                            DO UPDATE SET predicted_wp=excluded.predicted_wp,
                                          features_hash=excluded.features_hash,
                                          scored_at=excluded.scored_at
                        """, (slot_name, model_version, gdate, sport, ht, at,
                              "spreads", sv, round(wp, 6), fhash,
                              datetime.now(timezone.utc).isoformat()))
                        slot_scored += 1
                    except Exception as exc:
                        log.debug("[shadow] insert error: %s", exc)
        con.commit()
        results[slot_name] = {"scored": slot_scored, "type": "totals" if is_totals else "ats"}
        total_scored += slot_scored
    con.close()
    return {"status": "ok", "scored": total_scored, "slots": results, "date": today}

# ── Config shadows ──────────────────────────────────────────────────────
# A config shadow uses the ACTIVE V3 model but applies a different
# post-processing path.  Each config provides a transform_fn:
#   transform_fn(base_wp: float, row: pd.Series, spread_value: float,
#                ctx: dict) → float | None
# where ctx holds pre-loaded context (CLV lookup, per-rung WR table, etc.)
# The base_wp is the active V3 model's raw output (0.0–1.0).

_SHADOW_CONFIG_SLOTS_PATH = os.path.join(_DATA_DIR, "shadow_config_slots.json")

# In-memory registry of transform functions (keyed by slot_name).
# Populated by register_shadow_config() at import time or startup.
_config_transforms: dict[str, callable] = {}

def _load_config_slots():
    if os.path.exists(_SHADOW_CONFIG_SLOTS_PATH):
        with open(_SHADOW_CONFIG_SLOTS_PATH) as f:
            return json.load(f)
    return {"slots": {}}

def _save_config_slots(data):
    with open(_SHADOW_CONFIG_SLOTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def register_shadow_config(slot_name, transform_fn, meta=None):
    """Register a config-level shadow slot.

    Parameters
    ----------
    slot_name : str — unique name (e.g. "config_d1_v3_alts")
    transform_fn : callable(base_wp, row, spread_value, ctx) → float|None
    meta : optional dict of metadata
    """
    _config_transforms[slot_name] = transform_fn
    slots = _load_config_slots()
    existing = slots["slots"].get(slot_name)
    if existing:
        # Preserve active state — don't re-activate a deactivated config
        existing["meta"] = meta or existing.get("meta", {})
        existing["transform_fn_registered"] = True
    else:
        slots["slots"][slot_name] = {
            "type": "config",
            "meta": meta or {},
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "active": True,
        }
    _save_config_slots(slots)
    log.info("[shadow-config] registered slot '%s'", slot_name)
    return slots["slots"][slot_name]

def list_shadow_configs():
    return _load_config_slots().get("slots", {})

def _get_base_v3_prediction(clf, sc, imp, feat_cols, row, spread_value):
    """Get the active V3 model prediction for a game/spread.

    Returns the raw probability (0.0-1.0) or None on failure.
    This is the BASE that config transforms adjust.
    """
    return _predict_shadow(clf, sc, imp, feat_cols, row, spread_value)

def run_shadow_config_scoring(db=None):
    """Score today's slate with all registered config shadows.

    Uses the ACTIVE V3 model as the base, then applies each config's
    transform_fn to produce adjusted predictions.
    """
    from ml_model import _get_feature_matrix, _canonical_sport
    config_slots = _load_config_slots().get("slots", {})
    active_configs = {k: v for k, v in config_slots.items()
                      if v.get("active", True) and k in _config_transforms}
    if not active_configs:
        return {"status": "no_config_slots", "scored": 0}

    today = datetime.utcnow().strftime("%Y-%m-%d")
    con = _get_con()
    sport = "MLB"

    try:
        matrix = _get_feature_matrix(_canonical_sport(sport))
    except Exception as exc:
        con.close()
        return {"status": "matrix_error", "error": str(exc)}
    if matrix.empty:
        con.close()
        return {"status": "empty_matrix", "scored": 0}

    today_games = matrix[matrix["game_date"].astype(str).str[:10] == today]
    if today_games.empty:
        today_games = matrix[matrix["game_date"].astype(str).str[:10] >= today].head(50)
    if today_games.empty:
        con.close()
        return {"status": "no_games_today", "scored": 0}

    # Load the active V3 model (same one used for model-shadow "shadow_v3")
    model_slots = _load_slots().get("slots", {})
    v3_slot = model_slots.get("shadow_v3")
    if not v3_slot:
        con.close()
        return {"status": "no_v3_base", "error": "shadow_v3 slot not found"}
    try:
        clf, sc, imp, feat_cols = _load_shadow_artifacts(v3_slot)
    except Exception as exc:
        con.close()
        return {"status": "v3_load_error", "error": str(exc)}

    # Spread universe covers actual MLB alt-spread bet range (live ALE sees ±1.5 to ±6.5)
    spreads = [-2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    total_scored = 0
    results = {}

    for config_name, config_cfg in active_configs.items():
        slot_scored = 0
        transform_fn = _config_transforms[config_name]
        model_version = config_cfg.get("meta", {}).get("version", config_name)

        # Build context for the transform function (loaded once per config)
        ctx = _build_config_context(config_name, con)

        for _, row in today_games.iterrows():
            ht = row.get("home_team", "")
            at = row.get("away_team", "")
            gdate = str(row.get("game_date", today))[:10]

            for sv in spreads:
                base_wp = _get_base_v3_prediction(clf, sc, imp, feat_cols, row, sv)
                if base_wp is None:
                    continue
                try:
                    adjusted_wp = transform_fn(base_wp, row, sv, ctx)
                except Exception as exc:
                    log.debug("[shadow-config] %s transform error: %s", config_name, exc)
                    adjusted_wp = None
                if adjusted_wp is None:
                    continue
                adjusted_wp = max(0.001, min(0.999, adjusted_wp))

                try:
                    con.execute("""
                        INSERT INTO shadow_predictions
                            (shadow_slot, model_version, game_date, sport,
                             home_team, away_team, market_type, spread_value,
                             predicted_wp, features_hash, scored_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(shadow_slot, game_date, home_team, away_team,
                                    market_type, spread_value)
                        DO UPDATE SET predicted_wp=excluded.predicted_wp,
                                      features_hash=excluded.features_hash,
                                      scored_at=excluded.scored_at
                    """, (config_name, model_version, gdate, sport, ht, at,
                          "spreads", sv, round(adjusted_wp, 6),
                          f"base={round(base_wp,4)}",
                          datetime.now(timezone.utc).isoformat()))
                    slot_scored += 1
                except Exception as exc:
                    log.debug("[shadow-config] insert error: %s", exc)

        con.commit()
        results[config_name] = {"scored": slot_scored}
        total_scored += slot_scored

    con.close()
    return {"status": "ok", "scored": total_scored, "configs": results, "date": today}


def _build_config_context(config_name, con):
    """Pre-load any data the config's transform_fn needs.

    Each config can stash lookup tables, calibration arrays, etc.
    This is called once per scoring run, NOT per-leg.
    """
    ctx = {}

    # CLV lookup: game_date + teams → clv_cents (for D2)
    if "clv" in config_name:
        try:
            bets_con = sqlite3.connect(_DB_PATH)
            clv_rows = bets_con.execute("""
                SELECT mbl.resolved_home_team, mbl.resolved_away_team,
                       mb.game_date, mbl.spread_value, mbl.clv_cents
                FROM mock_bet_legs mbl
                JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
                WHERE mbl.clv_cents IS NOT NULL
                  AND mbl.leg_result IN ('WIN','LOSS')
                  AND mbl.sport = 'MLB'
                  AND mbl.market_type IN ('alternate_spreads','spreads')
            """).fetchall()
            # Build per-spread CLV avg
            from collections import defaultdict
            clv_by_spread = defaultdict(list)
            for ht, at, gd, sv, clv in clv_rows:
                if sv is not None:
                    clv_by_spread[round(sv, 1)].append(clv)
            ctx["clv_avg_by_spread"] = {
                sv: sum(vs)/len(vs) for sv, vs in clv_by_spread.items() if vs
            }
            # Overall CLV signal direction
            all_clv = [c for _, _, _, _, c in clv_rows if c is not None]
            ctx["clv_overall_avg"] = sum(all_clv) / len(all_clv) if all_clv else 0.0
            bets_con.close()
        except Exception:
            ctx["clv_avg_by_spread"] = {}
            ctx["clv_overall_avg"] = 0.0

    # Per-rung WR lookup (for D3)
    if "rung" in config_name:
        try:
            bets_con = sqlite3.connect(_DB_PATH)
            rung_rows = bets_con.execute("""
                SELECT line_delta,
                       count(*) as n,
                       avg(CASE WHEN leg_result='WIN' THEN 1.0 ELSE 0.0 END) as wr
                FROM mock_bet_legs
                WHERE line_delta IS NOT NULL
                  AND leg_result IN ('WIN','LOSS')
                  AND sport = 'MLB'
                GROUP BY line_delta
                HAVING n >= 10
            """).fetchall()
            ctx["rung_wr"] = {round(ld, 1): (wr, n) for ld, n, wr in rung_rows}

            # Directional signal: line_direction → WR
            dir_rows = bets_con.execute("""
                SELECT line_direction,
                       count(*) as n,
                       avg(CASE WHEN leg_result='WIN' THEN 1.0 ELSE 0.0 END) as wr
                FROM mock_bet_legs
                WHERE line_direction IS NOT NULL
                  AND leg_result IN ('WIN','LOSS')
                  AND sport = 'MLB'
                GROUP BY line_direction
                HAVING n >= 20
            """).fetchall()
            ctx["direction_wr"] = {d: (wr, n) for d, n, wr in dir_rows}
            bets_con.close()
        except Exception:
            ctx["rung_wr"] = {}
            ctx["direction_wr"] = {}

    return ctx


# ── Settlement ───────────────────────────────────────────────────────────
def _fetch_espn_completed_scores(days_from: int = 3) -> dict:
    """Fetch completed MLB scores from ESPN → {(home_team, away_team, game_date): (home_score, away_score)}."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from espn_scores import fetch_espn_scores
    except ImportError:
        log.warning("espn_scores module not available for shadow settlement")
        return {}

    scores = fetch_espn_scores("baseball_mlb", days_from=days_from)
    result_map: dict[tuple, tuple] = {}
    for g in scores:
        if not g.get("completed"):
            continue
        ht = g.get("home_team", "")
        at = g.get("away_team", "")
        sc_list = g.get("scores", [])
        if len(sc_list) < 2 or not ht or not at:
            continue
        # Parse scores from the scores list (keyed by team name)
        hs = as_ = None
        for s in sc_list:
            try:
                val = int(s["score"])
            except (ValueError, KeyError, TypeError):
                continue
            if s.get("name") == ht:
                hs = val
            elif s.get("name") == at:
                as_ = val
        if hs is None or as_ is None:
            continue
        # Derive game_date (CT) from commence_time
        ct = g.get("commence_time", "")
        if ct:
            from datetime import timedelta
            try:
                # ESPN commence_time is ISO UTC like "2026-06-23T02:00Z"
                utc_str = ct.replace("Z", "+00:00")
                from datetime import datetime as _dt
                utc_dt = _dt.fromisoformat(utc_str)
                ct_dt = utc_dt - timedelta(hours=5)  # UTC → CDT (approx)
                gdate = ct_dt.strftime("%Y-%m-%d")
            except Exception:
                continue
        else:
            continue
        result_map[(ht, at, gdate)] = (hs, as_)
    return result_map


def settle_shadow_predictions(db=None):
    """Grade unsettled shadow predictions against ESPN final scores."""
    con = _get_con()
    unsettled = con.execute("""
        SELECT id, game_date, home_team, away_team, spread_value, market_type
        FROM shadow_predictions WHERE actual_result IS NULL
    """).fetchall()
    if not unsettled:
        con.close()
        return {"settled": 0}

    # Determine date range needed
    game_dates = {row[1] for row in unsettled}
    from datetime import date
    today = date.today().isoformat()
    max_days = 1
    for gd in game_dates:
        try:
            diff = (date.fromisoformat(today) - date.fromisoformat(gd)).days + 1
            max_days = max(max_days, diff)
        except Exception:
            pass
    max_days = min(max_days, 7)  # cap at 7 days

    score_map = _fetch_espn_completed_scores(days_from=max_days)
    if not score_map:
        con.close()
        return {"settled": 0, "reason": "no ESPN scores available"}

    settled_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for pred_id, gdate, ht, at, sv, mkt in unsettled:
        scores = score_map.get((ht, at, gdate))
        if scores is None:
            continue  # game not completed yet (or postponed) — leave NULL
        home_score, away_score = scores
        margin = home_score - away_score

        if mkt == "spreads" or "spread" in (mkt or "").lower():
            cover = margin + sv
            result = "WIN" if cover > 0 else ("PUSH" if cover == 0 else "LOSS")
        elif mkt == "h2h":
            result = "WIN" if margin > 0 else ("PUSH" if margin == 0 else "LOSS")
        elif mkt in ("totals", "alternate_totals"):
            total = home_score + away_score
            diff = total - sv
            result = "WIN" if diff > 0 else ("PUSH" if diff == 0 else "LOSS")
        else:
            # Default: spread grading (current shadow scoring only produces spreads)
            cover = margin + sv
            result = "WIN" if cover > 0 else ("PUSH" if cover == 0 else "LOSS")

        con.execute(
            "UPDATE shadow_predictions SET actual_result=?, settled_at=? WHERE id=?",
            (result, now_iso, pred_id),
        )
        settled_count += 1
    con.commit()
    con.close()
    print(f"[shadow] Settled {settled_count}/{len(unsettled)} shadow predictions")
    return {"settled": settled_count, "total_unsettled": len(unsettled)}

# ── Per-shadow self-calibration (settled only — leak-safe) ───────────────
_CAL_BUCKETS = [(0.0,0.35,"<35%"),(0.35,0.45,"35-45%"),(0.45,0.55,"45-55%"),
                (0.55,0.65,"55-65%"),(0.65,1.01,"65%+")]

def refresh_shadow_calibration(db=None):
    con = _get_con()
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = con.execute("""
        SELECT shadow_slot, sport, market_type, predicted_wp, actual_result
        FROM shadow_predictions WHERE actual_result IN ('WIN','LOSS')
    """).fetchall()
    if not rows:
        con.close()
        return {"updated": 0}
    from collections import defaultdict
    buckets = defaultdict(lambda: {"preds": [], "wins": 0, "n": 0})
    for slot, sport, mkt, wp, result in rows:
        bl = "65%+"
        for lo, hi, label in _CAL_BUCKETS:
            if lo <= wp < hi:
                bl = label; break
        key = (slot, sport, mkt, bl)
        buckets[key]["preds"].append(wp)
        buckets[key]["n"] += 1
        if result == "WIN":
            buckets[key]["wins"] += 1
    updated = 0
    for (slot, sport, mkt, bucket), data in buckets.items():
        if data["n"] < 5:
            continue
        avg_pred = sum(data["preds"]) / data["n"]
        avg_actual = data["wins"] / data["n"]
        haircut = avg_pred - avg_actual
        con.execute("""
            INSERT INTO shadow_calibration
                (shadow_slot, sport, market_type, wp_bucket,
                 avg_predicted, avg_actual, haircut_pp, sample_size, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(shadow_slot, sport, market_type, wp_bucket) DO UPDATE SET
                avg_predicted=excluded.avg_predicted, avg_actual=excluded.avg_actual,
                haircut_pp=excluded.haircut_pp, sample_size=excluded.sample_size,
                last_updated=excluded.last_updated
        """, (slot, sport, mkt, bucket, round(avg_pred,4), round(avg_actual,4),
              round(haircut,4), data["n"], now_iso))
        updated += 1
    con.commit(); con.close()
    return {"updated": updated}

# ── Selection pass (config shadows only) ─────────────────────────────────
# The live edge is SELECTION (which legs to bet), not probability accuracy.
# Config shadows (D1/D3) ask: "does different post-processing change which
# legs ALE selects, and do those selections win more?"
#
# For each config shadow, we simulate ALE selection:
#   1. Group predictions by fixture (home_team, away_team, game_date)
#   2. Compute LOS for each spread — same formula as parlay_builder._compute_los()
#   3. Pick the best rung per fixture (highest LOS = ALE's choice)
#   4. Apply confidence gates (MARKET_ELIGIBILITY min_wp, sport threshold)
#   5. Mark the selected predictions with selected=1
#
# NOTE: This is an approximation over the shadow's spread universe
# ([-2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), not the full set of
# alt-lines ALE evaluates from live odds. Selected-WR measures selection
# quality within this universe, not exact reproduction of live ALE picks.

# LOS weights (mirrored from parlay_builder.py)
_LOS_WIN_W   = 0.50
_LOS_EDGE_W  = 0.30
_LOS_LQS_W   = 0.20
_LOS_EDGE_SCALE = 15.0
# CUSHION weights (most alt-spread legs get this in the live system)
_LOS_WIN_W_C  = 0.60
_LOS_EDGE_W_C = 0.15
_LOS_LQS_W_C  = 0.25
# Selection gates — must match live path EXACTLY for selected-WR to be comparable.
#
# Live path (recommender.py:893): _model_wp >= 55.0 (on 0-100 scale).
# This is the ATS sub-model's raw win_prob passed through score_leg().
# Config shadows adjust this same model's output, so the gate must match.
#
# Previously 62.0 (from MARKET_ELIGIBILITY), but that gate applies to
# market-implied WP on the live alt-spread INJECTION path (recommender.py:991),
# NOT to the primary model_wp gate at recommender.py:893.
# The 62% gate over-filtered: configs pushing 56%→63% looked like "new selection"
# in shadow but live already had those legs at 56%.
_MLB_ALT_SPREAD_MIN_WP = 55.0   # match recommender.py:893 (model_wp gate)
_LOS_FLOOR = 0.52               # from parlay_builder


def _shadow_compute_los(predicted_wp: float, edge_pp: float = 0.0,
                        lqs: float = 50.0) -> float:
    """Compute LOS for a shadow prediction.

    For alt-spread legs, edge_pp is 0.0 (same as live system — recommender
    hardcodes edge=0 for alt-lines). Uses CUSHION weights since most MLB
    alt-spread legs have CUSHION margin grade in practice.
    """
    wp = predicted_wp  # already 0-1
    norm_edge = max(0.0, min(edge_pp, 15.0)) / _LOS_EDGE_SCALE
    lqs_norm = lqs / 100.0
    # Use CUSHION weights — alt-spreads are the primary config shadow market
    return round(wp * _LOS_WIN_W_C + norm_edge * _LOS_EDGE_W_C
                 + lqs_norm * _LOS_LQS_W_C, 4)


def run_shadow_selection(db=None):
    """Run ALE-style selection on config shadow predictions.

    Processes all config shadow predictions that haven't been selected yet
    (selected IS NULL). For each config shadow slot:
      - Group predictions by fixture (game_date + teams)
      - Compute LOS per spread, pick the best rung per fixture
      - Apply min_wp gate
      - Mark selected=1 on chosen predictions, selected=0 on others

    Model shadows (shadow_v3, shadow_v3_noaug) are NOT selected — they
    test model quality via Brier/AUC, not selection.
    """
    con = _get_con()
    config_slots = _load_config_slots().get("slots", {})
    active_configs = {k for k, v in config_slots.items() if v.get("active", True)}
    if not active_configs:
        con.close()
        return {"status": "no_config_slots", "selected": 0}

    total_selected = 0
    results = {}

    # Get average LQS for MLB alt-spreads from recent mock_bet_legs (proxy —
    # LQS doesn't depend on win_prob for alt-lines so it's slot-invariant)
    try:
        avg_lqs_row = con.execute("""
            SELECT AVG(COALESCE(l.scout_grade, 50.0))
            FROM mock_bet_legs l JOIN mock_bets b ON b.id = l.mock_bet_id
            WHERE l.sport='MLB' AND l.market_type='alternate_spreads'
              AND b.game_date >= date('now', '-14 days')
              AND l.leg_result IN ('WIN','LOSS')
        """).fetchone()
        base_lqs = float(avg_lqs_row[0]) if avg_lqs_row and avg_lqs_row[0] else 50.0
    except Exception:
        base_lqs = 50.0

    for slot_name in active_configs:
        # Fetch this config's unselected predictions (selected IS NULL)
        preds = con.execute("""
            SELECT id, home_team, away_team, game_date, spread_value, predicted_wp
            FROM shadow_predictions
            WHERE shadow_slot=? AND selected IS NULL
            ORDER BY game_date, home_team, away_team, spread_value
        """, (slot_name,)).fetchall()

        if not preds:
            results[slot_name] = {"selected": 0, "total": 0}
            continue

        # Group by fixture
        from collections import defaultdict
        fixtures = defaultdict(list)
        for pid, ht, at, gd, sv, wp in preds:
            fixtures[(ht, at, gd)].append({"id": pid, "sv": sv, "wp": wp})

        slot_selected_ids = []
        slot_total = len(preds)

        for fixture_key, legs in fixtures.items():
            # ── Both-sides selection (matches live ALE) ──────────
            # The shadow stores predictions from the HOME perspective:
            #   sv=+1.5: P(home covers +1.5)
            #   sv=-1.5: P(home covers -1.5)
            #
            # The live ALE creates legs for BOTH teams:
            #   Home at +1.5: win_prob = P(home +1.5)
            #   Away at +1.5: win_prob = 1 - P(home -1.5)
            #
            # Build a lookup so we can compute both sides.
            by_sv = {leg["sv"]: leg for leg in legs}

            # Live ALE rung range for alt-spread selection
            _LIVE_RUNGS = [1.5, 2.5, 3.5, 4.5]

            # Find qualifying candidates: check both sides at each rung
            candidates = []  # (rung, pick_side, wp, leg_id)
            for rung in _LIVE_RUNGS:
                # Home side at +rung
                home_leg = by_sv.get(rung)
                if home_leg:
                    home_wp = home_leg["wp"]
                    if home_wp * 100.0 >= _MLB_ALT_SPREAD_MIN_WP:
                        candidates.append((rung, "home", home_wp, home_leg["id"]))

                # Away side at +rung = 1 - P(home covers -rung)
                neg_leg = by_sv.get(-rung)
                if neg_leg:
                    away_wp = 1.0 - neg_leg["wp"]
                    if away_wp * 100.0 >= _MLB_ALT_SPREAD_MIN_WP:
                        candidates.append((rung, "away", away_wp, neg_leg["id"]))

            if not candidates:
                continue

            # Pick lowest qualifying rung (closest to main line = best value).
            # Within same rung, prefer the side with higher model WP.
            candidates.sort(key=lambda c: (c[0], -c[2]))
            best_rung, best_side, best_wp, best_id = candidates[0]

            # LOS gate
            los = _shadow_compute_los(best_wp, edge_pp=0.0, lqs=base_lqs)
            if los < _LOS_FLOOR:
                continue

            slot_selected_ids.append((best_id, best_side))

        # Write selection: selected=1 for chosen, selected=0 for others
        pred_ids = [p[0] for p in preds]
        selected_map = {sid: side for sid, side in slot_selected_ids}
        for pid in pred_ids:
            if pid in selected_map:
                con.execute(
                    "UPDATE shadow_predictions SET selected=1, pick_side=? WHERE id=?",
                    (selected_map[pid], pid),
                )
            else:
                con.execute(
                    "UPDATE shadow_predictions SET selected=0 WHERE id=?",
                    (pid,),
                )

        con.commit()
        total_selected += len(slot_selected_ids)
        results[slot_name] = {
            "selected": len(slot_selected_ids),
            "total": slot_total,
            "fixtures": len(fixtures),
        }
        print(f"[shadow-selection] {slot_name}: {len(slot_selected_ids)}/{len(fixtures)} "
              f"fixtures selected (LOS≥{_LOS_FLOOR}, wp≥{_MLB_ALT_SPREAD_MIN_WP}%)")

    con.close()
    return {"status": "ok", "total_selected": total_selected, "configs": results}


# ── Tripwire comparator (recommendation-only — NEVER auto-promotes) ─────
def compare_shadows_vs_active(db=None):
    con = _get_con()
    try:
        active_rows = con.execute("""
            SELECT mbl.win_prob, mbl.leg_result
            FROM mock_bet_legs mbl JOIN mock_bets mb ON mb.id=mbl.mock_bet_id
            WHERE mbl.sport='MLB' AND mbl.market_type IN ('alternate_spreads','spreads')
              AND mbl.leg_result IN ('WIN','LOSS')
              AND COALESCE(mbl.excluded_from_metrics,0)=0
              AND mb.game_date >= date('now','-30 days')
        """).fetchall()
    except Exception:
        active_rows = []

    def _metrics(predictions):
        if not predictions:
            return {"n": 0}
        n = len(predictions)
        wins = sum(1 for _,r in predictions if r == "WIN")
        wr = wins / n
        preds = [p for p,_ in predictions]
        actuals = [1.0 if r=="WIN" else 0.0 for _,r in predictions]
        brier = sum((p-a)**2 for p,a in zip(preds, actuals)) / n
        auc = None
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(actuals)) > 1:
                auc = roc_auc_score(actuals, preds)
        except Exception:
            pass
        wp_w = [p for p,r in predictions if r=="WIN"]
        wp_l = [p for p,r in predictions if r=="LOSS"]
        edge = None
        if wp_w and wp_l:
            edge = (sum(wp_w)/len(wp_w)) - (sum(wp_l)/len(wp_l))
        return {"n": n, "per_leg_wr": round(wr*100,2), "brier": round(brier,4),
                "auc": round(auc,4) if auc else None,
                "cal_gap_pp": round((sum(preds)/n - sum(actuals)/n)*100, 2),
                "discrimination_edge": round(edge,4) if edge else None}

    active_m = _metrics([(wp/100.0, r) for wp, r in active_rows if wp])

    # Active model's SELECTED legs: the legs that actually got bet
    # (alternate_spreads with stake>0 or in non-excluded bets)
    try:
        active_sel_rows = con.execute("""
            SELECT mbl.win_prob, mbl.leg_result
            FROM mock_bet_legs mbl JOIN mock_bets mb ON mb.id=mbl.mock_bet_id
            WHERE mbl.sport='MLB' AND mbl.market_type='alternate_spreads'
              AND mbl.leg_result IN ('WIN','LOSS')
              AND COALESCE(mbl.excluded_from_metrics,0)=0
              AND mb.game_date >= date('now','-30 days')
              AND mb.amount > 0
        """).fetchall()
    except Exception:
        active_sel_rows = []
    active_sel_m = _metrics([(wp/100.0, r) for wp, r in active_sel_rows if wp])

    shadow_reports = {}
    config_slot_names = set(_load_config_slots().get("slots", {}).keys())
    model_slot_names = set(_load_slots().get("slots", {}).keys())
    all_slot_names = list(model_slot_names) + list(config_slot_names)

    for slot_name in all_slot_names:
        is_config = slot_name in config_slot_names

        # Full-universe metrics (Brier/AUC — all predictions)
        s_rows = con.execute("""
            SELECT predicted_wp, actual_result FROM shadow_predictions
            WHERE shadow_slot=? AND actual_result IN ('WIN','LOSS')
        """, (slot_name,)).fetchall()
        m = _metrics(s_rows)

        # Config shadows: add SELECTED metrics (WR/PnL on ALE-selected legs)
        if is_config:
            sel_rows_raw = con.execute("""
                SELECT predicted_wp, actual_result, pick_side FROM shadow_predictions
                WHERE shadow_slot=? AND actual_result IN ('WIN','LOSS')
                  AND selected=1
            """, (slot_name,)).fetchall()
            # For away-side picks, flip predicted_wp and actual_result
            # (stored data is home-perspective; away bet is the inverse)
            sel_rows = []
            for wp, result, side in sel_rows_raw:
                if side == "away":
                    wp = 1.0 - wp
                    result = "LOSS" if result == "WIN" else "WIN"
                sel_rows.append((wp, result))
            sel_m = _metrics(sel_rows)
            m["selected_n"] = sel_m.get("n", 0)
            m["selected_wr"] = sel_m.get("per_leg_wr")
            m["selected_brier"] = sel_m.get("brier")
            m["selected_auc"] = sel_m.get("auc")
            m["selection_rate"] = round(sel_m["n"] / m["n"] * 100, 1) if m["n"] > 0 else 0
            m["note"] = ("Selection approximated over 8-rung universe "
                         "[-2.5..6.5], not full ALE alt-line set")

        # Tripwire logic — different criteria for model vs config shadows
        fires = False; reason = ""
        if m["n"] < TRIPWIRE_MIN_LEGS:
            reason = f"insufficient data (n={m['n']}<{TRIPWIRE_MIN_LEGS})"
        else:
            span = con.execute("""
                SELECT MIN(game_date), MAX(game_date) FROM shadow_predictions
                WHERE shadow_slot=? AND actual_result IN ('WIN','LOSS')
            """, (slot_name,)).fetchone()
            span_days = 0
            if span and span[0] and span[1]:
                d0 = datetime.strptime(span[0], "%Y-%m-%d")
                d1 = datetime.strptime(span[1], "%Y-%m-%d")
                span_days = (d1-d0).days + 1
            if span_days < TRIPWIRE_MIN_DAYS:
                reason = f"insufficient span ({span_days}d<{TRIPWIRE_MIN_DAYS}d)"
            elif is_config:
                # CONFIG shadows: tripwire fires on SELECTION edge
                # Does this config's selected-WR beat active's selected-WR?
                sel_wr = m.get("selected_wr")
                sel_n = m.get("selected_n", 0)
                active_sel_wr = active_sel_m.get("per_leg_wr")
                if sel_n < 20:
                    reason = f"insufficient selected legs (n={sel_n}<20)"
                elif sel_wr is None or active_sel_wr is None:
                    reason = "missing selected-WR data"
                else:
                    # Also check Brier as secondary calibration gate
                    brier_ok = (m.get("brier") or 1.0) <= (active_m.get("brier") or 0.0) + 0.02
                    wr_beats = sel_wr > active_sel_wr
                    if wr_beats and brier_ok:
                        fires = True
                        reason = (f"selected-WR {sel_wr:.1f}% > active {active_sel_wr:.1f}% "
                                  f"(n={sel_n}, {span_days}d, Brier ok)")
                    elif wr_beats:
                        reason = (f"selected-WR {sel_wr:.1f}% > active {active_sel_wr:.1f}% "
                                  f"but Brier degraded — needs review")
                    else:
                        reason = (f"selected-WR {sel_wr:.1f}% ≤ active {active_sel_wr:.1f}% "
                                  f"(n={sel_n}, {span_days}d)")
            else:
                # MODEL shadows: tripwire fires on calibration/discrimination
                beats = 0; comps = 0
                for mk in ["brier", "auc", "discrimination_edge"]:
                    sv = m.get(mk); av = active_m.get(mk)
                    if sv is None or av is None: continue
                    comps += 1
                    better = sv < av if mk == "brier" else sv > av
                    if better: beats += 1
                if comps > 0 and beats >= 2:
                    fires = True
                    reason = f"outperforms on {beats}/{comps} calibration metrics over {span_days}d"
                else:
                    reason = f"does not outperform ({beats}/{comps} calibration metrics)"

        m["tripwire"] = {"fires": fires, "reason": reason,
                         "action": "RECOMMENDATION ONLY" if fires else "no action",
                         "criteria": "selected-WR" if is_config else "Brier/AUC"}
        shadow_reports[slot_name] = m
    con.close()
    return {"active": active_m, "active_selected": active_sel_m,
            "shadows": shadow_reports,
            "note": "Tripwire is RECOMMENDATION ONLY. No auto-promotion."}
