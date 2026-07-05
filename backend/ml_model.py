"""
ml_model.py — Model v3 (Phase 5)

Improvements over v2:
  1. Combined training data — personal bets (Pikkit) + historical games (historical.db)
  2. Sample weights — historical:1× / personal >90d:3× / personal ≤90d:6×
  3. CLV_READY hook — add_clv_features() called in pipeline; fill it when
     TheOddsAPI paid plan activates, nothing else changes
  4. SimpleImputer added before StandardScaler to handle NaN from historical
     team-quality cols that personal bets don't have
  5. HIST_EXTRA_COLS — 8 rolling team-strength features from features.py,
     populated for historical rows, NaN for personal bets (imputer fills median)

──────────────────────────────────────────────────────────────────────────────
TWO SEPARATE MODEL PATHS — do not confuse them
──────────────────────────────────────────────────────────────────────────────
PRODUCTION  train(db)
  • Combines personal bets + historical.db game rows into one DataFrame.
  • Feature space: FEATURE_COLS (personal-bet schema — odds, sport_id, legs,
    implied_prob, etc.). Historical rows are mapped into this schema.
  • Saved to data/model.pkl. Used by predict_bet() for every recommendation.
  • Last known AUC: 0.5972 (NFL as sole HISTORICAL_ATS_SPORTS sport).
  • Endpoint: POST /api/model/train

DIAGNOSTIC  train_diagnostic_sport_model(sport)
  • Trains on historical ATS rows for a single sport in isolation.
  • Feature space: MODEL_FEATURE_COLS (rolling team stats, schedule, SOS —
    features.py). No personal-bet features.
  • NOT saved to model.pkl; optional save to data/submodels/{sport}_ats_clf.pkl
    only when AUC >= 0.55 (activation threshold).
  • Intended for feature research and sport-level signal detection, not
    production inference.
  • Endpoint: POST /api/model/train-diagnostic-sport

SPORT ROUTING (final):
  NHL     → nhl_ats_v1         AUC 0.6372  threshold 65%  ACTIVE  (retrained 2026-05-11, sklearn 1.2.1)
  MLB     → mlb_ats_v1         AUC 0.5472  threshold 60%  ACTIVE  (retrained 2026-05-11, sklearn 1.2.1)
  NOTE: AUC for ATS submodels is measured on ATS spread-coverage holdout data (random baseline=0.50).
        Combined model AUC 0.5972 is on a heterogeneous personal-bet schema (different feature
        space, different task). The two AUCs are NOT comparable. MLB at 0.5472 = +4.72pp over
        random on the ATS task; combined_v1 has no ATS rolling-stat features and cannot compete.
  Soccer  → soccer_total_v1    AUC 0.5578  threshold 55%  ACTIVE (totals only)
            soccer_ml_v1       AUC 0.5250  (below 0.54 gate — using combined fallback)
  NFL     → combined_v1        AUC 0.5972  threshold 55%  PATH B (market efficient)
  NBA     → combined_v1        AUC 0.5972  threshold 55%  PATH B (market efficient)

NBA diagnostic: 0.4904 — confirmed Path B, dropna fix had no impact
NFL diagnostic: 0.4928 — confirmed Path B, dropna fix had no impact
Both sports removed from future diagnostic retrain queue.
Soccer moneyline: retrain when ~150+ resolved ML legs are available.
No further isolation testing planned for NFL or NBA unless
a new data source becomes available (real-time injury API,
line movement history > 6 months, or pace-adjusted NBA metrics).
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import math
import os
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from database import Bet, BetLeg, ModelRun, Prediction

import logging
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH       = os.path.join(DATA_DIR, "model.pkl")
SCALER_PATH      = os.path.join(DATA_DIR, "scaler.pkl")
IMPUTER_PATH     = os.path.join(DATA_DIR, "imputer.pkl")
ACTIVE_COLS_PATH = os.path.join(DATA_DIR, "active_feature_cols.pkl")
SUBMODEL_DIR = os.path.join(DATA_DIR, "submodels")
DRIFT_PATH   = os.path.join(DATA_DIR, "drift_state.json")
os.makedirs(SUBMODEL_DIR, exist_ok=True)

# ── Model versioning ─────────────────────────────────────────────────────────
# Every model save writes to a versioned path AND copies to the live path.
# A JSON registry tracks all versions with metadata.  This makes artifact
# loss impossible: the live file can always be restored from any version.
_VERSION_DIR      = os.path.join(DATA_DIR, "model_versions")
_VERSION_REGISTRY = os.path.join(_VERSION_DIR, "registry.json")
os.makedirs(_VERSION_DIR, exist_ok=True)


def _load_version_registry() -> dict:
    """Load the version registry or return empty structure."""
    if os.path.exists(_VERSION_REGISTRY):
        with open(_VERSION_REGISTRY, "r") as f:
            return json.load(f)
    return {"versions": [], "active": {}}


def _save_version_registry(reg: dict) -> None:
    with open(_VERSION_REGISTRY, "w") as f:
        json.dump(reg, f, indent=2)


def _next_version(reg: dict, model_key: str) -> int:
    """Return the next version number for this model_key."""
    existing = [v["version"] for v in reg["versions"] if v["model_key"] == model_key]
    return max(existing, default=0) + 1


def _versioned_save(artifact: object, live_path: str, model_key: str,
                    metadata: dict | None = None) -> str:
    """Save artifact to a versioned path and copy to the live path.

    Returns the versioned path.  Never overwrites a versioned file.
    """
    import shutil
    from datetime import datetime as _dt

    reg = _load_version_registry()
    ver = _next_version(reg, model_key)
    ts  = _dt.utcnow().strftime("%Y%m%d_%H%M%S")

    # Versioned filename: {model_key}_v{N}_{timestamp}.pkl
    ext = os.path.splitext(live_path)[1] or ".pkl"
    ver_name = f"{model_key}_v{ver}_{ts}{ext}"
    ver_path = os.path.join(_VERSION_DIR, ver_name)

    # Write versioned copy
    if ext == ".json":
        with open(ver_path, "w") as f:
            json.dump(artifact, f, indent=2)
    else:
        with open(ver_path, "wb") as f:
            pickle.dump(artifact, f)

    # Copy to live path (atomic: write tmp then rename)
    tmp_path = live_path + ".tmp"
    shutil.copy2(ver_path, tmp_path)
    os.replace(tmp_path, live_path)

    # Update registry
    entry = {
        "model_key": model_key,
        "version":   ver,
        "timestamp": ts,
        "file":      ver_name,
        "live_path": os.path.relpath(live_path, DATA_DIR),
        "size_bytes": os.path.getsize(ver_path),
    }
    if metadata:
        entry["metadata"] = metadata
    reg["versions"].append(entry)
    reg["active"][model_key] = ver
    _save_version_registry(reg)

    log.info("versioned_save: %s v%d → %s", model_key, ver, ver_name)
    return ver_path


def get_model_versions(model_key: str | None = None) -> list[dict]:
    """Return version history, optionally filtered by model_key."""
    reg = _load_version_registry()
    versions = reg["versions"]
    if model_key:
        versions = [v for v in versions if v["model_key"] == model_key]
    return versions


def revert_model(model_key: str, version: int) -> dict:
    """Revert a model to a specific version by copying the versioned file to the live path."""
    import shutil
    reg = _load_version_registry()
    match = [v for v in reg["versions"] if v["model_key"] == model_key and v["version"] == version]
    if not match:
        return {"error": f"No version {version} for {model_key}"}
    entry = match[0]
    ver_path = os.path.join(_VERSION_DIR, entry["file"])
    if not os.path.exists(ver_path):
        return {"error": f"Versioned file missing: {entry['file']}"}
    live_path = os.path.join(DATA_DIR, entry["live_path"])
    tmp_path = live_path + ".tmp"
    shutil.copy2(ver_path, tmp_path)
    os.replace(tmp_path, live_path)
    reg["active"][model_key] = version
    _save_version_registry(reg)
    log.info("revert_model: %s → v%d (%s)", model_key, version, entry["file"])
    return {"reverted": model_key, "version": version, "file": entry["file"]}

# ── Post-hoc win_prob calibrator (Platt scaling) ─────────────────────────────
# Reversible: remove the _calibrate_prob() calls to disable.
_CALIBRATOR_PATH = os.path.join(SUBMODEL_DIR, "win_prob_calibrator.pkl")
_calibrator_cache: dict = {}  # singleton cache

def _load_calibrator():
    """Load the Platt calibrator (lazy, cached). Returns the LogisticRegression or None."""
    if "obj" in _calibrator_cache:
        return _calibrator_cache["obj"]
    if not os.path.exists(_CALIBRATOR_PATH):
        _calibrator_cache["obj"] = None
        return None
    try:
        with open(_CALIBRATOR_PATH, "rb") as f:
            cal = pickle.load(f)
        _calibrator_cache["obj"] = cal
        return cal
    except Exception as exc:
        log.warning("_load_calibrator: failed (%s) — using raw probabilities", exc)
        _calibrator_cache["obj"] = None
        return None

def _calibrate_prob(p: float) -> float:
    """Apply post-hoc Platt calibration to a raw model probability.
    Returns the calibrated probability, or the original if no calibrator loaded."""
    cal = _load_calibrator()
    if cal is None:
        return p
    try:
        import numpy as _np
        lo = _np.log(max(p, 1e-6) / max(1 - p, 1e-6))
        return float(cal.predict_proba([[lo]])[0][1])
    except Exception:
        return p  # graceful fallback to raw

# ── Training environment pin ─────────────────────────────────────────────────
# Models pickled with one numpy/sklearn version cannot be loaded by another.
# If save=True and the runtime doesn't match the server's .venv, raise so we
# never silently save incompatible .pkl files again (broke 2026-06-09..06-13).
_REQUIRED_TRAIN_VERSIONS = {"numpy": "2.2.6", "scikit-learn": "1.7.2"}

def _assert_training_env():
    """Raise RuntimeError if numpy/sklearn don't match the server's .venv."""
    import importlib.metadata as _meta
    for pkg, expected in _REQUIRED_TRAIN_VERSIONS.items():
        actual = _meta.version(pkg)
        if actual != expected:
            raise RuntimeError(
                f"Cannot save model: {pkg} {actual} != {expected}. "
                f"Run with .venv/bin/python to match server environment."
            )


# ── Sample weight configuration ────────────────────────────────────────────────
# Adjust weights here without touching model code.
WEIGHT_CONFIG = {
    "historical_base":      1.0,   # historical.db ATS games: foundation / volume
    "personal_base":        3.0,   # personal bets older than personal_recent_days
    "personal_recent":      6.0,   # personal bets within personal_recent_days (3× source × 2× recency)
    "personal_recent_days": 90,    # recency cutoff in days
}

# Sports in historical.db that are candidates for ATS training data.
# Each sport is gated by cover_rate_check() at runtime — if a sport's cover rate
# falls outside [46%, 54%] it is silently excluded from that training run.
# This lets us keep the list broad without risking data-quality regressions.
HISTORICAL_ATS_SPORTS = ["NFL", "MLB", "NHL"]
# NFL:  4 seasons. Isolation AUC = 0.4928. PATH B — market efficient against rolling
#       stats. Remains in combined model for training diversity (cover_rate ~50.5% ✓).
#       No further diagnostic retrain unless new data source available.
# NBA:  4 seasons (2022-26), 4,863 ATS rows. Isolation AUC = 0.4904. PATH B —
#       confirmed market efficient. Excluded from combined model (dropped AUC to 0.3994).
#       No further diagnostic retrain unless real-time injury API, line movement
#       history > 6 months, or pace-adjusted metrics become available.
# MLB:  4 seasons (2022-25). Isolation AUC = 0.5472 ✓ ACTIVE → mlb_ats_v1.
#       Retrained 2026-05-11 (sklearn 1.2.1 + numpy 1.23.5). Threshold 60%.
# NHL:  5 seasons (2021-22 to 2025-26). Isolation AUC = 0.6372 ✓ ACTIVE → nhl_ats_v1.
#       Retrained 2026-05-11 (sklearn 1.2.1 + numpy 1.23.5). Threshold 65%.
#       ot_so_game is dominant feature: OT/SO always 1-goal margin.


# ── CLV_READY hook ─────────────────────────────────────────────────────────────
def add_clv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich *df* with CLV columns. Called on both personal and historical
    DataFrames before the final concat — so filling it once enriches everything.

    CLV_READY: currently a no-op. When TheOddsAPI paid plan is active:
        1. Query opening odds from TheOddsAPI for each row's game_id / event_id
        2. df["clv"]           = log(close_odds) - log(open_odds)
        3. df["clv_available"] = 1
        4. Return enriched df
    """
    # CLV_READY — no-op until TheOddsAPI paid plan active
    return df


import re
_SPREAD_RE = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*(?:\(|$)')

def _extract_spread_value(description: str, market_type: str) -> float:
    """
    Extract the numeric spread from a leg description.
    Examples:
        "Cincinnati Reds +1.5 (Spread)"     → +1.5
        "NY Yankees -1.5 (Alt Spread)"      → -1.5
        "Detroit Pistons +5.5 (Alt Spread)"  → +5.5
    Returns 0.0 for non-spread markets (h2h, totals, props).
    """
    if "spread" not in (market_type or "").lower():
        return 0.0
    if not description:
        return 0.0
    m = _SPREAD_RE.search(description)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def _build_recency_features(db_session) -> dict:
    """
    Build a lookup dict of recency features from settled mock_bet_legs.
    Returns: {(team_lower, market_type): {cover_rate, recent_wr, form_trend}}

    cover_rate:  rolling win rate on this market type (last 10 settled legs)
    recent_wr:   rolling straight-up win rate (last 10 h2h legs)
    form_trend:  last-5 WR minus last-20 WR (positive = hot streak)
    """
    from sqlalchemy import text as _text
    try:
        rows = db_session.execute(_text("""
            SELECT mbl.description, mbl.market_type, mbl.leg_result,
                   mbl.resolved_home_team, mbl.resolved_away_team,
                   mb.game_date, mbl.sport
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mbl.sport IS NOT NULL
            ORDER BY mb.game_date DESC
        """)).fetchall()
    except Exception as exc:
        log.warning("_build_recency_features: %s", exc)
        return {}

    from collections import defaultdict
    # Accumulate per (team, market_type) and per (team, 'h2h')
    team_market_results = defaultdict(list)   # (team, mkt) → [1/0, ...]
    team_h2h_results    = defaultdict(list)   # team → [1/0, ...]

    for r in rows:
        desc, mkt, result = r[0] or "", r[1] or "", r[2]
        home, away = (r[3] or "").lower().strip(), (r[4] or "").lower().strip()
        won = 1.0 if result == "WIN" else 0.0

        # Determine which team this leg is about from description
        desc_lower = desc.lower()
        team = None
        if home and home in desc_lower:
            team = home
        elif away and away in desc_lower:
            team = away

        if not team:
            continue

        team_market_results[(team, mkt)].append(won)
        # Also accumulate for straight-up WR (all markets)
        team_h2h_results[team].append(won)

    lookup = {}
    for (team, mkt), results in team_market_results.items():
        last_10 = results[:10]  # already sorted DESC
        cover_rate = sum(last_10) / len(last_10) if last_10 else 0.5

        h2h_all = team_h2h_results.get(team, [])
        recent_wr = sum(h2h_all[:10]) / min(len(h2h_all), 10) if h2h_all else 0.5

        last_5  = h2h_all[:5]
        last_20 = h2h_all[:20]
        wr_5  = sum(last_5)  / len(last_5)  if len(last_5)  >= 3 else 0.5
        wr_20 = sum(last_20) / len(last_20) if len(last_20) >= 5 else 0.5
        form_trend = wr_5 - wr_20

        lookup[(team, mkt)] = {
            "cover_rate": round(cover_rate, 4),
            "recent_wr":  round(recent_wr,  4),
            "form_trend": round(form_trend, 4),
        }

    log.info("_build_recency_features: %d team-market combos indexed", len(lookup))
    return lookup


# ── Mappings ───────────────────────────────────────────────────────────────────
SPORT_MAP = {
    "Basketball": 0, "American Football": 1, "Baseball": 2,
    "Soccer": 3, "Ice Hockey": 4, "Tennis": 5, "other": 6,
}
LEAGUE_MAP = {
    "NBA": 0, "NFL": 1, "MLB": 2, "NCAAM": 3, "NCAAFB": 4,
    "La Liga": 5, "UEFA Champions League": 6, "NHL": 7,
    "WNBA": 8, "English Premier League": 9, "other": 10,
}
SUBMODEL_SPORTS = ["Basketball", "American Football", "Baseball", "Soccer", "Ice Hockey"]

# historical.db sport key → SPORT_MAP / LEAGUE_MAP ids
_SPORT_KEY_MAP: dict[str, dict] = {
    "NBA":        {"sport_id": SPORT_MAP["Basketball"],        "league_id": LEAGUE_MAP["NBA"]},
    "NFL":        {"sport_id": SPORT_MAP["American Football"], "league_id": LEAGUE_MAP["NFL"]},
    "MLB":        {"sport_id": SPORT_MAP["Baseball"],          "league_id": LEAGUE_MAP["MLB"]},
    "NHL":        {"sport_id": SPORT_MAP["Ice Hockey"],        "league_id": LEAGUE_MAP["NHL"]},
    "CFB":        {"sport_id": SPORT_MAP["American Football"], "league_id": LEAGUE_MAP["NCAAFB"]},
    "CBB":        {"sport_id": SPORT_MAP["Basketball"],        "league_id": LEAGUE_MAP["NCAAM"]},
    "EPL":        {"sport_id": SPORT_MAP["Soccer"],            "league_id": LEAGUE_MAP["English Premier League"]},
    "UCL":        {"sport_id": SPORT_MAP["Soccer"],            "league_id": LEAGUE_MAP["UEFA Champions League"]},
    "LaLiga":     {"sport_id": SPORT_MAP["Soccer"],            "league_id": LEAGUE_MAP["La Liga"]},
    "Bundesliga": {"sport_id": SPORT_MAP["Soccer"],            "league_id": LEAGUE_MAP["other"]},
    "Ligue1":     {"sport_id": SPORT_MAP["Soccer"],            "league_id": LEAGUE_MAP["other"]},
    "SerieA":     {"sport_id": SPORT_MAP["Soccer"],            "league_id": LEAGUE_MAP["other"]},
}

# ── Feature schema ─────────────────────────────────────────────────────────────
# Personal-bet features — unchanged from v2 (extract_features() keys)
_PERSONAL_FEATURE_COLS: list[str] = [
    "legs", "odds", "log_odds", "implied_prob", "stake", "is_parlay",
    "sport_id", "league_id", "hour_placed", "day_of_week",
    "ml_pct", "spread_pct", "total_pct", "prop_pct",
    "multi_sport", "n_sports", "has_ev", "ev_value", "closing_line_diff",
    "odds_bracket", "leg_diversity", "clv", "clv_available",
    "multi_league", "n_leagues", "prop_density", "stake_ratio",
    "month_sin", "month_cos", "hour_sin", "hour_cos",
    # New: Kelly sizing quality
    "stake_vs_kelly",   # log(actual_stake / kelly_optimal_stake); 0=right-sized
]

# Historical-only team-quality features from features.py.
# Populated for historical rows; NaN for personal bets → imputer fills median.
HIST_EXTRA_COLS: list[str] = [
    # Core rolling quality (original 8)
    "margin_diff_10g",    # home − away 10-game avg margin  (top ATS predictor)
    "win_pct_diff_10g",   # home − away 10-game win%
    "form_diff_3g",       # home − away 3-game scoring form
    "h_margin_10g",       # home team 10-game avg margin
    "a_margin_10g",       # away team 10-game avg margin
    "h_win_pct_10g",      # home team 10-game win%
    "a_win_pct_10g",      # away team 10-game win%
    "rest_advantage",     # home rest days − away rest days
    # New: momentum (days_since_last_win) + strength-of-schedule (opp ML implied prob)
    "days_since_win_diff",   # home_days_since_win − away_days_since_win (+ve = home colder)
    "h_days_since_win",      # home team: days since last win (momentum)
    "a_days_since_win",      # away team: days since last win
    "h_opp_impl_prob_5g",    # home: avg implied prob of last-5 opponents (SOS)
    "a_opp_impl_prob_5g",    # away: avg implied prob of last-5 opponents
    "opp_strength_diff",     # h_opp_impl_prob_5g − a_opp_impl_prob_5g
    # Pi (Elo-like) team power ratings — margin-based, opponent-quality-adjusted
    "home_pi",               # home team's home Pi rating (0-centered)
    "away_pi",               # away team's away Pi rating (0-centered)
    "pi_diff",               # home_pi − away_pi (positive = home stronger)
    "pi_confidence",         # min(home_gp, away_gp) / 30 — rating stability
]

# Per-leg signals shipped May 2026 — populated on mock_bet_legs, NaN elsewhere.
# HistGradientBoostingClassifier handles NaN natively; SimpleImputer fills median
# for the legacy GBC path.  Features accumulate non-null values over time.
NEW_LEG_FEATURE_COLS: list[str] = [
    "injury_net_advantage",      # net injury impact (home − away), shipped 2026-05-11
    "line_movement_signal",      # steam/reverse-line movement indicator, shipped 2026-05-11
    "scout_grade_encoded",       # scout quality A=4 B=3 C=2 D=1 NULL=NaN, shipped 2026-05-11
    # clv_cents removed from training — not available at inference time (0.0009 importance)
    "home_team_injury_impact",   # home-side injury severity (0−1), shipped 2026-05-11
    "away_team_injury_impact",   # away-side injury severity (0−1), shipped 2026-05-11
    # ── Spread-aware features — shipped 2026-06-01 ──
    # The model previously treated -1.5 and +2.5 spreads identically (both ≈75% pred).
    # Actual: -1.5 covers 31%, +2.5 covers 76%. These features fix that blind spot.
    "spread_value",              # signed numeric spread (+1.5, -2.5, etc.); 0 for non-spread
    "spread_abs",                # absolute distance from pick-line (1.5, 2.5, etc.)
    "is_favorite_spread",        # 1 if covering as favorite (negative spread), 0 otherwise
    # ── Recency features — shipped 2026-06-01 ──
    # Prevents model from repeatedly picking teams that can't cover (e.g. Angels -1.5 0/19).
    "team_recent_cover_rate",    # rolling 10-game cover rate for this team+market
    "team_recent_wr",            # rolling 10-game straight-up win rate
    "team_form_trend",           # 5-game WR minus 20-game WR (+ = hot, − = cold)
    # ── MLB-specific features — shipped 2026-06-01 ──
    # Park factors (static lookup, always available for MLB; 1.0 for non-MLB)
    "park_runs_factor",          # park runs factor (>1.0 = hitter-friendly)
    "park_hr_factor",            # park HR factor
    # Starting pitcher quality (from MLB Stats API)
    "home_starter_era",          # home starting pitcher season ERA
    "home_starter_whip",         # home starter WHIP
    "home_starter_k9",           # home starter K/9
    "home_starter_hr9",          # home starter HR/9
    "away_starter_era",          # away starting pitcher season ERA
    "away_starter_whip",         # away starter WHIP
    "away_starter_k9",           # away starter K/9
    "away_starter_hr9",          # away starter HR/9
    "starter_era_diff",          # home ERA - away ERA (negative = home has better pitcher)
    "starter_whip_diff",         # home WHIP - away WHIP
    "starter_k9_diff",           # home K/9 - away K/9
    # Weather features (from Open-Meteo, MLB outdoor games only)
    "game_temp_f",               # game-time temperature in °F
    "game_wind_mph",             # game-time wind speed in mph
    "wind_helps_scoring",        # 1.0=blowing out, 0.0=blowing in, 0.5=crosswind
    # Statcast batting quality (from Baseball Savant, weekly cache)
    "home_team_xwoba",           # home team avg expected wOBA
    "away_team_xwoba",           # away team avg expected wOBA
    "xwoba_diff",                # home xwoba - away xwoba
    "home_team_barrel_pct",      # home team avg barrel%
    "away_team_barrel_pct",      # away team avg barrel%
    "barrel_diff",               # home barrel% - away barrel%
    # ── Bullpen quality features — shipped 2026-06-01 ──
    "home_bullpen_era",          # home team bullpen ERA (season, relief pitchers)
    "away_bullpen_era",          # away team bullpen ERA
    "bullpen_era_diff",          # home bullpen ERA - away bullpen ERA (negative = home better)
    # ── H2H historical matchup features — shipped 2026-06-01 ──
    "h2h_home_wr",               # home team win rate in H2H matchups (from settled legs)
    "h2h_n_games",               # number of settled H2H games between these teams
]

# Unified schema used for training and inference (v3)
FEATURE_COLS: list[str] = _PERSONAL_FEATURE_COLS + HIST_EXTRA_COLS + NEW_LEG_FEATURE_COLS

# Original 46-feature set (GBC baseline, no NaN-only columns).
# Kept for backward compatibility with older model pkl files.
ORIGINAL_FEATURE_COLS: list[str] = _PERSONAL_FEATURE_COLS + HIST_EXTRA_COLS


# ── Feature engineering — personal bets ───────────────────────────────────────
def extract_features(bet: Bet) -> Optional[dict]:
    if not bet.odds or bet.odds <= 1:
        return None

    hour   = bet.time_placed.hour      if bet.time_placed else 12
    dow    = bet.time_placed.weekday() if bet.time_placed else 0
    month  = bet.time_placed.month     if bet.time_placed else 6

    sport_raw  = (bet.sports  or "").split("|")[0].strip()
    league_raw = (bet.leagues or "").split("|")[0].strip()
    log_odds   = math.log(float(bet.odds))
    imp_prob   = 1 / bet.odds

    info = (bet.bet_info or "").lower()
    ml_c   = info.count("moneyline")
    spr_c  = sum(1 for kw in ["run line", "spread", "point spread"] if kw in info)
    tot_c  = sum(1 for kw in ["over", "under", "total"] if kw in info)
    prop_c = sum(1 for kw in ["to score", "shots on target", "player", "corners",
                               "goals", "assists", "rebounds", "strikeouts"] if kw in info)
    t_mkts = max(ml_c + spr_c + tot_c + prop_c, 1)

    n_mkt_types  = sum([ml_c > 0, spr_c > 0, tot_c > 0, prop_c > 0])
    leg_diversity = n_mkt_types / 4.0

    if   bet.odds < 2:   odds_bracket = 0
    elif bet.odds < 5:   odds_bracket = 1
    elif bet.odds < 15:  odds_bracket = 2
    elif bet.odds < 50:  odds_bracket = 3
    else:                odds_bracket = 4

    clv = clv_avail = 0
    if bet.closing_line and bet.odds and bet.closing_line > 1:
        clv       = math.log(bet.closing_line) - log_odds
        clv_avail = 1

    leagues_list = [l.strip() for l in (bet.leagues or "").split("|") if l.strip()]
    n_leagues    = len(set(leagues_list))

    # stake_vs_kelly: log(actual / kelly_optimal).
    # Kelly fraction = ev_value / (decimal_odds - 1) where ev_value = prob*odds - 1.
    # We use the bet's own implied_prob as a proxy for true win probability, which
    # assumes fair odds (no edge); the signal is in RELATIVE sizing — bets where
    # the bettor stakes more than Kelly optimal at given EV tend to be over-confident.
    # For historical rows this is set to 0.0 (neutral / not applicable).
    ev_val    = float(bet.ev) if bet.ev else 0.0
    dec_odds  = float(bet.odds) if bet.odds else 1.1
    kelly_f   = max(ev_val, 0.001) / max(dec_odds - 1, 0.001)
    stake_amt = float(bet.amount or 5)
    # Normalise by a reference stake of 5 units to make it scale-independent,
    # then take log so the feature is symmetric around 0.
    stake_vs_kelly = math.log(max(stake_amt / 5.0, 0.01) / kelly_f)
    stake_vs_kelly = float(np.clip(stake_vs_kelly, -5, 5))   # bound outliers

    return {
        "legs":              bet.legs,
        "odds":              float(bet.odds),
        "log_odds":          log_odds,
        "implied_prob":      round(imp_prob, 6),
        "stake":             float(bet.amount or 5),
        "is_parlay":         1 if bet.bet_type == "parlay" else 0,
        "sport_id":          SPORT_MAP.get(sport_raw, 6),
        "league_id":         LEAGUE_MAP.get(league_raw, 10),
        "hour_placed":       hour,
        "day_of_week":       dow,
        "ml_pct":            ml_c   / t_mkts,
        "spread_pct":        spr_c  / t_mkts,
        "total_pct":         tot_c  / t_mkts,
        "prop_pct":          prop_c / t_mkts,
        "multi_sport":       1 if "|" in (bet.sports or "") else 0,
        "n_sports":          len(set((bet.sports or "").split("|"))),
        "has_ev":            1 if bet.ev is not None else 0,
        "ev_value":          float(bet.ev) if bet.ev is not None else 0.0,
        "closing_line_diff": float(bet.closing_line - bet.odds) if (bet.closing_line and bet.odds) else 0.0,
        "odds_bracket":      odds_bracket,
        "leg_diversity":     leg_diversity,
        "clv":               clv,
        "clv_available":     clv_avail,
        "multi_league":      1 if n_leagues > 1 else 0,
        "n_leagues":         n_leagues,
        "prop_density":      prop_c / max(bet.legs, 1),
        "stake_ratio":       float(bet.amount or 5) / 6.8,
        "month_sin":         math.sin(2 * math.pi * month / 12),
        "month_cos":         math.cos(2 * math.pi * month / 12),
        "hour_sin":          math.sin(2 * math.pi * hour / 24),
        "hour_cos":          math.cos(2 * math.pi * hour / 24),
        "stake_vs_kelly":    stake_vs_kelly,
    }


# ── Data-source builders ───────────────────────────────────────────────────────

def _american_to_decimal(ml: Optional[float]) -> float:
    """American moneyline → decimal odds. Returns 1.909 (≈−110) if unavailable."""
    if ml is None or (isinstance(ml, float) and math.isnan(ml)) or ml == 0:
        return 1.909
    return (ml / 100 + 1) if ml > 0 else (100 / abs(ml) + 1)


def _build_personal_df(bets: list, cutoff: datetime) -> pd.DataFrame:
    """
    Convert settled Bet objects → DataFrame aligned with FEATURE_COLS.
    HIST_EXTRA_COLS are NaN (filled by imputer during training/inference).

    Extra metadata columns (prefix _) are dropped before fitting:
        _label        : 1 = WIN, 0 = LOSS
        _weight       : per WEIGHT_CONFIG
        _date         : bet placement datetime (for chronological sort)
        _sport_label  : human-readable sport name (for sub-model routing)
    """
    # ── Pre-build spread lookup from bet_legs ─────────────────────────────
    # For each bet_id, compute average spread_value across spread legs.
    # Run Line descriptions: "Team 1.5 Run Line ..." (always +1.5, no sign)
    # Spread descriptions: "Team +10.5 Alternate Spread ..." (has sign)
    _spread_lookup: dict[str, tuple[float, float, float]] = {}
    try:
        import sqlite3 as _sq3
        _db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        _con = _sq3.connect(_db_path, timeout=10)
        _leg_rows = _con.execute(
            "SELECT bet_id, description, market_type "
            "FROM bet_legs WHERE market_type IN ('Spread','Run Line')"
        ).fetchall()
        _con.close()

        _SPREAD_NUM = re.compile(r'([+-]?\d+(?:\.\d+)?)\s+(?:Run Line|Alternate\s+Spread|Spread)')
        from collections import defaultdict
        _by_bet: dict[str, list[float]] = defaultdict(list)
        for bid, desc, mkt in _leg_rows:
            m = _SPREAD_NUM.search(desc or "")
            if m:
                sv = float(m.group(1))
                # Run Line without sign is always underdog = +1.5
                if mkt == "Run Line" and sv > 0 and "+" not in (m.group(1)):
                    sv = abs(sv)  # keep positive (underdog)
                _by_bet[bid].append(sv)

        for bid, vals in _by_bet.items():
            avg_sv  = sum(vals) / len(vals)
            avg_abs = sum(abs(v) for v in vals) / len(vals)
            avg_fav = sum(1.0 for v in vals if v < 0) / len(vals)
            _spread_lookup[bid] = (avg_sv, avg_abs, avg_fav)
    except Exception as _sle:
        log.warning("_build_personal_df: spread lookup failed: %s", _sle)

    rows = []
    for b in bets:
        f = extract_features(b)
        if f is None:
            continue
        age_days = (cutoff - b.time_placed).days if b.time_placed else 999
        weight = (
            WEIGHT_CONFIG["personal_recent"]
            if age_days <= WEIGHT_CONFIG["personal_recent_days"]
            else WEIGHT_CONFIG["personal_base"]
        )
        row: dict = {c: f.get(c, np.nan) for c in _PERSONAL_FEATURE_COLS}
        for c in HIST_EXTRA_COLS:
            row[c] = np.nan
        for c in NEW_LEG_FEATURE_COLS:
            row[c] = np.nan   # personal bets have no per-leg signal data
        # Spread features from bet_legs (avg across spread legs, 0.0 if none)
        _sp = _spread_lookup.get(b.id)
        if _sp:
            row["spread_value"]       = _sp[0]
            row["spread_abs"]         = _sp[1]
            row["is_favorite_spread"] = _sp[2]
        else:
            row["spread_value"]       = 0.0
            row["spread_abs"]         = 0.0
            row["is_favorite_spread"] = 0.0
        row["_label"]       = 1.0 if b.status == "SETTLED_WIN" else 0.0
        row["_weight"]      = weight
        row["_date"]        = b.time_placed or datetime(2020, 1, 1)
        row["_sport_label"] = (b.sports or "other").split("|")[0].strip()
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


_COVER_RATE_LOW  = 0.46
_COVER_RATE_HIGH = 0.54

def cover_rate_check(sport: str, df: "pd.DataFrame") -> bool:
    """
    Validate that a sport's ATS labels have a realistic cover rate before
    including them in training.

    A cover rate far from 50% signals a data problem (e.g. synthetic run lines
    where the favorite side is always assigned, or a systematic label error).
    Real markets should produce cover rates of 46-54% over large samples.

    Parameters
    ----------
    sport : e.g. "NFL", "NBA", "MLB"
    df    : DataFrame with a '_label' column (0/1 ATS labels)

    Returns
    -------
    True  — cover rate is healthy; safe to include in training
    False — cover rate is out of bounds; skip this sport with a warning
    """
    import pandas as _pd
    labeled = df["_label"].dropna()
    if len(labeled) < 50:
        log.warning("cover_rate_check %s: only %d labeled rows — skipping sport",
                    sport, len(labeled))
        return False

    rate = labeled.mean()
    if _COVER_RATE_LOW <= rate <= _COVER_RATE_HIGH:
        log.info("cover_rate_check %s: %.1f%% cover rate — OK (%d rows)",
                 sport, rate * 100, len(labeled))
        return True
    else:
        log.warning(
            "cover_rate_check %s: %.1f%% cover rate is OUTSIDE [%.0f%%, %.0f%%] "
            "— skipping sport (fix spread data before re-enabling)",
            sport, rate * 100, _COVER_RATE_LOW * 100, _COVER_RATE_HIGH * 100,
        )
        return False


def _load_historical_df() -> pd.DataFrame:
    """
    Load ATS-labeled rows from historical.db, mapped to FEATURE_COLS schema.
    Reads only HISTORICAL_ATS_SPORTS (currently NFL — the only sport with
    close_spread populated).  Add sports to HISTORICAL_ATS_SPORTS as more
    spread data is ingested.

    Extra metadata columns:
        _label        : covered_spread (1/0)
        _weight       : WEIGHT_CONFIG["historical_base"]
        _date         : game_date
        _sport_label  : human-readable sport name
    """
    try:
        from features import build_feature_matrix, TARGET_ATS
    except ImportError:
        log.warning("features.py not importable — skipping historical data")
        return pd.DataFrame()

    frames = []
    for sport in HISTORICAL_ATS_SPORTS:
        try:
            matrix = build_feature_matrix(sport)
        except Exception as exc:
            log.warning("historical load failed for %s: %s", sport, exc)
            continue

        labeled = matrix[matrix[TARGET_ATS].notna()].copy()
        if labeled.empty:
            log.info("historical: no ATS labels for %s", sport)
            continue

        meta = _SPORT_KEY_MAP.get(sport, {"sport_id": 6, "league_id": 10})
        SPORT_MAP_INV = {v: k for k, v in SPORT_MAP.items()}
        sport_label = SPORT_MAP_INV.get(meta["sport_id"], "other")

        rows = []
        for _, r in labeled.iterrows():
            dec_odds     = _american_to_decimal(r.get("close_ml_home"))
            imp_prob     = 1.0 / dec_odds
            log_odds_val = math.log(dec_odds)

            if   dec_odds < 2:   ob = 0
            elif dec_odds < 5:   ob = 1
            elif dec_odds < 15:  ob = 2
            elif dec_odds < 50:  ob = 3
            else:                ob = 4

            gd    = pd.to_datetime(r.get("game_date"))
            month = gd.month     if pd.notna(gd) else 6
            dow   = gd.dayofweek if pd.notna(gd) else 0

            row: dict = {
                # ── Personal-bet cols mapped from game context ────────────
                "legs":              1,
                "odds":              dec_odds,
                "log_odds":          log_odds_val,
                "implied_prob":      round(imp_prob, 6),
                "stake":             1.0,
                "is_parlay":         0,
                "sport_id":          meta["sport_id"],
                "league_id":         meta["league_id"],
                "hour_placed":       np.nan,   # unknown; imputer fills median
                "day_of_week":       dow,
                "ml_pct":            0.0,
                "spread_pct":        1.0,      # all historical rows are ATS bets
                "total_pct":         0.0,
                "prop_pct":          0.0,
                "multi_sport":       0,
                "n_sports":          1,
                "has_ev":            0,
                "ev_value":          0.0,
                "closing_line_diff": 0.0,      # CLV_READY
                "odds_bracket":      ob,
                "leg_diversity":     0.25,     # single market type
                "clv":               0.0,      # CLV_READY
                "clv_available":     0,
                "multi_league":      0,
                "n_leagues":         1,
                "prop_density":      0.0,
                "stake_ratio":       1.0 / 6.8,
                "month_sin":         math.sin(2 * math.pi * month / 12),
                "month_cos":         math.cos(2 * math.pi * month / 12),
                "hour_sin":          np.nan,   # unknown; imputer fills median
                "hour_cos":          np.nan,
                # ── Historical team-quality cols ──────────────────────────
                "margin_diff_10g":   float(r.get("margin_diff_10g",    np.nan)),
                "win_pct_diff_10g":  float(r.get("win_pct_diff_10g",  np.nan)),
                "form_diff_3g":      float(r.get("form_diff_3g",       np.nan)),
                "h_margin_10g":      float(r.get("h_margin_10g",       np.nan)),
                "a_margin_10g":      float(r.get("a_margin_10g",       np.nan)),
                "h_win_pct_10g":     float(r.get("h_win_pct_10g",     np.nan)),
                "a_win_pct_10g":     float(r.get("a_win_pct_10g",     np.nan)),
                "rest_advantage":    float(r.get("rest_advantage",     np.nan)),
                # New: momentum + SOS
                "stake_vs_kelly":    0.0,   # neutral for historical (no EV info)
                "days_since_win_diff":  float(r.get("days_since_win_diff",  np.nan)),
                "h_days_since_win":     float(r.get("h_days_since_win",     np.nan)),
                "a_days_since_win":     float(r.get("a_days_since_win",     np.nan)),
                "h_opp_impl_prob_5g":   float(r.get("h_opp_impl_prob_5g",  np.nan)),
                "a_opp_impl_prob_5g":   float(r.get("a_opp_impl_prob_5g",  np.nan)),
                "opp_strength_diff":    float(r.get("opp_strength_diff",   np.nan)),
                # Pi ratings (from features.py, populated for MLB)
                "home_pi":              float(r.get("home_pi",             np.nan)),
                "away_pi":              float(r.get("away_pi",             np.nan)),
                "pi_diff":              float(r.get("pi_diff",             np.nan)),
                "pi_confidence":        float(r.get("pi_confidence",       np.nan)),
                # ── Per-leg signals (NaN for historical game rows) ────────
                **{c: np.nan for c in NEW_LEG_FEATURE_COLS},
                # Override spread features from actual close_spread (all historical rows are ATS)
                "spread_value":       float(r.get("close_spread", 0)) if pd.notna(r.get("close_spread")) else 0.0,
                "spread_abs":         abs(float(r.get("close_spread", 0))) if pd.notna(r.get("close_spread")) else 0.0,
                "is_favorite_spread": 1.0 if pd.notna(r.get("close_spread")) and float(r.get("close_spread", 0)) < 0 else 0.0,
                # ── Metadata ──────────────────────────────────────────────
                "_label":       float(r[TARGET_ATS]),
                "_weight":      WEIGHT_CONFIG["historical_base"],
                "_date":        gd if pd.notna(gd) else datetime(2020, 1, 1),
                "_sport_label": sport_label,
            }
            rows.append(row)

        if rows:
            sport_df = pd.DataFrame(rows)
            if cover_rate_check(sport, sport_df):
                frames.append(sport_df)
                log.info("historical_df: %d labeled rows loaded for %s", len(rows), sport)
            else:
                log.warning("historical_df: %s excluded — failed cover_rate_check", sport)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_mock_leg_df(db: Session) -> pd.DataFrame:
    """
    Build labeled training rows from settled mock_bet_legs.

    This is the only data source that carries the new per-leg signals
    (injury, line movement, scout grade, CLV cents, spread value, recency).
    Personal bets and historical game rows have NaN for these columns.

    The open_odds IS NOT NULL filter keeps only legs with real market odds.
    Exploration/simulation legs now have open_odds backfilled from their
    source alt_lines/fixtures/player_props records, so they pass this filter.
    New exploration legs get open_odds set at creation time via _dec_to_american().

    Label  : leg_result = 'WIN' → 1, 'LOSS' → 0
    Weight : personal_base (3×) — same as older personal bets
    """
    from sqlalchemy import text as _text

    # Build recency lookup before iterating rows
    recency_lookup = _build_recency_features(db)

    # Pre-load park factor helper
    try:
        from mlb_features import get_park_features
    except ImportError:
        get_park_features = lambda h, s: {"park_runs_factor": 1.0, "park_hr_factor": 1.0}

    # Tier-A training fallback: local game_logs pitcher features (no API, no leak)
    try:
        from mlb_features import get_pitcher_features_from_gamelogs as _gl_pitcher
    except ImportError:
        _gl_pitcher = None
    _gl_cache: dict[tuple[str, str, str], dict] = {}  # (home, away, date) → features

    try:
        rows = db.execute(_text("""
            SELECT
                mbl.open_odds,
                mbl.close_odds,
                mbl.market_type,
                mbl.sport,
                mbl.is_alt_line,
                mbl.predicted_win_prob,
                mbl.predicted_edge_pp,
                mbl.injury_net_advantage,
                mbl.line_movement_signal,
                mbl.clv_cents,
                mbl.home_team_injury_impact,
                mbl.away_team_injury_impact,
                mbl.leg_result,
                mb.generated_at,
                CASE mbl.scout_grade
                    WHEN 'A' THEN 4 WHEN 'B' THEN 3
                    WHEN 'C' THEN 2 WHEN 'D' THEN 1
                    ELSE NULL END AS scout_grade_encoded,
                mbl.description,
                mbl.resolved_home_team,
                mbl.resolved_away_team,
                mb.game_date,
                mbl.home_team_home_pi,
                mbl.away_team_away_pi,
                mbl.pi_diff,
                mbl.spread_value,
                mbl.spread_abs,
                mbl.is_favorite_spread,
                mbl.home_starter_era,
                mbl.home_starter_whip,
                mbl.home_starter_k9,
                mbl.home_starter_hr9,
                mbl.away_starter_era,
                mbl.away_starter_whip,
                mbl.away_starter_k9,
                mbl.away_starter_hr9,
                mbl.starter_era_diff,
                mbl.starter_whip_diff,
                mbl.starter_k9_diff,
                mbl.game_temp_f,
                mbl.game_wind_mph,
                mbl.wind_helps_scoring,
                mbl.home_team_xwoba,
                mbl.away_team_xwoba,
                mbl.xwoba_diff,
                mbl.home_team_barrel_pct,
                mbl.away_team_barrel_pct,
                mbl.barrel_diff,
                mbl.home_bullpen_era,
                mbl.away_bullpen_era,
                mbl.bullpen_era_diff,
                mbl.h2h_home_wr,
                mbl.h2h_n_games
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mbl.open_odds IS NOT NULL
        """)).fetchall()
    except Exception as exc:
        log.warning("_build_mock_leg_df: query failed: %s", exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        open_odds_am  = r[0]
        close_odds_am = r[1]
        market_type   = (r[2] or "").lower()
        sport_raw     = r[3] or ""
        pred_wp       = r[5]   # 0–100
        injury_net    = r[7]
        lms           = r[8]
        clv_c         = r[9]
        h_injury      = r[10]
        a_injury      = r[11]
        leg_result    = r[12]
        generated_at  = r[13]
        scout_enc     = r[14]  # 0–4
        description   = r[15] or ""
        home_team     = (r[16] or "").lower().strip()
        away_team     = (r[17] or "").lower().strip()
        game_date     = r[18] or ""
        _pi_home      = r[19]   # home_team_home_pi (may be None)
        _pi_away      = r[20]   # away_team_away_pi (may be None)
        _pi_d         = r[21]   # pi_diff (may be None)
        _db_spread_val = r[22]  # spread_value from DB (may be None)
        _db_spread_abs = r[23]  # spread_abs from DB (may be None)
        _db_is_fav     = r[24]  # is_favorite_spread from DB (may be None)
        # MLB enrichment features (indices 25-49, may be None)
        _db_home_era   = r[25]; _db_home_whip  = r[26]; _db_home_k9   = r[27]; _db_home_hr9  = r[28]
        _db_away_era   = r[29]; _db_away_whip  = r[30]; _db_away_k9   = r[31]; _db_away_hr9  = r[32]
        _db_era_diff   = r[33]; _db_whip_diff  = r[34]; _db_k9_diff   = r[35]
        _db_temp       = r[36]; _db_wind       = r[37]; _db_wind_help = r[38]
        _db_h_xwoba    = r[39]; _db_a_xwoba    = r[40]; _db_xwoba_d   = r[41]
        _db_h_barrel   = r[42]; _db_a_barrel   = r[43]; _db_barrel_d  = r[44]
        _db_h_bullpen  = r[45]; _db_a_bullpen  = r[46]; _db_bullpen_d = r[47]
        _db_h2h_wr     = r[48]; _db_h2h_n      = r[49]

        dec_odds  = _american_to_decimal(open_odds_am)
        dec_close = _american_to_decimal(close_odds_am) if close_odds_am else None

        imp_prob     = 1.0 / dec_odds
        log_odds_val = math.log(dec_odds) if dec_odds > 0 else 0.0

        if   dec_odds < 2:   ob = 0
        elif dec_odds < 5:   ob = 1
        elif dec_odds < 15:  ob = 2
        elif dec_odds < 50:  ob = 3
        else:                ob = 4

        # Market-type fractions (single-leg → one type = 1.0)
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

        # ── Tier-A training fallback: pitcher features from game_logs ──
        # Only for MLB legs missing pitcher data. No API calls, no leak.
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
            if _gl.get("home_starter_era") is not None:
                v = _gl["home_starter_era"]
                if v == v: _db_home_era = v  # not NaN
            if _gl.get("away_starter_era") is not None:
                v = _gl["away_starter_era"]
                if v == v: _db_away_era = v
            if _gl.get("home_starter_whip") is not None:
                v = _gl["home_starter_whip"]
                if v == v: _db_home_whip = v
            if _gl.get("away_starter_whip") is not None:
                v = _gl["away_starter_whip"]
                if v == v: _db_away_whip = v
            if _gl.get("home_starter_k9") is not None:
                v = _gl["home_starter_k9"]
                if v == v: _db_home_k9 = v
            if _gl.get("away_starter_k9") is not None:
                v = _gl["away_starter_k9"]
                if v == v: _db_away_k9 = v
            if _gl.get("starter_era_diff") is not None:
                v = _gl["starter_era_diff"]
                if v == v: _db_era_diff = v
            if _gl.get("starter_whip_diff") is not None:
                v = _gl["starter_whip_diff"]
                if v == v: _db_whip_diff = v
            if _gl.get("starter_k9_diff") is not None:
                v = _gl["starter_k9_diff"]
                if v == v: _db_k9_diff = v

        # ── Spread-aware features (prefer stored DB values, fallback to parsing) ─
        if _db_spread_val is not None:
            sv            = float(_db_spread_val)
            spread_abs    = float(_db_spread_abs) if _db_spread_abs is not None else abs(sv)
            is_fav_spread = float(_db_is_fav)     if _db_is_fav     is not None else (1.0 if sv < 0 else 0.0)
        else:
            sv            = _extract_spread_value(description, market_type)
            spread_abs    = abs(sv)
            is_fav_spread = 1.0 if sv < 0 else 0.0

        # ── Recency features ─────────────────────────────────────────────
        # Determine team from description
        desc_lower = description.lower()
        _team = None
        if home_team and home_team in desc_lower:
            _team = home_team
        elif away_team and away_team in desc_lower:
            _team = away_team
        rec = recency_lookup.get((_team, market_type), {}) if _team else {}

        record: dict = {
            "legs":              1,
            "odds":              dec_odds,
            "log_odds":          log_odds_val,
            "implied_prob":      round(imp_prob, 6),
            "stake":             10.0,
            "is_parlay":         0,
            "sport_id":          SPORT_MAP.get(sport_raw, 6),
            "league_id":         10,
            "hour_placed":       hour,
            "day_of_week":       dow,
            "ml_pct":            ml_pct,
            "spread_pct":        spread_pct,
            "total_pct":         total_pct,
            "prop_pct":          prop_pct,
            "multi_sport":       0,
            "n_sports":          1,
            "has_ev":            1,
            "ev_value":          round(ev_val, 6),
            "closing_line_diff": closing_line_diff,
            "odds_bracket":      ob,
            "leg_diversity":     0.25,
            "clv":               clv_val,
            "clv_available":     clv_avail,
            "multi_league":      0,
            "n_leagues":         1,
            "prop_density":      1.0 if "player" in market_type else 0.0,
            "stake_ratio":       10.0 / 6.8,
            "month_sin":         math.sin(2 * math.pi * month / 12),
            "month_cos":         math.cos(2 * math.pi * month / 12),
            "hour_sin":          math.sin(2 * math.pi * hour / 24),
            "hour_cos":          math.cos(2 * math.pi * hour / 24),
            "stake_vs_kelly":    0.0,
            # HIST_EXTRA_COLS — game-level stats not available at leg grain
            **{c: np.nan for c in HIST_EXTRA_COLS},
            # Override Pi features from stored values (backfilled on mock_bet_legs)
            "home_pi":        float(_pi_home) if _pi_home is not None else np.nan,
            "away_pi":        float(_pi_away) if _pi_away is not None else np.nan,
            "pi_diff":        float(_pi_d)    if _pi_d    is not None else np.nan,
            "pi_confidence":  1.0 if _pi_home is not None else np.nan,
            # NEW_LEG_FEATURE_COLS — original signals
            "injury_net_advantage":    float(injury_net) if injury_net is not None else np.nan,
            "line_movement_signal":    float(lms)        if lms       is not None else np.nan,
            "scout_grade_encoded":     float(scout_enc)  if scout_enc is not None else np.nan,
            "clv_cents":               float(clv_c)      if clv_c     is not None else np.nan,
            "home_team_injury_impact": float(h_injury)   if h_injury  is not None else np.nan,
            "away_team_injury_impact": float(a_injury)   if a_injury  is not None else np.nan,
            # Spread-aware features
            "spread_value":            sv,
            "spread_abs":              spread_abs,
            "is_favorite_spread":      is_fav_spread,
            # Recency features
            "team_recent_cover_rate":  float(rec.get("cover_rate", np.nan)) if rec else np.nan,
            "team_recent_wr":          float(rec.get("recent_wr",  np.nan)) if rec else np.nan,
            "team_form_trend":         float(rec.get("form_trend", np.nan)) if rec else np.nan,
            # Park factors (static lookup — always available for MLB)
            **get_park_features(home_team or "", sport_raw),
            # MLB enrichment — read stored values from mock_bet_legs (NaN if not yet populated)
            "home_starter_era":    float(_db_home_era)   if _db_home_era   is not None else np.nan,
            "home_starter_whip":   float(_db_home_whip)  if _db_home_whip  is not None else np.nan,
            "home_starter_k9":     float(_db_home_k9)    if _db_home_k9    is not None else np.nan,
            "home_starter_hr9":    float(_db_home_hr9)   if _db_home_hr9   is not None else np.nan,
            "away_starter_era":    float(_db_away_era)   if _db_away_era   is not None else np.nan,
            "away_starter_whip":   float(_db_away_whip)  if _db_away_whip  is not None else np.nan,
            "away_starter_k9":     float(_db_away_k9)    if _db_away_k9    is not None else np.nan,
            "away_starter_hr9":    float(_db_away_hr9)   if _db_away_hr9   is not None else np.nan,
            "starter_era_diff":    float(_db_era_diff)   if _db_era_diff   is not None else np.nan,
            "starter_whip_diff":   float(_db_whip_diff)  if _db_whip_diff  is not None else np.nan,
            "starter_k9_diff":     float(_db_k9_diff)    if _db_k9_diff    is not None else np.nan,
            "game_temp_f":         float(_db_temp)       if _db_temp       is not None else np.nan,
            "game_wind_mph":       float(_db_wind)       if _db_wind       is not None else np.nan,
            "wind_helps_scoring":  float(_db_wind_help)  if _db_wind_help  is not None else np.nan,
            "home_team_xwoba":     float(_db_h_xwoba)    if _db_h_xwoba    is not None else np.nan,
            "away_team_xwoba":     float(_db_a_xwoba)    if _db_a_xwoba    is not None else np.nan,
            "xwoba_diff":          float(_db_xwoba_d)    if _db_xwoba_d    is not None else np.nan,
            "home_team_barrel_pct":float(_db_h_barrel)   if _db_h_barrel   is not None else np.nan,
            "away_team_barrel_pct":float(_db_a_barrel)   if _db_a_barrel   is not None else np.nan,
            "barrel_diff":         float(_db_barrel_d)   if _db_barrel_d   is not None else np.nan,
            "home_bullpen_era":    float(_db_h_bullpen)  if _db_h_bullpen  is not None else np.nan,
            "away_bullpen_era":    float(_db_a_bullpen)  if _db_a_bullpen  is not None else np.nan,
            "bullpen_era_diff":    float(_db_bullpen_d)  if _db_bullpen_d  is not None else np.nan,
            "h2h_home_wr":         float(_db_h2h_wr)     if _db_h2h_wr     is not None else np.nan,
            "h2h_n_games":         float(_db_h2h_n)      if _db_h2h_n      is not None else np.nan,
            # Metadata
            "_label":       1.0 if leg_result == "WIN" else 0.0,
            "_weight":      float(WEIGHT_CONFIG["personal_base"]),
            "_date":        gdt if gdt else datetime(2020, 1, 1),
            "_sport_label": sport_raw,
        }
        records.append(record)

    df = pd.DataFrame(records)
    log.info("_build_mock_leg_df: %d settled mock legs loaded for training", len(df))
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_clf(algorithm: str, calibrate: bool = True):
    if algorithm == "hist_gradient_boost":
        # HistGBT handles NaN natively and is scale-invariant.
        # Wrapped in CalibratedClassifierCV(isotonic) for better probability
        # calibration (HistGBT's raw Platt probabilities can be overconfident).
        base = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=20, random_state=42,
        )
        return CalibratedClassifierCV(base, cv=5, method="isotonic")
    base = {
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=5, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42),
        "logistic_regression": LogisticRegression(
            max_iter=2000, C=0.5, random_state=42),
    }.get(algorithm, GradientBoostingClassifier(n_estimators=300, random_state=42))
    if calibrate and algorithm != "logistic_regression":
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")
    return base


def _importances(clf, cols: list[str],
                 X_val=None, y_val=None) -> dict:
    """
    Extract feature importances from a fitted estimator or CalibratedClassifierCV.

    GBC path  : MDI feature_importances_ averaged across cv folds.
    HistGBT path (direct or inside CalibratedCV): permutation importance on a
    500-sample validation slice — sklearn 1.2 HistGBT has no feature_importances_.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _perm(estimator, X, y, feature_cols):
        if X is None or y is None or len(y) == 0:
            return {}
        try:
            from sklearn.inspection import permutation_importance
            n   = min(len(y), 500)
            idx = np.random.default_rng(42).choice(len(y), n, replace=False)
            r   = permutation_importance(
                estimator, X[idx], y[idx],
                n_repeats=10, random_state=42, n_jobs=1,
            )
            return {c: round(float(v), 4) for c, v in zip(feature_cols, r.importances_mean)}
        except Exception as exc:
            log.warning("_importances: permutation_importance failed: %s", exc)
            return {}

    # ── CalibratedClassifierCV wrapper ────────────────────────────────────────
    if hasattr(clf, "calibrated_classifiers_"):
        # GBC path: average MDI importances across cv folds
        all_imp = []
        for cal_clf in clf.calibrated_classifiers_:
            inner = getattr(cal_clf, "estimator", None)
            if inner is not None and hasattr(inner, "feature_importances_"):
                all_imp.append(inner.feature_importances_)
        if all_imp:
            mean_imp = np.mean(all_imp, axis=0)
            return {c: round(float(v), 4) for c, v in zip(cols, mean_imp)}

        # HistGBT-inside-CalibratedCV: permutation importance on first fold estimator
        first_inner = getattr(clf.calibrated_classifiers_[0], "estimator", None)
        if first_inner is not None and isinstance(first_inner, HistGradientBoostingClassifier):
            return _perm(first_inner, X_val, y_val, cols)
        return {}

    # ── Direct estimator ──────────────────────────────────────────────────────
    base = getattr(clf, "estimator", clf)
    if isinstance(base, HistGradientBoostingClassifier):
        return _perm(base, X_val, y_val, cols)
    if hasattr(base, "feature_importances_"):
        return {c: round(float(v), 4) for c, v in zip(cols, base.feature_importances_)}
    if hasattr(base, "coef_"):
        coef = np.abs(base.coef_[0])
        return {c: round(float(v / coef.sum()), 4) for c, v in zip(cols, coef)}
    return {}


_MODEL_CACHE: tuple | None = None   # cached (clf, scaler, imputer, active_cols) — None = not yet loaded
_MODEL_LOAD_FAILED: bool   = False  # True after a failed load (skip future attempts)
import threading as _threading
_MODEL_LOCK = _threading.Lock()

def load_model():
    """Return (clf, scaler, imputer, active_cols).

    active_cols is a list[str] of feature columns actually used by the model.
    Falls back to FEATURE_COLS if the saved list isn't available.
    Callers that unpack 3 values still work — active_cols is the 4th element.
    """
    global _MODEL_CACHE, _MODEL_LOAD_FAILED
    # Fast path — already loaded or already failed
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if _MODEL_LOAD_FAILED:
        return None, None, None, None
    # Slow path — serialize so only one thread attempts the pickle load
    with _MODEL_LOCK:
        # Re-check after acquiring lock (another thread may have loaded it)
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE
        if _MODEL_LOAD_FAILED:
            return None, None, None, None
        if not os.path.exists(MODEL_PATH):
            _MODEL_LOAD_FAILED = True
            return None, None, None, None
        try:
            with open(MODEL_PATH,  "rb") as f: clf     = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: scaler  = pickle.load(f)
            imputer = None
            if os.path.exists(IMPUTER_PATH):
                with open(IMPUTER_PATH, "rb") as f: imputer = pickle.load(f)
            active_cols = None
            if os.path.exists(ACTIVE_COLS_PATH):
                with open(ACTIVE_COLS_PATH, "rb") as f: active_cols = pickle.load(f)
            _MODEL_CACHE = (clf, scaler, imputer, active_cols)
            return _MODEL_CACHE
        except Exception as _e:
            print(f"[load_model] WARNING: global model load failed ({_e}). Returning None.")
            _MODEL_LOAD_FAILED = True
            return None, None, None, None


def _load_submodel(sport: str):
    # Legacy path — kept for backwards-compat with any manually placed files.
    path = os.path.join(SUBMODEL_DIR, f"{sport.lower().replace(' ', '_')}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Sport key normalization ──────────────────────────────────────────────────
# Callers may pass any of:  "baseball_mlb", "mlb", "MLB", "Baseball"
# Model files use lowercase short form ("mlb"), feature matrix / team_stats
# use uppercase short form ("MLB").
_SPORT_TO_CANONICAL: dict[str, str] = {
    "baseball_mlb": "MLB", "mlb": "MLB", "baseball": "MLB",
    "icehockey_nhl": "NHL", "nhl": "NHL", "ice hockey": "NHL",
    "basketball_nba": "NBA", "nba": "NBA", "basketball": "NBA",
    "americanfootball_nfl": "NFL", "nfl": "NFL", "american football": "NFL",
}


def _canonical_sport(sport: str) -> str:
    """Normalize any sport key variant to the canonical uppercase form (MLB, NHL, …)."""
    return _SPORT_TO_CANONICAL.get(sport.lower().strip(), sport)


def _load_ats_submodel(sport: str):
    """
    Load the diagnostic ATS sub-model artifacts for *sport*.
    Returns (clf, scaler, imputer, feat_cols) or None if not available.
    """
    prefix = _canonical_sport(sport).lower()
    paths = {
        "clf":   os.path.join(SUBMODEL_DIR, f"{prefix}_ats_clf.pkl"),
        "sc":    os.path.join(SUBMODEL_DIR, f"{prefix}_ats_scaler.pkl"),
        "imp":   os.path.join(SUBMODEL_DIR, f"{prefix}_ats_imputer.pkl"),
        "cols":  os.path.join(SUBMODEL_DIR, f"{prefix}_ats_feat_cols.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    try:
        with open(paths["clf"],  "rb") as f: clf       = pickle.load(f)
        with open(paths["sc"],   "rb") as f: sc        = pickle.load(f)
        with open(paths["imp"],  "rb") as f: imp       = pickle.load(f)
        with open(paths["cols"], "rb") as f: feat_cols = pickle.load(f)
    except Exception as _e:
        print(f"[_load_ats_submodel] WARNING: {sport} ATS model load failed ({_e}). Sub-model unavailable.")
        return None
    return clf, sc, imp, feat_cols


# ── Feature-matrix cache (keyed by sport, TTL = 6 hours) ─────────────────────
# Raised from 1h → 6h: the MLB matrix build takes 10-14s clean but can blow up
# to 30+ minutes under GIL contention from concurrent HTTP threads.  Historical
# data changes at most once daily (nightly ETL) so a 6-hour TTL is safe and
# avoids a second expensive rebuild during the PM mock-gen pass.
_FM_CACHE: dict = {}          # sport → (timestamp, DataFrame)
_FM_CACHE_TTL = 21600         # 6 hours


def _get_feature_matrix(sport: str):
    """Return cached feature matrix for sport, rebuilding if stale.

    MLB reverted to build_feature_matrix (V1 builder) — V1 submodel expects
    season-level pitcher/batting stats that V2 builder doesn't produce.
    V2 builder kept available for shadow/retrain use only.
    """
    import time
    now = time.time()
    if sport in _FM_CACHE:
        ts, df = _FM_CACHE[sport]
        if now - ts < _FM_CACHE_TTL:
            return df
    from features import build_feature_matrix
    df = build_feature_matrix(sport)
    _FM_CACHE[sport] = (now, df)
    return df


# NHL full name → 3-letter abbreviation (used in historical.db)
_NHL_NAME_TO_CODE: dict[str, str] = {
    "anaheim ducks":           "ANA",
    "arizona coyotes":         "ARI",
    "utah hockey club":        "UTA",
    "boston bruins":           "BOS",
    "buffalo sabres":          "BUF",
    "carolina hurricanes":     "CAR",
    "columbus blue jackets":   "CBJ",
    "calgary flames":          "CGY",
    "chicago blackhawks":      "CHI",
    "colorado avalanche":      "COL",
    "dallas stars":            "DAL",
    "detroit red wings":       "DET",
    "edmonton oilers":         "EDM",
    "florida panthers":        "FLA",
    "los angeles kings":       "LAK",
    "minnesota wild":          "MIN",
    "montreal canadiens":      "MTL",
    "new jersey devils":       "NJD",
    "nashville predators":     "NSH",
    "new york islanders":      "NYI",
    "new york rangers":        "NYR",
    "ottawa senators":         "OTT",
    "philadelphia flyers":     "PHI",
    "pittsburgh penguins":     "PIT",
    "seattle kraken":          "SEA",
    "san jose sharks":         "SJS",
    "st. louis blues":         "STL",
    "tampa bay lightning":     "TBL",
    "toronto maple leafs":     "TOR",
    "vancouver canucks":       "VAN",
    "vegas golden knights":    "VGK",
    "winnipeg jets":           "WPG",
    "washington capitals":     "WSH",
}


def _normalize_team_name(sport: str, name: str) -> str:
    """Convert full team name to the format stored in historical.db."""
    if sport == "NHL":
        return _NHL_NAME_TO_CODE.get(name.lower().strip(), name)
    return name   # MLB and others use full names


def predict_game_ats(
    sport:        str,
    home_team:    str,
    away_team:    str,
    game_date:    str | None = None,
    spread_value: float | None = None,
    total_line:   float | None = None,
) -> float | None:
    """
    Predict the probability that the home team covers the spread for a game,
    using the diagnostic ATS sub-model for *sport*.

    Parameters
    ----------
    spread_value : The spread line from the home team's perspective (e.g. -1.5
                   means home is favored by 1.5).  Required for spread-aware
                   models (MLB); ignored for non-augmented models (NHL).
                   If the model expects spread features but spread_value is
                   None, falls back to the standard run-line (-1.5 for MLB).

    Returns a float in [0, 1] or None if:
      - No sub-model exists for this sport
      - No matching game found in the feature matrix

    The prediction is based on rolling team stats, pitcher ERA, etc. — NOT
    the global bet-slip model features.
    """
    sport = _canonical_sport(sport)

    artifacts = _load_ats_submodel(sport)
    if artifacts is None:
        return None
    clf, sc, imp, feat_cols = artifacts

    # ── V2 guard: if V2 submodel (has _spread_sq) is accidentally loaded,
    #    skip it — miscalibrated isotonic on augmented spreads.
    #    V1 submodel (has _is_fav_spread, no _spread_sq) is safe to use.
    if sport == "MLB" and "_spread_sq" in feat_cols:
        return None

    # Does this model include spread augmentation features?
    _spread_aware = "_spread_value" in feat_cols

    try:
        matrix = _get_feature_matrix(sport)
        if matrix.empty:
            return None

        # Normalise team names to the format stored in historical.db
        ht = _normalize_team_name(sport, home_team).lower().strip()
        at = _normalize_team_name(sport, away_team).lower().strip()
        mask = (
            matrix["home_team"].str.lower().str.strip() == ht
        ) & (
            matrix["away_team"].str.lower().str.strip() == at
        )
        if game_date:
            date_str = str(game_date)[:10]
            mask = mask & (matrix["game_date"].astype(str).str[:10] == date_str)

        matched = matrix[mask]
        if matched.empty:
            return None

        # Use the most recent matching row (today's / closest future game)
        row = matched.sort_values("game_date", ascending=False).iloc[0]

        # Resolve spread_value for spread-aware models
        _sv = spread_value
        if _spread_aware and _sv is None:
            _sv = -1.5 if sport == "MLB" else -1.5  # default run line

        def _val(col: str) -> float:
            # Spread augmentation features — injected at inference time
            if col == "_spread_value":
                return float(_sv) if _sv is not None else np.nan
            if col == "_spread_abs":
                return abs(float(_sv)) if _sv is not None else np.nan
            if col == "_spread_sq":
                return float(_sv) ** 2 if _sv is not None else np.nan
            if col == "_is_fav_spread":
                return 1.0 if (_sv is not None and _sv < 0) else 0.0
            # For future NHL games, ot_so_game is unknown — explicitly use 0
            # (regulation finish assumption) rather than the imputer mean (~0.25).
            # The model learned regulation = more decisive puck-line; 0.25 would
            # hedge toward OT and understate cover probability for clear favourites.
            if col == "ot_so_game" and sport == "NHL":
                return 0.0
            return row[col] if col in row.index and not pd.isna(row[col]) else np.nan

        X = np.array([[_val(c) for c in feat_cols]], dtype=float)
        X_imp = imp.transform(X)
        X_s   = sc.transform(X_imp)
        raw_prob = float(clf.predict_proba(X_s)[0][1])
        # Both V1 and V2 ATS models use CalibratedClassifierCV (isotonic).
        # Skip the external Platt calibrator to avoid double-calibration.
        return raw_prob

    except Exception as exc:
        log.warning("predict_game_ats(%s, %s vs %s, spread=%s): %s",
                    sport, home_team, away_team, spread_value, exc)
        return None


# ── Soccer sub-model ──────────────────────────────────────────────────────────

def _load_soccer_submodel(market: str):
    """
    Load soccer market-specific pkl artifacts.
    market: 'total_goals' | 'moneyline'
    Returns (clf, imputer, scaler, feat_cols) or None.
    """
    prefix = os.path.join(SUBMODEL_DIR, f"soccer_{market}")
    paths = {
        "clf":  f"{prefix}_clf.pkl",
        "imp":  f"{prefix}_imputer.pkl",
        "sc":   f"{prefix}_scaler.pkl",
        "cols": f"{prefix}_feat_cols.pkl",
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    try:
        with open(paths["clf"],  "rb") as f: clf       = pickle.load(f)
        with open(paths["imp"],  "rb") as f: imp       = pickle.load(f)
        with open(paths["sc"],   "rb") as f: sc        = pickle.load(f)
        with open(paths["cols"], "rb") as f: feat_cols = pickle.load(f)
    except Exception as _e:
        print(f"[_load_soccer_submodel] WARNING: soccer {market} model load failed ({_e}).")
        return None
    return clf, imp, sc, feat_cols


def _soccer_team_form(team_name: str, as_of_date: str) -> dict:
    """
    Compute rolling form stats for a team as of a given date.
    Queries soccer_results from bets.db directly (no SQLAlchemy dependency).
    Returns dict of form features or empty dict if no data found.
    """
    import re as _re
    import unicodedata as _uc
    import sqlite3 as _sq

    _BETS_DB = os.path.join(DATA_DIR, "bets.db")
    if not os.path.exists(_BETS_DB):
        return {}

    _STROKE_TRANS = str.maketrans({
        'ø': 'o', 'Ø': 'o', 'ł': 'l', 'Ł': 'l', 'ð': 'd',
        'þ': 'th', 'æ': 'ae', 'œ': 'oe', 'ß': 'ss',
    })
    _ALIASES = {
        "man city": "manchester city", "man utd": "manchester united",
        "paris st-g": "paris saint-germain", "lyon": "olympique lyonnais",
        "inter": "internazionale", "inter milan": "internazionale milan",
        "dortmund": "borussia dortmund", "leverkusen": "bayer leverkusen",
        "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
        "newcastle": "newcastle united", "wolves": "wolverhampton wanderers",
        "sheff utd": "sheffield united", "nottm forest": "nottingham forest",
        "brighton": "brighton hove albion",
    }
    _SUFFIX_RE = _re.compile(
        r"\b(fc|sc|cf|ac|as|rc|ud|fk|sk|bk|sporting|club|de|del|la|el)\b",
        _re.IGNORECASE,
    )

    def _norm(n: str) -> str:
        s = _uc.normalize("NFKD", n)
        s = "".join(c for c in s if not _uc.combining(c))
        s = s.translate(_STROKE_TRANS).lower().strip()
        s = _ALIASES.get(s, s)
        s = _SUFFIX_RE.sub("", s).replace("/", " ")
        s = s.replace("munchen", "munich")
        return _re.sub(r"\s+", " ", s).strip()

    def _match(needle: str, candidate: str) -> bool:
        n, c = _norm(needle), _norm(candidate)
        if n == c:
            return True
        nw, cw = set(n.split()), set(c.split())
        if not nw or not cw:
            return False
        shorter, longer = (nw, cw) if len(nw) <= len(cw) else (cw, nw)
        if len(shorter) == 1:
            # Single-word match only works if candidate has ≤2 words.
            # Prevents "Barcelona" from matching "RCD Espanyol de Barcelona".
            if len(longer) > 2:
                return False
            return next(iter(shorter)) in longer
        return shorter.issubset(longer)

    try:
        con = _sq.connect(_BETS_DB)
        cur = con.execute(
            "SELECT date, home_team, away_team, home_goals, away_goals "
            "FROM soccer_results "
            "WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL "
            "AND date < ? ORDER BY date DESC LIMIT 800",
            (as_of_date,),
        )
        all_rows = cur.fetchall()
        con.close()
    except Exception:
        return {}

    matches = []
    for date, ht, at, hg, ag in all_rows:
        is_home = _match(team_name, ht)
        is_away = _match(team_name, at)
        if not is_home and not is_away:
            continue
        gf = hg if is_home else ag
        ga = ag if is_home else hg
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        matches.append({"date": date, "is_home": is_home, "gf": gf, "ga": ga,
                         "result": result, "total": gf + ga})
        if len(matches) >= 10:
            break

    if not matches:
        return {}

    def _form_stats(ms: list[dict]) -> dict:
        if not ms:
            return {}
        form = "".join(m["result"] for m in ms)
        return {
            "form": form,
            "wr": ms[0]["result"].count("W") / len(ms) if ms else 0,  # crude
            "wr_5":  sum(1 for m in ms[:5] if m["result"] == "W") / max(len(ms[:5]), 1),
            "wr_10": sum(1 for m in ms[:10] if m["result"] == "W") / max(len(ms[:10]), 1),
            "unbeaten_5": sum(1 for m in ms[:5] if m["result"] != "L") / max(len(ms[:5]), 1),
            "gf_avg_5":  sum(m["gf"] for m in ms[:5]) / max(len(ms[:5]), 1),
            "ga_avg_5":  sum(m["ga"] for m in ms[:5]) / max(len(ms[:5]), 1),
            "gf_avg_10": sum(m["gf"] for m in ms[:10]) / max(len(ms[:10]), 1),
            "ga_avg_10": sum(m["ga"] for m in ms[:10]) / max(len(ms[:10]), 1),
            "over25_r10": sum(1 for m in ms[:10] if m["total"] > 2.5) / max(len(ms[:10]), 1),
            "home_wr_5": (sum(1 for m in ms[:5] if m["is_home"] and m["result"] == "W")
                          / max(sum(1 for m in ms[:5] if m["is_home"]), 1)),
            "away_wr_5": (sum(1 for m in ms[:5] if not m["is_home"] and m["result"] == "W")
                          / max(sum(1 for m in ms[:5] if not m["is_home"]), 1)),
        }

    return _form_stats(matches)


def score_soccer_leg(leg: dict) -> dict | None:
    """
    Score a soccer leg using market-specific sub-models.

    Dispatches to:
      soccer_total_goals_clf  → h2h/totals Over/Under
      soccer_moneyline_clf    → h2h/moneyline + double_chance

    Returns dict with win_prob, model_used, or None to fall back to combined_v1.

    leg dict keys used:
      market: 'h2h' | 'totals' | 'spreads'
      pick: team name or "Over"/"Under"
      home_team, away_team
      game_date (YYYY-MM-DD)
      point (line for totals)
    """
    import re as _re

    market   = leg.get("market", "")
    sport    = leg.get("sport", "")
    home     = leg.get("home_team", "") or ""
    away     = leg.get("away_team", "") or ""
    pick_val = (leg.get("pick") or "").strip()
    today    = (leg.get("game_date") or "")[:10] or str(datetime.utcnow().date())

    if not home or not away:
        return None

    # ── Market routing ──────────────────────────────────────────────────────
    if market in ("totals", "alternate_totals"):
        market_key = "total_goals"
    elif market in ("h2h", "moneyline"):
        market_key = "moneyline"
    else:
        return None

    artifacts = _load_soccer_submodel(market_key)
    if artifacts is None:
        return None
    clf, imp, sc, feat_cols = artifacts

    # ── Feature computation ──────────────────────────────────────────────────
    hf = _soccer_team_form(home, today)
    af = _soccer_team_form(away, today)

    if not hf:
        return None  # No form data → graceful degradation

    if market_key == "total_goals":
        point = leg.get("point")
        direction = 1 if pick_val.lower() == "over" else (0 if pick_val.lower() == "under" else None)
        if direction is None or point is None:
            return None

        feature_dict = {
            "line":              float(point),
            "direction":         direction,
            "home_gf_avg_5":     hf.get("gf_avg_5"),
            "home_ga_avg_5":     hf.get("ga_avg_5"),
            "home_gf_avg_10":    hf.get("gf_avg_10"),
            "home_ga_avg_10":    hf.get("ga_avg_10"),
            "home_over25_r10":   hf.get("over25_r10"),
            "home_form_wr_5":    hf.get("wr_5"),
            "away_gf_avg_5":     af.get("gf_avg_5") if af else np.nan,
            "away_ga_avg_5":     af.get("ga_avg_5") if af else np.nan,
            "away_gf_avg_10":    af.get("gf_avg_10") if af else np.nan,
            "away_ga_avg_10":    af.get("ga_avg_10") if af else np.nan,
            "away_over25_r10":   af.get("over25_r10") if af else np.nan,
            "away_form_wr_5":    af.get("wr_5") if af else np.nan,
            "combined_goals_exp_5": (hf.get("gf_avg_5", 0) + (af.get("gf_avg_5", 0) if af else hf.get("gf_avg_5", 0))),
            "avg_over25_rate":   ((hf.get("over25_r10", 0.5) + (af.get("over25_r10", 0.5) if af else 0.5)) / 2),
        }
        model_name = "soccer_total_v1"

    else:  # moneyline
        pick_lower = pick_val.lower()
        if pick_lower in (home.lower(), _re.sub(r"\s+", "", home).lower()):
            is_home = 1.0
            pf, of = hf, af
        elif pick_lower in (away.lower(), _re.sub(r"\s+", "", away).lower()):
            is_home = 0.0
            pf, of = af if af else hf, hf
        else:
            is_home = 0.5
            pf, of = hf, af

        if not pf:
            return None

        # League home win rate prior
        _LEAGUE_HWR = {
            "epl": 0.44, "english premier league": 0.44,
            "la liga": 0.46, "bundesliga": 0.46,
            "ligue 1": 0.45, "ligue1": 0.45,
            "serie a": 0.46, "ucl": 0.48,
            "uefa champions league": 0.48,
        }
        lg_key = (leg.get("sport", "") or "").lower()
        league_hwr = _LEAGUE_HWR.get(lg_key, 0.45)

        feature_dict = {
            "is_home":         is_home,
            "is_double_chance": 0,
            "pick_wr_5":       pf.get("wr_5"),
            "pick_wr_10":      pf.get("wr_10"),
            "pick_unbeaten_5": pf.get("unbeaten_5"),
            "pick_gf_avg_5":   pf.get("gf_avg_5"),
            "pick_ga_avg_5":   pf.get("ga_avg_5"),
            "pick_gf_avg_10":  pf.get("gf_avg_10"),
            "pick_ga_avg_10":  pf.get("ga_avg_10"),
            "pick_home_wr_5":  pf.get("home_wr_5"),
            "pick_away_wr_5":  pf.get("away_wr_5"),
            "opp_wr_5":        of.get("wr_5") if of else np.nan,
            "opp_gf_avg_5":    of.get("gf_avg_5") if of else np.nan,
            "opp_ga_avg_5":    of.get("ga_avg_5") if of else np.nan,
            "opp_unbeaten_5":  of.get("unbeaten_5") if of else np.nan,
            "wr_diff_5":       ((pf.get("wr_5", 0.5) - (of.get("wr_5", 0.5) if of else 0.5))),
            "league_home_wr":  league_hwr,
        }
        model_name = "soccer_ml_v1"

    # ── Inference ────────────────────────────────────────────────────────────
    try:
        X = np.array([[feature_dict.get(c, np.nan) for c in feat_cols]], dtype=float)
        X_imp = imp.transform(X)
        X_sc  = sc.transform(X_imp)
        win_prob = float(clf.predict_proba(X_sc)[0][1])
    except Exception as exc:
        log.warning("score_soccer_leg error: %s", exc)
        return None

    dec_odds = leg.get("odds", 2.0)
    ev       = win_prob * (dec_odds - 1) - (1 - win_prob)

    return {
        "win_prob":   round(win_prob * 100, 2),
        "ev":         round(ev, 4),
        "model_used": model_name,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# World Cup 2026 — Market-Anchored Inference
# ═══════════════════════════════════════════════════════════════════════════════

_WC_DATA_CACHE: dict | None = None
_WC_CACHE_TS: float = 0.0


def _load_wc_data() -> dict | None:
    """Load and cache WC data files (feature matrix, odds snapshot, Pi, chemistry, dependency).

    Caches for 1 hour to avoid repeated disk reads during a single generation run.
    """
    import time as _time
    global _WC_DATA_CACHE, _WC_CACHE_TS

    if _WC_DATA_CACHE is not None and (_time.time() - _WC_CACHE_TS) < 3600:
        return _WC_DATA_CACHE

    import json, os
    _data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    try:
        import pandas as _pd
        feat_path = os.path.join(_data_dir, "wc2026_feature_matrix.csv")
        feat_df = _pd.read_csv(feat_path)

        with open(os.path.join(_data_dir, "wc2026_odds_snapshot.json")) as f:
            odds_snap = json.load(f)

        with open(os.path.join(_data_dir, "wc2026_pi_ratings_v2.json")) as f:
            pi_data = json.load(f)

        with open(os.path.join(_data_dir, "wc2026_club_chemistry.json")) as f:
            chem_data = json.load(f)

        with open(os.path.join(_data_dir, "wc2026_scoring_dependency.json")) as f:
            dep_data = json.load(f)

        # Optional: xG proxy and squad features (enrichment data)
        xg_path = os.path.join(_data_dir, "wc2026_xg_proxy.json")
        sq_path = os.path.join(_data_dir, "wc2026_squad_features.json")
        xg_proxy = json.load(open(xg_path)) if os.path.exists(xg_path) else {}
        squad_feat = json.load(open(sq_path)) if os.path.exists(sq_path) else {}
    except Exception as exc:
        log.warning("_load_wc_data: failed to load WC data: %s", exc)
        return None

    # Build team-name lookup for feature matrix rows
    _NAME_ALIASES = {
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
        "Czechia": "Czech Republic",
        "Türkiye": "Turkey", "Turkiye": "Turkey",
        "Congo DR": "DR Congo",
        "Korea Republic": "South Korea", "Republic of Korea": "South Korea",
        "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
        "USA": "United States", "Republic of Ireland": "Ireland",
        "IR Iran": "Iran", "Korea DPR": "North Korea",
        "Chinese Taipei": "Taiwan", "Curaçao": "Curacao",
    }

    def _norm(n: str) -> str:
        return _NAME_ALIASES.get(n, n)

    # Index feature matrix by (home_norm, away_norm)
    feat_lookup = {}
    for _, row in feat_df.iterrows():
        h = _norm(str(row.get("home_team", "")))
        a = _norm(str(row.get("away_team", "")))
        feat_lookup[(h, a)] = row.to_dict()

    # Index odds snapshot by (home_norm, away_norm)
    odds_lookup = {}
    for ev in odds_snap:
        h = _norm(ev.get("home_team", ""))
        a = _norm(ev.get("away_team", ""))
        odds_lookup[(h, a)] = ev

    _WC_DATA_CACHE = {
        "feat_lookup": feat_lookup,
        "odds_lookup": odds_lookup,
        "pi_data": pi_data,
        "chem_data": chem_data,
        "dep_data": dep_data,
        "xg_proxy": xg_proxy,
        "squad_feat": squad_feat,
        "_norm": _norm,
    }
    _WC_CACHE_TS = _time.time()
    return _WC_DATA_CACHE


def _devig_3way(odds_list: list[float]) -> list[float]:
    """Remove vig from 3-way decimal odds → true implied probabilities."""
    implied = [1.0 / o for o in odds_list if o > 1.0]
    total = sum(implied)
    if total <= 0:
        return [0.33, 0.33, 0.34]
    return [p / total for p in implied]


def _get_wc_market_odds(odds_event: dict, home: str, away: str) -> tuple[float, float, float] | None:
    """Extract consensus devigged 3-way odds from the odds snapshot event.

    Averages across all bookmakers to get a stable market-implied probability.
    """
    h2h_probs = []
    for bk in (odds_event.get("bookmakers") or []):
        for mkt in (bk.get("markets") or []):
            if mkt.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            h_price = outcomes.get(home)
            d_price = outcomes.get("Draw")
            a_price = outcomes.get(away)
            if h_price and d_price and a_price and h_price > 1 and d_price > 1 and a_price > 1:
                devigged = _devig_3way([h_price, d_price, a_price])
                h2h_probs.append(devigged)

    if not h2h_probs:
        return None

    n = len(h2h_probs)
    avg_h = sum(p[0] for p in h2h_probs) / n
    avg_d = sum(p[1] for p in h2h_probs) / n
    avg_a = sum(p[2] for p in h2h_probs) / n
    return (avg_h, avg_d, avg_a)


def predict_wc_match(leg: dict) -> dict | None:
    """Score a World Cup leg using the market-anchored approach.

    For knockout-stage matches (R32+), routes to the self-vetting knockout
    model (wc_knockout.py) which includes regularised lambdas, confederation
    corrections, and artifact detection.

    Supports h2h (moneyline), spreads, and totals markets.

    Returns dict with win_prob (0-100), ev, model_used, or None if data unavailable.
    """
    # ── Knockout routing: if odds snapshot has this match as a KO fixture, use knockout model ──
    try:
        from wc_knockout import predict_wc_knockout_match, _norm as _ko_norm
        import json as _json, os as _os
        _ko_data_dir = _os.path.join(_os.path.dirname(__file__), "..", "data")
        _ko_snap_path = _os.path.join(_ko_data_dir, "wc2026_odds_snapshot.json")
        if _os.path.exists(_ko_snap_path):
            with open(_ko_snap_path) as _f:
                _ko_snap = _json.load(_f)
            _home_n = _ko_norm(leg.get("home_team", "") or "")
            _away_n = _ko_norm(leg.get("away_team", "") or "")
            # Check if this match is in the current odds snapshot (which is R32+ now)
            for _ev in _ko_snap:
                _eh = _ko_norm(_ev.get("home_team", ""))
                _ea = _ko_norm(_ev.get("away_team", ""))
                if (_eh == _home_n and _ea == _away_n) or (_eh == _away_n and _ea == _home_n):
                    # Found in snapshot → route to knockout model
                    ko_result = predict_wc_knockout_match(leg)
                    if ko_result is not None:
                        return ko_result
                    break
    except Exception:
        pass  # Fall through to group-stage model
    wc = _load_wc_data()
    if wc is None:
        return None

    _norm = wc["_norm"]
    home = _norm(leg.get("home_team", "") or "")
    away = _norm(leg.get("away_team", "") or "")
    market = leg.get("market", "")
    pick = (leg.get("pick") or "").strip()

    if not home or not away or not pick:
        return None

    # ── Look up feature matrix row ──────────────────────────────────────────
    feat_row = wc["feat_lookup"].get((home, away))
    if feat_row is None:
        # Try reversed (away team listed as home in odds snapshot)
        feat_row = wc["feat_lookup"].get((away, home))
        if feat_row is not None:
            home, away = away, home  # Swap to match feature matrix perspective

    if feat_row is None:
        return None

    # ── Get fresh market odds ───────────────────────────────────────────────
    odds_ev = wc["odds_lookup"].get((home, away))
    if odds_ev is None:
        odds_ev = wc["odds_lookup"].get((away, home))

    # Build the adjustment input row
    mkt_home = float(feat_row.get("odds_home_implied", 0.33))
    mkt_draw = float(feat_row.get("odds_draw_implied", 0.25))
    mkt_away = float(feat_row.get("odds_away_implied", 0.20))

    # Override with fresh snapshot odds if available
    if odds_ev is not None:
        fresh = _get_wc_market_odds(odds_ev, home, away)
        if fresh is None:
            # Try with original event team names (pre-normalization)
            orig_home = odds_ev.get("home_team", home)
            orig_away = odds_ev.get("away_team", away)
            fresh = _get_wc_market_odds(odds_ev, orig_home, orig_away)
        if fresh is not None:
            mkt_home, mkt_draw, mkt_away = fresh

    adj_row = {
        "pi_delta":               float(feat_row.get("pi_delta", 0)),
        "odds_home_implied":      mkt_home,
        "odds_draw_implied":      mkt_draw,
        "odds_away_implied":      mkt_away,
        "qw_chemistry_delta":     float(feat_row.get("qw_chemistry_delta", 0)),
        "dependency_ratio_delta": float(feat_row.get("dependency_ratio_delta", 0)),
        "league_quality_delta":   float(feat_row.get("league_quality_delta", 0)),
        "form_win_pct_delta":     float(feat_row.get("form_win_pct_delta", 0)),
        "venue_alt_ft":           float(feat_row.get("venue_alt_ft", 0) or 0),
        "rest_delta":             float(feat_row.get("rest_delta", 0) or 0),
        "xg_diff_delta":          float(feat_row.get("xg_diff_delta", 0) or 0),
        "star_share_delta":       float(feat_row.get("star_share_delta", 0) or 0),
        "squad_goal_rate_delta":  float(feat_row.get("squad_goal_rate_delta", 0) or 0),
    }

    # ── Compute anchored 3-way probabilities ────────────────────────────────
    from wc_market_anchored import compute_anchored_probs
    probs = compute_anchored_probs(adj_row)
    p_home = probs["home_win"]
    p_draw = probs["draw"]
    p_away = probs["away_win"]

    # ── Map pick to win probability based on market type ────────────────────
    pick_lower = pick.lower().strip()
    dec_odds = leg.get("odds", 2.0)

    # Normalize pick for matching
    pick_norm = _norm(pick).lower()
    home_lower = home.lower()
    away_lower = away.lower()

    def _is_home_pick() -> bool:
        return pick_lower == home_lower or pick_norm == home_lower

    def _is_away_pick() -> bool:
        return pick_lower == away_lower or pick_norm == away_lower

    if market in ("h2h", "moneyline"):
        if _is_home_pick():
            win_prob = p_home
        elif _is_away_pick():
            win_prob = p_away
        elif pick_lower == "draw":
            win_prob = p_draw
        else:
            return None

    elif market in ("spreads", "alternate_spreads"):
        # Convert 3-way probabilities to spread coverage probability
        point = leg.get("point")
        if point is None:
            return None
        spread = float(point)

        # Determine if pick is home or away
        if _is_home_pick():
            # Home team covers spread
            # With a positive spread (e.g. +0.5), team wins or draws and covers
            # With zero spread, team must win outright
            # With negative spread (e.g. -1.5), team must win by 2+
            if spread >= 0.5:
                win_prob = p_home + p_draw  # covers with win or draw
            elif spread >= -0.5:
                win_prob = p_home + p_draw * 0.5  # draw is push at 0
            else:
                # Negative spread: estimate from moneyline strength
                # More negative spread → harder to cover
                ml_strength = p_home
                # Approximate: P(cover -N) ≈ P(win) * discount_factor
                discount = max(0.3, 1.0 - abs(spread) * 0.15)
                win_prob = ml_strength * discount
        elif _is_away_pick():
            if spread >= 0.5:
                win_prob = p_away + p_draw
            elif spread >= -0.5:
                win_prob = p_away + p_draw * 0.5
            else:
                ml_strength = p_away
                discount = max(0.3, 1.0 - abs(spread) * 0.15)
                win_prob = ml_strength * discount
        else:
            return None

        win_prob = max(0.05, min(0.95, win_prob))

    elif market in ("totals", "alternate_totals"):
        # Use expected goals from form data to estimate over/under
        point = leg.get("point")
        if point is None:
            return None
        line = float(point)

        # Expected goals — blend form data with xG proxy (competitive matches)
        h_gf = float(feat_row.get("home_form_avg_goals_scored", 1.3))
        h_ga = float(feat_row.get("home_form_avg_goals_conceded", 1.0))
        a_gf = float(feat_row.get("away_form_avg_goals_scored", 1.2))
        a_ga = float(feat_row.get("away_form_avg_goals_conceded", 1.1))

        # Blend with xG proxy if available (competitive match goal rates)
        h_xg = float(feat_row.get("home_xg_proxy", 0) or 0)
        a_xg = float(feat_row.get("away_xg_proxy", 0) or 0)
        h_xga = float(feat_row.get("home_xga_proxy", 0) or 0)
        a_xga = float(feat_row.get("away_xga_proxy", 0) or 0)
        if h_xg > 0 and a_xg > 0:
            # Blend form + xG proxy (50/50)
            h_gf = 0.5 * h_gf + 0.5 * h_xg
            a_gf = 0.5 * a_gf + 0.5 * a_xg
            h_ga = 0.5 * h_ga + 0.5 * h_xga
            a_ga = 0.5 * a_ga + 0.5 * a_xga

        # Expected total goals (average of team perspectives)
        exp_total = (h_gf + a_gf + h_ga + a_ga) / 2.0

        # WC group stage averages ~2.5 goals per game
        # Blend team expectation with tournament prior
        WC_AVG_GOALS = 2.5
        exp_total = 0.6 * exp_total + 0.4 * WC_AVG_GOALS

        # Poisson-based over probability
        import math as _math
        lam = max(0.5, exp_total)
        # P(total > line) using Poisson CDF complement
        # Approximate: sum P(X=k) for k=0..floor(line)
        floor_line = int(_math.floor(line))
        cdf = 0.0
        for k in range(floor_line + 1):
            cdf += (lam ** k * _math.exp(-lam)) / _math.factorial(k)

        # If line is integer (e.g., 2.0), exact = push; if half (2.5), clean over/under
        if line == floor_line:
            # Whole number: P(over) = 1 - cdf + P(X=line)/2 (push splits)
            p_exact = (lam ** floor_line * _math.exp(-lam)) / _math.factorial(floor_line)
            over_prob = 1.0 - cdf + p_exact * 0.5
        else:
            over_prob = 1.0 - cdf

        if pick_lower == "over":
            win_prob = over_prob
        elif pick_lower == "under":
            win_prob = 1.0 - over_prob
        else:
            return None

        win_prob = max(0.05, min(0.95, win_prob))

    else:
        return None

    ev = win_prob * (dec_odds - 1) - (1 - win_prob)

    # ── Confidence zone warning ──────────────────────────────────────────
    # 2022 backtest: 50-65% WP zone had only 41% accuracy (overconfident).
    # Flag picks in this zone so the display can add a caveat.
    wp_pct = round(win_prob * 100, 2)
    confidence = "high" if wp_pct >= 70 else ("low" if wp_pct >= 50 else "very_low")

    return {
        "win_prob":   wp_pct,
        "ev":         round(ev, 4),
        "model_used": "wc_market_anchored_v1",
        "confidence": confidence,
    }


def _prep_row(feature_dict: dict, scaler, imputer,
              active_cols: list[str] | None = None) -> np.ndarray:
    """
    Build a 2-D feature row for inference.

    active_cols: the feature columns the model was actually trained on (saved
    alongside the model artifact). Uses only these columns so the row shape
    matches the trained imputer/scaler/model.

    GBC path  (scaler is not None): apply imputer → StandardScaler.
    HistGBT path (scaler is None) : pass raw NaN-containing row directly.
    """
    if scaler is None:          # HistGBT — use full feature set, no preprocessing
        cols = active_cols or FEATURE_COLS
        row = np.array(
            [[feature_dict.get(c, np.nan) for c in cols]], dtype=float
        )
        return row
    # GBC path — determine which columns the model expects
    if active_cols is not None:
        cols = active_cols
    elif imputer is not None and imputer.n_features_in_ == len(FEATURE_COLS):
        cols = FEATURE_COLS
    else:
        cols = ORIGINAL_FEATURE_COLS
    row = np.array(
        [[feature_dict.get(c, np.nan) for c in cols]], dtype=float
    )
    if imputer is not None and imputer.n_features_in_ == len(cols):
        row = imputer.transform(row)
    else:
        row = np.nan_to_num(row, nan=0.0)
    return scaler.transform(row)


# ── Training ───────────────────────────────────────────────────────────────────
def train(db: Session, algorithm: str = "gradient_boost",
          train_submodels: bool = True,
          feature_cols: list[str] | None = None,
          dry_run: bool = False,
          mock_only: bool = False) -> dict:
    """
    Train the combined model.

    feature_cols: explicit list of columns to use; defaults to FEATURE_COLS.
                  Pass ORIGINAL_FEATURE_COLS for the 46-feature GBC baseline,
                  or a custom subset when only some new signals are ready.
    algorithm:    "gradient_boost"       — GBC + CalibratedClassifierCV (v3 baseline)
                  "hist_gradient_boost"  — HistGBT + CalibratedClassifierCV (isotonic)
    dry_run:      If True, skip saving pkl files and writing to model_runs.
                  Returns the same metrics dict with "dry_run": True added.
    mock_only:    If True, train only on mock bet legs (skips personal + historical).
    """
    now = datetime.utcnow()

    # ── 1. Personal bets ────────────────────────────────────────────────────
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).order_by(Bet.time_placed).all()

    personal_df = _build_personal_df(bets, cutoff=now)
    if len(personal_df) < 30:
        return {"error": "Need >= 30 settled bets to train."}

    # ── 2. Historical games ────────────────────────────────────────────────
    historical_df = _load_historical_df()

    # ── 3. Settled mock bet legs (carries new per-leg signals) ────────────
    mock_leg_df = _build_mock_leg_df(db)

    # ── 4. CLV hook — fills clv / clv_available when activated ────────────
    personal_df   = add_clv_features(personal_df)
    if not historical_df.empty:
        historical_df = add_clv_features(historical_df)

    # ── 5. Combine and sort chronologically ────────────────────────────────
    if mock_only:
        if mock_leg_df.empty:
            return {"error": "No settled mock legs for mock-only training."}
        combined = mock_leg_df.copy()
        n_mock_legs  = len(mock_leg_df)
        n_personal   = 0
        n_historical = 0
        log.info("train (mock-only): %d mock legs", n_mock_legs)
    else:
        frames = [personal_df]
        if not historical_df.empty:
            frames.append(historical_df[personal_df.columns])
        if not mock_leg_df.empty:
            frames.append(mock_leg_df[personal_df.columns])  # align cols
        combined = pd.concat(frames, ignore_index=True)
        n_personal   = int((combined["_weight"] > 1.0).sum())
        n_mock_legs  = len(mock_leg_df) if not mock_leg_df.empty else 0
        n_historical = len(combined) - n_personal
        log.info(
            "train: %d personal bets + %d historical rows + %d mock legs = %d total",
            n_personal - n_mock_legs, n_historical, n_mock_legs, len(combined),
        )
    combined = combined.sort_values("_date").reset_index(drop=True)

    # Resolve which feature columns to use for this training run
    _goal_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    # Add any missing columns as NaN so the subset is always available
    for _c in _goal_cols:
        if _c not in combined.columns:
            combined[_c] = np.nan

    # Split first, then drop all-NaN columns based on training portion only.
    # (Features that only have data in the last 20% would be all-NaN for the
    # imputer, causing a dimension mismatch between imputer and scaler.)
    _split_idx = int(len(combined) * 0.80)
    _train_portion = combined.iloc[:_split_idx]

    _nan_mask = _train_portion[_goal_cols].isna().all()
    _dropped  = [c for c in _goal_cols if _nan_mask.get(c, False)]
    _train_cols = [c for c in _goal_cols if not _nan_mask.get(c, False)]
    if _dropped:
        log.info("train: dropped %d all-NaN features (in train split): %s",
                 len(_dropped), ", ".join(_dropped[:5]) + ("..." if len(_dropped) > 5 else ""))
    log.info("train: active feature count = %d (of %d goal)", len(_train_cols), len(_goal_cols))

    X_all  = combined[_train_cols].values.astype(float)
    y_all  = combined["_label"].values.astype(float)
    sw_all = combined["_weight"].values.astype(float)

    # ── 6. Temporal 80/20 split (no shuffle) ──────────────────────────────
    split  = _split_idx
    X_tr,  X_te  = X_all[:split],  X_all[split:]
    y_tr,  y_te  = y_all[:split],  y_all[split:]
    sw_tr         = sw_all[:split]

    # ── 7. Preprocess — HistGBT handles NaN natively, GBC needs impute+scale
    use_hist = (algorithm == "hist_gradient_boost")
    if use_hist:
        # No imputer or scaler; HistGBT is scale-invariant and NaN-tolerant.
        X_tr_s, X_te_s = X_tr, X_te
        imputer = None
        scaler  = None
    else:
        imputer  = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr_imp)
        X_te_s   = scaler.transform(X_te_imp)

    # ── 8. Fit with sample weights ─────────────────────────────────────────
    clf = _build_clf(algorithm)
    clf.fit(X_tr_s, y_tr, sample_weight=sw_tr)

    # ── 9. Evaluate ────────────────────────────────────────────────────────
    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    acc    = float(accuracy_score(y_te, y_pred))
    auc    = float(roc_auc_score(y_te, y_prob)) if len(set(y_te)) > 1 else 0.5
    brier  = float(brier_score_loss(y_te, y_prob))
    imps   = _importances(clf, _train_cols, X_val=X_te_s, y_val=y_te)

    # Feature importances sorted descending — stored in notes for model-status
    imps_sorted = dict(sorted(imps.items(), key=lambda x: -x[1]))
    top_features = list(imps_sorted.items())[:10]  # top 10 for logging
    log.info("train: top features: %s",
             ", ".join(f"{n}={v:.4f}" for n, v in top_features))

    # ── 10. Persist (skipped when dry_run=True) ──────────────────────────────
    if not dry_run:
        _ver_meta = {"auc": round(auc, 4), "brier": round(brier, 4),
                     "n_train": int(len(X_tr)), "n_features": len(_train_cols)}
        _versioned_save(clf,         MODEL_PATH,       "global_clf",         _ver_meta)
        _versioned_save(scaler,      SCALER_PATH,      "global_scaler",      _ver_meta)
        _versioned_save(imputer,     IMPUTER_PATH,     "global_imputer",     _ver_meta)
        _versioned_save(_train_cols, ACTIVE_COLS_PATH,  "global_active_cols", _ver_meta)
        # Invalidate in-memory cache so next load_model() picks up new artifacts
        global _MODEL_CACHE, _MODEL_LOAD_FAILED
        _MODEL_CACHE = None
        _MODEL_LOAD_FAILED = False

    # ── 11. Sub-models (personal bets only, per sport) ─────────────────────
    sub_trained = {}
    if train_submodels:
        for _c in _train_cols:
            if _c not in personal_df.columns:
                personal_df[_c] = np.nan
        p_raw = personal_df[_train_cols].values.astype(float)
        if use_hist:
            p_processed = p_raw   # HistGBT sub-models also need no preprocessing
        else:
            p_processed = scaler.transform(imputer.transform(p_raw))
        p_y     = personal_df["_label"].values.astype(float)
        p_sw    = personal_df["_weight"].values.astype(float)
        p_sport = personal_df["_sport_label"].values

        for sport in SUBMODEL_SPORTS:
            mask = p_sport == sport
            if mask.sum() < 40:
                continue
            Xs, ys, sws = p_processed[mask], p_y[mask], p_sw[mask]
            sp = int(len(Xs) * 0.80)
            if sp < 20:
                continue
            clf_s = _build_clf(algorithm, calibrate=False)
            clf_s.fit(Xs[:sp], ys[:sp], sample_weight=sws[:sp])
            path = os.path.join(SUBMODEL_DIR, f"{sport.lower().replace(' ', '_')}.pkl")
            with open(path, "wb") as f:
                pickle.dump(clf_s, f)
            sub_trained[sport] = int(mask.sum())

    # ── 11. Per-sport test metrics ─────────────────────────────────────────
    SPORT_MAP_INV = {v: k for k, v in SPORT_MAP.items()}
    test_sport_ids = X_te[:, _train_cols.index("sport_id")]
    s_te = np.array([SPORT_MAP_INV.get(int(sid), "other") for sid in test_sport_ids])

    sport_metrics: dict = {}
    for sport in set(s_te):
        mask = s_te == sport
        if mask.sum() < 5:
            continue
        sport_metrics[sport] = {
            "n":        int(mask.sum()),
            "accuracy": round(float(accuracy_score(y_te[mask], y_pred[mask])) * 100, 1),
            "win_rate": round(float(y_te[mask].mean()) * 100, 1),
        }

    # ── 12. Drift baseline (skipped when dry_run=True) ────────────────────────
    n_personal_train   = int((combined["_weight"][:split] > 1.0).sum())
    n_historical_train = split - n_personal_train - n_mock_legs

    if not dry_run:
        with open(DRIFT_PATH, "w") as f:
            json.dump({
                "baseline_accuracy": round(acc, 4),
                "baseline_roc_auc":  round(auc, 4),
                "n_test":            len(X_te),
                "saved_at":          now.isoformat(),
            }, f)

        run = ModelRun(
            algorithm=algorithm,
            n_train=split, n_test=len(X_te),
            accuracy=round(acc, 4), roc_auc=round(auc, 4),
            feature_names=_train_cols,
            notes=json.dumps({
                "importances":         imps_sorted,
                "brier":               round(brier, 4),
                "sport_metrics":       sport_metrics,
                "submodels":           sub_trained,
                "algorithm":           algorithm,
                "n_personal_train":    n_personal_train,
                "n_historical_train":  n_historical_train,
                "n_mock_legs_train":   n_mock_legs,
                "feature_cols":        _train_cols,
                "n_features":          len(_train_cols),
                "weight_config":       WEIGHT_CONFIG,
            }),
        )
        db.add(run); db.commit(); db.refresh(run)
        run_id = run.id
    else:
        run_id = None

    result = {
        "run_id":               run_id,
        "algorithm":            algorithm,
        "n_features":           len(_train_cols),
        "n_goal_features":      len(_goal_cols),
        "n_dropped_nan":        len(_dropped),
        "dropped_features":     _dropped,
        "n_train":              split,
        "n_test":               len(X_te),
        "n_personal_train":     n_personal_train,
        "n_historical_train":   n_historical_train,
        "n_mock_legs_train":    n_mock_legs,
        "accuracy":             round(acc * 100, 2),
        "roc_auc":              round(auc, 4),
        "brier_score":          round(brier, 4),
        "feature_importance":   imps_sorted,
        "sport_metrics":        sport_metrics,
        "submodels_trained":    list(sub_trained.keys()),
        "weight_config":        WEIGHT_CONFIG,
        "class_report":         classification_report(y_te, y_pred, output_dict=True),
    }
    if dry_run:
        result["dry_run"] = True
    return result


# ── Model stats (for dashboard widget) ────────────────────────────────────────
def get_model_stats(db) -> dict:
    """
    Return validated model metrics for the stats widget.
    Reads the latest ModelRun from DB + eval_results.json if present.
    """
    import os

    # Latest training run
    run = db.query(ModelRun).order_by(ModelRun.id.desc()).first()
    if run is None:
        return {"error": "Model not trained yet"}

    notes = {}
    try:
        notes = json.loads(run.notes or "{}")
    except Exception:
        pass

    # Backtest results from eval (written by model_eval.py)
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "eval_results.json"
    )
    backtest_55 = None
    try:
        with open(eval_path) as f:
            ev = json.load(f)
        # eval_results.json structure: profit_sim["0.55"] = {n_bets, win_rate, roi, total_pnl}
        t55 = (ev.get("profit_sim") or {}).get("0.55")
        if t55:
            backtest_55 = {
                "win_rate":  round(t55.get("win_rate", 0) * 100, 1),
                "roi":       round(t55.get("roi", 0) * 100, 1),
                "n_bets":    t55.get("n_bets"),
                "total_pnl": t55.get("total_pnl"),
            }
    except Exception:
        pass

    # Cover rate validation — stored in notes by cover_rate_check
    cover_rates = notes.get("cover_rates", {
        "NFL": {"rate": 50.5, "pass": True},
    })

    # Diagnostic model inventory: {sport}_ats_clf.pkl files in SUBMODEL_DIR
    # Supplemented with static metadata for MLB/NHL (backfill complete, AUC pending)
    _DIAGNOSTIC_META: dict[str, dict] = {
        "MLB": {
            "seasons_covered":  ["2022", "2023", "2024", "2025"],
            "auc_threshold":    0.55,
            "data_sources":     ["mlb-statsapi", "espn-runline"],
            "notes":            "Run-line ±1.5. Backfill: --backfill-mlb. Gate AUC>=0.55.",
        },
        "NHL": {
            "seasons_covered":  ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"],
            "auc_threshold":    0.55,
            "data_sources":     ["nhl-web-api", "espn-puckline"],
            "notes":            "Puck-line ±1.5 with OT/SO flag. Gate AUC>=0.55.",
        },
    }

    diagnostic_models: list[dict] = []
    saved_sports: set[str] = set()
    try:
        for fname in sorted(os.listdir(SUBMODEL_DIR)):
            if fname.endswith("_ats_clf.pkl"):
                sport_label = fname.replace("_ats_clf.pkl", "").upper()
                entry: dict = {
                    "sport":       sport_label,
                    "path":        os.path.join(SUBMODEL_DIR, fname),
                    "saved":       True,
                    "model_class": "diagnostic",   # NOT production
                }
                entry.update(_DIAGNOSTIC_META.get(sport_label, {}))
                diagnostic_models.append(entry)
                saved_sports.add(sport_label)
    except Exception:
        pass

    # Include pending sports (no .pkl yet — backfill complete but not trained)
    for sport_label, meta in _DIAGNOSTIC_META.items():
        if sport_label not in saved_sports:
            diagnostic_models.append({
                "sport":       sport_label,
                "saved":       False,
                "model_class": "diagnostic",
                "status":      "pending_isolation_test",
                **meta,
            })

    # ── Load per-sport AUC from saved meta JSON files ────────────────────────
    import json as _json
    _SUBMODEL_META_STATIC: dict[str, dict] = {
        "NHL": {
            "threshold":   0.60,
            "top_feature": "opp_strength_diff",
            "calibration": "tight",
            "status":      "active",
        },
        "MLB": {
            "threshold":   0.55,
            "top_feature": "sp_era_diff",
            "calibration": "isotonic",
            "status":      "active",
        },
        "NBA": {
            "auc":    0.4952,
            "status": "path_b",
        },
        "NFL": {
            "auc":    0.5251,
            "status": "diagnostic_only",
        },
    }
    sub_models: dict[str, dict] = {}
    for sport_label, static in _SUBMODEL_META_STATIC.items():
        entry = dict(static)
        meta_path = os.path.join(SUBMODEL_DIR, f"{sport_label.lower()}_ats_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as _f:
                    _m = _json.load(_f)
                if _m.get("auc") is not None:
                    entry["auc"] = _m["auc"]
                entry["saved_at"]    = _m.get("saved_at")
                entry["n_features"]  = _m.get("n_features")
            except Exception:
                pass
        sub_models[sport_label] = entry

    # ── Soccer sub-models (form-based, trained 2026-04-26) ────────────────────
    _soccer_total_pkl = os.path.join(SUBMODEL_DIR, "soccer_total_goals_clf.pkl")
    sub_models["Soccer"] = {
        "soccer_model_status": {
            "total_goals": {
                "auc":          0.5578,
                "status":       "active",
                "algorithm":    "LogisticRegression",
                "threshold":    0.55,
                "pkl_exists":   os.path.exists(_soccer_total_pkl),
                "trained_at":   "2026-04-26",
                "n_features":   16,
                "top_features": ["line", "direction", "combined_goals_exp_5", "away_over25_r10"],
                "notes":        "Form-based. Totals only. Gate AUC>=0.54.",
            },
            "moneyline": {
                "auc":    0.5250,
                "status": "below_threshold",
                "notes":  "Below 0.54 gate. Falls back to combined_v1. Retrain at 150+ resolved legs.",
            },
            "shots_on_target": {
                "status": "pending_data",
                "notes":  "Requires API-Football premium for player stats. Deferred.",
            },
        },
        "soccer_total_v1_auc":   0.5578,
        "soccer_moneyline_v1":   "below_threshold",
        "kelly_fraction":        0.25,
        "status":                "active",
    }

    prod_auc = round(run.roc_auc or 0, 4)

    return {
        # ── New structured format ─────────────────────────────────────────────
        "production": {
            "model":       "combined_v1",
            "auc":         prod_auc,
            "sports":      ["NFL", "NBA"],
            "backtest_55": backtest_55["win_rate"] / 100 if backtest_55 else None,
        },
        "sub_models": sub_models,
        # ── Legacy flat fields (keep for backward compat with frontend widgets) ─
        "model_class":        "production",
        "run_id":             run.id,
        "last_retrain":       run.run_at.isoformat() if run.run_at else None,
        "algorithm":          run.algorithm,
        "production_auc":     prod_auc,
        "roc_auc":            prod_auc,
        "accuracy_pct":       round((run.accuracy or 0) * 100, 2),
        "n_train":            run.n_train,
        "n_test":             run.n_test,
        "n_personal_train":   notes.get("n_personal_train", 0),
        "n_historical_train": notes.get("n_historical_train", 0),
        "brier_score":        notes.get("brier", None),
        "backtest_55":        backtest_55,
        "cover_rates":        cover_rates,
        "version":            notes.get("version", "v3"),
        "weight_config":      notes.get("weight_config", {}),
        "diagnostic_models":  diagnostic_models,
        "sport_ats_models":   diagnostic_models,
    }


# ── SHAP-style explanation ─────────────────────────────────────────────────────
def explain_prediction(feature_dict: dict, top_n: int = 8) -> list[dict]:
    clf, scaler, imputer, active_cols = load_model()
    if clf is None:
        return []
    row_s  = _prep_row(feature_dict, scaler, imputer, active_cols)
    base_p = float(clf.predict_proba(row_s)[0][1])
    contribs = []
    _explain_cols = active_cols
    for i, col in enumerate(_explain_cols):
        abl        = row_s.copy(); abl[0, i] = 0.0
        impact     = base_p - float(clf.predict_proba(abl)[0][1])
        raw_val    = feature_dict.get(col, np.nan)
        contribs.append({
            "feature":    col,
            "value":      round(float(raw_val), 4) if not (isinstance(raw_val, float) and math.isnan(raw_val)) else None,
            "impact":     round(impact, 4),
            "direction":  "positive" if impact > 0 else "negative",
            "abs_impact": abs(impact),
        })
    contribs.sort(key=lambda x: -x["abs_impact"])
    return contribs[:top_n]


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_bet(feature_dict: dict, use_submodel: bool = True) -> dict:
    clf, scaler, imputer, active_cols = load_model()
    if clf is None:
        return {"error": "No trained model found. Run /api/model/train first."}

    sport   = feature_dict.get("sport_label", "")
    sub_clf = _load_submodel(sport) if (use_submodel and sport) else None
    active  = sub_clf if sub_clf else clf

    row_s    = _prep_row(feature_dict, scaler, imputer, active_cols)
    win_prob = _calibrate_prob(float(active.predict_proba(row_s)[0][1]))
    dec_odds = feature_dict.get("odds", 2.0)
    ev       = win_prob * (dec_odds - 1) - (1 - win_prob)
    explanation = explain_prediction(feature_dict, top_n=6)

    return {
        "win_probability": round(win_prob, 4),
        "expected_value":  round(ev, 4),
        "ev_pct":          round(ev * 100, 2),
        "recommendation":  _recommend(win_prob, ev, dec_odds),
        "model_used":      f"{sport} sub-model" if sub_clf else "global model",
        "explanation":     explanation,
    }


def _recommend(win_prob, ev, odds):
    imp  = 1 / odds if odds > 1 else 0.5
    edge = win_prob - imp
    if ev > 0.05 and edge > 0.02: return "STRONG BET — positive EV and edge over implied odds"
    if ev > 0:                    return "LEAN BET — slight positive EV"
    if ev > -0.05:                return "MARGINAL — small negative EV, use sparingly"
    return "AVOID — negative EV"


# ── Drift detection ────────────────────────────────────────────────────────────
def check_drift(db: Session, window: int = 30) -> dict:
    if not os.path.exists(DRIFT_PATH):
        return {"status": "no_baseline", "message": "Train a model first."}
    with open(DRIFT_PATH) as f:
        baseline = json.load(f)

    clf, scaler, imputer, active_cols = load_model()
    if clf is None:
        return {"status": "no_model"}

    recent = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).order_by(Bet.time_placed.desc()).limit(window).all()

    if len(recent) < 10:
        return {"status": "insufficient_data", "n_recent": len(recent)}

    correct = total = 0
    for b in recent:
        f = extract_features(b)
        if not f:
            continue
        row_s = _prep_row(f, scaler, imputer, active_cols)
        pred  = int(clf.predict(row_s)[0])
        actual = 1 if b.status == "SETTLED_WIN" else 0
        correct += (pred == actual); total += 1

    if total == 0:
        return {"status": "insufficient_data"}

    recent_acc   = correct / total
    baseline_acc = baseline["baseline_accuracy"]
    gap          = baseline_acc - recent_acc
    threshold    = 0.08

    status = (
        "drift_detected" if gap > threshold else
        "drift_warning"  if gap > threshold / 2 else
        "ok"
    )
    return {
        "status":            status,
        "baseline_accuracy": round(baseline_acc * 100, 2),
        "recent_accuracy":   round(recent_acc * 100, 2),
        "drift_gap_pct":     round(gap * 100, 2),
        "window":            total,
        "threshold_pct":     threshold * 100,
        "recommendation": (
            "Retrain the model — recent performance has degraded."
            if status == "drift_detected" else
            "Consider retraining soon."
            if status == "drift_warning" else
            "Model is performing consistently."
        ),
    }


# ── Backtest ───────────────────────────────────────────────────────────────────
def backtest(db: Session, start_date: Optional[str] = None) -> dict:
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).order_by(Bet.time_placed).all()

    if len(bets) < 40:
        return {"error": "Need >= 40 settled bets for backtest."}

    clf, scaler, imputer, active_cols = load_model()
    if clf is None:
        return {"error": "Train the model first."}

    results = []; correct = 0
    for b in bets[30:]:
        f = extract_features(b)
        if not f:
            continue
        row_s = _prep_row(f, scaler, imputer, active_cols)
        prob  = float(clf.predict_proba(row_s)[0][1])
        pred  = 1 if prob >= 0.5 else 0
        actual = 1 if b.status == "SETTLED_WIN" else 0
        results.append({
            "bet_id":              b.id,
            "date":                b.time_placed.strftime("%Y-%m-%d") if b.time_placed else "N/A",
            "legs":                b.legs,
            "odds":                b.odds,
            "predicted_win_prob":  round(prob, 3),
            "predicted":           pred,
            "actual":              actual,
            "correct":             pred == actual,
            "profit":              b.profit or 0,
            "sport":               (b.sports or "").split("|")[0].strip(),
        })
        correct += (pred == actual)

    acc = correct / len(results) if results else 0
    sport_bt: dict = {}
    for r in results:
        s = r["sport"] or "other"
        sport_bt.setdefault(s, {"correct": 0, "total": 0})
        sport_bt[s]["total"] += 1
        if r["correct"]:
            sport_bt[s]["correct"] += 1

    return {
        "total_evaluated":     len(results),
        "correct_predictions": correct,
        "accuracy_pct":        round(acc * 100, 2),
        "bets_recommended":    sum(1 for r in results if r["predicted"] == 1),
        "ev_if_followed":      round(sum(r["profit"] for r in results if r["predicted"] == 1), 2),
        "avg_predicted_prob":  round(sum(r["predicted_win_prob"] for r in results) / len(results), 3) if results else 0,
        "sport_accuracy":      {s: round(v["correct"] / v["total"] * 100, 1) for s, v in sport_bt.items() if v["total"] >= 5},
        "sample":              results[-10:],
    }


# ── Diagnostic per-sport model ─────────────────────────────────────────────────

def train_diagnostic_sport_model(
    sport:    str,
    save:     bool = False,
    min_season: int | None = None,
    feature_list: str = "auto",
) -> dict:
    """
    DIAGNOSTIC path — train a GradientBoostingClassifier on ATS rows for
    *sport* only, using rolling team stats (MODEL_FEATURE_COLS).

    This is NOT the production model.  It runs in a separate feature space
    from train(db) and is used exclusively for per-sport signal research.
    See module docstring for the full two-path explanation.

    Parameters
    ----------
    sport : e.g. "NBA", "NFL", "MLB"
    save  : if True AND AUC >= 0.55, persist:
              SUBMODEL_DIR/{sport_lower}_ats_clf.pkl
              SUBMODEL_DIR/{sport_lower}_ats_scaler.pkl
              SUBMODEL_DIR/{sport_lower}_ats_imputer.pkl
    min_season : if set, only use games from this year onward (e.g. 2025).
                 Recent seasons may be more predictive for spread-aware
                 models than full historical data.

    Decision thresholds (also controls save when save=True)
    ---------------------------------------------------------
    AUC >= 0.55  → "WORTH BUILDING"  (saves if save=True)
    0.50–0.54    → "WAIT"
    < 0.50       → "FEATURE WORK NEEDED"

    Does NOT overwrite any production model or main model.pkl.
    """
    if save:
        _assert_training_env()
    try:
        from features import (build_feature_matrix, TARGET_ATS,
                              MODEL_FEATURE_COLS, MODEL_FEATURE_COLS_V1_PRUNED)
    except ImportError:
        return {"error": "features.py not importable"}

    # ── 1. Load ATS feature matrix ────────────────────────────────────────────
    matrix = build_feature_matrix(sport)
    if matrix.empty:
        return {"error": f"No {sport} data in historical.db"}

    labeled = matrix[matrix[TARGET_ATS].notna()].copy()

    # ── 1b. Optional training window filter ───────────────────────────────────
    if min_season is not None:
        _gd = pd.to_datetime(labeled["game_date"], errors="coerce")
        _before = len(labeled)
        labeled = labeled[_gd.dt.year >= min_season].copy()
        log.info("  min_season=%d: %d → %d games", min_season, _before, len(labeled))

    if len(labeled) < 50:
        return {"error": f"Only {len(labeled)} labeled {sport} rows — need ≥50"}

    # ── 2. Build X / y ────────────────────────────────────────────────────────
    # feature_list selects which feature set to train on:
    #   "auto"   → V1_PRUNED for MLB (Top-20 + K-BB%), MODEL_FEATURE_COLS otherwise
    #   "full"   → MODEL_FEATURE_COLS (all 74 features, legacy)
    #   "pruned" → MODEL_FEATURE_COLS_V1_PRUNED (21 features)
    if feature_list == "auto":
        _feat_source = MODEL_FEATURE_COLS_V1_PRUNED if sport == "MLB" else MODEL_FEATURE_COLS
    elif feature_list == "pruned":
        _feat_source = MODEL_FEATURE_COLS_V1_PRUNED
    else:
        _feat_source = MODEL_FEATURE_COLS
    feat_cols = [c for c in _feat_source if c in labeled.columns]

    # ── 2b. MLB spread-aware data augmentation ────────────────────────────────
    # Expand each game into multiple rows — one per spread value — so the model
    # learns how cover probability changes with the spread line.
    # Labels are recomputed per-spread: margin + spread > 0 → home covers.
    _SPREAD_AWARE = (sport == "MLB")
    _AUGMENT_SPREADS = [-1.5, 1.5, 2.5, 3.5, 4.5]
    if _SPREAD_AWARE:
        # Compute raw margin from scores (home perspective)
        margin = (
            pd.to_numeric(labeled["home_score"], errors="coerce")
            - pd.to_numeric(labeled["away_score"], errors="coerce")
        )
        aug_rows = []
        for spread_val in _AUGMENT_SPREADS:
            chunk = labeled.copy()
            chunk["_spread_value"] = float(spread_val)
            chunk["_spread_abs"]   = abs(spread_val)
            chunk["_is_fav_spread"] = 1.0 if spread_val < 0 else 0.0
            # Recompute ATS label: home covers when margin + spread > 0
            ats_val = margin + spread_val
            chunk[TARGET_ATS] = np.where(ats_val > 0, 1.0,
                                np.where(ats_val < 0, 0.0, np.nan))
            aug_rows.append(chunk)
        labeled = pd.concat(aug_rows, ignore_index=True)
        # Drop pushes (ats_val == 0 → NaN)
        labeled = labeled[labeled[TARGET_ATS].notna()].copy()
        # Sort chronologically (within each spread group, order is preserved)
        labeled = labeled.sort_values("game_date").reset_index(drop=True)
        feat_cols = feat_cols + ["_spread_value", "_spread_abs", "_is_fav_spread"]
        log.info("  MLB augmented: %d rows across %d spread values",
                 len(labeled), len(_AUGMENT_SPREADS))

        # NOTE: Totals context features (_total_line, _is_high_total,
        # _park_expected_total) were tested 2026-06-13 and found to degrade
        # AUC (-0.007) and Brier (+0.003) without improving calibration.
        # The mid-total -22pp gap was already resolved by spread augmentation.
        # Totals features are NOT included in training. The inference function
        # still accepts total_line for future use but the model ignores it.

    X = labeled[feat_cols].values.astype(float)
    y = labeled[TARGET_ATS].values.astype(float)

    cover_rate  = float(y.mean())
    split       = int(len(X) * 0.80)
    X_tr, X_te  = X[:split], X[split:]
    y_tr, y_te  = y[:split], y[split:]

    if len(set(y_te)) < 2:
        return {"error": f"{sport} test set has only one class — need more data"}

    # ── 3. Impute → scale ──────────────────────────────────────────────────────
    imp      = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)

    sc       = StandardScaler()
    X_tr_s   = sc.fit_transform(X_tr_imp)
    X_te_s   = sc.transform(X_te_imp)

    # ── 4. Train GBC + isotonic calibration ───────────────────────────────────
    # Deeper tree (d4) + more estimators + isotonic recalibration fixes the
    # overconfident high-probability tail that plagues plain GBC on ATS data.
    from sklearn.calibration import CalibratedClassifierCV
    _base = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf = CalibratedClassifierCV(_base, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc   = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    acc   = float(accuracy_score(y_te, y_pred))

    # Extract importances from CalibratedClassifierCV by averaging across folds
    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])[:20]

    # ── 6. Calibration buckets ─────────────────────────────────────────────────
    buckets: list[dict] = []
    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
        mask = (y_prob >= lo) & (y_prob < hi)
        n    = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket":      f"{int(lo*100)}-{int(hi*100)}%",
            "n":           n,
            "pred_prob":   round(float(y_prob[mask].mean()) * 100, 1),
            "actual_rate": round(float(y_te[mask].mean()) * 100, 1),
            "gap":         round((float(y_te[mask].mean()) - float(y_prob[mask].mean())) * 100, 1),
        })

    # ── 6b. Per-spread calibration (MLB augmented models only) ───────────────
    spread_cal: list[dict] = []
    if _SPREAD_AWARE and "_spread_value" in labeled.columns:
        _te_spread = labeled.iloc[split:]["_spread_value"].values
        for sv in _AUGMENT_SPREADS:
            _sm = _te_spread == sv
            _n  = int(_sm.sum())
            if _n == 0:
                continue
            spread_cal.append({
                "spread":      sv,
                "n":           _n,
                "pred_prob":   round(float(y_prob[_sm].mean()) * 100, 1),
                "actual_rate": round(float(y_te[_sm].mean()) * 100, 1),
                "gap":         round((float(y_te[_sm].mean()) - float(y_prob[_sm].mean())) * 100, 1),
            })

    # ── 7. Decision ────────────────────────────────────────────────────────────
    if auc >= 0.54:
        decision = f"WORTH BUILDING: per-sport {sport} sub-model justified (AUC >= 0.54)"
        if save:
            import datetime as _dt
            prefix = sport.lower()
            _ver_meta = {"sport": sport.upper(), "auc": round(auc, 4),
                         "n_features": len(feat_cols), "n_train": int(len(X_tr_s)),
                         "spread_aware": _SPREAD_AWARE, "min_season": min_season}
            _versioned_save(clf,       os.path.join(SUBMODEL_DIR, f"{prefix}_ats_clf.pkl"),
                            f"{prefix}_ats_clf", _ver_meta)
            _versioned_save(sc,        os.path.join(SUBMODEL_DIR, f"{prefix}_ats_scaler.pkl"),
                            f"{prefix}_ats_scaler", _ver_meta)
            _versioned_save(imp,       os.path.join(SUBMODEL_DIR, f"{prefix}_ats_imputer.pkl"),
                            f"{prefix}_ats_imputer", _ver_meta)
            _versioned_save(feat_cols, os.path.join(SUBMODEL_DIR, f"{prefix}_ats_feat_cols.pkl"),
                            f"{prefix}_ats_feat_cols", _ver_meta)
            # Persist metadata (also versioned)
            meta = {
                "sport":         sport.upper(),
                "auc":           round(auc, 4),
                "n_features":    len(feat_cols),
                "n_train":       int(len(X_tr_s)),
                "spread_aware":  _SPREAD_AWARE,
                "min_season":    min_season,
                "saved_at":      _dt.datetime.utcnow().isoformat() + "Z",
            }
            _meta_path = os.path.join(SUBMODEL_DIR, f"{prefix}_ats_meta.json")
            _versioned_save(meta, _meta_path, f"{prefix}_ats_meta", _ver_meta)
            log.info("train_diagnostic_sport_model: saved %s diagnostic model (%d features, AUC=%.4f, spread_aware=%s)",
                     sport, len(feat_cols), auc, _SPREAD_AWARE)
    elif auc >= 0.50:
        decision = f"WAIT: per-sport {sport} model won't help yet — more data needed (0.50 <= AUC < 0.54)"
    else:
        decision = f"FEATURE WORK NEEDED: {sport} features need engineering (AUC < 0.50)"

    return {
        "sport":            sport,
        "n_labeled":        len(labeled),
        "n_train":          len(X_tr),
        "n_test":           len(X_te),
        "cover_rate_all":   round(cover_rate * 100, 1),
        "cover_rate_train": round(float(y_tr.mean()) * 100, 1),
        "cover_rate_test":  round(float(y_te.mean()) * 100, 1),
        "roc_auc":          round(auc, 4),
        "brier_score":      round(brier, 4),
        "accuracy_pct":     round(acc * 100, 1),
        "n_features":       len(feat_cols),
        "feature_cols":     feat_cols,
        "top20_importances": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "calibration_buckets": buckets,
        "spread_calibration": spread_cal if _SPREAD_AWARE else [],
        "spread_aware":     _SPREAD_AWARE,
        "saved":            save and (auc >= 0.54),
        "decision":         decision,
    }


def train_shadow_ats_experiment(
    experiment_name: str,
    custom_feat_cols: list[str],
    sport: str = "MLB",
    min_season: int | None = None,
) -> dict:
    """
    Train an ATS shadow experiment model with a custom feature list.

    Same architecture as train_diagnostic_sport_model (GBC+isotonic, spread-aware
    augmentation for MLB), but:
      - Uses custom_feat_cols instead of MODEL_FEATURE_COLS
      - Saves to shadow/ directory with experiment_name prefix
      - Never overwrites production models
      - Returns training metrics for comparison

    Example: train_shadow_ats_experiment("whip_ats", MODEL_FEATURE_COLS_WHIP_ATS)
    """
    _assert_training_env()
    try:
        from features import build_feature_matrix, TARGET_ATS
    except ImportError:
        return {"error": "features.py not importable"}

    matrix = build_feature_matrix(sport)
    if matrix.empty:
        return {"error": f"No {sport} data in historical.db"}

    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    if min_season is not None:
        _gd = pd.to_datetime(labeled["game_date"], errors="coerce")
        labeled = labeled[_gd.dt.year >= min_season].copy()

    if len(labeled) < 50:
        return {"error": f"Only {len(labeled)} labeled {sport} rows — need ≥50"}

    feat_cols = [c for c in custom_feat_cols if c in labeled.columns]
    missing = [c for c in custom_feat_cols if c not in labeled.columns]
    if missing:
        log.warning("[shadow-exp] %s: %d missing features: %s",
                    experiment_name, len(missing), missing[:10])

    # MLB spread-aware augmentation (same as production)
    _SPREAD_AWARE = (sport == "MLB")
    _AUGMENT_SPREADS = [-1.5, 1.5, 2.5, 3.5, 4.5]
    if _SPREAD_AWARE:
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

    X = labeled[feat_cols].values.astype(float)
    y = labeled[TARGET_ATS].values.astype(float)

    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if len(set(y_te)) < 2:
        return {"error": f"Test set has only one class"}

    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_te_s = sc.transform(X_te_imp)

    from sklearn.calibration import CalibratedClassifierCV
    _base = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf = CalibratedClassifierCV(_base, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    y_prob = clf.predict_proba(X_te_s)[:, 1]
    auc = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    acc = float(accuracy_score(y_te, (y_prob >= 0.5).astype(int)))

    # Feature importances
    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])[:20]

    # Save to shadow directory
    import datetime as _dt
    os.makedirs(_SHADOW_DIR, exist_ok=True)
    prefix = experiment_name.lower()
    for artifact, suffix in [(clf, "clf"), (sc, "scaler"), (imp, "imputer"), (feat_cols, "feat_cols")]:
        path = os.path.join(_SHADOW_DIR, f"{prefix}_{suffix}.pkl")
        with open(path, "wb") as f:
            pickle.dump(artifact, f)

    meta = {
        "experiment": experiment_name,
        "sport": sport.upper(),
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "accuracy": round(acc, 4),
        "n_features": len(feat_cols),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "spread_aware": _SPREAD_AWARE,
        "min_season": min_season,
        "missing_features": missing,
        "trained_at": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(_SHADOW_DIR, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("[shadow-exp] %s trained: AUC=%.4f Brier=%.4f (%d features, %d train rows)",
             experiment_name, auc, brier, len(feat_cols), len(X_tr))

    return {
        "experiment": experiment_name,
        "sport": sport,
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "accuracy_pct": round(acc * 100, 1),
        "n_features": len(feat_cols),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "missing_features": missing,
        "top20_importances": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "saved_to": _SHADOW_DIR,
        "artifacts": [f"{prefix}_{s}.pkl" for s in ("clf", "scaler", "imputer", "feat_cols")],
    }


def train_shadow_totals_experiment(
    experiment_name: str,
    custom_feat_cols: list[str],
    min_season: int | None = None,
) -> dict:
    """
    Train a TOTALS shadow experiment model with a custom feature list.

    Same architecture as train_mlb_totals_model (GBC+isotonic, total-line
    augmentation, per-line correction), but:
      - Uses custom_feat_cols instead of MODEL_FEATURE_COLS
      - Saves to shadow/ directory with experiment_name prefix
      - Never overwrites production models
      - Returns training metrics for comparison
    """
    _assert_training_env()
    try:
        from features import build_feature_matrix
    except ImportError:
        return {"error": "features.py not importable"}

    matrix = build_feature_matrix("MLB")
    if matrix.empty:
        return {"error": "No MLB data in historical.db"}

    matrix["_total_runs"] = (
        pd.to_numeric(matrix["home_score"], errors="coerce")
        + pd.to_numeric(matrix["away_score"], errors="coerce")
    )
    labeled = matrix[matrix["_total_runs"].notna() & (matrix["_total_runs"] > 0)].copy()

    if min_season is not None:
        _gd = pd.to_datetime(labeled["game_date"], errors="coerce")
        labeled = labeled[_gd.dt.year >= min_season].copy()

    if len(labeled) < 50:
        return {"error": f"Only {len(labeled)} labeled MLB rows — need ≥50"}

    feat_cols = [c for c in custom_feat_cols if c in labeled.columns]
    missing = [c for c in custom_feat_cols if c not in labeled.columns]
    if missing:
        log.warning("[shadow-totals] %s: %d missing features: %s",
                    experiment_name, len(missing), missing[:10])

    # Totals-aware data augmentation
    _AUGMENT_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    _all_totals = labeled["_total_runs"]
    _LINE_BASE_RATES = {}
    for lv in _AUGMENT_LINES:
        _over_ct = (_all_totals > lv).sum()
        _total_ct = ((_all_totals > lv) | (_all_totals < lv)).sum()
        _LINE_BASE_RATES[lv] = _over_ct / max(_total_ct, 1)

    total_runs = labeled["_total_runs"]
    aug_rows = []
    for line_val in _AUGMENT_LINES:
        chunk = labeled.copy()
        chunk["_total_line"] = float(line_val)
        chunk["_total_line_abs"] = abs(float(line_val))
        chunk["_is_high_total"] = 1.0 if line_val >= 9.5 else 0.0
        chunk["_line_over_base_rate"] = _LINE_BASE_RATES[line_val]
        ou_val = total_runs - line_val
        chunk["_ou_label"] = np.where(ou_val > 0, 1.0,
                             np.where(ou_val < 0, 0.0, np.nan))
        aug_rows.append(chunk)
    labeled = pd.concat(aug_rows, ignore_index=True)
    labeled = labeled[labeled["_ou_label"].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    feat_cols = feat_cols + ["_total_line", "_total_line_abs", "_is_high_total",
                             "_line_over_base_rate"]

    X = labeled[feat_cols].values.astype(float)
    y = labeled["_ou_label"].values.astype(float)

    # 70/10/20 split: train / calibration / test
    split_cal = int(len(X) * 0.70)
    split_te = int(len(X) * 0.80)
    X_tr, X_cal, X_te = X[:split_cal], X[split_cal:split_te], X[split_te:]
    y_tr, y_cal, y_te = y[:split_cal], y[split_cal:split_te], y[split_te:]

    if len(set(y_te)) < 2:
        return {"error": "Test set has only one class"}

    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_cal_imp = imp.transform(X_cal)
    X_te_imp = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_cal_s = sc.transform(X_cal_imp)
    X_te_s = sc.transform(X_te_imp)

    from sklearn.calibration import CalibratedClassifierCV
    _base = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf_iso = CalibratedClassifierCV(_base, cv=3, method="isotonic")
    clf_iso.fit(X_tr_s, y_tr)

    y_prob_raw = clf_iso.predict_proba(X_te_s)[:, 1]

    # Per-line correction (computed on calibration set)
    y_cal_raw = clf_iso.predict_proba(X_cal_s)[:, 1]
    _cal_lines = labeled.iloc[split_cal:split_te]["_total_line"].values
    _line_corrections = {}
    for lv in _AUGMENT_LINES:
        _lm = _cal_lines == lv
        if _lm.sum() < 10:
            continue
        _pred_mean = float(y_cal_raw[_lm].mean())
        _actual_mean = float(y_cal[_lm].mean())
        _line_corrections[lv] = _actual_mean - _pred_mean

    # Apply corrections to test set
    _te_lines = labeled.iloc[split_te:]["_total_line"].values
    y_prob = y_prob_raw.copy()
    for lv, corr in _line_corrections.items():
        _lm = _te_lines == lv
        y_prob[_lm] = np.clip(y_prob_raw[_lm] + corr, 0.01, 0.99)

    clf = PerLineCorrectedClassifier(clf_iso, _line_corrections)

    auc = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    acc = float(accuracy_score(y_te, (y_prob >= 0.5).astype(int)))

    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf_iso.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])[:20]

    # Save to shadow directory
    import datetime as _dt
    os.makedirs(_SHADOW_DIR, exist_ok=True)
    prefix = experiment_name.lower()
    for artifact, suffix in [(clf, "clf"), (sc, "scaler"), (imp, "imputer"), (feat_cols, "feat_cols")]:
        path = os.path.join(_SHADOW_DIR, f"{prefix}_{suffix}.pkl")
        with open(path, "wb") as f:
            pickle.dump(artifact, f)

    meta = {
        "experiment": experiment_name,
        "sport": "MLB",
        "target": "totals_over",
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "accuracy_pct": round(acc * 100, 1),
        "n_features": len(feat_cols),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "missing_features": missing,
        "line_corrections": {str(k): round(v, 4) for k, v in _line_corrections.items()},
        "trained_at": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(_SHADOW_DIR, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("[shadow-totals] %s trained: AUC=%.4f Brier=%.4f (%d features, %d train rows)",
             experiment_name, auc, brier, len(feat_cols), len(X_tr))

    return {
        "experiment": experiment_name,
        "sport": "MLB",
        "target": "totals_over",
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "accuracy_pct": round(acc * 100, 1),
        "n_features": len(feat_cols),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "missing_features": missing,
        "top20_importances": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "saved_to": _SHADOW_DIR,
        "artifacts": [f"{prefix}_{s}.pkl" for s in ("clf", "scaler", "imputer", "feat_cols")],
    }


def load_sport_ats_model(sport: str):
    """
    Load a saved ATS sub-model for *sport*.
    Returns (clf, scaler, imputer, feat_cols) or None if not saved.
    """
    prefix = sport.lower()
    clf_path  = os.path.join(SUBMODEL_DIR, f"{prefix}_ats_clf.pkl")
    if not os.path.exists(clf_path):
        return None
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_imputer.pkl"), "rb") as f:
        imp = pickle.load(f)
    with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_feat_cols.pkl"), "rb") as f:
        feat_cols = pickle.load(f)
    return clf, sc, imp, feat_cols


# ── MLB totals sub-model ─────────────────────────────────────────────────────

class PerLineCorrectedClassifier:
    """CalibratedClassifierCV wrapper with per-line additive correction."""
    def __init__(self, base_clf, corrections):
        self.base_clf = base_clf
        self.corrections = corrections  # {line: shift}
        self.classes_ = base_clf.classes_
    def predict_proba(self, X, total_lines=None):
        raw = self.base_clf.predict_proba(X)[:, 1]
        if total_lines is not None:
            cal = raw.copy()
            for i, tl in enumerate(total_lines):
                corr = self.corrections.get(tl)
                if corr is not None:
                    cal[i] = np.clip(raw[i] + corr, 0.01, 0.99)
                else:
                    # Interpolate correction from nearest lines
                    keys = sorted(self.corrections.keys())
                    if tl <= keys[0]:
                        cal[i] = np.clip(raw[i] + self.corrections[keys[0]], 0.01, 0.99)
                    elif tl >= keys[-1]:
                        cal[i] = np.clip(raw[i] + self.corrections[keys[-1]], 0.01, 0.99)
                    else:
                        for j in range(len(keys) - 1):
                            if keys[j] <= tl <= keys[j + 1]:
                                frac = (tl - keys[j]) / (keys[j + 1] - keys[j])
                                c = self.corrections[keys[j]] + frac * (
                                    self.corrections[keys[j + 1]] - self.corrections[keys[j]])
                                cal[i] = np.clip(raw[i] + c, 0.01, 0.99)
                                break
            return np.column_stack([1 - cal, cal])
        return np.column_stack([1 - raw, raw])


def _load_totals_submodel(sport: str = "MLB"):
    """
    Load the MLB totals sub-model artifacts.
    Returns (clf, scaler, imputer, feat_cols) or None if not available.
    """
    prefix = _canonical_sport(sport).lower()
    paths = {
        "clf":  os.path.join(SUBMODEL_DIR, f"{prefix}_totals_clf.pkl"),
        "sc":   os.path.join(SUBMODEL_DIR, f"{prefix}_totals_scaler.pkl"),
        "imp":  os.path.join(SUBMODEL_DIR, f"{prefix}_totals_imputer.pkl"),
        "cols": os.path.join(SUBMODEL_DIR, f"{prefix}_totals_feat_cols.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    try:
        with open(paths["clf"],  "rb") as f: clf       = pickle.load(f)
        with open(paths["sc"],   "rb") as f: sc        = pickle.load(f)
        with open(paths["imp"],  "rb") as f: imp       = pickle.load(f)
        with open(paths["cols"], "rb") as f: feat_cols = pickle.load(f)
    except Exception as _e:
        print(f"[_load_totals_submodel] WARNING: {sport} totals model load failed ({_e}).")
        return None
    return clf, sc, imp, feat_cols


_TOTALS_RE = re.compile(r'(?:Over|Under)\s+[+-]?(\d+(?:\.\d+)?)', re.IGNORECASE)

def _extract_total_line(description: str) -> tuple[float | None, str | None]:
    """
    Extract total line and direction from a totals leg description.
    Examples:
        "Over 8.5"   → (8.5, 'over')
        "Under 10.5" → (10.5, 'under')
    Returns (line, direction) or (None, None) if not parseable.
    """
    if not description:
        return None, None
    desc_lower = description.lower().strip()
    direction = None
    if desc_lower.startswith("over"):
        direction = "over"
    elif desc_lower.startswith("under"):
        direction = "under"
    m = _TOTALS_RE.search(description)
    if m:
        try:
            line = float(m.group(1))
            return line, direction
        except ValueError:
            pass
    return None, direction


# Historical over rates by line value (MLB 2025-2026, 2874 games).
# Used as a strong prior feature for the totals sub-model.
# Interpolate for lines not in the table.
_TOTALS_BASE_RATES = {
    5.5: 0.747, 6.5: 0.670, 7.5: 0.557, 8.5: 0.489,
    9.5: 0.396, 10.5: 0.329, 11.5: 0.251, 12.5: 0.203,
    13.5: 0.155, 14.5: 0.108,
}

def _line_over_base_rate(line: float) -> float:
    """Return historical over rate for a given total line via interpolation."""
    keys = sorted(_TOTALS_BASE_RATES.keys())
    if line <= keys[0]:
        return _TOTALS_BASE_RATES[keys[0]]
    if line >= keys[-1]:
        return _TOTALS_BASE_RATES[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= line <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (line - lo) / (hi - lo)
            return _TOTALS_BASE_RATES[lo] + frac * (_TOTALS_BASE_RATES[hi] - _TOTALS_BASE_RATES[lo])
    return 0.45  # fallback


def predict_game_totals(
    home_team:  str,
    away_team:  str,
    total_line: float,
    game_date:  str | None = None,
    sport:      str = "MLB",
) -> float | None:
    """
    Predict the probability that a game goes OVER the given total line.

    Parameters
    ----------
    total_line : The over/under line (e.g. 8.5).
    Returns a float in [0, 1] or None if sub-model unavailable or no game match.
    """
    sport = _canonical_sport(sport)

    artifacts = _load_totals_submodel(sport)
    if artifacts is None:
        return None
    clf, sc, imp, feat_cols = artifacts

    try:
        matrix = _get_feature_matrix(sport)
        if matrix.empty:
            return None

        ht = _normalize_team_name(sport, home_team).lower().strip()
        at = _normalize_team_name(sport, away_team).lower().strip()
        mask = (
            matrix["home_team"].str.lower().str.strip() == ht
        ) & (
            matrix["away_team"].str.lower().str.strip() == at
        )
        if game_date:
            date_str = str(game_date)[:10]
            mask = mask & (matrix["game_date"].astype(str).str[:10] == date_str)

        matched = matrix[mask]
        if matched.empty:
            return None

        row = matched.sort_values("game_date", ascending=False).iloc[0]

        def _val(col: str) -> float:
            if col == "_total_line":
                return float(total_line)
            if col == "_total_line_abs":
                return abs(float(total_line))
            if col == "_is_high_total":
                return 1.0 if total_line >= 9.5 else 0.0
            if col == "_line_over_base_rate":
                return _line_over_base_rate(total_line)
            return row[col] if col in row.index and not pd.isna(row[col]) else np.nan

        X = np.array([[_val(c) for c in feat_cols]], dtype=float)
        X_imp = imp.transform(X)
        X_s   = sc.transform(X_imp)
        # Pass total_line for per-line calibration if the model supports it
        if hasattr(clf, 'predict_proba'):
            try:
                proba = clf.predict_proba(X_s, total_lines=np.array([total_line]))
            except TypeError:
                proba = clf.predict_proba(X_s)
        else:
            proba = clf.predict_proba(X_s)
        return float(proba[0][1])

    except Exception as exc:
        log.warning("predict_game_totals(%s vs %s, line=%s): %s",
                    home_team, away_team, total_line, exc)
        return None


def train_mlb_totals_model(
    save:       bool = False,
    min_season: int | None = None,
) -> dict:
    """
    Train an MLB totals sub-model using the same architecture as the ATS
    sub-model (GBC + isotonic calibration) but targeting over/under outcomes.

    Data augmentation: each game is expanded into rows at standard total lines
    [6.5, 7.5, 8.5, 9.5, 10.5, 11.5].  Label = 1 if total_runs > line (over),
    0 if total_runs < line (under), excluded if push.

    Totals-specific features added: _total_line, _total_line_abs, _is_high_total.

    Saves to SUBMODEL_DIR/mlb_totals_*.pkl when save=True and AUC >= 0.54.
    """
    if save:
        _assert_training_env()
    try:
        from features import build_feature_matrix, MODEL_FEATURE_COLS
    except ImportError:
        return {"error": "features.py not importable"}

    # ── 1. Load feature matrix ────────────────────────────────────────────────
    matrix = build_feature_matrix("MLB")
    if matrix.empty:
        return {"error": "No MLB data in historical.db"}

    # Need home_score + away_score for label computation
    matrix["_total_runs"] = (
        pd.to_numeric(matrix["home_score"], errors="coerce")
        + pd.to_numeric(matrix["away_score"], errors="coerce")
    )
    labeled = matrix[matrix["_total_runs"].notna() & (matrix["_total_runs"] > 0)].copy()

    # ── 1b. Optional training window filter ───────────────────────────────────
    if min_season is not None:
        _gd = pd.to_datetime(labeled["game_date"], errors="coerce")
        _before = len(labeled)
        labeled = labeled[_gd.dt.year >= min_season].copy()
        log.info("  min_season=%d: %d → %d games", min_season, _before, len(labeled))

    if len(labeled) < 50:
        return {"error": f"Only {len(labeled)} labeled MLB rows — need ≥50"}

    # ── 2. Build feature columns ──────────────────────────────────────────────
    feat_cols = [c for c in MODEL_FEATURE_COLS if c in labeled.columns]

    # ── 2b. Totals-aware data augmentation ────────────────────────────────────
    _AUGMENT_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    # Pre-compute base over rate at each line from training data (strong prior)
    _all_totals = labeled["_total_runs"]
    _LINE_BASE_RATES = {}
    for lv in _AUGMENT_LINES:
        _over_ct = (_all_totals > lv).sum()
        _total_ct = ((_all_totals > lv) | (_all_totals < lv)).sum()  # excl pushes
        _LINE_BASE_RATES[lv] = _over_ct / max(_total_ct, 1)

    total_runs = labeled["_total_runs"]
    aug_rows = []
    for line_val in _AUGMENT_LINES:
        chunk = labeled.copy()
        chunk["_total_line"]          = float(line_val)
        chunk["_total_line_abs"]      = abs(float(line_val))
        chunk["_is_high_total"]       = 1.0 if line_val >= 9.5 else 0.0
        chunk["_line_over_base_rate"] = _LINE_BASE_RATES[line_val]
        # Label: 1 = over covers, 0 = under covers, NaN = push
        ou_val = total_runs - line_val
        chunk["_ou_label"] = np.where(ou_val > 0, 1.0,
                             np.where(ou_val < 0, 0.0, np.nan))
        aug_rows.append(chunk)
    labeled = pd.concat(aug_rows, ignore_index=True)
    labeled = labeled[labeled["_ou_label"].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    feat_cols = feat_cols + ["_total_line", "_total_line_abs", "_is_high_total",
                             "_line_over_base_rate"]
    log.info("  MLB totals augmented: %d rows across %d line values",
             len(labeled), len(_AUGMENT_LINES))

    X = labeled[feat_cols].values.astype(float)
    y = labeled["_ou_label"].values.astype(float)

    over_rate = float(y.mean())
    # 70/10/20 split: train / calibration / test
    split_cal = int(len(X) * 0.70)
    split_te  = int(len(X) * 0.80)
    X_tr, X_cal, X_te = X[:split_cal], X[split_cal:split_te], X[split_te:]
    y_tr, y_cal, y_te = y[:split_cal], y[split_cal:split_te], y[split_te:]

    if len(set(y_te)) < 2:
        return {"error": "MLB totals test set has only one class — need more data"}

    # ── 3. Impute → scale ─────────────────────────────────────────────────────
    imp      = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_cal_imp = imp.transform(X_cal)
    X_te_imp = imp.transform(X_te)

    sc       = StandardScaler()
    X_tr_s   = sc.fit_transform(X_tr_imp)
    X_cal_s  = sc.transform(X_cal_imp)
    X_te_s   = sc.transform(X_te_imp)

    # ── 4. Train GBC + isotonic calibration ───────────────────────────────────
    from sklearn.calibration import CalibratedClassifierCV
    _base = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf_iso = CalibratedClassifierCV(_base, cv=3, method="isotonic")
    clf_iso.fit(X_tr_s, y_tr)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    y_prob_raw = clf_iso.predict_proba(X_te_s)[:, 1]

    # ── 5b. Per-line correction (computed on calibration set, applied to test) ─
    # The global isotonic calibrator can't condition on line value, so predictions
    # at extreme lines (6.5, 11.5) are poorly calibrated.  Compute a per-line
    # additive correction on the held-out calibration set.
    y_cal_raw = clf_iso.predict_proba(X_cal_s)[:, 1]
    _cal_lines = labeled.iloc[split_cal:split_te]["_total_line"].values
    _line_corrections = {}
    for lv in _AUGMENT_LINES:
        _lm = _cal_lines == lv
        if _lm.sum() < 10:
            continue
        _pred_mean = float(y_cal_raw[_lm].mean())
        _actual_mean = float(y_cal[_lm].mean())
        _line_corrections[lv] = _actual_mean - _pred_mean

    # Apply corrections to TEST set for evaluation
    _te_lines = labeled.iloc[split_te:]["_total_line"].values
    y_prob = y_prob_raw.copy()
    for lv, corr in _line_corrections.items():
        _lm = _te_lines == lv
        y_prob[_lm] = np.clip(y_prob_raw[_lm] + corr, 0.01, 0.99)

    # Wrap clf + corrections
    clf = PerLineCorrectedClassifier(clf_iso, _line_corrections)

    y_pred = (y_prob >= 0.5).astype(int)

    auc   = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    acc   = float(accuracy_score(y_te, y_pred))

    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf_iso.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])[:15]

    # ── 6. Calibration buckets ────────────────────────────────────────────────
    buckets: list[dict] = []
    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
        mask = (y_prob >= lo) & (y_prob < hi)
        n    = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket":      f"{int(lo*100)}-{int(hi*100)}%",
            "n":           n,
            "pred_prob":   round(float(y_prob[mask].mean()) * 100, 1),
            "actual_rate": round(float(y_te[mask].mean()) * 100, 1),
            "gap":         round((float(y_te[mask].mean()) - float(y_prob[mask].mean())) * 100, 1),
        })

    # ── 6b. Per-line calibration ──────────────────────────────────────────────
    line_cal: list[dict] = []
    _te_line = labeled.iloc[split_te:]["_total_line"].values
    for lv in _AUGMENT_LINES:
        _sm = _te_line == lv
        _n  = int(_sm.sum())
        if _n == 0:
            continue
        line_cal.append({
            "line":        lv,
            "n":           _n,
            "pred_prob":   round(float(y_prob[_sm].mean()) * 100, 1),
            "actual_rate": round(float(y_te[_sm].mean()) * 100, 1),
            "gap":         round((float(y_te[_sm].mean()) - float(y_prob[_sm].mean())) * 100, 1),
        })

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    if auc >= 0.54 and save:
        import datetime as _dt
        prefix = "mlb"
        _ver_meta = {"sport": "MLB", "target": "totals_over", "auc": round(auc, 4),
                     "brier": round(brier, 4), "n_features": len(feat_cols),
                     "n_train": int(len(X_tr_s)), "min_season": min_season}
        _versioned_save(clf,       os.path.join(SUBMODEL_DIR, f"{prefix}_totals_clf.pkl"),
                        f"{prefix}_totals_clf", _ver_meta)
        _versioned_save(sc,        os.path.join(SUBMODEL_DIR, f"{prefix}_totals_scaler.pkl"),
                        f"{prefix}_totals_scaler", _ver_meta)
        _versioned_save(imp,       os.path.join(SUBMODEL_DIR, f"{prefix}_totals_imputer.pkl"),
                        f"{prefix}_totals_imputer", _ver_meta)
        _versioned_save(feat_cols, os.path.join(SUBMODEL_DIR, f"{prefix}_totals_feat_cols.pkl"),
                        f"{prefix}_totals_feat_cols", _ver_meta)
        meta = {
            "sport":        "MLB",
            "target":       "totals_over",
            "auc":          round(auc, 4),
            "brier":        round(brier, 4),
            "n_features":   len(feat_cols),
            "n_train":      int(len(X_tr_s)),
            "augment_lines": _AUGMENT_LINES,
            "min_season":   min_season,
            "saved_at":     _dt.datetime.utcnow().isoformat() + "Z",
        }
        _versioned_save(meta, os.path.join(SUBMODEL_DIR, f"{prefix}_totals_meta.json"),
                        f"{prefix}_totals_meta", _ver_meta)

        # Archive v1
        archive_dir = os.path.join(SUBMODEL_DIR, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        _date_tag = _dt.datetime.utcnow().strftime("%Y%m%d")
        for suffix in ("clf", "scaler", "imputer", "feat_cols"):
            src = os.path.join(SUBMODEL_DIR, f"{prefix}_totals_{suffix}.pkl")
            dst = os.path.join(archive_dir, f"{prefix}_totals_v1_{_date_tag}_{suffix}.pkl")
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)

        log.info("train_mlb_totals_model: saved (AUC=%.4f, Brier=%.4f, %d features)",
                 auc, brier, len(feat_cols))

    if auc >= 0.54:
        decision = f"WORTH BUILDING: MLB totals sub-model justified (AUC={auc:.4f} >= 0.54)"
    elif auc >= 0.50:
        decision = f"WAIT: MLB totals model won't help yet (0.50 <= AUC < 0.54)"
    else:
        decision = f"FEATURE WORK NEEDED: MLB totals features need engineering (AUC < 0.50)"

    return {
        "sport":             "MLB",
        "target":            "totals_over",
        "n_labeled":         len(labeled),
        "n_train":           len(X_tr),
        "n_test":            len(X_te),
        "over_rate_all":     round(over_rate * 100, 1),
        "over_rate_train":   round(float(y_tr.mean()) * 100, 1),
        "over_rate_test":    round(float(y_te.mean()) * 100, 1),
        "roc_auc":           round(auc, 4),
        "brier_score":       round(brier, 4),
        "accuracy_pct":      round(acc * 100, 1),
        "n_features":        len(feat_cols),
        "feature_cols":      feat_cols,
        "top15_importances": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "calibration_buckets": buckets,
        "line_calibration":    line_cal,
        "augment_lines":       _AUGMENT_LINES,
        "saved":               save and (auc >= 0.54),
        "decision":            decision,
    }


# ── NBA isolation test (backward compat wrapper) ───────────────────────────────

def train_nba_only_model() -> dict:
    """
    Backward-compatible wrapper for train_diagnostic_sport_model("NBA").
    Does NOT save the model.
    """
    return train_diagnostic_sport_model("NBA", save=False)


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW ATS RETRAIN — V2 leak-free, fixed spread encoding
# ═══════════════════════════════════════════════════════════════════════════════
# Saves artifacts to a SEPARATE path (mlb_ats_v2_shadow_*).
# Does NOT overwrite production models. Does NOT wire into inference.
# The production model + implied override continue running untouched.

_SHADOW_DIR = os.path.join(SUBMODEL_DIR, "shadow")

# V2 spread augmentation: symmetric, no _is_fav_spread
_V2_AUGMENT_SPREADS = [-4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5]


def train_ats_shadow(sport: str = "MLB", save: bool = True) -> dict:
    """
    Train the V2 ATS sub-model in shadow mode.

    Changes vs V1 (train_diagnostic_sport_model):
      1. Uses build_feature_matrix_v2 (leak-free: no season stats)
      2. Spread features: _spread_value + _spread_abs + _spread_sq (NO _is_fav_spread)
      3. Augmentation: symmetric [-4.5..+5.5] (was [-1.5..+4.5])
      4. monotone_constraints on _spread_value (more cushion → higher cover)
      5. Saves to shadow/ subdirectory, never overwrites production

    Returns a results dict with AUC, Brier, calibration, etc.
    """
    _assert_training_env()

    from features import (
        build_feature_matrix_v2,
        MODEL_FEATURE_COLS_V2,
        TARGET_ATS,
    )
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
    from sklearn.calibration import CalibratedClassifierCV

    log.info("train_ats_shadow: building V2 feature matrix for %s", sport)
    matrix = build_feature_matrix_v2(sport)
    if matrix.empty:
        return {"error": f"No feature data for {sport}"}

    # ── 1. Label: ATS target (home covered spread) ─────────────────────────
    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    log.info("  V2 labeled rows (pre-augmentation): %d", len(labeled))

    # ── 2. Feature columns (leak-free V2 set) ─────────────────────────────
    feat_cols = [c for c in MODEL_FEATURE_COLS_V2 if c in labeled.columns]
    log.info("  V2 base features: %d", len(feat_cols))

    # ── 3. Spread-aware augmentation (symmetric, no _is_fav_spread) ────────
    margin = (
        pd.to_numeric(labeled["home_score"], errors="coerce")
        - pd.to_numeric(labeled["away_score"], errors="coerce")
    )
    aug_rows = []
    for spread_val in _V2_AUGMENT_SPREADS:
        chunk = labeled.copy()
        chunk["_spread_value"] = float(spread_val)
        chunk["_spread_abs"]   = abs(spread_val)
        chunk["_spread_sq"]    = spread_val ** 2
        # NO _is_fav_spread — this is the fix
        ats_val = margin + spread_val
        chunk[TARGET_ATS] = np.where(ats_val > 0, 1.0,
                            np.where(ats_val < 0, 0.0, np.nan))
        aug_rows.append(chunk)

    labeled = pd.concat(aug_rows, ignore_index=True)
    labeled = labeled[labeled[TARGET_ATS].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    feat_cols = feat_cols + ["_spread_value", "_spread_abs", "_spread_sq"]
    log.info("  V2 augmented: %d rows across %d spread values, %d features",
             len(labeled), len(_V2_AUGMENT_SPREADS), len(feat_cols))

    # ── 4. Train/test split (temporal, 80/20) ──────────────────────────────
    X = labeled[feat_cols].values.astype(float)
    y = labeled[TARGET_ATS].values.astype(float)

    cover_rate = float(y.mean())
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if len(set(y_te)) < 2:
        return {"error": f"{sport} V2 test set has only one class"}

    # ── 5. Impute (median) → scale ─────────────────────────────────────────
    # NaN rows (~21% for pitcher features) are imputed, NOT dropped.
    # This avoids biasing toward long-history pitchers.
    # keep_empty_features=True preserves all-NaN columns (filled with 0)
    # so the feature count stays aligned with monotonic_cst.
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_te_s = sc.transform(X_te_imp)

    # ── 6. Train HistGBC with monotone constraint on _spread_value ────────
    # HistGradientBoostingClassifier supports monotonic_cst natively.
    # _spread_value gets +1: larger spread → higher P(home covers).
    from sklearn.ensemble import HistGradientBoostingClassifier as _HistGBC

    sv_idx = feat_cols.index("_spread_value")
    mono = [0] * len(feat_cols)
    mono[sv_idx] = 1  # enforce: more cushion → higher cover prob
    log.info("  V2 monotonic_cst: _spread_value at index %d = +1", sv_idx)

    _base = _HistGBC(
        max_iter=500,
        learning_rate=0.03,
        max_depth=4,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_features=0.8,
        monotonic_cst=mono,
        early_stopping=False,
        random_state=42,
    )
    clf = CalibratedClassifierCV(_base, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    # ── 7. Evaluate ────────────────────────────────────────────────────────
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc   = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    acc   = float(accuracy_score(y_te, y_pred))

    # Feature importances (averaged across calibration folds)
    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])[:20]

    # ── 8. Calibration buckets ─────────────────────────────────────────────
    buckets = []
    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55),
                   (0.55, 0.65), (0.65, 1.01)]:
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket":      f"{int(lo*100)}-{int(hi*100)}%",
            "n":           n,
            "pred_prob":   round(float(y_prob[mask].mean()) * 100, 1),
            "actual_rate": round(float(y_te[mask].mean()) * 100, 1),
            "gap":         round((float(y_te[mask].mean()) - float(y_prob[mask].mean())) * 100, 1),
        })

    # ── 9. Per-spread calibration ──────────────────────────────────────────
    _te_spread = labeled.iloc[split:]["_spread_value"].values
    spread_cal = []
    for sv in _V2_AUGMENT_SPREADS:
        _sm = _te_spread == sv
        _n = int(_sm.sum())
        if _n == 0:
            continue
        spread_cal.append({
            "spread":      sv,
            "n":           _n,
            "pred_prob":   round(float(y_prob[_sm].mean()) * 100, 1),
            "actual_rate": round(float(y_te[_sm].mean()) * 100, 1),
            "gap":         round((float(y_te[_sm].mean()) - float(y_prob[_sm].mean())) * 100, 1),
        })

    # ── 10. Save shadow artifacts (never overwrites production) ────────────
    saved = False
    if save and auc >= 0.50:
        import json as _json, datetime as _dt
        os.makedirs(_SHADOW_DIR, exist_ok=True)
        prefix = sport.lower()
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v2_clf.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v2_scaler.pkl"), "wb") as f:
            pickle.dump(sc, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v2_imputer.pkl"), "wb") as f:
            pickle.dump(imp, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v2_feat_cols.pkl"), "wb") as f:
            pickle.dump(feat_cols, f)
        meta = {
            "sport":          sport.upper(),
            "version":        "v2_shadow",
            "auc":            round(auc, 4),
            "brier":          round(brier, 4),
            "n_features":     len(feat_cols),
            "n_train":        int(len(X_tr_s)),
            "n_test":         int(len(X_te_s)),
            "spread_aware":   True,
            "augment_spreads": _V2_AUGMENT_SPREADS,
            "has_is_fav_spread": False,
            "monotone_constraints": True,
            "saved_at":       _dt.datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v2_meta.json"), "w") as f:
            _json.dump(meta, f)
        saved = True
        log.info("train_ats_shadow: saved V2 shadow model (AUC=%.4f, Brier=%.4f, %d features)",
                 auc, brier, len(feat_cols))

    return {
        "sport":              sport,
        "version":            "v2_shadow",
        "n_labeled":          len(labeled),
        "n_train":            len(X_tr),
        "n_test":             len(X_te),
        "cover_rate_all":     round(cover_rate * 100, 1),
        "cover_rate_train":   round(float(y_tr.mean()) * 100, 1),
        "cover_rate_test":    round(float(y_te.mean()) * 100, 1),
        "roc_auc":            round(auc, 4),
        "brier_score":        round(brier, 4),
        "accuracy_pct":       round(acc * 100, 1),
        "n_features":         len(feat_cols),
        "feature_cols":       feat_cols,
        "augment_spreads":    _V2_AUGMENT_SPREADS,
        "has_is_fav_spread":  False,
        "monotone_constraints": True,
        "top20_importances":  [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "calibration_buckets": buckets,
        "spread_calibration":  spread_cal,
        "saved":               saved,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ATS V3 SHADOW MODEL (V2 + rolling batting features)
# ═══════════════════════════════════════════════════════════════════════════════


def train_ats_v3_shadow(sport: str = "MLB", save: bool = True) -> dict:
    """
    Train ATS V3 shadow: V2 recipe + 38 rolling batting features.

    Identical to train_ats_shadow() except:
      - Uses MODEL_FEATURE_COLS_V3_ATS (V2 + batting)
      - Saves as v3 shadow artifacts (never overwrites V2)
      - Includes batting importance diagnostics

    Returns results dict with AUC, Brier, calibration, batting diagnostics.
    """
    _assert_training_env()

    from features import (
        build_feature_matrix_v2,
        MODEL_FEATURE_COLS_V3_ATS,
        _V2_ROLLING_BATTING_COLS,
        TARGET_ATS,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
    from sklearn.calibration import CalibratedClassifierCV

    log.info("train_ats_v3_shadow: building V2 feature matrix for %s", sport)
    matrix = build_feature_matrix_v2(sport)
    if matrix.empty:
        return {"error": f"No feature data for {sport}"}

    # ── 1. Label: ATS target (home covered spread) ─────────────────────────
    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    log.info("  V3 labeled rows (pre-augmentation): %d", len(labeled))

    # ── 2. Feature columns (V2 + batting) ──────────────────────────────────
    feat_cols = [c for c in MODEL_FEATURE_COLS_V3_ATS if c in labeled.columns]
    log.info("  V3 base features: %d (V2 had %d)", len(feat_cols),
             len(feat_cols) - len([c for c in _V2_ROLLING_BATTING_COLS if c in labeled.columns]))

    # ── 3. Spread-aware augmentation (same as V2) ─────────────────────────
    margin = (
        pd.to_numeric(labeled["home_score"], errors="coerce")
        - pd.to_numeric(labeled["away_score"], errors="coerce")
    )
    aug_rows = []
    for spread_val in _V2_AUGMENT_SPREADS:
        chunk = labeled.copy()
        chunk["_spread_value"] = float(spread_val)
        chunk["_spread_abs"]   = abs(spread_val)
        chunk["_spread_sq"]    = spread_val ** 2
        ats_val = margin + spread_val
        chunk[TARGET_ATS] = np.where(ats_val > 0, 1.0,
                            np.where(ats_val < 0, 0.0, np.nan))
        aug_rows.append(chunk)

    labeled = pd.concat(aug_rows, ignore_index=True)
    labeled = labeled[labeled[TARGET_ATS].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    feat_cols = feat_cols + ["_spread_value", "_spread_abs", "_spread_sq"]
    log.info("  V3 augmented: %d rows across %d spread values, %d features",
             len(labeled), len(_V2_AUGMENT_SPREADS), len(feat_cols))

    # ── 4. Train/test split (temporal, 80/20) ──────────────────────────────
    X = labeled[feat_cols].values.astype(float)
    y = labeled[TARGET_ATS].values.astype(float)

    cover_rate = float(y.mean())
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if len(set(y_te)) < 2:
        return {"error": f"{sport} V3 test set has only one class"}

    # ── 5. Impute (median) → scale ─────────────────────────────────────────
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_te_s = sc.transform(X_te_imp)

    # ── 6. Train HistGBC with monotone constraint on _spread_value ────────
    from sklearn.ensemble import HistGradientBoostingClassifier as _HistGBC

    sv_idx = feat_cols.index("_spread_value")
    mono = [0] * len(feat_cols)
    mono[sv_idx] = 1
    log.info("  V3 monotonic_cst: _spread_value at index %d = +1", sv_idx)

    _base = _HistGBC(
        max_iter=500,
        learning_rate=0.03,
        max_depth=4,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_features=0.8,
        monotonic_cst=mono,
        early_stopping=False,
        random_state=42,
    )
    clf = CalibratedClassifierCV(_base, cv=3, method="isotonic")
    clf.fit(X_tr_s, y_tr)

    # ── 7. Evaluate ────────────────────────────────────────────────────────
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc   = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    acc   = float(accuracy_score(y_te, y_pred))

    # Feature importances (averaged across calibration folds)
    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    _imp_sorted = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])
    imps = _imp_sorted[:30]

    # Batting feature diagnostics
    _batting_set = set(_V2_ROLLING_BATTING_COLS)
    _batting_feature_ranks = {}
    _batting_total_imp = 0.0
    for rank, (fname, fval) in enumerate(_imp_sorted, 1):
        if fname in _batting_set:
            _batting_feature_ranks[fname] = {
                "rank": rank, "importance_pct": round(float(fval) * 100, 2)
            }
            _batting_total_imp += fval

    # ── 8. Calibration buckets ─────────────────────────────────────────────
    buckets = []
    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55),
                   (0.55, 0.65), (0.65, 1.01)]:
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket":      f"{int(lo*100)}-{int(hi*100)}%",
            "n":           n,
            "pred_prob":   round(float(y_prob[mask].mean()) * 100, 1),
            "actual_rate": round(float(y_te[mask].mean()) * 100, 1),
            "gap":         round((float(y_te[mask].mean()) - float(y_prob[mask].mean())) * 100, 1),
        })

    # ── 9. Per-spread calibration ──────────────────────────────────────────
    _te_spread = labeled.iloc[split:]["_spread_value"].values
    spread_cal = []
    for sv in _V2_AUGMENT_SPREADS:
        _sm = _te_spread == sv
        _n = int(_sm.sum())
        if _n == 0:
            continue
        spread_cal.append({
            "spread":      sv,
            "n":           _n,
            "pred_prob":   round(float(y_prob[_sm].mean()) * 100, 1),
            "actual_rate": round(float(y_te[_sm].mean()) * 100, 1),
            "gap":         round((float(y_te[_sm].mean()) - float(y_prob[_sm].mean())) * 100, 1),
        })

    # ── 10. Leak-free proof ────────────────────────────────────────────────
    _KNOWN_LEAKY = {
        "h_ops", "a_ops", "h_slg", "a_slg",
        "h_k_pct", "a_k_pct", "h_bb_pct", "a_bb_pct",
        "h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip",
        "h_sp_k9", "a_sp_k9", "h_sp_wins", "a_sp_wins",
        "sp_era_diff", "sp_whip_diff", "sp_k9_diff", "sp_wins_diff",
        "h_sp_ip", "a_sp_ip",
    }
    _leaked = [c for c in feat_cols if c in _KNOWN_LEAKY]

    # ── 11. Save V3 shadow artifacts (never overwrites V2) ────────────────
    saved = False
    if save and auc >= 0.50:
        import json as _json, datetime as _dt
        os.makedirs(_SHADOW_DIR, exist_ok=True)
        prefix = sport.lower()
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v3_clf.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v3_scaler.pkl"), "wb") as f:
            pickle.dump(sc, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v3_imputer.pkl"), "wb") as f:
            pickle.dump(imp, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v3_feat_cols.pkl"), "wb") as f:
            pickle.dump(feat_cols, f)
        meta = {
            "sport":              sport.upper(),
            "version":            "v3_shadow",
            "auc":                round(auc, 4),
            "brier":              round(brier, 4),
            "n_features":         len(feat_cols),
            "n_train":            int(len(X_tr_s)),
            "n_test":             int(len(X_te_s)),
            "spread_aware":       True,
            "augment_spreads":    _V2_AUGMENT_SPREADS,
            "has_batting":        True,
            "batting_features":   len(_batting_feature_ranks),
            "monotone_constraints": True,
            "saved_at":           _dt.datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_ats_v3_meta.json"), "w") as f:
            _json.dump(meta, f)
        saved = True
        log.info("train_ats_v3_shadow: saved V3 shadow model (AUC=%.4f, Brier=%.4f, %d features)",
                 auc, brier, len(feat_cols))

    return {
        "sport":                   sport,
        "version":                 "v3_shadow_ats",
        "n_labeled":               len(labeled),
        "n_train":                 len(X_tr),
        "n_test":                  len(X_te),
        "cover_rate_all":          round(cover_rate * 100, 1),
        "cover_rate_train":        round(float(y_tr.mean()) * 100, 1),
        "cover_rate_test":         round(float(y_te.mean()) * 100, 1),
        "roc_auc":                 round(auc, 4),
        "brier_score":             round(brier, 4),
        "accuracy_pct":            round(acc * 100, 1),
        "n_features":              len(feat_cols),
        "feature_cols":            feat_cols,
        "augment_spreads":         _V2_AUGMENT_SPREADS,
        "monotone_constraints":    True,
        "batting_feature_ranks":   _batting_feature_ranks,
        "batting_total_importance_pct": round(_batting_total_imp * 100, 2),
        "leak_free_proof": {
            "leaky_features_found": _leaked,
            "leak_free":            len(_leaked) == 0,
        },
        "top30_importances": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "calibration_buckets":     buckets,
        "spread_calibration":      spread_cal,
        "saved":                   saved,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MLB TOTALS V2 SHADOW MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_mlb_totals_v2_shadow(
    save: bool = False,
) -> dict:
    """
    Train MLB totals V2 shadow model with three fixes vs V1:
      1. Uses build_feature_matrix_v2 (leak-free, matches inference)
      2. MLB-only training data (no multi-sport dilution)
      3. Park interaction features: park × SP ERA, park × team OPS proxies

    Architecture: GBC + CalibratedClassifierCV + PerLineCorrectedClassifier
    Saves to shadow/ subdirectory — never overwrites production.
    """
    _assert_training_env()

    from features import (
        build_feature_matrix_v2,
        MODEL_FEATURE_COLS_V2_TOTALS,
        _V2_ROLLING_BATTING_COLS,
    )
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
    from sklearn.calibration import CalibratedClassifierCV

    log.info("train_mlb_totals_v2_shadow: building V2 feature matrix for MLB")
    matrix = build_feature_matrix_v2("MLB")
    if matrix.empty:
        return {"error": "No MLB data from build_feature_matrix_v2"}

    # ── 1. Label: total runs ─────────────────────────────────────────────────
    matrix["_total_runs"] = (
        pd.to_numeric(matrix["home_score"], errors="coerce")
        + pd.to_numeric(matrix["away_score"], errors="coerce")
    )
    labeled = matrix[matrix["_total_runs"].notna() & (matrix["_total_runs"] > 0)].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    log.info("  V2 totals labeled rows (pre-augmentation): %d", len(labeled))

    if len(labeled) < 50:
        return {"error": f"Only {len(labeled)} labeled MLB rows — need ≥50"}

    # ── 2. Feature columns (V2 + rolling batting for totals) ─────────────────
    feat_cols = [c for c in MODEL_FEATURE_COLS_V2_TOTALS if c in labeled.columns]
    log.info("  V2 totals features (ATS + batting): %d", len(feat_cols))

    # ── 3. Park interaction features (Fix #3) ────────────────────────────────
    # park × SP ERA: high park factor + high ERA → inflated totals
    if "h_park_factor" in labeled.columns:
        if "h_sp_era_5" in labeled.columns:
            labeled["park_x_h_sp_era"] = labeled["h_park_factor"] * labeled["h_sp_era_5"]
        if "a_sp_era_5" in labeled.columns:
            labeled["park_x_a_sp_era"] = labeled["h_park_factor"] * labeled["a_sp_era_5"]
        # park × team offense proxy: pts_10g is the V2 analog of team OPS
        if "h_pts_10g" in labeled.columns:
            labeled["park_x_h_offense"] = labeled["h_park_factor"] * labeled["h_pts_10g"]
        if "a_pts_10g" in labeled.columns:
            labeled["park_x_a_offense"] = labeled["h_park_factor"] * labeled["a_pts_10g"]
        # park factor squared (nonlinear: extreme parks matter more)
        labeled["park_factor_sq"] = labeled["h_park_factor"] ** 2

    _park_interaction_cols = [
        c for c in [
            "park_x_h_sp_era", "park_x_a_sp_era",
            "park_x_h_offense", "park_x_a_offense",
            "park_factor_sq",
        ] if c in labeled.columns
    ]
    feat_cols = feat_cols + _park_interaction_cols
    log.info("  V2 totals park interaction features added: %d → %s",
             len(_park_interaction_cols), _park_interaction_cols)

    # ── 4. Totals-aware data augmentation ────────────────────────────────────
    _AUGMENT_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    _all_totals = labeled["_total_runs"]
    _LINE_BASE_RATES = {}
    for lv in _AUGMENT_LINES:
        _over_ct = (_all_totals > lv).sum()
        _total_ct = ((_all_totals > lv) | (_all_totals < lv)).sum()
        _LINE_BASE_RATES[lv] = _over_ct / max(_total_ct, 1)

    total_runs = labeled["_total_runs"]
    aug_rows = []
    for line_val in _AUGMENT_LINES:
        chunk = labeled.copy()
        chunk["_total_line"]          = float(line_val)
        chunk["_total_line_abs"]      = abs(float(line_val))
        chunk["_is_high_total"]       = 1.0 if line_val >= 9.5 else 0.0
        chunk["_line_over_base_rate"] = _LINE_BASE_RATES[line_val]
        ou_val = total_runs - line_val
        chunk["_ou_label"] = np.where(ou_val > 0, 1.0,
                             np.where(ou_val < 0, 0.0, np.nan))
        aug_rows.append(chunk)
    labeled = pd.concat(aug_rows, ignore_index=True)
    labeled = labeled[labeled["_ou_label"].notna()].copy()
    labeled = labeled.sort_values("game_date").reset_index(drop=True)
    feat_cols = feat_cols + ["_total_line", "_total_line_abs", "_is_high_total",
                             "_line_over_base_rate"]
    log.info("  V2 totals augmented: %d rows across %d line values, %d features",
             len(labeled), len(_AUGMENT_LINES), len(feat_cols))

    # ── 5. Train/calibration/test split (70/10/20 temporal) ──────────────────
    X = labeled[feat_cols].values.astype(float)
    y = labeled["_ou_label"].values.astype(float)

    over_rate = float(y.mean())
    split_cal = int(len(X) * 0.70)
    split_te  = int(len(X) * 0.80)
    X_tr, X_cal, X_te = X[:split_cal], X[split_cal:split_te], X[split_te:]
    y_tr, y_cal, y_te = y[:split_cal], y[split_cal:split_te], y[split_te:]

    if len(set(y_te)) < 2:
        return {"error": "V2 totals test set has only one class"}

    # ── 6. Impute → scale ────────────────────────────────────────────────────
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    X_tr_imp = imp.fit_transform(X_tr)
    X_cal_imp = imp.transform(X_cal)
    X_te_imp = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_cal_s = sc.transform(X_cal_imp)
    X_te_s = sc.transform(X_te_imp)

    # ── 7. Train GBC + isotonic calibration ──────────────────────────────────
    _base = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf_iso = CalibratedClassifierCV(_base, cv=3, method="isotonic")
    clf_iso.fit(X_tr_s, y_tr)

    # ── 8. Per-line correction (calibration set) ─────────────────────────────
    y_cal_raw = clf_iso.predict_proba(X_cal_s)[:, 1]
    _cal_lines = labeled.iloc[split_cal:split_te]["_total_line"].values
    _line_corrections = {}
    for lv in _AUGMENT_LINES:
        _lm = _cal_lines == lv
        if _lm.sum() < 10:
            continue
        _pred_mean = float(y_cal_raw[_lm].mean())
        _actual_mean = float(y_cal[_lm].mean())
        _line_corrections[lv] = _actual_mean - _pred_mean

    # Apply corrections to TEST set
    y_prob_raw = clf_iso.predict_proba(X_te_s)[:, 1]
    _te_lines = labeled.iloc[split_te:]["_total_line"].values
    y_prob = y_prob_raw.copy()
    for lv, corr in _line_corrections.items():
        _lm = _te_lines == lv
        y_prob[_lm] = np.clip(y_prob_raw[_lm] + corr, 0.01, 0.99)

    clf = PerLineCorrectedClassifier(clf_iso, _line_corrections)

    # ── 9. Evaluate ──────────────────────────────────────────────────────────
    y_pred = (y_prob >= 0.5).astype(int)
    auc   = float(roc_auc_score(y_te, y_prob))
    brier = float(brier_score_loss(y_te, y_prob))
    acc   = float(accuracy_score(y_te, y_pred))

    # Feature importances
    _all_imp = [
        cal.estimator.feature_importances_
        for cal in clf_iso.calibrated_classifiers_
        if hasattr(getattr(cal, "estimator", None), "feature_importances_")
    ]
    _mean_imp = np.mean(_all_imp, axis=0) if _all_imp else np.zeros(len(feat_cols))
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])[:20]

    # Park factor rank
    _imp_sorted = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])
    _park_rank = None
    _park_imp  = None
    for rank, (fname, fval) in enumerate(_imp_sorted, 1):
        if fname == "h_park_factor":
            _park_rank = rank
            _park_imp  = round(float(fval) * 100, 2)
            break
    # Also find park interaction ranks
    _park_interaction_ranks = {}
    for rank, (fname, fval) in enumerate(_imp_sorted, 1):
        if fname in _park_interaction_cols:
            _park_interaction_ranks[fname] = {
                "rank": rank, "importance_pct": round(float(fval) * 100, 2)
            }

    # Find batting feature ranks
    _batting_feature_ranks = {}
    _batting_total_imp = 0.0
    for rank, (fname, fval) in enumerate(_imp_sorted, 1):
        if fname in set(_V2_ROLLING_BATTING_COLS):
            _batting_feature_ranks[fname] = {
                "rank": rank, "importance_pct": round(float(fval) * 100, 2)
            }
            _batting_total_imp += fval

    # ── 10. Discrimination check ─────────────────────────────────────────────
    # r(predicted P_over, actual_total) — must beat V1's -0.057
    _te_actual_totals = labeled.iloc[split_te:]["_total_runs"].values
    # Use only one line value to avoid artificial correlation from augmentation
    _line_85_mask = _te_lines == 8.5
    if _line_85_mask.sum() > 30:
        from scipy.stats import pearsonr
        _disc_r, _disc_p = pearsonr(y_prob[_line_85_mask],
                                     _te_actual_totals[_line_85_mask])
        _disc_r = round(float(_disc_r), 4)
        _disc_p = round(float(_disc_p), 4)
    else:
        _disc_r, _disc_p = None, None

    # ── 11. Calibration buckets ──────────────────────────────────────────────
    buckets = []
    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55),
                   (0.55, 0.65), (0.65, 1.01)]:
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket":      f"{int(lo*100)}-{int(hi*100)}%",
            "n":           n,
            "pred_prob":   round(float(y_prob[mask].mean()) * 100, 1),
            "actual_rate": round(float(y_te[mask].mean()) * 100, 1),
            "gap":         round((float(y_te[mask].mean()) - float(y_prob[mask].mean())) * 100, 1),
        })

    # Per-line calibration
    line_cal = []
    for lv in _AUGMENT_LINES:
        _sm = _te_lines == lv
        _n  = int(_sm.sum())
        if _n == 0:
            continue
        line_cal.append({
            "line":        lv,
            "n":           _n,
            "pred_prob":   round(float(y_prob[_sm].mean()) * 100, 1),
            "actual_rate": round(float(y_te[_sm].mean()) * 100, 1),
            "gap":         round((float(y_te[_sm].mean()) - float(y_prob[_sm].mean())) * 100, 1),
        })

    # ── 12. Leak-free proof ──────────────────────────────────────────────────
    _KNOWN_LEAKY = {
        "h_ops", "a_ops", "h_slg", "a_slg",
        "h_k_pct", "a_k_pct", "h_bb_pct", "a_bb_pct",
        "h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip",
        "h_sp_k9", "a_sp_k9", "h_sp_wins", "a_sp_wins",
        "sp_era_diff", "sp_whip_diff", "sp_k9_diff", "sp_wins_diff",
        "h_sp_ip", "a_sp_ip",
    }
    _leaked = [c for c in feat_cols if c in _KNOWN_LEAKY]

    # ── 13. Save shadow artifacts ────────────────────────────────────────────
    saved = False
    if save and auc >= 0.50:
        import json as _json, datetime as _dt
        os.makedirs(_SHADOW_DIR, exist_ok=True)
        prefix = "mlb"
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_totals_v2_clf.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_totals_v2_scaler.pkl"), "wb") as f:
            pickle.dump(sc, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_totals_v2_imputer.pkl"), "wb") as f:
            pickle.dump(imp, f)
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_totals_v2_feat_cols.pkl"), "wb") as f:
            pickle.dump(feat_cols, f)
        meta = {
            "sport":              "MLB",
            "target":             "totals_over",
            "version":            "v2_shadow",
            "auc":                round(auc, 4),
            "brier":              round(brier, 4),
            "n_features":         len(feat_cols),
            "n_train":            int(len(X_tr_s)),
            "n_test":             int(len(X_te_s)),
            "augment_lines":      _AUGMENT_LINES,
            "park_interactions":   _park_interaction_cols,
            "line_corrections":   {str(k): round(v, 4) for k, v in _line_corrections.items()},
            "saved_at":           _dt.datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(_SHADOW_DIR, f"{prefix}_totals_v2_meta.json"), "w") as f:
            _json.dump(meta, f)
        saved = True
        log.info("train_mlb_totals_v2_shadow: saved (AUC=%.4f, Brier=%.4f, %d features)",
                 auc, brier, len(feat_cols))

    return {
        "version":               "v2_shadow_totals",
        "n_labeled":             len(labeled),
        "n_train":               len(X_tr),
        "n_calibration":         len(X_cal),
        "n_test":                len(X_te),
        "over_rate_all":         round(over_rate * 100, 1),
        "over_rate_train":       round(float(y_tr.mean()) * 100, 1),
        "over_rate_test":        round(float(y_te.mean()) * 100, 1),
        "roc_auc":               round(auc, 4),
        "brier_score":           round(brier, 4),
        "accuracy_pct":          round(acc * 100, 1),
        "n_features":            len(feat_cols),
        "feature_cols":          feat_cols,
        "park_interaction_cols": _park_interaction_cols,
        "augment_lines":         _AUGMENT_LINES,
        "line_corrections":      {str(k): round(v, 4) for k, v in _line_corrections.items()},
        "leak_free_proof": {
            "leaky_features_found": _leaked,
            "leak_free": len(_leaked) == 0,
        },
        "park_factor_rank":      _park_rank,
        "park_factor_importance_pct": _park_imp,
        "park_interaction_ranks": _park_interaction_ranks,
        "batting_feature_ranks": dict(sorted(
            _batting_feature_ranks.items(),
            key=lambda x: x[1]["rank"],
        )[:15]),
        "batting_total_importance_pct": round(_batting_total_imp * 100, 2),
        "discrimination": {
            "r_pred_vs_actual_total": _disc_r,
            "p_value": _disc_p,
            "line_used": 8.5,
            "n": int(_line_85_mask.sum()) if _disc_r is not None else 0,
        },
        "top30_importances": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "calibration_buckets":   buckets,
        "line_calibration":      line_cal,
        "saved":                 saved,
    }
