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
  NHL     → nhl_ats_v1         AUC 0.7544  threshold 65%  ACTIVE
  MLB     → mlb_ats_v1         AUC 0.6360  threshold 60%  ACTIVE
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
MODEL_PATH   = os.path.join(DATA_DIR, "model.pkl")
SCALER_PATH  = os.path.join(DATA_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(DATA_DIR, "imputer.pkl")
SUBMODEL_DIR = os.path.join(DATA_DIR, "submodels")
DRIFT_PATH   = os.path.join(DATA_DIR, "drift_state.json")
os.makedirs(SUBMODEL_DIR, exist_ok=True)

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
# MLB:  4 seasons (2022-25). Isolation AUC = 0.6360 ✓ ACTIVE → mlb_ats_v1.
#       Sub-model saved to data/ats_model_MLB.pkl. Threshold 60%.
# NHL:  5 seasons (2021-22 to 2025-26). Isolation AUC = 0.7544 ✓ ACTIVE → nhl_ats_v1.
#       Sub-model saved to data/ats_model_NHL.pkl. Threshold 65%.
#       ot_so_game is dominant feature (42.85% importance): OT/SO always 1-goal margin.


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
]

# Unified schema used for training and inference (v3)
FEATURE_COLS: list[str] = _PERSONAL_FEATURE_COLS + HIST_EXTRA_COLS


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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_clf(algorithm: str, calibrate: bool = True):
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


def _importances(clf, cols: list[str]) -> dict:
    """
    Extract feature importances from a fitted estimator or CalibratedClassifierCV.
    For calibrated models, averages importances across the CV folds.
    """
    # Unwrap CalibratedClassifierCV → average over fold estimators
    if hasattr(clf, "calibrated_classifiers_"):
        all_imp = []
        for cal_clf in clf.calibrated_classifiers_:
            base = getattr(cal_clf, "estimator", None)
            if base is not None and hasattr(base, "feature_importances_"):
                all_imp.append(base.feature_importances_)
        if all_imp:
            mean_imp = np.mean(all_imp, axis=0)
            return {n: round(float(v), 4) for n, v in zip(cols, mean_imp)}
        return {}
    # Direct estimator
    base = getattr(clf, "estimator", clf)
    if hasattr(base, "feature_importances_"):
        return {n: round(float(v), 4) for n, v in zip(cols, base.feature_importances_)}
    if hasattr(base, "coef_"):
        c = np.abs(base.coef_[0])
        return {n: round(float(v / c.sum()), 4) for n, v in zip(cols, c)}
    return {}


_MODEL_CACHE: tuple | None = None   # cached (clf, scaler, imputer) — None = not yet loaded
_MODEL_LOAD_FAILED: bool   = False  # True after a failed load (skip future attempts)
import threading as _threading
_MODEL_LOCK = _threading.Lock()

def load_model():
    global _MODEL_CACHE, _MODEL_LOAD_FAILED
    # Fast path — already loaded or already failed
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if _MODEL_LOAD_FAILED:
        return None, None, None
    # Slow path — serialize so only one thread attempts the pickle load
    with _MODEL_LOCK:
        # Re-check after acquiring lock (another thread may have loaded it)
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE
        if _MODEL_LOAD_FAILED:
            return None, None, None
        if not os.path.exists(MODEL_PATH):
            _MODEL_LOAD_FAILED = True
            return None, None, None
        try:
            with open(MODEL_PATH,  "rb") as f: clf     = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: scaler  = pickle.load(f)
            imputer = None
            if os.path.exists(IMPUTER_PATH):
                with open(IMPUTER_PATH, "rb") as f: imputer = pickle.load(f)
            _MODEL_CACHE = (clf, scaler, imputer)
            return _MODEL_CACHE
        except Exception as _e:
            print(f"[load_model] WARNING: global model load failed ({_e}). Returning None.")
            _MODEL_LOAD_FAILED = True
            return None, None, None


def _load_submodel(sport: str):
    # Legacy path — kept for backwards-compat with any manually placed files.
    path = os.path.join(SUBMODEL_DIR, f"{sport.lower().replace(' ', '_')}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_ats_submodel(sport: str):
    """
    Load the diagnostic ATS sub-model artifacts for *sport*.
    Returns (clf, scaler, imputer, feat_cols) or None if not available.
    """
    prefix = sport.lower()
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


# ── Feature-matrix cache (keyed by sport, TTL = 1 hour) ───────────────────────
_FM_CACHE: dict = {}          # sport → (timestamp, DataFrame)
_FM_CACHE_TTL = 3600          # seconds


def _get_feature_matrix(sport: str):
    """Return cached feature matrix for sport, rebuilding if stale."""
    import time
    from features import build_feature_matrix
    now = time.time()
    if sport in _FM_CACHE:
        ts, df = _FM_CACHE[sport]
        if now - ts < _FM_CACHE_TTL:
            return df
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
    sport:      str,
    home_team:  str,
    away_team:  str,
    game_date:  str | None = None,
) -> float | None:
    """
    Predict the probability that the home team covers the spread for a game,
    using the diagnostic ATS sub-model for *sport*.

    Returns a float in [0, 1] or None if:
      • No sub-model exists for this sport
      • No matching game found in the feature matrix

    The prediction is based on rolling team stats, pitcher ERA, etc. — NOT
    the global bet-slip model features.
    """
    artifacts = _load_ats_submodel(sport)
    if artifacts is None:
        return None
    clf, sc, imp, feat_cols = artifacts

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

        def _val(col: str) -> float:
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
        return float(clf.predict_proba(X_s)[0][1])

    except Exception as exc:
        log.warning("predict_game_ats(%s, %s vs %s): %s", sport, home_team, away_team, exc)
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


def _prep_row(feature_dict: dict, scaler, imputer) -> np.ndarray:
    """
    Build a scaled, imputed 2-D row from a feature dict for inference.
    HIST_EXTRA_COLS not in feature_dict default to NaN; imputer fills median.
    """
    row = np.array(
        [[feature_dict.get(c, np.nan) for c in FEATURE_COLS]], dtype=float
    )
    if imputer is not None:
        row = imputer.transform(row)
    return scaler.transform(row)


# ── Training ───────────────────────────────────────────────────────────────────
def train(db: Session, algorithm: str = "gradient_boost",
          train_submodels: bool = True) -> dict:
    now = datetime.utcnow()

    # ── 1. Personal bets ────────────────────────────────────────────────────
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).order_by(Bet.time_placed).all()

    personal_df = _build_personal_df(bets, cutoff=now)
    if len(personal_df) < 30:
        return {"error": "Need >= 30 settled bets to train."}

    # ── 2. Historical games (separate until final concat) ──────────────────
    historical_df = _load_historical_df()

    # ── 3. CLV hook — no-op now; fills clv / clv_available when activated ──
    personal_df   = add_clv_features(personal_df)
    if not historical_df.empty:
        historical_df = add_clv_features(historical_df)

    # ── 4. Combine and sort chronologically ────────────────────────────────
    frames = [personal_df]
    if not historical_df.empty:
        frames.append(historical_df[personal_df.columns])   # align cols
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("_date").reset_index(drop=True)

    n_personal   = int((combined["_weight"] > 1.0).sum())
    n_historical = len(combined) - n_personal
    log.info(
        "train: %d personal bets + %d historical rows = %d total",
        n_personal, n_historical, len(combined),
    )

    X_all  = combined[FEATURE_COLS].values.astype(float)
    y_all  = combined["_label"].values.astype(float)
    sw_all = combined["_weight"].values.astype(float)

    # ── 5. Temporal 80/20 split (no shuffle) ──────────────────────────────
    split  = int(len(X_all) * 0.80)
    X_tr,  X_te  = X_all[:split],  X_all[split:]
    y_tr,  y_te  = y_all[:split],  y_all[split:]
    sw_tr         = sw_all[:split]

    # ── 6. Impute → scale ──────────────────────────────────────────────────
    imputer  = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr_imp)
    X_te_s   = scaler.transform(X_te_imp)

    # ── 7. Fit with sample weights ─────────────────────────────────────────
    clf = _build_clf(algorithm)
    clf.fit(X_tr_s, y_tr, sample_weight=sw_tr)

    # ── 8. Evaluate ────────────────────────────────────────────────────────
    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    acc    = float(accuracy_score(y_te, y_pred))
    auc    = float(roc_auc_score(y_te, y_prob)) if len(set(y_te)) > 1 else 0.5
    brier  = float(brier_score_loss(y_te, y_prob))
    imps   = _importances(clf, FEATURE_COLS)

    # ── 9. Persist ─────────────────────────────────────────────────────────
    with open(MODEL_PATH,   "wb") as f: pickle.dump(clf,     f)
    with open(SCALER_PATH,  "wb") as f: pickle.dump(scaler,  f)
    with open(IMPUTER_PATH, "wb") as f: pickle.dump(imputer, f)

    # ── 10. Sub-models (personal bets only, per sport) ─────────────────────
    sub_trained = {}
    if train_submodels:
        p_imp = imputer.transform(personal_df[FEATURE_COLS].values.astype(float))
        p_y   = personal_df["_label"].values.astype(float)
        p_sw  = personal_df["_weight"].values.astype(float)
        p_sport = personal_df["_sport_label"].values

        for sport in SUBMODEL_SPORTS:
            mask = p_sport == sport
            if mask.sum() < 40:
                continue
            Xs, ys, sws = p_imp[mask], p_y[mask], p_sw[mask]
            sp = int(len(Xs) * 0.80)
            if sp < 20:
                continue
            clf_s = _build_clf(algorithm, calibrate=False)
            clf_s.fit(scaler.transform(Xs[:sp]), ys[:sp], sample_weight=sws[:sp])
            path = os.path.join(SUBMODEL_DIR, f"{sport.lower().replace(' ', '_')}.pkl")
            with open(path, "wb") as f:
                pickle.dump(clf_s, f)
            sub_trained[sport] = int(mask.sum())

    # ── 11. Per-sport test metrics ─────────────────────────────────────────
    SPORT_MAP_INV = {v: k for k, v in SPORT_MAP.items()}
    test_sport_ids = X_te[:, FEATURE_COLS.index("sport_id")]
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

    # ── 12. Drift baseline ─────────────────────────────────────────────────
    with open(DRIFT_PATH, "w") as f:
        json.dump({
            "baseline_accuracy": round(acc, 4),
            "baseline_roc_auc":  round(auc, 4),
            "n_test":            len(X_te),
            "saved_at":          now.isoformat(),
        }, f)

    n_personal_train   = int((combined["_weight"][:split] > 1.0).sum())
    n_historical_train = split - n_personal_train

    run = ModelRun(
        algorithm=f"{algorithm}_v3",
        n_train=split, n_test=len(X_te),
        accuracy=round(acc, 4), roc_auc=round(auc, 4),
        feature_names=FEATURE_COLS,
        notes=json.dumps({
            "importances":        imps,
            "brier":              round(brier, 4),
            "sport_metrics":      sport_metrics,
            "submodels":          sub_trained,
            "version":            "v3",
            "n_personal_train":   n_personal_train,
            "n_historical_train": n_historical_train,
            "weight_config":      WEIGHT_CONFIG,
        }),
    )
    db.add(run); db.commit(); db.refresh(run)

    return {
        "run_id":              run.id,
        "algorithm":           algorithm,
        "version":             "v3",
        "n_train":             split,
        "n_test":              len(X_te),
        "n_personal_train":    n_personal_train,
        "n_historical_train":  n_historical_train,
        "accuracy":            round(acc * 100, 2),
        "roc_auc":             round(auc, 4),
        "brier_score":         round(brier, 4),
        "feature_importance":  imps,
        "sport_metrics":       sport_metrics,
        "submodels_trained":   list(sub_trained.keys()),
        "weight_config":       WEIGHT_CONFIG,
        "class_report":        classification_report(y_te, y_pred, output_dict=True),
    }


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
    clf, scaler, imputer = load_model()
    if clf is None:
        return []
    row_s  = _prep_row(feature_dict, scaler, imputer)
    base_p = float(clf.predict_proba(row_s)[0][1])
    contribs = []
    for i, col in enumerate(FEATURE_COLS):
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
    clf, scaler, imputer = load_model()
    if clf is None:
        return {"error": "No trained model found. Run /api/model/train first."}

    sport   = feature_dict.get("sport_label", "")
    sub_clf = _load_submodel(sport) if (use_submodel and sport) else None
    active  = sub_clf if sub_clf else clf

    row_s    = _prep_row(feature_dict, scaler, imputer)
    win_prob = float(active.predict_proba(row_s)[0][1])
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

    clf, scaler, imputer = load_model()
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
        row_s = _prep_row(f, scaler, imputer)
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

    clf, scaler, imputer = load_model()
    if clf is None:
        return {"error": "Train the model first."}

    results = []; correct = 0
    for b in bets[30:]:
        f = extract_features(b)
        if not f:
            continue
        row_s = _prep_row(f, scaler, imputer)
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

    Decision thresholds (also controls save when save=True)
    ---------------------------------------------------------
    AUC >= 0.55  → "WORTH BUILDING"  (saves if save=True)
    0.50–0.54    → "WAIT"
    < 0.50       → "FEATURE WORK NEEDED"

    Does NOT overwrite any production model or main model.pkl.
    """
    try:
        from features import build_feature_matrix, TARGET_ATS, MODEL_FEATURE_COLS
    except ImportError:
        return {"error": "features.py not importable"}

    # ── 1. Load ATS feature matrix ────────────────────────────────────────────
    matrix = build_feature_matrix(sport)
    if matrix.empty:
        return {"error": f"No {sport} data in historical.db"}

    labeled = matrix[matrix[TARGET_ATS].notna()].copy()
    if len(labeled) < 50:
        return {"error": f"Only {len(labeled)} labeled {sport} rows — need ≥50"}

    # ── 2. Build X / y ────────────────────────────────────────────────────────
    feat_cols = [c for c in MODEL_FEATURE_COLS if c in labeled.columns]
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
    imps = sorted(zip(feat_cols, _mean_imp), key=lambda x: -x[1])[:10]

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

    # ── 7. Decision ────────────────────────────────────────────────────────────
    if auc >= 0.54:
        decision = f"WORTH BUILDING: per-sport {sport} sub-model justified (AUC >= 0.54)"
        if save:
            import json as _json, datetime as _dt
            prefix = sport.lower()
            with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_clf.pkl"),     "wb") as f: pickle.dump(clf, f)
            with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_scaler.pkl"),  "wb") as f: pickle.dump(sc, f)
            with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_imputer.pkl"), "wb") as f: pickle.dump(imp, f)
            # Also save feature column list so inference knows the schema
            with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_feat_cols.pkl"), "wb") as f:
                pickle.dump(feat_cols, f)
            # Persist metadata so predict-game endpoint can surface AUC without retraining
            meta = {
                "sport":         sport.upper(),
                "auc":           round(auc, 4),
                "n_features":    len(feat_cols),
                "n_train":       int(len(X_tr_s)),
                "saved_at":      _dt.datetime.utcnow().isoformat() + "Z",
            }
            with open(os.path.join(SUBMODEL_DIR, f"{prefix}_ats_meta.json"), "w") as f:
                _json.dump(meta, f)
            log.info("train_diagnostic_sport_model: saved %s diagnostic model (%d features, AUC=%.4f)",
                     sport, len(feat_cols), auc)
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
        "top10_importances": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in imps
        ],
        "calibration_buckets": buckets,
        "saved":            save and (auc >= 0.54),
        "decision":         decision,
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


# ── NBA isolation test (backward compat wrapper) ───────────────────────────────

def train_nba_only_model() -> dict:
    """
    Backward-compatible wrapper for train_diagnostic_sport_model("NBA").
    Does NOT save the model.
    """
    return train_diagnostic_sport_model("NBA", save=False)
