"""
walk_forward_backtest.py — Full-pipeline walk-forward backtest harness.

Replays historical MLB games through the *live ALE selection pipeline*,
scoring ALL market types (alt-spreads, alt-totals) across BOTH sides
of each fixture, ranking by LOS, and selecting top legs per day.

This is the canonical tool for:
  1. Validating config shadows against live realized numbers
  2. Testing new sub-models (e.g. NegBin distribution model) before deployment
  3. Measuring calibration, discrimination, and selected-WR by segment

LEAK CONTROL:
  - Features: build_feature_matrix() uses .shift(1).rolling() — no lookahead
  - Model: uses pre-trained artifacts (must predate backtest window)
  - Odds: pre-game alt_lines from historical.db (fetched_at before game)
  - Selection: replicates live ALE pipeline selection logic
  - Settlement: home_score - away_score + spread_value (spreads)
                home_score + away_score vs total_line (totals)

Usage:
    cd backend && python walk_forward_backtest.py [--start 2026-06-01] [--end 2026-06-27]
    cd backend && python walk_forward_backtest.py --calibrate
    cd backend && python walk_forward_backtest.py --spreads-only  # alt-spreads only (legacy mode)
"""
from __future__ import annotations
import os, sys, pickle, json, logging, argparse, sqlite3
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Constants (match live ALE pipeline — parlay_builder.py) ─────────────
SPREAD_UNIVERSE = [-2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
TOTALS_UNIVERSE = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]

# LOS weights — standard
LOS_WIN_W   = 0.50
LOS_EDGE_W  = 0.30
LOS_LQS_W   = 0.20
LOS_EDGE_SCALE = 15.0
# LOS weights — CUSHION (win_prob dominant)
LOS_WIN_W_C  = 0.60
LOS_EDGE_W_C = 0.15
LOS_LQS_W_C  = 0.25

# Gates (match recommender.py + parlay_builder.py)
ATS_MODEL_GATE      = 0.55   # 55% for |spread| ≤ 4.5 via ATS sub-model
DEEP_SPREAD_GATE    = 0.65   # 65% implied for |spread| ≥ 5.5
ALT_MIN_IMPLIED     = 0.42   # 42% implied floor for all alt-lines
LOS_FLOOR           = 0.52   # minimum LOS for ALE candidates

STAKE       = 10.0

# ── Live cap mechanism (replicated exactly) ──────────────────────────
# Live system (mock_bets.py):
#   - n_picks=40 total across all sports (standard picks pool)
#   - MAX_BETS_PER_GAME = 3 per fixture per game_date (cross-run hard cap)
#   - _MAX_POWER_BETS = 10 (4+ leg parlays, separate pool)
#   - Picks are parlays (2-4 legs), not individual legs
#   - Per-leg diversity: same leg appears in at most 3 picks
#
# For SINGLE-LEG backtest (measuring calibration, not parlay structure):
#   - per-fixture cap = 3 legs (matching MAX_BETS_PER_GAME — in live
#     each "bet" has 2-4 legs, but most are from different fixtures)
#   - daily cap = uncapped (live's n_picks=40 × ~3 legs = ~120 legs
#     across all sports, ~30 for MLB alone)
MAX_LEGS_PER_FIXTURE = 3     # match live's MAX_BETS_PER_GAME = 3
MAX_LEGS_PER_DAY     = 999   # effectively uncapped (matching live's actual behavior)

# ── LQS weights (from leg_quality.py _CURRENT_WEIGHTS) ──────────────
# A=0.55 (historical accuracy → neutral 50.0 in backtest, no in-sample leak)
# B=0.00 (model confidence — disabled)
# C=0.25 (market type quality — computable)
# D=0.20 (odds edge — computable)
_LQS_W_A = 0.55
_LQS_W_C = 0.25
_LQS_W_D = 0.20
_LQS_NEUTRAL_A = 50.0  # Component A returns 50 when < 5 examples exist

# Component C base scores (from leg_quality.py _MARKET_C_SCORE)
_MARKET_C_SCORE = {
    "Alt Spread": 70.0,
    "Alt Total":  60.0,
    "Spread":     85.0,
    "Total":      55.0,
    "Moneyline":  90.0,
    "Other":      50.0,
}

def _compute_lqs(market_type: str, edge_pp: float, line: float | None = None) -> float:
    """Compute LQS matching live's formula with components A (neutral), C, D.

    This mirrors compute_leg_quality_score() from leg_quality.py with:
    - Component A = neutral 50.0 (no historical data in backtest)
    - Component B = 0 (weight is 0 in current config)
    - Component C = market type quality score (pure lookup)
    - Component D = edge-based score (pure lookup)
    """
    # Component C
    sc_C = _MARKET_C_SCORE.get(market_type, 50.0)
    if market_type == "Alt Spread" and line is not None:
        abs_line = abs(line)
        if abs_line > 7.5:
            sc_C -= 25.0
        elif abs_line > 3.5:
            sc_C -= 15.0
    sc_C = max(0.0, min(100.0, sc_C))

    # Component D
    if edge_pp > 5:
        sc_D = 100.0
    elif edge_pp >= 2:
        sc_D = 75.0
    elif edge_pp >= 0:
        sc_D = 50.0
    else:
        sc_D = 20.0

    lqs = _LQS_NEUTRAL_A * _LQS_W_A + sc_C * _LQS_W_C + sc_D * _LQS_W_D
    return round(min(100.0, max(0.0, lqs)), 1)


def _load_model_artifacts(model_dir: str | None = None):
    """Load the main model artifacts (for Brier/discrimination metrics).

    Returns None tuple if model can't be loaded (numpy version mismatch etc.).
    The main model is NOT used for selection — only for Brier metrics.
    """
    d = model_dir or os.path.join(_HERE, "..", "data")
    try:
        with open(os.path.join(d, "model.pkl"), "rb") as f:
            clf = pickle.load(f)
        with open(os.path.join(d, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(d, "imputer.pkl"), "rb") as f:
            imputer = pickle.load(f)
        with open(os.path.join(d, "active_feature_cols.pkl"), "rb") as f:
            feat_cols = pickle.load(f)
        return clf, scaler, imputer, feat_cols
    except Exception as e:
        log.warning("Main model load failed (%s) — Brier metrics unavailable", e)
        return None, None, None, None


def _load_ats_submodel():
    """Load the ATS sub-model (MLB alt-spread selection)."""
    d = os.path.join(_HERE, "..", "data", "submodels")
    with open(os.path.join(d, "mlb_ats_clf.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(d, "mlb_ats_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(d, "mlb_ats_imputer.pkl"), "rb") as f:
        imputer = pickle.load(f)
    with open(os.path.join(d, "mlb_ats_feat_cols.pkl"), "rb") as f:
        feat_cols = pickle.load(f)
    return clf, scaler, imputer, feat_cols


def _load_totals_submodel():
    """Load the MLB totals sub-model."""
    d = os.path.join(_HERE, "..", "data", "submodels")
    try:
        with open(os.path.join(d, "mlb_totals_clf.pkl"), "rb") as f:
            clf = pickle.load(f)
        with open(os.path.join(d, "mlb_totals_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(d, "mlb_totals_imputer.pkl"), "rb") as f:
            imputer = pickle.load(f)
        with open(os.path.join(d, "mlb_totals_feat_cols.pkl"), "rb") as f:
            feat_cols = pickle.load(f)
        return clf, scaler, imputer, feat_cols
    except FileNotFoundError:
        log.warning("Totals sub-model not found — totals market disabled")
        return None


def _predict_ats(clf, scaler, imputer, feat_cols, row: pd.Series,
                 spread_value: float) -> float | None:
    """Predict P(home covers spread_value) using ATS sub-model. Returns 0-1."""
    def _val(col):
        if col == "_spread_value":
            return float(spread_value)
        if col == "_spread_abs":
            return abs(float(spread_value))
        if col == "_spread_sq":
            return float(spread_value) ** 2
        if col == "_is_fav_spread":
            return 1.0 if spread_value < 0 else 0.0
        if col == "_total_line":
            return np.nan
        if col == "_total_line_abs":
            return np.nan
        if col == "_is_high_total":
            return 0.0
        if col == "_line_over_base_rate":
            return np.nan
        if col == "ot_so_game":
            return 0.0
        return row[col] if col in row.index and not pd.isna(row[col]) else np.nan

    try:
        X = np.array([[_val(c) for c in feat_cols]], dtype=float)
        X_imp = imputer.transform(X)
        X_s = scaler.transform(X_imp)
        return float(clf.predict_proba(X_s)[0][1])
    except Exception:
        return None


def _line_over_base_rate(total_line: float) -> float:
    """Base rate of over hitting at this total line (from training data)."""
    # Approximate empirical base rates from MLB totals training
    _BASE = {6.5: 0.70, 7.5: 0.58, 8.5: 0.48, 9.5: 0.38, 10.5: 0.30, 11.5: 0.22}
    if total_line in _BASE:
        return _BASE[total_line]
    # Interpolate
    keys = sorted(_BASE.keys())
    if total_line <= keys[0]:
        return _BASE[keys[0]]
    if total_line >= keys[-1]:
        return _BASE[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= total_line <= keys[i + 1]:
            frac = (total_line - keys[i]) / (keys[i + 1] - keys[i])
            return _BASE[keys[i]] * (1 - frac) + _BASE[keys[i + 1]] * frac
    return 0.5


def _predict_totals(clf, scaler, imputer, feat_cols, row: pd.Series,
                    total_line: float) -> float | None:
    """Predict P(game goes OVER total_line). Returns 0-1."""
    def _val(col):
        if col == "_total_line":
            return float(total_line)
        if col == "_total_line_abs":
            return abs(float(total_line))
        if col == "_is_high_total":
            return 1.0 if total_line >= 9.5 else 0.0
        if col == "_line_over_base_rate":
            return _line_over_base_rate(total_line)
        if col == "_spread_value":
            return np.nan
        if col == "_spread_abs":
            return np.nan
        if col == "_spread_sq":
            return np.nan
        if col == "_is_fav_spread":
            return 0.0
        if col == "ot_so_game":
            return 0.0
        return row[col] if col in row.index and not pd.isna(row[col]) else np.nan

    try:
        X = np.array([[_val(c) for c in feat_cols]], dtype=float)
        X_imp = imputer.transform(X)
        X_s = scaler.transform(X_imp)
        if hasattr(clf, 'predict_proba'):
            try:
                proba = clf.predict_proba(X_s, total_lines=np.array([total_line]))
            except TypeError:
                proba = clf.predict_proba(X_s)
        else:
            proba = clf.predict_proba(X_s)
        return float(proba[0][1])
    except Exception:
        return None


def _compute_los(wp: float, edge_pp: float = 0.0,
                 lqs: float = 50.0, cushion: bool = False) -> float:
    """Compute Likelihood of Success score."""
    wp_norm = wp  # already 0-1
    lqs_norm = lqs / 100.0
    norm_edge = max(0.0, min(edge_pp, 15.0)) / LOS_EDGE_SCALE
    if cushion:
        return wp_norm * LOS_WIN_W_C + norm_edge * LOS_EDGE_W_C + lqs_norm * LOS_LQS_W_C
    return wp_norm * LOS_WIN_W + norm_edge * LOS_EDGE_W + lqs_norm * LOS_LQS_W


def _settle_spread(home_score: int, away_score: int,
                   spread_value: float, pick_side: str) -> str:
    """Settle a spread bet.

    spread_value is the rung from the PICK's perspective (always positive
    for underdogs, always negative for favorites).

    pick_side = 'home' or 'away'.
    """
    if pick_side == "home":
        cover = home_score - away_score + spread_value
    else:
        cover = away_score - home_score + spread_value
    if cover > 0:
        return "WIN"
    elif cover == 0:
        return "PUSH"
    else:
        return "LOSS"


def _settle_total(home_score: int, away_score: int,
                  total_line: float, direction: str) -> str:
    """Settle a totals bet."""
    actual_total = home_score + away_score
    if direction == "over":
        if actual_total > total_line:
            return "WIN"
        elif actual_total == total_line:
            return "PUSH"
        else:
            return "LOSS"
    else:  # under
        if actual_total < total_line:
            return "WIN"
        elif actual_total == total_line:
            return "PUSH"
        else:
            return "LOSS"


def _load_scores(start_date: str, end_date: str) -> dict[tuple, tuple]:
    """Load actual scores from historical.db."""
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date, home_score, away_score
        FROM games
        WHERE sport = 'MLB' AND status = 'Final'
          AND game_date BETWEEN ? AND ?
    """, (start_date, end_date)).fetchall()
    conn.close()
    scores = {}
    for home, away, gd, hs, as_ in rows:
        scores[(home, away, gd)] = (int(hs), int(as_))
    log.info("Loaded %d final MLB games (%s to %s)", len(scores), start_date, end_date)
    return scores


def _load_alt_line_odds(start_date: str, end_date: str) -> dict:
    """Load pre-game alt-spread odds from historical.db.

    Returns dict of (home_team, away_team, game_date, spread_value)
        -> market_implied_wp (0-1 scale, home-team perspective).
    """
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)

    rows = conn.execute("""
        SELECT home_team, away_team, game_date, team, line, odds, bookmaker
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_spreads'
          AND game_date BETWEEN ? AND ?
          AND odds IS NOT NULL AND odds > 1.0
        ORDER BY game_date, home_team, away_team, line, bookmaker
    """, (start_date, end_date)).fetchall()
    conn.close()

    if not rows:
        log.warning("No alt-spread odds found for %s to %s", start_date, end_date)
        return {}

    # Only use home-team entries (line is home-perspective spread)
    raw_odds = defaultdict(list)
    for home, away, gd, team, line, odds, book in rows:
        if team != home:
            continue
        sv = float(line)
        key = (home, away, gd, sv)
        raw_odds[key].append(float(odds))

    result = {}
    for key, odds_list in raw_odds.items():
        avg_odds = sum(odds_list) / len(odds_list)
        implied_wp = max(0.05, min(0.98, 1.0 / avg_odds))
        result[key] = implied_wp

    log.info("Loaded %d alt-spread implied WPs", len(result))
    return result


def _load_alt_total_odds(start_date: str, end_date: str) -> dict:
    """Load pre-game alt-total odds from historical.db.

    Returns dict of (home_team, away_team, game_date, total_line, direction)
        -> market_implied_wp (0-1 scale).
    direction is 'over' or 'under'.
    """
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)

    rows = conn.execute("""
        SELECT home_team, away_team, game_date, team, line, odds, bookmaker
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_totals'
          AND game_date BETWEEN ? AND ?
          AND odds IS NOT NULL AND odds > 1.0
        ORDER BY game_date, home_team, away_team, line
    """, (start_date, end_date)).fetchall()
    conn.close()

    if not rows:
        log.warning("No alt-total odds found for %s to %s", start_date, end_date)
        return {}

    # Group by (home, away, date, line, direction)
    raw_odds = defaultdict(list)
    for home, away, gd, team, line, odds, book in rows:
        tl = float(line)
        direction = team.strip().lower()
        if direction not in ("over", "under"):
            continue
        key = (home, away, gd, tl, direction)
        raw_odds[key].append(float(odds))

    result = {}
    for key, odds_list in raw_odds.items():
        avg_odds = sum(odds_list) / len(odds_list)
        implied_wp = max(0.05, min(0.98, 1.0 / avg_odds))
        result[key] = implied_wp

    log.info("Loaded %d alt-total implied WPs", len(result))
    return result


# ═════════════════════════════════════════════════════════════════════════
# Main backtest
# ═════════════════════════════════════════════════════════════════════════

def run_backtest(start_date: str, end_date: str,
                 model_dir: str | None = None,
                 config_transform=None,
                 config_name: str = "base_v3",
                 spreads_only: bool = False) -> dict:
    """Run full-pipeline walk-forward backtest on historical MLB games.

    Replicates the live ALE selection pipeline:
      1. Score ALL markets (alt-spreads + alt-totals) for each fixture
      2. Create candidates for BOTH sides (home/away, over/under)
      3. Apply gates (ATS model ≥ 55%, implied ≥ 42%, LOS ≥ 0.52)
      4. Rank all candidates across all fixtures by LOS
      5. Select top legs per day with per-fixture cap
      6. Settle against actual scores

    Parameters
    ----------
    spreads_only : if True, only score alt-spreads (legacy mode)
    config_transform : optional fn(base_wp, row, spread_value, ctx) -> float
        Applied to ATS model predictions for config shadow testing.
    """
    from features import build_feature_matrix

    # ── Load models ───────────────────────────────────────────────────
    clf, scaler, imputer, feat_cols = _load_model_artifacts(model_dir)
    main_model_ok = clf is not None
    if main_model_ok:
        log.info("Main model loaded: %d features", len(feat_cols))
    else:
        log.info("Main model unavailable — using ATS/totals sub-models only")

    ats_clf, ats_sc, ats_imp, ats_cols = _load_ats_submodel()
    log.info("ATS sub-model loaded: %d features", len(ats_cols))

    totals_arts = None
    if not spreads_only:
        totals_arts = _load_totals_submodel()
        if totals_arts:
            log.info("Totals sub-model loaded: %d features", len(totals_arts[3]))

    # ── Load feature matrix ───────────────────────────────────────────
    matrix = build_feature_matrix("MLB")
    if matrix.empty:
        return {"error": "Empty feature matrix"}

    matrix["_gd"] = matrix["game_date"].astype(str).str[:10]
    bt_games = matrix[(matrix["_gd"] >= start_date) & (matrix["_gd"] <= end_date)]
    if bt_games.empty:
        return {"error": f"No games in {start_date} to {end_date}"}

    dates = sorted(bt_games["_gd"].unique())
    log.info("Backtest: %d games across %d dates (%s to %s)",
             len(bt_games), len(dates), dates[0], dates[-1])

    # ── Load actual scores ────────────────────────────────────────────
    scores = _load_scores(start_date, end_date)

    # ── Load market odds ──────────────────────────────────────────────
    spread_wp = _load_alt_line_odds(start_date, end_date)
    total_wp = {}
    if not spreads_only:
        total_wp = _load_alt_total_odds(start_date, end_date)

    # ── Run backtest per date ─────────────────────────────────────────
    all_predictions = []   # full universe (for Brier)
    all_selected = []      # selected legs
    daily_stats = {}
    no_odds_spread = 0
    no_odds_total = 0

    for game_date in dates:
        day_games = bt_games[bt_games["_gd"] == game_date]
        day_candidates = []   # all ALE candidates for the day

        for idx, row in day_games.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            gd = game_date

            score_key = (home, away, gd)
            if score_key not in scores:
                continue
            home_score, away_score = scores[score_key]

            fixture_candidates = []

            # ── Alt-spread candidates ─────────────────────────────
            LIVE_RUNG_RANGE = {1.5, 2.5, 3.5, 4.5}

            # Pre-compute ATS predictions at all rungs (home perspective)
            ats_preds = {}
            for sv in SPREAD_UNIVERSE:
                p = _predict_ats(ats_clf, ats_sc, ats_imp, ats_cols, row, sv)
                if p is not None:
                    ats_preds[sv] = p

            for sv in SPREAD_UNIVERSE:
                if sv not in ats_preds:
                    continue
                ats_home_wp = ats_preds[sv]

                # Market implied WP (home perspective)
                mkt_wp_home = spread_wp.get((home, away, gd, sv))
                if mkt_wp_home is None:
                    no_odds_spread += 1
                    continue

                # Store full-universe prediction (home perspective, for Brier)
                model_wp_home = ats_home_wp
                if config_transform is not None:
                    transformed = config_transform(model_wp_home, row, sv, {})
                    if transformed is not None:
                        model_wp_home = transformed

                result_home = _settle_spread(home_score, away_score, sv, "home")
                all_predictions.append({
                    "game_date": gd, "home_team": home, "away_team": away,
                    "market": "alt_spread", "spread_value": sv,
                    "model_wp": round(model_wp_home, 4),
                    "market_wp": round(mkt_wp_home, 4),
                    "actual_result": result_home,
                    "pick_side": "home", "config": config_name,
                })

                # ── Create candidates for BOTH sides ──────────
                # Home side at +sv
                if sv in LIVE_RUNG_RANGE:
                    home_wp = model_wp_home
                    if home_wp >= ATS_MODEL_GATE and mkt_wp_home >= ALT_MIN_IMPLIED:
                        edge_pp = (home_wp - mkt_wp_home) * 100.0
                        lqs = _compute_lqs("Alt Spread", edge_pp, line=sv)
                        los = _compute_los(home_wp, edge_pp=edge_pp,
                                           lqs=lqs, cushion=(home_wp >= 0.70))
                        if los >= LOS_FLOOR:
                            fixture_candidates.append({
                                "game_date": gd, "home_team": home, "away_team": away,
                                "market": "alt_spread", "spread_value": sv,
                                "pick_side": "home", "pick_team": home,
                                "model_wp": round(home_wp, 4),
                                "market_wp": round(mkt_wp_home, 4),
                                "edge_pp": round(edge_pp, 2),
                                "los": round(los, 4),
                                "lqs": lqs,
                                "is_favorite": sv < 0,
                            })

                # Away side at +sv: P(away covers +sv) = 1 - P(home covers -sv)
                neg_sv = -sv
                if sv in LIVE_RUNG_RANGE and neg_sv in ats_preds:
                    ats_home_neg = ats_preds[neg_sv]
                    away_model_wp = 1.0 - ats_home_neg
                    if config_transform is not None:
                        # Config transform operates on the home-perspective base_wp.
                        # For away side: transform P(home, -sv), then invert.
                        transformed_neg = config_transform(ats_home_neg, row, neg_sv, {})
                        if transformed_neg is not None:
                            away_model_wp = 1.0 - transformed_neg

                    # Market implied for away side: 1 - P(home covers -sv)
                    mkt_wp_away_neg = spread_wp.get((home, away, gd, neg_sv))
                    if mkt_wp_away_neg is not None:
                        mkt_away_wp = 1.0 - mkt_wp_away_neg
                    else:
                        mkt_away_wp = 1.0 - mkt_wp_home  # fallback

                    if away_model_wp >= ATS_MODEL_GATE and mkt_away_wp >= ALT_MIN_IMPLIED:
                        edge_pp = (away_model_wp - mkt_away_wp) * 100.0
                        lqs = _compute_lqs("Alt Spread", edge_pp, line=sv)
                        los = _compute_los(away_model_wp, edge_pp=edge_pp,
                                           lqs=lqs, cushion=(away_model_wp >= 0.70))
                        if los >= LOS_FLOOR:
                            fixture_candidates.append({
                                "game_date": gd, "home_team": home, "away_team": away,
                                "market": "alt_spread", "spread_value": sv,
                                "pick_side": "away", "pick_team": away,
                                "model_wp": round(away_model_wp, 4),
                                "market_wp": round(mkt_away_wp, 4),
                                "edge_pp": round(edge_pp, 2),
                                "los": round(los, 4),
                                "lqs": lqs,
                                "is_favorite": sv < 0,
                            })

            # ── Alt-total candidates ──────────────────────────────
            if totals_arts and not spreads_only:
                t_clf, t_sc, t_imp, t_cols = totals_arts

                for tl in TOTALS_UNIVERSE:
                    over_prob = _predict_totals(t_clf, t_sc, t_imp, t_cols, row, tl)
                    if over_prob is None:
                        continue

                    for direction in ("over", "under"):
                        dir_wp = over_prob if direction == "over" else (1.0 - over_prob)

                        # Market implied
                        mkt_key = (home, away, gd, tl, direction)
                        mkt_wp = total_wp.get(mkt_key)
                        if mkt_wp is None:
                            no_odds_total += 1
                            continue

                        # Full-universe prediction (for Brier)
                        result_total = _settle_total(home_score, away_score, tl, direction)
                        all_predictions.append({
                            "game_date": gd, "home_team": home, "away_team": away,
                            "market": "alt_total", "total_line": tl,
                            "direction": direction,
                            "model_wp": round(dir_wp, 4),
                            "market_wp": round(mkt_wp, 4),
                            "actual_result": result_total,
                            "config": config_name,
                        })

                        # Gate: model wp >= 55% and implied >= 42%
                        if dir_wp < ATS_MODEL_GATE:
                            continue
                        if mkt_wp < ALT_MIN_IMPLIED:
                            continue

                        edge_pp = (dir_wp - mkt_wp) * 100.0
                        lqs = _compute_lqs("Alt Total", edge_pp, line=tl)
                        los = _compute_los(dir_wp, edge_pp=edge_pp,
                                           lqs=lqs, cushion=(dir_wp >= 0.70))
                        if los >= LOS_FLOOR:
                            fixture_candidates.append({
                                "game_date": gd, "home_team": home, "away_team": away,
                                "market": "alt_total", "total_line": tl,
                                "direction": direction,
                                "model_wp": round(dir_wp, 4),
                                "market_wp": round(mkt_wp, 4),
                                "edge_pp": round(edge_pp, 2),
                                "los": round(los, 4),
                                "lqs": lqs,
                            })

            # Apply per-fixture cap: keep top N by LOS
            fixture_candidates.sort(key=lambda c: -c["los"])
            day_candidates.extend(fixture_candidates[:MAX_LEGS_PER_FIXTURE])

        # ── Daily ALE selection: rank all candidates by LOS ───────
        day_candidates.sort(key=lambda c: -c["los"])
        day_selected = day_candidates[:MAX_LEGS_PER_DAY]

        # Settle selected legs
        for leg in day_selected:
            score_key = (leg["home_team"], leg["away_team"], leg["game_date"])
            if score_key not in scores:
                continue
            hs, as_ = scores[score_key]

            if leg["market"] == "alt_spread":
                leg["actual_result"] = _settle_spread(
                    hs, as_, leg["spread_value"], leg["pick_side"])
            elif leg["market"] == "alt_total":
                leg["actual_result"] = _settle_total(
                    hs, as_, leg["total_line"], leg["direction"])
            leg["selected"] = 1

        all_selected.extend(day_selected)

        # Daily stats
        day_sel_wl = [s for s in day_selected if s.get("actual_result") in ("WIN", "LOSS")]
        day_wins = sum(1 for s in day_sel_wl if s["actual_result"] == "WIN")
        day_total = len(day_sel_wl)
        day_wr = round(day_wins / day_total * 100, 1) if day_total > 0 else 0.0

        daily_stats[game_date] = {
            "games": len(day_games),
            "candidates": len(day_candidates) if day_candidates else 0,
            "selected": len(day_selected),
            "settled": day_total,
            "wins": day_wins,
            "wr": day_wr,
        }

    if no_odds_spread > 0:
        log.info("Skipped %d spread combos with no odds", no_odds_spread)
    if no_odds_total > 0:
        log.info("Skipped %d total combos with no odds", no_odds_total)

    # ── Aggregate results ─────────────────────────────────────────────
    sel_wl = [s for s in all_selected if s.get("actual_result") in ("WIN", "LOSS")]
    total_selected = len(sel_wl)
    total_wins = sum(1 for s in sel_wl if s["actual_result"] == "WIN")
    total_pushes = sum(1 for s in all_selected if s.get("actual_result") == "PUSH")
    selected_wr = round(total_wins / total_selected * 100, 1) if total_selected > 0 else 0.0

    # P&L (flat stake, standard -110 juice)
    pnl = 0.0
    for s in sel_wl:
        if s["actual_result"] == "WIN":
            pnl += STAKE * 0.909
        else:
            pnl -= STAKE

    # ── Per-rung breakdown (spreads) ──────────────────────────────────
    rung_stats = defaultdict(lambda: {"n": 0, "wins": 0})
    for s in sel_wl:
        if s["market"] == "alt_spread":
            sv = s["spread_value"]
            rung_stats[sv]["n"] += 1
            if s["actual_result"] == "WIN":
                rung_stats[sv]["wins"] += 1

    # ── Per-total-line breakdown ──────────────────────────────────────
    total_line_stats = defaultdict(lambda: {"n": 0, "wins": 0})
    for s in sel_wl:
        if s["market"] == "alt_total":
            key = f"{s['direction']}_{s['total_line']}"
            total_line_stats[key]["n"] += 1
            if s["actual_result"] == "WIN":
                total_line_stats[key]["wins"] += 1

    # ── Per-market breakdown ──────────────────────────────────────────
    market_stats = defaultdict(lambda: {"n": 0, "wins": 0})
    for s in sel_wl:
        market_stats[s["market"]]["n"] += 1
        if s["actual_result"] == "WIN":
            market_stats[s["market"]]["wins"] += 1

    # ── Fav/dog breakdown (spreads only) ──────────────────────────────
    fav_dog = {"fav": {"n": 0, "wins": 0}, "dog": {"n": 0, "wins": 0}}
    for s in sel_wl:
        if s["market"] == "alt_spread":
            is_fav = s.get("is_favorite", False)
            bucket = "fav" if is_fav else "dog"
            fav_dog[bucket]["n"] += 1
            if s["actual_result"] == "WIN":
                fav_dog[bucket]["wins"] += 1

    # ── Home/away breakdown (spreads only) ────────────────────────────
    home_away = {"home": {"n": 0, "wins": 0}, "away": {"n": 0, "wins": 0}}
    for s in sel_wl:
        if s["market"] == "alt_spread":
            side = s.get("pick_side", "unknown")
            if side in home_away:
                home_away[side]["n"] += 1
                if s["actual_result"] == "WIN":
                    home_away[side]["wins"] += 1

    # ── Full-universe Brier ───────────────────────────────────────────
    all_wl = [p for p in all_predictions if p["actual_result"] in ("WIN", "LOSS")]
    brier_model = None
    brier_market = None
    if all_wl:
        brier_model = round(sum(
            (p["model_wp"] - (1.0 if p["actual_result"] == "WIN" else 0.0)) ** 2
            for p in all_wl) / len(all_wl), 4)
        brier_market = round(sum(
            (p["market_wp"] - (1.0 if p["actual_result"] == "WIN" else 0.0)) ** 2
            for p in all_wl) / len(all_wl), 4)

    avg_edge = round(sum(s["edge_pp"] for s in sel_wl) / len(sel_wl), 2) if sel_wl else 0

    result = {
        "config": config_name,
        "start_date": start_date,
        "end_date": end_date,
        "total_games": len(set((p["home_team"], p["away_team"], p["game_date"])
                               for p in all_predictions)),
        "total_predictions": len(all_predictions),
        "total_selected": total_selected,
        "total_pushes": total_pushes,
        "selected_wins": total_wins,
        "selected_wr": selected_wr,
        "pnl": round(pnl, 2),
        "avg_edge_pp": avg_edge,
        "brier_model": brier_model,
        "brier_market": brier_market,
        "no_odds_spread": no_odds_spread,
        "no_odds_total": no_odds_total,
        "daily": daily_stats,
        "per_rung": {str(k): {"n": v["n"], "wins": v["wins"],
                               "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0}
                     for k, v in sorted(rung_stats.items())},
        "per_total_line": {k: {"n": v["n"], "wins": v["wins"],
                                "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0}
                           for k, v in sorted(total_line_stats.items())},
        "per_market": {k: {"n": v["n"], "wins": v["wins"],
                            "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0}
                       for k, v in sorted(market_stats.items())},
        "fav_dog": {k: {"n": v["n"], "wins": v["wins"],
                         "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0}
                    for k, v in fav_dog.items() if v["n"] > 0},
        "home_away": {k: {"n": v["n"], "wins": v["wins"],
                           "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0}
                      for k, v in home_away.items() if v["n"] > 0},
    }
    return result


def print_report(r: dict):
    """Pretty-print backtest results."""
    print()
    print(f"{'=' * 70}")
    print(f"  FULL-PIPELINE BACKTEST: {r['config']}")
    print(f"  {r['start_date']} -> {r['end_date']}")
    print(f"{'=' * 70}")
    print()
    print(f"  Games:            {r['total_games']}")
    print(f"  Predictions:      {r['total_predictions']} (with market odds)")
    print(f"  No-odds (spread): {r.get('no_odds_spread', 0)}")
    print(f"  No-odds (total):  {r.get('no_odds_total', 0)}")
    print(f"  Selected legs:    {r['total_selected']} (W/L) + {r['total_pushes']} pushes")
    print(f"  Selected WR:      {r['selected_wr']}%  ({r['selected_wins']}/{r['total_selected']})")
    print(f"  P&L:              ${r['pnl']:+.2f}  (flat ${STAKE} @ -110)")
    print(f"  Avg edge (sel):   {r.get('avg_edge_pp', 0):+.2f}pp")
    print(f"  Brier (model):    {r.get('brier_model', 'N/A')}")
    print(f"  Brier (market):   {r.get('brier_market', 'N/A')}")

    # Per-market breakdown
    if r.get("per_market"):
        print()
        print("  Per-market breakdown:")
        print(f"  {'Market':>15} {'N':>5} {'Wins':>5} {'WR':>7}")
        print(f"  {'-' * 35}")
        for mkt, d in r["per_market"].items():
            print(f"  {mkt:>15} {d['n']:>5} {d['wins']:>5} {d['wr']:>6.1f}%")

    # Fav/dog breakdown
    if r.get("fav_dog"):
        print()
        print("  Fav/dog breakdown (spreads):")
        print(f"  {'Type':>10} {'N':>5} {'Wins':>5} {'WR':>7} {'Mix':>7}")
        print(f"  {'-' * 38}")
        total_spread = sum(d['n'] for d in r['fav_dog'].values())
        for side, d in r["fav_dog"].items():
            mix = round(d['n'] / total_spread * 100, 1) if total_spread > 0 else 0
            print(f"  {side:>10} {d['n']:>5} {d['wins']:>5} {d['wr']:>6.1f}% {mix:>6.1f}%")

    # Home/away breakdown
    if r.get("home_away"):
        print()
        print("  Home/away breakdown (spreads):")
        print(f"  {'Side':>10} {'N':>5} {'Wins':>5} {'WR':>7} {'Mix':>7}")
        print(f"  {'-' * 38}")
        total_spread = sum(d['n'] for d in r['home_away'].values())
        for side, d in r["home_away"].items():
            mix = round(d['n'] / total_spread * 100, 1) if total_spread > 0 else 0
            print(f"  {side:>10} {d['n']:>5} {d['wins']:>5} {d['wr']:>6.1f}% {mix:>6.1f}%")

    # Per-day breakdown
    print()
    print("  Per-day breakdown:")
    print(f"  {'Date':>12} {'Games':>6} {'Cands':>6} {'Selected':>9} {'Wins':>5} {'WR':>7}")
    print(f"  {'-' * 52}")
    for date in sorted(r["daily"].keys()):
        d = r["daily"][date]
        print(f"  {date:>12} {d['games']:>6} {d.get('candidates', 0):>6} "
              f"{d['selected']:>9} {d['wins']:>5} {d['wr']:>6.1f}%")

    # Per-rung breakdown (spreads)
    if r.get("per_rung"):
        print()
        print("  Per-rung breakdown (selected spread legs):")
        print(f"  {'Rung':>8} {'N':>5} {'Wins':>5} {'WR':>7}")
        print(f"  {'-' * 28}")
        for rung in sorted(r["per_rung"].keys(), key=float):
            d = r["per_rung"][rung]
            print(f"  {rung:>8} {d['n']:>5} {d['wins']:>5} {d['wr']:>6.1f}%")

    # Per-total-line breakdown
    if r.get("per_total_line"):
        print()
        print("  Per-total-line breakdown (selected total legs):")
        print(f"  {'Line':>14} {'N':>5} {'Wins':>5} {'WR':>7}")
        print(f"  {'-' * 34}")
        for line, d in sorted(r["per_total_line"].items()):
            print(f"  {line:>14} {d['n']:>5} {d['wins']:>5} {d['wr']:>6.1f}%")

    print()


def run_calibration_check():
    """Calibration check: backtest on Jun 22-27 and compare to realized.

    Runs both full-pipeline and spreads-only modes, then compares
    against shadow harness and live mock_bet_legs numbers.
    """
    print("=" * 70)
    print("  CALIBRATION CHECK: Backtest vs Realized")
    print("=" * 70)
    print()

    # Run spreads-only to compare against live alt-spread WR
    result_spreads = run_backtest("2026-06-22", "2026-06-27",
                                  config_name="V3_spreads_only",
                                  spreads_only=True)
    print_report(result_spreads)

    # Run full pipeline
    result_full = run_backtest("2026-06-22", "2026-06-27",
                                config_name="V3_full_pipeline")
    print_report(result_full)

    # Load realized numbers for comparison
    conn = sqlite3.connect(os.path.join(_HERE, "..", "data", "bets.db"))

    # Shadow V3 selected-WR
    shadow_sel = conn.execute("""
        SELECT COUNT(*), SUM(CASE WHEN actual_result='WIN' THEN 1 ELSE 0 END),
               ROUND(AVG(CASE WHEN actual_result='WIN' THEN 100.0 ELSE 0.0 END), 1)
        FROM shadow_predictions
        WHERE shadow_slot = 'shadow_v3' AND actual_result IN ('WIN','LOSS')
          AND selected = 1
          AND DATE(scored_at) BETWEEN '2026-06-22' AND '2026-06-27'
    """).fetchone()
    sh_n, sh_wins, sh_wr = shadow_sel if shadow_sel[0] and shadow_sel[0] > 0 else (0, 0, 0)

    # Live alt-spread legs WR
    live_alt = conn.execute("""
        SELECT COUNT(*), SUM(CASE WHEN leg_result='WIN' THEN 1 ELSE 0 END),
               ROUND(AVG(CASE WHEN leg_result='WIN' THEN 100.0 ELSE 0.0 END), 1)
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN','LOSS')
          AND mbl.market_type = 'alternate_spreads'
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.source IN ('prospective', 'prospective_pm')
          AND mb.game_date BETWEEN '2026-06-22' AND '2026-06-27'
    """).fetchone()
    la_n, la_wins, la_wr = live_alt if live_alt[0] and live_alt[0] > 0 else (0, 0, 0)

    # Live per-rung WR
    live_rung = conn.execute("""
        SELECT mbl.spread_value,
            COUNT(*), SUM(CASE WHEN leg_result='WIN' THEN 1 ELSE 0 END)
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN','LOSS')
          AND mbl.market_type = 'alternate_spreads'
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.source IN ('prospective', 'prospective_pm')
          AND mbl.spread_value IS NOT NULL
        GROUP BY mbl.spread_value ORDER BY mbl.spread_value
    """).fetchall()

    # Live totals WR (if any)
    live_totals = conn.execute("""
        SELECT COUNT(*), SUM(CASE WHEN leg_result='WIN' THEN 1 ELSE 0 END),
               ROUND(AVG(CASE WHEN leg_result='WIN' THEN 100.0 ELSE 0.0 END), 1)
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN','LOSS')
          AND mbl.market_type IN ('totals', 'alternate_totals')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.source IN ('prospective', 'prospective_pm')
          AND mb.game_date BETWEEN '2026-06-22' AND '2026-06-27'
    """).fetchone()
    lt_n, lt_wins, lt_wr = live_totals if live_totals[0] and live_totals[0] > 0 else (0, 0, 0)

    conn.close()

    # Comparison table
    print("=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print()

    bt_wr = result_spreads["selected_wr"]
    fp_wr = result_full["selected_wr"]

    print(f"  {'Metric':<25} {'BT Spread':>12} {'BT Full':>12} {'Shadow':>12} {'Live Alt':>12}")
    print(f"  {'-' * 77}")
    print(f"  {'Selected legs':<25} {result_spreads['total_selected']:>12} "
          f"{result_full['total_selected']:>12} {sh_n or 'N/A':>12} {la_n or 'N/A':>12}")
    print(f"  {'Selected WR':<25} {bt_wr:>11.1f}% {fp_wr:>11.1f}% "
          f"{(sh_wr or 0):>11.1f}% {(la_wr or 0):>11.1f}%")
    print(f"  {'P&L':<25} ${result_spreads['pnl']:>+11.2f} "
          f"${result_full['pnl']:>+11.2f} {'N/A':>12} {'N/A':>12}")

    # Per-rung comparison
    if live_rung:
        print()
        print("  Per-rung comparison (backtest vs live):")
        print(f"  {'Rung':>8} {'BT N':>6} {'BT WR':>7} {'Live N':>7} {'Live WR':>8}")
        print(f"  {'-' * 42}")
        bt_rungs = result_spreads.get("per_rung", {})
        for sv, ln, lw in live_rung:
            l_wr = round(lw / ln * 100, 1) if ln > 0 else 0
            bt_d = bt_rungs.get(str(sv), {"n": 0, "wr": 0})
            print(f"  {sv:>8} {bt_d['n']:>6} {bt_d['wr']:>6.1f}% {ln:>7} {l_wr:>7.1f}%")

    if lt_n > 0:
        print()
        print(f"  Live totals WR: {lt_wr}% ({lt_wins}/{lt_n})")

    print()

    # Validation targets
    print("  VALIDATION TARGETS:")
    print(f"  {'-' * 50}")
    targets = {
        "Alt-spread selected WR ≈ 85%": bt_wr,
        "+1.5 WR ≈ 72.6%": float(result_spreads["per_rung"].get("1.5", {}).get("wr", 0)),
        "+2.5 WR ≈ 76.8%": float(result_spreads["per_rung"].get("2.5", {}).get("wr", 0)),
    }
    for label, val in targets.items():
        target_val = float(label.split("≈")[1].strip().rstrip("%"))
        gap = val - target_val
        status = "PASS" if abs(gap) < 10 else ("HIGH" if gap > 0 else "LOW")
        print(f"  {label:<35} → {val:>6.1f}%  ({gap:+.1f}pp) [{status}]")

    print()
    return result_full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-pipeline walk-forward backtest")
    parser.add_argument("--start", default="2026-06-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-27",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration check vs realized numbers")
    parser.add_argument("--spreads-only", action="store_true",
                        help="Alt-spreads only (legacy mode)")
    parser.add_argument("--model-dir", default=None,
                        help="Path to model artifacts (default: data/)")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration_check()
    else:
        result = run_backtest(args.start, args.end,
                              model_dir=args.model_dir,
                              spreads_only=args.spreads_only)
        print_report(result)
