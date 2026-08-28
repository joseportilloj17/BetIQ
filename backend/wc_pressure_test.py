"""
wc_pressure_test.py — World Cup 2026 Model Pressure Test Framework

Three analyses:
  1. Historical WC backtest (2014 / 2018 / 2022) — train on earlier data, predict later
  2. Market calibration — compare model vs bookmaker implied probs for WC 2026
  3. Feature correlation — rank features by predictive power, check multicollinearity

Run:
    cd backend && python wc_pressure_test.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Name normalization (must match wc_features.py) ──────────────────────────

_NAME_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States",
    "Republic of Ireland": "Ireland",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "Chinese Taipei": "Taiwan",
}


def _norm(name: str) -> str:
    return _NAME_ALIASES.get(name, name)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Historical WC Backtest
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_pi_ratings(results_df: pd.DataFrame, before_date: str,
                        lookback_years: int = 4) -> dict:
    """
    Compute Pi ratings from international results up to `before_date`,
    using matches from the last `lookback_years` years.
    Uses opponent-weighted K (same algorithm as wc_features.py v2).
    """
    cutoff = pd.Timestamp(before_date)
    start = cutoff - pd.DateOffset(years=lookback_years)

    df = results_df[
        (results_df["date"] >= start) & (results_df["date"] < cutoff)
    ].copy()

    ratings = {}  # team -> {"home_pi": float, "away_pi": float, "games": int}

    K_BASE = 0.1

    for _, row in df.iterrows():
        home = _norm(str(row["home_team"]))
        away = _norm(str(row["away_team"]))
        hs, as_ = float(row["home_score"]), float(row["away_score"])
        tourn = str(row.get("tournament", ""))

        margin = hs - as_

        # Importance weighting
        is_competitive = any(kw in tourn.lower() for kw in
                            ["world cup", "euro", "copa america", "nations league",
                             "african cup", "asian cup", "gold cup", "confederations",
                             "qualification"])
        importance = 1.0 if is_competitive else 0.5

        for team in [home, away]:
            if team not in ratings:
                ratings[team] = {"home_pi": 0.0, "away_pi": 0.0, "games": 0}

        # Opponent strength adjustment
        home_combined = ratings[home]["home_pi"] + ratings[home]["away_pi"]
        away_combined = ratings[away]["home_pi"] + ratings[away]["away_pi"]

        k_home = K_BASE * importance * (1 + max(away_combined, 0) / 5.0)
        k_away = K_BASE * importance * (1 + max(home_combined, 0) / 5.0)

        ratings[home]["home_pi"] += k_home * margin
        ratings[home]["games"] += 1
        ratings[away]["away_pi"] -= k_away * margin
        ratings[away]["games"] += 1

    return ratings


def _compute_form_features(results_df: pd.DataFrame, team: str,
                           before_date: str, n_matches: int = 10) -> dict:
    """Compute recent form stats for a team from results before a date."""
    t = _norm(team)
    cutoff = pd.Timestamp(before_date)
    df = results_df[results_df["date"] < cutoff].copy()

    # Filter team matches
    mask = (df["home_team"].apply(_norm) == t) | (df["away_team"].apply(_norm) == t)
    team_df = df[mask].sort_values("date", ascending=False).head(n_matches)

    if len(team_df) == 0:
        return {
            "form_win_pct": 0.5, "form_draw_pct": 0.2, "form_loss_pct": 0.3,
            "form_goals_scored_avg": 1.2, "form_goals_conceded_avg": 1.0,
            "form_clean_sheets_pct": 0.2, "avg_total_goals": 2.5,
        }

    wins, draws, losses = 0, 0, 0
    goals_for, goals_against = 0, 0
    clean_sheets = 0

    for _, row in team_df.iterrows():
        home = _norm(str(row["home_team"]))
        hs, as_ = float(row["home_score"]), float(row["away_score"])

        if home == t:
            gf, ga = hs, as_
        else:
            gf, ga = as_, hs

        goals_for += gf
        goals_against += ga
        if ga == 0:
            clean_sheets += 1
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    n = len(team_df)
    return {
        "form_win_pct": round(wins / n, 4),
        "form_draw_pct": round(draws / n, 4),
        "form_loss_pct": round(losses / n, 4),
        "form_goals_scored_avg": round(goals_for / n, 4),
        "form_goals_conceded_avg": round(goals_against / n, 4),
        "form_clean_sheets_pct": round(clean_sheets / n, 4),
        "avg_total_goals": round((goals_for + goals_against) / n, 4),
    }


def _compute_strength_features(results_df: pd.DataFrame, team: str,
                                before_date: str) -> dict:
    """Competitive vs friendly win rates."""
    t = _norm(team)
    cutoff = pd.Timestamp(before_date)
    start = cutoff - pd.DateOffset(years=4)
    df = results_df[(results_df["date"] >= start) & (results_df["date"] < cutoff)]

    mask = (df["home_team"].apply(_norm) == t) | (df["away_team"].apply(_norm) == t)
    team_df = df[mask]

    comp_wins, comp_total = 0, 0
    for _, row in team_df.iterrows():
        tourn = str(row.get("tournament", ""))
        is_comp = any(kw in tourn.lower() for kw in
                      ["world cup", "euro", "copa america", "nations league",
                       "african cup", "asian cup", "gold cup", "qualification"])
        if not is_comp:
            continue
        home = _norm(str(row["home_team"]))
        hs, as_ = float(row["home_score"]), float(row["away_score"])
        gf, ga = (hs, as_) if home == t else (as_, hs)
        comp_total += 1
        if gf > ga:
            comp_wins += 1

    return {
        "competitive_win_pct": round(comp_wins / max(comp_total, 1), 4),
        "competitive_matches": comp_total,
    }


def _build_historical_features(results_df: pd.DataFrame,
                                home: str, away: str,
                                before_date: str) -> dict:
    """
    Build feature vector for a historical match using only data available
    before that match date. This is the key anti-leakage constraint.
    """
    pi = _compute_pi_ratings(results_df, before_date)

    h, a = _norm(home), _norm(away)
    hp = pi.get(h, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
    ap = pi.get(a, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})

    home_combined = hp["home_pi"] + hp["away_pi"]
    away_combined = ap["home_pi"] + ap["away_pi"]

    features = {
        "home_pi_combined": round(home_combined, 4),
        "away_pi_combined": round(away_combined, 4),
        "pi_delta": round(home_combined - away_combined, 4),
    }

    # Form features
    home_form = _compute_form_features(results_df, home, before_date)
    away_form = _compute_form_features(results_df, away, before_date)
    features.update({f"home_{k}": v for k, v in home_form.items()})
    features.update({f"away_{k}": v for k, v in away_form.items()})

    # Derived
    features["form_win_pct_delta"] = round(
        features.get("home_form_win_pct", 0.5) - features.get("away_form_win_pct", 0.5), 4
    )

    # Strength features
    home_str = _compute_strength_features(results_df, home, before_date)
    away_str = _compute_strength_features(results_df, away, before_date)
    features.update({f"home_{k}": v for k, v in home_str.items()})
    features.update({f"away_{k}": v for k, v in away_str.items()})

    features["competitive_wp_delta"] = round(
        features.get("home_competitive_win_pct", 0.5)
        - features.get("away_competitive_win_pct", 0.5), 4
    )

    features["combined_avg_goals"] = round(
        (features.get("home_avg_total_goals", 2.5)
         + features.get("away_avg_total_goals", 2.5)) / 2, 4
    )

    return features


def _get_wc_matches(results_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Extract World Cup finals matches for a given year."""
    wc = results_df[results_df["tournament"] == "FIFA World Cup"].copy()
    return wc[wc["date"].dt.year == year].reset_index(drop=True)


def _outcome_label(hs: float, as_: float) -> int:
    """0=away win, 1=draw, 2=home win."""
    if hs > as_:
        return 2
    elif hs == as_:
        return 1
    else:
        return 0


def _outcome_home_win(hs: float, as_: float) -> int:
    """Binary: 1 if home win or draw (home doesn't lose), 0 if away win."""
    return 1 if hs >= as_ else 0


def run_historical_backtest(results_df: pd.DataFrame) -> dict:
    """
    Walk-forward backtest: for each tournament, train on prior WC + qualifying
    data, predict outcomes.

    Strategy:
      - 2014 WC: train on all results before 2014-06-01
      - 2018 WC: train on results before 2018-06-01 (including 2014 WC)
      - 2022 WC: train on results before 2022-11-01 (including 2014+2018 WC)
    """
    print("\n" + "=" * 75)
    print("  PART 1: HISTORICAL WORLD CUP BACKTEST (2014 / 2018 / 2022)")
    print("=" * 75)

    tournaments = [
        {"year": 2014, "cutoff": "2014-06-12"},
        {"year": 2018, "cutoff": "2018-06-14"},
        {"year": 2022, "cutoff": "2022-11-20"},
    ]

    all_results = {}
    combined_preds = []
    combined_actuals = []
    combined_probs = []

    # Pre-compute Pi ratings for each cutoff (expensive, do once per tournament)
    print("\nComputing Pi ratings for each tournament cutoff...")

    for t in tournaments:
        year = t["year"]
        cutoff = t["cutoff"]

        print(f"\n{'─' * 70}")
        print(f"  WC {year} — training on data before {cutoff}")
        print(f"{'─' * 70}")

        wc_matches = _get_wc_matches(results_df, year)
        print(f"  Matches: {len(wc_matches)}")

        # Pre-compute Pi ratings once for this cutoff
        pi_cache = _compute_pi_ratings(results_df, cutoff)

        # Build features for each WC match
        rows = []
        labels = []
        labels_binary = []

        for _, match in wc_matches.iterrows():
            home = str(match["home_team"])
            away = str(match["away_team"])
            match_date = match["date"].strftime("%Y-%m-%d")
            hs = float(match["home_score"])
            as_ = float(match["away_score"])

            # Use pre-computed Pi + form/strength features
            h, a = _norm(home), _norm(away)
            hp = pi_cache.get(h, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
            ap = pi_cache.get(a, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})

            home_combined = hp["home_pi"] + hp["away_pi"]
            away_combined = ap["home_pi"] + ap["away_pi"]

            features = {
                "home_pi_combined": round(home_combined, 4),
                "away_pi_combined": round(away_combined, 4),
                "pi_delta": round(home_combined - away_combined, 4),
            }

            # Form features (use cutoff, not per-match date, for speed)
            home_form = _compute_form_features(results_df, home, cutoff)
            away_form = _compute_form_features(results_df, away, cutoff)
            features.update({f"home_{k}": v for k, v in home_form.items()})
            features.update({f"away_{k}": v for k, v in away_form.items()})

            features["form_win_pct_delta"] = round(
                features.get("home_form_win_pct", 0.5)
                - features.get("away_form_win_pct", 0.5), 4)

            home_str = _compute_strength_features(results_df, home, cutoff)
            away_str = _compute_strength_features(results_df, away, cutoff)
            features.update({f"home_{k}": v for k, v in home_str.items()})
            features.update({f"away_{k}": v for k, v in away_str.items()})

            features["competitive_wp_delta"] = round(
                features.get("home_competitive_win_pct", 0.5)
                - features.get("away_competitive_win_pct", 0.5), 4)

            features["combined_avg_goals"] = round(
                (features.get("home_avg_total_goals", 2.5)
                 + features.get("away_avg_total_goals", 2.5)) / 2, 4)

            # Neutral venue flag (most WC games are neutral)
            features["is_neutral"] = 1 if str(match.get("neutral", "")).upper() == "TRUE" else 0

            rows.append(features)
            labels.append(_outcome_label(hs, as_))
            labels_binary.append(_outcome_home_win(hs, as_))

        feature_df = pd.DataFrame(rows)
        feature_cols = [c for c in feature_df.columns if feature_df[c].dtype in [np.float64, np.int64, float, int]]

        X = feature_df[feature_cols].fillna(0).values
        y = np.array(labels)
        y_bin = np.array(labels_binary)

        # ── Train on pre-tournament international results ──
        # Build a training set from *all* international results (not just WC)
        # in the 4 years before the tournament
        print("  Building training set from pre-tournament results...")
        cutoff_dt = pd.Timestamp(cutoff)
        train_start = cutoff_dt - pd.DateOffset(years=4)
        train_df = results_df[
            (results_df["date"] >= train_start) & (results_df["date"] < cutoff_dt)
        ].copy()

        # Subsample: competitive matches + random friendlies
        comp_mask = train_df["tournament"].str.lower().str.contains(
            "world cup|euro |copa america|nations league|african|asian cup|gold cup|qualification",
            na=False
        )
        train_comp = train_df[comp_mask]
        train_friendly = train_df[~comp_mask].sample(
            min(500, len(train_df[~comp_mask])), random_state=42
        ) if len(train_df[~comp_mask]) > 0 else pd.DataFrame()
        train_subset = pd.concat([train_comp, train_friendly]).sort_values("date")

        train_rows = []
        train_labels = []
        for _, row in train_subset.iterrows():
            home = str(row["home_team"])
            away = str(row["away_team"])
            hs = float(row["home_score"])
            as_ = float(row["away_score"])

            h, a = _norm(home), _norm(away)
            hp = pi_cache.get(h, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
            ap = pi_cache.get(a, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})

            home_combined = hp["home_pi"] + hp["away_pi"]
            away_combined = ap["home_pi"] + ap["away_pi"]

            feat = {
                "home_pi_combined": round(home_combined, 4),
                "away_pi_combined": round(away_combined, 4),
                "pi_delta": round(home_combined - away_combined, 4),
            }

            hf = _compute_form_features(results_df, home, cutoff)
            af = _compute_form_features(results_df, away, cutoff)
            feat.update({f"home_{k}": v for k, v in hf.items()})
            feat.update({f"away_{k}": v for k, v in af.items()})
            feat["form_win_pct_delta"] = round(
                feat.get("home_form_win_pct", 0.5) - feat.get("away_form_win_pct", 0.5), 4)

            hs_feat = _compute_strength_features(results_df, home, cutoff)
            as_feat = _compute_strength_features(results_df, away, cutoff)
            feat.update({f"home_{k}": v for k, v in hs_feat.items()})
            feat.update({f"away_{k}": v for k, v in as_feat.items()})
            feat["competitive_wp_delta"] = round(
                feat.get("home_competitive_win_pct", 0.5)
                - feat.get("away_competitive_win_pct", 0.5), 4)
            feat["combined_avg_goals"] = round(
                (feat.get("home_avg_total_goals", 2.5)
                 + feat.get("away_avg_total_goals", 2.5)) / 2, 4)
            feat["is_neutral"] = 1 if str(row.get("neutral", "")).upper() == "TRUE" else 0

            train_rows.append(feat)
            train_labels.append(_outcome_home_win(hs, as_))

        train_feat_df = pd.DataFrame(train_rows)
        # Ensure same columns
        for c in feature_cols:
            if c not in train_feat_df.columns:
                train_feat_df[c] = 0.0
        X_train = train_feat_df[feature_cols].fillna(0).values
        y_train = np.array(train_labels)

        print(f"  Training set: {len(X_train)} matches")
        print(f"  Test set:     {len(X)} matches (WC {year})")

        # Train model
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)

        # Predict on WC matches
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        # If only 2 classes in training, model predicts binary
        if y_prob.shape[1] == 2:
            home_win_prob = y_prob[:, 1]
        else:
            home_win_prob = y_prob[:, 1] if 1 in model.classes_ else np.full(len(X), 0.5)

        acc = accuracy_score(y_bin, y_pred)
        brier = brier_score_loss(y_bin, home_win_prob)

        try:
            auc = roc_auc_score(y_bin, home_win_prob)
        except ValueError:
            auc = 0.5

        print(f"\n  Results for WC {year}:")
        print(f"    Accuracy (home doesn't lose): {acc:.1%}")
        print(f"    Brier score:                  {brier:.4f}")
        print(f"    AUC-ROC:                      {auc:.4f}")
        print(f"    Home win rate (actual):        {(y_bin == 1).mean():.1%}")
        print(f"    Home win rate (predicted):     {(y_pred == 1).mean():.1%}")

        # Calibration buckets
        print(f"\n    Calibration (home win/draw prob):")
        buckets = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        for lo, hi in buckets:
            mask = (home_win_prob >= lo) & (home_win_prob < hi)
            if mask.sum() > 0:
                actual_rate = y_bin[mask].mean()
                pred_mean = home_win_prob[mask].mean()
                print(f"      [{lo:.0%}-{hi:.0%}): n={mask.sum():2d}  "
                      f"predicted={pred_mean:.1%}  actual={actual_rate:.1%}  "
                      f"gap={abs(actual_rate - pred_mean):.1%}")

        # Notable upsets (predicted home win but away won, or vice versa)
        upsets = []
        for i, (_, match) in enumerate(wc_matches.iterrows()):
            pred_hw = home_win_prob[i]
            actual = y_bin[i]
            if (pred_hw > 0.65 and actual == 0) or (pred_hw < 0.35 and actual == 1):
                upsets.append({
                    "match": f"{match['home_team']} vs {match['away_team']}",
                    "score": f"{int(match['home_score'])}-{int(match['away_score'])}",
                    "pred_home_win_prob": f"{pred_hw:.1%}",
                    "upset": "underdog won" if pred_hw > 0.5 else "favorite won"
                })

        if upsets:
            print(f"\n    Notable upsets ({len(upsets)}):")
            for u in upsets[:8]:
                print(f"      {u['match']} ({u['score']}) — "
                      f"p(home)={u['pred_home_win_prob']}, {u['upset']}")

        all_results[year] = {
            "matches": len(wc_matches),
            "accuracy": round(acc, 4),
            "brier": round(brier, 4),
            "auc": round(auc, 4),
            "upsets": len(upsets),
        }

        combined_preds.extend(y_pred.tolist())
        combined_actuals.extend(y_bin.tolist())
        combined_probs.extend(home_win_prob.tolist())

    # ── Combined stats across all 3 tournaments ──
    combined_preds = np.array(combined_preds)
    combined_actuals = np.array(combined_actuals)
    combined_probs = np.array(combined_probs)

    total_acc = accuracy_score(combined_actuals, combined_preds)
    total_brier = brier_score_loss(combined_actuals, combined_probs)
    try:
        total_auc = roc_auc_score(combined_actuals, combined_probs)
    except ValueError:
        total_auc = 0.5

    print(f"\n{'─' * 70}")
    print(f"  COMBINED (192 WC matches, 2014+2018+2022)")
    print(f"{'─' * 70}")
    print(f"    Accuracy:    {total_acc:.1%}")
    print(f"    Brier:       {total_brier:.4f}")
    print(f"    AUC-ROC:     {total_auc:.4f}")

    # Baseline: always predict home doesn't lose
    baseline_acc = combined_actuals.mean()
    print(f"    Baseline:    {baseline_acc:.1%} (always predict home doesn't lose)")
    print(f"    Lift:        {total_acc - baseline_acc:+.1%}")

    # Random baseline Brier
    random_brier = brier_score_loss(combined_actuals,
                                     np.full_like(combined_probs, combined_actuals.mean()))
    print(f"    Brier (random): {random_brier:.4f}")
    print(f"    Brier improvement: {(random_brier - total_brier) / random_brier:.1%}")

    all_results["combined"] = {
        "matches": len(combined_actuals),
        "accuracy": round(total_acc, 4),
        "brier": round(total_brier, 4),
        "auc": round(total_auc, 4),
        "baseline_acc": round(baseline_acc, 4),
    }

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Market Calibration Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_market_calibration() -> dict:
    """
    Compare model feature-based predictions vs bookmaker odds for WC 2026.
    Since WC hasn't happened yet, we analyze:
      1. Agreement between model and market
      2. Where model diverges from market (potential edges)
      3. Market-implied vs Pi-derived rankings
    """
    print("\n\n" + "=" * 75)
    print("  PART 2: MARKET CALIBRATION — MODEL vs BOOKMAKERS (WC 2026)")
    print("=" * 75)

    # Load feature matrix (already contains odds columns from wc_features.py)
    feat_path = os.path.join(_DATA_DIR, "wc2026_feature_matrix.csv")

    if not os.path.exists(feat_path):
        print("  [SKIP] Feature matrix not found")
        return {}

    df = pd.read_csv(feat_path)

    print(f"\n  Feature matrix: {len(df)} matches x {len(df.columns)} features")

    # Use odds columns already in the feature matrix (from wc_features.py)
    needed = ["pi_delta", "odds_home_implied", "odds_away_implied", "odds_draw_implied"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [SKIP] Missing columns: {missing}")
        return {}

    has_odds = df["odds_home_implied"].notna() & (df["odds_home_implied"] > 0)
    print(f"  Matches with odds: {has_odds.sum()}")

    # Convert pi_delta to implied home win probability using logistic
    # Calibrated on historical data: a 1.0 pi delta ~ 65% home win prob
    def _pi_to_prob(pi_delta: float) -> float:
        return 1.0 / (1.0 + math.exp(-0.5 * pi_delta))

    rows_for_analysis = []

    for _, row in df[has_odds].iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        pi_delta = float(row.get("pi_delta", 0))
        model_home_prob = _pi_to_prob(pi_delta)

        # Also check form
        form_delta = float(row.get("form_win_pct_delta", 0))
        comp_delta = float(row.get("competitive_wp_delta", 0))

        # Blend: 50% Pi, 25% form, 25% competitive strength
        blended_home = 0.5 * model_home_prob + 0.25 * (0.5 + form_delta) + 0.25 * (0.5 + comp_delta)
        blended_home = max(0.05, min(0.95, blended_home))

        market_home_win = float(row["odds_home_implied"])
        market_draw = float(row["odds_draw_implied"])
        market_home = market_home_win + market_draw * 0.5  # home doesn't lose

        edge = blended_home - market_home
        edge_win = model_home_prob - market_home_win

        rows_for_analysis.append({
            "match": f"{home} vs {away}",
            "pi_delta": pi_delta,
            "model_home_prob": round(blended_home, 3),
            "market_home_prob": round(market_home, 3),
            "market_home_win": round(market_home_win, 3),
            "edge": round(edge, 3),
            "edge_win": round(edge_win, 3),
            "agreement": abs(edge) < 0.10,
        })

    if not rows_for_analysis:
        print("  [SKIP] No matches with both features and odds")
        return {}

    analysis_df = pd.DataFrame(rows_for_analysis)
    agreement_rate = analysis_df["agreement"].mean()
    avg_edge = analysis_df["edge"].abs().mean()

    print(f"\n  Agreement rate (<10% divergence): {agreement_rate:.1%}")
    print(f"  Average absolute edge:           {avg_edge:.3f}")
    print(f"  Correlation (model vs market):    "
          f"{analysis_df['model_home_prob'].corr(analysis_df['market_home_prob']):.3f}")

    # Distribution of edges
    print(f"\n  Edge distribution:")
    for bucket, label in [
        (analysis_df["edge"] > 0.15, "  Strong model > market (>15%)"),
        ((analysis_df["edge"] > 0.05) & (analysis_df["edge"] <= 0.15), "  Mild model > market (5-15%)"),
        ((analysis_df["edge"] >= -0.05) & (analysis_df["edge"] <= 0.05), "  Agreement (±5%)"),
        ((analysis_df["edge"] < -0.05) & (analysis_df["edge"] >= -0.15), "  Mild market > model (5-15%)"),
        (analysis_df["edge"] < -0.15, "  Strong market > model (>15%)"),
    ]:
        n = bucket.sum()
        print(f"    {label}: {n} matches ({n/len(analysis_df):.0%})")

    # Top model edges (where model disagrees most with market)
    top_edges = analysis_df.reindex(
        analysis_df["edge"].abs().sort_values(ascending=False).index
    ).head(10)

    print(f"\n  Top 10 model-vs-market divergences:")
    print(f"    {'Match':<35} {'Model':>7} {'Market':>7} {'Edge':>7} {'Pi Δ':>6}")
    for _, r in top_edges.iterrows():
        direction = "MODEL" if r["edge"] > 0 else "MKT"
        print(f"    {r['match']:<35} {r['model_home_prob']:>6.1%} {r['market_home_prob']:>6.1%} "
              f"{r['edge']:>+6.1%}  {r['pi_delta']:>+5.2f}")

    # Market-implied rankings vs Pi rankings
    print(f"\n  Pi rating vs Market ranking comparison:")
    sorted_by_pi = analysis_df.sort_values("pi_delta", ascending=False)
    sorted_by_mkt = analysis_df.sort_values("market_home_win", ascending=False)

    # Extract unique teams and rank them
    team_pi_scores = defaultdict(list)
    team_mkt_scores = defaultdict(list)
    for _, r in analysis_df.iterrows():
        match = r["match"]
        home, away = match.split(" vs ")
        pi = r["pi_delta"]
        mkt_hw = r["market_home_win"]
        team_pi_scores[home].append(pi)
        team_pi_scores[away].append(-pi)
        team_mkt_scores[home].append(mkt_hw)
        team_mkt_scores[away].append(1 - mkt_hw)  # approximate

    team_avg_pi = {t: np.mean(v) for t, v in team_pi_scores.items()}
    team_avg_mkt = {t: np.mean(v) for t, v in team_mkt_scores.items()}

    pi_ranking = sorted(team_avg_pi.items(), key=lambda x: -x[1])
    mkt_ranking = sorted(team_avg_mkt.items(), key=lambda x: -x[1])

    pi_rank_map = {t: i+1 for i, (t, _) in enumerate(pi_ranking)}
    mkt_rank_map = {t: i+1 for i, (t, _) in enumerate(mkt_ranking)}

    print(f"    {'Team':<25} {'Pi Rank':>8} {'Mkt Rank':>9} {'Diff':>6}")
    for team, _ in pi_ranking[:15]:
        pr = pi_rank_map.get(team, 99)
        mr = mkt_rank_map.get(team, 99)
        diff = mr - pr
        flag = " ⚠" if abs(diff) >= 10 else ""
        print(f"    {team:<25} {pr:>8} {mr:>9} {diff:>+5}{flag}")

    results = {
        "matches_analyzed": len(analysis_df),
        "agreement_rate": round(agreement_rate, 4),
        "avg_abs_edge": round(avg_edge, 4),
        "correlation": round(
            analysis_df["model_home_prob"].corr(analysis_df["market_home_prob"]), 4),
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Feature Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_feature_analysis(results_df: pd.DataFrame) -> dict:
    """
    Analyze which features are most predictive, check multicollinearity,
    and rank by importance using gradient boosting on historical WC data.
    """
    print("\n\n" + "=" * 75)
    print("  PART 3: FEATURE CORRELATION & IMPORTANCE ANALYSIS")
    print("=" * 75)

    # Build a combined dataset from all 3 WC tournaments
    all_rows = []
    all_labels = []

    for year, cutoff in [(2014, "2014-06-12"), (2018, "2018-06-14"), (2022, "2022-11-20")]:
        wc = _get_wc_matches(results_df, year)
        pi_cache = _compute_pi_ratings(results_df, cutoff)

        for _, match in wc.iterrows():
            home = str(match["home_team"])
            away = str(match["away_team"])
            hs = float(match["home_score"])
            as_ = float(match["away_score"])

            h, a = _norm(home), _norm(away)
            hp = pi_cache.get(h, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
            ap = pi_cache.get(a, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})

            home_combined = hp["home_pi"] + hp["away_pi"]
            away_combined = ap["home_pi"] + ap["away_pi"]

            feat = {
                "home_pi_combined": round(home_combined, 4),
                "away_pi_combined": round(away_combined, 4),
                "pi_delta": round(home_combined - away_combined, 4),
            }

            hf = _compute_form_features(results_df, home, cutoff)
            af = _compute_form_features(results_df, away, cutoff)
            feat.update({f"home_{k}": v for k, v in hf.items()})
            feat.update({f"away_{k}": v for k, v in af.items()})

            feat["form_win_pct_delta"] = round(
                feat.get("home_form_win_pct", 0.5) - feat.get("away_form_win_pct", 0.5), 4)

            hs_feat = _compute_strength_features(results_df, home, cutoff)
            as_feat = _compute_strength_features(results_df, away, cutoff)
            feat.update({f"home_{k}": v for k, v in hs_feat.items()})
            feat.update({f"away_{k}": v for k, v in as_feat.items()})

            feat["competitive_wp_delta"] = round(
                feat.get("home_competitive_win_pct", 0.5)
                - feat.get("away_competitive_win_pct", 0.5), 4)
            feat["combined_avg_goals"] = round(
                (feat.get("home_avg_total_goals", 2.5)
                 + feat.get("away_avg_total_goals", 2.5)) / 2, 4)
            feat["is_neutral"] = 1 if str(match.get("neutral", "")).upper() == "TRUE" else 0

            all_rows.append(feat)
            all_labels.append(_outcome_home_win(hs, as_))

    df = pd.DataFrame(all_rows)
    y = np.array(all_labels)

    feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, float, int]]
    X = df[feature_cols].fillna(0)

    print(f"\n  Dataset: {len(X)} matches x {len(feature_cols)} features")
    print(f"  Home win/draw rate: {y.mean():.1%}")

    # ── 1. Feature importance via GradientBoosting ──
    print(f"\n  {'─' * 60}")
    print(f"  Feature Importance (GradientBoosting)")
    print(f"  {'─' * 60}")

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X.values, y)

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    print(f"\n    {'Feature':<40} {'Importance':>10}")
    print(f"    {'─' * 50}")
    for feat_name, imp in importances.head(20).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat_name:<40} {imp:>9.4f}  {bar}")

    # ── 2. Cross-validated performance ──
    print(f"\n  {'─' * 60}")
    print(f"  Cross-Validated Performance (5-fold)")
    print(f"  {'─' * 60}")

    cv_scores = cross_val_score(model, X.values, y, cv=5, scoring="accuracy")
    cv_brier = cross_val_score(model, X.values, y, cv=5, scoring="neg_brier_score")
    cv_auc = cross_val_score(model, X.values, y, cv=5, scoring="roc_auc")

    print(f"    Accuracy:  {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"    Brier:     {-cv_brier.mean():.4f} ± {cv_brier.std():.4f}")
    print(f"    AUC-ROC:   {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # ── 3. Correlation matrix (top features) ──
    print(f"\n  {'─' * 60}")
    print(f"  Multicollinearity Check (top 10 features)")
    print(f"  {'─' * 60}")

    top_features = importances.head(10).index.tolist()
    corr = X[top_features].corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i, f1 in enumerate(top_features):
        for j, f2 in enumerate(top_features):
            if i < j and abs(corr.loc[f1, f2]) > 0.7:
                high_corr_pairs.append((f1, f2, round(corr.loc[f1, f2], 3)))

    if high_corr_pairs:
        print(f"\n    High correlation pairs (|r| > 0.7):")
        for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"      {f1} <-> {f2}: r={r}")
    else:
        print(f"\n    No highly correlated pairs found among top 10 features")

    # ── 4. Feature-outcome correlations ──
    print(f"\n  {'─' * 60}")
    print(f"  Feature-Outcome Correlations (point biserial)")
    print(f"  {'─' * 60}")

    correlations = {}
    for col in feature_cols:
        r = np.corrcoef(X[col].values, y)[0, 1]
        if not np.isnan(r):
            correlations[col] = round(r, 4)

    sorted_corr = sorted(correlations.items(), key=lambda x: -abs(x[1]))

    print(f"\n    {'Feature':<40} {'Corr(outcome)':>13}")
    print(f"    {'─' * 53}")
    for feat_name, r in sorted_corr[:15]:
        direction = "+" if r > 0 else "-"
        bar = "█" * int(abs(r) * 50)
        print(f"    {feat_name:<40} {r:>+12.4f}  {bar}")

    # ── 5. Feature redundancy analysis ──
    print(f"\n  {'─' * 60}")
    print(f"  Feature Redundancy — Candidates for Removal")
    print(f"  {'─' * 60}")

    # Features that have near-zero importance AND high correlation with a better feature
    low_importance = importances[importances < 0.01].index.tolist()
    redundant = []
    for feat in low_importance:
        for top_feat in importances.head(5).index:
            if feat != top_feat and feat in X.columns and top_feat in X.columns:
                r = abs(X[feat].corr(X[top_feat]))
                if r > 0.6:
                    redundant.append((feat, top_feat, round(r, 3),
                                     round(float(importances[feat]), 4)))
                    break

    if redundant:
        print(f"\n    Low-importance features correlated with top features:")
        for feat, corr_with, r, imp in redundant[:10]:
            print(f"      {feat:<35} imp={imp:.4f}  corr({corr_with})={r}")
    else:
        print(f"\n    No clearly redundant features found")

    # ── 6. Summary for WC 2026 feature matrix ──
    print(f"\n  {'─' * 60}")
    print(f"  Summary: Feature Quality Assessment")
    print(f"  {'─' * 60}")

    n_useful = len([f for f, imp in importances.items() if imp > 0.01])
    n_low = len([f for f, imp in importances.items() if imp <= 0.01])

    print(f"    Total features:     {len(feature_cols)}")
    print(f"    Useful (imp>1%):    {n_useful}")
    print(f"    Low value (imp<1%): {n_low}")
    print(f"    Top 5 features explain: {importances.head(5).sum():.1%} of variance")
    print(f"    Top 10 features explain: {importances.head(10).sum():.1%} of variance")

    return {
        "total_features": len(feature_cols),
        "useful_features": n_useful,
        "top_5_importance_sum": round(float(importances.head(5).sum()), 4),
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "cv_auc": round(float(cv_auc.mean()), 4),
        "high_corr_pairs": len(high_corr_pairs),
        "top_features": importances.head(10).to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Monte Carlo Tournament Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def _american_to_implied(american: float) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100.0 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def _match_probs(pi_home: float, pi_away: float) -> tuple:
    """
    Convert Pi delta to 3-way probabilities (home_win, draw, away_win).
    Calibrated from historical WC data: ~30% of WC matches are draws,
    and Pi delta of 1.0 corresponds to ~55-60% higher-rated team winning.
    """
    delta = pi_home - pi_away

    # Logistic for home win advantage
    home_strength = 1.0 / (1.0 + math.exp(-0.35 * delta))

    # WC base draw rate ~25-30%
    draw_base = 0.27
    # Draws are more likely when teams are evenly matched
    draw_prob = draw_base * math.exp(-0.04 * delta ** 2)
    draw_prob = max(0.10, min(0.40, draw_prob))

    remaining = 1.0 - draw_prob
    home_win = remaining * home_strength
    away_win = remaining * (1.0 - home_strength)

    return home_win, draw_prob, away_win


def _simulate_match(home_pi: float, away_pi: float, rng: np.random.Generator,
                    allow_draw: bool = True) -> str:
    """Simulate a single match. Returns 'home', 'draw', or 'away'."""
    hw, dr, aw = _match_probs(home_pi, away_pi)
    if not allow_draw:
        # Knockout: redistribute draw to home/away
        hw = hw + dr * (hw / (hw + aw))
        aw = 1.0 - hw
        dr = 0.0

    r = rng.random()
    if r < hw:
        return "home"
    elif r < hw + dr:
        return "draw"
    else:
        return "away"


def run_monte_carlo(n_sims: int = 10000) -> dict:
    """
    Monte Carlo simulation of WC 2026.
    Simulate the full tournament 10,000 times using Pi ratings.
    Compare advancement/win probabilities to market futures.
    """
    print("\n\n" + "=" * 75)
    print(f"  PART 4: MONTE CARLO TOURNAMENT SIMULATION ({n_sims:,} runs)")
    print("=" * 75)

    # Load data
    def _lj(fn):
        path = os.path.join(_DATA_DIR, fn)
        with open(path) as f:
            return json.load(f)

    try:
        groups = _lj("wc2026_groups.json")
        pi_data = _lj("wc2026_pi_ratings_v2.json")
        winner_odds_raw = _lj("wc2026_winner_odds.json")
    except FileNotFoundError as e:
        print(f"  [SKIP] Missing data: {e}")
        return {}

    # Parse market futures (average across bookmakers)
    market_futures = {}
    for entry in winner_odds_raw:
        for bm in entry.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "outrights":
                    for outcome in mkt["outcomes"]:
                        team = _norm(outcome["name"])
                        prob = _american_to_implied(outcome["price"])
                        if team not in market_futures:
                            market_futures[team] = []
                        market_futures[team].append(prob)

    # Average across books
    market_futures = {t: np.mean(probs) for t, probs in market_futures.items()}

    def _get_pi(team: str) -> float:
        t = _norm(team)
        data = pi_data.get(t, {"home_pi": 0.0, "away_pi": 0.0})
        return data["home_pi"] + data["away_pi"]

    # Clean up group data — each team appears in exactly one group
    # (the JSON has some teams in multiple groups due to scheduling format)
    team_primary_group = {}
    for grp, teams in groups.items():
        if len(grp) > 1:
            continue  # Skip compound groups like "A/B"
        for team in teams:
            t = _norm(team)
            if t not in team_primary_group:
                team_primary_group[t] = grp

    # Build clean groups
    clean_groups = defaultdict(list)
    for team, grp in team_primary_group.items():
        clean_groups[grp].append(team)

    # Only simulate groups with exactly 4 teams
    valid_groups = {g: teams for g, teams in clean_groups.items() if len(teams) == 4}

    all_teams = set()
    for teams in valid_groups.values():
        all_teams.update(teams)

    print(f"\n  Groups: {len(valid_groups)} (with 4 teams each)")
    print(f"  Teams: {len(all_teams)}")
    print(f"  Market futures for: {len(market_futures)} teams")
    print(f"  Simulations: {n_sims:,}")

    # Tracking
    group_wins = defaultdict(int)     # team -> times won group
    advanced = defaultdict(int)       # team -> times advanced past group
    quarterfinals = defaultdict(int)  # team -> times reached QF
    semifinals = defaultdict(int)     # team -> times reached SF
    finals = defaultdict(int)         # team -> times reached final
    champion = defaultdict(int)       # team -> times won tournament

    rng = np.random.default_rng(42)

    for sim in range(n_sims):
        # ── Group stage ──
        group_winners = {}   # group -> (1st, 2nd, 3rd)

        for grp, teams in sorted(valid_groups.items()):
            # Each team plays 3 matches (round-robin)
            points = {t: 0 for t in teams}
            gd = {t: 0.0 for t in teams}

            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    home, away = teams[i], teams[j]
                    pi_h, pi_a = _get_pi(home), _get_pi(away)
                    result = _simulate_match(pi_h, pi_a, rng, allow_draw=True)

                    if result == "home":
                        points[home] += 3
                        gd[home] += 1
                        gd[away] -= 1
                    elif result == "draw":
                        points[home] += 1
                        points[away] += 1
                    else:
                        points[away] += 3
                        gd[away] += 1
                        gd[home] -= 1

            # Sort by points, then GD, then Pi (tiebreaker)
            ranked = sorted(teams, key=lambda t: (-points[t], -gd[t], -_get_pi(t)))
            group_winners[grp] = ranked

            # Top 2 advance (in 2026: top 2 + best 3rd places, but simplify to top 2)
            for t in ranked[:2]:
                advanced[t] += 1
            group_wins[ranked[0]] += 1

        # ── Knockout stage (simplified bracket) ──
        # 2026 WC has 12 groups → 24 teams advance (top 2) + 8 best 3rd → 32 in R32
        # Simplify: take top 2 from each group → 24 teams
        # Then pair winners vs runners-up from different groups

        advancing = []
        for grp in sorted(group_winners.keys()):
            advancing.append((grp, 1, group_winners[grp][0]))
            advancing.append((grp, 2, group_winners[grp][1]))

        # Shuffle advancing teams into bracket (deterministic pairing by group)
        # Simple: pair group A 1st vs group B 2nd, etc.
        group_list = sorted(valid_groups.keys())
        r32_matches = []

        # Pair groups: A-B, C-D, E-F, G-H, I-J, K-L
        for i in range(0, len(group_list) - 1, 2):
            g1, g2 = group_list[i], group_list[i+1]
            if g1 in group_winners and g2 in group_winners:
                r32_matches.append((group_winners[g1][0], group_winners[g2][1]))
                r32_matches.append((group_winners[g2][0], group_winners[g1][1]))

        # Round of 32 → QF
        qf_teams = []
        for home, away in r32_matches:
            result = _simulate_match(_get_pi(home), _get_pi(away), rng, allow_draw=False)
            winner = home if result == "home" else away
            qf_teams.append(winner)
            quarterfinals[winner] += 1

        # QF → SF
        sf_teams = []
        for i in range(0, len(qf_teams) - 1, 2):
            home, away = qf_teams[i], qf_teams[i+1]
            result = _simulate_match(_get_pi(home), _get_pi(away), rng, allow_draw=False)
            winner = home if result == "home" else away
            sf_teams.append(winner)
            semifinals[winner] += 1

        # SF → Final
        final_teams = []
        for i in range(0, len(sf_teams) - 1, 2):
            home, away = sf_teams[i], sf_teams[i+1]
            result = _simulate_match(_get_pi(home), _get_pi(away), rng, allow_draw=False)
            winner = home if result == "home" else away
            final_teams.append(winner)
            finals[winner] += 1

        # Final
        if len(final_teams) >= 2:
            result = _simulate_match(
                _get_pi(final_teams[0]), _get_pi(final_teams[1]),
                rng, allow_draw=False
            )
            winner = final_teams[0] if result == "home" else final_teams[1]
            champion[winner] += 1

    # ── Results ──
    print(f"\n  {'─' * 65}")
    print(f"  Tournament Win Probabilities — Model vs Market")
    print(f"  {'─' * 65}")

    # Sort by model win probability
    model_probs = {t: champion.get(t, 0) / n_sims for t in all_teams}
    sorted_teams = sorted(model_probs.items(), key=lambda x: -x[1])

    print(f"\n    {'Team':<22} {'Model':>7} {'Market':>7} {'Edge':>7} "
          f"{'Advnce':>7} {'QF':>5} {'SF':>5} {'Final':>6}")
    print(f"    {'─' * 75}")

    edges = []
    for team, win_pct in sorted_teams[:25]:
        mkt = market_futures.get(team, 0)
        adv = advanced.get(team, 0) / n_sims
        qf = quarterfinals.get(team, 0) / n_sims
        sf = semifinals.get(team, 0) / n_sims
        fn = finals.get(team, 0) / n_sims
        edge = win_pct - mkt

        if mkt > 0:
            edges.append({"team": team, "model": win_pct, "market": mkt, "edge": edge})

        print(f"    {team:<22} {win_pct:>6.1%} {mkt:>6.1%} {edge:>+6.1%}  "
              f"{adv:>6.1%} {qf:>4.1%} {sf:>4.1%} {fn:>5.1%}")

    # ── Value bets (model > market by significant margin) ──
    print(f"\n  {'─' * 65}")
    print(f"  Potential Value Bets (model probability > market implied)")
    print(f"  {'─' * 65}")

    value_bets = sorted(edges, key=lambda x: -x["edge"])
    for vb in value_bets[:10]:
        if vb["edge"] > 0.005:
            ratio = vb["model"] / max(vb["market"], 0.001)
            print(f"    {vb['team']:<22} model={vb['model']:.1%}  "
                  f"market={vb['market']:.1%}  edge={vb['edge']:+.1%}  "
                  f"ratio={ratio:.2f}x")

    # ── Overvalued by market ──
    print(f"\n  {'─' * 65}")
    print(f"  Overvalued by Market (market probability > model)")
    print(f"  {'─' * 65}")

    overvalued = sorted(edges, key=lambda x: x["edge"])
    for ov in overvalued[:10]:
        if ov["edge"] < -0.005:
            print(f"    {ov['team']:<22} model={ov['model']:.1%}  "
                  f"market={ov['market']:.1%}  edge={ov['edge']:+.1%}")

    # ── Group advancement rates ──
    print(f"\n  {'─' * 65}")
    print(f"  Group Stage Advancement Rates")
    print(f"  {'─' * 65}")

    for grp in sorted(valid_groups.keys()):
        teams = valid_groups[grp]
        print(f"\n    Group {grp}:")
        ranked = sorted(teams, key=lambda t: -advanced.get(t, 0))
        for t in ranked:
            adv_pct = advanced.get(t, 0) / n_sims
            win_pct = group_wins.get(t, 0) / n_sims
            print(f"      {t:<22} advance={adv_pct:.0%}  win_group={win_pct:.0%}  "
                  f"pi={_get_pi(t):+.2f}")

    return {
        "simulations": n_sims,
        "champion_probs": {t: round(champion.get(t, 0) / n_sims, 4)
                           for t in sorted(all_teams)},
        "value_bets": [{"team": vb["team"], "model": round(vb["model"], 4),
                        "market": round(vb["market"], 4),
                        "edge": round(vb["edge"], 4)}
                       for vb in value_bets[:10] if vb["edge"] > 0.005],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 75)
    print("  WC 2026 PRESSURE TEST FRAMEWORK")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 75)

    # Load international results
    results_path = os.path.join(_DATA_DIR, "international_results.csv")
    results_df = pd.read_csv(results_path)
    results_df["date"] = pd.to_datetime(results_df["date"])
    print(f"\nLoaded {len(results_df)} international results")

    # Run all four analyses
    backtest_results = run_historical_backtest(results_df)
    calibration_results = run_market_calibration()
    feature_results = run_feature_analysis(results_df)
    mc_results = run_monte_carlo(n_sims=10000)

    # ── Final summary ──
    print("\n\n" + "█" * 75)
    print("  PRESSURE TEST SUMMARY")
    print("█" * 75)

    print("\n  Historical Backtest:")
    for year in [2014, 2018, 2022]:
        r = backtest_results.get(year, {})
        print(f"    WC {year}: Acc={r.get('accuracy',0):.1%}  "
              f"Brier={r.get('brier',0):.4f}  "
              f"AUC={r.get('auc',0):.4f}")
    c = backtest_results.get("combined", {})
    print(f"    Combined: Acc={c.get('accuracy',0):.1%}  "
          f"Brier={c.get('brier',0):.4f}  "
          f"AUC={c.get('auc',0):.4f}  "
          f"(baseline={c.get('baseline_acc',0):.1%})")

    if calibration_results:
        print(f"\n  Market Calibration:")
        print(f"    Matches: {calibration_results.get('matches_analyzed', 0)}")
        print(f"    Agreement: {calibration_results.get('agreement_rate', 0):.1%}")
        print(f"    Correlation: {calibration_results.get('correlation', 0):.3f}")

    if feature_results:
        print(f"\n  Feature Analysis:")
        print(f"    Features: {feature_results.get('total_features', 0)} total, "
              f"{feature_results.get('useful_features', 0)} useful")
        print(f"    CV Accuracy: {feature_results.get('cv_accuracy', 0):.1%}")
        print(f"    CV AUC: {feature_results.get('cv_auc', 0):.4f}")
        print(f"    Top 5 explain: {feature_results.get('top_5_importance_sum', 0):.1%}")

    if mc_results:
        print(f"\n  Monte Carlo ({mc_results.get('simulations', 0):,} sims):")
        champs = mc_results.get("champion_probs", {})
        top3 = sorted(champs.items(), key=lambda x: -x[1])[:3]
        for team, prob in top3:
            print(f"    {team}: {prob:.1%}")
        vb = mc_results.get("value_bets", [])
        if vb:
            print(f"    Value bets found: {len(vb)}")

    # Save full results
    out_path = os.path.join(_DATA_DIR, "wc2026_pressure_test_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "backtest": backtest_results,
            "calibration": calibration_results,
            "feature_analysis": feature_results,
            "monte_carlo": mc_results,
        }, f, indent=2, default=str)
    print(f"\n  Full results saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
