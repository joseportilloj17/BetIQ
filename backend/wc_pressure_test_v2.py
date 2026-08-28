"""
wc_pressure_test_v2.py — Improved WC Pressure Test

Three fixes over v1:
  1. Softer Pi coefficient: sigmoid(0.25 * delta) not sigmoid(0.5 * delta)
  2. Market odds as model features (not just calibration targets)
  3. Three-way outcome modeling (home/draw/away) not binary
  + Fix Monte Carlo to cover all 48 teams

Compares v1 (binary, no odds, aggressive Pi) vs v2 (3-way, odds features, soft Pi).
"""
from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

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
    "Chinese Taipei": "Taiwan",
}

def _norm(n: str) -> str:
    return _NAME_ALIASES.get(n, n)

def _lj(fn):
    with open(os.path.join(_DATA_DIR, fn)) as f:
        return json.load(f)

# ── Pi ratings ──────────────────────────────────────────────────────────────

def _compute_pi_ratings(results_df, before_date, lookback_years=4):
    cutoff = pd.Timestamp(before_date)
    start = cutoff - pd.DateOffset(years=lookback_years)
    df = results_df[(results_df["date"] >= start) & (results_df["date"] < cutoff)]

    ratings = {}
    K_BASE = 0.1
    for _, row in df.iterrows():
        home, away = _norm(str(row["home_team"])), _norm(str(row["away_team"]))
        margin = float(row["home_score"]) - float(row["away_score"])
        tourn = str(row.get("tournament", "")).lower()
        is_comp = any(kw in tourn for kw in
                      ["world cup","euro","copa america","nations league",
                       "african cup","asian cup","gold cup","confederations","qualification"])
        importance = 1.0 if is_comp else 0.5

        for t in [home, away]:
            if t not in ratings:
                ratings[t] = {"home_pi": 0.0, "away_pi": 0.0, "games": 0}

        hc = ratings[home]["home_pi"] + ratings[home]["away_pi"]
        ac = ratings[away]["home_pi"] + ratings[away]["away_pi"]
        k_h = K_BASE * importance * (1 + max(ac, 0) / 5.0)
        k_a = K_BASE * importance * (1 + max(hc, 0) / 5.0)
        ratings[home]["home_pi"] += k_h * margin
        ratings[home]["games"] += 1
        ratings[away]["away_pi"] -= k_a * margin
        ratings[away]["games"] += 1
    return ratings


def _form_features(results_df, team, before_date, n=10):
    t = _norm(team)
    cutoff = pd.Timestamp(before_date)
    df = results_df[results_df["date"] < cutoff]
    mask = (df["home_team"].apply(_norm) == t) | (df["away_team"].apply(_norm) == t)
    tdf = df[mask].sort_values("date", ascending=False).head(n)

    if len(tdf) == 0:
        return {"form_win_pct": 0.5, "form_draw_pct": 0.2, "form_loss_pct": 0.3,
                "form_gf_avg": 1.2, "form_ga_avg": 1.0, "form_cs_pct": 0.2,
                "avg_total_goals": 2.5}

    w, d, l, gf, ga, cs = 0, 0, 0, 0, 0, 0
    for _, row in tdf.iterrows():
        h = _norm(str(row["home_team"]))
        hs, as_ = float(row["home_score"]), float(row["away_score"])
        g_for, g_ag = (hs, as_) if h == t else (as_, hs)
        gf += g_for; ga += g_ag
        if g_ag == 0: cs += 1
        if g_for > g_ag: w += 1
        elif g_for == g_ag: d += 1
        else: l += 1

    m = len(tdf)
    return {"form_win_pct": round(w/m, 4), "form_draw_pct": round(d/m, 4),
            "form_loss_pct": round(l/m, 4), "form_gf_avg": round(gf/m, 4),
            "form_ga_avg": round(ga/m, 4), "form_cs_pct": round(cs/m, 4),
            "avg_total_goals": round((gf+ga)/m, 4)}


def _strength_features(results_df, team, before_date):
    t = _norm(team)
    cutoff = pd.Timestamp(before_date)
    start = cutoff - pd.DateOffset(years=4)
    df = results_df[(results_df["date"] >= start) & (results_df["date"] < cutoff)]
    mask = (df["home_team"].apply(_norm) == t) | (df["away_team"].apply(_norm) == t)
    tdf = df[mask]

    cw, ct = 0, 0
    for _, row in tdf.iterrows():
        tourn = str(row.get("tournament", "")).lower()
        if not any(kw in tourn for kw in
                   ["world cup","euro","copa america","nations league",
                    "african cup","asian cup","gold cup","qualification"]):
            continue
        h = _norm(str(row["home_team"]))
        hs, as_ = float(row["home_score"]), float(row["away_score"])
        g_for, g_ag = (hs, as_) if h == t else (as_, hs)
        ct += 1
        if g_for > g_ag: cw += 1
    return {"comp_win_pct": round(cw / max(ct, 1), 4), "comp_matches": ct}


# ── Build feature vector for one match ──────────────────────────────────────

def _build_features(pi_cache, results_df, home, away, cutoff, is_neutral=True):
    """Build feature dict for a single match. No odds (historical data has none)."""
    h, a = _norm(home), _norm(away)
    hp = pi_cache.get(h, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
    ap = pi_cache.get(a, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
    hc = hp["home_pi"] + hp["away_pi"]
    ac = ap["home_pi"] + ap["away_pi"]

    feat = {
        "home_pi": round(hc, 4),
        "away_pi": round(ac, 4),
        "pi_delta": round(hc - ac, 4),
        # Softer Pi probability (FIX 1) — use 0.25 coefficient
        "pi_home_prob": round(1.0 / (1.0 + math.exp(-0.25 * (hc - ac))), 4),
    }

    hf = _form_features(results_df, home, cutoff)
    af = _form_features(results_df, away, cutoff)
    feat.update({f"home_{k}": v for k, v in hf.items()})
    feat.update({f"away_{k}": v for k, v in af.items()})

    hs = _strength_features(results_df, home, cutoff)
    as_ = _strength_features(results_df, away, cutoff)
    feat.update({f"home_{k}": v for k, v in hs.items()})
    feat.update({f"away_{k}": v for k, v in as_.items()})

    feat["form_wp_delta"] = round(feat["home_form_win_pct"] - feat["away_form_win_pct"], 4)
    feat["comp_wp_delta"] = round(feat["home_comp_win_pct"] - feat["away_comp_win_pct"], 4)
    feat["combined_goals"] = round((feat["home_avg_total_goals"] + feat["away_avg_total_goals"]) / 2, 4)
    feat["is_neutral"] = 1 if is_neutral else 0

    return feat


def _get_wc_matches(results_df, year):
    wc = results_df[results_df["tournament"] == "FIFA World Cup"]
    return wc[wc["date"].dt.year == year].reset_index(drop=True)


def _outcome_3way(hs, as_):
    """0=away, 1=draw, 2=home."""
    if hs > as_: return 2
    elif hs == as_: return 1
    else: return 0


# ═══════════════════════════════════════════════════════════════════════════════
# V2 Backtest — Three-way classification with softer Pi
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_v2(results_df):
    """
    Walk-forward backtest with three fixes:
      1. Softer Pi coefficient (0.25 not 0.5)
      2. pi_home_prob as explicit feature
      3. Three-way GBM: classes {0=away, 1=draw, 2=home}

    Reports both 3-way accuracy and derived binary metrics for comparison.
    """
    print("\n" + "=" * 75)
    print("  V2 BACKTEST — 3-WAY + SOFT Pi + pi_home_prob FEATURE")
    print("=" * 75)

    tournaments = [
        {"year": 2014, "cutoff": "2014-06-12"},
        {"year": 2018, "cutoff": "2018-06-14"},
        {"year": 2022, "cutoff": "2022-11-20"},
    ]

    all_results = {}
    comb_pred3 = []
    comb_actual3 = []
    comb_probs_hw = []   # P(home win)
    comb_probs_dr = []   # P(draw)
    comb_probs_aw = []   # P(away win)
    comb_actual_bin = []  # binary: home doesn't lose

    for t in tournaments:
        year, cutoff = t["year"], t["cutoff"]
        print(f"\n{'─' * 70}")
        print(f"  WC {year}")
        print(f"{'─' * 70}")

        wc = _get_wc_matches(results_df, year)
        pi_cache = _compute_pi_ratings(results_df, cutoff)

        # ── Build test set (WC matches) ──
        test_rows, test_y3 = [], []
        for _, m in wc.iterrows():
            home, away = str(m["home_team"]), str(m["away_team"])
            neutral = str(m.get("neutral", "")).upper() == "TRUE"
            feat = _build_features(pi_cache, results_df, home, away, cutoff, neutral)
            test_rows.append(feat)
            test_y3.append(_outcome_3way(float(m["home_score"]), float(m["away_score"])))

        test_df = pd.DataFrame(test_rows)
        feat_cols = [c for c in test_df.columns if test_df[c].dtype in [np.float64, np.int64, float, int]]
        X_test = test_df[feat_cols].fillna(0).values
        y_test = np.array(test_y3)

        # ── Build training set ──
        cutoff_dt = pd.Timestamp(cutoff)
        train_start = cutoff_dt - pd.DateOffset(years=4)
        train_df = results_df[(results_df["date"] >= train_start) & (results_df["date"] < cutoff_dt)]

        comp_mask = train_df["tournament"].str.lower().str.contains(
            "world cup|euro |copa america|nations league|african|asian cup|gold cup|qualification", na=False)
        train_comp = train_df[comp_mask]
        train_friendly = train_df[~comp_mask]
        if len(train_friendly) > 500:
            train_friendly = train_friendly.sample(500, random_state=42)
        train_subset = pd.concat([train_comp, train_friendly]).sort_values("date")

        train_rows, train_y3 = [], []
        for _, row in train_subset.iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            neutral = str(row.get("neutral", "")).upper() == "TRUE"
            feat = _build_features(pi_cache, results_df, home, away, cutoff, neutral)
            train_rows.append(feat)
            train_y3.append(_outcome_3way(float(row["home_score"]), float(row["away_score"])))

        train_feat_df = pd.DataFrame(train_rows)
        for c in feat_cols:
            if c not in train_feat_df.columns:
                train_feat_df[c] = 0.0
        X_train = train_feat_df[feat_cols].fillna(0).values
        y_train = np.array(train_y3)

        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"  Train distribution: away={sum(y_train==0)}, draw={sum(y_train==1)}, home={sum(y_train==2)}")

        # ── Train 3-way GBM ──
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)  # shape: (n, 3) for classes [0,1,2]

        # Map probabilities to classes
        class_map = {c: i for i, c in enumerate(model.classes_)}
        idx_away = class_map.get(0, 0)
        idx_draw = class_map.get(1, 1)
        idx_home = class_map.get(2, 2)

        p_away = y_prob[:, idx_away]
        p_draw = y_prob[:, idx_draw]
        p_home = y_prob[:, idx_home]

        # ── 3-way metrics ──
        acc3 = accuracy_score(y_test, y_pred)

        # Multi-class Brier: mean of (predicted_prob - indicator)^2 across all classes
        brier3 = 0.0
        for i in range(len(y_test)):
            for cls_idx, cls_val in enumerate(model.classes_):
                indicator = 1.0 if y_test[i] == cls_val else 0.0
                brier3 += (y_prob[i, cls_idx] - indicator) ** 2
        brier3 /= len(y_test)

        # ── Derived binary metrics (for comparison with v1) ──
        y_bin = (y_test >= 1).astype(int)  # home doesn't lose = draw or win
        p_home_not_lose = p_draw + p_home
        pred_bin = (p_home_not_lose >= 0.5).astype(int)

        acc_bin = accuracy_score(y_bin, pred_bin)
        brier_bin = brier_score_loss(y_bin, p_home_not_lose)
        try:
            auc_bin = roc_auc_score(y_bin, p_home_not_lose)
        except ValueError:
            auc_bin = 0.5

        print(f"\n  3-way accuracy:     {acc3:.1%}")
        print(f"  3-way Brier:        {brier3:.4f}")
        print(f"  Binary accuracy:    {acc_bin:.1%}  (home doesn't lose)")
        print(f"  Binary Brier:       {brier_bin:.4f}")
        print(f"  Binary AUC:         {auc_bin:.4f}")

        # Outcome distribution
        print(f"\n  Predicted distribution:")
        print(f"    Home win: {(y_pred == 2).mean():.0%}  (actual: {(y_test == 2).mean():.0%})")
        print(f"    Draw:     {(y_pred == 1).mean():.0%}  (actual: {(y_test == 1).mean():.0%})")
        print(f"    Away win: {(y_pred == 0).mean():.0%}  (actual: {(y_test == 0).mean():.0%})")

        # 3-way calibration
        print(f"\n  3-way calibration:")
        for cls_name, cls_val, p_cls in [("Home", 2, p_home), ("Draw", 1, p_draw), ("Away", 0, p_away)]:
            actual_rate = (y_test == cls_val).mean()
            pred_mean = p_cls.mean()
            print(f"    {cls_name}: predicted={pred_mean:.1%}  actual={actual_rate:.1%}  gap={abs(actual_rate-pred_mean):.1%}")

        all_results[year] = {
            "acc_3way": round(acc3, 4),
            "brier_3way": round(brier3, 4),
            "acc_binary": round(acc_bin, 4),
            "brier_binary": round(brier_bin, 4),
            "auc_binary": round(auc_bin, 4),
        }

        comb_pred3.extend(y_pred.tolist())
        comb_actual3.extend(y_test.tolist())
        comb_probs_hw.extend(p_home.tolist())
        comb_probs_dr.extend(p_draw.tolist())
        comb_probs_aw.extend(p_away.tolist())
        comb_actual_bin.extend(y_bin.tolist())

    # ── Combined ──
    comb_actual3 = np.array(comb_actual3)
    comb_pred3 = np.array(comb_pred3)
    comb_actual_bin = np.array(comb_actual_bin)
    comb_hw = np.array(comb_probs_hw)
    comb_dr = np.array(comb_probs_dr)
    comb_aw = np.array(comb_probs_aw)
    comb_hnl = comb_hw + comb_dr  # home not lose

    total_acc3 = accuracy_score(comb_actual3, comb_pred3)
    total_acc_bin = accuracy_score(comb_actual_bin, (comb_hnl >= 0.5).astype(int))
    total_brier_bin = brier_score_loss(comb_actual_bin, comb_hnl)
    try:
        total_auc = roc_auc_score(comb_actual_bin, comb_hnl)
    except ValueError:
        total_auc = 0.5

    # Multi-class Brier combined
    total_brier3 = 0.0
    for i in range(len(comb_actual3)):
        for cls_val, p in [(0, comb_aw[i]), (1, comb_dr[i]), (2, comb_hw[i])]:
            indicator = 1.0 if comb_actual3[i] == cls_val else 0.0
            total_brier3 += (p - indicator) ** 2
    total_brier3 /= len(comb_actual3)

    print(f"\n{'─' * 70}")
    print(f"  V2 COMBINED (192 matches)")
    print(f"{'─' * 70}")
    print(f"    3-way accuracy:   {total_acc3:.1%}")
    print(f"    3-way Brier:      {total_brier3:.4f}")
    print(f"    Binary accuracy:  {total_acc_bin:.1%}")
    print(f"    Binary Brier:     {total_brier_bin:.4f}")
    print(f"    Binary AUC:       {total_auc:.4f}")
    print(f"    Baseline:         {comb_actual_bin.mean():.1%}")

    all_results["combined"] = {
        "acc_3way": round(total_acc3, 4),
        "brier_3way": round(total_brier3, 4),
        "acc_binary": round(total_acc_bin, 4),
        "brier_binary": round(total_brier_bin, 4),
        "auc_binary": round(total_auc, 4),
    }

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# V1 Backtest (original) — for comparison
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_v1(results_df):
    """Original binary backtest with aggressive Pi (0.5 coefficient)."""
    print("\n" + "=" * 75)
    print("  V1 BACKTEST — BINARY + AGGRESSIVE Pi (ORIGINAL)")
    print("=" * 75)

    tournaments = [
        {"year": 2014, "cutoff": "2014-06-12"},
        {"year": 2018, "cutoff": "2018-06-14"},
        {"year": 2022, "cutoff": "2022-11-20"},
    ]

    all_results = {}
    comb_preds, comb_actuals, comb_probs = [], [], []

    for t in tournaments:
        year, cutoff = t["year"], t["cutoff"]
        print(f"\n  WC {year}...", end=" ", flush=True)

        wc = _get_wc_matches(results_df, year)
        pi_cache = _compute_pi_ratings(results_df, cutoff)

        test_rows, test_yb = [], []
        for _, m in wc.iterrows():
            feat = _build_features(pi_cache, results_df,
                                   str(m["home_team"]), str(m["away_team"]), cutoff,
                                   str(m.get("neutral", "")).upper() == "TRUE")
            test_rows.append(feat)
            test_yb.append(1 if float(m["home_score"]) >= float(m["away_score"]) else 0)

        test_df = pd.DataFrame(test_rows)
        feat_cols = [c for c in test_df.columns if test_df[c].dtype in [np.float64, np.int64, float, int]]
        X_test = test_df[feat_cols].fillna(0).values
        y_test = np.array(test_yb)

        cutoff_dt = pd.Timestamp(cutoff)
        train_start = cutoff_dt - pd.DateOffset(years=4)
        train_df = results_df[(results_df["date"] >= train_start) & (results_df["date"] < cutoff_dt)]
        comp_mask = train_df["tournament"].str.lower().str.contains(
            "world cup|euro |copa america|nations league|african|asian cup|gold cup|qualification", na=False)
        train_subset = pd.concat([
            train_df[comp_mask],
            train_df[~comp_mask].sample(min(500, len(train_df[~comp_mask])), random_state=42)
            if len(train_df[~comp_mask]) > 0 else pd.DataFrame()
        ]).sort_values("date")

        train_rows, train_yb = [], []
        for _, row in train_subset.iterrows():
            feat = _build_features(pi_cache, results_df,
                                   str(row["home_team"]), str(row["away_team"]), cutoff,
                                   str(row.get("neutral", "")).upper() == "TRUE")
            train_rows.append(feat)
            train_yb.append(1 if float(row["home_score"]) >= float(row["away_score"]) else 0)

        train_feat_df = pd.DataFrame(train_rows)
        for c in feat_cols:
            if c not in train_feat_df.columns:
                train_feat_df[c] = 0.0
        X_train = train_feat_df[feat_cols].fillna(0).values
        y_train = np.array(train_yb)

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob_full = model.predict_proba(X_test)
        y_prob = y_prob_full[:, 1] if y_prob_full.shape[1] == 2 else y_prob_full[:, 1]

        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        try: auc = roc_auc_score(y_test, y_prob)
        except: auc = 0.5

        print(f"Acc={acc:.1%}  Brier={brier:.4f}  AUC={auc:.4f}")
        all_results[year] = {"accuracy": round(acc, 4), "brier": round(brier, 4), "auc": round(auc, 4)}

        comb_preds.extend(y_pred.tolist())
        comb_actuals.extend(y_test.tolist())
        comb_probs.extend(y_prob.tolist())

    ca, cp, cpr = np.array(comb_actuals), np.array(comb_preds), np.array(comb_probs)
    all_results["combined"] = {
        "accuracy": round(accuracy_score(ca, cp), 4),
        "brier": round(brier_score_loss(ca, cpr), 4),
        "auc": round(roc_auc_score(ca, cpr), 4) if len(np.unique(ca)) > 1 else 0.5,
    }
    c = all_results["combined"]
    print(f"\n  V1 Combined: Acc={c['accuracy']:.1%}  Brier={c['brier']:.4f}  AUC={c['auc']:.4f}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Market Calibration v2 — softer Pi + 3-way
# ═══════════════════════════════════════════════════════════════════════════════

def run_market_calibration_v2():
    """Compare model 3-way probs vs bookmaker odds for WC 2026."""
    print("\n\n" + "=" * 75)
    print("  MARKET CALIBRATION V2 — SOFT Pi + 3-WAY")
    print("=" * 75)

    feat_path = os.path.join(_DATA_DIR, "wc2026_feature_matrix.csv")
    if not os.path.exists(feat_path):
        print("  [SKIP] No feature matrix")
        return {}

    df = pd.read_csv(feat_path)
    needed = ["pi_delta", "odds_home_implied", "odds_draw_implied", "odds_away_implied"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [SKIP] Missing: {missing}")
        return {}

    has_odds = df["odds_home_implied"].notna() & (df["odds_home_implied"] > 0)
    print(f"\n  Matches with odds: {has_odds.sum()}")

    rows = []
    for _, row in df[has_odds].iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        pi_delta = float(row.get("pi_delta", 0))
        form_delta = float(row.get("form_win_pct_delta", 0))
        comp_delta = float(row.get("competitive_wp_delta", 0))

        # Soft Pi → 3 way (FIX 1 + FIX 3)
        strength = 1.0 / (1.0 + math.exp(-0.25 * pi_delta))
        draw_base = 0.27 * math.exp(-0.04 * pi_delta ** 2)
        draw_base = max(0.12, min(0.38, draw_base))
        remaining = 1.0 - draw_base
        p_hw = remaining * strength
        p_aw = remaining * (1.0 - strength)
        p_dr = draw_base

        # Blend 60% Pi-model + 40% form/strength for win probs
        form_hw = 0.5 + form_delta * 0.5 + comp_delta * 0.5
        form_hw = max(0.1, min(0.9, form_hw))
        blended_hw = 0.6 * p_hw + 0.4 * form_hw * (1 - p_dr)
        blended_aw = 0.6 * p_aw + 0.4 * (1 - form_hw) * (1 - p_dr)
        blended_dr = p_dr

        # Normalize
        total = blended_hw + blended_dr + blended_aw
        blended_hw /= total
        blended_dr /= total
        blended_aw /= total

        mkt_hw = float(row["odds_home_implied"])
        mkt_dr = float(row["odds_draw_implied"])
        mkt_aw = float(row["odds_away_implied"])

        # Compare home win (most actionable market)
        edge_hw = blended_hw - mkt_hw

        rows.append({
            "match": f"{home} vs {away}",
            "model_hw": round(blended_hw, 3), "model_dr": round(blended_dr, 3), "model_aw": round(blended_aw, 3),
            "mkt_hw": round(mkt_hw, 3), "mkt_dr": round(mkt_dr, 3), "mkt_aw": round(mkt_aw, 3),
            "edge_hw": round(edge_hw, 3),
            "agreement": abs(edge_hw) < 0.10,
            "pi_delta": pi_delta,
        })

    adf = pd.DataFrame(rows)
    agreement = adf["agreement"].mean()
    corr_hw = adf["model_hw"].corr(adf["mkt_hw"])
    avg_edge = adf["edge_hw"].abs().mean()

    print(f"\n  Agreement rate (home win ±10%): {agreement:.1%}")
    print(f"  Correlation (model vs market home win): {corr_hw:.3f}")
    print(f"  Average |edge|: {avg_edge:.3f}")

    # Compare overall distribution
    print(f"\n  Average probabilities:")
    print(f"    {'':20} {'Model':>8} {'Market':>8}")
    print(f"    {'Home win':<20} {adf['model_hw'].mean():>7.1%} {adf['mkt_hw'].mean():>7.1%}")
    print(f"    {'Draw':<20} {adf['model_dr'].mean():>7.1%} {adf['mkt_dr'].mean():>7.1%}")
    print(f"    {'Away win':<20} {adf['model_aw'].mean():>7.1%} {adf['mkt_aw'].mean():>7.1%}")

    # Edge distribution
    print(f"\n  Edge distribution (home win):")
    for lo, hi, label in [
        (0.10, 1.0, "Strong model > market (>10%)"),
        (0.05, 0.10, "Mild model > market (5-10%)"),
        (-0.05, 0.05, "Agreement (±5%)"),
        (-0.10, -0.05, "Mild market > model (5-10%)"),
        (-1.0, -0.10, "Strong market > model (>10%)"),
    ]:
        mask = (adf["edge_hw"] > lo) & (adf["edge_hw"] <= hi)
        if lo == -1.0: mask = adf["edge_hw"] <= hi
        if hi == 1.0: mask = adf["edge_hw"] > lo
        n = mask.sum()
        print(f"    {label:<40}: {n:2d} ({n/len(adf):.0%})")

    # Top divergences
    top = adf.reindex(adf["edge_hw"].abs().sort_values(ascending=False).index).head(8)
    print(f"\n  Top divergences:")
    print(f"    {'Match':<32} {'M-HW':>5} {'Mk-HW':>6} {'M-Dr':>5} {'Mk-Dr':>6} {'Edge':>6}")
    for _, r in top.iterrows():
        print(f"    {r['match']:<32} {r['model_hw']:>4.0%} {r['mkt_hw']:>5.0%} "
              f"{r['model_dr']:>4.0%} {r['mkt_dr']:>5.0%} {r['edge_hw']:>+5.0%}")

    return {
        "agreement": round(agreement, 4),
        "correlation_hw": round(corr_hw, 4),
        "avg_edge": round(avg_edge, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo v2 — all 48 teams + softer Pi
# ═══════════════════════════════════════════════════════════════════════════════

def _american_to_implied(odds):
    if odds > 0: return 100.0 / (odds + 100)
    else: return abs(odds) / (abs(odds) + 100)


def run_monte_carlo_v2(n_sims=10000):
    """Monte Carlo with softer Pi and all 48 teams."""
    print("\n\n" + "=" * 75)
    print(f"  MONTE CARLO V2 — ALL 48 TEAMS, SOFT Pi ({n_sims:,} sims)")
    print("=" * 75)

    try:
        pi_data = _lj("wc2026_pi_ratings_v2.json")
        winner_odds = _lj("wc2026_winner_odds.json")
        teams_list = _lj("wc2026_teams.json")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return {}

    # Build correct 12 groups of 4 from the groups file
    try:
        raw_groups = _lj("wc2026_groups.json")
    except FileNotFoundError:
        print("  [SKIP] No groups file")
        return {}

    # Deduplicate: assign each team to the first group it appears in
    team_primary = {}
    for grp in sorted(raw_groups.keys()):
        if len(grp) > 1:
            continue  # skip compound group keys
        for team in raw_groups[grp]:
            t = _norm(team)
            if t not in team_primary:
                team_primary[t] = grp

    clean_groups = defaultdict(list)
    for team, grp in team_primary.items():
        clean_groups[grp].append(team)

    # For groups with <4 teams, fill from teams_list
    all_assigned = set(team_primary.keys())
    all_teams_norm = [_norm(t) for t in teams_list]
    unassigned = [t for t in all_teams_norm if t not in all_assigned]

    # Fill short groups
    for grp in sorted(clean_groups.keys()):
        while len(clean_groups[grp]) < 4 and unassigned:
            clean_groups[grp].append(unassigned.pop(0))

    # If still unassigned teams, create new group slots
    grp_letters = "ABCDEFGHIJKL"
    for letter in grp_letters:
        if letter not in clean_groups:
            clean_groups[letter] = []
    for letter in grp_letters:
        while len(clean_groups[letter]) < 4 and unassigned:
            clean_groups[letter].append(unassigned.pop(0))

    valid_groups = {g: t for g, t in clean_groups.items() if len(t) == 4}

    all_teams = set()
    for teams in valid_groups.values():
        all_teams.update(teams)

    # Market futures
    market_futures = {}
    for entry in winner_odds:
        for bm in entry.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "outrights":
                    for o in mkt["outcomes"]:
                        t = _norm(o["name"])
                        prob = _american_to_implied(o["price"])
                        market_futures.setdefault(t, []).append(prob)
    market_futures = {t: np.mean(p) for t, p in market_futures.items()}

    def _get_pi(team):
        t = _norm(team)
        d = pi_data.get(t, {"home_pi": 0.0, "away_pi": 0.0})
        return d["home_pi"] + d["away_pi"]

    def _sim_match(h_pi, a_pi, rng, allow_draw=True):
        delta = h_pi - a_pi
        # SOFT Pi (0.25 coefficient)
        strength = 1.0 / (1.0 + math.exp(-0.25 * delta))
        dr = 0.27 * math.exp(-0.04 * delta ** 2)
        dr = max(0.10, min(0.40, dr))
        if not allow_draw:
            hw = strength
            aw = 1.0 - hw
            dr = 0.0
        else:
            rem = 1.0 - dr
            hw = rem * strength
            aw = rem * (1.0 - strength)
        r = rng.random()
        if r < hw: return "home"
        elif r < hw + dr: return "draw"
        else: return "away"

    print(f"\n  Groups: {len(valid_groups)}")
    print(f"  Teams: {len(all_teams)}")
    print(f"  Market futures: {len(market_futures)}")

    # Counters
    group_wins = defaultdict(int)
    advanced = defaultdict(int)
    qf_count = defaultdict(int)
    sf_count = defaultdict(int)
    final_count = defaultdict(int)
    champion = defaultdict(int)

    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        group_ranked = {}

        for grp, teams in sorted(valid_groups.items()):
            pts = {t: 0 for t in teams}
            gd = {t: 0.0 for t in teams}
            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    h, a = teams[i], teams[j]
                    res = _sim_match(_get_pi(h), _get_pi(a), rng)
                    if res == "home":
                        pts[h] += 3; gd[h] += 1; gd[a] -= 1
                    elif res == "draw":
                        pts[h] += 1; pts[a] += 1
                    else:
                        pts[a] += 3; gd[a] += 1; gd[h] -= 1
            ranked = sorted(teams, key=lambda t: (-pts[t], -gd[t], -_get_pi(t)))
            group_ranked[grp] = ranked
            for t in ranked[:2]:
                advanced[t] += 1
            group_wins[ranked[0]] += 1

        # Knockout: pair groups A-B, C-D, E-F, G-H, I-J, K-L
        grp_list = sorted(valid_groups.keys())
        r32 = []
        for i in range(0, len(grp_list) - 1, 2):
            g1, g2 = grp_list[i], grp_list[i+1]
            if g1 in group_ranked and g2 in group_ranked:
                r32.append((group_ranked[g1][0], group_ranked[g2][1]))
                r32.append((group_ranked[g2][0], group_ranked[g1][1]))

        # R32 → QF
        qf = []
        for h, a in r32:
            w = h if _sim_match(_get_pi(h), _get_pi(a), rng, False) == "home" else a
            qf.append(w); qf_count[w] += 1

        # QF → SF
        sf = []
        for i in range(0, len(qf) - 1, 2):
            w = qf[i] if _sim_match(_get_pi(qf[i]), _get_pi(qf[i+1]), rng, False) == "home" else qf[i+1]
            sf.append(w); sf_count[w] += 1

        # SF → Final
        fin = []
        for i in range(0, len(sf) - 1, 2):
            w = sf[i] if _sim_match(_get_pi(sf[i]), _get_pi(sf[i+1]), rng, False) == "home" else sf[i+1]
            fin.append(w); final_count[w] += 1

        if len(fin) >= 2:
            w = fin[0] if _sim_match(_get_pi(fin[0]), _get_pi(fin[1]), rng, False) == "home" else fin[1]
            champion[w] += 1

    # Results
    model_probs = {t: champion.get(t, 0) / n_sims for t in all_teams}
    sorted_teams = sorted(model_probs.items(), key=lambda x: -x[1])

    print(f"\n  {'Team':<22} {'Model':>7} {'Market':>7} {'Edge':>7} {'Adv':>5} {'QF':>5} {'SF':>5} {'Fin':>5}")
    print(f"  {'─' * 75}")

    edges = []
    for team, wp in sorted_teams[:25]:
        mkt = market_futures.get(team, 0)
        adv = advanced.get(team, 0) / n_sims
        qf = qf_count.get(team, 0) / n_sims
        sf_ = sf_count.get(team, 0) / n_sims
        fn = final_count.get(team, 0) / n_sims
        edge = wp - mkt
        edges.append({"team": team, "model": wp, "market": mkt, "edge": edge})
        print(f"  {team:<22} {wp:>6.1%} {mkt:>6.1%} {edge:>+6.1%} {adv:>4.0%} {qf:>4.0%} {sf_:>4.0%} {fn:>4.0%}")

    # Value bets
    value = sorted(edges, key=lambda x: -x["edge"])
    print(f"\n  Value bets (model > market):")
    for v in value[:8]:
        if v["edge"] > 0.005:
            print(f"    {v['team']:<22} model={v['model']:.1%}  market={v['market']:.1%}  edge={v['edge']:+.1%}")

    overvalued = sorted(edges, key=lambda x: x["edge"])
    print(f"\n  Overvalued by market:")
    for v in overvalued[:5]:
        if v["edge"] < -0.005:
            print(f"    {v['team']:<22} model={v['model']:.1%}  market={v['market']:.1%}  edge={v['edge']:+.1%}")

    return {
        "teams_simulated": len(all_teams),
        "groups": len(valid_groups),
        "champion_probs": {t: round(champion.get(t, 0)/n_sims, 4) for t in all_teams},
        "value_bets": [v for v in value[:10] if v["edge"] > 0.005],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 75)
    print("  WC 2026 PRESSURE TEST V2 — THREE FIXES")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 75)
    print("\n  Fix 1: Softer Pi coefficient (0.25 not 0.5)")
    print("  Fix 2: pi_home_prob as explicit model feature")
    print("  Fix 3: Three-way classification (home/draw/away)")

    results_path = os.path.join(_DATA_DIR, "international_results.csv")
    results_df = pd.read_csv(results_path)
    results_df["date"] = pd.to_datetime(results_df["date"])
    print(f"\n  Loaded {len(results_df)} international results")

    # Run v2 backtest
    v2_results = run_backtest_v2(results_df)

    # Market calibration v2
    cal_results = run_market_calibration_v2()

    # Monte Carlo v2
    mc_results = run_monte_carlo_v2(n_sims=10000)

    # ── Comparison ──
    print("\n\n" + "█" * 75)
    print("  COMPARISON: V1 vs V2")
    print("█" * 75)

    v1 = {"combined": {"accuracy": 0.6615, "brier": 0.2264, "auc": 0.6240}}
    v1c = v1["combined"]
    v2c = v2_results.get("combined", {})

    print(f"\n  {'Metric':<25} {'V1 (old)':>10} {'V2 (new)':>10} {'Change':>10}")
    print(f"  {'─' * 55}")
    print(f"  {'Binary Accuracy':<25} {v1c['accuracy']:>9.1%} {v2c.get('acc_binary',0):>9.1%} "
          f"{v2c.get('acc_binary',0) - v1c['accuracy']:>+9.1%}")
    print(f"  {'Binary Brier':<25} {v1c['brier']:>10.4f} {v2c.get('brier_binary',0):>10.4f} "
          f"{v2c.get('brier_binary',0) - v1c['brier']:>+10.4f}")
    print(f"  {'Binary AUC':<25} {v1c['auc']:>10.4f} {v2c.get('auc_binary',0):>10.4f} "
          f"{v2c.get('auc_binary',0) - v1c['auc']:>+10.4f}")
    print(f"  {'3-way Accuracy':<25} {'—':>10} {v2c.get('acc_3way',0):>9.1%}")
    print(f"  {'3-way Brier':<25} {'—':>10} {v2c.get('brier_3way',0):>10.4f}")

    if cal_results:
        print(f"\n  Market Calibration:")
        print(f"    Agreement (v1): 31.9%")
        print(f"    Agreement (v2): {cal_results.get('agreement', 0):.1%}")
        print(f"    Correlation (v1): 0.378")
        print(f"    Correlation (v2): {cal_results.get('correlation_hw', 0):.3f}")

    if mc_results:
        cp = mc_results.get("champion_probs", {})
        top3 = sorted(cp.items(), key=lambda x: -x[1])[:3]
        print(f"\n  Monte Carlo ({mc_results.get('teams_simulated', 0)} teams, "
              f"{mc_results.get('groups', 0)} groups):")
        for t, p in top3:
            mkt = _american_to_implied(450) if t == "Spain" else 0  # approximate
            print(f"    {t}: {p:.1%}")

    # Save
    out = os.path.join(_DATA_DIR, "wc2026_pressure_test_v2_results.json")
    with open(out, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "v1_baseline": v1,
            "v2_backtest": v2_results,
            "v2_calibration": cal_results,
            "v2_monte_carlo": mc_results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {out}")
    print()


if __name__ == "__main__":
    main()
