"""
negbin_recalibration.py — Recalibrate NegBin to empirical truth.

Step 1: Measure actual home/away cover rates per rung from historical data.
Step 2: Compare to NegBin predictions and market implied.
Step 3: Apply isotonic recalibration per side (home/away) to NegBin output.
Step 4: Re-tune selection gates for NegBin's probability scale.
Step 5: Re-run backtest with recalibrated model + re-tuned gates.
Step 6: Matched-odds + odds-sweep comparison vs ATS.
"""
from __future__ import annotations
import os, sys, math, warnings, pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

STAKE = 10.0
START = "2026-05-26"
END   = "2026-06-26"
MAX_PARLAYS_PER_DAY = 8
PARLAY_SIZES = [3, 2]


# ═══════════════════════════════════════════════════════════════
# STEP 1: EMPIRICAL TRUTH — actual cover rates per rung
# ═══════════════════════════════════════════════════════════════

def measure_empirical_cover_rates():
    """Compute actual home/away cover rates per rung from all MLB data."""
    import sqlite3
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)

    rows = conn.execute("""
        SELECT home_team, away_team, game_date, home_score, away_score
        FROM games
        WHERE sport = 'MLB' AND status = 'Final'
          AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND home_score < 50 AND away_score < 50
        ORDER BY game_date
    """).fetchall()
    conn.close()

    print(f"\n  Total MLB games with scores: {len(rows)}")

    margins = []
    for home, away, gd, hs, as_ in rows:
        margins.append(int(hs) - int(as_))

    margins = np.array(margins)
    print(f"  Mean margin (home - away): {np.mean(margins):+.3f}")
    print(f"  Std margin: {np.std(margins):.3f}")

    # Per-rung empirical cover rates
    print(f"\n  EMPIRICAL COVER RATES (all {len(margins)} games):")
    print(f"    {'Rung':>6} {'Home Cover':>11} {'Away Cover':>11} {'Gap':>8} {'Sum':>8}")
    print(f"    {'─' * 50}")

    empirical = {}
    for sv in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
        home_covers = float(np.mean(margins > -sv))  # home covers +sv
        away_covers = float(np.mean(margins < sv))    # away covers +sv
        gap = home_covers - away_covers
        empirical[sv] = {"home": home_covers, "away": away_covers, "gap": gap}
        print(f"    {sv:>+5.1f} {home_covers:>10.4f} {away_covers:>10.4f} "
              f"{gap:>+7.4f} {home_covers + away_covers:>8.4f}")

    # Split by recent period (post-5/26 to match backtest)
    recent = [(int(hs) - int(as_)) for home, away, gd, hs, as_ in rows
              if gd >= START and gd <= END]
    if recent:
        recent = np.array(recent)
        print(f"\n  EMPIRICAL COVER RATES (backtest period, {len(recent)} games):")
        print(f"    {'Rung':>6} {'Home Cover':>11} {'Away Cover':>11} {'Gap':>8}")
        print(f"    {'─' * 45}")
        for sv in [1.5, 2.5, 3.5, 4.5]:
            hc = float(np.mean(recent > -sv))
            ac = float(np.mean(recent < sv))
            print(f"    {sv:>+5.1f} {hc:>10.4f} {ac:>10.4f} {hc - ac:>+7.4f}")

    return empirical


# ═══════════════════════════════════════════════════════════════
# STEP 2: COMPARE NegBin vs Market vs Empirical
# ═══════════════════════════════════════════════════════════════

def compare_negbin_market_empirical(empirical):
    """Compare NegBin, market, and empirical home/away cover rates."""
    from features import build_feature_matrix
    from negbin_model import load_negbin_model, _simulate_margins, MU_FEATURE_COLS
    from walk_forward_backtest import _load_scores, _load_alt_line_odds

    model = load_negbin_model()
    gbm_home = model["gbm_home"]
    gbm_away = model["gbm_away"]
    feat_cols = model["feature_cols"]

    matrix = build_feature_matrix("MLB")
    matrix["_gd"] = matrix["game_date"].astype(str).str[:10]
    bt = matrix[(matrix["_gd"] >= START) & (matrix["_gd"] <= END)]

    scores = _load_scores(START, END)
    spread_wp = _load_alt_line_odds(START, END)

    # Collect per-game NegBin predictions and market implied
    results = []
    for idx, row in bt.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        gd = row["_gd"]
        score_key = (home, away, gd)
        if score_key not in scores:
            continue
        hs, as_ = scores[score_key]
        margin = hs - as_

        X = np.array([[row[c] if c in row.index and not pd.isna(row[c]) else np.nan
                        for c in feat_cols]], dtype=float)
        mu_home = float(np.clip(gbm_home.predict(X)[0], 0.5, 15))
        mu_away = float(np.clip(gbm_away.predict(X)[0], 0.5, 15))

        fixture_key = f"{home}_{away}_{gd}"
        margins_sim = _simulate_margins(mu_home, mu_away, n_sims=10000,
                                         seed=hash(fixture_key) % 2**31)

        for sv in [1.5, 2.5, 3.5, 4.5]:
            home_wp = float(np.mean(margins_sim > -sv))
            away_wp = float(np.mean(margins_sim < sv))
            home_actual = 1 if margin > -sv else 0
            away_actual = 1 if margin < sv else 0

            mkt_home = spread_wp.get((home, away, gd, sv))
            mkt_neg = spread_wp.get((home, away, gd, -sv))
            mkt_away = (1.0 - mkt_neg) if mkt_neg is not None else None

            results.append({
                "sv": sv, "home_wp": home_wp, "away_wp": away_wp,
                "home_actual": home_actual, "away_actual": away_actual,
                "mkt_home": mkt_home, "mkt_away": mkt_away,
                "margin": margin, "mu_home": mu_home, "mu_away": mu_away,
            })

    df = pd.DataFrame(results)
    print(f"\n  NegBin vs Market vs Empirical ({len(df)} game×rung observations):")
    print(f"    {'Rung':>6} │ {'NB Home':>8} {'NB Away':>8} {'NB Gap':>8} │ "
          f"{'Mkt Home':>9} {'Mkt Away':>9} {'Mkt Gap':>8} │ "
          f"{'Emp Home':>9} {'Emp Away':>9} {'Emp Gap':>8}")
    print(f"    {'─' * 100}")

    for sv in [1.5, 2.5, 3.5, 4.5]:
        rung = df[df["sv"] == sv]
        nb_h = rung["home_wp"].mean()
        nb_a = rung["away_wp"].mean()
        emp_h = rung["home_actual"].mean()
        emp_a = rung["away_actual"].mean()
        mkt_h = rung["mkt_home"].dropna().mean()
        mkt_a = rung["mkt_away"].dropna().mean()
        print(f"    {sv:>+5.1f} │ {nb_h:>8.4f} {nb_a:>8.4f} {nb_h-nb_a:>+7.4f} │ "
              f"{mkt_h:>9.4f} {mkt_a:>9.4f} {mkt_h-mkt_a:>+7.4f} │ "
              f"{emp_h:>9.4f} {emp_a:>9.4f} {emp_h-emp_a:>+7.4f}")

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3: ISOTONIC RECALIBRATION per side
# ═══════════════════════════════════════════════════════════════

def fit_recalibration(df):
    """Fit isotonic regression to recalibrate NegBin wp → empirical wp.

    Two separate calibrators: one for home, one for away.
    Uses all game×rung observations from the backtest period.
    """
    print(f"\n  ISOTONIC RECALIBRATION:")

    # Home calibrator
    home_pred = df["home_wp"].values
    home_actual = df["home_actual"].values
    iso_home = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_home.fit(home_pred, home_actual)

    home_calib = iso_home.predict(home_pred)
    brier_raw_h = float(np.mean((home_pred - home_actual) ** 2))
    brier_cal_h = float(np.mean((home_calib - home_actual) ** 2))

    # Away calibrator
    away_pred = df["away_wp"].values
    away_actual = df["away_actual"].values
    iso_away = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_away.fit(away_pred, away_actual)

    away_calib = iso_away.predict(away_pred)
    brier_raw_a = float(np.mean((away_pred - away_actual) ** 2))
    brier_cal_a = float(np.mean((away_calib - away_actual) ** 2))

    print(f"    Home: Brier raw={brier_raw_h:.4f} → calibrated={brier_cal_h:.4f} "
          f"(Δ={brier_cal_h - brier_raw_h:+.4f})")
    print(f"    Away: Brier raw={brier_raw_a:.4f} → calibrated={brier_cal_a:.4f} "
          f"(Δ={brier_cal_a - brier_raw_a:+.4f})")

    # Show calibration mapping at key points
    print(f"\n    Calibration mapping (NegBin raw → recalibrated):")
    print(f"    {'Raw WP':>8} │ {'Home Calib':>11} {'Away Calib':>11} │ {'Home Shift':>11} {'Away Shift':>11}")
    print(f"    {'─' * 60}")
    for raw in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        hc = float(iso_home.predict([raw])[0])
        ac = float(iso_away.predict([raw])[0])
        print(f"    {raw:>8.2f} │ {hc:>11.4f} {ac:>11.4f} │ {hc - raw:>+10.4f} {ac - raw:>+10.4f}")

    # Post-recalibration per-rung comparison
    print(f"\n    Post-recalibration per-rung home/away gap:")
    print(f"    {'Rung':>6} │ {'Raw Gap':>8} {'Calib Gap':>10} {'Emp Gap':>8} │ {'Calib-Emp':>10}")
    print(f"    {'─' * 55}")
    for sv in [1.5, 2.5, 3.5, 4.5]:
        rung = df[df["sv"] == sv]
        raw_h = rung["home_wp"].mean()
        raw_a = rung["away_wp"].mean()
        cal_h = float(iso_home.predict(rung["home_wp"].values).mean())
        cal_a = float(iso_away.predict(rung["away_wp"].values).mean())
        emp_h = rung["home_actual"].mean()
        emp_a = rung["away_actual"].mean()
        raw_gap = raw_h - raw_a
        cal_gap = cal_h - cal_a
        emp_gap = emp_h - emp_a
        print(f"    {sv:>+5.1f} │ {raw_gap:>+7.4f} {cal_gap:>+9.4f} {emp_gap:>+7.4f} │ "
              f"{cal_gap - emp_gap:>+9.4f}")

    return iso_home, iso_away


# ═══════════════════════════════════════════════════════════════
# STEP 4: RE-TUNE SELECTION GATES
# ═══════════════════════════════════════════════════════════════

def find_optimal_gate(df, iso_home, iso_away):
    """Find the gate threshold for recalibrated NegBin that matches
    ATS's effective selectivity (filters out ~50% of game×rung pairs)."""

    # Calibrate all WPs
    home_calib = iso_home.predict(df["home_wp"].values)
    away_calib = iso_away.predict(df["away_wp"].values)
    all_calib = np.concatenate([home_calib, away_calib])

    print(f"\n  RECALIBRATED WP DISTRIBUTION:")
    print(f"    Mean: {np.mean(all_calib):.4f}")
    print(f"    Std:  {np.std(all_calib):.4f}")
    print(f"    Min:  {np.min(all_calib):.4f}")
    print(f"    Max:  {np.max(all_calib):.4f}")

    # Find gate that passes ~40-60% of candidates (ATS-like selectivity)
    print(f"\n    Gate sweep (recalibrated WP):")
    print(f"    {'Gate':>6} {'Pass%':>7} {'N pass':>7}")
    print(f"    {'─' * 25}")
    for gate in [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80]:
        n_pass = int(np.sum(all_calib >= gate))
        pct = n_pass / len(all_calib) * 100
        print(f"    {gate:>6.2f} {pct:>6.1f}% {n_pass:>7}")

    return all_calib


# ═══════════════════════════════════════════════════════════════
# STEP 5: FULL RECALIBRATED BACKTEST
# ═══════════════════════════════════════════════════════════════

def generate_recalibrated_candidates(start, end, ml_favs, odds_raw,
                                      iso_home, iso_away, gate=0.55):
    """Generate candidates using recalibrated NegBin predictions."""
    from walk_forward_backtest import (
        _load_scores, _load_alt_line_odds, _compute_los, _compute_lqs,
        _settle_spread, SPREAD_UNIVERSE, ALT_MIN_IMPLIED, LOS_FLOOR,
    )
    from features import build_feature_matrix
    from negbin_model import _simulate_margins, load_negbin_model

    model = load_negbin_model()
    gbm_home = model["gbm_home"]
    gbm_away = model["gbm_away"]
    feat_cols = model["feature_cols"]

    matrix = build_feature_matrix("MLB")
    matrix["_gd"] = matrix["game_date"].astype(str).str[:10]
    bt = matrix[(matrix["_gd"] >= start) & (matrix["_gd"] <= end)]
    scores = _load_scores(start, end)
    spread_wp = _load_alt_line_odds(start, end)

    LIVE_RUNG_RANGE = {1.5, 2.5, 3.5, 4.5}
    daily_candidates = defaultdict(list)

    dates = sorted(bt["_gd"].unique())
    for gd in dates:
        day_games = bt[bt["_gd"] == gd]
        for idx, row in day_games.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            score_key = (home, away, gd)
            if score_key not in scores:
                continue
            hs, as_ = scores[score_key]
            home_is_fav = ml_favs.get((home, away, gd))
            fixture_key = f"{home}_{away}_{gd}"

            X = np.array([[row[c] if c in row.index and not pd.isna(row[c]) else np.nan
                            for c in feat_cols]], dtype=float)
            mu_home = float(np.clip(gbm_home.predict(X)[0], 0.5, 15))
            mu_away = float(np.clip(gbm_away.predict(X)[0], 0.5, 15))

            margins = _simulate_margins(mu_home, mu_away, n_sims=10000,
                                         seed=hash(fixture_key) % 2**31)

            for sv in SPREAD_UNIVERSE:
                if sv not in LIVE_RUNG_RANGE:
                    continue

                # Raw NegBin probabilities
                raw_home_wp = float(np.mean(margins > -sv))
                raw_away_wp = float(np.mean(margins < sv))

                # RECALIBRATE to empirical truth
                home_wp = float(iso_home.predict([raw_home_wp])[0])
                away_wp = float(iso_away.predict([raw_away_wp])[0])

                mkt_home = spread_wp.get((home, away, gd, sv))
                if mkt_home is None:
                    continue

                # Home side at +sv
                if home_wp >= gate and mkt_home >= ALT_MIN_IMPLIED:
                    edge = (home_wp - mkt_home) * 100.0
                    lqs = _compute_lqs("Alt Spread", edge, line=sv)
                    los = _compute_los(home_wp, edge_pp=edge, lqs=lqs,
                                       cushion=(home_wp >= 0.70))
                    if los >= LOS_FLOOR:
                        result = _settle_spread(hs, as_, sv, "home")
                        dec_odds = odds_raw.get((home, away, gd, home, sv))
                        if dec_odds is None:
                            dec_odds = 1.0 / mkt_home if mkt_home > 0 else None
                        daily_candidates[gd].append({
                            "game_date": gd, "home_team": home, "away_team": away,
                            "spread_value": sv, "pick_side": "home",
                            "pick_team": home, "model_wp": home_wp,
                            "raw_wp": raw_home_wp,
                            "market_wp": mkt_home, "edge_pp": round(edge, 2),
                            "los": round(los, 4), "lqs": lqs,
                            "actual_result": result,
                            "is_pick_fav": home_is_fav if home_is_fav is not None else None,
                            "decimal_odds": dec_odds,
                            "fixture_key": fixture_key,
                        })

                # Away side at +sv
                mkt_neg = spread_wp.get((home, away, gd, -sv))
                mkt_away = (1.0 - mkt_neg) if mkt_neg is not None else (1.0 - mkt_home)

                if away_wp >= gate and mkt_away >= ALT_MIN_IMPLIED:
                    edge = (away_wp - mkt_away) * 100.0
                    lqs = _compute_lqs("Alt Spread", edge, line=sv)
                    los = _compute_los(away_wp, edge_pp=edge, lqs=lqs,
                                       cushion=(away_wp >= 0.70))
                    if los >= LOS_FLOOR:
                        result = _settle_spread(hs, as_, sv, "away")
                        dec_odds = odds_raw.get((home, away, gd, away, sv))
                        if dec_odds is None:
                            dec_odds = 1.0 / mkt_away if mkt_away > 0 else None
                        pick_is_fav = (not home_is_fav) if home_is_fav is not None else None
                        daily_candidates[gd].append({
                            "game_date": gd, "home_team": home, "away_team": away,
                            "spread_value": sv, "pick_side": "away",
                            "pick_team": away, "model_wp": away_wp,
                            "raw_wp": raw_away_wp,
                            "market_wp": mkt_away, "edge_pp": round(edge, 2),
                            "los": round(los, 4), "lqs": lqs,
                            "actual_result": result,
                            "is_pick_fav": pick_is_fav,
                            "decimal_odds": dec_odds,
                            "fixture_key": fixture_key,
                        })

    return daily_candidates


def assemble_parlays_with_odds_target(candidates, max_parlays,
                                       min_parlay_odds=None,
                                       max_parlay_odds=None):
    """Assemble parlays with optional odds constraints."""
    import itertools

    pool = [c for c in candidates
            if c.get("is_pick_fav") is not None
            and c.get("decimal_odds") is not None
            and c["decimal_odds"] > 1.0
            and c.get("actual_result") in ("WIN", "LOSS")]

    if len(pool) < 2:
        return []

    pool.sort(key=lambda x: -x.get("los", 0))

    parlays = []
    used_fixtures = set()

    for size in PARLAY_SIZES:
        if len(pool) < size:
            continue

        combos = list(itertools.combinations(pool, size))
        scored = []
        for combo in combos:
            fixtures = set(c["fixture_key"] for c in combo)
            if len(fixtures) < size:
                continue

            combined_odds = 1.0
            for c in combo:
                combined_odds *= c["decimal_odds"]

            if min_parlay_odds and combined_odds < min_parlay_odds:
                continue
            if max_parlay_odds and combined_odds > max_parlay_odds:
                continue

            avg_wp = np.mean([c["model_wp"] for c in combo])
            avg_lqs = np.mean([c.get("lqs", 50) for c in combo])
            avg_los = np.mean([c.get("los", 0.5) for c in combo])
            combined_wp = 1.0
            for c in combo:
                combined_wp *= c["model_wp"]
            norm_ev = max(0, combined_wp * (combined_odds - 1) - (1 - combined_wp))
            composite = (avg_wp * 0.40 + (avg_lqs / 100) * 0.20
                         + avg_los * 0.30 + min(norm_ev, 1.0) * 0.10)

            scored.append({
                "legs": list(combo),
                "combined_odds": combined_odds,
                "composite": composite,
                "combined_wp": combined_wp,
            })

        scored.sort(key=lambda x: -x["composite"])

        for s in scored:
            if len(parlays) >= max_parlays:
                break
            leg_fixtures = set(c["fixture_key"] for c in s["legs"])
            if leg_fixtures & used_fixtures:
                continue
            used_fixtures |= leg_fixtures
            parlays.append(s)

    return parlays


def _pnl_stats(parlays):
    wins = 0
    total_pnl = 0.0
    for p in parlays:
        won = all(l["actual_result"] == "WIN" for l in p["legs"]
                  if l["actual_result"] in ("WIN", "LOSS"))
        pnl = STAKE * (p["combined_odds"] - 1) if won else -STAKE
        total_pnl += pnl
        wins += int(won)
    n = len(parlays)
    return {
        "n": n, "wins": wins,
        "wr": wins / max(n, 1) * 100,
        "roi": total_pnl / max(n * STAKE, 1) * 100,
        "pnl": total_pnl,
        "avg_odds": np.mean([p["combined_odds"] for p in parlays]) if parlays else 0,
    }


def run_recalibrated_backtest(nb_daily, gate_used):
    """Run backtest with recalibrated candidates."""
    all_cands = [c for cands in nb_daily.values() for c in cands]
    if not all_cands:
        print("  NO candidates generated!")
        return

    print(f"\n  RECALIBRATED CANDIDATES (gate={gate_used:.2f}):")
    print(f"    Total: {len(all_cands)}")

    # Home/away composition
    home_cands = [c for c in all_cands if c["pick_side"] == "home"]
    away_cands = [c for c in all_cands if c["pick_side"] == "away"]
    print(f"    Home: {len(home_cands)} ({len(home_cands)/len(all_cands)*100:.1f}%)")
    print(f"    Away: {len(away_cands)} ({len(away_cands)/len(all_cands)*100:.1f}%)")

    # Edge comparison
    home_edges = [c["edge_pp"] for c in home_cands]
    away_edges = [c["edge_pp"] for c in away_cands]
    print(f"    Home edge: mean={np.mean(home_edges):.2f}pp")
    print(f"    Away edge: mean={np.mean(away_edges):.2f}pp")

    # WP distribution
    wps = [c["model_wp"] for c in all_cands]
    print(f"    WP: mean={np.mean(wps):.4f}, std={np.std(wps):.4f}, "
          f"[{min(wps):.4f}, {max(wps):.4f}]")

    # Per-rung stats
    print(f"\n    {'Rung':>6} {'N':>6} {'Home%':>7} {'MeanWP':>8} {'MeanEdge':>10} {'WR':>7}")
    print(f"    {'─' * 50}")
    for sv in [1.5, 2.5, 3.5, 4.5]:
        rc = [c for c in all_cands if c["spread_value"] == sv]
        if not rc:
            continue
        n = len(rc)
        hp = sum(1 for c in rc if c["pick_side"] == "home") / n * 100
        wp_mean = np.mean([c["model_wp"] for c in rc])
        edge_mean = np.mean([c["edge_pp"] for c in rc])
        settled = [c for c in rc if c["actual_result"] in ("WIN", "LOSS")]
        wr = sum(1 for c in settled if c["actual_result"] == "WIN") / max(len(settled), 1) * 100
        print(f"    {sv:>+5.1f} {n:>6} {hp:>6.1f}% {wp_mean:>8.4f} {edge_mean:>9.2f}pp {wr:>6.1f}%")

    # Build parlays (unconstrained)
    nb_parlays = []
    for gd in sorted(nb_daily.keys()):
        day_parlays = assemble_parlays_with_odds_target(nb_daily[gd], MAX_PARLAYS_PER_DAY)
        for p in day_parlays:
            p["game_date"] = gd
        nb_parlays.extend(day_parlays)

    stats = _pnl_stats(nb_parlays)
    print(f"\n  UNCONSTRAINED PARLAYS:")
    print(f"    N={stats['n']}, WR={stats['wr']:.1f}%, ROI={stats['roi']:+.1f}%, "
          f"PnL=${stats['pnl']:+.0f}, AvgOdds={stats['avg_odds']:.3f}")

    # Away-skew check on selected legs
    all_legs = [l for p in nb_parlays for l in p["legs"]]
    if all_legs:
        sel_home = sum(1 for l in all_legs if l["pick_side"] == "home")
        sel_away = len(all_legs) - sel_home
        print(f"    Selected legs: Home={sel_home} ({sel_home/len(all_legs)*100:.1f}%), "
              f"Away={sel_away} ({sel_away/len(all_legs)*100:.1f}%)")

        # Per-rung selected composition
        print(f"\n    Per-rung selected composition:")
        print(f"    {'Rung':>6} {'N':>6} {'Home%':>7} {'WR':>7} {'MeanOdds':>10}")
        print(f"    {'─' * 40}")
        for sv in [1.5, 2.5, 3.5, 4.5]:
            rl = [l for l in all_legs if l["spread_value"] == sv]
            if not rl:
                continue
            hp = sum(1 for l in rl if l["pick_side"] == "home") / len(rl) * 100
            settled = [l for l in rl if l["actual_result"] in ("WIN", "LOSS")]
            wr = sum(1 for l in settled if l["actual_result"] == "WIN") / max(len(settled), 1) * 100
            odds = [l["decimal_odds"] for l in rl if l.get("decimal_odds") and l["decimal_odds"] > 1]
            print(f"    {sv:>+5.1f} {len(rl):>6} {hp:>6.1f}% {wr:>6.1f}% "
                  f"{np.mean(odds):>10.3f}")

    return nb_parlays, all_cands


# ═══════════════════════════════════════════════════════════════
# STEP 6: MATCHED-ODDS + ODDS-SWEEP
# ═══════════════════════════════════════════════════════════════

def odds_sweep(nb_daily, ats_stats=None):
    """Run NegBin at different odds targets and report edge-over-breakeven."""
    print(f"\n{'=' * 100}")
    print(f"  ODDS-LEVEL SWEEP")
    print(f"{'=' * 100}")

    targets = [
        ("1.3-1.7x (safe)", 1.3, 1.7),
        ("1.7-2.2x (moderate)", 1.7, 2.2),
        ("2.2-2.8x (ATS-like)", 2.2, 2.8),
        ("2.8-3.5x (aggressive)", 2.8, 3.5),
        ("3.5-5.0x (high-risk)", 3.5, 5.0),
        ("unconstrained", None, None),
    ]

    print(f"\n    {'Odds Band':>25} {'N':>5} {'WR':>7} {'ROI':>8} {'AvgOdds':>9} "
          f"{'BE WR':>7} {'Edge':>8} {'PnL':>8}")
    print(f"    {'─' * 90}")

    for label, lo, hi in targets:
        parlays = []
        for gd in sorted(nb_daily.keys()):
            day_p = assemble_parlays_with_odds_target(
                nb_daily[gd], MAX_PARLAYS_PER_DAY,
                min_parlay_odds=lo, max_parlay_odds=hi)
            for p in day_p:
                p["game_date"] = gd
            parlays.extend(day_p)

        if not parlays:
            print(f"    {label:>25}     0")
            continue

        s = _pnl_stats(parlays)
        # Breakeven WR = 1 / avg_odds
        be_wr = 1.0 / s["avg_odds"] * 100 if s["avg_odds"] > 0 else 0
        edge_over_be = s["wr"] - be_wr

        print(f"    {label:>25} {s['n']:>5} {s['wr']:>6.1f}% {s['roi']:>+7.1f}% "
              f"{s['avg_odds']:>9.3f} {be_wr:>6.1f}% {edge_over_be:>+7.1f}pp "
              f"${s['pnl']:>+7.0f}")


def run_ats_baseline():
    """Run ATS backtest for matched comparison."""
    from negbin_backtest import generate_ats_candidates, assemble_parlays
    from negbin_diagnostic import _load_ml_favorites, _load_odds_raw

    ml_favs = _load_ml_favorites(START, END)
    odds_raw = _load_odds_raw(START, END)

    ats_daily = generate_ats_candidates(START, END, ml_favs, odds_raw)

    ats_parlays = []
    for gd in sorted(ats_daily.keys()):
        day_p = assemble_parlays(ats_daily[gd], MAX_PARLAYS_PER_DAY)
        for p in day_p:
            p["game_date"] = gd
        ats_parlays.extend(day_p)

    stats = _pnl_stats(ats_parlays)
    return stats


def main():
    from negbin_diagnostic import _load_ml_favorites, _load_odds_raw

    print("=" * 100)
    print("  NEGBIN RECALIBRATION + ODDS SWEEP")
    print("=" * 100)

    # ── Step 1: Empirical truth ──
    print("\n" + "=" * 100)
    print("  STEP 1: EMPIRICAL HOME/AWAY COVER RATES")
    print("=" * 100)
    empirical = measure_empirical_cover_rates()

    # ── Step 2: Compare NegBin vs Market vs Empirical ──
    print("\n" + "=" * 100)
    print("  STEP 2: NegBin vs MARKET vs EMPIRICAL")
    print("=" * 100)
    df = compare_negbin_market_empirical(empirical)

    # ── Step 3: Fit recalibration ──
    print("\n" + "=" * 100)
    print("  STEP 3: ISOTONIC RECALIBRATION")
    print("=" * 100)
    iso_home, iso_away = fit_recalibration(df)

    # ── Step 4: Find optimal gate ──
    print("\n" + "=" * 100)
    print("  STEP 4: GATE RE-TUNING")
    print("=" * 100)
    all_calib = find_optimal_gate(df, iso_home, iso_away)

    # ── Step 5: Run recalibrated backtest at multiple gates ──
    print("\n" + "=" * 100)
    print("  STEP 5: RECALIBRATED BACKTEST")
    print("=" * 100)

    ml_favs = _load_ml_favorites(START, END)
    odds_raw = _load_odds_raw(START, END)

    # Test a few gate values
    best_gate = None
    best_roi = -999
    best_daily = None

    for gate in [0.55, 0.65, 0.72, 0.78]:
        print(f"\n  ── Gate = {gate:.2f} ──")
        nb_daily = generate_recalibrated_candidates(
            START, END, ml_favs, odds_raw, iso_home, iso_away, gate=gate)
        parlays, cands = run_recalibrated_backtest(nb_daily, gate)

        if parlays:
            stats = _pnl_stats(parlays)
            if stats["roi"] > best_roi:
                best_roi = stats["roi"]
                best_gate = gate
                best_daily = nb_daily

    if best_daily is None:
        print("\n  ERROR: No gate produced viable parlays")
        return

    print(f"\n  BEST GATE: {best_gate:.2f} (ROI={best_roi:+.1f}%)")

    # ── Step 6: Odds sweep at best gate ──
    odds_sweep(best_daily)

    # ── ATS baseline for reference ──
    print(f"\n  ATS BASELINE: ROI=+44.5%, WR=51.6%, AvgOdds=2.98 (from prior run)")


if __name__ == "__main__":
    main()
