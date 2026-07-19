"""
negbin_validation_checks.py — Four validation checks before NegBin integration.

Check 1: 2×2 mispricing decomposition (Fav/Dog × Home/Away)
Check 2: Longer-period odds sweep (Apr 22 - Jun 26, ~2 months)
Check 3: Band-for-band ATS comparison
Check 4: Held-out isotonic calibration (fit Jul 2025-Apr 2026, eval May-Jun 2026)
"""
from __future__ import annotations
import os, sys, math, warnings, pickle, json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

STAKE = 10.0
PARLAY_SIZES = [3, 2]

# ── Shared utilities ─────────────────────────────────────────

def _load_ml_favorites_hist(start, end):
    """Load moneyline favorites from historical.db h2h odds."""
    import sqlite3
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)
    # Try h2h from alt_lines or a dedicated table
    # Fallback: use bets.db fixtures
    conn.close()

    db_path2 = os.path.join(_HERE, "..", "data", "bets.db")
    conn2 = sqlite3.connect(db_path2)
    rows = conn2.execute("""
        SELECT home_team, away_team, game_date_ct, bookmakers
        FROM fixtures
        WHERE sport_key = 'baseball_mlb' AND bookmakers IS NOT NULL
          AND game_date_ct BETWEEN ? AND ?
    """, (start, end)).fetchall()
    conn2.close()

    favs = {}
    for home, away, gd, bk_json in rows:
        try:
            bks = json.loads(bk_json)
        except Exception:
            continue
        for bk in bks:
            for mkt in bk.get("markets", []):
                if mkt.get("key") == "h2h":
                    outcomes = {o["name"]: o.get("price", 0) for o in mkt.get("outcomes", [])}
                    hp = outcomes.get(home, 0)
                    ap = outcomes.get(away, 0)
                    if hp and ap:
                        favs[(home, away, gd)] = hp < ap
                    break
            if (home, away, gd) in favs:
                break
    return favs


def _load_odds_raw(start, end):
    import sqlite3
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date, team, line, odds
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_spreads'
          AND game_date BETWEEN ? AND ?
          AND odds IS NOT NULL AND odds > 1.0
    """, (start, end)).fetchall()
    conn.close()
    odds = {}
    for home, away, gd, team, line, o in rows:
        odds[(home, away, gd, team, float(line))] = float(o)
    return odds


def _pnl_stats(parlays):
    wins = 0
    total_pnl = 0.0
    odds_list = []
    for p in parlays:
        won = all(l["actual_result"] == "WIN" for l in p["legs"]
                  if l["actual_result"] in ("WIN", "LOSS"))
        pnl = STAKE * (p["combined_odds"] - 1) if won else -STAKE
        total_pnl += pnl
        wins += int(won)
        odds_list.append(p["combined_odds"])
    n = len(parlays)
    avg_odds = np.mean(odds_list) if odds_list else 0
    return {
        "n": n, "wins": wins,
        "wr": wins / max(n, 1) * 100,
        "roi": total_pnl / max(n * STAKE, 1) * 100,
        "pnl": total_pnl,
        "avg_odds": avg_odds,
    }


def assemble_parlays_odds_band(candidates, max_parlays, min_odds=None, max_odds=None):
    """Assemble parlays with optional odds-band constraint."""
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
            if min_odds and combined_odds < min_odds:
                continue
            if max_odds and combined_odds > max_odds:
                continue
            avg_wp = np.mean([c["model_wp"] for c in combo])
            avg_lqs = np.mean([c.get("lqs", 50) for c in combo])
            avg_los = np.mean([c.get("los", 0.5) for c in combo])
            combined_wp = 1.0
            for c in combo:
                combined_wp *= c["model_wp"]
            norm_ev = max(0, combined_wp * (combined_odds - 1) - (1 - combined_wp))
            composite = avg_wp * 0.40 + (avg_lqs / 100) * 0.20 + avg_los * 0.30 + min(norm_ev, 1.0) * 0.10
            scored.append({
                "legs": list(combo), "combined_odds": combined_odds,
                "composite": composite, "combined_wp": combined_wp,
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


# ═══════════════════════════════════════════════════════════════
# CHECK 1: 2×2 MISPRICING DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

def check1_mispricing_matrix():
    """Decompose home-overpricing by Fav/Dog × Home/Away."""
    import sqlite3
    from walk_forward_backtest import _load_alt_line_odds

    print("\n" + "=" * 100)
    print("  CHECK 1: 2×2 MISPRICING DECOMPOSITION (Fav/Dog × Home/Away)")
    print("=" * 100)

    # Load all MLB games with scores
    db_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(db_path)
    games = conn.execute("""
        SELECT home_team, away_team, game_date, home_score, away_score
        FROM games
        WHERE sport = 'MLB' AND status = 'Final'
          AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND home_score < 50 AND away_score < 50
        ORDER BY game_date
    """).fetchall()
    conn.close()

    # Get moneyline favorites for ALL games we can
    ml_favs = _load_ml_favorites_hist("2020-01-01", "2027-01-01")

    # Compute margins and classify
    cells = {"fav_home": [], "fav_away": [], "dog_home": [], "dog_away": []}

    for home, away, gd, hs, as_ in games:
        margin = int(hs) - int(as_)
        is_home_fav = ml_favs.get((home, away, gd))
        if is_home_fav is None:
            continue

        # Home side is always home; classify by fav/dog status
        if is_home_fav:
            cells["fav_home"].append(margin)   # home is fav, looking at home covers
            cells["dog_away"].append(margin)   # away is dog
        else:
            cells["dog_home"].append(margin)   # home is dog
            cells["fav_away"].append(margin)   # away is fav

    print(f"\n  Games with ML data: {sum(len(v) for v in cells.values()) // 2}")
    print(f"  Home-fav games: {len(cells['fav_home'])}")
    print(f"  Away-fav games: {len(cells['fav_away'])}")

    # Empirical cover rates per quadrant per rung
    print(f"\n  A. EMPIRICAL COVER RATES by quadrant:")

    for sv in [1.5, 2.5, 3.5, 4.5]:
        print(f"\n    Rung +{sv}:")
        print(f"      {'Quadrant':>15} {'N':>6} {'Cover%':>8} {'Mean Margin':>12}")
        print(f"      {'─' * 45}")

        for label, side in [
            ("Fav@Home", "fav_home"), ("Dog@Home", "dog_home"),
            ("Fav@Away", "fav_away"), ("Dog@Away", "dog_away"),
        ]:
            margins_cell = np.array(cells[side])
            n = len(margins_cell)

            if "home" in side.split("_")[1]:
                # Home perspective: home covers +sv when margin > -sv
                cover_rate = float(np.mean(margins_cell > -sv))
            else:
                # Away perspective: away covers +sv when margin < sv
                cover_rate = float(np.mean(margins_cell < sv))

            mean_margin = float(np.mean(margins_cell))
            print(f"      {label:>15} {n:>6} {cover_rate:>7.4f} {mean_margin:>+11.3f}")

    # Now compare to market implied per quadrant (using alt-lines period)
    spread_wp = _load_alt_line_odds("2026-04-22", "2026-06-26")

    # Match games in the alt-lines period to quadrants
    print(f"\n  B. MARKET IMPLIED vs EMPIRICAL per quadrant (Apr 22 - Jun 26):")

    period_games = [(h, a, gd, hs, as_) for h, a, gd, hs, as_ in games
                    if gd >= "2026-04-22" and gd <= "2026-06-26"]

    for sv in [1.5, 2.5, 3.5, 4.5]:
        print(f"\n    Rung +{sv}:")
        print(f"      {'Quadrant':>15} {'N':>5} {'Emp Cover':>10} {'Mkt Impl':>10} "
              f"{'Mkt-Emp':>8} {'Mispriced?':>11}")
        print(f"      {'─' * 65}")

        for label, is_home_fav_val, pick_side in [
            ("Fav@Home", True, "home"),
            ("Dog@Home", False, "home"),
            ("Fav@Away", False, "away"),  # away is fav when home is dog
            ("Dog@Away", True, "away"),   # away is dog when home is fav
        ]:
            emp_covers = []
            mkt_implieds = []

            for home, away, gd, hs, as_ in period_games:
                is_hf = ml_favs.get((home, away, gd))
                if is_hf is None or is_hf != is_home_fav_val:
                    continue

                margin = int(hs) - int(as_)
                if pick_side == "home":
                    covered = 1 if margin > -sv else 0
                    mkt = spread_wp.get((home, away, gd, sv))
                else:
                    covered = 1 if margin < sv else 0
                    mkt_neg = spread_wp.get((home, away, gd, -sv))
                    mkt = (1.0 - mkt_neg) if mkt_neg is not None else None

                emp_covers.append(covered)
                if mkt is not None:
                    mkt_implieds.append(mkt)

            if not emp_covers:
                continue

            emp_rate = np.mean(emp_covers)
            mkt_rate = np.mean(mkt_implieds) if mkt_implieds else float("nan")
            gap = mkt_rate - emp_rate if not np.isnan(mkt_rate) else float("nan")
            mispriced = "YES" if abs(gap) > 0.02 else "no" if not np.isnan(gap) else "?"

            print(f"      {label:>15} {len(emp_covers):>5} {emp_rate:>10.4f} "
                  f"{mkt_rate:>10.4f} {gap:>+7.4f} {mispriced:>11}")


# ═══════════════════════════════════════════════════════════════
# CHECK 2 + 3: LONGER-PERIOD ODDS SWEEP + BAND-FOR-BAND ATS
# ═══════════════════════════════════════════════════════════════

def generate_candidates_for_period(start, end, iso_home=None, iso_away=None, gate=0.55):
    """Generate NegBin candidates for a date range, optionally with calibration."""
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

    ml_favs = _load_ml_favorites_hist(start, end)
    odds_raw = _load_odds_raw(start, end)

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

                raw_home_wp = float(np.mean(margins > -sv))
                raw_away_wp = float(np.mean(margins < sv))

                if iso_home is not None and iso_away is not None:
                    home_wp = float(iso_home.predict([raw_home_wp])[0])
                    away_wp = float(iso_away.predict([raw_away_wp])[0])
                else:
                    home_wp = raw_home_wp
                    away_wp = raw_away_wp

                mkt_home = spread_wp.get((home, away, gd, sv))
                if mkt_home is None:
                    continue

                # Home side
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
                            "market_wp": mkt_home, "edge_pp": round(edge, 2),
                            "los": round(los, 4), "lqs": lqs,
                            "actual_result": result,
                            "is_pick_fav": home_is_fav if home_is_fav is not None else None,
                            "decimal_odds": dec_odds,
                            "fixture_key": fixture_key,
                        })

                # Away side
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
                            "market_wp": mkt_away, "edge_pp": round(edge, 2),
                            "los": round(los, 4), "lqs": lqs,
                            "actual_result": result,
                            "is_pick_fav": pick_is_fav,
                            "decimal_odds": dec_odds,
                            "fixture_key": fixture_key,
                        })

    return daily_candidates


def generate_ats_candidates_for_period(start, end):
    """Generate ATS candidates for a date range using existing backtest infra."""
    from walk_forward_backtest import (
        _load_scores, _load_alt_line_odds, _compute_los, _compute_lqs,
        _settle_spread, _predict_ats, _load_ats_submodel,
        SPREAD_UNIVERSE, ATS_MODEL_GATE, ALT_MIN_IMPLIED, LOS_FLOOR,
    )
    from features import build_feature_matrix

    ats_clf, ats_sc, ats_imp, ats_cols = _load_ats_submodel()
    ml_favs = _load_ml_favorites_hist(start, end)
    odds_raw = _load_odds_raw(start, end)

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

            # Pre-compute ATS predictions at all rungs
            ats_preds = {}
            for sv_ in SPREAD_UNIVERSE:
                p = _predict_ats(ats_clf, ats_sc, ats_imp, ats_cols, row, sv_)
                if p is not None:
                    ats_preds[sv_] = p

            for sv in SPREAD_UNIVERSE:
                if sv not in LIVE_RUNG_RANGE:
                    continue
                if sv not in ats_preds:
                    continue

                mkt_wp_home = spread_wp.get((home, away, gd, sv))
                if mkt_wp_home is None:
                    continue

                # Home side at +sv
                home_wp = ats_preds[sv]
                if home_wp >= ATS_MODEL_GATE and mkt_wp_home >= ALT_MIN_IMPLIED:
                    edge_pp = (home_wp - mkt_wp_home) * 100.0
                    lqs = _compute_lqs("Alt Spread", edge_pp, line=sv)
                    los = _compute_los(home_wp, edge_pp=edge_pp, lqs=lqs,
                                       cushion=(home_wp >= 0.70))
                    if los >= LOS_FLOOR:
                        result = _settle_spread(hs, as_, sv, "home")
                        dec_odds = odds_raw.get((home, away, gd, home, sv))
                        if dec_odds is None:
                            dec_odds = 1.0 / mkt_wp_home if mkt_wp_home > 0 else None
                        daily_candidates[gd].append({
                            "game_date": gd, "home_team": home, "away_team": away,
                            "spread_value": sv, "pick_side": "home",
                            "pick_team": home, "model_wp": round(home_wp, 4),
                            "market_wp": round(mkt_wp_home, 4),
                            "edge_pp": round(edge_pp, 2),
                            "los": round(los, 4), "lqs": lqs,
                            "actual_result": result,
                            "is_pick_fav": home_is_fav if home_is_fav is not None else None,
                            "decimal_odds": dec_odds,
                            "fixture_key": fixture_key,
                        })

                # Away side at +sv
                neg_sv = -sv
                if neg_sv in ats_preds:
                    away_model_wp = 1.0 - ats_preds[neg_sv]
                    mkt_neg = spread_wp.get((home, away, gd, neg_sv))
                    mkt_away = (1.0 - mkt_neg) if mkt_neg is not None else (1.0 - mkt_wp_home)
                    if away_model_wp >= ATS_MODEL_GATE and mkt_away >= ALT_MIN_IMPLIED:
                        edge_pp = (away_model_wp - mkt_away) * 100.0
                        lqs = _compute_lqs("Alt Spread", edge_pp, line=sv)
                        los = _compute_los(away_model_wp, edge_pp=edge_pp, lqs=lqs,
                                           cushion=(away_model_wp >= 0.70))
                        if los >= LOS_FLOOR:
                            result = _settle_spread(hs, as_, sv, "away")
                            dec_odds = odds_raw.get((home, away, gd, away, sv))
                            if dec_odds is None:
                                dec_odds = 1.0 / mkt_away if mkt_away > 0 else None
                            pick_is_fav = (not home_is_fav) if home_is_fav is not None else None
                            daily_candidates[gd].append({
                                "game_date": gd, "home_team": home, "away_team": away,
                                "spread_value": sv, "pick_side": "away",
                                "pick_team": away, "model_wp": round(away_model_wp, 4),
                                "market_wp": round(mkt_away, 4),
                                "edge_pp": round(edge_pp, 2),
                                "los": round(los, 4), "lqs": lqs,
                                "actual_result": result,
                                "is_pick_fav": pick_is_fav,
                                "decimal_odds": dec_odds,
                                "fixture_key": fixture_key,
                            })

    return daily_candidates


def odds_sweep_comparison(nb_daily, ats_daily, label=""):
    """Run odds-band sweep for both models, report band-for-band."""
    bands = [
        ("1.3-1.7x", 1.3, 1.7),
        ("1.7-2.2x", 1.7, 2.2),
        ("2.2-2.8x", 2.2, 2.8),
        ("2.8-3.5x", 2.8, 3.5),
        ("3.5-5.0x", 3.5, 5.0),
        ("unconstrained", None, None),
    ]

    print(f"\n    {'Band':>15} │ {'NB N':>5} {'NB WR':>7} {'NB ROI':>8} {'NB Edge':>8} │ "
          f"{'ATS N':>6} {'ATS WR':>8} {'ATS ROI':>9} {'ATS Edge':>9} │ {'NB-ATS':>7}")
    print(f"    {'─' * 105}")

    for band_label, lo, hi in bands:
        # NegBin
        nb_parlays = []
        for gd in sorted(nb_daily.keys()):
            day_p = assemble_parlays_odds_band(nb_daily[gd], 8, lo, hi)
            for p in day_p:
                p["game_date"] = gd
            nb_parlays.extend(day_p)

        # ATS
        ats_parlays = []
        for gd in sorted(ats_daily.keys()):
            day_p = assemble_parlays_odds_band(ats_daily[gd], 8, lo, hi)
            for p in day_p:
                p["game_date"] = gd
            ats_parlays.extend(day_p)

        nb_s = _pnl_stats(nb_parlays)
        ats_s = _pnl_stats(ats_parlays)

        nb_be = (1.0 / nb_s["avg_odds"] * 100) if nb_s["avg_odds"] > 0 else 0
        ats_be = (1.0 / ats_s["avg_odds"] * 100) if ats_s["avg_odds"] > 0 else 0
        nb_edge = nb_s["wr"] - nb_be
        ats_edge = ats_s["wr"] - ats_be
        delta = nb_s["roi"] - ats_s["roi"] if nb_s["n"] > 0 and ats_s["n"] > 0 else float("nan")

        nb_str = (f"{nb_s['n']:>5} {nb_s['wr']:>6.1f}% {nb_s['roi']:>+7.1f}% {nb_edge:>+7.1f}pp"
                  if nb_s["n"] > 0 else f"{'0':>5} {'—':>7} {'—':>8} {'—':>8}")
        ats_str = (f"{ats_s['n']:>6} {ats_s['wr']:>7.1f}% {ats_s['roi']:>+8.1f}% {ats_edge:>+8.1f}pp"
                   if ats_s["n"] > 0 else f"{'0':>6} {'—':>8} {'—':>9} {'—':>9}")
        delta_str = f"{delta:>+6.1f}pp" if not np.isnan(delta) else f"{'—':>7}"

        print(f"    {band_label:>15} │ {nb_str} │ {ats_str} │ {delta_str}")


# ═══════════════════════════════════════════════════════════════
# CHECK 4: HELD-OUT CALIBRATION
# ═══════════════════════════════════════════════════════════════

def check4_held_out_calibration():
    """Fit isotonic on earlier period (NegBin predictions + outcomes),
    evaluate on May-Jun 2026."""
    from features import build_feature_matrix
    from negbin_model import load_negbin_model, _simulate_margins
    from walk_forward_backtest import _load_scores

    print("\n" + "=" * 100)
    print("  CHECK 4: HELD-OUT ISOTONIC CALIBRATION")
    print("=" * 100)

    model = load_negbin_model()
    gbm_home = model["gbm_home"]
    gbm_away = model["gbm_away"]
    feat_cols = model["feature_cols"]

    matrix = build_feature_matrix("MLB")
    matrix["_gd"] = matrix["game_date"].astype(str).str[:10]

    # Training period for calibration: Aug 2025 - Apr 2026
    # (Jul 2025 is first month of NegBin test set, skip it for warm-up)
    CAL_START = "2025-08-01"
    CAL_END = "2026-04-30"
    EVAL_START = "2026-05-01"
    EVAL_END = "2026-06-26"

    scores_cal = _load_scores(CAL_START, CAL_END)
    scores_eval = _load_scores(EVAL_START, EVAL_END)

    def _get_predictions(start, end, scores_dict):
        bt = matrix[(matrix["_gd"] >= start) & (matrix["_gd"] <= end)]
        results = []
        for idx, row in bt.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            gd = row["_gd"]
            score_key = (home, away, gd)
            if score_key not in scores_dict:
                continue
            hs, as_ = scores_dict[score_key]
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
                results.append({
                    "sv": sv, "home_wp": home_wp, "away_wp": away_wp,
                    "home_actual": home_actual, "away_actual": away_actual,
                })
        return pd.DataFrame(results)

    print(f"\n  Calibration period: {CAL_START} to {CAL_END}")
    print(f"  Evaluation period: {EVAL_START} to {EVAL_END}")

    print(f"\n  Generating calibration predictions...")
    cal_df = _get_predictions(CAL_START, CAL_END, scores_cal)
    print(f"  Calibration observations: {len(cal_df)}")

    print(f"  Generating evaluation predictions...")
    eval_df = _get_predictions(EVAL_START, EVAL_END, scores_eval)
    print(f"  Evaluation observations: {len(eval_df)}")

    if cal_df.empty or eval_df.empty:
        print("  ERROR: insufficient data for held-out calibration")
        return None, None

    # Fit isotonic on calibration period
    iso_home = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_home.fit(cal_df["home_wp"].values, cal_df["home_actual"].values)

    iso_away = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_away.fit(cal_df["away_wp"].values, cal_df["away_actual"].values)

    # Evaluate on held-out period
    eval_home_raw = eval_df["home_wp"].values
    eval_home_cal = iso_home.predict(eval_home_raw)
    eval_home_actual = eval_df["home_actual"].values

    eval_away_raw = eval_df["away_wp"].values
    eval_away_cal = iso_away.predict(eval_away_raw)
    eval_away_actual = eval_df["away_actual"].values

    brier_raw_h = float(np.mean((eval_home_raw - eval_home_actual) ** 2))
    brier_cal_h = float(np.mean((eval_home_cal - eval_home_actual) ** 2))
    brier_raw_a = float(np.mean((eval_away_raw - eval_away_actual) ** 2))
    brier_cal_a = float(np.mean((eval_away_cal - eval_away_actual) ** 2))

    print(f"\n  HELD-OUT Brier scores (evaluation period):")
    print(f"    Home: raw={brier_raw_h:.4f} → calibrated={brier_cal_h:.4f} (Δ={brier_cal_h - brier_raw_h:+.4f})")
    print(f"    Away: raw={brier_raw_a:.4f} → calibrated={brier_cal_a:.4f} (Δ={brier_cal_a - brier_raw_a:+.4f})")

    # Show calibration mapping
    print(f"\n    Held-out calibration mapping:")
    print(f"    {'Raw WP':>8} │ {'Home Calib':>11} {'Away Calib':>11}")
    print(f"    {'─' * 35}")
    for raw in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        hc = float(iso_home.predict([raw])[0])
        ac = float(iso_away.predict([raw])[0])
        print(f"    {raw:>8.2f} │ {hc:>11.4f} {ac:>11.4f}")

    # Per-rung held-out comparison
    print(f"\n    Per-rung held-out home/away gap:")
    print(f"    {'Rung':>6} │ {'Raw Gap':>8} {'Calib Gap':>10} {'Emp Gap':>8}")
    print(f"    {'─' * 40}")
    for sv in [1.5, 2.5, 3.5, 4.5]:
        rung = eval_df[eval_df["sv"] == sv]
        raw_h = rung["home_wp"].mean()
        raw_a = rung["away_wp"].mean()
        cal_h = float(iso_home.predict(rung["home_wp"].values).mean())
        cal_a = float(iso_away.predict(rung["away_wp"].values).mean())
        emp_h = rung["home_actual"].mean()
        emp_a = rung["away_actual"].mean()
        print(f"    {sv:>+5.1f} │ {raw_h - raw_a:>+7.4f} {cal_h - cal_a:>+9.4f} {emp_h - emp_a:>+7.4f}")

    return iso_home, iso_away


def main():
    print("=" * 100)
    print("  NEGBIN VALIDATION — FOUR CHECKS")
    print("=" * 100)

    # ── CHECK 1 ──
    check1_mispricing_matrix()

    # ── CHECK 4 (before 2+3, since calibrators are needed) ──
    iso_home_ho, iso_away_ho = check4_held_out_calibration()

    # ── CHECK 2 + 3: Longer period, band-for-band ──
    print("\n" + "=" * 100)
    print("  CHECK 2 + 3: LONGER-PERIOD ODDS SWEEP + BAND-FOR-BAND ATS")
    print("=" * 100)

    LONG_START = "2026-04-22"
    LONG_END = "2026-06-26"
    SHORT_START = "2026-05-26"

    print(f"\n  Generating NegBin candidates (raw, {LONG_START} to {LONG_END})...")
    nb_daily_long_raw = generate_candidates_for_period(LONG_START, LONG_END, gate=0.55)
    nb_cands_long = sum(len(v) for v in nb_daily_long_raw.values())
    nb_days_long = len(nb_daily_long_raw)
    print(f"    NegBin raw: {nb_days_long} days, {nb_cands_long} candidates")

    if iso_home_ho is not None:
        print(f"\n  Generating NegBin candidates (held-out calibrated, {LONG_START} to {LONG_END})...")
        nb_daily_long_cal = generate_candidates_for_period(
            LONG_START, LONG_END, iso_home=iso_home_ho, iso_away=iso_away_ho, gate=0.55)
        nb_cands_cal = sum(len(v) for v in nb_daily_long_cal.values())
        print(f"    NegBin calibrated: {len(nb_daily_long_cal)} days, {nb_cands_cal} candidates")

    print(f"\n  Generating ATS candidates ({LONG_START} to {LONG_END})...")
    ats_daily_long = generate_ats_candidates_for_period(LONG_START, LONG_END)
    ats_cands_long = sum(len(v) for v in ats_daily_long.values())
    print(f"    ATS: {len(ats_daily_long)} days, {ats_cands_long} candidates")

    # 2a. Raw NegBin vs ATS, full period
    print(f"\n  ── RAW NegBin vs ATS, BAND-FOR-BAND ({LONG_START} to {LONG_END}) ──")
    odds_sweep_comparison(nb_daily_long_raw, ats_daily_long, "full-raw")

    # 2b. Held-out calibrated NegBin vs ATS, full period
    if iso_home_ho is not None:
        print(f"\n  ── HELD-OUT CALIBRATED NegBin vs ATS, BAND-FOR-BAND ({LONG_START} to {LONG_END}) ──")
        odds_sweep_comparison(nb_daily_long_cal, ats_daily_long, "full-cal")

    # 2c. Short period (May-Jun) for comparison
    nb_daily_short = {gd: v for gd, v in nb_daily_long_raw.items() if gd >= SHORT_START}
    ats_daily_short = {gd: v for gd, v in ats_daily_long.items() if gd >= SHORT_START}
    print(f"\n  ── RAW NegBin vs ATS, BAND-FOR-BAND ({SHORT_START} to {LONG_END}, original period) ──")
    odds_sweep_comparison(nb_daily_short, ats_daily_short, "short-raw")

    print(f"\n  ── PERIOD COMPARISON (does edge hold in earlier period Apr 22-May 25?) ──")
    nb_daily_early = {gd: v for gd, v in nb_daily_long_raw.items() if gd < SHORT_START}
    ats_daily_early = {gd: v for gd, v in ats_daily_long.items() if gd < SHORT_START}
    early_days = len(nb_daily_early)
    early_cands = sum(len(v) for v in nb_daily_early.values())
    print(f"    Early period: {early_days} days, {early_cands} NegBin candidates")
    if early_days > 0:
        odds_sweep_comparison(nb_daily_early, ats_daily_early, "early")


if __name__ == "__main__":
    main()
