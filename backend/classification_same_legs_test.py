"""
classification_same_legs_test.py — Same-legs test for is_favorite_spread bug.

For each historical leg where spread-sign and ML-based classification DISAGREE:
  - Re-score with the CORRECT is_fav_spread value (ML-based)
  - Compare win_prob: correct vs wrong label on the SAME leg
  - Does correcting the label improve the model's predictions?

This isolates the feature's effect from the population confound.
"""
from __future__ import annotations
import os, sys, json, math, pickle, re, sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

DB_PATH = os.path.join(_HERE, "..", "data", "bets.db")


def _load_ml_fav_map():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date_ct, bookmakers
        FROM fixtures
        WHERE sport_key = 'baseball_mlb' AND bookmakers IS NOT NULL
    """).fetchall()
    conn.close()
    result = {}
    for home, away, gd, bk_json in rows:
        if not bk_json:
            continue
        try:
            bks = json.loads(bk_json)
        except (json.JSONDecodeError, TypeError):
            continue
        home_price = None
        for pref in ["fanduel", "draftkings", "betmgm", "pinnacle", None]:
            for bk in bks:
                if pref is not None and bk.get("key") != pref:
                    continue
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    for oc in mkt.get("outcomes", []):
                        if oc.get("name", "").lower().strip() == home.lower().strip():
                            home_price = float(oc.get("price", 0))
                            break
                    if home_price: break
                if home_price: break
            if home_price: break
        if home_price and home_price > 0:
            result[(home, away, str(gd)[:10])] = home_price < 2.0
    return result


def _load_ats_model():
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


def _predict_with_fav(clf, scaler, imputer, feat_cols, row, spread_value, is_fav_override):
    """Predict P(home covers) with an explicit is_favorite_spread override."""
    def _val(col):
        if col == "_spread_value":
            return float(spread_value)
        if col == "_spread_abs":
            return abs(float(spread_value))
        if col == "_spread_sq":
            return float(spread_value) ** 2
        if col == "_is_fav_spread":
            return float(is_fav_override)  # THE OVERRIDE
        if col in ("_total_line", "_total_line_abs", "_line_over_base_rate"):
            return np.nan
        if col == "_is_high_total":
            return 0.0
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


def _wilson_ci(n, w, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = w / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round((centre - margin) * 100, 1), round((centre + margin) * 100, 1))


def _thin(n):
    if n < 10: return " ⚠THIN"
    if n < 20: return " ⚠thin"
    return ""


def main():
    from features import build_feature_matrix

    print("=" * 110)
    print("  SAME-LEGS TEST: correct vs wrong is_favorite_spread on identical legs")
    print("=" * 110)

    # Load model and features
    print("\nLoading ATS model...")
    clf, scaler, imputer, feat_cols = _load_ats_model()
    print(f"  Features: {len(feat_cols)}, has _is_fav_spread: {'_is_fav_spread' in feat_cols}")

    print("Loading ML fav map...")
    ml_favs = _load_ml_fav_map()
    print(f"  {len(ml_favs)} fixtures")

    print("Building feature matrix...")
    matrix = build_feature_matrix("MLB")
    matrix["_gd"] = matrix["game_date"].astype(str).str[:10]
    print(f"  {len(matrix)} rows")

    # Load settled legs with outcomes
    conn = sqlite3.connect(DB_PATH)
    legs = conn.execute("""
        SELECT mbl.id, mbl.description, mbl.leg_result,
               mbl.spread_value, mbl.is_favorite_spread,
               mbl.predicted_win_prob, mbl.open_odds,
               mb.game_date,
               mbl.resolved_home_team, mbl.resolved_away_team
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.market_type IN ('alternate_spreads', 'spreads')
          AND mbl.spread_value IS NOT NULL
          AND mb.sport LIKE '%%MLB%%'
    """).fetchall()
    conn.close()
    print(f"  {len(legs)} settled MLB spread legs")

    # For each leg: get both predictions (wrong vs correct is_fav)
    results = []
    skipped = 0

    for row_data in legs:
        lid, desc, result, sv, is_fav_stored, wp_stored, odds_am, gd, home, away = row_data
        if home is None or away is None or desc is None or sv is None:
            skipped += 1
            continue

        # Extract pick team
        m = re.match(r'^(.+?)\s+[+-]', desc)
        pick = m.group(1).strip() if m else None
        if pick is None:
            skipped += 1
            continue

        gd_str = str(gd)[:10]

        # ML classification
        home_is_fav = ml_favs.get((home, away, gd_str))
        if home_is_fav is None:
            skipped += 1
            continue

        # Spread-sign classification
        ss_fav = 1 if sv < 0 else 0

        # ML-based: is the PICK a favorite?
        if pick.lower().strip() == home.lower().strip():
            ml_pick_is_fav = 1 if home_is_fav else 0
        elif pick.lower().strip() == away.lower().strip():
            ml_pick_is_fav = 1 if not home_is_fav else 0
        else:
            skipped += 1
            continue

        # Only interested in DISAGREEMENT legs
        if ss_fav == ml_pick_is_fav:
            continue

        # Find this game in the feature matrix
        # The ATS model predicts P(HOME covers). For an away pick,
        # we need to flip perspective.
        pick_is_home = pick.lower().strip() == home.lower().strip()

        mask = (matrix["home_team"].str.lower().str.strip() == home.lower().strip()) & \
               (matrix["_gd"] == gd_str)
        matched = matrix[mask]
        if matched.empty:
            skipped += 1
            continue

        feat_row = matched.iloc[0]

        # For the ATS model, is_fav_spread is about the HOME team at this spread
        # If sv > 0, home is getting cushion → ss says NOT favorite
        # ML says: if home_is_fav and sv > 0, home IS a favorite taking cushion
        # The wrong value: ss_fav (0 for positive spreads)
        # The correct value for HOME: 1 if home_is_fav else 0
        wrong_is_fav = ss_fav  # spread-sign (what model currently uses)
        correct_is_fav = 1 if home_is_fav else 0

        # Get both predictions
        wp_wrong = _predict_with_fav(clf, scaler, imputer, feat_cols, feat_row, sv, wrong_is_fav)
        wp_correct = _predict_with_fav(clf, scaler, imputer, feat_cols, feat_row, sv, correct_is_fav)

        if wp_wrong is None or wp_correct is None:
            skipped += 1
            continue

        # For away picks, flip: P(pick wins) = 1 - P(home covers)
        if not pick_is_home:
            wp_wrong = 1.0 - wp_wrong
            wp_correct = 1.0 - wp_correct

        results.append({
            "lid": lid, "desc": desc, "result": result,
            "sv": sv, "gd": gd_str, "home": home, "away": away, "pick": pick,
            "ss_fav": ss_fav, "ml_fav": ml_pick_is_fav,
            "wp_wrong": wp_wrong * 100,
            "wp_correct": wp_correct * 100,
            "wp_delta": (wp_correct - wp_wrong) * 100,
            "pick_is_home": pick_is_home,
            "odds_am": odds_am,
        })

    print(f"\n  Disagreement legs re-scored: {len(results)}")
    print(f"  Skipped: {skipped}")

    if not results:
        print("  No results to analyze.")
        return

    # ═══════════════════════════════════════════════════════════
    # 1. OVERALL: does correcting the label change win_prob?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  1. OVERALL: win_prob change when is_fav_spread is corrected")
    print(f"{'='*110}")

    deltas = [r["wp_delta"] for r in results]
    avg_delta = sum(deltas) / len(deltas)
    median_delta = sorted(deltas)[len(deltas) // 2]
    pos = sum(1 for d in deltas if d > 0.1)
    neg = sum(1 for d in deltas if d < -0.1)
    zero = len(deltas) - pos - neg

    print(f"\n  N = {len(results)} disagreement legs")
    print(f"  Avg wp_delta (correct - wrong): {avg_delta:+.3f}pp")
    print(f"  Median wp_delta:                {median_delta:+.3f}pp")
    print(f"  wp increases (correct > wrong):  {pos} ({pos/len(results)*100:.0f}%)")
    print(f"  wp decreases (correct < wrong):  {neg} ({neg/len(results)*100:.0f}%)")
    print(f"  wp unchanged (<0.1pp):           {zero} ({zero/len(results)*100:.0f}%)")

    avg_wrong = sum(r["wp_wrong"] for r in results) / len(results)
    avg_correct = sum(r["wp_correct"] for r in results) / len(results)
    print(f"\n  Avg win_prob (wrong label):   {avg_wrong:.2f}%")
    print(f"  Avg win_prob (correct label): {avg_correct:.2f}%")

    # ═══════════════════════════════════════════════════════════
    # 2. BREAKDOWN: by disagreement type
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  2. BY DISAGREEMENT TYPE")
    print(f"{'='*110}")

    # Type 1: ss=fav(1), ml=dog(0) — model thinks fav, actually dog
    # Type 2: ss=dog(0), ml=fav(1) — model thinks dog, actually fav (the big group)
    for label, filt_fn in [
        ("SS=fav ML=dog (n=small: model wrongly told 'fav')",
         lambda r: r["ss_fav"] == 1 and r["ml_fav"] == 0),
        ("SS=dog ML=fav (n=big: fav taking +cushion, model wrongly told 'dog')",
         lambda r: r["ss_fav"] == 0 and r["ml_fav"] == 1),
    ]:
        subset = [r for r in results if filt_fn(r)]
        if not subset:
            continue
        avg_d = sum(r["wp_delta"] for r in subset) / len(subset)
        wins = sum(1 for r in subset if r["result"] == "WIN")
        wr = wins / len(subset) * 100
        avg_w = sum(r["wp_wrong"] for r in subset) / len(subset)
        avg_c = sum(r["wp_correct"] for r in subset) / len(subset)
        lo, hi = _wilson_ci(len(subset), wins)

        print(f"\n  {label}:")
        print(f"    N={len(subset)}, WR={wr:.1f}% [{lo:.1f}%,{hi:.1f}%]")
        print(f"    Avg wp (wrong):   {avg_w:.2f}%")
        print(f"    Avg wp (correct): {avg_c:.2f}%")
        print(f"    Avg delta:        {avg_d:+.3f}pp")

    # ═══════════════════════════════════════════════════════════
    # 3. BY SPREAD VALUE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  3. BY SPREAD VALUE")
    print(f"{'='*110}")

    print(f"\n  {'SV':>6} {'N':>5} {'WR':>7} {'WP Wrong':>10} {'WP Correct':>11} {'Delta':>8}")
    print(f"  {'─'*50}")
    for sv_val in sorted(set(r["sv"] for r in results)):
        g = [r for r in results if r["sv"] == sv_val]
        wins = sum(1 for r in g if r["result"] == "WIN")
        wr = wins / len(g) * 100
        avg_w = sum(r["wp_wrong"] for r in g) / len(g)
        avg_c = sum(r["wp_correct"] for r in g) / len(g)
        avg_d = sum(r["wp_delta"] for r in g) / len(g)
        print(f"  {sv_val:>+5.1f} {len(g):>5} {wr:>5.1f}% {avg_w:>9.2f}% {avg_c:>10.2f}% "
              f"{avg_d:>+7.3f}pp{_thin(len(g))}")

    # ═══════════════════════════════════════════════════════════
    # 4. CALIBRATION: does correcting improve Brier?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  4. CALIBRATION: Brier score comparison")
    print(f"{'='*110}")

    brier_wrong = 0
    brier_correct = 0
    n_brier = 0
    for r in results:
        actual = 1.0 if r["result"] == "WIN" else 0.0
        brier_wrong += (r["wp_wrong"] / 100 - actual) ** 2
        brier_correct += (r["wp_correct"] / 100 - actual) ** 2
        n_brier += 1

    brier_wrong /= n_brier
    brier_correct /= n_brier

    print(f"\n  Brier (wrong label):   {brier_wrong:.4f}")
    print(f"  Brier (correct label): {brier_correct:.4f}")
    print(f"  Delta:                 {brier_correct - brier_wrong:+.4f}")
    if brier_correct < brier_wrong:
        print(f"  → Correct label IMPROVES calibration by {(brier_wrong - brier_correct) / brier_wrong * 100:.1f}%")
    elif brier_correct > brier_wrong:
        print(f"  → Correct label WORSENS calibration by {(brier_correct - brier_wrong) / brier_wrong * 100:.1f}%")
    else:
        print(f"  → No difference")

    # ═══════════════════════════════════════════════════════════
    # 5. RANKING IMPACT: would correcting change which legs
    #    get selected (above wp threshold)?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  5. RANKING IMPACT: would correcting change selection?")
    print(f"{'='*110}")

    # Count legs that cross key thresholds
    for threshold in [55.0, 60.0, 65.0, 70.0]:
        wrong_above = sum(1 for r in results if r["wp_wrong"] >= threshold)
        correct_above = sum(1 for r in results if r["wp_correct"] >= threshold)
        both = sum(1 for r in results if r["wp_wrong"] >= threshold and r["wp_correct"] >= threshold)
        only_wrong = sum(1 for r in results if r["wp_wrong"] >= threshold and r["wp_correct"] < threshold)
        only_correct = sum(1 for r in results if r["wp_correct"] >= threshold and r["wp_wrong"] < threshold)
        print(f"  wp ≥ {threshold}%: wrong={wrong_above}, correct={correct_above}, "
              f"both={both}, only-wrong={only_wrong}, only-correct={only_correct}")

    # ═══════════════════════════════════════════════════════════
    # 6. REGIME SPLIT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  6. REGIME SPLIT")
    print(f"{'='*110}")

    for label, start, end in [("Pre-5/26", "2026-04-22", "2026-05-25"),
                               ("Post-5/26", "2026-05-26", "2026-06-26")]:
        sub = [r for r in results if start <= r["gd"] <= end]
        if not sub:
            continue
        avg_d = sum(r["wp_delta"] for r in sub) / len(sub)
        bw = sum((r["wp_wrong"]/100 - (1.0 if r["result"]=="WIN" else 0.0))**2 for r in sub) / len(sub)
        bc = sum((r["wp_correct"]/100 - (1.0 if r["result"]=="WIN" else 0.0))**2 for r in sub) / len(sub)
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    Avg wp_delta: {avg_d:+.3f}pp")
        print(f"    Brier wrong:  {bw:.4f}")
        print(f"    Brier correct:{bc:.4f}")
        print(f"    Brier delta:  {bc-bw:+.4f}")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    print(f"\n  Same-legs re-scoring: {len(results)} disagreement legs")
    print(f"  Avg wp change: {avg_delta:+.3f}pp")
    print(f"  Brier change:  {brier_correct - brier_wrong:+.4f}")

    if abs(avg_delta) < 0.5 and abs(brier_correct - brier_wrong) < 0.005:
        print(f"\n  → Feature has NEGLIGIBLE impact on model predictions.")
        print(f"  → Bug is truly costless — feature is weak/redundant in the trained model.")
        print(f"  → SKIP Shadow A retrain. Move to Thread 2 Step A.")
    elif brier_correct < brier_wrong:
        print(f"\n  → Correct label IMPROVES calibration.")
        print(f"  → Bug has a real cost, but distribution model (Step B) fixes natively.")
        print(f"  → Fix is_fav_spread definition for cleanliness, but retrain is low-priority.")
    else:
        print(f"\n  → Correct label doesn't improve (or worsens) calibration.")
        print(f"  → The model learned around the bug; correcting could destabilize.")
        print(f"  → SKIP Shadow A retrain. Move to Thread 2 Step A.")


if __name__ == "__main__":
    main()
