"""
classification_bug_cost.py — Measure the cost of the is_favorite_spread bug
WITHOUT retraining the ATS model.

The bug: is_favorite_spread = 1 if spread_value < 0 else 0 (spread-sign).
Correct: h2h moneyline decimal price < 2.0 = favorite (ML-based).

Questions:
1. How often do they disagree on historical legs?
2. Do disagreement legs have systematically worse outcomes (WR, ROI)?
3. How big is the classification error's impact on model quality?

If the bug's cost is measurable cheaply → no need to retrain the doomed ATS model.
"""
from __future__ import annotations
import os, sys, json, math, sqlite3
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

DB_PATH = os.path.join(_HERE, "..", "data", "bets.db")


def _load_ml_fav_map():
    """Load ML-based fav classification from fixtures h2h prices."""
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
                    if home_price:
                        break
                if home_price:
                    break
            if home_price:
                break
        if home_price and home_price > 0:
            result[(home, away, str(gd)[:10])] = home_price < 2.0
    return result


def _wilson_ci(n, w, z=1.96):
    if n == 0:
        return (0.0, 0.0)
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
    print("=" * 100)
    print("  CLASSIFICATION BUG COST ANALYSIS")
    print("  spread-sign (sv<0) vs ML-based (h2h<2.0)")
    print("=" * 100)

    ml_favs = _load_ml_fav_map()
    print(f"\n  ML fav map: {len(ml_favs)} fixtures")

    # Load settled mock_bet_legs with spread data
    conn = sqlite3.connect(DB_PATH)
    legs = conn.execute("""
        SELECT mbl.id, mbl.description, mbl.leg_result, mbl.market_type,
               mbl.spread_value, mbl.is_favorite_spread,
               mbl.predicted_win_prob, mbl.open_odds,
               mb.game_date,
               mbl.resolved_home_team, mbl.resolved_away_team
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.market_type IN ('alternate_spreads', 'spreads')
          AND mbl.spread_value IS NOT NULL
          AND mb.sport LIKE '%MLB%'
    """).fetchall()
    conn.close()

    print(f"  Settled MLB spread legs: {len(legs)}")

    # Try to get home_team/away_team from the leg or from description
    # Classify each leg
    agree = []
    disagree = []
    unclassifiable = 0

    for row in legs:
        lid, desc, result, mkt, sv, is_fav_spread, wp, odds_am, gd, home, away = row

        if home is None or away is None or desc is None:
            unclassifiable += 1
            continue

        # Extract pick team from description: "Team Name +1.5 (Alt Spread)"
        import re
        _pick_match = re.match(r'^(.+?)\s+[+-]', desc)
        pick = _pick_match.group(1).strip() if _pick_match else None
        if pick is None:
            unclassifiable += 1
            continue

        # Spread-sign classification (the bug)
        spread_sign_fav = 1 if sv is not None and sv < 0 else 0

        # ML-based classification (correct)
        home_is_fav = ml_favs.get((home, away, str(gd)[:10]))
        if home_is_fav is None:
            unclassifiable += 1
            continue

        # Is the pick a favorite per ML?
        if pick.strip().lower() == home.strip().lower():
            ml_fav = 1 if home_is_fav else 0
        elif pick.strip().lower() == away.strip().lower():
            ml_fav = 1 if not home_is_fav else 0
        else:
            unclassifiable += 1
            continue

        entry = {
            "id": lid, "desc": desc, "result": result, "sv": sv,
            "spread_sign_fav": spread_sign_fav, "ml_fav": ml_fav,
            "wp": wp, "odds_am": odds_am, "gd": gd,
            "is_fav_spread_stored": is_fav_spread,
        }

        if spread_sign_fav == ml_fav:
            agree.append(entry)
        else:
            disagree.append(entry)

    total = len(agree) + len(disagree)
    print(f"  Classifiable: {total}, Unclassifiable: {unclassifiable}")
    print(f"  Agree: {len(agree)} ({len(agree)/total*100:.1f}%)")
    print(f"  Disagree: {len(disagree)} ({len(disagree)/total*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # 1. DISAGREEMENT BREAKDOWN
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  1. DISAGREEMENT BREAKDOWN")
    print(f"{'='*100}")

    # What does disagreement look like?
    # spread_sign says fav but ML says dog = picking dog with wrong "fav" feature
    # spread_sign says dog but ML says fav = picking fav with wrong "dog" feature
    ss_fav_ml_dog = [e for e in disagree if e["spread_sign_fav"] == 1 and e["ml_fav"] == 0]
    ss_dog_ml_fav = [e for e in disagree if e["spread_sign_fav"] == 0 and e["ml_fav"] == 1]

    print(f"\n  Spread-sign=FAV but ML=DOG: {len(ss_fav_ml_dog)} legs")
    print(f"    (model thinks it's picking a fav, but it's actually a dog)")
    print(f"  Spread-sign=DOG but ML=FAV: {len(ss_dog_ml_fav)} legs")
    print(f"    (model thinks it's picking a dog, but it's actually a fav)")

    # Spread values where disagreement happens
    print(f"\n  Disagreement by spread value:")
    sv_counts = defaultdict(int)
    for e in disagree:
        sv_counts[e["sv"]] += 1
    for sv in sorted(sv_counts.keys()):
        pct = sv_counts[sv] / len(disagree) * 100
        print(f"    sv={sv:+.1f}: {sv_counts[sv]} ({pct:.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # 2. OUTCOME COMPARISON: agree vs disagree
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  2. OUTCOME COMPARISON: agree vs disagree legs")
    print(f"{'='*100}")

    def _stats(entries, label):
        n = len(entries)
        if n == 0:
            return
        wins = sum(1 for e in entries if e["result"] == "WIN")
        wr = wins / n * 100
        lo, hi = _wilson_ci(n, wins)

        # ROI via american odds
        pnl = 0
        staked = 0
        for e in entries:
            am = e.get("odds_am")
            if am is None:
                continue
            am = float(am)
            if am >= 100:
                dec = 1 + am / 100
            elif am <= -100:
                dec = 1 + 100 / abs(am)
            else:
                continue
            staked += 10
            pnl += (dec - 1) * 10 if e["result"] == "WIN" else -10
        roi = pnl / staked * 100 if staked > 0 else 0

        avg_wp = sum(e["wp"] for e in entries if e["wp"]) / sum(1 for e in entries if e["wp"]) if any(e["wp"] for e in entries) else 0

        print(f"  {label:<25} N={n:>4}  WR={wr:>5.1f}% [{lo:>5.1f}%,{hi:>5.1f}%]  "
              f"ROI={roi:>+7.1f}%  Avg WP={avg_wp:>5.1f}%{_thin(n)}")

    _stats(agree, "Agree (correct feature)")
    _stats(disagree, "Disagree (wrong feature)")
    _stats(ss_fav_ml_dog, "  SS=fav, ML=dog")
    _stats(ss_dog_ml_fav, "  SS=dog, ML=fav")

    # ═══════════════════════════════════════════════════════════
    # 3. REGIME SPLIT: pre vs post 5/26
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  3. REGIME SPLIT")
    print(f"{'='*100}")

    for regime_label, start, end in [("Pre-5/26", "2026-04-22", "2026-05-25"),
                                      ("Post-5/26", "2026-05-26", "2026-06-26")]:
        r_agree = [e for e in agree if start <= e["gd"] <= end]
        r_disagree = [e for e in disagree if start <= e["gd"] <= end]
        r_total = len(r_agree) + len(r_disagree)
        if r_total == 0:
            continue
        print(f"\n  {regime_label} (n={r_total}, disagree={len(r_disagree)} = {len(r_disagree)/r_total*100:.1f}%):")
        _stats(r_agree, f"  Agree")
        _stats(r_disagree, f"  Disagree")

    # ═══════════════════════════════════════════════════════════
    # 4. MODEL FEATURE IMPACT: does the wrong feature mislead?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  4. FEATURE IMPACT: is_favorite_spread stored value vs ML truth")
    print(f"{'='*100}")

    # For disagreement legs: what wp did the model assign?
    # If the model was misled by wrong is_fav, the wp should be
    # systematically over/under for disagreement legs
    print(f"\n  Average win_prob assigned by model:")
    _stats([e for e in agree if e["ml_fav"] == 1], "Agree + ML=fav")
    _stats([e for e in agree if e["ml_fav"] == 0], "Agree + ML=dog")
    _stats([e for e in disagree if e["ml_fav"] == 1], "Disagree + ML=fav")
    _stats([e for e in disagree if e["ml_fav"] == 0], "Disagree + ML=dog")

    # ═══════════════════════════════════════════════════════════
    # 5. CALIBRATION: is the model's win_prob less calibrated
    #    on disagreement legs?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  5. CALIBRATION: model accuracy on agree vs disagree")
    print(f"{'='*100}")

    for label, entries in [("Agree", agree), ("Disagree", disagree)]:
        if not entries:
            continue
        # Bucket by win_prob quintile
        wp_entries = [(e["wp"], e["result"]) for e in entries if e["wp"]]
        if not wp_entries:
            continue
        wp_entries.sort(key=lambda x: x[0])
        n = len(wp_entries)
        q = max(1, n // 4)
        print(f"\n  {label} ({n} legs with win_prob):")
        print(f"  {'WP Range':<20} {'N':>5} {'Predicted':>10} {'Actual WR':>10} {'Gap':>8}")
        print(f"  {'─'*55}")
        for i in range(0, n, q):
            chunk = wp_entries[i:i+q]
            if not chunk:
                continue
            avg_pred = sum(c[0] for c in chunk) / len(chunk)
            actual_wr = sum(1 for c in chunk if c[1] == "WIN") / len(chunk) * 100
            lo_wp = chunk[0][0]
            hi_wp = chunk[-1][0]
            gap = actual_wr - avg_pred
            print(f"  {lo_wp:>5.1f}% – {hi_wp:>5.1f}%    {len(chunk):>5} {avg_pred:>9.1f}% {actual_wr:>9.1f}% {gap:>+7.1f}pp")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  VERDICT")
    print(f"{'='*100}")

    disagree_pct = len(disagree) / total * 100
    print(f"\n  Disagreement rate: {disagree_pct:.1f}% ({len(disagree)}/{total})")
    if disagree_pct < 10:
        print(f"  → Low disagreement. Bug affects <10% of legs — limited upside from retrain.")
    elif disagree_pct < 25:
        print(f"  → Moderate disagreement. Bug affects {disagree_pct:.0f}% of legs.")
    else:
        print(f"  → High disagreement. Bug affects {disagree_pct:.0f}% of legs — significant.")

    # Compare outcomes
    a_wins = sum(1 for e in agree if e["result"] == "WIN")
    d_wins = sum(1 for e in disagree if e["result"] == "WIN")
    a_wr = a_wins / len(agree) * 100 if agree else 0
    d_wr = d_wins / len(disagree) * 100 if disagree else 0
    wr_gap = a_wr - d_wr

    print(f"  WR gap (agree - disagree): {wr_gap:+.1f}pp")
    if abs(wr_gap) < 3:
        print(f"  → Negligible WR gap. Wrong feature doesn't measurably hurt outcomes.")
        print(f"  → SKIP Shadow A retrain. Move to Thread 2 Step A (distribution model).")
    elif wr_gap > 3:
        print(f"  → Disagree legs underperform by {wr_gap:.1f}pp. Bug has a real cost.")
        print(f"  → But distribution model fixes this natively. Retrain is low-priority.")
    else:
        print(f"  → Disagree legs OUTPERFORM by {abs(wr_gap):.1f}pp. Bug may not hurt as expected.")


if __name__ == "__main__":
    main()
