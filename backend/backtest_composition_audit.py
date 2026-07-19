"""
backtest_composition_audit.py — Pre-build validation: home/away × fav/dog × rung splits.

IMPORTANT: "Favorite" = team with MONEYLINE implied prob > 50%
(i.e., h2h price < 2.0). NOT determined by spread sign.
A favorite can have a positive spread (e.g. SEA +2.5 at -670).
"""
from __future__ import annotations
import os, sys, sqlite3, math, json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def wilson_ci(n: int, w: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI, returns (lo%, hi%)."""
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round((centre - margin) * 100, 1), round((centre + margin) * 100, 1))


def wr_str(w: int, n: int) -> str:
    if n == 0:
        return "  —  "
    return f"{w/n*100:5.1f}%"


def _load_ml_favorites(start: str, end: str) -> dict[tuple, bool]:
    """Load moneyline-based favorite classification from fixtures.bookmakers.

    Returns dict of (home_team, away_team, game_date) → home_is_favorite (bool).
    Favorite = team with h2h price < 2.0 (implied prob > 50%).
    """
    db_path = os.path.join(_HERE, "..", "data", "bets.db")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date_ct, bookmakers
        FROM fixtures
        WHERE sport_key = 'baseball_mlb'
          AND bookmakers IS NOT NULL
          AND game_date_ct BETWEEN ? AND ?
    """, (start, end)).fetchall()
    conn.close()

    result = {}
    for home, away, gd, bk_json in rows:
        if not bk_json:
            continue
        try:
            bks = json.loads(bk_json)
        except (json.JSONDecodeError, TypeError):
            continue

        # Find h2h market, prefer fanduel → draftkings → any
        home_price = None
        for pref_book in ["fanduel", "draftkings", "betmgm", "pinnacle", None]:
            for bk in bks:
                if pref_book is not None and bk.get("key") != pref_book:
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
            # home_price < 2.0 → implied prob > 50% → home is favorite
            result[(home, away, gd)] = home_price < 2.0
        # else: can't determine, will be excluded

    return result


def run_backtest_raw(start: str, end: str, ml_favs: dict) -> list[dict]:
    """Run backtest and return raw selected legs with ML-based fav/dog."""
    from walk_forward_backtest import (
        _load_model_artifacts, _load_ats_submodel, _predict_ats,
        _load_scores, _load_alt_line_odds, _compute_los, _compute_lqs,
        _settle_spread, SPREAD_UNIVERSE, ATS_MODEL_GATE, ALT_MIN_IMPLIED,
        LOS_FLOOR, MAX_LEGS_PER_FIXTURE,
    )
    from features import build_feature_matrix

    clf, sc, imp, fcols = _load_model_artifacts()
    ats_clf, ats_sc, ats_imp, ats_cols = _load_ats_submodel()
    matrix = build_feature_matrix("MLB")
    matrix["_gd"] = matrix["game_date"].astype(str).str[:10]
    bt = matrix[(matrix["_gd"] >= start) & (matrix["_gd"] <= end)]
    scores = _load_scores(start, end)
    spread_wp = _load_alt_line_odds(start, end)

    LIVE_RUNG_RANGE = {1.5, 2.5, 3.5, 4.5}
    all_selected = []

    dates = sorted(bt["_gd"].unique())
    for gd in dates:
        day_games = bt[bt["_gd"] == gd]
        day_cands = []

        for idx, row in day_games.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            score_key = (home, away, gd)
            if score_key not in scores:
                continue
            hs, as_ = scores[score_key]

            # Moneyline-based favorite (from fixtures.bookmakers h2h)
            home_is_fav = ml_favs.get((home, away, gd))
            if home_is_fav is None:
                # Can't determine → skip fav/dog classification
                home_is_fav = None  # will mark as "unknown"

            fix_cands = []
            ats_preds = {}
            for sv in SPREAD_UNIVERSE:
                p = _predict_ats(ats_clf, ats_sc, ats_imp, ats_cols, row, sv)
                if p is not None:
                    ats_preds[sv] = p

            for sv in SPREAD_UNIVERSE:
                if sv not in ats_preds or sv not in LIVE_RUNG_RANGE:
                    continue
                ats_home = ats_preds[sv]
                mkt_home = spread_wp.get((home, away, gd, sv))
                if mkt_home is None:
                    continue

                # Home side
                home_wp = ats_home
                if home_wp >= ATS_MODEL_GATE and mkt_home >= ALT_MIN_IMPLIED:
                    edge = (home_wp - mkt_home) * 100.0
                    lqs = _compute_lqs("Alt Spread", edge, line=sv)
                    los = _compute_los(home_wp, edge_pp=edge, lqs=lqs,
                                       cushion=(home_wp >= 0.70))
                    if los >= LOS_FLOOR:
                        result = _settle_spread(hs, as_, sv, "home")
                        # Pick is home team → pick_is_fav = home_is_fav
                        pick_is_fav = home_is_fav
                        fix_cands.append({
                            "game_date": gd, "home_team": home, "away_team": away,
                            "spread_value": sv, "pick_side": "home",
                            "pick_team": home, "model_wp": home_wp,
                            "los": los, "actual_result": result,
                            "is_pick_favorite": pick_is_fav,
                        })

                # Away side
                neg_sv = -sv
                if neg_sv in ats_preds:
                    away_wp = 1.0 - ats_preds[neg_sv]
                    mkt_neg = spread_wp.get((home, away, gd, neg_sv))
                    mkt_away = (1.0 - mkt_neg) if mkt_neg is not None else (1.0 - mkt_home)

                    if away_wp >= ATS_MODEL_GATE and mkt_away >= ALT_MIN_IMPLIED:
                        edge = (away_wp - mkt_away) * 100.0
                        lqs = _compute_lqs("Alt Spread", edge, line=sv)
                        los = _compute_los(away_wp, edge_pp=edge, lqs=lqs,
                                           cushion=(away_wp >= 0.70))
                        if los >= LOS_FLOOR:
                            result = _settle_spread(hs, as_, sv, "away")
                            # Pick is away team → pick_is_fav = NOT home_is_fav
                            pick_is_fav = (not home_is_fav) if home_is_fav is not None else None
                            fix_cands.append({
                                "game_date": gd, "home_team": home, "away_team": away,
                                "spread_value": sv, "pick_side": "away",
                                "pick_team": away, "model_wp": away_wp,
                                "los": los, "actual_result": result,
                                "is_pick_favorite": pick_is_fav,
                            })

            fix_cands.sort(key=lambda c: -c["los"])
            day_cands.extend(fix_cands[:MAX_LEGS_PER_FIXTURE])

        day_cands.sort(key=lambda c: -c["los"])
        all_selected.extend(day_cands)

    return [s for s in all_selected if s["actual_result"] in ("WIN", "LOSS")]


def analyze_window(legs: list[dict], label: str):
    """Produce all splits for a window."""
    RUNGS = [1.5, 2.5, 3.5, 4.5]
    total = len(legs)
    total_wins = sum(1 for l in legs if l["actual_result"] == "WIN")

    # Filter legs with known fav/dog classification
    legs_classified = [l for l in legs if l.get("is_pick_favorite") is not None]
    n_unclassified = total - len(legs_classified)

    print(f"\n{'='*72}")
    print(f"  WINDOW: {label}")
    print(f"  {total} settled legs, WR={wr_str(total_wins, total)}")
    if n_unclassified:
        print(f"  ({n_unclassified} legs without ML data — excluded from fav/dog splits)")
    print(f"{'='*72}")

    # ── 1. HOME/AWAY SPLIT BY RUNG ───────────────────────────────────
    print(f"\n  1. HOME/AWAY SPLIT BY RUNG")
    print(f"  {'Rung':>6}  {'Home N':>7} {'Home WR':>8}  {'Away N':>7} {'Away WR':>8}  {'Gap':>6}  {'Home%':>6}")
    print(f"  {'-'*62}")

    for rung in RUNGS:
        rung_legs = [l for l in legs if l["spread_value"] == rung]
        h = [l for l in rung_legs if l["pick_side"] == "home"]
        a = [l for l in rung_legs if l["pick_side"] == "away"]
        hw = sum(1 for l in h if l["actual_result"] == "WIN")
        aw = sum(1 for l in a if l["actual_result"] == "WIN")
        h_wr = hw / len(h) * 100 if h else 0
        a_wr = aw / len(a) * 100 if a else 0
        gap = h_wr - a_wr if h and a else 0
        h_pct = len(h) / len(rung_legs) * 100 if rung_legs else 0
        print(f"  {rung:>+5.1f}  {len(h):>7} {wr_str(hw, len(h)):>8}  "
              f"{len(a):>7} {wr_str(aw, len(a)):>8}  {gap:>+5.1f}  {h_pct:>5.1f}%")

    h_all = [l for l in legs if l["pick_side"] == "home"]
    a_all = [l for l in legs if l["pick_side"] == "away"]
    hw_all = sum(1 for l in h_all if l["actual_result"] == "WIN")
    aw_all = sum(1 for l in a_all if l["actual_result"] == "WIN")
    h_wr_all = hw_all / len(h_all) * 100 if h_all else 0
    a_wr_all = aw_all / len(a_all) * 100 if a_all else 0
    gap_all = h_wr_all - a_wr_all if h_all and a_all else 0
    h_pct_all = len(h_all) / total * 100 if total else 0
    print(f"  {'ALL':>6}  {len(h_all):>7} {wr_str(hw_all, len(h_all)):>8}  "
          f"{len(a_all):>7} {wr_str(aw_all, len(a_all)):>8}  {gap_all:>+5.1f}  {h_pct_all:>5.1f}%")
    h_lo, h_hi = wilson_ci(len(h_all), hw_all)
    a_lo, a_hi = wilson_ci(len(a_all), aw_all)
    print(f"         CI: [{h_lo}%, {h_hi}%]       CI: [{a_lo}%, {a_hi}%]")

    # ── 2. FAVORITE/UNDERDOG SPLIT BY RUNG (ML-based) ────────────────
    print(f"\n  2. FAVORITE/UNDERDOG SPLIT BY RUNG (moneyline-based)")
    print(f"  {'Rung':>6}  {'Fav N':>7} {'Fav WR':>8}  {'Dog N':>7} {'Dog WR':>8}  {'Gap':>6}  {'Fav%':>6}")
    print(f"  {'-'*62}")

    for rung in RUNGS:
        rung_legs = [l for l in legs_classified if l["spread_value"] == rung]
        f = [l for l in rung_legs if l["is_pick_favorite"]]
        d = [l for l in rung_legs if not l["is_pick_favorite"]]
        fw = sum(1 for l in f if l["actual_result"] == "WIN")
        dw = sum(1 for l in d if l["actual_result"] == "WIN")
        f_wr = fw / len(f) * 100 if f else 0
        d_wr = dw / len(d) * 100 if d else 0
        gap = f_wr - d_wr if f and d else 0
        f_pct = len(f) / len(rung_legs) * 100 if rung_legs else 0
        print(f"  {rung:>+5.1f}  {len(f):>7} {wr_str(fw, len(f)):>8}  "
              f"{len(d):>7} {wr_str(dw, len(d)):>8}  {gap:>+5.1f}  {f_pct:>5.1f}%")

    f_all = [l for l in legs_classified if l["is_pick_favorite"]]
    d_all = [l for l in legs_classified if not l["is_pick_favorite"]]
    fw_all = sum(1 for l in f_all if l["actual_result"] == "WIN")
    dw_all = sum(1 for l in d_all if l["actual_result"] == "WIN")
    f_wr_all = fw_all / len(f_all) * 100 if f_all else 0
    d_wr_all = dw_all / len(d_all) * 100 if d_all else 0
    gap_all = f_wr_all - d_wr_all if f_all and d_all else 0
    f_pct_all = len(f_all) / len(legs_classified) * 100 if legs_classified else 0
    print(f"  {'ALL':>6}  {len(f_all):>7} {wr_str(fw_all, len(f_all)):>8}  "
          f"{len(d_all):>7} {wr_str(dw_all, len(d_all)):>8}  {gap_all:>+5.1f}  {f_pct_all:>5.1f}%")
    f_lo, f_hi = wilson_ci(len(f_all), fw_all)
    d_lo, d_hi = wilson_ci(len(d_all), dw_all)
    print(f"         CI: [{f_lo}%, {f_hi}%]       CI: [{d_lo}%, {d_hi}%]")

    # ── 3. THE 2×2 MATRIX ────────────────────────────────────────────
    print(f"\n  3. 2×2 MATRIX (fav/dog × home/away) — moneyline-based")

    cells = {
        "Fav:Home": [], "Fav:Away": [],
        "Dog:Home": [], "Dog:Away": [],
    }
    for l in legs_classified:
        fav = "Fav" if l["is_pick_favorite"] else "Dog"
        side = "Home" if l["pick_side"] == "home" else "Away"
        cells[f"{fav}:{side}"].append(l)

    n_classified = len(legs_classified)
    print(f"\n  Overall (n={n_classified}):")
    print(f"  {'':>12} {'Home':>22} {'Away':>22}")
    print(f"  {'-'*58}")
    for fav_label in ("Fav", "Dog"):
        h_cell = cells[f"{fav_label}:Home"]
        a_cell = cells[f"{fav_label}:Away"]
        h_w = sum(1 for l in h_cell if l["actual_result"] == "WIN")
        a_w = sum(1 for l in a_cell if l["actual_result"] == "WIN")
        h_pct = len(h_cell) / n_classified * 100 if n_classified else 0
        a_pct = len(a_cell) / n_classified * 100 if n_classified else 0
        print(f"  {fav_label:>10}  {wr_str(h_w, len(h_cell))} n={len(h_cell):>4} ({h_pct:4.1f}%)  "
              f"{wr_str(a_w, len(a_cell))} n={len(a_cell):>4} ({a_pct:4.1f}%)")

    # Per-rung matrix
    for rung in RUNGS:
        rung_legs = [l for l in legs_classified if l["spread_value"] == rung]
        if not rung_legs:
            continue
        rn = len(rung_legs)
        rung_wins = sum(1 for l in rung_legs if l["actual_result"] == "WIN")
        print(f"\n  Rung {rung:+.1f} (n={rn}, WR={wr_str(rung_wins, rn)}):")
        print(f"  {'':>12} {'Home':>22} {'Away':>22}")
        print(f"  {'-'*58}")
        for fav_label in ("Fav", "Dog"):
            h_cell = [l for l in rung_legs if l["is_pick_favorite"] == (fav_label == "Fav")
                       and l["pick_side"] == "home"]
            a_cell = [l for l in rung_legs if l["is_pick_favorite"] == (fav_label == "Fav")
                       and l["pick_side"] == "away"]
            h_w = sum(1 for l in h_cell if l["actual_result"] == "WIN")
            a_w = sum(1 for l in a_cell if l["actual_result"] == "WIN")
            h_pct = len(h_cell) / rn * 100 if rn else 0
            a_pct = len(a_cell) / rn * 100 if rn else 0
            print(f"  {fav_label:>10}  {wr_str(h_w, len(h_cell))} n={len(h_cell):>4} ({h_pct:4.1f}%)  "
                  f"{wr_str(a_w, len(a_cell))} n={len(a_cell):>4} ({a_pct:4.1f}%)")


def analyze_live(start: str, end: str, label: str, ml_favs: dict):
    """Produce the 2×2 matrix from live mock_bet_legs data using ML-based fav/dog."""
    db_path = os.path.join(_HERE, "..", "data", "bets.db")
    conn = sqlite3.connect(db_path)

    rows = conn.execute("""
        SELECT
            mbl.spread_value,
            mbl.description,
            mbl.leg_result,
            mbl.predicted_win_prob,
            mb.game_date,
            mbl.resolved_home_team,
            mbl.resolved_away_team
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN','LOSS')
          AND mbl.market_type = 'alternate_spreads'
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.source IN ('prospective', 'prospective_pm')
          AND mb.game_date BETWEEN ? AND ?
          AND mbl.spread_value IS NOT NULL
    """, (start, end)).fetchall()
    conn.close()

    if not rows:
        print(f"\n  4. LIVE 2×2 MATRIX — {label}: No data")
        return

    legs = []
    n_no_ml = 0
    for sv, desc, result, wp, gd, home, away in rows:
        sv = float(sv)
        desc_clean = (desc or "").strip()
        pick_side = None

        if home and away:
            home_l = home.lower().strip()
            away_l = away.lower().strip()
            desc_l = desc_clean.lower()
            if home_l in desc_l:
                pick_side = "home"
            elif away_l in desc_l:
                pick_side = "away"

        if pick_side is None:
            continue

        # ML-based favorite from fixtures.bookmakers
        home_is_fav = ml_favs.get((home, away, gd))
        if home_is_fav is None:
            n_no_ml += 1
            continue

        is_pick_fav = home_is_fav if pick_side == "home" else not home_is_fav

        legs.append({
            "spread_value": abs(sv),  # normalize to positive rung
            "pick_side": pick_side,
            "is_pick_favorite": is_pick_fav,
            "actual_result": result,
            "game_date": gd,
        })

    total = len(legs)
    total_wins = sum(1 for l in legs if l["actual_result"] == "WIN")

    print(f"\n{'='*72}")
    print(f"  4. LIVE 2×2 MATRIX — {label}")
    print(f"  {total} classified legs, WR={wr_str(total_wins, total)}")
    if n_no_ml:
        print(f"  ({n_no_ml} legs excluded — no ML data for fixture)")
    print(f"{'='*72}")

    RUNGS = [1.5, 2.5, 3.5, 4.5]

    # Overall matrix
    cells = {"Fav:Home": [], "Fav:Away": [], "Dog:Home": [], "Dog:Away": []}
    for l in legs:
        fav = "Fav" if l["is_pick_favorite"] else "Dog"
        side = "Home" if l["pick_side"] == "home" else "Away"
        cells[f"{fav}:{side}"].append(l)

    print(f"\n  Overall:")
    print(f"  {'':>12} {'Home':>22} {'Away':>22}")
    print(f"  {'-'*58}")
    for fav_label in ("Fav", "Dog"):
        h_cell = cells[f"{fav_label}:Home"]
        a_cell = cells[f"{fav_label}:Away"]
        h_w = sum(1 for l in h_cell if l["actual_result"] == "WIN")
        a_w = sum(1 for l in a_cell if l["actual_result"] == "WIN")
        h_pct = len(h_cell) / total * 100 if total else 0
        a_pct = len(a_cell) / total * 100 if total else 0
        print(f"  {fav_label:>10}  {wr_str(h_w, len(h_cell))} n={len(h_cell):>4} ({h_pct:4.1f}%)  "
              f"{wr_str(a_w, len(a_cell))} n={len(a_cell):>4} ({a_pct:4.1f}%)")

    # Marginals
    f_all = [l for l in legs if l["is_pick_favorite"]]
    d_all = [l for l in legs if not l["is_pick_favorite"]]
    h_all = [l for l in legs if l["pick_side"] == "home"]
    a_all = [l for l in legs if l["pick_side"] == "away"]
    print(f"\n  Marginals:")
    print(f"    Fav:  n={len(f_all):>4}  ({len(f_all)/total*100:.1f}%)  "
          f"WR={wr_str(sum(1 for l in f_all if l['actual_result']=='WIN'), len(f_all))}")
    print(f"    Dog:  n={len(d_all):>4}  ({len(d_all)/total*100:.1f}%)  "
          f"WR={wr_str(sum(1 for l in d_all if l['actual_result']=='WIN'), len(d_all))}")
    print(f"    Home: n={len(h_all):>4}  ({len(h_all)/total*100:.1f}%)  "
          f"WR={wr_str(sum(1 for l in h_all if l['actual_result']=='WIN'), len(h_all))}")
    print(f"    Away: n={len(a_all):>4}  ({len(a_all)/total*100:.1f}%)  "
          f"WR={wr_str(sum(1 for l in a_all if l['actual_result']=='WIN'), len(a_all))}")

    # Per-rung
    for rung in RUNGS:
        rung_legs = [l for l in legs if l["spread_value"] == rung]
        if not rung_legs:
            continue
        rn = len(rung_legs)
        rung_wins = sum(1 for l in rung_legs if l["actual_result"] == "WIN")
        print(f"\n  Rung ±{rung:.1f} (n={rn}, WR={wr_str(rung_wins, rn)}):")
        print(f"  {'':>12} {'Home':>22} {'Away':>22}")
        print(f"  {'-'*58}")
        for fav_label in ("Fav", "Dog"):
            h_cell = [l for l in rung_legs if l["is_pick_favorite"] == (fav_label == "Fav")
                       and l["pick_side"] == "home"]
            a_cell = [l for l in rung_legs if l["is_pick_favorite"] == (fav_label == "Fav")
                       and l["pick_side"] == "away"]
            h_w = sum(1 for l in h_cell if l["actual_result"] == "WIN")
            a_w = sum(1 for l in a_cell if l["actual_result"] == "WIN")
            h_pct = len(h_cell) / rn * 100 if rn else 0
            a_pct = len(a_cell) / rn * 100 if rn else 0
            print(f"  {fav_label:>10}  {wr_str(h_w, len(h_cell))} n={len(h_cell):>4} ({h_pct:4.1f}%)  "
                  f"{wr_str(a_w, len(a_cell))} n={len(a_cell):>4} ({a_pct:4.1f}%)")


def main():
    FULL_START, FULL_END = "2026-04-22", "2026-06-27"
    LAST30_START, LAST30_END = "2026-05-28", "2026-06-27"

    # Load moneyline-based favorites for ALL fixtures in the range
    print("Loading moneyline favorites from fixtures.bookmakers...")
    ml_favs = _load_ml_favorites(FULL_START, FULL_END)
    print(f"  → {len(ml_favs)} fixtures with ML classification")

    # Sanity check: what % are home favorites?
    n_home_fav = sum(1 for v in ml_favs.values() if v)
    print(f"  → {n_home_fav}/{len(ml_favs)} ({n_home_fav/len(ml_favs)*100:.1f}%) home favorites")

    print("\nLoading full-period backtest...")
    legs_full = run_backtest_raw(FULL_START, FULL_END, ml_favs)
    print(f"  → {len(legs_full)} settled legs")

    legs_30d = [l for l in legs_full if l["game_date"] >= LAST30_START]
    print(f"  → {len(legs_30d)} settled legs in last 30d\n")

    # ── BACKTEST WINDOWS ──────────────────────────────────────────────
    analyze_window(legs_full, f"FULL PERIOD ({FULL_START} to {FULL_END})")
    analyze_window(legs_30d, f"LAST 30 DAYS ({LAST30_START} to {LAST30_END})")

    # ── LIVE WINDOWS ──────────────────────────────────────────────────
    analyze_live(FULL_START, FULL_END, f"FULL PERIOD ({FULL_START} to {FULL_END})", ml_favs)
    analyze_live(LAST30_START, LAST30_END, f"LAST 30 DAYS ({LAST30_START} to {LAST30_END})", ml_favs)

    # ── STABILITY CHECK ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  5. STABILITY CHECK: Full vs Last-30d")
    print(f"{'='*72}")

    def _cell_stats(legs_list, side, fav):
        cell = [l for l in legs_list
                if l["pick_side"] == side
                and l.get("is_pick_favorite") == fav]
        w = sum(1 for l in cell if l["actual_result"] == "WIN")
        return len(cell), w

    print(f"\n  Backtest 2×2 WR comparison:")
    print(f"  {'Cell':>15} {'Full WR':>10} {'Full N':>7}  {'30d WR':>10} {'30d N':>7}  {'Drift':>7}")
    print(f"  {'-'*62}")
    for label, side, fav in [("Fav:Home", "home", True), ("Fav:Away", "away", True),
                              ("Dog:Home", "home", False), ("Dog:Away", "away", False)]:
        fn, fw = _cell_stats(legs_full, side, fav)
        tn, tw = _cell_stats(legs_30d, side, fav)
        f_wr = fw / fn * 100 if fn else 0
        t_wr = tw / tn * 100 if tn else 0
        drift = t_wr - f_wr if fn and tn else 0
        print(f"  {label:>15} {wr_str(fw, fn):>10} {fn:>7}  {wr_str(tw, tn):>10} {tn:>7}  {drift:>+6.1f}pp")

    # Per-rung WR stability
    RUNGS = [1.5, 2.5, 3.5, 4.5]
    print(f"\n  Per-rung WR stability:")
    print(f"  {'Rung':>6} {'Full WR':>10} {'Full N':>7}  {'30d WR':>10} {'30d N':>7}  {'Drift':>7}")
    print(f"  {'-'*55}")
    for rung in RUNGS:
        f_legs = [l for l in legs_full if l["spread_value"] == rung]
        t_legs = [l for l in legs_30d if l["spread_value"] == rung]
        fw = sum(1 for l in f_legs if l["actual_result"] == "WIN")
        tw = sum(1 for l in t_legs if l["actual_result"] == "WIN")
        f_wr = fw / len(f_legs) * 100 if f_legs else 0
        t_wr = tw / len(t_legs) * 100 if t_legs else 0
        drift = t_wr - f_wr if f_legs and t_legs else 0
        print(f"  {rung:>+5.1f} {wr_str(fw, len(f_legs)):>10} {len(f_legs):>7}  "
              f"{wr_str(tw, len(t_legs)):>10} {len(t_legs):>7}  {drift:>+6.1f}pp")

    # Fav% stability per rung
    print(f"\n  Favorite pick % stability (per rung):")
    for rung in RUNGS:
        f_legs = [l for l in legs_full if l["spread_value"] == rung and l.get("is_pick_favorite") is not None]
        t_legs = [l for l in legs_30d if l["spread_value"] == rung and l.get("is_pick_favorite") is not None]
        f_fav = len([l for l in f_legs if l["is_pick_favorite"]])
        t_fav = len([l for l in t_legs if l["is_pick_favorite"]])
        f_pct = f_fav / len(f_legs) * 100 if f_legs else 0
        t_pct = t_fav / len(t_legs) * 100 if t_legs else 0
        print(f"    Rung {rung:+.1f}: Full={f_pct:.1f}% ({f_fav}/{len(f_legs)})  "
              f"30d={t_pct:.1f}% ({t_fav}/{len(t_legs)})  drift={t_pct-f_pct:+.1f}pp")

    print()

    # ── PRIOR ANALYSIS CROSS-CHECK ───────────────────────────────────
    print(f"{'='*72}")
    print(f"  6. PRIOR ANALYSIS CROSS-CHECK")
    print(f"{'='*72}")
    print(f"""
  Previous fav/dog analyses used SPREAD-SIGN classification:
    is_favorite = (spread_value < 0)

  This is WRONG. A positive spread can be a favorite pick
  (e.g. SEA +2.5 at -670 when SEA is ML favorite at -162).

  Affected analyses:
    1. walk_forward_backtest.py: "is_favorite": sv < 0
       → All fav/dog breakdowns in backtest reports were wrong
    2. mock_bets.py _parse_spread_for_leg(): is_favorite_spread = 1 if sv < 0
       → The is_favorite_spread column in mock_bet_legs is wrong
    3. Previous "96% underdog" finding was an artifact of this bug

  The correct definition used in THIS audit:
    favorite = team with h2h/moneyline decimal price < 2.0
    (from fixtures.bookmakers JSON)
""")


if __name__ == "__main__":
    main()
