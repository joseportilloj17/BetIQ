"""
pnl_quadrant_audit.py — P&L by quadrant (fav/dog × home/away) for last 30 days.

Uses REAL odds (not just WR):
  - Backtest: alt-spread odds from historical.db (avg across books)
  - Live: open_odds from mock_bet_legs (American format)

Flat $10 stake per leg.  P&L = (decimal_odds - 1) * 10 on WIN, -10 on LOSS.
"""
from __future__ import annotations
import os, sys, sqlite3, json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

STAKE = 10.0
LAST30_START = "2026-05-28"
LAST30_END   = "2026-06-27"


def _american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    return 2.0  # fallback


def _load_ml_favorites(start: str, end: str) -> dict[tuple, bool]:
    """Moneyline-based favorite: h2h price < 2.0 → home is favorite."""
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
            result[(home, away, gd)] = home_price < 2.0
    return result


def _load_alt_line_odds_raw(start: str, end: str) -> dict:
    """Load raw decimal odds for alt-spreads from historical.db.

    Returns dict of (home, away, date, team, spread_value) → avg decimal odds.
    """
    hist_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(hist_path)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date, team, line, odds, bookmaker
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND game_date BETWEEN ? AND ?
          AND odds IS NOT NULL AND odds > 1.0
    """, (start, end)).fetchall()
    conn.close()

    raw = defaultdict(list)
    for home, away, gd, team, line, odds, book in rows:
        # Normalize game_date to YYYY-MM-DD
        gd_str = str(gd)[:10]
        key = (home, away, gd_str, team, float(line))
        raw[key].append(float(odds))

    result = {}
    for key, odds_list in raw.items():
        result[key] = sum(odds_list) / len(odds_list)
    return result


def run_backtest_pnl():
    """Run backtest for last 30d, attach real odds, compute P&L per quadrant."""
    from walk_forward_backtest import (
        _load_model_artifacts, _load_ats_submodel, _predict_ats,
        _load_scores, _load_alt_line_odds, _compute_los, _compute_lqs,
        _settle_spread, SPREAD_UNIVERSE, ATS_MODEL_GATE, ALT_MIN_IMPLIED,
        LOS_FLOOR, MAX_LEGS_PER_FIXTURE,
    )
    from features import build_feature_matrix

    start, end = LAST30_START, LAST30_END

    print("Loading ML favorites...")
    ml_favs = _load_ml_favorites(start, end)
    print(f"  {len(ml_favs)} fixtures classified")

    print("Loading alt-line odds (raw decimal)...")
    odds_raw = _load_alt_line_odds_raw(start, end)
    print(f"  {len(odds_raw)} team/spread/date odds entries")

    print("Loading model artifacts...")
    clf, sc, imp, fcols = _load_model_artifacts()
    ats_clf, ats_sc, ats_imp, ats_cols = _load_ats_submodel()

    print("Building feature matrix...")
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
            home_is_fav = ml_favs.get((home, away, gd))

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

                # Home side at +sv
                home_wp = ats_home
                if home_wp >= ATS_MODEL_GATE and mkt_home >= ALT_MIN_IMPLIED:
                    edge = (home_wp - mkt_home) * 100.0
                    lqs = _compute_lqs("Alt Spread", edge, line=sv)
                    los = _compute_los(home_wp, edge_pp=edge, lqs=lqs,
                                       cushion=(home_wp >= 0.70))
                    if los >= LOS_FLOOR:
                        result = _settle_spread(hs, as_, sv, "home")
                        # Look up actual odds for home team at +sv
                        dec_odds = odds_raw.get((home, away, gd, home, sv))
                        pick_is_fav = home_is_fav if home_is_fav is not None else None
                        fix_cands.append({
                            "game_date": gd, "home_team": home, "away_team": away,
                            "spread_value": sv, "pick_side": "home",
                            "pick_team": home, "model_wp": home_wp,
                            "market_wp": mkt_home,
                            "los": los, "actual_result": result,
                            "is_pick_favorite": pick_is_fav,
                            "decimal_odds": dec_odds,
                        })

                # Away side at +sv
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
                            # Away team at +sv: look up odds for away team
                            dec_odds = odds_raw.get((home, away, gd, away, sv))
                            pick_is_fav = (not home_is_fav) if home_is_fav is not None else None
                            fix_cands.append({
                                "game_date": gd, "home_team": home, "away_team": away,
                                "spread_value": sv, "pick_side": "away",
                                "pick_team": away, "model_wp": away_wp,
                                "market_wp": mkt_away,
                                "los": los, "actual_result": result,
                                "is_pick_favorite": pick_is_fav,
                                "decimal_odds": dec_odds,
                            })

            fix_cands.sort(key=lambda c: -c["los"])
            day_cands.extend(fix_cands[:MAX_LEGS_PER_FIXTURE])

        day_cands.sort(key=lambda c: -c["los"])
        all_selected.extend(day_cands)

    settled = [s for s in all_selected if s["actual_result"] in ("WIN", "LOSS")]
    return settled


def _compute_pnl(legs: list[dict], label: str, use_fallback_odds: bool = True):
    """Compute P&L stats for a group of legs at flat $STAKE."""
    n = len(legs)
    if n == 0:
        return {"n": 0, "wr": 0, "pnl": 0, "roi": 0, "avg_odds": 0, "no_odds": 0}

    wins = 0
    pnl = 0.0
    total_odds = 0.0
    n_with_odds = 0
    n_no_odds = 0

    for l in legs:
        dec = l.get("decimal_odds")
        if dec is None and use_fallback_odds:
            # Fallback: derive from market_wp (1/market_wp ≈ decimal odds with vig)
            mkt_wp = l.get("market_wp")
            if mkt_wp and mkt_wp > 0:
                dec = 1.0 / mkt_wp
            else:
                dec = None

        is_win = l["actual_result"] == "WIN"
        if is_win:
            wins += 1

        if dec is not None and dec > 1.0:
            n_with_odds += 1
            total_odds += dec
            if is_win:
                pnl += (dec - 1.0) * STAKE
            else:
                pnl -= STAKE
        else:
            n_no_odds += 1
            # Can't compute P&L without odds

    staked = n_with_odds * STAKE
    roi = (pnl / staked * 100) if staked > 0 else 0
    avg_odds = total_odds / n_with_odds if n_with_odds > 0 else 0

    return {
        "n": n,
        "n_priced": n_with_odds,
        "no_odds": n_no_odds,
        "wins": wins,
        "wr": wins / n * 100 if n > 0 else 0,
        "pnl": pnl,
        "staked": staked,
        "roi": roi,
        "avg_dec_odds": avg_odds,
    }


def _print_quadrant_table(cells: dict, label: str):
    """Print P&L table for all 4 quadrants."""
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")

    header = (f"  {'Quadrant':<12} {'N':>5} {'WR':>7} {'Avg Odds':>10} "
              f"{'Staked':>8} {'P&L':>9} {'ROI':>8}  {'No Odds':>8}")
    print(header)
    print(f"  {'-'*86}")

    total_pnl = 0.0
    total_staked = 0.0
    total_n = 0
    total_wins = 0

    for cell_name in ["Fav:Home", "Fav:Away", "Dog:Home", "Dog:Away"]:
        s = _compute_pnl(cells[cell_name], cell_name)
        total_pnl += s["pnl"]
        total_staked += s["staked"]
        total_n += s["n"]
        total_wins += s["wins"]
        wr_s = f"{s['wr']:5.1f}%" if s["n"] > 0 else "   — "
        odds_s = f"{s['avg_dec_odds']:.3f}" if s["n_priced"] > 0 else "  —"
        print(f"  {cell_name:<12} {s['n']:>5} {wr_s:>7} {odds_s:>10} "
              f"  ${s['staked']:>6.0f} ${s['pnl']:>+8.2f} {s['roi']:>+7.1f}%  "
              f"{s['no_odds']:>5}")

    # Totals
    total_roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
    total_wr = total_wins / total_n * 100 if total_n > 0 else 0
    print(f"  {'-'*86}")
    print(f"  {'TOTAL':<12} {total_n:>5} {total_wr:>5.1f}% {'':>10} "
          f"  ${total_staked:>6.0f} ${total_pnl:>+8.2f} {total_roi:>+7.1f}%")

    # Marginal P&L: Fav vs Dog
    fav_s = _compute_pnl(cells["Fav:Home"] + cells["Fav:Away"], "Fav total")
    dog_s = _compute_pnl(cells["Dog:Home"] + cells["Dog:Away"], "Dog total")
    print(f"\n  Marginal P&L:")
    print(f"    Fav total: n={fav_s['n']:>4}  WR={fav_s['wr']:5.1f}%  "
          f"avg_odds={fav_s['avg_dec_odds']:.3f}  P&L=${fav_s['pnl']:>+8.2f}  "
          f"ROI={fav_s['roi']:>+6.1f}%")
    print(f"    Dog total: n={dog_s['n']:>4}  WR={dog_s['wr']:5.1f}%  "
          f"avg_odds={dog_s['avg_dec_odds']:.3f}  P&L=${dog_s['pnl']:>+8.2f}  "
          f"ROI={dog_s['roi']:>+6.1f}%")


def _print_per_rung_quadrant(legs: list[dict], label: str):
    """Per-rung breakdown within each quadrant."""
    RUNGS = [1.5, 2.5, 3.5, 4.5]

    print(f"\n  Per-rung P&L within quadrants ({label}):")
    print(f"  {'Rung':<6} {'Quadrant':<12} {'N':>5} {'WR':>7} {'Avg Odds':>10} "
          f"{'P&L':>9} {'ROI':>8}")
    print(f"  {'-'*64}")

    for rung in RUNGS:
        rung_legs = [l for l in legs if l["spread_value"] == rung]
        if not rung_legs:
            continue

        for cell_name, fav_val, side_val in [
            ("Fav:Home", True, "home"), ("Fav:Away", True, "away"),
            ("Dog:Home", False, "home"), ("Dog:Away", False, "away"),
        ]:
            cell_legs = [l for l in rung_legs
                         if l.get("is_pick_favorite") == fav_val
                         and l["pick_side"] == side_val]
            if not cell_legs:
                continue
            s = _compute_pnl(cell_legs, f"rung{rung}_{cell_name}")
            odds_s = f"{s['avg_dec_odds']:.3f}" if s["n_priced"] > 0 else "  —"
            print(f"  {rung:>+5.1f} {cell_name:<12} {s['n']:>5} {s['wr']:>5.1f}% "
                  f"{odds_s:>10} ${s['pnl']:>+8.2f} {s['roi']:>+7.1f}%")
        print()  # blank line between rungs


def run_live_pnl():
    """Load live mock_bet_legs with open_odds, compute P&L per quadrant."""
    ml_favs = _load_ml_favorites(LAST30_START, LAST30_END)

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
            mbl.resolved_away_team,
            mbl.open_odds
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN','LOSS')
          AND mbl.market_type = 'alternate_spreads'
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.source IN ('prospective', 'prospective_pm')
          AND mb.game_date BETWEEN ? AND ?
          AND mbl.spread_value IS NOT NULL
    """, (LAST30_START, LAST30_END)).fetchall()
    conn.close()

    legs = []
    for sv, desc, result, wp, gd, home, away, open_odds in rows:
        sv = float(sv)
        desc_clean = (desc or "").strip()
        pick_side = None

        if home and away:
            if home.lower().strip() in desc_clean.lower():
                pick_side = "home"
            elif away.lower().strip() in desc_clean.lower():
                pick_side = "away"
        if pick_side is None:
            continue

        home_is_fav = ml_favs.get((home, away, gd))
        if home_is_fav is None:
            continue

        is_pick_fav = home_is_fav if pick_side == "home" else not home_is_fav

        dec_odds = None
        if open_odds is not None:
            dec_odds = _american_to_decimal(float(open_odds))

        legs.append({
            "spread_value": abs(sv),
            "pick_side": pick_side,
            "is_pick_favorite": is_pick_fav,
            "actual_result": result,
            "game_date": gd,
            "decimal_odds": dec_odds,
            "market_wp": (1.0 / dec_odds) if dec_odds and dec_odds > 0 else None,
        })

    return legs


def main():
    print("=" * 90)
    print("  P&L QUADRANT AUDIT — LAST 30 DAYS")
    print("  Flat $10 stake per leg.  P&L = (dec_odds-1)*10 on WIN, -10 on LOSS")
    print("=" * 90)

    # ── BACKTEST ──────────────────────────────────────────────────────
    print("\n── BACKTEST (last 30d) ──")
    bt_legs = run_backtest_pnl()
    bt_classified = [l for l in bt_legs if l.get("is_pick_favorite") is not None]
    print(f"\n{len(bt_legs)} settled legs, {len(bt_classified)} with ML classification")

    # Count odds availability
    n_real = sum(1 for l in bt_classified if l.get("decimal_odds") is not None)
    n_fallback = sum(1 for l in bt_classified if l.get("decimal_odds") is None
                     and l.get("market_wp") is not None)
    print(f"Odds: {n_real} from historical.db, {n_fallback} from market_wp fallback")

    cells_bt = {"Fav:Home": [], "Fav:Away": [], "Dog:Home": [], "Dog:Away": []}
    for l in bt_classified:
        fav = "Fav" if l["is_pick_favorite"] else "Dog"
        side = "Home" if l["pick_side"] == "home" else "Away"
        cells_bt[f"{fav}:{side}"].append(l)

    _print_quadrant_table(cells_bt, "BACKTEST P&L — Last 30 Days")
    _print_per_rung_quadrant(bt_classified, "Backtest")

    # ── LIVE ──────────────────────────────────────────────────────────
    print("\n── LIVE (last 30d) ──")
    live_legs = run_live_pnl()
    print(f"{len(live_legs)} classified live legs")

    n_live_odds = sum(1 for l in live_legs if l.get("decimal_odds") is not None)
    print(f"Odds: {n_live_odds}/{len(live_legs)} legs have open_odds")

    cells_live = {"Fav:Home": [], "Fav:Away": [], "Dog:Home": [], "Dog:Away": []}
    for l in live_legs:
        fav = "Fav" if l["is_pick_favorite"] else "Dog"
        side = "Home" if l["pick_side"] == "home" else "Away"
        cells_live[f"{fav}:{side}"].append(l)

    _print_quadrant_table(cells_live, "LIVE P&L — Last 30 Days")
    _print_per_rung_quadrant(live_legs, "Live")

    # ── KEY QUESTION ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  KEY QUESTION: Does the favorite-shift thesis hold on P&L?")
    print("=" * 90)

    fav_bt = _compute_pnl(cells_bt["Fav:Home"] + cells_bt["Fav:Away"], "BT Fav")
    dog_bt = _compute_pnl(cells_bt["Dog:Home"] + cells_bt["Dog:Away"], "BT Dog")
    print(f"\n  Backtest: Fav ROI={fav_bt['roi']:>+6.1f}% (odds={fav_bt['avg_dec_odds']:.3f}) "
          f"  vs  Dog ROI={dog_bt['roi']:>+6.1f}% (odds={dog_bt['avg_dec_odds']:.3f})")

    if live_legs:
        fav_live = _compute_pnl(cells_live["Fav:Home"] + cells_live["Fav:Away"], "Live Fav")
        dog_live = _compute_pnl(cells_live["Dog:Home"] + cells_live["Dog:Away"], "Live Dog")
        print(f"  Live:     Fav ROI={fav_live['roi']:>+6.1f}% (odds={fav_live['avg_dec_odds']:.3f}) "
              f"  vs  Dog ROI={dog_live['roi']:>+6.1f}% (odds={dog_live['avg_dec_odds']:.3f})")

    print(f"\n  If Fav ROI > Dog ROI → thesis holds (shift to favorites = more profit)")
    print(f"  If Fav ROI ≤ Dog ROI → WR advantage eaten by worse odds")


if __name__ == "__main__":
    main()
