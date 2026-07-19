"""
pnl_parlay_audit.py — Parlay-level P&L by favorite-composition.

The betting unit is the PARLAY TICKET (2-4 legs), not individual legs.
A parlay wins only if ALL legs hit.  P&L = (combined_decimal_odds - 1) * stake
on WIN, -stake on LOSS.

Runs BOTH full period (Apr 22 - Jun 26) and last 30 days, then compares.
"""
from __future__ import annotations
import os, sys, sqlite3, json, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

STAKE = 10.0
FULL_START   = "2026-04-22"
FULL_END     = "2026-06-26"
LAST30_START = "2026-05-28"
LAST30_END   = "2026-06-26"


def wilson_ci(n: int, w: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI, returns (lo%, hi%)."""
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round((centre - margin) * 100, 1), round((centre + margin) * 100, 1))


def _american_to_decimal(american: int | float) -> float:
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    return 2.0


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


def _classify_leg(leg: dict, ml_favs: dict) -> dict | None:
    """Classify a leg as Fav/Dog × Home/Away."""
    desc = (leg["description"] or "").strip()
    home = (leg["home"] or "").strip()
    away = (leg["away"] or "").strip()
    sv = leg["spread_value"]

    if not home or not away:
        return None

    pick_side = None
    if home.lower() in desc.lower():
        pick_side = "home"
    elif away.lower() in desc.lower():
        pick_side = "away"
    if pick_side is None:
        return None

    home_is_fav = ml_favs.get((home, away, leg["game_date"]))
    if home_is_fav is None:
        return None

    is_pick_fav = home_is_fav if pick_side == "home" else not home_is_fav

    return {
        **leg,
        "pick_side": pick_side,
        "is_pick_fav": is_pick_fav,
        "quadrant": f"{'Fav' if is_pick_fav else 'Dog'}:{'Home' if pick_side == 'home' else 'Away'}",
        "rung": abs(float(sv)) if sv else None,
    }


def load_parlays(ml_favs: dict, start: str, end: str) -> list[dict]:
    """Load all settled parlays (2+ legs) with their classified legs."""
    db_path = os.path.join(_HERE, "..", "data", "bets.db")
    conn = sqlite3.connect(db_path)

    parlays_raw = conn.execute("""
        SELECT mb.id, mb.legs, mb.odds, mb.actual_profit, mb.bet_type,
               mb.game_date
        FROM mock_bets mb
        WHERE mb.source IN ('prospective','prospective_pm')
          AND mb.game_date BETWEEN ? AND ?
          AND mb.actual_profit IS NOT NULL
          AND mb.bet_type = 'parlay'
          AND mb.legs >= 2
    """, (start, end)).fetchall()

    legs_raw = conn.execute("""
        SELECT mbl.mock_bet_id, mbl.description, mbl.spread_value,
               mbl.open_odds, mbl.leg_result, mbl.market_type,
               mbl.resolved_home_team, mbl.resolved_away_team
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mb.source IN ('prospective','prospective_pm')
          AND mb.game_date BETWEEN ? AND ?
          AND mb.actual_profit IS NOT NULL
          AND mb.bet_type = 'parlay'
          AND mb.legs >= 2
    """, (start, end)).fetchall()
    conn.close()

    # Group legs by parlay
    legs_by_parlay = defaultdict(list)
    for bet_id, desc, sv, open_odds, result, mkt, home, away in legs_raw:
        legs_by_parlay[bet_id].append({
            "description": desc,
            "spread_value": sv,
            "open_odds": open_odds,
            "leg_result": result,
            "market_type": mkt,
            "home": home,
            "away": away,
            "game_date": None,
        })

    parlays = []
    n_partial = 0
    for pid, n_legs, odds, profit, btype, gd in parlays_raw:
        pleg_list = legs_by_parlay.get(pid, [])
        if not pleg_list:
            continue

        for pl in pleg_list:
            pl["game_date"] = gd

        classified_legs = []
        for pl in pleg_list:
            cl = _classify_leg(pl, ml_favs)
            if cl:
                classified_legs.append(cl)

        # Only include parlays where ALL legs are classified
        if len(classified_legs) < len(pleg_list):
            n_partial += 1
            continue

        fav_count = sum(1 for cl in classified_legs if cl["is_pick_fav"])
        dog_count = len(classified_legs) - fav_count
        fav_frac = fav_count / len(classified_legs)

        parlay_won = profit > 0

        parlays.append({
            "id": pid,
            "n_legs": n_legs,
            "decimal_odds": float(odds),
            "profit": float(profit),
            "won": parlay_won,
            "game_date": gd,
            "legs": classified_legs,
            "fav_count": fav_count,
            "dog_count": dog_count,
            "fav_fraction": fav_frac,
            "composition": _classify_composition(classified_legs),
        })

    if n_partial:
        print(f"    ({n_partial} parlays excluded — some legs lacked ML classification)")

    return parlays


def _classify_composition(legs: list[dict]) -> str:
    n = len(legs)
    fav = sum(1 for l in legs if l["is_pick_fav"])
    if fav == n:
        return "all-fav"
    elif fav == 0:
        return "all-dog"
    elif fav > n / 2:
        return "fav-heavy"
    elif fav < n / 2:
        return "dog-heavy"
    else:
        return "even-split"


def _pnl_stats(parlays: list[dict]) -> dict:
    n = len(parlays)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0, "pnl": 0, "roi": 0,
                "avg_odds": 0, "staked": 0, "avg_legs": 0}
    wins = sum(1 for p in parlays if p["won"])
    pnl = sum(p["profit"] for p in parlays)
    staked = n * STAKE
    avg_odds = sum(p["decimal_odds"] for p in parlays) / n
    avg_legs = sum(p["n_legs"] for p in parlays) / n
    return {
        "n": n, "wins": wins, "wr": wins / n * 100,
        "pnl": pnl, "roi": pnl / staked * 100,
        "avg_odds": avg_odds, "staked": staked, "avg_legs": avg_legs,
    }


def _thin_flag(n: int) -> str:
    """Flag thin cells."""
    if n < 10:
        return " ⚠ THIN"
    elif n < 20:
        return " ⚠ thin"
    return ""


def analyze_window(parlays: list[dict], label: str):
    """Full analysis for one window."""
    print(f"\n{'='*95}")
    print(f"  {label}")
    print(f"  {len(parlays)} classified parlay tickets, $10/ticket")
    print(f"{'='*95}")

    if not parlays:
        print("  No parlays to analyze.")
        return {}

    all_parlay_legs = [l for p in parlays for l in p["legs"]]

    # ── 1. Composition table ─────────────────────────────────────
    print(f"\n  1. PARLAY P&L BY FAVORITE COMPOSITION")
    comps = ["all-fav", "fav-heavy", "even-split", "dog-heavy", "all-dog"]
    print(f"\n  {'Comp':<14} {'N':>5} {'WR':>7} {'CI 95%':>18} {'Avg Odds':>10} "
          f"{'P&L':>9} {'ROI':>8}")
    print(f"  {'─'*75}")

    comp_stats = {}
    for comp in comps:
        group = [p for p in parlays if p["composition"] == comp]
        if not group:
            continue
        s = _pnl_stats(group)
        lo, hi = wilson_ci(s["n"], s["wins"])
        flag = _thin_flag(s["n"])
        comp_stats[comp] = s
        print(f"  {comp:<14} {s['n']:>5} {s['wr']:>5.1f}% [{lo:>5.1f}%,{hi:>5.1f}%] "
              f"{s['avg_odds']:>10.3f} ${s['pnl']:>+8.2f} {s['roi']:>+7.1f}%{flag}")

    # Fav-majority vs Dog-majority summary
    fav_maj = [p for p in parlays if p["fav_fraction"] > 0.5]
    dog_maj = [p for p in parlays if p["fav_fraction"] < 0.5]
    sf = _pnl_stats(fav_maj)
    sd = _pnl_stats(dog_maj)
    fl, fh = wilson_ci(sf["n"], sf["wins"])
    dl, dh = wilson_ci(sd["n"], sd["wins"])
    print(f"\n  Fav-majority: n={sf['n']:>4}  WR={sf['wr']:>5.1f}% [{fl}%,{fh}%]  "
          f"odds={sf['avg_odds']:.3f}  ROI={sf['roi']:>+6.1f}%{_thin_flag(sf['n'])}")
    print(f"  Dog-majority: n={sd['n']:>4}  WR={sd['wr']:>5.1f}% [{dl}%,{dh}%]  "
          f"odds={sd['avg_odds']:.3f}  ROI={sd['roi']:>+6.1f}%{_thin_flag(sd['n'])}")

    # ── 2. 3-leg parlays by # fav legs ──────────────────────────
    print(f"\n  2. 3-LEG PARLAYS BY # FAVORITE LEGS")
    three_leg = [p for p in parlays if p["n_legs"] == 3]
    print(f"     ({len(three_leg)} tickets)")

    print(f"\n  {'Fav Legs':<10} {'N':>5} {'WR':>7} {'CI 95%':>18} {'Avg Odds':>10} "
          f"{'P&L':>9} {'ROI':>8}")
    print(f"  {'─'*72}")

    fav_leg_stats = {}
    for fav_n in [0, 1, 2, 3]:
        group = [p for p in three_leg if p["fav_count"] == fav_n]
        if not group:
            continue
        s = _pnl_stats(group)
        lo, hi = wilson_ci(s["n"], s["wins"])
        flag = _thin_flag(s["n"])
        fav_leg_stats[fav_n] = s
        print(f"  {fav_n} of 3     {s['n']:>5} {s['wr']:>5.1f}% [{lo:>5.1f}%,{hi:>5.1f}%] "
              f"{s['avg_odds']:>10.3f} ${s['pnl']:>+8.2f} {s['roi']:>+7.1f}%{flag}")

    # 2-leg parlays
    two_leg = [p for p in parlays if p["n_legs"] == 2]
    if two_leg:
        print(f"\n  2-LEG PARLAYS ({len(two_leg)} tickets)")
        print(f"\n  {'Fav Legs':<10} {'N':>5} {'WR':>7} {'CI 95%':>18} {'Avg Odds':>10} "
              f"{'P&L':>9} {'ROI':>8}")
        print(f"  {'─'*72}")
        for fav_n in [0, 1, 2]:
            group = [p for p in two_leg if p["fav_count"] == fav_n]
            if not group:
                continue
            s = _pnl_stats(group)
            lo, hi = wilson_ci(s["n"], s["wins"])
            flag = _thin_flag(s["n"])
            print(f"  {fav_n} of 2     {s['n']:>5} {s['wr']:>5.1f}% [{lo:>5.1f}%,{hi:>5.1f}%] "
                  f"{s['avg_odds']:>10.3f} ${s['pnl']:>+8.2f} {s['roi']:>+7.1f}%{flag}")

    # ── 3. Mechanism checks ─────────────────────────────────────
    print(f"\n  3. MECHANISM CHECKS")

    # 3a. Bust rate by quadrant
    print(f"\n  3a. Per-leg bust rate within parlays, by quadrant:")
    print(f"  {'Quadrant':<12} {'N legs':>7} {'Win%':>7} {'Bust%':>7} {'CI Bust':>18} {'Avg Odds':>10}")
    print(f"  {'─'*65}")

    bust_stats = {}
    for quad in ["Fav:Home", "Fav:Away", "Dog:Home", "Dog:Away"]:
        qlegs = [l for l in all_parlay_legs if l["quadrant"] == quad]
        if not qlegs:
            continue
        wins = sum(1 for l in qlegs if l["leg_result"] == "WIN")
        n_q = len(qlegs)
        bust = n_q - wins
        bust_lo, bust_hi = wilson_ci(n_q, bust)
        avg_odds = 0
        n_priced = 0
        for l in qlegs:
            if l["open_odds"] is not None:
                avg_odds += _american_to_decimal(float(l["open_odds"]))
                n_priced += 1
        avg_odds = avg_odds / n_priced if n_priced > 0 else 0
        bust_stats[quad] = {"bust_pct": bust / n_q * 100, "n": n_q, "avg_odds": avg_odds}

        print(f"  {quad:<12} {n_q:>7} {wins/n_q*100:>5.1f}% {bust/n_q*100:>5.1f}% "
              f"[{bust_lo:>5.1f}%,{bust_hi:>5.1f}%] {avg_odds:>10.3f}")

    # 3b. Aggregate bust rate: fav vs dog
    fav_legs = [l for l in all_parlay_legs if l["is_pick_fav"]]
    dog_legs = [l for l in all_parlay_legs if not l["is_pick_fav"]]
    fav_wins = sum(1 for l in fav_legs if l["leg_result"] == "WIN")
    dog_wins = sum(1 for l in dog_legs if l["leg_result"] == "WIN")
    fav_bust_pct = (len(fav_legs) - fav_wins) / len(fav_legs) * 100 if fav_legs else 0
    dog_bust_pct = (len(dog_legs) - dog_wins) / len(dog_legs) * 100 if dog_legs else 0
    fav_bl, fav_bh = wilson_ci(len(fav_legs), len(fav_legs) - fav_wins)
    dog_bl, dog_bh = wilson_ci(len(dog_legs), len(dog_legs) - dog_wins)

    print(f"\n  3b. Aggregate bust rate:")
    print(f"    Fav legs: {fav_bust_pct:>5.1f}% bust ({len(fav_legs)} legs)  "
          f"CI [{fav_bl}%,{fav_bh}%]")
    print(f"    Dog legs: {dog_bust_pct:>5.1f}% bust ({len(dog_legs)} legs)  "
          f"CI [{dog_bl}%,{dog_bh}%]")
    bust_gap = dog_bust_pct - fav_bust_pct
    print(f"    Gap: {bust_gap:+.1f}pp (fav busts less)")
    if fav_bh < dog_bl:
        print(f"    → CIs do NOT overlap — gap is statistically significant")
    elif fav_bl > dog_bh:
        print(f"    → CIs do NOT overlap — gap is statistically significant (reversed!)")
    else:
        print(f"    → CIs overlap — gap is NOT statistically significant yet")

    # 3c. Combined parlay odds: fav-heavy vs dog-heavy
    print(f"\n  3c. Combined parlay odds (fav-heavy vs dog-heavy):")
    print(f"    Fav-majority avg combined odds: {sf['avg_odds']:.3f}")
    print(f"    Dog-majority avg combined odds: {sd['avg_odds']:.3f}")
    odds_gap_pct = (sd['avg_odds'] - sf['avg_odds']) / sf['avg_odds'] * 100 if sf['avg_odds'] > 0 else 0
    print(f"    Dog pays {odds_gap_pct:+.1f}% more per ticket")

    # 3d. Expected parlay survival
    if fav_legs and dog_legs:
        fav_wr = fav_wins / len(fav_legs)
        dog_wr = dog_wins / len(dog_legs)
        print(f"\n  3d. Implied parlay ticket WR (if all legs same type):")
        for n_legs in [2, 3]:
            surv_fav = fav_wr ** n_legs * 100
            surv_dog = dog_wr ** n_legs * 100
            print(f"    {n_legs}-leg: all-fav={surv_fav:.1f}%  all-dog={surv_dog:.1f}%  "
                  f"gap={surv_fav - surv_dog:+.1f}pp")

    return {
        "fav_maj": sf, "dog_maj": sd,
        "fav_bust": fav_bust_pct, "dog_bust": dog_bust_pct,
        "fav_maj_odds": sf["avg_odds"], "dog_maj_odds": sd["avg_odds"],
        "comp_stats": comp_stats,
        "fav_leg_stats_3leg": fav_leg_stats,
    }


def main():
    print("=" * 95)
    print("  PARLAY-LEVEL P&L AUDIT — FULL PERIOD + LAST 30 DAYS")
    print("  Unit = parlay ticket ($10 stake). WIN = all legs hit. LOSS = any leg busts.")
    print("  Favorite = moneyline h2h price < 2.0 (NOT spread sign)")
    print("=" * 95)

    # Load ML favorites for full range
    print(f"\nLoading ML favorites ({FULL_START} to {FULL_END})...")
    ml_favs = _load_ml_favorites(FULL_START, FULL_END)
    print(f"  {len(ml_favs)} fixtures classified")

    # Load parlays for both windows
    print(f"\nLoading FULL PERIOD parlays ({FULL_START} to {FULL_END})...")
    parlays_full = load_parlays(ml_favs, FULL_START, FULL_END)
    print(f"  {len(parlays_full)} classified parlays")

    parlays_30d = [p for p in parlays_full if p["game_date"] >= LAST30_START]
    print(f"\n  Last 30d subset: {len(parlays_30d)} parlays")

    # ── Full period analysis ──────────────────────────────────────
    stats_full = analyze_window(parlays_full, f"FULL PERIOD ({FULL_START} to {FULL_END})")

    # ── Last 30d analysis ─────────────────────────────────────────
    stats_30d = analyze_window(parlays_30d, f"LAST 30 DAYS ({LAST30_START} to {LAST30_END})")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: Stability comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  5. STABILITY: FULL vs LAST 30 DAYS")
    print(f"{'='*95}")

    if stats_full and stats_30d:
        print(f"\n  Composition ROI stability:")
        print(f"  {'Comp':<14} {'Full N':>7} {'Full ROI':>9} {'30d N':>6} {'30d ROI':>9} {'Drift':>8}")
        print(f"  {'─'*58}")
        comps = ["all-fav", "fav-heavy", "even-split", "dog-heavy", "all-dog"]
        for comp in comps:
            sf = stats_full.get("comp_stats", {}).get(comp)
            s3 = stats_30d.get("comp_stats", {}).get(comp)
            if not sf and not s3:
                continue
            fn = sf["n"] if sf else 0
            fr = sf["roi"] if sf else 0
            tn = s3["n"] if s3 else 0
            tr = s3["roi"] if s3 else 0
            drift = tr - fr if sf and s3 else 0
            print(f"  {comp:<14} {fn:>7} {fr:>+8.1f}% {tn:>6} {tr:>+8.1f}% {drift:>+7.1f}pp")

        print(f"\n  Fav-majority vs Dog-majority:")
        print(f"    Full:  Fav ROI={stats_full['fav_maj']['roi']:>+6.1f}%  "
              f"Dog ROI={stats_full['dog_maj']['roi']:>+6.1f}%  "
              f"gap={stats_full['fav_maj']['roi'] - stats_full['dog_maj']['roi']:+.1f}pp")
        print(f"    30d:   Fav ROI={stats_30d['fav_maj']['roi']:>+6.1f}%  "
              f"Dog ROI={stats_30d['dog_maj']['roi']:>+6.1f}%  "
              f"gap={stats_30d['fav_maj']['roi'] - stats_30d['dog_maj']['roi']:+.1f}pp")

        print(f"\n  Bust rate stability:")
        print(f"    Full:  Fav bust={stats_full['fav_bust']:.1f}%  "
              f"Dog bust={stats_full['dog_bust']:.1f}%  "
              f"gap={stats_full['dog_bust'] - stats_full['fav_bust']:+.1f}pp")
        print(f"    30d:   Fav bust={stats_30d['fav_bust']:.1f}%  "
              f"Dog bust={stats_30d['dog_bust']:.1f}%  "
              f"gap={stats_30d['dog_bust'] - stats_30d['fav_bust']:+.1f}pp")

        print(f"\n  Combined odds stability:")
        print(f"    Full:  Fav-maj odds={stats_full['fav_maj_odds']:.3f}  "
              f"Dog-maj odds={stats_full['dog_maj_odds']:.3f}")
        print(f"    30d:   Fav-maj odds={stats_30d['fav_maj_odds']:.3f}  "
              f"Dog-maj odds={stats_30d['dog_maj_odds']:.3f}")

        # 3-leg fav-count pattern stability
        print(f"\n  3-leg fav-count ROI pattern stability:")
        print(f"  {'Fav Legs':<10} {'Full N':>7} {'Full ROI':>9} {'30d N':>6} {'30d ROI':>9} {'Drift':>8}")
        print(f"  {'─'*55}")
        for fav_n in [0, 1, 2, 3]:
            sf3 = stats_full.get("fav_leg_stats_3leg", {}).get(fav_n)
            s33 = stats_30d.get("fav_leg_stats_3leg", {}).get(fav_n)
            fn = sf3["n"] if sf3 else 0
            fr = sf3["roi"] if sf3 else 0
            tn = s33["n"] if s33 else 0
            tr = s33["roi"] if s33 else 0
            drift = tr - fr if sf3 and s33 else 0
            flag = _thin_flag(fn)
            print(f"  {fav_n} of 3     {fn:>7} {fr:>+8.1f}% {tn:>6} {tr:>+8.1f}% "
                  f"{drift:>+7.1f}pp{flag}")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  VERDICT")
    print(f"{'='*95}")

    if stats_full:
        sf = stats_full["fav_maj"]
        sd = stats_full["dog_maj"]
        fl, fh = wilson_ci(sf["n"], sf["wins"])
        dl, dh = wilson_ci(sd["n"], sd["wins"])

        print(f"\n  FULL PERIOD parlay-level:")
        print(f"    Fav-majority: n={sf['n']:>4}  WR={sf['wr']:>5.1f}% [{fl}%,{fh}%]  "
              f"odds={sf['avg_odds']:.3f}  ROI={sf['roi']:>+6.1f}%")
        print(f"    Dog-majority: n={sd['n']:>4}  WR={sd['wr']:>5.1f}% [{dl}%,{dh}%]  "
              f"odds={sd['avg_odds']:.3f}  ROI={sd['roi']:>+6.1f}%")

        if sf["n"] > 0 and sd["n"] > 0:
            if sf["roi"] > sd["roi"]:
                print(f"\n  → Fav-majority parlays have BETTER ticket ROI "
                      f"({sf['roi']:+.1f}% vs {sd['roi']:+.1f}%)")
                if fh < dl:
                    print(f"    WR CIs do NOT overlap → statistically significant")
                else:
                    print(f"    WR CIs overlap → directional but not yet significant")
            else:
                print(f"\n  → Dog-majority parlays have BETTER ticket ROI "
                      f"({sd['roi']:+.1f}% vs {sf['roi']:+.1f}%)")

        bust_gap = stats_full["dog_bust"] - stats_full["fav_bust"]
        print(f"\n  Mechanism check:")
        print(f"    Bust rate gap: {bust_gap:+.1f}pp (fav busts less)")
        print(f"    Odds convergence: fav-maj={sf['avg_odds']:.3f} vs "
              f"dog-maj={sd['avg_odds']:.3f} "
              f"({(sd['avg_odds']-sf['avg_odds'])/sf['avg_odds']*100:+.1f}% more payout)")
        print(f"    Compounding: each fav leg saves ~{bust_gap:.1f}pp bust risk → "
              f"in a 3-leg parlay that's ~{bust_gap*3:.0f}pp cumulative survival advantage")


if __name__ == "__main__":
    main()
