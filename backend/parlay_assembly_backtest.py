"""
parlay_assembly_backtest.py — Parlay-level backtest: blind vs polarized assembly.

Generates candidate legs per day using the walk_forward_backtest infrastructure,
then assembles them into parlays under two strategies:
  - BLIND (current): rank combos by avg LOS, no composition awareness
  - POLARIZED (Shadow B): penalize 1-of-N mixed fav/dog combos

Settles each parlay (all legs must win) and compares ticket-level ROI.

Post-5/26 only (current system regime).
"""
from __future__ import annotations
import os, sys, itertools, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

STAKE = 10.0
START = "2026-05-26"
END   = "2026-06-26"
MAX_PARLAYS_PER_DAY = 8  # match live ~6 avg, slight headroom
PARLAY_SIZES = [3, 2]    # try 3-leg first, then 2-leg


def wilson_ci(n: int, w: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round((centre - margin) * 100, 1), round((centre + margin) * 100, 1))


def _load_ml_favorites(start: str, end: str) -> dict:
    """ML-based fav/dog from fixtures.bookmakers h2h prices."""
    import sqlite3, json
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


def _load_alt_odds_raw(start: str, end: str) -> dict:
    """Load decimal odds from historical.db: (home, away, date, team, spread) → avg decimal odds."""
    import sqlite3
    hist_path = os.path.join(_HERE, "..", "data", "historical.db")
    conn = sqlite3.connect(hist_path)
    rows = conn.execute("""
        SELECT home_team, away_team, game_date, team, line, odds
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND game_date BETWEEN ? AND ?
          AND odds IS NOT NULL AND odds > 1.0
    """, (start, end)).fetchall()
    conn.close()

    raw = defaultdict(list)
    for home, away, gd, team, line, odds in rows:
        key = (home, away, str(gd)[:10], team, float(line))
        raw[key].append(float(odds))
    return {k: sum(v) / len(v) for k, v in raw.items()}


def generate_daily_candidates(start: str, end: str, ml_favs: dict, odds_raw: dict) -> dict:
    """Generate all candidate legs per day using the ATS model.

    Returns {game_date: [list of candidate leg dicts]}.
    Each leg has: game_date, home_team, away_team, spread_value, pick_side,
    pick_team, model_wp, market_wp, edge_pp, los, lqs, actual_result,
    is_pick_fav, decimal_odds, fixture_key.
    """
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
                        dec_odds = odds_raw.get((home, away, gd, home, sv))
                        if dec_odds is None:
                            dec_odds = 1.0 / mkt_home if mkt_home > 0 else None
                        pick_is_fav = home_is_fav if home_is_fav is not None else None
                        daily_candidates[gd].append({
                            "game_date": gd, "home_team": home, "away_team": away,
                            "spread_value": sv, "pick_side": "home",
                            "pick_team": home, "model_wp": home_wp,
                            "market_wp": mkt_home, "edge_pp": round(edge, 2),
                            "los": round(los, 4), "lqs": lqs,
                            "actual_result": result,
                            "is_pick_fav": pick_is_fav,
                            "decimal_odds": dec_odds,
                            "fixture_key": fixture_key,
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


def _classify_composition(legs: list) -> str:
    n = len(legs)
    fav = sum(1 for l in legs if l.get("is_pick_fav"))
    if fav == n:
        return "all-fav"
    elif fav == 0:
        return "all-dog"
    elif fav > n / 2:
        return "fav-heavy"
    elif fav < n / 2:
        return "dog-heavy"
    return "even-split"


def assemble_parlays(candidates: list, strategy: str, max_parlays: int) -> list:
    """Assemble parlays from a day's candidate legs.

    Strategies:
      - "blind": rank by avg LOS only (current system behavior)
      - "polarized": penalize 1-of-N mixed fav/dog combos
    """
    if len(candidates) < 2:
        return []

    # Filter to legs with known fav/dog and odds
    pool = [c for c in candidates
            if c.get("is_pick_fav") is not None
            and c.get("decimal_odds") is not None
            and c["decimal_odds"] > 1.0
            and c.get("actual_result") in ("WIN", "LOSS")]

    if len(pool) < 2:
        return []

    parlays = []
    used_sigs = set()

    for size in PARLAY_SIZES:
        if len(pool) < size:
            continue

        combos = list(itertools.combinations(range(len(pool)), size))
        # Cap for performance
        if len(combos) > 3000:
            import random
            combos = random.sample(combos, 3000)

        for combo in combos:
            legs = [pool[i] for i in combo]

            # No same-fixture constraint
            fixture_keys = [l["fixture_key"] for l in legs]
            if len(fixture_keys) != len(set(fixture_keys)):
                continue

            # Combined WP and odds
            combined_wp = 1.0
            combined_odds = 1.0
            for l in legs:
                combined_wp *= l["model_wp"]
                combined_odds *= l["decimal_odds"]

            # Combined WP floor (30% for 3-leg, 40% for 2-leg)
            wp_floor = {3: 0.30, 2: 0.40}.get(size, 0.30)
            if combined_wp < wp_floor:
                continue

            # Min combined odds (2.0 for 3-leg, 1.5 for 2-leg)
            min_odds = {3: 2.0, 2: 1.5}.get(size, 1.5)
            if combined_odds < min_odds:
                continue

            # Dedup
            sig = frozenset(id(l) for l in legs)
            if sig in used_sigs:
                continue
            used_sigs.add(sig)

            # Score
            avg_los = sum(l["los"] for l in legs) / len(legs)
            avg_lqs = sum(l["lqs"] for l in legs) / len(legs)
            avg_wp = sum(l["model_wp"] for l in legs) / len(legs)

            # Composite (simplified version of live's formula)
            norm_ev = max(0, combined_wp * (combined_odds - 1) - (1 - combined_wp))
            composite = avg_wp * 0.40 + (avg_lqs / 100) * 0.20 + avg_los * 0.30 + min(norm_ev, 1.0) * 0.10

            # Strategy-specific adjustments
            if strategy == "polarized":
                fav_count = sum(1 for l in legs if l["is_pick_fav"])
                comp = _classify_composition(legs)
                # Penalize 1-of-N mixed: exactly 1 fav in 3-leg, or mixed in general
                if size == 3 and fav_count == 1:
                    composite *= 0.80  # 20% penalty for 1-of-3 fav
                elif size == 3 and fav_count == 2:
                    composite *= 0.90  # 10% penalty for 2-of-3 (mild mixed)
                elif size == 2 and fav_count == 1:
                    composite *= 0.85  # 15% penalty for 1-of-2 mixed
                # Boost polarized
                if comp in ("all-fav", "all-dog"):
                    composite *= 1.10  # 10% boost for polarized

            # Settle: parlay wins if ALL legs win
            parlay_won = all(l["actual_result"] == "WIN" for l in legs)

            parlays.append({
                "legs": legs,
                "n_legs": size,
                "combined_odds": combined_odds,
                "combined_wp": combined_wp,
                "composite": composite,
                "avg_los": avg_los,
                "won": parlay_won,
                "profit": (combined_odds - 1) * STAKE if parlay_won else -STAKE,
                "fav_count": sum(1 for l in legs if l["is_pick_fav"]),
                "composition": _classify_composition(legs),
            })

    # Sort by composite and select top N
    parlays.sort(key=lambda p: -p["composite"])
    return parlays[:max_parlays]


def _pnl_stats(parlays: list) -> dict:
    n = len(parlays)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0, "pnl": 0, "roi": 0,
                "avg_odds": 0, "staked": 0}
    wins = sum(1 for p in parlays if p["won"])
    pnl = sum(p["profit"] for p in parlays)
    staked = n * STAKE
    avg_odds = sum(p["combined_odds"] for p in parlays) / n
    return {
        "n": n, "wins": wins, "wr": wins / n * 100,
        "pnl": pnl, "roi": pnl / staked * 100,
        "avg_odds": avg_odds, "staked": staked,
    }


def _thin_flag(n: int) -> str:
    if n < 10:
        return " ⚠THIN"
    elif n < 20:
        return " ⚠thin"
    return ""


def main():
    print("=" * 95)
    print("  PARLAY ASSEMBLY BACKTEST: BLIND vs POLARIZED")
    print(f"  Period: {START} to {END} (post-5/26 regime)")
    print(f"  $10/ticket, max {MAX_PARLAYS_PER_DAY} parlays/day")
    print("=" * 95)

    print("\nLoading ML favorites...")
    ml_favs = _load_ml_favorites(START, END)
    print(f"  {len(ml_favs)} fixtures")

    print("Loading alt-line odds...")
    odds_raw = _load_alt_odds_raw(START, END)
    print(f"  {len(odds_raw)} odds entries")

    print("Generating daily candidates...")
    daily_cands = generate_daily_candidates(START, END, ml_favs, odds_raw)
    total_cands = sum(len(v) for v in daily_cands.values())
    print(f"  {len(daily_cands)} days, {total_cands} total candidates")

    # ── Assemble under both strategies ────────────────────────────
    all_blind = []
    all_polar = []

    for gd in sorted(daily_cands.keys()):
        cands = daily_cands[gd]
        blind_parlays = assemble_parlays(cands, "blind", MAX_PARLAYS_PER_DAY)
        polar_parlays = assemble_parlays(cands, "polarized", MAX_PARLAYS_PER_DAY)

        for p in blind_parlays:
            p["game_date"] = gd
        for p in polar_parlays:
            p["game_date"] = gd

        all_blind.extend(blind_parlays)
        all_polar.extend(polar_parlays)

    print(f"\n  Blind:     {len(all_blind)} parlays across {len(daily_cands)} days")
    print(f"  Polarized: {len(all_polar)} parlays across {len(daily_cands)} days")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Overall comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  1. OVERALL: BLIND vs POLARIZED")
    print(f"{'='*95}")

    sb = _pnl_stats(all_blind)
    sp = _pnl_stats(all_polar)
    bl, bh = wilson_ci(sb["n"], sb["wins"])
    pl, ph = wilson_ci(sp["n"], sp["wins"])

    print(f"\n  {'Strategy':<12} {'N':>5} {'WR':>7} {'CI 95%':>18} {'Avg Odds':>10} "
          f"{'Staked':>8} {'P&L':>9} {'ROI':>8}")
    print(f"  {'─'*72}")
    print(f"  {'Blind':<12} {sb['n']:>5} {sb['wr']:>5.1f}% [{bl:>5.1f}%,{bh:>5.1f}%] "
          f"{sb['avg_odds']:>10.3f}   ${sb['staked']:>6.0f} ${sb['pnl']:>+8.2f} {sb['roi']:>+7.1f}%")
    print(f"  {'Polarized':<12} {sp['n']:>5} {sp['wr']:>5.1f}% [{pl:>5.1f}%,{ph:>5.1f}%] "
          f"{sp['avg_odds']:>10.3f}   ${sp['staked']:>6.0f} ${sp['pnl']:>+8.2f} {sp['roi']:>+7.1f}%")
    print(f"\n  ROI delta: {sp['roi'] - sb['roi']:+.1f}pp (polarized vs blind)")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: Composition distribution shift
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  2. COMPOSITION SHIFT: did polarization change the mix?")
    print(f"{'='*95}")

    comps = ["all-fav", "fav-heavy", "even-split", "dog-heavy", "all-dog"]
    print(f"\n  {'Comp':<14} {'Blind N':>8} {'Blind%':>7} {'Polar N':>8} {'Polar%':>7} {'Shift':>7}")
    print(f"  {'─'*55}")

    for comp in comps:
        nb = sum(1 for p in all_blind if p["composition"] == comp)
        np_ = sum(1 for p in all_polar if p["composition"] == comp)
        pct_b = nb / len(all_blind) * 100 if all_blind else 0
        pct_p = np_ / len(all_polar) * 100 if all_polar else 0
        print(f"  {comp:<14} {nb:>8} {pct_b:>5.1f}% {np_:>8} {pct_p:>5.1f}% {pct_p - pct_b:>+5.1f}pp")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: Per-composition ROI comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  3. PER-COMPOSITION ROI")
    print(f"{'='*95}")

    print(f"\n  {'Comp':<14} {'Blind':>30} {'Polarized':>30}")
    print(f"  {'':14} {'N':>5} {'WR':>7} {'ROI':>8} {'AvgOdds':>9}  "
          f"{'N':>5} {'WR':>7} {'ROI':>8} {'AvgOdds':>9}")
    print(f"  {'─'*75}")

    for comp in comps:
        gb = [p for p in all_blind if p["composition"] == comp]
        gp = [p for p in all_polar if p["composition"] == comp]
        sb_ = _pnl_stats(gb)
        sp_ = _pnl_stats(gp)
        flag_b = _thin_flag(sb_["n"])
        flag_p = _thin_flag(sp_["n"])
        b_wr = f"{sb_['wr']:5.1f}%" if sb_["n"] > 0 else "   — "
        p_wr = f"{sp_['wr']:5.1f}%" if sp_["n"] > 0 else "   — "
        b_roi = f"{sb_['roi']:+6.1f}%" if sb_["n"] > 0 else "   — "
        p_roi = f"{sp_['roi']:+6.1f}%" if sp_["n"] > 0 else "   — "
        b_odds = f"{sb_['avg_odds']:.3f}" if sb_["n"] > 0 else "  —"
        p_odds = f"{sp_['avg_odds']:.3f}" if sp_["n"] > 0 else "  —"
        print(f"  {comp:<14} {sb_['n']:>5} {b_wr:>7} {b_roi:>8} {b_odds:>9}{flag_b}  "
              f"{sp_['n']:>5} {p_wr:>7} {p_roi:>8} {p_odds:>9}{flag_p}")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: 3-leg parlays by fav count
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  4. 3-LEG PARLAYS BY FAV COUNT")
    print(f"{'='*95}")

    for label, group in [("Blind", all_blind), ("Polarized", all_polar)]:
        three = [p for p in group if p["n_legs"] == 3]
        print(f"\n  {label} ({len(three)} 3-leg tickets):")
        print(f"  {'Fav':>5} {'N':>5} {'WR':>7} {'Avg Odds':>10} {'ROI':>8}")
        print(f"  {'─'*40}")
        for fav_n in [0, 1, 2, 3]:
            g = [p for p in three if p["fav_count"] == fav_n]
            if not g:
                continue
            s = _pnl_stats(g)
            print(f"  {fav_n}/3   {s['n']:>5} {s['wr']:>5.1f}% {s['avg_odds']:>10.3f} "
                  f"{s['roi']:>+7.1f}%{_thin_flag(s['n'])}")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: Bust rate comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  5. LEG BUST RATES WITHIN SELECTED PARLAYS")
    print(f"{'='*95}")

    for label, group in [("Blind", all_blind), ("Polarized", all_polar)]:
        all_legs = [l for p in group for l in p["legs"]]
        fav_legs = [l for l in all_legs if l["is_pick_fav"]]
        dog_legs = [l for l in all_legs if not l["is_pick_fav"]]
        fb = sum(1 for l in fav_legs if l["actual_result"] != "WIN") / len(fav_legs) * 100 if fav_legs else 0
        db = sum(1 for l in dog_legs if l["actual_result"] != "WIN") / len(dog_legs) * 100 if dog_legs else 0
        print(f"\n  {label}:")
        print(f"    Fav legs: {fb:>5.1f}% bust ({len(fav_legs)} legs)")
        print(f"    Dog legs: {db:>5.1f}% bust ({len(dog_legs)} legs)")
        print(f"    Gap: {db - fb:+.1f}pp")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: Daily P&L comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  6. DAILY P&L: BLIND vs POLARIZED")
    print(f"{'='*95}")

    all_dates = sorted(set(p["game_date"] for p in all_blind + all_polar))
    blind_better = 0
    polar_better = 0
    ties = 0

    print(f"\n  {'Date':<12} {'B tickets':>10} {'B P&L':>8} {'P tickets':>10} {'P P&L':>8} {'Winner':>8}")
    print(f"  {'─'*60}")
    for gd in all_dates:
        b_day = [p for p in all_blind if p["game_date"] == gd]
        p_day = [p for p in all_polar if p["game_date"] == gd]
        b_pnl = sum(p["profit"] for p in b_day)
        p_pnl = sum(p["profit"] for p in p_day)
        if p_pnl > b_pnl:
            winner = "POLAR"
            polar_better += 1
        elif b_pnl > p_pnl:
            winner = "BLIND"
            blind_better += 1
        else:
            winner = "TIE"
            ties += 1
        print(f"  {gd:<12} {len(b_day):>10} ${b_pnl:>+7.2f} {len(p_day):>10} ${p_pnl:>+7.2f} {winner:>8}")

    print(f"\n  Daily wins: Blind={blind_better}  Polar={polar_better}  Ties={ties}")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  VERDICT")
    print(f"{'='*95}")

    sb = _pnl_stats(all_blind)
    sp = _pnl_stats(all_polar)
    delta = sp["roi"] - sb["roi"]

    print(f"\n  Blind:     {sb['n']} tickets, WR={sb['wr']:.1f}%, ROI={sb['roi']:+.1f}%")
    print(f"  Polarized: {sp['n']} tickets, WR={sp['wr']:.1f}%, ROI={sp['roi']:+.1f}%")
    print(f"  Delta:     {delta:+.1f}pp ROI")

    if delta > 5:
        print(f"\n  → Polarized WINS by {delta:.1f}pp. Deploy Shadow B.")
    elif delta > 0:
        print(f"\n  → Polarized is marginally better (+{delta:.1f}pp). Deploy Shadow B for forward confirmation.")
    elif delta > -5:
        print(f"\n  → Strategies are equivalent ({delta:+.1f}pp). Composition may not matter at this sample size.")
    else:
        print(f"\n  → Polarized is WORSE ({delta:+.1f}pp). The composition finding doesn't transfer to assembly.")

    # Composition shift summary
    blind_mixed_1of3 = sum(1 for p in all_blind if p["n_legs"] == 3 and p["fav_count"] == 1)
    polar_mixed_1of3 = sum(1 for p in all_polar if p["n_legs"] == 3 and p["fav_count"] == 1)
    blind_3 = sum(1 for p in all_blind if p["n_legs"] == 3) or 1
    polar_3 = sum(1 for p in all_polar if p["n_legs"] == 3) or 1
    print(f"\n  1-of-3 mixed (the valley):")
    print(f"    Blind:     {blind_mixed_1of3}/{blind_3} ({blind_mixed_1of3/blind_3*100:.0f}%) of 3-leg parlays")
    print(f"    Polarized: {polar_mixed_1of3}/{polar_3} ({polar_mixed_1of3/polar_3*100:.0f}%) of 3-leg parlays")


if __name__ == "__main__":
    main()
