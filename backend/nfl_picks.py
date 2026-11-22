"""
nfl_picks.py — NFL Lab "Picks": pre-registered FW-2 qualifying picks + budget planner.

READ-ONLY.  Picks = analysis/nfl_fw2_rules.slate_picks (FanDuel price, not past kickoff).
Card = prereg staking (FW-2c+2d by EV, 1/player, 2/game, 5u; Questionable benched).

Parlay suggestions are CROSS-GAME only (FanDuel prices cross-game parlays as the
product of legs — no correlation haircut).  Same-game correlation stacking is NOT
suggested: the 2026-10-03 SGP study found FanDuel's copula prices it (R²=98.2%;
book over-adjusts at moderate/deep lines).

Joint hit probability is haircut by the measured realized/independence ratio for
k-leg parlays from data/nfl_fw2_parlay_backtest.json (exploratory, mostly in-sample).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import time

logger = logging.getLogger(__name__)
_HERE = os.path.dirname(__file__)
_PARLAY_BT = os.path.abspath(os.path.join(_HERE, "..", "data", "nfl_fw2_parlay_backtest.json"))

_spec = importlib.util.spec_from_file_location("nfl_fw2_rules_p", os.path.join(_HERE, "analysis", "nfl_fw2_rules.py"))
fw2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw2)

_CACHE: dict = {}


def _parlay_bt():
    try:
        with open(_PARLAY_BT) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("[NFLPicks] parlay backtest missing at %s — joint probs un-haircut", _PARLAY_BT)
        return None


def _build_parlay(legs, ratio):
    price, p = 1.0, 1.0
    for l in legs:
        price *= l["price"]
        p *= l["model_p"]
    p_h = p * (ratio or 1.0)
    return {"k": len(legs), "legs": [{k: l[k] for k in ("player", "stat", "side", "line", "price", "rule", "game_id")}
                                     for l in legs],
            "price": round(price, 2), "p_model_indep": round(p, 4), "p_haircut": round(p_h, 4),
            "indep_ratio": ratio, "ev_model": round(p * price - 1, 3), "ev_haircut": round(p_h * price - 1, 3)}


def plan(season=2026, week=None, budget=50.0, unit=10.0, max_parlay_legs=8):
    if week is None:
        import fantasy_projections as fp
        season, week = fp.get_current_nfl_week()
    key = (season, week, int(time.time() // 300))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = fw2.slate_picks(season, week)
    sl = _CACHE[key]
    picks = sl["picks"]
    card, bench = fw2.prereg_card(picks)
    bt = _parlay_bt()
    ratios = {r["k"]: r["indep_ratio"] for r in (bt or {}).get("rows", [])}

    # Parlay pool: FW-2c/2d, not Questionable/Out, one best-EV leg per game
    pool, seen_g, seen_p = [], set(), set()
    for p in picks:
        if p["rule"] not in ("FW-2c", "FW-2d") or p["injury"] in ("Out", "IR", "Doubtful", "Questionable"):
            continue
        if p["game_id"] in seen_g or p["player"] in seen_p:
            continue
        pool.append(p)
        seen_g.add(p["game_id"])
        seen_p.add(p["player"])
    parlays = [_build_parlay(pool[:k], ratios.get(k)) for k in range(2, min(max_parlay_legs, len(pool)) + 1)]

    # Budget allocation: straights first (prereg stakes), then 2/3-leg, then one small lottery
    alloc, spent = [], 0.0
    for c in card:
        stake = round(c["stake_units"] * unit, 2)
        if spent + stake > budget:
            break
        alloc.append({"type": "straight", "stake": stake, "desc": f"{c['player']} {c['stat']} {c['side'].upper()} {c['line']}",
                      "price": c["price"], "p": c["model_p"], "ev_per_dollar": c["ev"]})
        spent += stake
    for pr in parlays:
        if pr["k"] in (2, 3) and spent + unit <= budget:
            alloc.append({"type": f"{pr['k']}-leg parlay", "stake": unit,
                          "desc": " + ".join(f"{l['player']} {l['side'][0].upper()}{l['line']}" for l in pr["legs"]),
                          "price": pr["price"], "p": pr["p_haircut"], "ev_per_dollar": pr["ev_haircut"]})
            spent += unit
    lot = next((pr for pr in reversed(parlays) if pr["k"] >= 6), None)
    rem = round(budget - spent, 2)
    if lot and rem > 0:
        stake = round(min(rem, unit / 2), 2)
        alloc.append({"type": f"{lot['k']}-leg lottery", "stake": stake,
                      "desc": " + ".join(f"{l['player']} {l['side'][0].upper()}{l['line']}" for l in lot["legs"]),
                      "price": lot["price"], "p": lot["p_haircut"], "ev_per_dollar": lot["ev_haircut"]})
        spent += stake
    exp_profit = sum(a["stake"] * a["ev_per_dollar"] for a in alloc)
    return {
        "sport": "NFL", "season": season, "week": week, "generated_at": sl["generated_at"],
        "budget": budget, "unit": unit, "spent": round(spent, 2), "expected_profit": round(exp_profit, 2),
        "picks": picks, "card": card, "bench": bench, "parlays": parlays, "allocation": alloc,
        "parlay_backtest": bt, "fd_missing": sl["fd_missing"], "n_future_main_preds": sl["n_future_main_preds"],
        "notes": ("Qualifying = pre-registered FW-2 rules (registered 2026-11-15). FW-2e is track-only — never bet. "
                  "Card = prereg staking. Parlays are cross-game only; joint P haircut by measured realized/indep "
                  "ratio (exploratory, NOT pre-registered). Same-game correlation stacks are not suggested: "
                  "FanDuel's SGP copula prices correlation (2026-10-03 study). Expected profit uses model P "
                  "(straights) and haircut P (parlays) — model P is uncalibrated at the top end."),
    }
