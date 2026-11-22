"""
nfl_performance.py — Week-over-week NFL prop performance with filter "dials".

READ-ONLY over historical.db:fantasy_forward_preds (graded rows only — the
Tuesday grader writes outcomes; nothing here fetches actuals).

Populations
  fw2_live : FW-2c + FW-2d (the live-bet rules)
  FW-2c / FW-2d / FW-2e : individual pre-registered rules (analysis/nfl_fw2_rules.py)
  all      : every graded pred, bet on the model-favoured side when edge ≥ min_edge

Per week: N, W, WR, avg breakeven (mean 1/price), ROI (flat 1u), game-clustered
bootstrap 95% CI, cumulative units.  Plus a falsification metric that does not
depend on bet selection: Brier(model P(over)) − Brier(devigged market) on the
same graded rows (negative = model beats the market; CI above 0 = market wins).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import random
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)

_HIST_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "historical.db"))
_YARD = ("pass_yds", "rush_yds", "rec_yds")

_spec = importlib.util.spec_from_file_location(
    "nfl_fw2_rules", os.path.join(os.path.dirname(__file__), "analysis", "nfl_fw2_rules.py"))
fw2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw2)


def _ro():
    c = sqlite3.connect(f"file:{_HIST_DB}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _boot(groups: dict, n=1000, seed=11):
    """groups: {cluster: [values]} → 95% CI of the pooled mean."""
    keys = list(groups)
    if len(keys) < 3:
        return None
    rng = random.Random(seed)
    st = []
    for _ in range(n):
        s = [x for k in (rng.choice(keys) for _ in keys) for x in groups[k]]
        st.append(sum(s) / len(s))
    st.sort()
    return [round(st[int(0.025 * n)], 4), round(st[int(0.975 * n)], 4)]


def _summ(picks):
    res = [p for p in picks if p["is_win"] is not None]
    n = len(res)
    if not n:
        return {"n": 0}
    w = sum(p["is_win"] for p in res)
    g = defaultdict(list)
    for p in res:
        g[p["game_id"] or p["player"]].append(p["roi"])
    ci = _boot(g)
    return {"n": n, "w": w, "wr": round(100 * w / n, 1),
            "be": round(100 * sum(1 / p["price"] for p in res) / n, 1),
            "roi": round(100 * sum(p["roi"] for p in res) / n, 1),
            "units": round(sum(p["roi"] for p in res), 2),
            "roi_ci": [round(100 * ci[0], 1), round(100 * ci[1], 1)] if ci else None,
            "n_games": len(g)}


def _all_population(rows, min_edge):
    picks = []
    for r in rows:
        if r["actual_value"] is None or r["prop_line"] is None or r["devigged_implied"] is None:
            continue
        pov = r["p_over_v12_gamma"] if r["stat_type"] in _YARD else r["p_over"]
        if pov is None:
            continue
        side = "over" if pov > 0.5 else "under"
        mp = pov if side == "over" else 1 - pov
        mk = r["devigged_implied"] if side == "over" else 1 - r["devigged_implied"]
        edge = (mp - mk) * 100
        if edge < min_edge:
            continue
        price = r["prop_over_price"] if side == "over" else r["prop_under_price"]
        if not price or price <= 1:
            continue
        o = fw2.outcome(r["actual_value"], float(r["prop_line"]), side, price)
        roi, win = o
        picks.append(dict(week=r["week"], stat=r["stat_type"], pos=r["position"], side=side,
                          depth=r["line_depth"], price=price, roi=roi, is_win=win,
                          game_id=r["game_id"], player=r["player_name"]))
    return picks


_CACHE: dict = {}
_TTL = 600


def weekly_performance(season=2026, population="fw2_live", stat=None, side=None, position=None,
                       depth="main", min_edge=3.0, price="fd"):
    import time
    key = (season, population, stat, side, position, depth, min_edge, price)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    out = _weekly_performance(season, population, stat, side, position, depth, min_edge, price)
    _CACHE[key] = (time.time(), out)
    return out


def _weekly_performance(season, population, stat, side, position, depth, min_edge, price):
    conn = _ro()
    q = ("SELECT * FROM fantasy_forward_preds WHERE season=? AND method='distributional' "
         "AND data_error IS NULL AND graded_at IS NOT NULL")
    args = [season]
    if depth:
        q += " AND line_depth=?"
        args.append(depth)
    rows = [dict(r) for r in conn.execute(q, args)]
    conn.close()
    # dedup (week, player, stat, line, depth) keep latest created
    best = {}
    for r in rows:
        k = (r["week"], fw2._normalize_player_name(r["player_name"]), r["stat_type"], r["prop_line"], r["line_depth"])
        if k not in best or (r["created_at"] or "") > (best[k]["created_at"] or ""):
            best[k] = r
    rows = list(best.values())
    if stat:
        rows = [r for r in rows if r["stat_type"] == stat]
    if position:
        rows = [r for r in rows if r["position"] == position]
    weeks = sorted({r["week"] for r in rows})

    if population == "all":
        picks = _all_population(rows, min_edge)
    else:
        if depth != "main":
            return {"error": "FW-2 rules are defined on main lines only — set depth=main."}
        fw2.PRICE_MODE = price
        picks, _miss = fw2.build_picks(season, weeks, grade_ungraded=False)
        fw2.PRICE_MODE = "fd"
        want = {"fw2_live": {"FW-2c", "FW-2d"}}.get(population, {population})
        picks = [p for p in picks if p["rule"] in want and p["roi"] is not None]
        if stat:
            picks = [p for p in picks if p["stat"] == stat]
        if position:
            picks = [p for p in picks if p["pos"] == position]
    if side:
        picks = [p for p in picks if p["side"] == side]

    # Brier skill vs market (selection-independent)
    brier_by_week = {}
    for wk in weeks:
        g = defaultdict(list)
        for r in rows:
            if r["week"] != wk or r["actual_value"] is None or r["devigged_implied"] is None:
                continue
            pov = r["p_over_v12_gamma"] if r["stat_type"] in _YARD else r["p_over"]
            if pov is None or r["actual_value"] == r["prop_line"]:
                continue
            y = 1.0 if r["actual_value"] > r["prop_line"] else 0.0
            g[r["game_id"] or r["player_name"]].append((pov - y) ** 2 - (r["devigged_implied"] - y) ** 2)
        vals = [x for v in g.values() for x in v]
        if vals:
            ci = _boot(g)
            brier_by_week[wk] = {"n": len(vals), "diff": round(sum(vals) / len(vals), 4), "ci": ci}

    out_weeks, cum = [], 0.0
    for wk in weeks:
        s = _summ([p for p in picks if p["week"] == wk])
        cum += s.get("units", 0) or 0
        s.update(week=wk, cum_units=round(cum, 2), brier=brier_by_week.get(wk))
        out_weeks.append(s)

    def _span(ws):
        return _summ([p for p in picks if p["week"] in ws])

    return {
        "sport": "NFL", "season": season,
        "filters": {"population": population, "stat": stat or "all", "side": side or "both",
                    "position": position or "all", "depth": depth or "all",
                    "min_edge_pp": min_edge if population == "all" else "per rule",
                    "price": price if population != "all" else "stored"},
        "weeks": out_weeks,
        "totals": {"season": _span(set(weeks)),
                   "in_sample_w2_w9": _span({w for w in weeks if w <= 9}),
                   "forward_w10_plus": _span({w for w in weeks if w >= 10})},
        "notes": "Graded rows only (Tuesday grader). ROI = flat 1u. CI = game-clustered bootstrap 95%. "
                 "Brier diff = model − devigged market on all graded rows matching filters "
                 "(negative = model better). FW-2 rules pre-injury-filter, uncapped.",
    }
