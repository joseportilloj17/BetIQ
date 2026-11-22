"""
GM4 pre-registered evaluation vs the OPENER (Tuesday 16:00Z consensus). Read-only.
Spec: OPEN_ITEMS NFL-GM4 step 3 (registered 2026-11-22 before the first GM4 run).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lab_nfl_games import _ABBR  # noqa: E402
import nfl_game_model_gm4 as gm4  # noqa: E402

_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "odds_history", "nfl"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "nfl_gm4_eval.json"))
GAP, BE = 1.5, 0.5238


def openers(label="tue"):
    """{(season, week, home_abbr, away_abbr): (median home spread, median total)} from snapshots."""
    out = {}
    for f in sorted(glob.glob(os.path.join(_DIR, "*.json"))):
        d = json.load(open(f))
        if d.get("label") != label:
            continue
        for e in d["data"]:
            h, a = _ABBR.get(e["home_team"]), _ABBR.get(e["away_team"])
            sp, tt = [], []
            for b in e.get("bookmakers", []):
                for m in b.get("markets", []):
                    o = {x["name"]: x for x in m.get("outcomes", [])}
                    if m["key"] == "spreads" and e["home_team"] in o and o[e["home_team"]].get("point") is not None:
                        sp.append(o[e["home_team"]]["point"])
                    if m["key"] == "totals" and "Over" in o and o["Over"].get("point") is not None:
                        tt.append(o["Over"]["point"])
            key = (d["season"], d["week"], h, a)
            # only the target week's games (snapshot lists the whole future schedule)
            if sp and key not in out:
                out[key] = (statistics.median(sp), statistics.median(tt) if tt else np.nan)
    return out


def _ci(x, n=4000, seed=1):
    x = np.asarray(x, float)
    if len(x) < 3:
        return None
    bs = np.random.default_rng(seed).choice(x, size=(n, len(x))).mean(1)
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def evaluate(pred_path=None, out_path=None):
    pr = pd.read_csv(pred_path or gm4.PRED_PATH)
    op = openers()
    pr["open_spread_home"] = [op.get((s, w, h, a), (np.nan,))[0] for s, w, h, a in zip(pr.season, pr.week, pr.home, pr.away)]
    pr["open_total"] = [op.get((s, w, h, a), (np.nan, np.nan))[1] for s, w, h, a in zip(pr.season, pr.week, pr.home, pr.away)]
    d = pr[pr.margin.notna() & pr.open_spread_home.notna() & pr.close_spread_home.notna()].copy()
    res = {"n_games_with_opener": int(len(d)), "markets": {}}
    for mk in ("spread", "total"):
        if mk == "spread":
            gap = d.open_spread_home - d.model_spread_home            # +ve → model likes HOME vs opener
            clv = (d.open_spread_home - d.close_spread_home) * np.sign(gap)   # pts of line moved toward model side
            r_open = (d.margin + d.open_spread_home) * np.sign(gap)
            r_close = (d.margin + d.close_spread_home) * np.sign(gap)
            mv = d.open_spread_home - d.close_spread_home
        else:
            dd = d[d.open_total.notna() & d.close_total.notna()]
            gap = dd.model_total - dd.open_total                      # +ve → model likes OVER
            clv = (dd.close_total - dd.open_total) * np.sign(gap)
            r_open = (dd.total - dd.open_total) * np.sign(gap)
            r_close = (dd.total - dd.close_total) * np.sign(gap)
            mv = dd.close_total - dd.open_total
        base = d if mk == "spread" else dd
        out = {"corr_gap_vs_move": round(float(np.corrcoef(gap, mv)[0, 1]), 3)}
        for lbl, mask in (("walk_forward_2023_25", base.season.between(2023, 2025)),
                          ("holdout_2026_w4_w11", (base.season == 2026) & base.week.between(4, 11)),
                          ("extra_2022", base.season == 2022)):
            sel = mask & (gap.abs() >= GAP)
            c, ro, rc = clv[sel], r_open[sel], r_close[sel]
            ro, rc = ro[ro != 0], rc[rc != 0]
            out[lbl] = {"n_bets": int(sel.sum()), "m1_clv_pts": round(float(c.mean()), 3) if len(c) else None,
                        "m1_ci95": _ci(c), "m2_hit_at_open": round(float((ro > 0).mean()) * 100, 1) if len(ro) else None,
                        "m3_hit_at_close": round(float((rc > 0).mean()) * 100, 1) if len(rc) else None,
                        "corr_gap_vs_move": round(float(np.corrcoef(gap[mask], mv[mask])[0, 1]), 3) if mask.sum() > 3 else None}
        wf, ho = out["walk_forward_2023_25"], out["holdout_2026_w4_w11"]
        out["take_gate_passed"] = bool(wf["m1_ci95"] and wf["m1_ci95"][0] > 0 and (ho["m1_clv_pts"] or -1) > 0
                                       and (wf["m2_hit_at_open"] or 0) > BE * 100)
        res["markets"][mk] = out
    with open(out_path or OUT, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    if "--gm41" in sys.argv:
        print(json.dumps(evaluate(gm4.PRED_PATH_41, OUT.replace("gm4_eval", "gm41_eval")), indent=1))
    else:
        print(json.dumps(evaluate(), indent=1))
