"""
Cross-game parlay backtest of FW-2 live picks (FW-2c + FW-2d), read-only.

For each graded week: sample random k-leg parlays with distinct game_ids and
distinct players, priced at FanDuel decimal odds (product — cross-game parlays
get no correlation adjustment).  Reports realized hit rate vs the independence
expectation (product of that week's per-leg WR), mean ROI per $1, and the share
of weeks a flat $1-per-parlay portfolio was profitable.

NOT pre-registered: exploratory.  Weeks W2-W9 are the FW-2 in-sample period.
"""
from __future__ import annotations

import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import nfl_fw2_rules as fw2  # noqa: E402


def run(season=2026, weeks=range(2, 12), ks=(1, 2, 3, 4, 5, 6, 7, 8), samples=4000, rules=("FW-2c", "FW-2d")):
    picks, _ = fw2.build_picks(season, list(weeks), grade_ungraded=False)
    picks = [p for p in picks if p["rule"] in rules and p["is_win"] is not None]
    by_wk = defaultdict(list)
    for p in picks:
        by_wk[p["week"]].append(p)
    rng = random.Random(17)
    print(f"Population: {'+'.join(rules)}, graded, FanDuel price, pushes excluded. N legs={len(picks)}\n")
    rows = []
    print(f"{'k':>2} {'weeks':>5} {'hit%':>6} {'indep%':>7} {'ROI/$':>7} {'med wk ROI':>10} {'wks>0':>6} {'worst wk':>9} {'best wk':>8}")
    for k in ks:
        wk_rois, hits, exp_hits, n_all = [], 0, 0.0, 0
        for wk, ps in sorted(by_wk.items()):
            games = defaultdict(list)
            for p in ps:
                games[p["game_id"]].append(p)
            if len(games) < k:
                continue
            wr = sum(p["is_win"] for p in ps) / len(ps)
            rets = []
            for _ in range(samples):
                gids = rng.sample(list(games), k)
                legs = [rng.choice(games[g]) for g in gids]
                if len({l["player"] for l in legs}) < k:
                    continue
                win = all(l["is_win"] for l in legs)
                price = 1.0
                for l in legs:
                    price *= l["price"]
                rets.append(price - 1 if win else -1.0)
                hits += win
                exp_hits += wr ** k
                n_all += 1
            if rets:
                wk_rois.append(sum(rets) / len(rets))
        if not wk_rois:
            continue
        srt = sorted(wk_rois)
        rows.append({"k": k, "weeks": len(wk_rois), "hit_pct": round(100 * hits / n_all, 1),
                     "indep_pct": round(100 * exp_hits / n_all, 1),
                     "indep_ratio": round(hits / exp_hits, 3) if exp_hits else None,
                     "roi_per_dollar": round(sum(wk_rois) / len(wk_rois), 3),
                     "weeks_profitable": sum(r > 0 for r in wk_rois),
                     "worst_week": round(srt[0], 3), "best_week": round(srt[-1], 3)})
        print(f"{k:>2} {len(wk_rois):>5} {100*hits/n_all:>6.1f} {100*exp_hits/n_all:>7.1f} "
              f"{sum(wk_rois)/len(wk_rois):>+7.2f} {srt[len(srt)//2]:>+10.2f} "
              f"{sum(r > 0 for r in wk_rois):>3}/{len(wk_rois):<2} {srt[0]:>+9.2f} {srt[-1]:>+8.2f}")


    return {"rows": rows, "n_legs": len(picks), "rules": list(rules),
            "weeks": sorted(by_wk), "note": "Exploratory, NOT pre-registered; W2-W9 in-sample for FW-2."}


if __name__ == "__main__":
    import json, time
    res = run()
    res["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nfl_fw2_parlay_backtest.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.abspath(out))
