"""
nfl_game_model_mr.py — MR-1: market + residual NFL game model.

Pre-registered 2026-11-22 (before any MR-1 result was seen):

  Base        closing market: home margin m_mkt = −spread_home, total t_mkt; implied team
              points h_mkt = t_mkt/2 + m_mkt/2, a_mkt = t_mkt/2 − m_mkt/2.
  Features    (all walk-forward, built only from data before the game's week)
    f1 proj_gap   : Sleeper bottom-up projected points (6·(pass_td+rush_td)+xpm+3·fgm, ×c)
                    minus market implied points   — home−away for margin, home+away for total
    f2 off_epa    : rolling 8-game offensive EPA/play
    f3 def_epa    : rolling 8-game defensive EPA/play allowed
    margin feats  = [f1_diff, off_epa_h − off_epa_a, def_epa_a − def_epa_h]
    total  feats  = [f1_sum,  off_epa_h + off_epa_a, def_epa_h + def_epa_a]
  Target      residual r_margin = actual_margin − m_mkt;  r_total = actual_total − t_mkt
  Fit         ridge (α=10, standardized) on 2023–2025 REG W4+  (train; evaluation on 2023–25 is burned)
  Test        2026 REG W4–W11 holdout (untouched by GM1/GAMMA-1)
  TAKE rule (per market, spread / total), BOTH required:
    (1) corr(predicted residual, actual residual) on holdout with bootstrap 95% CI lower bound > 0
    (2) hit rate when |predicted residual| ≥ 1.5 pts  >  52.38% (point estimate, N reported)
  ML is not tested separately (derived from margin).

READ-ONLY w.r.t. bets.db / historical.db.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pandas as pd

import nfl_game_model_gamma as ngg

logger = logging.getLogger(__name__)
_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
BACKTEST_PATH = os.path.join(_DATA, "nfl_game_model_mr_backtest.json")
_ALPHA = 10.0
_GAP_MIN = 1.5
_BE = 0.5238
TRAIN = (2023, 2024, 2025)
TEST_SEASON, TEST_WEEKS = 2026, range(4, 12)


def _epa_rolling(pbp, n=8):
    """{(season, week, team): (off_epa, def_epa_allowed)} using games strictly before that week."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()]
    off = p.groupby(["season", "week", "game_id", "posteam"])["epa"].mean().reset_index()
    dfn = p.groupby(["season", "week", "game_id", "defteam"])["epa"].mean().reset_index()
    off["order"] = off.season * 100 + off.week
    dfn["order"] = dfn.season * 100 + dfn.week
    out = {}
    orders = sorted(off.order.unique())
    teams = sorted(off.posteam.dropna().unique())
    for o in orders + [max(orders) + 1]:
        for t in teams:
            po = off[(off.posteam == t) & (off.order < o)].sort_values("order").tail(n)
            pd_ = dfn[(dfn.defteam == t) & (dfn.order < o)].sort_values("order").tail(n)
            if len(po) >= 3 and len(pd_) >= 3:
                out[(o, t)] = (float(po.epa.mean()), float(pd_.epa.mean()))
    return out


def build_frame(seasons):
    import nfl_data_py as nfl
    load = sorted({min(seasons) - 1} | set(seasons))
    sched = nfl.import_schedules(load)
    sched = sched[sched.game_type == "REG"]
    pbp = nfl.import_pbp_data(load)
    epa = _epa_rolling(pbp)
    tg = ngg._team_games([s for s in load if s >= 2023], sched[sched.season >= 2023])
    rows = []
    for season in seasons:
        for week in sorted(sched[sched.season == season].week.unique()):
            if week < 4:
                continue
            order = season * 100 + week
            prior = tg[(tg.order < order) & tg.actual.notna()]
            c = float(prior.actual.sum() / prior.raw.sum()) if len(prior) else 1.0
            cur = tg[tg.order == order]
            for _, g in sched[(sched.season == season) & (sched.week == week)].iterrows():
                if pd.isna(g.spread_line) or pd.isna(g.total_line):
                    continue
                h = cur[(cur.game_id == g.game_id) & (cur.side == "home")]
                a = cur[(cur.game_id == g.game_id) & (cur.side == "away")]
                eh, ea = epa.get((order, g.home_team)), epa.get((order, g.away_team))
                if h.empty or a.empty or eh is None or ea is None:
                    continue
                m_mkt = float(g.spread_line)          # nfl_data_py spread_line = home margin line (home fav > 0)
                t_mkt = float(g.total_line)
                hm, am = t_mkt / 2 + m_mkt / 2, t_mkt / 2 - m_mkt / 2
                ph, pa = c * h.raw.iloc[0], c * a.raw.iloc[0]
                rows.append({
                    "game_id": g.game_id, "season": int(season), "week": int(week),
                    "home": g.home_team, "away": g.away_team, "m_mkt": m_mkt, "t_mkt": t_mkt,
                    "mf1": (ph - hm) - (pa - am), "mf2": eh[0] - ea[0], "mf3": ea[1] - eh[1],
                    "tf1": (ph - hm) + (pa - am), "tf2": eh[0] + ea[0], "tf3": eh[1] + ea[1],
                    "margin": (g.home_score - g.away_score) if pd.notna(g.home_score) else np.nan,
                    "total": (g.home_score + g.away_score) if pd.notna(g.home_score) else np.nan,
                })
    return pd.DataFrame(rows)


def _ridge(X, y, alpha=_ALPHA):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(), "b0": float(y.mean())}


def _pred(model, X):
    Z = (X - np.array(model["mu"])) / np.array(model["sd"])
    return model["b0"] + Z @ np.array(model["beta"])


def _boot_corr(a, b, n=4000, seed=13):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    cs = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in idx])
    return [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))]


def run():
    df = build_frame(list(TRAIN) + [TEST_SEASON])
    tr = df[df.season.isin(TRAIN) & df.margin.notna()]
    te = df[(df.season == TEST_SEASON) & df.week.isin(list(TEST_WEEKS)) & df.margin.notna()]
    out = {"model": "MR-1", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_train": len(tr), "n_test": len(te),
           "rules": __doc__.split("TAKE rule")[1].split("ML is")[0].strip(), "markets": {}}
    for mk, feats, tgt, base in (("spread", ["mf1", "mf2", "mf3"], "margin", "m_mkt"),
                                 ("total", ["tf1", "tf2", "tf3"], "total", "t_mkt")):
        Xtr, ytr = tr[feats].values, (tr[tgt] - tr[base]).values
        mdl = _ridge(Xtr, ytr)
        tr_pred = _pred(mdl, Xtr)
        te_pred = _pred(mdl, te[feats].values)
        te_res = (te[tgt] - te[base]).values
        c = float(np.corrcoef(te_pred, te_res)[0, 1])
        ci = _boot_corr(te_pred, te_res)
        sel = np.abs(te_pred) >= _GAP_MIN
        res_sel, pred_sel = te_res[sel], te_pred[sel]
        push = res_sel == 0
        hits = (np.sign(pred_sel[~push]) == np.sign(res_sel[~push])).astype(float)
        hit = float(hits.mean()) if len(hits) else None
        passed = bool(ci[0] > 0 and hit is not None and hit > _BE)
        out["markets"][mk] = {
            "features": feats, "beta_std": [round(b, 3) for b in mdl["beta"]], "intercept": round(mdl["b0"], 3),
            "train_corr": round(float(np.corrcoef(tr_pred, ytr)[0, 1]), 3),
            "test_corr": round(c, 3), "test_corr_ci95": [round(x, 3) for x in ci],
            "test_n_gap": int(len(hits)), "test_hit_pct": round(hit * 100, 1) if hit is not None else None,
            "test_pred_sd": round(float(te_pred.std()), 2), "passed": passed, "model": mdl,
            "test_by_week": {int(w): round(float(np.corrcoef(te_pred[te.week.values == w], te_res[te.week.values == w])[0, 1]), 3)
                             for w in sorted(te.week.unique()) if (te.week.values == w).sum() > 3},
        }
    with open(BACKTEST_PATH, "w") as f:
        json.dump(out, f, indent=1)
    df.to_csv(BACKTEST_PATH.replace(".json", "_frame.csv"), index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    r = run()
    for mk, v in r["markets"].items():
        v = {k: x for k, x in v.items() if k != "model"}
        print(mk, json.dumps(v))
    print("n_train", r["n_train"], "n_test", r["n_test"])
