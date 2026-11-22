"""
nfl_game_model_gamma.py — GAMMA-1: game-level NFL model built from player projections.

Pre-registered 2026-11-22 (before any GAMMA-1 result was seen):

  Team mean   μ_team = c × [6·(Σ pass_td + Σ rush_td over the team's players)
                            + Σ K xpm + 3·Σ K fgm]          (Sleeper weekly projections)
  Team score  ~ shifted gamma via nfl_interim_dispersion.gamma_params_shifted(μ, cv, p_over_at_proj)
  Joint       Gaussian copula on the two team scores, correlation ρ
  c, cv, p_over_at_proj, ρ — fit walk-forward on all prior team-games only.

  Evaluation  identical game set as nfl_game_model (GM1 sim) backtest, same PASS_RULES
              vs closing lines; plus head-to-head vs GM1 on MAE (margin, total) and
              ML Brier with paired bootstrap CIs.

  W12 release gate
    (a) GAMMA-1 replaces GM1 in the Games & Edges "Model" column iff it beats GM1 on
        margin MAE, total MAE and ML Brier (point estimates) — a better descriptive model.
    (b) TAKE flags enabled only for a market whose PASS_RULES test vs the CLOSING line passes.

Sleeper projections are cached to data/sleeper_proj_cache/ (gitignored, durable).
READ-ONLY w.r.t. bets.db / historical.db.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

import nfl_game_model as gm1
from nfl_interim_dispersion import gamma_params_shifted

logger = logging.getLogger(__name__)
_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
_CACHE_DIR = os.path.join(_DATA, "sleeper_proj_cache")
BACKTEST_PATH = os.path.join(_DATA, "nfl_game_model_gamma_backtest.json")
_TEAM_FIX = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC"}


def _get(url, timeout=25):
    import requests
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("[GAMMA-1] fetch failed (%d/3) %s: %s", attempt + 1, url, exc)
            time.sleep(2)
    raise RuntimeError(f"Sleeper fetch failed: {url}")


def sleeper_week(season, week, refresh=False):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{season}_{week:02d}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return json.load(f)
    d = _get(f"https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular")
    with open(path, "w") as f:
        json.dump(d, f)
    return d


def team_points_proj(season, week, refresh=False):
    """{team: bottom-up projected points} from Sleeper player projections."""
    out = {}
    for x in sleeper_week(season, week, refresh):
        st = x.get("stats") or {}
        pos = (x.get("player") or {}).get("position")
        team = _TEAM_FIX.get(x.get("team"), x.get("team"))
        if not team or pos not in ("QB", "RB", "WR", "TE", "K"):
            continue
        if pos == "K":
            pts = st.get("xpm", 0) or 0
            pts += 3 * (st.get("fgm", 0) or 0)
        else:
            pts = 6 * ((st.get("pass_td", 0) or 0) + (st.get("rush_td", 0) or 0))
        out[team] = out.get(team, 0.0) + float(pts)
    return out


def _fit(prior):
    """prior: DataFrame of team-games with columns raw, actual, game_id, side."""
    c = prior["actual"].sum() / prior["raw"].sum()
    mu = c * prior["raw"]
    resid = prior["actual"] / mu
    cv = float(resid.std())
    p_over = float((prior["actual"] > mu).mean())
    # copula rho from normal scores of PIT
    piv = prior.assign(mu=mu).pivot_table(index="game_id", columns="side", values=["actual", "mu"])
    piv = piv.dropna()
    z = {}
    for s in ("home", "away"):
        a, m = piv[("actual", s)].values, piv[("mu", s)].values
        u = np.array([sps.gamma.cdf(ai, *_shape_scale(mi, cv, p_over)) for ai, mi in zip(a, m)])
        z[s] = sps.norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
    rho = float(np.corrcoef(z["home"], z["away"])[0, 1]) if len(piv) > 10 else 0.0
    return {"c": float(c), "cv": cv, "p_over_at_proj": p_over, "rho": rho, "n_team_games": int(len(prior))}


def _shape_scale(mu, cv, p_over):
    a, sc = gamma_params_shifted(mu, cv, p_over)
    return (a, 0, sc)


def sim_game(mu_h, mu_a, fit, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, fit["rho"]], [fit["rho"], 1]], size=n)
    u = sps.norm.cdf(z)
    ah, _, sh = _shape_scale(mu_h, fit["cv"], fit["p_over_at_proj"])
    aa, _, sa = _shape_scale(mu_a, fit["cv"], fit["p_over_at_proj"])
    hs = sps.gamma.ppf(u[:, 0], a=ah, scale=sh)
    as_ = sps.gamma.ppf(u[:, 1], a=aa, scale=sa)
    return {"margin": hs - as_, "total": hs + as_}


def _team_games(seasons, sched):
    rows = []
    for season in seasons:
        for week in sorted(sched[sched.season == season]["week"].unique()):
            proj = team_points_proj(int(season), int(week))
            for _, g in sched[(sched.season == season) & (sched.week == week)].iterrows():
                for side, team, act in (("home", g.home_team, g.home_score), ("away", g.away_team, g.away_score)):
                    if team in proj and proj[team] >= 5:
                        rows.append({"game_id": g.game_id, "season": int(season), "week": int(week), "side": side,
                                     "team": team, "raw": proj[team], "actual": act, "order": int(season) * 100 + int(week)})
    return pd.DataFrame(rows)


def backtest(seasons=(2023, 2024, 2025), min_week=4):
    import nfl_data_py as nfl
    sched = nfl.import_schedules(list(seasons))
    sched = sched[sched.game_type == "REG"]
    tg = _team_games(seasons, sched)
    recs = []
    for season in seasons:
        for week in sorted(sched[sched.season == season]["week"].unique()):
            if week < min_week:
                continue
            order = season * 100 + week
            prior = tg[(tg.order < order) & tg.actual.notna()]
            fit = _fit(prior)
            cur = tg[tg.order == order]
            for _, g in sched[(sched.season == season) & (sched.week == week)].iterrows():
                if pd.isna(g.home_score) or pd.isna(g.spread_line):
                    continue
                h = cur[(cur.game_id == g.game_id) & (cur.side == "home")]
                a = cur[(cur.game_id == g.game_id) & (cur.side == "away")]
                if h.empty or a.empty:
                    continue
                dist = sim_game(fit["c"] * h.raw.iloc[0], fit["c"] * a.raw.iloc[0], fit, seed=hash(g.game_id) % 2**31)
                spread_home = -float(g.spread_line)
                s = gm1.summarize(dist, spread_home, float(g.total_line))
                recs.append({"game_id": g.game_id, "season": int(season), "week": int(week),
                             "home": g.home_team, "away": g.away_team, "spread_home": spread_home,
                             "total_line": float(g.total_line),
                             "mkt_p_home": gm1._ml_prob(g.get("home_moneyline"), g.get("away_moneyline")),
                             "margin": float(g.home_score - g.away_score), "total": float(g.home_score + g.away_score),
                             "proj_margin_mean": float(fit["c"] * (h.raw.iloc[0] - a.raw.iloc[0])),
                             **s, **{f"fit_{k}": v for k, v in fit.items()}})
            logger.warning("[GAMMA-1] %d W%d done (%d games)", season, week, len(recs))
    return recs


def head_to_head(g_recs, s_csv):
    g = pd.DataFrame(g_recs)
    s = pd.read_csv(s_csv)
    m = g.merge(s[["game_id", "median_margin", "median_total", "p_home_win"]], on="game_id", suffixes=("", "_sim"))
    m = m[m.margin != 0]
    y = (m.margin > 0).astype(float)
    rng = np.random.default_rng(21)
    idx = rng.integers(0, len(m), size=(4000, len(m)))

    def paired(a, b):
        d = (a - b).values
        bs = d[idx].mean(axis=1)
        return {"gamma": round(float(a.mean()), 4), "sim": round(float(b.mean()), 4), "diff": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}

    out = {
        "n_games": int(len(m)),
        "margin_mae": paired((m.median_margin - m.margin).abs(), (m.median_margin_sim - m.margin).abs()),
        "total_mae": paired((m.median_total - m.total).abs(), (m.median_total_sim - m.total).abs()),
        "ml_brier": paired((m.p_home_win - y) ** 2, (m.p_home_win_sim - y) ** 2),
        "market_margin_mae": round(float((-m.spread_home - m.margin).abs().mean()), 3),
        "market_total_mae": round(float((m.total_line - m.total).abs().mean()), 3),
        "corr_proj_margin_vs_close": round(float(np.corrcoef(m.median_margin, -m.spread_home)[0, 1]), 3),
        "corr_proj_total_vs_close": round(float(np.corrcoef(m.median_total, m.total_line)[0, 1]), 3),
    }
    out["beats_sim_all3"] = bool(out["margin_mae"]["diff"] < 0 and out["total_mae"]["diff"] < 0 and out["ml_brier"]["diff"] < 0)
    return out


_LIVE: dict = {}


def predict_upcoming(season=2026):
    """{(home, away): summary+dist} for the next unplayed week, fit on all prior team-games."""
    import nfl_data_py as nfl
    key = (season, time.strftime("%Y-%m-%d-%H"))
    if key in _LIVE:
        return _LIVE[key]
    sched = nfl.import_schedules([season - 1, season])
    sched = sched[sched.game_type == "REG"]
    up = sched[(sched.season == season) & sched.home_score.isna()]
    if up.empty:
        return {}
    week = int(up.week.min())
    tg = _team_games([season - 1, season], sched)
    prior = tg[(tg.order < season * 100 + week) & tg.actual.notna()]
    fit = _fit(prior)
    proj = team_points_proj(season, week, refresh=True)
    out = {"_meta": {"model": "GAMMA-1", "season": season, "week": week, **fit}}
    for _, g in up[up.week == week].iterrows():
        if g.home_team in proj and g.away_team in proj:
            dist = sim_game(fit["c"] * proj[g.home_team], fit["c"] * proj[g.away_team], fit, seed=hash(g.game_id) % 2**31)
            out[(g.home_team, g.away_team)] = {"dist": dist, **gm1.summarize(dist)}
    _LIVE.clear()
    _LIVE[key] = out
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    recs = backtest()
    res = gm1.evaluate(recs)
    res["model"] = "GAMMA-1"
    res["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    res["head_to_head_vs_sim"] = head_to_head(recs, gm1.BACKTEST_PATH.replace(".json", "_games.csv"))
    res["fit_last"] = {k[4:]: recs[-1][k] for k in recs[-1] if k.startswith("fit_")}
    with open(BACKTEST_PATH, "w") as f:
        json.dump(res, f, indent=1)
    pd.DataFrame(recs).to_csv(BACKTEST_PATH.replace(".json", "_games.csv"), index=False)
    print(json.dumps({k: res[k] for k in ("n_games", "head_to_head_vs_sim", "fit_last", "calibration")}, indent=1))
    for mk in ("spread", "total", "moneyline"):
        print(mk, "PASSED" if res["markets"][mk]["passed"] else "FAILED")
