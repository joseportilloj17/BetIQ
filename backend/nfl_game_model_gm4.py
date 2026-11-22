"""
nfl_game_model_gm4.py — GM4: opponent-adjusted EPA ratings + QB layer + situational.

Design FIXED 2026-11-22 before any fitting/evaluation (no tuning on results):

Ratings (re-fit for every target week T, data strictly before T):
  plays      nflverse PBP, pass/run, wp ∈ [0.05, 0.95] (garbage-time filter), epa not null
  window     previous season + current season before T; weight 0.5^(age_weeks / 8)
             (offseason counts as 8 weeks of age)
  pass       epa/dropback  = μp + O_team + Q_qb + D_def + h·home      (ridge λ=_LAM on O, Q, D)
  rush       epa/carry     = μr + O_team + D_def + h·home              (ridge λ=_LAM)
  QB         target game's actual starter (most dropbacks in that game; announced pre-kick);
             unseen QB → Q = _BACKUP_Q_PRIOR
  pace/mix   team pass rate & plays/game: weighted means over the same window
Game layer (coefficients fit once per season on all PRIOR seasons' games):
  margin = b0 + b1·Δe·P + b2·rest_diff + b3·div            (Δe = exp. EPA/play home − away)
  total  = c0 + c1·Σe·P + c2·P + c3·dome + c4·wind15        (P = mean expected plays)
Outputs per game: model_spread_home (= −margin), model_total, plus components.

Evaluation is pre-registered in OPEN_ITEMS NFL-GM4 (run separately, after openers land).
READ-ONLY w.r.t. bets.db / historical.db.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PRED_PATH = os.path.join(_DATA, "nfl_gm4_predictions.csv")
_LAM = 40.0
_HL_WEEKS = 8.0
_OFFSEASON_WEEKS = 8
_BACKUP_Q_PRIOR = -0.05


def load_pbp(seasons):
    import nfl_data_py as nfl
    cols = ["game_id", "season", "week", "posteam", "defteam", "home_team", "away_team", "play_type",
            "epa", "wp", "qb_dropback", "passer_player_id", "passer_player_name", "rusher_player_id"]
    p = nfl.import_pbp_data(list(seasons), columns=cols, downcast=True)
    p = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.wp.between(0.05, 0.95)].copy()
    p["is_pass"] = (p.qb_dropback == 1).astype(int)
    p["home"] = (p.posteam == p.home_team).astype(int)
    p["t"] = p.season * 18 + p.week + (p.season - p.season.min()) * _OFFSEASON_WEEKS
    return p


def _ridge_fit(df, y, cats, lam, w):
    """Weighted ridge on one-hot categorical blocks + 'home'. Returns (mu, {cat: {level: coef}}, h)."""
    idx, cols = {}, 0
    for c in cats:
        lv = pd.Index(df[c].dropna().unique())
        idx[c] = (cols, lv)
        cols += len(lv)
    n = len(df)
    X = np.zeros((n, cols + 1))
    for c in cats:
        off, lv = idx[c]
        pos = lv.get_indexer(df[c])
        ok = pos >= 0
        X[np.arange(n)[ok], off + pos[ok]] = 1.0
    X[:, cols] = df["home"].values
    mu = float(np.average(y, weights=w))
    yc = y - mu
    W = w / w.mean()
    A = (X * W[:, None]).T @ X
    reg = np.full(cols + 1, lam)
    reg[cols] = 1e-3
    A[np.diag_indices_from(A)] += reg
    beta = np.linalg.solve(A, (X * W[:, None]).T @ yc)
    out = {c: dict(zip(idx[c][1], beta[idx[c][0]: idx[c][0] + len(idx[c][1])])) for c in cats}
    return mu, out, float(beta[cols])


def ratings_asof(p, season, week):
    t_target = season * 18 + week + (season - p.season.min()) * _OFFSEASON_WEEKS
    win = p[((p.season == season) & (p.week < week)) | (p.season == season - 1)]
    if win.empty:
        return None
    w_play = 0.5 ** ((t_target - win.t.values) / _HL_WEEKS)
    win = win.assign(w=w_play)
    # aggregate to team-game-unit rows for speed (weighted means)
    pa = win[win.is_pass == 1].groupby(["game_id", "posteam", "defteam", "passer_player_id", "home"]).agg(
        epa=("epa", "mean"), n=("epa", "size"), w=("w", "mean")).reset_index()
    pa = pa[pa.passer_player_id.notna()]
    ru = win[win.is_pass == 0].groupby(["game_id", "posteam", "defteam", "home"]).agg(
        epa=("epa", "mean"), n=("epa", "size"), w=("w", "mean")).reset_index()
    mp, cp, hp = _ridge_fit(pa.rename(columns={"posteam": "O", "defteam": "D", "passer_player_id": "Q"}),
                            pa.epa.values, ["O", "Q", "D"], _LAM, (pa.n * pa.w).values)
    mr, cr, hr = _ridge_fit(ru.rename(columns={"posteam": "O", "defteam": "D"}), ru.epa.values, ["O", "D"],
                            _LAM, (ru.n * ru.w).values)
    g = win.groupby(["game_id", "posteam"]).agg(n=("epa", "size"), pr=("is_pass", "mean"), w=("w", "mean")).reset_index()
    pace = g.groupby("posteam").apply(lambda x: pd.Series({
        "plays": np.average(x.n, weights=x.w), "pass_rate": np.average(x.pr, weights=x.w)}))
    return {"mp": mp, "cp": cp, "hp": hp, "mr": mr, "cr": cr, "hr": hr, "pace": pace}


def expected_epa(r, team, opp, qb, is_home):
    qv = r["cp"]["Q"].get(qb, _BACKUP_Q_PRIOR) if qb is not None else 0.0
    e_pass = r["mp"] + r["cp"]["O"].get(team, 0) + qv + r["cp"]["D"].get(opp, 0) + r["hp"] * is_home
    e_rush = r["mr"] + r["cr"]["O"].get(team, 0) + r["cr"]["D"].get(opp, 0) + r["hr"] * is_home
    pr = float(r["pace"]["pass_rate"].get(team, 0.58))
    plays = float(r["pace"]["plays"].get(team, 62))
    return pr * e_pass + (1 - pr) * e_rush, plays, e_pass, e_rush, qv


def game_features(seasons=(2021, 2022, 2023, 2024, 2025, 2026)):
    import nfl_data_py as nfl
    p = load_pbp(sorted({min(seasons) - 1} | set(seasons)))
    sched = nfl.import_schedules(list(seasons))
    sched = sched[sched.game_type == "REG"]
    starters = (p[p.is_pass == 1].groupby(["game_id", "posteam", "passer_player_id"]).size()
                .reset_index(name="n").sort_values("n").groupby(["game_id", "posteam"]).tail(1)
                .set_index(["game_id", "posteam"])["passer_player_id"])
    rows, t0 = [], time.time()
    for (season, week), games in sched.groupby(["season", "week"]):
        if week < 2:
            continue
        r = ratings_asof(p, season, week)
        if r is None:
            continue
        for _, g in games.iterrows():
            qh = starters.get((g.game_id, g.home_team))
            qa = starters.get((g.game_id, g.away_team))
            eh, ph, *_h = expected_epa(r, g.home_team, g.away_team, qh, 1)
            ea, pa_, *_a = expected_epa(r, g.away_team, g.home_team, qa, 0)
            P = (ph + pa_) / 2
            rows.append({
                "game_id": g.game_id, "season": int(season), "week": int(week),
                "home": g.home_team, "away": g.away_team, "e_home": eh, "e_away": ea, "P": P,
                "de_P": (eh - ea) * P, "se_P": (eh + ea) * P,
                "rest_diff": (g.home_rest or 7) - (g.away_rest or 7), "div": int(g.div_game or 0),
                "dome": int(str(g.roof) in ("dome", "closed")), "wind15": int((g.wind or 0) >= 15),
                "qb_home": qh, "qb_away": qa, "q_home": _h[2], "q_away": _a[2],
                "margin": (g.home_score - g.away_score) if pd.notna(g.home_score) else np.nan,
                "total": (g.home_score + g.away_score) if pd.notna(g.home_score) else np.nan,
                "close_spread_home": -g.spread_line if pd.notna(g.spread_line) else np.nan,
                "close_total": g.total_line,
            })
        logger.warning("[GM4] ratings %d W%d (%.0fs)", season, week, time.time() - t0)
    return pd.DataFrame(rows)


def fit_game_layer(df):
    """Per season S: fit margin/total coefficients on games from seasons < S (OLS)."""
    out = []
    for S in sorted(df.season.unique()):
        tr = df[(df.season < S) & df.margin.notna()]
        te = df[df.season == S].copy()
        if len(tr) < 150:
            continue
        Xm = np.c_[np.ones(len(tr)), tr.de_P, tr.rest_diff, tr["div"]]
        bm = np.linalg.lstsq(Xm, tr.margin.values, rcond=None)[0]
        Xt = np.c_[np.ones(len(tr)), tr.se_P, tr.P, tr.dome, tr.wind15]
        bt = np.linalg.lstsq(Xt, tr.total.values, rcond=None)[0]
        te["model_margin"] = np.c_[np.ones(len(te)), te.de_P, te.rest_diff, te["div"]] @ bm
        te["model_total"] = np.c_[np.ones(len(te)), te.se_P, te.P, te.dome, te.wind15] @ bt
        te["model_spread_home"] = -te.model_margin
        te["coef_margin"] = [list(np.round(bm, 3))] * len(te)
        te["coef_total"] = [list(np.round(bt, 3))] * len(te)
        out.append(te)
    return pd.concat(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    feats = game_features()
    preds = fit_game_layer(feats)
    preds.to_csv(PRED_PATH, index=False)
    d = preds[preds.margin.notna() & preds.close_spread_home.notna()]
    print("games", len(d), "seasons", sorted(d.season.unique()))
    print("descriptive only (eval is pre-registered separately): MAE margin model %.2f vs close %.2f; total model %.2f vs close %.2f"
          % ((d.model_margin - d.margin).abs().mean(), (-d.close_spread_home - d.margin).abs().mean(),
             (d.model_total - d.total).abs().mean(), (d.close_total - d.total).abs().mean()))
    print("corr(model spread, close spread) %.3f; corr(model total, close total) %.3f"
          % (np.corrcoef(d.model_spread_home, d.close_spread_home)[0, 1], np.corrcoef(d.model_total, d.close_total)[0, 1]))
