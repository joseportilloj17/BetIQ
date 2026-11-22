"""
nfl_game_model.py — Independent (market-free) NFL game model for Games & Edges.

Built on the archived drive-level sim (nfl_game_sim.py) but run UNANCHORED, so
nothing about the game's own market line enters the prediction:

  1. Offense profiles  — nfl_game_sim._build_profiles_from_plays (rolling games).
  2. Defense ratios    — per defteam allowed-rate / league rate for comp%, Y/C,
                         sack%, INT%, YPC, fumble% over the same rolling window,
                         shrunk toward 1.0 with k=_DEF_SHRINK_K games.
  3. Matchup profile   — offense efficiency × opponent defense ratio.
  4. Sim unanchored (total=0 → anchors = 1.0), then post-hoc
       scores × c   (c = league actual pts/team ÷ sim unanchored mean)
       margin ± HFA/2 (HFA measured from prior seasons' actual margins).
  5. Output: P(home win), median/mean margin, median/mean total, and
     P(cover)/P(over) at any line from the simulated distribution.

Walk-forward backtest (`backtest`) builds every input from plays strictly
before the game's week.  Pre-registered pass rules (2026-11-22, before any
result was seen) are in PASS_RULES; Games & Edges only shows a "take" flag for
a market whose rule PASSED (results persisted in data/nfl_game_model_backtest.json).

Known gaps: no QB-change/injury model (team offense profile only), no weather,
no OT, sim pts/yard ~93% of actual (handled by c).
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import nfl_game_sim as gs

logger = logging.getLogger(__name__)

_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
BACKTEST_PATH = os.path.join(_DATA, "nfl_game_model_backtest.json")
_DEF_SHRINK_K = 4.0
_DEF_FIELDS = ("comp_rate", "ypa_comp", "sack_rate", "int_rate", "ypc", "fumble_rate")

PASS_RULES = {
    "registered": "2026-11-22 (before backtest run)",
    "spread": "Some |gap| bucket (model median margin vs closing spread, pts) with N>=100 has ATS hit-rate "
              "95% CI lower bound > 52.38% after Bonferroni over buckets, AND same-sign (>52.38%) in each of "
              "2023, 2024, 2025.",
    "total": "Same as spread on (model median total − closing total).",
    "moneyline": "Model Brier < devigged closing-ML Brier with paired-bootstrap 95% CI excluding 0, "
                 "AND model better in each season.",
}
GAP_BUCKETS = [1.0, 2.0, 3.0, 4.0, 5.0]


# ── Play preprocessing (same as nfl_game_sim.validate_sim) ─────────────────────

def load_plays(seasons):
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data(seasons)
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    plays["situation"] = [
        gs._classify_situation(sd or 0, int(q or 1), g or 1800)
        for sd, q, g in zip(plays["score_differential"].fillna(0), plays["qtr"].fillna(1),
                            plays["game_seconds_remaining"].fillna(1800))]
    plays.sort_values(["game_id", "game_seconds_remaining"], ascending=[True, False], inplace=True)
    plays["next_gsr"] = plays.groupby(["game_id", "posteam", "fixed_drive"])["game_seconds_remaining"].shift(-1)
    plays["snap_gap"] = plays["game_seconds_remaining"] - plays["next_gsr"]
    plays.loc[(plays["snap_gap"] <= 0) | (plays["snap_gap"] > 55), "snap_gap"] = np.nan
    plays["is_dropback"] = ((plays["pass_attempt"] == 1) | (plays["qb_scramble"] == 1) |
                            (plays["sack"] == 1)).astype(float)
    plays["is_completion"] = ((plays["pass_attempt"] == 1) & (plays["complete_pass"] == 1)).astype(float)
    plays["comp_yards"] = np.where(plays["is_completion"] == 1, plays["yards_gained"], np.nan)
    if "no_huddle" not in plays.columns:
        plays["no_huddle"] = 0.0
    plays["_order"] = plays["season"] * 100 + plays["week"]
    return plays


def _rates(df):
    db = df[df["is_dropback"] == 1]
    pa = df[df["pass_attempt"] == 1]
    ru = df[(df["play_type"] == "run") & (df["qb_scramble"] != 1)]
    return {
        "comp_rate": pa["is_completion"].mean() if len(pa) else np.nan,
        "ypa_comp": df["comp_yards"].mean(),
        "sack_rate": db["sack"].mean() if len(db) else np.nan,
        "int_rate": pa["interception"].mean() if len(pa) else np.nan,
        "ypc": ru["yards_gained"].mean() if len(ru) else np.nan,
        "fumble_rate": df["fumble_lost"].mean() if "fumble_lost" in df else np.nan,
    }


def build_inputs(prior_plays):
    """Offense profiles + shrunk defense ratios from plays strictly before the target week."""
    off = gs._build_profiles_from_plays(prior_plays)
    lg = _rates(prior_plays)
    dfn = {}
    for team, tp in prior_plays.groupby("defteam"):
        gids = tp.drop_duplicates("game_id").sort_values("game_id", ascending=False)["game_id"].values
        recent = tp[tp["game_id"].isin(set(gids[:gs._PROFILE_ROLLING_GAMES]))]
        n = recent["game_id"].nunique()
        r = _rates(recent)
        w = n / (n + _DEF_SHRINK_K)
        dfn[team] = {f: (1.0 + w * (r[f] / lg[f] - 1.0)) if (lg[f] and not np.isnan(r[f])) else 1.0
                     for f in _DEF_FIELDS}
    return off, dfn


def matchup_profile(off_prof, def_ratio):
    p = dict(off_prof)
    if not def_ratio:
        return p
    p["comp_rate"] = float(np.clip(off_prof["comp_rate"] * def_ratio["comp_rate"], 0.45, 0.80))
    p["ypa_comp"] = max(6.0, off_prof["ypa_comp"] * def_ratio["ypa_comp"])
    p["sack_rate"] = float(np.clip(off_prof["sack_rate"] * def_ratio["sack_rate"], 0.01, 0.20))
    p["int_rate"] = float(np.clip(off_prof["int_rate"] * def_ratio["int_rate"], 0.005, 0.08))
    p["ypc"] = max(2.5, off_prof["ypc"] * def_ratio["ypc"])
    p["fumble_rate"] = float(np.clip(off_prof["fumble_rate"] * def_ratio["fumble_rate"], 0.0, 0.05))
    return p


def sim_game(home, away, off, dfn, c, hfa, n_sims=300, seed=7):
    if home not in off or away not in off:
        return None
    hp = matchup_profile(off[home], dfn.get(away))
    ap = matchup_profile(off[away], dfn.get(home))
    r = gs.simulate_game_n(n_sims, hp, ap, None, None, spread=0.0, total=0.0, seed=seed)
    hs = r["home_scores"] * c + hfa / 2.0
    as_ = r["away_scores"] * c - hfa / 2.0
    return {"margin": hs - as_, "total": hs + as_}


def summarize(dist, spread_home=None, total_line=None):
    """spread_home: home spread (negative = home favoured), e.g. -3.5."""
    m, t = dist["margin"], dist["total"]
    out = {
        "p_home_win": float(np.mean(m > 0) + 0.5 * np.mean(m == 0)),
        "median_margin": float(np.median(m)), "mean_margin": float(np.mean(m)),
        "median_total": float(np.median(t)), "mean_total": float(np.mean(t)),
        "model_spread_home": round(-float(np.median(m)) * 2) / 2,  # nearest half-point
        "model_total": round(float(np.median(t)) * 2) / 2,
    }
    if spread_home is not None:
        cov = m + spread_home
        out["p_home_cover"] = float(np.mean(cov > 0) / max(1e-9, 1 - np.mean(cov == 0)))
        out["spread_gap"] = float(np.median(m) + spread_home)  # +ve = model likes home vs line
    if total_line is not None:
        out["p_over"] = float(np.mean(t > total_line) / max(1e-9, 1 - np.mean(t == total_line)))
        out["total_gap"] = float(np.median(t) - total_line)
    return out


def _calib_c(prior_plays, n_sims=400):
    """c = actual mean pts/team in prior games ÷ sim unanchored league-average mean."""
    if not gs._SIM_LG_MEAN_SCORE or gs._SIM_LG_MEAN_SCORE <= 0:
        gs._calibrate(n_sims=n_sims)
    g = prior_plays.drop_duplicates("game_id")
    actual = float(((g["home_score"] + g["away_score"]) / 2).mean())
    return actual / gs._SIM_LG_MEAN_SCORE, actual


def _hfa(sched_prior):
    s = sched_prior.dropna(subset=["home_score", "away_score"])
    s = s[s["location"] != "Neutral"] if "location" in s else s
    return float((s["home_score"] - s["away_score"]).mean())


def _ml_prob(home_ml, away_ml):
    def imp(a):
        return 100 / (a + 100) if a > 0 else -a / (-a + 100)
    if pd.isna(home_ml) or pd.isna(away_ml):
        return None
    h, a = imp(home_ml), imp(away_ml)
    return h / (h + a)


# ── Walk-forward backtest ───────────────────────────────────────────────────────

def backtest(seasons=(2023, 2024, 2025), min_week=4, n_sims=300, progress=True):
    import nfl_data_py as nfl
    load_seasons = sorted({s - 1 for s in seasons} | set(seasons))
    plays = load_plays(load_seasons)
    sched = nfl.import_schedules(load_seasons)
    sched = sched[sched["game_type"] == "REG"]
    recs = []
    t0 = time.time()
    for season in seasons:
        for week in sorted(sched[(sched.season == season)]["week"].unique()):
            if week < min_week:
                continue
            order = season * 100 + week
            prior = plays[plays["_order"] < order]
            prior = prior[prior["_order"] >= (season - 1) * 100 + 1]  # last season + this season
            off, dfn = build_inputs(prior)
            c, lg_pts = _calib_c(prior)
            hfa = _hfa(sched[(sched.season * 100 + sched.week) < order].tail(600))
            games = sched[(sched.season == season) & (sched.week == week)]
            for _, gm in games.iterrows():
                if pd.isna(gm["home_score"]) or pd.isna(gm["spread_line"]):
                    continue
                dist = sim_game(gm["home_team"], gm["away_team"], off, dfn, c, hfa, n_sims=n_sims,
                                seed=hash(gm["game_id"]) % 2**31)
                if dist is None:
                    continue
                spread_home = -float(gm["spread_line"])   # nfl_data_py spread_line is home-minus-away margin line
                s = summarize(dist, spread_home, float(gm["total_line"]))
                margin = gm["home_score"] - gm["away_score"]
                tot = gm["home_score"] + gm["away_score"]
                recs.append({
                    "game_id": gm["game_id"], "season": int(season), "week": int(week),
                    "home": gm["home_team"], "away": gm["away_team"],
                    "spread_home": spread_home, "total_line": float(gm["total_line"]),
                    "mkt_p_home": _ml_prob(gm.get("home_moneyline"), gm.get("away_moneyline")),
                    "margin": float(margin), "total": float(tot), **s,
                    "c": c, "hfa": hfa,
                })
            if progress:
                logger.warning("[backtest] %d W%d done (%d games total, %.0fs)", season, week, len(recs),
                               time.time() - t0)
    return recs


def _hit_ci(hits, n_boot=4000, seed=3):
    a = np.array(hits, dtype=float)
    if len(a) < 2:
        return None
    rng = np.random.default_rng(seed)
    bs = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def evaluate(recs):
    df = pd.DataFrame(recs)
    BE = 0.5238
    out = {"n_games": len(df), "rules": PASS_RULES, "markets": {}}
    nb = len(GAP_BUCKETS)
    alpha_ci = 1 - 0.05 / nb   # Bonferroni
    for mkt, gap, res_fn in (
        ("spread", "spread_gap", lambda d: np.where(d.spread_gap > 0, d.margin + d.spread_home > 0,
                                                     d.margin + d.spread_home < 0)),
        ("total", "total_gap", lambda d: np.where(d.total_gap > 0, d.total > d.total_line, d.total < d.total_line)),
    ):
        push = (df.margin + df.spread_home == 0) if mkt == "spread" else (df.total == df.total_line)
        d = df[~push].copy()
        d["hit"] = res_fn(d).astype(float)
        buckets, passed = [], False
        for b in GAP_BUCKETS:
            sub = d[d[gap].abs() >= b]
            if not len(sub):
                continue
            rng = np.random.default_rng(5)
            bs = rng.choice(sub.hit.values, size=(4000, len(sub)), replace=True).mean(axis=1)
            lo = float(np.percentile(bs, 100 * (1 - alpha_ci))) if len(sub) > 1 else None
            per_season = {int(s): round(float(g.hit.mean()) * 100, 1) for s, g in sub.groupby("season")}
            ok = (len(sub) >= 100 and lo is not None and lo > BE and all(v > BE * 100 for v in per_season.values())
                  and len(per_season) == 3)
            passed |= ok
            buckets.append({"min_gap": b, "n": int(len(sub)), "hit_pct": round(float(sub.hit.mean()) * 100, 1),
                            "ci95": [round(x * 100, 1) for x in _hit_ci(sub.hit.values)] if len(sub) > 1 else None,
                            "bonf_lower": round(lo * 100, 1) if lo is not None else None,
                            "per_season": per_season, "pass": ok})
        corr = float(np.corrcoef(d[gap], (d.margin + d.spread_home) if mkt == "spread" else (d.total - d.total_line))[0, 1])
        out["markets"][mkt] = {"buckets": buckets, "passed": bool(passed), "breakeven_pct": BE * 100,
                               "corr_gap_vs_result_vs_line": round(corr, 3)}
    # moneyline
    m = df.dropna(subset=["mkt_p_home"]).copy()
    y = (m.margin > 0).astype(float)
    m = m[m.margin != 0]
    y = (m.margin > 0).astype(float)
    bd = (m.p_home_win - y) ** 2 - (m.mkt_p_home - y) ** 2
    rng = np.random.default_rng(9)
    idx = rng.integers(0, len(bd), size=(4000, len(bd)))
    bs = bd.values[idx].mean(axis=1)
    per_season = {int(s): round(float(((g.p_home_win - (g.margin > 0)) ** 2 - (g.mkt_p_home - (g.margin > 0)) ** 2).mean()), 4)
                  for s, g in m.groupby("season")}
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["markets"]["moneyline"] = {
        "n": int(len(m)), "brier_model": round(float(((m.p_home_win - y) ** 2).mean()), 4),
        "brier_market": round(float(((m.mkt_p_home - y) ** 2).mean()), 4),
        "diff": round(float(bd.mean()), 4), "ci95": [round(x, 4) for x in ci], "per_season_diff": per_season,
        "passed": bool(ci[1] < 0 and all(v < 0 for v in per_season.values())),
    }
    out["calibration"] = {
        "mean_model_margin": round(float(df.mean_margin.mean()), 2), "mean_actual_margin": round(float(df.margin.mean()), 2),
        "mean_model_total": round(float(df.mean_total.mean()), 2), "mean_actual_total": round(float(df.total.mean()), 2),
        "mae_margin_model": round(float((df.median_margin - df.margin).abs().mean()), 2),
        "mae_margin_market": round(float((-df.spread_home - df.margin).abs().mean()), 2),
        "mae_total_model": round(float((df.median_total - df.total).abs().mean()), 2),
        "mae_total_market": round(float((df.total_line - df.total).abs().mean()), 2),
    }
    return out


# ── Live predictions for upcoming games ────────────────────────────────────────

_LIVE_CACHE: dict = {}


def predict_upcoming(season=2026, n_sims=500):
    """{(home_abbr, away_abbr): summary-without-lines} for the next unplayed week."""
    import nfl_data_py as nfl
    key = (season, time.strftime("%Y-%m-%d-%H"))
    if key in _LIVE_CACHE:
        return _LIVE_CACHE[key]
    sched = nfl.import_schedules([season - 1, season])
    sched = sched[sched["game_type"] == "REG"]
    up = sched[(sched.season == season) & sched.home_score.isna()]
    if up.empty:
        return {}
    week = int(up.week.min())
    plays = load_plays([season - 1, season])
    prior = plays[plays["_order"] < season * 100 + week]
    off, dfn = build_inputs(prior)
    c, _ = _calib_c(prior)
    hfa = _hfa(sched[(sched.season * 100 + sched.week) < season * 100 + week].tail(600))
    out = {"_meta": {"season": season, "week": week, "c": c, "hfa": hfa, "n_sims": n_sims}}
    for _, gm in up[up.week == week].iterrows():
        dist = sim_game(gm["home_team"], gm["away_team"], off, dfn, c, hfa, n_sims=n_sims,
                        seed=hash(gm["game_id"]) % 2**31)
        if dist is not None:
            out[(gm["home_team"], gm["away_team"])] = {"dist": dist, **summarize(dist)}
    _LIVE_CACHE.clear()
    _LIVE_CACHE[key] = out
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "time"])
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument("--seasons", default="2023,2024,2025")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if a.cmd == "time":
        plays = load_plays([2025])
        off, dfn = build_inputs(plays[plays.week < 10])
        c, lg = _calib_c(plays[plays.week < 10])
        t = time.time()
        d = sim_game("KC", "BUF", off, dfn, c, 1.8, n_sims=a.n_sims)
        print(f"{a.n_sims} sims in {time.time()-t:.1f}s  c={c:.3f} lg_pts={lg:.1f}", summarize(d, -2.5, 47.5))
    else:
        recs = backtest(tuple(int(x) for x in a.seasons.split(",")), n_sims=a.n_sims)
        res = evaluate(recs)
        res["n_sims"] = a.n_sims
        res["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(BACKTEST_PATH, "w") as f:
            json.dump(res, f, indent=1)
        pd.DataFrame(recs).to_csv(BACKTEST_PATH.replace(".json", "_games.csv"), index=False)
        print(json.dumps(res, indent=1))
