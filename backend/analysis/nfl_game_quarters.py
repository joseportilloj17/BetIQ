"""
Per-quarter breakdown + rationale for one game from the GM1 unanchored sim (read-only).

GM1 FAILED its pre-registered backtest vs closing lines (OPEN_ITEMS NFL-GM1) — this
explains what the model believes; it is not a bet signal.

Usage: python analysis/nfl_game_quarters.py --home DAL --away TEN [--n 4000]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nfl_game_model as gm1  # noqa: E402
import nfl_game_sim as gs  # noqa: E402
import nfl_game_model_gamma as ngg  # noqa: E402

_LOG = []
_orig = gs._sim_drive


def _wrapped(rng, profile, qb, **kw):
    d = _orig(rng, profile, qb, **kw)
    _LOG.append((profile.get("_side"), kw.get("qtr"), d["outcome"], len(d["plays"]),
                 sum(1 for p in d["plays"] if p.get("is_pass") or p.get("play_type") == "pass")))
    return d


gs._sim_drive = _wrapped


def main(home, away, n, season=2026):
    import nfl_data_py as nfl
    sched = nfl.import_schedules([season - 1, season])
    sched = sched[sched.game_type == "REG"]
    up = sched[(sched.home_team == home) & (sched.away_team == away) & (sched.season == season)]
    week = int(up.week.iloc[0])
    plays = gm1.load_plays([season - 1, season])
    prior = plays[plays["_order"] < season * 100 + week]
    off, dfn = gm1.build_inputs(prior)
    c, lg = gm1._calib_c(prior)
    hfa = gm1._hfa(sched[(sched.season * 100 + sched.week) < season * 100 + week].tail(600))
    hp = gm1.matchup_profile(off[home], dfn.get(away)); hp["_side"] = "home"
    ap = gm1.matchup_profile(off[away], dfn.get(home)); ap["_side"] = "away"

    rng = np.random.default_rng(11)
    q_pts = {s: np.zeros((n, 4)) for s in ("home", "away")}
    finals = {s: np.zeros(n) for s in ("home", "away")}
    drives = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        _LOG.clear()
        r = gs.simulate_game(rng, hp, ap, None, None, 0.0, 0.0)
        raw = {"home": np.zeros(4), "away": np.zeros(4)}
        for side, qtr, outc, nplays, npass in _LOG:
            q = min(max(int(qtr or 1), 1), 4) - 1
            raw[side][q] += 7 if outc == "td" else 3 if outc == "fg" else 0
            drives[side]["drives"] += 1
            drives[side][outc] += 1
            drives[side]["plays"] += nplays
        for side, sc in (("home", r["home_score"]), ("away", r["away_score"])):
            final = sc * c + (hfa / 2 if side == "home" else -hfa / 2)
            finals[side][i] = final
            tot = raw[side].sum()
            q_pts[side][i] = raw[side] / tot * final if tot > 0 else final / 4
    m = finals["home"] - finals["away"]
    t = finals["home"] + finals["away"]
    print(f"\n{away} @ {home}  — 2026 W{week}, GM1 unanchored sim, N={n} (c={c:.3f}, HFA={hfa:.2f}, lg pts/team {lg:.1f})")
    print(f"Final (median): {away} {np.median(finals['away']):.1f} – {home} {np.median(finals['home']):.1f} | "
          f"mean margin {home} {m.mean():+.1f} | P({home} win) {np.mean(m > 0):.1%} | "
          f"P({home} cover -7.5) {np.mean(m > 7.5):.1%} | median total {np.median(t):.1f}")
    print("\nPoints by quarter (mean):      Q1    Q2    Q3    Q4   | 1H    2H")
    for side, tm in (("away", away), ("home", home)):
        q = q_pts[side].mean(0)
        print(f"  {tm:<4}                       {q[0]:5.1f} {q[1]:5.1f} {q[2]:5.1f} {q[3]:5.1f}  | {q[:2].sum():5.1f} {q[2:].sum():5.1f}")
    cum_h = np.cumsum(q_pts["home"], axis=1)
    cum_a = np.cumsum(q_pts["away"], axis=1)
    print("\nGame state after each quarter:   Q1     HALF   Q3")
    for k, lab in ((0, "Q1"), (1, "HALF"), (2, "Q3")):
        pass
    lead = cum_h - cum_a
    print(f"  {home} leads (P)               {np.mean(lead[:,0]>0):.0%}   {np.mean(lead[:,1]>0):.0%}   {np.mean(lead[:,2]>0):.0%}")
    print(f"  {home} leads by 8+ (P)         {np.mean(lead[:,0]>=8):.0%}   {np.mean(lead[:,1]>=8):.0%}   {np.mean(lead[:,2]>=8):.0%}")
    print(f"  {away} leads by 8+ (P)         {np.mean(lead[:,0]<=-8):.0%}   {np.mean(lead[:,1]<=-8):.0%}   {np.mean(lead[:,2]<=-8):.0%}")
    for i, lab in ((1, "halftime"),):
        up8 = lead[:, i] >= 8
        if up8.any():
            print(f"  If {home} up 8+ at {lab} ({up8.mean():.0%} of sims): {home} wins {np.mean(m[up8] > 0):.0%}, covers -7.5 {np.mean(m[up8] > 7.5):.0%}")
        close = np.abs(lead[:, i]) < 4
        print(f"  If within 3 at {lab} ({close.mean():.0%}): {home} wins {np.mean(m[close] > 0):.0%}, covers -7.5 {np.mean(m[close] > 7.5):.0%}")

    print("\nDrive outcomes per game (mean):  drives   TD    FG   punt  TO   plays")
    for side, tm in (("away", away), ("home", home)):
        d = drives[side]
        print(f"  {tm:<4}                          {d['drives']/n:5.1f} {d['td']/n:5.2f} {d['fg']/n:5.2f} {d.get('punt',0)/n:5.2f} "
              f"{d.get('turnover',0)/n:5.2f} {d['plays']/n:6.1f}")

    def show(team, opp):
        o, dr = off[team], dfn.get(opp, {})
        lgr = gm1._rates(prior)
        print(f"\n{team} offense (last {o.get('n_games')} games) vs {opp} defense (allowed/league ratio, shrunk):")
        for f, lab in (("comp_rate", "comp %"), ("ypa_comp", "yds/completion"), ("sack_rate", "sack %"),
                       ("int_rate", "INT %"), ("ypc", "yds/carry"), ("fumble_rate", "fumble %")):
            print(f"  {lab:<15} off {o[f]:.3f} (lg {lgr[f]:.3f})  × {opp} def ratio {dr.get(f, 1):.2f}"
                  f"  → matchup {gm1.matchup_profile(o, dr)[f]:.3f}")
        print(f"  proe {o.get('proe', 0):+.3f}  rel. efficiency (sim) {gs._estimate_rel_eff(o):.3f} → vs {opp}: {gs._estimate_rel_eff(gm1.matchup_profile(o, dr)):.3f}")

    show(away, home)
    show(home, away)
    qb = prior[prior.posteam.isin([home, away]) & (prior.pass_attempt == 1)]
    recent = qb[qb._order >= qb._order.max() - 8]
    print("\nPassers in profile window (attempts):")
    for tm in (away, home):
        s = recent[recent.posteam == tm].groupby("passer_player_name").size().sort_values(ascending=False).head(3)
        print(f"  {tm}: " + ", ".join(f"{k} {v}" for k, v in s.items()))
    try:
        p = ngg.team_points_proj(season, week)
        print(f"\nSleeper bottom-up projected pts (raw): {away} {p.get(away, 0):.1f}, {home} {p.get(home, 0):.1f}")
    except Exception as exc:
        print("Sleeper projection unavailable:", exc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    main(a.home, a.away, a.n)
