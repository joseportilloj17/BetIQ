"""
FW-2c / FW-2d / FW-2e pre-registered rule evaluator (read-only).

Implements the qualifying criteria registered 2026-11-15 (commit 8a172b3,
see docs/OPEN_ITEMS.md "FW-2c/2d/2e") against fantasy_forward_preds.

READ-ONLY: opens historical.db with mode=ro.  Never writes.  Ungraded weeks
(e.g. a TNF game before the Tuesday grader runs) are graded in memory from
Sleeper actuals.

Usage:
    python analysis/nfl_fw2_rules.py history            # per-week table W2..current
    python analysis/nfl_fw2_rules.py slate --week 11    # upcoming qualifying picks

Notes:
  * Price = FanDuel decimal price at the same main line, latest fetch before
    kickoff (player_props).  Rows with no FD price at that line are excluded
    and counted (fd_missing).  The stored prop_over_price is "latest book
    fetched", not FanDuel, so it is NOT used for the price band / ROI.
  * Injury filter (criterion 9) cannot be reconstructed historically; history
    mode is pre-injury-filter.  Slate mode prints Sleeper injury status.
  * Staking caps (1/player, 2/game, 5u/slate) apply to live bets only; the
    tracked population is uncapped.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fantasy_projections import (  # noqa: E402
    STAT_TO_PROP, _normalize_player_name, _fetch_sleeper_actuals, _sleeper_player_db,
)

logger = logging.getLogger("fw2_rules")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

HIST_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical.db"))
YARDAGE = ("pass_yds", "rush_yds", "rec_yds")


def _ro():
    c = sqlite3.connect(f"file:{HIST_DB}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def load_preds(conn, season, weeks=None):
    q = """SELECT * FROM fantasy_forward_preds
           WHERE season=? AND method='distributional' AND line_depth='main'
             AND data_error IS NULL"""
    args = [season]
    if weeks:
        q += f" AND week IN ({','.join('?' * len(weeks))})"
        args += list(weeks)
    rows = [dict(r) for r in conn.execute(q, args)]
    # dedup to one row per (week, player, stat): keep latest pre-kickoff created_at
    best = {}
    for r in rows:
        k = (r["week"], _normalize_player_name(r["player_name"]), r["stat_type"])
        if k not in best or (r["created_at"] or "") > (best[k]["created_at"] or ""):
            best[k] = r
    return list(best.values())


def load_fd_prices(conn, game_dates):
    """{(norm_name, market_key, point, side): [(fetched_at, price), ...]} for FanDuel."""
    q = f"""SELECT player_name, market_key, point, over_under, price, fetched_at
            FROM player_props
            WHERE sport_key='americanfootball_nfl' AND bookmaker='fanduel'
              AND market_key IN ({','.join('?' * len(STAT_TO_PROP))})
              AND game_date IN ({','.join('?' * len(game_dates))})"""
    out = defaultdict(list)
    for r in conn.execute(q, list(STAT_TO_PROP.values()) + list(game_dates)):
        out[(_normalize_player_name(r[0]), r[1], float(r[2]), r[3].lower())].append((r[5], r[4]))
    return out


PRICE_MODE = "fd"


def fd_price(fd, r, side):
    if PRICE_MODE == "stored":
        return r["prop_over_price"] if side == "over" else r["prop_under_price"]
    key = (_normalize_player_name(r["player_name"]), STAT_TO_PROP[r["stat_type"]], float(r["prop_line"]), side)
    ko = r["kickoff_utc"] or "9999"
    cands = [(t, p) for t, p in fd.get(key, []) if t < ko]
    return max(cands)[1] if cands else None


def classify(r, fd, now_iso=None):
    """Return (rule, side, model_p, mkt_p, edge_pp, price) or None."""
    st = r["stat_type"]
    if st not in STAT_TO_PROP or r["devigged_implied"] is None or r["prop_line"] is None:
        return None
    pov = r["p_over_v12_gamma"] if st in YARDAGE else r["p_over"]
    if pov is None:
        return None
    line = float(r["prop_line"])
    side = "over" if pov > 0.5 else "under"
    model_p = pov if side == "over" else 1 - pov
    mkt_p = r["devigged_implied"] if side == "over" else 1 - r["devigged_implied"]
    edge = (model_p - mkt_p) * 100
    # stale
    cm = r["consensus_mean"]
    if cm is not None and line >= 5 and abs(cm - line) > 0.40 * line and abs(cm - line) > 10:
        return None
    if now_iso and (r["kickoff_utc"] or "") <= now_iso:
        return None
    price = fd_price(fd, r, side)
    is_qb_rush = r["position"] == "QB" and st == "rush_yds"
    low = (st in YARDAGE and line < 10) or (st == "receptions" and line <= 1.5)

    def ok_price():
        return price is not None and price <= 2.0 and model_p * (price - 1) - (1 - model_p) > 0

    if is_qb_rush:
        rule = "FW-2d" if edge >= 3.0 else None
    elif low:
        rule = "FW-2c" if (side == "over" and edge >= 3.0) else None
    else:
        rule = "FW-2e" if edge > 15.0 else None
    if rule is None:
        return None
    if price is None:
        return (rule, side, model_p, mkt_p, edge, None)  # counted as fd_missing
    if not ok_price():
        return None
    return (rule, side, model_p, mkt_p, edge, price)


def outcome(actual, line, side, price):
    if actual is None:
        return None
    if actual == line:
        return 0.0, None
    win = actual > line if side == "over" else actual < line
    return ((price - 1.0) if win else -1.0), int(win)


def boot_ci(picks, n=4000, seed=7):
    games = defaultdict(list)
    for p in picks:
        games[p["game_id"] or p["player"]].append(p["roi"])
    keys = list(games)
    if len(keys) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        s = [x for k in (rng.choice(keys) for _ in keys) for x in games[k]]
        stats.append(sum(s) / len(s))
    stats.sort()
    return stats[int(0.025 * n)] * 100, stats[int(0.975 * n)] * 100


def summarize(picks):
    res = [p for p in picks if p["is_win"] is not None]
    n = len(res)
    if not n:
        return dict(n=0, w=0, wr=float("nan"), roi=float("nan"), be=float("nan"), ci=(float("nan"),) * 2)
    w = sum(p["is_win"] for p in res)
    return dict(n=n, w=w, wr=100 * w / n, roi=100 * sum(p["roi"] for p in res) / n,
                be=100 * sum(1 / p["price"] for p in res) / n, ci=boot_ci(res),
                games=len({p["game_id"] for p in res}))


def build_picks(season, weeks, grade_ungraded=True, only_game_dates=None):
    conn = _ro()
    preds = load_preds(conn, season, weeks)
    dates = sorted({(r["kickoff_utc"] or "")[:10] for r in preds} |
                   {d for d in conn.execute(
                       "SELECT DISTINCT game_date FROM player_props WHERE sport_key='americanfootball_nfl'").fetchall()
                    for d in d})
    fd = load_fd_prices(conn, [d for d in dates if d])
    conn.close()
    actual_cache = {}
    picks, fd_missing = [], defaultdict(int)
    for r in preds:
        c = classify(r, fd)
        if not c:
            continue
        rule, side, mp, mk, edge, price = c
        if price is None:
            fd_missing[(r["week"], rule)] += 1
            continue
        actual = r["actual_value"] if r["graded_at"] else None
        if actual is None and grade_ungraded:
            wk = r["week"]
            if wk not in actual_cache:
                actual_cache[wk] = _fetch_sleeper_actuals(season, wk)
            a = actual_cache[wk].get(_normalize_player_name(r["player_name"]), {})
            # only grade games that have kicked off >4h ago (finished)
            ko = r["kickoff_utc"]
            done = ko and (datetime.now(timezone.utc) - datetime.fromisoformat(ko.replace("Z", "+00:00"))).total_seconds() > 4 * 3600
            actual = a.get(r["stat_type"]) if done else None
        o = outcome(actual, float(r["prop_line"]), side, price)
        roi, is_win = (o if o else (None, None))
        picks.append(dict(week=r["week"], rule=rule, player=r["player_name"], pos=r["position"],
                          team=r["team"], opp=r["opponent"], stat=r["stat_type"], line=r["prop_line"],
                          side=side, model_p=mp, mkt_p=mk, edge=edge, price=price,
                          ev=mp * (price - 1) - (1 - mp), actual=actual, is_win=is_win, roi=roi,
                          game_id=r["game_id"], kickoff=r["kickoff_utc"],
                          stored_graded=bool(r["graded_at"])))
    return picks, fd_missing


def cmd_history(args):
    weeks = list(range(2, args.week + 1))
    picks, fd_missing = build_picks(args.season, weeks)
    print(f"FW-2 rules — per-week (FanDuel price, game-clustered 95% CI; pre-injury-filter; uncapped)\n")
    for rule in ("FW-2c", "FW-2d", "FW-2e"):
        print(f"== {rule}")
        print(f"{'wk':>3} {'N':>4} {'W':>4} {'WR%':>6} {'BE%':>6} {'ROI%':>7}  {'95% CI':>17}  games  fd_miss")
        for wk in weeks:
            s = summarize([p for p in picks if p["rule"] == rule and p["week"] == wk])
            if s["n"] == 0 and not fd_missing[(wk, rule)]:
                continue
            ci = s["ci"]
            print(f"{wk:>3} {s['n']:>4} {s['w']:>4} {s['wr']:>6.1f} {s['be']:>6.1f} {s['roi']:>+7.1f}  "
                  f"[{ci[0]:+6.1f}, {ci[1]:+6.1f}]  {s.get('games', 0):>5}  {fd_missing[(wk, rule)]:>7}")
        for label, ws in (("W2-W9 (in-sample backtest)", range(2, 10)),
                          (f"W10-W{args.week} (forward)", range(10, args.week + 1)),
                          (f"W2-W{args.week} season", range(2, args.week + 1))):
            s = summarize([p for p in picks if p["rule"] == rule and p["week"] in ws])
            ci = s["ci"]
            print(f"  {label:<28} N={s['n']:<4} WR={s['wr']:.1f}% BE={s['be']:.1f}% ROI={s['roi']:+.1f}% "
                  f"[{ci[0]:+.1f}, {ci[1]:+.1f}]")
        print()
    if args.detail_week:
        print(f"== Week {args.detail_week} pick detail")
        for p in sorted([p for p in picks if p["week"] == args.detail_week], key=lambda p: (p["rule"], -p["edge"])):
            res = "PEND" if p["is_win"] is None and p["roi"] is None else ("PUSH" if p["is_win"] is None else ("W" if p["is_win"] else "L"))
            print(f"  {p['rule']} {p['player']:<22} {p['pos'] or '':<3} {p['stat']:<10} {p['side']:<5} {p['line']:>6} "
                  f"@{p['price']:.2f} edge={p['edge']:+5.1f}pp act={p['actual']} {res}")


def cmd_slate(args):
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = _ro()
    preds = [r for r in load_preds(conn, args.season, [args.week]) if (r["kickoff_utc"] or "") > now_iso]
    fd = load_fd_prices(conn, sorted({(r["kickoff_utc"] or "")[:10] for r in preds} | {"2026-11-22", "2026-11-23", "2026-11-24"}))
    conn.close()
    pdb = _sleeper_player_db()
    inj = {}
    for v in pdb.values():
        if v.get("full_name") and v.get("injury_status"):
            inj[_normalize_player_name(v["full_name"])] = v["injury_status"]
    rows, miss = [], 0
    for r in preds:
        c = classify(r, fd, now_iso)
        if not c:
            continue
        rule, side, mp, mk, edge, price = c
        if price is None:
            miss += 1
            continue
        rows.append((rule, r, side, mp, mk, edge, price, inj.get(_normalize_player_name(r["player_name"]), "")))
    print(f"Upcoming W{args.week} qualifying picks (FanDuel price; generated {now_iso}); "
          f"{len(preds)} future main-line preds; fd_missing={miss}\n")
    for rule in ("FW-2c", "FW-2d", "FW-2e"):
        sel = sorted([x for x in rows if x[0] == rule], key=lambda x: -x[5])
        print(f"== {rule} ({len(sel)})")
        for _, r, side, mp, mk, edge, price, st in sel:
            ev = mp * (price - 1) - (1 - mp)
            flag = f" [{st}]" if st else ""
            print(f"  {r['kickoff_utc'][5:16]} {r['team'] or '':<4}v{r['opponent'] or '':<4} {r['player_name']:<22} "
                  f"{r['position'] or '':<3} {r['stat_type']:<10} {side.upper():<5} {r['prop_line']:>5} @{price:.2f}  "
                  f"model={mp:.3f} mkt={mk:.3f} edge={edge:+.1f}pp EV={ev:+.3f}{flag}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["history", "slate"])
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=11)
    ap.add_argument("--detail-week", type=int, default=None)
    ap.add_argument("--price", choices=["fd", "stored"], default="fd",
                    help="fd = FanDuel at same line (prereg); stored = row's latest-book price")
    a = ap.parse_args()
    PRICE_MODE = a.price
    cmd_history(a) if a.mode == "history" else cmd_slate(a)
