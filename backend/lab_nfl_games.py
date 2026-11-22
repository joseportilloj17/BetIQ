"""
lab_nfl_games.py — NFL Lab "Games & Edges": moneyline / spread / total per game.

READ-ONLY.  Source: bets.db:fixtures.bookmakers (TheOddsAPI snapshot, decimal odds).

There is NO validated NFL side/total prediction model (see memory: CLV verdict,
ATS model work is MLB-only; nfl_game_sim archived).  "Edge" here is therefore
market-structure only:

  * fair_prob   — median across books of per-book multiplicative-devig probability
                  at the consensus point (most common point across books).
  * shop_ev     — fair_prob × best_price − 1  (line-shopping EV vs no-vig consensus).
  * Wong teaser — validated structural edge: DOG at +1.5..+2.5 (blind 80.0% per leg,
                  N=135).  FAV at −8.5..−7.5 is tracked only (reverts to ~BE blind).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_BETS_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bets.db"))
_FD = "fanduel"


def _ro():
    c = sqlite3.connect(f"file:{_BETS_DB}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _parse_ts(s):
    if not s:
        return None
    s = str(s).replace("Z", "").replace("T", " ")
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("[GamesEdges] unparseable commence_time %r", s)
        return None


def _wong(spread):
    if spread is None:
        return None
    if 1.5 <= spread <= 2.5:
        return {"type": "dog", "validated": True,
                "note": "Wong dog teaser leg (+6 → crosses 3 & 7). Blind 80.0%/leg, N=135."}
    if -8.5 <= spread <= -7.5:
        return {"type": "fav", "validated": False,
                "note": "Wong fav — tracking only (blind 60.4%, reverts to ~BE)."}
    return None


def _market_summary(books, mkey, sides):
    """sides: list of outcome names (two).  Returns per-side summary dict."""
    # collect per-book (point, price) by side
    per_book = {}
    for b in books:
        for m in b.get("markets", []):
            if m.get("key") != mkey:
                continue
            o = {x.get("name"): x for x in m.get("outcomes", [])}
            if not all(s in o for s in sides):
                continue
            per_book[b.get("key")] = {s: (o[s].get("point"), o[s].get("price")) for s in sides}
    if not per_book:
        return None
    # consensus point = most common point of side[0]
    pts = Counter(v[sides[0]][0] for v in per_book.values())
    cons_pt = pts.most_common(1)[0][0]
    fair = {s: [] for s in sides}
    for v in per_book.values():
        if v[sides[0]][0] != cons_pt:
            continue
        pr = [v[s][1] for s in sides]
        if any(p is None or p <= 1 for p in pr):
            continue
        inv = [1 / p for p in pr]
        tot = sum(inv)
        for s, i in zip(sides, inv):
            fair[s].append(i / tot)
    out = {"consensus_point": cons_pt, "n_books": len(per_book),
           "n_books_at_consensus": len(fair[sides[0]]), "sides": {}}
    for s in sides:
        fp = statistics.median(fair[s]) if fair[s] else None
        at_cons = [(bk, v[s][1]) for bk, v in per_book.items() if v[sides[0]][0] == cons_pt and v[s][1]]
        best_bk, best_p = max(at_cons, key=lambda x: x[1]) if at_cons else (None, None)
        fd = per_book.get(_FD, {}).get(s)
        all_pts = sorted({v[s][0] for v in per_book.values() if v[s][0] is not None})
        side_pt = next((v[s][0] for v in per_book.values() if v[sides[0]][0] == cons_pt), None)
        out["sides"][s] = {
            "point": side_pt,
            "fair_prob": round(fp, 4) if fp else None,
            "fair_price": round(1 / fp, 3) if fp else None,
            "best_price": best_p, "best_book": best_bk,
            "shop_ev": round(fp * best_p - 1, 4) if (fp and best_p) else None,
            "fd_point": fd[0] if fd else None, "fd_price": fd[1] if fd else None,
            "fd_ev": round(fp * fd[1] - 1, 4) if (fp and fd and fd[1] and fd[0] == side_pt) else None,
            "point_range": [all_pts[0], all_pts[-1]] if all_pts else None,
        }
    return out


_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _backtest_status():
    import nfl_game_model as ngm
    try:
        with open(ngm.BACKTEST_PATH) as f:
            bt = json.load(f)
    except FileNotFoundError:
        return None
    return {"run_at": bt.get("run_at"), "n_games": bt.get("n_games"),
            "passed": {m: bool(v.get("passed")) for m, v in bt.get("markets", {}).items()},
            "markets": bt.get("markets"), "calibration": bt.get("calibration")}


def _attach_model(game, pred, status):
    """Add model prediction + gap per market side. take=True only if that market's backtest PASSED."""
    import numpy as np
    import nfl_game_model as ngm
    passed = (status or {}).get("passed", {})
    m, t = pred["dist"]["margin"], pred["dist"]["total"]
    home, away = game["home"], game["away"]
    ml = game.get("moneyline")
    if ml:
        ph = pred["p_home_win"]
        for side, p in ((home, ph), (away, 1 - ph)):
            s = ml["sides"][side]
            s["model_p"] = round(p, 4)
            s["gap"] = round((p - s["fair_prob"]) * 100, 1) if s.get("fair_prob") else None
            s["gap_unit"] = "pp"
            s["model_ev"] = round(p * s["best_price"] - 1, 3) if s.get("best_price") else None
            s["take"] = bool(passed.get("moneyline") and s["gap"] and s["gap"] > 0 and (s["model_ev"] or 0) > 0)
    sp = game.get("spread")
    if sp:
        for side, sign in ((home, 1.0), (away, -1.0)):
            s = sp["sides"][side]
            pt = s.get("point")
            if pt is None:
                continue
            cov = sign * m + pt           # side's margin + its spread
            pc = float(np.mean(cov > 0) / max(1e-9, 1 - np.mean(cov == 0)))
            s["model_line"] = round(-sign * float(np.median(m)) * 2) / 2   # model's spread for this side
            s["model_p"] = round(pc, 4)
            s["gap"] = round(sign * float(np.median(m)) + pt, 1)          # +ve = model likes this side vs line (pts)
            s["gap_unit"] = "pts"
            s["model_ev"] = round(pc * s["best_price"] - 1, 3) if s.get("best_price") else None
            s["take"] = bool(passed.get("spread") and s["gap"] > 0 and (s["model_ev"] or 0) > 0)
    tot = game.get("total")
    if tot:
        for side in ("Over", "Under"):
            s = tot["sides"][side]
            ln = s.get("point")
            if ln is None:
                continue
            po = float(np.mean(t > ln) / max(1e-9, 1 - np.mean(t == ln)))
            p = po if side == "Over" else 1 - po
            s["model_line"] = round(float(np.median(t)) * 2) / 2
            s["model_p"] = round(p, 4)
            g = float(np.median(t)) - ln
            s["gap"] = round(g if side == "Over" else -g, 1)
            s["gap_unit"] = "pts"
            s["model_ev"] = round(p * s["best_price"] - 1, 3) if s.get("best_price") else None
            s["take"] = bool(passed.get("total") and s["gap"] > 0 and (s["model_ev"] or 0) > 0)
    game["model"] = {"p_home_win": round(pred["p_home_win"], 4), "median_margin": round(pred["median_margin"], 1),
                     "median_total": round(pred["median_total"], 1),
                     "proj_home": round((pred["median_total"] + pred["median_margin"]) / 2, 1),
                     "proj_away": round((pred["median_total"] - pred["median_margin"]) / 2, 1)}


def build_nfl_games(days_ahead: int = 8, with_model: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    conn = _ro()
    try:
        rows = conn.execute(
            "SELECT id, home_team, away_team, commence_time, bookmakers, fetched_at "
            "FROM fixtures WHERE sport_key='americanfootball_nfl' ORDER BY commence_time").fetchall()
    finally:
        conn.close()
    games, skipped = [], 0
    for r in rows:
        ko = _parse_ts(r["commence_time"])
        if ko is None or ko < now - timedelta(hours=4) or ko > now + timedelta(days=days_ahead):
            continue
        try:
            books = json.loads(r["bookmakers"]) if isinstance(r["bookmakers"], str) else (r["bookmakers"] or [])
        except Exception as exc:
            logger.warning("[GamesEdges] bad bookmakers JSON fixture=%s: %s", r["id"], exc)
            skipped += 1
            continue
        home, away = r["home_team"], r["away_team"]
        ml = _market_summary(books, "h2h", [away, home])
        sp = _market_summary(books, "spreads", [away, home])
        tot = _market_summary(books, "totals", ["Over", "Under"])
        if sp:
            for side in (away, home):
                sp["sides"][side]["wong"] = _wong(sp["sides"][side]["point"])
        games.append({
            "fixture_id": r["id"], "home": home, "away": away,
            "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "started": ko <= now, "odds_fetched_at": str(r["fetched_at"]),
            "moneyline": ml, "spread": sp, "total": tot,
        })
    model_meta, status, n_model = None, None, 0
    if with_model and games:
        try:
            import nfl_game_model as ngm
            status = _backtest_status()
            preds = ngm.predict_upcoming(2026)
            model_meta = {k: v for k, v in preds.get("_meta", {}).items()}
            for g in games:
                key = (_ABBR.get(g["home"]), _ABBR.get(g["away"]))
                if key in preds:
                    _attach_model(g, preds[key], status)
                    n_model += 1
            if n_model < sum(1 for g in games if not g["started"] and
                             g["kickoff_utc"][:10] <= (now + timedelta(days=4)).strftime("%Y-%m-%d")):
                logger.warning("[GamesEdges] model attached to only %d/%d games", n_model, len(games))
        except Exception as exc:
            logger.warning("[GamesEdges] model prediction failed: %s", exc, exc_info=True)
            model_meta = {"error": str(exc)}
    return {
        "sport": "NFL", "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_games": len(games), "skipped_bad_json": skipped, "games": games,
        "model_meta": model_meta, "n_with_model": n_model, "backtest": status,
        "method": "fair = median per-book multiplicative devig at consensus point; "
                  "shop_ev = fair × best − 1. No validated NFL side/total model — "
                  "edges are line-shopping + Wong teaser structure only.",
    }
