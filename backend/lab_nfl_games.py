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


def build_nfl_games(days_ahead: int = 8) -> dict:
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
    return {
        "sport": "NFL", "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_games": len(games), "skipped_bad_json": skipped, "games": games,
        "method": "fair = median per-book multiplicative devig at consensus point; "
                  "shop_ev = fair × best − 1. No validated NFL side/total model — "
                  "edges are line-shopping + Wong teaser structure only.",
    }
