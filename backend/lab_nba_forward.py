"""
lab_nba_forward.py — NBA Lab data: the S5-4 frozen sim's stored forward predictions.

READ-ONLY over historical.db:nba_forward_preds.  Shows exactly what the S5-2
forward test will be graded on (sim P at every market-posted line), latest
snapshot per (player, stat, line, side).  No engine run (cheap, no drift from
the frozen config).

edge_pp = (sim P(side) − devig_implied) × 100, where devig_implied is the
market probability for that row's side (multiplicative for two-sided mains,
borrowed_margin for one-sided alts).
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
_HIST_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "historical.db"))


def _ro():
    c = sqlite3.connect(f"file:{_HIST_DB}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def build_nba_lab(game_date: str | None = None) -> dict:
    conn = _ro()
    try:
        if not game_date:
            today_ct = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%d")
            r = conn.execute("SELECT MIN(game_date) FROM nba_forward_preds WHERE game_date >= ? AND no_market=0",
                             (today_ct,)).fetchone()
            game_date = r[0] or today_ct
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM nba_forward_preds WHERE game_date=? AND data_error IS NULL", (game_date,))]
        dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT game_date FROM nba_forward_preds ORDER BY game_date DESC LIMIT 14")]
    finally:
        conn.close()

    latest = {}
    for r in rows:
        k = (r["player_name"], r["stat_type"], r["line"], r["side"])
        if k not in latest or (r["created_at"] or "") > (latest[k]["created_at"] or ""):
            latest[k] = r
    props, games, n_fallback, n_phantom = [], {}, 0, 0
    for r in latest.values():
        g = games.setdefault(r["game_id"], {
            "game_id": r["game_id"], "home": r["home_team"], "away": r["away_team"],
            "market_spread": r["market_spread"], "market_total": r["market_total"],
            "hours_to_tip": r["hours_to_tip"], "n_props": 0})
        g["n_props"] += 1
        if r["no_market"]:
            n_fallback += 1
            continue
        # Market never posts NBA alt UNDERs (518K alt props, all 'over'); stored
        # alt-under rows reuse the over price/devig → phantom.  OPEN_ITEMS OPS-8.
        if r["line_depth"] == "alt" and r["side"] == "under":
            n_phantom += 1
            continue
        p_side = r["sim_p_over"] if r["side"] == "over" else r["sim_p_under"]
        edge = (p_side - r["devig_implied"]) * 100 if (p_side is not None and r["devig_implied"]) else None
        ev = p_side * (r["best_price"] - 1) - (1 - p_side) if (p_side is not None and r["best_price"]) else None
        props.append({
            "player": r["player_name"], "team": r["team"], "game_id": r["game_id"],
            "stat": r["stat_type"], "line": r["line"], "side": r["side"], "depth": r["line_depth"],
            "sim_p": round(p_side, 4) if p_side is not None else None,
            "sim_median": r["sim_median"], "mkt_p": r["devig_implied"], "devig_method": r["devig_method"],
            "edge_pp": round(edge, 1) if edge is not None else None,
            "ev": round(ev, 3) if ev is not None else None,
            "best_price": r["best_price"], "best_book": r["best_book"], "n_books": r["n_books"],
            "injury": r["player_injury_status"], "snapshot": r["snapshot_type"], "created_at": r["created_at"],
            "actual": r["actual_value"], "hit": r["actual_hit"],
        })
    props.sort(key=lambda p: -(p["edge_pp"] or -999))
    no_anchor = [g for g in games.values() if not g["market_spread"] and not g["market_total"]]
    if no_anchor:
        logger.warning("[NBALab] %d/%d games on %s have no market spread/total anchor",
                       len(no_anchor), len(games), game_date)
    return {
        "sport": "NBA", "game_date": game_date, "available_dates": dates,
        "model_version": rows[0]["model_version"] if rows else None,
        "n_games": len(games), "n_props": len(props), "n_fallback_no_market": n_fallback,
        "n_excluded_phantom_alt_under": n_phantom,
        "n_graded": sum(1 for p in props if p["hit"] is not None),
        "games": sorted(games.values(), key=lambda g: g["hours_to_tip"] or 0),
        "games_missing_market_anchor": len(no_anchor),
        "props": props,
        "notes": "S5-4 frozen sim (tag s5-4-freeze-2026-11-15). Forward test UNCALIBRATED until "
                 "first Brier read at N≥100 games (~late-Dec). Latest snapshot per player/stat/line/side.",
    }
