"""
attribution.py — Phase 4A: Performance attribution dashboard.

Breaks down your actual edge and leaks across every dimension:
  - League, sport, market type, odds bracket, time of day/week
  - Hot/cold streak detection and bet-after-streak signal
  - Multi-dimension cross-tab (e.g. NFL x moneyline x 6-15x odds)
  - Leak finder: ranked list of your biggest ROI drains
  - Edge finder: ranked list of your strongest signals
"""
from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from database import Bet


# ── Helpers ───────────────────────────────────────────────────────────────────

def _odds_bracket(odds: float) -> str:
    if odds < 2:    return "favorite (<2x)"
    if odds < 3:    return "short (2-3x)"
    if odds < 6:    return "medium (3-6x)"
    if odds < 15:   return "long (6-15x)"
    if odds < 50:   return "longshot (15-50x)"
    return "moonshot (50x+)"


def _legs_bucket(legs: int) -> str:
    if legs <= 2:   return "1-2"
    if legs <= 4:   return "3-4"
    if legs <= 6:   return "5-6"
    if legs <= 9:   return "7-9"
    return "10+"


def _market_type(bet_info: str) -> str:
    info = (bet_info or "").lower()
    ml_c   = info.count("moneyline")
    spr_c  = sum(1 for kw in ["run line", "spread"] if kw in info)
    tot_c  = sum(1 for kw in ["over", "under", "total"] if kw in info)
    prop_c = sum(1 for kw in ["to score", "player", "shots on target",
                               "rebounds", "strikeouts"] if kw in info)
    counts = {"Moneyline": ml_c, "Spread": spr_c, "Total": tot_c, "Prop": prop_c}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else "Moneyline"


def _settled(db: Session, include_mock: bool = True) -> list[Bet]:
    q = db.query(Bet).filter(Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]))
    if not include_mock:
        q = q.filter(Bet.is_mock == False)
    return q.order_by(Bet.time_placed).all()


def _row(bets: list[Bet]) -> dict:
    if not bets:
        return {"n": 0, "win_rate": 0, "roi": 0, "profit": 0, "wagered": 0}
    wins    = sum(1 for b in bets if b.status == "SETTLED_WIN")
    profit  = sum((b.profit or 0) for b in bets)
    wagered = sum((b.amount or 0) for b in bets)
    return {
        "n":        len(bets),
        "wins":     wins,
        "win_rate": round(wins / len(bets) * 100, 2),
        "roi":      round(profit / wagered * 100, 2) if wagered else 0,
        "profit":   round(profit, 2),
        "wagered":  round(wagered, 2),
    }


# ── Core attribution dimensions ───────────────────────────────────────────────

def by_league(db: Session, min_bets: int = 5, include_mock: bool = False) -> list[dict]:
    bets  = _settled(db, include_mock)
    groups: dict[str, list] = {}
    for b in bets:
        lg = (b.leagues or "other").split("|")[0].strip()
        groups.setdefault(lg, []).append(b)
    rows = [{**_row(v), "league": k} for k, v in groups.items() if len(v) >= min_bets]
    return sorted(rows, key=lambda x: -x["roi"])


def by_sport(db: Session, min_bets: int = 5, include_mock: bool = False) -> list[dict]:
    bets  = _settled(db, include_mock)
    groups: dict[str, list] = {}
    for b in bets:
        sp = (b.sports or "other").split("|")[0].strip()
        groups.setdefault(sp, []).append(b)
    rows = [{**_row(v), "sport": k} for k, v in groups.items() if len(v) >= min_bets]
    return sorted(rows, key=lambda x: -x["roi"])


def by_market_type(db: Session, min_bets: int = 5, include_mock: bool = False) -> list[dict]:
    bets  = _settled(db, include_mock)
    groups: dict[str, list] = {}
    for b in bets:
        mt = _market_type(b.bet_info)
        groups.setdefault(mt, []).append(b)
    rows = [{**_row(v), "market_type": k} for k, v in groups.items() if len(v) >= min_bets]
    return sorted(rows, key=lambda x: -x["roi"])


def by_odds_bracket(db: Session, min_bets: int = 5, include_mock: bool = False) -> list[dict]:
    bets  = _settled(db, include_mock)
    order = ["favorite (<2x)", "short (2-3x)", "medium (3-6x)",
             "long (6-15x)", "longshot (15-50x)", "moonshot (50x+)"]
    groups: dict[str, list] = {}
    for b in bets:
        if b.odds:
            br = _odds_bracket(b.odds)
            groups.setdefault(br, []).append(b)
    rows = [{**_row(v), "bracket": k, "sort_order": order.index(k) if k in order else 99}
            for k, v in groups.items() if len(v) >= min_bets]
    return sorted(rows, key=lambda x: x["sort_order"])


def by_legs(db: Session, include_mock: bool = False) -> list[dict]:
    bets  = _settled(db, include_mock)
    order = ["1-2", "3-4", "5-6", "7-9", "10+"]
    groups: dict[str, list] = {}
    for b in bets:
        lb = _legs_bucket(b.legs or 1)
        groups.setdefault(lb, []).append(b)
    rows = [{**_row(v), "legs_bucket": k, "sort_order": order.index(k)}
            for k, v in groups.items()]
    return sorted(rows, key=lambda x: x["sort_order"])


def by_hour(db: Session, min_bets: int = 8, include_mock: bool = False) -> list[dict]:
    bets  = _settled(db, include_mock)
    groups: dict[int, list] = {}
    for b in bets:
        if b.time_placed:
            h = b.time_placed.hour
            groups.setdefault(h, []).append(b)
    rows = [{**_row(v), "hour": k, "hour_label": f"{k:02d}:00"}
            for k, v in groups.items() if len(v) >= min_bets]
    return sorted(rows, key=lambda x: x["hour"])


def by_day_of_week(db: Session, include_mock: bool = False) -> list[dict]:
    bets  = _settled(db, include_mock)
    days  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    groups: dict[int, list] = {}
    for b in bets:
        if b.time_placed:
            groups.setdefault(b.time_placed.weekday(), []).append(b)
    rows = [{**_row(v), "day": days[k], "day_num": k}
            for k, v in groups.items()]
    return sorted(rows, key=lambda x: x["day_num"])


def by_stake_size(db: Session, include_mock: bool = False) -> list[dict]:
    bets   = _settled(db, include_mock)
    labels = ["$1-3", "$4-7", "$8-12", "$13-20", "$20+"]
    def bucket(amt):
        if amt <= 3:    return "$1-3"
        if amt <= 7:    return "$4-7"
        if amt <= 12:   return "$8-12"
        if amt <= 20:   return "$13-20"
        return "$20+"
    groups: dict[str, list] = {}
    for b in bets:
        groups.setdefault(bucket(b.amount or 0), []).append(b)
    rows = [{**_row(v), "stake_bucket": k, "sort_order": labels.index(k) if k in labels else 99}
            for k, v in groups.items()]
    return sorted(rows, key=lambda x: x["sort_order"])


# ── Streak detection ──────────────────────────────────────────────────────────

def streak_analysis(db: Session, include_mock: bool = False) -> dict:
    """
    Detect current win/loss streak and compute
    historical performance after N-length streaks.
    """
    bets = _settled(db, include_mock)
    if not bets:
        return {}

    results = [b.status == "SETTLED_WIN" for b in bets]

    # Current streak
    current_streak = 1
    current_type   = "win" if results[-1] else "loss"
    for r in reversed(results[:-1]):
        if r == results[-1]:
            current_streak += 1
        else:
            break

    # After-streak win rates
    after_win_results  = []
    after_loss_results = []
    after_2win         = []
    after_2loss        = []
    after_3win         = []
    after_3loss        = []

    for i in range(1, len(results)):
        curr = results[i]
        prev = results[i - 1]
        if prev:
            after_win_results.append(curr)
        else:
            after_loss_results.append(curr)
        if i >= 2:
            if results[i-1] and results[i-2]:
                after_2win.append(curr)
            elif not results[i-1] and not results[i-2]:
                after_2loss.append(curr)
        if i >= 3:
            if all(results[i-3:i]):
                after_3win.append(curr)
            elif not any(results[i-3:i]):
                after_3loss.append(curr)

    def pct(lst): return round(sum(lst) / len(lst) * 100, 1) if lst else None

    # Hot/cold signal
    overall_wr = sum(results) / len(results)
    after_win_wr = pct(after_win_results) or 0
    after_loss_wr = pct(after_loss_results) or 0

    if current_type == "win" and current_streak >= 2:
        signal = "HOT" if after_win_wr > overall_wr * 100 else "NEUTRAL"
    elif current_type == "loss" and current_streak >= 2:
        signal = "COLD" if after_loss_wr < overall_wr * 100 else "NEUTRAL"
    else:
        signal = "NEUTRAL"

    return {
        "current_streak":    current_streak,
        "current_type":      current_type,
        "signal":            signal,
        "overall_win_rate":  round(overall_wr * 100, 1),
        "after_1win_wr":     pct(after_win_results),
        "after_1loss_wr":    pct(after_loss_results),
        "after_2win_wr":     pct(after_2win),
        "after_2loss_wr":    pct(after_2loss),
        "after_3win_wr":     pct(after_3win),
        "after_3loss_wr":    pct(after_3loss),
        "recommendation":    _streak_rec(current_type, current_streak, signal, after_win_wr, after_loss_wr),
    }


def _streak_rec(ctype, clen, signal, aw, al) -> str:
    if signal == "HOT":
        return f"On a {clen}-bet win streak. Historical win rate after similar streaks: {aw:.1f}% — slightly above average. Maintain current approach."
    if signal == "COLD":
        return f"On a {clen}-bet losing streak. Historical win rate after similar streaks: {al:.1f}% — consider reducing stake size or pausing."
    if ctype == "win":
        return f"Coming off {clen} win(s). Historical win rate after a win: {aw:.1f}%."
    return f"Coming off {clen} loss(es). Historical win rate after a loss: {al:.1f}%."


# ── Leak and edge finder ──────────────────────────────────────────────────────

def find_leaks_and_edges(db: Session, include_mock: bool = False) -> dict:
    """
    Rank your worst ROI patterns (leaks) and best ROI patterns (edges).
    Cross-tabs sport x market type, sport x legs, league x odds bracket.
    """
    bets = _settled(db, include_mock)
    if not bets:
        return {"leaks": [], "edges": []}

    cross: dict[str, list] = {}
    for b in bets:
        sport  = (b.sports   or "other").split("|")[0].strip()
        league = (b.leagues  or "other").split("|")[0].strip()
        mkt    = _market_type(b.bet_info)
        lb     = _legs_bucket(b.legs or 1)
        br     = _odds_bracket(b.odds or 2)

        keys = [
            f"{sport} | {mkt}",
            f"{sport} | {lb} legs",
            f"{league} | {br}",
            f"{mkt} | {lb} legs",
        ]
        for k in keys:
            cross.setdefault(k, []).append(b)

    rows = []
    for combo, group in cross.items():
        if len(group) < 8:
            continue
        r = _row(group)
        rows.append({**r, "combo": combo})

    rows.sort(key=lambda x: x["roi"])
    leaks  = rows[:5]
    edges  = list(reversed(rows[-5:]))

    return {
        "leaks":  leaks,
        "edges":  edges,
        "total_combos_analyzed": len(rows),
    }


# ── Full attribution report ────────────────────────────────────────────────────

def full_attribution(db: Session, include_mock: bool = False) -> dict:
    """Single call that returns all attribution dimensions + streak + leaks."""
    return {
        "by_league":      by_league(db, include_mock=include_mock),
        "by_sport":       by_sport(db, include_mock=include_mock),
        "by_market_type": by_market_type(db, include_mock=include_mock),
        "by_odds_bracket":by_odds_bracket(db, include_mock=include_mock),
        "by_legs":        by_legs(db, include_mock=include_mock),
        "by_hour":        by_hour(db, include_mock=include_mock),
        "by_day_of_week": by_day_of_week(db, include_mock=include_mock),
        "by_stake_size":  by_stake_size(db, include_mock=include_mock),
        "streaks":        streak_analysis(db, include_mock=include_mock),
        "leaks_edges":    find_leaks_and_edges(db, include_mock=include_mock),
    }
