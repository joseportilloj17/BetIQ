"""
kelly.py — Phase 4B: Kelly Criterion stake advisor + bankroll tracker.

Kelly formula: f* = (b*p - q) / b
  b = decimal_odds - 1 (net odds)
  p = win probability
  q = 1 - p (loss probability)

Uses fractional Kelly (25% by default) for safety.
Bankroll stored in DB as a simple key-value via SchedulerRun notes field,
or tracked client-side. Exposed via API.
"""
from __future__ import annotations
import math
import json
import os
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from database import Bet

BANKROLL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bankroll.json")
DEFAULT_BANKROLL = 500.0
KELLY_FRACTION   = 0.25   # 25% fractional Kelly — industry standard for safety


# ── Kelly math ────────────────────────────────────────────────────────────────

def kelly_stake(
    win_prob:     float,
    decimal_odds: float,
    bankroll:     float,
    fraction:     float = KELLY_FRACTION,
    min_stake:    float = 1.0,
    max_stake_pct:float = 0.10,   # never bet more than 10% of bankroll
) -> dict:
    """
    Compute fractional Kelly stake for a single bet.
    Returns recommended stake in dollars + supporting metrics.
    """
    if decimal_odds <= 1 or win_prob <= 0 or win_prob >= 1:
        return {"stake": 0, "kelly_pct": 0, "rationale": "Invalid odds or probability."}

    b = decimal_odds - 1
    q = 1 - win_prob
    raw_kelly = (b * win_prob - q) / b       # full Kelly fraction

    if raw_kelly <= 0:
        return {
            "stake":       0,
            "kelly_pct":   round(raw_kelly * 100, 2),
            "kelly_frac":  0,
            "rationale":   "Negative edge — Kelly advises no bet.",
            "ev_per_unit": round(win_prob * b - q, 4),
        }

    frac_kelly = raw_kelly * fraction
    max_kelly  = max_stake_pct                # hard cap
    used_kelly = min(frac_kelly, max_kelly)

    stake = round(bankroll * used_kelly, 2)
    stake = max(stake, min_stake)

    ev_per_dollar = win_prob * b - q
    ev_dollar     = round(ev_per_dollar * stake, 2)

    cap_note = " (capped at 10% bankroll limit)" if frac_kelly > max_kelly else ""

    return {
        "stake":          stake,
        "kelly_pct":      round(used_kelly * 100, 2),
        "full_kelly_pct": round(raw_kelly * 100, 2),
        "frac_kelly_pct": round(frac_kelly * 100, 2),
        "ev_per_dollar":  round(ev_per_dollar, 4),
        "ev_total":       ev_dollar,
        "bankroll":       bankroll,
        "rationale":      f"Fractional Kelly ({int(fraction*100)}%){cap_note}: "
                          f"bet ${stake} of ${bankroll:.0f} bankroll.",
    }


def kelly_parlay(
    legs: list[dict],           # each dict has win_prob (0-1) and odds (decimal)
    bankroll: float,
    fraction: float = KELLY_FRACTION,
) -> dict:
    """
    Kelly for a parlay: use combined win_prob and combined decimal odds.
    Also shows individual leg Kelly for comparison.
    """
    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= leg.get("odds", 1.0)
        combined_prob *= leg.get("win_prob", 0.5) / 100  # API returns 0-100

    main = kelly_stake(combined_prob, combined_odds, bankroll, fraction)

    leg_kellys = []
    for leg in legs:
        lk = kelly_stake(
            leg.get("win_prob", 50) / 100,
            leg.get("odds", 2.0),
            bankroll, fraction
        )
        leg_kellys.append({
            "description": leg.get("description", ""),
            "odds":        leg.get("odds"),
            "win_prob":    leg.get("win_prob"),
            **lk,
        })

    return {
        "parlay":           main,
        "combined_odds":    round(combined_odds, 3),
        "combined_win_prob":round(combined_prob * 100, 3),
        "leg_kellys":       leg_kellys,
        "note": "Parlay Kelly uses combined probability. Individual leg Kelly shown for reference.",
    }


# ── Bankroll management ────────────────────────────────────────────────────────

def load_bankroll() -> dict:
    """Load bankroll state from disk."""
    if not os.path.exists(BANKROLL_PATH):
        return {
            "starting_bankroll": DEFAULT_BANKROLL,
            "current_bankroll":  DEFAULT_BANKROLL,
            "peak_bankroll":     DEFAULT_BANKROLL,
            "updated_at":        datetime.utcnow().isoformat(),
            "kelly_fraction":    KELLY_FRACTION,
            "transactions":      [],
        }
    with open(BANKROLL_PATH) as f:
        return json.load(f)


def save_bankroll(state: dict):
    state["updated_at"] = datetime.utcnow().isoformat()
    with open(BANKROLL_PATH, "w") as f:
        json.dump(state, f, indent=2)


def set_bankroll(amount: float, kelly_fraction: float = KELLY_FRACTION) -> dict:
    """Set/reset the bankroll."""
    state = load_bankroll()
    state["starting_bankroll"] = amount
    state["current_bankroll"]  = amount
    state["peak_bankroll"]     = max(amount, state.get("peak_bankroll", 0))
    state["kelly_fraction"]    = kelly_fraction
    state["transactions"].append({
        "type": "set", "amount": amount, "at": datetime.utcnow().isoformat()
    })
    save_bankroll(state)
    return state


def update_bankroll_from_db(db: Session) -> dict:
    """
    Sync bankroll with actual settled bets from DB (app-placed only, not Pikkit imports).
    """
    state = load_bankroll()
    starting = state["starting_bankroll"]

    # Sum all app-placed (real only, not mock) settled bets
    real_settled = db.query(Bet).filter(
        Bet.source == "app",
        Bet.is_mock == False,
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS", "SETTLED_PUSH"]),
    ).all()

    total_profit = sum((b.profit or 0) for b in real_settled)
    current = round(starting + total_profit, 2)
    peak    = max(current, state.get("peak_bankroll", starting))

    state["current_bankroll"] = current
    state["peak_bankroll"]    = peak
    state["real_bets_tracked"]= len(real_settled)
    save_bankroll(state)
    return state


def bankroll_stats(db: Session) -> dict:
    """Full bankroll health report."""
    state  = update_bankroll_from_db(db)
    curr   = state["current_bankroll"]
    start  = state["starting_bankroll"]
    peak   = state["peak_bankroll"]
    drawdown = round((peak - curr) / peak * 100, 2) if peak > 0 else 0
    growth   = round((curr - start) / start * 100, 2) if start > 0 else 0

    # Sizing guidelines at current bankroll
    guidelines = []
    for pct, label in [(0.02, "Conservative (2%)"), (0.05, "Moderate (5%)"),
                       (0.10, "Aggressive (10%)")]:
        guidelines.append({"label": label, "stake": round(curr * pct, 2)})

    return {
        "starting_bankroll":  start,
        "current_bankroll":   curr,
        "peak_bankroll":      peak,
        "total_growth_pct":   growth,
        "drawdown_pct":       drawdown,
        "kelly_fraction":     state.get("kelly_fraction", KELLY_FRACTION),
        "sizing_guidelines":  guidelines,
        "real_bets_tracked":  state.get("real_bets_tracked", 0),
        "updated_at":         state.get("updated_at"),
    }


# ── Paper trading ledger ──────────────────────────────────────────────────────

def paper_ledger(db: Session) -> dict:
    """
    Virtual P&L for mock bets — deferred from Phase 3, delivered here.
    Shows what the model's recommendations would have earned paper trading.
    """
    mock_bets = db.query(Bet).filter(
        Bet.is_mock == True,
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS", "SETTLED_PUSH"]),
    ).order_by(Bet.time_placed).all()

    if not mock_bets:
        return {
            "total_bets":   0,
            "message":      "No settled mock bets yet. Place mock bets from Today's Picks to start tracking.",
        }

    wins    = sum(1 for b in mock_bets if b.status == "SETTLED_WIN")
    total   = len(mock_bets)
    wagered = sum((b.amount or 0) for b in mock_bets)
    profit  = sum((b.profit or 0) for b in mock_bets)

    # Running virtual bankroll (start at $1000 virtual)
    virtual_start = 1000.0
    running = virtual_start
    timeline = []
    for b in mock_bets:
        running += (b.profit or 0)
        timeline.append({
            "date":    b.time_placed.strftime("%Y-%m-%d") if b.time_placed else "—",
            "profit":  round(b.profit or 0, 2),
            "running": round(running, 2),
            "status":  b.status,
            "legs":    b.legs,
            "odds":    b.odds,
        })

    return {
        "total_bets":       total,
        "wins":             wins,
        "losses":           total - wins,
        "win_rate":         round(wins / total * 100, 2),
        "total_wagered":    round(wagered, 2),
        "total_profit":     round(profit, 2),
        "roi_pct":          round(profit / wagered * 100, 2) if wagered else 0,
        "virtual_start":    virtual_start,
        "virtual_current":  round(running, 2),
        "virtual_growth_pct": round((running - virtual_start) / virtual_start * 100, 2),
        "timeline":         timeline[-30:],   # last 30 for chart
    }


# ── Timing advisor ────────────────────────────────────────────────────────────

def timing_advice(db: Session) -> dict:
    """
    Based on historical data: best hours and days to place bets.
    Also surfaces line movement signal if CLV data exists.
    """
    from attribution import by_hour, by_day_of_week

    hours = by_hour(db, min_bets=8)
    days  = by_day_of_week(db)

    best_hours = sorted(hours, key=lambda x: -x["roi"])[:3]
    worst_hours= sorted(hours, key=lambda x:  x["roi"])[:3]
    best_days  = sorted(days,  key=lambda x: -x["roi"])[:3]
    worst_days = sorted(days,  key=lambda x:  x["roi"])[:3]

    # CLV signal
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
        Bet.closing_line.isnot(None),
    ).all()
    clv_signal = None
    if len(bets) >= 10:
        positive_clv = [b for b in bets if b.closing_line and b.odds and b.closing_line > b.odds]
        negative_clv = [b for b in bets if b.closing_line and b.odds and b.closing_line < b.odds]
        if positive_clv and negative_clv:
            pos_wr = sum(1 for b in positive_clv if b.status=="SETTLED_WIN") / len(positive_clv)
            neg_wr = sum(1 for b in negative_clv if b.status=="SETTLED_WIN") / len(negative_clv)
            clv_signal = {
                "positive_clv_bets": len(positive_clv),
                "positive_clv_wr":   round(pos_wr * 100, 1),
                "negative_clv_bets": len(negative_clv),
                "negative_clv_wr":   round(neg_wr * 100, 1),
                "edge":              round((pos_wr - neg_wr) * 100, 1),
                "interpretation":    "Bets placed before line moved in your favour win at a higher rate — "
                                     "suggests line shopping has value." if pos_wr > neg_wr else
                                     "No clear CLV edge detected in current data.",
            }

    return {
        "best_hours":    best_hours,
        "worst_hours":   worst_hours,
        "best_days":     best_days,
        "worst_days":    worst_days,
        "clv_signal":    clv_signal,
        "recommendation": _timing_rec(best_hours, best_days),
    }


def _timing_rec(best_hours: list, best_days: list) -> str:
    if not best_hours or not best_days:
        return "Not enough data for timing recommendations yet."
    bh = best_hours[0]
    bd = best_days[0]
    return (f"Your best ROI is at {bh['hour_label']} on {bd['day']}s. "
            f"Consider scheduling your bet review around these windows.")
