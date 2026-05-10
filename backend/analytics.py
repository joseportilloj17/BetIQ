"""
analytics.py — Compute betting statistics, risk profile, and EV metrics.
All functions take a SQLAlchemy Session and return plain dicts (JSON-serializable).
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
import math

from sqlalchemy.orm import Session
from sqlalchemy import func, case
from database import Bet, BetLeg, Prediction


# ─── Core stats ───────────────────────────────────────────────────────────────

def get_summary_stats(db: Session) -> dict:
    """High-level dashboard numbers."""
    q = db.query(Bet).filter(Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]))
    total   = q.count()
    wins    = q.filter(Bet.status == "SETTLED_WIN").count()
    losses  = total - wins

    wagered = db.query(func.sum(Bet.amount)).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).scalar() or 0.0

    profit = db.query(func.sum(Bet.profit)).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).scalar() or 0.0

    avg_odds = db.query(func.avg(Bet.odds)).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).scalar() or 0.0

    avg_legs = db.query(func.avg(Bet.legs)).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).scalar() or 0.0

    avg_stake = db.query(func.avg(Bet.amount)).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).scalar() or 0.0

    parlay_count   = q.filter(Bet.bet_type == "parlay").count()
    straight_count = q.filter(Bet.bet_type == "straight").count()

    # Mock bets
    mock_total  = db.query(Bet).filter(Bet.is_mock == True).count()
    mock_wins   = db.query(Bet).filter(Bet.is_mock == True, Bet.status == "SETTLED_WIN").count()

    return {
        "total_bets":       total,
        "wins":             wins,
        "losses":           losses,
        "win_rate":         round(wins / total * 100, 2) if total else 0,
        "total_wagered":    round(float(wagered), 2),
        "total_profit":     round(float(profit), 2),
        "roi_pct":          round(float(profit) / float(wagered) * 100, 2) if wagered else 0,
        "avg_odds":         round(float(avg_odds), 3),
        "avg_legs":         round(float(avg_legs), 2),
        "avg_stake":        round(float(avg_stake), 2),
        "parlay_count":     parlay_count,
        "straight_count":   straight_count,
        "mock_total":       mock_total,
        "mock_wins":        mock_wins,
        "mock_win_rate":    round(mock_wins / mock_total * 100, 2) if mock_total else 0,
    }


def get_stats_by_legs(db: Session) -> list[dict]:
    """Win rate and ROI grouped by number of legs (buckets)."""
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()

    buckets: dict[str, dict] = {}
    for b in bets:
        if b.legs <= 2:    key = "1-2"
        elif b.legs <= 4:  key = "3-4"
        elif b.legs <= 6:  key = "5-6"
        elif b.legs <= 9:  key = "7-9"
        else:              key = "10+"

        if key not in buckets:
            buckets[key] = {"legs_bucket": key, "bets": 0, "wins": 0,
                            "wagered": 0.0, "profit": 0.0, "avg_odds": []}
        b_stats = buckets[key]
        b_stats["bets"]    += 1
        b_stats["wagered"] += b.amount or 0
        b_stats["profit"]  += b.profit or 0
        if b.status == "SETTLED_WIN":
            b_stats["wins"] += 1
        if b.odds:
            b_stats["avg_odds"].append(b.odds)

    result = []
    for key in ["1-2", "3-4", "5-6", "7-9", "10+"]:
        if key not in buckets:
            continue
        s = buckets[key]
        avg_o = sum(s["avg_odds"]) / len(s["avg_odds"]) if s["avg_odds"] else 0
        result.append({
            "legs_bucket":  key,
            "bets":         s["bets"],
            "wins":         s["wins"],
            "win_rate":     round(s["wins"] / s["bets"] * 100, 2) if s["bets"] else 0,
            "roi_pct":      round(s["profit"] / s["wagered"] * 100, 2) if s["wagered"] else 0,
            "avg_odds":     round(avg_o, 2),
        })
    return result


def get_stats_by_sport(db: Session) -> list[dict]:
    """Performance breakdown by primary sport."""
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()

    sports: dict[str, dict] = {}
    for b in bets:
        sport = (b.sports or "Unknown").split("|")[0].strip()
        if sport not in sports:
            sports[sport] = {"sport": sport, "bets": 0, "wins": 0,
                             "wagered": 0.0, "profit": 0.0}
        s = sports[sport]
        s["bets"]    += 1
        s["wagered"] += b.amount or 0
        s["profit"]  += b.profit or 0
        if b.status == "SETTLED_WIN":
            s["wins"] += 1

    result = []
    for s in sorted(sports.values(), key=lambda x: -x["bets"]):
        result.append({
            "sport":     s["sport"],
            "bets":      s["bets"],
            "wins":      s["wins"],
            "win_rate":  round(s["wins"] / s["bets"] * 100, 2) if s["bets"] else 0,
            "roi_pct":   round(s["profit"] / s["wagered"] * 100, 2) if s["wagered"] else 0,
            "wagered":   round(s["wagered"], 2),
            "profit":    round(s["profit"], 2),
        })
    return result


def get_stats_by_market(db: Session) -> list[dict]:
    """Win rate by market type (Moneyline, Spread, Total, Player Prop)."""
    legs = db.query(BetLeg).all()
    leg_to_bet: dict[str, Bet] = {}
    for b in db.query(Bet).filter(Bet.status.in_(["SETTLED_WIN","SETTLED_LOSS"])).all():
        leg_to_bet[b.id] = b

    markets: dict[str, dict] = {}
    for leg in legs:
        bet = leg_to_bet.get(leg.bet_id)
        if not bet:
            continue
        mtype = leg.market_type or "Other"
        if mtype not in markets:
            markets[mtype] = {"market_type": mtype, "legs": 0, "bet_wins": 0,
                              "bet_total": 0, "seen_bets": set()}
        m = markets[mtype]
        m["legs"] += 1
        if leg.bet_id not in m["seen_bets"]:
            m["seen_bets"].add(leg.bet_id)
            m["bet_total"] += 1
            if bet.status == "SETTLED_WIN":
                m["bet_wins"] += 1

    return [
        {
            "market_type": m["market_type"],
            "legs":        m["legs"],
            "bet_total":   m["bet_total"],
            "win_rate":    round(m["bet_wins"] / m["bet_total"] * 100, 2) if m["bet_total"] else 0,
        }
        for m in sorted(markets.values(), key=lambda x: -x["legs"])
        if "seen_bets" not in m or True  # always include
    ]


def get_monthly_pnl(db: Session) -> list[dict]:
    """Month-by-month P&L for charting."""
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
        Bet.time_placed.isnot(None)
    ).order_by(Bet.time_placed).all()

    months: dict[str, dict] = {}
    for b in bets:
        key = b.time_placed.strftime("%Y-%m")
        if key not in months:
            months[key] = {"month": key, "bets": 0, "wins": 0,
                           "wagered": 0.0, "profit": 0.0}
        m = months[key]
        m["bets"]    += 1
        m["wagered"] += b.amount or 0
        m["profit"]  += b.profit or 0
        if b.status == "SETTLED_WIN":
            m["wins"] += 1

    result = []
    running = 0.0
    for key in sorted(months.keys()):
        m = months[key]
        running += m["profit"]
        result.append({
            "month":      m["month"],
            "bets":       m["bets"],
            "wins":       m["wins"],
            "win_rate":   round(m["wins"] / m["bets"] * 100, 2) if m["bets"] else 0,
            "wagered":    round(m["wagered"], 2),
            "profit":     round(m["profit"], 2),
            "cumulative": round(running, 2),
        })
    return result


# ─── Risk profile ─────────────────────────────────────────────────────────────

def get_risk_profile(db: Session) -> dict:
    """
    Infer betting style from historical patterns.
    Returns a risk score 1-10 and descriptive profile.
    """
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()
    if not bets:
        return {"risk_score": 5, "profile": "Insufficient data"}

    avg_legs  = sum(b.legs for b in bets) / len(bets)
    avg_odds  = sum(b.odds for b in bets if b.odds) / len(bets)
    parlay_pct= sum(1 for b in bets if b.bet_type == "parlay") / len(bets)
    avg_stake = sum(b.amount for b in bets if b.amount) / len(bets)
    max_stake = max(b.amount for b in bets if b.amount)

    # Score components (each 0-10, weighted)
    legs_score   = min(avg_legs / 2, 10)          # more legs = more risk
    odds_score   = min(math.log(avg_odds + 1), 10)
    parlay_score = parlay_pct * 10
    stake_score  = min(avg_stake / 5, 10)

    composite = (
        legs_score   * 0.40 +
        odds_score   * 0.25 +
        parlay_score * 0.25 +
        stake_score  * 0.10
    )
    score = round(min(composite, 10), 1)

    if score < 3:      profile = "Conservative"
    elif score < 5:    profile = "Moderate"
    elif score < 7:    profile = "Aggressive"
    elif score < 8.5:  profile = "High-roller"
    else:              profile = "Variance-seeker"

    return {
        "risk_score":     score,
        "profile":        profile,
        "avg_legs":       round(avg_legs, 2),
        "avg_odds":       round(avg_odds, 2),
        "parlay_pct":     round(parlay_pct * 100, 1),
        "avg_stake":      round(avg_stake, 2),
        "max_stake":      round(max_stake, 2),
        "recommended_legs":   _recommend_legs(avg_legs, score),
        "recommended_stake":  _recommend_stake(avg_stake, score),
    }


def _recommend_legs(avg: float, risk_score: float) -> str:
    if risk_score > 7:
        return f"Your {avg:.1f} avg legs is very high. Consider capping at 5 legs — your win rate data shows steep drop-off past 6."
    elif risk_score > 5:
        return f"Your {avg:.1f} avg legs is aggressive. Mixing in 3-4 leg parlays could improve ROI."
    return f"Your {avg:.1f} avg legs is reasonable. Maintain or slightly reduce for better variance control."


def _recommend_stake(avg: float, risk_score: float) -> str:
    if risk_score > 7:
        return f"Average stake ${avg:.2f} with high-variance parlays creates large drawdown risk. Consider a flat ${min(avg, 5):.2f} unit."
    return f"Stake sizing of ${avg:.2f} average is within normal range."


# ─── Promo performance ────────────────────────────────────────────────────────

def get_promo_performance(db: Session) -> dict:
    """
    Group settled bets by promo_type and return per-promo performance stats.
    Only includes bets with status SETTLED_WIN or SETTLED_LOSS.
    Requires n >= 5 before surfacing ROI (else roi_pct = None).
    """
    from promo_engine import PROMO_TYPES

    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()

    buckets: dict[str, dict] = {k: {
        "promo_type": k,
        "label":       PROMO_TYPES[k]["label"],
        "count":       0,
        "wins":        0,
        "wagered":     0.0,
        "profit":      0.0,
        "ev_lift_sum": 0.0,
        "stake_sum":   0.0,
    } for k in PROMO_TYPES}

    for b in bets:
        ptype = (b.promo_type or "none")
        if ptype not in buckets:
            ptype = "none"
        bkt = buckets[ptype]
        bkt["count"]   += 1
        bkt["wagered"] += b.amount or 0
        bkt["profit"]  += b.profit or 0
        bkt["stake_sum"] += b.amount or 0
        if b.status == "SETTLED_WIN":
            bkt["wins"] += 1
        if b.promo_ev_lift:
            bkt["ev_lift_sum"] += b.promo_ev_lift

    rows = []
    best_roi = None
    best_promo_label = None

    for k in PROMO_TYPES:
        bkt = buckets[k]
        n   = bkt["count"]
        row: dict = {
            "promo_type":   k,
            "label":        bkt["label"],
            "count":        n,
            "win_rate":     round(bkt["wins"] / n * 100, 1) if n else None,
            "avg_ev_lift":  round(bkt["ev_lift_sum"] / n, 2) if n else None,
            "total_profit": round(bkt["profit"], 2),
            "roi_pct":      round(bkt["profit"] / bkt["wagered"] * 100, 1) if (n >= 5 and bkt["wagered"]) else None,
            "avg_stake":    round(bkt["stake_sum"] / n, 2) if n else None,
        }
        rows.append(row)
        if row["roi_pct"] is not None and (best_roi is None or row["roi_pct"] > best_roi):
            best_roi         = row["roi_pct"]
            best_promo_label = bkt["label"]

    best_banner = None
    if best_promo_label and best_promo_label != "No Promo":
        best_banner = f"Your {best_promo_label} bets have the highest ROI at {best_roi}%"

    total_promo_bets = sum(r["count"] for r in rows if r["promo_type"] != "none")

    return {
        "rows":              rows,
        "best_banner":       best_banner,
        "total_promo_bets":  total_promo_bets,
        "has_data":          total_promo_bets > 0,
    }


# ─── EV helpers ───────────────────────────────────────────────────────────────

def american_to_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def decimal_to_prob(decimal_odds: float) -> float:
    return 1 / decimal_odds


def compute_ev(win_prob: float, decimal_odds: float, stake: float = 1.0) -> float:
    """Expected value given win probability and decimal odds."""
    payout  = (decimal_odds - 1) * stake
    ev      = win_prob * payout - (1 - win_prob) * stake
    return round(ev, 4)


# ─── Per-leg win rate analysis ────────────────────────────────────────────────

def get_leg_win_rates(db: Session) -> dict:
    """
    Compute per-leg win rate from historical bet data using two methods:

    1. Naive: legs in winning bets / total legs
       (understates because a 9-leg parlay counts all 9 as losses if 1 fails)

    2. Geometric mean (implied): parlay_win_rate ^ (1 / avg_legs)
       This estimates the true per-leg win rate assuming independence.
       For a 7-leg parlay with 24.7% win rate: 0.247^(1/7) = 82.25%

    The geometric mean is the more accurate representation.
    """
    import math

    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()

    if not bets:
        return {}

    total_bets    = len(bets)
    total_wins    = sum(1 for b in bets if b.status == "SETTLED_WIN")
    total_legs    = sum(b.legs or 1 for b in bets)
    legs_in_wins  = sum(b.legs or 1 for b in bets if b.status == "SETTLED_WIN")
    overall_wr    = total_wins / total_bets
    avg_legs      = total_legs / total_bets

    # Geometric mean per-leg win rate
    implied_per_leg = overall_wr ** (1 / avg_legs) if avg_legs > 0 else 0

    # Naive per-leg rate
    naive_per_leg = legs_in_wins / total_legs if total_legs > 0 else 0

    # Straight bets (exact, no estimation needed)
    straights     = [b for b in bets if (b.legs or 1) == 1]
    straight_wr   = sum(1 for b in straights if b.status == "SETTLED_WIN") / len(straights) if straights else None

    # Per sport (geometric mean)
    sport_rates = {}
    sports_seen: dict = {}
    for b in bets:
        sport = (b.sports or "other").split("|")[0].strip()
        if sport not in sports_seen:
            sports_seen[sport] = {"wins": 0, "total": 0, "total_legs": 0}
        s = sports_seen[sport]
        s["total"]      += 1
        s["total_legs"] += (b.legs or 1)
        if b.status == "SETTLED_WIN":
            s["wins"] += 1

    for sport, s in sports_seen.items():
        if s["total"] < 5:
            continue
        wr       = s["wins"] / s["total"]
        avg_l    = s["total_legs"] / s["total"]
        implied  = wr ** (1 / avg_l) if avg_l > 0 and wr > 0 else 0
        sport_rates[sport] = {
            "bets":              s["total"],
            "wins":              s["wins"],
            "bet_win_rate":      round(wr * 100, 2),
            "avg_legs":          round(avg_l, 1),
            "implied_leg_rate":  round(implied * 100, 2),
        }

    # Per legs bucket
    bucket_rates = {}
    buckets: dict = {}
    for b in bets:
        legs = b.legs or 1
        if legs <= 2:   bkt = "1-2"
        elif legs <= 4: bkt = "3-4"
        elif legs <= 6: bkt = "5-6"
        elif legs <= 9: bkt = "7-9"
        else:           bkt = "10+"
        if bkt not in buckets:
            buckets[bkt] = {"wins": 0, "total": 0, "leg_sum": 0}
        buckets[bkt]["total"]    += 1
        buckets[bkt]["leg_sum"]  += legs
        if b.status == "SETTLED_WIN":
            buckets[bkt]["wins"] += 1

    bucket_order = ["1-2", "3-4", "5-6", "7-9", "10+"]
    for bkt in bucket_order:
        if bkt not in buckets:
            continue
        v     = buckets[bkt]
        wr    = v["wins"] / v["total"]
        avg_l = v["leg_sum"] / v["total"]
        impl  = wr ** (1 / avg_l) if avg_l > 0 and wr > 0 else 0
        bucket_rates[bkt] = {
            "bets":             v["total"],
            "wins":             v["wins"],
            "bet_win_rate":     round(wr * 100, 2),
            "avg_legs":         round(avg_l, 1),
            "implied_leg_rate": round(impl * 100, 2),
        }

    return {
        "overall": {
            "total_bets":           total_bets,
            "total_wins":           total_wins,
            "bet_win_rate":         round(overall_wr * 100, 2),
            "avg_legs_per_bet":     round(avg_legs, 2),
            "total_legs":           total_legs,
            "legs_in_wins":         legs_in_wins,
            "naive_leg_win_rate":   round(naive_per_leg * 100, 2),
            "implied_leg_win_rate": round(implied_per_leg * 100, 2),
            "straight_bet_win_rate":round(straight_wr * 100, 2) if straight_wr else None,
            "n_straight_bets":      len(straights),
            "method_note": (
                "Implied rate uses geometric mean: bet_win_rate ^ (1/avg_legs). "
                "This estimates true per-leg accuracy assuming leg independence. "
                f"Example: {round(overall_wr*100,1)}% parlay win rate across "
                f"{round(avg_legs,1)} avg legs → {round(implied_per_leg*100,2)}% per leg."
            ),
        },
        "by_sport":        sport_rates,
        "by_legs_bucket":  bucket_rates,
    }
