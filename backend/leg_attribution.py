"""
leg_attribution.py — System 2: Leg-level performance attribution.

Parses bet_legs rows and bet_info strings to surface which leg
types, sports, and markets win or lose most often.

─── Two distinct metrics ──────────────────────────────────────────────────────

  parlay_win_pct  — % of parlays *containing* this leg type that won overall.
                    This is what the current data supports (Pikkit only exports
                    parlay-level outcomes). Affected by all other legs in the
                    parlay; not a measure of whether THIS leg won.

  leg_win_pct     — % of individual legs that won, regardless of parlay result.
                    Requires per-leg outcomes (leg_result column in bet_legs).
                    Pikkit CSV does NOT export this data. Populated only when
                    legs are imported with individual results (e.g. FanDuel sync
                    with per-leg grading, or manual entry).

─── Functions ─────────────────────────────────────────────────────────────────
  analyze_legs_60d(db, days)          aggregate leg stats for the last N days
  get_leg_detail(db, bet_id, leg_idx) stats for one specific leg
  top_losing_patterns(db, days, n)    top N losing leg patterns ranked by P&L drag
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from database import Bet, BetLeg
from etl import classify_market
import math


# ─── Market classification helpers ────────────────────────────────────────────

_ML_KEYWORDS = {"moneyline", "ml", "to win", "team to win", "match winner"}
_SPREAD_KEYWORDS = {"spread", "handicap", "ats", "+", "-", "pts", "points"}
_TOTAL_KEYWORDS = {"over", "under", "total", "o/u", "ou"}
_PROP_KEYWORDS = {
    "alt", "props", "td", "touchdown", "goals", "assists", "rebounds",
    "strikeouts", "hits", "yards", "rushing", "receiving",
    "to score", "anytime", "1st",
}


def _classify(description: str, market_type: Optional[str] = None) -> str:
    """Return a canonical market bucket: Moneyline | Spread | Total | Player Prop | Other."""
    if market_type:
        m = market_type.lower()
        if "moneyline" in m or "ml" in m:
            return "Moneyline"
        if "spread" in m or "handicap" in m:
            return "Spread"
        if "total" in m or "over" in m or "under" in m:
            return "Total"
        if "prop" in m:
            return "Player Prop"

    desc = description.lower()
    for kw in _ML_KEYWORDS:
        if kw in desc:
            return "Moneyline"
    for kw in _TOTAL_KEYWORDS:
        if kw in desc:
            return "Total"
    for kw in _PROP_KEYWORDS:
        if kw in desc:
            return "Player Prop"
    for kw in _SPREAD_KEYWORDS:
        if kw in desc:
            return "Spread"
    return "Other"


def _odds_bucket(odds_str: str) -> str:
    """Bin American odds string into a bucket."""
    try:
        odds = int(re.sub(r"[^0-9\-+]", "", str(odds_str)))
    except Exception:
        return "Unknown"
    if odds <= -200:
        return "Heavy Fav (≤-200)"
    if odds <= -110:
        return "Fav (-110 to -200)"
    if odds <= -101:
        return "Slight Fav"
    if odds <= 110:
        return "Pick / Even"
    if odds <= 200:
        return "Dog (+110 to +200)"
    return "Big Dog (>+200)"


def _parse_fanduel_legs(bet_info: str) -> list[dict]:
    """
    Parse legs from a pipe-delimited bet_info string.

    BetIQ format (from fanduel_importer._parse_legs):
      "Pick (Market) — Matchup Odds | Pick2 (Market2) — ..."

    Returns list of {description, market_type, odds_str, pick, matchup}.
    """
    if not bet_info:
        return []
    legs = []
    for block in str(bet_info).split("|"):
        block = block.strip()
        if not block:
            continue
        # Extract odds at end (e.g. "+150" or "-110")
        odds_match = re.search(r"([+-]\d+)\s*$", block)
        odds_str = odds_match.group(1) if odds_match else ""
        # Extract market from parentheses
        market_match = re.search(r"\(([^)]+)\)", block)
        market = market_match.group(1) if market_match else ""
        legs.append({
            "description": block,
            "market_type": _classify(block, market),
            "odds_str": odds_str,
            "market_raw": market,
        })
    return legs


def _pct(wins: int, total: int) -> Optional[float]:
    return round(wins / total * 100, 1) if total > 0 else None


# ─── Core analysis ─────────────────────────────────────────────────────────────

def analyze_legs_60d(db: Session, days: int = 60) -> dict:
    """
    Aggregate leg-level attribution stats from all settled bets in the last N days.

    Returns two types of win rate for each dimension:

      parlay_win_pct  — % of parlays containing this leg type that won overall.
                        This is what the Pikkit CSV supports; affected by all
                        other legs in the parlay. NOT the same as individual leg win%.

      leg_win_pct     — % of individual legs that won (requires leg_result column).
                        null when no per-leg outcome data is available.

    The response also includes `has_leg_results` (bool) indicating whether any
    per-leg outcome data exists in the database.

    Returns:
        {
          "days":               int,
          "has_leg_results":    bool,
          "leg_results_note":   str,
          "by_market":          [...],
          "by_sport":           [...],
          "by_odds":            [...],
          "top_losing":         [...],
          "resolution_breakdown": {
            "inferred_parlay_win": int,   # all WIN; upward-biased
            "resolved_from_db":    int,   # game results; unbiased
            "straight_bet":        int,   # exact; unbiased
            "unresolvable":        int,
            "total_with_result":   int,
          },
          "unbiased_leg_win_pct": float | None,   # resolved_from_db + straight_bet only
          "win_rate_by_leg_count": [...],          # parlay size breakdown
          "parlay_size_insight":  {...},           # expected vs actual hit rate
          "summary":              {...},
        }

    Each row in by_market / by_sport / by_odds contains:
        parlay_count    — number of legs seen across all parlays in this bucket
        parlay_wins     — number of those parlays that won
        parlay_win_pct  — parlay_wins / parlay_count × 100
        leg_count       — legs with known individual outcomes (subset)
        leg_wins        — legs that individually won
        leg_win_pct     — leg_wins / leg_count × 100  (null if leg_count == 0)
        pnl_contribution — summed profit from parlays in this bucket (by_market only)
    """
    since = datetime.utcnow() - timedelta(days=days)

    # Defensive load: normalize any ISO-T datetime strings SQLite may hold
    # (can occur if a bet was manually settled via raw SQL using isoformat()).
    from sqlalchemy import text as _sa_text
    _fix_sql = _sa_text(
        "UPDATE bets SET time_settled = REPLACE(time_settled, 'T', ' ', 1) "
        "WHERE time_settled LIKE '%T%'"
    )
    try:
        db.execute(_fix_sql)
        db.commit()
    except Exception:
        pass

    settled_bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
        Bet.time_settled >= since,
        Bet.is_mock.is_(False),
    ).all()

    if not settled_bets:
        return {
            "days": days,
            "has_leg_results": False,
            "leg_results_note": (
                "No settled bets in the selected window. "
                "Note: Pikkit CSV does not export per-leg outcomes — "
                "parlay_win_pct is the only metric available from historical imports."
            ),
            "by_market": [], "by_sport": [], "by_odds": [],
            "top_losing": [], "summary": {"total_legs": 0, "total_bets": 0},
        }

    bet_ids = [b.id for b in settled_bets]
    bet_lookup = {b.id: b for b in settled_bets}

    # ── 1. Pull from bet_legs table ──────────────────────────────────────────
    db_legs = db.query(BetLeg).filter(BetLeg.bet_id.in_(bet_ids)).all()
    leg_rows: list[dict] = []
    covered_ids: set[str] = set()
    any_leg_results = False

    for leg in db_legs:
        bet = bet_lookup.get(leg.bet_id)
        if not bet:
            continue
        covered_ids.add(leg.bet_id)

        # Individual leg outcome (null unless explicitly stored)
        lr = (leg.leg_result or "").upper() if leg.leg_result else None
        if lr in ("WIN", "LOSS", "PUSH"):
            any_leg_results = True
        else:
            lr = None

        # Use stored odds_str if available (populated for FanDuel legs and straight bets)
        odds_str = getattr(leg, "odds_str", None) or ""

        # Resolution source for bias tracking
        res_src = getattr(leg, "resolution_source", None) or ""

        leg_rows.append({
            "bet_id":            leg.bet_id,
            "sport":             leg.sport or bet.sports or "Unknown",
            "market":            _classify(leg.description or "", leg.market_type),
            "odds_str":          odds_str,
            "parlay_won":        bet.status == "SETTLED_WIN",
            "leg_result":        lr,          # WIN / LOSS / PUSH / None
            "profit":            (bet.profit or 0),
            "description":       leg.description or "",
            "resolution_source": res_src,
        })

    # ── 2. Fall back to parse_legs_from_bet_info for uncovered bets ─────────
    try:
        from fanduel_importer import parse_legs_from_bet_info as _parse_full
    except ImportError:
        _parse_full = None

    for bet in settled_bets:
        if bet.id in covered_ids:
            continue
        if not bet.bet_info:
            continue
        parsed = _parse_full(bet.bet_info, bet.sports or "") if _parse_full else _parse_fanduel_legs(bet.bet_info)
        for p in parsed:
            leg_rows.append({
                "bet_id":            bet.id,
                "sport":             p.get("sport") or bet.sports or "Unknown",
                "market":            p.get("market_type") or p.get("market_type", "Other"),
                "odds_str":          p.get("odds_str", ""),
                "parlay_won":        bet.status == "SETTLED_WIN",
                "leg_result":        None,    # fallback path never has per-leg result
                "profit":            (bet.profit or 0) / max(len(parsed), 1),
                "description":       p.get("description", ""),
                "resolution_source": "",
            })

    if not leg_rows:
        return {
            "days": days,
            "has_leg_results": False,
            "leg_results_note": (
                "Pikkit CSV does not export per-leg outcomes. "
                "Win rates shown are parlay-level outcomes, not individual leg win rates. "
                "Import leg-level outcomes from Pikkit (if available) or enable FanDuel "
                "sync with per-leg grading to unlock true per-leg analysis."
            ),
            "by_market": [], "by_sport": [], "by_odds": [],
            "top_losing": [], "summary": {"total_legs": 0, "total_bets": len(settled_bets)},
        }

    # ── 3. Aggregate ─────────────────────────────────────────────────────────
    # Parlay-level: keyed on (dimension, bet_id) so we count parlays, not legs
    # (a 9-leg parlay with 9 baseball legs counts as 1 baseball parlay, not 9)
    # Leg-level: each leg is its own data point

    # ── 3a. Parlay-level dedup ───────────────────────────────────────────────
    market_parlays:  dict[str, dict[str, bool]] = defaultdict(dict)
    sport_parlays:   dict[str, dict[str, bool]] = defaultdict(dict)
    odds_parlays:    dict[str, dict[str, bool]] = defaultdict(dict)
    market_pnl:      dict[str, float] = defaultdict(float)

    # ── 3b. Per-leg accumulators ─────────────────────────────────────────────
    market_legs: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})
    sport_legs:  dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})
    odds_legs:   dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})

    desc_drag: dict[str, dict] = defaultdict(lambda: {"appearances": 0, "drag": 0.0, "market": "", "sport": ""})

    # ── 3c. Resolution source breakdown + unbiased win rate accumulators ─────
    res_counts = {"inferred_parlay_win": 0, "resolved_from_db": 0,
                  "straight_bet": 0, "unresolvable": 0}
    unbiased_wins = 0
    unbiased_total = 0

    for row in leg_rows:
        m = row["market"]
        s = (row["sport"] or "Unknown").split("|")[0].strip()
        ob = _odds_bucket(row["odds_str"])
        bid = row["bet_id"]
        won = row["parlay_won"]
        lr = row["leg_result"]
        rsrc = row.get("resolution_source", "") or ""

        # Parlay-level (each bet_id counted once per dimension bucket)
        market_parlays[m][bid] = won
        sport_parlays[s][bid] = won
        odds_parlays[ob][bid] = won
        market_pnl[m] += row["profit"]

        # Per-leg (only when leg_result is known)
        if lr in ("WIN", "LOSS"):
            market_legs[m]["count"] += 1
            market_legs[m]["wins"]  += 1 if lr == "WIN" else 0
            sport_legs[s]["count"]  += 1
            sport_legs[s]["wins"]   += 1 if lr == "WIN" else 0
            odds_legs[ob]["count"]  += 1
            odds_legs[ob]["wins"]   += 1 if lr == "WIN" else 0

            # Resolution source tracking
            if rsrc == "inferred_parlay_win":
                res_counts["inferred_parlay_win"] += 1
            elif rsrc in ("historical_db", "pitcher_logs"):
                res_counts["resolved_from_db"] += 1
                unbiased_total += 1
                unbiased_wins  += 1 if lr == "WIN" else 0
            elif rsrc == "" and bet_lookup.get(bid) and bet_lookup[bid].legs == 1:
                # Straight bet inferred from parlay outcome column
                res_counts["straight_bet"] += 1
                unbiased_total += 1
                unbiased_wins  += 1 if lr == "WIN" else 0
            elif rsrc:
                res_counts["unresolvable"] += 1

        # Top losing patterns (parlay-level P&L drag)
        if not won and row["description"]:
            key = row["description"][:80]
            desc_drag[key]["appearances"] += 1
            desc_drag[key]["drag"] += abs(row["profit"])
            desc_drag[key]["market"] = m
            desc_drag[key]["sport"]  = row["sport"] or "Unknown"

    # ── 3d. Parlay size vs win rate ──────────────────────────────────────────
    # Each bucket: {leg_count_label: {parlays: int, wins: int}}
    size_buckets: dict[str, dict] = defaultdict(lambda: {"parlays": 0, "wins": 0})
    for bet in settled_bets:
        n = bet.legs or 1
        label = str(n) if n <= 6 else "7+"
        size_buckets[label]["parlays"] += 1
        size_buckets[label]["wins"]    += 1 if bet.status == "SETTLED_WIN" else 0

    # Sort: 1,2,3,4,5,6,7+
    _order = ["1", "2", "3", "4", "5", "6", "7+"]
    win_rate_by_leg_count = []
    for label in _order:
        if label not in size_buckets:
            continue
        bk = size_buckets[label]
        win_rate_by_leg_count.append({
            "legs":     label,
            "parlays":  bk["parlays"],
            "wins":     bk["wins"],
            "win_pct":  _pct(bk["wins"], bk["parlays"]),
        })

    # ── 3e. Parlay size insight ──────────────────────────────────────────────
    # Use unbiased win rate as the per-leg edge estimate
    per_leg_win_rate = (unbiased_wins / unbiased_total) if unbiased_total > 0 else None
    avg_legs = (
        sum((int(b.legs or 1) for b in settled_bets)) / len(settled_bets)
        if settled_bets else None
    )
    actual_parlay_win_pct = _pct(
        sum(1 for b in settled_bets if b.status == "SETTLED_WIN"),
        len(settled_bets)
    )

    if per_leg_win_rate and avg_legs:
        expected_pct = round(per_leg_win_rate ** avg_legs * 100, 1)
        target_3leg  = round(per_leg_win_rate ** 3  * 100, 1)
        target_4leg  = round(per_leg_win_rate ** 4  * 100, 1)
        parlay_size_insight = {
            "per_leg_win_rate_pct":   round(per_leg_win_rate * 100, 1),
            "avg_legs":               round(avg_legs, 1),
            "expected_hit_rate_pct":  expected_pct,
            "actual_hit_rate_pct":    actual_parlay_win_pct,
            "gap_pct":                round((actual_parlay_win_pct or 0) - expected_pct, 1),
            "target_3leg_hit_rate_pct": target_3leg,
            "target_4leg_hit_rate_pct": target_4leg,
            "unbiased_legs_sample":   unbiased_total,
        }
    else:
        parlay_size_insight = None

    # ── 3c. Build output rows ────────────────────────────────────────────────
    def _parlay_row(key, parlays_dict, legs_dict, pnl_dict=None):
        p_dict = parlays_dict[key]
        p_count = len(p_dict)
        p_wins  = sum(1 for w in p_dict.values() if w)
        l_stat  = legs_dict.get(key, {"count": 0, "wins": 0})
        row = {
            "parlay_count":   p_count,
            "parlay_wins":    p_wins,
            "parlay_win_pct": _pct(p_wins, p_count),
            "leg_count":      l_stat["count"],
            "leg_wins":       l_stat["wins"],
            "leg_win_pct":    _pct(l_stat["wins"], l_stat["count"]),
        }
        if pnl_dict is not None:
            row["pnl_contribution"] = round(pnl_dict.get(key, 0.0), 2)
        return row

    by_market = sorted([
        {"market_type": k, **_parlay_row(k, market_parlays, market_legs, market_pnl)}
        for k in market_parlays
    ], key=lambda x: -x["parlay_count"])

    by_sport = sorted([
        {"sport": k, **_parlay_row(k, sport_parlays, sport_legs)}
        for k in sport_parlays
    ], key=lambda x: -x["parlay_count"])

    by_odds = sorted([
        {"odds_bucket": k, **_parlay_row(k, odds_parlays, odds_legs)}
        for k in odds_parlays
    ], key=lambda x: -x["parlay_count"])

    top_losing = sorted([
        {
            "description": k,
            "appearances": v["appearances"],
            "drag":        round(v["drag"], 2),
            "market_type": v["market"],
            "sport":       v["sport"],
        }
        for k, v in desc_drag.items() if v["appearances"] >= 2
    ], key=lambda x: -x["drag"])[:10]

    total_parlays = len(settled_bets)
    parlay_wins   = sum(1 for b in settled_bets if b.status == "SETTLED_WIN")

    if any_leg_results:
        note = (
            "Per-leg outcomes available for some bets. "
            "Leg Win% shows individual leg performance. "
            "Parlay Win% shows % of parlays containing this leg type that won overall."
        )
    else:
        note = (
            "Win rates shown are parlay-level outcomes, not individual leg win rates. "
            "Pikkit CSV does not export per-leg results — only the overall parlay "
            "outcome (WIN/LOSS) is available. "
            "Leg Win% will populate as future bets are logged with per-leg grading."
        )

    res_counts["total_with_result"] = (
        res_counts["inferred_parlay_win"]
        + res_counts["resolved_from_db"]
        + res_counts["straight_bet"]
    )

    return {
        "days":              days,
        "has_leg_results":   any_leg_results,
        "leg_results_note":  note,
        "by_market":         by_market,
        "by_sport":          by_sport,
        "by_odds":           by_odds,
        "top_losing":        top_losing,
        "resolution_breakdown":  res_counts,
        "unbiased_leg_win_pct":  _pct(unbiased_wins, unbiased_total),
        "win_rate_by_leg_count": win_rate_by_leg_count,
        "parlay_size_insight":   parlay_size_insight,
        "summary": {
            "total_legs":        len(leg_rows),
            "total_bets":        total_parlays,
            "parlay_win_pct":    _pct(parlay_wins, total_parlays),
            "has_leg_results":   any_leg_results,
        },
    }


def get_leg_detail(db: Session, bet_id: str, leg_index: int) -> Optional[dict]:
    """
    Return detail for a specific leg: its description, market type,
    historical parlay win rate for similar legs, and the bet outcome.
    """
    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if not bet:
        return None

    # Try bet_legs table first
    leg = db.query(BetLeg).filter(
        BetLeg.bet_id == bet_id,
        BetLeg.leg_index == leg_index,
    ).first()

    if leg:
        market = _classify(leg.description or "", leg.market_type)
        description = leg.description or ""
        sport = leg.sport or bet.sports or "Unknown"
        leg_result = leg.leg_result
    else:
        # Parse from bet_info
        parts = _parse_fanduel_legs(bet.bet_info or "")
        if leg_index >= len(parts):
            return None
        p = parts[leg_index]
        market = p["market_type"]
        description = p["description"]
        sport = (bet.sports or "Unknown").split("|")[0].strip()
        leg_result = None

    # Historical parlay win rate for this market + sport combo (last 90 days)
    hist = analyze_legs_60d(db, days=90)
    market_parlay_win_pct = None
    market_leg_win_pct = None
    for row in hist["by_market"]:
        if row["market_type"] == market:
            market_parlay_win_pct = row["parlay_win_pct"]
            market_leg_win_pct    = row["leg_win_pct"]
            break

    return {
        "bet_id":                   bet_id,
        "leg_index":                leg_index,
        "description":              description,
        "market_type":              market,
        "sport":                    sport,
        "leg_result":               leg_result,
        "bet_outcome":              bet.status,
        "bet_profit":               bet.profit,
        "market_parlay_win_pct_90d": market_parlay_win_pct,
        "market_leg_win_pct_90d":   market_leg_win_pct,
        # Legacy alias kept for any existing callers
        "market_win_pct_90d":       market_parlay_win_pct,
    }
