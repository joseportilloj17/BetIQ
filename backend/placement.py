"""
placement.py — Placement recommendations layer for BetIQ.

Reads scouted_props for today and generates placement recommendations:
  - Grade-weighted Kelly sizing
  - Skip/play decisions based on threshold and grade
  - Parlay grouping suggestions (A-grade legs only)
  - Integration hook for recommender to attach scout grade to legs

Called by:
  - GET /api/placement/recommendations
  - POST /api/placement/size (single-bet Kelly with scout grade)
  - recommender.py (attaches scout_grade to each generated leg)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

# ── Constants ──────────────────────────────────────────────────────────────────

# Kelly fraction by grade (quarter-Kelly baseline scaled by confidence)
_KELLY_FRACTION = {
    "A": 0.25,   # full quarter-Kelly
    "B": 0.18,
    "C": 0.10,
    "D": 0.00,   # skip / bet opposite
}

# Minimum hit probability to recommend OVER (not UNDER/skip)
_PLAY_THRESHOLD = 0.55

# Grade → composite score multiplier (mirrors scout/base.py GRADE_MULTIPLIERS)
_GRADE_MULTIPLIERS = {
    "A": 1.5,
    "B": 1.2,
    "C": 1.0,
    "D": 0.5,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _kelly_stake(win_prob: float, decimal_odds: float, bankroll: float, fraction: float = 0.25) -> float:
    """Fractional Kelly criterion stake."""
    if decimal_odds <= 1 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    b = decimal_odds - 1
    q = 1 - win_prob
    k = (b * win_prob - q) / b
    return max(0.0, k * fraction * bankroll)


def _grade_multiplier(grade: str) -> float:
    return _GRADE_MULTIPLIERS.get(grade, 1.0)


def _load_bankroll() -> float:
    """Load current bankroll from kelly.py storage."""
    try:
        import kelly
        return _safe_float(kelly.load_bankroll().get("current_bankroll", 200.0))
    except Exception:
        return 200.0


# ── Main functions ─────────────────────────────────────────────────────────────

def get_todays_recommendations(
    db: Session,
    sport_filter: Optional[list] = None,
    min_grade: str = "B",
    limit: int = 30,
) -> dict:
    """
    Return placement recommendations for today's top-graded props.

    For each recommended prop:
      - action: PLAY_OVER | PLAY_UNDER | SKIP
      - kelly_stake: suggested stake in $ (fractional Kelly)
      - scout_multiplier: grade-based multiplier for composite score
      - reason: brief explanation string

    min_grade: minimum quality_grade to include (A|B|C|D).
    """
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bankroll   = _load_bankroll()

    grade_order = {"A": 4, "B": 3, "C": 2, "D": 1}
    min_rank    = grade_order.get(min_grade.upper(), 3)

    filters = ["scout_date = :date", "hit_probability >= :min_prob"]
    params: dict = {"date": scout_date, "min_prob": 0.50}

    if sport_filter:
        placeholders = ",".join(f":s{i}" for i in range(len(sport_filter)))
        filters.append(f"sport IN ({placeholders})")
        for i, s in enumerate(sport_filter):
            params[f"s{i}"] = s.upper()

    where = " AND ".join(filters)

    rows = db.execute(text(f"""
        SELECT id, sport, market_type, player_name, team, side,
               threshold, projected_value, hit_probability, quality_grade,
               confidence_factors, risk_factors, home_team, away_team,
               projected_low_95, projected_high_95, projected_std_dev
        FROM   scouted_props
        WHERE  {where}
        ORDER  BY hit_probability DESC
        LIMIT  :limit
    """), {**params, "limit": limit * 3}).fetchall()

    recommendations = []
    for row in rows:
        (prop_id, sport, market_type, player_name, team, side,
         threshold, projected_value, hit_prob, grade,
         confidence_json, risk_json, home_team, away_team,
         proj_low, proj_high, proj_std) = row

        grade_rank = grade_order.get(grade, 0)
        if grade_rank < min_rank:
            continue

        # Decide action
        if hit_prob >= _PLAY_THRESHOLD:
            action = f"PLAY_{(side or 'over').upper()}"
        else:
            action = "SKIP"

        # Kelly sizing (approximate decimal odds from hit_prob for unpriced props)
        approx_decimal_odds = 1.0 / max(0.01, hit_prob)   # no-vig implied odds
        kelly_frac = _KELLY_FRACTION.get(grade, 0.10)
        stake = _kelly_stake(hit_prob, approx_decimal_odds, bankroll, kelly_frac)

        # Confidence / risk parsing
        try:
            confidence = json.loads(confidence_json) if confidence_json else []
        except Exception:
            confidence = []
        try:
            risks = json.loads(risk_json) if risk_json else []
        except Exception:
            risks = []

        matchup = f"{away_team} @ {home_team}"
        desc = player_name or team or matchup
        reason = (
            f"Grade {grade} ({hit_prob*100:.0f}% hit prob) — {'; '.join(confidence[:2])}"
        )
        if risks:
            reason += f" | Risks: {risks[0]}"

        recommendations.append({
            "prop_id":          prop_id,
            "sport":            sport,
            "matchup":          matchup,
            "market_type":      market_type,
            "description":      desc,
            "side":             side,
            "threshold":        threshold,
            "projected_value":  projected_value,
            "projected_low_95": round(proj_low, 2) if proj_low is not None else None,
            "projected_high_95": round(proj_high, 2) if proj_high is not None else None,
            "projected_std_dev": round(proj_std, 2) if proj_std is not None else None,
            "hit_probability":  round(hit_prob, 4),
            "quality_grade":    grade,
            "action":           action,
            "kelly_stake":      round(stake, 2),
            "scout_multiplier": _grade_multiplier(grade),
            "reason":           reason,
            "confidence":       confidence,
            "risks":            risks,
        })

        if len(recommendations) >= limit:
            break

    play_recs = [r for r in recommendations if r["action"].startswith("PLAY")]
    total_kelly = sum(r["kelly_stake"] for r in play_recs)
    combined_prob = 1.0
    for r in play_recs[:4]:  # top 4 plays parlay probability
        combined_prob *= r["hit_probability"]

    return {
        "scout_date":         scout_date,
        "bankroll":           bankroll,
        "total":              len(recommendations),
        "play_count":         len(play_recs),
        "total_kelly_stake":  round(total_kelly, 2),
        "top4_parlay_prob":   round(combined_prob, 4) if len(play_recs) >= 2 else None,
        "recommendations":    recommendations,
    }


def size_single_bet(
    hit_probability: float,
    quality_grade: str,
    decimal_odds: float,
    bankroll: Optional[float] = None,
) -> dict:
    """
    Kelly sizing for a single scouted prop.
    Returns stake, fraction, and grade multiplier.
    """
    bankroll = bankroll or _load_bankroll()
    kelly_frac = _KELLY_FRACTION.get(quality_grade.upper(), 0.10)
    stake = _kelly_stake(hit_probability, decimal_odds, bankroll, kelly_frac)
    return {
        "quality_grade":    quality_grade,
        "hit_probability":  hit_probability,
        "decimal_odds":     decimal_odds,
        "bankroll":         bankroll,
        "kelly_fraction":   kelly_frac,
        "recommended_stake": round(stake, 2),
        "scout_multiplier": _grade_multiplier(quality_grade),
    }


def attach_scout_grade_to_leg(leg: dict, db: Session) -> dict:
    """
    Look up scouted_props for today matching this leg's description/fixture,
    and attach scout_grade + scout_hit_prob to the leg dict.
    Called by recommender when building parlay candidates.

    Matching strategy:
      1. game_id + market_type match (exact)
      2. player_name + market_type (fuzzy fallback)
    If no match found, scout_grade=None.
    """
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fixture_id  = leg.get("fixture_id", "")
    market_type = leg.get("market_type", "")
    player_name = leg.get("player_name", leg.get("description", ""))
    sport       = leg.get("sport", "")

    try:
        # Try exact game_id + market_type match first
        row = None
        if fixture_id:
            row = db.execute(text("""
                SELECT quality_grade, hit_probability, id
                FROM   scouted_props
                WHERE  scout_date = :d AND game_id = :gid AND market_type = :mt
                ORDER  BY hit_probability DESC
                LIMIT  1
            """), {"d": scout_date, "gid": fixture_id, "mt": market_type}).fetchone()

        # Fall back to player name match
        if not row and player_name:
            row = db.execute(text("""
                SELECT quality_grade, hit_probability, id
                FROM   scouted_props
                WHERE  scout_date = :d AND player_name LIKE :pn AND market_type = :mt
                ORDER  BY hit_probability DESC
                LIMIT  1
            """), {"d": scout_date, "pn": f"%{player_name}%", "mt": market_type}).fetchone()

        if row:
            leg["scout_grade"]    = row[0]
            leg["scout_hit_prob"] = round(float(row[1]), 4)
            leg["scouted_prop_id"] = row[2]
            leg["scout_multiplier"] = _grade_multiplier(row[0])
        else:
            leg["scout_grade"]    = None
            leg["scout_hit_prob"] = None
            leg["scouted_prop_id"] = None
            leg["scout_multiplier"] = 1.0

    except Exception as exc:
        print(f"[placement] attach_scout_grade error: {exc}")
        leg["scout_grade"]    = None
        leg["scout_hit_prob"] = None
        leg["scouted_prop_id"] = None
        leg["scout_multiplier"] = 1.0

    return leg


def get_parlay_suggestions(db: Session, max_legs: int = 3) -> dict:
    """
    Suggest top A-grade props as parlay legs.
    Returns up to `max_legs` combinations ranked by combined hit probability.
    """
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows = db.execute(text("""
        SELECT id, sport, market_type, player_name, team, side,
               threshold, hit_probability, quality_grade,
               home_team, away_team
        FROM   scouted_props
        WHERE  scout_date = :d AND quality_grade = 'A'
        ORDER  BY hit_probability DESC
        LIMIT  :n
    """), {"d": scout_date, "n": max_legs * 2}).fetchall()

    legs = []
    for r in rows:
        combined = 1.0
        player_or_team = r[3] or r[4] or ""
        legs.append({
            "prop_id":       r[0],
            "sport":         r[1],
            "market_type":   r[2],
            "description":   player_or_team,
            "side":          r[5],
            "threshold":     r[6],
            "hit_prob":      round(float(r[7]), 4),
            "grade":         r[8],
            "matchup":       f"{r[10]} @ {r[9]}",
        })

    # Combined parlay probability (independent legs assumption)
    top_n = legs[:max_legs]
    combined_prob = 1.0
    for lg in top_n:
        combined_prob *= lg["hit_prob"]

    return {
        "scout_date":     scout_date,
        "legs":           top_n,
        "combined_prob":  round(combined_prob, 4),
        "grade":          "A" if combined_prob >= 0.20 else "B",
    }
