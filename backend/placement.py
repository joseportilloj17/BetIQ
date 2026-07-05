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

# Fix A+B: market type normalization — leg market_type → scouted_props market_type
_MARKET_NORM: dict[str, str] = {
    "h2h":               "h2h",
    "spreads":           "spread",
    "alternate_spreads": "spread",
    "totals":            "totals",
    "alternate_totals":  "totals",
}

# Fix A: OddsAPI sport_key → scouted_props sport name
_SPORT_NORM: dict[str, str] = {
    "baseball_mlb":          "MLB",
    "icehockey_nhl":         "NHL",
    "basketball_nba":        "NBA",
    "americanfootball_nfl":  "NFL",
}

# Grade → Kelly sizing multiplier (for placement recommendations only — not composite scoring)
_GRADE_MULTIPLIERS = {
    "A": 1.5,
    "B": 1.2,
    "C": 1.0,
    "D": 0.5,
}

# ── Dynamic scout multipliers ──────────────────────────────────────────────────
# Multipliers relative to Grade B (= 1.00).  Used in composite scoring via
# recommender.py.  Updated from empirical validation data after each batch.
#
# First validation run — May 2026 (325 settled legs):
#   Grade A: 73.2% WR, 0.198 Brier  (n=112) → strong signal, moderate boost
#   Grade B: 60.0% WR, 0.240 Brier  (n=170) → baseline
#   Grade C: 72.2% WR, 0.234 Brier  (n=36)  → uncertain, hold at baseline
#   Grade D: 57.1% WR, 0.250 Brier  (n=7)   → directional only, conservative penalty
#
# Review thresholds:
#   A → raise to 1.35–1.50 once n≥200 and WR stays ≥70%
#   C → investigate and possibly increase if WR holds ≥68% at n≥80
#   D → lower to 0.50–0.60 once n≥30 and WR confirms <55%
_SCOUT_GRADE_MULTIPLIERS: dict[str, float] = {
    "A": 1.25,
    "B": 1.00,
    "C": 1.00,
    "D": 0.75,
}
_SCOUT_GRADE_MULT_MIN_N = 50   # minimum legs before a grade's dynamic multiplier is trusted

# Cache for dynamic multipliers (refreshed after each settlement batch)
_dynamic_scout_mult_cache: dict[str, float] = {}
_dynamic_scout_mult_ts: float = 0.0
_DYNAMIC_SCOUT_MULT_TTL = 600.0   # 10-minute TTL


def get_dynamic_scout_multipliers(db: Session, min_n: int = _SCOUT_GRADE_MULT_MIN_N) -> dict[str, float]:
    """
    Compute scout grade multipliers from empirical leg-level WR in the past 30 days.
    Grade B = 1.00 (baseline); other grades scaled relative to B.
    Returns cached result within TTL; falls back to _SCOUT_GRADE_MULTIPLIERS
    when Grade B has < min_n settled legs.

    Multipliers are clamped to [0.50, 1.75] to prevent extreme outliers.
    """
    import time

    global _dynamic_scout_mult_cache, _dynamic_scout_mult_ts
    if _dynamic_scout_mult_cache and (time.time() - _dynamic_scout_mult_ts) < _DYNAMIC_SCOUT_MULT_TTL:
        return _dynamic_scout_mult_cache

    try:
        rows = db.execute(text("""
            SELECT
                mbl.scout_grade,
                COUNT(*)                                                                  AS n,
                AVG(CASE WHEN mbl.leg_result = 'WIN' THEN 1.0 ELSE 0.0 END)             AS wr
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.scout_grade IS NOT NULL
              AND mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mb.game_date >= date('now', '-30 days')
            GROUP BY mbl.scout_grade
        """)).fetchall()

        b_wr = None
        grade_data: dict[str, tuple[int, float]] = {}
        for r in rows:
            grade = (r[0] or "").upper()
            n, wr = int(r[1] or 0), float(r[2] or 0)
            grade_data[grade] = (n, wr)
            if grade == "B" and n >= min_n:
                b_wr = wr

        if b_wr is None or b_wr == 0:
            # Not enough Grade B data — use hardcoded fallback
            return _SCOUT_GRADE_MULTIPLIERS

        result: dict[str, float] = {}
        for grade, targets in _SCOUT_GRADE_MULTIPLIERS.items():
            if grade in grade_data and grade_data[grade][0] >= min_n:
                raw = grade_data[grade][1] / b_wr
                result[grade] = round(max(0.50, min(1.75, raw)), 3)
            else:
                # Insufficient data: use hardcoded value for this grade
                result[grade] = _SCOUT_GRADE_MULTIPLIERS[grade]

        _dynamic_scout_mult_cache = result
        _dynamic_scout_mult_ts    = time.time()
        return result

    except Exception:
        return _SCOUT_GRADE_MULTIPLIERS


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


def _extract_player_name(description: str) -> str:
    """
    Extract likely player name from a leg description like
    'LeBron James OVER 25.5 Points' → 'LeBron James'.
    Strips trailing keywords and odds/numbers.
    """
    import re
    # Remove anything after OVER/UNDER/+/-/@ and numbers
    cleaned = re.split(r'\s+(?:OVER|UNDER|O|U|@|\+|-|\d)', description, maxsplit=1)[0]
    return cleaned.strip()


def _extract_team_from_description(description: str) -> str:
    """
    Extract team name from a game-level leg description.
      "Pittsburgh Pirates +1.5 (Spread)"   → "Pittsburgh Pirates"
      "Athletics +1.5 (alternate_spreads)"  → "Athletics"
      "Washington Nationals (h2h)"          → "Washington Nationals"
      "Over +8.5 (totals)"                  → "" (no team — caller handles)
    """
    import re
    # Strip everything from the first +/- followed by a digit
    cleaned = re.split(r'\s+[+-]\d', description, maxsplit=1)[0]
    # Strip trailing parenthetical market label e.g. " (h2h)"
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned).strip()
    # Reject known non-team tokens
    if cleaned.lower() in ("over", "under", "o", "u", ""):
        return ""
    return cleaned


def _scout_lookup_raw(fixture_id: str, market_type: str, description: str,
                      db_path: str) -> tuple:
    """
    Thread-safe scout lookup using a dedicated sqlite3 connection.
    Returns (quality_grade, hit_probability, id) or None.
    Avoids SQLAlchemy session contention in background threads.
    """
    import sqlite3 as _sql3
    scout_market = _MARKET_NORM.get(market_type)
    if not fixture_id or not scout_market:
        return None
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from datetime import timedelta
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    conn = _sql3.connect(db_path, timeout=10)
    try:
        fix_row = conn.execute(
            "SELECT home_team, away_team, COALESCE(game_date_ct, date(commence_time)) AS game_date, sport_key "
            "FROM fixtures WHERE id = ?", [fixture_id]).fetchone()
        if not fix_row:
            return None
        home_team, away_team, game_date, sport_key = fix_row
        sport = _SPORT_NORM.get(sport_key, sport_key)
        # Scout may have been run yesterday for today's game — check all candidate dates
        _dates = list(dict.fromkeys([game_date, scout_date, yesterday]))
        # Try both home/away orderings — scouted_props may store
        # the teams in a different order than the fixture.
        _team_pairs = [
            (home_team, away_team),
            (away_team, home_team),
        ]
        row = None
        if scout_market == "totals":
            for d in _dates:
                for _ht, _at in _team_pairs:
                    row = conn.execute(
                        "SELECT quality_grade, hit_probability, id FROM scouted_props "
                        "WHERE scout_date=? AND home_team=? AND away_team=? "
                        "AND sport=? AND market_type=? ORDER BY hit_probability DESC LIMIT 1",
                        [d, _ht, _at, sport, scout_market]).fetchone()
                    if row:
                        break
                if row:
                    break
        else:
            bet_team = _extract_team_from_description(description)
            for d in _dates:
                for _ht, _at in _team_pairs:
                    if bet_team:
                        row = conn.execute(
                            "SELECT quality_grade, hit_probability, id FROM scouted_props "
                            "WHERE scout_date=? AND home_team=? AND away_team=? "
                            "AND sport=? AND market_type=? AND team=? "
                            "ORDER BY hit_probability DESC LIMIT 1",
                            [d, _ht, _at, sport, scout_market, bet_team]).fetchone()
                        if not row:
                            last_word = bet_team.split()[-1]
                            if len(last_word) >= 4:
                                row = conn.execute(
                                    "SELECT quality_grade, hit_probability, id FROM scouted_props "
                                    "WHERE scout_date=? AND home_team=? AND away_team=? "
                                    "AND sport=? AND market_type=? AND team LIKE ? "
                                    "ORDER BY hit_probability DESC LIMIT 1",
                                    [d, _ht, _at, sport, scout_market,
                                     f"%{last_word}%"]).fetchone()
                    if not row:
                        row = conn.execute(
                            "SELECT quality_grade, hit_probability, id FROM scouted_props "
                            "WHERE scout_date=? AND home_team=? AND away_team=? "
                            "AND sport=? AND market_type=? "
                            "ORDER BY hit_probability DESC LIMIT 1",
                            [d, _ht, _at, sport, scout_market]).fetchone()
                    if row:
                        break
                if row:
                    break
        if not row:
            player_name = _extract_player_name(description)
            if player_name and len(player_name) >= 4:
                row = conn.execute(
                    "SELECT quality_grade, hit_probability, id FROM scouted_props "
                    "WHERE scout_date=? AND player_name=? "
                    "ORDER BY hit_probability DESC LIMIT 1",
                    [scout_date, player_name]).fetchone()
                if not row:
                    last = player_name.split()[-1]
                    if len(last) >= 4:
                        row = conn.execute(
                            "SELECT quality_grade, hit_probability, id FROM scouted_props "
                            "WHERE scout_date=? AND player_name LIKE ? "
                            "ORDER BY hit_probability DESC LIMIT 1",
                            [scout_date, f"%{last}%"]).fetchone()
        return row
    finally:
        conn.close()


def attach_scout_grade_to_leg(leg: dict, db: Session) -> dict:
    """
    Look up scouted_props for today matching this leg's description/fixture,
    and attach scout_grade + scout_hit_prob to the leg dict.
    Called by recommender when building parlay candidates.

    Uses a dedicated sqlite3 connection (_scout_lookup_raw) for thread safety
    in background generation threads, with ORM session as fallback.

    Matching strategy (Fix A + B + C):
      Primary — bridge via Fixtures table:
        1. fixture_id  → fixtures.id  → home_team, away_team, game_date, sport_key
        2. Normalize market_type  (alternate_spreads → spread, spreads → spread, etc.)
        3. Normalize sport_key    (baseball_mlb → MLB, icehockey_nhl → NHL, …)
        4. For spread/h2h: extract bet team from description, match scouted_props.team
           Fallback to best grade for the game if team match fails.
        5. For totals: just fetch best grade for the game.
      Secondary — player-name match (for future player props):
        6. player_name field or extracted from description
    """
    import os
    fixture_id  = leg.get("fixture_id", "") or leg.get("game_id", "")
    market_type = leg.get("market_type", "")
    description = leg.get("description", "") or ""

    try:
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
        row = _scout_lookup_raw(fixture_id, market_type, description, db_path)

        if row:
            leg["scout_grade"]     = row[0]
            leg["scout_hit_prob"]  = round(float(row[1]), 4)
            leg["scouted_prop_id"] = row[2]
            leg["scout_multiplier"] = _grade_multiplier(row[0])
        else:
            leg["scout_grade"]     = None
            leg["scout_hit_prob"]  = None
            leg["scouted_prop_id"] = None
            leg["scout_multiplier"] = 1.0

    except Exception as exc:
        print(f"[placement] attach_scout_grade error: {exc}")
        leg["scout_grade"]     = None
        leg["scout_hit_prob"]  = None
        leg["scouted_prop_id"] = None
        leg["scout_multiplier"] = 1.0

    return leg


def backfill_scout_grades_for_today() -> int:
    """
    Re-enrich mock_bet_legs that have NULL scout_grade for today's game_date.
    Called after the scout cron completes to patch legs that were generated
    before scout data was available.

    Returns the number of legs updated.
    """
    import os
    import sqlite3 as _sql3
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
    conn = _sql3.connect(db_path, timeout=10)
    updated = 0
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        from datetime import timedelta
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        # Find legs with NULL scout_grade for today's or yesterday's games
        rows = conn.execute(
            "SELECT mbl.id, mbl.fixture_id, mbl.market_type, mbl.description "
            "FROM mock_bet_legs mbl "
            "JOIN mock_bets mb ON mb.id = mbl.mock_bet_id "
            "WHERE mb.game_date IN (?, ?) "
            "AND mbl.scout_grade IS NULL "
            "AND mbl.fixture_id IS NOT NULL",
            [today, yesterday],
        ).fetchall()

        for leg_id, fixture_id, market_type, description in rows:
            row = _scout_lookup_raw(fixture_id, market_type, description or "", db_path)
            if row:
                conn.execute(
                    "UPDATE mock_bet_legs SET scout_grade=?, scout_hit_prob=?, "
                    "scouted_prop_id=? WHERE id=?",
                    [row[0], round(float(row[1]), 4), row[2], leg_id],
                )
                updated += 1

        if updated:
            conn.commit()
            print(f"[placement] backfill_scout_grades: updated {updated}/{len(rows)} legs")
    except Exception as exc:
        print(f"[placement] backfill_scout_grades error: {exc}")
    finally:
        conn.close()
    return updated


def rescore_composites_for_today() -> dict:
    """
    Recalculate composite_score on today's PENDING mock bets to reflect
    actual scout grades.  During generation, legs scored with NULL scout
    get _DEFAULT_SCOUT_NORM=0.50 in the scout component (weight 0.20).
    After backfill, the actual grade may be A(1.25), B(1.00), etc.

    Instead of recomputing the full formula (which includes directional,
    pattern, and projection adjustments already baked in), we add the
    scout delta:  Δ = (avg_actual_scout_norm − 0.50) × 0.20

    Returns {"updated": int, "details": [...]}.
    """
    import os
    import sqlite3 as _sql3
    _GRADE_NORM = {"A": 1.25, "B": 1.00, "C": 1.00, "D": 0.75}
    _DEFAULT = 0.50

    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
    conn = _sql3.connect(db_path, timeout=10)
    updated = 0
    details = []
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Get PENDING bets with composite_score for today
        bets = conn.execute(
            "SELECT id, composite_score, bet_info "
            "FROM mock_bets "
            "WHERE game_date = ? AND status = 'PENDING' AND composite_score IS NOT NULL",
            [today],
        ).fetchall()

        for bet_id, old_composite, bet_info in bets:
            # Get scout grades for this bet's legs
            legs = conn.execute(
                "SELECT scout_grade FROM mock_bet_legs WHERE mock_bet_id = ?",
                [bet_id],
            ).fetchall()
            if not legs:
                continue

            scout_norms = []
            for (grade,) in legs:
                scout_norms.append(_GRADE_NORM.get(grade, _DEFAULT) if grade else _DEFAULT)

            avg_actual = sum(scout_norms) / len(scout_norms)
            delta = (avg_actual - _DEFAULT) * 0.20

            if abs(delta) < 0.0001:
                continue  # no change

            new_composite = round(old_composite + delta, 6)
            conn.execute(
                "UPDATE mock_bets SET composite_score = ? WHERE id = ?",
                [new_composite, bet_id],
            )
            updated += 1
            details.append({
                "bet_id": bet_id,
                "old": round(old_composite, 6),
                "new": new_composite,
                "delta": round(delta, 6),
                "avg_scout_norm": round(avg_actual, 3),
                "info": (bet_info or "")[:60],
            })

        if updated:
            conn.commit()
            print(f"[placement] rescore_composites: updated {updated}/{len(bets)} bets")
    except Exception as exc:
        print(f"[placement] rescore_composites error: {exc}")
    finally:
        conn.close()
    return {"updated": updated, "details": details}


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
