"""
mock_bets.py — System 3: Mock Bet Training Loop.

Automatically generates paper bets from today's model recommendations,
tracks their outcomes, and converts results into model training signals.

Functions
---------
generate_mock_bets(db, stake, n_picks, max_legs)
    Pull today's top picks and write them as MockBet rows.

settle_mock_bets(db)
    Settle all PENDING MockBets whose fixture results are available.

get_mock_performance(db, days)
    Aggregate win rate, P&L, and trust metrics for settled mocks.

generate_mock_training_signal(db)
    Return feature rows from settled mocks suitable for model retraining.

generate_retroactive_mock_bets(db, lookback_days, n_per_day, job_id)
    Backfill historical mock bets from historical.db game outcomes.
    Uses the current model with accepted lookahead bias (weight=0.25).
"""
from __future__ import annotations

import os
import random
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, date as _date, timezone
from typing import Optional
from zoneinfo import ZoneInfo

_CT = ZoneInfo("America/Chicago")

from sqlalchemy.orm import Session

from database import MockBet, MockBetLeg, init_db
import recommender as rec
import auto_settle as asettler
import leg_quality as lq

# ─── Backfill job registry (module-level, thread-safe via GIL for dict ops) ───
_backfill_jobs: dict[str, dict] = {}
_backfill_lock = threading.Lock()

_HIST_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "historical.db"
)
_BETS_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "bets.db"
)

# ── Boost frequency (from 169-bet FanDuel history, 2026-01-26 → 2026-05-02) ──
# Historical profitable strategy: +25% boost → same CUSHION legs (pure EV add)
# Corrected strategy:             +30%/+50% → step-down to safer CUSHION lines
import random as _random


def _get_boost_for_bet() -> tuple[str | None, float]:
    """
    Randomly assign a boost type at historical FanDuel frequencies.

    Returns (promo_type, boost_pct):
        ('PROFIT_BOOST', 0.25)  — 24.3% of bets
        ('PROFIT_BOOST', 0.30)  — 16.6% of bets
        ('PROFIT_BOOST', 0.50)  — 13.0% of bets
        ('BONUS_BET',    0.0)   —  7.1% of bets  (free roll, $0 net stake risk)
        ('NO_SWEAT',     0.0)   —  4.1% of bets  (insurance, normal stake)
        (None,           0.0)   — 34.9% of bets  (no promo)
    """
    r = _random.random()
    if r < 0.243:   return ('PROFIT_BOOST', 0.25)
    elif r < 0.409: return ('PROFIT_BOOST', 0.30)
    elif r < 0.539: return ('PROFIT_BOOST', 0.50)
    elif r < 0.610: return ('BONUS_BET',    0.0)
    elif r < 0.651: return ('NO_SWEAT',     0.0)
    else:           return (None,           0.0)


def _is_sgp(pick: dict) -> bool:
    """Return True if all legs in the pick are from the same game (true SGP)."""
    legs = pick.get("legs", [])
    if len(legs) < 2:
        return False
    # Use fixture_id if present; fall back to game label
    ids = {leg.get("fixture_id") or leg.get("game") or "" for leg in legs}
    ids.discard("")
    return len(ids) == 1


def _is_single_sport(pick: dict) -> bool:
    """Return True if all legs in the pick cover the same sport (even if different games/leagues).

    Uses parlay_builder._normalize_sport so that EPL + Bundesliga + Ligue 1 all
    collapse to 'Soccer' and are correctly treated as same-sport for +50% Route B.
    """
    legs = pick.get("legs", [])
    if len(legs) < 2:
        return True  # single-leg is vacuously single-sport
    try:
        from parlay_builder import _normalize_sport as _pb_norm
        sports = {_pb_norm(leg.get("sport") or "") for leg in legs}
    except Exception:
        sports = {leg.get("sport") or "" for leg in legs}
    sports.discard("")
    return len(sports) <= 1


def _check_boost_eligible(
    boost_pct: float,
    n_legs: int,
    combined_odds: float,
    is_sgp: bool,
    is_single_sport: bool = False,
) -> tuple[bool, str]:
    """
    Check FanDuel boost eligibility rules.

    +25% PROFIT_BOOST:
        Combined odds >= 1.50 (-200). Works cross-sport or SGP.

    +30% PROFIT_BOOST:
        Combined odds >= 1.50 OR 3+ legs. Cross-sport ok.

    +50% PROFIT_BOOST:
        Route A: SGP 3+ legs (same-game, any sport).
        Route B: same-sport cross-game at combined odds >= 2.00 (+100).
                 Cross-sport multi-leg does NOT qualify for +50%, only +30% max.

    BONUS_BET / NO_SWEAT / None: always eligible (no odds gate).

    Returns (eligible: bool, reason: str)
    """
    if boost_pct == 0.25:
        ok = combined_odds >= 1.50
        return ok, ("✅ odds >= -200" if ok else "❌ needs odds >= -200 (1.50)")

    if boost_pct == 0.30:
        if combined_odds >= 1.50:
            return True, "✅ odds >= -200"
        if n_legs >= 3:
            return True, "✅ 3+ legs"
        return False, "❌ needs odds >= -200 or 3+ legs"

    if boost_pct == 0.50:
        # Route A: SGP 3+ legs (any sport)
        if is_sgp and n_legs >= 3:
            return True, "✅ Route A: SGP 3+ legs"
        # Route B: same-sport cross-game at +100 minimum
        if is_single_sport and not is_sgp and combined_odds >= 2.00:
            return True, "✅ Route B: same-sport +100"
        # Diagnose why it failed
        if is_sgp:
            return False, "❌ Route A: SGP needs 3+ legs"
        if not is_single_sport:
            return False, "❌ cross-sport: +50% only via SGP or same-sport multi"
        return False, "❌ Route B: same-sport needs odds >= +100 (2.00)"

    # BONUS_BET / NO_SWEAT / no boost — always eligible
    return True, "✅ no gate"


_ROUTE_B_MIN_LEG_WP = 60.0  # every leg must clear this to earn +50% Route B

def _route_b_wp_gate(pick: dict) -> tuple[bool, str]:
    """WP quality gate for Route B +50% boost.

    All legs must have individual win_prob >= _ROUTE_B_MIN_LEG_WP.
    Legs that earn high combined odds purely by stacking underdogs (~50% WP each)
    should not receive the premium boost tier.

    Returns (passes: bool, reason: str)
    """
    legs = pick.get("legs") or []
    if not legs:
        return True, "✅ no legs to check"
    leg_wps = [float(l.get("win_prob") or 50.0) for l in legs]
    min_wp = min(leg_wps)
    if min_wp < _ROUTE_B_MIN_LEG_WP:
        worst_idx = leg_wps.index(min_wp)
        worst_leg = legs[worst_idx].get("description") or legs[worst_idx].get("team") or f"leg {worst_idx+1}"
        return False, (
            f"❌ Route B WP gate: '{worst_leg}' WP={min_wp:.1f}% < {_ROUTE_B_MIN_LEG_WP:.0f}% required"
        )
    return True, f"✅ Route B WP gate: min leg WP={min_wp:.1f}%"


def _boost_ev(
    stake: float,
    decimal_odds: float,
    win_prob: float,  # 0-100 scale
    boost_pct: float,
) -> dict:
    """
    Compute base EV, boosted EV, and the EV lift from a profit boost.

    profit boost applies to profit only (not stake return):
        boosted_profit = base_profit * (1 + boost_pct)
        boosted_ev = win_prob * boosted_profit - (1 - win_prob) * stake
    """
    wp = win_prob / 100.0
    base_profit = stake * (decimal_odds - 1.0)
    base_ev = round(wp * base_profit - (1.0 - wp) * stake, 2)

    if boost_pct > 0:
        boosted_profit = base_profit * (1.0 + boost_pct)
        boosted_ev = round(wp * boosted_profit - (1.0 - wp) * stake, 2)
    else:
        boosted_ev = base_ev

    ev_lift = round(boosted_ev - base_ev, 2)
    return {
        "base_ev":    base_ev,
        "boosted_ev": boosted_ev,
        "ev_lift":    ev_lift,
    }


def _ensure_signal_columns() -> None:
    """
    Add signal + CLV columns to mock_bets / mock_bet_legs if the live schema
    predates these fields.  Safe to call on every import.
    """
    migrations = [
        ("mock_bets",      "predicted_odds",     "ALTER TABLE mock_bets ADD COLUMN predicted_odds INTEGER"),
        ("mock_bet_legs",  "predicted_win_prob",  "ALTER TABLE mock_bet_legs ADD COLUMN predicted_win_prob REAL"),
        ("mock_bet_legs",  "predicted_edge_pp",   "ALTER TABLE mock_bet_legs ADD COLUMN predicted_edge_pp REAL"),
        # CLV columns
        ("mock_bet_legs",  "open_odds",           "ALTER TABLE mock_bet_legs ADD COLUMN open_odds INTEGER"),
        ("mock_bet_legs",  "close_odds",          "ALTER TABLE mock_bet_legs ADD COLUMN close_odds INTEGER"),
        ("mock_bet_legs",  "clv_cents",           "ALTER TABLE mock_bet_legs ADD COLUMN clv_cents INTEGER"),
        ("mock_bet_legs",  "clv_available",       "ALTER TABLE mock_bet_legs ADD COLUMN clv_available INTEGER"),
        # ALE tracking columns
        ("mock_bet_legs",  "ale_considered",      "ALTER TABLE mock_bet_legs ADD COLUMN ale_considered BOOLEAN DEFAULT FALSE"),
        ("mock_bet_legs",  "ale_naive_pick",      "ALTER TABLE mock_bet_legs ADD COLUMN ale_naive_pick TEXT"),
        ("mock_bet_legs",  "ale_switched",        "ALTER TABLE mock_bet_legs ADD COLUMN ale_switched BOOLEAN DEFAULT FALSE"),
        ("mock_bet_legs",  "ale_los_improvement", "ALTER TABLE mock_bet_legs ADD COLUMN ale_los_improvement REAL"),
        # Line quality columns
        ("mock_bet_legs",  "main_market_line",   "ALTER TABLE mock_bet_legs ADD COLUMN main_market_line REAL"),
        ("mock_bet_legs",  "main_market_result", "ALTER TABLE mock_bet_legs ADD COLUMN main_market_result TEXT"),
        ("mock_bet_legs",  "direction_correct",  "ALTER TABLE mock_bet_legs ADD COLUMN direction_correct INTEGER"),
        ("mock_bet_legs",  "optimal_line",       "ALTER TABLE mock_bet_legs ADD COLUMN optimal_line REAL"),
        ("mock_bet_legs",  "line_delta",         "ALTER TABLE mock_bet_legs ADD COLUMN line_delta REAL"),
        ("mock_bet_legs",  "ab_alt_line",        "ALTER TABLE mock_bet_legs ADD COLUMN ab_alt_line REAL"),
        ("mock_bet_legs",  "ab_alt_result",      "ALTER TABLE mock_bet_legs ADD COLUMN ab_alt_result TEXT"),
        ("mock_bet_legs",  "ab_alt_odds",        "ALTER TABLE mock_bet_legs ADD COLUMN ab_alt_odds REAL"),
        ("mock_bet_legs",  "ab_alt_ev",          "ALTER TABLE mock_bet_legs ADD COLUMN ab_alt_ev REAL"),
        # Qualification tier (Item 1: CUSHION 4-tier filter)
        ("mock_bet_legs",  "qualification_tier", "ALTER TABLE mock_bet_legs ADD COLUMN qualification_tier TEXT"),
    ]
    try:
        con = sqlite3.connect(_BETS_DB_PATH)
        cur = con.cursor()
        for tbl, col, stmt in migrations:
            cur.execute(f"PRAGMA table_info({tbl})")
            existing = {row[1] for row in cur.fetchall()}
            if col not in existing:
                cur.execute(stmt)
                print(f"[mock_bets] migration: added {tbl}.{col}")
        con.commit()
        con.close()
    except Exception as _me:
        print(f"[mock_bets] migration warning: {_me}")

    # ── Normalize any T-separator datetimes (caused SQLAlchemy ORM crash) ────
    # SQLAlchemy's DateTime mapper for SQLite expects space separator, not 'T'.
    # One row was stored with isoformat() T-separator; normalize idempotently.
    try:
        con2 = sqlite3.connect(_BETS_DB_PATH)
        for col in ("settled_at", "generated_at"):
            con2.execute(
                f"UPDATE mock_bets SET {col} = replace({col}, 'T', ' ') "
                f"WHERE {col} LIKE '%T%'"
            )
        con2.commit()
        con2.close()
    except Exception:
        pass


_ensure_signal_columns()


# ─── CLV helpers ──────────────────────────────────────────────────────────────

def _decimal_to_american(price: float) -> int:
    """Convert decimal odds to American format."""
    if price >= 2.0:
        return round((price - 1) * 100)
    elif price > 1.0:
        return round(-100 / (price - 1))
    return 0


def _parse_outcome_from_desc(description: str, market_type: str) -> tuple[str, str]:
    """
    Return (market_key, outcome_name) suitable for matching line_snapshots.
    Examples:
      "Dallas Stars -1.5 (Spread)", "spread" → ("spreads", "Dallas Stars")
      "Over 220.5",                  "total"  → ("totals",  "Over")
      "New York Rangers ML",         "h2h"    → ("h2h",     "New York Rangers")
    """
    mkt_map = {
        "h2h": "h2h",       "moneyline": "h2h",  "ml": "h2h",
        "spread": "spreads", "spreads": "spreads",
        "total": "totals",   "totals": "totals",
    }
    mkt_key = mkt_map.get((market_type or "").lower().strip(), "")

    desc = (description or "").strip()
    lo = desc.lower()
    if lo.startswith("over"):
        return mkt_key or "totals", "Over"
    if lo.startswith("under"):
        return mkt_key or "totals", "Under"

    # Strip trailing " -1.5 (Spread)", " +2.5", " ML", " -110"
    import re as _re
    m = _re.match(r'^(.*?)\s+[+-]?\d', desc)
    if m:
        outcome = m.group(1).strip()
    else:
        outcome = desc.split(" (")[0].split(" ML")[0].strip()

    return mkt_key or "h2h", outcome


def _lookup_snap_odds(
    fixture_id: str,
    description: str,
    market_type: str,
    hist_db: str,
    before_dt: str | None = None,   # ISO8601; if None → earliest snapshot
) -> int | None:
    """
    Look up American odds from line_snapshots for a given leg.

    before_dt = None → earliest snapshot (open odds at generation)
    before_dt = ISO  → latest snapshot before that time (close odds)
    """
    if not fixture_id:
        return None
    mkt_key, outcome = _parse_outcome_from_desc(description, market_type)
    try:
        con = sqlite3.connect(hist_db)
        cur = con.cursor()
        if before_dt:
            order = "DESC"
            time_clause = "AND captured_at <= ?"
            params = [fixture_id, f"%{outcome}%", before_dt]
        else:
            order = "ASC"
            time_clause = ""
            params = [fixture_id, f"%{outcome}%"]

        if mkt_key:
            cur.execute(f"""
                SELECT price FROM line_snapshots
                WHERE event_id = ? AND market_key = ? AND outcome_name LIKE ?
                {time_clause}
                ORDER BY captured_at {order} LIMIT 1
            """, ([fixture_id, mkt_key, f"%{outcome}%"] + ([before_dt] if before_dt else [])))
        else:
            cur.execute(f"""
                SELECT price FROM line_snapshots
                WHERE event_id = ? AND outcome_name LIKE ?
                {time_clause}
                ORDER BY captured_at {order} LIMIT 1
            """, params)

        row = cur.fetchone()
        con.close()
        if row and row[0]:
            return _decimal_to_american(float(row[0]))
    except Exception as _ce:
        pass
    return None


# ─── Line quality helpers ─────────────────────────────────────────────────────

def _get_main_market_line(
    fixture_id: str | None,
    market: str,          # 'total' or 'spread'
    team: str | None,     # team name for spreads (used for fuzzy match)
    hist_db: str,
) -> float | None:
    """
    Return the standard (main) market line for a fixture.

    For totals : the Over/Under number (e.g. 8.5).
    For spreads: the spread for `team` (e.g. -1.5 or +1.5).

    Tries alt_lines (is_main_market=1) first, falls back to line_snapshots.
    Returns None when no data is available.
    """
    if not fixture_id:
        return None
    try:
        con = sqlite3.connect(hist_db)
        row = None
        if market == "total":
            row = con.execute(
                """SELECT line FROM alt_lines
                   WHERE event_id = ? AND market_key = 'totals' AND is_main_market = 1
                   ORDER BY fetched_at DESC LIMIT 1""",
                (fixture_id,),
            ).fetchone()
            if not row:
                row = con.execute(
                    """SELECT point FROM line_snapshots
                       WHERE event_id = ? AND market_key = 'totals' AND outcome_name = 'Over'
                       ORDER BY captured_at DESC LIMIT 1""",
                    (fixture_id,),
                ).fetchone()
        elif market == "spread" and team:
            slug = team[:12]
            row = con.execute(
                """SELECT line FROM alt_lines
                   WHERE event_id = ? AND market_key = 'spreads' AND is_main_market = 1
                     AND team LIKE ?
                   ORDER BY fetched_at DESC LIMIT 1""",
                (fixture_id, f"%{slug}%"),
            ).fetchone()
            if not row:
                row = con.execute(
                    """SELECT point FROM line_snapshots
                       WHERE event_id = ? AND market_key = 'spreads'
                         AND outcome_name LIKE ?
                       ORDER BY captured_at DESC LIMIT 1""",
                    (fixture_id, f"%{slug}%"),
                ).fetchone()
        con.close()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _get_alt_line_odds(
    fixture_id: str | None,
    market: str,       # 'total' or 'spread'
    direction: str,    # 'over'/'under' for totals; team name for spreads
    line: float,
    hist_db: str,
) -> float | None:
    """
    Return decimal odds from alt_lines for a specific (fixture, market, direction, line).
    Returns None when not found.
    """
    if not fixture_id:
        return None
    try:
        con = sqlite3.connect(hist_db)
        row = None
        if market == "total":
            for mkt_key in ("alternate_totals", "totals"):
                row = con.execute(
                    """SELECT odds FROM alt_lines
                       WHERE event_id = ? AND market_key = ?
                         AND over_under = ? AND ABS(line - ?) < 0.01
                       ORDER BY fetched_at DESC LIMIT 1""",
                    (fixture_id, mkt_key, direction.lower(), line),
                ).fetchone()
                if row:
                    break
        elif market == "spread":
            slug = direction[:12]
            for mkt_key in ("alternate_spreads", "spreads"):
                row = con.execute(
                    """SELECT odds FROM alt_lines
                       WHERE event_id = ? AND market_key = ?
                         AND team LIKE ? AND ABS(line - ?) < 0.01
                       ORDER BY fetched_at DESC LIMIT 1""",
                    (fixture_id, mkt_key, f"%{slug}%", line),
                ).fetchone()
                if row:
                    break
        con.close()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _compute_line_quality(leg, res: dict, hist_db: str) -> dict:
    """
    Compute two-dimensional line quality metrics for a settled leg.

    Requires resolved home_score / away_score in `res` (from _settle_mock_leg).
    Returns a dict with all nine line-quality fields; fields are None for
    markets where the concept doesn't apply (moneyline) or data is missing.
    """
    import re as _re

    _empty: dict = {
        "main_market_line": None, "main_market_result": None,
        "direction_correct": None, "optimal_line": None,
        "line_delta": None, "ab_alt_line": None,
        "ab_alt_result": None, "ab_alt_odds": None, "ab_alt_ev": None,
    }

    home_score = res.get("home_score")
    away_score = res.get("away_score")
    if home_score is None or away_score is None:
        return _empty

    desc       = (leg.description or "").strip()
    desc_lower = desc.lower()
    market     = (leg.market_type or "").lower()

    is_spread = (
        any(kw in market for kw in ("spread", "run_line", "puck_line")) or
        any(kw in desc_lower for kw in ("alt spread", "run line", "puck line", " spread)"))
    )
    is_total = (
        "total" in market or "totals" in market or
        _re.search(r"\b(over|under)\b", desc_lower) is not None
    )

    # ── Moneyline — direction only, no line precision ─────────────────────────
    if not is_spread and not is_total:
        outcome = res.get("outcome")
        return {
            **_empty,
            "direction_correct": (
                1 if outcome == "WIN" else (0 if outcome == "LOSS" else None)
            ),
        }

    # ── Totals ────────────────────────────────────────────────────────────────
    if is_total:
        m = _re.search(r"(over|under)\s+([\d.]+)", desc_lower)
        if not m:
            return _empty
        our_pick = m.group(1).lower()   # 'over' or 'under'
        our_line = float(m.group(2))
        actual_total = float(home_score) + float(away_score)

        main_line = _get_main_market_line(leg.fixture_id, "total", None, hist_db)
        if main_line is not None:
            if abs(actual_total - main_line) < 0.01:
                main_result = "PUSH"
            elif actual_total > main_line:
                main_result = "OVER"
            else:
                main_result = "UNDER"
            direction_correct = 1 if (
                (our_pick == "over"  and main_result == "OVER")  or
                (our_pick == "under" and main_result == "UNDER")
            ) else 0
        else:
            main_result = None
            direction_correct = None

        # Optimal: best line in the winning direction
        if our_pick == "over":
            optimal_line = actual_total - 0.5
        else:
            optimal_line = actual_total + 0.5

        line_delta = round(our_line - optimal_line, 2)

        # A/B: one step closer to main market
        ab_line = (our_line - 1.0) if our_pick == "over" else (our_line + 1.0)
        if abs(actual_total - ab_line) < 0.01:
            ab_result = "PUSH"
        elif our_pick == "over":
            ab_result = "WIN" if actual_total > ab_line else "LOSS"
        else:
            ab_result = "WIN" if actual_total < ab_line else "LOSS"

        ab_odds = _get_alt_line_odds(leg.fixture_id, "total", our_pick, ab_line, hist_db)
        ab_ev   = None  # populated in ALE calibration pass (step 7)

        return {
            "main_market_line":   main_line,
            "main_market_result": main_result,
            "direction_correct":  direction_correct,
            "optimal_line":       round(optimal_line, 2),
            "line_delta":         line_delta,
            "ab_alt_line":        ab_line,
            "ab_alt_result":      ab_result,
            "ab_alt_odds":        ab_odds,
            "ab_alt_ev":          ab_ev,
        }

    # ── Spreads ───────────────────────────────────────────────────────────────
    if is_spread:
        m = _re.search(r"([+-]?\d+\.?\d*)\s*\(", desc)
        if not m:
            return _empty
        our_line    = float(m.group(1))
        # team_margin already computed in _settle_mock_leg as team_score - opp_score
        team_margin = res.get("margin")
        if team_margin is None:
            return _empty
        team_margin = float(team_margin)

        # Extract team name for DB lookup
        pick_raw = _re.split(r"[+-]\d", desc)[0].strip()

        main_spread = _get_main_market_line(leg.fixture_id, "spread", pick_raw, hist_db)
        if main_spread is not None:
            main_adj     = team_margin + main_spread
            main_covered = main_adj > 0
            main_result  = "COVERED" if main_covered else "NOT_COVERED"
            direction_correct = 1 if main_covered else 0
        else:
            main_spread   = None
            main_result   = None
            direction_correct = None

        # Optimal: tightest line still winning in our direction
        if our_line < 0:
            optimal_line = -(team_margin - 0.5)   # favorite: tightest give-points line we'd cover
        else:
            optimal_line = team_margin + 0.5       # underdog: tightest get-points line we'd cover

        line_delta = round(our_line - optimal_line, 2)

        # A/B: one step toward main market
        ab_line = (our_line + 1.0) if our_line < 0 else (our_line - 1.0)
        ab_adj  = team_margin + ab_line
        ab_result = "WIN" if ab_adj > 0 else ("PUSH" if abs(ab_adj) < 0.01 else "LOSS")

        ab_odds = _get_alt_line_odds(leg.fixture_id, "spread", pick_raw, ab_line, hist_db)

        return {
            "main_market_line":   main_spread,
            "main_market_result": main_result,
            "direction_correct":  direction_correct,
            "optimal_line":       round(optimal_line, 2),
            "line_delta":         line_delta,
            "ab_alt_line":        ab_line,
            "ab_alt_result":      ab_result,
            "ab_alt_odds":        ab_odds,
            "ab_alt_ev":          None,
        }

    return _empty


# ─── Material-change detection (used by afternoon_only + require_change) ──────

# Minimum American-odds shift (in points) to count as a line move.
_LINE_MOVE_THRESHOLD_AM = 10   # e.g. -110 → -120 = 10 pts = material

def _detect_material_changes(
    fixture_ids: set[str],
    morning_cutoff_utc: str,              # ISO8601 — compare snapshots from here onward
    fixture_sport_map: dict[str, str] | None = None,  # {fixture_id: sport} for MLB-only checks
    bets_db:   str = _BETS_DB_PATH,
    hist_db:   str = _HIST_DB_PATH,
) -> dict[str, list[str]]:
    """
    For each fixture_id in fixture_ids, return a list of change reasons.
    A fixture with an empty list has had NO material changes since morning.

    Checks (in order):
      a) Line moved > _LINE_MOVE_THRESHOLD_AM American-odds points since morning
         snapshot (sharp money moved it)
      b) MLB confirmed pitcher now in pitcher_game_logs (wasn't at morning cutoff)
      c) Injury flag changed since morning (game_injury_flags star_out > 0 now
         but wasn't populated before, or star count increased)
      d) New alt lines posted since morning cutoff

    morning_cutoff_utc is the ISO timestamp of when the morning batch ran
    (stored as the generated_at of the first morning mock bet, or 8:00 AM UTC).
    """
    if not fixture_ids:
        return {}

    result: dict[str, list[str]] = {fid: [] for fid in fixture_ids}

    # ── (a) Line movement in line_snapshots ─────────────────────────────────
    # Compare the first snapshot price (open) vs latest snapshot price (now)
    # for each market/outcome combination.  A move ≥ threshold = sharp action.
    try:
        con_h = sqlite3.connect(hist_db)
        cur_h = con_h.cursor()

        for fid in fixture_ids:
            # Get all distinct market/outcome combos that have >= 2 snapshots
            cur_h.execute("""
                SELECT DISTINCT market_key, outcome_name
                FROM line_snapshots
                WHERE event_id = ?
            """, (fid,))
            combos = cur_h.fetchall()

            for combo in combos:
                mkt, out = combo[0], combo[1]
                # First snapshot AT OR AFTER morning cutoff (the "open" for today)
                cur_h.execute("""
                    SELECT price FROM line_snapshots
                    WHERE event_id = ? AND market_key = ? AND outcome_name = ?
                      AND captured_at >= ?
                    ORDER BY captured_at ASC LIMIT 1
                """, (fid, mkt, out, morning_cutoff_utc))
                first_row = cur_h.fetchone()
                # Latest snapshot (the current line)
                cur_h.execute("""
                    SELECT price FROM line_snapshots
                    WHERE event_id = ? AND market_key = ? AND outcome_name = ?
                    ORDER BY captured_at DESC LIMIT 1
                """, (fid, mkt, out))
                last_row = cur_h.fetchone()

                if not first_row or not last_row:
                    continue
                if first_row[0] == last_row[0]:
                    continue   # price unchanged

                try:
                    open_am   = _decimal_to_american(float(first_row[0]))
                    latest_am = _decimal_to_american(float(last_row[0]))
                    if abs(latest_am - open_am) >= _LINE_MOVE_THRESHOLD_AM:
                        result[fid].append(
                            f"line_move:{mkt}/{out} {open_am:+d}→{latest_am:+d}"
                        )
                        break   # one confirmed move per fixture is sufficient
                except Exception:
                    pass

        con_h.close()
    except Exception as _lme:
        print(f"[pm-change] line_move check failed: {_lme}")

    # ── (b) MLB confirmed pitcher ─────────────────────────────────────────────
    # Only applies to MLB fixtures (identified via fixture_sport_map).
    # Checks if pitcher_game_logs has entries for today added AFTER morning run.
    _mlb_fids = set()
    if fixture_sport_map:
        _mlb_fids = {fid for fid, sp in fixture_sport_map.items()
                     if "MLB" in (sp or "").upper()}
    if _mlb_fids:
        try:
            con_h2 = sqlite3.connect(hist_db)
            today_str = morning_cutoff_utc[:10]
            cur_h2 = con_h2.cursor()
            cur_h2.execute("""
                SELECT DISTINCT game_date FROM pitcher_game_logs
                WHERE game_date = ? AND game_date IS NOT NULL
            """, (today_str,))
            pitcher_dates = {r[0] for r in cur_h2.fetchall()}
            con_h2.close()
            if today_str in pitcher_dates:
                for fid in _mlb_fids:
                    if "pitcher_confirmed" not in " ".join(result.get(fid, [])):
                        result[fid].append("pitcher_confirmed:today")
        except Exception as _pe:
            pass

    # ── (c) Injury flags — any star_out or stars_missing > 0 ─────────────────
    try:
        con_h3 = sqlite3.connect(hist_db)
        cur_h3 = con_h3.cursor()
        cur_h3.execute("""
            SELECT game_id,
                   home_star_out + away_star_out +
                   home_stars_missing + away_stars_missing AS total_flags
            FROM game_injury_flags
            WHERE (home_star_out + away_star_out +
                   home_stars_missing + away_stars_missing) > 0
        """)
        flagged = {r[0]: r[1] for r in cur_h3.fetchall()}
        con_h3.close()
        for fid in fixture_ids:
            if fid in flagged:
                result[fid].append(f"injury_flag:{flagged[fid]}_stars_affected")
    except Exception:
        pass

    # ── (d) New alt lines posted since morning cutoff ─────────────────────────
    try:
        con_h4 = sqlite3.connect(hist_db)
        cur_h4 = con_h4.cursor()
        for fid in fixture_ids:
            cur_h4.execute("""
                SELECT COUNT(*) FROM alt_lines
                WHERE event_id = ? AND fetched_at >= ?
            """, (fid, morning_cutoff_utc))
            n_new = cur_h4.fetchone()[0]
            if n_new > 0:
                result[fid].append(f"new_alt_lines:{n_new}")
        con_h4.close()
    except Exception:
        pass

    return result


# Sports supported for retroactive scoring
_RETRO_SPORTS = ("NHL", "MLB")

# Confidence threshold: only score legs where model prob deviates from 0.5
_RETRO_CONF_THRESHOLD = 0.52   # lower than live threshold — more legs, less noise

# Known AUC values by sport for model_auc population
_AUC_BY_SPORT: dict[str, float] = {
    "NHL": 0.7544,
    "MLB": 0.6360,
}
_AUC_COMBINED = 0.5972

# Confidence bucketing thresholds — applied to avg PER-LEG probability
# (not parlay product, which is always <30% for 3-leg parlays)
# HIGH:    avg_leg_conf >= 0.62  — meaningful model edge
# MEDIUM:  avg_leg_conf 0.55-0.62 — moderate edge
# LOW:     avg_leg_conf < 0.55   — near-coin-flip legs
_CONF_HIGH   = 0.62
_CONF_MEDIUM = 0.55

# Soccer sport labels — used for draw-handling in ML resolution.
_SOCCER_SPORTS: frozenset[str] = frozenset({
    "EPL", "La Liga", "MLS", "UCL", "Bundesliga", "Ligue 1", "Serie A",
    "Europa League", "Soccer",
})

# Sports that have working settlement paths (OddsAPI scores endpoint or historical.db).
# Excluded: MMA (no completed scores in API), NCAAF (offseason in spring).
# NFL included but naturally absent in spring — picks simply won't appear.
# MLS excluded: no sub-model, no CLV data, no historical accuracy in bet_legs.
# Re-enable when an MLS-specific model exists.
_SUPPORTED_MOCK_SPORTS: frozenset[str] = frozenset({
    "MLB", "NHL", "NBA", "NFL",
}) | (_SOCCER_SPORTS - frozenset({"MLS"}))

# Hybrid sport weights for mock bet generation.
# One pass per sport — each generated bet contains legs from a single sport only.
# NHL/MLB use ATS sub-models; Soccer uses soccer_total_v1/soccer_ml_v1; NBA uses combined_v1.
# Tuple: (label, sport_filter_keys, has_submodel)
_SPORT_PASSES: list[tuple[str, list[str], bool]] = [
    ("NHL",    ["nhl"],     True),   # nhl_ats_v1
    ("MLB",    ["mlb"],     True),   # mlb_ats_v1
    ("NBA",    ["nba"],     False),  # combined_v1
    ("Soccer", ["soccer"],  True),   # soccer_total_v1 + soccer_ml_v1 (MLS excluded)
]

# Win-probability entry bar by model type.
# MLB sub-model is isotonic-calibrated at 55% (well-characterised threshold).
# NHL uses the recommender's own threshold (60%) to prevent low-confidence
# puck-line picks — especially important during playoffs when games are close.
# Passing None lets generate_todays_picks() use its _SPORT_CONFIDENCE_THRESHOLDS.
# Combined-model sports require 58% (generic model → higher confidence needed).
_MIN_CONF_SUBMODEL  = 55.0   # MLB only
_MIN_CONF_NHL       = None   # defer to recommender's 60% (or playoff override)
_MIN_CONF_COMBINED  = 58.0


# ─── Forced generation (no positive-EV requirement) ───────────────────────────

def _build_forced_picks(
    db,
    sport_filter:   list[str],
    stake:          float = 10.0,
    n_picks:        int   = 5,
    max_legs:       int   = 2,
    min_lqs:        float = 55.0,
    min_confidence: float = 50.0,
) -> list[dict]:
    """
    Build parlay picks directly from scored legs WITHOUT requiring positive edge.

    Used as a fallback when generate_todays_picks() returns 0 picks for a sport
    (e.g. all ML model outputs are below the confidence threshold, or no pitcher
    data is available for MLB). Ensures mock bets are always generated for
    simulation purposes regardless of market conditions.

    Quality filters kept:
      - LQS >= min_lqs (default 55 — lower than the top-picks threshold of 62)
      - win_prob >= min_confidence (default 50% — basic model signal)
      - Game starts today in CT timezone
      - Sport in SUPPORTED_MOCK_SPORTS

    Returns a list of pick dicts with is_forced_generation=True.
    """
    import itertools
    from parlay_builder import get_available_legs, score_leg

    all_raw = get_available_legs(db, markets=["h2h", "spreads", "totals"])
    if not all_raw:
        return []

    # Score all legs
    scored = [score_leg(l) for l in all_raw]

    # Apply sport filter — same expansion as recommender.py
    if sport_filter:
        _SPORT_GROUP = {
            "nba":    ["NBA"],
            "mlb":    ["MLB"],
            "nfl":    ["NFL"],
            "nhl":    ["NHL"],
            "soccer": ["EPL", "La Liga", "MLS", "UCL", "Bundesliga",
                       "Serie A", "Ligue 1", "Soccer"],
        }
        allowed: set[str] = set()
        for f in sport_filter:
            key = f.strip().lower()
            allowed.update(_SPORT_GROUP.get(key, [f.strip(), f.strip().upper()]))
        scored = [l for l in scored if l.get("sport", "") in allowed]

    if not scored:
        return []

    # Today CT filter + basic confidence
    today_ct = datetime.now(_CT).date()

    def _today(game_time_iso: Optional[str]) -> bool:
        if not game_time_iso:
            return True
        try:
            from datetime import timezone as _tz
            dt = datetime.fromisoformat(game_time_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_tz.utc)
            return dt.astimezone(_CT).date() == today_ct
        except Exception:
            return True

    scored = [
        l for l in scored
        if _today(l.get("game_time"))
        and (min_confidence is None or l.get("win_prob", 0) >= min_confidence)
        and l.get("sport", "OTHER") in _SUPPORTED_MOCK_SPORTS
    ]

    if not scored:
        return []

    # Compute LQS for each leg (inline — no augmentation block needed here)
    for l in scored:
        if l.get("lqs") is not None:
            continue
        try:
            _is_home = l.get("pick") == l.get("home_team")
            _opp     = l.get("away_team") if _is_home else l.get("home_team")
            cand = {
                "market_type":      l.get("market_label") or l.get("market"),
                "sport":            l.get("sport"),
                "team_or_player":   l.get("pick"),
                "model_confidence": l.get("win_prob"),
                "model_used":       l.get("model_used"),
                "edge_pp":          l.get("edge"),
                "line":             l.get("point"),
                "is_home":          _is_home,
                "opponent":         _opp,
            }
            r = lq.compute_leg_quality_score(cand, db)
            l["lqs"] = r.get("lqs", 0.0)
        except Exception:
            l["lqs"] = 0.0

    # Filter by LQS — relax floor if nothing qualifies at min_lqs
    pool = [l for l in scored if (l.get("lqs") or 0) >= min_lqs]
    if not pool:
        # Take best 10 by LQS regardless of floor
        pool = sorted(scored, key=lambda x: -(x.get("lqs") or 0))[:10]

    # Sort by LQS desc, cap to top 15 legs for combo generation
    pool.sort(key=lambda x: -(x.get("lqs") or 0))
    pool = pool[:15]

    picks: list[dict] = []
    used_sigs: set = set()

    for combo in itertools.combinations(range(len(pool)), 2):
        legs = [pool[i] for i in combo]

        # No same-fixture parlays
        fids = [l.get("fixture_id") for l in legs]
        if len(fids) != len(set(fids)):
            continue

        # No same-team parlays
        teams = [l.get("pick", "").strip().lower() for l in legs]
        real  = [t for t in teams if t not in ("over", "under", "")]
        if len(real) != len(set(real)):
            continue

        combined_odds = legs[0]["odds"] * legs[1]["odds"]
        if combined_odds < 2.0:
            continue

        sig = frozenset(l["leg_id"] for l in legs)
        if sig in used_sigs:
            continue
        used_sigs.add(sig)

        combined_win = (legs[0]["win_prob"] / 100) * (legs[1]["win_prob"] / 100)
        avg_lqs      = round(sum(l.get("lqs", 0) for l in legs) / 2, 1)

        picks.append({
            "legs":                 legs,
            "combined_odds":        round(combined_odds, 3),
            "win_prob":             round(combined_win * 100, 2),
            "combined_win_prob":    round(combined_win * 100, 2),
            "_avg_lqs":             avg_lqs,
            "model_used":           legs[0].get("model_used"),
            "is_forced_generation": True,
        })

        if len(picks) >= n_picks * 3:  # over-generate, dedup later
            break

    # Sort: best LQS first
    picks.sort(key=lambda x: (-x["_avg_lqs"], -x["combined_win_prob"]))
    return picks[:n_picks]


# ─── Leg description formatter ────────────────────────────────────────────────

def _format_leg_desc(leg: dict) -> str:
    """
    Build a context-rich bet_info segment for one leg.

    Format: "{pick}{line_str} — {away} @ {home} ({market})"
    Example: "Under 1.5 — Nice @ Marseille (Total)"
             "Yankees -1.5 — Orioles @ Yankees (Spread)"
             "New York Yankees — Orioles @ Yankees (Moneyline)"

    Falls back to the bare description when no game context is available
    (preserves backward compatibility for retroactively scored legs).
    """
    pick   = leg.get("pick", "")
    point  = leg.get("point")
    market = (leg.get("market_label") or
              {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}.get(
                  leg.get("market", ""), "") or
              leg.get("market_type") or "")

    # Omit the point suffix for h2h (Moneyline) legs — pick already contains the team.
    # For totals the line is always positive — show bare number (no "+" prefix).
    # For spreads show sign: "+1.5" (underdog) or "-1.5" (favorite).
    if point is not None and leg.get("market") != "h2h":
        try:
            pt   = float(point)
            is_spread = "spread" in (leg.get("market") or "")
            if is_spread:
                if pt == 0:
                    point_str = " PK"
                else:
                    sign = "+" if pt > 0 else ""
                    point_str = f" {sign}{pt:g}"
            else:
                point_str = f" {pt:g}"
        except (TypeError, ValueError):
            point_str = f" {point}"
    else:
        point_str = ""

    home = leg.get("home_team", "") or ""
    away = leg.get("away_team", "") or ""
    if home and away:
        game_str = f"{away} @ {home}"
    elif leg.get("game"):
        game_str = str(leg["game"])
    else:
        game_str = ""

    game_part   = f" — {game_str}" if game_str else ""
    market_part = f" ({market})"  if market    else ""
    desc        = leg.get("description", "")

    result = f"{pick}{point_str}{game_part}{market_part}"
    # If we produced nothing useful, fall back to the bare description
    return result if result.strip("() ") else desc


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_mock_bets(
    db:              Session,
    stake:           float       = 10.0,
    n_picks:         int         = 40,
    max_legs:        int         = 4,
    source:          str         = "prospective",
    tier_b_picks:    list | None = None,
    afternoon_only:  bool        = False,
    require_change:  bool        = False,
) -> dict:
    """
    Pull today's top parlay recommendations and record them as paper bets.

    Runs one generate_todays_picks call per sport (NHL, MLB, NBA, Soccer) so
    every generated bet contains legs from a single sport only.  Budget is
    split evenly across sports; any remainder goes to the first passes.

    Sport passes run sequentially to avoid Python GIL contention — parallel
    CPU-bound threads were each taking ~270s instead of ~56s due to GIL thrash.

    Deduplicates: if a MockBet with the same bet_info was already generated
    today it is skipped.

    Returns:
        {generated: int, skipped_dup: int, run_id: str}
    """
    run_id = f"mock_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    # Today's already-generated info strings (dedup).
    # Always dedup across ALL sources — prevents PM runs from re-generating
    # identical picks (same legs, same odds) that the morning run already produced.
    # If a line genuinely moves between morning and PM, the PM run will produce
    # a different bet_info (different odds encoded) or different legs, so it will
    # pass the dedup check and be stored. Identical bet_info = no new information.
    # prospective_legacy bets are excluded: they were superseded by a fresh pass.
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    existing_infos: set[str] = {
        row[0] for row in db.query(MockBet.bet_info).filter(
            MockBet.generated_at >= today_start,
            MockBet.source != "prospective_legacy",
        ).all()
        if row[0]
    }

    all_picks: list[dict] = []
    n_sports   = len(_SPORT_PASSES)
    n_per_sport = max(1, n_picks // n_sports)

    # Pre-compute today's CT date once — used to drop cross-date picks from
    # the primary path before deciding whether to fall back to forced gen.
    _gen_today_ct = datetime.now(_CT).date()

    def _all_legs_today(pick: dict) -> bool:
        """Return True if every leg in the pick starts today (CT)."""
        for leg in pick.get("legs", []):
            gt = leg.get("game_time")
            if not gt:
                continue   # no game_time → don't filter
            try:
                dt = datetime.fromisoformat(gt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt.astimezone(_CT).date() != _gen_today_ct:
                    return False
            except Exception:
                pass
        return True

    # ── Sport-pass worker (runs in its own thread with its own DB session) ───────
    def _run_sport_pass(sport_label: str, sport_filter: list, has_submodel: bool) -> list[dict]:
        """Execute one sport pass and return tagged picks. Thread-safe (own session)."""
        import time as _time_mod
        _t_pass = _time_mod.time()
        from database import SessionLocal as _SL
        _thread_db = _SL()
        try:
            sport_picks: list[dict] = []
            _primary_count = 0
            _forced_count  = 0
            if sport_label == "NHL":
                _min_conf = _MIN_CONF_NHL
            elif has_submodel:
                _min_conf = _MIN_CONF_SUBMODEL
            else:
                _min_conf = _MIN_CONF_COMBINED

            _sport_promo_type, _sport_boost_pct = _get_boost_for_bet()
            _t_pre_gen = _time_mod.time()
            try:
                resp = rec.generate_todays_picks(
                    _thread_db,
                    n_picks         = max(n_per_sport * 6, 8),
                    stake           = stake,
                    max_legs        = max_legs,
                    min_legs        = 2,
                    min_odds        = 2.0,
                    sport_filter    = sport_filter,
                    min_confidence  = _min_conf,
                    depth_lqs_min   = 55.0,
                    boost_pct       = _sport_boost_pct,
                )
                raw_picks  = resp.get("picks", []) if isinstance(resp, dict) else []
                raw_power  = resp.get("power_picks", []) if isinstance(resp, dict) else []
                sport_picks    = [p for p in (raw_picks + raw_power) if _all_legs_today(p)]
                _primary_count = len(sport_picks)
                _cross_date    = len(raw_picks) - _primary_count
                if _cross_date:
                    print(f"[mock-gen] {sport_label}: dropped {_cross_date} cross-date primary picks")
            except Exception as _e:
                print(f"[mock-gen] {sport_label} primary exception: {_e}")

            if not sport_picks:
                try:
                    sport_picks = _build_forced_picks(
                        _thread_db,
                        sport_filter  = sport_filter,
                        stake         = stake,
                        n_picks       = n_per_sport * 2,
                        max_legs      = min(max_legs, 2),
                        min_lqs       = 55.0,
                        min_confidence= _min_conf,
                    )
                    _forced_count = len(sport_picks)
                except Exception as _e:
                    print(f"[mock-gen] {sport_label} forced exception: {_e}")
                    sport_picks = []

            _t_gen_done = _time_mod.time()
            print(f"[mock-gen] {sport_label}: primary={_primary_count} forced={_forced_count} min_conf={_min_conf} gen_time={_t_gen_done-_t_pre_gen:.1f}s total_time={_t_gen_done-_t_pass:.1f}s")

            # Tag sport pass
            for p in sport_picks:
                p.setdefault("_sport_pass", sport_label)

            # Boost assignment + eligibility tagging
            for _p in sport_picks:
                _p_odds  = float(_p.get("combined_odds") or _p.get("odds") or 1.0)
                _p_legs  = len(_p.get("legs") or [])
                _p_sgp   = _is_sgp(_p)
                _p_ss    = _is_single_sport(_p)
                _p_wp    = float(_p.get("combined_win_prob") or _p.get("win_prob") or 50.0)
                _assigned_promo = _sport_promo_type
                _assigned_boost = _sport_boost_pct
                if _assigned_promo == "PROFIT_BOOST" and _assigned_boost > 0:
                    _eligible, _reason = _check_boost_eligible(_assigned_boost, _p_legs, _p_odds, _p_sgp, _p_ss)
                    # WP quality gate: Route B +50% requires every leg WP >= 60%
                    if _eligible and _assigned_boost == 0.50 and "Route B" in _reason:
                        _wp_ok, _wp_reason = _route_b_wp_gate(_p)
                        if not _wp_ok:
                            _eligible = False
                            _reason   = _wp_reason
                    if not _eligible:
                        for _fallback_pct in (0.30, 0.25):
                            if _fallback_pct < _assigned_boost:
                                _fb_ok, _fb_reason = _check_boost_eligible(_fallback_pct, _p_legs, _p_odds, _p_sgp, _p_ss)
                                if _fb_ok:
                                    _assigned_boost = _fallback_pct
                                    _reason         = f"downgraded to +{int(_fallback_pct*100)}%: {_fb_reason}"
                                    _eligible       = True
                                    break
                        if not _eligible:
                            _assigned_promo = None
                            _assigned_boost = 0.0
                            _reason         = "no boost — ineligible for all tiers"
                    _p.setdefault("_boost_eligibility", _reason)
                _p.setdefault("_promo_type",     _assigned_promo)
                _p.setdefault("_promo_boost_pct", _assigned_boost)
                if _assigned_promo == "PROFIT_BOOST" and _assigned_boost > 0:
                    _bev = _boost_ev(stake, _p_odds, _p_wp, _assigned_boost)
                    _p.setdefault("_boost_ev", _bev)
                    _tier_eligibility: dict = {}
                    for _t in (0.25, 0.30, 0.50):
                        _t_ok, _t_reason = _check_boost_eligible(_t, _p_legs, _p_odds, _p_sgp, _p_ss)
                        _t_ev = _boost_ev(stake, _p_odds, _p_wp, _t) if _t_ok else None
                        _tier_eligibility[f"+{int(_t*100)}%"] = {"eligible": _t_ok, "reason": _t_reason, "ev": _t_ev}
                    _p.setdefault("_boost_tiers", _tier_eligibility)

            return sport_picks
        finally:
            _thread_db.close()

    # ── Run sport passes sequentially ────────────────────────────────────────────
    # Parallel threads competed for the Python GIL on CPU-bound work, making each
    # pass take ~270s instead of ~56s.  Sequential execution avoids that overhead;
    # total wall-clock time is the sum of each pass (~2-4 min) rather than 4× the
    # longest pass due to GIL thrashing.
    for _sl, _sf, _hs in _SPORT_PASSES:
        try:
            all_picks += _run_sport_pass(_sl, _sf, _hs)
        except Exception as _e:
            print(f"[mock-gen] {_sl} exception: {_e}")

    # ── Inject positive-EV Section B parlays from the cached picks page ────────
    # The sport-by-sport loop only pulls from generate_todays_picks() which uses
    # stricter thresholds.  Section B fallback parlays (assembled from positive-EV
    # legs below the confidence gate) are passed in via tier_b_picks so they also
    # get paper-traded.
    _tier_b_added = 0
    if tier_b_picks:
        for _p in tier_b_picks:
            if (_p.get("expected_profit") or 0) <= 0:
                continue   # positive EV only
            if not _all_legs_today(_p):
                continue   # today's games only
            n_legs = len(_p.get("legs") or [])
            if n_legs < 2:
                continue   # parlays only in this injection
            _p.setdefault("_sport_pass", "tier_b")
            all_picks.append(_p)
            _tier_b_added += 1
    print(f"[mock-gen] tier_b injected={_tier_b_added}")

    # ── Stamp boost on any pick that bypassed _run_sport_pass() ─────────────────
    # tier_b picks and other injected picks skip the per-sport-pass boost loop.
    # Give each unstamped pick its own independent boost draw so the DB's
    # promo_type column reflects realistic distribution across ALL pick sources,
    # not just the per-sport pass picks.
    _boost_stamped_extra = 0
    for _p in all_picks:
        if "_promo_type" not in _p:
            _p_odds = float(_p.get("combined_odds") or _p.get("odds") or 1.0)
            _p_legs = len(_p.get("legs") or [])
            _p_sgp  = _is_sgp(_p)
            _p_ss   = _is_single_sport(_p)
            _p_wp   = float(_p.get("combined_win_prob") or _p.get("win_prob") or 50.0)
            _bt, _bp = _get_boost_for_bet()
            _assigned_promo = _bt
            _assigned_boost = _bp
            if _assigned_promo == "PROFIT_BOOST" and _assigned_boost > 0:
                _el, _er = _check_boost_eligible(_assigned_boost, _p_legs, _p_odds, _p_sgp, _p_ss)
                # WP quality gate: Route B +50% requires every leg WP >= 60%
                if _el and _assigned_boost == 0.50 and "Route B" in _er:
                    _wp_ok, _wp_reason = _route_b_wp_gate(_p)
                    if not _wp_ok:
                        _el = False
                        _er = _wp_reason
                if not _el:
                    for _fb in (0.30, 0.25):
                        if _fb < _assigned_boost:
                            _fbok, _ = _check_boost_eligible(_fb, _p_legs, _p_odds, _p_sgp, _p_ss)
                            if _fbok:
                                _assigned_boost = _fb
                                _el = True
                                break
                    if not _el:
                        _assigned_promo = None
                        _assigned_boost = 0.0
            _p["_promo_type"]      = _assigned_promo
            _p["_promo_boost_pct"] = _assigned_boost
            if _assigned_promo == "PROFIT_BOOST" and _assigned_boost > 0:
                _p["_boost_ev"] = _boost_ev(stake, _p_odds, _p_wp, _assigned_boost)
            _boost_stamped_extra += 1
    if _boost_stamped_extra:
        print(f"[mock-gen] boost-stamped {_boost_stamped_extra} injected picks")

    print(f"[mock-gen] all_picks total={len(all_picks)}")

    if not all_picks:
        return {"generated": 0, "skipped_dup": 0, "run_id": run_id,
                "error": "No high-confidence picks across any sport today."}

    # ── Score every assembled pick with LQS ─────────────────────────────────
    # Threshold logic:
    #   Single-leg (straight bet): avg LQS >= 55
    #   Parlay (2+ legs): every individual leg must score >= 55
    #     Matches depth_lqs_min=55 already passed to the recommender so that
    #     CUSHION ML parlays (Pistons lqs=58.5, Cavaliers lqs=59.5) are not
    #     double-rejected here after already passing the recommender gate.
    _LQS_MIN_SINGLE     = 55.0  # straight bets
    _LQS_MIN_PARLAY_LEG = 55.0  # minimum per-leg for any parlay

    def _pick_lqs_scores(p: dict) -> list[float]:
        """Return per-leg LQS scores for a pick."""
        legs_p = p.get("legs", [])
        if not legs_p:
            return [50.0]
        scores = []
        for leg in legs_p:
            try:
                candidate = {
                    "market_type":      leg.get("market_type") or leg.get("market"),
                    "sport":            leg.get("sport"),
                    "team_or_player":   leg.get("pick") or leg.get("team_or_player"),
                    "model_confidence": leg.get("win_prob"),
                    "model_used":       p.get("model_used") or leg.get("model_used"),
                    "edge_pp":          leg.get("edge"),
                    "line":             leg.get("point"),
                }
                result = lq.compute_leg_quality_score(candidate, db)
                scores.append(result["lqs"])
            except Exception:
                scores.append(50.0)
        return scores

    scored_picks = []
    for p in all_picks:
        leg_scores = _pick_lqs_scores(p)
        avg = round(sum(leg_scores) / len(leg_scores), 1)
        p["_avg_lqs"] = avg
        n_legs = len(p.get("legs", []))

        if n_legs >= 2:
            # Parlay: every leg must clear the per-leg floor
            min_leg = min(leg_scores)
            passes  = min_leg >= _LQS_MIN_PARLAY_LEG
        else:
            # Straight bet: average (= single leg score) >= 55
            passes = avg >= _LQS_MIN_SINGLE
            if passes and avg < 60.0:
                p["_lqs_band"] = "55-60"
                if not p.get("_source_tag"):
                    p["_source_tag"] = "prospective_low_lqs"

        if passes:
            scored_picks.append(p)

    # If filter removed everything, use all picks (avoid empty generation)
    if not scored_picks:
        scored_picks = all_picks

    # Sort: highest avg_lqs first, then by original win_prob
    scored_picks.sort(key=lambda x: (-x.get("_avg_lqs", 50), -x.get("win_prob", 0)))

    # ── Drop picks containing any unsupported-sport leg ────────────────────
    # Guard against the recommender surfacing MMA/NCAAF fixtures even
    # when sport_filter is set — those legs cannot be settled.
    scored_picks = [
        p for p in scored_picks
        if all(
            l.get("sport", "OTHER") in _SUPPORTED_MOCK_SPORTS
            for l in p.get("legs", [])
            if l.get("sport")   # ignore legs with no sport tag
        )
    ]

    # ── Drop picks with any leg not starting today (CT) ──────────────────────
    # Prevents legs for tomorrow's games entering mock bets that will be
    # settled in tonight's scheduler cycle. game_time comes from the
    # recommender's get_available_legs() — it is the fixture commence_time ISO.
    # NOTE: cross-date picks are also pre-filtered in the sport passes loop above
    # so that forced generation fires correctly. This is the final safety gate.
    today_ct = datetime.now(_CT).date()

    # 3 PM CT cutoff used for afternoon_only filter
    _3pm_ct = datetime.now(_CT).replace(hour=15, minute=0, second=0, microsecond=0)
    _3pm_utc = _3pm_ct.astimezone(timezone.utc).replace(tzinfo=None)

    def _is_today_ct(game_time_iso: Optional[str]) -> bool:
        if not game_time_iso:
            return True   # unknown time → don't filter
        try:
            dt = datetime.fromisoformat(game_time_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(_CT).date() == today_ct
        except Exception:
            return True   # parse failure → don't filter

    def _is_after_3pm_ct(game_time_iso: Optional[str]) -> bool:
        """For afternoon_only: leg game must start at or after 3 PM CT today."""
        if not game_time_iso:
            return True   # no time → don't exclude
        try:
            dt = datetime.fromisoformat(game_time_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(_CT) >= _3pm_ct
        except Exception:
            return True

    scored_picks = [
        p for p in scored_picks
        if all(_is_today_ct(l.get("game_time")) for l in p.get("legs", []))
    ]

    # Afternoon-only filter: drop picks where any leg starts before 3 PM CT
    if afternoon_only:
        scored_picks = [
            p for p in scored_picks
            if all(_is_after_3pm_ct(l.get("game_time")) for l in p.get("legs", []))
        ]
        print(f"[mock-gen] afternoon_only=True → {len(scored_picks)} picks with evening games")

    # require_change filter: only keep picks where at least one leg's fixture has
    # had a material change since the morning run.
    if require_change and afternoon_only:
        # Determine morning cutoff: generated_at of the earliest morning mock bet today,
        # or fall back to 8:00 AM UTC if none found.
        _morning_cutoff = "08:00:00"
        try:
            _mc_row = db.execute(  # type: ignore[union-attr]
                "SELECT MIN(generated_at) FROM mock_bets "
                "WHERE DATE(generated_at) = DATE('now') AND source = 'prospective'"
            ).fetchone()
            if _mc_row and _mc_row[0]:
                _morning_cutoff = str(_mc_row[0])
        except Exception:
            pass
        if not _morning_cutoff or len(_morning_cutoff) < 10:
            _morning_cutoff = datetime.utcnow().strftime("%Y-%m-%d") + "T08:00:00Z"
        elif "T" not in _morning_cutoff and " " in _morning_cutoff:
            _morning_cutoff = _morning_cutoff.replace(" ", "T") + "Z"

        # Collect all fixture_ids + their sport from the candidate picks
        _all_fids: set[str] = set()
        _fid_sport: dict[str, str] = {}
        for _p in scored_picks:
            for _l in _p.get("legs", []):
                _fid = _l.get("fixture_id") or _l.get("game_id")
                if _fid:
                    _all_fids.add(_fid)
                    if _l.get("sport"):
                        _fid_sport[_fid] = _l["sport"]

        # Detect material changes per fixture
        _changes = _detect_material_changes(
            _all_fids,
            morning_cutoff_utc=_morning_cutoff,
            fixture_sport_map=_fid_sport,
        )
        _changed_fids = {fid for fid, reasons in _changes.items() if reasons}

        # Log what was found
        _total_changed = len(_changed_fids)
        if _changed_fids:
            for _fid in list(_changed_fids)[:5]:
                print(f"[PM] material change on {_fid}: {_changes[_fid]}")
        else:
            print(f"[PM] No material changes detected — skipping {len(scored_picks)} picks")

        # Keep only picks where at least one leg's fixture changed
        scored_picks = [
            p for p in scored_picks
            if any(
                (l.get("fixture_id") or l.get("game_id")) in _changed_fids
                for l in p.get("legs", [])
            )
        ]
        print(f"[mock-gen] require_change=True → {len(scored_picks)} picks on changed fixtures "
              f"({_total_changed} fixtures changed)")

    # Deduplicate by bet_info signature across both passes.
    # Also enforce leg-level dedup within this run: once a (team, market, point)
    # leg has been placed in any bet, it cannot appear in another bet this run.
    # This prevents the same leg from inflating simulation counts (e.g. Angels -1.5
    # appearing in 20 different parlays → 20 correlated data points for one game).
    # Key = (fixture_id, market, pick, point) — or description as fallback.
    seen_sigs: set[str] = set()
    seen_leg_keys: set[tuple] = set()
    standard_picks: list[dict] = []
    power_picks_sel: list[dict] = []
    _MAX_POWER_BETS = 10  # cap power (4+ leg) bets per run

    def _leg_key(l: dict) -> tuple:
        return (
            l.get("fixture_id") or "",
            l.get("market") or l.get("market_type") or "",
            (l.get("pick") or l.get("team") or "").strip().lower(),
            str(l.get("point") or ""),
        )

    for p in scored_picks:
        legs    = p.get("legs", [])
        sig     = " | ".join(l.get("description", "") for l in legs if l.get("description"))
        if not sig or sig in seen_sigs:
            continue
        # Reject if any leg in this pick is already used in this run
        leg_keys = [_leg_key(l) for l in legs]
        if any(k in seen_leg_keys for k in leg_keys):
            continue
        seen_sigs.add(sig)
        seen_leg_keys.update(leg_keys)
        n_legs = len(legs)
        if n_legs >= 4:
            if len(power_picks_sel) < _MAX_POWER_BETS:
                power_picks_sel.append(p)
        else:
            if len(standard_picks) < n_picks:
                standard_picks.append(p)
        if len(standard_picks) >= n_picks and len(power_picks_sel) >= _MAX_POWER_BETS:
            break
    picks = standard_picks + power_picks_sel

    # ── Mandatory final boost pass ────────────────────────────────────────────
    # Runs on every pick with DIRECT ASSIGNMENT (not setdefault) before writing
    # to DB.  This is the single authoritative boost assignment — earlier passes
    # (per-sport-pass setdefault, tier_b extra stamp) are now defence-in-depth
    # only.  Direct assignment prevents a cache-mutation regression where the
    # same pick dict is reused across runs: once _promo_type is a key (even None),
    # setdefault / "not in" checks silently skip re-stamping on subsequent runs.
    _EXPL_SRCS = set(EXPLORATION_SOURCES)
    for _fp in picks:
        # Exploration / forced-generation picks carry $0 stake; skip boost draw.
        if source in _EXPL_SRCS or _fp.get("is_forced_generation"):
            _fp["_promo_type"]      = None
            _fp["_promo_boost_pct"] = 0.0
            continue
        _fpt, _fbp = _get_boost_for_bet()
        _fp_odds   = float(_fp.get("combined_odds") or _fp.get("odds") or 1.0)
        _fp_legs   = len(_fp.get("legs") or [])
        _fp_sgp    = _is_sgp(_fp)
        _fp_ss     = _is_single_sport(_fp)
        _fp_wp     = float(_fp.get("combined_win_prob") or _fp.get("win_prob") or 50.0)
        if _fpt == "PROFIT_BOOST" and _fbp > 0:
            _fel, _fer = _check_boost_eligible(_fbp, _fp_legs, _fp_odds, _fp_sgp, _fp_ss)
            # WP quality gate: Route B +50% requires every leg WP >= 60%
            if _fel and _fbp == 0.50 and "Route B" in _fer:
                _wp_ok, _wp_reason = _route_b_wp_gate(_fp)
                if not _wp_ok:
                    _fel = False
                    _fer = _wp_reason
            if not _fel:
                for _ffb in (0.30, 0.25):
                    if _ffb < _fbp:
                        _fbok, _fber = _check_boost_eligible(_ffb, _fp_legs, _fp_odds, _fp_sgp, _fp_ss)
                        if _fbok:
                            _fbp = _ffb
                            _fer = f"downgraded to +{int(_ffb*100)}%: {_fber}"
                            _fel = True
                            break
                if not _fel:
                    _fpt = None
                    _fbp = 0.0
            if _fel:
                _fp["_boost_eligibility"] = _fer
        _fp["_promo_type"]      = _fpt
        _fp["_promo_boost_pct"] = _fbp
        if _fpt == "PROFIT_BOOST" and _fbp > 0:
            _fp["_boost_ev"] = _boost_ev(stake, _fp_odds, _fp_wp, _fbp)
    _n_boosted_final = sum(1 for _fp in picks if _fp.get("_promo_type"))
    print(f"[mock-gen] final boost pass: {_n_boosted_final}/{len(picks)} picks boosted")

    generated = 0
    skipped   = 0

    for pick in picks:
        legs     = pick.get("legs", [])
        bet_info = " | ".join(
            _format_leg_desc(l) for l in legs if l.get("description") or l.get("pick")
        )
        if not bet_info or bet_info in existing_infos:
            skipped += 1
            continue
        existing_infos.add(bet_info)

        sport_set = {l.get("sport", "") for l in legs if l.get("sport")}
        dom_sport = next(iter(sport_set), None)

        mock_id = str(uuid.uuid4())
        _is_power = len(legs) >= 4 or bool(pick.get("is_power_pick"))
        _notes = "is_power_pick=True" if _is_power else None
        mock = MockBet(
            id                 = mock_id,
            generation_run_id  = run_id,
            game_date          = datetime.now(_CT).strftime("%Y-%m-%d"),
            sport              = " | ".join(sport_set) if sport_set else None,
            bet_type           = "parlay" if len(legs) > 1 else "straight",
            odds               = pick.get("combined_odds") or pick.get("odds"),
            # forced_generation and exploration bets track outcomes for calibration
            # but contribute $0 to P&L — they are control/baseline bets, not real picks.
            amount             = 0.0 if (
                pick.get("is_forced_generation")
                or source in EXPLORATION_SOURCES
                or pick.get("_promo_type") == "BONUS_BET"
            ) else stake,
            legs               = len(legs),
            bet_info           = bet_info,
            predicted_win_prob = pick.get("combined_win_prob") or pick.get("model_win_prob"),
            predicted_ev       = pick.get("expected_profit") or pick.get("combined_ev"),
            predicted_odds     = pick.get("combined_am"),
            confidence         = pick.get("confidence"),
            model_used         = pick.get("model_used"),
            model_auc          = pick.get("model_auc"),
            status             = "PENDING",
            source             = "forced_generation" if pick.get("is_forced_generation") else source,
            avg_lqs            = pick.get("_avg_lqs"),
            promo_type         = pick.get("_promo_type"),
            promo_boost_pct    = pick.get("_promo_boost_pct"),
            boost_strategy     = pick.get("_boost_eligibility"),
            promo_ev_lift      = (pick.get("_boost_ev") or {}).get("ev_lift"),
            promo_boosted_odds = (
                round(
                    (pick.get("combined_odds") or pick.get("odds") or 1.0)
                    * (1.0 + (pick.get("_promo_boost_pct") or 0.0)),
                    4,
                )
                if pick.get("_promo_type") == "PROFIT_BOOST" and pick.get("_promo_boost_pct")
                else None
            ),
            notes              = _notes,
        )
        db.add(mock)

        for i, leg in enumerate(legs):
            _fid  = leg.get("fixture_id") or leg.get("game_id")
            _desc = leg.get("description", "")
            _mkt  = leg.get("market_type") or leg.get("market")
            _open = _lookup_snap_odds(_fid, _desc, _mkt, _HIST_DB_PATH)

            # Enrich leg with scout grade from scouted_props
            _scout_grade    = leg.get("scout_grade")
            _scout_hit_prob = leg.get("scout_hit_prob")
            _scout_prop_id  = leg.get("scouted_prop_id")
            if _scout_grade is None:
                try:
                    import placement as _plc
                    _enriched       = _plc.attach_scout_grade_to_leg(dict(leg), db)
                    _scout_grade    = _enriched.get("scout_grade")
                    _scout_hit_prob = _enriched.get("scout_hit_prob")
                    _scout_prop_id  = _enriched.get("scouted_prop_id")
                except Exception:
                    pass

            leg_obj = MockBetLeg(
                mock_bet_id        = mock_id,
                leg_index          = i,
                description        = _desc,
                market_type        = _mkt,
                sport              = leg.get("sport"),
                win_prob           = leg.get("win_prob"),
                ev                 = leg.get("ev"),
                grade              = leg.get("grade"),
                model_used         = leg.get("model_used"),
                fixture_id         = _fid,
                is_alt_line        = bool(leg.get("is_alt_line", False)),
                predicted_win_prob = leg.get("win_prob"),
                predicted_edge_pp  = leg.get("edge"),
                open_odds          = _open,
                ale_considered     = bool(leg.get("ale_considered", False)),
                ale_naive_pick     = leg.get("ale_naive_pick"),
                ale_switched       = bool(leg.get("ale_switched", False)),
                ale_los_improvement = leg.get("ale_los_improvement"),
                qualification_tier  = leg.get("qualification_tier"),
            )
            db.add(leg_obj)
            db.flush()  # get leg_obj.id so we can update scout columns

            # Write scout columns via raw SQL (columns added by safe_add_column, not in ORM)
            if _scout_grade is not None:
                try:
                    from sqlalchemy import text as _text
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET scout_grade=:g, scout_hit_prob=:hp, "
                        "scouted_prop_id=:pid WHERE id=:lid"
                    ), {"g": _scout_grade, "hp": _scout_hit_prob,
                        "pid": _scout_prop_id, "lid": leg_obj.id})
                except Exception:
                    pass
        generated += 1

    forced_count = sum(1 for p in picks if p.get("is_forced_generation"))
    db.commit()
    # Auto-apply active theses to newly generated legs
    auto_excl = _apply_active_theses(run_id, db)
    if auto_excl:
        print(f"[mock-gen] auto-thesis pass excluded {auto_excl} leg(s)")
    result: dict = {"generated": generated, "skipped_dup": skipped, "run_id": run_id}
    if forced_count > 0:
        result["forced_generation_count"] = forced_count
        result["forced_generation_note"] = (
            f"{forced_count} bet(s) generated from best-available legs "
            f"(no positive-EV opportunity found for those sports today)."
        )
    return result


# ─── Auto-thesis application ──────────────────────────────────────────────────

def _apply_active_theses(run_id: str, db) -> int:
    """
    After generation, scan all active theses and auto-exclude matching legs
    from this run with Mode B (recalculate) as the default.

    Matching logic per thesis:
      1. Team name match: leg description contains thesis.team (case-insensitive)
      2. Market block: if leg.market_type is in market_filters.block → exclude
      3. Alt spread exception: if market is alternate_spreads AND
         line >= market_filters.alt_spreads_min_line → DO NOT exclude (allowed)

    Does not double-exclude legs that are already excluded.
    Updates parent bet has_excluded_legs / exclusion_mode_summary / recalculated_odds.
    Returns total legs auto-excluded.
    """
    import json as _json
    try:
        from database import UserThesis as _UT
    except ImportError:
        return 0

    theses = db.query(_UT).filter(_UT.active == True).all()
    if not theses:
        return 0

    # Fetch all legs written in this run (via their parent bets' generation_run_id)
    run_legs = (
        db.query(MockBetLeg)
        .join(MockBet, MockBet.id == MockBetLeg.mock_bet_id)
        .filter(MockBet.generation_run_id == run_id,
                MockBet.source != "exploration")
        .all()
    )
    if not run_legs:
        return 0

    total_excluded = 0

    for thesis in theses:
        if not thesis.team:
            continue
        mf_raw = thesis.market_filters
        if not mf_raw:
            continue
        try:
            mf = _json.loads(mf_raw) if isinstance(mf_raw, str) else mf_raw
        except Exception:
            continue

        blocked_markets = {m.lower() for m in mf.get("block", [])}
        alt_min_line    = mf.get("alt_spreads_min_line")  # e.g. 25
        team_lower      = thesis.team.lower()

        for leg in run_legs:
            if leg.user_excluded:
                continue  # already excluded by another thesis or manually

            desc = (leg.description or "").lower()
            if team_lower not in desc:
                continue  # doesn't involve this team

            mkt = (leg.market_type or "").lower()

            # Check if this market is blocked
            is_blocked = any(b in mkt for b in blocked_markets)
            if not is_blocked:
                continue

            # Alt-spread exception: if line is wide enough, allow it
            if "alternate_spread" in mkt and alt_min_line is not None:
                # Parse line from description: e.g. "Philadelphia 76ers +27.5 (Alt Spread)"
                import re as _re
                m = _re.search(r"([+\-][\d.]+)", leg.description or "")
                if m:
                    line_val = float(m.group(1))
                    if line_val >= float(alt_min_line):
                        continue  # allowed — wide enough cushion

            # Exclude this leg with Mode B
            leg.user_excluded           = True
            leg.user_excluded_reason    = f"Auto: thesis #{thesis.id} — {thesis.title}"
            leg.user_excluded_at        = datetime.utcnow()
            leg.user_excluded_thesis_id = thesis.id
            leg.exclusion_mode          = "recalculate"
            thesis.total_excluded_legs  = (thesis.total_excluded_legs or 0) + 1
            total_excluded += 1

            # Update parent bet
            bet = db.get(MockBet, leg.mock_bet_id)
            if bet:
                # Track recalculated bet count (only increment once per bet per thesis)
                if not bet.has_excluded_legs:
                    thesis.total_recalculated_bets = (thesis.total_recalculated_bets or 0) + 1
                bet.has_excluded_legs      = True
                bet.exclusion_mode_summary = "recalculate"
                # Recompute odds: divide out this leg's decimal
                if leg.open_odds:
                    current_dec = float(bet.recalculated_odds_decimal or bet.odds or 1.0)
                    def _am_to_dec(am):
                        try:
                            a = int(am)
                            return a / 100 + 1 if a >= 100 else 100 / abs(a) + 1
                        except Exception:
                            return None
                    def _dec_to_am(d):
                        try:
                            return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))
                        except Exception:
                            return None
                    leg_dec = _am_to_dec(leg.open_odds)
                    if leg_dec and leg_dec > 1.0:
                        new_dec = max(1.001, round(current_dec / leg_dec, 4))
                        bet.recalculated_odds_decimal           = new_dec
                        bet.recalculated_combined_odds_american = _dec_to_am(new_dec)

    if total_excluded > 0:
        db.commit()
        print(f"[auto-thesis] {total_excluded} leg(s) auto-excluded across {len(theses)} active thesis(es)")

    return total_excluded


# ─── Exploration bet generation ───────────────────────────────────────────────

def _parse_expl_pick_point(
    description: str,
    market_type: str,
) -> "tuple[str | None, float | None, str | None]":
    """
    Parse (team_or_direction, point_value, ou_direction) from a MockBetLeg description.

    ou_direction is 'over' or 'under' for totals; None for spreads/h2h.
    Returns (None, None, None) when parsing fails.
    """
    import re
    desc = (description or "").strip()
    # Strip " — game (market)" suffix so we work on the core pick string only
    clean = re.sub(r"\s*—.*$", "", desc).strip()
    clean = re.sub(r"\s*\([^)]+\)\s*$", "", clean).strip()

    mkt = (market_type or "").lower()

    if "h2h" in mkt:
        # "{team} ML"
        pick = re.sub(r"\s+ML\s*$", "", clean, flags=re.IGNORECASE).strip()
        return (pick or None), None, None

    if "total" in mkt:
        # "Over 224.5" or "Under 8.5"
        m = re.match(r"^(Over|Under)\s+([\d.]+)", clean, re.IGNORECASE)
        if m:
            return m.group(1), float(m.group(2)), m.group(1).lower()
        return None, None, None

    if "spread" in mkt:
        # "{team} +7.5" or "{team} PK" or "{team} -1.5"
        m = re.match(r"^(.+)\s+([+\-][\d.]+|PK)\s*$", clean)
        if m:
            team = m.group(1).strip()
            pt_str = m.group(2)
            pt = 0.0 if pt_str == "PK" else float(pt_str)
            return team, pt, None
        return None, None, None

    return None, None, None


def _get_adjacent_targets(
    market_type: str,
    team_or_dir: str,
    point: "float | None",
    ou_dir: "str | None",
) -> "list[tuple[str, str, float, str | None]]":
    """
    Compute adjacent line targets for an exploration bet.

    Returns list of (adj_market, adj_direction, adj_point, adj_ou_dir) where:
      adj_market   : 'spread' or 'total'
      adj_direction: team name (spreads) or 'over'/'under' (totals)
      adj_point    : numeric line value
      adj_ou_dir   : 'over'/'under' (totals) or None (spreads)

    Spec:
      moneyline  → same team at +1.5 and -1.5 spread
      Over  X    → Over (X-1.0)  and Under (X+2.0)
      Under X    → Under (X+1.0) and Over  (X-2.0)
      Spread +P  → +(P-1.0) tighter and +(P+1.0) wider
    """
    mkt = (market_type or "").lower()
    targets: list = []

    if "h2h" in mkt and team_or_dir:
        targets.append(("spread", team_or_dir, 1.5,  None))
        targets.append(("spread", team_or_dir, -1.5, None))

    elif "total" in mkt and point is not None and ou_dir:
        if "over" in ou_dir:
            targets.append(("total", "over",  round(point - 1.0, 1), "over"))
            targets.append(("total", "under", round(point + 2.0, 1), "under"))
        else:
            targets.append(("total", "under", round(point + 1.0, 1), "under"))
            targets.append(("total", "over",  round(point - 2.0, 1), "over"))

    elif "spread" in mkt and team_or_dir and point is not None:
        tighter = round(point - 1.0, 1)
        wider   = round(point + 1.0, 1)
        targets.append(("spread", team_or_dir, tighter, None))
        targets.append(("spread", team_or_dir, wider,   None))

    return targets


def generate_exploration_bets(
    db: Session,
    run_id: str | None = None,
    game_date: str | None = None,
) -> dict:
    """
    Create adjacent-line single-leg exploration bets for each CUSHION leg in
    today's Section A picks (prospective / prospective_pm / forced_generation /
    top_picks_page).

    Each CUSHION leg spawns up to 2 exploration bets (one tighter line, one
    opposite side or wider line).  Bets are written with:
      source   = 'exploration'
      amount   = 0.0  (zero stake — informational only)
      legs     = 1    (always single-leg)

    Already-existing exploration bets for the same bet_info string are skipped
    so this function is idempotent / safe to call multiple times.

    Returns: {"created": int, "skipped_dup": int, "cushion_legs": int}
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8] + "_expl"

    _today = game_date or datetime.now(_CT).strftime("%Y-%m-%d")
    _src_ph = ",".join("?" * len(_SECTION_A_SOURCES))

    # ── 1. Load today's CUSHION legs from Section A picks ────────────────────
    try:
        con = sqlite3.connect(_BETS_DB_PATH)
        con.row_factory = sqlite3.Row
        # "CUSHION legs" = alternate-line bets (alt spread / alt total) and moneylines,
        # which are the primary line-selection positions in Section A picks.
        # The CUSHION/AVOID designation lives in personal_edge_profile.margin_grade (not
        # mock_bet_legs.grade, which uses A-F LQS letter grades).
        cushion_rows = con.execute(
            f"""
            SELECT mbl.fixture_id, mbl.market_type, mbl.sport,
                   mbl.description, mb.bet_info AS parent_info
            FROM   mock_bet_legs mbl
            JOIN   mock_bets     mb  ON mb.id = mbl.mock_bet_id
            WHERE  mb.game_date = ?
              AND  mb.source    IN ({_src_ph})
              AND  mbl.market_type IN ('alternate_spreads', 'alternate_totals', 'h2h')
              AND  mb.status    = 'PENDING'
            """,
            [_today] + list(_SECTION_A_SOURCES),
        ).fetchall()

        # Existing exploration bet_infos for today → dedup
        existing_infos: set[str] = {
            r[0]
            for r in con.execute(
                "SELECT bet_info FROM mock_bets WHERE source = 'exploration' AND game_date = ?",
                (_today,),
            ).fetchall()
        }
        con.close()
    except Exception as exc:
        return {"error": str(exc), "created": 0, "skipped_dup": 0, "cushion_legs": 0}

    created = 0
    skipped = 0
    cushion_count = len(cushion_rows)

    for crow in cushion_rows:
        fixture_id  = crow["fixture_id"]
        market_type = crow["market_type"] or ""
        sport       = crow["sport"] or "Unknown"
        description = crow["description"] or ""
        parent_info = crow["parent_info"] or ""

        if not fixture_id:
            continue

        # ── 2. Parse pick / point from description ───────────────────────────
        team_or_dir, point, ou_dir = _parse_expl_pick_point(description, market_type)
        if team_or_dir is None:
            continue

        # ── 3. Compute adjacent line targets ─────────────────────────────────
        targets = _get_adjacent_targets(market_type, team_or_dir, point, ou_dir)

        for adj_mkt, adj_dir, adj_pt, adj_ou in targets:
            # Look up odds in alt_lines (historical.db)
            odds = _get_alt_line_odds(
                fixture_id, adj_mkt, adj_dir, adj_pt, _HIST_DB_PATH
            )
            if odds is None:
                continue  # line not available in alt_lines

            # ── 4. Build bet_info and description ────────────────────────────
            if adj_mkt == "spread":
                sign = "+" if adj_pt > 0 else ("" if adj_pt == 0 else "")
                pt_str = "PK" if adj_pt == 0 else f"{sign}{adj_pt:g}"
                bet_info = f"{adj_dir} {pt_str} (Alt Spread) [Expl]"
                leg_desc = f"{adj_dir} {pt_str} (Alt Spread)"
                leg_mkt  = "alternate_spreads"
            else:
                dir_cap  = "Over" if adj_ou == "over" else "Under"
                bet_info = f"{dir_cap} {adj_pt:g} (Alt Total) [Expl]"
                leg_desc = f"{dir_cap} {adj_pt:g} (Alt Total)"
                leg_mkt  = "alternate_totals"

            if bet_info in existing_infos:
                skipped += 1
                continue
            existing_infos.add(bet_info)

            # Implied probability from decimal odds
            win_prob = round(1.0 / odds, 4) if odds > 1.0 else 0.5

            # ── 5. Write MockBet + MockBetLeg ────────────────────────────────
            mock_id = str(uuid.uuid4())[:16]
            db.add(MockBet(
                id                = mock_id,
                generation_run_id = run_id,
                game_date         = _today,
                sport             = sport,
                bet_type          = "straight",
                odds              = odds,
                amount            = 0.0,   # no stake; informational only
                legs              = 1,
                bet_info          = bet_info,
                predicted_win_prob = win_prob,
                predicted_ev      = None,
                confidence        = None,
                model_used        = "exploration",
                model_auc         = None,
                status            = "PENDING",
                source            = "exploration",
                avg_lqs           = None,
                notes             = f"Exploration from Section A: {parent_info}",
            ))
            db.add(MockBetLeg(
                mock_bet_id        = mock_id,
                leg_index          = 0,
                description        = leg_desc,
                market_type        = leg_mkt,
                sport              = sport,
                win_prob           = win_prob,
                ev                 = None,
                grade              = None,
                model_used         = "exploration",
                fixture_id         = fixture_id,
                is_alt_line        = True,
                predicted_win_prob = win_prob,
                predicted_edge_pp  = None,
            ))
            created += 1

    if created > 0:
        db.commit()
        print(f"[exploration] created={created}  skipped_dup={skipped}  cushion_legs={cushion_count}")
    else:
        print(f"[exploration] no new bets created  (cushion_legs={cushion_count})")

    return {
        "created":      created,
        "skipped_dup":  skipped,
        "cushion_legs": cushion_count,
    }


# ─── Settlement helpers ───────────────────────────────────────────────────────

def _settle_curation_modes(mock, leg_objs: list, leg_outcomes: list[dict], stake: float, db) -> None:
    """
    After a bet settles, compute Mode B recalculated profit and Mode C counterfactual
    messages for any legs that were user-excluded.

    Called immediately after mock.actual_profit is written, while leg_outcomes is still
    in scope. Does not commit — caller owns the transaction.
    """
    has_excluded = getattr(mock, "has_excluded_legs", False)
    if not has_excluded:
        return

    mode_summary = getattr(mock, "exclusion_mode_summary", None)
    if mode_summary not in ("recalculate", "counterfactual", "null_bet"):
        return

    excluded_legs = [(lo, leg) for lo, leg in zip(leg_outcomes, leg_objs)
                     if getattr(leg, "user_excluded", False)]
    if not excluded_legs:
        return

    # ── Mode B: compute recalculated_actual_profit ─────────────────────────
    if mode_summary == "recalculate":
        recalc_dec = getattr(mock, "recalculated_odds_decimal", None)
        if recalc_dec and recalc_dec > 1.0:
            # Check outcome of non-excluded legs only
            remaining_outcomes = [lo["outcome"] for lo, leg in zip(leg_outcomes, leg_objs)
                                   if not getattr(leg, "user_excluded", False)]
            if None not in remaining_outcomes:
                if "LOSS" in remaining_outcomes:
                    mock.recalculated_actual_profit = -stake
                else:
                    mock.recalculated_actual_profit = round((recalc_dec - 1) * stake, 2)

    # ── Mode C: generate counterfactual_message ────────────────────────────
    elif mode_summary == "counterfactual":
        for lo, leg in excluded_legs:
            excl_leg_result  = lo.get("outcome")          # WIN / LOSS / PUSH
            original_outcome = mock.status                 # SETTLED_WIN / SETTLED_LOSS
            profit_str       = f"${abs(mock.actual_profit or 0):.2f}"

            if excl_leg_result == "WIN" and original_outcome == "SETTLED_WIN":
                msg = (f"Bet won despite your exclusion — the removed leg hit. "
                       f"Original profit would have been +{profit_str}.")
            elif excl_leg_result == "WIN" and original_outcome == "SETTLED_LOSS":
                msg = ("Bet lost on other legs — your excluded leg actually won but "
                       "didn't save the bet.")
            elif excl_leg_result == "LOSS" and original_outcome == "SETTLED_LOSS":
                msg = (f"Good call — excluded leg lost. Bet would have lost {profit_str} "
                       f"regardless (other legs also failed).")
            elif excl_leg_result == "LOSS" and original_outcome == "SETTLED_WIN":
                msg = ("Bet won on remaining legs — excellent call. Your excluded leg "
                       f"lost but you still booked +{profit_str}.")
            else:
                msg = f"Excluded leg result: {excl_leg_result}. Original outcome: {original_outcome}."

            mock.counterfactual_message = msg
            break  # use first excluded leg for the message


# ─── Settlement ───────────────────────────────────────────────────────────────

def _settle_mock_leg(
    leg,
    scores_by_id: dict,   # fixture_id → score dict (OddsAPI format)
    all_scores: list,     # fallback: all completed scores
) -> dict:
    """
    Resolve a single MockBetLeg against completed game scores.

    Returns a dict with keys:
        outcome   — 'WIN' | 'LOSS' | 'PUSH' | None (unresolvable)
        home_team, away_team, home_score, away_score
        margin              — team_score - opp_score
        adjusted_margin     — margin + line (spread) or total_diff (total)
        accuracy_delta      — same as adjusted_margin

    Pinning: if leg.fixture_id is in scores_by_id, only that game is
    checked — avoids the multi-game-same-team-in-3-day-window bug.
    """
    import re as _re

    EMPTY = {"outcome": None, "home_team": None, "away_team": None,
             "home_score": None, "away_score": None,
             "margin": None, "adjusted_margin": None, "accuracy_delta": None}

    desc       = (leg.description or "").strip()
    desc_lower = desc.lower()
    market     = (leg.market_type or "").lower()
    is_soccer  = (leg.sport or "") in _SOCCER_SPORTS

    # ── Market category — standard ────────────────────────────────────────────
    is_spread = any(kw in market for kw in ("spread", "run_line", "puck_line")) or \
                any(kw in desc_lower for kw in ("alt spread", "run line", "puck line", " spread)"))
    is_ml     = market in ("h2h",) or "moneyline" in desc_lower
    is_total  = ("total" in market or "totals" in market or
                 _re.search(r"\b(over|under)\b", desc_lower) is not None or
                 ("goal" in desc_lower and not is_ml))

    # ── Soccer-specific markets ────────────────────────────────────────────────
    is_double_chance = "double chance" in desc_lower
    is_btts          = ("both teams to score" in desc_lower or
                        desc_lower.startswith("btts"))
    is_corners       = "corner" in desc_lower
    # Soccer player prop: goal-scorer / shots / assists — no data source available
    is_soccer_prop   = (
        is_soccer and
        not is_btts and not is_double_chance and not is_corners and not is_total and
        any(kw in desc_lower for kw in ("to score", "shots on target", "to have",
                                         "assists", "anytime scorer"))
    )

    # ── Early exits for unresolvable soccer markets ───────────────────────────
    if is_corners:
        return {**EMPTY, "skip_reason": "UNRESOLVABLE_STAT: corners data not available"}
    if is_soccer_prop:
        return {**EMPTY, "skip_reason": "UNRESOLVABLE_PROP: soccer player stats not available"}

    # ── Parse numeric line ────────────────────────────────────────────────────
    line: Optional[float] = None
    total_direction: Optional[str] = None

    if is_spread and not is_double_chance:
        m = _re.search(r"([+-]?\d+\.?\d*)\s*\(", desc)
        if m:
            line = float(m.group(1))
    if is_total and not is_btts:
        m = _re.search(r"(over|under)\s+([\d.]+)", desc_lower)
        if m:
            total_direction = m.group(1)
            line = float(m.group(2))

    # ── Candidate games: fixture_id-pinned or name-based fallback ────────────
    # CRITICAL: if leg.fixture_id is set, ONLY check that exact fixture.
    # Do NOT fall back to all_scores when the fixture isn't scored yet —
    # the fallback would match a previous-day game for the same team, triggering
    # a false SETTLED_LOSS before the real game is even complete in the API.
    # Return None (unresolvable) instead; the bet stays PENDING until the game
    # appears as completed in the scores API.
    from auto_settle import _team_match
    if leg.fixture_id:
        if leg.fixture_id in scores_by_id:
            candidates: list[dict] = [scores_by_id[leg.fixture_id]]
        else:
            # Fixture is known but not yet in scores — game in progress or
            # API hasn't ingested it yet. Stay PENDING.
            return EMPTY
    else:
        candidates = all_scores

    for game in candidates:
        if not game.get("completed"):
            continue
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        score_map: dict[str, int] = {}
        for sc in (game.get("scores") or []):
            try:
                score_map[sc["name"]] = int(sc["score"])
            except (KeyError, ValueError):
                pass
        if len(score_map) < 2:
            continue
        home_score = score_map.get(home, 0)
        away_score = score_map.get(away, 0)

        # ── Moneyline ─────────────────────────────────────────────────────────
        if is_ml:
            pick = _re.sub(r"\(moneyline\)", "", desc, flags=_re.IGNORECASE).strip()

            # Soccer "Draw" pick — explicit bet on the draw result
            if pick.lower() == "draw":
                margin = home_score - away_score
                outcome = "WIN" if margin == 0 else "LOSS"
                return {"outcome": outcome, "home_team": home, "away_team": away,
                        "home_score": home_score, "away_score": away_score,
                        "margin": margin, "adjusted_margin": margin, "accuracy_delta": margin}

            side = _team_match(pick, home, away)
            if side is None:
                continue
            ts = home_score if side == "home" else away_score
            os = away_score if side == "home" else home_score
            margin = ts - os
            is_soccer = (leg.sport or "") in _SOCCER_SPORTS
            if margin > 0:
                outcome = "WIN"
            elif margin == 0:
                # Soccer: draw = LOSS for a team ML bet (not a push)
                outcome = "LOSS" if is_soccer else "PUSH"
            else:
                outcome = "LOSS"
            return {"outcome": outcome, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score,
                    "margin": margin, "adjusted_margin": margin, "accuracy_delta": margin}

        # ── Spread / Alt Spread / Run Line / Puck Line ────────────────────────
        if is_spread and line is not None:
            # Extract team name: everything before the +/- line value
            pick_raw = _re.split(r"[+-]\d", desc)[0].strip()
            side = _team_match(pick_raw, home, away)
            if side is None:
                continue
            ts = home_score if side == "home" else away_score
            os = away_score if side == "home" else home_score
            margin   = ts - os
            adjusted = margin + line
            if abs(adjusted) < 0.01:
                outcome = "PUSH"
            else:
                outcome = "WIN" if adjusted > 0 else "LOSS"
            return {"outcome": outcome, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score,
                    "margin": margin, "adjusted_margin": adjusted, "accuracy_delta": adjusted}

        # ── Total (Over/Under / Total Goals) ─────────────────────────────────
        if is_total and line is not None and total_direction is not None:
            combined = home_score + away_score
            diff = combined - line
            if abs(diff) < 0.01:
                outcome = "PUSH"
            elif total_direction == "over":
                outcome = "WIN" if combined > line else "LOSS"
            else:
                outcome = "WIN" if combined < line else "LOSS"
            return {"outcome": outcome, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score,
                    "margin": combined, "adjusted_margin": diff, "accuracy_delta": diff}

        # ── Double Chance (soccer) ────────────────────────────────────────────
        # "Barcelona And Draw Double Chance" → WIN if Barcelona wins OR draws
        # i.e. LOSS only when the OTHER team wins outright
        if is_double_chance:
            # Extract team: everything before "and draw"
            m_dc = _re.search(r"^(.+?)\s+and\s+draw\s+double\s+chance", desc_lower)
            if m_dc:
                team_pick = m_dc.group(1).strip()
                side = _team_match(team_pick, home, away)
                if side is None:
                    continue
                ts = home_score if side == "home" else away_score
                os = away_score if side == "home" else home_score
                # WIN if picked team wins or it's a draw; LOSS only if picked team loses
                outcome = "LOSS" if ts < os else "WIN"
                margin  = ts - os
                return {"outcome": outcome, "home_team": home, "away_team": away,
                        "home_score": home_score, "away_score": away_score,
                        "margin": margin, "adjusted_margin": margin, "accuracy_delta": margin}

        # ── Both Teams To Score (BTTS) ────────────────────────────────────────
        # "Both Teams To Score - Yes" → both teams score ≥ 1
        # "Both Teams To Score - No"  → at least one team scores 0
        if is_btts:
            both_scored = home_score > 0 and away_score > 0
            is_yes = "- no" not in desc_lower   # default is Yes unless "- No" present
            outcome = "WIN" if (both_scored == is_yes) else "LOSS"
            combined = home_score + away_score
            return {"outcome": outcome, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score,
                    "margin": combined, "adjusted_margin": None, "accuracy_delta": None}

    return EMPTY


def generate_retroactive_bets(
    db:               Session,
    fixture_ids:      list[str],
    known_scores:     dict,          # fixture_id → {home_team, away_team, home_score, away_score}
    stake:            float  = 10.0,
    weight:           float  = 0.25, # lookahead discount — results already known
    source:           str    = "retroactive_mock",
    retroactive_reason: str  = "",
    note:             str    = "",
    game_date:        str    = None,
    forced_legs_spec: list[dict] | None = None,  # if set, build exactly this parlay
    # forced_legs_spec format: [{fixture_id, market, pick, point}, ...]
) -> dict:
    """
    Generate mock bets for already-completed games and settle immediately.

    Unlike generate_mock_bets(), this function:
      - Bypasses the future-game-only filter in get_available_legs()
      - Only considers legs from the supplied fixture_ids
      - Immediately settles each bet against known_scores
      - Tags bets with source, weight, retroactive_reason, and note

    known_scores format:
        { fixture_id: {home_team, away_team, home_score, away_score} }

    Returns:
        {generated, settled, wins, losses, pushes, legs, bets, errors}
    """
    import itertools
    from parlay_builder import score_leg, SPORT_LABEL, MARKET_LABEL
    from database import Fixture

    run_id    = f"retro_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    gdate     = game_date or datetime.utcnow().strftime("%Y-%m-%d")
    errors: list[str] = []

    # ── Fetch legs for these specific fixtures (bypass future-only guard) ──────
    fixtures = db.query(Fixture).filter(Fixture.id.in_(fixture_ids)).all()
    if not fixtures:
        return {"generated": 0, "error": "No fixtures found for supplied IDs"}

    raw_legs: list[dict] = []
    markets  = ["h2h", "spreads", "totals"]
    for fix in fixtures:
        if not fix.bookmakers:
            continue
        sport_label = SPORT_LABEL.get(fix.sport_key, fix.sport_title or fix.sport_key)
        game_label  = f"{fix.away_team} @ {fix.home_team}"
        game_time   = fix.commence_time.isoformat() if fix.commence_time else None

        best: dict[tuple, float] = {}
        for bk in fix.bookmakers:
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in markets:
                    continue
                for outcome in mkt.get("outcomes", []):
                    name  = outcome.get("name", "")
                    price = float(outcome.get("price", 0))
                    point = outcome.get("point")
                    key   = (mkt_key, name, point)
                    if key not in best or price > best[key]:
                        best[key] = price

        for (mkt_key, name, point), price in best.items():
            if price <= 1.0:
                continue
            mkt_label = MARKET_LABEL.get(mkt_key, mkt_key)
            if mkt_key == "spreads" and point is not None:
                sign = "+" if float(point) > 0 else ""
                desc = f"{name} {sign}{point} ({mkt_label})"
            elif mkt_key == "totals" and point is not None:
                desc = f"{name} {point} ({mkt_label})"
            else:
                desc = f"{name} ({mkt_label})"
            raw_legs.append({
                "leg_id":     f"{fix.id}:{mkt_key}:{name}:{point}",
                "fixture_id": fix.id,
                "game":       game_label,
                "sport":      sport_label,
                "sport_key":  fix.sport_key,
                "market":     mkt_key,
                "market_label": mkt_label,
                "pick":       name,
                "point":      point,
                "description": desc,
                "odds":       round(price, 3),
                "implied_prob": round(1 / price * 100, 2),
                "game_time":  game_time,
                "home_team":  fix.home_team,
                "away_team":  fix.away_team,
            })

    if not raw_legs:
        return {"generated": 0, "error": "No odds data found for these fixtures"}

    # ── Score all legs through the current ATS model ──────────────────────────
    scored: list[dict] = []
    for leg in raw_legs:
        try:
            sl = score_leg(leg)
            if sl.get("win_prob", 0) >= 50.0 and sl.get("edge", -99) > 0:
                scored.append(sl)
        except Exception as exc:
            errors.append(f"score_leg {leg.get('leg_id')}: {exc}")

    if not scored:
        return {"generated": 0, "error": "No positive-EV legs found (all legs below 50% win_prob or negative edge)"}

    # Add LQS to each scored leg
    for leg in scored:
        try:
            _game_str  = leg.get("game", "")
            _pick_team = leg.get("pick", "")
            _home_team = _game_str.split(" @ ")[1].strip() if " @ " in _game_str else ""
            _away_team = _game_str.split(" @ ")[0].strip() if " @ " in _game_str else ""
            _is_home   = (True  if _home_team and _pick_team == _home_team else
                          False if _away_team and _pick_team == _away_team else None)
            q = lq.compute_leg_quality_score({
                "market_type":      leg.get("market"),
                "sport":            leg.get("sport"),
                "team_or_player":   _pick_team,
                "model_confidence": leg.get("win_prob"),
                "model_used":       leg.get("model_used"),
                "edge_pp":          leg.get("edge"),
                "line":             leg.get("point"),
                "is_home":          _is_home,
            }, db)
            leg["lqs"] = q.get("lqs", 65.0)
        except Exception:
            leg["lqs"] = 65.0

    # Sort by win_prob descending for parlay building
    scored.sort(key=lambda x: -x.get("win_prob", 0))

    # ── Forced-pick path: build a single specified parlay ─────────────────────
    # When forced_legs_spec is provided, look up each specified leg from scored
    # (matched by fixture_id + market + pick + point) and assemble as one bet.
    if forced_legs_spec:
        forced_legs: list[dict] = []
        for spec in forced_legs_spec:
            fid   = spec.get("fixture_id")
            mkt   = spec.get("market", "spreads")
            pick  = spec.get("pick", "")
            point = spec.get("point")
            match = None
            # Search scored legs for this fixture/market/outcome/point
            for sl in scored:
                if (sl.get("fixture_id") == fid
                        and sl.get("market") == mkt
                        and sl.get("pick", "").lower() == pick.lower()):
                    if point is None or sl.get("point") == point:
                        match = sl
                        break
            # If not in scored (e.g. win_prob < 50 for a cross-date game),
            # fall back to raw_legs so we can still build the calibration record.
            if match is None:
                for rl in raw_legs:
                    if (rl.get("fixture_id") == fid
                            and rl.get("market") == mkt
                            and rl.get("pick", "").lower() == pick.lower()):
                        if point is None or rl.get("point") == point:
                            try:
                                rl = score_leg(rl)
                            except Exception:
                                pass
                            if not rl.get("lqs"):
                                rl["lqs"] = 65.0
                            match = rl
                            break
            if match is None:
                errors.append(f"forced_leg not found: fid={fid} mkt={mkt} pick={pick} pt={point}")
            else:
                forced_legs.append(match)

        if len(forced_legs) != len(forced_legs_spec):
            return {
                "generated": 0,
                "error": f"Could not resolve all forced legs — found {len(forced_legs)}/{len(forced_legs_spec)}",
                "errors": errors,
            }

        combined_odds = 1.0
        combined_win  = 1.0
        for l in forced_legs:
            combined_odds *= l.get("odds", 1.0)
            combined_win  *= (l.get("win_prob", 50.0) / 100)
        payout         = (combined_odds - 1) * stake
        expected_profit = combined_win * payout - (1 - combined_win) * stake

        picks = [{
            "legs":              forced_legs,
            "combined_odds":     round(combined_odds, 3),
            "combined_win_prob": round(combined_win * 100, 2),
            "expected_profit":   round(expected_profit, 2),
            "n_legs":            len(forced_legs),
            "is_power_pick":     len(forced_legs) >= 4,
        }]
    else:
        # ── Build 2-leg parlays (one leg per fixture, no same-team repeats) ─────
        _MAX_LEG_APP = 2
        leg_use: dict[str, int] = {}
        candidates: list[dict] = []

        for leg_a, leg_b in itertools.combinations(scored, 2):
            # Enforce one leg per fixture
            if leg_a["fixture_id"] == leg_b["fixture_id"]:
                continue
            # Enforce team diversity
            teams_a = {leg_a.get("pick", "").strip().lower()}
            teams_b = {leg_b.get("pick", "").strip().lower()}
            if teams_a & teams_b:
                continue
            combined_odds = leg_a["odds"] * leg_b["odds"]
            if combined_odds < 2.0:
                continue
            combined_win  = (leg_a["win_prob"] / 100) * (leg_b["win_prob"] / 100)
            payout        = (combined_odds - 1) * stake
            expected_profit = combined_win * payout - (1 - combined_win) * stake
            if expected_profit <= 0:
                continue
            candidates.append({
                "legs": [leg_a, leg_b],
                "combined_odds": round(combined_odds, 3),
                "combined_win_prob": round(combined_win * 100, 2),
                "expected_profit": round(expected_profit, 2),
                "n_legs": 2,
            })

        candidates.sort(key=lambda x: (-x["combined_win_prob"], -x["expected_profit"]))

        # Select up to 3 unique bets with leg diversity
        seen_sigs: set = set()
        picks: list[dict] = []
        for cand in candidates:
            if len(picks) >= 3:
                break
            sig = frozenset(l["leg_id"] for l in cand["legs"])
            if sig in seen_sigs:
                continue
            if any(leg_use.get(l["leg_id"], 0) >= _MAX_LEG_APP for l in cand["legs"]):
                continue
            seen_sigs.add(sig)
            for l in cand["legs"]:
                leg_use[l["leg_id"]] = leg_use.get(l["leg_id"], 0) + 1
            picks.append(cand)

    # ── Build OddsAPI-format scores for immediate settlement ──────────────────
    scores_by_id: dict[str, dict] = {}
    for fid, sc in known_scores.items():
        scores_by_id[fid] = {
            "id":         fid,
            "completed":  True,
            "home_team":  sc["home_team"],
            "away_team":  sc["away_team"],
            "scores": [
                {"name": sc["home_team"], "score": str(sc["home_score"])},
                {"name": sc["away_team"], "score": str(sc["away_score"])},
            ],
        }

    # ── Write bets and settle immediately ─────────────────────────────────────
    generated = 0
    settled   = 0
    wins = losses = pushes = 0
    bet_summaries: list[dict] = []

    # Deduplicate against existing retroactive bets for the same game_date
    existing_infos = {
        row.bet_info
        for row in db.query(MockBet)
                     .filter(MockBet.game_date == gdate,
                             MockBet.source == source)
                     .all()
    }

    for pick in picks:
        legs     = pick["legs"]
        bet_info = " | ".join(
            _format_leg_desc(l) for l in legs if l.get("description") or l.get("pick")
        )
        if not bet_info or bet_info in existing_infos:
            continue

        existing_infos.add(bet_info)
        avg_lqs = round(sum(l.get("lqs", 65) for l in legs) / len(legs), 1)
        sport_set = {l.get("sport", "") for l in legs if l.get("sport")}
        mock_id   = str(uuid.uuid4())

        mock = MockBet(
            id                  = mock_id,
            generation_run_id   = run_id,
            game_date           = gdate,
            sport               = " | ".join(sport_set) if sport_set else None,
            bet_type            = "parlay",
            odds                = pick["combined_odds"],
            amount              = stake,
            legs                = len(legs),
            bet_info            = bet_info,
            predicted_win_prob  = pick["combined_win_prob"],
            predicted_ev        = pick["expected_profit"],
            confidence          = None,
            model_used          = legs[0].get("model_used") if legs else None,
            status              = "PENDING",
            source              = source,
            weight              = weight,
            avg_lqs             = avg_lqs,
            notes               = (
                f"{note}\nretroactive_reason={retroactive_reason}" if note
                else f"retroactive_reason={retroactive_reason}"
            ).strip(),
        )
        db.add(mock)

        leg_objs = []
        for i, leg in enumerate(legs):
            leg_obj = MockBetLeg(
                mock_bet_id         = mock_id,
                leg_index           = i,
                description         = leg.get("description", ""),
                market_type         = leg.get("market_type") or leg.get("market"),
                sport               = leg.get("sport"),
                win_prob            = leg.get("win_prob"),
                ev                  = leg.get("ev"),
                grade               = leg.get("grade"),
                model_used          = leg.get("model_used"),
                fixture_id          = leg.get("fixture_id"),
                is_alt_line         = bool(leg.get("is_alt_line", False)),
                ale_considered      = bool(leg.get("ale_considered", False)),
                ale_naive_pick      = leg.get("ale_naive_pick"),
                ale_switched        = bool(leg.get("ale_switched", False)),
                ale_los_improvement = leg.get("ale_los_improvement"),
            )
            db.add(leg_obj)
            leg_objs.append(leg_obj)

        generated += 1

        # ── Immediate settlement ───────────────────────────────────────────────
        db.flush()   # give leg_objs their PKs before settle query

        leg_outcomes: list[dict] = []
        for leg_obj in leg_objs:
            result = _settle_mock_leg(leg_obj, scores_by_id, list(scores_by_id.values()))
            outcome = result.get("outcome")
            leg_outcomes.append({
                "description":     leg_obj.description,
                "outcome":         outcome,
                "margin":          result.get("margin"),
                "adjusted_margin": result.get("adjusted_margin"),
                "accuracy_delta":  result.get("accuracy_delta"),
                "home_score":      result.get("home_score"),
                "away_score":      result.get("away_score"),
            })
            # Write resolution fields onto the leg row (correct ORM column names)
            leg_obj.leg_result               = outcome
            leg_obj.resolved_home_team       = result.get("home_team")
            leg_obj.resolved_away_team       = result.get("away_team")
            leg_obj.resolved_home_score      = result.get("home_score")
            leg_obj.resolved_away_score      = result.get("away_score")
            leg_obj.resolved_margin          = result.get("margin")
            leg_obj.resolved_adjusted_margin = result.get("adjusted_margin")
            leg_obj.accuracy_delta           = result.get("accuracy_delta")

        # Parlay outcome: ALL legs must WIN (any LOSS → LOSS, any PUSH → PUSH if no LOSS)
        outcomes = [lo["outcome"] for lo in leg_outcomes]
        if None in outcomes:
            bet_outcome = None      # unresolvable — leave PENDING
        elif "LOSS" in outcomes:
            bet_outcome = "SETTLED_LOSS"
            losses += 1
        elif "PUSH" in outcomes:
            bet_outcome = "SETTLED_WIN"   # treat push as win for parlay purposes
            wins += 1
        else:
            bet_outcome = "SETTLED_WIN"
            wins += 1

        if bet_outcome:
            mock.status     = bet_outcome
            mock.settled_at = datetime.utcnow()
            _bet_amount    = mock.amount or 0.0  # 0.0 for forced_generation / exploration / BONUS_BET

            if bet_outcome == "SETTLED_WIN":
                _raw_profit = round((mock.odds - 1) * _bet_amount, 2)
                if getattr(mock, "promo_type", None) == "PROFIT_BOOST" and getattr(mock, "promo_boost_pct", None):
                    # Boost applies to profit only (not stake return)
                    mock.actual_profit = round(_raw_profit * (1.0 + mock.promo_boost_pct), 2)
                elif getattr(mock, "promo_type", None) == "BONUS_BET":
                    # Free bet: stake not returned — profit only (stake was $0 anyway)
                    _real_stake = 10.0  # nominal stake for sizing (actual stored amount is 0)
                    mock.actual_profit = round((mock.odds - 1) * _real_stake, 2)
                else:
                    mock.actual_profit = _raw_profit
            else:
                mock.actual_profit = -_bet_amount  # SETTLED_LOSS: lose the stake (0.0 for free bets)
            settled += 1

            # ── Post-settlement curation: Mode B (recalculate) and Mode C (counterfactual) ──
            _settle_curation_modes(mock, leg_objs, leg_outcomes, _bet_amount, db)

        bet_summaries.append({
            "bet_id":   mock_id,
            "bet_info": bet_info,
            "odds":     pick["combined_odds"],
            "avg_lqs":  avg_lqs,
            "outcome":  bet_outcome,
            "legs":     leg_outcomes,
        })

    db.commit()

    return {
        "generated":    generated,
        "settled":      settled,
        "wins":         wins,
        "losses":       losses,
        "pushes":       pushes,
        "run_id":       run_id,
        "source":       source,
        "weight":       weight,
        "bets":         bet_summaries,
        **({"errors": errors} if errors else {}),
    }


def rescore_pending_lqs(db: Session) -> dict:
    """
    Re-score avg_lqs for all PENDING mock bets using the current LQS weights.

    Called after a component weight change so pending bets are evaluated
    consistently with bets generated after the change.

    Uses stored predicted_win_prob (model_confidence proxy) and
    predicted_edge_pp (edge_pp proxy) from mock_bet_legs. Component A
    will use whatever historical data exists for the market_type/sport;
    team_or_player is parsed from the leg description as a best-effort.
    """
    pending_bets = db.query(MockBet).filter(MockBet.status == "PENDING").all()
    updated = errors = 0

    for bet in pending_bets:
        legs = (db.query(MockBetLeg)
                .filter(MockBetLeg.mock_bet_id == bet.id)
                .order_by(MockBetLeg.leg_index)
                .all())
        if not legs:
            continue

        new_scores: list[float] = []
        for leg in legs:
            # Best-effort team/player extraction from description
            desc = leg.description or ""
            if " — " in desc:
                pick_part = desc.split(" — ")[0].strip()
            elif " (" in desc:
                pick_part = desc.split(" (")[0].strip()
            else:
                pick_part = desc

            candidate = {
                "market_type":     leg.market_type,
                "sport":           leg.sport or "",
                "team_or_player":  pick_part,
                "model_confidence": leg.predicted_win_prob,
                "model_used":      leg.model_used,
                "edge_pp":         leg.predicted_edge_pp,
            }
            try:
                result = lq.compute_leg_quality_score(candidate, db)
                new_scores.append(result["lqs"])
            except Exception:
                errors += 1

        if new_scores:
            bet.avg_lqs = round(sum(new_scores) / len(new_scores), 1)
            updated += 1

    db.commit()
    return {"updated": updated, "errors": errors}


def settle_mock_bets(db: Session) -> dict:
    """
    Settle all PENDING MockBets by checking if their games have results.

    Resolution order (first success wins):
      1. Twin-bet match — real settled bet with identical bet_info.
      2. Leg-by-leg resolution via _settle_mock_leg():
           a. Fixture-pinned OddsAPI scores (fixture_id → exact game, no cross-day
              contamination from same team playing multiple days)
           b. Historical.db scores (NHL/MLB, converted to OddsAPI format)
           c. All fetched OddsAPI scores (fallback when no fixture_id)

    game_date fallback: if NULL use DATE(generated_at).
    Bets older than 7 days with no result → EXPIRED.
    Per-leg audit written to mock_bet_legs resolution columns.

    Returns settlement summary.
    """
    from auto_settle import _fetch_scores_cached, SPORT_TO_KEYS
    from database import Bet, Fixture

    cutoff  = datetime.utcnow() - timedelta(days=7)
    now_utc = datetime.utcnow()
    pending = db.query(MockBet).filter(MockBet.status == "PENDING").all()

    settled = 0
    expired = 0
    skipped = 0

    # ── Pre-build fixture commence_time cache ─────────────────────────────────
    # Used to skip bets where any fixture hasn't started yet — prevents
    # settlement from matching a leg against a previous-day game of the same
    # team when the fixture_id isn't in the scores API yet.
    _fix_cache: dict[str, datetime] = {}
    for fix in db.query(Fixture).all():
        if fix.commence_time:
            ct = fix.commence_time
            if ct.tzinfo is not None:
                ct = ct.replace(tzinfo=None)  # strip to naive UTC
            _fix_cache[fix.id] = ct

    # ── Pre-fetch OddsAPI scores per sport (cached, shared across all bets) ──
    _api_scores_cache: dict[str, list] = {}

    def _get_api_scores(sport_str: str) -> list[dict]:
        # sport_str may be pipe-separated (e.g. "La Liga | EPL") — fetch all parts
        cache_key = sport_str
        if cache_key not in _api_scores_cache:
            pool: list[dict] = []
            seen_sk: set[str] = set()
            for part in sport_str.split("|"):
                sk_label = part.strip()
                for sk in SPORT_TO_KEYS.get(sk_label, []):
                    if sk in seen_sk:
                        continue
                    seen_sk.add(sk)
                    try:
                        pool.extend(_fetch_scores_cached(sk, days_from=3))
                    except Exception:
                        pass
            _api_scores_cache[cache_key] = pool
        return _api_scores_cache[cache_key]

    for mock in pending:
        # ── Expire stale bets ────────────────────────────────────────────────
        if mock.generated_at and mock.generated_at < cutoff:
            mock.status        = "EXPIRED"
            mock.settled_at    = datetime.utcnow()
            mock.actual_profit = 0.0
            expired += 1
            continue

        # ── game_date: fall back to DATE(generated_at) ───────────────────────
        game_date: Optional[str] = mock.game_date or (
            mock.generated_at.strftime("%Y-%m-%d") if mock.generated_at else None
        )

        # ── 1. Twin-bet match ────────────────────────────────────────────────
        if mock.bet_info:
            twin = db.query(Bet).filter(
                Bet.bet_info == mock.bet_info,
                Bet.is_mock.is_(False),
                Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
            ).first()
            if twin:
                won = twin.status == "SETTLED_WIN"
                mock.status        = "SETTLED_WIN" if won else "SETTLED_LOSS"
                mock.settled_at    = datetime.utcnow()
                mock.actual_profit = round(
                    mock.amount * ((mock.odds or 2.0) - 1) if won else -mock.amount, 2
                )
                settled += 1
                continue

        # ── 2. Leg-by-leg resolution ─────────────────────────────────────────
        legs = (
            db.query(MockBetLeg)
            .filter(MockBetLeg.mock_bet_id == mock.id)
            .order_by(MockBetLeg.leg_index)
            .all()
        )
        if not legs:
            skipped += 1
            continue

        # ── Game-started gate ─────────────────────────────────────────────────
        # If any leg's fixture hasn't started yet, skip the entire bet — do not
        # attempt settlement. Without this gate the fallback all_scores search
        # can match yesterday's game for the same team, triggering a premature
        # SETTLED_LOSS via the early-LOSS short-circuit before the real game plays.
        _any_future_fixture = False
        for _leg in legs:
            if _leg.fixture_id:
                _ct = _fix_cache.get(_leg.fixture_id)
                if _ct is not None and _ct > now_utc:
                    _any_future_fixture = True
                    break
        if _any_future_fixture:
            skipped += 1
            continue

        # Build flat scores list + id-keyed dict for fixture pinning
        api_scores  = _get_api_scores(mock.sport or "")
        hist_scores: list[dict] = []
        if game_date:
            for g in _get_historical_games_for_date(game_date):
                if g.get("home_score") is not None and g.get("away_score") is not None:
                    hist_scores.append({
                        "completed": True,
                        "home_team": g["home_team"],
                        "away_team": g["away_team"],
                        "scores": [
                            {"name": g["home_team"], "score": str(g["home_score"])},
                            {"name": g["away_team"], "score": str(g["away_score"])},
                        ],
                    })

        all_scores: list[dict] = hist_scores + api_scores
        # Build id→game map for fixture_id pinning (only OddsAPI games have "id")
        scores_by_id: dict[str, dict] = {
            g["id"]: g for g in api_scores if g.get("id")
        }

        if not all_scores:
            skipped += 1
            continue

        # Resolve every leg; collect outcomes + audit data
        leg_resolutions: list[dict] = [
            _settle_mock_leg(leg, scores_by_id, all_scores) for leg in legs
        ]
        leg_outcomes = [r["outcome"] for r in leg_resolutions]

        # Write audit fields back to each MockBetLeg row
        for leg, res in zip(legs, leg_resolutions):
            leg.leg_result               = res.get("outcome")
            leg.resolved_home_team       = res.get("home_team")
            leg.resolved_away_team       = res.get("away_team")
            leg.resolved_home_score      = res.get("home_score")
            leg.resolved_away_score      = res.get("away_score")
            leg.resolved_margin          = res.get("margin")
            leg.resolved_adjusted_margin = res.get("adjusted_margin")
            leg.accuracy_delta           = res.get("accuracy_delta")

            # Line quality: two-dimensional evaluation
            lq_data = _compute_line_quality(leg, res, _HIST_DB_PATH)
            leg.main_market_line   = lq_data["main_market_line"]
            leg.main_market_result = lq_data["main_market_result"]
            leg.direction_correct  = lq_data["direction_correct"]
            leg.optimal_line       = lq_data["optimal_line"]
            leg.line_delta         = lq_data["line_delta"]
            leg.ab_alt_line        = lq_data["ab_alt_line"]
            leg.ab_alt_result      = lq_data["ab_alt_result"]
            leg.ab_alt_odds        = lq_data["ab_alt_odds"]
            leg.ab_alt_ev          = lq_data["ab_alt_ev"]

            # CLV: look up latest snapshot before game start (close odds)
            if leg.fixture_id and leg.open_odds is not None:
                _commence = res.get("commence_time") or (
                    mock.game_date + "T23:59:59Z" if mock.game_date else None
                )
                _close = _lookup_snap_odds(
                    leg.fixture_id,
                    leg.description or "",
                    leg.market_type or "",
                    _HIST_DB_PATH,
                    before_dt=_commence,
                )
                if _close is not None:
                    leg.close_odds    = _close
                    leg.clv_cents     = _close - leg.open_odds
                    leg.clv_available = 1
                else:
                    leg.clv_available = 0

        # ── Early-LOSS short-circuit ─────────────────────────────────────────
        # A parlay fails as soon as any leg loses — no need to wait for
        # unresolved legs (e.g., game still in progress today).
        if any(o == "LOSS" for o in leg_outcomes):
            mock.status        = "SETTLED_LOSS"
            mock.actual_profit = round(-mock.amount, 2)
            mock.settled_at    = datetime.utcnow()
            settled += 1
            continue

        # Require all legs resolved before declaring WIN or PUSH
        if any(o is None for o in leg_outcomes):
            skipped += 1
            continue

        if all(o == "WIN" for o in leg_outcomes):
            mock.status        = "SETTLED_WIN"
            mock.actual_profit = round(mock.amount * ((mock.odds or 2.0) - 1), 2)
        else:
            # All PUSHes — stake returned
            mock.status        = "SETTLED_LOSS"
            mock.actual_profit = 0.0

        mock.settled_at = datetime.utcnow()
        settled += 1

    db.commit()
    return {"settled": settled, "expired": expired, "skipped": skipped}


# ─── Performance / trust metrics ──────────────────────────────────────────────

def get_mock_performance(db: Session, days: int = 30) -> dict:
    """
    Aggregate win rate, P&L, and trust metrics for settled mock bets.

    Returns:
        {
          win_rate, pnl, roi_pct, total_settled, total_pending,
          by_confidence: [{level, win_rate, bets}],
          by_sport: [{sport, win_rate, bets}],
          by_model: [{model_used, win_rate, auc, bets}],
          trust_level: "VALIDATED" | "BUILDING" | "INSUFFICIENT_DATA",
          baseline_win_rate: 0.30,   # theoretical baseline for 4-leg parlay
        }
    """
    since = datetime.utcnow() - timedelta(days=days)
    settled = db.query(MockBet).filter(
        MockBet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
        MockBet.settled_at >= since,
    ).all()

    total_settled = len(settled)
    wins          = sum(1 for m in settled if m.status == "SETTLED_WIN")
    pnl           = sum((m.actual_profit or 0) for m in settled)
    total_staked  = sum((m.amount or 0) for m in settled)
    win_rate      = round(wins / total_settled * 100, 1) if total_settled else None
    roi_pct       = round(pnl / total_staked * 100, 1) if total_staked else None

    # Avg legs across settled mocks (to compute baseline)
    avg_legs = sum(m.legs or 1 for m in settled) / max(total_settled, 1)
    # Baseline: theoretical win rate at 50% per leg
    baseline_win_rate = round((0.5 ** avg_legs) * 100, 1)

    # Trust level
    if total_settled < 10:
        trust_level = "INSUFFICIENT_DATA"
    elif win_rate is None:
        trust_level = "INSUFFICIENT_DATA"
    elif win_rate >= baseline_win_rate * 1.15:
        trust_level = "VALIDATED"
    elif win_rate >= baseline_win_rate * 0.90:
        trust_level = "BUILDING"
    else:
        trust_level = "UNDERPERFORMING"

    # By confidence — bucket using model_confidence_avg (per-leg avg) when available
    # Falls back to stored string label for legacy rows.
    # Thresholds: HIGH >= 0.62, MEDIUM 0.55-0.62, LOW < 0.55
    from collections import defaultdict
    conf_stats: dict = defaultdict(lambda: {"bets": 0, "wins": 0})
    sport_stats: dict = defaultdict(lambda: {"bets": 0, "wins": 0})
    model_stats: dict = defaultdict(lambda: {"bets": 0, "wins": 0, "auc": None})

    for m in settled:
        # Recompute confidence bucket from numeric avg if available
        mc_avg = getattr(m, "model_confidence_avg", None)
        if mc_avg is not None:
            if mc_avg >= _CONF_HIGH:
                c = "HIGH"
            elif mc_avg >= _CONF_MEDIUM:
                c = "MEDIUM"
            else:
                c = "LOW"
        else:
            c = m.confidence or "UNKNOWN"
        conf_stats[c]["bets"] += 1
        conf_stats[c]["wins"] += 1 if m.status == "SETTLED_WIN" else 0

        s = (m.sport or "Unknown").split("|")[0].strip()
        sport_stats[s]["bets"] += 1
        sport_stats[s]["wins"] += 1 if m.status == "SETTLED_WIN" else 0

        mu = m.model_used or "combined"
        model_stats[mu]["bets"] += 1
        model_stats[mu]["wins"] += 1 if m.status == "SETTLED_WIN" else 0
        if m.model_auc and not model_stats[mu]["auc"]:
            model_stats[mu]["auc"] = m.model_auc

    def _pct(wins, total):
        return round(wins / total * 100, 1) if total else None

    pending_count = db.query(MockBet).filter(MockBet.status == "PENDING").count()

    # Recent daily trend (last 7 days) — bucket by game_date (match start date),
    # not settlement date, so late-night games stay on the day they were played.
    _CT      = ZoneInfo("America/Chicago")
    today_ct = datetime.now(_CT).date()
    trend = []
    for offset in range(6, -1, -1):
        target_date     = today_ct - timedelta(days=offset)
        target_date_str = target_date.strftime("%Y-%m-%d")
        day_bets = [m for m in settled if m.game_date == target_date_str]
        day_wins = sum(1 for m in day_bets if m.status == "SETTLED_WIN")
        trend.append({
            "date":     target_date_str,
            "bets":     len(day_bets),
            "wins":     day_wins,
            "win_rate": _pct(day_wins, len(day_bets)),
            "pnl":      round(sum((m.actual_profit or 0) for m in day_bets), 2),
        })

    # ── Prospective-only P&L (excludes forced_generation / exploration noise) ──
    _PROSPECTIVE_SOURCES = frozenset({
        "prospective", "prospective_pm", "top_picks_page", "prospective_legacy",
    })
    prosp_bets  = [m for m in settled if getattr(m, "source", "") in _PROSPECTIVE_SOURCES]
    prosp_wins  = sum(1 for m in prosp_bets if m.status == "SETTLED_WIN")
    prosp_pnl   = round(sum((m.actual_profit or 0) for m in prosp_bets), 2)
    prosp_staked = sum((m.amount or 0) for m in prosp_bets)
    prosp_wr    = _pct(prosp_wins, len(prosp_bets)) if prosp_bets else None
    prosp_roi   = round(prosp_pnl / prosp_staked * 100, 1) if prosp_staked else None

    # ── Live vs retroactive split ──────────────────────────────────────────
    live_bets  = [m for m in settled if getattr(m, "source", "prospective") != "retroactive_mock"]
    retro_bets = [m for m in settled if getattr(m, "source", "prospective") == "retroactive_mock"]

    live_wins  = sum(1 for m in live_bets  if m.status == "SETTLED_WIN")
    retro_wins = sum(1 for m in retro_bets if m.status == "SETTLED_WIN")
    live_wr    = _pct(live_wins,  len(live_bets))
    retro_wr   = _pct(retro_wins, len(retro_bets))

    # Live-only trust level (retroactive is informational only)
    if len(live_bets) < 10:
        live_trust = "INSUFFICIENT_DATA"
    elif live_wr >= baseline_win_rate * 1.15:
        live_trust = "VALIDATED"
    elif live_wr >= baseline_win_rate * 0.90:
        live_trust = "BUILDING"
    else:
        live_trust = "UNDERPERFORMING"

    # Retroactive trend for chart (by game_date, not settled_at)
    retro_trend = []
    if retro_bets:
        retro_by_date: dict = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
        for m in retro_bets:
            gd = getattr(m, "game_date", None) or (
                m.generated_at.strftime("%Y-%m-%d") if m.generated_at else "unknown"
            )
            retro_by_date[gd]["bets"] += 1
            retro_by_date[gd]["wins"] += 1 if m.status == "SETTLED_WIN" else 0
            retro_by_date[gd]["pnl"]  += m.actual_profit or 0
        # Last 30 dates
        for d in sorted(retro_by_date.keys())[-30:]:
            v = retro_by_date[d]
            retro_trend.append({
                "date":     d,
                "bets":     v["bets"],
                "wins":     v["wins"],
                "win_rate": _pct(v["wins"], v["bets"]),
                "pnl":      round(v["pnl"], 2),
            })

    # Active backfill jobs
    active_jobs = [j for j in list_backfill_jobs() if j.get("status") in ("queued", "running")]

    # ── Alt line leg breakdown ──────────────────────────────────────────────
    # Count settled mocks that contained at least one alt-line leg
    from sqlalchemy import text as _sa_text
    alt_line_mock_ids = set()
    try:
        alt_rows = db.execute(_sa_text("""
            SELECT DISTINCT mbl.mock_bet_id
            FROM mock_bet_legs mbl
            WHERE mbl.is_alt_line = 1
        """)).fetchall()
        alt_line_mock_ids = {r[0] for r in alt_rows}
    except Exception:
        pass

    alt_bets      = [m for m in settled if m.id in alt_line_mock_ids]
    main_only     = [m for m in settled if m.id not in alt_line_mock_ids]
    alt_wins      = sum(1 for m in alt_bets  if m.status == "SETTLED_WIN")
    main_only_wins = sum(1 for m in main_only if m.status == "SETTLED_WIN")
    alt_line_leg_pct  = round(len(alt_bets) / total_settled * 100, 1) if total_settled else None
    alt_line_win_rate = _pct(alt_wins,       len(alt_bets))
    main_line_win_rate = _pct(main_only_wins, len(main_only))

    # ── LQS breakdown: high (≥70) / med (60–69) / low (<60) ──────────────
    # Only include post-launch bets with real LQS scores (≥ 2026-04-29).
    # Pre-simulation Pikkit-resolution bets predate the LQS system and would
    # pollute the correlation analysis with a large "No LQS" noise bucket.
    _LQS_CUTOFF = "2026-04-29"
    lqs_eligible = [
        m for m in settled
        if m.avg_lqs is not None
        and m.generated_at is not None
        and m.generated_at.strftime("%Y-%m-%d") >= _LQS_CUTOFF
    ]
    lqs_high  = [m for m in lqs_eligible if m.avg_lqs >= 70]
    lqs_med   = [m for m in lqs_eligible if 60 <= m.avg_lqs < 70]
    lqs_low   = [m for m in lqs_eligible if m.avg_lqs < 60]

    def _lqs_bucket(bets_subset):
        n = len(bets_subset)
        if n == 0:
            return None
        w = sum(1 for m in bets_subset if m.status == "SETTLED_WIN")

        # Legs breakdown: group by leg count to surface composition artifacts
        # (e.g. 4-leg parlays dragging down a bucket's win rate)
        legs_map: dict = {}
        for m in bets_subset:
            key = m.legs or 1
            if key not in legs_map:
                legs_map[key] = {"bets": 0, "wins": 0}
            legs_map[key]["bets"] += 1
            if m.status == "SETTLED_WIN":
                legs_map[key]["wins"] += 1

        by_legs = [
            {
                "legs":     k,
                "label":    "straight" if k == 1 else f"{k}-leg parlay",
                "bets":     v["bets"],
                "wins":     v["wins"],
                "win_rate": _pct(v["wins"], v["bets"]),
            }
            for k, v in sorted(legs_map.items())
        ]

        return {"bets": n, "wins": w, "win_rate": _pct(w, n), "by_legs": by_legs}

    by_lqs = []
    if lqs_high: by_lqs.append({"bucket": "High LQS (≥70)",  **_lqs_bucket(lqs_high)})
    if lqs_med:  by_lqs.append({"bucket": "Med LQS (60–69)", **_lqs_bucket(lqs_med)})
    if lqs_low:  by_lqs.append({"bucket": "Low LQS (<60)",   **_lqs_bucket(lqs_low)})

    # ── Compact line quality embed ─────────────────────────────────────────
    # Call get_line_quality_summary (reads its own DB conn, safe to nest).
    # Compact=True → skip the per-sport breakdown table to keep response slim.
    try:
        _lq = get_line_quality_summary(db)
        line_quality_embed = {
            "dir_accuracy_pct":    _lq.get("dir_accuracy_pct"),
            "avg_line_delta":      _lq.get("avg_line_delta"),
            "avg_delta_wins":      _lq.get("avg_delta_wins"),
            "avg_delta_losses":    _lq.get("avg_delta_losses"),
            "n_wins_with_delta":   _lq.get("n_wins_with_delta", 0),
            "n_losses_with_delta": _lq.get("n_losses_with_delta", 0),
            "real_loss_pct":       _lq.get("real_loss_pct"),
            "clv_win_rate_pct":    _lq.get("clv_win_rate_pct"),
            "clv_legs_evaluated":  _lq.get("clv_legs_evaluated", 0),
            "legs_evaluated":      _lq.get("n", 0),
            "calibration_ready":   _lq.get("calibration_ready", False),
            "interpretation":      _lq.get("interpretation"),
            "diagnosis":           _lq.get("diagnosis"),
            "diagnosis_detail":    _lq.get("diagnosis_detail"),
            "recommendation":      _lq.get("recommendation"),
        }
    except Exception:
        line_quality_embed = None

    return {
        "days":             days,
        "total_settled":    total_settled,
        "total_pending":    pending_count,
        "wins":             wins,
        "win_rate":         win_rate,
        "pnl":              round(pnl, 2),
        "roi_pct":          roi_pct,
        "baseline_win_rate": baseline_win_rate,
        "trust_level":      trust_level,
        # Live vs retroactive split
        # Prospective-only P&L (excludes forced_generation / exploration — $0 stake noise)
        "prospective_settled":  len(prosp_bets),
        "prospective_wins":     prosp_wins,
        "prospective_win_rate": prosp_wr,
        "prospective_pnl":      prosp_pnl,
        "prospective_roi_pct":  prosp_roi,
        # Live vs retroactive split
        "live_settled":     len(live_bets),
        "live_wins":        live_wins,
        "live_win_rate":    live_wr,
        "live_trust_level": live_trust,
        "retro_settled":    len(retro_bets),
        "retro_wins":       retro_wins,
        "retro_win_rate":   retro_wr,
        "by_confidence":    [
            {"level": k, "bets": v["bets"], "win_rate": _pct(v["wins"], v["bets"])}
            for k, v in conf_stats.items()
        ],
        "by_sport": [
            {"sport": k, "bets": v["bets"], "win_rate": _pct(v["wins"], v["bets"])}
            for k, v in sport_stats.items()
        ],
        "by_model": [
            {"model_used": k, "bets": v["bets"],
             "win_rate": _pct(v["wins"], v["bets"]), "auc": v["auc"]}
            for k, v in model_stats.items()
        ],
        "by_lqs":                by_lqs,
        "daily_trend":           trend,
        "retro_trend":           retro_trend,
        "active_backfills":      active_jobs,
        # Alt-line performance tracking
        "alt_line_leg_pct":      alt_line_leg_pct,
        "alt_line_win_rate":     alt_line_win_rate,
        "main_line_win_rate":    main_line_win_rate,
        # Embedded line quality (compact — no per-sport breakdown)
        "line_quality":          line_quality_embed,
    }


# ─── Line quality summary (API) ───────────────────────────────────────────────

# Date from which line quality metrics are displayed.
# April 29 2026 = first day CUSHION/AVOID margin grades + personal_edge_profile were live.
# Earlier bets still exist and feed historical learning (Component A), but are excluded
# from the performance display so the card reflects only the fully-configured system.
SYSTEM_LAUNCH_DATE = "2026-04-29"

# Ordered list of simulation sources — same set as signal_analysis.SIM_SOURCES.
_LQ_SIM_SOURCES = (
    "prospective", "prospective_pm", "top_picks_page",
    "forced_generation", "retroactive_mock", "prospective_legacy", "scenario_sim",
)

# Exploration bets are single-leg adjacent-line probes generated alongside Section A picks.
# They settle automatically (source='exploration') but are EXCLUDED from _LQ_SIM_SOURCES so
# they never contaminate T1/T2/T3 or the main line quality metrics.  Tracked separately.
EXPLORATION_SOURCES = ("exploration",)

# Section A sources whose CUSHION legs seed exploration bets each morning.
_SECTION_A_SOURCES = (
    "prospective", "prospective_pm", "forced_generation", "top_picks_page",
)


def get_line_quality_summary(db: Session, since: str | None = None) -> dict:
    """
    Aggregate line quality metrics from settled mock_bet_legs.

    since: ISO date string floor (default: SYSTEM_LAUNCH_DATE = 2026-04-29).
           Only bets generated on or after this date are included so the card
           reflects performance under the fully-configured CUSHION/AVOID system.

    Historical data before this date continues to feed model learning (Component A)
    but is excluded here so the display shows only post-launch performance.

    Returns overall directional accuracy, avg line delta, and A/B recovery
    rate, broken down by sport/market.
    """
    since_str = since or SYSTEM_LAUNCH_DATE
    src_placeholders = ",".join("?" * len(_LQ_SIM_SOURCES))
    try:
        con = sqlite3.connect(_BETS_DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            f"""
            SELECT
                ml.sport,
                ml.market_type,
                ml.leg_result,
                ml.direction_correct,
                ml.line_delta,
                ml.ab_alt_result,
                ml.ab_alt_odds,
                ml.main_market_line,
                ml.main_market_result,
                mb.status AS bet_status
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND ml.leg_result IN ('WIN', 'LOSS')
              AND ml.direction_correct IS NOT NULL
              AND mb.source IN ({src_placeholders})
              AND date(mb.generated_at) >= ?
            ORDER BY mb.generated_at
            """,
            list(_LQ_SIM_SOURCES) + [since_str],
        ).fetchall()

        # ── Exploration stats (separate query, NOT mixed into main metrics) ──
        exp_rows = con.execute(
            """
            SELECT ml.direction_correct, ml.leg_result, ml.sport, ml.market_type
            FROM   mock_bet_legs ml
            JOIN   mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE  mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND  ml.leg_result IN ('WIN', 'LOSS')
              AND  ml.direction_correct IS NOT NULL
              AND  mb.source = 'exploration'
              AND  date(mb.generated_at) >= ?
            ORDER BY mb.generated_at
            """,
            (since_str,),
        ).fetchall()
        con.close()
    except Exception as _e:
        return {"error": str(_e), "n": 0}

    if not rows:
        return {"n": 0, "insufficient_data": True}

    from collections import defaultdict

    total      = len(rows)
    dir_total  = sum(1 for r in rows if r["direction_correct"] == 1)
    losses     = [r for r in rows if r["leg_result"] == "LOSS"]
    ab_rescue  = sum(1 for r in losses if r["ab_alt_result"] == "WIN")
    deltas        = [r["line_delta"] for r in rows if r["line_delta"] is not None]
    deltas_wins   = [r["line_delta"] for r in rows if r["line_delta"] is not None and r["leg_result"] == "WIN"]
    deltas_losses = [r["line_delta"] for r in rows
                     if r["line_delta"] is not None
                     and r["leg_result"] == "LOSS"]

    # Typical line increment per sport — used to normalize delta into "steps off"
    # so that NBA +6.27 pts and MLB +0.18 pts can be compared meaningfully.
    _SPORT_INCREMENTS: dict[str, float] = {
        "NBA": 3.0, "NFL": 3.0, "NCAAB": 3.0, "NCAAF": 3.0,
        "NHL": 0.5, "MLB": 0.5,
        "EPL": 0.5, "La Liga": 0.5, "Ligue 1": 0.5,
        "Serie A": 0.5, "Bundesliga": 0.5, "Soccer": 0.5,
    }

    # Per-sport/market breakdown
    by_group: dict = defaultdict(lambda: {
        "n": 0, "dir_correct": 0, "losses": 0, "ab_win": 0,
        "deltas": [], "deltas_wins": [], "deltas_losses": [], "real_losses": 0,
    })
    for r in rows:
        sport  = (r["sport"] or "Unknown").split("|")[0].strip()
        market = (r["market_type"] or "unknown").lower()
        # Simplify market key for readability
        if "total" in market:
            mkey = "total"
        elif "spread" in market:
            mkey = "spread"
        else:
            mkey = market
        key = f"{sport} / {mkey}"
        g = by_group[key]
        g["n"] += 1
        if r["direction_correct"] == 1:
            g["dir_correct"] += 1
        if r["leg_result"] == "LOSS":
            g["losses"] += 1
            if r["ab_alt_result"] == "WIN":
                g["ab_win"] += 1
            else:
                g["real_losses"] += 1
        if r["line_delta"] is not None:
            g["deltas"].append(r["line_delta"])
            if r["leg_result"] == "WIN":
                g["deltas_wins"].append(r["line_delta"])
            else:
                g["deltas_losses"].append(r["line_delta"])

    def _avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else None

    breakdown = []
    for key, g in sorted(by_group.items(), key=lambda x: -x[1]["n"]):
        n = g["n"]
        _sport_key = key.split(" / ")[0].strip()
        _inc = _SPORT_INCREMENTS.get(_sport_key, 1.0)
        _ad  = _avg(g["deltas"])
        _adw = _avg(g["deltas_wins"])
        _adl = _avg(g["deltas_losses"])
        breakdown.append({
            "group":                  key,
            "n":                      n,
            "dir_acc_pct":            round(g["dir_correct"] / n * 100, 1) if n else None,
            "avg_delta":              _ad,
            "avg_delta_wins":         _adw,
            "avg_delta_losses":       _adl,
            "n_wins_delta":           len(g["deltas_wins"]),
            "n_losses_delta":         len(g["deltas_losses"]),
            "norm_delta":             round(_ad / _inc, 2) if _ad is not None else None,
            "norm_delta_wins":        round(_adw / _inc, 2) if _adw is not None else None,
            "norm_delta_losses":      round(_adl / _inc, 2) if _adl is not None else None,
            "sport_increment":        _inc,
            "ab_win_pct":             round(g["ab_win"] / g["losses"] * 100, 1) if g["losses"] else None,
            "real_loss_pct":          round(g["real_losses"] / n * 100, 1) if n else None,
        })

    dir_acc          = round(dir_total / total * 100, 1) if total else None
    avg_delta        = _avg(deltas)
    avg_delta_wins   = _avg(deltas_wins)
    avg_delta_losses = _avg(deltas_losses)
    ab_pct           = round(ab_rescue / len(losses) * 100, 1) if losses else None
    real_loss        = round((len(losses) - ab_rescue) / total * 100, 1) if total else None

    # ── Diagnosis: win/loss delta split reveals root cause ───────────────────
    # Sign convention for line_delta = our_line - optimal_line:
    #   Wins:   negative = tight/aggressive (extracting value — GOOD)
    #           positive = conservative cushion (safe, may leave EV)
    #   Losses: large positive = catastrophic blowout / wrong direction
    #           small positive = line was too tight but direction correct

    def _diagnose(avg_w, avg_l, dir_acc_, n_w, n_l):
        if avg_w is None:
            return ("Insufficient data", "Need more settled bets with delta tracking.", "")
        if avg_l is None or n_l == 0:
            return ("Wins only", f"Wins ({avg_w:+.2f}): well-priced. No loss-delta data yet.", "")

        if avg_l >= 3.0:
            diag = "Direction errors driving losses"
            detail = (
                f"Wins ({avg_w:+.2f}): ALE handling line selection correctly — "
                f"tight/aggressive lines on {n_w} winning legs.\n"
                f"Losses ({avg_l:+.2f}): catastrophic blowout magnitude across {n_l} losing legs "
                f"— model is picking wrong direction on uncertain matchups, not a line calibration issue.\n"
                f"Direction accuracy {dir_acc_:.1f}% confirms: the problem is directional edge, not line selection."
            )
            rec = (
                "Tighten CUSHION grade requirements; add abandon-if-uncertain gate "
                "(exclude legs where model_win_prob < 60%); improve regime classification "
                "to avoid uncertain matchups."
            )
        elif avg_l >= 1.0 and avg_w <= -1.0:
            diag = "Both wins and losses miscalibrated"
            detail = (
                f"Wins ({avg_w:+.2f}): leaving EV on table — too conservative on confident picks.\n"
                f"Losses ({avg_l:+.2f}): too aggressive on uncertain picks.\n"
                f"Direction accuracy {dir_acc_:.1f}%."
            )
            rec = (
                "Per-market calibration offset: push aggressive when CUSHION grade, "
                "pull conservative when CLOSE/MIXED."
            )
        elif avg_w >= -0.5 and avg_l <= 1.0:
            diag = "Line calibration on track"
            detail = (
                f"Wins ({avg_w:+.2f}) and losses ({avg_l:+.2f}) both within optimal range. "
                f"Continue current ALE strategy. Direction accuracy {dir_acc_:.1f}%."
            )
            rec = "Maintain current line selection parameters."
        elif avg_w <= -1.5:
            diag = "Leaving EV on the table — too conservative on wins"
            detail = (
                f"Wins ({avg_w:+.2f}): consistently winning with 1.5+ unit cushion — "
                f"ALE could push one step more aggressive.\n"
                f"Losses ({avg_l:+.2f}): acceptable."
            )
            rec = (
                "Increase aggression on CUSHION-grade legs: one step closer to main market."
            )
        else:
            diag = "Mixed signal — build sample"
            detail = (
                f"Wins {avg_w:+.2f} (n={n_w}), losses {avg_l:+.2f} (n={n_l}). "
                f"Direction accuracy {dir_acc_:.1f}%. "
                f"Wait for n≥300 before adjusting strategy."
            )
            rec = "No action yet — continue monitoring."

        return diag, detail, rec

    _n_w = len(deltas_wins)
    _n_l = len(deltas_losses)
    diagnosis, diagnosis_detail, recommendation = _diagnose(
        avg_delta_wins, avg_delta_losses, dir_acc or 0.0, _n_w, _n_l
    )

    # Legacy single-line interpretation (kept for existing callers)
    if dir_acc is not None and avg_delta is not None:
        interpretation = f"{dir_acc:.1f}% direction accuracy. {diagnosis}: {diagnosis_detail[:120]}..."
    else:
        interpretation = "Insufficient data for interpretation."

    # ── Exploration stats ─────────────────────────────────────────────────────
    exp_n = len(exp_rows)
    exp_dir_acc_pct = None
    exp_win_rate    = None
    exp_breakdown: list[dict] = []
    if exp_n > 0:
        exp_dir_correct = sum(1 for r in exp_rows if r["direction_correct"] == 1)
        exp_wins        = sum(1 for r in exp_rows if r["leg_result"] == "WIN")
        exp_dir_acc_pct = round(exp_dir_correct / exp_n * 100, 1)
        exp_win_rate    = round(exp_wins / exp_n * 100, 1)
        # Breakdown by sport/market
        from collections import defaultdict as _dd
        eg: dict = _dd(lambda: {"n": 0, "dir_correct": 0, "wins": 0})
        for r in exp_rows:
            sport_  = (r["sport"] or "Unknown").split("|")[0].strip()
            mkt_    = (r["market_type"] or "").lower()
            mkey_   = "total" if "total" in mkt_ else "spread" if "spread" in mkt_ else mkt_
            g_ = eg[f"{sport_} / {mkey_}"]
            g_["n"] += 1
            if r["direction_correct"] == 1:
                g_["dir_correct"] += 1
            if r["leg_result"] == "WIN":
                g_["wins"] += 1
        for key_, g_ in sorted(eg.items(), key=lambda x: -x[1]["n"]):
            n_ = g_["n"]
            exp_breakdown.append({
                "group":       key_,
                "n":           n_,
                "dir_acc_pct": round(g_["dir_correct"] / n_ * 100, 1) if n_ else None,
                "win_rate":    round(g_["wins"]        / n_ * 100, 1) if n_ else None,
            })

    # ── CLV proxy: % of legs where we beat the closing line ─────────────────
    # clv_cents = close_odds (American) - open_odds (American).
    # In American format: numerically higher = more favorable payout.
    # open_odds > close_odds (clv_cents < 0) → we got BETTER odds than close = positive CLV.
    clv_legs_evaluated = 0
    clv_legs_positive  = 0
    try:
        _con_clv = sqlite3.connect(_BETS_DB_PATH)
        _con_clv.row_factory = sqlite3.Row
        _clv_rows = _con_clv.execute(
            f"""
            SELECT ml.clv_cents, ml.clv_available
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE ml.clv_available = 1
              AND mb.source IN ({src_placeholders})
              AND date(mb.generated_at) >= ?
            """,
            list(_LQ_SIM_SOURCES) + [since_str],
        ).fetchall()
        _con_clv.close()
        clv_legs_evaluated = len(_clv_rows)
        # clv_cents < 0 → open_odds > close_odds → positive CLV (beat the line)
        clv_legs_positive  = sum(1 for r in _clv_rows if (r["clv_cents"] or 0) < 0)
    except Exception:
        pass

    clv_win_rate_pct = (
        round(clv_legs_positive / clv_legs_evaluated * 100, 1)
        if clv_legs_evaluated > 0 else None
    )

    return {
        "n":                   total,
        "since":               since_str,
        "dir_accuracy_pct":    dir_acc,
        # Line delta — aggregate + win/loss split
        # avg_delta_losses = direction-correct losses only (direction errors excluded)
        "avg_line_delta":      avg_delta,
        "avg_delta_wins":      avg_delta_wins,
        "avg_delta_losses":    avg_delta_losses,
        "n_wins_with_delta":   len(deltas_wins),
        "n_losses_with_delta": len(deltas_losses),   # dir-correct losses only
        "ab_recovery_pct":     ab_pct,
        "real_loss_pct":       real_loss,
        "breakdown":           breakdown,
        "interpretation":      interpretation,
        "diagnosis":           diagnosis,
        "diagnosis_detail":    diagnosis_detail,
        "recommendation":      recommendation,
        "calibration_ready":   total >= 20,
        # CLV proxy — strongest long-term edge predictor
        "clv_legs_evaluated": clv_legs_evaluated,
        "clv_legs_positive":  clv_legs_positive,
        "clv_win_rate_pct":   clv_win_rate_pct,   # target > 50%
        # Exploration bets — tracked separately, excluded from main metrics
        "exploration_n":           exp_n,
        "exploration_dir_acc_pct": exp_dir_acc_pct,
        "exploration_win_rate":    exp_win_rate,
        "exploration_breakdown":   exp_breakdown,
    }


# ─── True CLV summary ────────────────────────────────────────────────────────

def get_clv_summary(db: Session, days: int = 30) -> dict:
    """Return a dedicated CLV (Closing Line Value) summary.

    CLV sign convention:
        clv_cents = close_odds - open_odds  (American format)
        clv_cents < 0  →  open_odds > close_odds  →  POSITIVE CLV (beat the close)
        clv_cents > 0  →  open_odds < close_odds  →  NEGATIVE CLV (line moved against us)

    Returns overall beat-close rate, average clv_cents, per-market breakdown,
    and win-rate conditioned on positive vs negative CLV.
    """
    since_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    # CLV is only meaningful for real-time picks (retroactive/legacy have no live close_odds)
    _CLV_SOURCES = ("prospective", "prospective_pm", "forced_generation", "top_picks_page")
    src_ph = ", ".join("?" * len(_CLV_SOURCES))

    try:
        con = sqlite3.connect(_BETS_DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            f"""
            SELECT
                ml.clv_cents,
                ml.clv_available,
                ml.market_type,
                ml.sport,
                ml.leg_result
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE ml.clv_available = 1
              AND mb.source IN ({src_ph})
              AND date(mb.generated_at) >= ?
            """,
            list(_CLV_SOURCES) + [since_str],
        ).fetchall()
        con.close()
    except Exception as _e:
        return {"error": str(_e), "n": 0}

    n_total       = len(rows)
    n_positive    = 0  # beat the close (clv_cents < 0)
    n_negative    = 0  # line moved against us (clv_cents > 0)
    n_neutral     = 0  # clv_cents == 0
    sum_clv       = 0

    # Win rates conditioned on CLV direction (only settled legs)
    pos_wins = pos_settled = 0
    neg_wins = neg_settled = 0

    # Per-market breakdown
    mkt_stats: dict[str, dict] = {}

    for r in rows:
        cents     = r["clv_cents"] or 0
        market    = (r["market_type"] or "Unknown").strip()
        result    = r["leg_result"]  # WIN / LOSS / None (unsettled)
        sum_clv  += cents

        if cents < 0:
            n_positive += 1
            if result == "WIN":
                pos_wins    += 1
                pos_settled += 1
            elif result == "LOSS":
                pos_settled += 1
        elif cents > 0:
            n_negative += 1
            if result == "WIN":
                neg_wins    += 1
                neg_settled += 1
            elif result == "LOSS":
                neg_settled += 1
        else:
            n_neutral += 1

        bkt = mkt_stats.setdefault(market, {
            "n": 0, "positive": 0, "negative": 0, "neutral": 0,
            "sum_clv": 0, "wins": 0, "settled": 0,
        })
        bkt["n"]        += 1
        bkt["sum_clv"]  += cents
        if cents < 0:
            bkt["positive"] += 1
        elif cents > 0:
            bkt["negative"] += 1
        else:
            bkt["neutral"]  += 1
        if result == "WIN":
            bkt["wins"]    += 1
            bkt["settled"] += 1
        elif result == "LOSS":
            bkt["settled"] += 1

    beat_close_pct = round(n_positive / n_total * 100, 1) if n_total else None
    avg_clv        = round(sum_clv / n_total, 1) if n_total else None

    # Win rates
    pos_wr = round(pos_wins / pos_settled * 100, 1) if pos_settled >= 5 else None
    neg_wr = round(neg_wins / neg_settled * 100, 1) if neg_settled >= 5 else None

    breakdown = []
    for mkt, s in sorted(mkt_stats.items(), key=lambda x: -x[1]["n"]):
        n_ = s["n"]
        breakdown.append({
            "market":          mkt,
            "n":               n_,
            "beat_close_pct":  round(s["positive"] / n_ * 100, 1) if n_ else None,
            "avg_clv_cents":   round(s["sum_clv"] / n_, 1) if n_ else None,
            "win_rate":        round(s["wins"] / s["settled"] * 100, 1) if s["settled"] >= 3 else None,
            "n_settled":       s["settled"],
        })

    # CLV interpretation
    note = "strong edge retention" if (beat_close_pct or 0) >= 58 else (
           "moderate edge retention" if (beat_close_pct or 0) >= 52 else
           "line movement is neutral or adverse — revisit model timing"
    )

    return {
        "n":                    n_total,
        "since":                since_str,
        "beat_close_pct":       beat_close_pct,    # target > 50%
        "avg_clv_cents":        avg_clv,            # negative = positive CLV
        "n_positive_clv":       n_positive,
        "n_negative_clv":       n_negative,
        "n_neutral_clv":        n_neutral,
        # Win rates by CLV direction (min 5 settled legs)
        "win_rate_positive_clv": pos_wr,            # WR when we beat the close
        "win_rate_negative_clv": neg_wr,            # WR when line moved against us
        "n_positive_settled":   pos_settled,
        "n_negative_settled":   neg_settled,
        "breakdown":            breakdown,
        "interpretation":       note,
    }


# ─── Training signal ──────────────────────────────────────────────────────────

def generate_mock_training_signal(db: Session, min_settled: int = 50) -> dict:
    """
    Convert settled mock bets into feature rows suitable for retraining.

    Returns:
        {"rows": int, "ready": bool, "message": str}
    """
    settled = db.query(MockBet).filter(
        MockBet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()

    if len(settled) < min_settled:
        return {
            "rows":    len(settled),
            "ready":   False,
            "message": f"Need {min_settled} settled mock bets — have {len(settled)}.",
        }

    # Build minimal feature dicts (compatible with ml_model.train format)
    import math
    rows = []
    for m in settled:
        rows.append({
            "legs":         m.legs or 1,
            "odds":         m.odds or 2.0,
            "log_odds":     math.log(m.odds) if (m.odds or 1) > 1 else 0,
            "implied_prob": 1 / (m.odds or 2),
            "stake":        m.amount or 10.0,
            "is_parlay":    1 if (m.legs or 1) > 1 else 0,
            "win_prob":     m.predicted_win_prob or 0.5,
            "is_mock":      1,
            "label":        1 if m.status == "SETTLED_WIN" else 0,
        })

    return {
        "rows":    len(rows),
        "ready":   True,
        "message": f"{len(rows)} mock bets ready as training signal.",
        "data":    rows,
    }


# ─── Retroactive backfill (System 3F) ────────────────────────────────────────

def _hist_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_HIST_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _get_historical_games_for_date(target_date: str) -> list[dict]:
    """
    Pull settled games from historical.db for target_date (YYYY-MM-DD).
    Returns list of dicts with game_id, sport, home_team, away_team,
    home_score, away_score, close_spread, covered_pl.
    """
    sql = """
        SELECT g.game_id, g.sport, g.game_date, g.home_team, g.away_team,
               g.home_score, g.away_score,
               bl.close_spread, bl.covered_pl,
               bl.open_ml_home, bl.open_ml_away
        FROM games g
        LEFT JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.game_date = ?
          AND g.sport IN ('NHL', 'MLB')
          AND g.home_score IS NOT NULL
          AND g.home_score > 0
    """
    try:
        conn = _hist_conn()
        rows = conn.execute(sql, (target_date,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _score_retro_game(sport: str, home_team: str, away_team: str,
                      game_date: str) -> Optional[float]:
    """
    Score a historical game using the current sub-model.
    Returns probability that home team covers ATS, or None if unavailable.
    """
    try:
        import ml_model as ml
        prob = ml.predict_game_ats(
            sport=sport, home_team=home_team, away_team=away_team,
            game_date=game_date,
        )
        return prob
    except Exception:
        return None


def _settle_retro_leg(game: dict, home_covers: bool) -> bool:
    """
    Determine if a leg won given the actual outcome.
    home_covers=True  → bet is "home team covers" → win if covered_pl=1 (NHL)
                                                       or home_score > away_score (MLB)
    home_covers=False → bet is "away team covers" → opposite
    """
    sport = game.get("sport", "")
    if sport == "NHL" and game.get("covered_pl") is not None:
        covered = bool(game["covered_pl"])
    else:
        # MLB or no spread: use ML outcome
        covered = (game["home_score"] or 0) > (game["away_score"] or 0)
    return covered if home_covers else not covered


def _build_retro_decimal_odds(home_covers: bool, game: dict) -> float:
    """Derive decimal odds from ml lines; default to -110 (1.909)."""
    try:
        if home_covers:
            am = game.get("open_ml_home") or -110
        else:
            am = game.get("open_ml_away") or -110
        am = int(am)
        return round(1 + am / 100, 4) if am > 0 else round(1 + 100 / abs(am), 4)
    except Exception:
        return 1.909


def generate_retroactive_mock_bets(
    db:            Session,
    lookback_days: int = 180,
    n_per_day:     int = 30,
    job_id:        Optional[str] = None,
    parlay_size:   int = 3,
    stake:         float = 10.0,
) -> dict:
    """
    Generate and immediately settle mock parlays on historical game data.

    For each date in [today - lookback_days, yesterday]:
      1. Pull settled games from historical.db
      2. Score each game using the current sub-model
      3. Keep legs where |prob - 0.5| >= _RETRO_CONF_THRESHOLD - 0.5
      4. Build up to n_per_day random parlays of parlay_size legs
      5. Settle each parlay immediately (all legs must win)
      6. Store with source='retroactive_mock', weight=0.25

    Skips dates with < 5 games or < parlay_size confident legs.

    Accepted bias: current model features include games after target_date
    (lookahead). Retroactive rows are discounted to weight=0.25 at retrain time.

    Returns summary dict and updates _backfill_jobs[job_id] in real time.
    """
    run_id = f"retro_{datetime.utcnow().strftime('%Y%m%d_%H%M')}_{uuid.uuid4().hex[:6]}"

    if job_id:
        with _backfill_lock:
            _backfill_jobs[job_id].update({
                "status":      "running",
                "processed":   0,
                "total":       lookback_days,
                "generated":   0,
                "settled":     0,
                "skipped":     0,
                "run_id":      run_id,
            })

    today      = _date.today()
    start_date = today - timedelta(days=lookback_days)

    total_generated = 0
    total_settled   = 0
    dates_processed = 0
    dates_skipped   = 0

    for offset in range(lookback_days):
        target_date = start_date + timedelta(days=offset)
        if target_date >= today:
            break

        date_str = target_date.isoformat()

        if job_id:
            with _backfill_lock:
                _backfill_jobs[job_id]["processed"] = offset + 1
                _backfill_jobs[job_id]["generated"] = total_generated
                _backfill_jobs[job_id]["settled"]   = total_settled
                _backfill_jobs[job_id]["skipped"]   = dates_skipped

        # ── 1. Pull games for this date ──────────────────────────────────────
        games = _get_historical_games_for_date(date_str)
        if len(games) < 5:
            dates_skipped += 1
            continue

        # ── 2. Score each game ───────────────────────────────────────────────
        scored: list[dict] = []
        for g in games:
            prob = _score_retro_game(
                sport=g["sport"], home_team=g["home_team"],
                away_team=g["away_team"], game_date=date_str,
            )
            if prob is None:
                continue
            home_covers = prob >= 0.5
            conf = abs(prob - 0.5) * 2   # 0..1
            if conf < ((_RETRO_CONF_THRESHOLD - 0.5) * 2):
                continue
            scored.append({
                "game":        g,
                "prob":        prob,
                "home_covers": home_covers,
                "conf":        conf,
                "odds":        _build_retro_decimal_odds(home_covers, g),
            })

        if len(scored) < parlay_size:
            dates_skipped += 1
            continue

        # ── 3. Build parlays ─────────────────────────────────────────────────
        # Sort by confidence descending; sample without replacement
        scored.sort(key=lambda x: -x["conf"])
        pool    = scored[:min(len(scored), 15)]   # top 15 confident legs as pool
        n_build = min(n_per_day, max(1, len(pool) // parlay_size * 2))

        built_combos: set[frozenset] = set()
        day_generated = 0

        for _ in range(n_build * 3):   # attempts
            if day_generated >= n_per_day:
                break
            if len(pool) < parlay_size:
                break
            legs = random.sample(pool, parlay_size)
            key  = frozenset(l["game"]["game_id"] for l in legs)
            if key in built_combos:
                continue
            built_combos.add(key)

            # Combined odds (multiply leg odds)
            combined_odds = 1.0
            for leg in legs:
                combined_odds *= leg["odds"]
            combined_odds = round(combined_odds, 4)

            # Parlay win prob (product of leg probs — each prob is P(home covers))
            parlay_win_prob = 1.0
            for leg in legs:
                p = leg["prob"] if leg["home_covers"] else (1 - leg["prob"])
                parlay_win_prob *= p
            parlay_win_prob = round(parlay_win_prob, 4)

            bet_info = " | ".join(
                f"{'Home' if l['home_covers'] else 'Away'} cover — "
                f"{l['game']['home_team']} vs {l['game']['away_team']}"
                for l in legs
            )
            sport_set = {l["game"]["sport"] for l in legs}

            # Average per-leg confidence (distance from 0.5, normalised to 0-1)
            # This is the meaningful signal — parlay product is always tiny for 3-leg
            avg_leg_conf = sum(
                leg["prob"] if leg["home_covers"] else (1 - leg["prob"])
                for leg in legs
            ) / len(legs)

            # Confidence label uses per-leg avg, not parlay product
            if avg_leg_conf >= _CONF_HIGH:
                conf_label = "HIGH"
            elif avg_leg_conf >= _CONF_MEDIUM:
                conf_label = "MEDIUM"
            else:
                conf_label = "LOW"

            # AUC: look up by dominant sport
            from collections import Counter as _Counter
            dom_sport = _Counter(l["game"]["sport"] for l in legs).most_common(1)[0][0]
            auc_val   = _AUC_BY_SPORT.get(dom_sport, _AUC_COMBINED)

            mock_id = str(uuid.uuid4())
            mock = MockBet(
                id                     = mock_id,
                generation_run_id      = run_id,
                game_date              = date_str,
                generated_at           = datetime.combine(target_date, datetime.min.time()),
                sport                  = " | ".join(sport_set),
                bet_type               = "parlay",
                odds                   = combined_odds,
                amount                 = stake,
                legs                   = parlay_size,
                bet_info               = bet_info,
                predicted_win_prob     = parlay_win_prob,
                confidence             = conf_label,
                model_confidence_avg   = round(avg_leg_conf, 4),
                model_used             = "retroactive_submodel",
                model_auc              = auc_val,
                source                 = "retroactive_mock",
                weight                 = 0.25,
                status                 = "PENDING",  # will settle below
            )
            db.add(mock)

            for i, leg in enumerate(legs):
                db.add(MockBetLeg(
                    mock_bet_id = mock_id,
                    leg_index   = i,
                    description = f"{'Home' if leg['home_covers'] else 'Away'} cover — "
                                  f"{leg['game']['home_team']} vs {leg['game']['away_team']}",
                    market_type = "Spread" if leg["game"].get("close_spread") else "Moneyline",
                    sport       = leg["game"]["sport"],
                    win_prob    = leg["prob"],
                ))

            # ── 4. Settle immediately ────────────────────────────────────────
            all_win = all(
                _settle_retro_leg(l["game"], l["home_covers"]) for l in legs
            )
            mock.status     = "SETTLED_WIN" if all_win else "SETTLED_LOSS"
            mock.settled_at = datetime.combine(target_date, datetime.min.time())
            mock.actual_profit = round(
                stake * (combined_odds - 1) if all_win else -stake, 2
            )

            day_generated   += 1
            total_generated += 1
            if all_win:
                total_settled += 1   # repurpose: count wins for progress

        dates_processed += 1
        # Commit each day to avoid giant transactions
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[Retro] Commit error on {date_str}: {e}")

    summary = {
        "generated":       total_generated,
        "wins":            total_settled,
        "dates_processed": dates_processed,
        "dates_skipped":   dates_skipped,
        "run_id":          run_id,
    }

    if job_id:
        with _backfill_lock:
            _backfill_jobs[job_id].update({
                "status":        "completed",
                "processed":     lookback_days,
                "generated":     total_generated,
                "settled":       total_settled,
                "completed_at":  datetime.utcnow().isoformat(),
                **summary,
            })

    return summary


def start_backfill_job(
    lookback_days: int = 180,
    n_per_day:     int = 30,
) -> str:
    """
    Launch generate_retroactive_mock_bets() in a background thread.
    Returns a job_id to poll via get_backfill_job_status().
    """
    job_id = f"backfill_{uuid.uuid4().hex[:8]}"
    with _backfill_lock:
        _backfill_jobs[job_id] = {
            "status":      "queued",
            "processed":   0,
            "total":       lookback_days,
            "generated":   0,
            "settled":     0,
            "skipped":     0,
            "started_at":  datetime.utcnow().isoformat(),
            "completed_at": None,
            "error":       None,
        }

    def _run():
        from database import SessionLocal, init_db
        init_db()
        db = SessionLocal()
        try:
            generate_retroactive_mock_bets(
                db,
                lookback_days = lookback_days,
                n_per_day     = n_per_day,
                job_id        = job_id,
            )
        except Exception as exc:
            with _backfill_lock:
                _backfill_jobs[job_id]["status"] = "error"
                _backfill_jobs[job_id]["error"]  = str(exc)
        finally:
            db.close()

    t = threading.Thread(target=_run, daemon=True, name=f"BackfillJob-{job_id}")
    t.start()
    return job_id


def get_backfill_job_status(job_id: str) -> Optional[dict]:
    with _backfill_lock:
        return dict(_backfill_jobs.get(job_id, {})) or None


def list_backfill_jobs() -> list[dict]:
    with _backfill_lock:
        return [{"job_id": k, **v} for k, v in _backfill_jobs.items()]


# ─── Standalone entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    from database import SessionLocal
    db = SessionLocal()
    try:
        print("[Mock] Generating mock bets…")
        result = generate_mock_bets(db)
        print(f"[Mock] Generated: {result}")

        print("[Mock] Settling pending mock bets…")
        settle = settle_mock_bets(db)
        print(f"[Mock] Settled: {settle}")

        print("[Mock] Performance summary:")
        perf = get_mock_performance(db)
        print(f"  Win rate: {perf['win_rate']}%  Trust: {perf['trust_level']}")
    finally:
        db.close()
