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

import json
import logging
import os
import random
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, date as _date, timezone
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_CT = ZoneInfo("America/Chicago")

from sqlalchemy.orm import Session

from database import MockBet, MockBetLeg, init_db, DB_WRITE_LOCK
import recommender as rec
import auto_settle as asettler
import leg_quality as lq
import placement as _plc

# ─── Backfill job registry (module-level, thread-safe via GIL for dict ops) ───
_backfill_jobs: dict[str, dict] = {}
_backfill_lock = threading.Lock()

# ─── Bet category classification ──────────────────────────────────────────────
# All known team-bet market types (OddsAPI keys + legacy capitalized forms).
# Any market_type NOT in this set is treated as a player prop.
_TEAM_MARKET_KEYS: frozenset[str] = frozenset({
    "h2h", "spreads", "spread", "totals", "total",
    "alternate_spreads", "alternate_totals",
    "moneyline",   # legacy capitalized form stored as-is
})


def classify_bet_category(market_type: Optional[str]) -> str:
    """Return 'player_prop' or 'team' based on market_type string."""
    if not market_type:
        return "team"
    return "team" if market_type.lower() in _TEAM_MARKET_KEYS else "player_prop"


import re as _re
_SPREAD_PARSE_RE = _re.compile(r'([+-]\d+(?:\.\d+)?)')


def _parse_spread_for_leg(description: str, market_type: str) -> tuple:
    """
    Extract (spread_value, spread_abs, is_favorite_spread) from a leg.
    Returns (None, None, None) for non-spread markets.
    """
    if "spread" not in (market_type or "").lower():
        return None, None, None
    matches = _SPREAD_PARSE_RE.findall(description or "")
    if matches:
        sv = float(matches[-1])
        return sv, abs(sv), (1 if sv < 0 else 0)
    return None, None, None


# ── Tier classification ──────────────────────────────────────────────────────
# Markets validated via Brier < 0.20 or sustained edge over 50+ settled legs.
# Sport values use the short labels stored in mock_bet_legs.sport.
# Alt markets (alternate_spreads → spreads) are normalised before lookup.
PROVEN_MARKETS: set[tuple[str, str]] = {
    ("MLB", "spreads"),   ("MLB", "totals"),   ("MLB", "h2h"),
    ("NBA", "spreads"),   ("NBA", "totals"),   ("NBA", "h2h"),
    ("NHL", "spreads"),   ("NHL", "totals"),   ("NHL", "h2h"),
    ("EPL", "spreads"),   ("EPL", "totals"),   ("EPL", "h2h"),
    ("MLS", "spreads"),   ("MLS", "totals"),   ("MLS", "h2h"),
    ("La Liga", "spreads"), ("La Liga", "totals"), ("La Liga", "h2h"),
    ("Bundesliga", "spreads"), ("Bundesliga", "totals"), ("Bundesliga", "h2h"),
    ("Serie A", "spreads"),  ("Serie A", "totals"),  ("Serie A", "h2h"),
    ("Ligue 1", "spreads"),  ("Ligue 1", "totals"),  ("Ligue 1", "h2h"),
}

def _normalise_market(market: str) -> str:
    """Map alternate market keys to their parent for PROVEN_MARKETS lookup."""
    _ALT = {"alternate_spreads": "spreads", "alternate_totals": "totals"}
    return _ALT.get(market, market)


def classify_bet_tier(pick: dict, legs: list[dict], db=None) -> tuple[str, dict]:
    """
    Classify a mock bet as T1/T2/T3 based on signal quality.

    Returns (tier_str, signals_dict) where signals_dict records which
    checks passed for transparency.

    Signal checks (5 validated signals — positive_ev and calibrated_model
    were removed after backtesting showed them anti-predictive):
      1. scout_a_or_b:      any leg has scout grade A or B   (+12.3pp lift)
      2. wp_high:           all legs have win_prob >= 60      (+18.0pp lift)
      3. no_adverse_line:   no leg has confirmed adverse line movement
      4. no_adverse_injury: no leg has confirmed adverse injury direction
      5. proven_market:     all legs are in a validated market (+10.8pp lift)

    T1 = 4+ signals met  (highest conviction)
    T2 = 3 signals met   (medium conviction)
    T3 = 0-2 signals met (monitor only)
    """
    # 1. Scout grade A or B on any leg
    scout_grades = [l.get("scout_grade") for l in legs if l.get("scout_grade")]
    scout_a_or_b = any(g in ("A", "B") for g in scout_grades)

    # 2. All legs have win_prob >= 60
    leg_wps = [float(l.get("win_prob") or 0) for l in legs]
    wp_high = all(wp >= 60.0 for wp in leg_wps) if leg_wps else False

    # 3. No adverse line movement — only explicit "adverse" fails;
    #    NULL / None / "no_data" / "neutral" / "favorable" all pass.
    line_dirs = [l.get("_line_direction") or l.get("line_direction") for l in legs]
    no_adverse_line = all(d != "adverse" for d in line_dirs)

    # 4. No adverse injury direction — same logic: only explicit "adverse" fails.
    injury_dirs = [l.get("_injury_direction") or l.get("injury_direction") for l in legs]
    no_adverse_injury = all(d != "adverse" for d in injury_dirs)

    # 5. All legs in proven markets (normalise alt markets to parent)
    leg_markets = [
        (l.get("sport", ""), _normalise_market(l.get("market_type") or l.get("market") or ""))
        for l in legs
    ]
    proven_market = all((s, m) in PROVEN_MARKETS for s, m in leg_markets) if leg_markets else False

    signals = {
        "scout_a_or_b":      scout_a_or_b,
        "wp_high":           wp_high,
        "no_adverse_line":   no_adverse_line,
        "no_adverse_injury": no_adverse_injury,
        "proven_market":     proven_market,
    }
    signals_met = sum(signals.values())

    if signals_met >= 4:
        tier = "T1"
    elif signals_met >= 3:
        tier = "T2"
    else:
        tier = "T3"

    return tier, signals


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

# ── Two-strategy boost evaluation ─────────────────────────────────────────────
# Strategy A — amplify already-positive EV: base_ev > +0.50/10
# Strategy B — rescue high-WR chalk: base_ev in [-0.50, +0.50] AND wp >= 65%
#              AND all legs grade A or B, AND boosted_ev > +1.00/10
_STRAT_A_BASE_EV_THRESHOLD  = 0.50   # base_ev_per_10 must exceed this for Strat A
_STRAT_B_MIN_WP_TIER_30     = 70.0   # +30% boost in Strat B requires higher WP
_STRAT_B_MAX_BASE_EV        = 0.50   # Strat B only applies when base_ev is marginal
_STRAT_B_MIN_BASE_EV        = -0.50  # Strat B stops rescuing deeply-negative EV
_STRAT_B_TARGET_BOOSTED_EV  = 1.00   # boosted_ev_per_10 must exceed this
_SKIP_BOOSTED_EV_MIN        = 0.30   # even with boost, skip if boosted_ev < this
_SKIP_BASE_EV_MAX           = -0.50  # skip boost entirely on genuinely losing bets
_SKIP_WP_MIN                = 0.55   # skip boost when projected_wp < 55% (structurally weak)

_NORM_STAKE = 10.0   # EV computations normalized to $10 stake throughout

# ── Runtime-adjustable Strategy B WP threshold ───────────────────────────────
# Default: 65%. Auto-tightened to 70% by health check when 30-day Strategy B
# WR drops below 60% on n >= 30 settled picks. Persists in memory until restart,
# then re-evaluated by the next daily health check.
_BOOST_STRATEGY_CONFIG: dict = {
    "strat_b_min_wp":       65.0,   # WP floor for Strategy B rescue
    "tightened":            False,   # True when auto-tightened by health check
    "tightened_at":         None,    # ISO timestamp of last tighten event
    "tighten_reason":       None,    # human-readable reason string
}
_boost_config_lock = threading.Lock()


def get_strat_b_min_wp() -> float:
    """Return the current Strategy B minimum WP (may be tightened at runtime)."""
    return _BOOST_STRATEGY_CONFIG["strat_b_min_wp"]


def tighten_strat_b_threshold(new_wp: float = 70.0, reason: str = "auto") -> dict:
    """
    Raise the Strategy B WP threshold at runtime.

    Called by the health check when 30-day Strategy B WR < 60% on n >= 30 picks.
    Persists for the session; reset to 65% on server restart (health check re-evaluates).

    Returns the updated config dict.
    """
    from datetime import datetime as _dt, timezone as _tz
    with _boost_config_lock:
        old = _BOOST_STRATEGY_CONFIG["strat_b_min_wp"]
        _BOOST_STRATEGY_CONFIG["strat_b_min_wp"]   = new_wp
        _BOOST_STRATEGY_CONFIG["tightened"]         = True
        _BOOST_STRATEGY_CONFIG["tightened_at"]      = _dt.now(_tz.utc).isoformat()
        _BOOST_STRATEGY_CONFIG["tighten_reason"]    = reason
    return dict(_BOOST_STRATEGY_CONFIG)


def reset_strat_b_threshold(reason: str = "manual") -> dict:
    """Restore Strategy B WP threshold to the default 65%."""
    from datetime import datetime as _dt, timezone as _tz
    with _boost_config_lock:
        _BOOST_STRATEGY_CONFIG["strat_b_min_wp"]   = 65.0
        _BOOST_STRATEGY_CONFIG["tightened"]         = False
        _BOOST_STRATEGY_CONFIG["tightened_at"]      = _dt.now(_tz.utc).isoformat()
        _BOOST_STRATEGY_CONFIG["tighten_reason"]    = f"reset: {reason}"
    return dict(_BOOST_STRATEGY_CONFIG)


def _compute_ev_per_10(c_odds: float, c_wp_pct: float, boost_pct: float = 0.0) -> float:
    """Compute EV per $10 stake given decimal odds, win probability %, and optional boost."""
    wp     = c_wp_pct / 100.0
    profit = _NORM_STAKE * (c_odds - 1.0) * (1.0 + boost_pct)
    return round(wp * profit - (1.0 - wp) * _NORM_STAKE, 3)


def _evaluate_boost_strategy(
    pick: dict,
    eligible_tiers: list[float],
) -> tuple[float, str, float, float, float]:
    """
    Determine the optimal boost tier and strategy for a pick candidate.

    Evaluates both strategies and selects whichever produces the highest
    absolute boosted_ev_per_10 while satisfying the strategy gates.

    Args:
        pick: pick dict with combined_odds, combined_win_prob, legs
        eligible_tiers: list of float boost_pct values already checked for
                        FanDuel eligibility (e.g. [0.25, 0.30, 0.50])

    Returns:
        (boost_pct, strategy_label, base_ev_per_10, boosted_ev_per_10, boost_lift_ev)
        strategy_label ∈ {'amplify_positive_ev', 'rescue_high_wr_chalk', 'no_boost'}
    """
    if not eligible_tiers:
        c_odds = float(pick.get("combined_odds") or 1.0)
        c_wp   = float(pick.get("combined_win_prob") or 50.0)
        base_ev = _compute_ev_per_10(c_odds, c_wp)
        return 0.0, "no_boost", base_ev, base_ev, 0.0

    c_odds = float(pick.get("combined_odds") or 1.0)
    c_wp   = float(pick.get("combined_win_prob") or 50.0)
    base_ev = _compute_ev_per_10(c_odds, c_wp)

    # Hard skip gates — structurally weak or deeply negative
    if base_ev < _SKIP_BASE_EV_MAX or (c_wp / 100.0) < _SKIP_WP_MIN:
        return 0.0, "no_boost", base_ev, base_ev, 0.0

    # Scout grade check — Strategy B requires all legs A or B
    legs = pick.get("legs") or []
    all_high_grade = all(
        l.get("scout_grade") in ("A", "B", None)  # None = no scout run yet, allow through
        for l in legs
    )

    # Read the current (possibly tightened) Strategy B WP floor
    _strat_b_min_wp = get_strat_b_min_wp()

    best_pct      = 0.0
    best_strategy = "no_boost"
    best_boosted  = base_ev  # track best boosted_ev seen
    best_lift     = 0.0

    # Evaluate tiers highest-first
    for tier in sorted(eligible_tiers, reverse=True):
        boosted_ev = _compute_ev_per_10(c_odds, c_wp, tier)
        lift       = round(boosted_ev - base_ev, 3)

        # Skip if even with boost the edge is too thin
        if boosted_ev < _SKIP_BOOSTED_EV_MIN:
            continue

        # ── Strategy A: amplify already-positive EV ──────────────────────────
        if base_ev > _STRAT_A_BASE_EV_THRESHOLD:
            if boosted_ev > best_boosted:
                best_pct      = tier
                best_strategy = "amplify_positive_ev"
                best_boosted  = boosted_ev
                best_lift     = lift
            continue   # try higher tiers too (sorted desc, so this exits after one pass)

        # ── Strategy B: rescue high-WR chalk into +EV ────────────────────────
        # Gate: marginal base EV + high win probability + quality legs + target met
        if (
            _STRAT_B_MIN_BASE_EV <= base_ev <= _STRAT_B_MAX_BASE_EV
            and c_wp >= _strat_b_min_wp
            and all_high_grade
            and boosted_ev > _STRAT_B_TARGET_BOOSTED_EV
        ):
            # Tier-specific WP floor: +30% requires wp >= 70%; +25% insufficient for B
            if tier == 0.30 and c_wp < _STRAT_B_MIN_WP_TIER_30:
                continue
            if tier == 0.25:
                continue  # +25% almost never sufficient to rescue chalk
            if boosted_ev > best_boosted:
                best_pct      = tier
                best_strategy = "rescue_high_wr_chalk"
                best_boosted  = boosted_ev
                best_lift     = lift

    return best_pct, best_strategy, base_ev, best_boosted, best_lift


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
        # Boost strategy tracking (two-strategy framework: amplify vs rescue)
        ("mock_bets",  "base_ev_per_10",    "ALTER TABLE mock_bets ADD COLUMN base_ev_per_10 REAL"),
        ("mock_bets",  "boost_lift_ev",     "ALTER TABLE mock_bets ADD COLUMN boost_lift_ev REAL"),
        ("mock_bets",  "boosted_ev_per_10", "ALTER TABLE mock_bets ADD COLUMN boosted_ev_per_10 REAL"),
        # Bet category: 'team' | 'player_prop' — separates team-level from player prop tracking
        ("mock_bet_legs", "bet_category",   "ALTER TABLE mock_bet_legs ADD COLUMN bet_category TEXT"),
        # Bet tier: T1/T2/T3 conviction classification
        ("mock_bets",     "bet_tier",       "ALTER TABLE mock_bets ADD COLUMN bet_tier TEXT"),
        # Spread features for model training (alt_spreads / spreads)
        ("mock_bet_legs", "spread_value",        "ALTER TABLE mock_bet_legs ADD COLUMN spread_value REAL"),
        ("mock_bet_legs", "spread_abs",           "ALTER TABLE mock_bet_legs ADD COLUMN spread_abs REAL"),
        ("mock_bet_legs", "is_favorite_spread",   "ALTER TABLE mock_bet_legs ADD COLUMN is_favorite_spread INTEGER"),
        # ── MLB enrichment features for model training (shipped 2026-06-06) ──
        # Pitcher quality
        ("mock_bet_legs", "home_starter_era",     "ALTER TABLE mock_bet_legs ADD COLUMN home_starter_era REAL"),
        ("mock_bet_legs", "home_starter_whip",    "ALTER TABLE mock_bet_legs ADD COLUMN home_starter_whip REAL"),
        ("mock_bet_legs", "home_starter_k9",      "ALTER TABLE mock_bet_legs ADD COLUMN home_starter_k9 REAL"),
        ("mock_bet_legs", "home_starter_hr9",     "ALTER TABLE mock_bet_legs ADD COLUMN home_starter_hr9 REAL"),
        ("mock_bet_legs", "away_starter_era",     "ALTER TABLE mock_bet_legs ADD COLUMN away_starter_era REAL"),
        ("mock_bet_legs", "away_starter_whip",    "ALTER TABLE mock_bet_legs ADD COLUMN away_starter_whip REAL"),
        ("mock_bet_legs", "away_starter_k9",      "ALTER TABLE mock_bet_legs ADD COLUMN away_starter_k9 REAL"),
        ("mock_bet_legs", "away_starter_hr9",     "ALTER TABLE mock_bet_legs ADD COLUMN away_starter_hr9 REAL"),
        ("mock_bet_legs", "starter_era_diff",     "ALTER TABLE mock_bet_legs ADD COLUMN starter_era_diff REAL"),
        ("mock_bet_legs", "starter_whip_diff",    "ALTER TABLE mock_bet_legs ADD COLUMN starter_whip_diff REAL"),
        ("mock_bet_legs", "starter_k9_diff",      "ALTER TABLE mock_bet_legs ADD COLUMN starter_k9_diff REAL"),
        # Weather
        ("mock_bet_legs", "game_temp_f",          "ALTER TABLE mock_bet_legs ADD COLUMN game_temp_f REAL"),
        ("mock_bet_legs", "game_wind_mph",        "ALTER TABLE mock_bet_legs ADD COLUMN game_wind_mph REAL"),
        ("mock_bet_legs", "wind_helps_scoring",   "ALTER TABLE mock_bet_legs ADD COLUMN wind_helps_scoring REAL"),
        # Statcast batting
        ("mock_bet_legs", "home_team_xwoba",      "ALTER TABLE mock_bet_legs ADD COLUMN home_team_xwoba REAL"),
        ("mock_bet_legs", "away_team_xwoba",      "ALTER TABLE mock_bet_legs ADD COLUMN away_team_xwoba REAL"),
        ("mock_bet_legs", "xwoba_diff",           "ALTER TABLE mock_bet_legs ADD COLUMN xwoba_diff REAL"),
        ("mock_bet_legs", "home_team_barrel_pct", "ALTER TABLE mock_bet_legs ADD COLUMN home_team_barrel_pct REAL"),
        ("mock_bet_legs", "away_team_barrel_pct", "ALTER TABLE mock_bet_legs ADD COLUMN away_team_barrel_pct REAL"),
        ("mock_bet_legs", "barrel_diff",          "ALTER TABLE mock_bet_legs ADD COLUMN barrel_diff REAL"),
        # Hard hit rate (EV95+) — was fetched and discarded, now stored
        ("mock_bet_legs", "home_team_hard_hit_pct", "ALTER TABLE mock_bet_legs ADD COLUMN home_team_hard_hit_pct REAL"),
        ("mock_bet_legs", "away_team_hard_hit_pct", "ALTER TABLE mock_bet_legs ADD COLUMN away_team_hard_hit_pct REAL"),
        ("mock_bet_legs", "hard_hit_diff",          "ALTER TABLE mock_bet_legs ADD COLUMN hard_hit_diff REAL"),
        # Bullpen quality
        ("mock_bet_legs", "home_bullpen_era",     "ALTER TABLE mock_bet_legs ADD COLUMN home_bullpen_era REAL"),
        ("mock_bet_legs", "away_bullpen_era",     "ALTER TABLE mock_bet_legs ADD COLUMN away_bullpen_era REAL"),
        ("mock_bet_legs", "bullpen_era_diff",     "ALTER TABLE mock_bet_legs ADD COLUMN bullpen_era_diff REAL"),
        # H2H matchup
        ("mock_bet_legs", "h2h_home_wr",          "ALTER TABLE mock_bet_legs ADD COLUMN h2h_home_wr REAL"),
        ("mock_bet_legs", "h2h_n_games",          "ALTER TABLE mock_bet_legs ADD COLUMN h2h_n_games INTEGER"),
        # FD consensus signal (cross-bookmaker divergence)
        ("mock_bet_legs", "fd_field_divergence",   "ALTER TABLE mock_bet_legs ADD COLUMN fd_field_divergence REAL"),
        ("mock_bet_legs", "fd_consensus_flag",     "ALTER TABLE mock_bet_legs ADD COLUMN fd_consensus_flag TEXT"),
        # 60% parlay WR floor — forward-monitoring tags
        ("mock_bets",     "would_pass_floor",      "ALTER TABLE mock_bets ADD COLUMN would_pass_floor INTEGER"),
        ("mock_bets",     "floor_margin",           "ALTER TABLE mock_bets ADD COLUMN floor_margin REAL"),
        # Form-based rung nudge (line delta build, 2026-06-14)
        ("mock_bet_legs", "form_bucket",            "ALTER TABLE mock_bet_legs ADD COLUMN form_bucket TEXT"),
        ("mock_bet_legs", "line_delta_rec_rung",    "ALTER TABLE mock_bet_legs ADD COLUMN line_delta_rec_rung REAL"),
        # Shadow B composition tracking (2026-06-27)
        ("mock_bets",     "shadow_b_comp_adj",      "ALTER TABLE mock_bets ADD COLUMN shadow_b_comp_adj REAL"),
        ("mock_bets",     "shadow_b_fav_count",     "ALTER TABLE mock_bets ADD COLUMN shadow_b_fav_count INTEGER"),
        ("mock_bets",     "shadow_b_blind_rank",    "ALTER TABLE mock_bets ADD COLUMN shadow_b_blind_rank INTEGER"),
    ]
    try:
        con = sqlite3.connect(_BETS_DB_PATH, timeout=15)
        cur = con.cursor()
        for tbl, col, stmt in migrations:
            cur.execute(f"PRAGMA table_info({tbl})")
            existing = {row[1] for row in cur.fetchall()}
            if col not in existing:
                cur.execute(stmt)
                print(f"[mock_bets] migration: added {tbl}.{col}")
        con.commit()

        # Backfill bet_category for existing rows that have NULL
        cur.execute("SELECT id, market_type FROM mock_bet_legs WHERE bet_category IS NULL")
        to_backfill = cur.fetchall()
        if to_backfill:
            for row_id, mkt in to_backfill:
                cat = classify_bet_category(mkt)
                cur.execute("UPDATE mock_bet_legs SET bet_category=? WHERE id=?", (cat, row_id))
            con.commit()
            print(f"[mock_bets] backfilled bet_category for {len(to_backfill)} legs")

        con.close()
    except Exception as _me:
        print(f"[mock_bets] migration warning: {_me}")

    # ── Normalize any T-separator datetimes (caused SQLAlchemy ORM crash) ────
    # SQLAlchemy's DateTime mapper for SQLite expects space separator, not 'T'.
    # One row was stored with isoformat() T-separator; normalize idempotently.
    try:
        con2 = sqlite3.connect(_BETS_DB_PATH, timeout=15)
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
        "alternate_spreads": "spreads", "alt spread": "spreads",
        "total": "totals",   "totals": "totals",
        "alternate_totals": "totals", "alt total": "totals",
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


def _extract_leg_line(description: str) -> float | None:
    """
    Pull the numeric line value from a spread / total description.
    Works for:
      "Minnesota Twins +1.5 (Spread)"  → 1.5
      "Dallas Stars -1.5 (Spread)"     → -1.5
      "Over 8.5"                        → 8.5
      "Under 7.5"                       → 7.5
    Returns None for moneylines.
    """
    import re as _re3
    # For totals: "Over 8.5" or "Under 7.5"
    m = _re3.search(r'\b(?:over|under)\s+([+-]?\d+(?:\.\d+)?)', description, _re3.IGNORECASE)
    if m:
        return float(m.group(1))
    # For spreads: last "[+-]N.N" before optional "(" or end
    m = _re3.search(r'([+-]\d+(?:\.\d+)?)', description)
    if m:
        return float(m.group(1))
    return None


def _normalize_iso_dt(dt_str: str | None) -> str | None:
    """Normalize ISO datetime to 'T' separator format for consistent string comparison.

    Fixes bug where commence_time uses space separator ('2026-06-12 22:41:00')
    and fetched_at uses 'T' separator ('2026-06-12T12:10:32Z'), making
    string <= comparison fail because ord('T')=84 > ord(' ')=32.
    """
    if not dt_str:
        return dt_str
    # Replace space separator with 'T' for ISO consistency
    if len(dt_str) >= 11 and dt_str[10] == ' ':
        dt_str = dt_str[:10] + 'T' + dt_str[11:]
    return dt_str


def _lookup_alt_lines_odds(
    fixture_id: str,
    description: str,
    market_type: str,
    hist_db: str,
    before_dt: str | None = None,
    earliest: bool = False,
) -> int | None:
    """
    Look up American odds from the alt_lines table (alternate_spreads / alternate_totals).

    earliest=True  → earliest snapshot (opening proxy, order ASC)
    earliest=False → latest snapshot at or before before_dt (closing proxy, order DESC)

    Returns None for h2h (no alt_lines data), and None on any lookup failure.
    """
    if not fixture_id:
        return None

    mkt_key, outcome = _parse_outcome_from_desc(description, market_type)
    if mkt_key == "h2h":
        return None   # no h2h data in alt_lines

    order      = "ASC" if earliest else "DESC"
    leg_line   = _extract_leg_line(description)
    line_sql   = ""
    line_param: list = []
    if leg_line is not None:
        # Use signed line value — a bet at +3.5 and -3.5 are different
        line_sql   = "AND ABS(line - ?) < 0.26"
        line_param = [leg_line]

    time_sql   = ""
    time_param: list = []
    if not earliest and before_dt:
        time_sql   = "AND fetched_at <= ?"
        time_param = [_normalize_iso_dt(before_dt)]

    try:
        con = sqlite3.connect(hist_db, timeout=15)
        row = None

        if mkt_key == "totals":
            ou = "over" if outcome.lower() == "over" else "under"
            for mkt in ("alternate_totals", "totals"):
                row = con.execute(
                    f"""SELECT odds FROM alt_lines
                        WHERE event_id = ? AND market_key = ? AND over_under = ?
                        {line_sql} {time_sql}
                        ORDER BY fetched_at {order} LIMIT 1""",
                    [fixture_id, mkt, ou] + line_param + time_param,
                ).fetchone()
                if row:
                    break

        elif mkt_key == "spreads":
            team_slug = outcome[:15] if outcome else ""
            for mkt in ("alternate_spreads", "spreads"):
                row = con.execute(
                    f"""SELECT odds FROM alt_lines
                        WHERE event_id = ? AND market_key = ? AND team LIKE ?
                        {line_sql} {time_sql}
                        ORDER BY fetched_at {order} LIMIT 1""",
                    [fixture_id, mkt, f"%{team_slug}%"] + line_param + time_param,
                ).fetchone()
                if row:
                    break

        con.close()
        if row and row[0]:
            return _decimal_to_american(float(row[0]))
    except Exception:
        pass
    return None


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
    # For alt markets, mkt_key maps to the canonical market (spreads/totals)
    # which matches line_snapshots market_key values.
    try:
        con = sqlite3.connect(hist_db, timeout=15)
        cur = con.cursor()
        norm_dt = _normalize_iso_dt(before_dt) if before_dt else None
        if norm_dt:
            order = "DESC"
            time_clause = "AND captured_at <= ?"
        else:
            order = "ASC"
            time_clause = ""

        if mkt_key:
            cur.execute(f"""
                SELECT price FROM line_snapshots
                WHERE event_id = ? AND market_key = ? AND outcome_name LIKE ?
                {time_clause}
                ORDER BY captured_at {order} LIMIT 1
            """, ([fixture_id, mkt_key, f"%{outcome}%"] + ([norm_dt] if norm_dt else [])))
        else:
            cur.execute(f"""
                SELECT price FROM line_snapshots
                WHERE event_id = ? AND outcome_name LIKE ?
                {time_clause}
                ORDER BY captured_at {order} LIMIT 1
            """, ([fixture_id, f"%{outcome}%"] + ([norm_dt] if norm_dt else [])))

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
        con = sqlite3.connect(hist_db, timeout=15)
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
        con = sqlite3.connect(hist_db, timeout=15)
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
        con_h = sqlite3.connect(hist_db, timeout=15)
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
            con_h2 = sqlite3.connect(hist_db, timeout=15)
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
        con_h3 = sqlite3.connect(hist_db, timeout=15)
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
        con_h4 = sqlite3.connect(hist_db, timeout=15)
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


# ─── Directional helpers (line movement + injury vs pick side) ────────────────

# Map alt markets to their parent market for line movement lookups.
# LineMovementLog only tracks h2h/spreads/totals; alt markets share direction.
_MARKET_TO_PARENT = {
    "alternate_spreads": "spreads",
    "alternate_totals":  "totals",
}


def _model_n_features(model_used: str | None) -> int | None:
    """Return the feature count for a model_used tag from its meta.json, or None."""
    import json as _json, os as _os
    _SUBMODEL_DIR = _os.path.join(_os.path.dirname(__file__), "..", "data", "submodels")
    _MAP = {
        "mlb_ats_v1":     "mlb_ats_meta.json",
        "mlb_totals_v1":  "mlb_totals_meta.json",
        "nhl_ats_v1":     "nhl_ats_meta.json",
    }
    fname = _MAP.get(model_used or "")
    if not fname:
        # Global model
        if model_used in ("combined_v1", "global"):
            _cols_path = _os.path.join(_os.path.dirname(__file__), "..", "data", "active_feature_cols.pkl")
            if _os.path.exists(_cols_path):
                import pickle
                with open(_cols_path, "rb") as f:
                    return len(pickle.load(f))
        return None
    path = _os.path.join(_SUBMODEL_DIR, fname)
    if not _os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return _json.load(f).get("n_features")
    except Exception:
        return None


def _pick_to_side(leg: dict) -> str | None:
    """
    Map a leg's ``pick`` field to the LineMovementLog ``side`` value.

    Returns "home", "away", "over", "under", or None (unmappable — e.g. player props).
    """
    pick = (leg.get("pick") or "").strip()
    if not pick:
        return None
    pick_lower = pick.lower()
    if pick_lower == "over":
        return "over"
    if pick_lower == "under":
        return "under"
    home = (leg.get("home_team") or "").strip()
    away = (leg.get("away_team") or "").strip()
    if pick.lower() == home.lower():
        return "home"
    if pick.lower() == away.lower():
        return "away"
    return None


# Implied-probability movement thresholds for directional classification.
_LINE_DIR_ADVERSE_THRESHOLD  = -0.03   # ≥3% implied-prob drop = adverse
_LINE_DIR_FAVORABLE_THRESHOLD = 0.03   # ≥3% implied-prob gain = favorable


def _check_leg_line_direction(
    db,
    leg: dict,
) -> tuple[str, float]:
    """
    Check whether the line moved *for* or *against* our pick side since the
    first snapshot today.

    Returns ``(direction, implied_prob_delta)`` where direction is one of
    ``"favorable"``, ``"adverse"``, ``"neutral"``, ``"no_data"``.
    """
    side = _pick_to_side(leg)
    if side is None:
        return ("no_data", 0.0)

    mkt = leg.get("market") or leg.get("market_type") or ""
    # Alt markets → parent main market for movement lookup
    mkt = _MARKET_TO_PARENT.get(mkt, mkt)
    if mkt not in ("h2h", "spreads", "totals"):
        return ("no_data", 0.0)

    fid = leg.get("fixture_id") or ""
    if not fid:
        return ("no_data", 0.0)

    try:
        from database import LineMovementLog

        # Use ALL snapshots for this fixture+market+side, not just today's.
        # Line movement is meaningful across the full tracking window (opening
        # line → current line), which may span multiple days of odds fetches.
        snapshots = (
            db.query(LineMovementLog)
            .filter(
                LineMovementLog.fixture_id  == fid,
                LineMovementLog.market_type == mkt,
                LineMovementLog.side        == side,
            )
            .order_by(LineMovementLog.fetched_at.asc())
            .all()
        )
    except Exception:
        return ("no_data", 0.0)

    if len(snapshots) < 2:
        return ("no_data", 0.0)

    first_dec = snapshots[0].decimal_odds or 0.0
    last_dec  = snapshots[-1].decimal_odds or 0.0
    if not first_dec or not last_dec:
        return ("no_data", 0.0)

    # Implied-prob delta from our side's perspective.
    # Higher implied prob = market thinks our side is more likely to win = favorable.
    ip_first = 1.0 / first_dec
    ip_last  = 1.0 / last_dec
    delta    = ip_last - ip_first   # positive = favorable for our pick

    if delta <= _LINE_DIR_ADVERSE_THRESHOLD:
        return ("adverse", round(delta, 4))
    if delta >= _LINE_DIR_FAVORABLE_THRESHOLD:
        return ("favorable", round(delta, 4))
    return ("neutral", round(delta, 4))


# Injury-impact threshold for directional classification.
_INJURY_DIR_THRESHOLD = 0.05   # ≥5% injury-impact gap = directional


def _check_leg_injury_direction(
    leg: dict,
    db,
) -> tuple[str, float]:
    """
    Check whether injuries help or hurt our pick side.

    Calls ``attach_injury_signals`` (which reads the InjuryReport table —
    refreshed at 7 AM, 1 PM, 4 PM CT and 60 min pre-game) and maps the
    net-advantage to the pick's team.

    Returns ``(direction, advantage_for_pick_side)`` where direction is
    ``"favorable"``, ``"adverse"``, or ``"neutral"``.
    """
    pick = (leg.get("pick") or "").strip()
    home = (leg.get("home_team") or "").strip()
    away = (leg.get("away_team") or "").strip()
    if not pick or (not home and not away):
        return ("neutral", 0.0)

    # For totals (Over/Under) there is no "our team" — injury direction N/A.
    if pick.lower() in ("over", "under"):
        return ("neutral", 0.0)

    try:
        from injury_fetcher import attach_injury_signals as _ais
        enriched = _ais(dict(leg), db, datetime.now(_CT).strftime("%Y-%m-%d"))
    except Exception:
        return ("neutral", 0.0)

    net = enriched.get("injury_net_advantage") or 0.0
    # net = away_impact - home_impact
    #   positive → home team advantaged (away team more injured)
    #   negative → away team advantaged (home team more injured)
    is_home = pick.lower() == home.lower()
    advantage_for_us = net if is_home else -net

    if advantage_for_us <= -_INJURY_DIR_THRESHOLD:
        return ("adverse", round(advantage_for_us, 4))
    if advantage_for_us >= _INJURY_DIR_THRESHOLD:
        return ("favorable", round(advantage_for_us, 4))
    return ("neutral", round(advantage_for_us, 4))


# ── Form-based rung nudge helpers ─────────────────────────────────────────────
# Cold teams (<35% last-5 WR) cover at dramatically higher rates at deeper
# rungs — market overcorrects on recent form → generous lines + cushion stacks.
# Backtest: +0.7pp WR gain, 4 L→W flips, 0 negative flips at +1.5→+2.5.
_FORM_COLD_THRESHOLD  = 0.35   # last-5 WR below 35%
_FORM_HOT_THRESHOLD   = 0.65   # last-5 WR above 65%
_FORM_COLD_PENALTY    = 0.92   # 8% penalty on shallow-rung picks for cold teams
_FORM_COLD_BOOST      = 1.06   # 6% boost on deeper-rung picks for cold teams

# Cache: {team_name: (form_wr, form_bucket)} — lives for one generation run
_form_cache: dict[str, tuple[float, str]] = {}


def _get_team_form_bucket(team: str, game_date: str | None = None) -> tuple[float, str]:
    """
    Compute team's last-5 win rate from historical.db games table.

    Returns (win_rate, bucket) where bucket is 'cold', 'neutral', or 'hot'.
    Uses game_date < filter for leak-safety (no lookahead).
    """
    if not team:
        return (0.5, "neutral")

    cache_key = f"{team}|{game_date or 'today'}"
    if cache_key in _form_cache:
        return _form_cache[cache_key]

    try:
        con = sqlite3.connect(_HIST_DB_PATH, timeout=10)
        cur = con.cursor()
        gd = game_date or datetime.now(_CT).strftime("%Y-%m-%d")
        cur.execute("""
            SELECT home_team, away_team, home_score, away_score
            FROM games
            WHERE (home_team = ? OR away_team = ?)
              AND game_date < ?
              AND home_score IS NOT NULL
            ORDER BY game_date DESC
            LIMIT 5
        """, (team, team, gd))
        rows = cur.fetchall()
        con.close()

        if len(rows) < 3:
            _form_cache[cache_key] = (0.5, "neutral")
            return (0.5, "neutral")

        wins = sum(
            1 for ht, at, hs, as_ in rows
            if (ht == team and hs > as_) or (at == team and as_ > hs)
        )
        wr = wins / len(rows)

        if wr < _FORM_COLD_THRESHOLD:
            bucket = "cold"
        elif wr > _FORM_HOT_THRESHOLD:
            bucket = "hot"
        else:
            bucket = "neutral"

        _form_cache[cache_key] = (round(wr, 3), bucket)
        return (round(wr, 3), bucket)
    except Exception:
        _form_cache[cache_key] = (0.5, "neutral")
        return (0.5, "neutral")


# ── FD consensus signal: cross-bookmaker divergence ─────────────────────────
# Threshold: >5pp divergence in implied probability space is "strong divergence".
# Validated on 5,748 spread legs: both tails (soft & tight) underperform
# consensus by 9-15pp (z=-2.6 to -4.3, p<0.01).
_FD_CONSENSUS_THRESHOLD = 0.05  # 5pp in implied probability space

# Market key mapping for bookmakers JSON (h2h / spreads / totals)
_BK_MKT_MAP = {"h2h": "h2h", "spreads": "spreads", "totals": "totals",
                "alternate_spreads": "spreads", "alternate_totals": "totals"}


def _compute_fd_field_divergence(
    leg: dict, bookmakers_json: list[dict] | None, desc: str = "",
) -> tuple[float | None, str]:
    """
    Compute FD implied probability vs field median for a leg.

    Returns (divergence, flag):
      divergence = fd_implied - field_median  (negative = FD softer)
      flag = 'consensus' | 'soft' | 'tight' | 'insufficient_books'
    """
    if not bookmakers_json:
        return None, "insufficient_books"

    mkt_raw = leg.get("market_type") or leg.get("market") or ""
    mkt_key = _BK_MKT_MAP.get(mkt_raw.lower(), mkt_raw.lower())
    if mkt_key not in ("h2h", "spreads", "totals"):
        return None, "insufficient_books"

    # Determine pick name to match against outcomes
    pick = (leg.get("pick") or leg.get("team") or "").strip()
    if not pick and desc:
        # Parse from description: "Team Name -1.5 (Spread)" or "Over 8.5 (Total)"
        desc_lower = desc.lower()
        if mkt_key == "totals":
            if "over" in desc_lower:
                pick = "Over"
            elif "under" in desc_lower:
                pick = "Under"
        else:
            ht = (leg.get("home_team") or "").strip()
            at = (leg.get("away_team") or "").strip()
            for t in (ht, at):
                if t and t.lower() in desc_lower:
                    pick = t
                    break

    if not pick:
        return None, "insufficient_books"
    pick_lower = pick.lower()

    fd_imp = None
    field_imps: list[float] = []

    for bk in bookmakers_json:
        bk_key = bk.get("key", "")
        for mkt in bk.get("markets", []):
            if mkt.get("key") != mkt_key:
                continue
            for out in mkt.get("outcomes", []):
                oname = (out.get("name") or "").lower()
                price = out.get("price")
                if not price or price <= 1:
                    continue
                # Match pick to outcome
                if pick_lower in oname or oname in pick_lower:
                    imp = 1.0 / price
                    if bk_key == "fanduel":
                        fd_imp = imp
                    else:
                        field_imps.append(imp)

    if fd_imp is None or len(field_imps) < 3:
        return None, "insufficient_books"

    import statistics as _stats
    field_median = _stats.median(field_imps)
    divergence = round(fd_imp - field_median, 6)

    if divergence < -_FD_CONSENSUS_THRESHOLD:
        flag = "soft"
    elif divergence > _FD_CONSENSUS_THRESHOLD:
        flag = "tight"
    else:
        flag = "consensus"

    return divergence, flag


# ── 60% parlay WR floor ─────────────────────────────────────────────────────
# Target: 60% parlay-level WR.  Per-leg floor = 0.60^(1/n).
# Backtest (n=88 qualifying parlays): 88.0% WR, +25.8% ROI.
# Applied as a LIVE gate only; prospective picks tagged for forward monitoring.
_PARLAY_WR_TARGET = 0.60
_LEG_FLOOR: dict[int, float] = {}
for _n in range(1, 7):
    _LEG_FLOOR[_n] = round(_PARLAY_WR_TARGET ** (1.0 / _n) * 100.0, 2)
# Pre-computed: {1: 60.0, 2: 77.46, 3: 84.34, 4: 88.02, 5: 90.31, 6: 91.86}


def _compute_floor_pass(legs: list[dict], n_legs: int) -> tuple[bool, float]:
    """
    Check if every leg clears the per-depth WR floor.

    Returns (passes, margin) where margin = min(leg_wp - floor).
    Negative margin means at least one leg is below the floor.
    """
    floor = _LEG_FLOOR.get(n_legs, _LEG_FLOOR.get(6, 92.0))
    min_margin = float("inf")
    for leg in legs:
        wp = leg.get("win_prob") or leg.get("predicted_win_prob") or 0.0
        margin = wp - floor
        if margin < min_margin:
            min_margin = margin
    passes = min_margin >= 0.0
    return passes, round(min_margin, 2) if min_margin != float("inf") else 0.0


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
    "Europa League", "Soccer", "World Cup",
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
    include_started: bool = False,
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

    all_raw = get_available_legs(db, markets=["h2h", "spreads", "totals"], include_started=include_started)
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
                       "Serie A", "Ligue 1", "Soccer", "World Cup"],
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

        # No mixed team + player_prop parlays
        cats = {classify_bet_category(l.get("market_type") or l.get("market")) for l in legs}
        if len(cats) > 1:
            continue

        # Same-fixture and same-team checks only apply to team-market legs.
        # Player prop legs (different stats / players) may freely combine from
        # the same game — e.g. Josh Hart +8 pts AND +6 reb is one parlay.
        _team_legs_fp = [l for l in legs if classify_bet_category(
            l.get("market_type") or l.get("market")) == "team"]
        if _team_legs_fp:
            fids = [l.get("fixture_id") for l in _team_legs_fp]
            if len(fids) != len(set(fids)):
                continue
            teams = [l.get("pick", "").strip().lower() for l in _team_legs_fp]
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
        avg_lqs      = round(min(100.0, sum(l.get("lqs", 0) for l in legs) / 2), 1)

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
    include_started: bool        = False,
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

    # ── WAL checkpoint: ensure scout writes are visible ──────────────────
    # Scout runs on a separate SQLAlchemy session/connection.  In WAL mode,
    # a new sqlite3 connection (used by _scout_lookup_raw during leg
    # creation) may not see those writes until the WAL is checkpointed.
    # PASSIVE checkpoint flushes committed WAL pages without blocking readers.
    try:
        from sqlalchemy import text as _wal_text
        db.execute(_wal_text("PRAGMA wal_checkpoint(PASSIVE)"))
    except Exception:
        pass  # non-fatal — backfill will catch any misses

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
    _today = _gen_today_ct.strftime("%Y-%m-%d")

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
                    include_started = include_started,
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
                        include_started= include_started,
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
        avg = round(min(100.0, sum(leg_scores) / len(leg_scores)), 1)
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

    # ── Compute composite score for picks that lack one ────────────────────
    # Recommender picks arrive with _composite already set (30% WP + 15% EV
    # + 10% LQS + 20% scout + 10% sport + 5% market).
    # Forced picks and tier_b injections lack _composite — compute a fallback
    # using the fields available: WP, EV (normalised), and LQS.
    _max_ev = max(
        (abs(p.get("expected_profit") or p.get("predicted_ev") or 0) for p in scored_picks),
        default=1.0,
    ) or 1.0
    for p in scored_picks:
        if p.get("_composite") is not None:
            p["_composite_score"] = round(p["_composite"], 6)
        else:
            _wp  = (p.get("combined_win_prob") or p.get("win_prob") or 50.0) / 100.0
            _ev  = max(p.get("expected_profit") or 0, 0) / _max_ev
            _lqs = (p.get("_avg_lqs") or 50.0) / 100.0
            p["_composite_score"] = round(_wp * 0.50 + _ev * 0.30 + _lqs * 0.20, 6)

    # Sort: highest composite_score first, LQS as tiebreaker
    scored_picks.sort(key=lambda x: (-x.get("_composite_score", 0), -x.get("_avg_lqs", 50)))

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

    # ── Directional tagging (tracking only — does NOT influence scoring) ─────
    # Tag each leg with line_direction and injury_direction for post-hoc
    # analysis.  These signals are NOT proven to predict outcomes yet (all
    # historical data was no_data/neutral due to a date-format bug, now fixed).
    # Once we accumulate 2+ weeks of real data, re-evaluate whether to let
    # them influence composite scoring.
    _dir_adverse_count = 0
    _dir_favorable_count = 0
    for _p in scored_picks:
        for _leg in _p.get("legs", []):
            _ld, _lm = _check_leg_line_direction(db, _leg)
            _leg["_line_direction"] = _ld
            if _ld == "adverse":
                _dir_adverse_count += 1
            elif _ld == "favorable":
                _dir_favorable_count += 1

            _id, _im = _check_leg_injury_direction(_leg, db)
            _leg["_injury_direction"] = _id
            if _id == "adverse":
                _dir_adverse_count += 1
            elif _id == "favorable":
                _dir_favorable_count += 1

    if _dir_adverse_count or _dir_favorable_count:
        print(f"[mock-gen] directional tags: {_dir_adverse_count} adverse, "
              f"{_dir_favorable_count} favorable (tracking only, no scoring impact)")
    elif scored_picks:
        print("[mock-gen] directional tags: no adverse/favorable signals (no_data or neutral)")

    # ── Pattern signal adjustments ───────────────────────────────────────
    # Check BetPatternSignal table for sport+market patterns and adjust scoring.
    _pattern_adj_count = 0
    try:
        from database import BetPatternSignal
        _pattern_cache: dict[str, BetPatternSignal | None] = {}
        for _p in scored_picks:
            _pattern_adj = 1.0
            for _leg in _p.get("legs", []):
                _sport = _leg.get("sport") or _p.get("sport") or ""
                _mkt   = _leg.get("market") or _leg.get("market_type") or ""
                _team  = (_leg.get("pick") or _leg.get("team") or "").strip()

                # Sport + market pattern
                _sm_key = f"{_sport}|{_mkt}"
                if _sm_key not in _pattern_cache:
                    _pattern_cache[_sm_key] = db.query(BetPatternSignal).filter(
                        BetPatternSignal.pattern_key.like(f"{_sport}|{_mkt}|%")
                    ).order_by(BetPatternSignal.n_legs.desc()).first()
                _sm_pat = _pattern_cache.get(_sm_key)
                if _sm_pat and _sm_pat.n_legs and _sm_pat.n_legs >= 30:
                    if (_sm_pat.actual_wr or 0) < (_sm_pat.predicted_wr or 0) - 10:
                        _pattern_adj *= 0.85  # underperforming pattern
                    elif (_sm_pat.actual_wr or 0) > (_sm_pat.predicted_wr or 0) + 10:
                        _pattern_adj *= 1.10  # outperforming pattern

            if _pattern_adj != 1.0:
                _p["_composite_score"] = _p.get("_composite_score", 0) * _pattern_adj
                _pattern_adj_count += 1
    except Exception as _pe:
        print(f"[mock-gen] pattern signal error (non-fatal): {_pe}")
    if _pattern_adj_count:
        print(f"[mock-gen] pattern signal adjustments: {_pattern_adj_count} picks adjusted")

    # ── Player projection alignment ──────────────────────────────────────
    # For MLB spread/total legs, check if projections agree with pick direction.
    _proj_adj_count = 0
    try:
        from sqlalchemy import text as _proj_text
        for _p in scored_picks:
            _proj_adj = 1.0
            for _leg in _p.get("legs", []):
                _sport = (_leg.get("sport") or _p.get("sport") or "").upper()
                if _sport != "MLB":
                    continue
                _mkt = _leg.get("market") or _leg.get("market_type") or ""
                if "spread" not in _mkt and "total" not in _mkt:
                    continue
                _gd = _p.get("game_date") or _today
                _home = _leg.get("home_team") or _leg.get("resolved_home_team") or ""
                _away = _leg.get("away_team") or _leg.get("resolved_away_team") or ""
                if not _home or not _away:
                    continue
                # Sum projected runs for each team
                _rows = db.execute(_proj_text("""
                    SELECT team, SUM(projection) as total_proj
                    FROM player_projections
                    WHERE game_date = :gd AND sport = 'MLB'
                      AND stat_type = 'runs'
                      AND (team = :ht OR team = :at)
                    GROUP BY team
                """), {"gd": _gd, "ht": _home, "at": _away}).fetchall()
                if len(_rows) < 2:
                    continue
                _team_proj = {r[0]: float(r[1]) for r in _rows}
                _h_proj = _team_proj.get(_home, 0)
                _a_proj = _team_proj.get(_away, 0)
                _proj_margin = _h_proj - _a_proj  # positive = home favored

                _pick = (_leg.get("pick") or _leg.get("team") or "").strip()
                if "spread" in _mkt:
                    _pick_is_home = _pick.lower() in _home.lower() or _home.lower() in _pick.lower()
                    if _pick_is_home and _proj_margin > 0.5:
                        _proj_adj *= 1.08
                    elif _pick_is_home and _proj_margin < -0.5:
                        _proj_adj *= 0.90
                    elif not _pick_is_home and _proj_margin < -0.5:
                        _proj_adj *= 1.08
                    elif not _pick_is_home and _proj_margin > 0.5:
                        _proj_adj *= 0.90
                elif "total" in _mkt:
                    _total_proj = _h_proj + _a_proj
                    _pick_lower = _pick.lower()
                    _line = _leg.get("point") or _leg.get("line")
                    if _line and _total_proj > 0:
                        _line_f = float(_line)
                        if "over" in _pick_lower and _total_proj > _line_f + 0.5:
                            _proj_adj *= 1.08
                        elif "over" in _pick_lower and _total_proj < _line_f - 0.5:
                            _proj_adj *= 0.90
                        elif "under" in _pick_lower and _total_proj < _line_f - 0.5:
                            _proj_adj *= 1.08
                        elif "under" in _pick_lower and _total_proj > _line_f + 0.5:
                            _proj_adj *= 0.90

            if _proj_adj != 1.0:
                _p["_composite_score"] = _p.get("_composite_score", 0) * _proj_adj
                _proj_adj_count += 1
    except Exception as _pe2:
        print(f"[mock-gen] projection signal error (non-fatal): {_pe2}")
    if _proj_adj_count:
        print(f"[mock-gen] projection alignment adjustments: {_proj_adj_count} picks adjusted")

    # ── FD consensus divergence penalty (spread legs only) ─────────────
    # When FD strongly disagrees with the field (>5pp in implied prob),
    # both tails underperform consensus by 9-15pp.  Apply symmetric 0.88
    # multiplier to composite for spread/alt-spread legs.  Track-only for
    # h2h/totals (insufficient tail sample to penalize).
    _fd_penalty_count = 0
    _fd_track_count = 0
    try:
        from sqlalchemy import text as _fd_text
        _bk_cache: dict[str, list[dict] | None] = {}
        for _p in scored_picks:
            _fd_adj = 1.0
            for _leg in _p.get("legs", []):
                _fid = _leg.get("fixture_id") or _leg.get("game_id")
                if not _fid:
                    continue
                # Cache bookmakers JSON per fixture
                if _fid not in _bk_cache:
                    _bk_row = db.execute(_fd_text(
                        "SELECT bookmakers FROM fixtures WHERE id=:fid"
                    ), {"fid": _fid}).fetchone()
                    if _bk_row and _bk_row[0]:
                        try:
                            _bk_cache[_fid] = json.loads(_bk_row[0])
                        except Exception:
                            _bk_cache[_fid] = None
                    else:
                        _bk_cache[_fid] = None

                _bk_json = _bk_cache.get(_fid)
                _desc = _leg.get("description") or ""
                _div, _flag = _compute_fd_field_divergence(_leg, _bk_json, _desc)
                _leg["_fd_field_divergence"] = _div
                _leg["_fd_consensus_flag"] = _flag

                if _flag in ("soft", "tight"):
                    _mkt = (_leg.get("market_type") or _leg.get("market") or "").lower()
                    if "spread" in _mkt:
                        _fd_adj *= 0.88
                        _fd_penalty_count += 1
                    else:
                        _fd_track_count += 1

            if _fd_adj != 1.0:
                _p["_composite_score"] = _p.get("_composite_score", 0) * _fd_adj
    except Exception as _fd_err:
        print(f"[mock-gen] FD consensus signal error (non-fatal): {_fd_err}")
    if _fd_penalty_count or _fd_track_count:
        print(f"[mock-gen] FD consensus: {_fd_penalty_count} spread legs penalized, "
              f"{_fd_track_count} non-spread tracked only")

    # ── Form-based rung nudge (composite, not hard override) ─────────────
    # Cold teams (<35% last-5 WR) cover at higher rates at deeper rungs
    # because the market overcorrects on recent form.  Backtest on 541
    # game-sides: +0.7pp WR gain, 4 L→W flips, 0 negative flips.
    # Effect: penalize shallow-rung spread picks for cold teams (nudging
    # the system toward deeper-cushion alternatives when available) and
    # boost deeper-rung picks.  Tags each leg with form_bucket.
    _form_cache.clear()  # fresh cache each generation run
    _form_nudge_count = 0
    for _p in scored_picks:
        _form_adj = 1.0
        for _leg in _p.get("legs", []):
            _sport = (_leg.get("sport") or _p.get("sport") or "").upper()
            _mkt = (_leg.get("market_type") or _leg.get("market") or "").lower()
            if _sport != "MLB" or "spread" not in _mkt:
                continue
            _pick = (_leg.get("pick") or "").strip()
            if not _pick or _pick.lower() in ("over", "under"):
                continue
            _point = _leg.get("point")
            try:
                _rung = abs(float(_point)) if _point is not None else None
            except (TypeError, ValueError):
                _rung = None
            if _rung is None:
                continue

            _wr, _bucket = _get_team_form_bucket(_pick)
            _leg["_form_bucket"] = _bucket
            _leg["_form_wr"] = _wr

            if _bucket == "cold":
                if _rung <= 1.5:
                    # Shallow rung + cold team → penalize (prefer deeper cushion)
                    _form_adj *= _FORM_COLD_PENALTY
                    _leg["_line_delta_rec_rung"] = 2.5
                elif _rung >= 2.5:
                    # Deeper rung + cold team → boost (this IS the right cushion)
                    _form_adj *= _FORM_COLD_BOOST
                    _leg["_line_delta_rec_rung"] = _rung

        if _form_adj != 1.0:
            _p["_composite_score"] = _p.get("_composite_score", 0) * _form_adj
            _form_nudge_count += 1
    if _form_nudge_count:
        print(f"[mock-gen] form-rung nudge: {_form_nudge_count} picks adjusted "
              f"(cold-team shallow penalized / deep boosted)")

    # ── Pitcher ERA matchup nudge (line delta build, 2026-06-14) ─────────
    # Full-ladder counterfactual on 1044 game-sides: elite starters cover
    # at 76.5% at +1.5 vs 51.1% for poor starters (25.4pp gap).  On actual
    # settled legs (4478): unfavorable ERA matchup at +1.5 = 59.0% vs
    # favorable = 62.8% (4.6pp gap, reduced because base model already
    # prices pitcher quality).  Nudge unfavorable-matchup shallow picks down.
    # NOTE: pitcher ERA is not on the leg dict at this stage (enrichment runs
    # post-commit), so we fetch it live via get_pitcher_features, cached per
    # fixture.
    _PITCHER_UNFAV_PENALTY = 0.93   # 7% penalty: unfavorable ERA matchup + shallow rung
    # Favorable boost PULLED — within WP 60-70% at +1.5, favorable ERA
    # underperforms even (61.7% vs 64.8%): model already prices good pitching.
    # _PITCHER_FAV_BOOST = 1.04  ← would double-count base win_prob
    _pitcher_nudge_count = 0
    _pitcher_feat_cache: dict[str, dict] = {}
    try:
        from mlb_features import get_pitcher_features as _gpf
    except ImportError:
        _gpf = None
    if _gpf:
        for _p in scored_picks:
            _pitch_adj = 1.0
            for _leg in _p.get("legs", []):
                _sport = (_leg.get("sport") or _p.get("sport") or "").upper()
                _mkt = (_leg.get("market_type") or _leg.get("market") or "").lower()
                if _sport != "MLB" or "spread" not in _mkt:
                    continue
                _pick = (_leg.get("pick") or "").strip()
                if not _pick or _pick.lower() in ("over", "under"):
                    continue
                _fid = _leg.get("fixture_id") or ""
                _home = (_leg.get("home_team") or "").strip()
                _away = (_leg.get("away_team") or "").strip()
                if not _fid or not _home or not _away:
                    continue
                _is_home = _pick.lower() == _home.lower()

                # Fetch pitcher ERA (cached per fixture)
                if _fid not in _pitcher_feat_cache:
                    try:
                        _gd = _p.get("game_date") or datetime.now(_CT).strftime("%Y-%m-%d")
                        _pitcher_feat_cache[_fid] = _gpf(_home, _away, str(_gd)[:10], "MLB")
                    except Exception:
                        _pitcher_feat_cache[_fid] = {}
                _pf = _pitcher_feat_cache[_fid]
                _h_era = _pf.get("home_starter_era")
                _a_era = _pf.get("away_starter_era")
                import math as _math
                if _h_era is None or _a_era is None:
                    continue
                _pick_era = _h_era if _is_home else _a_era
                _opp_era  = _a_era if _is_home else _h_era
                try:
                    _era_diff = float(_pick_era) - float(_opp_era)
                except (TypeError, ValueError):
                    continue
                if _math.isnan(_era_diff):
                    continue

                _point = _leg.get("point")
                try:
                    _rung = abs(float(_point)) if _point is not None else None
                except (TypeError, ValueError):
                    _rung = None
                if _rung is None:
                    continue

                # Unfavorable matchup (our starter worse by >1.0 ERA) + shallow
                if _era_diff > 1.0 and _rung <= 1.5:
                    _pitch_adj *= _PITCHER_UNFAV_PENALTY
                # Favorable boost pulled — model already prices it (see comment above)

            if _pitch_adj != 1.0:
                _p["_composite_score"] = _p.get("_composite_score", 0) * _pitch_adj
                _pitcher_nudge_count += 1
    if _pitcher_nudge_count:
        print(f"[mock-gen] pitcher-era nudge: {_pitcher_nudge_count} picks adjusted")

    # ── Signal model investigation: CLOSED 2026-06-14 ──────────────────
    # Investigated stacking post-hoc signals (pi_diff_pick, scout_grade,
    # fd_soft) on top of the calibrated base model.  All four closed:
    #   - pi_diff_pick: structural redundancy — Pi is already a base model
    #     feature (rank 11-21/80).  Inverts at |Pi|>1.5 because the model
    #     correctly out-thinks raw Pi with pitcher/injury/form info.
    #     Bootstrap 95% CI spans zero.  No post-hoc Pi layer.
    #   - scout_grade: collinear with calibrated base (delta AUC = -0.006).
    #   - fd_soft: already handled here (0.88x penalty on spread legs).
    #     Signal model independently confirmed coef ≈ -0.88.
    #   - fd_tight: favorable (+2.8pp WR), no penalty needed.
    # Future Pi lever: feature engineering (rolling momentum, Pi-vs-line
    # disagreement) as NEW base model inputs, validated by retrain.
    # Full details: data/submodels/signal_model_meta.json
    # ─────────────────────────────────────────────────────────────────────

    # Re-sort after all adjustments
    scored_picks.sort(key=lambda x: (-x.get("_composite_score", 0), -x.get("_avg_lqs", 50)))

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

    # ── Mandatory final boost pass — two-strategy framework ──────────────────
    # Strategy A: amplify already-positive EV (base_ev > +0.50/10)
    # Strategy B: rescue high-WR chalk into +EV (wp ≥ 65%, grades A/B, boosted > +1.0)
    # Skip: structurally weak (base_ev < -0.50 OR wp < 55%) or boosted_ev < +0.30
    #
    # Availability draw: FanDuel offers boosts at ~39.7% of opportunities.
    # The draw determines WHICH tiers are available; strategy selects the optimal one.
    # Direct assignment (not setdefault) — authoritative for this run.
    _EXPL_SRCS = set(EXPLORATION_SOURCES)
    _strat_counts: dict[str, int] = {"amplify_positive_ev": 0, "rescue_high_wr_chalk": 0, "no_boost": 0}

    for _fp in picks:
        # Exploration / forced-generation picks carry $0 stake; skip boost.
        if source in _EXPL_SRCS or _fp.get("is_forced_generation"):
            _fp["_promo_type"]        = None
            _fp["_promo_boost_pct"]   = 0.0
            _fp["_boost_strategy"]    = "no_boost"
            _fp["_base_ev_per_10"]    = _compute_ev_per_10(
                float(_fp.get("combined_odds") or 1.0),
                float(_fp.get("combined_win_prob") or 50.0),
            )
            _fp["_boosted_ev_per_10"] = _fp["_base_ev_per_10"]
            _fp["_boost_lift_ev"]     = 0.0
            continue

        _fp_odds = float(_fp.get("combined_odds") or _fp.get("odds") or 1.0)
        _fp_legs = len(_fp.get("legs") or [])
        _fp_sgp  = _is_sgp(_fp)
        _fp_ss   = _is_single_sport(_fp)
        _fp_wp   = float(_fp.get("combined_win_prob") or _fp.get("win_prob") or 50.0)

        # Step 1: Draw which tiers FanDuel makes available this opportunity
        _fpt, _drawn_pct = _get_boost_for_bet()
        _available_tiers: list[float] = []
        if _fpt == "PROFIT_BOOST" and _drawn_pct > 0:
            # All tiers up to and including the drawn tier are available
            for _t in [0.25, 0.30, 0.50]:
                if _t <= _drawn_pct:
                    _t_ok, _ = _check_boost_eligible(_t, _fp_legs, _fp_odds, _fp_sgp, _fp_ss)
                    # +50% Route B: also must pass WP gate
                    if _t_ok and _t == 0.50:
                        _wp_ok, _ = _route_b_wp_gate(_fp)
                        _t_ok = _wp_ok
                    if _t_ok:
                        _available_tiers.append(_t)

        # Step 2: Evaluate strategy across all available tiers
        _opt_pct, _strat, _base_ev, _boosted_ev, _lift = _evaluate_boost_strategy(
            _fp, _available_tiers
        )

        # Step 3: Assign results
        if _opt_pct > 0 and _strat != "no_boost":
            _fp["_promo_type"]        = "PROFIT_BOOST"
            _fp["_promo_boost_pct"]   = _opt_pct
            _fp["_boost_eligibility"] = f"{_strat} +{int(_opt_pct*100)}%"
            _fp["_boost_ev"]          = _boost_ev(stake, _fp_odds, _fp_wp, _opt_pct)
        else:
            _fp["_promo_type"]        = None if _strat == "no_boost" else _fpt
            _fp["_promo_boost_pct"]   = 0.0
        _fp["_boost_strategy"]    = _strat
        _fp["_base_ev_per_10"]    = _base_ev
        _fp["_boosted_ev_per_10"] = _boosted_ev
        _fp["_boost_lift_ev"]     = _lift
        _strat_counts[_strat] = _strat_counts.get(_strat, 0) + 1

    _n_boosted_final = sum(1 for _fp in picks if _fp.get("_promo_type") == "PROFIT_BOOST")
    print(
        f"[mock-gen] final boost pass: {_n_boosted_final}/{len(picks)} picks boosted | "
        f"strat_A={_strat_counts['amplify_positive_ev']} "
        f"strat_B={_strat_counts['rescue_high_wr_chalk']} "
        f"no_boost={_strat_counts['no_boost']}"
    )

    generated = 0
    skipped   = 0

    # ── Game-level concentration cap ────────────────────────────────────
    # Prevents the same fixture from appearing in too many bets across
    # ALL runs for the same game_date.  Picks are already sorted by
    # composite score, so the best picks get priority; lower-ranked picks
    # reusing the same game are dropped.
    MAX_BETS_PER_GAME = 3
    from collections import defaultdict as _defaultdict
    from sqlalchemy import text as _gx_text
    _today_ct = datetime.now(_CT).strftime("%Y-%m-%d")
    _game_exposure: dict[str, int] = _defaultdict(int)

    # Seed with existing PRODUCTION bets already stored today (from prior runs).
    # Use an allowlist of production sources — shadow/exploration/simulation must
    # not count against the cap (they'd saturate every fixture).
    _prior_rows = db.execute(_gx_text(
        "SELECT mbl.fixture_id, COUNT(DISTINCT mb.id) "
        "FROM mock_bets mb "
        "JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id "
        "WHERE mb.game_date = :gd "
        "  AND mb.source IN ('prospective','prospective_pm','top_picks_page','forced_generation') "
        "GROUP BY mbl.fixture_id"
    ), {"gd": _today_ct}).fetchall()
    for _fid_row, _cnt_row in _prior_rows:
        if _fid_row:
            _game_exposure[_fid_row] = _cnt_row

    _cap_skipped = 0

    for pick in picks:
        legs     = pick.get("legs", [])
        bet_info = " | ".join(
            _format_leg_desc(l) for l in legs if l.get("description") or l.get("pick")
        )
        if not bet_info or bet_info in existing_infos:
            skipped += 1
            continue

        # Game concentration cap: skip if ANY fixture in this pick is at limit
        _fixture_ids = {l.get("fixture_id") or l.get("game_id") for l in legs} - {None}
        if any(_game_exposure[fid] >= MAX_BETS_PER_GAME for fid in _fixture_ids):
            _cap_skipped += 1
            continue
        # Accept: increment exposure for all fixtures in this pick
        for fid in _fixture_ids:
            _game_exposure[fid] += 1

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
            composite_score    = pick.get("_composite_score"),
            promo_type         = pick.get("_promo_type"),
            promo_boost_pct    = pick.get("_promo_boost_pct"),
            boost_strategy     = pick.get("_boost_strategy") or pick.get("_boost_eligibility"),
            promo_ev_lift      = (pick.get("_boost_ev") or {}).get("ev_lift"),
            base_ev_per_10     = pick.get("_base_ev_per_10"),
            boost_lift_ev      = pick.get("_boost_lift_ev"),
            boosted_ev_per_10  = pick.get("_boosted_ev_per_10"),
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
            cohort             = "clean",
        )
        db.add(mock)

        # ── Shadow B composition tracking ─────────────────────────
        _sb_adj = pick.get("_comp_adj")
        _sb_fav = pick.get("_comp_fav_count")
        _sb_br  = pick.get("_blind_rank")
        if _sb_adj is not None or _sb_fav is not None or _sb_br is not None:
            try:
                from sqlalchemy import text as _text
                db.execute(_text(
                    "UPDATE mock_bets SET shadow_b_comp_adj=:adj, "
                    "shadow_b_fav_count=:fc, shadow_b_blind_rank=:br WHERE id=:mid"
                ), {"adj": _sb_adj, "fc": _sb_fav, "br": _sb_br, "mid": mock_id})
            except Exception:
                pass  # column may not exist yet

        for i, leg in enumerate(legs):
            _fid  = leg.get("fixture_id") or leg.get("game_id")
            _desc = leg.get("description", "")
            _mkt  = leg.get("market_type") or leg.get("market")
            # Opening odds: try line_snapshots first, then alt_lines earliest
            # snapshot, then fall back to the leg's own decimal odds (always
            # present from recommender/parlay_builder — zero additional queries).
            _open = _lookup_snap_odds(_fid, _desc, _mkt, _HIST_DB_PATH)
            if _open is None:
                _open = _lookup_alt_lines_odds(_fid, _desc, _mkt, _HIST_DB_PATH, earliest=True)
            if _open is None and leg.get("odds"):
                _open = _decimal_to_american(float(leg["odds"]))

            # Enrich leg with scout grade from scouted_props
            _scout_grade    = leg.get("scout_grade")
            _scout_hit_prob = leg.get("scout_hit_prob")
            _scout_prop_id  = leg.get("scouted_prop_id")
            if _scout_grade is None:
                try:
                    _enriched       = _plc.attach_scout_grade_to_leg(dict(leg), db)
                    _scout_grade    = _enriched.get("scout_grade")
                    _scout_hit_prob = _enriched.get("scout_hit_prob")
                    _scout_prop_id  = _enriched.get("scouted_prop_id")
                except Exception as _scout_exc:
                    print(f"[mock-gen] scout enrichment failed for {_desc}: {_scout_exc}")

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
                bet_category        = classify_bet_category(_mkt),
            )
            db.add(leg_obj)
            db.flush()  # get leg_obj.id so we can update scout columns

            # Write spread features via raw SQL (not in ORM)
            _sv, _sa, _isf = _parse_spread_for_leg(_desc, _mkt)
            if _sv is not None:
                try:
                    from sqlalchemy import text as _text
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET spread_value=:sv, spread_abs=:sa, "
                        "is_favorite_spread=:isf WHERE id=:lid"
                    ), {"sv": _sv, "sa": _sa, "isf": _isf, "lid": leg_obj.id})
                except Exception:
                    pass

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

            # ── Attach injury signals ──────────────────────────────────────────
            try:
                from injury_fetcher import attach_injury_signals as _ais
                from sqlalchemy import text as _text
                _leg_enriched = _ais(dict(leg), db,
                                     datetime.now(_CT).strftime("%Y-%m-%d"))
                _hi  = _leg_enriched.get("home_team_injury_impact")
                _ai  = _leg_enriched.get("away_team_injury_impact")
                _net = _leg_enriched.get("injury_net_advantage")
                if _net is not None:
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET "
                        "home_team_injury_impact=:hi, away_team_injury_impact=:ai, "
                        "injury_net_advantage=:net WHERE id=:lid"
                    ), {"hi": _hi, "ai": _ai, "net": _net, "lid": leg_obj.id})
            except Exception:
                pass

            # ── Attach line movement signal ────────────────────────────────────
            try:
                from line_movement import get_line_movement_signal as _gls
                from sqlalchemy import text as _text
                _side_mapped = _pick_to_side(leg) or ""
                if _side_mapped:
                    _lm = _gls(db, _fid or "", _mkt or "", _side_mapped, _side_mapped)
                    if _lm != 0.0:
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET line_movement_signal=:lm WHERE id=:lid"
                        ), {"lm": _lm, "lid": leg_obj.id})
            except Exception:
                pass

            # ── Persist directional filter results ────────────────────────────
            _ld_val = leg.get("_line_direction")
            _id_val = leg.get("_injury_direction")
            if _ld_val or _id_val:
                try:
                    from sqlalchemy import text as _text
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET line_direction=:ld, "
                        "injury_direction=:id WHERE id=:lid"
                    ), {"ld": _ld_val, "id": _id_val, "lid": leg_obj.id})
                except Exception:
                    pass

            # ── Persist FD consensus divergence ──────────────────────────────
            _fd_div = leg.get("_fd_field_divergence")
            _fd_flag = leg.get("_fd_consensus_flag")
            if _fd_div is not None or _fd_flag:
                try:
                    from sqlalchemy import text as _text
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET fd_field_divergence=:div, "
                        "fd_consensus_flag=:flag WHERE id=:lid"
                    ), {"div": _fd_div, "flag": _fd_flag, "lid": leg_obj.id})
                except Exception:
                    pass

            # ── Persist form-based rung nudge tags ────────────────────────
            _fb = leg.get("_form_bucket")
            _rec = leg.get("_line_delta_rec_rung")
            if _fb or _rec is not None:
                try:
                    from sqlalchemy import text as _text
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET form_bucket=:fb, "
                        "line_delta_rec_rung=:rec WHERE id=:lid"
                    ), {"fb": _fb, "rec": _rec, "lid": leg_obj.id})
                except Exception:
                    pass

            # ── Persist model_n_features for era filtering ──────────────
            _mnf = _model_n_features(leg.get("model_used"))
            if _mnf is not None:
                try:
                    from sqlalchemy import text as _text
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET model_n_features=:nf WHERE id=:lid"
                    ), {"nf": _mnf, "lid": leg_obj.id})
                except Exception:
                    pass

            # ── Persist Pi rating features ────────────────────────────────
            try:
                _ht = leg.get("resolved_home_team") or leg.get("home_team")
                _at = leg.get("resolved_away_team") or leg.get("away_team")
                # Fallback: look up home/away from fixtures table if not on leg dict
                if (not _ht or not _at) and _fid:
                    _fix_row = db.execute(_text(
                        "SELECT home_team, away_team FROM fixtures WHERE id=:fid"
                    ), {"fid": _fid}).fetchone()
                    if _fix_row:
                        _ht = _ht or _fix_row[0]
                        _at = _at or _fix_row[1]
                _gd = pick.get("game_date") or _today
                if _ht and _at and leg.get("sport", "").lower() in ("mlb", "baseball_mlb"):
                    from pi_ratings import get_team_pi_for_date as _get_pi
                    _h_pi = _get_pi(_ht, str(_gd)[:10], "MLB")
                    _a_pi = _get_pi(_at, str(_gd)[:10], "MLB")
                    if _h_pi and _a_pi:
                        _hp = round(_h_pi["home_pi"], 4)
                        _ap = round(_a_pi["away_pi"], 4)
                        _pd = round(_hp - _ap, 4)  # raw home-away (for ML model)
                        # Pick-relative: positive = our picked side is stronger
                        _pick_team = (leg.get("pick") or "").strip().lower()
                        _at_lower = (_at or "").strip().lower()
                        _is_away_pick = bool(_pick_team and _at_lower and (
                            _pick_team in _at_lower or _at_lower in _pick_team
                        ))
                        _pd_pick = round(-_pd if _is_away_pick else _pd, 4)
                        from sqlalchemy import text as _text
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET home_team_home_pi=:hp, "
                            "away_team_away_pi=:ap, pi_diff=:pd, pi_diff_pick=:pdp WHERE id=:lid"
                        ), {"hp": _hp, "ap": _ap, "pd": _pd, "pdp": _pd_pick, "lid": leg_obj.id})
            except Exception as _pi_exc:
                print(f"[mock-gen] Pi enrichment failed: {_pi_exc}")

            # ── Persist MLB enrichment features (pitcher/weather/statcast/bullpen/H2H) ──
            try:
                _sport_norm = (leg.get("sport") or "").lower()
                if _sport_norm in ("mlb", "baseball_mlb") and _ht and _at:
                    from mlb_features import (
                        get_pitcher_features, get_weather_features,
                        get_statcast_features, get_bullpen_features,
                        get_h2h_features,
                    )
                    from sqlalchemy import text as _text
                    _commence = leg.get("commence_time") or ""
                    _pf = get_pitcher_features(_ht, _at, str(_gd)[:10], _sport_norm)
                    _wf = get_weather_features(_ht, _commence, _sport_norm)
                    _sf = get_statcast_features(_ht, _at, _sport_norm)
                    _bf = get_bullpen_features(_ht, _at, _sport_norm)
                    _hf = get_h2h_features(_ht, _at, db)
                    _vals = {}
                    for _d in (_pf, _wf, _sf, _bf, _hf):
                        for _k, _v in _d.items():
                            if _v is not None and _v == _v:  # skip NaN
                                _vals[_k] = round(float(_v), 4)
                    if _vals:
                        _sets = ", ".join(f"{k}=:{k}" for k in _vals)
                        _vals["lid"] = leg_obj.id
                        db.execute(_text(
                            f"UPDATE mock_bet_legs SET {_sets} WHERE id=:lid"
                        ), _vals)
            except Exception as _mlb_exc:
                print(f"[mock-gen] MLB enrichment failed for leg {i}: {_mlb_exc}")

        # ── Tier classification (after all legs + signals are persisted) ────
        try:
            _tier, _tier_signals = classify_bet_tier(pick, legs, db)
            from sqlalchemy import text as _text
            db.execute(_text(
                "UPDATE mock_bets SET bet_tier=:tier WHERE id=:bid"
            ), {"tier": _tier, "bid": mock_id})
        except Exception:
            pass

        # ── 60% parlay WR floor tags (forward monitoring) ────────────────
        try:
            _n_legs = len(legs)
            _passes, _margin = _compute_floor_pass(legs, _n_legs)
            from sqlalchemy import text as _text
            db.execute(_text(
                "UPDATE mock_bets SET would_pass_floor=:wpf, floor_margin=:fm "
                "WHERE id=:bid"
            ), {"wpf": int(_passes), "fm": _margin, "bid": mock_id})
        except Exception:
            pass

        generated += 1

    forced_count = sum(1 for p in picks if p.get("is_forced_generation"))
    with DB_WRITE_LOCK:
        db.commit()
    # Auto-apply active theses to newly generated legs
    auto_excl = _apply_active_theses(run_id, db)
    if auto_excl:
        print(f"[mock-gen] auto-thesis pass excluded {auto_excl} leg(s)")
    if _cap_skipped:
        print(f"[mock-gen] game concentration cap removed {_cap_skipped} pick(s) (MAX_BETS_PER_GAME={MAX_BETS_PER_GAME})")
    result: dict = {"generated": generated, "skipped_dup": skipped, "skipped_game_cap": _cap_skipped, "run_id": run_id}
    if forced_count > 0:
        result["forced_generation_count"] = forced_count
        result["forced_generation_note"] = (
            f"{forced_count} bet(s) generated from best-available legs "
            f"(no positive-EV opportunity found for those sports today)."
        )

    # Apply live-bet recommendation filter after all picks stored
    if source not in EXPLORATION_SOURCES and generated > 0:
        try:
            live_result = apply_live_recommendation(db)
            n_live = live_result.get("slate_summary", {}).get("passing_live_filter", 0)
            result["live_recommendation"] = {
                "qualifying": n_live,
                "slate_confidence": live_result.get("slate_confidence"),
            }
            if n_live:
                print(f"[mock-gen] live recommendations: {n_live} pick(s), slate={live_result['slate_confidence']}")
        except Exception as _lre:
            print(f"[mock-gen] live recommendation error (non-fatal): {_lre}")

    return result


# ─── Batch Pi enrichment (used by exploration + top_picks paths) ─────────────

def _batch_spread_enrich_run(run_id: str, db) -> int:
    """Backfill spread_value/spread_abs/is_favorite_spread for a generation run."""
    from sqlalchemy import text as _sqla_text
    rows = db.execute(_sqla_text(
        "SELECT mbl.id, mbl.description, mbl.market_type "
        "FROM mock_bet_legs mbl "
        "JOIN mock_bets mb ON mb.id = mbl.mock_bet_id "
        "WHERE mb.generation_run_id = :rid "
        "  AND mbl.spread_value IS NULL "
        "  AND LOWER(mbl.market_type) IN ('alternate_spreads', 'spreads')"
    ), {"rid": run_id}).fetchall()

    enriched = 0
    for leg_id, desc, mkt in rows:
        sv, sa, isf = _parse_spread_for_leg(desc, mkt)
        if sv is not None:
            db.execute(_sqla_text(
                "UPDATE mock_bet_legs SET spread_value=:sv, spread_abs=:sa, "
                "is_favorite_spread=:isf WHERE id=:lid"
            ), {"sv": sv, "sa": sa, "isf": isf, "lid": leg_id})
            enriched += 1
    if enriched:
        with DB_WRITE_LOCK:
            db.commit()
    return enriched


def _batch_pi_enrich_run(run_id: str, db) -> int:
    """
    Backfill home_team_home_pi / away_team_away_pi / pi_diff on all MLB legs
    belonging to the given generation run.  Returns number of legs enriched.

    Uses the fixtures table to resolve home/away teams, then looks up Pi ratings.
    Runs after db.commit() so all legs are visible.
    """
    try:
        from pi_ratings import get_team_pi_for_date as _get_pi
    except ImportError:
        return 0

    from sqlalchemy import text as _sqla_text
    rows = db.execute(_sqla_text(
        "SELECT mbl.id, mbl.fixture_id, mbl.sport, mb.game_date, mbl.description "
        "FROM mock_bet_legs mbl "
        "JOIN mock_bets mb ON mb.id = mbl.mock_bet_id "
        "WHERE mb.generation_run_id = :rid "
        "  AND LOWER(mbl.sport) IN ('mlb', 'baseball_mlb') "
        "  AND mbl.home_team_home_pi IS NULL"
    ), {"rid": run_id}).fetchall()

    enriched = 0
    for leg_id, fid, sport, game_date, description in rows:
        if not fid:
            continue
        try:
            fix = db.execute(_sqla_text(
                "SELECT home_team, away_team FROM fixtures WHERE id=:fid"
            ), {"fid": fid}).fetchone()
            if not fix:
                continue
            ht, at = fix[0], fix[1]
            gd = str(game_date)[:10] if game_date else None
            if not (ht and at and gd):
                continue
            h_pi = _get_pi(ht, gd, "MLB")
            a_pi = _get_pi(at, gd, "MLB")
            if h_pi and a_pi:
                hp = round(h_pi["home_pi"], 4)
                ap = round(a_pi["away_pi"], 4)
                pd = round(hp - ap, 4)  # raw home-away (for ML model)
                # Pick-relative: check if description names the away team
                _desc_lower = (description or "").lower()
                _at_lower = (at or "").lower()
                _is_away_pick = bool(_at_lower and _at_lower in _desc_lower)
                pd_pick = round(-pd if _is_away_pick else pd, 4)
                db.execute(_sqla_text(
                    "UPDATE mock_bet_legs SET home_team_home_pi=:hp, "
                    "away_team_away_pi=:ap, pi_diff=:pd, pi_diff_pick=:pdp WHERE id=:lid"
                ), {"hp": hp, "ap": ap, "pd": pd, "pdp": pd_pick, "lid": leg_id})
                enriched += 1
        except Exception:
            continue

    if enriched:
        with DB_WRITE_LOCK:
            db.commit()
        print(f"[pi-enrich] run={run_id} enriched={enriched} MLB legs")
    return enriched


def _batch_mlb_enrich_run(run_id: str, db) -> int:
    """
    Enrich MLB legs in a generation run with pitcher/weather/Statcast/bullpen
    features.  For same-day games the live API functions are pre-game-safe.
    Runs after db.commit() so all legs are visible.
    """
    from sqlalchemy import text as _sqla_text
    try:
        from mlb_features import (
            get_pitcher_features, get_weather_features,
            get_statcast_features, get_bullpen_features,
            get_h2h_features,
        )
    except ImportError:
        return 0

    rows = db.execute(_sqla_text(
        "SELECT mbl.id, mbl.fixture_id, mbl.sport, mb.game_date "
        "FROM mock_bet_legs mbl "
        "JOIN mock_bets mb ON mb.id = mbl.mock_bet_id "
        "WHERE mb.generation_run_id = :rid "
        "  AND LOWER(mbl.sport) IN ('mlb', 'baseball_mlb') "
        "  AND mbl.home_starter_era IS NULL"
    ), {"rid": run_id}).fetchall()

    if not rows:
        return 0

    enriched = 0
    # Cache per fixture to avoid redundant API calls
    _fix_cache: dict[str, tuple[str, str]] = {}
    _feat_cache: dict[tuple[str, str, str], dict] = {}

    for leg_id, fid, sport, game_date in rows:
        if not fid:
            continue
        try:
            # Resolve home/away teams from fixtures
            if fid not in _fix_cache:
                fix = db.execute(_sqla_text(
                    "SELECT home_team, away_team FROM fixtures WHERE id=:fid"
                ), {"fid": fid}).fetchone()
                if not fix or not fix[0] or not fix[1]:
                    continue
                _fix_cache[fid] = (fix[0], fix[1])
            ht, at = _fix_cache[fid]
            gd = str(game_date)[:10] if game_date else None
            if not gd:
                continue

            cache_key = (ht, at, gd)
            if cache_key not in _feat_cache:
                _vals = {}
                _sport_norm = (sport or "").upper().replace("BASEBALL_", "")
                try:
                    pf = get_pitcher_features(ht, at, gd, _sport_norm)
                    for k, v in pf.items():
                        if v is not None and v == v:
                            _vals[k] = round(float(v), 4)
                except Exception:
                    pass
                try:
                    sf = get_statcast_features(ht, at, _sport_norm)
                    for k, v in sf.items():
                        if v is not None and v == v:
                            _vals[k] = round(float(v), 4)
                except Exception:
                    pass
                try:
                    bf = get_bullpen_features(ht, at, _sport_norm)
                    for k, v in bf.items():
                        if v is not None and v == v:
                            _vals[k] = round(float(v), 4)
                except Exception:
                    pass
                try:
                    hf = get_h2h_features(ht, at, db)
                    for k, v in hf.items():
                        if v is not None and v == v:
                            _vals[k] = round(float(v), 4)
                except Exception:
                    pass
                _feat_cache[cache_key] = _vals

            vals = _feat_cache[cache_key]
            if vals:
                sets = ", ".join(f"{k}=:{k}" for k in vals)
                params = dict(vals)
                params["lid"] = leg_id
                db.execute(_sqla_text(
                    f"UPDATE mock_bet_legs SET {sets} WHERE id=:lid"
                ), params)
                enriched += 1
        except Exception:
            continue

    if enriched:
        with DB_WRITE_LOCK:
            db.commit()
        print(f"[mlb-enrich] run={run_id} enriched={enriched} MLB legs")
    return enriched


# ─── Top-picks seeding ────────────────────────────────────────────────────────

def seed_top_picks_mock_bets(
    cached_picks: list,
    db: Session,
    kelly_fraction: float = 0.25,
    bankroll: float       = 500.0,
) -> dict:
    """
    Convert the in-process picks-cache result into top_picks_page mock_bets.

    Unlike generate_mock_bets (which re-runs generate_todays_picks internally
    and then gets blocked by the prospective-source dedup), this function takes
    the exact picks already surfaced by the recommender, deletes any stale
    top_picks_page entries for today, and inserts fresh ones.

    Stake per bet = kelly_fraction * pick["kelly_500"]
      e.g. kelly_fraction=0.25, bankroll=500 → 25% Kelly on $500 bank.
    If kelly_500 is missing or zero, falls back to bankroll * 0.01 (1% flat).

    Returns: {"generated": int, "replaced": int, "run_id": str}
    """
    run_id   = f"tpp_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
    today_ct = datetime.now(_CT).strftime("%Y-%m-%d")

    # ── 1. Delete existing top_picks_page bets for today ───────────────────
    from sqlalchemy import text as _sqla_text
    replaced = db.execute(
        _sqla_text(
            "SELECT COUNT(*) FROM mock_bets "
            "WHERE source='top_picks_page' AND game_date=:d"
        ),
        {"d": today_ct},
    ).scalar() or 0

    if replaced > 0:
        # Cascade-delete legs first (FK constraint)
        stale_ids = [
            r[0] for r in db.execute(
                _sqla_text(
                    "SELECT id FROM mock_bets "
                    "WHERE source='top_picks_page' AND game_date=:d"
                ),
                {"d": today_ct},
            ).fetchall()
        ]
        for mid in stale_ids:
            db.execute(
                _sqla_text("DELETE FROM mock_bet_legs WHERE mock_bet_id=:mid"),
                {"mid": mid},
            )
        db.execute(
            _sqla_text(
                "DELETE FROM mock_bets "
                "WHERE source='top_picks_page' AND game_date=:d"
            ),
            {"d": today_ct},
        )
        db.flush()

    # ── 2. Insert fresh top_picks_page bets ─────────────────────────────────
    generated = 0
    seen_sigs: set[str] = set()

    # Game-level concentration cap (same as generate_mock_bets)
    MAX_BETS_PER_GAME = 3
    from collections import defaultdict as _dd
    _game_exposure: dict[str, int] = _dd(int)
    # Seed with bets from PRODUCTION sources already stored today
    # (prospective, prospective_pm, top_picks_page, forced_generation).
    # This ensures the cap counts across sources — if prospective already
    # used NYY twice, top_picks can only add 1 more.
    # Shadow/exploration/simulation/retro bets are NOT production and must
    # be excluded — otherwise their volume saturates every fixture and the
    # cap blocks all top_picks_page inserts (bug found 2026-06-29).
    _prior = db.execute(
        _sqla_text(
            "SELECT mbl.fixture_id, COUNT(DISTINCT mb.id) "
            "FROM mock_bets mb "
            "JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id "
            "WHERE mb.game_date = :gd "
            "  AND mb.source IN ('prospective','prospective_pm','top_picks_page','forced_generation') "
            "GROUP BY mbl.fixture_id"
        ),
        {"gd": today_ct},
    ).fetchall()
    for _f, _c in _prior:
        if _f:
            _game_exposure[_f] = _c
    _cap_skipped = 0

    for pick in cached_picks:
        legs = pick.get("legs", [])
        if not legs:
            continue

        sig = " | ".join(
            _format_leg_desc(l) for l in legs if l.get("description") or l.get("pick")
        )
        if not sig or sig in seen_sigs:
            continue

        # Game concentration cap: count ALL sources (prospective + pm + top_picks)
        # and ALL markets on a fixture together — correlated exposure means if a
        # game blows out, ML + spread + total on that fixture all lose.
        _fids = {l.get("fixture_id") or l.get("game_id") for l in legs} - {None}
        if any(_game_exposure[fid] >= MAX_BETS_PER_GAME for fid in _fids):
            _cap_skipped += 1
            continue
        # Accept: increment exposure for all fixtures in this pick
        for fid in _fids:
            _game_exposure[fid] += 1

        seen_sigs.add(sig)

        # Kelly-sized stake: 25% of full-Kelly suggestion at $500 bank
        k500   = float(pick.get("kelly_500") or 0.0)
        amount = round(kelly_fraction * k500, 2) if k500 > 0 else round(bankroll * 0.01, 2)

        sport_set = {l.get("sport", "") for l in legs if l.get("sport")}
        mock_id   = str(uuid.uuid4())

        mock = MockBet(
            id                 = mock_id,
            generation_run_id  = run_id,
            game_date          = today_ct,
            sport              = " | ".join(sport_set) if sport_set else None,
            bet_type           = "parlay" if len(legs) > 1 else "straight",
            odds               = pick.get("combined_odds") or pick.get("odds"),
            amount             = amount,
            legs               = len(legs),
            bet_info           = sig,
            predicted_win_prob = pick.get("combined_win_prob") or pick.get("model_win_prob"),
            predicted_ev       = pick.get("expected_profit") or pick.get("combined_ev"),
            predicted_odds     = pick.get("combined_am"),
            confidence         = pick.get("confidence"),
            model_used         = pick.get("model_used"),
            model_auc          = pick.get("model_auc"),
            status             = "PENDING",
            source             = "top_picks_page",
            avg_lqs            = pick.get("avg_lqs") or pick.get("_avg_lqs"),
            composite_score    = pick.get("_composite") or pick.get("_composite_score"),
            model_confidence_avg = pick.get("model_confidence"),
            notes              = f"kelly25pct={amount} kelly_full={k500}",
            cohort             = "clean",
        )
        db.add(mock)

        # ── Shadow B composition tracking ─────────────────────────
        _sb_adj = pick.get("_comp_adj")
        _sb_fav = pick.get("_comp_fav_count")
        _sb_br  = pick.get("_blind_rank")
        if _sb_adj is not None or _sb_fav is not None or _sb_br is not None:
            try:
                from sqlalchemy import text as _text
                db.execute(_text(
                    "UPDATE mock_bets SET shadow_b_comp_adj=:adj, "
                    "shadow_b_fav_count=:fc, shadow_b_blind_rank=:br WHERE id=:mid"
                ), {"adj": _sb_adj, "fc": _sb_fav, "br": _sb_br, "mid": mock_id})
            except Exception:
                pass

        for i, leg in enumerate(legs):
            _fid  = leg.get("fixture_id") or leg.get("game_id")
            _desc = leg.get("description", "")
            _mkt  = leg.get("market_type") or leg.get("market")
            _open = _lookup_snap_odds(_fid, _desc, _mkt, _HIST_DB_PATH)
            if _open is None:
                _open = _lookup_alt_lines_odds(_fid, _desc, _mkt, _HIST_DB_PATH, earliest=True)
            if _open is None and leg.get("odds"):
                _open = _decimal_to_american(float(leg["odds"]))

            # Enrich leg with scout grade from scouted_props
            _scout_grade    = leg.get("scout_grade")
            _scout_hit_prob = leg.get("scout_hit_prob")
            _scout_prop_id  = leg.get("scouted_prop_id")
            if _scout_grade is None:
                try:
                    _enriched = _plc.attach_scout_grade_to_leg(dict(leg), db)
                    _scout_grade    = _enriched.get("scout_grade")
                    _scout_hit_prob = _enriched.get("scout_hit_prob")
                    _scout_prop_id  = _enriched.get("scouted_prop_id")
                except Exception as _scout_exc:
                    print(f"[tpp-gen] scout enrichment failed for {_desc}: {_scout_exc}")

            leg_obj = MockBetLeg(
                mock_bet_id        = mock_id,
                leg_index          = i,
                description        = _desc,
                market_type        = _mkt,
                sport              = leg.get("sport"),
                win_prob           = leg.get("win_prob"),
                ev                 = leg.get("ev"),
                grade              = leg.get("grade"),
                model_used         = leg.get("model_used") or pick.get("model_used"),
                fixture_id         = _fid,
                is_alt_line        = bool(leg.get("is_alt_line", False)),
                predicted_win_prob = leg.get("win_prob"),
                predicted_edge_pp  = leg.get("edge"),
                open_odds          = _open,
                ale_considered     = bool(leg.get("ale_considered", False)),
                bet_category        = classify_bet_category(_mkt),
            )
            db.add(leg_obj)
            db.flush()

            # Write spread features via raw SQL (not in ORM)
            _sv, _sa, _isf = _parse_spread_for_leg(_desc, _mkt)
            if _sv is not None:
                try:
                    db.execute(_sqla_text(
                        "UPDATE mock_bet_legs SET spread_value=:sv, spread_abs=:sa, "
                        "is_favorite_spread=:isf WHERE id=:lid"
                    ), {"sv": _sv, "sa": _sa, "isf": _isf, "lid": leg_obj.id})
                except Exception:
                    pass

            # Write scout columns via raw SQL (added by safe_add_column, not in ORM)
            if _scout_grade is not None:
                try:
                    db.execute(_sqla_text(
                        "UPDATE mock_bet_legs SET scout_grade=:g, scout_hit_prob=:hp, "
                        "scouted_prop_id=:pid WHERE id=:lid"
                    ), {"g": _scout_grade, "hp": _scout_hit_prob,
                        "pid": _scout_prop_id, "lid": leg_obj.id})
                except Exception:
                    pass

            # ── Persist FD consensus divergence ──────────────────────────────
            try:
                _bk_row = db.execute(_sqla_text(
                    "SELECT bookmakers FROM fixtures WHERE id=:fid"
                ), {"fid": _fid}).fetchone() if _fid else None
                _bk_json = json.loads(_bk_row[0]) if _bk_row and _bk_row[0] else None
                _fd_div, _fd_flag = _compute_fd_field_divergence(leg, _bk_json, _desc)
                if _fd_div is not None or _fd_flag:
                    db.execute(_sqla_text(
                        "UPDATE mock_bet_legs SET fd_field_divergence=:div, "
                        "fd_consensus_flag=:flag WHERE id=:lid"
                    ), {"div": _fd_div, "flag": _fd_flag, "lid": leg_obj.id})
            except Exception:
                pass

            # ── MLB enrichment features (pitcher/weather/statcast/bullpen/H2H) ──
            try:
                _sport_norm = (leg.get("sport") or "").lower()
                _ht = leg.get("resolved_home_team") or leg.get("home_team")
                _at = leg.get("resolved_away_team") or leg.get("away_team")
                if _sport_norm in ("mlb", "baseball_mlb") and _ht and _at:
                    from mlb_features import (
                        get_pitcher_features, get_weather_features,
                        get_statcast_features, get_bullpen_features,
                        get_h2h_features,
                    )
                    _gd_str = pick.get("game_date") or str(today)
                    _commence = leg.get("commence_time") or ""
                    _pf = get_pitcher_features(_ht, _at, str(_gd_str)[:10], _sport_norm)
                    _wf = get_weather_features(_ht, _commence, _sport_norm)
                    _sf = get_statcast_features(_ht, _at, _sport_norm)
                    _bf = get_bullpen_features(_ht, _at, _sport_norm)
                    _hf = get_h2h_features(_ht, _at, db)
                    _vals = {}
                    for _d in (_pf, _wf, _sf, _bf, _hf):
                        for _k, _v in _d.items():
                            if _v is not None and _v == _v:
                                _vals[_k] = round(float(_v), 4)
                    if _vals:
                        _sets = ", ".join(f"{k}=:{k}" for k in _vals)
                        _vals["lid"] = leg_obj.id
                        db.execute(_sqla_text(
                            f"UPDATE mock_bet_legs SET {_sets} WHERE id=:lid"
                        ), _vals)
            except Exception:
                pass

        # ── Tier classification ──────────────────────────────────────────
        try:
            _tier, _ = classify_bet_tier(pick, legs, db)
            db.execute(
                _sqla_text("UPDATE mock_bets SET bet_tier=:tier WHERE id=:bid"),
                {"tier": _tier, "bid": mock_id},
            )
        except Exception:
            pass

        # ── 60% parlay WR floor tags (forward monitoring) ────────────────
        try:
            _n_legs = len(legs)
            _passes, _margin = _compute_floor_pass(legs, _n_legs)
            db.execute(
                _sqla_text("UPDATE mock_bets SET would_pass_floor=:wpf, floor_margin=:fm WHERE id=:bid"),
                {"wpf": int(_passes), "fm": _margin, "bid": mock_id},
            )
        except Exception:
            pass

        generated += 1

    # Guard: if we deleted existing rows but inserted nothing, rollback
    # to preserve the old picks rather than leaving a gap.
    if generated == 0 and replaced > 0:
        db.rollback()
        print(f"[top-picks-seed] ROLLBACK — deleted {replaced} but inserted 0 (cap_skipped={_cap_skipped}). Old picks preserved.")
        return {"generated": 0, "replaced": 0, "skipped_game_cap": _cap_skipped, "run_id": run_id, "rolled_back": True}

    with DB_WRITE_LOCK:
        db.commit()
    _batch_spread_enrich_run(run_id, db)
    _batch_pi_enrich_run(run_id, db)
    _batch_mlb_enrich_run(run_id, db)
    if _cap_skipped:
        print(f"[top-picks-seed] game concentration cap removed {_cap_skipped} pick(s) (MAX_BETS_PER_GAME={MAX_BETS_PER_GAME})")
    print(f"[top-picks-seed] generated={generated} replaced={replaced} cap_skipped={_cap_skipped} run_id={run_id}")
    return {"generated": generated, "replaced": replaced, "skipped_game_cap": _cap_skipped, "run_id": run_id}


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
        with DB_WRITE_LOCK:
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


def _count_exploration_legs_this_week(
    sport: str, market_type: str, bets_db_path: str, db_session=None
) -> int:
    """Return how many exploration legs for this sport+market were created this ISO week."""
    import datetime as _dt
    today = _dt.date.today()
    monday = today - _dt.timedelta(days=today.weekday())
    try:
        if db_session is not None:
            row = db_session.execute(
                _sqla_text("""
                    SELECT COUNT(*) FROM mock_bet_legs mbl
                    JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
                    WHERE mb.source = 'exploration'
                      AND mbl.sport = :sport
                      AND mbl.market_type = :mkt
                      AND date(mb.generated_at) >= :monday
                """),
                {"sport": sport, "mkt": market_type, "monday": monday.isoformat()},
            ).fetchone()
        else:
            con = sqlite3.connect(bets_db_path, timeout=15)
            row = con.execute(
                """
                SELECT COUNT(*) FROM mock_bet_legs mbl
                JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
                WHERE mb.source = 'exploration'
                  AND mbl.sport = ?
                  AND mbl.market_type = ?
                  AND date(mb.generated_at) >= ?
                """,
                (sport, market_type, monday.isoformat()),
            ).fetchone()
            con.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _devig_alt_line(hist_con, event_id: str, market_key: str,
                    team: str, line: float | None,
                    our_odds: float) -> float:
    """Compute devigged (vig-removed) win probability for an alt-line leg.

    For alternate_spreads: the other side is the other team at the negated line.
    For alternate_totals:  the other side is the opposite over/under at the same line.

    Uses multiplicative devig: true_prob = implied_ours / (implied_ours + implied_theirs).
    Falls back to raw implied probability (with vig) if the other side isn't available.
    """
    our_implied = 1.0 / our_odds  # e.g. 0.769 for 1.30 odds

    other_odds = None
    try:
        if market_key == "alternate_spreads" and line is not None:
            # Other side: other team, negated line
            other_line = -line
            row = hist_con.execute(
                "SELECT odds FROM alt_lines "
                "WHERE event_id = ? AND market_key = ? AND team != ? AND line = ? "
                "ORDER BY ABS(odds - 1.9) ASC LIMIT 1",
                (event_id, market_key, team, other_line),
            ).fetchone()
            if row:
                other_odds = float(row[0])
        elif market_key == "alternate_totals" and line is not None:
            # Other side: opposite over/under at same line
            # team may be "Over"/"Under"/"over"/"under" — use case-insensitive compare
            row = hist_con.execute(
                "SELECT odds FROM alt_lines "
                "WHERE event_id = ? AND market_key = ? AND line = ? "
                "AND LOWER(team) != LOWER(?) "
                "ORDER BY ABS(odds - 1.9) ASC LIMIT 1",
                (event_id, market_key, line, team),
            ).fetchone()
            if row:
                other_odds = float(row[0])
    except Exception:
        pass

    if other_odds and other_odds > 1.0:
        other_implied = 1.0 / other_odds
        overround = our_implied + other_implied  # e.g. 1.05 for 5% vig
        devigged = our_implied / overround
        return round(devigged * 100, 4)

    # Fallback: raw implied probability (vig included)
    return round(our_implied * 100, 4)


def devig_alt_line_standalone(event_id: str, market_key: str,
                              team: str, line: float | None,
                              our_odds: float) -> float:
    """Devig an alt-line using historical.db. Opens its own connection.

    Importable by other modules (e.g. recommender.py).
    """
    hist_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
    if not os.path.exists(hist_path):
        return round(100.0 / our_odds, 4)
    try:
        con = sqlite3.connect(hist_path, timeout=10)
        con.row_factory = sqlite3.Row
        result = _devig_alt_line(con, event_id, market_key, team, line, our_odds)
        con.close()
        return result
    except Exception:
        return round(100.0 / our_odds, 4)


def _get_alt_lines_for_tier(
    sport: str,
    market_type: str,
    wp_min: float,
    wp_max: float,
    today: str,
    bets_db_path: str,
    hist_db_path: str,
    db_session=None,        # Optional SQLAlchemy session — reused for fixtures query
                            # to avoid WAL/SHM conflicts when called from scheduler
) -> list[dict]:
    """
    Pull candidates for a given sport+market in the WP range.

    Two data sources depending on market_type:
      alternate_spreads / alternate_totals
          → historical.db  alt_lines table (per-event fetch, many bookmakers)
      h2h / spreads / totals
          → bets.db  fixtures.bookmakers JSON (already paid for by main odds fetch,
            zero additional credit cost)
      player_* props
          → deliberately empty until player prop fetch is enabled

    Returns list of dicts: fixture_id, sport, home_team, away_team,
                           team, odds, win_prob, market_type, line.
    """
    _ALT_MARKETS    = {"alternate_spreads", "alternate_totals"}
    _MAIN_MARKETS   = {"h2h", "spreads", "totals"}
    # Preferred bookmaker order for main-market odds extraction
    _PREFERRED_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pinnacle"]

    # Map exploration market_type aliases → actual OddsAPI market_key(s) in player_props table
    _PROP_MARKET_MAP: dict[str, list[str]] = {
        "player_hits":         ["batter_hits", "batter_hits_alternate"],
        "player_total_bases":  ["batter_total_bases", "batter_total_bases_alternate"],
        "player_points":       ["player_points", "player_points_alternate"],
        "player_rebounds":     ["player_rebounds", "player_rebounds_alternate"],
        "player_assists":      ["player_assists", "player_assists_alternate"],
        "player_shots_on_goal":["player_shots_on_goal", "player_shots_on_goal_alternate"],
    }
    # Also accept raw OddsAPI keys directly
    _ALL_PROP_KEYS = {k for keys in _PROP_MARKET_MAP.values() for k in keys}
    if market_type in _ALL_PROP_KEYS:
        _PROP_MARKET_MAP[market_type] = [market_type]

    # Map sport name to sport_key values used in fixtures table
    _sport_key_map: dict[str, list[str]] = {
        "MLB":       ["baseball_mlb"],
        "NBA":       ["basketball_nba"],
        "NHL":       ["icehockey_nhl"],
        "NFL":       ["americanfootball_nfl"],
        "NCAAF":     ["americanfootball_ncaaf"],
        "NCAAB":     ["basketball_ncaab"],
        "MLS":       ["soccer_usa_mls"],
        "WORLD CUP": ["soccer_fifa_world_cup"],
    }
    sport_sport_keys = {
        "MLB": "baseball_mlb", "NBA": "basketball_nba",
        "NHL": "icehockey_nhl", "NFL": "americanfootball_nfl",
    }
    sport_keys = _sport_key_map.get(sport.upper(), [])

    _is_prop = market_type in _PROP_MARKET_MAP

    # Unknown market type — return empty
    if market_type not in _ALT_MARKETS and market_type not in _MAIN_MARKETS and not _is_prop:
        return []

    # ── Fetch today's fixtures (bookmakers JSON included for main-market path) ──
    # Prefers the caller's SQLAlchemy session to avoid WAL/SHM conflicts that
    # produce SQLITE_IOERR when a raw sqlite3 connection races the session's WAL.
    fid_rows = []
    try:
        if db_session is not None:
            # Use existing session connection — no WAL conflict possible
            from sqlalchemy import text as _sqla_text
            if sport_keys:
                _ph = ",".join(f"'{k}'" for k in sport_keys)
                _rows = db_session.execute(
                    _sqla_text(
                        f"SELECT id, home_team, away_team, bookmakers FROM fixtures "
                        f"WHERE game_date_ct = :d AND sport_key IN ({_ph})"
                    ),
                    {"d": today},
                ).fetchall()
            else:
                _rows = db_session.execute(
                    _sqla_text(
                        "SELECT id, home_team, away_team, bookmakers FROM fixtures "
                        "WHERE game_date_ct = :d AND UPPER(sport_title) LIKE :s"
                    ),
                    {"d": today, "s": f"%{sport.upper()}%"},
                ).fetchall()
            fid_rows = [tuple(r) for r in _rows]
        else:
            # Fallback: raw sqlite3 with retry (standalone/test callers)
            for _attempt in range(3):
                try:
                    con = sqlite3.connect(bets_db_path, timeout=15)
                    if sport_keys:
                        placeholders = ",".join("?" * len(sport_keys))
                        fid_rows = con.execute(
                            f"SELECT id, home_team, away_team, bookmakers FROM fixtures "
                            f"WHERE game_date_ct = ? AND sport_key IN ({placeholders})",
                            [today] + sport_keys,
                        ).fetchall()
                    else:
                        fid_rows = con.execute(
                            "SELECT id, home_team, away_team, bookmakers FROM fixtures "
                            "WHERE game_date_ct = ? AND UPPER(sport_title) LIKE ?",
                            (today, f"%{sport.upper()}%"),
                        ).fetchall()
                    con.close()
                    break
                except Exception as _e:
                    if _attempt < 2:
                        import time as _time
                        _time.sleep(1 + _attempt)
                    else:
                        print(f"[exploration] fixtures query error: {_e}")
                        return []
    except Exception as _e:
        print(f"[exploration] fixtures query error: {_e}")
        return []

    if not fid_rows:
        return []

    results: list[dict] = []

    # ── Path A: alternate_spreads / alternate_totals → alt_lines table ───────
    if market_type in _ALT_MARKETS:
        try:
            hist_con = sqlite3.connect(hist_db_path, timeout=15)
            hist_con.row_factory = sqlite3.Row
            for fid_row in fid_rows:
                fid  = fid_row[0]
                home = fid_row[1] or ""
                away = fid_row[2] or ""
                alt_rows = hist_con.execute(
                    """
                    SELECT team, odds, line, market_key
                    FROM alt_lines
                    WHERE event_id = ? AND market_key = ?
                    ORDER BY ABS(odds - 1.9) ASC
                    LIMIT 6
                    """,
                    (fid, market_type),
                ).fetchall()
                for ar in alt_rows:
                    odds_dec = float(ar["odds"] or 0)
                    if odds_dec <= 1.01:
                        continue
                    wp_pct = _devig_alt_line(
                        hist_con, fid, market_type,
                        ar["team"] or "", ar["line"], odds_dec,
                    )
                    if not (wp_min <= wp_pct <= wp_max):
                        continue
                    results.append({
                        "fixture_id":  fid,
                        "sport":       sport,
                        "home_team":   home,
                        "away_team":   away,
                        "team":        ar["team"] or "",
                        "odds":        odds_dec,
                        "win_prob":    wp_pct,
                        "market_type": market_type,
                        "line":        ar["line"],
                    })
            hist_con.close()
        except Exception as _e:
            print(f"[exploration] alt_lines query error: {_e}")

    # ── Path C: player props → historical.db player_props table ─────────────
    elif _is_prop:
        prop_keys = _PROP_MARKET_MAP[market_type]
        sport_key_val = sport_sport_keys.get(sport.upper(), "")
        key_placeholders = ",".join("?" * len(prop_keys))
        try:
            hist_con = sqlite3.connect(hist_db_path, timeout=15)
            hist_con.row_factory = sqlite3.Row
            # Best odds per player/market/side across bookmakers for today's games
            # Try today first; fall back to the most recent available date if empty.
            # Player prop lines don't shift dramatically overnight, so same-day +1 is fine.
            for _prop_date in [today, None]:
                if _prop_date is not None:
                    _date_clause = "DATE(fetched_at) = ?"
                    _date_params = [_prop_date, sport_key_val] + prop_keys
                else:
                    # Use MAX(DATE(fetched_at)) as the fallback date
                    _max_date_row = hist_con.execute(
                        f"SELECT MAX(DATE(fetched_at)) FROM player_props "
                        f"WHERE sport_key = ? AND market_key IN ({key_placeholders})",
                        [sport_key_val] + prop_keys,
                    ).fetchone()
                    _fallback_date = _max_date_row[0] if _max_date_row else None
                    if not _fallback_date:
                        break
                    _date_clause = "DATE(fetched_at) = ?"
                    _date_params = [_fallback_date, sport_key_val] + prop_keys
                prop_rows = hist_con.execute(
                    f"""
                    SELECT player_name, market_key, over_under, point,
                           MAX(price) AS best_price,
                           COUNT(DISTINCT bookmaker) AS n_books,
                           event_id, home_team, away_team
                    FROM player_props
                    WHERE {_date_clause}
                      AND sport_key = ?
                      AND market_key IN ({key_placeholders})
                    GROUP BY player_name, market_key, over_under, point, event_id
                    HAVING n_books >= 2
                    """,
                    _date_params,
                ).fetchall()
                if prop_rows:
                    break
            hist_con.close()
            # Only keep props whose event_id is a fixture playing TODAY.
            # Without this, the fallback-date path returns event_ids from past
            # games (different CDT date), which the stale-fixture gate then
            # correctly expires — but the bet should never have been created.
            today_fixture_ids = {r[0] for r in fid_rows}
            prop_rows = [pr for pr in prop_rows if pr["event_id"] in today_fixture_ids]
            # Build fixture_id lookup: event_id → fixture id from bets.db
            fid_by_event = {r[0]: r[0] for r in fid_rows}  # event_id IS the fixture id
            for pr in prop_rows:
                odds_dec = float(pr["best_price"] or 0)
                if odds_dec <= 1.01:
                    continue
                wp_pct = round(100.0 / odds_dec, 4)
                if not (wp_min <= wp_pct <= wp_max):
                    continue
                desc = (f"{pr['player_name']} {pr['over_under'].capitalize()} "
                        f"{pr['point']} {pr['market_key'].replace('_',' ').title()}")
                results.append({
                    "fixture_id":  pr["event_id"],
                    "sport":       sport,
                    "home_team":   pr["home_team"] or "",
                    "away_team":   pr["away_team"] or "",
                    "team":        desc,
                    "odds":        odds_dec,
                    "win_prob":    wp_pct,
                    "market_type": market_type,
                    "line":        pr["point"],
                })
        except Exception as _e:
            print(f"[exploration] player_props query error: {_e}")

    # ── Path B: h2h / spreads / totals → fixtures.bookmakers JSON ────────────
    else:
        for fid_row in fid_rows:
            fid        = fid_row[0]
            home       = fid_row[1] or ""
            away       = fid_row[2] or ""
            bm_raw     = fid_row[3]
            if not bm_raw:
                continue
            try:
                bookmakers = json.loads(bm_raw)
            except Exception:
                continue

            # Pick preferred bookmaker
            bk = None
            for preferred in _PREFERRED_BOOKS:
                bk = next((b for b in bookmakers if b.get("key") == preferred), None)
                if bk:
                    break
            if bk is None and bookmakers:
                bk = bookmakers[0]
            if bk is None:
                continue

            # Find the target market inside this bookmaker
            mkt_data = next(
                (m for m in bk.get("markets", []) if m.get("key") == market_type),
                None,
            )
            if mkt_data is None:
                continue

            for oc in mkt_data.get("outcomes", []):
                odds_dec = float(oc.get("price") or 0)
                if odds_dec <= 1.01:
                    continue
                wp_pct = round(100.0 / odds_dec, 4)
                if not (wp_min <= wp_pct <= wp_max):
                    continue
                results.append({
                    "fixture_id":  fid,
                    "sport":       sport,
                    "home_team":   home,
                    "away_team":   away,
                    "team":        oc.get("name") or "",
                    "odds":        odds_dec,
                    "win_prob":    wp_pct,
                    "market_type": market_type,
                    "line":        oc.get("point"),
                })

    return results


def _dec_to_american(dec: float) -> int:
    """Convert decimal odds (e.g. 2.50) to American odds (+150 / -200)."""
    if dec >= 2.0:
        return round((dec - 1) * 100)
    elif dec > 1.0:
        return round(-100 / (dec - 1))
    return 0


def generate_tiered_exploration_bets(
    db: Session,
    run_id: str | None = None,
    game_date: str | None = None,
    daily_target: int | None = None,
) -> dict:
    """
    Generate exploration bets structured by diagnostic tier.

    Tier 1 Validation (80%): proven-edge markets, full WP range — confirm calibration.
    Tier 2 Expansion (15%): new markets — build data to decide if worth promoting.
    Tier 3 Boundary Testing (5%): banned zones — detect recovery.

    All bets: source='exploration', exploration_tier=<tier>, amount=0.0.
    Performance is tracked separately — never contaminates prospective metrics.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8] + "_expl_tiered"
    if daily_target is None:
        daily_target = _EXPLORATION_DAILY_TARGET

    _today = game_date or datetime.now(_CT).strftime("%Y-%m-%d")

    # Dedup: existing exploration bet_infos for today — use the session to avoid WAL conflicts
    try:
        existing_infos: set[str] = {
            r[0]
            for r in db.execute(
                _sqla_text(
                    "SELECT bet_info FROM mock_bets WHERE source = 'exploration' AND game_date = :d"
                ),
                {"d": _today},
            ).fetchall()
        }
    except Exception:
        existing_infos = set()

    created_by_tier: dict[str, int] = {"validation": 0, "expansion": 0, "boundary_testing": 0}
    created_by_market: dict[tuple, int] = {}   # (sport, market_type) → count today
    skipped = 0

    for tier_name, config in EXPLORATION_BUDGET.items():
        tier_target = max(1, int(daily_target * config["allocation"]))
        wp_min, wp_max = config["wp_range"]
        markets = config["markets"]
        max_per_week = config.get("max_per_market_per_week")

        # Boundary testing runs weekly per market (not daily)
        if config.get("frequency") == "weekly":
            import datetime as _dt
            today_obj = _dt.date.fromisoformat(_today)
            # Only run on Monday (weekday 0)
            if today_obj.weekday() != 0:
                continue

        per_market_target = max(1, tier_target // max(len(markets), 1))
        tier_created = 0

        for sport, market_type in markets:
            if tier_created >= tier_target:
                break

            # Weekly rate limit
            if max_per_week is not None:
                week_count = _count_exploration_legs_this_week(sport, market_type, _BETS_DB_PATH, db_session=db)
                if week_count >= max_per_week:
                    continue
                available = min(per_market_target, max_per_week - week_count)
            else:
                available = per_market_target

            # Pull candidates in WP range
            candidates = _get_alt_lines_for_tier(
                sport=sport,
                market_type=market_type,
                wp_min=wp_min,
                wp_max=wp_max,
                today=_today,
                bets_db_path=_BETS_DB_PATH,
                hist_db_path=_HIST_DB_PATH,
                db_session=db,
            )

            # Shuffle for variety; pick up to available
            random.shuffle(candidates)
            candidates = candidates[:available]

            for cand in candidates:
                # Build bet_info for dedup
                team_str = cand.get("team") or ""
                line_str = f" {cand['line']:+g}" if cand.get("line") is not None else ""
                bet_info = f"{team_str}{line_str} ({market_type}) [{tier_name[:3].upper()}]"

                if bet_info in existing_infos:
                    skipped += 1
                    continue
                existing_infos.add(bet_info)

                wp = cand["win_prob"]
                odds = cand["odds"]
                mock_id = str(uuid.uuid4())[:16]

                try:
                    db.add(MockBet(
                        id                = mock_id,
                        generation_run_id = run_id,
                        game_date         = _today,
                        sport             = sport,
                        bet_type          = "straight",
                        odds              = odds,
                        amount            = 0.0,
                        legs              = 1,
                        bet_info          = bet_info,
                        predicted_win_prob = wp / 100.0,
                        predicted_ev      = None,
                        confidence        = None,
                        model_used        = "exploration",
                        model_auc         = None,
                        status            = "PENDING",
                        source            = "exploration",
                        exploration_tier  = tier_name,
                        avg_lqs           = None,
                        notes             = f"Tiered exploration [{tier_name}]: {sport} {market_type}",
                    ))
                    _open_american = _dec_to_american(odds) if odds > 1.0 else None
                    db.add(MockBetLeg(
                        mock_bet_id        = mock_id,
                        leg_index          = 0,
                        description        = bet_info.split(" [")[0],
                        market_type        = market_type,
                        sport              = sport,
                        win_prob           = wp,
                        ev                 = None,
                        grade              = None,
                        model_used         = "exploration",
                        fixture_id         = cand.get("fixture_id"),
                        is_alt_line        = True,
                        predicted_win_prob = wp / 100.0,
                        predicted_edge_pp  = None,
                        bet_category        = classify_bet_category(market_type),
                        open_odds          = _open_american,
                    ))
                    tier_created += 1
                    created_by_tier[tier_name] = created_by_tier.get(tier_name, 0) + 1
                    _mkey = (sport, market_type)
                    created_by_market[_mkey] = created_by_market.get(_mkey, 0) + 1
                except Exception:
                    skipped += 1

    # ── Player props guarantee ────────────────────────────────────────────────
    # Ensure every active prop market gets at least _PROP_DAILY_MIN exploration
    # bets each day regardless of tier-budget crowding.  Without this, spreads
    # dominate the 6-slot expansion budget and props never accumulate signal.
    _PROP_DAILY_MIN = 2
    _PROP_GUARANTEE_MARKETS = [
        ("NBA", "player_points"),
        ("NBA", "player_rebounds"),
        ("NBA", "player_assists"),
        ("MLB", "player_hits"),
        ("MLB", "player_total_bases"),
        ("NHL", "player_shots_on_goal"),
    ]
    props_guaranteed = 0
    for sport, market_type in _PROP_GUARANTEE_MARKETS:
        already = created_by_market.get((sport, market_type), 0)
        need = _PROP_DAILY_MIN - already
        if need <= 0:
            continue
        try:
            candidates = _get_alt_lines_for_tier(
                sport=sport,
                market_type=market_type,
                wp_min=55.0,
                wp_max=85.0,
                today=_today,
                bets_db_path=_BETS_DB_PATH,
                hist_db_path=_HIST_DB_PATH,
                db_session=db,
            )
        except Exception:
            continue
        if not candidates:
            continue
        random.shuffle(candidates)
        for cand in candidates[:need]:
            team_str = cand.get("team") or ""
            line_str = f" {cand['line']:+g}" if cand.get("line") is not None else ""
            bet_info = f"{team_str}{line_str} ({market_type}) [EXP]"
            if bet_info in existing_infos:
                skipped += 1
                continue
            existing_infos.add(bet_info)
            wp = cand["win_prob"]
            odds = cand["odds"]
            mock_id = str(uuid.uuid4())[:16]
            try:
                db.add(MockBet(
                    id                = mock_id,
                    generation_run_id = run_id,
                    game_date         = _today,
                    sport             = sport,
                    bet_type          = "straight",
                    odds              = odds,
                    amount            = 0.0,
                    legs              = 1,
                    bet_info          = bet_info,
                    predicted_win_prob = wp / 100.0,
                    predicted_ev      = None,
                    confidence        = None,
                    model_used        = "exploration",
                    model_auc         = None,
                    status            = "PENDING",
                    source            = "exploration",
                    exploration_tier  = "expansion",
                    avg_lqs           = None,
                    notes             = f"Props guarantee: {sport} {market_type}",
                ))
                _open_american = _dec_to_american(odds) if odds > 1.0 else None
                db.add(MockBetLeg(
                    mock_bet_id        = mock_id,
                    leg_index          = 0,
                    description        = bet_info.split(" [")[0],
                    market_type        = market_type,
                    sport              = sport,
                    win_prob           = wp,
                    ev                 = None,
                    grade              = None,
                    model_used         = "exploration",
                    fixture_id         = cand.get("fixture_id"),
                    is_alt_line        = True,
                    predicted_win_prob = wp / 100.0,
                    predicted_edge_pp  = None,
                    bet_category        = classify_bet_category(market_type),
                    open_odds          = _open_american,
                ))
                created_by_tier["expansion"] = created_by_tier.get("expansion", 0) + 1
                props_guaranteed += 1
            except Exception:
                skipped += 1

    # ── MLB standard spreads guarantee ────────────────────────────────────────
    # Production bets MLB spreads (~6/day) but boundary_testing only generates
    # ~0.7/day (weekly, max 5).  This floor ensures the calibration table's
    # MLB spreads 75-85 cell (the +27.7pp correction) stays data-fed.
    _SPREAD_DAILY_MIN = 6
    _SPREAD_GUARANTEE_MARKETS = [
        ("MLB", "spreads"),
        ("NHL", "alternate_totals"),
    ]
    spreads_guaranteed = 0
    for sport, market_type in _SPREAD_GUARANTEE_MARKETS:
        already = created_by_market.get((sport, market_type), 0)
        need = _SPREAD_DAILY_MIN - already
        if need <= 0:
            continue
        try:
            candidates = _get_alt_lines_for_tier(
                sport=sport,
                market_type=market_type,
                wp_min=55.0,
                wp_max=90.0,
                today=_today,
                bets_db_path=_BETS_DB_PATH,
                hist_db_path=_HIST_DB_PATH,
                db_session=db,
            )
        except Exception:
            continue
        if not candidates:
            continue
        random.shuffle(candidates)
        for cand in candidates[:need]:
            team_str = cand.get("team") or ""
            line_str = f" {cand['line']:+g}" if cand.get("line") is not None else ""
            bet_info = f"{team_str}{line_str} ({market_type}) [CAL]"
            if bet_info in existing_infos:
                skipped += 1
                continue
            existing_infos.add(bet_info)
            wp = cand["win_prob"]
            odds = cand["odds"]
            mock_id = str(uuid.uuid4())[:16]
            try:
                db.add(MockBet(
                    id                = mock_id,
                    generation_run_id = run_id,
                    game_date         = _today,
                    sport             = sport,
                    bet_type          = "straight",
                    odds              = odds,
                    amount            = 0.0,
                    legs              = 1,
                    bet_info          = bet_info,
                    predicted_win_prob = wp / 100.0,
                    predicted_ev      = None,
                    confidence        = None,
                    model_used        = "exploration",
                    model_auc         = None,
                    status            = "PENDING",
                    source            = "exploration",
                    exploration_tier  = "validation",
                    avg_lqs           = None,
                    notes             = f"Calibration guarantee: {sport} {market_type}",
                ))
                _open_american = _dec_to_american(odds) if odds > 1.0 else None
                db.add(MockBetLeg(
                    mock_bet_id        = mock_id,
                    leg_index          = 0,
                    description        = bet_info.split(" [")[0],
                    market_type        = market_type,
                    sport              = sport,
                    win_prob           = wp,
                    ev                 = None,
                    grade              = None,
                    model_used         = "exploration",
                    fixture_id         = cand.get("fixture_id"),
                    is_alt_line        = market_type.startswith("alternate"),
                    predicted_win_prob = wp / 100.0,
                    predicted_edge_pp  = None,
                    bet_category        = classify_bet_category(market_type),
                    open_odds          = _open_american,
                ))
                created_by_tier["validation"] = created_by_tier.get("validation", 0) + 1
                spreads_guaranteed += 1
                _mkey = (sport, market_type)
                created_by_market[_mkey] = created_by_market.get(_mkey, 0) + 1
            except Exception:
                skipped += 1

    # ── MLB totals calibration guarantee ─────────────────────────────────────
    # 40 legs/day across 12 cells (6 lines × over/under).  Sources from
    # alt_lines (alternate_totals) — data already fetched, zero API cost.
    # Feeds the mlb_totals sub-model's calibration pipeline.
    _TOTALS_DAILY_MIN = 40
    _TOTALS_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    _TOTALS_DIRS  = ["over", "under"]
    _TOTALS_PER_CELL = max(1, _TOTALS_DAILY_MIN // (len(_TOTALS_LINES) * len(_TOTALS_DIRS)))  # ~3
    totals_guaranteed = 0
    totals_already = created_by_market.get(("MLB", "alternate_totals"), 0)
    if totals_already < _TOTALS_DAILY_MIN:
        try:
            _hist_con = sqlite3.connect(_HIST_DB_PATH, timeout=15)
            _hist_con.row_factory = sqlite3.Row
            # Fetch all MLB alternate_totals for today's games from alt_lines
            _tot_rows = _hist_con.execute(
                """
                SELECT al.event_id, al.home_team, al.away_team, al.line,
                       al.over_under, al.odds, al.bookmaker
                FROM alt_lines al
                WHERE al.sport_key = 'baseball_mlb'
                  AND al.market_key = 'alternate_totals'
                  AND COALESCE(al.game_date_ct, al.game_date) = :today
                  AND al.line IN (6.5, 7.5, 8.5, 9.5, 10.5, 11.5)
                  AND al.odds > 1.05
                ORDER BY al.line, al.over_under, ABS(al.odds - 2.0)
                """,
                {"today": _today},
            ).fetchall()
            _hist_con.close()

            # Group by (line, direction) and pick best odds per game
            from collections import defaultdict as _ddict
            _cell_cands: dict[tuple, list] = _ddict(list)
            _seen_games: dict[tuple, set] = _ddict(set)
            for r in _tot_rows:
                _key = (r["line"], r["over_under"])
                _game_key = (r["home_team"], r["away_team"])
                # One candidate per game per cell (best odds first due to ORDER BY)
                if _game_key in _seen_games[_key]:
                    continue
                _seen_games[_key].add(_game_key)
                _cell_cands[_key].append(dict(r))

            # Try to get model WP; fall back to implied
            try:
                from ml_model import predict_game_totals as _pgt
            except ImportError:
                _pgt = None

            for line_val in _TOTALS_LINES:
                for direction in _TOTALS_DIRS:
                    _key = (line_val, direction)
                    cands = _cell_cands.get(_key, [])
                    random.shuffle(cands)
                    for cand in cands[:_TOTALS_PER_CELL]:
                        odds_dec = float(cand["odds"])
                        impl_wp = devig_alt_line_standalone(
                            cand["event_id"], "alternate_totals",
                            cand.get("over_under", ""), line_val, odds_dec,
                        )

                        # Model WP
                        _model_wp = None
                        if _pgt is not None:
                            try:
                                _over_p = _pgt(
                                    cand["home_team"], cand["away_team"],
                                    float(line_val), _today,
                                )
                                if _over_p is not None:
                                    _model_wp = round(
                                        (_over_p if direction == "over" else 1.0 - _over_p) * 100, 2
                                    )
                            except Exception:
                                pass
                        wp = _model_wp if _model_wp is not None else impl_wp

                        dir_label = direction.capitalize()
                        bet_info = f"{dir_label} {line_val} ({cand['home_team']} vs {cand['away_team']}) [CAL-TOT]"
                        if bet_info in existing_infos:
                            skipped += 1
                            continue
                        existing_infos.add(bet_info)

                        mock_id = str(uuid.uuid4())[:16]
                        try:
                            db.add(MockBet(
                                id                = mock_id,
                                generation_run_id = run_id,
                                game_date         = _today,
                                sport             = "MLB",
                                bet_type          = "straight",
                                odds              = odds_dec,
                                amount            = 0.0,
                                legs              = 1,
                                bet_info          = bet_info,
                                predicted_win_prob = wp / 100.0,
                                predicted_ev      = None,
                                confidence        = None,
                                model_used        = "exploration",
                                model_auc         = None,
                                status            = "PENDING",
                                source            = "exploration",
                                exploration_tier  = "validation",
                                avg_lqs           = None,
                                notes             = f"Totals calibration: {direction} {line_val}",
                            ))
                            _open_american = _dec_to_american(odds_dec) if odds_dec > 1.0 else None
                            db.add(MockBetLeg(
                                mock_bet_id        = mock_id,
                                leg_index          = 0,
                                description        = bet_info.split(" [")[0],
                                market_type        = "alternate_totals",
                                sport              = "MLB",
                                win_prob           = wp,
                                ev                 = None,
                                grade              = None,
                                model_used         = "mlb_totals_v1" if _model_wp is not None else "exploration",
                                fixture_id         = cand.get("event_id"),
                                is_alt_line        = True,
                                predicted_win_prob = wp / 100.0,
                                predicted_edge_pp  = None,
                                bet_category       = classify_bet_category("alternate_totals"),
                                open_odds          = _open_american,
                            ))
                            created_by_tier["validation"] = created_by_tier.get("validation", 0) + 1
                            totals_guaranteed += 1
                            _mkey = ("MLB", "alternate_totals")
                            created_by_market[_mkey] = created_by_market.get(_mkey, 0) + 1
                        except Exception:
                            skipped += 1
        except Exception as _tot_err:
            print(f"[exploration] totals guarantee error: {_tot_err}")

    total_created = sum(created_by_tier.values())
    if total_created > 0:
        try:
            with DB_WRITE_LOCK:
                db.commit()
        except Exception as exc:
            db.rollback()
            return {"error": str(exc), "created_by_tier": created_by_tier}
        _batch_spread_enrich_run(run_id, db)
        _batch_pi_enrich_run(run_id, db)
        _batch_mlb_enrich_run(run_id, db)
        print(
            f"[exploration-tiered] created={total_created} props_guaranteed={props_guaranteed} spreads_guaranteed={spreads_guaranteed} totals_guaranteed={totals_guaranteed} skipped={skipped} "
            + "  ".join(f"{k}={v}" for k, v in created_by_tier.items())
        )
    else:
        print(f"[exploration-tiered] no new bets created (skipped={skipped})")

    return {
        "created":          total_created,
        "props_guaranteed": props_guaranteed,
        "skipped_dup":      skipped,
        "created_by_tier":  created_by_tier,
    }


def generate_daily_sims(
    db: Session,
    game_date: str | None = None,
    bets_per_game: int = 3,
) -> dict:
    """
    Post-picks simulation pass — generates simulation bets for ALL of today's games.

    Unlike generate_mock_bets (which runs the recommender and picks the top N),
    this ensures every game on the slate has at least `bets_per_game` simulation
    bets regardless of whether it competed successfully in the main picks round.
    Run after the morning AND PM mock-generate cycles so the Knicks/Cavs or any
    late-starting game always has coverage to build signal.

    Markets attempted per game (in priority order):
      alternate_spreads → h2h → spreads

    Bets are tagged source='simulation', amount=0, and deduped against
    existing simulation bets for today so re-running is idempotent.
    """
    from sqlalchemy import text as _sqla_text
    _today = game_date or datetime.now(_CT).strftime("%Y-%m-%d")

    # Dedup: existing simulation bet_infos for today
    try:
        existing_infos: set[str] = {
            r[0]
            for r in db.execute(
                _sqla_text(
                    "SELECT bet_info FROM mock_bets WHERE source = 'simulation' AND game_date = :d"
                ),
                {"d": _today},
            ).fetchall()
        }
    except Exception:
        existing_infos = set()

    _SPORTS = [
        ("NBA", "basketball_nba"),
        ("MLB", "baseball_mlb"),
        ("NHL", "icehockey_nhl"),
    ]
    _MARKET_PRIORITY = ["alternate_spreads", "h2h", "spreads"]

    created = 0
    skipped = 0
    by_sport: dict[str, int] = {}

    for sport, _sport_key in _SPORTS:
        # Get today's fixtures for this sport (CT date — DST-correct)
        try:
            fid_rows = db.execute(
                _sqla_text(
                    "SELECT id, home_team, away_team FROM fixtures "
                    "WHERE game_date_ct = :d AND sport_key = :sk"
                ),
                {"d": _today, "sk": _sport_key},
            ).fetchall()
        except Exception as _fe:
            print(f"[daily-sims] fixture fetch error for {sport}: {_fe}")
            continue

        if not fid_rows:
            continue

        sport_created = 0

        for fid_row in fid_rows:
            fid   = fid_row[0]
            home  = fid_row[1] or ""
            away  = fid_row[2] or ""
            game_created = 0

            for market_type in _MARKET_PRIORITY:
                if game_created >= bets_per_game:
                    break

                candidates = _get_alt_lines_for_tier(
                    sport        = sport,
                    market_type  = market_type,
                    wp_min       = 52.0,
                    wp_max       = 92.0,
                    today        = _today,
                    bets_db_path = _BETS_DB_PATH,
                    hist_db_path = _HIST_DB_PATH,
                    db_session   = db,
                )
                # Filter to this specific fixture
                candidates = [c for c in candidates if c.get("fixture_id") == fid]
                if not candidates:
                    # If no fixture-specific candidates, use any for the matchup via team name
                    candidates = _get_alt_lines_for_tier(
                        sport        = sport,
                        market_type  = market_type,
                        wp_min       = 52.0,
                        wp_max       = 92.0,
                        today        = _today,
                        bets_db_path = _BETS_DB_PATH,
                        hist_db_path = _HIST_DB_PATH,
                        db_session   = db,
                    )
                    candidates = [
                        c for c in candidates
                        if (home.split()[-1] in (c.get("team") or "")
                            or away.split()[-1] in (c.get("team") or ""))
                    ]

                random.shuffle(candidates)
                need = bets_per_game - game_created
                for cand in candidates[:need]:
                    team_str = cand.get("team") or ""
                    line_str = f" {cand['line']:+g}" if cand.get("line") is not None else ""
                    bet_info = f"{team_str}{line_str} ({market_type}) [SIM]"
                    if bet_info in existing_infos:
                        skipped += 1
                        continue
                    existing_infos.add(bet_info)
                    wp   = cand["win_prob"]
                    odds = cand["odds"]
                    mock_id = str(uuid.uuid4())[:16]
                    try:
                        db.add(MockBet(
                            id                 = mock_id,
                            generation_run_id  = f"daily_sims_{_today}",
                            game_date          = _today,
                            sport              = sport,
                            bet_type           = "straight",
                            odds               = odds,
                            amount             = 0.0,
                            legs               = 1,
                            bet_info           = bet_info,
                            predicted_win_prob = wp / 100.0,
                            predicted_ev       = None,
                            confidence         = None,
                            model_used         = "daily_sims",
                            model_auc          = None,
                            status             = "PENDING",
                            source             = "simulation",
                            exploration_tier   = None,
                            avg_lqs            = None,
                            notes              = f"Daily sim: {sport} {home} vs {away} [{market_type}]",
                        ))
                        _open_american = _dec_to_american(odds) if odds > 1.0 else None
                        db.add(MockBetLeg(
                            mock_bet_id        = mock_id,
                            leg_index          = 0,
                            description        = bet_info.split(" [")[0],
                            market_type        = market_type,
                            sport              = sport,
                            win_prob           = wp,
                            ev                 = None,
                            grade              = None,
                            model_used         = "daily_sims",
                            fixture_id         = fid,
                            is_alt_line        = market_type in {"alternate_spreads", "alternate_totals"},
                            predicted_win_prob = wp / 100.0,
                            predicted_edge_pp  = None,
                            bet_category       = classify_bet_category(market_type),
                            open_odds          = _open_american,
                        ))
                        game_created  += 1
                        sport_created += 1
                        created += 1
                    except Exception:
                        skipped += 1

        by_sport[sport] = sport_created

    _sims_run_id = f"daily_sims_{_today}"
    if created > 0:
        try:
            with DB_WRITE_LOCK:
                db.commit()
        except Exception as exc:
            db.rollback()
            return {"error": str(exc), "created": 0, "by_sport": by_sport}
        _batch_spread_enrich_run(_sims_run_id, db)
        _batch_pi_enrich_run(_sims_run_id, db)
        _batch_mlb_enrich_run(_sims_run_id, db)
        print(f"[daily-sims] created={created} skipped={skipped} "
              + " ".join(f"{k}={v}" for k, v in by_sport.items()))
    else:
        print(f"[daily-sims] no new sims created (skipped={skipped})")

    return {"created": created, "skipped_dup": skipped, "by_sport": by_sport}


def get_exploration_calibration(days: int = 30) -> dict:
    """
    Per-tier per-market performance for the exploration calibration endpoint.

    Returns data to drive promotion/demotion decisions:
      - Expansion tier markets with 30+ legs and WR >= expected - 5pp → 'promote'
      - Expansion tier markets with WR < expected - 15pp → 'demote'
      - Boundary testing markets showing improvement → 'monitor'
    """
    since = (datetime.now(_CT) - timedelta(days=days)).strftime("%Y-%m-%d")

    results: dict[str, list] = {
        "validation": [],
        "expansion": [],
        "boundary_testing": [],
    }

    try:
        con = sqlite3.connect(_BETS_DB_PATH, timeout=15)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                mb.exploration_tier,
                mbl.sport,
                mbl.market_type,
                COUNT(*) as n,
                SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END) as wins,
                ROUND(100.0 * SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0), 1) as actual_wr_pct,
                ROUND(AVG(mbl.win_prob), 1) as avg_wp
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source = 'exploration'
              AND mb.exploration_tier IS NOT NULL
              AND mbl.leg_result IN ('WIN','LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND date(mb.generated_at) >= ?
            GROUP BY mb.exploration_tier, mbl.sport, mbl.market_type
            ORDER BY mb.exploration_tier, COUNT(*) DESC
            """,
            (since,),
        ).fetchall()
        con.close()
    except Exception as exc:
        return {"error": str(exc)}

    for r in rows:
        tier   = r["exploration_tier"] or "unknown"
        sport  = r["sport"] or ""
        mkt    = r["market_type"] or ""
        n      = r["n"] or 0
        actual_wr = (r["actual_wr_pct"] or 0) / 100.0
        avg_wp    = r["avg_wp"] or 0

        expected_wr = _EXPLORATION_EXPECTED_WR.get((sport, mkt), 0.58)

        # Verdict logic
        if n < 30:
            verdict = "insufficient_data"
        elif tier == "expansion":
            if actual_wr >= expected_wr - 0.05:
                verdict = "promote"
            elif actual_wr < expected_wr - 0.15:
                verdict = "demote"
            else:
                verdict = "monitor"
        elif tier == "boundary_testing":
            if actual_wr >= 0.55:
                verdict = "consider_expand"
            elif actual_wr >= 0.50:
                verdict = "slight_improvement"
            else:
                verdict = "still_broken"
        elif tier == "validation":
            if abs(actual_wr - (avg_wp / 100.0)) <= 0.08:
                verdict = "calibrated"
            elif actual_wr < (avg_wp / 100.0) - 0.15:
                verdict = "overconfident_zone"
            else:
                verdict = "watch"
        else:
            verdict = "unknown"

        entry = {
            "sport":       sport,
            "market_type": mkt,
            "n_legs":      n,
            "actual_wr":   round(actual_wr, 3),
            "expected_wr": expected_wr,
            "avg_model_wp": avg_wp,
            "verdict":     verdict,
        }

        tier_list = results.get(tier)
        if tier_list is not None:
            tier_list.append(entry)
        else:
            results.setdefault(tier, []).append(entry)

    return results


def generate_exploration_bets(
    db: Session,
    run_id: str | None = None,
    game_date: str | None = None,
) -> dict:
    """
    Create adjacent-line single-leg exploration bets for each CUSHION leg in
    today's Section A picks (prospective / prospective_pm / forced_generation /
    top_picks_page).

    Legacy adjacent-line exploration. Tiered exploration runs separately via
    generate_tiered_exploration_bets(). Both are tagged source='exploration'
    but legacy bets have exploration_tier=NULL while tiered bets have a tier tag.

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
        con = sqlite3.connect(_BETS_DB_PATH, timeout=15)
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
                bet_category        = classify_bet_category(leg_mkt),
            ))
            created += 1

    if created > 0:
        with DB_WRITE_LOCK:
            db.commit()
        _batch_spread_enrich_run(run_id, db)
        _batch_pi_enrich_run(run_id, db)
        _batch_mlb_enrich_run(run_id, db)
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


# ─── Player prop settlement (MLB hits/TB, NHL shots on goal) ──────────────────

import unicodedata as _unicodedata
import requests as _requests

_PROP_STAT_CACHE: dict = {}   # (sport, game_date, "away@home") → {last_name: stat_dict}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in _unicodedata.normalize("NFD", s)
        if _unicodedata.category(c) != "Mn"
    ).lower()


_NAME_SUFFIXES = frozenset({"jr.", "sr.", "ii", "iii", "iv", "jr", "sr"})


def _norm_player_name(full_name: str) -> str:
    """Return lowercase last name for player lookup key, stripping Jr./Sr./II suffixes."""
    parts = [p for p in full_name.strip().split() if p.lower() not in _NAME_SUFFIXES]
    return parts[-1].lower() if parts else full_name.strip().lower()


def _mlb_team_match(db_name: str, api_name: str) -> bool:
    """True if db_name and api_name refer to the same MLB team."""
    d = db_name.strip().lower()
    a = api_name.strip().lower()
    if d == a:
        return True
    # Last word of either contained in the other  (e.g. "Athletics" ↔ "Athletics")
    d_last = d.split()[-1] if d.split() else d
    a_last = a.split()[-1] if a.split() else a
    return d_last in a or a_last in d


def _nhl_team_match(db_name: str, abbrev: str, place_name: str) -> bool:
    """True if db_name matches an NHL team identified by its abbreviation or place name."""
    dn = _strip_accents(db_name)
    pn = _strip_accents(place_name)
    # "Montreal Canadiens" → "montreal" appears in "montreal"
    for word in dn.split():
        if len(word) >= 4 and (word in pn or pn.startswith(word)):
            return True
    # Last word of DB name as fallback ("Canadiens" → "canadiens")
    last = dn.split()[-1] if dn.split() else ""
    if last and (last in pn or _strip_accents(abbrev) == last[:3]):
        return True
    return False


def _fetch_mlb_all_batters(game_date: str) -> dict:
    """
    Fetch ALL MLB batter stats for every game on game_date.

    Returns {player_last_name: {"full_name", "hits", "total_bases"}}
    Pools all games so player lookup works regardless of which fixture_id
    was stored on the leg.
    """
    cache_key = ("MLB_ALL", game_date)
    if cache_key in _PROP_STAT_CACHE:
        return _PROP_STAT_CACHE[cache_key]

    try:
        r = _requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": game_date, "hydrate": "team"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("[prop_settler] MLB schedule failed for %s: %s", game_date, exc)
        return {}

    game_pks = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            pk = game.get("gamePk")
            if not pk:
                continue
            # Only include games that have started or finished.
            # Pre-game boxscores list roster players with 0 stats,
            # which is indistinguishable from "player went 0-for-X"
            # and causes premature LOSS settlement.
            status_code = (game.get("status") or {}).get("statusCode", "")
            if status_code in ("S", "P", "PW"):
                # S=Scheduled, P=Pre-game, PW=Postponed/Warmup — skip
                continue
            game_pks.append(pk)

    if not game_pks:
        logger.info("[prop_settler] MLB: no games found for %s", game_date)
        _PROP_STAT_CACHE[cache_key] = {}
        return {}

    result: dict = {}
    for game_pk in game_pks:
        try:
            r = _requests.get(
                f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore",
                timeout=10,
            )
            r.raise_for_status()
            bdata = r.json()
        except Exception as exc:
            logger.warning("[prop_settler] MLB boxscore failed for gamePk %s: %s", game_pk, exc)
            continue
        for side in ("home", "away"):
            team_block = bdata.get("teams", {}).get(side, {})
            players = team_block.get("players", {})
            for pid in team_block.get("batters", []):
                pdata = players.get(f"ID{pid}", {})
                full_name = (pdata.get("person") or {}).get("fullName", "")
                if not full_name:
                    continue
                stats = (pdata.get("stats") or {}).get("batting", {})
                last = _norm_player_name(full_name)
                result[last] = {
                    "full_name":    full_name,
                    "hits":         int(stats.get("hits") or 0),
                    "total_bases":  int(stats.get("totalBases") or 0),
                }

    _PROP_STAT_CACHE[cache_key] = result
    logger.info("[prop_settler] MLB all-batters: %d players across %d games on %s",
                len(result), len(game_pks), game_date)
    return result


def _fetch_mlb_boxscore_stats(home_team: str, away_team: str, game_date: str) -> dict:
    """Wrapper: returns all-batters pool (team args kept for API compatibility)."""
    return _fetch_mlb_all_batters(game_date)


def _fetch_nhl_all_skaters(game_date: str) -> dict:
    """
    Fetch ALL NHL player SOG stats for every completed game on game_date.

    Returns {player_last_name: {"abbrev_name", "sog"}}
    Pools all completed games so player lookup works regardless of which
    fixture_id was stored on the leg.
    """
    cache_key = ("NHL_ALL", game_date)
    if cache_key in _PROP_STAT_CACHE:
        return _PROP_STAT_CACHE[cache_key]

    try:
        r = _requests.get(
            f"https://api-web.nhle.com/v1/schedule/{game_date}",
            timeout=10,
        )
        r.raise_for_status()
        sched = r.json()
    except Exception as exc:
        logger.warning("[prop_settler] NHL schedule failed for %s: %s", game_date, exc)
        return {}

    game_ids = []
    for gw in sched.get("gameWeek", []):
        for game in gw.get("games", []):
            if game.get("gameState") == "OFF":
                gid = game.get("id")
                if gid:
                    game_ids.append(gid)

    if not game_ids:
        logger.info("[prop_settler] NHL: no completed games found for %s", game_date)
        _PROP_STAT_CACHE[cache_key] = {}
        return {}

    result: dict = {}
    for game_id in game_ids:
        try:
            r = _requests.get(
                f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore",
                timeout=10,
            )
            r.raise_for_status()
            bdata = r.json()
        except Exception as exc:
            logger.warning("[prop_settler] NHL boxscore failed for game %s: %s", game_id, exc)
            continue
        pstats = bdata.get("playerByGameStats", {})
        for side in ("homeTeam", "awayTeam"):
            for pos_key in ("forwards", "defense", "goalies"):
                for p in (pstats.get(side) or {}).get(pos_key, []):
                    abbrev = (p.get("name") or {}).get("default", "")
                    if not abbrev:
                        continue
                    parts = abbrev.split()
                    last = parts[-1].lower() if parts else abbrev.lower()
                    result[last] = {
                        "abbrev_name": abbrev,
                        "sog": int(p.get("sog") or 0),
                    }

    _PROP_STAT_CACHE[cache_key] = result
    logger.info("[prop_settler] NHL all-skaters: %d players across %d games on %s",
                len(result), len(game_ids), game_date)
    return result


def _fetch_nhl_boxscore_stats(home_team: str, away_team: str, game_date: str) -> dict:
    """Wrapper: returns all-skaters pool (team args kept for API compatibility)."""
    return _fetch_nhl_all_skaters(game_date)


_PLAYER_PROP_MARKETS = frozenset({
    "player_hits", "player_total_bases", "player_shots_on_goal",
})


def _settle_player_prop_leg(leg, game_date: str) -> dict:
    """
    Settle a player prop exploration leg using free boxscore APIs.

    Supports: player_hits, player_total_bases (MLB Stats API)
              player_shots_on_goal (NHL official API)

    Returns dict with "outcome": "WIN"|"LOSS"|"PUSH"|None
    """
    import re as _re

    EMPTY: dict = {"outcome": None}

    desc   = (leg.description or "").strip()
    market = (leg.market_type or "").lower()
    sport  = (leg.sport or "").upper()

    if market not in _PLAYER_PROP_MARKETS:
        return EMPTY
    if not game_date:
        return EMPTY

    # Parse "<Player Name> Over/Under <line> ..."
    m = _re.match(r"^(.+?)\s+(Over|Under)\s+([\d.]+)", desc, _re.IGNORECASE)
    if not m:
        return EMPTY

    player_name = m.group(1).strip()
    direction   = m.group(2).upper()   # "OVER" | "UNDER"
    line        = float(m.group(3))

    # Fetch all players from all games on game_date (pool approach — independent of
    # which fixture_id was stored on the leg, since props are often mis-assigned).
    if sport == "MLB":
        stats    = _fetch_mlb_all_batters(game_date)
        stat_key = "hits" if market == "player_hits" else "total_bases"
    elif sport == "NHL":
        stats    = _fetch_nhl_all_skaters(game_date)
        stat_key = "sog"
    else:
        return EMPTY

    if not stats:
        return EMPTY

    # Look up player by last name
    last_name    = _norm_player_name(player_name)
    player_stats = stats.get(last_name)

    if player_stats is None:
        # Secondary scan: find any entry whose key is a suffix match
        for k, v in stats.items():
            if k.endswith(last_name):
                player_stats = v
                break

    if player_stats is None:
        # If the game date is already past (boxscore should be final), mark as DNP.
        # This prevents the bet from looping as PENDING for 7 days before expiring.
        _today_ct_str = datetime.now(_CT).strftime("%Y-%m-%d")
        if game_date < _today_ct_str:
            logger.info("[prop_settler] DNP: '%s' not in %s boxscore for %s (game in past). Available: %s",
                        player_name, sport, game_date,
                        sorted(stats.keys())[:12])
            return {"outcome": "DNP", "player_name": player_name}
        logger.info("[prop_settler] Player '%s' not found in %s pool for %s (game may still be live). Available: %s",
                    player_name, sport, game_date,
                    sorted(stats.keys())[:12])
        return EMPTY

    actual = int(player_stats.get(stat_key) or 0)

    if direction == "OVER":
        outcome = "WIN" if actual > line else ("PUSH" if actual == line else "LOSS")
    else:
        outcome = "WIN" if actual < line else ("PUSH" if actual == line else "LOSS")

    logger.info("[prop_settler] %s (%s) %s %s %.1f → actual=%d → %s",
                player_name, sport, stat_key, direction, line, actual, outcome)
    return {
        "outcome":      outcome,
        "player_name":  player_name,
        "actual_stat":  actual,
        "stat_key":     stat_key,
        "line":         line,
        "direction":    direction,
    }


# ─── Settlement ───────────────────────────────────────────────────────────────

def _settle_mock_leg(
    leg,
    scores_by_id: dict,          # scores-API-id → score dict (ESPN numeric IDs)
    all_scores: list,            # fallback: all completed scores
    fix_teams_cache: dict = {},  # OddsAPI fixture_id → (home_team, away_team)
    game_date: str = "",         # CDT date string for player prop settlement
) -> dict:
    """
    Resolve a single MockBetLeg against completed game scores.

    Returns a dict with keys:
        outcome   — 'WIN' | 'LOSS' | 'PUSH' | None (unresolvable)
        home_team, away_team, home_score, away_score
        margin              — team_score - opp_score
        adjusted_margin     — margin + line (spread) or total_diff (total)
        accuracy_delta      — same as adjusted_margin

    Pinning strategy (in order):
      1. scores_by_id[fixture_id]  — exact ESPN-ID match (fastest)
      2. fix_teams_cache[fixture_id] name-match in all_scores — bridges
         OddsAPI hex IDs to ESPN numeric IDs via home/away team names
      3. all_scores team-name search — used when no fixture_id set
    """
    import re as _re

    EMPTY = {"outcome": None, "home_team": None, "away_team": None,
             "home_score": None, "away_score": None,
             "margin": None, "adjusted_margin": None, "accuracy_delta": None}

    # ── Player prop dispatch — delegate before team-score logic ──────────────
    _market_lower = (leg.market_type or "").lower()
    if _market_lower in _PLAYER_PROP_MARKETS and game_date:
        _prop_res = _settle_player_prop_leg(leg, game_date)
        _prop_outcome = _prop_res.get("outcome")
        if _prop_outcome is not None:
            # WIN / LOSS / PUSH / DNP — all propagated as-is
            return {**EMPTY, "outcome": _prop_outcome}
        # Game not yet available or player not found — stay PENDING
        return EMPTY

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
        if not m:
            # Fallback: "Team +3.5" without trailing "(Spread)" label
            m = _re.search(r"([+-]\d+\.?\d*)\s*$", desc)
        if m:
            line = float(m.group(1))
    if is_total and not is_btts:
        m = _re.search(r"(over|under)\s+\+?([\d.]+)", desc_lower)
        if m:
            total_direction = m.group(1)
            line = float(m.group(2))

    # ── Candidate games: fixture_id-pinned or name-based fallback ────────────
    # Priority:
    #   1. scores_by_id[fixture_id] — exact match (ESPN numeric ID)
    #   2. fix_teams_cache pin — OddsAPI hex ID → home/away teams →
    #      first completed game in all_scores where both teams match.
    #      This bridges the OddsAPI←→ESPN ID format mismatch.
    #   3. No fixture_id → search all_scores by team name (unchanged).
    #
    # IMPORTANT: only fall back to all_scores when fixture_id is NOT set.
    # If fixture_id is set but missing from all sources, stay PENDING so we
    # don't accidentally match a different-day game with the same team name.
    from auto_settle import _team_match
    if leg.fixture_id:
        if leg.fixture_id in scores_by_id:
            # Fast path: ESPN ID direct match
            candidates: list[dict] = [scores_by_id[leg.fixture_id]]
        else:
            # Slow path: look up home/away from Fixture table, find in all_scores
            fix_teams = fix_teams_cache.get(leg.fixture_id)
            if fix_teams:
                fix_home, fix_away = fix_teams
                _pinned = next(
                    (g for g in all_scores
                     if g.get("completed")
                     and _team_match(fix_home, g.get("home_team", ""), g.get("away_team", "")) == "home"
                     and _team_match(fix_away, g.get("home_team", ""), g.get("away_team", "")) == "away"),
                    None,
                )
                if _pinned:
                    candidates = [_pinned]
                else:
                    # Game not completed in scores yet — stay PENDING
                    return EMPTY
            else:
                # Fixture not in DB (no teams known) — stay PENDING
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
        avg_lqs = round(min(100.0, sum(l.get("lqs", 65) for l in legs) / len(legs)), 1)
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
            result = _settle_mock_leg(leg_obj, scores_by_id, list(scores_by_id.values()),
                                      game_date=gdate)
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

    with DB_WRITE_LOCK:
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
            bet.avg_lqs = round(min(100.0, sum(new_scores) / len(new_scores)), 1)
            updated += 1

    with DB_WRITE_LOCK:
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
    dnp     = 0

    # ── Pre-build fixture caches ──────────────────────────────────────────────
    # _fix_cache:       fixture_id → naive-UTC commence_time
    # _fix_teams_cache: fixture_id → (home_team, away_team)
    # The team cache lets us pin a score by team name when the fixture_id key
    # format differs from the scores-API ID format (OddsAPI hex ≠ ESPN numeric).
    _fix_cache: dict[str, datetime] = {}
    _fix_teams_cache: dict[str, tuple[str, str]] = {}
    for fix in db.query(Fixture).all():
        if fix.commence_time:
            ct = fix.commence_time
            if ct.tzinfo is not None:
                ct = ct.replace(tzinfo=None)  # strip to naive UTC
            _fix_cache[fix.id] = ct
        if fix.home_team and fix.away_team:
            _fix_teams_cache[fix.id] = (fix.home_team, fix.away_team)

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
                        pool.extend(_fetch_scores_cached(sk, days_from=7))
                    except Exception:
                        pass
            _api_scores_cache[cache_key] = pool
        return _api_scores_cache[cache_key]

    for mock in pending:
        # ── Expire stale bets ────────────────────────────────────────────────
        _gen_at = mock.generated_at
        if _gen_at and _gen_at.tzinfo is not None:
            _gen_at = _gen_at.replace(tzinfo=None)
        if _gen_at and _gen_at < cutoff:
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
                # Player prop legs don't use fixture pinning — skip this gate.
                if (getattr(_leg, "market_type", "") or "").lower() in _PLAYER_PROP_MARKETS:
                    continue
                _ct = _fix_cache.get(_leg.fixture_id)
                if _ct is not None and _ct > now_utc:
                    _any_future_fixture = True
                    break
        if _any_future_fixture:
            skipped += 1
            continue

        # ── Same-CDT-day gate ──────────────────────────────────────────────────
        # MLB/NBA/NHL games starting at 7-10 PM CDT have a UTC commence_time on
        # the NEXT calendar day (e.g. 7 PM CDT = 00:00 UTC+1). A bet generated on
        # 5/16 CDT can therefore reference a fixture whose UTC date is "5/16" but
        # whose CDT date is "5/15" (it's actually yesterday's game). The existing
        # future-gate passes these because the game has already started in UTC.
        # This gate catches that: if ANY leg's fixture started BEFORE the bet's
        # CDT game_date, the bet can't have a valid result yet — skip it.
        # Expiry path: if game_date is in the past AND fixture CDT date < game_date,
        # the bet was generated against the wrong fixture — expire it so it doesn't
        # loop forever.
        if game_date:
            _utc_tz = ZoneInfo("UTC")
            _has_stale_fixture = False
            _today_ct_str = datetime.now(_CT).strftime("%Y-%m-%d")
            for _leg in legs:
                if _leg.fixture_id:
                    # Player prop legs don't use fixture pinning — their fixture_id
                    # often references a different game than the prop itself.
                    # Skip the stale-fixture check; settlement uses resolved_teams + game_date.
                    if (getattr(_leg, "market_type", "") or "").lower() in _PLAYER_PROP_MARKETS:
                        continue
                    _ct_utc = _fix_cache.get(_leg.fixture_id)
                    if _ct_utc is not None:
                        # Convert naive UTC → aware UTC → CDT date string
                        _aware_utc = _ct_utc.replace(tzinfo=_utc_tz)
                        _fix_date_ct = _aware_utc.astimezone(_CT).strftime("%Y-%m-%d")
                        if _fix_date_ct < game_date:
                            _has_stale_fixture = True
                            break
            if _has_stale_fixture:
                # If the bet's game_date is in the past, the game already played
                # on the earlier fixture date — bet is unresolvable, expire it.
                if game_date < _today_ct_str:
                    mock.status        = "EXPIRED"
                    mock.settled_at    = datetime.utcnow()
                    mock.actual_profit = 0.0
                    expired += 1
                else:
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

        # ── Filter api_scores to game_date (CDT) to prevent cross-day contamination ──
        # OddsAPI returns multiple days of scores in one call. Without a date gate the
        # fix_teams_cache pin path finds yesterday's COMPLETED game for the same teams
        # and settles today's bet with yesterday's score.
        # hist_scores are already date-filtered via _get_historical_games_for_date.
        def _api_scores_for_date(scores: list, target_date: str) -> list:
            out = []
            for g in scores:
                ct_str = g.get("commence_time") or ""
                if not ct_str:
                    out.append(g)   # no date info — keep
                    continue
                try:
                    # Parse ISO datetime (may or may not have trailing Z / timezone)
                    _dt_str = ct_str.replace("Z", "").replace("+00:00", "")
                    _dt_utc = datetime.fromisoformat(_dt_str)
                    _dt_ct  = _dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(_CT)
                    if _dt_ct.strftime("%Y-%m-%d") == target_date:
                        out.append(g)
                except Exception:
                    out.append(g)   # keep on parse error
            return out

        _filtered_api = _api_scores_for_date(api_scores, game_date) if game_date else api_scores

        # Also guard hist_scores: only include games that have actually completed
        # (status != "Scheduled") to prevent 0-0 scheduled placeholders being
        # treated as final scores.
        _filtered_hist = [
            h for h in hist_scores
            if any(
                int(s.get("score", 0) or 0) > 0
                for s in h.get("scores", [])
            )
        ]

        all_scores: list[dict] = _filtered_hist + _filtered_api
        # Build id→game map for fixture_id pinning (only OddsAPI games have "id")
        scores_by_id: dict[str, dict] = {
            g["id"]: g for g in _filtered_api if g.get("id")
        }

        if not all_scores:
            skipped += 1
            continue

        # Resolve every leg; collect outcomes + audit data
        leg_resolutions: list[dict] = [
            _settle_mock_leg(leg, scores_by_id, all_scores, _fix_teams_cache,
                             game_date=game_date or "") for leg in legs
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

            # CLV: closing-line proxy = latest alt_lines snapshot before
            # game start (afternoon fetch is the proxy for the true close).
            # Falls back to line_snapshots if alt_lines has no data.
            if leg.fixture_id and leg.open_odds is not None:
                _commence = res.get("commence_time") or (
                    mock.game_date + "T23:59:59Z" if mock.game_date else None
                )
                # Try alt_lines first (populated by daily morning + PM fetch)
                _close = _lookup_alt_lines_odds(
                    leg.fixture_id,
                    leg.description or "",
                    leg.market_type or "",
                    _HIST_DB_PATH,
                    before_dt=_commence,
                    earliest=False,
                )
                # Fall back to line_snapshots (older data, pre-May 2026)
                if _close is None:
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

        # ── DNP: player Did Not Play — void the bet (stake returned, not a loss) ──
        # Applies when all unresolved legs are player props where the player
        # wasn't in any boxscore for game_date (confirmed DNP, not just pending).
        if any(o == "DNP" for o in leg_outcomes):
            mock.status        = "DNP"
            mock.actual_profit = 0.0
            mock.settled_at    = datetime.utcnow()
            dnp     += 1
            settled += 1
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

    with DB_WRITE_LOCK:
        db.commit()

    # Change 1: refresh calibration + Change 4: update pattern signals + Change 2: auto-evaluate markets
    cal_result: dict = {}
    if settled > 0:
        try:
            # Collect IDs of bets just settled in this batch
            _newly_settled_ids = [
                b.id for b in pending
                if b.status in ("SETTLED_WIN", "SETTLED_LOSS") and b.settled_at is not None
            ]
            cal_result = _refresh_calibration_after_settlement(db, settled)
            update_pattern_signals(db, _newly_settled_ids)
            auto_evaluate_markets(db)
        except Exception as _fe:
            cal_result = {"error": str(_fe)}

    # Post-settlement validation (runs after each batch that settled ≥1 bet)
    validation_result: dict = {}
    if settled > 0:
        try:
            import validation_tracker as _vt
            validation_result = _vt.run_all_validations()
        except Exception as _vfe:
            validation_result = {"error": str(_vfe)}

    # ── Pi concentration tracking (post-settlement) ─────────────────────
    # Populates pi_concentration_flags for every settled leg that appeared
    # in 3+ bets for its fixture.  NOT wired to live generation — analysis only.
    _pi_flags_inserted = 0
    if settled > 0:
        try:
            _pi_flags_inserted = _populate_pi_concentration_flags(db)
        except Exception as _pcfe:
            print(f"[settle] pi_concentration_flags error (non-fatal): {_pcfe}")

    # ── ATS gap-sizing asymmetry monitor (post-settlement) ────────────
    _atscap_report = None
    if settled > 0:
        try:
            from negbin_shadow import _report_atscap_asymmetry
            _report_atscap_asymmetry(db)
        except Exception as _ace:
            print(f"[settle] atscap asymmetry report error (non-fatal): {_ace}")

    return {
        "settled": settled,
        "expired": expired,
        "skipped": skipped,
        "dnp":     dnp,
        "calibration_refreshed": settled > 0,
        "calibration_result": cal_result,
        "validation": validation_result,
        "pi_flags_inserted": _pi_flags_inserted,
    }


# ─── Pi concentration tracking ────────────────────────────────────────────────

def _populate_pi_concentration_flags(db: Session) -> int:
    """
    Populate pi_concentration_flags for settled legs not yet tracked.

    For each settled leg (from production sources), count how many bets
    contain the same (fixture_id, description) on that game_date, then
    flag whether Pi filtering would have removed it.

    Returns number of rows inserted.
    """
    from sqlalchemy import text as _pct
    from database import PiConcentrationFlag

    # Find settled legs not yet in the tracking table
    rows = db.execute(_pct("""
        SELECT mbl.description, mbl.fixture_id, mbl.pi_diff_pick, mbl.leg_result,
               mb.game_date, mb.settled_at,
               COUNT(DISTINCT mb.id) as n_bets,
               mbl.pi_diff
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND mb.source IN ('prospective','prospective_pm','prospective_legacy','top_picks_page')
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND NOT EXISTS (
              SELECT 1 FROM pi_concentration_flags pcf
              WHERE pcf.game_date = mb.game_date
                AND pcf.fixture_id = mbl.fixture_id
                AND pcf.bet_info = mbl.description
          )
        GROUP BY mbl.description, mbl.fixture_id, mb.game_date
    """)).fetchall()

    inserted = 0
    for desc, fid, pi_diff_pick, leg_result, gdate, settled_at, n_bets, pi_diff_raw in rows:
        # Use pick-relative pi_diff for filtering decisions
        _pd = pi_diff_pick if pi_diff_pick is not None else pi_diff_raw
        if _pd is not None and _pd < -1.0:
            flag = "pi_disqualify"
        elif _pd is not None and _pd < 0 and n_bets >= 3:
            flag = "pi_cap_2"
        else:
            flag = "none"

        # Parse settled_at to datetime if it comes back as string
        _sa = settled_at
        if isinstance(_sa, str):
            try:
                _sa = datetime.fromisoformat(_sa)
            except (ValueError, TypeError):
                _sa = None

        db.add(PiConcentrationFlag(
            game_date=gdate,
            fixture_id=fid,
            bet_info=desc,
            pi_diff=_pd,
            n_bets_containing=n_bets,
            leg_result=leg_result,
            would_have_filtered=flag,
            settled_at=_sa,
        ))
        inserted += 1

    if inserted:
        with DB_WRITE_LOCK:
            db.commit()
        print(f"[settle] pi_concentration_flags: {inserted} new rows")
    return inserted


# ─── Manual settlement ────────────────────────────────────────────────────────

def manual_settle_bet(
    db: Session,
    bet_id: str,
    home_score: int,
    away_score: int,
) -> dict:
    """
    Settle a single PENDING MockBet by injecting the home/away score directly.

    Useful when the automatic score fetcher can't reach the source (e.g. Ligue 1
    post-season barrage games that ESPN/OddsAPI don't expose).

    Resolves every leg against the injected score using the same _settle_mock_leg
    logic, then sets status/profit identically to the automated path.
    """
    mock = db.query(MockBet).filter(MockBet.id == bet_id).first()
    if not mock:
        return {"error": f"Bet {bet_id} not found"}
    if mock.status != "PENDING":
        return {"error": f"Bet {bet_id} is already {mock.status}"}

    legs = (
        db.query(MockBetLeg)
        .filter(MockBetLeg.mock_bet_id == bet_id)
        .order_by(MockBetLeg.leg_index)
        .all()
    )
    if not legs:
        return {"error": "No legs found for this bet"}

    # Build a synthetic scores entry that matches OddsAPI format.
    # For parlays, the injected score may only apply to one of the legs.
    # We build one synthetic entry per distinct fixture_id, using the correct
    # home/away teams from the fixture cache.
    from database import Fixture
    _fix_teams_cache: dict[str, tuple[str, str]] = {}
    for fix in db.query(Fixture).all():
        if fix.home_team and fix.away_team:
            _fix_teams_cache[fix.id] = (fix.home_team, fix.away_team)

    # Collect all distinct fixture_ids across legs so we can build synthetic
    # scores for every leg, not just leg0.  For legs whose fixture resolves via
    # the normal scores path (e.g. NHL in a mixed parlay) those will be handled
    # by all_scores; for the leg we're manually settling, we inject it here.
    all_fix_ids = list({leg.fixture_id for leg in legs if leg.fixture_id})

    synthetic_scores: list[dict] = []
    scores_by_id: dict[str, dict] = {}

    for fid in all_fix_ids:
        fix_teams = _fix_teams_cache.get(fid)
        if fix_teams:
            h, a = fix_teams
        else:
            h, a = "Home", "Away"

        # Only inject the manual score for fixtures that match the provided
        # home/away teams (case-insensitive partial match).
        # For other fixtures (different sport legs), leave the score out so
        # normal settlement logic can resolve them.
        synthetic = {
            "id":            fid,
            "completed":     True,
            "home_team":     h,
            "away_team":     a,
            "scores": [
                {"name": h, "score": str(home_score)},
                {"name": a, "score": str(away_score)},
            ],
            "commence_time": (mock.game_date + "T20:00:00Z") if mock.game_date else "",
            "source":        "manual",
        }
        synthetic_scores.append(synthetic)
        scores_by_id[fid] = synthetic

    # Use first fixture's teams for display
    if all_fix_ids:
        first_fid = all_fix_ids[0]
        h_name, a_name = _fix_teams_cache.get(first_fid, ("Home", "Away"))
    else:
        h_name, a_name = "Home", "Away"

    # Also pull live scores for non-manual legs (e.g. NHL leg in a mixed parlay)
    from auto_settle import _fetch_scores_cached, SPORT_TO_KEYS
    sport_str = mock.sport or ""
    live_pool: list[dict] = []
    seen_live: set[str] = set()
    for part in sport_str.split("|"):
        sk_label = part.strip()
        for sk in SPORT_TO_KEYS.get(sk_label, []):
            try:
                for g in _fetch_scores_cached(sk, days_from=7):
                    gid = g.get("id", "")
                    if gid and gid not in seen_live:
                        live_pool.append(g)
                        seen_live.add(gid)
            except Exception:
                pass

    all_scores = synthetic_scores + live_pool
    game_date  = mock.game_date or ""

    leg_resolutions = [
        _settle_mock_leg(leg, scores_by_id, all_scores, _fix_teams_cache,
                         game_date=game_date) for leg in legs
    ]
    leg_outcomes = [r["outcome"] for r in leg_resolutions]

    # Write audit fields
    for leg, res in zip(legs, leg_resolutions):
        leg.leg_result          = res.get("outcome")
        leg.resolved_home_team  = res.get("home_team") or h_name
        leg.resolved_away_team  = res.get("away_team") or a_name
        leg.resolved_home_score = res.get("home_score") if res.get("home_score") is not None else home_score
        leg.resolved_away_score = res.get("away_score") if res.get("away_score") is not None else away_score

    stake = mock.amount or 0.0

    if any(o == "LOSS" for o in leg_outcomes):
        mock.status        = "SETTLED_LOSS"
        mock.actual_profit = round(-stake, 2)
    elif any(o is None for o in leg_outcomes):
        return {"error": f"Could not resolve all legs. Outcomes: {leg_outcomes}. "
                         f"Check that home/away teams match the leg descriptions."}
    elif all(o == "WIN" for o in leg_outcomes):
        mock.status        = "SETTLED_WIN"
        mock.actual_profit = round(stake * ((mock.odds or 2.0) - 1), 2)
    else:
        mock.status        = "SETTLED_LOSS"
        mock.actual_profit = 0.0

    mock.settled_at = datetime.utcnow()
    with DB_WRITE_LOCK:
        db.commit()

    return {
        "settled": 1,
        "bet_id":  bet_id,
        "status":  mock.status,
        "profit":  mock.actual_profit,
        "score":   f"{h_name} {home_score} – {a_name} {away_score}",
        "leg_outcomes": leg_outcomes,
    }


# ─── Per-leg accuracy breakdown ───────────────────────────────────────────────

# Epoch boundaries corresponding to major architecture changes.
# Each label covers [start, end) in game_date CDT strings.
_LEG_ACCURACY_EPOCHS: list[tuple[str, str, str]] = [
    ("pre-5/18",  "0000-00-00", "2026-05-17"),
    ("5/18–5/27", "2026-05-18", "2026-05-27"),
    ("5/28+",     "2026-05-28", "9999-99-99"),
]


def get_leg_accuracy(
    db: Session,
    days: int = 90,
    source: str = "simulation",
    game_date: str = None,
) -> dict:
    """
    Return per-leg accuracy stats for mock bets from `source`.

    Groups by sport × market_type, with win-rate broken out per epoch
    (pre-5/18, 5/18–5/27, 5/28+) to track improvement over time.

    game_date: optional exact date filter ("today", "yesterday", or "YYYY-MM-DD").
               When set, overrides the `days` window.

    Returns:
        {
          "source": str,
          "total_legs": int,
          "overall_win_rate": float,
          "epochs": [{"label", "start", "end", "legs", "win_rate"}],
          "by_sport_market": [
              {"sport", "market_type", "total", "wins", "losses",
               "win_rate", "by_epoch": {"label": {"legs","win_rate"}}}
          ]
        }
    """
    _today_ct = datetime.now(_CT).strftime("%Y-%m-%d")
    _yest_ct  = (datetime.now(_CT) - timedelta(days=1)).strftime("%Y-%m-%d")

    if game_date in ("today", "yesterday"):
        exact_date = _today_ct if game_date == "today" else _yest_ct
        since = exact_date
        until = exact_date
    elif game_date:
        since = game_date
        until = game_date
    else:
        since = (datetime.now(_CT) - timedelta(days=days)).strftime("%Y-%m-%d")
        until = None

    # "simulation" is an alias for the full set of sim sources.
    # The source name changed over time (prospective → simulation on 5/25),
    # so we always query all sim sources together.
    if source == "simulation":
        src_list = _LQ_SIM_SOURCES
    elif source == "exploration":
        src_list = EXPLORATION_SOURCES
    else:
        src_list = (source,)
    _src_in_la = ",".join(f"'{s}'" for s in src_list)

    from sqlalchemy import text as _text_la
    until_clause_la = "AND mb.game_date <= :until" if until else ""
    until_params_la: dict = {"until": until} if until else {}
    rows = db.execute(
        _text_la(f"""
            SELECT mbl.sport, mbl.market_type, mbl.leg_result, mb.game_date
            FROM   mock_bet_legs  mbl
            JOIN   mock_bets      mb  ON mb.id = mbl.mock_bet_id
            WHERE  mb.source      IN ({_src_in_la})
              AND  mbl.leg_result IN ('WIN','LOSS')
              AND  COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND  mb.game_date  >= :since
              {until_clause_la}
            ORDER BY mb.game_date
        """),
        {"since": since, **until_params_la},
    ).fetchall()

    # Aggregate by (sport, market_type, epoch)
    from collections import defaultdict
    Cell = lambda: {"wins": 0, "losses": 0}
    by_sm_epoch: dict[tuple, dict] = defaultdict(lambda: defaultdict(Cell))
    epoch_totals: dict[str, dict] = {lbl: Cell() for lbl, _, _ in _LEG_ACCURACY_EPOCHS}
    grand = Cell()

    for r in rows:
        sport   = r[0] or "Unknown"
        market  = r[1] or "unknown"
        result  = r[2]
        gd      = r[3] or ""
        epoch_lbl = next(
            (lbl for lbl, start, end in _LEG_ACCURACY_EPOCHS if start <= gd <= end),
            "other",
        )
        cell = by_sm_epoch[(sport, market)][epoch_lbl]
        etot = epoch_totals.get(epoch_lbl)
        if result == "WIN":
            cell["wins"] += 1
            if etot: etot["wins"] += 1
            grand["wins"] += 1
        else:
            cell["losses"] += 1
            if etot: etot["losses"] += 1
            grand["losses"] += 1

    def _wr(wins, losses):
        total = wins + losses
        return round(wins / total * 100, 1) if total else None

    total_legs = grand["wins"] + grand["losses"]

    # Per-epoch summary
    epochs_out = []
    for lbl, start, end in _LEG_ACCURACY_EPOCHS:
        et = epoch_totals[lbl]
        t  = et["wins"] + et["losses"]
        epochs_out.append({
            "label":    lbl,
            "start":    start if start != "0000-00-00" else None,
            "end":      end   if end   != "9999-99-99" else None,
            "legs":     t,
            "win_rate": _wr(et["wins"], et["losses"]),
        })

    # Per-sport-market breakdown
    rows_out = []
    for (sport, market), epoch_cells in sorted(by_sm_epoch.items()):
        wins   = sum(c["wins"]   for c in epoch_cells.values())
        losses = sum(c["losses"] for c in epoch_cells.values())
        by_ep  = {}
        for lbl, _, _ in _LEG_ACCURACY_EPOCHS:
            c = epoch_cells.get(lbl, {"wins": 0, "losses": 0})
            t = c["wins"] + c["losses"]
            by_ep[lbl] = {"legs": t, "win_rate": _wr(c["wins"], c["losses"])}
        rows_out.append({
            "sport":       sport,
            "market_type": market,
            "wins":        wins,
            "losses":      losses,
            "total":       wins + losses,
            "win_rate":    _wr(wins, losses),
            "by_epoch":    by_ep,
        })
    # Sort by total legs descending
    rows_out.sort(key=lambda x: -x["total"])

    return {
        "source":           source,
        "days":             days,
        "total_legs":       total_legs,
        "overall_win_rate": _wr(grand["wins"], grand["losses"]),
        "epochs":           epochs_out,
        "by_sport_market":  rows_out,
    }


# ─── Live bet recommendation layer ────────────────────────────────────────────

_PROD_SOURCES = ("prospective", "prospective_pm", "prospective_legacy", "top_picks_page")
_WEAK_SOCCER_LEAGUES = {"Bundesliga", "Ligue 1"}
_MAX_LIVE_PICKS = 5
_LIVE_FIXTURE_CAP = 2
_KELLY_SHADOW_BANKROLL = 500.0  # reference bankroll for shadow Kelly sizing

# Leg-count threshold multipliers: parlays have inherently lower composite
# (multiplicative WP) so the top-25% cutoff is scaled by leg count.
_LIVE_THRESHOLD_MULT = {1: 1.00, 2: 0.70, 3: 0.55}


def _compute_14d_cover_rates(db: Session) -> dict[tuple[str, str], float]:
    """
    Compute 14-day cover rate for each (team-side description, market_type).

    Returns dict mapping description → win rate (0-100).
    Uses only production settled legs.
    """
    from sqlalchemy import text as _t
    rows = db.execute(_t("""
        SELECT mbl.description,
               SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END) as wins,
               COUNT(*) as total
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mb.source IN ('prospective','prospective_pm','prospective_legacy','top_picks_page')
          AND mb.status LIKE 'SETTLED%'
          AND mbl.leg_result IN ('WIN','LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.game_date >= date(:today, '-14 days')
        GROUP BY mbl.description
        HAVING total >= 3
    """), {"today": datetime.now(_CT).strftime("%Y-%m-%d")}).fetchall()
    return {r[0]: round(100.0 * r[1] / r[2], 1) for r in rows}


def apply_live_recommendation(db: Session, game_date: str | None = None) -> dict:
    """
    Apply live-bet criteria to today's production picks and flag qualifiers.

    Criteria (ALL must pass for a pick):
      1. Pi-adjusted composite >= leg-count-scaled top-25% threshold
         - Pi penalty: max(0, (1.0 - worst_pi_diff) * 0.03) applied to composite
         - Threshold scaled by leg count: 1L=100%, 2L=70%, 3L=55%, 4L+=45%
      2. bet_tier = 'T1'
      3. ALL legs have scout_grade IN ('A','B') OR scout_grade IS NULL
      4. No leg has line_direction = 'adverse'
      5. No leg has injury_direction = 'adverse'
      6. 14d cover rate >= 60% on each leg's description (if data exists)
      7. Max 2 fixtures in the live set

    Returns slate confidence, qualifying picks, and red flags.
    """
    from sqlalchemy import text as _t

    today = game_date or datetime.now(_CT).strftime("%Y-%m-%d")

    # Reset today's live flags
    db.execute(_t(
        "UPDATE mock_bets SET live_recommendation = 0 "
        "WHERE game_date = :gd AND live_recommendation = 1"
    ), {"gd": today})
    db.flush()

    # Load all production picks for today
    picks = db.execute(_t("""
        SELECT mb.id, mb.composite_score, mb.bet_tier, mb.bet_info, mb.odds,
               mb.predicted_win_prob, mb.status, mb.actual_profit, mb.source
        FROM mock_bets mb
        WHERE mb.game_date = :gd
          AND mb.source IN ('prospective','prospective_pm','top_picks_page')
          AND mb.composite_score IS NOT NULL
        ORDER BY mb.composite_score DESC
    """), {"gd": today}).fetchall()

    if not picks:
        return {
            "slate_confidence": "PASS",
            "recommended_bet_count": "No bets today",
            "qualifying_picks": [],
            "slate_summary": {
                "total_production_picks": 0,
                "passing_live_filter": 0,
                "avg_composite_qualifying": None,
                "avg_pi_qualifying": None,
                "flags": ["No production picks for this date"],
            },
        }

    # Composite top-25% threshold
    scores = sorted([p[1] for p in picks], reverse=True)
    top25_idx = max(0, len(scores) // 4 - 1)
    comp_threshold = scores[top25_idx] if top25_idx < len(scores) else scores[0]

    # Load legs for all picks
    leg_data: dict[str, list[dict]] = {}
    leg_rows = db.execute(_t("""
        SELECT mbl.mock_bet_id, mbl.fixture_id, mbl.description,
               mbl.pi_diff, mbl.scout_grade, mbl.line_direction,
               mbl.injury_direction, mbl.sport, mbl.leg_result,
               mbl.pi_diff_pick, mbl.win_prob
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mb.game_date = :gd
          AND mb.source IN ('prospective','prospective_pm','top_picks_page')
    """), {"gd": today}).fetchall()
    for r in leg_rows:
        leg_data.setdefault(r[0], []).append({
            "fixture_id": r[1], "description": r[2], "pi_diff": r[3],
            "scout_grade": r[4], "line_direction": r[5],
            "injury_direction": r[6], "sport": r[7], "leg_result": r[8],
            "pi_diff_pick": r[9], "win_prob": r[10],
        })

    # 14d cover rates
    cover_rates = _compute_14d_cover_rates(db)

    # Red flags collection
    flags: list[str] = []
    all_pi_diffs = [l["pi_diff_pick"] for legs in leg_data.values() for l in legs if l.get("pi_diff_pick") is not None]
    bad_pi_count = sum(1 for p in all_pi_diffs if p < -0.5)
    if bad_pi_count:
        flags.append(f"{bad_pi_count} legs have pi_diff_pick < -0.5 -- weak picked-side exposure")

    # Check for weak soccer leagues
    weak_soccer = sum(1 for legs in leg_data.values() for l in legs if l.get("sport") in _WEAK_SOCCER_LEAGUES)
    if weak_soccer:
        flags.append(f"Bundesliga/Ligue1 legs present ({weak_soccer}) -- historically poor")

    # Scout A coverage
    all_scouts = [l["scout_grade"] for legs in leg_data.values() for l in legs if l["scout_grade"]]
    if not any(g == "A" for g in all_scouts):
        flags.append("No Scout A grades today -- lower signal quality")

    # Low cover rate warning
    low_cover = sum(1 for desc, cr in cover_rates.items() if cr < 50)
    if low_cover:
        flags.append(f"{low_cover} leg descriptions have 14d cover rate < 50% -- unproven markets")

    # Apply criteria
    qualifying: list[dict] = []
    fixture_exposure: dict[str, int] = {}

    for pick in picks:
        pid, comp, tier, bet_info, odds, wp, status, profit, source = pick
        legs = leg_data.get(pid, [])
        if not legs:
            continue

        # 1. Pi-adjusted composite vs leg-count-scaled threshold
        n_legs = len(legs)
        pi_with_values = [l["pi_diff_pick"] for l in legs if l.get("pi_diff_pick") is not None]
        worst_pi = min(pi_with_values) if pi_with_values else None
        pi_penalty = max(0.0, (1.0 - worst_pi) * 0.03) if worst_pi is not None else 0.0
        adj_comp = comp * (1.0 - pi_penalty)
        leg_mult = _LIVE_THRESHOLD_MULT.get(n_legs, 0.45)
        adj_threshold = comp_threshold * leg_mult
        if adj_comp < adj_threshold:
            continue
        # 2. T1 only
        if tier != "T1":
            continue
        # 3. Scout grade A/B or NULL
        if any(l["scout_grade"] and l["scout_grade"] not in ("A", "B") for l in legs):
            continue
        # 4. No adverse line direction
        if any(l["line_direction"] == "adverse" for l in legs):
            continue
        # 5. No adverse injury direction
        if any(l["injury_direction"] == "adverse" for l in legs):
            continue
        # 6. 14d cover rate >= 60%
        cover_fail = False
        leg_covers = []
        for l in legs:
            cr = cover_rates.get(l["description"])
            if cr is not None:
                if cr < 60.0:
                    cover_fail = True
                    break
                leg_covers.append(cr)
            else:
                leg_covers.append(None)
        if cover_fail:
            continue
        # 7. Fixture concentration cap (max 2 in live set)
        fids = {l["fixture_id"] for l in legs if l["fixture_id"]}
        if any(fixture_exposure.get(f, 0) >= _LIVE_FIXTURE_CAP for f in fids):
            continue
        # 8. 60% parlay WR floor: every leg must clear LEG_FLOOR[n_legs]
        _floor = _LEG_FLOOR.get(n_legs, _LEG_FLOOR.get(6, 92.0))
        if any((l.get("win_prob") or 0) < _floor for l in legs):
            continue

        # Accept
        for f in fids:
            fixture_exposure[f] = fixture_exposure.get(f, 0) + 1

        pi_vals = pi_with_values
        scout_grades = [l["scout_grade"] or "?" for l in legs]
        cover_display = [cr if cr is not None else "N/A" for cr in leg_covers]

        # Shadow Kelly computation
        _k_edge = None
        _k_frac = 0.0
        _k_bet  = 5.0
        if odds and odds > 1.0 and wp:
            _p = wp / 100.0
            _b = odds - 1.0
            _k_edge = round((_p - 1.0 / odds) * 100, 2)
            _raw_kf = (_b * _p - (1.0 - _p)) / _b
            _k_frac = round(max(_raw_kf, 0.0), 6)
            _qk = _k_frac * 0.25
            _k_bet = round(max(_KELLY_SHADOW_BANKROLL * _qk, 5.0), 2)
            _k_bet = min(_k_bet, round(_KELLY_SHADOW_BANKROLL * 0.20, 2))

        qualifying.append({
            "id": pid,
            "legs": bet_info,
            "composite": round(comp, 4),
            "pi_adjusted_composite": round(adj_comp, 4),
            "pi_penalty": round(pi_penalty, 4),
            "threshold_used": round(adj_threshold, 4),
            "n_legs": n_legs,
            "avg_pi": round(sum(pi_vals) / len(pi_vals), 2) if pi_vals else None,
            "scout_grades": scout_grades,
            "cover_rates": cover_display,
            "signals_passing": 7,
            "combined_odds": _format_american_odds(odds) if odds else None,
            "status": status,
            "actual_profit": profit,
            "why": _build_live_why(legs, pi_vals, scout_grades, cover_display),
            "kelly_edge": _k_edge,
            "kelly_fraction": _k_frac,
            "kelly_bet_size": _k_bet,
        })

        if len(qualifying) >= _MAX_LIVE_PICKS:
            break

    # Flag qualifying picks in DB with shadow Kelly values
    for q in qualifying:
        db.execute(_t(
            "UPDATE mock_bets SET live_recommendation = 1, "
            "kelly_fraction = :kf, kelly_bet_size = :kb, kelly_edge = :ke "
            "WHERE id = :id"
        ), {
            "id": q["id"],
            "kf": q.get("kelly_fraction"),
            "kb": q.get("kelly_bet_size"),
            "ke": q.get("kelly_edge"),
        })
    with DB_WRITE_LOCK:
        db.commit()

    # Fixture concentration flag
    max_fixture_bets = max(fixture_exposure.values()) if fixture_exposure else 0
    total_fids = len({l["fixture_id"] for legs in leg_data.values() for l in legs if l["fixture_id"]})
    if total_fids and max_fixture_bets > 0:
        top_pct = max_fixture_bets / max(len(qualifying), 1) * 100
        if top_pct > 60:
            flags.append(f"{top_pct:.0f}% of live picks share the same fixture -- concentration risk")

    # Slate confidence
    n_qual = len(qualifying)
    avg_comp = round(sum(q["composite"] for q in qualifying) / n_qual, 4) if n_qual else None
    pi_all = [q["avg_pi"] for q in qualifying if q["avg_pi"] is not None]
    avg_pi = round(sum(pi_all) / len(pi_all), 2) if pi_all else None

    if n_qual >= 4 and avg_pi is not None and avg_pi > 0.3:
        slate_conf = "STRONG"
        rec_count = "3-4 bets"
    elif n_qual >= 2:
        slate_conf = "MODERATE"
        rec_count = "1-2 bets"
    elif n_qual == 1:
        slate_conf = "WEAK"
        rec_count = "1 bet max or skip"
    else:
        slate_conf = "PASS"
        rec_count = "No bets today"

    # ── Near-misses: top 5 picks by composite with per-criteria diagnostics ──
    near_misses: list[dict] = []
    if n_qual == 0:
        for pick in picks[:5]:
            pid, comp, tier, bet_info, odds, wp, status, profit, source = pick
            legs = leg_data.get(pid, [])
            if not legs:
                continue

            checks: list[dict] = []
            # 1. Composite threshold
            checks.append({
                "criterion": "Composite",
                "passed": comp >= comp_threshold,
                "detail": f"{comp:.3f} (threshold {comp_threshold:.3f})",
            })
            # 2. T1 tier
            checks.append({
                "criterion": "T1 tier",
                "passed": tier == "T1",
                "detail": tier or "unclassified",
            })
            # 3. Pi diff (pick-relative) >= 0
            worst_pi = None
            worst_pi_desc = None
            pi_pass = True
            for l in legs:
                _pdp = l.get("pi_diff_pick")
                if _pdp is not None and _pdp < 0:
                    pi_pass = False
                    if worst_pi is None or _pdp < worst_pi:
                        worst_pi = _pdp
                        worst_pi_desc = l["description"][:40]
            checks.append({
                "criterion": "Pi diff (pick-relative)",
                "passed": pi_pass,
                "detail": f"worst={worst_pi:+.2f} on {worst_pi_desc}" if worst_pi is not None else "all >= 0 or N/A",
            })
            # 4. Scout A/B or NULL
            scout_pass = not any(l["scout_grade"] and l["scout_grade"] not in ("A", "B") for l in legs)
            bad_scouts = [l["scout_grade"] for l in legs if l["scout_grade"] and l["scout_grade"] not in ("A", "B")]
            checks.append({
                "criterion": "Scout grade",
                "passed": scout_pass,
                "detail": "no grades" if not any(l["scout_grade"] for l in legs) else (
                    f"has {','.join(bad_scouts)}" if bad_scouts else "A/B only"
                ),
            })
            # 5. No adverse line movement
            adv_line = any(l["line_direction"] == "adverse" for l in legs)
            checks.append({
                "criterion": "Line direction",
                "passed": not adv_line,
                "detail": "adverse movement detected" if adv_line else "OK",
            })
            # 6. No adverse injury
            adv_inj = any(l["injury_direction"] == "adverse" for l in legs)
            checks.append({
                "criterion": "Injury direction",
                "passed": not adv_inj,
                "detail": "adverse injury impact" if adv_inj else "OK",
            })
            # 7. Cover rate >= 60%
            cr_pass = True
            cr_detail = "OK"
            for l in legs:
                cr = cover_rates.get(l["description"])
                if cr is not None and cr < 60.0:
                    cr_pass = False
                    cr_detail = f"{cr:.0f}% on {l['description'][:35]}"
                    break
            if cr_pass and not any(cover_rates.get(l["description"]) for l in legs):
                cr_detail = "no data"
            checks.append({
                "criterion": "14d cover rate",
                "passed": cr_pass,
                "detail": cr_detail,
            })

            n_passed = sum(1 for c in checks if c["passed"])
            near_misses.append({
                "legs": bet_info,
                "composite": round(comp, 4),
                "combined_odds": _format_american_odds(odds) if odds else None,
                "model_wp": round(wp, 1) if wp else None,
                "checks": checks,
                "passed_count": n_passed,
                "total_checks": len(checks),
                "gap": f"{len(checks) - n_passed} of {len(checks)} criteria failed",
            })

    return {
        "slate_confidence": slate_conf,
        "recommended_bet_count": rec_count,
        "qualifying_picks": qualifying,
        "near_misses": near_misses,
        "slate_summary": {
            "total_production_picks": len(picks),
            "passing_live_filter": n_qual,
            "avg_composite_qualifying": avg_comp,
            "avg_pi_qualifying": avg_pi,
            "composite_threshold": round(comp_threshold, 4),
            "flags": flags,
        },
    }


def _format_american_odds(decimal_odds: float) -> str:
    if decimal_odds is None or decimal_odds <= 0:
        return "?"
    if decimal_odds >= 2.0:
        return f"+{int(round((decimal_odds - 1) * 100))}"
    return f"-{int(round(100 / (decimal_odds - 1)))}"


def _build_live_why(legs, pi_vals, scout_grades, cover_rates):
    parts = []
    if pi_vals and all(p >= 0 for p in pi_vals):
        parts.append("All legs positive Pi")
    sg_str = "/".join(g for g in scout_grades if g != "?")
    if sg_str:
        parts.append(f"Scout {sg_str}")
    num_covers = [c for c in cover_rates if isinstance(c, (int, float))]
    if num_covers:
        parts.append(f"{sum(num_covers)/len(num_covers):.0f}% avg cover rate")
    return ", ".join(parts) if parts else "Passes all live criteria"


def backtest_live_filter(db: Session, days: int = 10) -> list[dict]:
    """
    Backtest: apply live criteria to each of the last N days.

    Returns per-day results comparing live-qualifying picks vs full slate.
    """
    from sqlalchemy import text as _t

    today = datetime.now(_CT).strftime("%Y-%m-%d")
    dates = db.execute(_t("""
        SELECT DISTINCT game_date FROM mock_bets
        WHERE game_date >= date(:today, :offset)
          AND game_date <= :today
          AND source IN ('prospective','prospective_pm','prospective_legacy','top_picks_page')
          AND status LIKE 'SETTLED%'
        ORDER BY game_date
    """), {"today": today, "offset": f"-{days} days"}).fetchall()

    results = []
    for (gdate,) in dates:
        slate = apply_live_recommendation(db, game_date=gdate)

        # Full-slate stats for this day
        full = db.execute(_t("""
            SELECT COUNT(*) as bets,
                   SUM(CASE WHEN status='SETTLED_WIN' THEN 1 ELSE 0 END) as wins,
                   ROUND(SUM(actual_profit), 2) as pnl
            FROM mock_bets
            WHERE game_date = :gd
              AND source IN ('prospective','prospective_pm','prospective_legacy','top_picks_page')
              AND status LIKE 'SETTLED%'
        """), {"gd": gdate}).fetchone()

        # Live qualifying stats
        live_picks = slate["qualifying_picks"]
        live_wins = sum(1 for p in live_picks if p.get("status") == "SETTLED_WIN")
        live_losses = sum(1 for p in live_picks if p.get("status") == "SETTLED_LOSS")
        live_settled = live_wins + live_losses
        live_pnl = sum(p.get("actual_profit") or 0 for p in live_picks
                       if p.get("status") in ("SETTLED_WIN", "SETTLED_LOSS"))

        results.append({
            "game_date": gdate,
            "slate_confidence": slate["slate_confidence"],
            "full_bets": full[0],
            "full_wins": full[1],
            "full_wr": round(100 * full[1] / full[0], 1) if full[0] else 0,
            "full_pnl": full[2],
            "live_qualifying": len(live_picks),
            "live_wins": live_wins,
            "live_wr": round(100 * live_wins / live_settled, 1) if live_settled else None,
            "live_pnl": round(live_pnl, 2),
            "flags": slate["slate_summary"]["flags"],
        })

    return results


# ─── Performance / trust metrics ──────────────────────────────────────────────

def get_mock_performance(
    db: Session,
    days: int = 30,
    bet_category: Optional[str] = None,
    game_date: Optional[str] = None,
    source: Optional[str] = "simulation",
) -> dict:
    """
    Aggregate win rate, P&L, and trust metrics for settled mock bets.

    source: 'simulation' (default) filters to _LQ_SIM_SOURCES only.
            None or 'all' includes everything.
    bet_category: 'team' | 'player_prop' | None (all).
    game_date:    'YYYY-MM-DD' or 'today' — filter by game_date instead of settled_at window.

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
    # Resolve date sentinels to CDT date strings
    if game_date == "today":
        game_date = datetime.now(_CT).strftime("%Y-%m-%d")
    elif game_date == "yesterday":
        game_date = (datetime.now(_CT) - timedelta(days=1)).strftime("%Y-%m-%d")

    since = datetime.utcnow() - timedelta(days=days)

    # Resolve source filter to a list of allowed source values
    _src_filter: list[str] | None = None
    if source == "simulation":
        _src_filter = list(_LQ_SIM_SOURCES)
    elif source and source != "all":
        _src_filter = [source]
    # None / "all" → no source filter

    if bet_category:
        # Only include bets where every leg matches the category.
        # Use a raw SQL subquery to avoid loading thousands of IDs into Python
        # (large IN clauses cause SQLite page errors on big databases).
        from sqlalchemy import text as _text
        _src_clause = ""
        _params: dict = {"cat": bet_category}
        if _src_filter:
            _src_placeholders = ",".join(f":src{i}" for i in range(len(_src_filter)))
            _src_clause = f" AND source IN ({_src_placeholders})"
            _params.update({f"src{i}": s for i, s in enumerate(_src_filter)})
        if game_date:
            cat_subq = _text(f"""
                SELECT id FROM mock_bets
                WHERE status IN ('SETTLED_WIN','SETTLED_LOSS')
                  AND game_date = :gdate{_src_clause}
                  AND id IN (
                      SELECT mock_bet_id FROM mock_bet_legs
                      GROUP BY mock_bet_id
                      HAVING SUM(CASE WHEN COALESCE(bet_category,'team')=:cat THEN 1 ELSE 0 END)=COUNT(*)
                  )
            """)
            _params["gdate"] = game_date
            matched_ids = [row[0] for row in db.execute(cat_subq, _params)]
        else:
            cat_subq = _text(f"""
                SELECT id FROM mock_bets
                WHERE status IN ('SETTLED_WIN','SETTLED_LOSS')
                  AND settled_at >= :since{_src_clause}
                  AND id IN (
                      SELECT mock_bet_id FROM mock_bet_legs
                      GROUP BY mock_bet_id
                      HAVING SUM(CASE WHEN COALESCE(bet_category,'team')=:cat THEN 1 ELSE 0 END)=COUNT(*)
                  )
            """)
            _params["since"] = since
            matched_ids = [row[0] for row in db.execute(cat_subq, _params)]
        if not matched_ids:
            settled = []
        else:
            # Batch into chunks of 500 to keep IN clause safe
            settled = []
            for i in range(0, len(matched_ids), 500):
                chunk = matched_ids[i:i+500]
                settled += db.query(MockBet).filter(MockBet.id.in_(chunk)).all()
    elif game_date:
        q = db.query(MockBet).filter(
            MockBet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
            MockBet.game_date == game_date,
        )
        if _src_filter:
            q = q.filter(MockBet.source.in_(_src_filter))
        settled = q.all()
    else:
        q = db.query(MockBet).filter(
            MockBet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
            MockBet.settled_at >= since,
        )
        if _src_filter:
            q = q.filter(MockBet.source.in_(_src_filter))
        settled = q.all()

    total_settled = len(settled)
    wins          = sum(1 for m in settled if m.status == "SETTLED_WIN")
    pnl           = sum((m.actual_profit or 0) for m in settled)
    total_staked  = sum((m.amount or 0) for m in settled)
    win_rate      = round(wins / total_settled * 100, 1) if total_settled else None
    roi_pct       = round(pnl / total_staked * 100, 1) if total_staked else None

    # Avg legs across settled mocks (to compute baseline)
    def _safe_legs(v) -> int:
        try:
            return int(v) if v is not None else 1
        except (ValueError, TypeError):
            return 1
    avg_legs = sum(_safe_legs(m.legs) for m in settled) / max(total_settled, 1)
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
    sport_stats: dict = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
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
        sport_stats[s]["pnl"]  += m.actual_profit or 0

        mu = m.model_used or "combined"
        model_stats[mu]["bets"] += 1
        model_stats[mu]["wins"] += 1 if m.status == "SETTLED_WIN" else 0
        if m.model_auc and not model_stats[mu]["auc"]:
            model_stats[mu]["auc"] = m.model_auc

    def _pct(wins, total):
        return round(wins / total * 100, 1) if total else None

    if bet_category:
        from sqlalchemy import text as _pend_text
        if game_date:
            _pend_subq = _pend_text("""
                SELECT COUNT(*) FROM mock_bets
                WHERE status='PENDING'
                  AND game_date = :gdate
                  AND id IN (
                      SELECT mock_bet_id FROM mock_bet_legs
                      GROUP BY mock_bet_id
                      HAVING SUM(CASE WHEN COALESCE(bet_category,'team')=:cat THEN 1 ELSE 0 END)=COUNT(*)
                  )
            """)
            pending_count = db.execute(_pend_subq, {"gdate": game_date, "cat": bet_category}).scalar() or 0
        else:
            _pend_subq = _pend_text("""
                SELECT COUNT(*) FROM mock_bets
                WHERE status='PENDING'
                  AND id IN (
                      SELECT mock_bet_id FROM mock_bet_legs
                      GROUP BY mock_bet_id
                      HAVING SUM(CASE WHEN COALESCE(bet_category,'team')=:cat THEN 1 ELSE 0 END)=COUNT(*)
                  )
            """)
            pending_count = db.execute(_pend_subq, {"cat": bet_category}).scalar() or 0
    elif game_date:
        _pq = db.query(MockBet).filter(
            MockBet.status == "PENDING",
            MockBet.game_date == game_date,
        )
        if _src_filter:
            _pq = _pq.filter(MockBet.source.in_(_src_filter))
        pending_count = _pq.count()
    else:
        _pq = db.query(MockBet).filter(MockBet.status == "PENDING")
        if _src_filter:
            _pq = _pq.filter(MockBet.source.in_(_src_filter))
        pending_count = _pq.count()

    # Recent daily trend (last 7 days) — bucket by game_date (match start date),
    # not settlement date, so late-night games stay on the day they were played.
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

    # ── Retroactive stats — use a 180-day independent query so retro bets
    # are always visible regardless of the live performance window (days param).
    # retro_bets above may be empty if the performance window < age of retro data.
    retro_all = retro_bets  # start with any retro within the current window
    if not retro_all:
        # Expand to 180-day lookback specifically for retro bets
        try:
            from sqlalchemy import text as _retro_text
            _retro_since = datetime.utcnow() - timedelta(days=180)
            _retro_raw = db.query(MockBet).filter(
                MockBet.source == "retroactive_mock",
                MockBet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
                MockBet.settled_at >= _retro_since,
            ).all()
            if _retro_raw:
                retro_all = _retro_raw
                retro_wins = sum(1 for m in retro_all if m.status == "SETTLED_WIN")
                retro_wr   = _pct(retro_wins, len(retro_all))
        except Exception:
            pass

    # Retroactive trend for chart (by game_date, not settled_at)
    retro_trend = []
    if retro_all:
        retro_by_date: dict = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
        for m in retro_all:
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

    # ── Tier breakdown (signal-based bet_tier column) ────────────────────
    def _tier_stats(subset):
        n    = len(subset)
        wins = sum(1 for m in subset if m.status == "SETTLED_WIN")
        pnl  = round(sum((m.actual_profit or 0) for m in subset), 2)
        return {"n": n, "wins": wins, "win_rate": _pct(wins, n), "pnl": pnl}

    t1_bets = [m for m in settled if getattr(m, "bet_tier", "") == "T1"]
    t2_bets = [m for m in settled if getattr(m, "bet_tier", "") == "T2"]
    t3_bets = [m for m in settled if getattr(m, "bet_tier", "") == "T3"]
    by_tier = [
        {"tier": "T1", "label": "High conviction", **_tier_stats(t1_bets)},
        {"tier": "T2", "label": "Moderate conviction", **_tier_stats(t2_bets)},
        {"tier": "T3", "label": "Monitor only",    **_tier_stats(t3_bets)},
    ]

    # ── Compact line quality embed ─────────────────────────────────────────
    # Pass the same date window used for performance so the signal is in sync.
    _lq_since = game_date if game_date else (datetime.now(_CT) - timedelta(days=days)).strftime("%Y-%m-%d")
    _lq_until = game_date  # None unless exact-day filter
    try:
        _lq = get_line_quality_summary(db, since=_lq_since, until=_lq_until)
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
        "retro_settled":    len(retro_all),
        "retro_wins":       retro_wins,
        "retro_win_rate":   retro_wr,
        "by_confidence":    [
            {"level": k, "bets": v["bets"], "win_rate": _pct(v["wins"], v["bets"])}
            for k, v in conf_stats.items()
        ],
        "by_sport": [
            {"sport": k, "bets": v["bets"], "win_rate": _pct(v["wins"], v["bets"]),
             "pnl": round(v["pnl"], 2)}
            for k, v in sport_stats.items()
        ],
        "by_model": [
            {"model_used": k, "bets": v["bets"],
             "win_rate": _pct(v["wins"], v["bets"]), "auc": v["auc"]}
            for k, v in model_stats.items()
        ],
        "by_tier":               by_tier,
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

# ── Tiered exploration budget (Phase B diagnostic architecture) ───────────────
# Exploration bets are NOT betting picks — they are data collection instruments.
# Each tier has a distinct diagnostic purpose, volume budget, and eligibility rules.
# All exploration bets: source='exploration', amount=0.0, single-leg, zero stake.
# Only prospective picks use MARKET_ELIGIBILITY gates; exploration bypasses them.
EXPLORATION_BUDGET: dict = {
    # Tier 1: Validation — 80% of volume
    # Purpose: confirm calibration holds as data accumulates in proven-edge markets.
    # Full WP range ensures we see overconfident AND well-calibrated zones.
    "validation": {
        "allocation": 0.80,
        "markets": [
            ("MLB", "alternate_spreads"),
            ("NBA", "alternate_spreads"),
            ("NBA", "h2h"),
            ("NHL", "h2h"),
        ],
        "wp_range": (55.0, 95.0),
        "frequency": "daily",
        "max_per_market_per_week": None,   # no cap — fill allocation
    },
    # Tier 2: Expansion — 15% of volume
    # Purpose: build initial data for markets not yet in prospective generation.
    # When 30+ legs accumulate with WR >= expected - 5pp, promote to prospective.
    "expansion": {
        "allocation": 0.15,
        "markets": [
            ("NBA", "player_points"),
            ("NBA", "player_rebounds"),
            ("NBA", "player_assists"),
            ("MLB", "player_hits"),
            ("MLB", "player_total_bases"),
            ("MLB", "totals"),
            ("MLB", "alternate_totals"),
            ("NHL", "totals"),
            ("NHL", "player_shots_on_goal"),
            ("World Cup", "h2h"),
            ("World Cup", "spreads"),
            ("World Cup", "totals"),
        ],
        "wp_range": (55.0, 85.0),
        "frequency": "daily",
        "max_per_market_per_week": 30,
    },
    # Tier 3: Boundary testing — 5% of volume
    # Purpose: detect when broken zones start recovering (warrant re-testing for prospective).
    # Very low volume to not waste resources on confirmed bad markets.
    "boundary_testing": {
        "allocation": 0.05,
        "markets": [
            ("MLB", "h2h"),
            ("MLB", "spreads"),
            ("NHL", "spreads"),
        ],
        "wp_range": (55.0, 75.0),
        "frequency": "weekly",           # once per week per market, not daily
        "max_per_market_per_week": 5,
    },
}

# Expected WR per market (from Phase A diagnostic + historical baseline)
# Used by promotion logic: promote if actual_wr >= expected_wr - 5pp.
_EXPLORATION_EXPECTED_WR: dict[tuple, float] = {
    ("MLB", "alternate_spreads"):    0.68,
    ("NBA", "alternate_spreads"):    0.65,
    ("NBA", "h2h"):                  0.58,
    ("NHL", "h2h"):                  0.62,
    ("NBA", "player_points"):        0.60,
    ("NBA", "player_rebounds"):      0.60,
    ("NBA", "player_assists"):       0.58,
    ("MLB", "player_hits"):          0.58,
    ("MLB", "player_total_bases"):   0.58,
    ("MLB", "totals"):               0.58,
    ("MLB", "alternate_totals"):     0.60,
    ("NHL", "totals"):               0.58,
    ("NHL", "player_shots_on_goal"): 0.58,
    ("World Cup", "h2h"):            0.55,    # new — building calibration data
    ("World Cup", "spreads"):        0.55,
    ("World Cup", "totals"):         0.55,
    ("MLB", "h2h"):                  0.47,    # known broken — baseline for detection
    ("MLB", "spreads"):              0.33,    # known broken
    ("NHL", "spreads"):              0.09,    # known broken
}

_EXPLORATION_DAILY_TARGET = 40    # total exploration bets per day across all tiers


def get_line_quality_summary(
    db: Session,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    """
    Aggregate line quality metrics from settled mock_bet_legs.

    since: ISO date string floor (default: SYSTEM_LAUNCH_DATE = 2026-04-29).
    until: ISO date string ceiling (default: no upper bound).
           When both are the same date, filters to that single day.

    Historical data before SYSTEM_LAUNCH_DATE continues to feed model learning
    (Component A) but is excluded here so the display shows post-launch performance.

    Returns overall directional accuracy, avg line delta, and A/B recovery
    rate, broken down by sport/market.
    """
    since_str = since or SYSTEM_LAUNCH_DATE
    _src_in = ",".join(f"'{s}'" for s in _LQ_SIM_SOURCES)
    until_clause = "AND mb.game_date <= :until" if until else ""
    until_params: dict = {"until": until} if until else {}
    try:
        from sqlalchemy import text as _lq_text
        rows_raw = db.execute(
            _lq_text(f"""
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
              AND COALESCE(ml.excluded_from_metrics, 0) = 0
              AND ml.direction_correct IS NOT NULL
              AND mb.source IN ({_src_in})
              AND mb.game_date >= :since
              {until_clause}
            ORDER BY mb.game_date
            """),
            {"since": since_str, **until_params},
        ).fetchall()
        rows = [dict(r._mapping) for r in rows_raw]

        # ── Exploration stats (separate query, NOT mixed into main metrics) ──
        exp_rows_raw = db.execute(
            _lq_text(f"""
            SELECT ml.direction_correct, ml.leg_result, ml.sport, ml.market_type
            FROM   mock_bet_legs ml
            JOIN   mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE  mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND  ml.leg_result IN ('WIN', 'LOSS')
              AND  COALESCE(ml.excluded_from_metrics, 0) = 0
              AND  ml.direction_correct IS NOT NULL
              AND  mb.source = 'exploration'
              AND  mb.game_date >= :since
              {until_clause}
            ORDER BY mb.game_date
            """),
            {"since": since_str, **until_params},
        ).fetchall()
        exp_rows = [dict(r._mapping) for r in exp_rows_raw]
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
        from sqlalchemy import text as _lq_text2
        _clv_rows = db.execute(
            _lq_text2(f"""
            SELECT ml.clv_cents, ml.clv_available
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE ml.clv_available = 1
              AND mb.source IN ({_src_in})
              AND date(mb.generated_at) >= :since
            """),
            {"since": since_str},
        ).fetchall()
        clv_legs_evaluated = len(_clv_rows)
        # clv_cents < 0 → open_odds > close_odds → positive CLV (beat the line)
        clv_legs_positive  = sum(1 for r in _clv_rows if (r._mapping.get("clv_cents") or 0) < 0)
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


# ─── CLV backfill ────────────────────────────────────────────────────────────

def backfill_clv_opening(db: Session) -> dict:
    """
    One-time / idempotent backfill of open_odds for legs that have a
    fixture_id but no open_odds stored.

    Priority:
      1. alt_lines earliest snapshot (morning fetch is opening proxy)
      2. line_snapshots earliest record (stale but better than nothing)

    h2h (moneyline) legs are skipped — no spread/total data in alt_lines.

    Returns {"updated": N, "skipped_no_data": M, "skipped_h2h": K}.
    """
    from sqlalchemy import text as _t

    rows = db.execute(_t("""
        SELECT id, fixture_id, description, market_type
        FROM mock_bet_legs
        WHERE fixture_id IS NOT NULL AND fixture_id != ''
          AND open_odds IS NULL
    """)).fetchall()

    updated = skipped_no_data = skipped_h2h = 0

    for row in rows:
        lid, fid, desc, mkt = row[0], row[1], row[2] or "", row[3] or ""
        mkt_key, _ = _parse_outcome_from_desc(desc, mkt)
        if mkt_key == "h2h":
            skipped_h2h += 1
            continue

        # Try alt_lines earliest record
        am = _lookup_alt_lines_odds(fid, desc, mkt, _HIST_DB_PATH, earliest=True)
        # Fall back to line_snapshots
        if am is None:
            am = _lookup_snap_odds(fid, desc, mkt, _HIST_DB_PATH, before_dt=None)

        if am is not None:
            db.execute(_t("UPDATE mock_bet_legs SET open_odds = :v WHERE id = :id"),
                       {"v": am, "id": lid})
            updated += 1
        else:
            skipped_no_data += 1

    with DB_WRITE_LOCK:
        db.commit()
    print(f"[backfill_clv_opening] updated={updated} skipped_h2h={skipped_h2h} "
          f"skipped_no_data={skipped_no_data}")
    return {"updated": updated, "skipped_h2h": skipped_h2h, "skipped_no_data": skipped_no_data}


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
        con = sqlite3.connect(_BETS_DB_PATH, timeout=15)
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
                      game_date: str,
                      spread_value: Optional[float] = None) -> Optional[float]:
    """
    Score a historical game using the current sub-model.
    Returns probability that home team covers ATS, or None if unavailable.
    """
    try:
        import ml_model as ml
        prob = ml.predict_game_ats(
            sport=sport, home_team=home_team, away_team=away_team,
            game_date=game_date,
            spread_value=spread_value,
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
            with DB_WRITE_LOCK:
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


# ─── Top-picks calibration / validation signal ────────────────────────────────

def get_top_picks_calibration(db: Session, days: int = 30) -> dict:
    """
    Calibration and validation signal for top_picks_page simulation bets.

    Answers three questions:
      1. Parlay-level calibration — does predicted_win_prob (displayed as
         "parlay win probability") actually predict parlay outcomes?
      2. Leg-level accuracy — does per-leg win_prob predict leg wins?
      3. Min-leg-confidence vs parlay outcome — does higher model_confidence_avg
         (displayed as "min leg conf") correlate with parlay wins?
      4. Kelly ROI — does a higher Kelly stake (more confident picks) produce
         better ROI than lower-stake picks?

    Returns
    -------
    {
      n_settled           : int          # settled top_picks_page bets
      parlay_calibration  : [            # grouped by predicted_win_prob bucket
          { bucket, label, n, wins, actual_win_pct, predicted_mid, delta_pp }
      ]
      leg_accuracy        : [            # grouped by per-leg win_prob bucket
          { bucket, n_legs, wins, actual_win_pct, predicted_mid, delta_pp }
      ]
      min_leg_conf_buckets : [           # grouped by model_confidence_avg
          { bucket, n, wins, actual_win_pct, avg_roi_pct }
      ]
      kelly_roi_buckets   : [            # grouped by Kelly stake amount
          { bucket, n, wins, actual_win_pct, avg_roi_pct }
      ]
      overall : {
          win_rate_pct, roi_pct, avg_predicted_wp, brier_score,
          avg_leg_wp, leg_win_rate_pct, leg_brier_score
      }
      interpretation : str
      data_warning   : str | None        # shown when n < 20
    }
    """
    since_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        from sqlalchemy import text as _text

        # ── Parlay-level rows — use the shared SQLAlchemy session ──────────────
        parlay_raw = db.execute(
            _text("""
            SELECT
                mb.predicted_win_prob,
                mb.model_confidence_avg,
                mb.amount,
                mb.status,
                mb.actual_profit
            FROM mock_bets mb
            WHERE mb.source = 'top_picks_page'
              AND mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND date(mb.generated_at) >= :since
            ORDER BY mb.generated_at
            """),
            {"since": since_str},
        ).fetchall()
        parlay_rows = [dict(r._mapping) for r in parlay_raw]

        # ── Leg-level rows ─────────────────────────────────────────────────────
        leg_raw = db.execute(
            _text("""
            SELECT
                ml.win_prob,
                ml.leg_result
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE mb.source = 'top_picks_page'
              AND mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND date(mb.generated_at) >= :since
              AND ml.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(ml.excluded_from_metrics, 0) = 0
            """),
            {"since": since_str},
        ).fetchall()
        leg_rows = [dict(r._mapping) for r in leg_raw]

    except Exception as exc:
        return {"error": str(exc), "n_settled": 0}

    n_settled = len(parlay_rows)
    data_warning = None
    if n_settled < 20:
        data_warning = (
            f"Only {n_settled} settled top_picks_page bets — calibration curves "
            "will stabilize once ≥20 bets are settled."
        )

    # ── Parlay-level calibration ───────────────────────────────────────────────
    # Buckets: <40, 40-50, 50-60, 60-70, 70-80, ≥80
    PARLAY_BUCKETS = [
        (0,   40,  "<40%"),
        (40,  50,  "40–50%"),
        (50,  60,  "50–60%"),
        (60,  70,  "60–70%"),
        (70,  80,  "70–80%"),
        (80,  101, "≥80%"),
    ]
    from collections import defaultdict
    pb_stats: dict[str, dict] = {}
    for lo, hi, label in PARLAY_BUCKETS:
        pb_stats[label] = {"lo": lo, "hi": hi, "n": 0, "wins": 0,
                           "sum_pred": 0.0, "sum_profit": 0.0, "sum_staked": 0.0}

    brier_sum_parlay = 0.0
    sum_pred_wp = 0.0
    wins_total  = 0
    total_staked = 0.0
    total_profit = 0.0

    for r in parlay_rows:
        wp   = (r["predicted_win_prob"] or 0.0)   # already 0-100
        won  = 1 if r["status"] == "SETTLED_WIN" else 0
        amt  = r["amount"] or 0.0
        prof = r["actual_profit"] or 0.0

        sum_pred_wp  += wp
        wins_total   += won
        total_staked += amt
        total_profit += prof

        # Brier score: (predicted_prob - actual_outcome)²
        p_frac = wp / 100.0
        brier_sum_parlay += (p_frac - won) ** 2

        # Bucket assignment
        for lo, hi, label in PARLAY_BUCKETS:
            if lo <= wp < hi:
                pb_stats[label]["n"]          += 1
                pb_stats[label]["wins"]        += won
                pb_stats[label]["sum_pred"]   += wp
                pb_stats[label]["sum_profit"] += prof
                pb_stats[label]["sum_staked"] += amt
                break

    overall_wr  = round(wins_total / n_settled * 100, 1) if n_settled else None
    overall_roi = round(total_profit / total_staked * 100, 1) if total_staked else None
    avg_pred_wp = round(sum_pred_wp / n_settled, 1) if n_settled else None
    parlay_brier = round(brier_sum_parlay / n_settled, 4) if n_settled else None

    parlay_calibration = []
    for lo, hi, label in PARLAY_BUCKETS:
        s = pb_stats[label]
        if s["n"] == 0:
            continue
        actual_pct  = round(s["wins"] / s["n"] * 100, 1)
        pred_mid    = round(s["sum_pred"] / s["n"], 1)
        delta_pp    = round(actual_pct - pred_mid, 1)
        avg_roi_pct = round(s["sum_profit"] / s["sum_staked"] * 100, 1) if s["sum_staked"] else None
        parlay_calibration.append({
            "bucket":          label,
            "lo":              lo,
            "hi":              hi,
            "n":               s["n"],
            "wins":            s["wins"],
            "actual_win_pct":  actual_pct,
            "predicted_mid":   pred_mid,
            "delta_pp":        delta_pp,   # +ve = model under-confident, -ve = over-confident
            "avg_roi_pct":     avg_roi_pct,
        })

    # ── Leg-level accuracy ────────────────────────────────────────────────────
    # Buckets by win_prob: <55, 55-65, 65-75, 75-85, ≥85
    LEG_BUCKETS = [
        (0,   55, "<55%"),
        (55,  65, "55–65%"),
        (65,  75, "65–75%"),
        (75,  85, "75–85%"),
        (85, 101, "≥85%"),
    ]
    lb_stats: dict[str, dict] = {}
    for lo, hi, label in LEG_BUCKETS:
        lb_stats[label] = {"lo": lo, "hi": hi, "n": 0, "wins": 0, "sum_pred": 0.0}

    brier_sum_leg = 0.0
    leg_wins_total = 0
    n_legs_total   = 0
    sum_leg_wp     = 0.0

    for r in leg_rows:
        wp  = float(r["win_prob"] or 0.0)   # already 0-100
        won = 1 if r["leg_result"] == "WIN" else 0

        sum_leg_wp    += wp
        leg_wins_total += won
        n_legs_total  += 1

        p_frac = wp / 100.0
        brier_sum_leg += (p_frac - won) ** 2

        for lo, hi, label in LEG_BUCKETS:
            if lo <= wp < hi:
                lb_stats[label]["n"]        += 1
                lb_stats[label]["wins"]     += won
                lb_stats[label]["sum_pred"] += wp
                break

    leg_wr    = round(leg_wins_total / n_legs_total * 100, 1) if n_legs_total else None
    avg_leg_wp = round(sum_leg_wp / n_legs_total, 1) if n_legs_total else None
    leg_brier  = round(brier_sum_leg / n_legs_total, 4) if n_legs_total else None

    leg_accuracy = []
    for lo, hi, label in LEG_BUCKETS:
        s = lb_stats[label]
        if s["n"] == 0:
            continue
        actual_pct = round(s["wins"] / s["n"] * 100, 1)
        pred_mid   = round(s["sum_pred"] / s["n"], 1)
        delta_pp   = round(actual_pct - pred_mid, 1)
        leg_accuracy.append({
            "bucket":         label,
            "lo":             lo,
            "hi":             hi,
            "n_legs":         s["n"],
            "wins":           s["wins"],
            "actual_win_pct": actual_pct,
            "predicted_mid":  pred_mid,
            "delta_pp":       delta_pp,
        })

    # ── Min-leg-confidence buckets ────────────────────────────────────────────
    # model_confidence_avg is already stored as 0-100
    MLC_BUCKETS = [
        (0,   60,  "<60%"),
        (60,  70,  "60–70%"),
        (70,  80,  "70–80%"),
        (80,  90,  "80–90%"),
        (90, 101,  "≥90%"),
    ]
    mlc_stats: dict[str, dict] = {}
    for lo, hi, label in MLC_BUCKETS:
        mlc_stats[label] = {"lo": lo, "hi": hi, "n": 0, "wins": 0,
                            "sum_profit": 0.0, "sum_staked": 0.0}

    for r in parlay_rows:
        mc  = (r["model_confidence_avg"] or 0.0)
        won = 1 if r["status"] == "SETTLED_WIN" else 0
        amt = r["amount"] or 0.0
        prof = r["actual_profit"] or 0.0
        for lo, hi, label in MLC_BUCKETS:
            if lo <= mc < hi:
                mlc_stats[label]["n"]          += 1
                mlc_stats[label]["wins"]        += won
                mlc_stats[label]["sum_profit"] += prof
                mlc_stats[label]["sum_staked"] += amt
                break

    min_leg_conf_buckets = []
    for lo, hi, label in MLC_BUCKETS:
        s = mlc_stats[label]
        if s["n"] == 0:
            continue
        actual_pct  = round(s["wins"] / s["n"] * 100, 1)
        avg_roi_pct = round(s["sum_profit"] / s["sum_staked"] * 100, 1) if s["sum_staked"] else None
        min_leg_conf_buckets.append({
            "bucket":         label,
            "lo":             lo,
            "hi":             hi,
            "n":              s["n"],
            "wins":           s["wins"],
            "actual_win_pct": actual_pct,
            "avg_roi_pct":    avg_roi_pct,
        })

    # ── Kelly-stake ROI buckets ───────────────────────────────────────────────
    KELLY_BUCKETS = [
        (0,    3,   "<$3"),
        (3,    6,   "$3–6"),
        (6,    10,  "$6–10"),
        (10,   15,  "$10–15"),
        (15,   999, "≥$15"),
    ]
    kelly_stats: dict[str, dict] = {}
    for lo, hi, label in KELLY_BUCKETS:
        kelly_stats[label] = {"lo": lo, "hi": hi, "n": 0, "wins": 0,
                              "sum_profit": 0.0, "sum_staked": 0.0}

    for r in parlay_rows:
        amt  = r["amount"] or 0.0
        won  = 1 if r["status"] == "SETTLED_WIN" else 0
        prof = r["actual_profit"] or 0.0
        for lo, hi, label in KELLY_BUCKETS:
            if lo <= amt < hi:
                kelly_stats[label]["n"]          += 1
                kelly_stats[label]["wins"]        += won
                kelly_stats[label]["sum_profit"] += prof
                kelly_stats[label]["sum_staked"] += amt
                break

    kelly_roi_buckets = []
    for lo, hi, label in KELLY_BUCKETS:
        s = kelly_stats[label]
        if s["n"] == 0:
            continue
        actual_pct  = round(s["wins"] / s["n"] * 100, 1)
        avg_roi_pct = round(s["sum_profit"] / s["sum_staked"] * 100, 1) if s["sum_staked"] else None
        kelly_roi_buckets.append({
            "bucket":         label,
            "lo":             lo,
            "hi":             hi,
            "n":              s["n"],
            "wins":           s["wins"],
            "actual_win_pct": actual_pct,
            "avg_roi_pct":    avg_roi_pct,
        })

    # ── Plain-English interpretation ──────────────────────────────────────────
    interp_parts = []

    if n_settled < 10:
        interp_parts.append(
            f"Only {n_settled} settled bets — too early for reliable calibration."
        )
    else:
        # Parlay WP accuracy
        if avg_pred_wp and overall_wr is not None:
            delta = round(overall_wr - avg_pred_wp, 1)
            if abs(delta) <= 5:
                interp_parts.append(
                    f"Parlay WP is well-calibrated: model predicts {avg_pred_wp}%, "
                    f"actual win rate is {overall_wr}% (Δ{delta:+.1f}pp)."
                )
            elif delta > 5:
                interp_parts.append(
                    f"Model is under-confident on parlays: predicts {avg_pred_wp}% "
                    f"but actual win rate is {overall_wr}% (Δ{delta:+.1f}pp)."
                )
            else:
                interp_parts.append(
                    f"Model is over-confident on parlays: predicts {avg_pred_wp}% "
                    f"but actual win rate is {overall_wr}% (Δ{delta:+.1f}pp)."
                )

        # Leg accuracy
        if avg_leg_wp and leg_wr is not None:
            leg_delta = round(leg_wr - avg_leg_wp, 1)
            if abs(leg_delta) <= 5:
                interp_parts.append(
                    f"Per-leg confidence is well-calibrated: avg {avg_leg_wp}% predicted, "
                    f"{leg_wr}% actual leg win rate (Δ{leg_delta:+.1f}pp)."
                )
            elif leg_delta > 5:
                interp_parts.append(
                    f"Legs are outperforming their confidence: {avg_leg_wp}% predicted, "
                    f"{leg_wr}% actual (Δ{leg_delta:+.1f}pp) — model is conservative."
                )
            else:
                interp_parts.append(
                    f"Legs are under-performing their confidence: {avg_leg_wp}% predicted, "
                    f"{leg_wr}% actual (Δ{leg_delta:+.1f}pp) — model assigns too-high WP."
                )

        # Brier score context
        if parlay_brier is not None:
            skill = "good" if parlay_brier < 0.20 else ("fair" if parlay_brier < 0.25 else "poor")
            interp_parts.append(
                f"Parlay Brier score: {parlay_brier} ({skill} calibration; "
                f"perfect=0, random-50%=0.25)."
            )

    interpretation = " ".join(interp_parts) if interp_parts else "Insufficient data."

    return {
        "n_settled":            n_settled,
        "n_legs":               n_legs_total,
        "days":                 days,
        "parlay_calibration":   parlay_calibration,
        "leg_accuracy":         leg_accuracy,
        "min_leg_conf_buckets": min_leg_conf_buckets,
        "kelly_roi_buckets":    kelly_roi_buckets,
        "overall": {
            "win_rate_pct":     overall_wr,
            "roi_pct":          overall_roi,
            "avg_predicted_wp": avg_pred_wp,
            "brier_score":      parlay_brier,
            "avg_leg_wp":       avg_leg_wp,
            "leg_win_rate_pct": leg_wr,
            "leg_brier_score":  leg_brier,
        },
        "interpretation": interpretation,
        "data_warning":   data_warning,
    }


# ─── Brier Score Tracking ─────────────────────────────────────────────────────

def get_brier_scores(db: Session, days: int = 30) -> dict:
    """
    Compute Brier scores for per-leg and parlay predictions, with Murphy
    decomposition by sport/market, bucketed calibration, and trend deltas.

    Returns a comprehensive dict suitable for the /api/health/brier-scores endpoint.
    """
    from sqlalchemy import text as _text
    from collections import defaultdict
    import math

    since_str       = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    # Trend windows
    older_start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    older_end_str   = (datetime.utcnow() - timedelta(days=days - 7)).strftime("%Y-%m-%d")
    recent_start_str = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    data_warning = None

    # ── 1. Overall per-leg Brier ───────────────────────────────────────────────
    leg_rows = db.execute(_text("""
        SELECT mbl.win_prob, mbl.leg_result
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND date(mb.generated_at) >= :since
          AND mbl.win_prob IS NOT NULL
    """), {"since": since_str}).fetchall()

    n_legs = len(leg_rows)
    if n_legs < 5:
        data_warning = f"Only {n_legs} settled legs in last {days} days — Brier scores unreliable."

    def _leg_brier(rows) -> float | None:
        if not rows:
            return None
        sq = [(r[0] / 100.0 - (1.0 if r[1] == 'WIN' else 0.0)) ** 2 for r in rows]
        return round(sum(sq) / len(sq), 4)

    per_leg_brier = _leg_brier(leg_rows)

    # ── 2. Overall parlay Brier ────────────────────────────────────────────────
    parlay_rows = db.execute(_text("""
        SELECT predicted_win_prob, status
        FROM mock_bets
        WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND source IN ('top_picks_page', 'prospective', 'prospective_pm')
          AND date(generated_at) >= :since
          AND predicted_win_prob IS NOT NULL
    """), {"since": since_str}).fetchall()

    n_parlays = len(parlay_rows)

    def _parlay_brier(rows) -> float | None:
        if not rows:
            return None
        sq = [(r[0] / 100.0 - (1.0 if r[1] == 'SETTLED_WIN' else 0.0)) ** 2 for r in rows]
        return round(sum(sq) / len(sq), 4)

    parlay_brier = _parlay_brier(parlay_rows)

    # ── 3. Trend (7d vs older window within the days window) ──────────────────
    def _leg_brier_window(start_str, end_str) -> float | None:
        rows = db.execute(_text("""
            SELECT mbl.win_prob, mbl.leg_result
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND date(mb.generated_at) >= :start
              AND date(mb.generated_at) < :end
              AND mbl.win_prob IS NOT NULL
        """), {"start": start_str, "end": end_str}).fetchall()
        return _leg_brier(rows)

    recent_brier = _leg_brier_window(recent_start_str,
                                     datetime.utcnow().strftime("%Y-%m-%d"))
    older_brier  = _leg_brier_window(older_start_str, older_end_str)

    trend_7d  = None
    trend_30d = None
    if recent_brier is not None and older_brier is not None:
        trend_7d = round(recent_brier - older_brier, 4)
    # trend_30d: compare current window vs prior window of same length
    prior_start_str = (datetime.utcnow() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
    prior_end_str   = since_str
    prior_brier = _leg_brier_window(prior_start_str, prior_end_str)
    if per_leg_brier is not None and prior_brier is not None:
        trend_30d = round(per_leg_brier - prior_brier, 4)

    # ── 4. By sport/market breakdown with Murphy decomposition ─────────────────
    sport_mkt_rows = db.execute(_text("""
        SELECT mbl.sport, mbl.market_type, mbl.win_prob, mbl.leg_result, mbl.grade
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND date(mb.generated_at) >= :since
          AND mbl.win_prob IS NOT NULL
          AND mbl.sport IS NOT NULL
          AND mbl.market_type IS NOT NULL
    """), {"since": since_str}).fetchall()

    # Group by (sport, market_type)
    cell_data: dict[tuple, list] = defaultdict(list)
    for r in sport_mkt_rows:
        sport, mkt, wp, result, grade = r[0], r[1], r[2], r[3], r[4]
        cell_data[(sport, mkt)].append({
            "p": wp / 100.0,
            "y": 1.0 if result == 'WIN' else 0.0,
        })

    def _murphy_decomp(items: list[dict]) -> dict:
        """Murphy decomposition using decile buckets of predicted prob."""
        N = len(items)
        if N < 2:
            return {"reliability": None, "resolution": None, "uncertainty": None}

        # Overall mean actual
        overall_mean = sum(i["y"] for i in items) / N

        # Bucket by decile of win_prob (0-10, 10-20, ... 90-100)
        buckets: dict[int, list] = defaultdict(list)
        for i in items:
            b = min(int(i["p"] * 10), 9)  # decile index 0-9
            buckets[b].append(i)

        reliability = 0.0
        resolution  = 0.0
        for b, bkt in buckets.items():
            n_k         = len(bkt)
            mean_pred_k = sum(i["p"] for i in bkt) / n_k
            mean_act_k  = sum(i["y"] for i in bkt) / n_k
            reliability += (n_k / N) * (mean_pred_k - mean_act_k) ** 2
            resolution  += (n_k / N) * (mean_act_k - overall_mean) ** 2

        uncertainty = overall_mean * (1.0 - overall_mean)

        return {
            "reliability": round(reliability, 4),
            "resolution":  round(resolution,  4),
            "uncertainty": round(uncertainty,  4),
        }

    _TIER_THRESHOLDS = [
        (0.18, "excellent"),
        (0.22, "good"),
        (0.25, "mediocre"),
    ]

    def _quality_tier(brier: float) -> str:
        for thresh, label in _TIER_THRESHOLDS:
            if brier <= thresh:
                return label
        return "poor"

    by_sport_market = []
    for (sport, mkt), items in cell_data.items():
        if len(items) < 5:
            continue
        brier = round(sum((i["p"] - i["y"]) ** 2 for i in items) / len(items), 4)
        tier  = _quality_tier(brier)
        decomp = _murphy_decomp(items)
        by_sport_market.append({
            "sport":           sport,
            "market_type":     mkt,
            "per_leg_brier":   brier,
            "n_legs":          len(items),
            "quality_tier":    tier,
            "flag_for_review": tier == "poor" and len(items) >= 30,
            "decomposition":   decomp,
        })

    by_sport_market.sort(key=lambda x: x["per_leg_brier"], reverse=True)

    # ── 5. By grade ────────────────────────────────────────────────────────────
    grade_data: dict[str, list] = defaultdict(list)
    for r in sport_mkt_rows:
        grade = r[4]
        if grade:
            grade_data[grade].append({
                "p": r[2] / 100.0,
                "y": 1.0 if r[3] == 'WIN' else 0.0,
            })
    # Also pull from main leg_rows (in case sport/market is NULL)
    for r in leg_rows:
        pass  # leg_rows doesn't include grade; sport_mkt_rows already covers it

    by_grade = []
    for grade, items in sorted(grade_data.items()):
        if not items:
            continue
        brier = round(sum((i["p"] - i["y"]) ** 2 for i in items) / len(items), 4)
        by_grade.append({"grade": grade, "brier": brier, "n": len(items)})

    # ── 6. By win_prob bucket (10-wide bins, 0–100) ────────────────────────────
    WP_BINS = [(i, i + 10) for i in range(0, 100, 10)]
    bucket_data: dict[int, list] = defaultdict(list)
    for r in leg_rows:
        wp   = r[0]
        y    = 1.0 if r[1] == 'WIN' else 0.0
        b_lo = min(int(wp / 10) * 10, 90)
        bucket_data[b_lo].append({"p": wp / 100.0, "y": y})

    by_wp_bucket = []
    for lo in range(0, 100, 10):
        items = bucket_data[lo]
        if not items:
            continue
        brier = round(sum((i["p"] - i["y"]) ** 2 for i in items) / len(items), 4)
        mid   = (lo + lo + 10) / 2.0 / 100.0
        act_wr = round(sum(i["y"] for i in items) / len(items), 4)
        by_wp_bucket.append({
            "bucket":        f"{lo}–{lo+10}",
            "predicted_mid": round(mid, 3),
            "brier":         brier,
            "n":             len(items),
            "actual_win_pct": act_wr,
        })

    return {
        "overall": {
            "per_leg_brier":  per_leg_brier,
            "parlay_brier":   parlay_brier,
            "n_legs":         n_legs,
            "n_parlays":      n_parlays,
            "trend_7d":       trend_7d,
            "trend_30d":      trend_30d,
        },
        "by_sport_market": by_sport_market,
        "by_grade":        by_grade,
        "by_wp_bucket":    by_wp_bucket,
        "days":            days,
        "data_warning":    data_warning,
    }


# ─── Brier Decomposition (Priority 3) ─────────────────────────────────────────

def get_brier_decomposition(db: Session, days: int = 30) -> dict:
    """
    Full Murphy decomposition per (sport, market_type) cell.

    Returns:
      • cells: sorted by Reliability desc (highest calibration leverage first)
        Each cell has: sport, market_type, n_legs, brier, reliability, resolution,
        uncertainty, calibration_ceiling, quality_tier, flag_for_review,
        suggested_intervention.
      • parlay_summary: parlay-level Brier + decomposition
      • high_wp_bucket: breakdown of the 90-100% WP bucket (Priority 5)
      • overall: aggregate Brier + components
    """
    from sqlalchemy import text as _text
    from collections import defaultdict

    since_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # ── Leg rows with all needed fields ──────────────────────────────────────────
    # Filter to production sources only — exploration/simulation legs are
    # intentionally low-confidence (edge-case testing) and inflate Brier.
    _PRODUCTION_SOURCES = ("prospective", "prospective_pm", "top_picks_page")
    _src_placeholders = ",".join(f":src{i}" for i in range(len(_PRODUCTION_SOURCES)))
    _src_params = {f"src{i}": s for i, s in enumerate(_PRODUCTION_SOURCES)}
    leg_rows = db.execute(_text(f"""
        SELECT mbl.sport, mbl.market_type, mbl.win_prob, mbl.leg_result,
               mbl.grade, mbl.fixture_id, mb.source
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND date(mb.generated_at) >= :since
          AND mbl.win_prob IS NOT NULL
          AND mbl.sport IS NOT NULL
          AND mbl.market_type IS NOT NULL
          AND mb.source IN ({_src_placeholders})
    """), {"since": since_str, **_src_params}).fetchall()

    cell_data: dict[tuple, list] = defaultdict(list)
    all_items: list[dict] = []

    for r in leg_rows:
        sport, mkt, wp, result, grade, fix_id, source = (
            r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        )
        item = {
            "p":      wp / 100.0,
            "y":      1.0 if result == 'WIN' else 0.0,
            "grade":  grade,
            "source": source,
            "wp_raw": wp,
        }
        cell_data[(sport, mkt)].append(item)
        all_items.append(item)

    def _murphy(items: list[dict]) -> dict:
        N = len(items)
        if N < 2:
            return {"reliability": None, "resolution": None, "uncertainty": None,
                    "calibration_ceiling": None}
        overall_mean = sum(i["y"] for i in items) / N
        buckets: dict[int, list] = defaultdict(list)
        for i in items:
            b = min(int(i["p"] * 10), 9)
            buckets[b].append(i)
        reliability = 0.0
        resolution  = 0.0
        for b, bkt in buckets.items():
            n_k         = len(bkt)
            mean_pred_k = sum(i["p"] for i in bkt) / n_k
            mean_act_k  = sum(i["y"] for i in bkt) / n_k
            reliability += (n_k / N) * (mean_pred_k - mean_act_k) ** 2
            resolution  += (n_k / N) * (mean_act_k - overall_mean) ** 2
        uncertainty         = overall_mean * (1.0 - overall_mean)
        calibration_ceiling = max(0.0, uncertainty - resolution)
        return {
            "reliability":         round(reliability,         4),
            "resolution":          round(resolution,          4),
            "uncertainty":         round(uncertainty,         4),
            "calibration_ceiling": round(calibration_ceiling, 4),
        }

    def _tier(brier: float) -> str:
        if brier <= 0.18: return "excellent"
        if brier <= 0.22: return "good"
        if brier <= 0.25: return "mediocre"
        return "poor"

    def _intervention(decomp: dict, sport: str, mkt: str) -> str:
        rel  = decomp.get("reliability") or 0.0
        res  = decomp.get("resolution")  or 0.0
        ceil = decomp.get("calibration_ceiling") or 0.0
        if rel > 0.02:
            return "calibration — apply/tighten WP haircuts for this cell"
        if ceil < 0.005:
            return "feature engineering — add discriminating signals (low resolution ceiling)"
        if res < 0.01:
            return "feature engineering — model lacks discrimination in this cell"
        return "monitor — no high-priority intervention"

    cells = []
    for (sport, mkt), items in cell_data.items():
        if len(items) < 5:
            continue
        brier  = round(sum((i["p"] - i["y"]) ** 2 for i in items) / len(items), 4)
        tier   = _tier(brier)
        decomp = _murphy(items)
        actual_wr = round(sum(i["y"] for i in items) / len(items), 4)
        cells.append({
            "sport":               sport,
            "market_type":         mkt,
            "n_legs":              len(items),
            "actual_win_rate":     actual_wr,
            "brier":               brier,
            "quality_tier":        tier,
            "flag_for_review":     tier == "poor" and len(items) >= 30,
            "reliability":         decomp["reliability"],
            "resolution":          decomp["resolution"],
            "uncertainty":         decomp["uncertainty"],
            "calibration_ceiling": decomp["calibration_ceiling"],
            "suggested_intervention": _intervention(decomp, sport, mkt),
        })

    # Sort by reliability descending (most calibration leverage first)
    cells.sort(key=lambda x: (x["reliability"] or 0.0), reverse=True)

    # ── Overall aggregate ─────────────────────────────────────────────────────────
    overall_brier  = None
    overall_decomp = {}
    if all_items:
        overall_brier  = round(sum((i["p"] - i["y"]) ** 2 for i in all_items) / len(all_items), 4)
        overall_decomp = _murphy(all_items)

    # ── Parlay-level decomposition ────────────────────────────────────────────────
    parlay_rows = db.execute(_text("""
        SELECT predicted_win_prob, status
        FROM mock_bets
        WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND source IN ('top_picks_page', 'prospective', 'prospective_pm')
          AND date(generated_at) >= :since
          AND predicted_win_prob IS NOT NULL
    """), {"since": since_str}).fetchall()

    parlay_items = [
        {"p": r[0] / 100.0, "y": 1.0 if r[1] == 'SETTLED_WIN' else 0.0}
        for r in parlay_rows
    ]
    parlay_brier  = None
    parlay_decomp = {}
    cal_parlay_brier = None
    if parlay_items:
        parlay_brier  = round(sum((i["p"] - i["y"]) ** 2 for i in parlay_items) / len(parlay_items), 4)
        parlay_decomp = _murphy(parlay_items)
        # Also compute Brier using calibrated parlay WP
        try:
            import os as _os
            from win_prob_calibration import get_calibrated_parlay_wp as _gcpwp
            _db = _os.path.join(_os.path.dirname(__file__), "..", "data", "bets.db")
            cal_items = [{"p": _gcpwp(r[0], _db) / 100.0, "y": 1.0 if r[1] == 'SETTLED_WIN' else 0.0}
                         for r in parlay_rows]
            cal_parlay_brier = round(sum((i["p"] - i["y"]) ** 2 for i in cal_items) / len(cal_items), 4)
        except Exception:
            pass

    # ── 90-100% WP bucket investigation (Priority 5) ─────────────────────────────
    high_wp_items = [i for i in all_items if i["wp_raw"] >= 90.0]
    high_wp_by_grade: dict[str, list] = defaultdict(list)
    for i in [r for r in leg_rows if r[2] >= 90.0]:
        high_wp_by_grade[r[4] or "unknown"].append({
            "p": r[2] / 100.0, "y": 1.0 if r[3] == 'WIN' else 0.0,
            "sport": r[0], "mkt": r[1],
        })

    high_wp_cells: list[dict] = []
    for grade, glist in sorted(high_wp_by_grade.items()):
        if glist:
            wr = round(sum(i["y"] for i in glist) / len(glist), 4)
            avg_pred = round(sum(i["p"] for i in glist) / len(glist), 4)
            high_wp_cells.append({
                "grade": grade, "n": len(glist),
                "avg_predicted_wp": avg_pred, "actual_win_rate": wr,
                "overconfidence_pp": round((avg_pred - wr) * 100, 2),
            })

    high_wp_overall = None
    if high_wp_items:
        wr_hw = sum(i["y"] for i in high_wp_items) / len(high_wp_items)
        avg_pred_hw = sum(i["p"] for i in high_wp_items) / len(high_wp_items)
        high_wp_overall = {
            "n": len(high_wp_items),
            "avg_predicted_wp": round(avg_pred_hw, 4),
            "actual_win_rate":  round(wr_hw, 4),
            "overconfidence_pp": round((avg_pred_hw - wr_hw) * 100, 2),
        }

    return {
        "cells":       cells,
        "overall": {
            "brier":               overall_brier,
            "n_legs":              len(all_items),
            "quality_tier":        _tier(overall_brier) if overall_brier else None,
            **overall_decomp,
        },
        "parlay_summary": {
            "brier":                  parlay_brier,
            "calibrated_brier":       cal_parlay_brier,
            "n":                      len(parlay_items),
            "quality_tier":           _tier(parlay_brier) if parlay_brier else None,
            "calibrated_quality_tier": _tier(cal_parlay_brier) if cal_parlay_brier else None,
            **parlay_decomp,
        },
        "high_wp_bucket_90_100": {
            "overall": high_wp_overall,
            "by_grade": high_wp_cells,
            "finding": (
                "Soccer alt_spreads (exempt from calibration) and soccer totals drive "
                "extreme overconfidence at 90-100% WP. n=9 too small to act on, but "
                "monitor: if n >= 30 and actual WR < 0.70, remove _ALT_MARKETS exemption "
                "for soccer alt_spreads and add a separate calibration bucket."
            ),
        },
        "feature_work_backlog": [
            {
                "cell": "NBA/alternate_spreads",
                "current_tier": "poor",
                "resolution_score": 0.0020,
                "root_cause": "resolution-limited — model lacks discriminating signals",
                "interventions": [
                    "Add pace differential (team pace vs opponent pace for the specific game)",
                    "Rest advantage: B2B flag, days since last game",
                    "Confirm lineup (starter load management, DNP flags)",
                    "Line value only — skip if alt_spread is priced < +0.5pp edge",
                ],
            },
            {
                "cell": "NHL/alternate_spreads",
                "current_tier": "poor",
                "resolution_score": 0.0344,
                "root_cause": "mix of calibration error and low resolution",
                "interventions": [
                    "Goalie matchup score: confirmed starter quality vs opponent",
                    "Home-ice advantage factor for alt puck-line bets",
                    "Score differential expectation from recent 5-game form",
                    "Avoid alt_spreads when implied probability < 55% (market uncertainty too high)",
                ],
            },
            {
                "cell": "Soccer/alternate_spreads (all leagues)",
                "current_tier": "poor",
                "resolution_score": "varies — high reliability (calibration error) is primary driver",
                "root_cause": "market-implied probability exemption (_ALT_MARKETS) is incorrect for soccer: "
                             "soccer alternate spreads are model outputs, not pure market-implied values",
                "interventions": [
                    "CRITICAL: Remove soccer alt_spreads from _ALT_MARKETS exemption",
                    "Build separate soccer alt_spread calibration buckets (< 45 bucket shows -60pp inverted haircut)",
                    "Add expected goals (xG) differential as primary feature",
                    "Home/away form split (separate calibration for home vs away alt_spread legs)",
                ],
            },
            {
                "cell": "MLB/spreads",
                "current_tier": "poor (flagged)",
                "resolution_score": "moderate",
                "root_cause": "persistent calibration error (55-65 bucket: 46.6pp haircut, POOR DB cell)",
                "interventions": [
                    "Calibration already applied (+46.6pp haircut at confidence=0.47) — let data accumulate",
                    "Add pitcher quality score: era_plus, whip, xfip vs opponent batting",
                    "Park factor correction for run-line bets",
                ],
            },
        ],
        "days": days,
    }


# ─── Change 1 helper: post-settlement calibration refresh ─────────────────────

def _refresh_calibration_after_settlement(db: Session, settled_count: int) -> dict:
    """
    Called at the end of settle_mock_bets when settled_count > 0.
    Rebuilds win_prob_calibration + parlay_calibration from fresh data.
    Returns a brief summary dict.
    """
    summary: dict = {}
    try:
        from win_prob_calibration import (
            build_calibration_table as _build_cal,
            build_parlay_calibration_table as _build_pcal,
        )
        cal_result  = _build_cal(_BETS_DB_PATH)
        pcal_result = _build_pcal(_BETS_DB_PATH)
        summary["calibration"]        = cal_result
        summary["parlay_calibration"] = pcal_result
        print(f"[Settlement] Calibration refreshed after settling {settled_count} bets — "
              f"rows={cal_result.get('written', '?')} parlay_rows={pcal_result.get('written', '?')}")
    except Exception as exc:
        summary["calibration_error"] = str(exc)
        print(f"[Settlement] Calibration refresh error (non-fatal): {exc}")
    return summary


# ─── Change 4: Pattern signals ────────────────────────────────────────────────

_PATTERN_SIGNAL_THRESHOLD = 5.0   # |actual - predicted| * sqrt(n) must exceed this


def _wp_bucket_label(wp: float) -> str:
    """Map a 0-100 win probability to a calibration bucket label."""
    if wp < 45:   return "<45"
    if wp < 55:   return "45-55"
    if wp < 65:   return "55-65"
    if wp < 75:   return "65-75"
    if wp < 85:   return "75-85"
    return "85+"


def update_pattern_signals(db: Session, settled_bet_ids: list[str]) -> int:
    """
    After a settlement batch, recompute pattern signals for affected patterns.

    Patterns tracked per leg:
      sport|market_type|wp_bucket
      sport|market_type|day_of_week       (Mon=0 … Sun=6)
      sport|n_legs_in_parlay
      cushion|sport
      sport|market_type|injury_advantage  (home / away / neutral)

    Updates bet_pattern_signals table.  Returns count of patterns updated.
    """
    from sqlalchemy import text as _text
    from datetime import timezone as _tz
    from database import BetPatternSignal

    if not settled_bet_ids:
        return 0

    # Pull all resolved legs from these bets (joined to parent for parlay leg count + date)
    placeholders = ",".join(f":id{i}" for i in range(len(settled_bet_ids)))
    params = {f"id{i}": bid for i, bid in enumerate(settled_bet_ids)}
    params["win"]  = "WIN"
    params["loss"] = "LOSS"

    rows = db.execute(_text(f"""
        SELECT mbl.sport, mbl.market_type, mbl.win_prob, mbl.leg_result,
               mbl.ale_switched, mbl.injury_net_advantage,
               mb.game_date, mb.id AS bet_id,
               COUNT(mbl2.id) AS n_legs_in_parlay
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        JOIN mock_bet_legs mbl2 ON mbl2.mock_bet_id = mb.id
        WHERE mb.id IN ({placeholders})
          AND mbl.leg_result IN (:win, :loss)
          AND mbl.win_prob IS NOT NULL
          AND mbl.sport IS NOT NULL
          AND mbl.market_type IS NOT NULL
        GROUP BY mbl.id
    """), params).fetchall()

    if not rows:
        return 0

    # Accumulate per pattern_key
    from collections import defaultdict
    buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0, "wp_sum": 0.0, "desc": ""})

    for r in rows:
        sport, mkt, wp, result, cushion, injury_adv, gdate, bet_id, n_legs = (
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]
        )
        wp    = float(wp or 50.0)
        won   = 1 if result == "WIN" else 0
        bucket = _wp_bucket_label(wp)

        # Day of week from game_date
        dow = "?"
        if gdate:
            try:
                _gd = datetime.strptime(str(gdate)[:10], "%Y-%m-%d")
                dow = ["mon","tue","wed","thu","fri","sat","sun"][_gd.weekday()]
            except Exception:
                pass

        injury_dir = "home" if (injury_adv or 0) > 0.05 else ("away" if (injury_adv or 0) < -0.05 else "neutral")

        keys_descs = [
            (f"{sport}|{mkt}|{bucket}",        f"{sport} {mkt} WP {bucket}"),
            (f"{sport}|{mkt}|dow_{dow}",        f"{sport} {mkt} {dow.upper()}"),
            (f"{sport}|nlegs_{n_legs}",         f"{sport} {n_legs}-leg parlay"),
            (f"cushion_{1 if cushion else 0}|{sport}", f"{'CUSHION' if cushion else 'NO-CUSHION'} {sport}"),
            (f"{sport}|{mkt}|inj_{injury_dir}", f"{sport} {mkt} injury={injury_dir}"),
        ]
        for key, desc in keys_descs:
            b = buckets[key]
            b["wins"]   += won
            b["total"]  += 1
            b["wp_sum"] += wp
            b["desc"]    = desc

    # Upsert pattern signals (merge with existing rows)
    updated = 0
    for key, data in buckets.items():
        n      = data["total"]
        actual = round(100.0 * data["wins"] / n, 2) if n else 0.0
        pred   = round(data["wp_sum"] / n, 2) if n else 50.0

        # Fetch existing row to accumulate running totals
        existing = db.execute(_text(
            "SELECT n_legs, actual_wr, predicted_wr FROM bet_pattern_signals WHERE pattern_key = :k"
        ), {"k": key}).fetchone()

        if existing:
            prev_n    = existing[0] or 0
            prev_wins = round((existing[1] or 0) * prev_n / 100.0)
            new_n     = prev_n + n
            new_wins  = prev_wins + data["wins"]
            actual    = round(100.0 * new_wins / new_n, 2)
            pred      = round(((existing[2] or 50.0) * prev_n + data["wp_sum"]) / new_n, 2)
            n         = new_n

        strength = round(abs(actual - pred) * (n ** 0.5), 3)

        db.execute(_text("""
            INSERT INTO bet_pattern_signals
                (pattern_key, description, n_legs, actual_wr, predicted_wr, signal_strength, last_updated)
            VALUES (:k, :d, :n, :awr, :pwr, :ss, :ts)
            ON CONFLICT(pattern_key) DO UPDATE SET
                description     = excluded.description,
                n_legs          = excluded.n_legs,
                actual_wr       = excluded.actual_wr,
                predicted_wr    = excluded.predicted_wr,
                signal_strength = excluded.signal_strength,
                last_updated    = excluded.last_updated
        """), {
            "k": key, "d": data["desc"], "n": n,
            "awr": actual, "pwr": pred, "ss": strength,
            "ts": datetime.utcnow().isoformat(),
        })
        updated += 1

        if strength > _PATTERN_SIGNAL_THRESHOLD:
            print(f"[Pattern] {key}: actual={actual}% predicted={pred}% "
                  f"n={n} strength={strength:.1f}")

    with DB_WRITE_LOCK:
        db.commit()
    return updated


def get_top_pattern_signals(db: Session, min_n: int = 20, top_k: int = 15) -> list[dict]:
    """
    Return the strongest pattern signals (highest |actual - predicted| × sqrt(n)).
    Used in daily summary and retraining readiness.
    """
    from sqlalchemy import text as _text
    rows = db.execute(_text("""
        SELECT pattern_key, description, n_legs, actual_wr, predicted_wr, signal_strength, last_updated
        FROM bet_pattern_signals
        WHERE n_legs >= :min_n
        ORDER BY signal_strength DESC
        LIMIT :top_k
    """), {"min_n": min_n, "top_k": top_k}).fetchall()

    return [
        {
            "pattern_key":     r[0],
            "description":     r[1],
            "n_legs":          r[2],
            "actual_wr":       round(r[3], 1) if r[3] is not None else None,
            "predicted_wr":    round(r[4], 1) if r[4] is not None else None,
            "signal_strength": round(r[5], 2) if r[5] is not None else None,
            "calibration_error_pp": round(abs((r[3] or 50) - (r[4] or 50)), 1),
            "last_updated":    r[6],
        }
        for r in rows
    ]


# ─── Change 2: Auto-evaluate exploration markets ──────────────────────────────

def auto_evaluate_markets(db: Session) -> list[dict]:
    """
    After calibration refresh, scan all expansion-tier markets and flag
    promote/demote candidates into pending_market_actions.

    Thresholds (matching the spec):
      PROMOTE: tier=expansion, n>=30, actual_wr >= expected_wr - 5pp
      DEMOTE:  tier=expansion, n>=30, actual_wr < expected_wr - 15pp

    Only writes NEW actions (skips markets already pending review).
    Returns list of actions written.
    """
    from sqlalchemy import text as _text

    calib = get_exploration_calibration(days=30)
    expansion_markets = calib.get("expansion", [])

    actions: list[dict] = []
    for m in expansion_markets:
        n          = m.get("n_legs", 0)
        actual_wr  = m.get("actual_wr")
        avg_wp     = m.get("avg_model_wp")
        sport      = m.get("sport", "")
        mkt        = m.get("market_type", "")
        verdict    = m.get("verdict", "")

        if n < 30 or actual_wr is None:
            continue

        expected_wr = avg_wp if avg_wp is not None else 50.0
        delta       = actual_wr - expected_wr

        if delta >= -5.0:
            action = "promote"
            reason = f"WR {actual_wr:.1f}% meets threshold (≥{expected_wr - 5:.1f}%) on {n} legs"
        elif delta < -15.0:
            action = "demote"
            reason = f"WR {actual_wr:.1f}% well below threshold (<{expected_wr - 15:.1f}%) on {n} legs"
        else:
            continue

        # Check: already pending review?
        existing = db.execute(_text("""
            SELECT id FROM pending_market_actions
            WHERE sport = :s AND market_type = :m AND action = :a AND reviewed = 0
        """), {"s": sport, "m": mkt, "a": action}).fetchone()
        if existing:
            continue  # already flagged, don't duplicate

        db.execute(_text("""
            INSERT INTO pending_market_actions
                (action, sport, market_type, tier, actual_wr, expected_wr, n_legs, reason, flagged_at, reviewed)
            VALUES (:action, :sport, :mkt, 'expansion', :awr, :ewr, :n, :reason, :ts, 0)
        """), {
            "action": action, "sport": sport, "mkt": mkt,
            "awr": round(actual_wr, 2), "ewr": round(expected_wr, 2),
            "n": n, "reason": reason,
            "ts": datetime.utcnow().isoformat(),
        })
        actions.append({
            "action": action, "sport": sport, "market_type": mkt,
            "actual_wr": round(actual_wr, 1), "expected_wr": round(expected_wr, 1),
            "n_legs": n, "reason": reason,
        })

    if actions:
        with DB_WRITE_LOCK:
            db.commit()
        print(f"[AutoEval] {len(actions)} market action(s) pending review: "
              + ", ".join(f"{a['action']} {a['sport']} {a['market_type']}" for a in actions))

    return actions


def get_pending_market_actions(db: Session) -> list[dict]:
    """Return all unreviewed pending market actions for daily summary."""
    from sqlalchemy import text as _text
    rows = db.execute(_text("""
        SELECT id, action, sport, market_type, tier,
               actual_wr, expected_wr, n_legs, reason, flagged_at
        FROM pending_market_actions
        WHERE reviewed = 0
        ORDER BY flagged_at DESC
        LIMIT 20
    """)).fetchall()
    return [
        {
            "id":           r[0],
            "action":       r[1],
            "sport":        r[2],
            "market_type":  r[3],
            "tier":         r[4],
            "actual_wr":    round(r[5], 1) if r[5] is not None else None,
            "expected_wr":  round(r[6], 1) if r[6] is not None else None,
            "n_legs":       r[7],
            "reason":       r[8],
            "flagged_at":   r[9],
            "confirm_url":  f"/api/exploration/{'promote' if r[1] == 'promote' else 'demote'}-market",
        }
        for r in rows
    ]


# ─── Change 5: Retraining readiness ──────────────────────────────────────────

def check_retraining_readiness(db: Session) -> dict:
    """
    Evaluate whether model retraining would be productive right now.

    Checks:
      1. New settled legs since last training run (threshold: 500)
      2. Calibration drift — max reliability score across cells (threshold: 0.05)
      3. Brier degradation vs trained-on Brier (threshold: +0.02)

    Does NOT auto-retrain. Surfaces reasons in daily summary.
    """
    from sqlalchemy import text as _text
    from database import ModelRun

    reasons: list[str] = []
    ready = False

    # ── Last training run ──────────────────────────────────────────────────────
    last_run = db.query(ModelRun).order_by(ModelRun.id.desc()).first()
    last_trained_at  = last_run.run_at.strftime("%Y-%m-%d") if last_run else None
    last_trained_auc = float(last_run.roc_auc) if last_run and last_run.roc_auc else None

    # ── New legs since last train ──────────────────────────────────────────────
    since_clause = f"AND date(mb.settled_at) > '{last_trained_at}'" if last_trained_at else ""
    new_legs_row = db.execute(_text(f"""
        SELECT COUNT(*)
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          {since_clause}
    """)).fetchone()
    new_legs = int(new_legs_row[0]) if new_legs_row else 0

    if new_legs >= 500:
        reasons.append(f"{new_legs} new settled legs since last train (threshold: 500)")
        ready = True

    # ── Calibration drift (max reliability from recent cells) ─────────────────
    max_reliability = 0.0
    drift_cell      = None
    try:
        cells = get_brier_decomposition(db, days=30).get("cells", [])
        for cell in cells:
            r = cell.get("reliability", 0.0)
            if r > max_reliability:
                max_reliability = r
                drift_cell = f"{cell['sport']} {cell['market_type']}"
    except Exception:
        pass

    if max_reliability > 0.05:
        reasons.append(
            f"Calibration drift {max_reliability:.3f} in {drift_cell} exceeds threshold (0.05)"
        )
        ready = True

    # ── Brier degradation vs last training ────────────────────────────────────
    current_brier = None
    try:
        overall = get_brier_decomposition(db, days=30).get("overall", {})
        current_brier = overall.get("brier")
    except Exception:
        pass

    # Use model_performance_log to get Brier at training time if available
    last_trained_brier: float | None = None
    try:
        mpl_row = db.execute(_text("""
            SELECT brier FROM model_performance_log
            ORDER BY evaluated_at DESC LIMIT 1
        """)).fetchone()
        if mpl_row:
            last_trained_brier = float(mpl_row[0])
    except Exception:
        pass

    if current_brier is not None and last_trained_brier is not None:
        degradation = current_brier - last_trained_brier
        if degradation > 0.02:
            reasons.append(
                f"Brier degraded {last_trained_brier:.3f} → {current_brier:.3f} "
                f"(+{degradation:.3f}, threshold: +0.02)"
            )
            ready = True

    # ── Top pattern signals as additional context ──────────────────────────────
    top_patterns = get_top_pattern_signals(db, min_n=30, top_k=3)

    # ── B-weight (model confidence) correlation analysis ───────────────────
    # Gate: 50+ post-fix (>= 2026-05-16), non-alt-spread, single-leg resolved legs.
    # Computes corr(predicted_win_prob, outcome) and corr(win_prob, edge_pp).
    # If B-outcome corr > 0.15 AND B-D collinearity < 0.60: flag for B restoration.
    b_weight_analysis: dict = {"status": "insufficient_data", "eligible_legs": 0}
    _FIX_DATE = "2026-05-16"
    try:
        b_rows = db.execute(_text("""
            SELECT mbl.predicted_win_prob, mbl.predicted_edge_pp,
                   CASE WHEN mbl.leg_result = 'WIN' THEN 1.0 ELSE 0.0 END AS outcome
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.game_date >= :fix_date
              AND mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mbl.is_alt_line = 0
              AND mb.source IN ('prospective', 'top_picks')
              AND mbl.predicted_win_prob IS NOT NULL
              AND mbl.predicted_edge_pp  IS NOT NULL
        """), {"fix_date": _FIX_DATE}).fetchall()

        n_eligible = len(b_rows)
        b_weight_analysis["eligible_legs"] = n_eligible

        if n_eligible >= 50:
            import numpy as _np
            wp_arr  = _np.array([r[0] for r in b_rows], dtype=float)
            ep_arr  = _np.array([r[1] for r in b_rows], dtype=float)
            out_arr = _np.array([r[2] for r in b_rows], dtype=float)

            def _safe_corr(a, b):
                if _np.std(a) < 1e-8 or _np.std(b) < 1e-8:
                    return 0.0
                return float(_np.corrcoef(a, b)[0, 1])

            b_outcome_corr = round(_safe_corr(wp_arr, out_arr), 4)
            b_d_collinearity = round(_safe_corr(wp_arr, ep_arr), 4)

            if b_outcome_corr > 0.15 and b_d_collinearity < 0.60:
                recommendation_b = "restore_at_10pct"
                b_note = (
                    f"B-outcome corr={b_outcome_corr:.3f} > 0.15 AND "
                    f"B-D collinearity={b_d_collinearity:.3f} < 0.60. "
                    "Restore Component B weight to 10–15%."
                )
            else:
                recommendation_b = "keep_zero"
                b_note = (
                    f"B-outcome corr={b_outcome_corr:.3f} "
                    f"({'below 0.15 threshold' if b_outcome_corr <= 0.15 else 'OK'}), "
                    f"B-D collinearity={b_d_collinearity:.3f} "
                    f"({'too high' if b_d_collinearity >= 0.60 else 'OK'}). "
                    "Keep B at 0%."
                )

            b_weight_analysis = {
                "status":               "analyzed",
                "eligible_legs":        n_eligible,
                "b_outcome_correlation": b_outcome_corr,
                "b_d_collinearity":      b_d_collinearity,
                "recommendation":        recommendation_b,
                "note":                  b_note,
            }
        else:
            b_weight_analysis["status"] = "insufficient_data"
            b_weight_analysis["note"] = (
                f"{n_eligible} post-fix non-alt straight-bet legs settled "
                f"(need 50). Re-check when more resolve."
            )
    except Exception as _be:
        b_weight_analysis = {"status": "error", "error": str(_be)}

    recommendation = None
    if ready:
        if max_reliability > 0.08 and drift_cell:
            recommendation = f"Retrain {drift_cell.split()[0]} submodel — calibration drift {max_reliability:.3f}"
        elif new_legs >= 500:
            recommendation = f"Full retrain with {new_legs} new settled legs available"
        else:
            recommendation = "Retrain recommended — see reasons"

    return {
        "ready":                  ready,
        "reasons":                reasons,
        "recommendation":         recommendation,
        "new_legs_since_train":   new_legs,
        "last_trained_at":        last_trained_at,
        "last_trained_auc":       last_trained_auc,
        "current_brier":          round(current_brier, 4) if current_brier else None,
        "max_calibration_drift":  round(max_reliability, 4),
        "drift_cell":             drift_cell,
        "top_patterns":           top_patterns,
        "b_weight_analysis":      b_weight_analysis,
    }


# ─── Tier summary + activation gate ──────────────────────────────────────────


def get_tier_summary(db: Session) -> dict:
    """
    Return tier breakdown with today's bets + settled stats for 7d and 30d windows.
    Production sources only.
    """
    from sqlalchemy import text as _text
    from zoneinfo import ZoneInfo as _ZI
    _CT_TZ = _ZI("America/Chicago")
    today = datetime.now(_CT_TZ).strftime("%Y-%m-%d")

    result: dict = {}
    for tier in ("T1", "T2", "T3"):
        # Today's bets
        today_row = db.execute(_text("""
            SELECT COUNT(*) as bets, COALESCE(SUM(legs), 0) as legs
            FROM mock_bets
            WHERE bet_tier = :tier AND game_date = :today
              AND source IN ('prospective','prospective_pm','top_picks_page')
        """), {"tier": tier, "today": today}).fetchone()

        settled_windows: dict = {}
        for window_days in (7, 30):
            sr = db.execute(_text("""
                SELECT
                    COUNT(*) as bets,
                    SUM(CASE WHEN status='SETTLED_WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(COALESCE(actual_profit, 0)) as pnl,
                    SUM(amount) as wagered
                FROM mock_bets
                WHERE bet_tier = :tier
                  AND status IN ('SETTLED_WIN','SETTLED_LOSS')
                  AND source IN ('prospective','prospective_pm','top_picks_page')
                  AND game_date >= date(:today, :offset)
            """), {"tier": tier, "today": today,
                   "offset": f"-{window_days} days"}).fetchone()

            bets    = sr[0] or 0
            wins    = sr[1] or 0
            pnl     = round(float(sr[2] or 0), 2)
            wagered = float(sr[3] or 0)
            wr      = round(wins / bets * 100, 1) if bets > 0 else 0.0
            roi     = round(pnl / wagered * 100, 1) if wagered > 0 else 0.0

            settled_windows[f"{window_days}d"] = {
                "bets": bets, "wins": wins, "wr": wr,
                "roi": roi, "pnl": pnl,
            }

        result[tier] = {
            "today_bets": today_row[0] or 0,
            "today_legs": today_row[1] or 0,
            **settled_windows,
        }

    return result


def check_activation_readiness(db: Session) -> dict:
    """
    Check whether the system meets activation criteria for real-money betting.
    11 criteria covering profitability, calibration, consistency, and data quality.
    Returns criteria status dict with overall readiness flag.
    """
    from sqlalchemy import text as _text
    from zoneinfo import ZoneInfo as _ZI
    _CT_TZ = _ZI("America/Chicago")
    today = datetime.now(_CT_TZ).strftime("%Y-%m-%d")

    _PROD = "('prospective','prospective_pm','top_picks_page')"
    criteria: dict = {}

    # ── Shared queries ───────────────────────────────────────────────────────

    # T1 stats over 14 days
    t1_14d = db.execute(_text(f"""
        SELECT
            COUNT(*) as bets,
            SUM(CASE WHEN status='SETTLED_WIN' THEN 1 ELSE 0 END) as wins,
            SUM(COALESCE(actual_profit, 0)) as pnl,
            SUM(amount) as wagered
        FROM mock_bets
        WHERE bet_tier = 'T1'
          AND status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND source IN {_PROD}
          AND game_date >= date(:today, '-14 days')
    """), {"today": today}).fetchone()
    t1_settled = int(t1_14d[0] or 0)
    t1_wins = int(t1_14d[1] or 0)
    t1_pnl = float(t1_14d[2] or 0)
    t1_wagered = float(t1_14d[3] or 0)

    # ── 1. T1 ROI 14d >= 12% ────────────────────────────────────────────────
    t1_roi = round(t1_pnl / t1_wagered * 100, 1) if t1_wagered > 0 else 0.0
    criteria["min_t1_roi_14d"] = {
        "value": t1_roi, "threshold": 12.0,
        "met": t1_roi >= 12.0,
        "detail": f"{t1_roi}% ROI on {t1_settled} settled T1 bets (14d)",
    }

    # ── 2. T1 settled count >= 50 ───────────────────────────────────────────
    criteria["min_t1_bets_settled"] = {
        "value": t1_settled, "threshold": 50,
        "met": t1_settled >= 50,
        "detail": f"{t1_settled}/50 T1 bets settled (14d)",
    }

    # ── 3. T1 win rate >= 55% ───────────────────────────────────────────────
    t1_wr = round(t1_wins / max(t1_settled, 1) * 100, 1)
    criteria["min_t1_wr"] = {
        "value": t1_wr, "threshold": 55.0,
        "met": t1_wr >= 55.0,
        "detail": f"T1 WR = {t1_wr}% over {t1_settled} bets",
    }

    # ── 4. Brier 7d < 0.18 ─────────────────────────────────────────────────
    try:
        brier_val = db.execute(_text(f"""
            SELECT AVG(
              POWER(CAST(mbl.win_prob AS REAL)/100.0 -
                CASE WHEN mbl.leg_result='WIN' THEN 1.0 ELSE 0.0 END, 2))
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source IN {_PROD}
            AND mbl.leg_result IN ('WIN','LOSS')
            AND COALESCE(mbl.excluded_from_metrics, 0) = 0
            AND mbl.win_prob IS NOT NULL
            AND mb.game_date >= date(:today, '-7 days')
        """), {"today": today}).scalar() or 0
        brier_7d = round(float(brier_val), 4)
    except Exception:
        brier_7d = 1.0
    criteria["max_brier_7d"] = {
        "value": brier_7d, "threshold": 0.18,
        "met": brier_7d < 0.18,
        "detail": f"7d production Brier = {brier_7d}",
    }

    # ── 5. Per-leg WR 14d >= 65% ────────────────────────────────────────────
    try:
        leg_row = db.execute(_text(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END) as wins
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source IN {_PROD}
            AND mb.bet_tier = 'T1'
            AND mbl.leg_result IN ('WIN','LOSS')
            AND COALESCE(mbl.excluded_from_metrics, 0) = 0
            AND mb.game_date >= date(:today, '-14 days')
        """), {"today": today}).fetchone()
        leg_total = int(leg_row[0] or 0)
        leg_wins = int(leg_row[1] or 0)
        leg_wr = round(leg_wins / max(leg_total, 1) * 100, 1)
    except Exception:
        leg_wr = 0.0
        leg_total = 0
    criteria["min_per_leg_wr_14d"] = {
        "value": leg_wr, "threshold": 65.0,
        "met": leg_wr >= 65.0,
        "detail": f"T1 leg WR = {leg_wr}% ({leg_total} legs, 14d)",
    }

    # ── 6. Max single-day loss < 30% of that day's wager ────────────────────
    try:
        worst = db.execute(_text(f"""
            SELECT
                game_date,
                ROUND(SUM(COALESCE(actual_profit, 0)) * 100.0
                      / NULLIF(SUM(amount), 0), 1) as day_roi
            FROM mock_bets
            WHERE bet_tier = 'T1'
              AND status IN ('SETTLED_WIN','SETTLED_LOSS')
              AND source IN {_PROD}
              AND game_date >= date(:today, '-14 days')
            GROUP BY game_date
            ORDER BY day_roi ASC
            LIMIT 1
        """), {"today": today}).fetchone()
        worst_day_roi = float(worst[1]) if worst and worst[1] is not None else 0.0
        worst_day_loss_pct = abs(worst_day_roi) if worst_day_roi < 0 else 0.0
    except Exception:
        worst_day_loss_pct = 0.0
    criteria["max_single_day_loss_pct"] = {
        "value": worst_day_loss_pct, "threshold": 30.0,
        "met": worst_day_loss_pct < 30.0,
        "detail": f"Worst single-day loss = {worst_day_loss_pct}% of wager",
    }

    # ── 7. Positive ROI days >= 10 of last 14 ───────────────────────────────
    try:
        daily_rows = db.execute(_text(f"""
            SELECT
                game_date,
                SUM(COALESCE(actual_profit, 0)) as day_pnl
            FROM mock_bets
            WHERE bet_tier = 'T1'
              AND status IN ('SETTLED_WIN','SETTLED_LOSS')
              AND source IN {_PROD}
              AND game_date >= date(:today, '-14 days')
            GROUP BY game_date
        """), {"today": today}).fetchall()
        pos_days = sum(1 for r in daily_rows if float(r[1] or 0) > 0)
        total_days = len(daily_rows)
    except Exception:
        pos_days = 0
        total_days = 0
    criteria["min_positive_roi_days_of_14"] = {
        "value": pos_days, "threshold": 10,
        "met": pos_days >= 10,
        "detail": f"{pos_days}/{total_days} days with positive P&L (14d window)",
    }

    # ── 8. Max consecutive loss days <= 3 ───────────────────────────────────
    try:
        daily_pnls = db.execute(_text(f"""
            SELECT game_date, SUM(COALESCE(actual_profit, 0)) as day_pnl
            FROM mock_bets
            WHERE bet_tier = 'T1'
              AND status IN ('SETTLED_WIN','SETTLED_LOSS')
              AND source IN {_PROD}
              AND game_date >= date(:today, '-14 days')
            GROUP BY game_date
            ORDER BY game_date
        """), {"today": today}).fetchall()
        max_consec_loss = 0
        cur_streak = 0
        for r in daily_pnls:
            if float(r[1] or 0) < 0:
                cur_streak += 1
                max_consec_loss = max(max_consec_loss, cur_streak)
            else:
                cur_streak = 0
    except Exception:
        max_consec_loss = 0
    criteria["max_consecutive_loss_days"] = {
        "value": max_consec_loss, "threshold": 3,
        "met": max_consec_loss <= 3,
        "detail": f"Longest losing streak = {max_consec_loss} days",
    }

    # ── 9. CLV beat rate >= 40% ─────────────────────────────────────────────
    try:
        clv_row = db.execute(_text(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN clv_cents > 0 THEN 1 ELSE 0 END) as beats
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source IN {_PROD}
            AND mb.bet_tier = 'T1'
            AND mbl.clv_cents IS NOT NULL
            AND mb.game_date >= date(:today, '-14 days')
        """), {"today": today}).fetchone()
        clv_total = int(clv_row[0] or 0)
        clv_beats = int(clv_row[1] or 0)
        clv_rate = round(clv_beats / max(clv_total, 1) * 100, 1)
    except Exception:
        clv_rate = 0.0
        clv_total = 0
    criteria["min_clv_beat_rate"] = {
        "value": clv_rate, "threshold": 40.0,
        "met": clv_rate >= 40.0 if clv_total >= 10 else True,  # pass if insufficient data
        "detail": f"CLV beat rate = {clv_rate}% ({clv_total} legs with CLV data)",
    }

    # ── 10. Scout coverage >= 60% of T1 legs ────────────────────────────────
    try:
        sc_row = db.execute(_text(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN mbl.scout_grade IS NOT NULL THEN 1 ELSE 0 END) as scouted
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source IN {_PROD}
            AND mb.bet_tier = 'T1'
            AND mbl.leg_result IN ('WIN','LOSS')
            AND COALESCE(mbl.excluded_from_metrics, 0) = 0
            AND mb.game_date >= date(:today, '-14 days')
        """), {"today": today}).fetchone()
        sc_total = int(sc_row[0] or 0)
        sc_scouted = int(sc_row[1] or 0)
        sc_coverage = round(sc_scouted / max(sc_total, 1) * 100, 1)
    except Exception:
        sc_coverage = 0.0
        sc_total = 0
    criteria["min_scout_coverage"] = {
        "value": sc_coverage, "threshold": 60.0,
        "met": sc_coverage >= 60.0,
        "detail": f"Scout coverage = {sc_coverage}% ({sc_scouted}/{sc_total} T1 legs)",
    }

    # ── 11. Reliability (calibration error) 7d < 0.02 ───────────────────────
    #    Reliability = weighted avg of (bin_mean_prob - bin_actual_wr)^2
    #    Computed from raw legs in 10 bins (0-10%, 10-20%, ..., 90-100%).
    try:
        cal_rows = db.execute(_text(f"""
            SELECT
                CAST(CAST(mbl.win_prob AS REAL) / 10 AS INTEGER) as bin,
                COUNT(*) as n,
                AVG(CAST(mbl.win_prob AS REAL) / 100.0) as mean_prob,
                AVG(CASE WHEN mbl.leg_result='WIN' THEN 1.0 ELSE 0.0 END) as actual_wr
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source IN {_PROD}
            AND mbl.leg_result IN ('WIN','LOSS')
            AND COALESCE(mbl.excluded_from_metrics, 0) = 0
            AND mbl.win_prob IS NOT NULL
            AND mb.game_date >= date(:today, '-7 days')
            GROUP BY bin
            HAVING COUNT(*) >= 5
        """), {"today": today}).fetchall()
        if cal_rows:
            total_n = sum(int(r[1]) for r in cal_rows)
            reliability_7d = round(sum(
                int(r[1]) * (float(r[2]) - float(r[3])) ** 2
                for r in cal_rows
            ) / max(total_n, 1), 4)
        else:
            reliability_7d = None
    except Exception:
        reliability_7d = None
    criteria["max_reliability_7d"] = {
        "value": reliability_7d, "threshold": 0.02,
        "met": reliability_7d is not None and reliability_7d < 0.02,
        "detail": f"7d calibration error = {reliability_7d}" if reliability_7d is not None
                  else "Insufficient data for reliability calculation",
    }

    # ── Summary ─────────────────────────────────────────────────────────────
    met_count = sum(c["met"] for c in criteria.values())
    all_met = met_count == len(criteria)
    blocking = [k for k, v in criteria.items() if not v["met"]]

    # Rough readiness estimate
    if all_met:
        est_ready = today
    elif met_count >= 9:
        est_ready = "~1-2 weeks"
    elif met_count >= 7:
        est_ready = "~3-4 weeks"
    else:
        est_ready = "~6+ weeks (multiple criteria far from threshold)"

    return {
        "ready": all_met,
        "criteria_met": met_count,
        "criteria_total": len(criteria),
        "blocking": blocking,
        "estimated_ready": est_ready,
        "criteria": criteria,
    }


# ─── Change 3: Rolling Brier windows ─────────────────────────────────────────

def get_brier_windows(db: Session, windows: list[int] | None = None) -> dict:
    """
    Return Brier decomposition for multiple rolling windows in one call.

    Args:
        windows: list of day counts, e.g. [7, 14, 30]. Defaults to [7, 14, 30].

    Returns:
        {
          "7d":  {brier, reliability, resolution, uncertainty, n_legs, quality_tier},
          "14d": {...},
          "30d": {...},
          "trajectory": "improving" | "stable" | "degrading" | "volatile"
        }
    """
    if windows is None:
        windows = [7, 14, 30]

    result: dict = {}
    brier_by_window: dict[int, float] = {}

    for days in sorted(windows):
        try:
            decomp   = get_brier_decomposition(db, days=days)
            overall  = decomp.get("overall", {})
            b        = overall.get("brier")
            brier_by_window[days] = b
            result[f"{days}d"] = {
                "brier":       round(b, 4) if b is not None else None,
                "reliability": round(overall.get("reliability", 0), 4),
                "resolution":  round(overall.get("resolution",  0), 4),
                "uncertainty": round(overall.get("uncertainty", 0), 4),
                "n_legs":      overall.get("n_legs", 0),
                "quality_tier": overall.get("quality_tier", "unknown"),
            }
        except Exception as exc:
            result[f"{days}d"] = {"error": str(exc)}

    # Trajectory: compare shortest vs longest window
    min_w = min(windows)
    max_w = max(windows)
    b_short = brier_by_window.get(min_w)
    b_long  = brier_by_window.get(max_w)
    if b_short is not None and b_long is not None:
        diff = b_short - b_long   # negative = recent is better = improving
        if diff < -0.02:
            trajectory = "improving"     # recent < historical
        elif diff > 0.02:
            trajectory = "degrading"     # recent > historical
        elif len(windows) >= 3:
            # Check for volatility: middle window out of range
            mid_w = sorted(windows)[len(windows) // 2]
            b_mid = brier_by_window.get(mid_w)
            if b_mid is not None and abs(b_mid - (b_short + b_long) / 2) > 0.015:
                trajectory = "volatile"
            else:
                trajectory = "stable"
        else:
            trajectory = "stable"
    else:
        trajectory = "unknown"

    result["trajectory"] = trajectory
    result["windows"]    = sorted(windows)

    return result


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
