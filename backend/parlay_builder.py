"""
parlay_builder.py — Phase 2A: Parlay builder and EV optimizer.

Core functions:
  - get_available_legs(db)        All scoreable legs from current fixtures
  - score_leg(leg)                Model EV for a single leg
  - build_parlay(legs)            Combine N legs into a parlay with true EV
  - optimize_parlays(db, n, size) Find the best EV combos from available legs
  - correlation_check(legs)       Warn on correlated / same-game legs
"""
from __future__ import annotations
import math
import itertools
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from database import Fixture

# Sport key → human label
SPORT_LABEL = {
    "basketball_nba":             "NBA",
    "basketball_ncaab":           "NCAAB",
    "americanfootball_nfl":       "NFL",
    "americanfootball_ncaaf":     "NCAAF",
    "baseball_mlb":               "MLB",
    "soccer_epl":                 "EPL",
    "soccer_spain_la_liga":       "La Liga",
    "soccer_usa_mls":             "MLS",
    "soccer_uefa_champs_league":  "UCL",
    "soccer_germany_bundesliga":  "Bundesliga",
    "soccer_france_ligue_one":    "Ligue 1",
    "soccer_italy_serie_a":       "Serie A",
    "soccer_uefa_europa_league":  "Europa League",
    "icehockey_nhl":              "NHL",
    "tennis_atp_french_open":     "Tennis",
    "mma_mixed_martial_arts":     "MMA",
}

MARKET_LABEL = {
    "h2h":     "Moneyline",
    "spreads": "Spread",
    "totals":  "Total",
}

# ─── Leg extraction ───────────────────────────────────────────────────────────

def get_available_legs(db: Session, markets: list[str] = None) -> list[dict]:
    """
    Parse all stored fixtures into individual scoreable legs.
    Each leg = one pick on one market for one fixture.
    Only returns fixtures with commence_time TODAY (CT timezone) that haven't started yet.

    The fixture table may hold a 7-day API window. The CT-date gate ensures only
    today's games enter the pool — no leakage of tomorrow's or next-week's fixtures
    into picks that would be impossible to settle today.
    """
    markets = markets or ["h2h", "spreads", "totals"]
    fixtures = db.query(Fixture).order_by(Fixture.commence_time).all()
    legs: list[dict] = []

    now = datetime.now(timezone.utc)
    from datetime import timedelta as _td
    from zoneinfo import ZoneInfo as _ZI
    _CT_TZ    = _ZI("America/Chicago")
    _now_ct   = datetime.now(_CT_TZ)
    # Betting-day cutoff: 4 AM CT defines the slate boundary.
    # Games before next 4 AM CT are "today's slate" — includes West Coast games
    # that start near midnight CT (10 PM PT) and run past midnight.
    _4am_base = _now_ct.replace(hour=4, minute=0, second=0, microsecond=0)
    if _now_ct >= _4am_base:
        _slate_end_ct = _4am_base + _td(days=1)   # next 4 AM CT (tomorrow morning)
    else:
        _slate_end_ct = _4am_base                  # today's 4 AM CT (early morning run)
    _slate_end_utc = _slate_end_ct.astimezone(timezone.utc)

    # ── EMERGENCY PATCH 2026-05-03 ─────────────────────���─────────────────────────
    # Morning generation missed the early-slate MLB games. Allow recently-started
    # games to enter the pool so the 12:30 PM PT slate is not lost.
    # After tonight's settlement: set EMERGENCY_INCLUDE_STARTED = False.
    # Long-term fix: 11 AM CT safety check (see scheduler) regenerates picks
    # before first pitch when pick count is below expected threshold.
    # Emergency override: allow recently-started games to re-enter the pool.
    # Set True only on days where the morning generation missed early-slate games.
    # MUST be reverted to False the same day after settlement.
    EMERGENCY_INCLUDE_STARTED = False
    EMERGENCY_WINDOW_HOURS    = 2

    # Bookmakers-freshness threshold: reject fixtures where all bookmaker data is
    # older than 48 hours. TheOddsAPI pre-populates phantom future playoff games
    # (series games 4-7 before they're confirmed) with placeholder odds from the
    # series open — these have stale last_update timestamps and must never enter
    # the leg pool. A real game for tonight will always have been refreshed within
    # the last ~24 hours by the scheduler's odds fetch.
    _FRESHNESS_HOURS = 48
    _freshness_cutoff = now - _td(hours=_FRESHNESS_HOURS)

    import json as _json

    for fix in fixtures:
        if not fix.bookmakers:
            continue

        # ── Bookmakers freshness gate ──────────────────────────────────────────
        # Skip fixtures where ALL bookmakers have stale last_update (> 48h old).
        # Phantom pre-populated playoff games have last_update from series open.
        try:
            bms = fix.bookmakers if isinstance(fix.bookmakers, list) else _json.loads(fix.bookmakers or '[]')
            if bms:
                latest_ts = None
                for bm in bms:
                    lu = bm.get("last_update", "")
                    if lu:
                        try:
                            ts = datetime.fromisoformat(lu.replace("Z", "+00:00"))
                            if latest_ts is None or ts > latest_ts:
                                latest_ts = ts
                        except Exception:
                            pass
                if latest_ts is not None and latest_ts < _freshness_cutoff:
                    continue  # all bookmaker data is stale — phantom fixture
        except Exception:
            pass  # if we can't parse bookmakers, let the time gate handle it

        # ── Time gate: today-only (CT) + not yet started ───────────────────────
        live_bet_patch = False
        if fix.commence_time:
            ct = fix.commence_time
            if ct.tzinfo is None:
                from datetime import timezone as tz
                ct = ct.replace(tzinfo=tz.utc)

            # Slate cutoff: reject fixtures at or after next 4 AM CT.
            # Prevents the 7-day API window from leaking future fixtures into picks
            # while correctly including West Coast games that start near midnight CT.
            if ct >= _slate_end_utc:
                continue

            if ct > now:
                pass  # today's game, not yet started — include normally
            elif EMERGENCY_INCLUDE_STARTED:
                hours_since = (now - ct).total_seconds() / 3600
                if hours_since > EMERGENCY_WINDOW_HOURS:
                    continue
                live_bet_patch = True
            else:
                continue  # today's game already started — skip

        sport_label = SPORT_LABEL.get(fix.sport_key, fix.sport_title or fix.sport_key)
        game_label  = f"{fix.away_team} @ {fix.home_team}"
        game_time   = fix.commence_time.isoformat() if fix.commence_time else None

        # Collect best odds per market+outcome across all bookmakers
        best: dict[tuple, float] = {}  # (market, outcome_name, point) -> best_price
        for bk in fix.bookmakers:
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in markets:
                    continue
                for outcome in mkt.get("outcomes", []):
                    name  = outcome.get("name", "")
                    price = float(outcome.get("price", 0))
                    point = outcome.get("point")        # spread/total value
                    key   = (mkt_key, name, point)
                    if key not in best or price > best[key]:
                        best[key] = price

        for (mkt_key, name, point), price in best.items():
            if price <= 1.0:
                continue

            # For recently-started games (live_bet_patch), exclude h2h legs with extreme
            # odds that indicate live in-game lines (e.g. Marlins +3300 down 8-0).
            # Normal pregame MLB underdogs are ≤ +700 (8.0 decimal). Cap at 9.0 to allow
            # genuine big underdogs while blocking obviously live odds.
            if live_bet_patch and mkt_key == "h2h" and price > 9.0:
                continue

            # Soccer totals: only allow 0.5-increment lines in range 1.5–4.5
            if mkt_key == "totals" and fix.sport_key.startswith("soccer_") and point is not None:
                pt = float(point)
                if abs(pt % 0.5) >= 0.01:   # non-0.5-step (e.g. 2.25, 2.75)
                    continue
                if pt < 1.5 or pt > 4.5:    # outside valid FanDuel range
                    continue
            # Soccer spreads: only allow 0.5-increment lines (Asian handicap lines
            # like +1.25, +2.75 are not available on FanDuel)
            if mkt_key == "spreads" and fix.sport_key.startswith("soccer_") and point is not None:
                if abs(float(point) % 0.5) >= 0.01:
                    continue

            # Build a readable description
            mkt_label  = MARKET_LABEL.get(mkt_key, mkt_key)
            if mkt_key == "spreads" and point is not None:
                sign = "+" if float(point) > 0 else ""
                desc = f"{name} {sign}{point} ({mkt_label})"
            elif mkt_key == "totals" and point is not None:
                desc = f"{name} {point} ({mkt_label})"
            elif name.lower() == "draw":
                # Soccer 3-way market — include game context so display isn't bare "Draw"
                desc = f"Draw — {fix.away_team} vs {fix.home_team} ({mkt_label})"
            else:
                desc = f"{name} ({mkt_label})"

            leg_entry = {
                "leg_id":       f"{fix.id}:{mkt_key}:{name}:{point}",
                "fixture_id":   fix.id,
                "game":         game_label,
                "sport":        sport_label,
                "sport_key":    fix.sport_key,
                "market":       mkt_key,
                "market_label": mkt_label,
                "pick":         name,
                "point":        point,
                "description":  desc,
                "odds":         round(price, 3),
                "implied_prob": round(1 / price * 100, 2),
                "game_time":    game_time,
                "home_team":    fix.home_team,
                "away_team":    fix.away_team,
            }
            if live_bet_patch:
                leg_entry["live_bet_patch"] = True
            legs.append(leg_entry)

    return legs


# ─── Single-leg scoring ───────────────────────────────────────────────────────

def score_leg(leg: dict) -> dict:
    """
    Add model win_prob and EV to a leg dict.
    Returns the leg with prediction fields added.

    For spread (ATS) legs on sports with a trained diagnostic sub-model
    (currently NHL, MLB), the game-level rolling-stats model is used instead
    of the global bet-slip model.  All other legs use the global model.

    For Soccer legs, the market-specific sub-model is used if available
    (soccer_total_v1 for Over/Under, soccer_ml_v1 for Moneyline).
    Falls back to the global combined_v1 model if sub-model unavailable.
    """
    from ml_model import load_model, FEATURE_COLS, SPORT_MAP, LEAGUE_MAP, predict_game_ats, score_soccer_leg
    import numpy as np

    try:
        clf, scaler, imputer = load_model()
    except Exception as _load_err:
        # Graceful fallback when the pickled model was saved with an incompatible
        # numpy version (e.g. numpy._core vs numpy.core). Picks are still generated
        # using implied probability from odds; sub-models (ATS/soccer) are unaffected.
        print(f"[score_leg] WARNING: global model load failed ({_load_err}). Falling back to implied prob.")
        clf, scaler, imputer = None, None, None
    dec_odds   = leg["odds"]
    imp_prob   = 1 / dec_odds
    mkt        = leg.get("market", "")
    sport_raw  = leg.get("sport", "")
    sport_norm = _normalize_sport(sport_raw)
    model_used = "global"

    # ── ATS sub-model path (NHL / MLB spreads) ────────────────────────────────
    # _normalize_sport maps "MLB" → "Baseball" and "NHL" → "Ice Hockey", so we
    # must check both the raw label and the normalized form.
    _ATS_RAW  = {"NHL", "MLB"}
    _ATS_NORM = {"Ice Hockey", "Baseball"}
    _ATS_CANONICAL = {"MLB": "mlb_ats_v1", "NHL": "nhl_ats_v1",
                      "Baseball": "mlb_ats_v1", "Ice Hockey": "nhl_ats_v1"}
    # Map to canonical "MLB"/"NHL" for predict_game_ats routing
    _TO_CANONICAL = {"Baseball": "MLB", "Ice Hockey": "NHL"}
    sport_canonical = _TO_CANONICAL.get(sport_norm, sport_raw)

    if mkt in ("spreads", "alternate_spreads") and (
        sport_raw in _ATS_RAW or sport_norm in _ATS_NORM
    ):
        ats_prob = predict_game_ats(
            sport      = sport_canonical,
            home_team  = leg.get("home_team", ""),
            away_team  = leg.get("away_team", ""),
            game_date  = leg.get("game_date", ""),
        )
        if ats_prob is not None:
            # ats_prob = probability the HOME TEAM covers the spread.
            # Invert for away-team legs so each side gets the correct probability.
            pick = leg.get("pick", "").lower().strip()
            away = leg.get("away_team", "").lower().strip()
            win_prob   = (1.0 - ats_prob) if (pick and away and pick == away) else ats_prob

            # Post-hoc calibration correction for MLB run line:
            # Retrain (Apr 2026) showed mlb_ats_v1 is overconfident by 3.8pp in the
            # 55–65% confidence band (predicted 57.6%, actual 53.8%).  Apply a flat
            # correction to keep Component B EV scores from inflating MLB spread picks.
            if sport_canonical == "MLB" and 0.55 <= win_prob <= 0.65:
                win_prob = max(0.50, win_prob - 0.038)

            ev         = win_prob * (dec_odds - 1) - (1 - win_prob)
            edge       = win_prob - imp_prob
            model_used = _ATS_CANONICAL.get(sport_raw,
                         _ATS_CANONICAL.get(sport_norm, "combined_v1"))
            return {
                **leg,
                "win_prob":    round(win_prob * 100, 2),
                "ev":          round(ev, 4),
                "ev_pct":      round(ev * 100, 2),
                "edge":        round(edge * 100, 2),
                "grade":       _grade(ev, edge),
                "model_used":  model_used,
            }

    # ── Soccer sub-model path ─────────────────────────────────────────────────
    _SOCCER_SPORTS = {"Soccer", "soccer", "EPL", "La Liga", "Ligue 1", "Bundesliga",
                      "Serie A", "UCL", "Europa League"}
    if sport_raw in _SOCCER_SPORTS or sport_norm == "Soccer":
        soccer_result = score_soccer_leg(leg)
        if soccer_result is not None:
            win_prob = soccer_result["win_prob"] / 100
            ev       = soccer_result["ev"]
            edge     = win_prob - imp_prob
            return {
                **leg,
                "win_prob":   soccer_result["win_prob"],
                "ev":         round(ev, 4),
                "ev_pct":     round(ev * 100, 2),
                "edge":       round(edge * 100, 2),
                "grade":      _grade(ev, edge),
                "model_used": soccer_result["model_used"],
            }
        # Sub-model unavailable or no form data → fall through to global model

    # ── Global model path ─────────────────────────────────────────────────────
    if clf is None:
        win_prob = imp_prob
        ev       = 0.0
    else:
        sport_id  = SPORT_MAP.get(sport_norm, 6)
        league_id = LEAGUE_MAP.get(leg.get("sport", ""), 10)

        feature_dict = {
            "legs":         1,
            "odds":         dec_odds,
            "log_odds":     math.log(dec_odds),
            "implied_prob": imp_prob,
            "stake":        10.0,
            "is_parlay":    0,
            "sport_id":     sport_id,
            "league_id":    league_id,
            "hour_placed":  datetime.now().hour,
            "day_of_week":  datetime.now().weekday(),
            "ml_pct":       1.0 if mkt == "h2h"     else 0.0,
            "spread_pct":   1.0 if mkt == "spreads"  else 0.0,
            "total_pct":    1.0 if mkt == "totals"   else 0.0,
            "prop_pct":     0.0,
            "multi_sport":  0,
            "n_sports":     1,
            "has_ev":       0,
            "ev_value":     0.0,
            "closing_line_diff": 0.0,
        }
        # Use np.nan (not 0.0) for missing features so the imputer fills median.
        row    = np.array([[feature_dict.get(c, np.nan) for c in FEATURE_COLS]], dtype=float)
        if imputer is not None:
            row = imputer.transform(row)
        row_s  = scaler.transform(row)
        win_prob = float(clf.predict_proba(row_s)[0][1])
        ev       = win_prob * (dec_odds - 1) - (1 - win_prob)

    edge = win_prob - imp_prob

    return {
        **leg,
        "win_prob":    round(win_prob * 100, 2),
        "ev":          round(ev, 4),
        "ev_pct":      round(ev * 100, 2),
        "edge":        round(edge * 100, 2),
        "grade":       _grade(ev, edge),
        "model_used":  model_used,
    }


def _normalize_sport(label: str) -> str:
    mapping = {
        "NBA": "Basketball", "NCAAB": "Basketball",
        "NFL": "American Football", "NCAAF": "American Football",
        "MLB": "Baseball",
        "EPL": "Soccer", "La Liga": "Soccer", "MLS": "Soccer", "UCL": "Soccer",
        "Bundesliga": "Soccer", "Ligue 1": "Soccer", "Serie A": "Soccer",
        "Europa League": "Soccer",
        "NHL": "Ice Hockey",
        "Tennis": "Tennis", "MMA": "other",
    }
    return mapping.get(label, label)


def _grade(ev: float, edge: float) -> str:
    if ev > 0.08 and edge > 0.03:  return "A"
    if ev > 0.04 and edge > 0.01:  return "B"
    if ev > 0.0:                   return "C"
    if ev > -0.05:                 return "D"
    return "F"


# ─── Alternative Line Evaluation (ALE) ───────────────────────────────────────

def _classify_line(market_key: str) -> str:
    """Map a market key to a human-readable line type for ALE deduplication."""
    if market_key == "h2h":               return "moneyline"
    if market_key == "spreads":            return "spread"
    if market_key == "alternate_spreads":  return "alt_spread"
    if market_key == "totals":             return "total"
    if market_key == "alternate_totals":   return "alt_total"
    return "other"


def _pick_direction(pick: str) -> str:
    """
    Normalise a pick label to its directional key for CLOSE→CUSHION matching.
    Over/Under → direction token; team names → lowercased name.
    Used to match a CLOSE leg with a CUSHION alternative on the same game.
    """
    p = (pick or "").strip().lower()
    if p in ("over", "under"):
        return p
    return p


# LOS weights and scale — tunable without code change
# Standard (non-CUSHION) weights:
_LOS_WIN_W      = 0.50   # win probability (dominant — win first, EV second)
_LOS_EDGE_W     = 0.30   # normalised edge vs market
_LOS_LQS_W      = 0.20   # LQS signal quality
# CUSHION-grade overrides (personal history validates this market — win_prob primary):
_LOS_WIN_W_C    = 0.60   # CUSHION: win_prob weight (personal WR confirms quality)
_LOS_EDGE_W_C   = 0.15   # CUSHION: edge is secondary (CUSHION markets price efficiently)
_LOS_LQS_W_C    = 0.25   # CUSHION: LQS gets more weight (quality signal more reliable)
_LOS_EDGE_SCALE = 15.0   # edge of 15pp maps to norm_edge = 1.0
_LOS_MIN        = 0.52   # legs below this floor are excluded from ALE candidates

_SOCCER_SPORT_TOKENS = frozenset([
    "soccer", "ucl", "epl", "la liga", "bundesliga",
    "serie a", "ligue 1", "champions", "europa",
])


def _is_valid_soccer_line(sport: str, market_key: str, point) -> bool:
    """
    Soccer totals and spreads must use exactly x.5-increment lines (-0.5, 1.5, 2.5 …).
    Integers (0, 1, 2, 3) and quarter-goal Asian-handicap lines (2.25, 2.75, +1.25 …)
    are not available on FanDuel and are excluded.
    Non-soccer sports and non-total/spread markets always pass through.
    """
    sport_lower = (sport or "").lower()
    is_soccer  = any(tok in sport_lower for tok in _SOCCER_SPORT_TOKENS)
    mkt_lower  = (market_key or "").lower()
    is_total   = "total"  in mkt_lower
    is_spread  = "spread" in mkt_lower
    if is_soccer and (is_total or is_spread) and point is not None:
        try:
            # % 1.0 gives 0.0 for integers, 0.5 for x.5 lines (Python floor mod)
            return abs(float(point) % 1.0 - 0.5) < 0.01   # must be exactly x.5
        except (TypeError, ValueError):
            pass
    return True


_MLB_SPORT_KEYS = frozenset({"baseball_mlb", "baseball_mlb_preseason"})


def _is_allowed_mlb_leg(sport_key: str, market_key: str, point, win_prob) -> bool:
    """
    DEPRECATED — restriction removed 2026-04-29.
    ALE + sim WR gate + margin grades handle MLB line selection.
    All MLB markets (ML, spreads -1.5/+1.5, totals) are now evaluated by ALE.
    The sim WR gate (×0.70 penalty at <20% WR) penalises -1.5 naturally.
    """
    return True


def _sim_win_rate(db, sport: str, market_type: str, team: str,
                  since: str = "2026-04-22") -> tuple:
    """
    Query mock_bet_legs for historical win rate of a sport/market/team combination.
    Returns (win_rate, n_settled) where win_rate is None when n_settled < 5.
    """
    try:
        import sqlite3 as _sql3
        from database import DB_PATH as _DB_PATH
        con = _sql3.connect(_DB_PATH)
        row = con.execute("""
            SELECT
                AVG(CASE WHEN mbl.leg_result = 'WIN' THEN 1.0 ELSE 0.0 END),
                COUNT(*)
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
            WHERE mbl.sport = ?
              AND mbl.market_type = ?
              AND mbl.description LIKE ?
              AND mb.generated_at >= ?
              AND mbl.leg_result IS NOT NULL
              AND mbl.leg_result != 'PUSH'
        """, (sport, market_type, f"%{team}%", since)).fetchone()
        con.close()
        if row and row[1] and row[1] >= 5:
            return float(row[0]), int(row[1])
        return None, 0
    except Exception:
        return None, 0


def _compute_los(leg: dict, margin_grade: str = None) -> float:
    """
    Line Optimization Score — composite ranking used by ALE pre-assembly selection.

    Inputs (already on the scored leg dict):
        win_prob  — model win probability (0–100 %)
        edge      — edge in percentage points vs implied prob
        lqs       — Leg Quality Score (0–100)
        margin_grade (optional) — personal profile grade; CUSHION uses different weights

    For alt lines edge=0 (win_prob == implied_prob); their LOS advantage comes
    from high implied probability on high-probability cushion lines + LQS.

    CUSHION grade: win_prob is the primary signal (personal WR confirms reliability);
    edge is secondary because CUSHION markets often have efficient pricing (-400 etc.).
    For a -400 (83% implied, 0 edge, lqs=70):
      Standard:  0.83×0.50 + 0×0.30 + 0.70×0.20 = 0.415 + 0.14 = 0.555
      CUSHION:   0.83×0.60 + 0×0.15 + 0.70×0.25 = 0.498 + 0.175 = 0.673 ✓ (after ×1.10 → 0.74)
    """
    wp        = (leg.get("win_prob") or 0.0) / 100.0
    edge_pp   = (leg.get("edge")     or 0.0)
    lqs       = (leg.get("lqs")      or 0.0) / 100.0
    norm_edge = max(0.0, min(edge_pp, 15.0)) / _LOS_EDGE_SCALE
    if margin_grade in ("CUSHION", "NEUTRAL"):
        # CUSHION: personal WR validates this market; win_prob is primary signal.
        # NEUTRAL: no personal data — evaluate on model confidence alone (edge ≈ 0
        #          for alt spread lines, so standard formula undersells win_prob).
        # Both use win_prob-dominant weights to avoid penalising zero-edge legs.
        return round(wp * _LOS_WIN_W_C + norm_edge * _LOS_EDGE_W_C + lqs * _LOS_LQS_W_C, 4)
    return round(wp * _LOS_WIN_W + norm_edge * _LOS_EDGE_W + lqs * _LOS_LQS_W, 4)


# Markets extracted from fixture.bookmakers JSON for ALE scoring
_ALE_BK_MARKETS = {"h2h", "spreads", "totals", "alternate_spreads", "alternate_totals"}

# Zone filter for alt lines (replaces the old hard 1.25–3.50 decimal range).
# Zone A: implied >= 55% (decimal <= ~1.82, odds <= -122) — CUSHION/high-confidence range.
#          Allows heavy favorites like -400/-500 that the 1.25 min previously blocked.
# Zone B: implied 42–55% (decimal ~1.82–2.38, odds -122 to +138) — value-spread range;
#          LOS filter gates further by model confidence.
# Exclude: implied < 42% (decimal > 2.38, odds > +138) — pure longshot territory.
_ALE_ALT_MIN_IMPLIED = 42.0   # % — legs below this are excluded regardless of model signal

# ── Boost step-down constants ──────────────────────────────────────────────────
# When a boost is active, ALE prefers safer lines. The step-down defines how
# many handicap units toward the main line to prefer for each boost tier.
# +25% is pure EV addition (no step-down needed — CUSHION legs already optimal).
# +30%/+50% should target safer lines to restore correct EV.
_BOOST_STEP_DOWN: dict[float, float] = {
    0.25: 0.0,   # +25%: keep same CUSHION legs — boost is pure EV addition
    0.30: 1.0,   # +30%: step 1.0 unit closer to main line (safer spread/total)
    0.50: 1.5,   # +50%: step 1.5 units closer (much safer legs)
}


def _boost_adjusted_target_odds(target_decimal: float, boost_pct: float) -> float:
    """
    Given that we want equivalent payout to target_decimal after a profit boost,
    what base (unboosted) odds can we accept?

    Derivation:
        boosted_profit = base_profit * (1 + boost_pct)
        base_profit = (base_odds - 1) * stake
        We want boosted_profit / stake = (target_decimal - 1)
        → (base_odds - 1) * (1 + boost_pct) = target_decimal - 1
        → base_odds = 1 + (target_decimal - 1) / (1 + boost_pct)
    """
    if boost_pct <= 0:
        return target_decimal
    return round(1.0 + (target_decimal - 1.0) / (1.0 + boost_pct), 4)


def _boost_los_multiplier(implied_prob: float, boost_pct: float) -> float:
    """
    Safety premium applied to LOS when a boost is active.

    +25%: no change (pure EV addition, same legs).
    +30%/+50%: reward legs with higher implied_prob (safer) so ALE ranks them
    higher, implementing the step-down preference for safer lines.

    At 80% implied → full premium; at 50% implied → zero premium; below 50% → 0.
    """
    if boost_pct < 0.30:
        return 1.0
    # safety ∈ [0, 1] rising from 50% to 80% implied
    safety = max(0.0, min(1.0, (implied_prob - 50.0) / 30.0))
    return 1.0 + boost_pct * safety * 0.60


def evaluate_alt_lines(
    fix,                        # Fixture ORM object
    main_legs: list[dict],      # legs already scored via score_leg() for this fixture
    alt_lines_raw: list[dict],  # raw rows from alt_lines table (ct.get_scored_alt_lines)
    db,                         # SQLAlchemy Session (for LQS computation)
    boost_pct: float = 0.0,     # active boost tier — triggers safety-premium re-ranking
) -> list[dict]:
    """
    Score ALL available lines for a single fixture and rank by LOS.

    Candidate sources:
    1. Pre-scored main-market legs (h2h, spreads, totals) — already have win_prob
       and edge from score_leg(); only LQS and LOS are added here.
    2. Lines from fix.bookmakers for alternate_spreads / alternate_totals markets
       (if present — depends on what TheOddsAPI returned for this fetch).
    3. Rows from alt_lines table (passed in as alt_lines_raw) — scored with
       implied_prob as win_prob since the global ML model has no alt-line signal.

    Each candidate is scored:
        los = win_prob/100 * 0.50
            + max(0, min(edge_pp, 15)) / 15 * 0.30
            + lqs/100 * 0.20

    Returns: legs sorted by LOS descending, each with los, line_type, lqs added.
    Lines below _LOS_MIN (0.52) are excluded.

    No database writes.  No lookahead — uses only data available at selection time.
    """
    import leg_quality as lq_mod

    sport_label = SPORT_LABEL.get(fix.sport_key, fix.sport_title or fix.sport_key)
    game_label  = f"{fix.away_team} @ {fix.home_team}"
    game_time   = fix.commence_time.isoformat() if fix.commence_time else None

    def _lqs_for(leg: dict) -> float:
        """Compute LQS for a leg dict; returns 0.0 on any error."""
        is_home   = leg.get("pick") == fix.home_team
        opponent  = fix.away_team if is_home else fix.home_team
        try:
            r = lq_mod.compute_leg_quality_score({
                "market_type":      leg.get("market_label") or leg.get("market"),
                "sport":            leg.get("sport"),
                "team_or_player":   leg.get("pick"),
                "model_confidence": leg.get("win_prob"),
                "model_used":       leg.get("model_used"),
                "edge_pp":          leg.get("edge"),
                "line":             leg.get("point"),
                "is_home":          is_home,
                "opponent":         opponent,
            }, db)
            return float(r.get("lqs") or 0.0)
        except Exception:
            return 0.0

    candidates: list[dict] = []

    # ── Profile-grade lookup helper (cached per function call) ─────────────────
    # Used by Sources 1–3 to pass the correct margin_grade to _compute_los, enabling
    # CUSHION-aware LOS weights (win_prob 0.60 vs 0.50) for confirmed CUSHION legs.
    _grade_cache: dict[tuple, str | None] = {}
    def _pep_grade(sport: str, market_label: str, point) -> str | None:
        key = (sport, market_label, point)
        if key in _grade_cache:
            return _grade_cache[key]
        try:
            from personal_edge_profile import (
                normalize_sport        as _pns,
                normalize_market       as _pnm,
                classify_line_bucket   as _pcb,
                lookup_personal_profile as _plp,
            )
            prof = _plp(_pns(sport), _pnm(market_label), _pcb(_pnm(market_label), "", point))
            grade = prof.get("margin_grade") if prof else None
        except Exception:
            grade = None
        _grade_cache[key] = grade
        return grade

    # ── Source 1: pre-scored main-market legs ─────────────────────────────────
    seen_leg_ids: set[str] = set()
    for leg in main_legs:
        lqs = leg.get("lqs")
        if lqs is None:
            lqs = _lqs_for(leg)
        _s1_grade = _pep_grade(leg.get("sport", ""), leg.get("market_label") or leg.get("market") or "", leg.get("point"))
        candidates.append({
            **leg,
            "lqs":       lqs,
            "los":       _compute_los({**leg, "lqs": lqs}, margin_grade=_s1_grade),
            "line_type": _classify_line(leg.get("market") or ""),
        })
        seen_leg_ids.add(leg.get("leg_id", ""))

    # ── Source 2: alternate spreads/totals from bookmakers JSON ───────────────
    if fix.bookmakers:
        best_bk: dict[tuple, float] = {}
        for bk in fix.bookmakers:
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in ("alternate_spreads", "alternate_totals"):
                    continue   # main markets already covered by main_legs
                for outcome in mkt.get("outcomes", []):
                    name  = outcome.get("name", "")
                    price = float(outcome.get("price", 0))
                    point = outcome.get("point")
                    key   = (mkt_key, name, point)
                    if key not in best_bk or price > best_bk[key]:
                        best_bk[key] = price

        for (mkt_key, name, point), price in best_bk.items():
            if price <= 1.0:
                continue
            # Soccer alternate_totals: reject quarter-goal lines (4.25, 2.75, etc.)
            if not _is_valid_soccer_line(sport_label, mkt_key, point):
                continue
            mkt_label  = MARKET_LABEL.get(mkt_key, mkt_key)
            point_str  = (f" {float(point):+g}" if point is not None and float(point) != 0 else "")
            desc       = f"{name}{point_str} ({mkt_label})"
            leg_id     = f"{fix.id}:{mkt_key}:{name}:{point}"
            if leg_id in seen_leg_ids:
                continue
            seen_leg_ids.add(leg_id)

            raw_leg = {
                "leg_id":       leg_id,
                "fixture_id":   fix.id,
                "game":         game_label,
                "sport":        sport_label,
                "sport_key":    fix.sport_key,
                "market":       mkt_key,
                "market_label": mkt_label,
                "pick":         name,
                "point":        point,
                "description":  desc,
                "odds":         round(price, 3),
                "implied_prob": round(1 / price * 100, 2),
                "game_time":    game_time,
                "home_team":    fix.home_team,
                "away_team":    fix.away_team,
                "is_alt_line":  True,
            }
            try:
                sl = score_leg(raw_leg)
            except Exception:
                sl = {**raw_leg, "win_prob": raw_leg["implied_prob"],
                      "ev": 0.0, "edge": 0.0, "grade": "C", "model_used": "implied"}
            # Use model-assigned win_prob/edge directly.
            # MLB/NHL alternate_spreads route through mlb_ats_v1/nhl_ats_v1;
            # other sports fall back to the global model.  Implied-only fallback
            # only applies when score_leg() raises (model_used="implied" above).

            # Zero-edge gate: exclude low-probability legs when no model provides
            # a real edge signal (soccer spreads, global fallback).  Legs above 65%
            # implied are CUSHION-heavy favorites still worth including on LQS alone.
            if abs(sl.get("edge", 0)) < 0.5 and sl.get("implied_prob", 0) < 65.0:
                continue

            lqs = _lqs_for(sl)
            _s2_grade = _pep_grade(sport_label, mkt_label, point)
            los = _compute_los({**sl, "lqs": lqs}, margin_grade=_s2_grade)

            # Sim WR gate: penalise/boost LOS based on historical simulation results
            _swr, _sn = _sim_win_rate(db, sport_label, mkt_key, name)
            if _swr is not None:
                if _swr < 0.20:
                    los = round(los * 0.70, 4)
                elif _swr > 0.40:
                    los = round(los * 1.10, 4)

            candidates.append({
                **sl,
                "lqs":       lqs,
                "los":       los,
                "line_type": _classify_line(mkt_key),
            })

    # ── Source 3: alt_lines table rows ────────────────────────────────────────
    for al in (alt_lines_raw or []):
        al_odds   = float(al.get("odds") or 2.0)
        if al_odds <= 1.0:
            continue
        # Zone-based implied probability filter (replaces hard 1.25–3.50 odds cap):
        #   Zone A (implied >= 55%): CUSHION territory — allows heavy favorites (-400/-500)
        #   Zone B (implied 42–55%): value range — LOS gate handles model-confidence check
        #   Below 42%: pure longshot (odds > +138) — exclude
        _al_implied = (1.0 / al_odds) * 100
        if _al_implied < _ALE_ALT_MIN_IMPLIED:
            continue
        al_market = al.get("market_key", "")
        if al_market not in ("alternate_spreads", "alternate_totals",
                              "spreads", "totals"):
            continue
        # Soccer alternate_totals: reject quarter-goal lines (4.25, 2.75, etc.)
        if not _is_valid_soccer_line(sport_label, al_market, al.get("line")):
            continue
        al_team   = al.get("team", "")
        al_line   = al.get("line")
        mkt_label = "Alt Spread" if "spread" in al_market else "Alt Total"
        point_str = (f" {float(al_line):+g}" if al_line is not None
                     and float(al_line) != 0 else "")
        al_desc   = f"{al_team}{point_str} ({mkt_label})"
        leg_id    = f"{fix.id}:{al_market}:{al_team}:{al_line}:alt"
        if leg_id in seen_leg_ids:
            continue
        seen_leg_ids.add(leg_id)

        raw_leg = {
            "leg_id":       leg_id,
            "fixture_id":   fix.id,
            "game":         game_label,
            "sport":        sport_label,
            "sport_key":    fix.sport_key,
            "market":       al_market,
            "market_label": mkt_label,
            "pick":         al_team,
            "point":        al_line,
            "description":  al_desc,
            "odds":         round(al_odds, 3),
            "implied_prob": round(1 / al_odds * 100, 2),
            "game_time":    game_time,
            "home_team":    fix.home_team,
            "away_team":    fix.away_team,
            "is_alt_line":  True,
        }
        # Score via ML model — use result directly, not implied_prob override.
        # MLB/NHL alternate_spreads route through mlb_ats_v1/nhl_ats_v1 inside
        # score_leg(), giving real model win_prob and edge for alt lines.
        # Implied-only fallback only when score_leg() raises.
        try:
            sl = score_leg(raw_leg)
        except Exception:
            sl = {**raw_leg, "win_prob": raw_leg["implied_prob"],
                  "ev": 0.0, "edge": 0.0, "grade": "C", "model_used": "implied"}

        # Zero-edge gate: exclude low-probability legs when no model provides
        # a real edge signal (soccer spreads, global fallback).
        if abs(sl.get("edge", 0)) < 0.5 and sl.get("implied_prob", 0) < 65.0:
            continue

        lqs = _lqs_for(sl)
        _s3_grade = _pep_grade(sport_label, mkt_label, al_line)
        los = _compute_los({**sl, "lqs": lqs}, margin_grade=_s3_grade)

        # Sim WR gate: penalise/boost LOS based on historical simulation results
        _swr, _sn = _sim_win_rate(db, sport_label, al_market, al_team)
        if _swr is not None:
            if _swr < 0.20:
                los = round(los * 0.70, 4)
            elif _swr > 0.40:
                los = round(los * 1.10, 4)

        candidates.append({
            **sl,
            "lqs":       lqs,
            "los":       los,
            "line_type": _classify_line(al_market),
        })

    # ── Margin-adjusted LOS ───────────────────────────────────────────────────
    # Apply margin grade multipliers based on personal edge profiles.
    #
    # CUSHION ×1.10 — comfortable wins; boost this line's ranking
    # CLOSE   ×0.85 — narrow wins despite good WR; penalise for parlay safety
    # MIXED   ×0.92 — volatile margins; slight penalty
    # AVOID   →0.0  — poor WR or negative margins; remove from consideration
    # No data →×1.0 — pass through unchanged
    #
    # This ensures Soccer DC (88% WR, CLOSE) ranks below Soccer ML (82%, CUSHION)
    # in ALE ordering, and AVOID legs drop out entirely.
    try:
        from personal_edge_profile import (
            get_margin_adjusted_los as _mal_fn,
            normalize_sport         as _nsp_fn,
            normalize_market        as _nmkt_fn,
            classify_line_bucket    as _clf_bkt_fn,
            lookup_personal_profile as _lkp_fn,
        )
        for _c in candidates:
            _c_sport  = _nsp_fn(_c.get("sport", ""))
            _c_mt     = _nmkt_fn(_c.get("market_label") or _c.get("market") or "")
            _c_bucket = _clf_bkt_fn(_c_mt, _c.get("description", ""), _c.get("point"))
            _orig_los = _c["los"]

            # AVOID should not zero-out positive-point alt spread legs.
            # The AVOID grade was set on negative-point (covering-favourite) losses;
            # applying AVOID → 0.0 to underdog +2.5/+3.5 cushion lines removes
            # genuinely high-probability picks (78-80% implied, already gated upstream).
            _skip_avoid = False
            if _c.get("is_alt_line") and _c_mt in ("Alt Spread", "Spread"):
                _pt = _c.get("point")
                if _pt is not None:
                    try:
                        _skip_avoid = float(_pt) > 0
                    except (TypeError, ValueError):
                        pass

            if _skip_avoid:
                _new_los = _orig_los  # neutral — no AVOID/CLOSE penalty for cushion underdog lines
            else:
                _new_los = _mal_fn(_c_sport, _c_mt, _c_bucket, _orig_los)

            if _new_los != _orig_los:
                _prof  = _lkp_fn(_c_sport, _c_mt, _c_bucket)
                _grade = _prof.get("margin_grade") if _prof else None
                _c["los"] = _new_los
                _c["_margin_grade"] = _grade
    except Exception:
        pass

    # ── CLOSE → CUSHION switch annotation ────────────────────────────────────
    # For each CLOSE-graded candidate, check whether a CUSHION alternative
    # exists on the same fixture for the same pick direction (same team name
    # for ML/Spread, or same Over/Under label for Totals).
    # If found with LOS >= CLOSE_LOS × 0.90, annotate the CUSHION leg as the
    # recommended switch so the recommender picks it first.
    #
    # Note: ALE ordering already naturally prefers CUSHION legs (×1.10 vs ×0.85),
    # so this annotation is primarily for transparency / logging.
    try:
        _close_cands  = [c for c in candidates if c.get("_margin_grade") == "CLOSE"]
        _cushion_map: dict[tuple, dict] = {}
        for c in candidates:
            if c.get("_margin_grade") == "CUSHION":
                _key = (c.get("fixture_id"), _pick_direction(c.get("pick", "")))
                if _key not in _cushion_map or c["los"] > _cushion_map[_key]["los"]:
                    _cushion_map[_key] = c

        for close_c in _close_cands:
            _ck = (close_c.get("fixture_id"), _pick_direction(close_c.get("pick", "")))
            cushion_alt = _cushion_map.get(_ck)
            if cushion_alt and cushion_alt["los"] >= close_c["los"] * 0.90:
                cushion_alt["ale_switched"]  = True
                cushion_alt["ale_reason"]    = "margin_tightening"
                cushion_alt["_replaced_close_los"] = close_c["los"]
    except Exception:
        pass  # annotation is non-fatal

    # ── Boost safety re-ranking ───────────────────────────────────────────────
    # When boost_pct >= 0.30, apply a safety premium to LOS so that safer lines
    # (higher implied_prob) rank above aggressive lines, implementing the step-down
    # to equivalent EV at lower risk.
    if boost_pct >= 0.30:
        for c in candidates:
            multiplier = _boost_los_multiplier(c.get("implied_prob", 50.0), boost_pct)
            if multiplier != 1.0:
                c["los"] = round(c["los"] * multiplier, 4)
                c["_boost_adjusted"] = True

    # Filter by minimum LOS and sort
    result = [c for c in candidates if c["los"] >= _LOS_MIN]
    result.sort(key=lambda l: -l["los"])
    return result


# ─── Parlay construction ──────────────────────────────────────────────────────

def build_parlay(scored_legs: list[dict], stake: float = 10.0) -> dict:
    """
    Combine N pre-scored legs into a parlay.

    True parlay EV:
        combined_win_prob = product of individual win_probs (assumes independence)
        combined_odds     = product of decimal odds
        EV                = p_win * (combined_odds - 1) * stake - (1 - p_win) * stake
    """
    if not scored_legs:
        return {}

    combined_odds    = 1.0
    combined_win_prob = 1.0
    for leg in scored_legs:
        combined_odds    *= leg["odds"]
        combined_win_prob *= (leg["win_prob"] / 100)

    ev = combined_win_prob * (combined_odds - 1) * stake - (1 - combined_win_prob) * stake

    sports   = list({l["sport"] for l in scored_legs})
    games    = list({l["game"]  for l in scored_legs})
    warnings = correlation_check(scored_legs)

    return {
        "legs":              scored_legs,
        "n_legs":            len(scored_legs),
        "combined_odds":     round(combined_odds, 3),
        "combined_win_prob": round(combined_win_prob * 100, 3),
        "ev":                round(ev, 4),
        "ev_pct":            round(ev / stake * 100, 2),
        "payout":            round((combined_odds - 1) * stake, 2),
        "stake":             stake,
        "sports":            sports,
        "n_games":           len(games),
        "warnings":          warnings,
        "grade":             _grade(ev / stake, combined_win_prob - 1 / combined_odds),
    }


# ─── Correlation checker ──────────────────────────────────────────────────────

def correlation_check(legs: list[dict]) -> list[str]:
    """
    Detect known correlation risks:
    1. Same-game parlay (multiple legs from same fixture)
    2. Positively correlated legs (e.g. two team totals from same game)
    3. Negatively correlated legs (e.g. both sides of a spread)
    """
    warnings = []

    # Group by fixture
    fixture_legs: dict[str, list] = {}
    for leg in legs:
        fid = leg["fixture_id"]
        fixture_legs.setdefault(fid, []).append(leg)

    for fid, flegs in fixture_legs.items():
        if len(flegs) < 2:
            continue
        game = flegs[0]["game"]
        picks = [l["pick"] for l in flegs]
        markets = [l["market"] for l in flegs]

        # Opposite sides of same market (e.g. team A and team B moneyline)
        if len(set(markets)) == 1 and markets[0] == "h2h":
            warnings.append(
                f"CONFLICT: Both sides of {game} moneyline selected — only one can win."
            )
        elif len(flegs) >= 2:
            warnings.append(
                f"CORRELATED: {len(flegs)} legs from same game ({game}). "
                f"Sportsbooks may restrict or void same-game parlays."
            )

    # Opposite sides of totals (over + under same game)
    seen_totals: dict[str, list] = {}
    for leg in legs:
        if leg["market"] == "totals":
            key = f"{leg['fixture_id']}:{leg['point']}"
            seen_totals.setdefault(key, []).append(leg["pick"])
    for key, picks in seen_totals.items():
        if "Over" in picks and "Under" in picks:
            warnings.append(
                f"CONFLICT: Both Over and Under {key.split(':')[1]} selected for same game."
            )

    return warnings


# ─── Optimizer ────────────────────────────────────────────────────────────────

def optimize_parlays(
    db: Session,
    n_results:   int   = 5,
    parlay_size: int   = 4,
    stake:       float = 10.0,
    min_leg_grade: str = "C",        # only use legs graded C or better
    markets:     list  = None,
    sport_filter: list = None,       # e.g. ["NBA", "MLB"]
    max_same_game: int = 1,          # max legs from same fixture
    min_odds:    float = 1.0,        # min combined parlay odds (2.0 = -100 American payout minimum)
) -> dict:
    """
    Main optimizer entry point.
    1. Fetch + score all available legs
    2. Filter to quality legs (min_leg_grade)
    3. Enumerate combinations of parlay_size
    4. Return top n_results by EV
    """
    GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    min_rank    = GRADE_ORDER.get(min_leg_grade, 2)

    # Step 1: get legs
    raw_legs   = get_available_legs(db, markets=markets or ["h2h", "spreads", "totals"])
    if sport_filter:
        raw_legs = [l for l in raw_legs if l["sport"] in sport_filter]

    # Step 2: score
    scored = [score_leg(l) for l in raw_legs]

    # Step 3: filter by grade
    quality = [l for l in scored if GRADE_ORDER.get(l["grade"], 4) <= min_rank]

    # Hard cap to keep combinations manageable (C(40,4) = 91,390 — fine)
    if len(quality) > 60:
        # Keep best 60 by EV
        quality = sorted(quality, key=lambda x: -x["ev"])[:60]

    if len(quality) < parlay_size:
        return {
            "error": f"Not enough quality legs ({len(quality)}) to build a {parlay_size}-leg parlay. "
                     f"Try fetching fresh odds or relaxing the grade filter.",
            "total_legs_available": len(scored),
            "quality_legs": len(quality),
        }

    # Step 4: enumerate combos
    best_parlays = []
    combo_limit  = 5000   # safety cap for large leg pools

    combos = list(itertools.combinations(range(len(quality)), parlay_size))
    if len(combos) > combo_limit:
        # Sample evenly
        step   = len(combos) // combo_limit
        combos = combos[::step]

    for combo in combos:
        legs = [quality[i] for i in combo]

        # Enforce max_same_game constraint
        fixture_counts: dict[str, int] = {}
        skip = False
        for leg in legs:
            fid = leg["fixture_id"]
            fixture_counts[fid] = fixture_counts.get(fid, 0) + 1
            if fixture_counts[fid] > max_same_game:
                skip = True
                break
        if skip:
            continue

        parlay = build_parlay(legs, stake=stake)
        best_parlays.append(parlay)

    if not best_parlays:
        return {
            "error": "No valid parlays found with current constraints.",
            "total_legs_available": len(scored),
            "quality_legs": len(quality),
        }

    # Filter: parlay combined odds must meet the minimum (not individual legs)
    if min_odds > 1.0:
        best_parlays = [p for p in best_parlays if p.get("combined_odds", 1.0) >= min_odds]

    best_parlays.sort(key=lambda x: -x["ev"])
    top = best_parlays[:n_results]

    return {
        "total_legs_available": len(scored),
        "quality_legs":         len(quality),
        "combos_evaluated":     len(best_parlays),
        "parlay_size":          parlay_size,
        "stake":                stake,
        "top_parlays":          top,
        "best_ev":              round(top[0]["ev"], 4) if top else 0,
        "best_odds":            round(top[0]["combined_odds"], 2) if top else 0,
    }
