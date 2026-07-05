"""
cash_out_engine.py — Cash Out Decision Support (Phase 6B + Phase 6C Live Monitor)

Phase 6B (original): evaluate_cash_out() — static check against a known offer amount.
Phase 6C (new):      Live fair-value estimator, spread position monitor, and polling loop.

Public API — Phase 6B
---------------------
  evaluate_cash_out(bet, current_offer_amount, remaining_legs_status,
                    user_doubt_signal=False) -> dict

Public API — Phase 6C
---------------------
  poll_live_odds()          -> dict   — fetch live odds for in-progress LIVE bet games
  get_cashout_dashboard()   -> list   — build per-bet fair value + spread monitor + rec
  get_open_live_bets()      -> list   — find PENDING mock bets with live_recommendation=1
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_CT = ZoneInfo("America/Chicago")

# ─── DB paths ────────────────────────────────────────────────────────────────
_HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
_BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6C — Live odds snapshots table (in historical.db alongside line_snapshots)
# ══════════════════════════════════════════════════════════════════════════════

def _hist_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _bets_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(os.path.abspath(_BETS_DB))
    conn.row_factory = sqlite3.Row
    return conn


_TABLES_ENSURED = False

def _ensure_live_tables():
    global _TABLES_ENSURED
    if _TABLES_ENSURED:
        return
    conn = _hist_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS live_odds_snapshots (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at    TEXT NOT NULL,
            sport_key      TEXT NOT NULL,
            event_id       TEXT NOT NULL,
            game_date      TEXT,
            home_team      TEXT,
            away_team      TEXT,
            bookmaker      TEXT NOT NULL,
            market_key     TEXT NOT NULL,
            outcome_name   TEXT NOT NULL,
            price          REAL,
            point          REAL,
            is_live        INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_live_snap_event
            ON live_odds_snapshots(event_id, captured_at);
        CREATE INDEX IF NOT EXISTS idx_live_snap_date
            ON live_odds_snapshots(game_date, captured_at);
    """)
    conn.commit()
    conn.close()
    _TABLES_ENSURED = True


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6B — Static cash-out evaluation (preserved from original)
# ══════════════════════════════════════════════════════════════════════════════

_PENDING_STATUSES = {"pending", "in_game", "at_risk"}


def evaluate_cash_out(
    bet: dict,
    current_offer_amount: float,
    remaining_legs_status: list[dict],
    user_doubt_signal: bool = False,
) -> dict:
    """
    Evaluate whether to accept a sportsbook cash out offer.
    See module docstring for full input/output contract.
    """
    original_stake = float(bet.get("original_stake", 0))
    original_odds  = float(bet.get("original_odds", 1.0))
    current_offer  = float(current_offer_amount)

    full_payout = original_stake * original_odds

    # ── Walk legs ────────────────────────────────────────────────────────────
    combined_remaining_prob = 1.0
    any_at_risk             = False
    any_lost                = False
    all_pending_improving   = True

    at_risk_reasons: list[str] = []

    for leg in remaining_legs_status:
        status       = str(leg.get("status", "pending")).lower()
        updated_prob = float(leg.get("updated_prob", 0.5))
        original_prob= float(leg.get("original_prob", updated_prob))
        at_risk      = bool(leg.get("at_risk_flag", False))

        if status == "lost":
            any_lost = True
        elif status == "won":
            pass
        elif status in _PENDING_STATUSES:
            combined_remaining_prob *= max(0.0, min(1.0, updated_prob))
            if at_risk:
                any_at_risk = True
                reason = leg.get("at_risk_reason", "").strip()
                at_risk_reasons.append(reason or "unspecified")
            if updated_prob < original_prob:
                all_pending_improving = False

    if any_lost:
        combined_remaining_prob = 0.0

    # ── Core metrics ─────────────────────────────────────────────────────────
    safe_payout = full_payout if full_payout > 0 else 1.0

    cash_out_implied_prob = current_offer / safe_payout
    current_fair_value    = combined_remaining_prob * full_payout
    cash_out_vig          = current_fair_value - current_offer
    safe_fair_value       = current_fair_value if current_fair_value > 0 else 1.0
    cash_out_vig_pct      = (cash_out_vig / safe_fair_value) * 100

    ev_hold     = combined_remaining_prob * full_payout
    ev_cash_out = current_offer - original_stake
    ev_delta    = ev_hold - ev_cash_out

    # ── Recommendation logic (first match wins) ───────────────────────────────
    implied_overprice = cash_out_implied_prob > combined_remaining_prob + 0.05
    co_captures_90pct = ev_cash_out > ev_hold * 0.90 if ev_hold > 0 else False

    if implied_overprice or any_at_risk or co_captures_90pct:
        recommendation = "CASH_OUT_RECOMMENDED"
        if any_at_risk:
            r_str = "; ".join(at_risk_reasons[:3])
            reason = f"Leg flagged at risk ({r_str}). "
        elif implied_overprice:
            reason = (
                f"Book's offer implies {cash_out_implied_prob*100:.1f}% win probability "
                f"vs your current estimate of {combined_remaining_prob*100:.1f}% — "
                f"book is absorbing {(cash_out_implied_prob - combined_remaining_prob)*100:.1f}pp less "
                f"risk than your model suggests. "
            )
        else:
            reason = "Cash out offer captures 90%+ of hold value. "
        reason += (
            f"Offer ${current_offer:.2f} vs fair value ${current_fair_value:.2f}. "
            f"Book vig {cash_out_vig_pct:.1f}%."
        )
        confidence = "high" if (any_at_risk or combined_remaining_prob < 0.30) else "medium"

    elif (
        ev_hold > 0
        and ev_cash_out > 0
        and ev_hold > ev_cash_out * 1.30
        and cash_out_vig_pct > 15
        and all_pending_improving
    ):
        recommendation = "HOLD_STRONG"
        pct_better = (ev_hold / ev_cash_out - 1) * 100 if ev_cash_out > 0 else 0
        reason = (
            f"All legs tracking at or above original probability. "
            f"Hold EV ${ev_hold:.2f} vs cash out net ${ev_cash_out:.2f} "
            f"({pct_better:.0f}% better). "
            f"Book vig {cash_out_vig_pct:.1f}% — significantly overpriced offer. "
            f"Don't take it."
        )
        confidence = "high"

    elif (
        ev_hold > 0
        and ev_cash_out >= ev_hold * 0.90
        and user_doubt_signal
        and cash_out_vig_pct < 8
    ):
        recommendation = "CASH_OUT_OK"
        reason = (
            f"Hold EV ${ev_hold:.2f} vs cash out net ${ev_cash_out:.2f} — within 10%. "
            f"Book is offering relatively fair terms (vig {cash_out_vig_pct:.1f}%). "
            f"Acceptable to lock in profit given your qualitative concerns."
        )
        confidence = "medium"

    else:
        recommendation = "HOLD"
        offer_pct = (current_offer / safe_fair_value) * 100 if current_fair_value > 0 else 0
        reason = (
            f"Hold EV ${ev_hold:.2f} vs cash out net ${ev_cash_out:.2f}. "
            f"Book offering {offer_pct:.0f}% of fair value "
            f"(vig {cash_out_vig_pct:.1f}%). "
        )
        if cash_out_vig_pct > 10:
            reason += "Significant value lost if you cash out now."
        else:
            reason += "Hold has meaningfully higher expected value."
        confidence = "medium"

    return {
        "full_payout":             round(full_payout,             2),
        "combined_remaining_prob": round(combined_remaining_prob, 4),
        "cash_out_implied_prob":   round(cash_out_implied_prob,   4),
        "current_fair_value":      round(current_fair_value,      2),
        "cash_out_vig":            round(cash_out_vig,            2),
        "cash_out_vig_pct":        round(cash_out_vig_pct,        1),
        "ev_hold":                 round(ev_hold,                 2),
        "ev_cash_out":             round(ev_cash_out,             2),
        "ev_delta":                round(ev_delta,                2),
        "any_at_risk":             any_at_risk,
        "recommendation":          recommendation,
        "recommendation_reason":   reason,
        "confidence":              confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6C — Live Odds Polling
# ══════════════════════════════════════════════════════════════════════════════

# Sport key → TheOddsAPI sport key mapping (same as config.ODDS_SPORT_KEYS)
_SPORT_TO_API_KEY: dict[str, str] = {
    "NBA":        "basketball_nba",
    "NFL":        "americanfootball_nfl",
    "MLB":        "baseball_mlb",
    "NHL":        "icehockey_nhl",
    "CFB":        "americanfootball_ncaaf",
    "CBB":        "basketball_ncaab",
    "EPL":        "soccer_epl",
    "UCL":        "soccer_uefa_champs_league",
    "LaLiga":     "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1":     "soccer_france_ligue_one",
    "SerieA":     "soccer_italy_serie_a",
    "soccer_fifa_world_cup": "soccer_fifa_world_cup",
}

# Reverse mapping for convenience
_API_KEY_TO_SPORT = {v: k for k, v in _SPORT_TO_API_KEY.items()}

# FanDuel bookmaker key in TheOddsAPI
_FD_KEY = "fanduel"


def get_open_live_bets() -> list[dict]:
    """
    Find PENDING mock bets with live_recommendation=1 that have games today.
    Returns list of dicts with bet + leg info needed for live monitoring.
    """
    conn = _bets_conn()
    today = datetime.now(_CT).strftime("%Y-%m-%d")

    bets = conn.execute("""
        SELECT mb.id, mb.odds, mb.amount, mb.legs, mb.bet_info, mb.status,
               mb.composite_score, mb.bet_tier, mb.predicted_win_prob,
               mb.game_date, mb.source
        FROM mock_bets mb
        WHERE mb.status = 'PENDING'
          AND mb.live_recommendation = 1
          AND mb.game_date = ?
    """, (today,)).fetchall()

    result = []
    for b in bets:
        bet_id = b["id"]
        legs = conn.execute("""
            SELECT mbl.id, mbl.description, mbl.market_type, mbl.sport,
                   mbl.win_prob, mbl.fixture_id, mbl.leg_result,
                   mbl.is_alt_line, mbl.predicted_win_prob
            FROM mock_bet_legs mbl
            WHERE mbl.mock_bet_id = ?
            ORDER BY mbl.leg_index
        """, (bet_id,)).fetchall()

        result.append({
            "bet_id":       bet_id,
            "odds":         b["odds"],
            "stake":        b["amount"] or 10.0,
            "n_legs":       b["legs"],
            "bet_info":     b["bet_info"],
            "status":       b["status"],
            "composite":    b["composite_score"],
            "bet_tier":     b["bet_tier"],
            "win_prob":     b["predicted_win_prob"],
            "game_date":    b["game_date"],
            "source":       b["source"],
            "legs": [
                {
                    "leg_id":       l["id"],
                    "description":  l["description"],
                    "market_type":  l["market_type"],
                    "sport":        l["sport"],
                    "win_prob":     l["win_prob"] or l["predicted_win_prob"] or 50.0,
                    "fixture_id":   l["fixture_id"],
                    "leg_result":   l["leg_result"],
                    "is_alt_line":  bool(l["is_alt_line"]),
                }
                for l in legs
            ],
        })

    conn.close()
    return result


def _parse_spread_from_description(desc: str) -> tuple[str | None, float | None]:
    """
    Extract team name and spread line from a leg description.
    e.g. "Carolina Hurricanes +1.5 (spreads)" -> ("Carolina Hurricanes", 1.5)
         "New York Yankees -1.5 (alternate_spreads)" -> ("New York Yankees", -1.5)
    """
    m = re.match(r'^(.+?)\s+([+-]?\d+\.?\d*)\s*\(', desc or "")
    if m:
        return m.group(1).strip(), float(m.group(2))
    return None, None


def _parse_total_from_description(desc: str) -> tuple[str | None, float | None]:
    """
    Extract over/under and line from a leg description.
    e.g. "Over 8.5 (totals)" -> ("Over", 8.5)
    """
    m = re.match(r'^(Over|Under)\s+(\d+\.?\d*)\s*\(', desc or "", re.IGNORECASE)
    if m:
        return m.group(1).capitalize(), float(m.group(2))
    return None, None


def _parse_h2h_from_description(desc: str) -> str | None:
    """
    Extract team name from h2h leg description.
    e.g. "New York Yankees (h2h)" -> "New York Yankees"
    """
    m = re.match(r'^(.+?)\s*\(h2h\)', desc or "")
    if m:
        return m.group(1).strip()
    return None


def _get_sports_for_open_bets(open_bets: list[dict]) -> set[str]:
    """Get unique TheOddsAPI sport keys needed for open bets."""
    sports = set()
    for bet in open_bets:
        for leg in bet.get("legs", []):
            sport = leg.get("sport", "")
            api_key = _SPORT_TO_API_KEY.get(sport, sport)
            if api_key:
                sports.add(api_key)
    return sports


def poll_live_odds() -> dict:
    """
    Fetch live odds from TheOddsAPI for sports where we have open LIVE bets.
    Stores snapshots in live_odds_snapshots table.

    Returns: {polled_sports, events_found, rows_inserted, skipped_reason, credits_est}
    """
    _ensure_live_tables()

    open_bets = get_open_live_bets()
    if not open_bets:
        return {
            "polled_sports":  [],
            "events_found":   0,
            "rows_inserted":  0,
            "skipped_reason": "no_open_live_bets",
            "credits_est":    0,
        }

    # Determine which sport keys to poll
    needed_sport_keys = _get_sports_for_open_bets(open_bets)
    if not needed_sport_keys:
        return {
            "polled_sports":  [],
            "events_found":   0,
            "rows_inserted":  0,
            "skipped_reason": "no_sport_keys_mapped",
            "credits_est":    0,
        }

    # Collect fixture IDs we care about (for filtering)
    target_fixture_ids = set()
    for bet in open_bets:
        for leg in bet.get("legs", []):
            fid = leg.get("fixture_id")
            if fid:
                target_fixture_ids.add(fid)

    # Check credit budget before fetching
    try:
        from creator_tier import check_credit_budget
        budget = check_credit_budget()
        if budget == "suspended":
            return {
                "polled_sports":  list(needed_sport_keys),
                "events_found":   0,
                "rows_inserted":  0,
                "skipped_reason": "credits_suspended",
                "credits_est":    0,
            }
    except Exception:
        budget = "unknown"

    from odds_api import fetch_odds

    captured_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    total_rows    = 0
    total_events  = 0
    polled_sports: list[str] = []

    conn = _hist_conn()

    for sk in needed_sport_keys:
        try:
            # Fetch all events for this sport (h2h + spreads + totals = 3 credits)
            events = fetch_odds(sk, markets="h2h,spreads,totals", odds_format="decimal")
            if not events:
                continue

            polled_sports.append(sk)
            rows_to_insert: list[tuple] = []

            for event in events:
                event_id = event.get("id", "")

                # Only store snapshots for events we have bets on
                if target_fixture_ids and event_id not in target_fixture_ids:
                    continue

                total_events += 1
                game_date = (event.get("commence_time") or "")[:10]
                home      = event.get("home_team", "")
                away      = event.get("away_team", "")

                # Detect if the game is live (started but not completed)
                commence = event.get("commence_time", "")
                is_live = 0
                try:
                    game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    if game_dt <= datetime.now(timezone.utc):
                        is_live = 1
                except Exception:
                    pass

                for bk in event.get("bookmakers", []):
                    bk_key = bk.get("key", "")
                    for mkt in bk.get("markets", []):
                        for oc in mkt.get("outcomes", []):
                            rows_to_insert.append((
                                captured_at, sk, event_id, game_date,
                                home, away, bk_key, mkt.get("key", ""),
                                oc.get("name", ""),
                                oc.get("price"),
                                oc.get("point"),
                                is_live,
                            ))

            if rows_to_insert:
                conn.executemany("""
                    INSERT INTO live_odds_snapshots
                      (captured_at, sport_key, event_id, game_date, home_team, away_team,
                       bookmaker, market_key, outcome_name, price, point, is_live)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, rows_to_insert)
                conn.commit()
                total_rows += len(rows_to_insert)

        except Exception as exc:
            print(f"[CashoutPoll] Error fetching {sk}: {exc}")

    conn.close()

    result = {
        "polled_sports":  polled_sports,
        "events_found":   total_events,
        "rows_inserted":  total_rows,
        "skipped_reason": None,
        "credits_est":    len(polled_sports) * 3,
        "captured_at":    captured_at,
        "target_fixtures": len(target_fixture_ids),
    }
    print(f"[CashoutPoll] {result}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6C — Fair Value Computation
# ══════════════════════════════════════════════════════════════════════════════

def _get_latest_fd_odds(event_id: str, market_key: str, outcome_name: str) -> dict | None:
    """
    Get the latest FanDuel odds snapshot for a specific outcome.
    Searches live_odds_snapshots first, falls back to line_snapshots.
    Returns: {price, point, captured_at, bookmaker, is_live} or None.
    """
    conn = _hist_conn()

    # Try live_odds_snapshots first (most recent)
    row = conn.execute("""
        SELECT price, point, captured_at, bookmaker, is_live
        FROM live_odds_snapshots
        WHERE event_id = ? AND market_key = ? AND outcome_name = ?
          AND bookmaker = 'fanduel'
        ORDER BY captured_at DESC
        LIMIT 1
    """, (event_id, market_key, outcome_name)).fetchone()

    if row:
        conn.close()
        return dict(row)

    # Fallback to line_snapshots (pre-game capture)
    row = conn.execute("""
        SELECT price, point, captured_at, bookmaker, 0 as is_live
        FROM line_snapshots
        WHERE event_id = ? AND market_key = ? AND outcome_name = ?
          AND bookmaker = 'fanduel'
        ORDER BY captured_at DESC
        LIMIT 1
    """, (event_id, market_key, outcome_name)).fetchone()

    conn.close()
    return dict(row) if row else None


def _get_alt_line_odds(event_id: str, team: str, line: float) -> dict | None:
    """
    Get the latest alt_lines snapshot for a specific team + spread line.
    Used for alt spread legs to get a more accurate implied prob than the main market.
    """
    conn = _hist_conn()
    row = conn.execute("""
        SELECT odds, line, fetched_at
        FROM alt_lines
        WHERE event_id = ? AND team = ? AND line = ?
          AND bookmaker = 'fanduel'
          AND market_key = 'alternate_spreads'
        ORDER BY fetched_at DESC
        LIMIT 1
    """, (event_id, team, line)).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_opening_odds(event_id: str, market_key: str, outcome_name: str) -> dict | None:
    """
    Get the earliest (opening) odds for an outcome.
    Used to measure line movement for spread position monitor.
    """
    conn = _hist_conn()
    row = conn.execute("""
        SELECT price, point, captured_at, bookmaker
        FROM line_snapshots
        WHERE event_id = ? AND market_key = ? AND outcome_name = ?
          AND bookmaker = 'fanduel'
        ORDER BY captured_at ASC
        LIMIT 1
    """, (event_id, market_key, outcome_name)).fetchone()
    conn.close()
    return dict(row) if row else None


def _implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def classify_leg_status(
    leg: dict,
) -> dict:
    """
    Classify a single leg's live status.

    Returns: {
        status: 'WON' | 'LOST' | 'PENDING' | 'UNKNOWN',
        live_prob: float (0-1),
        live_odds: float | None (decimal),
        live_point: float | None,
        opening_odds: float | None,
        opening_point: float | None,
        data_source: 'live_snapshot' | 'line_snapshot' | 'model_wp' | 'none',
    }
    """
    leg_result = leg.get("leg_result")

    # Already settled
    if leg_result == "WIN":
        return {"status": "WON", "live_prob": 1.0, "live_odds": None,
                "live_point": None, "opening_odds": None, "opening_point": None,
                "data_source": "settlement"}
    if leg_result == "LOSS":
        return {"status": "LOST", "live_prob": 0.0, "live_odds": None,
                "live_point": None, "opening_odds": None, "opening_point": None,
                "data_source": "settlement"}
    if leg_result == "PUSH":
        return {"status": "WON", "live_prob": 1.0, "live_odds": None,
                "live_point": None, "opening_odds": None, "opening_point": None,
                "data_source": "settlement"}

    # Still pending — try to get live odds from snapshots
    fixture_id  = leg.get("fixture_id")
    description = leg.get("description", "")
    market_type = leg.get("market_type", "")

    # Determine market_key and outcome_name for lookup
    market_key = _normalize_market_key(market_type)
    # If market_type normalization fails, try extracting from description
    if market_key == market_type.lower().strip() and "(" in description:
        # Extract from parenthetical: "Team +1.5 (Alt Spread)" → "Alt Spread"
        paren_match = re.search(r'\(([^)]+)\)', description)
        if paren_match:
            market_key = _normalize_market_key(paren_match.group(1))
    outcome_name = _extract_outcome_name(description, market_key)

    live_snap  = None
    open_snap  = None

    if fixture_id and outcome_name:
        live_snap = _get_latest_fd_odds(fixture_id, market_key, outcome_name)
        open_snap = _get_opening_odds(fixture_id, market_key, outcome_name)

    # For alt spread legs, try to get the exact line from alt_lines first
    is_alt = (market_type or "").lower() in ("alternate_spreads", "alt spread")
    if is_alt and fixture_id and outcome_name:
        _, our_spread = _parse_spread_from_description(description)
        if our_spread is not None:
            alt_data = _get_alt_line_odds(fixture_id, outcome_name, our_spread)
            if alt_data and alt_data.get("odds"):
                alt_price = alt_data["odds"]
                return {
                    "status":        "PENDING",
                    "live_prob":     _implied_prob(alt_price),
                    "live_odds":     alt_price,
                    "live_point":    live_snap.get("point") if live_snap else None,
                    "opening_odds":  alt_price,  # alt_lines IS the opening for alt spreads
                    "opening_point": open_snap.get("point") if open_snap else None,
                    "data_source":   "alt_lines",
                }

    if live_snap and live_snap.get("price"):
        live_price = live_snap["price"]
        return {
            "status":        "PENDING",
            "live_prob":     _implied_prob(live_price),
            "live_odds":     live_price,
            "live_point":    live_snap.get("point"),
            "opening_odds":  open_snap.get("price") if open_snap else None,
            "opening_point": open_snap.get("point") if open_snap else None,
            "data_source":   "live_snapshot" if live_snap.get("is_live") else "line_snapshot",
        }

    # Fallback: use model win_prob from generation time
    wp = leg.get("win_prob", 50.0)
    return {
        "status":        "PENDING",
        "live_prob":     wp / 100.0 if wp > 1.0 else wp,
        "live_odds":     None,
        "live_point":    None,
        "opening_odds":  open_snap.get("price") if open_snap else None,
        "opening_point": open_snap.get("point") if open_snap else None,
        "data_source":   "model_wp",
    }


def _normalize_market_key(market_type: str) -> str:
    """Map leg market_type to TheOddsAPI market_key for snapshot lookup."""
    mt = (market_type or "").lower().strip()
    mapping = {
        "h2h": "h2h",
        "moneyline": "h2h",
        "spreads": "spreads",
        "spread": "spreads",
        "alt spread": "spreads",
        "alternate_spreads": "spreads",
        "totals": "totals",
        "total": "totals",
        "alt total": "totals",
        "alternate_totals": "totals",
    }
    return mapping.get(mt, mt)


def _extract_outcome_name(description: str, market_key: str) -> str | None:
    """
    Extract the outcome_name that matches TheOddsAPI's format.
    - h2h: team name (e.g. "New York Yankees")
    - spreads: team name (e.g. "Carolina Hurricanes")
    - totals: "Over" or "Under"
    """
    desc = description or ""

    if market_key == "totals":
        ou, _ = _parse_total_from_description(desc)
        return ou  # "Over" or "Under"

    if market_key == "h2h":
        team = _parse_h2h_from_description(desc)
        return team

    if market_key == "spreads":
        team, _ = _parse_spread_from_description(desc)
        return team

    # Fallback: try to extract team name before first special char
    m = re.match(r'^(.+?)\s*[\(\-\+\d]', desc)
    if m:
        return m.group(1).strip()
    return None


def compute_fair_value(bet: dict) -> dict:
    """
    Compute fair cashout value for a bet based on live odds.

    fair_value = stake × combined_decimal_odds × product(P(each pending leg))

    For legs that are WON, P = 1.0 (already locked in).
    For legs that are LOST, the bet is dead (fair_value = 0).
    For PENDING legs, P = implied prob from latest live odds.

    Returns: {
        fair_value, full_payout, combined_live_prob,
        leg_statuses: [{description, status, live_prob, ...}],
        any_lost, any_live_data, all_won,
        cashout_guidance: str,
    }
    """
    stake     = bet.get("stake", 10.0)
    odds      = bet.get("odds", 1.0)  # decimal odds
    legs      = bet.get("legs", [])
    full_payout = stake * odds

    leg_statuses   = []
    combined_prob  = 1.0
    any_lost       = False
    any_live_data  = False
    all_won        = True

    for leg in legs:
        ls = classify_leg_status(leg)
        ls["description"] = leg.get("description", "")
        ls["market_type"] = leg.get("market_type", "")
        leg_statuses.append(ls)

        if ls["status"] == "LOST":
            any_lost = True
            all_won  = False
        elif ls["status"] == "PENDING":
            combined_prob *= ls["live_prob"]
            all_won = False
            if ls["data_source"] in ("live_snapshot", "line_snapshot"):
                any_live_data = True
        # WON legs contribute P=1.0 (no multiplication needed)

    if any_lost:
        fair_value    = 0.0
        combined_prob = 0.0
    elif all_won:
        fair_value = full_payout
    else:
        fair_value = full_payout * combined_prob

    return {
        "fair_value":         round(fair_value, 2),
        "full_payout":        round(full_payout, 2),
        "combined_live_prob": round(combined_prob, 4),
        "stake":              stake,
        "decimal_odds":       odds,
        "leg_statuses":       leg_statuses,
        "any_lost":           any_lost,
        "any_live_data":      any_live_data,
        "all_won":            all_won,
        "n_legs":             len(legs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6C — Spread Position Monitor
# ══════════════════════════════════════════════════════════════════════════════

# Position thresholds (in spread points)
_SPREAD_SAFE_THRESHOLD    = 1.0   # line moved ≥1pt in our favor
_SPREAD_DANGER_THRESHOLD  = 1.0   # line moved ≥1pt against us


def classify_spread_position(leg_status: dict, leg: dict) -> dict:
    """
    For spread legs, compare our line at placement vs the live line.

    SAFE:    live line moved in our favor (e.g. we took +1.5, now +2.5)
    NEUTRAL: within ±threshold
    DANGER:  live line moved against us (e.g. we took +1.5, now -0.5)

    For non-spread legs, returns position='N/A'.
    """
    market_type = (leg.get("market_type") or "").lower()
    if market_type not in ("spreads", "alternate_spreads"):
        return {"position": "N/A", "our_line": None, "live_line": None,
                "line_move": None, "detail": "Not a spread leg"}

    description = leg.get("description", "")
    _, our_line = _parse_spread_from_description(description)

    if our_line is None:
        return {"position": "UNKNOWN", "our_line": None, "live_line": None,
                "line_move": None, "detail": "Could not parse spread from description"}

    live_point = leg_status.get("live_point")
    if live_point is None:
        # Check opening_point as fallback
        opening_point = leg_status.get("opening_point")
        if opening_point is not None:
            return {"position": "NEUTRAL", "our_line": our_line,
                    "live_line": opening_point, "line_move": 0.0,
                    "detail": "No live spread data; using opening line"}
        return {"position": "UNKNOWN", "our_line": our_line, "live_line": None,
                "line_move": None, "detail": "No live or opening spread data"}

    # Line move interpretation for spreads:
    # TheOddsAPI point = team handicap. More negative = team stronger.
    # We bet on this team → team getting stronger = SAFE for us.
    #
    # Example: took CAR +1.5 (our_line=+1.5), live shows CAR -1.5 (live_point=-1.5)
    #   line_move = -1.5 - 1.5 = -3.0 → team became 3pts stronger → SAFE
    # Example: took CAR +1.5, live shows CAR +3.5 (live_point=+3.5)
    #   line_move = +3.5 - 1.5 = +2.0 → team became bigger underdog → DANGER
    line_move = live_point - our_line

    # Negative line_move = team stronger = SAFE for us
    if line_move <= -_SPREAD_SAFE_THRESHOLD:
        position = "SAFE"
        detail = f"Line moved {abs(line_move):.1f}pts in our favor ({our_line:+.1f} → {live_point:+.1f})"
    elif line_move >= _SPREAD_DANGER_THRESHOLD:
        position = "DANGER"
        detail = f"Line moved {line_move:+.1f}pts against us ({our_line:+.1f} → {live_point:+.1f})"
    else:
        position = "NEUTRAL"
        detail = f"Line stable ({our_line:+.1f} → {live_point:+.1f}, move={line_move:+.1f})"

    return {
        "position":  position,
        "our_line":  our_line,
        "live_line": live_point,
        "line_move": round(line_move, 1),
        "detail":    detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6C — Recommendation Bands
# ══════════════════════════════════════════════════════════════════════════════

def compute_recommendation(
    fair_val: dict,
    spread_positions: list[dict],
    stake: float,
) -> dict:
    """
    Assign a recommendation band based on fair value and leg positions.

    LOCK_PROFIT:  fair_value >= 120% of stake AND has a shaky leg (DANGER or low prob)
    FAIR_HOLD:    tracking near expected value, no urgent action needed
    SALVAGE:      a leg is in DANGER but overall value is still positive
    DEAD:         a leg has been lost — bet cannot win
    GRADED_WIN:   all legs won — awaiting payout

    Returns: {band, reason, urgency: 'high'|'medium'|'low'}
    """
    fv           = fair_val.get("fair_value", 0)
    any_lost     = fair_val.get("any_lost", False)
    all_won      = fair_val.get("all_won", False)
    combined_prob = fair_val.get("combined_live_prob", 0)
    full_payout  = fair_val.get("full_payout", 0)

    has_danger = any(sp.get("position") == "DANGER" for sp in spread_positions)
    has_safe   = any(sp.get("position") == "SAFE"   for sp in spread_positions)

    # Any leg with live_prob < 0.35 is "shaky"
    shaky_legs = [ls for ls in fair_val.get("leg_statuses", [])
                  if ls.get("status") == "PENDING" and ls.get("live_prob", 1.0) < 0.35]

    # 1. DEAD — leg lost
    if any_lost:
        return {
            "band":    "DEAD",
            "reason":  "A leg has been lost. This bet cannot win.",
            "urgency": "high",
        }

    # 2. GRADED_WIN — all legs won
    if all_won:
        return {
            "band":    "GRADED_WIN",
            "reason":  f"All legs won! Full payout: ${full_payout:.2f}.",
            "urgency": "low",
        }

    # 3. LOCK_PROFIT — fair value >= 120% stake AND shaky leg or danger
    if fv >= stake * 1.20 and (shaky_legs or has_danger):
        shaky_detail = ""
        if shaky_legs:
            worst = min(shaky_legs, key=lambda x: x.get("live_prob", 1))
            shaky_detail = f" Weakest leg: \"{worst.get('description','')}\" at {worst.get('live_prob',0)*100:.0f}%."
        return {
            "band":    "LOCK_PROFIT",
            "reason":  (f"Fair value ${fv:.2f} is {fv/stake*100:.0f}% of stake. "
                        f"Consider locking profit — you have unrealized gains "
                        f"with risk still on the table.{shaky_detail}"),
            "urgency": "high",
        }

    # 4. SALVAGE — leg in DANGER but value still positive
    if has_danger and fv > 0:
        danger_legs = [sp for sp in spread_positions if sp.get("position") == "DANGER"]
        detail_str = "; ".join(d.get("detail", "") for d in danger_legs[:2])
        return {
            "band":    "SALVAGE",
            "reason":  (f"Spread position moved against you ({detail_str}). "
                        f"Fair value still ${fv:.2f}. "
                        f"If FD offers ≥85% of fair value, consider cashing out."),
            "urgency": "medium",
        }

    # 5. FAIR_HOLD — tracking near expected
    fv_pct = (fv / stake * 100) if stake > 0 else 0
    prob_pct = combined_prob * 100

    reason_parts = [f"Fair value ${fv:.2f} ({fv_pct:.0f}% of stake)."]
    if has_safe:
        reason_parts.append("Spread position is favorable.")
    reason_parts.append(f"Combined remaining probability: {prob_pct:.1f}%.")
    reason_parts.append("Hold — no urgent action needed.")

    return {
        "band":    "FAIR_HOLD",
        "reason":  " ".join(reason_parts),
        "urgency": "low",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6C — Dashboard Assembly
# ══════════════════════════════════════════════════════════════════════════════

def get_cashout_dashboard() -> list[dict]:
    """
    Build the full cashout dashboard for all open LIVE bets.
    Combines: fair value, spread position, recommendation band.

    Returns list of per-bet dashboard entries, sorted by urgency.
    """
    open_bets = get_open_live_bets()
    if not open_bets:
        return []

    dashboard = []

    for bet in open_bets:
        # 1. Fair value computation
        fv = compute_fair_value(bet)

        # 2. Spread position for each leg
        spread_positions = []
        for ls, leg in zip(fv["leg_statuses"], bet["legs"]):
            sp = classify_spread_position(ls, leg)
            spread_positions.append(sp)

        # 3. Recommendation band
        rec = compute_recommendation(fv, spread_positions, bet["stake"])

        # 4. Cashout guidance — what to compare FD's offer against
        fair_value = fv["fair_value"]
        if fair_value > 0:
            good_threshold = fair_value * 0.85
            bad_threshold  = fair_value * 0.70
            guidance = (
                f"Compare FD app offer to fair value ${fair_value:.2f}: "
                f"≥${good_threshold:.2f} (85%) = good cashout; "
                f"<${bad_threshold:.2f} (70%) = bad cashout, hold."
            )
        else:
            guidance = "Bet is dead or has no remaining value."

        # 5. Combine leg details
        leg_details = []
        for ls, sp, leg in zip(fv["leg_statuses"], spread_positions, bet["legs"]):
            leg_details.append({
                "description":    leg.get("description", ""),
                "market_type":    leg.get("market_type", ""),
                "sport":          leg.get("sport", ""),
                "status":         ls["status"],
                "live_prob":      round(ls["live_prob"], 4),
                "live_odds":      ls.get("live_odds"),
                "data_source":    ls["data_source"],
                "spread_position": sp,
            })

        entry = {
            "bet_id":           bet["bet_id"],
            "bet_info":         bet["bet_info"],
            "n_legs":           bet["n_legs"],
            "stake":            bet["stake"],
            "decimal_odds":     bet["odds"],
            "full_payout":      fv["full_payout"],
            "fair_value":       fv["fair_value"],
            "combined_prob":    fv["combined_live_prob"],
            "any_live_data":    fv["any_live_data"],
            "recommendation":   rec,
            "cashout_guidance": guidance,
            "legs":             leg_details,
        }
        dashboard.append(entry)

    # Sort by urgency: high > medium > low
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    dashboard.sort(key=lambda x: urgency_order.get(
        x.get("recommendation", {}).get("urgency", "low"), 3))

    return dashboard


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6C — Scheduler helpers
# ══════════════════════════════════════════════════════════════════════════════

def should_poll_live_odds() -> bool:
    """
    Guard: only poll if we have open LIVE bets AND we're in a game window.
    Game window: 11 AM – 1 AM CT (covers day games through west coast evening).
    """
    ct = datetime.now(_CT)
    hour = ct.hour

    # Game window: 11 AM – 1 AM next day
    if not (11 <= hour or hour < 1):
        return False

    # Check for open bets (cheap — just a count query)
    today = ct.strftime("%Y-%m-%d")
    try:
        conn = _bets_conn()
        count = conn.execute("""
            SELECT COUNT(*) FROM mock_bets
            WHERE status = 'PENDING'
              AND live_recommendation = 1
              AND game_date = ?
        """, (today,)).fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False
