"""
f5_shadow.py — First-5-Innings (F5) shadow arm for MLB.

Fetches F5 lines (h2h, spreads, totals) from TheOddsAPI per-event endpoint,
scores them with the existing ATS model's signal, and stores as shadow bets.

Settlement: MLB Stats API linescore (innings 1-5 scores).

Why F5 might work where full-game spreads didn't:
  - F5 is PITCHER-DOMINATED — starters throw most/all of the first 5
  - Our SP-quality features (sp_era_diff, sp_whip_diff) are MOST predictive here
  - Bullpen variance (which we're blind to) is REMOVED
  - F5 is softer / less public action

Architecture:
  1. fetch_f5_odds_for_event()  — per-event endpoint, returns F5 odds
  2. fetch_f5_odds_batch()      — batch all today's MLB events
  3. compute_f5_scores()        — inning-level MLB Stats API → F5 scores
  4. settle_f5_bets()           — resolve F5 mock_bets against F5 scores
  5. generate_f5_shadow()       — score + store F5 shadow bets

Source: 'shadow_f5' in mock_bets table.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger("f5_shadow")
_CT = ZoneInfo("America/Chicago")

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_BETS_DB = os.path.join(_DATA_DIR, "bets.db")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# F5 market keys on TheOddsAPI
F5_MARKETS = [
    "h2h_1st_5_innings",
    "spreads_1st_5_innings",
    "totals_1st_5_innings",
]

# Map F5 OddsAPI market keys to our internal market_type labels
_F5_MARKET_MAP = {
    "h2h_1st_5_innings":     "f5_h2h",
    "spreads_1st_5_innings": "f5_spreads",
    "totals_1st_5_innings":  "f5_totals",
}

_SOURCE = "shadow_f5"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Fetch F5 odds
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_f5_odds_batch(today_str: str | None = None) -> list[dict]:
    """
    Fetch F5 odds for all of today's MLB events using the per-event endpoint.

    Returns list of dicts: {event_id, home, away, game_date, market_type,
                            pick, point, odds, side}

    Cost: 1 credit per event (same as alt-lines fetch).
    """
    from odds_api import _get, ODDS_API_KEY, DEV_MODE, BASE_URL

    if DEV_MODE:
        print("[f5] DEV_MODE: skipping F5 fetch — no credits used")
        return []

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    # Step 1: free event-list fetch
    import requests
    params = {"apiKey": ODDS_API_KEY, "dateFormat": "iso"}
    try:
        r = requests.get(f"{BASE_URL}/sports/baseball_mlb/events",
                         params=params, timeout=15)
        if r.status_code != 200:
            log.warning("F5 event-list fetch failed: HTTP %d", r.status_code)
            return []
        events_list = r.json()
    except Exception as e:
        log.error("F5 event-list fetch error: %s", e)
        return []

    # Filter to today's games (CT date)
    now_utc = datetime.utcnow()
    ct_now = datetime.now(_CT)
    _4am = ct_now.replace(hour=4, minute=0, second=0, microsecond=0)
    if ct_now >= _4am:
        slate_end = _4am + timedelta(days=1)
    else:
        slate_end = _4am
    cutoff = slate_end.astimezone(timezone.utc).replace(tzinfo=None)

    upcoming = []
    for ev in events_list:
        ct_str = ev.get("commence_time", "")
        try:
            game_dt = datetime.fromisoformat(
                ct_str.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            continue
        if now_utc <= game_dt <= cutoff:
            upcoming.append(ev)

    if not upcoming:
        print(f"[f5] No upcoming MLB events for today ({today_str})")
        return []

    print(f"[f5] {len(upcoming)} MLB events today — fetching F5 odds")

    # Step 2: per-event F5 odds fetch
    all_legs: list[dict] = []
    import time as _time

    for ev in upcoming:
        event_id = ev.get("id", "")
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        commence = ev.get("commence_time", "")

        data = _get(
            f"/sports/baseball_mlb/events/{event_id}/odds",
            {
                "regions": "us",
                "markets": ",".join(F5_MARKETS),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
                "bookmakers": "fanduel",
            },
        )
        _time.sleep(0.3)  # rate limit

        if not data or not isinstance(data, dict):
            continue

        for bk in data.get("bookmakers", []):
            if bk.get("key") != "fanduel":
                continue
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in _F5_MARKET_MAP:
                    continue
                internal_mkt = _F5_MARKET_MAP[mkt_key]

                for oc in mkt.get("outcomes", []):
                    name = oc.get("name", "")
                    price = oc.get("price")
                    point = oc.get("point")
                    if not price:
                        continue

                    # Determine side
                    if internal_mkt == "f5_totals":
                        side = name.lower()  # "Over" or "Under"
                        pick = name
                    elif name == home:
                        side = "home"
                        pick = home
                    elif name == away:
                        side = "away"
                        pick = away
                    else:
                        side = name.lower()
                        pick = name

                    all_legs.append({
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence,
                        "game_date": today_str,
                        "market_type": internal_mkt,
                        "pick": pick,
                        "point": float(point) if point is not None else None,
                        "odds": float(price),
                        "side": side,
                    })

    print(f"[f5] Fetched {len(all_legs)} F5 legs across {len(upcoming)} events")
    return all_legs


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: F5 Settlement — compute F5 scores from MLB Stats API linescore
# ═══════════════════════════════════════════════════════════════════════════════

def compute_f5_scores(game_date: str) -> list[dict]:
    """
    Fetch inning-by-inning scores for completed MLB games on a date.
    Returns list of {home, away, f5_home, f5_away, f5_total, completed, game_pk}.

    Uses MLB Stats API linescore (free, no auth).
    """
    import requests as _req

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date}&hydrate=linescore"
    )
    try:
        r = _req.get(url, timeout=15)
        data = r.json()
    except Exception as e:
        log.error("F5 scores fetch error: %s", e)
        return []

    results = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            completed = status in ("Final", "Completed Early")
            if not completed:
                continue

            teams = game.get("teams", {})
            home_name = teams.get("home", {}).get("team", {}).get("name", "")
            away_name = teams.get("away", {}).get("team", {}).get("name", "")

            linescore = game.get("linescore") or {}
            innings = linescore.get("innings", [])

            # Need at least 5 innings for F5 settlement
            if len(innings) < 5:
                continue

            # Sum runs for innings 1-5
            f5_home = sum(
                inn.get("home", {}).get("runs", 0) or 0
                for inn in innings if inn.get("num", 0) <= 5
            )
            f5_away = sum(
                inn.get("away", {}).get("runs", 0) or 0
                for inn in innings if inn.get("num", 0) <= 5
            )

            # Full-game scores for context
            full_home = linescore.get("teams", {}).get("home", {}).get("runs")
            full_away = linescore.get("teams", {}).get("away", {}).get("runs")

            results.append({
                "home_team": home_name,
                "away_team": away_name,
                "f5_home": f5_home,
                "f5_away": f5_away,
                "f5_total": f5_home + f5_away,
                "full_home": full_home,
                "full_away": full_away,
                "completed": completed,
                "game_pk": game.get("gamePk"),
            })

    return results


def _resolve_f5_leg(leg_desc: str, market_type: str, point: float | None,
                    side: str, f5_scores: list[dict]) -> str | None:
    """
    Resolve a single F5 leg against F5 scores.
    Returns 'WIN', 'LOSS', 'PUSH', or None (not found / not completed).
    """
    desc_lower = leg_desc.lower()

    for game in f5_scores:
        home = game["home_team"]
        away = game["away_team"]

        # Match game by team name in description
        if home.lower() not in desc_lower and away.lower() not in desc_lower:
            continue

        f5_home = game["f5_home"]
        f5_away = game["f5_away"]
        f5_total = game["f5_total"]

        # ── F5 Moneyline ──
        if market_type == "f5_h2h":
            if side == "home":
                if f5_home > f5_away:
                    return "WIN"
                elif f5_home == f5_away:
                    return "PUSH"
                else:
                    return "LOSS"
            else:
                if f5_away > f5_home:
                    return "WIN"
                elif f5_away == f5_home:
                    return "PUSH"
                else:
                    return "LOSS"

        # ── F5 Spread ──
        if market_type == "f5_spreads" and point is not None:
            if side == "home":
                covered = f5_home - f5_away + point
            else:
                covered = f5_away - f5_home + point
            if abs(covered) < 0.01:
                return "PUSH"
            return "WIN" if covered > 0 else "LOSS"

        # ── F5 Totals ──
        if market_type == "f5_totals" and point is not None:
            diff = f5_total - point
            if abs(diff) < 0.01:
                return "PUSH"
            if side == "over":
                return "WIN" if f5_total > point else "LOSS"
            else:
                return "WIN" if f5_total < point else "LOSS"

    return None


def settle_f5_bets(days_back: int = 3) -> dict:
    """
    Settle pending shadow_f5 bets using F5 scores from MLB Stats API.
    Returns {checked, settled, errors}.
    """
    conn = sqlite3.connect(_BETS_DB)
    conn.row_factory = sqlite3.Row

    # Find pending F5 bets
    rows = conn.execute("""
        SELECT mb.id, mb.game_date, mb.bet_info, mb.odds,
               mbl.description, mbl.market_type, mbl.fixture_id,
               mbl.id as leg_id
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.source = ?
          AND mb.status = 'PENDING'
          AND mb.game_date >= date('now', '-' || ? || ' days')
    """, (_SOURCE, days_back)).fetchall()

    if not rows:
        conn.close()
        return {"checked": 0, "settled": 0, "errors": 0}

    # Group by game_date for batch F5 score fetching
    dates = set(r["game_date"] for r in rows)
    all_f5_scores = {}
    for d in dates:
        all_f5_scores[d] = compute_f5_scores(d)

    settled = 0
    errors = 0
    for r in rows:
        game_date = r["game_date"]
        scores = all_f5_scores.get(game_date, [])
        if not scores:
            continue

        # Parse leg info from notes or bet_info
        desc = r["description"] or r["bet_info"] or ""
        mkt = r["market_type"] or ""

        # Extract point and side from notes stored at generation time
        notes = conn.execute(
            "SELECT notes FROM mock_bets WHERE id = ?", (r["id"],)
        ).fetchone()
        notes_str = notes["notes"] if notes and notes["notes"] else ""

        # Parse side and point from notes: "side=home point=-0.5 mkt=f5_spreads"
        side_match = re.search(r"side=(\w+)", notes_str)
        point_match = re.search(r"point=([+-]?\d+\.?\d*)", notes_str)
        side = side_match.group(1) if side_match else ""
        point = float(point_match.group(1)) if point_match else None

        result = _resolve_f5_leg(desc, mkt, point, side, scores)
        if result is None:
            continue

        try:
            # Find the matching game for resolved scores
            matched_game = None
            for game in scores:
                if (game["home_team"].lower() in desc.lower() or
                        game["away_team"].lower() in desc.lower()):
                    matched_game = game
                    break

            # Update bet status
            status = f"SETTLED_{result}" if result in ("WIN", "LOSS") else result
            conn.execute(
                "UPDATE mock_bets SET status = ?, settled_at = ? WHERE id = ?",
                (status, datetime.utcnow().isoformat(), r["id"]),
            )

            # Update leg result
            if matched_game:
                conn.execute("""
                    UPDATE mock_bet_legs SET
                        leg_result = ?,
                        resolved_home_team = ?,
                        resolved_away_team = ?,
                        resolved_home_score = ?,
                        resolved_away_score = ?
                    WHERE mock_bet_id = ? AND leg_index = 0
                """, (
                    result,
                    matched_game["home_team"],
                    matched_game["away_team"],
                    matched_game["f5_home"],
                    matched_game["f5_away"],
                    r["id"],
                ))
            else:
                conn.execute(
                    "UPDATE mock_bet_legs SET leg_result = ? WHERE mock_bet_id = ? AND leg_index = 0",
                    (result, r["id"]),
                )

            settled += 1
        except Exception as e:
            log.error("F5 settle error for bet %s: %s", r["id"], e)
            errors += 1

    conn.commit()
    conn.close()
    print(f"[f5] Settlement: {settled} settled, {errors} errors (from {len(rows)} pending)")
    return {"checked": len(rows), "settled": settled, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Score F5 legs with existing ATS model + store as shadow bets
# ═══════════════════════════════════════════════════════════════════════════════

def _score_f5_with_ats(leg: dict) -> dict | None:
    """
    Score an F5 spread leg using the existing ATS model.

    The ATS model predicts P(home covers full-game spread). For F5 spreads,
    the model's signal (SP quality features) is arguably MORE relevant since
    the starter pitches most of the first 5 innings.

    We use the full-game ATS model as-is — the F5 shadow tests whether its
    signal transfers to the F5 market without retraining.
    """
    from ml_model import predict_game_ats

    home = leg["home_team"]
    away = leg["away_team"]
    game_date = leg["game_date"]
    point = leg.get("point")
    mkt = leg["market_type"]

    # Only score F5 spreads (ATS model is a spread model)
    if mkt != "f5_spreads":
        return None

    # The ATS model expects spread from home perspective
    # For F5, the spread is typically ±0.5
    sv = point if leg["side"] == "home" else (-point if point is not None else None)

    prob = predict_game_ats(
        sport="MLB",
        home_team=home,
        away_team=away,
        game_date=game_date,
        spread_value=sv,
    )
    if prob is None:
        return None

    # prob = P(home covers). For away picks, invert.
    win_prob = (1.0 - prob) if leg["side"] == "away" else prob
    odds = leg["odds"]
    imp_prob = 1.0 / odds
    ev = win_prob * (odds - 1) - (1 - win_prob)
    edge = win_prob - imp_prob

    return {
        **leg,
        "win_prob": round(win_prob * 100, 2),
        "ev": round(ev, 4),
        "edge_pp": round(edge * 100, 2),
        "model_used": "mlb_ats_v1",
    }


def generate_f5_shadow(db=None, today_str: str | None = None) -> list[dict]:
    """
    Main entry point: fetch F5 odds, score with ATS model, store as shadow bets.

    Returns list of stored shadow legs.
    """
    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[f5] Generating F5 shadow for {today_str}")

    # Step 1: Fetch F5 odds
    all_legs = fetch_f5_odds_batch(today_str)
    if not all_legs:
        print("[f5] No F5 legs available")
        return []

    # Step 2: Filter to F5 spread legs (ATS model domain)
    # Also keep h2h and totals for logging/future use
    f5_spread_legs = [l for l in all_legs if l["market_type"] == "f5_spreads"]
    f5_h2h_legs = [l for l in all_legs if l["market_type"] == "f5_h2h"]
    f5_total_legs = [l for l in all_legs if l["market_type"] == "f5_totals"]

    print(f"[f5] F5 legs: {len(f5_spread_legs)} spreads, "
          f"{len(f5_h2h_legs)} h2h, {len(f5_total_legs)} totals")

    # Step 3: Score F5 spread legs with ATS model
    scored = []
    for leg in f5_spread_legs:
        try:
            result = _score_f5_with_ats(leg)
            if result is not None:
                scored.append(result)
        except Exception as e:
            log.warning("F5 score error: %s", e)
            continue

    print(f"[f5] Scored {len(scored)} F5 spread legs")

    # Step 4: Store as shadow bets
    if db is not None and scored:
        stored = _store_f5_shadow(db, scored, today_str)
        print(f"[f5] Stored {stored} F5 shadow legs (source={_SOURCE})")

    # Also store h2h and total legs (unscored, for settlement tracking)
    if db is not None:
        _store_f5_unscored(db, f5_h2h_legs + f5_total_legs, today_str)

    return scored


def _store_f5_shadow(db, scored_legs: list[dict], today_str: str) -> int:
    """Store scored F5 spread legs as 1-leg mock_bets."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    run_id = str(uuid.uuid4())
    stored = 0

    for leg in scored_legs:
        mock_id = str(uuid.uuid4())
        desc = (f"{leg['pick']} {leg.get('point', '')} "
                f"(F5 Spread)").strip()
        wp = leg.get("win_prob", 0)
        odds = leg.get("odds", 1.0)
        ev = leg.get("ev", 0)
        point = leg.get("point")
        edge_pp = leg.get("edge_pp", 0)

        bet_info = f"{desc} | wp={wp:.1f}"

        # Dedup
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
            "AND source=:src AND bet_info=:bi"
        ), {"gd": today_str, "src": _SOURCE, "bi": bet_info}).scalar()
        if existing > 0:
            continue

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=odds,
            amount=0.0,
            legs=1,
            bet_info=bet_info,
            predicted_win_prob=wp,
            model_used=leg.get("model_used", "mlb_ats_v1"),
            status="PENDING",
            source=_SOURCE,
            cohort="clean",
            notes=(f"side={leg['side']} point={point} mkt={leg['market_type']} "
                   f"edge_pp={edge_pp:.1f}"),
        )
        db.add(mock)

        mock_leg = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=desc,
            market_type=leg["market_type"],
            sport="MLB",
            win_prob=wp,
            ev=ev,
            fixture_id=leg.get("event_id"),
        )
        db.add(mock_leg)
        stored += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        log.error("F5 shadow commit error: %s", exc)
        return 0

    return stored


def _store_f5_unscored(db, legs: list[dict], today_str: str) -> int:
    """Store unscored F5 h2h/total legs for settlement tracking only."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    run_id = str(uuid.uuid4())
    stored = 0

    for leg in legs:
        mock_id = str(uuid.uuid4())
        mkt = leg["market_type"]
        point = leg.get("point")

        if mkt == "f5_h2h":
            desc = f"{leg['pick']} (F5 Moneyline)"
        elif mkt == "f5_totals":
            desc = f"{leg['side'].title()} {point} (F5 Total)"
        else:
            desc = f"{leg['pick']} {point} (F5)"

        odds = leg.get("odds", 1.0)
        imp_prob = round(1.0 / odds * 100, 2) if odds > 0 else 0
        bet_info = f"{desc} | imp={imp_prob:.1f}"

        # Dedup
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
            "AND source=:src AND bet_info=:bi"
        ), {"gd": today_str, "src": _SOURCE, "bi": bet_info}).scalar()
        if existing > 0:
            continue

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=odds,
            amount=0.0,
            legs=1,
            bet_info=bet_info,
            predicted_win_prob=imp_prob,  # implied prob only (no model score)
            status="PENDING",
            source=_SOURCE,
            cohort="clean",
            notes=(f"side={leg['side']} point={point} mkt={mkt} "
                   f"unscored=true"),
        )
        db.add(mock)

        mock_leg = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=desc,
            market_type=mkt,
            sport="MLB",
            win_prob=imp_prob,
            fixture_id=leg.get("event_id"),
        )
        db.add(mock_leg)
        stored += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        log.error("F5 unscored commit error: %s", exc)
        return 0

    if stored:
        print(f"[f5] Stored {stored} unscored F5 legs (h2h+totals)")
    return stored


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Historical F5 backtest — does existing ATS signal predict F5 outcomes?
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_ats_on_f5():
    """
    Historical test: does the existing ATS model's signal predict F5 outcomes?

    Uses MLB Stats API linescore to compute F5 scores for all historical games
    in our database, then checks whether our ATS features predict F5 cover.
    """
    import numpy as np

    print("=" * 70)
    print("F5 BACKTEST: Does ATS signal predict First-5-Innings outcomes?")
    print("=" * 70)

    # Load feature matrix with target
    from features import build_feature_matrix, TARGET_ATS
    fm = build_feature_matrix("MLB")
    fm = fm[fm["sport"] == "MLB"].copy() if "sport" in fm.columns else fm

    # We need game_pk to fetch linescores — extract from game_id
    if "game_id" not in fm.columns:
        print("[f5-backtest] No game_id in feature matrix")
        return
    fm["game_pk"] = fm["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")

    # Load F5 scores from historical.db (compute from stored linescores if available)
    conn = sqlite3.connect(_HIST_DB)

    # Check if we have inning-level data
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "game_innings" not in tables:
        print("[f5-backtest] No game_innings table — need to backfill inning scores first")
        print("[f5-backtest] Running backfill_f5_scores()...")
        backfill_f5_scores()

    # Now load F5 scores
    try:
        f5_rows = conn.execute("""
            SELECT game_pk, home_team, away_team, f5_home, f5_away
            FROM game_innings
            WHERE f5_home IS NOT NULL AND f5_away IS NOT NULL
        """).fetchall()
    except Exception:
        print("[f5-backtest] game_innings table not ready yet")
        conn.close()
        return

    conn.close()

    if not f5_rows:
        print("[f5-backtest] No F5 scores available")
        return

    # Build F5 outcome: did home team lead after 5 innings?
    f5_data = {}
    for gpk, home, away, f5h, f5a in f5_rows:
        f5_data[gpk] = {
            "f5_home": f5h, "f5_away": f5a,
            "f5_home_covers_05": 1 if (f5h - f5a + 0.5) > 0 else 0,
            "f5_home_covers_neg05": 1 if (f5h - f5a - 0.5) > 0 else 0,
            "f5_home_ml": 1 if f5h > f5a else (0 if f5h < f5a else None),
        }

    # Merge with feature matrix
    fm["f5_home_covers_05"] = fm["game_pk"].map(
        lambda pk: f5_data.get(pk, {}).get("f5_home_covers_05"))
    fm["f5_home_ml"] = fm["game_pk"].map(
        lambda pk: f5_data.get(pk, {}).get("f5_home_ml"))

    # Filter to games with both features and F5 outcomes
    valid_mask = fm["f5_home_covers_05"].notna()
    if TARGET_ATS in fm.columns:
        valid_mask = valid_mask & fm[TARGET_ATS].notna()
    fm_valid = fm[valid_mask].copy()

    print(f"\n  Games with F5 outcomes: {len(fm_valid)}")
    print(f"  Games with full-game ATS target: {fm_valid[TARGET_ATS].notna().sum()}")

    if len(fm_valid) < 100:
        print("  INSUFFICIENT DATA — need more F5 scores")
        return

    # Load ATS model and score
    from ml_model import _load_ats_submodel
    artifacts = _load_ats_submodel("MLB")
    if artifacts is None:
        print("  Can't load MLB ATS sub-model")
        return
    clf, scaler, imputer, feat_cols = artifacts

    # Filter feat_cols to only base features (no spread augmentation for this test)
    base_feat_cols = [c for c in feat_cols if not c.startswith("_")]

    # Build X matrix
    available = [c for c in base_feat_cols if c in fm_valid.columns]
    missing = [c for c in feat_cols if c not in fm_valid.columns and not c.startswith("_")]
    if missing:
        print(f"  Missing features: {missing[:5]}")

    X = fm_valid[feat_cols].copy() if all(c in fm_valid.columns for c in feat_cols) else None
    if X is None:
        # Build with available features, add spread augmentation
        X = fm_valid[available].copy()
        # Add spread features at default -1.5
        if "_spread_value" in feat_cols:
            X["_spread_value"] = -1.5
        if "_spread_abs" in feat_cols:
            X["_spread_abs"] = 1.5
        if "_is_fav_spread" in feat_cols:
            X["_is_fav_spread"] = 1
        X = X.reindex(columns=feat_cols, fill_value=0)

    X = X.fillna(X.median())

    # Score
    X_scaled = scaler.transform(X) if scaler else X.values
    X_imputed = imputer.transform(X_scaled) if imputer else X_scaled

    probs = clf.predict_proba(X_imputed)[:, 1]  # P(home covers full game)
    fm_valid = fm_valid.copy()
    fm_valid["ats_prob"] = probs

    # Test 1: Does ATS prob predict FULL-GAME cover? (sanity check)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y_full = fm_valid[TARGET_ATS].values.astype(int)
    auc_full = roc_auc_score(y_full, probs)
    brier_full = brier_score_loss(y_full, probs)
    print(f"\n  ATS model → full-game cover: AUC={auc_full:.4f}, Brier={brier_full:.4f}")

    # Test 2: Does ATS prob predict F5 cover (+0.5)?
    y_f5 = fm_valid["f5_home_covers_05"].values.astype(int)
    auc_f5 = roc_auc_score(y_f5, probs)
    brier_f5 = brier_score_loss(y_f5, probs)
    print(f"  ATS model → F5 cover (+0.5):  AUC={auc_f5:.4f}, Brier={brier_f5:.4f}")

    # Test 3: F5 moneyline
    ml_mask = fm_valid["f5_home_ml"].notna()
    if ml_mask.sum() > 100:
        y_ml = fm_valid.loc[ml_mask, "f5_home_ml"].values.astype(int)
        p_ml = fm_valid.loc[ml_mask, "ats_prob"].values
        auc_ml = roc_auc_score(y_ml, p_ml)
        print(f"  ATS model → F5 ML (no ties): AUC={auc_ml:.4f} (N={ml_mask.sum()})")

    # Correlation between full-game and F5 outcomes
    corr = np.corrcoef(y_full, y_f5)[0, 1]
    print(f"\n  r(full_cover, f5_cover) = {corr:.4f}")
    agreement = np.mean(y_full == y_f5)
    print(f"  Agreement rate: {agreement:.3f}")

    # Quintile analysis: sort by ATS prob, check F5 cover rate per quintile
    print(f"\n  ── Quintile analysis (ATS prob → F5 cover) ──")
    print(f"  {'Q':<6} {'ATS prob':<12} {'F5 WR':<10} {'Full WR':<10} {'N'}")
    sorted_idx = np.argsort(probs)
    q_size = len(sorted_idx) // 5
    for i in range(5):
        start = i * q_size
        end = start + q_size if i < 4 else len(sorted_idx)
        idx = sorted_idx[start:end]
        avg_p = np.mean(probs[idx])
        f5_wr = np.mean(y_f5[idx])
        full_wr = np.mean(y_full[idx])
        print(f"  Q{i+1:<5} {avg_p:.4f}      {f5_wr:.3f}     {full_wr:.3f}     {len(idx)}")


def backfill_f5_scores():
    """
    Backfill F5 scores from MLB Stats API for all games in our historical DB.
    Creates/populates game_innings table.
    """
    import requests as _req
    import time as _time

    conn = sqlite3.connect(_HIST_DB)

    # Create table if needed
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_innings (
            game_pk     INTEGER PRIMARY KEY,
            home_team   TEXT,
            away_team   TEXT,
            game_date   TEXT,
            f5_home     INTEGER,
            f5_away     INTEGER,
            total_innings INTEGER,
            full_home   INTEGER,
            full_away   INTEGER
        )
    """)
    conn.commit()

    # Get all game_pks from our data that we don't have F5 scores for yet
    existing = set(r[0] for r in conn.execute(
        "SELECT game_pk FROM game_innings"
    ).fetchall())

    # Get game_pks from team_game_batting (our historical data)
    all_games = conn.execute("""
        SELECT DISTINCT game_pk, game_date, team_name, is_home
        FROM team_game_batting
        WHERE season >= '2023'
        ORDER BY game_date
    """).fetchall()

    # Pivot to game-level
    game_dates = {}
    for gpk, gdate, team, is_home in all_games:
        if gpk not in game_dates:
            game_dates[gpk] = {"date": gdate, "home": None, "away": None}
        if is_home:
            game_dates[gpk]["home"] = team
        else:
            game_dates[gpk]["away"] = team

    need_fetch = {gpk: info for gpk, info in game_dates.items()
                  if gpk not in existing}

    print(f"[f5-backfill] {len(need_fetch)} games need F5 scores "
          f"(of {len(game_dates)} total, {len(existing)} already done)")

    if not need_fetch:
        conn.close()
        return

    # Batch by date for efficient API calls
    dates = set(info["date"] for info in need_fetch.values())
    dates = sorted(dates)

    fetched = 0
    for d in dates:
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=1&date={d}&hydrate=linescore")
        try:
            r = _req.get(url, timeout=15)
            data = r.json()
        except Exception as e:
            log.warning("F5 backfill fetch error for %s: %s", d, e)
            continue

        rows_to_insert = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                gpk = game.get("gamePk")
                if gpk not in need_fetch:
                    continue

                status = game.get("status", {}).get("abstractGameState", "")
                if status not in ("Final", "Completed Early"):
                    continue

                linescore = game.get("linescore") or {}
                innings = linescore.get("innings", [])
                if len(innings) < 5:
                    continue

                home_name = game["teams"]["home"]["team"]["name"]
                away_name = game["teams"]["away"]["team"]["name"]

                f5_home = sum(
                    inn.get("home", {}).get("runs", 0) or 0
                    for inn in innings if inn.get("num", 0) <= 5
                )
                f5_away = sum(
                    inn.get("away", {}).get("runs", 0) or 0
                    for inn in innings if inn.get("num", 0) <= 5
                )
                full_home = linescore.get("teams", {}).get("home", {}).get("runs")
                full_away = linescore.get("teams", {}).get("away", {}).get("runs")

                rows_to_insert.append((
                    gpk, home_name, away_name, d,
                    f5_home, f5_away, len(innings),
                    full_home, full_away,
                ))

        if rows_to_insert:
            conn.executemany("""
                INSERT OR REPLACE INTO game_innings
                  (game_pk, home_team, away_team, game_date,
                   f5_home, f5_away, total_innings,
                   full_home, full_away)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, rows_to_insert)
            conn.commit()
            fetched += len(rows_to_insert)

        _time.sleep(0.2)  # rate limit

    conn.close()
    print(f"[f5-backfill] Stored {fetched} F5 scores")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest_ats_on_f5()
    elif len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill_f5_scores()
    elif len(sys.argv) > 1 and sys.argv[1] == "settle":
        result = settle_f5_bets()
        print(f"Settlement result: {result}")
    else:
        # Default: generate shadow
        from database import SessionLocal, init_db
        init_db()
        db = SessionLocal()
        try:
            legs = generate_f5_shadow(db)
            if legs:
                print(f"\n[f5] Summary: {len(legs)} F5 spread legs scored")
                for leg in sorted(legs, key=lambda l: l.get("win_prob", 0),
                                  reverse=True)[:10]:
                    wp = leg.get("win_prob", 0)
                    desc = f"{leg['pick']} {leg.get('point', '')}"
                    print(f"  wp={wp:5.1f}% odds={leg['odds']:.3f} "
                          f"edge={leg.get('edge_pp', 0):+.1f}pp  {desc}")
        finally:
            db.close()
