"""
odds_api.py — TheOddsAPI integration.
Fetches live fixtures + odds, stores to DB, and computes EV against model predictions.
"""
from __future__ import annotations
import os
import requests
from datetime import datetime, timezone as _utc_tz
from typing import Optional
from zoneinfo import ZoneInfo as _ZoneInfo

from sqlalchemy.orm import Session
from database import Fixture

_CT = _ZoneInfo("America/Chicago")


def _utc_to_ct_date(utc_dt: datetime | None) -> str | None:
    """Convert a UTC-naive or UTC-aware datetime to a CT date string (YYYY-MM-DD)."""
    if utc_dt is None:
        return None
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=_utc_tz.utc)
    return utc_dt.astimezone(_CT).strftime("%Y-%m-%d")


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal. Backend always stores decimal."""
    try:
        n = float(american)
        if n > 0:  return round(1 + n / 100, 4)
        if n < 0:  return round(1 + 100 / abs(n), 4)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return 1.0

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
if not ODDS_API_KEY:
    try:
        from config import ODDS_API_KEY as _cfg_key
        ODDS_API_KEY = _cfg_key
    except Exception:
        pass
BASE_URL     = "https://api.the-odds-api.com/v4"

# ── Dev-mode credit guard ─────────────────────────────────────────────────────
# Set BETIQ_DEV_MODE=true in the environment during all build/test sessions.
# When active, all fetch_odds() / fetch_scores() calls are stubbed with empty
# lists so zero credits are consumed.  Disable only for production verification.
#
#   export BETIQ_DEV_MODE=true   # before uvicorn start
#   unset BETIQ_DEV_MODE         # for production runs
# ─────────────────────────────────────────────────────────────────────────────
DEV_MODE = os.environ.get("BETIQ_DEV_MODE", "").strip().lower() == "true"

# Map our internal sport labels → OddsAPI sport keys
SPORT_KEY_MAP = {
    "Basketball":         "basketball_nba",
    "American Football":  "americanfootball_nfl",
    "Baseball":           "baseball_mlb",
    "Soccer":             "soccer_epl",
    "Ice Hockey":         "icehockey_nhl",
    "ncaab":              "basketball_ncaab",
    "ncaaf":              "americanfootball_ncaaf",
}

# ── Player props deferral note ────────────────────────────────────────────────
# Player-prop markets (pitcher_strikeouts, player_points, etc.) require the
# Business plan ($99/mo) on TheOddsAPI to access via the event-level endpoint.
# Current plan: Creator tier.  Evaluate upgrading on or after 2026-05-14.
# Implementation hook: creator_tier._PROP_MARKETS + fetch_player_props() are
# already wired and scheduled; the upgrade unlocks them automatically.
# ─────────────────────────────────────────────────────────────────────────────

ALL_SPORT_KEYS = [
    "basketball_nba",
    "americanfootball_nfl",
    "baseball_mlb",
    # Soccer — MLS excluded (no sub-model, no CLV data).
    # Re-enable soccer_usa_mls when an MLS-specific model exists.
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_italy_serie_a",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_fifa_world_cup",
    "icehockey_nhl",
    "basketball_ncaab",
    "americanfootball_ncaaf",
    "tennis_atp_french_open",
    # MMA excluded: no personal betting history, no sub-model, ML-only market structure.
    # Re-enable when dedicated MMA model and history exist.
    # "mma_mixed_martial_arts",
]


def _get(endpoint: str, params: dict) -> dict | list | None:
    from http_retry import get_with_retry
    params["apiKey"] = ODDS_API_KEY
    r = get_with_retry(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=10,
        label="OddsAPI",
    )
    if r is None:
        return None
    # Log to credits_log so all API spend is visible (previously unlogged)
    try:
        rem_str  = r.headers.get("x-requests-remaining", "")
        used_str = r.headers.get("x-requests-last", "")
        rem  = int(rem_str)  if rem_str.isdigit()  else None
        used = int(used_str) if used_str.isdigit() else None
        if used is not None or rem is not None:
            from creator_tier import _log_credits
            _log_credits(endpoint, used, rem)
    except Exception:
        pass  # never block a response for logging failure
    return r.json()


# ─── Sports list ──────────────────────────────────────────────────────────────

def list_sports() -> list[dict]:
    """All sports OddsAPI supports (active only)."""
    data = _get("/sports", {"all": "false"})
    return data or []


# ─── Fixtures + odds ──────────────────────────────────────────────────────────

def fetch_odds(sport_key: str, regions: str = "us",
               markets: str = "h2h,spreads,totals",
               odds_format: str = "decimal") -> list[dict]:
    """Fetch upcoming events + odds for one sport."""
    if DEV_MODE:
        print(f"[API] DEV_MODE: skipping fetch_odds({sport_key}) — no credits used")
        return []
    return _get(f"/sports/{sport_key}/odds", {
        "regions":     regions,
        "markets":     markets,
        "oddsFormat":  odds_format,
        "dateFormat":  "iso",
    }) or []


def get_active_sport_keys(requested_keys: list[str]) -> list[str]:
    """
    Filter the requested sport keys to only those currently active
    according to the /sports endpoint, to avoid wasting API quota.
    """
    if DEV_MODE:
        print("[API] DEV_MODE: skipping /sports check — returning all requested keys")
        return list(requested_keys)
    active = _get("/sports", {"all": "false"}) or []
    active_keys = {s["key"] for s in active if not s.get("has_outrights", False)}
    return [k for k in requested_keys if k in active_keys]


def fetch_all_fixtures(db: Session, sport_keys: list[str] | None = None) -> dict:
    """
    Fetch odds for all (or specified) sports and upsert into fixtures table.
    Only fetches sports that are currently active (checked via /sports endpoint).
    Returns summary of what was fetched.
    """
    from line_movement import record_odds_snapshot as _record_snap

    requested = sport_keys or ALL_SPORT_KEYS
    keys = get_active_sport_keys(requested)

    if not keys:
        return {"new": 0, "updated": 0, "sports_fetched": 0,
                "message": "No active sports found right now."}

    total_new = total_updated = 0
    _SNAPSHOT_MARKETS = {"h2h", "spreads", "totals"}
    _SNAPSHOT_BOOK    = "fanduel"

    for sport_key in keys:
        events = fetch_odds(sport_key)
        for event in events:
            existing = db.query(Fixture).filter(Fixture.id == event["id"]).first()
            commence_raw = event.get("commence_time", "")
            try:
                commence = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
            except Exception:
                commence = None

            _gd_ct = _utc_to_ct_date(commence)

            if existing:
                existing.bookmakers   = event.get("bookmakers", [])
                existing.fetched_at   = datetime.utcnow()
                if _gd_ct and not existing.game_date_ct:
                    existing.game_date_ct = _gd_ct
                total_updated += 1
            else:
                fix = Fixture(
                    id            = event["id"],
                    sport_key     = event.get("sport_key", sport_key),
                    sport_title   = event.get("sport_title", ""),
                    home_team     = event.get("home_team", ""),
                    away_team     = event.get("away_team", ""),
                    commence_time = commence,
                    game_date_ct  = _gd_ct,
                    bookmakers    = event.get("bookmakers", []),
                )
                db.add(fix)
                total_new += 1

            # ── Record odds snapshot for line-movement tracking (FanDuel only) ──
            _fid        = event["id"]
            _sport      = event.get("sport_title", sport_key)
            _home       = event.get("home_team", "")
            _away       = event.get("away_team", "")
            for bk in (event.get("bookmakers") or []):
                if bk.get("key") != _SNAPSHOT_BOOK:
                    continue
                for mkt in (bk.get("markets") or []):
                    mkt_key = mkt.get("key", "")
                    if mkt_key not in _SNAPSHOT_MARKETS:
                        continue
                    for outcome in (mkt.get("outcomes") or []):
                        _price = outcome.get("price")
                        _point = outcome.get("point")
                        _name  = outcome.get("name", "")
                        # Normalise side: team name → "home"/"away", Over/Under → "over"/"under"
                        if mkt_key == "totals":
                            _side = _name.lower()  # "over" or "under"
                        elif _name == _home:
                            _side = "home"
                        elif _name == _away:
                            _side = "away"
                        else:
                            _side = _name.lower()
                        try:
                            _record_snap(
                                db,
                                fixture_id    = _fid,
                                sport         = _sport,
                                home_team     = _home,
                                away_team     = _away,
                                market_type   = mkt_key,
                                point         = float(_point) if _point is not None else None,
                                side          = _side,
                                american_odds = None,
                                decimal_odds  = float(_price) if _price is not None else None,
                                book          = _SNAPSHOT_BOOK,
                            )
                        except Exception:
                            pass
                break  # Only one FanDuel block per event

    db.commit()
    return {
        "new":           total_new,
        "updated":       total_updated,
        "sports_fetched":len(keys),
        "sports":        keys,
    }


# ─── Odds extraction helpers ──────────────────────────────────────────────────

def get_best_odds(fixture: Fixture, market: str = "h2h") -> dict:
    """
    Return the best available odds for h2h / spreads / totals
    across all bookmakers in the fixture.
    """
    best: dict[str, float] = {}
    for bk in (fixture.bookmakers or []):
        for mkt in bk.get("markets", []):
            if mkt.get("key") != market:
                continue
            for outcome in mkt.get("outcomes", []):
                name  = outcome.get("name", "")
                price = float(outcome.get("price", 0))
                if name not in best or price > best[name]:
                    best[name] = price
    return best


def fixtures_with_ev(db: Session, min_win_prob: float = 0.0) -> list[dict]:
    """
    Return upcoming fixtures enriched with model-predicted EV.
    Requires a trained model.
    """
    from ml_model import load_model, FEATURE_COLS, predict_bet
    import math

    clf, scaler, _imputer, _active_cols = load_model()
    fixtures = db.query(Fixture).order_by(Fixture.commence_time).all()
    results  = []

    for fix in fixtures:
        if not fix.bookmakers:
            continue
        best_odds = get_best_odds(fix, "h2h")
        if not best_odds:
            continue

        for team, dec_odds in best_odds.items():
            if dec_odds <= 1.0:
                continue
            # Build a minimal feature set for a 1-leg straight bet
            feature_dict = {
                "legs":         1,
                "odds":         dec_odds,
                "log_odds":     math.log(dec_odds),
                "implied_prob": 1 / dec_odds,
                "stake":        10.0,
                "is_parlay":    0,
                "sport_id":     0,
                "ml_pct":       1.0,
                "spread_pct":   0.0,
                "total_pct":    0.0,
                "prop_pct":     0.0,
                "multi_sport":  0,
                "n_sports":     1,
                "has_ev":       0,
                "ev_value":     0.0,
                "closing_line_diff": 0.0,
                "hour_placed":  datetime.utcnow().hour,
                "day_of_week":  datetime.utcnow().weekday(),
                "league_id":    0,
            }
            pred = predict_bet(feature_dict) if clf else {"win_probability": 1/dec_odds, "expected_value": 0}

            results.append({
                "fixture_id":   fix.id,
                "sport":        fix.sport_title,
                "home_team":    fix.home_team,
                "away_team":    fix.away_team,
                "commence_time":fix.commence_time.isoformat() if fix.commence_time else None,
                "pick":         team,
                "decimal_odds": dec_odds,
                "implied_prob": round(1 / dec_odds * 100, 2),
                "model_win_prob":round(pred.get("win_probability", 0) * 100, 2),
                "ev":           pred.get("expected_value", 0),
                "recommendation": pred.get("recommendation", "N/A"),
            })

    # Sort by EV descending
    results.sort(key=lambda x: x["ev"], reverse=True)
    return results


def get_scores(sport_key: str, days_from: int = 1) -> list[dict]:
    """Fetch completed game scores for result settlement."""
    return _get(f"/sports/{sport_key}/scores", {
        "daysFrom": days_from,
        "dateFormat": "iso",
    }) or []
