"""
shadow_teaser.py -- Forward shadow arm for NFL Wong teaser edge validation.

Purpose: validate the Wong DOG teaser structural edge confirmed by blind test
(78.6% per-leg WR on 2024-2025 blind set, N=126, sustained from 76.8% train).
The edge is BET-CONSTRUCTION (margin clustering at 3/7), not prediction.

BLIND TEST RESULT (Aug 2026):
  - DOG legs: Train 76.8% → Blind 78.6% (N=126) — SUSTAINS
  - FAV legs: Train 78.1% → Blind 60.4% (N=48) — REVERTS to near-BE
  - Deploy DOG ONLY. Zero favorite legs.
  - 2-leg parlay WR: 78.6%^2 = 61.7%, at -120 → ROI +13.1%

Architecture:
  - Identifies qualifying Wong teaser legs from NFL closing spreads
  - DOG-ONLY mode (default): only logs underdog legs (+1.5 to +2.5)
  - Two modes:
    a) BACKFILL: populate from nfl.db historical data (2020-2025)
    b) FORWARD: score upcoming NFL games from TheOddsAPI spreads
  - Stores individual teased legs as 1-leg mock_bets, source='shadow_teaser'
  - STRAIGHT bets (1 leg per mock_bet), FLAT stake, $0 paper
  - Settlement: leg covers if actual_margin + teased_spread > 0
  - Notes JSON tracks: original spread, teased spread, wong type, side,
    assumed real-world teaser price, deploy eligibility

Qualifying legs (Wong narrow, DOG ONLY):
  - Dog: closing spread in [+1.5, +2.5] -- teases through both 3 and 7
  - (Favorites [-8.5, -7.5] still logged for tracking but flagged non-deploy)

Price assumptions (from August 2026 sportsbook research):
  - DraftKings/bet365/BetMGM: -120 (best regulated US, DEPLOYABLE)
  - FanDuel/Caesars: -130 to -134 (too expensive)
  - ESPN BET: -150 to -165 dynamic (actively kills Wong legs)
  - Offshore (Bookmaker.eu): -110 to -115
  - Reference price for shadow: -120 (DraftKings, capturable)

Sizing (fractional Kelly on blind-confirmed parlay WR):
  - Per-leg WR (blind): 78.6% for dogs
  - 2-leg parlay WR: 61.7% at -120 (1.833 dec)
  - Kelly fraction: ~14%, deploy at 3-5% (quarter-Kelly for N uncertainty)
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_CT = ZoneInfo("America/Chicago")
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_BETS_DB = os.path.join(_DATA_DIR, "bets.db")
_NFL_DB = os.path.join(_DATA_DIR, "nfl.db")

# Teaser parameters
_TEASER_POINTS = 6.0

# Wong narrow qualifying ranges (closing spread, team-perspective)
_WONG_FAV_MIN = -8.5  # most negative favorite spread
_WONG_FAV_MAX = -7.5  # least negative favorite spread
_WONG_DOG_MIN = 1.5   # least positive dog spread
_WONG_DOG_MAX = 2.5   # most positive dog spread

# Sportsbook teaser pricing (2-team 6-point, from Aug 2026 research)
# Key: book name -> {"price_american": int, "deployable": bool}
TEASER_PRICING = {
    "draftkings":   {"price_am": -120, "per_leg_be": 0.7385, "deployable": True},
    "bet365":       {"price_am": -120, "per_leg_be": 0.7385, "deployable": True},
    "betmgm":       {"price_am": -120, "per_leg_be": 0.7385, "deployable": True},
    "fanduel":      {"price_am": -134, "per_leg_be": 0.7579, "deployable": False},
    "caesars":      {"price_am": -130, "per_leg_be": 0.7516, "deployable": False},
    "espnbet":      {"price_am": -155, "per_leg_be": 0.7810, "deployable": False},
    "bookmaker_eu": {"price_am": -110, "per_leg_be": 0.7237, "deployable": True},
}

# Reference price for shadow tracking (DraftKings -120, best regulated US)
_REF_PRICE_AM = -120
_REF_PRICE_DEC = 1.0 + 100.0 / 120.0   # 1.8333
_REF_PER_LEG_BE = 0.7385


def _is_wong_narrow(spread: float) -> bool:
    """Check if a closing spread qualifies for Wong narrow teaser."""
    return (_WONG_FAV_MIN <= spread <= _WONG_FAV_MAX) or \
           (_WONG_DOG_MIN <= spread <= _WONG_DOG_MAX)


def _is_wong_dog(spread: float) -> bool:
    """True if spread qualifies as Wong narrow DOG (the deployable side)."""
    return _WONG_DOG_MIN <= spread <= _WONG_DOG_MAX


def _is_wong_broad(spread: float) -> bool:
    """Check if a closing spread qualifies for Wong broad teaser."""
    return (-9.0 <= spread <= -7.0) or (1.0 <= spread <= 3.0)


def _teased_spread(spread: float) -> float:
    """Apply 6-point teaser to spread (always in bettor's favor)."""
    return spread + _TEASER_POINTS


def _covers_teased(actual_margin: float, teased: float) -> str:
    """Check if actual margin covers the teased spread.

    Returns 'WIN', 'LOSS', or 'PUSH'.
    Cover condition: actual_margin + teased_spread > 0
    """
    result = actual_margin + teased
    if abs(result) < 0.001:
        return "PUSH"
    return "WIN" if result > 0 else "LOSS"


def generate_teaser_shadow_backfill(db=None):
    """Backfill shadow_teaser from all historical NFL data in nfl.db.

    This populates mock_bets with settled Wong teaser legs from 2020-2025,
    providing the historical baseline for the forward shadow comparison.
    Returns count of legs stored.
    """
    if not os.path.exists(_NFL_DB):
        print("[shadow_teaser] nfl.db not found -- run nfl_foundation.py first")
        return 0

    nfl_conn = sqlite3.connect(_NFL_DB)
    nfl_conn.row_factory = sqlite3.Row

    rows = nfl_conn.execute("""
        SELECT f.game_id, f.team, f.opp_team, f.season, f.week,
               f.game_date, f.close_spread, f.is_home,
               g.home_team, g.away_team, g.home_score, g.away_score
        FROM nfl_features f
        JOIN nfl_games g ON f.game_id = g.game_id
        WHERE f.close_spread IS NOT NULL
          AND g.home_score IS NOT NULL
        ORDER BY f.game_date, f.game_id
    """).fetchall()
    nfl_conn.close()

    print(f"[shadow_teaser] Backfill: {len(rows)} feature rows from nfl.db")

    legs_to_store = []
    for r in rows:
        spread = r["close_spread"]
        if not _is_wong_narrow(spread):
            continue

        team = r["team"]
        if team == r["home_team"]:
            actual_margin = r["home_score"] - r["away_score"]
        else:
            actual_margin = r["away_score"] - r["home_score"]

        teased = _teased_spread(spread)
        result = _covers_teased(actual_margin, teased)
        side = "fav" if spread < 0 else "dog"

        legs_to_store.append({
            "game_id": r["game_id"],
            "team": team,
            "opp_team": r["opp_team"],
            "season": r["season"],
            "week": r["week"],
            "game_date": r["game_date"],
            "original_spread": spread,
            "teased_spread": teased,
            "actual_margin": actual_margin,
            "result": result,
            "side": side,
            "is_home": r["is_home"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
        })

    print(f"[shadow_teaser] {len(legs_to_store)} Wong narrow qualifying legs")

    if db is not None and legs_to_store:
        stored = _store_teaser_legs(db, legs_to_store, backfill=True)
        print(f"[shadow_teaser] Stored {stored} backfill legs")
        return stored

    return len(legs_to_store)


def generate_teaser_shadow(db=None, today_str=None, dog_only=True):
    """Score this week's NFL games for Wong teaser qualifying legs.

    For forward tracking: uses TheOddsAPI NFL spreads to identify
    qualifying legs BEFORE games are played. After settlement, the
    leg_result is updated by the settlement pipeline.

    Falls back to nfl_data_py closing lines for post-game retrospective.

    Args:
        dog_only: If True (default), only generate DOG legs (+1.5 to +2.5).
                  Favorites collapsed blind (60.4% vs 78.1% train) — not deployable.
                  Still logs all qualifying legs for tracking, but flags fav legs
                  with deploy_eligible=False in notes.

    Returns list of qualifying leg dicts.
    """
    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    mode_str = "DOG-ONLY" if dog_only else "all sides"
    print(f"\n[shadow_teaser] Generating Wong teaser shadow for {today_str} ({mode_str})")

    qualifying = []

    # Strategy 1: Check TheOddsAPI for NFL spreads (forward, pre-game)
    try:
        qualifying = _get_legs_from_odds_api(today_str)
    except Exception as exc:
        print(f"[shadow_teaser] OddsAPI lookup failed: {exc}")

    # Strategy 2: Fall back to nfl.db closing lines (post-game retrospective)
    if not qualifying and os.path.exists(_NFL_DB):
        qualifying = _get_legs_from_nfl_db(today_str)

    if not qualifying:
        print("[shadow_teaser] No qualifying Wong legs found")
        return []

    # Tag deploy eligibility based on blind test results
    for leg in qualifying:
        leg["deploy_eligible"] = _is_wong_dog(leg["original_spread"])

    dogs = sum(1 for l in qualifying if l["deploy_eligible"])
    favs = len(qualifying) - dogs
    print(f"[shadow_teaser] {len(qualifying)} qualifying legs: "
          f"{dogs} dogs (DEPLOY), {favs} favs (track-only)")

    # If dog_only, filter to dogs for storage (still log favs for comparison)
    if dog_only:
        deploy_legs = [l for l in qualifying if l["deploy_eligible"]]
        track_legs = [l for l in qualifying if not l["deploy_eligible"]]
    else:
        deploy_legs = qualifying
        track_legs = []

    # Store deploy-eligible legs
    if db is not None and deploy_legs:
        stored = _store_teaser_legs(db, deploy_legs, backfill=False)
        print(f"[shadow_teaser] Stored {stored} DOG shadow legs (deploy-eligible)")

    # Also store fav legs for tracking comparison (non-deploy)
    if db is not None and track_legs:
        stored_track = _store_teaser_legs(db, track_legs, backfill=False)
        print(f"[shadow_teaser] Stored {stored_track} FAV legs (track-only, not deployable)")

    return qualifying


def _get_legs_from_odds_api(today_str: str) -> list:
    """Pull NFL spreads from TheOddsAPI fixtures table for today's games."""
    try:
        bets_conn = sqlite3.connect(_BETS_DB)
        bets_conn.row_factory = sqlite3.Row

        # Check for NFL fixtures with closing spreads
        rows = bets_conn.execute("""
            SELECT id, sport_key, home_team, away_team, commence_time,
                   bookmakers
            FROM fixtures
            WHERE sport_key = 'americanfootball_nfl'
              AND date(commence_time) = ?
        """, (today_str,)).fetchall()
        bets_conn.close()

        if not rows:
            return []

        qualifying = []
        for r in rows:
            bookmakers_raw = r["bookmakers"]
            if not bookmakers_raw:
                continue

            try:
                bookmakers = json.loads(bookmakers_raw)
            except Exception:
                continue

            # Find the consensus/pinnacle closing spread
            spread = _extract_best_spread(bookmakers, r["home_team"])
            if spread is None:
                continue

            # Check both sides
            home_spread = spread
            away_spread = -spread

            for team, sp, is_home in [
                (r["home_team"], home_spread, 1),
                (r["away_team"], away_spread, 0),
            ]:
                if not _is_wong_narrow(sp):
                    continue

                opp = r["away_team"] if is_home else r["home_team"]
                qualifying.append({
                    "game_id": r["id"],
                    "team": team,
                    "opp_team": opp,
                    "season": None,  # filled by settlement
                    "week": None,
                    "game_date": today_str,
                    "original_spread": sp,
                    "teased_spread": _teased_spread(sp),
                    "actual_margin": None,  # PENDING
                    "result": None,
                    "side": "fav" if sp < 0 else "dog",
                    "is_home": is_home,
                    "home_score": None,
                    "away_score": None,
                })

        return qualifying

    except Exception as exc:
        print(f"[shadow_teaser] OddsAPI fixtures error: {exc}")
        return []


def _extract_best_spread(bookmakers: list, home_team: str) -> float | None:
    """Extract home team closing spread from bookmakers JSON.

    Prefers pinnacle > draftkings > fanduel > first available.
    """
    priority = ["pinnacle", "draftkings", "fanduel", "betmgm"]
    spreads_by_book = {}

    for bk in bookmakers:
        bk_key = bk.get("key", "").lower()
        for market in bk.get("markets", []):
            if market.get("key") != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_team:
                    point = outcome.get("point")
                    if point is not None:
                        spreads_by_book[bk_key] = float(point)

    for prio in priority:
        if prio in spreads_by_book:
            return spreads_by_book[prio]

    # Fall back to first available
    if spreads_by_book:
        return next(iter(spreads_by_book.values()))

    return None


def _get_legs_from_nfl_db(today_str: str) -> list:
    """Pull qualifying Wong legs from nfl.db for a specific date."""
    nfl_conn = sqlite3.connect(_NFL_DB)
    nfl_conn.row_factory = sqlite3.Row

    rows = nfl_conn.execute("""
        SELECT f.game_id, f.team, f.opp_team, f.season, f.week,
               f.game_date, f.close_spread, f.is_home,
               g.home_team, g.away_team, g.home_score, g.away_score
        FROM nfl_features f
        JOIN nfl_games g ON f.game_id = g.game_id
        WHERE f.close_spread IS NOT NULL
          AND f.game_date = ?
    """, (today_str,)).fetchall()
    nfl_conn.close()

    qualifying = []
    for r in rows:
        spread = r["close_spread"]
        if not _is_wong_narrow(spread):
            continue

        actual_margin = None
        result = None
        if r["home_score"] is not None:
            team = r["team"]
            if team == r["home_team"]:
                actual_margin = r["home_score"] - r["away_score"]
            else:
                actual_margin = r["away_score"] - r["home_score"]
            teased = _teased_spread(spread)
            result = _covers_teased(actual_margin, teased)

        qualifying.append({
            "game_id": r["game_id"],
            "team": r["team"],
            "opp_team": r["opp_team"],
            "season": r["season"],
            "week": r["week"],
            "game_date": r["game_date"],
            "original_spread": spread,
            "teased_spread": _teased_spread(spread),
            "actual_margin": actual_margin,
            "result": result,
            "side": "fav" if spread < 0 else "dog",
            "is_home": r["is_home"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
        })

    return qualifying


def _store_teaser_legs(db, legs: list, backfill: bool = False) -> int:
    """Persist Wong teaser legs as 1-leg mock_bets."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    run_id = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc)
    stored = 0

    for leg in legs:
        mock_id = str(uuid.uuid4())
        team = leg["team"]
        opp = leg["opp_team"]
        spread = leg["original_spread"]
        teased = leg["teased_spread"]
        side = leg["side"]
        game_date = leg["game_date"]
        result = leg.get("result")

        desc = (f"{team} {spread:+.1f} -> {teased:+.1f} "
                f"(Wong teaser, {side})")
        bet_info = f"{desc} | {side}"

        # Dedup
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
            "AND source='shadow_teaser' AND bet_info=:bi"
        ), {"gd": game_date, "bi": bet_info}).scalar()
        if existing > 0:
            continue

        # Status
        if result == "WIN":
            status = "SETTLED_WIN"
        elif result == "LOSS":
            status = "SETTLED_LOSS"
        elif result == "PUSH":
            status = "SETTLED_PUSH"
        else:
            status = "PENDING"

        # PnL at reference price (-120 = 1.8333 decimal)
        actual_profit = None
        if status == "SETTLED_WIN":
            actual_profit = 10.0 * (_REF_PRICE_DEC - 1.0)  # win $8.33 on $10
        elif status == "SETTLED_LOSS":
            actual_profit = -10.0

        deploy_eligible = leg.get("deploy_eligible", _is_wong_dog(spread))
        # Blind-confirmed per-leg WR: dogs 78.6%, favs 60.4%
        blind_wr = 78.6 if side == "dog" else 60.4

        notes_dict = {
            "wong_type": "narrow",
            "side": side,
            "deploy_eligible": deploy_eligible,
            "original_spread": spread,
            "teased_spread": teased,
            "teaser_points": _TEASER_POINTS,
            "ref_price_am": _REF_PRICE_AM,
            "ref_price_dec": round(_REF_PRICE_DEC, 4),
            "ref_per_leg_be": _REF_PER_LEG_BE,
            "blind_per_leg_wr": blind_wr,
            "actual_margin": leg.get("actual_margin"),
            "season": leg.get("season"),
            "week": leg.get("week"),
            "is_backfill": backfill,
            "pricing": {
                "draftkings": -120,
                "bet365": -120,
                "betmgm": -120,
                "fanduel": -134,
                "bookmaker_eu": -110,
            },
        }
        notes_json = json.dumps(notes_dict, separators=(",", ":"))

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            generated_at=now_utc,
            game_date=game_date,
            sport="NFL",
            bet_type="straight",
            odds=_REF_PRICE_DEC,
            amount=0.0,
            legs=1,
            bet_info=bet_info,
            predicted_win_prob=blind_wr,
            model_used="wong_teaser_structural",
            status=status,
            source="shadow_teaser",
            cohort="clean",
            notes=notes_json,
            actual_profit=actual_profit,
        )
        db.add(mock)

        mock_leg = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=desc,
            market_type="teaser_6pt",
            sport="NFL",
            win_prob=blind_wr,
            fixture_id=str(leg.get("game_id", "")),
            is_alt_line=False,
            leg_result=result,
            resolved_home_score=leg.get("home_score"),
            resolved_away_score=leg.get("away_score"),
            resolved_margin=leg.get("actual_margin"),
            resolved_adjusted_margin=(
                (leg["actual_margin"] + teased)
                if leg.get("actual_margin") is not None else None
            ),
        )
        db.add(mock_leg)
        stored += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[shadow_teaser] Commit error: {exc}")
        return 0

    return stored


def settle_teaser_legs(db):
    """Settle pending shadow_teaser legs using nfl.db scores.

    Called by the scheduler after NFL games complete.
    """
    from sqlalchemy import text as _text

    # Get pending teaser legs
    pending = db.execute(_text("""
        SELECT mb.id, mb.game_date, mb.notes,
               mbl.id as leg_id, mbl.fixture_id, mbl.description
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.source = 'shadow_teaser'
          AND mb.status = 'PENDING'
    """)).fetchall()

    if not pending:
        return 0

    # Load NFL scores
    if not os.path.exists(_NFL_DB):
        print("[shadow_teaser] nfl.db not found for settlement")
        return 0

    nfl_conn = sqlite3.connect(_NFL_DB)
    nfl_conn.row_factory = sqlite3.Row

    scores = {}
    for row in nfl_conn.execute("""
        SELECT game_id, home_team, away_team, home_score, away_score
        FROM nfl_games WHERE home_score IS NOT NULL
    """).fetchall():
        scores[str(row["game_id"])] = dict(row)

    nfl_conn.close()

    settled = 0
    for row in pending:
        game_id = row.fixture_id if hasattr(row, 'fixture_id') else row[4]
        mock_id = row.id if hasattr(row, 'id') else row[0]
        leg_id = row.leg_id if hasattr(row, 'leg_id') else row[3]
        notes_raw = row.notes if hasattr(row, 'notes') else row[2]

        if str(game_id) not in scores:
            continue

        game = scores[str(game_id)]
        try:
            notes = json.loads(notes_raw or "{}")
        except Exception:
            notes = {}

        teased = notes.get("teased_spread")
        team = notes.get("side")  # need team from description
        original_spread = notes.get("original_spread")

        if teased is None or original_spread is None:
            continue

        # Determine actual margin from the description
        desc = row.description if hasattr(row, 'description') else row[5]
        # Description format: "TEAM +2.5 -> +8.5 (Wong teaser, dog)"
        team_name = desc.split(" ")[0] if desc else None
        if not team_name:
            continue

        if team_name == game["home_team"]:
            actual_margin = game["home_score"] - game["away_score"]
        elif team_name == game["away_team"]:
            actual_margin = game["away_score"] - game["home_score"]
        else:
            continue

        result = _covers_teased(actual_margin, teased)

        if result == "WIN":
            status = "SETTLED_WIN"
            profit = 10.0 * (_REF_PRICE_DEC - 1.0)
        elif result == "LOSS":
            status = "SETTLED_LOSS"
            profit = -10.0
        else:
            status = "SETTLED_PUSH"
            profit = 0.0

        db.execute(_text(
            "UPDATE mock_bets SET status=:st, actual_profit=:pr, "
            "settled_at=:sa WHERE id=:mid"
        ), {"st": status, "pr": profit, "sa": datetime.now(timezone.utc),
            "mid": mock_id})

        db.execute(_text(
            "UPDATE mock_bet_legs SET leg_result=:lr, "
            "resolved_home_score=:hs, resolved_away_score=:as_, "
            "resolved_margin=:rm, resolved_adjusted_margin=:ram "
            "WHERE id=:lid"
        ), {"lr": result, "hs": game["home_score"],
            "as_": game["away_score"], "rm": actual_margin,
            "ram": actual_margin + teased, "lid": leg_id})

        settled += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[shadow_teaser] Settlement commit error: {exc}")
        return 0

    print(f"[shadow_teaser] Settled {settled} teaser legs")
    return settled


if __name__ == "__main__":
    from database import SessionLocal, init_db
    init_db()
    dbs = SessionLocal()
    try:
        # Backfill from historical data
        count = generate_teaser_shadow_backfill(dbs)
        print(f"\n[shadow_teaser] Backfill complete: {count} legs stored")
    finally:
        dbs.close()
