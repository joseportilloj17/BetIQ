"""
live_monitor.py — Live Bet Monitor: resolve PLACED bets and generate cash-out advice.

Public API:
  resolve_placed_bets(db, hist_db_path, auto_settle) → list[dict]
  evaluate_cashout(bet_row, resolved_legs, db)        → dict
  log_cashout_recommendation(bet, rec_dict, db)       → None
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text as _sqla_text

from database import Bet, BetLeg, Fixture, SessionLocal
import leg_resolver as lr
import odds_api

# ── Paths & constants ──────────────────────────────────────────────────────────
_HIST_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

_SPORT_KEY_FOR_SCORES: dict[str, str] = {
    "MLB": "baseball_mlb",
    "NBA": "basketball_nba",
    "NHL": "icehockey_nhl",
    "NFL": "americanfootball_nfl",
}

# OddsAPI sport_key → internal sport label (reverse of the above)
_SPORT_FROM_KEY: dict[str, str] = {v: k for k, v in _SPORT_KEY_FOR_SCORES.items()}

# Try sports in this order when bet.sports field is wrong
_SPORT_PRIORITY = ["MLB", "NBA", "NHL", "NFL"]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _plain_conn(path: str | None = None) -> sqlite3.Connection:
    """Plain SQLite connection (no row_factory) — compatible with leg_resolver._find_game."""
    return sqlite3.connect(path or _HIST_DB_PATH)


def _parse_compact_leg(desc: str) -> dict:
    """
    Parse the compact BetIQ-app leg format: '{team} +/-line {line} ({market}) {odds}'
    Used when bet_info has no ' @ ' matchup separator.

    Examples handled:
      'Cubs  +2.5 +2.5 (Alt Spread) -390'
      'Dodgers (Moneyline) -310'
      'Yankees +1.5 alt sprad +1.5 (Spread) -330'
      'Red Sox +2.5 +2.5 (Alt Spread) -390'
    """
    text = desc.strip()

    # 1 — trailing American odds  (-390, +168, -110 …)
    odds: Optional[int] = None
    m = re.search(r'\s([+-]\d{2,4})\s*$', text)
    if m:
        odds = int(m.group(1))
        text = text[:m.start()].strip()

    # 2 — market from parentheses, then fallback to raw keyword scan
    market_type = "Spread"
    # ── Soccer-specific detection (highest priority — checked before parentheses) ──
    tl_early = text.lower()
    if "double chance" in tl_early:
        market_type = "Double Chance"
    elif "both teams to score" in tl_early or tl_early.startswith("btts"):
        market_type = "BTTS"
    elif "corner" in tl_early:
        market_type = "Corners"
    elif re.search(r'\bgoals?\b', tl_early) and re.search(r'\b(over|under)\b', tl_early):
        market_type = "Total"
    elif re.search(r'\bgoals?\b', tl_early) and "to score" not in tl_early:
        market_type = "Total"

    if market_type not in ("Double Chance", "BTTS", "Corners", "Total"):
        m = re.search(r'\(([^)]+)\)', text)
        if m:
            mkt_raw = m.group(1).lower()
            if "moneyline" in mkt_raw or "3-way" in mkt_raw:
                market_type = "Moneyline"
            elif "total" in mkt_raw or "over" in mkt_raw or "under" in mkt_raw:
                market_type = "Total"
            elif "prop" in mkt_raw:
                market_type = "Player Prop"
            else:
                market_type = "Spread"
            text = re.sub(r'\s*\([^)]+\)\s*', ' ', text).strip()
        else:
            # No parentheses — check the raw text for market keywords
            tl = text.lower()
            if "moneyline" in tl:
                market_type = "Moneyline"
                text = re.sub(r'\bmoneyline\b', '', text, flags=re.I).strip()
            elif "run line" in tl or "puck line" in tl or "alt spread" in tl:
                market_type = "Spread"
                text = re.sub(r'\b(run line|puck line|alt spread)\b', '', text, flags=re.I).strip()
            elif "over " in tl or "under " in tl or "total" in tl:
                market_type = "Total"
            # else stays "Spread" (default for spread/run-line legs without keywords)

    # 3 — team name = leading words before first numeric line token
    _JUNK = {"alt", "sprad", "spread", "alternate", "run", "line", "puck", "sprd"}
    team_parts: list[str] = []
    line: Optional[float] = None
    direction: Optional[str] = None

    for token in text.split():
        tok_lower = token.lower()
        # Numeric token: [+-]N.N (spread/total line)
        m_num = re.match(r'^([+-]?\d+\.?\d*)\+?$', token)
        if m_num and ('+' in token or '-' in token or '.' in token):
            if line is None:
                try:
                    line = float(m_num.group(1))
                    direction = "+" if line >= 0 else "-"
                except ValueError:
                    pass
            # once we've seen a line, stop collecting team tokens
        elif line is None and tok_lower not in _JUNK:
            team_parts.append(token)

    team_raw = " ".join(team_parts).strip()
    selected = lr.normalize_team(team_raw) if team_raw else ""

    return {
        "selected_team_or_player": selected,
        "market_type":  market_type,
        "line":         line,
        "direction":    direction,
        "odds":         odds,
        "away_team":    "",
        "home_team":    "",
        "stat_type":    None,
    }


def _infer_sport_for_team(team_canonical: str, stated_sport: str, cur: sqlite3.Cursor) -> str:
    """
    Return the sport that actually has this team in historical.db.
    Falls back to stated_sport if nothing found.
    """
    stated = lr.infer_sport("", stated_sport) or ""

    # Try stated sport first (fast path when correct)
    if stated in _SPORT_PRIORITY:
        db_team = lr.to_db_team(team_canonical, stated)
        try:
            cur.execute(
                "SELECT 1 FROM games WHERE sport=? AND (home_team=? OR away_team=? "
                "OR home_team LIKE ? OR away_team LIKE ?) LIMIT 1",
                (stated, db_team, db_team, f"%{db_team}%", f"%{db_team}%"),
            )
            if cur.fetchone():
                return stated
        except Exception:
            pass

    # Try each sport in priority order
    for sport in _SPORT_PRIORITY:
        if sport == stated:
            continue
        db_team = lr.to_db_team(team_canonical, sport)
        try:
            cur.execute(
                "SELECT 1 FROM games WHERE sport=? AND (home_team=? OR away_team=? "
                "OR home_team LIKE ? OR away_team LIKE ?) LIMIT 1",
                (sport, db_team, db_team, f"%{db_team}%", f"%{db_team}%"),
            )
            if cur.fetchone():
                return sport
        except Exception:
            pass

    return stated or "MLB"


def _lookup_commence_time(
    team_canonical: str,
    sport: str,
    time_placed: datetime,
    db: Session,
) -> Optional[datetime]:
    """
    Find the fixture commence_time for a leg's game.

    Searches the fixtures table for games involving team_canonical in the window
    (time_placed - 72h, time_placed + 4d) and returns the one whose commence_time
    is CLOSEST to time_placed — not necessarily the earliest.

    72-hour lookback handles retroactively-logged bets (placed in FanDuel before
    the game, logged in BetIQ after). Closest-to-placement selection ensures that
    when the same teams play two days in a row the correct game is chosen.

    Staleness guard: rejects fixtures more than 72 hours before time_placed.
    """
    sport_key = _SPORT_KEY_FOR_SCORES.get(sport, "")
    if not sport_key or not team_canonical:
        return None

    # Use the last word of the canonical name as a robust LIKE match
    # (e.g. "Pittsburgh Pirates" → "Pirates", "Los Angeles Dodgers" → "Dodgers")
    nickname = team_canonical.strip().split()[-1] if team_canonical.strip() else ""
    if not nickname or len(nickname) < 3:
        return None

    window_start = time_placed - timedelta(hours=72)
    window_end   = time_placed + timedelta(days=4)

    try:
        from sqlalchemy import or_
        fixtures = (
            db.query(Fixture)
            .filter(
                Fixture.sport_key == sport_key,
                Fixture.commence_time >= window_start,
                Fixture.commence_time <= window_end,
                or_(
                    Fixture.home_team.ilike(f"%{nickname}%"),
                    Fixture.away_team.ilike(f"%{nickname}%"),
                ),
            )
            .all()
        )
        if not fixtures:
            return None

        def _to_dt(val):
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(str(val).replace("Z", "").split("+")[0])

        _3H = timedelta(hours=3)

        # Split into preferred (pregame or logged within 3h of start) vs past
        preferred = [
            f for f in fixtures
            if _to_dt(f.commence_time) >= time_placed           # pregame — always preferred
            or time_placed - _to_dt(f.commence_time) <= _3H     # ≤3h post-start — still same game
        ]

        if preferred:
            # Pick the preferred fixture closest to time_placed.
            # When the same team plays two days in a row this naturally selects
            # the right game: pregame bets pick the imminent game, in-game logs
            # (≤3h) stay on today's game.
            best = min(preferred, key=lambda f: abs((_to_dt(f.commence_time) - time_placed).total_seconds()))
        else:
            # Logged >3h after all fixtures in window already started.
            # No upcoming game found → retroactive log, fall back to the most
            # recent past game so the resolver can still close it out.
            past = [f for f in fixtures if _to_dt(f.commence_time) < time_placed]
            if not past:
                return None
            best = min(past, key=lambda f: (time_placed - _to_dt(f.commence_time)).total_seconds())

        ct = _to_dt(best.commence_time)

        # Staleness guard: reject if still more than 72h before placement
        if ct < window_start:
            return None

        return ct
    except Exception:
        return None


def _resolve_leg_from_historical(
    parsed: dict,
    game_date: str,
    sport: str,
    cur: sqlite3.Cursor,
) -> dict:
    """Try to resolve a parsed leg from historical.db. Returns status dict."""
    selected = parsed.get("selected_team_or_player") or ""
    away     = parsed.get("away_team") or ""
    home     = parsed.get("home_team") or ""

    db_sel  = lr.to_db_team(lr.normalize_team(selected), sport) if selected else ""
    db_away = lr.to_db_team(lr.normalize_team(away), sport)     if away     else ""
    db_home = lr.to_db_team(lr.normalize_team(home), sport)     if home     else ""

    # Prefer full matchup lookup; fall back to single-team
    game = None
    if db_away and db_home:
        game = lr._find_game(cur, sport, db_away, db_home, game_date)
    if game is None and db_sel:
        game = lr._find_game(cur, sport, db_sel, "", game_date)
    if game is None and db_away:
        game = lr._find_game(cur, sport, db_away, "", game_date)
    if game is None:
        # Try with window=2 days
        if db_away and db_home:
            game = lr._find_game(cur, sport, db_away, db_home, game_date, window=2)
        if game is None and db_sel:
            game = lr._find_game(cur, sport, db_sel, "", game_date, window=2)

    if game is None:
        return {"status": "UNKNOWN", "reason": "game not found in historical.db"}

    res = lr._resolve_from_game(parsed, game, sport)
    leg_result = res.get("leg_result")

    return {
        "status":           {"WIN": "WON", "LOSS": "LOST", "PUSH": "PUSH"}.get(leg_result or "", "UNKNOWN"),
        "result":           leg_result,
        "accuracy_delta":   res.get("accuracy_delta"),
        "actual_value":     res.get("actual_value"),
        "resolution_source": res.get("resolution_source", "historical_db"),
        "game": {
            "home":       game[0],
            "away":       game[1],
            "home_score": game[2],
            "away_score": game[3],
            "date":       game[4],
        },
    }


def _resolve_leg_from_scores_api(
    parsed: dict,
    game_date: str,
    sport: str,
) -> dict:
    """Fallback: resolve via TheOddsAPI /scores endpoint for very recent games."""
    sport_key = _SPORT_KEY_FOR_SCORES.get(sport)
    if not sport_key:
        return {"status": "UNKNOWN", "reason": f"no scores key for {sport}"}

    try:
        scores = odds_api.get_scores(sport_key, days_from=3)
    except Exception:
        return {"status": "UNKNOWN", "reason": "scores API error"}

    if not scores:
        return {"status": "UNKNOWN", "reason": "no scores returned"}

    selected = (parsed.get("selected_team_or_player") or "").lower()
    away     = (parsed.get("away_team") or "").lower()
    home     = (parsed.get("home_team") or "").lower()

    matched = None
    for g in scores:
        if not g.get("completed"):
            continue
        g_home = (g.get("home_team") or "").lower()
        g_away = (g.get("away_team") or "").lower()

        home_hit = home and (home in g_home or g_home in home)
        away_hit = away and (away in g_away or g_away in away)
        sel_hit  = selected and (selected in g_home or selected in g_away
                                 or g_home in selected or g_away in selected)

        if (home_hit and away_hit) or (not away and sel_hit):
            matched = g
            break

    if not matched:
        return {"status": "UNKNOWN", "reason": "game not found in scores API"}

    scores_map = {s["name"]: s["score"] for s in (matched.get("scores") or []) if s.get("score")}
    g_home = matched.get("home_team", "")
    g_away = matched.get("away_team", "")
    try:
        hs = int(float(scores_map.get(g_home, 0)))
        as_ = int(float(scores_map.get(g_away, 0)))
    except (ValueError, TypeError):
        return {"status": "UNKNOWN", "reason": "invalid scores from API"}

    game_tuple = (g_home, g_away, hs, as_, game_date)
    res = lr._resolve_from_game(parsed, game_tuple, sport)
    leg_result = res.get("leg_result")
    return {
        "status":            {"WIN": "WON", "LOSS": "LOST", "PUSH": "PUSH"}.get(leg_result or "", "UNKNOWN"),
        "result":            leg_result,
        "accuracy_delta":    res.get("accuracy_delta"),
        "resolution_source": "odds_api_scores",
    }


def _resolve_single_leg(
    desc: str,
    game_date: str,
    stated_sport: str,
    cur: sqlite3.Cursor,
    expected_game_start: Optional[datetime] = None,
) -> dict:
    """
    Resolve one pipe-delimited leg description.
    Returns enriched result dict including parsed info for display.

    expected_game_start — when provided:
      • If still in the future → return UNKNOWN immediately (pregame guard)
      • Otherwise use its date as the game_date for historical.db lookup
    """
    now = datetime.utcnow()

    # ── Pregame guard ─────────────────────────────────────────────────────────
    # Parse first so we have team info for the returned dict even when guarded.
    has_matchup = " @ " in desc or " v " in desc or " vs " in desc

    if has_matchup:
        try:
            parsed = lr.parse_leg_details(desc)
        except Exception as exc:
            return {"status": "UNKNOWN", "reason": f"parse error: {exc}", "parsed": {}}
        # parse_leg_details strips market keywords before classification, so it can
        # mis-classify legs whose market word (e.g. "Moneyline") appears mid-string.
        # Recover the correct type from the raw description when result is "Other".
        if parsed.get("market_type") == "Other":
            dl = desc.lower()
            if "double chance" in dl:
                parsed["market_type"] = "Double Chance"
            elif "both teams to score" in dl or dl.startswith("btts"):
                parsed["market_type"] = "BTTS"
            elif "corner" in dl:
                parsed["market_type"] = "Corners"
            elif "moneyline" in dl or "3-way" in dl:
                parsed["market_type"] = "Moneyline"
            elif "run line" in dl or "puck line" in dl or "spread" in dl:
                parsed["market_type"] = "Spread"
            elif "over" in dl or "under" in dl or "total" in dl or "goal" in dl:
                parsed["market_type"] = "Total"
    else:
        parsed = _parse_compact_leg(desc)

    selected = parsed.get("selected_team_or_player") or ""

    # Infer correct sport for this leg (handles wrong bet.sports field)
    canonical = lr.normalize_team(selected) if selected else ""
    sport = _infer_sport_for_team(canonical, stated_sport, cur)

    if expected_game_start is not None and expected_game_start > now:
        return {
            "status":   "UNKNOWN",
            "reason":   f"pregame — scheduled {expected_game_start.strftime('%Y-%m-%d %H:%M')} UTC",
            "parsed":   parsed,
            "sport":    sport,
        }

    # ── Choose the authoritative game date ────────────────────────────────────
    # Use fixture commence_time date when available — handles bets placed the
    # night before a day game or multi-day parlays.
    if expected_game_start is not None:
        effective_date = expected_game_start.strftime("%Y-%m-%d")
    else:
        effective_date = game_date

    # Try historical.db first
    result = _resolve_leg_from_historical(parsed, effective_date, sport, cur)

    # Fallback to TheOddsAPI scores for very recent games
    if result["status"] == "UNKNOWN":
        api_result = _resolve_leg_from_scores_api(parsed, effective_date, sport)
        if api_result.get("status") not in ("UNKNOWN", None):
            result = {**api_result, "parsed": parsed, "sport": sport}

    result.setdefault("parsed", parsed)
    result.setdefault("sport", sport)
    return result


# ── PRIMARY PUBLIC FUNCTION ────────────────────────────────────────────────────

def resolve_placed_bets(
    db: Session,
    hist_db_path: Optional[str] = None,
    auto_settle: bool = True,
) -> list[dict]:
    """
    Resolve all non-mock PLACED bets and optionally auto-settle when outcome is clear.

    Returns list of bet status dicts — one per bet — with:
      bet_id, bet_outcome, legs_won, legs_lost, legs_remaining,
      won_multiplier, resolved_legs, auto_settled (if settled), profit
    """
    bets = (
        db.query(Bet)
        .filter(Bet.status == "PLACED", Bet.is_mock.is_(False))
        .all()
    )
    if not bets:
        return []

    hist_path = hist_db_path or _HIST_DB_PATH
    conn = _plain_conn(hist_path)
    cur  = conn.cursor()

    now = datetime.utcnow()

    # Bets placed before this date are from completed seasons — auto-settle as loss
    # since player-prop stats are unresolvable and the season is long over.
    _STALE_CUTOFF = datetime(2026, 1, 1)

    results: list[dict] = []

    for bet in bets:
        bet_info = (bet.bet_info or "").strip()
        if not bet_info:
            continue

        placed_dt = bet.time_placed or datetime.utcnow()

        # ── Stale-bet fast path ───────────────────────────────────────────────
        # Bets from before 2026 are from completed seasons. Player prop stats
        # are not in historical.db so they will never resolve. Auto-settle now.
        if auto_settle and placed_dt < _STALE_CUTOFF:
            bet.status      = "SETTLED_LOSS"
            bet.profit      = -round(float(bet.amount or 0), 2)
            bet.time_settled = now
            try:
                db.execute(
                    _sqla_text("UPDATE bets SET status='SETTLED_LOSS', profit=:p, time_settled=:ts WHERE id=:id"),
                    {"p": bet.profit, "ts": now.isoformat(), "id": bet.id},
                )
            except Exception:
                pass
            results.append({
                "bet_id":      bet.id,
                "bet_outcome": "LOST",
                "auto_settled": "SETTLED_LOSS",
                "profit":       bet.profit,
                "reason":       "stale — placed before 2026, season complete",
                "legs_won": 0, "legs_push": 0, "legs_lost": 0, "legs_remaining": 0,
                "won_multiplier": 0.0, "remaining_legs": [], "resolved_legs": [],
                "bet": {"id": bet.id, "amount": bet.amount, "odds": bet.odds,
                        "time_placed": placed_dt.isoformat(), "status": "SETTLED_LOSS"},
                "cashout": None,
            })
            continue
        game_date    = placed_dt.strftime("%Y-%m-%d")     # fallback when no fixture found
        stated_sport = (bet.sports or "").split("|")[0].strip()

        leg_descs = [d.strip() for d in bet_info.split("|") if d.strip()]
        resolved_legs: list[dict] = []

        for i, desc in enumerate(leg_descs):
            # ── Determine expected game start (commence_time anchor) ──────────
            # 1. Check stored game_commence_time in bet_legs table (set at placement)
            bl_row = db.query(BetLeg).filter(
                BetLeg.bet_id    == bet.id,
                BetLeg.leg_index == i,
            ).first()

            expected_game_start: Optional[datetime] = None

            if bl_row and getattr(bl_row, "game_commence_time", None):
                try:
                    raw = bl_row.game_commence_time
                    expected_game_start = datetime.fromisoformat(raw.split("+")[0].rstrip("Z"))
                except Exception:
                    pass

            if expected_game_start is None:
                # 2. Fall back: parse team from description, look up fixture
                has_mu = " @ " in desc or " v " in desc or " vs " in desc
                try:
                    tmp = lr.parse_leg_details(desc) if has_mu else _parse_compact_leg(desc)
                except Exception:
                    tmp = {}
                team_raw = tmp.get("selected_team_or_player") or ""
                canonical = lr.normalize_team(team_raw) if team_raw else ""
                inferred_sport = _infer_sport_for_team(canonical, stated_sport, cur)
                expected_game_start = _lookup_commence_time(canonical, inferred_sport, placed_dt, db)

                # 3. Cache it back to bet_legs so future calls skip the fixture query
                if expected_game_start is not None and bl_row is not None:
                    try:
                        db.execute(
                            _sqla_text(
                                "UPDATE bet_legs SET game_commence_time=:gct WHERE id=:id"
                            ),
                            {"gct": expected_game_start.isoformat(), "id": bl_row.id},
                        )
                    except Exception:
                        pass

            r = _resolve_single_leg(
                desc, game_date, stated_sport, cur,
                expected_game_start=expected_game_start,
            )
            p = r.get("parsed") or {}
            resolved_legs.append({
                "index":             i,
                "description":       desc,
                "status":            r.get("status", "UNKNOWN"),
                "result":            r.get("result"),
                "sport":             r.get("sport", stated_sport),
                "team":              p.get("selected_team_or_player", ""),
                "market_type":       p.get("market_type", ""),
                "line":              p.get("line"),
                "odds":              p.get("odds"),
                "accuracy_delta":    r.get("accuracy_delta"),
                "game":              r.get("game"),
                "resolution_source": r.get("resolution_source"),
                "reason":            r.get("reason"),
            })

        # Tally statuses
        statuses  = [l["status"] for l in resolved_legs]
        n_won     = statuses.count("WON")
        n_push    = statuses.count("PUSH")
        n_lost    = statuses.count("LOST")
        n_unk     = statuses.count("UNKNOWN")

        # Won-leg multiplier (product of decimal odds for WON/PUSH legs)
        won_mult = 1.0
        for l in resolved_legs:
            if l["status"] in ("WON", "PUSH"):
                am = l.get("odds")
                if am is not None:
                    dec = (1 + am / 100) if am > 0 else (1 + 100 / abs(am))
                    won_mult *= dec
        won_mult = round(won_mult, 3)

        # Determine overall outcome
        if n_lost > 0:
            bet_outcome = "LOST"
        elif n_unk == 0:
            bet_outcome = "WON"
        elif n_won > 0 or n_push > 0:
            bet_outcome = "IN_PROGRESS"
        else:
            bet_outcome = "PENDING"

        bet_dict: dict = {
            "bet_id":         bet.id,
            "bet_outcome":    bet_outcome,
            "legs_won":       n_won,
            "legs_push":      n_push,
            "legs_lost":      n_lost,
            "legs_remaining": n_unk,
            "won_multiplier": won_mult,
            "remaining_legs": [l for l in resolved_legs if l["status"] == "UNKNOWN"],
            "resolved_legs":  resolved_legs,
            "bet": {
                "id":          bet.id,
                "amount":      bet.amount,
                "odds":        bet.odds,
                "legs":        bet.legs,
                "sports":      bet.sports,
                "bet_info":    bet.bet_info,
                "time_placed": bet.time_placed.isoformat() if bet.time_placed else None,
                "status":      bet.status,
                "cashed_out":  bool(getattr(bet, "cashed_out", False)),
                "cash_out_amount": getattr(bet, "cash_out_amount", None),
            },
        }

        # Auto-settle when outcome is fully determined
        if auto_settle and bet_outcome in ("WON", "LOST"):
            if bet_outcome == "WON":
                bet.status       = "SETTLED_WIN"
                bet.profit       = round((bet.amount * (bet.odds or 1)) - bet.amount, 2)
                bet.time_settled = now
            else:
                bet.status       = "SETTLED_LOSS"
                bet.profit       = -round(bet.amount or 0, 2)
                bet.time_settled = now

            # Retroactive cash-out verdict
            if getattr(bet, "cashed_out", False):
                verdict = "UNNECESSARY" if bet_outcome == "WON" else "WISE"
                try:
                    db.execute(
                        _sqla_text("UPDATE bets SET cash_out_vs_final_decision=:v WHERE id=:id"),
                        {"v": verdict, "id": bet.id},
                    )
                except Exception:
                    pass
                bet_dict["cash_out_verdict"] = verdict

            try:
                db.commit()
            except Exception:
                db.rollback()

            bet_dict["auto_settled"] = bet.status
            bet_dict["profit"]       = bet.profit

        results.append(bet_dict)

    conn.close()
    return results


# ── CASH-OUT ADVISOR ───────────────────────────────────────────────────────────

def compute_cashout_target(
    amount: float,
    odds: float,
    n_legs: int,
    avg_lqs: float = 65.0,
) -> dict:
    """
    Compute a placement-time cash-out target for a parlay.

    Base pct by n_legs:
      2  → 60%   3  → 55%   4  → 50%
      5  → 45%   6  → 40%   7+ → 35%

    LQS adjustment: ±10pp based on (avg_lqs - 65) / 100 × 0.10
    Clamped [20%, 80%]. Minimum target = stake × 1.5.

    Returns dict with:
      target_amount, target_pct, minimum_amount, full_payout, rationale
    """
    _base_pcts = {1: 0.70, 2: 0.60, 3: 0.55, 4: 0.50, 5: 0.45, 6: 0.40}
    base_pct = _base_pcts.get(n_legs, 0.35)  # 7+ legs → 35%

    lqs_adj = ((avg_lqs - 65.0) / 100.0) * 0.10
    raw_pct = base_pct + lqs_adj
    target_pct = round(max(0.20, min(0.80, raw_pct)), 4)

    full_payout = round(amount * odds, 2)
    target_amount = round(full_payout * target_pct, 2)
    minimum_amount = round(amount * 1.5, 2)
    target_amount = max(target_amount, minimum_amount)

    adj_dir = "+" if lqs_adj >= 0 else ""
    rationale = (
        f"{n_legs}-leg base {base_pct:.0%}"
        f", LQS {avg_lqs:.0f} adj {adj_dir}{lqs_adj*100:.1f}pp"
        f" → {target_pct:.0%} of ${full_payout:.2f} max payout"
        f" = ${target_amount:.2f}"
    )

    return {
        "target_amount":  target_amount,
        "target_pct":     target_pct,
        "minimum_amount": minimum_amount,
        "full_payout":    full_payout,
        "rationale":      rationale,
    }


def evaluate_cashout(
    bet_row: Bet,
    resolved_legs: list[dict],
    db: Session,
) -> dict:
    """
    Compute cash-out recommendation for a bet with some WON and some UNKNOWN legs.

    Returns dict with:
      recommendation: CASH_OUT | HOLD | MONITOR | DEAD
      confidence:     HIGH | MEDIUM | LOW
      fair_cashout, expected_fd_offer, full_payout, won_multiplier,
      remaining_prob, legs_won, legs_remaining, legs_lost, reasons,
      last_updated, next_refresh
    """
    now          = datetime.utcnow()
    next_refresh = now + timedelta(minutes=30)
    amount       = float(bet_row.amount or 10)
    bet_odds     = float(bet_row.odds or 1)
    full_payout  = round(amount * bet_odds, 2)

    statuses = [l["status"] for l in resolved_legs]
    n_won    = statuses.count("WON")
    n_lost   = statuses.count("LOST")
    n_unk    = statuses.count("UNKNOWN")

    _base = {
        "full_payout":    full_payout,
        "last_updated":   now.isoformat(),
        "next_refresh":   next_refresh.isoformat(),
        "legs_won":       n_won,
        "legs_remaining": n_unk,
        "legs_lost":      n_lost,
    }

    if n_lost > 0:
        return {**_base,
                "recommendation": "DEAD", "confidence": "HIGH",
                "fair_cashout": 0.0, "expected_fd_offer": 0.0,
                "won_multiplier": 0.0, "remaining_prob": 0.0,
                "reasons": ["Parlay dead — at least one leg lost"]}

    # Won-leg multiplier
    won_mult = 1.0
    for l in resolved_legs:
        if l["status"] in ("WON", "PUSH"):
            am = l.get("odds")
            if am is not None:
                dec = (1 + am / 100) if am > 0 else (1 + 100 / abs(am))
                won_mult *= dec
    won_mult = round(won_mult, 3)

    # Remaining legs: implied win prob from odds; LQS if available
    remaining = [l for l in resolved_legs if l["status"] == "UNKNOWN"]
    probs: list[float] = []
    lqs_list: list[int] = []

    for l in remaining:
        am = l.get("odds")
        if am is not None:
            implied = (100 / (100 + am)) if am > 0 else (abs(am) / (abs(am) + 100))
            probs.append(max(0.05, min(0.95, implied)))
        else:
            probs.append(0.50)
        lqs_list.append(l.get("lqs") or 65)

    rem_prob = 1.0
    for p in probs:
        rem_prob *= p
    rem_prob = round(rem_prob, 4)

    min_lqs = min(lqs_list) if lqs_list else 65

    fair_cashout      = round(amount * won_mult * rem_prob, 2)
    expected_fd_offer = round(fair_cashout * 0.90, 2)

    # Decision signals
    reasons: list[str] = []
    cash_signals = hold_signals = 0

    # CASH OUT triggers
    if min_lqs < 55:
        reasons.append(f"Weak remaining leg (LQS {min_lqs}) risks {won_mult:.1f}x buildup")
        cash_signals += 2
    if won_mult > 4.0 and rem_prob < 0.25:
        reasons.append(f"High buildup ({won_mult:.1f}x) with uncertain finish ({rem_prob:.0%})")
        cash_signals += 2
    if full_payout > 0 and expected_fd_offer > full_payout * 0.75:
        reasons.append(f"Cash out offer ~${expected_fd_offer:.2f} exceeds 75% of max ${full_payout:.2f}")
        cash_signals += 1

    # HOLD triggers
    if min_lqs >= 65:
        hold_signals += 1
    if rem_prob >= 0.30:
        hold_signals += 1
    if full_payout > 0 and expected_fd_offer < full_payout * 0.50:
        hold_signals += 1

    # (initial rec/conf will be set below after target adjustment)
    rec = "MONITOR"
    conf = "LOW"

    # Pull placement-time target if stored
    stored_target     = getattr(bet_row, "cash_out_target", None)
    stored_target_pct = getattr(bet_row, "cash_out_target_pct", None)

    target_signal: str | None = None
    if stored_target is not None and expected_fd_offer > 0:
        ratio = expected_fd_offer / stored_target
        if ratio >= 1.0:
            target_signal = f"Offer ~${expected_fd_offer:.2f} meets your target ${stored_target:.2f} ✓"
            cash_signals += 1
        elif ratio >= 0.90:
            target_signal = f"Offer ~${expected_fd_offer:.2f} within 10% of target ${stored_target:.2f}"
        elif ratio >= 0.75:
            target_signal = f"Offer ~${expected_fd_offer:.2f} below target ${stored_target:.2f} (75%+ threshold)"
        else:
            target_signal = f"Offer ~${expected_fd_offer:.2f} well below target ${stored_target:.2f}"
            hold_signals += 1

        if target_signal:
            reasons.append(target_signal)

    # Re-evaluate rec after target adjustment
    if cash_signals >= 2:
        rec = "CASH_OUT"
        conf = "HIGH" if cash_signals >= 3 else "MEDIUM"
        if not reasons:
            reasons.append("Multiple risk factors — consider cashing out")
    elif hold_signals >= 3:
        rec = "HOLD"
        conf = "HIGH"
        lqs_str = "/".join(str(q) for q in lqs_list) if lqs_list else "—"
        reasons.append(f"Strong remaining legs — LQS {lqs_str} · {rem_prob:.0%} combined prob")
        reasons.append(f"Expected offer ~${expected_fd_offer:.2f} vs ${full_payout:.2f} max payout")
    elif hold_signals >= 2:
        rec = "HOLD"
        conf = "MEDIUM"
        reasons.append(f"Good remaining legs ({rem_prob:.0%} combined prob)")
    else:
        rec = "MONITOR"
        conf = "LOW"
        reasons.append("Track closely — reassess at next refresh")

    return {**_base,
            "recommendation":    rec,
            "confidence":        conf,
            "fair_cashout":      fair_cashout,
            "expected_fd_offer": expected_fd_offer,
            "won_multiplier":    round(won_mult, 2),
            "remaining_prob":    rem_prob,
            "min_lqs":           min_lqs,
            "cash_out_target":   stored_target,
            "cash_out_target_pct": stored_target_pct,
            "reasons":           reasons}


def log_cashout_recommendation(bet: Bet, rec: dict, db: Session) -> None:
    """Append a cashout recommendation snapshot to cash_out_offers_log JSON array."""
    try:
        existing = json.loads(getattr(bet, "cash_out_offers_log", None) or "[]")
    except Exception:
        existing = []

    entry = {
        "timestamp":      rec.get("last_updated") or datetime.utcnow().isoformat(),
        "recommendation": rec.get("recommendation"),
        "won_multiplier": rec.get("won_multiplier"),
        "remaining_prob": rec.get("remaining_prob"),
        "fair_cashout":   rec.get("fair_cashout"),
        "reasons":        rec.get("reasons", []),
        "changed":        not existing or existing[-1].get("recommendation") != rec.get("recommendation"),
    }
    existing.append(entry)

    try:
        db.execute(
            _sqla_text("UPDATE bets SET cash_out_offers_log=:v WHERE id=:id"),
            {"v": json.dumps(existing), "id": bet.id},
        )
        db.commit()
    except Exception:
        db.rollback()
