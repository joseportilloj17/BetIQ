"""
shadow_altline.py — Forward shadow arm for MLB alt-spread edge validation.

Purpose: validate the mlb_ats_v1 alt-spread edge found in the multi-lever
analysis (L2 realized PnL = +10.7% ROI, z=3.73, concentrated at +2.5 rung).
The shadow resolves the fair-price mapping debate: forward realized PnL on
real alt prices is mapping-INDEPENDENT ground truth.

Architecture:
  - Scores today's MLB alt-spread legs through the ATS model (same score_leg
    path as the live pipeline)
  - Logs REAL actionable alt-line prices per bookmaker from the alt_lines table
    (not derived/fair prices — actual book odds you could bet)
  - Stores individual legs as 1-leg mock_bets, source='shadow_altline'
  - STRAIGHT bets (not parlays), FLAT stake, $0 paper
  - Settled by the same ESPN pipeline as all other shadow arms

Selection logic (mirrors live pipeline for mlb_ats_v1 alt-spreads):
  - MLB alternate_spreads only
  - ATS model wp >= 55.0 for rungs |spread| <= 4.5 (same gate as recommender.py)
  - All rungs included (forward read confirms rung coherence vs cherry-pick)
  - Full slate, independent (no dedup — each leg stored separately)

Forward read (three levers, at N~300-500 settled):
  - L1: Realized PnL forward (ground truth, mapping-independent) — PRIMARY
  - L2: Entry vs devigged main line (sharp benchmark) — secondary
  - L3: Alt-close CLV (expected to understate — soft-close finding)
  - Per-rung breakdown for survivorship check
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
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# US-accessible books with real alt-spread limits ($50-250+).
# Pinnacle is sharp/benchmark only — high limits but not US-accessible for most bettors.
# Entry price must come from a BETTABLE book, not the sharp benchmark.
_BETTABLE_BOOKS = {"fanduel", "draftkings", "betmgm", "caesars", "betrivers",
                   "espnbet", "fanatics", "pointsbetus"}
_SHARP_BOOKS = {"pinnacle"}


def _decimal_to_american(dec: float) -> int:
    """Convert decimal odds to American."""
    if dec >= 2.0:
        return round((dec - 1.0) * 100)
    elif dec > 1.0:
        return round(-100.0 / (dec - 1.0))
    return 0


def _get_alt_line_prices(event_id: str, team: str, line: float,
                         today_str: str) -> dict:
    """Look up real per-bookmaker alt-line prices from alt_lines table.

    Returns dict with:
      all_prices: list of {bookmaker, odds_decimal, odds_american, fetched_at, bettable}
      bettable_prices: subset from US-accessible books (the actionable prices)
      sharp_prices: subset from sharp benchmarks (Pinnacle)
      best_bettable: best price from a bettable book (entry price for PnL)
      best_sharp: best sharp book price (benchmark only)
      best_overall: absolute best across all books (for reference — NOT entry price)
      liquidity: "liquid" (3+ bettable), "moderate" (2), "thin" (1), "none" (0)
      n_bettable: count of bettable books carrying this line
    """
    result = {
        "all_prices": [], "bettable_prices": [], "sharp_prices": [],
        "best_bettable": None, "best_sharp": None, "best_overall": None,
        "liquidity": "none", "n_bettable": 0,
    }
    try:
        conn = sqlite3.connect(_HIST_DB)
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            "SELECT MAX(fetched_at) FROM alt_lines "
            "WHERE event_id=? AND fetched_at >= ?",
            (event_id, f"{today_str}T00:00:00Z")
        ).fetchone()
        latest = row[0] if row and row[0] else None

        if not latest:
            conn.close()
            return result

        rows = conn.execute(
            "SELECT bookmaker, odds, fetched_at FROM alt_lines "
            "WHERE event_id=? AND team=? AND line=? "
            "AND market_key='alternate_spreads' AND fetched_at=?",
            (event_id, team, line, latest)
        ).fetchall()
        conn.close()

        for r in rows:
            dec = float(r["odds"])
            bk = r["bookmaker"].lower()
            is_bettable = bk in _BETTABLE_BOOKS
            entry = {
                "bookmaker": r["bookmaker"],
                "odds_decimal": round(dec, 4),
                "odds_american": _decimal_to_american(dec),
                "fetched_at": r["fetched_at"],
                "bettable": is_bettable,
            }
            result["all_prices"].append(entry)
            if is_bettable:
                result["bettable_prices"].append(entry)
            if bk in _SHARP_BOOKS:
                result["sharp_prices"].append(entry)

        # Sort each list by odds descending
        for key in ("all_prices", "bettable_prices", "sharp_prices"):
            result[key].sort(key=lambda p: -p["odds_decimal"])

        if result["bettable_prices"]:
            result["best_bettable"] = result["bettable_prices"][0]
        if result["sharp_prices"]:
            result["best_sharp"] = result["sharp_prices"][0]
        if result["all_prices"]:
            result["best_overall"] = result["all_prices"][0]

        n_bet = len(result["bettable_prices"])
        result["n_bettable"] = n_bet
        if n_bet >= 3:
            result["liquidity"] = "liquid"
        elif n_bet == 2:
            result["liquidity"] = "moderate"
        elif n_bet == 1:
            result["liquidity"] = "thin"
        else:
            result["liquidity"] = "none"

        return result
    except Exception as exc:
        print(f"[shadow_altline] Price lookup error: {exc}")
        return result


def _get_opposite_side_price(event_id: str, team: str, line: float,
                             home_team: str, away_team: str,
                             today_str: str) -> float | None:
    """Get best opposite-side alt-line price for devigging.

    If we're looking at TeamA +2.5, we need TeamB -2.5 to devig.
    """
    try:
        opp_team = away_team if team == home_team else home_team
        opp_line = -line

        conn = sqlite3.connect(_HIST_DB)
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            "SELECT MAX(fetched_at) FROM alt_lines "
            "WHERE event_id=? AND fetched_at >= ?",
            (event_id, f"{today_str}T00:00:00Z")
        ).fetchone()
        latest = row[0] if row and row[0] else None
        if not latest:
            conn.close()
            return None

        rows = conn.execute(
            "SELECT odds FROM alt_lines "
            "WHERE event_id=? AND team=? AND line=? "
            "AND market_key='alternate_spreads' AND fetched_at=?",
            (event_id, opp_team, opp_line, latest)
        ).fetchall()
        conn.close()

        if not rows:
            return None
        # Return best (lowest) opposite-side odds
        return min(float(r["odds"]) for r in rows)
    except Exception:
        return None


def generate_altline_shadow(db=None, today_str=None):
    """Score today's MLB alt-spread legs, log real book prices, store as shadow.

    Returns list of stored leg dicts.
    """
    from parlay_builder import get_available_legs, score_leg

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[shadow_altline] Generating alt-line shadow for {today_str}")

    # ── Step 1: Get all legs including alt-spreads ────────────────────────────
    all_legs = get_available_legs(
        db, markets=["h2h", "spreads", "totals", "alternate_spreads"]
    )
    if not all_legs:
        print("[shadow_altline] No legs available")
        return []

    # Filter to MLB alternate_spreads only
    alt_spread_legs = [
        leg for leg in all_legs
        if leg.get("sport", "").upper() in ("MLB", "BASEBALL")
        and leg.get("market", "") == "alternate_spreads"
    ]
    print(f"[shadow_altline] {len(alt_spread_legs)} MLB alt-spread legs from fixtures")

    # ── Step 2: Also pull alt-lines from alt_lines table ─────────────────────
    # The fixtures table has limited alt-line coverage from bookmakers JSON.
    # The alt_lines table (populated by creator_tier.py) has broader coverage
    # across 6-8 bookmakers. Supplement with these.
    try:
        from creator_tier import get_scored_alt_lines
        # Get unique fixture IDs from all MLB legs (any market) to look up alts
        mlb_legs = [l for l in all_legs
                    if l.get("sport", "").upper() in ("MLB", "BASEBALL")]
        fixture_ids = set(l.get("fixture_id") for l in mlb_legs if l.get("fixture_id"))

        # Build lookup: fixture_id -> leg info (for home/away team context)
        fix_info = {}
        for l in mlb_legs:
            fid = l.get("fixture_id")
            if fid and fid not in fix_info:
                fix_info[fid] = {
                    "home_team": l.get("home_team", ""),
                    "away_team": l.get("away_team", ""),
                    "game": l.get("game", ""),
                    "game_time": l.get("game_time"),
                    "sport_key": l.get("sport_key", ""),
                }

        # Collect alt-line legs from alt_lines table
        existing_ids = set(l.get("leg_id") for l in alt_spread_legs)
        supplemented = 0

        for fid in fixture_ids:
            try:
                alt_lines = get_scored_alt_lines(fid, min_label="ALL")
            except Exception:
                continue

            info = fix_info.get(fid, {})
            for al in alt_lines:
                if al.get("market_key", "") != "alternate_spreads":
                    continue
                al_line = al.get("line")
                al_team = al.get("team", "")
                al_odds = float(al.get("odds", 0))
                if al_odds <= 1.0:
                    continue

                # Only keep reasonable alt-spread rungs (|line| <= 6.5)
                if al_line is not None and abs(float(al_line)) > 6.5:
                    continue

                leg_id = f"{fid}:alternate_spreads:{al_team}:{al_line}:alt"
                if leg_id in existing_ids:
                    continue
                existing_ids.add(leg_id)

                point_str = (f" {al_line:+g}" if al_line is not None
                             and float(al_line) != 0 else "")
                desc = f"{al_team}{point_str} (Alt Spread)"

                alt_spread_legs.append({
                    "leg_id": leg_id,
                    "fixture_id": fid,
                    "game": info.get("game", ""),
                    "sport": "MLB",
                    "sport_key": info.get("sport_key", "baseball_mlb"),
                    "market": "alternate_spreads",
                    "market_label": "Alt Spread",
                    "pick": al_team,
                    "point": al_line,
                    "description": desc,
                    "odds": round(al_odds, 3),
                    "implied_prob": round(1 / al_odds * 100, 2),
                    "game_time": info.get("game_time"),
                    "home_team": info.get("home_team", ""),
                    "away_team": info.get("away_team", ""),
                    "is_alt_line": True,
                })
                supplemented += 1

        if supplemented > 0:
            print(f"[shadow_altline] +{supplemented} legs from alt_lines table "
                  f"(total: {len(alt_spread_legs)})")
    except Exception as exc:
        print(f"[shadow_altline] Alt-lines supplement failed: {exc}")

    if not alt_spread_legs:
        print("[shadow_altline] No MLB alt-spread legs")
        return []

    # ── Step 3: Score each leg with the ATS model ────────────────────────────
    scored = []
    for leg in alt_spread_legs:
        try:
            scored_leg = score_leg(leg)
            wp = scored_leg.get("win_prob")
            if wp is None:
                continue

            # Apply same gate as live pipeline (recommender.py line 925-930):
            # MLB alt-spreads |spread| <= 4.5: model wp >= 55.0
            # |spread| > 4.5: implied prob >= 65.0 (no model signal)
            point = leg.get("point")
            spread_abs = abs(float(point)) if point is not None else 0

            if spread_abs <= 4.5:
                if wp >= 55.0:
                    scored_leg["gate"] = f"model_wp_{wp:.0f}"
                    scored.append(scored_leg)
            else:
                implied = leg.get("implied_prob", 0)
                if implied >= 65.0:
                    scored_leg["win_prob"] = implied
                    scored_leg["gate"] = f"implied_{implied:.0f}"
                    scored.append(scored_leg)
        except Exception as exc:
            print(f"[shadow_altline] Score error: {exc}")
            continue

    print(f"[shadow_altline] {len(scored)} legs pass gate (from {len(alt_spread_legs)} candidates)")

    # ── Step 4: Spread distribution ──────────────────────────────────────────
    spread_counts = {}
    for leg in scored:
        point = leg.get("point")
        sv = float(point) if point is not None else None
        spread_counts[sv] = spread_counts.get(sv, 0) + 1
    print(f"[shadow_altline] Spread distribution: {spread_counts}")

    # ── Step 5: Look up real per-book prices & store ─────────────────────────
    if db is not None and scored:
        stored = _store_altline_legs(db, scored, today_str)
        print(f"[shadow_altline] Stored {stored} shadow legs (source=shadow_altline)")

    return scored


def _store_altline_legs(db, scored_legs, today_str):
    """Persist scored alt-spread legs as 1-leg mock_bets with real book prices."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    run_id = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc)
    stored = 0

    for leg in scored_legs:
        mock_id = str(uuid.uuid4())
        desc = leg.get("description", "")
        wp = leg.get("win_prob", 0)
        odds_dec = leg.get("odds", 1.0)
        ev = leg.get("ev", 0)
        point = leg.get("point")
        sv = float(point) if point is not None else None
        fid = leg.get("fixture_id", "")
        team = leg.get("pick", "")
        home_team = leg.get("home_team", "")
        away_team = leg.get("away_team", "")
        gate = leg.get("gate", "")

        bet_info = f"{desc} | wp={wp:.1f}"

        # Dedup: skip if same leg already stored today
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
            "AND source='shadow_altline' AND bet_info=:bi"
        ), {"gd": today_str, "bi": bet_info}).scalar()
        if existing > 0:
            continue

        # Look up real per-book prices (bettable vs sharp separated)
        price_data = _get_alt_line_prices(fid, team, sv, today_str)
        liquidity = price_data["liquidity"]
        n_bettable = price_data["n_bettable"]

        # ENTRY PRICE: best price from a BETTABLE US book (not sharp benchmark).
        # This is the price you could actually bet — forward PnL is real money.
        # Falls back to best_overall only if no bettable book carries the line.
        entry_source = price_data["best_bettable"] or price_data["best_overall"]
        if entry_source:
            best_dec = entry_source["odds_decimal"]
            best_am = entry_source["odds_american"]
            best_bookmaker = entry_source["bookmaker"]
            price_type = "bettable" if price_data["best_bettable"] else "fallback"
        else:
            best_dec = odds_dec
            best_am = _decimal_to_american(odds_dec)
            best_bookmaker = "unknown"
            price_type = "fallback"

        # Sharp benchmark (Pinnacle) — logged separately, NOT used as entry price
        sharp_entry = price_data["best_sharp"]
        sharp_dec = sharp_entry["odds_decimal"] if sharp_entry else None

        # Devig: look up opposite side for no-vig probability
        opp_dec = _get_opposite_side_price(fid, team, sv, home_team, away_team, today_str)
        devig_prob = None
        overround = None
        if opp_dec and opp_dec > 1.0:
            imp_our = 1.0 / best_dec
            imp_opp = 1.0 / opp_dec
            overround = round(imp_our + imp_opp, 4)
            devig_prob = round(imp_our / (imp_our + imp_opp), 6)

        # Build per-book price snapshot for notes
        all_prices = {p["bookmaker"]: p["odds_decimal"] for p in price_data["all_prices"]}
        bettable_prices = {p["bookmaker"]: p["odds_decimal"] for p in price_data["bettable_prices"]}

        notes_dict = {
            "gate": gate,
            "spread": sv,
            "model_wp": round(wp, 2),
            "best_book": best_bookmaker,
            "best_dec": best_dec,
            "best_am": best_am,
            "price_type": price_type,          # "bettable" or "fallback"
            "liquidity": liquidity,            # "liquid"/"moderate"/"thin"/"none"
            "n_bettable": n_bettable,          # how many US books carry this line
            "bettable_prices": bettable_prices, # US-book prices only
            "all_prices": all_prices,           # all books including sharp
        }
        if sharp_dec is not None:
            notes_dict["pinnacle_dec"] = sharp_dec  # sharp benchmark, not entry
        if devig_prob is not None:
            notes_dict["devig_prob"] = devig_prob
            notes_dict["overround"] = overround

        notes_json = json.dumps(notes_dict, separators=(",", ":"))

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            generated_at=now_utc,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=best_dec,
            amount=0.0,
            legs=1,
            bet_info=bet_info,
            predicted_win_prob=wp,
            model_used=leg.get("model_used", "mlb_ats_v1"),
            status="PENDING",
            source="shadow_altline",
            cohort="clean",
            notes=notes_json,
        )
        db.add(mock)

        mock_leg = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=desc,
            market_type="alternate_spreads",
            sport="MLB",
            win_prob=wp,
            ev=ev,
            fixture_id=fid,
            is_alt_line=True,
            open_odds=best_am,
            devig_entry_prob=devig_prob,
            overround=overround,
        )
        db.add(mock_leg)
        stored += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[shadow_altline] Commit error: {exc}")
        return 0

    return stored


if __name__ == "__main__":
    from database import SessionLocal, init_db
    init_db()
    db = SessionLocal()
    try:
        legs = generate_altline_shadow(db)
        if legs:
            print(f"\n[shadow_altline] Summary: {len(legs)} legs stored")
            # Per-rung breakdown
            rung_counts = {}
            for leg in legs:
                sv = float(leg.get("point", 0)) if leg.get("point") is not None else None
                rung_counts[sv] = rung_counts.get(sv, 0) + 1
            for sv in sorted(rung_counts.keys(), key=lambda x: x or 0):
                print(f"  rung {sv:+.1f}: {rung_counts[sv]} legs")
            # Top 10 by WP
            print("\n  Top 10 by model WP:")
            for leg in sorted(legs, key=lambda l: l.get("win_prob", 0), reverse=True)[:10]:
                wp = leg.get("win_prob", 0)
                desc = leg.get("description", "")[:50]
                odds = leg.get("odds", 0)
                print(f"  wp={wp:5.1f}% odds={odds:.3f} {desc}")
    finally:
        db.close()
