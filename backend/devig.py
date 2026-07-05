"""
devig.py — Group F: True no-vig probability computation.

MEASURE-ONLY.  Computes devigged implied probabilities and stores them
ALONGSIDE existing win_prob.  Does NOT replace win_prob, does NOT change
live picks, does NOT resize stakes, does NOT feed Kelly.

Devig formula (multiplicative / proportional):
  2-way: true_prob = (1/our_odds) / (1/our_odds + 1/other_odds)
  3-way: true_prob = (1/our_odds) / (1/odds_A + 1/odds_B + 1/odds_C)

Columns written to mock_bet_legs:
  devig_entry_prob  — devigged implied prob at entry (0-1 scale)
  devig_close_prob  — devigged implied prob at close (0-1 scale)
  overround         — total implied prob before devig (e.g. 1.05 = 5% vig)
  devig_available   — 1 if paired side found, 0 if not
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import logging

log = logging.getLogger(__name__)

_BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
_HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _american_to_decimal(american: int | float) -> float | None:
    """Convert American odds to decimal odds.  Returns None for invalid."""
    try:
        american = int(american)
        if american > 0:
            return 1.0 + american / 100.0
        elif american < 0:
            return 1.0 + 100.0 / abs(american)
        return None
    except (TypeError, ValueError):
        return None


def _parse_spread_from_desc(desc: str) -> float | None:
    """Extract spread value from description like 'Yankees +1.5 (Alt Spread)'."""
    m = re.search(r'([+-]?\d+\.?\d*)\s*\(', desc)
    return float(m.group(1)) if m else None


def _parse_ou_line_from_desc(desc: str) -> tuple[str | None, float | None]:
    """Extract over/under and line from 'Over 8.5 (Total)' or 'Under +4.5 (Alt Total)'."""
    m = re.search(r'(Over|Under)\s+\+?(\d+\.?\d*)', desc, re.IGNORECASE)
    if m:
        return m.group(1).lower(), float(m.group(2))
    return None, None


def _parse_team_from_desc(desc: str, home: str | None, away: str | None) -> str | None:
    """Determine which team is the pick from description."""
    if home and home in desc:
        return home
    if away and away in desc:
        return away
    return None


def _devig_2way(our_implied: float, other_implied: float) -> tuple[float, float]:
    """Multiplicative devig for 2-way market.
    Returns (devigged_prob, overround)."""
    overround = our_implied + other_implied
    if overround <= 0:
        return our_implied, 1.0
    return our_implied / overround, overround


def _devig_3way(our_implied: float, all_implied: list[float]) -> tuple[float, float]:
    """Multiplicative devig for 3-way market (e.g. soccer h2h).
    Returns (devigged_prob, overround)."""
    overround = sum(all_implied)
    if overround <= 0:
        return our_implied, 1.0
    return our_implied / overround, overround


# ═══════════════════════════════════════════════════════════════════════════
# Entry devig — from bookmakers JSON or alt_lines table
# ═══════════════════════════════════════════════════════════════════════════

def _entry_devig_main_market(leg: dict, bookmakers_json) -> tuple[float | None, float | None]:
    """Compute entry devig for main-market legs (spreads/totals/h2h).

    Uses the fixtures.bookmakers JSON which contains both sides.
    Returns (devig_prob, overround) or (None, None).
    """
    if not bookmakers_json:
        return None, None

    bk = json.loads(bookmakers_json) if isinstance(bookmakers_json, str) else bookmakers_json
    if not bk:
        return None, None

    mkt_key = leg["market_type"]
    desc = leg["description"] or ""
    home = leg.get("home_team") or ""
    away = leg.get("away_team") or ""

    # Map market_type to bookmakers JSON key
    bk_key_map = {"spreads": "spreads", "totals": "totals", "h2h": "h2h"}
    bk_key = bk_key_map.get(mkt_key)
    if not bk_key:
        return None, None

    # Find the market in the first bookmaker that has it
    for b in bk:
        for mkt in b.get("markets", []):
            if mkt["key"] != bk_key:
                continue
            outcomes = mkt.get("outcomes", [])
            if len(outcomes) < 2:
                continue

            # Identify which outcome is "ours"
            our_outcome = None
            other_outcomes = []

            if mkt_key == "h2h":
                # Match by team name
                pick_team = _parse_team_from_desc(desc, home, away)
                for o in outcomes:
                    if pick_team and pick_team in o.get("name", ""):
                        our_outcome = o
                    else:
                        other_outcomes.append(o)
            elif mkt_key == "spreads":
                pick_team = _parse_team_from_desc(desc, home, away)
                spread = _parse_spread_from_desc(desc)
                for o in outcomes:
                    name = o.get("name", "")
                    pt = o.get("point")
                    if pick_team and pick_team in name:
                        our_outcome = o
                    else:
                        other_outcomes.append(o)
            elif mkt_key == "totals":
                ou, line = _parse_ou_line_from_desc(desc)
                for o in outcomes:
                    name = o.get("name", "").lower()
                    if ou and ou in name:
                        our_outcome = o
                    else:
                        other_outcomes.append(o)

            if not our_outcome or not other_outcomes:
                continue

            our_price = our_outcome.get("price")
            if not our_price or our_price <= 1.0:
                continue

            our_implied = 1.0 / our_price
            all_implied = [our_implied] + [1.0 / o["price"] for o in other_outcomes if o.get("price", 0) > 1.0]

            if len(all_implied) < 2:
                continue

            if len(outcomes) >= 3 and mkt_key == "h2h":
                # 3-way (soccer)
                return _devig_3way(our_implied, all_implied)
            else:
                # 2-way
                other_implied = sum(all_implied) - our_implied
                return _devig_2way(our_implied, other_implied)

    return None, None


def _entry_devig_alt_line(leg: dict, hist_con) -> tuple[float | None, float | None]:
    """Compute entry devig for alt-line legs using alt_lines table.

    Returns (devig_prob, overround) or (None, None).
    """
    fid = leg.get("fixture_id")
    mkt = leg.get("market_type")
    desc = leg.get("description") or ""
    home = leg.get("home_team") or ""
    away = leg.get("away_team") or ""
    open_odds_am = leg.get("open_odds")

    if not fid or not mkt:
        return None, None

    # Get our decimal odds from open_odds (American)
    our_dec = _american_to_decimal(open_odds_am) if open_odds_am else None

    if mkt == "alternate_spreads":
        spread = _parse_spread_from_desc(desc)
        pick_team = _parse_team_from_desc(desc, home, away)
        if not spread or not pick_team:
            return None, None

        other_team = away if pick_team == home else home
        neg_spread = -spread

        # Get our side odds from alt_lines (most accurate — not converted from American)
        our_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND team=? AND line=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, pick_team, spread),
        ).fetchone()

        other_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND team=? AND line=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, other_team, neg_spread),
        ).fetchone()

        if our_row and other_row:
            our_price = float(our_row[0])
            other_price = float(other_row[0])
            if our_price > 1.0 and other_price > 1.0:
                return _devig_2way(1.0 / our_price, 1.0 / other_price)
        elif other_row and our_dec and our_dec > 1.0:
            # Fallback: use open_odds for our side
            other_price = float(other_row[0])
            if other_price > 1.0:
                return _devig_2way(1.0 / our_dec, 1.0 / other_price)

    elif mkt == "alternate_totals":
        ou, line = _parse_ou_line_from_desc(desc)
        if not ou or line is None:
            return None, None

        other_ou = "under" if ou == "over" else "over"

        our_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND line=? AND LOWER(team)=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, line, ou),
        ).fetchone()

        other_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND line=? AND LOWER(team)=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, line, other_ou),
        ).fetchone()

        if our_row and other_row:
            our_price = float(our_row[0])
            other_price = float(other_row[0])
            if our_price > 1.0 and other_price > 1.0:
                return _devig_2way(1.0 / our_price, 1.0 / other_price)
        elif other_row and our_dec and our_dec > 1.0:
            other_price = float(other_row[0])
            if other_price > 1.0:
                return _devig_2way(1.0 / our_dec, 1.0 / other_price)

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Close devig — from line_movement_log (main) or alt_lines latest (alt)
# ═══════════════════════════════════════════════════════════════════════════

def _pick_to_lml_side(desc: str, home: str | None, away: str | None,
                      market_type: str) -> str | None:
    """Map a leg's description to LineMovementLog side value."""
    if market_type in ("totals", "alternate_totals"):
        ou, _ = _parse_ou_line_from_desc(desc)
        return ou  # "over" or "under"
    team = _parse_team_from_desc(desc, home, away)
    if team == home:
        return "home"
    if team == away:
        return "away"
    return None


def _close_devig_main_market(leg: dict, bets_con) -> tuple[float | None, float | None]:
    """Compute closing devig for main-market legs using line_movement_log.

    Uses the latest snapshot before/at game time for both sides.
    Returns (devig_prob, overround) or (None, None).
    """
    fid = leg.get("fixture_id")
    mkt = leg.get("market_type")
    desc = leg.get("description") or ""
    home = leg.get("home_team") or ""
    away = leg.get("away_team") or ""

    if not fid or not mkt:
        return None, None

    our_side = _pick_to_lml_side(desc, home, away, mkt)
    if not our_side:
        return None, None

    # Determine opposite side
    side_pairs = {"home": "away", "away": "home", "over": "under", "under": "over"}
    other_side = side_pairs.get(our_side)
    if not other_side:
        return None, None

    # For h2h → check if 3-way (draw exists)
    lml_mkt = mkt  # spreads/totals/h2h map directly

    # Get latest snapshot for our side
    our_row = bets_con.execute(
        "SELECT decimal_odds FROM line_movement_log "
        "WHERE fixture_id=? AND market_type=? AND side=? "
        "ORDER BY fetched_at DESC LIMIT 1",
        (fid, lml_mkt, our_side),
    ).fetchone()

    other_row = bets_con.execute(
        "SELECT decimal_odds FROM line_movement_log "
        "WHERE fixture_id=? AND market_type=? AND side=? "
        "ORDER BY fetched_at DESC LIMIT 1",
        (fid, lml_mkt, other_side),
    ).fetchone()

    if not our_row or not other_row:
        return None, None

    our_dec = float(our_row[0])
    other_dec = float(other_row[0])

    if our_dec <= 1.0 or other_dec <= 1.0:
        return None, None

    our_implied = 1.0 / our_dec
    other_implied = 1.0 / other_dec

    # Check for 3-way (soccer h2h — draw side)
    if mkt == "h2h":
        draw_row = bets_con.execute(
            "SELECT decimal_odds FROM line_movement_log "
            "WHERE fixture_id=? AND market_type=? AND side='draw' "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, lml_mkt),
        ).fetchone()
        if draw_row:
            draw_dec = float(draw_row[0])
            if draw_dec > 1.0:
                all_implied = [our_implied, other_implied, 1.0 / draw_dec]
                return _devig_3way(our_implied, all_implied)

    return _devig_2way(our_implied, other_implied)


def _close_devig_alt_line(leg: dict, hist_con) -> tuple[float | None, float | None]:
    """Compute closing devig for alt-line legs.

    Uses the latest alt_lines snapshot (closest to game time).
    Returns (devig_prob, overround) or (None, None).
    """
    fid = leg.get("fixture_id")
    mkt = leg.get("market_type")
    desc = leg.get("description") or ""
    home = leg.get("home_team") or ""
    away = leg.get("away_team") or ""

    if not fid or not mkt:
        return None, None

    if mkt == "alternate_spreads":
        spread = _parse_spread_from_desc(desc)
        pick_team = _parse_team_from_desc(desc, home, away)
        if not spread or not pick_team:
            return None, None

        other_team = away if pick_team == home else home
        neg_spread = -spread

        # Latest snapshot = closest to closing
        our_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND team=? AND line=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, pick_team, spread),
        ).fetchone()
        other_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND team=? AND line=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, other_team, neg_spread),
        ).fetchone()

        if our_row and other_row:
            our_price = float(our_row[0])
            other_price = float(other_row[0])
            if our_price > 1.0 and other_price > 1.0:
                return _devig_2way(1.0 / our_price, 1.0 / other_price)

    elif mkt == "alternate_totals":
        ou, line = _parse_ou_line_from_desc(desc)
        if not ou or line is None:
            return None, None
        other_ou = "under" if ou == "over" else "over"

        our_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND line=? AND LOWER(team)=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, line, ou),
        ).fetchone()
        other_row = hist_con.execute(
            "SELECT odds FROM alt_lines WHERE event_id=? AND market_key=? AND line=? AND LOWER(team)=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (fid, mkt, line, other_ou),
        ).fetchone()

        if our_row and other_row:
            our_price = float(our_row[0])
            other_price = float(other_row[0])
            if our_price > 1.0 and other_price > 1.0:
                return _devig_2way(1.0 / our_price, 1.0 / other_price)

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Backfill — compute devig for all settled legs
# ═══════════════════════════════════════════════════════════════════════════

def backfill_devig(force: bool = False):
    """Compute and store devig probabilities for all settled mock_bet_legs.

    Skips legs that already have devig_available set (unless force=True).
    Safe to re-run — idempotent.
    """
    bets_con = sqlite3.connect(os.path.abspath(_BETS_DB), timeout=30)
    bets_con.row_factory = sqlite3.Row

    hist_path = os.path.abspath(_HIST_DB)
    hist_con = None
    if os.path.exists(hist_path):
        hist_con = sqlite3.connect(hist_path, timeout=30)
        hist_con.row_factory = sqlite3.Row

    # Fetch all settled legs that need devig
    where = "WHERE l.leg_result IN ('WIN','LOSS')"
    if not force:
        where += " AND l.devig_available IS NULL"

    rows = bets_con.execute(f"""
        SELECT l.id, l.fixture_id, l.market_type, l.sport, l.description,
               l.open_odds, l.close_odds, l.is_alt_line, l.win_prob,
               f.bookmakers, f.home_team, f.away_team
        FROM mock_bet_legs l
        JOIN fixtures f ON f.id = l.fixture_id
        {where}
    """).fetchall()

    log.info(f"[devig] Processing {len(rows)} legs")
    print(f"[devig] Processing {len(rows)} legs")

    updated = 0
    available = 0
    entry_ok = 0
    close_ok = 0

    for row in rows:
        leg = dict(row)
        is_alt = bool(leg.get("is_alt_line"))
        mkt = leg.get("market_type") or ""

        # ── Entry devig ──
        entry_prob, entry_or = None, None
        if is_alt and mkt in ("alternate_spreads", "alternate_totals") and hist_con:
            entry_prob, entry_or = _entry_devig_alt_line(leg, hist_con)
        elif mkt in ("spreads", "totals", "h2h"):
            entry_prob, entry_or = _entry_devig_main_market(leg, leg.get("bookmakers"))

        # ── Close devig ──
        close_prob, close_or = None, None
        if is_alt and mkt in ("alternate_spreads", "alternate_totals") and hist_con:
            close_prob, close_or = _close_devig_alt_line(leg, hist_con)
        elif mkt in ("spreads", "totals", "h2h"):
            close_prob, close_or = _close_devig_main_market(leg, bets_con)

        # ── Store ──
        has_devig = entry_prob is not None
        if has_devig:
            available += 1
            entry_ok += 1
        if close_prob is not None:
            close_ok += 1

        bets_con.execute(
            "UPDATE mock_bet_legs SET devig_entry_prob=?, devig_close_prob=?, "
            "overround=?, devig_available=? WHERE id=?",
            (
                round(entry_prob, 6) if entry_prob is not None else None,
                round(close_prob, 6) if close_prob is not None else None,
                round(entry_or, 6) if entry_or is not None else None,
                1 if has_devig else 0,
                leg["id"],
            ),
        )
        updated += 1

        if updated % 1000 == 0:
            bets_con.commit()
            print(f"  ... {updated}/{len(rows)} processed")

    bets_con.commit()
    bets_con.close()
    if hist_con:
        hist_con.close()

    summary = (
        f"[devig] Backfill complete: {updated} legs processed, "
        f"{available} entry devig ({100*available/max(1,updated):.1f}%), "
        f"{close_ok} close devig ({100*close_ok/max(1,updated):.1f}%)"
    )
    log.info(summary)
    print(summary)
    return {"processed": updated, "entry_devig": available, "close_devig": close_ok}


# ═══════════════════════════════════════════════════════════════════════════
# Validation (F4) — devig calibration vs outcomes
# ═══════════════════════════════════════════════════════════════════════════

def validate_devig():
    """Compare devig'd prob vs vig-included prob against actual WR.

    Returns a dict of validation results by segment.
    """
    con = sqlite3.connect(os.path.abspath(_BETS_DB), timeout=30)
    con.row_factory = sqlite3.Row

    rows = con.execute("""
        SELECT market_type, sport, is_alt_line, leg_result, win_prob,
               open_odds, devig_entry_prob, devig_close_prob, overround, devig_available
        FROM mock_bet_legs
        WHERE leg_result IN ('WIN','LOSS')
          AND devig_available = 1
          AND devig_entry_prob IS NOT NULL
    """).fetchall()

    con.close()

    if not rows:
        return {"error": "No devig'd legs found"}

    # Bucket by devig'd prob and vig-included prob, compare to actual WR
    from collections import defaultdict
    buckets_devig = defaultdict(lambda: {"wins": 0, "total": 0, "sum_prob": 0.0})
    buckets_vig = defaultdict(lambda: {"wins": 0, "total": 0, "sum_prob": 0.0})
    segment_stats = defaultdict(lambda: {
        "n": 0, "wins": 0, "sum_devig": 0.0, "sum_vig": 0.0,
        "sum_overround": 0.0, "sum_devig_sq_err": 0.0, "sum_vig_sq_err": 0.0,
    })

    for row in rows:
        won = 1 if row["leg_result"] == "WIN" else 0
        devig_p = row["devig_entry_prob"]
        overround = row["overround"] or 1.0

        # Vig-included implied prob from open_odds
        open_am = row["open_odds"]
        if open_am:
            dec = _american_to_decimal(open_am)
            vig_p = 1.0 / dec if dec and dec > 1.0 else None
        else:
            vig_p = None

        # Bucket by 5pp bands
        devig_bucket = int(devig_p * 20) * 5  # 0, 5, 10, ..., 95
        if vig_p is not None:
            vig_bucket = int(vig_p * 20) * 5

        buckets_devig[devig_bucket]["wins"] += won
        buckets_devig[devig_bucket]["total"] += 1
        buckets_devig[devig_bucket]["sum_prob"] += devig_p

        if vig_p is not None:
            buckets_vig[vig_bucket]["wins"] += won
            buckets_vig[vig_bucket]["total"] += 1
            buckets_vig[vig_bucket]["sum_prob"] += vig_p

        # Per-segment stats
        mkt = row["market_type"] or "unknown"
        sport = row["sport"] or "unknown"
        seg_key = f"{sport}|{mkt}"
        s = segment_stats[seg_key]
        s["n"] += 1
        s["wins"] += won
        s["sum_devig"] += devig_p
        s["sum_overround"] += overround
        s["sum_devig_sq_err"] += (devig_p - won) ** 2
        if vig_p is not None:
            s["sum_vig"] += vig_p
            s["sum_vig_sq_err"] += (vig_p - won) ** 2

    # Compute calibration: Brier scores
    total_devig_brier = sum((b["sum_prob"] / b["total"] - b["wins"] / b["total"]) ** 2 * b["total"]
                            for b in buckets_devig.values() if b["total"] >= 5)
    total_devig_n = sum(b["total"] for b in buckets_devig.values() if b["total"] >= 5)

    total_vig_brier = sum((b["sum_prob"] / b["total"] - b["wins"] / b["total"]) ** 2 * b["total"]
                          for b in buckets_vig.values() if b["total"] >= 5)
    total_vig_n = sum(b["total"] for b in buckets_vig.values() if b["total"] >= 5)

    return {
        "total_legs": len(rows),
        "buckets_devig": {k: dict(v) for k, v in sorted(buckets_devig.items())},
        "buckets_vig": {k: dict(v) for k, v in sorted(buckets_vig.items())},
        "segment_stats": {k: dict(v) for k, v in sorted(segment_stats.items())},
        "devig_calibration_error": total_devig_brier / max(1, total_devig_n),
        "vig_calibration_error": total_vig_brier / max(1, total_vig_n),
    }


# ═══════════════════════════════════════════════════════════════════════════
# F5a: Probability-space CLV
# ═══════════════════════════════════════════════════════════════════════════

def compute_prob_space_clv():
    """Compute CLV in probability space: clv_prob = devig_close_prob - devig_entry_prob.

    Positive = we got a better number than the close (beat the close).
    Segmented by sport / market / spread rung.
    """
    con = sqlite3.connect(os.path.abspath(_BETS_DB), timeout=30)
    con.row_factory = sqlite3.Row

    rows = con.execute("""
        SELECT l.market_type, l.sport, l.description, l.leg_result,
               l.devig_entry_prob, l.devig_close_prob, l.is_alt_line
        FROM mock_bet_legs l
        WHERE l.leg_result IN ('WIN','LOSS')
          AND l.devig_available = 1
          AND l.devig_entry_prob IS NOT NULL
          AND l.devig_close_prob IS NOT NULL
    """).fetchall()
    con.close()

    from collections import defaultdict
    segments = defaultdict(lambda: {"n": 0, "sum_clv": 0.0, "positive": 0, "negative": 0,
                                     "wins": 0, "sum_clv_winners": 0.0, "sum_clv_losers": 0.0})

    for row in rows:
        # clv_prob: positive = we beat the close (our entry was better than close)
        # If our entry implied prob is LOWER than close (market moved toward us), we beat it
        # devig_entry_prob = our implied prob at entry
        # devig_close_prob = implied prob at close
        # If close_prob > entry_prob → market moved in our direction → we got a bargain
        clv = row["devig_close_prob"] - row["devig_entry_prob"]

        mkt = row["market_type"] or "unknown"
        sport = row["sport"] or "unknown"
        won = row["leg_result"] == "WIN"

        # Parse spread rung from description
        spread = _parse_spread_from_desc(row["description"] or "")
        rung = f"rung_{spread}" if spread is not None else "no_rung"

        # Record in multiple segments
        for seg_key in [f"{sport}|{mkt}", f"{sport}|{mkt}|{rung}", f"ALL|ALL"]:
            s = segments[seg_key]
            s["n"] += 1
            s["sum_clv"] += clv
            if clv > 0:
                s["positive"] += 1
            else:
                s["negative"] += 1
            if won:
                s["wins"] += 1
                s["sum_clv_winners"] += clv
            else:
                s["sum_clv_losers"] += clv

    # Compute averages
    results = {}
    for key, s in sorted(segments.items()):
        if s["n"] < 10:
            continue
        avg_clv = s["sum_clv"] / s["n"]
        beat_rate = s["positive"] / s["n"]
        wr = s["wins"] / s["n"]
        avg_clv_win = s["sum_clv_winners"] / max(1, s["wins"])
        avg_clv_loss = s["sum_clv_losers"] / max(1, s["n"] - s["wins"])
        results[key] = {
            "n": s["n"], "avg_clv_pp": round(avg_clv * 100, 2),
            "beat_close_rate": round(beat_rate * 100, 1),
            "wr": round(wr * 100, 1),
            "avg_clv_winners_pp": round(avg_clv_win * 100, 2),
            "avg_clv_losers_pp": round(avg_clv_loss * 100, 2),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# F5b: Edge-vs-implied
# ═══════════════════════════════════════════════════════════════════════════

def compute_edge_vs_implied():
    """Compute edge = model_prob - devig_entry_prob per leg.

    Reports whether positive edge predicts higher WR.
    """
    con = sqlite3.connect(os.path.abspath(_BETS_DB), timeout=30)
    con.row_factory = sqlite3.Row

    rows = con.execute("""
        SELECT l.market_type, l.sport, l.description, l.leg_result,
               l.win_prob, l.devig_entry_prob, l.is_alt_line
        FROM mock_bet_legs l
        WHERE l.leg_result IN ('WIN','LOSS')
          AND l.devig_available = 1
          AND l.devig_entry_prob IS NOT NULL
          AND l.win_prob IS NOT NULL
    """).fetchall()
    con.close()

    from collections import defaultdict

    # Bin edges into quintiles
    edges = []
    for row in rows:
        model_prob = row["win_prob"] / 100.0  # win_prob is 0-100
        devig_prob = row["devig_entry_prob"]   # 0-1 scale
        edge = model_prob - devig_prob
        won = 1 if row["leg_result"] == "WIN" else 0
        mkt = row["market_type"] or "unknown"
        sport = row["sport"] or "unknown"
        edges.append({"edge": edge, "won": won, "mkt": mkt, "sport": sport,
                       "model_prob": model_prob, "devig_prob": devig_prob})

    if not edges:
        return {"error": "No legs with both model prob and devig"}

    # Sort by edge, compute quintile WR
    edges.sort(key=lambda x: x["edge"])
    n = len(edges)
    quintile_size = n // 5

    quintile_results = []
    for qi in range(5):
        start = qi * quintile_size
        end = start + quintile_size if qi < 4 else n
        subset = edges[start:end]
        wins = sum(e["won"] for e in subset)
        avg_edge = sum(e["edge"] for e in subset) / len(subset)
        avg_model = sum(e["model_prob"] for e in subset) / len(subset)
        avg_devig = sum(e["devig_prob"] for e in subset) / len(subset)
        wr = wins / len(subset)
        quintile_results.append({
            "quintile": f"Q{qi+1}",
            "n": len(subset),
            "avg_edge_pp": round(avg_edge * 100, 2),
            "avg_model_prob": round(avg_model * 100, 1),
            "avg_devig_prob": round(avg_devig * 100, 1),
            "wr": round(wr * 100, 1),
        })

    # Overall stats
    all_edges = [e["edge"] for e in edges]
    all_wins = [e["won"] for e in edges]
    pos_edge = [e for e in edges if e["edge"] > 0]
    neg_edge = [e for e in edges if e["edge"] <= 0]

    return {
        "total_legs": n,
        "avg_edge_pp": round(sum(all_edges) / n * 100, 2),
        "median_edge_pp": round(sorted(all_edges)[n // 2] * 100, 2),
        "pct_positive_edge": round(len(pos_edge) / n * 100, 1),
        "pos_edge_wr": round(sum(e["won"] for e in pos_edge) / max(1, len(pos_edge)) * 100, 1),
        "neg_edge_wr": round(sum(e["won"] for e in neg_edge) / max(1, len(neg_edge)) * 100, 1),
        "quintiles": quintile_results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running devig backfill...")
    result = backfill_devig(force=True)
    print(f"\nResult: {result}")
