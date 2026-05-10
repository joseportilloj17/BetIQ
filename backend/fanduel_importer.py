"""
fanduel_importer.py — Import and weekly-sync bet CSV exports into bets.db.

Supports three CSV formats (auto-detected):
  pikkit           — Pikkit export: one row per parlay, pipe-delimited bet_info
  fanduel          — FanDuel export: one row per parlay, legs_summary column
  sportsbook_scout — Sportsbook Scout: one row per leg, with per-leg Leg Result

Functions
---------
detect_csv_format(df)                    detect which format a loaded CSV is
import_fanduel_csv(path, db)             import FanDuel CSV
import_sportsbook_scout_csv(df, db)      import Sportsbook Scout CSV
find_settled_bets(df, db)                rows in FanDuel df settled but missing from db
parse_legs_from_bet_info(bet_info, sport) parse bet_info string into structured leg dicts
backfill_bet_legs_from_bet_info(db)      enrich/create bet_legs for all bets; infer
                                          leg_result for straight bets (legs=1)
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import SessionLocal, init_db, Bet, BetLeg
from etl import parse_dt, classify_market


# ─── Helpers ──────────────────────────────────────────────────────────────────

_STATUS_MAP = {
    "WON":        "SETTLED_WIN",
    "LOST":       "SETTLED_LOSS",
    "CASHED_OUT": "SETTLED_WIN",
}


def _norm_bet_type(t: str) -> str:
    return "straight" if str(t).upper() == "SGL" else "parlay"


def _am_to_dec(am) -> Optional[float]:
    try:
        am = int(am)
        return round(1 + am / 100, 4) if am > 0 else round(1 + 100 / abs(am), 4)
    except Exception:
        return None


def _calc_profit(row: pd.Series) -> float:
    result = str(row.get("result", "")).upper()
    pandl  = float(row.get("pandl", 0) or 0)
    stake  = float(row.get("wager_amount", 0) or 0)
    if result == "WON":
        return round(pandl, 2)
    if result == "CASHED_OUT":
        return round(pandl - stake, 2) if pandl > 0 else 0.0
    return round(-stake, 2)


def _dedup_key(row: pd.Series) -> str:
    """MD5 of (placed_date_minute, legs_summary, wager_amount) — catches content dupes."""
    placed  = str(row.get("placed_date", ""))[:16]
    legs    = str(row.get("legs_summary", ""))
    amount  = str(round(float(row.get("wager_amount", 0)), 2))
    return hashlib.md5(f"{placed}|{legs}|{amount}".encode()).hexdigest()


# ─── Market classification (extended) ────────────────────────────────────────
# Extends etl.classify_market with patterns not covered there.

_MARKET_PATTERNS_EXT = [
    # Spreads — must come before generic spread/run checks
    (r"alternate run lines?|alt run lines?",            "Spread"),
    (r"puck line|alternate puck lines?",                "Spread"),
    (r"alt spread|alternate spread",                    "Spread"),
    (r"\d+\.?\d* run line|run line",                   "Spread"),
    (r"spread|\bats\b|handicap",                        "Spread"),
    # ── Player props BEFORE generic over/under ──────────────────────────────
    # These stat keywords are unambiguous — always props, never game totals.
    # Must run before the Total pattern so "Under 8.5 Assists" isn't Total.
    (r"^player prop$|player prop\b",                               "Player Prop"),
    (r"rushing yds?|receiving yds?|passing yds?|passing yards?",   "Player Prop"),
    (r"\bassists?\b(?!\s+wins)",                                    "Player Prop"),  # assists (not "assists wins")
    (r"\brebounds?\b",                                              "Player Prop"),
    (r"strikeouts?|home runs?|\bhits?\b",                          "Player Prop"),
    (r"to score \d+\+?\s*points?|to have \d+|to score or assist",  "Player Prop"),
    (r"alt points?|alt rebounds?|alt assists?|alt yards?",          "Player Prop"),
    (r"shots? on target",                                           "Player Prop"),
    (r"rushing yards?|receiving yards?",                            "Player Prop"),
    (r"to score|anytime|1st|\btd\b|touchdowns?",                   "Player Prop"),
    (r"points?\s+\(?\d|rebounds?\s+\(?\d|\d+\+?\s*pts?",           "Player Prop"),
    # Totals — generic over/under (game-level, checked after prop keywords)
    (r"over/under|o/u|\btotal\b|total runs|total goals|total corners", "Total"),
    (r"\bover\b|\bunder\b",                                            "Total"),
    # Moneyline
    (r"moneyline|\bml\b|\bto win\b|match winner|\(3-way\)", "Moneyline"),
]

_ODDS_RE = re.compile(r'([+-]\d{2,4})(?:\s*$|\s+[-–])')


def _classify_extended(description: str, market_hint: str = "") -> str:
    """
    Classify a leg description into a canonical market bucket.
    Checks extended patterns that cover FanDuel alt markets.
    Falls back to etl.classify_market.
    """
    text = (description + " " + market_hint).lower()
    for pattern, label in _MARKET_PATTERNS_EXT:
        if re.search(pattern, text):
            return label
    return classify_market(description)


def _extract_odds_str(description: str) -> str:
    """
    Extract trailing American odds token from a description string.

    FanDuel-format descriptions end with the odds: "... @ Team -229"
    Pikkit descriptions do NOT include per-leg odds — returns "".
    """
    m = re.search(r'([+-]\d{2,4})\s*$', description.strip())
    return m.group(1) if m else ""


def _extract_team_or_player(description: str) -> str:
    """
    Best-effort extraction of team/player name from a leg description.

    FanDuel format: "{pick} ({market}) — {matchup} {odds}"
      → text before the first " (" is the team/player/pick label
      e.g. "New York Knicks (Moneyline) — ..." → "New York Knicks"
      e.g. "Jalen Brunson Over 22.5 (Jalen Brunson - Alt Points) — ..." → "Jalen Brunson Over 22.5"

    Pikkit format: "{pick_description} {matchup}"
      → text before first market keyword
      e.g. "New York Yankees Moneyline Miami Marlins..." → "New York Yankees"
      e.g. "Over 3.5 Total Runs Chicago Cubs..." → "" (market is first word)
    """
    # FanDuel format: has " (" with market keyword inside
    if " (" in description and ") —" in description:
        return description.split(" (")[0].strip()

    # Pikkit / generic: find first market keyword and take text before it
    _MARKET_KEYWORDS = [
        "moneyline", "run line", "puck line", "spread", "total",
        " over ", " under ", "player to ", "to score", "to have",
        "anytime", "shots on target", "corners", "assists", "rebounds",
    ]
    desc_lower = description.lower()
    earliest = len(description)
    for kw in _MARKET_KEYWORDS:
        idx = desc_lower.find(kw)
        if 0 < idx < earliest:
            earliest = idx
    if earliest < len(description):
        return description[:earliest].strip()
    return ""


# ─── Leg parser ───────────────────────────────────────────────────────────────

def parse_legs_from_bet_info(bet_info: str, sport: str = "") -> list[dict]:
    """
    Parse a pipe-delimited bet_info string into structured leg dicts.

    Handles two stored formats:

    FanDuel (created by _parse_legs from CSV import):
      "{pick} ({market}) — {matchup} {odds}"
      e.g. "New York Knicks (Moneyline) — Atlanta Hawks @ New York Knicks -229"

    Pikkit (raw Pikkit CSV descriptions, no per-leg odds):
      "{pick_description} {matchup}"
      e.g. "New York Yankees Moneyline Miami Marlins (P Fairbanks) @ New York Yankees (M Fried)"

    Returns list of dicts:
      {
        description:    str,           # the raw segment
        team_or_player: str,           # best-effort pick name
        market_type:    str,           # Moneyline | Spread | Total | Player Prop | Other
        odds_str:       str,           # American odds e.g. "-229" or "" when not embedded
        matchup:        str,           # "Team A @ Team B" or ""
        sport:          str,           # passed-through from parent bet
      }
    """
    if not bet_info:
        return []

    segments = [s.strip() for s in str(bet_info).split("|") if s.strip()]
    legs = []
    for seg in segments:
        # Extract market hint from parentheses (FanDuel format)
        mkt_hint_m = re.search(r'\(([^)]+)\)', seg)
        mkt_hint = mkt_hint_m.group(1) if mkt_hint_m else ""

        # Extract matchup: "Team @ Team" or "Team v Team"
        matchup_m = re.search(r'([A-Z][^@\-–]+(?:@|v\s)[A-Z][^@\-–\d]+)', seg)
        matchup = matchup_m.group(0).strip() if matchup_m else ""

        legs.append({
            "description":    seg,
            "team_or_player": _extract_team_or_player(seg),
            "market_type":    _classify_extended(seg, mkt_hint),
            "odds_str":       _extract_odds_str(seg),
            "matchup":        matchup,
            "sport":          sport,
        })
    return legs


# ─── Backfill / enrichment ────────────────────────────────────────────────────

def backfill_bet_legs_from_bet_info(db: Session) -> dict:
    """
    Enrich and gap-fill bet_legs for all real bets in the database.

    Three operations:

    1. Create missing legs — for any bet with bet_info and legs>0 but
       NO rows yet in bet_legs (e.g. 'app' source manual entries).

    2. Enrich existing legs — for each bet_leg where odds_str IS NULL
       and/or team IS NULL, re-parse its description to populate those
       fields.  Also fixes market_type = "Other" for legs that are
       actually Spreads (Alternate Run Lines, Puck Line, etc.).

    3. Infer leg_result for straight bets — for bets with legs=1 and a
       settled status, the single leg outcome equals the parlay outcome.
       This is 100% reliable.  Parlay leg_result stays NULL.

    Returns:
      {
        legs_created:         int,   # new BetLeg rows inserted
        legs_enriched:        int,   # existing rows updated (odds/team/market)
        leg_results_inferred: int,   # straight-bet leg_result set
      }
    """
    from sqlalchemy import text as sqla_text

    legs_created = 0
    legs_enriched = 0
    leg_results_inferred = 0

    # ── 1. Create missing legs ────────────────────────────────────────────────
    bets_needing_legs = (
        db.query(Bet)
        .filter(
            Bet.bet_info.isnot(None),
            Bet.bet_info != "",
            Bet.is_mock.is_(False),
        )
        .all()
    )

    existing_bet_ids_with_legs: set[str] = {
        row[0] for row in db.query(BetLeg.bet_id).distinct().all()
    }

    for bet in bets_needing_legs:
        if bet.id in existing_bet_ids_with_legs:
            continue  # already has legs — handled in step 2
        parsed = parse_legs_from_bet_info(bet.bet_info, bet.sports or "")
        for i, p in enumerate(parsed):
            db.add(BetLeg(
                bet_id      = bet.id,
                leg_index   = i,
                description = p["description"],
                market_type = p["market_type"],
                team        = p["team_or_player"] or None,
                sport       = p["sport"] or None,
                odds_str    = p["odds_str"] or None,
            ))
            legs_created += 1

    if legs_created:
        db.flush()

    # ── 2. Enrich existing legs ───────────────────────────────────────────────
    # Load all legs that need enrichment: odds_str null, team null, or market Other
    all_legs = db.query(BetLeg).all()

    for leg in all_legs:
        if not leg.description:
            continue

        changed = False

        # Parse the stored description to extract fields
        p = parse_legs_from_bet_info(leg.description, leg.sport or "")
        if not p:
            continue
        parsed_leg = p[0]  # description was a single segment

        # Populate odds_str from description (FanDuel legs have trailing odds)
        if not leg.odds_str and parsed_leg["odds_str"]:
            leg.odds_str = parsed_leg["odds_str"]
            changed = True

        # Populate team when missing
        if not leg.team and parsed_leg["team_or_player"]:
            leg.team = parsed_leg["team_or_player"]
            changed = True

        # Fix market_type=Other legs that are actually Spread/Prop/etc.
        if leg.market_type == "Other":
            new_market = parsed_leg["market_type"]
            if new_market != "Other":
                leg.market_type = new_market
                changed = True

        if changed:
            legs_enriched += 1

    if legs_enriched:
        db.flush()

    # ── 3. Infer leg_result for straight bets (legs=1, 100% reliable) ─────────
    straight_bets = (
        db.query(Bet)
        .filter(
            Bet.legs == 1,
            Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
            Bet.is_mock.is_(False),
        )
        .all()
    )
    straight_ids = {b.id for b in straight_bets}
    straight_outcome = {b.id: ("WIN" if b.status == "SETTLED_WIN" else "LOSS") for b in straight_bets}

    straight_legs = (
        db.query(BetLeg)
        .filter(BetLeg.bet_id.in_(straight_ids))
        .all()
    )
    for leg in straight_legs:
        if leg.leg_result is None:
            leg.leg_result = straight_outcome.get(leg.bet_id)
            if leg.leg_result:
                leg_results_inferred += 1

    db.commit()

    return {
        "legs_created":         legs_created,
        "legs_enriched":        legs_enriched,
        "leg_results_inferred": leg_results_inferred,
    }


# ─── Format detection ─────────────────────────────────────────────────────────

def detect_csv_format(df: pd.DataFrame) -> str:
    """
    Detect which CSV format a loaded DataFrame represents.

    Returns one of: "sportsbook_scout" | "fanduel" | "pikkit" | "unknown"

    Detection rules (checked in order):
      sportsbook_scout — has "External Bet ID", "Primary", AND "Leg Result" columns
      fanduel          — has "legs_summary" OR "bet_id" + "american_odds" columns
      pikkit           — has "bet_id" + "bet_info" columns (Pikkit export)
    """
    cols = set(df.columns)

    if {"External Bet ID", "Primary", "Leg Result"}.issubset(cols):
        return "sportsbook_scout"

    if "legs_summary" in cols or ("bet_id" in cols and "american_odds" in cols):
        return "fanduel"

    if "bet_id" in cols and "bet_info" in cols:
        return "pikkit"

    return "unknown"


# ─── Sportsbook Scout status / leg result maps ────────────────────────────────

_SS_STATUS_MAP = {
    "WIN":  "SETTLED_WIN",
    "LOSS": "SETTLED_LOSS",
    "PUSH": "SETTLED_PUSH",
}

_SS_LEG_RESULT_MAP = {
    "WIN":  "WIN",
    "LOSS": "LOSS",
    "PUSH": "PUSH",
}


def _ss_parse_dt(val) -> Optional[datetime]:
    """Parse a Sportsbook Scout datetime string; tolerates multiple formats."""
    if pd.isna(val) or not val:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    return None


def _ss_profit(result: str, amount: float, payout: float) -> float:
    """Compute profit from a Sportsbook Scout row."""
    r = result.upper()
    if r == "WIN":
        return round(payout - amount, 2)
    if r == "PUSH":
        return 0.0
    return round(-amount, 2)


def _pikkit_crosswalk_key(b: Bet) -> Optional[tuple]:
    """
    Build a (date, amount, legs) tuple for matching Pikkit bets to SS bets.
    Returns None if time_placed is missing.
    """
    if not b.time_placed:
        return None
    return (
        b.time_placed.strftime("%Y-%m-%d"),
        round(b.amount or 0, 2),
        b.legs or 1,
    )


# ─── Sportsbook Scout importer ────────────────────────────────────────────────

def import_sportsbook_scout_csv(df: pd.DataFrame, db: Session) -> dict:
    """
    Import a Sportsbook Scout CSV into bets.db.

    Sportsbook Scout exports one row per leg.  Each group of rows sharing the
    same "External Bet ID" represents one parlay/straight bet.

    Deduplication strategy (three tiers):
      1. ss_{External Bet ID} already in bets table → skip entirely.
      2. Matching Pikkit bet found by (date, amount, legs_count) → backfill
         leg_result onto its existing bet_legs rows; don't create a new bet.
         Match is skipped if all matched legs already have leg_result set.
      3. No existing record found → create new bet + bet_legs with ss_ prefix.

    Returns:
      {
        inserted:            int,  # new bets created (ss_ prefix)
        skipped_id:          int,  # ss_ bet_id already existed
        pikkit_crosswalked:  int,  # Pikkit bets whose legs got leg_result backfilled
        legs_created:        int,  # total bet_legs rows written for new ss_ bets
        leg_results_written: int,  # leg_result values set (new + crosswalk)
        total_groups:        int,  # total distinct External Bet IDs processed
      }
    """
    # Normalise column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Required column aliases — Sportsbook Scout may vary capitalisation
    col = {
        "ext_id":    next((c for c in df.columns if c.lower() == "external bet id"), "External Bet ID"),
        "primary":   next((c for c in df.columns if c.lower() == "primary"), "Primary"),
        "leg_result":next((c for c in df.columns if c.lower() == "leg result"), "Leg Result"),
        "sport":     next((c for c in df.columns if c.lower() == "sport"), "Sport"),
        "market":    next((c for c in df.columns if c.lower() == "market"), "Market"),
        "odds":      next((c for c in df.columns if c.lower() == "odds"), "Odds"),
        "selection": next((c for c in df.columns if c.lower() == "selection"), "Selection"),
        "event":     next((c for c in df.columns if c.lower() == "event"), "Event"),
        "amount":    next((c for c in df.columns if c.lower() == "bet amount"), "Bet Amount"),
        "payout":    next((c for c in df.columns if c.lower() == "potential payout"), "Potential Payout"),
        "placed":    next((c for c in df.columns if c.lower() == "placed date"), "Placed Date"),
        "settled":   next((c for c in df.columns if c.lower() == "settled date"), "Settled Date"),
        "result":    next((c for c in df.columns if c.lower() == "result"), "Result"),
    }

    # ── Build Pikkit crosswalk index ─────────────────────────────────────────
    # Index: (date_str, amount, legs) → Pikkit Bet object
    pikkit_bets = (
        db.query(Bet)
        .filter(Bet.source == "pikkit", Bet.is_mock.is_(False))
        .all()
    )
    pikkit_index: dict[tuple, Bet] = {}
    for b in pikkit_bets:
        k = _pikkit_crosswalk_key(b)
        if k:
            pikkit_index[k] = b  # last one wins if collision (rare)

    # ── Existing ss_ bet IDs ─────────────────────────────────────────────────
    existing_ss_ids: set[str] = {
        row[0] for row in db.query(Bet.id).filter(Bet.source == "sportsbook_scout").all()
    }

    inserted = 0
    skipped_id = 0
    pikkit_crosswalked = 0
    legs_created = 0
    leg_results_written = 0

    groups = df.groupby(col["ext_id"], sort=False)

    for ext_id, group in groups:
        ss_bet_id = f"ss_{ext_id}"

        # ── Tier 1: already imported ─────────────────────────────────────────
        if ss_bet_id in existing_ss_ids:
            skipped_id += 1
            continue

        # Extract primary row (Primary == 1) for parlay-level fields
        primary_mask = group[col["primary"]].astype(str).str.strip() == "1"
        primary_rows = group[primary_mask]
        primary_row  = primary_rows.iloc[0] if not primary_rows.empty else group.iloc[0]

        amount        = float(primary_row.get(col["amount"], 0) or 0)
        payout        = float(primary_row.get(col["payout"], 0) or 0)
        result_raw    = str(primary_row.get(col["result"], "")).strip()
        result_upper  = result_raw.upper()
        placed_dt     = _ss_parse_dt(primary_row.get(col["placed"]))
        settled_dt    = _ss_parse_dt(primary_row.get(col["settled"]))
        n_legs        = len(group)

        # ── Tier 2: Pikkit crosswalk ─────────────────────────────────────────
        if placed_dt:
            crosswalk_key = (
                placed_dt.strftime("%Y-%m-%d"),
                round(amount, 2),
                n_legs,
            )
            pikkit_bet = pikkit_index.get(crosswalk_key)
        else:
            pikkit_bet = None

        if pikkit_bet is not None:
            # Fetch existing bet_legs for this Pikkit bet
            existing_legs = (
                db.query(BetLeg)
                .filter(BetLeg.bet_id == pikkit_bet.id)
                .order_by(BetLeg.leg_index)
                .all()
            )

            # Skip if all legs already have leg_result (already crosswalked)
            if existing_legs and all(l.leg_result for l in existing_legs):
                skipped_id += 1
                continue

            # Write leg_result onto matched legs
            leg_rows = group.reset_index(drop=True)
            for i, leg_orm in enumerate(existing_legs):
                if i >= len(leg_rows):
                    break
                lr_raw = str(leg_rows.at[i, col["leg_result"]]).strip().upper()
                lr     = _SS_LEG_RESULT_MAP.get(lr_raw)
                if lr and leg_orm.leg_result != lr:
                    leg_orm.leg_result = lr
                    # Also enrich odds_str and market_type if missing
                    if not leg_orm.odds_str:
                        raw_odds = leg_rows.at[i, col["odds"]]
                        if pd.notna(raw_odds):
                            leg_orm.odds_str = str(int(float(raw_odds))) if str(raw_odds).lstrip("-+").isdigit() else str(raw_odds)
                    leg_results_written += 1

            db.flush()
            pikkit_crosswalked += 1
            continue

        # ── Tier 3: new ss_ bet ──────────────────────────────────────────────
        # Build pipe-delimited bet_info from leg descriptions
        leg_descs = []
        for _, leg_row in group.iterrows():
            sel = str(leg_row.get(col["selection"], "") or "").strip()
            mkt = str(leg_row.get(col["market"],    "") or "").strip()
            evt = str(leg_row.get(col["event"],     "") or "").strip()
            raw_odds = leg_row.get(col["odds"])
            odds_part = f" {int(float(raw_odds))}" if pd.notna(raw_odds) else ""
            leg_descs.append(f"{sel} ({mkt}) — {evt}{odds_part}".strip(" —"))
        bet_info_str = " | ".join(leg_descs)

        # Most common sport across legs (handle ties by first occurrence)
        sports_list = [
            str(group.iloc[i].get(col["sport"], "") or "").strip()
            for i in range(len(group))
            if str(group.iloc[i].get(col["sport"], "") or "").strip()
        ]
        from collections import Counter
        dominant_sport = Counter(sports_list).most_common(1)[0][0] if sports_list else ""

        # Compute parlay decimal odds from payout/stake
        dec_odds = round(payout / amount, 4) if amount > 0 else None

        bet = Bet(
            id           = ss_bet_id,
            source       = "sportsbook_scout",
            sportsbook   = "FanDuel",          # SS exports are typically FanDuel
            bet_type     = "straight" if n_legs == 1 else "parlay",
            status       = _SS_STATUS_MAP.get(result_upper, "SETTLED_LOSS"),
            odds         = dec_odds,
            amount       = amount,
            profit       = _ss_profit(result_raw, amount, payout),
            legs         = n_legs,
            sports       = dominant_sport,
            bet_info     = bet_info_str,
            is_mock      = False,
            time_placed  = placed_dt,
            time_settled = settled_dt,
        )
        db.add(bet)

        # Create bet_legs with per-leg outcomes
        for i, (_, leg_row) in enumerate(group.iterrows()):
            sel      = str(leg_row.get(col["selection"], "") or "").strip()
            mkt_raw  = str(leg_row.get(col["market"],    "") or "").strip()
            evt      = str(leg_row.get(col["event"],     "") or "").strip()
            spt      = str(leg_row.get(col["sport"],     "") or "").strip()
            raw_odds = leg_row.get(col["odds"])
            lr_raw   = str(leg_row.get(col["leg_result"], "") or "").strip().upper()

            odds_str_val = None
            if pd.notna(raw_odds):
                try:
                    odds_str_val = str(int(float(raw_odds)))
                except (ValueError, TypeError):
                    odds_str_val = str(raw_odds)

            lr = _SS_LEG_RESULT_MAP.get(lr_raw)

            description = f"{sel} ({mkt_raw}) — {evt}"
            if odds_str_val:
                description += f" {odds_str_val}"

            db.add(BetLeg(
                bet_id      = ss_bet_id,
                leg_index   = i,
                description = description.strip(" —"),
                market_type = _classify_extended(mkt_raw + " " + sel, mkt_raw),
                team        = sel or None,
                sport       = spt or None,
                odds_str    = odds_str_val,
                leg_result  = lr,
            ))
            legs_created += 1
            if lr:
                leg_results_written += 1

        existing_ss_ids.add(ss_bet_id)
        inserted += 1

    db.commit()

    return {
        "inserted":            inserted,
        "skipped_id":          skipped_id,
        "pikkit_crosswalked":  pikkit_crosswalked,
        "legs_created":        legs_created,
        "leg_results_written": leg_results_written,
        "total_groups":        len(groups),
    }


def _parse_legs(summary: str, bet_id: str, sport: str) -> tuple[str, list[BetLeg]]:
    """
    Parse FanDuel legs_summary into pipe-delimited bet_info string + BetLeg rows.

    FanDuel format: "N | Away @ Home | Market | Pick | Odds || N | ..."
    """
    if not summary or (isinstance(summary, float) and pd.isna(summary)):
        return "", []
    blocks    = [b.strip() for b in str(summary).split("||") if b.strip()]
    desc_parts: list[str] = []
    leg_rows:   list[BetLeg] = []
    for i, block in enumerate(blocks):
        parts   = [p.strip() for p in block.split("|")]
        matchup = parts[1] if len(parts) > 1 else ""
        market  = parts[2] if len(parts) > 2 else ""
        pick    = parts[3] if len(parts) > 3 else ""
        odds    = parts[4] if len(parts) > 4 else ""
        desc    = f"{pick} ({market}) — {matchup} {odds}".strip(" —")
        desc_parts.append(desc)
        leg_rows.append(BetLeg(
            bet_id      = bet_id,
            leg_index   = i,
            description = desc,
            market_type = classify_market(market + " " + pick),
            sport       = sport,
        ))
    return " | ".join(desc_parts), leg_rows


def _promo_fields(row: pd.Series) -> tuple[str, Optional[int]]:
    reward = str(row.get("reward_type", "") or "").upper()
    boost  = row.get("price_boost_pct")
    if "BOOST" in reward:
        promo_type = "profit_boost"
    elif "FREE" in reward:
        promo_type = "free_bet"
    else:
        promo_type = "none"
    boost_val = int(boost) if pd.notna(boost) and boost else None
    return promo_type, boost_val


# ─── Core import ──────────────────────────────────────────────────────────────

def import_fanduel_csv(path: str, db: Optional[Session] = None) -> dict:
    """
    Import all rows from a FanDuel CSV into bets.db.

    Deduplicates on both:
      - Primary key  : fd_{bet_id}
      - Content hash : (placed_date_minute, legs_summary, wager_amount)

    Returns summary dict: {inserted, skipped_id, skipped_key, cashed_out, total}
    """
    own_db = db is None
    if own_db:
        init_db()
        db = SessionLocal()

    try:
        df = pd.read_csv(path)
        inserted = skipped_id = skipped_key = 0
        cashed_out_ids: list[str] = []
        seen_keys: set[str] = set()

        # Seed seen_keys from already-imported fanduel rows to catch re-imports
        for eb_info, eamt, etm in db.query(
            Bet.bet_info, Bet.amount, Bet.time_placed
        ).filter(Bet.source == "fanduel").all():
            if etm:
                ek = hashlib.md5(
                    f"{str(etm)[:16]}|{eb_info}|{round(eamt or 0, 2)}".encode()
                ).hexdigest()
                seen_keys.add(ek)

        for _, row in df.iterrows():
            bet_id = f"fd_{row['bet_id']}"

            if db.query(Bet).filter(Bet.id == bet_id).first():
                skipped_id += 1
                continue

            dk = _dedup_key(row)
            if dk in seen_keys:
                skipped_key += 1
                continue
            seen_keys.add(dk)

            result         = str(row.get("result", "")).upper()
            sport          = str(row.get("sport", ""))
            bet_info, legs = _parse_legs(row.get("legs_summary", ""), bet_id, sport)
            promo_type, boost_val = _promo_fields(row)

            bet = Bet(
                id           = bet_id,
                source       = "fanduel",
                sportsbook   = "FanDuel",
                bet_type     = _norm_bet_type(row.get("bet_type", "parlay")),
                status       = _STATUS_MAP.get(result, "SETTLED_LOSS"),
                odds         = _am_to_dec(row.get("american_odds")),
                amount       = float(row.get("wager_amount", 0) or 0),
                profit       = _calc_profit(row),
                legs         = int(row.get("num_legs", 1) or 1),
                sports       = sport,
                bet_info     = bet_info,
                is_mock      = False,
                time_placed  = parse_dt(row.get("placed_date")),
                time_settled = parse_dt(row.get("settled_date")),
                promo_type         = promo_type,
                promo_boosted_odds = boost_val,
            )
            db.add(bet)
            for leg in legs:
                db.add(leg)

            if result == "CASHED_OUT":
                cashed_out_ids.append(bet_id)
            inserted += 1

        db.commit()

        # Set cashed_out=1 via raw SQL (column added via migration, not in ORM)
        for bid in cashed_out_ids:
            db.execute(text("UPDATE bets SET cashed_out=1 WHERE id=:id"), {"id": bid})
        if cashed_out_ids:
            db.commit()

        return {
            "inserted":    inserted,
            "skipped_id":  skipped_id,
            "skipped_key": skipped_key,
            "cashed_out":  len(cashed_out_ids),
            "total":       len(df),
            "file":        os.path.basename(path),
        }

    finally:
        if own_db:
            db.close()


# ─── Weekly sync helper ───────────────────────────────────────────────────────

def find_settled_bets(df: pd.DataFrame, db: Session) -> pd.DataFrame:
    """
    Filter df to rows that:
      - are settled on FanDuel (result in WON / LOST / CASHED_OUT)
      - do NOT yet exist in bets table by either primary key or content hash

    Returns the subset of df rows that should be upserted.
    """
    existing_ids: set[str] = {
        row[0] for row in db.query(Bet.id).filter(Bet.source == "fanduel").all()
    }
    existing_keys: set[str] = set()
    for eb_info, eamt, etm in db.query(
        Bet.bet_info, Bet.amount, Bet.time_placed
    ).filter(Bet.source == "fanduel").all():
        if etm:
            ek = hashlib.md5(
                f"{str(etm)[:16]}|{eb_info}|{round(eamt or 0, 2)}".encode()
            ).hexdigest()
            existing_keys.add(ek)

    settled_results = {"WON", "LOST", "CASHED_OUT"}
    mask = []
    for _, row in df.iterrows():
        result = str(row.get("result", "")).upper()
        if result not in settled_results:
            mask.append(False)
            continue
        bet_id = f"fd_{row['bet_id']}"
        if bet_id in existing_ids:
            mask.append(False)
            continue
        if _dedup_key(row) in existing_keys:
            mask.append(False)
            continue
        mask.append(True)

    return df[mask].reset_index(drop=True)


def _latest_fanduel_csv(imports_dir: str) -> Optional[str]:
    """Return the most recently modified fanduel_*.csv in imports_dir, or None."""
    if not os.path.isdir(imports_dir):
        return None
    candidates = [
        os.path.join(imports_dir, f)
        for f in os.listdir(imports_dir)
        if f.startswith("fanduel_") and f.endswith(".csv")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run_weekly_sync(imports_dir: Optional[str] = None) -> dict:
    """
    Load the latest fanduel_*.csv from data/imports/, find new settled bets,
    upsert them, and return a summary.

    Called by the Sunday 8 AM scheduler job and POST /api/sync/fanduel.
    """
    if imports_dir is None:
        imports_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "imports"
        )

    csv_path = _latest_fanduel_csv(imports_dir)

    # Also check data/ root for manually dropped files
    if csv_path is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        candidates = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("fanduel_") and f.endswith(".csv")
        ]
        if candidates:
            csv_path = max(candidates, key=os.path.getmtime)

    if csv_path is None:
        return {
            "status":  "no_file",
            "message": "No fanduel_*.csv found in data/imports/ or data/",
            "new_bets": 0, "settled": 0, "skipped": 0, "errors": 0,
        }

    init_db()
    db = SessionLocal()
    try:
        df = pd.read_csv(csv_path)
        new_df = find_settled_bets(df, db)

        if new_df.empty:
            return {
                "status":   "up_to_date",
                "message":  f"No new settled bets in {os.path.basename(csv_path)}",
                "file":     os.path.basename(csv_path),
                "new_bets": 0, "settled": 0, "skipped": 0, "errors": 0,
            }

        # Write a temp CSV of just the new rows and run the importer
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            new_df.to_csv(tmp, index=False)
            tmp_path = tmp.name

        try:
            summary = import_fanduel_csv(tmp_path, db)
        finally:
            os.unlink(tmp_path)

        # Backfill cash-out decisions for newly inserted rows
        try:
            from auto_settle import _backfill_cash_out_decisions
            _backfill_cash_out_decisions(db)
        except Exception:
            pass

        # Enrich leg data for newly inserted bets
        try:
            backfill_bet_legs_from_bet_info(db)
        except Exception:
            pass

        return {
            "status":   "synced",
            "file":     os.path.basename(csv_path),
            "new_bets": summary["inserted"],
            "settled":  summary["inserted"],
            "skipped":  summary["skipped_id"] + summary["skipped_key"],
            "errors":   0,
            "synced_at": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "new_bets": 0, "settled": 0, "skipped": 0, "errors": 1,
        }
    finally:
        db.close()
