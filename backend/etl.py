"""
etl.py — Import Pikkit CSV export into the bets database.

Usage:
    python etl.py --csv path/to/transactions.csv
    python etl.py --csv path/to/transactions.csv --reset   # wipe + reimport
"""
from __future__ import annotations

import argparse
import re
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from database import init_db, SessionLocal, Bet, BetLeg


# ─── Soccer description parser ────────────────────────────────────────────────

def parse_soccer_description(desc: str) -> dict:
    """
    Parse a Pikkit soccer leg description into structured fields.

    Pikkit soccer format: {pick_description} {market_label} {home_team} v {away_team}
    The fixture always appears at the end separated by ' v '.

    Returns dict with keys: home_team, away_team, pick_team, player,
                             line, direction, market_type, subtype
    Empty dict if not a soccer-style description.
    """
    v_idx = desc.rfind(' v ')
    if v_idx == -1:
        return {}

    away_team = desc[v_idx + 3:].strip()
    left = desc[:v_idx].strip()

    # ── Moneyline (3-way) ────────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+Moneyline\s*\(3-way\)\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'pick_team': m.group(1).strip(),
            'market_type': 'Moneyline', 'subtype': 'moneyline_3way',
        }

    # ── Simple Moneyline (without 3-way) ─────────────────────────────────────
    m = re.match(r'^(.+?)\s+Moneyline\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'pick_team': m.group(1).strip(),
            'market_type': 'Moneyline', 'subtype': 'moneyline_3way',
        }

    # ── Double Chance ────────────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+And Draw Double Chance\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'pick_team': m.group(1).strip(),
            'market_type': 'Moneyline', 'subtype': 'double_chance',
        }

    # ── 1st Half Total Goals ─────────────────────────────────────────────────
    m = re.match(r'^1st Half\s+(Over|Under)\s+([\d.]+)\s+Goals\s+1st Half\s+Over/Under\s+[\d.]+\s+Goals\s+(.+)$',
                 left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(3).strip(), 'away_team': away_team,
            'direction': m.group(1), 'line': float(m.group(2)),
            'market_type': 'Total', 'subtype': 'total_goals_1h',
        }

    # ── Full-match Total Goals ───────────────────────────────────────────────
    m = re.match(r'^(Over|Under)\s+([\d.]+)\s+Goals\s+Over/Under\s+[\d.]+\s+Goals\s+(.+)$',
                 left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(3).strip(), 'away_team': away_team,
            'direction': m.group(1), 'line': float(m.group(2)),
            'market_type': 'Total', 'subtype': 'total_goals',
        }

    # ── Corners ──────────────────────────────────────────────────────────────
    m = re.match(r'^(Over|Under)\s+([\d.]+)\s+Corners\s+Total Corners\s+[\d.]+\s+(.+)$',
                 left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(3).strip(), 'away_team': away_team,
            'direction': m.group(1), 'line': float(m.group(2)),
            'market_type': 'Total', 'subtype': 'corners',
        }

    # ── Match Shots On Target (team total) ───────────────────────────────────
    m = re.match(r'^(\d+)\s+Or More Shots On Target\s+Match Shots On Target\s+(.+)$',
                 left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'player': 'Match', 'line': int(m.group(1)),
            'market_type': 'Player Prop', 'subtype': 'shots_on_target',
        }

    # ── Player Shots On Target ───────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+Player To Have\s+(\d+)\s+Or More Shots On Target\s+(.+)$',
                 left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(3).strip(), 'away_team': away_team,
            'player': m.group(1).strip(), 'line': int(m.group(2)),
            'market_type': 'Player Prop', 'subtype': 'shots_on_target',
        }

    # ── Player Shots (total, not on-target) ──────────────────────────────────
    m = re.match(r'^(.+?)\s+Player To Have\s+(\d+)\s+Or More Shots\s+(.+)$',
                 left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(3).strip(), 'away_team': away_team,
            'player': m.group(1).strip(), 'line': int(m.group(2)),
            'market_type': 'Player Prop', 'subtype': 'shots_total',
        }

    # ── To Score Or Assist ───────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+To Score Or Assist\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'player': m.group(1).strip(),
            'market_type': 'Player Prop', 'subtype': 'score_or_assist',
        }

    # ── Anytime Goalscorer ───────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+Anytime Goalscorer\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'player': m.group(1).strip(),
            'market_type': 'Player Prop', 'subtype': 'anytime_goalscorer',
        }

    # ── Anytime Assist ───────────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+Anytime Assist\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'player': m.group(1).strip(),
            'market_type': 'Player Prop', 'subtype': 'anytime_assist',
        }

    # ── Both Teams To Score ──────────────────────────────────────────────────
    m = re.match(r'^(Yes|No)\s+Both Teams To Score\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'pick_team': m.group(1).strip(),
            'market_type': 'Total', 'subtype': 'btts',
        }

    # ── Team To Score First Goal ─────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+Team To Score the First Goal\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'pick_team': m.group(1).strip(),
            'market_type': 'Other', 'subtype': 'first_goal',
        }

    # ── To Qualify ───────────────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+To Qualify for the Next Round\s+(.+)$', left, re.IGNORECASE)
    if m:
        return {
            'home_team': m.group(2).strip(), 'away_team': away_team,
            'pick_team': m.group(1).strip(),
            'market_type': 'Moneyline', 'subtype': 'to_qualify',
        }

    return {}


def classify_soccer_market(desc: str) -> tuple[str, Optional[str]]:
    """Return (market_type, subtype) for a soccer leg description."""
    p = parse_soccer_description(desc)
    if p:
        return p.get('market_type', 'Other'), p.get('subtype')
    return 'Other', None


# ─── General market classifier ────────────────────────────────────────────────

MARKET_PATTERNS = [
    (r"moneyline",            "Moneyline"),
    (r"\d+\.?\d* run line",   "Run Line"),
    (r"\d+\.?\d* (point )?spread|alternate spread", "Spread"),
    (r"over|under|\d+\.?\d+ total|alternate total", "Total"),
    (r"to score|shots on target|player to have|to have \d+", "Player Prop"),
    (r"corners|goals|assists",  "Player Prop"),
]

def classify_market(desc: str) -> str:
    d = desc.lower()
    for pattern, label in MARKET_PATTERNS:
        if re.search(pattern, d):
            return label
    return "Other"


def parse_legs(bet_info: str, bet_id: str, sports_str: str, leagues_str: str):
    """Split pipe-delimited leg string into BetLeg rows."""
    if not bet_info or bet_info != bet_info:   # NaN check
        return []
    sports  = [s.strip() for s in str(sports_str).split("|")] if pd.notna(sports_str)  else []
    leagues = [l.strip() for l in str(leagues_str).split("|")] if pd.notna(leagues_str) else []
    raw_legs = [l.strip() for l in str(bet_info).split("|") if l.strip()]
    legs = []
    for i, desc in enumerate(raw_legs):
        sport = sports[i] if i < len(sports) else (sports[0] if sports else None)
        is_soccer = (sport or "").lower() == "soccer"

        if is_soccer:
            parsed = parse_soccer_description(desc)
            market_type = parsed.get('market_type') or classify_market(desc)
            subtype     = parsed.get('subtype')
            # team: pick_team for ML/DC, player for props, home_team for totals
            team = (
                parsed.get('pick_team') or
                parsed.get('player') or
                parsed.get('home_team')
            )
            opponent = parsed.get('away_team')
        else:
            market_type = classify_market(desc)
            subtype     = None
            team        = None
            opponent    = None

        legs.append(BetLeg(
            bet_id     = bet_id,
            leg_index  = i,
            description= desc,
            market_type= market_type,
            subtype    = subtype,
            team       = team,
            opponent   = opponent,
            sport      = sport,
            league     = leagues[i] if i < len(leagues) else (leagues[0] if leagues else None),
        ))
    return legs


def parse_dt(val) -> datetime | None:
    if pd.isna(val):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%m/%d/%Y %H:%M:%S GMT", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(val), fmt)
        except ValueError:
            continue
    return None


# ─── Main import ──────────────────────────────────────────────────────────────

def import_csv(path: str, reset: bool = False):
    init_db()
    db = SessionLocal()

    if reset:
        db.query(BetLeg).delete()
        db.query(Bet).delete()
        db.commit()
        print("[ETL] Reset: cleared existing bets and legs.")

    df = pd.read_csv(path)
    df["legs_count"] = df["bet_info"].apply(
        lambda x: len(str(x).split("|")) if pd.notna(x) else 1
    )

    inserted = skipped = 0
    for _, row in df.iterrows():
        bet_id = str(row.get("bet_id", uuid.uuid4()))

        if db.query(Bet).filter(Bet.id == bet_id).first():
            skipped += 1
            continue

        bet = Bet(
            id           = bet_id,
            source       = "pikkit",
            sportsbook   = row.get("sportsbook", "FanDuel"),
            bet_type     = row.get("type", "parlay"),
            status       = row.get("status", "UNKNOWN"),
            odds         = float(row["odds"]) if pd.notna(row.get("odds")) else None,
            closing_line = float(row["closing_line"]) if pd.notna(row.get("closing_line")) else None,
            ev           = float(row["ev"]) if pd.notna(row.get("ev")) else None,
            amount       = float(row["amount"]) if pd.notna(row.get("amount")) else 0.0,
            profit       = float(row["profit"]) if pd.notna(row.get("profit")) else None,
            legs         = int(row["legs_count"]),
            sports       = str(row.get("sports", "")) if pd.notna(row.get("sports")) else None,
            leagues      = str(row.get("leagues", "")) if pd.notna(row.get("leagues")) else None,
            bet_info     = str(row.get("bet_info", "")) if pd.notna(row.get("bet_info")) else None,
            tags         = str(row.get("tags", "")) if pd.notna(row.get("tags")) else None,
            is_mock      = False,
            time_placed  = parse_dt(row.get("time_placed_iso") or row.get("time_placed")),
            time_settled = parse_dt(row.get("time_settled_iso") or row.get("time_settled")),
        )
        db.add(bet)

        for leg in parse_legs(row.get("bet_info"), bet_id,
                               row.get("sports"), row.get("leagues")):
            db.add(leg)

        inserted += 1

    db.commit()
    db.close()
    print(f"[ETL] Done — inserted {inserted}, skipped {skipped} duplicates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True, help="Path to Pikkit CSV export")
    parser.add_argument("--reset", action="store_true", help="Wipe DB before import")
    args = parser.parse_args()
    import_csv(args.csv, args.reset)
