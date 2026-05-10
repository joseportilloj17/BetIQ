"""
fanduel_api_importer.py — Import FanDuel bet history from the FanDuel sportsbook API.

This module handles the richer JSON API format (fetch-my-bets endpoint) which
contains boost metadata, per-leg player prop flags, SGM indicators, and exact
boost EV via original vs boosted price comparison — none of which is available
in the CSV export.

Tables
------
  fanduel_bets       — one row per bet (header + aggregate fields)
  fanduel_bet_legs   — one row per leg (all leg-level detail)

Both tables live in data/bets.db alongside the existing bets/bet_legs tables.

Entry points
------------
  create_tables()                    ensure schema exists
  parse_and_upsert(raw_json, conn)   parse API response dict, upsert all bets
  fetch_all_pages(token, ...)        paginate through fetch-my-bets until done
  import_from_file(path, conn)       load JSON file, parse, upsert
  run_analysis(conn)                 run the five analysis queries, return dict
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

# ── DB path (same bets.db used by the rest of the backend) ────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_BETS_DB  = os.path.abspath(os.path.join(_DATA_DIR, "bets.db"))

# ── FanDuel API constants ─────────────────────────────────────────────────────
_FD_BASE_URL = "https://api.sportsbook.fanduel.com/sbapi/fetch-my-bets"
_FD_APP_KEY  = "FhMFpcPWXMeyZxOx"
_FD_PAGE     = 20   # records per page
_FD_REGION   = "IL"


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS fanduel_bets (
    bet_receipt_id      TEXT PRIMARY KEY,
    bet_id              TEXT,
    bet_type            TEXT,       -- ACC4, ACC7, SGL, TBL, SGM etc
    legs                INTEGER,
    stake               REAL,
    result              TEXT,       -- WON / LOST / CASHED_OUT / OPEN
    odds_final          REAL,       -- betPrices.betPrice.decimalPrice (boosted)
    odds_original       REAL,       -- betPrices.originalBetPrice.decimalPrice (pre-boost)
    pnl                 REAL,       -- actual profit/loss (net of stake)
    placed_date         TEXT,       -- ISO 8601
    settled_date        TEXT,       -- ISO 8601, NULL if still open
    reward_type         TEXT,       -- rewardUsed.type: PROFIT_BOOST / NO_SWEAT / BONUS_BET / NULL
    boost_pct           REAL,       -- features.priceBoost.generosityPercentage / 100 (e.g. 0.25)
    is_sgm              INTEGER,    -- 1 if sameGameMultis=true
    has_player_props    INTEGER,    -- 1 if any leg has isPlayerSelection=true
    imported_at         TEXT        -- UTC timestamp of import
);

CREATE TABLE IF NOT EXISTS fanduel_bet_legs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_receipt_id      TEXT NOT NULL,
    leg_number          INTEGER,
    result              TEXT,       -- WON / LOST / PLACED / VOID
    selection           TEXT,       -- e.g. "Chicago Cubs -1.5"
    market_type         TEXT,       -- MONEY_LINE, ALTERNATE_RUN_LINES, TO_SCORE_20+_POINTS etc
    competition         TEXT,       -- league / competition name
    price               REAL,       -- decimal price for this leg
    original_price      REAL,       -- pre-boost decimal price (NULL if not boosted)
    is_player_selection INTEGER,    -- 1 if player prop leg
    handicap            TEXT,       -- spread / handicap value (string, may be "+1.5")
    over_under          TEXT,       -- "over" / "under" for totals legs
    FOREIGN KEY (bet_receipt_id) REFERENCES fanduel_bets(bet_receipt_id)
);

CREATE INDEX IF NOT EXISTS idx_fdlegs_receipt ON fanduel_bet_legs(bet_receipt_id);
CREATE INDEX IF NOT EXISTS idx_fdbets_placed  ON fanduel_bets(placed_date);
CREATE INDEX IF NOT EXISTS idx_fdbets_result  ON fanduel_bets(result);
"""


def create_tables(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create fanduel_bets and fanduel_bet_legs tables if they don't exist."""
    close = conn is None
    if conn is None:
        conn = sqlite3.connect(_BETS_DB)
    try:
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()
    finally:
        if close:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _iso(ts) -> Optional[str]:
    """Normalise any FanDuel timestamp string to ISO 8601 UTC."""
    if not ts:
        return None
    s = str(ts).replace("T", " ").rstrip("Z")
    return s


def _parse_bet(raw: dict) -> tuple[dict, list[dict]]:
    """
    Parse one raw bet object from the fetch-my-bets response into
    (bet_row, [leg_row, ...]) dicts ready for SQLite upsert.

    Actual API structure (verified 2026-05-03):
      raw.currentSize              — stake
      raw.betPrices.betPrice       — final (boosted) odds object
      raw.betPrices.originalBetPrice — pre-boost odds object (only on boosted bets)
      raw.features.priceBoost.generosityPercentage — boost % (integer)
      raw.rewardUsed.type          — NO_SWEAT / PROFIT_BOOST / BONUS_BET
      raw.sameGameMultis           — bool
      raw.pandl                    — profit on WON/CASHED_OUT (0 on LOST)
      raw.legs[].result            — leg result
      raw.legs[].parts[0]          — the actual selection detail:
          .selectionName, .marketType, .competitionName,
          .price, .originalPrice, .isPlayerSelection,
          .handicap / .parsedHandicap, .overOrUnder
    """
    receipt_id = raw.get("betReceiptId") or raw.get("betId", "")
    bet_id     = raw.get("betId", "")
    bet_type   = raw.get("betType", "")

    # ── Stake: currentSize ────────────────────────────────────────────────
    stake = _safe_float(raw.get("currentSize")) or 0.0

    # ── Result ────────────────────────────────────────────────────────────
    result = (raw.get("result") or "OPEN").upper()

    # ── Odds (boosted vs original) ────────────────────────────────────────
    bet_prices    = raw.get("betPrices") or {}
    price_obj     = bet_prices.get("betPrice") or {}
    orig_obj      = bet_prices.get("originalBetPrice") or {}
    odds_final    = _safe_float(price_obj.get("decimalPrice"))
    odds_original = _safe_float(orig_obj.get("decimalPrice"))
    # Fallback: top-level betPrice field
    if odds_final is None:
        odds_final = _safe_float(raw.get("betPrice"))

    # ── P&L ───────────────────────────────────────────────────────────────
    # pandl = profit on WON/CASHED_OUT bets; 0 on LOST (FanDuel shows 0 net)
    pandl = _safe_float(raw.get("pandl")) or 0.0
    if result == "WON":
        pnl = round(pandl, 2)          # pandl = actual winnings (net of stake)
    elif result == "LOST":
        pnl = round(-stake, 2)
    elif result == "CASHED_OUT":
        pnl = round(pandl, 2)          # can be negative if cashed out at a loss
    else:
        pnl = 0.0

    # ── Dates ────────────────────────────────────────────────────────────
    placed_date  = _iso(raw.get("placedDate"))
    settled_date = _iso(raw.get("settledDate"))

    # ── Reward / boost ───────────────────────────────────────────────────
    reward      = raw.get("rewardUsed") or {}
    reward_type = reward.get("type")  # PROFIT_BOOST / NO_SWEAT / BONUS_BET
    boost_pct   = None
    features    = raw.get("features") or {}
    price_boost = features.get("priceBoost") or {}
    if price_boost:
        gen = _safe_float(price_boost.get("generosityPercentage"))
        if gen is not None:
            boost_pct = round(gen / 100, 4)
    # Also check rewardUsed.generosity as fallback
    if boost_pct is None and reward_type == "PROFIT_BOOST":
        gen2 = _safe_float(reward.get("generosity"))
        if gen2:
            boost_pct = round(gen2 / 100, 4)

    # ── SGM flag ──────────────────────────────────────────────────────────
    is_sgm = 1 if raw.get("sameGameMultis") else 0

    # ── Legs — each leg has a "parts" list with the actual selection ───────
    raw_legs  = raw.get("legs") or []
    has_props = 0
    leg_rows: list[dict] = []

    for leg in raw_legs:
        leg_result = (leg.get("result") or "PLACED").upper()
        leg_num    = leg.get("legNumber", len(leg_rows) + 1)

        # parts[0] holds the real selection data
        parts = leg.get("parts") or []
        part  = parts[0] if parts else {}

        selection     = part.get("selectionName", "")
        market_type   = part.get("marketType", "")
        competition   = part.get("competitionName", "")
        leg_price     = _safe_float(part.get("price"))
        leg_orig      = _safe_float(part.get("originalPrice"))
        is_player     = 1 if part.get("isPlayerSelection") else 0
        if is_player:
            has_props = 1

        # Handicap: prefer parsedHandicap (clean number), fall back to handicap
        handicap = str(part.get("parsedHandicap") or part.get("handicap") or "") or None

        # Over/Under
        over_under_raw = part.get("overOrUnder")
        over_under = over_under_raw.lower() if over_under_raw else None  # "over"/"under"

        leg_rows.append({
            "bet_receipt_id":      receipt_id,
            "leg_number":          leg_num,
            "result":              leg_result,
            "selection":           selection,
            "market_type":         market_type,
            "competition":         competition,
            "price":               leg_price,
            "original_price":      leg_orig,
            "is_player_selection": is_player,
            "handicap":            handicap,
            "over_under":          over_under,
        })

    n_legs = len(raw_legs) or _legs_from_bet_type(bet_type)

    bet_row = {
        "bet_receipt_id":   receipt_id,
        "bet_id":           bet_id,
        "bet_type":         bet_type,
        "legs":             n_legs,
        "stake":            stake,
        "result":           result,
        "odds_final":       odds_final,
        "odds_original":    odds_original,
        "pnl":              pnl,
        "placed_date":      placed_date,
        "settled_date":     settled_date,
        "reward_type":      reward_type,
        "boost_pct":        boost_pct,
        "is_sgm":           is_sgm,
        "has_player_props": has_props,
        "imported_at":      datetime.now(timezone.utc).isoformat(),
    }
    return bet_row, leg_rows


def _legs_from_bet_type(bt: str) -> int:
    """Infer leg count from bet type code: ACC4 → 4, SGL/DBL/TBL → 1/2/3."""
    bt = bt.upper()
    if bt == "SGL":  return 1
    if bt == "DBL":  return 2
    if bt == "TBL":  return 3
    import re
    m = re.search(r"(\d+)", bt)
    return int(m.group(1)) if m else 1


# ─────────────────────────────────────────────────────────────────────────────
# Upsert
# ─────────────────────────────────────────────────────────────────────────────

def parse_and_upsert(raw_json: dict, conn: sqlite3.Connection) -> dict:
    """
    Parse a single fetch-my-bets API response page and upsert all bets.
    Returns {"inserted": N, "updated": N, "skipped": N}.
    """
    bets_list = (
        raw_json.get("bets")
        or raw_json.get("betList")
        or raw_json.get("data")
        or []
    )
    inserted = updated = skipped = 0

    for raw in bets_list:
        try:
            bet_row, leg_rows = _parse_bet(raw)
        except Exception as e:
            print(f"[FD API] parse error on bet {raw.get('betReceiptId','?')}: {e}")
            skipped += 1
            continue

        if not bet_row["bet_receipt_id"]:
            skipped += 1
            continue

        # ── Check existing ────────────────────────────────────────────────
        exists = conn.execute(
            "SELECT 1 FROM fanduel_bets WHERE bet_receipt_id = ?",
            (bet_row["bet_receipt_id"],),
        ).fetchone()

        if exists:
            conn.execute("""
                UPDATE fanduel_bets SET
                    result = ?, odds_final = ?, odds_original = ?, pnl = ?,
                    settled_date = ?, reward_type = ?, boost_pct = ?,
                    is_sgm = ?, has_player_props = ?, imported_at = ?
                WHERE bet_receipt_id = ?
            """, (
                bet_row["result"], bet_row["odds_final"], bet_row["odds_original"],
                bet_row["pnl"], bet_row["settled_date"], bet_row["reward_type"],
                bet_row["boost_pct"], bet_row["is_sgm"], bet_row["has_player_props"],
                bet_row["imported_at"], bet_row["bet_receipt_id"],
            ))
            updated += 1
        else:
            conn.execute("""
                INSERT INTO fanduel_bets (
                    bet_receipt_id, bet_id, bet_type, legs, stake, result,
                    odds_final, odds_original, pnl, placed_date, settled_date,
                    reward_type, boost_pct, is_sgm, has_player_props, imported_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                bet_row["bet_receipt_id"], bet_row["bet_id"], bet_row["bet_type"],
                bet_row["legs"], bet_row["stake"], bet_row["result"],
                bet_row["odds_final"], bet_row["odds_original"], bet_row["pnl"],
                bet_row["placed_date"], bet_row["settled_date"],
                bet_row["reward_type"], bet_row["boost_pct"],
                bet_row["is_sgm"], bet_row["has_player_props"], bet_row["imported_at"],
            ))
            inserted += 1

        # ── Legs: delete-and-reinsert for clean upsert ────────────────────
        conn.execute(
            "DELETE FROM fanduel_bet_legs WHERE bet_receipt_id = ?",
            (bet_row["bet_receipt_id"],),
        )
        for lr in leg_rows:
            conn.execute("""
                INSERT INTO fanduel_bet_legs (
                    bet_receipt_id, leg_number, result, selection, market_type,
                    competition, price, original_price, is_player_selection,
                    handicap, over_under
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                lr["bet_receipt_id"], lr["leg_number"], lr["result"],
                lr["selection"], lr["market_type"], lr["competition"],
                lr["price"], lr["original_price"], lr["is_player_selection"],
                lr["handicap"], lr["over_under"],
            ))

    conn.commit()
    return {"inserted": inserted, "updated": updated, "skipped": skipped}


# ─────────────────────────────────────────────────────────────────────────────
# Paginated fetch from FanDuel API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_pages(
    token: str,
    since_date: str = "2024-01-04",
    page_size: int = _FD_PAGE,
    region: str = _FD_REGION,
    delay_s: float = 0.4,
) -> list[dict]:
    """
    Fetch all settled bets from FanDuel API, paginating until:
      - moreAvailable == false, OR
      - oldest bet on page is before since_date

    Returns the flat list of all raw bet dicts.
    """
    all_bets: list[dict] = []
    from_record = 1
    page = 0

    headers = {
        "accept":             "application/json",
        "x-app-version":      "2.142.2",
        "x-application":      _FD_APP_KEY,
        "x-authentication":   token,
        "x-sportsbook-region": region,
        "sec-fetch-dest":     "empty",
        "sec-fetch-mode":     "cors",
        "sec-fetch-site":     "same-site",
        "referer":            f"https://{region.lower()}.sportsbook.fanduel.com/",
    }

    while True:
        page += 1
        params = {
            "isSettled":            "true",
            "fromRecord":           from_record,
            "toRecord":             from_record + page_size - 1,
            "sortDir":              "DESC",
            "sortParam":            "SETTLEMENT_DATE",
            "adaptiveTokenEnabled": "true",
            "rewardsClubEnabled":   "false",
            "_ak":                  _FD_APP_KEY,
        }

        try:
            resp = requests.get(_FD_BASE_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[FD API] fetch error page {page}: {e}")
            break

        bets = data.get("bets") or data.get("betList") or data.get("data") or []
        if not bets:
            break

        all_bets.extend(bets)
        print(f"[FD API] page {page}: fetched {len(bets)} bets (total so far: {len(all_bets)})")

        # ── Check if oldest bet on this page pre-dates cutoff ─────────────
        oldest = bets[-1]
        oldest_date = (
            oldest.get("settledDate") or oldest.get("settlementDate")
            or oldest.get("placedDate") or ""
        )
        if oldest_date and oldest_date[:10] < since_date:
            print(f"[FD API] Reached cutoff ({since_date}), stopping.")
            break

        if not data.get("moreAvailable", False):
            print(f"[FD API] moreAvailable=false, done.")
            break

        from_record += page_size
        time.sleep(delay_s)

    return all_bets


# ─────────────────────────────────────────────────────────────────────────────
# File-based import
# ─────────────────────────────────────────────────────────────────────────────

def import_from_file(path: str, conn: Optional[sqlite3.Connection] = None) -> dict:
    """
    Load a JSON file (single API response or list of responses), parse and upsert.
    If conn is None, opens bets.db automatically.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    close = conn is None
    if conn is None:
        conn = sqlite3.connect(_BETS_DB)

    create_tables(conn)

    try:
        # If the file is a list of API responses (multiple pages), merge them
        if isinstance(raw, list):
            total = {"inserted": 0, "updated": 0, "skipped": 0}
            for page_data in raw:
                r = parse_and_upsert(page_data, conn)
                for k in total:
                    total[k] += r[k]
            return total
        else:
            return parse_and_upsert(raw, conn)
    finally:
        if close:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis queries
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(conn: Optional[sqlite3.Connection] = None) -> dict:
    """Run all five analysis queries; return structured results."""
    close = conn is None
    if conn is None:
        conn = sqlite3.connect(_BETS_DB)
        conn.row_factory = sqlite3.Row

    try:
        def rows(sql, params=()):
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

        boost_perf = rows("""
            SELECT
                COALESCE(reward_type, 'NONE')        AS reward_type,
                COALESCE(CAST(ROUND(boost_pct*100) AS INTEGER), 0) AS boost_pct_int,
                COUNT(*)                              AS bets,
                ROUND(100.0 * SUM(CASE WHEN result='WON' THEN 1 ELSE 0 END) / COUNT(*), 1) AS wr,
                ROUND(SUM(pnl), 2)                    AS total_pnl
            FROM fanduel_bets
            WHERE result IN ('WON','LOST','CASHED_OUT')
            GROUP BY reward_type, boost_pct_int
            ORDER BY total_pnl DESC
        """)

        sgm_split = rows("""
            SELECT
                is_sgm,
                COUNT(*)                              AS bets,
                ROUND(100.0 * SUM(CASE WHEN result='WON' THEN 1 ELSE 0 END) / COUNT(*), 1) AS wr,
                ROUND(SUM(pnl), 2)                    AS total_pnl
            FROM fanduel_bets
            WHERE result IN ('WON','LOST','CASHED_OUT')
            GROUP BY is_sgm
        """)

        props_split = rows("""
            SELECT
                has_player_props,
                COUNT(*)                              AS bets,
                ROUND(100.0 * SUM(CASE WHEN result='WON' THEN 1 ELSE 0 END) / COUNT(*), 1) AS wr,
                ROUND(SUM(pnl), 2)                    AS total_pnl
            FROM fanduel_bets
            WHERE result IN ('WON','LOST','CASHED_OUT')
            GROUP BY has_player_props
        """)

        prop_markets = rows("""
            SELECT
                fl.market_type,
                COUNT(*)                              AS appearances,
                ROUND(100.0 * SUM(CASE WHEN fl.result='WON' THEN 1 ELSE 0 END) / COUNT(*), 1) AS leg_wr
            FROM fanduel_bet_legs fl
            WHERE fl.is_player_selection = 1
            GROUP BY fl.market_type
            HAVING COUNT(*) >= 3
            ORDER BY appearances DESC
        """)

        by_legs_boost = rows("""
            SELECT
                legs,
                COALESCE(reward_type, 'NONE')         AS reward_type,
                COUNT(*)                              AS bets,
                ROUND(100.0 * SUM(CASE WHEN result='WON' THEN 1 ELSE 0 END) / COUNT(*), 1) AS wr,
                ROUND(SUM(pnl), 2)                    AS total_pnl
            FROM fanduel_bets
            WHERE result IN ('WON','LOST','CASHED_OUT')
            GROUP BY legs, reward_type
            ORDER BY legs, reward_type
        """)

        # Summary counts
        summary = rows("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN result='WON'        THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN result='LOST'       THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN result='CASHED_OUT' THEN 1 ELSE 0 END) AS cashed_out,
                SUM(CASE WHEN result='OPEN'       THEN 1 ELSE 0 END) AS open_bets,
                ROUND(SUM(stake), 2)  AS total_staked,
                ROUND(SUM(pnl), 2)    AS total_pnl,
                MIN(placed_date)      AS earliest_bet,
                MAX(placed_date)      AS latest_bet
            FROM fanduel_bets
        """)[0]

        return {
            "summary":       summary,
            "boost_perf":    boost_perf,
            "sgm_split":     sgm_split,
            "props_split":   props_split,
            "prop_markets":  prop_markets,
            "by_legs_boost": by_legs_boost,
        }

    finally:
        if close:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Full import: token → fetch all pages → upsert → return analysis
# ─────────────────────────────────────────────────────────────────────────────

def import_from_token(
    token: str,
    since_date: str = "2024-01-04",
    region: str = _FD_REGION,
) -> dict:
    """
    High-level entry point used by the API endpoint.
    Fetches all pages, upserts, returns {upsert_stats, analysis}.
    """
    conn = sqlite3.connect(_BETS_DB)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    all_bets = fetch_all_pages(token, since_date=since_date, region=region)

    if not all_bets:
        conn.close()
        return {"error": "No bets returned — token may be expired", "fetched": 0}

    # Wrap in the standard API response envelope so parse_and_upsert works
    upsert_stats = parse_and_upsert({"bets": all_bets}, conn)
    analysis     = run_analysis(conn)
    conn.close()

    return {
        "fetched":     len(all_bets),
        "upsert":      upsert_stats,
        "analysis":    analysis,
    }
