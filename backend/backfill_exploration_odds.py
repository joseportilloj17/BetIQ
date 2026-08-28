"""
Backfill real market odds onto exploration/simulation legs that have NULL open_odds.

Links each leg back to its source odds record via fixture_id + market_type + team/line,
converts decimal → American, and writes open_odds + close_odds + clv_cents.

This removes the circular-odds problem: exploration legs previously derived dec_odds
from predicted_win_prob (1/wp), which means odds/implied_prob/ev_value features encoded
the model's own prediction. Real market odds break that circularity.

Run once:  python backfill_exploration_odds.py
"""
import json
import re
import sqlite3
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
BETS_DB = os.path.join(_DIR, "..", "data", "bets.db")
HIST_DB = os.path.join(_DIR, "..", "data", "historical.db")

# ── Decimal → American conversion ────────────────────────────────────────────

def _dec_to_american(dec: float) -> int:
    """Convert decimal odds (e.g. 2.50) to American odds (+150 / -200)."""
    if dec >= 2.0:
        return round((dec - 1) * 100)
    elif dec > 1.0:
        return round(-100 / (dec - 1))
    return 0


# ── Description parsers ──────────────────────────────────────────────────────

# "Tampa Bay Rays +1.5 (Alt Spread)" → team="Tampa Bay Rays", line=1.5
# "Anaheim Ducks +1.5 (spreads)"     → team="Anaheim Ducks", line=1.5
_SPREAD_DESC_RE = re.compile(r'^(.+?)\s+([+-]?\d+(?:\.\d+)?)\s+\(')

# "Over +5.5 (totals)" / "Under +12.5 (alternate_totals)" → side="Over"/"Under", line=5.5/12.5
_TOTAL_DESC_RE = re.compile(r'^(Over|Under)\s+\+?(\d+(?:\.\d+)?)\s+\(', re.IGNORECASE)

# "San Antonio Spurs (h2h)" → team="San Antonio Spurs"
_H2H_DESC_RE = re.compile(r'^(.+?)\s+\(h2h\)$')

# "Ronald Acuna Jr. Over 0.5 Batter Hits +0.5 (player_hits)"
# → player="Ronald Acuna Jr.", over_under="Over", point=0.5
_PROP_DESC_RE = re.compile(
    r'^(.+?)\s+(Over|Under)\s+(\d+(?:\.\d+)?)\s+.+?\s+[+-]?\d+(?:\.\d+)?\s+\(',
    re.IGNORECASE,
)

# Market type → player_props market_key mapping
_PROP_MARKET_MAP = {
    "player_hits":          ["batter_hits", "batter_hits_alternate"],
    "player_total_bases":   ["batter_total_bases", "batter_total_bases_alternate"],
    "player_points":        ["player_points", "player_points_alternate"],
    "player_rebounds":      ["player_rebounds", "player_rebounds_alternate"],
    "player_assists":       ["player_assists", "player_assists_alternate"],
    "player_shots_on_goal": ["player_shots_on_goal", "player_shots_on_goal_alternate"],
}

# Preferred bookmaker order (same as _get_alt_lines_for_tier)
_PREFERRED_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pinnacle"]


def _best_bookmaker_odds(bookmakers_json: list, market_key: str, team: str, line: float | None):
    """Extract best odds from fixtures.bookmakers JSON for a given market+team+line."""
    for preferred in _PREFERRED_BOOKS:
        bk = next((b for b in bookmakers_json if b.get("key") == preferred), None)
        if not bk:
            continue
        mkt_data = next((m for m in bk.get("markets", []) if m.get("key") == market_key), None)
        if not mkt_data:
            continue
        for oc in mkt_data.get("outcomes", []):
            oc_name = (oc.get("name") or "").strip()
            oc_point = oc.get("point")
            if oc_name.lower() == team.lower():
                if market_key in ("h2h",):
                    return float(oc.get("price", 0))
                # spreads / totals — also match line
                if line is not None and oc_point is not None:
                    if abs(float(oc_point) - line) < 0.01:
                        return float(oc.get("price", 0))
                elif line is None:
                    return float(oc.get("price", 0))
    # Fallback: any bookmaker
    for bk in bookmakers_json:
        mkt_data = next((m for m in bk.get("markets", []) if m.get("key") == market_key), None)
        if not mkt_data:
            continue
        for oc in mkt_data.get("outcomes", []):
            oc_name = (oc.get("name") or "").strip()
            oc_point = oc.get("point")
            if oc_name.lower() == team.lower():
                if market_key == "h2h":
                    return float(oc.get("price", 0))
                if line is not None and oc_point is not None:
                    if abs(float(oc_point) - line) < 0.01:
                        return float(oc.get("price", 0))
    return None


def run_backfill():
    bets_con = sqlite3.connect(BETS_DB, timeout=30)
    bets_con.row_factory = sqlite3.Row
    hist_con = sqlite3.connect(HIST_DB, timeout=30)
    hist_con.row_factory = sqlite3.Row

    # ── Load all exploration/simulation legs with NULL open_odds ──────────
    legs = bets_con.execute("""
        SELECT mbl.id, mbl.fixture_id, mbl.market_type, mbl.description,
               mbl.win_prob, mbl.sport, mb.source
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.open_odds IS NULL
          AND mbl.fixture_id IS NOT NULL
          AND mb.source IN ('exploration', 'simulation')
    """).fetchall()

    print(f"[backfill] {len(legs)} legs to process")

    # ── Pre-load fixtures bookmakers JSON (for h2h/spreads/totals path) ──
    fixture_bm_cache: dict[str, list] = {}

    # ── Counters ─────────────────────────────────────────────────────────
    matched = 0
    matched_close = 0
    unmatched = 0
    by_market: dict[str, dict[str, int]] = {}

    updates: list[tuple] = []  # (open_odds, close_odds, clv_cents, leg_id)

    for leg in legs:
        leg_id = leg["id"]
        fid = leg["fixture_id"]
        mkt = leg["market_type"]
        desc = leg["description"] or ""

        mkt_stats = by_market.setdefault(mkt, {"matched": 0, "unmatched": 0})

        open_dec = None
        close_dec = None

        # ── Path A: alternate_spreads / alternate_totals → alt_lines ─────
        if mkt in ("alternate_spreads", "alternate_totals"):
            if mkt == "alternate_totals":
                m = _TOTAL_DESC_RE.match(desc)
                if not m:
                    mkt_stats["unmatched"] += 1
                    unmatched += 1
                    continue
                team = m.group(1)  # "Over" or "Under"
                line = float(m.group(2))
            else:
                m = _SPREAD_DESC_RE.match(desc)
                if not m:
                    mkt_stats["unmatched"] += 1
                    unmatched += 1
                    continue
                team = m.group(1).strip()
                line = float(m.group(2))

            # Query alt_lines: earliest snapshot = open, latest = close
            rows = hist_con.execute("""
                SELECT odds, fetched_at FROM alt_lines
                WHERE event_id = ? AND market_key = ? AND team = ? AND ABS(line - ?) < 0.01
                ORDER BY fetched_at ASC
            """, (fid, mkt, team, line)).fetchall()

            if rows:
                open_dec = float(rows[0]["odds"])
                close_dec = float(rows[-1]["odds"]) if len(rows) > 1 else open_dec

        # ── Path B: h2h / spreads / totals → fixtures.bookmakers JSON ────
        elif mkt in ("h2h", "spreads", "totals"):
            if mkt == "h2h":
                m = _H2H_DESC_RE.match(desc)
                team = m.group(1).strip() if m else desc.replace(" (h2h)", "").strip()
                line = None
            elif mkt == "totals":
                m = _TOTAL_DESC_RE.match(desc)
                if not m:
                    mkt_stats["unmatched"] += 1
                    unmatched += 1
                    continue
                team = m.group(1)  # "Over" or "Under"
                line = float(m.group(2))
            else:  # spreads
                m = _SPREAD_DESC_RE.match(desc)
                if not m:
                    mkt_stats["unmatched"] += 1
                    unmatched += 1
                    continue
                team = m.group(1).strip()
                line = float(m.group(2))

            # Load bookmakers JSON from fixtures table
            if fid not in fixture_bm_cache:
                bm_row = bets_con.execute(
                    "SELECT bookmakers FROM fixtures WHERE id = ?", (fid,)
                ).fetchone()
                if bm_row and bm_row["bookmakers"]:
                    try:
                        fixture_bm_cache[fid] = json.loads(bm_row["bookmakers"])
                    except Exception:
                        fixture_bm_cache[fid] = []
                else:
                    fixture_bm_cache[fid] = []

            bm_json = fixture_bm_cache[fid]
            if bm_json:
                dec = _best_bookmaker_odds(bm_json, mkt, team, line)
                if dec and dec > 1.0:
                    open_dec = dec
                    close_dec = dec  # fixtures JSON is a single snapshot

        # ── Path C: player_* props → player_props table ──────────────────
        elif mkt.startswith("player_"):
            m = _PROP_DESC_RE.match(desc)
            if not m:
                mkt_stats["unmatched"] += 1
                unmatched += 1
                continue
            player = m.group(1).strip()
            over_under = m.group(2).lower()  # "over" / "under"
            point = float(m.group(3))

            prop_keys = _PROP_MARKET_MAP.get(mkt, [mkt])
            key_ph = ",".join("?" * len(prop_keys))

            rows = hist_con.execute(f"""
                SELECT price, fetched_at FROM player_props
                WHERE event_id = ?
                  AND market_key IN ({key_ph})
                  AND player_name = ?
                  AND ABS(point - ?) < 0.01
                  AND over_under = ?
                ORDER BY fetched_at ASC
            """, [fid] + prop_keys + [player, point, over_under]).fetchall()

            if rows:
                open_dec = float(rows[0]["price"])
                close_dec = float(rows[-1]["price"]) if len(rows) > 1 else open_dec

        else:
            mkt_stats["unmatched"] += 1
            unmatched += 1
            continue

        # ── Convert and queue update ─────────────────────────────────────
        if open_dec and open_dec > 1.0:
            open_american = _dec_to_american(open_dec)
            close_american = _dec_to_american(close_dec) if close_dec and close_dec > 1.0 else open_american

            # CLV = close_implied_prob - open_implied_prob (positive = line moved in our favor)
            open_ip = 1.0 / open_dec
            close_ip = 1.0 / close_dec if close_dec and close_dec > 1.0 else open_ip
            clv_cents = round((open_ip - close_ip) * 100, 2)  # in percentage points

            updates.append((open_american, close_american, clv_cents, leg_id))
            matched += 1
            mkt_stats["matched"] += 1
            if close_dec != open_dec:
                matched_close += 1
        else:
            mkt_stats["unmatched"] += 1
            unmatched += 1

    # ── Apply updates ────────────────────────────────────────────────────
    if updates:
        bets_con.executemany("""
            UPDATE mock_bet_legs
            SET open_odds = ?, close_odds = ?, clv_cents = ?
            WHERE id = ?
        """, updates)
        bets_con.commit()

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n[backfill] Results:")
    print(f"  Matched:   {matched}/{len(legs)} ({100*matched/max(len(legs),1):.1f}%)")
    print(f"  With distinct close: {matched_close}")
    print(f"  Unmatched: {unmatched}")
    print(f"\n  By market:")
    for mkt, stats in sorted(by_market.items(), key=lambda x: -x[1]["matched"]):
        total = stats["matched"] + stats["unmatched"]
        print(f"    {mkt:25s}  {stats['matched']:4d}/{total:4d} matched")

    # ── Verify ───────────────────────────────────────────────────────────
    remaining = bets_con.execute("""
        SELECT COUNT(*) FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.open_odds IS NULL AND mbl.fixture_id IS NOT NULL
          AND mb.source IN ('exploration', 'simulation')
    """).fetchone()[0]
    print(f"\n  Remaining NULL open_odds: {remaining}")

    bets_con.close()
    hist_con.close()

    return {"matched": matched, "unmatched": unmatched, "remaining": remaining}


if __name__ == "__main__":
    run_backfill()
