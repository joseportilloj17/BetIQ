"""
fanduel_parser.py — FanDuel bet slip parser for A/B test ingestion.

Two input modes:
  • Image (PNG/JPG screenshot) — sent to Claude Haiku vision
  • Text (pasted bet slip text)  — sent to Claude Haiku text-only

Returns structured ParsedPick dicts ready for user review and submission.
No OddsAPI calls are made — alt_lines lookup uses existing historical.db data.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sqlite3
from datetime import datetime
from typing import Optional

import anthropic

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_HIST_DB       = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
)
_HAIKU_MODEL   = "claude-haiku-4-5-20251001"

# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are parsing a FanDuel sports betting bet slip.
Extract ALL structured data and return ONLY valid JSON — no explanation, no markdown.

Identify:

1. BET STATUS (overall pick):
   PENDING       — all legs unsettled, games not started
   PARTIAL       — some legs settled, some still live (game in progress)
   SETTLED_WIN   — all legs hit, payout finalized
   SETTLED_LOSS  — at least one leg lost (parlay busted)
   CASHED_OUT    — user took early cash-out before final settlement

2. PARLAY STRUCTURE:
   n_legs                  — total number of legs
   combined_odds_american  — combined parlay odds as shown (e.g. +145 → 145, -110 → -110)
   boost                   — null OR {type, pct, boosted_odds_american}
                             type: "PROFIT_BOOST" | "BONUS_BET" | "NO_SWEAT"
                             pct: 0.25 / 0.30 / 0.50 (as decimal fraction)
                             boosted_odds_american: final odds after boost
   stake                   — dollar amount wagered (float, e.g. 10.00)
   potential_payout        — total payout shown (stake + profit) (float or null)
   cashout_value           — cash-out amount if shown (float or null)
   is_sgp                  — true if entire bet is a Same Game Parlay
   has_sgp_subset          — true if some legs are an SGP sub-group
   sgp_combined_odds       — combined odds of SGP subset if shown (integer or null)
   fanduel_bet_id          — alphanumeric bet ID if visible (string or null)
   bet_placed_at           — ISO timestamp when bet was placed (string or null, from bet history view)

3. PER LEG (extract every leg):
   sport          — infer from teams: NBA, MLB, NHL, NFL, NCAAB, NCAAF, MMA, soccer
   market_type    — h2h | spread | alt_spreads | totals | alt_totals |
                    player_points | player_assists | player_rebounds | player_hits |
                    player_strikeouts | player_home_runs | player_props
   team           — full team name (null for player props)
   player         — full player name (null for team bets)
   point          — numeric line value if applicable (+1.5 → 1.5, Over 209.5 → 209.5, null for moneyline)
   over_under     — "over" | "under" | null
   bet_info       — exact text shown on the slip for this leg (e.g. "Oklahoma City Thunder ML")
   home_team      — home team of the game this leg belongs to
   away_team      — away team of the game this leg belongs to
   extracted_price — American odds for this leg IF clearly visible as a standalone number
                     (null for SGP-internal legs where individual odds aren't shown)
   leg_status     — WON | LOST | PUSH | VOID | PENDING
   is_part_of_sgp — true if this leg is inside an SGP sub-group

Rules:
- For American odds: positive numbers have no +/- ambiguity → 145 means +145
  Negative numbers: -880 means -880. Include the sign in the integer.
- combined_odds_american: if +145 is shown → 145. If -110 → -110.
- stake: parse "$10.00" → 10.0
- For SGP-internal legs where individual prices aren't shown → extracted_price = null
- If you cannot determine a field, use null

Return this exact JSON structure:
{
  "n_legs": 5,
  "bet_status": "PARTIAL",
  "combined_odds_american": 145,
  "boost": null,
  "stake": 10.00,
  "potential_payout": 24.50,
  "cashout_value": null,
  "fanduel_bet_id": null,
  "bet_placed_at": null,
  "is_sgp": false,
  "has_sgp_subset": true,
  "sgp_combined_odds": 126,
  "legs": [
    {
      "sport": "NBA",
      "market_type": "h2h",
      "team": "Oklahoma City Thunder",
      "player": null,
      "point": null,
      "over_under": null,
      "bet_info": "Oklahoma City Thunder ML",
      "home_team": "Los Angeles Lakers",
      "away_team": "Oklahoma City Thunder",
      "extracted_price": null,
      "leg_status": "PENDING",
      "is_part_of_sgp": true
    }
  ]
}"""


# ── Claude API helpers ────────────────────────────────────────────────────────

def _call_claude_vision(image_bytes: bytes, mime_type: str) -> dict:
    """Send image to Claude Haiku vision and return parsed JSON dict."""
    if not _ANTHROPIC_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client  = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    b64data = base64.standard_b64encode(image_bytes).decode("utf-8")

    resp = client.messages.create(
        model      = _HAIKU_MODEL,
        max_tokens = 3000,
        messages   = [{
            "role":    "user",
            "content": [
                {
                    "type":   "image",
                    "source": {"type": "base64", "media_type": mime_type, "data": b64data},
                },
                {"type": "text", "text": _EXTRACTION_PROMPT},
            ],
        }],
    )
    raw = resp.content[0].text.strip()
    return _parse_claude_json(raw)


def _call_claude_text(slip_text: str) -> dict:
    """Send pasted text to Claude Haiku and return parsed JSON dict."""
    if not _ANTHROPIC_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    prompt = (
        _EXTRACTION_PROMPT
        + f"\n\nHere is the bet slip text to parse:\n\n{slip_text}"
    )

    resp = client.messages.create(
        model      = _HAIKU_MODEL,
        max_tokens = 3000,
        messages   = [{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    return _parse_claude_json(raw)


def _parse_claude_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


# ── Odds conversion helpers ───────────────────────────────────────────────────

def _american_to_decimal(american: int | float) -> float:
    """Convert American odds integer to decimal odds."""
    a = float(american)
    if a > 0:
        return round(1.0 + a / 100.0, 6)
    else:
        return round(1.0 + 100.0 / abs(a), 6)


def _decimal_to_american(dec: float) -> int:
    """Convert decimal odds to American odds integer."""
    if dec >= 2.0:
        return round((dec - 1.0) * 100)
    else:
        return round(-100.0 / (dec - 1.0))


def _combined_decimal(american_list: list[int]) -> Optional[float]:
    """Compute combined decimal odds from list of American odds."""
    if not american_list:
        return None
    combined = 1.0
    for a in american_list:
        combined *= _american_to_decimal(a)
    return round(combined, 6)


# ── Alt_lines price lookup ────────────────────────────────────────────────────

def _hist_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_HIST_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _fuzzy_team_match(name_a: str, name_b: str) -> bool:
    """True if two team names share a meaningful word (ignoring common words)."""
    stop = {"at", "the", "vs", "fc", "sc", "united", "city", "of"}
    tokens_a = {t.lower() for t in re.split(r"\s+", name_a) if len(t) > 2 and t.lower() not in stop}
    tokens_b = {t.lower() for t in re.split(r"\s+", name_b) if len(t) > 2 and t.lower() not in stop}
    return bool(tokens_a & tokens_b)


def resolve_leg_price(parsed_leg: dict, game_date: Optional[str] = None) -> dict:
    """
    Resolve American odds for a parsed leg.

    If extracted_price is visible in screenshot → use it directly (confidence 1.0).
    Otherwise → query alt_lines table (alternate_spreads / alternate_totals).
    For h2h / player props → unresolved (requires manual entry).

    Returns dict with:
      matched_price       int | None
      matched_event_id    str | None
      price_source        'screenshot_visible' | 'alt_lines_lookup' | 'unresolved'
      price_confidence    0.0 – 1.0
      requires_manual_entry bool
    """
    # ── If price is visible in screenshot, use it directly ────────────────────
    if parsed_leg.get("extracted_price") is not None:
        return {
            "matched_price":        int(parsed_leg["extracted_price"]),
            "matched_event_id":     None,
            "price_source":         "screenshot_visible",
            "price_confidence":     1.0,
            "requires_manual_entry": False,
        }

    market = (parsed_leg.get("market_type") or "").lower()
    home   = parsed_leg.get("home_team") or ""
    away   = parsed_leg.get("away_team") or ""
    team   = parsed_leg.get("team") or ""
    point  = parsed_leg.get("point")
    ou     = (parsed_leg.get("over_under") or "").lower()

    # ── h2h and player props: alt_lines has no data ────────────────────────────
    if market in ("h2h", "moneyline") or market.startswith("player_"):
        return {
            "matched_price":        None,
            "matched_event_id":     None,
            "price_source":         "unresolved",
            "price_confidence":     0.0,
            "requires_manual_entry": True,
        }

    # ── Spread / totals lookup ─────────────────────────────────────────────────
    try:
        conn = _hist_conn()

        # Find event_id by team names
        today_str = game_date or datetime.utcnow().strftime("%Y-%m-%d")
        rows = conn.execute(
            """SELECT DISTINCT event_id, home_team, away_team
               FROM alt_lines
               WHERE game_date = ?
               ORDER BY fetched_at DESC""",
            (today_str,),
        ).fetchall()

        event_id = None
        for row in rows:
            if (_fuzzy_team_match(row["home_team"], home) or
                    _fuzzy_team_match(row["home_team"], away) or
                    _fuzzy_team_match(row["away_team"], home) or
                    _fuzzy_team_match(row["away_team"], away)):
                event_id = row["event_id"]
                break

        if not event_id:
            conn.close()
            return {
                "matched_price":        None,
                "matched_event_id":     None,
                "price_source":         "unresolved",
                "price_confidence":     0.0,
                "requires_manual_entry": True,
            }

        # Map market_type → alt_lines market_key
        mkt_key_map = {
            "spread":       "alternate_spreads",
            "alt_spreads":  "alternate_spreads",
            "totals":       "alternate_totals",
            "alt_totals":   "alternate_totals",
        }
        mkt_key = mkt_key_map.get(market)
        if not mkt_key:
            conn.close()
            return {
                "matched_price":        None,
                "matched_event_id":     event_id,
                "price_source":         "unresolved",
                "price_confidence":     0.0,
                "requires_manual_entry": True,
            }

        # Build query
        params: list = [event_id, mkt_key]
        sql = "SELECT * FROM alt_lines WHERE event_id=? AND market_key=?"

        if point is not None:
            # Match within ±0.5 of requested line
            sql += " AND ABS(line - ?) < 0.6"
            params.append(float(point))

        if ou in ("over", "under"):
            sql += " AND over_under = ?"
            params.append(ou)
        elif team:
            # Spread leg — match by team name
            sql += " AND team LIKE ?"
            team_tok = team.split()[-1]  # last word (city stripped)
            params.append(f"%{team_tok}%")

        sql += " ORDER BY fetched_at DESC LIMIT 1"
        match = conn.execute(sql, params).fetchone()
        conn.close()

        if match:
            dec = float(match["odds"])
            am  = _decimal_to_american(dec)
            return {
                "matched_price":        am,
                "matched_event_id":     match["event_id"],
                "price_source":         "alt_lines_lookup",
                "price_confidence":     0.92,
                "requires_manual_entry": False,
            }

        return {
            "matched_price":        None,
            "matched_event_id":     event_id,
            "price_source":         "unresolved",
            "price_confidence":     0.0,
            "requires_manual_entry": True,
        }

    except Exception as exc:
        print(f"[fanduel_parser] alt_lines lookup error: {exc}")
        return {
            "matched_price":        None,
            "matched_event_id":     None,
            "price_source":         "unresolved",
            "price_confidence":     0.0,
            "requires_manual_entry": True,
        }


# ── Enrich raw Claude output ──────────────────────────────────────────────────

def enrich_parsed_pick(raw: dict, game_date: Optional[str] = None) -> dict:
    """
    Take raw Claude-extracted dict and:
      1. Resolve each leg's price via alt_lines lookup
      2. Compute combined_odds_decimal from resolved prices (if needed)
      3. Collect warnings (unresolved legs, etc.)
    """
    legs    = raw.get("legs", [])
    warnings: list[str] = []

    enriched_legs = []
    resolved_prices = []

    for leg in legs:
        resolution = resolve_leg_price(leg, game_date=game_date)
        enriched_leg = {
            "sport":         leg.get("sport"),
            "market_type":   leg.get("market_type"),
            "team":          leg.get("team"),
            "player":        leg.get("player"),
            "point":         leg.get("point"),
            "over_under":    leg.get("over_under"),
            "bet_info":      leg.get("bet_info"),
            "home_team":     leg.get("home_team"),
            "away_team":     leg.get("away_team"),
            "leg_status":    leg.get("leg_status", "PENDING"),
            "is_part_of_sgp": bool(leg.get("is_part_of_sgp", False)),
            # Extracted price (visible in screenshot)
            "extracted_price": leg.get("extracted_price"),
            # Resolved price (from screenshot or alt_lines)
            "matched_price":       resolution["matched_price"],
            "matched_event_id":    resolution["matched_event_id"],
            "price_source":        resolution["price_source"],
            "price_confidence":    resolution["price_confidence"],
            "requires_manual_entry": resolution["requires_manual_entry"],
        }
        enriched_legs.append(enriched_leg)

        if resolution["matched_price"] is not None:
            resolved_prices.append(resolution["matched_price"])
        elif resolution["requires_manual_entry"]:
            bet_desc = leg.get("bet_info") or leg.get("team") or "Unknown leg"
            if leg.get("is_part_of_sgp"):
                warnings.append(
                    f"SGP leg '{bet_desc}' price not in alt_lines — will be listed as unresolved."
                )
            else:
                warnings.append(
                    f"Leg '{bet_desc}' price unresolved — enter manually."
                )

    # Combined decimal odds
    combined_am  = raw.get("combined_odds_american")
    if combined_am is not None:
        combined_dec = _american_to_decimal(int(combined_am))
    elif resolved_prices:
        combined_dec = _combined_decimal(resolved_prices)
        if combined_dec:
            combined_am = _decimal_to_american(combined_dec)
    else:
        combined_dec = None

    # Boost
    boost     = raw.get("boost")  # {type, pct, boosted_odds_american} or null
    stake     = raw.get("stake", 10.0) or 10.0
    pot_payout = raw.get("potential_payout")

    # Compute potential_profit from boosted or base odds
    if boost and boost.get("boosted_odds_american") is not None:
        b_dec = _american_to_decimal(int(boost["boosted_odds_american"]))
        potential_profit = round((b_dec - 1) * stake, 2)
    elif combined_dec:
        potential_profit = round((combined_dec - 1) * stake, 2)
    else:
        potential_profit = None

    # Derive bet_status-level warnings
    bet_status = raw.get("bet_status", "PENDING")
    cashout_val = raw.get("cashout_value")

    return {
        "n_legs":                 len(enriched_legs),
        "combined_odds_american": combined_am,
        "combined_odds_decimal":  round(combined_dec, 4) if combined_dec else None,
        "boost":                  boost,
        "is_sgp":                 bool(raw.get("is_sgp", False)),
        "has_sgp_subset":         bool(raw.get("has_sgp_subset", False)),
        "sgp_combined_odds":      raw.get("sgp_combined_odds"),
        "fanduel_bet_id":         raw.get("fanduel_bet_id"),
        "bet_placed_at":          raw.get("bet_placed_at"),
        "bet_status":             bet_status,
        "stake":                  stake,
        "potential_payout":       pot_payout,
        "potential_profit":       potential_profit,
        "cashout_value":          cashout_val,
        "parsed_legs":            enriched_legs,
        "warnings":               warnings,
    }


# ── Free OCR parser (no API key required) ─────────────────────────────────────

# UI chrome that Tesseract picks up from FanDuel app — never a real bet leg
_UI_NOISE_FRAGMENTS = [
    "game lines", "home runs", "hits", "total bases", "total runs",
    "all sports", "my bets", "live now", "account", "home",
    "alternate run lines", "alternate spreads", "alternate totals",
    "cash out", "cashout", "fanduel", "sportsbook", "not secure",
    "bet slip", "open bets", "settled bets", "same game parlay",
    "share", "sgp+", "same game", "parlay insurance",
    "more wagers", "more markets", "see all", "view all",
    "add to betslip", "place bet", "bet now",
    "betslip", "bet slip", "slip",
    "spreads", "moneylines", "totals", "game props",
    "player props", "featured", "popular",
    "http", "www.", ".com", "secure",
]

def _is_ui_noise(line: str) -> bool:
    """Return True if the line is FanDuel app chrome, not bet content."""
    lower = line.lower().strip()
    if not lower:
        return True
    # Exact match against known single-word noise
    if lower in {"home", "account", "total", "totals", "spreads", "live"}:
        return True
    # Contains a noise fragment
    return any(frag in lower for frag in _UI_NOISE_FRAGMENTS)

def _is_pitcher_line(line: str) -> bool:
    """'P: Clayton Kershaw' — pitcher attribution, not a leg."""
    return bool(re.match(r"^P:\s", line.strip()))

def _is_time_line(line: str) -> bool:
    return bool(re.match(r"^\d{1,2}:\d{2}\s*[AaPp][Mm]", line.strip()))

def _is_dollar_amount(line: str) -> bool:
    return bool(re.fullmatch(r"\$[\d,\.]+", line.strip()))


# ── Browsing-page detector ─────────────────────────────────────────────────────

_RE_GAME_HEADER = re.compile(
    r"^(.+?)\s+@\s+(.+?)(?:\s*>)?\s*$"
)

def _detect_page_type(lines: list[str]) -> str:
    """
    Return 'browsing_page' or 'bet_slip' based on OCR content.

    Browsing pages have @ game headers and no stake/payout dollar amounts.
    Bet slips have stake/payout amounts and per-leg status labels.
    """
    has_game_header  = False
    has_stake        = False
    has_leg_status   = False

    for line in lines:
        stripped = line.strip()
        if _RE_GAME_HEADER.search(stripped):
            has_game_header = True
        lower = stripped.lower()
        if re.search(r"\$\d", stripped):
            has_stake = True
        if any(k in lower for k in ("won", "lost", "settled", "pending", "wager", "stake")):
            has_leg_status = True

    # Browsing page: has game headers, no stake amounts, no leg statuses
    if has_game_header and not has_stake and not has_leg_status:
        return "browsing_page"
    return "bet_slip"


# ── Browsing-page game parser ──────────────────────────────────────────────────

_RE_ODDS_TOKEN    = re.compile(r"([+\-]\d{3,4})(?!\d)")
_RE_SPREAD_VAL    = re.compile(r"([+\-]\d{1,2}(?:\.\d)?)\s+([+\-]\d{3,4})")
_RE_TOTAL_LINE    = re.compile(r"[OoUu]\s*(\d+\.?\d?)\s+([+\-]\d{3,4})")
_RE_TIME_LINE     = re.compile(r"(\d{1,2}:\d{2}\s*[AaPp][Mm](?:\s*[A-Z]{2,3})?)")


def _parse_team_row(line: str, team_name: str, home: str, away: str) -> list[dict]:
    """
    Parse one team row from a game lines browsing page.
    FanDuel layout:  TeamName  spread+odds  ML_odds  O/U+odds
    Returns list of line dicts.
    """
    result: list[dict] = []

    # All American-odds tokens on the line
    all_odds = _RE_ODDS_TOKEN.findall(line)

    # Spread: first occurrence of "±N.N  ±NNN"
    spread_m = _RE_SPREAD_VAL.search(line)
    spread_odds_used: set[str] = set()
    if spread_m:
        sv   = spread_m.group(1)
        so   = spread_m.group(2)
        spread_odds_used.add(so)
        result.append({
            "team":        team_name,
            "type":        "spread",
            "description": f"{team_name} {sv}",
            "odds":        int(so),
            "market_type": "spread",
            "point":       float(sv),
            "over_under":  None,
            "home_team":   home,
            "away_team":   away,
        })

    # Total: O/U N.N  ±NNN
    total_m = _RE_TOTAL_LINE.search(line)
    total_odds_used: set[str] = set()
    if total_m:
        tv  = total_m.group(1)
        to  = total_m.group(2)
        total_odds_used.add(to)
        ou  = "over" if re.search(r"^[Oo]", line.lstrip()) or "O " in line else "under"
        result.append({
            "team":        team_name,
            "type":        "total",
            "description": f"{'Over' if ou == 'over' else 'Under'} {tv}",
            "odds":        int(to),
            "market_type": "totals",
            "point":       float(tv),
            "over_under":  ou,
            "home_team":   home,
            "away_team":   away,
        })

    # Moneyline: remaining odds not captured by spread or total
    used = spread_odds_used | total_odds_used
    ml_candidates = [o for o in all_odds if o not in used]
    if ml_candidates:
        # Take the first remaining odds as ML
        result.append({
            "team":        team_name,
            "type":        "h2h",
            "description": f"{team_name} ML",
            "odds":        int(ml_candidates[0]),
            "market_type": "h2h",
            "point":       None,
            "over_under":  None,
            "home_team":   home,
            "away_team":   away,
        })

    return result


def _extract_team_name(line: str) -> str:
    """Pull the team name prefix before the first odds token."""
    # Strip leading/trailing whitespace
    line = line.strip()
    # Remove anything that looks like odds or numbers
    m = re.match(r"^([A-Za-z][A-Za-z\s'\.]+?)(?:\s+[+\-\dOU])", line)
    if m:
        return m.group(1).strip()
    # Fallback: first 2 words
    words = line.split()
    return " ".join(words[:2]) if len(words) >= 2 else words[0] if words else line


def parse_browsing_page(raw_text: str) -> dict:
    """
    Parse a FanDuel game lines browsing page into structured game objects.
    Returns dict with page_type='browsing_page' and a 'games' list.
    """
    lines  = raw_text.split("\n")
    games: list[dict] = []
    current: dict | None = None
    team_rows_seen = 0   # how many team-row lines we've seen for current game

    for raw_line in lines:
        line = raw_line.strip()
        if not line or len(line) < 3:
            continue
        if _is_ui_noise(line):
            continue
        if _is_pitcher_line(line):
            continue
        if _is_dollar_amount(line):
            continue

        # Game header: "Cubs @ Cardinals" or "SD Padres @ MIL Brewers >"
        header_m = _RE_GAME_HEADER.search(line)
        if header_m:
            if current:
                games.append(current)
            away_raw = header_m.group(1).strip()
            home_raw = header_m.group(2).strip()
            current = {
                "away":       away_raw,
                "home":       home_raw,
                "time":       None,
                "lines":      [],
            }
            team_rows_seen = 0
            continue

        # Time line
        time_m = _RE_TIME_LINE.search(line)
        if time_m and current and not current["time"]:
            current["time"] = time_m.group(1).strip()
            continue

        # Team row: has at least one American-odds token
        odds_tokens = _RE_ODDS_TOKEN.findall(line)
        if odds_tokens and current is not None:
            # Infer team name — first row = away team, second = home team
            home  = current["home"]
            away  = current["away"]
            team  = away if team_rows_seen == 0 else home
            extracted = _extract_team_name(line)
            # Use OCR-extracted name if it's plausible (>3 chars, not pure numbers)
            if len(extracted) > 3 and not re.fullmatch(r"[\d\s+\-\.]+", extracted):
                team = extracted

            row_lines = _parse_team_row(line, team, home, away)
            current["lines"].extend(row_lines)
            team_rows_seen += 1
            continue

    if current:
        games.append(current)

    # Warn if we found nothing meaningful
    warnings: list[str] = []
    total_lines = sum(len(g["lines"]) for g in games)
    if not games:
        warnings.append("No game blocks detected — check raw_text and try a cleaner screenshot.")
    elif total_lines == 0:
        warnings.append("Games detected but no odds lines extracted — OCR quality may be low.")

    return {
        "parse_method": "ocr_free",
        "page_type":    "browsing_page",
        "raw_text":     raw_text,
        "games":        games,
        "n_games":      len(games),
        "warnings":     warnings,
    }


def _ocr_text_from_bytes(image_bytes: bytes) -> str:
    """Run Tesseract OCR on raw image bytes. Returns extracted text."""
    try:
        import pytesseract
        from PIL import Image as _PILImage
        import io
        img = _PILImage.open(io.BytesIO(image_bytes))

        # Mobile screenshots are typically high-res — scale to ~1200px width
        # for best OCR accuracy without being too slow
        w, h = img.size
        if w > 1400:
            scale = 1400 / w
            img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)

        # Convert to greyscale and boost contrast for better OCR
        img = img.convert("L")

        # Tesseract config: PSM 6 = uniform block of text (works well for bet slips)
        text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
        return text
    except ImportError:
        return ""
    except Exception as exc:
        print(f"[fanduel_parser] OCR error: {exc}")
        return ""


# Patterns used by the free parser
_RE_AMERICAN_ODDS = re.compile(r"([+\-]\d{3,4})(?!\d)")
_RE_MONEY         = re.compile(r"\$\s*(\d{1,6}(?:\.\d{1,2})?)")
_RE_OU            = re.compile(r"\b(Over|Under)\s+(\d+\.?\d*)\b", re.IGNORECASE)
_RE_SPREAD        = re.compile(r"([+\-]\d{1,2}(?:\.\d)?)\b")
_RE_TOTAL_ODDS    = re.compile(
    r"(?:Total Odds|Combined Odds|Parlay Odds)[:\s]+([+\-]\d{3,4})", re.IGNORECASE
)
_RE_BET_PLACED    = re.compile(
    r"(?:Bet Placed|Placed|Wager placed)[:\s]+(.+)", re.IGNORECASE
)
_RE_BET_ID        = re.compile(r"Bet\s*(?:ID|#)[:\s]*([A-Z0-9\-]+)", re.IGNORECASE)

# Common FanDuel section headers to skip
_SKIP_LINES = {
    "fanduel", "sportsbook", "bet slip", "my bets", "open bets",
    "settled bets", "same game parlay", "sgp", "parlay",
    "legs", "leg", "total", "stake", "wager", "payout",
    "potential payout", "potential winnings", "share",
    "cash out", "cashout",
}

_MARKET_KEYWORDS = {
    "moneyline": "h2h", "ml": "h2h",
    "spread":    "spread",
    "over":      "totals", "under": "totals",
    "total":     "totals",
    "points":    "player_points",
    "rebounds":  "player_rebounds",
    "assists":   "player_assists",
    "3-pointers": "player_threes", "threes": "player_threes",
    "strikeouts": "player_strikeouts",
    "hits":       "player_hits",
    "home runs":  "player_home_runs",
}

_STATUS_KEYWORDS = {
    "won": "WON", "win": "WON", "winner": "WON",
    "lost": "LOST", "loss": "LOST",
    "push": "PUSH", "void": "VOID", "voided": "VOID",
    "pending": "PENDING", "live": "PENDING",
}


def _infer_market(line_text: str) -> str:
    """Infer market_type from a line of OCR text."""
    lower = line_text.lower()
    for kw, mkt in _MARKET_KEYWORDS.items():
        if kw in lower:
            return mkt
    return "player_props"


def _infer_leg_status(line_text: str) -> str:
    lower = line_text.lower()
    for kw, status in _STATUS_KEYWORDS.items():
        if kw in lower:
            return status
    return "PENDING"


def _parse_ocr_legs(lines: list[str]) -> list[dict]:
    """
    Walk OCR lines and group them into leg dicts.

    FanDuel slip layout (approximate):
      <bet description / player prop label>   ← one or two lines
      <optional odds for that leg>
      <optional status: Won / Lost / Pending>

    We use the presence of an American-odds token to "close" the current leg
    and start a new one on the next substantive line.
    """
    legs: list[dict] = []
    current: dict | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()

        # Skip pure header / footer / UI chrome lines
        if lower in _SKIP_LINES:
            continue
        # Skip very short tokens that are likely OCR noise
        if len(line) < 3:
            continue
        # Skip pure dollar amounts (stake / payout header lines)
        if re.fullmatch(r"\$[\d,\.]+", line):
            continue

        odds_m  = _RE_AMERICAN_ODDS.search(line)
        ou_m    = _RE_OU.search(line)
        money_m = _RE_MONEY.search(line)

        # Lines that are ONLY an odds token → attach to current leg
        stripped_odds = _RE_AMERICAN_ODDS.sub("", line).strip()
        is_pure_odds  = odds_m and len(stripped_odds) <= 2

        if is_pure_odds and current is not None:
            if current.get("extracted_price") is None:
                current["extracted_price"] = int(odds_m.group(1))
            continue

        # Status line → attach to current leg
        status = _infer_leg_status(line)
        if status != "PENDING" and current is not None:
            current["leg_status"] = status
            continue

        # New leg description line
        if current is not None:
            legs.append(current)

        market = _infer_market(line)
        over_under = None
        point = None

        if ou_m:
            over_under = ou_m.group(1).lower()
            try:
                point = float(ou_m.group(2))
            except ValueError:
                pass

        current = {
            "bet_info":       line,
            "market_type":    market,
            "over_under":     over_under,
            "point":          point,
            "extracted_price": int(odds_m.group(1)) if odds_m and not is_pure_odds else None,
            "leg_status":     "PENDING",
            "player":         None,
            "team":           None,
            "sport":          None,
            "home_team":      None,
            "away_team":      None,
            "is_part_of_sgp": False,
            "price_source":   "ocr_extracted",
            "price_confidence": 0.6,
            "requires_manual_entry": True,
            "matched_price":  None,
            "matched_event_id": None,
        }

        # Try to extract price even on description line
        if odds_m and not is_pure_odds and current["extracted_price"] is None:
            current["extracted_price"] = int(odds_m.group(1))
            current["matched_price"]   = current["extracted_price"]
            current["price_source"]    = "screenshot_visible"
            current["price_confidence"] = 1.0
            current["requires_manual_entry"] = False

    if current is not None:
        legs.append(current)

    # De-dupe: remove legs whose bet_info is a subset of the next leg's bet_info
    # (OCR sometimes splits one label across two lines)
    merged: list[dict] = []
    for leg in legs:
        if (merged and
                merged[-1].get("extracted_price") is None and
                len(merged[-1].get("bet_info", "")) < 6):
            merged[-1]["bet_info"] = merged[-1]["bet_info"] + " " + leg["bet_info"]
            merged[-1]["market_type"]   = leg["market_type"]
            merged[-1]["over_under"]    = leg["over_under"] or merged[-1]["over_under"]
            merged[-1]["point"]         = leg["point"] or merged[-1]["point"]
            merged[-1]["extracted_price"] = leg["extracted_price"]
            merged[-1]["leg_status"]    = leg["leg_status"]
        else:
            merged.append(leg)

    return merged


def parse_screenshot_free(image_bytes: bytes,
                           game_date: Optional[str] = None) -> dict:
    """
    Parse a FanDuel bet slip screenshot using Tesseract OCR + regex.
    No API key required.

    Returns a dict with:
      parse_method    'ocr_free'
      raw_text        full OCR output (for manual correction reference)
      legs            list of partially-parsed leg dicts
      n_legs          int
      bet_type        'straight' | 'parlay' | None
      bet_status      'PENDING' | 'SETTLED_WIN' | 'SETTLED_LOSS' etc.
      combined_odds_american  int | None
      stake           float | None
      potential_payout float | None
      warnings        list[str]
    """
    raw_text = _ocr_text_from_bytes(image_bytes)
    lines    = raw_text.split("\n")

    # Route to the right parser based on page type
    page_type = _detect_page_type(lines)
    if page_type == "browsing_page":
        result = parse_browsing_page(raw_text)
        result["bet_date"] = None
        result["bet_time"] = None
        result["timestamp_source"] = "current"
        return result

    # Bet-slip path — filter noise before leg extraction
    clean_lines = [
        l for l in lines
        if not _is_ui_noise(l.strip())
        and not _is_pitcher_line(l.strip())
        and not _is_dollar_amount(l.strip())
    ]
    legs = _parse_ocr_legs(clean_lines)

    # Global fields
    stake = None
    payout = None
    combined_am = None
    bet_status = "PENDING"

    for line in lines:
        lower = line.strip().lower()
        money_m = _RE_MONEY.search(line)
        total_m = _RE_TOTAL_ODDS.search(line)

        if total_m:
            try:
                combined_am = int(total_m.group(1))
            except ValueError:
                pass

        if money_m:
            amount = float(money_m.group(1))
            if any(k in lower for k in ("wager", "stake", "bet:", "risking", "risk")):
                stake = amount
            elif any(k in lower for k in ("payout", "winning", "return", "profit")):
                payout = amount

        # Overall bet status
        if "settled" in lower or "graded" in lower:
            if any(k in lower for k in ("won", "win", "winner")):
                bet_status = "SETTLED_WIN"
            elif any(k in lower for k in ("lost", "loss")):
                bet_status = "SETTLED_LOSS"
        elif "cashed out" in lower or "cashout" in lower:
            bet_status = "CASHED_OUT"

    # If no combined odds found but individual odds present, compute
    if combined_am is None:
        prices = [l["extracted_price"] for l in legs if l.get("extracted_price") is not None]
        if prices:
            dec = _combined_decimal(prices)
            if dec:
                combined_am = _decimal_to_american(dec)

    warnings: list[str] = []
    needs_manual = [l["bet_info"] for l in legs if l.get("requires_manual_entry")]
    if needs_manual:
        warnings.append(
            f"{len(needs_manual)} leg(s) need manual review: "
            + "; ".join(needs_manual[:3])
            + ("…" if len(needs_manual) > 3 else "")
        )

    n_legs = len(legs)
    bet_type = None
    if n_legs == 1:
        bet_type = "straight"
    elif n_legs > 1:
        bet_type = "parlay"

    return {
        "parse_method":           "ocr_free",
        "page_type":              "bet_slip",
        "raw_text":               raw_text,
        "n_legs":                 n_legs,
        "bet_type":               bet_type,
        "bet_status":             bet_status,
        "combined_odds_american": combined_am,
        "combined_odds_decimal":  round(_american_to_decimal(combined_am), 4) if combined_am else None,
        "stake":                  stake,
        "potential_payout":       payout,
        "potential_profit":       round((_american_to_decimal(combined_am) - 1) * stake, 2)
                                  if combined_am and stake else None,
        "parsed_legs":            legs,
        "warnings":               warnings,
        "is_sgp":                 False,
        "has_sgp_subset":         False,
        "boost":                  None,
        "fanduel_bet_id":         None,
    }


# ── Public parse functions ────────────────────────────────────────────────────

def parse_screenshot(image_bytes: bytes,
                     mime_type: str = "image/png",
                     game_date: Optional[str] = None) -> dict:
    """
    Parse a FanDuel screenshot.
    Returns enriched pick dict ready for review, or {"error": "..."}.
    """
    try:
        raw = _call_claude_vision(image_bytes, mime_type)
        return enrich_parsed_pick(raw, game_date=game_date)
    except json.JSONDecodeError as e:
        return {"error": f"Claude returned invalid JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}


def parse_slip_text(slip_text: str,
                    game_date: Optional[str] = None) -> dict:
    """
    Parse pasted FanDuel bet slip text.
    Returns enriched pick dict ready for review, or {"error": "..."}.
    """
    try:
        raw = _call_claude_text(slip_text)
        return enrich_parsed_pick(raw, game_date=game_date)
    except json.JSONDecodeError as e:
        return {"error": f"Claude returned invalid JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ── Validation ────────────────────────────────────────────────────────────────

def validate_pick_submission(parsed_pick: dict, mode: str, db) -> dict:
    """
    Validate a parsed pick before submitting to user_picks.

    mode: 'pre_game' | 'retroactive'

    Returns:
      {passed, warnings, blockers, requires_user_confirmation}
    """
    from database import UserPick  # late import to avoid circular

    warnings: list[str] = []
    blockers: list[str] = []

    bet_status = parsed_pick.get("bet_status", "PENDING")

    if mode == "pre_game":
        if bet_status not in ("PENDING", None):
            warnings.append(
                f"Bet status is '{bet_status}' but mode is 'pre_game'. "
                "Use retroactive mode for settled or partial bets."
            )

    elif mode == "retroactive":
        # Check 1: fanduel_bet_id dedup (only if bet_id present)
        fid = parsed_pick.get("fanduel_bet_id")
        if fid:
            existing = db.query(UserPick).filter(
                UserPick.fanduel_bet_id == fid
            ).first()
            if existing:
                blockers.append(
                    f"Bet ID {fid} already submitted at "
                    f"{existing.submitted_at.strftime('%Y-%m-%d %H:%M') if existing.submitted_at else '?'} "
                    f"with status {existing.status}."
                )

        # Check 2: timing plausibility (if bet_placed_at provided)
        bet_placed_at = parsed_pick.get("bet_placed_at")
        if bet_placed_at:
            warnings.append(
                "Bet placed timestamp noted — verify placement was before game start."
            )

        # Check 3: warn about legs with no resolved price
        for leg in parsed_pick.get("parsed_legs", []):
            if leg.get("requires_manual_entry"):
                warnings.append(
                    f"Leg '{leg.get('bet_info', '?')}' has no resolved price. "
                    "It will be recorded without odds contribution."
                )

        # Check 4: settled legs present → retroactive is expected
        leg_statuses = {l.get("leg_status") for l in parsed_pick.get("parsed_legs", [])}
        if bet_status == "PENDING" and not (leg_statuses - {"PENDING"}):
            warnings.append(
                "All legs show PENDING status in retroactive mode. "
                "Ensure this bet is not still live."
            )

    return {
        "passed":                    len(blockers) == 0,
        "warnings":                  warnings,
        "blockers":                  blockers,
        "requires_user_confirmation": len(warnings) > 0,
    }
