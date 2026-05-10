"""
leg_resolver.py — Retrospective leg outcome resolution from historical.db.

Cross-references game results against parsed bet leg descriptions to populate
leg_result, accuracy_delta, resolution_source, and actual_value on bet_legs rows.

Resolution hierarchy
────────────────────
1. historical_db   — moneyline, spread, total resolved from games + scores
2. pitcher_logs    — MLB strikeout props from pitcher_game_logs
3. inferred_parlay_win — parlay SETTLED_WIN → all legs must have won
4. unresolvable    — sport in DB but no matching game found
5. sport_not_in_db — Tennis, MMA, etc. (no historical data)

accuracy_delta convention
─────────────────────────
  Positive  = cushion (won with room to spare)
  Negative  = miss   (lost or squeaked past line)

  Spread:  delta = selected_score + line - opponent_score
           e.g. Yankees +1.5, won 5-3 → 5 + 1.5 - 3 = +3.5
  Total:   delta = actual_combined - line  (Over bets)
           delta = line - actual_combined  (Under bets)
           e.g. Brunson Over 22.5, scored 19 → 19 - 22.5 = -3.5
  Moneyline: delta = selected_score - opponent_score (margin of victory)
"""
from __future__ import annotations

import re
import sqlite3
import os
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from database import Bet, BetLeg
from soccer_data import get_soccer_result, fetch_soccer_results, store_soccer_results


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEAM ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

# Keys = canonical display name.
# Values = list of aliases (nicknames, abbreviations, alt spellings).
# IMPORTANT: Historical.db uses:
#   MLB → full names  (New York Yankees)
#   NBA → 3-letter    (NYK)
#   NHL → 3-letter    (PIT)
#   NFL → 2-3 letter  (KC, LAR, GB)
# _TEAM_TO_DB maps canonical → historical.db exact string.

TEAM_ALIASES: dict[str, list[str]] = {
    # ── MLB ───────────────────────────────────────────────────────────────────
    "Arizona Diamondbacks":  ["Diamondbacks", "ARI", "D-Backs", "D'Backs"],
    "Atlanta Braves":        ["Braves", "ATL"],
    "Baltimore Orioles":     ["Orioles", "BAL"],
    "Boston Red Sox":        ["Red Sox", "BOS"],
    "Chicago Cubs":          ["Cubs", "CHC"],
    "Chicago White Sox":     ["White Sox", "CWS", "CHW"],
    "Cincinnati Reds":       ["Reds", "CIN"],
    "Cleveland Guardians":   ["Guardians", "CLE", "Cleveland"],
    "Colorado Rockies":      ["Rockies", "COL"],
    "Detroit Tigers":        ["Tigers", "DET"],
    "Houston Astros":        ["Astros", "HOU"],
    "Kansas City Royals":    ["Royals", "KC", "KCR", "KCR"],
    "Los Angeles Angels":    ["Angels", "LAA", "Halos"],
    "Los Angeles Dodgers":   ["Dodgers", "LAD", "LA Dodgers"],
    "Miami Marlins":         ["Marlins", "MIA", "Florida Marlins", "FLA"],
    "Milwaukee Brewers":     ["Brewers", "MIL"],
    "Minnesota Twins":       ["Twins", "MIN"],
    "New York Mets":         ["Mets", "NYM"],
    "New York Yankees":      ["Yankees", "NYY", "NY Yankees"],
    "Athletics":             ["A's", "OAK", "Oakland Athletics", "Las Vegas Athletics", "LVA"],
    "Philadelphia Phillies": ["Phillies", "PHI"],
    "Pittsburgh Pirates":    ["Pirates", "PIT"],
    "San Diego Padres":      ["Padres", "SD", "SDP"],
    "San Francisco Giants":  ["Giants", "SF", "SFG"],
    "Seattle Mariners":      ["Mariners", "SEA"],
    "St. Louis Cardinals":   ["Cardinals", "STL", "Cards"],
    "Tampa Bay Rays":        ["Rays", "TB", "TBR"],
    "Texas Rangers":         ["Rangers", "TEX"],
    "Toronto Blue Jays":     ["Blue Jays", "TOR", "Jays"],
    "Washington Nationals":  ["Nationals", "WSH", "WAS", "Nats"],
    # ── NBA ───────────────────────────────────────────────────────────────────
    "Atlanta Hawks":             ["Hawks", "ATL"],
    "Boston Celtics":            ["Celtics", "BOS"],
    "Brooklyn Nets":             ["Nets", "BKN", "NJN"],
    "Charlotte Hornets":         ["Hornets", "CHA", "Charlotte"],
    "Chicago Bulls":             ["Bulls", "CHI"],
    "Cleveland Cavaliers":       ["Cavaliers", "CLE", "Cavs"],
    "Dallas Mavericks":          ["Mavericks", "DAL", "Mavs"],
    "Denver Nuggets":            ["Nuggets", "DEN"],
    "Detroit Pistons":           ["Pistons", "DET"],
    "Golden State Warriors":     ["Warriors", "GSW", "GS"],
    "Houston Rockets":           ["Rockets", "HOU"],
    "Indiana Pacers":            ["Pacers", "IND"],
    "Los Angeles Clippers":      ["Clippers", "LAC"],
    "Los Angeles Lakers":        ["Lakers", "LAL"],
    "Memphis Grizzlies":         ["Grizzlies", "MEM"],
    "Miami Heat":                ["Heat", "MIA"],
    "Milwaukee Bucks":           ["Bucks", "MIL"],
    "Minnesota Timberwolves":    ["Timberwolves", "MIN", "Wolves"],
    "New Orleans Pelicans":      ["Pelicans", "NOP", "NO"],
    "New York Knicks":           ["Knicks", "NYK"],
    "Oklahoma City Thunder":     ["Thunder", "OKC"],
    "Orlando Magic":             ["Magic", "ORL"],
    "Philadelphia 76ers":        ["76ers", "PHI", "Sixers"],
    "Phoenix Suns":              ["Suns", "PHX", "PHO"],
    "Portland Trail Blazers":    ["Trail Blazers", "POR", "Blazers"],
    "Sacramento Kings":          ["Kings", "SAC"],
    "San Antonio Spurs":         ["Spurs", "SAS", "SA"],
    "Toronto Raptors":           ["Raptors", "TOR"],
    "Utah Jazz":                 ["Jazz", "UTA"],
    "Washington Wizards":        ["Wizards", "WAS", "WSH"],
    # ── NHL ───────────────────────────────────────────────────────────────────
    "Anaheim Ducks":             ["Ducks", "ANA"],
    "Boston Bruins":             ["Bruins", "BOS"],
    "Buffalo Sabres":            ["Sabres", "BUF"],
    "Calgary Flames":            ["Flames", "CGY"],
    "Carolina Hurricanes":       ["Hurricanes", "CAR", "Canes"],
    "Chicago Blackhawks":        ["Blackhawks", "CHI"],
    "Colorado Avalanche":        ["Avalanche", "COL", "Avs"],
    "Columbus Blue Jackets":     ["Blue Jackets", "CBJ"],
    "Dallas Stars":              ["Stars", "DAL"],
    "Detroit Red Wings":         ["Red Wings", "DET"],
    "Edmonton Oilers":           ["Oilers", "EDM"],
    "Florida Panthers":          ["Panthers", "FLA", "FLO"],
    "Los Angeles Kings":         ["Kings", "LAK", "LA Kings"],
    "Minnesota Wild":            ["Wild", "MIN"],
    "Montreal Canadiens":        ["Canadiens", "MTL", "Habs"],
    "Nashville Predators":       ["Predators", "NSH", "Preds"],
    "New Jersey Devils":         ["Devils", "NJD"],
    "New York Islanders":        ["Islanders", "NYI"],
    "New York Rangers":          ["Rangers", "NYR"],
    "Ottawa Senators":           ["Senators", "OTT"],
    "Philadelphia Flyers":       ["Flyers", "PHI"],
    "Pittsburgh Penguins":       ["Penguins", "PIT"],
    "San Jose Sharks":           ["Sharks", "SJS"],
    "Seattle Kraken":            ["Kraken", "SEA"],
    "St. Louis Blues":           ["Blues", "STL"],
    "Tampa Bay Lightning":       ["Lightning", "TBL", "TB Lightning"],
    "Toronto Maple Leafs":       ["Maple Leafs", "TOR", "Leafs"],
    "Utah Mammoth":              ["Mammoth", "UTA", "Arizona Coyotes", "Coyotes", "ARI"],
    "Vancouver Canucks":         ["Canucks", "VAN"],
    "Vegas Golden Knights":      ["Golden Knights", "VGK", "Vegas"],
    "Washington Capitals":       ["Capitals", "WSH", "WSH", "Caps"],
    "Winnipeg Jets":             ["Jets", "WPG"],
    # ── NFL ───────────────────────────────────────────────────────────────────
    "Arizona Cardinals":         ["Cardinals", "ARI"],
    "Atlanta Falcons":           ["Falcons", "ATL"],
    "Baltimore Ravens":          ["Ravens", "BAL"],
    "Buffalo Bills":             ["Bills", "BUF"],
    "Carolina Panthers":         ["Panthers", "CAR"],
    "Chicago Bears":             ["Bears", "CHI"],
    "Cincinnati Bengals":        ["Bengals", "CIN"],
    "Cleveland Browns":          ["Browns", "CLE"],
    "Dallas Cowboys":            ["Cowboys", "DAL"],
    "Denver Broncos":            ["Broncos", "DEN"],
    "Detroit Lions":             ["Lions", "DET"],
    "Green Bay Packers":         ["Packers", "GB", "Green Bay"],
    "Houston Texans":            ["Texans", "HOU"],
    "Indianapolis Colts":        ["Colts", "IND"],
    "Jacksonville Jaguars":      ["Jaguars", "JAX"],
    "Kansas City Chiefs":        ["Chiefs", "KC", "Kansas City"],
    "Los Angeles Rams":          ["Rams", "LA", "LAR"],
    "Los Angeles Chargers":      ["Chargers", "LAC"],
    "Las Vegas Raiders":         ["Raiders", "LV", "Oakland Raiders"],
    "Miami Dolphins":            ["Dolphins", "MIA"],
    "Minnesota Vikings":         ["Vikings", "MIN"],
    "New England Patriots":      ["Patriots", "NE", "New England"],
    "New Orleans Saints":        ["Saints", "NO"],
    "New York Giants":           ["Giants", "NYG"],
    "New York Jets":             ["Jets", "NYJ"],
    "Philadelphia Eagles":       ["Eagles", "PHI"],
    "Pittsburgh Steelers":       ["Steelers", "PIT"],
    "Seattle Seahawks":          ["Seahawks", "SEA"],
    "San Francisco 49ers":       ["49ers", "SF", "Niners"],
    "Tampa Bay Buccaneers":      ["Buccaneers", "TB", "Bucs"],
    "Tennessee Titans":          ["Titans", "TEN"],
    "Washington Commanders":     ["Commanders", "WAS", "Washington Football Team", "Redskins"],
}

# historical.db canonical string per sport
# MLB → full name,  NBA/NHL/NFL → abbreviation
_TEAM_TO_DB: dict[str, dict[str, str]] = {
    # MLB (full names match historical.db)
    "MLB": {k: k for k in [
        "Arizona Diamondbacks","Atlanta Braves","Baltimore Orioles","Boston Red Sox",
        "Chicago Cubs","Chicago White Sox","Cincinnati Reds","Cleveland Guardians",
        "Colorado Rockies","Detroit Tigers","Houston Astros","Kansas City Royals",
        "Los Angeles Angels","Los Angeles Dodgers","Miami Marlins","Milwaukee Brewers",
        "Minnesota Twins","New York Mets","New York Yankees","Athletics",
        "Philadelphia Phillies","Pittsburgh Pirates","San Diego Padres","San Francisco Giants",
        "Seattle Mariners","St. Louis Cardinals","Tampa Bay Rays","Texas Rangers",
        "Toronto Blue Jays","Washington Nationals",
    ]},
    # NBA (3-letter codes)
    "NBA": {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    },
    # NHL (3-letter codes)
    "NHL": {
        "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
        "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
        "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
        "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
        "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
        "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
        "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
        "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
        "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
        "Utah Mammoth": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
        "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
    },
    # NFL (2-3 letter codes)
    "NFL": {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
        "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
        "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
        "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    },
}

# Build reverse alias index: any alias/abbrev → canonical full name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in TEAM_ALIASES.items():
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical
    for _a in _aliases:
        _ALIAS_TO_CANONICAL[_a.lower()] = _canonical

# Sports that have game data in historical.db
_RESOLVABLE_SPORTS = {"MLB", "NBA", "NHL", "NFL"}
_SPORT_ABBREV_MAP = {
    # FanDuel/Pikkit sport strings → historical.db sport key
    "baseball":             "MLB",
    "major league baseball":"MLB",
    "mlb":                  "MLB",
    "basketball":           "NBA",
    "nba":                  "NBA",
    "ice hockey":           "NHL",
    "nhl":                  "NHL",
    "nhl games":            "NHL",
    "american football":    "NFL",
    "nfl":                  "NFL",
    "soccer":               "Soccer",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEAM NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_team(name: str) -> str:
    """
    Map any team name variant to canonical form (full name).

    Tries in order:
      1. Exact match as canonical key
      2. Alias lookup (case-insensitive)
      3. Last-word fallback (e.g. "Warriors" → "Golden State Warriors")
      4. Returns original name unchanged if no match
    """
    if not name:
        return name
    name = name.strip()
    # Exact canonical
    if name in TEAM_ALIASES:
        return name
    # Case-insensitive alias lookup
    canonical = _ALIAS_TO_CANONICAL.get(name.lower())
    if canonical:
        return canonical
    # Last-word fallback
    last = name.split()[-1].lower()
    for canonical, aliases in TEAM_ALIASES.items():
        if last == canonical.split()[-1].lower():
            return canonical
        if any(last == a.lower() for a in aliases):
            return canonical
    return name


def to_db_team(canonical: str, sport: str) -> str:
    """
    Convert canonical team name to historical.db exact team string.
    Returns canonical unchanged if no mapping found (e.g. MLB names match directly).
    """
    db_map = _TEAM_TO_DB.get(sport, {})
    return db_map.get(canonical, canonical)


def infer_sport(leg_sport: str, bet_sport: str) -> str:
    """
    Return the normalised sport key for historical.db lookup.
    Returns None if the sport is not in historical.db.
    """
    for src in (leg_sport, bet_sport):
        if not src:
            continue
        key = _SPORT_ABBREV_MAP.get(src.lower().strip())
        if key:
            return key
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LEG DESCRIPTION PARSER
# ═══════════════════════════════════════════════════════════════════════════════

_STAT_KEYWORDS: dict[str, list[str]] = {
    "points":         ["points", "pts", "alt points", "to score"],
    "rebounds":       ["rebounds", "reb", "alt rebounds"],
    "assists":        ["assists", "ast", "alt assists"],
    "strikeouts":     ["strikeouts", "k's", "ks"],
    "home_runs":      ["home run", "hr", "to hit a home run"],
    "hits":           ["hits"],
    "walks":          ["walks", "bb"],
    "rushing_yards":  ["rushing yds", "rushing yards", "alt rushing yds"],
    "receiving_yards":["receiving yds", "receiving yards"],
    "passing_yards":  ["passing yds", "passing yards"],
    "goals":          ["goals", "to score or assist", "shots on target"],
    "corners":        ["corners"],
    "saves":          ["saves"],
}


def _detect_stat_type(text: str) -> Optional[str]:
    t = text.lower()
    for stat, keywords in _STAT_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return stat
    return None


def _strip_parens(text: str) -> str:
    """Remove parenthesised content including pitcher names like (M Fried)."""
    return re.sub(r'\s*\([^)]*\)', '', text).strip()


def _extract_trailing_odds(text: str) -> tuple[str, Optional[int]]:
    """Remove and return trailing American odds like -162 or +350."""
    m = re.search(r'\s+([+-]\d{2,4})\s*$', text)
    if m:
        return text[:m.start()].strip(), int(m.group(1))
    return text, None


def _extract_matchup(raw: str) -> tuple[str, str]:
    """
    Extract away_team and home_team from a matchup string.
    Handles:
      "Kansas City Royals (pitcher) @ New York Yankees (pitcher) -320"
      "Atletico Madrid v Barcelona"
    Returns ("", "") on failure.
    """
    cleaned, _ = _extract_trailing_odds(raw)
    cleaned = _strip_parens(cleaned).strip()

    for sep in (" @ ", " v ", " vs ", " VS "):
        if sep in cleaned:
            parts = cleaned.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return "", ""


def parse_leg_details(description: str) -> dict:
    """
    Extract structured fields from a raw leg description string.

    Handles both FanDuel format ({selection} ({market}) — {matchup} {odds})
    and Pikkit format ({selection} {matchup} without separator).

    Returns dict:
      selected_team_or_player: str
      market_type:  Moneyline | Spread | Total | Player Prop | Other
      line:         float | None      — numeric threshold
      direction:    str | None        — "Over" / "Under" / "+" / "-"
      stat_type:    str | None        — points / rebounds / strikeouts / …
      away_team:    str               — normalized canonical name
      home_team:    str               — normalized canonical name
      odds:         int | None        — American odds
    """
    if not description:
        return _empty_parse()

    desc = description.strip()
    market_hint = ""
    sel_raw = ""
    matchup_raw = ""

    # ── FanDuel format: has " — " separator ──────────────────────────────────
    for sep in (" — ", " – ", " - "):
        if sep in desc:
            idx = desc.index(sep)
            sel_raw = desc[:idx].strip()
            matchup_raw = desc[idx + len(sep):].strip()
            break
    else:
        # ── Pikkit / app format: find the @ or v matchup marker ───────────────
        if " @ " in desc or " v " in desc:
            # ── Soccer Pikkit: "...{Away Team} v {Home Team}" at end of string ──
            # Use rfind so we always pick the matchup separator, not an
            # accidental " v " that appears in market label text.
            # Walk backwards from " v " collecting Title-Case words that are
            # NOT known soccer market stat words.
            if " v " in desc and " @ " not in desc:
                _SOCCER_NON_TEAM = frozenset({
                    "Goals", "Goal", "Corners", "Corner", "Shots", "Shot",
                    "Target", "Targets", "Points", "Point", "Assists", "Assist",
                    "Penalty", "Penalties", "Cards", "Card", "Yellow", "Red",
                    "Over", "Under", "Total", "Match", "Or", "More", "Than",
                    "Have", "Score", "And", "The", "To", "In", "Of", "On",
                    "Player", "Anytime", "Scorer",
                })
                # Soccer Pikkit format: "Home v Away"
                # LEFT of " v " = home team, RIGHT of " v " = away team
                v_idx = desc.rfind(" v ")
                away_cand = desc[v_idx + 3:].strip()   # right = away
                left_of_v = desc[:v_idx]
                words = left_of_v.split()
                home_words = []
                for w in reversed(words):
                    clean = re.sub(r"[^a-zA-ZÀ-ÿ\']", "", w)
                    if clean and clean[0].isupper() and clean not in _SOCCER_NON_TEAM:
                        home_words.insert(0, w)
                    else:
                        break
                home_cand = " ".join(home_words).strip()
                # Build matchup in standard "Away @ Home" format for _extract_matchup
                matchup_raw = (away_cand + " @ " + home_cand).strip(" @")
                # Selection = everything before the home team name in left_of_v
                if home_cand:
                    boundary = left_of_v.rfind(home_cand)
                    sel_raw = left_of_v[:boundary].strip() if boundary >= 0 else left_of_v.strip()
                else:
                    sel_raw = left_of_v.strip()
            else:
                # Find where matchup starts by locating the @ then walking left
                # to the start of the away team name (first capital letter word)
                at_idx = desc.find(" @ ") if " @ " in desc else desc.find(" v ")
                left_of_at = desc[:at_idx]
                right_of_at = desc[at_idx + 3:]

                # Market keyword positions — selection ends just before the away team
                _MKT_MARKERS = [
                    "moneyline", "run line", "puck line", "spread", "total runs",
                    "total goals", "total points", "over/under", " over ", " under ",
                    "alternate run lines", "alt spread", "alternate spread",
                ]
                mkt_pos = len(left_of_at)
                for kw in _MKT_MARKERS:
                    pos = left_of_at.lower().find(kw)
                    if pos != -1 and pos < mkt_pos:
                        mkt_pos = pos

                if mkt_pos < len(left_of_at):
                    sel_raw = left_of_at[:mkt_pos].strip()
                    # The rest of left_of_at might be partial away team name or market words
                    away_candidate = left_of_at[mkt_pos:].strip()
                    # Strip market keywords to get the away team
                    for kw in [
                        "Alternate Run Lines", "Run Line", "Puck Line", "Spread",
                        "Total Runs", "Total Goals", "Total Points", "Over/Under",
                        "Total", "Moneyline",
                    ]:
                        away_candidate = re.sub(re.escape(kw), "", away_candidate, flags=re.I).strip()
                    # Remove numeric line values
                    away_candidate = re.sub(r'\s*[+-]?\d+\.?\d*\s*', ' ', away_candidate).strip()
                    matchup_raw = (away_candidate + " @ " + right_of_at).strip(" @")
                else:
                    # Couldn't find market marker — treat whole left as selection
                    sel_raw = left_of_at.strip()
                    matchup_raw = right_of_at.strip()
        else:
            sel_raw = desc

    # ── Extract odds ─────────────────────────────────────────────────────────
    matchup_raw, odds = _extract_trailing_odds(matchup_raw)
    if odds is None:
        sel_raw, odds = _extract_trailing_odds(sel_raw)

    # ── Extract market hint from parentheses in selection ────────────────────
    hint_m = re.search(r'\(([^)]+)\)', sel_raw)
    if hint_m:
        market_hint = hint_m.group(1)
        sel_raw_clean = _strip_parens(sel_raw)
    else:
        sel_raw_clean = sel_raw

    # ── Soccer market sub-type detection (before _classify_extended) ─────────
    _desc_lower = desc.lower()
    _subtype: Optional[str] = None
    if "double chance" in _desc_lower:
        market_type = "Double Chance"
        _subtype    = "double_chance"
    elif "both teams to score" in _desc_lower or _desc_lower.startswith("btts"):
        market_type = "BTTS"
        _subtype    = "btts"
    elif "corner" in _desc_lower and re.search(r'\b(over|under|\d)', _desc_lower):
        market_type = "Corners"
        _subtype    = "corners"
    else:
        # ── Classify market ───────────────────────────────────────────────────
        from fanduel_importer import _classify_extended
        market_type = _classify_extended(sel_raw_clean + " " + market_hint, market_hint)

    # ── Extract direction and line ───────────────────────────────────────────
    direction: Optional[str] = None
    line: Optional[float] = None
    stat_type = _detect_stat_type(market_hint + " " + sel_raw_clean)

    # Direction from "Over" / "Under" keyword
    dir_m = re.search(r'\b(Over|Under)\b', sel_raw_clean + " " + market_hint, re.I)
    if dir_m:
        direction = dir_m.group(1).capitalize()
        # Remove direction word to isolate line
        text_for_line = re.sub(r'\b(over|under)\b', '', sel_raw_clean + " " + market_hint, flags=re.I)
    else:
        text_for_line = sel_raw_clean

    # Extract numeric line from: "Over 22.5", "+1.5", "15+", "-1.5", "1.5 Run"
    line_m = re.search(r'([+-]?\d+\.?\d*)\s*\+?\s*(?:points?|pts?|runs?|goals?|assists?|rebounds?|strikeouts?|yards?|yds?|corners?|home runs?|hits?)?', text_for_line)
    if line_m:
        raw_line = line_m.group(1)
        try:
            line = float(raw_line)
            if line == 0:
                line = None
        except ValueError:
            line = None

    # Direction from spread sign when no Over/Under
    if direction is None and line is not None and market_type == "Spread":
        direction = "+" if line >= 0 else "-"

    # ── Extract team / player ────────────────────────────────────────────────
    sel_clean = sel_raw_clean

    # For totals / direction-first format, selection may be empty or just Over/Under
    if direction and line is not None:
        # Remove direction + line from selection
        sel_clean = re.sub(
            r'\b(Over|Under)\b\s*[+-]?\d+\.?\d*\+?\s*', '', sel_clean, flags=re.I
        ).strip()
        sel_clean = re.sub(r'^[+-]?\d+\.?\d*\+?\s*', '', sel_clean).strip()
    elif line is not None and market_type == "Spread":
        sel_clean = re.sub(r'\s*[+-]?\d+\.?\d*\s*$', '', sel_clean).strip()

    selected = sel_clean.strip()

    # ── Parse matchup teams ───────────────────────────────────────────────────
    away_raw, home_raw = _extract_matchup(matchup_raw)
    away_team = normalize_team(away_raw)
    home_team = normalize_team(home_raw)

    # If selected team not yet cleaned up to just a name, strip remaining markers
    for kw in ["Moneyline", "Run Line", "Puck Line", "Spread", "Alternate"]:
        selected = re.sub(re.escape(kw), "", selected, flags=re.I).strip()
    selected = normalize_team(selected.strip())

    return {
        "selected_team_or_player": selected,
        "market_type":  market_type,
        "line":         line,
        "direction":    direction,
        "stat_type":    stat_type,
        "away_team":    away_team,
        "home_team":    home_team,
        "odds":         odds,
        "subtype":      _subtype,
    }


def _empty_parse() -> dict:
    return {
        "selected_team_or_player": "",
        "market_type": "Other",
        "line": None, "direction": None, "stat_type": None,
        "away_team": "", "home_team": "", "odds": None,
        "subtype": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GAME LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

def _find_game(
    cur: sqlite3.Cursor,
    sport: str,
    team_a: str,
    team_b: str,
    game_date: str,
    window: int = 1,
) -> Optional[tuple]:
    """
    Find a game row (home_team, away_team, home_score, away_score, game_date)
    within ±window days of game_date matching either team_a or team_b on each side.

    Returns the first matching row or None.
    """
    from datetime import datetime, timedelta
    try:
        base = datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        return None

    dates = [
        (base + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(-window, window + 1)
    ]
    placeholders = ",".join("?" * len(dates))

    # Try both orderings (either team could be home or away)
    for t1, t2 in [(team_a, team_b), (team_b, team_a)]:
        if not t1 or not t2:
            continue
        cur.execute(f"""
            SELECT home_team, away_team, home_score, away_score, game_date
            FROM games
            WHERE sport = ?
              AND game_date IN ({placeholders})
              AND (
                (home_team = ? AND away_team = ?)
                OR (home_team LIKE ? AND away_team LIKE ?)
              )
              AND home_score IS NOT NULL
        """, [sport] + dates + [t1, t2, f"%{t1}%", f"%{t2}%"])
        row = cur.fetchone()
        if row:
            return row

    # Fallback: find any game on those dates involving team_a
    if team_a:
        cur.execute(f"""
            SELECT home_team, away_team, home_score, away_score, game_date
            FROM games
            WHERE sport = ?
              AND game_date IN ({placeholders})
              AND (home_team = ? OR away_team = ? OR home_team LIKE ? OR away_team LIKE ?)
              AND home_score IS NOT NULL
        """, [sport] + dates + [team_a, team_a, f"%{team_a}%", f"%{team_a}%"])
        row = cur.fetchone()
        if row:
            return row

    return None


def _pitcher_strikeouts(
    cur: sqlite3.Cursor,
    pitcher_name: str,
    game_date: str,
) -> Optional[int]:
    """Look up pitcher strikeout total from pitcher_game_logs."""
    # Try exact name then partial
    for query in [
        "SELECT strikeouts FROM pitcher_game_logs WHERE pitcher_name = ? AND game_date = ?",
        "SELECT strikeouts FROM pitcher_game_logs WHERE pitcher_name LIKE ? AND game_date = ?",
    ]:
        name = pitcher_name if "?" not in query else f"%{pitcher_name.split()[-1]}%"
        try:
            cur.execute(query, (name, game_date))
            row = cur.fetchone()
            if row:
                return row[0]
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RESOLUTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_from_game(
    parsed: dict,
    game: tuple,
    sport: str,
) -> dict:
    """
    Given parsed leg details and a game row, compute leg_result, accuracy_delta,
    actual_value.

    game = (home_team, away_team, home_score, away_score, game_date)
    """
    home_team, away_team, home_score, away_score, _ = game

    selected = normalize_team(parsed["selected_team_or_player"])
    db_selected = to_db_team(selected, sport) if selected else None

    # Identify which side the selected team is on
    is_home = db_selected and (
        db_selected == home_team or
        (db_selected and db_selected.lower() in home_team.lower())
    )
    is_away = db_selected and (
        db_selected == away_team or
        (db_selected and db_selected.lower() in away_team.lower())
    )

    if is_home:
        sel_score, opp_score = home_score, away_score
    elif is_away:
        sel_score, opp_score = away_score, home_score
    else:
        # Can't identify which side — fall back to canonical name matching
        sel_norm = selected.lower() if selected else ""
        if sel_norm and sel_norm in home_team.lower():
            sel_score, opp_score = home_score, away_score
        elif sel_norm and sel_norm in away_team.lower():
            sel_score, opp_score = away_score, home_score
        else:
            # For totals, no selected team needed
            sel_score, opp_score = None, None

    mkt   = parsed["market_type"]
    line  = parsed["line"]
    dirn  = parsed["direction"]

    # ── Moneyline ────────────────────────────────────────────────────────────
    if mkt == "Moneyline":
        if sel_score is None:
            return _unresolvable("can't identify selected team side")
        actual = float(sel_score)
        delta  = float(sel_score - opp_score)
        if sel_score > opp_score:
            result = "WIN"
        elif sel_score < opp_score:
            result = "LOSS"
        else:
            result = "PUSH"
        return dict(leg_result=result, accuracy_delta=delta,
                    actual_value=actual, resolution_source="historical_db")

    # ── Spread / Run Line / Puck Line ────────────────────────────────────────
    if mkt == "Spread":
        if sel_score is None or line is None:
            return _unresolvable("missing score or line for spread")
        # delta = sel_score + line - opp_score  (positive → covered)
        delta  = float(sel_score + line - opp_score)
        actual = float(sel_score)
        if delta > 0:
            result = "WIN"
        elif delta < 0:
            result = "LOSS"
        else:
            result = "PUSH"
        return dict(leg_result=result, accuracy_delta=delta,
                    actual_value=actual, resolution_source="historical_db")

    # ── Total (Over/Under) ────────────────────────────────────────────────────
    if mkt == "Total":
        if line is None:
            return _unresolvable("missing line for total")
        combined = float(home_score + away_score)
        # Sanity check: if the combined game score vastly exceeds the line,
        # this leg was almost certainly a player prop misrouted here.
        # Game-total lines: NFL ~47, NHL ~5.5, MLB ~8.5, NBA ~230.
        # Player prop lines: points ≤50, assists ≤20, yards ≤400.
        # A combined > 100 against a line < 50 is always wrong.
        if combined > 100 and line < 50:
            return _unresolvable(
                "sanity_fail: combined game score too large vs line "
                f"(combined={combined}, line={line}) — likely misclassified player prop"
            )
        if dirn == "Over":
            delta  = combined - line
            result = "WIN" if combined > line else ("PUSH" if combined == line else "LOSS")
        elif dirn == "Under":
            delta  = line - combined
            result = "WIN" if combined < line else ("PUSH" if combined == line else "LOSS")
        else:
            # No direction — can still store actual combined
            delta  = combined - line if line else None
            result = None
            return dict(leg_result=result, accuracy_delta=delta,
                        actual_value=combined, resolution_source="historical_db")
        return dict(leg_result=result, accuracy_delta=delta,
                    actual_value=combined, resolution_source="historical_db")

    return _unresolvable(f"market {mkt} not resolvable from scores")


def _unresolvable(reason: str = "") -> dict:
    return dict(leg_result=None, accuracy_delta=None,
                actual_value=None, resolution_source="unresolvable")


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. SOCCER RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

_SOCCER_NORM_RE = re.compile(
    # "real", "athletic", "city", "united" excluded — see soccer_data.py for rationale.
    r"\b(fc|sc|cf|ac|as|rc|ud|fk|sk|bk|sporting|club|de|del|la|el)\b",
    re.IGNORECASE,
)


_SOCCER_TEAM_ALIASES: dict[str, str] = {
    "man city":          "manchester city",
    "man utd":           "manchester united",
    "man united":        "manchester united",
    "inter":             "internazionale",
    "inter milan":       "internazionale milan",
    "lyon":              "olympique lyonnais",
    "paris st-g":        "paris saint-germain",
    "paris sg":          "paris saint-germain",
    "psg":               "paris saint-germain",
    "sporting lisbon":   "sporting clube portugal",
    "sporting cp":       "sporting clube portugal",
    "benfica":           "sl benfica",
    "fc copenhagen":     "fc kobenhavn",
    "copenhagen":        "kobenhavn",
    "crvena zvezda":     "crvena zvezda",
    "red star":          "crvena zvezda",
    "red star belgrade": "crvena zvezda",
    "oviedo":            "real oviedo",
}


_SOCCER_STROKE_TRANS = str.maketrans({
    'ø': 'o', 'Ø': 'o',
    'ł': 'l', 'Ł': 'l',
    'ð': 'd', 'Ð': 'd',
    'þ': 'th',
    'æ': 'ae', 'Æ': 'ae',
    'œ': 'oe', 'Œ': 'oe',
    'ß': 'ss',
})


def _norm_soccer_team(name: str) -> str:
    import unicodedata as _ud
    s = _ud.normalize("NFKD", name)
    s = "".join(c for c in s if not _ud.combining(c))
    s = s.translate(_SOCCER_STROKE_TRANS)
    s = s.lower().strip()
    s = _SOCCER_TEAM_ALIASES.get(s, s)
    s = _SOCCER_NORM_RE.sub("", s)
    s = s.replace("/", " ")
    s = s.replace("munchen", "munich")
    return re.sub(r"\s+", " ", s).strip()


def _soccer_team_match(needle: str, candidate: str) -> bool:
    n, c = _norm_soccer_team(needle), _norm_soccer_team(candidate)
    if n == c:
        return True
    nw, cw = set(n.split()), set(c.split())
    if not nw or not cw:
        return False
    shorter, longer = (nw, cw) if len(nw) <= len(cw) else (cw, nw)
    if len(shorter) == 1:
        return next(iter(shorter)) in longer
    return shorter.issubset(longer)


def resolve_soccer_leg(parsed: dict, game: "SoccerResult") -> dict:  # type: ignore[name-defined]
    """
    Resolve a single soccer bet leg against a SoccerResult row.

    Handles: Moneyline (3-way), Total Goals, Spread (Asian handicap),
             Double Chance, BTTS, Corners.

    Returns same shape as _resolve_from_game():
      {leg_result, accuracy_delta, actual_value, resolution_source}
    """
    home_goals: Optional[int] = game.home_goals
    away_goals: Optional[int] = game.away_goals

    if home_goals is None or away_goals is None:
        return _unresolvable("soccer game scores not available")

    h = float(home_goals)
    a = float(away_goals)
    combined = h + a

    mkt      = parsed.get("market_type", "")
    subtype  = parsed.get("subtype")
    line     = parsed.get("line")
    dirn     = parsed.get("direction")
    selected = parsed.get("selected_team_or_player", "")

    # Identify which side the bettor picked
    side: Optional[str] = None   # "home" | "away" | "draw"
    if selected.lower() in ("draw", "the draw", "x"):
        side = "draw"
    elif _soccer_team_match(selected, game.home_team):
        side = "home"
    elif _soccer_team_match(selected, game.away_team):
        side = "away"

    # ── Double Chance ─────────────────────────────────────────────────────────
    if subtype == "double_chance" or mkt == "Double Chance":
        if side is None:
            return _unresolvable("can't identify selected team for double chance")
        if side == "home":
            result = "WIN" if h >= a else "LOSS"   # win or draw both count
        elif side == "away":
            result = "WIN" if a >= h else "LOSS"
        else:
            # "Draw or Home" style — less common; treat as unresolvable
            return _unresolvable("draw double-chance variant not supported")
        delta = h - a if side == "home" else a - h
        return dict(leg_result=result, accuracy_delta=round(delta, 2),
                    actual_value=delta, resolution_source="soccer_api")

    # ── BTTS ─────────────────────────────────────────────────────────────────
    if subtype == "btts" or mkt == "BTTS":
        desc_lower = (parsed.get("selected_team_or_player", "") + " " +
                      (parsed.get("stat_type") or "")).lower()
        is_yes = "no" not in desc_lower
        both_scored = h > 0 and a > 0
        result = "WIN" if (both_scored == is_yes) else "LOSS"
        return dict(leg_result=result, accuracy_delta=None,
                    actual_value=combined, resolution_source="soccer_api")

    # ── Corners ───────────────────────────────────────────────────────────────
    if subtype == "corners" or mkt == "Corners":
        home_c = game.home_corners
        away_c = game.away_corners
        if home_c is None or away_c is None:
            return _unresolvable("corner stats not fetched yet")
        total_c = float(home_c + away_c)
        if line is None:
            return _unresolvable("missing line for corners total")
        if dirn == "Over":
            delta  = total_c - line
            result = "WIN" if total_c > line else ("PUSH" if total_c == line else "LOSS")
        elif dirn == "Under":
            delta  = line - total_c
            result = "WIN" if total_c < line else ("PUSH" if total_c == line else "LOSS")
        else:
            return _unresolvable("missing direction for corners")
        return dict(leg_result=result, accuracy_delta=round(delta, 2),
                    actual_value=total_c, resolution_source="soccer_api")

    # ── Total Goals ───────────────────────────────────────────────────────────
    if mkt == "Total":
        if line is None:
            return _unresolvable("missing line for total goals")
        if dirn == "Over":
            delta  = combined - line
            result = "WIN" if combined > line else ("PUSH" if combined == line else "LOSS")
        elif dirn == "Under":
            delta  = line - combined
            result = "WIN" if combined < line else ("PUSH" if combined == line else "LOSS")
        else:
            return _unresolvable("missing direction for total goals")
        return dict(leg_result=result, accuracy_delta=round(delta, 2),
                    actual_value=combined, resolution_source="soccer_api")

    # ── Moneyline (3-way) ─────────────────────────────────────────────────────
    if mkt == "Moneyline":
        if side is None:
            return _unresolvable("can't identify selected team for soccer moneyline")
        if side == "draw":
            result = "WIN" if h == a else "LOSS"
            delta  = 0.0
        elif side == "home":
            result = "WIN" if h > a else "LOSS"  # draw = LOSS for team ML
            delta  = h - a
        else:
            result = "WIN" if a > h else "LOSS"
            delta  = a - h
        return dict(leg_result=result, accuracy_delta=round(delta, 2),
                    actual_value=delta, resolution_source="soccer_api")

    # ── Spread (Asian handicap) ───────────────────────────────────────────────
    if mkt == "Spread":
        if side is None or line is None:
            return _unresolvable("missing side or line for soccer spread")
        if side == "home":
            delta = h + line - a
        elif side == "away":
            delta = a + line - h
        else:
            return _unresolvable("can't apply spread to draw selection")
        result = "WIN" if delta > 0 else ("PUSH" if delta == 0 else "LOSS")
        return dict(leg_result=result, accuracy_delta=round(delta, 2),
                    actual_value=round(delta, 2), resolution_source="soccer_api")

    return _unresolvable(f"soccer market '{mkt}' not resolvable")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BATCH RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

_HIST_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "historical.db"
)


def resolve_all_legs(
    db: Session,
    hist_db_path: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Resolve leg_result for all unresolved bet_legs in bets.db.

    Algorithm
    ---------
    For each settled bet:
      1. If SETTLED_WIN parlay: any leg still unresolved after DB lookup
         gets leg_result=WIN, resolution_source=inferred_parlay_win.
      2. For each leg: parse description → look up game → compute result.
      3. Pitcher strikeout props: try pitcher_game_logs.
      4. Other player props / Soccer player props: unresolvable.
      5. Unsupported sports (Tennis, MMA): sport_not_in_db.

    Parameters
    ----------
    overwrite : if True, re-resolve legs that already have a result set.

    Returns
    -------
    {
      resolved:           int,
      inferred_parlay:    int,
      skipped_no_sport:   int,
      already_resolved:   int,
      unresolvable:       int,
      errors:             int,
      by_sport:           dict,
    }
    """
    hist_path = hist_db_path or _HIST_DB_PATH
    hist_conn = sqlite3.connect(hist_path)
    cur = hist_conn.cursor()

    resolved = inferred_parlay = skipped = already_set = unresolvable_count = errors = 0
    by_sport: dict[str, int] = {}

    # Fetch all settled non-mock bets with legs
    settled_bets = (
        db.query(Bet)
        .filter(
            Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
            Bet.is_mock.is_(False),
        )
        .all()
    )

    for bet in settled_bets:
        game_date = (
            bet.time_placed.strftime("%Y-%m-%d") if bet.time_placed else None
        )
        if not game_date:
            continue

        bet_sport  = (bet.sports or "").split("|")[0].strip()
        hist_sport = infer_sport("", bet_sport)
        is_parlay_win = bet.status == "SETTLED_WIN" and (bet.legs or 1) > 1

        legs = (
            db.query(BetLeg)
            .filter(BetLeg.bet_id == bet.id)
            .order_by(BetLeg.leg_index)
            .all()
        )

        for leg in legs:
            # Skip already resolved unless overwrite
            if leg.leg_result is not None and not overwrite:
                already_set += 1
                continue

            desc = leg.description or ""
            if not desc:
                unresolvable_count += 1
                continue

            leg_sport = infer_sport(leg.sport or "", bet_sport)

            # Sport not in historical.db (Tennis, MMA, Soccer player props…)
            if leg_sport not in _RESOLVABLE_SPORTS:
                # Soccer totals/moneylines can still be resolved if game data exists
                if leg_sport == "Soccer" or (bet_sport.lower() == "soccer"):
                    # Attempt Soccer resolution below (games table has soccer scores)
                    pass
                else:
                    leg.resolution_source = "sport_not_in_db"
                    skipped += 1
                    continue

            try:
                parsed = parse_leg_details(desc)
            except Exception:
                errors += 1
                continue

            mkt = parsed["market_type"]

            # ── Trust stored market_type for Player Props ─────────────────────
            # The stored leg.market_type was set by the FanDuel importer which
            # had access to the "(Player Prop)" column in the original data.
            # If re-parsing the bare description text reclassifies it (e.g.
            # "Under 8.5 Assists" → Total), restore the correct label so we
            # don't accidentally route props through game-score resolution.
            stored_mkt = (leg.market_type or "").strip()
            if stored_mkt == "Player Prop" and mkt != "Player Prop":
                mkt = "Player Prop"
                parsed["market_type"] = "Player Prop"

            # ── Pitcher strikeout prop ────────────────────────────────────────
            if mkt == "Player Prop" and parsed.get("stat_type") == "strikeouts":
                player = parsed["selected_team_or_player"]
                k_total = _pitcher_strikeouts(cur, player, game_date)
                if k_total is not None and parsed["line"] is not None:
                    line = parsed["line"]
                    dirn = parsed["direction"] or "Over"
                    actual = float(k_total)
                    delta  = actual - line if dirn == "Over" else line - actual
                    if dirn == "Over":
                        result = "WIN" if actual > line else ("PUSH" if actual == line else "LOSS")
                    else:
                        result = "WIN" if actual < line else ("PUSH" if actual == line else "LOSS")
                    leg.leg_result         = result
                    leg.accuracy_delta     = round(delta, 2)
                    leg.actual_value       = actual
                    leg.resolution_source  = "pitcher_logs"
                    resolved += 1
                    by_sport[leg_sport or "MLB"] = by_sport.get(leg_sport or "MLB", 0) + 1
                    continue
                # Fall through to unresolvable/inferred if no pitcher log
                if is_parlay_win:
                    leg.leg_result        = "WIN"
                    leg.resolution_source = "inferred_parlay_win"
                    inferred_parlay += 1
                else:
                    leg.resolution_source = "unresolvable"
                    unresolvable_count += 1
                continue

            # ── Other player props — can't resolve ───────────────────────────
            if mkt == "Player Prop":
                if is_parlay_win:
                    leg.leg_result        = "WIN"
                    leg.resolution_source = "inferred_parlay_win"
                    inferred_parlay += 1
                else:
                    leg.resolution_source = "unresolvable"
                    unresolvable_count += 1
                continue

            # ── Persist soccer subtype to leg record ──────────────────────────
            if parsed.get("subtype") and not leg.subtype:
                leg.subtype = parsed["subtype"]

            # ── Soccer path: resolve via soccer_results cache ─────────────────
            if leg_sport == "Soccer" or bet_sport.lower() == "soccer":
                soccer_home = parsed.get("home_team") or ""
                soccer_away = parsed.get("away_team") or ""
                soccer_game = get_soccer_result(soccer_home, soccer_away, game_date, db)

                # Widen date window by ±1 if not found on exact date
                if soccer_game is None:
                    from datetime import timedelta
                    from database import SoccerResult as _SR
                    for delta_d in (-1, 1):
                        alt_date = (
                            datetime.strptime(game_date, "%Y-%m-%d") + timedelta(days=delta_d)
                        ).strftime("%Y-%m-%d")
                        soccer_game = get_soccer_result(soccer_home, soccer_away, alt_date, db)
                        if soccer_game:
                            break

                if soccer_game is None:
                    if is_parlay_win:
                        leg.leg_result        = "WIN"
                        leg.resolution_source = "inferred_parlay_win"
                        inferred_parlay += 1
                    else:
                        leg.resolution_source = "unresolvable_soccer_no_cache"
                        unresolvable_count += 1
                    continue

                outcome = resolve_soccer_leg(parsed, soccer_game)

                if outcome["leg_result"] is not None:
                    leg.leg_result        = outcome["leg_result"]
                    leg.accuracy_delta    = (
                        round(outcome["accuracy_delta"], 3)
                        if outcome["accuracy_delta"] is not None else None
                    )
                    leg.actual_value      = outcome["actual_value"]
                    leg.resolution_source = outcome["resolution_source"]
                    resolved += 1
                    by_sport["Soccer"] = by_sport.get("Soccer", 0) + 1
                elif is_parlay_win:
                    leg.leg_result        = "WIN"
                    leg.resolution_source = "inferred_parlay_win"
                    inferred_parlay += 1
                else:
                    leg.resolution_source = outcome["resolution_source"]
                    unresolvable_count += 1
                continue

            # ── Moneyline / Spread / Total — look up game ─────────────────────
            if not leg_sport:
                leg.resolution_source = "sport_not_in_db"
                skipped += 1
                continue

            away = to_db_team(normalize_team(parsed["away_team"]), leg_sport)
            home = to_db_team(normalize_team(parsed["home_team"]), leg_sport)

            game = _find_game(cur, leg_sport, home, away, game_date)

            if game is None:
                # Try alternate date (same-day settlement sometimes off by 1 day)
                game = _find_game(cur, leg_sport, home, away, game_date, window=2)

            if game is None:
                if is_parlay_win:
                    leg.leg_result        = "WIN"
                    leg.resolution_source = "inferred_parlay_win"
                    inferred_parlay += 1
                else:
                    leg.resolution_source = "unresolvable"
                    unresolvable_count += 1
                continue

            outcome = _resolve_from_game(parsed, game, leg_sport)

            if outcome["leg_result"] is not None:
                leg.leg_result        = outcome["leg_result"]
                leg.accuracy_delta    = (
                    round(outcome["accuracy_delta"], 3)
                    if outcome["accuracy_delta"] is not None else None
                )
                leg.actual_value      = outcome["actual_value"]
                leg.resolution_source = outcome["resolution_source"]
                resolved += 1
                by_sport[leg_sport] = by_sport.get(leg_sport, 0) + 1
            elif is_parlay_win:
                leg.leg_result        = "WIN"
                leg.resolution_source = "inferred_parlay_win"
                inferred_parlay += 1
            else:
                leg.resolution_source = outcome["resolution_source"]
                unresolvable_count += 1

    db.commit()
    hist_conn.close()

    return {
        "resolved":          resolved,
        "inferred_parlay":   inferred_parlay,
        "skipped_no_sport":  skipped,
        "already_resolved":  already_set,
        "unresolvable":      unresolvable_count,
        "errors":            errors,
        "total_processed":   resolved + inferred_parlay + skipped + already_set + unresolvable_count + errors,
        "by_sport":          by_sport,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INDIVIDUAL LEG RESOLUTION AGAINST GAME SCORES
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_leg_from_game(
    leg_id:      Optional[int],
    description: str,
    sport:       str,
    game_date:   str,
    db:          Session,
    cache:       bool = True,
    hist_db_path: Optional[str] = None,
) -> dict:
    """
    Resolve a single leg against historical.db game scores.

    Parses description to extract team/market/line/direction, queries
    historical.db for the matching game, and applies the same resolution
    logic used by settle_mock_bets().

    Args:
        leg_id:      mock_bet_legs.id (used for caching; None = skip dedup check)
        description: raw leg description string
        sport:       sport key as stored in mock_bets (MLB / NHL / NBA / etc.)
        game_date:   YYYY-MM-DD of the game
        db:          bets.db SQLAlchemy session
        cache:       if True, insert result into leg_historical_resolution
        hist_db_path: override for historical.db location

    Returns:
        {result: 'WIN'|'LOSS'|'PUSH'|None, margin: float|None, game_found: bool}
    """
    from datetime import datetime as _dt
    import json as _json

    hist_path = hist_db_path or _HIST_DB_PATH

    # ── Parse the description ─────────────────────────────���──────────────────
    parsed = parse_leg_details(description)
    team_raw    = parsed.get("selected_team_or_player") or ""
    market_type = parsed.get("market_type") or "Other"
    line        = parsed.get("line")
    direction   = parsed.get("direction") or ""

    if not team_raw or market_type == "Player Prop":
        return {"result": None, "margin": None, "game_found": False,
                "reason": "player_prop_or_no_team"}

    # ── Map sport to historical.db key ───────────────────────────────────────
    hist_sport = infer_sport(sport, sport)
    if not hist_sport or hist_sport == "Soccer":
        return {"result": None, "margin": None, "game_found": False,
                "reason": f"sport_not_in_db:{sport}"}

    canonical = normalize_team(team_raw)
    db_team   = to_db_team(canonical, hist_sport)

    # ── Query historical.db for the game ──────────────────────────────���──────
    try:
        conn = sqlite3.connect(hist_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT home_team, away_team, home_score, away_score, game_date
            FROM   games
            WHERE  sport  = ?
              AND  status = 'Final'
              AND  game_date = ?
              AND  (home_team = ? OR away_team = ?
                   OR home_team LIKE ? OR away_team LIKE ?)
            LIMIT 1
        """, (hist_sport, game_date, db_team, db_team,
              f"%{db_team}%", f"%{db_team}%")).fetchall()
        conn.close()
    except Exception as e:
        return {"result": None, "margin": None, "game_found": False,
                "reason": f"db_error:{e}"}

    if not rows:
        # Try with canonical name directly in case db_team mapping failed
        try:
            conn = sqlite3.connect(hist_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT home_team, away_team, home_score, away_score, game_date
                FROM   games
                WHERE  sport  = ?
                  AND  status = 'Final'
                  AND  game_date = ?
                  AND  (home_team LIKE ? OR away_team LIKE ?)
                LIMIT 1
            """, (hist_sport, game_date,
                  f"%{canonical.split()[-1]}%",
                  f"%{canonical.split()[-1]}%")).fetchall()
            conn.close()
        except Exception:
            rows = []

    if not rows:
        return {"result": None, "margin": None, "game_found": False,
                "reason": "game_not_found_in_historical_db"}

    game = tuple(rows[0])   # (home_team, away_team, home_score, away_score, game_date)

    # ── Apply resolution logic ────────────────────────────────────────────────
    outcome = _resolve_from_game(parsed, game, hist_sport)

    result = outcome.get("leg_result")
    margin = outcome.get("accuracy_delta")

    # ── Cache into leg_historical_resolution ─────────────────────────────────
    if cache and result in ("WIN", "LOSS", "PUSH"):
        try:
            from database import LegHistoricalResolution
            from sqlalchemy import text as _sqla_text

            # Dedup: skip if already cached for this leg_id
            if leg_id is not None:
                existing = db.execute(
                    _sqla_text("SELECT id FROM leg_historical_resolution WHERE bet_leg_id = :lid LIMIT 1"),
                    {"lid": str(leg_id)}
                ).fetchone()
                if existing:
                    return {"result": result, "margin": margin, "game_found": True,
                            "cached": False, "reason": "already_cached"}

            row = LegHistoricalResolution(
                bet_leg_id        = str(leg_id) if leg_id is not None else None,
                game_date         = game_date,
                sport             = hist_sport,
                team              = canonical,
                market_type       = market_type,
                line              = line,
                result            = result,
                margin            = round(margin, 3) if margin is not None else None,
                resolution_source = "historical_db",
                resolved_at       = _dt.utcnow().isoformat(),
            )
            db.add(row)
            db.commit()
        except Exception:
            db.rollback()

    return {"result": result, "margin": margin, "game_found": True}
