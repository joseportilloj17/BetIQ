"""
backfill_player_props.py — Resolve Player Prop bet legs using ESPN boxscore data.

For each unresolved Player Prop leg on a settled (non-mock) bet:
  1. Parse player name, stat type, threshold, direction from description
  2. Find the ESPN game event via scoreboard (team-match + ±7 day window)
  3. Fetch boxscore → build {player_name: {stat: value}} map
  4. Match player by name, read stat, compare to threshold → WIN / LOSS
  5. Write leg_result, actual_value, accuracy_delta back to bet_legs

Supported sports: NBA/Basketball, NFL/American Football, MLB/Baseball, NHL/Ice Hockey
"""
from __future__ import annotations

import re
import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

log = logging.getLogger(__name__)

# ── ESPN URLs ─────────────────────────────────────────────────────────────────
_ESPN = {
    "NBA":  "https://site.api.espn.com/apis/site/v2/sports/basketball/nba",
    "NFL":  "https://site.api.espn.com/apis/site/v2/sports/football/nfl",
    "MLB":  "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb",
    "NHL":  "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl",
}

_SPORT_MAP = {
    "basketball": "NBA", "nba": "NBA",
    "american football": "NFL", "nfl": "NFL",
    "baseball": "MLB", "major league baseball": "MLB", "mlb": "MLB",
    "ice hockey": "NHL", "nhl": "NHL",
}

# ── Stat type detection ───────────────────────────────────────────────────────
# Maps keyword patterns in description → (espn_category, espn_key, is_combined)
# is_combined: list of keys to sum (e.g. passing+rushing yards)
_STAT_PATTERNS: list[tuple[re.Pattern, str, str, list[str]]] = [
    # NBA
    (re.compile(r'\bpoints?\b|\bto score\b|\bpts\b',      re.I), "NBA",  "points",                 []),
    (re.compile(r'\brebounds?\b|\brebs?\b',               re.I), "NBA",  "rebounds",               []),
    (re.compile(r'\bassists?\b|\bast\b',                  re.I), "NBA",  "assists",                []),
    (re.compile(r'\bsteals?\b',                           re.I), "NBA",  "steals",                 []),
    (re.compile(r'\bblocks?\b',                           re.I), "NBA",  "blocks",                 []),
    (re.compile(r'\bthrees?\b|\b3pm\b|\b3-point',         re.I), "NBA",  "threes",                 []),
    # NFL — combined first so they match before single-stat patterns
    (re.compile(r'passing\s*\+\s*rushing\s*yds?',         re.I), "NFL",  "passing+rushing",        ["passingYards", "rushingYards"]),
    (re.compile(r'\bpassing\s+yds?\b|\bpassing\s+yards?\b', re.I), "NFL","passingYards",           []),
    (re.compile(r'\brushing\s+yds?\b|\brushing\s+yards?\b', re.I), "NFL","rushingYards",           []),
    (re.compile(r'\breceiving\s+yds?\b|\breceiving\s+yards?\b', re.I), "NFL", "receivingYards",    []),
    (re.compile(r'\breceptions?\b|\brec\b',               re.I), "NFL",  "receptions",             []),
    (re.compile(r'\btouchdowns?\b|\btds?\b',              re.I), "NFL",  "touchdowns",             []),
    # Generic yards (after specific patterns) — infer from context
    (re.compile(r'\byards?\b|\byds?\b',                   re.I), "NFL",  "yards_generic",          []),
    # MLB
    (re.compile(r'\bhome\s+run\b|\bhr\b',                 re.I), "MLB",  "homeRuns",               []),
    (re.compile(r'\bhits?\b',                             re.I), "MLB",  "hits",                   []),
    (re.compile(r'\brbi\b|\bruns?\s+batted\b',            re.I), "MLB",  "RBI",                    []),
    (re.compile(r'\btotal\s+bases?\b',                    re.I), "MLB",  "totalBases",             []),
    (re.compile(r'\bstrikeouts?\b|\bk\'?s\b',             re.I), "MLB",  "strikeouts",             []),
    # NHL
    (re.compile(r'\bshots?\s+on\s+goal\b|\bsog\b',        re.I), "NHL",  "shots",                  []),
    (re.compile(r'\bgoals?\b',                            re.I), "NHL",  "goals",                  []),
    (re.compile(r'\bassists?\b',                          re.I), "NHL",  "assists",                []),
    (re.compile(r'\bpoints?\b',                           re.I), "NHL",  "points",                 []),
]

# ESPN boxscore key lookup per sport+stat
# Format: (category_name, key_index_or_name)
# key_index: position in keys[] list; key_name: match substring in keys[]
_ESPN_STAT_LOOKUP: dict[str, dict[str, tuple[str, str]]] = {
    "NBA": {
        "points":   ("",         "points"),
        "rebounds": ("",         "rebounds"),
        "assists":  ("",         "assists"),
        "steals":   ("",         "steals"),
        "blocks":   ("",         "blocks"),
        "threes":   ("",         "threePointFieldGoalsMade"),  # split key like "6-12" → take [0]
    },
    "NFL": {
        "passingYards":   ("passing",   "passingYards"),
        "rushingYards":   ("rushing",   "rushingYards"),
        "receivingYards": ("receiving", "receivingYards"),
        "receptions":     ("receiving", "receptions"),
        "touchdowns":     ("",          "Touchdowns"),   # partial match across categories
        "passingTouchdowns": ("passing","passingTouchdowns"),
        "rushingTouchdowns": ("rushing","rushingTouchdowns"),
        "receivingTouchdowns": ("receiving","receivingTouchdowns"),
        "passingYards":   ("passing",   "passingYards"),
        "rushingYards":   ("rushing",   "rushingYards"),
    },
    "MLB": {
        "hits":       ("batting", "hits"),
        "homeRuns":   ("batting", "homeRuns"),
        "RBI":        ("batting", "RBI"),
        "totalBases": ("batting", "totalBases"),
        "strikeouts": ("pitching", "strikeouts"),
    },
    "NHL": {
        "shots":   ("", "shots"),
        "goals":   ("", "goals"),
        "assists": ("", "assists"),
        "points":  ("", "points"),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTION PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


_NON_TEAM_WORDS = frozenset({
    # stat / market words that can appear near team names but aren't part of them
    "over", "under", "alt", "alternate", "total", "points", "yards", "yds",
    "pts", "rushing", "passing", "receiving", "hits", "runs", "goals",
    "shots", "moneyline", "spread", "line", "puck", "run", "to", "score",
    "or", "more", "and", "the", "at", "of", "in", "on", "a",
})


def _extract_matchup_teams(desc: str) -> tuple[str, str]:
    """Extract away_team, home_team from description containing ' @ ' or ' v '."""
    for sep in (" @ ", " v ", " vs "):
        if sep in desc:
            parts = desc.split(sep)
            home_raw = parts[-1].strip()
            home_raw = re.sub(r'\s*[+-]\d{2,4}\s*$', '', home_raw).strip()
            home_raw = re.sub(r'\s*\([^)]*\)', '', home_raw).strip()

            # Extract away team: take consecutive Title-cased, non-stat words
            # from the rightmost end of the segment before the separator.
            before_sep = parts[-2].strip()
            before_sep = re.sub(r'\s*\([^)]*\)', '', before_sep).strip()
            words = before_sep.split()
            away_words = []
            for w in reversed(words):
                # Stop at: numbers, market keywords, punctuation-only tokens
                if re.match(r'^[+-]?\d', w):
                    break
                if w.lower() in _NON_TEAM_WORDS:
                    break
                if not re.match(r'^[A-Z]', w) and w not in ("'s",):
                    break
                away_words.insert(0, w)
                if len(away_words) >= 4:   # max 4-word team name
                    break

            away_candidate = " ".join(away_words).strip()
            return away_candidate, home_raw
    return "", ""


def parse_prop_description(desc: str, sport_key: str) -> dict:
    """
    Parse a player prop leg description into structured fields.

    Returns:
      player_name: str
      stat_type: str         — ESPN stat key (e.g. "points", "receivingYards")
      stat_keys: list[str]   — ESPN key(s) to sum (for combined stats)
      threshold: float
      direction: "over" | "under"
      away_team: str
      home_team: str
    """
    result = {
        "player_name": None,
        "stat_type":   None,
        "stat_keys":   [],
        "threshold":   None,
        "direction":   "over",   # default — props are almost always over
        "away_team":   "",
        "home_team":   "",
    }

    away, home = _extract_matchup_teams(desc)
    result["away_team"] = away
    result["home_team"] = home

    # ── Build prop_part: strip matchup suffix ────────────────────────────────
    # Find the separator and trim everything from the away team onward.
    # Use the away_team's first word to locate it rather than double-stripping,
    # so we don't accidentally eat stat keywords (e.g. "Rushing Yds").
    prop_part = desc
    for sep in (" @ ", " v ", " vs "):
        if sep in desc:
            idx = desc.rfind(sep)
            before_sep = desc[:idx]
            # Try to strip the away team from the end of before_sep
            stripped = before_sep
            if away:
                first_word = re.escape(away.split()[0])
                # rfind the away team's first word and cut there
                m = re.search(rf'\s+{first_word}\b', before_sep, re.I)
                if m:
                    stripped = before_sep[:m.start()].strip()
            prop_part = _norm(stripped)
            break

    # Clean FanDuel format parens/odds/separators
    prop_part = re.sub(r'^\(|\)$', '', prop_part.strip())
    prop_part = re.sub(r'\s*[+-]\d{2,4}\s*$', '', prop_part).strip()
    prop_part = re.sub(r'\s*[—–]\s*', ' ', prop_part).strip()

    # Direction detection
    if re.search(r'\bunder\b', prop_part, re.I):
        result["direction"] = "under"

    # ── Threshold: first "N+" or "over N.N" or "N or more" ──────────────────
    thr_m = re.search(r'(?:over\s+)?(\d+\.?\d*)\s*\+?(?:\s+or\s+more)?',
                      prop_part, re.I)
    if thr_m:
        result["threshold"] = float(thr_m.group(1))

    # ── Stat type: scan full desc (preserves "Passing + Rushing Yds" before
    # the matchup strips it from prop_part) ──────────────────────────────────
    for pattern, sport_scope, stat_key, combined_keys in _STAT_PATTERNS:
        if sport_scope and sport_scope != sport_key:
            continue
        if pattern.search(desc):
            result["stat_type"]  = stat_key
            result["stat_keys"]  = combined_keys or [stat_key]
            break

    # Infer generic yards subtype from surrounding context
    if result["stat_type"] == "yards_generic":
        lower = prop_part.lower()
        if "receiv" in lower or "alt rec" in lower:
            result["stat_type"] = "receivingYards"
            result["stat_keys"] = ["receivingYards"]
        elif "rush" in lower or "alt rush" in lower:
            result["stat_type"] = "rushingYards"
            result["stat_keys"] = ["rushingYards"]
        elif "pass" in lower or "alt pass" in lower:
            result["stat_type"] = "passingYards"
            result["stat_keys"] = ["passingYards"]
        else:
            result["stat_type"] = "receivingYards"  # most common yard prop
            result["stat_keys"] = ["receivingYards"]

    # ── Player name: everything before the first digit (threshold boundary) ─
    # e.g. "A.J. Brown 50+ Yards ..." → stop at "50", name = "A.J. Brown"
    #      "Karl-Anthony Towns To Score 15+ Points" → stop at "15", name = "Karl-Anthony Towns To Score"
    digit_m = re.search(r'\d', prop_part)
    if digit_m and digit_m.start() > 0:
        raw_name = prop_part[:digit_m.start()].strip()
    else:
        raw_name = prop_part

    # Strip trailing market keywords from name candidate
    _NAME_TRAIL_RE = re.compile(
        r'\s*(to\s+score|to\s+record|to\s+have|to\s+hit|to\s+get|'
        r'anytime|scorer)\s*$', re.I
    )
    raw_name = _NAME_TRAIL_RE.sub('', raw_name).strip()
    raw_name = re.sub(r'[\(\)\-—–]+$', '', raw_name).strip()

    if raw_name and len(raw_name.split()) <= 6:
        result["player_name"] = raw_name

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ESPN SCOREBOARD + BOXSCORE
# ═══════════════════════════════════════════════════════════════════════════════

_scoreboard_cache: dict[tuple, list] = {}   # (sport, date) → list of {id, home, away}
_boxscore_cache:   dict[str, dict]   = {}   # event_id → {player_name_lower: {stat: val}}


def _get(url: str, params: dict | None = None) -> dict | None:
    from http_retry import get_with_retry
    r = get_with_retry(url, params=params, timeout=15, max_attempts=3,
                       label="PropBackfill")
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _fetch_scoreboard(sport_key: str, date_str: str) -> list[dict]:
    """Return list of {event_id, home_team, away_team} for a sport+date."""
    cache_key = (sport_key, date_str)
    if cache_key in _scoreboard_cache:
        return _scoreboard_cache[cache_key]

    base = _ESPN.get(sport_key)
    if not base:
        _scoreboard_cache[cache_key] = []
        return []

    date_esc = date_str.replace("-", "")
    data = _get(f"{base}/scoreboard", params={"dates": date_esc, "limit": 50})
    events = []
    if data:
        for ev in data.get("events", []):
            comps = ev.get("competitions", [{}])
            if not comps:
                continue
            comp = comps[0]
            teams = {}
            for t in comp.get("competitors", []):
                hoa = t.get("homeAway", "")
                name = (t.get("team") or {}).get("displayName", "") or \
                       (t.get("team") or {}).get("shortDisplayName", "")
                teams[hoa] = name
            events.append({
                "event_id":  ev.get("id", ""),
                "home_team": teams.get("home", ""),
                "away_team": teams.get("away", ""),
                "name":      ev.get("name", ""),
            })

    _scoreboard_cache[cache_key] = events
    time.sleep(0.15)
    return events


def _team_matches(a: str, b: str) -> bool:
    """Fuzzy team name match — last word, or any word overlap."""
    if not a or not b:
        return False
    a_l, b_l = a.lower().strip(), b.lower().strip()
    if a_l == b_l:
        return True
    a_words = set(a_l.split())
    b_words = set(b_l.split())
    # Ignore very common words
    _STOP = {"at", "the", "of", "fc", "sc", "city", "united"}
    a_sig = a_words - _STOP
    b_sig = b_words - _STOP
    if not a_sig or not b_sig:
        return False
    return bool(a_sig & b_sig)


def find_espn_event(
    sport_key: str,
    away_team: str,
    home_team: str,
    game_date: str,
    window_days: int = 7,
) -> str | None:
    """
    Search ESPN scoreboard ±window_days around game_date for a game matching
    the given teams. Returns the ESPN event_id or None.
    """
    try:
        base_dt = datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        return None

    for delta in range(0, window_days + 1):
        for sign in ([0] if delta == 0 else [1, -1]):
            search_dt = base_dt + timedelta(days=delta * sign)
            search_date = search_dt.strftime("%Y-%m-%d")
            events = _fetch_scoreboard(sport_key, search_date)
            for ev in events:
                h_match = _team_matches(home_team, ev["home_team"])
                a_match = _team_matches(away_team, ev["away_team"])
                # Also try swapped (some data sources swap home/away)
                h2 = _team_matches(home_team, ev["away_team"])
                a2 = _team_matches(away_team, ev["home_team"])
                if (h_match and a_match) or (h2 and a2):
                    return ev["event_id"]
                # Single-team fallback if we couldn't parse both teams
                if not away_team and h_match:
                    return ev["event_id"]
                if not home_team and a_match:
                    return ev["event_id"]
    return None


def fetch_boxscore(sport_key: str, event_id: str) -> dict[str, dict]:
    """
    Fetch ESPN boxscore for an event.
    Returns {player_name_lower: {stat_key: float_value}}.
    """
    if event_id in _boxscore_cache:
        return _boxscore_cache[event_id]

    base = _ESPN.get(sport_key, "")
    data = _get(f"{base}/summary", params={"event": event_id})
    players: dict[str, dict] = {}

    if not data:
        _boxscore_cache[event_id] = players
        return players

    boxscore = data.get("boxscore", {})
    for team_entry in boxscore.get("players", []):
        for stat_group in team_entry.get("statistics", []):
            category   = stat_group.get("name", "")
            keys_raw   = stat_group.get("keys", [])
            # Build index: key_name → column_index
            key_idx: dict[str, int] = {}
            for i, k in enumerate(keys_raw):
                # Some keys are "made-attempts" — index both the full key and first part
                key_idx[k.lower()] = i
                base_k = k.split("-")[0].split("/")[0]
                key_idx[base_k.lower()] = i

            for athlete_entry in stat_group.get("athletes", []):
                ath   = athlete_entry.get("athlete", {})
                name  = (ath.get("displayName") or ath.get("fullName", "?")).lower()
                stats_raw = athlete_entry.get("stats", [])

                if name not in players:
                    players[name] = {}

                for stat_name, col_idx in key_idx.items():
                    if col_idx < len(stats_raw):
                        raw_val = stats_raw[col_idx]
                        # Handle "15-22" or "3/4" style — take numerator
                        if isinstance(raw_val, str) and ("-" in raw_val or "/" in raw_val):
                            raw_val = re.split(r'[-/]', raw_val)[0]
                        try:
                            val = float(raw_val)
                            # Accumulate across categories (same stat may appear in multiple)
                            players[name][stat_name] = players[name].get(stat_name, 0) + val
                        except (ValueError, TypeError):
                            pass

    _boxscore_cache[event_id] = players
    time.sleep(0.15)
    return players


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLAYER NAME MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_name(name: str) -> str:
    """Normalise for comparison: lowercase, strip punctuation."""
    return re.sub(r'[^a-z\s]', '', name.lower()).strip()


def match_player(target: str, player_stats: dict[str, dict]) -> dict | None:
    """
    Find target player in player_stats dict (keyed by lowercase display name).
    Returns the stat dict or None.
    """
    if not target:
        return None

    t_norm = _norm_name(target)
    t_words = t_norm.split()

    # 1. Exact normalised match
    for name, stats in player_stats.items():
        if _norm_name(name) == t_norm:
            return stats

    # 2. Last name match (if last name is distinctive enough)
    if t_words:
        last = t_words[-1]
        if len(last) >= 4:
            candidates = [
                (name, stats) for name, stats in player_stats.items()
                if _norm_name(name).split() and _norm_name(name).split()[-1] == last
            ]
            if len(candidates) == 1:
                return candidates[0][1]

    # 3. All tokens present in name (for "Karl-Anthony Towns" → "karl anthony towns")
    if len(t_words) >= 2:
        for name, stats in player_stats.items():
            n_norm = _norm_name(name)
            if all(w in n_norm for w in t_words):
                return stats

    # 4. First + last token
    if len(t_words) >= 2:
        first, last = t_words[0], t_words[-1]
        for name, stats in player_stats.items():
            n_words = _norm_name(name).split()
            if n_words and n_words[0] == first and n_words[-1] == last:
                return stats

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STAT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_stat_value(
    stat_type: str,
    stat_keys: list[str],
    player_stat_dict: dict,
    sport_key: str,
) -> float | None:
    """
    Pull the relevant stat value(s) from a player's stat dict.
    For combined stats (passing+rushing), sums the components.
    """
    if not player_stat_dict:
        return None

    if stat_type == "passing+rushing":
        py = player_stat_dict.get("passingyards", player_stat_dict.get("yards", 0))
        ry = player_stat_dict.get("rushingyards", 0)
        # Try alternate key names
        if not py:
            for k, v in player_stat_dict.items():
                if "passing" in k and "yard" in k:
                    py = v; break
        if not ry:
            for k, v in player_stat_dict.items():
                if "rushing" in k and "yard" in k:
                    ry = v; break
        return (py or 0) + (ry or 0) or None

    # Try each key in stat_keys list
    for sk in (stat_keys or [stat_type]):
        sk_lower = sk.lower()
        # Direct key lookup
        val = player_stat_dict.get(sk_lower)
        if val is not None:
            return float(val)
        # Partial match
        for k, v in player_stat_dict.items():
            if sk_lower in k:
                return float(v)

    # Sport-specific fallbacks
    if sport_key == "NBA":
        _fallbacks = {
            "points":   ["points"],
            "rebounds": ["rebounds", "totalrebounds"],
            "assists":  ["assists"],
            "steals":   ["steals"],
            "blocks":   ["blocks"],
            "threes":   ["threepointfieldgoalsmade", "3pm"],
        }
        for fk in _fallbacks.get(stat_type, []):
            val = player_stat_dict.get(fk)
            if val is not None:
                return float(val)

    if sport_key == "NFL":
        _fallbacks = {
            "receivingyards": ["receivingyards", "yards"],
            "rushingyards":   ["rushingyards"],
            "passingyards":   ["passingyards"],
            "receptions":     ["receptions"],
        }
        for fk in _fallbacks.get(stat_type.lower(), [stat_type.lower()]):
            val = player_stat_dict.get(fk)
            if val is not None:
                return float(val)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN BACKFILL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_backfill(db, dry_run: bool = False) -> dict:
    """
    Resolve all unresolved Player Prop legs in bet_legs for settled non-mock bets.

    Returns summary dict: {resolved, no_parse, no_event, no_player, no_stat, errors}
    """
    from database import Bet, BetLeg
    from sqlalchemy import and_

    _SUPPORTED_SPORTS = {
        "basketball", "nba", "american football", "nfl",
        "baseball", "major league baseball", "mlb",
        "ice hockey", "nhl",
    }

    # Fetch all unresolved player prop legs
    legs = (
        db.query(BetLeg)
        .join(Bet, BetLeg.bet_id == Bet.id)
        .filter(
            Bet.is_mock.is_(False),
            Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
            BetLeg.leg_result.is_(None),
            BetLeg.market_type == "Player Prop",
        )
        .all()
    )

    # Build a bet lookup
    bet_ids = {leg.bet_id for leg in legs}
    bets = {b.id: b for b in db.query(Bet).filter(Bet.id.in_(bet_ids)).all()}

    stats = {
        "total":     len(legs),
        "resolved":  0,
        "no_parse":  0,
        "no_event":  0,
        "no_player": 0,
        "no_stat":   0,
        "errors":    0,
        "by_sport":  {},
    }

    for leg in legs:
        try:
            bet = bets.get(leg.bet_id)
            if not bet:
                stats["errors"] += 1
                continue

            # Normalise sport
            raw_sport = (leg.sport or bet.sports or "").split("|")[0].strip().lower()
            sport_key = _SPORT_MAP.get(raw_sport)
            if not sport_key:
                continue   # unsupported sport (Tennis, Soccer player props, etc.)

            # Game date from time_placed
            if not bet.time_placed:
                stats["no_parse"] += 1
                continue
            game_date = bet.time_placed.strftime("%Y-%m-%d")

            # Parse description
            parsed = parse_prop_description(leg.description or "", sport_key)
            if not parsed["player_name"] or parsed["threshold"] is None or not parsed["stat_type"]:
                log.debug(f"[backfill] leg {leg.id}: bad parse {parsed}")
                stats["no_parse"] += 1
                leg.resolution_source = "prop_parse_failed"
                continue

            # Find ESPN event
            event_id = find_espn_event(
                sport_key,
                parsed["away_team"],
                parsed["home_team"],
                game_date,
                window_days=7,
            )
            if not event_id:
                log.debug(f"[backfill] leg {leg.id}: no event for {parsed['away_team']} @ {parsed['home_team']} ~{game_date}")
                stats["no_event"] += 1
                leg.resolution_source = "prop_game_not_found"
                continue

            # Fetch boxscore
            player_stats = fetch_boxscore(sport_key, event_id)
            if not player_stats:
                stats["no_event"] += 1
                leg.resolution_source = "prop_boxscore_empty"
                continue

            # Match player
            pstats = match_player(parsed["player_name"], player_stats)
            if pstats is None:
                log.debug(f"[backfill] leg {leg.id}: player '{parsed['player_name']}' not in boxscore (event {event_id})")
                stats["no_player"] += 1
                leg.resolution_source = "prop_player_not_found"
                continue

            # Extract stat value
            actual = extract_stat_value(
                parsed["stat_type"],
                parsed["stat_keys"],
                pstats,
                sport_key,
            )
            if actual is None:
                log.debug(f"[backfill] leg {leg.id}: stat '{parsed['stat_type']}' not found for {parsed['player_name']}")
                stats["no_stat"] += 1
                leg.resolution_source = "prop_stat_not_found"
                continue

            # Resolve
            threshold  = parsed["threshold"]
            direction  = parsed["direction"]
            if direction == "over":
                result  = "WIN" if actual > threshold else "LOSS"
                delta   = actual - threshold
            else:
                result  = "WIN" if actual < threshold else "LOSS"
                delta   = threshold - actual

            if not dry_run:
                leg.leg_result        = result
                leg.actual_value      = actual
                leg.accuracy_delta    = round(delta, 2)
                leg.resolution_source = "espn_boxscore"

            stats["resolved"] += 1
            stats["by_sport"][sport_key] = stats["by_sport"].get(sport_key, 0) + 1
            log.info(
                f"[backfill] leg {leg.id} {parsed['player_name']} "
                f"{direction} {threshold} {parsed['stat_type']} → actual={actual} → {result}"
            )

        except Exception as e:
            log.warning(f"[backfill] leg {leg.id} error: {e}", exc_info=True)
            stats["errors"] += 1

    if not dry_run:
        db.commit()

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Backfill player prop leg results via ESPN")
    parser.add_argument("--dry-run", action="store_true", help="Parse and match but don't write to DB")
    args = parser.parse_args()

    from database import SessionLocal
    db = SessionLocal()
    try:
        result = run_backfill(db, dry_run=args.dry_run)
        print("\n=== Player Prop Backfill Result ===")
        print(f"  Total legs:     {result['total']}")
        print(f"  Resolved:       {result['resolved']}")
        print(f"  No parse:       {result['no_parse']}")
        print(f"  No ESPN event:  {result['no_event']}")
        print(f"  Player missing: {result['no_player']}")
        print(f"  Stat missing:   {result['no_stat']}")
        print(f"  Errors:         {result['errors']}")
        print(f"  By sport:       {result['by_sport']}")
    finally:
        db.close()
