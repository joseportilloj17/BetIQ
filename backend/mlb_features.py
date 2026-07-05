"""
mlb_features.py — MLB-specific model features: park factors, pitcher quality, weather.

All functions return dicts of feature values keyed by the FEATURE_COLS names.
Callers (mock_bets.py, parlay_builder.py) merge these into the feature_dict
before passing to _prep_row().
"""
from __future__ import annotations

import math
import time
import logging
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# ─── Park Factors ────────────────────────────────────────────────────────────
# Source: FanGraphs / ESPN Park Factors (2024-2025 average).
# (runs_factor, hr_factor) — 1.0 = neutral. >1.0 = hitter-friendly.

PARK_FACTORS: dict[str, tuple[float, float]] = {
    # Team name fragment → (runs, hr)
    "rockies":        (1.38, 1.42),   # Coors Field
    "rangers":        (1.08, 1.15),   # Globe Life Field
    "red sox":        (1.05, 1.10),   # Fenway Park
    "reds":           (1.06, 1.12),   # Great American Ball Park
    "diamondbacks":   (1.04, 1.08),   # Chase Field
    "cubs":           (1.03, 1.05),   # Wrigley Field
    "phillies":       (1.02, 1.04),   # Citizens Bank Park
    "braves":         (1.01, 1.03),   # Truist Park
    "yankees":        (1.01, 1.06),   # Yankee Stadium
    "angels":         (1.00, 1.00),   # Angel Stadium
    "white sox":      (1.00, 1.02),   # Guaranteed Rate Field
    "tigers":         (0.99, 0.98),   # Comerica Park
    "cardinals":      (0.99, 0.99),   # Busch Stadium
    "twins":          (0.99, 1.01),   # Target Field
    "brewers":        (0.98, 0.97),   # American Family Field
    "astros":         (0.98, 0.99),   # Minute Maid Park
    "nationals":      (0.97, 0.98),   # Nationals Park
    "blue jays":      (0.97, 0.99),   # Rogers Centre
    "royals":         (0.97, 0.96),   # Kauffman Stadium
    "orioles":        (0.96, 0.99),   # Camden Yards
    "mets":           (0.96, 0.94),   # Citi Field
    "guardians":      (0.95, 0.93),   # Progressive Field
    "pirates":        (0.94, 0.90),   # PNC Park
    "rays":           (0.94, 0.92),   # Tropicana Field
    "dodgers":        (0.93, 0.92),   # Dodger Stadium
    "marlins":        (0.93, 0.88),   # loanDepot Park
    "mariners":       (0.92, 0.90),   # T-Mobile Park
    "padres":         (0.92, 0.88),   # Petco Park
    "athletics":      (0.91, 0.87),   # Sutter Health Park
    "giants":         (0.85, 0.78),   # Oracle Park
}


def _match_team_to_park(team_name: str) -> str | None:
    """Find the park factor key matching a team name."""
    lower = team_name.lower()
    for key in PARK_FACTORS:
        if key in lower:
            return key
    return None


def get_park_features(home_team: str, sport: str) -> dict[str, float]:
    """
    Return park factor features for a game.
    Non-MLB sports default to 1.0 (neutral).
    """
    if sport.upper() != "MLB":
        return {"park_runs_factor": 1.0, "park_hr_factor": 1.0}
    key = _match_team_to_park(home_team)
    if key is None:
        return {"park_runs_factor": 1.0, "park_hr_factor": 1.0}
    runs, hr = PARK_FACTORS[key]
    return {"park_runs_factor": runs, "park_hr_factor": hr}


# ─── Starting Pitcher Quality ───────────────────────────────────────────────
# Free MLB Stats API — no auth required.
# Endpoint: https://statsapi.mlb.com/api/v1/people/{id}/stats?stats=season&group=pitching

_PITCHER_CACHE: dict[int, dict] = {}   # pitcher_id → stats dict
_PITCHER_CACHE_TTL = 6 * 3600          # 6 hours

import requests


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float, returning default for non-numeric strings like '.---'."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _fetch_pitcher_season_stats(pitcher_id: int) -> dict | None:
    """Fetch season pitching stats from MLB Stats API. Returns stat dict or None."""
    now = time.time()
    cached = _PITCHER_CACHE.get(pitcher_id)
    if cached and now - cached.get("_fetched_at", 0) < _PITCHER_CACHE_TTL:
        return cached

    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
    try:
        r = requests.get(url, params={"stats": "season", "group": "pitching"},
                         timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.debug("pitcher stats fetch failed for %d: %s", pitcher_id, exc)
        return None

    splits = (data.get("stats") or [{}])[0].get("splits", [])
    if not splits:
        return None

    s = splits[0].get("stat", {})
    result = {
        "era":             _safe_float(s.get("era", 0)),
        "whip":            _safe_float(s.get("whip", 0)),
        "k_per_9":         _safe_float(s.get("strikeoutsPer9Inn", 0)),
        "hr_per_9":        _safe_float(s.get("homeRunsPer9", 0)),
        "innings_pitched": 0.0,
        "win_pct":         _safe_float(s.get("winPercentage", 0)),
        "_fetched_at":     now,
    }
    # Parse innings pitched properly: "95.1" means 95 and 1/3 innings
    try:
        ip_str = str(s.get("inningsPitched", "0"))
        if "." in ip_str:
            whole, frac = ip_str.split(".")
            result["innings_pitched"] = int(whole) + int(frac) / 3.0
        else:
            result["innings_pitched"] = float(ip_str)
    except (ValueError, TypeError):
        result["innings_pitched"] = 0.0

    _PITCHER_CACHE[pitcher_id] = result
    return result


_PROBABLE_PITCHERS_CACHE: dict[str, dict] = {}  # date → {(home,away) → (h_id, a_id)}
_PROBABLE_PITCHERS_CACHE_TIME: float = 0


def _load_probable_pitchers(game_date: str) -> dict[tuple[str, str], tuple[int | None, int | None]]:
    """
    Fetch probable pitchers for all MLB games on a date from the schedule API.
    Returns {(home_team_lower, away_team_lower): (home_pitcher_id, away_pitcher_id)}.
    Caches for 6 hours.
    """
    global _PROBABLE_PITCHERS_CACHE_TIME
    now = time.time()
    if game_date in _PROBABLE_PITCHERS_CACHE and now - _PROBABLE_PITCHERS_CACHE_TIME < _PITCHER_CACHE_TTL:
        return _PROBABLE_PITCHERS_CACHE[game_date]

    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": "1", "date": game_date, "hydrate": "probablePitcher"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.debug("probable pitchers fetch failed for %s: %s", game_date, exc)
        return {}

    result: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    for date_obj in data.get("dates", []):
        for game in date_obj.get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_name = (home.get("team", {}).get("name") or "").lower().strip()
            away_name = (away.get("team", {}).get("name") or "").lower().strip()
            hp = home.get("probablePitcher", {})
            ap = away.get("probablePitcher", {})
            result[(home_name, away_name)] = (
                hp.get("id"), ap.get("id"),
            )
    _PROBABLE_PITCHERS_CACHE[game_date] = result
    _PROBABLE_PITCHERS_CACHE_TIME = now
    return result


def _get_probable_pitchers(home_team: str, away_team: str,
                           game_date: str) -> tuple[int | None, int | None]:
    """
    Get probable pitcher IDs for a matchup on a date.
    Returns (home_pitcher_id, away_pitcher_id).
    """
    lookup = _load_probable_pitchers(game_date)
    if not lookup:
        return (None, None)
    # Try exact match first, then substring
    ht = home_team.lower().strip()
    at = away_team.lower().strip()
    if (ht, at) in lookup:
        return lookup[(ht, at)]
    # Fuzzy: last word match
    for (h, a), ids in lookup.items():
        if ht.split()[-1] == h.split()[-1] and at.split()[-1] == a.split()[-1]:
            return ids
    return (None, None)


def get_pitcher_features(home_team: str, away_team: str, game_date: str,
                         sport: str) -> dict[str, float]:
    """
    Return starting pitcher quality features for both starters.
    Non-MLB or missing data returns NaN (imputer fills median).
    """
    nan = float("nan")
    defaults = {
        "home_starter_era":   nan, "home_starter_whip":  nan,
        "home_starter_k9":    nan, "home_starter_hr9":   nan,
        "away_starter_era":   nan, "away_starter_whip":  nan,
        "away_starter_k9":    nan, "away_starter_hr9":   nan,
        "starter_era_diff":   nan, "starter_whip_diff":  nan,
        "starter_k9_diff":    nan,
    }
    if sport.upper() != "MLB":
        return defaults

    home_pid, away_pid = _get_probable_pitchers(home_team, away_team, game_date)

    home_stats = _fetch_pitcher_season_stats(home_pid) if home_pid else None
    away_stats = _fetch_pitcher_season_stats(away_pid) if away_pid else None

    result = dict(defaults)
    if home_stats:
        result["home_starter_era"]  = home_stats["era"]
        result["home_starter_whip"] = home_stats["whip"]
        result["home_starter_k9"]   = home_stats["k_per_9"]
        result["home_starter_hr9"]  = home_stats["hr_per_9"]
    if away_stats:
        result["away_starter_era"]  = away_stats["era"]
        result["away_starter_whip"] = away_stats["whip"]
        result["away_starter_k9"]   = away_stats["k_per_9"]
        result["away_starter_hr9"]  = away_stats["hr_per_9"]
    if home_stats and away_stats:
        result["starter_era_diff"]  = home_stats["era"]      - away_stats["era"]
        result["starter_whip_diff"] = home_stats["whip"]     - away_stats["whip"]
        result["starter_k9_diff"]   = home_stats["k_per_9"]  - away_stats["k_per_9"]

    return result


# ─── Weather Features ────────────────────────────────────────────────────────
# Wraps weather.py's get_game_weather() and returns model features.

# Wind direction relative to common park orientations.
# Most MLB parks have home plate roughly facing NE (45°).
# Wind blowing OUT from home plate assists scoring.
_DEFAULT_PARK_FACING = 45   # degrees, NE (average)

# Park-specific home plate facing directions (degrees from N, clockwise)
_PARK_FACING: dict[str, int] = {
    "yankees":      57, "mets":        135, "red sox":    110,
    "cubs":         25, "white sox":   180, "dodgers":    355,
    "giants":       145, "athletics":   0,  "padres":     0,
    "angels":       170, "astros":      0,  "rangers":    0,   # dome
    "cardinals":    0,  "brewers":     0,   "pirates":    45,
    "reds":         0,  "tigers":      130, "guardians":  170,
    "royals":       90, "twins":       100, "orioles":    0,
    "blue jays":    0,  "rays":        0,   "marlins":    0,   # dome
    "nationals":    355, "phillies":    0,  "braves":     0,
    "rockies":      0,  "diamondbacks": 0,  "mariners":   0,  # dome
}


def get_weather_features(home_team: str, commence_time: str | None,
                         sport: str) -> dict[str, float]:
    """
    Return weather model features. Non-MLB or indoor → defaults.
    """
    nan = float("nan")
    defaults = {"game_temp_f": nan, "game_wind_mph": nan, "wind_helps_scoring": nan}

    if sport.upper() != "MLB" or not commence_time:
        return defaults

    try:
        from weather import get_game_weather
        w = get_game_weather(home_team, commence_time, sport="mlb")
    except Exception:
        return defaults

    if w.get("is_dome"):
        return {"game_temp_f": 72.0, "game_wind_mph": 0.0, "wind_helps_scoring": 0.5}

    if w.get("error"):
        return defaults

    temp_f   = w.get("temperature_f")
    wind_mph = w.get("wind_speed_mph")
    wind_dir = w.get("wind_direction_deg")

    # Compute wind_helps_scoring: 1.0 if blowing out, 0.0 if blowing in
    wind_score = 0.5  # crosswind default
    if wind_dir is not None and wind_mph is not None and wind_mph > 3:
        key = _match_team_to_park(home_team)
        facing = _PARK_FACING.get(key, _DEFAULT_PARK_FACING) if key else _DEFAULT_PARK_FACING
        # "Blowing out" = wind direction roughly opposite to where HP faces
        # If HP faces 45° (NE), wind FROM 225° (SW) blows out
        out_dir = (facing + 180) % 360
        angle_diff = abs(((wind_dir - out_dir) + 180) % 360 - 180)
        if angle_diff < 45:
            wind_score = 1.0   # blowing out
        elif angle_diff > 135:
            wind_score = 0.0   # blowing in
        else:
            wind_score = 0.5   # crosswind

    return {
        "game_temp_f":         temp_f if temp_f is not None else nan,
        "game_wind_mph":       wind_mph if wind_mph is not None else nan,
        "wind_helps_scoring":  wind_score,
    }


# ─── Statcast Batting Quality ───────────────────────────────────────────────
# Aggregated per-team from Baseball Savant's CSV endpoints (free, no auth).
# Fetched weekly — Statcast aggregates are stable over short periods.

_STATCAST_CACHE: dict[str, dict] = {}   # team_lower → stats
_STATCAST_CACHE_TIME: float = 0
_STATCAST_CACHE_TTL = 7 * 24 * 3600    # 1 week

# MLB team abbreviation → team name fragment (for matching)
_MLB_TEAM_IDS: dict[int, str] = {
    108: "angels", 109: "diamondbacks", 110: "orioles", 111: "red sox",
    112: "cubs", 113: "reds", 114: "guardians", 115: "rockies",
    116: "tigers", 117: "astros", 118: "royals", 119: "dodgers",
    120: "nationals", 121: "mets", 133: "athletics", 134: "pirates",
    135: "padres", 136: "mariners", 137: "giants", 138: "cardinals",
    139: "rays", 140: "rangers", 141: "blue jays", 142: "twins",
    143: "phillies", 144: "braves", 145: "white sox", 146: "marlins",
    147: "yankees", 158: "brewers",
}


def _load_statcast_batting() -> dict[str, dict]:
    """
    Fetch xwOBA and barrel% per team from Baseball Savant.
    Returns {team_fragment_lower: {xwoba, barrel_pct, hard_hit_pct}}.
    """
    global _STATCAST_CACHE, _STATCAST_CACHE_TIME
    now = time.time()
    if _STATCAST_CACHE and now - _STATCAST_CACHE_TIME < _STATCAST_CACHE_TTL:
        return _STATCAST_CACHE

    import csv
    from io import StringIO
    from collections import defaultdict

    team_stats: dict[int, list[dict]] = defaultdict(list)

    # 1. Expected stats (xwoba)
    try:
        r = requests.get(
            "https://baseballsavant.mlb.com/leaderboard/expected_statistics",
            params={"type": "batter", "year": "2026", "min": "50", "csv": "true"},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        # Strip UTF-8 BOM that Baseball Savant injects — without this,
        # the first column name gets a \ufeff prefix and all lookups fail.
        _csv_text = r.text.encode().decode("utf-8-sig")
        reader = csv.DictReader(StringIO(_csv_text))
        for row in reader:
            pid = int(row.get("player_id", 0))
            xwoba = _safe_float(row.get("est_woba"))
            if pid and xwoba > 0:
                # Look up player's team via the player search API
                team_stats[pid].append({"xwoba": xwoba})
    except Exception as exc:
        log.debug("Statcast xwoba fetch failed: %s", exc)

    # 2. Barrel / hard-hit stats
    try:
        r = requests.get(
            "https://baseballsavant.mlb.com/leaderboard/statcast",
            params={"type": "batter", "year": "2026", "min": "50", "csv": "true"},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        _csv_text = r.text.encode().decode("utf-8-sig")
        reader = csv.DictReader(StringIO(_csv_text))
        for row in reader:
            pid = int(row.get("player_id", 0))
            barrel = _safe_float(row.get("brl_percent"))
            hard_hit = _safe_float(row.get("ev95percent"))
            if pid:
                team_stats[pid].append({"barrel_pct": barrel, "hard_hit_pct": hard_hit})
    except Exception as exc:
        log.debug("Statcast barrel fetch failed: %s", exc)

    # 3. Map players to teams via per-team roster endpoint
    #    (hydrate=roster doesn't return rosters reliably)
    team_agg: dict[str, dict] = defaultdict(lambda: {"xwoba": [], "barrel_pct": [], "hard_hit_pct": []})
    try:
        for team_id, team_key in _MLB_TEAM_IDS.items():
            try:
                r = requests.get(
                    f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
                    params={"rosterType": "active", "season": "2026"},
                    timeout=10,
                )
                r.raise_for_status()
                roster = r.json().get("roster", [])
                for player in roster:
                    pid = player.get("person", {}).get("id")
                    if pid and pid in team_stats:
                        for stat_dict in team_stats[pid]:
                            for k, v in stat_dict.items():
                                if v and v > 0:
                                    team_agg[team_key][k].append(v)
                time.sleep(0.15)  # rate limit
            except Exception:
                pass
    except Exception as exc:
        log.debug("MLB roster fetch for Statcast mapping failed: %s", exc)

    # Compute team averages
    result: dict[str, dict] = {}
    for team_key, stats in team_agg.items():
        result[team_key] = {
            "xwoba":        sum(stats["xwoba"]) / max(len(stats["xwoba"]), 1) if stats["xwoba"] else 0.0,
            "barrel_pct":   sum(stats["barrel_pct"]) / max(len(stats["barrel_pct"]), 1) if stats["barrel_pct"] else 0.0,
            "hard_hit_pct": sum(stats["hard_hit_pct"]) / max(len(stats["hard_hit_pct"]), 1) if stats["hard_hit_pct"] else 0.0,
        }

    _STATCAST_CACHE = result
    _STATCAST_CACHE_TIME = now
    log.info("Loaded Statcast batting for %d teams", len(result))
    return result


def get_statcast_features(home_team: str, away_team: str,
                          sport: str) -> dict[str, float]:
    """
    Return Statcast batting quality features for both teams + diffs.
    Non-MLB returns NaN.
    """
    nan = float("nan")
    defaults = {
        "home_team_xwoba": nan, "away_team_xwoba": nan, "xwoba_diff": nan,
        "home_team_barrel_pct": nan, "away_team_barrel_pct": nan, "barrel_diff": nan,
        "home_team_hard_hit_pct": nan, "away_team_hard_hit_pct": nan, "hard_hit_diff": nan,
    }
    if sport.upper() != "MLB":
        return defaults

    try:
        batting = _load_statcast_batting()
    except Exception:
        return defaults

    if not batting:
        return defaults

    h_key = _match_team_to_park(home_team)  # reuse the fuzzy matcher
    a_key = _match_team_to_park(away_team)

    h_stats = batting.get(h_key, {}) if h_key else {}
    a_stats = batting.get(a_key, {}) if a_key else {}

    h_xwoba    = h_stats.get("xwoba", nan)
    a_xwoba    = a_stats.get("xwoba", nan)
    h_barrel   = h_stats.get("barrel_pct", nan)
    a_barrel   = a_stats.get("barrel_pct", nan)
    h_hard_hit = h_stats.get("hard_hit_pct", nan)
    a_hard_hit = a_stats.get("hard_hit_pct", nan)

    result = {
        "home_team_xwoba":          h_xwoba if h_xwoba else nan,
        "away_team_xwoba":          a_xwoba if a_xwoba else nan,
        "home_team_barrel_pct":     h_barrel if h_barrel else nan,
        "away_team_barrel_pct":     a_barrel if a_barrel else nan,
        "home_team_hard_hit_pct":   h_hard_hit if h_hard_hit else nan,
        "away_team_hard_hit_pct":   a_hard_hit if a_hard_hit else nan,
    }
    if h_xwoba and a_xwoba:
        result["xwoba_diff"] = h_xwoba - a_xwoba
    else:
        result["xwoba_diff"] = nan
    if h_barrel and a_barrel:
        result["barrel_diff"] = h_barrel - a_barrel
    else:
        result["barrel_diff"] = nan
    if h_hard_hit and a_hard_hit:
        result["hard_hit_diff"] = h_hard_hit - a_hard_hit
    else:
        result["hard_hit_diff"] = nan

    return result


# ─── Bullpen Quality Features ───────────────────────────────────────────────
# Uses MLB Stats API team pitching stats (no auth required).
# Season ERA for bullpen derived by subtracting starter stats from team total.

_BULLPEN_CACHE: dict[str, dict] = {}
_BULLPEN_CACHE_TIME: float = 0
_BULLPEN_CACHE_TTL = 6 * 3600


def _load_bullpen_stats() -> dict[str, dict]:
    """
    Fetch team-level bullpen stats for all MLB teams from the Stats API.
    Derives bullpen ERA by subtracting starter ERA contribution from team total.
    Returns {team_fragment_lower: {bullpen_era, bullpen_ip_pct}}.
    """
    global _BULLPEN_CACHE, _BULLPEN_CACHE_TIME
    now = time.time()
    if _BULLPEN_CACHE and now - _BULLPEN_CACHE_TIME < _BULLPEN_CACHE_TTL:
        return _BULLPEN_CACHE

    result: dict[str, dict] = {}
    team_id_to_key = _MLB_TEAM_IDS   # {int_team_id: "team_fragment"}
    team_pitching: dict[str, dict] = {}

    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/teams/stats",
            params={"sportId": "1", "season": "2026", "group": "pitching",
                    "stats": "season", "gameType": "R"},
            timeout=12,
        )
        r.raise_for_status()
        for stat_group in r.json().get("stats", []):
            for split in stat_group.get("splits", []):
                tid = split.get("team", {}).get("id")
                tkey = team_id_to_key.get(tid, "")
                if not tkey:
                    continue
                s = split.get("stat", {})
                team_pitching[tkey] = {
                    "era":  _safe_float(s.get("era")),
                    "whip": _safe_float(s.get("whip")),
                    "ip":   _safe_float(s.get("inningsPitched")),
                    "er":   _safe_float(s.get("earnedRuns")),
                }
    except Exception as exc:
        log.debug("Team pitching stats fetch failed: %s", exc)
        return result

    # Try starter-only split to derive bullpen by subtraction
    starter_pitching: dict[str, dict] = {}
    try:
        r2 = requests.get(
            "https://statsapi.mlb.com/api/v1/teams/stats",
            params={"sportId": "1", "season": "2026", "group": "pitching",
                    "stats": "season", "gameType": "R",
                    "sitCodes": "sp"},
            timeout=12,
        )
        r2.raise_for_status()
        for stat_group in r2.json().get("stats", []):
            for split in stat_group.get("splits", []):
                tid = split.get("team", {}).get("id")
                tkey = team_id_to_key.get(tid, "")
                if not tkey:
                    continue
                s = split.get("stat", {})
                starter_pitching[tkey] = {
                    "era": _safe_float(s.get("era")),
                    "ip":  _safe_float(s.get("inningsPitched")),
                    "er":  _safe_float(s.get("earnedRuns")),
                }
    except Exception as exc:
        log.debug("Starter split fetch failed, using team ERA as fallback: %s", exc)

    for tkey, total in team_pitching.items():
        sp = starter_pitching.get(tkey)
        if sp and sp.get("ip", 0) > 10:
            total_ip = total.get("ip", 0.0)
            sp_ip = sp.get("ip", 0.0)
            bp_ip = total_ip - sp_ip
            if bp_ip > 10:
                total_er = total.get("er", 0.0)
                sp_er = sp.get("er", 0.0)
                bp_er = max(total_er - sp_er, 0)
                bp_era = bp_er / bp_ip * 9.0
                result[tkey] = {
                    "bullpen_era": round(bp_era, 2),
                    "bullpen_ip_pct": round(bp_ip / max(total_ip, 1), 3),
                }
                continue
        # Fallback: use team ERA as proxy
        result[tkey] = {
            "bullpen_era": total.get("era", 4.50),
            "bullpen_ip_pct": 0.33,
        }

    _BULLPEN_CACHE = result
    _BULLPEN_CACHE_TIME = now
    log.info("Loaded bullpen stats for %d teams", len(result))
    return result


def get_bullpen_features(home_team: str, away_team: str,
                         sport: str) -> dict[str, float]:
    """
    Return bullpen quality features for both teams + diff.
    Non-MLB returns NaN.
    """
    nan = float("nan")
    defaults = {
        "home_bullpen_era": nan, "away_bullpen_era": nan, "bullpen_era_diff": nan,
    }
    if sport.upper() != "MLB":
        return defaults

    try:
        bp = _load_bullpen_stats()
    except Exception:
        return defaults

    if not bp:
        return defaults

    h_key = _match_team_to_park(home_team)
    a_key = _match_team_to_park(away_team)

    h_stats = bp.get(h_key, {}) if h_key else {}
    a_stats = bp.get(a_key, {}) if a_key else {}

    h_era = h_stats.get("bullpen_era") if h_stats else None
    a_era = a_stats.get("bullpen_era") if a_stats else None

    result = {
        "home_bullpen_era": h_era if h_era is not None else nan,
        "away_bullpen_era": a_era if a_era is not None else nan,
    }
    if h_era is not None and a_era is not None:
        result["bullpen_era_diff"] = h_era - a_era
    else:
        result["bullpen_era_diff"] = nan

    return result


# ─── H2H Historical Matchup Features ────────────────────────────────────────
# Derived from settled mock_bet_legs in the local database.

_H2H_CACHE: dict[tuple[str, str], dict] = {}
_H2H_CACHE_TIME: float = 0
_H2H_CACHE_TTL = 12 * 3600


def _load_h2h_data(db) -> dict[tuple[str, str], dict]:
    """
    Build H2H matchup stats from settled mock_bet_legs (h2h market only).
    Returns {(home_lower, away_lower): {h2h_home_wr, h2h_n_games}}.
    """
    global _H2H_CACHE, _H2H_CACHE_TIME
    now = time.time()
    if _H2H_CACHE and now - _H2H_CACHE_TIME < _H2H_CACHE_TTL:
        return _H2H_CACHE

    from sqlalchemy import text as _text
    from collections import defaultdict

    matchups: dict[tuple[str, str], list[int]] = defaultdict(list)
    try:
        rows = db.execute(_text("""
            SELECT mbl.resolved_home_team, mbl.resolved_away_team,
                   mbl.leg_result, mbl.pick
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.leg_result IN ('WIN', 'LOSS')
              AND mbl.market_type = 'h2h'
              AND mbl.resolved_home_team IS NOT NULL
              AND mbl.resolved_away_team IS NOT NULL
            ORDER BY mb.game_date DESC
        """)).fetchall()
    except Exception as exc:
        log.debug("H2H data fetch failed: %s", exc)
        return {}

    for r in rows:
        home = (r[0] or "").strip().lower()
        away = (r[1] or "").strip().lower()
        result_str = r[2]
        pick = (r[3] or "").strip().lower()
        if not home or not away:
            continue

        if home in pick:
            home_won = 1 if result_str == "WIN" else 0
        elif away in pick:
            home_won = 0 if result_str == "WIN" else 1
        else:
            continue

        matchups[(home, away)].append(home_won)

    result: dict[tuple[str, str], dict] = {}
    for (home, away), games in matchups.items():
        n = len(games)
        if n >= 2:
            home_wins = sum(games)
            result[(home, away)] = {
                "h2h_home_wr": round(home_wins / n, 3),
                "h2h_n_games": n,
            }

    _H2H_CACHE = result
    _H2H_CACHE_TIME = now
    log.info("Loaded H2H matchup data for %d pairs", len(result))
    return result


def get_h2h_features(home_team: str, away_team: str, db=None) -> dict[str, float]:
    """
    Return H2H historical matchup features.
    """
    nan = float("nan")
    defaults = {"h2h_home_wr": nan, "h2h_n_games": nan}

    if not db:
        return defaults

    try:
        h2h = _load_h2h_data(db)
    except Exception:
        return defaults

    if not h2h:
        return defaults

    ht = home_team.strip().lower()
    at = away_team.strip().lower()

    stats = h2h.get((ht, at))

    if not stats:
        ht_last = ht.split()[-1] if ht else ""
        at_last = at.split()[-1] if at else ""
        for (h, a), s in h2h.items():
            if h.split()[-1] == ht_last and a.split()[-1] == at_last:
                stats = s
                break

    if not stats:
        return defaults

    return {
        "h2h_home_wr":  stats["h2h_home_wr"],
        "h2h_n_games":  float(stats["h2h_n_games"]),
    }


# ─── Tier-A Recompute Engine (leak-proof) ──────────────────────────────────
# Computes pitcher features ONLY from pitcher_game_logs entries strictly
# BEFORE the game date.  Never calls the MLB season-stats API.
# HR9 is unavailable (game_logs lacks home_runs) and returned as NaN.

import os as _os
import sqlite3 as _sqlite3

_HIST_DB_PATH = _os.path.join(_os.path.dirname(__file__), "..", "data", "historical.db")


def get_pitcher_features_from_gamelogs(
    home_team: str, away_team: str, game_date: str,
) -> dict[str, float]:
    """
    Leak-proof pitcher features from local pitcher_game_logs.

    1. Identifies starters by joining game_logs → games (highest IP per side).
    2. Computes as-of-date ERA / WHIP / K9 using only game_date < :gd rows.
    3. HR9 returned as NaN (column not in game_logs).

    The starter is the *actual* starter (soft leak vs. probable pitcher),
    accepted as documented in the Phase-0 audit.
    """
    nan = float("nan")
    defaults = {
        "home_starter_era": nan, "home_starter_whip": nan,
        "home_starter_k9":  nan, "home_starter_hr9":  nan,
        "away_starter_era": nan, "away_starter_whip": nan,
        "away_starter_k9":  nan, "away_starter_hr9":  nan,
        "starter_era_diff": nan, "starter_whip_diff": nan,
        "starter_k9_diff":  nan,
    }

    if not _os.path.exists(_HIST_DB_PATH):
        return defaults

    try:
        conn = _sqlite3.connect(_HIST_DB_PATH)
        conn.row_factory = _sqlite3.Row

        # ── Step 1: find starters (top-IP pitcher per side for this game) ──
        # We need the game_pk for this matchup on this date.
        game_row = conn.execute(
            """
            SELECT g.game_id
            FROM games g
            WHERE g.game_date = ?
              AND g.sport = 'MLB'
              AND g.home_team = ?
              AND g.away_team = ?
            LIMIT 1
            """,
            (game_date, home_team, away_team),
        ).fetchone()

        if not game_row:
            conn.close()
            return defaults

        # game_id is like 'MLB_823538' — extract the numeric part
        game_id_str = game_row["game_id"]
        game_pk_str = game_id_str.replace("MLB_", "") if game_id_str.startswith("MLB_") else game_id_str

        # Get all pitchers who logged for this game, sorted by IP desc
        pitchers = conn.execute(
            """
            SELECT pitcher_name, player_id, ip
            FROM pitcher_game_logs
            WHERE game_pk = ? AND game_date = ?
            ORDER BY ip DESC
            """,
            (int(game_pk_str), game_date),
        ).fetchall()

        if not pitchers:
            conn.close()
            return defaults

        # Identify which pitcher belongs to which team using frequency:
        # a pitcher's OWN team appears in ALL their games; the opponent
        # varies. Count appearances and assign by majority.
        home_starter_pid = None
        away_starter_pid = None

        def _pitcher_team(player_id: int) -> str | None:
            """Return the team name this pitcher most likely belongs to."""
            rows = conn.execute(
                """
                SELECT g.home_team, g.away_team
                FROM pitcher_game_logs pgl
                JOIN games g ON g.game_id = 'MLB_' || pgl.game_pk
                WHERE pgl.player_id = ?
                  AND pgl.season = substr(?, 1, 4)
                """,
                (player_id, game_date),
            ).fetchall()
            if not rows:
                return None
            # Count how often each team appears across all this pitcher's games
            from collections import Counter
            team_counts: Counter = Counter()
            for r in rows:
                team_counts[r["home_team"]] += 1
                team_counts[r["away_team"]] += 1
            # The pitcher's own team appears in EVERY game (home or away).
            # Opponents rotate, so the pitcher's team has the highest count.
            top_team = team_counts.most_common(1)[0][0]
            return top_team

        for p in pitchers:
            if home_starter_pid and away_starter_pid:
                break
            pid = p["player_id"]
            team = _pitcher_team(pid)
            if team == home_team and not home_starter_pid:
                home_starter_pid = pid
            elif team == away_team and not away_starter_pid:
                away_starter_pid = pid
            elif not home_starter_pid:
                home_starter_pid = pid
            elif not away_starter_pid:
                away_starter_pid = pid

        # ── Step 2: compute as-of-date stats from game_logs ──
        def _compute_asof_stats(player_id: int, gd: str) -> dict | None:
            row = conn.execute(
                """
                SELECT SUM(ip) as total_ip, SUM(er) as total_er,
                       SUM(hits) as total_h, SUM(walks) as total_bb,
                       SUM(strikeouts) as total_k, COUNT(*) as n_games
                FROM pitcher_game_logs
                WHERE player_id = ?
                  AND game_date < ?
                  AND season = substr(?, 1, 4)
                """,
                (player_id, gd, gd),
            ).fetchone()
            if not row or not row["total_ip"] or row["total_ip"] <= 0:
                return None
            ip = row["total_ip"]
            return {
                "era":  row["total_er"] / ip * 9,
                "whip": (row["total_h"] + row["total_bb"]) / ip,
                "k9":   row["total_k"] / ip * 9,
            }

        result = dict(defaults)
        home_stats = _compute_asof_stats(home_starter_pid, game_date) if home_starter_pid else None
        away_stats = _compute_asof_stats(away_starter_pid, game_date) if away_starter_pid else None

        if home_stats:
            result["home_starter_era"]  = round(home_stats["era"], 4)
            result["home_starter_whip"] = round(home_stats["whip"], 4)
            result["home_starter_k9"]   = round(home_stats["k9"], 4)
        if away_stats:
            result["away_starter_era"]  = round(away_stats["era"], 4)
            result["away_starter_whip"] = round(away_stats["whip"], 4)
            result["away_starter_k9"]   = round(away_stats["k9"], 4)
        if home_stats and away_stats:
            result["starter_era_diff"]  = round(home_stats["era"]  - away_stats["era"], 4)
            result["starter_whip_diff"] = round(home_stats["whip"] - away_stats["whip"], 4)
            result["starter_k9_diff"]   = round(home_stats["k9"]   - away_stats["k9"], 4)

        conn.close()
        return result

    except Exception as exc:
        log.warning("get_pitcher_features_from_gamelogs failed: %s", exc)
        return defaults
