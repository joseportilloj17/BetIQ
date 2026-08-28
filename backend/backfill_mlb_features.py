"""
backfill_mlb_features.py — Backfill pitcher/weather/statcast/bullpen/H2H
features on settled MLB mock_bet_legs.

Reads from data/bets.db, calls MLB Stats API + Open-Meteo archive + Savant,
and updates columns in-place.  Does NOT touch .pkl files or retrain.

Usage:
    cd /Users/joseportillo/Downloads/BetIQ/backend
    python backfill_mlb_features.py
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone

# Ensure backend/ is on the path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from mlb_features import (
    get_pitcher_features,
    get_weather_features,
    get_bullpen_features,
    get_statcast_features,
    _match_team_to_park,
    _safe_float,
)
from weather import MLB_STADIUMS, _is_dome, _find_stadium

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bets.db")

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _is_nan(v):
    """Check if value is NaN (works for None too)."""
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (ValueError, TypeError):
        return True


def _val_or_none(v):
    """Convert NaN to None for SQLite storage."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# 1. PITCHER BACKFILL — MLB Stats API probable pitchers for each game date
# ────────────────────────────────────────────────────────────────────────────

def backfill_pitcher(conn: sqlite3.Connection):
    """Backfill pitcher features for all settled MLB legs missing them."""
    cur = conn.cursor()

    # Get all distinct games that need pitcher data
    cur.execute("""
        SELECT DISTINCT mbl.resolved_home_team, mbl.resolved_away_team, mb.game_date
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.sport = 'MLB'
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.home_starter_era IS NULL
          AND mbl.resolved_home_team IS NOT NULL
          AND mbl.resolved_away_team IS NOT NULL
          AND mb.game_date IS NOT NULL
        ORDER BY mb.game_date
    """)
    games = cur.fetchall()
    total_games = len(games)
    log(f"PITCHER: {total_games} distinct games need pitcher data")

    if total_games == 0:
        return

    success, fail, legs_updated = 0, 0, 0
    for idx, (home, away, game_date) in enumerate(games):
        try:
            feats = get_pitcher_features(home, away, game_date, "MLB")

            # Check if we actually got data (not all NaN)
            has_data = any(not _is_nan(v) for v in feats.values())

            # Update all legs for this game
            cur.execute("""
                UPDATE mock_bet_legs
                SET home_starter_era  = ?,
                    home_starter_whip = ?,
                    home_starter_k9   = ?,
                    home_starter_hr9  = ?,
                    away_starter_era  = ?,
                    away_starter_whip = ?,
                    away_starter_k9   = ?,
                    away_starter_hr9  = ?,
                    starter_era_diff  = ?,
                    starter_whip_diff = ?,
                    starter_k9_diff   = ?
                WHERE id IN (
                    SELECT mbl.id
                    FROM mock_bet_legs mbl
                    JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
                    WHERE mbl.sport = 'MLB'
                      AND mbl.leg_result IN ('WIN', 'LOSS')
                      AND mbl.resolved_home_team = ?
                      AND mbl.resolved_away_team = ?
                      AND mb.game_date = ?
                      AND mbl.home_starter_era IS NULL
                )
            """, (
                _val_or_none(feats.get("home_starter_era")),
                _val_or_none(feats.get("home_starter_whip")),
                _val_or_none(feats.get("home_starter_k9")),
                _val_or_none(feats.get("home_starter_hr9")),
                _val_or_none(feats.get("away_starter_era")),
                _val_or_none(feats.get("away_starter_whip")),
                _val_or_none(feats.get("away_starter_k9")),
                _val_or_none(feats.get("away_starter_hr9")),
                _val_or_none(feats.get("starter_era_diff")),
                _val_or_none(feats.get("starter_whip_diff")),
                _val_or_none(feats.get("starter_k9_diff")),
                home, away, game_date,
            ))
            n = cur.rowcount
            legs_updated += n

            if has_data:
                success += 1
            else:
                fail += 1

        except Exception as exc:
            fail += 1
            log(f"  PITCHER ERROR game {idx+1} ({home} vs {away} on {game_date}): {exc}")

        # Commit every 50 games
        if (idx + 1) % 50 == 0:
            conn.commit()
            log(f"  PITCHER progress: {idx+1}/{total_games} games, {legs_updated} legs updated")

        # Rate limit: 0.5s between API calls
        time.sleep(0.5)

    conn.commit()
    log(f"PITCHER DONE: {success} games with data, {fail} without data, {legs_updated} legs updated")


# ────────────────────────────────────────────────────────────────────────────
# 2. WEATHER BACKFILL — Open-Meteo ARCHIVE API for historical games
# ────────────────────────────────────────────────────────────────────────────

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get_historical_weather(home_team: str, game_date: str) -> dict:
    """
    Fetch historical weather from Open-Meteo Archive API.
    game_date is YYYY-MM-DD. Returns dict with temp_f, wind_mph, wind_dir.
    """
    stadium_info = _find_stadium(home_team, "mlb")
    if not stadium_info:
        return {"error": f"No stadium for {home_team}"}

    lat, lon, stadium_name = stadium_info

    if _is_dome(stadium_name):
        return {"is_dome": True, "temperature_f": 72.0, "wind_speed_mph": 0.0,
                "wind_direction_deg": None}

    try:
        r = requests.get(
            _ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": game_date,
                "end_date": game_date,
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "timezone": "America/New_York",  # most MLB games in US timezones
            },
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        return {"error": str(exc)}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    dirs = hourly.get("wind_direction_10m", [])

    if not times:
        return {"error": "No hourly data"}

    # Pick typical game start hour: 7 PM ET = index 19
    # (most MLB games are between 6-8 PM local)
    game_hour = 19
    idx = min(game_hour, len(times) - 1)

    temp_c = temps[idx] if idx < len(temps) else None
    wind_ms = winds[idx] if idx < len(winds) else None
    wind_dir = dirs[idx] if idx < len(dirs) else None

    temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
    wind_mph = round(wind_ms * 2.237, 1) if wind_ms is not None else None

    return {
        "temperature_f": temp_f,
        "wind_speed_mph": wind_mph,
        "wind_direction_deg": wind_dir,
        "is_dome": False,
        "stadium": stadium_name,
    }


def _compute_wind_helps_scoring(home_team: str, wind_dir, wind_mph) -> float:
    """
    Compute wind_helps_scoring score (0.0 = blowing in, 0.5 = crosswind, 1.0 = blowing out).
    Replicates logic from mlb_features.py.
    """
    from mlb_features import _PARK_FACING, _DEFAULT_PARK_FACING, _match_team_to_park

    if wind_dir is None or wind_mph is None or wind_mph <= 3:
        return 0.5

    key = _match_team_to_park(home_team)
    facing = _PARK_FACING.get(key, _DEFAULT_PARK_FACING) if key else _DEFAULT_PARK_FACING

    out_dir = (facing + 180) % 360
    angle_diff = abs(((wind_dir - out_dir) + 180) % 360 - 180)

    if angle_diff < 45:
        return 1.0
    elif angle_diff > 135:
        return 0.0
    else:
        return 0.5


def backfill_weather(conn: sqlite3.Connection):
    """Backfill weather features for settled MLB legs."""
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT mbl.resolved_home_team, mb.game_date
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.sport = 'MLB'
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.game_temp_f IS NULL
          AND mbl.resolved_home_team IS NOT NULL
          AND mb.game_date IS NOT NULL
        ORDER BY mb.game_date
    """)
    game_venues = cur.fetchall()
    total = len(game_venues)
    log(f"WEATHER: {total} distinct (home_team, date) combos need weather data")

    if total == 0:
        return

    success, fail, legs_updated = 0, 0, 0
    for idx, (home, game_date) in enumerate(game_venues):
        try:
            w = _get_historical_weather(home, game_date)

            if w.get("error"):
                fail += 1
                if idx < 5:
                    log(f"  WEATHER skip: {home} on {game_date}: {w['error']}")
                continue

            temp_f = w.get("temperature_f")
            wind_mph = w.get("wind_speed_mph")
            wind_dir = w.get("wind_direction_deg")
            is_dome = w.get("is_dome", False)

            if is_dome:
                wind_score = 0.5
            else:
                wind_score = _compute_wind_helps_scoring(home, wind_dir, wind_mph)

            cur.execute("""
                UPDATE mock_bet_legs
                SET game_temp_f = ?,
                    game_wind_mph = ?,
                    wind_helps_scoring = ?
                WHERE id IN (
                    SELECT mbl.id
                    FROM mock_bet_legs mbl
                    JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
                    WHERE mbl.sport = 'MLB'
                      AND mbl.leg_result IN ('WIN', 'LOSS')
                      AND mbl.resolved_home_team = ?
                      AND mb.game_date = ?
                      AND mbl.game_temp_f IS NULL
                )
            """, (
                temp_f if not is_dome else 72.0,
                wind_mph if not is_dome else 0.0,
                wind_score,
                home, game_date,
            ))
            n = cur.rowcount
            legs_updated += n
            success += 1

        except Exception as exc:
            fail += 1
            log(f"  WEATHER ERROR {home} on {game_date}: {exc}")

        if (idx + 1) % 50 == 0:
            conn.commit()
            log(f"  WEATHER progress: {idx+1}/{total} venues, {legs_updated} legs updated")

        # Open-Meteo archive: be respectful, 0.3s between requests
        time.sleep(0.3)

    conn.commit()
    log(f"WEATHER DONE: {success} venues with data, {fail} failed, {legs_updated} legs updated")


# ────────────────────────────────────────────────────────────────────────────
# 3. BULLPEN BACKFILL — MLB Stats API team pitching stats
# ────────────────────────────────────────────────────────────────────────────

def backfill_bullpen(conn: sqlite3.Connection):
    """
    Backfill bullpen ERA features. Uses current-season aggregate stats
    (not as-of-date), so accuracy degrades for older games. Still better than NaN.
    """
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT mbl.resolved_home_team, mbl.resolved_away_team
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.sport = 'MLB'
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.home_bullpen_era IS NULL
          AND mbl.resolved_home_team IS NOT NULL
          AND mbl.resolved_away_team IS NOT NULL
    """)
    pairs = cur.fetchall()
    total = len(pairs)
    log(f"BULLPEN: {total} distinct team pairs need bullpen data")

    if total == 0:
        return

    # Fetch bullpen stats once (current-season snapshot for all teams)
    try:
        feats_cache = {}
        # Pre-fetch all team bullpen stats (one API call via get_bullpen_features)
        # The first call loads the cache; subsequent calls are instant.
        for idx, (home, away) in enumerate(pairs):
            cache_key = (home, away)
            if cache_key not in feats_cache:
                feats = get_bullpen_features(home, away, "MLB")
                feats_cache[cache_key] = feats
                if idx == 0:
                    # Only the first call hits the API; the rest use cache
                    time.sleep(0.5)
    except Exception as exc:
        log(f"BULLPEN pre-fetch error: {exc}")
        return

    legs_updated = 0
    success = 0
    for idx, (home, away) in enumerate(pairs):
        feats = feats_cache.get((home, away), {})
        has_data = any(not _is_nan(v) for v in feats.values())

        cur.execute("""
            UPDATE mock_bet_legs
            SET home_bullpen_era = ?,
                away_bullpen_era = ?,
                bullpen_era_diff = ?
            WHERE id IN (
                SELECT mbl.id
                FROM mock_bet_legs mbl
                JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
                WHERE mbl.sport = 'MLB'
                  AND mbl.leg_result IN ('WIN', 'LOSS')
                  AND mbl.resolved_home_team = ?
                  AND mbl.resolved_away_team = ?
                  AND mbl.home_bullpen_era IS NULL
            )
        """, (
            _val_or_none(feats.get("home_bullpen_era")),
            _val_or_none(feats.get("away_bullpen_era")),
            _val_or_none(feats.get("bullpen_era_diff")),
            home, away,
        ))
        legs_updated += cur.rowcount
        if has_data:
            success += 1

        if (idx + 1) % 50 == 0:
            conn.commit()
            log(f"  BULLPEN progress: {idx+1}/{total} pairs, {legs_updated} legs updated")

    conn.commit()
    log(f"BULLPEN DONE: {success}/{total} pairs with data, {legs_updated} legs updated")


# ────────────────────────────────────────────────────────────────────────────
# 4. H2H BACKFILL — Pure local DB, no API calls
# ────────────────────────────────────────────────────────────────────────────

def backfill_h2h(conn: sqlite3.Connection):
    """
    Backfill H2H features using only local settled legs data.
    CRITICAL: only uses games BEFORE the current game's date to prevent leakage.
    """
    cur = conn.cursor()

    # Get all settled MLB legs needing H2H, ordered by date
    cur.execute("""
        SELECT mbl.id, mbl.resolved_home_team, mbl.resolved_away_team,
               mb.game_date, mbl.leg_result, mbl.description, mbl.market_type
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.sport = 'MLB'
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.h2h_home_wr IS NULL
          AND mbl.resolved_home_team IS NOT NULL
          AND mbl.resolved_away_team IS NOT NULL
          AND mb.game_date IS NOT NULL
        ORDER BY mb.game_date
    """)
    legs = cur.fetchall()
    total = len(legs)
    log(f"H2H: {total} legs need H2H data")

    if total == 0:
        return

    # Pre-load ALL settled MLB h2h legs for fast lookups
    cur.execute("""
        SELECT mbl.resolved_home_team, mbl.resolved_away_team,
               mb.game_date, mbl.leg_result, mbl.description
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.sport = 'MLB'
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.market_type = 'h2h'
          AND mbl.resolved_home_team IS NOT NULL
          AND mbl.resolved_away_team IS NOT NULL
          AND mb.game_date IS NOT NULL
        ORDER BY mb.game_date
    """)
    all_h2h_legs = cur.fetchall()
    log(f"H2H: loaded {len(all_h2h_legs)} settled h2h legs for reference")

    # Build a lookup: for each matchup pair (sorted), accumulate game results by date
    # Key: frozenset({team_a, team_b}) -> list of (game_date, home_team, home_won)
    matchup_history = defaultdict(list)
    for (h, a, gd, result, desc) in all_h2h_legs:
        h_lower = (h or "").strip().lower()
        a_lower = (a or "").strip().lower()
        p_lower = (desc or "").strip().lower()
        if not h_lower or not a_lower:
            continue

        # Determine if home team won
        if h_lower in p_lower:
            home_won = 1 if result == "WIN" else 0
        elif a_lower in p_lower:
            home_won = 0 if result == "WIN" else 1
        else:
            continue

        key = frozenset({h_lower, a_lower})
        matchup_history[key].append((gd, h_lower, home_won))

    legs_updated = 0
    for idx, (leg_id, home, away, game_date, result, desc, mkt) in enumerate(legs):
        h_lower = (home or "").strip().lower()
        a_lower = (away or "").strip().lower()

        key = frozenset({h_lower, a_lower})
        prior_games = matchup_history.get(key, [])

        # Only count games BEFORE this game's date (no leakage!)
        prior = [g for g in prior_games if g[0] < game_date]

        if len(prior) >= 1:
            # Compute home win rate from perspective of the current game's home team
            home_wins = 0
            total_prior = len(prior)
            for (gd, hist_home, hist_home_won) in prior:
                if hist_home == h_lower:
                    home_wins += hist_home_won
                else:
                    # Teams are swapped — away team was home in that game
                    home_wins += (1 - hist_home_won)

            h2h_home_wr = round(home_wins / total_prior, 4)
            h2h_n_games = total_prior
        else:
            h2h_home_wr = None
            h2h_n_games = None

        cur.execute("""
            UPDATE mock_bet_legs SET h2h_home_wr = ?, h2h_n_games = ? WHERE id = ?
        """, (h2h_home_wr, h2h_n_games, leg_id))
        legs_updated += cur.rowcount

        if (idx + 1) % 100 == 0:
            conn.commit()
            log(f"  H2H progress: {idx+1}/{total} legs")

    conn.commit()
    filled = sum(1 for (lid, h, a, gd, r, p, m) in legs
                 if len([g for g in matchup_history.get(frozenset({(h or "").strip().lower(), (a or "").strip().lower()}), []) if g[0] < gd]) >= 1)
    log(f"H2H DONE: {legs_updated} legs updated, {filled} had prior matchup data")


# ────────────────────────────────────────────────────────────────────────────
# 5. STATCAST BACKFILL — Baseball Savant (current season snapshot)
# ────────────────────────────────────────────────────────────────────────────

def _load_statcast_batting_fixed() -> dict:
    """
    Load Statcast batting data (xwOBA, barrel%) per team.
    Fixed version: fetches rosters per team instead of using hydrate.
    Returns {team_fragment_lower: {xwoba, barrel_pct}}.
    """
    import csv
    from io import StringIO
    from collections import defaultdict

    # Player-level stats
    player_xwoba: dict[int, float] = {}
    player_barrel: dict[int, float] = {}

    # 1. Expected stats (xwoba)
    try:
        r = requests.get(
            "https://baseballsavant.mlb.com/leaderboard/expected_statistics",
            params={"type": "batter", "year": "2026", "min": "50", "csv": "true"},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        _csv_text = r.text.encode().decode("utf-8-sig")
        reader = csv.DictReader(StringIO(_csv_text))
        for row in reader:
            pid = int(row.get("player_id", 0))
            xwoba = _safe_float(row.get("est_woba"))
            if pid and xwoba > 0:
                player_xwoba[pid] = xwoba
        log(f"  STATCAST: loaded xwOBA for {len(player_xwoba)} players")
    except Exception as exc:
        log(f"  STATCAST xwoba fetch failed: {exc}")

    # 2. Barrel stats
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
            if pid and barrel > 0:
                player_barrel[pid] = barrel
        log(f"  STATCAST: loaded barrel% for {len(player_barrel)} players")
    except Exception as exc:
        log(f"  STATCAST barrel fetch failed: {exc}")

    if not player_xwoba and not player_barrel:
        return {}

    # 3. Map players to teams using per-team roster endpoint
    from mlb_features import _MLB_TEAM_IDS
    team_agg: dict[str, dict] = defaultdict(lambda: {"xwoba": [], "barrel_pct": []})

    all_player_ids = set(player_xwoba.keys()) | set(player_barrel.keys())

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
                if pid:
                    if pid in player_xwoba:
                        team_agg[team_key]["xwoba"].append(player_xwoba[pid])
                    if pid in player_barrel:
                        team_agg[team_key]["barrel_pct"].append(player_barrel[pid])
            time.sleep(0.2)  # rate limit
        except Exception as exc:
            log(f"  STATCAST roster fetch failed for team {team_key}: {exc}")

    result: dict[str, dict] = {}
    for team_key, stats in team_agg.items():
        xwoba_list = stats["xwoba"]
        barrel_list = stats["barrel_pct"]
        result[team_key] = {
            "xwoba": sum(xwoba_list) / len(xwoba_list) if xwoba_list else 0.0,
            "barrel_pct": sum(barrel_list) / len(barrel_list) if barrel_list else 0.0,
        }
    log(f"  STATCAST: computed stats for {len(result)} teams")
    return result


def backfill_statcast(conn: sqlite3.Connection):
    """
    Backfill Statcast features. Uses current-season aggregate stats
    (not as-of-date). Fixed version with per-team roster fetching.
    """
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT mbl.resolved_home_team, mbl.resolved_away_team
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.sport = 'MLB'
          AND mbl.leg_result IN ('WIN', 'LOSS')
          AND mbl.home_team_xwoba IS NULL
          AND mbl.resolved_home_team IS NOT NULL
          AND mbl.resolved_away_team IS NOT NULL
    """)
    pairs = cur.fetchall()
    total = len(pairs)
    log(f"STATCAST: {total} distinct team pairs need statcast data")

    if total == 0:
        return

    # Load statcast data with fixed roster fetching
    try:
        batting = _load_statcast_batting_fixed()
    except Exception as exc:
        log(f"STATCAST: failed to load data: {exc}")
        return

    if not batting:
        log("STATCAST: no data loaded, skipping")
        return

    nan = float("nan")
    legs_updated = 0
    success = 0
    for idx, (home, away) in enumerate(pairs):
        h_key = _match_team_to_park(home)
        a_key = _match_team_to_park(away)

        h_stats = batting.get(h_key, {}) if h_key else {}
        a_stats = batting.get(a_key, {}) if a_key else {}

        h_xwoba = h_stats.get("xwoba") or None
        a_xwoba = a_stats.get("xwoba") or None
        h_barrel = h_stats.get("barrel_pct") or None
        a_barrel = a_stats.get("barrel_pct") or None

        xwoba_diff = (h_xwoba - a_xwoba) if h_xwoba and a_xwoba else None
        barrel_diff = (h_barrel - a_barrel) if h_barrel and a_barrel else None

        has_data = h_xwoba is not None or a_xwoba is not None

        cur.execute("""
            UPDATE mock_bet_legs
            SET home_team_xwoba = ?,
                away_team_xwoba = ?,
                xwoba_diff = ?,
                home_team_barrel_pct = ?,
                away_team_barrel_pct = ?,
                barrel_diff = ?
            WHERE id IN (
                SELECT mbl.id
                FROM mock_bet_legs mbl
                JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
                WHERE mbl.sport = 'MLB'
                  AND mbl.leg_result IN ('WIN', 'LOSS')
                  AND mbl.resolved_home_team = ?
                  AND mbl.resolved_away_team = ?
                  AND mbl.home_team_xwoba IS NULL
            )
        """, (
            h_xwoba, a_xwoba, xwoba_diff,
            h_barrel, a_barrel, barrel_diff,
            home, away,
        ))
        legs_updated += cur.rowcount
        if has_data:
            success += 1

        if (idx + 1) % 50 == 0:
            conn.commit()
            log(f"  STATCAST progress: {idx+1}/{total} pairs")

    conn.commit()
    log(f"STATCAST DONE: {success}/{total} pairs with data, {legs_updated} legs updated")


# ────────────────────────────────────────────────────────────────────────────
# 6. INJURY COVERAGE CHECK
# ────────────────────────────────────────────────────────────────────────────

def check_injury_coverage(conn: sqlite3.Connection):
    """Report injury_net_advantage coverage stats."""
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN injury_net_advantage IS NOT NULL THEN 1 ELSE 0 END) as has_injury
        FROM mock_bet_legs
        WHERE leg_result IN ('WIN', 'LOSS')
    """)
    total, has_injury = cur.fetchone()
    pct = (has_injury / total * 100) if total else 0
    log(f"INJURY CHECK: {has_injury}/{total} settled legs have injury_net_advantage ({pct:.1f}%)")

    # Check by sport
    cur.execute("""
        SELECT sport, COUNT(*) as total,
               SUM(CASE WHEN injury_net_advantage IS NOT NULL THEN 1 ELSE 0 END) as has_injury
        FROM mock_bet_legs
        WHERE leg_result IN ('WIN', 'LOSS')
        GROUP BY sport
        ORDER BY total DESC
    """)
    for sport, t, h in cur.fetchall():
        pct = (h / t * 100) if t else 0
        log(f"  {sport}: {h}/{t} ({pct:.1f}%)")


# ────────────────────────────────────────────────────────────────────────────
# 7. FINAL COVERAGE REPORT
# ────────────────────────────────────────────────────────────────────────────

def coverage_report(conn: sqlite3.Connection):
    """Report coverage of all MLB features on settled legs."""
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM mock_bet_legs
        WHERE sport = 'MLB' AND leg_result IN ('WIN', 'LOSS')
    """)
    total = cur.fetchone()[0]

    cols = [
        "home_starter_era", "home_starter_whip", "home_starter_k9", "home_starter_hr9",
        "away_starter_era", "away_starter_whip", "away_starter_k9", "away_starter_hr9",
        "starter_era_diff", "starter_whip_diff", "starter_k9_diff",
        "game_temp_f", "game_wind_mph", "wind_helps_scoring",
        "home_team_xwoba", "away_team_xwoba", "xwoba_diff",
        "home_team_barrel_pct", "away_team_barrel_pct", "barrel_diff",
        "home_bullpen_era", "away_bullpen_era", "bullpen_era_diff",
        "h2h_home_wr", "h2h_n_games",
    ]

    log(f"\n{'='*60}")
    log(f"COVERAGE REPORT — {total} settled MLB legs")
    log(f"{'='*60}")

    groups = {
        "PITCHER (11)": cols[0:11],
        "WEATHER (3)": cols[11:14],
        "STATCAST (6)": cols[14:20],
        "BULLPEN (3)": cols[20:23],
        "H2H (2)": cols[23:25],
    }

    for group_name, group_cols in groups.items():
        log(f"\n  {group_name}:")
        for col in group_cols:
            cur.execute(f"""
                SELECT SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END)
                FROM mock_bet_legs
                WHERE sport = 'MLB' AND leg_result IN ('WIN', 'LOSS')
            """)
            filled = cur.fetchone()[0] or 0
            pct = (filled / total * 100) if total else 0
            log(f"    {col:30s}: {filled:5d}/{total} ({pct:5.1f}%)")


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("MLB FEATURES BACKFILL — Starting")
    log(f"Database: {os.path.abspath(DB_PATH)}")
    log("=" * 60)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")

    try:
        # Pre-check: how many legs to backfill?
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM mock_bet_legs
            WHERE sport = 'MLB' AND leg_result IN ('WIN', 'LOSS')
        """)
        total_mlb = cur.fetchone()[0]
        log(f"Total settled MLB legs: {total_mlb}")

        # 1. Pitcher (highest priority)
        log("\n" + "─" * 40)
        log("PHASE 1: PITCHER FEATURES")
        log("─" * 40)
        backfill_pitcher(conn)

        # 2. Weather
        log("\n" + "─" * 40)
        log("PHASE 2: WEATHER FEATURES")
        log("─" * 40)
        backfill_weather(conn)

        # 3. Bullpen
        log("\n" + "─" * 40)
        log("PHASE 3: BULLPEN FEATURES")
        log("─" * 40)
        backfill_bullpen(conn)

        # 4. H2H (no API calls)
        log("\n" + "─" * 40)
        log("PHASE 4: H2H FEATURES")
        log("─" * 40)
        backfill_h2h(conn)

        # 5. Statcast (lowest priority)
        log("\n" + "─" * 40)
        log("PHASE 5: STATCAST FEATURES")
        log("─" * 40)
        backfill_statcast(conn)

        # 6. Injury coverage check
        log("\n" + "─" * 40)
        log("PHASE 6: INJURY COVERAGE CHECK")
        log("─" * 40)
        check_injury_coverage(conn)

        # 7. Final coverage report
        coverage_report(conn)

    except Exception as exc:
        log(f"FATAL ERROR: {exc}")
        traceback.print_exc()
    finally:
        conn.close()

    log("\nBACKFILL COMPLETE")


if __name__ == "__main__":
    main()
