"""
weather.py — Open-Meteo weather data for outdoor sports (MLB, NFL).

Open-Meteo is free with no API key required.
Forecast API: https://api.open-meteo.com/v1/forecast

Stadium coordinates are baked in for all MLB and NFL venues.
For each game, we return:
  - temperature_f: temperature at game time in °F
  - wind_speed_mph: wind speed at game time in mph
  - wind_direction_deg: wind direction in degrees
  - precipitation_prob_pct: % chance of rain/snow at game time
  - weather_alert: True if conditions likely to affect scoring
    (wind ≥ 15 mph OR precip ≥ 40%)

Usage
-----
    from weather import get_game_weather
    w = get_game_weather("New York Mets", "2026-05-01T19:10:00Z")
    # → {"temperature_f": 68, "wind_speed_mph": 12, ...}
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# ─── Stadium coordinates ──────────────────────────────────────────────────────

# MLB stadiums: {team_name_fragment: (lat, lon, stadium_name)}
MLB_STADIUMS: dict[str, tuple[float, float, str]] = {
    "yankees":          (40.8296, -73.9262, "Yankee Stadium"),
    "mets":             (40.7571, -73.8458, "Citi Field"),
    "red sox":          (42.3467, -71.0972, "Fenway Park"),
    "cubs":             (41.9484, -87.6553, "Wrigley Field"),
    "white sox":        (41.8300, -87.6339, "Guaranteed Rate Field"),
    "dodgers":          (34.0739, -118.2400, "Dodger Stadium"),
    "giants":           (37.7786, -122.3893, "Oracle Park"),
    "athletics":        (37.7516, -122.2005, "Sutter Health Park"),
    "padres":           (32.7073, -117.1566, "Petco Park"),
    "angels":           (33.8003, -117.8827, "Angel Stadium"),
    "astros":           (29.7573, -95.3555, "Minute Maid Park"),
    "rangers":          (32.7473, -97.0829, "Globe Life Field"),
    "cardinals":        (38.6226, -90.1928, "Busch Stadium"),
    "brewers":          (43.0280, -87.9712, "American Family Field"),
    "pirates":          (40.4469, -80.0058, "PNC Park"),
    "reds":             (39.0974, -84.5069, "Great American Ball Park"),
    "tigers":           (42.3390, -83.0485, "Comerica Park"),
    "indians":          (41.4962, -81.6852, "Progressive Field"),
    "guardians":        (41.4962, -81.6852, "Progressive Field"),
    "royals":           (39.0517, -94.4803, "Kauffman Stadium"),
    "twins":            (44.9817, -93.2778, "Target Field"),
    "orioles":          (39.2838, -76.6217, "Camden Yards"),
    "blue jays":        (43.6414, -79.3894, "Rogers Centre"),
    "rays":             (27.7682, -82.6534, "Tropicana Field"),
    "marlins":          (25.7781, -80.2197, "loanDepot Park"),
    "nationals":        (38.8730, -77.0074, "Nationals Park"),
    "phillies":         (39.9061, -75.1665, "Citizens Bank Park"),
    "braves":           (33.8907, -84.4677, "Truist Park"),
    "rockies":          (39.7559, -104.9942, "Coors Field"),
    "diamondbacks":     (33.4453, -112.0667, "Chase Field"),
    "mariners":         (47.5914, -122.3325, "T-Mobile Park"),
}

# NFL stadiums
NFL_STADIUMS: dict[str, tuple[float, float, str]] = {
    "chiefs":           (39.0489, -94.4839, "Arrowhead Stadium"),
    "bills":            (42.7738, -78.7868, "Highmark Stadium"),
    "patriots":         (42.0909, -71.2643, "Gillette Stadium"),
    "steelers":         (40.4468, -80.0158, "Acrisure Stadium"),
    "ravens":           (39.2780, -76.6227, "M&T Bank Stadium"),
    "browns":           (41.5061, -81.6995, "Huntington Bank Field"),
    "bengals":          (39.0955, -84.5160, "Paycor Stadium"),
    "titans":           (36.1665, -86.7713, "Nissan Stadium"),
    "colts":            (39.7601, -86.1638, "Lucas Oil Stadium"),
    "jaguars":          (30.3239, -81.6373, "EverBank Stadium"),
    "texans":           (29.6847, -95.4107, "NRG Stadium"),
    "cowboys":          (32.7473, -97.0929, "AT&T Stadium"),
    "eagles":           (39.9008, -75.1675, "Lincoln Financial Field"),
    "giants":           (40.8135, -74.0745, "MetLife Stadium"),
    "jets":             (40.8135, -74.0745, "MetLife Stadium"),
    "commanders":       (38.9078, -76.8644, "FedExField"),
    "bears":            (41.8623, -87.6167, "Soldier Field"),
    "packers":          (44.5013, -88.0622, "Lambeau Field"),
    "vikings":          (44.9739, -93.2575, "U.S. Bank Stadium"),
    "lions":            (42.3400, -83.0456, "Ford Field"),
    "saints":           (29.9508, -90.0810, "Caesars Superdome"),
    "falcons":          (33.7553, -84.4006, "Mercedes-Benz Stadium"),
    "panthers":         (35.2258, -80.8529, "Bank of America Stadium"),
    "buccaneers":       (27.9759, -82.5033, "Raymond James Stadium"),
    "rams":             (33.9534, -118.3390, "SoFi Stadium"),
    "chargers":         (33.9534, -118.3390, "SoFi Stadium"),
    "seahawks":         (47.5952, -122.3316, "Lumen Field"),
    "49ers":            (37.4032, -121.9698, "Levi's Stadium"),
    "cardinals":        (33.5276, -112.2626, "State Farm Stadium"),
    "broncos":          (39.7439, -105.0201, "Empower Field"),
    "raiders":          (36.0909, -115.1833, "Allegiant Stadium"),
    "dolphins":         (25.9580, -80.2388, "Hard Rock Stadium"),
}

# Dome/retractable roof stadiums — weather irrelevant
_DOMES = {
    "rogers centre", "tropicana field", "loandepot park",
    "chase field",   "minute maid park",
    # NFL domes
    "at&t stadium", "nissan stadium", "lucas oil stadium",
    "u.s. bank stadium", "allegiant stadium", "caesars superdome",
    "mercedes-benz stadium", "ford field", "sofi stadium",
    "state farm stadium",
}


def _find_stadium(team_name: str, sport: str = "mlb"
                  ) -> tuple[float, float, str] | None:
    """Look up (lat, lon, stadium_name) by partial team name match."""
    lookup = team_name.lower()
    stadiums = MLB_STADIUMS if sport.lower() == "mlb" else NFL_STADIUMS
    for fragment, coords in stadiums.items():
        if fragment in lookup or lookup in fragment:
            return coords
    return None


def _is_dome(stadium_name: str) -> bool:
    return stadium_name.lower() in _DOMES


def _c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _ms_to_mph(ms: float) -> float:
    return round(ms * 2.237, 1)


def get_game_weather(home_team: str, commence_time: str,
                     sport: str = "mlb") -> dict:
    """
    Return weather forecast for a game.

    Parameters
    ----------
    home_team     : team name (used to look up stadium)
    commence_time : ISO 8601 UTC string, e.g. "2026-05-01T19:10:00Z"
    sport         : "mlb" or "nfl"

    Returns
    -------
    dict with keys: temperature_f, wind_speed_mph, wind_direction_deg,
                    precipitation_prob_pct, weather_alert, stadium,
                    is_dome, game_time_utc, error (if any)
    """
    stadium_info = _find_stadium(home_team, sport)
    if not stadium_info:
        return {"error": f"No stadium found for '{home_team}'",
                "is_dome": False, "weather_alert": False}

    lat, lon, stadium_name = stadium_info

    if _is_dome(stadium_name):
        return {
            "stadium":              stadium_name,
            "is_dome":              True,
            "weather_alert":        False,
            "temperature_f":        None,
            "wind_speed_mph":       None,
            "wind_direction_deg":   None,
            "precipitation_prob_pct": None,
            "note":                 "Indoor/dome venue — weather not applicable",
        }

    # Parse game time
    try:
        game_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except Exception:
        return {"error": f"Invalid commence_time: {commence_time}",
                "is_dome": False, "weather_alert": False}

    game_date = game_dt.strftime("%Y-%m-%d")

    # Open-Meteo forecast only covers 16 days out
    game_dt_utc = game_dt if game_dt.tzinfo else game_dt.replace(tzinfo=timezone.utc)
    now_utc     = datetime.now(timezone.utc)
    days_out    = (game_dt_utc - now_utc).days
    if days_out > 16:
        return {
            "stadium":              stadium_name,
            "is_dome":              False,
            "weather_alert":        False,
            "temperature_f":        None,
            "wind_speed_mph":       None,
            "wind_direction_deg":   None,
            "precipitation_prob_pct": None,
            "note":                 f"Forecast unavailable — game is {days_out} days out (max 16)",
        }
    elif days_out < -7:
        return {
            "stadium":              stadium_name,
            "is_dome":              False,
            "weather_alert":        False,
            "note":                 "Game already played",
        }

    # Open-Meteo forecast: request enough days to cover game date
    forecast_days = max(2, days_out + 2)  # at least 2 days, always cover game date

    try:
        r = requests.get(
            _BASE_URL,
            params={
                "latitude":               lat,
                "longitude":              lon,
                "hourly":                 "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
                "temperature_unit":       "celsius",
                "wind_speed_unit":        "ms",
                "timezone":               "UTC",
                "forecast_days":          forecast_days,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        return {"error": str(exc), "is_dome": False, "weather_alert": False}

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    temps  = hourly.get("temperature_2m", [])
    winds  = hourly.get("wind_speed_10m", [])
    dirs   = hourly.get("wind_direction_10m", [])
    precip = hourly.get("precipitation_probability", [])

    if not times:
        return {"error": "No hourly data returned", "is_dome": False, "weather_alert": False}

    # Find the hour matching game date + game time
    target_prefix = f"{game_date}T{game_dt.strftime('%H')}"
    idx = None
    for i, t in enumerate(times):
        if t.startswith(target_prefix):
            idx = i
            break
    if idx is None:
        # Fallback: closest hour on game date
        matching = [i for i, t in enumerate(times) if t.startswith(game_date)]
        if matching:
            target_hour = game_dt.hour
            idx = min(matching, key=lambda i: abs(int(times[i][11:13]) - target_hour))
    if idx is None:
        return {"error": f"No forecast data for {game_date}", "is_dome": False, "weather_alert": False}

    temp_c      = temps[idx]  if idx < len(temps)  else None
    wind_ms     = winds[idx]  if idx < len(winds)  else None
    wind_dir    = dirs[idx]   if idx < len(dirs)   else None
    precip_prob = precip[idx] if idx < len(precip) else None

    temp_f    = _c_to_f(temp_c)    if temp_c  is not None else None
    wind_mph  = _ms_to_mph(wind_ms) if wind_ms is not None else None

    alert = bool(
        (wind_mph  is not None and wind_mph  >= 15) or
        (precip_prob is not None and precip_prob >= 40)
    )

    return {
        "stadium":                stadium_name,
        "is_dome":                False,
        "game_time_utc":          commence_time,
        "temperature_f":          temp_f,
        "wind_speed_mph":         wind_mph,
        "wind_direction_deg":     wind_dir,
        "precipitation_prob_pct": precip_prob,
        "weather_alert":          alert,
    }


def enrich_fixtures_with_weather(fixtures: list[dict],
                                 sport: str = "mlb") -> list[dict]:
    """
    Given a list of fixture dicts (home_team, commence_time), return
    the same list with a 'weather' key added to each.
    """
    for fix in fixtures:
        home      = fix.get("home_team", "")
        commence  = fix.get("commence_time", "")
        fix["weather"] = get_game_weather(home, commence, sport)
        time.sleep(0.1)   # rate limit
    return fixtures
