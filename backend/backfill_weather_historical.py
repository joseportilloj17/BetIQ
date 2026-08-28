"""
backfill_weather_historical.py — Backfill game_weather table in historical.db
with temperature/wind data from Open-Meteo Archive API.

Batches by stadium × season: one API call per (lat/lon, Apr-Nov) returns
the full season's hourly data.  ~31 stadiums × 5 seasons = ~155 API calls
instead of ~10K individual calls.

Usage:
    .venv/bin/python backend/backfill_weather_historical.py
"""
import os
import sys
import time
import sqlite3
import requests
import logging
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather import MLB_STADIUMS, _is_dome, _find_stadium

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "historical.db")
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_season_weather(lat: float, lon: float, start_date: str,
                          end_date: str) -> dict:
    """
    Fetch hourly weather for a location across a date range.
    Returns {date_str: {temperature_f, wind_speed_mph, wind_direction_deg}}.
    """
    try:
        r = requests.get(
            _ARCHIVE_URL,
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "timezone": "America/New_York",
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        return {"error": str(exc)}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    dirs_ = hourly.get("wind_direction_10m", [])

    if not times:
        return {"error": "No hourly data"}

    # Group by date, pick 7 PM (index 19 within each day's 24 hours)
    result = {}
    for i, t in enumerate(times):
        date_str = t[:10]  # "2022-04-07T19:00" → "2022-04-07"
        hour = int(t[11:13]) if len(t) >= 13 else 0
        if hour == 19:  # typical game start
            temp_c = temps[i] if i < len(temps) and temps[i] is not None else None
            wind_ms = winds[i] if i < len(winds) and winds[i] is not None else None
            wind_dir = dirs_[i] if i < len(dirs_) and dirs_[i] is not None else None

            temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
            wind_mph = round(wind_ms * 2.237, 1) if wind_ms is not None else None

            result[date_str] = {
                "temperature_f": temp_f,
                "wind_speed_mph": wind_mph,
                "wind_direction_deg": wind_dir,
            }

    return result


def backfill():
    conn = sqlite3.connect(DB_PATH)

    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home_team TEXT NOT NULL,
            game_date TEXT NOT NULL,
            temperature_f REAL,
            wind_speed_mph REAL,
            wind_direction_deg REAL,
            is_dome INTEGER,
            stadium TEXT,
            UNIQUE(home_team, game_date)
        )
    """)
    conn.commit()

    # Get all (home_team, game_date) needing weather, grouped by team + year
    all_games = conn.execute("""
        SELECT DISTINCT g.home_team, g.game_date
        FROM games g
        WHERE g.sport = 'MLB' AND g.status = 'Final'
          AND NOT EXISTS (
              SELECT 1 FROM game_weather gw
              WHERE gw.home_team = g.home_team AND gw.game_date = g.game_date
          )
        ORDER BY g.home_team, g.game_date
    """).fetchall()

    total = len(all_games)
    log.info("Weather backfill: %d games to fetch across %d teams",
             total, len(set(r[0] for r in all_games)))
    if total == 0:
        conn.close()
        return

    # Group by (home_team, year)
    team_year_games = defaultdict(list)
    for home_team, game_date in all_games:
        year = game_date[:4]
        team_year_games[(home_team, year)].append(game_date)

    log.info("Grouped into %d (team, year) batches", len(team_year_games))

    total_written = 0
    total_dome = 0
    total_fail = 0
    batch_num = 0

    for (home_team, year), game_dates in sorted(team_year_games.items()):
        batch_num += 1
        stadium_info = _find_stadium(home_team, "mlb")
        if not stadium_info:
            log.warning("  No stadium for %s — skipping %d games", home_team, len(game_dates))
            total_fail += len(game_dates)
            continue

        lat, lon, stadium_name = stadium_info

        if _is_dome(stadium_name):
            # Dome: no API call needed, fixed weather
            rows = [(home_team, gd, 72.0, 0.0, None, 1, stadium_name)
                    for gd in game_dates]
            conn.executemany("""
                INSERT OR REPLACE INTO game_weather
                    (home_team, game_date, temperature_f, wind_speed_mph,
                     wind_direction_deg, is_dome, stadium)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            total_dome += len(rows)
            total_written += len(rows)
            if batch_num % 20 == 0:
                log.info("  batch %d/%d: %s %s (dome, %d games)",
                         batch_num, len(team_year_games), home_team, year, len(rows))
            continue

        # Outdoor: fetch entire season's weather in one API call
        min_date = min(game_dates)
        max_date = max(game_dates)

        # Clamp max_date to yesterday (archive API doesn't have today)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if max_date >= today:
            max_date = today  # will be missing for today's games, that's fine

        weather_data = _fetch_season_weather(lat, lon, min_date, max_date)

        if isinstance(weather_data, dict) and weather_data.get("error"):
            log.warning("  API fail: %s %s — %s", home_team, year, weather_data["error"])
            total_fail += len(game_dates)
            time.sleep(1)
            continue

        # Match game dates to weather data
        rows = []
        missed = 0
        for gd in game_dates:
            wx = weather_data.get(gd)
            if wx:
                rows.append((home_team, gd, wx["temperature_f"], wx["wind_speed_mph"],
                             wx["wind_direction_deg"], 0, stadium_name))
            else:
                missed += 1

        if rows:
            conn.executemany("""
                INSERT OR REPLACE INTO game_weather
                    (home_team, game_date, temperature_f, wind_speed_mph,
                     wind_direction_deg, is_dome, stadium)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            total_written += len(rows)

        if batch_num % 10 == 0 or missed > 0:
            log.info("  batch %d/%d: %s %s — %d games, %d matched, %d missed",
                     batch_num, len(team_year_games), home_team, year,
                     len(game_dates), len(rows), missed)

        time.sleep(0.3)  # brief pause between API calls

    # Summary
    total_rows = conn.execute("SELECT COUNT(*) FROM game_weather").fetchone()[0]
    stats = conn.execute("""
        SELECT ROUND(AVG(temperature_f), 1),
               ROUND(AVG(CASE WHEN is_dome=0 THEN wind_speed_mph END), 1),
               SUM(is_dome),
               COUNT(*) - SUM(is_dome)
        FROM game_weather
    """).fetchone()

    log.info("Weather backfill complete: %d written, %d dome, %d fail, %d total rows",
             total_written, total_dome, total_fail, total_rows)
    log.info("  Stats: avg_temp=%.1f°F, avg_outdoor_wind=%.1f mph, domes=%d, outdoor=%d",
             stats[0] or 0, stats[1] or 0, stats[2] or 0, stats[3] or 0)

    conn.close()


if __name__ == "__main__":
    backfill()
