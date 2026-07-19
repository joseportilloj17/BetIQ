"""
backfill_weather_forecast.py — Backfill forecast-quality wind speed using
Open-Meteo Previous Runs API (wind_speed_10m_previous_day1).

This gives us the 1-day-ahead FORECAST of wind speed for each game,
matching what we'd have in live serving (where we fetch forecast, not
observed actuals). Covers Jan 2024 onward (API archive start).

Pre-2024 games get NULL — HistGradientBoosting handles NaN natively.

Usage:
    .venv/bin/python backend/backfill_weather_forecast.py
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
_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# API coverage starts Jan 2024
_MIN_DATE = "2024-01-01"


def _fetch_forecast_wind(lat: float, lon: float, start_date: str,
                         end_date: str) -> dict:
    """
    Fetch 1-day-ahead forecast wind speed for a location across a date range.
    Uses Open-Meteo Previous Runs API: wind_speed_10m_previous_day1.

    Returns {date_str: wind_speed_forecast_mph} or {"error": ...}.
    """
    try:
        r = requests.get(
            _PREVIOUS_RUNS_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "wind_speed_10m_previous_day1",
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
    winds = hourly.get("wind_speed_10m_previous_day1", [])

    if not times:
        return {"error": "No hourly data returned"}

    # Group by date, pick 7 PM (typical game start hour ET)
    result = {}
    for i, t in enumerate(times):
        date_str = t[:10]
        hour = int(t[11:13]) if len(t) >= 13 else 0
        if hour == 19:
            wind_ms = winds[i] if i < len(winds) and winds[i] is not None else None
            if wind_ms is not None:
                wind_mph = round(wind_ms * 2.237, 1)
                result[date_str] = wind_mph

    return result


def backfill():
    conn = sqlite3.connect(DB_PATH)

    # Add column if not exists
    cols = [c[1] for c in conn.execute("PRAGMA table_info(game_weather)").fetchall()]
    if "wind_speed_forecast_mph" not in cols:
        conn.execute("ALTER TABLE game_weather ADD COLUMN wind_speed_forecast_mph REAL")
        conn.commit()
        log.info("Added wind_speed_forecast_mph column to game_weather")

    # Get all (home_team, game_date) since 2024 that need forecast wind
    all_games = conn.execute("""
        SELECT DISTINCT gw.home_team, gw.game_date, gw.stadium, gw.is_dome
        FROM game_weather gw
        WHERE gw.game_date >= ?
          AND gw.wind_speed_forecast_mph IS NULL
        ORDER BY gw.home_team, gw.game_date
    """, (_MIN_DATE,)).fetchall()

    total = len(all_games)
    log.info("Forecast wind backfill: %d rows to fetch (since %s)", total, _MIN_DATE)
    if total == 0:
        conn.close()
        return

    # Group by (home_team, year) for batch API calls
    team_year_games = defaultdict(list)
    dome_count = 0
    for home_team, game_date, stadium, is_dome in all_games:
        if is_dome:
            # Dome stadiums: forecast wind = 0 (indoor)
            conn.execute("""
                UPDATE game_weather SET wind_speed_forecast_mph = 0.0
                WHERE home_team = ? AND game_date = ?
            """, (home_team, game_date))
            dome_count += 1
            continue
        year = game_date[:4]
        team_year_games[(home_team, year)].append(game_date)

    if dome_count > 0:
        conn.commit()
        log.info("  Set %d dome rows to 0.0 mph forecast wind", dome_count)

    log.info("  %d outdoor (team, year) batches to fetch via API",
             len(team_year_games))

    total_written = 0
    total_fail = 0
    batch_num = 0

    for (home_team, year), game_dates in sorted(team_year_games.items()):
        batch_num += 1
        stadium_info = _find_stadium(home_team, "mlb")
        if not stadium_info:
            log.warning("  No stadium for %s — skipping %d games",
                        home_team, len(game_dates))
            total_fail += len(game_dates)
            continue

        lat, lon, stadium_name = stadium_info

        min_date = min(game_dates)
        max_date = max(game_dates)

        # Clamp to yesterday (API might not have today)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if max_date >= today:
            max_date = today

        # Previous Runs API has a max range per request (~3 months recommended)
        # Split into quarterly chunks if needed
        from datetime import timedelta
        import datetime as dt

        start = dt.date.fromisoformat(min_date)
        end = dt.date.fromisoformat(max_date)
        chunk_size = timedelta(days=90)

        all_weather = {}
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + chunk_size, end)
            weather_data = _fetch_forecast_wind(
                lat, lon,
                chunk_start.isoformat(),
                chunk_end.isoformat(),
            )
            if isinstance(weather_data, dict) and weather_data.get("error"):
                log.warning("  API fail: %s %s chunk %s–%s — %s",
                            home_team, year, chunk_start, chunk_end,
                            weather_data["error"])
                total_fail += len([d for d in game_dates
                                   if chunk_start.isoformat() <= d <= chunk_end.isoformat()])
            else:
                all_weather.update(weather_data)
            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(0.3)

        # Update matched rows
        matched = 0
        for gd in game_dates:
            wind_mph = all_weather.get(gd)
            if wind_mph is not None:
                conn.execute("""
                    UPDATE game_weather SET wind_speed_forecast_mph = ?
                    WHERE home_team = ? AND game_date = ?
                """, (wind_mph, home_team, gd))
                matched += 1

        conn.commit()
        total_written += matched

        if batch_num % 10 == 0:
            log.info("  batch %d/%d: %s %s — %d/%d matched",
                     batch_num, len(team_year_games), home_team, year,
                     matched, len(game_dates))

    # Summary
    filled = conn.execute("""
        SELECT COUNT(*) FROM game_weather
        WHERE wind_speed_forecast_mph IS NOT NULL
    """).fetchone()[0]
    total_rows = conn.execute("SELECT COUNT(*) FROM game_weather").fetchone()[0]

    # Compare forecast vs observed for sanity
    comparison = conn.execute("""
        SELECT AVG(wind_speed_mph), AVG(wind_speed_forecast_mph),
               AVG(ABS(wind_speed_mph - wind_speed_forecast_mph))
        FROM game_weather
        WHERE wind_speed_forecast_mph IS NOT NULL
          AND wind_speed_mph IS NOT NULL
          AND is_dome = 0
    """).fetchone()

    log.info("Forecast wind backfill complete:")
    log.info("  Written: %d, Failed: %d, Domes: %d", total_written, total_fail, dome_count)
    log.info("  Coverage: %d/%d rows have forecast wind (%.1f%%)",
             filled, total_rows, 100 * filled / total_rows if total_rows else 0)
    if comparison[0]:
        log.info("  Observed avg: %.1f mph, Forecast avg: %.1f mph, MAE: %.1f mph",
                 comparison[0], comparison[1], comparison[2])

    conn.close()


if __name__ == "__main__":
    backfill()
