"""
data_fetcher.py — Historical game data fetchers for BetIQ Phase 5.

Each public function returns a normalised pandas DataFrame.  No side-effects;
all writing is handled by historical_etl.py.

Sources
-------
  NBA         nba-api   (stats.nba.com)
  NFL         nfl-data-py (nflverse GitHub parquet files)
  MLB         pybaseball  (Baseball Reference)
  EPL/LaLiga/Bundesliga/Ligue1/SerieA   football-data.org (REST, key from config)
  UCL         football-data.org  (REST, key from config)

Note on soccerdata / FBref
--------------------------
  soccerdata is installed and would provide richer per-match stats (xG, shots,
  possession) but fbref.com currently returns 403 for automated requests.
  Switch fetch_soccer_games / fetch_soccer_team_stats back to the FBref path
  once fbref access is restored; the _FBREF_LEAGUE_MAP dict is kept below.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Iterable

import pandas as pd
import requests

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    FOOTBALL_DATA_KEY,
    FOOTBALL_DATA_BASE,
    HISTORICAL_SEASONS,
)

log = logging.getLogger(__name__)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _iso(val) -> str | None:
    """Return ISO-8601 date string or None."""
    if val is None or (isinstance(val, float) and val != val):
        return None
    try:
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return str(val)


# ── football-data.org shared helpers ──────────────────────────────────────────
# Used by both domestic leagues (EPL/LaLiga/Bundesliga/Ligue1/SerieA) and UCL.

# season param = calendar year the season STARTS
# "2022-23" → 2022,  "2023-24" → 2023,  "2024-25" → 2024
def _fdo_season_year(season_str: str) -> int:
    return int(season_str.split("-")[0]) if "-" in season_str else int(season_str)

# Alias used in the domestic soccer section
_fdo_domestic_season_year = _fdo_season_year


def _fdo_get(path: str, params: dict | None = None) -> dict | list | None:
    """Single GET against football-data.org with auth header + 3-attempt retry."""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url     = f"{FOOTBALL_DATA_BASE}{path}"
    for attempt in range(3):
        try:
            time.sleep(6.5)   # free tier: 10 req/min → ~6 s spacing
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code == 429:
                log.warning("football-data.org 429 — waiting 60 s")
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            log.warning("football-data.org attempt %d: %s", attempt + 1, exc)
            time.sleep(10 * (attempt + 1))
    return None


def _fetch_fdo_matches(competition_code: str, season_year: int) -> list[dict]:
    """Fetch all matches for one competition + season from football-data.org."""
    data = _fdo_get(f"/competitions/{competition_code}/matches",
                    {"season": season_year})
    if not data or "matches" not in data:
        return []
    return data["matches"]


def _fdo_match_to_row(m: dict, league_key: str, season_str: str) -> dict:
    """Normalise a football-data.org match dict to the shared games schema."""
    home = (m.get("homeTeam") or {}).get("shortName") or (m.get("homeTeam") or {}).get("name", "")
    away = (m.get("awayTeam") or {}).get("shortName") or (m.get("awayTeam") or {}).get("name", "")
    score = m.get("score") or {}
    ft    = score.get("fullTime") or {}
    hs    = _safe_int(ft.get("home"))
    as_   = _safe_int(ft.get("away"))
    date  = _iso(m.get("utcDate"))
    status_raw = m.get("status", "")
    return {
        "game_id":    f"{league_key}_{season_str}_{m.get('id', '')}",
        "sport":      league_key,
        "season":     season_str,
        "game_date":  date,
        "home_team":  home,
        "away_team":  away,
        "home_score": hs,
        "away_score": as_,
        "status":     "Final" if status_raw == "FINISHED" else status_raw,
        "source":     "football-data.org",
    }


# ══════════════════════════════════════════════════════════════════════════════
# NBA  (nba-api → stats.nba.com)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_nba_games(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Return one row per unique game (home-team perspective) for every
    requested season.

    seasons : list of strings like ["2022-23", "2023-24", "2024-25"]
              defaults to HISTORICAL_SEASONS["NBA"]
    """
    from nba_api.stats.endpoints import leaguegamefinder

    if seasons is None:
        seasons = HISTORICAL_SEASONS["NBA"]

    frames = []
    for season in seasons:
        log.info("NBA games: fetching %s", season)
        try:
            time.sleep(0.6)   # respect stats.nba.com rate limit
            finder = leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            df = finder.get_data_frames()[0]

            # Keep only home-side rows (MATCHUP contains "vs.")
            home = df[df["MATCHUP"].str.contains("vs\\.", na=False)].copy()
            away = df[df["MATCHUP"].str.contains("@", na=False)].copy()

            # Build normalised game rows
            home = home.rename(columns={
                "GAME_ID":           "game_id_raw",
                "TEAM_ABBREVIATION": "home_team",
                "GAME_DATE":         "game_date",
                "WL":                "home_result",
                "PTS":               "home_score",
            })
            away = away[["GAME_ID", "TEAM_ABBREVIATION", "PTS", "WL"]].rename(columns={
                "GAME_ID":           "game_id_raw",
                "TEAM_ABBREVIATION": "away_team",
                "PTS":               "away_score",
                "WL":                "away_result",
            })

            merged = home.merge(away, on="game_id_raw", how="inner")
            merged["sport"]    = "NBA"
            merged["season"]   = season
            merged["game_id"]  = "NBA_" + merged["game_id_raw"].astype(str)
            merged["game_date"]= merged["game_date"].apply(_iso)
            merged["status"]   = "Final"
            merged["source"]   = "nba-api"

            frames.append(merged[[
                "game_id", "sport", "season", "game_date",
                "home_team", "away_team", "home_score", "away_score",
                "status", "source",
            ]])
        except Exception as exc:
            log.error("NBA games %s failed: %s", season, exc)

    return pd.concat(frames, ignore_index=True) if frames else _empty_games()


def fetch_nba_spreads(
    seasons: list[int] | None = None,
    date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch NBA final game scores from BallDontLie API.

    seasons : BDL season ints — 2024 = 2024-25 season, 2025 = 2025-26 season.
              Defaults to [2024, 2025].
    date    : optional ISO date "YYYY-MM-DD"; if given, only return games
              played on that date (ignored if filtering by season is enough).

    Returns one row per completed regular-season game:
        bdl_id | game_date | home_abbr | away_abbr |
        home_full | away_full | home_score | away_score | season
    """
    _BDL_API_KEY = "81b9acae-4bd2-44b6-addc-d493dc4e57bd"
    _BDL_BASE    = "https://api.balldontlie.io/v1"

    if seasons is None:
        seasons = [2024, 2025]

    headers = {"Authorization": _BDL_API_KEY}
    rows: list[dict] = []

    def _fetch_page(params: dict) -> dict | None:
        url = f"{_BDL_BASE}/games"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=20)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                log.warning("BDL attempt %d: %s", attempt + 1, exc)
                time.sleep(5 * (attempt + 1))
        return None

    for season in seasons:
        log.info("BDL NBA scores: fetching season %d", season)
        cursor = None

        while True:
            params: dict = {
                "seasons[]": season,
                "per_page":  100,
            }
            if date:
                params["dates[]"] = date
            if cursor is not None:
                params["cursor"] = cursor

            data = _fetch_page(params)
            if data is None:
                log.error("BDL season %d: fetch failed after retries", season)
                break

            games_page = data.get("data", [])
            meta       = data.get("meta", {})

            for g in games_page:
                if g.get("status") != "Final":
                    continue
                if g.get("postseason"):
                    continue   # regular season only

                hs = g.get("home_team_score")
                vs = g.get("visitor_team_score")
                if hs is None or vs is None:
                    continue

                ht = g.get("home_team") or {}
                vt = g.get("visitor_team") or {}

                rows.append({
                    "bdl_id":     g["id"],
                    "game_date":  (g.get("date") or "")[:10],   # "YYYY-MM-DD"
                    "home_abbr":  ht.get("abbreviation", ""),
                    "away_abbr":  vt.get("abbreviation", ""),
                    "home_full":  ht.get("full_name", ""),
                    "away_full":  vt.get("full_name", ""),
                    "home_score": int(hs),
                    "away_score": int(vs),
                    "season":     season,
                })

            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor
            time.sleep(0.35)   # BDL free tier: ≤60 req/min

    if not rows:
        return pd.DataFrame(columns=[
            "bdl_id", "game_date", "home_abbr", "away_abbr",
            "home_full", "away_full", "home_score", "away_score", "season",
        ])
    return pd.DataFrame(rows)


def fetch_nba_moneylines(
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch NBA closing moneylines from ESPN Core API.

    Uses the same ESPN scoreboard + odds pipeline as backfill_lines.py but returns
    a DataFrame for inspection / pipeline use.  Actual DB writes happen in
    backfill_lines.backfill_nba_espn().

    seasons : nba-api style strings e.g. ["2022-23", "2023-24", "2024-25"].
              Defaults to HISTORICAL_SEASONS["NBA"].

    Returns one row per game with odds:
        game_date | home_abbr | away_abbr | espn_event_id |
        spread | over_under | ml_home | ml_away | season
    """
    import sqlite3 as _sqlite3
    from backfill_lines import (
        _build_date_event_map, _fetch_espn_odds, ESPN_TO_BDL, _DELAY_ODDS,
    )

    if seasons is None:
        seasons = HISTORICAL_SEASONS["NBA"]

    _DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
    conn = _sqlite3.connect(os.path.abspath(_DB))
    cur  = conn.cursor()

    placeholders = ",".join("?" * len(seasons))
    cur.execute(f"""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team, g.season
        FROM games g
        JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.sport  = 'NBA'
          AND g.status = 'Final'
          AND g.season IN ({placeholders})
          AND bl.close_ml_home IS NULL
        ORDER BY g.game_date
    """, seasons)
    db_games = cur.fetchall()
    conn.close()

    if not db_games:
        log.info("fetch_nba_moneylines: no games with missing ML for seasons %s", seasons)
        return pd.DataFrame(columns=[
            "game_id", "game_date", "home_abbr", "away_abbr",
            "espn_event_id", "spread", "over_under", "ml_home", "ml_away", "season",
        ])

    unique_dates = sorted(set(r[1] for r in db_games))
    log.info("fetch_nba_moneylines: %d games across %d dates", len(db_games), len(unique_dates))

    event_map = _build_date_event_map(unique_dates)
    rows: list[dict] = []

    for game_id, game_date, home_team, away_team, season in db_games:
        event_id = event_map.get((game_date, home_team))
        if not event_id:
            continue
        time.sleep(_DELAY_ODDS)
        odds = _fetch_espn_odds(event_id)
        if not odds:
            continue
        rows.append({
            "game_id":       game_id,
            "game_date":     game_date,
            "home_abbr":     home_team,
            "away_abbr":     away_team,
            "espn_event_id": event_id,
            "spread":        odds.get("spread"),
            "over_under":    odds.get("over_under"),
            "ml_home":       odds.get("ml_home"),
            "ml_away":       odds.get("ml_away"),
            "season":        season,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "game_id", "game_date", "home_abbr", "away_abbr",
        "espn_event_id", "spread", "over_under", "ml_home", "ml_away", "season",
    ])
    log.info("fetch_nba_moneylines: fetched ML for %d/%d games", len(df), len(db_games))
    return df


def fetch_nba_game_logs(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Fetch all player-game appearances for given seasons using nba-api LeagueGameLog.
    One API call per season (not per game).

    Returns one row per (player, game) with:
        PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GAME_ID, GAME_DATE,
        MIN (float, minutes played), season (str)
    Only includes players who actually logged minutes (DNP/inactive excluded).
    """
    from nba_api.stats.endpoints import leaguegamelog

    if seasons is None:
        seasons = HISTORICAL_SEASONS["NBA"]

    frames = []
    for season in seasons:
        log.info("NBA game log: fetching %s", season)
        try:
            time.sleep(0.6)
            lg = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
            )
            df = lg.get_data_frames()[0]
            df["season"] = season
            # Ensure MIN is numeric (can be stored as float or int from nba-api)
            df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0)
            df["GAME_DATE"] = df["GAME_DATE"].apply(_iso)
            frames.append(df[[
                "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
                "GAME_ID", "GAME_DATE", "MIN", "season",
            ]])
        except Exception as exc:
            log.error("NBA game log %s failed: %s", season, exc)

    if not frames:
        return pd.DataFrame(columns=[
            "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
            "GAME_ID", "GAME_DATE", "MIN", "season",
        ])
    return pd.concat(frames, ignore_index=True)


def compute_nba_injury_flags(
    game_log_df: pd.DataFrame,
    star_min_thresh: float = 20.0,
    rotation_min_thresh: float = 15.0,
    min_games_for_star: int = 15,
) -> pd.DataFrame:
    """
    Derive per-game injury flags from a LeagueGameLog DataFrame.

    A player is a 'star' for a team-season if:
      - They averaged >= star_min_thresh minutes across their appearances for that team
      - They appeared in >= min_games_for_star games for that team

    A player is a 'rotation' player if they averaged >= rotation_min_thresh but < star_min_thresh.

    For each game, a star/rotation player is considered 'out' if:
      - They played for this team in at least one of the 20 games immediately preceding this game
        (i.e., they were on the active roster recently)
      - AND they do NOT appear in this game's box score

    Returns DataFrame with one row per (GAME_ID, team side perspective):
        nba_game_id: '0022400XXX' format
        team_abbr: team abbreviation
        stars_missing: count of star players absent
        rotation_missing: count of rotation players absent
        star_out: 1 if any star is absent, else 0
    """
    if game_log_df.empty:
        return pd.DataFrame(columns=[
            "nba_game_id", "team_abbr",
            "stars_missing", "rotation_missing", "star_out",
        ])

    df = game_log_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # ── 1. Compute per-player season averages per team ─────────────────────────
    player_season = (
        df.groupby(["PLAYER_ID", "TEAM_ABBREVIATION", "season"])
        .agg(avg_min=("MIN", "mean"), games_played=("GAME_ID", "count"))
        .reset_index()
    )

    stars = player_season[
        (player_season["avg_min"] >= star_min_thresh) &
        (player_season["games_played"] >= min_games_for_star)
    ][["PLAYER_ID", "TEAM_ABBREVIATION", "season"]].copy()
    stars["player_class"] = "star"

    rotation = player_season[
        (player_season["avg_min"] >= rotation_min_thresh) &
        (player_season["avg_min"] < star_min_thresh) &
        (player_season["games_played"] >= min_games_for_star)
    ][["PLAYER_ID", "TEAM_ABBREVIATION", "season"]].copy()
    rotation["player_class"] = "rotation"

    key_players = pd.concat([stars, rotation], ignore_index=True)

    if key_players.empty:
        return pd.DataFrame(columns=[
            "nba_game_id", "team_abbr",
            "stars_missing", "rotation_missing", "star_out",
        ])

    # ── 2. For each game×team, determine which key players were active recently ─
    # Sort game log by date
    df_sorted = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]).copy()

    # Build: (player, team, season) → sorted list of game dates they appeared
    appear_index: dict = {}
    for _, row in df_sorted.iterrows():
        key = (row["PLAYER_ID"], row["TEAM_ABBREVIATION"], row["season"])
        appear_index.setdefault(key, []).append(row["GAME_DATE"])

    # Build: (game_id, team) → set of player_ids who played
    game_team_players: dict = {}
    for _, row in df.iterrows():
        gt_key = (row["GAME_ID"], row["TEAM_ABBREVIATION"])
        game_team_players.setdefault(gt_key, set()).add(row["PLAYER_ID"])

    # ── 3. For each unique (game, team), count absences ────────────────────────
    game_meta = (
        df.drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION", "season"])
          [["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "season"]]
    )

    rows_out = []
    LOOKBACK = 20   # require player appeared in last 20 team games to be "expected"

    for _, gm in game_meta.iterrows():
        gid   = gm["GAME_ID"]
        team  = gm["TEAM_ABBREVIATION"]
        gdate = gm["GAME_DATE"]
        seas  = gm["season"]

        played_in_game = game_team_players.get((gid, team), set())

        # Key players for this team-season
        team_keys = key_players[
            (key_players["TEAM_ABBREVIATION"] == team) &
            (key_players["season"] == seas)
        ]

        stars_missing = rotation_missing = 0
        for _, kp in team_keys.iterrows():
            pid   = kp["PLAYER_ID"]
            pclass = kp["player_class"]
            # Check: did this player appear in any of the team's last LOOKBACK games?
            key = (pid, team, seas)
            appearances = appear_index.get(key, [])
            prior = [d for d in appearances if d < gdate]
            if len(prior) == 0:
                continue  # player hadn't yet appeared for this team this season
            if len(prior) < 5:
                continue  # too few games to be "expected"
            recent = prior[-LOOKBACK:]
            recent_games_count = len(recent)
            if recent_games_count < 3:
                continue  # not established yet

            if pid not in played_in_game:
                if pclass == "star":
                    stars_missing += 1
                else:
                    rotation_missing += 1

        rows_out.append({
            "nba_game_id":      gid,
            "team_abbr":        team,
            "stars_missing":    stars_missing,
            "rotation_missing": rotation_missing,
            "star_out":         int(stars_missing > 0),
        })

    return pd.DataFrame(rows_out)


def fetch_nba_team_stats(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Return per-game team box-score stats (both home and away rows per game).
    Stat columns are encoded as JSON in stats_json for flexibility.
    """
    from nba_api.stats.endpoints import leaguegamefinder

    if seasons is None:
        seasons = HISTORICAL_SEASONS["NBA"]

    STAT_COLS = [
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
    ]

    frames = []
    for season in seasons:
        log.info("NBA team stats: fetching %s", season)
        try:
            time.sleep(0.6)
            finder = leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            df = finder.get_data_frames()[0]

            # Build per-game score lookup: game_id → {team_abbr: pts}
            # Used to populate opp_score without a separate DB query.
            game_score_map: dict[str, dict] = {}
            for _, r in df.iterrows():
                gid  = "NBA_" + str(r["GAME_ID"])
                team = str(r["TEAM_ABBREVIATION"])
                pts  = _safe_int(r.get("PTS"))
                game_score_map.setdefault(gid, {})[team] = pts

            rows = []
            for _, r in df.iterrows():
                stats   = {c: r.get(c) for c in STAT_COLS if c in df.columns}
                gid     = "NBA_" + str(r["GAME_ID"])
                my_team = str(r["TEAM_ABBREVIATION"])
                all_scores = game_score_map.get(gid, {})
                opp_entries = {t: s for t, s in all_scores.items() if t != my_team}
                # Exactly one opponent per game
                opp_score = next(iter(opp_entries.values()), None)
                rows.append({
                    "game_id":    gid,
                    "sport":      "NBA",
                    "season":     season,
                    "team":       my_team,
                    "is_home":    int("vs." in str(r.get("MATCHUP", ""))),
                    "score":      _safe_int(r.get("PTS")),
                    "opp_score":  opp_score,
                    "result":     r.get("WL"),
                    "stats_json": json.dumps(stats),
                })
            frames.append(pd.DataFrame(rows))
        except Exception as exc:
            log.error("NBA team stats %s failed: %s", season, exc)

    return pd.concat(frames, ignore_index=True) if frames else _empty_team_stats()


# ══════════════════════════════════════════════════════════════════════════════
# NFL  (nfl-data-py → nflverse GitHub parquet files)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_nfl_games(years: Iterable[int] | None = None) -> pd.DataFrame:
    """
    One row per game.  nfl-data-py import_schedules already includes
    closing spread/total/moneyline lines, which get written to betting_lines.
    """
    import nfl_data_py as nfl

    if years is None:
        years = HISTORICAL_SEASONS["NFL"]

    log.info("NFL games: fetching years %s", list(years))
    try:
        df = nfl.import_schedules(list(years))
    except Exception as exc:
        log.error("NFL import_schedules failed: %s", exc)
        return _empty_games()

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "game_id":    "NFL_" + str(r["game_id"]),
            "sport":      "NFL",
            "season":     str(r.get("season", "")),
            "game_date":  _iso(r.get("gameday")),
            "home_team":  r.get("home_team"),
            "away_team":  r.get("away_team"),
            "home_score": _safe_int(r.get("home_score")),
            "away_score": _safe_int(r.get("away_score")),
            "status":     "Final" if pd.notna(r.get("home_score")) else "Scheduled",
            "source":     "nfl-data-py",
            # Pass-through betting lines (close only — no open available)
            # nfl_data_py stores spread_line from the AWAY team's perspective
            # (positive = away is underdog). We negate it here so close_spread
            # is always stored from the HOME team's perspective throughout the
            # codebase (negative = home favoured).
            "_spread_line":    -r["spread_line"] if pd.notna(r.get("spread_line")) else None,
            "_total_line":        r.get("total_line"),
            "_home_moneyline":    r.get("home_moneyline"),
            "_away_moneyline":    r.get("away_moneyline"),
        })

    return pd.DataFrame(rows) if rows else _empty_games()


def fetch_nfl_team_stats(years: Iterable[int] | None = None) -> pd.DataFrame:
    """
    Weekly offensive stats per player aggregated to team level via
    import_weekly_data.  Returns one row per team per week with key stats.
    """
    import nfl_data_py as nfl

    if years is None:
        years = HISTORICAL_SEASONS["NFL"]

    log.info("NFL team stats: fetching years %s", list(years))
    try:
        df = nfl.import_schedules(list(years))
    except Exception as exc:
        log.error("NFL team stats failed: %s", exc)
        return _empty_team_stats()

    rows = []
    for _, r in df.iterrows():
        base = {
            "sport":  "NFL",
            "season": str(r.get("season", "")),
            "game_id": "NFL_" + str(r["game_id"]),
        }
        for side, opp in (("home", "away"), ("away", "home")):
            stats = {
                "spread_line":   r.get("spread_line"),
                "total_line":    r.get("total_line"),
                "moneyline":     r.get(f"{side}_moneyline"),
                "rest_days":     r.get(f"{side}_rest"),
                "result":        r.get("result"),
                "overtime":      r.get("overtime"),
                "roof":          r.get("roof"),
                "surface":       r.get("surface"),
                "temp":          r.get("temp"),
                "wind":          r.get("wind"),
            }
            score     = _safe_int(r.get(f"{side}_score"))
            opp_score = _safe_int(r.get(f"{opp}_score"))
            result    = None
            if score is not None and opp_score is not None:
                result = "W" if score > opp_score else ("T" if score == opp_score else "L")
            rows.append({
                **base,
                "team":       r.get(f"{side}_team"),
                "is_home":    int(side == "home"),
                "score":      score,
                "opp_score":  opp_score,
                "result":     result,
                "stats_json": json.dumps(stats),
            })

    return pd.DataFrame(rows) if rows else _empty_team_stats()


# ══════════════════════════════════════════════════════════════════════════════
# MLB  (mlb-statsapi → MLB Stats API, official, no key required)
#
# Replaces pybaseball schedule_and_record (Baseball Reference scraping,
# rate-limited) and pybaseball FanGraphs endpoints (403).
# mlb-statsapi hits the official MLB Stats API — no auth, no scraping.
# ══════════════════════════════════════════════════════════════════════════════

# Regular-season date windows per calendar year
_MLB_SEASON_DATES = {
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-30"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-27", "2026-09-27"),   # 2026 regular season
}

# statsapi.schedule() hardcodes a heavy hydrate string that includes
# broadcasts and game(content(media(epg))), causing 503 timeouts for older
# completed seasons. For seasons <= 2023 we call statsapi.get("schedule")
# directly with a minimal hydrate — scores, decisions, probable pitchers,
# and series status only.
_MLB_HYDRATE_FULL  = "decisions,probablePitcher(note),linescore,seriesStatus"
_MLB_HYDRATE_LIGHT = "linescore,decisions,probablePitcher(note),seriesStatus"


def _mlb_schedule_raw(start: str, end: str, season: int) -> list[dict]:
    """
    Return game-dicts equivalent to statsapi.schedule() output, but with a
    minimal hydrate for seasons <= 2023 to avoid 503/500 server errors.
    Older seasons are fetched in monthly chunks to keep payload size small.
    Falls back to full statsapi.schedule() for 2024+.
    """
    import statsapi
    from datetime import date, timedelta

    if int(season) >= 2024:
        return statsapi.schedule(start_date=start, end_date=end, sportId=1)

    def _parse_chunk(r: dict) -> list[dict]:
        games = []
        for date_obj in r.get("dates", []):
            for game in date_obj.get("games", []):
                home = game["teams"]["home"]
                away = game["teams"]["away"]
                games.append({
                    "game_id":                game["gamePk"],
                    "game_date":              date_obj["date"],
                    "game_type":              game["gameType"],
                    "status":                 game["status"]["detailedState"],
                    "home_name":              home["team"].get("name", ""),
                    "away_name":              away["team"].get("name", ""),
                    "home_id":                home["team"]["id"],
                    "away_id":                away["team"]["id"],
                    "home_score":             home.get("score"),
                    "away_score":             away.get("score"),
                    "doubleheader":           game.get("doubleHeader"),
                    "venue_name":             game.get("venue", {}).get("name"),
                    "winning_pitcher":        game.get("decisions", {}).get("winner", {}).get("fullName"),
                    "losing_pitcher":         game.get("decisions", {}).get("loser",  {}).get("fullName"),
                    "save_pitcher":           game.get("decisions", {}).get("save",   {}).get("fullName"),
                    "series_status":          game.get("seriesStatus", {}).get("description"),
                    "home_probable_pitcher":  home.get("probablePitcher", {}).get("fullName", ""),
                    "away_probable_pitcher":  away.get("probablePitcher", {}).get("fullName", ""),
                })
        return games

    # Chunk by month to keep individual requests small
    all_games: list[dict] = []
    chunk_start = date.fromisoformat(start)
    season_end  = date.fromisoformat(end)

    while chunk_start <= season_end:
        # End of month or season end, whichever is first
        if chunk_start.month == 12:
            next_month = chunk_start.replace(year=chunk_start.year + 1, month=1, day=1)
        else:
            next_month = chunk_start.replace(month=chunk_start.month + 1, day=1)
        chunk_end = min(next_month - timedelta(days=1), season_end)

        params = {
            "startDate": chunk_start.isoformat(),
            "endDate":   chunk_end.isoformat(),
            "sportId":   "1",
            "hydrate":   _MLB_HYDRATE_LIGHT,
        }
        try:
            r = statsapi.get("schedule", params)
            all_games.extend(_parse_chunk(r))
        except Exception as exc:
            log.warning("MLB chunk %s→%s failed: %s", chunk_start, chunk_end, exc)

        chunk_start = next_month

    return all_games


def fetch_mlb_games(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """
    Fetch all regular-season MLB games via the official MLB Stats API
    (mlb-statsapi).  Returns one row per game with home/away scores.
    Single bulk call per season — no per-team loops, no scraping.
    """
    import statsapi

    if seasons is None:
        seasons = HISTORICAL_SEASONS["MLB"]

    rows = []
    for season in seasons:
        start, end = _MLB_SEASON_DATES.get(int(season), (f"{season}-03-28", f"{season}-10-01"))
        log.info("MLB games: %s  (%s → %s)", season, start, end)
        try:
            games = _mlb_schedule_raw(start, end, season)
            # Keep only regular-season final games (game_type='R')
            final = [g for g in games if g.get("game_type") == "R"]
            log.info("  MLB %s: %d regular-season games", season, len(final))
            for g in final:
                rows.append({
                    "game_id":    f"MLB_{g['game_id']}",
                    "sport":      "MLB",
                    "season":     str(season),
                    "game_date":  g.get("game_date"),
                    "home_team":  g.get("home_name", ""),
                    "away_team":  g.get("away_name", ""),
                    "home_score": _safe_int(g.get("home_score")),
                    "away_score": _safe_int(g.get("away_score")),
                    "status":     g.get("status", ""),
                    "source":     "mlb-statsapi",
                })
        except Exception as exc:
            log.error("MLB games %s: %s", season, exc)

    return pd.DataFrame(rows) if rows else _empty_games()


def fetch_mlb_team_stats(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """
    Build per-game, per-team rows from MLB Stats API game data.
    Stats available on free tier: score, result, venue, winning/losing pitcher.
    stats_json holds the full game metadata for feature engineering.
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS["MLB"]

    rows = []
    for season in seasons:
        start, end = _MLB_SEASON_DATES.get(int(season), (f"{season}-03-28", f"{season}-10-01"))
        log.info("MLB team stats: %s", season)
        try:
            games = _mlb_schedule_raw(start, end, season)
            final = [g for g in games if g.get("game_type") == "R"]
            for g in final:
                gid = f"MLB_{g['game_id']}"
                meta = {
                    "venue":                   g.get("venue_name"),
                    "doubleheader":            g.get("doubleheader"),
                    "winning_pitcher":         g.get("winning_pitcher"),
                    "losing_pitcher":          g.get("losing_pitcher"),
                    "save_pitcher":            g.get("save_pitcher"),
                    "series_status":           g.get("series_status"),
                    "home_probable_pitcher":   g.get("home_probable_pitcher", ""),
                    "away_probable_pitcher":   g.get("away_probable_pitcher", ""),
                }
                hs = _safe_int(g.get("home_score"))
                as_ = _safe_int(g.get("away_score"))
                for side, opp_side, team_key in (
                    ("home", "away", "home_name"),
                    ("away", "home", "away_name"),
                ):
                    score     = hs if side == "home" else as_
                    opp_score = as_ if side == "home" else hs
                    result    = None
                    if score is not None and opp_score is not None:
                        result = "W" if score > opp_score else ("T" if score == opp_score else "L")
                    rows.append({
                        "game_id":    gid,
                        "sport":      "MLB",
                        "season":     str(season),
                        "team":       g.get(team_key, ""),
                        "is_home":    int(side == "home"),
                        "score":      score,
                        "opp_score":  opp_score,
                        "result":     result,
                        "stats_json": json.dumps(meta),
                    })
        except Exception as exc:
            log.error("MLB team stats %s: %s", season, exc)

    return pd.DataFrame(rows) if rows else _empty_team_stats()


def fetch_mlb_todays_probable_pitchers(date_str: str | None = None) -> list[dict]:
    """
    Fetch today's MLB probable pitchers from the free MLB Stats API.

    Endpoint: GET https://statsapi.mlb.com/api/v1/schedule
      ?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher

    Returns a list of dicts, one per game:
        {
            "game_pk":              int,
            "game_date":            str,       # YYYY-MM-DD
            "home_team":            str,
            "away_team":            str,
            "home_probable_pitcher": str | None,
            "away_probable_pitcher": str | None,
        }

    Returns [] on failure (API down, no games, etc.).
    Network timeout: 10 seconds.
    """
    import requests
    from datetime import date

    if date_str is None:
        date_str = date.today().isoformat()

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId":  "1",
        "date":     date_str,
        "hydrate":  "probablePitcher",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("fetch_mlb_todays_probable_pitchers: %s", exc)
        return []

    games = []
    for date_obj in data.get("dates", []):
        for game in date_obj.get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            games.append({
                "game_pk":               game.get("gamePk"),
                "game_date":             date_obj.get("date", date_str),
                "home_team":             home.get("team", {}).get("name", ""),
                "away_team":             away.get("team", {}).get("name", ""),
                "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName") or None,
                "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName") or None,
            })
    return games


def fetch_mlb_pitcher_stats(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """
    Fetch season-level pitching stats for all MLB pitchers via the official
    Stats API bulk endpoint.  Returns one row per (pitcher_name, season) with
    ERA, WHIP, K/9, BB/9, FIP, games started, and innings pitched.

    Used by features.py to look up starting pitcher quality at game-time.
    Only includes pitchers with >= 5 games started to filter relievers.
    """
    import statsapi

    if seasons is None:
        seasons = HISTORICAL_SEASONS["MLB"]

    rows = []
    for season in seasons:
        log.info("MLB pitcher stats: %s", season)
        try:
            r = statsapi.get("stats", {
                "stats":      "season",
                "group":      "pitching",
                "season":     str(season),
                "sportId":    "1",
                "playerPool": "All",
                "limit":      "2000",
            })
            splits = []
            for stat_obj in r.get("stats", []):
                splits.extend(stat_obj.get("splits", []))

            for s in splits:
                st      = s.get("stat", {})
                player  = s.get("player", {})
                name    = player.get("fullName", "")
                pid     = player.get("id")          # MLB player ID (int)
                gs      = int(st.get("gamesStarted", 0) or 0)
                if gs < 5 or not name:
                    continue
                ip_raw = st.get("inningsPitched", "0") or "0"
                try:
                    ip = float(ip_raw)
                except ValueError:
                    ip = 0.0
                rows.append({
                    "pitcher_name": name,
                    "player_id":    pid,
                    "season":       str(season),
                    "era":          float(st.get("era",  99.0) or 99.0),
                    "whip":         float(st.get("whip",  9.9) or 9.9),
                    "k9":           float(st.get("strikeOutsPer9Inn", 0.0) or 0.0),
                    "bb9":          float(st.get("walksPer9Inn", 0.0) or 0.0),
                    "fip":          float(st.get("fieldingIndependentPitching", 99.0) or 99.0),
                    "gs":           gs,
                    "ip":           ip,
                })
            log.info("  MLB pitcher stats %s: %d starters", season, len([r for r in rows if r["season"] == str(season)]))
        except Exception as exc:
            log.error("MLB pitcher stats %s: %s", season, exc)

    if not rows:
        return pd.DataFrame(columns=["pitcher_name","player_id","season","era","whip","k9","bb9","fip","gs","ip"])
    return pd.DataFrame(rows)


def fetch_mlb_pitcher_game_logs(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """
    Fetch per-start pitching logs for all qualifying starters (gs >= 5) across
    the given seasons.  Returns one row per start with:
        pitcher_name, player_id, season, game_date, game_pk,
        ip, er, hits, walks, strikeouts, gs_flag

    Used by features.py to compute rolling ERA/WHIP over the last 3 or 5
    starts before each game — no lookahead bias since we always use only
    starts that occurred *before* the game date in question.

    API: GET /api/v1/people/{id}/stats?stats=gameLog&group=pitching&season={year}
    One call per (pitcher × season).  With ~250 starters × 4 seasons ≈ 1,000
    calls at 0.25s spacing = ~4 min total.
    """
    import time
    import requests

    if seasons is None:
        seasons = HISTORICAL_SEASONS["MLB"]

    # First get player IDs from season stats
    season_pitcher_ids: dict[str, list[tuple[str, int]]] = {}  # season → [(name, id)]
    for season in seasons:
        try:
            import statsapi
            r = statsapi.get("stats", {
                "stats":      "season",
                "group":      "pitching",
                "season":     str(season),
                "sportId":    "1",
                "playerPool": "All",
                "limit":      "2000",
            })
            ids = []
            for stat_obj in r.get("stats", []):
                for s in stat_obj.get("splits", []):
                    st     = s.get("stat", {})
                    player = s.get("player", {})
                    name   = player.get("fullName", "")
                    pid    = player.get("id")
                    gs     = int(st.get("gamesStarted", 0) or 0)
                    if gs >= 5 and pid and name:
                        ids.append((name, pid))
            season_pitcher_ids[str(season)] = ids
            log.info("MLB game logs: %s — %d qualifying starters", season, len(ids))
        except Exception as exc:
            log.error("MLB game logs player ID fetch %s: %s", season, exc)
            season_pitcher_ids[str(season)] = []

    rows = []
    total_pitchers = sum(len(v) for v in season_pitcher_ids.values())
    done = 0

    for season, pitchers in season_pitcher_ids.items():
        for name, pid in pitchers:
            try:
                url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                resp = requests.get(url, params={
                    "stats":  "gameLog",
                    "group":  "pitching",
                    "season": season,
                }, timeout=20)
                resp.raise_for_status()
                splits = resp.json().get("stats", [{}])[0].get("splits", [])
                for s in splits:
                    st = s.get("stat", {})
                    if not int(st.get("gamesStarted", 0) or 0):
                        continue   # relief appearance — skip
                    ip_raw = st.get("inningsPitched", "0") or "0"
                    try:
                        ip = float(ip_raw)
                    except ValueError:
                        ip = 0.0
                    rows.append({
                        "pitcher_name": name,
                        "player_id":    pid,
                        "season":       season,
                        "game_date":    s.get("date"),
                        "game_pk":      s.get("game", {}).get("gamePk"),
                        "ip":           ip,
                        "er":           int(st.get("earnedRuns", 0) or 0),
                        "hits":         int(st.get("hits",        0) or 0),
                        "walks":        int(st.get("baseOnBalls", 0) or 0),
                        "strikeouts":   int(st.get("strikeOuts",  0) or 0),
                    })
            except Exception as exc:
                log.warning("MLB game log %s %s %s: %s", name, season, pid, exc)

            done += 1
            if done % 100 == 0:
                log.info("  MLB game logs: %d/%d pitchers fetched", done, total_pitchers)
            time.sleep(0.2)   # ~200ms between calls to stay well within rate limits

    if not rows:
        return pd.DataFrame(columns=["pitcher_name","player_id","season","game_date","game_pk","ip","er","hits","walks","strikeouts"])
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def fetch_mlb_team_batting_stats(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """
    Fetch season-level team batting stats (OPS, SLG, OBP, AVG, HR, K%, BB%)
    for all 30 MLB teams across the given seasons.

    Returns one row per (team_name, season).
    """
    import statsapi

    if seasons is None:
        seasons = list(_MLB_SEASON_DATES.keys())

    r = statsapi.get("teams", {"sportId": 1})
    teams = [(t["id"], t["name"]) for t in r["teams"]]

    rows = []
    for season in seasons:
        log.info("MLB team batting stats: %s (%d teams)", season, len(teams))
        for tid, tname in teams:
            try:
                resp = statsapi.get("team_stats", {
                    "teamId": tid,
                    "stats":  "season",
                    "group":  "hitting",
                    "season": season,
                })
                splits = resp.get("stats", [{}])[0].get("splits", [])
                if not splits:
                    continue
                st = splits[0]["stat"]
                pa = int(st.get("plateAppearances") or 1)
                rows.append({
                    "team_name": tname,
                    "season":    str(season),
                    "ops":       float(st.get("ops",  0) or 0),
                    "slg":       float(st.get("slg",  0) or 0),
                    "obp":       float(st.get("obp",  0) or 0),
                    "avg":       float(st.get("avg",  0) or 0),
                    "hr":        int(st.get("homeRuns", 0) or 0),
                    "k_pct":     round(int(st.get("strikeOuts", 0) or 0) / pa, 4),
                    "bb_pct":    round(int(st.get("baseOnBalls", 0) or 0) / pa, 4),
                    "babip":     float(st.get("babip", 0) or 0),
                })
            except Exception as exc:
                log.warning("MLB team batting %s %s: %s", tname, season, exc)
            time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["team_name","season","ops","slg","obp","avg","hr","k_pct","bb_pct","babip"])
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Soccer — domestic leagues (football-data.org REST API)
# EPL, LaLiga, Bundesliga, Ligue1, SerieA — all TIER_ONE on free plan
#
# FBref / soccerdata fallback
# ──────────────────────────
# _FBREF_LEAGUE_MAP is kept here for when fbref.com access is restored.
# To switch back: replace the football-data.org calls below with:
#   reader = sd.FBref(leagues=[_FBREF_LEAGUE_MAP[league_key]], seasons=season_years)
#   sched  = reader.read_schedule()
# ══════════════════════════════════════════════════════════════════════════════

# football-data.org competition codes for the 5 domestic leagues
_FDO_LEAGUE_MAP = {
    "EPL":        "PL",
    "LaLiga":     "PD",
    "Bundesliga": "BL1",
    "Ligue1":     "FL1",
    "SerieA":     "SA",
}

# Kept for future FBref re-integration
_FBREF_LEAGUE_MAP = {
    "EPL":        "ENG-Premier League",
    "LaLiga":     "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Ligue1":     "FRA-Ligue 1",
    "SerieA":     "ITA-Serie A",
}

# Config soccer seasons are "2022-23" etc.
# football-data.org season param = the calendar year the season STARTS
# "2022-23" → 2022,  "2023-24" → 2023,  "2024-25" → 2024
def _fdo_domestic_season_year(season_str: str) -> int:
    return int(season_str.split("-")[0]) if "-" in season_str else int(season_str)


def _fetch_fdo_matches(competition_code: str, season_year: int) -> list[dict]:
    """Single football-data.org matches fetch with rate-limit guard."""
    data = _fdo_get(f"/competitions/{competition_code}/matches",
                    {"season": season_year})
    if not data or "matches" not in data:
        return []
    return data["matches"]


def _fdo_match_to_row(m: dict, league_key: str, season_str: str) -> dict:
    """Normalise a football-data.org match dict to the games schema."""
    home = (m.get("homeTeam") or {}).get("shortName") or (m.get("homeTeam") or {}).get("name", "")
    away = (m.get("awayTeam") or {}).get("shortName") or (m.get("awayTeam") or {}).get("name", "")
    score = m.get("score") or {}
    ft    = score.get("fullTime") or {}
    hs    = _safe_int(ft.get("home"))
    as_   = _safe_int(ft.get("away"))
    date  = _iso(m.get("utcDate"))
    status_raw = m.get("status", "")
    return {
        "game_id":    f"{league_key}_{season_str}_{m.get('id', '')}",
        "sport":      league_key,
        "season":     season_str,
        "game_date":  date,
        "home_team":  home,
        "away_team":  away,
        "home_score": hs,
        "away_score": as_,
        "status":     "Final" if status_raw == "FINISHED" else status_raw,
        "source":     "football-data.org",
    }


def fetch_soccer_games(
    leagues: Iterable[str] | None = None,
    seasons: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch schedule + scores for each domestic league × season via
    football-data.org (TIER_ONE, all 5 leagues accessible on free key).

    leagues : BetIQ keys — ["EPL", "LaLiga", "Bundesliga", "Ligue1", "SerieA"]
              defaults to all five
    seasons : strings like ["2022-23", "2023-24", "2024-25"]
              defaults to HISTORICAL_SEASONS["SOCCER"]["EPL"]
    """
    if leagues is None:
        leagues = list(_FDO_LEAGUE_MAP.keys())
    if seasons is None:
        seasons = HISTORICAL_SEASONS["SOCCER"]["EPL"]

    rows = []
    for league_key in leagues:
        comp_code = _FDO_LEAGUE_MAP.get(league_key)
        if not comp_code:
            log.warning("fetch_soccer_games: unknown league key %s", league_key)
            continue

        for season_str in seasons:
            year = _fdo_domestic_season_year(season_str)
            log.info("Soccer games: %s %s (season year %s)", league_key, season_str, year)
            try:
                matches = _fetch_fdo_matches(comp_code, year)
                for m in matches:
                    rows.append(_fdo_match_to_row(m, league_key, season_str))
                log.info("  %s %s: %d matches", league_key, season_str, len(matches))
            except Exception as exc:
                log.error("Soccer %s %s: %s", league_key, season_str, exc)

    return pd.DataFrame(rows) if rows else _empty_games()


def fetch_soccer_team_stats(
    leagues: Iterable[str] | None = None,
    seasons: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Per-game team rows from football-data.org match data.
    Free tier provides goals and result only; stats_json will hold those
    plus any available match metadata (venue, stage, matchday).

    When FBref access is restored, swap in sd.FBref.read_team_match_stats()
    here for richer stats (xG, shots, possession, etc.).
    """
    if leagues is None:
        leagues = list(_FDO_LEAGUE_MAP.keys())
    if seasons is None:
        seasons = HISTORICAL_SEASONS["SOCCER"]["EPL"]

    rows = []
    for league_key in leagues:
        comp_code = _FDO_LEAGUE_MAP.get(league_key)
        if not comp_code:
            continue

        for season_str in seasons:
            year = _fdo_domestic_season_year(season_str)
            log.info("Soccer team stats: %s %s", league_key, season_str)
            try:
                matches = _fetch_fdo_matches(comp_code, year)
                for m in matches:
                    g = _fdo_match_to_row(m, league_key, season_str)
                    hs, as_ = g["home_score"], g["away_score"]
                    meta = {
                        "matchday": m.get("matchday"),
                        "stage":    m.get("stage"),
                        "venue":    m.get("venue"),
                        "referee":  (m.get("referees") or [{}])[0].get("name"),
                    }
                    for side, opp_side in (("home", "away"), ("away", "home")):
                        score     = g[f"{side}_score"]
                        opp_score = g[f"{opp_side}_score"]
                        result    = None
                        if score is not None and opp_score is not None:
                            result = "W" if score > opp_score else ("D" if score == opp_score else "L")
                        rows.append({
                            "game_id":    g["game_id"],
                            "sport":      league_key,
                            "season":     season_str,
                            "team":       g[f"{side}_team"],
                            "is_home":    int(side == "home"),
                            "score":      score,
                            "opp_score":  opp_score,
                            "result":     result,
                            "stats_json": json.dumps(meta),
                        })
            except Exception as exc:
                log.error("Soccer team stats %s %s: %s", league_key, season_str, exc)

    return pd.DataFrame(rows) if rows else _empty_team_stats()


# ══════════════════════════════════════════════════════════════════════════════
# UCL  (football-data.org REST API, league code "CL")
# ══════════════════════════════════════════════════════════════════════════════
# Shared _fdo_get / _fdo_season_year / _fetch_fdo_matches helpers are defined
# above the NBA section alongside the other football-data.org utilities.

def fetch_ucl_games(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Fetch UEFA Champions League matches from football-data.org (competition CL).

    seasons : list of strings like ["2022-23", "2023-24", "2024-25"]
              defaults to HISTORICAL_SEASONS["SOCCER"]["UCL"]
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS["SOCCER"]["UCL"]

    rows = []
    for season_str in seasons:
        year = _fdo_season_year(season_str)
        log.info("UCL games: season %s (year param %s)", season_str, year)
        try:
            matches = _fetch_fdo_matches("CL", year)
            if not matches:
                log.warning("UCL %s: no data returned", season_str)
                continue
            for m in matches:
                rows.append(_fdo_match_to_row(m, "UCL", season_str))
            log.info("  UCL %s: %d matches", season_str, len(matches))
        except Exception as exc:
            log.error("UCL %s: %s", season_str, exc)

    return pd.DataFrame(rows) if rows else _empty_games()


def fetch_ucl_team_stats(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Build per-game team-level rows from UCL match data (goals, result only —
    football-data.org free tier does not provide advanced match stats).
    """
    games = fetch_ucl_games(seasons)
    if games.empty:
        return _empty_team_stats()

    rows = []
    for _, g in games.iterrows():
        for side, opp_side in (("home", "away"), ("away", "home")):
            score     = g[f"{side}_score"]
            opp_score = g[f"{opp_side}_score"]
            result    = None
            if score is not None and opp_score is not None:
                result = "W" if score > opp_score else ("D" if score == opp_score else "L")
            rows.append({
                "game_id":    g["game_id"],
                "sport":      "UCL",
                "season":     g["season"],
                "team":       g[f"{side}_team"],
                "is_home":    int(side == "home"),
                "score":      score,
                "opp_score":  opp_score,
                "result":     result,
                "stats_json": json.dumps({}),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# NHL  (NHL Web API — api-web.nhle.com, no key, no auth required)
# ══════════════════════════════════════════════════════════════════════════════

# All 32 current NHL franchises.
# Arizona Coyotes relocated → Utah Hockey Club for 2024-25 onward.
_NHL_TEAMS_CURRENT = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]
# Pre-2024-25: ARI instead of UTA
_NHL_TEAMS_PRE_2024 = [t if t != "UTA" else "ARI" for t in _NHL_TEAMS_CURRENT]


def _nhl_season_code(season_str: str) -> str:
    """
    Convert "2021-22" → "20212022", "2024-25" → "20242025".
    Pass-through for 8-digit codes.
    """
    if "-" in season_str:
        start, end_short = season_str.split("-", 1)
        end_full = (start[:2] + end_short) if len(end_short) == 2 else end_short
        return start + end_full
    return season_str


def _nhl_api_get(url: str) -> dict | None:
    """GET NHL Web API with 3-attempt retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "BetIQ/1.0 (research)"})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            log.debug("NHL API HTTP %d: %s", r.status_code, url[-80:])
            return None
        except requests.RequestException as exc:
            log.warning("NHL API error (attempt %d): %s", attempt + 1, exc)
            time.sleep(2)
    return None


def fetch_nhl_games(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Fetch all regular-season NHL games via the NHL Web API (api-web.nhle.com).

    Uses per-team schedule endpoint and deduplicates by game_id so each game
    appears exactly once.  Stores last_period_type ("REG"/"OT"/"SO") in the
    ``_last_period`` column for puck-line OT/SO logic in historical_etl.py.

    seasons: ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
             defaults to HISTORICAL_SEASONS["NHL"]

    Rate limit: 0.5 s between requests (NHL API is lenient).
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS.get("NHL", ["2022-23", "2023-24", "2024-25"])

    seen_ids: set = set()
    rows: list[dict] = []

    for season in seasons:
        season_code = _nhl_season_code(season)
        # Pick the right team list based on season
        start_year = int(season_code[:4])
        teams = _NHL_TEAMS_CURRENT if start_year >= 2024 else _NHL_TEAMS_PRE_2024
        log.info("NHL games: season %s (%s) — %d teams", season, season_code, len(teams))

        for team in teams:
            url = (f"https://api-web.nhle.com/v1/club-schedule-season"
                   f"/{team}/{season_code}")
            time.sleep(0.5)
            data = _nhl_api_get(url)
            if not data:
                log.warning("NHL %s %s: no data returned", team, season)
                continue

            for g in data.get("games", []):
                # gameType 2 = regular season; 3 = playoffs; 1 = preseason
                if g.get("gameType") != 2:
                    continue
                gid_raw = g.get("id")
                if not gid_raw or gid_raw in seen_ids:
                    continue
                seen_ids.add(gid_raw)

                home  = g.get("homeTeam") or {}
                away  = g.get("awayTeam") or {}
                outcome = g.get("gameOutcome") or {}

                home_score = _safe_int(home.get("score"))
                away_score = _safe_int(away.get("score"))
                last_period = (outcome.get("lastPeriodType") or "").upper()

                # Determine completion — NHL API uses gameState "OFF" for final
                game_state = (g.get("gameState") or "").upper()
                is_final   = game_state in ("OFF", "FINAL") or (
                    home_score is not None and away_score is not None
                    and game_state not in ("FUT", "PRE", "LIVE")
                )

                rows.append({
                    "game_id":      f"NHL_{gid_raw}",
                    "sport":        "NHL",
                    "season":       season,
                    "game_date":    _iso(g.get("gameDate")),
                    "home_team":    home.get("abbrev", ""),
                    "away_team":    away.get("abbrev", ""),
                    "home_score":   home_score,
                    "away_score":   away_score,
                    "status":       "Final" if is_final else "Scheduled",
                    "source":       "nhl-web-api",
                    "_last_period": last_period,   # REG / OT / SO — for ot_so_game flag
                })

        season_games = sum(1 for r in rows if r["season"] == season)
        log.info("NHL %s: %d unique games", season, season_games)

    return pd.DataFrame(rows) if rows else _empty_games()


def fetch_nhl_team_stats(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Build per-game, per-team rows from NHL game data.
    Stores ``last_period`` in stats_json for downstream use.
    """
    games_df = fetch_nhl_games(seasons=seasons)
    if games_df.empty:
        return _empty_team_stats()

    rows = []
    for _, g in games_df.iterrows():
        hs  = g["home_score"]
        as_ = g["away_score"]
        lp  = g.get("_last_period", "") or ""
        for side, opp_side in (("home", "away"), ("away", "home")):
            score     = hs  if side == "home" else as_
            opp_score = as_ if side == "home" else hs
            result    = None
            if score is not None and opp_score is not None:
                result = "W" if score > opp_score else ("T" if score == opp_score else "L")
            rows.append({
                "game_id":    g["game_id"],
                "sport":      "NHL",
                "season":     g["season"],
                "team":       g[f"{side}_team"],
                "is_home":    int(side == "home"),
                "score":      score,
                "opp_score":  opp_score,
                "result":     result,
                "stats_json": json.dumps({"last_period": lp}),
            })

    return pd.DataFrame(rows) if rows else _empty_team_stats()


def fetch_nhl_scores_for_date(date_str: str) -> list[dict]:
    """
    Fetch all completed NHL games for *date_str* (ISO "YYYY-MM-DD") via the
    NHL daily schedule endpoint.  Used by the nightly scheduler.

    Returns a list of dicts: {game_id, home_team, away_team, home_score,
    away_score, last_period}
    """
    url  = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    data = _nhl_api_get(url)
    if not data:
        return []

    results = []
    for day in data.get("gameWeek", []):
        if day.get("date") != date_str:
            continue
        for g in day.get("games", []):
            if g.get("gameType") != 2:
                continue
            game_state = (g.get("gameState") or "").upper()
            if game_state not in ("OFF", "FINAL"):
                continue
            home    = g.get("homeTeam") or {}
            away    = g.get("awayTeam") or {}
            outcome = g.get("gameOutcome") or {}
            results.append({
                "game_id":     f"NHL_{g['id']}",
                "home_team":   home.get("abbrev", ""),
                "away_team":   away.get("abbrev", ""),
                "home_score":  _safe_int(home.get("score")),
                "away_score":  _safe_int(away.get("score")),
                "last_period": (outcome.get("lastPeriodType") or "").upper(),
            })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: fetch all sports at once
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_games() -> pd.DataFrame:
    """Aggregate games from every sport. Skips sources that fail."""
    parts = []
    for fn, label in [
        (fetch_nba_games,    "NBA"),
        (fetch_nfl_games,    "NFL"),
        (fetch_mlb_games,    "MLB"),
        (fetch_nhl_games,    "NHL"),
        (fetch_soccer_games, "Soccer"),
        (fetch_ucl_games,    "UCL"),
    ]:
        log.info("fetch_all_games: %s", label)
        try:
            parts.append(fn())
        except Exception as exc:
            log.error("%s fetch failed: %s", label, exc)

    return pd.concat(parts, ignore_index=True) if parts else _empty_games()


def fetch_all_team_stats() -> pd.DataFrame:
    parts = []
    for fn, label in [
        (fetch_nba_team_stats,    "NBA"),
        (fetch_nfl_team_stats,    "NFL"),
        (fetch_mlb_team_stats,    "MLB"),
        (fetch_nhl_team_stats,    "NHL"),
        (fetch_soccer_team_stats, "Soccer"),
        (fetch_ucl_team_stats,    "UCL"),
    ]:
        log.info("fetch_all_team_stats: %s", label)
        try:
            parts.append(fn())
        except Exception as exc:
            log.error("%s team stats failed: %s", label, exc)

    return pd.concat(parts, ignore_index=True) if parts else _empty_team_stats()


# ── Empty-frame sentinels ──────────────────────────────────────────────────────

def _empty_games() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "game_id", "sport", "season", "game_date",
        "home_team", "away_team", "home_score", "away_score",
        "status", "source",
    ])


def _empty_team_stats() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "game_id", "sport", "season", "team",
        "is_home", "score", "opp_score", "result", "stats_json",
    ])
