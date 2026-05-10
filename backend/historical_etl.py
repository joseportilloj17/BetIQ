"""
historical_etl.py — Build and populate historical.db for BetIQ Phase 5.

Schema
------
  games         One row per unique game across all sports / seasons.
  team_stats    Per-game, per-team stats (both sides).  Sport-specific fields
                live in stats_json; universal columns (score, result, is_home)
                are queryable directly.
  betting_lines One row per game.  All CLV columns are present and NULL now.
                # CLV_READY: populate open_* / close_* / clv_* once
                # TheOddsAPI paid plan is active and historical pulls are live.

Usage
-----
  # Full ingest — all sports, all seasons from config.py
  python historical_etl.py

  # Single sport
  python historical_etl.py --sport NBA
  python historical_etl.py --sport NFL
  python historical_etl.py --sport MLB
  python historical_etl.py --sport NHL
  python historical_etl.py --sport EPL
  python historical_etl.py --sport UCL

  # Wipe and re-ingest
  python historical_etl.py --reset

  # Dry-run: print row counts without writing
  python historical_etl.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime,
    create_engine, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Ensure the backend directory is on sys.path so config imports cleanly
sys.path.insert(0, os.path.dirname(__file__))
import data_fetcher as df_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── DB setup ──────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
DB_URL  = f"sqlite:///{os.path.abspath(DB_PATH)}"

engine       = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Game(Base):
    """
    One row per unique game across all sports and seasons.
    game_id format: {SPORT}_{source_id}
    """
    __tablename__ = "games"

    game_id    = Column(String, primary_key=True)
    sport      = Column(String, nullable=False)   # NBA / NFL / MLB / EPL / UCL / ...
    season     = Column(String, nullable=False)   # "2023-24" / "2023" (calendar year for MLB)
    game_date  = Column(String, nullable=True)    # ISO YYYY-MM-DD
    home_team  = Column(String, nullable=True)
    away_team  = Column(String, nullable=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    status     = Column(String, nullable=True)    # Final / Scheduled / Postponed
    source     = Column(String, nullable=True)    # nba-api / nfl-data-py / pybaseball / ...
    fetched_at = Column(DateTime, default=datetime.utcnow)


class TeamStat(Base):
    """
    Per-game, per-team stats.  Two rows per game (home + away).
    Sport-specific metrics live in stats_json to keep the schema universal.
    Common query columns (score, result, is_home) are stored directly.
    """
    __tablename__ = "team_stats"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    game_id    = Column(String, nullable=False)
    sport      = Column(String, nullable=False)
    season     = Column(String, nullable=False)
    team       = Column(String, nullable=False)
    is_home    = Column(Integer, nullable=True)   # 1 = home, 0 = away
    score      = Column(Integer, nullable=True)
    opp_score  = Column(Integer, nullable=True)
    result     = Column(String, nullable=True)    # W / L / D / T
    stats_json = Column(Text, nullable=True)      # JSON string of all raw stats


class BettingLine(Base):
    """
    Per-game betting lines.  Open and close lines plus derived CLV columns.

    # CLV_READY ─────────────────────────────────────────────────────────────
    # All open_*, close_*, and clv_* columns are NULL and ready to be filled.
    # Populate them once TheOddsAPI paid plan is active and the historical
    # odds pull (TheOddsAPI /v4/historical/sports/{sport}/odds) is available.
    #
    # Suggested fill query (run via historical_etl --fill-clv after pulling):
    #   UPDATE betting_lines
    #   SET    clv_spread   = close_spread   - open_spread,
    #          clv_total    = close_total    - open_total,
    #          clv_ml_home  = close_ml_home  - open_ml_home,
    #          clv_ml_away  = close_ml_away  - open_ml_away
    #   WHERE  open_spread IS NOT NULL AND close_spread IS NOT NULL;
    # ────────────────────────────────────────────────────────────────────────
    """
    __tablename__ = "betting_lines"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    game_id       = Column(String, nullable=False)
    sport         = Column(String, nullable=False)
    season        = Column(String, nullable=False)
    game_date     = Column(String, nullable=True)
    home_team     = Column(String, nullable=True)
    away_team     = Column(String, nullable=True)
    bookmaker     = Column(String, nullable=True)

    # ── Spread (home-team perspective; negative = home favoured) ─────────────
    # CLV_READY: populate when TheOddsAPI paid plan is active
    open_spread   = Column(Float, nullable=True)
    close_spread  = Column(Float, nullable=True)
    clv_spread    = Column(Float, nullable=True)   # close_spread - open_spread

    # ── Game total ────────────────────────────────────────────────────────────
    # CLV_READY: populate when TheOddsAPI paid plan is active
    open_total    = Column(Float, nullable=True)
    close_total   = Column(Float, nullable=True)
    clv_total     = Column(Float, nullable=True)   # close_total - open_total

    # ── Moneyline (American odds) ─────────────────────────────────────────────
    # CLV_READY: populate when TheOddsAPI paid plan is active
    open_ml_home  = Column(Float, nullable=True)
    open_ml_away  = Column(Float, nullable=True)
    close_ml_home = Column(Float, nullable=True)
    close_ml_away = Column(Float, nullable=True)
    clv_ml_home   = Column(Float, nullable=True)   # close_ml_home - open_ml_home
    clv_ml_away   = Column(Float, nullable=True)   # close_ml_away - open_ml_away

    # ── Run-line / spread metadata ────────────────────────────────────────────
    # 1 = home team is the ML favourite (gets -1.5 in MLB run line / NHL puck line)
    # 0 = away team is the ML favourite (home gets +1.5)
    # NULL = unknown (row was written before this column was added)
    home_is_favorite = Column(Integer, nullable=True)

    # ── Puck-line columns (NHL only) ──────────────────────────────────────────
    # covered_pl: 1 = home team covered the ±1.5 puck line, 0 = did not, NULL = no data
    # Computed after backfill_nhl_espn() sets close_spread and home_is_favorite.
    covered_pl    = Column(Integer, nullable=True)
    # ot_so_game: 1 = game decided in OT or shootout (affects puck-line coverage)
    # -1.5 favorite ALWAYS fails to cover in OT/SO (margin = ±1, need ±2+).
    ot_so_game    = Column(Integer, default=0)

    source        = Column(String, nullable=True)  # "the-odds-api" / "nfl-schedules" / etc.
    fetched_at    = Column(DateTime, default=datetime.utcnow)


class PitcherStats(Base):
    """
    Season-level pitching stats per pitcher, used for MLB run-line features.
    One row per (pitcher_name, season).  Only starting pitchers (gs >= 5).
    Populated by fetch_mlb_pitcher_stats() and ingested via run(['MLB']).
    """
    __tablename__ = "pitcher_stats"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    pitcher_name = Column(String, nullable=False)
    player_id    = Column(Integer, nullable=True)  # MLB player ID
    season       = Column(String, nullable=False)  # "2022", "2023", etc.
    era          = Column(Float, nullable=True)
    whip         = Column(Float, nullable=True)
    k9           = Column(Float, nullable=True)    # strikeouts per 9 innings
    bb9          = Column(Float, nullable=True)    # walks per 9 innings
    fip          = Column(Float, nullable=True)    # fielding-independent pitching
    gs           = Column(Integer, nullable=True)  # games started
    ip           = Column(Float, nullable=True)    # innings pitched


class PitcherGameLog(Base):
    """
    Per-start pitching log for qualifying MLB starters.
    One row per (player_id, season, game_date).
    Used to compute rolling ERA/WHIP over last 3 or 5 starts in features.py.
    """
    __tablename__ = "pitcher_game_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    pitcher_name = Column(String, nullable=False)
    player_id    = Column(Integer, nullable=False)
    season       = Column(String, nullable=False)
    game_date    = Column(String, nullable=True)   # ISO YYYY-MM-DD
    game_pk      = Column(Integer, nullable=True)  # MLB gamePk
    ip           = Column(Float, nullable=True)    # innings pitched in this start
    er           = Column(Integer, nullable=True)  # earned runs
    hits         = Column(Integer, nullable=True)
    walks        = Column(Integer, nullable=True)
    strikeouts   = Column(Integer, nullable=True)


class TeamBattingStats(Base):
    """
    Season-level team batting stats for MLB teams.
    One row per (team_name, season).
    Provides OPS/SLG/OBP signal independent of runs-per-game rolling averages.
    """
    __tablename__ = "team_batting_stats"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String, nullable=False)
    season    = Column(String, nullable=False)
    ops       = Column(Float, nullable=True)
    slg       = Column(Float, nullable=True)
    obp       = Column(Float, nullable=True)
    avg       = Column(Float, nullable=True)
    hr        = Column(Integer, nullable=True)
    k_pct     = Column(Float, nullable=True)
    bb_pct    = Column(Float, nullable=True)
    babip     = Column(Float, nullable=True)


def backfill_nba_injuries(seasons: list[str] | None = None) -> dict:
    """
    Backfill per-game injury flags (home/away star & rotation absences) for NBA.

    Steps
    -----
    1. fetch_nba_game_logs(seasons)       — one nba-api call per season
    2. compute_nba_injury_flags(df)       — derive absences from box-score gaps
    3. Pivot per-team rows to per-game    — home_star_out, away_star_out, etc.
    4. INSERT OR REPLACE into game_injury_flags

    Returns
    -------
    dict: seasons, game_log_rows, games_written
    """
    import sqlite3 as _sqlite3
    import pandas as _pd

    from config import HISTORICAL_SEASONS

    if seasons is None:
        seasons = HISTORICAL_SEASONS["NBA"]

    log.info("backfill_nba_injuries: seasons=%s", seasons)

    game_log_df = df_mod.fetch_nba_game_logs(seasons=seasons)
    if game_log_df.empty:
        log.warning("backfill_nba_injuries: no game log data returned")
        return {"seasons": seasons, "game_log_rows": 0, "games_written": 0}

    log.info("backfill_nba_injuries: %d player-game rows fetched", len(game_log_df))

    flags_df = df_mod.compute_nba_injury_flags(game_log_df)
    log.info("backfill_nba_injuries: %d team-game injury rows computed", len(flags_df))

    if flags_df.empty:
        return {"seasons": seasons, "game_log_rows": len(game_log_df), "games_written": 0}

    # Map nba-api GAME_ID (e.g. "0022400001") → our game_id ("NBA_0022400001")
    flags_df["game_id"] = "NBA_" + flags_df["nba_game_id"].astype(str)

    # Load games table to determine which team is home vs away for each game
    _db_path = os.path.abspath(DB_PATH)
    _conn = _sqlite3.connect(_db_path)
    try:
        games_lookup = _pd.read_sql_query(
            "SELECT game_id, home_team, away_team FROM games WHERE sport='NBA'",
            _conn,
        )
    finally:
        _conn.close()

    if games_lookup.empty:
        log.warning("backfill_nba_injuries: no NBA games in games table — run ETL first")
        return {"seasons": seasons, "game_log_rows": len(game_log_df), "games_written": 0}

    # Merge to associate each flag row with a home/away side
    merged = flags_df.merge(games_lookup, on="game_id", how="inner")

    home_mask = merged["team_abbr"] == merged["home_team"]
    away_mask = merged["team_abbr"] == merged["away_team"]

    home_flags = (
        merged[home_mask][["game_id", "star_out", "stars_missing", "rotation_missing"]]
        .rename(columns={
            "star_out":         "home_star_out",
            "stars_missing":    "home_stars_missing",
            "rotation_missing": "home_rotation_missing",
        })
    )
    away_flags = (
        merged[away_mask][["game_id", "star_out", "stars_missing", "rotation_missing"]]
        .rename(columns={
            "star_out":         "away_star_out",
            "stars_missing":    "away_stars_missing",
            "rotation_missing": "away_rotation_missing",
        })
    )

    per_game = home_flags.merge(away_flags, on="game_id", how="outer")
    int_cols = [
        "home_star_out", "away_star_out",
        "home_stars_missing", "away_stars_missing",
        "home_rotation_missing", "away_rotation_missing",
    ]
    per_game[int_cols] = per_game[int_cols].fillna(0).astype(int)

    # Write to DB (raw sqlite3 — this table is not managed by SQLAlchemy ORM)
    _conn = _sqlite3.connect(_db_path)
    try:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS game_injury_flags (
                game_id              TEXT PRIMARY KEY,
                home_star_out        INTEGER DEFAULT 0,
                away_star_out        INTEGER DEFAULT 0,
                home_stars_missing   INTEGER DEFAULT 0,
                away_stars_missing   INTEGER DEFAULT 0,
                home_rotation_missing INTEGER DEFAULT 0,
                away_rotation_missing INTEGER DEFAULT 0
            )
        """)
        written = 0
        for _, row in per_game.iterrows():
            _conn.execute(
                """
                INSERT OR REPLACE INTO game_injury_flags
                (game_id,
                 home_star_out, away_star_out,
                 home_stars_missing, away_stars_missing,
                 home_rotation_missing, away_rotation_missing)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["game_id"],
                    int(row["home_star_out"]),
                    int(row["away_star_out"]),
                    int(row["home_stars_missing"]),
                    int(row["away_stars_missing"]),
                    int(row["home_rotation_missing"]),
                    int(row["away_rotation_missing"]),
                ),
            )
            written += 1
        _conn.commit()
        log.info("backfill_nba_injuries: wrote %d rows to game_injury_flags", written)
        return {
            "seasons":       seasons,
            "game_log_rows": len(game_log_df),
            "games_written": written,
        }
    except Exception as exc:
        _conn.rollback()
        log.error("backfill_nba_injuries: DB write error: %s", exc)
        return {
            "seasons":       seasons,
            "game_log_rows": len(game_log_df),
            "games_written": 0,
            "error":         str(exc),
        }
    finally:
        _conn.close()


def backfill_mlb_scores(seasons: list[int] | None = None) -> dict:
    """
    Full MLB run-line backfill for 4 seasons (2022-2025).

    Steps
    -----
    1. run(['MLB']) — ingest games + team_stats + betting_lines stubs
    2. backfill_mlb_espn() — populate close_spread (±1.5), close_ml_*, home_is_favorite
    3. cover_rate_check("MLB") — expect 48-52% on run line
    4. Return summary

    Returns
    -------
    dict: games_inserted, lines_filled, cover_rate_pct, passed
    """
    from config import HISTORICAL_SEASONS as _HS
    if seasons is None:
        seasons = _HS.get("MLB", [2022, 2023, 2024, 2025])

    log.info("backfill_mlb_scores: seasons=%s", seasons)
    init_db()

    # Step 1 — ingest via standard ETL (uses config.py seasons)
    run(sports=["MLB"], reset=False, dry_run=False)

    # Step 2 — ESPN run-line backfill
    try:
        from backfill_lines import backfill_mlb_espn
        lines_result = backfill_mlb_espn(dry_run=False)
        lines_filled = lines_result.get("filled", 0)
        log.info("backfill_mlb_scores: espn lines filled=%d missing=%d",
                 lines_filled, lines_result.get("missing", 0))
    except Exception as exc:
        log.error("backfill_mlb_scores: ESPN backfill failed: %s", exc)
        lines_filled = 0

    # Step 3 — cover rate check
    import sqlite3 as _sqlite3
    import pandas as _pd
    import numpy as _np

    cover_rate = None
    passed     = False
    try:
        _db_path = os.path.abspath(DB_PATH)
        _conn    = _sqlite3.connect(_db_path)
        rows = _pd.read_sql_query("""
            SELECT g.home_score, g.away_score, bl.close_spread
            FROM   betting_lines bl
            JOIN   games g ON bl.game_id = g.game_id
            WHERE  bl.sport = 'MLB'
              AND  bl.close_spread IS NOT NULL
              AND  g.home_score   IS NOT NULL
              AND  g.away_score   IS NOT NULL
        """, _conn)
        _conn.close()

        if not rows.empty:
            ats_val = (_pd.to_numeric(rows["home_score"], errors="coerce")
                       - _pd.to_numeric(rows["away_score"],  errors="coerce")
                       + _pd.to_numeric(rows["close_spread"], errors="coerce"))
            labels = _np.where(ats_val > 0, 1.0, _np.where(ats_val < 0, 0.0, _np.nan))
            labeled = _pd.Series(labels).dropna()
            if len(labeled) >= 50:
                cover_rate = float(labeled.mean())
                passed = 0.46 <= cover_rate <= 0.54
                log.info("backfill_mlb_scores: cover_rate=%.1f%%  passed=%s  n=%d",
                         cover_rate * 100, passed, len(labeled))
    except Exception as exc:
        log.error("backfill_mlb_scores: cover rate check error: %s", exc)

    return {
        "sport":          "MLB",
        "seasons":        seasons,
        "lines_filled":   lines_filled,
        "cover_rate_pct": round(cover_rate * 100, 1) if cover_rate else None,
        "passed":         passed,
    }


def update_mlb_scores_for_date(date_str: str) -> dict:
    """
    Fetch yesterday's MLB scores from MLB Stats API and write any
    missing home_score / away_score into the games table.

    Used by the nightly scheduler.  Only updates rows where home_score IS NULL.
    """
    log.info("update_mlb_scores_for_date: %s", date_str)
    try:
        import statsapi
    except ImportError:
        log.error("update_mlb_scores_for_date: mlb-statsapi not installed")
        return {"date": date_str, "fetched": 0, "updated": 0}

    try:
        games_api = statsapi.schedule(start_date=date_str, end_date=date_str, sportId=1)
    except Exception as exc:
        log.error("update_mlb_scores_for_date: statsapi error: %s", exc)
        return {"date": date_str, "fetched": 0, "updated": 0, "error": str(exc)}

    final = [g for g in games_api
             if g.get("game_type") == "R" and g.get("status") == "Final"]
    log.info("update_mlb_scores_for_date: %d final games on %s", len(final), date_str)

    if not final:
        return {"date": date_str, "fetched": 0, "updated": 0}

    # Build lookup: (date, home_name, away_name) → (home_score, away_score)
    lookup: dict = {}
    for g in final:
        key = (date_str, g.get("home_name", ""), g.get("away_name", ""))
        hs  = g.get("home_score")
        as_ = g.get("away_score")
        if hs is not None and as_ is not None:
            lookup[key] = (int(hs), int(as_))

    db = SessionLocal()
    updated = 0
    try:
        null_games = (
            db.query(Game)
            .filter(Game.sport == "MLB", Game.game_date == date_str,
                    Game.home_score.is_(None))
            .all()
        )
        for g in null_games:
            key = (date_str, g.home_team or "", g.away_team or "")
            if key in lookup:
                g.home_score, g.away_score = lookup[key]
                updated += 1
        if updated:
            db.commit()
        log.info("update_mlb_scores_for_date: updated %d games", updated)
        return {"date": date_str, "fetched": len(final), "updated": updated}
    except Exception as exc:
        db.rollback()
        log.error("update_mlb_scores_for_date: DB error: %s", exc)
        return {"date": date_str, "fetched": len(final), "updated": 0, "error": str(exc)}
    finally:
        db.close()


def backfill_nhl_scores(seasons: list[str] | None = None) -> dict:
    """
    Full NHL puck-line backfill for 5 seasons (2021-22 through 2025-26).

    Steps
    -----
    1. run(['NHL']) — ingest games + team_stats + betting_lines with ot_so_game flag
    2. backfill_nhl_espn() — populate close_spread (±1.5), close_ml_*, home_is_favorite
    3. Compute covered_pl from scores + close_spread (standard formula handles OT/SO)
    4. cover_rate_check("NHL") — REG ~50%, OT/SO ~0% from home -1.5 perspective
    5. Return summary with OT/SO breakdown

    Returns
    -------
    dict: games_inserted, ot_so_count, lines_filled, cover_rate_pct,
          ot_so_cover_rate_pct, passed
    """
    from config import HISTORICAL_SEASONS as _HS
    if seasons is None:
        seasons = _HS.get("NHL", ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])

    log.info("backfill_nhl_scores: seasons=%s", seasons)
    init_db()

    # Step 1 — ingest via standard ETL
    run(sports=["NHL"], reset=False, dry_run=False)

    # Step 2 — ESPN moneyline backfill (sets close_spread and home_is_favorite)
    lines_filled = 0
    try:
        from backfill_lines import backfill_nhl_espn
        lines_result = backfill_nhl_espn(dry_run=False)
        lines_filled = lines_result.get("filled", 0)
        log.info("backfill_nhl_scores: espn lines filled=%d missing=%d",
                 lines_filled, lines_result.get("missing", 0))
    except Exception as exc:
        log.error("backfill_nhl_scores: ESPN backfill failed: %s", exc)

    # Step 3 — Compute covered_pl using standard formula
    # Formula: (home_score - away_score + close_spread) > 0 → 1, < 0 → 0, else NULL
    # This correctly handles OT/SO: home -1.5 always yields ats_val ≤ -0.5 < 0 in OT/SO
    import sqlite3 as _sqlite3
    import pandas as _pd
    import numpy as _np

    try:
        _db_path = os.path.abspath(DB_PATH)
        _conn    = _sqlite3.connect(_db_path)
        # Update covered_pl for all NHL rows that have scores and a spread
        _conn.execute("""
            UPDATE betting_lines
            SET covered_pl = (
                SELECT CASE
                    WHEN g.home_score IS NULL OR g.away_score IS NULL THEN NULL
                    WHEN betting_lines.close_spread IS NULL THEN NULL
                    WHEN (g.home_score - g.away_score + betting_lines.close_spread) > 0 THEN 1
                    WHEN (g.home_score - g.away_score + betting_lines.close_spread) < 0 THEN 0
                    ELSE NULL
                END
                FROM games g WHERE g.game_id = betting_lines.game_id
            )
            WHERE sport = 'NHL'
        """)
        _conn.commit()
        log.info("backfill_nhl_scores: covered_pl computed for NHL rows")
        _conn.close()
    except Exception as exc:
        log.error("backfill_nhl_scores: covered_pl update error: %s", exc)

    # Step 4 — Cover rate check with OT/SO breakdown
    cover_rate = ot_so_cover_rate = None
    ot_so_count = 0
    passed = False
    try:
        _conn = _sqlite3.connect(os.path.abspath(DB_PATH))
        rows = _pd.read_sql_query("""
            SELECT g.home_score, g.away_score, bl.close_spread, bl.ot_so_game
            FROM   betting_lines bl
            JOIN   games g ON bl.game_id = g.game_id
            WHERE  bl.sport = 'NHL'
              AND  bl.close_spread IS NOT NULL
              AND  g.home_score   IS NOT NULL
              AND  g.away_score   IS NOT NULL
        """, _conn)
        _conn.close()

        if not rows.empty:
            ats_val = (_pd.to_numeric(rows["home_score"], errors="coerce")
                       - _pd.to_numeric(rows["away_score"],  errors="coerce")
                       + _pd.to_numeric(rows["close_spread"], errors="coerce"))
            labels = _np.where(ats_val > 0, 1.0, _np.where(ats_val < 0, 0.0, _np.nan))
            labeled = _pd.Series(labels)
            ot_flag = _pd.to_numeric(rows["ot_so_game"], errors="coerce").fillna(0)

            all_labeled = labeled.dropna()
            if len(all_labeled) >= 50:
                cover_rate = float(all_labeled.mean())
                passed = 0.44 <= cover_rate <= 0.56   # wider bound due to OT/SO suppression
                log.info("backfill_nhl_scores: cover_rate=%.1f%%  n=%d  passed=%s",
                         cover_rate * 100, len(all_labeled), passed)

            ot_mask = (ot_flag == 1) & labeled.notna()
            ot_so_count = int(ot_mask.sum())
            if ot_so_count >= 10:
                ot_so_cover_rate = float(labeled[ot_mask].mean())
                log.info("backfill_nhl_scores: OT/SO cover_rate=%.1f%%  n=%d  "
                         "(expected ~0%% for home -1.5 perspective)",
                         ot_so_cover_rate * 100, ot_so_count)
    except Exception as exc:
        log.error("backfill_nhl_scores: cover rate check error: %s", exc)

    return {
        "sport":              "NHL",
        "seasons":            seasons,
        "lines_filled":       lines_filled,
        "ot_so_count":        ot_so_count,
        "cover_rate_pct":     round(cover_rate * 100, 1) if cover_rate else None,
        "ot_so_cover_pct":    round(ot_so_cover_rate * 100, 1) if ot_so_cover_rate else None,
        "passed":             passed,
    }


def update_nhl_scores_for_date(date_str: str) -> dict:
    """
    Fetch yesterday's NHL scores from the NHL Web API and write any
    missing home_score / away_score into the games table.

    Used by the nightly scheduler.  Only updates rows where home_score IS NULL.
    """
    log.info("update_nhl_scores_for_date: %s", date_str)
    games_list = df_mod.fetch_nhl_scores_for_date(date_str)
    if not games_list:
        log.info("update_nhl_scores_for_date: no completed NHL games found for %s", date_str)
        return {"date": date_str, "fetched": 0, "updated": 0}

    # Build lookup: (home_team_abbrev, away_team_abbrev) → (home_score, away_score, last_period)
    lookup: dict = {}
    for g in games_list:
        key = (g["home_team"].upper(), g["away_team"].upper())
        if g["home_score"] is not None and g["away_score"] is not None:
            lookup[key] = g

    db = SessionLocal()
    updated = 0
    try:
        null_games = (
            db.query(Game)
            .filter(Game.sport == "NHL", Game.game_date == date_str,
                    Game.home_score.is_(None))
            .all()
        )
        for g in null_games:
            key = ((g.home_team or "").upper(), (g.away_team or "").upper())
            if key in lookup:
                entry = lookup[key]
                g.home_score = entry["home_score"]
                g.away_score = entry["away_score"]
                # Update ot_so_game in betting_lines via raw SQL
                last_period = (entry.get("last_period") or "").upper()
                ot_so = 1 if last_period in ("OT", "SO") else 0
                db.execute(text("""
                    UPDATE betting_lines SET ot_so_game = :flag
                    WHERE  game_id = :gid AND sport = 'NHL'
                """), {"flag": ot_so, "gid": g.game_id})
                updated += 1

        if updated:
            db.commit()
        log.info("update_nhl_scores_for_date: updated %d games", updated)
        return {"date": date_str, "fetched": len(games_list), "updated": updated}
    except Exception as exc:
        db.rollback()
        log.error("update_nhl_scores_for_date: DB error: %s", exc)
        return {"date": date_str, "fetched": len(games_list), "updated": 0, "error": str(exc)}
    finally:
        db.close()


def update_nba_scores_for_date(date_str: str) -> dict:
    """
    Fetch BDL final scores for *date_str* (ISO "YYYY-MM-DD") and write any
    missing home_score / away_score into the games table.

    Used by the nightly scheduler for same-day ATS coverage.
    Only updates rows where home_score IS NULL to avoid overwriting nba-api data.
    """
    log.info("update_nba_scores_for_date: %s", date_str)
    scores_df = df_mod.fetch_nba_spreads(seasons=[2024, 2025], date=date_str)

    if scores_df.empty:
        log.info("update_nba_scores_for_date: no BDL games found for %s", date_str)
        return {"date": date_str, "bdl_fetched": 0, "updated": 0}

    # Build lookup (date, home_abbr, away_abbr) → (home_score, away_score)
    lookup: dict[tuple, tuple] = {
        (row["game_date"], row["home_abbr"].upper(), row["away_abbr"].upper()):
        (int(row["home_score"]), int(row["away_score"]))
        for _, row in scores_df.iterrows()
    }

    db = SessionLocal()
    updated = 0
    try:
        null_games = (
            db.query(Game)
            .filter(
                Game.sport == "NBA",
                Game.game_date == date_str,
                Game.home_score.is_(None),
            )
            .all()
        )
        for g in null_games:
            key = (
                (g.game_date or "")[:10],
                (g.home_team or "").upper(),
                (g.away_team or "").upper(),
            )
            if key in lookup:
                g.home_score, g.away_score = lookup[key]
                updated += 1

        if updated:
            db.commit()
        log.info("update_nba_scores_for_date: %s → updated %d game(s)", date_str, updated)
        return {"date": date_str, "bdl_fetched": len(scores_df), "updated": updated}
    except Exception as exc:
        db.rollback()
        log.error("update_nba_scores_for_date: DB error: %s", exc)
        return {"date": date_str, "bdl_fetched": len(scores_df), "updated": 0, "error": str(exc)}
    finally:
        db.close()


def backfill_nba_ats(seasons: list[int] | None = None) -> dict:
    """
    Full NBA ATS backfill using BallDontLie scores.

    Steps
    -----
    1. Fetch all final regular-season scores from BDL for *seasons*.
    2. Update games.home_score / away_score where currently NULL.
    3. Query betting_lines JOIN games for rows with spread + scores.
    4. Compute ATS labels inline (no lookahead — just close_spread + final score).
    5. Run cover_rate_check("NBA").  Log recommendation.

    Returns
    -------
    dict: bdl_fetched, score_updates, ats_rows, cover_rate_pct, passed
    """
    import numpy as np
    import pandas as pd

    if seasons is None:
        seasons = [2024, 2025]

    log.info("backfill_nba_ats: fetching BDL scores for seasons %s", seasons)
    bdl_df = df_mod.fetch_nba_spreads(seasons=seasons)
    log.info("backfill_nba_ats: %d BDL games fetched", len(bdl_df))

    if bdl_df.empty:
        return {"bdl_fetched": 0, "score_updates": 0, "ats_rows": 0,
                "cover_rate_pct": None, "passed": False}

    # ── Build lookup ──────────────────────────────────────────────────────────
    lookup: dict[tuple, tuple] = {
        (row["game_date"], row["home_abbr"].upper(), row["away_abbr"].upper()):
        (int(row["home_score"]), int(row["away_score"]))
        for _, row in bdl_df.iterrows()
    }

    # ── Update missing game scores ────────────────────────────────────────────
    db = SessionLocal()
    score_updates = 0
    try:
        null_games = (
            db.query(Game)
            .filter(Game.sport == "NBA", Game.home_score.is_(None))
            .all()
        )
        for g in null_games:
            key = (
                (g.game_date or "")[:10],
                (g.home_team or "").upper(),
                (g.away_team or "").upper(),
            )
            if key in lookup:
                g.home_score, g.away_score = lookup[key]
                score_updates += 1

        if score_updates:
            db.commit()
            log.info("backfill_nba_ats: wrote scores for %d games", score_updates)

        # ── Compute ATS labels from DB ────────────────────────────────────────
        rows = db.execute(text("""
            SELECT g.home_score, g.away_score, bl.close_spread
            FROM   betting_lines bl
            JOIN   games g ON bl.game_id = g.game_id
            WHERE  bl.sport = 'NBA'
              AND  bl.close_spread IS NOT NULL
              AND  g.home_score   IS NOT NULL
              AND  g.away_score   IS NOT NULL
        """)).fetchall()

        if not rows:
            log.warning("backfill_nba_ats: no NBA rows with spread + scores — "
                        "run historical_etl.py --sport NBA first")
            return {
                "bdl_fetched": len(bdl_df),
                "score_updates": score_updates,
                "ats_rows": 0,
                "cover_rate_pct": None,
                "passed": False,
            }

        ats_df = pd.DataFrame(rows, columns=["home_score", "away_score", "close_spread"])
        ats_val = (
            pd.to_numeric(ats_df["home_score"], errors="coerce")
            - pd.to_numeric(ats_df["away_score"],  errors="coerce")
            + pd.to_numeric(ats_df["close_spread"], errors="coerce")
        )
        ats_df["_label"] = np.where(ats_val > 0, 1.0,
                           np.where(ats_val < 0, 0.0, np.nan))
        labeled = ats_df["_label"].dropna()

        # ── cover_rate_check ──────────────────────────────────────────────────
        passed = False
        try:
            from ml_model import cover_rate_check
            passed = cover_rate_check("NBA", ats_df)
        except ImportError:
            log.warning("backfill_nba_ats: ml_model not importable; checking inline")
            if len(labeled) >= 50:
                rate = float(labeled.mean())
                passed = 0.46 <= rate <= 0.54

        cover_rate = float(labeled.mean()) if len(labeled) >= 50 else None
        summary = {
            "bdl_fetched":    len(bdl_df),
            "score_updates":  score_updates,
            "ats_rows":       int(len(labeled)),
            "cover_rate_pct": round(cover_rate * 100, 1) if cover_rate is not None else None,
            "passed":         passed,
        }
        log.info(
            "backfill_nba_ats: ats_rows=%d  cover_rate=%.1f%%  passed=%s",
            summary["ats_rows"],
            summary["cover_rate_pct"] or 0.0,
            summary["passed"],
        )
        if passed:
            log.info(
                "backfill_nba_ats: NBA cover rate OK — "
                "add 'NBA' to HISTORICAL_ATS_SPORTS in ml_model.py to include in training"
            )
        else:
            log.warning(
                "backfill_nba_ats: cover rate check FAILED — "
                "keep NBA out of HISTORICAL_ATS_SPORTS until more data accumulates"
            )
        return summary

    except Exception as exc:
        db.rollback()
        log.error("backfill_nba_ats: error: %s", exc)
        return {"bdl_fetched": len(bdl_df), "score_updates": 0,
                "ats_rows": 0, "cover_rate_pct": None, "passed": False,
                "error": str(exc)}
    finally:
        db.close()


def _migrate_puck_line_columns() -> None:
    """Add NHL puck-line columns to betting_lines if they don't exist yet."""
    migrations = [
        "ALTER TABLE betting_lines ADD COLUMN covered_pl INTEGER DEFAULT NULL",
        "ALTER TABLE betting_lines ADD COLUMN ot_so_game INTEGER DEFAULT 0",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                pass  # column already exists


def init_db() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _migrate_puck_line_columns()
    log.info("historical.db initialised at %s", os.path.abspath(DB_PATH))


def reset_db(sport: str | None = None) -> None:
    """Drop all rows (or rows for a single sport) from all three tables."""
    db = SessionLocal()
    try:
        for Model in (BettingLine, TeamStat, Game):
            q = db.query(Model)
            if sport:
                q = q.filter(Model.sport == sport)
            deleted = q.delete(synchronize_session=False)
            log.info("reset_db: deleted %d rows from %s (sport=%s)",
                     deleted, Model.__tablename__, sport or "ALL")
        db.commit()
    finally:
        db.close()


# ── Write helpers ─────────────────────────────────────────────────────────────

def _upsert_games(session, rows: list[dict]) -> int:
    written = 0
    for r in rows:
        gid = r.get("game_id")
        if not gid:
            continue
        existing = session.get(Game, gid)
        if existing:
            continue   # skip duplicates; use --reset to re-ingest
        g = Game(
            game_id    = gid,
            sport      = r.get("sport"),
            season     = str(r.get("season", "")),
            game_date  = r.get("game_date"),
            home_team  = str(r.get("home_team", "") or ""),
            away_team  = str(r.get("away_team", "") or ""),
            home_score = r.get("home_score"),
            away_score = r.get("away_score"),
            status     = r.get("status"),
            source     = r.get("source"),
        )
        session.add(g)
        written += 1
    return written


def _upsert_team_stats(session, rows: list[dict]) -> int:
    written = 0
    for r in rows:
        gid  = r.get("game_id")
        team = r.get("team")
        if not gid or not team:
            continue
        # Use game_id + team as de-dup key
        exists = (
            session.query(TeamStat)
            .filter(TeamStat.game_id == gid, TeamStat.team == str(team))
            .first()
        )
        if exists:
            continue
        ts = TeamStat(
            game_id    = gid,
            sport      = r.get("sport"),
            season     = str(r.get("season", "")),
            team       = str(team),
            is_home    = r.get("is_home"),
            score      = r.get("score"),
            opp_score  = r.get("opp_score"),
            result     = r.get("result"),
            stats_json = r.get("stats_json"),
        )
        session.add(ts)
        written += 1
    return written


def _upsert_betting_lines(session, rows: list[dict]) -> int:
    written = 0
    for r in rows:
        gid = r.get("game_id")
        if not gid:
            continue
        exists = (
            session.query(BettingLine)
            .filter(BettingLine.game_id == gid)
            .first()
        )
        if exists:
            continue
        bl = BettingLine(
            game_id       = gid,
            sport         = r.get("sport"),
            season        = str(r.get("season", "")),
            game_date     = r.get("game_date"),
            home_team     = str(r.get("home_team", "") or ""),
            away_team     = str(r.get("away_team", "") or ""),
            bookmaker     = r.get("bookmaker"),

            # CLV_READY: all NULL until TheOddsAPI paid plan is active
            open_spread   = r.get("open_spread"),    # NULL
            close_spread  = r.get("close_spread"),   # NULL (or NFL closing line if available)
            clv_spread    = r.get("clv_spread"),     # NULL

            open_total    = r.get("open_total"),     # NULL
            close_total   = r.get("close_total"),    # NULL (or NFL total_line if available)
            clv_total     = r.get("clv_total"),      # NULL

            open_ml_home  = r.get("open_ml_home"),   # NULL
            open_ml_away  = r.get("open_ml_away"),   # NULL
            close_ml_home = r.get("close_ml_home"),  # NULL (or NFL home_moneyline)
            close_ml_away = r.get("close_ml_away"),  # NULL (or NFL away_moneyline)
            clv_ml_home   = r.get("clv_ml_home"),    # NULL
            clv_ml_away   = r.get("clv_ml_away"),    # NULL

            # NHL puck-line fields (NULL/0 for non-NHL sports)
            covered_pl    = r.get("covered_pl"),
            ot_so_game    = r.get("ot_so_game", 0) or 0,

            source        = r.get("source"),
        )
        session.add(bl)
        written += 1
    return written


# ── Sport-level ingest functions ──────────────────────────────────────────────

def _ingest_sport(session, sport_label: str, games_df, stats_df) -> dict:
    """
    Write games, team_stats, and betting_lines rows for one sport.
    Returns a summary dict.
    """
    # ── games ──────────────────────────────────────────────────────────────
    game_rows = games_df.where(games_df.notna(), None).to_dict("records")
    g_written = _upsert_games(session, game_rows)

    # ── team_stats ──────────────────────────────────────────────────────────
    stat_rows = stats_df.where(stats_df.notna(), None).to_dict("records")
    s_written = _upsert_team_stats(session, stat_rows)

    # ── betting_lines (one row per game, all CLV columns NULL) ───────────────
    bl_rows = []
    for r in game_rows:
        gid = r.get("game_id")
        if not gid:
            continue

        bl = {
            "game_id":   gid,
            "sport":     r.get("sport"),
            "season":    r.get("season"),
            "game_date": r.get("game_date"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "bookmaker": None,
            "source":    None,
            # CLV_READY — all NULL by default
            "open_spread":   None,
            "close_spread":  None,
            "clv_spread":    None,
            "open_total":    None,
            "close_total":   None,
            "clv_total":     None,
            "open_ml_home":  None,
            "open_ml_away":  None,
            "close_ml_home": None,
            "close_ml_away": None,
            "clv_ml_home":   None,
            "clv_ml_away":   None,
        }

        # NFL: import_schedules already includes closing lines — pre-fill close_*
        # (open_* and clv_* remain NULL until TheOddsAPI historical data is available)
        if r.get("sport") == "NFL":
            bl["close_spread"]  = r.get("_spread_line")
            bl["close_total"]   = r.get("_total_line")
            bl["close_ml_home"] = r.get("_home_moneyline")
            bl["close_ml_away"] = r.get("_away_moneyline")
            bl["source"]        = "nfl-data-py"

        # NHL: store OT/SO flag from _last_period (fetched by fetch_nhl_games)
        if r.get("sport") == "NHL":
            last_period = (r.get("_last_period") or "").upper()
            bl["ot_so_game"] = 1 if last_period in ("OT", "SO") else 0

        bl_rows.append(bl)

    bl_written = _upsert_betting_lines(session, bl_rows)

    return {
        "sport":   sport_label,
        "games":   g_written,
        "stats":   s_written,
        "lines":   bl_written,
    }


def _upsert_pitcher_game_logs(session, rows: list[dict]) -> int:
    """
    INSERT OR REPLACE pitcher_game_logs rows keyed on (player_id, season, game_date).
    Returns count of rows written.
    """
    written = 0
    for r in rows:
        pid       = r.get("player_id")
        season    = r.get("season", "")
        game_date = r.get("game_date")
        if game_pk := r.get("game_pk"):
            game_pk = int(game_pk) if game_pk else None
        if not pid or not season or not game_date:
            continue
        # Normalize date to string
        if hasattr(game_date, "isoformat"):
            game_date = game_date.strftime("%Y-%m-%d")
        existing = (
            session.query(PitcherGameLog)
            .filter_by(player_id=int(pid), season=season, game_date=game_date)
            .first()
        )
        if existing:
            existing.ip         = r.get("ip")
            existing.er         = r.get("er")
            existing.hits       = r.get("hits")
            existing.walks      = r.get("walks")
            existing.strikeouts = r.get("strikeouts")
        else:
            session.add(PitcherGameLog(
                pitcher_name = r.get("pitcher_name", ""),
                player_id    = int(pid),
                season       = season,
                game_date    = game_date,
                game_pk      = game_pk,
                ip           = r.get("ip"),
                er           = r.get("er"),
                hits         = r.get("hits"),
                walks        = r.get("walks"),
                strikeouts   = r.get("strikeouts"),
            ))
        written += 1
    return written


def _upsert_pitcher_stats(session, rows: list[dict]) -> int:
    """
    INSERT OR REPLACE pitcher_stats rows keyed on (pitcher_name, season).
    Returns count of rows written.
    """
    written = 0
    for r in rows:
        name   = r.get("pitcher_name", "")
        season = r.get("season", "")
        if not name or not season:
            continue
        existing = (
            session.query(PitcherStats)
            .filter_by(pitcher_name=name, season=season)
            .first()
        )
        if existing:
            existing.player_id = r.get("player_id")
            existing.era  = r.get("era")
            existing.whip = r.get("whip")
            existing.k9   = r.get("k9")
            existing.bb9  = r.get("bb9")
            existing.fip  = r.get("fip")
            existing.gs   = r.get("gs")
            existing.ip   = r.get("ip")
        else:
            session.add(PitcherStats(
                pitcher_name = name,
                player_id    = r.get("player_id"),
                season       = season,
                era          = r.get("era"),
                whip         = r.get("whip"),
                k9           = r.get("k9"),
                bb9          = r.get("bb9"),
                fip          = r.get("fip"),
                gs           = r.get("gs"),
                ip           = r.get("ip"),
            ))
        written += 1
    return written


def _upsert_team_batting_stats(session, rows: list[dict]) -> int:
    """
    INSERT OR REPLACE team_batting_stats rows keyed on (team_name, season).
    Returns count of rows written.
    """
    written = 0
    for r in rows:
        name   = r.get("team_name", "")
        season = r.get("season", "")
        if not name or not season:
            continue
        existing = (
            session.query(TeamBattingStats)
            .filter_by(team_name=name, season=season)
            .first()
        )
        if existing:
            existing.ops   = r.get("ops")
            existing.slg   = r.get("slg")
            existing.obp   = r.get("obp")
            existing.avg   = r.get("avg")
            existing.hr    = r.get("hr")
            existing.k_pct = r.get("k_pct")
            existing.bb_pct= r.get("bb_pct")
            existing.babip = r.get("babip")
        else:
            session.add(TeamBattingStats(
                team_name = name,
                season    = season,
                ops       = r.get("ops"),
                slg       = r.get("slg"),
                obp       = r.get("obp"),
                avg       = r.get("avg"),
                hr        = r.get("hr"),
                k_pct     = r.get("k_pct"),
                bb_pct    = r.get("bb_pct"),
                babip     = r.get("babip"),
            ))
        written += 1
    return written


# ── Main orchestrator ─────────────────────────────────────────────────────────

SPORT_DISPATCH = {
    "NBA": (df_mod.fetch_nba_games,    df_mod.fetch_nba_team_stats),
    "NFL": (df_mod.fetch_nfl_games,    df_mod.fetch_nfl_team_stats),
    "MLB": (df_mod.fetch_mlb_games,    df_mod.fetch_mlb_team_stats),
    "NHL": (df_mod.fetch_nhl_games,    df_mod.fetch_nhl_team_stats),
    "UCL": (df_mod.fetch_ucl_games,    df_mod.fetch_ucl_team_stats),
    # Domestic soccer leagues — all fetched together via soccerdata
    "EPL":        (lambda: df_mod.fetch_soccer_games(["EPL"]),        lambda: df_mod.fetch_soccer_team_stats(["EPL"])),
    "LaLiga":     (lambda: df_mod.fetch_soccer_games(["LaLiga"]),     lambda: df_mod.fetch_soccer_team_stats(["LaLiga"])),
    "Bundesliga": (lambda: df_mod.fetch_soccer_games(["Bundesliga"]), lambda: df_mod.fetch_soccer_team_stats(["Bundesliga"])),
    "Ligue1":     (lambda: df_mod.fetch_soccer_games(["Ligue1"]),     lambda: df_mod.fetch_soccer_team_stats(["Ligue1"])),
    "SerieA":     (lambda: df_mod.fetch_soccer_games(["SerieA"]),     lambda: df_mod.fetch_soccer_team_stats(["SerieA"])),
}


def upsert_mlb_today_pitchers(date_str: str | None = None) -> dict:
    """
    Lightweight morning fetch: pull today's MLB probable pitchers from the free
    MLB Stats API and write/update Game + TeamStat rows in historical.db.

    Designed to run at 6:45 AM UTC (before mock bet generation at 7:00 AM) so
    the MLB ATS sub-model has pitcher context when scoring today's games.

    Returns {date, games_found, games_written, stats_written, error?}.
    """
    import json as _json
    from datetime import date

    if date_str is None:
        date_str = date.today().isoformat()

    try:
        games = df_mod.fetch_mlb_todays_probable_pitchers(date_str)
    except Exception as exc:
        return {"date": date_str, "games_found": 0, "error": str(exc)}

    if not games:
        return {"date": date_str, "games_found": 0, "games_written": 0, "stats_written": 0}

    season = date_str[:4]
    db = SessionLocal()
    games_written = 0
    stats_written = 0

    try:
        for g in games:
            game_pk = g.get("game_pk")
            if not game_pk:
                continue
            gid  = f"MLB_{game_pk}"
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            h_sp = g.get("home_probable_pitcher") or ""
            a_sp = g.get("away_probable_pitcher") or ""

            # Create game row if missing (today's games may not be in DB yet)
            if not db.get(Game, gid):
                db.add(Game(
                    game_id   = gid,
                    sport     = "MLB",
                    season    = season,
                    game_date = date_str,
                    home_team = home,
                    away_team = away,
                    status    = "Scheduled",
                    source    = "mlb_stats_api_pitcher_fetch",
                ))
                games_written += 1

            # Build meta dict for stats_json
            meta = {
                "home_probable_pitcher": h_sp,
                "away_probable_pitcher": a_sp,
                "venue": None, "doubleheader": None,
                "winning_pitcher": None, "losing_pitcher": None, "save_pitcher": None,
                "series_status": None,
            }
            meta_json = _json.dumps(meta)

            # Upsert home team_stats row
            for team, is_home in [(home, 1), (away, 0)]:
                existing = (
                    db.query(TeamStat)
                    .filter(TeamStat.game_id == gid, TeamStat.team == team)
                    .first()
                )
                if existing:
                    # Update stats_json to inject pitcher info
                    try:
                        old = _json.loads(existing.stats_json or "{}")
                    except Exception:
                        old = {}
                    old["home_probable_pitcher"] = h_sp
                    old["away_probable_pitcher"] = a_sp
                    existing.stats_json = _json.dumps(old)
                else:
                    db.add(TeamStat(
                        game_id    = gid,
                        sport      = "MLB",
                        season     = season,
                        team       = team,
                        is_home    = is_home,
                        score      = None,
                        opp_score  = None,
                        result     = None,
                        stats_json = meta_json,
                    ))
                    stats_written += 1

        db.commit()
    except Exception as exc:
        db.rollback()
        return {"date": date_str, "games_found": len(games), "error": str(exc)}
    finally:
        db.close()

    # Invalidate ML model feature-matrix cache for MLB
    try:
        import ml_model
        ml_model._FM_CACHE.pop("MLB", None)
    except Exception:
        pass

    return {
        "date":           date_str,
        "games_found":    len(games),
        "games_written":  games_written,
        "stats_written":  stats_written,
    }


def run(
    sports: list[str] | None = None,
    reset: bool = False,
    dry_run: bool = False,
    quick: bool = False,
) -> None:
    """
    Main entry point.

    sports   : list of sport keys to ingest; None = all
    reset    : wipe matching rows before re-ingesting
    dry_run  : fetch and log counts without writing to DB
    """
    init_db()

    if sports is None:
        sports = list(SPORT_DISPATCH.keys())

    totals = {"games": 0, "stats": 0, "lines": 0}

    for sport in sports:
        if sport not in SPORT_DISPATCH:
            log.warning("Unknown sport key: %s — skipping", sport)
            continue

        fetch_games_fn, fetch_stats_fn = SPORT_DISPATCH[sport]

        log.info("━━━  %s  ━━━", sport)
        games_df = df_mod._empty_games()
        stats_df = df_mod._empty_team_stats()

        try:
            games_df = fetch_games_fn()
            log.info("%s: %d game rows fetched", sport, len(games_df))
        except Exception as exc:
            log.error("%s games fetch error: %s", sport, exc)

        try:
            stats_df = fetch_stats_fn()
            log.info("%s: %d stat rows fetched", sport, len(stats_df))
        except Exception as exc:
            log.error("%s stats fetch error: %s", sport, exc)

        if dry_run:
            log.info("[DRY RUN] %s — would write %d games, %d stat rows",
                     sport, len(games_df), len(stats_df))
            continue

        # MLB: also fetch per-season pitcher stats and per-start game logs
        pitcher_df  = None
        game_log_df = None
        batting_df  = None
        if sport == "MLB":
            try:
                pitcher_df = df_mod.fetch_mlb_pitcher_stats()
                log.info("MLB: %d pitcher stat rows fetched", len(pitcher_df))
            except Exception as exc:
                log.error("MLB pitcher stats fetch error: %s", exc)
            if not quick:
                try:
                    game_log_df = df_mod.fetch_mlb_pitcher_game_logs()
                    log.info("MLB: %d pitcher game log rows fetched", len(game_log_df))
                except Exception as exc:
                    log.error("MLB pitcher game logs fetch error: %s", exc)
            else:
                log.info("MLB: skipping pitcher game log fetch (--quick mode)")
            try:
                batting_df = df_mod.fetch_mlb_team_batting_stats()
                log.info("MLB: %d team batting rows fetched", len(batting_df))
            except Exception as exc:
                log.error("MLB team batting fetch error: %s", exc)

        db = SessionLocal()
        try:
            if reset:
                for Model in (BettingLine, TeamStat, Game):
                    db.query(Model).filter(Model.sport == sport).delete(
                        synchronize_session=False
                    )
                db.commit()
                log.info("%s: reset complete", sport)

            summary = _ingest_sport(db, sport, games_df, stats_df)

            # MLB: ingest pitcher stats alongside game/team data
            if pitcher_df is not None and not pitcher_df.empty:
                p_rows = pitcher_df.where(pitcher_df.notna(), None).to_dict("records")
                p_written = _upsert_pitcher_stats(db, p_rows)
                log.info("MLB: wrote %d pitcher_stats rows", p_written)

            # MLB: ingest per-start game logs
            if game_log_df is not None and not game_log_df.empty:
                gl_rows = game_log_df.where(game_log_df.notna(), None).to_dict("records")
                gl_written = _upsert_pitcher_game_logs(db, gl_rows)
                log.info("MLB: wrote %d pitcher_game_logs rows", gl_written)

            # MLB: ingest season-level team batting stats
            if batting_df is not None and not batting_df.empty:
                b_rows = batting_df.where(batting_df.notna(), None).to_dict("records")
                b_written = _upsert_team_batting_stats(db, b_rows)
                log.info("MLB: wrote %d team_batting_stats rows", b_written)

            db.commit()

            log.info(
                "%s: wrote %d games, %d team_stats, %d betting_lines",
                sport, summary["games"], summary["stats"], summary["lines"],
            )
            for k in totals:
                totals[k] += summary[k]

        except Exception as exc:
            db.rollback()
            log.error("%s DB write failed: %s", sport, exc)
        finally:
            db.close()

    if not dry_run:
        log.info(
            "━━━  TOTAL  ━━━  games=%d  team_stats=%d  betting_lines=%d",
            totals["games"], totals["stats"], totals["lines"],
        )
        log.info("historical.db written to %s", os.path.abspath(DB_PATH))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest historical game data into historical.db"
    )
    parser.add_argument(
        "--sport",
        choices=sorted(SPORT_DISPATCH.keys()),
        default=None,
        help="Ingest a single sport (omit for all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing rows for the target sport(s) before re-ingesting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and log counts without writing to the database",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow MLB pitcher game-log fetch (for daily refresh; logs already in DB)",
    )
    parser.add_argument(
        "--backfill-nba",
        action="store_true",
        help="Run NBA ATS backfill using BallDontLie (fill missing scores + cover_rate_check)",
    )
    parser.add_argument(
        "--nba-seasons",
        nargs="+",
        type=int,
        default=[2024, 2025],
        metavar="SEASON",
        help="BDL season ints for --backfill-nba (default: 2024 2025)",
    )
    parser.add_argument(
        "--backfill-injuries",
        action="store_true",
        help="Backfill NBA per-game injury flags (star/rotation absences) using nba-api",
    )
    parser.add_argument(
        "--injury-seasons",
        nargs="+",
        default=None,
        metavar="SEASON",
        help='NBA season strings for --backfill-injuries (e.g. "2022-23" "2023-24"); '
             "default: all seasons from config.py HISTORICAL_SEASONS['NBA']",
    )
    parser.add_argument(
        "--backfill-mlb",
        action="store_true",
        help="Run MLB 4-season backfill (2022-2025): ingest + ESPN run-line + cover rate",
    )
    parser.add_argument(
        "--mlb-seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Calendar years for --backfill-mlb (default: 2022 2023 2024 2025)",
    )
    parser.add_argument(
        "--backfill-nhl",
        action="store_true",
        help="Run NHL 5-season backfill (2021-22 to 2025-26): ingest + ESPN + puck line",
    )
    parser.add_argument(
        "--nhl-seasons",
        nargs="+",
        default=None,
        metavar="SEASON",
        help='Season strings for --backfill-nhl (e.g. "2021-22" "2022-23"); '
             "default: all seasons from config.py HISTORICAL_SEASONS['NHL']",
    )
    args = parser.parse_args()

    if args.backfill_nba:
        init_db()
        result = backfill_nba_ats(seasons=args.nba_seasons)
        print(result)
    elif args.backfill_injuries:
        init_db()
        result = backfill_nba_injuries(seasons=args.injury_seasons)
        print(result)
    elif args.backfill_mlb:
        result = backfill_mlb_scores(seasons=args.mlb_seasons)
        print(result)
    elif args.backfill_nhl:
        result = backfill_nhl_scores(seasons=args.nhl_seasons)
        print(result)
    else:
        run(
            sports  = [args.sport] if args.sport else None,
            reset   = args.reset,
            dry_run = args.dry_run,
            quick   = args.quick,
        )
