"""
scout/runner.py — Top-level orchestrator for the daily scout pipeline.

Called by scheduler._run_daily_scout() at 9:00 AM CT.

Flow:
  1. Ensure scouted_props table exists (safe_migrate)
  2. Fetch today's games for NBA / MLB / NHL
  3. For each game, run player prop + team market scouts
  4. Persist all ScoutedProp rows to DB
  5. Return summary dict
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

from scout.base import GameInfo, ScoutedProp

# ── Schema ─────────────────────────────────────────────────────────────────────

_CREATE_SCOUTED_PROPS = """
CREATE TABLE IF NOT EXISTS scouted_props (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scout_date          TEXT    NOT NULL,
    sport               TEXT    NOT NULL,
    game_id             TEXT    NOT NULL,
    home_team           TEXT    NOT NULL,
    away_team           TEXT    NOT NULL,
    commence_time       TEXT,
    market_type         TEXT    NOT NULL,
    player_name         TEXT,
    player_id           TEXT,
    team                TEXT,
    side                TEXT,
    threshold           REAL,
    projected_value     REAL    NOT NULL,
    projected_low_95    REAL,
    projected_high_95   REAL,
    projected_std_dev   REAL,
    hit_probability     REAL    NOT NULL,
    quality_grade       TEXT    NOT NULL,
    confidence_factors  TEXT,
    risk_factors        TEXT,
    actual_outcome_value REAL,
    actual_hit          INTEGER,
    scout_accuracy      REAL,
    data_source         TEXT,
    projection_version  TEXT,
    created_at          TEXT    DEFAULT (datetime('now'))
)
"""

_CREATE_SCOUT_CALIBRATION_LOG = """
CREATE TABLE IF NOT EXISTS scout_calibration_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT NOT NULL,
    sport           TEXT,
    quality_grade   TEXT,
    sample_size     INTEGER,
    expected_hit_pct REAL,
    actual_hit_pct  REAL,
    drift_pp        REAL,
    alert_level     TEXT
)
"""


# ── Helper ─────────────────────────────────────────────────────────────────────

def _ensure_schema(engine) -> None:
    """Idempotently create scout tables via safe_migrate."""
    try:
        import safe_migrate as sm
        sm.initialize(engine)
        sm.safe_migrate(engine, _CREATE_SCOUTED_PROPS,
                        "create_scouted_props_v1",
                        "Create scouted_props table for scout pipeline")
        sm.safe_migrate(engine, _CREATE_SCOUT_CALIBRATION_LOG,
                        "create_scout_calibration_log_v1",
                        "Create scout_calibration_log table")
    except Exception as exc:
        # If safe_migrate unavailable, fall back to direct exec
        print(f"[scout.runner] safe_migrate warning: {exc} — attempting direct create")
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text(_CREATE_SCOUTED_PROPS))
                conn.execute(text(_CREATE_SCOUT_CALIBRATION_LOG))
                conn.commit()
        except Exception as exc2:
            print(f"[scout.runner] schema init error: {exc2}")


def _games_to_game_infos(games: list[dict], sport: str) -> List[GameInfo]:
    """Convert raw game dicts (from data_sources) to GameInfo objects."""
    infos = []
    for g in games:
        try:
            # ESPN format
            home = g.get("home_team", {})
            away = g.get("away_team", {})
            if not home or not away:
                continue
            infos.append(GameInfo(
                game_id       = str(g.get("id", g.get("game_id", ""))),
                sport         = sport,
                home_team     = home.get("displayName") or home.get("name", ""),
                away_team     = away.get("displayName") or away.get("name", ""),
                home_team_id  = str(home.get("id", "")),
                away_team_id  = str(away.get("id", "")),
                commence_time = str(g.get("date", g.get("commence_time", ""))),
                venue         = g.get("venue", {}).get("fullName") if g.get("venue") else None,
                extra         = {},
            ))
        except Exception as exc:
            print(f"[scout.runner] game parse error ({sport}): {exc}")
    return infos


def _persist_props(props: List[ScoutedProp], db) -> int:
    """Insert ScoutedProp rows into scouted_props table. Returns count inserted."""
    if not props:
        return 0

    from sqlalchemy import text
    count = 0
    for p in props:
        try:
            db.execute(text("""
                INSERT INTO scouted_props
                    (scout_date, sport, game_id, home_team, away_team,
                     commence_time, market_type, player_name, player_id,
                     team, side, threshold, projected_value,
                     projected_low_95, projected_high_95, projected_std_dev,
                     hit_probability, quality_grade,
                     confidence_factors, risk_factors,
                     data_source, projection_version)
                VALUES
                    (:scout_date, :sport, :game_id, :home_team, :away_team,
                     :commence_time, :market_type, :player_name, :player_id,
                     :team, :side, :threshold, :projected_value,
                     :projected_low_95, :projected_high_95, :projected_std_dev,
                     :hit_probability, :quality_grade,
                     :confidence_factors, :risk_factors,
                     :data_source, :projection_version)
            """), {
                "scout_date":        p.scout_date,
                "sport":             p.sport,
                "game_id":           p.game_id,
                "home_team":         p.home_team,
                "away_team":         p.away_team,
                "commence_time":     p.commence_time,
                "market_type":       p.market_type,
                "player_name":       p.player_name,
                "player_id":         p.player_id,
                "team":              p.team,
                "side":              p.side,
                "threshold":         p.threshold,
                "projected_value":   p.projected_value,
                "projected_low_95":  p.projected_low_95,
                "projected_high_95": p.projected_high_95,
                "projected_std_dev": p.projected_std_dev,
                "hit_probability":   p.hit_probability,
                "quality_grade":     p.quality_grade,
                "confidence_factors": json.dumps(p.confidence_factors),
                "risk_factors":       json.dumps(p.risk_factors),
                "data_source":       p.data_source,
                "projection_version":p.projection_version,
            })
            count += 1
        except Exception as exc:
            print(f"[scout.runner] persist error for {p.player_name} / {p.market_type}: {exc}")

    db.commit()
    return count


# ── Main entry point ───────────────────────────────────────────────────────────

def run_daily_scout(engine, db) -> dict:
    """
    Full daily scout pipeline.
    Called by scheduler at 9:00 AM CT.
    Returns summary dict.
    """
    _ensure_schema(engine)

    import scout.data_sources as ds
    import scout.nba_player_props  as nba_props
    import scout.nba_team_markets  as nba_teams
    import scout.mlb_player_props  as mlb_props
    import scout.mlb_team_markets  as mlb_teams
    import scout.nhl_player_props  as nhl_props
    import scout.nhl_team_markets  as nhl_teams

    total_props = 0
    by_sport: dict[str, int] = {}
    errors: list[str] = []

    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Delete stale props from today (re-run idempotently) ───────────────────
    try:
        from sqlalchemy import text
        deleted = db.execute(
            text("DELETE FROM scouted_props WHERE scout_date = :d"),
            {"d": scout_date}
        ).rowcount
        db.commit()
        if deleted:
            print(f"[scout.runner] Cleared {deleted} stale props from {scout_date}")
    except Exception as exc:
        print(f"[scout.runner] Failed to clear stale props: {exc}")

    # ── NBA ────────────────────────────────────────────────────────────────────
    try:
        nba_games_raw = ds.get_nba_games_today()
        nba_games     = _games_to_game_infos(nba_games_raw, "NBA")
        nba_count = 0
        for game in nba_games:
            try:
                props: list[ScoutedProp] = []
                props += nba_props.scout_game(game)
                props += nba_teams.scout_game(game)
                nba_count += _persist_props(props, db)
            except Exception as exc:
                err = f"NBA game {game.game_id}: {exc}"
                errors.append(err)
                print(f"[scout.runner] {err}")
        total_props    += nba_count
        by_sport["NBA"] = nba_count
        print(f"[scout.runner] NBA: {len(nba_games)} games → {nba_count} props")
    except Exception as exc:
        errors.append(f"NBA fetch: {exc}")
        print(f"[scout.runner] NBA fetch error: {exc}")

    # ── MLB ────────────────────────────────────────────────────────────────────
    try:
        mlb_games_raw = ds.get_mlb_games_today()
        mlb_games     = _games_to_game_infos(mlb_games_raw, "MLB")
        mlb_count = 0
        for game in mlb_games:
            try:
                props = []
                props += mlb_props.scout_game(game)
                props += mlb_teams.scout_game(game)
                mlb_count += _persist_props(props, db)
            except Exception as exc:
                err = f"MLB game {game.game_id}: {exc}"
                errors.append(err)
                print(f"[scout.runner] {err}")
        total_props    += mlb_count
        by_sport["MLB"] = mlb_count
        print(f"[scout.runner] MLB: {len(mlb_games)} games → {mlb_count} props")
    except Exception as exc:
        errors.append(f"MLB fetch: {exc}")
        print(f"[scout.runner] MLB fetch error: {exc}")

    # ── NHL ────────────────────────────────────────────────────────────────────
    try:
        nhl_games_raw = ds.get_nhl_games_today()
        nhl_games     = _games_to_game_infos(nhl_games_raw, "NHL")
        nhl_count = 0
        for game in nhl_games:
            try:
                props = []
                props += nhl_props.scout_game(game)
                props += nhl_teams.scout_game(game)
                nhl_count += _persist_props(props, db)
            except Exception as exc:
                err = f"NHL game {game.game_id}: {exc}"
                errors.append(err)
                print(f"[scout.runner] {err}")
        total_props    += nhl_count
        by_sport["NHL"] = nhl_count
        print(f"[scout.runner] NHL: {len(nhl_games)} games → {nhl_count} props")
    except Exception as exc:
        errors.append(f"NHL fetch: {exc}")
        print(f"[scout.runner] NHL fetch error: {exc}")

    return {
        "scout_date":  scout_date,
        "total_props": total_props,
        "by_sport":    by_sport,
        "errors":      errors,
        "status":      "ok" if not errors else "partial",
    }
