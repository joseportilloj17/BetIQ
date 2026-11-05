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

    # S5-2: Ensure nba_forward_preds table in historical.db (created here so
    # grading can run even if no scout has written yet today).
    try:
        _ensure_nba_forward_preds_table()
    except Exception as exc:
        print(f"[scout.runner] nba_forward_preds schema error: {exc}")


def _ensure_game_infos(games) -> List[GameInfo]:
    """
    Accept either List[GameInfo] (already parsed by data_sources) or
    a list of raw dicts (legacy / test callers).  Returns List[GameInfo].
    """
    if not games:
        return []
    # If already GameInfo objects, pass through
    if isinstance(games[0], GameInfo):
        return games
    # Fallback: raw dict → GameInfo (ESPN-style)
    infos = []
    for g in games:
        try:
            home = g.get("home_team", {})
            away = g.get("away_team", {})
            infos.append(GameInfo(
                game_id       = str(g.get("id", g.get("game_id", ""))),
                sport         = g.get("sport", ""),
                home_team     = home.get("displayName") or home.get("name", ""),
                away_team     = away.get("displayName") or away.get("name", ""),
                home_team_id  = str(home.get("id", "")),
                away_team_id  = str(away.get("id", "")),
                commence_time = str(g.get("date", g.get("commence_time", ""))),
                venue         = g.get("venue", {}).get("fullName") if isinstance(g.get("venue"), dict) else None,
                extra         = {},
            ))
        except Exception as exc:
            print(f"[scout.runner] game parse error: {exc}")
    return infos


def _persist_props(props: List[ScoutedProp], db=None) -> int:
    """Insert ScoutedProp rows into scouted_props table. Returns count inserted.

    Always opens a fresh session from SessionLocal so that a failed transaction
    on the caller's shared `db` session never corrupts future persist calls.
    """
    if not props:
        return 0

    from sqlalchemy import text
    from database import SessionLocal, DB_WRITE_LOCK
    session = SessionLocal()
    count = 0
    try:
        for p in props:
            try:
                session.execute(text("""
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
                try:
                    session.rollback()
                except Exception:
                    pass
        try:
            with DB_WRITE_LOCK:
                session.commit()
        except Exception as exc:
            print(f"[scout.runner] commit error: {exc}")
            try:
                session.rollback()
            except Exception:
                pass
    finally:
        session.close()
    return count

# ── Legacy single-row insert (kept for callers that pass individual props) ────
def _persist_props_UNUSED(props: List[ScoutedProp], db) -> int:
    """Replaced by _persist_props above."""
    return _persist_props(props)


# ── NBA forward-test pre-registration ────────────────────────────────────────
# Market type → sim stat name
_MKT_TO_SIM = {
    "player_points": "PTS", "player_rebounds": "REB",
    "player_assists": "AST", "player_threes": "FG3M",
    "player_steals": "STL", "player_blocks": "BLK",
    "player_pra": "PRA",
}

_NBA_FP_DDL = """
CREATE TABLE IF NOT EXISTS nba_forward_preds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date TEXT NOT NULL,
    game_id TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market_total REAL,
    market_spread REAL,
    player_name TEXT NOT NULL,
    player_id TEXT,
    team TEXT NOT NULL,
    stat_type TEXT NOT NULL,
    line REAL NOT NULL,
    side TEXT NOT NULL,
    sim_p_over REAL NOT NULL,
    sim_p_under REAL NOT NULL,
    sim_median REAL,
    blend_projected REAL,
    blend_hit_prob REAL,
    model_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    n_draws INTEGER,
    hours_to_tip REAL,
    snapshot_type TEXT NOT NULL DEFAULT 'scout',
    created_at TEXT NOT NULL,
    player_injury_status TEXT,
    injury_context_json TEXT,
    actual_value REAL,
    actual_hit INTEGER,
    graded_at TEXT,
    is_reconstructed INTEGER DEFAULT 0,
    player_name_norm TEXT
)
"""


def _normalize_player_name(name: str | None) -> str | None:
    """Normalize NBA player name for cross-source matching.

    Strips diacritics, trailing Jr./Sr./II/III/IV, lowercases, collapses
    whitespace.  Handles ESPN ↔ BDL/NBA.com name mismatches.
    """
    if not name:
        return None
    import unicodedata
    import re
    # NFD decompose → strip combining marks (diacritics)
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().strip()
    # Remove common suffixes
    s = re.sub(r'\s+(jr\.?|sr\.?|ii+|iv|v)$', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def _ensure_nba_forward_preds_table():
    """Create nba_forward_preds + indexes in historical.db if not exists."""
    import os, sqlite3
    hist_path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                             "historical.db")
    hist_path = os.path.normpath(hist_path)
    con = sqlite3.connect(hist_path, timeout=30)
    con.execute("PRAGMA busy_timeout = 30000")
    con.execute(_NBA_FP_DDL)
    con.execute("CREATE INDEX IF NOT EXISTS idx_nfp_game_date "
                "ON nba_forward_preds(game_date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_nfp_lookup "
                "ON nba_forward_preds(game_id, player_name, stat_type)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_nfp_ungraded "
                "ON nba_forward_preds(game_date) WHERE actual_hit IS NULL")
    con.commit()
    con.close()


def _persist_nba_forward_preds(
    props: List[ScoutedProp],
    sim,
    game: GameInfo,
    market_total,
    market_spread,
    injury_statuses: dict | None,
    hours_to_tip: float | None,
    snapshot_type: str = "scout",
) -> int:
    """Write NBA sim predictions to nba_forward_preds in historical.db.

    One row per player × stat × line × snapshot.  Never rewritten.
    """
    if sim is None or not props:
        return 0

    import os
    import sqlite3
    import hashlib
    import numpy as np

    hist_path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                             "historical.db")
    hist_path = os.path.normpath(hist_path)

    con = sqlite3.connect(hist_path, timeout=30)
    con.execute("PRAGMA busy_timeout = 30000")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute(_NBA_FP_DDL)
    con.execute("CREATE INDEX IF NOT EXISTS idx_nfp_game_date "
                "ON nba_forward_preds(game_date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_nfp_lookup "
                "ON nba_forward_preds(game_id, player_name, stat_type)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_nfp_ungraded "
                "ON nba_forward_preds(game_date) WHERE actual_hit IS NULL")

    # Config hash: identifies the exact sim configuration
    inj_sorted = sorted((injury_statuses or {}).items())
    cfg = f"{sim.n_draws}|{market_total}|{market_spread}|{inj_sorted}"
    config_hash = hashlib.md5(cfg.encode()).hexdigest()[:12]

    inj_json = json.dumps(injury_statuses) if injury_statuses else None
    now_str = datetime.now(timezone.utc).isoformat()
    # NBA uses ET dates — convert UTC commence_time to ET for matching game_logs
    try:
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        _tip = datetime.fromisoformat(
            game.commence_time.replace("Z", "+00:00"))
        game_date = _tip.astimezone(_et).strftime("%Y-%m-%d")
    except Exception:
        game_date = (game.commence_time or "")[:10]

    count = 0
    for p in props:
        if p.sport != "NBA":
            continue
        sim_stat = _MKT_TO_SIM.get(p.market_type)
        if sim_stat is None or p.player_name not in sim._player_idx:
            continue
        try:
            sp_over = sim.p_over(p.player_name, sim_stat, p.threshold)
            sp_under = sim.p_under(p.player_name, sim_stat, p.threshold)
            arr = sim._get(p.player_name, sim_stat)
            avail = sim._player_available(p.player_name)
            n_avail = int(avail.sum())
            median_val = float(np.median(arr[avail])) if n_avail > 0 else None
        except (KeyError, Exception):
            continue

        con.execute("""
            INSERT INTO nba_forward_preds (
                game_date, game_id, home_team, away_team,
                market_total, market_spread,
                player_name, player_id, team, stat_type,
                line, side, sim_p_over, sim_p_under, sim_median,
                blend_projected, blend_hit_prob,
                model_version, config_hash, n_draws,
                hours_to_tip, snapshot_type, created_at,
                player_injury_status, injury_context_json,
                is_reconstructed, player_name_norm
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            game_date, p.game_id, p.home_team, p.away_team,
            market_total, market_spread,
            p.player_name, p.player_id, p.team, p.market_type,
            p.threshold, p.side,
            round(sp_over, 6), round(sp_under, 6),
            round(median_val, 2) if median_val is not None else None,
            p.projected_value, p.hit_probability,
            p.projection_version, config_hash, sim.n_draws,
            hours_to_tip, snapshot_type, now_str,
            (injury_statuses or {}).get(p.player_name),
            inj_json,
            0,
            _normalize_player_name(p.player_name),
        ))
        count += 1

    con.commit()
    con.close()
    return count


# ── NBA forward-test grading ────────────────────────────────────────────────
# Maps market_type in nba_forward_preds → column(s) in nba_player_game_logs
_STAT_COL_MAP = {
    "player_points":   ("pts",),
    "player_rebounds":  ("reb",),
    "player_assists":   ("ast",),
    "player_threes":    ("fg3m",),
    "player_steals":    ("stl",),
    "player_blocks":    ("blk",),
    "player_pra":       ("pts", "reb", "ast"),
}


def grade_nba_forward_preds() -> dict:
    """Grade ungraded rows in nba_forward_preds from nba_player_game_logs.

    Matches on normalized player_name + game_date (ET dates in both tables).
    Sets actual_value, actual_hit, graded_at.  Returns summary dict.
    """
    import os
    import sqlite3

    hist_path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                             "historical.db")
    hist_path = os.path.normpath(hist_path)

    con = sqlite3.connect(hist_path, timeout=30)
    con.execute("PRAGMA busy_timeout = 30000")
    con.row_factory = sqlite3.Row

    # Bail early if table doesn't exist yet (first scout hasn't run)
    tbl = con.execute(
        "SELECT name FROM sqlite_master WHERE name='nba_forward_preds'"
    ).fetchone()
    if not tbl:
        con.close()
        return {"graded": 0, "ungraded_remaining": 0, "note": "table not created yet"}

    # Get distinct game_dates with ungraded preds
    dates = con.execute("""
        SELECT DISTINCT game_date FROM nba_forward_preds
        WHERE actual_hit IS NULL
    """).fetchall()

    if not dates:
        con.close()
        return {"graded": 0, "ungraded_remaining": 0}

    graded = 0
    not_found = 0

    for (gdate,) in dates:
        # Fetch all game logs for this date
        logs = con.execute("""
            SELECT player_name, pts, reb, ast, fg3m, stl, blk
            FROM nba_player_game_logs
            WHERE game_date = ?
        """, (gdate,)).fetchall()

        if not logs:
            continue

        # Build lookup: normalized_name -> {col: value}
        log_map: dict = {}
        for row in logs:
            norm = _normalize_player_name(row["player_name"])
            if norm:
                log_map[norm] = {
                    "pts": row["pts"], "reb": row["reb"], "ast": row["ast"],
                    "fg3m": row["fg3m"], "stl": row["stl"], "blk": row["blk"],
                }

        # Get ungraded preds for this date
        preds = con.execute("""
            SELECT id, player_name, player_name_norm, stat_type, line, side
            FROM nba_forward_preds
            WHERE game_date = ? AND actual_hit IS NULL
        """, (gdate,)).fetchall()

        now_str = datetime.now(timezone.utc).isoformat()

        for pred in preds:
            # Use stored norm if available, else compute on the fly
            norm = pred["player_name_norm"] or _normalize_player_name(pred["player_name"])
            stat_type = pred["stat_type"]
            line = pred["line"]
            side = pred["side"]

            if norm not in log_map:
                not_found += 1
                continue

            cols = _STAT_COL_MAP.get(stat_type)
            if cols is None:
                continue

            plog = log_map[norm]
            actual = sum(plog.get(c, 0) or 0 for c in cols)

            if side == "over":
                hit = 1 if actual > line else 0
            else:
                hit = 1 if actual < line else 0

            con.execute("""
                UPDATE nba_forward_preds
                SET actual_value = ?, actual_hit = ?, graded_at = ?
                WHERE id = ?
            """, (round(actual, 1), hit, now_str, pred["id"]))
            graded += 1

    con.commit()

    remaining = con.execute(
        "SELECT COUNT(*) FROM nba_forward_preds WHERE actual_hit IS NULL"
    ).fetchone()[0]

    con.close()

    if graded:
        print(f"[scout.runner] S5-2 grading: {graded} preds graded, "
              f"{remaining} ungraded remaining, {not_found} player-not-found")

    return {"graded": graded, "ungraded_remaining": remaining,
            "not_found": not_found}


def run_pretip_nba_forward_preds(min_before_low: int = 30,
                                  min_before_high: int = 60) -> dict:
    """Re-run NBA sim for games tipping soon and persist pre-tip snapshot.

    Called by scheduler every cycle during game hours.  Only targets games
    whose tip-off is within [min_before_low, min_before_high] minutes from now.
    Returns summary dict with game count and rows written.
    """
    import scout.data_sources as ds
    import scout.nba_player_props as nba_props
    import scout.nba_team_markets as nba_teams

    games_raw = ds.get_nba_games_today()
    games = _ensure_game_infos(games_raw)
    if not games:
        return {"games": 0, "rows": 0}

    now_utc = datetime.now(timezone.utc)
    targeted = []
    for g in games:
        try:
            tip = datetime.fromisoformat(
                g.commence_time.replace("Z", "+00:00"))
            mins = (tip - now_utc).total_seconds() / 60
            if min_before_low <= mins <= min_before_high:
                targeted.append(g)
        except Exception:
            continue

    if not targeted:
        return {"games": 0, "rows": 0}

    total_rows = 0
    game_ids = []
    for game in targeted:
        try:
            _mkt = nba_teams._fetch_market_lines(game)
            if _mkt and "market_total" in _mkt and "market_spread" in _mkt:
                game.extra["market_total"] = _mkt["market_total"]
                game.extra["market_spread"] = _mkt["market_spread"]

            props = nba_props.scout_game(game)
            if not props:
                continue

            extra = game.extra if isinstance(game.extra, dict) else {}
            _h = extra.get("home_abbr", game.home_team or "")
            _a = extra.get("away_abbr", game.away_team or "")
            _sim = nba_props.get_game_sim(_a, _h)
            _ctx = nba_props.get_game_sim_context(_a, _h)

            if _sim is not None:
                rows = _persist_nba_forward_preds(
                    props, _sim, game,
                    extra.get("market_total"),
                    extra.get("market_spread"),
                    _ctx.get("injury_statuses") if _ctx else None,
                    _ctx.get("hours_to_tip") if _ctx else None,
                    snapshot_type="pre_tip",
                )
                total_rows += rows
                game_ids.append(game.game_id)
                if rows:
                    print(f"[scout.runner] S5-2 pre-tip: {rows} preds for "
                          f"{_a}@{_h}")
        except Exception as exc:
            print(f"[scout.runner] S5-2 pre-tip error {game.game_id}: {exc}")

    return {"games": len(game_ids), "rows": total_rows, "game_ids": game_ids}


def __persist_props_old(props: List[ScoutedProp], db) -> int:
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
            try:
                db.rollback()
            except Exception:
                pass

    try:
        db.commit()
    except Exception as exc:
        print(f"[scout.runner] commit error: {exc}")
        try:
            db.rollback()
        except Exception:
            pass
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
        from database import DB_WRITE_LOCK as _WL
        deleted = db.execute(
            text("DELETE FROM scouted_props WHERE scout_date = :d"),
            {"d": scout_date}
        ).rowcount
        with _WL:
            db.commit()
        if deleted:
            print(f"[scout.runner] Cleared {deleted} stale props from {scout_date}")
    except Exception as exc:
        print(f"[scout.runner] Failed to clear stale props: {exc}")
        try:
            db.rollback()
        except Exception:
            pass

    def _safe_persist(props, db):
        """Persist with rollback on failure so session stays usable."""
        try:
            return _persist_props(props, db)
        except Exception as exc:
            print(f"[scout.runner] persist batch error: {exc}")
            try:
                db.rollback()
            except Exception:
                pass
            return 0

    # ── NBA ────────────────────────────────────────────────────────────────────
    try:
        nba_games_raw = ds.get_nba_games_today()
        nba_games     = _ensure_game_infos(nba_games_raw)
        nba_count = 0
        for game in nba_games:
            try:
                # B3-F1: Wire market spreads/totals into game.extra before
                # any scout_game call.  Fail loudly if missing.
                _mkt = nba_teams._fetch_market_lines(game)
                if _mkt and "market_total" in _mkt and "market_spread" in _mkt:
                    game.extra["market_total"]  = _mkt["market_total"]
                    game.extra["market_spread"] = _mkt["market_spread"]
                    game.extra["market_book"]   = _mkt.get("book", "unknown")
                    print(f"[scout.runner] {game.away_team} @ {game.home_team}: "
                          f"spread={_mkt['market_spread']:+.1f}  "
                          f"total={_mkt['market_total']:.1f}  "
                          f"book={_mkt.get('book')}")
                else:
                    print(f"[scout.runner] [NO MARKET ANCHOR] "
                          f"{game.away_team} @ {game.home_team} — "
                          f"no spread/total in fixtures; "
                          f"projections will use model-only fallback")
                    game.extra["_no_market_anchor"] = True

                props: list[ScoutedProp] = []
                nba_player_props = nba_props.scout_game(game)
                props += nba_player_props
                props += nba_teams.scout_game(game)
                nba_count += _safe_persist(props, db)

                # S5-2: Persist sim predictions for forward test
                try:
                    extra = game.extra if hasattr(game, "extra") and isinstance(game.extra, dict) else {}
                    _h = extra.get("home_abbr", game.home_team or "")
                    _a = extra.get("away_abbr", game.away_team or "")
                    _sim = nba_props.get_game_sim(_a, _h)
                    _ctx = nba_props.get_game_sim_context(_a, _h)
                    if _sim is not None and nba_player_props:
                        _fp = _persist_nba_forward_preds(
                            nba_player_props, _sim, game,
                            extra.get("market_total"),
                            extra.get("market_spread"),
                            _ctx.get("injury_statuses") if _ctx else None,
                            _ctx.get("hours_to_tip") if _ctx else None,
                            snapshot_type="scout",
                        )
                        if _fp:
                            print(f"[scout.runner] S5-2: {_fp} forward preds "
                                  f"written for {_a}@{_h}")
                except Exception as fp_exc:
                    print(f"[scout.runner] S5-2 forward preds error: {fp_exc}")
            except Exception as exc:
                err = f"NBA game {game.game_id}: {exc}"
                errors.append(err)
                print(f"[scout.runner] {err}")
                try:
                    db.rollback()
                except Exception:
                    pass
        total_props    += nba_count
        by_sport["NBA"] = nba_count
        print(f"[scout.runner] NBA: {len(nba_games)} games → {nba_count} props")
    except Exception as exc:
        errors.append(f"NBA fetch: {exc}")
        print(f"[scout.runner] NBA fetch error: {exc}")

    # ── MLB ────────────────────────────────────────────────────────────────────
    try:
        mlb_games_raw = ds.get_mlb_games_today()
        mlb_games     = _ensure_game_infos(mlb_games_raw)
        mlb_count = 0
        for game in mlb_games:
            try:
                props = []
                props += mlb_props.scout_game(game)
                props += mlb_teams.scout_game(game)
                mlb_count += _safe_persist(props, db)
            except Exception as exc:
                err = f"MLB game {game.game_id}: {exc}"
                errors.append(err)
                print(f"[scout.runner] {err}")
                try:
                    db.rollback()
                except Exception:
                    pass
        total_props    += mlb_count
        by_sport["MLB"] = mlb_count
        print(f"[scout.runner] MLB: {len(mlb_games)} games → {mlb_count} props")
    except Exception as exc:
        errors.append(f"MLB fetch: {exc}")
        print(f"[scout.runner] MLB fetch error: {exc}")

    # ── NHL ────────────────────────────────────────────────────────────────────
    try:
        nhl_games_raw = ds.get_nhl_games_today()
        nhl_games     = _ensure_game_infos(nhl_games_raw)
        nhl_count = 0
        for game in nhl_games:
            try:
                props = []
                props += nhl_props.scout_game(game)
                props += nhl_teams.scout_game(game)
                nhl_count += _safe_persist(props, db)
            except Exception as exc:
                err = f"NHL game {game.game_id}: {exc}"
                errors.append(err)
                print(f"[scout.runner] {err}")
                try:
                    db.rollback()
                except Exception:
                    pass
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
