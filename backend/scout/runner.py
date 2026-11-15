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
import logging
import traceback
from datetime import datetime, timezone
from typing import List

from scout.base import GameInfo, ScoutedProp

logger = logging.getLogger(__name__)

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
    "player_points_rebounds_assists": "PRA",
}

# Odds API market_key → nba_forward_preds stat_type mapping
# Handles alternate-market keys and the PRA market name difference
_MKT_KEY_TO_STAT = {
    "player_points": "player_points",
    "player_points_alternate": "player_points",
    "player_rebounds": "player_rebounds",
    "player_rebounds_alternate": "player_rebounds",
    "player_assists": "player_assists",
    "player_assists_alternate": "player_assists",
    "player_threes": "player_threes",
    "player_steals": "player_steals",
    "player_blocks": "player_blocks",
    "player_points_rebounds_assists": "player_pra",
}

# Market keys that are alternate lines (not the main market)
_ALT_MARKET_KEYS = {
    "player_points_alternate",
    "player_rebounds_alternate",
    "player_assists_alternate",
}

# Normalize Odds-API team names → ESPN team names for game matching
_TEAM_NORM = {
    "Los Angeles Clippers": "LA Clippers",
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
    player_name_norm TEXT,
    bdl_player_id INTEGER,
    market_key TEXT,
    line_depth TEXT,
    best_price REAL,
    best_book TEXT,
    devig_implied REAL,
    devig_method TEXT,
    n_books INTEGER,
    no_market INTEGER DEFAULT 0
)
"""

_NBA_ALIAS_DDL = """
CREATE TABLE IF NOT EXISTS nba_player_alias (
    alias_name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    bdl_player_id INTEGER,
    source TEXT DEFAULT 'manual',
    PRIMARY KEY (alias_name)
)
"""

# Seed aliases for known Odds API ↔ BDL mismatches
_SEED_ALIASES = [
    ("A.J. Green",         "AJ Green",           1631260),
    ("C.J. McCollum",      "CJ McCollum",         203468),
    ("Carlton Carrington", "Bub Carrington",     1642267),
    ("G.G. Jackson",       "GG Jackson",         1641713),
    ("Herb Jones",         "Herbert Jones",      1630529),
    ("R.J. Barrett",       "RJ Barrett",         1629628),
    ("Ron Holland",        "Ronald Holland II",  1641842),
]


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
    """Create nba_forward_preds + alias table + indexes in historical.db."""
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
    # Add columns if table existed before they were added
    for _col, _type in [("bdl_player_id", "INTEGER"),
                        ("data_error", "TEXT"),
                        ("market_key", "TEXT"),
                        ("line_depth", "TEXT"),
                        ("best_price", "REAL"),
                        ("best_book", "TEXT"),
                        ("devig_implied", "REAL"),
                        ("devig_method", "TEXT"),
                        ("n_books", "INTEGER"),
                        ("no_market", "INTEGER DEFAULT 0")]:
        try:
            con.execute(f"ALTER TABLE nba_forward_preds ADD COLUMN {_col} {_type}")
        except Exception:
            pass  # column already exists
    # Alias table for Odds API ↔ BDL name mismatches
    con.execute(_NBA_ALIAS_DDL)
    for alias, canonical, bdl_id in _SEED_ALIASES:
        con.execute(
            "INSERT OR IGNORE INTO nba_player_alias "
            "(alias_name, canonical_name, bdl_player_id, source) "
            "VALUES (?, ?, ?, 'seed')",
            (alias, canonical, bdl_id),
        )
    con.commit()
    con.close()


def _resolve_bdl_player_id(con, player_name: str, game_date: str) -> int | None:
    """Look up BDL player_id via alias table, then fall back to game_logs norm match.

    Returns bdl_player_id (int) or None.
    """
    # 1. Check alias table (exact match on Odds API name)
    row = con.execute(
        "SELECT bdl_player_id FROM nba_player_alias WHERE alias_name = ?",
        (player_name,),
    ).fetchone()
    if row and row[0]:
        return row[0]

    # 2. Fall back: match normalized name against game_logs for this date
    norm = _normalize_player_name(player_name)
    if not norm:
        return None
    row = con.execute(
        "SELECT player_id FROM nba_player_game_logs "
        "WHERE game_date = ? LIMIT 1",
        (game_date,),
    ).fetchone()
    # If game_logs exist for the date, try matching by normalized name
    if row:
        # nba_player_game_logs has player_name; match on normalized
        rows = con.execute(
            "SELECT DISTINCT player_id, player_name FROM nba_player_game_logs "
            "WHERE game_date = ?",
            (game_date,),
        ).fetchall()
        for r in rows:
            if _normalize_player_name(r[1]) == norm:
                return r[0]
    return None


def _get_market_lines_for_game(
    game_date: str,
    home_team: str,
    away_team: str,
) -> list[dict]:
    """Query player_props for all market-posted lines for a game.

    Returns list of dicts with keys: player_name, market_key, stat_type,
    line_depth, point, over_price, under_price, best_book, n_books,
    devig_implied, devig_method.

    Groups by (player_name, stat_type, point) across bookmakers.
    For two-sided lines, devig via multiplicative removal.
    For one-sided alts, store raw implied prob with method='one_sided_raw'.
    """
    import os
    import sqlite3

    hist_path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                             "historical.db")
    hist_path = os.path.normpath(hist_path)
    con = sqlite3.connect(hist_path, timeout=30)
    con.execute("PRAGMA busy_timeout = 30000")
    con.row_factory = sqlite3.Row

    # Normalize team names: Odds API → ESPN
    norm_home = _TEAM_NORM.get(home_team, home_team)
    norm_away = _TEAM_NORM.get(away_team, away_team)

    # Fetch latest snapshot per (bookmaker, market_key, player_name, over_under, point)
    rows = con.execute("""
        SELECT p1.player_name, p1.market_key, p1.point, p1.over_under,
               p1.price, p1.bookmaker
        FROM player_props p1
        JOIN (SELECT bookmaker, market_key, player_name, over_under, point,
                     MAX(fetched_at) AS max_fa
              FROM player_props
              WHERE sport_key = 'basketball_nba'
                AND game_date = ?
                AND (home_team = ? OR home_team = ?)
              GROUP BY bookmaker, market_key, player_name, over_under, point
             ) p2
          ON p1.bookmaker   = p2.bookmaker
         AND p1.market_key  = p2.market_key
         AND p1.player_name = p2.player_name
         AND p1.over_under  = p2.over_under
         AND p1.point       = p2.point
         AND p1.fetched_at  = p2.max_fa
        WHERE p1.sport_key = 'basketball_nba'
          AND p1.game_date = ?
          AND (p1.home_team = ? OR p1.home_team = ?)
    """, (game_date, home_team, norm_home,
          game_date, home_team, norm_home)).fetchall()

    # If no rows matched on home_team, try away_team (in case team order differs)
    if not rows:
        rows = con.execute("""
            SELECT p1.player_name, p1.market_key, p1.point, p1.over_under,
                   p1.price, p1.bookmaker
            FROM player_props p1
            JOIN (SELECT bookmaker, market_key, player_name, over_under, point,
                         MAX(fetched_at) AS max_fa
                  FROM player_props
                  WHERE sport_key = 'basketball_nba'
                    AND game_date = ?
                    AND (away_team = ? OR away_team = ?)
                  GROUP BY bookmaker, market_key, player_name, over_under, point
                 ) p2
              ON p1.bookmaker   = p2.bookmaker
             AND p1.market_key  = p2.market_key
             AND p1.player_name = p2.player_name
             AND p1.over_under  = p2.over_under
             AND p1.point       = p2.point
             AND p1.fetched_at  = p2.max_fa
            WHERE p1.sport_key = 'basketball_nba'
              AND p1.game_date = ?
              AND (p1.away_team = ? OR p1.away_team = ?)
        """, (game_date, away_team, norm_away,
              game_date, away_team, norm_away)).fetchall()

    con.close()

    if not rows:
        return []

    # Group by (player_name, market_key, point) → collect best over/under prices
    from collections import defaultdict
    grouped = defaultdict(lambda: {"over_prices": [], "under_prices": [],
                                   "books": set()})
    for r in rows:
        key = (r["player_name"], r["market_key"], r["point"])
        price = r["price"]
        if price is None or price <= 0:
            continue
        grouped[key]["books"].add(r["bookmaker"])
        if r["over_under"] == "over":
            grouped[key]["over_prices"].append((price, r["bookmaker"]))
        elif r["over_under"] == "under":
            grouped[key]["under_prices"].append((price, r["bookmaker"]))

    # ── Pass 1: compute per-(player, stat_type, book) overround from main lines ─
    #    Used to devig one-sided alt lines.
    # overround_map[(player, stat_type, book)] = total_implied (e.g. 1.05)
    overround_map: dict[tuple, float] = {}
    # Also track per-(stat_type, book) as fallback
    overround_by_stat_book: dict[tuple, list] = defaultdict(list)

    for (player_name, market_key, point), data in grouped.items():
        stat_type = _MKT_KEY_TO_STAT.get(market_key)
        if stat_type is None:
            continue
        if market_key in _ALT_MARKET_KEYS:
            continue  # only main lines contribute overround
        over_prices = data["over_prices"]
        under_prices = data["under_prices"]
        if not over_prices or not under_prices:
            continue
        # Per-book overround: use each book's own over+under for the most accurate margin
        book_prices: dict[str, dict] = {}
        for price, book in over_prices:
            book_prices.setdefault(book, {})["over"] = price
        for price, book in under_prices:
            book_prices.setdefault(book, {})["under"] = price
        for book, bp in book_prices.items():
            if "over" in bp and "under" in bp:
                total_imp = 1.0 / bp["over"] + 1.0 / bp["under"]
                key = (player_name, stat_type, book)
                # Keep the overround closest to 1.0 (least vig) if multiple main lines
                if key not in overround_map or abs(total_imp - 1.0) < abs(overround_map[key] - 1.0):
                    overround_map[key] = total_imp
                overround_by_stat_book[(stat_type, book)].append(total_imp)

    # Precompute median overround per (stat_type, book) as last-resort fallback
    median_overround: dict[tuple, float] = {}
    for (st, bk), vals in overround_by_stat_book.items():
        sv = sorted(vals)
        n = len(sv)
        median_overround[(st, bk)] = sv[n // 2] if n % 2 else (sv[n//2 - 1] + sv[n//2]) / 2

    # Global fallback: median across all main lines
    all_overrounds = [v for vals in overround_by_stat_book.values() for v in vals]
    global_median_overround = 1.05  # safe default
    if all_overrounds:
        sv = sorted(all_overrounds)
        n = len(sv)
        global_median_overround = sv[n // 2] if n % 2 else (sv[n//2 - 1] + sv[n//2]) / 2

    # ── Pass 2: build results with devig ─────────────────────────────────────
    results = []
    for (player_name, market_key, point), data in grouped.items():
        stat_type = _MKT_KEY_TO_STAT.get(market_key)
        if stat_type is None or stat_type not in _MKT_TO_SIM:
            continue

        line_depth = "alt" if market_key in _ALT_MARKET_KEYS else "main"

        over_prices = data["over_prices"]
        under_prices = data["under_prices"]

        best_over = max(over_prices, key=lambda x: x[0]) if over_prices else None
        best_under = max(under_prices, key=lambda x: x[0]) if under_prices else None

        if best_over and best_under:
            # Two-sided: multiplicative devig
            imp_over = 1.0 / best_over[0]
            imp_under = 1.0 / best_under[0]
            total_imp = imp_over + imp_under
            devig_over = imp_over / total_imp
            devig_method = "multiplicative"
            if best_over[0] >= best_under[0]:
                best_price = best_over[0]
                best_book = best_over[1]
                side = "over"
                devig_implied = devig_over
            else:
                best_price = best_under[0]
                best_book = best_under[1]
                side = "under"
                devig_implied = 1.0 - devig_over
        elif best_over:
            # One-sided over: borrow overround from main line
            best_price = best_over[0]
            best_book = best_over[1]
            side = "over"
            raw_imp = 1.0 / best_over[0]
            # Look up borrowed overround: player/stat/book → stat/book → global
            borrowed = overround_map.get((player_name, stat_type, best_book))
            if borrowed is None:
                borrowed = median_overround.get((stat_type, best_book))
            if borrowed is None:
                borrowed = global_median_overround
            devig_implied = raw_imp / borrowed
            devig_method = "borrowed_margin"
        elif best_under:
            best_price = best_under[0]
            best_book = best_under[1]
            side = "under"
            raw_imp = 1.0 / best_under[0]
            borrowed = overround_map.get((player_name, stat_type, best_book))
            if borrowed is None:
                borrowed = median_overround.get((stat_type, best_book))
            if borrowed is None:
                borrowed = global_median_overround
            devig_implied = raw_imp / borrowed
            devig_method = "borrowed_margin"
        else:
            continue

        results.append({
            "player_name": player_name,
            "market_key": market_key,
            "stat_type": stat_type,
            "line_depth": line_depth,
            "point": point,
            "side": side,
            "best_price": round(best_price, 4),
            "best_book": best_book,
            "n_books": len(data["books"]),
            "devig_implied": round(devig_implied, 6),
            "devig_method": devig_method,
        })

    return results


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

    Stores one row per player × stat × market-posted line × snapshot.
    Queries player_props for all lines the market posted for this game,
    then computes sim P(over)/P(under) at each line.

    Falls back to ScoutedProp thresholds for players in the sim but not
    in player_props (ensuring at least one row per sim player×stat).
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
    # Migrate columns if table pre-dates them (matches _ensure_nba_forward_preds_table)
    for _col, _type in [("bdl_player_id", "INTEGER"),
                        ("data_error", "TEXT"),
                        ("market_key", "TEXT"),
                        ("line_depth", "TEXT"),
                        ("best_price", "REAL"),
                        ("best_book", "TEXT"),
                        ("devig_implied", "REAL"),
                        ("devig_method", "TEXT"),
                        ("n_books", "INTEGER"),
                        ("no_market", "INTEGER DEFAULT 0")]:
        try:
            con.execute(f"ALTER TABLE nba_forward_preds ADD COLUMN {_col} {_type}")
        except Exception:
            pass  # column already exists

    # Config hash: identifies the exact sim configuration
    inj_sorted = sorted((injury_statuses or {}).items())
    cfg = f"{sim.n_draws}|{market_total}|{market_spread}|{inj_sorted}"
    config_hash = hashlib.md5(cfg.encode()).hexdigest()[:12]

    inj_json = json.dumps(injury_statuses) if injury_statuses else None
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.isoformat()
    # NBA uses ET dates — convert UTC commence_time to ET for matching game_logs
    _tip_utc = None
    try:
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        _tip_utc = datetime.fromisoformat(
            game.commence_time.replace("Z", "+00:00"))
        game_date = _tip_utc.astimezone(_et).strftime("%Y-%m-%d")
    except Exception:
        game_date = (game.commence_time or "")[:10]

    # Per-game tip freeze: never write predictions after the game has tipped
    if _tip_utc is not None and now_utc > _tip_utc:
        logger.warning("[scout.runner] FREEZE: skipping nba_forward_preds for "
                       "%s (tip %s, now %s)",
                       game.game_id, _tip_utc.isoformat(), now_str)
        con.close()
        return 0

    # Pre-build alias lookup (Odds API name → bdl_player_id) for this batch
    _alias_cache: dict[str, int | None] = {}

    # Build set of sim-available players (for matching market names)
    sim_players = set(sim._player_idx.keys())

    # Load alias table for Odds API → sim name resolution
    alias_to_sim: dict[str, str] = {}  # odds_api_name → sim_name
    try:
        alias_rows = con.execute(
            "SELECT alias_name, canonical_name FROM nba_player_alias"
        ).fetchall()
        for a_name, c_name in alias_rows:
            alias_to_sim[a_name] = c_name
    except Exception:
        pass

    def _resolve_sim_name(odds_api_name: str) -> str | None:
        """Resolve an Odds API player name to a sim player name."""
        if odds_api_name in sim_players:
            return odds_api_name
        if odds_api_name in alias_to_sim:
            canon = alias_to_sim[odds_api_name]
            if canon in sim_players:
                return canon
        # Normalized match
        norm = _normalize_player_name(odds_api_name)
        for sp in sim_players:
            if _normalize_player_name(sp) == norm:
                return sp
        return None

    # ── Query market-posted lines from player_props ──
    market_lines = _get_market_lines_for_game(
        game_date, game.home_team, game.away_team
    )

    # Build a lookup: ScoutedProp player_name → ScoutedProp (for fallback metadata)
    prop_lookup: dict[str, ScoutedProp] = {}
    for p in props:
        if p.sport == "NBA":
            prop_lookup[p.player_name] = p

    # Track which (sim_name, stat_type) combos were stored from market lines
    stored_from_market: set[tuple[str, str]] = set()

    count = 0

    # ── Primary path: store one row per market-posted line ──
    for mkt in market_lines:
        sim_name = _resolve_sim_name(mkt["player_name"])
        if sim_name is None:
            continue
        sim_stat = _MKT_TO_SIM.get(mkt["stat_type"])
        if sim_stat is None:
            continue

        try:
            sp_over = sim.p_over(sim_name, sim_stat, mkt["point"])
            sp_under = sim.p_under(sim_name, sim_stat, mkt["point"])
            arr = sim._get(sim_name, sim_stat)
            avail = sim._player_available(sim_name)
            n_avail = int(avail.sum())
            median_val = float(np.median(arr[avail])) if n_avail > 0 else None
        except (KeyError, Exception):
            continue

        # Determine side based on sim probability (over if P(over) >= 0.5)
        side = "over" if sp_over >= 0.5 else "under"

        # Get ScoutedProp metadata if available (for blend_projected, etc.)
        sp = prop_lookup.get(sim_name)

        # Resolve bdl_player_id
        if sim_name not in _alias_cache:
            _alias_cache[sim_name] = _resolve_bdl_player_id(
                con, mkt["player_name"], game_date)
        bdl_pid = _alias_cache[sim_name]

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
                is_reconstructed, player_name_norm, bdl_player_id,
                market_key, line_depth, best_price, best_book,
                devig_implied, devig_method, n_books
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            game_date, game.game_id, game.home_team, game.away_team,
            market_total, market_spread,
            sim_name,
            sp.player_id if sp else None,
            sp.team if sp else "",
            mkt["stat_type"],
            mkt["point"], side,
            round(sp_over, 6), round(sp_under, 6),
            round(median_val, 2) if median_val is not None else None,
            sp.projected_value if sp else None,
            sp.hit_probability if sp else None,
            sp.projection_version if sp else "sim_only",
            config_hash, sim.n_draws,
            hours_to_tip, snapshot_type, now_str,
            (injury_statuses or {}).get(sim_name),
            inj_json,
            0,
            _normalize_player_name(sim_name),
            bdl_pid,
            mkt["market_key"], mkt["line_depth"],
            mkt["best_price"], mkt["best_book"],
            mkt["devig_implied"], mkt["devig_method"],
            mkt["n_books"],
        ))
        count += 1
        stored_from_market.add((sim_name, mkt["stat_type"]))

    # ── Fallback: for sim players with no market lines, store at ScoutedProp threshold ──
    for p in props:
        if p.sport != "NBA":
            continue
        sim_stat = _MKT_TO_SIM.get(p.market_type)
        if sim_stat is None or p.player_name not in sim._player_idx:
            continue
        if (p.player_name, p.market_type) in stored_from_market:
            continue  # already stored from market lines

        try:
            sp_over = sim.p_over(p.player_name, sim_stat, p.threshold)
            sp_under = sim.p_under(p.player_name, sim_stat, p.threshold)
            arr = sim._get(p.player_name, sim_stat)
            avail = sim._player_available(p.player_name)
            n_avail = int(avail.sum())
            median_val = float(np.median(arr[avail])) if n_avail > 0 else None
        except (KeyError, Exception):
            continue

        if p.player_name not in _alias_cache:
            _alias_cache[p.player_name] = _resolve_bdl_player_id(
                con, p.player_name, game_date)
        bdl_pid = _alias_cache[p.player_name]

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
                is_reconstructed, player_name_norm, bdl_player_id,
                market_key, line_depth, best_price, best_book,
                devig_implied, devig_method, n_books,
                no_market
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            bdl_pid,
            None, "fallback",  # market_key, line_depth
            None, None,        # best_price, best_book
            None, None,        # devig_implied, devig_method
            None,              # n_books
            1,                 # no_market
        ))
        count += 1

    mkt_count = len(stored_from_market)
    fallback_count = count - sum(1 for _ in [])  # will log separately
    if market_lines:
        logger.info("[scout.runner] S5-2: %d market lines found, %d stored "
                    "(from %d player×stat combos)",
                    len(market_lines), count, mkt_count)
    else:
        logger.warning("[scout.runner] S5-2: 0 market lines found for %s on %s "
                       "— using fallback thresholds only (%d rows)",
                       game.game_id, game_date, count)

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

    Resolution order for matching:
      1. bdl_player_id (if stored) → game_logs.player_id
      2. Alias table: alias_name → canonical norm → game_logs norm
      3. Normalized name direct match

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

    # Load alias table once (alias_name → canonical_name norm)
    alias_map: dict[str, str] = {}  # normalized alias → normalized canonical
    try:
        aliases = con.execute(
            "SELECT alias_name, canonical_name FROM nba_player_alias"
        ).fetchall()
        for a in aliases:
            anorm = _normalize_player_name(a["alias_name"])
            cnorm = _normalize_player_name(a["canonical_name"])
            if anorm and cnorm:
                alias_map[anorm] = cnorm
    except Exception:
        pass  # alias table may not exist yet on first run

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
            SELECT player_id, player_name, pts, reb, ast, fg3m, stl, blk
            FROM nba_player_game_logs
            WHERE game_date = ?
        """, (gdate,)).fetchall()

        if not logs:
            continue

        # Build lookups: norm_name → stats, bdl_player_id → stats
        log_map_norm: dict = {}
        log_map_id: dict = {}
        for row in logs:
            stats = {
                "pts": row["pts"], "reb": row["reb"], "ast": row["ast"],
                "fg3m": row["fg3m"], "stl": row["stl"], "blk": row["blk"],
            }
            norm = _normalize_player_name(row["player_name"])
            if norm:
                log_map_norm[norm] = stats
            if row["player_id"]:
                log_map_id[row["player_id"]] = stats

        # Get ungraded preds for this date
        preds = con.execute("""
            SELECT id, player_name, player_name_norm, stat_type, line, side,
                   bdl_player_id
            FROM nba_forward_preds
            WHERE game_date = ? AND actual_hit IS NULL
        """, (gdate,)).fetchall()

        now_str = datetime.now(timezone.utc).isoformat()

        for pred in preds:
            stat_type = pred["stat_type"]
            line = pred["line"]
            side = pred["side"]

            # Resolution: bdl_player_id → alias → normalized name
            plog = None
            bdl_pid = pred["bdl_player_id"]
            if bdl_pid and bdl_pid in log_map_id:
                plog = log_map_id[bdl_pid]
            else:
                norm = pred["player_name_norm"] or _normalize_player_name(pred["player_name"])
                # Try alias resolution
                resolved = alias_map.get(norm, norm)
                if resolved in log_map_norm:
                    plog = log_map_norm[resolved]
                elif norm != resolved and norm in log_map_norm:
                    plog = log_map_norm[norm]

            if plog is None:
                not_found += 1
                continue

            cols = _STAT_COL_MAP.get(stat_type)
            if cols is None:
                continue

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
                    logger.error("[scout.runner] S5-2 forward preds error: %s\n%s",
                                 fp_exc, traceback.format_exc())
            except Exception as exc:
                err = f"NBA game {game.game_id}: {exc}"
                errors.append(err)
                logger.error("[scout.runner] %s\n%s", err, traceback.format_exc())
                try:
                    db.rollback()
                except Exception:
                    pass
        total_props    += nba_count
        by_sport["NBA"] = nba_count
        print(f"[scout.runner] NBA: {len(nba_games)} games → {nba_count} props")

        # S5-2 outcome check: count actual rows in nba_forward_preds for today
        _nfp_today = 0
        _nfp_pretip = 0
        if nba_games:
            try:
                import sqlite3 as _sq
                _hp = os.path.normpath(os.path.join(
                    os.path.dirname(__file__), "..", "..", "data", "historical.db"))
                _hc = _sq.connect(_hp, timeout=5)
                _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                _nfp_today = _hc.execute(
                    "SELECT COUNT(*) FROM nba_forward_preds "
                    "WHERE game_date=? AND snapshot_type='scout'",
                    (_today,)).fetchone()[0]
                _nfp_pretip = _hc.execute(
                    "SELECT COUNT(*) FROM nba_forward_preds "
                    "WHERE game_date=? AND snapshot_type='pre_tip'",
                    (_today,)).fetchone()[0]
                _hc.close()
            except Exception:
                pass
            print(f"[scout.runner] S5-2: {_nfp_today} scout + {_nfp_pretip} "
                  f"pre_tip forward preds for {len(nba_games)} games")
        # Dedicated pipeline_health entry: rows = actual forward preds written
        try:
            from pipeline_health import record_run as _rec_nfp
            _rec_nfp("nba_forward_preds",
                     rows_written=_nfp_today,
                     errors=0 if _nfp_today > 0 else 1,
                     error_details=(f"{len(nba_games)} games but 0 forward preds"
                                    if nba_games and _nfp_today == 0 else None),
                     games_exist=len(nba_games) > 0)
        except Exception:
            pass
        # Record NBA scout health
        try:
            from pipeline_health import record_run
            _nba_errs = [e for e in errors if "NBA" in e]
            record_run("nba_scout",
                       rows_written=nba_count,
                       errors=len(_nba_errs),
                       error_details="; ".join(_nba_errs[:3]) if _nba_errs else None,
                       games_exist=len(nba_games) > 0)
        except Exception:
            pass
    except Exception as exc:
        errors.append(f"NBA fetch: {exc}")
        logger.error("[scout.runner] NBA fetch error: %s\n%s", exc, traceback.format_exc())
        try:
            from pipeline_health import record_run
            record_run("nba_scout", errors=1,
                       error_details=f"NBA fetch: {exc}",
                       games_exist=None)
        except Exception:
            pass

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
