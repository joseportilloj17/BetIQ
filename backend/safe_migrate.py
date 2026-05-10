"""
safe_migrate.py — Schema migration safety wrapper for BetIQ.

Provides:
  • initialize(engine)        — create migration_history table, called once at startup
  • safe_migrate(engine, sql, migration_id, description="")
                              — idempotent migration with pre-run backup + audit trail
  • safe_add_column(engine, table, column, definition)
                              — idempotent ALTER TABLE ADD COLUMN wrapper

Backup policy:
  • One .db backup per calendar day (UTC) to /tmp/betiq_backups/
  • Backup is written BEFORE the first migration that day runs
  • Files older than 30 days are pruned automatically
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

BACKUP_DIR = "/tmp/betiq_backups"
_BACKUP_DONE_TODAY: set[str] = set()   # in-process guard (avoid repeat stat calls)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _db_path_from_engine(engine) -> str:
    """Extract the filesystem path from a SQLAlchemy engine URL."""
    url_str = str(engine.url)
    # sqlite:////absolute/path/to/bets.db  or  sqlite:///relative
    if url_str.startswith("sqlite:////"):
        return url_str[len("sqlite:///"):]
    elif url_str.startswith("sqlite:///"):
        return os.path.abspath(url_str[len("sqlite:///"):])
    raise ValueError(f"safe_migrate: unsupported engine URL: {url_str}")


def _ensure_backup_dir():
    os.makedirs(BACKUP_DIR, exist_ok=True)


def _prune_old_backups(keep_days: int = 30):
    """Delete backup files older than keep_days."""
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400
        for fname in os.listdir(BACKUP_DIR):
            fpath = os.path.join(BACKUP_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                log.info(f"[safe_migrate] Pruned old backup: {fname}")
    except Exception as exc:
        log.warning(f"[safe_migrate] Prune failed (non-fatal): {exc}")


def _backup_if_needed(db_path: str):
    """Create a dated backup of the DB if we haven't done so today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today in _BACKUP_DONE_TODAY:
        return
    _ensure_backup_dir()
    backup_name = f"{today}_{datetime.now(timezone.utc).strftime('%H-%M')}_bets.db"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    if not os.path.exists(db_path):
        log.warning(f"[safe_migrate] DB not found at {db_path}, skipping backup")
        _BACKUP_DONE_TODAY.add(today)
        return
    try:
        shutil.copy2(db_path, backup_path)
        log.info(f"[safe_migrate] Backup written: {backup_path}")
        _BACKUP_DONE_TODAY.add(today)
        _prune_old_backups()
    except Exception as exc:
        log.error(f"[safe_migrate] Backup FAILED: {exc}")
        # Do NOT set the flag — allow retry on next call


def _raw_conn(db_path: str) -> sqlite3.Connection:
    """Open a bare sqlite3 connection (bypasses SQLAlchemy engine pool)."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


def _ensure_migration_history(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS migration_history (
            migration_id   TEXT PRIMARY KEY,
            description    TEXT,
            applied_at     TEXT NOT NULL,
            sql_executed   TEXT
        )
    """)
    conn.commit()


def _already_applied(conn: sqlite3.Connection, migration_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM migration_history WHERE migration_id = ?", (migration_id,)
    ).fetchone()
    return row is not None


def _record_migration(conn: sqlite3.Connection, migration_id: str, description: str, sql: str):
    conn.execute(
        """
        INSERT OR REPLACE INTO migration_history (migration_id, description, applied_at, sql_executed)
        VALUES (?, ?, ?, ?)
        """,
        (migration_id, description, datetime.now(timezone.utc).isoformat(), sql),
    )
    conn.commit()


# ─── Public API ────────────────────────────────────────────────────────────────

def initialize(engine) -> None:
    """
    Create the migration_history table if it doesn't exist.
    Call once at app startup (in init_db).
    """
    db_path = _db_path_from_engine(engine)
    conn = _raw_conn(db_path)
    try:
        _ensure_migration_history(conn)
    finally:
        conn.close()
    log.info("[safe_migrate] migration_history table ready")


def safe_migrate(engine, sql: str, migration_id: str, description: str = "") -> bool:
    """
    Run sql exactly once, guarded by migration_history.

    Returns True if the migration ran, False if already applied.
    Raises on any unexpected error.
    """
    db_path = _db_path_from_engine(engine)
    _backup_if_needed(db_path)
    conn = _raw_conn(db_path)
    try:
        _ensure_migration_history(conn)
        if _already_applied(conn, migration_id):
            return False
        conn.execute(sql)
        conn.commit()
        _record_migration(conn, migration_id, description, sql)
        log.info(f"[safe_migrate] Applied: {migration_id}")
        return True
    finally:
        conn.close()


def safe_add_column(engine, table: str, column: str, definition: str) -> bool:
    """
    Idempotent ADD COLUMN wrapper.

    Checks PRAGMA table_info to see if the column already exists before attempting
    the ALTER TABLE.  Also records in migration_history for audit.

    Example:
        safe_add_column(engine, "user_pick_legs", "point", "REAL DEFAULT NULL")
    """
    migration_id = f"add_column__{table}__{column}"
    db_path = _db_path_from_engine(engine)
    _backup_if_needed(db_path)
    conn = _raw_conn(db_path)
    try:
        _ensure_migration_history(conn)
        if _already_applied(conn, migration_id):
            return False
        # Check actual schema
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if column in cols:
            # Column exists but wasn't tracked — record it so future calls skip fast
            _record_migration(
                conn, migration_id,
                f"column {column} already existed in {table} at registration time",
                f"-- skipped (column exists): ALTER TABLE {table} ADD COLUMN {column} {definition}",
            )
            return False
        sql = f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        conn.execute(sql)
        conn.commit()
        _record_migration(conn, migration_id, f"Add {column} to {table}", sql)
        log.info(f"[safe_migrate] {sql}")
        return True
    finally:
        conn.close()


def migration_history(engine) -> list[dict]:
    """Return all recorded migrations as a list of dicts (for health endpoint)."""
    db_path = _db_path_from_engine(engine)
    conn = _raw_conn(db_path)
    try:
        _ensure_migration_history(conn)
        rows = conn.execute(
            "SELECT migration_id, description, applied_at FROM migration_history ORDER BY applied_at DESC"
        ).fetchall()
        return [{"migration_id": r[0], "description": r[1], "applied_at": r[2]} for r in rows]
    finally:
        conn.close()


def last_backup_info() -> dict:
    """Return info about the most recent backup file."""
    try:
        files = [
            f for f in os.listdir(BACKUP_DIR)
            if f.endswith(".db") and os.path.isfile(os.path.join(BACKUP_DIR, f))
        ]
        if not files:
            return {"backup_dir": BACKUP_DIR, "last_backup": None, "backup_count": 0}
        files.sort()
        latest = files[-1]
        stat = os.stat(os.path.join(BACKUP_DIR, latest))
        return {
            "backup_dir": BACKUP_DIR,
            "last_backup": latest,
            "last_backup_size_kb": round(stat.st_size / 1024, 1),
            "backup_count": len(files),
        }
    except Exception:
        return {"backup_dir": BACKUP_DIR, "last_backup": None, "backup_count": 0}
