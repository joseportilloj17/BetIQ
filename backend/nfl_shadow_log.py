"""
nfl_shadow_log.py — Manual WR-CB shadow coverage log.

Tracks projected and confirmed shadow matchups (CB travels with WR1) from
free articles. This is the only in-season source for WR-CB assignment data
without paid subscriptions.

Usage:
    python nfl_shadow_log.py add --week 6 --def DAL --cb "Trevon Diggs" \
        --off PHI --wr "A.J. Brown" --status projected --source PFF \
        --url "https://..." --scope outside_only
    python nfl_shadow_log.py list --week 6
    python nfl_shadow_log.py confirm --week 5 --def DAL --cb "Trevon Diggs" \
        --result confirmed   # or confirmed_no_shadow
    python nfl_shadow_log.py backfill   # mark all as backfilled
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_HERE     = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_MANUAL_DIR = os.path.join(_DATA_DIR, "manual")
_CSV_PATH = os.path.join(_MANUAL_DIR, "shadow_log.csv")
_HIST_DB  = os.path.join(_DATA_DIR, "historical.db")

_VALID_STATUSES = {"projected", "confirmed", "confirmed_no_shadow"}
_VALID_SCOPES = {"outside_only", "full_travel", "partial", "unknown"}
_VALID_SOURCES = {"PFF", "RotoBaller", "ESPN", "FTN", "FantasyPros", "Twitter",
                  "Reddit", "manual", "other"}

# ── NFL team abbreviations ──
_NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

# Cache for roster lookups
_roster_cache: dict | None = None


def _load_roster(season: int = 2026) -> dict:
    """Load nflverse roster and build name→gsis_id lookup by (team, position_group)."""
    global _roster_cache
    if _roster_cache is not None:
        return _roster_cache

    try:
        import nfl_data_py as nfl
        r = nfl.import_weekly_rosters([season])
        # Deduplicate to latest week per player
        r = r.sort_values("week", ascending=False).drop_duplicates("player_id")

        lookup = {}
        for _, row in r.iterrows():
            name = row["player_name"]
            pid = row["player_id"]
            team = row["team"]
            pos = row.get("depth_chart_position", row.get("position", ""))
            if name and pid:
                # Store by (name, team) and by (name_lower, team)
                lookup[(name, team)] = pid
                lookup[(name.lower(), team)] = pid
                # Also by name only (less specific, used as fallback)
                lookup[name] = pid
                lookup[name.lower()] = pid
        _roster_cache = lookup
        return lookup
    except Exception as exc:
        logger.warning("Failed to load roster: %s", exc)
        return {}


_NAME_ALIASES = {
    "Deebo Samuel": "Deebo Samuel Sr.",
    "Michael Pittman": "Michael Pittman Jr.",
    "Odell Beckham": "Odell Beckham Jr.",
    "Marvin Harrison": "Marvin Harrison Jr.",
    "Pat Surtain": "Pat Surtain II",
    "Patrick Surtain": "Pat Surtain II",
}


def _resolve_gsis_id(name: str, team: str, season: int = 2026) -> str:
    """Resolve a player name + team to a gsis_id. Returns '' if unmatched."""
    roster = _load_roster(season)
    # Try (name, team) first, then name-only fallback
    gsis = roster.get((name, team)) or roster.get((name.lower(), team))
    if not gsis:
        gsis = roster.get(name) or roster.get(name.lower())
    # Try alias if still unmatched
    if not gsis and name in _NAME_ALIASES:
        alias = _NAME_ALIASES[name]
        gsis = roster.get((alias, team)) or roster.get((alias.lower(), team))
        if not gsis:
            gsis = roster.get(alias) or roster.get(alias.lower())
    if not gsis:
        logger.warning("UNMATCHED: %s (%s) — no gsis_id found in roster", name, team)
        return ""
    return gsis


def _normalize_team(team: str) -> str:
    """Normalize team abbreviation."""
    team = team.upper().strip()
    if team == "LAR":
        team = "LA"
    if team not in _NFL_TEAMS:
        raise ValueError(f"Unknown team: {team}. Valid: {sorted(_NFL_TEAMS)}")
    return team


# ══════════════════════════════════════════════════════════════════════════════
# CSV operations
# ══════════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "season", "week", "game_id", "def_team", "cb_name", "cb_gsis_id",
    "off_team", "wr_name", "wr_gsis_id", "status", "shadow_scope",
    "source", "source_url", "entered_at", "notes",
]


def add_entry(season: int, week: int, def_team: str, cb_name: str,
              off_team: str, wr_name: str,
              status: str = "projected",
              shadow_scope: str = "unknown",
              source: str = "manual",
              source_url: str = "",
              notes: str = "",
              game_id: str = "") -> dict:
    """Add a shadow log entry to CSV and DB."""
    # Validate
    def_team = _normalize_team(def_team)
    off_team = _normalize_team(off_team)

    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Valid: {_VALID_STATUSES}")
    if shadow_scope not in _VALID_SCOPES:
        raise ValueError(f"Invalid scope: {shadow_scope}. Valid: {_VALID_SCOPES}")

    # Resolve gsis IDs
    cb_gsis = _resolve_gsis_id(cb_name, def_team, season)
    wr_gsis = _resolve_gsis_id(wr_name, off_team, season)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = {
        "season": season,
        "week": week,
        "game_id": game_id or f"{season}_{week:02d}_{off_team}_{def_team}",
        "def_team": def_team,
        "cb_name": cb_name,
        "cb_gsis_id": cb_gsis,
        "off_team": off_team,
        "wr_name": wr_name,
        "wr_gsis_id": wr_gsis,
        "status": status,
        "shadow_scope": shadow_scope,
        "source": source,
        "source_url": source_url,
        "entered_at": now,
        "notes": notes,
    }

    # Append to CSV
    os.makedirs(_MANUAL_DIR, exist_ok=True)
    file_exists = os.path.exists(_CSV_PATH) and os.path.getsize(_CSV_PATH) > 0
    with open(_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

    # Also store in DB
    _store_to_db([entry])

    # Report unmatched IDs
    warnings = []
    if not cb_gsis:
        warnings.append(f"  WARNING: CB '{cb_name}' ({def_team}) — no gsis_id found")
    if not wr_gsis:
        warnings.append(f"  WARNING: WR '{wr_name}' ({off_team}) — no gsis_id found")

    print(f"Added: {cb_name} ({def_team}) shadows {wr_name} ({off_team}) "
          f"— Week {week}, {status}, {shadow_scope}")
    for w in warnings:
        print(w)

    return entry


def list_entries(season: int = 2026, week: int | None = None) -> list[dict]:
    """List shadow log entries from CSV."""
    if not os.path.exists(_CSV_PATH):
        print("No shadow log found.")
        return []

    entries = []
    with open(_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("season", 0)) != season:
                continue
            if week is not None and int(row.get("week", 0)) != week:
                continue
            entries.append(row)

    if not entries:
        print(f"No entries for {season}" + (f" week {week}" if week else ""))
        return []

    print(f"\n{'='*80}")
    print(f"Shadow Log — {season}" + (f" Week {week}" if week else " (all weeks)"))
    print(f"{'='*80}")
    print(f"  {'Wk':<3} {'DEF':<4} {'CB':<22} {'OFF':<4} {'WR':<22} {'Status':<20} {'Scope':<12} {'Src':<8}")
    print(f"  {'-'*95}")
    for e in entries:
        print(f"  {e['week']:<3} {e['def_team']:<4} {e['cb_name']:<22} "
              f"{e['off_team']:<4} {e['wr_name']:<22} "
              f"{e['status']:<20} {e['shadow_scope']:<12} {e['source']:<8}")

    return entries


def confirm_entry(season: int, week: int, def_team: str, cb_name: str,
                  result: str = "confirmed") -> bool:
    """Update a projected entry to confirmed or confirmed_no_shadow."""
    if result not in ("confirmed", "confirmed_no_shadow"):
        raise ValueError(f"Result must be 'confirmed' or 'confirmed_no_shadow'")

    def_team = _normalize_team(def_team)

    # Read all entries, find and update the matching one
    if not os.path.exists(_CSV_PATH):
        print("No shadow log found.")
        return False

    entries = []
    found = False
    with open(_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row.get("season", 0)) == season
                    and int(row.get("week", 0)) == week
                    and row.get("def_team") == def_team
                    and row.get("cb_name", "").lower() == cb_name.lower()
                    and row.get("status") == "projected"):
                row["status"] = result
                row["entered_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                found = True
                print(f"Updated: {cb_name} ({def_team}) Week {week} → {result}")
            entries.append(row)

    if not found:
        print(f"No projected entry found for {cb_name} ({def_team}) Week {week}")
        return False

    # Rewrite CSV
    with open(_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(entries)

    # Update DB
    _store_to_db(entries, replace=True)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Database storage
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_shadow_table():
    """Create the shadow_coverage_log table if it doesn't exist."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_coverage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT,
            def_team TEXT NOT NULL,
            cb_name TEXT NOT NULL,
            cb_gsis_id TEXT,
            off_team TEXT NOT NULL,
            wr_name TEXT NOT NULL,
            wr_gsis_id TEXT,
            status TEXT NOT NULL,
            shadow_scope TEXT,
            source TEXT,
            source_url TEXT,
            entered_at TEXT NOT NULL,
            notes TEXT,
            UNIQUE(season, week, def_team, cb_name, off_team, wr_name)
        )
    """)
    conn.commit()
    conn.close()


def _store_to_db(entries: list[dict], replace: bool = False):
    """Store entries into the shadow_coverage_log table."""
    _ensure_shadow_table()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    for e in entries:
        try:
            conn.execute(f"""
                {verb} INTO shadow_coverage_log
                (season, week, game_id, def_team, cb_name, cb_gsis_id,
                 off_team, wr_name, wr_gsis_id, status, shadow_scope,
                 source, source_url, entered_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(e["season"]), int(e["week"]), e.get("game_id", ""),
                e["def_team"], e["cb_name"], e.get("cb_gsis_id", ""),
                e["off_team"], e["wr_name"], e.get("wr_gsis_id", ""),
                e["status"], e.get("shadow_scope", "unknown"),
                e.get("source", "manual"), e.get("source_url", ""),
                e["entered_at"], e.get("notes", ""),
            ))
        except Exception as exc:
            logger.warning("Failed to store shadow entry: %s", exc)
    conn.commit()
    conn.close()


def ingest_csv_to_db():
    """Load all CSV entries into the DB table (for re-sync)."""
    if not os.path.exists(_CSV_PATH):
        print("No shadow log CSV found.")
        return

    entries = []
    with open(_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    if entries:
        _store_to_db(entries, replace=True)
        print(f"Ingested {len(entries)} entries into shadow_coverage_log")
    else:
        print("CSV is empty.")


def get_shadow_status(season: int, week: int, off_team: str,
                      wr_name: str) -> dict | None:
    """
    Query whether a WR has a shadow projected/confirmed this week.

    Returns dict with cb_name, status, shadow_scope, or None.
    """
    _ensure_shadow_table()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT cb_name, def_team, status, shadow_scope, source
        FROM shadow_coverage_log
        WHERE season = ? AND week = ? AND off_team = ? AND wr_name = ?
        ORDER BY CASE status
            WHEN 'confirmed' THEN 1
            WHEN 'projected' THEN 2
            WHEN 'confirmed_no_shadow' THEN 3
        END
        LIMIT 1
    """, (season, week, off_team, wr_name)).fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="NFL Shadow Coverage Log")
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add a shadow entry")
    p_add.add_argument("--season", type=int, default=2026)
    p_add.add_argument("--week", type=int, required=True)
    p_add.add_argument("--def", dest="def_team", required=True, help="Defensive team")
    p_add.add_argument("--cb", required=True, help="CB name")
    p_add.add_argument("--off", dest="off_team", required=True, help="Offensive team")
    p_add.add_argument("--wr", required=True, help="WR name")
    p_add.add_argument("--status", default="projected",
                        choices=["projected", "confirmed", "confirmed_no_shadow"])
    p_add.add_argument("--scope", default="unknown",
                        choices=["outside_only", "full_travel", "partial", "unknown"])
    p_add.add_argument("--source", default="manual")
    p_add.add_argument("--url", default="", help="Source URL")
    p_add.add_argument("--notes", default="")

    # list
    p_list = sub.add_parser("list", help="List shadow entries")
    p_list.add_argument("--season", type=int, default=2026)
    p_list.add_argument("--week", type=int, default=None)

    # confirm
    p_conf = sub.add_parser("confirm", help="Confirm or deny a projected shadow")
    p_conf.add_argument("--season", type=int, default=2026)
    p_conf.add_argument("--week", type=int, required=True)
    p_conf.add_argument("--def", dest="def_team", required=True)
    p_conf.add_argument("--cb", required=True)
    p_conf.add_argument("--result", default="confirmed",
                         choices=["confirmed", "confirmed_no_shadow"])

    # ingest
    p_ing = sub.add_parser("ingest", help="Re-sync CSV to DB")

    args = parser.parse_args()

    if args.command == "add":
        add_entry(
            season=args.season, week=args.week,
            def_team=args.def_team, cb_name=args.cb,
            off_team=args.off_team, wr_name=args.wr,
            status=args.status, shadow_scope=args.scope,
            source=args.source, source_url=args.url,
            notes=args.notes,
        )
    elif args.command == "list":
        list_entries(args.season, args.week)
    elif args.command == "confirm":
        confirm_entry(args.season, args.week, args.def_team, args.cb, args.result)
    elif args.command == "ingest":
        ingest_csv_to_db()
    else:
        parser.print_help()
