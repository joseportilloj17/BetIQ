"""
soccer_backfill.py — Phase 1b+1c: Build soccer feature tables.

Creates three new tables in bets.db:
  player_soccer_stats   — per-player match stats (API-Football, premium required for historical)
  fixture_soccer_stats  — per-team match stats (shots, corners, possession)
  team_soccer_form      — rolling form per team as of each bet date (computed from soccer_results)

Usage:
    cd /Users/joseportillo/Downloads/BetIQ

    # Phase 1c: Team form from existing soccer_results (no API calls needed)
    python backend/soccer_backfill.py --team-form

    # Phase 1b: Player/fixture stats from API-Football (requires premium for historical dates)
    python backend/soccer_backfill.py --fixture-stats --date 2026-04-25

    # Full backfill (team form only — player stats require API-Football premium)
    python backend/soccer_backfill.py --team-form --dry-run
    python backend/soccer_backfill.py --team-form

    # Check what was built
    python backend/soccer_backfill.py --status
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import sqlite3
import time
import unicodedata
from datetime import datetime, date as _date, timedelta
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")

# ── API-Football config ───────────────────────────────────────────────────────
try:
    from config import API_FOOTBALL_KEY, API_FOOTBALL_BASE, LEAGUE_IDS
    from config import FOOTBALL_DATA_KEY, FOOTBALL_DATA_BASE
except ImportError:
    API_FOOTBALL_KEY   = os.environ.get("API_FOOTBALL_KEY", "")
    API_FOOTBALL_BASE  = "https://v3.football.api-sports.io"
    FOOTBALL_DATA_KEY  = os.environ.get("FOOTBALL_DATA_KEY", "")
    FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
    LEAGUE_IDS         = {}

_AF_HEADERS = {
    "x-rapidapi-key":  API_FOOTBALL_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io",
}
_FD_HEADERS = {"X-Auth-Token": FOOTBALL_DATA_KEY}

# football-data.org competition codes mapped to league names
_FD_COMP_IDS = {
    "EPL": "PL", "La Liga": "PD", "Bundesliga": "BL1",
    "Ligue1": "FL1", "Serie A": "SA", "UCL": "CL", "Europa League": "EL",
}

# Normalize league names from bet_legs to match FD comp codes
_LEAGUE_NORM = {
    "english premier league": "EPL",
    "la liga": "La Liga", "ligue1": "Ligue1", "ligue 1": "Ligue1",
    "serie a": "Serie A", "bundesliga": "Bundesliga",
    "uefa champions league": "UCL", "europa league": "Europa League",
}

# ─────────────────────────────────────────────────────────────────────────────
# Schema creation
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS player_soccer_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL,
    player_name     TEXT    NOT NULL,
    team_name       TEXT,
    shots_total     INTEGER,
    shots_on_target INTEGER,
    goals           INTEGER,
    assists         INTEGER,
    minutes_played  INTEGER,
    game_date       TEXT,
    league_id       INTEGER,
    fetched_at      TEXT,
    UNIQUE(fixture_id, player_name)
);

CREATE TABLE IF NOT EXISTS fixture_soccer_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL,
    team_name       TEXT    NOT NULL,
    is_home         INTEGER,
    shots_on_target INTEGER,
    shots_total     INTEGER,
    corners         INTEGER,
    possession      REAL,
    game_date       TEXT,
    league_id       INTEGER,
    fetched_at      TEXT,
    UNIQUE(fixture_id, team_name)
);

CREATE TABLE IF NOT EXISTS team_soccer_form (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name        TEXT    NOT NULL,
    as_of_date       TEXT    NOT NULL,
    league           TEXT,
    form_5           TEXT,
    form_10          TEXT,
    wins_5           INTEGER,
    draws_5          INTEGER,
    losses_5         INTEGER,
    goals_scored_5   REAL,
    goals_conceded_5 REAL,
    goals_scored_10  REAL,
    goals_conceded_10 REAL,
    home_form_5      TEXT,
    away_form_5      TEXT,
    over25_rate_10   REAL,
    games_found      INTEGER,
    computed_at      TEXT,
    UNIQUE(team_name, as_of_date)
);
"""

def _ensure_tables(con: sqlite3.Connection) -> None:
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Team name normalization (mirrors soccer_data._normalize_team)
# ─────────────────────────────────────────────────────────────────────────────

_COMMON_SUFFIXES = re.compile(
    r"\b(fc|sc|cf|ac|as|rc|ud|fk|sk|bk|sporting|club|de|del|la|el)\b",
    re.IGNORECASE,
)
_STROKE_TRANS = str.maketrans({
    'ø': 'o', 'Ø': 'o', 'ł': 'l', 'Ł': 'l', 'ð': 'd', 'Ð': 'd',
    'þ': 'th', 'æ': 'ae', 'Æ': 'ae', 'œ': 'oe', 'Œ': 'oe', 'ß': 'ss',
})
_ALIASES: dict[str, str] = {
    "man city": "manchester city", "man utd": "manchester united",
    "man united": "manchester united", "inter": "internazionale",
    "inter milan": "internazionale milan", "lyon": "olympique lyonnais",
    "paris st-g": "paris saint-germain", "paris sg": "paris saint-germain",
    "psg": "paris saint-germain", "sporting lisbon": "sporting clube portugal",
    "sporting cp": "sporting clube portugal", "benfica": "sl benfica",
    "fc copenhagen": "fc kobenhavn", "copenhagen": "kobenhavn",
    "red star": "crvena zvezda", "red star belgrade": "crvena zvezda",
    "oviedo": "real oviedo", "dortmund": "borussia dortmund",
    "bvb": "borussia dortmund", "leverkusen": "bayer leverkusen",
    "gladbach": "borussia monchengladbach", "m'gladbach": "borussia monchengladbach",
    "atletico": "atletico madrid", "real": "real madrid",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
    "newcastle": "newcastle united", "wolves": "wolverhampton wanderers",
    "villa": "aston villa", "brentford": "brentford",
    "leeds": "leeds united", "sheff utd": "sheffield united",
    "sheffield utd": "sheffield united", "west ham": "west ham united",
    "nottm forest": "nottingham forest", "coventry": "coventry city",
    "brighton": "brighton hove albion", "luton": "luton town",
    "burnley": "burnley",
}


def _normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.translate(_STROKE_TRANS)
    s = s.lower().strip()
    s = _ALIASES.get(s, s)
    s = _COMMON_SUFFIXES.sub("", s)
    s = s.replace("/", " ").replace("munchen", "munich")
    return re.sub(r"\s+", " ", s).strip()


def _team_match(needle: str, candidate: str) -> bool:
    n, c = _normalize(needle), _normalize(candidate)
    if n == c:
        return True
    nw, cw = set(n.split()), set(c.split())
    if not nw or not cw:
        return False
    shorter, longer = (nw, cw) if len(nw) <= len(cw) else (cw, nw)
    if len(shorter) == 1:
        return next(iter(shorter)) in longer
    return shorter.issubset(longer)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1c: Team form computation from soccer_results
# ─────────────────────────────────────────────────────────────────────────────

def _load_soccer_results(con: sqlite3.Connection) -> list[dict]:
    """Load all match results from soccer_results table."""
    cur = con.execute(
        "SELECT date, home_team, away_team, home_goals, away_goals, league_name "
        "FROM soccer_results WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL "
        "ORDER BY date"
    )
    rows = []
    for date, ht, at, hg, ag, league in cur.fetchall():
        rows.append({
            "date": date, "home_team": ht, "away_team": at,
            "home_goals": hg, "away_goals": ag, "league": league,
        })
    return rows


def _get_team_results_before(
    all_results: list[dict],
    team_name: str,
    before_date: str,
    max_games: int = 10,
) -> list[dict]:
    """
    Find the last N results for a team before a given date.
    Returns list of dicts: {date, is_home, goals_for, goals_against, result}
    Sorted newest first.
    """
    matches = []
    for r in all_results:
        if r["date"] >= before_date:
            continue
        is_home = _team_match(team_name, r["home_team"])
        is_away = _team_match(team_name, r["away_team"])
        if not is_home and not is_away:
            continue

        if is_home:
            gf, ga = r["home_goals"], r["away_goals"]
        else:
            gf, ga = r["away_goals"], r["home_goals"]

        if gf > ga:
            result = "W"
        elif gf == ga:
            result = "D"
        else:
            result = "L"

        matches.append({
            "date": r["date"],
            "is_home": is_home,
            "goals_for": gf,
            "goals_against": ga,
            "total_goals": gf + ga,
            "result": result,
        })

    # Sort newest first, take last N
    matches.sort(key=lambda x: x["date"], reverse=True)
    return matches[:max_games]


def _compute_form(matches: list[dict], n: int) -> dict:
    """Compute form stats from last N matches (newest first)."""
    subset = matches[:n]
    if not subset:
        return {}

    form_str = "".join(m["result"] for m in subset)
    wins   = form_str.count("W")
    draws  = form_str.count("D")
    losses = form_str.count("L")
    gf_avg = sum(m["goals_for"]  for m in subset) / len(subset)
    ga_avg = sum(m["goals_against"] for m in subset) / len(subset)

    home_matches = [m for m in subset if m["is_home"]]
    away_matches = [m for m in subset if not m["is_home"]]
    home_form = "".join(m["result"] for m in home_matches) or None
    away_form = "".join(m["result"] for m in away_matches) or None

    return {
        "form": form_str,
        "wins": wins, "draws": draws, "losses": losses,
        "gf_avg": round(gf_avg, 2),
        "ga_avg": round(ga_avg, 2),
        "home_form": home_form,
        "away_form": away_form,
    }


def build_team_form(dry_run: bool = False) -> None:
    """
    Phase 1c: Compute team_soccer_form for every unique (team, bet_date) pair.

    Uses soccer_results table — no API calls required.
    """
    con = sqlite3.connect(DB_PATH)
    _ensure_tables(con)

    all_results = _load_soccer_results(con)
    print(f"[form] Loaded {len(all_results)} match results from soccer_results")

    # Get unique (team, bet_date, league) combinations from soccer bet_legs
    cur = con.cursor()
    cur.execute("""
        SELECT DISTINCT
            CASE
                WHEN bl.market_type IN ('Moneyline') THEN bl.team
                WHEN bl.market_type = 'Total'        THEN bl.team
                ELSE NULL
            END as team_name,
            DATE(b.time_placed) as bet_date,
            bl.league
        FROM bet_legs bl
        JOIN bets b ON bl.bet_id = b.id
        WHERE bl.sport='Soccer'
          AND bl.team IS NOT NULL
          AND bl.market_type IN ('Moneyline', 'Total')
        ORDER BY bet_date
    """)
    moneyline_pairs = cur.fetchall()

    # Also get away teams (opponent field for Moneyline legs)
    cur.execute("""
        SELECT DISTINCT bl.opponent, DATE(b.time_placed), bl.league
        FROM bet_legs bl
        JOIN bets b ON bl.bet_id = b.id
        WHERE bl.sport='Soccer'
          AND bl.opponent IS NOT NULL
          AND bl.market_type IN ('Moneyline', 'Total')
        ORDER BY 2
    """)
    away_pairs = cur.fetchall()

    all_pairs = list({(t, d, lg) for t, d, lg in moneyline_pairs + away_pairs if t})
    all_pairs.sort(key=lambda x: (x[0], x[1]))
    print(f"[form] Computing form for {len(all_pairs)} unique (team, date) pairs")

    inserted = updated = skipped_no_data = 0
    for team, bet_date, league in all_pairs:
        if not team or not bet_date:
            continue

        matches = _get_team_results_before(all_results, team, bet_date, max_games=10)

        if not matches:
            skipped_no_data += 1
            if dry_run:
                print(f"  [SKIP] {team} @ {bet_date} — no matches found in soccer_results")
            continue

        f5 = _compute_form(matches, 5)
        f10 = _compute_form(matches, 10)

        # over25_rate_10: fraction of last 10 games with 2.5+ total goals
        subset10 = matches[:10]
        over25 = sum(1 for m in subset10 if m["total_goals"] > 2.5)
        over25_rate = round(over25 / len(subset10), 3) if subset10 else None

        row = {
            "team_name": team, "as_of_date": bet_date, "league": league,
            "form_5":  f5.get("form"), "form_10": f10.get("form"),
            "wins_5":  f5.get("wins"), "draws_5": f5.get("draws"), "losses_5": f5.get("losses"),
            "goals_scored_5":   f5.get("gf_avg"),  "goals_conceded_5":  f5.get("ga_avg"),
            "goals_scored_10":  f10.get("gf_avg"), "goals_conceded_10": f10.get("ga_avg"),
            "home_form_5": f5.get("home_form"),    "away_form_5": f5.get("away_form"),
            "over25_rate_10": over25_rate, "games_found": len(matches),
            "computed_at": datetime.utcnow().isoformat(),
        }

        if dry_run:
            print(f"  [DRY] {team:25s} @ {bet_date}  "
                  f"form5={row['form_5']:5s}  gf={row['goals_scored_5']}  "
                  f"ga={row['goals_conceded_5']}  over25={over25_rate}")
            inserted += 1
            continue

        # Upsert
        existing = con.execute(
            "SELECT id FROM team_soccer_form WHERE team_name=? AND as_of_date=?",
            (team, bet_date),
        ).fetchone()

        if existing:
            con.execute("""
                UPDATE team_soccer_form
                SET league=?, form_5=?, form_10=?, wins_5=?, draws_5=?, losses_5=?,
                    goals_scored_5=?, goals_conceded_5=?, goals_scored_10=?,
                    goals_conceded_10=?, home_form_5=?, away_form_5=?,
                    over25_rate_10=?, games_found=?, computed_at=?
                WHERE team_name=? AND as_of_date=?
            """, (
                row["league"], row["form_5"], row["form_10"],
                row["wins_5"], row["draws_5"], row["losses_5"],
                row["goals_scored_5"], row["goals_conceded_5"],
                row["goals_scored_10"], row["goals_conceded_10"],
                row["home_form_5"], row["away_form_5"],
                row["over25_rate_10"], row["games_found"], row["computed_at"],
                team, bet_date,
            ))
            updated += 1
        else:
            con.execute("""
                INSERT OR IGNORE INTO team_soccer_form
                  (team_name, as_of_date, league, form_5, form_10,
                   wins_5, draws_5, losses_5,
                   goals_scored_5, goals_conceded_5,
                   goals_scored_10, goals_conceded_10,
                   home_form_5, away_form_5,
                   over25_rate_10, games_found, computed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                team, bet_date, row["league"],
                row["form_5"], row["form_10"],
                row["wins_5"], row["draws_5"], row["losses_5"],
                row["goals_scored_5"], row["goals_conceded_5"],
                row["goals_scored_10"], row["goals_conceded_10"],
                row["home_form_5"], row["away_form_5"],
                row["over25_rate_10"], row["games_found"],
                row["computed_at"],
            ))
            inserted += 1

    if not dry_run:
        con.commit()

    con.close()
    status = "DRY RUN" if dry_run else "COMMITTED"
    print(f"\n[form] {status}: inserted={inserted}, updated={updated}, "
          f"no_data={skipped_no_data}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1b: Fixture stats from API-Football (recent dates only on free plan)
# ─────────────────────────────────────────────────────────────────────────────

_AF_LEAGUE_IDS = {v for k, v in LEAGUE_IDS.items() if "soccer" in k}

def _season_for_date(date_str: str) -> int:
    """European leagues start in August. Season = year season started."""
    year, month = int(date_str[:4]), int(date_str[5:7])
    return year if month >= 8 else year - 1


def fetch_fixture_stats(fixture_id: int, game_date: str, league_id: Optional[int],
                        dry_run: bool = False) -> int:
    """
    Fetch player + team stats for one fixture from API-Football.
    Returns number of rows written (0 on failure or dry run).
    """
    con = sqlite3.connect(DB_PATH)
    _ensure_tables(con)
    now = datetime.utcnow().isoformat()

    # ── Player stats ──────────────────────────────────────────────────────────
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures/players",
            headers=_AF_HEADERS,
            params={"fixture": fixture_id},
            timeout=15,
        )
        data = r.json()
    except Exception as exc:
        print(f"  [fixture_stats] player fetch error for {fixture_id}: {exc}")
        con.close()
        return 0

    errors = data.get("errors")
    if errors:
        plan_err = str(errors).lower()
        if "plan" in plan_err or "free" in plan_err:
            print(f"  [fixture_stats] API-Football free plan restriction for fixture {fixture_id}")
            print(f"    Error: {errors}")
            con.close()
            return 0

    player_rows = 0
    for team_block in data.get("response", []):
        team_name = team_block.get("team", {}).get("name", "")
        for player_data in team_block.get("players", []):
            p       = player_data.get("player", {})
            stats   = (player_data.get("statistics") or [{}])[0]
            shots   = stats.get("shots", {}) or {}
            goals   = stats.get("goals", {}) or {}
            games   = stats.get("games", {}) or {}

            row = (
                fixture_id,
                p.get("name", ""),
                team_name,
                shots.get("total"),
                shots.get("on"),
                goals.get("total"),
                goals.get("assists"),
                games.get("minutes"),
                game_date,
                league_id,
                now,
            )
            if dry_run:
                print(f"    [DRY player] {row[2]:20s} {row[1]:20s} "
                      f"shots={row[3]} sot={row[4]} goals={row[5]} ast={row[6]}")
            else:
                con.execute("""
                    INSERT OR REPLACE INTO player_soccer_stats
                      (fixture_id, player_name, team_name,
                       shots_total, shots_on_target, goals, assists,
                       minutes_played, game_date, league_id, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, row)
            player_rows += 1

    time.sleep(0.3)  # Respect rate limits

    # ── Team (fixture) stats ──────────────────────────────────────────────────
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures/statistics",
            headers=_AF_HEADERS,
            params={"fixture": fixture_id},
            timeout=15,
        )
        data = r.json()
    except Exception as exc:
        print(f"  [fixture_stats] team stats fetch error for {fixture_id}: {exc}")
        if not dry_run:
            con.commit()
        con.close()
        return player_rows

    fix_rows = 0
    for i, team_block in enumerate(data.get("response", [])):
        team_name = team_block.get("team", {}).get("name", "")
        is_home   = 1 if i == 0 else 0
        stat_map  = {s["type"]: s["value"] for s in team_block.get("statistics", [])}

        def _int(val) -> Optional[int]:
            try:
                return int(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        def _float(val) -> Optional[float]:
            try:
                if isinstance(val, str) and val.endswith("%"):
                    return float(val.rstrip("%"))
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        row = (
            fixture_id, team_name, is_home,
            _int(stat_map.get("Shots on Goal") or stat_map.get("Shots on Target")),
            _int(stat_map.get("Total Shots")),
            _int(stat_map.get("Corner Kicks")),
            _float(stat_map.get("Ball Possession")),
            game_date, league_id, now,
        )
        if dry_run:
            print(f"    [DRY fixture] is_home={is_home} {team_name:20s} "
                  f"sot={row[3]} shots={row[4]} corners={row[5]} poss={row[6]}")
        else:
            con.execute("""
                INSERT OR REPLACE INTO fixture_soccer_stats
                  (fixture_id, team_name, is_home,
                   shots_on_target, shots_total, corners, possession,
                   game_date, league_id, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, row)
        fix_rows += 1

    if not dry_run:
        con.commit()
    con.close()
    time.sleep(0.3)
    return player_rows + fix_rows


def fetch_all_fixture_stats(date_str: str, dry_run: bool = False) -> None:
    """
    Fetch fixture + player stats for all fixtures on a given date.
    Only works for dates within API-Football free plan window (recent 2 days).
    """
    con = sqlite3.connect(DB_PATH)
    _ensure_tables(con)
    cur = con.cursor()

    season = _season_for_date(date_str)
    total_written = 0

    # Get all league IDs with bets on this date
    cur.execute("""
        SELECT DISTINCT bl.league FROM bet_legs bl
        JOIN bets b ON bl.bet_id=b.id
        WHERE bl.sport='Soccer' AND DATE(b.time_placed)=?
    """, (date_str,))
    leagues_in_bets = {r[0] for r in cur.fetchall() if r[0]}
    con.close()

    print(f"\n[fixture_stats] Fetching fixtures for {date_str} (season={season})")
    print(f"[fixture_stats] Leagues in bets that day: {leagues_in_bets}")

    # Map league names to API-Football league IDs
    from config import LEAGUE_IDS as _LID
    league_id_map = {
        "english premier league": _LID.get("soccer_epl", 39),
        "epl": _LID.get("soccer_epl", 39),
        "la liga": _LID.get("soccer_spain_la_liga", 140),
        "bundesliga": _LID.get("soccer_germany_bundesliga", 78),
        "ligue1": _LID.get("soccer_france_ligue_one", 61),
        "ligue 1": _LID.get("soccer_france_ligue_one", 61),
        "serie a": _LID.get("soccer_italy_serie_a", 135),
        "uefa champions league": _LID.get("soccer_uefa_champs_league", 2),
        "ucl": _LID.get("soccer_uefa_champs_league", 2),
        "europa league": _LID.get("soccer_uefa_europa_league", 3),
    }
    target_league_ids = {
        league_id_map[lg.lower()] for lg in leagues_in_bets
        if lg and lg.lower() in league_id_map
    }
    if not target_league_ids:
        target_league_ids = set(_AF_LEAGUE_IDS)

    fetched_fixtures = set()
    calls_used = 0

    for league_id in sorted(target_league_ids):
        try:
            r = requests.get(
                f"{API_FOOTBALL_BASE}/fixtures",
                headers=_AF_HEADERS,
                params={"date": date_str, "league": league_id, "season": season, "status": "FT"},
                timeout=15,
            )
            calls_used += 1
            data = r.json()
        except Exception as exc:
            print(f"  Error fetching fixtures league={league_id}: {exc}")
            continue

        errors = data.get("errors")
        if errors:
            plan_err = str(errors).lower()
            if "plan" in plan_err or "free" in plan_err:
                print(f"  [fixture_stats] Free plan date restriction: {errors}")
                print("  NOTE: API-Football free plan only allows ±2 days from today.")
                print("  Historical fixture stats require an API-Football paid plan.")
                return
            print(f"  Errors for league {league_id}: {errors}")
            continue

        for fix in data.get("response", []):
            fid = fix.get("fixture", {}).get("id")
            if fid and fid not in fetched_fixtures:
                fetched_fixtures.add(fid)
                league_name = fix.get("league", {}).get("name", "")
                print(f"  Fixture {fid}: {fix['teams']['home']['name']} v {fix['teams']['away']['name']}")
                n = fetch_fixture_stats(fid, date_str, league_id, dry_run=dry_run)
                total_written += n
                calls_used += 2  # player + stats calls

        time.sleep(0.5)

    print(f"[fixture_stats] Done: {len(fetched_fixtures)} fixtures, "
          f"{total_written} rows written, ~{calls_used} API calls used")


# ─────────────────────────────────────────────────────────────────────────────
# Status report
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print("\n━━━ Soccer Pipeline Status ━━━")

    # Phase 1a
    cur.execute("SELECT COUNT(*) FROM bet_legs WHERE sport='Soccer'")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM bet_legs WHERE sport='Soccer' AND subtype IS NOT NULL")
    parsed = cur.fetchone()[0]
    cur.execute("SELECT subtype, COUNT(*) FROM bet_legs WHERE sport='Soccer' AND subtype IS NOT NULL GROUP BY subtype ORDER BY 2 DESC")
    subtypes = cur.fetchall()
    print(f"\nPhase 1a — ETL parse: {parsed}/{total} soccer legs have subtype")
    for st, cnt in subtypes:
        print(f"  {st:30s}  {cnt}")

    # Phase 1b
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "player_soccer_stats" in tables:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT fixture_id) FROM player_soccer_stats")
        prows, pfixtures = cur.fetchone()
        print(f"\nPhase 1b — player_soccer_stats: {prows} rows across {pfixtures} fixtures")
    else:
        print("\nPhase 1b — player_soccer_stats: table not yet created")

    if "fixture_soccer_stats" in tables:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT fixture_id) FROM fixture_soccer_stats")
        frows, ffixtures = cur.fetchone()
        print(f"Phase 1b — fixture_soccer_stats: {frows} rows across {ffixtures} fixtures")
    else:
        print("Phase 1b — fixture_soccer_stats: table not yet created")

    # Phase 1c
    if "team_soccer_form" in tables:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT team_name) FROM team_soccer_form")
        trows, tteams = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM team_soccer_form WHERE games_found >= 5")
        good = cur.fetchone()[0]
        print(f"\nPhase 1c — team_soccer_form: {trows} rows, {tteams} teams, "
              f"{good} with ≥5 games of history")
    else:
        print("\nPhase 1c — team_soccer_form: table not yet created")

    # Coverage summary
    cur.execute("""
        SELECT COUNT(*) FROM bet_legs bl
        JOIN bets b ON bl.bet_id=b.id
        WHERE bl.sport='Soccer' AND bl.market_type='Total' AND bl.subtype='total_goals'
    """)
    total_goals_legs = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM bet_legs bl
        JOIN bets b ON bl.bet_id=b.id
        WHERE bl.sport='Soccer' AND bl.market_type='Moneyline'
    """)
    ml_legs = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM bet_legs bl
        JOIN bets b ON bl.bet_id=b.id
        WHERE bl.sport='Soccer' AND bl.subtype='shots_on_target'
    """)
    shots_legs = cur.fetchone()[0]

    print(f"\n━━━ Training Sample Preview ━━━")
    print(f"  Total Goals legs:       {total_goals_legs}  (form data: can train now)")
    print(f"  Moneyline/DC legs:      {ml_legs}  (form data: can train now)")
    print(f"  Shots On Target legs:   {shots_legs}  (needs API-Football premium for player stats)")

    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer feature table backfill")
    parser.add_argument("--team-form",     action="store_true", help="Phase 1c: build team_soccer_form")
    parser.add_argument("--fixture-stats", action="store_true", help="Phase 1b: fetch fixture/player stats")
    parser.add_argument("--date",          type=str,            help="Date for fixture stats (YYYY-MM-DD)")
    parser.add_argument("--status",        action="store_true", help="Print status report")
    parser.add_argument("--dry-run",       action="store_true", help="Print plan without writing")

    args = parser.parse_args()

    if not any([args.team_form, args.fixture_stats, args.status]):
        parser.print_help()
        sys.exit(0)

    if args.status:
        print_status()

    if args.team_form:
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Building team_soccer_form ...")
        build_team_form(dry_run=args.dry_run)

    if args.fixture_stats:
        if not args.date:
            print("--fixture-stats requires --date YYYY-MM-DD")
            sys.exit(1)
        fetch_all_fixture_stats(args.date, dry_run=args.dry_run)

    if not args.status and any([args.team_form, args.fixture_stats]):
        print_status()
