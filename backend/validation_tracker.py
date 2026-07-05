"""
validation_tracker.py — Post-settlement validation for BetIQ prediction layers.

Four validation layers run after each settlement batch:
  1. Scout predictions  — scout_grade/scout_hit_prob vs actual leg outcomes
  2. Player projections — player_projections vs ESPN box-score actuals
  3. Placement         — ROI by source (proxy for rec_type)
  4. Signal agreement  — WR by count of aligned signals

All results are persisted to validation tables and surfaced via run_all_validations().
"""
from __future__ import annotations

import os
import math
import sqlite3
import logging
import urllib.request
import json
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# ── DB path ───────────────────────────────────────────────────────────────────
_BETS_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
)


# ─── Table DDL ────────────────────────────────────────────────────────────────

def _ensure_tables(db_path: str = _BETS_DB_PATH) -> None:
    """Create validation tables if they don't already exist. Safe to call many times."""
    con = sqlite3.connect(db_path, timeout=15)
    try:
        # Migrate existing scout_validation rows: add trajectory columns if absent
        for col, defn in [("window_start", "TEXT"), ("window_end", "TEXT")]:
            try:
                con.execute(f"ALTER TABLE scout_validation ADD COLUMN {col} {defn}")
                con.commit()
            except Exception:
                pass  # column already exists

        con.executescript("""
        -- 1. Scout prediction accuracy
        CREATE TABLE IF NOT EXISTS scout_validation (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at    TEXT    NOT NULL,
            window_days     INTEGER NOT NULL,
            n_legs          INTEGER NOT NULL,
            n_with_scout    INTEGER NOT NULL,
            per_grade       TEXT,          -- JSON: {grade: {n, wins, wr, brier}}
            overall_brier   REAL,
            insufficient_data INTEGER DEFAULT 0,
            window_start    TEXT,          -- earliest game_date in sample
            window_end      TEXT           -- latest game_date in sample
        );

        -- 2. Player projection accuracy vs ESPN actuals
        CREATE TABLE IF NOT EXISTS projection_validation (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name     TEXT NOT NULL,
            stat_type       TEXT NOT NULL,
            game_date       TEXT NOT NULL,
            sport           TEXT,
            projected       REAL,
            actual          REAL,
            error           REAL,           -- actual - projected
            abs_error       REAL,
            pct_error       REAL,           -- (actual - projected) / projected
            direction       TEXT,           -- over / under / exact
            validated_at    TEXT NOT NULL,
            UNIQUE(player_name, stat_type, game_date)
        );

        -- 3. Placement validation (ROI by rec source)
        CREATE TABLE IF NOT EXISTS placement_validation (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at    TEXT    NOT NULL,
            window_days     INTEGER NOT NULL,
            source          TEXT    NOT NULL,
            n               INTEGER NOT NULL,
            wins            INTEGER NOT NULL,
            win_rate        REAL,
            total_stake     REAL,
            total_profit    REAL,
            roi_pct         REAL
        );

        -- 4. Signal agreement log
        CREATE TABLE IF NOT EXISTS signal_agreement_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at    TEXT    NOT NULL,
            window_days     INTEGER NOT NULL,
            signals_aligned INTEGER NOT NULL,   -- 0, 1, 2, or 3
            n_bets          INTEGER NOT NULL,
            wins            INTEGER NOT NULL,
            win_rate        REAL,
            sample_note     TEXT
        );
        """)
        con.commit()
    finally:
        con.close()


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _db_conn(db_path: str = _BETS_DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=15)
    con.row_factory = sqlite3.Row
    return con


def _pct(num: int, denom: int, places: int = 3) -> Optional[float]:
    if not denom:
        return None
    return round(num / denom, places)


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─── 1. Scout validation ──────────────────────────────────────────────────────

def validate_scout_predictions(
    db_path: str = _BETS_DB_PATH,
    window_days: int = 30,
) -> dict:
    """
    Measure how well scout_grade / scout_hit_prob predicts actual leg outcomes.

    Returns immediately with insufficient_data=True when < 5 legs have scout data,
    since 0 legs are currently populated (builds infra for future use).
    """
    _ensure_tables(db_path)
    since = (datetime.utcnow() - timedelta(days=window_days)).strftime("%Y-%m-%d")

    con = _db_conn(db_path)
    try:
        rows = con.execute("""
            SELECT
                mbl.scout_grade,
                mbl.scout_hit_prob,
                mbl.leg_result,
                mbl.win_prob,
                mb.game_date
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND mbl.scout_grade IS NOT NULL
              AND mbl.scout_hit_prob IS NOT NULL
              AND mb.settled_at >= :since
        """, {"since": since}).fetchall()

        n_total_settled = con.execute("""
            SELECT COUNT(*) FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND mb.settled_at >= :since
        """, {"since": since}).fetchone()[0]

    finally:
        con.close()

    n_with_scout = len(rows)
    now_iso = datetime.utcnow().isoformat()

    # Date range of sample for trajectory tracking
    game_dates = [r["game_date"] for r in rows if r["game_date"]]
    window_start = min(game_dates) if game_dates else None
    window_end   = max(game_dates) if game_dates else None

    if n_with_scout < 5:
        result = {
            "n_legs":           n_total_settled,
            "n_with_scout":     n_with_scout,
            "insufficient_data": True,
            "message":          f"Only {n_with_scout} leg(s) have scout data — need ≥5 to compute metrics",
            "window_days":      window_days,
        }
        con = _db_conn(db_path)
        try:
            con.execute("""
                INSERT INTO scout_validation
                  (evaluated_at, window_days, n_legs, n_with_scout, insufficient_data,
                   window_start, window_end)
                VALUES (:ts, :w, :nl, :ns, 1, :ws, :we)
            """, {"ts": now_iso, "w": window_days, "nl": n_total_settled,
                  "ns": n_with_scout, "ws": window_start, "we": window_end})
            con.commit()
        finally:
            con.close()
        return result

    # ── Compute per-grade metrics ─────────────────────────────────────────────
    from collections import defaultdict
    grade_buckets: dict[str, list] = defaultdict(list)

    brier_sum = 0.0
    for row in rows:
        grade = (row["scout_grade"] or "?").upper()
        hit_prob = _safe_float(row["scout_hit_prob"]) or 0.5
        outcome = 1 if row["leg_result"] == "WIN" else 0
        grade_buckets[grade].append((hit_prob, outcome))
        brier_sum += (hit_prob - outcome) ** 2

    overall_brier = round(brier_sum / n_with_scout, 4)

    per_grade: dict[str, dict] = {}
    for grade, items in sorted(grade_buckets.items()):
        n = len(items)
        wins = sum(o for _, o in items)
        b = sum((p - o) ** 2 for p, o in items) / n
        per_grade[grade] = {
            "n":       n,
            "wins":    wins,
            "wr":      _pct(wins, n),
            "brier":   round(b, 4),
        }

    result = {
        "n_legs":           n_total_settled,
        "n_with_scout":     n_with_scout,
        "insufficient_data": False,
        "window_days":      window_days,
        "window_start":     window_start,
        "window_end":       window_end,
        "overall_brier":    overall_brier,
        "per_grade":        per_grade,
    }

    con = _db_conn(db_path)
    try:
        con.execute("""
            INSERT INTO scout_validation
              (evaluated_at, window_days, n_legs, n_with_scout, per_grade,
               overall_brier, insufficient_data, window_start, window_end)
            VALUES (:ts, :w, :nl, :ns, :pg, :ob, 0, :ws, :we)
        """, {
            "ts": now_iso, "w": window_days,
            "nl": n_total_settled, "ns": n_with_scout,
            "pg": json.dumps(per_grade), "ob": overall_brier,
            "ws": window_start, "we": window_end,
        })
        con.commit()
    finally:
        con.close()

    return result


# ─── 2. Player projection validation ─────────────────────────────────────────

# Mapping: projection stat_type → ESPN box score stat key
_ESPN_STAT_KEYS: dict[str, dict[str, str]] = {
    "MLB": {
        "hits":        "hits",
        "home_runs":   "homeRuns",
        "rbi":         "RBIs",
        "runs":        "runs",
        "strikeouts":  "strikeouts",
        "total_bases": "totalBases",   # may not exist; falls back gracefully
    },
    "NHL": {
        "shots_on_goal": "shotsOnGoal",
        "goals":         "goals",
        "assists":        "assists",
    },
}

# ESPN sport path fragments
_ESPN_SPORT_PATH: dict[str, str] = {
    "MLB": "baseball/mlb",
    "NHL": "hockey/nhl",
}


def _fetch_espn_box_score(sport: str, event_id: str, timeout: int = 8) -> Optional[dict]:
    """Fetch ESPN summary JSON for an event. Returns the full dict or None on error."""
    path = _ESPN_SPORT_PATH.get(sport.upper())
    if not path:
        return None
    url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/summary?event={event_id}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BetIQ/1.0 (projection-validator)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.debug("ESPN box score fetch failed for %s %s: %s", sport, event_id, e)
        return None


def _extract_player_stats(box_data: dict, sport: str) -> dict[str, dict[str, float]]:
    """
    Parse ESPN summary JSON into {player_name_lower: {stat_key: value}}.

    Handles the nested ESPN boxscore → players → statistics structure.
    Returns an empty dict on any parsing failure.
    """
    result: dict[str, dict[str, float]] = {}
    try:
        boxscore = box_data.get("boxscore", {})
        players_sections = boxscore.get("players", [])
        for section in players_sections:
            statistics = section.get("statistics", [])
            for stat_group in statistics:
                keys = stat_group.get("keys", [])         # list of stat names
                athletes = stat_group.get("athletes", []) # list of player rows
                for athlete_entry in athletes:
                    name = (
                        athlete_entry.get("athlete", {}).get("displayName", "")
                        or athlete_entry.get("athlete", {}).get("fullName", "")
                    )
                    if not name:
                        continue
                    stats_raw = athlete_entry.get("stats", [])
                    name_key = name.strip().lower()
                    if name_key not in result:
                        result[name_key] = {}
                    for k, v in zip(keys, stats_raw):
                        try:
                            result[name_key][k] = float(v)
                        except (TypeError, ValueError):
                            pass
    except Exception as e:
        log.debug("ESPN stat extraction error: %s", e)
    return result


def _fuzzy_player_match(name: str, candidates: dict[str, dict]) -> Optional[str]:
    """
    Find the best matching player name from ESPN data.
    Uses substring and fuzzy matching.
    """
    import difflib
    name_lower = name.strip().lower()

    # Exact match first
    if name_lower in candidates:
        return name_lower

    # Substring match (e.g. "J. Smith" vs "John Smith")
    for cand in candidates:
        if name_lower in cand or cand in name_lower:
            return cand

    # Last-name match
    name_parts = name_lower.split()
    last_name = name_parts[-1] if name_parts else ""
    if len(last_name) >= 4:
        for cand in candidates:
            if last_name in cand:
                return cand

    # Difflib fallback
    matches = difflib.get_close_matches(name_lower, candidates.keys(), n=1, cutoff=0.70)
    return matches[0] if matches else None


def _get_event_ids_for_date(game_date: str, db_path: str) -> list[tuple[str, str]]:
    """
    Return [(event_id, sport)] for fixtures on a given game_date from the Fixture table.
    Falls back to settle_log if Fixture has nothing.
    """
    con = _db_conn(db_path)
    try:
        rows = con.execute("""
            SELECT DISTINCT espn_event_id, sport_key
            FROM fixtures
            WHERE game_date_ct = :gd
              AND espn_event_id IS NOT NULL
              AND espn_event_id != ''
        """, {"gd": game_date}).fetchall()
        result = [(r["espn_event_id"], r["sport_key"]) for r in rows if r["espn_event_id"]]
    except Exception:
        result = []
    finally:
        con.close()
    return result


def _sport_key_to_sport(sport_key: str) -> str:
    """Convert OddsAPI sport key (e.g. baseball_mlb) to short name (MLB)."""
    sk = sport_key.upper()
    if "MLB" in sk or "BASEBALL" in sk:
        return "MLB"
    if "NHL" in sk or "HOCKEY" in sk:
        return "NHL"
    if "NFL" in sk or "FOOTBALL" in sk:
        return "NFL"
    if "NBA" in sk or "BASKETBALL" in sk:
        return "NBA"
    return sk


def validate_player_projections(
    db_path: str = _BETS_DB_PATH,
    game_date: Optional[str] = None,
) -> dict:
    """
    Compare player_projections to ESPN box score actuals.

    Skips game_date/player/stat combos already in projection_validation (idempotent).
    Returns summary stats: {validated, skipped_existing, not_found, by_stat: {stat: {mae, rmse, bias, n}}}
    """
    _ensure_tables(db_path)
    now_iso = datetime.utcnow().isoformat()

    # Default: validate yesterday + today
    if game_date is None:
        today = datetime.utcnow().date()
        dates_to_check = [
            (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
        ]
    else:
        dates_to_check = [game_date]

    con = _db_conn(db_path)
    try:
        projections_by_date: dict[str, list] = {}
        for gd in dates_to_check:
            rows = con.execute("""
                SELECT player_name, stat_type, sport, projection, game_date
                FROM player_projections
                WHERE game_date = :gd
            """, {"gd": gd}).fetchall()
            if rows:
                projections_by_date[gd] = [dict(r) for r in rows]
    finally:
        con.close()

    if not projections_by_date:
        return {"validated": 0, "skipped_existing": 0, "not_found": 0,
                "message": "No projections found for requested dates", "by_stat": {}}

    total_validated = 0
    total_skipped   = 0
    total_not_found = 0
    errors_by_stat: dict[str, list[float]] = {}   # stat_type → list of (actual - projected)

    for gd, proj_rows in projections_by_date.items():
        # Get ESPN event IDs for this date
        event_ids = _get_event_ids_for_date(gd, db_path)
        if not event_ids:
            log.info("[validation] No ESPN event IDs for %s — skipping projection validation", gd)
            total_not_found += len(proj_rows)
            continue

        # Build combined player stat dict across all events for this date
        # {sport: {player_name_lower: {stat_key: value}}}
        espn_by_sport: dict[str, dict[str, dict[str, float]]] = {}
        for event_id, sport_key in event_ids:
            sport = _sport_key_to_sport(sport_key)
            if sport not in _ESPN_STAT_KEYS:
                continue
            if sport not in espn_by_sport:
                espn_by_sport[sport] = {}
            box = _fetch_espn_box_score(sport, event_id)
            if box:
                player_stats = _extract_player_stats(box, sport)
                espn_by_sport[sport].update(player_stats)

        # Match each projection to ESPN data
        insert_rows = []
        for proj in proj_rows:
            player  = proj["player_name"]
            stat    = proj["stat_type"]
            sport   = proj["sport"].upper()
            proj_v  = _safe_float(proj["projection"])

            if proj_v is None:
                total_not_found += 1
                continue

            # Look up ESPN key for this stat
            espn_key = _ESPN_STAT_KEYS.get(sport, {}).get(stat)
            if not espn_key:
                total_not_found += 1
                continue

            espn_players = espn_by_sport.get(sport, {})
            matched_key  = _fuzzy_player_match(player, espn_players)

            if not matched_key:
                total_not_found += 1
                continue

            actual_v = espn_players[matched_key].get(espn_key)
            if actual_v is None:
                # Try totalBases workaround (TB = H + 2B×1 + 3B×2 + HR×3 — not available atomically)
                total_not_found += 1
                continue

            error     = actual_v - proj_v
            abs_err   = abs(error)
            pct_err   = (error / proj_v) if proj_v != 0 else None
            direction = "over" if actual_v > proj_v else ("under" if actual_v < proj_v else "exact")

            insert_rows.append({
                "player_name": player,
                "stat_type":   stat,
                "game_date":   gd,
                "sport":       sport,
                "projected":   proj_v,
                "actual":      actual_v,
                "error":       round(error, 3),
                "abs_error":   round(abs_err, 3),
                "pct_error":   round(pct_err, 4) if pct_err is not None else None,
                "direction":   direction,
                "validated_at": now_iso,
            })

            if stat not in errors_by_stat:
                errors_by_stat[stat] = []
            errors_by_stat[stat].append(error)

        # Write new rows (INSERT OR IGNORE for idempotency)
        if insert_rows:
            con = _db_conn(db_path)
            try:
                for r in insert_rows:
                    try:
                        con.execute("""
                            INSERT OR IGNORE INTO projection_validation
                              (player_name, stat_type, game_date, sport,
                               projected, actual, error, abs_error, pct_error,
                               direction, validated_at)
                            VALUES
                              (:player_name, :stat_type, :game_date, :sport,
                               :projected, :actual, :error, :abs_error, :pct_error,
                               :direction, :validated_at)
                        """, r)
                        if con.execute("SELECT changes()").fetchone()[0]:
                            total_validated += 1
                        else:
                            total_skipped += 1
                    except Exception as ie:
                        log.debug("projection_validation insert error: %s", ie)
                        total_skipped += 1
                con.commit()
            finally:
                con.close()

    # ── Aggregate stats ──────────────────────────────────────────────────────
    by_stat: dict[str, dict] = {}
    for stat, errs in errors_by_stat.items():
        n   = len(errs)
        mae = round(sum(abs(e) for e in errs) / n, 3) if n else None
        rmse = round(math.sqrt(sum(e**2 for e in errs) / n), 3) if n else None
        bias = round(sum(errs) / n, 3) if n else None
        over = sum(1 for e in errs if e > 0)
        by_stat[stat] = {
            "n":    n,
            "mae":  mae,
            "rmse": rmse,
            "bias": bias,
            "over_pct":  _pct(over, n),
        }

    return {
        "validated":        total_validated,
        "skipped_existing": total_skipped,
        "not_found":        total_not_found,
        "by_stat":          by_stat,
        "dates_checked":    dates_to_check,
    }


# ─── 3. Placement validation ──────────────────────────────────────────────────

# Default cutoff: calibration fixes shipped 2026-05-16 (win_prob scale bug + _ALT_MARKETS fix).
# Pre-fix data had systematic MLB alt-spread overconfidence and should not pollute placement QA.
_PLACEMENT_DEFAULT_SINCE = "2026-05-16"


def validate_placement(
    db_path: str = _BETS_DB_PATH,
    since: Optional[str] = None,
    window_days: Optional[int] = None,
    bet_category: Optional[str] = None,
) -> dict:
    """
    Compute ROI by source (proxy for rec_type) for settled bets on or after `since`.

    Date precedence:
      1. `since` (explicit YYYY-MM-DD) — use as-is
      2. `window_days` (int) — compute since = today - N days
      3. Default: _PLACEMENT_DEFAULT_SINCE ('2026-05-16', post-calibration-fix)

    bet_category filter: 'team' | 'player_prop' | None (all).
    When set, only bets where ALL legs match the category are included.
    Rows are inserted fresh each run (snapshot log, not idempotent).
    """
    _ensure_tables(db_path)
    now_iso = datetime.utcnow().isoformat()

    if since is not None:
        since_date = since
    elif window_days is not None:
        since_date = (datetime.utcnow() - timedelta(days=window_days)).strftime("%Y-%m-%d")
    else:
        since_date = _PLACEMENT_DEFAULT_SINCE

    con = _db_conn(db_path)
    try:
        # Build category sub-filter: include only bet IDs where all legs match category
        cat_join = ""
        cat_where = ""
        params: dict = {"since": since_date}
        if bet_category:
            cat_join = """
                JOIN (
                    SELECT mock_bet_id
                    FROM mock_bet_legs
                    GROUP BY mock_bet_id
                    HAVING SUM(CASE WHEN COALESCE(bet_category,'team') = :cat THEN 1 ELSE 0 END) = COUNT(*)
                ) cat_filter ON cat_filter.mock_bet_id = mock_bets.id
            """
            params["cat"] = bet_category

        rows = con.execute(f"""
            SELECT
                source,
                COUNT(*)                                                            AS n,
                SUM(CASE WHEN status = 'SETTLED_WIN' THEN 1 ELSE 0 END)            AS wins,
                SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END)                   AS total_stake,
                SUM(COALESCE(actual_profit, 0))                                    AS total_profit
            FROM mock_bets
            {cat_join}
            WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND game_date >= :since
              AND source IS NOT NULL
            GROUP BY source
            ORDER BY total_profit DESC
        """, params).fetchall()
    finally:
        con.close()

    results = []
    for r in rows:
        source  = r["source"]
        n       = r["n"] or 0
        wins    = r["wins"] or 0
        stake   = r["total_stake"] or 0.0
        profit  = r["total_profit"] or 0.0
        wr      = _pct(wins, n)
        roi     = round(profit / stake, 4) if stake > 0 else None
        results.append({
            "source":       source,
            "n":            n,
            "wins":         wins,
            "win_rate":     wr,
            "total_stake":  round(stake, 2),
            "total_profit": round(profit, 2),
            "roi_pct":      round(roi * 100, 2) if roi is not None else None,
        })

    if results:
        con = _db_conn(db_path)
        try:
            for r in results:
                con.execute("""
                    INSERT INTO placement_validation
                      (evaluated_at, window_days, source, n, wins, win_rate,
                       total_stake, total_profit, roi_pct)
                    VALUES
                      (:ts, :w, :src, :n, :wins, :wr, :stake, :profit, :roi)
                """, {
                    "ts":     now_iso,
                    "w":      since_date,   # stores the since date string in the window_days column
                    "src":    r["source"],
                    "n":      r["n"],
                    "wins":   r["wins"],
                    "wr":     r["win_rate"],
                    "stake":  r["total_stake"],
                    "profit": r["total_profit"],
                    "roi":    r["roi_pct"],
                })
            con.commit()
        finally:
            con.close()

    # Surface T1/T2/T3 rollups using signal-based bet_tier column
    def _tier_rollup_from_results(tier_val: str, label: str) -> dict:
        """Roll up by bet_tier.  Falls back to source-mapping for pre-tier bets."""
        # Bets with bet_tier populated: use it directly
        # Bets without bet_tier (legacy): mapped via source as before
        _LEGACY_T1 = {"prospective_pm", "top_picks_page"}
        _LEGACY_T2 = {"prospective"}
        n, wins, stake, profit = 0, 0, 0.0, 0.0
        for r in results:
            src = r["source"]
            # Approximate: source-level stats can't split by bet_tier.
            # For the tier_rollup we map: pm/top→T1, prospective→T2, rest→T3.
            # This is the legacy mapping — once all bets have bet_tier, this
            # should be replaced with a dedicated SQL query on bet_tier.
            if tier_val == "T1" and src in _LEGACY_T1:
                n += r["n"]; wins += r["wins"]; stake += r["total_stake"]; profit += r["total_profit"]
            elif tier_val == "T2" and src in _LEGACY_T2:
                n += r["n"]; wins += r["wins"]; stake += r["total_stake"]; profit += r["total_profit"]
            elif tier_val == "T3" and src not in (_LEGACY_T1 | _LEGACY_T2):
                n += r["n"]; wins += r["wins"]; stake += r["total_stake"]; profit += r["total_profit"]
        return {
            "label":        label,
            "n":            n,
            "wins":         wins,
            "win_rate":     _pct(wins, n),
            "roi_pct":      round(profit / stake * 100, 2) if stake > 0 else None,
            "total_profit": round(profit, 2),
        }

    return {
        "since":       since_date,
        "by_source":   results,
        "tier_rollup": {
            "T1": _tier_rollup_from_results("T1", "High conviction"),
            "T2": _tier_rollup_from_results("T2", "Moderate conviction"),
            "T3": _tier_rollup_from_results("T3", "Monitor only"),
        },
    }


# ─── 4. Signal agreement ──────────────────────────────────────────────────────

# Signal definitions (4th signal activates when scout data flows)
_SIG_WIN_PROB_THRESH  = 0.65
_SIG_CLV_POSITIVE     = False  # clv_cents < 0 means we beat close → positive CLV
_SIG_GRADE_SET        = {"A", "B"}


def _count_signals_for_bet(win_prob, clv_cents, grade) -> int:
    """Count how many of the 3 core signals are aligned for a given bet."""
    count = 0
    if win_prob is not None and win_prob >= _SIG_WIN_PROB_THRESH:
        count += 1
    if clv_cents is not None and clv_cents < 0:
        count += 1
    if grade is not None and str(grade).upper() in _SIG_GRADE_SET:
        count += 1
    return count


def compute_signal_agreement(
    db_path: str = _BETS_DB_PATH,
    window_days: int = 30,
    bet_category: Optional[str] = None,
) -> dict:
    """
    Compute win rate by number of aligned signals (0-3) across settled bets.

    Signals:
      1. model win_prob ≥ 0.65 (leg-level average)
      2. clv_cents < 0 (beat close, i.e. positive CLV) — uses average across legs
      3. grade in {A, B} — leg-level average (majority rule)

    bet_category filter: 'team' | 'player_prop' | None (all).
    A 4th scout signal will be added once scout_hit_prob data flows.
    """
    _ensure_tables(db_path)
    now_iso = datetime.utcnow().isoformat()
    since   = (datetime.utcnow() - timedelta(days=window_days)).strftime("%Y-%m-%d")

    con = _db_conn(db_path)
    try:
        # Build category sub-filter: include only bets where all legs match category
        cat_having = ""
        params: dict = {"since": since}
        if bet_category:
            cat_having = (
                "AND SUM(CASE WHEN COALESCE(mbl.bet_category,'team') = :cat THEN 1 ELSE 0 END) = COUNT(*)"
            )
            params["cat"] = bet_category

        # Pull settled bets with their leg aggregates
        rows = con.execute(f"""
            SELECT
                mb.id,
                mb.status,
                AVG(mbl.win_prob)                                                      AS avg_wp,
                AVG(mbl.clv_cents)                                                     AS avg_clv,
                SUM(CASE WHEN UPPER(mbl.grade) IN ('A','B') THEN 1 ELSE 0 END) * 1.0
                    / COUNT(mbl.id)                                                    AS ab_grade_frac
            FROM mock_bets mb
            JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
            WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND mb.settled_at >= :since
            GROUP BY mb.id
            HAVING 1=1 {cat_having}
        """, params).fetchall()
    finally:
        con.close()

    # Bucket bets by signal count
    from collections import defaultdict
    buckets: dict[int, list[int]] = defaultdict(list)  # count → [1 win, 0 loss]

    for row in rows:
        wp      = _safe_float(row["avg_wp"])
        clv     = _safe_float(row["avg_clv"])
        ab_frac = _safe_float(row["ab_grade_frac"])
        grade_proxy = "A" if (ab_frac is not None and ab_frac >= 0.5) else "C"

        n_signals = _count_signals_for_bet(wp, clv, grade_proxy)
        is_win    = 1 if row["status"] == "SETTLED_WIN" else 0
        buckets[n_signals].append(is_win)

    # Persist and build result
    signal_rows = []
    for k in range(4):
        outcomes = buckets.get(k, [])
        n    = len(outcomes)
        wins = sum(outcomes)
        wr   = _pct(wins, n)
        note = "insufficient" if n < 10 else None
        signal_rows.append({
            "signals_aligned": k,
            "n_bets":          n,
            "wins":            wins,
            "win_rate":        wr,
            "sample_note":     note,
        })

    if signal_rows:
        con = _db_conn(db_path)
        try:
            for r in signal_rows:
                con.execute("""
                    INSERT INTO signal_agreement_log
                      (evaluated_at, window_days, signals_aligned, n_bets, wins, win_rate, sample_note)
                    VALUES (:ts, :w, :sa, :nb, :wins, :wr, :note)
                """, {
                    "ts":   now_iso,
                    "w":    window_days,
                    "sa":   r["signals_aligned"],
                    "nb":   r["n_bets"],
                    "wins": r["wins"],
                    "wr":   r["win_rate"],
                    "note": r["sample_note"],
                })
            con.commit()
        finally:
            con.close()

    # Summarise edge: WR at max signal count vs 0
    wr_3 = signal_rows[3]["win_rate"] if signal_rows[3]["n_bets"] >= 5 else None
    wr_0 = signal_rows[0]["win_rate"] if signal_rows[0]["n_bets"] >= 5 else None
    edge = None
    if wr_3 is not None and wr_0 is not None:
        edge = round(wr_3 - wr_0, 3)

    return {
        "window_days":         window_days,
        "by_signal_count":     signal_rows,
        "signal_definitions":  [
            f"win_prob ≥ {_SIG_WIN_PROB_THRESH}",
            "clv_cents < 0 (beat close)",
            "grade in {A, B}",
        ],
        "edge_3v0_signals":    edge,
        "total_bets":          sum(r["n_bets"] for r in signal_rows),
    }


# ─── Master runner ────────────────────────────────────────────────────────────

def run_all_validations(
    db_path: str = _BETS_DB_PATH,
    window_days: int = 30,
) -> dict:
    """
    Run all four validation layers and return a combined summary dict.
    Designed to be called in the post-settlement hook.
    Failures in any layer are caught and returned as errors (never raises).
    """
    result: dict = {
        "run_at":     datetime.utcnow().isoformat(),
        "window_days": window_days,
    }

    # Scout accuracy
    try:
        result["scout_accuracy"] = validate_scout_predictions(db_path, window_days)
    except Exception as e:
        log.warning("scout validation failed: %s", e)
        result["scout_accuracy"] = {"error": str(e)}

    # Player projection accuracy
    try:
        result["projection_accuracy"] = validate_player_projections(db_path)
    except Exception as e:
        log.warning("projection validation failed: %s", e)
        result["projection_accuracy"] = {"error": str(e)}

    # Placement accuracy — default to post-calibration-fix period only
    try:
        result["placement_accuracy"] = validate_placement(db_path)
    except Exception as e:
        log.warning("placement validation failed: %s", e)
        result["placement_accuracy"] = {"error": str(e)}

    # Signal agreement
    try:
        result["signal_agreement"] = compute_signal_agreement(db_path, window_days)
    except Exception as e:
        log.warning("signal agreement computation failed: %s", e)
        result["signal_agreement"] = {"error": str(e)}

    return result


# ─── Convenience: latest snapshot ────────────────────────────────────────────

def get_latest_validation_summary(db_path: str = _BETS_DB_PATH) -> dict:
    """
    Return the most recent persisted row from each validation table for dashboards.
    Does not re-run validations; use run_all_validations() for fresh data.
    """
    _ensure_tables(db_path)
    con = _db_conn(db_path)
    try:
        # Scout
        scout_row = con.execute("""
            SELECT * FROM scout_validation ORDER BY id DESC LIMIT 1
        """).fetchone()

        # Projections (last 7 days summary)
        proj_row = con.execute("""
            SELECT
                stat_type,
                COUNT(*) as n,
                AVG(abs_error) as mae,
                SUM(CASE WHEN direction = 'over' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as over_rate
            FROM projection_validation
            WHERE validated_at >= date('now', '-7 days')
            GROUP BY stat_type
        """).fetchall()

        # Placement (latest run)
        placement_rows = con.execute("""
            SELECT * FROM placement_validation
            WHERE evaluated_at = (SELECT MAX(evaluated_at) FROM placement_validation)
        """).fetchall()

        # Signal agreement (latest run)
        signal_rows = con.execute("""
            SELECT * FROM signal_agreement_log
            WHERE evaluated_at = (SELECT MAX(evaluated_at) FROM signal_agreement_log)
            ORDER BY signals_aligned
        """).fetchall()

    finally:
        con.close()

    return {
        "scout": dict(scout_row) if scout_row else None,
        "projections_7d": [
            {"stat_type": r["stat_type"], "n": r["n"],
             "mae": round(r["mae"] or 0, 3), "over_rate": round(r["over_rate"] or 0, 3)}
            for r in proj_row
        ],
        "placement": [dict(r) for r in placement_rows],
        "signal_agreement": [dict(r) for r in signal_rows],
    }


# ─── Scout trajectory & grade driver analysis ─────────────────────────────────

# Per-grade improvement targets (updated monthly as sample grows)
SCOUT_TARGETS = {
    "A": {"wr": 0.75, "brier": 0.185},  # current baseline: 73.2% WR, 0.198 Brier (n=112)
    "B": {"wr": 0.65, "brier": 0.220},  # current baseline: 60.0% WR, 0.240 Brier (n=170)
    "C": {"wr": 0.60, "brier": 0.230},  # uncertain at n=36 — 72.2% likely to regress
    "D": {"wr": 0.55, "brier": 0.245},  # directional only at n=7, 57.1% WR
}


def get_scout_trajectory(db_path: str = _BETS_DB_PATH, lookback_runs: int = 10) -> dict:
    """
    Return per-grade WR and Brier trend across the last N scout_validation runs,
    plus vs-target deltas for the most recent run.

    Used in the daily summary to track continuous improvement.
    """
    _ensure_tables(db_path)
    con = _db_conn(db_path)
    try:
        runs = con.execute("""
            SELECT id, evaluated_at, window_start, window_end, per_grade,
                   overall_brier, n_with_scout, insufficient_data
            FROM scout_validation
            WHERE insufficient_data = 0
            ORDER BY id DESC
            LIMIT :n
        """, {"n": lookback_runs}).fetchall()
    finally:
        con.close()

    if not runs:
        return {"runs": [], "current": None, "targets": SCOUT_TARGETS}

    parsed_runs = []
    for r in runs:
        try:
            pg = json.loads(r["per_grade"] or "{}")
        except Exception:
            pg = {}
        parsed_runs.append({
            "evaluated_at": r["evaluated_at"],
            "window_start": r["window_start"],
            "window_end":   r["window_end"],
            "n":            r["n_with_scout"],
            "overall_brier": r["overall_brier"],
            "per_grade":    pg,
        })

    # Most recent run — compute vs-target deltas
    current = parsed_runs[0]
    grade_status: dict[str, dict] = {}
    for grade, targets in SCOUT_TARGETS.items():
        g = current["per_grade"].get(grade, {})
        n   = g.get("n", 0)
        wr  = g.get("wr")
        brier = g.get("brier")

        # Trend: compare to second-most-recent run if available
        prev_wr    = None
        prev_brier = None
        if len(parsed_runs) > 1:
            prev_g = parsed_runs[1]["per_grade"].get(grade, {})
            prev_wr    = prev_g.get("wr")
            prev_brier = prev_g.get("brier")

        trend_wr    = None
        trend_brier = None
        if wr is not None and prev_wr is not None:
            delta = round(wr - prev_wr, 3)
            trend_wr = "improving" if delta > 0.01 else ("degrading" if delta < -0.01 else "stable")
        elif wr is not None:
            trend_wr = "new_signal"

        if brier is not None and prev_brier is not None:
            delta_b = round(brier - prev_brier, 4)
            trend_brier = "improving" if delta_b < -0.005 else ("degrading" if delta_b > 0.005 else "stable")
        elif brier is not None:
            trend_brier = "new_signal"

        grade_status[grade] = {
            "current_wr":    wr,
            "prev_wr":       prev_wr,
            "trend_wr":      trend_wr,
            "current_brier": brier,
            "prev_brier":    prev_brier,
            "trend_brier":   trend_brier,
            "n":             n,
            "target_wr":     targets["wr"],
            "target_brier":  targets["brier"],
            "wr_vs_target":  round(wr - targets["wr"], 3) if wr is not None else None,
            "brier_vs_target": round(brier - targets["brier"], 4) if brier is not None else None,
            "target_met":    (
                (wr is not None and wr >= targets["wr"]) and
                (brier is not None and brier <= targets["brier"])
            ),
            "reliable":      n >= 50,   # flag grades with thin samples
        }

    return {
        "current":       {**current, "grade_status": grade_status},
        "history":       parsed_runs[1:],   # previous runs for sparkline
        "targets":       SCOUT_TARGETS,
        "total_runs":    len(parsed_runs),
    }


def analyze_grade_drivers(
    db_path: str = _BETS_DB_PATH,
    grade: str = "B",
    since: Optional[str] = None,
) -> dict:
    """
    For legs with a specific scout grade, identify which sport/market cells
    differ most between wins and losses.

    Surfaces patterns like: 'Grade B MLB alt_spreads hit at 65% but
    Grade B NHL h2h hit at 45% — scout grades NHL matchups poorly'.
    """
    since = since or "2026-05-10"   # default: post-scout-activation
    con = _db_conn(db_path)
    try:
        rows = con.execute("""
            SELECT
                mb.sport,
                mbl.market_type,
                COUNT(*)                                                                  AS n,
                SUM(CASE WHEN mbl.leg_result = 'WIN'  THEN 1 ELSE 0 END)                AS wins,
                ROUND(AVG(CASE WHEN mbl.leg_result = 'WIN' THEN 1.0 ELSE 0.0 END), 3)  AS wr,
                ROUND(AVG(mbl.win_prob), 3)                                              AS avg_wp,
                ROUND(AVG(mbl.clv_cents), 3)                                            AS avg_clv,
                ROUND(AVG(mbl.injury_net_advantage), 3)                                 AS avg_injury
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.scout_grade = :grade
              AND mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mb.game_date >= :since
            GROUP BY mb.sport, mbl.market_type
            HAVING n >= 5
            ORDER BY wr DESC
        """, {"grade": grade.upper(), "since": since}).fetchall()
    finally:
        con.close()

    cells = [dict(r) for r in rows]

    # Flag outliers: cells more than 10pp above/below grade average
    if cells:
        all_wins  = sum(c["wins"] for c in cells)
        all_n     = sum(c["n"]    for c in cells)
        grade_avg = all_wins / all_n if all_n else 0.0
        for c in cells:
            delta = round(c["wr"] - grade_avg, 3)
            c["delta_vs_grade_avg"] = delta
            c["flag"] = (
                "outperforms" if delta >  0.10 else
                "underperforms" if delta < -0.10 else
                "inline"
            )
    else:
        grade_avg = None

    return {
        "grade":       grade.upper(),
        "since":       since,
        "grade_avg_wr": round(grade_avg, 3) if grade_avg is not None else None,
        "cells":       cells,
        "outliers": {
            "outperforming":  [c for c in cells if c.get("flag") == "outperforms"],
            "underperforming": [c for c in cells if c.get("flag") == "underperforms"],
        },
    }
