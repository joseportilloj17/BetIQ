"""
auto_settle.py — Phase 2B: Auto-settlement engine.

Flow:
  1. Pull all PLACED bets from DB
  2. For each, determine which sport_keys to check
  3. Fetch scores from OddsAPI (completed games, last N days)
  4. Match each bet's legs against completed game results
  5. Determine WIN / LOSS / PUSH
  6. Settle in DB, log to settle_log
  7. Optionally trigger model retrain if enough new data

Matching strategy:
  - Parlay: ALL legs must win → WIN, any loss → LOSS
  - Straight: single leg outcome
  - Leg matching: fuzzy team name match against home/away in scores
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time
import difflib
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

# ─── Retrain subprocess state (read by scheduler health endpoint) ─────────────
RETRAIN_STATE: dict = {
    "status":       "idle",   # idle | running | completed | completed_no_save | failed | killed_timeout
    "pid":          None,
    "sport":        None,     # None = combined, else sport label
    "started_ct":   None,
    "completed_ct": None,
    "log_path":     None,
    "error":        None,
}

_RETRAIN_PROC: Optional[subprocess.Popen] = None  # live handle so we can poll/kill

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_WORKER   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrain_worker.py")


def _now_ct_str() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Chicago")).isoformat()


def _trigger_retrain_subprocess(sport: Optional[str] = None) -> None:
    """
    Launch retrain in a background subprocess. Returns immediately.
    sport=None → combined model retrain; sport="mlb" → MLB sub-model, etc.
    """
    global _RETRAIN_PROC

    from zoneinfo import ZoneInfo
    ct = datetime.now(ZoneInfo("America/Chicago"))
    timestamp = ct.strftime("%Y%m%d_%H%M")
    label     = sport.lower() if sport else "combined"
    log_path  = os.path.join(_DATA_DIR, f"retrain_{label}_{timestamp}.log")

    cmd = [sys.executable, _WORKER]
    if sport:
        cmd += ["--sport", sport.lower()]
    else:
        cmd += ["--combined"]

    try:
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        _RETRAIN_PROC = proc
        RETRAIN_STATE.update({
            "status":       "running",
            "pid":          proc.pid,
            "sport":        sport,
            "started_ct":   ct.isoformat(),
            "completed_ct": None,
            "log_path":     log_path,
            "error":        None,
        })
        print(f"[AutoSettle] Retrain subprocess started: PID={proc.pid} sport={label} log={log_path}", flush=True)
    except Exception as exc:
        print(f"[AutoSettle] Failed to start retrain subprocess: {exc}", flush=True)
        RETRAIN_STATE.update({
            "status": "failed",
            "error":  str(exc),
        })


def check_retrain_status() -> None:
    """
    Poll the running retrain subprocess and update RETRAIN_STATE.
    Called by the scheduler watchdog every cycle while status='running'.
    """
    global _RETRAIN_PROC

    if RETRAIN_STATE.get("status") != "running":
        return

    pid = RETRAIN_STATE.get("pid")
    if not pid:
        return

    # CHECK 1: Is the process still alive?
    still_running = False
    try:
        os.kill(pid, 0)   # signal 0 = existence check
        still_running = True
    except (ProcessLookupError, OSError):
        still_running = False

    if still_running:
        # Check for timeout (8h)
        started_str = RETRAIN_STATE.get("started_ct")
        if started_str:
            try:
                started = datetime.fromisoformat(started_str).replace(tzinfo=timezone.utc)
                elapsed_hrs = (datetime.now(timezone.utc) - started).total_seconds() / 3600
                if elapsed_hrs > 8:
                    print(f"[AutoSettle] Retrain PID={pid} running {elapsed_hrs:.1f}h — killing", flush=True)
                    try:
                        if _RETRAIN_PROC:
                            _RETRAIN_PROC.kill()
                        else:
                            os.kill(pid, 9)
                    except Exception:
                        pass
                    RETRAIN_STATE["status"] = "killed_timeout"
                    RETRAIN_STATE["pid"]    = None
                    _RETRAIN_PROC = None
            except Exception:
                pass
        return  # still running normally

    # Process ended — determine success or failure
    _RETRAIN_PROC = None

    # CHECK 2: Did pkl files update since retrain started?
    sport = RETRAIN_STATE.get("sport")
    label = sport.lower() if sport else "mlb"
    pkl_path = os.path.join(_DATA_DIR, "submodels", f"{label}_ats_clf.pkl")
    started_str = RETRAIN_STATE.get("started_ct")
    pkl_updated = False
    if os.path.exists(pkl_path) and started_str:
        try:
            started_ts = datetime.fromisoformat(started_str).astimezone(timezone.utc).timestamp()
            pkl_updated = os.path.getmtime(pkl_path) > started_ts
        except Exception:
            pass

    if pkl_updated:
        RETRAIN_STATE["status"]       = "completed"
        RETRAIN_STATE["completed_ct"] = _now_ct_str()
        RETRAIN_STATE["pid"]          = None
        print(f"[AutoSettle] Retrain completed — pkl updated", flush=True)
        return

    # CHECK 3: Scan log for error patterns
    log_path = RETRAIN_STATE.get("log_path")
    error_found = None
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path) as f:
                content = f.read()
            for pattern in ("Traceback", "MemoryError", "ModuleNotFoundError",
                            "Exception", "Error", "Killed"):
                if pattern in content:
                    error_found = pattern
                    break
            if error_found:
                tail = "\n".join(content.splitlines()[-5:])
                print(f"[AutoSettle] Retrain failed — {error_found}\n{tail}", flush=True)
        except Exception:
            pass

    if error_found:
        RETRAIN_STATE["status"] = "failed"
        RETRAIN_STATE["error"]  = error_found
    else:
        # Ended cleanly but pkl not written (AUC gate or no new data)
        RETRAIN_STATE["status"]       = "completed_no_save"
        RETRAIN_STATE["completed_ct"] = _now_ct_str()
        print(f"[AutoSettle] Retrain ended — no pkl update (AUC gate or no new data)", flush=True)
    RETRAIN_STATE["pid"] = None
from database import Bet, SettleLog, SchedulerRun, SessionLocal

from odds_api import get_scores, ALL_SPORT_KEYS, ODDS_API_KEY
from parlay_builder import SPORT_LABEL   # sport_key → label


# ─── Sport key routing ────────────────────────────────────────────────────────

# Map our stored sport strings → OddsAPI sport keys to check
SPORT_TO_KEYS = {
    "Basketball":        ["basketball_nba", "basketball_ncaab"],
    "NBA":               ["basketball_nba"],
    "NCAAB":             ["basketball_ncaab"],
    "American Football": ["americanfootball_nfl", "americanfootball_ncaaf"],
    "NFL":               ["americanfootball_nfl"],
    "Baseball":          ["baseball_mlb"],
    "MLB":               ["baseball_mlb"],
    "Soccer":            ["soccer_epl", "soccer_spain_la_liga", "soccer_usa_mls",
                          "soccer_uefa_champs_league", "soccer_germany_bundesliga",
                          "soccer_france_ligue_one", "soccer_italy_serie_a",
                          "soccer_uefa_europa_league"],
    "EPL":               ["soccer_epl"],
    "La Liga":           ["soccer_spain_la_liga"],
    "UCL":               ["soccer_uefa_champs_league"],
    "Bundesliga":        ["soccer_germany_bundesliga"],
    "Ligue 1":           ["soccer_france_ligue_one"],
    "Serie A":           ["soccer_italy_serie_a"],
    "MLS":               ["soccer_usa_mls"],
    "Europa League":     ["soccer_uefa_europa_league"],
    "Ice Hockey":        ["icehockey_nhl"],
    "NHL":               ["icehockey_nhl"],
    "Tennis":            ["tennis_atp_french_open", "tennis_wta_french_open"],
    "MMA":               ["mma_mixed_martial_arts"],
}


# ─── Team name normaliser ────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _team_match(pick: str, home: str, away: str, threshold: float = 0.55) -> Optional[str]:
    """
    Returns 'home', 'away', or None.
    Uses difflib ratio for fuzzy matching to handle
    'NY Knicks' vs 'New York Knicks', city-only names, etc.
    """
    p = _normalise(pick)
    h = _normalise(home)
    a = _normalise(away)

    # Exact or substring match first (fastest path)
    if p == h or p in h or h in p:
        return "home"
    if p == a or p in a or a in p:
        return "away"

    # Fuzzy ratio
    h_ratio = difflib.SequenceMatcher(None, p, h).ratio()
    a_ratio = difflib.SequenceMatcher(None, p, a).ratio()

    if h_ratio >= threshold and h_ratio > a_ratio:
        return "home"
    if a_ratio >= threshold and a_ratio > h_ratio:
        return "away"

    return None


# ─── Score cache ──────────────────────────────────────────────────────────────

_score_cache: dict[str, tuple[list, datetime]] = {}
CACHE_TTL_MINUTES = 15


def _fetch_scores_cached(sport_key: str, days_from: int = 3) -> list[dict]:
    """Cache score fetches per sport_key to avoid hammering the API."""
    now = datetime.utcnow()
    cached = _score_cache.get(sport_key)
    if cached and (now - cached[1]).total_seconds() < CACHE_TTL_MINUTES * 60:
        return cached[0]
    scores = get_scores(sport_key, days_from=days_from)
    _score_cache[sport_key] = (scores, now)
    return scores


# ─── Leg outcome resolver ─────────────────────────────────────────────────────

def _resolve_leg(leg_desc: str, scores: list[dict]) -> Optional[str]:
    """
    Given a pipe-split leg description string and a list of OddsAPI score objects,
    return 'WIN', 'LOSS', 'PUSH', or None (not found / not yet completed).

    Handles:
      - Moneyline:  "Celtics (Moneyline)"  →  pick wins if their score > opponent
      - Run Line / Spread: "Yankees +1.5 (Spread)"  →  cover check
      - Total:  "Over 220.5 (Total)"  →  combined score check
    """
    desc = leg_desc.strip()
    desc_lower = desc.lower()

    for game in scores:
        if not game.get("completed"):
            continue

        home = game.get("home_team", "")
        away = game.get("away_team", "")
        scores_list = game.get("scores") or []
        score_map: dict[str, int] = {}
        for sc in scores_list:
            try:
                score_map[sc["name"]] = int(sc["score"])
            except (KeyError, ValueError):
                pass

        if len(score_map) < 2:
            continue

        home_score = score_map.get(home, 0)
        away_score = score_map.get(away, 0)
        total_score = home_score + away_score

        # ── Moneyline ──
        if "moneyline" in desc_lower:
            # Extract pick = everything before "(Moneyline)"
            pick = re.sub(r"\(moneyline\)", "", desc, flags=re.IGNORECASE).strip()
            side = _team_match(pick, home, away)
            if side is None:
                continue
            if side == "home":
                result = "WIN" if home_score > away_score else ("PUSH" if home_score == away_score else "LOSS")
            else:
                result = "WIN" if away_score > home_score else ("PUSH" if away_score == home_score else "LOSS")
            return result

        # ── Spread / Run Line ──
        spread_match = re.search(r"([+-]?\d+\.?\d*)\s*\(spread\|run line\|point spread\)", desc_lower)
        if not spread_match:
            # Try simpler pattern: "Team +1.5 (Run Line)" or "Team -3.5 (Spread)"
            spread_match = re.search(r"([+-]\d+\.?\d*)", desc)
        if spread_match and ("spread" in desc_lower or "run line" in desc_lower):
            spread_val = float(spread_match.group(1))
            # Determine which team has the spread applied
            pick_raw = re.split(r"[+-]\d", desc)[0].strip()
            side = _team_match(pick_raw, home, away)
            if side is None:
                continue
            if side == "home":
                covered_margin = home_score - away_score + spread_val
            else:
                covered_margin = away_score - home_score + spread_val
            if abs(covered_margin) < 0.01:
                return "PUSH"
            return "WIN" if covered_margin > 0 else "LOSS"

        # ── Total (Over/Under) ──
        total_match = re.search(r"(over|under)\s+(\d+\.?\d*)", desc_lower)
        if total_match:
            direction = total_match.group(1)
            line = float(total_match.group(2))
            if abs(total_score - line) < 0.01:
                return "PUSH"
            if direction == "over":
                return "WIN" if total_score > line else "LOSS"
            else:
                return "WIN" if total_score < line else "LOSS"

    return None   # no matching completed game found


# ─── Parlay outcome resolver ──────────────────────────────────────────────────

def _resolve_parlay(bet: Bet, scores_by_key: dict[str, list]) -> tuple[str, str]:
    """
    Returns (outcome, notes):
      outcome: 'WIN' | 'LOSS' | 'PUSH' | 'PENDING'
    """
    if not bet.bet_info:
        return "PENDING", "No bet_info to match legs against"

    legs = [l.strip() for l in bet.bet_info.split("|") if l.strip()]
    all_scores = []
    for scores in scores_by_key.values():
        all_scores.extend(scores)

    results = []
    unresolved = []
    for leg in legs:
        outcome = _resolve_leg(leg, all_scores)
        if outcome is None:
            unresolved.append(leg)
        else:
            results.append(outcome)

    if unresolved:
        return "PENDING", f"Could not resolve {len(unresolved)}/{len(legs)} legs: {unresolved[:2]}"

    if "LOSS" in results:
        return "LOSS", f"Lost on leg(s): {[l for l,r in zip(legs,results) if r=='LOSS'][:2]}"
    if all(r == "PUSH" for r in results):
        return "PUSH", "All legs pushed"
    return "WIN", f"All {len(results)} legs won"


# ─── Retrospective cash-out verdict ──────────────────────────────────────────

def _backfill_cash_out_decisions(db: Session) -> None:
    """
    Two passes run after every settle cycle:

    Pass 1 — Cashed-out bets:
      For every bet where cashed_out=1 and cash_out_vs_final_decision is NULL,
      check the SettleLog to determine what *would have happened* had the user
      held.  Writes "cashed_wisely" or "cashed_unnecessarily".

    Pass 2 — Held bets with CASH_OUT_RECOMMENDED in their log:
      For every settled bet (cashed_out=0, status SETTLED_WIN/LOSS) whose
      cash_out_offers_log contains a CASH_OUT_RECOMMENDED entry without a
      final_outcome annotation, write final_outcome = "won" or "lost" into
      each such entry.  This powers the "held through red" accuracy metric.
    """
    import json as _json
    from sqlalchemy import text as _text

    # ── Pass 1: cashed-out bets ───────────────────────────────────────────────
    try:
        cashed = db.execute(_text("""
            SELECT id, cash_out_amount, amount, odds
            FROM bets
            WHERE cashed_out = 1
              AND cash_out_vs_final_decision IS NULL
        """)).fetchall()
    except Exception:
        return   # columns not yet migrated

    for row in cashed:
        bet_id          = row[0]
        cash_out_amount = float(row[1] or 0.0)
        original_stake  = float(row[2] or 0.0)
        original_odds   = float(row[3] or 1.0)

        settle = (
            db.query(SettleLog)
            .filter(
                SettleLog.bet_id == bet_id,
                SettleLog.result.in_(["WIN", "LOSS", "PUSH"]),
            )
            .order_by(SettleLog.settled_at.desc())
            .first()
        )
        if settle is None:
            continue

        verdict = "cashed_unnecessarily" if settle.result == "WIN" else "cashed_wisely"
        try:
            db.execute(_text(
                "UPDATE bets SET cash_out_vs_final_decision = :v WHERE id = :id"
            ), {"v": verdict, "id": bet_id})
        except Exception:
            pass

    # ── Pass 2: settled held bets — annotate CASH_OUT_RECOMMENDED log entries ─
    try:
        held_settled = db.execute(_text("""
            SELECT id, status, cash_out_offers_log
            FROM bets
            WHERE cashed_out = 0
              AND status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND cash_out_offers_log IS NOT NULL
        """)).fetchall()
    except Exception:
        held_settled = []

    for row in held_settled:
        bet_id     = row[0]
        status     = row[1]
        log_raw    = row[2]

        try:
            log_list = _json.loads(log_raw)
        except Exception:
            continue

        # Only touch entries that are CASH_OUT_RECOMMENDED and lack final_outcome
        needs_update = any(
            e.get("recommendation") == "CASH_OUT_RECOMMENDED"
            and "final_outcome" not in e
            for e in log_list
        )
        if not needs_update:
            continue

        final_outcome = "won" if status == "SETTLED_WIN" else "lost"
        for entry in log_list:
            if (entry.get("recommendation") == "CASH_OUT_RECOMMENDED"
                    and "final_outcome" not in entry):
                entry["final_outcome"] = final_outcome

        try:
            db.execute(_text(
                "UPDATE bets SET cash_out_offers_log = :v WHERE id = :id"
            ), {"v": _json.dumps(log_list), "id": bet_id})
        except Exception:
            pass

    db.commit()


# ─── Main settle runner ───────────────────────────────────────────────────────

def _new_settled_since_last_train(db: Session) -> int:
    """
    Count personal bets settled since the last model training run.
    Returns 0 if no training run exists yet.
    """
    from database import ModelRun
    last_run = db.query(ModelRun).order_by(ModelRun.id.desc()).first()
    if last_run is None:
        return 0
    last_train_time = last_run.run_at
    count = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS", "SETTLED_PUSH"]),
        Bet.time_settled > last_train_time,
        Bet.is_mock.is_(False),
    ).count()
    return count


# Sub-model sports and their retrain thresholds (lower than the combined 50
# because sub-models have less data to begin with)
_SUBMODEL_RETRAIN_SPORTS: dict[str, int] = {
    "NHL": 30,
    "MLB": 30,
}
# Sport label strings as they appear in Bet.sports
_SPORT_BET_LABELS: dict[str, list[str]] = {
    "NHL": ["NHL", "Ice Hockey"],
    "MLB": ["MLB", "Baseball"],
}


def _new_settled_sport(db: Session, sport: str, since) -> int:
    """Count personal bets for a given sport settled since a timestamp."""
    labels = _SPORT_BET_LABELS.get(sport, [sport])
    from sqlalchemy import or_
    clauses = [Bet.sports.contains(label) for label in labels]
    count = db.query(Bet).filter(
        or_(*clauses),
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS", "SETTLED_PUSH"]),
        Bet.time_settled > since,
        Bet.is_mock.is_(False),
    ).count()
    return count


def _settle_scout_props(db: Session, days_back: int = 3) -> int:
    """
    Mark actual_hit on scouted_props rows where the outcome is now known.

    Strategy:
      - Find scouted_props where actual_hit IS NULL and scout_date >= today - days_back
      - For each row, look up a settled mock_bet_leg that matches game_id + market_type
        (or falls back to matching team names + market_type)
      - If found and leg_result = WIN → actual_hit = 1 (for OVER/home side)
                                  LOSS → actual_hit = 0
      - Updates actual_outcome_value from resolved_home/away scores where available

    Returns count of props settled.
    """
    from sqlalchemy import text
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    settled = 0

    try:
        # Fetch unsettled scouted props within window
        rows = db.execute(text("""
            SELECT id, game_id, market_type, side, threshold, sport, scout_date
            FROM   scouted_props
            WHERE  actual_hit IS NULL
            AND    scout_date >= :cutoff
        """), {"cutoff": cutoff}).fetchall()

        for (prop_id, game_id, market_type, side, threshold, sport, scout_date) in rows:
            # Try to find a settled mock_bet_leg for this game/market
            leg_row = None

            # Match on fixture_id (= game_id) + market_type
            if game_id:
                leg_row = db.execute(text("""
                    SELECT mbl.leg_result,
                           mbl.resolved_home_score, mbl.resolved_away_score
                    FROM   mock_bet_legs mbl
                    JOIN   mock_bets mb ON mb.id = mbl.mock_bet_id
                    WHERE  mbl.fixture_id  = :gid
                    AND    mbl.market_type = :mt
                    AND    mbl.leg_result  IS NOT NULL
                    AND    mbl.leg_result  != 'PENDING'
                    ORDER  BY mb.created_at DESC
                    LIMIT  1
                """), {"gid": game_id, "mt": market_type}).fetchone()

            if not leg_row:
                continue   # no settled game data yet

            leg_result, home_score, away_score = leg_row

            # Determine hit: WIN → 1, LOSS → 0, PUSH → skip
            if leg_result == "PUSH":
                continue
            actual_hit = 1 if leg_result == "WIN" else 0

            # Infer actual outcome value from scores where meaningful
            actual_value = None
            if home_score is not None and away_score is not None:
                if market_type == "totals":
                    actual_value = home_score + away_score
                elif market_type in ("h2h", "spread"):
                    actual_value = home_score - away_score if side == "home" else away_score - home_score

            db.execute(text("""
                UPDATE scouted_props
                SET    actual_hit           = :hit,
                       actual_outcome_value = :val
                WHERE  id = :pid
            """), {"hit": actual_hit, "val": actual_value, "pid": prop_id})
            settled += 1

        db.commit()
        if settled:
            print(f"[AutoSettle] Settled {settled} scouted props")
        return settled

    except Exception as exc:
        print(f"[AutoSettle] _settle_scout_props error: {exc}")
        return 0


def run_auto_settle(
    db: Session,
    days_back: int = 3,
    auto_retrain: bool = True,
    retrain_threshold: int = 50,   # retrain when 50+ new personal bets settled since last train
) -> dict:
    """
    Main entry point. Checks all PLACED bets, settles what it can.
    Returns a summary dict.
    """
    t_start = time.time()
    placed_bets = db.query(Bet).filter(Bet.status == "PLACED").all()

    if not placed_bets:
        return {
            "bets_checked": 0,
            "bets_settled": 0,
            "pending":      0,
            "errors":       0,
            "retrained":    False,
            "message":      "No PLACED bets to settle.",
        }

    # Determine which sport keys we need
    needed_keys: set[str] = set()
    for bet in placed_bets:
        sport_str = (bet.sports or "").split("|")[0].strip()
        for key in SPORT_TO_KEYS.get(sport_str, ALL_SPORT_KEYS[:6]):
            needed_keys.add(key)

    # Fetch scores (cached)
    scores_by_key: dict[str, list] = {}
    for key in needed_keys:
        scores_by_key[key] = _fetch_scores_cached(key, days_from=days_back)

    settled_count = errors = pending = 0
    settle_logs: list[SettleLog] = []

    for bet in placed_bets:
        try:
            sport_str = (bet.sports or "").split("|")[0].strip()
            relevant_keys = SPORT_TO_KEYS.get(sport_str, list(needed_keys))
            relevant_scores = {k: scores_by_key.get(k, []) for k in relevant_keys}

            if bet.bet_type == "straight" and bet.legs == 1:
                all_scores = [s for ss in relevant_scores.values() for s in ss]
                info = (bet.bet_info or "").split("|")[0].strip()
                outcome = _resolve_leg(info, all_scores)
                if outcome is None:
                    outcome, notes = "PENDING", "Game not yet completed or not found"
                else:
                    notes = f"Resolved from scores API"
            else:
                outcome, notes = _resolve_parlay(bet, relevant_scores)

            if outcome == "PENDING":
                pending += 1
                settle_logs.append(SettleLog(
                    bet_id=bet.id, sport_key=sport_str,
                    result="SKIP", method="auto", notes=notes
                ))
                continue

            # Settle the bet
            bet.status       = f"SETTLED_{outcome}" if outcome in ("WIN","LOSS") else "SETTLED_PUSH"
            bet.time_settled = datetime.utcnow()

            if outcome == "WIN":
                bet.profit = round((bet.odds - 1) * bet.amount, 2)
            elif outcome == "LOSS":
                bet.profit = -bet.amount
            else:  # PUSH
                bet.profit = 0.0

            settle_logs.append(SettleLog(
                bet_id=bet.id, sport_key=sport_str,
                result=outcome, profit=bet.profit,
                method="auto", notes=notes
            ))
            settled_count += 1

        except Exception as e:
            errors += 1
            settle_logs.append(SettleLog(
                bet_id=bet.id, result="ERROR", method="auto",
                notes=str(e)[:300]
            ))

    for log in settle_logs:
        db.add(log)
    db.commit()

    # ── Retrospective cash-out analysis ───────────────────────────────────────
    # For every bet that was cashed out AND has now settled via the score lookup
    # above, compute whether the user made the right call.
    _backfill_cash_out_decisions(db)

    # ── Personal edge profile continuous learning ─────────────────────────────
    # After each settlement cycle, refresh personal_edge_profile so margin grades
    # and win rates stay current with the latest resolved bets. This is lightweight
    # (re-aggregates from existing tables) and runs only when bets were settled.
    if settled_count > 0:
        try:
            from personal_edge_profile import refresh_personal_edge_profiles
            _pep_result = refresh_personal_edge_profiles()
            print(f"[AutoSettle] Personal edge profile refreshed after {settled_count} settlements: "
                  f"{_pep_result.get('merged_profiles')} profiles, "
                  f"{_pep_result.get('deleted_stale', 0)} stale removed")
        except Exception as _pep_err:
            print(f"[AutoSettle] Personal edge profile refresh error (non-fatal): {_pep_err}")

    # ── Scout prop settlement ─────────────────────────────────────────────────
    # After bets settle, mark scouted_props.actual_hit where game outcomes are known.
    # Runs as a best-effort pass — failure never blocks the settle cycle.
    if settled_count > 0:
        try:
            _settle_scout_props(db, days_back=days_back)
        except Exception as _sp_err:
            print(f"[AutoSettle] Scout prop settlement error (non-fatal): {_sp_err}")

    # Auto-retrain — launched as a background subprocess so the server never blocks
    retrained = False
    new_accuracy = None
    new_since_train = _new_settled_since_last_train(db)
    submodel_retrains: dict[str, str] = {}

    if auto_retrain and RETRAIN_STATE.get("status") not in ("running",):
        if new_since_train >= retrain_threshold:
            _trigger_retrain_subprocess(sport=None)  # combined model
            retrained = True
            print(f"[AutoSettle] Combined retrain queued — {new_since_train} new bets since last train")
        else:
            # Per-sport sub-model auto-retrain (lower threshold: 30 bets each)
            from database import ModelRun
            last_run = db.query(ModelRun).order_by(ModelRun.id.desc()).first()
            since_ts = last_run.run_at if last_run else datetime(2000, 1, 1, tzinfo=timezone.utc)
            for sport, threshold in _SUBMODEL_RETRAIN_SPORTS.items():
                n_sport = _new_settled_sport(db, sport, since_ts)
                if n_sport >= threshold:
                    _trigger_retrain_subprocess(sport=sport)
                    submodel_retrains[sport] = "queued"
                    print(f"[AutoSettle] Sub-model retrain queued: {sport} ({n_sport} new bets)")
                    break  # one retrain at a time

    duration = round(time.time() - t_start, 2)

    # Log this scheduler run
    run = SchedulerRun(
        bets_checked  = len(placed_bets),
        bets_settled  = settled_count,
        retrained     = retrained,
        new_accuracy  = new_accuracy,
        errors        = str(errors) if errors else None,
        duration_secs = duration,
    )
    db.add(run)
    db.commit()

    return {
        "bets_checked":         len(placed_bets),
        "bets_settled":         settled_count,
        "pending":              pending,
        "errors":               errors,
        "retrained":            retrained,
        "new_accuracy":         new_accuracy,
        "new_bets_since_train": new_since_train,
        "retrain_threshold":    retrain_threshold,
        "submodel_retrains":    submodel_retrains,
        "duration_secs":        duration,
        "message":              f"Settled {settled_count}/{len(placed_bets)} bets. "
                                f"{pending} still pending. "
                                + (f"Model retrained ({new_accuracy}% acc)." if retrained else
                                   f"{new_since_train}/{retrain_threshold} new bets toward auto-retrain.")
                                + (f" Sub-model retrains: {submodel_retrains}" if submodel_retrains else ""),
    }


# ─── Settle history ──────────────────────────────────────────────────────────

def get_settle_log(db: Session, limit: int = 50) -> list[dict]:
    logs = db.query(SettleLog).order_by(SettleLog.settled_at.desc()).limit(limit).all()
    return [{
        "id":         l.id,
        "bet_id":     l.bet_id,
        "sport_key":  l.sport_key,
        "result":     l.result,
        "profit":     l.profit,
        "method":     l.method,
        "notes":      l.notes,
        "settled_at": l.settled_at.isoformat() if l.settled_at else None,
        "retrained":  l.retrained,
    } for l in logs]


def get_scheduler_history(db: Session, limit: int = 20) -> list[dict]:
    runs = db.query(SchedulerRun).order_by(SchedulerRun.run_at.desc()).limit(limit).all()
    return [{
        "id":            r.id,
        "run_at":        r.run_at.isoformat() if r.run_at else None,
        "bets_checked":  r.bets_checked,
        "bets_settled":  r.bets_settled,
        "retrained":     r.retrained,
        "new_accuracy":  r.new_accuracy,
        "errors":        r.errors,
        "duration_secs": r.duration_secs,
    } for r in runs]
