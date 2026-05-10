"""
leg_quality.py — Leg Quality Score (LQS) system.

Scores a candidate leg 0–100 before placement using four weighted components:

  A. Historical accuracy   (40%) — avg accuracy_delta + win_rate from resolved bet_legs
  B. Model confidence      (30%) — recommender win_prob, model identity bonus
  C. Market type quality   (20%) — calibrated from unbiased resolver win rates
  D. Odds value            (10%) — edge_pp vs breakeven

Grade:  A ≥ 80  |  B 65–79  |  C 50–64  |  D < 50
Action: ADD ≥ 65  |  CONSIDER 50–64  |  AVOID < 50

Calibration baseline (resolver output 2026-04-21, unbiased legs):
  Moneyline  : 92.5%  →  C-score 90
  Spread     : 92.5%  →  C-score 85 (line variance penalty)
  Total      : 61.9%  →  C-score 55 (weakest market)
  Player Prop: 95.8% biased → C-score 65 (conservative until more unbiased data)
  Alt Spread : starts 70, penalised by line size
"""
from __future__ import annotations

import os
import re as _re
import sqlite3
import statistics
import time as _time
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text as sqla_text

# ── Component-A result cache ──────────────────────────────────────────────────
# Caches (team, sport, market, line_bucket) → (score, timestamp).
# TTL = 5 minutes so the cache stays warm during a generate_mock_bets call
# (which runs 4 sport passes in parallel, each scoring many identical alt-lines)
# but refreshes between scheduler runs.
_COMP_A_CACHE: dict = {}
_COMP_A_TTL = 300  # 5 minutes

# ── Historical DB (read-only, used for Component E matchup context) ───────────
_HIST_DB_LQ = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

def _hist_conn_lq() -> sqlite3.Connection:
    conn = sqlite3.connect(_HIST_DB_LQ)
    conn.row_factory = sqlite3.Row
    return conn


def query_historical(sql: str, params: list) -> list:
    """Execute a read-only query against historical.db and return list of row dicts."""
    try:
        conn = sqlite3.connect(_HIST_DB_LQ)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


# Module-level cache for Component E results (keyed on inputs, cleared on server restart)
_e_cache: dict = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Component C base scores (calibrated to unbiased resolver data)
_MARKET_C_SCORE: dict[str, float] = {
    "Moneyline":   90.0,
    "Spread":      85.0,
    "Total":       55.0,
    "Player Prop": 65.0,
    "Alt Spread":  70.0,
    "Alt Total":   60.0,
    "Other":       50.0,
}

# Sub-model bonus for component B
_SUBMODEL_BONUS: dict[str, float] = {
    "mlb_ats_v1":  10.0,
    "nhl_ats_v1":  10.0,
    "combined_v1":  0.0,
}

# Resolution sources considered unbiased (game result from DB or pitcher logs)
_UNBIASED_SOURCES = {"historical_db", "pitcher_logs"}

# Probability boost per unit of line shift (percentage points per unit), by sport + market
_PP_PER_UNIT: dict[str, dict[str, float]] = {
    "MLB": {"Total": 8.0,  "Spread": 3.0},   # 8pp per run for MLB totals
    "NBA": {"Total": 1.0,  "Spread": 3.0},   # 1pp per point for NBA totals
    "NHL": {"Total": 10.0, "Spread": 3.0},   # 10pp per goal for NHL totals
    "NFL": {"Total": 1.0,  "Spread": 3.0},   # 1pp per point for NFL totals
}
_PP_DEFAULT = {"Total": 5.0, "Spread": 3.0}  # fallback for unknown sport
_MAX_JUICE_AMERICAN: int = -250              # never recommend pivot past this

# Sports that support Component E (have games data in historical.db)
_E_SUPPORTED_SPORTS = {"MLB", "NHL", "NBA"}

# Full name → 3-letter abbreviation for NHL and NBA
# (historical.db stores abbreviations; fixtures store full names)
_NHL_TO_ABBR: dict[str, str] = {
    "Anaheim Ducks":         "ANA", "Boston Bruins":         "BOS",
    "Buffalo Sabres":        "BUF", "Calgary Flames":        "CGY",
    "Carolina Hurricanes":   "CAR", "Chicago Blackhawks":    "CHI",
    "Colorado Avalanche":    "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars":          "DAL", "Detroit Red Wings":     "DET",
    "Edmonton Oilers":       "EDM", "Florida Panthers":      "FLA",
    "Los Angeles Kings":     "LAK", "Minnesota Wild":        "MIN",
    "Montréal Canadiens":    "MTL", "Montreal Canadiens":    "MTL",
    "Nashville Predators":   "NSH", "New Jersey Devils":     "NJD",
    "New York Islanders":    "NYI", "New York Rangers":      "NYR",
    "Ottawa Senators":       "OTT", "Philadelphia Flyers":   "PHI",
    "Pittsburgh Penguins":   "PIT", "San Jose Sharks":       "SJS",
    "Seattle Kraken":        "SEA", "St Louis Blues":        "STL",
    "St. Louis Blues":       "STL", "Tampa Bay Lightning":   "TBL",
    "Toronto Maple Leafs":   "TOR", "Utah Mammoth":          "UTA",
    "Utah Hockey Club":      "UTA", "Vancouver Canucks":     "VAN",
    "Vegas Golden Knights":  "VGK", "Washington Capitals":   "WSH",
    "Winnipeg Jets":         "WPG", "Arizona Coyotes":       "ARI",
}
_NBA_TO_ABBR: dict[str, str] = {
    "Atlanta Hawks":          "ATL", "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN", "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI", "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL", "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET", "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU", "Indiana Pacers":         "IND",
    "Los Angeles Clippers":   "LAC", "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM", "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP", "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC", "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI", "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS", "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA", "Washington Wizards":     "WAS",
}


def _to_hist_team(name: str, sport: str) -> str:
    """
    Normalise a team name to the format used in historical.db games table.
    MLB: stored as full name → pass through.
    NHL/NBA: stored as 3-letter abbreviation → map via lookup dict.
    """
    if not name:
        return name
    sp = (sport or "").upper()
    if sp == "NHL":
        return _NHL_TO_ABBR.get(name, name)
    if sp == "NBA":
        return _NBA_TO_ABBR.get(name, name)
    return name   # MLB: full names match


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MARKET TYPE NORMALISER
# ═══════════════════════════════════════════════════════════════════════════════

def _canonical_market(market_type: str) -> str:
    """Normalise raw market type string to a canonical key for _MARKET_C_SCORE."""
    mt = (market_type or "").strip().lower()
    if not mt:
        return "Other"
    if "alt spread" in mt or "alternate spread" in mt or "alt run line" in mt or "alt puck line" in mt:
        return "Alt Spread"
    if "alt total" in mt or "alternate total" in mt:
        return "Alt Total"
    if "player prop" in mt or mt == "prop":
        return "Player Prop"
    if mt in ("moneyline", "ml", "h2h", "team to win", "match winner"):
        return "Moneyline"
    if "spread" in mt or "handicap" in mt or "run line" in mt or "puck line" in mt:
        return "Spread"
    if "total" in mt or "over" in mt or "under" in mt or "o/u" in mt:
        return "Total"
    return "Other"


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. ODDS / PROBABILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _dec_to_american(dec: float) -> int:
    """Convert decimal odds to American odds (integer)."""
    if dec >= 2.0:
        return round((dec - 1.0) * 100)
    else:
        return round(-100.0 / (dec - 1.0))


def _american_to_dec(am: int) -> float:
    """Convert American odds to decimal odds."""
    if am > 0:
        return 1.0 + am / 100.0
    else:
        return 1.0 + 100.0 / abs(am)


def _ev_dollar(prob: float, dec_odds: float, stake: float = 10.0) -> float:
    """EV on a given stake: E[profit] = prob*(dec-1)*stake - (1-prob)*stake."""
    return (prob * (dec_odds - 1.0) - (1.0 - prob)) * stake


def _pp_per_unit_shift(sport: str, market_type: str) -> float:
    """Return probability-shift per 1-unit line move as a 0-1 fraction."""
    mt  = _canonical_market(market_type)
    sp  = (sport or "").upper()
    tbl = _PP_PER_UNIT.get(sp, _PP_DEFAULT)
    return tbl.get(mt, 3.0) / 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL COMPONENT SCORERS
# ═══════════════════════════════════════════════════════════════════════════════

def _score_A(
    avg_delta: Optional[float],
    win_rate:  Optional[float],
    sample_size: int,
) -> tuple[float, list[str]]:
    """Component A: Historical accuracy (0–100)."""
    warnings: list[str] = []
    if sample_size < 5:
        warnings.append("⚠️ No historical data for this profile")
        return 50.0, warnings

    # Base from win rate
    if win_rate is None:
        base = 50.0
    elif win_rate >= 0.70:
        base = 80.0 + (win_rate - 0.70) / 0.30 * 20.0
    elif win_rate >= 0.55:
        base = 60.0 + (win_rate - 0.55) / 0.15 * 20.0
    elif win_rate >= 0.40:
        base = 40.0 + (win_rate - 0.40) / 0.15 * 20.0
    else:
        base = max(0.0, win_rate / 0.40 * 40.0)

    # Delta bonus / penalty
    delta_bonus = 0.0
    if avg_delta is not None:
        if avg_delta > 2.0:
            delta_bonus = 10.0
            warnings.append(f"✅ Strong accuracy history — avg delta +{avg_delta:.1f}")
        elif avg_delta >= 0.5:
            delta_bonus = 5.0
        elif avg_delta > -0.5:
            delta_bonus = 0.0
        elif avg_delta > -1.5:
            delta_bonus = -5.0
        else:
            delta_bonus = -10.0
            warnings.append(f"⚠️ Consistently missing — avg delta {avg_delta:.1f}")

    return min(100.0, max(0.0, base + delta_bonus)), warnings


# ─── Component A v3: personal edge profile as primary source ─────────────────

def compute_component_a(
    team_or_player: Optional[str],
    sport:          str,
    market_type:    str,
    line:           Optional[float],
    db:             Session,
) -> float:
    """
    Compute Component A score (0–100) anchored on personal historical win rates.

    Data sources (in priority order):
      1. personal_edge_profile — Jose's actual WR for sport/market/line_bucket.
         Built from leg_quality_profiles (Pikkit imports) + mock_bet_legs sims.
         This is the ground truth: e.g. Soccer ML = 82%, NBA ML = 89%.

      2. Team-specific recency overlay — settled mock_bet_legs and
         leg_historical_resolution for the specific team_or_player.
         Blended at 70% profile / 30% team-specific when n_team >= 5.

    Scoring:
      base_score = personal_wr × 100
      delta_adj  = clamp(mean_delta × 5, -15, +15)
      consist_adj = clamp((2 - std_delta) × 2, -8, +5) when std_delta available
      confidence  = min(1.0, sample_size / 30)
      score = 50 + confidence × (base + delta_adj + consist_adj − 50)

    Returns 50.0 (neutral) when no profile and fewer than 5 team-specific rows.
    Does NOT use inferred_parlay_win data.
    """
    from datetime import date as _date

    # ── Cache check ───────────────────────────────────────────────────────────
    # Build a cache key from the slow-changing inputs (team, sport, market, line).
    # team_or_player varies per fixture; sport/market/line_bucket determine the
    # profile lookup.  DB data doesn't change within a 5-minute window, so caching
    # eliminates the repeated LIKE queries during ALE (730 alt-lines per soccer run).
    try:
        from personal_edge_profile import (
            classify_line_bucket as _clf_bucket_c,
            normalize_sport      as _norm_sport_c,
            normalize_market     as _norm_market_c,
        )
        _ck_bucket = _clf_bucket_c(_norm_market_c(market_type or ""), "", line)
        _cache_key = (team_or_player or "", sport or "", market_type or "", _ck_bucket)
    except Exception:
        _cache_key = None

    if _cache_key is not None:
        _cached = _COMP_A_CACHE.get(_cache_key)
        if _cached is not None:
            _score, _ts = _cached
            if _time.time() - _ts < _COMP_A_TTL:
                return _score
        # Evict stale entries periodically (keep cache small)
        if len(_COMP_A_CACHE) > 2000:
            _now_t = _time.time()
            _COMP_A_CACHE.clear()  # simple full eviction when oversized

    # ── Step 1: personal_edge_profile lookup (sport / market / line_bucket) ──
    try:
        from personal_edge_profile import (
            classify_line_bucket  as _clf_bucket,
            lookup_personal_profile as _lookup,
            normalize_sport       as _norm_sport,
            normalize_market      as _norm_market,
            get_contextual_wr_adjustment as _ctx_adj,
        )
        sp_norm  = _norm_sport(sport or "")
        mt_norm  = _norm_market(market_type or "")
        bucket   = _clf_bucket(mt_norm, "", line)
        profile  = _lookup(sp_norm, mt_norm, bucket)
    except Exception:
        profile  = None

    profile_score: Optional[float] = None
    if profile and profile["sample_size"] >= 5 and profile["personal_wr"] is not None:
        personal_wr = profile["personal_wr"]
        mean_d      = profile["mean_delta"] or 0.0
        std_d       = profile["std_delta"]

        # Step 2: base score from personal win rate
        base = personal_wr * 100.0

        # Step 3: margin quality adjustments
        delta_adj   = max(-15.0, min(15.0, mean_d * 5.0))
        consist_adj = 0.0
        if std_d is not None:
            consist_adj = max(-8.0, min(5.0, (2.0 - std_d) * 2.0))

        # Step 4: sample-size confidence weight (toward neutral=50 when data is sparse)
        confidence = min(1.0, profile["sample_size"] / 30.0)
        raw = base + delta_adj + consist_adj
        profile_score = 50.0 + confidence * (raw - 50.0)

        # Step 5: contextual team-form / odds-value modifiers
        try:
            ctx_pp = _ctx_adj(team_or_player or "", sport or "")
            # Translate pp adjustment (e.g. +3pp form bonus) into score space
            profile_score += ctx_pp
        except Exception:
            pass

        profile_score = max(0.0, min(100.0, profile_score))

    # ── Step 6: team-specific recency overlay ───────────────────────────────
    # Query settled mock_bet_legs and leg_historical_resolution for this team.
    # Blended 70/30 when n_team >= 5; used as sole score when no profile.
    mt = (market_type or "").strip()
    sp = (sport or "").strip()

    sim_score: Optional[float] = None
    if team_or_player:
        team_frag = f"%{team_or_player}%"

        # mock_bet_legs (simulation)
        try:
            mock_rows = db.execute(sqla_text("""
                SELECT bl.leg_result, mb.game_date
                FROM   mock_bet_legs bl
                JOIN   mock_bets mb ON mb.id = bl.mock_bet_id
                WHERE  bl.leg_result IN ('WIN', 'LOSS')
                  AND  bl.sport        = :sport
                  AND  bl.market_type  = :market_type
                  AND  bl.description  LIKE :team_frag
                  AND  mb.game_date IS NOT NULL
                  AND  mb.source IN (
                           'prospective', 'prospective_pm', 'top_picks_page',
                           'forced_generation', 'retroactive_mock'
                       )
                ORDER BY mb.game_date DESC
                LIMIT 100
            """), {"sport": sp, "market_type": mt, "team_frag": team_frag}).fetchall()
        except Exception:
            mock_rows = []

        # leg_historical_resolution
        line_clause = ""
        hist_params: dict = {"sport": sp, "market_type": mt, "team_frag": team_frag}
        if line is not None:
            line_clause = " AND ABS(COALESCE(lhr.line, 0) - :line) < 0.6"
            hist_params["line"] = line
        try:
            hist_rows = db.execute(sqla_text(f"""
                SELECT lhr.result AS leg_result, lhr.game_date
                FROM   leg_historical_resolution lhr
                WHERE  lhr.sport       = :sport
                  AND  lhr.market_type = :market_type
                  AND  lhr.team        LIKE :team_frag
                  AND  lhr.result      IN ('WIN', 'LOSS')
                  AND  lhr.resolution_source = 'historical_db'
                {line_clause}
                ORDER BY lhr.game_date DESC
                LIMIT 200
            """), hist_params).fetchall()
        except Exception:
            hist_rows = []

        all_resolved = list(mock_rows) + list(hist_rows)

        if len(all_resolved) >= 5:
            today = _date.today()
            total_w = 0.0
            win_w   = 0.0
            for row in all_resolved:
                result    = row[0]
                game_date = row[1]
                try:
                    gd   = _date.fromisoformat(game_date[:10])
                    days = max(0, (today - gd).days)
                except Exception:
                    days = 365
                w = max(0.1, 1.0 - days / 1095.0)
                total_w += w
                if result == "WIN":
                    win_w += w
            if total_w > 0:
                cover = win_w / total_w
                sim_score = max(0.0, min(100.0, 50.0 + (cover - 0.5) * 200.0))

    # ── Combine profile + team-specific sim ──────────────────────────────────
    if profile_score is not None and sim_score is not None:
        _final = round(profile_score * 0.70 + sim_score * 0.30, 1)
    elif profile_score is not None:
        _final = round(profile_score, 1)
    elif sim_score is not None:
        _final = round(sim_score, 1)
    else:
        _final = 50.0   # insufficient data → neutral

    if _cache_key is not None:
        _COMP_A_CACHE[_cache_key] = (_final, _time.time())
    return _final


def _score_B(
    model_confidence: Optional[float],
    model_used:       Optional[str],
) -> tuple[float, list[str]]:
    """Component B: Model confidence (0–100)."""
    warnings: list[str] = []
    if model_confidence is None:
        return 50.0, warnings

    # Normalise to 0–1
    conf = model_confidence / 100.0 if model_confidence > 1.5 else model_confidence

    if conf >= 0.65:
        base = 100.0
    elif conf >= 0.60:
        base = 80.0 + (conf - 0.60) / 0.05 * 20.0
    elif conf >= 0.55:
        base = 60.0 + (conf - 0.55) / 0.05 * 20.0
    elif conf >= 0.50:
        base = 40.0 + (conf - 0.50) / 0.05 * 20.0
    else:
        base = max(10.0, conf / 0.50 * 40.0)

    bonus = _SUBMODEL_BONUS.get(model_used or "", 0.0)
    if bonus > 0:
        warnings.append(f"✅ Sub-model confidence ({model_used})")

    return min(100.0, base + bonus), warnings


def _score_C(
    market_type: Optional[str],
    line:        Optional[float],
) -> tuple[float, list[str]]:
    """Component C: Market type quality (0–100), with alt spread line penalty."""
    warnings: list[str] = []
    canonical = _canonical_market(market_type or "")
    base = _MARKET_C_SCORE.get(canonical, 50.0)

    if canonical == "Alt Spread" and line is not None:
        abs_line = abs(line)
        if abs_line > 7.5:
            base -= 25.0
            warnings.append("⚠️ Large alt spread (>7.5) — historically high drag")
        elif abs_line > 3.5:
            base -= 15.0
            warnings.append("⚠️ Alt spread > 3.5 — historically low accuracy")
        else:
            warnings.append("⚠️ Alt spread leg — monitor accuracy")

    if canonical == "Total":
        warnings.append("⚠️ Total leg — weakest market type (61.9% win rate)")

    return min(100.0, max(0.0, base)), warnings


def _score_D(edge_pp: Optional[float]) -> tuple[float, list[str]]:
    """Component D: Odds value / edge (0–100)."""
    if edge_pp is None:
        return 50.0, []
    if edge_pp > 5:
        return 100.0, []
    if edge_pp >= 2:
        return 75.0, []
    if edge_pp >= 0:
        return 50.0, []
    return 20.0, []


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HISTORICAL PROFILE LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

def _lookup_accuracy_profile(
    market_type:    str,
    sport:          str,
    team_or_player: Optional[str],
    db:             Session,
) -> dict:
    """
    Query resolved bet_legs for historical win_rate + avg accuracy_delta.

    Prefers unbiased resolution sources (historical_db, pitcher_logs).
    Falls back to all resolved legs (including inferred_parlay_win) if < 3 unbiased.
    """
    mt = (market_type or "").strip()
    sp = (sport       or "").strip()

    base_q = """
        SELECT bl.accuracy_delta, bl.leg_result, bl.resolution_source
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result IN ('WIN','LOSS')
          AND  bl.accuracy_delta IS NOT NULL
    """
    params: dict = {}

    if mt:
        base_q += " AND bl.market_type = :market_type"
        params["market_type"] = mt
    if sp:
        base_q += " AND (bl.sport = :sport OR bl.sport LIKE :sport_like)"
        params["sport"]       = sp
        params["sport_like"]  = f"%{sp}%"

    rows = db.execute(sqla_text(base_q), params).fetchall()

    # Split biased vs unbiased
    unbiased = [r for r in rows if r[2] in _UNBIASED_SOURCES]
    working  = unbiased if len(unbiased) >= 3 else rows

    if not working:
        return {
            "avg_delta":       None,
            "win_rate":        None,
            "sample_size":     0,
            "unbiased_sample": 0,
        }

    deltas = [r[0] for r in working if r[0] is not None]
    wins   = sum(1 for r in working if r[1] == "WIN")
    total  = len(working)

    return {
        "avg_delta":       round(statistics.mean(deltas), 2) if deltas else None,
        "win_rate":        wins / total if total > 0 else None,
        "sample_size":     total,
        "unbiased_sample": len(unbiased),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. COMPONENT E — MATCHUP CONTEXT (optional, MLB/NBA/NHL only)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_E(
    team:        str,
    opponent:    str,
    sport:       str,
    market_type: str,
    line:        Optional[float],
    is_home:     Optional[bool],
) -> tuple[float, dict, bool]:
    """
    Component E: Matchup context scoring.  Returns (score_0_100, context_dict, data_available).

    Three 0-100 sub-scores combined as E = E1*0.40 + E2*0.40 + E3*0.20:
      E1  Opponent quality — opponent win rate last 30 days  (higher = weaker opp = better)
      E2  Team recent form — win/cover rate last 10 games    (higher = hotter team = better)
      E3  Home/away edge  — team's venue-specific split      (higher = venue advantage)

    Only fires for MLB, NHL, NBA with at least 5 games of data in historical.db.
    Returns (50, {..., data_available: False}, False) when insufficient data.
    """
    sp = (sport or "").upper()
    if sp not in _E_SUPPORTED_SPORTS:
        return 50.0, {"data_available": False}, False

    team_hist     = _to_hist_team(team or "", sp)
    opponent_hist = _to_hist_team(opponent or "", sp)
    if not team_hist:
        return 50.0, {"data_available": False}, False

    mt_canonical = _canonical_market(market_type)

    # Check module-level cache (invalidated only on server restart)
    cache_key = (team_hist, opponent_hist, sp, mt_canonical, str(line), str(is_home))
    if cache_key in _e_cache:
        return _e_cache[cache_key]

    cutoff_30 = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_90 = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

    team_rows: list = []
    opp_data_ok = False

    try:
        conn = _hist_conn_lq()

        # ── E1: Opponent quality (last 30 days, ≥5 games required) ───────────
        e1_score = 50.0   # neutral default — used when opp data insufficient
        e1_note  = ""
        if opponent_hist:
            opp_rows = conn.execute("""
                SELECT home_team, away_team, home_score, away_score
                FROM   games
                WHERE  sport = ? AND status = 'Final'
                  AND  game_date >= ?
                  AND  (home_team = ? OR away_team = ?)
                ORDER BY game_date DESC
            """, (sp, cutoff_30, opponent_hist, opponent_hist)).fetchall()

            if len(opp_rows) >= 5:
                opp_data_ok = True
                opp_wins = sum(
                    1 for r in opp_rows
                    if (r["home_team"] == opponent_hist and (r["home_score"] or 0) > (r["away_score"] or 0))
                    or (r["away_team"] == opponent_hist and (r["away_score"] or 0) > (r["home_score"] or 0))
                )
                opp_wr = opp_wins / len(opp_rows)
                # Weak opponent → high E1 score (good for us)
                if   opp_wr < 0.30: e1_score = 85.0
                elif opp_wr < 0.40: e1_score = 70.0
                elif opp_wr < 0.50: e1_score = 55.0
                elif opp_wr < 0.60: e1_score = 45.0
                elif opp_wr < 0.70: e1_score = 35.0
                else:               e1_score = 20.0
                e1_note = f"{opponent} {opp_wr:.0%} win rate → E1={e1_score:.0f}"

        # ── E2: Team recent form — last 10 games (90-day window) ─────────────
        e2_score  = 50.0   # neutral default
        e2_note   = ""
        form_rate = None
        team_rows = conn.execute("""
            SELECT home_team, away_team, home_score, away_score
            FROM   games
            WHERE  sport = ? AND status = 'Final'
              AND  (home_team = ? OR away_team = ?)
              AND  game_date >= ?
            ORDER BY game_date DESC
            LIMIT  10
        """, (sp, team_hist, team_hist, cutoff_90)).fetchall()

        if len(team_rows) >= 5:
            if mt_canonical in ("Spread", "Alt Spread") and line is not None:
                # Cover condition: team point-margin + spread line > 0
                covered = sum(
                    1 for r in team_rows
                    if (r["home_team"] == team_hist and
                        (r["home_score"] or 0) - (r["away_score"] or 0) + float(line) > 0)
                    or (r["away_team"] == team_hist and
                        (r["away_score"] or 0) - (r["home_score"] or 0) + float(line) > 0)
                )
                form_rate = covered / len(team_rows)
                label = "cover rate"
            else:
                # Moneyline / Total: straight win rate
                wins = sum(
                    1 for r in team_rows
                    if (r["home_team"] == team_hist and (r["home_score"] or 0) > (r["away_score"] or 0))
                    or (r["away_team"] == team_hist and (r["away_score"] or 0) > (r["home_score"] or 0))
                )
                form_rate = wins / len(team_rows)
                label = "win rate"

            if   form_rate >= 0.65: e2_score = 85.0
            elif form_rate >= 0.55: e2_score = 70.0
            elif form_rate >= 0.45: e2_score = 55.0
            elif form_rate >= 0.35: e2_score = 40.0
            else:                   e2_score = 25.0
            e2_note = f"{team} {form_rate:.0%} recent {label} → E2={e2_score:.0f}"

        # ── E3: Home/away split (from same 10-game window) ───────────────────
        e3_score = 50.0   # neutral default when venue split data is thin
        e3_note  = ""
        if is_home is not None and len(team_rows) >= 5:
            if is_home:
                venue_games = [r for r in team_rows if r["home_team"] == team_hist]
                venue_wins  = sum(1 for r in venue_games if (r["home_score"] or 0) > (r["away_score"] or 0))
                venue_label = "home"
            else:
                venue_games = [r for r in team_rows if r["away_team"] == team_hist]
                venue_wins  = sum(1 for r in venue_games if (r["away_score"] or 0) > (r["home_score"] or 0))
                venue_label = "away"

            if len(venue_games) >= 3:
                split_wr = venue_wins / len(venue_games)
                if   split_wr >= 0.65: e3_score = 75.0
                elif split_wr >= 0.55: e3_score = 60.0
                elif split_wr >= 0.45: e3_score = 50.0
                elif split_wr >= 0.35: e3_score = 40.0
                else:                  e3_score = 25.0
                e3_note = f"{team} {split_wr:.0%} {venue_label} W% → E3={e3_score:.0f}"

        conn.close()

    except Exception:
        return 50.0, {"data_available": False}, False

    data_available = (len(team_rows) >= 5) or opp_data_ok
    if not data_available:
        result = (50.0, {"data_available": False}, False)
        _e_cache[cache_key] = result
        return result

    # Weighted combination: E1 opponent × 0.40, E2 form × 0.40, E3 venue × 0.20
    e_score = round(min(100.0, max(0.0,
        e1_score * 0.40 + e2_score * 0.40 + e3_score * 0.20
    )), 1)

    notes = [n for n in [e1_note, e2_note, e3_note] if n]
    context = {
        "data_available": True,
        "score":          e_score,
        "e1_opponent":    round(e1_score, 1),
        "e2_team_form":   round(e2_score, 1),
        "e3_home_away":   round(e3_score, 1),
        "form_rate":      round(form_rate, 3) if form_rate is not None else None,
        "summary":        "; ".join(notes) if notes else "neutral matchup",
    }
    result = (e_score, context, True)
    _e_cache[cache_key] = result
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_leg_quality_score(leg_candidate: dict, db: Session) -> dict:
    """
    Score a candidate leg 0–100 before placement.

    leg_candidate keys (all optional except market_type):
        market_type      str   "Moneyline" / "Spread" / "Alt Spread" / "Total" / "Player Prop"
        sport            str   "MLB" / "NHL" / "NBA" / "NFL"
        team_or_player   str
        direction        str   "over" / "under" / team name (used for close call check)
        odds             int   American odds e.g. -110
        model_confidence float 0–100 win probability from recommender
        model_used       str   "mlb_ats_v1" / "nhl_ats_v1" / "combined_v1"
        edge_pp          float edge in percentage points vs breakeven
        line             float spread or total line value (for alt spread penalty)
    """
    market_type    = leg_candidate.get("market_type") or "Other"
    sport          = leg_candidate.get("sport") or ""
    team_or_player = leg_candidate.get("team_or_player")
    direction      = leg_candidate.get("direction") or ""
    model_conf     = leg_candidate.get("model_confidence")
    model_used     = leg_candidate.get("model_used")
    edge_pp        = leg_candidate.get("edge_pp")
    line           = leg_candidate.get("line")
    # Component E inputs (optional — callers set these when available)
    opponent       = leg_candidate.get("opponent")
    is_home        = leg_candidate.get("is_home")    # bool or None

    # ── Component scores ────────────────────────────────────────────────────
    # Component A: individually-resolved leg outcomes only (no inferred_parlay_win).
    # Returns 50.0 (neutral) when < 5 clean examples are available.
    sc_A    = compute_component_a(team_or_player, sport, market_type, line, db)
    w_A     = []   # no warnings from v2 path (sparse data returns 50 silently)
    profile = {}   # kept for downstream references to profile["sample_size"] etc.
    sc_B, w_B = _score_B(model_conf, model_used)
    sc_C, w_C = _score_C(market_type, line)
    sc_D, w_D = _score_D(edge_pp)

    # ── Component E: Matchup context (optional) ──────────────────────────────
    sc_E, matchup_ctx, e_available = _score_E(
        team        = team_or_player or "",
        opponent    = opponent or "",
        sport       = sport,
        market_type = market_type,
        line        = line,
        is_home     = is_home,
    )

    # Conditional weights: redistribute to include E when data is available.
    # When E unavailable fall back to A/B/C/D weights — no penalty.
    #
    # Weight rationale (2026-04-27 calibration pass — Component A v2):
    #   A (historical accuracy) at 20%/22%: now uses compute_component_a() which
    #     queries ONLY individually-resolved mock_bet_legs + leg_historical_resolution.
    #     inferred_parlay_win data completely removed from Component A.
    #     Returns neutral 50.0 when < 5 clean examples exist (no contamination).
    #   B (model confidence) raised 30→44%: corr(win_prob, leg_won) = +0.07 (weak
    #     positive) but the only forward-looking unbiased signal available.
    #   C (market type quality) 19%: structural / unbiased.
    #   D (odds edge) 15%: corr(edge, leg_won) = -0.20 currently — miscalibrated,
    #     monitor until n≥200 post-ATS-fix settlements.
    #   E (matchup context) 10%: unchanged signal.
    #
    # Weights are read from _CURRENT_WEIGHTS so a single dict change takes effect.
    # When E is available, redistribute its 10% proportionally across A/C/D
    # (B=0 so it receives nothing).
    _w = _CURRENT_WEIGHTS   # {"A": 0.55, "B": 0.00, "C": 0.25, "D": 0.20}
    if e_available:
        # Reserve 10% for E, scale remaining A/C/D proportionally (B stays 0).
        _non_e_total = _w["A"] + _w["C"] + _w["D"]   # 1.00 (B=0)
        _scale = 0.90 / _non_e_total if _non_e_total > 0 else 0.90
        lqs = (sc_A * _w["A"] * _scale
               + sc_B * _w["B"]                        # 0
               + sc_C * _w["C"] * _scale
               + sc_D * _w["D"] * _scale
               + sc_E * 0.10)
    else:
        lqs = (sc_A * _w["A"] + sc_B * _w["B"]
               + sc_C * _w["C"] + sc_D * _w["D"])

    # ── Close call pattern check (Total / Spread only) ──────────────────────
    mt_canonical = _canonical_market(market_type)
    cc_history: Optional[dict] = None
    if mt_canonical in ("Total", "Spread", "Alt Spread") and line is not None:
        cc_history = check_close_call_history(market_type, sport, direction, line, db)
        if cc_history.get("warning"):
            lqs -= 5.0          # slight penalty: this leg type is risky near the line
            w_A = w_A + [cc_history["warning"]]

    lqs = round(min(100.0, max(0.0, lqs)), 1)

    if lqs >= 80:   grade = "A"
    elif lqs >= 65: grade = "B"
    elif lqs >= 50: grade = "C"
    else:           grade = "D"

    recommendation = "ADD" if lqs >= 65 else ("CONSIDER" if lqs >= 50 else "AVOID")

    comp_scores = {
        "accuracy":  round(sc_A, 1),
        "model":     round(sc_B, 1),
        "market":    round(sc_C, 1),
        "odds":      round(sc_D, 1),
    }
    if e_available:
        comp_scores["matchup"] = round(sc_E, 1)

    result = {
        "lqs":              lqs,
        "lqs_grade":        grade,
        "recommendation":   recommendation,
        "component_scores": comp_scores,
        "accuracy_profile": profile,
        "warnings":         w_A + w_B + w_C + w_D,
        "matchup_context":  matchup_ctx,
    }
    if cc_history is not None:
        result["close_call_history"] = cc_history
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ALT-LINE PIVOT EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_alt_line_pivot(
    main_leg:  dict,
    alt_leg:   dict,
    main_prob: float,
    db:        Session,
) -> dict:
    """
    Compare main line vs alt line on cushion-adjusted EV.

    main_leg / alt_leg keys (all optional except market_type):
        market_type  str    "Total" / "Spread" / "Alt Spread"
        direction    str    "over" / "under" / team name
        line         float  the spread or total line
        odds         float|int  decimal (≥1.1) OR American (<0 or >100)
        sport        str    "MLB" / "NBA" / "NHL" / "NFL"

    main_prob: model win probability (0-1 OR 0-100; auto-normalised).

    Pivot is recommended ONLY when:
        alt_ev > main_ev  (strictly better EV)
        AND alt_odds >= -250 (juice not excessive)
        AND prob_jump >= 6pp (meaningful cushion)

    Returns:
        main_ev, alt_ev, ev_improvement, main_prob_pct, alt_prob_pct,
        prob_jump_pp, pivot_recommended, pivot_reason, cushion_units,
        odds_penalty, main_odds, alt_odds, main_line, alt_line
    """
    # ── Normalise prob ──────────────────────────────────────────────────────
    if main_prob > 1.5:
        main_prob /= 100.0
    main_prob = max(0.05, min(0.95, main_prob))

    sport       = (main_leg.get("sport") or alt_leg.get("sport") or "").upper()
    market_type = main_leg.get("market_type") or "Total"
    direction   = (main_leg.get("direction") or "over").lower()
    main_line   = main_leg.get("line")
    alt_line_v  = alt_leg.get("line")

    # ── Convert odds to American ────────────────────────────────────────────
    def _to_am(raw):
        if raw is None: return -110
        raw = float(raw)
        if 1.01 <= raw <= 30.0:      # decimal range
            return _dec_to_american(raw)
        return int(raw)

    main_am = _to_am(main_leg.get("odds"))
    alt_am  = _to_am(alt_leg.get("odds"))

    # ── Unit shift → prob adjustment ────────────────────────────────────────
    unit_shift = 0.0
    cushion    = 0.0
    mt = _canonical_market(market_type)

    if main_line is not None and alt_line_v is not None:
        diff = float(main_line) - float(alt_line_v)
        if mt == "Total":
            # Over: lower line = easier → positive shift when alt_line < main_line
            # Under: higher line = easier → positive shift when alt_line > main_line
            unit_shift = diff if direction == "over" else -diff
        else:
            # Spread: smaller number = easier (e.g. -1.5 → +0.5 is +2 units easier)
            unit_shift = diff
        cushion = abs(diff)

    pp        = _pp_per_unit_shift(sport, market_type)
    alt_prob  = max(0.05, min(0.95, main_prob + unit_shift * pp))
    prob_jump = alt_prob - main_prob

    main_dec = _american_to_dec(main_am)
    alt_dec  = _american_to_dec(alt_am)
    main_ev  = _ev_dollar(main_prob, main_dec)
    alt_ev   = _ev_dollar(alt_prob,  alt_dec)
    ev_improvement = alt_ev - main_ev

    pivot_recommended = (
        alt_ev > main_ev
        and alt_am >= _MAX_JUICE_AMERICAN
        and prob_jump >= 0.06
    )

    if pivot_recommended:
        pivot_reason = (
            f"+{round(cushion, 1)} unit cushion · +{round(prob_jump * 100, 1)}pp probability. "
            f"EV ${round(main_ev, 2)} → ${round(alt_ev, 2)} on $10 stake."
        )
    elif alt_am < _MAX_JUICE_AMERICAN:
        pivot_reason = (
            f"Juice at {alt_am} exceeds {_MAX_JUICE_AMERICAN} limit — "
            "cushion rarely justifies the cost in a parlay."
        )
    elif prob_jump < 0.06:
        pivot_reason = (
            f"Only +{round(prob_jump * 100, 1)}pp probability gain — "
            "insufficient cushion benefit (need ≥6pp)."
        )
    else:
        pivot_reason = (
            "Alt EV worse despite probability boost — odds penalty exceeds cushion value."
        )

    return {
        "main_ev":           round(main_ev, 3),
        "alt_ev":            round(alt_ev, 3),
        "ev_improvement":    round(ev_improvement, 3),
        "main_prob_pct":     round(main_prob * 100, 1),
        "alt_prob_pct":      round(alt_prob * 100, 1),
        "prob_jump_pp":      round(prob_jump * 100, 1),
        "pivot_recommended": pivot_recommended,
        "pivot_reason":      pivot_reason,
        "cushion_units":     round(cushion, 2),
        "odds_penalty":      abs(alt_am - main_am),
        "main_odds":         main_am,
        "alt_odds":          alt_am,
        "main_line":         main_line,
        "alt_line":          alt_line_v,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLOSE CALL PATTERN CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

def check_close_call_history(
    market_type: str,
    sport:       str,
    direction:   str,
    line:        float,
    db:          Session,
) -> dict:
    """
    Look up resolved bet_legs for similar legs that were close calls
    (accuracy_delta between -1.0 and 0 — just missed).

    Similar = same market_type + sport + direction, line within ±3 units.

    Returns:
        close_call_count  int   legs that nearly missed
        close_call_rate   float fraction of similar legs that were close calls
        avg_miss          float avg delta on close calls
        warning           str or None  (if count >= 2 or rate >= 20%)
        total_similar     int   total similar resolved legs found
    """
    line = float(line or 0)
    dir_lower = (direction or "").lower()

    # Safe direction clause — only inject "over" or "under", nothing else
    dir_clause = ""
    if dir_lower in ("over", "under"):
        dir_clause = f" AND LOWER(bl.description) LIKE :dir_pattern"

    # bet_legs has no separate point/line column — filter by market/sport/direction only.
    # Line proximity matching is omitted (no column to filter on); we rely on
    # market_type + sport + direction to find similar legs.
    q = f"""
        SELECT bl.accuracy_delta, bl.leg_result
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result IN ('WIN','LOSS')
          AND  bl.accuracy_delta IS NOT NULL
          AND  bl.market_type = :market_type
          AND  (UPPER(bl.sport) = :sport OR :sport = '')
        {dir_clause}
    """
    params: dict = {
        "market_type": market_type,
        "sport":       (sport or "").upper(),
    }
    if dir_clause:
        params["dir_pattern"] = f"%{dir_lower}%"

    rows = db.execute(sqla_text(q), params).fetchall()
    total = len(rows)

    if total == 0:
        return {
            "close_call_count": 0,
            "close_call_rate":  0.0,
            "avg_miss":         None,
            "warning":          None,
            "total_similar":    0,
        }

    close_calls = [r for r in rows if r[0] is not None and -1.0 <= r[0] <= 0.0]
    count    = len(close_calls)
    rate     = count / total
    avg_miss = (
        round(sum(r[0] for r in close_calls) / len(close_calls), 2)
        if close_calls else None
    )

    warning = None
    if count >= 2 or rate >= 0.20:
        dir_label = dir_lower.upper() + " " if dir_lower in ("over", "under") else ""
        warning = (
            f"⚠️ Pattern: {count} similar {dir_label}{market_type} legs "
            f"missed by {abs(avg_miss or 0):.1f} units in history. "
            "Consider alt line for extra cushion."
        )

    return {
        "close_call_count": count,
        "close_call_rate":  round(rate, 3),
        "avg_miss":         avg_miss,
        "warning":          warning,
        "total_similar":    total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEVIATION PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_leg_deviation_profile(
    market_type:    str,
    sport:          str,
    team_or_player: Optional[str],
    db:             Session,
) -> dict:
    """
    Statistical deviation profile for a (market_type, sport) profile.

    Uses ALL resolved legs for win_rate and sample_size.
    Delta statistics (mean, std, p25, p75) are computed only from the subset
    that has accuracy_delta populated — they are None when no delta data exists.

    High mean + low std  → consistent winner (high quality)
    High mean + high std → volatile (use with caution)
    Negative mean        → systematically on wrong side
    High bad_loss_rate   → when it loses, loses badly
    """
    base_params: dict = {"market_type": market_type}
    sport_clause = ""
    team_clause  = ""
    if sport:
        sport_clause = " AND bl.sport = :sport"
        base_params["sport"] = sport
    if team_or_player:
        team_clause = " AND bl.team = :team"
        base_params["team"] = team_or_player

    # ── All resolved legs (for win_rate + sample_size) ────────────────────────
    all_rows = db.execute(sqla_text(f"""
        SELECT bl.leg_result
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result IN ('WIN','LOSS')
          AND  bl.market_type = :market_type
        {sport_clause}{team_clause}
    """), base_params).fetchall()

    if not all_rows:
        return {"sample_size": 0, "market_type": market_type, "sport": sport}

    all_results = [r[0] for r in all_rows]
    sample      = len(all_results)
    win_rate    = sum(1 for r in all_results if r == "WIN") / sample

    # ── Unbiased legs: exclude inferred_parlay_win (parlay-win inflation) ─────
    # These legs have leg_result=WIN inferred from parlay outcome, not direct resolution.
    # They inflate win_rate for multi-leg bets and must be excluded for honest stats.
    unbiased_rows = db.execute(sqla_text(f"""
        SELECT bl.leg_result
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result IN ('WIN','LOSS')
          AND  COALESCE(bl.resolution_source, '') != 'inferred_parlay_win'
          AND  bl.market_type = :market_type
        {sport_clause}{team_clause}
    """), base_params).fetchall()

    unbiased_results  = [r[0] for r in unbiased_rows]
    unbiased_sample   = len(unbiased_results)
    unbiased_win_rate = (
        sum(1 for r in unbiased_results if r == "WIN") / unbiased_sample
        if unbiased_sample > 0 else None
    )

    # ── Delta-equipped legs (for accuracy statistics) ─────────────────────────
    delta_rows = db.execute(sqla_text(f"""
        SELECT bl.accuracy_delta, bl.leg_result
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result IN ('WIN','LOSS')
          AND  bl.accuracy_delta IS NOT NULL
          AND  bl.market_type = :market_type
        {sport_clause}{team_clause}
    """), base_params).fetchall()

    mean_d = std_d = p25_d = p75_d = None
    close_loss_rate = bad_loss_rate = consistency = None

    if len(delta_rows) >= 2:
        deltas  = [r[0] for r in delta_rows]
        d_res   = [r[1] for r in delta_rows]
        losses  = [d for d, r in zip(deltas, d_res) if r == "LOSS"]
        n_d     = len(deltas)

        sorted_d    = sorted(deltas)
        p25_idx     = max(0, int(n_d * 0.25) - 1)
        p75_idx     = min(n_d - 1, int(n_d * 0.75))
        mean_d      = statistics.mean(deltas)
        std_d       = statistics.stdev(deltas)
        p25_d       = sorted_d[p25_idx]
        p75_d       = sorted_d[p75_idx]

        close_loss_rate = (sum(1 for d in losses if d > -1.0) / len(losses)) if losses else 0.0
        bad_loss_rate   = (sum(1 for d in losses if d < -3.0) / len(losses)) if losses else 0.0
        consistency     = max(0.0, min(1.0, 1.0 - std_d / max(abs(mean_d), 1.0)))

    return {
        "market_type":        market_type,
        "sport":              sport,
        "team_or_player":     team_or_player,
        "sample_size":        sample,
        "unbiased_sample":    unbiased_sample,
        "delta_sample":       len(delta_rows),
        "mean_delta":         round(mean_d, 3)          if mean_d          is not None else None,
        "std_delta":          round(std_d, 3)           if std_d           is not None else None,
        "p25_delta":          round(p25_d, 3)           if p25_d           is not None else None,
        "p75_delta":          round(p75_d, 3)           if p75_d           is not None else None,
        "win_rate":           round(win_rate, 3),
        "unbiased_win_rate":  round(unbiased_win_rate, 3) if unbiased_win_rate is not None else None,
        "close_loss_rate":    round(close_loss_rate, 3) if close_loss_rate is not None else None,
        "bad_loss_rate":      round(bad_loss_rate, 3)   if bad_loss_rate   is not None else None,
        "consistency_score":  round(consistency, 3)     if consistency     is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PROFILE TABLE REBUILD
# ═══════════════════════════════════════════════════════════════════════════════

def update_quality_profiles(db: Session) -> dict:
    """
    Rebuild leg_quality_profiles from all resolved bet_legs.
    Groups by (market_type, sport). Run after every resolve-legs call.
    """
    # Query all resolved legs regardless of accuracy_delta — delta stats are
    # computed separately (and may be NULL) inside compute_leg_deviation_profile.
    combos = db.execute(sqla_text("""
        SELECT DISTINCT bl.market_type, bl.sport
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result IN ('WIN','LOSS')
          AND  bl.market_type IS NOT NULL
    """)).fetchall()

    updated = skipped = 0
    now_ts  = datetime.utcnow().isoformat()

    for market_type, sport in combos:
        if not market_type:
            skipped += 1
            continue

        p = compute_leg_deviation_profile(market_type, sport or "", None, db)
        if p.get("sample_size", 0) < 2:
            skipped += 1
            continue

        existing = db.execute(sqla_text("""
            SELECT id FROM leg_quality_profiles
            WHERE  market_type = :mt AND sport = :sp AND team_or_player IS NULL
        """), {"mt": market_type, "sp": sport or ""}).fetchone()

        vals = dict(
            mean=p["mean_delta"],   std=p["std_delta"],
            p25=p["p25_delta"],     p75=p["p75_delta"],
            wr=p["win_rate"],       uwr=p["unbiased_win_rate"],
            clr=p["close_loss_rate"], blr=p["bad_loss_rate"],
            cs=p["consistency_score"],
            ss=p["sample_size"],    uss=p["unbiased_sample"],
            ts=now_ts,
        )
        if existing:
            db.execute(sqla_text("""
                UPDATE leg_quality_profiles
                SET    mean_delta=:mean,        std_delta=:std,
                       p25_delta=:p25,          p75_delta=:p75,
                       win_rate=:wr,            unbiased_win_rate=:uwr,
                       close_loss_rate=:clr,    bad_loss_rate=:blr,
                       consistency_score=:cs,
                       sample_size=:ss,         unbiased_sample=:uss,
                       last_updated=:ts
                WHERE  id=:id
            """), {**vals, "id": existing[0]})
        else:
            db.execute(sqla_text("""
                INSERT INTO leg_quality_profiles
                  (market_type, sport, team_or_player,
                   mean_delta, std_delta, p25_delta, p75_delta,
                   win_rate, unbiased_win_rate,
                   close_loss_rate, bad_loss_rate,
                   consistency_score, sample_size, unbiased_sample, last_updated)
                VALUES
                  (:mt, :sp, NULL,
                   :mean, :std, :p25, :p75,
                   :wr, :uwr,
                   :clr, :blr,
                   :cs, :ss, :uss, :ts)
            """), {**vals, "mt": market_type, "sp": sport or ""})

        updated += 1

    db.commit()
    return {"profiles_updated": updated, "profiles_skipped": skipped}


# ═══════════════════════════════════════════════════════════════════════════════
# 10. RETROACTIVE LQS BACKFILL
# ═══════════════════════════════════════════════════════════════════════════════

def backfill_lqs_on_bet_legs(db: Session) -> dict:
    """
    Retroactively populate lqs / lqs_grade on existing bet_legs that have no score.

    For each bet_leg where lqs IS NULL and market_type IS NOT NULL:
      - Build a leg_candidate dict from the stored fields
      - Call compute_leg_quality_score()
      - UPDATE bet_legs SET lqs=?, lqs_grade=? WHERE id=?

    Also updates avg_lqs on parent bets after all legs are scored.

    Returns: {scored, skipped, errors}
    """
    # bet_legs has no separate odds/point columns; use odds_str and parse line from description
    rows = db.execute(sqla_text("""
        SELECT bl.id, bl.market_type, bl.sport, bl.team,
               bl.odds_str, bl.description, bl.bet_id
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.lqs IS NULL
          AND  bl.market_type IS NOT NULL
    """)).fetchall()

    scored = skipped = errors = 0
    affected_bet_ids: set = set()

    # Parse a numeric line from description text (e.g. "Over 8.5", "Cubs -1.5")
    import re as _re
    _line_re = _re.compile(r"([-+]?\d+\.?\d*)")

    def _parse_line(desc: str, team: str) -> float | None:
        text = (desc or team or "")
        m = _line_re.search(text)
        return float(m.group(1)) if m else None

    def _parse_am_odds(odds_str: str) -> int | None:
        if not odds_str:
            return None
        m = _re.search(r"([-+]?\d+)", str(odds_str))
        return int(m.group(1)) if m else None

    for row in rows:
        leg_id, market_type, sport, team, odds_str, description, bet_id = row
        try:
            line    = _parse_line(description, team)
            am_odds = _parse_am_odds(odds_str)
            # Infer direction from description
            desc_l   = (description or "").lower()
            direction = "over" if "over" in desc_l else ("under" if "under" in desc_l else "")
            candidate = {
                "market_type":    market_type,
                "sport":          sport or "",
                "team_or_player": team or description or "",
                "direction":      direction,
                "odds":           am_odds,
                "line":           line,
                # model_confidence / model_used / edge_pp unknown for historical legs → defaults
            }
            result = compute_leg_quality_score(candidate, db)
            db.execute(sqla_text("""
                UPDATE bet_legs
                SET    lqs = :lqs, lqs_grade = :grade
                WHERE  id  = :id
            """), {"lqs": result["lqs"], "grade": result["lqs_grade"], "id": leg_id})
            affected_bet_ids.add(bet_id)
            scored += 1
        except Exception:
            errors += 1
            # Don't abort the whole run for one bad leg
            continue

    db.commit()

    # Count legs that were skipped (no market_type)
    skipped_rows = db.execute(sqla_text("""
        SELECT COUNT(*) FROM bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.lqs IS NULL
          AND  bl.market_type IS NULL
    """)).scalar() or 0
    skipped = int(skipped_rows)

    return {
        "scored":           scored,
        "skipped":          skipped,
        "errors":           errors,
        "bets_updated":     len(affected_bet_ids),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. WEIGHT TUNING (OLS)
# ═══════════════════════════════════════════════════════════════════════════════


def tune_lqs_weights(db: Session) -> dict:
    """
    Correlate stored LQS against actual leg outcomes using UNBIASED legs only.

    Excludes resolution_source = 'inferred_parlay_win' because parlay-win inference
    inflates high-LQS win counts (all legs of a winning parlay are marked WIN even if
    some individual legs were weak picks). This causes lqs_mean_when_lost > lqs_mean_when_won.

    Unbiased sources: 'historical_db', 'pitcher_logs', 'straight_bet', or NULL
    (NULL = legacy rows resolved before source tracking was added).

    Requires minimum 50 unbiased resolved legs with LQS stored.
    """
    rows = db.execute(sqla_text("""
        SELECT bl.leg_result, bl.lqs
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result IN ('WIN','LOSS')
          AND  bl.lqs IS NOT NULL
          AND  COALESCE(bl.resolution_source, '') != 'inferred_parlay_win'
    """)).fetchall()

    n = len(rows)
    if n < 50:
        return {
            "error": f"Need 50 unbiased resolved legs with LQS stored, have {n}. "
                     "Run /api/attribution/resolve-legs then /api/legs/update-profiles first.",
            "n":           n,
            "unbiased_n":  n,
            "note":        "insufficient unbiased data",
        }

    y_vals = [1.0 if r[0] == "WIN" else 0.0 for r in rows]
    x_vals = [r[1] for r in rows]
    mx, my = sum(x_vals) / n, sum(y_vals) / n
    cov    = sum((x - mx) * (y - my) for x, y in zip(x_vals, y_vals)) / n
    sx     = (sum((x - mx) ** 2 for x in x_vals) / n) ** 0.5
    sy     = (sum((y - my) ** 2 for y in y_vals) / n) ** 0.5
    corr   = cov / (sx * sy) if sx * sy > 0 else 0.0

    won_lqs  = [x for x, y in zip(x_vals, y_vals) if y == 1.0]
    lost_lqs = [x for x, y in zip(x_vals, y_vals) if y == 0.0]

    # Grade-level win rates
    grade_stats: dict[str, dict] = {}
    for result, lqs in rows:
        g = "A" if lqs >= 80 else ("B" if lqs >= 65 else ("C" if lqs >= 50 else "D"))
        if g not in grade_stats:
            grade_stats[g] = {"wins": 0, "total": 0}
        grade_stats[g]["total"] += 1
        if result == "WIN":
            grade_stats[g]["wins"] += 1

    grade_win_rates = {
        g: round(v["wins"] / v["total"] * 100, 1) if v["total"] > 0 else None
        for g, v in sorted(grade_stats.items())
    }

    unbiased_note = (
        "insufficient unbiased data — collect more straight-bet resolutions"
        if n < 100 else
        "Positive correlation confirms LQS predictive value. "
        "Re-run after 200+ settlements to calibrate component weights via OLS regression."
    )

    return {
        "n":                    n,
        "unbiased_n":           n,
        "lqs_win_corr":         round(corr, 3),
        "lqs_mean_when_won":    round(sum(won_lqs)  / len(won_lqs),  1) if won_lqs  else None,
        "lqs_mean_when_lost":   round(sum(lost_lqs) / len(lost_lqs), 1) if lost_lqs else None,
        "grade_win_rates":      grade_win_rates,
        "note":                 unbiased_note,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SELF-CALIBRATION TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════

# Current component weights — 2026-04-28: B set to 0%
# Signal analysis confirmed corr(B_proxy=win_prob, leg_won)=-0.0643 (negative).
# B is anti-predictive; removing it entirely. Using A+C+D only.
# Without B: A=0.55, C=0.25, D=0.20  (sum=1.00)
_CURRENT_WEIGHTS: dict[str, float] = {"A": 0.55, "B": 0.00, "C": 0.25, "D": 0.20}
_DRIFT_THRESHOLD = 0.05   # 5pp difference triggers drift warning


def run_self_calibration(db: Session) -> dict:
    """
    Full LQS self-calibration loop. Intended to run after every FanDuel sync.

    Steps:
      1. backfill_lqs_on_bet_legs()    — score any new unscored legs
      2. update_quality_profiles()     — rebuild deviation profiles
      3. tune_lqs_weights()            — correlate LQS vs outcomes (unbiased only)
      4. Compute optimal component weights via univariate OLS per component
      5. Compare to _CURRENT_WEIGHTS — flag drift > 5pp
      6. Write to lqs_calibration_log
      7. Return calibration summary

    Optimal weight estimation: for each component column (A/B/C/D), compute the
    correlation with win outcome using unbiased legs. Normalise to sum=1.
    """
    import json as _json
    from database import LqsCalibrationLog

    summary: dict = {}

    # Step 1: backfill
    try:
        bf = backfill_lqs_on_bet_legs(db)
        summary["backfill"] = bf
    except Exception as e:
        summary["backfill_error"] = str(e)

    # Step 2: profiles
    try:
        pr = update_quality_profiles(db)
        summary["profiles"] = pr
    except Exception as e:
        summary["profiles_error"] = str(e)

    # Step 3: tune (unbiased)
    tune = tune_lqs_weights(db)
    summary["tune"] = tune

    n_unbiased  = tune.get("unbiased_n", tune.get("n", 0))
    correlation = tune.get("lqs_win_corr")

    # Step 4: estimate per-component correlations using unbiased leg data
    # We don't store per-component scores in bet_legs, so we use grade-level win rates
    # as a proxy. Grade A/B/C/D → assign mid-point LQS (90, 72, 57, 40) and correlate.
    grade_wr   = tune.get("grade_win_rates", {})
    grade_mid  = {"A": 90, "B": 72, "C": 57, "D": 40}
    optimal    = dict(_CURRENT_WEIGHTS)   # start from current; update where data exists
    drift_detected = False

    if grade_wr and len(grade_wr) >= 3:
        # Simple univariate: map grade → win_rate, fit linear slope
        xs = [grade_mid[g] for g in ("A","B","C","D") if g in grade_wr and grade_wr[g] is not None]
        ys = [grade_wr[g] / 100.0 for g in ("A","B","C","D") if g in grade_wr and grade_wr[g] is not None]
        if len(xs) >= 3:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov    = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            var_x  = sum((x - mx) ** 2 for x in xs) or 1.0
            slope  = cov / var_x
            # Map slope back to component weight hints (heuristic)
            # Higher slope → LQS predicts well → keep or raise accuracy weight (A)
            if slope > 0.003:
                optimal["A"] = min(0.55, _CURRENT_WEIGHTS["A"] + round(slope * 10, 2))
            elif slope < 0.001:
                optimal["A"] = max(0.30, _CURRENT_WEIGHTS["A"] - 0.05)
            # Normalise so weights sum to 1
            total = sum(optimal.values())
            optimal = {k: round(v / total, 3) for k, v in optimal.items()}

    # Step 5: drift check
    drift_details = {}
    for comp, cur_w in _CURRENT_WEIGHTS.items():
        opt_w = optimal.get(comp, cur_w)
        diff  = abs(opt_w - cur_w)
        if diff >= _DRIFT_THRESHOLD:
            drift_detected = True
            drift_details[comp] = {"current": cur_w, "optimal": round(opt_w, 3), "diff": round(diff, 3)}

    drift_note = None
    if drift_detected:
        drift_note = (
            "LQS weights drift detected — consider updating component weights. "
            "Drifted components: " + ", ".join(
                f"{k}: {v['current']} → {v['optimal']}" for k, v in drift_details.items()
            )
        )
    elif n_unbiased < 50:
        drift_note = "insufficient unbiased data — collect more straight-bet resolutions before re-weighting"

    # Step 6: log to DB
    try:
        log_entry = LqsCalibrationLog(
            run_date        = datetime.utcnow().strftime("%Y-%m-%d"),
            n_unbiased      = n_unbiased,
            current_weights = _json.dumps(_CURRENT_WEIGHTS),
            optimal_weights = _json.dumps(optimal),
            correlation     = correlation,
            drift_detected  = drift_detected,
            note            = drift_note or tune.get("note", ""),
        )
        db.add(log_entry)
        db.commit()
        summary["logged"] = True
    except Exception as e:
        summary["log_error"] = str(e)

    summary.update({
        "n_unbiased":           n_unbiased,
        "correlation":          correlation,
        "current_weights":      _CURRENT_WEIGHTS,
        "optimal_weights":      optimal,
        "recommended_weights":  optimal,   # alias — same dict, clearer name for API consumers
        "drift_detected":       drift_detected,
        "drift_details":        drift_details,
        "drift_note":           drift_note,
    })
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 13. ATTRIBUTION QUALITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def get_lqs_attribution(db: Session) -> dict:
    """
    LQS distribution, close calls, and bad misses for the attribution UI.
    """
    grade_rows = db.execute(sqla_text("""
        SELECT bl.lqs_grade,
               COUNT(*)                                                  AS leg_count,
               SUM(CASE WHEN bl.leg_result = 'WIN' THEN 1 ELSE 0 END)   AS wins,
               AVG(bl.accuracy_delta)                                    AS avg_delta
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.lqs_grade IS NOT NULL
          AND  bl.leg_result IN ('WIN','LOSS')
        GROUP BY bl.lqs_grade
        ORDER BY bl.lqs_grade
    """)).fetchall()

    grade_dist = []
    for r in grade_rows:
        total = r[1] or 0
        wins  = r[2] or 0
        grade_dist.append({
            "grade":     r[0],
            "leg_count": total,
            "win_rate":  round(wins / total * 100, 1) if total > 0 else None,
            "avg_delta": round(r[3], 2) if r[3] is not None else None,
        })

    # Close calls: losses where delta > -1.0 (just missed)
    close_calls = db.execute(sqla_text("""
        SELECT bl.team, bl.market_type, bl.sport, bl.accuracy_delta, bl.description
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result = 'LOSS'
          AND  bl.accuracy_delta > -1.0
          AND  bl.accuracy_delta IS NOT NULL
        ORDER BY bl.accuracy_delta DESC
        LIMIT 15
    """)).fetchall()

    # Bad misses: losses where delta < -3.0 (wrong analysis)
    bad_misses = db.execute(sqla_text("""
        SELECT bl.team, bl.market_type, bl.sport, bl.accuracy_delta, bl.description
        FROM   bet_legs bl
        JOIN   bets b ON b.id = bl.bet_id
        WHERE  b.is_mock = 0
          AND  bl.leg_result = 'LOSS'
          AND  bl.accuracy_delta < -3.0
          AND  bl.accuracy_delta IS NOT NULL
        ORDER BY bl.accuracy_delta ASC
        LIMIT 15
    """)).fetchall()

    def _row(r):
        return {
            "team":        r[0],
            "market_type": r[1],
            "sport":       r[2],
            "delta":       round(r[3], 2),
            "description": (r[4] or "")[:100],
        }

    return {
        "grade_distribution": grade_dist,
        "close_calls":        [_row(r) for r in close_calls],
        "bad_misses":         [_row(r) for r in bad_misses],
        "has_lqs_data":       len(grade_dist) > 0,
    }
