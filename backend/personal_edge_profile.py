"""
personal_edge_profile.py
=======================
Builds and maintains the personal_edge_profile table —
Jose's actual win rates by sport × market_type × line_bucket.

This is the primary source for Component A in LQS scoring.

Data source: bet_legs (real bets, is_mock=0)
  - Win rate:    legs WHERE resolution_source != 'inferred_parlay_win'
                 (excludes parlay-inference artifact that inflated Basketball/Football/Baseball WR to 100%)
  - Margin data: legs WHERE accuracy_delta IS NOT NULL
                 (historical_db and soccer_api resolutions only)

Why not leg_quality_profiles?
  leg_quality_profiles uses win_rate (ALL legs including inferred_parlay_win) for its primary WR.
  Analysis showed Basketball Moneyline is 100% WR from 237 parlay-inferred legs — not real performance.
  Correct unbiased WRs from bet_legs: Soccer ML 76%, NBA ML 75%, MLB ML 75%, Soccer Total 70.3%.

Margin grade thresholds (updated after full analysis 2026-04-29):
  CUSHION: wr >= 0.75 AND mean_delta > 1.0 AND edge_ratio < 2.0
  CLOSE:   wr >= 0.60 but not CUSHION (tight margins or WR 60-75%)
  MIXED:   edge_ratio > 3.0 AND wr >= 0.65 (volatile results)
  AVOID:   wr < 0.50 OR mean_delta < 0 OR insufficient data

Refresh: call refresh_personal_edge_profiles(db=None) weekly (Sunday scheduler).
         Also called on server start the first time the table is empty.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from datetime import datetime
from typing import Optional

# ── DB path ───────────────────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")


# ── Sport normalization ───────────────────────────────────────────────────────
# Maps odds-API sport labels → bet_legs.sport labels (the source of truth).
#
# bet_legs uses these sport labels (from Pikkit import + direct bets):
#   "Soccer", "Basketball", "NBA", "American Football",
#   "Major League Baseball", "Ice Hockey", "NHL Games", "Baseball"
#
# Analysis 2026-04-29 showed:
#   - "Basketball" in bet_legs = WNBA + NCAAB + NBA combined → 74.2% Total WR (unbiased)
#   - "NBA" in bet_legs = 8 real legs, 75% WR (specific, reliable)
#   - "NHL Games" ≠ "Ice Hockey" in bet_legs (keep separate — different profiles)
#   - "Baseball" = alt spreads/totals; "Major League Baseball" = MLB specific
_SPORT_NORM: dict[str, str] = {
    # Odds-API league labels → bet_legs sport labels
    "La Liga":       "Soccer",
    "EPL":           "Soccer",
    "Serie A":       "Soccer",
    "MLS":           "Soccer",
    "UCL":           "Soccer",
    "Ligue 1":       "Soccer",
    "Bundesliga":    "Soccer",
    "Eredivisie":    "Soccer",
    "Europa League": "Soccer",
    # NBA: map to "NBA" — bet_legs has 8 real unbiased NBA legs at 75% WR
    # "Basketball" in bet_legs is 151-leg Total profile (WNBA/NCAAB mix, 74.2% WR)
    "NBA":   "NBA",
    "NCAAB": "Basketball",  # college basketball → generic Basketball profile
    # NFL
    "NFL":   "American Football",
    "NCAAF": "American Football",
    # MLB: bet_legs uses "Major League Baseball" for MLB-specific legs
    "MLB":      "Major League Baseball",
    # NHL: normalize both "NHL Games" and "Ice Hockey" → "Ice Hockey".
    # bet_legs uses "NHL Games" for Moneyline legs and "Ice Hockey" for Total legs
    # (inconsistent Pikkit import labels for the same sport). Consolidating to
    # "Ice Hockey" ensures that NHL ML and Total both land in the same profile group,
    # and API lookups with sport="NHL" find both Moneyline and Total profiles.
    "NHL":       "Ice Hockey",
    "NHL Games": "Ice Hockey",
    # Already canonical (from bet_legs import)
    "Soccer":                "Soccer",
    "Basketball":            "Basketball",
    "American Football":     "American Football",
    "Major League Baseball": "Major League Baseball",
    "Ice Hockey":            "Ice Hockey",
    "Baseball":              "Baseball",
}


def normalize_sport(sport: str) -> str:
    """Return the canonical sport name used in personal_edge_profile."""
    return _SPORT_NORM.get((sport or "").strip(), (sport or "").strip())


# ── Market-type normalization ─────────────────────────────────────────────────
# Converts odds-API market_key values to canonical market_type labels.
_MARKET_NORM: dict[str, str] = {
    "h2h":               "Moneyline",
    "moneyline":         "Moneyline",
    "spreads":           "Spread",
    "spread":            "Spread",
    "totals":            "Total",
    "total":             "Total",
    "alternate_spreads": "Alt Spread",
    "alt_spread":        "Alt Spread",
    "alternate_totals":  "Alt Total",
    "alt_total":         "Alt Total",
    # Already canonical
    "Moneyline":  "Moneyline",
    "Spread":     "Spread",
    "Total":      "Total",
    "Alt Spread": "Alt Spread",
    "Alt Total":  "Alt Total",
    "Other":      "Other",
}


def normalize_market(market_type: str) -> str:
    return _MARKET_NORM.get((market_type or "").strip(), (market_type or "").strip())


# ── Line-bucket classifier ────────────────────────────────────────────────────

def classify_line_bucket(
    market_type: str,
    description: str = "",
    point: Optional[float] = None,
) -> str:
    """
    Classify a leg into a line bucket for personal_edge_profile lookup.

    Priority: point float > description regex > market_type.

    Buckets:
      'ML'     — moneyline (any sport)
      'DC'     — double chance (soccer)
      '+1.5'   — underdog spread / alt spread at +1.5
      '-1.5'   — favourite spread at -1.5
      '+2.5'   — underdog spread / alt spread at +2.5
      '-2.5'   — favourite spread at -2.5
      'other'  — any other spread or unclassified point
      'Total'  — over/under totals
      'Other'  — player props, exotic markets
    """
    mt = (market_type or "").strip()
    mt_norm = normalize_market(mt)
    desc = (description or "").lower()

    # Moneyline / double-chance detection
    if mt_norm == "Moneyline":
        if "double_chance" in mt.lower() or "double chance" in desc or "dc" == mt.lower():
            return "DC"
        return "ML"

    # Total
    if mt_norm in ("Total", "Alt Total"):
        return "Total"

    # Spread / Alt Spread — classify by point value
    if mt_norm in ("Spread", "Alt Spread"):
        p: Optional[float] = point

        # Try to parse from description if not supplied
        if p is None:
            m = re.search(r"([+-]\d+\.?\d*)\s*\(", description or "")
            if m:
                try:
                    p = float(m.group(1))
                except ValueError:
                    pass

        if p is not None:
            try:
                pf = float(p)
                if abs(pf - 1.5)  < 0.1: return "+1.5"
                if abs(pf + 1.5)  < 0.1: return "-1.5"
                if abs(pf - 2.5)  < 0.1: return "+2.5"
                if abs(pf + 2.5)  < 0.1: return "-2.5"
                if abs(pf - 3.5)  < 0.1: return "+3.5"
                if abs(pf + 3.5)  < 0.1: return "-3.5"
            except (TypeError, ValueError):
                pass
        return "other"

    # Other
    return "Other"


# ── Table DDL ─────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS personal_edge_profile (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    sport             TEXT NOT NULL,
    market_type       TEXT NOT NULL,
    line_bucket       TEXT NOT NULL,
    sample_size       INTEGER NOT NULL DEFAULT 0,
    personal_wr       REAL,
    mean_delta        REAL,
    std_delta         REAL,
    edge_ratio        REAL,    -- std_delta / ABS(mean_delta): lower = more consistent
    close_call_rate   REAL,    -- fraction of WINS where delta < 0.5 (close calls)
    narrow_loss_rate  REAL,    -- fraction of LOSSES where delta > -0.5 (near-misses)
    margin_grade      TEXT,    -- 'CUSHION' | 'CLOSE' | 'MIXED' | 'AVOID'
    max_parlay_legs   INTEGER NOT NULL DEFAULT 2, -- max legs this profile may appear in
    avg_odds          REAL,
    avg_ev            REAL,
    data_sources      TEXT,
    last_updated      TEXT,
    UNIQUE(sport, market_type, line_bucket)
)
"""

# ── Margin grade rules ────────────────────────────────────────────────────────
# Updated 2026-04-29 based on full 594-bet analysis with correct unbiased WRs.
#
# CUSHION: wr >= 0.70 AND mean_delta > 0 AND (edge_ratio < 3.0 OR n < 15)
#   → These legs anchor parlay construction. They have confirmed positive win
#     margins. Small-n (< 15) profiles pass the edge_ratio gate because we don't
#     have enough data to reject them — confidence-damping in Component A handles
#     the uncertainty.
#
# MIXED:   wr 0.55–0.65, OR wr >= 0.65 with edge_ratio >= 3.0 AND n >= 15
#   → Volatile. NFL Total: 73% WR but edge_ratio=5.29, n=30 → MIXED.
#   → 1-leg only for tracking/calibration.
#
# CLOSE:   wr >= 0.65 with positive/no delta but not CUSHION or MIXED
#   → Includes Basketball ML (100% WR but no delta data — can't verify margin).
#   → 1-leg only until enough delta data accumulates.
#
# AVOID:   wr < 0.55 OR mean_delta < 0 → excluded entirely.

def compute_margin_grade(
    personal_wr:     Optional[float],
    mean_delta:      Optional[float],
    edge_ratio:      Optional[float],
    close_call_rate: Optional[float],
    sample_size:     int = 0,
) -> str:
    """
    Assign a margin quality grade. Priority: AVOID → CUSHION → MIXED → CLOSE.

    CUSHION: wr >= 0.70 AND mean_delta > 0 AND (edge_ratio < 3.0 OR n < 15)
    MIXED:   wr 0.55-0.65, OR (wr >= 0.65 AND edge_ratio >= 3.0 AND n >= 15)
    CLOSE:   wr >= 0.65 (remaining — positive/no delta, not CUSHION/MIXED)
    AVOID:   wr < 0.55, or mean_delta < 0
    """
    if personal_wr is None:
        return "AVOID"
    if personal_wr < 0.55:
        return "AVOID"
    if mean_delta is not None and mean_delta < 0:
        return "AVOID"

    # CUSHION: solid WR + positive margin + not excessively volatile
    # Small-n profiles (< 15 legs) pass even with unknown edge_ratio since we
    # don't have enough data to disqualify them — confidence weight handles it.
    if (personal_wr >= 0.70
            and mean_delta is not None and mean_delta > 0
            and (edge_ratio is None or edge_ratio < 3.0 or sample_size < 15)):
        return "CUSHION"

    # MIXED: volatile (high edge_ratio with sufficient sample to be certain),
    # or borderline WR in 55-65% range
    if (edge_ratio is not None and edge_ratio >= 3.0
            and sample_size >= 15 and personal_wr >= 0.65):
        return "MIXED"
    if 0.55 <= personal_wr <= 0.65:
        return "MIXED"

    # CLOSE: decent WR but insufficient/missing delta data, or WR in 65-70% range
    if personal_wr >= 0.65:
        return "CLOSE"

    return "AVOID"


def compute_max_parlay_legs(
    market_type:  str,
    personal_wr:  Optional[float],
    margin_grade: str,
    mean_delta:   Optional[float] = None,
    edge_ratio:   Optional[float] = None,
) -> int:
    """
    Return the max number of legs this profile can appear in per parlay.

    AVOID  → 0  (blocked entirely)
    MIXED  → 1  (1-leg straight only — volatile edge_ratio or borderline WR)
    CLOSE  → 2  (2-leg max — decent WR but missing/narrow delta data)

    CUSHION anchor tier → 5:
      wr >= 0.75 AND mean_delta > 1.0 AND edge_ratio < 2.5
      These are the most reliable legs: confirmed WR, wide win cushion,
      and consistent results. They can anchor 4-5 leg parlays.
      Examples: Soccer ML (76%, δ+1.24, er1.42), MLB ML (75%, δ+2.94, er1.31).

    CUSHION supporting tier → 3:
      wr >= 0.70 but doesn't meet anchor criteria (large delta variance, or
      mean_delta ≤ 1.0, or small sample passing edge_ratio gate).
      Examples: Basketball Total (74.2%, δ+61, er1.53 — high absolute delta
      but WR < 0.75), Soccer Total (70.3%, δ+1.23, er1.62 — WR just above floor).

    The anchor/supporting split ensures 5-leg parlays are built only from
    the most proven profiles. Supporting legs still contribute to 2-3 leg combos.
    """
    if margin_grade == "AVOID":
        return 0
    if margin_grade == "MIXED":
        return 1
    if margin_grade in ("CLOSE", "NEUTRAL"):
        # NEUTRAL = no personal data yet; treat conservatively same as CLOSE.
        return 2
    if margin_grade == "CUSHION":
        wr = personal_wr or 0.0
        # Anchor tier: high WR + large, consistent win margin
        if (wr >= 0.75
                and mean_delta is not None and mean_delta > 1.0
                and edge_ratio is not None and edge_ratio < 2.5):
            return 5   # anchor: any parlay size up to 5-leg
        return 3       # supporting: 2-3 leg; edge_ratio too high or WR < 0.75
    return 2           # fallback (unknown grade)


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL)
    conn.commit()
    # Migrate: add margin columns if they don't exist yet (table created before this version)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(personal_edge_profile)")}
    migrations = [
        ("edge_ratio",        "REAL"),
        ("close_call_rate",   "REAL"),
        ("narrow_loss_rate",  "REAL"),
        ("margin_grade",      "TEXT"),
        ("max_parlay_legs",   "INTEGER NOT NULL DEFAULT 2"),
    ]
    for col, col_type in migrations:
        if col not in existing:
            conn.execute(f"ALTER TABLE personal_edge_profile ADD COLUMN {col} {col_type}")
    conn.commit()


# ── Data builders ─────────────────────────────────────────────────────────────
# Minimum sample sizes for profile inclusion
_MIN_N_FOR_PROFILE = 5    # must have at least 5 unbiased legs to include
_MIN_N_FOR_DELTA   = 3    # must have at least 3 legs with accuracy_delta


def _personal_rows(conn: sqlite3.Connection) -> list[dict]:
    """
    Build profile rows directly from bet_legs (real bets, is_mock=0).

    Win rate: legs WHERE resolution_source != 'inferred_parlay_win'
      — This excludes the parlay-inference artifact that inflated Basketball/Football/
        Baseball Moneyline to 100% WR. Those legs have 0 accuracy_delta and were
        inferred as WIN because all other legs in the parlay won — not a real signal.

    Margin metrics: legs WHERE accuracy_delta IS NOT NULL
      — historical_db and soccer_api resolutions provide real game outcome margins.
      — accuracy_delta = actual_outcome_value - bet_line (positive = won by that margin)

    Groups by sport × market_type. Line bucket derived from market_type.
    Minimum: _MIN_N_FOR_PROFILE unbiased legs to include a profile row.
    """
    rows = conn.execute("""
        SELECT
            -- Normalize "NHL Games" Moneyline → "Ice Hockey" so the 6 NHL ML legs
            -- land in the same group as any "Ice Hockey" Moneyline legs.
            -- NOTE: Only normalize Moneyline — NHL Games Total (3 legs, 33% WR) is
            -- intentionally kept SEPARATE so it doesn't corrupt Ice Hockey Total
            -- (9 legs, 77.8% WR). With n=3 it falls below min_n and gets excluded.
            CASE
                WHEN bl.sport = 'NHL Games' AND bl.market_type = 'Moneyline'
                THEN 'Ice Hockey'
                ELSE bl.sport
            END as sport,
            bl.market_type,
            COUNT(*) as n_unbiased,
            AVG(CASE WHEN bl.leg_result='WIN' THEN 1.0 ELSE 0.0 END) as wr,
            COUNT(CASE WHEN bl.accuracy_delta IS NOT NULL THEN 1 END) as n_delta,
            AVG(CASE WHEN bl.accuracy_delta IS NOT NULL THEN bl.accuracy_delta END) as mean_d,
            AVG(CASE WHEN bl.accuracy_delta IS NOT NULL
                     THEN bl.accuracy_delta * bl.accuracy_delta END) as mean_sq,
            -- close_call_rate: % of WINS with |delta| < 0.5 (narrow wins)
            AVG(CASE WHEN bl.leg_result='WIN' AND bl.accuracy_delta IS NOT NULL
                     AND ABS(bl.accuracy_delta) < 0.5 THEN 1.0 ELSE 0.0 END) as ccr,
            -- narrow_loss_rate: % of LOSSES with |delta| < 0.5 (near-miss losses)
            AVG(CASE WHEN bl.leg_result='LOSS' AND bl.accuracy_delta IS NOT NULL
                     AND ABS(bl.accuracy_delta) < 0.5 THEN 1.0 ELSE 0.0 END) as nlr
        FROM bet_legs bl
        JOIN bets b ON bl.bet_id = b.id
        WHERE b.is_mock = 0
          AND bl.resolution_source != 'inferred_parlay_win'
          AND bl.leg_result IN ('WIN', 'LOSS')
        GROUP BY
            CASE
                WHEN bl.sport = 'NHL Games' AND bl.market_type = 'Moneyline'
                THEN 'Ice Hockey'
                ELSE bl.sport
            END,
            bl.market_type
        HAVING COUNT(*) >= :min_n
    """, {"min_n": _MIN_N_FOR_PROFILE}).fetchall()

    result = []
    for r in rows:
        sport, mt, n_unbiased, wr, n_delta, mean_d, mean_sq, ccr, nlr = r

        sport_norm = normalize_sport(sport or "")
        mt_norm    = normalize_market(mt or "")

        # Line bucket from market_type alone (no line value in this aggregation)
        if mt_norm == "Moneyline":
            bucket = "ML"
        elif mt_norm in ("Spread", "Run Line", "Alt Spread"):
            bucket = "Spread"
        elif mt_norm in ("Total", "Alt Total"):
            bucket = "Total"
        elif mt_norm in ("Player Prop",):
            bucket = "Player Prop"
        else:
            bucket = mt_norm or "Other"

        # Compute std_delta and edge_ratio from mean/mean_sq aggregates
        mean_d_f = float(mean_d) if (mean_d is not None and n_delta >= _MIN_N_FOR_DELTA) else None
        std_d_f  = None
        er       = None
        if mean_d_f is not None and mean_sq is not None and n_delta >= 2:
            variance = float(mean_sq) - mean_d_f ** 2
            std_d_f  = math.sqrt(max(0.0, variance))
            if abs(mean_d_f) > 0.01:
                er = round(std_d_f / abs(mean_d_f), 3)

        ccr_f = float(ccr) if ccr is not None else None
        nlr_f = float(nlr) if nlr is not None else None

        grade    = compute_margin_grade(float(wr), mean_d_f, er, ccr_f,
                                      sample_size=int(n_unbiased))
        max_legs = compute_max_parlay_legs(mt_norm, float(wr), grade,
                                           mean_delta=mean_d_f, edge_ratio=er)

        result.append({
            "sport":            sport_norm,
            "market_type":      mt_norm,
            "line_bucket":      bucket,
            "sample_size":      int(n_unbiased),
            "personal_wr":      round(float(wr), 6),
            "mean_delta":       round(mean_d_f, 4) if mean_d_f is not None else None,
            "std_delta":        round(std_d_f, 4)  if std_d_f  is not None else None,
            "edge_ratio":       er,
            "close_call_rate":  round(ccr_f, 4)    if ccr_f    is not None else None,
            "narrow_loss_rate": round(nlr_f, 4)    if nlr_f    is not None else None,
            "margin_grade":     grade,
            "max_parlay_legs":  max_legs,
            "avg_odds":         None,
            "avg_ev":           None,
            "data_sources":     json.dumps({"bet_legs": int(n_unbiased),
                                            "n_delta": int(n_delta or 0)}),
        })
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def refresh_personal_edge_profiles(db=None, db_path: str = None) -> dict:
    """
    Rebuild personal_edge_profile from all available resolved leg data.

    Args:
        db:      SQLAlchemy Session (ignored — we open our own sqlite3 connection)
        db_path: path to bets.db (defaults to standard location)

    Returns a summary dict with counts of rows processed.
    """
    path = db_path or _DB_PATH
    conn = sqlite3.connect(path)
    try:
        _ensure_table(conn)
        profiles = _personal_rows(conn)

        # Seed rows: direction-split profiles for spread markets that have no real data.
        # These prevent positive-spread (+1.5/+2.5/+3.5) legs from inheriting the
        # AVOID grade earned by negative-spread (favourite-laying) losses.
        # Rows are protected from deletion below and inserted with OR IGNORE so real
        # data is never overwritten.
        _SEED_PROFILES: list[dict] = [
            {"sport": "Major League Baseball", "market_type": "Spread", "line_bucket": "+1.5",
             "margin_grade": "CLOSE",   "max_parlay_legs": 2},
            {"sport": "Major League Baseball", "market_type": "Spread", "line_bucket": "+2.5",
             "margin_grade": "NEUTRAL", "max_parlay_legs": 2},
            {"sport": "Major League Baseball", "market_type": "Spread", "line_bucket": "+3.5",
             "margin_grade": "NEUTRAL", "max_parlay_legs": 2},
            {"sport": "Ice Hockey",            "market_type": "Spread", "line_bucket": "+1.5",
             "margin_grade": "CLOSE",   "max_parlay_legs": 2},
            {"sport": "Ice Hockey",            "market_type": "Spread", "line_bucket": "+2.5",
             "margin_grade": "NEUTRAL", "max_parlay_legs": 2},
            {"sport": "Basketball",            "market_type": "Spread", "line_bucket": "-1.5",
             "margin_grade": "CLOSE",   "max_parlay_legs": 1},
            {"sport": "Basketball",            "market_type": "Spread", "line_bucket": "+1.5",
             "margin_grade": "CLOSE",   "max_parlay_legs": 2},
        ]
        _SEED_KEYS = {(s["sport"], s["market_type"], s["line_bucket"])
                      for s in _SEED_PROFILES}

        # Build key set for stale-row removal; protect seed keys from deletion
        current_keys = {(r["sport"], r["market_type"], r["line_bucket"]) for r in profiles}
        current_keys |= _SEED_KEYS   # seed rows are never deleted
        db_keys = conn.execute(
            "SELECT sport, market_type, line_bucket FROM personal_edge_profile"
        ).fetchall()
        deleted = 0
        for dk in db_keys:
            if dk not in current_keys:
                conn.execute(
                    "DELETE FROM personal_edge_profile "
                    "WHERE sport=? AND market_type=? AND line_bucket=?", dk
                )
                deleted += 1

        now = datetime.utcnow().isoformat()
        upserted = 0
        for row in profiles:
            params = {
                "sport":             row.get("sport"),
                "market_type":       row.get("market_type"),
                "line_bucket":       row.get("line_bucket"),
                "sample_size":       row.get("sample_size"),
                "personal_wr":       row.get("personal_wr"),
                "mean_delta":        row.get("mean_delta"),
                "std_delta":         row.get("std_delta"),
                "edge_ratio":        row.get("edge_ratio"),
                "close_call_rate":   row.get("close_call_rate"),
                "narrow_loss_rate":  row.get("narrow_loss_rate"),
                "margin_grade":      row.get("margin_grade"),
                "max_parlay_legs":   row.get("max_parlay_legs", 2),
                "avg_odds":          row.get("avg_odds"),
                "avg_ev":            row.get("avg_ev"),
                "data_sources":      row.get("data_sources"),
                "last_updated":      now,
            }
            conn.execute("""
                INSERT INTO personal_edge_profile
                    (sport, market_type, line_bucket, sample_size, personal_wr,
                     mean_delta, std_delta, edge_ratio, close_call_rate,
                     narrow_loss_rate, margin_grade, max_parlay_legs,
                     avg_odds, avg_ev, data_sources, last_updated)
                VALUES (:sport, :market_type, :line_bucket, :sample_size, :personal_wr,
                        :mean_delta, :std_delta, :edge_ratio, :close_call_rate,
                        :narrow_loss_rate, :margin_grade, :max_parlay_legs,
                        :avg_odds, :avg_ev, :data_sources, :last_updated)
                ON CONFLICT(sport, market_type, line_bucket) DO UPDATE SET
                    sample_size       = excluded.sample_size,
                    personal_wr       = excluded.personal_wr,
                    mean_delta        = excluded.mean_delta,
                    std_delta         = excluded.std_delta,
                    edge_ratio        = excluded.edge_ratio,
                    close_call_rate   = excluded.close_call_rate,
                    narrow_loss_rate  = excluded.narrow_loss_rate,
                    margin_grade      = excluded.margin_grade,
                    max_parlay_legs   = excluded.max_parlay_legs,
                    avg_odds          = excluded.avg_odds,
                    avg_ev            = excluded.avg_ev,
                    data_sources      = excluded.data_sources,
                    last_updated      = excluded.last_updated
            """, params)
            upserted += 1

        # Insert seed rows — OR IGNORE so real accumulated data is never overwritten
        seed_inserted = 0
        for seed in _SEED_PROFILES:
            conn.execute("""
                INSERT OR IGNORE INTO personal_edge_profile
                    (sport, market_type, line_bucket, sample_size,
                     margin_grade, max_parlay_legs, data_sources, last_updated)
                VALUES (?, ?, ?, 0, ?, ?, 'seed', ?)
            """, (seed["sport"], seed["market_type"], seed["line_bucket"],
                  seed["margin_grade"], seed["max_parlay_legs"], now))
            seed_inserted += conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    summary = {
        "profiles_built":  len(profiles),
        "upserted":        upserted,
        "seed_inserted":   seed_inserted,
        "deleted_stale":   deleted,
    }
    print(f"[PersonalEdge] Refresh complete: {summary}")
    return summary


def lookup_personal_profile(
    sport: str,
    market_type: str,
    line_bucket: str,
    db_path: str = None,
) -> Optional[dict]:
    """
    Look up one profile row by sport / market_type / line_bucket.
    Returns None when not found or table doesn't exist yet.

    Two-axis fallback: market_type first, then bucket.

    Market-type fallbacks (tried in order):
      1. Exact normalized market type (e.g. "Alt Spread")
      2. Base market type (e.g. "Spread" for "Alt Spread", "Total" for "Alt Total")
         _personal_rows() aggregates bet_legs by market_type without alt/main distinction,
         so "Alt Spread" MLB legs should inherit the "Spread" profile.

    Bucket fallbacks (within each market_type attempt):
      1. Exact bucket (e.g. "-1.5", "+2.5")
      2. Base market-type bucket (e.g. "Spread", "Total", "ML")
      3. "other" — catchall for unclassified buckets

    Example: MLB Alt Spread -1.5 leg →
      try (MLB, Alt Spread, -1.5) → miss
      try (MLB, Alt Spread, Spread) → miss
      try (MLB, Spread, -1.5) → miss
      try (MLB, Spread, Spread) → HIT → AVOID/0 ✓
    """
    path = db_path or _DB_PATH
    sport_norm = normalize_sport(sport)
    mt_norm    = normalize_market(market_type)

    # Market-type fallback: alt markets inherit their base market profile
    _MT_FALLBACKS: dict[str, str] = {
        "Alt Spread": "Spread",
        "Alt Total":  "Total",
    }
    mt_chain = [mt_norm]
    if mt_norm in _MT_FALLBACKS:
        mt_chain.append(_MT_FALLBACKS[mt_norm])

    # Bucket fallback sequence (computed once, reused for each mt_try)
    mt_base_bucket = {
        "Moneyline":  "ML",
        "Spread":     "Spread",
        "Alt Spread": "Spread",
        "Total":      "Total",
        "Alt Total":  "Total",
    }.get(mt_norm, line_bucket)

    buckets_to_try = []
    if line_bucket:
        buckets_to_try.append(line_bucket)
    if mt_base_bucket and mt_base_bucket != line_bucket:
        buckets_to_try.append(mt_base_bucket)
    if "other" not in buckets_to_try:
        buckets_to_try.append("other")

    _SELECT = """
        SELECT sport, market_type, line_bucket, sample_size,
               personal_wr, mean_delta, std_delta,
               edge_ratio, close_call_rate, narrow_loss_rate, margin_grade,
               max_parlay_legs, avg_odds, avg_ev, data_sources
        FROM   personal_edge_profile
        WHERE  sport = ? AND market_type = ? AND line_bucket = ?
    """

    try:
        conn = sqlite3.connect(path)
        row = None
        for mt_try in mt_chain:
            for bkt in buckets_to_try:
                row = conn.execute(_SELECT, (sport_norm, mt_try, bkt)).fetchone()
                if row:
                    break
            if row:
                break
        conn.close()
        if row:
            profile = {
                "sport":             row[0],  "market_type":      row[1],
                "line_bucket":       row[2],  "sample_size":      row[3],
                "personal_wr":       row[4],  "mean_delta":       row[5],
                "std_delta":         row[6],  "edge_ratio":       row[7],
                "close_call_rate":   row[8],  "narrow_loss_rate": row[9],
                "margin_grade":      row[10], "max_parlay_legs":  row[11],
                "avg_odds":          row[12], "avg_ev":           row[13],
                "data_sources":      row[14],
            }
            # Direction-aware AVOID override for spread markets.
            # AVOID grade on "Spread" bucket was earned from negative-point (favourite)
            # losses (e.g. MLB -1.5).  Positive-point underdog lines (+1.5, +2.5, +3.5,
            # or "other" > 0) are a completely different market direction — opposite risk.
            # Rather than inheriting AVOID, treat them as NEUTRAL (no penalty, 2-leg max).
            #
            # Only applies when:
            #   1. The matched profile row is AVOID (earned on favourite/negative spread)
            #   2. The ORIGINAL requested line_bucket is a positive-spread bucket
            #   3. Market is a spread type (Spread or Alt Spread)
            _POSITIVE_SPREAD_BUCKETS = frozenset({"+0.5", "+1.5", "+2.5", "+3.5"})
            if (profile.get("margin_grade") == "AVOID"
                    and line_bucket in _POSITIVE_SPREAD_BUCKETS
                    and normalize_market(market_type) in ("Spread", "Alt Spread")):
                return {
                    **profile,
                    "margin_grade":   "NEUTRAL",
                    "max_parlay_legs": 2,
                    "_synthetic":     True,   # flag: not real historical data
                }
            return profile
        return None
    except Exception:
        return None


def get_max_legs_for_personal_profile(
    sport: str,
    market_type: str,
    line_bucket: str,
) -> int:
    """
    Return the maximum parlay leg count allowed for this combo based on
    personal win-rate history AND margin grade.

    Margin grade (primary gate — overrides WR tier):
      AVOID        → 0 (blocked entirely — never include in any combo)
      CLOSE        → 1 (straight bet only — WR 60-74% or tight margins)
      MIXED        → 1 (volatile edge_ratio > 3.0 — 1-leg only)
      CUSHION/None → apply WR tier below

    WR tier (only reached when grade is CUSHION or unknown):
      ≥80% WR AND sample_size >= 10 → 3-leg parlays
      ≥75% WR → 2-leg parlays (minimum CUSHION threshold)
      no data → 2-leg default (conservative)

    CUSHION threshold is 75% (updated after analysis 2026-04-29).
    Only CUSHION-graded legs qualify for multi-leg parlays.
    """
    profile = lookup_personal_profile(sport, market_type, line_bucket)
    if profile is None or profile["personal_wr"] is None:
        return 2  # no profile: allow 2-leg but not 3-leg

    # Use the pre-computed stored value (set at refresh time by compute_max_parlay_legs)
    stored = profile.get("max_parlay_legs")
    if stored is not None:
        return int(stored)

    # Fallback: derive from grade if column missing (old rows before migration)
    grade = profile.get("margin_grade")
    if grade == "AVOID":  return 0
    if grade == "MIXED":  return 1
    if grade == "CLOSE":  return 2
    if grade == "CUSHION":
        wr     = profile.get("personal_wr") or 0.0
        md     = profile.get("mean_delta")
        er     = profile.get("edge_ratio")
        if (wr >= 0.75
                and md is not None and md > 1.0
                and er is not None and er < 2.5):
            return 5
        return 3
    return 2


def get_personal_edge_score(
    sport: str,
    market_type: str,
    line_bucket: str,
) -> Optional[float]:
    """
    Personal edge score = personal_wr × (1 + mean_delta / 10).
    Higher WR and larger win cushion → higher score.
    Returns None when no profile or sample_size < 5.
    Used by ALE to boost line candidates that match proven personal patterns.
    """
    profile = lookup_personal_profile(sport, market_type, line_bucket)
    if not profile or profile["personal_wr"] is None or profile["sample_size"] < 5:
        return None
    wr  = profile["personal_wr"]
    md  = profile["mean_delta"] or 0.0
    return wr * (1.0 + md / 10.0)


def get_margin_adjusted_los(
    sport: str,
    market_type: str,
    line_bucket: str,
    base_los: float,
) -> float:
    """
    Apply margin grade multiplier to a base Line Optimization Score (LOS).

    CUSHION → ×1.10  (comfortable wins; boost the line)
    CLOSE   → ×0.85  (narrow wins; penalize even if WR is high)
    MIXED   → ×0.92  (volatile; slight penalty)
    AVOID   → 0.0    (blocked — return 0 to remove from candidates)
    NEUTRAL → ×1.00  (no personal data — pass through unchanged, accumulating)
    No data → ×1.00  (no change)

    Caps at 1.0 after multiplication.
    """
    profile = lookup_personal_profile(sport, market_type, line_bucket)
    if profile is None:
        return base_los  # no profile: pass through unchanged

    grade = profile.get("margin_grade")
    if grade == "AVOID":
        return 0.0
    elif grade == "CUSHION":
        return round(min(1.0, base_los * 1.10), 4)
    elif grade == "CLOSE":
        return round(base_los * 0.85, 4)
    elif grade == "MIXED":
        return round(base_los * 0.92, 4)
    # NEUTRAL: no personal data — pass through unchanged (same as no data)
    return base_los


def get_contextual_wr_adjustment(
    team: str,
    sport: str,
    market_type: str,
    current_odds: Optional[float] = None,
    avg_profile_odds: Optional[float] = None,
    db_path: str = None,
) -> float:
    """
    Compute contextual win-rate adjustments (pp) for a specific team/game.

    Returns a float adjustment in probability points (e.g. +3.0 or -5.0).
    These are additive modifiers on top of the personal_wr base.

    Currently implemented:
      A. Team form (soccer only): +3pp for 4-5 wins / -5pp for 0-1 wins in last 5
      E. Odds value: +2pp if current_odds > avg_profile_odds (better value than usual)
    """
    path = db_path or _DB_PATH
    adjustment = 0.0

    # ── A. Soccer team form ───────────────────────────────────────────────────
    sport_norm = normalize_sport(sport)
    if sport_norm == "Soccer" and team:
        try:
            conn = sqlite3.connect(path)
            row = conn.execute("""
                SELECT wins_5
                FROM   team_soccer_form
                WHERE  team_name LIKE ?
                ORDER  BY as_of_date DESC
                LIMIT  1
            """, (f"%{team}%",)).fetchone()
            conn.close()
            if row and row[0] is not None:
                wins_5 = int(row[0])
                if wins_5 >= 4:
                    adjustment += 3.0
                elif wins_5 <= 1:
                    adjustment -= 5.0
        except Exception:
            pass

    # ── B. Rest days (NBA / NHL only) ────────────────────────────────────────
    # Back-to-back = 0 rest days: -15pp.  1 day rest: -5pp.  2+ days: no change.
    # Only fires for NBA and NHL where schedule fatigue is well-documented.
    if team and sport_norm in ("NBA", "NHL"):
        try:
            sport_keys = (
                ["basketball_nba"] if sport_norm == "NBA"
                else ["icehockey_nhl"]
            )
            sk_placeholders = ",".join("?" * len(sport_keys))
            conn = sqlite3.connect(path)
            row = conn.execute(f"""
                SELECT MAX(date(commence_time)) AS last_game
                FROM   fixtures
                WHERE  (home_team = ? OR away_team = ?)
                  AND  date(commence_time) < date('now', '-5 hours')
                  AND  sport_key IN ({sk_placeholders})
            """, [team, team] + sport_keys).fetchone()
            conn.close()
            if row and row[0]:
                import datetime as _dt
                last_game = _dt.date.fromisoformat(row[0])
                today = _dt.date.today()
                rest_days = (today - last_game).days - 1   # 0 = played yesterday
                if rest_days <= 0:
                    adjustment -= 15.0
                elif rest_days == 1:
                    adjustment -= 5.0
        except Exception:
            pass

    # ── E. Odds value vs historical avg ──────────────────────────────────────
    if (current_odds is not None and avg_profile_odds is not None
            and current_odds > 1.0 and avg_profile_odds > 1.0):
        # Better decimal odds → better payout for same probability → value bet
        if current_odds >= avg_profile_odds * 1.05:
            adjustment += 2.0

    return adjustment


def ensure_populated(db_path: str = None) -> None:
    """
    Called on server start: create table and populate if empty.
    """
    path = db_path or _DB_PATH
    try:
        conn = sqlite3.connect(path)
        _ensure_table(conn)
        n = conn.execute("SELECT COUNT(*) FROM personal_edge_profile").fetchone()[0]
        conn.close()
        if n == 0:
            refresh_personal_edge_profiles(db_path=path)
    except Exception as e:
        print(f"[PersonalEdge] ensure_populated error: {e}")
