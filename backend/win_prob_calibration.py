"""
win_prob_calibration.py — Data-driven, market-type-aware win probability calibration.

Replaces the hardcoded Phase-A multipliers in recommender.calibrated_wp() with
per-(sport, market_type, wp_bucket) corrections derived from settled mock_bet_legs.

How it works
────────────
1. build_calibration_table(bets_db_path) queries mock_bet_legs and computes:
     haircut_pp = avg(win_prob) − actual_wr   (positive = overconfident)
   for each (sport, market_type, wp_bucket) slice with ≥ 10 resolved legs.

2. get_calibrated_wp(raw_wp, sport, market_type, bets_db_path) looks up the
   correction and applies it, confidence-weighted by sample size:
     correction = haircut_pp × confidence      (confidence = min(1.0, n/100))
     calibrated = raw_wp − correction

3. Alt lines (alternate_spreads, alternate_totals) are exempted — their win_prob
   is the market-implied probability (1/odds), which IS the calibration signal.

4. refresh_calibration_table() is called nightly after settlement propagation
   so corrections tighten as data accumulates (precise after 30+ days).

Fallback
────────
If the table is empty or the specific bucket has n < 30, the function falls back
to sport-level default haircuts (conservative values from Phase A diagnostic).
These defaults are kept accurate: the old NHL ×0.40 disaster zone has been removed
because it applied a spread-market correction to h2h legs that win 91%.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── Alt-line market types: skip calibration (market implied prob, not model output) ──
# alternate_totals remain exempt — no usable calibration data and the market odds
# ARE the correct signal for over/under total lines.
# alternate_spreads are NO LONGER exempt: post-normalization we have DB entries for
# MLB (n=281-162), NBA (n=275-204), NHL (n=38-28), and aggregate soccer fallbacks.
_ALT_MARKETS: frozenset[str] = frozenset({"alternate_totals"})

# Soccer leagues stored as sport in mock_bet_legs (no canonical normalization)
_SOCCER_LEAGUES: frozenset[str] = frozenset({
    "EPL", "La Liga", "Ligue 1", "Serie A", "Bundesliga",
    "UCL", "MLS", "Soccer", "soccer_epl",
})

# ── WP bucket boundaries (lower-inclusive, upper-exclusive) ──────────────────
_BUCKETS: list[tuple[float, float, str]] = [
    (0.0,  45.0, "<45"),
    (45.0, 55.0, "45-55"),
    (55.0, 65.0, "55-65"),
    (65.0, 75.0, "65-75"),
    (75.0, 85.0, "75-85"),
    (85.0, 100.1, "85+"),
]

# ── Fallback sport-level defaults (conservative, used when n < 30) ────────────
# These replace the old hardcoded multipliers.  Key changes vs Phase A version:
#   • Removed NHL ×0.40 disaster zone — derived from banned spreads, wrongly hit h2h.
#   • NHL h2h 55-65% actual WR = 90.9% → neutral correction (let DB take over).
#   • NBA h2h 55-65% actual WR = 90.9% → neutral correction.
#
# Format: (sport_normalised, market_type) → haircut_pp by bucket.
# Positive = overconfident (reduce WP). Negative = underconfident (boost WP).
# None = no correction (trust the model for this bucket).
_FALLBACK_HAIRCUTS: dict[tuple[str, str], dict[str, float]] = {
    ("MLB", "h2h"): {
        "<45":    0.0,
        "45-55":  3.6,   # n=120, mild overconfidence
        "55-65":  21.4,  # n=30,  significant overconfidence (blocked by min_wp=75 gate anyway)
        "65-75":  0.0,   # n=24,  model underconfident (actual 83%) — don't haircut
        "75-85":  7.9,   # Phase A: 75%+ bucket calibrated at 70.5% WR
        "85+":    5.0,   # conservative
    },
    ("NBA", "h2h"): {
        "<45":    6.0,   # n=29, mild overconfidence
        "45-55":  0.0,   # no data
        "55-65":  0.0,   # n=11, actual WR 90.9% — model underconfident, don't cut
        "65-75":  12.7,  # Phase A: 12.7pp overconfident
        "75-85":  5.9,   # Phase A: mostly calibrated
        "85+":    5.0,
    },
    ("NHL", "h2h"): {
        "<45":    0.0,
        "45-55":  15.4,  # n=26, significant overconfidence (also blocked by min_wp=55 gate)
        "55-65":  0.0,   # n=11, actual WR 90.9% — model underconfident, do NOT reduce
        "65-75":  0.0,   # no data — conservative, don't apply stale spread-era correction
        "75-85":  8.7,   # Phase A
        "85+":    5.0,
    },
    # Spreads are banned — entries here are for completeness / future use
    ("MLB", "spreads"):  {"45-55": 30.0, "55-65": 30.0, "65-75": 23.0, "75-85": 20.0, "85+": 15.0},
    ("NHL", "spreads"):  {"45-55": 30.0, "55-65": 30.0, "65-75": 25.0, "75-85": 20.0, "85+": 15.0},
    ("NBA", "spreads"):  {"45-55": 10.0, "55-65": 10.0, "65-75": 8.0,  "75-85": 6.0,  "85+": 4.0},
    # Totals — small samples, conservative universal correction
    ("MLB", "totals"):   {"55-65": 15.0, "65-75": 10.0, "75-85": 7.9, "85+": 5.0},
    ("NBA", "totals"):   {"55-65": 12.0, "65-75": 8.0,  "75-85": 5.0, "85+": 3.0},
    ("NHL", "totals"):   {"55-65": 15.0, "65-75": 10.0, "75-85": 8.0, "85+": 5.0},
    # Alt-spreads — derived from normalized win_prob data (all sports, 2026-04-29+).
    # MLB/NBA/NHL will primarily use DB entries (n=162-281 per major bucket).
    # These fallbacks kick in when the bucket has n < 30 or no DB row yet.
    ("MLB", "alternate_spreads"): {
        "<45":   3.0,    # n=61,  avg_pred=33.89, actual=31.15  (+2.74 in DB)
        "45-55": 14.0,   # n=20,  avg_pred=48.60, actual=35.00  (+13.60 in DB)
        "55-65": 0.0,    # n=57,  underconfident (-2.49) — don't haircut
        "65-75": 2.0,    # n=162, well-calibrated  (+2.38 in DB)
        "75-85": 3.0,    # n=281, well-calibrated  (+3.34 in DB)
        "85+":   1.0,    # n=43,  near-perfect     (+0.49 in DB)
    },
    ("NBA", "alternate_spreads"): {
        "<45":   15.0,   # n=13, too small for DB — conservative
        "45-55": 5.0,    # no DB data — conservative
        "55-65": 2.0,    # n=16, near neutral
        "65-75": 16.0,   # n=204, overconfident (+16.43 in DB)
        "75-85": 10.0,   # n=275, overconfident (+9.85 in DB)
        "85+":   5.0,    # conservative
    },
    ("NHL", "alternate_spreads"): {
        "<45":   0.0,    # n=16, near neutral
        "45-55": 5.0,    # no DB data — conservative
        "55-65": 5.0,    # no DB data — conservative
        "65-75": 13.0,   # n=28, overconfident (+13.39 in DB)
        "75-85": 0.0,    # n=38, underconfident (-2.90) — don't haircut
        "85+":   0.0,    # n=19, well-calibrated
    },
    # Soccer alt_spreads — aggregate across all leagues (n=24-42 per bucket).
    # Used as fallback for EPL, La Liga, Ligue 1, Serie A, Bundesliga, UCL, MLS.
    # DB lookup uses per-league sport names; if n < 30, falls through to these.
    ("Soccer", "alternate_spreads"): {
        "<45":   0.0,    # n=24, slight underconfidence (-2.64) — neutral
        "45-55": 14.0,   # n=19, +14.16pp overconfident
        "55-65": 15.0,   # n=9,  +15.42pp (small sample, conservative)
        "65-75": 2.0,    # n=18, nearly perfect (+1.83)
        "75-85": 21.0,   # n=26, +21.28pp overconfident
        "85+":   9.0,    # n=42, +8.68pp overconfident
    },
}
_FALLBACK_DEFAULT_HAIRCUT = 5.0   # for unrecognised (sport, market) combos


def _normalize_sport(sport: str) -> str:
    """Map display sport names → canonical keys matching recommender._normalize_sport."""
    mapping = {
        "Basketball":        "NBA",
        "basketball_nba":    "NBA",
        "American Football": "NFL",
        "Baseball":          "MLB",
        "baseball_mlb":      "MLB",
        "Ice Hockey":        "NHL",
        "icehockey_nhl":     "NHL",
        "Soccer":            "Soccer",
        "soccer_epl":        "Soccer",
    }
    return mapping.get(sport, sport.upper() if sport else "")


def _get_bucket(wp: float) -> str:
    for lo, hi, label in _BUCKETS:
        if lo <= wp < hi:
            return label
    return "85+"


# ── Table management ──────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS win_prob_calibration (
    sport                  TEXT NOT NULL,
    market_type            TEXT NOT NULL,
    wp_bucket              TEXT NOT NULL,
    observed_predicted_wp  REAL,
    observed_actual_wr     REAL,
    calibration_haircut_pp REAL,
    sample_size            INTEGER,
    confidence             REAL,
    last_updated           TEXT NOT NULL,
    PRIMARY KEY (sport, market_type, wp_bucket)
)
"""


def _ensure_table(con: sqlite3.Connection) -> None:
    con.execute(_CREATE_SQL)
    con.commit()


def build_calibration_table(
    bets_db_path: str,
    since_date: str = "2026-04-29",
    min_legs: int = 10,
) -> dict:
    """
    Compute per-(sport, market_type, wp_bucket) calibration corrections from
    settled mock_bet_legs and write them to win_prob_calibration in bets.db.

    Parameters
    ──────────
    bets_db_path  Path to data/bets.db.
    since_date    Only use legs from this game_date onward (YYYY-MM-DD).
    min_legs      Minimum resolved legs to write a row (rows below this
                  threshold are skipped — not enough data to calibrate).

    Returns summary dict with counts.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(bets_db_path)
    _ensure_table(con)

    rows = con.execute("""
        SELECT
            mbl.sport,
            mbl.market_type,
            CASE
                WHEN mbl.win_prob < 45  THEN '<45'
                WHEN mbl.win_prob < 55  THEN '45-55'
                WHEN mbl.win_prob < 65  THEN '55-65'
                WHEN mbl.win_prob < 75  THEN '65-75'
                WHEN mbl.win_prob < 85  THEN '75-85'
                ELSE '85+'
            END                                                       AS wp_bucket,
            COUNT(*)                                                   AS n,
            AVG(mbl.win_prob)                                          AS avg_pred,
            100.0 * SUM(CASE WHEN mbl.leg_result = 'WIN' THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0)                                  AS actual_wr
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN', 'LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.game_date  >= ?
          AND mbl.win_prob  IS NOT NULL
        GROUP BY mbl.sport, mbl.market_type, wp_bucket
        HAVING COUNT(*) >= ?
    """, (since_date, min_legs)).fetchall()

    written = skipped = 0
    for sport, market_type, bucket, n, avg_pred, actual_wr in rows:
        haircut = round(avg_pred - actual_wr, 4)
        conf    = round(min(1.0, n / 100.0), 4)
        con.execute("""
            INSERT INTO win_prob_calibration
                (sport, market_type, wp_bucket,
                 observed_predicted_wp, observed_actual_wr,
                 calibration_haircut_pp, sample_size, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sport, market_type, wp_bucket) DO UPDATE SET
                observed_predicted_wp  = excluded.observed_predicted_wp,
                observed_actual_wr     = excluded.observed_actual_wr,
                calibration_haircut_pp = excluded.calibration_haircut_pp,
                sample_size            = excluded.sample_size,
                confidence             = excluded.confidence,
                last_updated           = excluded.last_updated
        """, (sport, market_type, bucket,
              round(avg_pred, 4), round(actual_wr, 4),
              haircut, n, conf, now_iso))
        written += 1

    con.commit()
    con.close()
    log.info(f"[WPCal] calibration table refreshed: {written} rows written, {skipped} skipped")
    return {"written": written, "skipped": skipped, "since": since_date}


def refresh_calibration_table(bets_db_path: str) -> dict:
    """Incremental refresh — same as build, but always uses the last 30 days."""
    from datetime import timedelta
    since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    return build_calibration_table(bets_db_path, since_date=since, min_legs=10)


# ── Public API ────────────────────────────────────────────────────────────────

def get_calibrated_wp(
    raw_wp: float,
    sport: str,
    market_type: str,
    bets_db_path: Optional[str] = None,
) -> float:
    """
    Return calibrated win probability for a single leg, applying the
    data-driven correction from win_prob_calibration (if available).

    Rules
    ─────
    • Alt-line markets (alternate_spreads / alternate_totals) are exempted —
      their win_prob is the market-implied probability, which IS the calibration.
    • If the DB row has sample_size < 30, falls back to hardcoded defaults.
    • Correction is confidence-weighted: correction = haircut_pp × confidence.
    • Result is clamped to [5.0, 95.0].
    """
    if not raw_wp:
        return 0.0
    if (market_type or "").lower() in _ALT_MARKETS:
        return raw_wp  # alt lines: market-implied probability is correct signal

    norm_sport = _normalize_sport(sport)
    bucket     = _get_bucket(raw_wp)

    # ── Try DB lookup ──────────────────────────────────────────────────────────
    haircut: Optional[float] = None
    confidence = 0.0

    if bets_db_path and os.path.exists(bets_db_path):
        try:
            con = sqlite3.connect(bets_db_path)
            row = con.execute("""
                SELECT calibration_haircut_pp, sample_size, confidence
                FROM win_prob_calibration
                WHERE sport = ? AND market_type = ? AND wp_bucket = ?
            """, (norm_sport, market_type, bucket)).fetchone()
            con.close()
            if row and row[1] >= 30:
                haircut, confidence = row[0], row[2]
        except Exception as exc:
            log.debug(f"[WPCal] DB lookup failed: {exc}")

    # ── Fallback to hardcoded defaults ─────────────────────────────────────────
    if haircut is None:
        mkt_lower = (market_type or "").lower()
        key       = (norm_sport, mkt_lower)
        defaults  = _FALLBACK_HAIRCUTS.get(key, {})
        raw_cut   = defaults.get(bucket)
        # Secondary fallback: for soccer leagues not in _FALLBACK_HAIRCUTS keys,
        # try the generic ("Soccer", market_type) entry.
        if raw_cut is None and sport in _SOCCER_LEAGUES and norm_sport not in {"MLB", "NBA", "NHL"}:
            soccer_defaults = _FALLBACK_HAIRCUTS.get(("Soccer", mkt_lower), {})
            raw_cut = soccer_defaults.get(bucket, soccer_defaults.get("default"))
        if raw_cut is None:
            raw_cut = defaults.get("default", _FALLBACK_DEFAULT_HAIRCUT)
        haircut    = raw_cut
        confidence = 0.5   # half-weight for fallback values

    correction = haircut * confidence
    calibrated = raw_wp - correction
    return max(5.0, min(95.0, round(calibrated, 4)))


# ─── Parlay Calibration Table ─────────────────────────────────────────────────
# Tracks parlay-level WP overconfidence in 10% buckets.
# Built from settled mock_bets (top_picks_page / prospective / prospective_pm).
# Applied at display time — raw predicted_win_prob stays in DB, calibrated value
# is returned by get_calibrated_parlay_wp() for frontend and scoring purposes.

_PARLAY_CAL_CREATE = """
CREATE TABLE IF NOT EXISTS parlay_calibration (
    wp_bucket          TEXT PRIMARY KEY,   -- e.g. "20-30"
    bucket_lo          REAL NOT NULL,
    bucket_hi          REAL NOT NULL,
    observed_pred_wp   REAL,
    observed_actual_wr REAL,
    haircut_pp         REAL,
    sample_size        INTEGER DEFAULT 0,
    confidence         REAL DEFAULT 0.0,
    last_updated       TEXT
);
"""

_MIN_PARLAY_SAMPLES = 20    # require at least this many parlays per bucket


def build_parlay_calibration_table(bets_db_path: Optional[str] = None) -> dict:
    """
    Compute per-bucket parlay WP haircuts from settled mock_bets and write them
    into the parlay_calibration table.  Called nightly after settlement.

    Returns a summary dict with buckets written and total parlays used.
    """
    if not bets_db_path or not os.path.exists(bets_db_path):
        return {"error": "bets_db_path not found", "written": 0}

    try:
        con = sqlite3.connect(bets_db_path, timeout=30)
        con.execute(_PARLAY_CAL_CREATE)

        rows = con.execute("""
            SELECT predicted_win_prob, status
            FROM mock_bets
            WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND source IN ('top_picks_page', 'prospective', 'prospective_pm')
              AND predicted_win_prob IS NOT NULL
        """).fetchall()

        if not rows:
            con.close()
            return {"written": 0, "n_parlays": 0}

        from collections import defaultdict
        buckets: dict[int, list] = defaultdict(list)
        for wp, status in rows:
            b_lo = min(int(float(wp) / 10) * 10, 90)
            buckets[b_lo].append((float(wp), 1.0 if status == 'SETTLED_WIN' else 0.0))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        written = 0

        for b_lo in range(0, 100, 10):
            b_hi  = b_lo + 10
            label = f"{b_lo}-{b_hi}"
            items = buckets.get(b_lo, [])
            n     = len(items)
            if n < _MIN_PARLAY_SAMPLES:
                continue
            avg_pred = sum(wp for wp, _ in items) / n
            actual_wr = sum(y for _, y in items) / n * 100.0  # in pp
            haircut  = round(avg_pred - actual_wr, 4)
            conf     = round(min(1.0, n / 100), 4)

            con.execute("""
                INSERT INTO parlay_calibration
                    (wp_bucket, bucket_lo, bucket_hi,
                     observed_pred_wp, observed_actual_wr, haircut_pp,
                     sample_size, confidence, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(wp_bucket) DO UPDATE SET
                    observed_pred_wp   = excluded.observed_pred_wp,
                    observed_actual_wr = excluded.observed_actual_wr,
                    haircut_pp         = excluded.haircut_pp,
                    sample_size        = excluded.sample_size,
                    confidence         = excluded.confidence,
                    last_updated       = excluded.last_updated
            """, (label, b_lo, b_hi, round(avg_pred, 4), round(actual_wr, 4),
                  haircut, n, conf, now_str))
            written += 1

        con.commit()
        con.close()
        log.info(f"[ParlayCalibration] Wrote {written} bucket rows from {len(rows)} parlays.")
        return {"written": written, "n_parlays": len(rows)}

    except Exception as exc:
        log.error(f"[ParlayCalibration] build failed: {exc}")
        return {"error": str(exc), "written": 0}


def get_calibrated_parlay_wp(raw_wp: float, bets_db_path: Optional[str] = None) -> float:
    """
    Return calibrated parlay win probability by applying the per-bucket haircut
    from the parlay_calibration table.

    Falls back to a conservative 15pp haircut if no DB entry exists.
    Confidence-weighted: correction = haircut_pp × confidence.
    Clamped to [2.0, 90.0] (parlays are inherently low-probability events).
    """
    if not raw_wp:
        return 0.0

    b_lo  = min(int(raw_wp / 10) * 10, 90)
    label = f"{b_lo}-{b_lo + 10}"

    haircut: Optional[float] = None
    confidence = 0.0

    if bets_db_path and os.path.exists(bets_db_path):
        try:
            con = sqlite3.connect(bets_db_path, timeout=15)
            row = con.execute("""
                SELECT haircut_pp, sample_size, confidence
                FROM parlay_calibration
                WHERE wp_bucket = ?
            """, (label,)).fetchone()
            con.close()
            if row and row[1] >= _MIN_PARLAY_SAMPLES:
                haircut, confidence = row[0], row[2]
        except Exception as exc:
            log.debug(f"[ParlayCalibration] DB lookup failed: {exc}")

    if haircut is None:
        haircut    = 15.0   # conservative fallback
        confidence = 0.5

    correction = haircut * confidence
    calibrated = raw_wp - correction
    return max(2.0, min(90.0, round(calibrated, 4)))
