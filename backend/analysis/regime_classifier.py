#!/usr/bin/env python3
"""
regime_classifier.py — Standalone BetIQ market regime classifier.

Reads today's picks_cache.json + bets.db to classify the market into one
of 5 regimes and suggest composite weights.

NEVER imported by main.py or any other production backend file.
Called as a subprocess by scheduler.py or run directly:

    python3 backend/analysis/regime_classifier.py
    python3 backend/analysis/regime_classifier.py --write-db
    python3 backend/analysis/regime_classifier.py --output data/regime_log.json

Read-only unless --write-db is passed (writes to market_regime_log table).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime

# ── Locate databases relative to this script ─────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_BETS_DB    = os.path.join(_ROOT, "data", "bets.db")
_CACHE_FILE = os.path.join(_ROOT, "data", "picks_cache.json")

# AUC per sport — mirrors recommender / mock_bets constants
_AUC_BY_SPORT: dict[str, float] = {
    "NHL": 0.7544,
    "MLB": 0.6360,
}
_AUC_DEFAULT = 0.6082

# ─────────────────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def get_weighted_model_confidence(db_path: str) -> dict:
    """
    Average model confidence across today's top legs, weighted by AUC of the
    model that scored each leg.  Covers all sports.
    """
    try:
        con = _connect(db_path)
        cur = con.cursor()
        cur.execute("""
            SELECT
                mbl.predicted_win_prob,
                mb.sport,
                CASE
                    WHEN mb.sport LIKE '%NHL%' THEN 0.7544
                    WHEN mb.sport LIKE '%MLB%' THEN 0.6360
                    ELSE 0.6082
                END AS model_auc
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
            WHERE DATE(mb.generated_at) = DATE('now')
              AND mbl.predicted_win_prob IS NOT NULL
            ORDER BY mbl.predicted_win_prob DESC
            LIMIT 40
        """)
        legs = [dict(r) for r in cur.fetchall()]
        con.close()
    except Exception as e:
        print(f"  [warn] confidence query failed: {e}", file=sys.stderr)
        return {"weighted_conf": 0.52, "breakdown": {}, "n_legs": 0}

    if not legs:
        return {"weighted_conf": 0.52, "breakdown": {}, "n_legs": 0}

    total_weight  = sum(l["model_auc"] for l in legs)
    weighted_conf = sum(
        (l["predicted_win_prob"] / 100) * l["model_auc"]
        for l in legs
    ) / total_weight

    breakdown: dict[str, dict] = {}
    for sport_raw in {l["sport"] for l in legs}:
        sport = (sport_raw or "combined").split("|")[0].strip()
        sl = [l for l in legs if l["sport"] == sport_raw]
        breakdown[sport] = {
            "legs":     len(sl),
            "avg_conf": round(sum(l["predicted_win_prob"] for l in sl) / len(sl), 1),
            "auc":      sl[0]["model_auc"],
        }

    return {
        "weighted_conf": round(weighted_conf, 4),
        "breakdown":     breakdown,
        "n_legs":        len(legs),
    }


def get_avg_implied_prob_today(db_path: str) -> float:
    """
    Average implied probability across all mock bet legs generated today.
    Uses the decimal odds stored on mock_bets to estimate market-implied win
    probability per leg.  High (>0.65) indicates heavy-favourite day.
    """
    try:
        con = _connect(db_path)
        cur = con.cursor()
        # Use per-leg win_prob as proxy for implied (reciprocal of odds)
        cur.execute("""
            SELECT mbl.win_prob
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
            WHERE DATE(mb.generated_at) = DATE('now')
              AND mbl.win_prob IS NOT NULL
        """)
        rows = cur.fetchall()
        con.close()
        if not rows:
            return 0.52
        probs = [r["win_prob"] / 100 for r in rows]
        return round(sum(probs) / len(probs), 4)
    except Exception:
        return 0.52


def classify_market_regime(db_path: str = _BETS_DB, cache_file: str = _CACHE_FILE) -> dict:
    """
    Classify today's market into one of 5 regimes.
    Returns classification + suggested weights (NOT applied to picks).
    """
    # ── Load picks pool assessment from cache ────────────────────────────────
    pool: dict = {}
    try:
        with open(cache_file) as f:
            payload = json.load(f)
        raw_picks = payload.get("picks", {})
        pool = raw_picks.get("pool_assessment", {})
    except Exception:
        pass

    legs_total   = pool.get("legs_evaluated",  0)
    legs_pos_ev  = pool.get("legs_positive_ev", 0)
    hq_legs      = pool.get("high_quality_legs", legs_pos_ev)  # fallback to pos_ev if not present
    pos_ev_pct   = legs_pos_ev / max(legs_total, 1)

    # ── Weighted model confidence ─────────────────────────────────────────────
    conf_data     = get_weighted_model_confidence(db_path)
    weighted_conf = conf_data["weighted_conf"]

    # ── Average implied probability ───────────────────────────────────────────
    avg_implied = get_avg_implied_prob_today(db_path)

    # ── Regime classification (priority order) ────────────────────────────────
    if legs_total < 20 or legs_pos_ev < 3:
        regime  = "sparse"
        weights = {"win_prob": 0.60, "ev": 0.10, "lqs": 0.30}
        note    = "Few legs available — anchor bets only"

    elif pos_ev_pct >= 0.15 and weighted_conf >= 0.60:
        regime  = "sharp"
        weights = {"win_prob": 0.45, "ev": 0.40, "lqs": 0.15}
        note    = "Strong multi-sport signal — trust model and EV"

    elif avg_implied > 0.65 and pos_ev_pct < 0.05:
        regime  = "efficient"
        weights = {"win_prob": 0.50, "ev": 0.10, "lqs": 0.40}
        note    = "Market efficient — lean on quality anchors"

    elif weighted_conf < 0.55 and pos_ev_pct < 0.10:
        regime  = "low_signal"
        weights = {"win_prob": 0.40, "ev": 0.20, "lqs": 0.40}
        note    = "Weak model signal across all sports — LQS anchor"

    else:
        regime  = "mixed"
        weights = {"win_prob": 0.50, "ev": 0.30, "lqs": 0.20}
        note    = "Mixed signals — standard weights apply"

    return {
        "date":    str(date.today()),
        "regime":  regime,
        "signals": {
            "legs_evaluated":            legs_total,
            "legs_positive_ev":          legs_pos_ev,
            "pos_ev_pct":                round(pos_ev_pct, 3),
            "hq_legs":                   hq_legs,
            "weighted_model_confidence": weighted_conf,
            "model_breakdown":           conf_data["breakdown"],
            "n_conf_legs":               conf_data["n_legs"],
            "avg_implied_prob":          round(avg_implied, 3),
        },
        "suggested_weights": weights,
        "current_weights":   {"win_prob": 0.50, "ev": 0.30, "lqs": 0.20},
        "weights_applied":   False,   # always False until Phase 2
        "note":              note,
    }


def write_regime_to_db(result: dict, db_path: str = _BETS_DB) -> None:
    """
    Upsert today's classification into market_regime_log.
    Creates the table if it doesn't exist yet.
    """
    sigs = result.get("signals", {})
    con = sqlite3.connect(db_path)
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS market_regime_log (
                date                      TEXT PRIMARY KEY,
                regime                    TEXT,
                legs_evaluated            INTEGER,
                legs_positive_ev          INTEGER,
                pos_ev_pct                REAL,
                hq_legs                   INTEGER,
                weighted_model_confidence REAL,
                avg_implied_prob          REAL,
                model_breakdown           TEXT,
                suggested_weights         TEXT,
                actual_weights            TEXT,
                mock_bets_generated       INTEGER,
                mock_win_rate             REAL,
                mock_pnl                  REAL,
                clv_avg                   REAL,
                note                      TEXT,
                created_at                TEXT
            )
        """)
        con.execute("""
            INSERT INTO market_regime_log
                (date, regime, legs_evaluated, legs_positive_ev, pos_ev_pct,
                 hq_legs, weighted_model_confidence, avg_implied_prob,
                 model_breakdown, suggested_weights, actual_weights,
                 note, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(date) DO UPDATE SET
                regime                    = excluded.regime,
                legs_evaluated            = excluded.legs_evaluated,
                legs_positive_ev          = excluded.legs_positive_ev,
                pos_ev_pct                = excluded.pos_ev_pct,
                hq_legs                   = excluded.hq_legs,
                weighted_model_confidence = excluded.weighted_model_confidence,
                avg_implied_prob          = excluded.avg_implied_prob,
                model_breakdown           = excluded.model_breakdown,
                suggested_weights         = excluded.suggested_weights,
                note                      = excluded.note
        """, (
            result["date"],
            result["regime"],
            sigs.get("legs_evaluated"),
            sigs.get("legs_positive_ev"),
            sigs.get("pos_ev_pct"),
            sigs.get("hq_legs"),
            sigs.get("weighted_model_confidence"),
            sigs.get("avg_implied_prob"),
            json.dumps(sigs.get("model_breakdown", {})),
            json.dumps(result.get("suggested_weights", {})),
            json.dumps(result.get("current_weights", {})),
            result.get("note"),
            datetime.utcnow().isoformat(),
        ))
        con.commit()
    finally:
        con.close()


def update_regime_settlement(db_path: str = _BETS_DB) -> dict:
    """
    After a settlement cycle, update today's market_regime_log row with
    actual win rate, P&L, and average CLV for today's settled bets.
    Called by scheduler after each settle run.
    """
    today = str(date.today())
    try:
        con = _connect(db_path)
        cur = con.cursor()

        # Today's settled bets
        cur.execute("""
            SELECT
                COUNT(*) AS n,
                SUM(CASE WHEN status='SETTLED_WIN' THEN 1 ELSE 0 END) AS wins,
                SUM(COALESCE(actual_profit, 0)) AS pnl
            FROM mock_bets
            WHERE DATE(settled_at) = ? AND status IN ('SETTLED_WIN','SETTLED_LOSS')
        """, (today,))
        row = cur.fetchone()
        n, wins, pnl = (row["n"] or 0), (row["wins"] or 0), (row["pnl"] or 0.0)
        wr  = round(wins / n, 4) if n else None

        # Average CLV from settled legs today
        cur.execute("""
            SELECT AVG(mbl.clv_cents) AS avg_clv
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
            WHERE DATE(mb.settled_at) = ? AND mbl.clv_available = 1
        """, (today,))
        clv_row = cur.fetchone()
        clv_avg = round(clv_row["avg_clv"], 1) if clv_row["avg_clv"] is not None else None

        cur.execute("""
            UPDATE market_regime_log
            SET mock_win_rate = ?, mock_pnl = ?, clv_avg = ?
            WHERE date = ?
        """, (wr, round(pnl, 2), clv_avg, today))
        con.commit()
        con.close()
        return {"date": today, "n": n, "win_rate": wr, "pnl": round(pnl, 2), "clv_avg": clv_avg}
    except Exception as e:
        return {"error": str(e)}


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_regime(result: dict) -> None:
    sigs = result.get("signals", {})
    sw   = result.get("suggested_weights", {})
    cw   = result.get("current_weights", {})
    bd   = sigs.get("model_breakdown", {})

    print("\n" + "═" * 60)
    print(f"MARKET REGIME — {result['date']}")
    print("═" * 60)
    print(f"  Regime:   {result['regime'].upper()}")
    print(f"  Note:     {result['note']}")
    print()
    print(f"  Signals:")
    print(f"    legs_evaluated      = {sigs.get('legs_evaluated')}")
    print(f"    legs_positive_ev    = {sigs.get('legs_positive_ev')}  ({sigs.get('pos_ev_pct', 0)*100:.1f}%)")
    print(f"    hq_legs             = {sigs.get('hq_legs')}")
    print(f"    weighted_confidence = {sigs.get('weighted_model_confidence'):.4f}")
    print(f"    avg_implied_prob    = {sigs.get('avg_implied_prob'):.3f}")
    print()

    if bd:
        print(f"  Model breakdown:")
        for sport, info in bd.items():
            print(f"    {sport:<12} legs={info['legs']:>3}  avg_conf={info['avg_conf']:>5.1f}%  auc={info['auc']:.4f}")
        print()

    print(f"  Suggested weights:  WP={sw.get('win_prob'):.0%}  EV={sw.get('ev'):.0%}  LQS={sw.get('lqs'):.0%}")
    print(f"  Current weights:    WP={cw.get('win_prob'):.0%}  EV={cw.get('ev'):.0%}  LQS={cw.get('lqs'):.0%}")
    print(f"  weights_applied:    {result.get('weights_applied')}  ← informational only")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BetIQ regime classifier — standalone, read-only"
    )
    parser.add_argument(
        "--write-db", action="store_true",
        help="Write classification to market_regime_log table in bets.db",
    )
    parser.add_argument(
        "--output", "-o", metavar="FILE",
        help="Append JSON result to this file",
    )
    parser.add_argument(
        "--db", default=_BETS_DB, metavar="PATH",
        help=f"SQLite bets database (default: {_BETS_DB})",
    )
    parser.add_argument(
        "--cache", default=_CACHE_FILE, metavar="PATH",
        help=f"picks_cache.json path (default: {_CACHE_FILE})",
    )
    args = parser.parse_args()

    db_path = args.db if os.path.isabs(args.db) else os.path.join(_ROOT, args.db)
    cache   = args.cache if os.path.isabs(args.cache) else os.path.join(_ROOT, args.cache)

    print(f"\nBetIQ Regime Classifier  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Database:   {db_path}")
    print(f"Cache file: {cache}")

    result = classify_market_regime(db_path=db_path, cache_file=cache)
    print_regime(result)

    if args.write_db:
        write_regime_to_db(result, db_path=db_path)
        print(f"\n  ✓ Written to market_regime_log (date={result['date']})")

    if args.output:
        out_path = (args.output if os.path.isabs(args.output)
                    else os.path.join(_ROOT, args.output))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        log: list = []
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                    log = existing if isinstance(existing, list) else [existing]
            except Exception:
                pass
        log.append(result)
        with open(out_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"  Results appended → {out_path}")

    print()
    return result


if __name__ == "__main__":
    main()
