#!/usr/bin/env python3
"""
signal_analysis.py — Standalone BetIQ signal analysis.

Reads settled mock bets from data/bets.db (mock_bets / mock_bet_legs tables)
and optionally enriches with CLV data from data/historical.db (betting_lines).

NEVER imported by main.py or any production backend file.
Runs standalone only:

    python3 backend/analysis/signal_analysis.py
    python3 backend/analysis/signal_analysis.py --output data/analysis_log.json
    python3 backend/analysis/signal_analysis.py --db data/bets.db --output data/analysis_log.json

Safe to run at any time — read-only, no DB writes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta

# ── Locate databases relative to this script ─────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_DB_PATH     = os.path.join(_ROOT, "data", "bets.db")
_HIST_DB     = os.path.join(_ROOT, "data", "historical.db")

ATS_FIX_CUTOFF     = "2026-04-23 20:00:00"   # UTC — ATS inversion bug fixed here
SIM_START_DATE     = "2026-04-22"            # First day BetIQ generated simulation bets
WEIGHT_CHANGE_DATE = "2026-04-26"            # LQS weights updated: A 40→22%, B 30→44%
SYSTEM_LAUNCH_DATE = "2026-04-29"            # CUSHION/AVOID margin grades + personal_edge_profile live

# All source values that represent genuine simulation bets
SIM_SOURCES = frozenset({
    "prospective",
    "prospective_pm",
    "top_picks_page",
    "forced_generation",
    "retroactive_mock",
    "prospective_legacy",   # Tier 3 — morning/stale bets superseded by fresh generation
    "scenario_sim",         # Tier 2 — human-hypothesis testing; directional accuracy analysis
    "exploration",          # Adjacent-line probes; settles automatically but excluded from T1/T2/T3
})

# Tier 2: human-curated scenario bets tracked separately from model-generated bets
T2_SOURCES = frozenset({"scenario_sim"})

# Exploration bets: adjacent-line single-leg probes generated alongside Section A picks.
# Excluded from T1/T2/T3 performance metrics; feed line quality analysis ONLY.
EXPLORATION_SOURCES = frozenset({"exploration"})


# ── Math helpers (no scipy dependency) ───────────────────────────────────────

def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num  = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dsx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dsy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dsx == 0 or dsy == 0:
        return None
    return round(num / (dsx * dsy), 4)


def quintile_buckets(rows: list[dict], signal_key: str) -> list[dict]:
    """Split rows into 5 equal-size buckets by signal_key, compute win stats."""
    valid = [r for r in rows if r.get(signal_key) is not None]
    if not valid:
        return []
    valid.sort(key=lambda r: r[signal_key])
    n   = len(valid)
    q   = n / 5
    out = []
    labels = ["Bottom 20%", "20–40%", "40–60%", "60–80%", "Top 20%"]
    for i in range(5):
        lo     = round(i * q)
        hi     = round((i + 1) * q)
        bucket = valid[lo:hi]
        if not bucket:
            continue
        wins   = sum(r["won"] for r in bucket)
        pnl    = sum(r["actual_profit"] or 0 for r in bucket)
        sig_lo = min(r[signal_key] for r in bucket)
        sig_hi = max(r[signal_key] for r in bucket)
        out.append({
            "label":        labels[i],
            "n":            len(bucket),
            "win_rate":     round(wins / len(bucket) * 100, 1),
            "avg_pnl":      round(pnl / len(bucket), 2),
            "signal_range": (round(sig_lo, 2), round(sig_hi, 2)),
        })
    return out


def _is_monotone(buckets: list[dict]) -> bool:
    rates = [b["win_rate"] for b in buckets]
    return all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


def power_n(p0: float = 0.25, delta: float = 0.05,
            alpha: float = 0.05, power: float = 0.80) -> int:
    """Minimum n to detect a `delta` improvement at given alpha and power."""
    z_alpha = 1.645   # one-sided α = 0.05
    z_beta  = 0.842   # power = 0.80
    p1      = p0 + delta
    p_bar   = (p0 + p1) / 2
    n       = ((z_alpha + z_beta) ** 2 * p_bar * (1 - p_bar)) / (delta ** 2)
    return math.ceil(n)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rows(db_path: str) -> list[dict]:
    """Load settled mock bets with the signals we need."""
    if not os.path.exists(db_path):
        print(f"[error] Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Check which columns exist (schema evolves over time)
    cur.execute("PRAGMA table_info(mock_bets)")
    cols = {row["name"] for row in cur.fetchall()}

    avg_lqs_sel = "avg_lqs" if "avg_lqs" in cols else "NULL AS avg_lqs"
    ev_sel      = "predicted_ev" if "predicted_ev" in cols else "NULL AS predicted_ev"
    wp_sel      = "predicted_win_prob" if "predicted_win_prob" in cols else "NULL AS predicted_win_prob"

    cur.execute(f"""
        SELECT
            id,
            generated_at,
            COALESCE(source, 'prospective') AS source,
            status,
            {avg_lqs_sel},
            {wp_sel},
            {ev_sel},
            actual_profit,
            COALESCE(amount, 10.0) AS amount,
            legs
        FROM mock_bets
        WHERE COALESCE(source, 'prospective') IN (
            'prospective', 'top_picks_page', 'forced_gen', 'tier_b', 'prospective_pm',
            'forced_generation', 'retroactive_mock', 'prospective_legacy', 'scenario_sim'
        )
          AND status IN ('SETTLED_WIN', 'SETTLED_LOSS')
        ORDER BY generated_at
    """)
    rows = []
    for r in cur.fetchall():
        rows.append({
            "id":              r["id"],
            "generated_at":    r["generated_at"],
            "source":          r["source"],
            "status":          r["status"],
            "won":             1 if r["status"] == "SETTLED_WIN" else 0,
            "avg_lqs":         r["avg_lqs"],
            "win_prob":        r["predicted_win_prob"],
            "expected_profit": r["predicted_ev"],
            "actual_profit":   r["actual_profit"],
            "amount":          r["amount"],
            "legs":            r["legs"],
        })
    con.close()
    return rows


def load_leg_rows(db_path: str) -> list[dict]:
    """Load per-leg signals from mock_bet_legs for supplemental analysis."""
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    try:
        cur.execute("""
            SELECT
                ml.mock_bet_id,
                ml.description,
                ml.market_type,
                ml.sport,
                ml.win_prob,
                ml.ev,
                ml.grade,
                ml.leg_result,
                mb.status AS bet_status,
                mb.generated_at,
                mb.avg_lqs
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND ml.leg_result IS NOT NULL
            ORDER BY mb.generated_at
        """)
        legs = []
        for r in cur.fetchall():
            legs.append({
                "mock_bet_id":  r["mock_bet_id"],
                "description":  r["description"],
                "market_type":  r["market_type"],
                "sport":        r["sport"],
                "win_prob":     r["win_prob"],
                "ev":           r["ev"],
                "grade":        r["grade"],
                "leg_result":   r["leg_result"],
                "won":          1 if r["leg_result"] == "won" else 0,
                "generated_at": r["generated_at"],
                "avg_lqs":      r["avg_lqs"],
            })
        return legs
    except Exception as e:
        print(f"  [warn] leg load failed: {e}", file=sys.stderr)
        return []
    finally:
        con.close()


# ── Analyses ──────────────────────────────────────────────────────────────────

def _load_component_proxies(db_path: str) -> list[dict]:
    """
    Load per-leg proxy signals from mock_bet_legs for component-level analysis.

    Proxies:
      B (model confidence) → predicted_win_prob
      D (odds edge)        → predicted_edge_pp

    Component A (historical accuracy) has no direct per-leg column — the
    bias it introduces is visible indirectly via avg_lqs on the parent bet.

    Returns rows with bet_won (parlay outcome) and leg_won (individual result).
    Only includes simulation bets (source IN SIM_SOURCES, generated_at >= SIM_START_DATE).
    """
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    src_ph = ",".join("?" * len(SIM_SOURCES))
    try:
        cur.execute(f"""
            SELECT
                ml.predicted_win_prob   AS comp_b,
                ml.predicted_edge_pp    AS comp_d,
                CASE WHEN ml.leg_result = 'WIN'  THEN 1
                     WHEN ml.leg_result = 'LOSS' THEN 0
                     ELSE NULL END       AS leg_won,
                CASE WHEN mb.status = 'SETTLED_WIN'  THEN 1
                     WHEN mb.status = 'SETTLED_LOSS' THEN 0
                     ELSE NULL END       AS bet_won,
                mb.avg_lqs
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE mb.source IN ({src_ph})
              AND mb.generated_at >= ?
              AND mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND ml.leg_result IN ('WIN', 'LOSS')
        """, list(SIM_SOURCES) + [SIM_START_DATE])
        rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"  [warn] component proxy load failed: {e}", file=sys.stderr)
        rows = []
    finally:
        con.close()
    return rows


def analysis_1_correlations(rows: list[dict], db_path: str | None = None) -> dict:
    """Pearson correlation of signals with outcome (won=0/1)."""
    valid = [r for r in rows if r["avg_lqs"] is not None]
    n = len(valid)

    won  = [r["won"]                   for r in valid]
    lqs  = [r["avg_lqs"]               for r in valid]
    wp   = [r["win_prob"]   or 0       for r in valid]
    ev   = [r["expected_profit"] or 0  for r in valid]

    dates  = [r["generated_at"] for r in valid if r["generated_at"]]
    drange = (min(dates)[:10], max(dates)[:10]) if dates else ("?", "?")
    sources = Counter(r["source"] for r in valid)

    result = {
        "n":            n,
        "date_range":   drange,
        "sources":      dict(sources),
        "win_rate":     round(sum(won) / n * 100, 1) if n else None,
        "corr_lqs_won": pearson(lqs, won),
        "corr_wp_won":  pearson(wp,  won),
        "corr_ev_won":  pearson(ev,  won),
    }

    print("\n" + "═" * 60)
    print("ANALYSIS 1 — Signal Correlation with Outcomes")
    print("═" * 60)
    print(f"  n = {n}  ({drange[0]} → {drange[1]})")
    print(f"  Win rate:  {result['win_rate']}%")
    print(f"  Sources:   {dict(sources)}")
    print()
    print(f"  corr(avg_lqs,         won) = {result['corr_lqs_won']}")
    print(f"  corr(win_prob,        won) = {result['corr_wp_won']}")
    print(f"  corr(expected_profit, won) = {result['corr_ev_won']}")
    if n < 30:
        print(f"\n  ⚠  Only {n} settled bets with LQS — correlations unreliable below ~50")
    elif n < 100:
        print(f"\n  ⚠  {n} bets — treat correlations as directional only until n≥100")

    # Also report bets without LQS
    n_all  = len(rows)
    n_nolqs = sum(1 for r in rows if r["avg_lqs"] is None)
    if n_nolqs:
        won_nolqs = sum(r["won"] for r in rows if r["avg_lqs"] is None)
        print(f"\n  Note: {n_nolqs}/{n_all} bets lack LQS (older bets pre-LQS feature)  "
              f"— win_rate={round(won_nolqs/n_nolqs*100,1)}%")

    # Morning vs PM CLV comparison
    am_bets = [r for r in rows if r["source"] == "prospective"]
    pm_bets = [r for r in rows if r["source"] == "prospective_pm"]
    if am_bets and pm_bets:
        print(f"\n  Morning vs PM batch comparison:")
        for label, batch in [("Morning (8 AM)", am_bets), ("Afternoon (3 PM)", pm_bets)]:
            bn    = len(batch)
            bwr   = round(sum(r["won"] for r in batch) / bn * 100, 1)
            print(f"    {label:<20} n={bn:<5} win_rate={bwr}%")
    elif pm_bets:
        bwr = round(sum(r["won"] for r in pm_bets) / len(pm_bets) * 100, 1)
        print(f"\n  PM batch (prospective_pm): n={len(pm_bets)}  win_rate={bwr}%")

    # ── Per-component proxy correlations ─────────────────────────────────────
    # Component A (historical accuracy): no direct column — its bias is visible
    #   indirectly because avg_lqs(LOSS) > avg_lqs(WIN) when A is inflated.
    # Component B (model confidence): predicted_win_prob in mock_bet_legs.
    # Component D (odds edge):          predicted_edge_pp in mock_bet_legs.
    # Both correlated at the INDIVIDUAL LEG level (not parlay level).
    if db_path:
        leg_proxies = _load_component_proxies(db_path)
        n_legs = len(leg_proxies)
        if n_legs >= 5:
            b_vals  = [r["comp_b"] or 0  for r in leg_proxies]
            d_vals  = [r["comp_d"] or 0  for r in leg_proxies]
            l_won   = [r["leg_won"]       for r in leg_proxies]
            c_b_leg = pearson(b_vals, l_won)
            c_d_leg = pearson(d_vals, l_won)

            # Component A inversion check (at bet level)
            lqs_win  = [r["avg_lqs"] for r in valid if r["won"] == 1 and r["avg_lqs"] is not None]
            lqs_loss = [r["avg_lqs"] for r in valid if r["won"] == 0 and r["avg_lqs"] is not None]
            avg_lqs_win  = round(sum(lqs_win)  / len(lqs_win),  2) if lqs_win  else None
            avg_lqs_loss = round(sum(lqs_loss) / len(lqs_loss), 2) if lqs_loss else None
            a_inverted = (avg_lqs_loss is not None
                          and avg_lqs_win  is not None
                          and avg_lqs_loss > avg_lqs_win)

            print(f"\n  Component proxy correlations  (n_legs={n_legs}):")
            print(f"    A (historical accuracy) — inversion check:")
            print(f"      avg_lqs  WIN bets = {avg_lqs_win}")
            print(f"      avg_lqs LOSS bets = {avg_lqs_loss}")
            if a_inverted:
                diff = round(avg_lqs_loss - avg_lqs_win, 2)
                print(f"      → ⚠  INVERTED (+{diff}) — A is biased from inferred_parlay_win data")
                print(f"         Fix: reduce A weight (current: 22%) until unbiased n≥200")
            else:
                print(f"      → ✓ Not inverted — A bias not detected at this sample size")

            print(f"    B (model confidence)     corr(win_prob,   leg_won) = {c_b_leg}")
            print(f"    D (odds edge)            corr(edge_pp,    leg_won) = {c_d_leg}")
            if c_b_leg is not None and c_d_leg is not None:
                top = max(("B", c_b_leg), ("D", c_d_leg), key=lambda x: x[1])
                bot = min(("B", c_b_leg), ("D", c_d_leg), key=lambda x: x[1])
                print(f"\n    Best signal: Component {top[0]} ({top[1]:+.4f})")
                if bot[1] < -0.05:
                    print(f"    ⚠  Component {bot[0]} is negatively correlated — "
                          f"consider reducing its weight further")

            result["comp_proxies"] = {
                "n_legs":         n_legs,
                "corr_b_leg_won": c_b_leg,
                "corr_d_leg_won": c_d_leg,
                "avg_lqs_win":    avg_lqs_win,
                "avg_lqs_loss":   avg_lqs_loss,
                "a_inverted":     a_inverted,
            }

    return result


def analysis_2_monotonicity(rows: list[dict]) -> dict:
    """Quintile win rate tables for LQS, win_prob, and EV."""
    # Use all rows for LQS; rows with the field for WP/EV
    rows_lqs  = [r for r in rows if r["avg_lqs"] is not None]
    rows_wp   = [r for r in rows if r["win_prob"] is not None]
    rows_ev   = [r for r in rows if r["expected_profit"] is not None]

    def _print_table(signal_rows: list[dict], signal_key: str, label: str) -> list[dict]:
        buckets = quintile_buckets(signal_rows, signal_key)
        n_sig = len(signal_rows)
        print(f"\n  {label} quintiles  (n={n_sig}):")
        if not buckets:
            print("    (insufficient data)")
            return []
        print(f"  {'Bucket':<12} {'n':>5} {'WR%':>7} {'Avg P&L':>10} {'Range':>22}")
        print("  " + "─" * 60)
        for b in buckets:
            lo, hi = b["signal_range"]
            print(f"  {b['label']:<12} {b['n']:>5} {b['win_rate']:>6.1f}%"
                  f" {b['avg_pnl']:>+9.2f}  ({lo} – {hi})")
        mono = _is_monotone(buckets)
        tag  = "✓ monotone" if mono else "✗ not monotone"
        print(f"  → {tag}")
        return buckets

    print("\n" + "═" * 60)
    print("ANALYSIS 2 — Monotonicity Check")
    print("═" * 60)

    result = {
        "lqs_quintiles": _print_table(rows_lqs, "avg_lqs",         "LQS"),
        "wp_quintiles":  _print_table(rows_wp,  "win_prob",        "Win Prob"),
        "ev_quintiles":  _print_table(rows_ev,  "expected_profit", "Expected Profit"),
    }
    return result


def analysis_3_ats_impact(rows: list[dict]) -> dict:
    """Pre/post ATS bug fix comparison."""
    pre  = [r for r in rows if (r["generated_at"] or "") < ATS_FIX_CUTOFF]
    post = [r for r in rows if (r["generated_at"] or "") >= ATS_FIX_CUTOFF]

    def _stats(group: list[dict], label: str) -> dict:
        if not group:
            return {"label": label, "n": 0}
        n         = len(group)
        wins      = sum(r["won"] for r in group)
        valid_lqs = [r["avg_lqs"]  for r in group if r["avg_lqs"]  is not None]
        valid_wp  = [r["win_prob"] for r in group if r["win_prob"] is not None]
        total_pnl = sum(r["actual_profit"] or 0 for r in group)
        return {
            "label":    label,
            "n":        n,
            "win_rate": round(wins / n * 100, 1),
            "avg_lqs":  round(sum(valid_lqs) / len(valid_lqs), 1) if valid_lqs else None,
            "avg_wp":   round(sum(valid_wp)  / len(valid_wp),  1) if valid_wp  else None,
            "total_pnl": round(total_pnl, 2),
        }

    pre_s  = _stats(pre,  f"Pre-fix  (before {ATS_FIX_CUTOFF[:10]})")
    post_s = _stats(post, f"Post-fix (on/after {ATS_FIX_CUTOFF[:10]})")

    print("\n" + "═" * 60)
    print("ANALYSIS 3 — ATS Bug Impact Isolation")
    print("═" * 60)
    print(f"  Cutoff: {ATS_FIX_CUTOFF} UTC")
    print()
    for s in (pre_s, post_s):
        if s["n"] == 0:
            print(f"  {s['label']}: n=0")
            continue
        print(f"  {s['label']}")
        print(f"    n={s['n']}  win_rate={s['win_rate']}%  "
              f"avg_lqs={s['avg_lqs']}  avg_wp={s['avg_wp']}%  "
              f"total_pnl=${s['total_pnl']:+.2f}")

    if pre_s["n"] > 0 and post_s["n"] > 0:
        diff = round(post_s["win_rate"] - pre_s["win_rate"], 1)
        sign = "+" if diff >= 0 else ""
        print(f"\n  Post vs Pre win rate delta: {sign}{diff}pp")
        if abs(diff) >= 10:
            print("  ⚠  Large delta — pre-fix bets likely distort correlation analysis.")
            print("     Recommend filtering to post-fix bets only for signal evaluation.")
        else:
            print("  ✓ Delta < 10pp — pre-fix bets probably safe to include.")
    elif post_s["n"] == 0:
        print("\n  ⚠  No post-fix bets settled yet — all data is pre-fix.")

    return {"pre": pre_s, "post": post_s}


def analysis_4_sample_size(rows: list[dict]) -> dict:
    """Days-to-significance projections based on current data."""
    post = [r for r in rows if (r["generated_at"] or "") >= ATS_FIX_CUTOFF]
    n_clean = len(post)

    # Win rate from post-fix window; fall back to assumption if too few
    actual_wr = round(sum(r["won"] for r in post) / n_clean, 4) if n_clean else 0.25
    p = actual_wr if n_clean >= 10 else 0.25

    # Daily generation rate: bets / unique days in the post-fix window
    days_seen: set[str] = set()
    for r in post:
        ga = r.get("generated_at") or ""
        if ga:
            days_seen.add(ga[:10])
    daily_rate = round(n_clean / max(len(days_seen), 1), 1)

    n_power   = power_n(p0=p, delta=0.05)
    days_pwr  = round(n_power / max(daily_rate, 0.1))

    print("\n" + "═" * 60)
    print("ANALYSIS 4 — Sample Size Projections")
    print("═" * 60)
    print(f"  Post-fix bets: {n_clean}  over {len(days_seen)} days")
    wr_note = f"{round(p*100,1)}% (observed)" if n_clean >= 10 else f"25.0% (assumed — only {n_clean} post-fix bets)"
    print(f"  Win rate used: {wr_note}")
    print(f"  Daily generation rate: ~{daily_rate} bets/day")
    print()
    print(f"  {'n':>5}  {'Days needed':>12}  {'95% CI (±pp)':>14}  Note")
    print("  " + "─" * 60)
    milestones = {}
    for target_n in (30, 50, 100, 200, 500):
        days  = round(target_n / max(daily_rate, 0.1))
        ci_95 = round(1.96 * math.sqrt(p * (1 - p) / target_n) * 100, 1)
        note  = "← minimum for basic signal" if target_n == 30 else \
                "← reliable correlation"     if target_n == 100 else \
                "← publishable quality"      if target_n == 200 else ""
        print(f"  {target_n:>5}  {days:>10} d  ±{ci_95:>10.1f}pp  {note}")
        milestones[str(target_n)] = {"days": days, "ci_95_pp": ci_95}

    print()
    print(f"  Min n to detect +5pp improvement (80% power): {n_power}")
    print(f"  → ~{days_pwr} days at current generation rate")

    return {
        "n_post_fix":        n_clean,
        "n_days_seen":       len(days_seen),
        "daily_rate":        daily_rate,
        "win_rate_pct":      round(p * 100, 1),
        "milestones":        milestones,
        "n_for_5pp_power":   n_power,
        "days_for_power":    days_pwr,
    }


def analysis_5_correlation_matrix(rows: list[dict]) -> dict:
    """Pairwise signal correlation matrix (signal interdependence)."""
    valid = [r for r in rows
             if r["avg_lqs"]         is not None
             and r["win_prob"]       is not None
             and r["expected_profit"] is not None]
    n = len(valid)

    lqs = [r["avg_lqs"]          for r in valid]
    wp  = [r["win_prob"]         for r in valid]
    ev  = [r["expected_profit"]  for r in valid]

    c_lqs_wp = pearson(lqs, wp)
    c_lqs_ev = pearson(lqs, ev)
    c_wp_ev  = pearson(wp,  ev)

    def _fmt(v: float | None) -> str:
        return f"{v:+.4f}" if v is not None else "  n/a "

    print("\n" + "═" * 60)
    print("ANALYSIS 5 — Signal Correlation Matrix")
    print("═" * 60)
    print(f"  n = {n}  (bets with all three signals populated)")
    print()
    print(f"  {'':12}  {'LQS':>8}  {'WinProb':>8}  {'EV':>8}")
    print("  " + "─" * 44)
    print(f"  {'LQS':<12}  {'  1.000':>8}  {_fmt(c_lqs_wp):>8}  {_fmt(c_lqs_ev):>8}")
    print(f"  {'WinProb':<12}  {_fmt(c_lqs_wp):>8}  {'  1.000':>8}  {_fmt(c_wp_ev):>8}")
    print(f"  {'EV':<12}  {_fmt(c_lqs_ev):>8}  {_fmt(c_wp_ev):>8}  {'  1.000':>8}")
    print()

    def _interp(r: float | None, a: str, b: str) -> str:
        if r is None:
            return f"  corr({a}, {b}): insufficient data"
        if abs(r) > 0.7:
            return (f"  corr({a}, {b}) = {r:+.4f}  HIGH — signals redundant; "
                    "weighting both unlikely to help")
        elif abs(r) > 0.4:
            return f"  corr({a}, {b}) = {r:+.4f}  MODERATE — some shared information"
        else:
            return (f"  corr({a}, {b}) = {r:+.4f}  LOW — independent signals; "
                    "weighting both adds value")

    print(_interp(c_lqs_wp, "LQS", "WinProb"))
    print(_interp(c_lqs_ev, "LQS", "EV"))
    print(_interp(c_wp_ev,  "WinProb", "EV"))

    if n < 10:
        print(f"\n  ⚠  Only {n} bets with all three signals — matrix unreliable")

    return {
        "n":           n,
        "corr_lqs_wp": c_lqs_wp,
        "corr_lqs_ev": c_lqs_ev,
        "corr_wp_ev":  c_wp_ev,
    }


def analysis_6_regime_performance(db_path: str) -> dict:
    """
    Regime performance table: for each regime, show days seen, avg win rate,
    avg P&L, and avg weighted model confidence.
    Also: corr(weighted_model_confidence, mock_win_rate) across all days.
    """
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT date, regime, mock_win_rate, mock_pnl, weighted_model_confidence
            FROM market_regime_log
            WHERE mock_win_rate IS NOT NULL
            ORDER BY date
        """)
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
    except Exception as e:
        print(f"\n  [warn] market_regime_log query failed: {e}")
        rows = []

    print("\n" + "═" * 60)
    print("ANALYSIS 6 — Regime Performance Table")
    print("═" * 60)

    if not rows:
        print("  No settled regime days yet — run again after bets settle.")
        return {"n": 0}

    regimes = sorted({r["regime"] for r in rows})
    print(f"  {'Regime':<12} {'Days':>5}  {'Avg WR':>8}  {'Avg P&L':>10}  {'Avg Conf':>10}")
    print("  " + "─" * 52)
    regime_stats: dict = {}
    for reg in regimes:
        sub = [r for r in rows if r["regime"] == reg]
        wr  = round(sum(r["mock_win_rate"] for r in sub) / len(sub) * 100, 1)
        pnl = round(sum(r["mock_pnl"] or 0 for r in sub) / len(sub), 2)
        conf_vals = [r["weighted_model_confidence"] for r in sub
                     if r["weighted_model_confidence"] is not None]
        conf = round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else None
        conf_s = f"{conf:.3f}" if conf is not None else "  n/a"
        print(f"  {reg:<12} {len(sub):>5}  {wr:>7.1f}%  {pnl:>+9.2f}  {conf_s:>10}")
        regime_stats[reg] = {"days": len(sub), "avg_wr_pct": wr, "avg_pnl": pnl, "avg_conf": conf}

    # corr(weighted_model_confidence, mock_win_rate)
    conf_wp = [(r["weighted_model_confidence"], r["mock_win_rate"])
               for r in rows
               if r["weighted_model_confidence"] is not None and r["mock_win_rate"] is not None]
    if len(conf_wp) >= 3:
        xs = [v[0] for v in conf_wp]
        ys = [v[1] for v in conf_wp]
        c = pearson(xs, ys)
        print()
        print(f"  corr(weighted_model_confidence, mock_win_rate) = {c}")
        if c is not None:
            if c > 0.4:
                print("  → Positive: higher confidence days produce better win rates")
            elif c < -0.2:
                print("  → Negative: confidence not predictive — check model calibration")
            else:
                print("  → Near-zero: confidence not yet predicting daily win rates")
    else:
        c = None
        print(f"\n  corr(confidence, win_rate): insufficient data ({len(conf_wp)} days)")

    return {
        "n":            len(rows),
        "regimes":      regime_stats,
        "corr_conf_wr": c,
    }


# ── LQS correlation trend tracking ───────────────────────────────────────────

def _corr_lqs_since(db_path: str, cutoff_date: str) -> tuple[float | None, int]:
    """
    Compute corr(avg_lqs, won) for T1+T2 simulation bets settled after cutoff_date.
    Returns (correlation, n).
    """
    if not os.path.exists(db_path):
        return None, 0
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    src_ph = ",".join("?" * len({"prospective", "prospective_pm", "top_picks_page"}))
    try:
        cur.execute(f"""
            SELECT avg_lqs,
                   CASE WHEN status='SETTLED_WIN' THEN 1 ELSE 0 END AS won
            FROM mock_bets
            WHERE source IN ('prospective','prospective_pm','top_picks_page')
              AND generated_at >= ?
              AND status IN ('SETTLED_WIN','SETTLED_LOSS')
              AND avg_lqs IS NOT NULL
        """, (cutoff_date,))
        rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        con.close()

    if len(rows) < 3:
        return None, len(rows)
    lqs = [r["avg_lqs"] for r in rows]
    won = [r["won"]     for r in rows]
    return pearson(lqs, won), len(rows)


def analysis_7_lqs_trend(db_path: str) -> dict:
    """
    Track corr(avg_lqs, won) across two windows:
      1. Since simulation start (SIM_START_DATE = 2026-04-22)
      2. Since weight change   (WEIGHT_CHANGE_DATE = 2026-04-26)

    Target: positive and increasing week-over-week.
    Decision rules:
      - Still negative after 2 weeks of new weights → investigate whether
        predicted_win_prob at leg level predicts individual leg outcomes.
      - Positive but below +0.05 after 4 weeks → fine, keep accumulating data.
      - Reaches +0.10 → weights are working; schedule formal OLS calibration.
    """
    c_sim,     n_sim     = _corr_lqs_since(db_path, SIM_START_DATE)
    c_weights, n_weights = _corr_lqs_since(db_path, WEIGHT_CHANGE_DATE)

    print("\n" + "═" * 60)
    print("ANALYSIS 7 — LQS Correlation Trend")
    print("═" * 60)
    print(f"  Weight change date: {WEIGHT_CHANGE_DATE}  "
          f"(A 40→22%, B 30→44%)")
    print()

    def _status(c: float | None, n: int, window: str) -> str:
        if c is None:
            return f"  {window:<28} n={n:<4}  — (insufficient data)"
        arrow = "↑" if c > 0 else "↓"
        tag = ("✓ predictive" if c >  0.05
               else "⚠ not yet predictive" if c > -0.05
               else "✗ INVERTED — investigate")
        return f"  {window:<28} n={n:<4}  corr={c:+.4f}  {arrow}  {tag}"

    print(_status(c_sim,     n_sim,     f"Since sim start ({SIM_START_DATE})"))
    print(_status(c_weights, n_weights, f"Since weight change ({WEIGHT_CHANGE_DATE})"))

    if c_weights is not None and c_sim is not None:
        delta = round(c_weights - c_sim, 4)
        sign  = "+" if delta >= 0 else ""
        print(f"\n  Δ since weight change: {sign}{delta}")
        if delta > 0:
            print("  → Weight change is improving LQS predictiveness")
        else:
            print("  → No improvement yet — wait for more post-change settlements")

    if c_weights is not None and c_weights < -0.05 and n_weights >= 30:
        print("\n  ⚠  DECISION TRIGGER: corr still negative after n≥30 with new weights.")
        print("     Next step: check if predicted_win_prob predicts individual leg outcomes")
        print("     vs parlay outcomes (these are different prediction problems).")
        print("     Consider decoupling LQS from parlay outcome → score per-leg instead.")

    if c_weights is not None and c_weights >= 0.10:
        print("\n  ✓ MILESTONE: corr≥0.10 reached.")
        print("     Schedule formal OLS calibration of component weights.")

    return {
        "corr_sim_start":    c_sim,
        "n_sim_start":       n_sim,
        "corr_new_weights":  c_weights,
        "n_new_weights":     n_weights,
        "weight_change_date": WEIGHT_CHANGE_DATE,
    }


# ── Performance segmentation (weekly snapshot) ────────────────────────────────

def migrate_performance_log(db_path: str) -> None:
    """Idempotent migration: create / update model_performance_log."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_performance_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            week_ending       TEXT    NOT NULL UNIQUE,
            computed_at       TEXT    NOT NULL,
            t1_n              INTEGER,
            t1_wins           INTEGER,
            t1_win_rate       REAL,
            t1_pnl            REAL,
            t2_n              INTEGER,
            t2_wins           INTEGER,
            t2_win_rate       REAL,
            t2_pnl            REAL,
            t3_n              INTEGER,
            t3_wins           INTEGER,
            t3_win_rate       REAL,
            t3_pnl            REAL,
            total_n           INTEGER,
            total_wins        INTEGER,
            overall_win_rate  REAL,
            overall_pnl       REAL,
            sport_breakdown   TEXT,
            corr_lqs_won      REAL,
            corr_wp_won       REAL,
            corr_ev_won       REAL,
            n_corr            INTEGER,
            trust_gate_passed INTEGER,
            trust_gate_reason TEXT,
            lqs_monotone      INTEGER,
            corr_lqs_won_sim        REAL,
            n_sim                   INTEGER,
            corr_lqs_won_post_wt    REAL,
            n_post_wt               INTEGER
        )
    """)
    # Idempotent column additions for rows written before this migration
    for col, typ in [
        ("corr_lqs_won_sim",       "REAL"),
        ("n_sim",                  "INTEGER"),
        ("corr_lqs_won_post_wt",   "REAL"),
        ("n_post_wt",              "INTEGER"),
        # Regime A/B tracking columns (added 2026-04-27)
        ("regime_ab_days_logged",  "INTEGER"),
        ("regime_ab_standard_wr",  "REAL"),
        ("regime_ab_shadow_wr",    "REAL"),
        ("regime_ab_flip_ready",   "INTEGER"),
        ("regime_ab_by_regime",    "TEXT"),   # JSON snapshot
        # System launch boundary columns (added 2026-04-29)
        ("system_launch_date",       "TEXT"),
        ("pre_launch_bets_excluded", "INTEGER"),
        # Pairwise CUSHION parlay tracking (added 2026-05-01)
        ("top_cushion_pairings",     "TEXT"),   # JSON list of top-5 pairing dicts
    ]:
        try:
            cur.execute(f"ALTER TABLE model_performance_log ADD COLUMN {col} {typ}")
        except Exception:
            pass   # column already exists
    con.commit()
    con.close()


def load_all_settled(db_path: str) -> list[dict]:
    """
    Load simulation mock bets only — excludes legacy Pikkit bets.

    Filters:
      - source IN SIM_SOURCES  (the 5 known simulation source types)
      - generated_at >= SIM_START_DATE  (2026-04-22 is the first simulation day)
      - status IN ('SETTLED_WIN', 'SETTLED_LOSS')

    This prevents historical/attribution bets from leaking into tier analysis.
    """
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("PRAGMA table_info(mock_bets)")
    cols = {row["name"] for row in cur.fetchall()}

    avg_lqs_sel = "mb.avg_lqs" if "avg_lqs" in cols else "NULL"
    ev_sel      = "mb.predicted_ev" if "predicted_ev" in cols else "NULL"
    wp_sel      = "mb.predicted_win_prob" if "predicted_win_prob" in cols else "NULL"

    # Build IN clause from SIM_SOURCES constant so it stays in sync
    src_placeholders = ",".join("?" * len(SIM_SOURCES))
    src_list = list(SIM_SOURCES)

    cur.execute(f"""
        SELECT
            mb.id,
            mb.generated_at,
            mb.source,
            mb.status,
            {avg_lqs_sel}  AS avg_lqs,
            {wp_sel}        AS win_prob,
            {ev_sel}        AS expected_profit,
            mb.actual_profit,
            COALESCE(mb.amount, 10.0) AS amount,
            (SELECT ml.sport FROM mock_bet_legs ml
             WHERE ml.mock_bet_id = mb.id LIMIT 1) AS sport
        FROM mock_bets mb
        WHERE mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND mb.source IN ({src_placeholders})
          AND mb.generated_at >= ?
        ORDER BY mb.generated_at
    """, src_list + [SIM_START_DATE])
    rows = []
    for r in cur.fetchall():
        rows.append({
            "id":              r["id"],
            "generated_at":    r["generated_at"],
            "source":          r["source"],
            "won":             1 if r["status"] == "SETTLED_WIN" else 0,
            "avg_lqs":         r["avg_lqs"],
            "win_prob":        r["win_prob"],
            "expected_profit": r["expected_profit"],
            "actual_profit":   r["actual_profit"],
            "amount":          r["amount"],
            "sport":           r["sport"] or "Unknown",
        })
    con.close()
    return rows


def _compute_cushion_pairings(db_path: str, since: str) -> list[dict]:
    """
    Find the top-5 most common CUSHION leg pairings across multi-leg settled bets
    and compute each pairing's win rate.

    Returns a list of dicts sorted by occurrence count:
      [{"pair": ["NBA|spreads", "MLB|totals"], "n": 12, "wins": 7, "wr": 0.583}, ...]
    """
    if not os.path.exists(db_path):
        return []
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        # Load settled multi-leg bets with their CUSHION legs since launch date
        rows = con.execute("""
            SELECT mb.id, mb.status, mbl.sport, mbl.market_type, mbl.grade
            FROM   mock_bets mb
            JOIN   mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
            WHERE  mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND  mb.legs > 1
              AND  mb.generated_at >= ?
              AND  mbl.grade = 'CUSHION'
        """, (since,)).fetchall()
        con.close()

        # Group CUSHION legs by bet id
        from collections import defaultdict
        bet_legs: dict[str, dict] = defaultdict(lambda: {"won": False, "combos": []})
        for r in rows:
            bid = r["id"]
            bet_legs[bid]["won"] = (r["status"] == "SETTLED_WIN")
            combo = f"{(r['sport'] or 'UNK').strip()}|{(r['market_type'] or 'UNK').strip()}"
            if combo not in bet_legs[bid]["combos"]:
                bet_legs[bid]["combos"].append(combo)

        # Count pairings
        pair_counts: dict[tuple, dict] = {}
        for bet in bet_legs.values():
            combos = sorted(bet["combos"])
            won = bet["won"]
            # Generate all 2-combos
            for i in range(len(combos)):
                for j in range(i + 1, len(combos)):
                    key = (combos[i], combos[j])
                    if key not in pair_counts:
                        pair_counts[key] = {"n": 0, "wins": 0}
                    pair_counts[key]["n"] += 1
                    if won:
                        pair_counts[key]["wins"] += 1

        # Sort by count desc, take top 5
        top5 = sorted(pair_counts.items(), key=lambda x: x[1]["n"], reverse=True)[:5]
        result = []
        for (a, b), stats in top5:
            n = stats["n"]
            w = stats["wins"]
            result.append({
                "pair": [a, b],
                "n": n,
                "wins": w,
                "wr": round(w / n, 4) if n else None,
            })
        return result
    except Exception:
        return []


def compute_weekly_snapshot(db_path: str, week_ending: str | None = None) -> dict:
    """
    Compute a weekly performance snapshot and upsert into model_performance_log.

    week_ending: ISO date string e.g. '2026-04-27'.  Defaults to the coming Sunday.
    Covers ALL time — not just one week — so the snapshot always reflects cumulative state.
    """
    if week_ending is None:
        today = datetime.utcnow()
        days_ahead = (6 - today.weekday()) % 7   # days until next Sunday (0=Mon…6=Sun)
        week_ending = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    print(f"\n  Computing weekly snapshot  week_ending={week_ending}")

    all_rows = load_all_settled(db_path)
    if not all_rows:
        print("  No settled bets — snapshot skipped.")
        return {}

    # ── System launch boundary ────────────────────────────────────────────────
    # Apr 22–28 bets exist and feed historical learning (Component A / LQS training)
    # but are excluded from weekly performance metrics because CUSHION/AVOID margin
    # grades and personal_edge_profile were not yet active.
    pre_launch_count = sum(
        1 for r in all_rows if (r.get("generated_at") or "") < SYSTEM_LAUNCH_DATE
    )
    rows = [r for r in all_rows if (r.get("generated_at") or "") >= SYSTEM_LAUNCH_DATE]
    if pre_launch_count:
        print(f"  Pre-launch bets excluded from snapshot: {pre_launch_count} "
              f"(generated before {SYSTEM_LAUNCH_DATE})")

    # ── Tier definitions ─────────────────────────────────────────────────────
    # T1: model chose + high information state (PM batch + top picks page)
    T1_SOURCES = {"prospective_pm", "top_picks_page"}
    # T2: model made a genuine morning pick (prospective only)
    #     WR trend here = is the model's morning signal improving?
    T2_SOURCES = {"prospective"}
    # T3: no-signal override OR post-hoc retroactive bets — baseline control
    #     forced_gen fires when model said "no qualifying picks"; retroactive_mock
    #     is created after game start/end. Both are non-genuine signals.
    #     prospective_legacy = morning batch superseded by fresh generation the
    #     same day; still settles and adds calibration data but does NOT count
    #     toward T1/T2 performance metrics.
    #     WR trend here = is the floor (random pick quality) rising?
    T3_SOURCES = {"forced_generation", "retroactive_mock", "prospective_legacy"}
    # LEGACY: anything else or pre-simulation — excluded by load_all_settled()

    def _stats(tier_rows: list[dict]) -> dict:
        if not tier_rows:
            return {"n": 0, "wins": 0, "win_rate": None, "pnl": 0.0}
        n    = len(tier_rows)
        wins = sum(r["won"] for r in tier_rows)
        pnl  = round(sum(r["actual_profit"] or 0 for r in tier_rows), 2)
        return {"n": n, "wins": wins, "win_rate": round(wins / n, 4), "pnl": pnl}

    t1 = _stats([r for r in rows if r["source"] in T1_SOURCES])
    t2 = _stats([r for r in rows if r["source"] in T2_SOURCES])
    t3 = _stats([r for r in rows if r["source"] in T3_SOURCES])

    # Primary = T1 + T2: genuine model picks.
    # T3 excluded — forced/retroactive bets intentionally omit model signal.
    primary = [r for r in rows if r["source"] in (T1_SOURCES | T2_SOURCES)]
    total_n    = len(primary)
    total_wins = sum(r["won"] for r in primary)
    overall_wr  = round(total_wins / total_n, 4) if total_n else None
    overall_pnl = round(sum(r["actual_profit"] or 0 for r in primary), 2)

    # Sport breakdown — split pipe-separated multi-sport strings (e.g. "EPL | Serie A")
    # and attribute each bet fractionally (1/n weight per sport)
    sport_map: dict[str, dict] = {}
    for r in primary:
        raw_sport = r["sport"] or "Unknown"
        parts = [s.strip() for s in raw_sport.split("|") if s.strip()]
        if not parts:
            parts = ["Unknown"]
        weight = 1.0 / len(parts)
        for sp in parts:
            if sp not in sport_map:
                sport_map[sp] = {"n": 0.0, "wins": 0.0, "pnl": 0.0}
            sport_map[sp]["n"]    += weight
            sport_map[sp]["wins"] += weight * r["won"]
            sport_map[sp]["pnl"]  += weight * (r["actual_profit"] or 0)
    sport_breakdown: dict = {}
    for sp, agg in sorted(sport_map.items()):
        sn   = agg["n"]
        sw   = agg["wins"]
        spnl = round(agg["pnl"], 2)
        sport_breakdown[sp] = {
            "n": round(sn, 2), "wins": round(sw, 2),
            "win_rate": round(sw / sn, 4) if sn else None,
            "pnl": spnl,
        }

    # Signal correlations — post-ATS-fix clean data only
    corr_rows = [r for r in primary
                 if (r["generated_at"] or "") >= ATS_FIX_CUTOFF
                 and r["avg_lqs"] is not None]
    n_corr = len(corr_rows)
    if n_corr >= 3:
        lqs = [r["avg_lqs"]              for r in corr_rows]
        wps = [r["win_prob"]     or 0    for r in corr_rows]
        evs = [r["expected_profit"] or 0 for r in corr_rows]
        won = [r["won"]                  for r in corr_rows]
        corr_lqs_won = pearson(lqs, won)
        corr_wp_won  = pearson(wps, won)
        corr_ev_won  = pearson(evs, won)
    else:
        corr_lqs_won = corr_wp_won = corr_ev_won = None

    # Trust gate: n≥30 and win_rate≥20%
    if total_n < 30:
        trust_gate_passed = 0
        trust_gate_reason = f"insufficient data: n={total_n} (<30)"
    elif overall_wr is not None and overall_wr >= 0.20:
        trust_gate_passed = 1
        trust_gate_reason = f"n={total_n}  win_rate={round(overall_wr*100,1)}%"
    else:
        trust_gate_passed = 0
        trust_gate_reason = f"win_rate too low: {round((overall_wr or 0)*100,1)}%"

    # LQS monotonicity over primary rows
    lqs_rows    = [r for r in primary if r["avg_lqs"] is not None]
    lqs_buckets = quintile_buckets(lqs_rows, "avg_lqs")
    lqs_monotone = (1 if _is_monotone(lqs_buckets) else 0) if lqs_buckets else None

    # Print summary
    def _wr(stats: dict) -> str:
        return f"{round((stats['win_rate'] or 0)*100,1)}%" if stats["n"] else "—"

    print(f"  T1 (PM/top_picks): n={t1['n']:>4}  wr={_wr(t1)}  pnl=${t1['pnl']:+.2f}")
    print(f"  T2 (morning):      n={t2['n']:>4}  wr={_wr(t2)}  pnl=${t2['pnl']:+.2f}")
    print(f"  T3 (retro/forced): n={t3['n']:>4}  wr={_wr(t3)}  pnl=${t3['pnl']:+.2f}")
    print(f"  Overall (T1+T2):   n={total_n:>4}  wr={round((overall_wr or 0)*100,1)}%  pnl=${overall_pnl:+.2f}")
    print(f"  Trust gate: {'PASSED' if trust_gate_passed else 'FAILED'} — {trust_gate_reason}")
    if lqs_monotone is not None:
        print(f"  LQS monotone: {bool(lqs_monotone)}  corr_lqs_won={corr_lqs_won}")
    else:
        print("  LQS monotone: insufficient data")

    # ── Pairwise CUSHION parlay tracking ─────────────────────────────────────
    # For each multi-leg settled bet, find all CUSHION leg pairs and record
    # the bet outcome.  Top-5 most common pairs + their win rate stored as JSON.
    top_cushion_pairings = _compute_cushion_pairings(db_path, SYSTEM_LAUNCH_DATE)

    # Upsert into model_performance_log
    con = sqlite3.connect(db_path)
    # LQS trend correlations for the two tracking windows
    c_lqs_sim,    n_sim    = _corr_lqs_since(db_path, SIM_START_DATE)
    c_lqs_post_wt, n_post_wt = _corr_lqs_since(db_path, WEIGHT_CHANGE_DATE)

    # Regime A/B snapshot
    ab = analysis_8_regime_ab(db_path)
    ab_days     = ab.get("days_logged", 0)
    ab_std_wr   = ab.get("standard_overall_wr")
    ab_shad_wr  = ab.get("regime_overall_wr")
    ab_flip     = 1 if ab.get("flip_ready") else 0
    ab_by_rg    = json.dumps(ab.get("by_regime", {}))

    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO model_performance_log (
            week_ending, computed_at,
            t1_n, t1_wins, t1_win_rate, t1_pnl,
            t2_n, t2_wins, t2_win_rate, t2_pnl,
            t3_n, t3_wins, t3_win_rate, t3_pnl,
            total_n, total_wins, overall_win_rate, overall_pnl,
            sport_breakdown,
            corr_lqs_won, corr_wp_won, corr_ev_won, n_corr,
            trust_gate_passed, trust_gate_reason, lqs_monotone,
            corr_lqs_won_sim, n_sim,
            corr_lqs_won_post_wt, n_post_wt,
            regime_ab_days_logged, regime_ab_standard_wr, regime_ab_shadow_wr,
            regime_ab_flip_ready, regime_ab_by_regime,
            system_launch_date, pre_launch_bets_excluded,
            top_cushion_pairings
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        week_ending, datetime.utcnow().isoformat(),
        t1["n"], t1["wins"], t1["win_rate"], t1["pnl"],
        t2["n"], t2["wins"], t2["win_rate"], t2["pnl"],
        t3["n"], t3["wins"], t3["win_rate"], t3["pnl"],
        total_n, total_wins, overall_wr, overall_pnl,
        json.dumps(sport_breakdown),
        corr_lqs_won, corr_wp_won, corr_ev_won, n_corr,
        trust_gate_passed, trust_gate_reason, lqs_monotone,
        c_lqs_sim, n_sim,
        c_lqs_post_wt, n_post_wt,
        ab_days, ab_std_wr, ab_shad_wr, ab_flip, ab_by_rg,
        SYSTEM_LAUNCH_DATE, 1 if pre_launch_count > 0 else 0,
        json.dumps(top_cushion_pairings),
    ))
    con.commit()
    con.close()
    print(f"  → Snapshot written to model_performance_log  (week_ending={week_ending})")
    print(f"     corr_lqs_won since sim start:     {c_lqs_sim}  (n={n_sim})")
    print(f"     corr_lqs_won since weight change: {c_lqs_post_wt}  (n={n_post_wt})")
    print(f"     regime_ab: {ab_days} days logged, flip_ready={bool(ab_flip)}")

    return {
        "week_ending":          week_ending,
        "total_n":              total_n,
        "overall_win_rate":     overall_wr,
        "overall_pnl":          overall_pnl,
        "trust_gate":           trust_gate_passed,
        "lqs_monotone":         lqs_monotone,
        "t1":                   t1,
        "t2":                   t2,
        "t3":                   t3,
        "sport_breakdown":      sport_breakdown,
        "corr_lqs_won_sim":     c_lqs_sim,
        "corr_lqs_won_post_wt": c_lqs_post_wt,
    }


# ─── Analysis 8 — Regime A/B Test ────────────────────────────────────────────

def analysis_8_regime_ab(db_path: str) -> dict:
    """
    Compare standard composite weights vs regime-suggested weights.

    Decision trigger:
      - flip_ready = True when regime_wr > standard_wr + 5pp for ≥ 20 days
        across ≥ 2 regimes, advantage consistent 3+ consecutive weeks
    """
    if not os.path.exists(db_path):
        return {"days_logged": 0, "flip_ready": False}

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    try:
        rows = con.execute(
            "SELECT * FROM regime_ab_log ORDER BY date"
        ).fetchall()
    except Exception:
        con.close()
        return {"days_logged": 0, "flip_ready": False, "note": "table_not_yet_created"}
    finally:
        con.close()

    days_logged = len(rows)
    if days_logged == 0:
        return {"days_logged": 0, "flip_ready": False, "note": "no_data_yet"}

    # Aggregate by regime
    by_regime: dict = {}
    for r in rows:
        rn = r["regime"] or "unknown"
        rd = by_regime.setdefault(rn, {"days": 0, "std_sum": 0, "std_n": 0,
                                       "reg_sum": 0, "reg_n": 0,
                                       "weights": None})
        rd["days"] += 1
        if r["standard_win_rate"] is not None:
            rd["std_sum"] += r["standard_win_rate"]; rd["std_n"] += 1
        if r["regime_win_rate"] is not None:
            rd["reg_sum"] += r["regime_win_rate"];   rd["reg_n"] += 1
        if r["regime_weights"] and not rd["weights"]:
            try: rd["weights"] = json.loads(r["regime_weights"])
            except Exception: pass

    regime_rows: list[dict] = []
    for rn, rd in sorted(by_regime.items()):
        std_wr = round(rd["std_sum"] / rd["std_n"], 4) if rd["std_n"] else None
        reg_wr = round(rd["reg_sum"] / rd["reg_n"], 4) if rd["reg_n"] else None
        if std_wr is None or reg_wr is None:
            verdict = "insufficient_data"
        elif reg_wr > std_wr + 0.05:
            verdict = "regime_better"
        elif reg_wr < std_wr - 0.05:
            verdict = "standard_better"
        else:
            verdict = "no_difference"
        regime_rows.append({"regime": rn, "days": rd["days"],
                             "standard_wr": std_wr, "regime_wr": reg_wr,
                             "weights": rd["weights"], "verdict": verdict})

    regimes_with_data = sum(1 for r in regime_rows if r["regime_wr"] is not None)
    all_std = [r["standard_wr"] for r in regime_rows if r["standard_wr"] is not None]
    all_reg = [r["regime_wr"]   for r in regime_rows if r["regime_wr"]   is not None]
    overall_std = round(sum(all_std) / len(all_std), 4) if all_std else None
    overall_reg = round(sum(all_reg) / len(all_reg), 4) if all_reg else None

    flip_ready = (
        days_logged >= 20
        and regimes_with_data >= 2
        and overall_reg is not None and overall_std is not None
        and overall_reg > overall_std + 0.05
    )

    # Print table
    print(f"\n  ANALYSIS 8 — Regime A/B Test (n={days_logged} days logged)")
    col_w = 12
    hdr   = f"{'Regime':<12} {'Days':>4}  {'Standard WR':>12}  {'Regime WR':>11}"
    print("  " + hdr)
    print("  " + "─" * len(hdr))
    for r in regime_rows:
        std_s = f"{r['standard_wr']*100:.1f}%" if r["standard_wr"] is not None else "--"
        reg_s = f"{r['regime_wr']*100:.1f}%"   if r["regime_wr"]   is not None else "--"
        print(f"  {r['regime']:<12} {r['days']:>4}  {std_s:>12}  {reg_s:>11}  {r['verdict']}")
    if flip_ready:
        print("  ✅ FLIP READY — regime weights outperform standard by > 5pp")
    else:
        print(f"  Flip status: NOT READY  "
              f"(need 20 days [{days_logged} logged], "
              f"2 regimes with data [{regimes_with_data} have data])")

    return {
        "days_logged":     days_logged,
        "regimes_seen":    [r["regime"] for r in regime_rows],
        "standard_overall_wr": overall_std,
        "regime_overall_wr":   overall_reg,
        "by_regime":       {r["regime"]: r for r in regime_rows},
        "flip_ready":      flip_ready,
    }


# ─── Analysis 9 — Margin Profile Report ─────────────────────────────────────

def analysis_9_margin_profile(db_path: str) -> dict:
    """
    Weekly margin quality report from personal_edge_profile.

    Prints a table of all profiles sorted by personal_wr, with:
      - margin_grade (CUSHION / CLOSE / MIXED / AVOID)
      - mean_delta (average win/loss margin)
      - edge_ratio (std/mean_delta — consistency metric)
      - close_call_rate (% of wins that were narrow)
      - max_legs allowed in parlays

    Decision triggers:
      - Flag any CUSHION profile that has degraded (WR dropped below 70% since
        last refresh) — these need investigation before next recommendation cycle.
      - Flag any new AVOID profiles that were previously CUSHION/CLOSE.
    """
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    if not os.path.exists(db_path):
        return {"profiles": [], "note": "db_not_found"}

    try:
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(db_path)
        rows = con.execute("""
            SELECT sport, market_type, line_bucket, sample_size,
                   personal_wr, mean_delta, edge_ratio, close_call_rate,
                   narrow_loss_rate, margin_grade, last_updated
            FROM   personal_edge_profile
            ORDER  BY
                CASE margin_grade
                    WHEN 'CUSHION' THEN 1
                    WHEN 'CLOSE'   THEN 2
                    WHEN 'MIXED'   THEN 3
                    WHEN 'AVOID'   THEN 4
                    ELSE 5
                END,
                personal_wr DESC
        """).fetchall()
        con.close()
    except Exception as e:
        return {"profiles": [], "note": str(e)}

    if not rows:
        return {"profiles": [], "note": "table_empty"}

    from personal_edge_profile import get_max_legs_for_personal_profile

    profiles = []
    cushion_count = close_count = mixed_count = avoid_count = 0
    degraded = []

    for r in rows:
        sport, mt, bkt, n, wr, md, er, ccr, nlr, grade, updated = r
        max_legs = get_max_legs_for_personal_profile(sport, mt, bkt)
        entry = {
            "sport": sport, "market_type": mt, "line_bucket": bkt,
            "sample_size": n, "personal_wr": wr, "mean_delta": md,
            "edge_ratio": er, "close_call_rate": ccr,
            "narrow_loss_rate": nlr, "margin_grade": grade,
            "max_legs": max_legs, "last_updated": updated,
        }
        profiles.append(entry)

        if grade == "CUSHION":   cushion_count += 1
        elif grade == "CLOSE":   close_count   += 1
        elif grade == "MIXED":   mixed_count   += 1
        elif grade == "AVOID":   avoid_count   += 1

        # Flag degraded CUSHION (WR below 70% with CUSHION grade — unusual)
        if grade == "CUSHION" and wr is not None and wr < 0.70:
            degraded.append({"sport": sport, "market_type": mt, "wr": wr, "grade": grade})

    # Print table
    total = len(profiles)
    print(f"\n  ANALYSIS 9 — Margin Profile Report ({total} profiles)")
    print(f"  CUSHION={cushion_count}  CLOSE={close_count}  MIXED={mixed_count}  AVOID={avoid_count}")
    print()
    hdr = f"  {'SPORT':<24} {'MARKET':<12} {'BUCKET':<8} {'WR%':>6}  {'δ':>6}  {'ER':>5}  {'CCR':>6}  {'GRADE':<9}  {'MAX'}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for p in profiles:
        if p["margin_grade"] not in ("CUSHION", "CLOSE", "MIXED"):
            continue  # skip AVOID rows in printed table to keep it readable
        wr_s   = "%.1f%%" % (p["personal_wr"] * 100) if p["personal_wr"] is not None else "N/A"
        md_s   = "%+.2f" % p["mean_delta"] if p["mean_delta"] is not None else "N/A"
        er_s   = "%.2f"  % p["edge_ratio"] if p["edge_ratio"] is not None else "N/A"
        ccr_s  = "%.0f%%" % (p["close_call_rate"] * 100) if p["close_call_rate"] is not None else "N/A"
        grade  = p["margin_grade"] or "None"
        ml_s   = str(p["max_legs"]) if p["max_legs"] > 0 else "BLOCKED"
        print(f"  {p['sport']:<24} {p['market_type']:<12} {p['line_bucket']:<8} "
              f"{wr_s:>6}  {md_s:>6}  {er_s:>5}  {ccr_s:>6}  {grade:<9}  {ml_s}")

    # AVOID summary (compact — just counts by sport)
    avoid_profiles = [p for p in profiles if p["margin_grade"] == "AVOID"]
    if avoid_profiles:
        print(f"\n  AVOID profiles ({len(avoid_profiles)}): "
              + ", ".join(f"{p['sport']} {p['market_type']} {p['line_bucket']} "
                          f"({('%.0f%%' % (p['personal_wr']*100)) if p['personal_wr'] else 'no WR'})"
                          for p in avoid_profiles))

    if degraded:
        print(f"\n  ⚠ DEGRADED CUSHION (WR < 70%): "
              + ", ".join(f"{d['sport']} {d['market_type']} {d['wr']:.1%}" for d in degraded))

    return {
        "total_profiles": total,
        "cushion_count":  cushion_count,
        "close_count":    close_count,
        "mixed_count":    mixed_count,
        "avoid_count":    avoid_count,
        "profiles":       profiles,
        "degraded_cushion": degraded,
    }


def analysis_10_parlay_wr_targets(db_path: str, rolling_days: int = 30) -> dict:
    """
    Analysis 10 — Parlay WR vs targets table.

    Compares actual win rates from mock_bets by leg count against
    the margin-tightened targets set by the personal edge profile system.

    Targets (from 594-bet baseline + CUSHION-grade projections):
      2-leg: 65%  |  3-leg: 57%  |  4-leg: 53%  |  5-leg: 53%  |  overall: 35%

    Historical baseline (pre-margin-tightening):
      2-leg: 29.3%  |  4-leg: 33.3%  |  1-leg: 59.3%

    Shows rolling window (default 30 days) AND all-time from mock_bets.
    Also reports grade_change detection: profiles that flipped grade in the
    last 7 days (CUSHION→CLOSE, CLOSE→CUSHION, etc.) from personal_edge_profile.
    """
    if not os.path.exists(db_path):
        return {"rows": [], "note": "db_not_found"}

    _TARGETS: dict[int, float] = {1: 0.593, 2: 0.65, 3: 0.57, 4: 0.53, 5: 0.53}
    _OVERALL_TARGET = 0.35

    try:
        import sqlite3 as _sq
        from datetime import timedelta as _td
        con = _sq.connect(db_path)

        # All-time parlay WR by leg count
        alltime_rows = con.execute("""
            SELECT n_legs,
                   COUNT(*) as n,
                   SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins
            FROM   mock_bets
            WHERE  is_mock = 1
              AND  outcome IN ('win', 'loss')
            GROUP  BY n_legs
            ORDER  BY n_legs
        """).fetchall()

        # Rolling window
        cutoff_iso = (datetime.utcnow() - _td(days=rolling_days)).isoformat()
        rolling_rows = con.execute("""
            SELECT n_legs,
                   COUNT(*) as n,
                   SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins
            FROM   mock_bets
            WHERE  is_mock = 1
              AND  outcome IN ('win', 'loss')
              AND  created_at >= ?
            GROUP  BY n_legs
            ORDER  BY n_legs
        """, (cutoff_iso,)).fetchall()

        # Grade changes in last 7 days — compare current grade vs grade 7+ days ago
        # We use the personal_edge_profile last_updated field as a proxy.
        # Any profile updated within the last 7 days is considered "recently changed".
        grade_change_cutoff = (datetime.utcnow() - _td(days=7)).isoformat()
        grade_change_rows = con.execute("""
            SELECT sport, market_type, line_bucket, margin_grade, personal_wr, last_updated
            FROM   personal_edge_profile
            WHERE  last_updated >= ?
            ORDER  BY last_updated DESC
        """, (grade_change_cutoff,)).fetchall()

        con.close()
    except Exception as e:
        return {"rows": [], "note": str(e)}

    def _build_wr_table(raw_rows: list) -> dict:
        by_legs: dict[int, dict] = {}
        total_n = total_wins = 0
        for leg_count, n, wins in raw_rows:
            wr = wins / n if n else 0.0
            target = _TARGETS.get(leg_count)
            by_legs[leg_count] = {
                "n_legs": leg_count,
                "n": n,
                "wins": wins,
                "actual_wr": round(wr, 4),
                "target_wr": target,
                "vs_target": round(wr - target, 4) if target else None,
                "meets_target": (wr >= target) if target else None,
            }
            total_n   += n
            total_wins += wins
        overall_wr = total_wins / total_n if total_n else 0.0
        return {
            "by_legs": by_legs,
            "overall_n": total_n,
            "overall_wins": total_wins,
            "overall_wr": round(overall_wr, 4),
            "overall_target": _OVERALL_TARGET,
            "overall_meets_target": overall_wr >= _OVERALL_TARGET,
        }

    alltime  = _build_wr_table(alltime_rows)
    rolling  = _build_wr_table(rolling_rows)

    # Print table
    print(f"\n  ANALYSIS 10 — Parlay WR vs Targets  (rolling={rolling_days}d / all-time)")
    print()
    hdr = f"  {'LEGS':>4}  {'TARGET':>7}  {'ACTUAL(roll)':>12}  {'vs TARGET':>10}  {'OK?':>4}  {'N(roll)':>8}  {'ACTUAL(all)':>12}  {'N(all)':>8}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    all_leg_counts = sorted(set(list(alltime["by_legs"].keys()) + list(rolling["by_legs"].keys())))
    for lc in all_leg_counts:
        r = rolling["by_legs"].get(lc)
        a = alltime["by_legs"].get(lc)
        target = _TARGETS.get(lc)
        t_s = "%.1f%%" % (target * 100) if target else "  —  "
        r_wr_s = ("%.1f%%" % (r["actual_wr"] * 100)) if r else "  —  "
        r_n_s  = str(r["n"]) if r else "—"
        r_vs_s = ("%+.1f%%" % (r["vs_target"] * 100)) if (r and r["vs_target"] is not None) else "  —  "
        r_ok_s = ("YES" if r["meets_target"] else "NO ") if (r and r["meets_target"] is not None) else " — "
        a_wr_s = ("%.1f%%" % (a["actual_wr"] * 100)) if a else "  —  "
        a_n_s  = str(a["n"]) if a else "—"
        print(f"  {lc:>4}  {t_s:>7}  {r_wr_s:>12}  {r_vs_s:>10}  {r_ok_s:>4}  {r_n_s:>8}  {a_wr_s:>12}  {a_n_s:>8}")

    # Overall line
    r_overall = rolling["overall_wr"]
    a_overall = alltime["overall_wr"]
    r_ok = "YES" if rolling["overall_meets_target"] else "NO "
    print(f"  {'ALL':>4}  {'35.0%':>7}  {'%.1f%%' % (r_overall*100):>12}  "
          f"{'%+.1f%%' % ((r_overall - _OVERALL_TARGET)*100):>10}  {r_ok:>4}  "
          f"{rolling['overall_n']:>8}  {'%.1f%%' % (a_overall*100):>12}  {alltime['overall_n']:>8}")

    # Grade changes this week
    if grade_change_rows:
        print(f"\n  GRADE CHANGES (last 7 days):")
        for sport, mt, bkt, grade, wr, updated in grade_change_rows:
            wr_s = "%.1f%%" % (wr * 100) if wr is not None else "N/A"
            print(f"    {sport} {mt} {bkt}: {grade} ({wr_s}) — updated {updated[:10]}")
    else:
        print(f"\n  No grade changes in the last 7 days.")

    return {
        "rolling_days":  rolling_days,
        "alltime":       alltime,
        "rolling":       rolling,
        "grade_changes": [
            {"sport": r[0], "market_type": r[1], "line_bucket": r[2],
             "grade": r[3], "personal_wr": r[4], "last_updated": r[5]}
            for r in grade_change_rows
        ],
    }


def analysis_11_line_quality(db_path: str) -> dict:
    """
    Analysis 11 — Line Quality Analysis.

    Measures two-dimensional line selection quality for settled mock bet legs:
      1. Directional accuracy  — are we picking the right side of the main market?
      2. Line precision        — how far is our line from optimal?
      3. A/B comparison        — of our losses, how many were one step too far?

    Requires the new line-quality columns added in mock_bet_legs (2026-04-30).
    Falls back gracefully if columns don't exist yet.
    """
    if not os.path.exists(db_path):
        return {"error": "db not found"}

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Check if new columns exist
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(mock_bet_legs)").fetchall()}
        if "direction_correct" not in cols:
            con.close()
            print("\n  ANALYSIS 11 — Line Quality Analysis")
            print("  [skip] line-quality columns not yet migrated — run settle to populate")
            return {"skipped": True, "reason": "columns_missing"}
    except Exception as e:
        con.close()
        return {"error": str(e)}

    src_ph = ",".join("?" * len(SIM_SOURCES))
    try:
        rows = con.execute(f"""
            SELECT
                ml.sport,
                ml.market_type,
                ml.leg_result,
                ml.direction_correct,
                ml.line_delta,
                ml.ab_alt_result,
                mb.status  AS bet_status
            FROM mock_bet_legs ml
            JOIN mock_bets mb ON ml.mock_bet_id = mb.id
            WHERE mb.source IN ({src_ph})
              AND mb.generated_at >= ?
              AND mb.status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND ml.leg_result IN ('WIN', 'LOSS')
              AND ml.direction_correct IS NOT NULL
            ORDER BY mb.generated_at
        """, list(SIM_SOURCES) + [SIM_START_DATE]).fetchall()
    except Exception as e:
        con.close()
        return {"error": str(e)}
    finally:
        con.close()

    n_total = len(rows)
    print(f"\n  ANALYSIS 11 — Line Quality Analysis  (n={n_total} legs with direction data)")

    if n_total == 0:
        print("  [skip] No legs with line-quality data yet — settle more bets.")
        return {"n": 0, "insufficient_data": True}

    # ── Overall metrics ───────────────────────────────────────────────────────
    dir_correct = sum(1 for r in rows if r["direction_correct"] == 1)
    dir_acc     = dir_correct / n_total
    losses      = [r for r in rows if r["leg_result"] == "LOSS"]
    n_losses    = len(losses)
    ab_rescue   = sum(1 for r in losses if r["ab_alt_result"] == "WIN")
    real_losses = n_losses - ab_rescue
    deltas      = [r["line_delta"] for r in rows if r["line_delta"] is not None]
    avg_delta   = sum(deltas) / len(deltas) if deltas else None

    print(f"\n  Directional accuracy:  {dir_acc*100:.1f}%  ({dir_correct}/{n_total} legs on right side of main market)")
    if avg_delta is not None:
        bias = "too aggressive ↑" if avg_delta > 0.3 else ("too conservative ↓" if avg_delta < -0.3 else "well-calibrated ✓")
        print(f"  Avg line delta:        {avg_delta:+.2f}  ({bias})")
    if n_losses:
        print(f"  A/B recovery rate:     {ab_rescue/n_losses*100:.1f}%  of losses would win with one-step-closer line")
        print(f"  Real loss rate:        {real_losses/n_total*100:.1f}%  wrong direction entirely (no line fix helps)")

    # ── Per-sport/market breakdown ────────────────────────────────────────────
    from collections import defaultdict
    groups: dict = defaultdict(lambda: {"n": 0, "dir": 0, "losses": 0, "ab_win": 0,
                                         "real_loss": 0, "deltas": []})
    for r in rows:
        sport  = (r["sport"] or "Unknown").split("|")[0].strip()[:18]
        market = (r["market_type"] or "unknown").lower()
        mkey   = "total" if "total" in market else ("spread" if "spread" in market else market)
        key    = f"{sport} / {mkey}"
        g = groups[key]
        g["n"] += 1
        if r["direction_correct"] == 1:
            g["dir"] += 1
        if r["leg_result"] == "LOSS":
            g["losses"] += 1
            if r["ab_alt_result"] == "WIN":
                g["ab_win"] += 1
            else:
                g["real_loss"] += 1
        if r["line_delta"] is not None:
            g["deltas"].append(r["line_delta"])

    # Print table
    print()
    W = 26
    hdr = (f"  {'Sport / Market':<{W}}  {'n':>5}  {'Dir Acc%':>8}  {'Avg Δ':>7}"
           f"  {'AB Win%':>7}  {'Real Loss%':>10}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    breakdown_out = []
    for key, g in sorted(groups.items(), key=lambda x: -x[1]["n"]):
        n    = g["n"]
        da_s = f"{g['dir']/n*100:.1f}%" if n else "  —  "
        dl_s = (f"{sum(g['deltas'])/len(g['deltas']):+.2f}" if g["deltas"] else "  —  ")
        ab_s = (f"{g['ab_win']/g['losses']*100:.1f}%" if g["losses"] else "  —  ")
        rl_s = f"{g['real_loss']/n*100:.1f}%" if n else "  —  "
        print(f"  {key:<{W}}  {n:>5}  {da_s:>8}  {dl_s:>7}  {ab_s:>7}  {rl_s:>10}")
        breakdown_out.append({
            "group": key, "n": n,
            "dir_acc_pct":   round(g["dir"] / n * 100, 1) if n else None,
            "avg_delta":     round(sum(g["deltas"]) / len(g["deltas"]), 2) if g["deltas"] else None,
            "ab_win_pct":    round(g["ab_win"] / g["losses"] * 100, 1) if g["losses"] else None,
            "real_loss_pct": round(g["real_loss"] / n * 100, 1) if n else None,
        })

    # Interpretation
    if n_total >= 10 and avg_delta is not None:
        print()
        if dir_acc >= 0.80:
            print("  ✓ Directional accuracy ≥80% — direction selection is solid.")
        elif dir_acc >= 0.65:
            print("  ~ Directional accuracy 65–80% — acceptable, monitor for improvement.")
        else:
            print("  ✗ Directional accuracy <65% — model may be picking wrong sides of lines.")

        if avg_delta > 0.5:
            print(f"  → Avg delta +{avg_delta:.2f}: systematically too aggressive."
                  "  ALE calibration: try one step closer to main market.")
        elif avg_delta < -0.5:
            print(f"  → Avg delta {avg_delta:.2f}: leaving EV on the table."
                  "  ALE may safely go one step further from main.")

    return {
        "n":               n_total,
        "dir_accuracy":    round(dir_acc, 4),
        "avg_line_delta":  round(avg_delta, 4) if avg_delta is not None else None,
        "ab_recovery_pct": round(ab_rescue / n_losses * 100, 1) if n_losses else None,
        "real_loss_pct":   round(real_losses / n_total * 100, 1) if n_total else None,
        "breakdown":       breakdown_out,
    }


def analysis_12_boost_performance(db_path: str) -> dict:
    """
    Analysis 12 — Boost Performance (smart strategy).

    Measures mock bet performance broken down by promo_type and promo_boost_pct.
    Compares against the historical FanDuel baselines established from real bets.

    Historical baselines (169 FanDuel bets, 2026-01-26 → 2026-05-02):
        +25% PROFIT_BOOST:  31.7% WR  (already profitable — target: ≥ 31.7%)
        +30% PROFIT_BOOST:  10.7% WR  (corrected strategy target: ≥ 25.0%)
        +50% PROFIT_BOOST:   9.1% WR  (corrected strategy target: ≥ 30.0%)
        BONUS_BET:           0.0% WR  (free roll — any win is pure profit)
        NO_SWEAT:           28.6% WR
        No promo:           28.8% WR
    """
    import sqlite3
    if not os.path.exists(db_path):
        return {"error": "db not found"}

    _TARGETS = {
        ("PROFIT_BOOST", 0.25): 31.7,
        ("PROFIT_BOOST", 0.30): 25.0,
        ("PROFIT_BOOST", 0.50): 30.0,
        ("BONUS_BET",    0.0):   0.0,  # any win = profit
        ("NO_SWEAT",     0.0):  28.6,
        (None,           0.0):  28.8,
    }

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Check columns exist
    cols = {r[1] for r in con.execute("PRAGMA table_info(mock_bets)").fetchall()}
    if "promo_type" not in cols:
        con.close()
        print("\n  ANALYSIS 12 — Boost Performance")
        print("  [skip] promo_type column not yet present — restart server to migrate")
        return {"skipped": True, "reason": "columns_missing"}

    src_ph = ",".join("?" * len(SIM_SOURCES))
    rows = con.execute(f"""
        SELECT
            promo_type,
            ROUND(COALESCE(promo_boost_pct, 0), 2)   AS boost_pct,
            COUNT(*)                                   AS bets,
            SUM(CASE WHEN status = 'SETTLED_WIN' THEN 1 ELSE 0 END) AS wins,
            ROUND(SUM(COALESCE(actual_profit, 0)), 2)  AS pnl,
            ROUND(AVG(COALESCE(predicted_win_prob, 0)), 1) AS avg_wp,
            ROUND(AVG(COALESCE(avg_lqs, 0)), 1)        AS avg_lqs
        FROM mock_bets
        WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS')
          AND source IN ({src_ph})
        GROUP BY promo_type, boost_pct
        ORDER BY boost_pct DESC, promo_type
    """, list(SIM_SOURCES)).fetchall()
    con.close()

    results = []
    print("\n  ANALYSIS 12 — Boost Performance (smart strategy)")
    print(f"  {'Promo':<18} {'Boost':>6}  {'Bets':>5}  {'WR%':>6}  {'P&L':>8}  {'AvgWP':>6}  {'Target':>7}  {'Status':>8}")
    print("  " + "─" * 80)

    for r in rows:
        promo   = r["promo_type"] or "None"
        boost   = r["boost_pct"] or 0.0
        n       = r["bets"]
        w       = r["wins"]
        wr      = round(w / n * 100, 1) if n else 0.0
        pnl     = r["pnl"] or 0.0
        avg_wp  = r["avg_wp"] or 0.0
        avg_lqs_val = r["avg_lqs"] or 0.0

        key     = (r["promo_type"], boost)
        target  = _TARGETS.get(key)
        status  = ""
        if target is not None and n >= 5:
            status = "✓ on track" if wr >= target else f"✗ need {target:.0f}%"

        print(f"  {promo:<18} {boost:>5.0%}  {n:>5}  {wr:>5.1f}%  {pnl:>8.2f}  {avg_wp:>5.1f}%  "
              f"  {(str(target)+'%') if target else '—':>7}  {status}")

        results.append({
            "promo_type":   promo,
            "boost_pct":    boost,
            "bets":         n,
            "wins":         w,
            "win_rate":     wr,
            "pnl":          pnl,
            "avg_win_prob": avg_wp,
            "avg_lqs":      avg_lqs_val,
            "target_wr":    target,
            "on_track":     (wr >= target) if (target is not None and n >= 5) else None,
        })

    return {"by_boost_tier": results}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BetIQ signal analysis — standalone, read-only"
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Append results as JSON to this file (e.g. data/analysis_log.json)",
    )
    parser.add_argument(
        "--db",
        default=_DB_PATH,
        metavar="PATH",
        help=f"SQLite mock-bets database (default: {_DB_PATH})",
    )
    args = parser.parse_args()

    # Resolve relative paths from repo root
    db_path = args.db if os.path.isabs(args.db) else os.path.join(_ROOT, args.db)

    print(f"\nBetIQ Signal Analysis  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Database:     {db_path}")
    print(f"ATS cutoff:   {ATS_FIX_CUTOFF} UTC")

    rows = load_rows(db_path)
    n_all = len(rows)
    wr_all = round(sum(r["won"] for r in rows) / n_all * 100, 1) if n_all else 0
    print(f"Settled bets: {n_all}  (win_rate={wr_all}%)")

    if n_all == 0:
        print("\n  No settled bets found — nothing to analyse.")
        return {}

    results: dict = {
        "run_at":    datetime.utcnow().isoformat(),
        "db_path":   db_path,
        "n_total":   n_all,
        "win_rate":  wr_all,
    }

    results["analysis_1"] = analysis_1_correlations(rows, db_path=db_path)
    results["analysis_2"] = analysis_2_monotonicity(rows)
    results["analysis_3"] = analysis_3_ats_impact(rows)
    results["analysis_4"] = analysis_4_sample_size(rows)
    results["analysis_5"] = analysis_5_correlation_matrix(rows)
    results["analysis_6"] = analysis_6_regime_performance(db_path)
    results["analysis_7"] = analysis_7_lqs_trend(db_path)
    results["analysis_8"] = analysis_8_regime_ab(db_path)
    results["analysis_9"]  = analysis_9_margin_profile(db_path)
    results["analysis_10"] = analysis_10_parlay_wr_targets(db_path)
    results["analysis_11"] = analysis_11_line_quality(db_path)
    results["analysis_12"] = analysis_12_boost_performance(db_path)

    # Scenario sim summary (T2 — human-hypothesis bets, tracked separately)
    scenario_rows = [r for r in rows if r.get("source") == "scenario_sim"]
    if scenario_rows:
        s_n    = len(scenario_rows)
        s_wins = sum(r["won"] for r in scenario_rows)
        s_wr   = round(s_wins / s_n * 100, 1) if s_n else None
        print("\n" + "═" * 60)
        print("SCENARIO SIM — Human-hypothesis bets (Tier 2)")
        print("═" * 60)
        print(f"  n={s_n}  wins={s_wins}  win_rate={s_wr}%")
        results["scenario_sim"] = {"n": s_n, "wins": s_wins, "win_rate": s_wr}
    else:
        results["scenario_sim"] = {"n": 0, "note": "no settled scenario_sim bets yet"}

    # Performance snapshot (writes to DB — not read-only like analyses 1-6)
    print("\n" + "═" * 60)
    print("PERFORMANCE SNAPSHOT — Weekly segmentation")
    print("═" * 60)
    migrate_performance_log(db_path)
    snapshot = compute_weekly_snapshot(db_path)
    results["performance_snapshot"] = snapshot

    print("\n" + "═" * 60)
    print("Done.")

    if args.output:
        out_path = (args.output if os.path.isabs(args.output)
                    else os.path.join(_ROOT, args.output))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        log: list = []
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                    log = existing if isinstance(existing, list) else [existing]
            except Exception:
                pass
        log.append(results)

        with open(out_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"Results appended → {out_path}")

    return results


if __name__ == "__main__":
    main()
