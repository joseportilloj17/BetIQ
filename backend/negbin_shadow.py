"""
negbin_shadow.py — NegBin forward shadow with band-for-band tracking.

Runs alongside the live ATS pipeline. Does NOT replace live picks.
Stores shadow parlays as MockBet/MockBetLeg with source="shadow_negbin".
Tracks band-for-band edge-over-breakeven for forward validation.

Deployment posture:
  - Anchor on 1.7-2.2x band (proven stable edge ~+5pp)
  - Higher-odds bands are opportunistic upside (conservative +3-6pp baseline)
  - 2×2 cell rule: back Fav@Away, fade Dog@Home (market mispricing)
"""
from __future__ import annotations
import os, sys, uuid, json, sqlite3, warnings
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_CT = timezone(timedelta(hours=-6))
_BETS_DB   = os.path.join(_HERE, "..", "data", "bets.db")
_HIST_DB   = os.path.join(_HERE, "..", "data", "historical.db")

# ── Selection gates (tuned for raw NegBin scale) ──
NEGBIN_GATE      = 0.55    # model WP floor (same as ATS — raw NegBin is already calibrated OOS)
MKT_IMPLIED_FLOOR = 0.42   # market implied floor
LOS_FLOOR        = 0.52    # LOS floor
LIVE_RUNGS       = [1.5, 2.5, 3.5, 4.5]
MAX_SHADOW_PARLAYS = 50    # total across all bands
PER_BAND_TARGET   = 10     # aim for ~10 per band
STAKE            = 10.0

# ── 2×2 cell rule (from Check 1 validation) ──
# Dog@Home overpriced 7-9pp → penalize. Fav@Away underpriced 6-10pp → boost.
CELL_BOOST = {
    "fav_away": 1.10,    # 10% composite boost (underpriced by market)
    "dog_home": 0.70,    # 30% composite penalty (overpriced by market)
    "fav_home": 1.00,    # fairly priced — no adjustment
    "dog_away": 1.00,    # fairly priced — no adjustment
}

# ── Odds bands for tracking ──
ODDS_BANDS = [
    ("1.3-1.7x", 1.3, 1.7),
    ("1.7-2.2x", 1.7, 2.2),
    ("2.2-2.8x", 2.2, 2.8),
    ("2.8-3.5x", 2.8, 3.5),
    ("3.5-5.0x", 3.5, 5.0),
]


def _ensure_shadow_columns():
    """Add NegBin shadow columns to mock_bet_legs if missing."""
    migrations = [
        ("mock_bet_legs", "negbin_wp",         "ALTER TABLE mock_bet_legs ADD COLUMN negbin_wp REAL"),
        ("mock_bet_legs", "negbin_mu_home",    "ALTER TABLE mock_bet_legs ADD COLUMN negbin_mu_home REAL"),
        ("mock_bet_legs", "negbin_mu_away",    "ALTER TABLE mock_bet_legs ADD COLUMN negbin_mu_away REAL"),
        ("mock_bet_legs", "negbin_cell",       "ALTER TABLE mock_bet_legs ADD COLUMN negbin_cell TEXT"),
        ("mock_bet_legs", "negbin_edge_pp",    "ALTER TABLE mock_bet_legs ADD COLUMN negbin_edge_pp REAL"),
        ("mock_bet_legs", "ats_cover_wp",      "ALTER TABLE mock_bet_legs ADD COLUMN ats_cover_wp REAL"),
        ("mock_bet_legs", "ats_dist_wp",       "ALTER TABLE mock_bet_legs ADD COLUMN ats_dist_wp REAL"),
        ("mock_bet_legs", "bayes_team_wp",     "ALTER TABLE mock_bet_legs ADD COLUMN bayes_team_wp REAL"),
        ("mock_bet_legs", "mkt_disagree_dist", "ALTER TABLE mock_bet_legs ADD COLUMN mkt_disagree_dist REAL"),
        ("mock_bet_legs", "ridge_composite",   "ALTER TABLE mock_bet_legs ADD COLUMN ridge_composite REAL"),
        ("mock_bets",     "negbin_odds_band",  "ALTER TABLE mock_bets ADD COLUMN negbin_odds_band TEXT"),
        ("mock_bets",     "negbin_cell_adj",   "ALTER TABLE mock_bets ADD COLUMN negbin_cell_adj REAL"),
        ("mock_bets",     "negbin_leg_count",  "ALTER TABLE mock_bets ADD COLUMN negbin_leg_count INTEGER"),
        ("mock_bets",     "live_realistic",    "ALTER TABLE mock_bets ADD COLUMN live_realistic INTEGER DEFAULT 0"),
        ("mock_bets",     "mean_leg_dist_wp",  "ALTER TABLE mock_bets ADD COLUMN mean_leg_dist_wp REAL"),
        ("mock_bets",     "min_leg_dist_wp",   "ALTER TABLE mock_bets ADD COLUMN min_leg_dist_wp REAL"),
        ("mock_bets",     "concentrated_risk", "ALTER TABLE mock_bets ADD COLUMN concentrated_risk INTEGER DEFAULT 0"),
    ]
    try:
        con = sqlite3.connect(_BETS_DB, timeout=15)
        cur = con.cursor()
        for tbl, col, stmt in migrations:
            cur.execute(f"PRAGMA table_info({tbl})")
            existing = {row[1] for row in cur.fetchall()}
            if col not in existing:
                cur.execute(stmt)
                print(f"[negbin-shadow] migration: added {tbl}.{col}")
        con.commit()
        con.close()
    except Exception as e:
        print(f"[negbin-shadow] migration error: {e}")


# ── Bayesian team cover rate cache ──
_BAYES_TEAM_CACHE: dict | None = None


def _build_bayes_team_cache():
    """Build James-Stein shrunk team cover rates from settled Shadow A legs.

    OOS discrimination confirmed: rho=+0.435, quintile Q0=45%→Q4=98%.
    Uses only source='shadow_negbin' settled legs to avoid leakage.
    Retries once on transient SQLite lock (fix for 2026-07-18 silent failure).
    """
    global _BAYES_TEAM_CACHE
    import time as _time

    for attempt in range(2):
        try:
            con = sqlite3.connect(_BETS_DB, timeout=30)
            rows = con.execute("""
                SELECT mbl.description, mbl.leg_result
                FROM mock_bet_legs mbl
                JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
                WHERE mb.source = 'shadow_negbin'
                AND mbl.leg_result IN ('WIN', 'LOSS')
                AND mbl.negbin_wp IS NOT NULL
            """).fetchall()
            con.close()

            if len(rows) < 100:
                print(f"[negbin-shadow] Bayes cache: insufficient data ({len(rows)} legs)")
                return

            # Parse team name from description (e.g. "Detroit Tigers +1.5" → "Detroit Tigers")
            import re
            team_wins = defaultdict(int)
            team_n = defaultdict(int)
            for desc, result in rows:
                if not desc:
                    continue
                m = re.match(r'(.+?)\s+[+-]?\d+\.?\d*$', desc.strip())
                if m:
                    team = m.group(1).strip()
                    team_n[team] += 1
                    if result == "WIN":
                        team_wins[team] += 1

            if not team_n:
                return

            grand_wr = sum(team_wins.values()) / sum(team_n.values())
            grand_var = grand_wr * (1 - grand_wr)

            # James-Stein shrinkage
            team_rates = {t: team_wins[t] / team_n[t] for t in team_n}
            avg_n = np.mean(list(team_n.values()))
            rate_var = np.var(list(team_rates.values()))
            between_var = max(rate_var - grand_var / avg_n, 0.001)

            cache = {}
            for team in team_n:
                n = team_n[team]
                shrinkage = between_var / (between_var + grand_var / n)
                raw_rate = team_wins[team] / n
                shrunk = shrinkage * raw_rate + (1 - shrinkage) * grand_wr
                cache[team] = shrunk

            _BAYES_TEAM_CACHE = cache
            print(f"[negbin-shadow] Bayes cache built: {len(cache)} teams, "
                  f"grand WR={grand_wr:.3f}, avg shrinkage={np.mean([between_var / (between_var + grand_var / team_n[t]) for t in team_n]):.3f}")
            return  # success
        except Exception as exc:
            if attempt == 0:
                print(f"[negbin-shadow] Bayes cache attempt 1 failed ({exc}), retrying in 5s...")
                _time.sleep(5)
            else:
                print(f"CRITICAL [negbin-shadow] Bayes cache FAILED after retry: {exc}")
                print(f"[negbin-shadow] All downstream signals (ridge_composite, mkt_disagree) will be NULL this run")


def _lookup_market_implied(fixture_id: str, team: str, spread_value) -> float | None:
    """Look up market implied cover probability from alt_lines in historical.db.

    Returns the latest (close) odds converted to implied probability, or None.
    """
    if not fixture_id or not team or spread_value is None:
        return None
    try:
        con = sqlite3.connect(_HIST_DB, timeout=15)
        row = con.execute("""
            SELECT odds FROM alt_lines
            WHERE event_id = ? AND team = ? AND line = ?
            AND market_key = 'alternate_spreads'
            ORDER BY fetched_at DESC LIMIT 1
        """, (fixture_id, team, float(spread_value))).fetchone()
        con.close()
        if row and row[0] and row[0] > 1.0:
            return 1.0 / row[0]  # decimal odds → implied prob
        return None
    except Exception:
        return None


# ── Ridge composite model cache ──
# Ridge logistic: bayes_team_wp + ats_dist_wp, C=0.005 (nested-CV selected).
# OOS validated: Brier 0.1340 (nested CV), +0.017 over raw dist_wp.
# Tracked column only — not wired into gates or sizing.
_RIDGE_MODEL: tuple | None = None  # (scaler, model) or None


def _build_ridge_model():
    """Fit ridge logistic on settled shadow_negbin legs with both signals.

    Uses C=0.005 (selected by nested purged CV, not in-sample picked).
    Requires ≥200 legs with both bayes_team_wp and ats_dist_wp.
    """
    global _RIDGE_MODEL
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        con = sqlite3.connect(_BETS_DB, timeout=15)
        rows = con.execute("""
            SELECT mbl.bayes_team_wp, mbl.ats_dist_wp, mbl.leg_result
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source = 'shadow_negbin'
            AND mbl.leg_result IN ('WIN', 'LOSS')
            AND mbl.bayes_team_wp IS NOT NULL
            AND mbl.ats_dist_wp IS NOT NULL
        """).fetchall()
        con.close()

        if len(rows) < 200:
            print(f"[negbin-shadow] Ridge model: insufficient data ({len(rows)} legs)")
            return

        X = np.array([[r[0], r[1]] for r in rows])
        y = np.array([1 if r[2] == "WIN" else 0 for r in rows])

        sc = StandardScaler()
        X_s = sc.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, C=0.005, penalty="l2", solver="lbfgs")
        lr.fit(X_s, y)

        _RIDGE_MODEL = (sc, lr)
        print(f"[negbin-shadow] Ridge model built: {len(rows)} legs, "
              f"coefs=[bayes={lr.coef_[0][0]:+.3f}, dist={lr.coef_[0][1]:+.3f}], "
              f"intercept={lr.intercept_[0]:+.3f}")
    except Exception as exc:
        print(f"[negbin-shadow] Ridge model error: {exc}")


def _ridge_predict(bayes_wp: float, dist_wp: float) -> float | None:
    """Score a single leg with the ridge composite model."""
    if _RIDGE_MODEL is None:
        return None
    sc, lr = _RIDGE_MODEL
    X = sc.transform([[bayes_wp, dist_wp]])
    return float(lr.predict_proba(X)[0, 1])


def _get_todays_fixtures(today_str: str, only_unstarted: bool = True):
    """Get today's MLB fixtures from bets.db.

    Args:
        only_unstarted: When True (default), exclude games whose
            commence_time has already passed.  Safe for late catch-up
            runs triggered by the morning-chain safety net.
    """
    con = sqlite3.connect(_BETS_DB, timeout=15)
    rows = con.execute("""
        SELECT id, home_team, away_team, game_date_ct, bookmakers, commence_time
        FROM fixtures
        WHERE sport_key = 'baseball_mlb'
          AND game_date_ct = ?
          AND bookmakers IS NOT NULL
    """, (today_str,)).fetchall()
    con.close()

    # Filter out already-started games when running as a late catch-up.
    if only_unstarted:
        from datetime import datetime as _dt, timezone as _tz
        _now_str = _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S")
        _before = len(rows)
        rows = [r for r in rows if not r[5] or r[5] > _now_str]
        _skipped = _before - len(rows)
        if _skipped:
            print(f"[NegBin] Skipped {_skipped} already-started fixtures")

    fixtures = []
    for fid, home, away, gd, bk_json, commence in rows:
        # Determine ML favorite from bookmakers
        home_is_fav = None
        try:
            bks = json.loads(bk_json) if bk_json else []
            for bk in bks:
                for mkt in bk.get("markets", []):
                    if mkt.get("key") == "h2h":
                        outcomes = {o["name"]: o.get("price", 0) for o in mkt.get("outcomes", [])}
                        hp = outcomes.get(home, 0)
                        ap = outcomes.get(away, 0)
                        if hp and ap:
                            home_is_fav = hp < ap
                        break
                if home_is_fav is not None:
                    break
        except Exception:
            pass

        fixtures.append({
            "fixture_id": fid, "home_team": home, "away_team": away,
            "game_date": gd, "home_is_fav": home_is_fav,
            "commence_time": commence,
        })

    return fixtures


def _get_market_implied(home_team: str, away_team: str, game_date: str,
                        rung: float, side: str):
    """Get market implied probability for a spread pick.

    Joins on home_team + away_team + game_date (fixture IDs differ between DBs).
    Falls back to game_date+1 for late-start games whose alt-lines are filed
    under the next UTC day (systematic: ~84% of missing coverage).
    Returns (mkt_wp, decimal_odds) or (None, None).
    """
    from datetime import date as _date, timedelta as _td
    con = sqlite3.connect(_HIST_DB, timeout=15)

    # Try game_date, then game_date+1 as fallback
    try:
        next_date = (
            _date.fromisoformat(game_date) + _td(days=1)
        ).isoformat()
    except Exception:
        next_date = None
    dates_to_try = [game_date]
    if next_date:
        dates_to_try.append(next_date)

    if side == "home":
        for gd in dates_to_try:
            row = con.execute("""
                SELECT odds FROM alt_lines
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                  AND market_key = 'alternate_spreads'
                  AND team = ?
                  AND ABS(line - ?) < 0.26
                  AND odds IS NOT NULL AND odds > 1.0
                ORDER BY fetched_at DESC LIMIT 1
            """, (home_team, away_team, gd, home_team, rung)).fetchone()
            if row is not None:
                con.close()
                dec_odds = float(row[0])
                mkt_wp = 1.0 / dec_odds
                return mkt_wp, dec_odds
        con.close()
        return None, None
    else:
        # Away at +rung: need 1 - P(home covers -rung)
        neg_rung = -rung
        for gd in dates_to_try:
            row = con.execute("""
                SELECT odds FROM alt_lines
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                  AND market_key = 'alternate_spreads'
                  AND team = ?
                  AND ABS(line - ?) < 0.26
                  AND odds IS NOT NULL AND odds > 1.0
                ORDER BY fetched_at DESC LIMIT 1
            """, (home_team, away_team, gd, home_team, neg_rung)).fetchone()

            row_away = con.execute("""
                SELECT odds FROM alt_lines
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                  AND market_key = 'alternate_spreads'
                  AND team = ?
                  AND ABS(line - ?) < 0.26
                  AND odds IS NOT NULL AND odds > 1.0
                ORDER BY fetched_at DESC LIMIT 1
            """, (home_team, away_team, gd, away_team, rung)).fetchone()

            if row is not None:
                con.close()
                home_neg_odds = float(row[0])
                mkt_home_neg = 1.0 / home_neg_odds
                mkt_away_wp = 1.0 - mkt_home_neg
                dec_odds = float(row_away[0]) if row_away else (1.0 / mkt_away_wp if mkt_away_wp > 0 else None)
                return mkt_away_wp, dec_odds

        con.close()
        return None, None


_LIVE_LQS_FLOOR = 55.0         # live's per-leg LQS minimum
_LIVE_MAX_BETS_PER_GAME = 3    # live's per-fixture cap
_LIVE_MAX_PARLAYS = 40         # live's standard-pick budget


def _select_live_realistic(all_scored: list) -> set:
    """
    Re-select from the full scored-combo pool using live's structural rules.

    Returns a set of bet_keys (tuple of sorted leg_ids) that would survive
    live's selection. This is a pure subset of the measurement set — same
    candidates, stricter assembly.

    Key differences from measurement assembly:
      - Strict no leg reuse (each leg_id in at most 1 parlay)
      - Pure composite rank (no round-robin band-filling)
      - MAX_BETS_PER_GAME = 3 per fixture
      - LQS >= 55 per leg
      - Cap at 40 parlays (live's standard budget)
    """
    # Sort ALL scored combos by composite descending (live's selection = top-N)
    ranked = sorted(all_scored, key=lambda x: -x["composite"])

    selected_keys = set()
    used_legs = set()                         # strict: each leg_id used at most once
    fixture_use = defaultdict(int)            # fixture_key -> count across parlays

    for s in ranked:
        if len(selected_keys) >= _LIVE_MAX_PARLAYS:
            break

        legs = s["legs"]
        bet_key = tuple(sorted(l["leg_id"] for l in legs))

        # Skip duplicates
        if bet_key in selected_keys:
            continue

        # LQS floor: every leg must have LQS >= 55
        if any(l.get("lqs", 50) < _LIVE_LQS_FLOOR for l in legs):
            continue

        # Strict no-reuse: no leg already used in a prior parlay
        if any(l["leg_id"] in used_legs for l in legs):
            continue

        # Per-fixture cap: each fixture in at most 3 parlays total
        fixture_keys = [l["fixture_key"] for l in legs]
        if any(fixture_use[fk] >= _LIVE_MAX_BETS_PER_GAME for fk in fixture_keys):
            continue

        # Passes all gates — select it
        selected_keys.add(bet_key)
        for l in legs:
            used_legs.add(l["leg_id"])
        for fk in fixture_keys:
            fixture_use[fk] += 1

    return selected_keys


_FH_MAX_PARLAYS = 15   # cap for fav_home carve-out (measurement volume)
_FH_MAX_LEG_APP = 3    # same reuse cap as main shadow


def _assemble_fav_home_carveout(candidates, odds_bands, cell_boost):
    """
    Build parlays from fav_home candidates ONLY — isolated allocation.

    Same assembly mechanics as the main shadow but restricted to the
    fav_home candidate pool. Uses pure composite rank (no band round-robin)
    since the pool is small.
    """
    import itertools

    fh_pool = [c for c in candidates
               if c.get("cell") == "fav_home"
               and c.get("decimal_odds") is not None
               and c["decimal_odds"] > 1.0]

    if len(fh_pool) < 2:
        print(f"[negbin-shadow-fh] {len(fh_pool)} fav_home candidates — "
              f"need >= 2 for parlays, skipping")
        return []

    fh_pool.sort(key=lambda x: -x["los"])

    for c in fh_pool:
        if "leg_id" not in c:
            c["leg_id"] = f"{c['fixture_key']}:{c['pick_team']}:{c['spread_value']}"

    def _score_combo(combo):
        fixtures_used = set(c["fixture_key"] for c in combo)
        if len(fixtures_used) < len(combo):
            return None
        combined_odds = 1.0
        for c in combo:
            combined_odds *= c["decimal_odds"]
        band = "other"
        for band_label, lo, hi in odds_bands:
            if lo <= combined_odds < hi:
                band = band_label
                break
        avg_wp = np.mean([c["ats_dist_wp"] for c in combo])
        avg_lqs = np.mean([c.get("lqs", 50) for c in combo])
        avg_los = np.mean([c.get("los", 0.5) for c in combo])
        combined_wp = 1.0
        for c in combo:
            combined_wp *= c["ats_dist_wp"]
        norm_ev = max(0, combined_wp * (combined_odds - 1) - (1 - combined_wp))
        composite = (avg_wp * 0.40 + (avg_lqs / 100) * 0.20
                     + avg_los * 0.30 + min(norm_ev, 1.0) * 0.10)
        cell_adj = 1.0
        for c in combo:
            cell_adj *= cell_boost.get(c["cell"], 1.0)
        composite *= cell_adj
        return {
            "legs": list(combo),
            "combined_odds": combined_odds,
            "composite": composite,
            "combined_wp": combined_wp,
            "odds_band": band,
            "cell_adj": round(cell_adj, 4),
            "leg_count": len(combo),
        }

    all_scored = []
    pool_cap = min(len(fh_pool), 30)
    for size in [2, 3]:
        if len(fh_pool) < size:
            continue
        for combo in itertools.combinations(fh_pool[:pool_cap], size):
            result = _score_combo(combo)
            if result is not None:
                all_scored.append(result)

    # Pure composite rank — no band round-robin
    all_scored.sort(key=lambda x: -x["composite"])

    parlays = []
    seen_keys = set()
    leg_use = defaultdict(int)

    for s in all_scored:
        if len(parlays) >= _FH_MAX_PARLAYS:
            break
        bet_key = tuple(sorted(l["leg_id"] for l in s["legs"]))
        if bet_key in seen_keys:
            continue
        if any(leg_use[l["leg_id"]] >= _FH_MAX_LEG_APP for l in s["legs"]):
            continue
        seen_keys.add(bet_key)
        for l in s["legs"]:
            leg_use[l["leg_id"]] += 1
        # Tag for live-realistic check within carve-out
        s["live_realistic"] = 0  # carve-out is measurement-only
        parlays.append(s)

    print(f"[negbin-shadow-fh] {len(fh_pool)} fav_home candidates → "
          f"{len(all_scored)} combos → {len(parlays)} parlays assembled")

    return parlays


def _fit_wp_calibration():
    """Fit isotonic WP recalibration from settled Shadow A legs.

    Uses settled shadow_negbin legs (negbin_wp → leg_result) to learn the
    WP→cover mapping.  Returns a callable wp_transform(raw_wp) → calibrated_wp,
    or None if insufficient data (< 500 settled legs).
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("[negbin-recal] sklearn not available, skipping calibration")
        return None

    try:
        con = sqlite3.connect(_BETS_DB, timeout=15)
        rows = con.execute("""
            SELECT mbl.negbin_wp, mbl.leg_result
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.source = 'shadow_negbin'
              AND mbl.leg_result IN ('WIN', 'LOSS')
              AND mbl.negbin_wp IS NOT NULL
              AND mb.cohort = 'clean'
        """).fetchall()
        con.close()
    except Exception as e:
        print(f"[negbin-recal] DB error: {e}")
        return None

    if len(rows) < 500:
        print(f"[negbin-recal] Only {len(rows)} settled legs — need 500+, skipping")
        return None

    wps = np.array([r[0] for r in rows], dtype=float)
    covers = np.array([1.0 if r[1] == "WIN" else 0.0 for r in rows], dtype=float)

    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(wps, covers)

    # Log the mapping for transparency
    print(f"[negbin-recal] Isotonic fit on {len(rows)} settled legs")
    for probe in [0.55, 0.65, 0.75, 0.85, 0.95]:
        cal = float(iso.predict([probe])[0])
        print(f"[negbin-recal]   WP {probe:.2f} → {cal:.3f} ({cal-probe:+.3f})")

    def wp_transform(raw_wp):
        return float(iso.predict([raw_wp])[0])

    return wp_transform


def generate_negbin_shadow_picks(db=None, today_str=None,
                                 wp_transform=None, source_tag="shadow_negbin",
                                 fixtures=None):
    """Generate NegBin shadow picks for today.

    Returns list of shadow parlay dicts, each with legs, odds, band, cell info.
    Also stores them as MockBet/MockBetLeg with source=source_tag.

    Args:
        wp_transform: Optional callable(raw_wp) → calibrated_wp. When provided,
            all WP-based decisions (gate, edge, LOS, combo scoring) use the
            calibrated value. Used by Shadow B (recalibrated) variant.
        source_tag: DB source tag for stored parlays. Default 'shadow_negbin' (A).
        fixtures: Optional pre-fetched fixture list. When provided, skips
            _get_todays_fixtures() — guarantees A/B see the exact same slate.
    """
    from negbin_model import load_negbin_model, _simulate_margins
    from ml_model import _get_feature_matrix, ats_distributional_cover_prob
    from walk_forward_backtest import _compute_los, _compute_lqs
    import itertools

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    tag_label = "recal" if source_tag == "shadow_negbin_recal" else "shadow"
    print(f"\n[negbin-{tag_label}] Generating {tag_label} picks for {today_str}")

    # Build Bayesian team cover cache if needed (lazy, once per session)
    if _BAYES_TEAM_CACHE is None:
        _build_bayes_team_cache()

    # Build ridge composite model if needed (lazy, once per session)
    if _RIDGE_MODEL is None:
        _build_ridge_model()

    model = load_negbin_model()
    if model is None:
        print(f"[negbin-{tag_label}] ERROR: NegBin model not found")
        return []

    gbm_home = model["gbm_home"]
    gbm_away = model["gbm_away"]
    feat_cols = model["feature_cols"]

    # Get feature matrix (cached, same as live pipeline)
    matrix = _get_feature_matrix("MLB")
    if matrix.empty:
        print(f"[negbin-{tag_label}] ERROR: empty feature matrix")
        return []

    matrix["_gd"] = matrix["game_date"].astype(str).str[:10]

    if fixtures is None:
        fixtures = _get_todays_fixtures(today_str)
    if not fixtures:
        print(f"[negbin-{tag_label}] No MLB fixtures for {today_str}")
        return []

    print(f"[negbin-{tag_label}] {len(fixtures)} MLB fixtures")

    # ── Generate candidates ──
    candidates = []
    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]
        home_is_fav = fix["home_is_fav"]

        # Find this game in the feature matrix
        mask = (
            (matrix["home_team"].str.lower().str.strip() == home.lower().strip())
            & (matrix["away_team"].str.lower().str.strip() == away.lower().strip())
            & (matrix["_gd"] == today_str)
        )
        matched = matrix[mask]
        if matched.empty:
            # Try without date (for games not yet in matrix)
            mask2 = (
                (matrix["home_team"].str.lower().str.strip() == home.lower().strip())
                & (matrix["away_team"].str.lower().str.strip() == away.lower().strip())
            )
            matched = matrix[mask2]
            if matched.empty:
                continue
            matched = matched.sort_values("game_date", ascending=False).head(1)

        row = matched.iloc[0]

        X = np.array([[row[c] if c in row.index and not pd.isna(row[c]) else np.nan
                        for c in feat_cols]], dtype=float)
        mu_home = float(np.clip(gbm_home.predict(X)[0], 0.5, 15))
        mu_away = float(np.clip(gbm_away.predict(X)[0], 0.5, 15))

        fixture_key = f"{home}_{away}_{today_str}"
        margins = _simulate_margins(mu_home, mu_away, n_sims=10000,
                                     seed=hash(fixture_key) % 2**31)

        # P(blowout) filter: suppress tight rungs for extreme-blowout fixtures
        p_blow = float(np.mean(np.abs(margins) >= 5))
        blow_suppress = p_blow >= 0.50  # extreme zone only
        if blow_suppress:
            print(f"[negbin-{tag_label}] {home} v {away}: P(blow)={p_blow:.1%} ≥50% — suppressing +1.5/+2.5")

        for sv in LIVE_RUNGS:
            # Extreme-zone filter: skip tight rungs for high-blowout fixtures
            if blow_suppress and sv <= 2.5:
                continue
            # Home side
            home_wp_raw = float(np.mean(margins > -sv))
            home_wp = wp_transform(home_wp_raw) if wp_transform else home_wp_raw
            mkt_home, dec_odds_home = _get_market_implied(home, away, today_str, sv, "home")
            if mkt_home is not None and home_wp >= NEGBIN_GATE and mkt_home >= MKT_IMPLIED_FLOOR:
                edge = (home_wp - mkt_home) * 100.0
                lqs = _compute_lqs("Alt Spread", edge, line=sv)
                los = _compute_los(home_wp, edge_pp=edge, lqs=lqs,
                                   cushion=(home_wp >= 0.70))
                if los >= LOS_FLOOR:
                    # 2×2 cell classification
                    if home_is_fav is True:
                        cell = "fav_home"
                    elif home_is_fav is False:
                        cell = "dog_home"
                    else:
                        cell = "unknown"

                    # dist_wp: calibrated spread-cover probability (vs model_wp which is overconfident)
                    _dist_wp_home = ats_distributional_cover_prob(
                        mu_home - mu_away, sv, "home")

                    candidates.append({
                        "fixture_id": fix["fixture_id"],
                        "fixture_key": fixture_key,
                        "home_team": home, "away_team": away,
                        "game_date": today_str,
                        "spread_value": sv, "pick_side": "home",
                        "pick_team": home,
                        "model_wp": home_wp, "market_wp": mkt_home,
                        "ats_dist_wp": round(_dist_wp_home, 4),
                        "edge_pp": round(edge, 2),
                        "los": round(los, 4), "lqs": lqs,
                        "decimal_odds": dec_odds_home,
                        "is_pick_fav": home_is_fav,
                        "cell": cell,
                        "mu_home": mu_home, "mu_away": mu_away,
                        "commence_time": fix["commence_time"],
                    })

            # Away side
            away_wp_raw = float(np.mean(margins < sv))
            away_wp = wp_transform(away_wp_raw) if wp_transform else away_wp_raw
            mkt_away, dec_odds_away = _get_market_implied(home, away, today_str, sv, "away")
            if mkt_away is not None and away_wp >= NEGBIN_GATE and mkt_away >= MKT_IMPLIED_FLOOR:
                edge = (away_wp - mkt_away) * 100.0
                lqs = _compute_lqs("Alt Spread", edge, line=sv)
                los = _compute_los(away_wp, edge_pp=edge, lqs=lqs,
                                   cushion=(away_wp >= 0.70))
                if los >= LOS_FLOOR:
                    if home_is_fav is True:
                        cell = "dog_away"
                    elif home_is_fav is False:
                        cell = "fav_away"
                    else:
                        cell = "unknown"

                    _dist_wp_away = ats_distributional_cover_prob(
                        mu_home - mu_away, sv, "away")

                    candidates.append({
                        "fixture_id": fix["fixture_id"],
                        "fixture_key": fixture_key,
                        "home_team": home, "away_team": away,
                        "game_date": today_str,
                        "spread_value": sv, "pick_side": "away",
                        "pick_team": away,
                        "model_wp": away_wp, "market_wp": mkt_away,
                        "ats_dist_wp": round(_dist_wp_away, 4),
                        "edge_pp": round(edge, 2),
                        "los": round(los, 4), "lqs": lqs,
                        "decimal_odds": dec_odds_away,
                        "is_pick_fav": not home_is_fav if home_is_fav is not None else None,
                        "cell": cell,
                        "mu_home": mu_home, "mu_away": mu_away,
                        "commence_time": fix["commence_time"],
                    })

    print(f"[negbin-{tag_label}] {len(candidates)} candidates "
          f"(Home: {sum(1 for c in candidates if c['pick_side']=='home')}, "
          f"Away: {sum(1 for c in candidates if c['pick_side']=='away')})")

    if not candidates:
        return []

    # ── Cell breakdown ──
    cell_counts = defaultdict(int)
    for c in candidates:
        cell_counts[c["cell"]] += 1
    print(f"[negbin-{tag_label}] Cells: {dict(cell_counts)}")

    # ── Assemble parlays ──
    # Design: LEG-LEVEL capped reuse, matching live's construction rules.
    #
    # - 2-3 legs only: matches the backtest-validated population
    # - Same fixture allowed across parlays (live does this — limited games require it)
    # - Same fixture at DIFFERENT team:rung = different legs, different outcomes = OK
    # - Same leg (fixture:team:rung) capped at MAX_LEG_APP appearances (matches live)
    # - No duplicate parlays (same leg set)
    # - This mirrors live's _MAX_LEG_APP reuse rules, making the forward comparison
    #   apples-to-apples on construction method
    _MAX_LEG_APP = 2  # reduced from 3: halves per-fixture blowout exposure (12→8 max infected parlays)

    pool = [c for c in candidates
            if c.get("decimal_odds") is not None and c["decimal_odds"] > 1.0]

    # ── Fix 2: epp >= 12 toxicity filter (applied to ALL arms via pool) ──
    pre_epp = len(pool)
    pool = [c for c in pool
            if c.get("edge_pp") is None or c["edge_pp"] < _EDGE_GATE_THRESHOLD_PP]
    if pre_epp - len(pool) > 0:
        print(f"[negbin-{tag_label}] epp≥{_EDGE_GATE_THRESHOLD_PP} filter: "
              f"{pre_epp} → {len(pool)} ({pre_epp - len(pool)} toxic legs removed)")

    # ── Fix 3: min-rung per fixture-team (evaware logic on Shadow A) ──
    # Group by fixture_key:pick_team, keep only the lowest rung per group.
    # Lower rungs (+1.5/+2.5) have thicker edge at longer odds; max-dist_wp
    # funneled 84% of legs to +4.5 (thinnest edge, shortest odds).
    ft_groups = defaultdict(list)
    for c in pool:
        ft_key = f"{c['fixture_key']}:{c['pick_team']}"
        ft_groups[ft_key].append(c)
    pool = []
    for ft_key, group in ft_groups.items():
        group.sort(key=lambda x: x["spread_value"])
        pool.append(group[0])  # lowest rung

    pool.sort(key=lambda x: -x["los"])

    # Assign a leg_id to each candidate: fixture_key:pick_team:spread_value
    for c in pool:
        c["leg_id"] = f"{c['fixture_key']}:{c['pick_team']}:{c['spread_value']}"

    pool_cap = min(len(pool), 50)

    def _score_combo(combo):
        fixtures_used = set(c["fixture_key"] for c in combo)
        if len(fixtures_used) < len(combo):
            return None  # no same-game legs within a parlay
        combined_odds = 1.0
        for c in combo:
            combined_odds *= c["decimal_odds"]
        band = "other"
        for band_label, lo, hi in ODDS_BANDS:
            if lo <= combined_odds < hi:
                band = band_label
                break
        avg_wp = np.mean([c["ats_dist_wp"] for c in combo])
        avg_lqs = np.mean([c.get("lqs", 50) for c in combo])
        avg_los = np.mean([c.get("los", 0.5) for c in combo])
        combined_wp = 1.0
        for c in combo:
            combined_wp *= c["ats_dist_wp"]
        norm_ev = max(0, combined_wp * (combined_odds - 1) - (1 - combined_wp))
        composite = avg_wp * 0.40 + (avg_lqs / 100) * 0.20 + avg_los * 0.30 + min(norm_ev, 1.0) * 0.10
        cell_adj = 1.0
        for c in combo:
            cell_adj *= CELL_BOOST.get(c["cell"], 1.0)
        composite *= cell_adj
        return {
            "legs": list(combo),
            "combined_odds": combined_odds,
            "composite": composite,
            "combined_wp": combined_wp,
            "odds_band": band,
            "cell_adj": round(cell_adj, 4),
            "leg_count": len(combo),
        }

    # Generate combos for sizes 2-3 only (validated population)
    all_scored = []
    for size in [2, 3]:
        if len(pool) < size:
            continue
        combos = list(itertools.combinations(pool[:pool_cap], size))
        for combo in combos:
            result = _score_combo(combo)
            if result is not None:
                all_scored.append(result)

    # Bucket by band, sort each by composite
    band_buckets = defaultdict(list)
    for s in all_scored:
        band_buckets[s["odds_band"]].append(s)
    for band in band_buckets:
        band_buckets[band].sort(key=lambda x: -x["composite"])

    # Fill bands via round-robin with leg-level capped reuse (mirrors live's _MAX_LEG_APP).
    # Round-robin ensures high-odds bands get slots before popular legs hit their cap.
    parlays = []
    seen_bet_infos = set()
    leg_use = defaultdict(int)  # leg_id -> appearance count across ALL parlays
    band_filled = defaultdict(int)  # band -> count selected
    band_iters = {}  # band -> iterator over bucket
    for band_label, _, _ in ODDS_BANDS:
        band_iters[band_label] = iter(band_buckets.get(band_label, []))

    any_progress = True
    while any_progress and len(parlays) < MAX_SHADOW_PARLAYS:
        any_progress = False
        for band_label, _, _ in ODDS_BANDS:
            if band_filled[band_label] >= PER_BAND_TARGET:
                continue
            if len(parlays) >= MAX_SHADOW_PARLAYS:
                break
            # Try to find one valid parlay for this band
            for s in band_iters[band_label]:
                bet_key = tuple(sorted(l["leg_id"] for l in s["legs"]))
                if bet_key in seen_bet_infos:
                    continue
                if any(leg_use[l["leg_id"]] >= _MAX_LEG_APP for l in s["legs"]):
                    continue
                seen_bet_infos.add(bet_key)
                for l in s["legs"]:
                    leg_use[l["leg_id"]] += 1
                parlays.append(s)
                band_filled[band_label] += 1
                any_progress = True
                break  # one per band per round

    print(f"[negbin-{tag_label}] {len(parlays)} {tag_label} parlays assembled "
          f"(from {len(all_scored)} scored combos, pool={pool_cap})")

    # ── Band distribution + underfill diagnostic ──
    band_counts = defaultdict(int)
    for p in parlays:
        band_counts[p["odds_band"]] += 1
    avail_counts = {b: len(v) for b, v in band_buckets.items() if v}
    print(f"[negbin-{tag_label}] Bands selected: {dict(band_counts)}")
    print(f"[negbin-{tag_label}] Bands available: {avail_counts}")

    # Underfill diagnostic: distinguish leg-consumption vs no-constructable-parlays
    for band_label, _, _ in ODDS_BANDS:
        filled = band_filled[band_label]
        if filled >= PER_BAND_TARGET:
            continue
        avail = len(band_buckets.get(band_label, []))
        if avail == 0:
            reason = "NO_CONSTRUCTABLE — no 2-3 leg combo reaches this odds range"
        elif filled == 0 and avail > 0:
            reason = f"LEG_CONSUMPTION — {avail} combos exist but all legs at cap ({_MAX_LEG_APP}x)"
        else:
            # Partially filled: some combos worked, rest hit cap
            reason = f"LEG_CONSUMPTION — {avail} combos exist, {filled}/{PER_BAND_TARGET} filled before legs capped"
        print(f"[negbin-{tag_label}]   {band_label} underfill: {reason}")

    # Leg reuse stats
    leg_dist = defaultdict(int)
    for p in parlays:
        leg_dist[p["leg_count"]] += 1
    reuse_hist = defaultdict(int)  # appearance_count -> number_of_legs
    for cnt in leg_use.values():
        reuse_hist[cnt] += 1
    at_cap = sum(1 for cnt in leg_use.values() if cnt >= _MAX_LEG_APP)
    print(f"[negbin-{tag_label}] Leg counts: {dict(leg_dist)} | "
          f"reuse histogram: {dict(sorted(reuse_hist.items()))} | "
          f"at cap ({_MAX_LEG_APP}x): {at_cap}")

    # ── Log candidate-level summary ──
    if candidates:
        home_cands = [c for c in candidates if c["pick_side"] == "home"]
        away_cands = [c for c in candidates if c["pick_side"] == "away"]
        h_edge = np.mean([c["edge_pp"] for c in home_cands]) if home_cands else 0
        a_edge = np.mean([c["edge_pp"] for c in away_cands]) if away_cands else 0
        print(f"[negbin-{tag_label}] Home edge: {h_edge:+.1f}pp, Away edge: {a_edge:+.1f}pp")

    # ── Live-realistic subset ──
    # Re-select from scored combos using live's structural constraints:
    #   - Strict no leg reuse (live uses seen_leg_keys — each leg in at most 1 parlay)
    #   - Pure composite rank (no band round-robin)
    #   - MAX_BETS_PER_GAME = 3 per fixture across all parlays
    #   - LQS >= 55 per leg
    live_real_keys = _select_live_realistic(all_scored)
    for p in parlays:
        bet_key = tuple(sorted(l["leg_id"] for l in p["legs"]))
        p["live_realistic"] = 1 if bet_key in live_real_keys else 0

    lr_count = sum(1 for p in parlays if p["live_realistic"])
    print(f"[negbin-{tag_label}] live-realistic subset: {lr_count} of {len(parlays)} parlays")

    # ── Fav@Home carved-out allocation ──
    # Separate assembly from fav_home candidates ONLY.
    # Tracks whether fav_home is profitable on its own when not competed out.
    fh_parlays = _assemble_fav_home_carveout(candidates, ODDS_BANDS, CELL_BOOST)

    # ── Store as MockBet/MockBetLeg ──
    if db is not None and parlays:
        _store_shadow_parlays(db, parlays, today_str, source_tag=source_tag)
    if db is not None and fh_parlays:
        fh_tag = f"{source_tag}_fh"
        _store_shadow_parlays(db, fh_parlays, today_str, source_tag=fh_tag)

    return parlays


def generate_negbin_recal_shadow_picks(db=None, today_str=None, fixtures=None):
    """Shadow B: same NegBin model with isotonic WP recalibration.

    Parallel to Shadow A (uncalibrated). Identical model, features, gate logic,
    construction — ONLY the WP calibration differs. Tagged as
    source='shadow_negbin_recal' for clean A/B comparison on the same slates.

    Args:
        fixtures: Pre-fetched fixture list from Shadow A's run. Guarantees
            both shadows see the exact same slate (no timing divergence).
    """
    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    # Snapshot fixtures once so A and B see identical slates
    if fixtures is None:
        fixtures = _get_todays_fixtures(today_str)

    cal = _fit_wp_calibration()
    if cal is None:
        print("[negbin-recal] No calibration available — Shadow B skipped")
        return []
    return generate_negbin_shadow_picks(
        db=db, today_str=today_str,
        wp_transform=cal, source_tag="shadow_negbin_recal",
        fixtures=fixtures,
    )


def generate_atscap_counterfactual(db=None, today_str=None):
    """ATS Continuous Gap-Weighted Sizing — counterfactual shadow on Shadow A.

    Keeps ALL of Shadow A's parlays but sizes each one down proportionally
    to the ATS disagreement on its worst +1.5 leg. Uses the smooth monotonic
    gap->bust gradient (Spearman rho=-0.375, p<0.0001) — no hand-picked
    threshold, degrades gracefully.

    Design:
      - weight = min(min_ats_leg / 0.60, 1.0) per parlay
        * All +1.5 legs ATS >= 60% → weight = 1.0 (no change)
        * Worst +1.5 leg ATS = 45% → weight = 0.75 (size down 25%)
        * Worst +1.5 leg ATS = 30% → weight = 0.50 (size down 50%)
        * No +1.5 ATS legs → weight = 1.0 (unchanged)
      - Stores amount = $10 * weight (notional sized stake)
      - Stores weight in mock_bets.weight column
      - Same parlays as A, same settlement — only sizing differs
      - PnL computed using sized amount, not equal $10

    Separate from recalibration A/B (WP-selection edge).
    This is the fragility SIZING edge — independent construction lever.

    Two-gate graduation:
      Gate 1 (gradient, fixture-level): ATS gap→bust monotonicity holds
        forward. Spearman rho < -0.20 at p<0.05 on OOS fixtures.
        PASSED: rho=-0.634 (p=0.005) on N=18 OOS fixtures (7/6-7/8).
        Fixture-level EV positive: +$0.836 net (busts saved $0.895,
        wins forgone $0.059). Bootstrap CI excludes zero (p=0.02).
      Gate 2 (P&L asymmetry, parlay-level): netted drawdown cut vs
        upside cost > 1.5x on forward days. Needs more variance-time.
        Currently 60.9x on 10 backfilled days (accumulating forward).

    Deploy path: counterfactual shadow now → graduate to live sizing
    once Gate 2 confirms on genuinely-live slates. Start conservative
    (current curve), confirm live P&L tracks expected asymmetry.
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    _NOTIONAL_STAKE = 10.0

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[atscap] Generating continuous gap-sizing counterfactual for {today_str}")

    # ── Check for existing atscap parlays today (dedup) ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source='shadow_negbin_atscap' AND game_date=:gd"
    ), {"gd": today_str}).scalar()
    if existing > 0:
        print(f"[atscap] Already {existing} atscap parlays for {today_str} — skipping")
        return []

    # ── Read Shadow A's parlays for today ──
    a_parlays = db.execute(_text(
        "SELECT id, bet_info, odds, composite_score, negbin_odds_band, "
        "       negbin_leg_count, predicted_win_prob "
        "FROM mock_bets "
        "WHERE source='shadow_negbin' AND game_date=:gd "
        "  AND cohort='clean' "
        "ORDER BY composite_score DESC"
    ), {"gd": today_str}).fetchall()

    if not a_parlays:
        print("[atscap] No Shadow A parlays found for today")
        return []

    # ── Read all legs with ATS scores ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.description, l.win_prob, l.ats_cover_wp, "
        "       l.fixture_id, l.leg_index "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    parlay_legs = defaultdict(list)
    for mid, desc, wp, ats, fix_id, idx in leg_data:
        parlay_legs[mid].append({
            "description": desc, "win_prob": wp, "ats_cover_wp": ats,
            "fixture_id": fix_id, "leg_index": idx,
        })

    # ── Compute per-parlay fragility weight ──
    # weight = min(min_ats_among_+1.5_legs / 0.60, 1.0)
    parlay_weights = {}
    for pid, _, _, _, _, _, _ in a_parlays:
        legs = parlay_legs.get(pid, [])
        ats_scores = [
            leg["ats_cover_wp"] for leg in legs
            if "+1.5" in leg["description"] and leg["ats_cover_wp"] is not None
        ]
        if ats_scores:
            min_ats = min(ats_scores)
            weight = min(min_ats / 0.60, 1.0)
        else:
            weight = 1.0
        parlay_weights[pid] = round(weight, 4)

    # Log sizing summary
    sized_down = {pid: w for pid, w in parlay_weights.items() if w < 1.0}
    n_total = len(a_parlays)
    n_sized = len(sized_down)
    if sized_down:
        avg_weight = sum(sized_down.values()) / len(sized_down)
        total_staked = sum(parlay_weights[pid] * _NOTIONAL_STAKE
                          for pid, _, _, _, _, _, _ in a_parlays)
        print(f"[atscap] {n_total} parlays: {n_sized} sized down "
              f"(avg weight={avg_weight:.2f}), "
              f"${total_staked:.2f} / ${n_total * _NOTIONAL_STAKE:.2f} staked")

        # Show the fragility legs driving sizing
        seen_legs = {}
        for pid in sized_down:
            for leg in parlay_legs.get(pid, []):
                desc = leg["description"]
                ats = leg["ats_cover_wp"]
                if "+1.5" in desc and ats is not None and ats < 0.60:
                    if desc not in seen_legs:
                        seen_legs[desc] = ats
        for desc, ats in sorted(seen_legs.items(), key=lambda x: x[1]):
            w = min(ats / 0.60, 1.0)
            print(f"  {desc}: ATS={ats*100:.1f}% → weight={w:.2f}")
    else:
        print(f"[atscap] {n_total} parlays: no fragility legs — all weight=1.0")

    # ── Store as shadow_negbin_atscap with sized amounts ──
    run_id = str(uuid.uuid4())
    stored = 0
    for row in a_parlays:
        pid, bet_info, odds, comp, band, lc, wp = row
        weight = parlay_weights[pid]
        sized_amount = round(_NOTIONAL_STAKE * weight, 2)
        mock_id = str(uuid.uuid4())

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if (lc or 2) > 1 else "straight",
            odds=odds,
            amount=sized_amount,
            legs=lc or 2,
            bet_info=bet_info,
            predicted_win_prob=wp,
            model_used="negbin_v1",
            status="PENDING",
            source="shadow_negbin_atscap",
            composite_score=comp,
            cohort="clean",
            weight=weight,
        )
        db.add(mock)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_leg_count=:lc "
                "WHERE id=:mid"
            ), {"band": band, "lc": lc, "mid": mock_id})
        except Exception:
            pass

        # Copy legs from A's parlay
        legs = parlay_legs.get(pid, [])
        for leg in legs:
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=leg["leg_index"],
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["win_prob"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()
            if leg["ats_cover_wp"] is not None:
                try:
                    db.execute(_text(
                        "UPDATE mock_bet_legs SET ats_cover_wp=:ats WHERE id=:lid"
                    ), {"ats": leg["ats_cover_wp"], "lid": leg_obj.id})
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[atscap] DB commit failed: {e}")
        return []

    print(f"[atscap] Stored {stored} atscap parlays (source=shadow_negbin_atscap)")

    # ── Graduation-bar tracking ──
    _report_atscap_asymmetry(db)

    return a_parlays


# ═══════════════════════════════════════════════════════════════
# EDGE GATE COUNTERFACTUAL SHADOW
# ═══════════════════════════════════════════════════════════════

_EDGE_GATE_THRESHOLD_PP = 12  # max negbin_edge_pp per parlay

def generate_edgegate_counterfactual(db=None, today_str=None):
    """Edge-gate counterfactual: reject parlays where max(negbin_edge_pp) >= threshold.

    When NegBin claims a large edge over the market (edge_pp >= 12),
    the market is typically right — those legs cover at 60-67% despite
    NegBin predicting 85%+.  Legs with modest edge (3-9pp) cover at
    88-96% (genuine small edge).

    This gate rejects toxic-overconfidence parlays while keeping the
    sweet spot.  100% coverage (all legs have edge_pp, unlike ATS
    which only scores +1.5).

    Deflated SR status (2026-07-11, T=12 days):
      Raw best: 15.6% ROI at 12pp gate
      Search-free avg: 8.1% (across 11 thresholds tested)
      Deflated SR: -0.36 (DIRECTIONAL — not yet confirmed)
      Needs ~50+ forward days for DSR confirmation

    This is a SEPARATE counterfactual arm — does not affect live
    selection, atscap sizing, or recalibration A/B.
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    _SOURCE = "shadow_negbin_edgegate"
    _NOTIONAL_STAKE = 10.0

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[edgegate] Generating edge-gate counterfactual for {today_str} "
          f"(threshold={_EDGE_GATE_THRESHOLD_PP}pp)")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[edgegate] Already {existing} edgegate parlays for {today_str} — skipping")
        return []

    # ── Read Shadow A's parlays for today ──
    a_parlays = db.execute(_text(
        "SELECT id, bet_info, odds, composite_score, negbin_odds_band, "
        "       negbin_leg_count, predicted_win_prob "
        "FROM mock_bets "
        "WHERE source='shadow_negbin' AND game_date=:gd "
        "  AND cohort='clean' "
        "ORDER BY composite_score DESC"
    ), {"gd": today_str}).fetchall()

    if not a_parlays:
        print("[edgegate] No Shadow A parlays found for today")
        return []

    # ── Read all legs with edge_pp ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.fixture_id, l.leg_index, l.ats_cover_wp, l.negbin_wp, "
        "       l.negbin_mu_home, l.negbin_mu_away, l.negbin_cell "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    parlay_legs = defaultdict(list)
    for mid, desc, wp, edge, fix_id, idx, ats, nb_wp, mu_h, mu_a, cell in leg_data:
        parlay_legs[mid].append({
            "description": desc, "win_prob": wp, "negbin_edge_pp": edge,
            "fixture_id": fix_id, "leg_index": idx, "ats_cover_wp": ats,
            "negbin_wp": nb_wp, "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
            "negbin_cell": cell,
        })

    # ── Gate: reject parlays where max(edge_pp) >= threshold ──
    kept = []
    rejected = []
    rejected_edges = []  # for logging
    for row in a_parlays:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        edges = [leg["negbin_edge_pp"] for leg in legs
                 if leg["negbin_edge_pp"] is not None]
        max_edge = max(edges) if edges else 0
        if max_edge >= _EDGE_GATE_THRESHOLD_PP:
            rejected.append(row)
            rejected_edges.append(max_edge)
        else:
            kept.append(row)

    n_total = len(a_parlays)
    n_kept = len(kept)
    n_rejected = len(rejected)
    print(f"[edgegate] {n_total} A parlays: {n_kept} kept, "
          f"{n_rejected} rejected (max edge >= {_EDGE_GATE_THRESHOLD_PP}pp)")
    if rejected_edges:
        print(f"  Rejected edge range: {min(rejected_edges):.1f}pp - "
              f"{max(rejected_edges):.1f}pp (avg {sum(rejected_edges)/len(rejected_edges):.1f}pp)")

    # ── Store kept parlays as shadow_negbin_edgegate ──
    run_id = str(uuid.uuid4())
    stored = 0
    for row in kept:
        pid, bet_info, odds, comp, band, lc, wp = row
        mock_id = str(uuid.uuid4())

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if (lc or 2) > 1 else "straight",
            odds=odds,
            amount=_NOTIONAL_STAKE,
            legs=lc or 2,
            bet_info=bet_info,
            predicted_win_prob=wp,
            model_used="negbin_v1",
            status="PENDING",
            source=_SOURCE,
            composite_score=comp,
            cohort="clean",
            weight=1.0,
        )
        db.add(mock)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_leg_count=:lc "
                "WHERE id=:mid"
            ), {"band": band, "lc": lc, "mid": mock_id})
        except Exception:
            pass

        # Copy legs from A
        legs = parlay_legs.get(pid, [])
        for leg in legs:
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=leg["leg_index"],
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["win_prob"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()
            # Copy NegBin + ATS columns
            updates = {}
            if leg["negbin_wp"] is not None:
                updates["negbin_wp"] = leg["negbin_wp"]
            if leg["negbin_edge_pp"] is not None:
                updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
            if leg["negbin_mu_home"] is not None:
                updates["negbin_mu_home"] = leg["negbin_mu_home"]
            if leg["negbin_mu_away"] is not None:
                updates["negbin_mu_away"] = leg["negbin_mu_away"]
            if leg["negbin_cell"] is not None:
                updates["negbin_cell"] = leg["negbin_cell"]
            if leg["ats_cover_wp"] is not None:
                updates["ats_cover_wp"] = leg["ats_cover_wp"]
            if updates:
                set_clause = ", ".join(f"{k}=:{k}" for k in updates)
                updates["lid"] = leg_obj.id
                try:
                    db.execute(_text(
                        f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                    ), updates)
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[edgegate] DB commit failed: {e}")
        return []

    print(f"[edgegate] Stored {stored} edgegate parlays (source={_SOURCE})")

    # ── Graduation tracker — locked validation standard ──
    _report_edgegate_tracker(db)

    return kept


def _report_edgegate_tracker(db):
    """Edge-gate tracker: reports raw + search-free + deflated SR + bad-period.

    Locked validation standard (2026-07-11):
      - Every report shows deflated SR, not just raw
      - Trial count N=11 (thresholds tested in discovery)
      - Graduation requires deflated SR > 0 at p < 0.05
    """
    from sqlalchemy import text as _text

    # ── Get settled data for edge-gated shadow AND Shadow A ──
    gate_rows = db.execute(_text(
        "SELECT game_date, status, odds, amount "
        "FROM mock_bets WHERE source='shadow_negbin_edgegate' AND cohort='clean' "
        "  AND status IN ('SETTLED_WIN','SETTLED_LOSS')"
    )).fetchall()
    a_rows = db.execute(_text(
        "SELECT game_date, status, odds "
        "FROM mock_bets WHERE source='shadow_negbin' AND cohort='clean' "
        "  AND status IN ('SETTLED_WIN','SETTLED_LOSS')"
    )).fetchall()

    if not gate_rows:
        print("[edgegate] No settled data yet — tracker starts after first settlement")
        return

    # Per-day PnL
    a_day = defaultdict(float)
    gate_day = defaultdict(float)
    gate_n_day = defaultdict(int)
    a_n_day = defaultdict(int)
    for gd, st, odds in a_rows:
        a_day[gd] += (odds - 1) * 10.0 if st == 'SETTLED_WIN' else -10.0
        a_n_day[gd] += 1
    for gd, st, odds, amt in gate_rows:
        gate_day[gd] += (odds - 1) * amt if st == 'SETTLED_WIN' else -amt
        gate_n_day[gd] += 1

    overlap = sorted(set(a_day) & set(gate_day))
    if len(overlap) < 2:
        print(f"[edgegate] Only {len(overlap)} settled days — need ≥2 for tracking")
        return

    # ── Per-day report ──
    lifts = []
    print(f"\n[edgegate] Forward tracker ({len(overlap)} settled days, "
          f"gate={_EDGE_GATE_THRESHOLD_PP}pp):")
    print(f"  {'Day':<12} {'A_N':>4} {'Gate_N':>6} {'A_PnL':>9} {'Gate_PnL':>9} "
          f"{'Lift':>7} {'Type':>5}")
    for d in overlap:
        lift = gate_day[d] - a_day[d]
        lifts.append(lift)
        dtype = "BAD" if a_day[d] < -50 else ("GOOD" if a_day[d] > 50 else "FLAT")
        print(f"  {d:<12} {a_n_day[d]:4d} {gate_n_day[d]:6d} "
              f"{a_day[d]:+9.2f} {gate_day[d]:+9.2f} {lift:+7.2f} {dtype:>5}")

    lifts = np.array(lifts)
    T = len(lifts)

    # ── Summary stats ──
    total_a = sum(a_day[d] for d in overlap)
    total_gate = sum(gate_day[d] for d in overlap)
    total_lift = total_gate - total_a
    n_gate_total = sum(gate_n_day[d] for d in overlap)
    n_a_total = sum(a_n_day[d] for d in overlap)

    gate_roi = total_gate / (n_gate_total * 10) * 100 if n_gate_total > 0 else 0
    a_roi = total_a / (n_a_total * 10) * 100 if n_a_total > 0 else 0

    print(f"\n  Shadow A:    {n_a_total} parlays, P&L=${total_a:+.2f}, ROI={a_roi:+.1f}%")
    print(f"  Edge-gated:  {n_gate_total} parlays, P&L=${total_gate:+.2f}, ROI={gate_roi:+.1f}%")
    print(f"  Total lift:  ${total_lift:+.2f}")

    # ── Bad-period survival ──
    bad_days = [d for d in overlap if a_day[d] < -50]
    if bad_days:
        bad_a = sum(a_day[d] for d in bad_days)
        bad_gate = sum(gate_day[d] for d in bad_days)
        print(f"  Bad-period ({len(bad_days)} days): A=${bad_a:+.0f}, "
              f"Gate=${bad_gate:+.0f}, saved=${bad_gate - bad_a:+.0f}")

    # ── Deflated Sharpe (locked standard) ──
    if T >= 3 and lifts.std() > 0:
        daily_sr = lifts.mean() / lifts.std()
        N_TRIALS = 11  # thresholds tested in discovery (locked)

        # E[max SR under null] for N trials at T days
        E_max_sr = np.sqrt(2 * np.log(N_TRIALS)) / np.sqrt(T)
        deflated_sr = daily_sr - E_max_sr

        # p-value
        try:
            from scipy.stats import norm, ttest_1samp
            se_sr = 1.0 / np.sqrt(T)
            dsr_z = deflated_sr / se_sr
            dsr_p = 1 - norm.cdf(dsr_z)

            # Paired t-test (not search-adjusted)
            t_stat, t_p = ttest_1samp(lifts, 0)
            t_p_one = t_p / 2  # one-sided
        except ImportError:
            dsr_p = None
            t_p_one = None

        verdict = ("CONFIRMED" if deflated_sr > 0 and (dsr_p or 1) < 0.05
                   else "DIRECTIONAL" if daily_sr > 0
                   else "COLLAPSED")

        print(f"\n  ── Locked Validation Standard ──")
        print(f"  Raw daily SR:       {daily_sr:.4f}")
        print(f"  E[max SR null]:     {E_max_sr:.4f} (N={N_TRIALS} trials, T={T} days)")
        print(f"  Deflated SR:        {deflated_sr:+.4f}")
        if dsr_p is not None:
            print(f"  DSR p-value:        {dsr_p:.4f}")
        if t_p_one is not None:
            print(f"  Paired t (raw):     t={t_stat:.3f}, p={t_p_one:.4f}")
        print(f"  Verdict:            {verdict}")

        # Min track record
        if daily_sr > 0 and daily_sr > E_max_sr:
            print(f"  → Signal ABOVE null maximum — accumulating toward significance")
        elif daily_sr > 0:
            needed_T = (np.sqrt(2 * np.log(N_TRIALS)) / daily_sr) ** 2
            print(f"  → Need T > {needed_T:.0f} days for DSR confirmation "
                  f"(have {T}, need {max(0, needed_T - T):.0f} more)")
    else:
        print(f"\n  ── Validation: insufficient data (T={T}) ──")

    # ── Edge-bucket breakdown ──
    edge_rows = db.execute(_text(
        "SELECT l.negbin_edge_pp, l.leg_result "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin_edgegate' AND b.cohort='clean' "
        "  AND b.status IN ('SETTLED_WIN','SETTLED_LOSS') "
        "  AND l.negbin_edge_pp IS NOT NULL AND l.leg_result IN ('WIN','LOSS')"
    )).fetchall()

    if edge_rows:
        print(f"\n  ── Edge-bucket cover rates (gated legs) ──")
        buckets = [(0, 3, '0-3pp'), (3, 6, '3-6pp'), (6, 9, '6-9pp'), (9, 12, '9-12pp')]
        print(f"  {'Bucket':<10} {'N':>5} {'Cover%':>8}")
        for lo, hi, label in buckets:
            subset = [(e, r) for e, r in edge_rows if lo <= e < hi]
            if subset:
                cover = sum(1 for _, r in subset if r == 'WIN') / len(subset)
                print(f"  {label:<10} {len(subset):5d} {cover*100:7.1f}%")


# ═══════════════════════════════════════════════════════════════
# FULL-STACK COUNTERFACTUAL: NegBin select → edge gate → dist_wp Kelly sizing
# ═══════════════════════════════════════════════════════════════

_FULLSTACK_SOURCE = "shadow_negbin_fullstack"
_KELLY_FRACTION = 0.25       # quarter-Kelly
_KELLY_BASE_STAKE = 10.0     # notional bankroll unit
_KELLY_MAX_WEIGHT = 2.5      # cap at 2.5x base (conservative)
_KELLY_MIN_WEIGHT = 0.10     # floor (no zero-weight parlays)
_CORR_FIXTURE_CAP = 0.70     # daily fixture concentration cap for correlation adj
_NOTIONAL_BANKROLL = 1000.0  # notional bankroll for sizing ($10 base = 1% per unit)
_DAILY_EXPOSURE_CAP = 0.15   # max 15% of notional bankroll at risk per day
                              # Rationale: effective-N per day ≈ 22 (8.6x correlated),
                              # worst day -37% → -8.9% at zero Sharpe cost (0.428 constant).
                              # Correlated-slate risk = the drawdown driver, not concentration.


def generate_fullstack_counterfactual(db=None, today_str=None):
    """Full-stack counterfactual: NegBin select → edge<12pp gate → dist_wp quarter-Kelly.

    Architecture:
      1. NegBin selects fixtures/rungs (Shadow A parlays)
      2. Edge gate rejects toxic overconfidence (edge_pp >= 12)
      3. dist_wp provides discrimination (replaces NegBin WP)
      4. Quarter-Kelly sizes each parlay using dist_wp as the cover probability
      5. Correlation-aware: daily fixture concentration dampening

    This is the COMBINED strategy deployed as a counterfactual shadow.
    Individual layers are OOS-confirmed; this tests the STACK as a unit.

    Kelly formula:
      edge = dist_wp_parlay - (1 / odds)
      kelly_frac = edge / (odds - 1)
      weight = quarter_kelly = max(min(kelly_frac * 0.25, MAX), MIN)
      amount = base_stake * weight

    dist_wp_parlay = product of leg dist_wp values (independent legs assumption,
    valid because builder enforces fixture uniqueness within each parlay).

    Correlation adjustment: if a fixture appears in >CAP fraction of today's
    kept parlays, dampen those parlays' weights by the excess concentration.
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text
    import re

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[fullstack] Generating full-stack counterfactual for {today_str} "
          f"(edge<{_EDGE_GATE_THRESHOLD_PP}pp, dist_wp quarter-Kelly)")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _FULLSTACK_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[fullstack] Already {existing} fullstack parlays for {today_str} — skipping")
        return []

    # ── Read Shadow A's parlays for today ──
    a_parlays = db.execute(_text(
        "SELECT id, bet_info, odds, composite_score, negbin_odds_band, "
        "       negbin_leg_count, predicted_win_prob "
        "FROM mock_bets "
        "WHERE source='shadow_negbin' AND game_date=:gd "
        "  AND cohort='clean' "
        "ORDER BY composite_score DESC"
    ), {"gd": today_str}).fetchall()

    if not a_parlays:
        print("[fullstack] No Shadow A parlays found for today")
        return []

    # ── Read all legs with edge_pp and dist_wp ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.fixture_id, l.leg_index, l.ats_cover_wp, l.negbin_wp, "
        "       l.negbin_mu_home, l.negbin_mu_away, l.negbin_cell, l.ats_dist_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    parlay_legs = defaultdict(list)
    for (mid, desc, wp, edge, fix_id, idx, ats, nb_wp,
         mu_h, mu_a, cell, dist_wp) in leg_data:
        parlay_legs[mid].append({
            "description": desc, "win_prob": wp, "negbin_edge_pp": edge,
            "fixture_id": fix_id, "leg_index": idx, "ats_cover_wp": ats,
            "negbin_wp": nb_wp, "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
            "negbin_cell": cell, "ats_dist_wp": dist_wp,
        })

    # ── Layer 1: Edge gate — reject parlays with max(edge_pp) >= 12 ──
    after_gate = []
    n_rejected_edge = 0
    for row in a_parlays:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        edges = [leg["negbin_edge_pp"] for leg in legs
                 if leg["negbin_edge_pp"] is not None]
        max_edge = max(edges) if edges else 0
        if max_edge >= _EDGE_GATE_THRESHOLD_PP:
            n_rejected_edge += 1
        else:
            after_gate.append(row)

    print(f"[fullstack] Edge gate: {len(a_parlays)} → {len(after_gate)} "
          f"({n_rejected_edge} rejected, threshold={_EDGE_GATE_THRESHOLD_PP}pp)")

    if not after_gate:
        print("[fullstack] All parlays rejected by edge gate")
        return []

    # ── Layer 2: Compute dist_wp parlay-level cover probability ──
    # Score dist_wp on-the-fly if not already stored
    rung_re = re.compile(r'[+-](\d+\.5)')
    mu_cache = {}  # (home, away, date) → mu

    try:
        from ml_model import predict_game_ats_distributional, ats_distributional_cover_prob
        _has_dist_model = True
    except ImportError:
        _has_dist_model = False

    n_scored_live = 0
    n_fallback = 0

    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        for leg in legs:
            if leg["ats_dist_wp"] is not None:
                continue
            if not _has_dist_model:
                n_fallback += 1
                continue
            # Parse rung and teams from description
            desc = leg["description"] or ""
            m = rung_re.search(desc)
            if not m:
                n_fallback += 1
                continue
            sv = float(m.group(1))
            # Determine home/away team and pick_side from leg data
            ht = leg.get("negbin_mu_home")
            at = leg.get("negbin_mu_away")
            fix_id = leg["fixture_id"]
            # We need team names — look them up from fixtures table
            try:
                fix_row = db.execute(_text(
                    "SELECT home_team, away_team "
                    "FROM fixtures WHERE id=:fid LIMIT 1"
                ), {"fid": fix_id}).fetchone()
                if fix_row:
                    home_team, away_team = fix_row[0], fix_row[1]
                else:
                    n_fallback += 1
                    continue

                key = (home_team, away_team, today_str)
                if key not in mu_cache:
                    mu_cache[key] = predict_game_ats_distributional(
                        home_team, away_team, today_str)
                mu = mu_cache[key]
                if mu is None:
                    n_fallback += 1
                    continue

                # Determine pick_side
                pick_side = 'away'
                if home_team and desc.lower().startswith(home_team.lower()[:6]):
                    pick_side = 'home'

                dist_wp = ats_distributional_cover_prob(mu, sv, pick_side)
                leg["ats_dist_wp"] = round(dist_wp, 4)
                n_scored_live += 1

                # Also store it back to the source leg
                try:
                    src_legs = db.execute(_text(
                        "SELECT l.id FROM mock_bet_legs l "
                        "JOIN mock_bets b ON l.mock_bet_id = b.id "
                        "WHERE b.source='shadow_negbin' AND b.game_date=:gd "
                        "  AND l.fixture_id=:fid AND l.description=:desc "
                        "  AND l.ats_dist_wp IS NULL"
                    ), {"gd": today_str, "fid": fix_id, "desc": desc}).fetchall()
                    for (lid,) in src_legs:
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET ats_dist_wp=:dwp WHERE id=:lid"
                        ), {"dwp": round(dist_wp, 4), "lid": lid})
                except Exception:
                    pass

            except Exception as exc:
                n_fallback += 1
                continue

    if n_scored_live > 0:
        print(f"[fullstack] Scored {n_scored_live} legs with dist_wp on-the-fly")
        try:
            db.commit()
        except Exception:
            pass

    parlay_dist_wp = {}
    n_missing_dist = 0

    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        leg_probs = []
        for leg in legs:
            dist = leg["ats_dist_wp"]
            if dist is not None and dist > 0:
                leg_probs.append(dist)
            else:
                # Fallback: use market-implied (conservative), NOT win_prob (overconfident)
                nb_wp = leg.get("negbin_wp")
                epp = leg.get("negbin_edge_pp")
                if nb_wp is not None and epp is not None:
                    mkt_fb = nb_wp - epp / 100.0
                    leg_probs.append(max(mkt_fb, 0.30))
                else:
                    leg_probs.append(0.65)  # conservative default
                n_missing_dist += 1

        # Parlay cover prob = product (legs are independent: different fixtures)
        parlay_prob = 1.0
        for p in leg_probs:
            parlay_prob *= p
        parlay_dist_wp[pid] = parlay_prob

    if n_missing_dist > 0:
        print(f"[fullstack] {n_missing_dist} legs still missing dist_wp — used market-implied fallback")
    if n_fallback > 0 and n_scored_live == 0:
        print(f"[fullstack] {n_fallback} legs could not be scored with dist_wp")

    # ── Layer 3: dist_wp discrimination-weighted sizing (quarter-Kelly) ──
    #
    # dist_wp is honest and CONSERVATIVE: actual WR exceeds dist_wp by
    # ~4pp per leg. At the parlay level, product(dist_wp) significantly
    # undershoots actual WR (65.3% parlay WR vs ~50% dist_wp product).
    #
    # The edge comes from NegBin SELECTION, not from dist_wp > market.
    # dist_wp's role is DISCRIMINATION: high dist_wp parlays cover much
    # more than low dist_wp ones (Q1=69% → Q5=97% within gated universe).
    #
    # Sizing strategy: Kelly on the EMPIRICAL edge, modulated by dist_wp
    # rank. The edge-gated universe has +9.3pp edge over breakeven on
    # settled data (WR=65.3%, BE=56.0%). dist_wp quintile determines
    # how much of that edge each parlay captures:
    #   Q5 (high dist_wp): full weight — these cover at ~97%
    #   Q1 (low dist_wp):  minimum weight — these cover at ~54%
    #
    # Kelly applied to empirical parlay-level edge:
    #   p_parlay = dist_wp_rank_mapped (Q1→0.55, Q5→0.90, linear interp)
    #   f* = (p * odds - 1) / (odds - 1), quarter-Kelly
    #
    # The rank mapping converts dist_wp discrimination into calibrated
    # parlay-level WRs based on settled quintile data.

    # Rank all parlays by their dist_wp and assign quintile-calibrated WRs
    dist_pairs = [(row[0], parlay_dist_wp[row[0]]) for row in after_gate]
    dist_pairs.sort(key=lambda x: x[1])
    n_parlays = len(dist_pairs)

    # Quintile-calibrated WRs from settled data (edge-gated universe):
    # Q1=69.3%, Q2=80.0%, Q3=84.0%, Q4=89.3%, Q5=97.4%
    # Conservative: discount by 20% toward breakeven (forward uncertainty)
    _Q_WRS = [0.60, 0.70, 0.75, 0.80, 0.90]  # conservative mappings

    parlay_weights = {}
    for rank_idx, (pid, dp) in enumerate(dist_pairs):
        # Map rank to quintile WR via linear interpolation
        rank_frac = rank_idx / max(n_parlays - 1, 1)  # 0.0 to 1.0
        # Interpolate across 5 quintile WRs
        q_idx = rank_frac * 4  # 0 to 4
        q_lo = int(q_idx)
        q_hi = min(q_lo + 1, 4)
        frac = q_idx - q_lo
        p_mapped = _Q_WRS[q_lo] * (1 - frac) + _Q_WRS[q_hi] * frac

        odds = dict((r[0], r[2] or 2.0) for r in after_gate)[pid]

        # Kelly: f* = (p * odds - 1) / (odds - 1)
        edge = p_mapped * odds - 1.0
        if edge <= 0 or odds <= 1.0:
            kelly_frac = 0.0
        else:
            kelly_frac = edge / (odds - 1.0)

        # Quarter-Kelly with floor and cap
        quarter_kelly = kelly_frac * _KELLY_FRACTION
        weight = max(min(quarter_kelly, _KELLY_MAX_WEIGHT), _KELLY_MIN_WEIGHT)
        parlay_weights[pid] = weight

    # ── Layer 4: Correlation-aware fixture concentration dampening ──
    # Count fixture appearances across kept parlays
    fixture_counts = defaultdict(int)
    n_kept = len(after_gate)
    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        seen = set()
        for leg in legs:
            fid = leg["fixture_id"]
            if fid and fid not in seen:
                fixture_counts[fid] += 1
                seen.add(fid)

    # Dampen parlays with over-concentrated fixtures
    concentrated = {fid: cnt for fid, cnt in fixture_counts.items()
                    if cnt / n_kept > _CORR_FIXTURE_CAP}
    if concentrated:
        print(f"[fullstack] Fixture concentration dampening: "
              f"{len(concentrated)} fixtures above {_CORR_FIXTURE_CAP*100:.0f}% cap")
        for row in after_gate:
            pid = row[0]
            legs = parlay_legs.get(pid, [])
            max_conc = 0
            for leg in legs:
                fid = leg["fixture_id"]
                if fid in concentrated:
                    conc_ratio = fixture_counts[fid] / n_kept
                    max_conc = max(max_conc, conc_ratio)
            if max_conc > _CORR_FIXTURE_CAP:
                # Dampen: weight *= cap / actual_concentration
                dampen = _CORR_FIXTURE_CAP / max_conc
                parlay_weights[pid] = max(parlay_weights[pid] * dampen,
                                          _KELLY_MIN_WEIGHT)

    # ── Layer 5: Daily exposure cap ──
    # Scale all weights proportionally so total daily staked ≤ 15% of bankroll.
    # Effective-N ≈ 22/day (8.6x correlated) → the slate is ~1 bet.
    # Cap is a pure volatility dial: Sharpe constant at 0.428 regardless of level.
    max_daily = _DAILY_EXPOSURE_CAP * _NOTIONAL_BANKROLL
    pre_cap_staked = sum(w * _KELLY_BASE_STAKE for w in parlay_weights.values())
    if pre_cap_staked > max_daily:
        scale = max_daily / pre_cap_staked
        for pid in parlay_weights:
            parlay_weights[pid] = round(parlay_weights[pid] * scale, 6)
        print(f"[fullstack] Daily exposure cap: ${pre_cap_staked:.1f} → "
              f"${max_daily:.1f} (scale={scale:.3f})")
    else:
        print(f"[fullstack] Daily exposure: ${pre_cap_staked:.1f} "
              f"within ${max_daily:.1f} cap — no scaling needed")

    # ── Summary stats ──
    weights = list(parlay_weights.values())
    total_staked = sum(w * _KELLY_BASE_STAKE for w in weights)
    equal_staked = n_kept * _KELLY_BASE_STAKE
    neg_edge = sum(1 for row in after_gate
                   if parlay_weights[row[0]] <= _KELLY_MIN_WEIGHT)

    print(f"[fullstack] {n_kept} kept parlays: "
          f"avg_weight={np.mean(weights):.3f}, "
          f"median={np.median(weights):.3f}, "
          f"staked=${total_staked:.1f}/${equal_staked:.1f} "
          f"({total_staked/equal_staked*100:.0f}%), "
          f"neg_edge={neg_edge}")

    # ── Store as shadow_negbin_fullstack ──
    run_id = str(uuid.uuid4())
    stored = 0
    for row in after_gate:
        pid, bet_info, odds, comp, band, lc, wp = row
        weight = round(parlay_weights[pid], 4)
        sized_amount = round(_KELLY_BASE_STAKE * weight, 2)
        dist_prob = parlay_dist_wp[pid]
        mock_id = str(uuid.uuid4())

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if (lc or 2) > 1 else "straight",
            odds=odds,
            amount=sized_amount,
            legs=lc or 2,
            bet_info=bet_info,
            predicted_win_prob=dist_prob,  # dist_wp replaces negbin_wp
            model_used="negbin_v1+dist_v1",
            status="PENDING",
            source=_FULLSTACK_SOURCE,
            composite_score=comp,
            cohort="clean",
            weight=weight,
        )
        db.add(mock)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_leg_count=:lc "
                "WHERE id=:mid"
            ), {"band": band, "lc": lc, "mid": mock_id})
        except Exception:
            pass

        # Copy legs with all columns
        legs = parlay_legs.get(pid, [])
        for leg in legs:
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=leg["leg_index"],
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["win_prob"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()
            updates = {}
            if leg["negbin_wp"] is not None:
                updates["negbin_wp"] = leg["negbin_wp"]
            if leg["negbin_edge_pp"] is not None:
                updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
            if leg["negbin_mu_home"] is not None:
                updates["negbin_mu_home"] = leg["negbin_mu_home"]
            if leg["negbin_mu_away"] is not None:
                updates["negbin_mu_away"] = leg["negbin_mu_away"]
            if leg["negbin_cell"] is not None:
                updates["negbin_cell"] = leg["negbin_cell"]
            if leg["ats_cover_wp"] is not None:
                updates["ats_cover_wp"] = leg["ats_cover_wp"]
            if leg["ats_dist_wp"] is not None:
                updates["ats_dist_wp"] = leg["ats_dist_wp"]
            if updates:
                set_clause = ", ".join(f"{k}=:{k}" for k in updates)
                updates["lid"] = leg_obj.id
                try:
                    db.execute(_text(
                        f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                    ), updates)
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[fullstack] DB commit failed: {e}")
        return []

    print(f"[fullstack] Stored {stored} fullstack parlays "
          f"(source={_FULLSTACK_SOURCE})")

    # ── Tracker ──
    _report_fullstack_tracker(db)

    return after_gate


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE ARM — ridge_composite leg selection + bayes_team_wp floor
# ═══════════════════════════════════════════════════════════════════════════
_COMPOSITE_SOURCE = "shadow_negbin_composite"
_COMPOSITE_RIDGE_FLOOR = 0.80       # leg must have ridge_composite >= this
_COMPOSITE_BAYES_FLOOR = 0.65       # leg must have bayes_team_wp >= this
_COMPOSITE_PARLAY_MIN_RC = 0.75     # min(ridge_composite) across parlay legs


def generate_composite_counterfactual(db=None, today_str=None):
    """Composite counterfactual: ridge_composite leg selection + bayes floor.

    Architecture:
      1. Start from Shadow A parlays (same as fullstack)
      2. Edge gate (same as fullstack: reject max edge_pp >= 12)
      3. Leg screening: reject legs where ridge_composite < 0.80 OR bayes_team_wp < 0.65
         → kills CIN (bayes 0.24, composite 0.49) and OAK (bayes 0.11, composite 0.36)
      4. Parlay screening: reject parlays where min(ridge_composite) < 0.75
      5. Quarter-Kelly sizing on remaining parlays (same as fullstack)

    Source tag: shadow_negbin_composite
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text
    import re

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[composite] Generating composite counterfactual for {today_str} "
          f"(ridge>={_COMPOSITE_RIDGE_FLOOR}, bayes>={_COMPOSITE_BAYES_FLOOR})")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _COMPOSITE_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[composite] Already {existing} composite parlays for {today_str} — skipping")
        return []

    # ── Read Shadow A's parlays for today ──
    a_parlays = db.execute(_text(
        "SELECT id, bet_info, odds, composite_score, negbin_odds_band, "
        "       negbin_leg_count, predicted_win_prob "
        "FROM mock_bets "
        "WHERE source='shadow_negbin' AND game_date=:gd "
        "  AND cohort='clean' "
        "ORDER BY composite_score DESC"
    ), {"gd": today_str}).fetchall()

    if not a_parlays:
        print("[composite] No Shadow A parlays found for today")
        return []

    # ── Read all legs with signals (including ridge_composite + bayes_team_wp) ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.fixture_id, l.leg_index, l.ats_cover_wp, l.negbin_wp, "
        "       l.negbin_mu_home, l.negbin_mu_away, l.negbin_cell, l.ats_dist_wp, "
        "       l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    parlay_legs = defaultdict(list)
    for row_l in leg_data:
        (mid, desc, wp, edge, fix_id, idx, ats, nb_wp,
         mu_h, mu_a, cell, dist_wp, ridge_c, bayes_wp) = row_l
        parlay_legs[mid].append({
            "description": desc, "win_prob": wp, "negbin_edge_pp": edge,
            "fixture_id": fix_id, "leg_index": idx, "ats_cover_wp": ats,
            "negbin_wp": nb_wp, "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
            "negbin_cell": cell, "ats_dist_wp": dist_wp,
            "ridge_composite": ridge_c, "bayes_team_wp": bayes_wp,
        })

    # ── Layer 1: Edge gate (same as fullstack) ──
    after_gate = []
    n_rejected_edge = 0
    for row in a_parlays:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        edges = [leg["negbin_edge_pp"] for leg in legs
                 if leg["negbin_edge_pp"] is not None]
        max_edge = max(edges) if edges else 0
        if max_edge >= _EDGE_GATE_THRESHOLD_PP:
            n_rejected_edge += 1
        else:
            after_gate.append(row)

    print(f"[composite] Edge gate: {len(a_parlays)} → {len(after_gate)} "
          f"({n_rejected_edge} rejected)")

    if not after_gate:
        print("[composite] All parlays rejected by edge gate")
        return []

    # ── Layer 2: Leg-level composite screening ──
    # Reject parlays that contain ANY leg with ridge_composite < floor OR bayes_team_wp < floor
    after_leg_screen = []
    n_rejected_leg = 0
    n_missing_signal = 0

    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        reject = False
        has_signal = False
        for leg in legs:
            rc = leg["ridge_composite"]
            bwp = leg["bayes_team_wp"]
            if rc is None or bwp is None:
                # Can't evaluate — reject (conservative: don't pass unscored legs)
                reject = True
                n_missing_signal += 1
                break
            has_signal = True
            if rc < _COMPOSITE_RIDGE_FLOOR:
                reject = True
                break
            if bwp < _COMPOSITE_BAYES_FLOOR:
                reject = True
                break
        if reject:
            n_rejected_leg += 1
        else:
            after_leg_screen.append(row)

    print(f"[composite] Leg screen: {len(after_gate)} → {len(after_leg_screen)} "
          f"({n_rejected_leg} rejected, {n_missing_signal} missing signals)")

    if not after_leg_screen:
        print("[composite] All parlays rejected by leg screening")
        return []

    # ── Layer 3: Parlay-level min(ridge_composite) check ──
    after_parlay_screen = []
    n_rejected_parlay = 0

    for row in after_leg_screen:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        min_rc = min(leg["ridge_composite"] for leg in legs)
        if min_rc < _COMPOSITE_PARLAY_MIN_RC:
            n_rejected_parlay += 1
        else:
            after_parlay_screen.append(row)

    print(f"[composite] Parlay min-RC screen: {len(after_leg_screen)} → {len(after_parlay_screen)} "
          f"({n_rejected_parlay} rejected, floor={_COMPOSITE_PARLAY_MIN_RC})")

    if not after_parlay_screen:
        print("[composite] All parlays rejected by parlay min-RC screen")
        return []

    # ── Layer 4: dist_wp parlay-level cover probability ──
    parlay_dist_wp = {}
    n_missing_dist = 0

    for row in after_parlay_screen:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        leg_probs = []
        for leg in legs:
            dist = leg["ats_dist_wp"]
            if dist is not None and dist > 0:
                leg_probs.append(dist)
            else:
                # Fallback: market-implied (conservative), NOT win_prob (overconfident)
                nb_wp = leg.get("negbin_wp")
                epp = leg.get("negbin_edge_pp")
                if nb_wp is not None and epp is not None:
                    mkt_fb = nb_wp - epp / 100.0
                    leg_probs.append(max(mkt_fb, 0.30))
                else:
                    leg_probs.append(0.65)
                n_missing_dist += 1

        parlay_prob = 1.0
        for p in leg_probs:
            parlay_prob *= p
        parlay_dist_wp[pid] = parlay_prob

    if n_missing_dist > 0:
        print(f"[composite] {n_missing_dist} legs missing dist_wp — used market fallback")

    # ── Layer 5: Quarter-Kelly sizing (same logic as fullstack) ──
    dist_pairs = [(row[0], parlay_dist_wp[row[0]]) for row in after_parlay_screen]
    dist_pairs.sort(key=lambda x: x[1])
    n_parlays = len(dist_pairs)

    _Q_WRS = [0.60, 0.70, 0.75, 0.80, 0.90]

    parlay_weights = {}
    for rank_idx, (pid, dp) in enumerate(dist_pairs):
        rank_frac = rank_idx / max(n_parlays - 1, 1)
        q_idx = rank_frac * 4
        q_lo = int(q_idx)
        q_hi = min(q_lo + 1, 4)
        frac = q_idx - q_lo
        p_mapped = _Q_WRS[q_lo] * (1 - frac) + _Q_WRS[q_hi] * frac

        odds = dict((r[0], r[2] or 2.0) for r in after_parlay_screen)[pid]

        edge = p_mapped * odds - 1.0
        if edge <= 0 or odds <= 1.0:
            kelly_frac = 0.0
        else:
            kelly_frac = edge / (odds - 1.0)

        quarter_kelly = kelly_frac * _KELLY_FRACTION
        weight = max(min(quarter_kelly, _KELLY_MAX_WEIGHT), _KELLY_MIN_WEIGHT)
        parlay_weights[pid] = weight

    # ── Layer 6: Correlation-aware fixture concentration dampening ──
    fixture_counts = defaultdict(int)
    n_kept = len(after_parlay_screen)
    for row in after_parlay_screen:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        seen = set()
        for leg in legs:
            fid = leg["fixture_id"]
            if fid and fid not in seen:
                fixture_counts[fid] += 1
                seen.add(fid)

    concentrated = {fid: cnt for fid, cnt in fixture_counts.items()
                    if cnt / n_kept > _CORR_FIXTURE_CAP}
    if concentrated:
        print(f"[composite] Fixture concentration dampening: "
              f"{len(concentrated)} fixtures above {_CORR_FIXTURE_CAP*100:.0f}% cap")
        for row in after_parlay_screen:
            pid = row[0]
            legs = parlay_legs.get(pid, [])
            max_conc = 0
            for leg in legs:
                fid = leg["fixture_id"]
                if fid in concentrated:
                    conc_ratio = fixture_counts[fid] / n_kept
                    max_conc = max(max_conc, conc_ratio)
            if max_conc > _CORR_FIXTURE_CAP:
                dampen = _CORR_FIXTURE_CAP / max_conc
                parlay_weights[pid] = max(parlay_weights[pid] * dampen,
                                          _KELLY_MIN_WEIGHT)

    # ── Layer 7: Daily exposure cap ──
    max_daily = _DAILY_EXPOSURE_CAP * _NOTIONAL_BANKROLL
    pre_cap_staked = sum(w * _KELLY_BASE_STAKE for w in parlay_weights.values())
    if pre_cap_staked > max_daily:
        scale = max_daily / pre_cap_staked
        for pid in parlay_weights:
            parlay_weights[pid] = round(parlay_weights[pid] * scale, 6)
        print(f"[composite] Daily exposure cap: ${pre_cap_staked:.1f} → "
              f"${max_daily:.1f} (scale={scale:.3f})")
    else:
        print(f"[composite] Daily exposure: ${pre_cap_staked:.1f} "
              f"within ${max_daily:.1f} cap — no scaling needed")

    # ── Summary ──
    weights = list(parlay_weights.values())
    total_staked = sum(w * _KELLY_BASE_STAKE for w in weights)
    equal_staked = n_kept * _KELLY_BASE_STAKE
    neg_edge = sum(1 for row in after_parlay_screen
                   if parlay_weights[row[0]] <= _KELLY_MIN_WEIGHT)

    print(f"[composite] {n_kept} kept parlays: "
          f"avg_weight={np.mean(weights):.3f}, "
          f"median={np.median(weights):.3f}, "
          f"staked=${total_staked:.1f}/${equal_staked:.1f} "
          f"({total_staked/equal_staked*100:.0f}%), "
          f"neg_edge={neg_edge}")

    # ── Store as shadow_negbin_composite ──
    run_id = str(uuid.uuid4())
    stored = 0
    for row in after_parlay_screen:
        pid, bet_info, odds, comp, band, lc, wp = row
        weight = round(parlay_weights[pid], 4)
        sized_amount = round(_KELLY_BASE_STAKE * weight, 2)
        dist_prob = parlay_dist_wp[pid]
        mock_id = str(uuid.uuid4())

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if (lc or 2) > 1 else "straight",
            odds=odds,
            amount=sized_amount,
            legs=lc or 2,
            bet_info=bet_info,
            predicted_win_prob=dist_prob,
            model_used="negbin_v1+ridge_composite",
            status="PENDING",
            source=_COMPOSITE_SOURCE,
            composite_score=comp,
            cohort="clean",
            weight=weight,
        )
        db.add(mock)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_leg_count=:lc "
                "WHERE id=:mid"
            ), {"band": band, "lc": lc, "mid": mock_id})
        except Exception:
            pass

        # Copy legs with all columns
        legs = parlay_legs.get(pid, [])
        for leg in legs:
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=leg["leg_index"],
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["win_prob"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()
            updates = {}
            if leg["negbin_wp"] is not None:
                updates["negbin_wp"] = leg["negbin_wp"]
            if leg["negbin_edge_pp"] is not None:
                updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
            if leg["negbin_mu_home"] is not None:
                updates["negbin_mu_home"] = leg["negbin_mu_home"]
            if leg["negbin_mu_away"] is not None:
                updates["negbin_mu_away"] = leg["negbin_mu_away"]
            if leg["negbin_cell"] is not None:
                updates["negbin_cell"] = leg["negbin_cell"]
            if leg["ats_cover_wp"] is not None:
                updates["ats_cover_wp"] = leg["ats_cover_wp"]
            if leg["ats_dist_wp"] is not None:
                updates["ats_dist_wp"] = leg["ats_dist_wp"]
            if leg["bayes_team_wp"] is not None:
                updates["bayes_team_wp"] = leg["bayes_team_wp"]
            if leg["ridge_composite"] is not None:
                updates["ridge_composite"] = leg["ridge_composite"]
            if updates:
                set_clause = ", ".join(f"{k}=:{k}" for k in updates)
                updates["lid"] = leg_obj.id
                try:
                    db.execute(_text(
                        f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                    ), updates)
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[composite] DB commit failed: {e}")
        return []

    print(f"[composite] Stored {stored} composite parlays "
          f"(source={_COMPOSITE_SOURCE})")

    return after_parlay_screen


# ═══════════════════════════════════════════════════════════════════════════
# REAL MARKET-IMPLIED HELPER — used by dist_wp gate and both-gates arms
# ═══════════════════════════════════════════════════════════════════════════

# Cache: fixture_id → (home_team, away_team) — populated per-day
_fixture_teams_cache = {}


def _get_fixture_teams(fixture_id, db):
    """Resolve fixture_id → (home_team, away_team) from betiq.db fixtures table."""
    if fixture_id in _fixture_teams_cache:
        return _fixture_teams_cache[fixture_id]
    from sqlalchemy import text as _text
    row = db.execute(_text(
        "SELECT home_team, away_team FROM fixtures WHERE id=:fid"
    ), {"fid": fixture_id}).first()
    if row:
        _fixture_teams_cache[fixture_id] = (row[0], row[1])
        return (row[0], row[1])
    return (None, None)


def _real_market_implied_for_leg(leg, game_date, db):
    """Get real market-implied probability from alt_lines for a counterfactual leg.

    Parses leg description ("TeamName +1.5") to extract pick_team and rung,
    then queries alt_lines for actual decimal odds.
    Falls back to back-derivation (negbin_wp - edge_pp/100) if alt_lines miss.

    Returns: market_implied probability (float) or None if completely unavailable.
    """
    import re
    desc = leg.get("description", "")
    # Parse "TeamName +1.5" or "TeamName +2.5" etc.
    m = re.match(r"^(.+?)\s+\+(\d+\.?\d*)$", desc)
    if not m:
        # Fallback: back-derive
        nb_wp = leg.get("negbin_wp")
        edge_pp = leg.get("negbin_edge_pp")
        if nb_wp is not None and edge_pp is not None:
            return nb_wp - edge_pp / 100.0
        return None

    pick_team = m.group(1).strip()
    rung = float(m.group(2))

    # Resolve fixture to get home/away
    fixture_id = leg.get("fixture_id")
    if fixture_id:
        home_team, away_team = _get_fixture_teams(fixture_id, db)
    else:
        home_team, away_team = None, None

    if home_team is None or away_team is None:
        # Fallback: back-derive
        nb_wp = leg.get("negbin_wp")
        edge_pp = leg.get("negbin_edge_pp")
        if nb_wp is not None and edge_pp is not None:
            return nb_wp - edge_pp / 100.0
        return None

    # Determine side
    if pick_team == home_team:
        side = "home"
    elif pick_team == away_team:
        side = "away"
    else:
        # Fuzzy: team name might be abbreviated — fallback
        nb_wp = leg.get("negbin_wp")
        edge_pp = leg.get("negbin_edge_pp")
        if nb_wp is not None and edge_pp is not None:
            return nb_wp - edge_pp / 100.0
        return None

    mkt_wp, dec_odds = _get_market_implied(home_team, away_team, game_date, rung, side)
    if mkt_wp is not None:
        return mkt_wp

    # Final fallback: back-derive
    nb_wp = leg.get("negbin_wp")
    edge_pp = leg.get("negbin_edge_pp")
    if nb_wp is not None and edge_pp is not None:
        return nb_wp - edge_pp / 100.0
    return None


# ═══════════════════════════════════════════════════════════════════════════
# DIST_WP GATE ARM — dist_wp > market_implied per-leg inclusion gate
# ═══════════════════════════════════════════════════════════════════════════
_DISTWP_GATE_SOURCE = "shadow_negbin_distwp_gate"


def generate_distwp_gated_counterfactual(db=None, today_str=None):
    """dist_wp-gated counterfactual: keep legs only where dist_wp > market implied.

    Architecture:
      1. Start from Shadow A parlays (same fixture snapshot)
      2. For each leg, derive market_implied = negbin_wp - negbin_edge_pp/100
      3. Gate: keep leg only if ats_dist_wp >= market_implied
      4. Reject parlay if ANY leg fails the gate (no partial parlays)
      5. Quarter-Kelly sizing on remaining parlays (same dist_wp quintile-mapped WRs)

    Rationale:
      dist_wp has rho=0.285 discrimination vs NegBin rho=0.077 (3.7x).
      On disagreement legs (50%), dist_wp is within 2.65pp of actual;
      NegBin overconfident by 14.44pp. Gate validated at +0.118/unit on real rung odds.

    Source tag: shadow_negbin_distwp_gate
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[distwp-gate] Generating dist_wp-gated counterfactual for {today_str}")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _DISTWP_GATE_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[distwp-gate] Already {existing} parlays for {today_str} — skipping")
        return []

    # ── Read Shadow A's parlays for today ──
    a_parlays = db.execute(_text(
        "SELECT id, bet_info, odds, composite_score, negbin_odds_band, "
        "       negbin_leg_count, predicted_win_prob "
        "FROM mock_bets "
        "WHERE source='shadow_negbin' AND game_date=:gd "
        "  AND cohort='clean' "
        "ORDER BY composite_score DESC"
    ), {"gd": today_str}).fetchall()

    if not a_parlays:
        print("[distwp-gate] No Shadow A parlays found for today")
        return []

    # ── Read all legs with signals ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.fixture_id, l.leg_index, l.ats_cover_wp, l.negbin_wp, "
        "       l.negbin_mu_home, l.negbin_mu_away, l.negbin_cell, l.ats_dist_wp, "
        "       l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    parlay_legs = defaultdict(list)
    for row_l in leg_data:
        (mid, desc, wp, edge, fix_id, idx, ats, nb_wp,
         mu_h, mu_a, cell, dist_wp, ridge_c, bayes_wp) = row_l
        parlay_legs[mid].append({
            "description": desc, "win_prob": wp, "negbin_edge_pp": edge,
            "fixture_id": fix_id, "leg_index": idx, "ats_cover_wp": ats,
            "negbin_wp": nb_wp, "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
            "negbin_cell": cell, "ats_dist_wp": dist_wp,
            "ridge_composite": ridge_c, "bayes_team_wp": bayes_wp,
        })

    # ── Gate: dist_wp >= market_implied per leg ──
    # Uses REAL market-implied from alt_lines (actual odds), back-derive as fallback
    after_gate = []
    n_rejected = 0
    n_missing = 0
    n_legs_rejected = 0
    n_legs_total = 0
    n_real_odds = 0
    n_fallback = 0

    n_epp_toxic = 0
    for row in a_parlays:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        reject = False
        for leg in legs:
            n_legs_total += 1

            # epp >= 12 toxicity filter
            epp = leg.get("negbin_edge_pp")
            if epp is not None and epp >= _EDGE_GATE_THRESHOLD_PP:
                reject = True
                n_epp_toxic += 1
                break

            dist_wp = leg["ats_dist_wp"]

            if dist_wp is None:
                reject = True
                n_missing += 1
                break

            # Get real market-implied from alt_lines
            market_implied = _real_market_implied_for_leg(leg, today_str, db)

            if market_implied is None:
                reject = True
                n_missing += 1
                break

            # Track source (real odds vs fallback)
            nb_wp = leg.get("negbin_wp")
            edge_pp = leg.get("negbin_edge_pp")
            if nb_wp is not None and edge_pp is not None:
                back_derived = nb_wp - edge_pp / 100.0
                if abs(market_implied - back_derived) > 0.001:
                    n_real_odds += 1  # meaningfully different = came from real odds
                else:
                    n_fallback += 1
            else:
                n_real_odds += 1

            if dist_wp < market_implied:
                reject = True
                n_legs_rejected += 1
                break

        if reject:
            n_rejected += 1
        else:
            after_gate.append(row)

    print(f"[distwp-gate] Market-implied source: {n_real_odds} real-odds, "
          f"{n_fallback} back-derived (matched)")

    print(f"[distwp-gate] dist_wp + epp<{_EDGE_GATE_THRESHOLD_PP} gate: {len(a_parlays)} → {len(after_gate)} "
          f"({n_rejected} rejected, {n_missing} missing signals, "
          f"{n_legs_rejected} legs failed gate, {n_epp_toxic} epp toxic)")

    if not after_gate:
        print("[distwp-gate] All parlays rejected by dist_wp gate")
        return []

    # ── dist_wp parlay-level cover probability ──
    parlay_dist_wp = {}
    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        leg_probs = []
        for leg in legs:
            dist = leg["ats_dist_wp"]
            if dist is not None and dist > 0:
                leg_probs.append(dist)
            else:
                # Market-implied fallback (conservative), not win_prob
                nb_wp = leg.get("negbin_wp")
                epp = leg.get("negbin_edge_pp")
                if nb_wp is not None and epp is not None:
                    leg_probs.append(max(nb_wp - epp / 100.0, 0.30))
                else:
                    leg_probs.append(0.65)

        parlay_prob = 1.0
        for p in leg_probs:
            parlay_prob *= p
        parlay_dist_wp[pid] = parlay_prob

    # ── Quarter-Kelly sizing (same logic as fullstack) ──
    dist_pairs = [(row[0], parlay_dist_wp[row[0]]) for row in after_gate]
    dist_pairs.sort(key=lambda x: x[1])
    n_parlays = len(dist_pairs)

    _Q_WRS = [0.60, 0.70, 0.75, 0.80, 0.90]

    parlay_weights = {}
    for rank_idx, (pid, dp) in enumerate(dist_pairs):
        rank_frac = rank_idx / max(n_parlays - 1, 1)
        q_idx = rank_frac * 4
        q_lo = int(q_idx)
        q_hi = min(q_lo + 1, 4)
        frac = q_idx - q_lo
        p_mapped = _Q_WRS[q_lo] * (1 - frac) + _Q_WRS[q_hi] * frac

        odds = dict((r[0], r[2] or 2.0) for r in after_gate)[pid]

        edge = p_mapped * odds - 1.0
        if edge <= 0 or odds <= 1.0:
            kelly_frac = 0.0
        else:
            kelly_frac = edge / (odds - 1.0)

        quarter_kelly = kelly_frac * _KELLY_FRACTION
        weight = max(min(quarter_kelly, _KELLY_MAX_WEIGHT), _KELLY_MIN_WEIGHT)
        parlay_weights[pid] = weight

    # ── Correlation-aware fixture concentration dampening ──
    fixture_counts = defaultdict(int)
    n_kept = len(after_gate)
    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        seen = set()
        for leg in legs:
            fid = leg["fixture_id"]
            if fid and fid not in seen:
                fixture_counts[fid] += 1
                seen.add(fid)

    concentrated = {fid: cnt for fid, cnt in fixture_counts.items()
                    if cnt / n_kept > _CORR_FIXTURE_CAP}
    if concentrated:
        print(f"[distwp-gate] Fixture concentration dampening: "
              f"{len(concentrated)} fixtures above {_CORR_FIXTURE_CAP*100:.0f}% cap")
        for row in after_gate:
            pid = row[0]
            legs = parlay_legs.get(pid, [])
            max_conc = 0
            for leg in legs:
                fid = leg["fixture_id"]
                if fid in concentrated:
                    conc_ratio = fixture_counts[fid] / n_kept
                    max_conc = max(max_conc, conc_ratio)
            if max_conc > _CORR_FIXTURE_CAP:
                dampen = _CORR_FIXTURE_CAP / max_conc
                parlay_weights[pid] = max(parlay_weights[pid] * dampen,
                                          _KELLY_MIN_WEIGHT)

    # ── Daily exposure cap ──
    max_daily = _DAILY_EXPOSURE_CAP * _NOTIONAL_BANKROLL
    pre_cap_staked = sum(w * _KELLY_BASE_STAKE for w in parlay_weights.values())
    if pre_cap_staked > max_daily:
        scale = max_daily / pre_cap_staked
        for pid in parlay_weights:
            parlay_weights[pid] = round(parlay_weights[pid] * scale, 6)
        print(f"[distwp-gate] Daily exposure cap: ${pre_cap_staked:.1f} → "
              f"${max_daily:.1f} (scale={scale:.3f})")
    else:
        print(f"[distwp-gate] Daily exposure: ${pre_cap_staked:.1f} "
              f"within ${max_daily:.1f} cap — no scaling needed")

    # ── Summary ──
    weights = list(parlay_weights.values())
    total_staked = sum(w * _KELLY_BASE_STAKE for w in weights)
    equal_staked = n_kept * _KELLY_BASE_STAKE
    neg_edge = sum(1 for row in after_gate
                   if parlay_weights[row[0]] <= _KELLY_MIN_WEIGHT)

    print(f"[distwp-gate] {n_kept} kept parlays: "
          f"avg_weight={np.mean(weights):.3f}, "
          f"median={np.median(weights):.3f}, "
          f"staked=${total_staked:.1f}/${equal_staked:.1f} "
          f"({total_staked/equal_staked*100:.0f}%), "
          f"neg_edge={neg_edge}")

    # ── Store as shadow_negbin_distwp_gate ──
    run_id = str(uuid.uuid4())
    stored = 0
    for row in after_gate:
        pid, bet_info, odds, comp, band, lc, wp = row
        weight = round(parlay_weights[pid], 4)
        sized_amount = round(_KELLY_BASE_STAKE * weight, 2)
        dist_prob = parlay_dist_wp[pid]
        mock_id = str(uuid.uuid4())

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if (lc or 2) > 1 else "straight",
            odds=odds,
            amount=sized_amount,
            legs=lc or 2,
            bet_info=bet_info,
            predicted_win_prob=dist_prob,
            model_used="negbin_v1+distwp_gate",
            status="PENDING",
            source=_DISTWP_GATE_SOURCE,
            composite_score=comp,
            cohort="clean",
            weight=weight,
        )
        db.add(mock)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_leg_count=:lc "
                "WHERE id=:mid"
            ), {"band": band, "lc": lc, "mid": mock_id})
        except Exception:
            pass

        # Copy legs with all columns
        legs = parlay_legs.get(pid, [])
        for leg in legs:
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=leg["leg_index"],
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["win_prob"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()
            updates = {}
            if leg["negbin_wp"] is not None:
                updates["negbin_wp"] = leg["negbin_wp"]
            if leg["negbin_edge_pp"] is not None:
                updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
            if leg["negbin_mu_home"] is not None:
                updates["negbin_mu_home"] = leg["negbin_mu_home"]
            if leg["negbin_mu_away"] is not None:
                updates["negbin_mu_away"] = leg["negbin_mu_away"]
            if leg["negbin_cell"] is not None:
                updates["negbin_cell"] = leg["negbin_cell"]
            if leg["ats_cover_wp"] is not None:
                updates["ats_cover_wp"] = leg["ats_cover_wp"]
            if leg["ats_dist_wp"] is not None:
                updates["ats_dist_wp"] = leg["ats_dist_wp"]
            if leg["bayes_team_wp"] is not None:
                updates["bayes_team_wp"] = leg["bayes_team_wp"]
            if leg["ridge_composite"] is not None:
                updates["ridge_composite"] = leg["ridge_composite"]
            if updates:
                set_clause = ", ".join(f"{k}=:{k}" for k in updates)
                updates["lid"] = leg_obj.id
                try:
                    db.execute(_text(
                        f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                    ), updates)
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[distwp-gate] DB commit failed: {e}")
        return []

    print(f"[distwp-gate] Stored {stored} dist_wp-gated parlays "
          f"(source={_DISTWP_GATE_SOURCE})")

    return after_gate


# ═══════════════════════════════════════════════════════════════════════════
# BOTH-GATES ARM — dist_wp >= market_implied AND edge < 12pp (toxicity gate)
# ═══════════════════════════════════════════════════════════════════════════
_BOTHGATES_SOURCE = "shadow_negbin_bothgates"


def generate_bothgates_counterfactual(db=None, today_str=None):
    """Both-gates counterfactual: dist_wp gate + edge toxicity gate per leg.

    Architecture:
      1. Start from Shadow A parlays (same fixture snapshot)
      2. Per-leg dual gate:
         (a) dist_wp >= market_implied (real odds from alt_lines)
         (b) negbin_edge_pp < 12 (toxicity gate — reject overconfident legs)
      3. Reject parlay if ANY leg fails EITHER gate
      4. Quarter-Kelly sizing on remaining parlays (same dist_wp quintile-mapped WRs)

    Rationale:
      dist_wp gate alone: +0.118/unit. Both gates: +0.158/unit (higher rate,
      slightly fewer bets). The edge toxicity gate removes legs where NegBin
      is overconfident (>12pp claimed edge) which are empirically -EV.

    Source tag: shadow_negbin_bothgates
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[bothgates] Generating both-gates counterfactual for {today_str} "
          f"(dist_wp>=mkt_implied AND edge<{_EDGE_GATE_THRESHOLD_PP}pp)")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _BOTHGATES_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[bothgates] Already {existing} parlays for {today_str} — skipping")
        return []

    # ── Read Shadow A's parlays for today ──
    a_parlays = db.execute(_text(
        "SELECT id, bet_info, odds, composite_score, negbin_odds_band, "
        "       negbin_leg_count, predicted_win_prob "
        "FROM mock_bets "
        "WHERE source='shadow_negbin' AND game_date=:gd "
        "  AND cohort='clean' "
        "ORDER BY composite_score DESC"
    ), {"gd": today_str}).fetchall()

    if not a_parlays:
        print("[bothgates] No Shadow A parlays found for today")
        return []

    # ── Read all legs with signals ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.fixture_id, l.leg_index, l.ats_cover_wp, l.negbin_wp, "
        "       l.negbin_mu_home, l.negbin_mu_away, l.negbin_cell, l.ats_dist_wp, "
        "       l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    parlay_legs = defaultdict(list)
    for row_l in leg_data:
        (mid, desc, wp, edge, fix_id, idx, ats, nb_wp,
         mu_h, mu_a, cell, dist_wp, ridge_c, bayes_wp) = row_l
        parlay_legs[mid].append({
            "description": desc, "win_prob": wp, "negbin_edge_pp": edge,
            "fixture_id": fix_id, "leg_index": idx, "ats_cover_wp": ats,
            "negbin_wp": nb_wp, "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
            "negbin_cell": cell, "ats_dist_wp": dist_wp,
            "ridge_composite": ridge_c, "bayes_team_wp": bayes_wp,
        })

    # ── Dual gate: dist_wp >= market_implied AND edge < 12pp ──
    after_gate = []
    n_rejected = 0
    n_missing = 0
    n_legs_failed_distwp = 0
    n_legs_failed_edge = 0
    n_legs_total = 0
    n_real_odds = 0
    n_fallback = 0

    for row in a_parlays:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        reject = False
        for leg in legs:
            n_legs_total += 1
            dist_wp = leg["ats_dist_wp"]
            edge_pp = leg["negbin_edge_pp"]

            # Gate (b): edge toxicity — reject if edge >= 12pp
            if edge_pp is not None and edge_pp >= _EDGE_GATE_THRESHOLD_PP:
                reject = True
                n_legs_failed_edge += 1
                break

            # Gate (a): dist_wp >= market_implied
            if dist_wp is None:
                reject = True
                n_missing += 1
                break

            market_implied = _real_market_implied_for_leg(leg, today_str, db)

            if market_implied is None:
                reject = True
                n_missing += 1
                break

            # Track source
            nb_wp = leg.get("negbin_wp")
            if nb_wp is not None and edge_pp is not None:
                back_derived = nb_wp - edge_pp / 100.0
                if abs(market_implied - back_derived) > 0.001:
                    n_real_odds += 1
                else:
                    n_fallback += 1
            else:
                n_real_odds += 1

            if dist_wp < market_implied:
                reject = True
                n_legs_failed_distwp += 1
                break

        if reject:
            n_rejected += 1
        else:
            after_gate.append(row)

    print(f"[bothgates] Dual gate: {len(a_parlays)} → {len(after_gate)} "
          f"({n_rejected} rejected: {n_legs_failed_distwp} dist_wp, "
          f"{n_legs_failed_edge} edge, {n_missing} missing)")
    print(f"[bothgates] Market-implied source: {n_real_odds} real-odds, "
          f"{n_fallback} back-derived (matched)")

    if not after_gate:
        print("[bothgates] All parlays rejected by dual gate")
        return []

    # ── dist_wp parlay-level cover probability ──
    parlay_dist_wp = {}
    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        leg_probs = []
        for leg in legs:
            dist = leg["ats_dist_wp"]
            if dist is not None and dist > 0:
                leg_probs.append(dist)
            else:
                nb_wp = leg.get("negbin_wp")
                epp = leg.get("negbin_edge_pp")
                if nb_wp is not None and epp is not None:
                    leg_probs.append(max(nb_wp - epp / 100.0, 0.30))
                else:
                    leg_probs.append(0.65)

        parlay_prob = 1.0
        for p in leg_probs:
            parlay_prob *= p
        parlay_dist_wp[pid] = parlay_prob

    # ── Quarter-Kelly sizing ──
    dist_pairs = [(row[0], parlay_dist_wp[row[0]]) for row in after_gate]
    dist_pairs.sort(key=lambda x: x[1])
    n_parlays = len(dist_pairs)

    _Q_WRS = [0.60, 0.70, 0.75, 0.80, 0.90]

    parlay_weights = {}
    for rank_idx, (pid, dp) in enumerate(dist_pairs):
        rank_frac = rank_idx / max(n_parlays - 1, 1)
        q_idx = rank_frac * 4
        q_lo = int(q_idx)
        q_hi = min(q_lo + 1, 4)
        frac = q_idx - q_lo
        p_mapped = _Q_WRS[q_lo] * (1 - frac) + _Q_WRS[q_hi] * frac

        odds = dict((r[0], r[2] or 2.0) for r in after_gate)[pid]

        edge = p_mapped * odds - 1.0
        if edge <= 0 or odds <= 1.0:
            kelly_frac = 0.0
        else:
            kelly_frac = edge / (odds - 1.0)

        quarter_kelly = kelly_frac * _KELLY_FRACTION
        weight = max(min(quarter_kelly, _KELLY_MAX_WEIGHT), _KELLY_MIN_WEIGHT)
        parlay_weights[pid] = weight

    # ── Fixture concentration dampening ──
    fixture_counts = defaultdict(int)
    n_kept = len(after_gate)
    for row in after_gate:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        seen = set()
        for leg in legs:
            fid = leg["fixture_id"]
            if fid and fid not in seen:
                fixture_counts[fid] += 1
                seen.add(fid)

    concentrated = {fid: cnt for fid, cnt in fixture_counts.items()
                    if cnt / n_kept > _CORR_FIXTURE_CAP}
    if concentrated:
        print(f"[bothgates] Fixture concentration dampening: "
              f"{len(concentrated)} fixtures above {_CORR_FIXTURE_CAP*100:.0f}% cap")
        for row in after_gate:
            pid = row[0]
            legs = parlay_legs.get(pid, [])
            max_conc = 0
            for leg in legs:
                fid = leg["fixture_id"]
                if fid in concentrated:
                    conc_ratio = fixture_counts[fid] / n_kept
                    max_conc = max(max_conc, conc_ratio)
            if max_conc > _CORR_FIXTURE_CAP:
                dampen = _CORR_FIXTURE_CAP / max_conc
                parlay_weights[pid] = max(parlay_weights[pid] * dampen,
                                          _KELLY_MIN_WEIGHT)

    # ── Daily exposure cap ──
    max_daily = _DAILY_EXPOSURE_CAP * _NOTIONAL_BANKROLL
    pre_cap_staked = sum(w * _KELLY_BASE_STAKE for w in parlay_weights.values())
    if pre_cap_staked > max_daily:
        scale = max_daily / pre_cap_staked
        for pid in parlay_weights:
            parlay_weights[pid] = round(parlay_weights[pid] * scale, 6)
        print(f"[bothgates] Daily exposure cap: ${pre_cap_staked:.1f} → "
              f"${max_daily:.1f} (scale={scale:.3f})")
    else:
        print(f"[bothgates] Daily exposure: ${pre_cap_staked:.1f} "
              f"within ${max_daily:.1f} cap — no scaling needed")

    # ── Summary ──
    weights = list(parlay_weights.values())
    total_staked = sum(w * _KELLY_BASE_STAKE for w in weights)
    equal_staked = n_kept * _KELLY_BASE_STAKE
    neg_edge = sum(1 for row in after_gate
                   if parlay_weights[row[0]] <= _KELLY_MIN_WEIGHT)

    print(f"[bothgates] {n_kept} kept parlays: "
          f"avg_weight={np.mean(weights):.3f}, "
          f"median={np.median(weights):.3f}, "
          f"staked=${total_staked:.1f}/${equal_staked:.1f} "
          f"({total_staked/equal_staked*100:.0f}%), "
          f"neg_edge={neg_edge}")

    # ── Store as shadow_negbin_bothgates ──
    run_id = str(uuid.uuid4())
    stored = 0
    for row in after_gate:
        pid, bet_info, odds, comp, band, lc, wp = row
        weight = round(parlay_weights[pid], 4)
        sized_amount = round(_KELLY_BASE_STAKE * weight, 2)
        dist_prob = parlay_dist_wp[pid]
        mock_id = str(uuid.uuid4())

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if (lc or 2) > 1 else "straight",
            odds=odds,
            amount=sized_amount,
            legs=lc or 2,
            bet_info=bet_info,
            predicted_win_prob=dist_prob,
            model_used="negbin_v1+distwp_gate+edge_gate",
            status="PENDING",
            source=_BOTHGATES_SOURCE,
            composite_score=comp,
            cohort="clean",
            weight=weight,
        )
        db.add(mock)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_leg_count=:lc "
                "WHERE id=:mid"
            ), {"band": band, "lc": lc, "mid": mock_id})
        except Exception:
            pass

        # Copy legs with all columns
        legs = parlay_legs.get(pid, [])
        for leg in legs:
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=leg["leg_index"],
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["win_prob"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()
            updates = {}
            if leg["negbin_wp"] is not None:
                updates["negbin_wp"] = leg["negbin_wp"]
            if leg["negbin_edge_pp"] is not None:
                updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
            if leg["negbin_mu_home"] is not None:
                updates["negbin_mu_home"] = leg["negbin_mu_home"]
            if leg["negbin_mu_away"] is not None:
                updates["negbin_mu_away"] = leg["negbin_mu_away"]
            if leg["negbin_cell"] is not None:
                updates["negbin_cell"] = leg["negbin_cell"]
            if leg["ats_cover_wp"] is not None:
                updates["ats_cover_wp"] = leg["ats_cover_wp"]
            if leg["ats_dist_wp"] is not None:
                updates["ats_dist_wp"] = leg["ats_dist_wp"]
            if leg["bayes_team_wp"] is not None:
                updates["bayes_team_wp"] = leg["bayes_team_wp"]
            if leg["ridge_composite"] is not None:
                updates["ridge_composite"] = leg["ridge_composite"]
            if updates:
                set_clause = ", ".join(f"{k}=:{k}" for k in updates)
                updates["lid"] = leg_obj.id
                try:
                    db.execute(_text(
                        f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                    ), updates)
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[bothgates] DB commit failed: {e}")
        return []

    print(f"[bothgates] Stored {stored} both-gates parlays "
          f"(source={_BOTHGATES_SOURCE})")

    return after_gate


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-CONVICTION ARM — absolute dist_wp ≥ 0.85 per-leg threshold
# ═══════════════════════════════════════════════════════════════════════════
_HIGHCONV_SOURCE = "shadow_negbin_highconv"
_HIGHCONV_DISTWP_FLOOR = 0.85   # 85-90% band: +16.3pp -1.5 margin, $4.77 EV/$10

# Evidence (conviction_ladder.py, 2026-07-27, N=290 de-duped fixtures):
#   dist_wp 85-90%: -1.5 WR=50.4% vs BE=34.1% = +16.3pp, split-half STABLE
#                   +4.5 WR=87.4% vs BE=83.1% = +4.3pp
#   dist_wp 90%+:   +3.5 WR=89.3% vs BE=77.3% = +11.9pp, split-half STABLE
#                   -1.5 WR=37.5% vs BE=34.1% = +3.4pp (in noise at N=56)
# The 0.85 threshold maximises the +EV pool while staying split-half stable.
# Volume: ~7.3 unique fixtures/day at 0.85+, ~2.8/day at 0.90+.


def generate_highconv_counterfactual(db=None, today_str=None):
    """High-conviction counterfactual: keep legs only where ats_dist_wp >= 0.85.

    Absolute dist_wp threshold (not relative to market).
    Equal $10 stake (like edgegate) — sizing comes later once -1.5 rung is live.

    Source tag: shadow_negbin_highconv
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[highconv] Generating high-conviction counterfactual for {today_str}")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _HIGHCONV_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[highconv] Already {existing} parlays for {today_str} — skipping")
        return []

    # ── Read Shadow A's parlays for today ──
    a_parlays = db.execute(_text(
        "SELECT id, bet_info, odds, composite_score, negbin_odds_band, "
        "       negbin_leg_count, predicted_win_prob "
        "FROM mock_bets "
        "WHERE source='shadow_negbin' AND game_date=:gd "
        "  AND cohort='clean' "
        "ORDER BY composite_score DESC"
    ), {"gd": today_str}).fetchall()

    if not a_parlays:
        print("[highconv] No Shadow A parlays found for today")
        return []

    # ── Read all legs with signals ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.fixture_id, l.leg_index, l.ats_cover_wp, l.negbin_wp, "
        "       l.negbin_mu_home, l.negbin_mu_away, l.negbin_cell, l.ats_dist_wp, "
        "       l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    parlay_legs = defaultdict(list)
    for row_l in leg_data:
        (mid, desc, wp, edge, fix_id, idx, ats, nb_wp,
         mu_h, mu_a, cell, dist_wp, ridge_c, bayes_wp) = row_l
        parlay_legs[mid].append({
            "description": desc, "win_prob": wp, "negbin_edge_pp": edge,
            "fixture_id": fix_id, "leg_index": idx, "ats_cover_wp": ats,
            "negbin_wp": nb_wp, "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
            "negbin_cell": cell, "ats_dist_wp": dist_wp,
            "ridge_composite": ridge_c, "bayes_team_wp": bayes_wp,
        })

    # ── Gate: ats_dist_wp >= _HIGHCONV_DISTWP_FLOOR per leg (absolute) ──
    after_gate = []
    n_rejected = 0
    n_missing = 0

    n_epp_toxic = 0
    for row in a_parlays:
        pid = row[0]
        legs = parlay_legs.get(pid, [])
        reject = False
        for leg in legs:
            # epp >= 12 toxicity filter
            epp = leg.get("negbin_edge_pp")
            if epp is not None and epp >= _EDGE_GATE_THRESHOLD_PP:
                reject = True
                n_epp_toxic += 1
                break
            dist_wp = leg["ats_dist_wp"]
            if dist_wp is None:
                reject = True
                n_missing += 1
                break
            if dist_wp < _HIGHCONV_DISTWP_FLOOR:
                reject = True
                break
        if reject:
            n_rejected += 1
        else:
            after_gate.append(row)

    print(f"[highconv] dist_wp≥{_HIGHCONV_DISTWP_FLOOR} + epp<{_EDGE_GATE_THRESHOLD_PP} gate: "
          f"{len(a_parlays)} → {len(after_gate)} "
          f"({n_rejected} rejected, {n_missing} missing dist_wp, {n_epp_toxic} epp toxic)")

    if not after_gate:
        print("[highconv] All parlays rejected by high-conviction gate")
        return []

    # ── Equal $10 stake (flat sizing — same as edgegate) ──
    run_id = str(uuid.uuid4())
    stored = 0
    for row in after_gate:
        pid, bet_info, odds, comp, band, lc, wp = row
        legs = parlay_legs.get(pid, [])

        # Parlay-level dist_wp
        leg_probs = []
        for leg in legs:
            d = leg["ats_dist_wp"]
            if d and d > 0:
                leg_probs.append(d)
            else:
                nb_wp = leg.get("negbin_wp")
                epp = leg.get("negbin_edge_pp")
                if nb_wp is not None and epp is not None:
                    leg_probs.append(max(nb_wp - epp / 100.0, 0.30))
                else:
                    leg_probs.append(0.65)
        parlay_prob = 1.0
        for p in leg_probs:
            parlay_prob *= p

        mock_id = str(uuid.uuid4())
        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if (lc or 2) > 1 else "straight",
            odds=odds,
            amount=10.0,
            legs=lc or 2,
            bet_info=bet_info,
            predicted_win_prob=parlay_prob,
            model_used="negbin_v1+highconv",
            status="PENDING",
            source=_HIGHCONV_SOURCE,
            composite_score=comp,
            cohort="clean",
            weight=1.0,
        )
        db.add(mock)
        db.flush()

        # Set mean/min dist_wp via raw SQL (columns added by ALTER TABLE, not in ORM model)
        try:
            db.execute(_text(
                "UPDATE mock_bets SET mean_leg_dist_wp=:mdw, min_leg_dist_wp=:mindw "
                "WHERE id=:mid"
            ), {"mdw": round(float(np.mean(leg_probs)), 4),
                "mindw": round(float(min(leg_probs)), 4), "mid": mock_id})
        except Exception:
            pass

        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_leg_count=:lc "
                "WHERE id=:mid"
            ), {"band": band, "lc": lc, "mid": mock_id})
        except Exception:
            pass

        for leg in legs:
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=leg["leg_index"],
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["win_prob"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()
            updates = {}
            if leg["negbin_wp"] is not None:
                updates["negbin_wp"] = leg["negbin_wp"]
            if leg["negbin_edge_pp"] is not None:
                updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
            if leg["negbin_mu_home"] is not None:
                updates["negbin_mu_home"] = leg["negbin_mu_home"]
            if leg["negbin_mu_away"] is not None:
                updates["negbin_mu_away"] = leg["negbin_mu_away"]
            if leg["negbin_cell"] is not None:
                updates["negbin_cell"] = leg["negbin_cell"]
            if leg["ats_cover_wp"] is not None:
                updates["ats_cover_wp"] = leg["ats_cover_wp"]
            if leg["ats_dist_wp"] is not None:
                updates["ats_dist_wp"] = leg["ats_dist_wp"]
            if leg["bayes_team_wp"] is not None:
                updates["bayes_team_wp"] = leg["bayes_team_wp"]
            if leg["ridge_composite"] is not None:
                updates["ridge_composite"] = leg["ridge_composite"]
            if updates:
                set_clause = ", ".join(f"{k}=:{k}" for k in updates)
                updates["lid"] = leg_obj.id
                try:
                    db.execute(_text(
                        f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                    ), updates)
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[highconv] DB commit failed: {e}")
        return []

    print(f"[highconv] Stored {stored} high-conviction parlays "
          f"(source={_HIGHCONV_SOURCE})")

    return after_gate


# ═══════════════════════════════════════════════════════════════════════════
# STRAIGHT -1.5 ARM — bet team -1.5 (win by 2+) on 85-90% dist_wp fixtures
# ═══════════════════════════════════════════════════════════════════════════
_NEG15_SOURCE = "shadow_negbin_neg15"
_NEG15_DISTWP_LO = 0.85
_NEG15_DISTWP_HI = 0.90
_NEG15_RUNG = -1.5
_NEG15_STAKE = 10.0
_NEG15_FALLBACK_ODDS = 2.929  # avg -1.5 odds from alt_lines (2026-06-28 to 2026-07-26)
_NEG15_MAX_ODDS = 3.50  # odds ceiling — above this, WR collapses (heavy underdogs can't win by 2+)
# 3.00-3.50 bucket: WR=53.3% vs BE=32.1% = +21.2pp (keep)
# 3.50+: WR=16.7% vs BE=27.2% = -10.6pp (cut)
# N=13 above threshold — mechanically justified (not precisely measured)

# Evidence (conviction_ladder.py + neg15_durability.py, 2026-07-27):
#   dist_wp 85-90% × -1.5: WR=47.0% vs BE=34.1% = +12.9pp, split-half STABLE
#   Straight $10: +$505 PnL, +37.7% ROI, 17/24 profit days
#   Same fixtures at +4.5: +$68, +5.0% ROI — -1.5 is 7.5x the ROI
#   H2 durability: +5.0pp above BE (edgegate went -4.4pp below — different)
#   Discrimination IS the lever: P(team wins)=59.7%, P(decisive|won)=78.8% (MLB constant)


def _get_spread_odds(pick_team, home_team, away_team, game_date, target_line, fallback_odds):
    """Look up spread odds from alt_lines for a specific team and line.

    Falls back to fallback_odds if not found.
    """
    con = sqlite3.connect(_HIST_DB, timeout=15)
    from datetime import date as _date, timedelta as _td
    try:
        next_date = (_date.fromisoformat(game_date) + _td(days=1)).isoformat()
    except Exception:
        next_date = None

    dates_to_try = [game_date]
    if next_date:
        dates_to_try.append(next_date)

    for gd in dates_to_try:
        row = con.execute("""
            SELECT odds FROM alt_lines
            WHERE home_team = ? AND away_team = ? AND game_date = ?
              AND market_key = 'alternate_spreads'
              AND team = ?
              AND ABS(line - ?) < 0.26
              AND odds IS NOT NULL AND odds > 1.0
            ORDER BY fetched_at DESC LIMIT 1
        """, (home_team, away_team, gd, pick_team, target_line)).fetchone()
        if row is not None:
            con.close()
            return float(row[0])

    con.close()
    return fallback_odds


def generate_neg15_straight(db=None, today_str=None):
    """Straight -1.5 bets on 85-90% dist_wp fixtures.

    Architecture:
      1. Read Shadow A's legs for today (same fixture snapshot)
      2. De-duplicate by fixture_id → one entry per game-side
      3. For each fixture: take the max dist_wp across all rungs
      4. Gate: keep fixtures where max_dist_wp ∈ [0.85, 0.90)
      5. Look up actual -1.5 odds from alt_lines
      6. Create one STRAIGHT bet per fixture at $10 flat

    Why straight, not parlay:
      Non-overlapping parlays: PnL $433 mean (lower than straight $505)
      Straight: deterministic, uncorrelated, highest absolute PnL
      The parlay "higher ROI" is on fewer bets with more variance.

    Source tag: shadow_negbin_neg15
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text
    import re

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[neg15] Generating straight -1.5 bets for {today_str}")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _NEG15_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[neg15] Already {existing} bets for {today_str} — skipping")
        return []

    # ── Read Shadow A's legs for today ──
    leg_data = db.execute(_text(
        "SELECT l.fixture_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.ats_cover_wp, l.negbin_wp, l.negbin_mu_home, l.negbin_mu_away, "
        "       l.negbin_cell, l.ats_dist_wp, l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    if not leg_data:
        print("[neg15] No Shadow A legs found for today")
        return []

    # ── De-duplicate by fixture: one entry per game-side, max dist_wp ──
    fixture_legs = {}  # fixture_id → leg dict with max dist_wp
    for row in leg_data:
        (fix_id, desc, wp, edge, ats, nb_wp, mu_h, mu_a,
         cell, dist_wp, ridge, bayes) = row

        if dist_wp is None or fix_id is None:
            continue

        if fix_id not in fixture_legs or dist_wp > fixture_legs[fix_id]["ats_dist_wp"]:
            fixture_legs[fix_id] = {
                "fixture_id": fix_id, "description": desc, "win_prob": wp,
                "negbin_edge_pp": edge, "ats_cover_wp": ats, "negbin_wp": nb_wp,
                "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
                "negbin_cell": cell, "ats_dist_wp": dist_wp,
                "ridge_composite": ridge, "bayes_team_wp": bayes,
            }

    print(f"[neg15] {len(fixture_legs)} unique fixtures found")

    # ── Gate: dist_wp in [0.85, 0.90) + epp < 12 ──
    qualifying = {}
    n_epp_toxic = 0
    for fix_id, leg in fixture_legs.items():
        dwp = leg["ats_dist_wp"]
        epp = leg.get("negbin_edge_pp")
        if epp is not None and epp >= _EDGE_GATE_THRESHOLD_PP:
            n_epp_toxic += 1
            continue
        if _NEG15_DISTWP_LO <= dwp < _NEG15_DISTWP_HI:
            qualifying[fix_id] = leg

    print(f"[neg15] {len(qualifying)} fixtures pass dist_wp "
          f"[{_NEG15_DISTWP_LO}, {_NEG15_DISTWP_HI}) + epp<{_EDGE_GATE_THRESHOLD_PP} gate "
          f"({n_epp_toxic} epp toxic)")

    if not qualifying:
        print("[neg15] No qualifying fixtures")
        return []

    # ── Build straight -1.5 bets ──
    run_id = str(uuid.uuid4())
    stored = 0
    n_real_odds = 0
    n_fallback = 0

    for fix_id, leg in qualifying.items():
        desc = leg["description"]
        # Parse team name from "TeamName +X.X"
        m = re.match(r"^(.+?)\s+\+(\d+\.?\d*)$", desc)
        if not m:
            print(f"[neg15] Skipping unparseable: {desc}")
            continue
        pick_team = m.group(1).strip()

        # Resolve fixture to home/away teams
        home_team, away_team = _get_fixture_teams(fix_id, db)
        if home_team is None or away_team is None:
            print(f"[neg15] Cannot resolve fixture {fix_id}")
            continue

        # Look up -1.5 odds
        neg15_odds = _get_spread_odds(
            pick_team, home_team, away_team, today_str,
            _NEG15_RUNG, _NEG15_FALLBACK_ODDS)
        if abs(neg15_odds - _NEG15_FALLBACK_ODDS) < 0.01:
            n_fallback += 1
        else:
            n_real_odds += 1

        # Odds ceiling: skip heavy underdogs where WR collapses
        if neg15_odds >= _NEG15_MAX_ODDS:
            print(f"[neg15] Skipping {pick_team}: odds {neg15_odds:.3f} >= ceiling {_NEG15_MAX_ODDS} "
                  f"(heavy underdog, WR collapses at long odds)")
            continue

        # Create straight -1.5 bet
        neg15_desc = f"{pick_team} -1.5"
        # Model WP for -1.5 is not directly available; use 1/odds as proxy
        model_wp = 1.0 / neg15_odds

        mock_id = str(uuid.uuid4())
        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=neg15_odds,
            amount=_NEG15_STAKE,
            legs=1,
            bet_info=json.dumps({
                "type": "straight_neg15",
                "team": pick_team,
                "rung": -1.5,
                "dist_wp_at_best_rung": round(leg["ats_dist_wp"], 4),
                "negbin_cell": leg["negbin_cell"],
                "neg15_odds": round(neg15_odds, 3),
                "baseline_rung": float(m.group(2)),
            }),
            predicted_win_prob=model_wp,
            model_used="negbin_v1+neg15_straight",
            status="PENDING",
            source=_NEG15_SOURCE,
            cohort="clean",
            weight=1.0,
        )
        db.add(mock)
        db.flush()

        # Set mean/min dist_wp via raw SQL (columns added by ALTER TABLE, not in ORM model)
        try:
            db.execute(_text(
                "UPDATE mock_bets SET mean_leg_dist_wp=:mdw, min_leg_dist_wp=:mindw "
                "WHERE id=:mid"
            ), {"mdw": round(leg["ats_dist_wp"], 4),
                "mindw": round(leg["ats_dist_wp"], 4), "mid": mock_id})
        except Exception:
            pass

        # Create the single leg
        leg_obj = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=neg15_desc,
            market_type="alternate_spreads",
            sport="MLB",
            win_prob=model_wp,
            fixture_id=fix_id,
            is_alt_line=True,
        )
        db.add(leg_obj)
        db.flush()

        # Set spread_value via raw SQL (column added by ALTER TABLE, not in ORM model)
        try:
            db.execute(_text(
                "UPDATE mock_bet_legs SET spread_value=:sv WHERE id=:lid"
            ), {"sv": -1.5, "lid": leg_obj.id})
        except Exception:
            pass

        # Copy signal columns
        updates = {}
        if leg["negbin_wp"] is not None:
            updates["negbin_wp"] = leg["negbin_wp"]
        if leg["negbin_edge_pp"] is not None:
            updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
        if leg["negbin_mu_home"] is not None:
            updates["negbin_mu_home"] = leg["negbin_mu_home"]
        if leg["negbin_mu_away"] is not None:
            updates["negbin_mu_away"] = leg["negbin_mu_away"]
        if leg["negbin_cell"] is not None:
            updates["negbin_cell"] = leg["negbin_cell"]
        if leg["ats_cover_wp"] is not None:
            updates["ats_cover_wp"] = leg["ats_cover_wp"]
        if leg["ats_dist_wp"] is not None:
            updates["ats_dist_wp"] = leg["ats_dist_wp"]
        if leg["bayes_team_wp"] is not None:
            updates["bayes_team_wp"] = leg["bayes_team_wp"]
        if leg["ridge_composite"] is not None:
            updates["ridge_composite"] = leg["ridge_composite"]
        if updates:
            set_clause = ", ".join(f"{k}=:{k}" for k in updates)
            updates["lid"] = leg_obj.id
            try:
                db.execute(_text(
                    f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                ), updates)
            except Exception:
                pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[neg15] DB commit failed: {e}")
        return []

    print(f"[neg15] Stored {stored} straight -1.5 bets "
          f"({n_real_odds} real odds, {n_fallback} fallback) "
          f"(source={_NEG15_SOURCE})")

    return list(qualifying.values())


# ═══════════════════════════════════════════════════════════════════════════
# STRAIGHT +3.5 ARM — bet team +3.5 (don't lose by 4+) on 90%+ dist_wp fixtures
# ═══════════════════════════════════════════════════════════════════════════
_PLUS35_SOURCE = "shadow_negbin_plus35"
_PLUS35_DISTWP_LO = 0.90
_PLUS35_RUNG = 3.5
_PLUS35_STAKE = 10.0
_PLUS35_FALLBACK_ODDS = 1.299  # avg +3.5 odds from alt_lines
_PLUS35_MIN_ODDS = 1.12  # odds floor — below this, BE > WR (edge inverts)

# Evidence (conviction_ladder.py, 2026-07-27):
#   dist_wp 90%+ × +3.5: +$1.54/$10, split-half stable
#   Disjoint from neg15 (neg15 = [0.85, 0.90), this = 0.90+)
#   Odds floor: odds < 1.12 → BE ~90%+ → edge 0.3pp (noise); odds >= 1.12 → edge +2.7pp


def generate_plus35_straight(db=None, today_str=None):
    """Straight +3.5 bets on 90%+ dist_wp fixtures.

    Companion to neg15: captures the second validated conviction-ladder cell.
    dist_wp >= 0.90 → bet +3.5 (team gets 3.5, covers unless loses by 4+).
    Disjoint from neg15 (different band, different rung).

    Source tag: shadow_negbin_plus35
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text
    import re

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[plus35] Generating straight +3.5 bets for {today_str}")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _PLUS35_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[plus35] Already {existing} bets for {today_str} — skipping")
        return []

    # ── Read Shadow A's legs for today ──
    leg_data = db.execute(_text(
        "SELECT l.fixture_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.ats_cover_wp, l.negbin_wp, l.negbin_mu_home, l.negbin_mu_away, "
        "       l.negbin_cell, l.ats_dist_wp, l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    if not leg_data:
        print("[plus35] No Shadow A legs found for today")
        return []

    # ── De-duplicate by fixture: one entry per game-side, max dist_wp ──
    fixture_legs = {}
    for row in leg_data:
        (fix_id, desc, wp, edge, ats, nb_wp, mu_h, mu_a,
         cell, dist_wp, ridge, bayes) = row

        if dist_wp is None or fix_id is None:
            continue

        if fix_id not in fixture_legs or dist_wp > fixture_legs[fix_id]["ats_dist_wp"]:
            fixture_legs[fix_id] = {
                "fixture_id": fix_id, "description": desc, "win_prob": wp,
                "negbin_edge_pp": edge, "ats_cover_wp": ats, "negbin_wp": nb_wp,
                "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
                "negbin_cell": cell, "ats_dist_wp": dist_wp,
                "ridge_composite": ridge, "bayes_team_wp": bayes,
            }

    print(f"[plus35] {len(fixture_legs)} unique fixtures found")

    # ── Gate: dist_wp >= 0.90 + epp < 12 ──
    qualifying = {}
    n_epp_toxic = 0
    for fix_id, leg in fixture_legs.items():
        dwp = leg["ats_dist_wp"]
        epp = leg.get("negbin_edge_pp")
        if epp is not None and epp >= _EDGE_GATE_THRESHOLD_PP:
            n_epp_toxic += 1
            continue
        if dwp >= _PLUS35_DISTWP_LO:
            qualifying[fix_id] = leg

    print(f"[plus35] {len(qualifying)} fixtures pass dist_wp >= {_PLUS35_DISTWP_LO} "
          f"+ epp<{_EDGE_GATE_THRESHOLD_PP} gate ({n_epp_toxic} epp toxic)")

    if not qualifying:
        print("[plus35] No qualifying fixtures")
        return []

    # ── Build straight +3.5 bets ──
    run_id = str(uuid.uuid4())
    stored = 0
    n_real_odds = 0
    n_fallback = 0

    for fix_id, leg in qualifying.items():
        desc = leg["description"]
        m = re.match(r"^(.+?)\s+\+(\d+\.?\d*)$", desc)
        if not m:
            print(f"[plus35] Skipping unparseable: {desc}")
            continue
        pick_team = m.group(1).strip()

        # Resolve fixture to home/away teams
        home_team, away_team = _get_fixture_teams(fix_id, db)
        if home_team is None or away_team is None:
            print(f"[plus35] Cannot resolve fixture {fix_id}")
            continue

        # Look up +3.5 odds
        plus35_odds = _get_spread_odds(
            pick_team, home_team, away_team, today_str,
            _PLUS35_RUNG, _PLUS35_FALLBACK_ODDS)
        if abs(plus35_odds - _PLUS35_FALLBACK_ODDS) < 0.01:
            n_fallback += 1
        else:
            n_real_odds += 1

        # Odds floor: skip bets where BE > historical WR (edge inverts at short odds)
        if plus35_odds < _PLUS35_MIN_ODDS:
            print(f"[plus35] Skipping {pick_team}: odds {plus35_odds:.3f} < floor {_PLUS35_MIN_ODDS} "
                  f"(BE={1/plus35_odds*100:.1f}% too high)")
            continue

        # Create straight +3.5 bet
        plus35_desc = f"{pick_team} +3.5"
        model_wp = 1.0 / plus35_odds

        mock_id = str(uuid.uuid4())
        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=plus35_odds,
            amount=_PLUS35_STAKE,
            legs=1,
            bet_info=json.dumps({
                "type": "straight_plus35",
                "team": pick_team,
                "rung": 3.5,
                "dist_wp_at_best_rung": round(leg["ats_dist_wp"], 4),
                "negbin_cell": leg["negbin_cell"],
                "plus35_odds": round(plus35_odds, 3),
                "baseline_rung": float(m.group(2)),
            }),
            predicted_win_prob=model_wp,
            model_used="negbin_v1+plus35_straight",
            status="PENDING",
            source=_PLUS35_SOURCE,
            cohort="clean",
            weight=1.0,
        )
        db.add(mock)
        db.flush()

        # Set mean/min dist_wp via raw SQL (columns added by ALTER TABLE, not in ORM model)
        try:
            db.execute(_text(
                "UPDATE mock_bets SET mean_leg_dist_wp=:mdw, min_leg_dist_wp=:mindw "
                "WHERE id=:mid"
            ), {"mdw": round(leg["ats_dist_wp"], 4),
                "mindw": round(leg["ats_dist_wp"], 4), "mid": mock_id})
        except Exception:
            pass

        leg_obj = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=plus35_desc,
            market_type="alternate_spreads",
            sport="MLB",
            win_prob=model_wp,
            fixture_id=fix_id,
            is_alt_line=True,
        )
        db.add(leg_obj)
        db.flush()

        # Set spread_value via raw SQL (column added by ALTER TABLE, not in ORM model)
        try:
            db.execute(_text(
                "UPDATE mock_bet_legs SET spread_value=:sv WHERE id=:lid"
            ), {"sv": 3.5, "lid": leg_obj.id})
        except Exception:
            pass

        # Copy signal columns
        updates = {}
        if leg["negbin_wp"] is not None:
            updates["negbin_wp"] = leg["negbin_wp"]
        if leg["negbin_edge_pp"] is not None:
            updates["negbin_edge_pp"] = leg["negbin_edge_pp"]
        if leg["negbin_mu_home"] is not None:
            updates["negbin_mu_home"] = leg["negbin_mu_home"]
        if leg["negbin_mu_away"] is not None:
            updates["negbin_mu_away"] = leg["negbin_mu_away"]
        if leg["negbin_cell"] is not None:
            updates["negbin_cell"] = leg["negbin_cell"]
        if leg["ats_cover_wp"] is not None:
            updates["ats_cover_wp"] = leg["ats_cover_wp"]
        if leg["ats_dist_wp"] is not None:
            updates["ats_dist_wp"] = leg["ats_dist_wp"]
        if leg["bayes_team_wp"] is not None:
            updates["bayes_team_wp"] = leg["bayes_team_wp"]
        if leg["ridge_composite"] is not None:
            updates["ridge_composite"] = leg["ridge_composite"]
        if updates:
            set_clause = ", ".join(f"{k}=:{k}" for k in updates)
            updates["lid"] = leg_obj.id
            try:
                db.execute(_text(
                    f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                ), updates)
            except Exception:
                pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[plus35] DB commit failed: {e}")
        return []

    print(f"[plus35] Stored {stored} straight +3.5 bets "
          f"({n_real_odds} real odds, {n_fallback} fallback) "
          f"(source={_PLUS35_SOURCE})")

    return list(qualifying.values())


# ═══════════════════════════════════════════════════════════════════════════
# NEG15 + EDGE_PP FILTER — variant arm: exclude high-edge legs
# ═══════════════════════════════════════════════════════════════════════════
_NEG15_EPP_SOURCE = "shadow_negbin_neg15_epp"
_NEG15_EPP_THRESHOLD = 5.0  # PROVISIONAL — failed split-half at N=54 (H1=31.6%/H2=58.8%); revisit at N~120

# Evidence (2026-08-02, N=54 neg15 settled):
#   edge_pp < 5: 44.4% WR, +5.4pp, EV=$+1.09 (N=36) — flips arm to +EV
#   edge_pp >= 5: 22.2% WR, -15.9pp (N=18) — deeply -EV
#   The inversion (rho=-0.127): NegBin overconfident → market right → loses
#   Split-half on dist_wp>=0.85 shadow: 66.3%/66.1% — STABLE


def generate_neg15_epp_filtered(db=None, today_str=None):
    """Straight -1.5 bets with edge_pp filter.

    Same as neg15 but excludes legs where negbin_edge_pp >= _NEG15_EPP_THRESHOLD.
    Deployed as SEPARATE variant arm to forward-test the edge_pp filter
    independently of the plain neg15 arm.

    Source tag: shadow_negbin_neg15_epp
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text
    import re

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[neg15_epp] Generating edge_pp-filtered -1.5 bets for {today_str}")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _NEG15_EPP_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[neg15_epp] Already {existing} bets for {today_str} — skipping")
        return []

    # ── Read Shadow A's legs for today ──
    leg_data = db.execute(_text(
        "SELECT l.fixture_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.ats_cover_wp, l.negbin_wp, l.negbin_mu_home, l.negbin_mu_away, "
        "       l.negbin_cell, l.ats_dist_wp, l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    if not leg_data:
        print("[neg15_epp] No Shadow A legs found for today")
        return []

    # ── De-duplicate by fixture: one entry per game-side, max dist_wp ──
    fixture_legs = {}
    for row in leg_data:
        (fix_id, desc, wp, edge, ats, nb_wp, mu_h, mu_a,
         cell, dist_wp, ridge, bayes) = row
        if dist_wp is None or fix_id is None:
            continue
        if fix_id not in fixture_legs or dist_wp > fixture_legs[fix_id]["ats_dist_wp"]:
            fixture_legs[fix_id] = {
                "fixture_id": fix_id, "description": desc, "win_prob": wp,
                "negbin_edge_pp": edge, "ats_cover_wp": ats, "negbin_wp": nb_wp,
                "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
                "negbin_cell": cell, "ats_dist_wp": dist_wp,
                "ridge_composite": ridge, "bayes_team_wp": bayes,
            }

    # ── Gate 1: dist_wp in [0.85, 0.90) ──
    qualifying = {}
    for fix_id, leg in fixture_legs.items():
        dwp = leg["ats_dist_wp"]
        if _NEG15_DISTWP_LO <= dwp < _NEG15_DISTWP_HI:
            qualifying[fix_id] = leg

    # ── Gate 2: edge_pp < threshold ──
    n_epp_rejected = 0
    epp_qualifying = {}
    for fix_id, leg in qualifying.items():
        epp = leg["negbin_edge_pp"]
        if epp is not None and epp >= _NEG15_EPP_THRESHOLD:
            n_epp_rejected += 1
            continue
        epp_qualifying[fix_id] = leg

    print(f"[neg15_epp] {len(fixture_legs)} fixtures → "
          f"{len(qualifying)} pass dist_wp gate → "
          f"{len(epp_qualifying)} pass edge_pp<{_NEG15_EPP_THRESHOLD} "
          f"({n_epp_rejected} rejected by edge_pp)")

    if not epp_qualifying:
        print("[neg15_epp] No qualifying fixtures")
        return []

    # ── Build straight -1.5 bets ──
    run_id = str(uuid.uuid4())
    stored = 0
    n_real_odds = 0
    n_fallback = 0

    for fix_id, leg in epp_qualifying.items():
        desc = leg["description"]
        m = re.match(r"^(.+?)\s+\+(\d+\.?\d*)$", desc)
        if not m:
            print(f"[neg15_epp] Skipping unparseable: {desc}")
            continue
        pick_team = m.group(1).strip()

        home_team, away_team = _get_fixture_teams(fix_id, db)
        if home_team is None or away_team is None:
            continue

        neg15_odds = _get_spread_odds(
            pick_team, home_team, away_team, today_str,
            _NEG15_RUNG, _NEG15_FALLBACK_ODDS)
        if abs(neg15_odds - _NEG15_FALLBACK_ODDS) < 0.01:
            n_fallback += 1
        else:
            n_real_odds += 1

        if neg15_odds >= _NEG15_MAX_ODDS:
            print(f"[neg15_epp] Skipping {pick_team}: odds {neg15_odds:.3f} >= ceiling {_NEG15_MAX_ODDS}")
            continue

        neg15_desc = f"{pick_team} -1.5"
        model_wp = 1.0 / neg15_odds

        mock_id = str(uuid.uuid4())
        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=neg15_odds,
            amount=_NEG15_STAKE,
            legs=1,
            bet_info=json.dumps({
                "type": "straight_neg15_epp",
                "team": pick_team,
                "rung": -1.5,
                "dist_wp_at_best_rung": round(leg["ats_dist_wp"], 4),
                "negbin_cell": leg["negbin_cell"],
                "neg15_odds": round(neg15_odds, 3),
                "edge_pp_filter": _NEG15_EPP_THRESHOLD,
                "edge_pp_value": round(leg["negbin_edge_pp"], 2) if leg["negbin_edge_pp"] is not None else None,
            }),
            predicted_win_prob=model_wp,
            model_used="negbin_v1+neg15_epp_filtered",
            status="PENDING",
            source=_NEG15_EPP_SOURCE,
            cohort="clean",
            weight=1.0,
        )
        db.add(mock)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bets SET mean_leg_dist_wp=:mdw, min_leg_dist_wp=:mindw "
                "WHERE id=:mid"
            ), {"mdw": round(leg["ats_dist_wp"], 4),
                "mindw": round(leg["ats_dist_wp"], 4), "mid": mock_id})
        except Exception:
            pass

        leg_obj = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=neg15_desc,
            market_type="alternate_spreads",
            sport="MLB",
            win_prob=model_wp,
            fixture_id=fix_id,
            is_alt_line=True,
        )
        db.add(leg_obj)
        db.flush()

        try:
            db.execute(_text(
                "UPDATE mock_bet_legs SET spread_value=:sv WHERE id=:lid"
            ), {"sv": -1.5, "lid": leg_obj.id})
        except Exception:
            pass

        # Copy signal columns
        updates = {}
        for col in ("negbin_wp", "negbin_edge_pp", "negbin_mu_home", "negbin_mu_away",
                     "negbin_cell", "ats_cover_wp", "ats_dist_wp", "bayes_team_wp",
                     "ridge_composite"):
            if leg[col] is not None:
                updates[col] = leg[col]
        if updates:
            set_clause = ", ".join(f"{k}=:{k}" for k in updates)
            updates["lid"] = leg_obj.id
            try:
                db.execute(_text(
                    f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                ), updates)
            except Exception:
                pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[neg15_epp] DB commit failed: {e}")
        return []

    print(f"[neg15_epp] Stored {stored} edge_pp-filtered -1.5 bets "
          f"({n_real_odds} real odds, {n_fallback} fallback) "
          f"(source={_NEG15_EPP_SOURCE})")

    return list(epp_qualifying.values())


# ═══════════════════════════════════════════════════════════════════════════
# EV-AWARE RUNG SELECTION — variant parlay arm: pick the +EV rung per fixture
# ═══════════════════════════════════════════════════════════════════════════
_EVAWARE_SOURCE = "shadow_negbin_evaware"

# EV-optimal rung from cell×rung grid (2026-08-02, N=4852):
#   +1.5 is the ONLY +EV rung at aggregate (+2.9pp, EV=$+0.35)
#   +2.5 breakeven, +3.5/-EV, +4.5 deeply -EV (-8.7pp)
#   max-dist_wp funnels to +4.5 (worst): 84% of selections are +4.5
# Fix: for each fixture-team, pick the LOWEST rung Shadow A generated
# dist_wp floor 0.70 — first +EV point in backtest (edge=+3.9pp, EV=$+0.37,
# split-half stable H1=+$0.34/H2=+$0.39 on N=244). Without floor, low-dist_wp
# +1.5 legs dilute edge (correct side but terrible odds → EV=-$0.35).
# dist_wp varies by rung (+1.5 dist_wp ~0.65, +4.5 dist_wp ~0.88 for same team)
_EVAWARE_DISTWP_FLOOR = 0.70


def generate_evaware_parlay(db=None, today_str=None):
    """EV-aware rung selection: for each leg, prefer shortest rung (+1.5/+2.5).

    Reads Shadow A's legs, re-selects per fixture-team: pick the LOWEST
    available rung instead of max-dist_wp (which funnels to -EV +4.5).

    dist_wp floor 0.70 applied after min-rung selection — first +EV point in
    historical backtest (N=244, edge=+3.9pp, split-half stable). Without floor,
    low-dist_wp +1.5 legs are correct-side but terrible-odds → EV negative.

    Variant arm to compare against max-dist_wp (current) selection.
    Source tag: shadow_negbin_evaware
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text
    import re

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[evaware] Generating EV-aware rung selection parlays for {today_str}")

    # ── Dedup ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _EVAWARE_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[evaware] Already {existing} parlays for {today_str} — skipping")
        return []

    # ── Read ALL Shadow A legs ──
    leg_data = db.execute(_text(
        "SELECT l.mock_bet_id, l.fixture_id, l.description, l.win_prob, "
        "       l.negbin_edge_pp, l.ats_cover_wp, l.negbin_wp, "
        "       l.negbin_mu_home, l.negbin_mu_away, l.negbin_cell, l.ats_dist_wp, "
        "       l.ridge_composite, l.bayes_team_wp, l.scout_grade "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    if not leg_data:
        print("[evaware] No Shadow A legs found for today")
        return []

    # ── Group by fixture-team, pick LOWEST rung per fixture-team ──
    fixture_team_legs = defaultdict(list)
    for row in leg_data:
        (mid, fix_id, desc, wp, edge, ats, nb_wp, mu_h, mu_a,
         cell, dist_wp, ridge, bayes, scout) = row
        if not desc:
            continue
        m = re.match(r"^(.+?)\s+\+(\d+\.?\d*)$", desc)
        if not m:
            continue
        team = m.group(1).strip()
        rung = float(m.group(2))
        key = f"{fix_id}_{team}"
        fixture_team_legs[key].append({
            "mock_bet_id": mid, "fixture_id": fix_id, "description": desc,
            "win_prob": wp, "negbin_edge_pp": edge, "ats_cover_wp": ats,
            "negbin_wp": nb_wp, "negbin_mu_home": mu_h, "negbin_mu_away": mu_a,
            "negbin_cell": cell, "ats_dist_wp": dist_wp or 0.5,
            "ridge_composite": ridge, "bayes_team_wp": bayes,
            "scout_grade": scout, "team": team, "rung": rung,
        })

    # For each fixture-team, pick the LOWEST available rung
    selected_legs = []
    for key, legs in fixture_team_legs.items():
        legs.sort(key=lambda x: x["rung"])
        selected_legs.append(legs[0])

    pre_floor = len(selected_legs)

    # Apply dist_wp floor — removes marginal low-dist_wp legs
    # (low-rung legs at dist_wp < 0.70 are correct-side but terrible-odds → -EV)
    selected_legs = [l for l in selected_legs
                     if l["ats_dist_wp"] >= _EVAWARE_DISTWP_FLOOR]

    # epp >= 12 toxicity filter
    pre_epp = len(selected_legs)
    selected_legs = [l for l in selected_legs
                     if l.get("negbin_edge_pp") is None
                     or l["negbin_edge_pp"] < _EDGE_GATE_THRESHOLD_PP]

    print(f"[evaware] {len(fixture_team_legs)} fixture-teams → "
          f"{pre_floor} min-rung → {pre_epp} pass dist_wp≥{_EVAWARE_DISTWP_FLOOR} "
          f"→ {len(selected_legs)} pass epp<{_EDGE_GATE_THRESHOLD_PP}")

    rung_dist = defaultdict(int)
    for leg in selected_legs:
        rung_dist[leg["rung"]] += 1
    rung_str = ', '.join(f"+{r:.1f}:{c}" for r, c in sorted(rung_dist.items()))
    print(f"[evaware] Rung distribution: {rung_str}")

    if len(selected_legs) < 2:
        print("[evaware] Not enough legs for parlays")
        return []

    # ── Look up actual odds for each selected leg ──
    import itertools

    pool = list(selected_legs)
    pool.sort(key=lambda x: -(x["win_prob"] or 0.5))
    pool_cap = min(len(pool), 50)

    for c in pool:
        c["leg_id"] = f"{c['fixture_id']}:{c['team']}:{c['rung']}"

    for leg in pool[:pool_cap]:
        fix_id = leg["fixture_id"]
        team = leg["team"]
        rung = leg["rung"]
        home_team, away_team = _get_fixture_teams(fix_id, db)
        if home_team and away_team:
            fallback = {1.5: 1.885, 2.5: 1.667, 3.5: 1.400, 4.5: 1.250}.get(rung, 1.500)
            odds = _get_spread_odds(team, home_team, away_team, today_str, rung, fallback)
            leg["decimal_odds"] = odds
        else:
            leg["decimal_odds"] = None

    pool = [c for c in pool[:pool_cap] if c.get("decimal_odds") and c["decimal_odds"] > 1.0]
    print(f"[evaware] {len(pool)} legs with valid odds")

    # ── Assemble parlays: 2-3 leg combos ──
    _MAX_LEG_APP = 2

    all_scored = []
    for size in [2, 3]:
        if len(pool) < size:
            continue
        combos = list(itertools.combinations(pool[:pool_cap], size))
        for combo in combos:
            fixtures_used = set(c["fixture_id"] for c in combo)
            if len(fixtures_used) < len(combo):
                continue
            combined_odds = 1.0
            for c in combo:
                combined_odds *= c["decimal_odds"]
            avg_wp = np.mean([c.get("ats_dist_wp") or 0.5 for c in combo])
            combined_wp = 1.0
            for c in combo:
                combined_wp *= (c.get("ats_dist_wp") or 0.5)
            norm_ev = max(0, combined_wp * (combined_odds - 1) - (1 - combined_wp))
            composite = avg_wp * 0.40 + 0.10 + 0.15 + min(norm_ev, 1.0) * 0.10
            band = "other"
            for band_label, lo, hi in ODDS_BANDS:
                if lo <= combined_odds < hi:
                    band = band_label
                    break
            all_scored.append({
                "legs": list(combo),
                "combined_odds": combined_odds,
                "composite": composite,
                "combined_wp": combined_wp,
                "odds_band": band,
                "leg_count": len(combo),
            })

    # Top-N selection with leg reuse caps (no per-band bucketing)
    all_scored.sort(key=lambda x: -x["composite"])
    parlays = []
    seen = set()
    leg_use = defaultdict(int)
    _MAX_PARLAYS = 15

    for combo in all_scored:
        if len(parlays) >= _MAX_PARLAYS:
            break
        leg_ids = tuple(sorted(c["leg_id"] for c in combo["legs"]))
        if leg_ids in seen:
            continue
        if any(leg_use[c["leg_id"]] >= _MAX_LEG_APP for c in combo["legs"]):
            continue
        seen.add(leg_ids)
        for c in combo["legs"]:
            leg_use[c["leg_id"]] += 1
        parlays.append(combo)

    print(f"[evaware] Assembled {len(parlays)} parlays from {len(all_scored)} scored combos")

    if not parlays:
        return []

    # ── Store ──
    run_id = str(uuid.uuid4())
    stored = 0
    for parlay in parlays:
        legs = parlay["legs"]
        combined_odds = parlay["combined_odds"]

        bet_info = json.dumps({
            "type": "evaware_parlay",
            "rung_selection": "min_rung",
            "dist_wp_floor": _EVAWARE_DISTWP_FLOOR,
            "legs": [{"team": l["team"], "rung": l["rung"],
                      "dist_wp": round(l["ats_dist_wp"], 4)} for l in legs],
        })

        mock_id = str(uuid.uuid4())
        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if len(legs) > 1 else "straight",
            odds=round(combined_odds, 3),
            amount=10.0,
            legs=len(legs),
            bet_info=bet_info,
            predicted_win_prob=round(parlay["combined_wp"], 6),
            model_used="negbin_v1+evaware_rung",
            status="PENDING",
            source=_EVAWARE_SOURCE,
            cohort="clean",
            weight=1.0,
        )
        db.add(mock)
        db.flush()

        # Set mean/min dist_wp
        leg_dps = [l["ats_dist_wp"] for l in legs if l["ats_dist_wp"]]
        if leg_dps:
            try:
                db.execute(_text(
                    "UPDATE mock_bets SET mean_leg_dist_wp=:mdw, min_leg_dist_wp=:mindw "
                    "WHERE id=:mid"
                ), {"mdw": round(np.mean(leg_dps), 4),
                    "mindw": round(min(leg_dps), 4), "mid": mock_id})
            except Exception:
                pass

        for i, leg in enumerate(legs):
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=i,
                description=leg["description"],
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg.get("win_prob") or (1.0 / leg["decimal_odds"]),
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
            )
            db.add(leg_obj)
            db.flush()

            try:
                db.execute(_text(
                    "UPDATE mock_bet_legs SET spread_value=:sv WHERE id=:lid"
                ), {"sv": leg["rung"], "lid": leg_obj.id})
            except Exception:
                pass

            # Copy signal columns
            updates = {}
            for col in ("negbin_wp", "negbin_edge_pp", "negbin_mu_home", "negbin_mu_away",
                         "negbin_cell", "ats_cover_wp", "ats_dist_wp", "bayes_team_wp",
                         "ridge_composite"):
                if leg.get(col) is not None:
                    updates[col] = leg[col]
            if leg.get("scout_grade") is not None:
                updates["scout_grade"] = leg["scout_grade"]
            if updates:
                set_clause = ", ".join(f"{k}=:{k}" for k in updates)
                updates["lid"] = leg_obj.id
                try:
                    db.execute(_text(
                        f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                    ), updates)
                except Exception:
                    pass

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[evaware] DB commit failed: {e}")
        return []

    print(f"[evaware] Stored {stored} EV-aware parlays (source={_EVAWARE_SOURCE})")
    return parlays


def _report_fullstack_tracker(db):
    """Full-stack tracker: combined P&L, Kelly sizing quality, layer contributions.

    Reports:
    - Daily P&L: fullstack vs Shadow A vs edge-gate-only
    - dist_wp quintile cover rates (forward confirmation of discrimination)
    - Kelly sizing quality (are higher-weighted parlays winning more?)
    - Locked validation standard (deflated SR, N=1 trial)
    """
    from sqlalchemy import text as _text

    # ── Get settled data ──
    fs_rows = db.execute(_text(
        "SELECT game_date, status, odds, amount, weight "
        "FROM mock_bets WHERE source=:src AND cohort='clean' "
        "  AND status IN ('SETTLED_WIN','SETTLED_LOSS')"
    ), {"src": _FULLSTACK_SOURCE}).fetchall()

    if not fs_rows:
        print("[fullstack] No settled data yet — tracker starts after first settlement")
        return

    a_rows = db.execute(_text(
        "SELECT game_date, status, odds "
        "FROM mock_bets WHERE source='shadow_negbin' AND cohort='clean' "
        "  AND status IN ('SETTLED_WIN','SETTLED_LOSS')"
    )).fetchall()

    gate_rows = db.execute(_text(
        "SELECT game_date, status, odds, amount "
        "FROM mock_bets WHERE source='shadow_negbin_edgegate' AND cohort='clean' "
        "  AND status IN ('SETTLED_WIN','SETTLED_LOSS')"
    )).fetchall()

    # Per-day PnL
    a_day = defaultdict(float)
    gate_day = defaultdict(float)
    fs_day = defaultdict(float)
    fs_n_day = defaultdict(int)

    for gd, st, odds in a_rows:
        a_day[gd] += (odds - 1) * 10.0 if st == 'SETTLED_WIN' else -10.0
    for gd, st, odds, amt in gate_rows:
        gate_day[gd] += (odds - 1) * amt if st == 'SETTLED_WIN' else -amt
    for gd, st, odds, amt, wt in fs_rows:
        fs_day[gd] += (odds - 1) * amt if st == 'SETTLED_WIN' else -amt
        fs_n_day[gd] += 1

    overlap = sorted(set(fs_day.keys()))
    if len(overlap) < 2:
        print(f"[fullstack] Only {len(overlap)} settled day(s) — need ≥2")
        return

    # ── Per-day report ──
    print(f"\n[fullstack] Forward tracker ({len(overlap)} settled days):")
    print(f"  {'Day':<12} {'FS_N':>4} {'ShadA':>9} {'EGate':>9} {'FullSt':>9} "
          f"{'vs_A':>7} {'vs_Gate':>7}")
    for d in overlap:
        a_pnl = a_day.get(d, 0)
        g_pnl = gate_day.get(d, 0)
        f_pnl = fs_day[d]
        print(f"  {d:<12} {fs_n_day[d]:4d} "
              f"{a_pnl:+9.2f} {g_pnl:+9.2f} {f_pnl:+9.2f} "
              f"{f_pnl - a_pnl:+7.2f} {f_pnl - g_pnl:+7.2f}")

    # ── Summary ──
    total_fs = sum(fs_day[d] for d in overlap)
    total_a = sum(a_day.get(d, 0) for d in overlap)
    total_staked = sum(amt for gd, st, odds, amt, wt in fs_rows)
    fs_roi = total_fs / total_staked * 100 if total_staked > 0 else 0

    print(f"\n  Full-stack P&L: ${total_fs:+.2f} "
          f"(staked ${total_staked:.2f}, ROI={fs_roi:+.1f}%)")
    print(f"  Shadow A P&L:   ${total_a:+.2f} "
          f"(equal $10, ROI={total_a / (sum(1 for _ in a_rows) * 10) * 100:+.1f}%)")
    print(f"  vs Shadow A:    ${total_fs - total_a:+.2f}")

    # ── Kelly sizing quality: weight quintile → WR ──
    print(f"\n  ── Kelly sizing quality (weight quintile → WR) ──")
    weight_data = [(wt, 1 if st == 'SETTLED_WIN' else 0)
                   for gd, st, odds, amt, wt in fs_rows if wt is not None]
    if len(weight_data) >= 20:
        weight_data.sort(key=lambda x: x[0])
        n_w = len(weight_data)
        q_sz = n_w // 5
        for qi in range(5):
            s = qi * q_sz
            e = s + q_sz if qi < 4 else n_w
            q = weight_data[s:e]
            avg_w = np.mean([x[0] for x in q])
            wr = np.mean([x[1] for x in q])
            print(f"    Q{qi+1}: avg_weight={avg_w:.3f}  WR={wr:.3f}  N={len(q)}")

        # Spearman: weight → outcome
        from scipy.stats import spearmanr
        rho, p = spearmanr([x[0] for x in weight_data], [x[1] for x in weight_data])
        print(f"    weight → outcome: rho={rho:+.3f} (p={p:.4f})")
    else:
        print(f"    Only {len(weight_data)} settled parlays — need ≥20 for quintiles")

    # ── Deflated SR (locked standard, N=1 trial for the combined stack) ──
    T = len(overlap)
    returns = [fs_day[d] for d in overlap]
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if T > 1 else 1.0
    daily_sr = mean_r / std_r if std_r > 0 else 0

    N_TRIALS = 1  # single combined strategy, no threshold search
    E_max_sr = np.sqrt(2 * np.log(max(N_TRIALS, 1))) / np.sqrt(T) if N_TRIALS > 1 else 0
    deflated_sr = daily_sr - E_max_sr

    try:
        from scipy.stats import norm
        se_sr = 1.0 / np.sqrt(T)
        z = deflated_sr / se_sr if se_sr > 0 else 0
        p_dsr = 1 - norm.cdf(z)
    except ImportError:
        p_dsr = None

    verdict = ("CONFIRMED" if deflated_sr > 0 and (p_dsr or 1) < 0.05
               else "DIRECTIONAL" if daily_sr > 0
               else "COLLAPSED")

    print(f"\n  ── Locked Validation Standard ──")
    print(f"  Daily SR:      {daily_sr:.4f}")
    print(f"  Deflated SR:   {deflated_sr:+.4f} (N={N_TRIALS} trial)")
    if p_dsr is not None:
        print(f"  p-value:       {p_dsr:.4f}")
    print(f"  T:             {T} days")
    print(f"  Verdict:       {verdict}")

    # ── Bad-period survival ──
    bad_days = [d for d in overlap if a_day.get(d, 0) < -50]
    if bad_days:
        bad_a = sum(a_day.get(d, 0) for d in bad_days)
        bad_fs = sum(fs_day[d] for d in bad_days)
        print(f"  Bad-period ({len(bad_days)} days): A=${bad_a:+.0f}, "
              f"FullStack=${bad_fs:+.0f}, saved=${bad_fs - bad_a:+.0f}")


def _report_atscap_asymmetry(db):
    """Report continuous gap-sizing asymmetry: drawdown cut vs upside cost.

    Compares Shadow A (equal $10 sizing) vs ATSCAP (gap-weighted sizing)
    on settled days. The cap graduates only if drawdown reduction on bad
    days exceeds upside cost on good days (asymmetry ratio > 1.5x).
    """
    from sqlalchemy import text as _text

    # Get settled days for ATSCAP (which has sized amounts)
    cap_rows = db.execute(_text(
        "SELECT game_date, status, odds, amount "
        "FROM mock_bets WHERE source='shadow_negbin_atscap' AND cohort='clean' "
        "  AND status IN ('SETTLED_WIN','SETTLED_LOSS')"
    )).fetchall()
    a_rows = db.execute(_text(
        "SELECT game_date, status, odds "
        "FROM mock_bets WHERE source='shadow_negbin' AND cohort='clean' "
        "  AND status IN ('SETTLED_WIN','SETTLED_LOSS')"
    )).fetchall()

    if not cap_rows or not a_rows:
        print("[atscap] No settled data yet — asymmetry tracking starts after settlement")
        return

    # Per-day PnL
    a_day = defaultdict(float)
    cap_day = defaultdict(float)
    for gd, st, odds in a_rows:
        a_day[gd] += (odds - 1) * 10.0 if st == 'SETTLED_WIN' else -10.0
    for gd, st, odds, amt in cap_rows:
        cap_day[gd] += (odds - 1) * amt if st == 'SETTLED_WIN' else -amt

    overlap = sorted(set(a_day) & set(cap_day))
    if not overlap:
        return

    bad_saved = 0.0
    good_cost = 0.0
    print(f"\n[atscap] Asymmetry tracker ({len(overlap)} settled days):")
    print(f"  {'Day':<12} {'A_PnL':>9} {'Cap_PnL':>9} {'Δ':>9} {'Type':>5}")
    for d in overlap:
        delta = cap_day[d] - a_day[d]
        dtype = "BAD" if a_day[d] < -50 else ("GOOD" if a_day[d] > 50 else "FLAT")
        print(f"  {d:<12} {a_day[d]:+9.2f} {cap_day[d]:+9.2f} {delta:+9.2f} {dtype:>5}")
        if dtype == "BAD":
            bad_saved += delta
        elif dtype == "GOOD":
            good_cost += delta

    print(f"\n  Bad-day drawdown saved: {bad_saved:+.2f}")
    print(f"  Good-day upside cost:   {good_cost:+.2f}")
    if bad_saved > 0 and good_cost < 0:
        ratio = bad_saved / abs(good_cost)
        print(f"  Asymmetry ratio: {ratio:.1f}x (>1.5 = graduate)")
    elif bad_saved > 0 and good_cost >= 0:
        print(f"  Asymmetry ratio: inf (saves on bad, no cost on good)")
    elif bad_saved <= 0:
        print(f"  No drawdown saved — cap not helping")

    # ── Fixture-level gradient monitor (fast confirmation) ──
    _report_fixture_gradient(db)


def _report_fixture_gradient(db):
    """Monitor ATS gap→bust gradient at fixture level.

    Faster confirmation than parlay P&L: ~9 fixtures/day, each independent.
    Reports Spearman rho and bust rates by ATS bin, split by discovery window.
    Discovery cutoff: 2026-07-06 (signal established 2026-07-07 from 6/28-7/5 data).
    """
    from sqlalchemy import text as _text

    rows = db.execute(_text(
        "SELECT b.game_date, l.fixture_id, l.ats_cover_wp, l.leg_result "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.cohort='clean' "
        "  AND l.ats_cover_wp IS NOT NULL "
        "  AND l.description LIKE '%+1.5%' "
        "  AND l.leg_result IN ('WIN','LOSS')"
    )).fetchall()

    if not rows:
        return

    # Deduplicate to fixture level
    fixtures = {}
    for gd, fix_id, ats, result in rows:
        key = (gd, fix_id)
        if key not in fixtures:
            fixtures[key] = {"date": gd, "ats": ats, "bust": result == "LOSS"}

    oos = {k: v for k, v in fixtures.items() if k[0] >= "2026-07-06"}
    if len(oos) < 5:
        return

    ats_vals = [f["ats"] for f in oos.values()]
    bust_vals = [1 if f["bust"] else 0 for f in oos.values()]

    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(ats_vals, bust_vals)
    except Exception:
        return

    # Bust rates by bin
    bins = [("<60%", lambda a: a < 0.60), ("≥60%", lambda a: a >= 0.60)]
    print(f"\n  [fixture-gradient] OOS (7/6+): N={len(oos)}, "
          f"Spearman rho={rho:.3f}, p={p:.4f}")
    for label, filt in bins:
        subset = [f for f in oos.values() if filt(f["ats"])]
        if subset:
            br = sum(1 for f in subset if f["bust"]) / len(subset)
            print(f"    ATS {label}: {len(subset)} fixtures, bust={br*100:.0f}%")


def _store_shadow_parlays(db, parlays, today_str, source_tag="shadow_negbin"):
    """Persist shadow parlays to mock_bets + mock_bet_legs."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    _ensure_shadow_columns()

    run_id = str(uuid.uuid4())
    stored = 0

    for p in parlays:
        mock_id = str(uuid.uuid4())
        legs = p["legs"]

        bet_info = " | ".join(
            f"{l['pick_team']} +{l['spread_value']}" for l in legs
        )

        # Check for existing shadow with same bet_info today
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd AND source=:src "
            "AND bet_info=:bi"
        ), {"gd": today_str, "src": source_tag, "bi": bet_info}).scalar()
        if existing > 0:
            continue

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="parlay" if len(legs) > 1 else "straight",
            odds=p["combined_odds"],
            amount=0.0,         # shadow — $0 stake (not live)
            legs=len(legs),
            bet_info=bet_info,
            predicted_win_prob=p["combined_wp"],
            model_used="negbin_v1",
            status="PENDING",
            source=source_tag,
            composite_score=p["composite"],
            cohort="clean",
        )
        db.add(mock)
        db.flush()

        # Write odds band + cell adj + leg count + live_realistic via raw SQL
        try:
            db.execute(_text(
                "UPDATE mock_bets SET negbin_odds_band=:band, negbin_cell_adj=:adj, "
                "negbin_leg_count=:lc, live_realistic=:lr WHERE id=:mid"
            ), {"band": p["odds_band"], "adj": p["cell_adj"],
                "lc": len(legs), "lr": p.get("live_realistic", 0),
                "mid": mock_id})
        except Exception:
            pass

        leg_dist_wps = []  # collect per-leg dist_wp for parlay-level stats

        for i, leg in enumerate(legs):
            leg_obj = MockBetLeg(
                mock_bet_id=mock_id,
                leg_index=i,
                description=f"{leg['pick_team']} +{leg['spread_value']}",
                market_type="alternate_spreads",
                sport="MLB",
                win_prob=leg["model_wp"],
                fixture_id=leg["fixture_id"],
                is_alt_line=True,
                predicted_win_prob=leg["model_wp"],
                predicted_edge_pp=leg["edge_pp"],
            )
            db.add(leg_obj)
            db.flush()

            # Write NegBin-specific columns
            try:
                db.execute(_text(
                    "UPDATE mock_bet_legs SET negbin_wp=:wp, negbin_mu_home=:mh, "
                    "negbin_mu_away=:ma, negbin_cell=:cell, negbin_edge_pp=:edge "
                    "WHERE id=:lid"
                ), {
                    "wp": leg["model_wp"], "mh": leg["mu_home"], "ma": leg["mu_away"],
                    "cell": leg["cell"], "edge": leg["edge_pp"], "lid": leg_obj.id,
                })
            except Exception:
                pass

            # ATS cross-estimator score (only +1.5 rung — same question as ATS binary)
            if leg.get("spread_value") == 1.5:
                try:
                    from ml_model import predict_game_ats
                    ht = leg.get("home_team", "")
                    at = leg.get("away_team", "")
                    gd = leg.get("game_date", today_str)
                    ats_raw = predict_game_ats("MLB", ht, at, gd, spread_value=-1.5)
                    if ats_raw is not None:
                        # ATS returns P(home covers -1.5). Convert to picked side.
                        if leg.get("pick_side") == "home":
                            # Home at +1.5 (underdog): call with spread=+1.5
                            ats_raw2 = predict_game_ats("MLB", ht, at, gd, spread_value=1.5)
                            ats_cover = ats_raw2 if ats_raw2 is not None else ats_raw
                        else:
                            # Away at +1.5: P(away covers +1.5) = 1 - P(home covers -1.5)
                            ats_cover = 1.0 - ats_raw
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET ats_cover_wp=:ats WHERE id=:lid"
                        ), {"ats": round(ats_cover, 4), "lid": leg_obj.id})
                except Exception as exc:
                    print(f"[negbin-shadow] ATS score error: {exc}")

            # ATS distributional score (all rungs — margin regression model)
            dist_wp = None  # used by market disagree scorer below
            try:
                from ml_model import predict_game_ats_distributional, ats_distributional_cover_prob
                ht = leg.get("home_team", "")
                at = leg.get("away_team", "")
                gd = leg.get("game_date", today_str)
                sv = leg.get("spread_value")
                ps = leg.get("pick_side", "away")
                if sv is not None:
                    mu = predict_game_ats_distributional(ht, at, gd)
                    if mu is not None:
                        dist_wp = ats_distributional_cover_prob(mu, float(sv), ps)
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET ats_dist_wp=:dwp WHERE id=:lid"
                        ), {"dwp": round(dist_wp, 4), "lid": leg_obj.id})
                        leg_dist_wps.append(dist_wp)
            except Exception as exc:
                print(f"[negbin-shadow] ATS distributional score error: {exc}")

            # ── Bayesian team shrunk cover rate ──
            # James-Stein shrinkage of picked-team historical cover rate toward
            # grand mean.  Provides genuine discrimination (OOS rho=+0.435,
            # quintile Q0=45% → Q4=98%, monotonic rho=+0.422).
            bwp = None  # used by ridge composite scorer below
            try:
                picked_team_name = leg.get("pick_team", "")
                if picked_team_name and _BAYES_TEAM_CACHE is not None:
                    bwp = _BAYES_TEAM_CACHE.get(picked_team_name)
                    if bwp is not None:
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET bayes_team_wp=:bwp WHERE id=:lid"
                        ), {"bwp": round(bwp, 4), "lid": leg_obj.id})
            except Exception as exc:
                print(f"[negbin-shadow] Bayes team WP score error: {exc}")

            # ── Market disagreement with dist_wp ──
            # mkt_disagree_dist = market_implied_prob - dist_wp.
            # Requires alt_lines data for this fixture + a valid dist_wp.
            try:
                if dist_wp is not None and leg.get("fixture_id"):
                    _mkt_prob = _lookup_market_implied(
                        leg["fixture_id"], leg.get("pick_team", ""),
                        leg.get("spread_value"))
                    if _mkt_prob is not None:
                        mdd = _mkt_prob - dist_wp
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET mkt_disagree_dist=:mdd WHERE id=:lid"
                        ), {"mdd": round(mdd, 4), "lid": leg_obj.id})
            except Exception as exc:
                print(f"[negbin-shadow] Market disagree score error: {exc}")

            # ── Ridge composite (bayes_team_wp + ats_dist_wp, C=0.005) ──
            # Fitted probability from ridge logistic. Tracked column only.
            try:
                if bwp is not None and dist_wp is not None:
                    rc = _ridge_predict(bwp, dist_wp)
                    if rc is not None:
                        db.execute(_text(
                            "UPDATE mock_bet_legs SET ridge_composite=:rc WHERE id=:lid"
                        ), {"rc": round(rc, 4), "lid": leg_obj.id})
            except Exception as exc:
                print(f"[negbin-shadow] Ridge composite score error: {exc}")

        # ── Parlay-level dist_wp stats (mean + min + concentrated-risk flag) ──
        if leg_dist_wps:
            _mean_dw = sum(leg_dist_wps) / len(leg_dist_wps)
            _min_dw = min(leg_dist_wps)
            _max_dw = max(leg_dist_wps)
            _conc_risk = 1 if (_max_dw - _min_dw) > 0.15 else 0
            try:
                db.execute(_text(
                    "UPDATE mock_bets SET mean_leg_dist_wp=:mdw, min_leg_dist_wp=:mindw, "
                    "concentrated_risk=:cr WHERE id=:mid"
                ), {"mdw": round(_mean_dw, 4), "mindw": round(_min_dw, 4),
                    "cr": _conc_risk, "mid": mock_id})
            except Exception as exc:
                print(f"[negbin-shadow] mean/min dist_wp store error: {exc}")

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[negbin-shadow] DB commit failed: {e}")
        return

    print(f"[negbin-shadow] Stored {stored} shadow parlays (source={source_tag})")

    # ── Post-dispatch validation: verify signal scoring populated ──
    # Catches silent failures like the 2026-07-18 bayes cache lock issue.
    if stored > 0 and source_tag == "shadow_negbin":
        try:
            check = db.execute(_text(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN bayes_team_wp IS NOT NULL THEN 1 ELSE 0 END) as has_bayes, "
                "SUM(CASE WHEN ats_dist_wp IS NOT NULL THEN 1 ELSE 0 END) as has_dwp "
                "FROM mock_bet_legs WHERE mock_bet_id IN "
                "(SELECT id FROM mock_bets WHERE source='shadow_negbin' AND game_date=:gd)"
            ), {"gd": today_str}).fetchone()
            total, has_bayes, has_dwp = check
            if total > 0 and has_bayes == 0:
                print(f"CRITICAL [negbin-shadow] Post-dispatch validation FAILED: "
                      f"0/{total} legs have bayes_team_wp — cache likely failed to build. "
                      f"Ridge composite will also be NULL.")
            elif total > 0 and has_dwp == 0:
                print(f"CRITICAL [negbin-shadow] Post-dispatch validation FAILED: "
                      f"0/{total} legs have ats_dist_wp — distributional model error.")
            else:
                print(f"[negbin-shadow] Post-dispatch OK: {has_bayes}/{total} bayes, "
                      f"{has_dwp}/{total} dist_wp")
        except Exception:
            pass  # validation check itself shouldn't block the pipeline


# ═══════════════════════════════════════════════════════════════
# FORWARD TRACKING: band-for-band edge report
# ═══════════════════════════════════════════════════════════════

def report_forward_tracking(days_back: int = 30):
    """Report band-for-band edge on settled shadow picks.

    Compares NegBin shadow (source=shadow_negbin) vs live ATS (source=prospective*)
    on settled results, broken down by odds band AND leg count.

    IMPORTANT: fixture reuse across parlays means outcomes are correlated.
    Multiple shadow parlays sharing a game outcome are NOT independent bets.
    Band ROI variance is understated — treat N as effective-N < raw-N.
    """
    con = sqlite3.connect(_BETS_DB, timeout=15)

    # NegBin shadow parlays — include leg count and descriptions for reuse tracking
    nb_rows = con.execute("""
        SELECT mb.id, mb.odds, mb.negbin_odds_band, mb.status, mb.game_date,
               mb.legs, mb.negbin_leg_count,
               GROUP_CONCAT(mbl.leg_result, ',') as leg_results,
               GROUP_CONCAT(mbl.fixture_id, ',') as fixture_ids,
               GROUP_CONCAT(mbl.description, '||') as leg_descs
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.source = 'shadow_negbin'
          AND mb.game_date >= date('now', ? || ' days')
        GROUP BY mb.id
    """, (f"-{days_back}",)).fetchall()

    # ATS live parlays (MLB only)
    ats_rows = con.execute("""
        SELECT mb.id, mb.odds, mb.status, mb.game_date, mb.legs,
               GROUP_CONCAT(mbl.leg_result, ',') as leg_results
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.source IN ('prospective', 'prospective_pm')
          AND mb.sport = 'MLB'
          AND mb.game_date >= date('now', ? || ' days')
        GROUP BY mb.id
    """, (f"-{days_back}",)).fetchall()
    con.close()

    def _classify_band(odds):
        if odds is None:
            return "other"
        for label, lo, hi in ODDS_BANDS:
            if lo <= odds < hi:
                return label
        return "other"

    def _is_won(leg_results_str):
        if not leg_results_str:
            return None
        results = leg_results_str.split(",")
        if any(r.strip() == "" or r.strip() not in ("WIN", "LOSS") for r in results):
            return None
        return all(r.strip() == "WIN" for r in results)

    def _empty_bucket():
        return {"n": 0, "wins": 0, "pnl": 0.0, "odds_sum": 0.0}

    # Process NegBin — keyed by (band, leg_count)
    nb_by_band = defaultdict(_empty_bucket)
    nb_by_band_legs = defaultdict(lambda: defaultdict(_empty_bucket))
    # Track leg-level reuse (leg_desc = "Team +rung", unique per fixture+team+rung)
    leg_appearances = defaultdict(int)    # leg_desc -> total appearances across parlays
    band_leg_apps = defaultdict(lambda: defaultdict(int))  # band -> leg_desc -> count
    fixtures_per_date = defaultdict(lambda: defaultdict(int))  # date -> fixture_id -> count

    for mid, odds, band, status, gd, legs_col, neg_lc, lr, fix_ids, leg_descs in nb_rows:
        if band is None:
            band = _classify_band(odds)
        lc = neg_lc or legs_col or (len(lr.split(",")) if lr else 0)
        won = _is_won(lr)
        if won is None:
            continue

        # Track fixture and leg-level reuse
        if fix_ids:
            for fid in fix_ids.split(","):
                fid = fid.strip()
                if fid:
                    fixtures_per_date[gd][fid] += 1
        if leg_descs:
            for desc in leg_descs.split("||"):
                desc = desc.strip()
                if desc:
                    leg_key = f"{gd}:{desc}"  # date-scoped leg identity
                    leg_appearances[leg_key] += 1
                    band_leg_apps[band][leg_key] += 1

        for bucket in [nb_by_band[band], nb_by_band_legs[band][lc]]:
            bucket["n"] += 1
            bucket["wins"] += int(won)
            bucket["pnl"] += STAKE * ((odds or 2) - 1) if won else -STAKE
            bucket["odds_sum"] += (odds or 2)

    # Process ATS
    ats_by_band = defaultdict(_empty_bucket)
    for mid, odds, status, gd, legs_col, lr in ats_rows:
        band = _classify_band(odds)
        won = _is_won(lr)
        if won is None:
            continue
        ats_by_band[band]["n"] += 1
        ats_by_band[band]["wins"] += int(won)
        ats_by_band[band]["pnl"] += STAKE * ((odds or 2) - 1) if won else -STAKE
        ats_by_band[band]["odds_sum"] += (odds or 2)

    # ── Compute fixture reuse stats ──
    max_reuse = 0
    total_reused_fixtures = 0
    for date_fixes in fixtures_per_date.values():
        for fid, cnt in date_fixes.items():
            if cnt > 1:
                total_reused_fixtures += 1
                max_reuse = max(max_reuse, cnt)

    def _fmt_bucket(b):
        if b["n"] == 0:
            return None, 0
        wr = b["wins"] / b["n"] * 100
        roi = b["pnl"] / (b["n"] * STAKE) * 100
        avg_odds = b["odds_sum"] / b["n"]
        be = 1.0 / avg_odds * 100
        edge = wr - be
        return f"N={b['n']:>3} WR={wr:>5.1f}% ROI={roi:>+6.1f}% edge={edge:>+5.1f}pp", edge

    # ── Main band-for-band table ──
    print(f"\n{'=' * 110}")
    print(f"  NEGBIN FORWARD SHADOW — BAND-FOR-BAND TRACKING (last {days_back} days)")
    print(f"{'=' * 110}")

    print(f"\n  {'Band':>15} │ {'NB (all)':^40} │ {'ATS':^35} │ {'NB-ATS':>7}")
    print(f"  {'─' * 105}")

    for label, lo, hi in ODDS_BANDS:
        nb = nb_by_band[label]
        ats = ats_by_band[label]

        nb_str, nb_edge = _fmt_bucket(nb)
        ats_str, ats_edge = _fmt_bucket(ats)

        nb_disp = nb_str or f"{'—':^40}"
        ats_disp = ats_str or f"{'—':^35}"
        delta_str = f"{nb_edge - ats_edge:>+6.1f}pp" if nb_str and ats_str else f"{'—':>7}"

        print(f"  {label:>15} │ {nb_disp:^40} │ {ats_disp:^35} │ {delta_str}")

    # ── Leg-count breakdown within each band ──
    has_leg_split = any(len(v) > 1 or (len(v) == 1 and list(v.keys())[0] != 2)
                        for v in nb_by_band_legs.values() if v)

    print(f"\n  LEG-COUNT SPLIT (validation was on 2-3 leg parlays):")
    print(f"  {'Band':>15} │ {'Legs':>4} │ {'N':>4} {'WR':>7} {'Edge':>8} │ {'Comparable?':>12}")
    print(f"  {'─' * 65}")

    for label, lo, hi in ODDS_BANDS:
        legs_data = nb_by_band_legs.get(label, {})
        if not legs_data:
            print(f"  {label:>15} │  —")
            continue
        for lc in sorted(legs_data.keys()):
            b = legs_data[lc]
            if b["n"] == 0:
                continue
            wr = b["wins"] / b["n"] * 100
            avg_odds = b["odds_sum"] / b["n"]
            be = 1.0 / avg_odds * 100
            edge = wr - be
            # Flag whether this leg count was in the backtest validation
            comparable = "YES" if lc in (2, 3) else "NO — new"
            print(f"  {label:>15} │ {lc:>4} │ {b['n']:>4} {wr:>6.1f}% {edge:>+7.1f}pp │ {comparable:>12}")

    # ── Totals ──
    nb_total_n = sum(v["n"] for v in nb_by_band.values())
    nb_total_w = sum(v["wins"] for v in nb_by_band.values())
    nb_total_pnl = sum(v["pnl"] for v in nb_by_band.values())
    ats_total_n = sum(v["n"] for v in ats_by_band.values())
    ats_total_w = sum(v["wins"] for v in ats_by_band.values())
    ats_total_pnl = sum(v["pnl"] for v in ats_by_band.values())

    print(f"\n  Totals: NB {nb_total_n} settled (WR={nb_total_w/max(nb_total_n,1)*100:.1f}%, "
          f"PnL=${nb_total_pnl:+.0f}), ATS {ats_total_n} settled "
          f"(WR={ats_total_w/max(ats_total_n,1)*100:.1f}%, PnL=${ats_total_pnl:+.0f})")

    # ── Leg reuse transparency ──
    print(f"\n  LEG REUSE (fixture:team:rung, capped at 3x — matches live construction):")

    # Global leg reuse histogram
    reuse_hist = defaultdict(int)  # appearance_count -> num_legs
    for cnt in leg_appearances.values():
        reuse_hist[cnt] += 1
    total_legs = len(leg_appearances)
    at_cap = sum(1 for c in leg_appearances.values() if c >= 3)
    print(f"    Leg appearance histogram: {dict(sorted(reuse_hist.items()))}")
    print(f"    {total_legs} distinct legs, {at_cap} at cap (3x)")

    # Per-band effective-N estimate
    # Effective-N accounts for correlated parlays sharing legs.
    # Conservative: eff_N = raw_N / avg_leg_appearances_in_band
    print(f"\n  EFFECTIVE-N PER BAND (raw-N discounted by avg leg reuse in band):")
    print(f"  {'Band':>15} │ {'Raw N':>6} {'Avg reuse':>10} {'Eff-N':>7} │ {'Read quality':>15}")
    print(f"  {'─' * 65}")
    for label, lo, hi in ODDS_BANDS:
        nb = nb_by_band[label]
        if nb["n"] == 0:
            print(f"  {label:>15} │ {'0':>6} {'—':>10} {'—':>7} │ {'no data':>15}")
            continue
        band_legs = band_leg_apps.get(label, {})
        if band_legs:
            avg_reuse = sum(band_legs.values()) / len(band_legs)
            eff_n = nb["n"] / avg_reuse
        else:
            avg_reuse = 1.0
            eff_n = nb["n"]
        quality = "RELIABLE" if eff_n >= 30 else ("DIRECTIONAL" if eff_n >= 10 else "TOO THIN")
        print(f"  {label:>15} │ {nb['n']:>6} {avg_reuse:>9.1f}x {eff_n:>6.0f} │ {quality:>15}")

    # Fixture reuse context
    if total_reused_fixtures > 0:
        print(f"\n    {total_reused_fixtures} fixtures appear in multiple parlays "
              f"(max {max_reuse}x). Leg-level cap bounds correlation.")
        print(f"    Per-band reads use effective-N. Aggregate totals are cross-band correlated.")
    else:
        print(f"\n    No fixture reuse detected yet in settled parlays.")

    nb_settled = nb_total_n
    print(f"\n  Status: {'COLLECTING' if nb_settled < 50 else 'EVALUABLE'} "
          f"({nb_settled} settled, need ~50+ for reliable band-level reads)")

    return {
        "nb_by_band": dict(nb_by_band),
        "nb_by_band_legs": {b: dict(v) for b, v in nb_by_band_legs.items()},
        "ats_by_band": dict(ats_by_band),
        "nb_total": nb_total_n,
        "fixture_reuse": {"total_reused": total_reused_fixtures, "max_reuse": max_reuse},
    }


# ═══════════════════════════════════════════════════════════════════════════
# STRAIGHT-BET STACK — Two-stage model (NegBin select → discriminator rank)
# ═══════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   STAGE 1 (NegBin): game selection — which games are in play
#     NegBin is a good SELECTOR but bad DISCRIMINATOR (overconfident
#     within-pool, drifts/reverses at +4.5). Keep it ONLY for game selection.
#
#   STAGE 2 (discriminator stack): within NegBin's pool, rank quality:
#     1. epp < 12 — hard filter (+15-30pp all rungs, universal)
#     2. ridge_composite — continuous rank (+10-15pp all rungs, equal-N)
#     3. ats_dist_wp — continuous rank (monotonic +1.5 to +3.5)
#     4. fav cell preference (+7-15pp all rungs)
#
#   BET STRAIGHT at the rung where EDGE-vs-BREAKEVEN is highest.
#
# Independence check (on existing pool):
#   epp vs ridge:   r = -0.265 (weakly anti-correlated — orthogonal enough)
#   epp vs dist_wp: r = -0.391
#   ridge vs dist:  r = +0.532 (moderate — partially redundant)
#   Stacking does NOT increase edge-vs-breakeven: WR gains = odds shortening.
#   Ridge and dist_wp are partially redundant but ridge adds +3pp over
#   dist_wp alone after epp filter. Keep both for ranking; edge validation
#   must come FORWARD, not from this in-sample pool.
#
# Source tag: shadow_straight_stack
# ═══════════════════════════════════════════════════════════════════════════

_STRAIGHT_STACK_SOURCE = "shadow_straight_stack"
_STRAIGHT_STACK_STAKE = 10.0
_STRAIGHT_STACK_MAX_BETS = 8  # cap daily volume (manageable straight bets)

# Rung-specific fallback odds (from historical medians)
_RUNG_FALLBACK_ODDS = {
    1.5: 1.885,
    2.5: 1.667,
    3.5: 1.400,
    4.5: 1.250,
}

# Rung-specific base rates (from game-deduped synthesis)
# Used to compute edge-vs-breakeven per rung
_RUNG_BASE_WR = {
    1.5: 0.665,
    2.5: 0.742,
    3.5: 0.799,
    4.5: 0.855,
}


def generate_straight_stack(db=None, today_str=None):
    """Two-stage straight-bet model: NegBin select → discriminator rank.

    Reads Shadow A's legs, applies the validated discriminator stack,
    and bets STRAIGHT at the rung with highest edge-vs-breakeven per game.

    Stage 1 (already done by Shadow A): NegBin selected games into pool.
    Stage 2 (this function):
      1. Hard filter: epp < 12
      2. Group by fixture-team
      3. For each fixture-team: find available rungs, look up odds,
         compute edge-vs-breakeven per rung, pick the best rung
      4. Rank all fixture-teams by discriminator score:
         score = 0.5 * ridge_composite + 0.3 * ats_dist_wp + 0.2 * cell_boost
      5. Bet the top N straight

    Source tag: shadow_straight_stack
    """
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text
    import re

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[straight-stack] Generating straight-bet stack for {today_str}")

    # ── Dedup check ──
    existing = db.execute(_text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source=:src AND game_date=:gd"
    ), {"src": _STRAIGHT_STACK_SOURCE, "gd": today_str}).scalar()
    if existing > 0:
        print(f"[straight-stack] Already {existing} bets for {today_str} — skipping")
        return []

    # ── Read Shadow A's legs for today ──
    leg_data = db.execute(_text(
        "SELECT l.fixture_id, l.description, l.win_prob, l.negbin_edge_pp, "
        "       l.ats_cover_wp, l.negbin_wp, l.negbin_mu_home, l.negbin_mu_away, "
        "       l.negbin_cell, l.ats_dist_wp, l.ridge_composite, l.bayes_team_wp "
        "FROM mock_bet_legs l "
        "JOIN mock_bets b ON l.mock_bet_id = b.id "
        "WHERE b.source='shadow_negbin' AND b.game_date=:gd AND b.cohort='clean'"
    ), {"gd": today_str}).fetchall()

    if not leg_data:
        print("[straight-stack] No Shadow A legs found for today")
        return []

    print(f"[straight-stack] {len(leg_data)} Shadow A legs loaded")

    # ── Parse and filter ──
    all_candidates = []
    n_epp_toxic = 0
    for row in leg_data:
        (fix_id, desc, wp, edge_pp, ats, nb_wp, mu_h, mu_a,
         cell, dist_wp, ridge, bayes) = row

        if fix_id is None or desc is None:
            continue

        # Hard filter: epp < 12
        if edge_pp is not None and edge_pp >= _EDGE_GATE_THRESHOLD_PP:
            n_epp_toxic += 1
            continue

        # Parse team and rung from description
        m = re.match(r"^(.+?)\s+\+(\d+\.?\d*)$", desc)
        if not m:
            continue
        team = m.group(1).strip()
        rung = float(m.group(2))

        if rung not in _RUNG_FALLBACK_ODDS:
            continue

        all_candidates.append({
            "fixture_id": fix_id,
            "description": desc,
            "team": team,
            "rung": rung,
            "win_prob": wp,
            "negbin_edge_pp": edge_pp,
            "negbin_wp": nb_wp,
            "negbin_mu_home": mu_h,
            "negbin_mu_away": mu_a,
            "negbin_cell": cell,
            "ats_dist_wp": dist_wp or 0.5,
            "ats_cover_wp": ats,
            "ridge_composite": ridge or 0.5,
            "bayes_team_wp": bayes,
        })

    print(f"[straight-stack] {len(all_candidates)} candidates after epp<12 "
          f"({n_epp_toxic} epp toxic)")

    if not all_candidates:
        print("[straight-stack] No candidates after epp filter")
        return []

    # ── Group by fixture-team, find best rung per game ──
    fixture_team = defaultdict(list)
    for c in all_candidates:
        key = f"{c['fixture_id']}_{c['team']}"
        fixture_team[key].append(c)

    print(f"[straight-stack] {len(fixture_team)} unique fixture-teams")

    # For each fixture-team: look up odds at each available rung,
    # compute edge-vs-breakeven, pick the best rung
    best_per_game = []
    rung_chosen = defaultdict(int)

    for ft_key, candidates in fixture_team.items():
        fix_id = candidates[0]["fixture_id"]
        team = candidates[0]["team"]

        home_team, away_team = _get_fixture_teams(fix_id, db)
        if home_team is None or away_team is None:
            continue

        # Evaluate each available rung
        rung_options = []
        for c in candidates:
            rung = c["rung"]
            fallback = _RUNG_FALLBACK_ODDS.get(rung, 1.500)
            odds = _get_spread_odds(team, home_team, away_team,
                                    today_str, rung, fallback)

            if odds <= 1.0:
                continue

            # Edge-vs-breakeven: model WR at this rung minus breakeven
            # Use negbin_wp as the model's estimate for this specific rung
            model_wp = c["negbin_wp"] or _RUNG_BASE_WR.get(rung, 0.7)
            breakeven = 1.0 / odds
            edge_vs_be = model_wp - breakeven

            # Also compute expected ROI for ranking
            expected_roi = model_wp * odds - 1.0

            rung_options.append({
                **c,
                "actual_odds": odds,
                "breakeven": breakeven,
                "edge_vs_be": edge_vs_be,
                "expected_roi": expected_roi,
                "home_team": home_team,
                "away_team": away_team,
            })

        if not rung_options:
            continue

        # Pick the rung with highest edge-vs-breakeven
        rung_options.sort(key=lambda x: x["edge_vs_be"], reverse=True)
        best = rung_options[0]

        # Skip if edge-vs-breakeven is negative (model thinks this is -EV)
        if best["edge_vs_be"] <= 0:
            continue

        # Compute discriminator score for ranking across games
        cell = best.get("negbin_cell") or ""
        cell_boost = 1.0
        if "fav" in cell:
            cell_boost = 1.15
        elif "dog" in cell:
            cell_boost = 0.85

        disc_score = (0.50 * best["ridge_composite"]
                      + 0.30 * best["ats_dist_wp"]
                      + 0.20 * cell_boost)

        best["disc_score"] = disc_score
        best_per_game.append(best)
        rung_chosen[best["rung"]] += 1

    print(f"[straight-stack] {len(best_per_game)} fixture-teams with positive edge-vs-BE")
    rung_str = ", ".join(f"+{r:.1f}:{c}" for r, c in sorted(rung_chosen.items()))
    print(f"[straight-stack] Rung distribution (by edge-vs-BE): {rung_str}")

    if not best_per_game:
        print("[straight-stack] No positive-edge fixture-teams")
        return []

    # ── Rank by discriminator score, take top N ──
    best_per_game.sort(key=lambda x: x["disc_score"], reverse=True)
    selected = best_per_game[:_STRAIGHT_STACK_MAX_BETS]

    print(f"[straight-stack] Selected top {len(selected)} by discriminator score")

    # ── Store straight bets ──
    run_id = str(uuid.uuid4())
    stored = 0

    for pick in selected:
        team = pick["team"]
        rung = pick["rung"]
        odds = pick["actual_odds"]
        fix_id = pick["fixture_id"]

        bet_desc = f"{team} +{rung:.1f}"
        model_wp = pick["negbin_wp"] or (1.0 / odds)

        mock_id = str(uuid.uuid4())
        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=odds,
            amount=_STRAIGHT_STACK_STAKE,
            legs=1,
            bet_info=json.dumps({
                "type": "straight_stack",
                "team": team,
                "rung": rung,
                "edge_vs_be": round(pick["edge_vs_be"], 4),
                "disc_score": round(pick["disc_score"], 4),
                "ridge_composite": round(pick["ridge_composite"], 4),
                "ats_dist_wp": round(pick["ats_dist_wp"], 4),
                "negbin_cell": pick.get("negbin_cell"),
                "negbin_edge_pp": round(pick["negbin_edge_pp"], 2) if pick.get("negbin_edge_pp") else None,
                "actual_odds": round(odds, 3),
                "breakeven": round(pick["breakeven"], 4),
            }),
            predicted_win_prob=model_wp,
            model_used="negbin_select+disc_stack_v1",
            status="PENDING",
            source=_STRAIGHT_STACK_SOURCE,
            cohort="clean",
            weight=1.0,
        )
        db.add(mock)
        db.flush()

        # Set mean/min dist_wp
        try:
            db.execute(_text(
                "UPDATE mock_bets SET mean_leg_dist_wp=:mdw, min_leg_dist_wp=:mindw "
                "WHERE id=:mid"
            ), {"mdw": round(pick["ats_dist_wp"], 4),
                "mindw": round(pick["ats_dist_wp"], 4), "mid": mock_id})
        except Exception:
            pass

        # Create the single leg
        leg_obj = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=bet_desc,
            market_type="alternate_spreads",
            sport="MLB",
            win_prob=model_wp,
            fixture_id=fix_id,
            is_alt_line=True,
        )
        db.add(leg_obj)
        db.flush()

        # Set spread_value
        try:
            db.execute(_text(
                "UPDATE mock_bet_legs SET spread_value=:sv WHERE id=:lid"
            ), {"sv": rung, "lid": leg_obj.id})
        except Exception:
            pass

        # Copy signal columns
        updates = {}
        for col, val in [
            ("negbin_wp", pick.get("negbin_wp")),
            ("negbin_edge_pp", pick.get("negbin_edge_pp")),
            ("negbin_mu_home", pick.get("negbin_mu_home")),
            ("negbin_mu_away", pick.get("negbin_mu_away")),
            ("negbin_cell", pick.get("negbin_cell")),
            ("ats_cover_wp", pick.get("ats_cover_wp")),
            ("ats_dist_wp", pick.get("ats_dist_wp")),
            ("bayes_team_wp", pick.get("bayes_team_wp")),
            ("ridge_composite", pick.get("ridge_composite")),
        ]:
            if val is not None:
                updates[col] = val
        if updates:
            set_clause = ", ".join(f"{k}=:{k}" for k in updates)
            updates["lid"] = leg_obj.id
            try:
                db.execute(_text(
                    f"UPDATE mock_bet_legs SET {set_clause} WHERE id=:lid"
                ), updates)
            except Exception:
                pass

        stored += 1
        print(f"  [{stored}] {bet_desc} odds={odds:.3f} "
              f"edge_vs_be={pick['edge_vs_be']:.3f} "
              f"disc={pick['disc_score']:.3f} "
              f"ridge={pick['ridge_composite']:.3f} "
              f"dist_wp={pick['ats_dist_wp']:.3f} "
              f"cell={pick.get('negbin_cell', '?')}")

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[straight-stack] DB commit failed: {e}")
        return []

    print(f"[straight-stack] Stored {stored} straight bets "
          f"(source={_STRAIGHT_STACK_SOURCE})")

    return selected


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        report_forward_tracking(days)
    elif len(sys.argv) > 1 and sys.argv[1] == "recal":
        # Generate Shadow B (recalibrated) picks
        from database import SessionLocal
        db = SessionLocal()
        try:
            picks = generate_negbin_recal_shadow_picks(db)
            if picks:
                print(f"\n[negbin-recal] Summary:")
                for i, p in enumerate(picks):
                    legs_desc = " + ".join(f"{l['pick_team']} +{l['spread_value']}" for l in p["legs"])
                    print(f"  #{i+1}: {legs_desc} | odds={p['combined_odds']:.3f} "
                          f"| band={p['odds_band']} | cell_adj={p['cell_adj']}")
        finally:
            db.close()
    else:
        # Generate Shadow A (uncalibrated) picks
        from database import SessionLocal
        db = SessionLocal()
        try:
            picks = generate_negbin_shadow_picks(db)
            if picks:
                print(f"\n[negbin-shadow] Summary:")
                for i, p in enumerate(picks):
                    legs_desc = " + ".join(f"{l['pick_team']} +{l['spread_value']}" for l in p["legs"])
                    print(f"  #{i+1}: {legs_desc} | odds={p['combined_odds']:.3f} "
                          f"| band={p['odds_band']} | cell_adj={p['cell_adj']}")
        finally:
            db.close()
