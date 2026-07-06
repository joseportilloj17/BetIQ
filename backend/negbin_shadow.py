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
        ("mock_bets",     "negbin_odds_band",  "ALTER TABLE mock_bets ADD COLUMN negbin_odds_band TEXT"),
        ("mock_bets",     "negbin_cell_adj",   "ALTER TABLE mock_bets ADD COLUMN negbin_cell_adj REAL"),
        ("mock_bets",     "negbin_leg_count",  "ALTER TABLE mock_bets ADD COLUMN negbin_leg_count INTEGER"),
        ("mock_bets",     "live_realistic",    "ALTER TABLE mock_bets ADD COLUMN live_realistic INTEGER DEFAULT 0"),
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
        avg_wp = np.mean([c["model_wp"] for c in combo])
        avg_lqs = np.mean([c.get("lqs", 50) for c in combo])
        avg_los = np.mean([c.get("los", 0.5) for c in combo])
        combined_wp = 1.0
        for c in combo:
            combined_wp *= c["model_wp"]
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
    from ml_model import _get_feature_matrix
    from walk_forward_backtest import _compute_los, _compute_lqs
    import itertools

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    tag_label = "recal" if source_tag == "shadow_negbin_recal" else "shadow"
    print(f"\n[negbin-{tag_label}] Generating {tag_label} picks for {today_str}")

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

                    candidates.append({
                        "fixture_id": fix["fixture_id"],
                        "fixture_key": fixture_key,
                        "home_team": home, "away_team": away,
                        "game_date": today_str,
                        "spread_value": sv, "pick_side": "home",
                        "pick_team": home,
                        "model_wp": home_wp, "market_wp": mkt_home,
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

                    candidates.append({
                        "fixture_id": fix["fixture_id"],
                        "fixture_key": fixture_key,
                        "home_team": home, "away_team": away,
                        "game_date": today_str,
                        "spread_value": sv, "pick_side": "away",
                        "pick_team": away,
                        "model_wp": away_wp, "market_wp": mkt_away,
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
        avg_wp = np.mean([c["model_wp"] for c in combo])
        avg_lqs = np.mean([c.get("lqs", 50) for c in combo])
        avg_los = np.mean([c.get("los", 0.5) for c in combo])
        combined_wp = 1.0
        for c in combo:
            combined_wp *= c["model_wp"]
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

        stored += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[negbin-shadow] DB commit failed: {e}")
        return

    print(f"[negbin-shadow] Stored {stored} shadow parlays (source={source_tag})")


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
