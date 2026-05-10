"""
recommender.py — Phase 3: Proactive bet recommendation engine.

Strategy:
  - Pick the legs the model is MOST CONFIDENT will win (highest win_prob)
  - Build short parlays (2-4 legs) so combined win_prob stays high
  - Require the final parlay to pay at least min_payout on the stake
  - Sort by combined win_prob × payout (expected profit)

Reality check on math:
  - 2 legs at 70% each = 49% combined win, ~+$18 payout on $10
  - 3 legs at 65% each = 27% combined win, ~+$30 payout on $10
  - The model's 85.71% accuracy means it correctly identifies the winning
    side 85.71% of the time — leverage that by picking its top-confidence legs
"""
from __future__ import annotations
import math
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from database import Bet, Fixture
import kelly as kly
import strategy as strat


# ── Model version labels ───────────────────────────────────────────────────────
_MODEL_LABEL = {
    "NHL":          "nhl_ats_v1",
    "MLB":          "mlb_ats_v1",
    "Soccer":       "soccer_v1",      # routes to soccer_total_v1 or soccer_ml_v1 per leg
    "soccer_total_v1": "soccer_total_v1",
    "soccer_ml_v1":    "soccer_ml_v1",
}
_DEFAULT_MODEL_LABEL = "combined_v1"

# ── Sub-model AUC (used to scale Kelly fraction) ──────────────────────────────
# Higher AUC → more edge → higher fractional Kelly.
# Fractions: NHL 0.35 (AUC 0.6372), MLB 0.25 (AUC 0.5509), combined 0.30 (AUC 0.5972)
_SUBMODEL_AUC = {
    "NHL":          0.7544,   # retrained 2026-04-21 (was 0.6372)
    "MLB":          0.6360,   # retrained 2026-04-21 (was 0.5509)
    "soccer_total_v1": 0.5529,  # trained 2026-04-26 (total goals form model)
    "soccer_ml_v1":    0.5542,  # trained 2026-04-26 (moneyline form model)
    "Soccer":       0.5535,   # average of soccer sub-models for parlay AUC
}
_COMBINED_AUC = 0.5972
_SUBMODEL_KELLY_FRACTION = {
    "NHL":          0.45,   # scaled up with AUC 0.7544
    "MLB":          0.35,   # scaled up with AUC 0.6360
    "Soccer":       0.25,   # conservative — soccer first-gen model (AUC ~0.554)
    "soccer_total_v1": 0.25,
    "soccer_ml_v1":    0.25,
}
_DEFAULT_KELLY_FRACTION = 0.30

# Per-sport Section A confidence thresholds (min per-leg win_prob for high-conf section)
_SECTION_A_THRESHOLD = {
    "NHL": 65.0,
    "MLB": 60.0,
}

# ── Pattern cache (for attribution display) ───────────────────────────────────
_pattern_cache = None
_cache_built_at = None
CACHE_TTL_SECONDS = 300


def _build_pattern_cache(db: Session) -> dict:
    bets = db.query(Bet).filter(
        Bet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()
    combos: dict = {}
    for b in bets:
        sport  = (b.sports or "other").split("|")[0].strip()
        legs_b = _legs_bucket(b.legs or 1)
        won    = 1 if b.status == "SETTLED_WIN" else 0
        roi    = (b.profit or 0) / (b.amount or 1)
        key    = (sport, legs_b)
        if key not in combos:
            combos[key] = {"wins": 0, "total": 0, "roi_sum": 0.0}
        combos[key]["wins"]    += won
        combos[key]["total"]   += 1
        combos[key]["roi_sum"] += roi

    patterns: dict = {"_global": {}}
    for (sport, legs_b), v in combos.items():
        n = v["total"]
        if n < 5:
            continue
        patterns["_global"][(sport, legs_b)] = {
            "win_rate": round(v["wins"] / n, 4),
            "roi":      round(v["roi_sum"] / n, 4),
            "n":        n,
        }
    return patterns


def _get_patterns(db: Session) -> dict:
    global _pattern_cache, _cache_built_at
    now = datetime.utcnow()
    if (_pattern_cache is None or _cache_built_at is None or
            (now - _cache_built_at).total_seconds() > CACHE_TTL_SECONDS):
        _pattern_cache = _build_pattern_cache(db)
        _cache_built_at = now
    return _pattern_cache


def _legs_bucket(legs: int) -> str:
    if legs <= 2:   return "1-2"
    if legs <= 4:   return "3-4"
    if legs <= 6:   return "5-6"
    if legs <= 9:   return "7-9"
    return "10+"


def _top_patterns(patterns: dict) -> list[dict]:
    rows = []
    for (sport, legs_b), v in patterns.get("_global", {}).items():
        if v["n"] >= 10:
            rows.append({
                "sport":    sport,
                "legs_bucket": legs_b,
                "win_rate": round(v["win_rate"] * 100, 1),
                "roi_pct":  round(v["roi"] * 100, 1),
                "n":        v["n"],
            })
    rows.sort(key=lambda x: -x["roi_pct"])
    return rows[:6]


# ── Quality rating ─────────────────────────────────────────────────────────────

def compute_quality(
    combined_win_prob: float,   # 0-100
    combined_odds:     float,   # decimal
    stake:             float,
    n_legs:            int,
    warnings:          list,
) -> dict:
    """
    Rate a parlay 0-10 based on:
      - Combined win probability (50%)
      - Expected profit relative to stake (30%)
      - Parlay structure / leg count (20%)
    """
    # 1. Win prob score — 60% combined = 6/10, 40% = 4/10, 80% = 8/10
    wp_score = max(0, min(10, combined_win_prob / 10))

    # 2. Expected profit score
    payout        = (combined_odds - 1) * stake
    expected_profit = (combined_win_prob / 100) * payout - (1 - combined_win_prob / 100) * stake
    # Normalise: +$10 expected profit = 10/10, -$5 = 0/10
    profit_score = max(0, min(10, (expected_profit + 5) / 1.5))

    # 3. Structure score — prefer 2-4 legs
    if n_legs <= 2:   struct = 9.0
    elif n_legs <= 3: struct = 8.0
    elif n_legs <= 4: struct = 7.0
    elif n_legs <= 5: struct = 5.5
    else:             struct = 3.0
    struct = max(0, struct - len(warnings) * 0.5)

    raw = wp_score * 0.50 + profit_score * 0.30 + struct * 0.20
    rating = round(max(0, min(10, raw)), 1)
    grade  = "A" if rating >= 8 else ("B" if rating >= 6 else ("C" if rating >= 4 else "D"))

    return {
        "rating":  rating,
        "grade":   grade,
        "breakdown": {
            "win_prob_score":    round(wp_score, 1),
            "profit_score":      round(profit_score, 1),
            "structure_score":   round(struct, 1),
            "expected_profit":   round(expected_profit, 2),
        },
    }


# ── Parlay scoring ─────────────────────────────────────────────────────────────

def score_parlay(legs: list[dict], stake: float, db: Session) -> dict:
    from parlay_builder import build_parlay, correlation_check
    if not legs:
        return {"error": "No legs provided"}

    parlay   = build_parlay(legs, stake=stake)
    warnings = parlay.get("warnings", [])

    # Use 85.71% model accuracy as actual win rate per leg
    MODEL_ACCURACY = 0.8571
    n = len(legs)
    actual_win_prob = MODEL_ACCURACY ** n * 100   # e.g. 73.5% for 2-leg
    quality = compute_quality(
        combined_win_prob = actual_win_prob,
        combined_odds     = parlay["combined_odds"],
        stake             = stake,
        n_legs            = n,
        warnings          = warnings,
    )

    payout_table = []
    for s in [5, 10, 25, 50]:
        p_win = round((parlay["combined_odds"] - 1) * s, 2)
        p_ev  = round(actual_win_prob / 100 * p_win
                      - (1 - actual_win_prob / 100) * s, 2)
        payout_table.append({"stake": s, "payout": p_win, "ev": p_ev})

    # Historical context
    patterns  = _get_patterns(db)
    sport     = (legs[0].get("sport", "other")) if legs else "other"
    legs_b    = _legs_bucket(len(legs))
    hist      = patterns.get("_global", {}).get((sport, legs_b),
                {"win_rate": 0.0, "roi": 0.0, "n": 0})

    # Kelly criterion stake recommendation (25% fractional Kelly)
    try:
        bankroll_state = kly.load_bankroll()
        bankroll       = bankroll_state.get("current_bankroll", 500.0)
        kelly_result   = kly.kelly_stake(
            win_prob     = actual_win_prob / 100,
            decimal_odds = parlay["combined_odds"],
            bankroll     = bankroll,
            fraction     = 0.25,   # 25% fractional Kelly
        )
    except Exception:
        kelly_result = {"stake": stake, "kelly_pct": 0, "rationale": "Set bankroll to enable Kelly sizing"}

    return {
        **parlay,
        "stake":              kelly_result.get("stake", stake),  # Kelly-recommended stake
        "kelly_stake":        kelly_result.get("stake", stake),
        "kelly_pct":          kelly_result.get("kelly_pct", 0),
        "kelly_rationale":    kelly_result.get("rationale", ""),
        "bankroll":           kelly_result.get("bankroll", 500),
        "combined_win_prob":  round(actual_win_prob, 2),
        "dominant_sport":     sport,
        "legs_bucket":        legs_b,
        "historical":         {**hist, "confidence": "medium" if hist["n"] >= 10 else "low"},
        "quality":            quality,
        "payout_table":       payout_table,
        "accuracy_note":      f"Win prob based on {MODEL_ACCURACY*100:.2f}% model accuracy per leg",
    }


# ── Regime A/B shadow run ─────────────────────────────────────────────────────

# Standard composite weights used in production
_STANDARD_WEIGHTS = {"win_prob": 0.40, "ev": 0.40, "lqs": 0.20}


def _weights_differ_meaningfully(w1: dict, w2: dict, threshold: float = 0.05) -> bool:
    """Return True if any weight key differs by more than threshold."""
    for key in ("win_prob", "ev", "lqs"):
        if abs(w1.get(key, 0) - w2.get(key, 0)) > threshold:
            return True
    return False


def _do_regime_shadow(
    candidates:    list,
    standard_picks: list,
    cand_max_ev:   float,
    stake:         float,
    db,
) -> None:
    """
    Shadow-run today's picks under regime-suggested weights and log to regime_ab_log.
    Never modifies standard_picks; never served to the UI.

    Only runs when:
      - market_regime_log has an entry for today
      - suggested_weights differ meaningfully from _STANDARD_WEIGHTS (> 5pp any key)
    """
    import json as _json
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI
    from sqlalchemy import text as _sqla_text

    today_ct = _dt.now(_ZI("America/Chicago")).strftime("%Y-%m-%d")

    # ── Load today's regime ───────────────────────────────────────────────────
    try:
        row = db.execute(_sqla_text(
            "SELECT regime, suggested_weights FROM market_regime_log WHERE date = :d LIMIT 1"
        ), {"d": today_ct}).fetchone()
    except Exception:
        return

    if not row or not row[1]:
        return   # no regime today

    try:
        regime_weights = _json.loads(row[1])   # {"win_prob": 0.45, "ev": 0.4, "lqs": 0.15}
    except Exception:
        return

    if not _weights_differ_meaningfully(regime_weights, _STANDARD_WEIGHTS):
        return   # weights too close — skip pointless shadow run

    regime_name = row[0] or "unknown"

    # ── Skip if already logged today ──────────────────────────────────────────
    try:
        existing = db.execute(_sqla_text(
            "SELECT date FROM regime_ab_log WHERE date = :d LIMIT 1"
        ), {"d": today_ct}).fetchone()
        if existing:
            return
    except Exception:
        pass

    if not candidates:
        return

    # ── Re-score candidates with regime weights ───────────────────────────────
    rw_wp  = regime_weights.get("win_prob", 0.4)
    rw_ev  = regime_weights.get("ev",       0.4)
    rw_lqs = regime_weights.get("lqs",      0.2)
    _max_ev = cand_max_ev or 1.0

    regime_cands = []
    for c in candidates:
        raw_lqs  = sum(l.get("lqs") or 50 for l in c["legs"]) / max(len(c["legs"]), 1)
        norm_ev  = max(c["expected_profit"], 0) / _max_ev
        score    = (
            (c["combined_win_prob"] / 100) * rw_wp +
            norm_ev * rw_ev +
            (raw_lqs / 100) * rw_lqs
        )
        regime_cands.append((score, c))

    regime_cands.sort(key=lambda x: -x[0])

    # ── Assemble shadow picks (same dedup logic as standard) ──────────────────
    _MAX_APP = 2
    shadow_picks = []
    used_sigs: set = set()
    leg_app_cnt: dict = {}

    for _sc, cand in regime_cands:
        if len(shadow_picks) >= max(len(standard_picks), 5):
            break
        sig = frozenset(l["leg_id"] for l in cand["legs"])
        if sig in used_sigs:
            continue
        if any(leg_app_cnt.get(l["leg_id"], 0) >= _MAX_APP for l in cand["legs"]):
            continue
        if any(
            len(sig & frozenset(l["leg_id"] for l in p_["legs"])) / max(len(sig), 1) > 0.5
            for p_ in shadow_picks
        ):
            continue
        shadow_picks.append(cand)
        used_sigs.add(sig)
        for l in cand["legs"]:
            lid = l["leg_id"]
            leg_app_cnt[lid] = leg_app_cnt.get(lid, 0) + 1

    # ── Compute overlap stats ─────────────────────────────────────────────────
    std_sigs    = {frozenset(l["leg_id"] for l in p["legs"]) for p in standard_picks}
    shad_sigs   = {frozenset(l["leg_id"] for l in c["legs"]) for c in shadow_picks}
    overlap_n   = len(std_sigs & shad_sigs)

    # ── Build shadow pick id list (use fixture/leg signatures — no mock_bet ids yet) ──
    shadow_ids = [
        "+".join(sorted(l["leg_id"] for l in c["legs"]))
        for c in shadow_picks
    ]
    standard_ids = [p.get("bet_id") or "" for p in standard_picks]

    # ── Write to regime_ab_log ────────────────────────────────────────────────
    try:
        from database import RegimeAbLog
        ab_row = RegimeAbLog(
            date              = today_ct,
            regime            = regime_name,
            standard_weights  = _json.dumps(_STANDARD_WEIGHTS),
            standard_pick_ids = _json.dumps(standard_ids),
            standard_n        = len(standard_picks),
            regime_weights    = _json.dumps(regime_weights),
            regime_pick_ids   = _json.dumps(shadow_ids),
            regime_n          = len(shadow_picks),
            overlap_n         = overlap_n,
            only_standard_n   = len(std_sigs - shad_sigs),
            only_regime_n     = len(shad_sigs - std_sigs),
            created_at        = _dt.utcnow().isoformat(),
        )
        db.add(ab_row)
        db.commit()
        print(f"[Recommender] Regime A/B logged: {regime_name} shadow={len(shadow_picks)} "
              f"standard={len(standard_picks)} overlap={overlap_n}")
    except Exception as _we:
        db.rollback()
        print(f"[Recommender] Regime A/B log write failed: {_we}")


# ── Core pick generation ───────────────────────────────────────────────────────

def generate_todays_picks(
    db:                  Session,
    n_picks:             int        = 5,
    stake:               float      = 10.0,
    max_legs:            int        = 5,
    min_legs:            int        = 2,
    min_odds:            float      = 2.0,
    sort_by:             str        = "win_prob",
    sport_filter:        list       = None,   # e.g. ["NBA", "MLB"] — None means all sports
    min_confidence:      float      = None,   # override per-sport thresholds (e.g. 50.0 for paper trading)
    depth_lqs_min:       float      = None,   # override _DEPTH_LQS_MIN floor (e.g. 55.0 for paper trading)
    boost_pct:           float      = 0.0,    # active boost tier — threaded to ALE for safety re-ranking
) -> dict:
    """
    Build today's top picks by:
    1. Scoring every available leg with the ML model
    2. Taking only legs the model is most confident will WIN (highest win_prob)
    3. Building short parlays (2-4 legs) to keep combined win_prob high
    4. Requiring the final parlay to pay at least min_odds
    5. Sorting by combined win_prob (highest chance of actually hitting)

    The model's 85.71% accuracy means its top-confidence legs are very reliable.
    We exploit that by using only the model's most confident picks.
    """
    from parlay_builder import get_available_legs, score_leg, build_parlay, correlation_check
    import itertools

    all_raw  = get_available_legs(db, markets=["h2h", "spreads", "totals"])
    if not all_raw:
        return {
            "error": "No fixture legs available. Go to Fixtures → Fetch Latest Odds first.",
            "picks": [],
        }

    # Always exclude sports that have no trained model and cannot be settled.
    # These inflate _legs_evaluated without contributing picks.
    _EXCLUDED_SPORTS = frozenset({"MMA", "NCAAF", "NCAAB", "Tennis"})
    all_raw = [l for l in all_raw if l.get("sport", "") not in _EXCLUDED_SPORTS]

    # Sport label mapping — used for both pre-filter (before scoring) and
    # post-filter (exact label match).
    SPORT_GROUP = {
        "nba":     ["NBA"],
        "mlb":     ["MLB"],
        "nfl":     ["NFL"],
        "nhl":     ["NHL"],
        "ncaab":   ["NCAAB"],
        "ncaaf":   ["NCAAF"],
        "soccer":  ["EPL", "La Liga", "UCL", "Bundesliga",
                    "Serie A", "Ligue 1", "Soccer"],
        # MLS excluded: no sub-model, no CLV data, poor historical accuracy.
        # Re-enable when an MLS-specific model is built.
        "tennis":  ["Tennis"],
        "mma":     ["MMA"],
    }

    # Pre-filter by sport BEFORE scoring when a sport_filter is specified.
    # This is a major speedup for per-sport mock-gen passes (e.g. MLB-only scores
    # 114 legs instead of all 1038, reducing ALE + score time by ~10×).
    if sport_filter:
        allowed_labels = set()
        for f in sport_filter:
            key = f.strip().lower()
            if key in SPORT_GROUP:
                allowed_labels.update(SPORT_GROUP[key])
            else:
                allowed_labels.add(f.strip())
                allowed_labels.add(f.strip().upper())
        all_raw = [l for l in all_raw
                   if not l.get("sport") or l.get("sport", "") in allowed_labels]

    # Score every leg
    scored = [score_leg(l) for l in all_raw]
    _legs_evaluated_total = len(scored)  # count before (now redundant) post-filter

    # Post-filter — applies only when sport_filter is given AND some legs slipped
    # through the pre-filter (e.g. legs with no sport tag).
    if sport_filter:
        # allowed_labels already built above
        scored = [l for l in scored if l.get("sport", "") in allowed_labels]

        if not scored:
            sport_name = ", ".join(sport_filter).upper()
            return {
                "error": f"No {sport_name} legs available right now. "
                         f"This sport may be out of season, or try Fixtures → Fetch Latest Odds.",
                "picks": [],
                "pool_assessment": {
                    "total_legs": 0, "legs_evaluated": _legs_evaluated_total,
                    "legs_positive_ev": 0,
                    "reason_no_picks": f"No {sport_name} fixtures with future start times found.",
                    "next_refresh": "Next refresh: 7:45 AM CT tomorrow",
                },
            }

    # ── Only surface picks where model predicted probability >= threshold ───
    # Validated threshold from backtest: 58.3% win rate, +11.4% ROI at cutoff.
    # NHL uses 60% because the sub-model is underconfident at the top bucket:
    # 60% predicted ≈ 67%+ actual outcomes.  MLB uses standard 55% (isotonic
    # calibration already corrected overconfidence).
    _SPORT_CONFIDENCE_THRESHOLDS = {
        # NHL: 60% predicted ≈ 67%+ actual outcomes (sub-model underconfident).
        # Playoff consideration: puck line (-1.5) requires winning by 2 in a sport
        # where most games end within 1 goal → raise to 65-68% when playoffs detected.
        # TODO: add is_nhl_playoffs() check and bump NHL to 68.0 in Apr-Jun window.
        "NHL": 60.0,   # base threshold; raise to 68.0 for playoffs when flagged
        "MLB": 55.0,   # isotonic-calibrated, standard threshold
        # Soccer uses 3-way markets (h2h outcomes each ~33% base). A model
        # win_prob of 45%+ on a 33% implied side is genuinely confident.
        # Applying the 55% binary-sport bar would suppress all soccer picks.
        "EPL":        45.0,
        "La Liga":    45.0,
        "Bundesliga": 45.0,
        "Serie A":    45.0,
        "Ligue 1":    45.0,
        "UCL":        45.0,
        "Soccer":     45.0,
        # MLS intentionally omitted — excluded from scoring pool
    }
    _DEFAULT_THRESHOLD = 55.0
    # No fallbacks — surfacing negative-EV legs is worse than showing no picks.
    # The post-augmentation filter below is the authoritative gate.

    # ── Augment pool with high-probability cushion alt lines ─────────────────
    # Root cause fix: `get_scored_alt_lines` sorts by EV descending which surfaces
    # extreme deep alt lines (e.g. Over 14.5 at 9.0x, spread -41.5 at 10.9x).
    # These have 9-15% win probability and LQS ~51 — always below main-line LQS.
    #
    # What we want: HIGH-PROBABILITY cushion lines in parlay-sensible odds range
    # (1.25–3.5 decimal ≈ -300 to +250 American, ~29–80% implied prob).
    # These offer easier coverage than the main market line while staying viable
    # for parlay inclusion.
    try:
        import creator_tier as ct
        import leg_quality as lq_mod

        # Odds range for "cushion alt line" — high enough to add parlay value,
        # low enough to actually hit.  Widened to capture alternate_spreads at
        # ±2.5 (underdog +2.5 ~1.25–1.72; favorite −2.5 ~2.06–3.50).
        _ALT_MIN_ODDS = 1.25   # ~−300 American (underdog +2.5 lines start here)
        _ALT_MAX_ODDS = 3.50   # ~+250 American (captures favorite −2.5 at +170–250)

        # Build a lookup: fixture_id → best main-market LQS for that fixture
        main_lqs_by_fixture: dict[str, float] = {}
        for l in scored:
            fid = l.get("fixture_id")
            if not fid:
                continue
            _is_home   = l.get("pick") == l.get("home_team")
            _opponent  = l.get("away_team") if _is_home else l.get("home_team")
            cand = {
                "market_type":      l.get("market_label") or l.get("market"),
                "sport":            l.get("sport"),
                "team_or_player":   l.get("pick"),
                "model_confidence": l.get("win_prob"),
                "model_used":       l.get("model_used"),
                "edge_pp":          l.get("edge"),
                "line":             l.get("point"),
                "is_home":          _is_home,
                "opponent":         _opponent,
            }
            try:
                lqs_r = lq_mod.compute_leg_quality_score(cand, db)
                _lqs  = lqs_r.get("lqs") or 0
                existing = main_lqs_by_fixture.get(fid, 0)
                main_lqs_by_fixture[fid] = max(existing, _lqs)
                l["lqs"] = _lqs  # store on leg for later depth assessment
            except Exception:
                pass

        unique_fixtures = list({l.get("fixture_id"): l for l in scored
                                 if l.get("fixture_id")}.values())

        n_evaluated = 0
        n_added     = 0

        for main_leg in unique_fixtures:
            fid = main_leg.get("fixture_id")
            if not fid:
                continue
            try:
                # Fetch ALL stored alt lines, then filter by odds range ourselves.
                # min_label='ALL' avoids the EV-bias of the VALUE/ANCHOR filter.
                all_alt_lines = ct.get_scored_alt_lines(fid, min_label="ALL")
            except Exception:
                continue

            # Keep only reasonable-odds lines (cushion range: ~29–80% implied prob)
            cushion_lines = [
                al for al in all_alt_lines
                if _ALT_MIN_ODDS <= float(al.get("odds") or 0) <= _ALT_MAX_ODDS
                and al.get("market_key", "") in (
                    "alternate_spreads", "alternate_totals", "spreads", "totals"
                )
            ]

            for al in cushion_lines:
                n_evaluated += 1
                al_odds_dec = float(al.get("odds"))
                al_market   = al.get("market_key", "")
                al_line     = al.get("line")
                al_team     = al.get("team", "")
                al_sport    = main_leg.get("sport", "")
                mkt_label   = "Alt Spread" if "spread" in al_market else "Alt Total"
                point_str   = (f" {al_line:+g}" if al_line is not None
                               and float(al_line) != 0 else "")
                al_desc     = f"{al_team}{point_str} ({mkt_label})"
                al_leg_id   = f"{fid}:{al_market}:{al_team}:{al_line}:alt"

                # Skip if already in main pool (same leg_id)
                if any(l.get("leg_id") == al_leg_id for l in scored):
                    continue

                # Score through ML model
                al_leg_raw = {
                    "leg_id":         al_leg_id,
                    "fixture_id":     fid,
                    "game":           main_leg.get("game", ""),
                    "sport":          al_sport,
                    "sport_key":      main_leg.get("sport_key", ""),
                    "market":         al_market,
                    "market_label":   mkt_label,
                    "pick":           al_team,
                    "point":          al_line,
                    "description":    al_desc,
                    "odds":           round(al_odds_dec, 3),
                    "implied_prob":   round(1.0 / al_odds_dec * 100, 2),
                    "game_time":      main_leg.get("game_time"),
                    "is_alt_line":    True,
                    "main_line_odds": main_leg.get("odds"),
                }
                try:
                    from parlay_builder import score_leg as _score_leg
                    al_scored = _score_leg(al_leg_raw)
                except Exception:
                    al_scored = {**al_leg_raw, "win_prob": al_leg_raw["implied_prob"],
                                  "ev": 0.0, "edge": 0.0, "grade": "C", "model_used": "implied"}

                # Score LQS
                # IMPORTANT: use market implied_prob as model_confidence, NOT the
                # ML model's win_prob. The global ML model was trained on main-market
                # legs and returns ~49% for any spread input regardless of odds —
                # it has no signal for alt lines. The market odds ARE the signal.
                alt_lqs = 0.0
                implied_conf = al_leg_raw["implied_prob"]  # e.g. 70% for 1.43x odds
                try:
                    _al_is_home  = al_team == main_leg.get("home_team")
                    _al_opponent = main_leg.get("away_team") if _al_is_home else main_leg.get("home_team")
                    al_cand = {
                        "market_type":      mkt_label,
                        "sport":            al_sport,
                        "team_or_player":   al_team,
                        "model_confidence": implied_conf,
                        "edge_pp":          0.0,
                        "line":             al_line,
                        "is_home":          _al_is_home,
                        "opponent":         _al_opponent,
                    }
                    lqs_r = lq_mod.compute_leg_quality_score(al_cand, db)
                    alt_lqs = lqs_r.get("lqs") or 0.0
                except Exception:
                    pass

                # Alt-spread gate: market implied probability is the only reliable signal.
                #
                # The ML model returns ~49% for any spread input regardless of odds —
                # it was trained on main-market legs and has no alt-spread signal.
                # Using model edge (49% vs 30% implied = +19pp "edge") creates false
                # positives: these legs beat real ML legs in ALE ranking (LOS 0.655
                # vs 0.530), replacing high-wp ML picks with 49%-wp alt spreads and
                # emptying Section A.
                #
                # Rule: for alternate_spreads, accept only if market implies ≥65%
                # coverage probability (1.54x or better odds).  These are genuine
                # cushion picks — the market is pricing a high-probability outcome.
                # Use implied_conf as win_prob (the market IS the signal here).
                #
                # For other alt markets (totals etc.) keep the original LQS gate.
                _is_alt_spread = al_market in ("alternate_spreads", "spreads")

                if _is_alt_spread:
                    _passes = implied_conf >= 65.0
                    # Use market implied prob as win_prob (honest signal for cushion lines)
                    al_scored["win_prob"] = implied_conf
                    al_scored["edge"]     = 0.0
                else:
                    main_lqs  = main_lqs_by_fixture.get(fid, 0.0)
                    _passes   = alt_lqs >= main_lqs + 3.0

                # Floor lqs at 62 for high-implied alt spreads so that LOS passes
                # the ALE _LOS_MIN=0.52 gate:
                #   LOS = implied*0.50 + 62/100*0.20 = 0.40 + 0.124 = 0.524
                # Without this, neutral Component A (50.0 — no personal data) gives
                # lqs=55, LOS=0.51 — just below the floor.
                if _is_alt_spread and implied_conf >= 65.0:
                    alt_lqs = max(alt_lqs, 62.0)

                if _passes:
                    al_scored["lqs"]       = alt_lqs
                    al_scored["_alt_lqs"]  = alt_lqs
                    al_scored["_main_lqs"] = main_lqs_by_fixture.get(fid, 0.0)
                    scored.append(al_scored)
                    n_added += 1
                    print(f"[alt-line] ✓ {al_desc} | implied={implied_conf:.0f}% odds={al_odds_dec:.3f}")

        print(f"[alt-line] evaluated={n_evaluated} added={n_added} / {len(unique_fixtures)} fixtures")

    except Exception as _alt_err:
        print(f"[alt-line] augmentation skipped: {_alt_err}")

    # Personal edge profile imports — must be before ALE (rescue pass inside ALE
    # uses _pep_available) AND before pool helpers are defined (they reference it).
    try:
        from personal_edge_profile import (
            get_max_legs_for_personal_profile as _max_legs_pep,
            normalize_sport  as _nsp_rec,
            normalize_market as _nmkt_rec,
            classify_line_bucket as _clf_rec,
        )
        _pep_available = True
    except Exception:
        _pep_available = False

    # ── ALE: per-fixture optimal line selection ───────────────────────────────
    # Replaces the single-direction "score the candidate leg" model with a
    # full per-game candidate sweep:
    #   • Both teams on every market (ML, spread, total, alt-spread if available)
    #   • Multiple spread points where bookmakers differ (e.g. -2.5 vs -3.0)
    #   • Favorite +1.5 / +2.5 cushion lines from alt_lines table when present
    # All candidates are ranked by LOS; the top line per fixture (plus an
    # optional 2nd of a different market type if LOS ≥ 0.57) enter the pool.
    #
    # Naive pick = the leg with the highest win_prob among main-market legs only
    # (what the old pipeline would have selected by default win_prob sort).
    # ale_switched=True when ALE chose a different line.
    _ALE_SECOND_LOS_MIN = 0.57   # 2nd market-type leg threshold (slightly above _LOS_MIN)
    n_ale_switched = 0   # initialise so ale_summary is always available at return
    n_ale_fav_plus = 0
    try:
        from parlay_builder import (
            evaluate_alt_lines as _run_ale,
            _classify_line     as _clf_line,
            _compute_los       as _los_fn,
        )
        from database import Fixture as _Fixture

        # Build fixture lookup — needed to access bookmakers JSON for alt markets
        _fix_by_id: dict[str, object] = {
            f.id: f for f in db.query(_Fixture).all()
        }

        # Group pre-scored main-market legs by fixture_id
        _by_fixture: dict[str, list] = {}
        _legs_no_fixture: list       = []
        for _leg in scored:
            _fid = _leg.get("fixture_id")
            if _fid:
                _by_fixture.setdefault(_fid, []).append(_leg)
            else:
                _legs_no_fixture.append(_leg)

        ale_pool:       list = []
        n_ale_fixtures  = len(_by_fixture)
        n_ale_switched  = 0
        n_ale_fav_plus  = 0   # times a +line was chosen for the market-favourite

        for _fid, _main_legs in _by_fixture.items():
            _fix = _fix_by_id.get(_fid)
            if not _fix:
                ale_pool.extend(_main_legs)
                continue

            # Fetch alt lines for this fixture (empty list if none stored today)
            _alt_raw: list = []
            try:
                _alt_raw = ct.get_scored_alt_lines(_fid, min_label="ALL")
            except Exception:
                pass

            # Full ALE ranking — all available lines for this fixture
            _ranked = _run_ale(_fix, _main_legs, _alt_raw, db, boost_pct=boost_pct)
            if not _ranked:
                ale_pool.extend(_main_legs)
                continue

            # Naive = highest win_prob among main-market (spread/ML/total) legs
            _main_only = [l for l in _main_legs
                          if _clf_line(l.get("market") or "") in
                          ("spread", "moneyline", "total")]
            _naive = (max(_main_only, key=lambda l: l.get("win_prob") or 0)
                      if _main_only else _main_legs[0])

            _best     = _ranked[0]
            _switched = _best.get("leg_id") != _naive.get("leg_id")
            if _switched:
                n_ale_switched += 1

            # Count times ALE chose a + line for the market favourite
            _bpt = _best.get("point")
            if _bpt is not None:
                try:
                    if float(_bpt) > 0:
                        n_ale_fav_plus += 1
                except (TypeError, ValueError):
                    pass

            _los_gain = round(_best["los"] - _los_fn(_naive), 4)

            _alts = [
                {
                    "description": _l.get("description"),
                    "pick":        _l.get("pick"),
                    "win_prob":    _l.get("win_prob"),
                    "edge":        _l.get("edge"),
                    "lqs":         _l.get("lqs"),
                    "los":         _l["los"],
                    "line_type":   _l["line_type"],
                }
                for _l in _ranked[1:4]
            ]

            _best["ale_considered"]      = True
            _best["ale_naive_pick"]      = _naive.get("description", "")
            _best["ale_switched"]        = _switched
            _best["ale_los_improvement"] = _los_gain
            _best["ale_alternatives"]    = _alts
            ale_pool.append(_best)

            # Optional second entry: best leg of a different market type
            _second = next(
                (
                    _l for _l in _ranked[1:]
                    if _clf_line(_l.get("market") or "") != _best["line_type"]
                    and _l["los"] >= _ALE_SECOND_LOS_MIN
                ),
                None,
            )
            if _second:
                _second["ale_considered"]      = True
                _second["ale_naive_pick"]      = _naive.get("description", "")
                _second["ale_switched"]        = False
                _second["ale_los_improvement"] = 0.0
                _second["ale_alternatives"]    = []
                ale_pool.append(_second)

        ale_pool.extend(_legs_no_fixture)

        # ── CUSHION ML direct injection ────────────────────────────────────────
        # MLB/NBA/NHL Moneylines for heavy favorites have edge=0 when the global
        # model is unavailable (win_prob = implied_prob).  This gives LOS ≈ 0.49
        # — below the 0.52 LOS floor — so these legs never enter _ranked, and
        # the ALE never considers them.  Yet they're CUSHION max=5 with personal
        # WR of 75-83%.  The personal WR IS the quality signal for these legs;
        # the model's silence doesn't make them bad picks.
        #
        # Direct injection: for any pre-ALE leg that is CUSHION (max_parlay_legs
        # >= 3) with win_prob >= 60% and is not already in ale_pool, inject it.
        # This ensures heavy favorites enter the confident filter and combo loop.
        if _pep_available:
            try:
                _ale_pool_ids: set = {l.get("leg_id") for l in ale_pool}
                _cushion_added = 0
                for _rfid, _pre_ale_legs in _by_fixture.items():
                    for _rl in sorted(_pre_ale_legs, key=lambda x: -(x.get("win_prob") or 0)):
                        _lid = _rl.get("leg_id", "")
                        if _lid in _ale_pool_ids:
                            continue  # already in pool
                        _wp = _rl.get("win_prob") or 0
                        if _wp < 55.0:
                            break  # legs are sorted desc by win_prob; no point continuing
                        mkt_n = _nmkt_rec(_rl.get("market_label") or _rl.get("market") or "")
                        bkt_n = _clf_rec(mkt_n, _rl.get("description", ""), _rl.get("point"))
                        ml    = _max_legs_pep(_nsp_rec(_rl.get("sport", "")), mkt_n, bkt_n)
                        if ml >= 3:  # CUSHION anchor (5) or supporting (3)
                            _ale_pool_ids.add(_lid)
                            ale_pool.append(_rl)
                            _cushion_added += 1
                if _cushion_added:
                    print(f"[ALE-cushion] injected {_cushion_added} CUSHION ML leg(s) (wp≥55%)")
            except Exception as _rescue_err:
                print(f"[ALE-cushion] skipped: {_rescue_err}")

        # ── Alt-spread direct injection ────────────────────────────────────────
        # High-implied alt spread legs (≥65% implied, is_alt_line=True) arrive
        # from the augmentation block with edge=0 by design.  Their LOS is
        # dominated by win_prob (0.50 weight) with no edge contribution —
        # even at 80% implied the LOS is only 0.524, right at the floor, and
        # any AVOID profile inherited from the opposite-direction bucket pushes
        # LOS to 0.0 via the margin multiplier.
        # Inject them directly here, the same way CUSHION ML legs are injected,
        # so they survive into the combo loop as 1-2 leg CLOSE-tier candidates.
        try:
            _ale_pool_ids_alt: set = {l.get("leg_id") for l in ale_pool}
            _alt_spread_added = 0
            for _rfid, _pre_ale_legs in _by_fixture.items():
                for _rl in _pre_ale_legs:
                    if not _rl.get("is_alt_line"):
                        continue
                    _lid = _rl.get("leg_id", "")
                    if _lid in _ale_pool_ids_alt:
                        continue  # already selected by ALE
                    _impl = _rl.get("implied_prob") or _rl.get("win_prob") or 0
                    if _impl < 65.0:
                        continue  # below cushion gate
                    # Ensure lqs floor is applied (may have been computed pre-floor-fix)
                    if (_rl.get("lqs") or 0) < 62.0:
                        _rl["lqs"]      = 62.0
                        _rl["_alt_lqs"] = 62.0
                    ale_pool.append(_rl)
                    _ale_pool_ids_alt.add(_lid)
                    _alt_spread_added += 1
            if _alt_spread_added:
                print(f"[ALE-altspread] injected {_alt_spread_added} alt-spread leg(s) (implied≥65%)")
        except Exception as _alt_inj_err:
            print(f"[ALE-altspread] injection skipped: {_alt_inj_err}")

        scored = ale_pool
        print(
            f"[ALE] {n_ale_fixtures} fixtures → {len(ale_pool)} optimal legs "
            f"| switched={n_ale_switched} | fav_plus={n_ale_fav_plus}"
        )

    except Exception as _ale_err:
        print(f"[ALE] pass skipped: {_ale_err}")

    # Re-sort after augmentation then apply authoritative quality gate:
    #   Main-line legs:  edge > 0  (model win_prob > market implied_prob)
    #                    AND win_prob >= per-sport threshold (or min_confidence override)
    #   Alt-line legs:   edge = 0 by design (win_prob == implied_prob)
    #                    — quality is ensured by alt_lqs >= main_lqs + 3
    scored.sort(key=lambda x: -x.get("win_prob", 0))

    def _confidence_threshold(sport_label: str) -> float:
        if min_confidence is not None:
            return min_confidence
        return _SPORT_CONFIDENCE_THRESHOLDS.get(_normalize_sport(sport_label), _DEFAULT_THRESHOLD)

    _legs_evaluated = len(scored)
    _legs_positive_ev = sum(
        1 for l in scored
        if not l.get("is_alt_line")
        and l.get("edge", -99) > 0
        and l.get("win_prob", 0) >= _confidence_threshold(l.get("sport", ""))
    )

    # CT-date filter removed: get_available_legs() already enforces commence_time > now,
    # so only future games enter the pool. The old strict "today only" filter blocked
    # all valid legs in the evening once today's games had started, leaving an empty pool.

    # Remove soccer Draw moneylines from the recommendation pool.
    # Draw is a valid 3-way market outcome but requires a dedicated soccer model
    # with draw-specific calibration. Without one, implied probs (~28-33%) are
    # poorly estimated and the legs drag down parlay win rates.
    scored = [l for l in scored if (l.get("pick") or "").lower() != "draw"]

    # ── Personal-profile pool helpers ─────────────────────────────────────────
    def _pep_max_legs_for(leg: dict) -> int:
        """Return personal max_parlay_legs for a leg (0 if AVOID, -1 if unknown)."""
        if not _pep_available:
            return -1
        try:
            sp_n = _nsp_rec(leg.get("sport", ""))
            mt_n = _nmkt_rec(leg.get("market_label") or leg.get("market") or "")
            bkt  = _clf_rec(mt_n, leg.get("description", ""), leg.get("point"))
            return _max_legs_pep(sp_n, mt_n, bkt)
        except Exception:
            return -1

    def _is_cushion_main_line(leg: dict) -> bool:
        """Return True if this main-line leg's personal profile is CUSHION.

        NOTE: do NOT check lqs here — at confident-filter time, main-line legs
        have lqs=None (LQS is computed in the pool LQS loop, which runs after
        the confident filter). The combo loop's _lqs_ok gate enforces the
        quality floor (lqs >= 62) for CUSHION legs before combo assembly.
        """
        if leg.get("is_alt_line"):
            return False  # alt lines handled by their own branch
        return _pep_max_legs_for(leg) >= 3  # CUSHION anchor=5 or supporting=3

    def _is_avoid_main_line(leg: dict) -> bool:
        """Return True if this main-line leg's personal profile is AVOID (max_legs=0).
        AVOID legs are blocked from the pool even when the model shows positive edge —
        personal win rate / margin grade overrides model signal on this market."""
        if leg.get("is_alt_line"):
            return False  # alt lines are already LOS-gated upstream
        return _pep_max_legs_for(leg) == 0  # AVOID grade → 0 legs allowed

    def _qualify_leg(leg: dict) -> str | None:
        """Assign a qualification tier requiring BOTH model_wp and personal_wr to agree.

        The prior max(model_wp, personal_wr) approach inflated tier assignments when
        one signal was strong and the other was near 50%.  Both signals must clear
        their respective thresholds to earn a tier; failing either → None (excluded).

        TIER_1: CUSHION + personal_wr ≥ 75% AND model_wp ≥ 65%   (both strongly agree)
        TIER_2: CUSHION + personal_wr ≥ 65% AND model_wp ≥ 58%   (both agree, moderate)
        TIER_3: CLOSE   + model_wp ≥ 68%  AND personal_wr ≥ 55%  (model drives, personal confirms)
        TIER_4: MIXED/CLOSE + model_wp ≥ 72%                     (model dominates, personal optional)

        Returns None → leg excluded from main-line pool by the pool entry gate.
        Alt-line legs always return None (quality-gated by LQS, not market tier).
        """
        if leg.get("is_alt_line"):
            return None

        pep_max     = _pep_max_legs_for(leg)
        model_wp    = (leg.get("win_prob") or 0.0)           # 0-100 scale
        personal_wr = leg.get("_personal_wr")                # 0-1 scale, may be None

        # TIER_1: CUSHION market, both signals strongly agree
        if pep_max >= 3 and personal_wr is not None:
            if personal_wr >= 0.75 and model_wp >= 65.0:
                return "TIER_1"

        # TIER_2: CUSHION market, both signals agree at moderate conviction
        if pep_max >= 3 and personal_wr is not None:
            if personal_wr >= 0.65 and model_wp >= 58.0:
                return "TIER_2"

        # TIER_3: CLOSE market, high model conviction + personal at least neutral.
        # If personal_wr is None (no history), raise model threshold to 70% to compensate.
        if pep_max == 2 and model_wp >= 68.0:
            if personal_wr is None and model_wp >= 70.0:
                return "TIER_3"
            if personal_wr is not None and personal_wr >= 0.55:
                return "TIER_3"

        # TIER_4: MIXED or CLOSE market, very high model conviction (personal not required).
        # This is the "model dominates" tier — reserved for ≥72% model WP only.
        if pep_max in (1, 2) and model_wp >= 72.0:
            return "TIER_4"

        return None  # below dual-signal threshold — excluded from main-line pool

    def _is_close_tier3(leg: dict) -> bool:
        """Tier 3: CLOSE market (pep_max==2) + model win_prob >= 65%.
        Model confidence overrides the conservative market grade."""
        if leg.get("is_alt_line"):
            return False
        return _pep_max_legs_for(leg) == 2 and (leg.get("win_prob") or 0) >= 65.0

    def _is_mixed_tier4(leg: dict) -> bool:
        """Tier 4: MIXED market (pep_max==1) + model win_prob >= 70%.
        Requires very high model confidence to enter pool."""
        if leg.get("is_alt_line"):
            return False
        return _pep_max_legs_for(leg) == 1 and (leg.get("win_prob") or 0) >= 70.0

    confident = [
        l for l in scored
        if l.get("is_alt_line")           # alt lines: quality-gated by LQS, not edge
        or _is_cushion_main_line(l)       # CUSHION profile legs: personal WR is the signal
        or _is_close_tier3(l)             # Tier 3: CLOSE + model wp>=65%
        or _is_mixed_tier4(l)             # Tier 4: MIXED + model wp>=70%
        or (
            l.get("edge", -99) > 0
            and l.get("win_prob", 0) >= _confidence_threshold(l.get("sport", ""))
            and not _is_avoid_main_line(l)  # AVOID-graded legs blocked regardless of model edge
        )
    ]
    # Split confident legs into two buckets so CUSHION main-line legs (MLB ML,
    # NHL ML, NBA ML, Soccer ML, totals) are always represented in the pool.
    # Without this, the sort places alt lines first (they have _alt_lqs > 0)
    # and all 30 slots fill with alt lines — CUSHION main-line legs never enter,
    # which means no leg has max_parlay_legs >= 3, making 3+ leg combos impossible.
    _alt_pool     = sorted([l for l in confident if l.get("is_alt_line")],
                            key=lambda x: -x.get("_alt_lqs", 0))
    # Sort CUSHION main-line legs by EV (positive-value underdogs first), not win_prob.
    # Sorting by win_prob would always prefer short-odds favorites whose combined EV
    # is negative; an MLB +245 underdog with ev=+0.68 should outrank a -270 favorite.
    # Also stamp each CUSHION leg with its personal_wr from the profile so the combo
    # loop can use it as a win_prob proxy when the global model falls back to implied_prob
    # (e.g. sklearn version mismatch makes clf=None → win_prob = implied_prob for MLB h2h).
    # Step 1: collect all CUSHION main-line legs and stamp personal_wr FIRST.
    # personal_wr is used as effective WP proxy when the global model falls back to
    # implied_prob (e.g. sklearn version mismatch → MLB h2h gets 29% implied for +245,
    # but personal history says 75% WR → effective WP = max(29, 75) = 75%).
    _cushion_pool_all = [l for l in confident if not l.get("is_alt_line")
                          and _is_cushion_main_line(l)]
    for _cl in _cushion_pool_all:
        if _cl.get("_personal_wr") is None:
            try:
                from personal_edge_profile import lookup_personal_profile as _lpp
                _sp_n  = _nsp_rec(_cl.get("sport", ""))
                _mt_n  = _nmkt_rec(_cl.get("market_label") or _cl.get("market") or "")
                _bkt_n = _clf_rec(_mt_n, _cl.get("description", ""), _cl.get("point"))
                _prow  = _lpp(_sp_n, _mt_n, _bkt_n)
                if _prow and _prow.get("personal_wr") is not None:
                    _cl["_personal_wr"]     = float(_prow["personal_wr"])
                    _cl["_personal_sample"] = _prow.get("sample_size") or _prow.get("n_bets")
            except Exception:
                pass
    # Step 2: pre-filter using EFFECTIVE WP = max(model_wp, personal_wr * 100).
    # This allows +EV underdogs like Orioles +245 (model_wp=29%, personal_wr=75%) to
    # pass the 48% floor, while still removing true longshots with no proven history.
    _CUSHION_MIN_WP = 48.0
    _cushion_pool_raw = [
        l for l in _cushion_pool_all
        if max((l.get("win_prob") or 0), (l.get("_personal_wr") or 0) * 100) >= _CUSHION_MIN_WP
    ]
    # Sort by win_prob descending (primary), then ev descending (tiebreaker).
    # Sorting by ev alone caused soccer h2h sub-model to assign ~49% to +400 longshots,
    # generating fake ev=1.4+ that monopolized all 12 slots.
    _cushion_pool = sorted(_cushion_pool_raw, key=lambda x: (-(x.get("win_prob") or 0), -(x.get("ev") or 0)))
    # Build CUSHION pool with sport-market diversity.
    # Problem: sorting by WP alone lets soccer totals (82-95%) take all 12 slots,
    # pushing out MLB/NBA/NHL ML legs at 48-75%.
    # Problem: sorting by EV alone lets soccer h2h sub-model longshots (ev=1.4 at
    # wp=49%) monopolize slots while Orioles +245 (ev=0.70, wp=48.7%) is excluded.
    # Solution: group legs into (sport × market_type) buckets; within each bucket sort
    # by EV descending (so underdog value legs like Orioles rank above break-even favs
    # like Yankees -270); then interleave buckets by their best-leg WP (soccer totals
    # first, then NBA ML, then MLB ML, etc.); cap at 3 per bucket, 12 total.
    _cushion_buckets: dict[str, list] = {}
    for _cl in _cushion_pool:
        _sp_key = _nsp_rec(_cl.get("sport", ""))
        _mt_key = _nmkt_rec(_cl.get("market_label") or _cl.get("market") or "")
        _bkt_key = f"{_sp_key}|{_mt_key}"
        _cushion_buckets.setdefault(_bkt_key, []).append(_cl)
    # Within each bucket, sort by effective EV descending.
    # When the global model fails (ev=0.0 for MLB/NBA h2h), compute EV from personal_wr
    # and actual odds so Orioles +245 (personal_wr=0.75, ev=0.875) outranks Yankees -270
    # (personal_wr=0.75, ev=-0.0625).  ev_effective = max(ev, personal_wr*(odds-1)-(1-wr)).
    def _eff_ev(leg: dict) -> float:
        model_ev = leg.get("ev") or 0.0
        wr = leg.get("_personal_wr")
        if wr is not None:
            odds = leg.get("odds") or 1.0
            wr_ev = wr * (float(odds) - 1.0) - (1.0 - wr)
            return max(model_ev, wr_ev)
        return model_ev
    for _bk in _cushion_buckets:
        _cushion_buckets[_bk].sort(key=lambda x: -_eff_ev(x))
    # Assemble CUSHION pool with per-bucket quotas so no single sport monopolizes slots.
    # Each bucket gets a quota based on its profile sample size:
    #   n >= 20 → quota 3  (well-proven: MLB ML n=20, Soccer ML n=75, Soccer Total n=111)
    #   n >= 8  → quota 2  (moderate: NBA ML n=8)
    #   n < 8   → quota 1  (small sample: NHL ML n=6)
    # After quotas are filled, remaining slots go to any bucket's overflow (sorted by WP).
    # Total CUSHION pool target: min(20, total qualifying CUSHION legs).
    # Pool is then padded with alt lines: total = up to 30 legs.
    _CUSHION_TOTAL = 20
    _quota_used: dict[str, int] = {}
    _cushion_diverse: list[dict] = []
    # First pass: fill per-bucket quotas
    for _bk in _cushion_buckets:
        _legs = _cushion_buckets[_bk]
        # Determine quota from profile sample size (stored on first leg if stamped)
        _sample = next(
            (l.get("_personal_sample") for l in _legs if l.get("_personal_sample") is not None),
            None,
        )
        # Fall back: count legs as proxy for sample size if _personal_sample not stamped
        _n = _sample if _sample is not None else len(_legs)
        _quota = 3 if _n >= 20 else (2 if _n >= 8 else 1)
        _added = 0
        for _cl in _legs:
            if _added < _quota:
                _cushion_diverse.append(_cl)
                _added += 1
        _quota_used[_bk] = _added
    # Second pass: fill remaining slots from overflow (any bucket, sorted by eff_ev)
    if len(_cushion_diverse) < _CUSHION_TOTAL:
        _overflow = []
        for _bk in _cushion_buckets:
            quota = _quota_used.get(_bk, 0)
            _overflow.extend(_cushion_buckets[_bk][quota:])
        _overflow.sort(key=lambda x: -_eff_ev(x))
        for _cl in _overflow:
            if len(_cushion_diverse) >= _CUSHION_TOTAL:
                break
            _cushion_diverse.append(_cl)
    _cushion_pool_final = _cushion_diverse

    # Tier 3/4 pool: CLOSE (pep_max==2, wp>=65%) and MIXED (pep_max==1, wp>=70%) legs.
    # These pass the confident filter via _is_close_tier3()/_is_mixed_tier4() but are NOT
    # CUSHION, so they don't enter _cushion_pool_all. Collect them separately.
    # In the combo loop: CLOSE legs can form 1-2 leg combos, MIXED legs only singles.
    _tier34_pool = sorted(
        [l for l in confident
         if not l.get("is_alt_line")
         and not _is_cushion_main_line(l)
         and (_is_close_tier3(l) or _is_mixed_tier4(l))],
        key=lambda x: -(x.get("win_prob") or 0),
    )

    # Combine: CUSHION main-line (up to 20) + Tier 3/4 (up to 6) + alt lines (up to 6) = max 32.
    _n_cushion = min(_CUSHION_TOTAL, len(_cushion_pool_final))
    _n_t34     = min(6, len(_tier34_pool))
    _n_alt     = min(32 - _n_cushion - _n_t34, len(_alt_pool))
    pool = _cushion_pool_final[:_n_cushion] + _tier34_pool[:_n_t34] + _alt_pool[:_n_alt]

    # Stamp qualification_tier on every pool leg (used at generation time to store
    # tier in mock_bet_legs for downstream signal analysis).
    for _pl in pool:
        if _pl.get("qualification_tier") is None:
            _pl["qualification_tier"] = _qualify_leg(_pl)

    # Pool entry gate: main-line legs that fail the dual-signal threshold are removed.
    # Alt-line legs are exempt — they are quality-gated by LQS, not market tier.
    # This enforces tier qualification uniformly across all main-line entry paths
    # (CUSHION, CLOSE, MIXED) rather than only stamping a label post-assembly.
    _pre_gate_n = len(pool)
    pool = [
        l for l in pool
        if l.get("is_alt_line")                       # alt lines: LQS is the gate
        or l.get("qualification_tier") is not None    # main lines: must earn a tier
    ]
    _gate_removed = _pre_gate_n - len(pool)
    if _gate_removed:
        print(f"[pool-gate] Removed {_gate_removed} main-line legs below dual-signal threshold "
              f"({_pre_gate_n} → {len(pool)} pool legs)")

    # Section A sim WR floor: exclude legs where simulation shows < 15% win rate
    # (>= 10 settled legs for that team/market). These are consistently failing
    # combinations that shouldn't be served as top picks regardless of LQS.
    def _passes_sim_floor(leg: dict) -> bool:
        sport  = leg.get("sport", "")
        market = leg.get("market", "")
        team   = leg.get("pick", "")
        if not team or not sport or not market:
            return True
        try:
            wr, n = _swr_fn(None, sport, market, team)
            if wr is not None and n >= 10 and wr < 0.15:
                print(f"[pool-filter] Excluded {team} {market} — sim WR={wr:.0%} ({n} legs)")
                return False
        except Exception:
            pass
        return True

    pool = [l for l in pool if _passes_sim_floor(l)]

    # ── Form-based parlay-block filter (CUSHION legs only) ────────────────────
    # CUSHION legs with poor recent form AND a strong opponent are flagged as
    # parlay-ineligible for today. They remain in the pool for single-leg tracking
    # (tier_b/c/d) but are skipped in the section_a combo loop.
    # Poor form: form_rate < 0.35 (less than 35% wins in last 10 games)
    # Strong opp: e1_opponent < 40 (opponent win rate ≥ 60% last 30 days)
    for _pl in pool:
        if not _is_cushion_main_line(_pl):
            continue
        try:
            _mc  = _pl.get("matchup_context") or {}
            _fr  = _mc.get("form_rate")
            _e1  = _mc.get("e1_opponent")
            if _fr is not None and _e1 is not None:
                if _fr < 0.35 and _e1 < 40:
                    _pl["_poor_form_today"] = True
        except Exception:
            pass

    if not pool:
        return _build_tiered_fallback(
            scored, [], stake, db,
            _legs_evaluated, _legs_positive_ev,
        )

    # ── Ensure all pool legs have an LQS score ────────────────────────────────
    # Main-line legs get LQS computed here (alt lines were scored in the augmentation block).
    for l in pool:
        if l.get("lqs") is not None:
            continue
        try:
            _is_home_p  = l.get("pick") == l.get("home_team")
            _opponent_p = l.get("away_team") if _is_home_p else l.get("home_team")
            _lqs_cand = {
                "market_type":      l.get("market_label") or l.get("market"),
                "sport":            l.get("sport"),
                "team_or_player":   l.get("pick"),
                "model_confidence": l.get("win_prob"),
                "model_used":       l.get("model_used"),
                "edge_pp":          l.get("edge"),
                "line":             l.get("point"),
                "is_home":          _is_home_p,
                "opponent":         _opponent_p,
            }
            _r = lq_mod.compute_leg_quality_score(_lqs_cand, db)
            l["lqs"] = _r.get("lqs")
        except Exception as _lqs_err:
            print(f"[pool-lqs] WARNING: LQS computation failed for {l.get('pick')} "
                  f"({l.get('sport')} {l.get('market_label') or l.get('market')}): {_lqs_err}")
            # Fallback: assign a neutral LQS of 62 so CUSHION legs aren't silently
            # blocked by the (None or 0) >= 62 check in the combo loop.
            l["lqs"] = 62.0

    # ── Per-depth LQS minimums (prevents padding deeper parlays with weak legs) ──
    # Targets: 5-leg 53%, 4-leg 53%, 3-leg 57%, 2-leg 65% — all require quality legs.
    # depth_lqs_min overrides the floor when set (e.g. 55.0 for paper trading).
    if depth_lqs_min is not None:
        _DEPTH_LQS_MIN: dict[int, float] = {2: depth_lqs_min, 3: depth_lqs_min + 2,
                                             4: depth_lqs_min + 4, 5: depth_lqs_min + 5}
    else:
        _DEPTH_LQS_MIN: dict[int, float] = {2: 62.0, 3: 64.0, 4: 66.0, 5: 68.0}

    # Hard cap: never exceed 5 legs (historical collapse at 6+, 12.7% WR)
    max_legs = min(max_legs, 5)

    def _leg_max_parlay(l: dict) -> int:
        """Return max_parlay_legs for a single leg from its personal profile."""
        if not _pep_available:
            return 5  # no profile data: allow through
        result = _max_legs_pep(
            _nsp_rec(l.get("sport", "")),
            _nmkt_rec(l.get("market_label") or l.get("market") or ""),
            _clf_rec(
                _nmkt_rec(l.get("market_label") or l.get("market") or ""),
                l.get("description", ""),
                l.get("point"),
            ),
        )
        # Alt spread legs with a positive point (underdog cushion lines like +1.5/+2.5)
        # are graded CLOSE by the personal profile (max_legs=2). This is too conservative:
        # alt spreads have 63% WR and 54.7% CLV beat-close rate — stronger than most
        # main-line markets. Raise positive-point alt spreads to supporting CUSHION (3)
        # so they can combine with CUSHION ML legs in 3-leg cross-market parlays.
        # They are already gated by implied_conf >= 65% at pool entry (ALE LOS filter).
        if l.get("is_alt_line"):
            _pt = l.get("point")
            if _pt is not None:
                try:
                    if float(_pt) > 0:
                        return max(result, 3)  # at least supporting CUSHION — allow 3-leg combos
                except (TypeError, ValueError):
                    pass
        return result

    # ── Pre-compute personal_wr for each pool leg ─────────────────────────────
    # Needed for the CUSHION 2-leg wp floor check (requires personal_wr ≥ 0.80).
    # Done once here to avoid per-combo DB lookups in the inner combo loop.
    _pool_wr: dict = {}  # leg_id → personal_wr (float or None)
    if _pep_available:
        try:
            from personal_edge_profile import lookup_personal_profile as _lpp
            for _pl in pool:
                _pid = _pl.get("leg_id", id(_pl))
                _prof = _lpp(
                    _nsp_rec(_pl.get("sport", "")),
                    _nmkt_rec(_pl.get("market_label") or _pl.get("market") or ""),
                    _clf_rec(
                        _nmkt_rec(_pl.get("market_label") or _pl.get("market") or ""),
                        _pl.get("description", ""),
                        _pl.get("point"),
                    ),
                )
                _pool_wr[_pid] = _prof.get("personal_wr") if _prof else None
        except Exception:
            pass

    # ── Build all parlay combinations — priority order: 5→4→3→2→1 ────────────
    # For each combo: max allowed size = min(max_parlay_legs across all legs).
    # AVOID legs (max_parlay_legs=0): blocked from all combos.
    # CLOSE/MIXED legs (max_parlay_legs=1): 1-leg only.
    # CUSHION supporting legs (max_parlay_legs=3): can appear in 2/3-leg.
    # CUSHION anchor legs (max_parlay_legs=5): can appear in any size up to 5.
    candidates = []
    used_sigs  = set()

    # Try larger parlays first — prefer 4-5 leg combos (best historical P&L)
    _sizes = sorted(range(min_legs, max_legs + 1), reverse=True)

    for size in _sizes:
        if len(pool) < size:
            continue

        combos = list(itertools.combinations(range(len(pool)), size))
        # Limit combinations for performance.
        # For larger sizes (≥4), random-sample instead of head-truncating so that
        # CUSHION anchor legs spread throughout the pool (not just first N indices)
        # get a fair chance to appear in deep combinations.
        _combo_limit = 5000 if size >= 4 else 2000
        if len(combos) > _combo_limit:
            import random as _rnd
            combos = _rnd.sample(combos, _combo_limit)

        # Non-CUSHION win_prob floors by parlay size (model-edge legs).
        _WIN_PROB_MIN = {5: 58.0, 4: 57.0, 3: 55.0, 2: 55.0}
        _min_wp_for_size = _WIN_PROB_MIN.get(size, 55.0)

        # CUSHION win_prob floors by parlay size.
        # For CUSHION legs the personal WR (≥75%) is the primary signal — the global
        # model assigns ~50% to all h2h legs and has no real MLB/NHL/NBA ML signal.
        # Requiring 55% on CUSHION 2-leg blocks every valid CUSHION leg.
        # Floor lowered to 48% for 2-leg (implied-prob parity for +105 underdogs).
        # 5-leg: 55% (no exceptions at max depth).
        # 4-leg: 53%. 3-leg: 51%. 2-leg: 48%.
        _CUSHION_WP_FLOORS = {5: 55.0, 4: 53.0, 3: 51.0, 2: 48.0}
        _cushion_wp_floor  = _CUSHION_WP_FLOORS.get(size, 48.0)

        for combo in combos:
            legs = [pool[i] for i in combo]

            # Form block: skip combo if any leg is flagged as poor-form today.
            if any(l.get("_poor_form_today") for l in legs):
                continue

            # Personal profile parlay tier enforcement.
            # Compute max_parlay_legs for every leg once; reuse for LQS, tier, and WP checks.
            try:
                _leg_maxes = [_leg_max_parlay(l) for l in legs]
                _max_allowed = min(_leg_maxes)

                # Base gate: combo size must not exceed any leg's max
                if size > _max_allowed:
                    continue

                # Depth-specific anchor requirements:
                #   5-leg → ALL legs must be anchor tier (max_parlay_legs = 5)
                #   4-leg → at least 3 anchor legs (1 supporting CUSHION allowed)
                #   3-leg → at least 2 legs must be CUSHION (max_parlay_legs >= 3)
                if size == 5 and _pep_available:
                    if any(m < 5 for m in _leg_maxes):
                        continue
                elif size == 4 and _pep_available:
                    if sum(1 for m in _leg_maxes if m >= 5) < 3:
                        continue
                elif size == 3 and _pep_available:
                    if sum(1 for m in _leg_maxes if m >= 3) < 2:
                        continue
            except Exception:
                _leg_maxes = [-1] * len(legs)  # unknown — allow through with depth floor

            # Win_prob gate (uses _leg_maxes computed above):
            #   CUSHION legs (pep_max ≥ 3): size-dependent floor (see _CUSHION_WP_FLOORS).
            #     For 2-leg: 48% floor (model has no h2h signal; personal WR is the gate).
            #     When the global model fails (e.g. sklearn version mismatch), MLB h2h
            #     win_prob falls back to implied_prob (29% for +245), which is below 48%.
            #     In that case, use personal_wr (stamped on pool assembly) as proxy.
            #   Non-CUSHION: standard _WIN_PROB_MIN floor (55-58%).
            def _wp_floor_for(leg, idx):
                _ml = _leg_maxes[idx] if idx < len(_leg_maxes) else -1
                if _ml < 3:
                    return _min_wp_for_size  # non-CUSHION: strict floor
                return _cushion_wp_floor  # CUSHION: personal WR is signal, not model edge

            def _effective_wp(leg, idx):
                wp = (leg.get("win_prob") or 0)
                _ml = _leg_maxes[idx] if idx < len(_leg_maxes) else -1
                if _ml >= 3:
                    # For CUSHION legs, use personal_wr * 100 as floor proxy if it's higher.
                    # This prevents model fallback to implied_prob from blocking +EV underdogs.
                    _pwr = leg.get("_personal_wr")
                    if _pwr is not None:
                        wp = max(wp, _pwr * 100)
                return wp

            if any(
                _effective_wp(l, i) < _wp_floor_for(l, i)
                for i, l in enumerate(legs)
            ):
                continue

            # Per-depth LQS minimum — CUSHION legs use a flat floor of 55;
            # non-CUSHION legs use the depth-scaled floor from _DEPTH_LQS_MIN.
            depth_floor = _DEPTH_LQS_MIN.get(size, 62.0)
            _lqs_ok = all(
                (l.get("lqs") or 0) >= (55.0 if (_leg_maxes[i] if i < len(_leg_maxes) else -1) >= 3 else depth_floor)
                for i, l in enumerate(legs)
            )
            if not _lqs_ok:
                continue

            # Enforce: max 1 leg per fixture (no same-game parlays)
            fixture_ids = [l["fixture_id"] for l in legs]
            if len(fixture_ids) != len(set(fixture_ids)):
                continue

            # Enforce: no same team appearing twice across different fixtures
            team_names = [l.get("pick", "").strip().lower() for l in legs]
            real_teams = [t for t in team_names if t not in ("over", "under", "")]
            if len(real_teams) != len(set(real_teams)):
                continue

            # Build parlay
            combined_odds = 1.0
            combined_win  = 1.0
            for l in legs:
                combined_odds *= l["odds"]
                # Use personal_wr when available (calibrated historical rate).
                # This is critical for legs where the global model fails (e.g. MLB ML
                # when sklearn version mismatch → win_prob = implied_prob ~28-51%).
                # personal_wr reflects actual bet outcomes and is the authoritative
                # signal for CUSHION legs.
                _pwr = l.get("_personal_wr")
                if _pwr is not None:
                    combined_win *= _pwr
                else:
                    combined_win *= (l["win_prob"] / 100)

            # Pre-construction combined WP floor for 3+ leg parlays.
            # Target: 40% bet WR on 4-leg parlays requires ~80% per-leg WP.
            # Any 3+ leg combo with combined WP < 30% cannot reach the WR target
            # and should not be constructed regardless of other filters.
            # 0.74^4 ≈ 0.30 — the floor implies each leg must average ≥74% for 4-leg.
            # Size-dependent: 3-leg requires ≥30% (0.67^3=0.30), 4-leg ≥30%, 5-leg ≥20%.
            _COMBINED_WP_FLOOR = {3: 0.30, 4: 0.30, 5: 0.20}
            if size >= 3 and combined_win < _COMBINED_WP_FLOOR.get(size, 0.30):
                continue

            # Per-size minimum payout threshold.
            # 2-leg uses 1.5 so that two 55%+ CUSHION favorites (e.g. -350/-385)
            # still qualify even when their combined decimal odds are below 2.0.
            _MIN_ODDS_BY_SIZE = {5: 3.0, 4: 2.5, 3: 2.0, 2: 1.5}
            _effective_min_odds = _MIN_ODDS_BY_SIZE.get(size, min_odds)
            if combined_odds < _effective_min_odds:
                continue

            # Pre-filter: use conservative 0.80 per-leg rate to let more through.
            # Final grading uses per-market historical rates (strategy.py).
            # Use strict "< 0" (not <= 0) so that breakeven cushion lines at
            # 1.250 odds (EV exactly 0.0 at 80% pre_win) are allowed through.
            pre_win_check  = 0.80 ** size
            payout_check   = (combined_odds - 1) * stake
            exp_profit_chk = pre_win_check * payout_check - (1 - pre_win_check) * stake
            if exp_profit_chk < 0:
                continue

            sig = frozenset(l["leg_id"] for l in legs)
            if sig in used_sigs:
                continue
            used_sigs.add(sig)

            # Use actual per-leg win probabilities — not the hardcoded model accuracy.
            payout          = (combined_odds - 1) * stake
            expected_profit = combined_win * payout - (1 - combined_win) * stake

            candidates.append({
                "legs":              legs,
                "combined_odds":     round(combined_odds, 3),
                "combined_win_prob": round(combined_win * 100, 2),  # actual product of per-leg probs
                "model_win_prob":    round(combined_win * 100, 2),  # same; kept for backwards compat
                "payout":            round(payout, 2),
                "expected_profit":   round(expected_profit, 2),
                "n_legs":            size,
            })

    if not candidates:
        return _build_tiered_fallback(
            scored, pool, stake, db,
            _legs_evaluated, _legs_positive_ev,
            depth_lqs_min=depth_lqs_min,
        )

    # ── Sort by strategy type then quality ────────────────────────────────────
    for c in candidates:
        try:
            strategy_legs = []
            for l in c["legs"]:
                am = l.get("american_odds_raw") or (
                    round((l["odds"]-1)*100) if l["odds"] >= 2
                    else -round(100/(l["odds"]-1))
                )
                mkt = l.get("market_label", "Moneyline")
                strategy_legs.append({"market": mkt, "american_odds": am})

            cam = round((c["combined_odds"]-1)*100) if c["combined_odds"] >= 2                   else -round(100/(c["combined_odds"]-1))

            graded              = strat.grade_bet(strategy_legs, c["combined_odds"], cam, stake)
            c["bet_class"]      = graded["bet_class"]
            c["grade"]          = graded["grade"]
            c["grade_color"]    = graded["color"]
            c["grade_summary"]  = graded["summary"]
            c["strategy_note"]  = graded["strategy_note"]
            c["hist_win_prob"]  = graded["win_prob"]
            c["hist_ev"]        = graded["ev"]
            c["kelly_500"]      = graded["kelly_500"]
            c["combined_am"]    = cam
        except Exception:
            # If grading fails, assign safe defaults so pick still shows
            c.setdefault("bet_class",     "mixed")
            c.setdefault("grade",         "C")
            c.setdefault("grade_color",   "amber")
            c.setdefault("grade_summary", "")
            c.setdefault("strategy_note", "")
            c.setdefault("hist_win_prob", c.get("combined_win_prob", 0))
            c.setdefault("hist_ev",       c.get("expected_profit", 0))
            c.setdefault("kelly_500",     0)
            cam = round((c["combined_odds"]-1)*100) if c["combined_odds"] >= 2                   else -round(100/(c["combined_odds"]-1))
            c.setdefault("combined_am", cam)

    # Save candidate snapshot for regime shadow run (before standard sort modifies _composite)
    _candidates_shadow = list(candidates)

    # Composite sort: avg effective win_prob (50%) + avg LQS (30%) + normalised EV (20%).
    # Uses avg per-leg win_prob (not combined product) so 5×65% scores the same as 2×65%.
    # Combined product collapses to ~2.8% for 5×49% legs (indistinguishable from noise).
    # For CUSHION legs (model unavailable → win_prob ≈ implied_prob), use effective WP:
    #   effective_wp = max(model_wp, personal_wr * 100)
    # This ensures MLB ML at +245 (personal_wr=75%, model_wp=29%) ranks alongside
    # soccer totals (sub-model WP 82-95%) rather than always being crowded out.
    _cand_max_ev = max(abs(c["expected_profit"]) for c in candidates) or 1.0

    def _eff_wp_for_leg(leg: dict) -> float:
        """Effective WP for composite scoring.

        For CUSHION legs (personal_wr stamped): use personal_wr * 100 directly.
        This normalizes all CUSHION legs to their PROVEN historical win rate rather
        than unreliable model outputs:
          - Soccer total sub-model assigns 82-95% (overconfident vs 70% personal WR)
          - Global model fails for MLB/NBA h2h → implied_prob 40-75%
        Using personal_wr gives a consistent, calibrated score across all CUSHION legs.
        Without this fix, soccer totals (95% sub-model) always crowd out MLB ML (75% WR).
        """
        pwr = leg.get("_personal_wr")
        if pwr is not None:
            return pwr * 100  # use calibrated personal WR for all CUSHION legs
        return leg.get("win_prob") or 0  # non-CUSHION: raw model WP

    # Pre-compute max boost lift for normalization across the candidate pool.
    # Inline version of the boost eligibility check (can't import mock_bets —
    # mock_bets imports recommender, creating a circular dependency).
    # Lift formula: boost_pct * stake * (combined_odds - 1) * (win_prob / 100)
    _BOOST_STAKE = 10.0

    def _best_boost_lift(legs: list, c_odds: float, c_wp: float) -> float:
        """Return EV lift from the best eligible boost tier for this candidate.

        is_single_sport uses _normalize_sport so that EPL + Bundesliga + Ligue 1
        all count as 'Soccer' and qualify for +50% Route B together.
        """
        _sgp = len({l.get("fixture_id", "") for l in legs if l.get("fixture_id")}) == 1 and len(legs) >= 2
        _ss  = len({_normalize_sport(l.get("sport") or "") for l in legs if l.get("sport")}) <= 1
        _n   = len(legs)
        _wp  = c_wp / 100.0
        _profit = _BOOST_STAKE * (c_odds - 1.0)
        # Check tiers highest-first: return lift for first eligible
        for _pct, _ok in [
            (0.50, (_sgp and _n >= 3) or (_ss and not _sgp and c_odds >= 2.00)),
            (0.30, c_odds >= 1.50 or _n >= 3),
            (0.25, c_odds >= 1.50),
        ]:
            if _ok:
                return round(_wp * _profit * _pct, 3)  # ev_lift = win_prob * profit * boost_pct
        return 0.0

    _boost_lifts = [_best_boost_lift(_c["legs"], _c["combined_odds"], _c["combined_win_prob"]) for _c in candidates]
    _max_boost_lift = max(_boost_lifts) if _boost_lifts else 1.0

    for _i, _c in enumerate(candidates):
        _raw_lqs    = sum(l.get("lqs") or 50 for l in _c["legs"]) / max(len(_c["legs"]), 1)
        _avg_wp     = sum(_eff_wp_for_leg(l) for l in _c["legs"]) / max(len(_c["legs"]), 1)
        _norm_ev    = max(_c["expected_profit"], 0) / _cand_max_ev
        _norm_boost = _boost_lifts[_i] / (_max_boost_lift or 1.0)
        # Weights: avg_wp 47%, avg_lqs 28%, norm_ev 19%, best_boost_ev 6%
        # The boost term naturally promotes same-sport CUSHION clusters:
        #   same-sport +100 → +50% eligible → higher lift → higher composite
        #   cross-sport     → +30% max      → lower lift  → lower composite
        _c["_composite"] = (
            (_avg_wp  / 100) * 0.47 +
            (_raw_lqs / 100) * 0.28 +
            _norm_ev          * 0.19 +
            _norm_boost        * 0.06
        )
        _c["_avg_lqs"] = _raw_lqs  # cache for depth-first selection below
        _c["_avg_wp"]  = _avg_wp   # cache for section_a sort key
    candidates.sort(key=lambda x: -x["_composite"])

    # ── Section A: depth-first CUSHION pick selection ─────────────────────────
    # When CUSHION legs dominate the pool, the composite score (40% win_prob)
    # always favours 2-leg parlays (42% combined) over 4-5 leg (10-20% combined).
    # This pass selects the BEST combo per depth bucket (5→4→3→2) so Section A
    # always includes the deepest valid pick alongside a high-WR shorter combo.
    #
    # Scoring within each bucket: avg_lqs (model-independent quality proxy).
    # Diversity: each leg_id may appear at most once across Section A picks.
    # Top 3 per size, then assemble: primary=largest bucket, secondary=smaller.
    _sa_by_size: dict[int, list] = {}
    for _c in candidates:
        _sa_by_size.setdefault(_c["n_legs"], []).append(_c)

    section_a_picks: list = []
    _sa_leg_used: dict[str, int] = {}  # leg_id → appearances in section_a

    for _sz in sorted(_sa_by_size.keys(), reverse=True):   # 5 → 4 → 3 → 2
        _bucket = _sa_by_size[_sz]
        # Sort bucket by composite (50% avg_win_prob + 30% avg_lqs) × same-sport bonus.
        # avg_win_prob is dominant so high-LQS/low-WP combos don't outrank genuine picks.
        def _sa_sort_key(x):
            _lqs    = x.get("_avg_lqs", 0)
            _wp     = x.get("_avg_wp", 0)
            _sports = {l.get("sport", "") for l in x.get("legs", [])}
            _bonus  = 1.05 if len(_sports) == 1 else 1.0
            _base   = (_wp / 100) * 0.50 + (_lqs / 100) * 0.30
            return -(_base * _bonus)
        _bucket_sorted = sorted(_bucket, key=_sa_sort_key)
        _added_this_sz = 0
        _sa_sigs: set = set()
        for _bc in _bucket_sorted:
            sig = frozenset(l["leg_id"] for l in _bc["legs"])
            if sig in _sa_sigs:
                continue
            # Each leg may appear in at most 1 Section A pick (strict diversity)
            if any(_sa_leg_used.get(lid, 0) >= 1 for lid in sig):
                continue
            # Near-duplicate: skip if >50% overlap with any section_a pick already added
            if any(
                len(sig & frozenset(l["leg_id"] for l in _ep["legs"])) / max(len(sig), 1) > 0.5
                for _ep in section_a_picks
            ):
                continue

            # Accept
            _sa_sigs.add(sig)
            for lid in sig:
                _sa_leg_used[lid] = _sa_leg_used.get(lid, 0) + 1
            _fp = _format_pick(_bc, stake, db)
            _fp["confidence_tier"] = "A"
            _fp["tier_label"]      = "Sharp Picks"
            _fp["show_warning"]    = False
            _fp["section_a_depth"] = _sz   # tag with depth for UI
            section_a_picks.append(_fp)
            _added_this_sz += 1
            # Per-depth cap: 5-leg → 3, 4-leg → 2, 3-leg → 2, 2-leg → 3
            _SA_SIZE_CAP = {5: 3, 4: 2, 3: 2, 2: 3}
            if _added_this_sz >= _SA_SIZE_CAP.get(_sz, 3):
                break

    # Final content-based dedup: identical parlay combos can appear across depth
    # buckets when the same alt-line leg has two leg_id variants (ALE-sourced vs
    # CUSHION-injected), or when the game label string differs slightly between
    # sources.  Key on sorted (pick, point) tuples only — order-independent and
    # robust to game-label formatting differences.
    _seen_parlay_content: set = set()
    _deduped_section_a: list = []
    for _sp in section_a_picks:
        _content_key = tuple(sorted(
            (l.get("pick", ""), str(l.get("point", 0)))
            for l in _sp.get("legs", [])
        ))
        if _content_key not in _seen_parlay_content:
            _seen_parlay_content.add(_content_key)
            _deduped_section_a.append(_sp)
    section_a_picks = _deduped_section_a

    # ── Split into anchor and core pools ──────────────────────────────────────
    # Leg diversity: each individual leg_id may appear in at most 2 of the top picks.
    # This prevents a single high-LQS leg (e.g. ARI +1.5) from dominating all 5 picks.
    _MAX_LEG_APPEARANCES = 2
    leg_use_count: dict[str, int] = {}

    anchor_picks = []
    core_picks   = []
    mixed_picks  = []
    used_sigs    = set()

    for cand in candidates:
        sig = frozenset(l["leg_id"] for l in cand["legs"])
        if sig in used_sigs:
            continue

        # Leg diversity gate: skip if any leg would exceed its appearance budget
        if any(leg_use_count.get(l["leg_id"], 0) >= _MAX_LEG_APPEARANCES
               for l in cand["legs"]):
            continue

        # Near-duplicate gate: skip if >50% of legs overlap with any existing pick
        all_selected = anchor_picks + core_picks + mixed_picks
        overlap = any(
            len(sig & frozenset(l["leg_id"] for l in p["legs"])) / max(len(sig), 1) > 0.5
            for p in all_selected
        )
        if overlap:
            continue

        # Accept this pick — register it and update leg counts
        used_sigs.add(sig)
        for l in cand["legs"]:
            leg_use_count[l["leg_id"]] = leg_use_count.get(l["leg_id"], 0) + 1

        bc = cand["bet_class"]
        if bc == "anchor" and len(anchor_picks) < 3:
            anchor_picks.append(_format_pick(cand, stake, db))
        elif bc == "core" and len(core_picks) < n_picks:
            core_picks.append(_format_pick(cand, stake, db))
        elif bc == "mixed" and len(mixed_picks) < 2:
            mixed_picks.append(_format_pick(cand, stake, db))

    # Fallback: if no anchor/core split, put everything in picks for backwards compat
    # Apply the same leg diversity constraint so the fallback is also clean.
    # Use the shared leg_use_count so power picks downstream honour the same budget.
    all_picks = anchor_picks + core_picks + mixed_picks
    if not all_picks:
        for c in candidates:
            if len(all_picks) >= n_picks:
                break
            if any(leg_use_count.get(l["leg_id"], 0) >= _MAX_LEG_APPEARANCES
                   for l in c["legs"]):
                continue
            all_picks.append(_format_pick(c, stake, db))
            for l in c["legs"]:
                leg_use_count[l["leg_id"]] = leg_use_count.get(l["leg_id"], 0) + 1

    # ── Pool assessment metadata ───────────────────────────────────────────────
    n_high_quality = sum(1 for l in pool if (l.get("lqs") or 0) >= 60)
    n_good_quality = sum(1 for l in pool if 55 <= (l.get("lqs") or 0) < 60)
    # Count picks by depth for assessment note
    _picks_by_depth: dict[int, int] = {}
    for _p in anchor_picks + core_picks + mixed_picks + all_picks:
        _n = _p.get("n_legs", len(_p.get("legs", [])))
        _picks_by_depth[_n] = _picks_by_depth.get(_n, 0) + 1
    _depth_note_parts = [f"{v}×{k}-leg" for k, v in sorted(_picks_by_depth.items(), reverse=True)]
    pool_assessment  = {
        "total_legs":        len(pool),
        "legs_evaluated":    _legs_evaluated,
        "legs_positive_ev":  _legs_positive_ev,
        "high_quality_legs": n_high_quality,   # LQS ≥ 60
        "good_quality_legs": n_good_quality,   # LQS 55–59
        "depth_note": (
            ", ".join(_depth_note_parts) if _depth_note_parts
            else f"{n_high_quality} high-quality legs available"
        ),
    }

    # Annotate all normal-path picks as Section A (sharp picks)
    for _p in anchor_picks + core_picks + mixed_picks + all_picks:
        _p.setdefault("confidence_tier", "A")
        _p.setdefault("tier_label",      "Sharp Picks")
        _p.setdefault("show_warning",    False)

    # ── Regime A/B shadow run (never changes what gets served) ───────────────
    # Logged for post-hoc win-rate comparison.  Standard picks always returned.
    try:
        _do_regime_shadow(_candidates_shadow, anchor_picks + core_picks + mixed_picks + all_picks,
                          _cand_max_ev, stake, db)
    except Exception as _rse:
        print(f"[Recommender] Regime shadow run error (non-fatal): {_rse}")

    # Depth summary for pool assessment note
    _sa_depth_note = ", ".join(
        f"1×{sz}-leg" for sz in sorted(set(p.get("section_a_depth", p.get("n_legs", 0))
                                           for p in section_a_picks), reverse=True)
    ) if section_a_picks else ""
    if _sa_depth_note:
        pool_assessment["section_a_note"] = _sa_depth_note

    return {
        "anchor":          anchor_picks,
        "core":            core_picks,
        "mixed":           mixed_picks,
        "picks":           all_picks,        # backwards compat
        "section_a":       section_a_picks,  # depth-diverse CUSHION picks (primary + secondary)
        "power_picks":     [],               # deprecated — priority loop handles 4-5 leg combos
        "section_a_empty": len(section_a_picks) == 0,
        "pool_assessment": pool_assessment,
        "ale_summary": {
            "ale_switched":   n_ale_switched,
            "ale_fav_plus":   n_ale_fav_plus,
            "legs_evaluated": _legs_evaluated,
        },
    }


_BREAKEVEN_PROB = 52.38   # break-even at standard -110 ATS juice


def _dominant_sport(legs: list[dict]) -> str:
    """Return the most common sport label in a leg list."""
    from collections import Counter
    sports = [_normalize_sport(l.get("sport", "")) for l in legs if l.get("sport")]
    if not sports:
        return ""
    return Counter(sports).most_common(1)[0][0]


def _normalize_sport(label: str) -> str:
    """
    Map any sport label to the canonical key used in recommender lookup dicts.

    parlay_builder._normalize_sport converts "MLB" → "Baseball" and
    "NHL" → "Ice Hockey", which breaks lookups in _MODEL_LABEL, _SECTION_A_THRESHOLD,
    etc. that use "MLB"/"NHL" as keys.  This wrapper maps back to canonical form.
    """
    from parlay_builder import _normalize_sport as _pb_norm
    _CANONICAL = {
        "Baseball":        "MLB",
        "Ice Hockey":      "NHL",
        "Basketball":      "NBA",
        "American Football": "NFL",
        "Soccer":          "Soccer",
    }
    return _CANONICAL.get(_pb_norm(label), label)


def _profile_hist_edge(leg: dict, db) -> Optional[float]:
    """
    Return historical_edge_pp for one leg using LegQualityProfile.

    Formula: unbiased_win_rate * 100 - implied_prob_pct
    Returns None (not 0.0) when no profile exists so the UI can show "—".
    """
    if db is None:
        return None
    from database import LegQualityProfile
    market_type = leg.get("market_label") or leg.get("market_type") or leg.get("market", "")
    sport       = leg.get("sport", "") or ""
    implied_p   = leg.get("implied_prob")   # already percent, e.g. 71.43
    if implied_p is None:
        odds = leg.get("odds")
        implied_p = round(1.0 / odds * 100, 2) if odds and odds > 1 else None
    if implied_p is None:
        return None

    # Try sport-specific aggregate profile first, then sport-agnostic
    for sp_filter in [sport, None]:
        q = db.query(LegQualityProfile).filter(
            LegQualityProfile.market_type == market_type,
            LegQualityProfile.team_or_player.is_(None),
        )
        if sp_filter:
            q = q.filter(LegQualityProfile.sport == sp_filter)
        else:
            q = q.filter(LegQualityProfile.sport.is_(None))
        profile = q.first()
        if profile and profile.unbiased_win_rate is not None:
            return round(profile.unbiased_win_rate * 100 - implied_p, 2)
    return None


def _format_pick(cand: dict, stake: float, db=None) -> dict:
    """Enrich a candidate pick with display fields."""
    legs_out = []
    for l in cand["legs"]:
        am = l.get("american_odds_raw") or (
            round((l["odds"]-1)*100) if l["odds"] >= 2
            else -round(100/(l["odds"]-1))
        )
        is_alt = l.get("is_alt_line", False)
        legs_out.append({
            **l,
            "american_odds_display": f"+{am}" if am > 0 else str(am),
            "market_label":          l.get("market_label", "Moneyline"),
            "is_alt_line":           is_alt,
            "main_line_odds":        l.get("main_line_odds"),
            "alt_line_label":        "⚡ Alt Line — Better Quality" if is_alt else None,
            "hist_win_rate":         round(strat.get_leg_win_rate(
                                        l.get("market_label","Moneyline"))*100, 1),
            # model_used already present on leg from score_leg(); make canonical label
            "model_used": _MODEL_LABEL.get(
                _normalize_sport(l.get("sport", "")),
                l.get("model_used", _DEFAULT_MODEL_LABEL),
            ) if l.get("model_used", "global") != "global" else _DEFAULT_MODEL_LABEL,
        })

    payout_10 = round((cand["combined_odds"]-1)*10, 2)
    cam       = cand.get("combined_am", 0)
    am_str    = f"+{cam}" if cam > 0 else str(cam)

    # ── Dominant sport / model for this parlay ────────────────────────────
    dom_sport   = _dominant_sport(cand["legs"])
    parlay_model = _MODEL_LABEL.get(dom_sport, _DEFAULT_MODEL_LABEL)
    parlay_auc   = _SUBMODEL_AUC.get(dom_sport, _COMBINED_AUC)

    # ── Model confidence fields ────────────────────────────────────────────
    # model_confidence = average per-leg win probability (not the combined product).
    # The combined product is exposed as combined_win_prob on the pick itself.
    # This ensures section_a_qualified comparisons are on a per-leg basis.
    leg_win_probs_raw = [l.get("win_prob", 0) for l in cand.get("legs", [])]
    model_conf = round(sum(leg_win_probs_raw) / len(leg_win_probs_raw), 1) if leg_win_probs_raw else 0.0

    # historical_edge_pp = avg(unbiased_win_rate - implied_prob) across legs.
    # Sourced from LegQualityProfile so it reflects actual historical outcomes,
    # not the model's current win_prob estimate.
    # Returns None (not 0.0) when no profile exists — UI shows "—" for None.
    leg_hist_edges = [_profile_hist_edge(l, db) for l in cand.get("legs", [])]
    valid_edges = [e for e in leg_hist_edges if e is not None]
    hist_edge = round(sum(valid_edges) / len(valid_edges), 2) if valid_edges else None

    # Per-leg min win_prob — used by Section A filter in frontend
    leg_win_probs = [l.get("win_prob", 0) for l in cand.get("legs", [])]
    min_leg_win_prob = round(min(leg_win_probs), 1) if leg_win_probs else 0.0
    section_a_threshold = _SECTION_A_THRESHOLD.get(dom_sport, 55.0)

    # Kelly stake at $500 bankroll — fraction scales with sub-model AUC
    combined_odds   = cand.get("combined_odds", 1.91)
    kelly_fraction  = _SUBMODEL_KELLY_FRACTION.get(dom_sport, _DEFAULT_KELLY_FRACTION)
    p               = model_conf / 100.0
    b               = combined_odds - 1.0
    kelly_f         = max(0.0, (p * b - (1 - p)) / b)
    kelly_suggestion = round(kelly_f * kelly_fraction * 500, 2)   # $500 bankroll

    low_conf_warning = model_conf < 45.0

    return {
        **cand,
        "legs":              legs_out,
        "american_odds":     am_str,
        "payout_10":         payout_10,
        "grade":             cand.get("grade", "C"),
        "grade_color":       cand.get("grade_color", "amber"),
        "grade_summary":     cand.get("grade_summary", ""),
        "strategy_note":     cand.get("strategy_note", ""),
        "bet_class":         cand.get("bet_class", "mixed"),
        "hist_win_prob":     cand.get("hist_win_prob", cand["combined_win_prob"]),
        "hist_ev":           cand.get("hist_ev", cand["expected_profit"]),
        "kelly_500":         cand.get("kelly_500", 0),
        # ── validated model fields ──
        "model_used":            parlay_model,
        "model_auc":             parlay_auc,
        "model_confidence":      round(model_conf, 1),
        "historical_edge_pp":    hist_edge,
        "kelly_suggestion":      kelly_suggestion,
        "kelly_fraction_used":   kelly_fraction,
        "low_conf_warning":      low_conf_warning,
        "dominant_sport":        dom_sport,
        "min_leg_win_prob":      min_leg_win_prob,
        "section_a_threshold":   section_a_threshold,
        "section_a_qualified":   min_leg_win_prob >= section_a_threshold,
        "has_alt_line":          any(l.get("is_alt_line") for l in legs_out),
        # Sport composition flags for boost eligibility (FanDuel +50% Route B requires same sport).
        # Use _normalize_sport so EPL/Bundesliga/Ligue 1 all collapse to "Soccer".
        "is_sgp":           len({l.get("fixture_id", "") for l in legs_out if l.get("fixture_id")}) == 1
                            and len(legs_out) >= 2,
        "is_single_sport":  len({_normalize_sport(l.get("sport") or "") for l in legs_out
                                  if l.get("sport")}) <= 1,
        # Emergency patch tag: any pick with a recently-started leg gets a distinct source
        # so it can be excluded from clean prospective WR/P&L analysis in signal_analysis.
        "source": (
            "prospective_live_patched_20260503"
            if any(l.get("live_bet_patch") for l in cand.get("legs", []))
            else "prospective"
        ),
    }


def _wrap_single_leg(leg: dict, stake: float, db=None) -> dict:
    """Wrap a single scored leg into a pick-dict (single-leg 'parlay')."""
    # Normalize field names — some ingest paths use 'line' instead of 'point'
    leg = {
        **leg,
        "point": leg.get("point") or leg.get("line"),
    }
    odds       = max(leg.get("odds", 1.91), 1.001)
    win_prob   = leg.get("win_prob", leg.get("implied_prob", 50.0))
    payout     = round((odds - 1) * stake, 2)
    exp_profit = round((win_prob / 100) * payout - (1 - win_prob / 100) * stake, 2)
    cam        = round((odds - 1) * 100) if odds >= 2 else -round(100 / (odds - 1))

    cand = {
        "legs":              [leg],
        "combined_odds":     round(odds, 3),
        "combined_win_prob": round(win_prob, 2),
        "model_win_prob":    round(win_prob, 2),
        "payout":            payout,
        "expected_profit":   exp_profit,
        "n_legs":            1,
        "bet_class":         "mixed",
        "grade":             "C",
        "grade_color":       "amber",
        "grade_summary":     "",
        "strategy_note":     "",
        "hist_win_prob":     win_prob,
        "hist_ev":           exp_profit,
        "kelly_500":         0,
        "combined_am":       cam,
    }
    return _format_pick(cand, stake, db)


def _build_fallback_parlays(
    scored: list,
    stake: float,
    db,
    n: int = 5,
) -> tuple[list, list]:
    """
    Assemble 2- and 3-leg parlays from positive-EV legs when the standard
    path produced no candidates.  Returns (tier_b_parlays, tier_c_parlays).

    Section B: combined EV > 0  (genuinely positive expected-value parlay)
    Section C: combined EV <= 0 but both legs individually positive-EV
               (quality combination worth monitoring)
    """
    import itertools as _it

    pos_ev = [l for l in scored if (l.get("edge") or 0) > 0 and l.get("odds") is not None]
    if len(pos_ev) < 2:
        return [], []

    # Composite per-leg score: balance win_prob (50%) and edge (50%)
    max_edge = max(l.get("edge", 0) for l in pos_ev) or 1.0
    leg_scores = {
        id(l): (l.get("win_prob", 50) / 100) * 0.5 + (l.get("edge", 0) / max_edge) * 0.5
        for l in pos_ev
    }
    pos_ev = sorted(pos_ev, key=lambda l: -leg_scores[id(l)])
    top12 = pos_ev[:12]
    top8  = pos_ev[:8]

    raw_candidates: list[dict] = []

    def _try_combo(legs: list) -> dict | None:
        # Same-fixture guard
        fids = [l.get("fixture_id") for l in legs if l.get("fixture_id")]
        if len(fids) != len(set(fids)):
            return None
        # Same-team guard (allow Over/Under totals to coexist)
        teams = [l.get("pick", "").strip().lower() for l in legs]
        real  = [t for t in teams if t not in ("over", "under", "")]
        if len(real) != len(set(real)):
            return None

        combined_odds = 1.0
        combined_win  = 1.0
        for l in legs:
            combined_odds *= l["odds"]
            combined_win  *= (l.get("win_prob", 50) / 100)

        # Minimum win_prob floor by depth — filters out high-variance longshots
        _WIN_FLOOR = {2: 0.20, 3: 0.12, 4: 0.08}
        n_legs = len(legs)
        if combined_win < _WIN_FLOOR.get(n_legs, 0.08):
            return None

        # Maximum odds cap — above +1500 variance makes EV unrealizable
        cam = (round((combined_odds - 1) * 100) if combined_odds >= 2
               else -round(100 / (combined_odds - 1)))
        if cam > 1500:
            return None

        payout  = (combined_odds - 1) * stake
        ev      = combined_win * payout - (1 - combined_win) * stake
        avg_lqs = sum(l.get("lqs") or 50 for l in legs) / len(legs)

        return {
            "legs":              legs,
            "combined_odds":     round(combined_odds, 3),
            "combined_win_prob": round(combined_win * 100, 2),
            "model_win_prob":    round(combined_win * 100, 2),
            "payout":            round(payout, 2),
            "expected_profit":   round(ev, 2),
            "n_legs":            n_legs,
            "bet_class":         "mixed",
            "grade":             "B" if ev > 0 else "C",
            "grade_color":       "blue" if ev > 0 else "amber",
            "grade_summary":     "",
            "strategy_note":     "",
            "hist_win_prob":     round(combined_win * 100, 2),
            "hist_ev":           round(ev, 2),
            "kelly_500":         0,
            "combined_am":       cam,
            "_fallback_avg_lqs": avg_lqs,
        }

    # 2-leg combos from top 12
    for i, j in _it.combinations(range(len(top12)), 2):
        c = _try_combo([top12[i], top12[j]])
        if c:
            raw_candidates.append(c)

    # 3-leg combos from top 8
    for i, j, k in _it.combinations(range(len(top8)), 3):
        c = _try_combo([top8[i], top8[j], top8[k]])
        if c:
            raw_candidates.append(c)

    if not raw_candidates:
        return [], []

    # Sort by composite: win_prob (50%) + EV (30%) + avg_lqs (20%)
    # Win_prob is weighted highest — ensures actionable picks surface first.
    max_ev = max(abs(c["expected_profit"]) for c in raw_candidates) or 1.0
    for c in raw_candidates:
        c["_sort_score"] = (
            (c["combined_win_prob"] / 100) * 0.5 +
            (max(c["expected_profit"], 0) / max_ev) * 0.3 +
            (c["_fallback_avg_lqs"] / 100) * 0.2
        )
    raw_candidates.sort(key=lambda c: -c["_sort_score"])

    # Deduplicate: skip if >50% leg overlap with an already-selected pick,
    # or if any single leg has already appeared twice (diversity gate).
    _MAX_LEG_APP = 2
    selected: list[dict] = []
    used_sigs: set = set()
    leg_app_count: dict[str, int] = {}
    _safe_lid = lambda l: l.get("leg_id") or l.get("description") or str(id(l))
    for c in raw_candidates:
        sig = frozenset(_safe_lid(l) for l in c["legs"])
        if sig in used_sigs:
            continue
        # Per-leg appearance cap — prevents one strong leg from dominating every parlay
        if any(leg_app_count.get(_safe_lid(l), 0) >= _MAX_LEG_APP for l in c["legs"]):
            continue
        if any(
            len(sig & frozenset(_safe_lid(l) for l in s["legs"])) / max(len(sig), 1) > 0.5
            for s in selected
        ):
            continue
        selected.append(c)
        used_sigs.add(sig)
        for l in c["legs"]:
            k = _safe_lid(l)
            leg_app_count[k] = leg_app_count.get(k, 0) + 1

    tier_b: list[dict] = []
    tier_c: list[dict] = []
    for c in selected:
        if len(tier_b) + len(tier_c) >= n * 2:
            break
        try:
            formatted = _format_pick(c, stake, db)
        except Exception as _fe:
            print(f"[fallback-parlays] format error: {_fe}")
            continue

        if c["expected_profit"] > 0:
            formatted["confidence_tier"] = "B"
            formatted["tier_label"]      = "Model Favored — Market Efficient"
            formatted["show_warning"]    = False
            if len(tier_b) < n:
                tier_b.append(formatted)
        else:
            formatted["confidence_tier"] = "C"
            formatted["tier_label"]      = "Quality Leg — Monitor for line movement"
            formatted["show_warning"]    = True
            if len(tier_c) < n:
                tier_c.append(formatted)

    print(f"[fallback-parlays] built {len(tier_b)} tier-B and {len(tier_c)} tier-C parlays "
          f"from {len(pos_ev)} positive-EV legs")
    return tier_b, tier_c


def _build_tiered_fallback(
    scored:            list,
    pool:              list,
    stake:             float,
    db,
    _legs_evaluated:   int,
    _legs_positive_ev: int,
    depth_lqs_min:     float = None,
) -> dict:
    """
    Build Section B / C / D picks when no Section A parlays can be assembled.
    Never returns a blank page — always surfaces the best available legs.

    Section B: model_confidence ≥ 52%, edge ≥ -3pp  (Model Favored — Market Efficient)
    Section C: LQS ≥ 65 regardless of edge           (Quality Leg — Monitor for line movement)
    Section D: remaining legs, sorted by win_prob     (Simulation Legs — tracking only)
    """
    # ── Fallback parlay assembly ──────────────────────────────────────────────
    # Attempt to build 2-3 leg parlays from positive-EV legs BEFORE falling
    # back to single-leg wraps.  Parlays go to B (positive combined EV) or
    # C (negative combined EV but both legs individually positive-EV).
    _fb_parlays_b, _fb_parlays_c = _build_fallback_parlays(scored, stake, db)

    # ── Section B ────────────────────────────────────────────────────────────
    # Qualifying criteria (any of the following):
    #   • edge > 0  (genuine positive-EV vs market — the primary signal)
    #   • win_prob ≥ 52%  (model high-confidence, even when edge is negative)
    # Progressive edge gate for the win_prob-only candidates:
    #   Try -3pp → -5pp → drop entirely on heavy-favourite days.
    _pos_ev_single = [l for l in scored if (l.get("edge") or 0) > 0]
    _hi_conf_only  = [l for l in scored if l.get("win_prob", 0) >= 52.0
                      and (l.get("edge") or 0) <= 0]

    _edge_threshold_b: float | None = -3.0
    if _hi_conf_only:
        if not any(l.get("edge", -99) >= -3.0 for l in _hi_conf_only):
            _edge_threshold_b = -5.0
        if not any(l.get("edge", -99) >= -5.0 for l in _hi_conf_only):
            _edge_threshold_b = None

    _hi_conf_filtered = (
        _hi_conf_only if _edge_threshold_b is None
        else [l for l in _hi_conf_only if l.get("edge", -99) >= _edge_threshold_b]
    )

    # Composite sort for single-leg B: EV (50%) + win_prob (50%)
    _max_edge_b = max((abs(l.get("edge") or 0) for l in _pos_ev_single + _hi_conf_filtered), default=1.0) or 1.0
    def _b_score(l: dict) -> float:
        return (l.get("win_prob", 50) / 100) * 0.5 + (max(l.get("edge") or 0, 0) / _max_edge_b) * 0.5

    sec_b_raw = sorted(
        {id(l): l for l in _pos_ev_single + _hi_conf_filtered}.values(),
        key=_b_score, reverse=True,
    )[:10]

    # ── Section C ────────────────────────────────────────────────────────────
    sec_c_raw = sorted(
        [l for l in scored if (l.get("lqs") or 0) >= 65.0],
        key=lambda x: -(x.get("lqs") or 0),
    )[:10]

    # ── Section D ────────────────────────────────────────────────────────────
    # Remaining legs not already surfaced in B or C (by object identity)
    b_ids     = {id(l) for l in sec_b_raw}
    c_ids     = {id(l) for l in sec_c_raw}
    remaining = [l for l in scored if id(l) not in b_ids and id(l) not in c_ids]
    sec_d_raw = sorted(remaining, key=lambda x: -x.get("win_prob", 0))[:5]

    # Augment Section D with any today's mock-bet legs not yet shown.
    # Mock descriptions often include the line ("San Francisco Giants -1.5")
    # while scored-leg descriptions are team-only ("San Francisco Giants").
    # Normalise both sides to team name before comparing.
    def _team_from_desc(desc: str) -> str:
        """Strip trailing line/market suffix: 'Team -1.5 (Spread)' → 'Team'."""
        d = re.sub(r'\s*[+-]?\d+\.?\d*\s*\([^)]*\)\s*$', '', desc).strip()
        d = re.sub(r'\s*\([^)]*\)\s*$', '', d).strip()
        d = re.sub(r'\s+[+-]?\d+\.?\d+$', '', d).strip()
        return d

    if db is not None:
        try:
            from database import MockBet, MockBetLeg as _MBL
            from zoneinfo import ZoneInfo as _ZI_TD
            today_str = datetime.now(_ZI_TD("America/Chicago")).strftime("%Y-%m-%d")
            mock_descs = [
                row[0]
                for row in db.query(_MBL.description)
                              .join(MockBet, _MBL.mock_bet_id == MockBet.id)
                              .filter(MockBet.game_date == today_str)
                              .all()
                if row[0]
            ]
            if mock_descs:
                # already_shown keyed on normalised team name
                already_shown_teams = (
                    {_team_from_desc(l.get("description", "")) for l in sec_b_raw}
                    | {_team_from_desc(l.get("description", "")) for l in sec_c_raw}
                    | {_team_from_desc(l.get("description", "")) for l in sec_d_raw}
                )
                # scored pool indexed by both full description and normalised team
                desc_to_leg  = {l.get("description", ""): l for l in scored if l.get("description")}
                team_to_leg  = {_team_from_desc(l.get("description", "")): l
                                for l in scored if l.get("description")}

                added_teams: set[str] = set()
                for mock_desc in sorted(mock_descs):
                    mock_team = _team_from_desc(mock_desc)
                    if mock_team in already_shown_teams or mock_team in added_teams:
                        continue
                    # Prefer exact description match; fall back to team-name match
                    leg = desc_to_leg.get(mock_desc) or team_to_leg.get(mock_team)
                    if leg:
                        sec_d_raw.append(leg)
                        added_teams.add(mock_team)
                        print(f"[tiered-fallback] mock augment: added '{mock_team}' to Section D")
                    else:
                        print(f"[tiered-fallback] mock augment: no scored leg found for '{mock_desc}'")
        except Exception as _e:
            print(f"[tiered-fallback] mock-bet augment error: {_e}")

    # Leg descriptions already used in parlay picks — avoid duplicating as singles
    _parlay_leg_descs: set[str] = {
        l.get("description", "")
        for p in (_fb_parlays_b + _fb_parlays_c)
        for l in p.get("legs", [])
        if l.get("description")
    }

    def _pick_list(legs, tier, label, warning, skip_descs: set | None = None):
        out = []
        for leg in legs:
            if skip_descs and leg.get("description", "") in skip_descs:
                continue
            try:
                p = _wrap_single_leg(leg, stake, db)
                p["confidence_tier"] = tier
                p["tier_label"]      = label
                p["show_warning"]    = warning
                out.append(p)
            except Exception as _e:
                print(f"[tiered-fallback] wrap error {tier}: {_e}")
        return out

    # Build Section B singles; track their descriptions so C doesn't repeat them.
    _singles_b = _pick_list(sec_b_raw, "B", "Model Favored — Market Efficient",               False, _parlay_leg_descs)
    _b_descs = _parlay_leg_descs | {
        l.get("description", "")
        for p in _singles_b
        for l in p.get("legs", [])
        if l.get("description")
    }
    _singles_c = _pick_list(sec_c_raw, "C", "Quality Leg — Monitor for line movement",         True,  _b_descs)
    _singles_d = _pick_list(sec_d_raw, "D", "Simulation Legs — Low confidence, tracking only", True)

    tier_b = _fb_parlays_b + _singles_b
    tier_c = _fb_parlays_c + _singles_c
    tier_d = _singles_d

    reason = (
        f"{len(pool)} leg(s) passed confidence gate but no valid parlay "
        f"combinations met LQS depth requirements (min {depth_lqs_min or 62})."
        if pool else
        f"All {_legs_evaluated} legs showed negative edge vs implied odds."
    )

    return {
        "anchor":          [],
        "core":            [],
        "mixed":           [],
        "picks":           tier_b + tier_c + tier_d,   # backwards-compat field
        "power_picks":     [],
        "section_a_empty": True,
        "section_a_note":  "No sharp edges today — market is efficient across all available legs.",
        "tier_b":          tier_b,
        "tier_c":          tier_c,
        "tier_d":          tier_d,
        "pool_assessment": {
            "total_legs":       len(scored),
            "legs_evaluated":   _legs_evaluated,
            "legs_positive_ev": _legs_positive_ev,
            "reason_no_picks":  reason,
            "next_refresh":     "Next refresh: 7:45 AM CT tomorrow",
        },
    }


def _pick_label(legs: list[dict], win_prob: float) -> str:
    sports = list({l.get("sport", "") for l in legs})
    sport  = sports[0] if len(sports) == 1 else "Multi-sport"
    n      = len(legs)
    conf   = "High confidence" if win_prob >= 50 else "Moderate confidence"
    return f"{conf} {n}-leg {sport} parlay"


# ── Parlay modification ────────────────────────────────────────────────────────

def modify_parlay(
    current_legs:   list[dict],
    add_leg_ids:    list[str],
    remove_leg_ids: list[str],
    stake:          float,
    db:             Session,
) -> dict:
    from parlay_builder import get_available_legs, score_leg
    remaining = [l for l in current_legs if l["leg_id"] not in set(remove_leg_ids)]
    if add_leg_ids:
        all_avail = get_available_legs(db)
        leg_map   = {l["leg_id"]: l for l in all_avail}
        for lid in add_leg_ids:
            if lid in leg_map:
                remaining.append(score_leg(leg_map[lid]))
    if not remaining:
        return {"error": "Parlay would have no legs after modification."}
    return score_parlay(remaining, stake=stake, db=db)
