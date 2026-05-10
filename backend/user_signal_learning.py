"""
user_signal_learning.py — Bidirectional learning from user picks.

Flow:
  1. extract_signals(pick, legs)       → writes user_pick_signals rows
  2. update_signal_performance(db)     → aggregates wins/losses per (type, key)
  3. classify_signal_pattern(perf)     → labels EDGE_POSITIVE / EDGE_NEGATIVE / NEUTRAL
  4. get_user_signal_adjustment(leg)   → returns multiplier for recommender scoring
  5. evaluate_user_pick(pick, legs, db) → ADOPT | AVOID | NEUTRAL decision

Called at:
  - Pick submission: extract_signals()
  - Pick settlement: update_signal_performance() + classify_all()
  - Recommendation time: get_user_signal_adjustment()
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from database import UserPick, UserPickLeg, UserPickSignal, UserSignalPerformance


# ─── Signal extraction ────────────────────────────────────────────────────────

_FORM_HOT_KEYWORDS  = ["hot", "streak", "rolling", "fire", "on fire", "winning streak"]
_FORM_COLD_KEYWORDS = ["cold", "slump", "losing streak", "struggling", "cold streak"]

def extract_signals(
    pick:  UserPick,
    legs:  list[UserPickLeg],
    db:    Session,
) -> list[UserPickSignal]:
    """
    Derive interpretable signal rows from a pick and its legs.
    Writes rows to user_pick_signals and returns them.

    Signal types extracted:
      team_preference    — per leg: team:{TeamName}
      market_preference  — per leg: market:{Moneyline|Spread|Total|Prop}
      sport_preference   — per leg: sport:{MLB|NBA|...}
      leg_count_preference — pick-level: legs:{N}
      form_aware         — from pick notes: hot | cold keywords
    """
    signals: list[UserPickSignal] = []

    notes_lower = (pick.notes or "").lower()

    # Pick-level: leg count
    n_legs = len(legs)
    if n_legs:
        sig = UserPickSignal(
            user_pick_id      = pick.id,
            user_pick_leg_id  = None,
            signal_type       = "leg_count_preference",
            signal_key        = f"legs:{n_legs}",
            signal_value      = float(n_legs),
            reasoning_excerpt = None,
            outcome           = "PENDING",
        )
        signals.append(sig)

    # Pick-level: form awareness from notes
    for kw in _FORM_HOT_KEYWORDS:
        if kw in notes_lower:
            sig = UserPickSignal(
                user_pick_id      = pick.id,
                user_pick_leg_id  = None,
                signal_type       = "form_aware",
                signal_key        = "form_aware:hot",
                reasoning_excerpt = _excerpt(pick.notes, kw),
                outcome           = "PENDING",
            )
            signals.append(sig)
            break  # one hot signal per pick

    for kw in _FORM_COLD_KEYWORDS:
        if kw in notes_lower:
            sig = UserPickSignal(
                user_pick_id      = pick.id,
                user_pick_leg_id  = None,
                signal_type       = "form_aware",
                signal_key        = "form_aware:cold",
                reasoning_excerpt = _excerpt(pick.notes, kw),
                outcome           = "PENDING",
            )
            signals.append(sig)
            break

    # Per-leg signals
    for leg in legs:
        # Team preference
        if leg.team:
            sig = UserPickSignal(
                user_pick_id      = pick.id,
                user_pick_leg_id  = leg.id,
                signal_type       = "team_preference",
                signal_key        = f"team:{leg.team.strip()}",
                outcome           = "PENDING",
            )
            signals.append(sig)

        # Market type preference
        if leg.market_type:
            sig = UserPickSignal(
                user_pick_id      = pick.id,
                user_pick_leg_id  = leg.id,
                signal_type       = "market_preference",
                signal_key        = f"market:{leg.market_type.strip()}",
                outcome           = "PENDING",
            )
            signals.append(sig)

        # Sport preference
        if leg.sport:
            sig = UserPickSignal(
                user_pick_id      = pick.id,
                user_pick_leg_id  = leg.id,
                signal_type       = "sport_preference",
                signal_key        = f"sport:{leg.sport.strip()}",
                outcome           = "PENDING",
            )
            signals.append(sig)

    for sig in signals:
        db.add(sig)
    db.flush()  # get ids without full commit
    return signals


def _excerpt(text: Optional[str], keyword: str, window: int = 60) -> Optional[str]:
    """Return ±window chars around keyword in text."""
    if not text:
        return None
    i = text.lower().find(keyword)
    if i < 0:
        return None
    return text[max(0, i - 20): i + window].strip()


# ─── Signal performance update ────────────────────────────────────────────────

def update_signal_performance(db: Session) -> int:
    """
    Recompute user_signal_performance table from all settled user_pick_signals.
    Returns number of distinct (signal_type, signal_key) rows updated/created.
    """
    # Load all settled signals
    settled = db.query(UserPickSignal).filter(
        UserPickSignal.outcome.in_(["WON", "LOST"])
    ).all()

    # Aggregate
    agg: dict[tuple[str, str], dict] = {}
    for sig in settled:
        k = (sig.signal_type, sig.signal_key or "")
        if k not in agg:
            agg[k] = {"wins": 0, "losses": 0}
        if sig.outcome == "WON":
            agg[k]["wins"] += 1
        else:
            agg[k]["losses"] += 1

    updated = 0
    for (stype, skey), counts in agg.items():
        perf = db.query(UserSignalPerformance).filter(
            UserSignalPerformance.signal_type == stype,
            UserSignalPerformance.signal_key  == skey,
        ).first()

        wins   = counts["wins"]
        losses = counts["losses"]
        total  = wins + losses
        wr     = round(wins / total * 100, 2) if total else None

        p_class, weight = classify_signal_pattern_values(total, wr)

        if perf:
            perf.total_uses        = total
            perf.wins              = wins
            perf.losses            = losses
            perf.wr_pct            = wr
            perf.pattern_class     = p_class
            perf.confidence_weight = weight
            perf.last_updated      = datetime.utcnow()
        else:
            perf = UserSignalPerformance(
                signal_type        = stype,
                signal_key         = skey,
                total_uses         = total,
                wins               = wins,
                losses             = losses,
                wr_pct             = wr,
                pattern_class      = p_class,
                confidence_weight  = weight,
                last_updated       = datetime.utcnow(),
            )
            db.add(perf)

        updated += 1

    db.commit()
    return updated


def classify_signal_pattern_values(total: int, wr_pct: Optional[float]) -> tuple[str, float]:
    """
    Return (pattern_class, confidence_weight).

    Weight range: 0.5 (strong avoid) → 1.0 (neutral) → 1.5 (strong adopt).
    Evaluated top-down; first matching rule wins.
    """
    if total < 5 or wr_pct is None:
        return ("INSUFFICIENT_DATA", 1.0)

    # Strong adopt
    if wr_pct >= 60 and total >= 10:
        return ("EDGE_POSITIVE", 1.5)

    # Moderate adopt
    if wr_pct >= 55 and total >= 15:
        return ("EDGE_POSITIVE", 1.2)

    # Strong avoid
    if wr_pct <= 40 and total >= 10:
        return ("EDGE_NEGATIVE", 0.5)

    # Moderate avoid
    if wr_pct <= 45 and total >= 15:
        return ("EDGE_NEGATIVE", 0.7)

    return ("NEUTRAL", 1.0)


# ─── Model scoring adjustment ─────────────────────────────────────────────────

def get_user_signal_adjustment(
    leg: dict,
    db:  Session,
) -> tuple[float, list[str]]:
    """
    Compute composite multiplier from user signal performance for a given leg.
    Returns (multiplier, list_of_notes).

    Called by recommender.py when scoring legs.
    Only applies when a signal has ≥5 settled picks (INSUFFICIENT_DATA is ignored).
    """
    multiplier = 1.0
    notes: list[str] = []

    checks = []

    # Team
    team = (leg.get("team") or "").strip()
    if team:
        checks.append(("team_preference", f"team:{team}"))

    # Market
    market = (leg.get("market_type") or "").strip()
    if market:
        checks.append(("market_preference", f"market:{market}"))

    # Sport
    sport = (leg.get("sport") or "").strip()
    if sport:
        checks.append(("sport_preference", f"sport:{sport}"))

    for stype, skey in checks:
        perf = db.query(UserSignalPerformance).filter(
            UserSignalPerformance.signal_type == stype,
            UserSignalPerformance.signal_key  == skey,
        ).first()

        if not perf or perf.pattern_class in ("INSUFFICIENT_DATA", "NEUTRAL"):
            continue

        w = perf.confidence_weight or 1.0
        multiplier *= w

        if perf.pattern_class == "EDGE_POSITIVE":
            notes.append(
                f"User edge on {skey}: {perf.wr_pct:.0f}% WR over {perf.total_uses} picks — adopt (+{w:.2f}x)"
            )
        elif perf.pattern_class == "EDGE_NEGATIVE":
            notes.append(
                f"User weakness on {skey}: {perf.wr_pct:.0f}% WR over {perf.total_uses} picks — avoid ({w:.2f}x)"
            )

    return round(multiplier, 4), notes


# ─── ADOPT / AVOID / NEUTRAL pick evaluation ─────────────────────────────────

def evaluate_user_pick(
    pick: UserPick,
    legs: list[UserPickLeg],
    db:   Session,
) -> dict:
    """
    Determine whether the model should ADOPT, AVOID, or treat as NEUTRAL
    based on the user's historical track record on signals from this pick.

    Returns:
      {
        "decision": "ADOPT" | "AVOID" | "NEUTRAL",
        "adopt_score": float,
        "avoid_score": float,
        "signal_notes": [str],
      }
    """
    all_checks = []

    # Leg-level signals
    for leg in legs:
        for stype, skey in [
            ("team_preference",   f"team:{(leg.team or '').strip()}"),
            ("market_preference", f"market:{(leg.market_type or '').strip()}"),
            ("sport_preference",  f"sport:{(leg.sport or '').strip()}"),
        ]:
            if not skey.split(":", 1)[1]:
                continue
            all_checks.append((stype, skey))

    # Pick-level: leg count
    all_checks.append(("leg_count_preference", f"legs:{len(legs)}"))

    adopt_score = 0.0
    avoid_score  = 0.0
    signal_notes: list[str] = []

    for stype, skey in all_checks:
        perf = db.query(UserSignalPerformance).filter(
            UserSignalPerformance.signal_type == stype,
            UserSignalPerformance.signal_key  == skey,
        ).first()

        if not perf or perf.total_uses < 5:
            continue

        w = perf.confidence_weight or 1.0
        if perf.pattern_class == "EDGE_POSITIVE":
            adopt_score  += w - 1.0
            signal_notes.append(f"✅ {skey} — {perf.wr_pct:.0f}% WR ({perf.total_uses} picks)")
        elif perf.pattern_class == "EDGE_NEGATIVE":
            avoid_score  += 1.0 - w
            signal_notes.append(f"❌ {skey} — {perf.wr_pct:.0f}% WR ({perf.total_uses} picks)")

    DECISION_THRESHOLD = 0.3
    if adopt_score > avoid_score and adopt_score >= DECISION_THRESHOLD:
        decision = "ADOPT"
    elif avoid_score > adopt_score and avoid_score >= DECISION_THRESHOLD:
        decision = "AVOID"
    else:
        decision = "NEUTRAL"

    return {
        "decision":      decision,
        "adopt_score":   round(adopt_score, 3),
        "avoid_score":   round(avoid_score, 3),
        "signal_notes":  signal_notes,
    }


# ─── Comparison-vs-model signals ─────────────────────────────────────────────

def extract_comparison_signals(
    pick: UserPick,
    legs: list[UserPickLeg],
    db:   Session,
) -> list[UserPickSignal]:
    """
    For each leg in a user pick, check if the model generated a matching bet
    on the same game_date.  Writes two signal types:
      overlap_with_model  — user and model picked the same direction
      user_unique_pick    — user picked this, model did NOT

    These reveal whether user divergence from model is profitable.
    Uses only data already in the DB — no new fetches.
    """
    import sqlite3 as _sq
    from database import _BETS_DB_PATH

    signals: list[UserPickSignal] = []
    if not legs:
        return signals

    try:
        con = _sq.connect(_BETS_DB_PATH)
        con.row_factory = _sq.Row
        # Load all model mock bet leg descriptions for this game_date
        model_descs: set[str] = set()
        rows = con.execute(
            """
            SELECT LOWER(mbl.description) AS desc
            FROM   mock_bet_legs mbl
            JOIN   mock_bets     mb ON mb.id = mbl.mock_bet_id
            WHERE  mb.game_date = ?
              AND  mb.source IN ('prospective','prospective_pm','forced_generation','top_picks_page')
            """,
            (pick.game_date,),
        ).fetchall()
        con.close()
        model_descs = {r["desc"] for r in rows if r["desc"]}
    except Exception:
        return signals  # non-fatal — no model data available

    for leg in legs:
        leg_desc_lower = (leg.description or "").lower().strip()
        if not leg_desc_lower:
            continue

        # Fuzzy match: check if any model description contains key tokens from leg
        tokens = [t for t in leg_desc_lower.split() if len(t) >= 4]
        matched = any(
            all(tok in md for tok in tokens[:3])   # first 3 meaningful tokens
            for md in model_descs
        ) if tokens else False

        signal_type = "overlap_with_model" if matched else "user_unique_pick"
        sig = UserPickSignal(
            user_pick_id     = pick.id,
            user_pick_leg_id = leg.id,
            signal_type      = signal_type,
            signal_key       = f"{signal_type}:{leg.sport or 'unknown'}",
            outcome          = "PENDING",
            feature_source   = "keyword",
        )
        signals.append(sig)
        db.add(sig)

    db.flush()
    return signals


# ─── Market alignment signals (extracted at settlement time) ──────────────────

def extract_market_alignment_signals(
    pick: UserPick,
    legs: list[UserPickLeg],
    db:   Session,
) -> list[UserPickSignal]:
    """
    For each leg, check if the line moved TOWARD or AWAY from the user's pick
    using alt_lines data already in the DB.  Writes:
      line_movement_validation  signal_key='line_moved_toward' | 'line_moved_away'

    Only meaningful for spread/total legs where we have opening + closing lines.
    Uses existing alt_lines data — no new fetches.
    """
    import sqlite3 as _sq
    from database import _BETS_DB_PATH

    signals: list[UserPickSignal] = []

    try:
        con = _sq.connect(_BETS_DB_PATH)
        con.row_factory = _sq.Row
    except Exception:
        return signals

    try:
        for leg in legs:
            if not leg.odds_american or not leg.description:
                continue
            market = (leg.market_type or "").lower()
            if market not in ("spread", "alternate_spreads", "total", "alternate_totals"):
                continue

            # Look up alt_lines rows for this fixture around game_date
            # Movement heuristic: compare earliest vs latest odds_american for same description
            rows = con.execute(
                """
                SELECT al.odds, al.fetched_at
                FROM   alt_lines al
                JOIN   fixtures  f ON f.id = al.fixture_id
                WHERE  f.commence_time BETWEEN datetime(?, '-1 day') AND datetime(?, '+2 days')
                  AND  LOWER(al.description) LIKE ?
                ORDER BY al.fetched_at ASC
                """,
                (pick.game_date, pick.game_date, f"%{(leg.team or '').lower()[:8]}%"),
            ).fetchall()

            if len(rows) < 2:
                continue

            earliest_odds = rows[0]["odds"]
            latest_odds   = rows[-1]["odds"]

            # If odds improved (number went up) toward user pick = market agreed
            # American odds: closer to 0 (less negative or more positive) = shorter = more likely
            # "line moved toward user" = odds got shorter (more negative for favorites)
            user_odds = leg.odds_american
            if user_odds is None:
                continue

            if user_odds < 0:
                # User picked a favorite; line moved toward = odds became more negative (shorter)
                moved_toward = latest_odds <= earliest_odds
            else:
                # User picked an underdog; line moved toward = odds became more positive (longer)
                moved_toward = latest_odds >= earliest_odds

            direction = "line_moved_toward" if moved_toward else "line_moved_away"
            sig = UserPickSignal(
                user_pick_id     = pick.id,
                user_pick_leg_id = leg.id,
                signal_type      = "line_movement_validation",
                signal_key       = direction,
                outcome          = "PENDING",
                feature_source   = "keyword",
            )
            signals.append(sig)
            db.add(sig)

    except Exception:
        pass  # non-fatal
    finally:
        con.close()

    db.flush()
    return signals


# ─── Settle signals — called when a pick settles ─────────────────────────────

def settle_pick_signals(
    pick:        UserPick,
    db:          Session,
) -> None:
    """
    Update outcome on all user_pick_signals for this pick to WON/LOST,
    then rebuild signal performance aggregates.
    Also extracts market alignment signals at settlement time (uses closing data).
    """
    outcome = "WON" if pick.status == "SETTLED_WIN" else "LOST"
    signals = db.query(UserPickSignal).filter(
        UserPickSignal.user_pick_id == pick.id
    ).all()
    for sig in signals:
        sig.outcome = outcome
    db.flush()

    # Extract market alignment signals at settlement (best time — closing lines available)
    try:
        legs = db.query(UserPickLeg).filter(
            UserPickLeg.user_pick_id == pick.id
        ).all()
        new_sigs = extract_market_alignment_signals(pick, legs, db)
        # Immediately settle the new signals too
        for s in new_sigs:
            s.outcome = outcome
        db.flush()
    except Exception:
        pass  # non-fatal

    update_signal_performance(db)


# ─── Learning curve report ────────────────────────────────────────────────────

def learning_curve_report(db: Session, date_floor: Optional[str] = None) -> dict:
    """
    Summary report of all signal pattern classifications, top adopt/avoid
    patterns, and model improvement attribution.

    date_floor: YYYY-MM-DD — only count signals extracted from picks on or after this date.
    """
    # Total signals ingested
    sq = db.query(UserPickSignal)
    if date_floor:
        # Approximate: filter by extracted_at date
        sq = sq.filter(UserPickSignal.extracted_at >= date_floor)
    n_signals = sq.count()

    # All signal performances
    perfs = db.query(UserSignalPerformance).all()

    classes: dict[str, int] = {
        "edge_positive":    0,
        "edge_negative":    0,
        "neutral":          0,
        "insufficient_data":0,
    }
    adopt_patterns: list[str] = []
    avoid_patterns: list[str] = []

    for p in perfs:
        cls = (p.pattern_class or "INSUFFICIENT_DATA").lower()
        classes[cls] = classes.get(cls, 0) + 1

        if p.pattern_class == "EDGE_POSITIVE" and p.wr_pct is not None:
            adopt_patterns.append(
                f"{p.signal_key} — {p.wr_pct:.0f}% WR over {p.total_uses} picks "
                f"(+{p.confidence_weight:.2f}x weight)"
            )
        elif p.pattern_class == "EDGE_NEGATIVE" and p.wr_pct is not None:
            avoid_patterns.append(
                f"{p.signal_key} — {p.wr_pct:.0f}% WR over {p.total_uses} picks "
                f"({p.confidence_weight:.2f}x weight)"
            )

    # Sort: adopt by WR desc, avoid by WR asc
    edge_pos_perfs = [p for p in perfs if p.pattern_class == "EDGE_POSITIVE" and p.wr_pct]
    edge_neg_perfs = [p for p in perfs if p.pattern_class == "EDGE_NEGATIVE" and p.wr_pct]
    edge_pos_perfs.sort(key=lambda p: p.wr_pct or 0, reverse=True)
    edge_neg_perfs.sort(key=lambda p: p.wr_pct or 100)

    adopt_patterns = [
        f"{p.signal_key} ({p.wr_pct:.0f}% WR, n={p.total_uses}) — +{p.confidence_weight:.2f}x weight"
        for p in edge_pos_perfs[:10]
    ]
    avoid_patterns = [
        f"{p.signal_key} ({p.wr_pct:.0f}% WR, n={p.total_uses}) — {p.confidence_weight:.2f}x weight"
        for p in edge_neg_perfs[:10]
    ]

    # Model improvement attribution (baseline = model bets before user picks, current = after)
    # Simple approximation: compare early vs recent mock bet WR
    from database import MockBet as _MB
    all_settled_mocks = db.query(_MB).filter(
        _MB.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
        _MB.source.notin_(["exploration", "forced_generation"]),
    ).order_by(_MB.generated_at).all()

    baseline_wr = None
    current_wr  = None
    if len(all_settled_mocks) >= 20:
        half   = len(all_settled_mocks) // 2
        first_half  = all_settled_mocks[:half]
        second_half = all_settled_mocks[half:]
        bw = sum(1 for b in first_half  if b.status == "SETTLED_WIN") / max(len(first_half),  1) * 100
        cw = sum(1 for b in second_half if b.status == "SETTLED_WIN") / max(len(second_half), 1) * 100
        baseline_wr = round(bw, 1)
        current_wr  = round(cw, 1)

    return {
        "user_signals_ingested":  n_signals,
        "patterns_classified":    classes,
        "top_adopt_patterns":     adopt_patterns,
        "top_avoid_patterns":     avoid_patterns,
        "model_improvement_attribution": {
            "baseline_model_wr_pct": baseline_wr,
            "current_model_wr_pct":  current_wr,
            "note": "First-half vs second-half mock bet WR — rough attribution only",
        },
    }
