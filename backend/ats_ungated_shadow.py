"""
ATS Ungated Shadow — score ALL MLB spread legs through the ATS sub-model
with no gates (no market ban, no WP floor, no tier requirement).

Purpose: measure whether the model discriminates within sub-gate and -1.5
(fav-side) legs. Historical backtest shows the model CAN rank -1.5 legs
well (rank corr 0.985, top-20% covers 50.5% vs BE 37.3% = +13pp edge).
This forward shadow confirms that finding on live data.

Architecture:
  - Reuses get_available_legs() with ALL spread markets (including banned MLB spreads)
  - Scores each leg with score_leg() (same ATS sub-model as live pipeline)
  - Stores INDIVIDUAL legs as 1-leg mock_bets, source='ats_ungated'
  - amount=0, cohort='clean', settled by the same ESPN pipeline
  - Leg-level storage (not parlays) enables per-leg discrimination analysis
"""

import os
import sys
import uuid
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

_CT = ZoneInfo("America/Chicago")
_BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")


def generate_ungated_shadow(db=None, today_str=None):
    """Score all MLB spread legs (no gates) and store as shadow bets.

    Returns list of stored leg dicts.
    """
    from parlay_builder import get_available_legs, score_leg

    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[ats-ungated] Generating ungated shadow for {today_str}")

    # Get ALL legs including banned markets (spreads + alternate_spreads)
    all_legs = get_available_legs(
        db, markets=["h2h", "spreads", "totals", "alternate_spreads"]
    )
    if not all_legs:
        print("[ats-ungated] No legs available")
        return []

    # Filter to MLB spread legs only (the model's domain)
    mlb_spread_legs = [
        leg for leg in all_legs
        if leg.get("sport", "").upper() in ("MLB", "BASEBALL")
        and leg.get("market", "") in ("spreads", "alternate_spreads")
    ]
    print(f"[ats-ungated] {len(mlb_spread_legs)} MLB spread legs "
          f"(from {len(all_legs)} total legs across all sports/markets)")

    if not mlb_spread_legs:
        print("[ats-ungated] No MLB spread legs today")
        return []

    # Score each leg with the ATS sub-model (same model as live pipeline)
    scored = []
    for leg in mlb_spread_legs:
        try:
            scored_leg = score_leg(leg)
            if scored_leg.get("win_prob") is not None:
                scored.append(scored_leg)
        except Exception as exc:
            print(f"[ats-ungated] Score error: {exc}")
            continue

    print(f"[ats-ungated] {len(scored)} legs scored")

    # Classify by spread value
    spread_counts = {}
    for leg in scored:
        point = leg.get("point")
        sv = float(point) if point is not None else None
        spread_counts[sv] = spread_counts.get(sv, 0) + 1
    print(f"[ats-ungated] Spread distribution: {spread_counts}")

    # Classify by gate status (would the live pipeline accept this leg?)
    # Live gates: MLB alt_spreads WP >= 62%, MLB spreads = BANNED
    gated = 0
    ungated = 0
    for leg in scored:
        wp = leg.get("win_prob", 0)
        mkt = leg.get("market", "")
        if mkt == "spreads":
            ungated += 1  # banned market — gate rejects ALL
        elif wp < 62.0:
            ungated += 1  # below WP floor
        else:
            gated += 1    # would pass gate
    print(f"[ats-ungated] Gate status: {gated} would pass, {ungated} gate-rejected")

    # Store as individual mock_bets (1 leg per bet)
    if db is not None and scored:
        stored = _store_ungated_legs(db, scored, today_str)
        print(f"[ats-ungated] Stored {stored} shadow legs (source=ats_ungated)")

    return scored


def _store_ungated_legs(db, scored_legs, today_str):
    """Persist individual scored legs as 1-leg mock_bets."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    run_id = str(uuid.uuid4())
    stored = 0

    for leg in scored_legs:
        mock_id = str(uuid.uuid4())
        desc = leg.get("description", "")
        wp = leg.get("win_prob", 0)
        odds = leg.get("odds", 1.0)
        ev = leg.get("ev", 0)
        point = leg.get("point")
        sv = float(point) if point is not None else None
        mkt = leg.get("market", "")

        bet_info = f"{desc} | wp={wp:.1f}"

        # Dedup: skip if same leg already stored today
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
            "AND source='ats_ungated' AND bet_info=:bi"
        ), {"gd": today_str, "bi": bet_info}).scalar()
        if existing > 0:
            continue

        # Determine gate status for this leg
        if mkt == "spreads":
            gate_status = "banned_market"
        elif wp < 62.0:
            gate_status = f"below_wp_floor_{wp:.0f}"
        else:
            gate_status = "would_pass"

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=odds,
            amount=0.0,
            legs=1,
            bet_info=bet_info,
            predicted_win_prob=wp,
            model_used=leg.get("model_used", "mlb_ats_v1"),
            status="PENDING",
            source="ats_ungated",
            cohort="clean",
            notes=f"gate={gate_status} spread={sv} mkt={mkt}",
        )
        db.add(mock)

        mock_leg = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=desc,
            market_type=mkt,
            sport="MLB",
            win_prob=wp,
            ev=ev,
            fixture_id=leg.get("fixture_id"),
            is_alt_line=(mkt == "alternate_spreads"),
        )
        db.add(mock_leg)
        stored += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[ats-ungated] Commit error: {exc}")
        return 0

    return stored


if __name__ == "__main__":
    from database import SessionLocal, init_db
    init_db()
    db = SessionLocal()
    try:
        legs = generate_ungated_shadow(db)
        if legs:
            print(f"\n[ats-ungated] Summary: {len(legs)} legs scored")
            # Show a sample
            for leg in sorted(legs, key=lambda l: l.get("win_prob", 0), reverse=True)[:10]:
                wp = leg.get("win_prob", 0)
                desc = leg.get("description", "")[:50]
                mkt = leg.get("market", "")
                print(f"  wp={wp:5.1f}% {mkt:20s} {desc}")
    finally:
        db.close()
