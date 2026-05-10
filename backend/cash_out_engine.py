"""
cash_out_engine.py — Cash Out Decision Support (Phase 6B)

Public API
----------
  evaluate_cash_out(bet, current_offer_amount, remaining_legs_status,
                    user_doubt_signal=False) -> dict

Input contract
--------------
  bet: {
    original_stake: float   — amount wagered
    original_odds:  float   — decimal odds (e.g. 3.5 = +250)
    legs:           int
  }

  remaining_legs_status: list of {
    leg_id:           str
    status:           "won" | "lost" | "pending" | "in_game" | "at_risk"
    updated_prob:     float  — current estimated win probability (0–1)
    at_risk_flag:     bool   — user has a specific concern (injury, weather, etc.)
    at_risk_reason:   str    — human description (optional)
    original_prob:    float  — probability at bet placement (optional; used for
                               HOLD_STRONG check; defaults to updated_prob if omitted)
  }

Output keys
-----------
  full_payout              total dollars received if bet wins (stake × decimal odds)
  combined_remaining_prob  product of updated_prob for all non-won, non-lost legs
  cash_out_implied_prob    offer / full_payout  (what the book thinks you'll win)
  current_fair_value       combined_remaining_prob × full_payout
  cash_out_vig             fair_value − offer  (always ≥ 0 for a rational book)
  cash_out_vig_pct         vig / fair_value × 100
  ev_hold                  combined_prob × full_payout  (no downside term; stake spent)
  ev_cash_out              current_offer_amount − original_stake  (net profit)
  ev_delta                 ev_hold − ev_cash_out  (+ve → holding is worth more)
  any_at_risk              bool — any leg flagged at_risk
  recommendation           "CASH_OUT_RECOMMENDED" | "CASH_OUT_OK" | "HOLD" | "HOLD_STRONG"
  recommendation_reason    human-readable explanation
  confidence               "high" | "medium" | "low"

Recommendation priority (evaluated in order; first match wins)
--------------------------------------------------------------
  CASH_OUT_RECOMMENDED  cash_out_implied_prob > combined_prob + 5pp
                        OR any at_risk leg
                        OR ev_cash_out > ev_hold × 0.90
  HOLD_STRONG           ev_hold > ev_cash_out × 1.30
                        AND vig_pct > 15
                        AND all pending legs improving vs original_prob
  CASH_OUT_OK           ev_cash_out ≥ ev_hold × 0.90
                        AND user_doubt_signal = True
                        AND vig_pct < 8
  HOLD                  default (ev_hold meaningfully better, or default)
"""
from __future__ import annotations


_PENDING_STATUSES = {"pending", "in_game", "at_risk"}


def evaluate_cash_out(
    bet: dict,
    current_offer_amount: float,
    remaining_legs_status: list[dict],
    user_doubt_signal: bool = False,
) -> dict:
    """
    Evaluate whether to accept a sportsbook cash out offer.

    See module docstring for full input/output contract.
    """
    original_stake = float(bet.get("original_stake", 0))
    original_odds  = float(bet.get("original_odds", 1.0))
    current_offer  = float(current_offer_amount)

    full_payout = original_stake * original_odds   # total payout on win

    # ── Walk legs ────────────────────────────────────────────────────────────
    combined_remaining_prob = 1.0
    any_at_risk             = False
    any_lost                = False
    all_pending_improving   = True   # True until a pending leg shows declining prob

    at_risk_reasons: list[str] = []

    for leg in remaining_legs_status:
        status       = str(leg.get("status", "pending")).lower()
        updated_prob = float(leg.get("updated_prob", 0.5))
        original_prob= float(leg.get("original_prob", updated_prob))
        at_risk      = bool(leg.get("at_risk_flag", False))

        if status == "lost":
            any_lost = True
        elif status == "won":
            pass   # locked in; does not affect remaining probability
        elif status in _PENDING_STATUSES:
            combined_remaining_prob *= max(0.0, min(1.0, updated_prob))
            if at_risk:
                any_at_risk = True
                reason = leg.get("at_risk_reason", "").strip()
                at_risk_reasons.append(reason or "unspecified")
            if updated_prob < original_prob:
                all_pending_improving = False

    if any_lost:
        combined_remaining_prob = 0.0

    # ── Core metrics ─────────────────────────────────────────────────────────
    safe_payout = full_payout if full_payout > 0 else 1.0   # guard div/0

    cash_out_implied_prob = current_offer / safe_payout
    current_fair_value    = combined_remaining_prob * full_payout
    cash_out_vig          = current_fair_value - current_offer
    safe_fair_value       = current_fair_value if current_fair_value > 0 else 1.0
    cash_out_vig_pct      = (cash_out_vig / safe_fair_value) * 100

    ev_hold     = combined_remaining_prob * full_payout   # no downside term
    ev_cash_out = current_offer - original_stake          # net profit from offer
    ev_delta    = ev_hold - ev_cash_out

    # ── Recommendation logic (first match wins) ───────────────────────────────

    # 1 — CASH_OUT_RECOMMENDED
    implied_overprice = cash_out_implied_prob > combined_remaining_prob + 0.05
    co_captures_90pct = ev_cash_out > ev_hold * 0.90 if ev_hold > 0 else False

    if implied_overprice or any_at_risk or co_captures_90pct:
        recommendation = "CASH_OUT_RECOMMENDED"

        if any_at_risk:
            r_str = "; ".join(at_risk_reasons[:3])
            reason = f"Leg flagged at risk ({r_str}). "
        elif implied_overprice:
            reason = (
                f"Book's offer implies {cash_out_implied_prob*100:.1f}% win probability "
                f"vs your current estimate of {combined_remaining_prob*100:.1f}% — "
                f"book is absorbing {(cash_out_implied_prob - combined_remaining_prob)*100:.1f}pp less "
                f"risk than your model suggests. "
            )
        else:
            reason = "Cash out offer captures 90%+ of hold value. "

        reason += (
            f"Offer ${current_offer:.2f} vs fair value ${current_fair_value:.2f}. "
            f"Book vig {cash_out_vig_pct:.1f}%."
        )
        confidence = (
            "high"   if (any_at_risk or combined_remaining_prob < 0.30)
            else "medium"
        )

    # 2 — HOLD_STRONG
    elif (
        ev_hold > 0
        and ev_cash_out > 0
        and ev_hold > ev_cash_out * 1.30
        and cash_out_vig_pct > 15
        and all_pending_improving
    ):
        recommendation = "HOLD_STRONG"
        pct_better     = (ev_hold / ev_cash_out - 1) * 100 if ev_cash_out > 0 else 0
        reason = (
            f"All legs tracking at or above original probability. "
            f"Hold EV ${ev_hold:.2f} vs cash out net ${ev_cash_out:.2f} "
            f"({pct_better:.0f}% better). "
            f"Book vig {cash_out_vig_pct:.1f}% — significantly overpriced offer. "
            f"Don't take it."
        )
        confidence = "high"

    # 3 — CASH_OUT_OK
    elif (
        ev_hold > 0
        and ev_cash_out >= ev_hold * 0.90
        and user_doubt_signal
        and cash_out_vig_pct < 8
    ):
        recommendation = "CASH_OUT_OK"
        reason = (
            f"Hold EV ${ev_hold:.2f} vs cash out net ${ev_cash_out:.2f} — within 10%. "
            f"Book is offering relatively fair terms (vig {cash_out_vig_pct:.1f}%). "
            f"Acceptable to lock in profit given your qualitative concerns."
        )
        confidence = "medium"

    # 4 — HOLD (default)
    else:
        recommendation = "HOLD"
        offer_pct = (current_offer / safe_fair_value) * 100 if current_fair_value > 0 else 0
        reason = (
            f"Hold EV ${ev_hold:.2f} vs cash out net ${ev_cash_out:.2f}. "
            f"Book offering {offer_pct:.0f}% of fair value "
            f"(vig {cash_out_vig_pct:.1f}%). "
        )
        if cash_out_vig_pct > 10:
            reason += "Significant value lost if you cash out now."
        else:
            reason += "Hold has meaningfully higher expected value."
        confidence = "medium"

    return {
        "full_payout":             round(full_payout,             2),
        "combined_remaining_prob": round(combined_remaining_prob, 4),
        "cash_out_implied_prob":   round(cash_out_implied_prob,   4),
        "current_fair_value":      round(current_fair_value,      2),
        "cash_out_vig":            round(cash_out_vig,            2),
        "cash_out_vig_pct":        round(cash_out_vig_pct,        1),
        "ev_hold":                 round(ev_hold,                 2),
        "ev_cash_out":             round(ev_cash_out,             2),
        "ev_delta":                round(ev_delta,                2),
        "any_at_risk":             any_at_risk,
        "recommendation":          recommendation,
        "recommendation_reason":   reason,
        "confidence":              confidence,
    }
