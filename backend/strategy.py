"""
strategy.py — Blended betting strategy logic.

Defines ANCHOR and CORE bet classifications, grading rubrics,
and per-market historical win rates derived from 536-bet Pikkit history.
"""

# ── Per-market historical win rates (geometric mean, 536-bet sample) ──────────
MARKET_WIN_RATES = {
    "Moneyline":   0.8194,
    "Alt Spread":  0.8195,
    "Total":       0.8088,
    "Alt Total":   0.7444,
    "Player Prop": 0.8399,
    "Spread":      0.8088,  # treated same as Total (small sample)
}

DEFAULT_WIN_RATE = 0.8194   # fallback = Moneyline rate


def get_leg_win_rate(market: str) -> float:
    return MARKET_WIN_RATES.get(market, DEFAULT_WIN_RATE)


def american_to_decimal(am: float) -> float:
    if am > 0:  return round(1 + am / 100, 5)
    if am < 0:  return round(1 + 100 / abs(am), 5)
    return 1.0


# ── Classification ─────────────────────────────────────────────────────────────

def classify_bet(legs: list, combined_american_odds: int = None) -> str:
    """
    Classify a bet as 'anchor', 'core', or 'mixed'.

    ANCHOR — high win probability, tight odds, cushion role.
      • 1-2 legs
      • Combined American odds between -166 and -400
      • Market: Moneyline or Total only

    CORE — high EV, positive combined odds, growth role.
      • 2-4 legs
      • Combined American odds > 0 (positive)
      • Market: Moneyline or Total preferred

    MIXED — doesn't fit either cleanly.
    """
    n = len(legs)

    # Determine combined odds
    if combined_american_odds is not None:
        cam = combined_american_odds
    else:
        dec = 1.0
        for leg in legs:
            dec *= american_to_decimal(leg.get("american_odds", -110))
        cam = round((dec - 1) * 100) if dec >= 2 else -round(100 / (dec - 1))

    anchor_markets = {"Moneyline", "Total", "Spread"}
    core_markets   = {"Moneyline", "Total", "Spread", "Alt Spread"}

    markets = [leg.get("market", "Moneyline") for leg in legs]
    all_anchor_mkt = all(m in anchor_markets for m in markets)
    all_core_mkt   = all(m in core_markets   for m in markets)

    if n <= 2 and -400 <= cam <= -166 and all_anchor_mkt:
        return "anchor"
    if n >= 2 and cam > 0 and all_core_mkt:
        return "core"
    if n <= 2 and -165 <= cam <= -1:
        return "anchor"   # treat tight negative as anchor too
    return "mixed"


# ── EV using per-market win rates ──────────────────────────────────────────────

def calc_ev(legs: list, combined_decimal: float, stake: float = 10.0) -> dict:
    """
    Calculate EV using per-market historical win rates instead of flat 85.71%.
    """
    win_prob = 1.0
    for leg in legs:
        win_prob *= get_leg_win_rate(leg.get("market", "Moneyline"))

    payout   = (combined_decimal - 1) * stake
    ev       = round(win_prob * payout - (1 - win_prob) * stake, 4)
    n        = len(legs)

    # Kelly (25% fractional, 10% cap)
    b        = combined_decimal - 1
    q        = 1 - win_prob
    raw_k    = (b * win_prob - q) / b if b > 0 else -1
    kelly_25 = min(max(raw_k * 0.25, 0), 0.10)

    return {
        "win_prob":      round(win_prob * 100, 2),
        "ev":            ev,
        "payout":        round(payout, 2),
        "kelly_pct":     round(kelly_25 * 100, 4),
        "kelly_500":     round(500 * kelly_25, 2),
        "breakeven_met": ev > 0,
    }


# ── Grading ───────────────────────────────────────────────────────────────────

def grade_anchor(win_prob_pct: float, ev: float, cam: int) -> dict:
    """
    ANCHOR rubric: prioritises win probability and cushion reliability.
      A — win_prob >= 75% AND ev > 0
      B — win_prob >= 65% AND ev > 0
      C — win_prob >= 55% AND ev > 0
      D — ev > 0 but win_prob < 55%
      F — ev <= 0
    """
    if ev <= 0:
        return {"grade": "F", "color": "red",
                "summary": f"Negative EV (${ev:.2f}). Not worth the anchor slot."}
    if win_prob_pct >= 75:
        return {"grade": "A", "color": "green",
                "summary": f"Strong anchor. {win_prob_pct}% win prob with positive EV."}
    if win_prob_pct >= 65:
        return {"grade": "B", "color": "blue",
                "summary": f"Solid anchor. {win_prob_pct}% win prob — reliable cushion."}
    if win_prob_pct >= 55:
        return {"grade": "C", "color": "amber",
                "summary": f"Marginal anchor. {win_prob_pct}% win prob — consider tighter odds."}
    return {"grade": "D", "color": "red",
            "summary": f"Too risky for anchor role at {win_prob_pct}% win prob."}


def grade_core(win_prob_pct: float, ev: float, cam: int, n_legs: int) -> dict:
    """
    CORE rubric: prioritises EV and positive odds.
      A — positive odds AND ev > $10 AND win_prob >= 20%
      B — positive odds AND ev > $5  AND win_prob >= 15%
      C — positive odds AND ev > $0  AND win_prob >= 10%
      D — negative combined odds but ev > 0
      F — ev <= 0 or negative odds without merit
    """
    positive = cam > 0
    if ev <= 0:
        return {"grade": "F", "color": "red",
                "summary": f"Negative EV (${ev:.2f}). Doesn't qualify as a core bet."}
    if positive and ev >= 10 and win_prob_pct >= 20:
        return {"grade": "A", "color": "green",
                "summary": f"Elite core bet. +${ev:.2f} EV, {win_prob_pct}% win prob at positive odds."}
    if positive and ev >= 5 and win_prob_pct >= 15:
        return {"grade": "B", "color": "blue",
                "summary": f"Strong core bet. +${ev:.2f} EV at positive odds."}
    if positive and ev > 0 and win_prob_pct >= 10:
        return {"grade": "C", "color": "amber",
                "summary": f"Acceptable core bet. Positive odds but modest EV (${ev:.2f})."}
    if not positive and ev > 0:
        return {"grade": "D", "color": "amber",
                "summary": f"Positive EV but negative combined odds — better suited as anchor."}
    return {"grade": "F", "color": "red",
            "summary": "Doesn't meet core bet criteria."}


def grade_bet(legs: list, combined_decimal: float,
              combined_american_odds: int = None,
              stake: float = 10.0) -> dict:
    """
    Master grading function. Classifies the bet then applies the right rubric.
    Returns full analysis including classification, EV, grade, and recommendation.
    """
    n   = len(legs)
    ev_data = calc_ev(legs, combined_decimal, stake)

    if combined_american_odds is None:
        cam = round((combined_decimal - 1) * 100) if combined_decimal >= 2 \
              else -round(100 / (combined_decimal - 1))
    else:
        cam = combined_american_odds

    bet_class = classify_bet(legs, cam)

    if bet_class == "anchor":
        grade_info = grade_anchor(ev_data["win_prob"], ev_data["ev"], cam)
    elif bet_class == "core":
        grade_info = grade_core(ev_data["win_prob"], ev_data["ev"], cam, n)
    else:
        # Mixed: use stricter of the two rubrics
        anch = grade_anchor(ev_data["win_prob"], ev_data["ev"], cam)
        core = grade_core(ev_data["win_prob"], ev_data["ev"], cam, n)
        # Take the worse grade
        order = ["A", "B", "C", "D", "F"]
        grade_info = anch if order.index(anch["grade"]) >= order.index(core["grade"]) else core
        grade_info["summary"] = f"Mixed bet type. " + grade_info["summary"]

    return {
        "bet_class":       bet_class,
        "n_legs":          n,
        "combined_decimal":combined_decimal,
        "combined_am":     cam,
        **ev_data,
        **grade_info,
        "strategy_note":   _strategy_note(bet_class, ev_data, cam),
    }


def _strategy_note(bet_class: str, ev_data: dict, cam: int) -> str:
    if bet_class == "anchor":
        return (f"Use as your session ANCHOR — "
                f"{ev_data['win_prob']}% win prob provides a cushion "
                f"against core bet misses. Pair with a positive-odds core bet.")
    if bet_class == "core":
        return (f"Use as your session CORE — "
                f"positive EV (${ev_data['ev']:+.2f}) drives bankroll growth. "
                f"Pair with a -166 to -400 anchor bet for session stability.")
    return "Mixed structure — consider restructuring into a pure anchor or core bet."
