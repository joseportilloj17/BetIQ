"""
promo_engine.py — Phase 6A: Promo-aware EV scoring.

Handles boost promos (+25/30/50%) and free-bet-back by computing:
  - boosted odds (american)
  - EV with and without the promo
  - Kelly sizing update
  - strategy leg-count recommendation
  - use/save recommendation

Boost formula:
    new_decimal = 1 + (old_decimal - 1) × (1 + boost_pct)

Bet-back EV (free bet — no loss term):
    EV = win_prob × (decimal - 1) × stake
"""
from __future__ import annotations
import math


# ── Promo catalogue ────────────────────────────────────────────────────────────

PROMO_TYPES: dict[str, dict] = {
    "none":     {"boost": None,  "label": "No Promo",   "is_free": False},
    "boost_25": {"boost": 0.25,  "label": "+25% Boost", "is_free": False},
    "boost_30": {"boost": 0.30,  "label": "+30% Boost", "is_free": False},
    "boost_50": {"boost": 0.50,  "label": "+50% Boost", "is_free": False},
    "bet_back": {"boost": None,  "label": "Bet Back",   "is_free": True},
}

# Minimum meaningful EV lift to recommend using the promo
_USE_THRESHOLD = 1.00   # $1.00 lift on a $10 stake


# ── Conversion helpers ─────────────────────────────────────────────────────────

def american_to_decimal(american: int | float) -> float:
    """Convert American odds integer to decimal odds."""
    if american >= 100:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds to American (rounded to nearest integer)."""
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100)
    else:
        return round(-100.0 / (decimal - 1.0))


# ── Core functions ─────────────────────────────────────────────────────────────

def apply_boost(american_odds: int | float, boost_pct: float) -> int:
    """
    Apply a percentage boost to American odds.
    Returns new American odds (integer).

    Example: -110, boost=0.50
        decimal = 1 + 100/110 = 1.9091
        boosted = 1 + (1.9091 - 1) * 1.50 = 2.3636
        american = round(1.3636 * 100) = +136
    """
    dec = american_to_decimal(american_odds)
    boosted_dec = 1.0 + (dec - 1.0) * (1.0 + boost_pct)
    return decimal_to_american(boosted_dec)


def score_with_promo(
    win_prob:      float,   # model win probability 0-1
    american_odds: int | float,
    promo_type:    str  = "none",
    stake:         float = 10.0,
) -> dict:
    """
    Compute EV metrics with and without the promo.

    Returns a dict with:
        base_ev, boosted_ev, ev_lift,
        base_kelly, boosted_kelly,
        boosted_odds_american (None if no boost),
        strategy_rec, use_promo, recommendation_label, promo_label
    """
    promo = PROMO_TYPES.get(promo_type, PROMO_TYPES["none"])
    boost_pct = promo["boost"]
    is_free   = promo["is_free"]
    label     = promo["label"]

    dec_base = american_to_decimal(american_odds)
    p = float(win_prob)
    q = 1.0 - p

    # ── Base EV ────────────────────────────────────────────────────────────
    base_payout = (dec_base - 1.0) * stake
    base_ev     = p * base_payout - q * stake

    # ── Boosted EV ─────────────────────────────────────────────────────────
    boosted_odds_am = None
    boosted_ev      = base_ev

    if is_free:
        # Bet-back / free bet: no loss term — only win side
        boosted_ev = p * (dec_base - 1.0) * stake
        boosted_odds_am = None   # same odds, different risk structure
    elif boost_pct is not None:
        boosted_odds_am = apply_boost(american_odds, boost_pct)
        dec_boosted      = american_to_decimal(boosted_odds_am)
        boosted_payout   = (dec_boosted - 1.0) * stake
        boosted_ev       = p * boosted_payout - q * stake

    ev_lift = boosted_ev - base_ev

    # ── Kelly (25% fractional) ─────────────────────────────────────────────
    def kelly25(dec_odds: float, win_p: float, free_bet: bool = False) -> float:
        b = dec_odds - 1.0
        if b <= 0:
            return 0.0
        if free_bet:
            # Free bet Kelly: EV = p*b*stake, no downside → stake whole thing if EV > 0
            f = win_p if win_p > 0 else 0.0
        else:
            f = (win_p * b - (1.0 - win_p)) / b
        return max(0.0, f * 0.25)

    base_kelly    = kelly25(dec_base, p)
    boosted_kelly = base_kelly

    if is_free:
        boosted_kelly = kelly25(dec_base, p, free_bet=True)
    elif boost_pct is not None and boosted_odds_am is not None:
        dec_b = american_to_decimal(boosted_odds_am)
        boosted_kelly = kelly25(dec_b, p)

    # ── Strategy recommendation (leg count sweet spot) ─────────────────────
    if promo_type == "none":
        strategy_rec = "3-leg unboosted — standard strategy"
    elif promo_type in ("boost_25", "boost_30"):
        strategy_rec = "4-leg sweet spot — target +320 to +500 combined odds"
    elif promo_type == "boost_50":
        strategy_rec = "4-5 leg recommended — target +500 to +700 combined odds"
    elif promo_type == "bet_back":
        strategy_rec = "Free bet mode — no loss on this wager; go for higher payout"
    else:
        strategy_rec = ""

    # ── Use or save ────────────────────────────────────────────────────────
    use_promo = ev_lift >= _USE_THRESHOLD and promo_type != "none"

    if use_promo:
        recommendation_label = f"USE PROMO HERE ✓  +${ev_lift:.2f} lift"
    elif promo_type == "none":
        recommendation_label = "No promo selected"
    else:
        recommendation_label = f"SAVE PROMO — only +${ev_lift:.2f} lift here"

    return {
        "promo_type":           promo_type,
        "promo_label":          label,
        "base_ev":              round(base_ev, 2),
        "boosted_ev":           round(boosted_ev, 2),
        "ev_lift":              round(ev_lift, 2),
        "base_kelly_pct":       round(base_kelly * 100, 2),
        "boosted_kelly_pct":    round(boosted_kelly * 100, 2),
        "base_kelly_500":       round(base_kelly * 500, 2),
        "boosted_kelly_500":    round(boosted_kelly * 500, 2),
        "boosted_odds_american": boosted_odds_am,
        "strategy_rec":         strategy_rec,
        "use_promo":            use_promo,
        "recommendation_label": recommendation_label,
        "is_free_bet":          is_free,
    }
