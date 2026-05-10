"""
slip_parser.py — Bet slip image parser using Claude vision API.

Sends a bet slip screenshot to Claude claude-sonnet-4-20250514, extracts all legs
into structured JSON, and records them as a placed bet in the DB.
"""
from __future__ import annotations
import base64
import json
import strategy as strat
import os
import math
import uuid
import requests
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from database import Bet

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-5"


# ── Vision extraction ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a sports betting slip parser. 
Extract every leg from the bet slip image and return ONLY valid JSON, no other text.

Return this exact structure:
{
  "sportsbook": "FanDuel",
  "bet_type": "parlay",
  "n_legs": 13,
  "combined_american_odds": 428,
  "legs": [
    {
      "team": "Miami Heat",
      "market": "Moneyline",
      "point": null,
      "american_odds": -2200,
      "game": "Miami Heat @ Washington Wizards",
      "game_time": "6:10PM CT"
    }
  ]
}

Rules:
- market must be one of: Moneyline, Spread, Alt Spread, Total, Alt Total, Player Prop
- point is null for Moneyline, a number for spreads/totals (e.g. 6.5, -3.5, 220.5)
- american_odds is always an integer (e.g. -2200, +428, -110)
- combined_american_odds is the total parlay odds shown (e.g. 428 for +428)
- If it's a single straight bet, set bet_type to "straight" and n_legs to 1
- game_time can be null if not visible
- Return ONLY the JSON object, no markdown, no explanation
"""

def parse_slip_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    """
    Send bet slip image to Claude vision API.
    Returns structured dict with legs and parlay info.
    """
    if not ANTHROPIC_API_KEY:
        return {
            "error": "ANTHROPIC_API_KEY not set.",
            "detail": "Run in terminal: export ANTHROPIC_API_KEY=your-key-here then restart START.sh",
            "setup": True
        }

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": 2000,
        "system":     SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type":   "image",
                        "source": {
                            "type":       "base64",
                            "media_type": mime_type,
                            "data":       image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract all legs from this bet slip. Return only JSON."
                    }
                ],
            }
        ],
    }

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data     = r.json()
        raw_text = data["content"][0]["text"].strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        return _enrich(parsed)

    except requests.exceptions.RequestException as e:
        status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        body   = ""
        try:
            body = e.response.text[:300] if hasattr(e, "response") and e.response else ""
        except Exception:
            pass
        return {"error": f"API request failed (HTTP {status}): {body or str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse Claude response as JSON: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def _enrich(parsed: dict) -> dict:
    """Add computed fields to the parsed slip for display and EV analysis."""
    legs  = parsed.get("legs", [])
    n     = len(legs)

    # Combined decimal odds
    combined_am  = parsed.get("combined_american_odds")
    if combined_am is not None:
        if combined_am > 0:
            combined_dec = round(1 + combined_am / 100, 4)
        else:
            combined_dec = round(1 + 100 / abs(combined_am), 4)
    else:
        # Compute from legs
        combined_dec = 1.0
        for leg in legs:
            am = leg.get("american_odds", -110)
            if am > 0:  combined_dec *= (1 + am / 100)
            elif am < 0: combined_dec *= (1 + 100 / abs(am))
        combined_dec = round(combined_dec, 4)

    # Per-leg implied win probability
    for leg in legs:
        am = leg.get("american_odds", -110)
        if am > 0:   dec = 1 + am / 100
        elif am < 0: dec = 1 + 100 / abs(am)
        else:        dec = 1.0
        leg["decimal_odds"]  = round(dec, 4)
        leg["implied_prob"]  = round(1 / dec * 100, 1) if dec > 1 else 0

    # Model-based combined win prob (85.71% accuracy per leg)
    MODEL_ACCURACY   = 0.8571
    model_combined   = MODEL_ACCURACY ** n
    naive_combined   = 1.0
    for leg in legs:
        naive_combined *= (leg["implied_prob"] / 100)

    payout_10        = round((combined_dec - 1) * 10, 2)
    ev_model         = round(model_combined * payout_10 - (1 - model_combined) * 10, 2)
    ev_implied       = round(naive_combined * payout_10 - (1 - naive_combined) * 10, 2)

    # Kelly
    b          = combined_dec - 1
    q          = 1 - model_combined
    raw_kelly  = (b * model_combined - q) / b if b > 0 else -1
    kelly_rec  = max(raw_kelly * 0.25, 0)

    enriched = {
        **parsed,
        "n_legs":            n,
        "combined_decimal":  combined_dec,
        "model_win_prob":    round(model_combined * 100, 2),
        "implied_win_prob":  round(naive_combined * 100, 2),
        "payout_on_10":      payout_10,
        "ev_model":          ev_model,
        "ev_implied":        ev_implied,
        "kelly_fraction":    round(kelly_rec * 100, 4),
        "kelly_rec":         "BET ${:.2f} per $100".format(kelly_rec * 100) if kelly_rec > 0
                             else "DO NOT BET — negative EV",
        "verdict":           _verdict(n, model_combined, ev_model, kelly_rec),
    }
    # Add BetIQ recommendation
    # Strategy classification and grading
    legs_for_strat = [
        {"market": l.get("market", "Moneyline"),
         "american_odds": l.get("american_odds", -110)}
        for l in parsed.get("legs", [])
    ]
    cam = parsed.get("combined_american_odds")
    if cam is None:
        cam = round((combined_dec-1)*100) if combined_dec >= 2               else -round(100/(combined_dec-1))
    graded = strat.grade_bet(legs_for_strat, combined_dec, cam, 10.0)
    enriched.update({
        "bet_class":     graded["bet_class"],
        "grade":         graded["grade"],
        "grade_color":   graded["color"],
        "grade_summary": graded["summary"],
        "strategy_note": graded["strategy_note"],
        "hist_win_prob": graded["win_prob"],
        "hist_ev":       graded["ev"],
        "kelly_500":     graded["kelly_500"],
    })
    enriched["recommendation"] = build_recommendation(enriched, stake=10.0)
    return enriched


def _verdict(n_legs: int, win_prob: float, ev: float, kelly: float) -> dict:
    if kelly <= 0 or ev < 0:
        grade  = "F"
        color  = "red"
        summary = (
            f"Negative EV (${ev:.2f} per $10). Kelly says don't bet. "
            f"The {n_legs}-leg chain reduces win prob to {win_prob*100:.1f}%."
        )
    elif win_prob >= 0.60:
        grade  = "A"
        color  = "green"
        summary = f"Strong pick. {win_prob*100:.1f}% win prob, positive EV."
    elif win_prob >= 0.45:
        grade  = "B"
        color  = "blue"
        summary = f"Decent pick. {win_prob*100:.1f}% win prob."
    elif win_prob >= 0.30:
        grade  = "C"
        color  = "amber"
        summary = f"Risky. Only {win_prob*100:.1f}% win prob."
    else:
        grade  = "D"
        color  = "red"
        summary = (
            f"Very risky — {win_prob*100:.1f}% win prob. "
            f"Consider breaking into {max(2, n_legs//3)}-leg parlays."
        )

    suggestion = None
    if n_legs > 4:
        suggestion = (
            f"Instead of {n_legs} legs: pick your 2-3 highest confidence legs "
            f"for ~{0.8571**3*100:.0f}% win prob with positive EV."
        )

    return {
        "grade":      grade,
        "color":      color,
        "summary":    summary,
        "suggestion": suggestion,
    }



# ── Recommendation engine ─────────────────────────────────────────────────────

import itertools as _itertools

def _ev_for_subset(legs: list, stake: float = 10.0) -> tuple:
    """Returns (ev, win_prob_pct, combined_dec, combined_am)."""
    n            = len(legs)
    combined_dec = 1.0
    for l in legs:
        am = l.get("american_odds", -110)
        combined_dec *= (1 + am/100) if am > 0 else (1 + 100/abs(am))
    win_prob     = MODEL_ACCURACY ** n
    payout       = (combined_dec - 1) * stake
    ev           = win_prob * payout - (1 - win_prob) * stake
    combined_am  = round((combined_dec-1)*100) if combined_dec >= 2                    else -round(100/(combined_dec-1))
    return round(ev,4), round(win_prob*100,2), round(combined_dec,4), combined_am


MODEL_ACCURACY = 0.8571


def build_recommendation(parsed: dict, stake: float = 10.0) -> dict:
    """
    Given a parsed slip, produce a BetIQ recommendation:
      - AGREE: full parlay has positive EV and reasonable win prob
      - TRUNCATE: recommend a shorter subset with reasoning for each dropped leg
      - SKIP: even the best subset has negative EV (common for SGP+)

    Returns a recommendation dict that gets merged into the parsed slip.
    """
    legs    = parsed.get("legs", [])
    n       = len(legs)
    is_sgp  = "same game" in (parsed.get("bet_type","") + parsed.get("sportsbook","")).lower()

    if not legs:
        return {"action": "SKIP", "reason": "No legs found."}

    # ── Full parlay analysis ──────────────────────────────────────────────────
    full_ev, full_wp, full_dec, full_am = _ev_for_subset(legs, stake)
    full_payout = round((full_dec-1)*stake, 2)

    # ── Find best subset (2 to min(4, n-1) legs) ─────────────────────────────
    best_ev      = full_ev
    best_combo   = tuple(range(n))   # start with full set
    best_n       = n

    # Try shorter combos — cap at 4 legs max for high win prob
    search_max = min(4, n-1)
    for size in range(2, search_max+1):
        for combo in _itertools.combinations(range(n), size):
            subset = [legs[i] for i in combo]
            ev, wp, dec, am = _ev_for_subset(subset, stake)
            if ev > best_ev:
                best_ev    = ev
                best_combo = combo
                best_n     = size

    best_legs    = [legs[i] for i in best_combo]
    kept_indices = set(best_combo)
    dropped      = [(i, legs[i]) for i in range(n) if i not in kept_indices]

    best_ev_val, best_wp, best_dec, best_am = _ev_for_subset(best_legs, stake)
    best_payout  = round((best_dec-1)*stake, 2)

    # ── Kelly for best subset ─────────────────────────────────────────────────
    b             = best_dec - 1
    q             = 1 - best_wp/100
    raw_k         = (b*(best_wp/100) - q) / b if b > 0 else -1
    kelly_25_pct  = max(raw_k * 0.25, 0)
    kelly_stake   = round(500 * min(kelly_25_pct, 0.10), 2)

    # ── Decision logic ────────────────────────────────────────────────────────
    if is_sgp:
        # SGP legs are correlated — can't partially place
        if full_ev >= 0:
            action  = "AGREE"
            reason  = (
                f"This SGP+ has estimated positive EV (${full_ev:+.2f}) and "
                f"{full_wp}% win probability. Same-game parlays cannot be partially "
                f"placed — it's all-or-nothing. BetIQ agrees with the full bet."
            )
        else:
            action  = "SKIP"
            reason  = (
                f"This SGP+ has negative EV (${full_ev:+.2f} on ${stake}). "
                f"Same-game parlays can't be partially placed so truncation isn't "
                f"an option. The +{parsed.get('combined_american_odds',0)} payout "
                f"doesn't compensate for the {full_wp}% win probability. "
                f"BetIQ recommends skipping this one."
            )
        return {
            "action":           action,
            "is_sgp":           True,
            "full_ev":          full_ev,
            "full_win_prob":    full_wp,
            "full_payout":      full_payout,
            "reason":           reason,
            "rec_legs":         legs if action == "AGREE" else [],
            "dropped_legs":     [],
            "kelly_stake":      kelly_stake if action == "AGREE" else 0,
        }

    # Regular parlay decision
    if full_ev >= 2.0 and best_n == n:
        # Full parlay is already the best and has strong positive EV
        action = "AGREE"
        reason = (
            f"BetIQ agrees with this {n}-leg parlay. "
            f"Expected profit is ${full_ev:+.2f} per ${stake} bet with "
            f"{full_wp}% win probability at +{full_am} odds. "
            f"All legs contribute positively — no truncation needed."
        )
        return {
            "action":        "AGREE",
            "is_sgp":        False,
            "full_ev":       full_ev,
            "full_win_prob": full_wp,
            "full_payout":   full_payout,
            "reason":        reason,
            "rec_legs":      legs,
            "dropped_legs":  [],
            "kelly_stake":   kelly_stake,
            "kelly_pct":     round(kelly_25_pct*100, 2),
        }

    elif best_ev > full_ev or best_n < n:
        # Truncation improves EV
        action = "TRUNCATE"

        # Explain each dropped leg
        drop_reasons = []
        for i, leg in dropped:
            am  = leg.get("american_odds", -110)
            dec = (1+am/100) if am>0 else (1+100/abs(am))
            imp = round(1/dec*100, 1)
            # Find what removing this leg does to EV improvement
            without = [legs[j] for j in range(n) if j != i and j in kept_indices or j in kept_indices]
            reason_leg = (
                f"Implied win prob {imp}% is "
                f"{'the lowest in this parlay' if imp < 70 else 'weaker than kept legs'} — "
                f"removing it raises combined win probability without enough payout gain"
            )
            drop_reasons.append({
                "leg":    leg,
                "reason": reason_leg,
            })

        ev_gain  = round(best_ev - full_ev, 2)
        wp_gain  = round(best_wp - full_wp, 1)
        reason = (
            f"BetIQ recommends truncating from {n} to {best_n} legs. "
            f"Dropping {n-best_n} leg{'s' if n-best_n>1 else ''} improves "
            f"win probability from {full_wp}% → {best_wp}% "
            f"and expected profit from ${full_ev:+.2f} → ${best_ev:+.2f} per ${stake}. "
            f"The kept legs have the best combination of odds and implied probability."
        )
        return {
            "action":           "TRUNCATE",
            "is_sgp":           False,
            "full_ev":          full_ev,
            "full_win_prob":    full_wp,
            "full_payout":      full_payout,
            "rec_n_legs":       best_n,
            "rec_ev":           best_ev,
            "rec_win_prob":     best_wp,
            "rec_payout":       best_payout,
            "rec_american_odds":best_am,
            "rec_legs":         best_legs,
            "dropped_legs":     drop_reasons,
            "ev_improvement":   ev_gain,
            "wp_improvement":   wp_gain,
            "reason":           reason,
            "kelly_stake":      kelly_stake,
            "kelly_pct":        round(kelly_25_pct*100, 2),
        }

    else:
        # Even the best subset has negative or marginal EV
        action = "SKIP"
        reason = (
            f"Even the best {best_n}-leg subset has only ${best_ev:+.2f} EV "
            f"at {best_wp}% win probability. "
            f"The payout (+{best_am}) doesn't compensate for the risk. "
            f"BetIQ recommends skipping this bet entirely."
        )
        return {
            "action":        "SKIP",
            "is_sgp":        False,
            "full_ev":       full_ev,
            "full_win_prob": full_wp,
            "full_payout":   full_payout,
            "rec_legs":      [],
            "dropped_legs":  [],
            "reason":        reason,
            "kelly_stake":   0,
        }

# ── Save to DB ────────────────────────────────────────────────────────────────

def save_parsed_slip(parsed: dict, stake: float, is_mock: bool,
                     db: Session) -> dict:
    """
    Save a parsed bet slip as a placed bet in the DB.
    All legs are recorded in bet_info as pipe-separated descriptions.
    """
    legs     = parsed.get("legs", [])
    n        = len(legs)

    if n == 0:
        return {"error": "No legs found in parsed slip."}

    # Build bet_info string (same format as Pikkit CSV)
    bet_info_parts = []
    for leg in legs:
        point_str = ""
        if leg.get("point") is not None:
            p = leg["point"]
            point_str = f" {'+' if p > 0 else ''}{p}"
        bet_info_parts.append(
            f"{leg['team']}{point_str} ({leg['market']}) {leg.get('american_odds','')}"
        )
    bet_info  = " | ".join(bet_info_parts)

    sports    = " | ".join(set(leg.get("sport", "Basketball") for leg in legs))
    leagues   = "NBA"  # default for NBA slips; could be detected

    combined_dec = parsed.get("combined_decimal")
    if not combined_dec:
        # Compute from combined_american_odds
        cam = parsed.get("combined_american_odds")
        if cam is not None:
            combined_dec = round((1 + cam/100) if cam > 0 else (1 + 100/abs(cam)), 4)
        else:
            # Compute from legs
            combined_dec = 1.0
            for leg in legs:
                am = leg.get("american_odds", -110)
                combined_dec *= (1 + am/100) if am > 0 else (1 + 100/abs(am))
            combined_dec = round(combined_dec, 4)

    # Detect sport from legs
    sport_keywords = {
        "baseball": "Baseball", "mlb": "Baseball",
        "basketball": "Basketball", "nba": "Basketball",
        "football": "American Football", "nfl": "American Football",
        "hockey": "Ice Hockey", "nhl": "Ice Hockey",
        "soccer": "Soccer",
    }
    detected_sport = "Basketball"  # default
    all_text = " ".join(str(l.get("team","")) + " " + str(l.get("game","")) for l in legs).lower()
    for kw, sp in sport_keywords.items():
        if kw in all_text:
            detected_sport = sp
            break
    sports  = detected_sport
    leagues = parsed.get("sport_title", "") or detected_sport

    bet = Bet(
        id          = str(uuid.uuid4()),
        source      = "app",
        bet_type    = parsed.get("bet_type", "parlay"),
        sportsbook  = parsed.get("sportsbook", "FanDuel"),
        odds        = combined_dec,
        amount      = stake,
        legs        = n,
        sports      = sports,
        leagues     = leagues,
        bet_info    = bet_info,
        status      = "PLACED",
        is_mock     = is_mock,
        time_placed = datetime.utcnow(),
    )
    db.add(bet)
    db.commit()
    db.refresh(bet)

    return {
        "bet_id":    bet.id,
        "n_legs":    n,
        "odds":      combined_dec,
        "amount":    stake,
        "is_mock":   is_mock,
        "bet_info":  bet_info,
        "message":   f"{'Mock' if is_mock else 'Real'} bet recorded from slip image.",
    }
