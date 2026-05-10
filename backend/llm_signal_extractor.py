"""
llm_signal_extractor.py — LLM-based signal extraction from user reasoning (v2).

Uses Claude claude-haiku-4-5-20251001 to extract structured features from a user's
pick reasoning text.  Runs alongside the v1 keyword extractor (user_signal_learning.py)
and writes to the same user_pick_signals table with feature_source='llm'.

Cost: ~$0.0013 per pick at claude-haiku-4-5-20251001 pricing.
Failure mode: on any error, log and return empty features — never blocks pick submission.

Signal types produced (richer than keyword extractor):
  pitcher_command_aware, pitcher_mechanics_aware, pitcher_recent_form_aware,
  lineup_quality_aware, bullpen_exposure_aware, platoon_aware,
  weather_aware, rest_aware, motivation_aware,
  causal_reasoner, specific_predictor, contrarian_reasoner,
  high_complexity_reasoning, low_complexity_reasoning,
  confidence_lock, confidence_lean,
  bias_detected:revenge, bias_detected:hometown, bias_detected:hot_hand,
  bias_detected:chasing_loss, bias_detected:narrative_driven,
  bias_detected:regression_ignored
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from database import UserPick, UserPickLeg, UserPickSignal

# ── Client setup ──────────────────────────────────────────────────────────────
# Reuses ANTHROPIC_API_KEY from environment (same as slip_parser.py).
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model: claude-haiku-4-5-20251001 — fastest/cheapest, sufficient for JSON extraction
_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 700

# Minimum reasoning length before we bother calling the LLM
_MIN_REASONING_LEN = 30


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are analyzing a sports bettor's reasoning for a parlay pick.
Extract structured features that capture HOW they are handicapping,
not just WHAT teams they picked.

Pick details:
- Sport(s): {sports}
- Markets: {markets}
- Legs: {legs_summary}

User's reasoning:
\"\"\"{reasoning}\"\"\"

Extract these feature categories. Only include features explicitly
supported by the reasoning text. Do NOT invent features.

1. pitcher_factors — specific pitcher analysis
   Options: command_issues, mechanics_breakdown, recent_form_decline,
   platoon_advantage, splits_lefty, velocity_drop, pitch_mix_change,
   bb_rate_high, era_inflated, war_declining

2. matchup_factors — specific matchup analysis
   Options: lineup_patience, hot_lineup, lefty_heavy, home_road_split,
   divisional_rival, recent_head_to_head, bullpen_exposure_likely,
   lineup_platoon_favorable, pace_of_play_edge

3. situational — non-team factors
   Options: weather_wind_out, weather_cold, weather_rain_risk,
   day_game, travel_disadvantage, rest_advantage, back_to_back,
   playoff_atmosphere, must_win_motivation, schedule_spot

4. causal_chain — reasoning structure quality
   Options: mechanism_explained (user states WHY it will happen),
   prediction_specific (user predicts specific outcome or threshold),
   contrarian_logic (user explains why public/market is wrong),
   data_cited (user references specific stats)

5. confidence_signal — user's certainty language
   Options: lock, strong, love, lean, risky, value_play,
   longshot, hedge, just_enough, fade

6. potential_bias — cognitive bias flags
   Options: revenge_pick, hometown_bias, hot_hand,
   chasing_loss, narrative_driven, regression_ignored,
   recency_bias, anchoring

Return ONLY valid JSON, no other text:
{{
  "pitcher_factors": [],
  "matchup_factors": [],
  "situational": [],
  "causal_chain": [],
  "confidence_signal": [],
  "potential_bias": [],
  "key_excerpt": "the most informative sentence from their reasoning, or empty string",
  "complexity_score": 0.0
}}

complexity_score: 0.0 = vague/no reasoning, 1.0 = rich multi-factor analysis.
"""

# ── Feature-type → signal_type mapping ───────────────────────────────────────

# Maps (category, value) → signal_type stored in user_pick_signals
_CATEGORY_SIGNAL_MAP = {
    "pitcher_factors": {
        "command_issues":      "pitcher_command_aware",
        "mechanics_breakdown": "pitcher_mechanics_aware",
        "recent_form_decline": "pitcher_recent_form_aware",
        "platoon_advantage":   "platoon_aware",
        "splits_lefty":        "platoon_aware",
        "velocity_drop":       "pitcher_mechanics_aware",
        "pitch_mix_change":    "pitcher_mechanics_aware",
        "bb_rate_high":        "pitcher_command_aware",
        "era_inflated":        "pitcher_recent_form_aware",
        "war_declining":       "pitcher_recent_form_aware",
    },
    "matchup_factors": {
        "lineup_patience":         "lineup_quality_aware",
        "hot_lineup":              "lineup_quality_aware",
        "lefty_heavy":             "platoon_aware",
        "home_road_split":         "lineup_quality_aware",
        "divisional_rival":        "matchup_history_aware",
        "recent_head_to_head":     "matchup_history_aware",
        "bullpen_exposure_likely": "bullpen_exposure_aware",
        "lineup_platoon_favorable":"platoon_aware",
        "pace_of_play_edge":       "lineup_quality_aware",
    },
    "situational": {
        "weather_wind_out":    "weather_aware",
        "weather_cold":        "weather_aware",
        "weather_rain_risk":   "weather_aware",
        "day_game":            "situational_aware",
        "travel_disadvantage": "rest_aware",
        "rest_advantage":      "rest_aware",
        "back_to_back":        "rest_aware",
        "playoff_atmosphere":  "motivation_aware",
        "must_win_motivation": "motivation_aware",
        "schedule_spot":       "rest_aware",
    },
    "causal_chain": {
        "mechanism_explained":  "causal_reasoner",
        "prediction_specific":  "specific_predictor",
        "contrarian_logic":     "contrarian_reasoner",
        "data_cited":           "data_driven_reasoner",
    },
    "confidence_signal": {
        "lock":       "confidence_lock",
        "strong":     "confidence_lock",
        "love":       "confidence_lock",
        "lean":       "confidence_lean",
        "risky":      "confidence_lean",
        "value_play": "confidence_lean",
        "longshot":   "confidence_lean",
        "hedge":      "confidence_lean",
        "just_enough":"confidence_lean",
        "fade":       "confidence_lean",
    },
    "potential_bias": {
        "revenge_pick":         "bias_detected:revenge",
        "hometown_bias":        "bias_detected:hometown",
        "hot_hand":             "bias_detected:hot_hand",
        "chasing_loss":         "bias_detected:chasing_loss",
        "narrative_driven":     "bias_detected:narrative_driven",
        "regression_ignored":   "bias_detected:regression_ignored",
        "recency_bias":         "bias_detected:recency_bias",
        "anchoring":            "bias_detected:anchoring",
    },
}


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> Optional[dict]:
    """Call Claude and return parsed JSON dict, or None on failure."""
    if not _ANTHROPIC_KEY:
        print("[llm-extractor] ANTHROPIC_API_KEY not set — skipping LLM extraction")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        response = client.messages.create(
            model       = _MODEL,
            max_tokens  = _MAX_TOKENS,
            messages    = [{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[llm-extractor] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"[llm-extractor] API error: {e}")
        return None


def _build_prompt(pick: UserPick, legs: list[UserPickLeg]) -> str:
    sports   = ", ".join({l.sport for l in legs if l.sport}) or "Unknown"
    markets  = ", ".join({l.market_type for l in legs if l.market_type}) or "Unknown"
    legs_sum = " | ".join(
        f"{l.description}" + (f" ({l.team})" if l.team else "")
        for l in legs
    ) or "No legs"
    reasoning = (pick.notes or "").strip()
    return _EXTRACTION_PROMPT.format(
        sports      = sports,
        markets     = markets,
        legs_summary= legs_sum,
        reasoning   = reasoning,
    )


# ── Signal row builder ────────────────────────────────────────────────────────

def _features_to_signals(
    pick:       UserPick,
    features:   dict,
    complexity: float,
) -> list[dict]:
    """
    Convert LLM-extracted feature dict into signal rows ready for DB insert.
    Returns list of dicts (not ORM objects — caller creates them).
    """
    rows: list[dict] = []
    key_excerpt = features.get("key_excerpt") or None

    for category, values in features.items():
        if category in ("key_excerpt", "complexity_score"):
            continue
        if not isinstance(values, list):
            continue
        cat_map = _CATEGORY_SIGNAL_MAP.get(category, {})
        seen_signal_types: set[str] = set()   # dedupe within category

        for val in values:
            if not isinstance(val, str):
                continue
            signal_type = cat_map.get(val, f"{category}:{val}")
            if signal_type in seen_signal_types:
                continue
            seen_signal_types.add(signal_type)
            rows.append({
                "signal_type":       signal_type,
                "signal_key":        f"{category}:{val}",
                "feature_source":    "llm",
                "feature_complexity":complexity,
                "reasoning_excerpt": key_excerpt,
            })

    # Complexity-level meta-signals
    if complexity >= 0.7:
        rows.append({
            "signal_type":       "high_complexity_reasoning",
            "signal_key":        f"complexity:{complexity:.1f}",
            "feature_source":    "llm",
            "feature_complexity":complexity,
            "reasoning_excerpt": key_excerpt,
        })
    elif complexity < 0.3:
        rows.append({
            "signal_type":       "low_complexity_reasoning",
            "signal_key":        f"complexity:{complexity:.1f}",
            "feature_source":    "llm",
            "feature_complexity":complexity,
            "reasoning_excerpt": key_excerpt,
        })

    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def extract_llm_signals(
    pick: UserPick,
    legs: list[UserPickLeg],
    db:   Session,
) -> tuple[list[UserPickSignal], float]:
    """
    Run LLM extraction on pick reasoning.  Writes user_pick_signals rows
    with feature_source='llm'.  Updates pick.llm_features_extracted and
    pick.reasoning_complexity_score.

    Returns (signals_written, complexity_score).
    Never raises — failures are logged and an empty list is returned.
    """
    reasoning = (pick.notes or "").strip()
    if len(reasoning) < _MIN_REASONING_LEN:
        print(f"[llm-extractor] pick {pick.id}: reasoning too short ({len(reasoning)} chars), skipping")
        return [], 0.0

    prompt   = _build_prompt(pick, legs)
    features = _call_llm(prompt)

    if not features:
        _mark_pick(pick, extracted=False, complexity=0.0)
        db.flush()
        return [], 0.0

    complexity = float(features.get("complexity_score", 0.0))
    signal_rows = _features_to_signals(pick, features, complexity)

    written: list[UserPickSignal] = []
    for row in signal_rows:
        sig = UserPickSignal(
            user_pick_id       = pick.id,
            user_pick_leg_id   = None,      # LLM signals are pick-level
            signal_type        = row["signal_type"],
            signal_key         = row["signal_key"],
            feature_source     = row.get("feature_source", "llm"),
            feature_complexity = row.get("feature_complexity"),
            reasoning_excerpt  = row.get("reasoning_excerpt"),
            outcome            = "PENDING",
        )
        db.add(sig)
        written.append(sig)

    _mark_pick(pick, extracted=True, complexity=complexity)
    db.flush()

    print(
        f"[llm-extractor] pick {pick.id}: extracted {len(written)} signals "
        f"(complexity={complexity:.2f})"
    )
    return written, complexity


def _mark_pick(pick: UserPick, extracted: bool, complexity: float) -> None:
    try:
        pick.llm_features_extracted      = 1 if extracted else 0
        pick.reasoning_complexity_score  = complexity
    except Exception:
        pass  # columns may not exist on older rows


# ── Backfill ──────────────────────────────────────────────────────────────────

def backfill_llm_features(db: Session, limit: int = 100) -> dict:
    """
    Run LLM extraction on all user_picks where llm_features_extracted=0
    and notes is non-empty.  Processes at most `limit` picks per call.

    Returns summary: {processed, skipped_short, skipped_no_notes, errors, total_signals}
    """
    picks = db.query(UserPick).filter(
        UserPick.llm_features_extracted != 1,
        UserPick.notes.isnot(None),
        UserPick.notes != "",
    ).limit(limit).all()

    processed = skipped_short = errors = total_signals = 0

    for pick in picks:
        if len((pick.notes or "").strip()) < _MIN_REASONING_LEN:
            # Mark so we don't retry endlessly
            _mark_pick(pick, extracted=False, complexity=0.0)
            skipped_short += 1
            continue
        try:
            legs = db.query(UserPickLeg).filter(
                UserPickLeg.user_pick_id == pick.id
            ).all()
            sigs, _ = extract_llm_signals(pick, legs, db)
            total_signals += len(sigs)
            processed += 1
        except Exception as e:
            print(f"[llm-extractor] backfill error for {pick.id}: {e}")
            errors += 1

    db.commit()
    return {
        "processed":      processed,
        "skipped_short":  skipped_short,
        "errors":         errors,
        "total_signals":  total_signals,
        "picks_remaining": db.query(UserPick).filter(
            UserPick.llm_features_extracted != 1,
            UserPick.notes.isnot(None),
            UserPick.notes != "",
        ).count(),
    }


# ── Feature-source comparison (for learning-curve endpoint) ──────────────────

def feature_source_comparison(db: Session) -> dict:
    """
    Compare prediction quality of keyword vs LLM extracted signals.
    Returns stats per source and LLM advantage in percentage points.
    """
    from database import UserSignalPerformance

    # All signal performances — group by feature_source
    # (UserPickSignal has feature_source; join to UserSignalPerformance by signal_key)
    # Simpler: query UserPickSignal directly, group settled outcomes by source
    from database import UserPickSignal as _UPS
    from sqlalchemy import func

    result: dict = {}
    for source in ("keyword", "llm"):
        sigs = db.query(_UPS).filter(
            _UPS.feature_source == source,
            _UPS.outcome.in_(["WON", "LOST"]),
        ).all()

        n      = len(sigs)
        wins   = sum(1 for s in sigs if s.outcome == "WON")
        losses = n - wins
        wr     = round(wins / n * 100, 1) if n else None

        # Count distinct (signal_type, signal_key) patterns
        from database import UserSignalPerformance as _USP
        # We need to count patterns classified per source:
        # Proxy: count distinct signal keys that appear in settled signals for this source
        pattern_keys = {(s.signal_type, s.signal_key) for s in sigs}
        edge_pos = edge_neg = neutral = insuf = 0
        for stype, skey in pattern_keys:
            perf = db.query(_USP).filter(
                _USP.signal_type == stype,
                _USP.signal_key  == skey,
            ).first()
            if not perf:
                insuf += 1
            elif perf.pattern_class == "EDGE_POSITIVE":
                edge_pos += 1
            elif perf.pattern_class == "EDGE_NEGATIVE":
                edge_neg += 1
            elif perf.pattern_class == "NEUTRAL":
                neutral += 1
            else:
                insuf += 1

        result[source] = {
            "settled_signals":        n,
            "patterns_classified":    len(pattern_keys),
            "edge_positive":          edge_pos,
            "edge_negative":          edge_neg,
            "neutral":                neutral,
            "insufficient_data":      insuf,
            "overall_win_rate_pct":   wr,
        }

    # LLM advantage
    kw_wr  = result.get("keyword", {}).get("overall_win_rate_pct")
    llm_wr = result.get("llm",     {}).get("overall_win_rate_pct")
    if kw_wr is not None and llm_wr is not None:
        result["llm_advantage_pct_pts"] = round(llm_wr - kw_wr, 1)
    else:
        result["llm_advantage_pct_pts"] = None

    return result
