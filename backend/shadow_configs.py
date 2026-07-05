"""
shadow_configs.py — Config-level shadow transform functions for Group D.

Each config shadow uses the active V3 model as the base prediction, then
applies a different post-processing transform.  The transform_fn signature:

    transform_fn(base_wp: float, row: pd.Series, spread_value: float,
                 ctx: dict) → float | None

base_wp is the V3 model's raw probability (0.0–1.0).
ctx holds pre-loaded lookup data (populated by _build_config_context in
shadow_registry.py).

IMPORTANT: These transforms NEVER touch the live scoring path.  They only
run inside the shadow harness (shadow_predictions table, bets nothing).
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# D1: V3 alt-spread discrimination
# ═══════════════════════════════════════════════════════════════════════════
# The live path overwrites V3's alt-spread output with market-implied prob
# (1/decimal_odds).  This config shadow passes V3's raw output through
# with a re-derived per-spread calibration correction.
#
# V3 spread calibration (from test-set evaluation):
#   spread -1.5: gap = -2.5pp (overconfident → subtract 2.5pp)
#   spread +1.5: gap = +1.0pp (underconfident → add 1.0pp)
#   spread +2.5: gap = +1.4pp
#   spread +3.5: gap = +1.5pp
#   spread +4.5: gap = +1.6pp
#
# These corrections are SPECIFIC to V3's augmented output — NOT the old
# implied_prob haircuts.  The live implied_prob haircuts stay live and
# untouched.

_V3_SPREAD_CAL = {
    -1.5: -0.025,
    1.5:   0.010,
    2.5:   0.014,
    3.5:   0.015,
    4.5:   0.016,
}

def d1_v3_alt_spread(base_wp, row, spread_value, ctx):
    """V3 model output + per-spread calibration correction.

    Instead of market-implied prob, use V3's actual prediction (which is
    spread-aware via augmentation) with a small additive calibration shift.
    """
    correction = _V3_SPREAD_CAL.get(spread_value, 0.0)
    return base_wp + correction


# ═══════════════════════════════════════════════════════════════════════════
# D2: CLV as a scoring signal — DISABLED (confound, not real signal)
# ═══════════════════════════════════════════════════════════════════════════
# The pooled inverse-CLV relationship (Q1 WR=64.5% > Q4 WR=54.0%) is a
# CONFOUND, not a real signal.  Analysis found:
#
# 1. Within-rung: direction inconsistent (+1.5 inverse, +2.5 normal, +3.5 flat)
# 2. Real pattern: U-shaped (extreme CLV either direction → worse), not
#    monotonic inversion.  Mid-CLV (|clv|<5) outperforms both extremes.
# 3. Extreme negative CLV driven by soccer/alt-line measurement artifacts
#    (team +2.5 in soccer priced at 99% → closing line collapses → huge
#    negative CLV, but the pick still wins because it was near-certain)
#
# DECISION: Do not invert.  Config deactivated.  If CLV is revisited,
# the signal is U-shaped penalization of extreme CLV, not directional.
#
# Original implementation preserved below for reference.

_CLV_SCALE = 0.02 / 5.0   # 2pp per 5 cents of CLV signal
_CLV_CAP   = 0.04          # max ±4pp adjustment

def d2_clv_signal(base_wp, row, spread_value, ctx):
    """Adjust V3 prediction by historical CLV signal for this spread bucket."""
    clv_by_spread = ctx.get("clv_avg_by_spread", {})
    clv_val = clv_by_spread.get(round(spread_value, 1))

    if clv_val is None:
        # No CLV data for this spread → use overall average
        clv_val = ctx.get("clv_overall_avg", 0.0)

    if clv_val is None or clv_val == 0.0:
        return base_wp

    # Negative CLV = market moved against us but we won → our picks are
    # undervalued at this spread → boost.  Positive CLV = we got a good
    # price but lost → our picks are overvalued → penalize.
    # INVERT the CLV signal because of the inverse relationship.
    adjustment = max(-_CLV_CAP, min(_CLV_CAP, -clv_val * _CLV_SCALE))
    return base_wp + adjustment


# ═══════════════════════════════════════════════════════════════════════════
# D3: Per-rung WR + directional accuracy
# ═══════════════════════════════════════════════════════════════════════════
# The spread "rung" is the distance from the main market line (line_delta).
# Historical data shows dramatic WR variation by rung:
#   rung  0 (main line):   WR = 100%  (pushes excluded, trivial)
#   rung +1 (1 tick out):  WR = 74.8%
#   rung +2 (2 ticks):     WR = 88.7%
#   rung +3:               WR = 38.1%
#   rung +5:               WR = 23.6%
#   rung -5:               WR = 77.1%
#
# The config adjusts V3's base_wp by comparing the rung's historical WR
# to the base_wp.  If the rung historically over/underperforms V3's
# prediction at that spread, we shift toward the empirical rate.
#
# Blending: adjusted_wp = 0.7 * base_wp + 0.3 * rung_empirical_wr
# (conservative — mostly trust V3, but nudge toward rung evidence)
#
# For spreads that map to specific rungs, we know the likely rung:
#   spread +1.5 → most common, rung varies by game line
#   We approximate: V3 spreads map to these typical rung distances:
#     -1.5 → favorite covering -1.5 → rung depends on main line
#     +1.5 → underdog +1.5 → closest rung = 0 or +1
#
# Since we don't know the exact main line here, we use the overall
# per-spread empirical WR as the rung proxy.

# Empirical WR from historical mock_bet_legs by spread_value
# (only for spreads with n >= 50 settled legs)
_SPREAD_EMPIRICAL_WR = {
    -1.5: 0.402,    # n=234, from active path data
    1.5:  0.636,    # n=3046
    2.5:  0.746,    # n=1010
    3.5:  0.800,    # n=579
    4.5:  0.761,    # n=180
}

_RUNG_BLEND = 0.3   # weight on empirical rung WR (0.7 on V3 base)

def d3_rung_directional(base_wp, row, spread_value, ctx):
    """Blend V3 prediction with per-spread empirical WR."""
    empirical_wr = _SPREAD_EMPIRICAL_WR.get(spread_value)
    if empirical_wr is None:
        return base_wp

    blended = (1.0 - _RUNG_BLEND) * base_wp + _RUNG_BLEND * empirical_wr
    return blended


# ═══════════════════════════════════════════════════════════════════════════
# D5: +1.5 confidence haircut
# ═══════════════════════════════════════════════════════════════════════════
# Diagnosis (6/22-6/25, 29 settled +1.5 legs):
#   Model predicts 86.6% WP at +1.5, actual WR = 72.4% → +14.2pp gap
#   Both alt and standard +1.5 show identical model predictions (86.6%)
#   — the alt/standard distinction is about juice, not the spread itself.
#   In MLB the run line is always ±1.5, so close_spread is always ±1.5.
#
# Anti-overfit sweep found NO clean EV floor (losses scattered across
# the full EV range, max EV = 0.263 so 0.30 kills everything).
#
# Mechanism: flat 5pp haircut on ALL spread_value=1.5 predictions.
# This is half the observed 14.2pp gap — conservative because:
#   1. 29 legs over 4 days is noisy
#   2. Gap may regress toward the long-run empirical +1.5 WR (63.6%)
#
# Forward validation: when D5 hits 100+ settled +1.5 predictions:
#   - Compare Brier on +1.5 subset vs shadow_v3 (base model)
#   - Check whether the actual gap held at ~14pp or regressed
#   - If gap held → grow haircut toward full gap; if shrank → shrink
#   - Don't lock 5pp — let the data size it

_15_HAIRCUT = 0.05   # 5pp discount on predicted probability

def d5_alt_15_haircut(base_wp, row, spread_value, ctx):
    """Discount V3 prediction at +1.5 spread.

    The model is systematically overconfident at +1.5 (predicts ~87%,
    actual ~72%).  Apply a flat 5pp haircut to test whether this
    improves calibration vs the base model.
    """
    if spread_value != 1.5:
        return base_wp

    return base_wp - _15_HAIRCUT


# ═══════════════════════════════════════════════════════════════════════════
# D3b: Light blend (10% weight on empirical WR vs D3's 30%)
# ═══════════════════════════════════════════════════════════════════════════
# Same mechanism as D3, but with a lighter touch.  If D3 at 30% is best,
# D3b at 10% should be worse — confirming that MORE blend is needed, not
# less.  If D3b beats D3, the optimal weight is below 30%.

_RUNG_BLEND_LIGHT = 0.10

def d3b_rung_light(base_wp, row, spread_value, ctx):
    """Light 10% blend of V3 prediction with per-spread empirical WR."""
    empirical_wr = _SPREAD_EMPIRICAL_WR.get(spread_value)
    if empirical_wr is None:
        return base_wp
    return (1.0 - _RUNG_BLEND_LIGHT) * base_wp + _RUNG_BLEND_LIGHT * empirical_wr


# ═══════════════════════════════════════════════════════════════════════════
# D3c: Targeted blend — only +2.5 and +3.5
# ═══════════════════════════════════════════════════════════════════════════
# D3's per-rung analysis shows V3 is underconfident at ALL positive spreads.
# D3c tests whether targeting just the biggest-gap rungs (+2.5, +3.5)
# avoids any harm from blending at other rungs.

_D3C_TARGET_SPREADS = {2.5, 3.5}

def d3c_rung_targeted(base_wp, row, spread_value, ctx):
    """30% blend only at +2.5 and +3.5; pass-through elsewhere."""
    if spread_value not in _D3C_TARGET_SPREADS:
        return base_wp
    empirical_wr = _SPREAD_EMPIRICAL_WR.get(spread_value)
    if empirical_wr is None:
        return base_wp
    return (1.0 - _RUNG_BLEND) * base_wp + _RUNG_BLEND * empirical_wr


# ═══════════════════════════════════════════════════════════════════════════
# D3d: Heavy blend (50% weight) — probing toward optimal
# ═══════════════════════════════════════════════════════════════════════════
# Optimal blend weights per rung are all >1.0 on 5-day data — meaning
# even 100% empirical WR isn't enough.  But that's likely overfit to a
# small sample.  50% is a step toward optimal while staying conservative.
# If D3d beats D3, the mechanism has room to run.

_RUNG_BLEND_HEAVY = 0.50

def d3d_rung_heavy(base_wp, row, spread_value, ctx):
    """Heavy 50% blend of V3 prediction with per-spread empirical WR."""
    empirical_wr = _SPREAD_EMPIRICAL_WR.get(spread_value)
    if empirical_wr is None:
        return base_wp
    return (1.0 - _RUNG_BLEND_HEAVY) * base_wp + _RUNG_BLEND_HEAVY * empirical_wr


# ═══════════════════════════════════════════════════════════════════════════
# Registration — called at import time by shadow_registry or scheduler
# ═══════════════════════════════════════════════════════════════════════════

def register_all_configs():
    """Register all D-group config shadows.  Idempotent — safe to call
    multiple times."""
    from shadow_registry import register_shadow_config

    register_shadow_config(
        "config_d1_v3_alts",
        d1_v3_alt_spread,
        meta={"version": "d1_v1", "description": "V3 alt-spread output + per-spread calibration",
              "group": "D1"},
    )
    register_shadow_config(
        "config_d2_clv",
        d2_clv_signal,
        meta={"version": "d2_v1", "description": "V3 + CLV signal adjustment",
              "group": "D2"},
    )
    register_shadow_config(
        "config_d3_rung",
        d3_rung_directional,
        meta={"version": "d3_v1", "description": "V3 blended with per-spread empirical WR",
              "group": "D3"},
    )
    register_shadow_config(
        "config_d5_alt15",
        d5_alt_15_haircut,
        meta={"version": "d5_v1",
              "description": "5pp haircut on all +1.5 predictions (model overconfident at run line)",
              "group": "D5"},
    )
    register_shadow_config(
        "config_d3b_light",
        d3b_rung_light,
        meta={"version": "d3b_v1",
              "description": "10% blend with per-spread empirical WR (light touch)",
              "group": "D3"},
    )
    register_shadow_config(
        "config_d3c_targeted",
        d3c_rung_targeted,
        meta={"version": "d3c_v1",
              "description": "30% blend at +2.5/+3.5 only, pass-through elsewhere",
              "group": "D3"},
    )
    register_shadow_config(
        "config_d3d_heavy",
        d3d_rung_heavy,
        meta={"version": "d3d_v1",
              "description": "50% blend with per-spread empirical WR (probing toward optimal)",
              "group": "D3"},
    )
    log.info("[shadow-configs] registered D1/D2/D3/D3b/D3c/D3d/D5 config shadows")
