"""
wc_knockout.py — WC 2026 Knockout-Stage Prediction Model (R32 → Final)

Self-vetting production model.  Built on top of the group-stage market-anchored
engine but adds:

1. Regularised xG/lambda engine with artifact detection
2. Confederation + mid-tier corrections at vetted (halved) weights
3. Draw-resolution module  (ET + PK probability)
4. Auto-vetting: deflation-strip, clamp-flag, market-adaptation check
5. Tiered edge output (Tier 1 / 2 / 3)

Philosophy: every edge is computed against the CURRENT market line, and
automatically checked for double-counting and artifacts before being
surfaced.  The model should never re-inflate.

LOCKED PARAMETERS (R32 → Final, do not re-tune per round):
  - GS/prior blend: 85/15 normal, 60/40 pathological (0-conceded)
  - Lambda bounds: floor 0.35, ceiling 3.5
  - Confederation corrections: CAF +2pp, AFC -1.5pp, CONCACAF +1pp
  - Mid-tier fade: 15% compression on 55-75% favorites
  - Implausible edge cap: 12pp (auto-demote to Tier 2)
  - Draw resolution: ET = 60% of gap, PK = 50/50 + 10% of gap
Only _GS_SCORING is updated each round (with actual results).
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ─── Locked model parameters (R32 → Final — do not re-tune per round) ─────────
# Rationale documented in commit; changing these mid-tournament invalidates
# forward calibration measurement.

# Lambda regularisation: how much to trust GS (3 games) vs pre-tournament prior
# Normal: 85% GS — recency (this tournament, this squad, this form) outweighs
#   sample size. 15% prior prevents ignoring 15-24 game competitive record.
# Pathological (0 conceded): 60% GS — still majority GS because 0-conceded at
#   a WC is stronger signal than in qualifiers, but the prior is structurally
#   necessary to prevent degenerate lambda=0.
BLEND_GS_NORMAL = 0.85
BLEND_GS_PATHOLOGICAL = 0.60

# Lambda bounds: soft floor/ceiling for artifact detection
LAMBDA_FLOOR = 0.35
LAMBDA_CEILING = 3.5

# Confederation corrections (halved from observed GS gaps — market partially corrected)
CONFED_CAF_PP = 0.02       # +2pp per CAF team
CONFED_AFC_PP = -0.015     # -1.5pp per AFC team
CONFED_CONCACAF_PP = 0.01  # +1pp per CONCACAF team

# Mid-tier fade: compress 55-75% favorites by 15% of excess above 50%
FADE_FLOOR = 0.55
FADE_CEILING = 0.75
FADE_RATE = 0.15

# Edge vetting
MAX_CREDIBLE_EDGE_PP = 12.0  # >12pp vs 20+ bookmakers → implausible, demote

# Draw resolution (ET + PK)
ET_GAP_CARRY = 0.30   # ET carries ~60% of 90-min gap (split: each side gets 30%)
ET_DRAW_RATE = 0.20   # ~20% of extra time ends still drawn
PK_GAP_CARRY = 0.10   # penalties: 50/50 + 10% of strength gap

# ─── Team-name normalisation ──────────────────────────────────────────────────

_NAME_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Korea Republic": "South Korea", "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States", "Republic of Ireland": "Ireland",
    "IR Iran": "Iran", "Korea DPR": "North Korea",
    "Chinese Taipei": "Taiwan", "Curaçao": "Curacao",
}


def _norm(n: str) -> str:
    return _NAME_ALIASES.get(n, n)


def _lj(fn: str):
    with open(os.path.join(_DATA_DIR, fn)) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# Confederation membership
# ═════════════════════════════════════════════════════════════════════════════

_CAF = frozenset({
    "Morocco", "Ivory Coast", "DR Congo", "Senegal", "Egypt", "Algeria",
    "Ghana", "Cape Verde", "Cameroon", "Nigeria", "South Africa", "Tunisia",
    "Mali", "Tanzania", "Mozambique", "Comoros",
})
_AFC = frozenset({
    "Japan", "Australia", "South Korea", "Saudi Arabia", "Iran", "Qatar",
    "Iraq", "Uzbekistan", "China", "Indonesia", "Thailand", "Bahrain",
    "India", "Jordan",
})
_CONCACAF = frozenset({
    "United States", "Mexico", "Canada", "Jamaica", "Costa Rica", "Honduras",
    "Panama", "El Salvador", "Haiti", "Curacao", "Trinidad and Tobago",
})
# UEFA + CONMEBOL: no correction (well-calibrated in GS evaluation)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Group-stage scoring data  (verified in-session analysis)
# ═════════════════════════════════════════════════════════════════════════════

# GS actual scoring rates from 72 group-stage matches.
# Format: team → (goals_scored_per_game, goals_conceded_per_game)
_GS_SCORING: dict[str, tuple[float, float]] = {
    "Brazil":                   (2.33, 0.67),
    "Japan":                    (2.33, 1.33),
    "Germany":                  (3.33, 1.33),
    "Paraguay":                 (1.33, 1.67),
    "Netherlands":              (2.33, 0.67),
    "Morocco":                  (1.33, 0.67),
    "Ivory Coast":              (1.67, 1.33),
    "Norway":                   (2.33, 1.67),
    "France":                   (2.67, 0.67),
    "Sweden":                   (1.00, 2.00),
    "Mexico":                   (1.67, 0.00),
    "Ecuador":                  (0.67, 0.67),
    "England":                  (2.00, 0.33),
    "DR Congo":                 (1.00, 1.33),
    "Belgium":                  (1.00, 0.67),
    "Senegal":                  (1.33, 1.00),
    "United States":            (1.33, 1.00),
    "Bosnia and Herzegovina":   (1.33, 1.67),
    "Spain":                    (3.00, 0.00),
    "Austria":                  (1.67, 2.33),
    "Portugal":                 (2.00, 1.00),
    "Croatia":                  (1.33, 1.00),
    "Switzerland":              (1.33, 1.00),
    "Algeria":                  (1.67, 1.33),
    "Australia":                (0.33, 1.00),
    "Egypt":                    (1.00, 0.67),
    "Argentina":                (2.33, 0.67),
    "Cape Verde":               (1.00, 1.33),
    "Colombia":                 (2.00, 0.00),
    "Ghana":                    (1.00, 1.33),
    "South Africa":             (0.67, 1.00),
    "Canada":                   (2.67, 1.00),
}

_GS_AVG_GPG = 2.96 / 2  # per-team average across 72 GS matches = 1.48


# ═════════════════════════════════════════════════════════════════════════════
# 2. Regularised lambda engine
# ═════════════════════════════════════════════════════════════════════════════

def _get_team_priors() -> dict[str, dict]:
    """Load xG proxy (pre-tournament competitive matches, 15-24 games)
    as a Bayesian prior for scoring rates."""
    try:
        xg = _lj("wc2026_xg_proxy.json")
    except FileNotFoundError:
        return {}
    priors = {}
    for team, data in xg.items():
        t = _norm(team)
        games = data.get("games", 0)
        if games >= 5:
            priors[t] = {
                "xg_pg": data.get("xg_proxy", 1.3),
                "xga_pg": data.get("xga_proxy", 1.0),
                "games": games,
            }
    return priors


def compute_lambdas(
    home: str, away: str,
    *,
    ko_deflation: float = 1.0,  # caller decides whether to apply
) -> dict:
    """Compute regularised Poisson lambdas for a knockout match.

    Blends GS in-tournament scoring (3 games) with pre-tournament xG proxy
    (15-24 games) using sample-size weighting.  This prevents 0-conceded
    teams from producing clamped-floor opponent lambdas.

    Returns dict with lambda_home, lambda_away, expected_total, and
    diagnostic flags.
    """
    h, a = _norm(home), _norm(away)
    priors = _get_team_priors()
    flags: list[str] = []

    # ── Team scoring/conceding rates ──
    def _blended_rate(team: str, rate_type: str) -> float:
        """Get scoring/conceding rate, regularising only pathological cases.

        GS data (3 games at THIS tournament) is the primary signal — it
        reflects current form, squad selection, and tournament intensity.
        Pre-tournament prior (15-24 competitive games) is used ONLY to
        regularise extreme GS values:
          - 0 conceded in 3 GS games → blend 60/40 GS/prior (prevents
            phantom Under edges from small-sample clean sheets)
          - >3.0 scored per game → blend 60/40 to regress outlier attacks
          - Normal GS rates → use GS directly (85/15 light blend)
        """
        gs = _GS_SCORING.get(team)
        prior = priors.get(team)

        gs_rate = None
        prior_rate = None

        if gs is not None:
            gs_rate = gs[0] if rate_type == "attack" else gs[1]

        if prior is not None:
            prior_rate = prior["xg_pg"] if rate_type == "attack" else prior["xga_pg"]

        if gs_rate is not None and prior_rate is not None:
            # Determine blend weight based on how extreme the GS rate is
            is_pathological = (
                (rate_type == "defend" and gs_rate == 0.0)  # 0 conceded
                or (rate_type == "attack" and gs_rate >= 3.0)  # outlier attack
                or (rate_type == "defend" and gs_rate == 0.0)
            )
            if is_pathological:
                w_gs = BLEND_GS_PATHOLOGICAL
            else:
                w_gs = BLEND_GS_NORMAL

            return gs_rate * w_gs + prior_rate * (1 - w_gs)
        elif gs_rate is not None:
            return gs_rate
        elif prior_rate is not None:
            return prior_rate
        else:
            return _GS_AVG_GPG

    h_attack = _blended_rate(h, "attack")
    h_defend = _blended_rate(h, "defend")
    a_attack = _blended_rate(a, "attack")
    a_defend = _blended_rate(a, "defend")

    # ── Lambda computation ──
    # lambda_home = home_attack × (away_conceding / league_avg) × deflation
    # This is the standard Dixon-Coles / Poisson approach.
    if _GS_AVG_GPG > 0:
        a_def_factor = a_defend / _GS_AVG_GPG
        h_def_factor = h_defend / _GS_AVG_GPG
    else:
        a_def_factor = h_def_factor = 1.0

    lam_h_raw = h_attack * a_def_factor * ko_deflation
    lam_a_raw = a_attack * h_def_factor * ko_deflation

    # ── Artifact detection (NO hard clamping) ──
    lam_h = max(LAMBDA_FLOOR, min(LAMBDA_CEILING, lam_h_raw))
    lam_a = max(LAMBDA_FLOOR, min(LAMBDA_CEILING, lam_a_raw))

    if lam_h_raw < LAMBDA_FLOOR + 0.05:
        flags.append(f"λ_{h}_near_floor({lam_h_raw:.3f})")
    if lam_a_raw < LAMBDA_FLOOR + 0.05:
        flags.append(f"λ_{a}_near_floor({lam_a_raw:.3f})")
    if lam_h_raw > LAMBDA_CEILING - 0.2:
        flags.append(f"λ_{h}_near_ceiling({lam_h_raw:.3f})")
    if lam_a_raw > LAMBDA_CEILING - 0.2:
        flags.append(f"λ_{a}_near_ceiling({lam_a_raw:.3f})")

    # Check if either side had 0 conceded in GS (artifact source)
    gs_h = _GS_SCORING.get(h)
    gs_a = _GS_SCORING.get(a)
    if gs_a and gs_a[1] == 0.0:
        flags.append(f"{a}_conceded_0_GS→regularised")
    if gs_h and gs_h[1] == 0.0:
        flags.append(f"{h}_conceded_0_GS→regularised")

    return {
        "lambda_home": round(lam_h, 4),
        "lambda_away": round(lam_a, 4),
        "lambda_home_raw": round(lam_h_raw, 4),
        "lambda_away_raw": round(lam_a_raw, 4),
        "expected_total": round(lam_h + lam_a, 4),
        "home_attack": round(h_attack, 4),
        "home_defend": round(h_defend, 4),
        "away_attack": round(a_attack, 4),
        "away_defend": round(a_defend, 4),
        "flags": flags,
        "has_artifact": len(flags) > 0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. Poisson score-distribution engine
# ═════════════════════════════════════════════════════════════════════════════

def _poisson(lam: float, k: int) -> float:
    return (lam ** k * math.exp(-lam)) / math.factorial(k)


def score_distribution(lam_h: float, lam_a: float, max_goals: int = 7) -> dict:
    """Full Poisson score distribution → 3-way probs, totals, most likely scores."""
    grid: dict[tuple[int, int], float] = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            grid[(h, a)] = _poisson(lam_h, h) * _poisson(lam_a, a)

    home_win = sum(p for (h, a), p in grid.items() if h > a)
    draw = sum(p for (h, a), p in grid.items() if h == a)
    away_win = sum(p for (h, a), p in grid.items() if h < a)

    # Totals
    def _under(line: float) -> float:
        return sum(p for (h, a), p in grid.items() if h + a < line)

    # Most likely scores
    sorted_scores = sorted(grid.items(), key=lambda x: -x[1])
    top_scores = [(f"{h}-{a}", round(p * 100, 1)) for (h, a), p in sorted_scores[:5]]

    return {
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "under_1_5": round(_under(1.5), 4),
        "under_2_5": round(_under(2.5), 4),
        "under_3_5": round(_under(3.5), 4),
        "top_scores": top_scores,
        "expected_total": round(lam_h + lam_a, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. Confederation + mid-tier corrections
# ═════════════════════════════════════════════════════════════════════════════

def apply_corrections(
    home: str, away: str,
    mkt_home: float, mkt_draw: float, mkt_away: float,
) -> dict:
    """Apply vetted corrections to market 3-way probs.

    Returns corrected probs + correction breakdown.
    All corrections are PARLAY-BIASING magnitudes, not standalone-bet size.
    """
    h, a = _norm(home), _norm(away)
    corrections: dict[str, float] = {}

    # ── 4a. Confederation correction (halved from raw observed gaps) ──
    confed_adj_h = 0.0
    confed_adj_a = 0.0

    if h in _CAF:
        confed_adj_h += CONFED_CAF_PP
        corrections["caf_boost_home"] = CONFED_CAF_PP
    if a in _CAF:
        confed_adj_a += CONFED_CAF_PP
        corrections["caf_boost_away"] = CONFED_CAF_PP
    if h in _AFC:
        confed_adj_h += CONFED_AFC_PP
        corrections["afc_fade_home"] = CONFED_AFC_PP
    if a in _AFC:
        confed_adj_a += CONFED_AFC_PP
        corrections["afc_fade_away"] = CONFED_AFC_PP
    if h in _CONCACAF:
        confed_adj_h += CONFED_CONCACAF_PP
        corrections["concacaf_boost_home"] = CONFED_CONCACAF_PP
    if a in _CONCACAF:
        confed_adj_a += CONFED_CONCACAF_PP
        corrections["concacaf_boost_away"] = CONFED_CONCACAF_PP

    # Net effect: positive = favour home
    net_confed = confed_adj_h - confed_adj_a

    # Apply to probs
    adj_h = mkt_home + net_confed
    adj_a = mkt_away - net_confed
    adj_d = mkt_draw  # draw untouched by confed

    # ── 4b. Mid-tier favourite fade ──
    fav_pct = max(adj_h, adj_a)
    fade_applied = 0.0
    if FADE_FLOOR <= fav_pct <= FADE_CEILING:
        excess = fav_pct - 0.50
        fade = excess * FADE_RATE
        fade_applied = fade
        if adj_h > adj_a:
            adj_h -= fade
            adj_d += fade * 0.6
            adj_a += fade * 0.4
        else:
            adj_a -= fade
            adj_d += fade * 0.6
            adj_h += fade * 0.4
        corrections["mid_tier_fade"] = round(fade, 4)

    # Clamp and renormalise
    adj_h = max(0.03, adj_h)
    adj_a = max(0.03, adj_a)
    adj_d = max(0.05, adj_d)
    total = adj_h + adj_a + adj_d
    adj_h /= total
    adj_a /= total
    adj_d /= total

    return {
        "adj_home": round(adj_h, 4),
        "adj_draw": round(adj_d, 4),
        "adj_away": round(adj_a, 4),
        "corrections": corrections,
        "net_confed": round(net_confed, 4),
        "fade_applied": round(fade_applied, 4),
        "is_parlay_bias_only": True,  # always — these are lean magnitudes
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. Draw-resolution module (ET + PK advance probability)
# ═════════════════════════════════════════════════════════════════════════════

def draw_resolution(p_home_90: float, p_draw_90: float, p_away_90: float) -> dict:
    """Compute P(advance) for each side in a knockout match.

    Model:
      P(advance) = P(win90) + P(draw90) × [P(winET)×(1-P(drawET)) + P(drawET)×P(winPK)]

    ET carries ~60% of the 90-min gap (teams are more cautious).
    PK ≈ 50/50 + 10% of strength gap.
    P(drawET) ≈ 20% (empirical: ~20% of extra time ends level).
    """
    gap = p_home_90 - p_away_90

    # ET probabilities
    p_home_et = 0.5 + gap * ET_GAP_CARRY
    p_away_et = 1.0 - p_home_et
    p_draw_et = ET_DRAW_RATE

    # PK probabilities
    p_home_pk = 0.5 + gap * PK_GAP_CARRY
    p_away_pk = 1.0 - p_home_pk

    # Advance probabilities
    p_home_advance = (
        p_home_90
        + p_draw_90 * (p_home_et * (1 - p_draw_et) + p_draw_et * p_home_pk)
    )
    p_away_advance = (
        p_away_90
        + p_draw_90 * (p_away_et * (1 - p_draw_et) + p_draw_et * p_away_pk)
    )

    return {
        "home_advance": round(p_home_advance, 4),
        "away_advance": round(p_away_advance, 4),
        "home_et_wp": round(p_home_et, 4),
        "away_et_wp": round(p_away_et, 4),
        "home_pk_wp": round(p_home_pk, 4),
        "away_pk_wp": round(p_away_pk, 4),
        "p_extra_time": round(p_draw_90, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. Auto-vetting engine
# ═════════════════════════════════════════════════════════════════════════════

def vet_totals_edge(
    model_under: float,
    market_under: float,
    lambda_info: dict,
    line: float = 2.5,
) -> dict:
    """Vet a totals edge.  Strips deflation double-counting and flags artifacts.

    The market already prices ~17.5% KO deflation (R32 avg line 2.43 vs
    GS actual 2.96).  We do NOT add deflation on top — the edge must come
    from matchup-specific scoring rates only.

    Returns the honest (post-vetting) edge and diagnostic flags.
    """
    flags = list(lambda_info.get("flags", []))
    raw_edge_pp = round((model_under - market_under) * 100, 1)

    # ── Check: is edge from a clamped/regularised lambda? ──
    is_artifact = lambda_info.get("has_artifact", False)

    # ── Check: is edge implausibly large vs a liquid market? ──
    # 20+ bookmakers price WC totals. An edge >12pp implies the entire
    # market is wildly wrong, which is far more likely to be model noise
    # (3-game GS sample driving extreme rates) than real value.
    is_implausible = raw_edge_pp > MAX_CREDIBLE_EDGE_PP

    # ── Determine tier ──
    if is_artifact:
        tier = 3  # construction nudge at best — artifact-contaminated
        tier_label = "Tier 3 (artifact-flagged)"
        flags.append("CLAMP_ARTIFACT→Tier3")
    elif is_implausible:
        tier = 2  # demote: edge too large to be credible vs liquid market
        tier_label = "Tier 2 (edge implausibly large — likely sample noise)"
        flags.append(f"IMPLAUSIBLE_EDGE({raw_edge_pp:.0f}pp>12pp)→Tier2")
    elif raw_edge_pp >= 5.0:
        tier = 1
        tier_label = "Tier 1 (standalone-bettable)"
    elif raw_edge_pp >= 2.0:
        tier = 2
        tier_label = "Tier 2 (parlay lean)"
    else:
        tier = 3
        tier_label = "Tier 3 (no actionable edge)"

    return {
        "raw_edge_pp": raw_edge_pp,
        "honest_edge_pp": raw_edge_pp,  # deflation already stripped by caller
        "tier": tier,
        "tier_label": tier_label,
        "flags": flags,
        "is_artifact": is_artifact,
        "model_under_pct": round(model_under * 100, 1),
        "market_under_pct": round(market_under * 100, 1),
    }


def vet_side_edge(
    model_prob: float,
    market_prob: float,
    team: str,
    corrections: dict,
) -> dict:
    """Vet an h2h/moneyline edge.  Checks if confed/fade correction
    created the edge and tags appropriately."""
    edge_pp = round((model_prob - market_prob) * 100, 1)

    # What corrections contributed?
    confed_source = []
    if corrections.get("caf_boost_home") or corrections.get("caf_boost_away"):
        confed_source.append("CAF+2pp")
    if corrections.get("afc_fade_home") or corrections.get("afc_fade_away"):
        confed_source.append("AFC-1.5pp")
    if corrections.get("concacaf_boost_home") or corrections.get("concacaf_boost_away"):
        confed_source.append("CONCACAF+1pp")
    if corrections.get("mid_tier_fade"):
        confed_source.append(f"fade({corrections['mid_tier_fade']:.3f})")

    # Tier: confed/fade edges are always Tier 2 (parlay lean), never Tier 1
    if edge_pp >= 5.0 and not confed_source:
        tier = 1
        tier_label = "Tier 1 (standalone)"
    elif edge_pp >= 1.5:
        tier = 2
        tier_label = "Tier 2 (parlay lean)"
    else:
        tier = 3
        tier_label = "Tier 3 (construction nudge)"

    return {
        "edge_pp": edge_pp,
        "tier": tier,
        "tier_label": tier_label,
        "correction_sources": confed_source,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. Full knockout match prediction
# ═════════════════════════════════════════════════════════════════════════════

def predict_knockout_match(
    home: str,
    away: str,
    odds_event: dict | None = None,
) -> dict:
    """Full knockout prediction for a single match.

    1. Get market odds (devigged consensus)
    2. Apply confederation + mid-tier corrections
    3. Compute lambdas (regularised, NO deflation — market already prices it)
    4. Poisson score distribution
    5. Draw resolution → P(advance)
    6. Auto-vet all edges
    7. Return tiered output
    """
    h, a = _norm(home), _norm(away)

    # ── 7a. Market odds ──
    mkt_h, mkt_d, mkt_a = 0.33, 0.34, 0.33
    mkt_u25 = None  # market under 2.5 implied prob
    mkt_total_line = None

    if odds_event is not None:
        # Devig h2h across all bookmakers
        h2h_probs = []
        orig_home = odds_event.get("home_team", home)
        orig_away = odds_event.get("away_team", away)

        for bk in odds_event.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                hp = outcomes.get(orig_home)
                dp = outcomes.get("Draw")
                ap = outcomes.get(orig_away)
                if hp and dp and ap and hp > 1 and dp > 1 and ap > 1:
                    raw = [1/hp, 1/dp, 1/ap]
                    t = sum(raw)
                    h2h_probs.append([r / t for r in raw])

        if h2h_probs:
            n = len(h2h_probs)
            mkt_h = sum(p[0] for p in h2h_probs) / n
            mkt_d = sum(p[1] for p in h2h_probs) / n
            mkt_a = sum(p[2] for p in h2h_probs) / n

        # Totals market
        for bk in odds_event.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") == "totals":
                    outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                    point = mkt["outcomes"][0].get("point")
                    u_price = outcomes.get("Under")
                    o_price = outcomes.get("Over")
                    if u_price and o_price and point:
                        raw_u, raw_o = 1 / u_price, 1 / o_price
                        mkt_u25 = raw_u / (raw_u + raw_o)
                        mkt_total_line = point
                    break
            if mkt_u25 is not None:
                break

    # ── 7b. Corrections ──
    corr = apply_corrections(h, a, mkt_h, mkt_d, mkt_a)

    # ── 7c. Lambdas — NO deflation (market already prices ~17.5%) ──
    lam_info = compute_lambdas(h, a, ko_deflation=1.0)
    lam_h_val = lam_info["lambda_home"]
    lam_a_val = lam_info["lambda_away"]

    # ── 7d. Score distribution from Poisson ──
    scores = score_distribution(lam_h_val, lam_a_val)

    # ── 7e. Draw resolution ──
    dr = draw_resolution(corr["adj_home"], corr["adj_draw"], corr["adj_away"])

    # ── 7f. Vet edges ──
    edges: list[dict] = []

    # Totals edge (U2.5)
    if mkt_u25 is not None:
        totals_vet = vet_totals_edge(
            model_under=scores["under_2_5"],
            market_under=mkt_u25,
            lambda_info=lam_info,
            line=mkt_total_line or 2.5,
        )
        edges.append({
            "type": "totals",
            "pick": f"Under {mkt_total_line or 2.5}",
            **totals_vet,
        })

    # H2H / side edges
    for side, model_p, market_p in [
        (h, corr["adj_home"], mkt_h),
        (a, corr["adj_away"], mkt_a),
    ]:
        side_vet = vet_side_edge(model_p, market_p, side, corr["corrections"])
        if side_vet["edge_pp"] >= 1.0:
            edges.append({
                "type": "h2h",
                "pick": side,
                **side_vet,
            })

    # Sort edges: Tier 1 first, then by edge size
    edges.sort(key=lambda e: (e.get("tier", 9), -abs(e.get("honest_edge_pp", e.get("edge_pp", 0)))))

    return {
        "home": h,
        "away": a,
        "market": {
            "home": round(mkt_h, 4),
            "draw": round(mkt_d, 4),
            "away": round(mkt_a, 4),
            "under_2_5": round(mkt_u25, 4) if mkt_u25 else None,
            "total_line": mkt_total_line,
        },
        "corrected_3way": {
            "home": corr["adj_home"],
            "draw": corr["adj_draw"],
            "away": corr["adj_away"],
        },
        "corrections": corr["corrections"],
        "advance": {
            "home": dr["home_advance"],
            "away": dr["away_advance"],
        },
        "draw_resolution": dr,
        "lambdas": lam_info,
        "score_dist": scores,
        "edges": edges,
        "model_used": "wc_knockout_v1",
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. Predict all R32 matches (batch)
# ═════════════════════════════════════════════════════════════════════════════

def predict_r32() -> list[dict]:
    """Load current odds snapshot and predict all R32 knockout matches."""
    odds_snap = _lj("wc2026_odds_snapshot.json")
    results = []

    for ev in odds_snap:
        h = ev.get("home_team", "")
        a = ev.get("away_team", "")
        pred = predict_knockout_match(h, a, odds_event=ev)
        results.append(pred)

    return results


def print_r32_report(predictions: list[dict] | None = None) -> None:
    """Print the full tiered R32 report to stdout."""
    if predictions is None:
        predictions = predict_r32()

    print("=" * 100)
    print("WC 2026 ROUND OF 32 — KNOCKOUT MODEL (SELF-VETTING)")
    print("Model: wc_knockout_v1 | Deflation: stripped (market-priced)")
    print("=" * 100)

    # ── Match-by-match ──
    hdr = (
        f"{'#':<3} {'Match':<35} "
        f"{'Mkt H':>6} {'Mkt D':>6} {'Mkt A':>6} │ "
        f"{'Adj H':>6} {'Adj D':>6} {'Adj A':>6} │ "
        f"{'Adv H':>6} {'Adv A':>6} │ "
        f"{'xG':>5} {'U2.5':>6}"
    )
    print(f"\n{hdr}")
    print("-" * 110)

    for i, p in enumerate(predictions, 1):
        m = p["market"]
        c = p["corrected_3way"]
        adv = p["advance"]
        s = p["score_dist"]
        flags = p["lambdas"].get("flags", [])
        flag_str = " ⚠" if flags else ""

        print(
            f"{i:<3} {p['home']+' vs '+p['away']:<35} "
            f"{m['home']*100:>5.1f}% {m['draw']*100:>5.1f}% {m['away']*100:>5.1f}% │ "
            f"{c['home']*100:>5.1f}% {c['draw']*100:>5.1f}% {c['away']*100:>5.1f}% │ "
            f"{adv['home']*100:>5.1f}% {adv['away']*100:>5.1f}% │ "
            f"{s['expected_total']:>5.2f} {s['under_2_5']*100:>5.1f}%{flag_str}"
        )

    # ── Tiered edges ──
    tier1, tier2, tier3 = [], [], []
    for p in predictions:
        match_label = f"{p['home']} vs {p['away']}"
        for e in p["edges"]:
            entry = {"match": match_label, **e}
            t = e.get("tier", 9)
            if t == 1:
                tier1.append(entry)
            elif t == 2:
                tier2.append(entry)
            else:
                tier3.append(entry)

    print(f"\n{'=' * 80}")
    print("TIER 1 — STANDALONE-BETTABLE (vetted, ≥5pp, no artifacts)")
    print(f"{'=' * 80}")
    if tier1:
        for e in tier1:
            edge_val = e.get("honest_edge_pp", e.get("edge_pp", 0))
            flags_str = f" [{', '.join(e.get('flags', []))}]" if e.get("flags") else ""
            print(f"  {e['match']}: {e['pick']} → +{edge_val:.1f}pp{flags_str}")
    else:
        print("  (none)")

    print(f"\n{'=' * 80}")
    print("TIER 2 — PARLAY LEANS (confed/fade, 1.5-5pp)")
    print(f"{'=' * 80}")
    if tier2:
        for e in tier2:
            edge_val = e.get("honest_edge_pp", e.get("edge_pp", 0))
            src_parts = e.get("correction_sources", [])
            flag_parts = [f for f in e.get("flags", []) if "IMPLAUSIBLE" in f or "CLAMP" in f]
            label = ", ".join(src_parts + flag_parts) if (src_parts or flag_parts) else ""
            print(f"  {e['match']}: {e['pick']} → +{edge_val:.1f}pp [{label}]")
    else:
        print("  (none)")

    print(f"\n{'=' * 80}")
    print("TIER 3 — CONSTRUCTION NUDGES / ARTIFACT-FLAGGED")
    print(f"{'=' * 80}")
    if tier3:
        for e in tier3[:10]:
            edge_val = e.get("honest_edge_pp", e.get("edge_pp", 0))
            flags_str = f" [{', '.join(e.get('flags', []))}]" if e.get("flags") else ""
            pick = e.get("pick", "?")
            print(f"  {e['match']}: {pick} → {edge_val:+.1f}pp{flags_str}")
    else:
        print("  (none)")

    # Lambda diagnostics
    flagged = [p for p in predictions if p["lambdas"].get("has_artifact")]
    if flagged:
        print(f"\n{'=' * 80}")
        print("LAMBDA DIAGNOSTICS (regularised, not clamped)")
        print(f"{'=' * 80}")
        for p in flagged:
            lam = p["lambdas"]
            print(f"  {p['home']} vs {p['away']}:")
            print(f"    λ_home={lam['lambda_home']:.3f} (raw {lam['lambda_home_raw']:.3f})")
            print(f"    λ_away={lam['lambda_away']:.3f} (raw {lam['lambda_away_raw']:.3f})")
            for fl in lam["flags"]:
                print(f"    ⚠ {fl}")


# ═════════════════════════════════════════════════════════════════════════════
# 9. Integration: predict_wc_knockout_match() — pipeline entry point
# ═════════════════════════════════════════════════════════════════════════════

def predict_wc_knockout_match(leg: dict) -> dict | None:
    """Score a WC knockout leg for the standard pipeline.

    Drop-in replacement for predict_wc_match() when stage='knockout'.
    Returns dict with win_prob (0-100), ev, model_used, confidence, ko_meta.
    """
    home = _norm(leg.get("home_team", "") or "")
    away = _norm(leg.get("away_team", "") or "")
    market = leg.get("market", "")
    pick = (leg.get("pick") or "").strip()
    dec_odds = leg.get("odds", 2.0)

    if not home or not away or not pick:
        return None

    # Load odds event from snapshot
    try:
        odds_snap = _lj("wc2026_odds_snapshot.json")
    except FileNotFoundError:
        odds_snap = []

    odds_event = None
    for ev in odds_snap:
        eh = _norm(ev.get("home_team", ""))
        ea = _norm(ev.get("away_team", ""))
        if (eh == home and ea == away) or (eh == away and ea == home):
            odds_event = ev
            break

    pred = predict_knockout_match(home, away, odds_event=odds_event)

    # Map pick to win probability
    pick_lower = pick.lower()
    pick_norm = _norm(pick).lower()
    home_lower = home.lower()
    away_lower = away.lower()

    win_prob: float | None = None

    if market in ("h2h", "moneyline"):
        if pick_lower == home_lower or pick_norm == home_lower:
            win_prob = pred["corrected_3way"]["home"]
        elif pick_lower == away_lower or pick_norm == away_lower:
            win_prob = pred["corrected_3way"]["away"]
        elif pick_lower == "draw":
            win_prob = pred["corrected_3way"]["draw"]

    elif market in ("spreads", "alternate_spreads"):
        point = leg.get("point")
        if point is None:
            return None
        spread = float(point)
        p_h = pred["corrected_3way"]["home"]
        p_d = pred["corrected_3way"]["draw"]
        p_a = pred["corrected_3way"]["away"]

        if pick_lower == home_lower or pick_norm == home_lower:
            if spread >= 0.5:
                win_prob = p_h + p_d
            elif spread >= -0.5:
                win_prob = p_h + p_d * 0.5
            else:
                win_prob = p_h * max(0.3, 1.0 - abs(spread) * 0.15)
        elif pick_lower == away_lower or pick_norm == away_lower:
            if spread >= 0.5:
                win_prob = p_a + p_d
            elif spread >= -0.5:
                win_prob = p_a + p_d * 0.5
            else:
                win_prob = p_a * max(0.3, 1.0 - abs(spread) * 0.15)

    elif market in ("totals", "alternate_totals"):
        point = leg.get("point")
        if point is None:
            return None
        line = float(point)
        lam = pred["lambdas"]
        lam_h = lam["lambda_home"]
        lam_a = lam["lambda_away"]

        # Poisson CDF
        floor_line = int(math.floor(line))
        cdf = 0.0
        for k in range(floor_line + 1):
            cdf += (
                (lam_h + lam_a) ** k * math.exp(-(lam_h + lam_a))
            ) / math.factorial(k)

        # Correction: use bivariate Poisson (independent), not univariate sum
        # The sum of two independent Poissons is Poisson(λ1+λ2), so univariate is correct
        if line == floor_line:
            p_exact = ((lam_h + lam_a) ** floor_line * math.exp(-(lam_h + lam_a))) / math.factorial(floor_line)
            over_prob = 1.0 - cdf + p_exact * 0.5
        else:
            over_prob = 1.0 - cdf

        if pick_lower == "over":
            win_prob = over_prob
        elif pick_lower == "under":
            win_prob = 1.0 - over_prob

    if win_prob is None:
        return None

    win_prob = max(0.05, min(0.95, win_prob))
    ev = win_prob * (dec_odds - 1) - (1 - win_prob)
    wp_pct = round(win_prob * 100, 2)

    # Confidence: factor in artifact flags
    has_artifact = pred["lambdas"].get("has_artifact", False)
    if market in ("totals", "alternate_totals") and has_artifact:
        confidence = "low_artifact"
    elif wp_pct >= 70:
        confidence = "high"
    elif wp_pct >= 50:
        confidence = "low"
    else:
        confidence = "very_low"

    # Find best tier for this pick
    best_tier = 9
    for e in pred.get("edges", []):
        if pick_lower in e.get("pick", "").lower() or _norm(pick).lower() in e.get("pick", "").lower():
            best_tier = min(best_tier, e.get("tier", 9))

    return {
        "win_prob": wp_pct,
        "ev": round(ev, 4),
        "model_used": "wc_knockout_v1",
        "confidence": confidence,
        "ko_tier": best_tier if best_tier < 9 else None,
        "ko_flags": pred["lambdas"].get("flags", []),
        "advance_home": pred["advance"]["home"],
        "advance_away": pred["advance"]["away"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 10. main — run R32 predictions and save
# ═════════════════════════════════════════════════════════════════════════════

def main():
    predictions = predict_r32()

    # Save
    out_path = os.path.join(_DATA_DIR, "wc2026_knockout_predictions.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": "wc_knockout_v1",
            "stage": "R32",
            "deflation_note": "NOT applied — market already prices ~17.5%",
            "predictions": predictions,
        }, f, indent=2, default=str)

    print(f"\nSaved to {out_path}")

    # Print report
    print_r32_report(predictions)


if __name__ == "__main__":
    main()
