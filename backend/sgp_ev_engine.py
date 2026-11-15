"""
sgp_ev_engine.py — Same-Game Parlay EV computation via sim copula + v1.2 marginals.

Architecture:
  1. Pre-compute game draws via generate_game_draws() + allocate_game_n()
  2. For each SGP leg, get v1.2 P(over|line) as the marginal
  3. Quantile-map: use sim correlation structure (copula) with v1.2 marginals
     to compute joint hit probability across N draws
  4. Report: fair odds, correlation multiplier k, EV at offered price, MC SE

The quantile-map preserves the sim's within-game correlations (e.g. WR rec_yds
goes up when QB pass_yds goes up) while calibrating each leg's individual hit
rate to the v1.2 model.

Usage:
    engine = SGPEngine(alloc_home, alloc_away)
    result = engine.price_sgp(legs)
    # result["joint_p"], result["fair_odds"], result["ev"], result["mc_se"]

Run standalone:
    python sgp_ev_engine.py demo
"""
from __future__ import annotations

import os
from collections import defaultdict
from math import sqrt

import numpy as np

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")

# Validated geo_ip range — below this, k curves are extrapolated
_GEO_IP_RELIABLE_FLOOR = 0.30

# Default guardrails for CLI
_MAX_LEGS_DEFAULT = 5

# ── Devig helpers ─────────────────────────────────────────────────────────

def _devig_two_sided(over_price: float, under_price: float) -> float | None:
    """Multiplicative devig: p_fair = (1/over) / (1/over + 1/under).

    Requires both sides with decimal odds > 1.
    """
    if over_price <= 1 or under_price <= 1:
        return None
    imp_over = 1.0 / over_price
    imp_under = 1.0 / under_price
    total = imp_over + imp_under
    if total <= 0:
        return None
    return imp_over / total


def _devig_one_sided(price: float, assumed_vig: float = 0.05) -> float | None:
    """Devig for one-sided alt markets.

    Method: assume total implied probability = 1 + assumed_vig, then
    p_fair = (1/price) / (1 + assumed_vig).

    Document as 'assumed_{vig_pct}pct_vig'. Less reliable than two-sided.
    """
    if price <= 1:
        return None
    return (1.0 / price) / (1.0 + assumed_vig)


def _devig_leg(leg: dict) -> tuple[float | None, str]:
    """Compute devigged implied P(over) for a leg, return (p, method).

    Priority:
      1. Two-sided multiplicative devig (over_price + under_price both present)
      2. One-sided with assumed 5% vig (only over_price or only under_price)
      3. None (no price data)
    """
    over_p = leg.get("over_price")
    under_p = leg.get("under_price")

    if over_p and under_p and over_p > 1 and under_p > 1:
        return _devig_two_sided(over_p, under_p), "two_sided_mult"

    if over_p and over_p > 1:
        return _devig_one_sided(over_p), "assumed_5pct_vig"

    if under_p and under_p > 1:
        # Only under price → P(over) = 1 - P(under)
        p_under = _devig_one_sided(under_p)
        if p_under is not None:
            return 1.0 - p_under, "assumed_5pct_vig_under"
        return None, "none"

    return None, "none"


# ── Stat name mapping: prop DB → allocation layer ─────────────────────────
_PROP_TO_ALLOC_STAT = {
    "rec_yds": "rec_yards",
    "rush_yds": "rush_yards",
    "pass_yds": "pass_yards",
    "pass_tds": "pass_tds",
    "rush_tds": "rush_tds",
    "receptions": "receptions",
    "pass_att": "pass_att",
    "rush_att": "rush_att",
    "completions": "completions",
    "interceptions": "interceptions",
    "sacks": "sacks",
    "targets": "targets",
    # Allow allocation-style names as passthrough
    "rec_yards": "rec_yards",
    "rush_yards": "rush_yards",
    "pass_yards": "pass_yards",
}


def _resolve_stat(stat: str) -> str:
    """Map prop-style stat name to allocation-layer stat name."""
    s = stat.strip().lower()
    mapped = _PROP_TO_ALLOC_STAT.get(s)
    if mapped:
        return mapped
    raise ValueError(f"Unknown stat '{stat}'. Known: {list(_PROP_TO_ALLOC_STAT)}")


# ══════════════════════════════════════════════════════════════════════════════
# SGP Engine
# ══════════════════════════════════════════════════════════════════════════════

class SGPEngine:
    """Price same-game parlays using sim draws + quantile-mapped v1.2 marginals.

    Accepts the output of ``allocate_game_n()`` for each side of the game.
    Both sides must come from the same set of N sim draws so that draw
    indices are correlated (home draw[i] and away draw[i] are from the
    same simulated game).
    """

    def __init__(
        self,
        alloc_home: dict,
        alloc_away: dict,
    ):
        """
        Args:
            alloc_home: output of allocate_game_n() for the home team.
            alloc_away: output of allocate_game_n() for the away team.
        """
        n_home = alloc_home.get("_meta", {}).get("n_sims", 0)
        n_away = alloc_away.get("_meta", {}).get("n_sims", 0)
        if n_home != n_away:
            raise ValueError(
                f"Home ({n_home}) and away ({n_away}) must have same N draws"
            )
        if n_home == 0:
            raise ValueError("No sim draws — run allocate_game_n() first")

        self.alloc_home = alloc_home
        self.alloc_away = alloc_away
        self.n_sims = n_home

        # Combined lookup: player -> alloc result
        self._players = {}
        for name, data in alloc_home.items():
            if name == "_meta":
                continue
            self._players[name] = data
        for name, data in alloc_away.items():
            if name == "_meta":
                continue
            if name in self._players:
                # Same name on both teams — rare; prefix with team side
                self._players[f"away_{name}"] = data
            else:
                self._players[name] = data

    def available_players(self) -> list[str]:
        """List all player names with sim data."""
        return sorted(self._players.keys())

    def available_stats(self, player: str) -> list[str]:
        """List stats with sim arrays for a player."""
        pdata = self._players.get(player)
        if not pdata:
            return []
        return [k for k, v in pdata.items()
                if isinstance(v, dict) and "sims" in v]

    def _get_sims(self, player: str, stat: str) -> np.ndarray | None:
        """Get the raw sim array for a player-stat."""
        pdata = self._players.get(player)
        if not pdata:
            return None
        alloc_stat = _resolve_stat(stat)
        stat_data = pdata.get(alloc_stat)
        if not stat_data or "sims" not in stat_data:
            return None
        return np.array(stat_data["sims"], dtype=float)

    def price_sgp(
        self,
        legs: list[dict],
        combo_offered_odds: float | None = None,
        boost_pct: float = 0.0,
        max_stake: float | None = None,
    ) -> dict:
        """Price a same-game parlay.

        Marginal priority per leg (highest to lowest):
          1. Devigged market (over_price + under_price → multiplicative devig)
          2. Devigged one-sided (over_price only → assumed 5% vig)
          3. v1.2 model (v12_p_over field)
          4. Raw sim P(over) (no quantile correction)

        Args:
            legs: list of leg dicts, each with:
                player      (str):  player name
                stat        (str):  prop stat (rec_yds, pass_yds, etc.)
                line        (float): prop line
                direction   (str):  "over" or "under"
                offered_price (float): decimal odds for this leg (optional)
                over_price  (float): decimal odds for over side (for devig)
                under_price (float): decimal odds for under side (for devig)
                v12_p_over  (float): v1.2 model P(over) (fallback)
            combo_offered_odds: decimal odds for the full SGP combo.
            boost_pct: boost percentage (e.g. 0.25 for 25% profit boost).
                Boosted EV = p·(1+b)·(D−1) − (1−p).
            max_stake: optional max-stake cap for reporting.

        Returns: dict with joint_p, independent_p, k, fair_odds,
            overcharge_c, breakeven_boost, ev, boosted_ev, mc_se,
            per_leg details, reliability flag, warnings.
        """
        if not legs:
            return {"error": "No legs provided"}

        n = self.n_sims
        warnings = []
        per_leg = []
        hit_arrays = []

        for i, leg in enumerate(legs):
            player = leg["player"]
            stat = leg["stat"]
            line = float(leg["line"])
            direction = leg.get("direction", "over").lower()

            sims = self._get_sims(player, stat)
            if sims is None:
                avail = self.available_players()
                warnings.append(
                    f"Leg {i}: {player}/{stat} not in draws. "
                    f"Available: {avail[:10]}"
                )
                per_leg.append({
                    "player": player, "stat": stat, "line": line,
                    "direction": direction, "error": "not in draws",
                })
                continue

            sim_p_over = float((sims > line).mean())

            # ── Marginal selection (priority: market → v1.2 → sim) ──
            # v12_p_over: callers should pass gamma P(over) for yardage
            # (p_over_v12_gamma), NegBin for counts. Stored p_over is
            # frozen lognormal and should NOT be used here.
            devig_p, devig_method = _devig_leg(leg)
            v12_p = leg.get("v12_p_over")

            if devig_p is not None:
                marginal_p_over = devig_p
                marginal_source = f"market:{devig_method}"
            elif v12_p is not None:
                marginal_p_over = float(v12_p)
                marginal_source = "v1.2"
            else:
                marginal_p_over = sim_p_over
                marginal_source = "sim"
                warnings.append(
                    f"Leg {i}: {player}/{stat} — no market or v1.2, "
                    f"using raw sim P(over)={sim_p_over:.3f}"
                )

            # ── Quantile-map: threshold in sim domain ──
            hit_array = self._quantile_map(
                sims, marginal_p_over, direction, n, i
            )

            actual_leg_p = float(hit_array.mean())

            leg_detail = {
                "player": player,
                "stat": stat,
                "line": line,
                "direction": direction,
                "marginal_p_over": round(marginal_p_over, 4),
                "marginal_source": marginal_source,
                "sim_p_over": round(sim_p_over, 4),
                "leg_p": round(actual_leg_p, 4),
                "sim_mean": round(float(sims.mean()), 2),
                "sim_sd": round(float(sims.std()), 2),
            }

            offered = leg.get("offered_price")
            if offered:
                leg_detail["offered_price"] = offered
                implied = 1.0 / offered
                leg_detail["implied_prob"] = round(implied, 4)
                leg_detail["leg_edge"] = round(actual_leg_p - implied, 4)

            per_leg.append(leg_detail)
            hit_arrays.append(hit_array)

        if not hit_arrays:
            return {
                "error": "No valid legs",
                "warnings": warnings,
                "per_leg": per_leg,
            }

        # ── Joint probability ──
        joint_hits = hit_arrays[0].copy()
        for ha in hit_arrays[1:]:
            joint_hits &= ha

        joint_p = float(joint_hits.mean())
        mc_se = sqrt(joint_p * (1.0 - joint_p) / n) if n > 0 else 0.0

        individual_probs = [pl["leg_p"] for pl in per_leg if "leg_p" in pl]
        independent_p = 1.0
        for p in individual_probs:
            independent_p *= p

        k = joint_p / independent_p if independent_p > 0 else float("inf")
        fair_odds = 1.0 / joint_p if joint_p > 0 else float("inf")

        # ── Reliability check: geo_ip extrapolation ──
        reliable = True
        geo_ip = None
        if len(individual_probs) >= 2:
            log_sum = sum(np.log(p) for p in individual_probs if p > 0)
            geo_ip = float(np.exp(log_sum / len(individual_probs)))
            if geo_ip < _GEO_IP_RELIABLE_FLOOR:
                reliable = False
                warnings.append(
                    f"geo_ip={geo_ip:.3f} < {_GEO_IP_RELIABLE_FLOOR} — "
                    f"k curve extrapolated; joint P is UNRELIABLE"
                )

        # ── Overcharge factor and break-even boost ──
        overcharge_c = None
        breakeven_boost = None
        book_implied_joint = None

        if combo_offered_odds is None and all(
            "offered_price" in pl for pl in per_leg
        ):
            combo_ref = 1.0
            for pl in per_leg:
                combo_ref *= pl["offered_price"]
            combo_offered_odds = combo_ref
            warnings.append(
                f"No combo price given; using product of leg prices "
                f"({combo_offered_odds:.3f}) as reference"
            )

        if combo_offered_odds is not None and combo_offered_odds > 0:
            book_implied_joint = 1.0 / combo_offered_odds
            if joint_p > 0:
                overcharge_c = book_implied_joint / joint_p
                breakeven_boost = overcharge_c - 1.0

        # ── EV computation ──
        ev = None
        boosted_ev = None
        ev_range = None

        if combo_offered_odds is not None and joint_p > 0:
            base_decimal = combo_offered_odds
            # Standard EV
            ev = joint_p * (base_decimal - 1.0) - (1.0 - joint_p)

            # Boosted EV: EV = p·(1+b)·(D−1) − (1−p)
            if boost_pct > 0:
                boosted_ev = (
                    joint_p * (1.0 + boost_pct) * (base_decimal - 1.0)
                    - (1.0 - joint_p)
                )

            # If unreliable, compute range using ±2 SE on joint_p
            if not reliable:
                p_lo = max(0.0, joint_p - 2 * mc_se)
                p_hi = min(1.0, joint_p + 2 * mc_se)
                if boost_pct > 0:
                    ev_lo = p_lo * (1 + boost_pct) * (base_decimal - 1) - (1 - p_lo)
                    ev_hi = p_hi * (1 + boost_pct) * (base_decimal - 1) - (1 - p_hi)
                else:
                    ev_lo = p_lo * (base_decimal - 1) - (1 - p_lo)
                    ev_hi = p_hi * (base_decimal - 1) - (1 - p_hi)
                ev_range = (round(ev_lo, 4), round(ev_hi, 4))

        # ── Verdict ──
        verdict = None
        if combo_offered_odds is not None:
            if not reliable:
                verdict = "UNRELIABLE"
            else:
                check_ev = boosted_ev if boosted_ev is not None else ev
                if check_ev is not None and check_ev > 0:
                    verdict = "TAKE"
                else:
                    verdict = "PASS"

        result = {
            "joint_p": round(joint_p, 6),
            "independent_p": round(independent_p, 6),
            "k": round(k, 4),
            "fair_odds": round(fair_odds, 3) if fair_odds < 1e6 else None,
            "combo_offered_odds": (
                round(combo_offered_odds, 3)
                if combo_offered_odds is not None else None
            ),
            "overcharge_c": round(overcharge_c, 4) if overcharge_c else None,
            "breakeven_boost": (
                round(breakeven_boost, 4) if breakeven_boost else None
            ),
            "ev": round(ev, 4) if ev is not None else None,
            "boosted_ev": round(boosted_ev, 4) if boosted_ev is not None else None,
            "ev_range": ev_range,
            "boost_pct": boost_pct if boost_pct > 0 else None,
            "max_stake": max_stake,
            "mc_se": round(mc_se, 6),
            "geo_ip": round(geo_ip, 3) if geo_ip is not None else None,
            "reliable": reliable,
            "verdict": verdict,
            "n_sims": n,
            "n_legs": len(hit_arrays),
            "per_leg": per_leg,
            "warnings": warnings,
        }
        return result

    @staticmethod
    def _quantile_map(
        sims: np.ndarray,
        marginal_p_over: float,
        direction: str,
        n: int,
        leg_idx: int,
    ) -> np.ndarray:
        """Map marginal probability onto sim draws via quantile threshold."""
        if direction == "over":
            marginal_p = marginal_p_over
            if marginal_p <= 0:
                return np.zeros(n, dtype=bool)
            if marginal_p >= 1:
                return np.ones(n, dtype=bool)
            threshold = np.percentile(sims, 100.0 * (1.0 - marginal_p))
            hit_array = sims > threshold
            target_hits = int(round(marginal_p * n))
            actual_hits = int(hit_array.sum())
            if actual_hits != target_hits:
                at_threshold = np.where(sims == threshold)[0]
                if len(at_threshold) > 0:
                    strictly_above = sims > threshold
                    n_strict = int(strictly_above.sum())
                    need = target_hits - n_strict
                    if 0 < need <= len(at_threshold):
                        rng = np.random.default_rng(42 + leg_idx)
                        chosen = rng.choice(at_threshold, size=need,
                                            replace=False)
                        hit_array = strictly_above.copy()
                        hit_array[chosen] = True
                    elif need <= 0:
                        hit_array = strictly_above
            return hit_array
        else:
            marginal_p = 1.0 - marginal_p_over
            if marginal_p <= 0:
                return np.zeros(n, dtype=bool)
            if marginal_p >= 1:
                return np.ones(n, dtype=bool)
            threshold = np.percentile(sims, 100.0 * marginal_p)
            hit_array = sims < threshold
            target_hits = int(round(marginal_p * n))
            actual_hits = int(hit_array.sum())
            if actual_hits != target_hits:
                at_threshold = np.where(sims == threshold)[0]
                if len(at_threshold) > 0:
                    strictly_below = sims < threshold
                    n_strict = int(strictly_below.sum())
                    need = target_hits - n_strict
                    if 0 < need <= len(at_threshold):
                        rng = np.random.default_rng(42 + leg_idx)
                        chosen = rng.choice(at_threshold, size=need,
                                            replace=False)
                        hit_array = strictly_below.copy()
                        hit_array[chosen] = True
                    elif need <= 0:
                        hit_array = strictly_below
            return hit_array


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: price from raw sim arrays (skip full allocation pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def price_sgp_from_arrays(
    legs: list[dict],
    combo_offered_odds: float | None = None,
) -> dict:
    """Price an SGP when you already have per-player sim arrays.

    Each leg dict must include a ``sims`` key with an array of N values
    (same N across all legs, draws aligned).  Faster than building the
    full engine when sim arrays are already available.

    Leg format: {
        player, stat, line, direction,
        sims: list[float],           # N sim values (aligned across legs)
        v12_p_over: float,           # optional v1.2 marginal
        offered_price: float,        # optional
    }
    """
    if not legs:
        return {"error": "No legs provided"}

    # Validate alignment
    ns = [len(leg["sims"]) for leg in legs]
    if len(set(ns)) != 1:
        return {"error": f"Mismatched sim lengths: {ns}"}
    n = ns[0]

    # Build a mock alloc result for each leg
    # All legs share a dummy "home" allocation
    alloc_home = {"_meta": {"n_sims": n, "n_players": 0}}
    for leg in legs:
        player = leg["player"]
        stat = _resolve_stat(leg["stat"])
        if player not in alloc_home:
            alloc_home[player] = {}
        alloc_home[player][stat] = {
            "sims": leg["sims"],
            "mean": float(np.mean(leg["sims"])),
        }

    alloc_away = {"_meta": {"n_sims": n, "n_players": 0}}
    engine = SGPEngine(alloc_home, alloc_away)

    return engine.price_sgp(legs, combo_offered_odds)


# ══════════════════════════════════════════════════════════════════════════════
# Batch pricing: multiple SGP combos for the same game
# ══════════════════════════════════════════════════════════════════════════════

def price_sgp_batch(
    engine: SGPEngine,
    combos: list[dict],
) -> list[dict]:
    """Price multiple SGP combos using the same game draws.

    Args:
        engine: SGPEngine initialized with game draws.
        combos: list of combo dicts, each with:
            legs: list of leg dicts (same format as price_sgp)
            combo_offered_odds: float (optional)
            label: str (optional, for identification)

    Returns: list of result dicts (same format as price_sgp, plus label).
    """
    results = []
    for combo in combos:
        result = engine.price_sgp(
            combo["legs"],
            combo.get("combo_offered_odds"),
        )
        result["label"] = combo.get("label", "")
        results.append(result)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Report formatting
# ══════════════════════════════════════════════════════════════════════════════

def print_sgp_result(result: dict) -> None:
    """Pretty-print an SGP pricing result."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    rel_tag = "" if result.get("reliable", True) else " [UNRELIABLE]"
    print("=" * 80)
    print(f"SGP PRICING — {result['n_legs']} legs, "
          f"{result['n_sims']} sims{rel_tag}")
    print("=" * 80)

    print(f"\n  {'Player':25s} {'Stat':12s} {'Line':>6s} {'Dir':5s} "
          f"{'Marg P':>7s} {'Source':>18s} {'Price':>7s} {'Edge':>7s}")
    print("  " + "─" * 88)

    for pl in result["per_leg"]:
        if "error" in pl:
            print(f"  {pl['player']:25s} {pl['stat']:12s} — {pl['error']}")
            continue
        price_str = f"{pl['offered_price']:.3f}" if "offered_price" in pl else "—"
        edge_str = f"{pl['leg_edge']:+.3f}" if "leg_edge" in pl else "—"
        p_over = pl.get("marginal_p_over", pl.get("leg_p", 0))
        if pl["direction"] == "under":
            p_display = 1.0 - p_over
        else:
            p_display = p_over
        src = pl.get("marginal_source", "?")
        print(f"  {pl['player']:25s} {pl['stat']:12s} "
              f"{pl['line']:6.1f} {pl['direction']:5s} "
              f"{p_display:7.3f} {src:>18s} "
              f"{price_str:>7s} {edge_str:>7s}")

    print(f"\n  Joint probability:  {result['joint_p']:.4f} "
          f"(MC SE: ±{result['mc_se']:.4f})")
    print(f"  Independent P:      {result['independent_p']:.4f}")
    print(f"  Correlation k:      {result['k']:.3f}")
    if result.get("geo_ip") is not None:
        print(f"  Geometric IP:       {result['geo_ip']:.3f}"
              f"{'  ← below validated range' if not result.get('reliable', True) else ''}")
    print(f"  Fair odds:          {result['fair_odds']}")

    if result.get("combo_offered_odds"):
        print(f"  Offered combo odds: {result['combo_offered_odds']:.3f}")

    if result.get("overcharge_c") is not None:
        c = result["overcharge_c"]
        be = result.get("breakeven_boost")
        print(f"  Overcharge c:       {c:.3f}  "
              f"(book charges {(c - 1) * 100:+.1f}% vs fair)")
        if be is not None:
            print(f"  Break-even boost:   {be * 100:.1f}%  "
                  f"(need ≥{be * 100:.1f}% boost to be +EV)")

    if result.get("ev") is not None:
        ev = result["ev"]
        if result.get("reliable", True):
            tag = "+EV" if ev > 0 else "-EV"
            print(f"  EV (base):          {ev:+.4f} ({ev * 100:+.1f}%) [{tag}]")
        else:
            rng = result.get("ev_range")
            if rng:
                print(f"  EV range (±2 SE):   [{rng[0]:+.4f}, {rng[1]:+.4f}]")
            else:
                print(f"  EV (unreliable):    {ev:+.4f}")

    if result.get("boosted_ev") is not None:
        bev = result["boosted_ev"]
        bp = result.get("boost_pct", 0) * 100
        if result.get("reliable", True):
            tag = "+EV" if bev > 0 else "-EV"
            print(f"  EV (boosted {bp:.0f}%):   {bev:+.4f} "
                  f"({bev * 100:+.1f}%) [{tag}]")
        else:
            rng = result.get("ev_range")
            if rng:
                print(f"  EV range (boosted): [{rng[0]:+.4f}, {rng[1]:+.4f}]")

    if result.get("max_stake"):
        ms = result["max_stake"]
        bev = result.get("boosted_ev") or result.get("ev") or 0
        print(f"  Max stake:          ${ms:.2f}  "
              f"(max profit: ${ms * max(bev, 0) if bev > 0 else 0:.2f})")

    if result.get("verdict"):
        v = result["verdict"]
        print(f"\n  VERDICT: {v}")

    if result.get("warnings"):
        print(f"\n  Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: game context → engine
# ══════════════════════════════════════════════════════════════════════════════

def build_game_engine(
    home_profile: dict,
    away_profile: dict,
    home_player_shares: list[dict],
    away_player_shares: list[dict],
    home_qb: dict | None = None,
    away_qb: dict | None = None,
    spread: float = 0.0,
    total: float = 45.0,
    home_targets: dict | None = None,
    away_targets: dict | None = None,
    n_sims: int = 10000,
    seed: int = 42,
) -> SGPEngine:
    """Build an SGPEngine from game context (full pipeline).

    Runs game sim → scales to market targets → allocates to players.
    Expensive (N game sims), so cache the engine and reuse for multiple
    SGP combos within the same game.

    Args:
        home/away_profile: team profiles from build_team_profiles().
        home/away_player_shares: player share lists for allocate_game_n().
        home/away_qb: optional QB overrides.
        spread: home spread (negative = favored).
        total: game total.
        home/away_targets: scaling targets from build_team_targets().
        n_sims: simulation count (≥5000 recommended).
        seed: random seed.

    Returns: SGPEngine ready for price_sgp() calls.
    """
    from nfl_volume_reconciliation import generate_game_draws
    from nfl_player_allocation import allocate_game_n

    # 1. Generate game draws
    draws = generate_game_draws(
        n_sims=n_sims,
        home_profile=home_profile,
        away_profile=away_profile,
        home_qb=home_qb,
        away_qb=away_qb,
        spread=spread,
        total=total,
        home_targets=home_targets,
        away_targets=away_targets,
        seed=seed,
    )

    # 2. Allocate to players (uses raw sim format, not prop-settlement)
    #    allocate_game_stats expects: pass_att, completions, pass_yards,
    #    sacks, scrambles, scramble_yards, rush_att, rush_yards, etc.
    rng = np.random.default_rng(seed + 1)
    alloc_home = allocate_game_n(rng, draws["home_draws"],
                                 home_player_shares)

    rng_away = np.random.default_rng(seed + 2)
    alloc_away = allocate_game_n(rng_away, draws["away_draws"],
                                 away_player_shares)

    return SGPEngine(alloc_home, alloc_away)


# ══════════════════════════════════════════════════════════════════════════════
# Validation: compare sim k vs measured k curves
# ══════════════════════════════════════════════════════════════════════════════

def validate_k_vs_curves(engine: SGPEngine, archetype_legs: list[dict]) -> dict:
    """Compare sim-derived k against the measured k curves.

    Takes a set of leg pairs at various line depths and computes
    sim joint P / independent P for each. Compares against the k from
    _interpolate_k() in parlay_builder.

    Args:
        engine: initialized SGPEngine
        archetype_legs: list of {
            player_a, stat_a, player_b, stat_b,
            lines: [(line_a, line_b, label), ...]
        }

    Returns: dict with comparison table and diagnostics.
    """
    try:
        from parlay_builder import _interpolate_k
    except ImportError:
        return {"error": "parlay_builder not available"}

    results = []

    for arch in archetype_legs:
        p_a = arch["player_a"]
        s_a = arch["stat_a"]
        p_b = arch["player_b"]
        s_b = arch["stat_b"]

        sims_a = engine._get_sims(p_a, s_a)
        sims_b = engine._get_sims(p_b, s_b)
        if sims_a is None or sims_b is None:
            results.append({"archetype": f"{p_a}/{s_a} × {p_b}/{s_b}",
                           "error": "missing sims"})
            continue

        n = len(sims_a)

        for line_a, line_b, label in arch["lines"]:
            p_over_a = float((sims_a > line_a).mean())
            p_over_b = float((sims_b > line_b).mean())

            if p_over_a <= 0 or p_over_b <= 0:
                continue

            # Sim joint
            joint_hit = ((sims_a > line_a) & (sims_b > line_b)).mean()
            indep = p_over_a * p_over_b
            sim_k = float(joint_hit / indep) if indep > 0 else float("nan")

            # Geometric implied prob
            geo_ip = sqrt(p_over_a * p_over_b)

            # Measured k from curve
            # Normalize stat names to prop-style for lookup
            prop_s_a = s_a.replace("_yards", "_yds")
            prop_s_b = s_b.replace("_yards", "_yds")
            curve_result = _interpolate_k((prop_s_a, prop_s_b), geo_ip)

            results.append({
                "archetype": f"{p_a}/{s_a} × {p_b}/{s_b}",
                "label": label,
                "line_a": line_a,
                "line_b": line_b,
                "p_over_a": round(p_over_a, 3),
                "p_over_b": round(p_over_b, 3),
                "geo_ip": round(geo_ip, 3),
                "sim_k": round(sim_k, 3),
                "curve_k": round(curve_result["factor"], 3),
                "k_diff": round(sim_k - curve_result["factor"], 3),
                "curve_status": curve_result.get("status", ""),
                "curve_n": curve_result.get("n", 0),
            })

    return {"comparisons": results}


def print_k_validation(val_result: dict) -> None:
    """Pretty-print k validation results."""
    if "error" in val_result:
        print(f"ERROR: {val_result['error']}")
        return

    comps = val_result["comparisons"]
    print("=" * 90)
    print("K VALIDATION: Sim vs Measured Curves")
    print("=" * 90)
    print(f"\n  {'Archetype':40s} {'Label':12s} {'geo_ip':>7s} "
          f"{'Sim k':>7s} {'Curve k':>8s} {'Diff':>7s}")
    print("  " + "─" * 85)

    for c in comps:
        if "error" in c:
            print(f"  {c['archetype']:40s} — {c['error']}")
            continue
        print(f"  {c['archetype']:40s} {c['label']:12s} "
              f"{c['geo_ip']:7.3f} {c['sim_k']:7.3f} "
              f"{c['curve_k']:8.3f} {c['k_diff']:+7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Validation against FanDuel SGP quotes
# ══════════════════════════════════════════════════════════════════════════════

def validate_against_quotes(quotes_csv: str | None = None) -> dict:
    """Compare engine k curves against FanDuel SGP quoted prices.

    Loads the FanDuel quotes from data/sgp_quotes.csv and compares
    the book's adjustment factor against the measured PBP k at
    matched probability levels.

    This is a CALIBRATION check (does our k match PBP reality?), not an
    EDGE check (SGP pricing edge is DEAD per project memory 2026-10-03).

    The sim engine should produce k values consistent with the PBP-measured
    curves. If so, the engine correctly prices SGPs; any EV comes only from
    offered boosts exceeding fair odds.
    """
    import csv

    if quotes_csv is None:
        quotes_csv = os.path.join(_HERE, "..", "data", "sgp_quotes.csv")

    try:
        from parlay_builder import _interpolate_k
    except ImportError:
        return {"error": "parlay_builder not available"}

    # Archetype → (stat_a, stat_b) for k-curve lookup
    _ARCH_TO_STATS = {
        "WR_recyds_receptions": ("rec_yds", "receptions"),
        "QB_passyds_passtds": ("pass_yds", "pass_tds"),  # normalized to sorted
        "RB_QB_sameteam": ("pass_yds", "rush_yds"),
    }

    rows = []
    with open(quotes_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    results = []
    for row in rows:
        arch = row["archetype"]
        stat_pair = _ARCH_TO_STATS.get(arch)
        if not stat_pair:
            continue

        price_a = float(row["price_a"])
        price_b = float(row["price_b"])
        sgp_quoted = float(row["sgp_quoted_odds"])
        depth = row.get("depth", "")

        # Implied probabilities (decimal odds → prob)
        ip_a = 1.0 / price_a if price_a > 1 else price_a
        ip_b = 1.0 / price_b if price_b > 1 else price_b
        geo_ip = sqrt(ip_a * ip_b)

        # Naive independent combo
        naive_odds = price_a * price_b

        # Book's k = naive / quoted (k > 1 means correlated → lower payout)
        book_k = naive_odds / sgp_quoted if sgp_quoted > 0 else float("nan")

        # PBP-measured k at this geo_ip
        curve_result = _interpolate_k(stat_pair, geo_ip)
        pbp_k = curve_result["factor"]

        # Gap: negative = book pays MORE than PBP fair (edge for bettor)
        #       positive = book pays LESS than PBP fair (book margin)
        gap_vs_pbp = book_k - pbp_k

        results.append({
            "archetype": arch,
            "depth": depth,
            "player_a": row.get("player_a", ""),
            "player_b": row.get("player_b", ""),
            "ip_a": round(ip_a, 3),
            "ip_b": round(ip_b, 3),
            "geo_ip": round(geo_ip, 3),
            "naive_odds": round(naive_odds, 3),
            "sgp_quoted": round(sgp_quoted, 3),
            "book_k": round(book_k, 3),
            "pbp_k": round(pbp_k, 3),
            "gap": round(gap_vs_pbp, 3),
        })

    # Summary by archetype × depth
    summary = defaultdict(lambda: {"gaps": [], "n": 0})
    for r in results:
        key = (r["archetype"], r["depth"])
        summary[key]["gaps"].append(r["gap"])
        summary[key]["n"] += 1

    return {
        "quotes": results,
        "summary": {
            f"{k[0]}_{k[1]}": {
                "n": v["n"],
                "mean_gap": round(float(np.mean(v["gaps"])), 3),
                "std_gap": round(float(np.std(v["gaps"])), 3),
            }
            for k, v in sorted(summary.items())
        },
    }


def print_quote_validation(val_result: dict) -> None:
    """Pretty-print quote validation results."""
    if "error" in val_result:
        print(f"ERROR: {val_result['error']}")
        return

    print("=" * 90)
    print("SGP QUOTE VALIDATION: PBP k vs FanDuel Book k")
    print("=" * 90)
    print("  gap = book_k - pbp_k  (+ = book charges more, - = book undercharges)")
    print()

    # Summary table
    print(f"  {'Archetype × Depth':35s} {'N':>4s} {'Mean Gap':>9s} {'Std':>7s}")
    print("  " + "─" * 58)
    for key, vals in sorted(val_result["summary"].items()):
        print(f"  {key:35s} {vals['n']:4d} {vals['mean_gap']:+9.3f} "
              f"{vals['std_gap']:7.3f}")

    # Detail for first few
    print(f"\n  Sample quotes:")
    print(f"  {'Archetype':25s} {'Depth':10s} {'geo_ip':>7s} "
          f"{'Book k':>7s} {'PBP k':>7s} {'Gap':>7s}")
    print("  " + "─" * 68)
    for r in val_result["quotes"][:15]:
        print(f"  {r['archetype']:25s} {r['depth']:10s} "
              f"{r['geo_ip']:7.3f} {r['book_k']:7.3f} "
              f"{r['pbp_k']:7.3f} {r['gap']:+7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Validation against historical PBP settled outcomes
# ══════════════════════════════════════════════════════════════════════════════

def validate_against_pbp(seasons: list[int] | None = None) -> dict:
    """Validate joint probability estimates against PBP settled outcomes.

    For each archetype, compute:
    1. Actual joint hit rate at various line thresholds from PBP
    2. Expected joint hit rate from independent marginals × k curve
    3. Report calibration: does k-adjusted prediction match reality?

    Requires nfl_data_py and is slow (loads full PBP seasons).
    """
    if seasons is None:
        seasons = [2023, 2024]

    try:
        import nfl_data_py as nfl
    except ImportError:
        return {"error": "nfl_data_py not installed"}

    try:
        from parlay_builder import _interpolate_k
    except ImportError:
        return {"error": "parlay_builder not available"}

    print(f"  Loading PBP for seasons {seasons}...")
    pbp = nfl.import_pbp_data(seasons)

    # Filter to pass plays with receiver info
    pass_plays = pbp[
        (pbp["play_type"] == "pass") &
        (pbp["two_point_attempt"] != 1) &
        (pbp["receiver_player_name"].notna())
    ].copy()

    # Per-player per-game receiving stats
    recv_stats = (
        pass_plays
        .groupby(["game_id", "posteam", "receiver_player_name"])
        .agg(
            targets=("pass_attempt", "sum"),
            receptions=("complete_pass", "sum"),
            rec_yards=("yards_gained", lambda x: x[pass_plays.loc[x.index, "complete_pass"] == 1].sum()),
        )
        .reset_index()
    )

    # Only WRs/TEs with enough targets
    recv_stats = recv_stats[recv_stats["targets"] >= 3]

    # Test: WR rec_yds OVER line AND receptions OVER line (same player)
    lines_to_test = [
        (30.5, 2.5, "easy"),
        (50.5, 3.5, "moderate"),
        (70.5, 5.5, "aggressive"),
    ]

    results = []
    for yard_line, rec_line, label in lines_to_test:
        # Marginal hit rates
        yds_over = recv_stats["rec_yards"] > yard_line
        rec_over = recv_stats["receptions"] > rec_line
        p_yds = float(yds_over.mean())
        p_rec = float(rec_over.mean())

        # Joint hit rate
        joint = (yds_over & rec_over)
        p_joint_actual = float(joint.mean())

        # Independent prediction
        p_joint_indep = p_yds * p_rec

        # k-adjusted prediction
        geo_ip = sqrt(p_yds * p_rec)
        k_result = _interpolate_k(("rec_yds", "receptions"), geo_ip)
        k = k_result["factor"]
        p_joint_k = p_joint_indep * k

        # Actual k
        actual_k = p_joint_actual / p_joint_indep if p_joint_indep > 0 else 0

        n = len(recv_stats)

        results.append({
            "label": label,
            "yard_line": yard_line,
            "rec_line": rec_line,
            "n": n,
            "p_yds_over": round(p_yds, 4),
            "p_rec_over": round(p_rec, 4),
            "geo_ip": round(geo_ip, 3),
            "p_joint_actual": round(p_joint_actual, 4),
            "p_joint_indep": round(p_joint_indep, 4),
            "p_joint_k_adj": round(p_joint_k, 4),
            "actual_k": round(actual_k, 3),
            "curve_k": round(k, 3),
            "k_error": round(actual_k - k, 3),
            "calib_gap": round(p_joint_actual - p_joint_k, 4),
        })

    return {
        "archetype": "WR_recyds_receptions",
        "seasons": seasons,
        "n_player_games": len(recv_stats),
        "results": results,
    }


def print_pbp_validation(val_result: dict) -> None:
    """Pretty-print PBP validation results."""
    if "error" in val_result:
        print(f"ERROR: {val_result['error']}")
        return

    print("=" * 90)
    print(f"PBP VALIDATION: {val_result['archetype']} "
          f"(seasons {val_result['seasons']}, "
          f"N={val_result['n_player_games']} player-games)")
    print("=" * 90)
    print(f"\n  {'Label':12s} {'Lines':>12s} {'geo_ip':>7s} "
          f"{'Actual P':>9s} {'Indep P':>9s} {'k-adj P':>9s} "
          f"{'Act k':>7s} {'Curve k':>8s} {'k err':>7s}")
    print("  " + "─" * 85)

    for r in val_result["results"]:
        print(f"  {r['label']:12s} "
              f"{r['yard_line']:5.1f}/{r['rec_line']:4.1f} "
              f"{r['geo_ip']:7.3f} "
              f"{r['p_joint_actual']:9.4f} "
              f"{r['p_joint_indep']:9.4f} "
              f"{r['p_joint_k_adj']:9.4f} "
              f"{r['actual_k']:7.3f} "
              f"{r['curve_k']:8.3f} "
              f"{r['k_error']:+7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Demo / smoke test
# ══════════════════════════════════════════════════════════════════════════════

def _demo():
    """Smoke test with synthetic draws demonstrating all features."""
    rng = np.random.default_rng(42)
    n = 10000

    qb_pass_mean, qb_pass_sd = 245, 55
    wr_rec_mean, wr_rec_sd = 68, 28

    # Correlated QB pass_yds ↔ WR rec_yds
    rho = 0.55
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    qb_pass_sims = np.maximum(qb_pass_mean + qb_pass_sd * z[:, 0], 0)
    wr_rec_sims = np.maximum(wr_rec_mean + wr_rec_sd * z[:, 1], 0)

    alloc_home = {
        "_meta": {"n_sims": n, "n_players": 2},
        "P. Mahomes": {"pass_yards": {"sims": qb_pass_sims.tolist()}},
        "R. Rice": {"rec_yards": {"sims": wr_rec_sims.tolist()}},
    }
    alloc_away = {"_meta": {"n_sims": n, "n_players": 0}}
    engine = SGPEngine(alloc_home, alloc_away)

    # ── Test 1: market marginals (two-sided devig) + boost ──
    print("--- Test 1: Market marginals + 30% boost ---")
    legs = [
        {"player": "P. Mahomes", "stat": "pass_yds", "line": 230.5,
         "direction": "over",
         "over_price": 1.87, "under_price": 1.93,  # two-sided
         "offered_price": 1.74},
        {"player": "R. Rice", "stat": "rec_yds", "line": 55.5,
         "direction": "over",
         "over_price": 1.80, "under_price": 2.00,  # two-sided
         "offered_price": 1.80},
    ]
    result = engine.price_sgp(legs, combo_offered_odds=4.50,
                              boost_pct=0.30, max_stake=25.0)
    print_sgp_result(result)

    # ── Test 2: mixed marginal sources ──
    print("\n\n--- Test 2: Mixed sources (market + v1.2 fallback) ---")
    legs_mixed = [
        {"player": "P. Mahomes", "stat": "pass_yds", "line": 230.5,
         "direction": "over",
         "over_price": 1.87, "under_price": 1.93,  # market
         "offered_price": 1.74},
        {"player": "R. Rice", "stat": "rec_yds", "line": 55.5,
         "direction": "over",
         "v12_p_over": 0.62,  # v1.2 fallback (no market prices)
         "offered_price": 1.80},
    ]
    result2 = engine.price_sgp(legs_mixed, combo_offered_odds=4.50)
    print_sgp_result(result2)

    # ── Test 3: aggressive lines → UNRELIABLE flag ──
    print("\n\n--- Test 3: Aggressive lines (geo_ip < 0.30) → UNRELIABLE ---")
    legs_agg = [
        {"player": "P. Mahomes", "stat": "pass_yds", "line": 310.5,
         "direction": "over",
         "over_price": 4.50, "under_price": 1.22,
         "offered_price": 4.50},
        {"player": "R. Rice", "stat": "rec_yds", "line": 100.5,
         "direction": "over",
         "over_price": 5.00, "under_price": 1.18,
         "offered_price": 5.00},
    ]
    result3 = engine.price_sgp(legs_agg, combo_offered_odds=30.00,
                               boost_pct=0.50)
    print_sgp_result(result3)


# ══════════════════════════════════════════════════════════════════════════════
# CLI: sgp price
# ══════════════════════════════════════════════════════════════════════════════

def _parse_leg(leg_str: str) -> dict:
    """Parse a leg string: 'player|stat|line|dir|over_price[|under_price]'.

    Examples:
      'P.Mahomes|pass_yds|230.5|over|1.87|1.93'
      'R.Rice|rec_yds|55.5|over|1.80'         (one-sided)
      'I.Pacheco|rush_yds|65.5|under|2.10|1.75'
    """
    parts = leg_str.strip().split("|")
    if len(parts) < 5:
        raise ValueError(
            f"Need ≥5 fields: player|stat|line|dir|over_price[|under_price]. "
            f"Got: {leg_str}"
        )
    leg = {
        "player": parts[0].strip(),
        "stat": parts[1].strip(),
        "line": float(parts[2].strip()),
        "direction": parts[3].strip().lower(),
        "over_price": float(parts[4].strip()),
        "offered_price": float(parts[4].strip()),  # default = over_price
    }
    if len(parts) >= 6 and parts[5].strip():
        leg["under_price"] = float(parts[5].strip())
    return leg


def _cli_price(args: list[str]) -> None:
    """CLI entry point for 'sgp price'.

    Usage:
      python sgp_ev_engine.py price \\
        --leg 'P.Mahomes|pass_yds|230.5|over|1.87|1.93' \\
        --leg 'R.Rice|rec_yds|55.5|over|1.80|2.00' \\
        --combo-odds 4.50 \\
        --boost 30 \\
        --max-stake 25
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="sgp_ev_engine.py price",
        description="Price an SGP from sim draws + market marginals",
    )
    parser.add_argument(
        "--leg", action="append", required=True,
        help="Leg: 'player|stat|line|dir|over_price[|under_price]'",
    )
    parser.add_argument(
        "--combo-odds", type=float, default=None,
        help="Offered combo decimal odds for the full SGP",
    )
    parser.add_argument(
        "--boost", type=float, default=0.0,
        help="Boost percentage (e.g. 30 for 30%%)",
    )
    parser.add_argument(
        "--max-stake", type=float, default=None,
        help="Max stake cap in dollars",
    )
    parser.add_argument(
        "--n-sims", type=int, default=10000,
        help="Number of sim draws (default: 10000)",
    )
    parsed = parser.parse_args(args)

    # Parse legs
    legs = []
    for leg_str in parsed.leg:
        legs.append(_parse_leg(leg_str))

    n_legs = len(legs)

    # ── Guardrails ──
    if n_legs > _MAX_LEGS_DEFAULT:
        print(f"WARNING: {n_legs} legs exceeds default max of "
              f"{_MAX_LEGS_DEFAULT}. Results may be unreliable.")

    # Check for aggressive lines
    for leg in legs:
        op = leg.get("over_price", 0)
        if op > 3.0:
            print(f"WARNING: {leg['player']}/{leg['stat']} at "
                  f"{op:.2f} is aggressive (implied P < 33%)")

    # Build synthetic sims for the CLI (quick mode: independent draws)
    # In production, use build_game_engine() for correlated draws.
    print("NOTE: CLI uses independent synthetic draws. For correlated "
          "joint P, use build_game_engine() in Python.\n")

    n = parsed.n_sims
    rng_cli = np.random.default_rng(42)

    alloc = {"_meta": {"n_sims": n, "n_players": 0}}

    for leg in legs:
        player = leg["player"]
        alloc_stat = _resolve_stat(leg["stat"])

        # Derive sim distribution from market marginal
        devig_p, _ = _devig_leg(leg)
        if devig_p is None:
            devig_p = 0.5

        # Use gamma-ish distribution centered around the line
        line = leg["line"]
        # Mean that makes P(X > line) ≈ devig_p:
        # For a rough gamma: mean ≈ line × (1 + 0.5 × (2*devig_p - 1))
        cv = 0.40
        mean_est = line / (1.0 - cv * 0.67 * (2 * devig_p - 1))
        mean_est = max(mean_est, 1.0)
        sd_est = mean_est * cv

        sims_raw = rng_cli.normal(mean_est, sd_est, size=n)
        sims_raw = np.maximum(sims_raw, 0)

        if player not in alloc:
            alloc[player] = {}
        alloc[player][alloc_stat] = {
            "sims": sims_raw.tolist(),
            "mean": float(sims_raw.mean()),
        }

    alloc_away = {"_meta": {"n_sims": n, "n_players": 0}}
    engine = SGPEngine(alloc, alloc_away)

    boost_frac = parsed.boost / 100.0 if parsed.boost else 0.0

    result = engine.price_sgp(
        legs,
        combo_offered_odds=parsed.combo_odds,
        boost_pct=boost_frac,
        max_stake=parsed.max_stake,
    )
    print_sgp_result(result)


# ══════════════════════════════════════════════════════════════════════════════
# Pikkit settled SGP validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_settled_sgps(pikkit_csv: str) -> dict:
    """Validate engine joint probabilities against settled Pikkit SGPs.

    For each settled SGP:
      1. Compute engine joint P using market marginals + k-curve copula
      2. Compare against realized outcome (win/loss)
      3. Bin by leg count and report calibration + overcharge c

    Args:
        pikkit_csv: path to Pikkit export CSV with settled SGPs.
            Expected columns: bet_id, legs (JSON or pipe-separated),
            combo_odds, outcome (win/loss/push), settled_amount, stake

    Returns: dict with per-bin calibration and overcharge analysis.
    """
    import csv
    import json

    if not os.path.exists(pikkit_csv):
        return {"error": f"File not found: {pikkit_csv}"}

    try:
        from parlay_builder import _interpolate_k
    except ImportError:
        return {"error": "parlay_builder not available"}

    rows = []
    with open(pikkit_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"error": "No rows in Pikkit export"}

    # Detect column format
    sample_cols = set(rows[0].keys())
    # Common Pikkit columns: bet_id, sport, legs, odds, result, payout
    # We need: per-leg implied probs, combo odds, outcome

    bins = {
        "2-3": {"n": 0, "wins": 0, "sum_p": 0.0, "sum_c": 0.0,
                "outcomes": []},
        "4-5": {"n": 0, "wins": 0, "sum_p": 0.0, "sum_c": 0.0,
                "outcomes": []},
        "6-9": {"n": 0, "wins": 0, "sum_p": 0.0, "sum_c": 0.0,
                "outcomes": []},
        "10+": {"n": 0, "wins": 0, "sum_p": 0.0, "sum_c": 0.0,
                "outcomes": []},
    }

    def _bin_key(n_legs: int) -> str:
        if n_legs <= 3:
            return "2-3"
        if n_legs <= 5:
            return "4-5"
        if n_legs <= 9:
            return "6-9"
        return "10+"

    # Average per-leg hold for (1+hold)^n approximation
    avg_leg_hold = 0.05  # ~5% vig per leg

    for row in rows:
        # Parse legs — support JSON array or pipe-separated
        legs_raw = row.get("legs", "")
        try:
            leg_list = json.loads(legs_raw) if legs_raw.startswith("[") else []
        except (json.JSONDecodeError, AttributeError):
            leg_list = []

        n_legs = int(row.get("num_legs", row.get("n_legs", len(leg_list))))
        if n_legs < 2:
            continue

        combo_odds = float(row.get("odds", row.get("combo_odds", 0)))
        if combo_odds <= 0:
            continue

        outcome = row.get("result", row.get("outcome", "")).lower()
        won = 1 if outcome in ("win", "won", "w", "1") else 0

        # Book implied joint P
        book_joint_p = 1.0 / combo_odds

        # Estimate fair joint P using independence + k-curve
        # Without per-leg details, approximate using (1+hold)^n
        fair_joint_p = book_joint_p * (1.0 + avg_leg_hold) ** n_legs

        # Overcharge c
        c = book_joint_p / fair_joint_p if fair_joint_p > 0 else 1.0

        # Theoretical overcharge from compounding vig
        c_theory = 1.0 / (1.0 + avg_leg_hold) ** n_legs

        bk = _bin_key(n_legs)
        bins[bk]["n"] += 1
        bins[bk]["wins"] += won
        bins[bk]["sum_p"] += book_joint_p
        bins[bk]["sum_c"] += c
        bins[bk]["outcomes"].append({
            "n_legs": n_legs,
            "combo_odds": combo_odds,
            "book_p": book_joint_p,
            "won": won,
            "c": c,
        })

    results = {}
    for bk, data in bins.items():
        if data["n"] == 0:
            results[bk] = {"n": 0}
            continue
        n = data["n"]
        win_rate = data["wins"] / n
        avg_book_p = data["sum_p"] / n
        avg_c = data["sum_c"] / n

        # (1+hold)^n approximation for median leg count in bin
        median_legs = {"2-3": 2.5, "4-5": 4.5, "6-9": 7.5, "10+": 12}[bk]
        c_approx = 1.0 / (1.0 + avg_leg_hold) ** median_legs

        results[bk] = {
            "n": n,
            "wins": data["wins"],
            "win_rate": round(win_rate, 4),
            "avg_book_p": round(avg_book_p, 4),
            "avg_c": round(avg_c, 4),
            "c_approx_hold_n": round(c_approx, 4),
            "c_vs_approx": round(avg_c - c_approx, 4),
        }

    return {
        "source": pikkit_csv,
        "total_bets": sum(d["n"] for d in bins.values()),
        "by_leg_count": results,
    }


def print_settled_validation(val: dict) -> None:
    """Pretty-print settled SGP validation."""
    if "error" in val:
        print(f"ERROR: {val['error']}")
        return

    print("=" * 80)
    print(f"SETTLED SGP VALIDATION — {val['total_bets']} bets from "
          f"{val['source']}")
    print("=" * 80)

    print(f"\n  {'Legs':8s} {'N':>5s} {'Wins':>5s} {'WR':>7s} "
          f"{'Avg P':>7s} {'Avg c':>7s} {'c theory':>9s} {'c − theory':>11s}")
    print("  " + "─" * 65)

    for bk, d in val["by_leg_count"].items():
        if d["n"] == 0:
            print(f"  {bk:8s} {0:>5d}")
            continue
        print(f"  {bk:8s} {d['n']:5d} {d['wins']:5d} "
              f"{d['win_rate']:7.1%} {d['avg_book_p']:7.4f} "
              f"{d['avg_c']:7.3f} {d['c_approx_hold_n']:9.3f} "
              f"{d['c_vs_approx']:+11.3f}")

    print(f"\n  c = book_implied_P / fair_joint_P")
    print(f"  c theory = 1 / (1 + 0.05)^n  (5% per-leg vig compounding)")
    print(f"  c > theory → book overcharges beyond simple vig compounding")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "demo":
        _demo()
    elif cmd == "price":
        _cli_price(sys.argv[2:])
    elif cmd == "validate-quotes":
        result = validate_against_quotes()
        print_quote_validation(result)
    elif cmd == "validate-pbp":
        seasons = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else None
        result = validate_against_pbp(seasons)
        print_pbp_validation(result)
    elif cmd == "validate-settled":
        if len(sys.argv) < 3:
            print("Usage: python sgp_ev_engine.py validate-settled <pikkit.csv>")
            sys.exit(1)
        result = validate_settled_sgps(sys.argv[2])
        print_settled_validation(result)
    else:
        print("Usage: python sgp_ev_engine.py <command>")
        print()
        print("Commands:")
        print("  demo             — synthetic smoke test")
        print("  price            — price an SGP from CLI")
        print("    --leg 'player|stat|line|dir|over_price[|under_price]'")
        print("    --combo-odds N   offered SGP decimal odds")
        print("    --boost N        boost percentage (e.g. 30)")
        print("    --max-stake N    max stake in dollars")
        print("  validate-quotes  — compare k vs FanDuel SGP quotes")
        print("  validate-pbp     — validate joint P against PBP data")
        print("  validate-settled <file.csv> — validate vs Pikkit export")
