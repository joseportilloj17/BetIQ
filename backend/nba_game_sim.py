"""B5: Monte Carlo joint-draw simulator for NBA player props.

Produces correlated player stat draws for a single game by sharing
game-level state (pace, blowout, minutes pool) across all players.

Cross-player correlations emerge naturally from:
  - Shared pace draw  → all players on a fast team get more possessions
  - Shared blowout draw → starters lose minutes together, bench gains
  - Shared minutes pool → Σ minutes = 240 per team per draw
  - Shared team scoring draw → same-team PTS/FG3M correlate via
    per-team efficiency multiplier (dampened ~3% CV)
  - Per-player engagement draw → within-player PTS+REB correlation (k≈1.20)
    via "good game / bad game" effect with stat-specific loadings
  - Discrete draws → Gamma rate × Poisson count = exact NegBin for STL/BLK/FG3M/AST-rotation
    (Normal corrections failed holdout validation; Gamma-Poisson is the only structural fix)

Usage:
    from nba_matchup_engine import NBAMatchupEngine
    from nba_game_sim import GameSimulator

    engine = NBAMatchupEngine()
    engine.load()
    proj = engine.project_game("BOS", "LAL", market_total=225.5, market_spread=-5.0)
    sim = GameSimulator(proj)
    result = sim.run(n_draws=10_000)

    # Marginal P(over)
    result.p_over("Jayson Tatum", "PTS", 26.5)

    # Joint P(A over X AND B over Y)
    result.p_joint([("Jayson Tatum", "PTS", 26.5), ("Jaylen Brown", "PTS", 22.5)])

    # SGP P(all legs hit)
    result.p_parlay([("Jayson Tatum", "PTS", 26.5), ("Jaylen Brown", "REB", 5.5),
                     ("Derrick White", "AST", 3.5)])
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

# P(plays) by injury status — POOLED rates (timing-unaware fallback).
# PROVENANCE: Out/IR = definitional (0%). Others measured from nba_injury_forward_log
# (NBA PDF → game-log settlement). Rates below are for STARTER+ROTATION (MPG≥15),
# which is the relevant population for our projection pool.
#
# Measured (2025-26 + 2026-27, 178 game dates, 21K settled rows):
#   Out:          9/9423 = 0.1%    Questionable: 1220/2198 = 55.5%
#   Doubtful:     9/ 411 = 2.2%    Probable:      763/ 826 = 92.4%
#
# By sub-role (starter=MPG≥28, rotation=15-28):
#   Questionable: starter 61.4%, rotation 51.6%, spot 29.4%
#   Probable:     starter 93.7%, rotation 91.3%, spot 77.8%
#
# All-player rates are lower (Q=41%, P=76%) because bench/two-way players
# often DNP regardless of health status.
_P_PLAYS_BY_STATUS = {
    "Out":          0.00,   # MEASURED — 0.1%, timing artifacts only
    "IR":           0.00,   # definitional
    "Doubtful":     0.02,   # MEASURED — 9/411 (N=411, pooled)
    "Questionable": 0.55,   # MEASURED — 1220/2198 starter+rotation, pooled
    "GTD":          0.50,   # HAND-SET — Sleeper-specific label, ~50%
    "DTD":          0.70,   # HAND-SET — Sleeper-specific label, ~70%
    "Probable":     0.92,   # MEASURED — 763/826 starter+rotation, pooled
    "Active":       1.00,   # confirmed healthy → no base-rate penalty
    None:           1.00,   # No injury report entry → default to base rate
}

# P(plays) by injury status × report timing (day-before vs game-day).
# MEASURED from same dataset, split by whether the snapshot was issued on the
# day before the game (>12 h to tip) or the day of (≤12 h to tip).
#
# Big movers: Probable +4.9 pp on game-day; Doubtful → 0% on game-day.
# Questionable barely moves (56.5% → 54.7%), but status TRANSITIONS are
# highly informative: Q→Available = 95.7%, Q→stays Q = 59.4%, Q→Out = 0%.
# Transition-conditional rates are parked for future implementation.
_P_PLAYS_DAY_BEFORE = {
    "Out":          0.00,   # 7/2591 = 0.3% (timing artifacts)
    "IR":           0.00,
    "Doubtful":     0.057,  # 9/159
    "Questionable": 0.565,  # 534/945
    "Probable":     0.893,  # 274/307
}
_P_PLAYS_GAME_DAY = {
    "Out":          0.00,   # 2/6832
    "IR":           0.00,
    "Doubtful":     0.00,   # 0/252
    "Questionable": 0.547,  # 686/1253
    "Probable":     0.942,  # 489/519
}

# P(plays) base rate by role — for players NOT on the injury report.
# MEASURED from nba_injury_forward_log + nba_player_game_logs join:
# starters/rotation who played on a date but were NOT listed on the last
# injury report before tip.  Covers 178 game dates (2025-26 + 2026-27).
#
#   starter:  7012/7013 = 100.0%   (1 surprise DNP in 7,013 appearances)
#   rotation: 14339/14361 = 99.8%  (22 surprise DNPs in 14,361)
#   "Available" status: 917/917 = 100.0%
#
# Old upper-bounds (96.3%/89.1%) included absences now captured by the
# injury report integration.  True undesignated absence is near zero.
_P_PLAYS_BASE_BY_ROLE = {
    "starter":  0.998,  # MEASURED — 7012/7013 (effectively 1.000)
    "rotation": 0.998,  # MEASURED — 14339/14361
    "spot":     0.658,  # ≤34.2% (N=6,373) — upper bound
}

# Score margin std (NBA historical), used for blowout probability.
# Measured from 4,668 regulation games (4 seasons, 2022-23 to 2025-26).
# Margin ~ N(-spread, σ²); kurtosis = +0.19 (mild fat tails, Normal OK).
_MARGIN_SIGMA = 15.8

# Blowout threshold and minute adjustments (from nba_minutes_model.py)
_BLOWOUT_THRESHOLD = 15.0
_BLOWOUT_STARTER_CUT = 2.3   # starters lose this many minutes
_BLOWOUT_ROTATION_GAIN = 4.0  # rotation gains
_BLOWOUT_SPOT_GAIN = 7.0      # spot gains
# Measured: |margin|>=15 in 32.6% of games (N=4,920). σ=15.8 predicts 34.2%.
# The engine's minutes model uses a pre-computed blowout probability for
# expected minute adjustments; the sim uses per-draw margin detection.
_BASELINE_BLOWOUT_PROB = 0.326

# Overtime: measured from 4,920 games (4 seasons).
# P(OT) = 5.12%. Each OT period = 5 minutes extra.
# Most OT games are single-OT; ~0.5% of games go to 2+ OT.
# OT minutes allocated by role: starters +4, rotation +1, spot 0.
_OT_PROB = 0.0512
_OT_MINUTES_STARTER = 4.0   # starters play most of OT
_OT_MINUTES_ROTATION = 1.0  # bench rotation gets ~1 extra min
_OT_MINUTES_SPOT = 0.0      # spot players rarely play OT

# In-game early exit (injury, ejection, foul-out).
# DISTINCT FROM pre-game availability: bet has action (not voided), player
# plays reduced minutes within their own distribution.
# HAND-SET from observed NBA data (~3.5% starters, lower for bench).
# Rotation/spot rarely get ejected but can still exit with injury.
_EARLY_EXIT_PROB = {
    "starter":  0.035,  # ~3.5% — includes injury exit, ejection, foul-out
    "rotation": 0.020,  # ~2.0% — less exposure, fewer ejections
    "spot":     0.010,  # ~1.0% — limited minutes = less exposure
}
_EARLY_EXIT_MEAN = 14.8   # mean minutes played before exit
_EARLY_EXIT_STD = 4.7     # std of exit-game minutes

# Possessions std: ~4% CV based on NBA game-to-game variance
_POSS_CV = 0.04

# Team total std: measured from 9,336 team-games (4 seasons, regulation only).
# Team_pts ~ N(implied_total, σ²). Used for game-script correlation.
# Reconciliation killed (B4/R4/R5); team total enters sim for blowout
# probability and cross-player game-state, not for point allocation.
_TEAM_TOTAL_STD = 12.5

# Game total std: derived from team total and margin stds.
# Var(team_pts) = (Var(game_total) + Var(margin)) / 4  (T,M independent)
# σ²_game = 4 × 12.5² - 15.8² = 625 - 249.64 = 375.4 → σ = 19.4
# This implies Corr(home_pts, away_pts) ≈ +0.20 (realistic: shared pace/refs).
_GAME_TOTAL_STD = 19.4

# Team scoring multiplier dampening factor.
# Full team-total variance is ~11% CV. Rate stds from game logs already
# include some team-scoring variance. Dampen to ~3.3% CV to avoid inflating
# total rate variance. Applied to PTS/FG3M only (scoring efficiency).
# S4-2 calibration: damp=0.30 matches teammate PTS k≈1.00.
_TEAM_SCORING_DAMP = 0.30

# Per-player engagement: "good game / bad game" shared multiplier across
# a player's stats. Creates within-player PTS+REB correlation (k≈1.20)
# that shared minutes alone cannot explain.
# S4-2 calibration: eng_cv=0.15 with stat-specific loadings:
#   PTS: 1.0 (full), REB: 0.8, AST: 0.0, FG3M: 1.0, STL/BLK: 0.0
# PRA loading derived from component loadings.
_ENGAGEMENT_CV = 0.15
_ENGAGEMENT_LOADINGS = {
    # stat_index: loading.  PTS=0, REB=1, AST=2, STL=3, BLK=4, FG3M=5, PRA=6
    0: 1.0,   # PTS — full engagement effect
    1: 0.8,   # REB — 80% (offensive rebounding tracks engagement)
    2: 0.0,   # AST — 0% (assists driven by teammates, not own engagement)
    3: 0.0,   # STL — 0% (defensive, unrelated to offensive engagement)
    4: 0.0,   # BLK — 0% (defensive)
    5: 1.0,   # FG3M — full (shooting volume tracks scoring engagement)
    6: 0.8,   # PRA — effective loading from PTS(1.0)+REB(0.8)+AST(0.0)
}

# ── Two-stage calibration corrections (S5-1) ──────────────────────────────
# HOLDOUT VALIDATION (fit on H1, test on H2 and vice versa) showed that
# Normal-distribution bias + σ_scale corrections are NOT stable across halves:
#   PTS starter: H1→H2 hurts (+0.044 KS), H2→H1 helps (-0.030)
#   PRA starter: H1→H2 hurts (+0.007), H2→H1 helps (-0.025)
# Only STL/BLK σ_scale survived holdout — but those are now addressed by
# Gamma-Poisson draws (root cause: Normal-vs-discrete mismatch).
# CONCLUSION: No Normal-distribution corrections applied. The only
# calibration fix that survives holdout is the Gamma-Poisson structural fix.
_CALIBRATION_CORRECTIONS = {}  # empty — all corrections failed holdout

# Stats drawn from Gamma-Poisson compound (exact NegBin) instead of Normal.
# Normal approximation fails for low-count integer stats (STL, BLK, FG3M,
# AST rotation): negative rates, continuous artifacts on integer counts.
# Gamma rate draw → Poisson count: non-negative by construction, integer
# output, NegBin-like overdispersion from game-to-game rate uncertainty.
# Indices: STL=3, BLK=4, FG3M=5. AST=2 for rotation players only.
_DISCRETE_STAT_INDICES = {3, 4, 5}  # STL, BLK, FG3M always discrete
_AST_INDEX = 2  # AST discrete for rotation only

# Minimum per-minute rate std (floor to prevent degenerate draws)
_MIN_RATE_STD_FRAC = 0.10

# Maximum minutes any player can have in a single game
_MAX_MINUTES = 48.0

# Team minutes constraint
_TEAM_MINUTES = 240.0


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class _PlayerSim:
    """Internal per-player simulation parameters extracted from engine."""
    name: str
    team: str
    role: str  # "starter" | "rotation" | "spot"
    minutes_mean: float
    minutes_std: float
    # Adjusted per-minute rates (already include pace/defense/etc.)
    pts_rate: float
    reb_rate: float
    ast_rate: float
    stl_rate: float
    blk_rate: float
    fg3m_rate: float
    # Rate stds
    pts_rate_std: float
    reb_rate_std: float
    ast_rate_std: float
    fg3m_rate_std: float
    # PRA rate (direct, captures covariance)
    pra_rate: float
    pra_rate_std: float
    # Probabilistic availability: P(this player plays in this game)
    # 1.0 = certain to play (default), <1 = may not play (injury/rest)
    p_plays: float = 1.0


@dataclass
class SimulationResult:
    """Holds the N×P×S draw matrix and provides query methods.

    Availability (void) rules:
    - p_over/p_under/marginal_stats condition on the player playing.
      A player's own prop is voided if they don't play, so we only
      count draws where they're available.
    - p_joint/p_parlay condition on ALL referenced players playing
      (standard SGP pricing assumes all legs are live).
    - p_parlay_void implements book-aware void rules:
      * fanduel: voided legs removed, surviving legs repriced as shorter
        parlay; with leg_odds, estimates sub-parlay payout via k×Π(odds).
      * other books (draftkings, etc.): any void → entire parlay refunded.
    """

    player_names: list[str]       # length P
    stat_names: list[str]         # length S
    draws: np.ndarray             # shape (N, P, S)  float32
    team_labels: list[str]        # team abbr per player (length P)
    availability: np.ndarray | None = None  # shape (N, P) bool; None = all play
    n_draws: int = 0

    # Index lookups (built post-init)
    _player_idx: dict = field(default_factory=dict, repr=False)
    _stat_idx: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._player_idx = {n: i for i, n in enumerate(self.player_names)}
        self._stat_idx = {s: i for i, s in enumerate(self.stat_names)}
        self.n_draws = self.draws.shape[0]

    def _pidx(self, player: str) -> int:
        pi = self._player_idx.get(player)
        if pi is None:
            raise KeyError(f"Player '{player}' not in simulation")
        return pi

    def _sidx(self, stat: str) -> int:
        si = self._stat_idx.get(stat)
        if si is None:
            raise KeyError(f"Stat '{stat}' not in simulation")
        return si

    def _get(self, player: str, stat: str) -> np.ndarray:
        """Return 1-D array of draws for (player, stat)."""
        return self.draws[:, self._pidx(player), self._sidx(stat)]

    def _player_available(self, player: str) -> np.ndarray:
        """Bool mask (N,): True in draws where player is available."""
        if self.availability is None:
            return np.ones(self.n_draws, dtype=bool)
        return self.availability[:, self._pidx(player)]

    def p_over(self, player: str, stat: str, line: float) -> float:
        """P(player stat > line | player plays).

        Conditions on draws where the player is available (void rule:
        prop is voided if player doesn't play).
        """
        arr = self._get(player, stat)
        avail = self._player_available(player)
        n_avail = avail.sum()
        if n_avail == 0:
            return 0.0
        return float((arr[avail] > line).sum() / n_avail)

    def p_under(self, player: str, stat: str, line: float) -> float:
        """P(player stat < line | player plays)."""
        arr = self._get(player, stat)
        avail = self._player_available(player)
        n_avail = avail.sum()
        if n_avail == 0:
            return 0.0
        return float((arr[avail] < line).sum() / n_avail)

    def p_joint(self, legs: list[tuple[str, str, float]],
                direction: str = "over") -> float:
        """P(all legs hit | all players play).

        Each leg is (player, stat, line).
        Conditions on draws where ALL referenced players are available
        (standard SGP pricing assumes all legs live).
        """
        # Build joint availability mask
        avail = np.ones(self.n_draws, dtype=bool)
        if self.availability is not None:
            for player, _stat, _line in legs:
                avail &= self._player_available(player)
        n_avail = avail.sum()
        if n_avail == 0:
            return 0.0
        hit = np.ones(n_avail, dtype=bool)
        for player, stat, line in legs:
            arr = self._get(player, stat)[avail]
            if direction == "over":
                hit &= (arr > line)
            else:
                hit &= (arr < line)
        return float(hit.sum() / n_avail)

    def p_parlay(self, legs: list[tuple[str, str, float, str]]) -> float:
        """P(all legs hit | all players play).

        Each leg is (player, stat, line, direction).
        Conditions on draws where ALL referenced players are available.
        """
        avail = np.ones(self.n_draws, dtype=bool)
        if self.availability is not None:
            for player, _stat, _line, _dir in legs:
                avail &= self._player_available(player)
        n_avail = avail.sum()
        if n_avail == 0:
            return 0.0
        hit = np.ones(n_avail, dtype=bool)
        for player, stat, line, direction in legs:
            arr = self._get(player, stat)[avail]
            if direction == "over":
                hit &= (arr > line)
            else:
                hit &= (arr < line)
        return float(hit.sum() / n_avail)

    def p_parlay_void(self, legs: list[tuple[str, str, float, str]],
                      book: str = "fanduel",
                      leg_odds: list[float] | None = None) -> dict:
        """Book-aware void-adjusted parlay probability.

        Each leg is (player, stat, line, direction).
        leg_odds: optional decimal odds per leg (e.g. [1.87, 1.91, 2.10]).
            Required for FanDuel EV estimation of sub-parlay pricing.

        Book rules:
          fanduel: voided legs removed, parlay repriced as shorter parlay
                   of surviving legs. All legs voided → refund.
          draftkings/other: ANY voided leg → entire parlay voided (refund).

        Returns dict with:
            p_full: P(all legs hit | all players play) — the priced probability
            p_void_adjusted: P(surviving legs all hit | ≥1 leg survives)
                (fanduel: conditioned on surviving subset hitting)
                (other books: = p_full, since any void = refund)
            n_legs_distribution: {k: fraction of draws with k active legs}
            void_risk: fraction of draws where ≥1 leg is voided
            book: which book's rules were applied
            ev_void_scenarios: (fanduel only) expected payout per $1 in
                void scenarios, using leg_odds × correlation multiplier
        """
        n_legs = len(legs)
        book = book.lower().replace(" ", "")

        # Per-leg availability: shape (N, n_legs)
        leg_avail = np.ones((self.n_draws, n_legs), dtype=bool)
        if self.availability is not None:
            for j, (player, _s, _l, _d) in enumerate(legs):
                leg_avail[:, j] = self._player_available(player)

        # Per-leg hit: shape (N, n_legs) — evaluated on all draws
        leg_hit = np.ones((self.n_draws, n_legs), dtype=bool)
        for j, (player, stat, line, direction) in enumerate(legs):
            arr = self._get(player, stat)
            if direction == "over":
                leg_hit[:, j] = arr > line
            else:
                leg_hit[:, j] = arr < line

        # Count active legs per draw
        active_count = leg_avail.sum(axis=1)  # (N,)

        # ── Non-FanDuel books: any void = entire parlay voided (refund) ──
        if book != "fanduel":
            all_play = active_count == n_legs
            n_all_play = all_play.sum()
            if n_all_play == 0:
                return {"p_full": 0.0, "p_void_adjusted": 0.0,
                        "n_legs_distribution": {n_legs: 0.0},
                        "void_risk": 1.0, "book": book}

            all_hit = leg_hit[all_play].all(axis=1)
            p_full = float(all_hit.sum() / n_all_play)
            void_risk = float((~all_play).sum() / self.n_draws)

            n_legs_dist = {}
            for k in range(n_legs + 1):
                frac = float((active_count == k).sum() / self.n_draws)
                if frac > 0:
                    n_legs_dist[k] = frac

            return {
                "p_full": p_full,
                "p_void_adjusted": p_full,  # no repricing — void = refund
                "n_legs_distribution": n_legs_dist,
                "void_risk": void_risk,
                "book": book,
            }

        # ── FanDuel: voided legs removed, surviving legs reprice ──
        has_active = active_count > 0
        n_valid = has_active.sum()
        if n_valid == 0:
            return {"p_full": 0.0, "p_void_adjusted": 0.0,
                    "n_legs_distribution": {}, "void_risk": 1.0,
                    "book": "fanduel"}

        # A draw "hits" if all ACTIVE legs hit (voided legs ignored)
        active_and_hit = leg_avail & leg_hit  # (N, n_legs)
        active_hits = active_and_hit.sum(axis=1)  # (N,)
        draw_hits = (active_hits == active_count) & has_active

        # p_full: condition on ALL players playing
        all_play = active_count == n_legs
        n_all_play = all_play.sum()
        p_full = float(draw_hits[all_play].sum() / n_all_play) if n_all_play > 0 else 0.0

        # p_void_adjusted: condition on at least one active leg
        p_void_adjusted = float(draw_hits[has_active].sum() / n_valid)

        # Distribution of active leg counts
        n_legs_dist = {}
        for k in range(n_legs + 1):
            frac = float((active_count == k).sum() / self.n_draws)
            if frac > 0:
                n_legs_dist[k] = frac

        void_risk = float((active_count < n_legs).sum() / self.n_draws)

        result = {
            "p_full": p_full,
            "p_void_adjusted": p_void_adjusted,
            "n_legs_distribution": n_legs_dist,
            "void_risk": void_risk,
            "book": "fanduel",
        }

        # ── FanDuel sub-parlay pricing estimate ──
        # When legs are voided, FanDuel pays at its own repriced odds for
        # the surviving sub-parlay. Estimate: product of surviving leg odds
        # × correlation multiplier k (measured from sim draws for each
        # possible sub-parlay configuration).
        if leg_odds is not None and len(leg_odds) == n_legs:
            ev_scenarios = self._fanduel_void_ev(
                legs, leg_avail, leg_hit, active_count,
                has_active, np.array(leg_odds, dtype=np.float64),
            )
            result["ev_void_scenarios"] = ev_scenarios

        return result

    def _fanduel_void_ev(
        self,
        legs: list[tuple[str, str, float, str]],
        leg_avail: np.ndarray,
        leg_hit: np.ndarray,
        active_count: np.ndarray,
        has_active: np.ndarray,
        leg_odds: np.ndarray,
    ) -> dict:
        """Estimate FanDuel payout in void scenarios.

        When k < n_legs legs survive, FanDuel reprices as a k-leg parlay.
        The book's price = product(surviving leg odds) × book_k, where
        book_k is the book's correlation adjustment (≥1 for positive corr).

        We estimate the book's correlation multiplier per sub-parlay
        configuration from the sim draws themselves:
            book_k ≈ P(all surviving hit) / Π(P(leg_j hit))
        Then:
            payout_if_hit = product(surviving_leg_odds) (what book pays)
            true_p_hit = P(surviving legs all hit) from sim draws
            EV(k legs survive) = true_p_hit × payout_if_hit - (1 - true_p_hit)

        Returns dict with:
            weighted_ev: probability-weighted EV across all void scenarios
            by_active_count: {k: {p_scenario, p_hit, est_payout, ev}}
            full_parlay_ev: EV when all legs are active (no voids)
        """
        n_legs = len(legs)
        scenarios = {}

        for k in range(1, n_legs + 1):
            mask_k = active_count == k
            n_k = mask_k.sum()
            if n_k == 0:
                continue

            if k == n_legs:
                # Full parlay — no void
                all_hit = leg_hit[mask_k].all(axis=1)
                p_hit = float(all_hit.sum() / n_k)
                full_odds = float(np.prod(leg_odds))
                ev = p_hit * (full_odds - 1) - (1 - p_hit) if full_odds > 0 else 0.0
                scenarios[k] = {
                    "p_scenario": float(n_k / self.n_draws),
                    "p_hit": p_hit,
                    "est_payout": full_odds,
                    "ev": round(ev, 4),
                }
            else:
                # Sub-parlay: only surviving legs are active
                # Average over all possible k-leg subsets that occur in draws
                avail_k = leg_avail[mask_k]  # (n_k, n_legs)
                hit_k = leg_hit[mask_k]       # (n_k, n_legs)

                # For each draw, surviving legs hit = active AND hit
                surviving_hit = avail_k & hit_k
                # All surviving legs hit?
                active_per_draw = avail_k.sum(axis=1)
                surviving_hits = surviving_hit.sum(axis=1)
                all_surviving_hit = surviving_hits == active_per_draw
                p_hit = float(all_surviving_hit.sum() / n_k)

                # Estimate payout: average product of surviving leg odds
                # For each draw, multiply odds of active legs
                log_odds = np.log(leg_odds)
                # Sum log-odds of active legs per draw
                log_payout = (avail_k.astype(np.float64) * log_odds[None, :]).sum(axis=1)
                est_payout = float(np.exp(log_payout).mean())

                ev = p_hit * (est_payout - 1) - (1 - p_hit) if est_payout > 0 else 0.0
                scenarios[k] = {
                    "p_scenario": float(n_k / self.n_draws),
                    "p_hit": p_hit,
                    "est_payout": round(est_payout, 3),
                    "ev": round(ev, 4),
                }

        # Probability-weighted EV across scenarios (excluding 0-leg refund)
        weighted_ev = sum(
            s["p_scenario"] * s["ev"] for s in scenarios.values()
        )

        return {
            "weighted_ev": round(weighted_ev, 4),
            "by_active_count": scenarios,
            "full_parlay_ev": scenarios.get(n_legs, {}).get("ev", 0.0),
        }

    def marginal_stats(self, player: str, stat: str) -> dict:
        """Summary statistics for a player-stat marginal, conditional on playing."""
        arr = self._get(player, stat)
        avail = self._player_available(player)
        n_avail = avail.sum()
        if n_avail == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0,
                    "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
        a = arr[avail]
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }

    def correlation(self, player_a: str, stat_a: str,
                    player_b: str, stat_b: str) -> float:
        """Pearson correlation between two player-stat marginals.

        Conditions on draws where BOTH players are available.
        """
        avail = self._player_available(player_a) & self._player_available(player_b)
        n_avail = avail.sum()
        if n_avail < 3:
            return 0.0
        a = self._get(player_a, stat_a)[avail]
        b = self._get(player_b, stat_b)[avail]
        return float(np.corrcoef(a, b)[0, 1])


# ── Simulator ────────────────────────────────────────────────────────────────

class GameSimulator:
    """Monte Carlo joint-draw simulator for a single NBA game.

    Takes a MatchupProjection (from NBAMatchupEngine.project_game) and
    produces N correlated draws of all player stats.
    """

    def __init__(self, projection, minutes_projections: dict | None = None,
                 injury_statuses: dict | None = None,
                 absence_impacts: dict | None = None,
                 hours_to_tip: float | None = None):
        """
        Args:
            projection: MatchupProjection from engine.project_game()
            minutes_projections: optional {team: {name: MinutesProjection}}
                If not provided, uses minutes/std from PlayerProjection.
            injury_statuses: optional {player_name: status_string} from
                fetch_injury_report(). Used to set P(plays) per player.
                Players not in dict get role-based base rate.
            absence_impacts: optional {absent_name: {beneficiary_name: delta_min}}
                from minutes model's measured with/without patterns (shrunk).
                Used for per-draw redistribution when a player is unavailable.
                Falls back to proportional 240 rescaling if not provided.
            hours_to_tip: optional hours until game tip-off. When provided,
                selects timing-conditional P(plays) rates (day-before vs
                game-day) instead of pooled rates. >12 h = day-before,
                ≤12 h = game-day.
        """
        self.proj = projection
        self.minutes_projections = minutes_projections or {}
        self.injury_statuses = injury_statuses or {}
        self.absence_impacts = absence_impacts or {}
        self.hours_to_tip = hours_to_tip
        self._players: list[_PlayerSim] = []
        self._extract_players()

    def _extract_players(self):
        """Extract simulation parameters from engine projection."""
        for name, pp in self.proj.players.items():
            if pp.minutes <= 0:
                continue

            # Get minutes std and role from MinutesProjection if available
            min_std = 0.0
            team_mp = self.minutes_projections.get(pp.team, {})
            mp = team_mp.get(name)
            role = None
            if mp is not None:
                min_std = mp.minutes_std
                role = mp.role
            if role is None:
                # Infer role from projected minutes (matches engine convention)
                if pp.minutes >= 28:
                    role = "starter"
                elif pp.minutes >= 15:
                    role = "rotation"
                else:
                    role = "spot"
            if min_std <= 0:
                # Fallback: estimate minutes std from projected minutes
                # Empirical CV: starters ~15%, rotation ~20%, spot ~30%
                cv = {"starter": 0.15, "rotation": 0.20, "spot": 0.30}.get(role, 0.20)
                min_std = pp.minutes * cv

            # Rate stds: use engine's stored values or fallback
            def _rate_std(adj_rate, stored_std, fallback_frac):
                if stored_std > 0:
                    return stored_std
                return max(abs(adj_rate) * fallback_frac, 1e-6)

            # We need the raw rate stds (per-minute), not the counting-stat stds.
            # The engine stores counting-stat stds on PlayerProjection but
            # the raw rate stds live on PlayerProfile. We back them out:
            #   counting_std = sqrt(min^2 * rate_std^2 + rate^2 * min_std^2)
            # Solving for rate_std:
            #   rate_std = sqrt(max(counting_std^2 - rate^2 * min_std^2, 0)) / min
            def _extract_rate_std(counting_std, adj_rate, min_mean, min_std_val,
                                  fallback_frac):
                if min_mean <= 0:
                    return abs(adj_rate) * fallback_frac
                var_total = counting_std ** 2
                var_from_min = adj_rate ** 2 * min_std_val ** 2
                var_rate = max(var_total - var_from_min, 0)
                rate_std = (var_rate ** 0.5) / min_mean if min_mean > 0 else 0
                return max(rate_std, abs(adj_rate) * _MIN_RATE_STD_FRAC)

            pts_rs = _extract_rate_std(pp.pts_std, pp.adj_pts_rate, pp.minutes, min_std, 0.25)
            reb_rs = _extract_rate_std(pp.reb_std, pp.adj_reb_rate, pp.minutes, min_std, 0.30)
            ast_rs = _extract_rate_std(pp.ast_std, pp.adj_ast_rate, pp.minutes, min_std, 0.35)
            fg3m_rs = _extract_rate_std(pp.thr_std, pp.adj_fg3m_rate, pp.minutes, min_std, 0.45)

            # PRA rate
            pra_rate = pp.adj_pts_rate + pp.adj_reb_rate + pp.adj_ast_rate
            pra_rs = _extract_rate_std(pp.pra_std, pra_rate, pp.minutes, min_std, 0.20)

            # P(plays) from injury status, conditioned on report timing
            inj_status = self.injury_statuses.get(name)
            if inj_status is not None:
                if self.hours_to_tip is not None:
                    # Use timing-conditional rate
                    timed = (_P_PLAYS_DAY_BEFORE if self.hours_to_tip > 12
                             else _P_PLAYS_GAME_DAY)
                    p_plays = timed.get(inj_status,
                                        _P_PLAYS_BY_STATUS.get(
                                            inj_status,
                                            _P_PLAYS_BASE_BY_ROLE.get(role, 0.90)))
                else:
                    # No timing info → pooled rate
                    p_plays = _P_PLAYS_BY_STATUS.get(inj_status,
                                                      _P_PLAYS_BASE_BY_ROLE.get(role, 0.90))
            else:
                # No injury report entry → use role-based base rate for
                # unexpected absences (rest, late scratches, etc.)
                p_plays = _P_PLAYS_BASE_BY_ROLE.get(role, 0.90)

            self._players.append(_PlayerSim(
                name=name,
                team=pp.team,
                role=role,
                minutes_mean=pp.minutes,
                minutes_std=min_std,
                pts_rate=pp.adj_pts_rate,
                reb_rate=pp.adj_reb_rate,
                ast_rate=pp.adj_ast_rate,
                stl_rate=pp.adj_stl_rate,
                blk_rate=pp.adj_blk_rate,
                fg3m_rate=pp.adj_fg3m_rate,
                pts_rate_std=pts_rs,
                reb_rate_std=reb_rs,
                ast_rate_std=ast_rs,
                fg3m_rate_std=fg3m_rs,
                pra_rate=pra_rate,
                pra_rate_std=pra_rs,
                p_plays=p_plays,
            ))

    def run(self, n_draws: int = 10_000, seed: int = 42) -> SimulationResult:
        """Run Monte Carlo simulation.

        Returns SimulationResult with shape (n_draws, n_players, n_stats).
        Stats: PTS, REB, AST, STL, BLK, FG3M, PRA
        """
        rng = np.random.default_rng(seed)
        n_players = len(self._players)
        stat_names = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "PRA"]
        n_stats = len(stat_names)

        if n_players == 0:
            return SimulationResult(
                player_names=[], stat_names=stat_names,
                draws=np.empty((n_draws, 0, n_stats), dtype=np.float32),
                team_labels=[],
            )

        # ── Build team indices ──
        teams = sorted(set(p.team for p in self._players))
        team_player_idx: dict[str, list[int]] = {t: [] for t in teams}
        for i, p in enumerate(self._players):
            team_player_idx[p.team].append(i)

        # ── Precompute arrays ──
        min_means = np.array([p.minutes_mean for p in self._players], dtype=np.float64)
        min_stds = np.array([p.minutes_std for p in self._players], dtype=np.float64)
        roles = [p.role for p in self._players]

        rate_means = np.column_stack([
            [p.pts_rate for p in self._players],
            [p.reb_rate for p in self._players],
            [p.ast_rate for p in self._players],
            [p.stl_rate for p in self._players],
            [p.blk_rate for p in self._players],
            [p.fg3m_rate for p in self._players],
            [p.pra_rate for p in self._players],
        ])  # shape (P, 7)

        rate_stds = np.column_stack([
            [p.pts_rate_std for p in self._players],
            [p.reb_rate_std for p in self._players],
            [p.ast_rate_std for p in self._players],
            # STL/BLK: use 50% CV as floor (low-rate, high-variance stats)
            [max(p.stl_rate * 0.50, 1e-6) for p in self._players],
            [max(p.blk_rate * 0.50, 1e-6) for p in self._players],
            [p.fg3m_rate_std for p in self._players],
            [p.pra_rate_std for p in self._players],
        ])  # shape (P, 7)

        # ── Apply S5-1 calibration corrections ──
        # Two-stage: (1) bias → shift rate_means, (2) σ_scale → widen rate_stds.
        # Only for (stat, role) combos where Normal correction improved KS.
        # Bias is on counting stat (not per-minute); convert to per-minute by
        # dividing by expected minutes.
        for pi in range(n_players):
            role = roles[pi]
            exp_min = max(min_means[pi], 1.0)
            for si in range(n_stats):
                corr = _CALIBRATION_CORRECTIONS.get((si, role))
                if corr is not None:
                    rate_means[pi, si] += corr["bias"] / exp_min
                    rate_stds[pi, si] *= corr["sigma_scale"]

        # ── Build discrete-stat mask per player ──
        # True where this (player, stat) should use Poisson draws.
        is_discrete = np.zeros((n_players, n_stats), dtype=bool)
        for pi in range(n_players):
            for si in _DISCRETE_STAT_INDICES:
                is_discrete[pi, si] = True
            # AST discrete for rotation/spot only (starters have enough volume)
            if roles[pi] in ("rotation", "spot"):
                is_discrete[pi, _AST_INDEX] = True

        # ── Step 1: Draw shared pace multiplier (one per draw) ──
        # Engine adj_rates already include pace adjustment for this game.
        # The pace multiplier adds game-to-game VARIANCE: some draws are
        # faster/slower than expected. Mean = 1.0 (no bias), std = _POSS_CV.
        # All players in the same draw share the same pace multiplier,
        # creating positive correlation in counting stats.
        pace_multiplier = rng.normal(1.0, _POSS_CV, size=n_draws)  # (N,)
        pace_multiplier = np.clip(pace_multiplier, 0.85, 1.15)

        # ── Step 2: Draw game margin + game-state flags ──
        # Margin draw → blowout detection (|margin| > threshold) and OT
        # detection (|margin| ≈ 0). These create correlated minute shifts
        # across all players on a team, which the sim uses for game-script
        # correlation in counting stats.
        spread = self.proj.projected_spread  # negative = home favored
        margin_draws = rng.normal(-spread, _MARGIN_SIGMA, size=n_draws)  # (N,) home margin

        # Per-draw game-state flags
        is_blowout = np.abs(margin_draws) > _BLOWOUT_THRESHOLD  # (N,)
        # OT: close game draws. |margin| < threshold doesn't guarantee OT,
        # but small margins correlate with OT probability. Use P(OT | margin)
        # ≈ flat _OT_PROB for games that aren't blowouts. Blowouts can't be OT.
        is_ot = (~is_blowout) & (rng.random(size=n_draws) < _OT_PROB)

        # Blowout minute deltas: applied RELATIVE TO mean (not double-counting).
        # Engine minutes already include E[blowout adjustment]. Per draw:
        #   blowout draw → extra cut for starters, extra gain for bench
        #   non-blowout draw → reverse (starters get a bit back, bench gives back)
        # Such that E[delta] = 0 (the mean is unchanged).
        # delta_blowout = adjustment if blowout - E[adjustment]
        # E[adjustment] = P(blowout) × adjustment
        p_blow = _BASELINE_BLOWOUT_PROB
        blowout_deltas = {
            "starter":  -_BLOWOUT_STARTER_CUT,
            "rotation": +_BLOWOUT_ROTATION_GAIN,
            "spot":     +_BLOWOUT_SPOT_GAIN,
        }
        # Build per-player blowout delta array: (n_players,) conditional on blowout
        blowout_delta_per_player = np.array([
            blowout_deltas.get(p.role, 0.0) for p in self._players
        ], dtype=np.float64)
        # Mean-subtract: delta if blowout = raw - E, delta if not = 0 - E
        # E[adjustment] = p_blow × raw_delta
        expected_delta = p_blow * blowout_delta_per_player  # already in mean
        # Actual per-draw delta:
        #   blowout:     raw_delta - expected_delta
        #   non-blowout: 0 - expected_delta = -expected_delta
        blowout_min_shift = np.where(
            is_blowout[:, np.newaxis],
            blowout_delta_per_player[np.newaxis, :] - expected_delta[np.newaxis, :],
            -expected_delta[np.newaxis, :],
        )  # (N, P)

        # OT minute additions (per role, mean-subtracted)
        ot_delta_per_player = np.array([
            {"starter": _OT_MINUTES_STARTER,
             "rotation": _OT_MINUTES_ROTATION,
             "spot": _OT_MINUTES_SPOT}.get(p.role, 0.0)
            for p in self._players
        ], dtype=np.float64)
        expected_ot = _OT_PROB * ot_delta_per_player
        ot_min_shift = np.where(
            is_ot[:, np.newaxis],
            ot_delta_per_player[np.newaxis, :] - expected_ot[np.newaxis, :],
            -expected_ot[np.newaxis, :],
        )  # (N, P)

        # ── Step 2a: Team scoring draws ──
        # Draw game total → per-team points → per-team scoring efficiency.
        # Creates within-team PTS/AST/FG3M correlation beyond game-level pace.
        # NOT reconciliation — no hard constraint on Σ player stats.  Soft
        # multiplier on scoring rates only (~5% CV after dampening).
        market_total = self.proj.projected_total
        home_team = self.proj.home
        away_team = self.proj.away

        # Per-player scoring multiplier: (N, P), default 1.0 (no-op)
        team_scoring_mult = np.ones((n_draws, n_players), dtype=np.float64)

        if market_total >= 150:  # guard: need a sensible total to derive multipliers
            game_total_draws = rng.normal(market_total, _GAME_TOTAL_STD, size=n_draws)
            # Clip to prevent degenerate draws (±30% of market total)
            game_total_draws = np.clip(game_total_draws, market_total * 0.70,
                                       market_total * 1.30)

            # Per-team points: margin_draws = home - away margin
            home_pts_draw = (game_total_draws + margin_draws) / 2  # (N,)
            away_pts_draw = (game_total_draws - margin_draws) / 2  # (N,)

            # Expected per-team totals (from market lines)
            # spread convention: negative = home favored
            # expected home margin = -spread → home_exp = (total - spread) / 2
            home_exp = (market_total - spread) / 2
            away_exp = (market_total + spread) / 2

            # Raw per-team scoring multiplier
            home_raw_mult = home_pts_draw / max(home_exp, 70.0)  # (N,)
            away_raw_mult = away_pts_draw / max(away_exp, 70.0)  # (N,)

            # Dampen: 1 + damp × (raw - 1).  At damp=0.5, ~11% CV → ~5.5% CV.
            home_mult = 1.0 + _TEAM_SCORING_DAMP * (home_raw_mult - 1.0)
            away_mult = 1.0 + _TEAM_SCORING_DAMP * (away_raw_mult - 1.0)

            # Clip to prevent extreme distortion
            home_mult = np.clip(home_mult, 0.85, 1.15)
            away_mult = np.clip(away_mult, 0.85, 1.15)

            # Assign to per-player array by team
            for team, idxs in team_player_idx.items():
                idxs_arr = np.array(idxs)
                if team == home_team:
                    team_scoring_mult[:, idxs_arr] = home_mult[:, np.newaxis]
                elif team == away_team:
                    team_scoring_mult[:, idxs_arr] = away_mult[:, np.newaxis]

        # ── Step 2b: Sample availability per player per draw ──
        # For players with P(plays) < 1.0, some draws have them absent.
        # Void rules: own props are conditioned on playing (SimulationResult).
        # Redistribution: use measured absence impacts when available,
        # fall back to proportional 240 rescaling.
        p_plays_arr = np.array([p.p_plays for p in self._players], dtype=np.float64)
        availability = np.ones((n_draws, n_players), dtype=bool)
        for i in range(n_players):
            if p_plays_arr[i] < 1.0 - 1e-9:
                availability[:, i] = rng.random(size=n_draws) < p_plays_arr[i]

        # ── Step 2c: Build absence impact matrix (P × P) ──
        # impact_matrix[beneficiary_i, absent_j] = delta_min when j is absent.
        # Built from minutes model's measured with/without patterns (shrunk).
        player_names = [p.name for p in self._players]
        name_to_idx = {n: i for i, n in enumerate(player_names)}
        impact_matrix = np.zeros((n_players, n_players), dtype=np.float64)
        has_impacts = False
        if self.absence_impacts:
            for absent_name, beneficiaries in self.absence_impacts.items():
                j = name_to_idx.get(absent_name)
                if j is None:
                    continue
                for ben_name, delta in beneficiaries.items():
                    i = name_to_idx.get(ben_name)
                    if i is not None:
                        impact_matrix[i, j] = delta
                        has_impacts = True

        # ── Step 3: Draw player minutes ──
        # Two separate draws for two distinct mechanisms:
        #   (a) Pre-game availability (Step 2b above): player doesn't enter →
        #       prop VOIDED, excluded from own P(over), redistributed to teammates.
        #   (b) In-game early exit (below): player enters but leaves early
        #       (injury/ejection/foul-out) → bet has ACTION, stays in player's
        #       own distribution with reduced minutes.
        minutes = np.empty((n_draws, n_players), dtype=np.float64)

        for i, p in enumerate(self._players):
            # In-game early exit: two-component mixture (normal + short-game)
            exit_prob = _EARLY_EXIT_PROB.get(roles[i], 0.02)
            is_short = rng.random(size=n_draws) < exit_prob
            main_draw = rng.normal(min_means[i], max(min_stds[i], 1.0), size=n_draws)
            short_draw = rng.normal(_EARLY_EXIT_MEAN, _EARLY_EXIT_STD, size=n_draws)
            minutes[:, i] = np.where(is_short, short_draw, main_draw)

        # Zero out minutes for unavailable players
        minutes *= availability

        # ── Step 3a: Apply blowout + OT minute shifts (mean-subtracted) ──
        # These are correlated shifts: all starters lose/gain together per draw.
        # Mean-subtracted so E[minutes] is unchanged (engine mean already includes
        # expected blowout/OT adjustments). Only applied to available players.
        minutes += (blowout_min_shift + ot_min_shift) * availability

        # ── Step 3b: Apply measured absence redistribution ──
        # For each draw, add measured delta_min to beneficiaries of absent players.
        # Only applied to available players (absent players stay at 0).
        # Cap total boost per player at +8 min (same as point projection).
        if has_impacts:
            # absent_float: (N, P), 1.0 where player absent, 0.0 where present
            absent_float = (~availability).astype(np.float64)
            # delta_minutes: (N, P) = absent_float @ impact_matrix.T
            # delta_minutes[d, i] = sum_j(absent_float[d, j] * impact_matrix[i, j])
            delta_minutes = absent_float @ impact_matrix.T  # (N, P)
            # Cap per-player boost
            delta_minutes = np.clip(delta_minutes, -4.0, 8.0)
            # Only apply to available players
            delta_minutes *= availability
            minutes += delta_minutes

        # Floor at 0, cap at 48
        minutes = np.clip(minutes, 0.0, _MAX_MINUTES)

        # ── Step 4: Enforce team minutes constraint via proportional rescaling ──
        # Target: 240 in regulation, 245 in OT draws (5 extra min per OT period).
        # After absence redistribution, residual Σ gap is closed by proportional
        # rescaling. When absence impacts are provided, most redistribution is
        # already done — this step handles the remainder.
        # Per-draw target: (N,) array — 245 for OT draws, 240 otherwise
        team_min_target = np.where(is_ot, _TEAM_MINUTES + 5.0, _TEAM_MINUTES)  # (N,)
        for team, idxs in team_player_idx.items():
            idxs_arr = np.array(idxs)
            team_mins = minutes[:, idxs_arr]  # (N, n_team_players)
            team_sum = team_mins.sum(axis=1, keepdims=True)  # (N, 1)
            # Avoid divide by zero
            safe_sum = np.where(team_sum > 0, team_sum, 1.0)
            scale = team_min_target[:, np.newaxis] / safe_sum  # (N, 1)
            # Cap rescaling at ±30% to avoid extreme distortion
            scale = np.clip(scale, 0.70, 1.30)
            minutes[:, idxs_arr] = team_mins * scale

        minutes = np.clip(minutes, 0.0, _MAX_MINUTES)

        # ── Step 4b: Per-player engagement draw ──
        # "Good game / bad game" effect: a shared per-player multiplier that
        # creates within-player stat correlation (PTS+REB k≈1.20).
        # Each player gets one z-score per draw; stat-specific loadings
        # control how much each stat participates.
        # The engagement multiplier is: 1 + loading × _ENGAGEMENT_CV × z
        # Total variance is preserved by reducing rate_stds accordingly.
        engagement_z = rng.normal(0.0, 1.0, size=(n_draws, n_players))

        # Adjust rate_stds: remove variance explained by engagement.
        # new_std² = original_std² - rate² × (eng_cv × loading)²
        # This keeps total variance (rate_std + engagement) = original.
        loadings = np.array([_ENGAGEMENT_LOADINGS.get(si, 0.0)
                             for si in range(n_stats)])  # (S,)
        adjusted_rate_stds = np.empty_like(rate_stds)
        for si in range(n_stats):
            ld = loadings[si]
            if ld > 0:
                var_eng = rate_means[:, si] ** 2 * (_ENGAGEMENT_CV * ld) ** 2
                var_resid = np.maximum(rate_stds[:, si] ** 2 - var_eng,
                                       0.01 * rate_means[:, si] ** 2)
                adjusted_rate_stds[:, si] = np.sqrt(var_resid)
            else:
                adjusted_rate_stds[:, si] = rate_stds[:, si]

        # ── Step 5: Draw per-minute rates (independent given game state) ──
        # Continuous stats: Normal(μ, σ) rate draws.
        # Discrete stats: Gamma(α, β) rate draws — non-negative by construction,
        # and Gamma-Poisson compound = exact NegBin marginal for counts.
        # Gamma params: shape α = (μ/σ)², scale β = σ²/μ (matching mean & variance).
        rate_draws = np.empty((n_draws, n_players, n_stats), dtype=np.float64)

        for si in range(n_stats):
            any_discrete_this_stat = np.any(is_discrete[:, si])
            for pi in range(n_players):
                mu = rate_means[pi, si]
                sigma = max(adjusted_rate_stds[pi, si], 1e-6)
                if is_discrete[pi, si] and mu > 1e-6:
                    # Gamma(shape=α, scale=β): E=αβ=μ, Var=αβ²=σ²
                    alpha = (mu / sigma) ** 2
                    beta = sigma ** 2 / mu
                    rate_draws[:, pi, si] = rng.gamma(shape=alpha, scale=beta,
                                                       size=n_draws)
                else:
                    rate_draws[:, pi, si] = rng.normal(loc=mu, scale=sigma,
                                                        size=n_draws)

        # Apply per-player engagement multiplier (stat-specific loadings)
        for si in range(n_stats):
            ld = loadings[si]
            if ld > 0:
                eng_mult = 1.0 + _ENGAGEMENT_CV * ld * engagement_z  # (N, P)
                eng_mult = np.clip(eng_mult, 0.75, 1.25)
                rate_draws[:, :, si] *= eng_mult

        # Floor rates at 0 (continuous stats can go negative from Normal draws)
        rate_draws = np.maximum(rate_draws, 0.0)

        # Apply shared pace multiplier to all rates (game-level possessions)
        # pace_multiplier shape (N,) → (N, 1, 1) for broadcasting
        rate_draws *= pace_multiplier[:, np.newaxis, np.newaxis]

        # Apply per-team scoring multiplier to scoring-related rates.
        # PTS=0, FG3M=5 correlate with team scoring efficiency.
        # AST=2 removed (S4-2: assists driven by teammate conversion, not
        # team scoring efficiency; including AST over-correlated PTS+AST).
        # REB=1, STL=3, BLK=4 correlate with possessions (pace only).
        # team_scoring_mult shape (N, P) → (N, P, 1) for broadcasting
        for si in (0, 5):  # PTS, FG3M only
            rate_draws[:, :, si] *= team_scoring_mult

        # ── Step 6: Compute counting stats ──
        # Continuous stats: counting = minutes × rate (Normal draws).
        # Discrete stats (STL, BLK, FG3M, AST rotation): Poisson(λ = minutes × rate).
        # Poisson respects integer nature and avoids Normal-CDF artifacts on
        # low-count stats (KS 0.17-0.23 with Normal vs ~0.05 target).
        stat_draws = minutes[:, :, np.newaxis] * rate_draws  # (N, P, S) default

        # Overwrite discrete stats with Poisson draws
        for pi in range(n_players):
            for si in range(n_stats):
                if is_discrete[pi, si]:
                    # λ = minutes × rate (after pace/scoring/engagement adjustments)
                    lam = minutes[:, pi] * rate_draws[:, pi, si]  # (N,)
                    lam = np.maximum(lam, 0.0)
                    # Poisson draw; clip λ to avoid overflow (Poisson(λ>700) ≈ Normal)
                    lam_safe = np.minimum(lam, 500.0)
                    stat_draws[:, pi, si] = rng.poisson(lam=lam_safe).astype(np.float64)

        # PRA (index 6) = PTS (0) + REB (1) + AST (2) from the draw.
        # Always the direct sum — PRA rate draw served as correlated input
        # for engagement, but actual PRA is the component sum.
        stat_draws[:, :, 6] = stat_draws[:, :, 0] + stat_draws[:, :, 1] + stat_draws[:, :, 2]

        # ── Build result ──
        # Pass availability so query methods can condition on playing
        # (void rules: own props conditional on playing, SGP void-leg repricing)
        has_uncertain = np.any(p_plays_arr < 1.0 - 1e-9)
        return SimulationResult(
            player_names=[p.name for p in self._players],
            stat_names=stat_names,
            draws=stat_draws.astype(np.float32),
            team_labels=[p.team for p in self._players],
            availability=availability if has_uncertain else None,
        )


# ── Convenience ──────────────────────────────────────────────────────────────

def simulate_game(engine, home: str, away: str,
                  market_total: float = 0.0,
                  market_spread: float = 0.0,
                  n_draws: int = 10_000,
                  seed: int = 42) -> SimulationResult:
    """One-shot: project + simulate a game.

    Args:
        engine: loaded NBAMatchupEngine instance
        home, away: team abbreviations
        market_total, market_spread: from lines
        n_draws: number of MC draws
        seed: RNG seed

    Returns:
        SimulationResult with query methods
    """
    proj = engine.project_game(
        home, away,
        market_total=market_total,
        market_spread=market_spread,
    )
    sim = GameSimulator(proj)
    return sim.run(n_draws=n_draws, seed=seed)
