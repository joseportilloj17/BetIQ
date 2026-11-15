"""
nba_minutes_model.py — Predict NBA player minutes for prop projections.

Features:
  1. Rotation role (starter/rotation/spot) from historical MPG
  2. B2B rest detection and historical minutes reduction
  3. Foul rate → foul-out risk → minutes reduction
  4. Blowout script: starter-minutes cut by game margin entering Q4
     (measured from historical PLUS_MINUS and final margins)
  5. Teammate-absence redistribution (with/without splits)

Inputs:
  - Player game logs (from NBAMatchupEngine.player_logs)
  - Team game logs (from NBAMatchupEngine.team_logs)
  - Market spread (proxy for expected margin → blowout probability)

Outputs:
  - MinutesProjection per player: projected minutes, std, role,
    adjustment factors (b2b, foul_risk, blowout, absence)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import erf, exp, sqrt
from statistics import mean, stdev
from typing import Optional

import pandas as pd

logger = logging.getLogger("NBAMinutes")

# ── Constants ────────────────────────────────────────────────────────────────

# Role thresholds (MPG cutoffs)
_STARTER_MPG = 28.0
_ROTATION_MPG = 15.0

# B2B minutes reduction from NBA research: starters lose ~2 min on B2B
_B2B_STARTER_CUT = 2.0     # minutes
_B2B_ROTATION_CUT = 1.2

# Foul rate: 6 personal fouls = fouled out.  Players with high PF/min
# get pulled earlier in foul trouble.
_FOUL_LIMIT = 6
_HIGH_FOUL_RATE = 4.5   # PF per 36 min — top decile

# Blowout script: data-measured minutes delta in |margin|>=20 vs <=10
_BLOWOUT_MARGIN = 15.0        # threshold for P(blowout) calculation
_BLOWOUT_STARTER_CUT = 2.3    # starters LOSE this many min (measured: -2.3)
_BLOWOUT_ROTATION_CUT = 4.0   # rotation GAINS this many min (measured: +4.0)
_BLOWOUT_SPOT_CUT = 7.0       # spot players GAIN (measured: +7.2, N=9, shrunk)

# Baseline blowout rate from 2025-26 data (|player plus_minus| >= 15).
# This is already embedded in the season-average minutes; only the EXCESS
# probability of this specific game vs baseline should produce an adjustment.
_BASELINE_BLOWOUT_PROB = 0.264

# Short-game mixture component for starters (injury exit, ejection,
# foul-out, deep blowout benching).  Measured from 2025-26 data:
# 3.54% of starter player-games are < 20 min, with mean 14.8 and std 4.7.
_STARTER_SHORT_GAME_PROB = 0.035
_STARTER_SHORT_GAME_MEAN = 14.8
_STARTER_SHORT_GAME_STD = 4.7

# Optimal blend weights (model recency vs last-5 average), fitted walk-forward
_BLEND_MODEL_WEIGHT = {"starter": 0.6, "rotation": 0.3, "spot": 0.2}


@dataclass
class MinutesProjection:
    """Projected minutes for a single player in a specific game."""
    player_name: str = ""
    team: str = ""
    opponent: str = ""

    # Core projection
    base_minutes: float = 0.0       # season average MPG
    projected_minutes: float = 0.0  # after all adjustments
    minutes_std: float = 0.0        # game-to-game variability

    # Role
    role: str = "rotation"          # starter | rotation | spot
    games_played: int = 0

    # Adjustment factors (additive, in minutes)
    b2b_adj: float = 0.0            # negative on B2B
    foul_adj: float = 0.0           # negative for foul-prone players
    blowout_adj: float = 0.0        # negative when blowout expected
    absence_adj: float = 0.0        # positive when teammates out

    # Context
    is_b2b: bool = False
    is_rest_day: bool = False        # known rest day (DNP-rest)
    foul_rate_per36: float = 0.0
    blowout_prob: float = 0.0       # P(margin > 15) from spread
    absent_teammates: list = field(default_factory=list)
    minutes_cap: float = 0.0        # 0 = no cap; >0 = hard ceiling from restriction

    def p_over(self, line: float) -> float:
        """P(minutes > line) from mixture distribution.

        Starters use a two-component mixture:
          (1-p_short) × Normal(mu, sigma) + p_short × Normal(short_mu, short_sigma)
        Other roles use a plain Normal.
        """
        if self.is_rest_day or self.projected_minutes <= 0:
            return 0.0
        return 1.0 - self.cdf(line)

    def cdf(self, x: float) -> float:
        """CDF: P(minutes <= x) using mixture for starters."""
        if self.is_rest_day:
            return 1.0 if x >= 0 else 0.0
        sigma = max(self.minutes_std, 1.0)

        # Main component
        z = (x - self.projected_minutes) / sigma
        main_cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))

        if self.role == "starter":
            # Short-game mixture component
            z_short = (x - _STARTER_SHORT_GAME_MEAN) / _STARTER_SHORT_GAME_STD
            short_cdf = 0.5 * (1.0 + erf(z_short / sqrt(2.0)))
            p_s = _STARTER_SHORT_GAME_PROB
            return (1.0 - p_s) * main_cdf + p_s * short_cdf

        return main_cdf

    def quantile(self, p: float) -> float:
        """Inverse CDF (approximate via bisection)."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 48.0
        lo, hi = 0.0, 48.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2.0, 1)

    def distribution_summary(self) -> dict:
        """Return percentiles and key distribution stats."""
        return {
            "mean": round(self.projected_minutes, 1),
            "std": round(self.minutes_std, 1),
            "p10": self.quantile(0.10),
            "p25": self.quantile(0.25),
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
        }


class NBAMinutesModel:
    """
    Minutes model built from game-log data.  Operates on the raw DataFrames
    stored by NBAMatchupEngine (player_logs, team_logs).
    """

    def __init__(self):
        # Per-player computed stats
        self._player_stats: dict[int, dict] = {}   # pid → stats dict
        # Per-team stats (for blowout calibration)
        self._team_stats: dict[str, dict] = {}      # abbr → stats dict
        # Teammate-absence matrix: {(team, absent_pid): {present_pid: min_delta}}
        self._absence_impact: dict[tuple, dict] = {}
        # B2: Absence rate-impact (REMOVED — killed by ablation, +14.9% worse MAE)
        self._built = False

    def build(self, player_logs: dict[int, pd.DataFrame],
              team_logs: dict[str, pd.DataFrame],
              player_profiles: dict | None = None,
              prior_player_logs: dict[int, pd.DataFrame] | None = None):
        """Build the minutes model from game-log data.

        Args:
            player_logs: {player_id: DataFrame} from NBAMatchupEngine
            team_logs: {team_abbr: DataFrame} from NBAMatchupEngine
            player_profiles: optional {name: PlayerProfile} for name→pid lookup
            prior_player_logs: optional prior-season game logs for role blending
        """
        self._build_player_stats(player_logs, prior_player_logs)
        self._build_team_stats(team_logs)
        self._build_absence_matrix(player_logs)
        self._built = True
        logger.info("[MinutesModel] Built: %d players, %d teams",
                     len(self._player_stats), len(self._team_stats))

    # ── Player-level stats ───────────────────────────────────────────────────

    @staticmethod
    def _recency_weighted_mean(values, half_life: float = 20.0) -> float:
        """Exponentially-weighted mean, most-recent game = highest weight.

        Half-life of 20 games: a game 20 games ago gets half the weight of
        the most recent game.  This naturally adapts to rest patterns, role
        changes, and minutes-limit ramp-ups without explicit detection.
        """
        n = len(values)
        if n == 0:
            return 0.0
        decay = 0.5 ** (1.0 / half_life)
        total_w = 0.0
        total_v = 0.0
        for i, v in enumerate(values):
            w = decay ** (n - 1 - i)          # most recent → highest weight
            total_w += w
            total_v += w * v
        return total_v / total_w if total_w > 0 else 0.0

    # Number of current-season games before we stop blending prior role
    _EARLY_SEASON_GAMES = 10

    def _build_player_stats(self, player_logs: dict[int, pd.DataFrame],
                            prior_player_logs: dict[int, pd.DataFrame] | None = None):
        """Compute per-player minutes statistics from game logs.

        When prior_player_logs is provided and a player has fewer than
        _EARLY_SEASON_GAMES of current-season data, blends the current role
        with the prior-season role.  Also detects team changes.
        """
        # Build prior-season reference for role blending
        prior_info: dict[int, dict] = {}  # pid → {avg_min, role, team}
        if prior_player_logs:
            for pid, pdf in prior_player_logs.items():
                if len(pdf) < 10:
                    continue
                p_mins = pdf["MIN"].astype(float)
                p_avg = float(p_mins.mean())
                p_team = str(pdf.iloc[-1].get("MATCHUP", ""))[:3].strip()
                if p_avg >= _STARTER_MPG:
                    p_role = "starter"
                elif p_avg >= _ROTATION_MPG:
                    p_role = "rotation"
                else:
                    p_role = "spot"
                prior_info[pid] = {"avg_min": p_avg, "role": p_role, "team": p_team}

        for pid, df in player_logs.items():
            if len(df) < 3:
                continue

            mins = df["MIN"].astype(float)
            fouls = df["PF"].astype(float)
            name = df["PLAYER_NAME"].iloc[0] if "PLAYER_NAME" in df.columns else str(pid)

            # Team from most recent game MATCHUP
            last_mu = str(df.iloc[-1].get("MATCHUP", ""))
            team = last_mu[:3].strip()

            # Recency-weighted mean: adapts to rest patterns, role changes
            avg_min = self._recency_weighted_mean(mins.tolist())
            std_min = float(mins.std()) if len(mins) >= 3 else avg_min * 0.15

            # Role classification from current data
            if avg_min >= _STARTER_MPG:
                role = "starter"
            elif avg_min >= _ROTATION_MPG:
                role = "rotation"
            else:
                role = "spot"

            # ── Early-season role blending ────────────────────────────────
            # When we have very few games, blend with prior-season role to
            # avoid volatile classification from 2-3 games.
            n_games = len(df)
            changed_team = False
            prior_role = None

            if pid in prior_info:
                prior_role = prior_info[pid]["role"]
                prior_team = prior_info[pid]["team"]
                if prior_team != team:
                    changed_team = True

            if n_games < self._EARLY_SEASON_GAMES and prior_role and not changed_team:
                # Blend: current_weight = n_games / _EARLY_SEASON_GAMES
                w_cur = n_games / self._EARLY_SEASON_GAMES
                prior_avg = prior_info[pid]["avg_min"]
                blended_avg = w_cur * avg_min + (1.0 - w_cur) * prior_avg
                if blended_avg >= _STARTER_MPG:
                    role = "starter"
                elif blended_avg >= _ROTATION_MPG:
                    role = "rotation"
                else:
                    role = "spot"
                # Don't change avg_min — use current data for projection,
                # only blend for ROLE classification

            # Foul rate per 36 minutes
            total_min = float(mins.sum())
            total_pf = float(fouls.sum())
            foul_rate_36 = (total_pf / total_min * 36.0) if total_min > 0 else 0.0

            # B2B minutes delta (measured from this player's actual game log)
            b2b_delta = self._measure_b2b_delta(df)

            # Blowout minutes delta (games with large PLUS_MINUS)
            blowout_delta = self._measure_blowout_delta(df)

            self._player_stats[pid] = {
                "name": name,
                "team": team,
                "avg_min": avg_min,
                "std_min": std_min,
                "role": role,
                "games": n_games,
                "foul_rate_36": foul_rate_36,
                "b2b_delta": b2b_delta,
                "blowout_delta": blowout_delta,
                "last5_avg": float(mean(mins.tolist()[-5:])) if n_games >= 5 else avg_min,
                "changed_team": changed_team,
                "prior_role": prior_role,
            }

    def _measure_b2b_delta(self, df: pd.DataFrame) -> float:
        """Measure actual minutes reduction on B2B for this player.

        Returns negative value (minutes lost on B2B) or 0 if insufficient data.
        """
        if "GAME_DATE_DT" not in df.columns:
            return 0.0

        dates = df["GAME_DATE_DT"].values
        mins = df["MIN"].astype(float).values
        b2b_mins = []
        non_b2b_mins = []

        for i in range(1, len(dates)):
            delta_days = (dates[i] - dates[i - 1]) / pd.Timedelta("1D")
            if delta_days <= 1.0:
                b2b_mins.append(mins[i])
            else:
                non_b2b_mins.append(mins[i])

        if len(b2b_mins) < 3 or len(non_b2b_mins) < 5:
            return 0.0  # insufficient data — use role-based default

        return mean(b2b_mins) - mean(non_b2b_mins)

    def _measure_blowout_delta(self, df: pd.DataFrame) -> float:
        """Measure minutes reduction in blowout games for this player.

        Uses PLUS_MINUS as a proxy for game margin (imperfect but available
        per-player without additional API calls).
        """
        mins = df["MIN"].astype(float).values
        plus_minus = df["PLUS_MINUS"].astype(float).values

        blowout_mins = []
        close_mins = []
        for m, pm in zip(mins, plus_minus):
            if abs(pm) >= _BLOWOUT_MARGIN:
                blowout_mins.append(m)
            elif abs(pm) <= 10:
                close_mins.append(m)

        if len(blowout_mins) < 3 or len(close_mins) < 5:
            return 0.0

        return mean(blowout_mins) - mean(close_mins)

    # ── Team-level stats ─────────────────────────────────────────────────────

    def _build_team_stats(self, team_logs: dict[str, pd.DataFrame]):
        """Compute team-level blowout frequency for context."""
        for abbr, df in team_logs.items():
            if len(df) == 0:
                continue
            # Count games by margin bucket
            # Team game logs don't have a clean margin column, but we can
            # compute from PTS and opponent PTS (if available) or WL + PLUS_MINUS
            if "PLUS_MINUS" in df.columns:
                pm = df["PLUS_MINUS"].astype(float)
                blowout_wins = (pm >= _BLOWOUT_MARGIN).sum()
                blowout_losses = (pm <= -_BLOWOUT_MARGIN).sum()
                total = len(df)
                self._team_stats[abbr] = {
                    "blowout_win_pct": blowout_wins / max(total, 1),
                    "blowout_loss_pct": blowout_losses / max(total, 1),
                    "blowout_any_pct": (blowout_wins + blowout_losses) / max(total, 1),
                    "games": total,
                }

    # ── Teammate-absence matrix ──────────────────────────────────────────────

    # Shrinkage strength for absence deltas: N_without / (N_without + K)
    # K=5 → N=2 gives 0.29 weight, N=5 gives 0.50, N=20 gives 0.80
    _ABSENCE_SHRINKAGE_K = 5

    def _build_absence_matrix(self, player_logs: dict[int, pd.DataFrame]):
        """Build with/without splits for teammate absences.

        For each (team, player) pair, measure how minutes redistribute when
        that player is absent.  All pairs are kept (no delta-magnitude filter)
        and shrunk toward zero by sample size to avoid noisy N=2 estimates
        dominating projections.
        """
        # Build game_id → set of player_ids who played
        game_players: dict[str, set[int]] = {}
        player_team: dict[int, str] = {}  # pid → team
        player_games: dict[int, set[str]] = {}  # pid → set of game_ids

        for pid, df in player_logs.items():
            if len(df) < 5:
                continue
            last_mu = str(df.iloc[-1].get("MATCHUP", ""))
            team = last_mu[:3].strip()
            player_team[pid] = team

            game_ids = set()
            for _, row in df.iterrows():
                gid = str(row.get("Game_ID", ""))
                if not gid:
                    continue
                game_ids.add(gid)
                if gid not in game_players:
                    game_players[gid] = set()
                game_players[gid].add(pid)
            player_games[pid] = game_ids

        # Store per-player last-game-date for timestamp gating
        for pid, df in player_logs.items():
            if pid in self._player_stats and "GAME_DATE_DT" in df.columns:
                self._player_stats[pid]["last_game_date"] = df["GAME_DATE_DT"].iloc[-1]
                self._player_stats[pid]["total_team_games"] = len(
                    self._team_stats.get(self._player_stats[pid].get("team", ""), {})
                ) or len(df)

        # For each team, find rotation+ players and compute with/without
        K = self._ABSENCE_SHRINKAGE_K
        for pid, stats in self._player_stats.items():
            if stats["role"] == "spot" or stats["games"] < 10:
                continue
            team = stats["team"]
            my_games = player_games.get(pid, set())
            if len(my_games) < 10:
                continue

            teammates = [
                tp for tp, ts in self._player_stats.items()
                if ts["team"] == team and tp != pid and ts["games"] >= 10
                and ts["role"] != "spot"
            ]

            for tm_pid in teammates:
                tm_games = player_games.get(tm_pid, set())
                both = my_games & tm_games
                without = my_games - tm_games

                if len(both) < 5 or len(without) < 2:
                    continue

                df = player_logs[pid]
                gid_col = "Game_ID"
                if gid_col not in df.columns:
                    continue

                with_mask = df[gid_col].astype(str).isin(both)
                without_mask = df[gid_col].astype(str).isin(without)
                with_mins = df.loc[with_mask, "MIN"].astype(float)
                without_mins = df.loc[without_mask, "MIN"].astype(float)

                if len(with_mins) < 5 or len(without_mins) < 2:
                    continue

                n_wo = len(without_mins)
                raw_delta = float(without_mins.mean()) - float(with_mins.mean())
                # Shrink toward zero: weight = N / (N + K)
                shrunk_delta = raw_delta * n_wo / (n_wo + K)

                key = (team, tm_pid)
                if key not in self._absence_impact:
                    self._absence_impact[key] = {}
                tm_name = self._player_stats.get(tm_pid, {}).get("name", str(tm_pid))
                self._absence_impact[key][pid] = {
                    "delta_min": round(shrunk_delta, 1),
                    "raw_delta": round(raw_delta, 1),
                    "n_without": n_wo,
                    "player_name": stats["name"],
                    "absent_name": tm_name,
                }

        n_pairs = sum(len(v) for v in self._absence_impact.values())
        logger.info("[MinutesModel] Absence matrix: %d pairs across %d absent players "
                     "(shrinkage K=%d)", n_pairs, len(self._absence_impact), K)

    # ── Projection ───────────────────────────────────────────────────────────

    def project(self, player_id: int, opponent: str,
                is_b2b: bool = False,
                market_spread: float = 0.0,
                absent_pids: list[int] | None = None,
                minutes_cap: float = 0.0,
                is_rest_day: bool = False,
                disable_foul_adj: bool = False,
                disable_absence_adj: bool = False) -> MinutesProjection | None:
        """Project minutes for a player in a specific game context.

        Args:
            player_id: nba_api player ID
            opponent: opponent team abbreviation
            is_b2b: True if second game of back-to-back
            market_spread: from home perspective (negative = home favored).
                           Used to estimate blowout probability.
            absent_pids: list of teammate player IDs known to be absent
            minutes_cap: hard ceiling on minutes (e.g., 24.0 for minutes
                         restriction). 0 = no cap. Applied after all adjustments.
            is_rest_day: True if player is known to be resting (DNP-rest).
                         Sets projected minutes to 0.
        """
        stats = self._player_stats.get(player_id)
        if not stats:
            return None

        # Blend recency-weighted model with last-5 average (role-specific)
        w_model = _BLEND_MODEL_WEIGHT.get(stats["role"], 0.5)
        blended_min = w_model * stats["avg_min"] + (1.0 - w_model) * stats["last5_avg"]

        proj = MinutesProjection(
            player_name=stats["name"],
            team=stats["team"],
            opponent=opponent,
            base_minutes=blended_min,
            minutes_std=stats["std_min"],
            role=stats["role"],
            games_played=stats["games"],
            foul_rate_per36=stats["foul_rate_36"],
            is_b2b=is_b2b,
        )

        # ── B2B adjustment ───────────────────────────────────────────────
        if is_b2b:
            measured = stats["b2b_delta"]
            if measured != 0:
                proj.b2b_adj = measured  # player-specific (already negative)
            elif stats["role"] == "starter":
                proj.b2b_adj = -_B2B_STARTER_CUT
            else:
                proj.b2b_adj = -_B2B_ROTATION_CUT

        # ── Foul rate adjustment ─────────────────────────────────────────
        # High foul rates → earlier benching in foul trouble
        # Expected minutes lost ≈ P(foul trouble) × minutes_if_in_trouble
        if not disable_foul_adj and stats["foul_rate_36"] >= _HIGH_FOUL_RATE:
            # Approximate P(5+ fouls by Q3) from Poisson
            # rate per minute = foul_rate_36 / 36
            rate = stats["foul_rate_36"] / 36.0
            expected_fouls_36 = rate * 36.0
            # Rough: if expected fouls > 4.5/36min, lose ~1 min per game average
            excess = max(0, expected_fouls_36 - 4.0) / 2.0
            proj.foul_adj = -min(excess, 2.5)  # cap at -2.5 min

        # ── Blowout script ───────────────────────────────────────────────
        # P(|margin| > 15): both-sided — starters sit in both blowout
        # wins AND blowout losses.  Only adjust for EXCESS probability vs
        # the season baseline (baseline blowout effect is already in avg_min).
        proj.blowout_prob = _p_blowout(market_spread, threshold=_BLOWOUT_MARGIN)
        excess_blowout = proj.blowout_prob - _BASELINE_BLOWOUT_PROB

        if abs(excess_blowout) > 0.02:
            measured = stats["blowout_delta"]
            if measured != 0:
                # Player-specific delta (negative for starters, positive for bench)
                proj.blowout_adj = measured * excess_blowout
            elif stats["role"] == "starter":
                proj.blowout_adj = -_BLOWOUT_STARTER_CUT * excess_blowout
            elif stats["role"] == "spot":
                proj.blowout_adj = _BLOWOUT_SPOT_CUT * excess_blowout
            else:
                # Rotation players get more minutes in blowouts (garbage time)
                proj.blowout_adj = _BLOWOUT_ROTATION_CUT * excess_blowout

        # ── Teammate absence ─────────────────────────────────────────────
        # Empirical test showed recency baseline does NOT absorb long-term
        # absences (residual bias +0.74 to +1.17 across all durations).
        # No timestamp gate — redistribute for all known absences.
        if absent_pids and not disable_absence_adj:
            total_delta = 0.0
            for absent_pid in absent_pids:
                key = (stats["team"], absent_pid)
                impacts = self._absence_impact.get(key, {})
                if player_id in impacts:
                    total_delta += impacts[player_id]["delta_min"]
                    proj.absent_teammates.append(
                        impacts[player_id].get("absent_name", str(absent_pid))
                    )
            # Cap total absence boost at +8 min (can't exceed ~44 min)
            proj.absence_adj = min(total_delta, 8.0)

        # ── Rest day / minutes restriction ──────────────────────────────
        proj.is_rest_day = is_rest_day
        proj.minutes_cap = minutes_cap

        if is_rest_day:
            proj.projected_minutes = 0.0
            return proj

        # ── Final projection ─────────────────────────────────────────────
        proj.projected_minutes = max(0.0, (
            proj.base_minutes
            + proj.b2b_adj
            + proj.foul_adj
            + proj.blowout_adj
            + proj.absence_adj
        ))
        # Hard ceiling at 44 min (even high-usage stars rarely exceed this)
        proj.projected_minutes = min(proj.projected_minutes, 44.0)
        # Minutes restriction cap (e.g., returning from injury)
        if minutes_cap > 0:
            proj.projected_minutes = min(proj.projected_minutes, minutes_cap)
        # Floor at 0 for spot players, 8 for rotation, 20 for starters
        if stats["role"] == "starter":
            proj.projected_minutes = max(proj.projected_minutes, 20.0)
        elif stats["role"] == "rotation":
            proj.projected_minutes = max(proj.projected_minutes, 8.0)

        proj.projected_minutes = round(proj.projected_minutes, 1)
        return proj

    # ── Walk-forward validation ──────────────────────────────────────────────

    def validate_walkforward(self, player_logs: dict[int, pd.DataFrame],
                             min_history: int = 10) -> dict:
        """Expanding-window walk-forward validation of minutes projections.

        For each test game, uses all prior games as training data (mirrors
        production behavior where the model retrains nightly with all
        available data).  Reports MAE and bias by role and by margin bucket.
        """
        results = {"starter": [], "rotation": [], "spot": []}

        for pid, df in player_logs.items():
            if len(df) < min_history + 3:
                continue

            mins = df["MIN"].astype(float).values
            dates = (df["GAME_DATE_DT"].values
                     if "GAME_DATE_DT" in df.columns else [None] * len(df))
            plus_minus = (df["PLUS_MINUS"].astype(float).values
                          if "PLUS_MINUS" in df.columns else [0.0] * len(df))

            for i in range(min_history, len(df)):
                # Training: all games 0..i-1 (expanding window)
                train_mins = mins[:i]
                model_avg = self._recency_weighted_mean(train_mins.tolist())
                last5_avg = float(mean(train_mins[-5:])) if i >= 5 else model_avg

                if model_avg >= _STARTER_MPG:
                    role = "starter"
                elif model_avg >= _ROTATION_MPG:
                    role = "rotation"
                else:
                    role = "spot"

                w = _BLEND_MODEL_WEIGHT.get(role, 0.5)
                predicted = w * model_avg + (1.0 - w) * last5_avg

                # B2B adjustment
                is_b2b = False
                if dates[i] is not None and dates[i - 1] is not None:
                    dd = (dates[i] - dates[i - 1]) / pd.Timedelta("1D")
                    is_b2b = dd <= 1.0
                if is_b2b:
                    b2b_m, non_b2b_m = [], []
                    for j in range(1, i):
                        if dates[j] is not None and dates[j - 1] is not None:
                            dd2 = (dates[j] - dates[j - 1]) / pd.Timedelta("1D")
                            (b2b_m if dd2 <= 1.0 else non_b2b_m).append(mins[j])
                    if len(b2b_m) >= 3 and len(non_b2b_m) >= 5:
                        predicted += mean(b2b_m) - mean(non_b2b_m)
                    elif role == "starter":
                        predicted += -_B2B_STARTER_CUT
                    else:
                        predicted += -_B2B_ROTATION_CUT

                # Blowout: no pre-game spread available in walk-forward,
                # so excess probability = 0 and no adjustment fires.
                # Record actual margin for bucket-level diagnostics.
                abs_margin = abs(plus_minus[i])

                # Clamp
                if role == "starter":
                    predicted = max(predicted, 20.0)
                elif role == "rotation":
                    predicted = max(predicted, 8.0)
                predicted = min(predicted, 44.0)

                actual = mins[i]
                error = actual - predicted
                results[role].append({
                    "predicted": round(predicted, 1),
                    "actual": actual,
                    "error": error,
                    "abs_error": abs(error),
                    "abs_margin": abs_margin,
                    "is_b2b": is_b2b,
                })

        # Aggregate
        all_errors = []
        cal = {}
        for role, errs in results.items():
            if not errs:
                cal[role] = {"mae": 0, "n": 0, "mean_error": 0}
                continue
            maes = [e["abs_error"] for e in errs]
            biases = [e["error"] for e in errs]
            cal[role] = {
                "mae": round(mean(maes), 2),
                "mean_error": round(mean(biases), 2),
                "n": len(errs),
            }
            all_errors.extend(maes)

        # Margin-bucket bias for starters
        spread_buckets = {}
        for e in results.get("starter", []):
            am = e["abs_margin"]
            if am < 5:
                bkt = "close_0_5"
            elif am < 10:
                bkt = "medium_5_10"
            elif am < 15:
                bkt = "decisive_10_15"
            else:
                bkt = "blowout_15+"
            spread_buckets.setdefault(bkt, []).append(e)
        spread_cal = {}
        for bkt, errs in spread_buckets.items():
            spread_cal[bkt] = {
                "mae": round(mean(e["abs_error"] for e in errs), 2),
                "mean_error": round(mean(e["error"] for e in errs), 2),
                "n": len(errs),
            }

        return {
            "overall_mae": round(mean(all_errors), 2) if all_errors else 0,
            "n_predictions": len(all_errors),
            "calibration_by_role": cal,
            "starter_by_spread": spread_cal,
        }

    # ── Lookup helpers ───────────────────────────────────────────────────────

    def get_player_stats(self, player_id: int) -> dict | None:
        return self._player_stats.get(player_id)

    def get_team_blowout_pct(self, team: str) -> float:
        ts = self._team_stats.get(team, {})
        return ts.get("blowout_any_pct", 0.10)

    def get_absence_impact(self, team: str, absent_pid: int) -> dict:
        return self._absence_impact.get((team, absent_pid), {})

    # get_absence_rate_impact REMOVED — killed by ablation (+14.9% worse MAE)

    def pit_calibration(self, player_logs: dict[int, pd.DataFrame],
                        min_history: int = 10) -> dict:
        """Probability Integral Transform calibration check.

        For each test observation, computes CDF(actual) using the model's
        predicted Normal(mu, sigma).  If the distribution is calibrated,
        the PIT values should be uniform(0, 1) — i.e., each decile bucket
        should contain ~10% of observations.

        Returns {role: {bucket: observed_fraction, n}} and a KS-like stat.
        """
        pit_by_role: dict[str, list[float]] = {"starter": [], "rotation": [], "spot": []}

        for pid, df in player_logs.items():
            if len(df) < min_history + 3:
                continue

            mins = df["MIN"].astype(float).values

            for i in range(min_history, len(df)):
                train_mins = mins[:i]
                avg_min = self._recency_weighted_mean(train_mins.tolist())
                std_min = (float(pd.Series(train_mins).std())
                           if len(train_mins) >= 3 else avg_min * 0.15)
                std_min = max(std_min, 1.0)

                if avg_min >= _STARTER_MPG:
                    role = "starter"
                elif avg_min >= _ROTATION_MPG:
                    role = "rotation"
                else:
                    role = "spot"

                actual = mins[i]
                # CDF(actual) under Normal(avg_min, std_min)
                z = (actual - avg_min) / std_min
                pit_val = 0.5 * (1.0 + erf(z / sqrt(2.0)))
                pit_by_role[role].append(pit_val)

        result = {}
        for role, pit_vals in pit_by_role.items():
            if len(pit_vals) < 20:
                continue
            n = len(pit_vals)
            # Decile buckets
            buckets = {}
            for d in range(10):
                lo = d * 0.1
                hi = (d + 1) * 0.1
                count = sum(1 for v in pit_vals if lo <= v < hi)
                buckets[f"{int(lo*100):02d}-{int(hi*100):02d}"] = round(count / n, 3)
            # Max deviation from uniform
            max_dev = max(abs(frac - 0.10) for frac in buckets.values())
            result[role] = {"buckets": buckets, "n": n, "max_dev": round(max_dev, 3)}

        return result

    @staticmethod
    def redistribute_team_minutes(
        projections: dict[str, "MinutesProjection"],
        target: float = 240.0,
    ) -> dict[str, "MinutesProjection"]:
        """Rescale a team's player projections so total ≈ target (240 min).

        Starters are held fixed; the surplus/deficit is distributed
        proportionally among rotation and spot players.  This enforces
        the physical constraint that 5 players are on court at all times
        (5 × 48 = 240) and channels starter blowout losses to the bench.
        """
        if not projections:
            return projections

        starter_total = 0.0
        bench_total = 0.0
        bench_keys: list[str] = []

        for name, proj in projections.items():
            if proj.is_rest_day:
                continue
            if proj.role == "starter":
                starter_total += proj.projected_minutes
            else:
                bench_total += proj.projected_minutes
                bench_keys.append(name)

        current_total = starter_total + bench_total
        if current_total <= 0 or not bench_keys:
            return projections

        # How many minutes the bench should absorb
        bench_target = target - starter_total
        if bench_target <= 0 or bench_total <= 0:
            return projections

        scale = bench_target / bench_total
        # Don't scale wildly — cap at ±30% adjustment
        scale = max(0.70, min(scale, 1.30))

        for name in bench_keys:
            p = projections[name]
            p.projected_minutes = round(p.projected_minutes * scale, 1)
            # Respect floors
            if p.role == "rotation":
                p.projected_minutes = max(p.projected_minutes, 8.0)
            p.projected_minutes = min(p.projected_minutes, 44.0)

        return projections


# ── Minutes Restrictions Store ──────────────────────────────────────────────

import json
import os
from datetime import datetime, date

_RESTRICTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nba_minutes_restrictions.json"
)


def load_restrictions() -> dict[str, dict]:
    """Load active minutes restrictions.

    Returns {player_name: {minutes_cap, is_rest_day, expires, source, set_at}}.
    Expired entries are automatically pruned on load.
    """
    if not os.path.exists(_RESTRICTIONS_PATH):
        return {}
    try:
        with open(_RESTRICTIONS_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    today = date.today().isoformat()
    active = {}
    for name, entry in data.items():
        if entry.get("expires", "9999-12-31") >= today:
            active[name] = entry
    # Prune expired if any were removed
    if len(active) < len(data):
        _save_restrictions(active)
    return active


def _save_restrictions(data: dict):
    with open(_RESTRICTIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def set_restriction(player_name: str, *,
                    minutes_cap: float = 0.0,
                    is_rest_day: bool = False,
                    expires: str = "",
                    source: str = "manual") -> dict:
    """Set a minutes restriction for a player.

    Args:
        player_name: exact player name (as in nba_api)
        minutes_cap: hard ceiling on minutes (0 = no cap)
        is_rest_day: True → player DNP-rest for this date
        expires: ISO date string (YYYY-MM-DD) when restriction ends.
                 Omit or empty for today-only.
        source: who/what set this (e.g., "manual", "morning_brief")

    Returns the stored entry.
    """
    data = load_restrictions()
    entry = {
        "minutes_cap": minutes_cap,
        "is_rest_day": is_rest_day,
        "expires": expires or date.today().isoformat(),
        "source": source,
        "set_at": datetime.now().isoformat(),
    }
    data[player_name] = entry
    _save_restrictions(data)
    logger.info("[MinutesRestriction] Set %s: cap=%.0f rest=%s expires=%s source=%s",
                player_name, minutes_cap, is_rest_day, entry["expires"], source)
    return entry


def clear_restriction(player_name: str) -> bool:
    """Remove a restriction. Returns True if it existed."""
    data = load_restrictions()
    if player_name in data:
        del data[player_name]
        _save_restrictions(data)
        return True
    return False


def list_restrictions() -> dict[str, dict]:
    """List all active restrictions."""
    return load_restrictions()


# ── Injury Watcher ───────────────────────────────────────────────────────────

# ESPN team ID → nba_api abbreviation
# fmt: off
_ESPN_ID_TO_ABBR = {
    "1": "ATL", "2": "BOS", "17": "BKN", "30": "CHA",
    "4": "CHI", "5": "CLE", "6": "DAL", "7": "DEN",
    "8": "DET", "9": "GSW", "10": "HOU", "11": "IND",
    "12": "LAC", "13": "LAL", "29": "MEM", "14": "MIA",
    "15": "MIL", "16": "MIN", "3": "NOP", "18": "NYK",
    "25": "OKC", "19": "ORL", "20": "PHI", "21": "PHX",
    "22": "POR", "23": "SAC", "24": "SAS", "28": "TOR",
    "26": "UTA", "27": "WAS",
}
# fmt: on

_ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Statuses that mean the player won't play
_OUT_STATUSES = {"Out", "Suspended"}
_QUESTIONABLE_STATUSES = {"Day-To-Day", "Questionable", "Doubtful"}


def fetch_nba_injuries() -> dict[str, list[dict]]:
    """Fetch current NBA injury report from ESPN.

    Returns: {team_abbr: [{player_name, player_id, status, detail}, ...]}
    Only includes players who are Out or Day-to-Day/Questionable.
    """
    import requests

    try:
        resp = requests.get(_ESPN_INJURIES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("[NBA Injuries] ESPN fetch failed: %s", e)
        return {}

    result: dict[str, list[dict]] = {}
    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        # ESPN doesn't always give abbreviation in injuries endpoint;
        # we need to extract it from the team object
        team_id = str(team_entry.get("id", ""))
        abbr = _ESPN_ID_TO_ABBR.get(team_id, "")
        injuries = []

        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            status_type = inj.get("type", {}).get("description", "")
            status = inj.get("status", status_type).strip()

            if status in _OUT_STATUSES or status in _QUESTIONABLE_STATUSES:
                injuries.append({
                    "player_name": athlete.get("displayName", ""),
                    "player_id": athlete.get("id", ""),
                    "status": status,
                    "detail": inj.get("details", {}).get("detail", ""),
                    "comment": (inj.get("longComment", "") or "")[:200],
                })

        if injuries:
            result[abbr] = injuries

    logger.info("[NBA Injuries] %d teams with injuries, %d total players",
                 len(result), sum(len(v) for v in result.values()))
    return result


# ── Utilities ────────────────────────────────────────────────────────────────

def _p_blowout(spread: float, threshold: float = 15.0, sigma: float = 12.0) -> float:
    """P(|margin| > threshold) — both directions, since both teams' starters sit.

    Uses NBA score-margin σ ≈ 12.  Computes P(margin > T) + P(margin < -T)
    where margin ~ N(spread, σ²).
    """
    # P(margin > +T): team wins by more than T
    z_win = (spread - threshold) / sigma
    p_win = 0.5 * (1.0 + erf(z_win / sqrt(2.0)))
    # P(margin < -T): team loses by more than T
    z_loss = (-spread - threshold) / sigma
    p_loss = 0.5 * (1.0 + erf(z_loss / sqrt(2.0)))
    return p_win + p_loss
