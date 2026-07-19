"""
features.py — Feature engineering for BetIQ ATS / O-U models.

Reads historical.db and builds a clean feature matrix per sport with:
  1. Rolling 10-game and 3-game team stats (no lookahead bias)
  2. Schedule features: rest days, back-to-back flag, games in last 7 days
  3. Home/away and neutral-site flags
  4. ATS target  — covered_spread (1/0) where spread data exists
  5. O-U target  — total_result (1 = over, 0 = under) where total data exists

Public API
----------
  build_feature_matrix(sport)          -> pd.DataFrame (one row per game)
  FEATURE_COLS                         -> list[str] of model-ready numeric cols
  TARGET_ATS                           -> 'covered_spread'
  TARGET_OU                            -> 'total_result'

No lookahead guarantee
----------------------
  Every rolling/schedule feature for game G uses only games whose
  game_date < G.game_date for that team.  The .shift(1) before each
  rolling transform enforces this even when two games share a date.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

# ── Target column names ────────────────────────────────────────────────────────
TARGET_ATS = "covered_spread"    # 1 = home team covered, 0 = did not
TARGET_OU  = "total_result"      # 1 = over,              0 = under

# ── Feature column groups (all numeric, NaN allowed for insufficient history) ─
ROLLING_COLS = [
    "h_pts_10g", "h_pts_allowed_10g", "h_win_pct_10g", "h_margin_10g",
    "h_pts_3g",  "h_pts_allowed_3g",  "h_win_pct_3g",
    "a_pts_10g", "a_pts_allowed_10g", "a_win_pct_10g", "a_margin_10g",
    "a_pts_3g",  "a_pts_allowed_3g",  "a_win_pct_3g",
]
SCHEDULE_COLS = [
    "h_rest_days", "h_is_btb", "h_games_7d",
    "a_rest_days", "a_is_btb", "a_games_7d",
    "rest_advantage",          # h_rest_days - a_rest_days
    "back_to_back_road",       # 1 = away team on BTB (historically the worst schedule spot)
    "season_phase",            # 0=early third, 1=mid third, 2=late third of season
]
DIFFERENTIAL_COLS = [
    "pts_diff_10g",       # h_pts_10g      - a_pts_10g
    "def_diff_10g",       # a_pts_allowed_10g - h_pts_allowed_10g (+ve = home better defence)
    "margin_diff_10g",    # h_margin_10g   - a_margin_10g
    "win_pct_diff_10g",   # h_win_pct_10g  - a_win_pct_10g
    "form_diff_3g",       # h_pts_3g       - a_pts_3g (recent form)
    # New: momentum + strength-of-schedule
    "days_since_win_diff",  # h_days_since_win - a_days_since_win (+ve = home colder)
    "h_days_since_win",     # home team: days since last win (momentum)
    "a_days_since_win",     # away team: days since last win
    "h_opp_impl_prob_5g",   # home team: avg implied prob of last-5 opponents (SOS)
    "a_opp_impl_prob_5g",   # away team: avg implied prob of last-5 opponents
    "opp_strength_diff",    # h_opp_impl_prob_5g - a_opp_impl_prob_5g
    # Injury flags (from game_injury_flags table; populated by backfill_nba_injuries)
    "h_star_out",           # 1 = home team missing a star player (20+ MPG)
    "a_star_out",           # 1 = away team missing a star player
    "h_stars_missing",      # count of home star players absent
    "a_stars_missing",      # count of away star players absent
    "h_rotation_missing",   # count of home rotation players absent
    "a_rotation_missing",   # count of away rotation players absent
    "star_out_diff",        # a_star_out - h_star_out (+ve = away team hurt more)
]
GAME_COLS = [
    "neutral_site",       # 0 by default (no source provides this currently)
    "ot_so_game",         # 1 = game decided in OT/SO (NHL puck line: -1.5 always fails in OT/SO)
    # MLB pitcher features (NaN for non-MLB sports; imputed to mean at train time)
    "h_sp_era",           # home starting pitcher season ERA
    "a_sp_era",           # away starting pitcher season ERA
    "h_sp_whip",          # home starting pitcher season WHIP
    "a_sp_whip",          # away starting pitcher season WHIP
    "sp_era_diff",        # a_sp_era - h_sp_era (+ve = home has better pitcher)
    "sp_whip_diff",       # a_sp_whip - h_sp_whip
    "h_sp_k9",            # home starting pitcher K/9
    "a_sp_k9",            # away starting pitcher K/9
    "h_sp_hr9",           # home starting pitcher HR/9  (8.9% fill — excluded from MODEL_FEATURE_COLS)
    "a_sp_hr9",           # away starting pitcher HR/9  (8.9% fill — excluded from MODEL_FEATURE_COLS)
    "sp_k9_diff",         # h_sp_k9 - a_sp_k9 (+ve = home SP has more strikeouts)
    "sp_hr9_diff",        # a_sp_hr9 - h_sp_hr9  (8.5% fill — excluded from MODEL_FEATURE_COLS)
    "h_sp_bb9",           # home starting pitcher BB/9
    "a_sp_bb9",           # away starting pitcher BB/9
    "sp_kbb_diff",        # (h_sp_k9 - h_sp_bb9) - (a_sp_k9 - a_sp_bb9)  K-BB% differential
    "h_sp_xera",          # home starting pitcher xERA (Baseball Savant expected ERA)
    "a_sp_xera",          # away starting pitcher xERA (Baseball Savant expected ERA)
    "sp_xera_diff",       # a_sp_xera - h_sp_xera (+ve = home pitcher better, luck-stripped)
    "h_park_factor",      # home ballpark run factor (>1 = hitter-friendly)
    # Pi (Elo-like) team power ratings — margin-based, opponent-quality-adjusted
    # 19.8% fill — excluded from MODEL_FEATURE_COLS until Group E backfill
    "home_pi",            # home team's Pi rating (home strength; 0-centered)
    "away_pi",            # away team's Pi rating (away strength; 0-centered)
    "pi_diff",            # home_pi - away_pi (positive = home is stronger)
    "pi_confidence",      # min(home_games, away_games) / 30 — how stable the ratings are
    # Rolling pitcher ERA/WHIP — last 3 and 5 starts before this game (no lookahead)
    "h_sp_era_3",         # home SP rolling ERA over last 3 starts
    "a_sp_era_3",         # away SP rolling ERA over last 3 starts
    "h_sp_era_5",         # home SP rolling ERA over last 5 starts
    "a_sp_era_5",         # away SP rolling ERA over last 5 starts
    "h_sp_whip_3",        # home SP rolling WHIP over last 3 starts
    "a_sp_whip_3",        # away SP rolling WHIP over last 3 starts
    "h_sp_whip_5",        # home SP rolling WHIP over last 5 starts
    "a_sp_whip_5",        # away SP rolling WHIP over last 5 starts
    "sp_era3_diff",       # a_sp_era_3 - h_sp_era_3
    "sp_era5_diff",       # a_sp_era_5 - h_sp_era_5
    "sp_whip3_diff",      # a_sp_whip_3 - h_sp_whip_3
    "sp_whip5_diff",      # a_sp_whip_5 - h_sp_whip_5
    # Rolling HR/9 — last 3 and 5 starts (needs B2 backfill for fill rate)
    "h_sp_hr9_3",         # home SP rolling HR/9 over last 3 starts
    "a_sp_hr9_3",         # away SP rolling HR/9 over last 3 starts
    "h_sp_hr9_5",         # home SP rolling HR/9 over last 5 starts
    "a_sp_hr9_5",         # away SP rolling HR/9 over last 5 starts
    "sp_hr9_3_diff",      # a_sp_hr9_3 - h_sp_hr9_3
    "sp_hr9_5_diff",      # a_sp_hr9_5 - h_sp_hr9_5
    # MLB team batting stats (season-level, NaN for non-MLB)
    "h_ops",              # home team season OPS
    "a_ops",              # away team season OPS
    "ops_diff",           # h_ops - a_ops
    "h_slg",              # home team season SLG
    "a_slg",              # away team season SLG
    "h_k_pct",            # home team strikeout rate (K/PA)
    "a_k_pct",            # away team strikeout rate (K/PA)
    "h_bb_pct",           # home team walk rate (BB/PA)
    "a_bb_pct",           # away team walk rate (BB/PA)
    # Statcast batted-ball quality (per-team season-level, from Baseball Savant)
    "h_barrel_pct",       # home team barrel rate (exit velo + launch angle quality)
    "a_barrel_pct",       # away team barrel rate
    "barrel_diff",        # h_barrel_pct - a_barrel_pct
    "h_hard_hit_pct",     # home team hard-hit rate (EV95+)
    "a_hard_hit_pct",     # away team hard-hit rate
    "hard_hit_diff",      # h_hard_hit_pct - a_hard_hit_pct
    # Weather features — game-day temperature/wind at venue (Open-Meteo Archive)
    "temperature_f",      # game-time temperature (°F), dome=72
    "wind_speed_mph",     # game-time wind speed (mph), dome=0 — OBSERVED actuals
    "wind_speed_forecast_mph",  # 1-day-ahead forecast wind (Previous Runs API) — train/serve matched
    "wind_direction_deg", # wind direction (degrees), dome=NaN
    "is_dome",            # 1=retractable/dome, 0=outdoor
]
LINE_COLS = [
    "close_spread",           # home perspective, negative = home favoured
    "close_total",
    "close_ml_home",          # American odds
    "close_ml_away",
    # Creator-tier CLV delta features (require open_spread / open_ml_home)
    "spread_move",            # close_spread - open_spread (line movement direction)
    "impl_prob_home_open",    # market-implied home win prob at open
    "impl_prob_home_close",   # market-implied home win prob at close
    "clv_ml_delta",           # close_impl_prob - open_impl_prob (sharp money direction)
]

FEATURE_COLS: list[str] = (
    ROLLING_COLS + SCHEDULE_COLS + DIFFERENTIAL_COLS + GAME_COLS + LINE_COLS
)

# Columns included in X for modelling (drop line features to avoid leakage
# when predicting before lines are available, but keep them here for analysis)
# Also exclude features with <20% fill rate — they silently impute to noise.
# These are kept in GAME_COLS / FEATURE_COLS for future use after backfill.
_MODEL_EXCLUDE = frozenset({
    "h_sp_hr9", "a_sp_hr9", "sp_hr9_diff",   # 8.9% fill (pitcher_stats HR data sparse)
    "home_pi", "away_pi", "pi_diff",          # 19.8% fill (Pi backfill pending Group E)
    "pi_confidence",                           # derived from Pi — meaningless without it
    "h_sp_hr9_3", "a_sp_hr9_3", "sp_hr9_3_diff",  # rolling HR/9 — pending B2 backfill
    "h_sp_hr9_5", "a_sp_hr9_5", "sp_hr9_5_diff",  # rolling HR/9 — pending B2 backfill
    "temperature_f", "wind_speed_mph",             # weather — observed actuals (train/serve mismatch)
    "wind_direction_deg", "is_dome",               # weather — noise (validated OOS)
    # NOTE: wind_speed_forecast_mph is NOT excluded — it's the train/serve-matched feature
})
MODEL_FEATURE_COLS: list[str] = [
    c for c in (ROLLING_COLS + SCHEDULE_COLS + DIFFERENTIAL_COLS + GAME_COLS)
    if c not in _MODEL_EXCLUDE
]

# ── Shadow experiment feature lists (Group E) ─────────────────────────────────
# E1: WHIP-emphasis ATS — swap ERA rolling for WHIP rolling.
# Thesis: rolling WHIP discriminates ATS better than rolling ERA (A2 finding).
_WHIP_ATS_DROP = frozenset({
    "h_sp_era_3", "a_sp_era_3", "h_sp_era_5", "a_sp_era_5",
    "sp_era3_diff", "sp_era5_diff",
})
MODEL_FEATURE_COLS_WHIP_ATS: list[str] = [
    c for c in MODEL_FEATURE_COLS if c not in _WHIP_ATS_DROP
]

# E2: Pi-enhanced ATS — adds Pi ratings to the standard feature set.
MODEL_FEATURE_COLS_PI_ATS: list[str] = MODEL_FEATURE_COLS + [
    "home_pi", "away_pi", "pi_diff",
]

# E3: Totals with HR/9 — adds rolling HR/9 features for totals prediction.
# Thesis: totals model was blind to run-environment pitching; HR/9 may revive it.
MODEL_FEATURE_COLS_HR9_TOTALS: list[str] = MODEL_FEATURE_COLS + [
    "h_sp_hr9_3", "a_sp_hr9_3", "sp_hr9_3_diff",
    "h_sp_hr9_5", "a_sp_hr9_5", "sp_hr9_5_diff",
]

# E4: Totals with weather — adds temperature/wind for totals prediction.
# Thesis: wind/temp are ORTHOGONAL to existing features (genuinely independent
# of pitcher stats). The cleanest test of whether totals is feature-blind.
MODEL_FEATURE_COLS_WEATHER_TOTALS: list[str] = MODEL_FEATURE_COLS + [
    "temperature_f", "wind_speed_mph", "wind_direction_deg", "is_dome",
]

# E5: V1 Pruned — Top-20 features by importance (tail 54 features actively
# degrade AUC and increase variance). Plus sp_kbb_diff (K-BB% differential),
# a free derivation from existing pitcher stats that captures command quality.
# Learning curve analysis (2026-07-04) confirmed model is feature-limited,
# not underfit: AUC plateaus at 50% of data. Pruning from 74→21 features
# raised CV AUC by +0.005 and cut variance from ±0.035 to ±0.027.
MODEL_FEATURE_COLS_V1_PRUNED: list[str] = [
    # Top-20 by GBC feature importance (importance-ordered)
    "sp_era_diff",          # #1: starting pitcher ERA differential (imp=0.31)
    "ops_diff",             # #2: team OPS differential
    "opp_strength_diff",    # #3: opponent strength (SOS) differential
    "h_sp_era",             # #4: home SP season ERA
    "sp_whip_diff",         # #5: starting pitcher WHIP differential
    "sp_era5_diff",         # #6: SP rolling 5-game ERA differential
    "a_ops",                # #7: away team season OPS
    "a_sp_era",             # #8: away SP season ERA
    "h_sp_era_5",           # #9: home SP rolling 5-game ERA
    "h_bb_pct",             # #10: home team walk rate
    "a_opp_impl_prob_5g",   # #11: away SOS (last 5 opponents)
    "h_opp_impl_prob_5g",   # #12: home SOS (last 5 opponents)
    "h_sp_era_3",           # #13: home SP rolling 3-game ERA
    "a_bb_pct",             # #14: away team walk rate
    "a_sp_era_5",           # #15: away SP rolling 5-game ERA
    "a_sp_whip_3",          # #16: away SP rolling 3-game WHIP
    "a_margin_10g",         # #17: away team 10-game avg margin
    "margin_diff_10g",      # #18: margin differential (10-game)
    "def_diff_10g",         # #19: defensive differential (10-game)
    "a_slg",                # #20: away team season SLG
    # New: K-BB% differential (command quality — derived from existing data)
    "sp_kbb_diff",          # K-BB% differential: (h_k9-h_bb9) - (a_k9-a_bb9)
]

# ── MLB park run factors ───────────────────────────────────────────────────────
# 2026 season park factors computed from actual game data (games table).
# venue_avg_total / league_avg_total.  Updated 2026-06-13.
# >1.0 = more runs scored here than league average; <1.0 = pitcher-friendly.
MLB_PARK_FACTORS: dict[str, float] = {
    "Pittsburgh Pirates":     1.59,
    "Athletics":              1.51,
    "Baltimore Orioles":      1.40,
    "Minnesota Twins":        1.32,
    "Colorado Rockies":       1.17,
    "Toronto Blue Jays":      1.13,
    "New York Yankees":       1.12,
    "Washington Nationals":   1.12,
    "Detroit Tigers":         1.06,
    "Philadelphia Phillies":  1.04,
    "Tampa Bay Rays":         1.00,
    "Los Angeles Dodgers":    0.99,
    "New York Mets":          0.99,
    "St. Louis Cardinals":    0.97,
    "Houston Astros":         0.97,
    "Milwaukee Brewers":      0.93,
    "Chicago White Sox":      0.93,
    "San Diego Padres":       0.93,
    "Kansas City Royals":     0.92,
    "Cincinnati Reds":        0.88,
    "Atlanta Braves":         0.88,
    "Cleveland Guardians":    0.88,
    "San Francisco Giants":   0.88,
    "Los Angeles Angels":     0.86,
    "Boston Red Sox":         0.80,
    "Texas Rangers":          0.77,
    "Seattle Mariners":       0.76,
    "Chicago Cubs":           0.75,
    "Arizona Diamondbacks":   0.69,
    "Miami Marlins":          0.69,
    "Oakland Athletics":      1.51,  # alias for Athletics
}


def _compute_rolling_pitcher_stats(
    conn: sqlite3.Connection,
    n: int,
) -> pd.DataFrame:
    """
    For each row in pitcher_game_logs, compute rolling ERA and WHIP over the
    last *n* starts (shift(1) before rolling — zero lookahead).

    Returns a DataFrame keyed on (player_id, game_date) with:
        rolling_era_{n}  rolling_whip_{n}
    """
    try:
        df = pd.read_sql_query(
            "SELECT player_id, game_date, ip, er, hits, walks "
            "FROM pitcher_game_logs ORDER BY player_id, game_date",
            conn,
        )
    except Exception:
        return pd.DataFrame(columns=["player_id","game_date",f"rolling_era_{n}",f"rolling_whip_{n}"])

    if df.empty:
        return pd.DataFrame(columns=["player_id","game_date",f"rolling_era_{n}",f"rolling_whip_{n}"])

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["player_id", "game_date"]).copy()

    # Vectorized rolling — avoids groupby().apply() which is 100-180× slower
    # under GIL contention from concurrent HTTP threads.
    grp = df.groupby("player_id")
    # shift(1) within each group → zero lookahead
    ip_shifted  = grp["ip"].shift(1)
    er_shifted  = grp["er"].shift(1)
    hw_shifted  = (df["hits"] + df["walks"]).groupby(df["player_id"]).shift(1)

    ip_roll  = ip_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
    er_roll  = er_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
    hw_roll  = hw_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)

    df[f"rolling_era_{n}"]  = (er_roll / ip_roll * 9).where(ip_roll > 0)
    df[f"rolling_whip_{n}"] = (hw_roll / ip_roll).where(ip_roll > 0)

    return df[["player_id", "game_date", f"rolling_era_{n}", f"rolling_whip_{n}"]].copy()


def _compute_rolling_pitcher_hr9(
    conn: sqlite3.Connection,
    n: int,
) -> pd.DataFrame:
    """
    Rolling HR/9 over the last *n* starts (shift(1) → zero lookahead).
    HR/9 = (home_runs / ip) × 9.

    Returns DataFrame keyed on (player_id, game_date) with:
        rolling_hr9_{n}
    """
    col = f"rolling_hr9_{n}"
    try:
        df = pd.read_sql_query(
            "SELECT player_id, game_date, ip, home_runs "
            "FROM pitcher_game_logs ORDER BY player_id, game_date",
            conn,
        )
    except Exception:
        return pd.DataFrame(columns=["player_id", "game_date", col])

    if df.empty or "home_runs" not in df.columns:
        return pd.DataFrame(columns=["player_id", "game_date", col])

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["player_id", "game_date"]).copy()

    grp = df.groupby("player_id")
    ip_shifted = grp["ip"].shift(1)
    hr_shifted = grp["home_runs"].shift(1)

    ip_roll = ip_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
    hr_roll = hr_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)

    df[col] = (hr_roll / ip_roll * 9).where(ip_roll > 0)

    return df[["player_id", "game_date", col]].copy()


def _load_pitcher_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load the pitcher_stats table.  Returns an empty DataFrame with correct
    columns if the table doesn't exist yet (pre-migration).
    """
    try:
        df = pd.read_sql_query(
            "SELECT pitcher_name, season, era, whip, k9, bb9, fip, hr9, xera FROM pitcher_stats",
            conn,
        )
        df["season"] = df["season"].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["pitcher_name", "season", "era", "whip", "k9", "bb9", "fip", "hr9", "xera"])


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _connect() -> sqlite3.Connection:
    return sqlite3.connect(os.path.abspath(DB_PATH))


def _load_game_history(conn: sqlite3.Connection, sport: str) -> pd.DataFrame:
    """
    Load all team-game rows with dates for a sport.
    Returns: game_id, team, is_home, game_date (datetime), score, opp_score, result,
             opp_ml (opponent's closing moneyline for SOS feature)
    """
    query = """
        SELECT
            ts.game_id,
            ts.team,
            ts.is_home,
            ts.score,
            ts.opp_score,
            ts.result,
            g.game_date,
            -- Opponent's closing ML: if we're home, opponent is away side
            CASE WHEN ts.is_home = 1
                 THEN bl.close_ml_away
                 ELSE bl.close_ml_home
            END AS opp_ml
        FROM team_stats ts
        JOIN games g ON ts.game_id = g.game_id
        LEFT JOIN betting_lines bl ON ts.game_id = bl.game_id
        WHERE ts.sport = ?
          AND ts.game_id IS NOT NULL
          AND g.game_date IS NOT NULL
          AND g.status = 'Final'
        ORDER BY g.game_date, ts.game_id
    """
    df = pd.read_sql_query(query, conn, params=[sport])
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["win"]       = (df["result"] == "W").astype(float)
    df["margin"]    = pd.to_numeric(df["score"],     errors="coerce") \
                    - pd.to_numeric(df["opp_score"], errors="coerce")
    df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
    df["opp_score"] = pd.to_numeric(df["opp_score"], errors="coerce")
    df["opp_ml"]    = pd.to_numeric(df["opp_ml"],    errors="coerce")
    return df


def _ml_to_impl_prob(ml: float) -> float:
    """American ML → implied probability (no vig removal)."""
    if pd.isna(ml) or ml == 0:
        return np.nan
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)


def _rolling_stats(history: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute rolling n-game stats per team using only PRIOR games.

    Key: .shift(1) before .rolling(n) excludes the current game row from
    every window — zero lookahead even within a single game_date.

    Returns history with added columns:
        pts_{n}g, pts_allowed_{n}g, win_pct_{n}g, margin_{n}g
        opp_impl_prob_{n}g  (avg implied prob of last n opponents — SOS proxy)
    """
    df = history.sort_values(["team", "game_date", "game_id"]).copy()

    def _roll(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=3).mean()

    grp = df.groupby("team", group_keys=False)

    df[f"pts_{n}g"]         = grp["score"]    .transform(lambda s: _roll(s, n))
    df[f"pts_allowed_{n}g"] = grp["opp_score"].transform(lambda s: _roll(s, n))
    df[f"win_pct_{n}g"]     = grp["win"]      .transform(lambda s: _roll(s, n))
    df[f"margin_{n}g"]      = grp["margin"]   .transform(lambda s: _roll(s, n))

    # Opponent strength: average implied probability of last n opponents.
    # Requires opp_ml from _load_game_history; NaN-safe (min_periods=3).
    if "opp_ml" in df.columns:
        df["opp_impl_prob"] = df["opp_ml"].apply(_ml_to_impl_prob)
        df[f"opp_impl_prob_{n}g"] = grp["opp_impl_prob"].transform(
            lambda s: _roll(s, n)
        )

    return df


def _schedule_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Add rest_days, is_btb, and games_7d to each team-game row.
    All computed using only games BEFORE the current row's date.
    """
    df = history.sort_values(["team", "game_date", "game_id"]).copy()

    # Rest days: days since previous game for this team
    df["prev_date"] = df.groupby("team")["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - df["prev_date"]).dt.days.fillna(21).clip(upper=30)
    df["is_btb"]    = (df["rest_days"] <= 1).astype(int)

    # Games in last 7 days BEFORE this game (per team)
    # Use time-indexed rolling within each team group
    games_7d_list: list[int] = []
    for team, grp in df.groupby("team", sort=False):
        grp_sorted = grp.sort_values("game_date").copy()
        grp_sorted = grp_sorted.set_index("game_date")
        # closed='left' excludes the current game; '7D' window covers 7 days prior
        counts = (
            grp_sorted["score"]
            .notna()
            .astype(int)
            .rolling("7D", closed="left")
            .sum()
            .values
        )
        # Re-attach in original order
        grp_sorted["games_7d"] = counts.astype(int)
        for idx in grp.index:
            orig_date = df.at[idx, "game_date"]
            val = grp_sorted.loc[orig_date, "games_7d"]
            # If multiple games on same date, .loc may return Series
            games_7d_list.append((idx, int(val.iloc[0]) if hasattr(val, "iloc") else int(val)))

    games_7d_series = pd.Series(
        {idx: v for idx, v in games_7d_list}, name="games_7d"
    )
    df["games_7d"] = games_7d_series.reindex(df.index).fillna(0).astype(int)

    # Days since last win — momentum signal.
    # For each game, looks back at prior games only (shift(1) on win dates).
    # A team on a long losing streak has high days_since_last_win.
    def _last_win_days(grp: pd.DataFrame) -> pd.Series:
        grp = grp.sort_values("game_date")
        # Mask: game_date where result was a win, NaT otherwise
        win_dates = grp["game_date"].where(grp["win"] == 1, other=pd.NaT)
        # Shift by 1 so current game's result is excluded, then forward-fill
        last_win = win_dates.shift(1).ffill()
        days = (grp["game_date"] - last_win).dt.days
        return days

    df["days_since_last_win"] = (
        df.groupby("team", group_keys=False)
        .apply(_last_win_days)
        .reset_index(level=0, drop=True)
        .reindex(df.index)
        .clip(upper=90)   # cap at 90 days to bound outliers (off-season gaps)
    )

    return df


def _compute_targets(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ATS and O-U target variables from scores and closing lines.

    covered_spread (home perspective)
    ----------------------------------
      close_spread = home team's spread (e.g. -3.5 means home -3.5 pts).
      Home covers when: (home_score - away_score) + close_spread > 0
      Example: close_spread=-3.5, margin=+7  → 7 + (-3.5) = 3.5 > 0 → covered
               close_spread=+3.5, margin=-1  → -1 + 3.5   = 2.5 > 0 → covered (dog covered)
      Push (== 0): set to NaN and exclude from training.

      # CLV_READY: open_spread available once TheOddsAPI paid plan active
      # — at that point also compute covered_open_spread for line value analysis

    total_result
    ------------
      1 = over  (actual total > close_total)
      0 = under (actual total < close_total)
      Push: NaN
    """
    df = games.copy()

    margin     = pd.to_numeric(df["home_score"], errors="coerce") \
               - pd.to_numeric(df["away_score"], errors="coerce")
    close_sprd = pd.to_numeric(df["close_spread"], errors="coerce")
    close_tot  = pd.to_numeric(df["close_total"],  errors="coerce")
    total_pts  = pd.to_numeric(df["home_score"], errors="coerce") \
               + pd.to_numeric(df["away_score"], errors="coerce")

    # ATS
    ats_val = margin + close_sprd
    covered = np.where(ats_val > 0, 1.0,
              np.where(ats_val < 0, 0.0, np.nan))
    df[TARGET_ATS] = np.where(
        close_sprd.isna() | margin.isna(),
        np.nan,
        covered,
    )

    # Note: MLB run line is always ±1.5, so ats_val can never equal 0 — there are
    # no pushes on the MLB run line. The standard formula above handles all cases.

    # O-U
    ou_val = total_pts - close_tot
    ou     = np.where(ou_val > 0, 1.0,
             np.where(ou_val < 0, 0.0, np.nan))
    df[TARGET_OU] = np.where(
        close_tot.isna() | total_pts.isna(),
        np.nan,
        ou,
    )

    return df


def _implied_prob(american_odds: float | None) -> float | None:
    """American odds → implied probability (vig included)."""
    if american_odds is None or np.isnan(american_odds):
        return None
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    sport: str,
    min_prior_games: int = 3,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """
    Build a clean feature matrix for *sport* from historical.db.

    Returns one row per game with:
      • Rolling 10-game and 3-game stats for home and away teams
      • Schedule features (rest, B2B, workload)
      • Differential features (relative team strength)
      • Closing line features where available
      • ATS and O-U target variables

    Parameters
    ----------
    sport           : e.g. 'NFL', 'NBA', 'MLB', 'EPL', 'UCL', …
    min_prior_games : minimum games a team must have played before a row
                      is included. Rows with fewer prior games have NaN
                      rolling features but are still returned — the caller
                      decides whether to drop them.
    conn            : optional existing sqlite3 connection (for testing)

    No lookahead guarantee
    ----------------------
    Every feature value for game G is derived exclusively from games
    whose game_date < G.game_date for the same team.
    """
    close_conn = conn is None
    if conn is None:
        conn = _connect()

    try:
        log.info("build_feature_matrix: sport=%s", sport)

        # ── 1. Load game history ───────────────────────────────────────────
        history = _load_game_history(conn, sport)
        if history.empty:
            log.warning("No team_stats found for sport=%s", sport)
            return pd.DataFrame()

        log.info("  Loaded %d team-game rows for %d unique teams",
                 len(history), history["team"].nunique())

        # ── 2. Rolling features (10-game, 5-game, and 3-game windows) ───────
        h10 = _rolling_stats(history, 10)
        h5  = _rolling_stats(history, 5)    # needed for opp_impl_prob_5g (SOS)
        h3  = _rolling_stats(history, 3)

        # Merge 3-game pts/pts_allowed/win_pct onto 10-game frame
        merge_cols = ["game_id", "team", "pts_3g", "pts_allowed_3g", "win_pct_3g"]
        h10 = h10.merge(
            h3[merge_cols].rename(columns={
                "pts_3g":         "pts_3g",
                "pts_allowed_3g": "pts_allowed_3g",
                "win_pct_3g":     "win_pct_3g",
            }),
            on=["game_id", "team"],
            how="left",
        )

        # Merge opp_impl_prob_5g (SOS proxy) from 5-game window
        if "opp_impl_prob_5g" in h5.columns:
            h10 = h10.merge(
                h5[["game_id", "team", "opp_impl_prob_5g"]],
                on=["game_id", "team"],
                how="left",
            )

        # ── 3. Schedule features ───────────────────────────────────────────
        h10 = _schedule_features(h10)

        # ── 4. Load game-level data with betting lines + injury flags ─────
        games_query = """
            SELECT
                g.game_id, g.sport, g.season, g.game_date,
                g.home_team, g.away_team, g.home_score, g.away_score,
                g.status,
                bl.open_spread,  bl.open_ml_home,  bl.open_ml_away,
                bl.close_spread, bl.close_total,
                bl.close_ml_home, bl.close_ml_away,
                COALESCE(bl.ot_so_game,            0) AS ot_so_game,
                COALESCE(inj.home_star_out,        0) AS home_star_out,
                COALESCE(inj.away_star_out,        0) AS away_star_out,
                COALESCE(inj.home_stars_missing,   0) AS home_stars_missing,
                COALESCE(inj.away_stars_missing,   0) AS away_stars_missing,
                COALESCE(inj.home_rotation_missing,0) AS home_rotation_missing,
                COALESCE(inj.away_rotation_missing,0) AS away_rotation_missing
            FROM games g
            LEFT JOIN betting_lines bl ON g.game_id = bl.game_id
            LEFT JOIN game_injury_flags inj ON g.game_id = inj.game_id
            WHERE g.sport = ?
              AND g.game_date IS NOT NULL
              AND g.status = 'Final'
            ORDER BY g.game_date
        """
        games = pd.read_sql_query(games_query, conn, params=[sport])
        games["game_date"] = pd.to_datetime(games["game_date"])

        log.info("  Loaded %d games", len(games))

        # ── 5. Join home and away rolling/schedule features ────────────────
        def _side_cols(side: str) -> dict[str, str]:
            """Rename feature columns with h_ or a_ prefix."""
            base = {
                "pts_10g":              f"{side}_pts_10g",
                "pts_allowed_10g":      f"{side}_pts_allowed_10g",
                "win_pct_10g":          f"{side}_win_pct_10g",
                "margin_10g":           f"{side}_margin_10g",
                "pts_3g":               f"{side}_pts_3g",
                "pts_allowed_3g":       f"{side}_pts_allowed_3g",
                "win_pct_3g":           f"{side}_win_pct_3g",
                "rest_days":            f"{side}_rest_days",
                "is_btb":               f"{side}_is_btb",
                "games_7d":             f"{side}_games_7d",
                "days_since_last_win":  f"{side}_days_since_win",
                "opp_impl_prob_5g":     f"{side}_opp_impl_prob_5g",
            }
            return base

        # Always-present cols + conditionally-present new cols
        feat_cols_base = [
            "game_id", "team",
            "pts_10g", "pts_allowed_10g", "win_pct_10g", "margin_10g",
            "pts_3g", "pts_allowed_3g", "win_pct_3g",
            "rest_days", "is_btb", "games_7d",
        ]
        for optional_col in ("days_since_last_win", "opp_impl_prob_5g"):
            if optional_col in h10.columns:
                feat_cols_base.append(optional_col)

        home_feats = (
            h10[feat_cols_base]
            .rename(columns={"team": "home_team", **_side_cols("h")})
        )
        away_feats = (
            h10[feat_cols_base]
            .rename(columns={"team": "away_team", **_side_cols("a")})
        )

        matrix = (
            games
            .merge(home_feats, on=["game_id", "home_team"], how="left")
            .merge(away_feats, on=["game_id", "away_team"], how="left")
        )

        # ── 6. Differential / composite features ──────────────────────────
        matrix["pts_diff_10g"]    = matrix["h_pts_10g"]       - matrix["a_pts_10g"]
        matrix["def_diff_10g"]    = matrix["a_pts_allowed_10g"] - matrix["h_pts_allowed_10g"]
        matrix["margin_diff_10g"] = matrix["h_margin_10g"]    - matrix["a_margin_10g"]
        matrix["win_pct_diff_10g"]= matrix["h_win_pct_10g"]   - matrix["a_win_pct_10g"]
        matrix["form_diff_3g"]    = matrix["h_pts_3g"]        - matrix["a_pts_3g"]
        matrix["rest_advantage"]  = matrix["h_rest_days"]     - matrix["a_rest_days"]

        # back_to_back_road: away team is playing on no rest (worst schedule spot).
        # a_is_btb is already computed; this column makes the signal explicit for
        # the model and avoids it being drowned out by the general a_is_btb weight.
        if "a_is_btb" in matrix.columns:
            matrix["back_to_back_road"] = matrix["a_is_btb"].fillna(0).astype(int)
        else:
            matrix["back_to_back_road"] = 0

        # season_phase: 0=early third, 1=mid third, 2=late third of season.
        # Computed per (sport, season) group by ranking game_date as a percentile
        # within the season — no hardcoded dates needed, works for any sport/year.
        if "season" in matrix.columns and not matrix.empty:
            def _phase(grp: pd.DataFrame) -> pd.Series:
                pct = grp["game_date"].rank(pct=True)
                return pd.cut(pct, bins=[0, 1/3, 2/3, 1.0],
                              labels=[0, 1, 2], include_lowest=True).astype(float)
            matrix["season_phase"] = (
                matrix.groupby("season", group_keys=False)
                .apply(_phase)
                .reindex(matrix.index)
            )
        else:
            matrix["season_phase"] = np.nan

        # New features (present only if backing columns were computed)
        if "h_days_since_win" in matrix.columns and "a_days_since_win" in matrix.columns:
            # Positive = home team is on a longer cold streak than away
            matrix["days_since_win_diff"] = (
                matrix["h_days_since_win"] - matrix["a_days_since_win"]
            )
        else:
            matrix["days_since_win_diff"] = np.nan

        if "h_opp_impl_prob_5g" in matrix.columns and "a_opp_impl_prob_5g" in matrix.columns:
            # Positive = home team has faced tougher opponents recently
            matrix["opp_strength_diff"] = (
                matrix["h_opp_impl_prob_5g"] - matrix["a_opp_impl_prob_5g"]
            )
        else:
            matrix["opp_strength_diff"] = np.nan

        # ── 6b-clv. Line-movement delta features ──────────────────────────
        # spread_move: how much the line moved from open to close.
        #   Negative = line moved toward the home team (public/sharp money on home).
        #   Requires open_spread AND close_spread to both be present.
        if "open_spread" in matrix.columns and "close_spread" in matrix.columns:
            matrix["spread_move"] = (
                pd.to_numeric(matrix["close_spread"], errors="coerce") -
                pd.to_numeric(matrix["open_spread"],  errors="coerce")
            )
        else:
            matrix["spread_move"] = np.nan

        # implied_prob_home_open / close: market implied probability of home win.
        # Derived from American ML odds stored as floats.
        def _imp_prob(american: pd.Series) -> pd.Series:
            """American odds → implied probability (handles +/-)."""
            a = pd.to_numeric(american, errors="coerce")
            return pd.Series(
                np.where(
                    a > 0, 100 / (a + 100),
                    np.where(a < 0, -a / (-a + 100), np.nan)
                ),
                index=american.index,
            )

        if "open_ml_home" in matrix.columns and "close_ml_home" in matrix.columns:
            matrix["impl_prob_home_open"]  = _imp_prob(matrix["open_ml_home"])
            matrix["impl_prob_home_close"] = _imp_prob(matrix["close_ml_home"])
            # clv_ml_delta: how much the market moved (positive = home became more favored)
            matrix["clv_ml_delta"] = (
                matrix["impl_prob_home_close"] - matrix["impl_prob_home_open"]
            )
        else:
            for col in ["impl_prob_home_open", "impl_prob_home_close", "clv_ml_delta"]:
                matrix[col] = np.nan

        # ── 6b. Injury differential features ──────────────────────────────
        # Rename raw injury columns to h_/a_ convention, then compute diff.
        # COALESCE in SQL ensures 0 when game_injury_flags has no row.
        if "home_star_out" in matrix.columns:
            matrix["h_star_out"]         = matrix["home_star_out"].astype(float)
            matrix["a_star_out"]         = matrix["away_star_out"].astype(float)
            matrix["h_stars_missing"]    = matrix["home_stars_missing"].astype(float)
            matrix["a_stars_missing"]    = matrix["away_stars_missing"].astype(float)
            matrix["h_rotation_missing"] = matrix["home_rotation_missing"].astype(float)
            matrix["a_rotation_missing"] = matrix["away_rotation_missing"].astype(float)
            # Positive = away team is more injured (home team has the edge)
            matrix["star_out_diff"]      = matrix["a_star_out"] - matrix["h_star_out"]
        else:
            for col in ["h_star_out", "a_star_out", "h_stars_missing",
                        "a_stars_missing", "h_rotation_missing",
                        "a_rotation_missing", "star_out_diff"]:
                matrix[col] = np.nan

        # ── 7. Game flags ──────────────────────────────────────────────────
        matrix["neutral_site"] = 0   # no source provides this; extend later
        # ot_so_game: 1 for NHL OT/SO games; 0 for all other sports/games.
        # Already fetched from betting_lines in games_query; ensure numeric.
        if "ot_so_game" in matrix.columns:
            matrix["ot_so_game"] = pd.to_numeric(matrix["ot_so_game"], errors="coerce").fillna(0).astype(int)
        else:
            matrix["ot_so_game"] = 0

        # ── 7b. MLB pitcher + park features ───────────────────────────────
        # Default NaN for all sports; populated below for MLB only.
        for col in ["h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip",
                    "sp_era_diff", "sp_whip_diff",
                    "h_sp_k9", "a_sp_k9", "h_sp_hr9", "a_sp_hr9",
                    "sp_k9_diff", "sp_hr9_diff",
                    "h_park_factor",
                    "home_pi", "away_pi", "pi_diff", "pi_confidence"]:
            matrix[col] = np.nan

        if sport == "MLB":
            # Park run factor: static lookup by home team name
            matrix["h_park_factor"] = (
                matrix["home_team"].map(MLB_PARK_FACTORS).fillna(1.0)
            )

            # ── Pi (Elo-like) power ratings ───────────────────────────────
            # Look up each game's home_pi and away_pi from pi_rating_history.
            # Uses the rating snapshot BEFORE the game date (no lookahead).
            try:
                _pi_hist = pd.read_sql_query(
                    "SELECT team, game_date, home_pi, away_pi, games_home, games_away "
                    "FROM pi_rating_history WHERE sport = ?",
                    conn, params=[sport],
                )
                if not _pi_hist.empty:
                    _pi_hist["game_date"] = pd.to_datetime(_pi_hist["game_date"])

                    # Build lookup: for each (team, game_date), find the latest Pi
                    # rating where pi.game_date < game.game_date.
                    # Vectorized approach: merge_asof on sorted dates.
                    _pi_hist = _pi_hist.sort_values("game_date")

                    # Home team Pi
                    _home_merge = matrix[["game_id", "home_team", "game_date"]].copy()
                    _home_merge = _home_merge.sort_values("game_date")
                    _home_pi = pd.merge_asof(
                        _home_merge,
                        _pi_hist.rename(columns={"team": "home_team",
                                                 "home_pi": "_h_home_pi",
                                                 "games_home": "_h_gp_home",
                                                 "games_away": "_h_gp_away"}),
                        on="game_date",
                        by="home_team",
                        direction="backward",
                        allow_exact_matches=False,  # strict < (no lookahead)
                    )

                    # Away team Pi
                    _away_merge = matrix[["game_id", "away_team", "game_date"]].copy()
                    _away_merge = _away_merge.sort_values("game_date")
                    _away_pi = pd.merge_asof(
                        _away_merge,
                        _pi_hist.rename(columns={"team": "away_team",
                                                 "away_pi": "_a_away_pi",
                                                 "games_home": "_a_gp_home",
                                                 "games_away": "_a_gp_away"}),
                        on="game_date",
                        by="away_team",
                        direction="backward",
                        allow_exact_matches=False,
                    )

                    # Merge back onto matrix
                    matrix = matrix.merge(
                        _home_pi[["game_id", "_h_home_pi", "_h_gp_home"]],
                        on="game_id", how="left",
                    )
                    matrix = matrix.merge(
                        _away_pi[["game_id", "_a_away_pi", "_a_gp_away"]],
                        on="game_id", how="left",
                    )
                    matrix["home_pi"] = matrix["_h_home_pi"]
                    matrix["away_pi"] = matrix["_a_away_pi"]
                    matrix["pi_diff"] = matrix["home_pi"] - matrix["away_pi"]
                    # Confidence: min of home/away games played, scaled by 30
                    _min_gp = pd.concat([
                        matrix["_h_gp_home"].fillna(0),
                        matrix["_a_gp_away"].fillna(0),
                    ], axis=1).min(axis=1)
                    matrix["pi_confidence"] = (_min_gp / 30.0).clip(upper=1.0)
                    matrix.drop(columns=["_h_home_pi", "_h_gp_home",
                                         "_a_away_pi", "_a_gp_away"],
                                inplace=True, errors="ignore")

                    _pi_fill = matrix["home_pi"].notna().mean()
                    log.info("  MLB Pi rating fill rate: %.1f%%", _pi_fill * 100)

            except Exception as exc:
                log.warning("Pi rating lookup failed: %s", exc)

            # Load pitcher season stats and join by probable pitcher name
            pitcher_stats = _load_pitcher_stats(conn)
            if not pitcher_stats.empty:
                # Build lookup: (pitcher_name, season) → stats
                ps = pitcher_stats.set_index(["pitcher_name", "season"])

                # Extract probable pitcher names from team_stats.stats_json
                # Query one row per game (home side) to get both pitcher names
                sp_query = """
                    SELECT
                        ts.game_id,
                        g.season,
                        MAX(CASE WHEN ts.is_home=1
                            THEN json_extract(ts.stats_json, '$.home_probable_pitcher')
                            END) AS h_sp_name,
                        MAX(CASE WHEN ts.is_home=0
                            THEN json_extract(ts.stats_json, '$.away_probable_pitcher')
                            END) AS a_sp_name
                    FROM team_stats ts
                    JOIN games g ON ts.game_id = g.game_id
                    WHERE ts.sport = ?
                    GROUP BY ts.game_id
                """
                sp_df = pd.read_sql_query(sp_query, conn, params=[sport])
                sp_df["season"] = sp_df["season"].astype(str)

                def _lookup_stat(name: str, season: str, stat: str) -> float:
                    if not name or pd.isna(name):
                        return np.nan
                    try:
                        return float(ps.loc[(name, season), stat])
                    except (KeyError, TypeError):
                        return np.nan

                sp_df["h_sp_era"]  = [_lookup_stat(n, s, "era")  for n, s in zip(sp_df["h_sp_name"], sp_df["season"])]
                sp_df["a_sp_era"]  = [_lookup_stat(n, s, "era")  for n, s in zip(sp_df["a_sp_name"], sp_df["season"])]
                sp_df["h_sp_whip"] = [_lookup_stat(n, s, "whip") for n, s in zip(sp_df["h_sp_name"], sp_df["season"])]
                sp_df["a_sp_whip"] = [_lookup_stat(n, s, "whip") for n, s in zip(sp_df["a_sp_name"], sp_df["season"])]
                sp_df["h_sp_k9"]   = [_lookup_stat(n, s, "k9")   for n, s in zip(sp_df["h_sp_name"], sp_df["season"])]
                sp_df["a_sp_k9"]   = [_lookup_stat(n, s, "k9")   for n, s in zip(sp_df["a_sp_name"], sp_df["season"])]
                sp_df["h_sp_hr9"]  = [_lookup_stat(n, s, "hr9")  for n, s in zip(sp_df["h_sp_name"], sp_df["season"])]
                sp_df["a_sp_hr9"]  = [_lookup_stat(n, s, "hr9")  for n, s in zip(sp_df["a_sp_name"], sp_df["season"])]
                sp_df["h_sp_bb9"]  = [_lookup_stat(n, s, "bb9")  for n, s in zip(sp_df["h_sp_name"], sp_df["season"])]
                sp_df["a_sp_bb9"]  = [_lookup_stat(n, s, "bb9")  for n, s in zip(sp_df["a_sp_name"], sp_df["season"])]
                sp_df["h_sp_xera"] = [_lookup_stat(n, s, "xera") for n, s in zip(sp_df["h_sp_name"], sp_df["season"])]
                sp_df["a_sp_xera"] = [_lookup_stat(n, s, "xera") for n, s in zip(sp_df["a_sp_name"], sp_df["season"])]

                _sp_cols = ["game_id", "h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip",
                            "h_sp_k9", "a_sp_k9", "h_sp_hr9", "a_sp_hr9",
                            "h_sp_bb9", "a_sp_bb9", "h_sp_xera", "a_sp_xera"]
                matrix = matrix.merge(
                    sp_df[_sp_cols],
                    on="game_id", how="left", suffixes=("", "_sp"),
                )
                # Use merged columns (overwrite the NaN placeholders)
                for col in _sp_cols[1:]:  # skip game_id
                    merged = col + "_sp"
                    if merged in matrix.columns:
                        matrix[col] = matrix[merged].combine_first(matrix[col])
                        matrix.drop(columns=[merged], inplace=True)

            # Differential: positive = home pitcher is better
            matrix["sp_era_diff"]  = matrix["a_sp_era"]  - matrix["h_sp_era"]
            matrix["sp_whip_diff"] = matrix["a_sp_whip"] - matrix["h_sp_whip"]
            matrix["sp_k9_diff"]   = matrix["h_sp_k9"]   - matrix["a_sp_k9"]
            matrix["sp_hr9_diff"]  = matrix["a_sp_hr9"]  - matrix["h_sp_hr9"]
            # K-BB% differential: higher = home pitcher has better command
            matrix["sp_kbb_diff"]  = (
                (matrix["h_sp_k9"] - matrix["h_sp_bb9"])
                - (matrix["a_sp_k9"] - matrix["a_sp_bb9"])
            )
            # xERA differential: luck-stripped ERA (Baseball Savant expected ERA)
            matrix["sp_xera_diff"] = matrix["a_sp_xera"] - matrix["h_sp_xera"]
            # NOTE: sp_xera_diff uses same sign convention as sp_era_diff:
            # positive = home pitcher is better (lower xERA)

            fill_rate = matrix["h_sp_era"].notna().mean()
            k9_fill   = matrix["h_sp_k9"].notna().mean()
            log.info("  MLB pitcher fill rate (season ERA): %.1f%%, K/9: %.1f%%",
                     fill_rate * 100, k9_fill * 100)

            # ── Rolling pitcher ERA/WHIP (last 3 and 5 starts) ────────────
            # Default NaN placeholders
            for col in ["h_sp_era_3","a_sp_era_3","h_sp_era_5","a_sp_era_5",
                        "h_sp_whip_3","a_sp_whip_3","sp_era3_diff","sp_era5_diff"]:
                matrix[col] = np.nan

            try:
                roll3 = _compute_rolling_pitcher_stats(conn, 3)
                roll5 = _compute_rolling_pitcher_stats(conn, 5)

                if not roll3.empty and not roll5.empty:
                    # Build (player_id, season) → {game_date: rolling_stats} map
                    # Load pitcher IDs from pitcher_stats table
                    pid_df = pd.read_sql_query(
                        "SELECT pitcher_name, season, player_id FROM pitcher_stats WHERE player_id IS NOT NULL",
                        conn,
                    )
                    pid_df["season"] = pid_df["season"].astype(str)
                    pid_map = dict(zip(
                        zip(pid_df["pitcher_name"], pid_df["season"]),
                        pid_df["player_id"].astype(int),
                    ))

                    # Merge rolling stats onto game matrix via SP name → player_id → game_date
                    def _join_rolling(sp_name_col: str, prefix: str):
                        """Look up rolling ERA/WHIP for each game row."""
                        rows_out_3 = []
                        rows_out_5 = []
                        # Build indexed lookups once
                        r3_idx = roll3.set_index(["player_id", "game_date"])
                        r5_idx = roll5.set_index(["player_id", "game_date"])

                        for _, row in sp_df[["game_id", "season", sp_name_col]].iterrows():
                            name    = row[sp_name_col]
                            season  = row["season"]
                            game_id = row["game_id"]
                            # Look up game_date for this game_id
                            gdate = matrix.loc[matrix["game_id"] == game_id, "game_date"]
                            if gdate.empty or pd.isna(name) or not name:
                                rows_out_3.append(np.nan)
                                rows_out_5.append(np.nan)
                                continue
                            gdate = gdate.iloc[0]
                            pid = pid_map.get((name, season))
                            if pid is None:
                                rows_out_3.append(np.nan)
                                rows_out_5.append(np.nan)
                                continue
                            try:
                                rows_out_3.append(float(r3_idx.loc[(pid, gdate), f"rolling_era_3"]))
                            except (KeyError, TypeError):
                                rows_out_3.append(np.nan)
                            try:
                                rows_out_5.append(float(r5_idx.loc[(pid, gdate), f"rolling_era_5"]))
                            except (KeyError, TypeError):
                                rows_out_5.append(np.nan)
                        return rows_out_3, rows_out_5

                    # Build vectorized join using merge instead of row-by-row
                    # Join roll3/roll5 onto sp_df via player_id + game_date
                    sp_df2 = sp_df.copy()
                    sp_df2["game_date"] = sp_df2["game_id"].map(
                        dict(zip(matrix["game_id"], matrix["game_date"]))
                    )

                    for side, name_col, pid_col in [("h", "h_sp_name", "h_pid"), ("a", "a_sp_name", "a_pid")]:
                        # Vectorized pid lookup via tuples — avoids apply(axis=1)
                        _keys = list(zip(sp_df2[name_col], sp_df2["season"]))
                        sp_df2[pid_col] = pd.to_numeric(
                            pd.Series([pid_map.get(k) for k in _keys], index=sp_df2.index),
                            errors="coerce",
                        )

                    # Merge rolling stats for home/away SP using merge_asof
                    # (backward join: most recent pitcher start <= game_date).
                    # Fixes live inference where exact-date merge returned NaN
                    # for upcoming games with no pitcher log on that date.
                    for side, pid_col in [("h", "h_pid"), ("a", "a_pid")]:
                        tmp = sp_df2[["game_id", "game_date", pid_col]].dropna(subset=[pid_col, "game_date"])
                        tmp = tmp.rename(columns={pid_col: "player_id"})
                        tmp["player_id"] = tmp["player_id"].astype(int)
                        tmp["game_date"] = pd.to_datetime(tmp["game_date"])
                        tmp = tmp.sort_values("game_date")

                        # Window 3: ERA, WHIP
                        r3 = roll3[["player_id", "game_date", f"rolling_era_3", f"rolling_whip_3"]].copy()
                        r3 = r3.dropna(subset=[f"rolling_era_3"], how="all")
                        r3["game_date"] = pd.to_datetime(r3["game_date"])
                        r3 = r3.sort_values("game_date")
                        tmp3 = pd.merge_asof(
                            tmp, r3.rename(columns={
                                "rolling_era_3": f"{side}_sp_era_3",
                                "rolling_whip_3": f"{side}_sp_whip_3",
                            }),
                            on="game_date", by="player_id",
                            direction="backward", allow_exact_matches=True,
                        )

                        # Window 5: ERA + WHIP
                        r5 = roll5[["player_id", "game_date", f"rolling_era_5", f"rolling_whip_5"]].copy()
                        r5 = r5.dropna(subset=[f"rolling_era_5"], how="all")
                        r5["game_date"] = pd.to_datetime(r5["game_date"])
                        r5 = r5.sort_values("game_date")
                        tmp5 = pd.merge_asof(
                            tmp, r5.rename(columns={
                                "rolling_era_5": f"{side}_sp_era_5",
                                "rolling_whip_5": f"{side}_sp_whip_5",
                            }),
                            on="game_date", by="player_id",
                            direction="backward", allow_exact_matches=True,
                        )

                        for col, src in [(f"{side}_sp_era_3", tmp3), (f"{side}_sp_whip_3", tmp3),
                                         (f"{side}_sp_era_5", tmp5), (f"{side}_sp_whip_5", tmp5)]:
                            if col in src.columns:
                                merged_col = src[["game_id", col]].dropna()
                                matrix = matrix.merge(merged_col, on="game_id", how="left", suffixes=("","_r"))
                                dup = col + "_r"
                                if dup in matrix.columns:
                                    matrix[col] = matrix[dup].combine_first(matrix[col])
                                    matrix.drop(columns=[dup], inplace=True)

                    matrix["sp_era3_diff"] = matrix["a_sp_era_3"] - matrix["h_sp_era_3"]
                    matrix["sp_era5_diff"] = matrix["a_sp_era_5"] - matrix["h_sp_era_5"]
                    matrix["sp_whip3_diff"] = matrix["a_sp_whip_3"] - matrix["h_sp_whip_3"]
                    matrix["sp_whip5_diff"] = matrix["a_sp_whip_5"] - matrix["h_sp_whip_5"]

                    # ── Rolling HR/9 (3-start and 5-start windows) ──
                    try:
                        hr9_3 = _compute_rolling_pitcher_hr9(conn, 3)
                        hr9_5 = _compute_rolling_pitcher_hr9(conn, 5)
                        for side, pid_col in [("h", "h_pid"), ("a", "a_pid")]:
                            tmp = sp_df2[["game_id", "game_date", pid_col]].dropna(subset=[pid_col, "game_date"])
                            tmp = tmp.rename(columns={pid_col: "player_id"})
                            tmp["player_id"] = tmp["player_id"].astype(int)
                            tmp["game_date"] = pd.to_datetime(tmp["game_date"])
                            tmp = tmp.sort_values("game_date")

                            for rdf, n_win in [(hr9_3, 3), (hr9_5, 5)]:
                                col_name = f"rolling_hr9_{n_win}"
                                if rdf.empty or col_name not in rdf.columns:
                                    continue
                                r = rdf[["player_id", "game_date", col_name]].copy()
                                r = r.dropna(subset=[col_name])
                                r["game_date"] = pd.to_datetime(r["game_date"])
                                r = r.sort_values("game_date")
                                merged = pd.merge_asof(
                                    tmp, r.rename(columns={col_name: f"{side}_sp_hr9_{n_win}"}),
                                    on="game_date", by="player_id",
                                    direction="backward", allow_exact_matches=True,
                                )
                                dest_col = f"{side}_sp_hr9_{n_win}"
                                if dest_col in merged.columns:
                                    mc = merged[["game_id", dest_col]].dropna()
                                    matrix = matrix.merge(mc, on="game_id", how="left", suffixes=("","_r"))
                                    dup = dest_col + "_r"
                                    if dup in matrix.columns:
                                        matrix[dest_col] = matrix[dup].combine_first(matrix[dest_col])
                                        matrix.drop(columns=[dup], inplace=True)

                        for col in ["h_sp_hr9_3","a_sp_hr9_3","h_sp_hr9_5","a_sp_hr9_5"]:
                            if col not in matrix.columns:
                                matrix[col] = np.nan
                        matrix["sp_hr9_3_diff"] = matrix["a_sp_hr9_3"] - matrix["h_sp_hr9_3"]
                        matrix["sp_hr9_5_diff"] = matrix["a_sp_hr9_5"] - matrix["h_sp_hr9_5"]

                        hr9_fill = matrix["h_sp_hr9_5"].notna().mean()
                        log.info("  MLB rolling HR/9-5 fill rate: %.1f%%", hr9_fill * 100)
                    except Exception as exc:
                        log.warning("MLB rolling HR/9 stats failed: %s", exc)

                    roll3_fill = matrix["h_sp_era_3"].notna().mean()
                    log.info("  MLB rolling ERA-3 fill rate: %.1f%%", roll3_fill * 100)
            except Exception as exc:
                log.warning("MLB rolling pitcher stats failed: %s", exc)

            # ── Team batting stats (season OPS/SLG/K%/BB%/barrel/hard_hit) ──
            _bat_cols = ["ops", "slg", "k_pct", "bb_pct", "barrel_pct", "hard_hit_pct"]
            for col in [f"{s}_{c}" for s in ("h", "a") for c in _bat_cols] + [
                "ops_diff", "barrel_diff", "hard_hit_diff"
            ]:
                matrix[col] = np.nan
            try:
                bat_df = pd.read_sql_query(
                    "SELECT team_name, season, ops, slg, k_pct, bb_pct, "
                    "barrel_pct, hard_hit_pct FROM team_batting_stats",
                    conn,
                )
                if not bat_df.empty:
                    bat_df["season"] = bat_df["season"].astype(str)
                    matrix["season"] = matrix["season"].astype(str)
                    # Vectorized merge replaces row-by-row apply (12K rows × N lookups)
                    for side, team_col in [("h", "home_team"), ("a", "away_team")]:
                        rename_map = {s: f"{side}_{s}" for s in _bat_cols}
                        bat_side = bat_df.rename(columns={
                            "team_name": team_col, **rename_map
                        })
                        matrix = matrix.merge(
                            bat_side[[team_col, "season"] + list(rename_map.values())],
                            on=[team_col, "season"], how="left", suffixes=("", "_bat"),
                        )
                        for col in rename_map.values():
                            dup = col + "_bat"
                            if dup in matrix.columns:
                                matrix[col] = matrix[dup].combine_first(matrix[col])
                                matrix.drop(columns=[dup], inplace=True)
                    matrix["ops_diff"]      = matrix["h_ops"]          - matrix["a_ops"]
                    matrix["barrel_diff"]   = matrix["h_barrel_pct"]   - matrix["a_barrel_pct"]
                    matrix["hard_hit_diff"] = matrix["h_hard_hit_pct"] - matrix["a_hard_hit_pct"]
                    fill = matrix["h_ops"].notna().mean()
                    barrel_fill = matrix["h_barrel_pct"].notna().mean()
                    log.info("  MLB team batting fill rate: %.1f%%, barrel: %.1f%%",
                             fill * 100, barrel_fill * 100)
            except Exception as exc:
                log.warning("MLB team batting stats failed: %s", exc)

        # ��─ 7b. Weather features (temperature, wind, dome) ────────────────
        _wx_cols = ["temperature_f", "wind_speed_mph", "wind_speed_forecast_mph",
                    "wind_direction_deg", "is_dome"]
        for col in _wx_cols:
            if col not in matrix.columns:
                matrix[col] = np.nan
        try:
            wx_df = pd.read_sql_query(
                "SELECT home_team, game_date, temperature_f, wind_speed_mph, "
                "wind_speed_forecast_mph, wind_direction_deg, is_dome FROM game_weather",
                conn,
            )
            if not wx_df.empty:
                wx_df["game_date"] = pd.to_datetime(wx_df["game_date"], errors="coerce")
                matrix["game_date"] = pd.to_datetime(matrix["game_date"], errors="coerce")
                matrix = matrix.merge(
                    wx_df, on=["home_team", "game_date"], how="left",
                    suffixes=("", "_wx"),
                )
                for col in _wx_cols:
                    dup = col + "_wx"
                    if dup in matrix.columns:
                        matrix[col] = matrix[dup].combine_first(matrix[col])
                        matrix.drop(columns=[dup], inplace=True)
                fill = matrix["wind_speed_forecast_mph"].notna().mean()
                log.info("  MLB weather fill rate (forecast wind): %.1f%%", fill * 100)
        except Exception as exc:
            log.warning("MLB weather join failed: %s", exc)

        # ── 8. Target variables ────────────────────────────────────────────
        matrix = _compute_targets(matrix)

        # ── 9. Numeric coerce on line columns ─────────────────────────────
        for col in ["close_spread", "close_total", "close_ml_home", "close_ml_away"]:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce")

        log.info(
            "  Feature matrix: %d rows × %d cols | "
            "ATS labels: %d  | O-U labels: %d",
            len(matrix), len(matrix.columns),
            int(matrix[TARGET_ATS].notna().sum()),
            int(matrix[TARGET_OU].notna().sum()),
        )

        return matrix

    finally:
        if close_conn:
            conn.close()


def get_model_ready(
    sport: str,
    target: str = TARGET_ATS,
    include_lines: bool = False,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Convenience wrapper: build matrix, select feature columns, return (X, y, cols).

    Parameters
    ----------
    sport         : sport key
    target        : TARGET_ATS or TARGET_OU
    include_lines : if True, include close_spread/close_total/close_ml_* in X
                    (useful for studying line value; set False for pre-game prediction)
    dropna        : drop rows where target or any feature is NaN

    Returns
    -------
    X      : pd.DataFrame of numeric features
    y      : pd.Series of 0/1 labels
    cols   : list of feature column names used
    """
    matrix = build_feature_matrix(sport)
    if matrix.empty:
        return pd.DataFrame(), pd.Series(dtype=float), []

    cols = MODEL_FEATURE_COLS + (LINE_COLS if include_lines else [])
    # Keep only columns that exist in the matrix
    cols = [c for c in cols if c in matrix.columns]

    sub = matrix[cols + [target]].copy()

    # Drop rows where target is missing
    sub = sub[sub[target].notna()]

    # Drop columns that are entirely null for this sport (e.g. pitcher ERA for NHL)
    # before the row-level dropna, so sport-irrelevant features don't erase all rows.
    all_null_cols = [c for c in cols if sub[c].isna().all()]
    if all_null_cols:
        cols = [c for c in cols if c not in all_null_cols]

    if dropna:
        sub = sub.dropna(subset=cols)

    X = sub[cols].astype(float)
    y = sub[target].astype(float)

    return X, y, cols


# ═══════════════════════════════════════════════════════════════════════════════
# V2 SHADOW — leak-free feature matrix for ATS retrain
# ═══════════════════════════════════════════════════════════════════════════════
# Changes vs V1:
#   1. DROPS 12 season-level pitcher features (h_sp_era, a_sp_era, etc.)
#      → replaced by rolling-only ERA/WHIP (already safe) + new rolling K/9
#   2. DROPS 9 season-level team batting features (h_ops, a_ops, etc.)
#      → no per-game batting source exists; clean drop
#   3. ADDS rolling K/9 at 3 and 5 starts (from pitcher_game_logs.strikeouts)
#   4. ADDS rolling ERA/WHIP at 10 starts (longer-term signal, replaces season)
# Total: 59 safe (kept) - 0 + 12 new rolling = 71 features (before spread aug)

_LEAKY_PITCHER_COLS = [
    "h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip",
    "sp_era_diff", "sp_whip_diff",
    "h_sp_k9", "a_sp_k9", "h_sp_hr9", "a_sp_hr9",
    "sp_k9_diff", "sp_hr9_diff",
]
_LEAKY_BATTING_COLS = [
    "h_ops", "a_ops", "ops_diff", "h_slg", "a_slg",
    "h_k_pct", "a_k_pct", "h_bb_pct", "a_bb_pct",
]

# New rolling columns added by v2
_V2_ROLLING_PITCHER_COLS = [
    # K/9 at 3 and 5 starts
    "h_sp_k9_3", "a_sp_k9_3", "h_sp_k9_5", "a_sp_k9_5",
    "sp_k9_3_diff", "sp_k9_5_diff",
    # ERA/WHIP/K9 at 10 starts (seasonal replacement)
    "h_sp_era_10", "a_sp_era_10", "sp_era10_diff",
    "h_sp_whip_10", "a_sp_whip_10", "sp_whip10_diff",
]

MODEL_FEATURE_COLS_V2: list[str] = [
    c for c in MODEL_FEATURE_COLS
    if c not in _LEAKY_PITCHER_COLS and c not in _LEAKY_BATTING_COLS
] + _V2_ROLLING_PITCHER_COLS

# Rolling batting columns — used by totals model, NOT by ATS
_V2_ROLLING_BATTING_COLS = [
    # OBP at 10 and 20 games
    "h_rolling_obp_10", "a_rolling_obp_10", "rolling_obp_10_diff",
    "h_rolling_obp_20", "a_rolling_obp_20", "rolling_obp_20_diff",
    # SLG at 10 and 20 games
    "h_rolling_slg_10", "a_rolling_slg_10", "rolling_slg_10_diff",
    "h_rolling_slg_20", "a_rolling_slg_20", "rolling_slg_20_diff",
    # ISO at 10 and 20 games
    "h_rolling_iso_10", "a_rolling_iso_10", "rolling_iso_10_diff",
    "h_rolling_iso_20", "a_rolling_iso_20", "rolling_iso_20_diff",
    # K% at 10 and 20 games
    "h_rolling_kpct_10", "a_rolling_kpct_10", "rolling_kpct_10_diff",
    "h_rolling_kpct_20", "a_rolling_kpct_20", "rolling_kpct_20_diff",
    # BB% at 10 and 20 games
    "h_rolling_bbpct_10", "a_rolling_bbpct_10", "rolling_bbpct_10_diff",
    "h_rolling_bbpct_20", "a_rolling_bbpct_20", "rolling_bbpct_20_diff",
    # Runs per game at 10 and 20
    "h_rolling_rpg_10", "a_rolling_rpg_10", "rolling_rpg_10_diff",
    "h_rolling_rpg_20", "a_rolling_rpg_20", "rolling_rpg_20_diff",
    # Park × SLG interactions
    "park_x_h_rolling_slg_10", "park_x_a_rolling_slg_10",
]

# Combined V2 feature set for totals model (ATS + batting)
MODEL_FEATURE_COLS_V2_TOTALS: list[str] = (
    MODEL_FEATURE_COLS_V2 + _V2_ROLLING_BATTING_COLS
)

# V3 ATS shadow: V2 ATS base + rolling batting features
MODEL_FEATURE_COLS_V3_ATS: list[str] = (
    MODEL_FEATURE_COLS_V2 + _V2_ROLLING_BATTING_COLS
)


def _compute_rolling_pitcher_stats_extended(
    conn: sqlite3.Connection,
    n: int,
) -> pd.DataFrame:
    """
    Extended rolling pitcher stats: ERA, WHIP, and K/9 over the last *n* starts.
    shift(1) before rolling → zero lookahead.

    Returns DataFrame keyed on (player_id, game_date) with:
        rolling_era_{n}   rolling_whip_{n}   rolling_k9_{n}
    """
    out_cols = [f"rolling_era_{n}", f"rolling_whip_{n}", f"rolling_k9_{n}"]
    try:
        df = pd.read_sql_query(
            "SELECT player_id, game_date, ip, er, hits, walks, strikeouts "
            "FROM pitcher_game_logs ORDER BY player_id, game_date",
            conn,
        )
    except Exception:
        return pd.DataFrame(columns=["player_id", "game_date"] + out_cols)

    if df.empty:
        return pd.DataFrame(columns=["player_id", "game_date"] + out_cols)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["player_id", "game_date"]).copy()

    grp = df.groupby("player_id")
    ip_shifted = grp["ip"].shift(1)
    er_shifted = grp["er"].shift(1)
    hw_shifted = (df["hits"] + df["walks"]).groupby(df["player_id"]).shift(1)
    k_shifted  = grp["strikeouts"].shift(1)

    ip_roll = ip_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
    er_roll = er_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
    hw_roll = hw_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
    k_roll  = k_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)

    df[f"rolling_era_{n}"]  = (er_roll / ip_roll * 9).where(ip_roll > 0)
    df[f"rolling_whip_{n}"] = (hw_roll / ip_roll).where(ip_roll > 0)
    df[f"rolling_k9_{n}"]   = (k_roll / ip_roll * 9).where(ip_roll > 0)

    return df[["player_id", "game_date"] + out_cols].copy()


def _compute_rolling_batting_stats(
    conn: sqlite3.Connection,
    n: int,
) -> pd.DataFrame:
    """
    Rolling team batting stats over the last *n* games.
    shift(1) before rolling → zero lookahead.

    Reads from team_game_batting (raw counting stats per game),
    computes rolling SUMS of counting stats, then derives rates:
      OBP = (H + BB + HBP) / PA
      SLG = TB / AB
      ISO = (TB - H) / AB
      K%  = K / PA
      BB% = BB / PA
      RPG = runs / games

    Returns DataFrame keyed on (team_name, game_date) with rate columns.
    """
    prefix = f"rolling"
    out_cols = [
        f"rolling_obp_{n}", f"rolling_slg_{n}", f"rolling_iso_{n}",
        f"rolling_kpct_{n}", f"rolling_bbpct_{n}", f"rolling_rpg_{n}",
    ]
    empty = pd.DataFrame(columns=["team_name", "game_date"] + out_cols)

    try:
        df = pd.read_sql_query(
            "SELECT team_name, game_date, plate_appearances, at_bats, "
            "hits, total_bases, walks, hit_by_pitch, strikeouts, runs "
            "FROM team_game_batting ORDER BY team_name, game_date",
            conn,
        )
    except Exception:
        return empty

    if df.empty:
        return empty

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["team_name", "game_date"]).copy()

    # Shift(1): each stat uses only PRIOR games (no same-game inclusion)
    grp = df.groupby("team_name")
    pa_s  = grp["plate_appearances"].shift(1)
    ab_s  = grp["at_bats"].shift(1)
    h_s   = grp["hits"].shift(1)
    tb_s  = grp["total_bases"].shift(1)
    bb_s  = grp["walks"].shift(1)
    hbp_s = grp["hit_by_pitch"].shift(1)
    k_s   = grp["strikeouts"].shift(1)
    r_s   = grp["runs"].shift(1)

    # Rolling sums over counting stats (min_periods=3 for stability)
    _mp = min(3, n)
    pa_roll  = pa_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)
    ab_roll  = ab_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)
    h_roll   = h_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)
    tb_roll  = tb_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)
    bb_roll  = bb_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)
    hbp_roll = hbp_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)
    k_roll   = k_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)
    r_roll   = r_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)

    # Count of games in window for RPG
    ones_s = grp["runs"].shift(1).notna().astype(float)
    games_roll = ones_s.groupby(df["team_name"]).rolling(n, min_periods=_mp).sum().droplevel(0)

    # Derive rate stats from rolling counting sums
    df[f"rolling_obp_{n}"]  = ((h_roll + bb_roll + hbp_roll) / pa_roll).where(pa_roll > 0)
    df[f"rolling_slg_{n}"]  = (tb_roll / ab_roll).where(ab_roll > 0)
    df[f"rolling_iso_{n}"]  = ((tb_roll - h_roll) / ab_roll).where(ab_roll > 0)
    df[f"rolling_kpct_{n}"] = (k_roll / pa_roll).where(pa_roll > 0)
    df[f"rolling_bbpct_{n}"]= (bb_roll / pa_roll).where(pa_roll > 0)
    df[f"rolling_rpg_{n}"]  = (r_roll / games_roll).where(games_roll > 0)

    return df[["team_name", "game_date"] + out_cols].copy()


def build_feature_matrix_v2(
    sport: str,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    """
    Leak-free feature matrix for ATS shadow retrain.

    Calls the V1 builder to get the full matrix (reusing all safe computation),
    then:
      1. Drops 21 leaky season-level features
      2. Adds rolling K/9 at windows 3, 5
      3. Adds rolling ERA/WHIP/K9 at window 10 (seasonal replacement)
      4. Adds rolling team batting stats at windows 10, 20 (totals model)
    """
    matrix = build_feature_matrix(sport, conn=conn)
    if matrix.empty:
        return matrix

    # ── 1. Drop leaky columns ──────────────────────────────────────────────
    drop_cols = [c for c in _LEAKY_PITCHER_COLS + _LEAKY_BATTING_COLS
                 if c in matrix.columns]
    matrix = matrix.drop(columns=drop_cols)
    log.info("  V2: dropped %d leaky columns", len(drop_cols))

    # ── 2. Add extended rolling pitcher stats (K/9 + window-10) ────────────
    if sport.upper() not in ("MLB", "BASEBALL_MLB"):
        # Only MLB has pitcher features; set NaN placeholders
        for col in _V2_ROLLING_PITCHER_COLS:
            matrix[col] = np.nan
        return matrix

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True

    try:
        # Compute extended rolling stats at 3, 5, and 10
        roll3  = _compute_rolling_pitcher_stats_extended(conn, 3)
        roll5  = _compute_rolling_pitcher_stats_extended(conn, 5)
        roll10 = _compute_rolling_pitcher_stats_extended(conn, 10)

        # Set NaN placeholders for all new columns
        for col in _V2_ROLLING_PITCHER_COLS:
            if col not in matrix.columns:
                matrix[col] = np.nan

        if roll3.empty and roll5.empty and roll10.empty:
            log.warning("  V2: no rolling pitcher data available")
            return matrix

        # Load pitcher name → player_id mapping
        pid_df = pd.read_sql_query(
            "SELECT pitcher_name, season, player_id FROM pitcher_stats "
            "WHERE player_id IS NOT NULL",
            conn,
        )
        pid_df["season"] = pid_df["season"].astype(str)
        pid_map = dict(zip(
            zip(pid_df["pitcher_name"], pid_df["season"]),
            pid_df["player_id"].astype(int),
        ))

        # Load SP names per game (same query as V1)
        sp_df = pd.read_sql_query(
            """SELECT ts.game_id, g.season,
                MAX(CASE WHEN ts.is_home=1 THEN json_extract(ts.stats_json, '$.home_probable_pitcher') END) AS h_sp_name,
                MAX(CASE WHEN ts.is_home=0 THEN json_extract(ts.stats_json, '$.away_probable_pitcher') END) AS a_sp_name
            FROM team_stats ts JOIN games g ON ts.game_id = g.game_id
            WHERE ts.sport = 'MLB' GROUP BY ts.game_id""",
            conn,
        )
        sp_df["season"] = sp_df["season"].astype(str)

        # Build game_id → game_date map
        gid_to_date = dict(zip(matrix["game_id"], matrix["game_date"]))
        sp_df["game_date"] = sp_df["game_id"].map(gid_to_date)

        # Resolve player_ids for home and away SPs
        for side, name_col, pid_col in [("h", "h_sp_name", "h_pid"),
                                         ("a", "a_sp_name", "a_pid")]:
            _keys = list(zip(sp_df[name_col].fillna(""), sp_df["season"]))
            sp_df[pid_col] = pd.to_numeric(
                pd.Series([pid_map.get(k) for k in _keys], index=sp_df.index),
                errors="coerce",
            )

        # Merge each rolling window's new stats onto matrix.
        # Use merge_asof (backward) so upcoming games get the pitcher's
        # most recent rolling stats — exact-date merge fails for future
        # games because the pitcher has no log entry on the game date.
        for side, pid_col in [("h", "h_pid"), ("a", "a_pid")]:
            tmp = sp_df[["game_id", "game_date", pid_col]].dropna(subset=[pid_col])
            tmp = tmp.rename(columns={pid_col: "player_id"})
            tmp["player_id"] = tmp["player_id"].astype(int)
            tmp["game_date"] = pd.to_datetime(tmp["game_date"])
            tmp = tmp.sort_values("game_date")

            # Window 3: K/9
            r3 = roll3[["player_id", "game_date", "rolling_k9_3"]].dropna(subset=["rolling_k9_3"]).copy()
            r3["game_date"] = pd.to_datetime(r3["game_date"])
            r3 = r3.sort_values("game_date")
            merged3 = pd.merge_asof(
                tmp, r3.rename(columns={"rolling_k9_3": f"{side}_sp_k9_3"}),
                on="game_date", by="player_id",
                direction="backward", allow_exact_matches=True,
            )
            for col in [f"{side}_sp_k9_3"]:
                if col in merged3.columns:
                    mcol = merged3[["game_id", col]].dropna()
                    matrix = matrix.merge(mcol, on="game_id", how="left", suffixes=("", "_v2"))
                    dup = col + "_v2"
                    if dup in matrix.columns:
                        matrix[col] = matrix[dup].combine_first(matrix[col])
                        matrix.drop(columns=[dup], inplace=True)

            # Window 5: K/9
            r5 = roll5[["player_id", "game_date", "rolling_k9_5"]].dropna(subset=["rolling_k9_5"]).copy()
            r5["game_date"] = pd.to_datetime(r5["game_date"])
            r5 = r5.sort_values("game_date")
            merged5 = pd.merge_asof(
                tmp, r5.rename(columns={"rolling_k9_5": f"{side}_sp_k9_5"}),
                on="game_date", by="player_id",
                direction="backward", allow_exact_matches=True,
            )
            for col in [f"{side}_sp_k9_5"]:
                if col in merged5.columns:
                    mcol = merged5[["game_id", col]].dropna()
                    matrix = matrix.merge(mcol, on="game_id", how="left", suffixes=("", "_v2"))
                    dup = col + "_v2"
                    if dup in matrix.columns:
                        matrix[col] = matrix[dup].combine_first(matrix[col])
                        matrix.drop(columns=[dup], inplace=True)

            # Window 10: ERA, WHIP (seasonal replacement)
            r10_cols = ["rolling_era_10", "rolling_whip_10"]
            r10 = roll10[["player_id", "game_date"] + r10_cols].dropna(subset=r10_cols, how="all").copy()
            r10["game_date"] = pd.to_datetime(r10["game_date"])
            r10 = r10.sort_values("game_date")
            merged10 = pd.merge_asof(
                tmp, r10.rename(columns={
                    "rolling_era_10": f"{side}_sp_era_10",
                    "rolling_whip_10": f"{side}_sp_whip_10",
                }),
                on="game_date", by="player_id",
                direction="backward", allow_exact_matches=True,
            )
            for col in [f"{side}_sp_era_10", f"{side}_sp_whip_10"]:
                if col in merged10.columns:
                    mcol = merged10[["game_id", col]].dropna()
                    matrix = matrix.merge(mcol, on="game_id", how="left", suffixes=("", "_v2"))
                    dup = col + "_v2"
                    if dup in matrix.columns:
                        matrix[col] = matrix[dup].combine_first(matrix[col])
                        matrix.drop(columns=[dup], inplace=True)

        # Compute differentials
        matrix["sp_k9_3_diff"]   = matrix["h_sp_k9_3"]   - matrix["a_sp_k9_3"]
        matrix["sp_k9_5_diff"]   = matrix["h_sp_k9_5"]   - matrix["a_sp_k9_5"]
        matrix["sp_era10_diff"]  = matrix["a_sp_era_10"]  - matrix["h_sp_era_10"]
        matrix["sp_whip10_diff"] = matrix["a_sp_whip_10"] - matrix["h_sp_whip_10"]

        # Fill rates
        for col in ["h_sp_k9_3", "h_sp_era_10"]:
            if col in matrix.columns:
                fr = matrix[col].notna().mean()
                log.info("  V2: %s fill rate: %.1f%%", col, fr * 100)

        # ── 3. Add rolling team batting stats (for totals model) ──────────
        # These columns are NOT in MODEL_FEATURE_COLS_V2 (ATS ignores them).
        # They appear in MODEL_FEATURE_COLS_V2_TOTALS only.
        try:
            bat10 = _compute_rolling_batting_stats(conn, 10)
            bat20 = _compute_rolling_batting_stats(conn, 20)

            # Set NaN placeholders for all batting columns
            for col in _V2_ROLLING_BATTING_COLS:
                if col not in matrix.columns:
                    matrix[col] = np.nan

            if not bat10.empty:
                bat10["game_date"] = pd.to_datetime(bat10["game_date"])
                bat20["game_date"] = pd.to_datetime(bat20["game_date"])

                # Doubleheaders: same team can have 2 rows on one date.
                # Keep the LAST row per (team, date) — it has the most
                # up-to-date rolling window (includes game 1 of the DH).
                bat10 = bat10.drop_duplicates(
                    subset=["team_name", "game_date"], keep="last",
                )
                bat20 = bat20.drop_duplicates(
                    subset=["team_name", "game_date"], keep="last",
                )

                _bat_stats = ["obp", "slg", "iso", "kpct", "bbpct", "rpg"]

                for side, team_col in [("h", "home_team"), ("a", "away_team")]:
                    # Window 10
                    rename10 = {f"rolling_{s}_10": f"{side}_rolling_{s}_10"
                                for s in _bat_stats}
                    tmp10 = bat10.rename(columns=rename10)
                    tmp10 = tmp10.rename(columns={"team_name": "_join_team",
                                                   "game_date": "_join_date"})
                    _m_date = pd.to_datetime(matrix["game_date"], errors="coerce")
                    _join = pd.DataFrame({
                        "_join_team": matrix[team_col],
                        "_join_date": _m_date,
                        "_idx": range(len(matrix)),
                    })
                    merged = _join.merge(
                        tmp10, on=["_join_team", "_join_date"], how="left",
                    )
                    merged = merged.sort_values("_idx").reset_index(drop=True)
                    for s in _bat_stats:
                        col = f"{side}_rolling_{s}_10"
                        if col in merged.columns:
                            matrix[col] = merged[col].values

                    # Window 20
                    rename20 = {f"rolling_{s}_20": f"{side}_rolling_{s}_20"
                                for s in _bat_stats}
                    tmp20 = bat20.rename(columns=rename20)
                    tmp20 = tmp20.rename(columns={"team_name": "_join_team",
                                                   "game_date": "_join_date"})
                    merged20 = _join.merge(
                        tmp20, on=["_join_team", "_join_date"], how="left",
                    )
                    merged20 = merged20.sort_values("_idx").reset_index(drop=True)
                    for s in _bat_stats:
                        col = f"{side}_rolling_{s}_20"
                        if col in merged20.columns:
                            matrix[col] = merged20[col].values

                # Compute differentials (home - away)
                for s in _bat_stats:
                    matrix[f"rolling_{s}_10_diff"] = (
                        matrix[f"h_rolling_{s}_10"] - matrix[f"a_rolling_{s}_10"]
                    )
                    matrix[f"rolling_{s}_20_diff"] = (
                        matrix[f"h_rolling_{s}_20"] - matrix[f"a_rolling_{s}_20"]
                    )

                # Park × SLG interactions
                if "h_park_factor" in matrix.columns:
                    matrix["park_x_h_rolling_slg_10"] = (
                        matrix["h_park_factor"] * matrix["h_rolling_slg_10"]
                    )
                    matrix["park_x_a_rolling_slg_10"] = (
                        matrix["h_park_factor"] * matrix["a_rolling_slg_10"]
                    )

                # Fill rates
                for col in ["h_rolling_obp_10", "h_rolling_slg_10",
                            "h_rolling_obp_20", "h_rolling_slg_20"]:
                    if col in matrix.columns:
                        fr = matrix[col].notna().mean()
                        log.info("  V2: %s fill rate: %.1f%%", col, fr * 100)

        except Exception as exc:
            log.warning("  V2: rolling batting stats failed: %s", exc)

    except Exception as exc:
        log.warning("  V2: extended rolling pitcher stats failed: %s", exc)
    finally:
        if close_conn:
            conn.close()

    return matrix
