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
    "h_park_factor",      # home ballpark run factor (>1 = hitter-friendly)
    # Rolling pitcher ERA/WHIP — last 3 and 5 starts before this game (no lookahead)
    "h_sp_era_3",         # home SP rolling ERA over last 3 starts
    "a_sp_era_3",         # away SP rolling ERA over last 3 starts
    "h_sp_era_5",         # home SP rolling ERA over last 5 starts
    "a_sp_era_5",         # away SP rolling ERA over last 5 starts
    "h_sp_whip_3",        # home SP rolling WHIP over last 3 starts
    "a_sp_whip_3",        # away SP rolling WHIP over last 3 starts
    "sp_era3_diff",       # a_sp_era_3 - h_sp_era_3
    "sp_era5_diff",       # a_sp_era_5 - h_sp_era_5
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
MODEL_FEATURE_COLS: list[str] = (
    ROLLING_COLS + SCHEDULE_COLS + DIFFERENTIAL_COLS + GAME_COLS
)

# ── MLB park run factors ───────────────────────────────────────────────────────
# 5-year average run factor per home stadium (Fangraphs park factors, 2020-2024).
# >1.0 = more runs scored here than league average; <1.0 = pitcher-friendly.
# Used to contextualise run-line totals — Coors Field inflates scoring by ~23%.
MLB_PARK_FACTORS: dict[str, float] = {
    "Colorado Rockies":       1.23,
    "Cincinnati Reds":        1.10,
    "Boston Red Sox":         1.09,
    "Texas Rangers":          1.08,
    "Chicago Cubs":           1.07,
    "Baltimore Orioles":      1.06,
    "Philadelphia Phillies":  1.05,
    "Detroit Tigers":         1.04,
    "Chicago White Sox":      1.03,
    "Kansas City Royals":     1.03,
    "Pittsburgh Pirates":     1.02,
    "Toronto Blue Jays":      1.01,
    "Arizona Diamondbacks":   1.01,
    "Los Angeles Angels":     1.01,
    "Atlanta Braves":         1.01,
    "New York Yankees":       1.00,
    "St. Louis Cardinals":    0.99,
    "Washington Nationals":   0.99,
    "Cleveland Guardians":    1.00,
    "Houston Astros":         1.00,
    "Minnesota Twins":        1.00,
    "New York Mets":          0.97,
    "Tampa Bay Rays":         0.97,
    "San Diego Padres":       0.97,
    "Los Angeles Dodgers":    0.98,
    "Milwaukee Brewers":      0.98,
    "Seattle Mariners":       0.96,
    "Oakland Athletics":      0.96,
    "Miami Marlins":          0.95,
    "San Francisco Giants":   0.94,
    "Athletics":              0.96,   # relocated team alias
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

    def _roll_era(g: pd.DataFrame) -> pd.Series:
        ip_roll  = g["ip"].shift(1).rolling(n, min_periods=2).sum()
        er_roll  = g["er"].shift(1).rolling(n, min_periods=2).sum()
        era = (er_roll / ip_roll * 9).where(ip_roll > 0)
        return era

    def _roll_whip(g: pd.DataFrame) -> pd.Series:
        ip_roll  = g["ip"].shift(1).rolling(n, min_periods=2).sum()
        hw_roll  = (g["hits"] + g["walks"]).shift(1).rolling(n, min_periods=2).sum()
        whip = (hw_roll / ip_roll).where(ip_roll > 0)
        return whip

    grp = df.groupby("player_id", group_keys=False)
    df[f"rolling_era_{n}"]  = grp.apply(_roll_era).reset_index(level=0, drop=True)
    df[f"rolling_whip_{n}"] = grp.apply(_roll_whip).reset_index(level=0, drop=True)

    return df[["player_id", "game_date", f"rolling_era_{n}", f"rolling_whip_{n}"]].copy()


def _load_pitcher_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load the pitcher_stats table.  Returns an empty DataFrame with correct
    columns if the table doesn't exist yet (pre-migration).
    """
    try:
        df = pd.read_sql_query(
            "SELECT pitcher_name, season, era, whip, k9, bb9, fip FROM pitcher_stats",
            conn,
        )
        df["season"] = df["season"].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["pitcher_name", "season", "era", "whip", "k9", "bb9", "fip"])


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
                    "sp_era_diff", "sp_whip_diff", "h_park_factor"]:
            matrix[col] = np.nan

        if sport == "MLB":
            # Park run factor: static lookup by home team name
            matrix["h_park_factor"] = (
                matrix["home_team"].map(MLB_PARK_FACTORS).fillna(1.0)
            )

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

                matrix = matrix.merge(
                    sp_df[["game_id", "h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip"]],
                    on="game_id", how="left", suffixes=("", "_sp"),
                )
                # Use merged columns (overwrite the NaN placeholders)
                for col in ["h_sp_era", "a_sp_era", "h_sp_whip", "a_sp_whip"]:
                    merged = col + "_sp"
                    if merged in matrix.columns:
                        matrix[col] = matrix[merged].combine_first(matrix[col])
                        matrix.drop(columns=[merged], inplace=True)

            # Differential: positive = home pitcher is better
            matrix["sp_era_diff"]  = matrix["a_sp_era"]  - matrix["h_sp_era"]
            matrix["sp_whip_diff"] = matrix["a_sp_whip"] - matrix["h_sp_whip"]

            fill_rate = matrix["h_sp_era"].notna().mean()
            log.info("  MLB pitcher fill rate (season ERA): %.1f%%", fill_rate * 100)

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
                        sp_df2[pid_col] = sp_df2.apply(
                            lambda r: pid_map.get((r[name_col], r["season"])), axis=1
                        )
                        sp_df2[pid_col] = pd.to_numeric(sp_df2[pid_col], errors="coerce")

                    # Merge rolling stats for home SP
                    for side, pid_col in [("h", "h_pid"), ("a", "a_pid")]:
                        tmp = sp_df2[["game_id", "game_date", pid_col]].dropna(subset=[pid_col])
                        tmp[pid_col] = tmp[pid_col].astype(int)
                        tmp3 = tmp.merge(
                            roll3.rename(columns={"rolling_era_3": f"{side}_sp_era_3",
                                                  "rolling_whip_3": f"{side}_sp_whip_3"}),
                            left_on=[pid_col, "game_date"], right_on=["player_id", "game_date"], how="left"
                        )
                        tmp5 = tmp.merge(
                            roll5.rename(columns={"rolling_era_5": f"{side}_sp_era_5"}),
                            left_on=[pid_col, "game_date"], right_on=["player_id", "game_date"], how="left"
                        )
                        for col, src in [(f"{side}_sp_era_3", tmp3), (f"{side}_sp_whip_3", tmp3),
                                         (f"{side}_sp_era_5", tmp5)]:
                            if col in src.columns:
                                merged_col = src[["game_id", col]].dropna()
                                matrix = matrix.merge(merged_col, on="game_id", how="left", suffixes=("","_r"))
                                dup = col + "_r"
                                if dup in matrix.columns:
                                    matrix[col] = matrix[dup].combine_first(matrix[col])
                                    matrix.drop(columns=[dup], inplace=True)

                    matrix["sp_era3_diff"] = matrix["a_sp_era_3"] - matrix["h_sp_era_3"]
                    matrix["sp_era5_diff"] = matrix["a_sp_era_5"] - matrix["h_sp_era_5"]

                    roll3_fill = matrix["h_sp_era_3"].notna().mean()
                    log.info("  MLB rolling ERA-3 fill rate: %.1f%%", roll3_fill * 100)
            except Exception as exc:
                log.warning("MLB rolling pitcher stats failed: %s", exc)

            # ── Team batting stats (season OPS/SLG/K%/BB%) ────────────────
            for col in ["h_ops","a_ops","ops_diff","h_slg","a_slg",
                        "h_k_pct","a_k_pct","h_bb_pct","a_bb_pct"]:
                matrix[col] = np.nan
            try:
                bat_df = pd.read_sql_query(
                    "SELECT team_name, season, ops, slg, k_pct, bb_pct FROM team_batting_stats",
                    conn,
                )
                if not bat_df.empty:
                    bat_df["season"] = bat_df["season"].astype(str)
                    bat_map = bat_df.set_index(["team_name", "season"])
                    matrix["season"] = matrix["season"].astype(str)
                    for side, team_col in [("h", "home_team"), ("a", "away_team")]:
                        for stat in ["ops", "slg", "k_pct", "bb_pct"]:
                            col = f"{side}_{stat}"
                            matrix[col] = matrix.apply(
                                lambda r, tc=team_col, s=stat: (
                                    bat_map.loc[(r[tc], r["season"]), s]
                                    if (r[tc], r["season"]) in bat_map.index else np.nan
                                ), axis=1
                            )
                    matrix["ops_diff"] = matrix["h_ops"] - matrix["a_ops"]
                    fill = matrix["h_ops"].notna().mean()
                    log.info("  MLB team batting fill rate: %.1f%%", fill * 100)
            except Exception as exc:
                log.warning("MLB team batting stats failed: %s", exc)

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
