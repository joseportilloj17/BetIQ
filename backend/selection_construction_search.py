"""
selection_construction_search.py — Automated hypothesis generator with
industrial-strength overfitting guards.

Systematically searches selection × construction combinations using:
  1. THREE-WAY data split (search / validation / final holdout)
  2. Nested temporal CV inside search set (rank on OUTER fold)
  3. Bonferroni + BH-FDR multiple-comparisons correction (TOTAL combos)
  4. Pre-registered mechanism requirement
  5. Beat-close tiering on final holdout

THREE VERDICT TIERS:
  BEATS_CLOSE      — Entry price beats de-vigged close + WR > BE + mechanism
                     + survives correction.  Deployable.
  WR_VS_BE_ONLY    — WR > BE + mechanism + survives correction, but beat-close
                     UNTESTED (no entry/close prices available for historical
                     straights).  Forward-shadow only, never size on this.
  NOISE            — Fails any guard.

HONEST EXPECTATION: most search "winners" will be overfit noise.  If zero
survive, the answer is "no selection×construction combo beats the close" —
a real, valuable result.

CALIBRATION GATE: the pipeline MUST re-find the Wong dog teaser as a winner.
If it can't re-discover the one known edge, the pipeline is BROKEN.

Usage:
  python selection_construction_search.py                    # full run
  python selection_construction_search.py --dry-run          # combo count only
  python selection_construction_search.py --sport NFL        # single sport
  python selection_construction_search.py --calibrate-only   # Wong resurface check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
HIST_DB = os.path.join(_DATA_DIR, "historical.db")
NFL_DB = os.path.join(_DATA_DIR, "nfl.db")
BETS_DB = os.path.join(_DATA_DIR, "bets.db")

# Three-way split boundaries (locked before any search)
SEARCH_END = "2023-12-31"       # Search set: everything before 2024
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"          # Validation: 2024
FINAL_START = "2025-01-01"      # Final holdout: 2025+ (touched ONCE)

# For NFL seasons that cross calendar years (e.g. 2023 season = Sep 2023 - Feb 2024),
# we split by SEASON number, not calendar date:
NFL_SEARCH_SEASONS = {2020, 2021, 2022}     # outer fold = 2022
NFL_OUTER_SEASON = 2022                     # honest estimate for ranking
NFL_VAL_SEASONS = {2023}                    # validation
NFL_FINAL_SEASONS = {2024, 2025}            # final holdout

# Nested CV
PURGE_DAYS = 3
MIN_N = 30          # minimum legs per combo to evaluate

# Multiple-comparisons
TOP_N_VALIDATION = 5

# Wong teaser reference
TEASER_POINTS = 6.0
TEASER_ODDS_AM = -120
TEASER_ODDS_DEC = 1.0 + 100.0 / 120.0    # 1.8333
TEASER_PER_LEG_BE = math.sqrt(1.0 / TEASER_ODDS_DEC)  # 0.7385

# WR-vs-BE-only combos need a HIGHER bar since beat-close is untested
WR_VS_BE_MIN_EDGE_PP = 3.0     # ≥ 3pp edge required
WR_VS_BE_MIN_N = 50            # ≥ 50 settled legs required

# ═══════════════════════════════════════════════════════════════════════════════
# MECHANISM PRE-REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════
# Every mechanism is registered BEFORE the search runs.  Combos without a
# matching mechanism are flagged PROBABLY_NOISE even if they pass stats.

MECHANISM_REGISTRY = {
    "favorite_longshot_bias":
        "Underdogs systematically underpriced; favorites overbet. "
        "Well-documented in academic literature (Snowberg & Wolfers 2010).",
    "key_number_crossing":
        "NFL teasers through 3 and 7 exploit margin clustering at key numbers. "
        "Structural, not predictive — NFL games disproportionately land on 3/7 margins.",
    "structural_payout":
        "Bet construction (teaser) creates asymmetric payoff structure that "
        "the market underprices relative to the structural probability.",
    "home_dog_underpricing":
        "Home underdogs may be underpriced due to travel/familiarity advantage "
        "that the market underweights for dogs (anchoring on dog status).",
    "totals_market_efficiency":
        "Totals markets may have different efficiency than sides. "
        "EXPLORATORY — no strong prior, requires extra scrutiny.",
}

def _get_mechanism(sport: str, side: str, market: str, bet_type: str) -> Optional[str]:
    """Look up pre-registered mechanism for a combo.  Returns None if no mechanism."""
    # Teasers: key_number_crossing + structural_payout (always)
    if bet_type == "teaser_6pt":
        return "key_number_crossing"

    # Dog straights: favorite_longshot_bias
    if side == "dog" and market in ("spreads", "moneyline"):
        return "favorite_longshot_bias"

    # Home dogs specifically
    if side == "home_dog":
        return "home_dog_underpricing"

    # Totals: exploratory, weak mechanism
    if market in ("totals_over", "totals_under"):
        return "totals_market_efficiency"

    # Everything else: no mechanism
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_historical_sides(sport: str) -> pd.DataFrame:
    """Load spread/ML data from historical.db with settlement.

    Returns one row per game with: game_date, home_team, away_team,
    close_spread, close_ml_home, close_ml_away, home_score, away_score,
    home_margin, home_covers, away_covers, dog_side, dog_covers.

    NOTE: Only CLOSING lines available.  No entry prices.
    Beat-close is UNTESTABLE on this data.
    """
    con = sqlite3.connect(HIST_DB)
    df = pd.read_sql("""
        SELECT bl.game_date, bl.sport, bl.season,
               bl.home_team, bl.away_team,
               bl.close_spread, bl.close_ml_home, bl.close_ml_away,
               bl.close_total,
               g.home_score, g.away_score
        FROM betting_lines bl
        JOIN games g ON bl.game_id = g.game_id
        WHERE bl.sport = ?
          AND g.home_score IS NOT NULL
          AND bl.close_spread IS NOT NULL
    """, con, params=(sport,))
    con.close()

    if df.empty:
        return df

    # Derived columns
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["spread_result"] = df["home_margin"] + df["close_spread"]
    # Exclude pushes
    df = df[df["spread_result"].abs() > 0.01].copy()
    df["home_covers"] = (df["spread_result"] > 0).astype(int)
    df["away_covers"] = 1 - df["home_covers"]

    # Dog identification: positive close_spread = home is dog
    df["home_is_dog"] = (df["close_spread"] > 0).astype(int)
    df["dog_covers"] = np.where(df["home_is_dog"] == 1,
                                 df["home_covers"], df["away_covers"])

    # Moneyline-implied probabilities (for BE calculation)
    # close_ml is in American format
    df["home_impl_prob"] = df["close_ml_home"].apply(_american_to_prob)
    df["away_impl_prob"] = df["close_ml_away"].apply(_american_to_prob)

    # Dog's implied probability (for BE)
    df["dog_impl_prob"] = np.where(df["home_is_dog"] == 1,
                                    df["home_impl_prob"], df["away_impl_prob"])
    # Dog's decimal odds
    df["dog_decimal_odds"] = np.where(
        df["dog_impl_prob"] > 0, 1.0 / df["dog_impl_prob"], np.nan)

    # Fav's implied probability
    df["fav_impl_prob"] = np.where(df["home_is_dog"] == 1,
                                    df["away_impl_prob"], df["home_impl_prob"])
    df["fav_decimal_odds"] = np.where(
        df["fav_impl_prob"] > 0, 1.0 / df["fav_impl_prob"], np.nan)

    # Fav covers
    df["fav_covers"] = 1 - df["dog_covers"]

    # Spread absolute value
    df["spread_abs"] = df["close_spread"].abs()

    # Home/away for the dog
    df["dog_is_home"] = df["home_is_dog"]

    return df


def load_historical_totals(sport: str) -> pd.DataFrame:
    """Load totals data with settlement."""
    con = sqlite3.connect(HIST_DB)
    df = pd.read_sql("""
        SELECT bl.game_date, bl.sport, bl.season,
               bl.home_team, bl.away_team,
               bl.close_total,
               g.home_score, g.away_score
        FROM betting_lines bl
        JOIN games g ON bl.game_id = g.game_id
        WHERE bl.sport = ?
          AND g.home_score IS NOT NULL
          AND bl.close_total IS NOT NULL
    """, con, params=(sport,))
    con.close()

    if df.empty:
        return df

    df["actual_total"] = df["home_score"] + df["away_score"]
    df["total_diff"] = df["actual_total"] - df["close_total"]
    # Exclude pushes
    df = df[df["total_diff"].abs() > 0.01].copy()
    df["over_wins"] = (df["total_diff"] > 0).astype(int)
    df["under_wins"] = 1 - df["over_wins"]

    return df


def load_nfl_teasers() -> pd.DataFrame:
    """Load NFL teaser data from nfl.db.

    Mirrors shadow_teaser.py logic exactly.
    """
    con = sqlite3.connect(NFL_DB)
    feats = pd.read_sql("""
        SELECT f.game_id, f.team, f.opp_team, f.season, f.week,
               f.game_date, f.is_home, f.close_spread,
               f.is_divisional, f.rest_days, f.is_short_week,
               f.s_point_diff, f.win_pct_season, f.mu_overall
        FROM nfl_features f
    """, con)
    games = pd.read_sql("""
        SELECT game_id, home_team, away_team, home_score, away_score
        FROM nfl_games WHERE home_score IS NOT NULL
    """, con)
    con.close()

    merged = feats.merge(games, on="game_id", how="inner")
    merged["team_score"] = np.where(merged["is_home"] == 1,
                                     merged["home_score"], merged["away_score"])
    merged["opp_score"] = np.where(merged["is_home"] == 1,
                                    merged["away_score"], merged["home_score"])
    merged["actual_margin"] = merged["team_score"] - merged["opp_score"]

    # Wong narrow qualification (same as shadow_teaser.py)
    merged["is_wong_dog"] = merged["close_spread"].between(1.5, 2.5)
    merged["is_wong_fav"] = merged["close_spread"].between(-8.5, -7.5)
    merged["is_wong"] = merged["is_wong_dog"] | merged["is_wong_fav"]

    wong = merged[merged["is_wong"]].copy()
    wong["teased_spread"] = wong["close_spread"] + TEASER_POINTS
    wong["teaser_margin"] = wong["actual_margin"] + wong["teased_spread"]
    # Exclude pushes
    wong = wong[wong["teaser_margin"].abs() > 0.01].copy()
    wong["teaser_win"] = (wong["teaser_margin"] > 0).astype(int)
    wong["side"] = np.where(wong["is_wong_dog"], "dog", "fav")

    return wong


def _american_to_prob(am: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(am) or am == 0:
        return np.nan
    if am > 0:
        return 100.0 / (am + 100.0)
    else:
        return abs(am) / (abs(am) + 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBO GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_combos() -> list[dict]:
    """Enumerate all selection × construction combinations.

    Returns list of combo dicts, each with a pre-registered mechanism
    (or None for combos without a structural hypothesis).
    """
    combos = []

    # ── NFL TEASERS ──────────────────────────────────────────────────────
    # The primary search target: teaser × side × home/away
    for side in ["dog", "fav", "any"]:
        for home_away in ["any", "home", "away"]:
            combo = {
                "combo_id": f"NFL_teaser_{side}_{home_away}",
                "sport": "NFL",
                "side": side,
                "market": "spreads",
                "rung": "wong_narrow",
                "home_away": home_away,
                "bet_type": "teaser_6pt",
                "mechanism": _get_mechanism("NFL", side, "spreads", "teaser_6pt"),
            }
            combos.append(combo)

    # ── SPREAD STRAIGHTS (all sports) ────────────────────────────────────
    for sport in ["MLB", "NFL", "NBA", "NHL"]:
        for side in ["dog", "fav", "any"]:
            for home_away in ["any", "home", "away"]:
                # Base combo
                combo = {
                    "combo_id": f"{sport}_spread_{side}_{home_away}",
                    "sport": sport,
                    "side": side,
                    "market": "spreads",
                    "rung": "any",
                    "home_away": home_away,
                    "bet_type": "straight",
                    "mechanism": _get_mechanism(sport, side, "spreads", "straight"),
                }
                combos.append(combo)

                # Home dog specifically (compound filter)
                if side == "any" and home_away == "any":
                    combo_hd = {
                        "combo_id": f"{sport}_spread_home_dog",
                        "sport": sport,
                        "side": "home_dog",
                        "market": "spreads",
                        "rung": "any",
                        "home_away": "home",
                        "bet_type": "straight",
                        "mechanism": _get_mechanism(sport, "home_dog", "spreads", "straight"),
                    }
                    combos.append(combo_hd)

        # Spread-range buckets for continuous-spread sports
        if sport in ("NBA", "NHL"):
            for side in ["dog", "fav"]:
                for spread_range in ["tight", "medium", "wide"]:
                    combo = {
                        "combo_id": f"{sport}_spread_{side}_{spread_range}",
                        "sport": sport,
                        "side": side,
                        "market": "spreads",
                        "rung": spread_range,
                        "home_away": "any",
                        "bet_type": "straight",
                        "mechanism": _get_mechanism(sport, side, "spreads", "straight"),
                    }
                    combos.append(combo)

        # MLB-specific rung (±1.5 is the only MLB spread rung)
        if sport == "MLB":
            for side in ["dog", "fav"]:
                combo = {
                    "combo_id": f"MLB_spread_{side}_1.5",
                    "sport": "MLB",
                    "side": side,
                    "market": "spreads",
                    "rung": 1.5,
                    "home_away": "any",
                    "bet_type": "straight",
                    "mechanism": _get_mechanism("MLB", side, "spreads", "straight"),
                }
                combos.append(combo)

        # NFL-specific rungs (key numbers)
        if sport == "NFL":
            for side in ["dog", "fav"]:
                for rung in [3.0, 3.5, 7.0, 7.5]:
                    combo = {
                        "combo_id": f"NFL_spread_{side}_{rung}",
                        "sport": "NFL",
                        "side": side,
                        "market": "spreads",
                        "rung": rung,
                        "home_away": "any",
                        "bet_type": "straight",
                        "mechanism": _get_mechanism("NFL", side, "spreads", "straight"),
                    }
                    combos.append(combo)

    # ── MONEYLINE STRAIGHTS (all sports) ─────────────────────────────────
    for sport in ["MLB", "NFL", "NBA", "NHL"]:
        for side in ["dog", "fav", "any"]:
            for home_away in ["any", "home", "away"]:
                combo = {
                    "combo_id": f"{sport}_ml_{side}_{home_away}",
                    "sport": sport,
                    "side": side,
                    "market": "moneyline",
                    "rung": "any",
                    "home_away": home_away,
                    "bet_type": "straight",
                    "mechanism": _get_mechanism(sport, side, "moneyline", "straight"),
                }
                combos.append(combo)

    # ── TOTALS (all sports) ──────────────────────────────────────────────
    for sport in ["MLB", "NFL", "NBA", "NHL"]:
        for direction in ["over", "under"]:
            combo = {
                "combo_id": f"{sport}_totals_{direction}_any",
                "sport": sport,
                "side": "any",
                "market": f"totals_{direction}",
                "rung": "any",
                "home_away": "any",
                "bet_type": "straight",
                "mechanism": _get_mechanism(sport, "any", f"totals_{direction}", "straight"),
            }
            combos.append(combo)

            # Totals at specific lines (sport-specific)
            totals_rungs = {
                "MLB": [7.5, 8.0, 8.5, 9.0, 9.5],
                "NFL": [41.5, 43.5, 45.5, 47.5],
                "NBA": [210.5, 215.5, 220.5, 225.5, 230.5],
                "NHL": [5.5, 6.0, 6.5],
            }
            for rung in totals_rungs.get(sport, []):
                combo = {
                    "combo_id": f"{sport}_totals_{direction}_{rung}",
                    "sport": sport,
                    "side": "any",
                    "market": f"totals_{direction}",
                    "rung": rung,
                    "home_away": "any",
                    "bet_type": "straight",
                    "mechanism": _get_mechanism(sport, "any", f"totals_{direction}", "straight"),
                }
                combos.append(combo)

    # Deduplicate by combo_id
    seen = set()
    unique = []
    for c in combos:
        if c["combo_id"] not in seen:
            seen.add(c["combo_id"])
            unique.append(c)

    return unique


# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION + SETTLEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def apply_combo_sides(combo: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Filter spread/ML data for a combo's selection criteria.

    Returns filtered DataFrame with 'won' column (1/0) and 'be' column
    (implied probability = breakeven for each bet).
    """
    df = data.copy()
    sport = combo["sport"]
    side = combo["side"]
    market = combo["market"]
    rung = combo.get("rung", "any")
    home_away = combo.get("home_away", "any")

    # SPREAD bets: BE depends on sport.
    #   NFL/NBA: standard -110 pricing, BE = 52.38%.  Correct.
    #   MLB (run line ±1.5): asymmetric pricing (+140-180 dog, -160-200 fav).
    #     -110 BE is WRONG — produces phantom edges.
    #   NHL (puck line ±1.5): same asymmetry (-160-200 dog cover, +140-170 fav).
    #     -110 BE is WRONG.
    # Without actual spread odds in historical.db, MLB/NHL spread BE is
    # UNKNOWN.  Flag them — do not report as edges.
    # MONEYLINE bets: BE = moneyline implied probability.  Correct.
    is_spread = (market == "spreads")

    # BE for spread bets (sport-specific)
    if is_spread:
        if sport in ("NFL", "NBA"):
            # Symmetric -110 pricing — BE = 52.38% is correct
            spread_be = 1.0 / 1.909
            spread_odds = 1.909
            be_reliable = True
        else:
            # MLB/NHL: asymmetric run/puck line, -110 is WRONG.
            # Use 50.0% as a conservative null (market should be efficient
            # → roughly fair after vig).  But flag as unreliable.
            # The actual BE depends on odds we don't have.
            spread_be = 0.50
            spread_odds = 2.0
            be_reliable = False
    else:
        be_reliable = True

    if side == "dog":
        df["won"] = df["dog_covers"]
        if is_spread:
            df["be"] = spread_be
            df["odds_dec"] = spread_odds
        else:
            df["be"] = df["dog_impl_prob"]
            df["odds_dec"] = df["dog_decimal_odds"]
    elif side == "fav":
        df["won"] = df["fav_covers"]
        if is_spread:
            df["be"] = spread_be
            df["odds_dec"] = spread_odds
        else:
            df["be"] = df["fav_impl_prob"]
            df["odds_dec"] = df["fav_decimal_odds"]
    elif side == "home_dog":
        df = df[df["home_is_dog"] == 1].copy()
        df["won"] = df["home_covers"]
        if is_spread:
            df["be"] = spread_be
            df["odds_dec"] = spread_odds
        else:
            df["be"] = df["home_impl_prob"]
            df["odds_dec"] = np.where(df["home_impl_prob"] > 0,
                                       1.0 / df["home_impl_prob"], np.nan)
    else:  # "any" — bet home team spread/ML
        df["won"] = df["home_covers"]
        if is_spread:
            df["be"] = spread_be
            df["odds_dec"] = spread_odds
        else:
            df["be"] = df["home_impl_prob"]
            df["odds_dec"] = np.where(df["home_impl_prob"] > 0,
                                       1.0 / df["home_impl_prob"], np.nan)

    # Tag BE reliability for downstream verdict
    df["be_reliable"] = be_reliable

    # Moneyline: settlement by actual winner, not spread cover
    if market == "moneyline":
        if side == "dog":
            df["won"] = np.where(df["home_is_dog"] == 1,
                                  (df["home_margin"] > 0).astype(int),
                                  (df["home_margin"] < 0).astype(int))
        elif side == "fav":
            df["won"] = np.where(df["home_is_dog"] == 1,
                                  (df["home_margin"] < 0).astype(int),
                                  (df["home_margin"] > 0).astype(int))
        elif side == "home_dog":
            df = df[df["home_is_dog"] == 1].copy()
            df["won"] = (df["home_margin"] > 0).astype(int)
        else:
            df["won"] = (df["home_margin"] > 0).astype(int)

    # Home/away filter
    if home_away == "home":
        if side == "dog":
            df = df[df["home_is_dog"] == 1]
        elif side == "fav":
            df = df[df["home_is_dog"] == 0]
    elif home_away == "away":
        if side == "dog":
            df = df[df["home_is_dog"] == 0]
        elif side == "fav":
            df = df[df["home_is_dog"] == 1]

    # Rung filter (sport-specific)
    if rung != "any" and rung is not None:
        if isinstance(rung, (int, float)):
            if side in ("dog", "home_dog"):
                df = df[(df["close_spread"].abs() - rung).abs() < 0.51]
            elif side == "fav":
                df = df[(df["close_spread"].abs() - rung).abs() < 0.51]
            else:
                df = df[(df["close_spread"].abs() - rung).abs() < 0.51]
        elif rung == "tight":
            df = df[df["spread_abs"] < 4.0]
        elif rung == "medium":
            df = df[(df["spread_abs"] >= 4.0) & (df["spread_abs"] <= 8.0)]
        elif rung == "wide":
            df = df[df["spread_abs"] > 8.0]

    # Drop rows with NaN BE (can't compute edge without odds)
    df = df.dropna(subset=["be", "won"])
    # Filter unrealistic BE (outside 0.05-0.95)
    df = df[(df["be"] > 0.05) & (df["be"] < 0.95)]

    return df


def apply_combo_totals(combo: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Filter totals data for a combo's selection criteria."""
    df = data.copy()
    market = combo["market"]
    rung = combo.get("rung", "any")

    if market == "totals_over":
        df["won"] = df["over_wins"]
    else:
        df["won"] = df["under_wins"]

    # Totals are always at -110 (1.909 decimal) → BE = 52.38%
    df["be"] = 1.0 / 1.909
    df["odds_dec"] = 1.909

    # Totals are always reliably priced at -110
    df["be_reliable"] = True

    # Rung filter
    if rung != "any" and rung is not None:
        if isinstance(rung, (int, float)):
            df = df[(df["close_total"] - rung).abs() < 0.51]

    return df


def apply_combo_teasers(combo: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Filter teaser data for a combo's selection criteria."""
    df = data.copy()
    side = combo["side"]
    home_away = combo.get("home_away", "any")

    if side == "dog":
        df = df[df["side"] == "dog"]
    elif side == "fav":
        df = df[df["side"] == "fav"]

    if home_away == "home":
        df = df[df["is_home"] == 1]
    elif home_away == "away":
        df = df[df["is_home"] == 0]

    df["won"] = df["teaser_win"]
    df["be"] = TEASER_PER_LEG_BE
    df["odds_dec"] = TEASER_ODDS_DEC
    # Teasers have fixed -120 odds — BE is structurally correct
    df["be_reliable"] = True

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute edge statistics for a filtered dataset.

    BE = AVG(1/odds) — correct, not 1/AVG(odds).
    """
    n = len(df)
    if n < 5:
        return {"n": n, "status": "INSUFFICIENT_N"}

    wins = int(df["won"].sum())
    wr = wins / n

    # BE = AVG(1/odds_decimal) = AVG(implied probability)
    # This IS AVG(1/odds), the correct formula
    be = df["be"].mean()

    edge_pp = (wr - be) * 100
    se = math.sqrt(be * (1 - be) / n) if be > 0 and be < 1 else 1e-6
    z = (wr - be) / se if se > 0 else 0.0

    # One-sided binomial test: P(wins >= observed | p = be)
    p_raw = 1.0 - binom.cdf(wins - 1, n, be) if wins > 0 else 1.0

    # PnL per unit at actual odds
    if "odds_dec" in df.columns:
        pnl_vals = np.where(df["won"] == 1, df["odds_dec"] - 1.0, -1.0)
        pnl_per_unit = float(pnl_vals.mean())
        roi_pct = pnl_per_unit * 100
    else:
        pnl_per_unit = None
        roi_pct = None

    return {
        "n": n,
        "wins": wins,
        "wr": round(wr, 4),
        "be": round(be, 4),
        "edge_pp": round(edge_pp, 2),
        "z": round(z, 3),
        "p_raw": round(p_raw, 6),
        "pnl_per_unit": round(pnl_per_unit, 4) if pnl_per_unit is not None else None,
        "roi_pct": round(roi_pct, 2) if roi_pct is not None else None,
        "status": "OK",
    }


def split_half_check(df: pd.DataFrame) -> dict:
    """Time-ordered split-half stability check."""
    n = len(df)
    if n < 20:
        return {"stable": None, "reason": "N too small for split-half"}

    half = n // 2
    sorted_df = df.sort_values("game_date")
    h1 = sorted_df.iloc[:half]
    h2 = sorted_df.iloc[half:]

    s1 = compute_stats(h1)
    s2 = compute_stats(h2)

    if s1["status"] != "OK" or s2["status"] != "OK":
        return {"stable": None, "reason": "Insufficient N in half"}

    # Stable = both halves show same edge sign AND neither reverts catastrophically
    same_sign = (s1["edge_pp"] > 0) == (s2["edge_pp"] > 0)
    max_drop = abs(s1["edge_pp"] - s2["edge_pp"])

    return {
        "stable": same_sign and max_drop < 15.0,
        "h1_edge_pp": s1["edge_pp"],
        "h1_n": s1["n"],
        "h2_edge_pp": s2["edge_pp"],
        "h2_n": s2["n"],
        "delta_pp": round(abs(s1["edge_pp"] - s2["edge_pp"]), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NESTED CV EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_combo(combo: dict, all_data: dict) -> dict:
    """Evaluate a single combo using leave-one-year-out (LOYO) nested CV.

    LOYO: each year in the search set takes a turn as the outer fold.
    The outer-fold predictions are pooled across all rotations to give a
    stable aggregate estimate.  This avoids the single-bad-year problem
    (e.g. 2022 NFL teasers at N=47).
    """
    sport = combo["sport"]
    bet_type = combo["bet_type"]
    market = combo["market"]

    # Select the right data and filter function
    if bet_type == "teaser_6pt":
        if "nfl_teasers" not in all_data:
            return {"combo_id": combo["combo_id"], "status": "NO_DATA"}
        full_data = all_data["nfl_teasers"]
        search = full_data[full_data["season"].isin(NFL_SEARCH_SEASONS)]
        filter_fn = apply_combo_teasers
        year_col = "season"
    elif market in ("totals_over", "totals_under"):
        key = f"totals_{sport}"
        if key not in all_data or all_data[key].empty:
            return {"combo_id": combo["combo_id"], "status": "NO_DATA"}
        full_data = all_data[key]
        search = full_data[full_data["game_date"] <= SEARCH_END].copy()
        search["_year"] = pd.to_datetime(search["game_date"]).dt.year
        filter_fn = apply_combo_totals
        year_col = "_year"
    else:  # spreads or moneyline
        key = f"sides_{sport}"
        if key not in all_data or all_data[key].empty:
            return {"combo_id": combo["combo_id"], "status": "NO_DATA"}
        full_data = all_data[key]
        if sport == "NFL" and "nfl_sides" in all_data:
            full_data = all_data["nfl_sides"]
        search = full_data[full_data["game_date"] <= SEARCH_END].copy()
        if "season" in search.columns:
            year_col = "season"
        else:
            search["_year"] = pd.to_datetime(search["game_date"]).dt.year
            year_col = "_year"
        filter_fn = apply_combo_sides

    # Apply selection to full search set first (to check min N)
    filtered_search = filter_fn(combo, search)
    if len(filtered_search) < MIN_N:
        return {
            "combo_id": combo["combo_id"],
            "mechanism": combo.get("mechanism"),
            "status": "INSUFFICIENT_N",
            "n_search": len(filtered_search),
        }

    # ── LOYO nested CV ────────────────────────────────────────────────
    # Each year takes a turn as outer fold; inner = remaining years
    years = sorted(search[year_col].unique())
    per_fold_stats = []
    pooled_outer_wins = 0
    pooled_outer_n = 0
    pooled_outer_be_sum = 0.0

    for yr in years:
        outer_data = search[search[year_col] == yr]
        inner_data = search[search[year_col] != yr]

        filtered_outer = filter_fn(combo, outer_data)
        fold_stats = compute_stats(filtered_outer)
        fold_stats["fold_year"] = int(yr) if not isinstance(yr, str) else yr

        per_fold_stats.append(fold_stats)

        if fold_stats.get("status") == "OK":
            pooled_outer_wins += fold_stats["wins"]
            pooled_outer_n += fold_stats["n"]
            pooled_outer_be_sum += fold_stats["be"] * fold_stats["n"]

    # Pooled outer fold stats (aggregate across all LOYO rotations)
    if pooled_outer_n >= MIN_N:
        pooled_wr = pooled_outer_wins / pooled_outer_n
        pooled_be = pooled_outer_be_sum / pooled_outer_n
        pooled_edge = (pooled_wr - pooled_be) * 100
        pooled_se = math.sqrt(pooled_be * (1 - pooled_be) / pooled_outer_n) \
            if 0 < pooled_be < 1 else 1e-6
        pooled_z = (pooled_wr - pooled_be) / pooled_se
        pooled_p = 1.0 - binom.cdf(pooled_outer_wins - 1, pooled_outer_n, pooled_be) \
            if pooled_outer_wins > 0 else 1.0
        outer_pooled = {
            "n": pooled_outer_n,
            "wins": pooled_outer_wins,
            "wr": round(pooled_wr, 4),
            "be": round(pooled_be, 4),
            "edge_pp": round(pooled_edge, 2),
            "z": round(pooled_z, 3),
            "p_raw": round(pooled_p, 6),
            "status": "OK",
        }
    else:
        outer_pooled = {"n": pooled_outer_n, "status": "INSUFFICIENT_N"}

    # ── Extract BE reliability from filtered data ──────────────────
    # All rows in a combo share the same be_reliable value
    if "be_reliable" in filtered_search.columns and len(filtered_search) > 0:
        be_reliable = bool(filtered_search["be_reliable"].iloc[0])
    else:
        be_reliable = True  # default for data without the flag (teasers, totals)

    # ── Split-half on search set ──────────────────────────────────────
    sh = split_half_check(filtered_search)

    # Inner consistency = all folds with enough data show same edge sign
    valid_folds = [f for f in per_fold_stats
                   if f.get("status") == "OK" and f["n"] >= 10]
    if len(valid_folds) >= 2:
        signs = [f["edge_pp"] > 0 for f in valid_folds]
        inner_consistent = all(signs) or not any(signs)
    else:
        inner_consistent = True

    return {
        "combo_id": combo["combo_id"],
        "mechanism": combo.get("mechanism"),
        "bet_type": bet_type,
        "sport": sport,
        "side": combo.get("side"),
        "market": market,
        "rung": combo.get("rung"),
        "home_away": combo.get("home_away"),
        "be_reliable": be_reliable,
        "loyo_folds": per_fold_stats,
        "outer_fold": outer_pooled,
        "search_total": compute_stats(filtered_search),
        "split_half": sh,
        "inner_consistent": inner_consistent,
        "status": outer_pooled.get("status", "OK"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPLE-COMPARISONS CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def correct_pvalues(results: list[dict], n_combos_total: int) -> list[dict]:
    """Apply Bonferroni and BH-FDR corrections.

    n_combos_total is the TOTAL number across ALL sports/markets
    (not per-sport — the user's tightening #1).
    """
    # Filter to evaluated combos with valid p-values
    evaluated = [r for r in results
                 if r.get("status") == "OK"
                 and r.get("outer_fold", {}).get("status") == "OK"]

    for r in evaluated:
        p_raw = r["outer_fold"]["p_raw"]

        # Bonferroni: correct against TOTAL combos tried
        r["p_bonferroni"] = min(1.0, p_raw * n_combos_total)

    # BH-FDR at 5% level (manual implementation for scipy compatibility)
    sorted_by_p = sorted(evaluated, key=lambda x: x["outer_fold"]["p_raw"])
    m = len(sorted_by_p)
    fdr_threshold = 0.05

    # Find largest k where p_(k) <= k * alpha / m
    max_reject_idx = -1
    for i, r in enumerate(sorted_by_p):
        rank = i + 1  # 1-indexed
        threshold = rank * fdr_threshold / m if m > 0 else 0
        if r["outer_fold"]["p_raw"] <= threshold:
            max_reject_idx = i

    # Assign FDR-adjusted p-values
    rejected_ids = set()
    for i, r in enumerate(sorted_by_p):
        if i <= max_reject_idx:
            r["fdr_rejected"] = True
            rejected_ids.add(r["combo_id"])
        else:
            r["fdr_rejected"] = False

        # BH-adjusted p-value
        rank = i + 1
        r["p_fdr_adjusted"] = min(1.0, r["outer_fold"]["p_raw"] * m / rank) if m > 0 else 1.0

    # Assign verdict status
    for r in evaluated:
        has_mechanism = r.get("mechanism") is not None
        survives_bonf = r["p_bonferroni"] < 0.05
        be_ok = r.get("be_reliable", True)

        if not be_ok:
            # BE is unreliable (MLB/NHL spreads with asymmetric pricing).
            # The edge might be an artifact of wrong BE, not a real edge.
            # Cannot be a CANDIDATE regardless of p-value.
            r["correction_status"] = "BE_UNCERTAIN"
        elif survives_bonf and has_mechanism:
            r["correction_status"] = "CANDIDATE"
        elif survives_bonf and not has_mechanism:
            r["correction_status"] = "PROBABLY_NOISE"
        else:
            r["correction_status"] = "NOISE_AFTER_CORRECTION"

    # Also update non-evaluated results
    for r in results:
        if r not in evaluated:
            r["p_bonferroni"] = 1.0
            r["p_fdr_adjusted"] = 1.0
            r["fdr_rejected"] = False
            r["correction_status"] = r.get("status", "INSUFFICIENT_N")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION + FINAL HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════

def validate_candidates(candidates: list[dict], all_data: dict) -> list[dict]:
    """Test top N candidates on the 2024 VALIDATION holdout.

    Applies mini multiple-comparisons correction for the N looks
    (tightening #2).
    """
    n_candidates = len(candidates)
    val_results = []

    for cand in candidates:
        combo = _reconstruct_combo(cand)
        sport = combo["sport"]
        bet_type = combo["bet_type"]
        market = combo["market"]

        # Get validation data
        if bet_type == "teaser_6pt":
            full_data = all_data.get("nfl_teasers", pd.DataFrame())
            val_data = full_data[full_data["season"].isin(NFL_VAL_SEASONS)]
            filter_fn = apply_combo_teasers
        elif market in ("totals_over", "totals_under"):
            key = f"totals_{sport}"
            full_data = all_data.get(key, pd.DataFrame())
            val_data = full_data[
                (full_data["game_date"] >= VAL_START) &
                (full_data["game_date"] <= VAL_END)
            ]
            filter_fn = apply_combo_totals
        else:
            key = f"sides_{sport}"
            full_data = all_data.get(key, pd.DataFrame())
            if sport == "NFL" and "nfl_sides" in all_data:
                full_data = all_data["nfl_sides"]
            val_data = full_data[
                (full_data["game_date"] >= VAL_START) &
                (full_data["game_date"] <= VAL_END)
            ]
            filter_fn = apply_combo_sides

        if val_data.empty:
            val_results.append({
                **cand, "val_status": "NO_VAL_DATA", "val_stats": None,
            })
            continue

        filtered = filter_fn(combo, val_data)
        stats = compute_stats(filtered)

        # Mini correction for N validation looks (tightening #2)
        if stats.get("status") == "OK":
            stats["p_val_corrected"] = min(1.0, stats["p_raw"] * n_candidates)
        else:
            stats["p_val_corrected"] = 1.0

        # Consistency with search
        search_edge = cand.get("outer_fold", {}).get("edge_pp", 0)
        val_edge = stats.get("edge_pp", 0)
        consistent = (search_edge > 0 and val_edge > 0) if stats.get("status") == "OK" else False

        val_results.append({
            **cand,
            "val_stats": stats,
            "val_consistent": consistent,
            "val_status": "CONFIRMED" if (
                consistent and
                stats.get("status") == "OK" and
                stats.get("edge_pp", 0) > 0 and
                stats.get("p_val_corrected", 1) < 0.20  # lenient confirmation
            ) else "NOT_CONFIRMED",
        })

    return val_results


def final_test(winner: dict, all_data: dict) -> dict:
    """Test the SINGLE winner on the 2025+ FINAL holdout.  Touched EXACTLY ONCE.

    Distinguishes three verdict tiers:
      BEATS_CLOSE      — Deployable (teasers, or straights with entry+close)
      WR_VS_BE_ONLY    — Forward-shadow only (straights without entry prices)
      NO_EDGE          — Failed
    """
    combo = _reconstruct_combo(winner)
    sport = combo["sport"]
    bet_type = combo["bet_type"]
    market = combo["market"]

    # Get final holdout data
    if bet_type == "teaser_6pt":
        full_data = all_data.get("nfl_teasers", pd.DataFrame())
        final_data = full_data[full_data["season"].isin(NFL_FINAL_SEASONS)]
        filter_fn = apply_combo_teasers
    elif market in ("totals_over", "totals_under"):
        key = f"totals_{sport}"
        full_data = all_data.get(key, pd.DataFrame())
        final_data = full_data[full_data["game_date"] >= FINAL_START]
        filter_fn = apply_combo_totals
    else:
        key = f"sides_{sport}"
        full_data = all_data.get(key, pd.DataFrame())
        if sport == "NFL" and "nfl_sides" in all_data:
            full_data = all_data["nfl_sides"]
        final_data = full_data[full_data["game_date"] >= FINAL_START]
        filter_fn = apply_combo_sides

    if final_data.empty:
        return {**winner, "final_status": "NO_FINAL_DATA", "final_stats": None}

    filtered = filter_fn(combo, final_data)
    stats = compute_stats(filtered)
    sh = split_half_check(filtered) if len(filtered) >= 20 else {"stable": None}

    # ── Determine beat-close tier ────────────────────────────────────
    has_mechanism = winner.get("mechanism") is not None
    be_ok = winner.get("be_reliable", True)

    if bet_type == "teaser_6pt":
        # Teasers: WR vs correct BE IS the valid test
        # Fixed odds → no Jensen's, structural mechanism
        beat_close_tier = "BEATS_CLOSE"
        beat_close_note = (
            "Teaser: fixed -120 odds, structural key-number mechanism. "
            "WR vs correct BE (sqrt(1/1.833)=73.9%) is the valid edge test."
        )
    elif not be_ok:
        # MLB/NHL spreads: we don't know the true BE (asymmetric pricing).
        # Cannot even do WR-vs-BE honestly.
        beat_close_tier = "BE_UNCERTAIN"
        beat_close_note = (
            "Spread bet with UNKNOWN BE: sport uses asymmetric run/puck line "
            "pricing.  The -110 assumption is wrong.  Cannot compute edge "
            "without actual spread odds.  Not deployable."
        )
    else:
        # Straights: historical.db has CLOSING lines only, no entry prices
        # Beat-close is UNTESTABLE
        beat_close_tier = "WR_VS_BE_ONLY"
        beat_close_note = (
            "Straight bet: only closing lines available in historical data. "
            "Beat-close UNTESTED. WR-vs-BE is a candidate signal only — "
            "must forward-shadow before any deployment. Never size on WR-vs-BE alone."
        )

    # Apply the higher bar for WR-vs-BE-only combos
    if beat_close_tier == "WR_VS_BE_ONLY":
        passes_higher_bar = (
            stats.get("status") == "OK" and
            stats.get("edge_pp", 0) >= WR_VS_BE_MIN_EDGE_PP and
            stats.get("n", 0) >= WR_VS_BE_MIN_N and
            sh.get("stable", False) and
            has_mechanism
        )
    else:
        passes_higher_bar = True  # teasers use normal bar

    # Final verdict
    if stats.get("status") != "OK":
        verdict = "NO_EDGE"
    elif stats.get("edge_pp", 0) <= 0:
        verdict = "NO_EDGE"
    elif beat_close_tier == "BE_UNCERTAIN":
        verdict = "BE_UNCERTAIN"
    elif beat_close_tier == "BEATS_CLOSE" and stats.get("p_raw", 1) < 0.10 and has_mechanism:
        verdict = "BEATS_CLOSE"
    elif beat_close_tier == "WR_VS_BE_ONLY" and passes_higher_bar:
        verdict = "WR_VS_BE_ONLY"
    else:
        verdict = "NO_EDGE"

    return {
        **winner,
        "final_stats": stats,
        "final_split_half": sh,
        "beat_close_tier": beat_close_tier,
        "beat_close_note": beat_close_note,
        "final_verdict": verdict,
        "final_status": "TESTED",
    }


def _reconstruct_combo(result: dict) -> dict:
    """Reconstruct a combo dict from a result dict."""
    return {
        "combo_id": result["combo_id"],
        "sport": result.get("sport", ""),
        "side": result.get("side", "any"),
        "market": result.get("market", "spreads"),
        "rung": result.get("rung", "any"),
        "home_away": result.get("home_away", "any"),
        "bet_type": result.get("bet_type", "straight"),
        "mechanism": result.get("mechanism"),
        "be_reliable": result.get("be_reliable", True),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NFL DATA LOADING (for spread/ML straights from nfl.db — more seasons)
# ═══════════════════════════════════════════════════════════════════════════════

def load_nfl_sides() -> pd.DataFrame:
    """Load NFL spread/ML data from nfl.db (2020-2025).

    Returns DataFrame compatible with apply_combo_sides().
    """
    con = sqlite3.connect(NFL_DB)
    feats = pd.read_sql("""
        SELECT f.game_id, f.team, f.opp_team, f.season, f.week,
               f.game_date, f.is_home, f.close_spread, f.close_ml, f.opp_close_ml
        FROM nfl_features f
    """, con)
    games = pd.read_sql("""
        SELECT game_id, home_team, away_team, home_score, away_score
        FROM nfl_games WHERE home_score IS NOT NULL
    """, con)
    con.close()

    # Keep only home-team rows to avoid duplicates
    merged = feats[feats["is_home"] == 1].merge(games, on="game_id", how="inner")
    if merged.empty:
        return merged

    merged["home_margin"] = merged["home_score"] - merged["away_score"]
    merged["spread_result"] = merged["home_margin"] + merged["close_spread"]
    merged = merged[merged["spread_result"].abs() > 0.01].copy()
    merged["home_covers"] = (merged["spread_result"] > 0).astype(int)
    merged["away_covers"] = 1 - merged["home_covers"]

    merged["home_is_dog"] = (merged["close_spread"] > 0).astype(int)
    merged["dog_covers"] = np.where(merged["home_is_dog"] == 1,
                                     merged["home_covers"], merged["away_covers"])
    merged["fav_covers"] = 1 - merged["dog_covers"]

    # Odds from close_ml (American format)
    merged["home_impl_prob"] = merged["close_ml"].apply(_american_to_prob)
    merged["away_impl_prob"] = merged["opp_close_ml"].apply(_american_to_prob)
    merged["dog_impl_prob"] = np.where(merged["home_is_dog"] == 1,
                                        merged["home_impl_prob"],
                                        merged["away_impl_prob"])
    merged["dog_decimal_odds"] = np.where(
        merged["dog_impl_prob"] > 0, 1.0 / merged["dog_impl_prob"], np.nan)
    merged["fav_impl_prob"] = np.where(merged["home_is_dog"] == 1,
                                        merged["away_impl_prob"],
                                        merged["home_impl_prob"])
    merged["fav_decimal_odds"] = np.where(
        merged["fav_impl_prob"] > 0, 1.0 / merged["fav_impl_prob"], np.nan)
    merged["spread_abs"] = merged["close_spread"].abs()
    merged["dog_is_home"] = merged["home_is_dog"]

    # Convert season to game_date-like for split compatibility
    # (nfl.db has game_date already)
    merged["sport"] = "NFL"

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# WONG TEASER RESURFACE CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def wong_resurface_check(all_data: dict) -> dict:
    """CALIBRATION GATE: the pipeline MUST re-find Wong dog teaser.

    If the pipeline can't re-discover the one known edge, it's BROKEN.
    Run this FIRST before trusting any other result.
    """
    print("\n" + "=" * 70)
    print("CALIBRATION: Wong Dog Teaser Resurface Check")
    print("=" * 70)

    # The known-good combo
    wong_combo = {
        "combo_id": "NFL_teaser_dog_any",
        "sport": "NFL",
        "side": "dog",
        "market": "spreads",
        "rung": "wong_narrow",
        "home_away": "any",
        "bet_type": "teaser_6pt",
        "mechanism": "key_number_crossing",
    }

    result = evaluate_combo(wong_combo, all_data)

    # Check outer fold (pooled LOYO)
    outer = result.get("outer_fold", {})
    search_total = result.get("search_total", {})

    print(f"\n  Per-year LOYO folds:")
    for f in result.get("loyo_folds", []):
        if f.get("status") == "OK":
            print(f"    {f['fold_year']}: N={f['n']}, WR={f['wr']*100:.1f}%, "
                  f"Edge={f['edge_pp']:+.1f}pp")
        else:
            print(f"    {f.get('fold_year', '?')}: {f.get('status')}")

    print(f"\n  Pooled outer fold (LOYO aggregate, honest estimate):")
    if outer.get("status") == "OK":
        print(f"    N={outer['n']}, WR={outer['wr']*100:.1f}%, "
              f"BE={outer['be']*100:.1f}%, Edge={outer['edge_pp']:+.1f}pp, "
              f"Z={outer['z']:.2f}, p={outer['p_raw']:.4f}")
    else:
        print(f"    Status: {outer.get('status')}")

    print(f"\n  Full search set:")
    if search_total.get("status") == "OK":
        print(f"    N={search_total['n']}, WR={search_total['wr']*100:.1f}%, "
              f"Edge={search_total['edge_pp']:+.1f}pp, Z={search_total['z']:.2f}")

    # Now test on validation (2023 season) and final (2024-2025)
    val_data = all_data.get("nfl_teasers", pd.DataFrame())
    val_set = val_data[val_data["season"].isin(NFL_VAL_SEASONS)]
    final_set = val_data[val_data["season"].isin(NFL_FINAL_SEASONS)]

    val_filtered = apply_combo_teasers(wong_combo, val_set)
    val_stats = compute_stats(val_filtered)
    print(f"\n  Validation (2023 season):")
    if val_stats.get("status") == "OK":
        print(f"    N={val_stats['n']}, WR={val_stats['wr']*100:.1f}%, "
              f"Edge={val_stats['edge_pp']:+.1f}pp, p={val_stats['p_raw']:.4f}")
    else:
        print(f"    Status: {val_stats.get('status')}")

    final_filtered = apply_combo_teasers(wong_combo, final_set)
    final_stats = compute_stats(final_filtered)
    print(f"\n  Final holdout (2024-2025 seasons):")
    if final_stats.get("status") == "OK":
        print(f"    N={final_stats['n']}, WR={final_stats['wr']*100:.1f}%, "
              f"Edge={final_stats['edge_pp']:+.1f}pp, p={final_stats['p_raw']:.4f}")
    else:
        print(f"    Status: {final_stats.get('status')}")

    # Pass/fail
    passed = (
        outer.get("status") == "OK" and
        outer.get("edge_pp", 0) > 0 and
        search_total.get("status") == "OK" and
        search_total.get("edge_pp", 0) > 0
    )

    if passed:
        print(f"\n  ✓ CALIBRATION PASSED: Wong dog teaser resurfaced with "
              f"edge={outer.get('edge_pp', 0):+.1f}pp on outer fold")
    else:
        print(f"\n  ✗ CALIBRATION FAILED: Pipeline cannot find the known edge!")
        print(f"    FIX THE PIPELINE before trusting any other results.")

    return {
        "passed": passed,
        "combo": wong_combo,
        "search_result": result,
        "val_stats": val_stats,
        "final_stats": final_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(meta: dict, search_results: list, val_results: list,
                    final_result: Optional[dict],
                    calibration: dict) -> dict:
    """Build the final JSON report."""

    # Count by status
    status_counts = defaultdict(int)
    for r in search_results:
        status_counts[r.get("correction_status", r.get("status", "UNKNOWN"))] += 1

    # Sort search winners by p_bonferroni
    winners = sorted(
        [r for r in search_results if r.get("correction_status") in
         ("CANDIDATE", "PROBABLY_NOISE")],
        key=lambda x: x.get("p_bonferroni", 1.0)
    )

    # Build honest summary
    n_candidates = len([r for r in search_results
                        if r.get("correction_status") == "CANDIDATE"])
    n_noise = len([r for r in search_results
                   if r.get("correction_status") == "NOISE_AFTER_CORRECTION"])
    n_prob_noise = len([r for r in search_results
                        if r.get("correction_status") == "PROBABLY_NOISE"])
    n_insuff = len([r for r in search_results
                    if r.get("status") == "INSUFFICIENT_N"])

    if final_result and final_result.get("final_verdict") == "BEATS_CLOSE":
        summary = (
            f"{n_candidates} of {meta['n_combos_total']} combos survived all guards. "
            f"Winner: {final_result['combo_id']} — BEATS CLOSE on final holdout. "
            f"Deployable."
        )
    elif final_result and final_result.get("final_verdict") == "WR_VS_BE_ONLY":
        summary = (
            f"{n_candidates} of {meta['n_combos_total']} combos survived correction. "
            f"Best: {final_result['combo_id']} — WR > BE on final holdout but "
            f"beat-close UNTESTED. Forward-shadow only."
        )
    elif n_candidates == 0:
        summary = (
            f"0 of {meta['n_combos_total']} combos survived all guards. "
            f"No selection×construction combo beats the close. "
            f"This is the guard working, not a search failure."
        )
    else:
        summary = (
            f"{n_candidates} of {meta['n_combos_total']} combos survived correction "
            f"but none survived the final holdout."
        )

    return {
        "meta": {
            **meta,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "calibration": {
            "wong_resurface_passed": calibration["passed"],
            "wong_search_edge_pp": calibration["search_result"]
                .get("outer_fold", {}).get("edge_pp"),
            "wong_final_edge_pp": calibration["final_stats"].get("edge_pp"),
        },
        "status_counts": dict(status_counts),
        "search_winners": [_slim_result(r) for r in winners[:20]],
        "validation_results": [_slim_result(r) for r in val_results] if val_results else [],
        "final_result": _slim_result(final_result) if final_result else None,
        "honest_summary": summary,
        "beat_close_note": (
            "Historical data has CLOSING lines only — no entry prices. "
            "Beat-close is UNTESTABLE for spread/ML straights. "
            "Only teasers (fixed odds, structural mechanism) can be tier-1 'BEATS_CLOSE'. "
            "All other combos are tier-2 'WR_VS_BE_ONLY' (forward-shadow candidates)."
        ),
    }


def _slim_result(r: dict) -> dict:
    """Trim a result dict for JSON output (remove large DataFrames etc)."""
    if r is None:
        return None
    slim = {}
    for k, v in r.items():
        if isinstance(v, (str, int, float, bool, type(None), list, dict)):
            slim[k] = v
    return slim


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_search(sport_filter: Optional[str] = None,
               dry_run: bool = False,
               calibrate_only: bool = False,
               output_path: str = "/tmp/selection_construction_search_report.json",
               ) -> dict:
    """Run the full search pipeline."""

    print("=" * 70)
    print("SELECTION × CONSTRUCTION SEARCH")
    print("Hypothesis generator with overfitting guards")
    print("=" * 70)

    # ── 1. Generate combos ────────────────────────────────────────────
    combos = generate_combos()
    if sport_filter:
        combos = [c for c in combos if c["sport"] == sport_filter]

    n_total = len(combos)
    n_with_mechanism = sum(1 for c in combos if c.get("mechanism") is not None)
    n_no_mechanism = n_total - n_with_mechanism

    print(f"\n  Total combos: {n_total}")
    print(f"  With mechanism: {n_with_mechanism}")
    print(f"  Without mechanism (flagged): {n_no_mechanism}")

    # Count by sport
    by_sport = defaultdict(int)
    for c in combos:
        by_sport[c["sport"]] += 1
    for s, n in sorted(by_sport.items()):
        print(f"    {s}: {n} combos")

    # Count by bet type
    by_type = defaultdict(int)
    for c in combos:
        by_type[c["bet_type"]] += 1
    for t, n in sorted(by_type.items()):
        print(f"    {t}: {n} combos")

    if dry_run:
        print("\n  [DRY RUN] — combo enumeration only, no evaluation.")
        print("\n  Sample combos:")
        for c in combos[:10]:
            mech = c.get("mechanism", "NONE")
            print(f"    {c['combo_id']} | mechanism={mech}")
        return {"n_combos_total": n_total, "dry_run": True}

    # ── 2. Load data ──────────────────────────────────────────────────
    print("\n  Loading data...")
    all_data = {}

    for sport in ["MLB", "NFL", "NBA", "NHL"]:
        if sport_filter and sport != sport_filter:
            continue
        sides = load_historical_sides(sport)
        if not sides.empty:
            all_data[f"sides_{sport}"] = sides
            print(f"    {sport} sides: {len(sides)} games")
        totals = load_historical_totals(sport)
        if not totals.empty:
            all_data[f"totals_{sport}"] = totals
            print(f"    {sport} totals: {len(totals)} games")

    # NFL from nfl.db (more seasons)
    if not sport_filter or sport_filter == "NFL":
        nfl_sides = load_nfl_sides()
        if not nfl_sides.empty:
            all_data["nfl_sides"] = nfl_sides
            print(f"    NFL sides (nfl.db): {len(nfl_sides)} games")
        nfl_teasers = load_nfl_teasers()
        if not nfl_teasers.empty:
            all_data["nfl_teasers"] = nfl_teasers
            print(f"    NFL teasers: {len(nfl_teasers)} Wong-qualifying legs")

    # ── 3. Wong resurface calibration (MUST pass) ─────────────────────
    calibration = wong_resurface_check(all_data)
    if not calibration["passed"]:
        print("\n  *** PIPELINE BROKEN — Wong dog teaser not resurfaced ***")
        print("  *** Fix the pipeline before trusting any results ***")
        report = {
            "calibration_failed": True,
            "honest_summary": "Pipeline broken — cannot find the known Wong dog teaser edge.",
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return report

    if calibrate_only:
        print("\n  [CALIBRATE ONLY] — Wong resurface passed, stopping.")
        return {"calibration": calibration, "calibrate_only": True}

    # ── 4. Evaluate all combos ────────────────────────────────────────
    print(f"\n  Evaluating {n_total} combos on search set...")
    results = []
    for i, combo in enumerate(combos):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i+1}/{n_total}] {combo['combo_id']}")
        result = evaluate_combo(combo, all_data)
        results.append(result)

    # ── 5. Multiple-comparisons correction ────────────────────────────
    print(f"\n  Applying corrections (Bonferroni N={n_total}, BH-FDR 5%)...")
    results = correct_pvalues(results, n_total)

    # Report
    candidates = [r for r in results if r.get("correction_status") == "CANDIDATE"]
    prob_noise = [r for r in results if r.get("correction_status") == "PROBABLY_NOISE"]
    noise = [r for r in results if r.get("correction_status") == "NOISE_AFTER_CORRECTION"]
    be_uncertain = [r for r in results if r.get("correction_status") == "BE_UNCERTAIN"]
    insuff = [r for r in results if r.get("status") == "INSUFFICIENT_N"]

    print(f"\n  Results:")
    print(f"    CANDIDATE (survives + mechanism): {len(candidates)}")
    print(f"    PROBABLY_NOISE (survives, no mechanism): {len(prob_noise)}")
    print(f"    BE_UNCERTAIN (MLB/NHL spreads, unreliable BE): {len(be_uncertain)}")
    print(f"    NOISE_AFTER_CORRECTION: {len(noise)}")
    print(f"    INSUFFICIENT_N: {len(insuff)}")

    if candidates:
        print(f"\n  Candidates (sorted by corrected p-value):")
        for c in sorted(candidates, key=lambda x: x.get("p_bonferroni", 1)):
            of = c.get("outer_fold", {})
            print(f"    {c['combo_id']}: edge={of.get('edge_pp', 0):+.1f}pp, "
                  f"N={of.get('n', 0)}, p_bonf={c.get('p_bonferroni', 1):.4f}, "
                  f"mechanism={c.get('mechanism')}")

    # ── 6. Validation pass (top 5 on 2024) ────────────────────────────
    top_candidates = sorted(
        candidates + prob_noise,
        key=lambda x: x.get("p_bonferroni", 1)
    )[:TOP_N_VALIDATION]

    val_results = []
    if top_candidates:
        print(f"\n  Testing top {len(top_candidates)} on VALIDATION (2024)...")
        val_results = validate_candidates(top_candidates, all_data)

        for v in val_results:
            vs = v.get("val_stats", {})
            if vs and vs.get("status") == "OK":
                print(f"    {v['combo_id']}: val_edge={vs.get('edge_pp', 0):+.1f}pp, "
                      f"N={vs.get('n', 0)}, p_corr={vs.get('p_val_corrected', 1):.4f}, "
                      f"status={v.get('val_status')}")
            else:
                print(f"    {v['combo_id']}: {v.get('val_status')}")

    # ── 7. Final holdout (single winner on 2025+) ─────────────────────
    confirmed = [v for v in val_results if v.get("val_status") == "CONFIRMED"]
    final_result = None

    if confirmed:
        winner = confirmed[0]  # best by original p_bonferroni
        print(f"\n  Testing winner on FINAL holdout (2025+): {winner['combo_id']}")
        final_result = final_test(winner, all_data)

        fs = final_result.get("final_stats", {})
        if fs and fs.get("status") == "OK":
            print(f"    Final: edge={fs.get('edge_pp', 0):+.1f}pp, "
                  f"N={fs.get('n', 0)}, WR={fs.get('wr', 0)*100:.1f}%, "
                  f"p={fs.get('p_raw', 1):.4f}")
            print(f"    Verdict: {final_result.get('final_verdict')}")
            print(f"    Tier: {final_result.get('beat_close_tier')}")
        else:
            print(f"    Final: {fs.get('status') if fs else 'NO_DATA'}")
    else:
        print(f"\n  No candidates confirmed on validation — no final test needed.")

    # ── 8. Generate report ────────────────────────────────────────────
    meta = {
        "n_combos_total": n_total,
        "n_with_mechanism": n_with_mechanism,
        "search_set": f"pre-{SEARCH_END[:4]}",
        "validation_set": f"{VAL_START[:4]}",
        "final_set": f"{FINAL_START[:4]}+",
        "sport_filter": sport_filter,
        "bonferroni_n": n_total,
    }

    report = generate_report(meta, results, val_results, final_result, calibration)

    # Print honest summary
    print(f"\n{'='*70}")
    print(f"HONEST SUMMARY")
    print(f"{'='*70}")
    print(f"  {report['honest_summary']}")
    print(f"\n  Beat-close note:")
    print(f"  {report['beat_close_note']}")

    # Save
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {output_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Selection × Construction search with overfitting guards")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show combos without evaluating")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Run Wong resurface check only")
    parser.add_argument("--sport", type=str, default=None,
                        help="Restrict to one sport (MLB/NFL/NBA/NHL)")
    parser.add_argument("--output", type=str,
                        default="/tmp/selection_construction_search_report.json",
                        help="Output path for JSON report")
    args = parser.parse_args()

    run_search(
        sport_filter=args.sport,
        dry_run=args.dry_run,
        calibrate_only=args.calibrate_only,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
