"""
soccer_features.py — Phase 2: Soccer bet feature engineering.

Standalone script (not imported by production code).
Reads from bets.db and produces labeled feature DataFrames for model training.

Usage:
    cd /Users/joseportillo/Downloads/BetIQ
    python backend/analysis/soccer_features.py              # print sample
    python backend/analysis/soccer_features.py --export     # write CSV files
    python backend/analysis/soccer_features.py --coverage   # show coverage stats

Feature sets:
  total_goals:  form-based Over/Under features
  moneyline:    form-based ML + Double Chance features

Output CSVs (--export):
  data/soccer_total_goals_features.csv
  data/soccer_moneyline_features.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bets.db")

# ── Historical home win rates by league (source: Wikipedia 5-year averages) ──
# Used as league-level prior for moneyline features.
_LEAGUE_HOME_WIN_RATE = {
    "epl":           0.44,
    "english premier league": 0.44,
    "la liga":       0.46,
    "bundesliga":    0.46,
    "ligue1":        0.45,
    "ligue 1":       0.45,
    "serie a":       0.46,
    "ucl":           0.48,
    "uefa champions league": 0.48,
    "europa league": 0.46,
    "default":       0.45,
}


def _parse_line_from_description(desc: str, subtype: str) -> float | None:
    """Extract the numeric line from a leg description."""
    if subtype == "total_goals":
        m = re.search(r"(Over|Under)\s+([\d.]+)\s+Goals", desc, re.IGNORECASE)
        return float(m.group(2)) if m else None
    if subtype == "corners":
        m = re.search(r"(Over|Under)\s+([\d.]+)\s+Corners", desc, re.IGNORECASE)
        return float(m.group(2)) if m else None
    if subtype in ("shots_on_target", "shots_total"):
        m = re.search(r"(\d+)\s+Or More Shots", desc, re.IGNORECASE)
        return float(m.group(1)) if m else None
    return None


def _parse_direction(desc: str) -> int | None:
    """1 for Over/Yes, 0 for Under/No."""
    d = desc.lower()
    if d.startswith("over ") or " over " in d[:20]:
        return 1
    if d.startswith("under ") or " under " in d[:20]:
        return 0
    if d.startswith("yes "):
        return 1
    if d.startswith("no "):
        return 0
    return None


def _parse_odds(odds_str: str | None) -> float | None:
    """Parse American odds string like '-110' or '+150' to implied probability."""
    if not odds_str:
        return None
    try:
        odds = float(str(odds_str).strip().replace("+", ""))
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except (ValueError, TypeError):
        return None


def _label(leg_result: str) -> int | None:
    if leg_result == "WIN":
        return 1
    if leg_result == "LOSS":
        return 0
    return None


def _form_to_win_rate(form: str | None, n: int = 5) -> float | None:
    """Convert form string 'WWDLW' to win rate over last n games."""
    if not form:
        return None
    subset = form[:n]
    if not subset:
        return None
    return subset.count("W") / len(subset)


def _form_to_unbeaten_rate(form: str | None, n: int = 5) -> float | None:
    """W+D rate over last n games."""
    if not form:
        return None
    subset = form[:n]
    if not subset:
        return None
    return (subset.count("W") + subset.count("D")) / len(subset)


def _league_home_win_rate(league: str | None) -> float:
    if not league:
        return _LEAGUE_HOME_WIN_RATE["default"]
    return _LEAGUE_HOME_WIN_RATE.get(league.lower(), _LEAGUE_HOME_WIN_RATE["default"])


# ─────────────────────────────────────────────────────────────────────────────
# Total Goals features
# ─────────────────────────────────────────────────────────────────────────────

def build_total_goals_features() -> pd.DataFrame:
    """
    Build feature matrix for Total Goals (Over/Under) legs.

    Returns DataFrame with features + target column 'label'.
    Includes only resolved legs (WIN/LOSS) with form data available.
    """
    con = sqlite3.connect(DB_PATH)

    # Load resolved total goals legs with bet date
    legs_df = pd.read_sql_query("""
        SELECT bl.id, bl.description, bl.subtype, bl.team as home_team,
               bl.opponent as away_team, bl.leg_result, bl.league, bl.odds_str,
               DATE(b.time_placed) as bet_date
        FROM bet_legs bl
        JOIN bets b ON bl.bet_id = b.id
        WHERE bl.sport='Soccer'
          AND bl.subtype = 'total_goals'
          AND bl.leg_result IN ('WIN', 'LOSS')
        ORDER BY bet_date
    """, con)

    # Load team form
    form_df = pd.read_sql_query(
        "SELECT * FROM team_soccer_form", con
    )
    con.close()

    if legs_df.empty or form_df.empty:
        return pd.DataFrame()

    # Build form lookup: (team_name, as_of_date) → row
    form_idx = form_df.set_index(["team_name", "as_of_date"])

    rows = []
    for _, leg in legs_df.iterrows():
        home = leg["home_team"]
        away = leg["away_team"]
        bdate = leg["bet_date"]

        if not home or not bdate:
            continue

        # Look up form for home and away teams
        def _lookup(team, date):
            if not team or not date or (team, date) not in form_idx.index:
                return None
            row = form_idx.loc[(team, date)]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row

        hf = _lookup(home, bdate)
        af = _lookup(away, bdate) if away else None

        # Require at least home team form
        if hf is None:
            continue

        line = _parse_line_from_description(leg["description"], leg["subtype"])
        direction = _parse_direction(leg["description"])
        implied_prob = _parse_odds(leg["odds_str"])
        label = _label(leg["leg_result"])

        if label is None:
            continue

        row = {
            # Identifiers
            "leg_id":          leg["id"],
            "bet_date":        bdate,
            "league":          leg["league"],
            "description":     leg["description"],
            # Target
            "label":           label,
            # Bet context
            "line":            line,
            "direction":       direction,
            "implied_prob":    implied_prob,
            # Home team form
            "home_gf_avg_5":   hf.get("goals_scored_5"),
            "home_ga_avg_5":   hf.get("goals_conceded_5"),
            "home_gf_avg_10":  hf.get("goals_scored_10"),
            "home_ga_avg_10":  hf.get("goals_conceded_10"),
            "home_over25_r10": hf.get("over25_rate_10"),
            "home_form_wr_5":  _form_to_win_rate(hf.get("form_5"), 5),
            "home_games_found": hf.get("games_found"),
        }

        # Away team form (optional — enrich if available)
        if af is not None:
            af_row = af
            row.update({
                "away_gf_avg_5":   af_row.get("goals_scored_5"),
                "away_ga_avg_5":   af_row.get("goals_conceded_5"),
                "away_gf_avg_10":  af_row.get("goals_scored_10"),
                "away_ga_avg_10":  af_row.get("goals_conceded_10"),
                "away_over25_r10": af_row.get("over25_rate_10"),
                "away_form_wr_5":  _form_to_win_rate(af_row.get("form_5"), 5),
                "away_games_found": af_row.get("games_found"),
            })
        else:
            row.update({
                "away_gf_avg_5": None, "away_ga_avg_5": None,
                "away_gf_avg_10": None, "away_ga_avg_10": None,
                "away_over25_r10": None, "away_form_wr_5": None,
                "away_games_found": 0,
            })

        # Derived: combined goal expectation
        if row["home_gf_avg_5"] is not None and row["away_gf_avg_5"] is not None:
            row["combined_goals_exp_5"] = round(row["home_gf_avg_5"] + row["away_gf_avg_5"], 2)
        elif row["home_gf_avg_5"] is not None:
            row["combined_goals_exp_5"] = row["home_gf_avg_5"] * 2  # rough estimate
        else:
            row["combined_goals_exp_5"] = None

        if row["home_over25_r10"] is not None and row["away_over25_r10"] is not None:
            row["avg_over25_rate"] = round(
                (row["home_over25_r10"] + row["away_over25_r10"]) / 2, 3
            )
        elif row["home_over25_r10"] is not None:
            row["avg_over25_rate"] = row["home_over25_r10"]
        else:
            row["avg_over25_rate"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("bet_date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Moneyline + Double Chance features
# ─────────────────────────────────────────────────────────────────────────────

def build_moneyline_features() -> pd.DataFrame:
    """
    Build feature matrix for Moneyline + Double Chance legs.

    team = pick_team (who was bet on)
    home_team derived from description parsing (team = home if in left position)
    is_home: 1 if pick_team == home_team, 0 if away, 0.5 if Double Chance
    """
    con = sqlite3.connect(DB_PATH)

    legs_df = pd.read_sql_query("""
        SELECT bl.id, bl.description, bl.subtype, bl.team as pick_team,
               bl.opponent as away_team, bl.leg_result, bl.league, bl.odds_str,
               DATE(b.time_placed) as bet_date
        FROM bet_legs bl
        JOIN bets b ON bl.bet_id = b.id
        WHERE bl.sport='Soccer'
          AND bl.market_type = 'Moneyline'
          AND bl.leg_result IN ('WIN', 'LOSS')
        ORDER BY bet_date
    """, con)

    form_df = pd.read_sql_query("SELECT * FROM team_soccer_form", con)
    con.close()

    if legs_df.empty or form_df.empty:
        return pd.DataFrame()

    form_idx = form_df.set_index(["team_name", "as_of_date"])

    def _get_form(team, date):
        if not team or not date:
            return None
        key = (team, date)
        if key not in form_idx.index:
            return None
        row = form_idx.loc[key]
        # If multiple matching rows → DataFrame; take first row as Series
        # If single matching row → Series already
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row  # already a Series

    rows = []
    for _, leg in legs_df.iterrows():
        pick_team = leg["pick_team"]
        away_team = leg["away_team"]  # from opponent field = away team in game
        bdate     = leg["bet_date"]
        subtype   = leg["subtype"]

        if not pick_team or not bdate:
            continue

        # Determine home team from description
        # Description: "{pick_team} Moneyline (3-way) {home_team} v {away_team}"
        # team field = pick_team, opponent field = away_team of the fixture
        # We need to figure out who is home
        # Re-parse from description
        desc = leg["description"]
        v_idx = desc.rfind(" v ")
        if v_idx != -1:
            # away of fixture = everything after last " v "
            fixture_away = desc[v_idx + 3:].strip()
            # home of fixture = last word(s) before " v " after the market label
            # We know away_team field was set to fixture_away
            # For home team, look at what's stored in the description
            # The parse already extracted: team=pick_team, opponent=away_team(=fixture_away)
            # But we need home_team = the first team in "X v Y" part
            # We can extract it from description using the market label end
            m = re.search(r'(?:Moneyline(?:\s*\(3-way\))?|Double Chance)\s+(.+)$',
                           desc[:v_idx], re.IGNORECASE)
            fixture_home = m.group(1).strip() if m else None
        else:
            fixture_home = None
            fixture_away = None

        # is_home: 1=home, 0=away, 0.5=double chance pick
        if subtype == "double_chance":
            is_home = 0.5  # DC doesn't distinguish home/away cleanly
        elif fixture_home and _normalize_simple(pick_team) == _normalize_simple(fixture_home):
            is_home = 1
        elif fixture_away and _normalize_simple(pick_team) == _normalize_simple(fixture_away):
            is_home = 0
        else:
            is_home = 0.5  # ambiguous

        # Get form for pick team (the team bet on)
        pf = _get_form(pick_team, bdate)
        # Get form for opponent
        opp_team = fixture_home if is_home == 0 else fixture_away
        of = _get_form(opp_team, bdate) if opp_team else None
        # Also try away_team field
        if of is None and away_team:
            of = _get_form(away_team, bdate)

        if pf is None:
            continue  # No form for pick team — skip

        implied_prob = _parse_odds(leg["odds_str"])
        label = _label(leg["leg_result"])
        if label is None:
            continue

        row = {
            "leg_id":          leg["id"],
            "bet_date":        bdate,
            "league":          leg["league"],
            "description":     leg["description"],
            "label":           label,
            "subtype":         subtype,
            # Bet context
            "is_home":         is_home,
            "is_double_chance": 1 if subtype == "double_chance" else 0,
            "implied_prob":    implied_prob,
            "league_home_wr":  _league_home_win_rate(leg["league"]),
            # Pick team form
            "pick_wr_5":       _form_to_win_rate(pf.get("form_5"), 5),
            "pick_wr_10":      _form_to_win_rate(pf.get("form_10"), 10),
            "pick_unbeaten_5": _form_to_unbeaten_rate(pf.get("form_5"), 5),
            "pick_gf_avg_5":   pf.get("goals_scored_5"),
            "pick_ga_avg_5":   pf.get("goals_conceded_5"),
            "pick_gf_avg_10":  pf.get("goals_scored_10"),
            "pick_ga_avg_10":  pf.get("goals_conceded_10"),
            "pick_home_wr_5":  _form_to_win_rate(pf.get("home_form_5"), 5),
            "pick_away_wr_5":  _form_to_win_rate(pf.get("away_form_5"), 5),
            "pick_games_found": pf.get("games_found"),
        }

        # Opponent form
        if of is not None:
            row.update({
                "opp_wr_5":       _form_to_win_rate(of.get("form_5"), 5),
                "opp_gf_avg_5":   of.get("goals_scored_5"),
                "opp_ga_avg_5":   of.get("goals_conceded_5"),
                "opp_unbeaten_5": _form_to_unbeaten_rate(of.get("form_5"), 5),
                "opp_games_found": of.get("games_found"),
            })
        else:
            row.update({
                "opp_wr_5": None, "opp_gf_avg_5": None,
                "opp_ga_avg_5": None, "opp_unbeaten_5": None,
                "opp_games_found": 0,
            })

        # Derived: form differential
        if row["pick_wr_5"] is not None and row["opp_wr_5"] is not None:
            row["wr_diff_5"] = round(row["pick_wr_5"] - row["opp_wr_5"], 3)
        else:
            row["wr_diff_5"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("bet_date").reset_index(drop=True)
    return df


def _normalize_simple(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", " ", name.lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
# Coverage report
# ─────────────────────────────────────────────────────────────────────────────

def print_coverage() -> None:
    tg = build_total_goals_features()
    ml = build_moneyline_features()

    print("\n━━━ Phase 2: Feature Coverage ━━━")

    print(f"\nTotal Goals feature matrix:")
    print(f"  Rows: {len(tg)}")
    if not tg.empty:
        print(f"  Date range: {tg['bet_date'].min()} → {tg['bet_date'].max()}")
        print(f"  Label dist: {tg['label'].value_counts().to_dict()} "
              f"({tg['label'].mean():.1%} WIN rate)")
        print(f"  Feature coverage (% non-null):")
        key_cols = ["line", "direction", "implied_prob",
                    "home_gf_avg_5", "away_gf_avg_5",
                    "home_over25_r10", "away_over25_r10",
                    "combined_goals_exp_5", "avg_over25_rate"]
        for col in key_cols:
            if col in tg.columns:
                pct = tg[col].notna().mean() * 100
                print(f"    {col:30s} {pct:.0f}%")

    print(f"\nMoneyline feature matrix:")
    print(f"  Rows: {len(ml)}")
    if not ml.empty:
        print(f"  Date range: {ml['bet_date'].min()} → {ml['bet_date'].max()}")
        print(f"  Label dist: {ml['label'].value_counts().to_dict()} "
              f"({ml['label'].mean():.1%} WIN rate)")
        print(f"  Subtype: {ml['subtype'].value_counts().to_dict()}")
        print(f"  Feature coverage (% non-null):")
        key_cols = ["is_home", "implied_prob", "pick_wr_5", "pick_unbeaten_5",
                    "pick_gf_avg_5", "opp_wr_5", "wr_diff_5"]
        for col in key_cols:
            if col in ml.columns:
                pct = ml[col].notna().mean() * 100
                print(f"    {col:30s} {pct:.0f}%")

    print(f"\nTotal model-ready rows: {len(tg) + len(ml)}")
    print("  Min required: 50 per model. "
          f"Total goals: {'✅' if len(tg) >= 50 else '❌'} ({len(tg)}). "
          f"Moneyline: {'✅' if len(ml) >= 50 else '❌'} ({len(ml)}).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", action="store_true", help="Print feature coverage report")
    parser.add_argument("--export",   action="store_true", help="Write feature CSVs to data/")
    parser.add_argument("--sample",   type=int, default=5, help="Print N sample rows")
    args = parser.parse_args()

    if args.coverage or not any([args.export]):
        print_coverage()

    tg = build_total_goals_features()
    ml = build_moneyline_features()

    if args.sample and not tg.empty:
        print("\n--- Total Goals sample ---")
        display_cols = ["bet_date", "description", "line", "direction",
                        "home_gf_avg_5", "away_gf_avg_5", "avg_over25_rate",
                        "implied_prob", "label"]
        avail = [c for c in display_cols if c in tg.columns]
        print(tg[avail].tail(args.sample).to_string(index=False))

    if args.sample and not ml.empty:
        print("\n--- Moneyline sample ---")
        display_cols = ["bet_date", "pick_wr_5", "opp_wr_5", "wr_diff_5",
                        "is_home", "is_double_chance", "implied_prob", "label"]
        avail = [c for c in display_cols if c in ml.columns]
        print(ml[avail].tail(args.sample).to_string(index=False))

    if args.export:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        if not tg.empty:
            p = os.path.join(out_dir, "soccer_total_goals_features.csv")
            tg.to_csv(p, index=False)
            print(f"\nExported {len(tg)} rows → {p}")
        if not ml.empty:
            p = os.path.join(out_dir, "soccer_moneyline_features.csv")
            ml.to_csv(p, index=False)
            print(f"Exported {len(ml)} rows → {p}")
