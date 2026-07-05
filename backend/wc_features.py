"""
wc_features.py — World Cup 2026 feature engineering module.

Computes match-level feature vectors for WC 2026 games, combining:
  - Pi ratings (Elo-style power ratings)
  - Head-to-head history
  - Venue/climate factors (altitude, temperature delta)
  - Goal scoring form
  - Tournament context (group stage, knockout)
  - Odds-implied probabilities

Usage:
    from wc_features import build_match_features
    features = build_match_features("Argentina", "Paraguay", venue="MetLife Stadium")
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Data paths ────────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_json(filename: str) -> dict:
    path = os.path.join(_DATA_DIR, filename)
    with open(path) as f:
        return json.load(f)


# ── Lazy-loaded data caches ───────────────────────────────────────────────────

_pi_ratings: Optional[dict] = None
_venues: Optional[dict] = None
_climate: Optional[dict] = None
_h2h: Optional[dict] = None
_results_df: Optional[pd.DataFrame] = None
_groups: Optional[dict] = None
_schedule: Optional[list] = None
_odds: Optional[list] = None
_scoring_dep: Optional[dict] = None


def _get_pi() -> dict:
    global _pi_ratings
    if _pi_ratings is None:
        _pi_ratings = _load_json("wc2026_pi_ratings_v2.json")
    return _pi_ratings


def _get_venues() -> dict:
    global _venues
    if _venues is None:
        _venues = _load_json("wc2026_venues.json")
    return _venues


def _get_climate() -> dict:
    global _climate
    if _climate is None:
        _climate = _load_json("wc2026_team_climate.json")
    return _climate


def _get_h2h() -> dict:
    global _h2h
    if _h2h is None:
        _h2h = _load_json("wc2026_h2h.json")
    return _h2h


def _get_results() -> pd.DataFrame:
    global _results_df
    if _results_df is None:
        path = os.path.join(_DATA_DIR, "international_results.csv")
        _results_df = pd.read_csv(path)
        _results_df["date"] = pd.to_datetime(_results_df["date"])
    return _results_df


def _get_scoring_dep() -> dict:
    global _scoring_dep
    if _scoring_dep is None:
        try:
            _scoring_dep = _load_json("wc2026_scoring_dependency.json")
        except FileNotFoundError:
            _scoring_dep = {}
    return _scoring_dep


def _get_groups() -> dict:
    global _groups
    if _groups is None:
        _groups = _load_json("wc2026_groups.json")
    return _groups


def _get_schedule() -> list:
    global _schedule
    if _schedule is None:
        _schedule = _load_json("wc2026_schedule.json")
    return _schedule


def _get_odds() -> list:
    global _odds
    if _odds is None:
        try:
            _odds = _load_json("wc2026_odds_snapshot.json")
        except FileNotFoundError:
            _odds = []
    return _odds


# ── Name normalization ────────────────────────────────────────────────────────

_NAME_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States",
    "Republic of Ireland": "Ireland",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "Chinese Taipei": "Taiwan",
}


def normalize_team(name: str) -> str:
    return _NAME_ALIASES.get(name, name)


# ── Pi rating features ───────────────────────────────────────────────────────

def get_pi_features(home: str, away: str) -> dict:
    pi = _get_pi()
    h = normalize_team(home)
    a = normalize_team(away)

    hp = pi.get(h, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})
    ap = pi.get(a, {"home_pi": 0.0, "away_pi": 0.0, "games": 0})

    home_combined = hp["home_pi"] + hp["away_pi"]
    away_combined = ap["home_pi"] + ap["away_pi"]

    return {
        "home_pi_combined": round(home_combined, 4),
        "away_pi_combined": round(away_combined, 4),
        "pi_delta": round(home_combined - away_combined, 4),
        "home_pi_home": round(hp["home_pi"], 4),
        "home_pi_away": round(hp["away_pi"], 4),
        "away_pi_home": round(ap["home_pi"], 4),
        "away_pi_away": round(ap["away_pi"], 4),
        "home_pi_games": hp["games"],
        "away_pi_games": ap["games"],
    }


# ── Head-to-head features ────────────────────────────────────────────────────

def get_h2h_features(home: str, away: str) -> dict:
    h2h = _get_h2h()
    h = normalize_team(home)
    a = normalize_team(away)

    key1 = f"{h} vs {a}"
    key2 = f"{a} vs {h}"
    data = h2h.get(key1) or h2h.get(key2)

    if not data:
        return {
            "h2h_matches": 0,
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "h2h_home_goals": 0,
            "h2h_away_goals": 0,
            "h2h_home_win_pct": 0.5,
            "h2h_goal_diff": 0.0,
        }

    total = data["total_matches"]
    t1, t2 = data["team1"], data["team2"]

    if h == t1:
        hw = data.get(f"{t1}_wins", 0)
        aw = data.get(f"{t2}_wins", 0)
        hg = data.get(f"{t1}_goals", 0)
        ag = data.get(f"{t2}_goals", 0)
    else:
        hw = data.get(f"{t2}_wins", 0)
        aw = data.get(f"{t1}_wins", 0)
        hg = data.get(f"{t2}_goals", 0)
        ag = data.get(f"{t1}_goals", 0)

    draws = data.get("draws", 0)

    return {
        "h2h_matches": total,
        "h2h_home_wins": hw,
        "h2h_away_wins": aw,
        "h2h_draws": draws,
        "h2h_home_goals": hg,
        "h2h_away_goals": ag,
        "h2h_home_win_pct": round((hw + 0.5 * draws) / max(total, 1), 4),
        "h2h_goal_diff": round((hg - ag) / max(total, 1), 4),
    }


# ── Form features (recent results) ───────────────────────────────────────────

def get_form_features(team: str, n_matches: int = 10,
                      before_date: str = "2026-06-11") -> dict:
    df = _get_results()
    t = normalize_team(team)
    cutoff = pd.Timestamp(before_date)

    team_matches = df[
        ((df["home_team"] == t) | (df["away_team"] == t))
        & (df["date"] < cutoff)
        & (df["date"] >= cutoff - pd.Timedelta(days=730))
    ].sort_values("date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "form_matches": 0,
            "form_wins": 0,
            "form_draws": 0,
            "form_losses": 0,
            "form_goals_scored": 0,
            "form_goals_conceded": 0,
            "form_win_pct": 0.5,
            "form_avg_goals_scored": 0.0,
            "form_avg_goals_conceded": 0.0,
            "form_clean_sheets": 0,
            "form_failed_to_score": 0,
        }

    wins = draws = losses = gf = ga = cs = fts = 0
    for _, row in team_matches.iterrows():
        hs, as_ = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        is_home = row["home_team"] == t
        scored = int(hs) if is_home else int(as_)
        conceded = int(as_) if is_home else int(hs)
        gf += scored
        ga += conceded
        if scored > conceded:
            wins += 1
        elif scored == conceded:
            draws += 1
        else:
            losses += 1
        if conceded == 0:
            cs += 1
        if scored == 0:
            fts += 1

    total = wins + draws + losses

    return {
        "form_matches": total,
        "form_wins": wins,
        "form_draws": draws,
        "form_losses": losses,
        "form_goals_scored": gf,
        "form_goals_conceded": ga,
        "form_win_pct": round((wins + 0.5 * draws) / max(total, 1), 4),
        "form_avg_goals_scored": round(gf / max(total, 1), 4),
        "form_avg_goals_conceded": round(ga / max(total, 1), 4),
        "form_clean_sheets": cs,
        "form_failed_to_score": fts,
    }


# ── Venue / climate / altitude features ──────────────────────────────────────

def get_venue_features(home: str, away: str,
                       venue: Optional[str] = None) -> dict:
    venues = _get_venues()
    climate = _get_climate()
    h = normalize_team(home)
    a = normalize_team(away)

    # Default venue if not specified
    v = venues.get(venue, {}) if venue else {}
    venue_alt = v.get("alt_ft", 200)
    venue_temp = v.get("avg_june_temp_f", 80)

    hc = climate.get(h, {"temp": 72, "alt": 500})
    ac = climate.get(a, {"temp": 72, "alt": 500})

    home_temp_delta = venue_temp - hc["temp"]
    away_temp_delta = venue_temp - ac["temp"]
    home_alt_delta = venue_alt - hc["alt"]
    away_alt_delta = venue_alt - ac["alt"]

    # Altitude stress: research shows >4000 ft impacts performance significantly
    home_alt_stress = max(0, (venue_alt - hc["alt"]) / 1000) if venue_alt > 3000 else 0
    away_alt_stress = max(0, (venue_alt - ac["alt"]) / 1000) if venue_alt > 3000 else 0

    # Heat stress: unfamiliar heat (>10°F above home temp) impacts performance
    home_heat_stress = max(0, home_temp_delta - 10) / 10
    away_heat_stress = max(0, away_temp_delta - 10) / 10

    return {
        "venue_alt_ft": venue_alt,
        "venue_temp_f": venue_temp,
        "home_temp_delta": round(home_temp_delta, 1),
        "away_temp_delta": round(away_temp_delta, 1),
        "home_alt_delta": round(home_alt_delta, 0),
        "away_alt_delta": round(away_alt_delta, 0),
        "home_alt_stress": round(home_alt_stress, 3),
        "away_alt_stress": round(away_alt_stress, 3),
        "home_heat_stress": round(home_heat_stress, 3),
        "away_heat_stress": round(away_heat_stress, 3),
        "venue_neutral": 1,  # All WC games are neutral (except host nation)
        "home_is_host": 1 if h in ("United States", "Mexico", "Canada") else 0,
        "away_is_host": 1 if a in ("United States", "Mexico", "Canada") else 0,
    }


# ── Odds-implied features ────────────────────────────────────────────────────

def get_odds_features(home: str, away: str) -> dict:
    odds_data = _get_odds()
    h = normalize_team(home)
    a = normalize_team(away)

    # Find matching game in odds data
    match = None
    for g in odds_data:
        gh = normalize_team(g.get("home_team", ""))
        ga = normalize_team(g.get("away_team", ""))
        if (gh == h and ga == a) or (gh == a and ga == h):
            match = g
            break

    if not match:
        return {
            "odds_home_implied": 0.33,
            "odds_away_implied": 0.33,
            "odds_draw_implied": 0.33,
            "odds_over25_implied": 0.5,
            "odds_under25_implied": 0.5,
            "odds_home_spread": 0.0,
            "odds_bookmaker_count": 0,
        }

    # Aggregate across bookmakers for consensus
    # Collect devigged triples per bookmaker so draw isn't inflated
    home_probs = []
    away_probs = []
    draw_probs = []
    over_probs = []
    spread_vals = []

    def _price_to_implied(price):
        """Convert decimal or American odds to raw implied probability."""
        if price <= 0:
            return None
        if price > 50 or price < -50:
            if price > 0:
                return 100 / (price + 100)
            else:
                return abs(price) / (abs(price) + 100)
        return 1 / price

    for bm in match.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            if mkt["key"] == "h2h":
                # Collect this bookmaker's h2h triple
                bm_home = bm_away = bm_draw = None
                for o in mkt["outcomes"]:
                    imp = _price_to_implied(o.get("price", 0))
                    if imp is None:
                        continue
                    nm = normalize_team(o["name"])
                    if nm == h:
                        bm_home = imp
                    elif nm == a:
                        bm_away = imp
                    elif o["name"] == "Draw":
                        bm_draw = imp

                # Devig: normalize this bookmaker's triple to sum to 1.0
                if bm_home and bm_away and bm_draw:
                    bm_total = bm_home + bm_away + bm_draw
                    home_probs.append(bm_home / bm_total)
                    away_probs.append(bm_away / bm_total)
                    draw_probs.append(bm_draw / bm_total)

            elif mkt["key"] == "totals":
                for o in mkt["outcomes"]:
                    imp = _price_to_implied(o.get("price", 0))
                    if imp is None:
                        continue
                    if o["name"] == "Over":
                        over_probs.append(imp)

            elif mkt["key"] == "spreads":
                for o in mkt["outcomes"]:
                    nm = normalize_team(o["name"])
                    if nm == h:
                        spread_vals.append(o.get("point", 0))

    def _avg(lst, default=0.33):
        return round(sum(lst) / len(lst), 4) if lst else default

    return {
        "odds_home_implied": _avg(home_probs, 0.33),
        "odds_away_implied": _avg(away_probs, 0.33),
        "odds_draw_implied": _avg(draw_probs, 0.33),
        "odds_over25_implied": _avg(over_probs, 0.5),
        "odds_under25_implied": round(1 - _avg(over_probs, 0.5), 4),
        "odds_home_spread": _avg(spread_vals, 0.0),
        "odds_bookmaker_count": len(match.get("bookmakers", [])),
    }


# ── Tournament context features ──────────────────────────────────────────────

def get_tournament_features(home: str, away: str) -> dict:
    groups = _get_groups()
    h = normalize_team(home)
    a = normalize_team(away)

    # Find groups for each team
    home_group = None
    away_group = None
    for grp, teams in groups.items():
        for t in teams:
            if normalize_team(t) == h:
                home_group = grp
            if normalize_team(t) == a:
                away_group = grp

    same_group = 1 if home_group and home_group == away_group else 0

    return {
        "home_group": home_group or "?",
        "away_group": away_group or "?",
        "same_group": same_group,
        "is_group_stage": 1,  # Will update for knockout rounds
    }


# ── Goal scoring patterns ────────────────────────────────────────────────────

def get_scoring_pattern_features(team: str, n_matches: int = 15,
                                  before_date: str = "2026-06-11") -> dict:
    """Analyze scoring patterns: BTTS, over/under, first-half goals, etc."""
    df = _get_results()
    t = normalize_team(team)
    cutoff = pd.Timestamp(before_date)

    matches = df[
        ((df["home_team"] == t) | (df["away_team"] == t))
        & (df["date"] < cutoff)
        & (df["date"] >= cutoff - pd.Timedelta(days=730))
    ].sort_values("date", ascending=False).head(n_matches)

    if matches.empty:
        return {
            "btts_pct": 0.5,
            "over25_pct": 0.5,
            "over15_pct": 0.7,
            "avg_total_goals": 2.5,
            "avg_margin": 0.0,
        }

    btts = o25 = o15 = total_goals = total_margin = count = 0

    for _, row in matches.iterrows():
        hs, as_ = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        hs, as_ = int(hs), int(as_)
        is_home = row["home_team"] == t
        scored = hs if is_home else as_
        conceded = as_ if is_home else hs

        total = hs + as_
        total_goals += total
        total_margin += scored - conceded
        count += 1

        if hs > 0 and as_ > 0:
            btts += 1
        if total > 2.5:
            o25 += 1
        if total > 1.5:
            o15 += 1

    n = max(count, 1)
    return {
        "btts_pct": round(btts / n, 4),
        "over25_pct": round(o25 / n, 4),
        "over15_pct": round(o15 / n, 4),
        "avg_total_goals": round(total_goals / n, 4),
        "avg_margin": round(total_margin / n, 4),
    }


# ── Competitive strength features ────────────────────────────────────────────

def get_strength_features(team: str, before_date: str = "2026-06-11") -> dict:
    """Strength of schedule + performance in competitive (non-friendly) matches."""
    df = _get_results()
    t = normalize_team(team)
    cutoff = pd.Timestamp(before_date)

    recent = df[
        ((df["home_team"] == t) | (df["away_team"] == t))
        & (df["date"] < cutoff)
        & (df["date"] >= cutoff - pd.Timedelta(days=730))
    ].copy()

    comp = recent[recent["tournament"] != "Friendly"]
    friendly = recent[recent["tournament"] == "Friendly"]

    def _win_pct(subset):
        w = d = l = 0
        for _, row in subset.iterrows():
            hs, as_ = row["home_score"], row["away_score"]
            if pd.isna(hs) or pd.isna(as_):
                continue
            is_home = row["home_team"] == t
            scored = int(hs) if is_home else int(as_)
            conceded = int(as_) if is_home else int(hs)
            if scored > conceded:
                w += 1
            elif scored == conceded:
                d += 1
            else:
                l += 1
        total = w + d + l
        return round((w + 0.5 * d) / max(total, 1), 4), total

    comp_wp, comp_n = _win_pct(comp)
    friendly_wp, friendly_n = _win_pct(friendly)

    return {
        "competitive_win_pct": comp_wp,
        "competitive_matches": comp_n,
        "friendly_win_pct": friendly_wp,
        "friendly_matches": friendly_n,
    }


# ── Scoring dependency features ───────────────────────────────────────────────

def get_scoring_dependency_features(team: str) -> dict:
    """Star dependency ratio and scoring depth from goalscorer data."""
    dep = _get_scoring_dep()
    t = normalize_team(team)
    data = dep.get(t, {})

    return {
        "dependency_ratio": data.get("dependency_ratio", 0.2),
        "scoring_depth": data.get("scoring_depth", 0.4),
        "unique_scorers": data.get("unique_scorers", 10),
        "total_goals": data.get("total_goals", 20),
    }


# ── Club chemistry and league quality features ───────────────────────────────

_chemistry: Optional[dict] = None
_league_quality: Optional[dict] = None


def _get_chemistry() -> dict:
    global _chemistry
    if _chemistry is None:
        try:
            _chemistry = _load_json("wc2026_club_chemistry.json")
        except FileNotFoundError:
            _chemistry = {}
    return _chemistry


def _get_league_quality() -> dict:
    global _league_quality
    if _league_quality is None:
        try:
            _league_quality = _load_json("wc2026_league_quality.json")
        except FileNotFoundError:
            _league_quality = {}
    return _league_quality


def get_squad_features(team: str) -> dict:
    """Club chemistry score and league quality for a team's squad."""
    t = normalize_team(team)
    # Try both raw and normalized name against JSON keys
    chem = _get_chemistry()
    lq = _get_league_quality()

    cdata = chem.get(t) or chem.get(team, {})
    ldata = lq.get(t) or lq.get(team, {})

    return {
        "chemistry_score": cdata.get("chemistry_score", 0),
        "largest_club_count": cdata.get("largest_club_count", 0),
        "club_concentration": cdata.get("club_concentration", 0.0),
        "unique_clubs": cdata.get("unique_clubs", 20),
        "league_quality": ldata.get("quality_score", 1.5),
        "tier1_players": ldata.get("tier1_players", 0),
        "quality_weighted_chemistry": cdata.get("quality_weighted_chemistry", 0.0),
    }


# ── Master feature builder ───────────────────────────────────────────────────

def build_match_features(home: str, away: str,
                         venue: Optional[str] = None,
                         before_date: str = "2026-06-11") -> dict:
    """
    Build a complete feature vector for a WC 2026 match.

    Returns a flat dict of ~50 features suitable for model input.
    """
    features = {}

    # Pi ratings
    features.update(get_pi_features(home, away))

    # Head-to-head
    features.update(get_h2h_features(home, away))

    # Form (last 10 matches each)
    home_form = get_form_features(home, n_matches=10, before_date=before_date)
    away_form = get_form_features(away, n_matches=10, before_date=before_date)
    features.update({f"home_{k}": v for k, v in home_form.items()})
    features.update({f"away_{k}": v for k, v in away_form.items()})

    # Venue / climate / altitude
    features.update(get_venue_features(home, away, venue))

    # Odds-implied
    features.update(get_odds_features(home, away))

    # Tournament context
    features.update(get_tournament_features(home, away))

    # Scoring patterns
    home_scoring = get_scoring_pattern_features(home, before_date=before_date)
    away_scoring = get_scoring_pattern_features(away, before_date=before_date)
    features.update({f"home_{k}": v for k, v in home_scoring.items()})
    features.update({f"away_{k}": v for k, v in away_scoring.items()})

    # Competitive strength
    home_strength = get_strength_features(home, before_date=before_date)
    away_strength = get_strength_features(away, before_date=before_date)
    features.update({f"home_{k}": v for k, v in home_strength.items()})
    features.update({f"away_{k}": v for k, v in away_strength.items()})

    # Scoring dependency
    home_dep = get_scoring_dependency_features(home)
    away_dep = get_scoring_dependency_features(away)
    features.update({f"home_{k}": v for k, v in home_dep.items()})
    features.update({f"away_{k}": v for k, v in away_dep.items()})

    # Squad features (chemistry + league quality)
    home_squad = get_squad_features(home)
    away_squad = get_squad_features(away)
    features.update({f"home_{k}": v for k, v in home_squad.items()})
    features.update({f"away_{k}": v for k, v in away_squad.items()})

    # Derived features
    features["form_win_pct_delta"] = round(
        features.get("home_form_win_pct", 0.5)
        - features.get("away_form_win_pct", 0.5), 4
    )
    features["competitive_wp_delta"] = round(
        features.get("home_competitive_win_pct", 0.5)
        - features.get("away_competitive_win_pct", 0.5), 4
    )
    features["combined_avg_goals"] = round(
        (features.get("home_avg_total_goals", 2.5)
         + features.get("away_avg_total_goals", 2.5)) / 2, 4
    )
    features["alt_stress_delta"] = round(
        features.get("away_alt_stress", 0) - features.get("home_alt_stress", 0), 4
    )
    features["heat_stress_delta"] = round(
        features.get("away_heat_stress", 0) - features.get("home_heat_stress", 0), 4
    )
    features["dependency_ratio_delta"] = round(
        features.get("home_dependency_ratio", 0.2)
        - features.get("away_dependency_ratio", 0.2), 4
    )
    features["scoring_depth_delta"] = round(
        features.get("home_scoring_depth", 0.4)
        - features.get("away_scoring_depth", 0.4), 4
    )
    features["chemistry_delta"] = (
        features.get("home_chemistry_score", 0)
        - features.get("away_chemistry_score", 0)
    )
    features["league_quality_delta"] = round(
        features.get("home_league_quality", 1.5)
        - features.get("away_league_quality", 1.5), 4
    )
    features["qw_chemistry_delta"] = round(
        features.get("home_quality_weighted_chemistry", 0.0)
        - features.get("away_quality_weighted_chemistry", 0.0), 4
    )

    return features


# ── Batch: build features for all scheduled WC matches ────────────────────────

def build_all_match_features() -> pd.DataFrame:
    """Build feature vectors for all 72 scheduled WC 2026 group stage matches."""
    schedule = _get_schedule()
    rows = []
    for match in schedule:
        home = match["home_team"]
        away = match["away_team"]
        before = match["commence_time"][:10]
        try:
            feat = build_match_features(home, away, before_date=before)
            feat["match_id"] = match.get("id", "")
            feat["home_team"] = home
            feat["away_team"] = away
            feat["commence_time"] = match["commence_time"]
            feat["group"] = match.get("group", "?")
            rows.append(feat)
        except Exception as e:
            print(f"[wc_features] Error for {home} vs {away}: {e}")

    df = pd.DataFrame(rows)
    return df


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== WC 2026 Feature Engineering Test ===\n")

    # Test: opening match Mexico vs South Africa
    feat = build_match_features("Mexico", "South Africa", venue="Estadio Azteca")
    print("Mexico vs South Africa @ Estadio Azteca:")
    for k, v in sorted(feat.items()):
        print(f"  {k:35s} = {v}")

    print(f"\nTotal features: {len(feat)}")

    # Build all match features
    print("\n--- Building features for all 72 matches ---")
    df = build_all_match_features()
    print(f"Feature matrix: {df.shape[0]} matches x {df.shape[1]} features")
    print(f"\nColumns: {sorted(df.columns.tolist())}")

    # Save feature matrix
    out_path = os.path.join(_DATA_DIR, "wc2026_feature_matrix.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
