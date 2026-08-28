#!/usr/bin/env python3
"""
pit_reconstruction.py — Point-in-time feature reconstruction for MLB games Jun 1 – Jul 12, 2026.

Steps:
  1. Load 562 out-of-sample MLB games (Jun 1–Jul 12 2026, settled, OOS)
  2. Fetch missing pitcher game logs (after Jun 14) from MLB Stats API
  3. Fetch missing team batting game logs (after Jun 17) from MLB Stats API
  4. Reconstruct all 21 model features with strict game_date < D cutoff
  5. Score with mlb_ats_distributional.pkl to get dist_wp per rung
  6. Spot-check one Jun-15 game for feature integrity
  7. Validate against live shadow_negbin dispatch (Jun 28–Jul 12)
  8. Run E-profit allocation sim: flat vs EP-filter vs EP-weighted
"""

import os
import sys
import sqlite3
import pickle
import time
import warnings
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy import stats as scipy_stats
from scipy.stats import norm, ttest_rel, ttest_1samp

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = "/Users/joseportillo/Downloads/BetIQ"
HIST_DB  = os.path.join(BASE_DIR, "data", "historical.db")
BETS_DB  = os.path.join(BASE_DIR, "data", "bets.db")
MODEL_PATH = os.path.join(BASE_DIR, "data", "submodels", "mlb_ats_distributional.pkl")

# ── Date window ────────────────────────────────────────────────────────────────
WINDOW_START = "2026-06-01"
WINDOW_END   = "2026-07-12"
OVERLAP_START = "2026-06-28"

# ── MLB Team → API ID ──────────────────────────────────────────────────────────
TEAM_TO_ID = {
    "Arizona Diamondbacks": 109, "Athletics": 133, "Atlanta Braves": 144,
    "Baltimore Orioles": 110, "Boston Red Sox": 111, "Chicago Cubs": 112,
    "Chicago White Sox": 145, "Cincinnati Reds": 113, "Cleveland Guardians": 114,
    "Colorado Rockies": 115, "Detroit Tigers": 116, "Houston Astros": 117,
    "Kansas City Royals": 118, "Los Angeles Angels": 108, "Los Angeles Dodgers": 119,
    "Miami Marlins": 146, "Milwaukee Brewers": 158, "Minnesota Twins": 142,
    "New York Mets": 121, "New York Yankees": 147, "Philadelphia Phillies": 143,
    "Pittsburgh Pirates": 134, "San Diego Padres": 135, "San Francisco Giants": 137,
    "Seattle Mariners": 136, "St. Louis Cardinals": 138, "Tampa Bay Rays": 139,
    "Texas Rangers": 140, "Toronto Blue Jays": 141, "Washington Nationals": 120,
}

# ── Typical alt-spread decimal odds (fallback if exploration lacks a rung) ─────
TYPICAL_ALT_ODDS = {1.5: 1.637, 2.5: 1.397, 3.5: 1.264, 4.5: 1.182}

# ── Utility ────────────────────────────────────────────────────────────────────
def hdb():
    return sqlite3.connect(HIST_DB)

def bdb():
    return sqlite3.connect(BETS_DB)

def ml_to_prob(ml):
    """American moneyline → implied probability (vig-inclusive)."""
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return -ml / (-ml + 100)

def ml_to_decimal(ml):
    """American moneyline → decimal odds."""
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        return ml / 100 + 1
    else:
        return 100 / (-ml) + 1

def ip_to_float(ip_str):
    """Convert innings pitched '5.2' (5 full + 2 outs) to float."""
    try:
        ip = float(ip_str)
        full = int(ip)
        thirds = round((ip - full) * 10)
        return full + thirds / 3.0
    except Exception:
        return np.nan

print("=" * 70)
print("BetIQ PIT RECONSTRUCTION — MLB Jun 1–Jul 12, 2026")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD BASE GAME LIST
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 1: Loading 562 OOS game list ──")

con = hdb()
games_df = pd.read_sql_query("""
    SELECT g.game_id, g.game_date, g.home_team, g.away_team,
           g.home_score, g.away_score,
           json_extract(ts_h.stats_json, '$.home_probable_pitcher') as h_sp_name,
           json_extract(ts_a.stats_json, '$.away_probable_pitcher') as a_sp_name
    FROM games g
    LEFT JOIN team_stats ts_h ON ts_h.game_id = g.game_id AND ts_h.is_home = 1
    LEFT JOIN team_stats ts_a ON ts_a.game_id = g.game_id AND ts_a.is_home = 0
    WHERE g.sport = 'MLB'
      AND g.game_date >= ? AND g.game_date <= ?
      AND g.home_score IS NOT NULL
    ORDER BY g.game_date, g.game_id
""", con, params=(WINDOW_START, WINDOW_END))
con.close()

print(f"  Games loaded: {len(games_df)}")
print(f"  Date range: {games_df.game_date.min()} → {games_df.game_date.max()}")
print(f"  SP name fill (home): {games_df.h_sp_name.notna().mean():.1%}")
print(f"  SP name fill (away): {games_df.a_sp_name.notna().mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD PITCHER GAME LOGS + FETCH MISSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 2: Pitcher game logs ──")

con = hdb()
pgl = pd.read_sql_query("""
    SELECT player_id, pitcher_name, game_date, ip, er, hits, walks, strikeouts
    FROM pitcher_game_logs
    WHERE season = '2026'
    ORDER BY player_id, game_date
""", con)
con.close()

# Convert ip to float
pgl["ip_f"] = pgl["ip"].apply(ip_to_float)
print(f"  Pitcher logs in DB: {len(pgl)} rows, last date: {pgl.game_date.max()}")

# Build pitcher name → player_id map from DB
name_to_id = (pgl.groupby("pitcher_name")["player_id"].first().to_dict())

# Find pitchers in our game window who might need API fetch
all_sp_names = set(games_df.h_sp_name.dropna()) | set(games_df.a_sp_name.dropna())
# Pitchers missing entirely from DB (started only after Jun 14)
in_db_names = set(pgl.pitcher_name.unique())
unknown_names = all_sp_names - in_db_names
print(f"  Unique SP names in window: {len(all_sp_names)}")
print(f"  SP names not in DB at all: {len(unknown_names)}")
if unknown_names:
    print(f"  Unknown: {sorted(unknown_names)[:10]}")

# Identify player_ids who have games after Jun 14 that we need logs for
games_post_cutoff = games_df[games_df.game_date > "2026-06-14"]
post_sp_names = set(games_post_cutoff.h_sp_name.dropna()) | set(games_post_cutoff.a_sp_name.dropna())
# Map to player_ids
post_pids = {name_to_id[n] for n in post_sp_names if n in name_to_id}
print(f"  Pitchers needing post-Jun14 logs: {len(post_pids)}")

def fetch_pitcher_gamelog_api(player_id: int, season: int = 2026) -> pd.DataFrame:
    """Fetch pitcher game log from MLB Stats API."""
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        d = r.json()
        splits = d.get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"    [WARN] API error for pitcher {player_id}: {e}")
        return pd.DataFrame()

    rows = []
    for s in splits:
        if s.get("gameType") != "R":
            continue
        stat = s.get("stat", {})
        ip_raw = stat.get("inningsPitched", "0")
        rows.append({
            "player_id": player_id,
            "pitcher_name": s.get("player", {}).get("fullName", ""),
            "game_date": s.get("date", ""),
            "ip": ip_raw,
            "ip_f": ip_to_float(ip_raw),
            "er": int(stat.get("earnedRuns", 0)),
            "hits": int(stat.get("hits", 0)),
            "walks": int(stat.get("baseOnBalls", 0)),
            "strikeouts": int(stat.get("strikeOuts", 0)),
        })
    return pd.DataFrame(rows)

# Fetch API logs for pitchers needing post-Jun14 data
api_pgl_rows = []
fetched_api_pitchers = 0
for pid in sorted(post_pids):
    df_api = fetch_pitcher_gamelog_api(pid)
    if not df_api.empty:
        # Only keep rows after Jun 14 (avoid duplicates with DB)
        df_api = df_api[df_api.game_date > "2026-06-14"]
        if not df_api.empty:
            api_pgl_rows.append(df_api)
            fetched_api_pitchers += 1
    time.sleep(0.25)

# Also fetch for pitchers with unknown names (might have different IDs)
for name in sorted(unknown_names):
    # Try to find via searching — skip for now, they'll get NaN imputed
    pass

if api_pgl_rows:
    api_pgl_df = pd.concat(api_pgl_rows, ignore_index=True)
    print(f"  API fetched: {len(api_pgl_df)} pitcher-start rows from {fetched_api_pitchers} pitchers")
    # Combine
    pgl_combined = pd.concat([pgl, api_pgl_df], ignore_index=True)
    pgl_combined = pgl_combined.drop_duplicates(subset=["player_id", "game_date"])
    # Update ip_f for DB rows
    pgl_combined["ip_f"] = pgl_combined.apply(
        lambda r: r.ip_f if pd.notna(r.ip_f) else ip_to_float(r.ip), axis=1
    )
else:
    pgl_combined = pgl.copy()
    print("  No API pitcher logs fetched")

print(f"  Combined pitcher logs: {len(pgl_combined)} rows, last date: {pgl_combined.game_date.max()}")

# Update name_to_id from combined
for _, row in pgl_combined.iterrows():
    name_to_id[row.pitcher_name] = row.player_id

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD TEAM BATTING LOGS + FETCH MISSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 3: Team batting game logs ──")

con = hdb()
tgb = pd.read_sql_query("""
    SELECT team_name, season, game_date, plate_appearances, at_bats, hits,
           walks, hit_by_pitch, sac_flies, total_bases, strikeouts, obp, slg, ops
    FROM team_game_batting
    WHERE season = '2026'
    ORDER BY team_name, game_date
""", con)
con.close()

print(f"  Team batting rows in DB: {len(tgb)}, last date: {tgb.game_date.max()}")

# Identify which teams need post-Jun17 data
all_teams = set(games_df.home_team) | set(games_df.away_team)
print(f"  Unique teams in window: {len(all_teams)}")

def fetch_team_batting_api(team_id: int, team_name: str, season: int = 2026) -> pd.DataFrame:
    """Fetch team batting game log from MLB Stats API."""
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    params = {"stats": "gameLog", "group": "hitting", "season": season}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        d = r.json()
        splits = d.get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"    [WARN] API error for team {team_name}: {e}")
        return pd.DataFrame()

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        rows.append({
            "team_name": team_name,
            "season": str(season),
            "game_date": s.get("date", ""),
            "plate_appearances": int(stat.get("plateAppearances", 0)),
            "at_bats": int(stat.get("atBats", 0)),
            "hits": int(stat.get("hits", 0)),
            "walks": int(stat.get("baseOnBalls", 0)),
            "hit_by_pitch": int(stat.get("hitByPitch", 0)),
            "sac_flies": int(stat.get("sacFlies", 0)),
            "total_bases": int(stat.get("totalBases", 0)),
            "strikeouts": int(stat.get("strikeOuts", 0)),
            "obp": float(stat.get("obp", 0) or 0),
            "slg": float(stat.get("slg", 0) or 0),
            "ops": float(stat.get("ops", 0) or 0),
        })
    return pd.DataFrame(rows)

# Fetch all teams for post-Jun17 data
print(f"  Fetching post-Jun17 team batting from API...")
api_tgb_rows = []
for team_name in sorted(all_teams):
    tid = TEAM_TO_ID.get(team_name)
    if tid is None:
        print(f"    [WARN] No team ID for: {team_name}")
        continue
    df_api = fetch_team_batting_api(tid, team_name)
    if not df_api.empty:
        df_api = df_api[df_api.game_date > "2026-06-17"]
        if not df_api.empty:
            api_tgb_rows.append(df_api)
    time.sleep(0.25)

if api_tgb_rows:
    api_tgb_df = pd.concat(api_tgb_rows, ignore_index=True)
    print(f"  API fetched: {len(api_tgb_df)} team-game batting rows")
    tgb_combined = pd.concat([tgb, api_tgb_df], ignore_index=True)
    tgb_combined = tgb_combined.drop_duplicates(subset=["team_name", "game_date"])
    tgb_combined = tgb_combined.sort_values(["team_name", "game_date"]).reset_index(drop=True)
else:
    tgb_combined = tgb.copy()
    print("  No API team batting logs fetched")

print(f"  Combined team batting: {len(tgb_combined)} rows, last date: {tgb_combined.game_date.max()}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOAD BETTING LINES + TEAM GAME MARGINS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 4: Betting lines + team margins ──")

con = hdb()
bl = pd.read_sql_query("""
    SELECT game_id, game_date, home_team, away_team, close_ml_home, close_ml_away
    FROM betting_lines
    WHERE sport = 'MLB' AND game_date >= '2022-01-01'
    ORDER BY game_date
""", con)

# Load all MLB games for margin computation (need pre-Jun 2026 data too)
all_games = pd.read_sql_query("""
    SELECT game_id, game_date, home_team, away_team, home_score, away_score
    FROM games
    WHERE sport = 'MLB' AND home_score IS NOT NULL AND away_score IS NOT NULL
    ORDER BY game_date
""", con)
con.close()

print(f"  Betting lines: {len(bl)} rows, ML fill: {bl.close_ml_home.notna().mean():.1%}")
print(f"  All MLB games: {len(all_games)} rows, last date: {all_games.game_date.max()}")

# Deduplicate betting lines: keep one row per game_id
bl = bl.sort_values("game_date").drop_duplicates(subset=["game_id"])

# Build team game ledger (both home and away perspective)
home_rows = all_games[["game_date", "home_team", "away_team", "home_score", "away_score"]].copy()
home_rows.columns = ["game_date", "team", "opp", "team_score", "opp_score"]
away_rows = all_games[["game_date", "away_team", "home_team", "away_score", "home_score"]].copy()
away_rows.columns = ["game_date", "team", "opp", "team_score", "opp_score"]
team_ledger = pd.concat([home_rows, away_rows], ignore_index=True)
team_ledger["margin"] = team_ledger["team_score"] - team_ledger["opp_score"]
team_ledger = team_ledger.sort_values(["team", "game_date"]).reset_index(drop=True)

# Merge betting lines onto ledger for opp_impl_prob
# For each game, we need the opponent's ML as a proxy of opponent strength
# bl has close_ml_home / close_ml_away — we need "opponent's implied prob from bet line"
# For opp_strength: use the opponent's WIN probability from the line

# Build a lookup: (game_date, home_team) or (game_date, away_team) → close_ml for each side
bl_home_prob = bl[["game_date", "home_team", "away_team", "close_ml_home", "close_ml_away"]].copy()
bl_home_prob["team"] = bl_home_prob["home_team"]
bl_home_prob["opp_ml"] = bl_home_prob["close_ml_away"]
bl_away_prob = bl[["game_date", "home_team", "away_team", "close_ml_home", "close_ml_away"]].copy()
bl_away_prob["team"] = bl_away_prob["away_team"]
bl_away_prob["opp_ml"] = bl_away_prob["close_ml_home"]

ml_lookup = pd.concat([
    bl_home_prob[["game_date", "team", "opp_ml"]],
    bl_away_prob[["game_date", "team", "opp_ml"]],
], ignore_index=True)
ml_lookup = ml_lookup.drop_duplicates(subset=["game_date", "team"])
ml_lookup["opp_impl_prob"] = ml_lookup["opp_ml"].apply(ml_to_prob)

# Merge opp_impl_prob onto team_ledger
team_ledger = team_ledger.merge(
    ml_lookup[["game_date", "team", "opp_impl_prob"]],
    on=["game_date", "team"], how="left"
)
print(f"  Opp impl prob fill on ledger: {team_ledger.opp_impl_prob.notna().mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FEATURE COMPUTATION PER GAME
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 5: Computing 21 features per game (strict pre-game cutoff) ──")

FEAT_COLS = [
    "sp_era_diff", "ops_diff", "opp_strength_diff", "h_sp_era", "sp_whip_diff",
    "sp_era5_diff", "a_ops", "a_sp_era", "h_sp_era_5", "h_bb_pct",
    "a_opp_impl_prob_5g", "h_opp_impl_prob_5g", "h_sp_era_3", "a_bb_pct",
    "a_sp_era_5", "a_sp_whip_3", "a_margin_10g", "margin_diff_10g",
    "def_diff_10g", "a_slg", "sp_kbb_diff",
]

# Index structures for fast lookups
pgl_idx = pgl_combined.set_index("player_id")

def get_pitcher_stats(player_id, game_date):
    """Return season-to-date + rolling pitcher stats, strictly before game_date."""
    if pd.isna(player_id):
        return {}
    pid = int(player_id)
    try:
        df = pgl_idx.loc[[pid]].copy() if pid in pgl_idx.index else pd.DataFrame()
    except KeyError:
        return {}
    if df.empty:
        return {}

    df = df[df["game_date"] < game_date].sort_values("game_date")
    if df.empty:
        return {}

    total_ip = df["ip_f"].sum()
    if total_ip <= 0:
        return {}

    total_er = df["er"].sum()
    total_hits = df["hits"].sum()
    total_walks = df["walks"].sum()
    total_k = df["strikeouts"].sum()

    season_era  = total_er / total_ip * 9
    season_whip = (total_hits + total_walks) / total_ip
    season_k9   = total_k / total_ip * 9
    season_bb9  = total_walks / total_ip * 9

    # Rolling last-3
    last3 = df.tail(3)
    ip3 = last3["ip_f"].sum()
    era3  = last3["er"].sum() / ip3 * 9 if ip3 > 0 else np.nan
    whip3 = (last3["hits"].sum() + last3["walks"].sum()) / ip3 if ip3 > 0 else np.nan

    # Rolling last-5
    last5 = df.tail(5)
    ip5 = last5["ip_f"].sum()
    era5  = last5["er"].sum() / ip5 * 9 if ip5 > 0 else np.nan

    return {
        "era": season_era, "whip": season_whip,
        "k9": season_k9, "bb9": season_bb9,
        "era3": era3, "whip3": whip3, "era5": era5,
    }

# Pre-index team batting by team_name
tgb_grp = {name: grp for name, grp in tgb_combined.groupby("team_name")}

def get_team_batting(team, game_date):
    """Season-to-date batting stats strictly before game_date."""
    if team not in tgb_grp:
        return {}
    df = tgb_grp[team]
    df = df[(df["game_date"] < game_date) & (df["season"] == "2026")]
    if df.empty:
        return {}
    total_ab = df["at_bats"].sum()
    total_pa = df["plate_appearances"].sum()
    total_bb = df["walks"].sum()
    total_tb = df["total_bases"].sum()
    total_hits = df["hits"].sum()
    total_hbp = df["hit_by_pitch"].sum()
    total_sf  = df["sac_flies"].sum()
    total_k   = df["strikeouts"].sum()

    slg = total_tb / total_ab if total_ab > 0 else np.nan
    obp_denom = total_ab + total_bb + total_hbp + total_sf
    obp = (total_hits + total_bb + total_hbp) / obp_denom if obp_denom > 0 else np.nan
    ops = (obp if not np.isnan(obp) else 0) + (slg if not np.isnan(slg) else 0)
    if np.isnan(obp) and np.isnan(slg):
        ops = np.nan
    bb_pct = total_bb / total_pa if total_pa > 0 else np.nan
    k_pct  = total_k  / total_pa if total_pa > 0 else np.nan

    return {"ops": ops, "slg": slg, "bb_pct": bb_pct, "k_pct": k_pct}

# Pre-index team ledger by team
ledger_grp = {name: grp.sort_values("game_date") for name, grp in team_ledger.groupby("team")}

def get_team_rolling(team, game_date):
    """Rolling 10-game margin and 5-game opp strength, strictly before game_date."""
    if team not in ledger_grp:
        return {}
    df = ledger_grp[team]
    df = df[df["game_date"] < game_date].tail(10)
    if df.empty:
        return {}

    margin_10g = df["margin"].mean()
    pts_allowed_10g = df["opp_score"].mean()

    # opp_impl_prob for last 5 games
    opp5 = df.tail(5)["opp_impl_prob"].dropna()
    opp_impl_5g = opp5.mean() if len(opp5) >= 1 else np.nan

    return {
        "margin_10g": margin_10g,
        "pts_allowed_10g": pts_allowed_10g,
        "opp_impl_prob_5g": opp_impl_5g,
    }

# Build feature matrix
feature_rows = []
print(f"  Processing {len(games_df)} games...")

for i, row in games_df.iterrows():
    gdate = row.game_date
    home  = row.home_team
    away  = row.away_team
    h_sp  = row.h_sp_name
    a_sp  = row.a_sp_name

    # Map pitcher names to IDs
    h_pid = name_to_id.get(h_sp) if pd.notna(h_sp) else None
    a_pid = name_to_id.get(a_sp) if pd.notna(a_sp) else None

    # Pitcher stats
    h_sp_stats = get_pitcher_stats(h_pid, gdate)
    a_sp_stats = get_pitcher_stats(a_pid, gdate)

    # Team batting
    h_bat = get_team_batting(home, gdate)
    a_bat = get_team_batting(away, gdate)

    # Team rolling
    h_roll = get_team_rolling(home, gdate)
    a_roll = get_team_rolling(away, gdate)

    # Compute 21 features
    h_era   = h_sp_stats.get("era", np.nan)
    a_era   = a_sp_stats.get("era", np.nan)
    h_whip  = h_sp_stats.get("whip", np.nan)
    a_whip  = a_sp_stats.get("whip", np.nan)
    h_k9    = h_sp_stats.get("k9",  np.nan)
    a_k9    = a_sp_stats.get("k9",  np.nan)
    h_bb9   = h_sp_stats.get("bb9", np.nan)
    a_bb9   = a_sp_stats.get("bb9", np.nan)
    h_era3  = h_sp_stats.get("era3", np.nan)
    a_era3  = a_sp_stats.get("era3", np.nan)
    h_era5  = h_sp_stats.get("era5", np.nan)
    a_era5  = a_sp_stats.get("era5", np.nan)
    a_whip3 = a_sp_stats.get("whip3", np.nan)

    h_ops   = h_bat.get("ops", np.nan)
    a_ops   = a_bat.get("ops", np.nan)
    h_slg   = h_bat.get("slg", np.nan)
    a_slg   = a_bat.get("slg", np.nan)
    h_bb_pct = h_bat.get("bb_pct", np.nan)
    a_bb_pct = a_bat.get("bb_pct", np.nan)

    h_margin_10g = h_roll.get("margin_10g", np.nan)
    a_margin_10g = a_roll.get("margin_10g", np.nan)
    h_pts_allowed_10g = h_roll.get("pts_allowed_10g", np.nan)
    a_pts_allowed_10g = a_roll.get("pts_allowed_10g", np.nan)
    h_opp_impl_5g = h_roll.get("opp_impl_prob_5g", np.nan)
    a_opp_impl_5g = a_roll.get("opp_impl_prob_5g", np.nan)

    # Derived features
    sp_era_diff    = a_era - h_era if not (np.isnan(a_era) or np.isnan(h_era)) else np.nan
    ops_diff       = h_ops - a_ops if not (np.isnan(h_ops) or np.isnan(a_ops)) else np.nan
    opp_str_diff   = h_opp_impl_5g - a_opp_impl_5g if not (np.isnan(h_opp_impl_5g) or np.isnan(a_opp_impl_5g)) else np.nan
    sp_whip_diff   = a_whip - h_whip if not (np.isnan(a_whip) or np.isnan(h_whip)) else np.nan
    sp_era5_diff   = a_era5 - h_era5 if not (np.isnan(a_era5) or np.isnan(h_era5)) else np.nan
    margin_diff_10g = h_margin_10g - a_margin_10g if not (np.isnan(h_margin_10g) or np.isnan(a_margin_10g)) else np.nan
    # def_diff_10g: away pts_allowed - home pts_allowed (+ve = home better defense)
    def_diff_10g   = a_pts_allowed_10g - h_pts_allowed_10g if not (np.isnan(a_pts_allowed_10g) or np.isnan(h_pts_allowed_10g)) else np.nan
    # sp_kbb_diff: (h_k9 - h_bb9) - (a_k9 - a_bb9)
    h_kbb = h_k9 - h_bb9 if not (np.isnan(h_k9) or np.isnan(h_bb9)) else np.nan
    a_kbb = a_k9 - a_bb9 if not (np.isnan(a_k9) or np.isnan(a_bb9)) else np.nan
    sp_kbb_diff = h_kbb - a_kbb if not (np.isnan(h_kbb) or np.isnan(a_kbb)) else np.nan

    feature_rows.append({
        "game_id": row.game_id,
        "game_date": gdate,
        "home_team": home,
        "away_team": away,
        "home_score": row.home_score,
        "away_score": row.away_score,
        "h_sp_name": h_sp,
        "a_sp_name": a_sp,
        # 21 features
        "sp_era_diff": sp_era_diff,
        "ops_diff": ops_diff,
        "opp_strength_diff": opp_str_diff,
        "h_sp_era": h_era,
        "sp_whip_diff": sp_whip_diff,
        "sp_era5_diff": sp_era5_diff,
        "a_ops": a_ops,
        "a_sp_era": a_era,
        "h_sp_era_5": h_era5,
        "h_bb_pct": h_bb_pct,
        "a_opp_impl_prob_5g": a_opp_impl_5g,
        "h_opp_impl_prob_5g": h_opp_impl_5g,
        "h_sp_era_3": h_era3,
        "a_bb_pct": a_bb_pct,
        "a_sp_era_5": a_era5,
        "a_sp_whip_3": a_whip3,
        "a_margin_10g": a_margin_10g,
        "margin_diff_10g": margin_diff_10g,
        "def_diff_10g": def_diff_10g,
        "a_slg": a_slg,
        "sp_kbb_diff": sp_kbb_diff,
    })

    if (i + 1) % 100 == 0:
        print(f"  ... {i+1}/{len(games_df)} games processed")

feat_df = pd.DataFrame(feature_rows)
print(f"\n  Feature matrix built: {feat_df.shape}")
print("  NaN rates per feature:")
for fc in FEAT_COLS:
    rate = feat_df[fc].isna().mean()
    bar = "!" if rate > 0.3 else ""
    print(f"    {fc:<30} {rate:.1%} {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SPOT CHECK: Jun 15 game
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 6: SPOT CHECK — Jun 15 game feature integrity ──")

spot_games = feat_df[feat_df.game_date == "2026-06-15"]
if spot_games.empty:
    spot_games = feat_df[feat_df.game_date >= "2026-06-15"].head(1)

if not spot_games.empty:
    sg = spot_games.iloc[0]
    print(f"\n  Game: {sg.home_team} vs {sg.away_team} on {sg.game_date}")
    print(f"  Home SP: {sg.h_sp_name}  |  Away SP: {sg.a_sp_name}")
    print(f"  Score: {sg.home_team} {sg.home_score} – {sg.away_team} {sg.away_score}")
    print(f"\n  Feature values (all computed from data BEFORE {sg.game_date}):")
    for fc in FEAT_COLS:
        val = sg[fc]
        print(f"    {fc:<30} = {val:.4f}" if pd.notna(val) else f"    {fc:<30} = NaN")

    # Anti-leak verification
    cutoff = sg.game_date
    h_pid = name_to_id.get(sg.h_sp_name)
    if h_pid:
        same_day_logs = pgl_combined[
            (pgl_combined.player_id == h_pid) &
            (pgl_combined.game_date == cutoff)
        ]
        print(f"\n  Anti-leak check — home SP starts ON {cutoff}: {len(same_day_logs)} (should be 0 in features)")
        prior_starts = pgl_combined[
            (pgl_combined.player_id == h_pid) &
            (pgl_combined.game_date < cutoff)
        ]
        print(f"  Prior starts used for h_sp_era: {len(prior_starts)} (latest: {prior_starts.game_date.max() if len(prior_starts) > 0 else 'none'})")

    h_bat_check = tgb_combined[
        (tgb_combined.team_name == sg.home_team) &
        (tgb_combined.game_date == cutoff) &
        (tgb_combined.season == "2026")
    ]
    print(f"  Anti-leak check — home team batting ON {cutoff}: {len(h_bat_check)} (should be 0 in features)")
    h_bat_prior = tgb_combined[
        (tgb_combined.team_name == sg.home_team) &
        (tgb_combined.game_date < cutoff) &
        (tgb_combined.season == "2026")
    ]
    print(f"  Prior batting games used for h_ops: {len(h_bat_prior)} (latest: {h_bat_prior.game_date.max() if len(h_bat_prior) > 0 else 'none'})")
else:
    print("  No Jun 15 games found in window")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SCORE WITH DISTRIBUTIONAL MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 7: Scoring with mlb_ats_distributional.pkl ──")

with open(MODEL_PATH, "rb") as f:
    art = pickle.load(f)

model   = art["model"]
imputer = art["imputer"]
sigma   = art["sigma"]
feat_names = art["features"]

print(f"  Model: {type(model).__name__}, sigma={sigma:.4f}, n_train={art.get('n_train', '?')}")
print(f"  Features used: {len(feat_names)}")

X_raw = feat_df[FEAT_COLS].values
X_imp = imputer.transform(X_raw)
mu    = model.predict(X_imp)  # E[home_score - away_score]

feat_df["mu_home"] = mu

# Compute dist_wp per rung (underdog = team needing the spread)
# For each game, the underdog gets the + side.
# We score all 4 rungs for the underdog.
# Determine underdog from close_ml in betting_lines
feat_df = feat_df.merge(
    bl[["game_id", "close_ml_home", "close_ml_away"]].drop_duplicates("game_id"),
    on="game_id", how="left"
)

RUNGS = [1.5, 2.5, 3.5, 4.5]

leg_rows = []
for _, row in feat_df.iterrows():
    mu_val = row.mu_home  # E[home - away]
    actual_margin = row.home_score - row.away_score  # home - away

    # Determine underdog
    h_ml = row.get("close_ml_home", np.nan)
    a_ml = row.get("close_ml_away", np.nan)

    # Home underdog if h_ml > 0 (positive = underdog)
    h_prob = ml_to_prob(h_ml) if pd.notna(h_ml) else 0.5
    a_prob = ml_to_prob(a_ml) if pd.notna(a_ml) else 0.5

    for spread in RUNGS:
        # Away team takes +spread (away covers if home_margin < spread, i.e., away_margin > -spread)
        # P(away covers +S) = P(home-away < S) = norm.cdf((S - mu) / sigma)
        a_cover_wp = float(norm.cdf((spread - mu_val) / sigma))
        # Home team takes -spread (home covers if home_margin > spread... wait — for underdogs)
        # We focus on the underdog side.
        # If home is underdog: home +S → P(home - away > -S) = P(home-away > -S) = 1-norm.cdf((-S-mu)/sigma)
        h_cover_wp = float(1 - norm.cdf((-spread - mu_val) / sigma))

        # Game outcome for each rung
        # Away covers +S if home_margin < S (i.e., home did NOT win by more than S)
        a_covered = int(actual_margin < spread)
        # Home covers +S if home_margin > -S
        h_covered  = int(actual_margin > -spread)

        # Add both legs
        leg_rows.append({
            "game_id": row.game_id,
            "game_date": row.game_date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "mu_home": mu_val,
            "team": row.away_team,
            "direction": "away_plus",
            "spread": spread,
            "dist_wp": a_cover_wp,
            "covered": a_covered,
            "is_home_udg": int(h_prob > 0.5),  # home is udg if home prob > 0.5
        })
        leg_rows.append({
            "game_id": row.game_id,
            "game_date": row.game_date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "mu_home": mu_val,
            "team": row.home_team,
            "direction": "home_plus",
            "spread": spread,
            "dist_wp": h_cover_wp,
            "covered": h_covered,
            "is_home_udg": int(h_prob > 0.5),
        })

legs_df = pd.DataFrame(leg_rows)

# Filter to underdog legs only
# away_plus is an underdog leg when away is the underdog (a_ml > 0 or h_ml < 0)
# We merge back to get the right underdog flag

feat_ml = feat_df[["game_id", "close_ml_home", "close_ml_away"]].drop_duplicates("game_id")
legs_df = legs_df.merge(feat_ml, on="game_id", how="left")

legs_df["away_is_udg"] = (legs_df["close_ml_away"] > 0) | (
    (legs_df["close_ml_away"].isna()) &
    (legs_df["close_ml_home"] < 0)
)
legs_df["home_is_udg"] = (legs_df["close_ml_home"] > 0) | (
    (legs_df["close_ml_home"].isna()) &
    (legs_df["close_ml_away"] < 0)
)

# Keep underdog legs: away_plus if away is udg; home_plus if home is udg
udg_legs = legs_df[
    ((legs_df.direction == "away_plus") & legs_df.away_is_udg) |
    ((legs_df.direction == "home_plus") & legs_df.home_is_udg)
].copy()

print(f"\n  Total leg rows (all): {len(legs_df)}")
print(f"  Underdog legs: {len(udg_legs)}")
print(f"  Games with ML data: {feat_df.close_ml_home.notna().sum()}/{len(feat_df)}")
print(f"\n  dist_wp summary (underdog legs):")
for s in RUNGS:
    sub = udg_legs[udg_legs.spread == s]
    print(f"    +{s}: n={len(sub)}, mean_wp={sub.dist_wp.mean():.3f}, actual_cover={sub.covered.mean():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — LOAD EXPLORATION ODDS FOR MARKET PRICING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 8: Loading exploration alt-spread odds ──")

bcon = bdb()
exp_legs = pd.read_sql_query("""
    SELECT DISTINCT l.fixture_id, l.description, l.spread_value, l.open_odds,
                    b.game_date, b.sport
    FROM mock_bet_legs l
    JOIN mock_bets b ON l.mock_bet_id = b.id
    WHERE b.source = 'exploration'
      AND l.is_alt_line = 1
      AND b.game_date >= ? AND b.game_date <= ?
      AND l.open_odds IS NOT NULL
      AND l.spread_value IS NOT NULL
      AND l.spread_value > 0
      AND l.description NOT LIKE '%total%'
      AND l.description NOT LIKE '%Total%'
      AND b.sport = 'MLB'
    ORDER BY b.game_date, l.fixture_id, l.spread_value
""", bcon, params=(WINDOW_START, WINDOW_END))
bcon.close()

print(f"  Exploration alt-spread legs: {len(exp_legs)}")
print(f"  Coverage: {exp_legs.game_date.nunique()} game-days, {exp_legs.fixture_id.nunique()} fixtures")

# Extract team name from description: "TeamName +X.X (Alt Spread)"
def extract_team_from_desc(desc):
    if not desc:
        return None
    m = re.match(r"^(.+?)\s+\+[\d.]+", str(desc))
    return m.group(1).strip() if m else None

exp_legs["team"] = exp_legs["description"].apply(extract_team_from_desc)

# Average odds per (fixture_id, spread_value, team) — deduplicate multiple bookmakers
exp_odds = (
    exp_legs.groupby(["fixture_id", "team", "spread_value", "game_date"])
    .agg(avg_open_odds=("open_odds", "median"))
    .reset_index()
)
exp_odds["market_decimal"] = exp_odds["avg_open_odds"].apply(ml_to_decimal)
print(f"  Deduped exploration odds: {len(exp_odds)} (fixture × team × spread)")

# Now match to our underdog legs by (game_date, team, spread)
# We need to connect fixture_id → game_id

# Build game date+teams → game_id
game_key = feat_df[["game_id", "game_date", "home_team", "away_team"]].copy()
game_key["home_key"] = game_key["game_date"] + "|" + game_key["home_team"]
game_key["away_key"] = game_key["game_date"] + "|" + game_key["away_team"]

exp_odds["team_date_key"] = exp_odds["game_date"] + "|" + exp_odds["team"].fillna("")

# Match by team+date
team_date_to_game = {}
for _, r in game_key.iterrows():
    team_date_to_game[r.home_key] = r.game_id
    team_date_to_game[r.away_key] = r.game_id

exp_odds["game_id"] = exp_odds["team_date_key"].map(team_date_to_game)
matched_exp = exp_odds[exp_odds.game_id.notna()].copy()
print(f"  Exploration odds matched to game_id: {len(matched_exp)}/{len(exp_odds)}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MERGE MARKET ODDS INTO UNDERDOG LEGS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 9: Merging market odds ──")

udg_legs = udg_legs.merge(
    matched_exp[["game_id", "team", "spread_value", "market_decimal"]].rename(
        columns={"spread_value": "spread"}
    ),
    on=["game_id", "team", "spread"],
    how="left"
)

# Fill missing with typical odds
udg_legs["market_decimal"] = udg_legs.apply(
    lambda r: r.market_decimal if pd.notna(r.market_decimal)
    else TYPICAL_ALT_ODDS.get(r.spread, 1.397),
    axis=1
)

exp_fill = udg_legs.market_decimal.notna().mean()
print(f"  Market decimal fill from exploration: {(udg_legs.market_decimal != udg_legs.spread.map(TYPICAL_ALT_ODDS)).mean():.1%} real")
print(f"  Typical odds fallback used: {(udg_legs.apply(lambda r: r.market_decimal == TYPICAL_ALT_ODDS.get(r.spread, 1.397), axis=1)).mean():.1%}")

# Convert negative-odds entries (market_decimal < 1.0 from bad exploration data) to fallback
# Check if any decimals are < 1 (would be invalid)
bad_odds = (udg_legs.market_decimal < 1.05)
if bad_odds.any():
    udg_legs.loc[bad_odds, "market_decimal"] = udg_legs.loc[bad_odds, "spread"].map(TYPICAL_ALT_ODDS).fillna(1.397)
    print(f"  Fixed {bad_odds.sum()} rows with invalid decimal odds")

# Compute E-profit: dist_wp * (decimal - 1) - (1 - dist_wp)
udg_legs["ep"] = udg_legs["dist_wp"] * (udg_legs["market_decimal"] - 1) - (1 - udg_legs["dist_wp"])
# Compute PnL per $1 bet: win → +(decimal-1), loss → -1
udg_legs["pnl_flat"] = udg_legs["covered"] * (udg_legs["market_decimal"] - 1) - (1 - udg_legs["covered"])

print(f"\n  E-profit summary (underdog legs):")
print(f"    Mean EP:    {udg_legs.ep.mean():.4f}")
print(f"    EP > 0:     {(udg_legs.ep > 0).mean():.1%}")
print(f"    Mean PnL:   {udg_legs.pnl_flat.mean():.4f}")
print(f"    Cover rate: {udg_legs.covered.mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — OVERLAP VALIDATION (Jun 28 – Jul 12)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 10: OVERLAP VALIDATION vs live shadow_negbin (Jun 28–Jul 12) ──")

bcon = bdb()
live_legs = pd.read_sql_query("""
    SELECT l.description, l.ats_dist_wp, l.spread_value, l.fixture_id,
           b.game_date
    FROM mock_bet_legs l
    JOIN mock_bets b ON l.mock_bet_id = b.id
    WHERE b.source = 'shadow_negbin' AND b.cohort = 'clean'
      AND b.game_date >= ?
      AND b.game_date <= ?
      AND l.ats_dist_wp IS NOT NULL
""", bcon, params=(OVERLAP_START, WINDOW_END))
bcon.close()

print(f"  Live shadow_negbin legs: {len(live_legs)}")

live_legs["team"] = live_legs["description"].apply(extract_team_from_desc)
# spread_value is NULL in shadow_negbin — extract from description
def extract_spread_from_desc(desc):
    m = re.search(r'\+(\d+\.?\d*)', str(desc))
    return float(m.group(1)) if m else None
live_legs["spread"] = live_legs["spread_value"].apply(
    lambda v: v if pd.notna(v) else None
)
live_legs["spread"] = live_legs.apply(
    lambda r: r.spread if pd.notna(r.spread) else extract_spread_from_desc(r.description),
    axis=1
)
live_legs = live_legs.rename(columns={"ats_dist_wp": "live_dist_wp"})

# Match reconstructed to live by (game_date, team, spread)
# udg legs have game_date, team, spread
overlap_recon = udg_legs[udg_legs.game_date >= OVERLAP_START].copy()

comp = overlap_recon.merge(
    live_legs[["game_date", "team", "spread", "live_dist_wp"]].dropna().drop_duplicates(
        subset=["game_date", "team", "spread"]
    ),
    on=["game_date", "team", "spread"],
    how="inner"
)

print(f"  Matched pairs for validation: {len(comp)}")
if len(comp) >= 10:
    corr = comp[["dist_wp", "live_dist_wp"]].corr().iloc[0, 1]
    mad  = (comp.dist_wp - comp.live_dist_wp).abs().mean()
    bias = (comp.dist_wp - comp.live_dist_wp).mean()
    print(f"  Pearson correlation:   {corr:.4f}")
    print(f"  Mean absolute diff:    {mad:.4f}")
    print(f"  Mean bias (recon-live): {bias:+.4f}")
    print(f"\n  Sample comparisons (first 10):")
    print(f"  {'Date':<12} {'Team':<25} {'Spread':<8} {'Recon':<8} {'Live':<8} {'Diff':<8}")
    for _, r in comp.head(10).iterrows():
        diff = r.dist_wp - r.live_dist_wp
        print(f"  {r.game_date:<12} {str(r.team):<25} +{r.spread:<7} {r.dist_wp:.4f}  {r.live_dist_wp:.4f}  {diff:+.4f}")
else:
    print("  [WARN] Insufficient matched pairs for robust validation")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10B — NET-NEW BIAS MEASUREMENT (PIT vs current features, Jun 1–27)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 10B: NET-NEW BIAS (PIT vs current-feature scoring, Jun 1–27) ──")
print("  Scoring Jun 1–27 games with CURRENT build_feature_matrix (July-final features)")
print("  to measure how much lookahead inflates dist_wp relative to PIT reconstruction")

try:
    sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
    from features import build_feature_matrix

    current_fm = build_feature_matrix("MLB")
    # Get the same games as our net-new period
    net_new_game_ids = feat_df[feat_df.game_date < OVERLAP_START]["game_id"].values

    # Score with current features
    current_games = current_fm[current_fm["game_id"].isin(net_new_game_ids)].copy()
    if len(current_games) > 0:
        X_curr = imputer.transform(current_games[feat_names].values)
        mu_curr = model.predict(X_curr)
        current_games["mu_current"] = mu_curr

        # Match to PIT reconstruction
        pit_games = feat_df[feat_df.game_date < OVERLAP_START][["game_id", "mu_home"]].copy()
        pit_games = pit_games.rename(columns={"mu_home": "mu_pit"})

        bias_comp = current_games[["game_id", "mu_current"]].merge(pit_games, on="game_id")
        if len(bias_comp) > 0:
            mu_bias = (bias_comp.mu_current - bias_comp.mu_pit).mean()
            mu_mad = (bias_comp.mu_current - bias_comp.mu_pit).abs().mean()
            mu_corr = bias_comp[["mu_current", "mu_pit"]].corr().iloc[0, 1]

            # Also compute dist_wp bias at +1.5 rung (most common)
            dwp_pit = norm.cdf((1.5 - bias_comp.mu_pit.values) / sigma)
            dwp_curr = norm.cdf((1.5 - bias_comp.mu_current.values) / sigma)
            dwp_bias = np.mean(dwp_curr - dwp_pit)
            dwp_mad = np.mean(np.abs(dwp_curr - dwp_pit))

            print(f"  Matched games (Jun 1–27): {len(bias_comp)}")
            print(f"  mu bias (current - PIT):     {mu_bias:+.4f}")
            print(f"  mu MAD:                      {mu_mad:.4f}")
            print(f"  mu correlation:              {mu_corr:.4f}")
            print(f"  dist_wp@+1.5 bias (cur-PIT): {dwp_bias:+.4f}")
            print(f"  dist_wp@+1.5 MAD:            {dwp_mad:.4f}")
            print(f"\n  Compare to OVERLAP bias:     +0.009 (Jun 28–Jul 12)")
            print(f"  Net-new bias is {'LARGER' if abs(dwp_bias) > 0.009 else 'similar'} than overlap bias")
            print(f"  → Confirms {'user hypothesis: early games have more lookahead contamination' if abs(dwp_bias) > 0.015 else 'modest inflation across all periods'}")
        else:
            print("  [WARN] No matched games for bias comparison")
    else:
        print("  [WARN] No current-feature games found for Jun 1–27")
except Exception as e:
    print(f"  [ERROR] Could not compute net-new bias: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — E-PROFIT ALLOCATION SIM
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 11: E-PROFIT ALLOCATION SIM ──")

EP_THRESHOLD = 0.0  # filter legs with EP > 0

def run_sim(legs: pd.DataFrame, label: str) -> dict:
    """Run three allocation strategies on a set of legs."""
    n = len(legs)
    if n == 0:
        return {"label": label, "n": 0}

    # Strategy 1: Flat (bet all legs equally)
    flat_pnl = legs["pnl_flat"].values
    flat_daily = legs.groupby("game_date")["pnl_flat"].mean()

    # Strategy 2: EP-filter (bet only legs with EP > 0, flat among those)
    ep_legs = legs[legs["ep"] > EP_THRESHOLD]
    if len(ep_legs) > 0:
        ep_filter_pnl = ep_legs["pnl_flat"].values
        ep_filter_daily = ep_legs.groupby("game_date")["pnl_flat"].mean()
    else:
        ep_filter_pnl = np.array([])
        ep_filter_daily = pd.Series(dtype=float)

    # Strategy 3: EP-weighted (weight by EP magnitude, only positive EP legs)
    if len(ep_legs) > 0:
        # Normalize weights within each day
        def weighted_pnl_day(grp):
            w = grp["ep"].clip(lower=0)
            if w.sum() == 0:
                return grp["pnl_flat"].mean()
            return (grp["pnl_flat"] * w / w.sum()).sum()
        ep_weighted_daily = ep_legs.groupby("game_date").apply(weighted_pnl_day)
    else:
        ep_weighted_daily = pd.Series(dtype=float)

    def summarize(pnl_arr, daily_series, strat_name):
        if len(pnl_arr) == 0:
            return {}
        mean_pnl = np.mean(pnl_arr)
        std_pnl  = np.std(pnl_arr, ddof=1)
        sharpe   = mean_pnl / std_pnl * np.sqrt(len(pnl_arr)) if std_pnl > 0 else 0
        t_stat, p_val = ttest_1samp(pnl_arr, 0)
        daily_arr = daily_series.values
        daily_sharpe = np.mean(daily_arr) / np.std(daily_arr, ddof=1) * np.sqrt(len(daily_arr)) if len(daily_arr) > 1 and np.std(daily_arr, ddof=1) > 0 else 0
        return {
            "strat": strat_name,
            "n_legs": len(pnl_arr),
            "n_days": len(daily_arr),
            "mean_pnl": mean_pnl,
            "total_pnl": np.sum(pnl_arr),
            "std_pnl": std_pnl,
            "leg_sharpe": sharpe,
            "daily_sharpe": daily_sharpe,
            "t_stat": t_stat,
            "p_val": p_val,
            "cover_rate": (pnl_arr > 0).mean(),
        }

    s1 = summarize(flat_pnl, flat_daily, "Flat")
    s2 = summarize(ep_filter_pnl, ep_filter_daily, "EP-filter")
    s3 = summarize(ep_weighted_daily.values if len(ep_weighted_daily) else np.array([]),
                   ep_weighted_daily, "EP-weighted")

    return {"label": label, "n": n, "strategies": [s1, s2, s3]}

def print_sim_results(results: dict):
    label = results["label"]
    print(f"\n  [{label}]  Total legs: {results['n']}")
    if results["n"] == 0:
        return
    for s in results.get("strategies", []):
        if not s:
            continue
        print(f"  {'Strategy':<15} {'N legs':<8} {'Mean PnL':<10} {'Total PnL':<12} "
              f"{'Cover%':<8} {'DailySharpe':<13} {'p-val':<8}")
        break
    for s in results.get("strategies", []):
        if not s:
            continue
        print(f"  {s.get('strat','?'):<15} {s.get('n_legs',0):<8} "
              f"{s.get('mean_pnl',0):>+8.4f}  "
              f"{s.get('total_pnl',0):>+10.4f}   "
              f"{s.get('cover_rate',0):>6.1%}   "
              f"{s.get('daily_sharpe',0):>+8.3f}    "
              f"{s.get('p_val',1):.4f}")

# Split periods
early_legs = udg_legs[udg_legs.game_date < OVERLAP_START]
overlap_legs = udg_legs[udg_legs.game_date >= OVERLAP_START]
full_legs = udg_legs

print(f"\n  Period breakdown:")
print(f"    Net-new (Jun 1–27):    {len(early_legs)} legs, {early_legs.game_date.nunique()} days")
print(f"    Overlap (Jun 28–Jul12):{len(overlap_legs)} legs, {overlap_legs.game_date.nunique()} days")
print(f"    Combined (Jun 1–Jul12):{len(full_legs)} legs, {full_legs.game_date.nunique()} days")

print("\n  ── Net-new period (Jun 1–27, pure OOS) ──")
r1 = run_sim(early_legs, "Jun 1–27 (Net-new OOS)")
print_sim_results(r1)

print("\n  ── Overlap period (Jun 28–Jul 12) ──")
r2 = run_sim(overlap_legs, "Jun 28–Jul 12 (Overlap)")
print_sim_results(r2)

print("\n  ── Combined (Jun 1–Jul 12) ──")
r3 = run_sim(full_legs, "Jun 1–Jul 12 (Full OOS)")
print_sim_results(r3)

# Paired comparison: Flat vs EP-filter
print("\n  ── PAIRED t-TEST: Flat vs EP-filter (daily PnL, full period) ──")
flat_daily_full  = full_legs.groupby("game_date")["pnl_flat"].mean()
ep_legs_full     = full_legs[full_legs["ep"] > EP_THRESHOLD]
ep_daily_full    = ep_legs_full.groupby("game_date")["pnl_flat"].mean() if len(ep_legs_full) > 0 else pd.Series(dtype=float)

# Align on common dates
common_dates = flat_daily_full.index.intersection(ep_daily_full.index)
if len(common_dates) >= 5:
    t, p = ttest_rel(ep_daily_full[common_dates], flat_daily_full[common_dates])
    diff = (ep_daily_full[common_dates] - flat_daily_full[common_dates])
    print(f"  Common days: {len(common_dates)}")
    print(f"  Mean EP-filter daily PnL: {ep_daily_full[common_dates].mean():+.4f}")
    print(f"  Mean Flat daily PnL:      {flat_daily_full[common_dates].mean():+.4f}")
    print(f"  Mean difference (EP-Flat): {diff.mean():+.4f}")
    print(f"  Paired t-stat: {t:+.3f}   p={p:.4f}  {'SIGNIFICANT (p<0.10)' if p < 0.10 else 'NOT significant'}")
else:
    print("  Insufficient common days for paired t-test")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — PER-RUNG BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 12: Per-rung E-profit breakdown ──")

print(f"\n  {'Rung':<8} {'N_legs':<8} {'EP_mean':<10} {'Flat_PnL':<10} {'EPfilt_N':<10} {'EPfilt_PnL':<12} {'Cover%':<8}")
for s in RUNGS:
    sub = full_legs[full_legs.spread == s]
    ep_sub = sub[sub.ep > EP_THRESHOLD]
    if len(sub) == 0:
        continue
    print(f"  +{s:<7} {len(sub):<8} {sub.ep.mean():>+8.4f}  {sub.pnl_flat.mean():>+8.4f}   "
          f"{len(ep_sub):<10} {ep_sub.pnl_flat.mean() if len(ep_sub) > 0 else np.nan:>+8.4f}    "
          f"{sub.covered.mean():>6.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — SUMMARY & INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_games = len(feat_df)
overlap_games = len(feat_df[feat_df.game_date >= OVERLAP_START])
nan_rates = {fc: feat_df[fc].isna().mean() for fc in FEAT_COLS}
worst_feat = max(nan_rates, key=nan_rates.get)

print(f"\n  DATA STATUS")
print(f"    OOS games reconstructed:  {total_games} ({WINDOW_START} – {WINDOW_END})")
print(f"    Overlap games (Jun28+):   {overlap_games}")
print(f"    Pitcher logs (combined):  {len(pgl_combined)} rows, max date {pgl_combined.game_date.max()}")
print(f"    Team batting (combined):  {len(tgb_combined)} rows, max date {tgb_combined.game_date.max()}")
print(f"    Worst NaN feature:        {worst_feat} @ {nan_rates[worst_feat]:.1%}")

if len(comp) >= 10:
    print(f"\n  OVERLAP VALIDATION")
    print(f"    Pearson r (recon vs live): {corr:.4f}")
    print(f"    MAD:                       {mad:.4f}")
    print(f"    Bias (recon - live):      {bias:+.4f}")
    verdict = "PASS" if corr > 0.90 and mad < 0.05 else ("MARGINAL" if corr > 0.70 else "FAIL")
    print(f"    Verdict:                  {verdict}")

print(f"\n  E-PROFIT SIM")
for r in [r1, r2, r3]:
    if r["n"] == 0:
        continue
    strats = r.get("strategies", [])
    flat = next((s for s in strats if s.get("strat") == "Flat"), {})
    epf  = next((s for s in strats if s.get("strat") == "EP-filter"), {})
    print(f"\n    {r['label']}")
    if flat:
        print(f"      Flat    → mean PnL/leg {flat.get('mean_pnl',0):+.4f}, daily Sharpe {flat.get('daily_sharpe',0):+.3f}, p={flat.get('p_val',1):.4f}")
    if epf:
        print(f"      EP-filt → mean PnL/leg {epf.get('mean_pnl',0):+.4f}, daily Sharpe {epf.get('daily_sharpe',0):+.3f}, p={epf.get('p_val',1):.4f}")

print("\n  ANTI-LEAK GUARANTEE: All features computed from game_date < D only.")
print("  API fetch: pitcher logs (post Jun 14) + team batting (post Jun 17).")
print("  Market odds: FanDuel exploration legs where available, typical odds otherwise.")
print("\nDone.")
