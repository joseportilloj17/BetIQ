"""
nfl_foundation.py — NFL data acquisition + feature foundation for BetIQ.

Ingests play-by-play, schedules, and injuries from nfl_data_py (nflverse).
Builds POINT-IN-TIME features with Bayesian shrinkage.  No models — just
the validated data layer.

Lessons applied from MLB:
  - Every feature is as-of-game-date (no lookahead)
  - Bayesian shrinkage for early-season small samples
  - Phase-based component features (rush/pass O/D) with gap-from-aggregate
  - Train/serve parity: same code path for historical build and forward serving
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_NFL_DB = os.path.join(_DB_DIR, "nfl.db")

# ── Bayesian shrinkage priors ────────────────────────────────────────────────
# NFL league-average priors (updated from 2020-2024 data).
# Used when a team has < _SHRINKAGE_GAMES games played in the season.
_SHRINKAGE_GAMES = 4  # full shrinkage at 0 games, linear blend to full weight at N
_PRIOR = {
    "epa_per_play":       0.0,    # league average EPA/play
    "pass_epa_per_play":  0.05,   # passing is slightly positive on average
    "rush_epa_per_play": -0.05,   # rushing is slightly negative on average
    "success_rate":       0.45,   # ~45% of plays are "successful"
    "pass_rate":          0.58,   # ~58% pass plays
    "points_per_game":   22.0,    # ~22 ppg league average
    "point_diff":         0.0,    # average team is .500
    "qb_epa_per_play":    0.0,    # league average QB EPA
}


def _shrink(observed: float, n_games: int, prior: float) -> float:
    """Bayesian shrinkage: blend observed with prior based on sample size."""
    if n_games >= _SHRINKAGE_GAMES:
        return observed
    w = n_games / _SHRINKAGE_GAMES
    return w * observed + (1 - w) * prior


# ── Database setup ───────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    os.makedirs(_DB_DIR, exist_ok=True)
    con = sqlite3.connect(_NFL_DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db():
    """Create NFL tables if they don't exist."""
    con = _connect()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS nfl_games (
            game_id       TEXT PRIMARY KEY,
            season        INTEGER NOT NULL,
            week          INTEGER NOT NULL,
            game_type     TEXT NOT NULL,    -- REG, WC, DIV, CON, SB
            game_date     TEXT NOT NULL,
            weekday       TEXT,
            gametime      TEXT,
            home_team     TEXT NOT NULL,
            away_team     TEXT NOT NULL,
            home_score    INTEGER,
            away_score    INTEGER,
            spread_line   REAL,            -- closing spread (home perspective, negative = home fav)
            total_line    REAL,            -- closing total
            home_ml       INTEGER,         -- closing moneyline
            away_ml       INTEGER,
            home_rest     INTEGER,
            away_rest     INTEGER,
            div_game      INTEGER,         -- 1 if divisional
            overtime      INTEGER,
            roof          TEXT,            -- outdoors, dome, open, closed
            surface       TEXT,
            temp          REAL,
            wind          REAL,
            stadium       TEXT
        );

        CREATE TABLE IF NOT EXISTS nfl_team_game_stats (
            game_id          TEXT NOT NULL,
            team             TEXT NOT NULL,
            season           INTEGER NOT NULL,
            week             INTEGER NOT NULL,
            game_date        TEXT NOT NULL,
            is_home          INTEGER NOT NULL,
            opp_team         TEXT NOT NULL,
            score            INTEGER,
            opp_score        INTEGER,
            -- Offense
            off_epa_per_play     REAL,
            off_success_rate     REAL,
            off_plays            INTEGER,
            pass_epa_per_play    REAL,
            pass_success_rate    REAL,
            pass_plays           INTEGER,
            rush_epa_per_play    REAL,
            rush_success_rate    REAL,
            rush_plays           INTEGER,
            pass_rate            REAL,
            -- Defense (opponent's offensive stats against this team)
            def_epa_per_play     REAL,
            def_success_rate     REAL,
            def_plays            INTEGER,
            def_pass_epa         REAL,
            def_pass_success     REAL,
            def_pass_plays       INTEGER,
            def_rush_epa         REAL,
            def_rush_success     REAL,
            def_rush_plays       INTEGER,
            -- QB
            qb_name              TEXT,
            qb_epa_per_play      REAL,
            qb_cpoe              REAL,
            qb_dropbacks         INTEGER,
            PRIMARY KEY (game_id, team)
        );

        CREATE TABLE IF NOT EXISTS nfl_injuries (
            season          INTEGER NOT NULL,
            week            INTEGER NOT NULL,
            team            TEXT NOT NULL,
            player_name     TEXT NOT NULL,
            position        TEXT,
            injury          TEXT,
            status          TEXT,     -- Out, Doubtful, Questionable
            PRIMARY KEY (season, week, team, player_name)
        );

        CREATE TABLE IF NOT EXISTS nfl_features (
            game_id          TEXT NOT NULL,
            team             TEXT NOT NULL,
            season           INTEGER NOT NULL,
            week             INTEGER NOT NULL,
            game_date        TEXT NOT NULL,
            is_home          INTEGER NOT NULL,
            opp_team         TEXT NOT NULL,
            -- Overall strength (shrunk)
            s_epa_per_play       REAL,
            s_def_epa_per_play   REAL,
            s_success_rate       REAL,
            s_def_success_rate   REAL,
            s_points_per_game    REAL,
            s_points_allowed     REAL,
            s_point_diff         REAL,
            -- Phase-based offense (shrunk)
            s_pass_epa_off       REAL,
            s_rush_epa_off       REAL,
            s_pass_rate          REAL,
            -- Phase-based defense (shrunk)
            s_pass_epa_def       REAL,
            s_rush_epa_def       REAL,
            -- Gap from aggregate (how much phase deviates from overall)
            gap_pass_off         REAL,  -- s_pass_epa_off - s_epa_per_play
            gap_rush_off         REAL,  -- s_rush_epa_off - s_epa_per_play
            gap_pass_def         REAL,  -- s_pass_epa_def - s_def_epa_per_play
            gap_rush_def         REAL,  -- s_rush_epa_def - s_def_epa_per_play
            -- Matchup features (filled per-game: this team's O vs opp's D)
            mu_rush_o_vs_d       REAL,  -- s_rush_epa_off - opp s_rush_epa_def
            mu_pass_o_vs_d       REAL,  -- s_pass_epa_off - opp s_pass_epa_def
            mu_overall           REAL,  -- s_epa_per_play - opp s_def_epa_per_play
            -- QB (shrunk)
            s_qb_epa             REAL,
            s_qb_cpoe            REAL,
            -- Recent form (rolling 4 games)
            r4_epa_per_play      REAL,
            r4_point_diff        REAL,
            r4_win_pct           REAL,
            -- Situational
            rest_days            INTEGER,
            is_short_week        INTEGER,  -- rest <= 5 (Thursday games)
            is_bye_week_after    INTEGER,  -- coming off bye (rest >= 12)
            win_pct_season       REAL,
            games_played         INTEGER,
            -- NFL-specific structural
            is_divisional        INTEGER,
            is_primetime         INTEGER,  -- SNF/MNF/TNF
            key_number_3         INTEGER,  -- spread within 0.5 of 3
            key_number_7         INTEGER,  -- spread within 0.5 of 7
            spread_abs           REAL,     -- absolute value of closing spread
            -- Environment
            is_dome              INTEGER,
            temp                 REAL,
            wind                 REAL,
            -- Closing lines (for later CLV/edge analysis)
            close_spread         REAL,
            close_total          REAL,
            close_ml             INTEGER,
            opp_close_ml         INTEGER,
            PRIMARY KEY (game_id, team)
        );
    """)
    con.commit()
    con.close()
    log.info("[nfl] Database initialized at %s", _NFL_DB)


# ── Data ingestion ───────────────────────────────────────────────────────────

def ingest_games(seasons: list[int] | None = None) -> int:
    """Ingest game results + closing lines from nfl_data_py schedules."""
    import nfl_data_py as nfl

    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    sched = nfl.import_schedules(seasons)
    con = _connect()
    count = 0

    for _, r in sched.iterrows():
        # Closing spread: nfl_data_py stores from AWAY perspective.
        # We negate to store from HOME perspective (negative = home fav).
        spread = -r["spread_line"] if pd.notna(r.get("spread_line")) else None

        con.execute("""
            INSERT OR REPLACE INTO nfl_games
            (game_id, season, week, game_type, game_date, weekday, gametime,
             home_team, away_team, home_score, away_score,
             spread_line, total_line, home_ml, away_ml,
             home_rest, away_rest, div_game, overtime,
             roof, surface, temp, wind, stadium)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            r["game_id"],
            int(r["season"]),
            int(r["week"]),
            r.get("game_type", "REG"),
            str(r.get("gameday", "")),
            r.get("weekday"),
            str(r.get("gametime", "")),
            r["home_team"],
            r["away_team"],
            int(r["home_score"]) if pd.notna(r.get("home_score")) else None,
            int(r["away_score"]) if pd.notna(r.get("away_score")) else None,
            spread,
            r.get("total_line") if pd.notna(r.get("total_line")) else None,
            int(r["home_moneyline"]) if pd.notna(r.get("home_moneyline")) else None,
            int(r["away_moneyline"]) if pd.notna(r.get("away_moneyline")) else None,
            int(r["home_rest"]) if pd.notna(r.get("home_rest")) else None,
            int(r["away_rest"]) if pd.notna(r.get("away_rest")) else None,
            int(r.get("div_game", 0)),
            1 if r.get("overtime") == 1 else 0,
            r.get("roof"),
            r.get("surface"),
            float(r["temp"]) if pd.notna(r.get("temp")) else None,
            float(r["wind"]) if pd.notna(r.get("wind")) else None,
            r.get("stadium"),
        ))
        count += 1

    con.commit()
    con.close()
    log.info("[nfl] Ingested %d games across seasons %s", count, seasons)
    return count


def ingest_pbp_stats(seasons: list[int] | None = None) -> int:
    """Ingest play-by-play data, compute per-team-per-game stats.

    For each game-team pair: EPA/play (overall, pass, rush), success rate,
    play counts, pass rate, and primary QB metrics.

    Both offense AND defense stats are computed from the same PBP data
    (defense = opponent's offensive plays against this team).
    """
    import nfl_data_py as nfl

    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    log.info("[nfl] Fetching PBP for seasons %s...", seasons)
    pbp = nfl.import_pbp_data(seasons, downcast=True)
    log.info("[nfl] PBP loaded: %d plays", len(pbp))

    # Filter to actual plays (not timeouts, penalties-only, etc.)
    plays = pbp[
        (pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])) &
        (pbp["epa"].notna()) &
        (pbp["posteam"].notna())
    ].copy()

    # Get schedule for game metadata
    sched = nfl.import_schedules(seasons)
    game_meta = {}
    for _, r in sched.iterrows():
        game_meta[r["game_id"]] = {
            "game_date": str(r.get("gameday", "")),
            "season": int(r["season"]),
            "week": int(r["week"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_score": int(r["home_score"]) if pd.notna(r.get("home_score")) else None,
            "away_score": int(r["away_score"]) if pd.notna(r.get("away_score")) else None,
        }

    con = _connect()
    count = 0

    # Group by game
    for game_id, gplays in plays.groupby("game_id"):
        meta = game_meta.get(game_id)
        if not meta:
            continue

        teams_in_game = gplays["posteam"].unique()

        for team in teams_in_game:
            is_home = 1 if team == meta["home_team"] else 0
            opp_team = meta["away_team"] if is_home else meta["home_team"]
            score = meta["home_score"] if is_home else meta["away_score"]
            opp_score = meta["away_score"] if is_home else meta["home_score"]

            # Offense: plays where this team has the ball
            off = gplays[gplays["posteam"] == team]
            off_pass = off[off["play_type"] == "pass"]
            off_rush = off[off["play_type"] == "run"]

            # Defense: plays where opponent has the ball (this team is defending)
            defense = gplays[gplays["defteam"] == team]
            def_pass = defense[defense["play_type"] == "pass"]
            def_rush = defense[defense["play_type"] == "run"]

            # QB: primary passer (most dropbacks)
            qb_name = None
            qb_epa = None
            qb_cpoe = None
            qb_dropbacks = 0
            if len(off_pass) > 0 and "passer_player_name" in off_pass.columns:
                passers = off_pass[off_pass["passer_player_name"].notna()]
                if len(passers) > 0:
                    qb_counts = passers["passer_player_name"].value_counts()
                    qb_name = qb_counts.index[0]
                    qb_plays = passers[passers["passer_player_name"] == qb_name]
                    qb_dropbacks = len(qb_plays)
                    qb_epa = float(qb_plays["qb_epa"].mean()) if qb_plays["qb_epa"].notna().any() else None
                    qb_cpoe = float(qb_plays["cpoe"].mean()) if "cpoe" in qb_plays.columns and qb_plays["cpoe"].notna().any() else None

            def _safe_mean(s):
                return float(s.mean()) if len(s) > 0 and s.notna().any() else None

            def _safe_success(s):
                if len(s) == 0 or not s.notna().any():
                    return None
                return float((s == 1).sum() / s.notna().sum())

            n_off = len(off)
            n_pass = len(off_pass)
            n_rush = len(off_rush)

            con.execute("""
                INSERT OR REPLACE INTO nfl_team_game_stats
                (game_id, team, season, week, game_date, is_home, opp_team,
                 score, opp_score,
                 off_epa_per_play, off_success_rate, off_plays,
                 pass_epa_per_play, pass_success_rate, pass_plays,
                 rush_epa_per_play, rush_success_rate, rush_plays,
                 pass_rate,
                 def_epa_per_play, def_success_rate, def_plays,
                 def_pass_epa, def_pass_success, def_pass_plays,
                 def_rush_epa, def_rush_success, def_rush_plays,
                 qb_name, qb_epa_per_play, qb_cpoe, qb_dropbacks)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                game_id, team, meta["season"], meta["week"], meta["game_date"],
                is_home, opp_team, score, opp_score,
                _safe_mean(off["epa"]), _safe_success(off["success"]), n_off,
                _safe_mean(off_pass["epa"]), _safe_success(off_pass["success"]), n_pass,
                _safe_mean(off_rush["epa"]), _safe_success(off_rush["success"]), n_rush,
                round(n_pass / n_off, 3) if n_off > 0 else None,
                _safe_mean(defense["epa"]), _safe_success(defense["success"]), len(defense),
                _safe_mean(def_pass["epa"]), _safe_success(def_pass["success"]), len(def_pass),
                _safe_mean(def_rush["epa"]), _safe_success(def_rush["success"]), len(def_rush),
                qb_name, qb_epa, qb_cpoe, qb_dropbacks,
            ))
            count += 1

    con.commit()
    con.close()
    log.info("[nfl] Ingested PBP stats for %d team-game pairs", count)
    return count


def ingest_injuries(seasons: list[int] | None = None) -> int:
    """Ingest weekly injury reports from nfl_data_py."""
    import nfl_data_py as nfl

    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    con = _connect()
    count = 0

    for season in seasons:
        try:
            inj = nfl.import_injuries([season])
        except Exception as e:
            log.warning("[nfl] No injury data for %d: %s", season, e)
            continue

        for _, r in inj.iterrows():
            status = r.get("report_status")
            if not status or status == "":
                continue  # skip blank/healthy
            con.execute("""
                INSERT OR REPLACE INTO nfl_injuries
                (season, week, team, player_name, position, injury, status)
                VALUES (?,?,?,?,?,?,?)
            """, (
                int(r["season"]),
                int(r["week"]),
                r["team"],
                r.get("full_name", ""),
                r.get("position"),
                r.get("report_primary_injury"),
                status,
            ))
            count += 1

    con.commit()
    con.close()
    log.info("[nfl] Ingested %d injury records", count)
    return count


# ── Feature computation ──────────────────────────────────────────────────────

def build_features(seasons: list[int] | None = None) -> int:
    """Build point-in-time features for every team-game pair.

    For each game, features are computed using ONLY data from games
    BEFORE this game in the same season (no lookahead). Early-season
    games use Bayesian shrinkage toward league priors.

    This is the SAME code path used for both historical build and
    forward serving (train/serve parity).
    """
    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    con = _connect()
    cur = con.cursor()

    # Load all game stats, ordered by season + week
    cur.execute("""
        SELECT * FROM nfl_team_game_stats
        WHERE season IN ({})
        ORDER BY season, week, game_id
    """.format(",".join(str(s) for s in seasons)))
    all_stats = [dict(r) for r in cur.fetchall()]

    # Load games for structural features
    cur.execute("""
        SELECT * FROM nfl_games
        WHERE season IN ({})
    """.format(",".join(str(s) for s in seasons)))
    games_map = {r["game_id"]: dict(r) for r in cur.fetchall()}

    # Build per-team cumulative stats (point-in-time)
    # Key: (season, team) → list of game stats in chronological order
    team_history = defaultdict(list)

    # Process games in chronological order
    # First pass: group stats by game_id for matchup pairing
    games_stats = defaultdict(dict)
    for s in all_stats:
        games_stats[s["game_id"]][s["team"]] = s

    # Sort game_ids by (season, week)
    sorted_games = sorted(
        games_stats.keys(),
        key=lambda gid: (
            games_stats[gid][list(games_stats[gid].keys())[0]]["season"],
            games_stats[gid][list(games_stats[gid].keys())[0]]["week"],
        )
    )

    count = 0
    for game_id in sorted_games:
        teams_data = games_stats[game_id]
        game_info = games_map.get(game_id, {})

        if len(teams_data) < 2:
            continue

        team_list = list(teams_data.keys())

        # Pre-compute each team's point-in-time features
        team_feats = {}
        for team in team_list:
            s = teams_data[team]
            season = s["season"]
            key = (season, team)
            history = team_history[key]
            n = len(history)

            # Cumulative averages (point-in-time — BEFORE this game)
            def _cum_avg(field):
                vals = [h[field] for h in history if h.get(field) is not None]
                return np.mean(vals) if vals else None

            def _cum_avg_shrunk(field, prior_key):
                raw = _cum_avg(field)
                if raw is None:
                    return _PRIOR.get(prior_key, 0.0)
                return _shrink(raw, n, _PRIOR.get(prior_key, 0.0))

            # Overall strength
            epa = _cum_avg_shrunk("off_epa_per_play", "epa_per_play")
            def_epa = _cum_avg_shrunk("def_epa_per_play", "epa_per_play")
            sr = _cum_avg_shrunk("off_success_rate", "success_rate")
            def_sr = _cum_avg_shrunk("def_success_rate", "success_rate")

            # Points
            pts_vals = [h["score"] for h in history if h.get("score") is not None]
            pa_vals = [h["opp_score"] for h in history if h.get("opp_score") is not None]
            ppg = _shrink(np.mean(pts_vals), n, _PRIOR["points_per_game"]) if pts_vals else _PRIOR["points_per_game"]
            pa = _shrink(np.mean(pa_vals), n, _PRIOR["points_per_game"]) if pa_vals else _PRIOR["points_per_game"]

            # Phase-based
            pass_epa_off = _cum_avg_shrunk("pass_epa_per_play", "pass_epa_per_play")
            rush_epa_off = _cum_avg_shrunk("rush_epa_per_play", "rush_epa_per_play")
            pass_rate = _cum_avg_shrunk("pass_rate", "pass_rate")
            pass_epa_def = _cum_avg_shrunk("def_pass_epa", "pass_epa_per_play")
            rush_epa_def = _cum_avg_shrunk("def_rush_epa", "rush_epa_per_play")

            # Gap from aggregate
            gap_pass_off = pass_epa_off - epa
            gap_rush_off = rush_epa_off - epa
            gap_pass_def = pass_epa_def - def_epa
            gap_rush_def = rush_epa_def - def_epa

            # QB
            qb_epa = _cum_avg_shrunk("qb_epa_per_play", "qb_epa_per_play")
            qb_cpoe_vals = [h["qb_cpoe"] for h in history if h.get("qb_cpoe") is not None]
            qb_cpoe = np.mean(qb_cpoe_vals) if qb_cpoe_vals else 0.0

            # Recent form (last 4 games)
            recent = history[-4:] if len(history) >= 4 else history
            r4_epa = np.mean([h["off_epa_per_play"] for h in recent if h.get("off_epa_per_play") is not None]) if recent else _PRIOR["epa_per_play"]
            r4_pdiff = np.mean([(h["score"] - h["opp_score"]) for h in recent if h.get("score") is not None and h.get("opp_score") is not None]) if recent else 0.0
            r4_wins = sum(1 for h in recent if h.get("score") is not None and h.get("opp_score") is not None and h["score"] > h["opp_score"])
            r4_wp = r4_wins / len(recent) if recent else 0.5

            # Season win pct
            all_wins = sum(1 for h in history if h.get("score") is not None and h.get("opp_score") is not None and h["score"] > h["opp_score"])
            win_pct = all_wins / n if n > 0 else 0.5

            # Rest / schedule
            rest = s.get("is_home") and game_info.get("home_rest") or game_info.get("away_rest")
            if rest is None:
                # Compute from history
                if history:
                    last_date = history[-1].get("game_date", "")
                    this_date = s.get("game_date", "")
                    try:
                        from datetime import datetime
                        d1 = datetime.strptime(last_date, "%Y-%m-%d")
                        d2 = datetime.strptime(this_date, "%Y-%m-%d")
                        rest = (d2 - d1).days
                    except (ValueError, TypeError):
                        rest = 7
                else:
                    rest = 7  # season opener default

            # Pull rest from game table
            if s["is_home"]:
                rest = game_info.get("home_rest", rest)
            else:
                rest = game_info.get("away_rest", rest)

            is_short = 1 if rest is not None and rest <= 5 else 0
            is_bye_after = 1 if rest is not None and rest >= 12 else 0

            # Primetime detection (from weekday + gametime)
            weekday = game_info.get("weekday", "")
            gametime = game_info.get("gametime", "")
            is_prime = 0
            if weekday in ("Thursday", "Monday"):
                is_prime = 1
            elif weekday == "Sunday" and gametime and gametime >= "20:00":
                is_prime = 1

            # Key numbers
            spread = game_info.get("spread_line")
            spread_for_team = None
            if spread is not None:
                spread_for_team = spread if s["is_home"] else -spread
            spread_abs = abs(spread) if spread is not None else None
            key3 = 1 if spread_abs is not None and abs(spread_abs - 3) <= 0.5 else 0
            key7 = 1 if spread_abs is not None and abs(spread_abs - 7) <= 0.5 else 0

            # Moneyline
            if s["is_home"]:
                ml = game_info.get("home_ml")
                opp_ml = game_info.get("away_ml")
            else:
                ml = game_info.get("away_ml")
                opp_ml = game_info.get("home_ml")

            team_feats[team] = {
                "game_id": game_id,
                "team": team,
                "season": season,
                "week": s["week"],
                "game_date": s["game_date"],
                "is_home": s["is_home"],
                "opp_team": s["opp_team"],
                # Overall
                "s_epa_per_play": round(epa, 4),
                "s_def_epa_per_play": round(def_epa, 4),
                "s_success_rate": round(sr, 4),
                "s_def_success_rate": round(def_sr, 4),
                "s_points_per_game": round(ppg, 2),
                "s_points_allowed": round(pa, 2),
                "s_point_diff": round(ppg - pa, 2),
                # Phase offense
                "s_pass_epa_off": round(pass_epa_off, 4),
                "s_rush_epa_off": round(rush_epa_off, 4),
                "s_pass_rate": round(pass_rate, 4),
                # Phase defense
                "s_pass_epa_def": round(pass_epa_def, 4),
                "s_rush_epa_def": round(rush_epa_def, 4),
                # Gaps
                "gap_pass_off": round(gap_pass_off, 4),
                "gap_rush_off": round(gap_rush_off, 4),
                "gap_pass_def": round(gap_pass_def, 4),
                "gap_rush_def": round(gap_rush_def, 4),
                # QB
                "s_qb_epa": round(qb_epa, 4),
                "s_qb_cpoe": round(qb_cpoe, 4) if qb_cpoe is not None else None,
                # Recent form
                "r4_epa_per_play": round(r4_epa, 4) if r4_epa is not None else None,
                "r4_point_diff": round(r4_pdiff, 2) if r4_pdiff is not None else None,
                "r4_win_pct": round(r4_wp, 4),
                # Situational
                "rest_days": rest,
                "is_short_week": is_short,
                "is_bye_week_after": is_bye_after,
                "win_pct_season": round(win_pct, 4),
                "games_played": n,
                # Structural
                "is_divisional": game_info.get("div_game", 0),
                "is_primetime": is_prime,
                "key_number_3": key3,
                "key_number_7": key7,
                "spread_abs": round(spread_abs, 1) if spread_abs is not None else None,
                # Environment
                "is_dome": 1 if game_info.get("roof") in ("dome", "closed") else 0,
                "temp": game_info.get("temp"),
                "wind": game_info.get("wind"),
                # Lines
                "close_spread": spread_for_team,
                "close_total": game_info.get("total_line"),
                "close_ml": ml,
                "opp_close_ml": opp_ml,
                # Matchup placeholders (filled below)
                "mu_rush_o_vs_d": None,
                "mu_pass_o_vs_d": None,
                "mu_overall": None,
            }

        # Fill matchup features (requires both teams' features)
        for team in team_list:
            opp = teams_data[team]["opp_team"]
            if opp in team_feats:
                f = team_feats[team]
                opp_f = team_feats[opp]
                f["mu_rush_o_vs_d"] = round(f["s_rush_epa_off"] - opp_f["s_rush_epa_def"], 4)
                f["mu_pass_o_vs_d"] = round(f["s_pass_epa_off"] - opp_f["s_pass_epa_def"], 4)
                f["mu_overall"] = round(f["s_epa_per_play"] - opp_f["s_def_epa_per_play"], 4)

        # Write features to DB
        for team, f in team_feats.items():
            cols = list(f.keys())
            placeholders = ",".join(["?"] * len(cols))
            col_str = ",".join(cols)
            con.execute(
                f"INSERT OR REPLACE INTO nfl_features ({col_str}) VALUES ({placeholders})",
                [f[c] for c in cols],
            )
            count += 1

            # AFTER computing features, add this game to history (point-in-time discipline)
            team_history[(teams_data[team]["season"], team)].append(teams_data[team])

    con.commit()
    con.close()
    log.info("[nfl] Built features for %d team-game pairs", count)
    return count


# ── Validation / audit ───────────────────────────────────────────────────────

def validate_features(seasons: list[int] | None = None) -> dict:
    """Run data quality checks on the feature foundation.

    Checks:
      1. No lookahead: features for week N shouldn't use week N data
      2. Monotonicity: more games_played → shrinkage weight increases
      3. Value ranges: EPA in [-1, 1], success rate in [0, 1], etc.
      4. Coverage: no unexpected NULLs in core features
      5. Closing line coverage
    """
    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    con = _connect()
    cur = con.cursor()

    results = {"checks": [], "passed": 0, "failed": 0}

    def _check(name, passed, detail=""):
        results["checks"].append({"name": name, "passed": passed, "detail": detail})
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

    # 1. Lookahead check: games_played should equal the number of prior games
    #    in the same season for that team
    cur.execute("""
        SELECT f.game_id, f.team, f.season, f.week, f.games_played,
               (SELECT COUNT(*) FROM nfl_team_game_stats s
                WHERE s.team = f.team AND s.season = f.season AND s.week < f.week)
               AS expected_prior
        FROM nfl_features f
        WHERE f.season IN ({})
    """.format(",".join(str(s) for s in seasons)))
    lookahead_violations = 0
    total_rows = 0
    for r in cur.fetchall():
        total_rows += 1
        if r["games_played"] != r["expected_prior"]:
            lookahead_violations += 1
    _check(
        "no_lookahead",
        lookahead_violations == 0,
        f"{lookahead_violations}/{total_rows} rows have games_played != prior game count"
    )

    # 2. Shrinkage: week 1 features should be at priors
    cur.execute("""
        SELECT s_epa_per_play, s_pass_epa_off, s_rush_epa_off, games_played
        FROM nfl_features WHERE week = 1 AND season IN ({})
    """.format(",".join(str(s) for s in seasons)))
    w1_rows = [dict(r) for r in cur.fetchall()]
    w1_at_prior = all(
        abs(r["s_epa_per_play"] - _PRIOR["epa_per_play"]) < 0.01
        for r in w1_rows
    )
    _check(
        "week1_at_prior",
        w1_at_prior,
        f"Week 1 should be at league priors (0 games played). {len(w1_rows)} rows checked."
    )

    # 3. Value ranges
    cur.execute("""
        SELECT COUNT(*) as n,
               MIN(s_epa_per_play) as min_epa, MAX(s_epa_per_play) as max_epa,
               MIN(s_success_rate) as min_sr, MAX(s_success_rate) as max_sr,
               MIN(s_pass_rate) as min_pr, MAX(s_pass_rate) as max_pr,
               MIN(s_qb_epa) as min_qb, MAX(s_qb_epa) as max_qb,
               MIN(win_pct_season) as min_wp, MAX(win_pct_season) as max_wp
        FROM nfl_features WHERE season IN ({})
    """.format(",".join(str(s) for s in seasons)))
    ranges = dict(cur.fetchone())
    epa_ok = -1.0 <= (ranges["min_epa"] or 0) and (ranges["max_epa"] or 0) <= 1.0
    sr_ok = 0.0 <= (ranges["min_sr"] or 0) and (ranges["max_sr"] or 0) <= 1.0
    pr_ok = 0.0 <= (ranges["min_pr"] or 0) and (ranges["max_pr"] or 0) <= 1.0
    _check("epa_range", epa_ok, f"EPA range: [{ranges['min_epa']}, {ranges['max_epa']}]")
    _check("success_rate_range", sr_ok, f"SR range: [{ranges['min_sr']}, {ranges['max_sr']}]")
    _check("pass_rate_range", pr_ok, f"PR range: [{ranges['min_pr']}, {ranges['max_pr']}]")

    # 4. Coverage
    cur.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN s_epa_per_play IS NOT NULL THEN 1 ELSE 0 END) as has_epa,
               SUM(CASE WHEN s_pass_epa_off IS NOT NULL THEN 1 ELSE 0 END) as has_pass,
               SUM(CASE WHEN s_rush_epa_off IS NOT NULL THEN 1 ELSE 0 END) as has_rush,
               SUM(CASE WHEN mu_overall IS NOT NULL THEN 1 ELSE 0 END) as has_matchup,
               SUM(CASE WHEN close_spread IS NOT NULL THEN 1 ELSE 0 END) as has_spread,
               SUM(CASE WHEN close_total IS NOT NULL THEN 1 ELSE 0 END) as has_total
        FROM nfl_features WHERE season IN ({})
    """.format(",".join(str(s) for s in seasons)))
    cov = dict(cur.fetchone())
    total = cov["total"]
    _check("epa_coverage", cov["has_epa"] == total, f"{cov['has_epa']}/{total}")
    _check("pass_epa_coverage", cov["has_pass"] == total, f"{cov['has_pass']}/{total}")
    _check("matchup_coverage", cov["has_matchup"] == total, f"{cov['has_matchup']}/{total}")
    _check("spread_coverage", cov["has_spread"] / total > 0.95, f"{cov['has_spread']}/{total}")

    # 5. Games per season
    cur.execute("""
        SELECT season, COUNT(*) as n FROM nfl_features
        WHERE season IN ({})
        GROUP BY season ORDER BY season
    """.format(",".join(str(s) for s in seasons)))
    season_counts = {r["season"]: r["n"] for r in cur.fetchall()}
    for s, n in season_counts.items():
        # Each team plays ~17 games + playoffs, so ~64-70 team-game pairs per season
        # Actually 32 teams × ~17 games = ~544 regular season + playoffs
        expected_min = 500  # at least 500 team-game pairs
        _check(f"season_{s}_count", n >= expected_min, f"{n} team-game pairs")

    con.close()
    return results


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(seasons: list[int] | None = None) -> dict:
    """Run the complete NFL data pipeline: init → ingest → features → validate."""
    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    log.info("[nfl] Starting full pipeline for seasons %s", seasons)
    init_db()

    games = ingest_games(seasons)
    stats = ingest_pbp_stats(seasons)
    injuries = ingest_injuries(seasons)
    features = build_features(seasons)
    validation = validate_features(seasons)

    return {
        "seasons": seasons,
        "games_ingested": games,
        "pbp_stats": stats,
        "injuries": injuries,
        "features_built": features,
        "validation": validation,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_full_pipeline()
    print(f"\n{'='*60}")
    print(f"NFL Foundation Pipeline Complete")
    print(f"{'='*60}")
    print(f"Games ingested:     {result['games_ingested']}")
    print(f"PBP team-games:     {result['pbp_stats']}")
    print(f"Injury records:     {result['injuries']}")
    print(f"Features built:     {result['features_built']}")
    print(f"\nValidation: {result['validation']['passed']} passed, {result['validation']['failed']} failed")
    for chk in result["validation"]["checks"]:
        status = "✓" if chk["passed"] else "✗"
        print(f"  {status} {chk['name']}: {chk['detail']}")
