"""
brief_builder.py — Structured NFL brief: Game → Team → Player hierarchy
with injury implication layer (beneficiaries, harmed, opponent effects).

Replaces the flat injury-feed dump with a scannable, decision-relevant brief
that surfaces dependency effects and connects to active prop lines.
"""
import logging
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
_NFL_DB = os.path.join(_DATA_DIR, "nfl.db")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# ─────────────────────────────────────────────────────────────
# Team name mapping (ESPN full → nflverse abbreviation)
# ─────────────────────────────────────────────────────────────
_TEAM_ABBREV = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
_ABBREV_TO_FULL = {v: k for k, v in _TEAM_ABBREV.items()}

# Injury statuses that matter for the brief (ranked by severity)
_INJURY_STATUSES = {"Out": 3, "Doubtful": 2, "Questionable": 1}
_ACTIVE_STATUSES = {"Active", "Probable"}

# Position groups for redistribution logic
_POS_GROUPS = {
    "WR": "pass_catcher", "TE": "pass_catcher", "WR/TE": "pass_catcher",
    "RB": "rusher", "FB": "rusher",
    "QB": "quarterback",
    "OT": "oline", "OG": "oline", "C": "oline", "OL": "oline",
    "T": "oline", "G": "oline", "LT": "oline", "RT": "oline",
    "LG": "oline", "RG": "oline",
}

# Position groups whose injuries have offensive implications
_OFFENSIVE_POSITIONS = {"WR", "TE", "RB", "FB", "QB", "OT", "OG", "C", "OL", "T", "G"}

# Prop market display names
_PROP_DISPLAY = {
    "player_pass_yds": "pass yds",
    "player_pass_tds": "pass TDs",
    "player_rush_yds": "rush yds",
    "player_reception_yds": "rec yds",
    "player_receptions": "receptions",
    "player_anytime_td": "anytime TD",
}

# Number of players to show per team (cap for scannability)
_MAX_PLAYERS_PER_TEAM = 8

# Minimum usage thresholds — OUT player must have this much share
# for their absence to generate implications
_MIN_SHARE_FOR_IMPLICATIONS = 0.08  # 8% target or carry share

# Max beneficiaries to list per injured player
_MAX_BENEFICIARIES_PER_INJURY = 3


# ─────────────────────────────────────────────────────────────
# Current-season roster + depth chart (source of truth for teams)
# ─────────────────────────────────────────────────────────────
def _load_current_roster(season: int) -> dict:
    """Load current-season roster and depth chart from nfl_data_py.

    This is the SOLE source of truth for team assignment and starter/backup
    determination. PBP usage data provides production baselines, but team
    membership comes from HERE.

    Returns:
        {normalized_name: {
            team, position, status, depth_rank, full_name, week
        }}

    depth_rank: 1=starter, 2=backup, 3+=deep backup, None=unknown
    """
    roster = {}

    try:
        import nfl_data_py as nfl
    except ImportError:
        logger.warning("[brief_builder] nfl_data_py not available")
        return roster

    # ── Step 1: Load seasonal roster (team + position + status) ──
    try:
        rosters_df = nfl.import_seasonal_rosters([season])
        # Use most recent week available
        latest_week = rosters_df["week"].max()
        latest = rosters_df[rosters_df["week"] == latest_week]
        # Only active players (ACT) and injured reserve (RES) — skip CUT, DEV
        active_statuses = {"ACT", "RES", "INA", "EXE"}
        latest = latest[latest["status"].isin(active_statuses)]

        for _, row in latest.iterrows():
            name = row.get("player_name", "")
            team = row.get("team", "")
            if not name or not team:
                continue
            norm = _normalize_name(name)
            roster[norm] = {
                "team": team,
                "position": row.get("position", ""),
                "status": row.get("status", ""),
                "depth_rank": None,  # filled from depth chart below
                "full_name": name,
                "week": int(latest_week),
            }
        logger.info("[brief_builder] Loaded %d roster entries (week %d, season %d)",
                    len(roster), latest_week, season)
    except Exception as e:
        logger.warning("[brief_builder] Failed to load seasonal rosters: %s", e)

    # ── Step 2: Load depth chart (starter/backup rank) ──
    try:
        dc = nfl.import_depth_charts([season])
        if dc.empty:
            logger.warning("[brief_builder] Empty depth chart for %d", season)
            return roster

        # Get most recent entry per player+team+position
        dc_sorted = dc.sort_values("dt", ascending=False)
        dc_latest = dc_sorted.drop_duplicates(subset=["team", "player_name", "pos_abb"], keep="first")

        for _, row in dc_latest.iterrows():
            name = row.get("player_name", "")
            team = row.get("team", "")
            if not name or not team:
                continue
            norm = _normalize_name(name)
            rank = row.get("pos_rank")
            pos = row.get("pos_abb", "")

            if norm in roster and roster[norm]["team"] == team:
                # Update existing roster entry with depth rank
                roster[norm]["depth_rank"] = int(rank) if rank is not None else None
            elif norm not in roster:
                # Player in depth chart but not in roster snapshot — add them
                roster[norm] = {
                    "team": team,
                    "position": _dc_pos_to_roster_pos(pos),
                    "status": "ACT",
                    "depth_rank": int(rank) if rank is not None else None,
                    "full_name": name,
                    "week": 0,
                }

        ranked = sum(1 for v in roster.values() if v["depth_rank"] is not None)
        logger.info("[brief_builder] Depth chart: %d/%d players have rank",
                    ranked, len(roster))
    except Exception as e:
        logger.warning("[brief_builder] Failed to load depth chart: %s", e)

    return roster


def _dc_pos_to_roster_pos(dc_pos: str) -> str:
    """Map depth chart position abbreviations to roster positions."""
    mapping = {
        "QB": "QB", "RB": "RB", "FB": "FB", "WR": "WR", "TE": "TE",
        "LT": "OT", "RT": "OT", "LG": "OG", "RG": "OG", "C": "C",
        "LDE": "DE", "RDE": "DE", "LDT": "DT", "RDT": "DT", "NT": "DT",
        "SLB": "LB", "MLB": "LB", "WLB": "LB", "LILB": "LB", "RILB": "LB",
        "LCB": "CB", "RCB": "CB", "NB": "CB",
        "FS": "S", "SS": "S",
        "K": "K", "PK": "K", "P": "P", "LS": "LS",
        "KR": "WR", "PR": "WR", "H": "WR",
    }
    return mapping.get(dc_pos, dc_pos)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
def _load_week_schedule(season: int, week: int) -> list[dict]:
    """Load NFL games for a specific week from nfl_data_py."""
    try:
        import nfl_data_py as nfl
        sched = nfl.import_schedules([season])
        week_games = sched[sched["week"] == week]
        games = []
        import math
        for _, row in week_games.iterrows():
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            # NaN means not yet played
            if isinstance(home_score, float) and math.isnan(home_score):
                home_score = None
            if isinstance(away_score, float) and math.isnan(away_score):
                away_score = None
            games.append({
                "game_id": row.get("game_id", ""),
                "gameday": str(row.get("gameday", "")),
                "gametime": str(row.get("gametime", "")),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "spread": row.get("spread_line"),
                "total": row.get("total_line"),
                "home_score": home_score,
                "away_score": away_score,
            })
        return games
    except Exception as e:
        logger.warning("[brief_builder] Failed to load schedule: %s", e)
        return []


def _find_current_week(target_date: str) -> tuple[int, int]:
    """Find the NFL season and week for a given date."""
    try:
        import nfl_data_py as nfl
        from datetime import datetime as dt
        target = dt.strptime(target_date, "%Y-%m-%d")
        year = target.year if target.month >= 3 else target.year - 1
        sched = nfl.import_schedules([year])
        # Find the week whose games bracket the target date
        for week in sorted(sched["week"].unique()):
            wk = sched[sched["week"] == week]
            min_date = wk["gameday"].min()
            max_date = wk["gameday"].max()
            if str(min_date) <= target_date <= str(max_date):
                return year, int(week)
            # Also catch dates between weeks (e.g., Tuesday before Sunday)
            if target_date < str(min_date):
                return year, int(week)
        return year, int(sched["week"].max())
    except Exception as e:
        logger.warning("[brief_builder] Failed to find current week: %s", e)
        return 2026, 4


def _load_roster_positions(season: int) -> dict:
    """Load player positions from nfl_data_py rosters.

    Returns {(abbreviated_name, team): position} for name matching.
    Stores multiple name variants for fuzzy matching.
    """
    try:
        import nfl_data_py as nfl
        rosters = nfl.import_seasonal_rosters([season])
        positions = {}
        for _, row in rosters.iterrows():
            name = row.get("player_name", "")
            team = row.get("team", "")
            pos = row.get("position", "")
            if not (name and team and pos):
                continue
            # Store full name
            positions[(name, team)] = pos
            # Store abbreviated: "Saquon Barkley" → "S.Barkley"
            abbrev = _normalize_name(name)
            positions[(abbrev, team)] = pos
            # Store last name for fallback: "Barkley"
            parts = name.split()
            if len(parts) >= 2:
                positions[(parts[-1], team)] = pos
                # PBP form: first initial + last name
                # "T.J. Hockenson" → "T.Hockenson", "Saquon Barkley" → "S.Barkley"
                pbp_name = name[0] + "." + parts[-1]
                positions[(pbp_name, team)] = pos
        return positions
    except Exception as e:
        logger.warning("[brief_builder] Failed to load rosters: %s", e)
        return {}


def _load_player_usage(season: int, current_roster: dict = None) -> dict:
    """Compute player usage from PBP data, remapped to current-season teams.

    Production baselines (target_share, carry_share, dropbacks) come from PBP.
    When current_roster is provided, each player's TEAM is overridden with
    their current roster team. Players who changed teams are flagged with
    pbp_team (old team) so callers know the baseline is from a different context.

    Players on the current roster with no PBP history (rookies, practice-squad
    elevations) are added with zero usage so they appear in teammate lookups.

    Returns {key: {player_name, team, position, targets, target_share,
             carries, carry_share, receptions, rec_yds, rush_yds, games,
             pbp_team (if different), depth_rank (if available)}}.
    """
    pbp_season = season
    try:
        import nfl_data_py as nfl
        import pandas as pd
        try:
            pbp = nfl.import_pbp_data([season])
        except Exception:
            pbp_season = season - 1
            logger.info("[brief_builder] PBP %d unavailable, using %d for usage", season, pbp_season)
            pbp = nfl.import_pbp_data([pbp_season])
    except Exception as e:
        logger.warning("[brief_builder] Failed to load PBP: %s", e)
        # If PBP entirely unavailable, build usage purely from roster
        return _build_usage_from_roster_only(current_roster)

    # Load roster positions for PBP-era position tagging
    roster_pos = _load_roster_positions(pbp_season)

    usage = {}

    # Receiving usage
    pass_plays = pbp[
        (pbp["pass_attempt"] == 1) & (pbp["receiver_player_name"].notna())
    ]
    if not pass_plays.empty:
        player_tgt = pass_plays.groupby(
            ["posteam", "receiver_player_name"]
        ).agg(
            targets=("pass_attempt", "sum"),
            receptions=("complete_pass", "sum"),
            rec_yds=("yards_gained", "sum"),
            games=("week", "nunique"),
        ).reset_index()

        team_tgt = pass_plays.groupby("posteam")["pass_attempt"].sum()

        for _, row in player_tgt.iterrows():
            name = row["receiver_player_name"]
            team = row["posteam"]
            team_total = team_tgt.get(team, 1)
            pos = roster_pos.get((name, team), "WR")
            if pos in ("RB", "FB"):
                role = "rusher"
            elif pos == "QB":
                role = "quarterback"
            else:
                role = "pass_catcher"
            usage[f"{name}_{team}"] = {
                "player_name": name,
                "team": team,
                "position": pos,
                "role": role,
                "targets": int(row["targets"]),
                "target_share": round(row["targets"] / team_total, 3),
                "receptions": int(row["receptions"]),
                "rec_yds": int(row["rec_yds"]),
                "games": int(row["games"]),
                "carries": 0,
                "carry_share": 0.0,
                "rush_yds": 0,
            }

    # Rushing usage
    rush_plays = pbp[
        (pbp["rush_attempt"] == 1) & (pbp["rusher_player_name"].notna())
    ]
    if not rush_plays.empty:
        player_rush = rush_plays.groupby(
            ["posteam", "rusher_player_name"]
        ).agg(
            carries=("rush_attempt", "sum"),
            rush_yds=("yards_gained", "sum"),
            games=("week", "nunique"),
        ).reset_index()

        team_rush = rush_plays.groupby("posteam")["rush_attempt"].sum()

        for _, row in player_rush.iterrows():
            name = row["rusher_player_name"]
            team = row["posteam"]
            key = f"{name}_{team}"
            team_total = team_rush.get(team, 1)
            if key in usage:
                usage[key]["carries"] = int(row["carries"])
                usage[key]["carry_share"] = round(row["carries"] / team_total, 3)
                usage[key]["rush_yds"] = int(row["rush_yds"])
            else:
                pos = roster_pos.get((name, team), "RB")
                usage[key] = {
                    "player_name": name,
                    "team": team,
                    "position": pos,
                    "role": "rusher" if pos in ("RB", "FB") else "quarterback",
                    "targets": 0,
                    "target_share": 0.0,
                    "receptions": 0,
                    "rec_yds": 0,
                    "games": int(row["games"]),
                    "carries": int(row["carries"]),
                    "carry_share": round(row["carries"] / team_total, 3),
                    "rush_yds": int(row["rush_yds"]),
                }

    # QB passing volume
    qb_plays = pbp[
        (pbp["qb_dropback"] == 1) & (pbp["passer_player_name"].notna())
    ]
    if not qb_plays.empty:
        qb_stats = qb_plays.groupby(
            ["posteam", "passer_player_name"]
        ).agg(
            dropbacks=("qb_dropback", "sum"),
            pass_yds=("yards_gained", "sum"),
            games=("week", "nunique"),
        ).reset_index()

        for _, row in qb_stats.iterrows():
            name = row["passer_player_name"]
            team = row["posteam"]
            key = f"{name}_{team}"
            if key not in usage:
                usage[key] = {
                    "player_name": name,
                    "team": team,
                    "position": "QB",
                    "role": "quarterback",
                    "targets": 0, "target_share": 0.0,
                    "receptions": 0, "rec_yds": 0,
                    "carries": 0, "carry_share": 0.0, "rush_yds": 0,
                    "games": int(row["games"]),
                }
            usage[key]["dropbacks"] = int(row["dropbacks"])
            usage[key]["pass_yds"] = int(row["pass_yds"])
            usage[key]["role"] = "quarterback"
            usage[key]["position"] = "QB"

    # ── Remap to current-season teams ──
    if current_roster:
        usage = _remap_usage_to_current_roster(usage, current_roster)

    return usage


def _remap_usage_to_current_roster(usage: dict, roster: dict) -> dict:
    """Remap PBP-derived usage entries to current-season team assignments.

    - Players who changed teams: re-key under new team, flag pbp_team
    - Players on roster with no PBP history: add with zero usage
    - Players in PBP but no longer on any roster: drop (retired/cut)
    """
    remapped = {}
    team_changes = 0
    added_from_roster = 0

    # Build a set of normalized names already processed
    processed_norms = set()

    for old_key, entry in usage.items():
        norm = _normalize_name(entry["player_name"])
        roster_info = roster.get(norm)

        if roster_info:
            current_team = roster_info["team"]
            pbp_team = entry["team"]

            new_entry = {**entry, "team": current_team}

            # Override position from current roster if available
            if roster_info.get("position"):
                new_entry["position"] = roster_info["position"]
                pg = _POS_GROUPS.get(roster_info["position"], "other")
                if pg == "quarterback":
                    new_entry["role"] = "quarterback"
                elif pg == "rusher":
                    new_entry["role"] = "rusher"
                elif pg == "pass_catcher":
                    new_entry["role"] = "pass_catcher"

            # Flag team change
            if current_team != pbp_team:
                new_entry["pbp_team"] = pbp_team
                team_changes += 1

            # Attach depth rank
            if roster_info.get("depth_rank") is not None:
                new_entry["depth_rank"] = roster_info["depth_rank"]

            new_key = f"{entry['player_name']}_{current_team}"
            remapped[new_key] = new_entry
            processed_norms.add(norm)
        else:
            # Not on current roster — keep as-is (could be a PBP-only player
            # from the prior season who is retired/cut, but may still be
            # referenced in injury reports or props)
            remapped[old_key] = entry
            processed_norms.add(norm)

    # Add roster players with no PBP history (rookies, new arrivals)
    for norm, info in roster.items():
        if norm in processed_norms:
            continue
        # Only add offensive skill players (QB, RB, WR, TE, OL)
        pos = info.get("position", "")
        if pos not in _OFFENSIVE_POSITIONS:
            continue
        if info.get("status") not in ("ACT", "RES"):
            continue

        pg = _POS_GROUPS.get(pos, "other")
        role = {"quarterback": "quarterback", "rusher": "rusher",
                "pass_catcher": "pass_catcher"}.get(pg, "other")

        new_key = f"{info['full_name']}_{info['team']}"
        # Use full name as PBP-style abbreviated name for the key
        pbp_name = _normalize_name(info["full_name"])
        new_key = f"{pbp_name}_{info['team']}"

        remapped[new_key] = {
            "player_name": pbp_name,
            "team": info["team"],
            "position": pos,
            "role": role,
            "targets": 0, "target_share": 0.0,
            "receptions": 0, "rec_yds": 0,
            "carries": 0, "carry_share": 0.0, "rush_yds": 0,
            "games": 0,
            "depth_rank": info.get("depth_rank"),
            "full_name": info.get("full_name", ""),
            "rookie": True,
        }
        added_from_roster += 1

    logger.info("[brief_builder] Usage remap: %d team changes, %d added from roster "
                "(no PBP history)", team_changes, added_from_roster)
    return remapped


def _build_usage_from_roster_only(roster: dict) -> dict:
    """Build minimal usage dict from roster when PBP is entirely unavailable."""
    if not roster:
        return {}
    usage = {}
    for norm, info in roster.items():
        pos = info.get("position", "")
        if pos not in _OFFENSIVE_POSITIONS:
            continue
        if info.get("status") not in ("ACT", "RES"):
            continue
        pg = _POS_GROUPS.get(pos, "other")
        role = {"quarterback": "quarterback", "rusher": "rusher",
                "pass_catcher": "pass_catcher"}.get(pg, "other")
        key = f"{norm}_{info['team']}"
        usage[key] = {
            "player_name": norm,
            "team": info["team"],
            "position": pos,
            "role": role,
            "targets": 0, "target_share": 0.0,
            "receptions": 0, "rec_yds": 0,
            "carries": 0, "carry_share": 0.0, "rush_yds": 0,
            "games": 0,
            "depth_rank": info.get("depth_rank"),
            "full_name": info.get("full_name", ""),
        }
    return usage


def _load_active_props(game_dates: list[str]) -> dict:
    """Load active player prop lines from historical.db.

    Returns {player_name: [{market, point, best_over, best_under}]}.
    """
    if not os.path.exists(_HIST_DB) or not game_dates:
        return {}

    try:
        con = sqlite3.connect(_HIST_DB)
        placeholders = ",".join("?" for _ in game_dates)
        # Only fetch PRIMARY markets (exclude alternates which have many lines)
        rows = con.execute(f"""
            SELECT player_name, market_key, point, over_under,
                   MAX(price) as best_price, game_date,
                   COUNT(*) as book_count
            FROM player_props
            WHERE game_date IN ({placeholders})
              AND market_key IN (
                'player_pass_yds','player_pass_tds','player_rush_yds',
                'player_reception_yds','player_receptions','player_anytime_td'
              )
            GROUP BY player_name, market_key, point, over_under, game_date
        """, game_dates).fetchall()
        con.close()
    except Exception as e:
        logger.warning("[brief_builder] Failed to load props: %s", e)
        return {}

    # For each player × market, keep only the MOST COMMON line
    # (the one offered by the most books = the consensus primary line)
    from collections import Counter
    line_counts = defaultdict(lambda: Counter())
    line_prices = defaultdict(lambda: defaultdict(dict))

    for name, market, point, ou, price, gdate, book_count in rows:
        line_counts[(name, market)][(point)] += book_count
        line_prices[(name, market, point)][ou] = price

    props = defaultdict(list)
    for (name, market), counts in line_counts.items():
        # Pick the line offered by the most books
        best_point = counts.most_common(1)[0][0]
        prices = line_prices[(name, market, best_point)]
        props[name].append({
            "market": market,
            "display": _PROP_DISPLAY.get(market, market),
            "point": best_point,
            "over": prices.get("over"),
            "under": prices.get("under"),
        })

    # Sort each player's props for consistent display
    for name in props:
        props[name].sort(key=lambda x: x["market"])

    return dict(props)


def _load_projections_and_edges(season: int, week: int) -> tuple[dict, dict]:
    """Load fantasy projections and consensus prop-gap edges.

    Returns:
        projections: {player_name: {stat_type: projected_value}}
        edges: {player_name: {stat_type: {mean, line, gap, gap_pct, direction, confidence}}}
    """
    projections = defaultdict(dict)
    edges = defaultdict(dict)

    if not os.path.exists(_HIST_DB):
        return dict(projections), dict(edges)

    try:
        con = sqlite3.connect(_HIST_DB)

        # ── Projections ──
        rows = con.execute("""
            SELECT player_name, team, stat_type, projected_value
            FROM fantasy_projections
            WHERE season = ? AND week = ?
        """, (season, week)).fetchall()
        for name, team, stat, val in rows:
            projections[name][stat] = val

        # ── Consensus edges (projection vs prop line) ──
        rows = con.execute("""
            SELECT player_name, stat_type,
                   consensus_mean, prop_line, gap, gap_pct,
                   direction, confidence_score
            FROM fantasy_consensus
            WHERE season = ? AND week = ?
              AND prop_line IS NOT NULL AND gap IS NOT NULL
        """, (season, week)).fetchall()
        for name, stat, mean, line, gap, gap_pct, direction, conf in rows:
            edges[name][stat] = {
                "mean": mean, "line": line, "gap": gap,
                "gap_pct": gap_pct, "direction": direction,
                "confidence": conf,
            }
        con.close()
    except Exception as e:
        logger.warning("[brief_builder] Failed to load projections/edges: %s", e)

    return dict(projections), dict(edges)


def _load_matchup_features(season: int) -> dict:
    """Load team advanced stats for matchup analysis.

    Returns {team_abbr: {pressure_rate_gen, sack_rate_gen, rz_td_rate,
             rz_td_rate_allowed, deep_comp_allowed, completion_pct_allowed,
             int_forced_rate, explosive_rate_allowed, run_epa_allowed, ...}}.
    """
    if not os.path.exists(_NFL_DB):
        return {}

    try:
        con = sqlite3.connect(_NFL_DB)
        # Use requested season; fall back to most recent available
        avail = con.execute(
            "SELECT MAX(season) FROM nfl_advanced_stats WHERE season <= ?",
            (season,),
        ).fetchone()
        use_season = avail[0] if avail and avail[0] else season
        if use_season != season:
            logger.info("[brief_builder] matchup features: no %d data, using %d",
                        season, use_season)

        rows = con.execute("""
            SELECT team,
                   AVG(pressure_rate_generated) as prg,
                   AVG(sack_rate_generated) as srg,
                   AVG(pressure_rate_allowed) as pra,
                   AVG(sack_rate_allowed) as sra,
                   AVG(redzone_td_rate) as rz_td,
                   AVG(redzone_td_rate_allowed) as rz_td_a,
                   AVG(deep_completion_allowed) as deep_a,
                   AVG(completion_pct_allowed) as comp_a,
                   AVG(int_forced_rate) as int_f,
                   AVG(explosive_rate_allowed) as exp_a,
                   AVG(run_epa_allowed) as run_epa_a,
                   AVG(fg_attempts) as fga,
                   AVG(fg_make_rate) as fgm_rate
            FROM nfl_advanced_stats
            WHERE season = ?
            GROUP BY team
        """, (use_season,)).fetchall()
        con.close()

        features = {}
        for row in rows:
            team = row[0]
            features[team] = {
                "pressure_rate_gen": row[1] or 0,
                "sack_rate_gen": row[2] or 0,
                "pressure_rate_allowed": row[3] or 0,
                "sack_rate_allowed": row[4] or 0,
                "rz_td_rate": row[5] or 0,
                "rz_td_rate_allowed": row[6] or 0,
                "deep_comp_allowed": row[7] or 0,
                "completion_pct_allowed": row[8] or 0,
                "int_forced_rate": row[9] or 0,
                "explosive_rate_allowed": row[10] or 0,
                "run_epa_allowed": row[11] or 0,
                "fg_attempts": row[12] or 0,
                "fg_make_rate": row[13] or 0,
            }
        return features
    except Exception as e:
        logger.warning("[brief_builder] Failed to load matchup features: %s", e)
        return {}


def _load_pressure_profiles(season: int) -> dict:
    """Load defensive pressure rates from PBP data.

    Returns {team_abbr: {pressure_rate, sack_rate}} for defensive teams.
    Falls back to computing directly from PBP if the full pressure chain
    function isn't available or the PBP lacks was_pressure.
    """
    try:
        import nfl_data_py as nfl
        import pandas as pd
        try:
            pbp = nfl.import_pbp_data([season])
        except Exception:
            # Current season PBP not yet available — fall back to prior season
            fallback = season - 1
            logger.info("[brief_builder] PBP %d unavailable, using %d", season, fallback)
            pbp = nfl.import_pbp_data([fallback])

        dropbacks = pbp[
            (pbp["qb_dropback"] == 1) &
            (pbp["defteam"].notna())
        ].copy()

        if dropbacks.empty:
            return {}

        team_pressure = {}

        # Compute sack rate (always available)
        for team in dropbacks["defteam"].unique():
            team_db = dropbacks[dropbacks["defteam"] == team]
            n = len(team_db)
            sack_rate = team_db["sack"].sum() / n if n > 0 else 0

            # was_pressure may not be in all PBP seasons
            pressure_rate = 0
            if "was_pressure" in dropbacks.columns:
                pressured = team_db["was_pressure"].fillna(False).astype(bool)
                pressure_rate = pressured.sum() / n if n > 0 else 0

            # qb_hit is always available and is a proxy for pressure
            hit_rate = team_db["qb_hit"].sum() / n if n > 0 else 0

            team_pressure[team] = {
                "pressure_rate": round(max(pressure_rate, hit_rate), 3),
                "sack_rate": round(sack_rate, 3),
            }

        return team_pressure
    except Exception as e:
        logger.warning("[brief_builder] Failed to load pressure profiles: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────
# PBP-derived matchup features (correct denominators)
# ─────────────────────────────────────────────────────────────
def _load_pbp_matchup_features(season: int) -> dict:
    """Compute matchup-relevant features directly from PBP data.

    Returns {team_abbr: {
        rz_td_rate_off, rz_td_rate_def,     # per RZ TRIP (correct denom)
        yards_per_drive_allowed,              # between-the-20s bend measure
        deep_comp_rate_off, deep_comp_rate_def,  # 20+ air yards
        ypc_allowed,                          # yards per carry allowed
        blitz_rate,                           # % of dropbacks with 5+ rushers
        avg_box_count,                        # avg defenders in box on runs
        light_box_rate,                       # % runs with <=6 in box
        separation_off,                       # NGS avg_separation (team avg)
    }}
    """
    try:
        import nfl_data_py as nfl
        import pandas as pd

        try:
            pbp = nfl.import_pbp_data([season])
        except Exception:
            fallback = season - 1
            logger.info("[brief_builder] PBP %d unavailable for matchup features, "
                        "using %d", season, fallback)
            pbp = nfl.import_pbp_data([fallback])

        features: dict[str, dict] = {}
        all_teams = set(pbp["posteam"].dropna().unique()) | set(pbp["defteam"].dropna().unique())
        all_teams = {t for t in all_teams if isinstance(t, str) and len(t) >= 2}

        # ── RZ TD rate per RZ TRIP (correct denominator) ──
        # A "trip" = a drive that enters the red zone (yardline_100 <= 20)
        rz_plays = pbp[pbp["yardline_100"] <= 20].copy()

        # Offensive RZ trips: group by game + posteam + drive
        off_rz = rz_plays.groupby(["game_id", "posteam", "fixed_drive"]).first()
        off_rz = off_rz.reset_index()
        off_trips = off_rz.groupby("posteam").size()
        off_tds = off_rz[off_rz["fixed_drive_result"] == "Touchdown"].groupby("posteam").size()

        # Defensive RZ trips: group by game + defteam + drive
        def_rz = rz_plays.groupby(["game_id", "defteam", "fixed_drive"]).first()
        def_rz = def_rz.reset_index()
        def_trips = def_rz.groupby("defteam").size()
        def_tds = def_rz[def_rz["fixed_drive_result"] == "Touchdown"].groupby("defteam").size()

        # ── Yards per drive allowed (between-the-20s) ──
        drive_data = pbp.groupby(["game_id", "defteam", "fixed_drive"]).agg(
            yards=("yards_gained", "sum"),
        ).reset_index()
        ypd_allowed = drive_data.groupby("defteam")["yards"].mean()

        # ── Deep ball (20+ air yards, excluding sacks) ──
        deep = pbp[(pbp["air_yards"] >= 20) & (pbp["pass_attempt"] == 1) & (pbp["sack"] == 0)]
        deep_off = deep.groupby("posteam").agg(
            deep_att=("complete_pass", "count"),
            deep_comp=("complete_pass", "sum"),
        )
        deep_def = deep.groupby("defteam").agg(
            deep_att=("complete_pass", "count"),
            deep_comp=("complete_pass", "sum"),
        )

        # ── YPC allowed ──
        runs = pbp[(pbp["rush_attempt"] == 1) & (pbp["defteam"].notna())]
        ypc_by_def = runs.groupby("defteam")["yards_gained"].mean()

        # ── Box count and blitz rate (columns may be absent in early-season PBP) ──
        has_box = "defenders_in_box" in pbp.columns
        has_rushers = "number_of_pass_rushers" in pbp.columns

        box_by_def = pd.DataFrame()
        light_by_def = pd.Series(dtype=float)
        blitz_total = pd.Series(dtype=float)
        blitz_count = pd.Series(dtype=float)

        if has_box:
            run_plays = pbp[(pbp["rush_attempt"] == 1) & (pbp["defenders_in_box"].notna())]
            if not run_plays.empty:
                box_by_def = run_plays.groupby("defteam").agg(
                    avg_box=("defenders_in_box", "mean"),
                    n_runs=("defenders_in_box", "count"),
                )
                light_box = run_plays[run_plays["defenders_in_box"] <= 6]
                light_by_def = light_box.groupby("defteam").size()

        if has_rushers:
            dropbacks = pbp[(pbp["qb_dropback"] == 1) & (pbp["number_of_pass_rushers"].notna())]
            if not dropbacks.empty:
                blitz_plays = dropbacks[dropbacks["number_of_pass_rushers"] > 4]
                blitz_total = dropbacks.groupby("defteam").size()
                blitz_count = blitz_plays.groupby("defteam").size()

        # ── NGS separation ──
        separation_by_team = {}
        try:
            ngs_rec = nfl.import_ngs_data("receiving", [season if season <= 2025 else 2025])
            if "avg_separation" in ngs_rec.columns:
                separation_by_team = ngs_rec.groupby("team_abbr")["avg_separation"].mean().to_dict()
        except Exception:
            pass

        # ── Assemble per-team ──
        for team in all_teams:
            t_off_trips = off_trips.get(team, 0)
            t_off_tds = off_tds.get(team, 0)
            t_def_trips = def_trips.get(team, 0)
            t_def_tds = def_tds.get(team, 0)

            d_off = deep_off.loc[team] if team in deep_off.index else pd.Series({"deep_att": 0, "deep_comp": 0})
            d_def = deep_def.loc[team] if team in deep_def.index else pd.Series({"deep_att": 0, "deep_comp": 0})

            b_total = blitz_total.get(team, 0) if len(blitz_total) else 0
            b_count = blitz_count.get(team, 0) if len(blitz_count) else 0

            box_info = (box_by_def.loc[team] if (not box_by_def.empty and team in box_by_def.index)
                        else pd.Series({"avg_box": 0, "n_runs": 0}))
            n_light = light_by_def.get(team, 0) if len(light_by_def) else 0
            n_box_runs = box_info["n_runs"] if box_info["n_runs"] > 0 else 1

            features[team] = {
                "rz_td_rate_off": round(t_off_tds / t_off_trips, 3) if t_off_trips > 0 else 0,
                "rz_td_rate_def": round(t_def_tds / t_def_trips, 3) if t_def_trips > 0 else 0,
                "rz_trips_off": int(t_off_trips),
                "rz_trips_def": int(t_def_trips),
                "yards_per_drive_allowed": round(ypd_allowed.get(team, 0), 1),
                "deep_comp_rate_off": round(d_off["deep_comp"] / d_off["deep_att"], 3) if d_off["deep_att"] > 10 else None,
                "deep_comp_rate_def": round(d_def["deep_comp"] / d_def["deep_att"], 3) if d_def["deep_att"] > 10 else None,
                "deep_att_off": int(d_off["deep_att"]),
                "deep_att_def": int(d_def["deep_att"]),
                "ypc_allowed": round(ypc_by_def.get(team, 0), 2),
                "blitz_rate": round(b_count / b_total, 3) if b_total > 0 else 0,
                "avg_box_count": round(box_info["avg_box"], 1),
                "light_box_rate": round(n_light / n_box_runs, 3),
                "separation_off": round(separation_by_team.get(team, 0), 2),
            }

        logger.info("[brief_builder] PBP matchup features loaded for %d teams", len(features))
        return features
    except Exception as e:
        logger.warning("[brief_builder] Failed to load PBP matchup features: %s", e)
        return {}


def _compute_percentile_ranks(features: dict) -> dict:
    """Add percentile ranks (0–100) for each metric across all teams.

    Mutates features in place, adding pct_<metric> keys.
    Returns the same dict.
    """
    if not features:
        return features

    # Collect all metric keys (exclude non-numeric and raw counts)
    skip = {"rz_trips_off", "rz_trips_def", "deep_att_off", "deep_att_def"}
    sample = next(iter(features.values()))

    def _is_numeric(v):
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return True
        # Handle numpy numeric types
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False

    metrics = [k for k in sample if k not in skip and _is_numeric(sample[k])]

    for metric in metrics:
        # Gather values, skipping None
        vals = [(team, features[team].get(metric)) for team in features]
        vals = [(t, v) for t, v in vals if v is not None]
        if len(vals) < 4:
            continue
        # Skip zero-variance metrics (e.g., all-zero when column was missing)
        unique_vals = set(float(v) for _, v in vals)
        if len(unique_vals) <= 1:
            continue
        # Sort and assign percentile rank
        sorted_vals = sorted(vals, key=lambda x: x[1])
        n = len(sorted_vals)
        for rank_idx, (team, _) in enumerate(sorted_vals):
            pct = round(rank_idx / (n - 1) * 100) if n > 1 else 50
            features[team][f"pct_{metric}"] = pct

    return features


# ─────────────────────────────────────────────────────────────
# Injury parsing
# ─────────────────────────────────────────────────────────────
def _parse_injury_items(items: list[dict]) -> list[dict]:
    """Parse ESPN injury items into structured player records.

    Returns list of {player_name, team_full, team_abbr, position,
                     status, severity, body}.
    """
    parsed = []
    # Pattern: [NFL] PlayerName (, Position) — Status
    pat = re.compile(
        r"\[NFL\]\s+(.+?)\s+\(,\s*(\w+)\)\s*[—–-]\s*(\w[\w\s]*\w|\w)"
    )

    for item in items:
        if item.get("source", "") != "espn_nfl_injuries":
            continue

        m = pat.match(item.get("headline", ""))
        if not m:
            continue

        player_name = m.group(1).strip()
        position = m.group(2).strip()
        status = m.group(3).strip()

        team_full = (item.get("teams") or [""])[0]
        team_abbr = _TEAM_ABBREV.get(team_full, "")

        parsed.append({
            "player_name": player_name,
            "team_full": team_full,
            "team_abbr": team_abbr,
            "position": position,
            "status": status,
            "severity": _INJURY_STATUSES.get(status, 0),
            "body": (item.get("body") or "")[:200],
            "pos_group": _POS_GROUPS.get(position, "other"),
        })

    return parsed


def _normalize_name(name: str) -> str:
    """Normalize player name for matching across data sources.

    PBP uses 'J.Smith-Njigba', props use 'Jaxon Smith-Njigba',
    ESPN uses 'Jaxon Smith-Njigba'.
    """
    # Convert PBP abbreviated first name to just the initial
    parts = name.split(".", 1)
    if len(parts) == 2 and len(parts[0]) <= 2:
        return parts[0].upper() + "." + parts[1].strip()
    # For full names, abbreviate
    parts = name.split(" ", 1)
    if len(parts) == 2:
        return parts[0][0].upper() + "." + parts[1].strip()
    return name


def _is_backup_with_starter_healthy(
    injured: dict, team: str, usage: dict, all_injuries: list[dict],
    current_roster: dict = None,
) -> bool:
    """Return True if this injured player is a backup whose starter is healthy.

    A backup who is OUT generates ZERO implications when the starter is healthy.
    But a backup who is OUT when the STARTER is also OUT is relevant (next man up).

    PRIMARY signal: current-season depth chart rank (pos_rank from nfl_data_py).
      - depth_rank > 1 AND the rank=1 player is healthy → backup, suppress.
      - depth_rank == 1 → starter, NEVER suppress regardless of PBP history.

    FALLBACK (no depth chart data): PBP usage comparison (carry/target share).
    """
    pos = injured["position"]
    pos_group = injured.get("pos_group", _POS_GROUPS.get(pos, "other"))
    inj_name_norm = _normalize_name(injured["player_name"])

    # ── PRIMARY: depth chart rank from current roster ──
    if current_roster:
        self_roster = current_roster.get(inj_name_norm)
        if self_roster and self_roster.get("depth_rank") is not None:
            self_rank = self_roster["depth_rank"]
            self_team = self_roster["team"]

            # Verify this player is on the expected team
            if self_team != team:
                # Player is on a different team than the injury report suggests.
                # Trust the roster — but this mismatch means the injury data
                # may be stale or the player was recently traded.
                logger.debug(
                    "[brief_builder] Team mismatch: %s injury says %s, roster says %s",
                    injured["player_name"], team, self_team,
                )

            if self_rank == 1:
                return False  # Starter — NEVER suppress

            # Rank > 1: check if the starter (rank=1) at this position is healthy
            for norm, info in current_roster.items():
                if info["team"] != self_team:
                    continue
                if norm == inj_name_norm:
                    continue
                # Same position group?
                r_pos = info.get("position", "")
                r_pg = _POS_GROUPS.get(r_pos, "other")
                if r_pg != pos_group:
                    continue
                if info.get("depth_rank") is not None and info["depth_rank"] < self_rank:
                    # Higher-ranked teammate exists — are they healthy?
                    higher_is_out = any(
                        _normalize_name(i["player_name"]) == norm and i["severity"] >= 2
                        for i in all_injuries
                    )
                    if not higher_is_out:
                        return True  # Starter healthy, this backup's absence is irrelevant

            # No healthy higher-ranked teammate found — this player is relevant
            # (could be next man up with starter also out)
            return False

    # ── FALLBACK: PBP usage comparison ──
    # Used when depth chart data is unavailable for this player
    teammates_at_pos = []
    for key, u in usage.items():
        if u["team"] != team:
            continue
        u_pos = u.get("position", _infer_position(u))
        u_pg = _POS_GROUPS.get(u_pos, "other")
        if u_pg != pos_group:
            continue
        u_norm = _normalize_name(u["player_name"])
        is_out = any(
            _normalize_name(i["player_name"]) == u_norm and i["severity"] >= 2
            for i in all_injuries
        )
        # Also check depth_rank if available on usage entry (from roster remap)
        u_rank = u.get("depth_rank")
        teammates_at_pos.append({**u, "is_out": is_out, "_norm": u_norm, "depth_rank": u_rank})

    if not teammates_at_pos:
        return False  # No data — assume relevant

    # If any teammate has depth_rank data, prefer that
    has_rank_data = any(t.get("depth_rank") is not None for t in teammates_at_pos)

    if has_rank_data:
        self_entry = next(
            (t for t in teammates_at_pos if t["_norm"] == inj_name_norm), None
        )
        if not self_entry or self_entry.get("depth_rank") is None:
            # Player not found in usage with rank — likely very deep backup
            return True
        self_rank = self_entry["depth_rank"]
        if self_rank == 1:
            return False
        for t in teammates_at_pos:
            if t["_norm"] == inj_name_norm:
                continue
            t_rank = t.get("depth_rank")
            if t_rank is not None and t_rank < self_rank and not t["is_out"]:
                return True
        return False

    # Pure PBP fallback (no depth chart for anyone)
    if pos_group == "quarterback":
        self_entry = next(
            (t for t in teammates_at_pos if t["_norm"] == inj_name_norm), None
        )
        if not self_entry:
            return True
        for t in teammates_at_pos:
            if t["_norm"] == inj_name_norm:
                continue
            if t.get("dropbacks", 0) > self_entry.get("dropbacks", 0) and not t["is_out"]:
                return True
        return False

    if pos_group == "rusher":
        self_entry = next(
            (t for t in teammates_at_pos if t["_norm"] == inj_name_norm), None
        )
        if not self_entry:
            return True
        self_share = self_entry.get("carry_share", 0)
        if self_share >= 0.15:
            return False
        for t in teammates_at_pos:
            if t["_norm"] == inj_name_norm:
                continue
            if t.get("carry_share", 0) > self_share and not t["is_out"]:
                return True
        return False

    if pos_group == "pass_catcher":
        self_entry = next(
            (t for t in teammates_at_pos if t["_norm"] == inj_name_norm), None
        )
        if not self_entry:
            return True
        self_share = self_entry.get("target_share", 0)
        if self_share >= _MIN_SHARE_FOR_IMPLICATIONS:
            return False
        for t in teammates_at_pos:
            if t["_norm"] == inj_name_norm:
                continue
            if t.get("target_share", 0) > self_share and not t["is_out"]:
                return True
        return False

    # OL, other — always relevant
    return False


# ─────────────────────────────────────────────────────────────
# Implication engine
# ─────────────────────────────────────────────────────────────
# Positional redistribution heuristics (when with/without splits unavailable)
_REDISTRIBUTION_HEURISTICS = {
    # When WR1 is out: WR2 gets +6pp target share, WR3 +3pp, TE +2pp
    "WR_out": {
        "WR": {"target_share_boost_pp": 6, "note": "target share up (volume redistribution)"},
        "TE": {"target_share_boost_pp": 2, "note": "red-zone targets up"},
        "RB": {"target_share_boost_pp": 1, "note": "check-down targets up slightly"},
    },
    # When TE1 is out: WR targets up, other TE up
    "TE_out": {
        "WR": {"target_share_boost_pp": 3, "note": "targets absorb"},
        "TE": {"target_share_boost_pp": 4, "note": "TE2 absorbs primary role"},
        "RB": {"target_share_boost_pp": 1, "note": "check-down targets up slightly"},
    },
    # When RB1 is out: RB2 carries up dramatically
    "RB_out": {
        "RB": {"carry_share_boost_pp": 20, "note": "lead-back role"},
    },
    # FB out: minor — RB may see more short-yardage carries
    "FB_out": {
        "RB": {"carry_share_boost_pp": 2, "note": "short-yardage carries up slightly"},
    },
}

# Harm effects from key player absences
_HARM_EFFECTS = {
    "WR": {
        "QB": "Lost top option → pass yds DOWN, efficiency drops",
        "WR_same_team": "Faces tighter coverage (defense concentrates on remaining options)",
    },
    "TE": {
        "QB": "Lost reliable target → passing efficiency down, may face more pressure",
    },
    "RB": {
        "QB": "If backup worse in pass-pro → pressure UP → INT rate UP",
    },
    "FB": {
        "RB": "Short-yardage blocking degrades slightly",
    },
    "OT": {
        "QB": "Pressure rate UP → sack rate UP → INT rate UP, pass yds DOWN",
        "RB": "Run blocking degrades → rush yds DOWN on affected side",
    },
    "OG": {
        "QB": "Interior pressure UP → hurries UP, pocket collapses faster",
        "RB": "Inside run blocking degrades → rush yds DOWN",
    },
    # "G" is an alias used by some injury reports for OG
    "G": {
        "QB": "Interior pressure UP → hurries UP, pocket collapses faster",
        "RB": "Inside run blocking degrades → rush yds DOWN",
    },
    "C": {
        "QB": "Protection calls disrupted → blitz pickup worse → pressure UP",
        "RB": "Interior run game degrades",
    },
    # "T" is an alias used by some injury reports for OT
    "T": {
        "QB": "Pressure rate UP → sack rate UP → INT rate UP, pass yds DOWN",
        "RB": "Run blocking degrades → rush yds DOWN on affected side",
    },
    # "OL" generic
    "OL": {
        "QB": "Pressure rate UP → sack rate UP, pass yds DOWN",
        "RB": "Run blocking degrades",
    },
}


def _compute_with_without_splits(
    injured_name: str, team: str, usage: dict
) -> Optional[dict]:
    """Compute actual with/without target/carry redistribution from PBP.

    This requires game-level data to see which games the injured player
    was active vs absent. Returns None if insufficient sample (N<5 games without).
    """
    # For now, return None — with/without splits require historical absence data
    # which needs multi-season PBP cross-referenced with injury reports.
    # TODO: implement when game-level absence data is available
    return None


def _compute_implications(
    injuries_by_team: dict,
    usage: dict,
    active_props: dict,
    pressure_profiles: dict,
    games: list[dict],
    projections: dict = None,
    edges: dict = None,
    current_roster: dict = None,
) -> dict:
    """Compute dependency effects for each injury.

    Returns {team_abbr: {
        "injured": [...],
        "beneficiaries": [...],
        "harmed": [...],
        "opponent_effects": [...],
    }}
    """
    # Build game lookup: team → opponent
    team_opponent = {}
    for game in games:
        team_opponent[game["home_team"]] = game["away_team"]
        team_opponent[game["away_team"]] = game["home_team"]

    # Build team rosters from usage data
    team_roster = defaultdict(list)
    for key, u in usage.items():
        team_roster[u["team"]].append(u)

    implications = {}

    for team_abbr, injured_list in injuries_by_team.items():
        out_players = [p for p in injured_list if p["severity"] >= 2]  # Out or Doubtful
        questionable = [p for p in injured_list if p["severity"] == 1]

        if not out_players and not questionable:
            continue

        impl = {
            "injured": injured_list,
            "beneficiaries": [],
            "harmed": [],
            "opponent_effects": [],
        }

        teammates = team_roster.get(team_abbr, [])
        opponent = team_opponent.get(team_abbr, "")

        for inj in out_players:
            pos = inj["position"]
            pos_group = inj["pos_group"]
            inj_name = inj["player_name"]
            inj_name_norm = _normalize_name(inj_name)

            # ── PARTICIPATION FILTER ──
            # Skip if this injured player is a backup whose starter is healthy.
            # A QB3 being out changes nothing; a QB1 being out changes everything.
            if _is_backup_with_starter_healthy(inj, team_abbr, usage, injured_list,
                                                current_roster=current_roster):
                logger.debug(
                    "[brief_builder] Skipping backup %s (%s) — starter healthy",
                    inj_name, pos,
                )
                continue

            # Find the injured player's usage for context
            inj_usage = None
            for key, u in usage.items():
                if u["team"] == team_abbr and _normalize_name(u["player_name"]) == inj_name_norm:
                    inj_usage = u
                    break

            # Skip low-usage OUT players — their absence doesn't move the needle
            inj_share = 0
            if inj_usage:
                inj_share = max(
                    inj_usage.get("target_share", 0),
                    inj_usage.get("carry_share", 0),
                )
            # QBs and OL always matter regardless of share metrics
            if pos_group not in ("quarterback", "oline") and inj_share < _MIN_SHARE_FOR_IMPLICATIONS:
                continue

            # ── BENEFICIARIES ──
            # Try with/without splits first
            splits = _compute_with_without_splits(inj_name, team_abbr, usage)

            if splits:
                for ben in splits.get("beneficiaries", []):
                    impl["beneficiaries"].append(ben)
            else:
                # Use positional redistribution heuristics
                heuristic_key = f"{pos}_out"
                heuristics = _REDISTRIBUTION_HEURISTICS.get(heuristic_key, {})

                # Collect candidate beneficiaries, then take top N by usage
                candidates = []
                for teammate in teammates:
                    tm_name = teammate["player_name"]
                    tm_pos = teammate.get("position", _infer_position(teammate))

                    if tm_pos not in heuristics:
                        continue
                    # Skip if teammate is also injured
                    if any(_normalize_name(i["player_name"]) == _normalize_name(tm_name)
                           and i["severity"] >= 2 for i in injured_list):
                        continue

                    boost_info = heuristics[tm_pos]
                    tgt_boost = boost_info.get("target_share_boost_pp", 0)
                    carry_boost = boost_info.get("carry_share_boost_pp", 0)

                    # Scale boost by injured player's share (bigger share = bigger redistribution)
                    scale = 1.0
                    if inj_usage:
                        if tgt_boost and inj_usage.get("target_share", 0) > 0:
                            scale = inj_usage["target_share"] / 0.25
                            scale = min(max(scale, 0.5), 2.0)
                        elif carry_boost and inj_usage.get("carry_share", 0) > 0:
                            scale = inj_usage["carry_share"] / 0.60
                            scale = min(max(scale, 0.5), 2.0)

                    # Only list meaningful beneficiaries (skip <5% share)
                    tm_share = max(
                        teammate.get("target_share", 0),
                        teammate.get("carry_share", 0),
                    )
                    if tm_share < 0.05:
                        continue

                    # Skip if boost rounds to 0
                    eff_boost = round((tgt_boost or carry_boost) * scale, 1)
                    if eff_boost < 1:
                        continue

                    eff_note = boost_info["note"]

                    # Check if beneficiary faces HARDER coverage (the tension)
                    tension = None
                    if pos in ("WR", "TE") and tm_pos in ("WR", "TE"):
                        tension = (
                            f"BUT faces tighter coverage — defense can concentrate "
                            f"with {inj_name.split()[-1]} out. "
                            f"Net effect genuinely unclear."
                        )

                    ben_props = _get_player_props(tm_name, active_props)

                    # Enrich with projection/edge data
                    player_proj = _get_player_projections(tm_name, projections)
                    player_edges = _get_player_edges(tm_name, edges)

                    candidates.append({
                        "player_name": tm_name,
                        "position": tm_pos,
                        "because_of": inj_name,
                        "boost_type": "target_share" if tgt_boost else "carry_share",
                        "boost_pp": round((tgt_boost or carry_boost) * scale, 1),
                        "note": eff_note,
                        "tension": tension,
                        "source": "heuristic",
                        "current_share": tm_share,
                        "props": ben_props,
                        "projections": player_proj,
                        "edges": player_edges,
                    })

                # Rank by usage (highest share first), cap at top N
                candidates.sort(key=lambda x: -x["current_share"])
                impl["beneficiaries"].extend(
                    candidates[:_MAX_BENEFICIARIES_PER_INJURY]
                )

            # ── HARMED PLAYERS ──
            harm_map = _HARM_EFFECTS.get(pos, {})

            for harm_target, harm_desc in harm_map.items():
                if harm_target == "QB":
                    # Find team's EXPECTED starting QB — highest dropbacks
                    # among QBs who are NOT themselves out/doubtful
                    qb_candidates = [
                        tm for tm in teammates
                        if (tm.get("position") == "QB" or tm.get("role") == "quarterback")
                        and not any(
                            _normalize_name(i["player_name"]) == _normalize_name(tm["player_name"])
                            and i["severity"] >= 2
                            for i in injured_list
                        )
                    ]
                    if qb_candidates:
                        starter = max(qb_candidates, key=lambda x: x.get("dropbacks", 0))
                        if starter.get("dropbacks", 0) > 20:
                            qb_props = _get_player_props(starter["player_name"], active_props)

                            # Wire pressure chain for OL injuries
                            pressure_note = ""
                            if pos_group == "oline" and opponent in pressure_profiles:
                                opp_pr = pressure_profiles[opponent].get("pressure_rate", 0)
                                if opp_pr > 0.30:
                                    pressure_note = (
                                        f" Opponent {opponent} already generates "
                                        f"{opp_pr:.0%} pressure — OL downgrade compounds this."
                                    )

                            impl["harmed"].append({
                                "player_name": starter["player_name"],
                                "position": "QB",
                                "because_of": inj_name,
                                "effect": harm_desc + pressure_note,
                                "props": qb_props,
                                "projections": _get_player_projections(starter["player_name"], projections),
                                "edges": _get_player_edges(starter["player_name"], edges),
                            })

                elif harm_target == "RB":
                    # Find lead RB
                    rb_candidates = [
                        tm for tm in teammates
                        if tm.get("position") in ("RB", "FB") and tm.get("carry_share", 0) > 0.15
                    ]
                    for rb in rb_candidates[:1]:  # Only lead back
                        rb_props = _get_player_props(rb["player_name"], active_props)
                        impl["harmed"].append({
                            "player_name": rb["player_name"],
                            "position": "RB",
                            "because_of": inj_name,
                            "effect": harm_desc,
                            "props": rb_props,
                            "projections": _get_player_projections(rb["player_name"], projections),
                            "edges": _get_player_edges(rb["player_name"], edges),
                        })

                elif harm_target == "WR_same_team":
                    # Top 2 remaining WRs by target share
                    wr_candidates = [
                        tm for tm in teammates
                        if tm.get("position") in ("WR", "TE")
                        and tm.get("target_share", 0) > 0.10
                        and _normalize_name(tm["player_name"]) != inj_name_norm
                    ]
                    wr_candidates.sort(key=lambda x: -x.get("target_share", 0))
                    for wr in wr_candidates[:2]:
                        wr_props = _get_player_props(wr["player_name"], active_props)
                        impl["harmed"].append({
                            "player_name": wr["player_name"],
                            "position": wr.get("position", "WR"),
                            "because_of": inj_name,
                            "effect": harm_desc,
                            "props": wr_props,
                            "projections": _get_player_projections(wr["player_name"], projections),
                            "edges": _get_player_edges(wr["player_name"], edges),
                        })

            # ── OPPONENT EFFECTS ──
            if pos_group == "pass_catcher" and inj_usage and inj_usage.get("target_share", 0) > 0.15:
                impl["opponent_effects"].append({
                    "note": (
                        f"With {inj_name} ({inj['position']}) out, {team_abbr}'s offense degrades — "
                        f"opposing {opponent} defense can concentrate coverage on remaining options. "
                        f"Consider {opponent} defensive props and team total."
                    ),
                })
            elif pos_group == "oline":
                impl["opponent_effects"].append({
                    "note": (
                        f"OL loss ({inj_name}, {pos}) for {team_abbr} — "
                        f"opposing {opponent} pass rush benefits. "
                        f"Watch {opponent} defensive player sack/pressure props."
                    ),
                })

        # Also process questionable players — lighter treatment
        for q in questionable:
            # Just note them; implications are conditional
            pass

        implications[team_abbr] = impl

    return implications


def _infer_position(usage_entry: dict) -> str:
    """Infer position from usage data (prefers roster-loaded position)."""
    # Use roster-loaded position if available
    pos = usage_entry.get("position", "")
    if pos and pos != "WR":
        return pos
    # Fall back to role-based inference
    role = usage_entry.get("role", "")
    if role == "quarterback":
        return "QB"
    if role == "rusher":
        return "RB"
    return pos or "WR"


def _get_player_props(player_name: str, active_props: dict) -> list[dict]:
    """Look up active props for a player, trying name variations."""
    # Direct match
    if player_name in active_props:
        return active_props[player_name]

    # Try normalized match
    norm = _normalize_name(player_name)
    for prop_name, props in active_props.items():
        if _normalize_name(prop_name) == norm:
            return props

    # Try last-name match (for PBP abbreviated names like "J.Smith-Njigba")
    if "." in player_name:
        last = player_name.split(".", 1)[1].strip()
        matches = [
            (pn, p) for pn, p in active_props.items()
            if pn.endswith(last) or last in pn
        ]
        if len(matches) == 1:
            return matches[0][1]

    return []


def _get_player_projections(player_name: str, projections: dict = None) -> dict:
    """Look up fantasy projections for a player, trying name variations."""
    if not projections:
        return {}
    # Direct match
    if player_name in projections:
        return projections[player_name]
    # Normalized match
    norm = _normalize_name(player_name)
    for pn, proj in projections.items():
        if _normalize_name(pn) == norm:
            return proj
    # Last-name match for PBP names
    if "." in player_name:
        last = player_name.split(".", 1)[1].strip()
        matches = [(pn, p) for pn, p in projections.items() if pn.endswith(last) or last in pn]
        if len(matches) == 1:
            return matches[0][1]
    return {}


def _get_player_edges(player_name: str, edges: dict = None) -> dict:
    """Look up consensus model edges for a player, trying name variations."""
    if not edges:
        return {}
    if player_name in edges:
        return edges[player_name]
    norm = _normalize_name(player_name)
    for pn, e in edges.items():
        if _normalize_name(pn) == norm:
            return e
    if "." in player_name:
        last = player_name.split(".", 1)[1].strip()
        matches = [(pn, e) for pn, e in edges.items() if pn.endswith(last) or last in pn]
        if len(matches) == 1:
            return matches[0][1]
    return {}


# ─────────────────────────────────────────────────────────────
# Usage-weighted player ranking
# ─────────────────────────────────────────────────────────────
def _rank_players_for_team(
    team_abbr: str,
    injuries: list[dict],
    usage: dict,
    active_props: dict,
) -> list[dict]:
    """Rank team's players by expected involvement for the brief.

    Ranking: usage (primary) × has_props (boost) × injury_relevance (boost).
    """
    # Get all players on this team from usage data
    team_players = {
        k: v for k, v in usage.items() if v["team"] == team_abbr
    }

    # Score each player
    scored = []
    for key, u in team_players.items():
        name = u["player_name"]

        # Usage score: blend of target share and carry share
        usage_score = max(
            u.get("target_share", 0),
            u.get("carry_share", 0),
        )
        # QB bonus — QBs don't have share metrics but are always relevant
        if u.get("role") == "quarterback" and u.get("dropbacks", 0) > 20:
            usage_score = max(usage_score, 0.80)

        # Prop boost: players with active props rank higher
        has_props = bool(_get_player_props(name, active_props))
        if has_props:
            usage_score += 0.15

        # Injury relevance boost
        is_injured = any(
            _normalize_name(inj["player_name"]) == _normalize_name(name)
            for inj in injuries
        )
        if is_injured:
            usage_score += 0.20

        scored.append({
            **u,
            "usage_score": usage_score,
            "has_props": has_props,
            "is_injured": is_injured,
        })

    # Sort by usage_score descending
    scored.sort(key=lambda x: -x["usage_score"])
    return scored[:_MAX_PLAYERS_PER_TEAM]


# ─────────────────────────────────────────────────────────────
# Active-player benefit detection
# ─────────────────────────────────────────────────────────────
def _find_active_beneficiaries(
    team_abbr: str,
    injuries: list[dict],
    usage: dict,
    active_props: dict,
    projections: dict = None,
    edges: dict = None,
    current_roster: dict = None,
) -> list[dict]:
    """Find ACTIVE players who benefit from teammate injuries.

    These are the highest-value lines — players whose competition is out
    but who aren't themselves injury items. Only reports meaningful OUT
    players (>= 8% share or QB/OL) and top beneficiaries.
    """
    out_players = [p for p in injuries if p["severity"] >= 2]
    if not out_players:
        return []

    # Filter to meaningful OUT players — skip backups whose starters are healthy
    meaningful_out = []
    for out_p in out_players:
        # Participation filter: backups with starter healthy generate no implications
        if _is_backup_with_starter_healthy(out_p, team_abbr, usage, injuries,
                                            current_roster=current_roster):
            continue
        out_name_norm = _normalize_name(out_p["player_name"])
        out_pos_group = out_p["pos_group"]
        # QB/OL always meaningful (already passed backup filter above)
        if out_pos_group in ("quarterback", "oline"):
            meaningful_out.append(out_p)
            continue
        # Check if this player had significant usage
        for key, u in usage.items():
            if u["team"] == team_abbr and _normalize_name(u["player_name"]) == out_name_norm:
                share = max(u.get("target_share", 0), u.get("carry_share", 0))
                if share >= _MIN_SHARE_FOR_IMPLICATIONS:
                    meaningful_out.append(out_p)
                break

    if not meaningful_out:
        return []

    # Collect beneficiaries, deduplicating by player (even if multiple teammates are out)
    seen_beneficiaries = {}  # player_name → best entry

    for out_p in meaningful_out:
        out_pos = out_p["position"]
        out_name_norm = _normalize_name(out_p["player_name"])

        candidates = []
        for key, u in usage.items():
            if u["team"] != team_abbr:
                continue
            tm_pos = u.get("position", _infer_position(u))
            tm_name_norm = _normalize_name(u["player_name"])

            if tm_name_norm == out_name_norm:
                continue
            if any(_normalize_name(i["player_name"]) == tm_name_norm
                   and i["severity"] >= 1 for i in injuries):
                continue

            # Same position group?
            out_pg = _POS_GROUPS.get(out_pos)
            tm_pg = _POS_GROUPS.get(tm_pos)
            if out_pg != tm_pg and not (
                out_pos in ("WR", "TE") and tm_pos in ("WR", "TE")
            ):
                continue

            share = max(u.get("target_share", 0), u.get("carry_share", 0))
            if share < 0.08:
                continue

            props = _get_player_props(u["player_name"], active_props)
            candidates.append({
                "player_name": u["player_name"],
                "position": tm_pos,
                "because_out": out_p["player_name"],
                "out_position": out_pos,
                "target_share": u.get("target_share", 0),
                "carry_share": u.get("carry_share", 0),
                "share": share,
                "props": props,
                "projections": _get_player_projections(u["player_name"], projections),
                "edges": _get_player_edges(u["player_name"], edges),
            })

        # Top 3 by usage
        candidates.sort(key=lambda x: -x["share"])
        for c in candidates[:_MAX_BENEFICIARIES_PER_INJURY]:
            name = c["player_name"]
            if name not in seen_beneficiaries or c["share"] > seen_beneficiaries[name]["share"]:
                seen_beneficiaries[name] = c

    return sorted(seen_beneficiaries.values(), key=lambda x: -x["share"])


# ─────────────────────────────────────────────────────────────
# Matchup edge rendering
# ─────────────────────────────────────────────────────────────
def _ordinal(n: int) -> str:
    """Return ordinal string for an integer (1st, 2nd, 3rd, 4th, ...)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][n % 10] if n % 10 < 4 else 'th'}"


def _format_game_matchup_edges(
    away: str, home: str,
    matchup_features: dict,
    pressure: dict,
    pbp_features: dict | None = None,
) -> list[str]:
    """Generate matchup edge bullets for a game.

    Uses percentile-based extreme-mismatch logic. Only surfaces edges where
    a top-quintile trait meets a bottom-quintile weakness. Target: 0-3 edges
    per game; games with no genuine mismatch get no edges.

    Requires pbp_features (from _load_pbp_matchup_features) for correct
    RZ denominators, deep-ball multi-factor checks, and RB box/blitz splits.
    """
    if not pbp_features:
        return []

    away_pbp = pbp_features.get(away, {})
    home_pbp = pbp_features.get(home, {})

    if not away_pbp and not home_pbp:
        return []

    # Legacy features (nfl_advanced_stats) — for pressure + INT only
    away_f = matchup_features.get(away, {})
    home_f = matchup_features.get(home, {})
    away_p = pressure.get(away, {})
    home_p = pressure.get(home, {})

    def _pct(team, metric):
        """Get percentile rank for a team's metric (0-100)."""
        return pbp_features.get(team, {}).get(f"pct_{metric}")

    def _val(team, metric):
        return pbp_features.get(team, {}).get(metric)

    def _is_top(pct, threshold=80):
        return pct is not None and pct >= threshold

    def _is_bot(pct, threshold=20):
        return pct is not None and pct <= threshold

    edges = []

    # ─────────────────────────────────────────────────────────
    # 1. PRESSURE MISMATCH
    # Top-quintile pressure generator vs bottom-quintile OL
    # ─────────────────────────────────────────────────────────
    for def_team, off_team in [(home, away), (away, home)]:
        pr = (pressure.get(def_team, {}).get("pressure_rate", 0)
              or matchup_features.get(def_team, {}).get("pressure_rate_gen", 0))
        sr = (pressure.get(def_team, {}).get("sack_rate", 0)
              or matchup_features.get(def_team, {}).get("sack_rate_gen", 0))
        pra = matchup_features.get(off_team, {}).get("pressure_rate_allowed", 0)

        # Use PBP blitz rate as secondary signal
        blitz = _val(def_team, "blitz_rate") or 0

        # Need BOTH: high pressure generation AND vulnerable OL
        pr_pct = _pct(def_team, "blitz_rate")  # blitz rate as proxy for aggression
        if pr > 0.30 and pra > 0.28:
            label = "elite" if pr > 0.35 else "high"
            edges.append((
                pr + pra,  # sort priority — combined mismatch severity
                f"- **Pressure mismatch:** {def_team} generates {pr:.0%} pressure "
                f"({label}), blitz {blitz:.0%} — vs {off_team} OL allows "
                f"{pra:.0%} pressure. Sack rate {sr:.0%}. "
                f"Watch {off_team} QB yds/TDs UNDER."
            ))

    # ─────────────────────────────────────────────────────────
    # 2. BEND-DON'T-BREAK (two conditions required)
    # High yards allowed (BEND) + low RZ TD rate (DON'T BREAK)
    # ─────────────────────────────────────────────────────────
    for def_team, off_team in [(home, away), (away, home)]:
        ypd_pct = _pct(def_team, "yards_per_drive_allowed")
        rz_def_pct = _pct(def_team, "rz_td_rate_def")
        ypd = _val(def_team, "yards_per_drive_allowed") or 0
        rz_def = _val(def_team, "rz_td_rate_def") or 0

        # Must BOTH be true: bends (top-40% yards allowed) AND doesn't break (bottom-35% RZ)
        if (ypd_pct is not None and ypd_pct >= 60
                and rz_def_pct is not None and rz_def_pct <= 35):
            off_rz = _val(off_team, "rz_td_rate_off") or 0
            off_rz_pct = _pct(off_team, "rz_td_rate_off")
            edges.append((
                ypd_pct - rz_def_pct,  # divergence magnitude
                f"- **Bend-don't-break:** {def_team} DEF allows {ypd:.0f} yds/drive "
                f"({_ordinal(ypd_pct)} pct) but only {rz_def:.0%} RZ TD rate "
                f"({_ordinal(rz_def_pct)} pct). Drives stall in FG range. "
                f"{off_team} OFF RZ TD {off_rz:.0%} ({_ordinal(off_rz_pct)} pct). "
                f"Kicker props UP for {off_team}."
            ))

    # ─────────────────────────────────────────────────────────
    # 3. DEEP BALL (multi-factor: QB accuracy + WR separation + defense weakness)
    # All three must align for a flag.
    # ─────────────────────────────────────────────────────────
    for off_team, def_team in [(away, home), (home, away)]:
        deep_off = _val(off_team, "deep_comp_rate_off")
        deep_def = _val(def_team, "deep_comp_rate_def")
        sep = _val(off_team, "separation_off") or 0

        deep_off_pct = _pct(off_team, "deep_comp_rate_off")
        deep_def_pct = _pct(def_team, "deep_comp_rate_def")
        sep_pct = _pct(off_team, "separation_off")

        if deep_off is None or deep_def is None:
            continue

        # Need at least 2 of 3: accurate deep QB (top-30%), separating WRs (top-30%),
        # weak deep defense (top-30% = high comp allowed)
        factors = 0
        factor_details = []
        if deep_off_pct is not None and deep_off_pct >= 70:
            factors += 1
            factor_details.append(f"QB deep comp {deep_off:.0%} ({_ordinal(deep_off_pct)} pct)")
        if sep_pct is not None and sep_pct >= 70:
            factors += 1
            factor_details.append(f"WR separation {sep:.2f} ({_ordinal(sep_pct)} pct)")
        if deep_def_pct is not None and deep_def_pct >= 70:
            factors += 1
            factor_details.append(f"{def_team} deep comp allowed {deep_def:.0%} ({_ordinal(deep_def_pct)} pct)")

        if factors >= 2:
            edges.append((
                factors + (deep_off or 0),
                f"- **Deep ball alignment:** {off_team} vs {def_team} — "
                + "; ".join(factor_details)
                + f". {off_team} WR receiving yds OVER."
            ))

    # ─────────────────────────────────────────────────────────
    # 4. RUN DEFENSE (yards-based + box/blitz interaction)
    # Weak run D (top quintile YPC allowed) PLUS defense uses
    # light boxes or low blitz = exploitable by ground game.
    # ─────────────────────────────────────────────────────────
    for off_team, def_team in [(away, home), (home, away)]:
        ypc_a = _val(def_team, "ypc_allowed") or 0
        ypc_pct = _pct(def_team, "ypc_allowed")
        light_box = _val(def_team, "light_box_rate") or 0
        light_pct = _pct(def_team, "light_box_rate")
        blitz = _val(def_team, "blitz_rate") or 0

        # Legacy EPA for context
        run_epa = matchup_features.get(def_team, {}).get("run_epa_allowed", 0)

        if ypc_pct is not None and ypc_pct >= 80:
            # Top-quintile bad run defense
            context_parts = [f"{def_team} allows {ypc_a:.1f} YPC ({_ordinal(ypc_pct)} pct)"]

            if run_epa > 0:
                context_parts.append(f"run EPA {run_epa:+.3f}")

            # Add box/blitz context if data is available and exploitable
            if light_pct is not None and light_pct >= 60 and light_box > 0:
                context_parts.append(
                    f"light box {light_box:.0%} of runs ({_ordinal(light_pct)} pct)"
                )
            blitz_pct = _pct(def_team, "blitz_rate")
            if blitz_pct is not None and blitz_pct <= 25 and blitz > 0:
                context_parts.append(f"low blitz {blitz:.0%} ({_ordinal(blitz_pct)} pct)")

            edges.append((
                ypc_pct + (light_pct or 50),
                f"- **Run D weakness:** " + "; ".join(context_parts)
                + f". {off_team} RB rush yds OVER."
            ))

    # ─────────────────────────────────────────────────────────
    # 5. TURNOVER MISMATCH (top INT-forcing D vs INT-prone QB matchup)
    # Only flag top-quintile INT-forcing defense.
    # ─────────────────────────────────────────────────────────
    for def_team, off_team in [(home, away), (away, home)]:
        int_f = matchup_features.get(def_team, {}).get("int_forced_rate", 0)
        # Use legacy percentile proxy: sort against all teams
        if int_f > 0.030:
            # Count how many teams have lower rate — manual percentile
            all_int = [matchup_features.get(t, {}).get("int_forced_rate", 0)
                       for t in matchup_features if matchup_features.get(t, {}).get("int_forced_rate")]
            if all_int:
                rank = sum(1 for v in all_int if v <= int_f) / len(all_int) * 100
                if rank >= 80:
                    edges.append((
                        rank,
                        f"- **INT pressure:** {def_team} DEF forces INTs at "
                        f"{int_f:.1%} ({_ordinal(int(rank))} pct). "
                        f"Watch {off_team} QB INT prop OVER."
                    ))

    # ─────────────────────────────────────────────────────────
    # Sort by mismatch severity, cap at 3 edges per game
    # ─────────────────────────────────────────────────────────
    edges.sort(key=lambda x: x[0], reverse=True)
    return [e[1] for e in edges[:3]]


# ─────────────────────────────────────────────────────────────
# Roster validation canary
# ─────────────────────────────────────────────────────────────
def _validate_roster_resolution(
    usage: dict, current_roster: dict, implications: dict
) -> dict:
    """Validate that brief-referenced players resolve to current roster teams.

    Logs a summary each run and flags dangerous conditions:
    - Team mismatches (player on wrong team)
    - Starters being suppressed (false-negative risk)

    Returns {resolved, mismatches, starters_suppressed, details}.
    """
    if not current_roster:
        logger.warning("[ROSTER VALIDATION] No current roster loaded — all team "
                       "assignments are from PBP (potentially stale)")
        return {"resolved": 0, "mismatches": 0, "starters_suppressed": 0}

    resolved = 0
    mismatches = 0
    mismatch_details = []

    for key, entry in usage.items():
        norm = _normalize_name(entry["player_name"])
        roster_info = current_roster.get(norm)
        if roster_info:
            resolved += 1
            if entry.get("pbp_team") and entry["pbp_team"] != entry["team"]:
                mismatches += 1
                mismatch_details.append(
                    f"{entry['player_name']}: PBP={entry['pbp_team']} → "
                    f"current={entry['team']} (pos={entry.get('position','')})"
                )

    # Check for suppressed starters — the DANGEROUS false-negative
    starters_suppressed = 0
    suppressed_details = []
    for team_abbr, impl in implications.items():
        for inj in impl.get("injured", []):
            if inj["severity"] < 2:
                continue
            norm = _normalize_name(inj["player_name"])
            roster_info = current_roster.get(norm)
            if not roster_info:
                continue
            # Was this player suppressed (not generating implications)?
            # A starter (depth_rank=1) should never be suppressed
            if roster_info.get("depth_rank") == 1:
                # Check if they generated any implications
                generated = any(
                    b.get("because_of") == inj["player_name"]
                    for b in impl.get("beneficiaries", [])
                ) or any(
                    h.get("because_of") == inj["player_name"]
                    for h in impl.get("harmed", [])
                )
                if not generated:
                    # Could be because they're in a non-offensive position,
                    # or have low usage — only flag offensive starters
                    if inj["position"] in _OFFENSIVE_POSITIONS:
                        starters_suppressed += 1
                        suppressed_details.append(
                            f"{inj['player_name']} ({inj['position']}, "
                            f"{team_abbr}, depth_rank=1) — OUT but no "
                            f"implications generated (REVIEW)"
                        )

    # Log summary
    logger.info(
        "[ROSTER VALIDATION] %d players resolved, %d team changes remapped, "
        "%d starters with no implications (review)",
        resolved, mismatches, starters_suppressed,
    )

    if mismatch_details:
        logger.info("[ROSTER VALIDATION] Team changes: %s",
                    "; ".join(mismatch_details[:10]))
        if len(mismatch_details) > 10:
            logger.info("[ROSTER VALIDATION] ... and %d more",
                        len(mismatch_details) - 10)

    if suppressed_details:
        for sd in suppressed_details:
            logger.warning("[ROSTER VALIDATION] STARTER NO-IMPL: %s", sd)

    return {
        "resolved": resolved,
        "mismatches": mismatches,
        "starters_suppressed": starters_suppressed,
        "mismatch_details": mismatch_details,
        "suppressed_details": suppressed_details,
    }


# ─────────────────────────────────────────────────────────────
# Brief rendering
# ─────────────────────────────────────────────────────────────
def build_structured_brief(game_date: str, items: list[dict]) -> str:
    """Build the full structured NFL brief: Game → Team → Player.

    Args:
        game_date: YYYY-MM-DD
        items: list of researcher item dicts (from fetch_all or DB)

    Returns:
        Markdown string ready for output.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 1. Find current week and load schedule
    season, week = _find_current_week(game_date)
    games = _load_week_schedule(season, week)
    logger.info("[brief_builder] Season %d Week %d: %d games", season, week, len(games))

    # 2. Load current-season roster (source of truth for team assignments)
    current_roster = _load_current_roster(season)

    # 3. Parse injuries
    all_injuries = _parse_injury_items(items)
    injuries_by_team = defaultdict(list)
    for inj in all_injuries:
        if inj["team_abbr"]:
            injuries_by_team[inj["team_abbr"]].append(inj)

    # 4. Load player usage data (remapped to current-season teams)
    usage = _load_player_usage(season, current_roster=current_roster)
    logger.info("[brief_builder] Loaded usage for %d players", len(usage))

    # 5. Collect game dates for prop lookup
    game_dates = sorted(set(g["gameday"] for g in games if g.get("gameday")))
    active_props = _load_active_props(game_dates)
    logger.info("[brief_builder] Loaded props for %d players", len(active_props))

    # 6. Load pressure profiles (for OL injury implications)
    pressure = _load_pressure_profiles(season)

    # 7. Load projections, consensus edges, and matchup features
    projections, edges = _load_projections_and_edges(season, week)
    logger.info("[brief_builder] Loaded projections for %d players, edges for %d",
                len(projections), len(edges))
    matchup_features = _load_matchup_features(season)
    logger.info("[brief_builder] Loaded matchup features for %d teams", len(matchup_features))

    # 7b. Load PBP-derived matchup features (correct RZ denominator, deep ball, run D)
    pbp_matchup = _load_pbp_matchup_features(season)
    if pbp_matchup:
        _compute_percentile_ranks(pbp_matchup)
    logger.info("[brief_builder] PBP matchup features: %d teams with percentiles",
                len(pbp_matchup))

    # 8. Compute implications (pass current_roster for depth-chart-aware filter)
    implications = _compute_implications(
        injuries_by_team, usage, active_props, pressure, games,
        projections=projections, edges=edges,
        current_roster=current_roster,
    )

    # 9. Validate roster resolution
    _validate_roster_resolution(usage, current_roster, implications)

    # 10. Render
    parts = []
    parts.append(f"# NFL Brief — Week {week} ({game_date})")
    parts.append(f"*Generated {now} · {len(all_injuries)} injury items across {len(injuries_by_team)} teams*\n")

    # ── High-value alerts: active players who benefit ──
    all_active_bens = []
    for game in games:
        for team in [game["home_team"], game["away_team"]]:
            team_injuries = injuries_by_team.get(team, [])
            bens = _find_active_beneficiaries(
                team, team_injuries, usage, active_props,
                projections=projections, edges=edges,
                current_roster=current_roster,
            )
            for b in bens:
                b["team"] = team
            all_active_bens.extend(bens)

    if all_active_bens:
        # Filter to those with props (highest actionability)
        prop_bens = [b for b in all_active_bens if b["props"]]
        no_prop_bens = [b for b in all_active_bens if not b["props"] and (
            b.get("target_share", 0) > 0.12 or b.get("carry_share", 0) > 0.15
        )]

        if prop_bens or no_prop_bens:
            parts.append("---")
            parts.append("## ACTIVE PLAYERS WITH INCREASED ROLE\n")
            parts.append("*Players who are healthy but benefit from teammate injuries — highest-value lines.*\n")

            for b in prop_bens:
                prop_str = _format_props(b["props"])
                share = b.get("target_share") or b.get("carry_share", 0)
                line = (
                    f"- **{b['player_name']}** ({b['team']} {b['position']}) — "
                    f"{b['because_out']} ({b['out_position']}) is OUT. "
                    f"Share: {share:.0%}. "
                    f"Props: {prop_str}"
                )
                # Add model edges if significant (>5pp gap)
                edge_str = _format_edges(b.get("edges", {}))
                if edge_str:
                    line += f"\n  - Model: {edge_str}"
                parts.append(line)

            for b in no_prop_bens:
                share = b.get("target_share") or b.get("carry_share", 0)
                line = (
                    f"- **{b['player_name']}** ({b['team']} {b['position']}) — "
                    f"{b['because_out']} ({b['out_position']}) is OUT. "
                    f"Share: {share:.0%}. No active props."
                )
                edge_str = _format_edges(b.get("edges", {}))
                if edge_str:
                    line += f"\n  - Model: {edge_str}"
                parts.append(line)

            parts.append("")

    # ── Game-by-game breakdown ──
    parts.append("---")

    for game in sorted(games, key=lambda g: (g["gameday"], g.get("gametime", ""))):
        away = game["away_team"]
        home = game["home_team"]
        spread = game.get("spread")
        total = game.get("total")

        # Skip completed games
        if game.get("home_score") is not None:
            continue

        spread_str = f"{spread:+.1f}" if spread is not None else "?"
        total_str = f"{total:.1f}" if total is not None else "?"
        time_str = game.get("gametime", "")

        parts.append(f"## {away} @ {home}")
        parts.append(
            f"*{game['gameday']} {time_str} · "
            f"Spread: {home} {spread_str} · Total: {total_str}*\n"
        )

        # Process each team in this game
        for team_abbr in [away, home]:
            team_injuries = injuries_by_team.get(team_abbr, [])
            out_doubt = [p for p in team_injuries if p["severity"] >= 2]
            questionable = [p for p in team_injuries if p["severity"] == 1]

            # Only show team section if there are injuries or implications
            impl = implications.get(team_abbr)
            if not out_doubt and not questionable and not impl:
                parts.append(f"### {team_abbr}\n")
                parts.append("No significant injury news.\n")
                continue

            parts.append(f"### {team_abbr}\n")

            # Injured players (sorted by severity, then usage)
            if out_doubt:
                parts.append("**OUT / DOUBTFUL:**\n")
                for p in sorted(out_doubt, key=lambda x: (-x["severity"], x["player_name"])):
                    status_label = p["status"].upper()
                    body_snip = f" — {p['body'][:100]}" if p["body"] else ""
                    inj_props = _get_player_props(p["player_name"], active_props)
                    prop_str = f" | Props: {_format_props(inj_props)}" if inj_props else ""
                    parts.append(
                        f"- **{p['player_name']}** ({p['position']}) — "
                        f"{status_label}{body_snip}{prop_str}"
                    )
                parts.append("")

            if questionable:
                parts.append("**QUESTIONABLE:**\n")
                for p in sorted(questionable, key=lambda x: x["player_name"]):
                    body_snip = f" — {p['body'][:100]}" if p["body"] else ""
                    inj_props = _get_player_props(p["player_name"], active_props)
                    prop_str = f" | Props: {_format_props(inj_props)}" if inj_props else ""
                    parts.append(
                        f"- **{p['player_name']}** ({p['position']}) — "
                        f"Questionable{body_snip}{prop_str}"
                    )
                parts.append("")

            # ── IMPLICATIONS ──
            if impl and (impl["beneficiaries"] or impl["harmed"]):
                parts.append("**IMPLICATIONS:**\n")

                if impl["beneficiaries"]:
                    parts.append("*Beneficiaries (volume redistribution):*\n")
                    seen = set()
                    for b in sorted(impl["beneficiaries"],
                                    key=lambda x: -x.get("boost_pp", 0)):
                        if b["player_name"] in seen:
                            continue
                        seen.add(b["player_name"])

                        boost = b.get("boost_pp", 0)
                        share = b.get("current_share", 0)
                        new_share = share + boost / 100.0
                        source_label = " [heuristic]" if b.get("source") == "heuristic" else " [with/without]"

                        line = (
                            f"- **{b['player_name']}** ({b['position']}): "
                            f"{b['note']} — share {share:.0%} → {new_share:.0%} "
                            f"(+{boost:.0f}pp){source_label}"
                        )

                        if b.get("tension"):
                            line += f"\n  - *Tension:* {b['tension']}"

                        # Props with model edges
                        if b.get("props"):
                            line += f"\n  - Props: {_format_props(b['props'])}"
                        edge_str = _format_edges(b.get("edges", {}))
                        if edge_str:
                            line += f"\n  - Model: {edge_str}"

                        parts.append(line)

                    parts.append("")

                if impl["harmed"]:
                    parts.append("*Harmed (inverse effects):*\n")
                    seen = set()
                    for h in impl["harmed"]:
                        key = f"{h['player_name']}_{h['because_of']}"
                        if key in seen:
                            continue
                        seen.add(key)

                        line = (
                            f"- **{h['player_name']}** ({h['position']}): "
                            f"{h['effect']} (because {h['because_of']} out)"
                        )
                        if h.get("props"):
                            line += f"\n  - Props: {_format_props(h['props'])}"
                        edge_str = _format_edges(h.get("edges", {}))
                        if edge_str:
                            line += f"\n  - Model: {edge_str}"
                        parts.append(line)
                    parts.append("")

                if impl["opponent_effects"]:
                    parts.append("*Opponent effects:*\n")
                    for oe in impl["opponent_effects"]:
                        parts.append(f"- {oe['note']}")
                    parts.append("")

        # ── MATCHUP EDGES (after both teams' injuries) ──
        matchup_lines = _format_game_matchup_edges(
            away, home, matchup_features, pressure,
            pbp_features=pbp_matchup,
        )
        if matchup_lines:
            parts.append("### Matchup Edges\n")
            for ml in matchup_lines:
                parts.append(ml)
            parts.append("")

    # ── YouTube analyst notes (if any) ──
    yt_items = [i for i in items if i.get("source", "").startswith("youtube:")]
    if yt_items:
        parts.append("---")
        parts.append(f"## Analyst Video Notes ({len(yt_items)})\n")
        for yt in yt_items:
            src = yt.get("source", "youtube")
            channel = src.replace("youtube:", "")
            headline = yt.get("headline", "")
            url = yt.get("url", "")
            body_snip = (yt.get("body") or "")[:200]
            link = f"[{headline}]({url})" if url else headline
            parts.append(f"- **{channel}**: {link}")
            if body_snip:
                parts.append(f"  - {body_snip}")
        parts.append("")

    # ── ESPN headlines (non-injury) ──
    news_items = [i for i in items
                  if i.get("source") in ("espn_nfl_rss",) and i.get("category") != "injury"]
    if news_items:
        parts.append("---")
        parts.append(f"## NFL Headlines ({len(news_items)})\n")
        for n in news_items[:10]:
            url = n.get("url", "")
            headline = n.get("headline", "")
            link = f"[{headline}]({url})" if url else headline
            parts.append(f"- {link}")
        if len(news_items) > 10:
            parts.append(f"- *... and {len(news_items) - 10} more*")
        parts.append("")

    return "\n".join(parts)


def _format_props(props: list[dict]) -> str:
    """Format prop lines for inline display."""
    if not props:
        return ""
    # Show up to 3 most relevant props
    display = []
    for p in props[:3]:
        name = p.get("display", p.get("market", "?"))
        point = p.get("point")
        if point is not None:
            display.append(f"{name} {point}")
        else:
            display.append(name)
    return ", ".join(display)


def _format_edges(player_edges: dict, threshold_pp: float = 5.0) -> str:
    """Format model edges for display. Only shows edges exceeding threshold.

    Args:
        player_edges: {stat_type: {mean, line, gap, gap_pct, direction, confidence}}
        threshold_pp: minimum gap_pct to display (default 5pp)

    Returns:
        Formatted string like "pass yds OVER 232.5 (+8.2%), rush yds UNDER 65.5 (-12.1%)"
        or empty string if no edges exceed threshold.
    """
    if not player_edges:
        return ""

    display = []
    for stat, info in sorted(player_edges.items()):
        gap_pct = abs(info.get("gap_pct", 0) or 0)
        if gap_pct < threshold_pp:
            continue
        direction = info.get("direction", "over").upper()
        line = info.get("line")
        mean = info.get("mean")
        stat_display = _PROP_DISPLAY.get(stat, stat.replace("player_", "").replace("_", " "))
        if line is not None and mean is not None:
            sign = "+" if info.get("gap", 0) >= 0 else ""
            display.append(
                f"{stat_display} {direction} {line} "
                f"(proj {mean:.1f}, {sign}{info.get('gap_pct', 0):.1f}%)"
            )
        elif line is not None:
            display.append(f"{stat_display} {direction} {line} ({gap_pct:.1f}% edge)")

    return ", ".join(display)
