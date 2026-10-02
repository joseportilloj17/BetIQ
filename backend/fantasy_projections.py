"""
fantasy_projections.py — NFL fantasy projection ingestion + prop gap analysis.

Architecture:
  1. Ingest per-player per-week component stat projections from multiple
     fantasy sources (Sleeper, ESPN Fantasy, FantasyPros).
  2. Build consensus: mean, min, max, SD across sources per (player, week, stat).
  3. Map to TheOddsAPI player prop lines and compute gaps.
  4. Rank gaps by size × consensus tightness (low SD) = candidate soft spots.

Sources:
  - Sleeper: free, no auth, weekly component projections (from RotoWire).
    Endpoint: api.sleeper.com/projections/nfl/{season}/{week}
  - ESPN Fantasy: free, no auth, unofficial API.
    Endpoint: site.api.espn.com/apis/fantasy/v2/games/ffl/...
  - FantasyPros: paid API ($70-130/yr HOF sub), consensus projections.
    Requires x-api-key header.

Key principle: compare COMPONENT stats (rush_yds, rec_yds, pass_yds, etc.)
to matching prop lines — NOT fantasy points to yards lines.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime
from math import sqrt
from typing import Optional

logger = logging.getLogger(__name__)

_HERE     = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_HIST_DB  = os.path.join(_DATA_DIR, "historical.db")

# ── Stat-to-prop mapping ──────────────────────────────────────────────────────
# Maps fantasy component stat names → TheOddsAPI player prop market keys.
# Each source may use different field names; we normalize to these canonical keys.
STAT_TO_PROP = {
    "pass_yds":  "player_pass_yds",
    "pass_tds":  "player_pass_tds",
    "rush_yds":  "player_rush_yds",
    "rec_yds":   "player_reception_yds",
    "receptions": "player_receptions",
}

# Alt market keys — same stats but alternate lines (full ladder)
STAT_TO_ALT_PROP = {
    "pass_yds":  "player_pass_yds_alternate",
    "rush_yds":  "player_rush_yds_alternate",
    "rec_yds":   "player_reception_yds_alternate",
    # No alt markets for pass_tds or receptions currently
}

# Canonical stat types we track (subset of what sources provide)
TRACKED_STATS = list(STAT_TO_PROP.keys())

# ── Source field name normalization ────────────────────────────────────────────
# Sleeper uses slightly different field names than our canonical keys.
_SLEEPER_FIELD_MAP = {
    "pass_yd":  "pass_yds",
    "pass_td":  "pass_tds",
    "rush_yd":  "rush_yds",
    "rec_yd":   "rec_yds",
    "rec":      "receptions",
}

# ESPN stat ID → canonical stat name
# ESPN uses numeric stat IDs in their fantasy API.
_ESPN_STAT_MAP = {
    "3":  "pass_yds",     # passingYards
    "4":  "pass_tds",     # passingTouchdowns
    "14": "rush_yds",     # rushingYards (corrected if needed)
    "24": "rec_yds",      # receivingYards
    "41": "receptions",   # receptions
}


# ══════════════════════════════════════════════════════════════════════════════
# HTTP helper
# ══════════════════════════════════════════════════════════════════════════════

def _get(url: str, headers: dict | None = None, timeout: int = 15) -> Optional[dict | list]:
    """Simple GET → JSON with error handling."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BetIQ/1.0 (fantasy projections; educational)",
                      **(headers or {})},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Fantasy projection fetch failed: %s — %s", url, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Source 1: SLEEPER (free, no auth)
# ══════════════════════════════════════════════════════════════════════════════

# Cache the Sleeper player database (large, ~5 MB) — fetched once per session.
_SLEEPER_PLAYERS: dict | None = None


def _sleeper_player_db() -> dict:
    """Fetch the Sleeper player database. Returns {player_id: {full_name, ...}}."""
    global _SLEEPER_PLAYERS
    if _SLEEPER_PLAYERS is not None:
        return _SLEEPER_PLAYERS
    data = _get("https://api.sleeper.app/v1/players/nfl", timeout=30)
    if data and isinstance(data, dict):
        _SLEEPER_PLAYERS = data
        logger.info("[Sleeper] Loaded %d players", len(data))
    else:
        _SLEEPER_PLAYERS = {}
        logger.warning("[Sleeper] Failed to load player database")
    return _SLEEPER_PLAYERS


def fetch_sleeper_projections(season: int, week: int) -> list[dict]:
    """
    Fetch week-level component stat projections from Sleeper.

    Returns list of:
        {"player_name": str, "position": str, "team": str,
         "stats": {"pass_yds": float, "rush_yds": float, ...},
         "source": "sleeper"}
    """
    url = f"https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular"
    raw = _get(url, timeout=20)
    if not raw or not isinstance(raw, list):
        logger.warning("[Sleeper] No projection data for %d week %d", season, week)
        return []

    player_db = _sleeper_player_db()
    results = []

    for entry in raw:
        pid = entry.get("player_id", "")
        stats_raw = entry.get("stats") or {}
        if not stats_raw:
            continue

        # Resolve player name from player DB
        pinfo = player_db.get(str(pid), {})
        full_name = pinfo.get("full_name") or pinfo.get("first_name", "?") + " " + pinfo.get("last_name", "?")
        position = pinfo.get("position", "")
        team = pinfo.get("team", "")

        # Only track offensive skill positions
        if position not in ("QB", "RB", "WR", "TE"):
            continue

        # Normalize field names and extract tracked stats
        normalized = {}
        for sleeper_key, canonical_key in _SLEEPER_FIELD_MAP.items():
            val = stats_raw.get(sleeper_key)
            if val is not None and val > 0:
                normalized[canonical_key] = float(val)

        if not normalized:
            continue

        results.append({
            "player_name": full_name.strip(),
            "player_id_sleeper": str(pid),
            "position": position,
            "team": team,
            "stats": normalized,
            "source": "sleeper",
        })

    logger.info("[Sleeper] Fetched %d player projections for %d week %d",
                len(results), season, week)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Source 2: ESPN FANTASY (free, no auth — unofficial API)
# ══════════════════════════════════════════════════════════════════════════════

# ESPN position IDs: 1=QB, 2=RB, 3=WR, 4=TE, 5=K, 16=D/ST
_ESPN_POS_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}

# ESPN stat category IDs for projected stats (used in fantasy API)
# statSourceId=1 is projected, statSourceId=0 is actual
# Full map verified from espn-api package constant.py
_ESPN_STAT_IDS = {
    "0":  "pass_att",       # passingAttempts
    "1":  "pass_cmp",       # passingCompletions
    "3":  "pass_yds",       # passingYards
    "4":  "pass_tds",       # passingTouchdowns
    "20": "pass_int",       # passingInterceptions
    "23": "rush_att",       # rushingAttempts
    "24": "rush_yds",       # rushingYards
    "25": "rush_tds",       # rushingTouchdowns
    "41": "receptions",     # receivingReceptions
    "42": "rec_yds",        # receivingYards
    "43": "rec_tds",        # receivingTouchdowns
    "58": "rec_targets",    # receivingTargets
}


def fetch_espn_projections(season: int, week: int) -> list[dict]:
    """
    Fetch week-level player projections from ESPN Fantasy API.

    ESPN uses a filter-based system where you request players with their
    projected stats for a specific scoring period (week).

    Returns list of:
        {"player_name": str, "position": str, "team": str,
         "stats": {"pass_yds": float, ...}, "source": "espn"}
    """
    # ESPN scoring periods: week 1 = period 1, etc.
    scoring_period = week

    # The public /players endpoint — no league ID required, no auth needed.
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{season}/players"
        f"?scoringPeriodId={scoring_period}&view=kona_player_info"
    )

    # ESPN requires a filter header to specify what data to return
    filter_json = json.dumps({
        "players": {
            "filterSlotIds": {"value": [0, 2, 4, 6]},  # QB=0, RB=2, WR=4, TE=6
            "filterStatsForSourceIds": {"value": [1]},  # 1 = projected
            "filterStatsForScoringPeriodIds": {"value": [scoring_period]},
            "sortDraftRanks": {
                "sortPriority": 101,
                "sortAscending": True,
                "value": "PPR",
            },
            "limit": 500,
            "offset": 0,
        }
    })

    headers = {"x-fantasy-filter": filter_json}
    data = _get(url, headers=headers, timeout=20)

    if not data:
        logger.warning("[ESPN] No projection data for %d week %d", season, week)
        return []

    # The /players endpoint returns a list directly; the /leagues endpoint
    # nests under {"players": [...]}.
    player_list = data if isinstance(data, list) else data.get("players", [])
    if not player_list:
        logger.warning("[ESPN] Empty player list for %d week %d", season, week)
        return []

    results = []
    for player_entry in player_list:
        # Each entry may be {player: {...}} or flat depending on endpoint
        pinfo = player_entry.get("player", player_entry) if isinstance(player_entry, dict) else {}
        full_name = pinfo.get("fullName", "")
        if not full_name:
            continue
        pos_id = pinfo.get("defaultPositionId", 0)
        position = _ESPN_POS_MAP.get(pos_id)
        if not position:
            continue

        # Get team abbreviation (ESPN uses numeric IDs)
        team = str(pinfo.get("proTeamId", ""))

        # Extract projected stats from the stats array
        normalized = {}
        for stat_entry in pinfo.get("stats", []):
            # statSourceId=1 is projected, statSplitTypeId=1 is by week
            if (stat_entry.get("statSourceId") == 1 and
                stat_entry.get("scoringPeriodId") == scoring_period):
                raw_stats = stat_entry.get("stats", {})
                for espn_id, canonical in _ESPN_STAT_IDS.items():
                    val = raw_stats.get(espn_id)
                    if val is not None and val > 0 and canonical in TRACKED_STATS:
                        normalized[canonical] = float(val)
                break  # found the right stats entry

        if not normalized:
            continue

        results.append({
            "player_name": full_name.strip(),
            "position": position,
            "team": team,
            "stats": normalized,
            "source": "espn",
        })

    logger.info("[ESPN] Fetched %d player projections for %d week %d",
                len(results), season, week)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Consensus builder
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_player_name(name: str) -> str:
    """Normalize player name for matching across sources.

    Strip suffixes (Jr., III, II), lowercase, collapse whitespace.
    """
    import re
    name = name.strip().lower()
    name = re.sub(r'\s+(jr\.?|sr\.?|ii+|iv|v)$', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def build_consensus(all_projections: list[list[dict]]) -> dict:
    """
    Build consensus projections across multiple sources.

    Args:
        all_projections: list of source results, each a list of player dicts
                         from fetch_sleeper_projections(), fetch_espn_projections(), etc.

    Returns:
        {normalized_player_name: {
            "display_name": str,
            "position": str,
            "team": str,
            "stats": {
                stat_type: {
                    "values": [v1, v2, ...],  # one per source
                    "sources": ["sleeper", "espn", ...],
                    "mean": float,
                    "min": float,
                    "max": float,
                    "sd": float,
                    "n_sources": int,
                }
            }
        }}
    """
    # Group by normalized player name
    player_data: dict[str, dict] = {}

    for source_projections in all_projections:
        for proj in source_projections:
            norm_name = _normalize_player_name(proj["player_name"])
            if norm_name not in player_data:
                player_data[norm_name] = {
                    "display_name": proj["player_name"],
                    "position": proj["position"],
                    "team": proj.get("team", ""),
                    "stats": defaultdict(lambda: {"values": [], "sources": []}),
                }

            for stat_type, value in proj["stats"].items():
                if stat_type in TRACKED_STATS:
                    player_data[norm_name]["stats"][stat_type]["values"].append(value)
                    player_data[norm_name]["stats"][stat_type]["sources"].append(proj["source"])

    # Compute consensus statistics
    consensus = {}
    for norm_name, pdata in player_data.items():
        stats_out = {}
        for stat_type, sdata in pdata["stats"].items():
            vals = sdata["values"]
            n = len(vals)
            if n == 0:
                continue
            mean_val = sum(vals) / n
            min_val = min(vals)
            max_val = max(vals)
            if n >= 2:
                variance = sum((v - mean_val) ** 2 for v in vals) / (n - 1)
                sd_val = sqrt(variance)
            else:
                sd_val = 0.0

            stats_out[stat_type] = {
                "values": vals,
                "sources": sdata["sources"],
                "mean": round(mean_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "sd": round(sd_val, 2),
                "n_sources": n,
            }

        if stats_out:
            consensus[norm_name] = {
                "display_name": pdata["display_name"],
                "position": pdata["position"],
                "team": pdata["team"],
                "stats": stats_out,
            }

    return consensus


# ══════════════════════════════════════════════════════════════════════════════
# Prop line matching + gap analysis
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_current_nfl_prop_lines() -> dict:
    """
    Fetch current NFL player prop lines from historical.db (player_props table).

    Returns:
        {normalized_player_name: {
            market_key: {"point": float, "over_price": float, "under_price": float,
                         "book": str}
        }}
    """
    if not os.path.exists(os.path.abspath(_HIST_DB)):
        logger.warning("[Props] historical.db not found at %s", _HIST_DB)
        return {}

    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    try:
        # Get today's NFL props — most recent fetch per (player, market)
        rows = conn.execute("""
            SELECT player_name, market_key, point, price, over_under, bookmaker,
                   fetched_at, game_date
            FROM player_props
            WHERE sport_key = 'americanfootball_nfl'
              AND game_date >= date('now', '-1 day')
            ORDER BY fetched_at DESC
        """).fetchall()
    except Exception as exc:
        logger.warning("[Props] Query failed: %s", exc)
        return {}
    finally:
        conn.close()

    # Deduplicate: keep most recent per (player, market, point, over/under)
    # For main lines: one point per market_key
    # For alt lines: multiple points per market_key (the ladder)
    props: dict[str, dict] = {}
    seen = set()
    for row in rows:
        pname = row["player_name"]
        mkey = row["market_key"]
        ou = row["over_under"]
        point = row["point"]
        is_alt = mkey.endswith("_alternate")

        # For alt lines, dedup by (player, market, POINT, ou) to keep the full ladder
        dedup_key = (pname, mkey, point, ou) if is_alt else (pname, mkey, ou)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        norm = _normalize_player_name(pname)
        if norm not in props:
            props[norm] = {}

        if is_alt:
            # Store alt lines as a list under the market key
            if mkey not in props[norm]:
                props[norm][mkey] = []
            # Find or create entry for this point
            entry = None
            for e in props[norm][mkey]:
                if e["point"] == point:
                    entry = e
                    break
            if entry is None:
                entry = {"point": point}
                props[norm][mkey].append(entry)
            if ou == "over":
                entry["over_price"] = row["price"]
            else:
                entry["under_price"] = row["price"]
            entry["book"] = row["bookmaker"]
        else:
            # Main lines: single dict per market
            if mkey not in props[norm]:
                props[norm][mkey] = {"point": point}
            if ou == "over":
                props[norm][mkey]["over_price"] = row["price"]
            else:
                props[norm][mkey]["under_price"] = row["price"]
            props[norm][mkey]["book"] = row["bookmaker"]

    return props


def _build_player_baselines(season: int, current_week: int,
                            lookback_weeks: int = 4) -> dict[str, dict[str, float]]:
    """
    Build per-player baseline projections from stored fantasy_projections.

    Baseline = mean projection over the last `lookback_weeks` stored weeks
    before `current_week`. Used to detect role changes (big projection jump).

    Returns: {normalized_player_name: {stat: baseline_mean}}
    """
    if not os.path.exists(os.path.abspath(_HIST_DB)):
        return {}

    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    try:
        rows = conn.execute("""
            SELECT player_name, stat_type, AVG(projected_value) as baseline
            FROM fantasy_projections
            WHERE season = ? AND week >= ? AND week < ?
            GROUP BY player_name, stat_type
        """, (season, max(1, current_week - lookback_weeks), current_week)).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    baselines: dict[str, dict[str, float]] = {}
    for row in rows:
        norm = _normalize_player_name(row[0])
        if norm not in baselines:
            baselines[norm] = {}
        baselines[norm][row[1]] = float(row[2])
    return baselines


def _build_player_variances(season: int, current_week: int) -> dict[str, dict[str, float]]:
    """
    Compute per-player game-to-game variance from stored projections.

    Uses stored raw projections as a proxy for actuals (we don't store actuals
    during forward mode — only projections). For variance-deviation, what matters
    is the SPREAD of a player's projections across weeks relative to other players.

    Returns: {normalized_name: {stat: variance}}
    """
    if not os.path.exists(os.path.abspath(_HIST_DB)):
        return {}

    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    try:
        # Get all stored projections for this player/season up to current week
        rows = conn.execute("""
            SELECT player_name, stat_type, projected_value
            FROM fantasy_projections
            WHERE season = ? AND week < ?
            ORDER BY player_name, stat_type, week
        """, (season, current_week)).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    # Group by player + stat
    from collections import defaultdict
    player_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        norm = _normalize_player_name(row[0])
        player_vals[norm][row[1]].append(float(row[2]))

    # Compute variance (need >= 3 data points)
    result: dict[str, dict[str, float]] = {}
    for norm, stats in player_vals.items():
        result[norm] = {}
        for stat, vals in stats.items():
            if len(vals) >= 3:
                m = sum(vals) / len(vals)
                v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
                result[norm][stat] = v
    return result


def _tag_gap_reason(projected: float, baseline: float | None,
                    role_change_threshold: float = 0.30) -> str:
    """
    Tag the reason for a projection-vs-line gap.

    Categories:
      "role_change" — projection jumped >30% vs recent baseline (depth chart/injury)
      "matchup"     — projection differs from line but baseline is stable
      "generic"     — no baseline available to distinguish

    Args:
        projected: this week's consensus projection
        baseline: average projection from recent weeks (None if no history)
        role_change_threshold: fractional jump to qualify as role change (0.30 = 30%)
    """
    if baseline is None or baseline <= 0:
        return "generic"

    pct_change = (projected - baseline) / baseline
    if abs(pct_change) >= role_change_threshold:
        return "role_change"
    return "matchup"


def _build_tendency_adjustments(prop_lines: dict) -> dict:
    """Build per-team tendency adjustments from nfl_tendency_features.

    Returns {nfl_abbr: {
        "opp_abbr": str,
        "rush_mean_adj": float,   # multiplier for rush consensus (1.0 = no change)
        "rec_te_rb_mean_adj": float,  # multiplier for TE/RB rec consensus
        "sigma_mult": float,      # variance multiplier from blitz profile
        "volume_mult": float,     # volume multiplier from QB mobility
    }}

    Uses player_props game matchups to determine opponent, then looks up
    tendency profiles from nfl_tendency_features.
    """
    try:
        from nfl_tendency_features import build_team_tendency_profiles, get_matchup_features
    except ImportError:
        logger.debug("[tendency] nfl_tendency_features not available")
        return {}

    # Map full team names → NFL abbreviations
    _FULL_TO_ABBR = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
        "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE",
        "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    }

    # Build matchup pairs from player_props data (home_team, away_team)
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    try:
        rows = conn.execute("""
            SELECT DISTINCT home_team, away_team FROM player_props
            WHERE sport_key = 'americanfootball_nfl'
              AND game_date >= date('now', '-1 day')
        """).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    if not rows:
        return {}

    # Build team→opponent mapping (in abbreviations)
    team_to_opp = {}
    matchup_pairs = set()
    for home_full, away_full in rows:
        home = _FULL_TO_ABBR.get(home_full, home_full)
        away = _FULL_TO_ABBR.get(away_full, away_full)
        team_to_opp[home] = away
        team_to_opp[away] = home
        matchup_pairs.add((home, away))

    # Build tendency profiles (cached — uses recent 8 games)
    try:
        profiles = build_team_tendency_profiles()
    except Exception as exc:
        logger.warning("[tendency] Profile build failed: %s", exc)
        return {}

    # Compute league-average baselines for relative adjustments
    adjustments = {}
    for team, opp in team_to_opp.items():
        adj = {"opp_abbr": opp, "rush_mean_adj": 1.0, "rec_te_rb_mean_adj": 1.0,
               "sigma_mult": 1.0, "volume_mult": 1.0}

        # Rush adjustment: opponent's rush_ypa_allowed vs league average
        if "rb_box" in profiles:
            rb = profiles["rb_box"]
            opp_row = rb[rb["defteam"] == opp]
            if not opp_row.empty:
                opp_ypa = opp_row.iloc[0].get("rush_ypa_allowed")
                if opp_ypa is not None:
                    league_avg_ypa = rb["rush_ypa_allowed"].mean()
                    if league_avg_ypa > 0:
                        # Cap adjustment at ±15% to avoid wild swings
                        ratio = opp_ypa / league_avg_ypa
                        adj["rush_mean_adj"] = max(0.85, min(1.15, ratio))

        # TE/RB rec adjustment: opponent's short_yds_per_target vs league avg
        if "mlb_coverage" in profiles:
            mc = profiles["mlb_coverage"]
            opp_row = mc[mc["defteam"] == opp]
            if not opp_row.empty:
                opp_ypt = opp_row.iloc[0].get("short_yds_per_target")
                if opp_ypt is not None:
                    league_avg_ypt = mc["short_yds_per_target"].mean()
                    if league_avg_ypt > 0:
                        ratio = opp_ypt / league_avg_ypt
                        adj["rec_te_rb_mean_adj"] = max(0.85, min(1.15, ratio))

        # Blitz sigma multiplier
        if "blitz" in profiles:
            bl = profiles["blitz"]
            opp_row = bl[bl["defteam"] == opp]
            if not opp_row.empty:
                sm = opp_row.iloc[0].get("sigma_multiplier")
                if sm is not None:
                    adj["sigma_mult"] = float(sm)

        # QB mobility volume multiplier
        if "qb_mobility" in profiles:
            qm = profiles["qb_mobility"]
            team_row = qm[qm["posteam"] == team]
            if not team_row.empty:
                vm = team_row.iloc[0].get("volume_multiplier")
                if vm is not None:
                    adj["volume_mult"] = float(vm)

        adjustments[team] = adj

    return adjustments


def _compute_segment_bias(season: int | None = None,
                          before_week: int | None = None) -> dict[tuple[str, str], float]:
    """Compute shrunk bias corrections from graded forward predictions.

    Uses a ROLLING PRIOR window: only data from weeks strictly before
    ``before_week`` are included.  This avoids the circularity of fitting
    and evaluating on the same data.

    Bias = mean(consensus_mean - actual_value) per (stat_type, position).
    Positive bias = consensus overshoots (optimistic) → subtract from future projections.
    Shrinkage: shrunk = raw_mean × N / (N + PRIOR_N), PRIOR_N=30.

    Returns: {(stat_type, position): shrunk_bias_value}
    """
    if not os.path.exists(os.path.abspath(_HIST_DB)):
        return {}

    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    try:
        query = """
            SELECT stat_type, position,
                   AVG(consensus_mean - actual_value) as raw_bias,
                   COUNT(*) as n
            FROM fantasy_forward_preds
            WHERE actual_value IS NOT NULL AND is_push = 0
        """
        params: list = []
        if season is not None:
            query += " AND season = ?"
            params.append(season)
        if before_week is not None:
            query += " AND week < ?"
            params.append(before_week)
        query += " GROUP BY stat_type, position HAVING COUNT(*) >= 5"

        rows = conn.execute(query, params).fetchall()
    except Exception as exc:
        logger.warning("[Bias] Query failed: %s", exc)
        return {}
    finally:
        conn.close()

    _PRIOR_N = 30
    biases: dict[tuple[str, str], float] = {}
    for row in rows:
        stat, pos, raw_bias, n = row
        shrunk = raw_bias * (n / (n + _PRIOR_N))
        biases[(stat, pos)] = round(shrunk, 2)

    return biases


def find_prop_gaps(consensus: dict, prop_lines: dict | None = None,
                   min_abs_gap: float = 1.0, min_line: float = 0.5,
                   player_baselines: dict | None = None,
                   player_variances: dict | None = None,
                   tendency_adjustments: dict | None = None,
                   segment_biases: dict | None = None,
                   min_edge_pp: float = 0.0) -> list[dict]:
    """
    Match consensus projections to sportsbook prop lines and find gaps.

    Ranking uses z-score (absolute gap / consensus SD) — not percentage.
    Percentage is kept as a display column only.

    Filters:
        min_abs_gap: minimum absolute gap between projection and line (default 1.0)
        min_line: minimum prop line value to consider (default 0.5 — filters noise)

    Args:
        player_baselines: {norm_name: {stat: baseline_mean}} for role-change detection.
            If None, all gaps tagged "generic". Build with _build_player_baselines().
        segment_biases: {(stat, pos): shrunk_bias} from _compute_segment_bias().
            Subtracted from consensus mean before tendency adjustment.
        min_edge_pp: minimum |edge_vs_market| to emit a gap (0.05 = 5pp no-bet zone).

    Returns sorted list of gap opportunities:
        [{"player": str, "position": str, "stat": str,
          "consensus_mean": float, "consensus_sd": float, "n_sources": int,
          "prop_line": float, "gap": float, "abs_gap": float, "gap_pct": float,
          "direction": "over" | "under",
          "z_score": float,  # |gap| / consensus SD — higher = tighter consensus on a real gap
          "gap_reason": "role_change" | "matchup" | "generic",
          "over_price": float, "under_price": float}]
    """
    if prop_lines is None:
        prop_lines = _fetch_current_nfl_prop_lines()

    gaps = []
    for norm_name, cdata in consensus.items():
        if norm_name not in prop_lines:
            continue

        player_props = prop_lines[norm_name]
        for stat_type, sdata in cdata["stats"].items():
            # Map stat to prop market key
            prop_key = STAT_TO_PROP.get(stat_type)
            if not prop_key or prop_key not in player_props:
                continue

            prop_info = player_props[prop_key]
            line = prop_info.get("point")
            if line is None or line < min_line:
                continue

            mean = sdata["mean"]
            sd = sdata["sd"]
            n = sdata["n_sources"]
            pos = cdata.get("position", "")

            # ── Segment bias correction ──
            # Subtract systematic consensus bias per (stat, position).
            # Positive bias = consensus overshoots → lower the projection.
            # Shrunk via N/(N+30) so thin segments auto-zero.
            bias_correction = 0.0
            if segment_biases:
                bias_correction = segment_biases.get((stat_type, pos), 0.0)
                if bias_correction != 0.0:
                    mean = round(mean - bias_correction, 2)

            # ── Tendency-based consensus adjustment ──
            # Adjusts the consensus mean using matchup-specific defensive
            # tendencies (box count for rush, MLB coverage for TE/RB rec,
            # QB mobility for volume). Capped at ±15% to avoid wild swings.
            tendency_adj = 1.0
            sigma_adj = 1.0
            if tendency_adjustments:
                team = cdata.get("team", "")
                tadj = tendency_adjustments.get(team, {})
                if tadj:
                    # Rush stats: adjust by opponent's rushing defense profile
                    if stat_type == "rush_yds":
                        tendency_adj = tadj.get("rush_mean_adj", 1.0)
                    # Rec stats for TE/RB: adjust by opponent's short coverage
                    elif stat_type == "rec_yds" and pos in ("TE", "RB"):
                        tendency_adj = tadj.get("rec_te_rb_mean_adj", 1.0)
                    # Variance: blitz sigma multiplier
                    sigma_adj = tadj.get("sigma_mult", 1.0)
                    # Volume: QB mobility multiplier (team-wide)
                    volume_mult = tadj.get("volume_mult", 1.0)
                    if stat_type in ("rush_yds", "rec_yds", "pass_yds", "receptions"):
                        tendency_adj *= volume_mult

            mean = round(mean * tendency_adj, 2)

            # Compute gap
            gap = mean - line
            abs_gap = abs(gap)
            gap_pct = (gap / line) * 100 if line > 0 else 0
            direction = "over" if gap > 0 else "under"

            # Filter: skip gaps below absolute threshold
            if abs_gap < min_abs_gap:
                continue

            # z-score: |gap| / consensus SD — measures how many SDs the gap
            # spans across sources. Higher = sources agree AND disagree with line.
            # Floor SD at 1.0 to prevent blow-up when 2 sources nearly agree
            # (SD~0.02 → z_score ~4000 is meaningless).
            # Single-source: SD=0, assign z_score = abs_gap.
            if n >= 2 and sd > 0:
                z_score = abs_gap / max(sd, 1.0)
            else:
                z_score = abs_gap  # single-source fallback: rank by raw gap size

            # Tag gap reason: role_change / matchup / generic
            baseline = None
            if player_baselines:
                baseline = (player_baselines.get(norm_name) or {}).get(stat_type)
            gap_reason = _tag_gap_reason(mean, baseline)

            # Distributional P(over line) — use consensus mean as center.
            dist_type = _STAT_DISTRIBUTION.get(stat_type, "lognormal")

            # Positional-norm CV (default when no per-player data)
            _typical_cv = {"pass_yds": 0.25, "rush_yds": 0.45, "rec_yds": 0.45,
                           "pass_tds": 0.80, "receptions": 0.40}
            norm_cv = _typical_cv.get(stat_type, 0.40)
            norm_var = (mean * norm_cv) ** 2 if mean > 0 else 1.0

            # Per-player variance if available (from game logs with shrinkage)
            player_cv = norm_cv  # default
            var_deviation = 0.0  # how far player deviates from norm
            pv = (player_variances or {}).get(norm_name, {}).get(stat_type)
            if pv is not None and mean > 0:
                player_cv = sqrt(pv) / mean if mean > 0 else norm_cv
                est_var = pv
                # Deviation: player's CV vs positional norm CV
                # >0 means MORE variable than norm (boom/bust)
                # <0 means LESS variable (steady)
                var_deviation = (player_cv - norm_cv) / norm_cv if norm_cv > 0 else 0
            else:
                est_var = norm_var

            # Apply blitz-based sigma adjustment to variance
            if sigma_adj != 1.0:
                est_var *= sigma_adj ** 2  # variance scales as sigma^2

            if dist_type == "lognormal":
                mu_ln, sigma_ln = _lognormal_params(max(mean, 0.1), est_var)
                dist_params = {"mu": mu_ln, "sigma": sigma_ln}
            else:
                mu_nb, r_nb = _negbin_params(max(mean, 0.01), est_var)
                dist_params = {"mu": mu_nb, "r": r_nb}

            p_ov = p_over_line(line, mean, dist_params, dist_type)

            # De-vig implied probability for over
            over_price = prop_info.get("over_price")
            under_price = prop_info.get("under_price")
            devigged = _devig_implied_prob(over_price, under_price)

            # Edge: P(over) - implied P(over) when direction is over,
            #        implied P(under) - P(under) when direction is under
            if devigged is not None:
                if direction == "over":
                    edge_vs_market = p_ov - devigged
                else:
                    edge_vs_market = (1 - devigged) - (1 - p_ov)  # = p_ov - devigged (same)
            else:
                edge_vs_market = None

            # ── No-bet threshold ──
            # Skip props where edge is below the minimum (no-bet zone).
            if min_edge_pp > 0 and edge_vs_market is not None:
                if abs(edge_vs_market) < min_edge_pp:
                    continue

            gaps.append({
                "player": cdata["display_name"],
                "position": cdata["position"],
                "team": cdata["team"],
                "stat": stat_type,
                "prop_market": prop_key,
                "consensus_mean": mean,
                "consensus_sd": sd,
                "n_sources": n,
                "prop_line": line,
                "gap": round(gap, 2),
                "abs_gap": round(abs_gap, 2),
                "gap_pct": round(gap_pct, 2),
                "direction": direction,
                "z_score": round(z_score, 2),
                "p_over": round(p_ov, 4),
                "devigged_implied": round(devigged, 4) if devigged is not None else None,
                "edge_vs_market": round(edge_vs_market, 4) if edge_vs_market is not None else None,
                "gap_reason": gap_reason,
                "line_depth": "main",
                "var_deviation": round(var_deviation, 3),
                "tendency_adj": round(tendency_adj, 3),
                "bias_correction": round(bias_correction, 2),
                "over_price": over_price,
                "under_price": under_price,
            })

            # ── Alt-line ladder evaluation ──
            # Evaluate each alt line using the SAME fitted distribution
            alt_prop_key = STAT_TO_ALT_PROP.get(stat_type)
            if alt_prop_key and alt_prop_key in player_props:
                alt_lines = player_props[alt_prop_key]
                if isinstance(alt_lines, list):
                    for alt_entry in alt_lines:
                        alt_line = alt_entry.get("point")
                        if alt_line is None or alt_line < min_line:
                            continue

                        # P(over alt_line) from the SAME distribution
                        alt_p_ov = p_over_line(alt_line, mean, dist_params, dist_type)

                        # Alt lines typically only have over price
                        alt_over_price = alt_entry.get("over_price")
                        alt_under_price = alt_entry.get("under_price")

                        # Compute implied under_price when not provided.
                        # Alt markets are one-sided (only over_price from the API).
                        # We estimate under_price assuming ~5% total vig (typical
                        # for alt props). This makes deep-alt UNDER recs evaluable.
                        # implied_over_prob = 1/over_price
                        # fair_under_prob = 1 - implied_over_prob + vig_share
                        # under_price = 1 / (fair_under_prob)
                        _ALT_VIG_ASSUMPTION = 0.05  # 5% total overround
                        if alt_over_price and alt_over_price > 1 and alt_under_price is None:
                            implied_over_prob = 1.0 / alt_over_price
                            # Total overround = implied_over + implied_under = 1 + vig
                            # implied_under = 1 + vig - implied_over
                            implied_under_prob = 1.0 + _ALT_VIG_ASSUMPTION - implied_over_prob
                            if 0 < implied_under_prob < 1:
                                alt_under_price = round(1.0 / implied_under_prob, 3)

                        # For alt lines: use raw 1/price as implied prob
                        # (includes vig — no two-way market to de-vig)
                        if alt_over_price and alt_over_price > 1:
                            alt_implied_over = 1.0 / alt_over_price
                        else:
                            alt_implied_over = None

                        # If under price exists (real or computed), de-vig properly
                        if alt_over_price and alt_under_price:
                            alt_devigged = _devig_implied_prob(alt_over_price, alt_under_price)
                        elif alt_implied_over is not None:
                            alt_devigged = alt_implied_over  # raw (includes vig)
                        else:
                            alt_devigged = None

                        # Classify line depth relative to main line
                        if line > 0:
                            depth_ratio = abs(alt_line - line) / line
                            if depth_ratio < 0.20:
                                depth = "near_alt"
                            else:
                                depth = "deep_alt"
                        else:
                            depth = "deep_alt"

                        # Alt props are OVER-only markets — direction is always "over"
                        alt_direction = "over"
                        alt_gap = mean - alt_line
                        alt_abs_gap = abs(alt_gap)

                        if alt_devigged is not None:
                            alt_edge = alt_p_ov - alt_devigged
                        else:
                            alt_edge = None

                        # BE from actual alt over odds (alt props are over-only)
                        alt_be = None
                        if alt_over_price and alt_over_price > 1:
                            alt_be = round(1.0 / alt_over_price * 100, 1)

                        # Vig estimate: raw overround from the line's prices
                        alt_vig_pct = None
                        if alt_over_price and alt_under_price and alt_over_price > 1 and alt_under_price > 1:
                            alt_vig_pct = round((1/alt_over_price + 1/alt_under_price - 1) * 100, 1)
                        elif alt_over_price and alt_over_price > 1:
                            # One-sided: can't compute vig exactly, estimate ~8-12%
                            alt_vig_pct = None  # unknown, flag in output

                        # No-bet threshold for alt lines
                        if min_edge_pp > 0 and alt_edge is not None:
                            if abs(alt_edge) < min_edge_pp:
                                continue

                        gaps.append({
                            "player": cdata["display_name"],
                            "position": cdata["position"],
                            "team": cdata["team"],
                            "stat": stat_type,
                            "prop_market": alt_prop_key,
                            "consensus_mean": mean,
                            "consensus_sd": sd,
                            "n_sources": n,
                            "prop_line": alt_line,
                            "gap": round(alt_gap, 2),
                            "abs_gap": round(alt_abs_gap, 2),
                            "gap_pct": round((alt_gap / alt_line) * 100, 2) if alt_line > 0 else 0,
                            "direction": alt_direction,
                            "z_score": round(alt_abs_gap / sd, 2) if sd > 0 else round(alt_abs_gap, 2),
                            "p_over": round(alt_p_ov, 4),
                            "devigged_implied": round(alt_devigged, 4) if alt_devigged is not None else None,
                            "edge_vs_market": round(alt_edge, 4) if alt_edge is not None else None,
                            "gap_reason": gap_reason,
                            "line_depth": depth,
                            "var_deviation": round(var_deviation, 3),
                            "alt_be_pct": alt_be,
                            "alt_vig_pct": alt_vig_pct,
                            "over_price": alt_over_price,
                            "under_price": alt_under_price,
                        })

    # Sort: main lines first, then near_alt, then deep_alt; within each tier by z-score desc
    _depth_order = {"main": 0, "near_alt": 1, "deep_alt": 2}
    gaps.sort(key=lambda x: (_depth_order.get(x.get("line_depth", "main"), 9), -x["z_score"]))
    return gaps


# ══════════════════════════════════════════════════════════════════════════════
# Storage — fantasy_projections table in historical.db
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_tables():
    """Create the fantasy_projections table if it doesn't exist."""
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fantasy_projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT,
            team TEXT,
            source TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            projected_value REAL NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(season, week, player_name, source, stat_type)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fantasy_consensus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT,
            stat_type TEXT NOT NULL,
            consensus_mean REAL,
            consensus_sd REAL,
            consensus_min REAL,
            consensus_max REAL,
            n_sources INTEGER,
            prop_line REAL,
            gap REAL,
            abs_gap REAL,
            gap_pct REAL,
            direction TEXT,
            z_score REAL,
            gap_reason TEXT,
            confidence_score REAL,
            computed_at TEXT NOT NULL,
            UNIQUE(season, week, player_name, stat_type)
        )
    """)
    # Forward test predictions — one row per (player, week, stat, method)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fantasy_forward_preds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT,
            stat_type TEXT NOT NULL,
            stat_category TEXT NOT NULL,       -- 'yardage' or 'count'
            method TEXT NOT NULL,              -- 'distributional' or 'mean_direction'
            consensus_mean REAL,
            consensus_sd REAL,
            n_sources INTEGER,
            prop_line REAL NOT NULL,
            prop_over_price REAL,
            prop_under_price REAL,
            direction TEXT NOT NULL,           -- 'over' or 'under'
            p_over REAL,                      -- distributional P(over line)
            devigged_implied REAL,            -- fair implied P(over)
            edge_vs_market REAL,              -- p_over - devigged_implied
            z_score REAL,
            gap_reason TEXT,                  -- 'role_change' / 'matchup' / 'generic'
            conviction_bucket TEXT,           -- 'low' / 'mid' / 'high'
            line_depth TEXT,                  -- 'main' / 'near_alt' / 'deep_alt'
            alt_be_pct REAL,                  -- BE from actual alt odds (wider vig)
            alt_vig_pct REAL,                 -- overround on this specific line
            var_deviation REAL,               -- player CV vs positional norm (>0 = boom/bust)
            -- grading fields (filled after settlement)
            actual_value REAL,
            is_win INTEGER,                   -- 1=win, 0=loss, NULL=unsettled
            is_push INTEGER,                  -- 1=push (actual==line)
            roi REAL,                         -- net of vig: (payout-1) if win, -1 if loss
            graded_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(season, week, player_name, stat_type, method, prop_line)
        )
    """)
    conn.commit()
    conn.close()


def store_raw_projections(season: int, week: int, projections: list[dict]):
    """Store raw per-source projections to fantasy_projections table."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    inserted = 0
    for proj in projections:
        for stat_type, value in proj["stats"].items():
            if stat_type not in TRACKED_STATS:
                continue
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO fantasy_projections
                    (season, week, player_name, position, team, source,
                     stat_type, projected_value, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (season, week, proj["player_name"], proj["position"],
                      proj.get("team", ""), proj["source"],
                      stat_type, value, now))
                inserted += 1
            except Exception as exc:
                logger.warning("[Store] Failed to insert %s/%s: %s",
                               proj["player_name"], stat_type, exc)
    conn.commit()
    conn.close()
    logger.info("[Store] Inserted %d raw projections for %d week %d", inserted, season, week)
    return inserted


def store_consensus(season: int, week: int, gaps: list[dict]):
    """Store consensus + gap analysis to fantasy_consensus table."""
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    inserted = 0
    for g in gaps:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO fantasy_consensus
                (season, week, player_name, position, stat_type,
                 consensus_mean, consensus_sd, consensus_min, consensus_max,
                 n_sources, prop_line, gap, abs_gap, gap_pct, direction,
                 z_score, gap_reason, confidence_score, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (season, week, g["player"], g["position"], g["stat"],
                  g["consensus_mean"], g["consensus_sd"],
                  g.get("consensus_min"), g.get("consensus_max"),
                  g["n_sources"], g.get("prop_line"),
                  g["gap"], g.get("abs_gap"), g["gap_pct"], g["direction"],
                  g.get("z_score"), g.get("gap_reason"),
                  g.get("z_score"),  # z_score in confidence_score for compat
                  now))
            inserted += 1
        except Exception as exc:
            logger.warning("[Store] Failed to insert consensus %s/%s: %s",
                           g["player"], g["stat"], exc)
    conn.commit()
    conn.close()
    return inserted


# Stat categories for forward segmentation
_YARDAGE_STATS = {"pass_yds", "rush_yds", "rec_yds"}
_COUNT_STATS = {"pass_tds", "receptions"}

# ── Pre-committed read rules (locked 2026-09-20) ────────────────────────────
# These define how forward results are interpreted. Do NOT change after data
# starts arriving — that's the whole point of pre-committing.
#
# PRIMARY HYPOTHESIS (one test, no Bonferroni needed):
#   Yardage + distributional + high-conviction bucket beats BE-at-actual-vig
#   on realized ROI, at adequate N.
#
# SECONDARY (exploratory — any winner needs Bonferroni across examined cells):
#   - Count stats (TDs, receptions): distributional vs mean_direction
#   - Gap reason segments: role_change vs matchup vs generic
#   - Alt-line depth: main vs near-alt vs deep-alt
#   - Variance-deviation from positional norm
#
# STOPPING RULE: No verdicts until N_PRIMARY_TARGET reached.
# Weekly reports are for MONITORING only.

_N_PRIMARY_TARGET = 300       # independent player-weeks for primary test
_EST_PRIMARY_PER_WEEK = 128   # estimated high-conviction yardage preds/week
_EST_READ_WEEK = 5            # ~Week 5 (Oct 11, 2026) at current rate
_PRIMARY_FILTER = {
    "stat_category": "yardage",
    "method": "distributional",
    "conviction_bucket": "high",
}
# Number of secondary cells examined (for Bonferroni):
# 2 stat-cats × 2 methods × 3 reasons × 3 convictions × 3 line-depths = 108
# But many are empty, so effective cells ~36. Use 36 for Bonferroni.
_N_SECONDARY_CELLS = 36
_BONFERRONI_ALPHA = 0.05 / _N_SECONDARY_CELLS  # ~0.0014


def _conviction_bucket(edge_pp: float) -> str:
    """Classify edge into conviction bucket."""
    pp = abs(edge_pp) * 100 if edge_pp is not None else 0
    if pp < 5:
        return "low"
    elif pp < 15:
        return "mid"
    return "high"


def store_forward_predictions(season: int, week: int, gaps: list[dict]):
    """
    Store forward predictions from pipeline gaps for later grading.

    For yardage stats: stores distributional method (primary).
    For count stats: stores BOTH distributional AND mean_direction methods.
    Alt lines: stored with line_depth, alt_be_pct, alt_vig_pct.
    """
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    inserted = 0

    for g in gaps:
        stat = g["stat"]
        stat_cat = "yardage" if stat in _YARDAGE_STATS else "count"
        line_depth = g.get("line_depth", "main")

        # Distributional prediction (primary for yardage, secondary for counts)
        p_ov = g.get("p_over")
        devigged = g.get("devigged_implied")
        edge = g.get("edge_vs_market")
        dist_direction = "over" if (p_ov or 0) > 0.5 else "under"
        conv = _conviction_bucket(edge)

        # ── Structural suppression filters ──
        # These segments are proven structural losers (vig > edge).
        # Suppress from forward predictions to avoid polluting the signal.
        pos = g.get("position", "")
        suppress = False

        # 1. near-alt and deep-alt OVER: -18 to -26% ROI, structural vig
        if line_depth in ("near_alt", "deep_alt") and dist_direction == "over":
            suppress = True

        # 2. rush OVER RB on main line: gate to high-conviction only
        #    (consensus overestimates RB rushing +8.5 yds, 45% WR overall)
        if (line_depth == "main" and stat == "rush_yds"
                and dist_direction == "over" and pos == "RB"
                and conv != "high"):
            suppress = True

        # 3. deep-alt UNDER for rec/pass: -3.4 to -3.7% ROI, vig eats edge
        if line_depth == "deep_alt" and dist_direction == "under" and stat in ("rec_yds", "pass_yds"):
            suppress = True

        if suppress:
            continue

        try:
            conn.execute("""
                INSERT OR REPLACE INTO fantasy_forward_preds
                (season, week, player_name, position, stat_type, stat_category,
                 method, consensus_mean, consensus_sd, n_sources,
                 prop_line, prop_over_price, prop_under_price,
                 direction, p_over, devigged_implied, edge_vs_market,
                 z_score, gap_reason, conviction_bucket,
                 line_depth, alt_be_pct, alt_vig_pct, var_deviation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (season, week, g["player"], g["position"], stat, stat_cat,
                  "distributional", g["consensus_mean"], g["consensus_sd"],
                  g["n_sources"], g["prop_line"],
                  g.get("over_price"), g.get("under_price"),
                  dist_direction, p_ov, devigged, edge,
                  g.get("z_score"), g.get("gap_reason"), conv,
                  line_depth, g.get("alt_be_pct"), g.get("alt_vig_pct"),
                  g.get("var_deviation"), now))
            inserted += 1
        except Exception as exc:
            logger.warning("[Forward] Failed to store dist pred %s/%s/%.1f: %s",
                           g["player"], stat, g.get("prop_line", 0), exc)

        # For count stats on MAIN lines, ALSO store mean-direction prediction
        if stat_cat == "count" and line_depth == "main":
            mean_dir = g["direction"]
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO fantasy_forward_preds
                    (season, week, player_name, position, stat_type, stat_category,
                     method, consensus_mean, consensus_sd, n_sources,
                     prop_line, prop_over_price, prop_under_price,
                     direction, p_over, devigged_implied, edge_vs_market,
                     z_score, gap_reason, conviction_bucket,
                     line_depth, alt_be_pct, alt_vig_pct, var_deviation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (season, week, g["player"], g["position"], stat, stat_cat,
                      "mean_direction", g["consensus_mean"], g["consensus_sd"],
                      g["n_sources"], g["prop_line"],
                      g.get("over_price"), g.get("under_price"),
                      mean_dir, p_ov, devigged, edge,
                      g.get("z_score"), g.get("gap_reason"), conv,
                      "main", None, None, g.get("var_deviation"), now))
                inserted += 1
            except Exception as exc:
                logger.warning("[Forward] Failed to store mean pred %s/%s: %s",
                               g["player"], stat, exc)

    conn.commit()
    conn.close()
    logger.info("[Forward] Stored %d predictions for %d week %d", inserted, season, week)
    return inserted


def grade_forward_predictions(season: int, week: int) -> dict:
    """
    Grade stored forward predictions against actual outcomes.

    Fetches actual stats from Sleeper, matches to stored predictions,
    computes win/loss/push and ROI for each prediction.

    Returns segmented results: by stat_category, method, gap_reason, conviction.
    """
    _ensure_tables()

    # Fetch actuals
    actuals = _fetch_sleeper_actuals(season, week)
    if not actuals:
        return {"error": f"No actuals available for {season} week {week}"}

    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Get ungraded predictions for this week
    preds = conn.execute("""
        SELECT id, player_name, stat_type, stat_category, method,
               prop_line, prop_over_price, prop_under_price,
               direction, p_over, devigged_implied, edge_vs_market,
               gap_reason, conviction_bucket, line_depth, var_deviation
        FROM fantasy_forward_preds
        WHERE season = ? AND week = ? AND is_win IS NULL
    """, (season, week)).fetchall()

    if not preds:
        conn.close()
        return {"error": f"No ungraded predictions for {season} week {week}"}

    graded = 0
    results = []

    for pred in preds:
        norm_name = _normalize_player_name(pred["player_name"])
        actual_stats = actuals.get(norm_name)
        if not actual_stats:
            continue

        actual_val = actual_stats.get(pred["stat_type"])
        if actual_val is None:
            continue

        line = pred["prop_line"]
        direction = pred["direction"]

        # Determine outcome
        is_push = (actual_val == line)
        if is_push:
            is_win = None  # pushes are excluded from WR
            roi = 0.0
        elif direction == "over":
            is_win = 1 if actual_val > line else 0
        else:  # under
            is_win = 1 if actual_val < line else 0

        # ROI: use the price for the bet direction
        if is_push:
            roi = 0.0
        elif is_win:
            price = pred["prop_over_price"] if direction == "over" else pred["prop_under_price"]
            roi = (price - 1.0) if price and price > 1 else 0.0  # decimal odds payout
        else:
            roi = -1.0  # lost the unit

        # Update the prediction row
        conn.execute("""
            UPDATE fantasy_forward_preds
            SET actual_value = ?, is_win = ?, is_push = ?, roi = ?, graded_at = ?
            WHERE id = ?
        """, (actual_val, is_win, 1 if is_push else 0, round(roi, 4), now, pred["id"]))
        graded += 1

        results.append({
            "player": pred["player_name"],
            "stat": pred["stat_type"],
            "stat_category": pred["stat_category"],
            "method": pred["method"],
            "direction": direction,
            "line": line,
            "actual": actual_val,
            "is_win": is_win,
            "is_push": is_push,
            "roi": roi,
            "gap_reason": pred["gap_reason"],
            "conviction": pred["conviction_bucket"],
            "line_depth": pred["line_depth"],
            "var_deviation": pred["var_deviation"],
            "edge_vs_market": pred["edge_vs_market"],
        })

    conn.commit()
    conn.close()

    if not results:
        return {"graded": 0, "error": "No matchable actuals"}

    # ── Segment results ──
    resolved = [r for r in results if not r["is_push"] and r["is_win"] is not None]
    pushes = [r for r in results if r["is_push"]]

    _N_MIN_READ = 30

    def _segment_summary(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "wins": 0, "wr": 0, "roi": 0, "readable": False}
        wins = sum(1 for r in subset if r["is_win"] == 1)
        total_roi = sum(r["roi"] for r in subset)
        n = len(subset)
        return {
            "n": n,
            "wins": wins,
            "wr": round(wins / n * 100, 1),
            "roi_pct": round(total_roi / n * 100, 1),
            "readable": n >= _N_MIN_READ,
        }

    summary = {
        "season": season,
        "week": week,
        "total_predictions": len(results),
        "graded": graded,
        "resolved": len(resolved),
        "pushes": len(pushes),
        "overall": _segment_summary(resolved),
        "by_stat_category": {
            "yardage": _segment_summary([r for r in resolved if r["stat_category"] == "yardage"]),
            "count": _segment_summary([r for r in resolved if r["stat_category"] == "count"]),
        },
        "by_method": {
            "distributional": _segment_summary([r for r in resolved if r["method"] == "distributional"]),
            "mean_direction": _segment_summary([r for r in resolved if r["method"] == "mean_direction"]),
        },
        "by_gap_reason": {
            reason: _segment_summary([r for r in resolved if r["gap_reason"] == reason])
            for reason in ["role_change", "matchup", "generic"]
        },
        "by_conviction": {
            bucket: _segment_summary([r for r in resolved if r["conviction"] == bucket])
            for bucket in ["low", "mid", "high"]
        },
        "by_line_depth": {
            depth: _segment_summary([r for r in resolved if r.get("line_depth") == depth])
            for depth in ["main", "near_alt", "deep_alt"]
        },
    }

    # Print report
    print(f"\n[Forward] {season} Week {week} — {len(results)} predictions, "
          f"{len(resolved)} resolved, {len(pushes)} pushes")
    ov = summary["overall"]
    print(f"  Overall: {ov['n']} resolved, {ov['wr']}% WR, {ov['roi_pct']}% ROI")

    def _print_grade_seg(name: str, data: dict, w: int = 12):
        if data["n"] <= 0:
            return
        tag = "" if data.get("readable", True) else "  << INSUFFICIENT"
        print(f"    {name:<{w}} N={data['n']:>4}, WR={data['wr']:>5.1f}%, ROI={data['roi_pct']:>6.1f}%{tag}")

    print(f"\n  By stat category:")
    for cat, data in summary["by_stat_category"].items():
        _print_grade_seg(cat, data)

    print(f"\n  By method:")
    for method, data in summary["by_method"].items():
        _print_grade_seg(method, data, 18)

    print(f"\n  By gap reason:")
    for reason, data in summary["by_gap_reason"].items():
        _print_grade_seg(reason, data, 14)

    print(f"\n  By conviction bucket:")
    for bucket, data in summary["by_conviction"].items():
        _print_grade_seg(bucket, data, 6)

    print(f"\n  By line depth:")
    for depth, data in summary["by_line_depth"].items():
        _print_grade_seg(depth, data, 10)

    return summary


def forward_report(season: int, weeks: list[int] | None = None) -> dict:
    """
    Aggregate forward test results across multiple weeks.

    Applies pre-committed read rules:
    - PRIMARY test: yardage + distributional + high-conviction, judged on ROI vs BE
    - SECONDARY segments: exploratory, Bonferroni-corrected, labeled as such
    - ACCUMULATING label until N_PRIMARY_TARGET reached
    """
    _ensure_tables()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT player_name, stat_type, stat_category, method,
               direction, prop_line, prop_over_price, prop_under_price,
               actual_value, is_win, is_push, roi,
               gap_reason, conviction_bucket, edge_vs_market,
               line_depth, alt_be_pct, alt_vig_pct, var_deviation,
               season, week
        FROM fantasy_forward_preds
        WHERE season = ? AND is_win IS NOT NULL
    """
    params: list = [season]

    if weeks:
        placeholders = ",".join("?" * len(weeks))
        query += f" AND week IN ({placeholders})"
        params.extend(weeks)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return {"error": "No graded forward predictions found"}

    results = [dict(r) for r in rows]
    resolved = [r for r in results if not r["is_push"]]
    pushes = [r for r in results if r["is_push"]]

    _N_MIN_READABLE = 30  # segments below this are not actionable

    def _seg(subset, label: str = "secondary"):
        if not subset:
            return {"n": 0, "wins": 0, "wr": 0, "roi_pct": 0,
                    "status": "no_data", "label": label, "readable": False}
        wins = sum(1 for r in subset if r["is_win"] == 1)
        total_roi = sum(r["roi"] or 0 for r in subset)
        n = len(subset)
        readable = n >= _N_MIN_READABLE
        return {
            "n": n,
            "wins": wins,
            "wr": round(wins / n * 100, 1),
            "roi_pct": round(total_roi / n * 100, 1),
            "label": label,
            "readable": readable,
            "status": "READABLE" if readable else "INSUFFICIENT",
        }

    weeks_covered = sorted(set(r["week"] for r in results))

    # ── Primary test: yardage + distributional + high-conviction ──
    primary_subset = [
        r for r in resolved
        if r["stat_category"] == _PRIMARY_FILTER["stat_category"]
        and r["method"] == _PRIMARY_FILTER["method"]
        and r["conviction_bucket"] == _PRIMARY_FILTER["conviction_bucket"]
    ]
    primary = _seg(primary_subset, label="PRIMARY")
    primary_n = primary["n"]
    primary_readable = primary_n >= _N_PRIMARY_TARGET

    if primary_readable:
        primary["status"] = "READABLE"
    else:
        primary["status"] = "ACCUMULATING"
        primary["n_remaining"] = _N_PRIMARY_TARGET - primary_n
        primary["n_target"] = _N_PRIMARY_TARGET
        primary["pct_complete"] = round(primary_n / _N_PRIMARY_TARGET * 100, 1)
        est_weeks_left = max(1, primary["n_remaining"] // max(_EST_PRIMARY_PER_WEEK, 1))
        primary["est_weeks_remaining"] = est_weeks_left

    # Compute actual BE from vig for primary subset
    primary_vig = [r for r in primary_subset
                   if r.get("prop_over_price") and r.get("prop_under_price")]
    if primary_vig:
        avg_be = sum(
            1.0 / (r["prop_over_price"] if r["direction"] == "over" else r["prop_under_price"])
            for r in primary_vig
        ) / len(primary_vig)
        primary["be_from_vig"] = round(avg_be * 100, 1)

    # ── Secondary segments (all exploratory, Bonferroni-labeled) ──
    sec_label = f"SECONDARY (exploratory, Bonferroni {_N_SECONDARY_CELLS} cells)"

    summary = {
        "season": season,
        "weeks": weeks_covered,
        "total": len(results),
        "resolved": len(resolved),
        "pushes": len(pushes),
        "read_rules": {
            "primary_hypothesis": "yardage + distributional + high-conviction beats BE on ROI",
            "n_target": _N_PRIMARY_TARGET,
            "bonferroni_cells": _N_SECONDARY_CELLS,
            "bonferroni_alpha": round(_BONFERRONI_ALPHA, 4),
            "est_read_week": _EST_READ_WEEK,
        },
        "primary": primary,
        "overall": _seg(resolved, label="overall"),
        "secondary": {
            "by_stat_category": {
                "yardage": _seg([r for r in resolved if r["stat_category"] == "yardage"], sec_label),
                "count": _seg([r for r in resolved if r["stat_category"] == "count"], sec_label),
            },
            "by_method": {
                "distributional": _seg([r for r in resolved if r["method"] == "distributional"], sec_label),
                "mean_direction": _seg([r for r in resolved if r["method"] == "mean_direction"], sec_label),
            },
            "by_stat": {
                stat: _seg([r for r in resolved if r["stat_type"] == stat], sec_label)
                for stat in TRACKED_STATS
            },
            "by_gap_reason": {
                reason: _seg([r for r in resolved if r["gap_reason"] == reason], sec_label)
                for reason in ["role_change", "matchup", "generic"]
            },
            "by_conviction": {
                bucket: _seg([r for r in resolved if r["conviction_bucket"] == bucket], sec_label)
                for bucket in ["low", "mid", "high"]
            },
            "counts_by_method": {
                method: _seg([r for r in resolved
                             if r["stat_category"] == "count" and r["method"] == method], sec_label)
                for method in ["distributional", "mean_direction"]
            },
            "by_line_depth": {
                depth: _seg([r for r in resolved if r.get("line_depth") == depth], sec_label)
                for depth in ["main", "near_alt", "deep_alt"]
            },
            "by_var_deviation": {
                "low_var": _seg([r for r in resolved
                                if r.get("var_deviation") is not None
                                and r["var_deviation"] < -0.15], sec_label),
                "norm_var": _seg([r for r in resolved
                                 if r.get("var_deviation") is not None
                                 and -0.15 <= r["var_deviation"] <= 0.15], sec_label),
                "high_var": _seg([r for r in resolved
                                  if r.get("var_deviation") is not None
                                  and r["var_deviation"] > 0.15], sec_label),
            },
        },
    }

    # Compute overall BE from vig
    vig_resolved = [r for r in resolved if r.get("prop_over_price") and r.get("prop_under_price")]
    if vig_resolved:
        avg_be = sum(
            1.0 / (r["prop_over_price"] if r["direction"] == "over" else r["prop_under_price"])
            for r in vig_resolved
        ) / len(vig_resolved)
        summary["overall"]["be_from_vig"] = round(avg_be * 100, 1)

    # ── Print report ──
    print(f"\n{'='*74}")
    print(f"  FANTASY PROJECTIONS — FORWARD TEST REPORT")
    print(f"  {season} Weeks {weeks_covered}")
    print(f"{'='*74}")

    # Primary hypothesis
    print(f"\n  ┌─ PRIMARY HYPOTHESIS (pre-registered, one test) ─────────────────┐")
    print(f"  │ Yardage + distributional + high-conviction beats BE on ROI      │")
    print(f"  │ N-target: {_N_PRIMARY_TARGET}    N-current: {primary_n:<5}                             │")
    if primary_readable:
        print(f"  │ STATUS: *** READABLE ***                                        │")
        print(f"  │ WR={primary['wr']}%  ROI={primary['roi_pct']}%  "
              f"BE={primary.get('be_from_vig', '?')}%                       │")
        beats_be = (primary.get("be_from_vig") and primary["wr"] > primary["be_from_vig"])
        verdict = "PASS — edge is real" if beats_be else "FAIL — does not beat BE"
        print(f"  │ VERDICT: {verdict:<54}│")
    else:
        pct = primary.get("pct_complete", 0)
        bar_len = int(pct / 100 * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  │ STATUS: ACCUMULATING — not readable                             │")
        print(f"  │ [{bar}] {pct:.0f}%           │")
        print(f"  │ ~{primary.get('est_weeks_remaining', '?')} weeks to readable "
              f"(est. Week {_EST_READ_WEEK})                       │")
        if primary_n > 0:
            print(f"  │ Interim (DO NOT ACT): WR={primary['wr']}%  ROI={primary['roi_pct']}%"
                  f"{'':>18}│")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  Total: {len(results)} predictions, {len(resolved)} resolved, {len(pushes)} pushes")
    ov = summary["overall"]
    print(f"  Overall: WR={ov['wr']}%, ROI={ov['roi_pct']}%")
    if ov.get("be_from_vig"):
        print(f"  Actual BE from vig: {ov['be_from_vig']}%")

    # Secondary segments (all labeled exploratory)
    print(f"\n  --- SECONDARY SEGMENTS (exploratory, Bonferroni/{_N_SECONDARY_CELLS}) ---")
    print(f"  Any winner here needs p < {_BONFERRONI_ALPHA:.4f} to be credible.")
    print(f"  Segments with N < {_N_MIN_READABLE} are labeled INSUFFICIENT — not actionable.")

    def _print_seg(name: str, d: dict, name_width: int = 12):
        if d["n"] <= 0:
            return
        tag = "" if d["readable"] else "  << INSUFFICIENT"
        print(f"    {name:<{name_width}} N={d['n']:>4}, WR={d['wr']:>5.1f}%, "
              f"ROI={d['roi_pct']:>6.1f}%{tag}")

    sec = summary["secondary"]
    print(f"\n  Stat category:")
    for cat in ["yardage", "count"]:
        _print_seg(cat, sec["by_stat_category"][cat])

    print(f"\n  Counts: distributional vs mean_direction:")
    for method in ["distributional", "mean_direction"]:
        _print_seg(method, sec["counts_by_method"][method], 18)

    print(f"\n  Gap reason:")
    for reason in ["role_change", "matchup", "generic"]:
        _print_seg(reason, sec["by_gap_reason"][reason], 14)

    print(f"\n  Conviction bucket:")
    for bucket in ["low", "mid", "high"]:
        _print_seg(bucket, sec["by_conviction"][bucket], 6)

    print(f"\n  Line depth (tail-vs-main test):")
    for depth in ["main", "near_alt", "deep_alt"]:
        _print_seg(depth, sec["by_line_depth"][depth], 10)

    print(f"\n  Variance deviation (boom/bust vs steady):")
    vd_labels = {"low_var": "steady (<-15%)", "norm_var": "norm (+-15%)", "high_var": "boom/bust (>+15%)"}
    for vd in ["low_var", "norm_var", "high_var"]:
        _print_seg(vd_labels[vd], sec["by_var_deviation"][vd], 22)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Distributional framework — P(over line) from fitted distributions
# ══════════════════════════════════════════════════════════════════════════════
#
# Instead of comparing mean projection vs prop line (directional only),
# model the player's OUTCOME DISTRIBUTION and compute P(actual > line).
# Compare that to the de-vigged implied probability from the odds.
#
# Distribution choices by stat type:
#   - TDs (pass_tds, rush_tds, rec_tds):  NegBin (overdispersed counts)
#   - Receptions:                         NegBin (overdispersed counts)
#   - Yardage (pass_yds, rush_yds, rec_yds): Lognormal (positive, right-skewed)
#
# Variance estimation: per-player from game logs with shrinkage toward
# position/usage-tier averages when individual N is small.

# Stat type → distribution family
_STAT_DISTRIBUTION = {
    "pass_yds":   "lognormal",
    "rush_yds":   "lognormal",
    "rec_yds":    "lognormal",
    "pass_tds":   "negbin",
    "receptions": "negbin",
}

# Minimum games for per-player variance (below this, shrink heavily)
_MIN_GAMES_INDIVIDUAL = 4
# Shrinkage weight: at exactly _MIN_GAMES_INDIVIDUAL games, the player's
# own variance gets this weight vs the tier prior. Ramps linearly to 1.0
# at _FULL_TRUST_GAMES.
_FULL_TRUST_GAMES = 12


def _estimate_player_variance(game_values: list[float], tier_mean: float,
                               tier_var: float, dist_type: str) -> dict:
    """
    Estimate a player's outcome distribution parameters with shrinkage.

    Args:
        game_values: list of actual stat values from recent games
        tier_mean: position/usage-tier mean for this stat
        tier_var: position/usage-tier variance for this stat
        dist_type: "lognormal" or "negbin"

    Returns dict with distribution parameters:
        lognormal: {"mu": float, "sigma": float}
        negbin:    {"mu": float, "r": float}  (r = dispersion)
    """
    n = len(game_values)

    if n == 0:
        # No individual data — use tier prior entirely
        if dist_type == "lognormal":
            mu_ln, sigma_ln = _lognormal_params(tier_mean, tier_var)
            return {"mu": mu_ln, "sigma": sigma_ln, "shrinkage": 1.0}
        else:
            mu, r = _negbin_params(tier_mean, tier_var)
            return {"mu": mu, "r": r, "shrinkage": 1.0}

    # Compute player-level stats
    player_mean = sum(game_values) / n
    if n >= 2:
        player_var = sum((v - player_mean) ** 2 for v in game_values) / (n - 1)
    else:
        player_var = tier_var  # single game, fall back to tier

    # Shrinkage weight: linear ramp from _MIN_GAMES to _FULL_TRUST_GAMES
    if n >= _FULL_TRUST_GAMES:
        w_player = 1.0
    elif n >= _MIN_GAMES_INDIVIDUAL:
        w_player = (n - _MIN_GAMES_INDIVIDUAL + 1) / (_FULL_TRUST_GAMES - _MIN_GAMES_INDIVIDUAL + 1)
    else:
        w_player = n / (_FULL_TRUST_GAMES + 1)  # very low weight

    w_tier = 1.0 - w_player

    # Shrunk estimates
    shrunk_mean = w_player * player_mean + w_tier * tier_mean
    shrunk_var = w_player * player_var + w_tier * tier_var
    # Ensure variance is positive and at least a small fraction of mean^2
    shrunk_var = max(shrunk_var, 0.01 * (shrunk_mean ** 2) if shrunk_mean > 0 else 0.1)

    if dist_type == "lognormal":
        mu_ln, sigma_ln = _lognormal_params(max(shrunk_mean, 0.1), shrunk_var)
        return {"mu": mu_ln, "sigma": sigma_ln, "shrinkage": round(w_tier, 3)}
    else:
        mu, r = _negbin_params(max(shrunk_mean, 0.01), shrunk_var)
        return {"mu": mu, "r": r, "shrinkage": round(w_tier, 3)}


def _lognormal_params(mean: float, var: float) -> tuple[float, float]:
    """Convert (mean, variance) to lognormal (mu, sigma) parameters.

    For lognormal: E[X] = exp(mu + sigma^2/2), Var[X] = (exp(sigma^2)-1)*exp(2*mu+sigma^2)
    Solving: sigma^2 = ln(1 + var/mean^2), mu = ln(mean) - sigma^2/2
    """
    from math import log
    if mean <= 0:
        return (0.0, 1.0)
    sigma_sq = log(1 + var / (mean ** 2))
    sigma = sqrt(sigma_sq)
    mu = log(mean) - sigma_sq / 2
    return (mu, sigma)


def _negbin_params(mean: float, var: float) -> tuple[float, float]:
    """Convert (mean, variance) to NegBin (mu, r) parameters.

    NegBin parameterized as: P(X=k) with mean=mu, variance=mu + mu^2/r.
    So r = mu^2 / (var - mu) when var > mu (overdispersed).
    If var <= mu (underdispersed), fall back to Poisson (r → ∞, use r=1000).
    """
    if mean <= 0:
        return (0.01, 1.0)
    if var <= mean:
        # Underdispersed or equidispersed — approximate as Poisson
        return (mean, 1000.0)
    r = (mean ** 2) / (var - mean)
    r = max(r, 0.1)  # floor to prevent degenerate distributions
    return (mean, r)


def p_over_line(line: float, mean: float, dist_params: dict,
                dist_type: str) -> float:
    """
    Compute P(actual > line) from a fitted distribution.

    Args:
        line: the prop line value
        mean: consensus projection mean (used as distribution center)
        dist_params: {"mu": float, "sigma": float} or {"mu": float, "r": float}
        dist_type: "lognormal" or "negbin"

    Returns: probability that the actual outcome exceeds the line.
    """
    from scipy import stats as sp_stats

    if dist_type == "lognormal":
        sigma = dist_params["sigma"]
        # Center the distribution on the consensus mean, keeping the estimated
        # sigma (variance structure from game logs + shrinkage).
        sigma_sq = sigma ** 2
        from math import log, exp
        mu_adj = log(max(mean, 0.1)) - sigma_sq / 2
        scale = exp(mu_adj)
        return float(1.0 - sp_stats.lognorm.cdf(line, s=sigma, scale=scale))

    elif dist_type == "negbin":
        r = dist_params["r"]
        # Center on consensus mean
        mu = max(mean, 0.01)
        if r >= 500:
            # Poisson approximation
            return float(1.0 - sp_stats.poisson.cdf(line, mu=mu))
        # scipy's nbinom: P(X=k) with params (n=r, p=r/(r+mu))
        p = r / (r + mu)
        # P(X > line) — for integer lines, P(X > line) = 1 - P(X <= floor(line))
        # For half-integer lines (e.g. 2.5), P(X > 2.5) = P(X >= 3) = 1 - P(X <= 2)
        from math import floor
        k = floor(line)  # P(X > line) = 1 - P(X <= k) for line in (k, k+1)
        if line == int(line):
            # Exact integer line: over means strictly > line
            return float(1.0 - sp_stats.nbinom.cdf(int(line), n=r, p=p))
        else:
            return float(1.0 - sp_stats.nbinom.cdf(k, n=r, p=p))

    return 0.5  # fallback


def _devig_implied_prob(over_price: float | None, under_price: float | None) -> float | None:
    """De-vig two-way odds to get fair implied probability for over.

    Uses multiplicative method: p_fair = (1/over) / (1/over + 1/under).
    Prices are decimal odds (e.g. 1.91 for -110).
    """
    if not over_price or not under_price or over_price <= 1 or under_price <= 1:
        return None
    imp_over = 1.0 / over_price
    imp_under = 1.0 / under_price
    total = imp_over + imp_under
    if total <= 0:
        return None
    return imp_over / total


def _build_tier_priors(all_actuals: dict[str, dict[str, list[float]]],
                       player_positions: dict[str, str]) -> dict[str, dict[str, dict]]:
    """
    Build position-tier variance priors from all player actuals.

    Args:
        all_actuals: {norm_name: {stat: [game_values...]}}
        player_positions: {norm_name: position}

    Returns: {position: {stat: {"mean": float, "var": float, "n_players": int}}}
    """
    from collections import defaultdict

    tier_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for norm_name, stats in all_actuals.items():
        pos = player_positions.get(norm_name, "")
        if not pos:
            continue
        for stat, values in stats.items():
            if values:
                # Use the player's mean as a single observation for the tier
                tier_data[pos][stat].extend(values)

    tier_priors = {}
    for pos, stats in tier_data.items():
        tier_priors[pos] = {}
        for stat, values in stats.items():
            if len(values) < 3:
                continue
            m = sum(values) / len(values)
            v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
            tier_priors[pos][stat] = {"mean": m, "var": max(v, 0.1), "n_players": len(values)}

    return tier_priors


# ══════════════════════════════════════════════════════════════════════════════
# Historical backtest — do fantasy projections beat the market?
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_sleeper_actuals(season: int, week: int) -> dict:
    """Fetch actual stats from Sleeper for a completed week.

    Returns {normalized_name: {"pass_yds": float, "rush_yds": float, ...}}
    """
    url = f"https://api.sleeper.com/stats/nfl/{season}/{week}?season_type=regular"
    raw = _get(url, timeout=20)
    if not raw or not isinstance(raw, list):
        return {}

    player_db = _sleeper_player_db()
    actuals = {}

    for entry in raw:
        pid = entry.get("player_id", "")
        stats_raw = entry.get("stats") or {}
        pinfo = player_db.get(str(pid), {})
        full_name = pinfo.get("full_name", "")
        if not full_name or not stats_raw:
            continue

        position = pinfo.get("position", "")
        if position not in ("QB", "RB", "WR", "TE"):
            continue

        normalized = {}
        for sleeper_key, canonical_key in _SLEEPER_FIELD_MAP.items():
            val = stats_raw.get(sleeper_key)
            if val is not None:
                normalized[canonical_key] = float(val)

        if normalized:
            actuals[_normalize_player_name(full_name)] = normalized

    return actuals


def backtest_projections(season: int, weeks: list[int] | None = None,
                         min_gap_pct: float = 0.0) -> dict:
    """
    Backtest: accuracy and directional performance of projection-based approach.

    Computes both:
      A) Old metric: mean-vs-line directional accuracy (for comparison)
      B) Distributional: P(over line) from fitted distributions with shrinkage

    Uses Sleeper projections vs Sleeper actuals (both free).
    Without stored prop lines, uses the player's OWN season median per stat
    as the proxy line (the hardest benchmark — maximally efficient).

    Returns:
        {"total_bets": int, "wins": int, "losses": int, "wr": float,
         "by_stat": {stat: {...}},
         "by_position": {...}}
    """
    if weeks is None:
        weeks = list(range(1, 19))  # NFL regular season

    # ── Phase 1: Collect all projections + actuals across weeks ──
    week_data: dict[int, dict] = {}   # week → {"projs": [...], "actuals": {...}}
    all_actuals_by_player: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    player_positions: dict[str, str] = {}

    print(f"[Backtest] Fetching {len(weeks)} weeks of data...")
    for week in weeks:
        projs = fetch_sleeper_projections(season, week)
        if not projs:
            continue
        time.sleep(0.3)

        actuals = _fetch_sleeper_actuals(season, week)
        if not actuals:
            continue
        time.sleep(0.3)

        week_data[week] = {"projs": projs, "actuals": actuals}

        # Accumulate game-level actuals for variance estimation
        for norm_name, stats in actuals.items():
            for stat, val in stats.items():
                if stat in TRACKED_STATS:
                    all_actuals_by_player[norm_name][stat].append(val)

        # Collect positions from projections
        for proj in projs:
            norm = _normalize_player_name(proj["player_name"])
            player_positions[norm] = proj["position"]

    if not week_data:
        return {"total_bets": 0, "error": "no data — need completed weeks"}

    # ── Phase 2: Build tier priors + player medians (proxy lines) ──
    tier_priors = _build_tier_priors(dict(all_actuals_by_player), player_positions)

    # Player season median per stat = proxy line (hardest benchmark)
    player_medians: dict[str, dict[str, float]] = {}
    for norm_name, stats in all_actuals_by_player.items():
        player_medians[norm_name] = {}
        for stat, values in stats.items():
            if len(values) >= 3:
                sv = sorted(values)
                mid = len(sv) // 2
                player_medians[norm_name][stat] = sv[mid] if len(sv) % 2 == 1 else (sv[mid-1] + sv[mid]) / 2

    # ── Phase 3: Evaluate each week's projections ──
    results = []
    for week in sorted(week_data.keys()):
        wd = week_data[week]
        for proj in wd["projs"]:
            norm_name = _normalize_player_name(proj["player_name"])
            if norm_name not in wd["actuals"]:
                continue
            actual = wd["actuals"][norm_name]
            pos = proj["position"]

            for stat_type, projected in proj["stats"].items():
                if stat_type not in TRACKED_STATS:
                    continue
                actual_val = actual.get(stat_type)
                if actual_val is None:
                    continue

                # Proxy line = player's season median
                median_line = (player_medians.get(norm_name) or {}).get(stat_type)
                if median_line is None or median_line <= 0:
                    continue

                dist_type = _STAT_DISTRIBUTION.get(stat_type, "lognormal")

                # Get player's game log UP TO this week (causal — no lookahead)
                all_vals = all_actuals_by_player.get(norm_name, {}).get(stat_type, [])
                # For a proper causal test we'd need per-week ordering,
                # but Sleeper actuals don't guarantee order. Use all games
                # as variance estimate (slight lookahead in variance only,
                # NOT in the line — line is the full-season median).
                # This is the same lookahead level as the original backtest.

                # Tier prior for this position + stat
                tier = (tier_priors.get(pos) or {}).get(stat_type,
                        {"mean": projected, "var": max(projected * 0.5, 1.0) ** 2})

                # Fit distribution
                params = _estimate_player_variance(
                    all_vals, tier["mean"], tier["var"], dist_type)

                # P(over median line) from distribution centered on projection
                p_over = p_over_line(median_line, projected, params, dist_type)

                # Push detection: actual exactly equals the line
                is_push = (actual_val == median_line)

                # Old metric: mean-vs-line direction
                mean_direction = "over" if projected > median_line else "under"
                mean_correct = (projected > median_line and actual_val > median_line) or \
                               (projected < median_line and actual_val < median_line)

                # Distributional metric: bet the side with P > 0.50
                dist_direction = "over" if p_over > 0.5 else "under"
                dist_correct = (p_over > 0.5 and actual_val > median_line) or \
                               (p_over < 0.5 and actual_val < median_line)

                # Edge: |P(over) - 0.50| — how much the model disagrees with 50/50
                edge = abs(p_over - 0.5)

                results.append({
                    "season": season,
                    "week": week,
                    "player": proj["player_name"],
                    "position": pos,
                    "stat": stat_type,
                    "projected": projected,
                    "actual": actual_val,
                    "median_line": median_line,
                    "p_over": round(p_over, 4),
                    "dist_direction": dist_direction,
                    "dist_correct": dist_correct,
                    "mean_direction": mean_direction,
                    "mean_correct": mean_correct,
                    "is_push": is_push,
                    "edge": round(edge, 4),
                    "shrinkage": params.get("shrinkage", 0),
                })

    if not results:
        return {"total_bets": 0, "error": "no matchable data"}

    # ── Phase 4: Aggregate results (excluding pushes from WR) ──
    n_total = len(results)
    n_pushes = sum(1 for r in results if r["is_push"])
    resolved = [r for r in results if not r["is_push"]]
    n = len(resolved)

    dist_wins = sum(1 for r in resolved if r["dist_correct"])
    mean_wins = sum(1 for r in resolved if r["mean_correct"])

    by_stat = defaultdict(lambda: {"n": 0, "pushes": 0, "dist_wins": 0, "mean_wins": 0})
    by_position = defaultdict(lambda: {"n": 0, "pushes": 0, "dist_wins": 0, "mean_wins": 0})
    by_edge_bucket = defaultdict(lambda: {"n": 0, "dist_wins": 0})

    for r in results:
        # By stat
        st = by_stat[r["stat"]]
        st["n"] += 1
        st["pushes"] += int(r["is_push"])
        if not r["is_push"]:
            st["dist_wins"] += int(r["dist_correct"])
            st["mean_wins"] += int(r["mean_correct"])

        # By position
        ps = by_position[r["position"]]
        ps["n"] += 1
        ps["pushes"] += int(r["is_push"])
        if not r["is_push"]:
            ps["dist_wins"] += int(r["dist_correct"])
            ps["mean_wins"] += int(r["mean_correct"])

        # By edge bucket (resolved only)
        if not r["is_push"]:
            if r["edge"] < 0.05:
                bucket = "low (<5pp)"
            elif r["edge"] < 0.15:
                bucket = "mid (5-15pp)"
            else:
                bucket = "high (>15pp)"
            eb = by_edge_bucket[bucket]
            eb["n"] += 1
            eb["dist_wins"] += int(r["dist_correct"])

    # Compute WR for each bucket (excluding pushes)
    for bucket_dict in [by_stat, by_position, by_edge_bucket]:
        for key, data in bucket_dict.items():
            resolved_n = data["n"] - data.get("pushes", 0)
            data["dist_wr"] = round(data["dist_wins"] / resolved_n * 100, 1) if resolved_n > 0 else 0
            if "mean_wins" in data:
                data["mean_wr"] = round(data["mean_wins"] / resolved_n * 100, 1) if resolved_n > 0 else 0
            data["push_rate"] = round(data.get("pushes", 0) / data["n"] * 100, 1) if data["n"] > 0 else 0

    # Over/under split — check for systematic over-bias (resolved only)
    over_bets = [r for r in resolved if r["dist_direction"] == "over"]
    under_bets = [r for r in resolved if r["dist_direction"] == "under"]
    over_correct = sum(1 for r in over_bets if r["dist_correct"])
    under_correct = sum(1 for r in under_bets if r["dist_correct"])

    summary = {
        "season": season,
        "weeks_tested": len(week_data),
        "total_observations": n_total,
        "pushes": n_pushes,
        "push_rate_pct": round(n_pushes / n_total * 100, 1),
        "resolved_observations": n,
        "distributional_wr": round(dist_wins / n * 100, 1) if n > 0 else 0,
        "mean_direction_wr": round(mean_wins / n * 100, 1) if n > 0 else 0,
        "over_under_split": {
            "over_n": len(over_bets),
            "over_wr": round(over_correct / len(over_bets) * 100, 1) if over_bets else 0,
            "under_n": len(under_bets),
            "under_wr": round(under_correct / len(under_bets) * 100, 1) if under_bets else 0,
        },
        "by_stat": dict(by_stat),
        "by_position": dict(by_position),
        "by_edge_bucket": dict(by_edge_bucket),
    }

    # Print summary
    print(f"\n[Backtest] {season} — {len(week_data)} weeks")
    print(f"  Total: {n_total} observations, {n_pushes} pushes ({summary['push_rate_pct']}%), {n} resolved")
    print(f"  Distributional WR: {summary['distributional_wr']}% (excl pushes)")
    print(f"  Mean-direction WR: {summary['mean_direction_wr']}% (excl pushes)")
    print(f"\n  Over/Under split (resolved only):")
    ou = summary["over_under_split"]
    print(f"    OVER:  N={ou['over_n']}, WR={ou['over_wr']}%")
    print(f"    UNDER: N={ou['under_n']}, WR={ou['under_wr']}%")
    print(f"\n  By stat type (WR excl pushes):")
    print(f"    {'Stat':<12} {'Total':>6} {'Push%':>6} {'Resolvd':>8} {'Dist WR':>8} {'Mean WR':>8}")
    for stat, data in sorted(by_stat.items()):
        resolved_n = data["n"] - data.get("pushes", 0)
        print(f"    {stat:<12} {data['n']:>6} {data['push_rate']:>5.1f}% {resolved_n:>8} "
              f"{data['dist_wr']:>7.1f}% {data['mean_wr']:>7.1f}%")
    print(f"\n  By edge bucket (resolved only):")
    for bucket, data in sorted(by_edge_bucket.items()):
        print(f"    {bucket:<16} N={data['n']:>5}, Dist WR={data['dist_wr']:.1f}%")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestration
# ══════════════════════════════════════════════════════════════════════════════

def get_current_nfl_week() -> tuple[int, int]:
    """Estimate current NFL season and week from the date.

    NFL season 2026 runs Sep 10, 2026 - Jan 3, 2027.
    Returns (season, week) — approximate.
    """
    from datetime import date
    today = date.today()
    # NFL 2026 season starts ~Sep 10, 2026
    # Each week is 7 days
    if today.month >= 9:
        season = today.year
        # Week 1 starts around Sep 10
        season_start = date(today.year, 9, 10)
    elif today.month <= 2:
        season = today.year - 1
        season_start = date(today.year - 1, 9, 10)
    else:
        # Offseason
        season = today.year
        season_start = date(today.year, 9, 10)

    days_since = (today - season_start).days
    week = max(1, min(18, (days_since // 7) + 1))
    return season, week


def run_weekly_pipeline(season: int | None = None, week: int | None = None,
                        fetch_props: bool = True) -> dict:
    """
    Full pipeline: fetch projections → build consensus → match to props → rank gaps.

    Args:
        season: NFL season year (default: current)
        week: NFL week number (default: current)
        fetch_props: if True, trigger a props fetch for NFL before analysis

    Returns dict with all results.
    """
    if season is None or week is None:
        s, w = get_current_nfl_week()
        season = season or s
        week = week or w

    print(f"[FantasyProj] Running pipeline for {season} week {week}")

    # 1. Fetch projections from all available sources
    all_sources = []

    # Sleeper (always available — free)
    sleeper = fetch_sleeper_projections(season, week)
    if sleeper:
        all_sources.append(sleeper)
        store_raw_projections(season, week, sleeper)
        print(f"  [Sleeper] {len(sleeper)} players")

    time.sleep(1)

    # ESPN (free — try it)
    espn = fetch_espn_projections(season, week)
    if espn:
        all_sources.append(espn)
        store_raw_projections(season, week, espn)
        print(f"  [ESPN]    {len(espn)} players")

    if not all_sources:
        return {"error": "No projection data available", "season": season, "week": week}

    # 2. Build consensus
    consensus = build_consensus(all_sources)
    print(f"  [Consensus] {len(consensus)} players with projections")

    # 3. Fetch prop lines and find gaps
    if fetch_props:
        try:
            from creator_tier import fetch_player_props
            print("  [Props] Fetching NFL props from TheOddsAPI...")
            fetch_player_props("americanfootball_nfl", hours_ahead=72)
        except Exception as exc:
            print(f"  [Props] Fetch failed (will use stored): {exc}")

    prop_lines = _fetch_current_nfl_prop_lines()
    print(f"  [Props] {len(prop_lines)} players with prop lines")

    # Build baselines for gap-reason tagging (role-change detection)
    baselines = _build_player_baselines(season, week)
    if baselines:
        print(f"  [Baselines] {len(baselines)} players with prior-week baselines")

    # Build per-player variance estimates for distributional precision
    player_vars = _build_player_variances(season, week)
    if player_vars:
        print(f"  [Variance] {len(player_vars)} players with variance estimates")

    # Build tendency-based adjustments (box count, blitz sigma, coverage, volume)
    tendency_adj = {}
    try:
        tendency_adj = _build_tendency_adjustments(prop_lines)
        if tendency_adj:
            print(f"  [Tendency] {len(tendency_adj)} teams with matchup adjustments")
    except Exception as exc:
        print(f"  [Tendency] Build failed (skipping): {exc}")

    # Compute segment bias corrections from PRIOR weeks only (rolling window).
    # Never fit and evaluate on the same data — forward honesty.
    segment_biases = _compute_segment_bias(season, before_week=week)
    if segment_biases:
        print(f"  [Bias] {len(segment_biases)} segment corrections (weeks < {week})")
        for (stat, bpos), bias in sorted(segment_biases.items()):
            if abs(bias) >= 1.0:
                print(f"    {stat} {bpos}: {bias:+.1f}")

    gaps = find_prop_gaps(consensus, prop_lines, player_baselines=baselines,
                          player_variances=player_vars,
                          tendency_adjustments=tendency_adj,
                          segment_biases=segment_biases,
                          min_edge_pp=0.05)
    if gaps:
        store_consensus(season, week, gaps)
        # Store forward predictions for later grading
        n_fwd = store_forward_predictions(season, week, gaps)
        print(f"  [Forward] Stored {n_fwd} predictions for grading")

    # 4. Format summary
    n_matched = len(set(g["player"] for g in gaps))
    n_tight = len([g for g in gaps if g["n_sources"] >= 2 and g["consensus_sd"] > 0])

    print(f"\n  === RESULTS: {len(gaps)} gaps found ({n_matched} players matched to props) ===")
    print(f"  {n_tight} gaps with multi-source consensus")

    if gaps:
        print(f"\n  Top 10 opportunities (by z-score):")
        print(f"  {'Player':<22} {'Stat':<12} {'Proj':>6} {'Line':>6} {'|Gap|':>6} {'SD':>6} {'Dir':<6} {'Z':>6} {'Reason':<12}")
        print(f"  {'-'*92}")
        for g in gaps[:10]:
            print(f"  {g['player']:<22} {g['stat']:<12} {g['consensus_mean']:>6.1f} "
                  f"{g['prop_line']:>6.1f} {g['abs_gap']:>6.1f} "
                  f"{g['consensus_sd']:>6.1f} {g['direction']:<6} {g['z_score']:>6.1f} "
                  f"{g.get('gap_reason', 'generic'):<12}")

    return {
        "season": season,
        "week": week,
        "sources": [s[0]["source"] for s in all_sources if s],
        "n_players_projected": len(consensus),
        "n_players_with_props": len(prop_lines),
        "n_matched": n_matched,
        "n_gaps": len(gaps),
        "top_gaps": gaps[:20],
        "all_gaps": gaps,
        "consensus": consensus,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="NFL Fantasy Projection Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run weekly pipeline")
    p_run.add_argument("--season", type=int)
    p_run.add_argument("--week", type=int)
    p_run.add_argument("--no-props", action="store_true",
                       help="Skip prop fetch (use stored props)")

    p_bt = sub.add_parser("backtest", help="Backtest projection accuracy")
    p_bt.add_argument("--season", type=int, required=True)
    p_bt.add_argument("--weeks", type=str, default="1-18",
                      help="Week range, e.g. '1-18' or '1,2,3'")

    p_grade = sub.add_parser("grade", help="Grade forward predictions for a completed week")
    p_grade.add_argument("--season", type=int, required=True)
    p_grade.add_argument("--week", type=int, required=True)

    p_report = sub.add_parser("report", help="Cumulative forward test report")
    p_report.add_argument("--season", type=int, required=True)
    p_report.add_argument("--weeks", type=str, default=None,
                          help="Week range, e.g. '2-5' or '2,3,4' (default: all graded)")

    p_test = sub.add_parser("test-fetch", help="Test fetching from each source")
    p_test.add_argument("--season", type=int, default=2025)
    p_test.add_argument("--week", type=int, default=1)

    args = parser.parse_args()

    if args.cmd == "run":
        result = run_weekly_pipeline(args.season, args.week,
                                     fetch_props=not args.no_props)
        print(f"\nPipeline complete: {result.get('n_gaps', 0)} gaps found")

    elif args.cmd == "backtest":
        if "-" in args.weeks:
            start, end = args.weeks.split("-")
            weeks = list(range(int(start), int(end) + 1))
        else:
            weeks = [int(w) for w in args.weeks.split(",")]
        result = backtest_projections(args.season, weeks)
        print(json.dumps(result, indent=2))

    elif args.cmd == "grade":
        result = grade_forward_predictions(args.season, args.week)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "report":
        if args.weeks:
            if "-" in args.weeks:
                start, end = args.weeks.split("-")
                weeks = list(range(int(start), int(end) + 1))
            else:
                weeks = [int(w) for w in args.weeks.split(",")]
        else:
            weeks = None
        result = forward_report(args.season, weeks)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "test-fetch":
        print(f"Testing Sleeper for {args.season} week {args.week}...")
        sleeper = fetch_sleeper_projections(args.season, args.week)
        print(f"  Sleeper: {len(sleeper)} players")
        if sleeper:
            sample = sleeper[0]
            print(f"  Sample: {sample['player_name']} ({sample['position']}) — {sample['stats']}")

        print(f"\nTesting ESPN for {args.season} week {args.week}...")
        espn = fetch_espn_projections(args.season, args.week)
        print(f"  ESPN: {len(espn)} players")
        if espn:
            sample = espn[0]
            print(f"  Sample: {sample['player_name']} ({sample['position']}) — {sample['stats']}")

    else:
        parser.print_help()
