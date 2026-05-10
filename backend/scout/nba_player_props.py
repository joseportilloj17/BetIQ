"""
scout/nba_player_props.py — NBA player prop projections.

Markets: player_points, player_rebounds, player_assists,
         player_pra (points+rebounds+assists), player_threes,
         player_steals, player_blocks.

For each player in both rosters we:
  1. Fetch season averages + last-N-game log from ESPN
  2. Blend season/recent (60/40)
  3. Estimate std dev from sport/stat CV table
  4. Project P(stat >= common thresholds) via normal CDF
  5. Return the best threshold that has meaningful edge
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from scout.base import GameInfo, ScoutedProp, PROJECTION_VERSION
from scout.projection_engine import (
    blend_season_recent,
    clamp_probability,
    compute_ci_95,
    grade_from_probability,
    project_over_under,
    std_from_average,
    validate_projection,
)
import scout.data_sources as ds

# ── Market definitions ─────────────────────────────────────────────────────────

_STAT_MAP = {
    # market_type       stat_key_season  stat_key_recent  sport_stat
    "player_points":   ("avg_points",   "points",        "points"),
    "player_rebounds": ("avg_rebounds", "rebounds",      "rebounds"),
    "player_assists":  ("avg_assists",  "assists",       "assists"),
    "player_threes":   ("avg_threes",   "threes",        "threes"),
    "player_steals":   ("avg_steals",   "steals",        "steals"),
    "player_blocks":   ("avg_blocks",   "blocks",        "blocks"),
}

# Typical lines bookmakers offer for each market
_COMMON_THRESHOLDS: dict[str, list[float]] = {
    "player_points":   [10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5, 24.5, 26.5, 28.5, 30.5],
    "player_rebounds": [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
    "player_assists":  [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    "player_pra":      [14.5, 17.5, 19.5, 21.5, 24.5, 27.5, 29.5, 32.5, 35.5, 38.5],
    "player_threes":   [0.5, 1.5, 2.5, 3.5, 4.5],
    "player_steals":   [0.5, 1.5, 2.5],
    "player_blocks":   [0.5, 1.5, 2.5],
}

# Minimum season avg required to project a market (filter out DNPs/fringe players)
_MIN_AVG: dict[str, float] = {
    "player_points":   5.0,
    "player_rebounds": 2.0,
    "player_assists":  1.5,
    "player_pra":      10.0,
    "player_threes":   0.3,
    "player_steals":   0.2,
    "player_blocks":   0.2,
}

# Minimum hit probability to emit a ScoutedProp (avoid flooding with D-grades)
_MIN_EMIT_PROB = 0.45


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _recent_avg(recent_games: list[dict], stat_key: str, n: int = 7) -> Optional[float]:
    """Mean of `stat_key` from the last n games. Returns None if no data."""
    vals = []
    for g in recent_games[:n]:
        v = _safe_float(g.get(stat_key))
        if v > 0:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pick_best_threshold(
    market: str,
    projected_mean: float,
    std: float,
) -> tuple[float, float]:
    """
    Choose the threshold closest to the projected mean (most uncertainty = most
    edge opportunity) but leaning toward lines where |hit_prob - 0.5| is
    maximised.  Returns (threshold, hit_probability).
    """
    thresholds = _COMMON_THRESHOLDS.get(market, [])
    if not thresholds:
        return projected_mean, 0.5

    best_thresh = thresholds[0]
    best_prob   = project_over_under(projected_mean, std, best_thresh)
    best_edge   = abs(best_prob - 0.50)

    for t in thresholds[1:]:
        p    = project_over_under(projected_mean, std, t)
        edge = abs(p - 0.50)
        if edge > best_edge:
            best_edge  = edge
            best_thresh = t
            best_prob   = p

    return best_thresh, best_prob


def _project_pra(season_stats: dict, recent_games: list[dict]) -> tuple[float, float]:
    """Project combined PRA mean and std."""
    pts_avg = _safe_float(season_stats.get("avg_points"))
    reb_avg = _safe_float(season_stats.get("avg_rebounds"))
    ast_avg = _safe_float(season_stats.get("avg_assists"))
    pra_season = pts_avg + reb_avg + ast_avg

    # Recent PRA per game
    pra_recent_vals = []
    for g in recent_games[:7]:
        v = (_safe_float(g.get("points")) +
             _safe_float(g.get("rebounds")) +
             _safe_float(g.get("assists")))
        if v > 0:
            pra_recent_vals.append(v)
    pra_recent = (sum(pra_recent_vals) / len(pra_recent_vals)) if pra_recent_vals else None

    pra_mean = blend_season_recent(pra_season, pra_recent)
    pra_std  = std_from_average(pra_mean, "NBA", "points") * 1.3  # PRA has more variance
    return pra_mean, pra_std


# ── Per-player projection ──────────────────────────────────────────────────────

def _scout_player(
    player: dict,
    game: GameInfo,
    team: str,
    scout_date: str,
) -> List[ScoutedProp]:
    """Project all markets for a single NBA player."""
    results: List[ScoutedProp] = []

    player_id   = str(player.get("id", ""))
    player_name = player.get("fullName") or player.get("displayName", "Unknown")
    position    = player.get("position", {})
    pos_abbr    = position.get("abbreviation", "") if isinstance(position, dict) else str(position)

    # Skip non-playable positions (coaches, two-way, etc.)
    if pos_abbr in ("", "C/PF"):  # keep all playing positions
        pass

    # Fetch season stats
    season_stats = ds.get_nba_player_season_stats(player_id)
    if not season_stats:
        return results

    # Fetch recent game log
    recent_games = ds.get_nba_player_recent_games(player_id, n=10)

    # Determine home/road split signal
    is_home = (team == game.home_team)
    home_road_note = "home" if is_home else "away"

    # ── Standard markets ──────────────────────────────────────────────────────
    for market_type, (season_key, recent_key, sport_stat) in _STAT_MAP.items():
        season_avg = _safe_float(season_stats.get(season_key))
        if season_avg < _MIN_AVG.get(market_type, 0.5):
            continue

        recent_avg_val = _recent_avg(recent_games, recent_key, n=7)
        projected_mean = blend_season_recent(season_avg, recent_avg_val)
        std            = std_from_average(projected_mean, "NBA", sport_stat)

        # Minutes trend — down-grade if avg minutes < 20 (limited opportunity)
        avg_min = _safe_float(season_stats.get("avg_minutes", 0))
        confidence_factors = [f"Season avg: {season_avg:.1f}", home_road_note]
        risk_factors       = []

        if avg_min > 0 and avg_min < 20:
            risk_factors.append(f"Low minutes ({avg_min:.0f} MPG) — bench risk")
        if recent_avg_val is not None:
            confidence_factors.append(f"L7 avg: {recent_avg_val:.1f}")
        else:
            risk_factors.append("No recent game log available")

        risk_factors = validate_projection(projected_mean, season_avg, std, risk_factors)

        threshold, hit_prob = _pick_best_threshold(market_type, projected_mean, std)
        if hit_prob < _MIN_EMIT_PROB:
            continue

        grade = grade_from_probability(hit_prob)
        side  = "over" if hit_prob >= 0.50 else "under"
        if side == "under":
            hit_prob = clamp_probability(1.0 - hit_prob)

        lo95, hi95 = compute_ci_95(projected_mean, std)

        results.append(ScoutedProp(
            scout_date       = scout_date,
            sport            = "NBA",
            game_id          = game.game_id,
            home_team        = game.home_team,
            away_team        = game.away_team,
            commence_time    = game.commence_time,
            market_type      = market_type,
            player_name      = player_name,
            player_id        = player_id,
            team             = team,
            side             = side,
            threshold        = threshold,
            projected_value  = round(projected_mean, 2),
            projected_low_95 = round(lo95, 2),
            projected_high_95= round(hi95, 2),
            projected_std_dev= round(std, 2),
            hit_probability  = round(hit_prob, 4),
            quality_grade    = grade,
            confidence_factors = confidence_factors,
            risk_factors       = risk_factors,
            data_source        = "espn",
            projection_version = PROJECTION_VERSION,
        ))

    # ── PRA market ────────────────────────────────────────────────────────────
    pts_avg = _safe_float(season_stats.get("avg_points"))
    reb_avg = _safe_float(season_stats.get("avg_rebounds"))
    ast_avg = _safe_float(season_stats.get("avg_assists"))
    if pts_avg + reb_avg + ast_avg >= _MIN_AVG["player_pra"]:
        pra_mean, pra_std = _project_pra(season_stats, recent_games)
        threshold, hit_prob = _pick_best_threshold("player_pra", pra_mean, pra_std)
        if hit_prob >= _MIN_EMIT_PROB:
            grade = grade_from_probability(hit_prob)
            side  = "over" if hit_prob >= 0.50 else "under"
            if side == "under":
                hit_prob = clamp_probability(1.0 - hit_prob)
            lo95, hi95 = compute_ci_95(pra_mean, pra_std)
            results.append(ScoutedProp(
                scout_date        = scout_date,
                sport             = "NBA",
                game_id           = game.game_id,
                home_team         = game.home_team,
                away_team         = game.away_team,
                commence_time     = game.commence_time,
                market_type       = "player_pra",
                player_name       = player_name,
                player_id         = player_id,
                team              = team,
                side              = side,
                threshold         = threshold,
                projected_value   = round(pra_mean, 2),
                projected_low_95  = round(lo95, 2),
                projected_high_95 = round(hi95, 2),
                projected_std_dev = round(pra_std, 2),
                hit_probability   = round(hit_prob, 4),
                quality_grade     = grade,
                confidence_factors= [f"PRA season: {pts_avg+reb_avg+ast_avg:.1f}", home_road_note],
                risk_factors      = [],
                data_source       = "espn",
                projection_version= PROJECTION_VERSION,
            ))

    return results


# ── Game-level entry point ─────────────────────────────────────────────────────

def scout_game(game: GameInfo) -> List[ScoutedProp]:
    """
    Generate all NBA player prop ScoutedProps for one game.
    Fetches rosters for both teams and projects all eligible players.
    """
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    for team_id, team_name in [
        (game.home_team_id, game.home_team),
        (game.away_team_id, game.away_team),
    ]:
        roster = ds.get_nba_team_roster(team_id)
        if not roster:
            continue
        for player in roster:
            try:
                props = _scout_player(player, game, team_name, scout_date)
                results.extend(props)
            except Exception as exc:
                print(f"[nba_player_props] skip {player.get('fullName', '?')}: {exc}")

    return results
