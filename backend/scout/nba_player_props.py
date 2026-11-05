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
from math import sqrt
from typing import List, Optional

from scout.base import GameInfo, ScoutedProp, PROJECTION_VERSION
from scout.projection_engine import (
    blend_season_recent,
    clamp_probability,
    compute_ci_95,
    detect_streak,
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
    def_factor: float = 1.0,
    engine_proj: Optional[dict] = None,
) -> List[ScoutedProp]:
    """Project all markets for a single NBA player.

    def_factor: multiplicative adjustment for opponent defense.
    1.0 = league average. >1.0 = soft defense. <1.0 = tough defense.
    Applied only to scoring markets (points, pra, threes).

    engine_proj: when supplied (from NBAMatchupEngine), provides
    matchup-specific projections with pace, defense, form, and rest
    adjustments already baked in. Falls back to ESPN blend when None.
    """
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

    # Scoring markets that benefit from defensive adjustment
    _DEF_ADJUSTED = {"player_points", "player_pra", "player_threes"}

    # Mapping from market_type → engine_proj key + std key
    _ENGINE_MAP = {
        "player_points":   ("points",   "pts_std"),
        "player_rebounds":  ("rebounds", "reb_std"),
        "player_assists":   ("assists",  "ast_std"),
        "player_threes":    ("threes",   "thr_std"),
        "player_steals":    ("steals",   None),
        "player_blocks":    ("blocks",   None),
    }

    # ── Standard markets ──────────────────────────────────────────────────────
    for market_type, (season_key, recent_key, sport_stat) in _STAT_MAP.items():
        season_avg = _safe_float(season_stats.get(season_key))
        if season_avg < _MIN_AVG.get(market_type, 0.5):
            continue

        recent_avg_val = _recent_avg(recent_games, recent_key, n=7)

        # ── Use matchup engine projection when available ──
        eng_key, eng_std_key = _ENGINE_MAP.get(market_type, (None, None))
        if engine_proj and eng_key and engine_proj.get(eng_key, 0) > 0:
            projected_mean = engine_proj[eng_key]
            # Use engine's game-log std dev if available
            if eng_std_key and engine_proj.get(eng_std_key, 0) > 0:
                std = engine_proj[eng_std_key]
            else:
                std = std_from_average(projected_mean, "NBA", sport_stat)
        else:
            # Fallback: ESPN season + recent blend
            projected_mean = blend_season_recent(season_avg, recent_avg_val)
            if market_type in _DEF_ADJUSTED and def_factor != 1.0:
                projected_mean = projected_mean * def_factor
            std = std_from_average(projected_mean, "NBA", sport_stat)

        # Minutes trend — down-grade if avg minutes < 20 (limited opportunity)
        avg_min = _safe_float(season_stats.get("avg_minutes", 0))
        confidence_factors = [f"Season avg: {season_avg:.1f}", home_road_note]
        risk_factors       = []

        # Engine adjustment annotations
        if engine_proj and eng_key:
            adj_parts = []
            if engine_proj.get("pace_adj", 1.0) != 1.0:
                adj_parts.append(f"pace {engine_proj['pace_adj']:.2f}x")
            if engine_proj.get("def_adj", 1.0) != 1.0:
                d = engine_proj["def_adj"]
                label = "soft" if d > 1.0 else "tough" if d < 1.0 else "avg"
                adj_parts.append(f"def {d:.2f}x ({label})")
            if engine_proj.get("form_adj", 1.0) != 1.0:
                adj_parts.append(f"form {engine_proj['form_adj']:.2f}x")
            if engine_proj.get("rest_adj", 1.0) != 1.0:
                adj_parts.append(f"rest {engine_proj['rest_adj']:.2f}x")
            if adj_parts:
                confidence_factors.append("Matchup adj: " + ", ".join(adj_parts))
            confidence_factors.append("src:matchup_engine")

        # Hot/cold streak detection (L3 games)
        streak_label, l3_avg = detect_streak(recent_games, recent_key, season_avg, n=3)
        if streak_label == "HOT":
            confidence_factors.append(f"HOT streak: L3 avg {l3_avg:.1f}")
        elif streak_label == "COLD":
            risk_factors.append(f"COLD streak: L3 avg {l3_avg:.1f}")

        if not engine_proj and def_factor != 1.0:
            dir_label = "soft" if def_factor > 1.0 else "tough" if def_factor < 1.0 else "avg"
            confidence_factors.append(f"Opp def adj: {def_factor:.2f}x ({dir_label})")
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
            risk_factors       = risk_factors + ["UNCALIBRATED"],
            data_source        = "matchup_engine" if engine_proj else "espn",
            projection_version = PROJECTION_VERSION,
        ))

    # ── PRA market ────────────────────────────────────────────────────────────
    pts_avg = _safe_float(season_stats.get("avg_points"))
    reb_avg = _safe_float(season_stats.get("avg_rebounds"))
    ast_avg = _safe_float(season_stats.get("avg_assists"))
    if pts_avg + reb_avg + ast_avg >= _MIN_AVG["player_pra"]:
        # Use engine PRA when available (already matchup-adjusted)
        if engine_proj and engine_proj.get("pra", 0) > 0:
            pra_mean = engine_proj["pra"]
            # B3: Use combined-rate PRA std (captures covariance through shared minutes)
            if engine_proj.get("pra_std", 0) > 0:
                pra_std = engine_proj["pra_std"]
            else:
                # Fallback: independent sum (underestimates)
                p_std = engine_proj.get("pts_std", 0)
                r_std = engine_proj.get("reb_std", 0)
                a_std = engine_proj.get("ast_std", 0)
                pra_std = sqrt(p_std**2 + r_std**2 + a_std**2) if (p_std + r_std + a_std) > 0 else pra_mean * 0.20
        else:
            pra_mean, pra_std = _project_pra(season_stats, recent_games)
            if def_factor != 1.0:
                pra_mean = pra_mean * (1.0 + (def_factor - 1.0) * 0.55)

        threshold, hit_prob = _pick_best_threshold("player_pra", pra_mean, pra_std)
        if hit_prob >= _MIN_EMIT_PROB:
            grade = grade_from_probability(hit_prob)
            side  = "over" if hit_prob >= 0.50 else "under"
            if side == "under":
                hit_prob = clamp_probability(1.0 - hit_prob)
            lo95, hi95 = compute_ci_95(pra_mean, pra_std)
            pra_conf = [f"PRA season: {pts_avg+reb_avg+ast_avg:.1f}", home_road_note]
            if engine_proj:
                pra_conf.append("src:matchup_engine")
            elif def_factor != 1.0:
                pra_conf.append(f"Opp def adj: {def_factor:.2f}x")
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
                confidence_factors= pra_conf,
                risk_factors      = ["UNCALIBRATED"],
                data_source       = "espn",
                projection_version= PROJECTION_VERSION,
            ))

    return results


# ── Opponent defensive adjustment ─────────────────────────────────────────────

_NBA_LEAGUE_AVG_PPG = 114.5   # ~2024-25 season average points allowed per game

# ── Matchup engine singleton ─────────────────────────────────────────────────
_engine = None

def _get_engine():
    """Lazy-load the NBAMatchupEngine (cached, loads once per process)."""
    global _engine
    if _engine is None:
        try:
            from nba_matchup_engine import NBAMatchupEngine
            _engine = NBAMatchupEngine(cache=True)  # auto-detects current season
            _engine.load()
            n_roster = len(getattr(_engine, '_roster_teams', {}))
            print(f"[nba_player_props] Matchup engine loaded: "
                  f"{len(_engine.player_profiles)} players, "
                  f"{len(_engine.trades)} trades, "
                  f"{n_roster} roster assignments")
            # Report any roster-vs-profile team mismatches
            changes = _engine.report_roster_changes()
            if changes:
                print(f"[nba_player_props] WARNING: {len(changes)} roster team mismatches")
        except Exception as e:
            print(f"[nba_player_props] Matchup engine unavailable: {e}")
            _engine = False  # sentinel: tried and failed
    return _engine if _engine is not False else None


def _opp_def_factor(opp_team_id: str, opp_abbr: str = "",
                    stat: str = "points") -> float:
    """
    Defensive adjustment factor for a stat vs an opponent.

    Prefers the matchup engine (per-stat defense from nba_api opponent stats).
    Falls back to ESPN season stats if engine unavailable.
    Returns multiplier: >1.0 = soft defense, <1.0 = tough defense.
    Clamped to [0.85, 1.15].
    """
    # Try matchup engine first (much richer data)
    engine = _get_engine()
    if engine and opp_abbr:
        raw = engine._defensive_factor(opp_abbr, stat)
        return max(0.85, min(1.15, raw))

    # Fallback: ESPN season stats
    try:
        stats = ds.get_nba_team_season_stats(opp_team_id)
        pts_allowed = _safe_float(stats.get("avg_points_allowed", 0))
        if pts_allowed <= 0:
            return 1.0
        factor = pts_allowed / _NBA_LEAGUE_AVG_PPG
        return max(0.85, min(1.15, factor))
    except Exception:
        return 1.0


def _proj_to_dict(proj) -> dict:
    """Serialize a PlayerProjection to a dict for _scout_player()."""
    return {
        "points": proj.points, "rebounds": proj.rebounds,
        "assists": proj.assists, "steals": proj.steals,
        "blocks": proj.blocks, "threes": proj.threes,
        "pra": proj.pra, "minutes": proj.minutes,
        "pts_std": proj.pts_std, "reb_std": proj.reb_std,
        "ast_std": proj.ast_std, "thr_std": proj.thr_std,
        "pra_std": proj.pra_std,
        "pace_adj": proj.pace_adj, "def_adj": proj.def_adj,
        "form_adj": proj.form_adj, "rest_adj": proj.rest_adj,
        "recon_pts_scale": proj.recon_pts_scale,
        "recon_reb_scale": proj.recon_reb_scale,
        "recon_ast_scale": proj.recon_ast_scale,
    }


def _engine_game_projection(
    home_abbr: str, away_abbr: str,
    market_total: float = 0.0,
    market_spread: float = 0.0,
    exclude_players: set | None = None,
) -> Optional[dict]:
    """Run project_game() and return {player_name: projection_dict}.

    Falls back to None if engine unavailable.
    exclude_players: explicit set of player names to exclude (e.g. from
        injury report). Merged with engine-level exclusion.
    """
    engine = _get_engine()
    if not engine:
        return None
    try:
        game_proj = engine.project_game(
            home_abbr, away_abbr,
            market_total=market_total,
            market_spread=market_spread,
            exclude_players=exclude_players,
        )
        return {
            name: _proj_to_dict(pp)
            for name, pp in game_proj.players.items()
        }
    except Exception as e:
        print(f"[nba_player_props] project_game failed: {e}")
        return None


def _engine_game_simulation(
    home_abbr: str, away_abbr: str,
    market_total: float = 0.0,
    market_spread: float = 0.0,
    n_draws: int = 10_000,
    exclude_players: set | None = None,
    injury_statuses: dict | None = None,
    hours_to_tip: float | None = None,
):
    """B5: Run Monte Carlo game simulation for SGP pricing.

    Returns (player_proj_dict, SimulationResult) or (None, None).
    """
    engine = _get_engine()
    if not engine:
        return None, None
    try:
        game_proj = engine.project_game(
            home_abbr, away_abbr,
            market_total=market_total,
            market_spread=market_spread,
            exclude_players=exclude_players,
        )
        proj_dict = {
            name: _proj_to_dict(pp)
            for name, pp in game_proj.players.items()
        }
        # Extract absence impacts from minutes model for per-draw redistribution
        absence_impacts = None
        try:
            absence_impacts = engine.get_absence_impacts_for_game(
                game_proj, home_abbr, away_abbr
            ) or None
        except Exception:
            pass
        # Run B5 joint-draw simulation
        from nba_game_sim import GameSimulator
        sim = GameSimulator(
            game_proj,
            injury_statuses=injury_statuses,
            absence_impacts=absence_impacts,
            hours_to_tip=hours_to_tip,
        )
        result = sim.run(n_draws=n_draws)
        return proj_dict, result
    except Exception as e:
        print(f"[nba_player_props] game simulation failed: {e}")
        return None, None


def _engine_projection(player_name: str, opp_abbr: str,
                       is_home: bool) -> Optional[dict]:
    """Get matchup-engine projection for a player if available.

    Single-player fallback (no reconciliation). Prefer _engine_game_projection
    when game context is available.
    """
    engine = _get_engine()
    if not engine:
        return None
    proj = engine.project_player(player_name, opp_abbr, is_home=is_home)
    if proj is None:
        return None
    return _proj_to_dict(proj)


# ── Game-level entry point ─────────────────────────────────────────────────────

def scout_game(game: GameInfo) -> List[ScoutedProp]:
    """
    Generate all NBA player prop ScoutedProps for one game.

    Uses project_game() (B3) for team-reconciled projections when the
    matchup engine is loaded. Falls back to per-player project_player()
    or ESPN season averages + generic defense factor.
    """
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    # Determine opponent abbreviations for matchup engine
    # GameInfo.extra may have {'home_abbr': 'PHI', 'away_abbr': 'CLE'}
    extra = game.extra if hasattr(game, "extra") and isinstance(game.extra, dict) else {}
    home_abbr = extra.get("home_abbr", game.home_team or "")
    away_abbr = extra.get("away_abbr", game.away_team or "")

    # B3-F1: Run project_game() once for team-reconciled projections.
    # market_total/market_spread are populated by runner.py from fixtures.
    market_total = float(extra.get("market_total", 0))
    market_spread = float(extra.get("market_spread", 0))
    _no_anchor = bool(extra.get("_no_market_anchor"))
    if _no_anchor:
        print(f"[nba_player_props] [NO MARKET ANCHOR] {game.away_team} @ "
              f"{game.home_team} — using model-only fallback")

    # Fetch injury exclusions (Out/IR players)
    injury_exclude = set()
    engine = _get_engine()
    if engine:
        try:
            if not hasattr(engine, "_injury_report"):
                engine.fetch_injury_report()
                engine.log_injury_snapshot()
            injury_exclude = engine.get_excluded_players()
            if injury_exclude:
                # Filter to players relevant to this game's teams
                game_exclude = {n for n in injury_exclude
                                if engine._injury_report.get(n, {}).get("team") in
                                (home_abbr, away_abbr)}
                if game_exclude:
                    print(f"[nba_player_props] Excluding {len(game_exclude)} "
                          f"injured players: {', '.join(sorted(game_exclude)[:5])}"
                          f"{'...' if len(game_exclude) > 5 else ''}")
                injury_exclude = game_exclude
        except Exception as e:
            print(f"[nba_player_props] Injury fetch failed: {e}")

    game_projs = _engine_game_projection(
        home_abbr, away_abbr,
        market_total=market_total,
        market_spread=market_spread,
        exclude_players=injury_exclude or None,
    )

    # Build injury_statuses dict for sim availability draws
    # Maps player_name -> status string for non-excluded players
    # (Out/IR already excluded above; this carries DTD/GTD/Questionable/Probable)
    injury_statuses = {}
    if engine and hasattr(engine, "_injury_report"):
        for pname, info in engine._injury_report.items():
            if info.get("team") in (home_abbr, away_abbr):
                status = info.get("status")
                if status and pname not in injury_exclude:
                    injury_statuses[pname] = status

    # Compute hours to tip for timing-conditional P(plays) rates
    hours_to_tip = None
    try:
        from datetime import datetime, timezone
        tip_utc = datetime.fromisoformat(game.commence_time.replace("Z", "+00:00"))
        now_utc = datetime.now(timezone.utc)
        hours_to_tip = (tip_utc - now_utc).total_seconds() / 3600
        if hours_to_tip < 0:
            hours_to_tip = 0.0
    except Exception:
        pass

    # B5: Run joint-draw simulation for SGP pricing (non-blocking)
    game_sim = None
    try:
        _, game_sim = _engine_game_simulation(
            home_abbr, away_abbr,
            market_total=market_total,
            market_spread=market_spread,
            exclude_players=injury_exclude or None,
            injury_statuses=injury_statuses or None,
            hours_to_tip=hours_to_tip,
        )
        if game_sim is not None:
            print(f"[nba_player_props] B5 sim: {game_sim.n_draws} draws, "
                  f"{len(game_sim.player_names)} players"
                  f"{f', {len(injury_statuses)} with injury status' if injury_statuses else ''}")
    except Exception as e:
        print(f"[nba_player_props] B5 sim skipped: {e}")

    # Pre-fetch defensive factors for both opponents (fallback path)
    home_def_pts = _opp_def_factor(game.away_team_id, away_abbr, "points")
    away_def_pts = _opp_def_factor(game.home_team_id, home_abbr, "points")

    for team_id, team_name, def_factor, is_home in [
        (game.home_team_id, game.home_team, home_def_pts, True),
        (game.away_team_id, game.away_team, away_def_pts, False),
    ]:
        opp_abbr = away_abbr if is_home else home_abbr
        roster = ds.get_nba_team_roster(team_id)
        if not roster:
            continue
        for player in roster:
            try:
                player_name = player.get("fullName") or player.get("displayName", "")

                # B3: Use reconciled game-level projection; fall back to per-player
                eng_proj = (game_projs or {}).get(player_name)
                if eng_proj is None:
                    eng_proj = _engine_projection(player_name, opp_abbr, is_home)

                props = _scout_player(
                    player, game, team_name, scout_date,
                    def_factor=def_factor,
                    engine_proj=eng_proj,
                )
                # B3-F1: Tag every prop from this game with NO MARKET ANCHOR
                if _no_anchor:
                    for p in props:
                        if hasattr(p, "risk_factors") and isinstance(p.risk_factors, list):
                            p.risk_factors.append("NO MARKET ANCHOR")
                results.extend(props)
            except Exception as exc:
                print(f"[nba_player_props] skip {player.get('fullName', '?')}: {exc}")

    # B5: Attach simulation result for SGP/parlay queries
    # Callers can access via scout_game_with_sim() or the module-level cache
    game_key = f"{away_abbr}@{home_abbr}"
    if game_sim is not None:
        _SIM_CACHE[game_key] = game_sim
    # S5-2: Store context needed for nba_forward_preds persistence
    _SIM_CONTEXT[game_key] = {
        "injury_statuses": injury_statuses or {},
        "hours_to_tip": hours_to_tip,
    }

    return results


# B5: Module-level sim cache for SGP queries after scout_game() runs
_SIM_CACHE: dict = {}
# S5-2: Parallel context cache for forward-test persistence
_SIM_CONTEXT: dict = {}


def get_game_sim(away_abbr: str, home_abbr: str):
    """Retrieve the SimulationResult for a game after scout_game() has run.

    Returns None if simulation wasn't run or game not found.
    """
    return _SIM_CACHE.get(f"{away_abbr}@{home_abbr}")


def get_game_sim_context(away_abbr: str, home_abbr: str) -> dict | None:
    """Retrieve injury_statuses and hours_to_tip stored at scout time.

    Returns dict with 'injury_statuses' and 'hours_to_tip', or None.
    """
    return _SIM_CONTEXT.get(f"{away_abbr}@{home_abbr}")
