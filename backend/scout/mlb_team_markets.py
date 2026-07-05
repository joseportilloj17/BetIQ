"""
scout/mlb_team_markets.py — MLB team market projections.

Markets: h2h (moneyline), spread (run line ±1.5), totals.

Methodology (enhanced):
  - Win probability: Log5 formula with +4% home-field adjustment
  - Starting pitcher quality: ERA differential adjusts expected runs
    Elite SP (ERA < 3.00) suppresses opposing runs by up to 15%
    Soft SP (ERA > 5.00) inflates opposing runs by up to 15%
  - Park factor: hardcoded run environment table per venue
    Range: 0.88 (Petco) → 1.13 (Coors Field)
  - Day/night: day games historically run ~0.15 runs higher (ball flies)
  - Weather: wind out ≥ 15 mph adds 0.5 runs; adverse weather risk flagged
  - Recent form: team last-10 RPG blended 30% vs season RPG 70%
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from scout.base import GameInfo, ScoutedProp, PROJECTION_VERSION
from scout.projection_engine import (
    clamp_probability,
    compute_ci_95,
    grade_from_probability,
    project_spread_cover,
    project_total_over,
    project_win_probability,
)
import scout.data_sources as ds

# ── Constants ──────────────────────────────────────────────────────────────────

_HOME_ADVANTAGE = 0.040   # MLB home field ~4pp
_MARGIN_STD     = 3.2     # run-differential std dev
_TOTAL_STD      = 3.0     # combined runs std dev
_DEFAULT_RPG    = 4.4     # league-average runs per game

_TOTAL_LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
_RUN_LINE    = 1.5        # MLB run line is always ±1.5

_MIN_EMIT_PROB = 0.45

# Park factor table: venue name fragment → run factor (1.0 = neutral)
# Values calibrated from multi-year park factor databases.
# Fragments matched case-insensitively against game.venue.
_PARK_FACTORS: list[tuple[str, float]] = [
    ("coors",        1.13),   # Coors Field (altitude, thin air)
    ("great american", 1.10), # Great American Ball Park
    ("fenway",       1.07),   # Fenway Park (Green Monster)
    ("wrigley",      1.05),   # Wrigley Field (wind variable, slight hitter)
    ("yankee",       1.04),   # Yankee Stadium (short porch RF)
    ("camden",       1.04),   # Camden Yards
    ("citizen",      1.03),   # Citizens Bank Park
    ("minute maid",  1.03),   # Minute Maid Park (Tal's Hill era gone but still warm)
    ("oracle",       0.96),   # Oracle Park (marine layer, cold)
    ("petco",        0.88),   # Petco Park (marine layer + spacious)
    ("tropicana",    0.94),   # Tropicana Field (dome, suppresses)
    ("t-mobile",     0.96),   # T-Mobile Park Seattle (marine layer)
    ("progressive",  0.98),   # Progressive Field
    ("angel",        0.98),   # Angel Stadium
    ("dodger",       0.98),   # Dodger Stadium (spacious)
    ("busch",        0.97),   # Busch Stadium
    ("globe life",   1.02),   # Globe Life Field (dome, warm)
    ("american family", 0.96), # American Family Field Milwaukee
    ("chase",        1.01),   # Chase Field (retractable roof, warm)
]
_DEFAULT_PARK_FACTOR = 1.00

# Cushion logic for run lines — aligns with NBA spread framework
# Favor -1.5 only when projected margin > 2.0 AND SP ERA differential ≥ 0.5
_MIN_MARGIN_FOR_RUNLINE_FAV = 2.0
_MIN_SP_ERA_DIFF_FOR_RUNLINE = 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _win_pct(stats: dict) -> float:
    w = _safe_float(stats.get("wins"))
    l = _safe_float(stats.get("losses"))
    if w + l == 0:
        return 0.500
    return w / (w + l)


def _runs_per_game(stats: dict, recent_weight: float = 0.30) -> float:
    """Blend season RPG (70%) with recent_runs_per_g (30%) if available."""
    season_rpg = _safe_float(stats.get("runs_per_game") or stats.get("avg_runs"))
    season_rpg = season_rpg if season_rpg > 1 else _DEFAULT_RPG
    recent_rpg = _safe_float(stats.get("recent_runs_per_g"))
    if recent_rpg > 1:
        return season_rpg * (1 - recent_weight) + recent_rpg * recent_weight
    return season_rpg


def _pitcher_era_factor(pitcher_stats: dict) -> float:
    """
    Return a run-adjustment multiplier based on SP ERA.
    ERA < 3.00 → factor < 1.0 (elite, suppresses runs).
    ERA > 5.00 → factor > 1.0 (soft, inflates runs).
    Range: [0.85, 1.15].
    """
    era = _safe_float(pitcher_stats.get("era"))
    if era <= 0:
        return 1.00
    # Neutral ERA ~ 4.20 (league average). Each 1.0 ERA above = +5% runs, below = -5%.
    delta = (era - 4.20) * 0.05
    return max(0.85, min(1.15, 1.0 + delta))


def _park_factor(venue: str | None) -> float:
    """Look up park factor by venue name fragment."""
    if not venue:
        return _DEFAULT_PARK_FACTOR
    low = venue.lower()
    for fragment, factor in _PARK_FACTORS:
        if fragment in low:
            return factor
    return _DEFAULT_PARK_FACTOR


def _day_game_adjustment(commence_time: str) -> float:
    """
    Day games (before 5 PM local) historically run ~0.15 more runs.
    Use UTC commence time as proxy — games before 21:00 UTC ≈ day game.
    """
    try:
        if "T" in commence_time:
            hour_utc = int(commence_time.split("T")[1][:2])
            if hour_utc < 21:   # before 5 PM ET ≈ 21:00 UTC
                return 0.15
    except (IndexError, ValueError):
        pass
    return 0.0


def _weather_adjustment(venue: str | None, home_team: str, commence_time: str) -> tuple[float, list]:
    """
    Fetch weather and return (run_adjustment, risk_notes).
    Wind blowing out (135–315°) ≥ 15 mph: +0.5 runs.
    Adverse conditions (rain): risk flagged.
    """
    try:
        import weather as wx
        w = wx.get_game_weather(home_team, commence_time, sport="MLB")
        if not w or w.get("temperature_f") is None:
            return 0.0, []
        notes = []
        adj = 0.0
        wind_mph = w.get("wind_speed_mph") or 0
        wind_dir = w.get("wind_direction_deg")
        if wind_mph >= 15 and wind_dir is not None:
            # Wind blowing out: roughly 135–315 degrees (SW through NW through NE)
            if 135 <= wind_dir <= 315:
                adj += 0.5
                notes.append(f"Wind out {wind_mph:.0f} mph: +0.5 runs")
            else:
                adj -= 0.3
                notes.append(f"Wind in {wind_mph:.0f} mph: -0.3 runs")
        if w.get("adverse_conditions"):
            notes.append("Adverse weather — total uncertainty elevated")
        return adj, notes
    except Exception:
        return 0.0, []


def _best_total_threshold(expected_total: float) -> tuple[float, float]:
    best_t, best_prob, best_edge = _TOTAL_LINES[0], 0.5, 0.0
    for t in _TOTAL_LINES:
        p    = project_total_over(expected_total, _TOTAL_STD, t)
        edge = abs(p - 0.50)
        if edge > best_edge:
            best_edge, best_prob, best_t = edge, p, t
    return best_t, best_prob


# ── Market builders ────────────────────────────────────────────────────────────

def _build_h2h(
    game: GameInfo,
    home_wp: float,
    away_wp: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    home_prob = project_win_probability(
        home_wp, away_wp,
        home_advantage=_HOME_ADVANTAGE,
        is_home=True,
    )
    away_prob = clamp_probability(1.0 - home_prob)
    results = []

    for team, prob, side in [
        (game.home_team, home_prob, "home"),
        (game.away_team, away_prob, "away"),
    ]:
        if prob < _MIN_EMIT_PROB:
            continue
        grade = grade_from_probability(prob)
        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "MLB",
            game_id           = game.game_id,
            home_team         = game.home_team,
            away_team         = game.away_team,
            commence_time     = game.commence_time,
            market_type       = "h2h",
            player_name       = None,
            player_id         = None,
            team              = team,
            side              = side,
            threshold         = None,
            projected_value   = round(prob, 4),
            projected_low_95  = round(max(0.0, prob - 0.10), 4),
            projected_high_95 = round(min(1.0, prob + 0.10), 4),
            projected_std_dev = 0.10,
            hit_probability   = round(prob, 4),
            quality_grade     = grade,
            confidence_factors= confidence_factors,
            risk_factors      = risk_factors,
            data_source       = "espn",
            projection_version= PROJECTION_VERSION,
        ))

    return results


def _build_run_line(
    game: GameInfo,
    expected_margin: float,
    sp_era_diff: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    """
    MLB run line ±1.5 with cushion logic:
    - Favor -1.5 only when projected margin > 2.0 AND SP ERA differential ≥ 0.5
    - Underdog +1.5 always strong when projected margin within 1.0 either way
    - Underdog +2.5 strong when projected margin within 2.0 (emitted as +1.5 via standard line)
    """
    home_cover_prob = project_spread_cover(expected_margin, _MARGIN_STD, -_RUN_LINE)
    away_cover_prob = clamp_probability(1.0 - home_cover_prob)

    results = []
    candidates = [
        (game.home_team, home_cover_prob, "home", -_RUN_LINE),
        (game.away_team, away_cover_prob, "away", +_RUN_LINE),
    ]
    for team, prob, side, line in candidates:
        if prob < _MIN_EMIT_PROB:
            continue
        # Cushion gate: home -1.5 requires projected margin > 2.0 AND SP quality signal
        if side == "home" and line < 0:
            if expected_margin < _MIN_MARGIN_FOR_RUNLINE_FAV or sp_era_diff < _MIN_SP_ERA_DIFF_FOR_RUNLINE:
                risk_factors.append(f"Home -1.5 skipped: margin={expected_margin:.1f}, SP ERA diff={sp_era_diff:.1f}")
                continue

        grade = grade_from_probability(prob)
        lo, hi = compute_ci_95(expected_margin, _MARGIN_STD)
        results.append(ScoutedProp(
            scout_date        = scout_date,
            sport             = "MLB",
            game_id           = game.game_id,
            home_team         = game.home_team,
            away_team         = game.away_team,
            commence_time     = game.commence_time,
            market_type       = "spread",
            player_name       = None,
            player_id         = None,
            team              = team,
            side              = side,
            threshold         = line,
            projected_value   = round(expected_margin, 2),
            projected_low_95  = round(lo, 2),
            projected_high_95 = round(hi, 2),
            projected_std_dev = _MARGIN_STD,
            hit_probability   = round(prob, 4),
            quality_grade     = grade,
            confidence_factors= confidence_factors,
            risk_factors      = risk_factors,
            data_source       = "espn",
            projection_version= PROJECTION_VERSION,
        ))

    return results


def _build_total(
    game: GameInfo,
    expected_total: float,
    scout_date: str,
    confidence_factors: list,
    risk_factors: list,
) -> List[ScoutedProp]:
    """
    Over/under: cushion framework.
    Over X.5 strong when projected total > X.5 + 1.0.
    Under X.5 strong when projected total < X.5 - 1.0.
    """
    total_line, hit_prob = _best_total_threshold(expected_total)

    side = "over"
    if hit_prob < 0.50:
        hit_prob = clamp_probability(1.0 - hit_prob)
        side     = "under"

    if hit_prob < _MIN_EMIT_PROB:
        return []

    # Cushion gate: require at least 1.0 run of margin from the line
    cushion = abs(expected_total - total_line)
    if cushion < 1.0:
        risk_factors.append(f"Total {total_line} skipped: only {cushion:.1f} run cushion (need ≥1.0)")
        return []

    grade  = grade_from_probability(hit_prob)
    lo, hi = compute_ci_95(expected_total, _TOTAL_STD)

    return [ScoutedProp(
        scout_date        = scout_date,
        sport             = "MLB",
        game_id           = game.game_id,
        home_team         = game.home_team,
        away_team         = game.away_team,
        commence_time     = game.commence_time,
        market_type       = "totals",
        player_name       = None,
        player_id         = None,
        team              = None,
        side              = side,
        threshold         = total_line,
        projected_value   = round(expected_total, 2),
        projected_low_95  = round(lo, 2),
        projected_high_95 = round(hi, 2),
        projected_std_dev = _TOTAL_STD,
        hit_probability   = round(hit_prob, 4),
        quality_grade     = grade,
        confidence_factors= confidence_factors,
        risk_factors      = risk_factors,
        data_source       = "espn",
        projection_version= PROJECTION_VERSION,
    )]


# ── Game-level entry point ─────────────────────────────────────────────────────

def scout_game(game: GameInfo) -> List[ScoutedProp]:
    """Generate MLB team market ScoutedProps for one game with full feature set."""
    scout_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[ScoutedProp] = []

    home_stats = ds.get_mlb_team_hitting_stats(game.home_team_id)
    away_stats = ds.get_mlb_team_hitting_stats(game.away_team_id)

    confidence_factors: list[str] = []
    risk_factors: list[str]       = []

    # ── Team win% and base runs-per-game ──────────────────────────────────────
    if not home_stats:
        risk_factors.append(f"No hitting stats for {game.home_team}")
        home_wp, home_rpg = 0.500, _DEFAULT_RPG
    else:
        home_wp  = _win_pct(home_stats)
        home_rpg = _runs_per_game(home_stats)
        confidence_factors.append(f"{game.home_team} W%={home_wp:.3f} RPG={home_rpg:.1f}")

    if not away_stats:
        risk_factors.append(f"No hitting stats for {game.away_team}")
        away_wp, away_rpg = 0.500, _DEFAULT_RPG
    else:
        away_wp  = _win_pct(away_stats)
        away_rpg = _runs_per_game(away_stats)
        confidence_factors.append(f"{game.away_team} W%={away_wp:.3f} RPG={away_rpg:.1f}")

    # ── Starting pitcher quality ───────────────────────────────────────────────
    extra = game.extra or {}
    home_pid = extra.get("home_pitcher_id", "")
    away_pid = extra.get("away_pitcher_id", "")

    home_sp_stats = ds.get_mlb_player_pitching_stats(home_pid) if home_pid else {}
    away_sp_stats = ds.get_mlb_player_pitching_stats(away_pid) if away_pid else {}

    # Away pitcher faces home hitters; home pitcher faces away hitters
    home_hitter_factor = _pitcher_era_factor(away_sp_stats)  # vs away SP
    away_hitter_factor = _pitcher_era_factor(home_sp_stats)  # vs home SP

    home_era = _safe_float(home_sp_stats.get("era"))
    away_era = _safe_float(away_sp_stats.get("era"))
    sp_era_diff = abs(home_era - away_era) if home_era and away_era else 0.0

    if home_sp_stats:
        confidence_factors.append(
            f"{extra.get('home_pitcher', 'Home SP')} ERA={home_era:.2f} factor={away_hitter_factor:.2f}x on away bats"
        )
    else:
        risk_factors.append("Home SP unknown — no pitcher quality adjustment")

    if away_sp_stats:
        confidence_factors.append(
            f"{extra.get('away_pitcher', 'Away SP')} ERA={away_era:.2f} factor={home_hitter_factor:.2f}x on home bats"
        )
    else:
        risk_factors.append("Away SP unknown — no pitcher quality adjustment")

    # ── Park factor ────────────────────────────────────────────────────────────
    pf = _park_factor(game.venue)
    if pf != _DEFAULT_PARK_FACTOR:
        confidence_factors.append(f"Park factor: {pf:.2f}x ({game.venue})")

    # ── Day/night adjustment ───────────────────────────────────────────────────
    day_adj = _day_game_adjustment(game.commence_time)
    if day_adj:
        confidence_factors.append(f"Day game: +{day_adj:.2f} runs")

    # ── Weather ────────────────────────────────────────────────────────────────
    wx_adj, wx_notes = _weather_adjustment(game.venue, game.home_team, game.commence_time)
    risk_factors.extend(wx_notes)

    # ── Projected runs with all adjustments ───────────────────────────────────
    # Formula: base_rpg × pitcher_factor × park_factor + day_adj + weather_adj
    adj_home_rpg = home_rpg * home_hitter_factor * pf + day_adj / 2 + wx_adj / 2
    adj_away_rpg = away_rpg * away_hitter_factor * pf + day_adj / 2 + wx_adj / 2

    expected_margin = (adj_home_rpg - adj_away_rpg) + 0.25   # +0.25 home advantage in runs
    expected_total  = adj_home_rpg + adj_away_rpg

    confidence_factors.append(
        f"Proj total={expected_total:.1f} ({game.home_team} {adj_home_rpg:.1f} / {game.away_team} {adj_away_rpg:.1f})"
    )

    # ── Emit markets ──────────────────────────────────────────────────────────
    results.extend(_build_h2h(game, home_wp, away_wp, scout_date, confidence_factors, risk_factors))
    results.extend(_build_run_line(game, expected_margin, sp_era_diff, scout_date, confidence_factors, risk_factors))
    results.extend(_build_total(game, expected_total, scout_date, confidence_factors, risk_factors))

    return results
