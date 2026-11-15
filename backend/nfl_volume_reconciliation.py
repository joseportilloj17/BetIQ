"""
nfl_volume_reconciliation.py — Phase 2b: Team volume model + share allocation.

Motivation: TB 24 – DAL 16 (2026 W5) exposed that the prop model projects
each player independently, allowing 4 TB receivers to be projected over
simultaneously because the QB passing-yard downgrade was never cascaded into
receiver shares. The CeeDee Lamb miss in that game (model 99.6, actual 9)
was coverage-driven (11.6% target share vs 25-30% expected); the *coverage
layer* addresses that, not this share model.

Checkpoint 1: Team volume predictions
  - Multivariate regression: rolling team pass/rush/total + opp defense
    + spread + total + is_home + backup_qb flag
  - Walk-forward validation 2023-2025 vs actuals
  - Beats league-mean baseline on pass_att, rush_att, pass_rate

Checkpoint 2: Joint share model (target + carry shares)
  - Player-specific EWMA priors (alpha=0.15, >=4 game history)
  - Other-pool estimation (blended rolling + independent window)
  - Named-player shares scaled to (1 − other-pool), explicit _OTHER bucket
  - Participation-probability weighting for sporadic players
  - Injury/inactive redistribution

Usage:
    python nfl_volume_reconciliation.py validate-volume
    python nfl_volume_reconciliation.py validate-shares
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from math import sqrt

import numpy as np

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")
_NFL_DB = os.path.join(_DATA_DIR, "nfl.db")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_team_game_data(seasons: list[int] | None = None) -> list[dict]:
    """Load team-game rows joined with game-level context."""
    db = sqlite3.connect(_NFL_DB)
    db.row_factory = sqlite3.Row

    where = ""
    if seasons:
        where = f"AND g.season IN ({','.join(str(s) for s in seasons)})"

    rows = db.execute(f"""
        SELECT tgs.game_id, tgs.team, tgs.off_plays, tgs.pass_plays,
               tgs.rush_plays, tgs.pass_rate, tgs.qb_name, tgs.qb_dropbacks,
               g.season, g.week, g.spread_line, g.total_line,
               g.home_team, g.away_team, g.home_score, g.away_score,
               g.home_rest, g.away_rest
        FROM nfl_team_game_stats tgs
        JOIN nfl_games g ON tgs.game_id = g.game_id
        WHERE tgs.off_plays IS NOT NULL
          AND g.season >= 2020
          {where}
        ORDER BY g.season, g.week
    """).fetchall()
    db.close()

    result = []
    for r in rows:
        is_home = r["team"] == r["home_team"]
        team_spread = r["spread_line"] if is_home else -r["spread_line"]
        team_score = (r["home_score"] if is_home else r["away_score"]) or 0
        opp_score = (r["away_score"] if is_home else r["home_score"]) or 0

        result.append({
            "game_id": r["game_id"],
            "team": r["team"],
            "opp": r["away_team"] if is_home else r["home_team"],
            "season": r["season"],
            "week": r["week"],
            "is_home": is_home,
            "off_plays": r["off_plays"],
            "pass_plays": r["pass_plays"],
            "rush_plays": r["rush_plays"],
            "pass_rate": r["pass_rate"],
            "qb_name": r["qb_name"],
            "qb_dropbacks": r["qb_dropbacks"],
            "spread": team_spread,
            "total": r["total_line"],
            "team_score": team_score,
            "opp_score": opp_score,
            "margin": team_score - opp_score,
            "rest_days": r["home_rest"] if is_home else r["away_rest"],
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Team volume model: multivariate regression on rolling + game features
# ══════════════════════════════════════════════════════════════════════════════

_LEAGUE_MEANS = {
    "pass_att": 35.4,
    "rush_att": 26.2,
    "total_plays": 62.5,
    "pass_rate": 0.566,
}
_ROLLING_WINDOW = 8
_SHRINK_K = 6

_FEATURE_NAMES = [
    "intercept", "roll_pass_att", "roll_rush_att", "roll_pass_rate",
    "opp_pass_att_allowed", "opp_rush_att_allowed",
    "spread", "total_c", "is_home", "is_backup_qb",
]
_N_FEATURES = len(_FEATURE_NAMES)


def _rolling_features(games: list[dict], team: str,
                      as_of_season: int, as_of_week: int) -> dict | None:
    """Compute rolling features for a team from prior games."""
    recent = [g for g in games
              if g["team"] == team
              and (g["season"], g["week"]) < (as_of_season, as_of_week)]
    recent = sorted(recent, key=lambda g: (g["season"], g["week"]),
                    reverse=True)[:_ROLLING_WINDOW]

    n = len(recent)
    if n == 0:
        return None

    raw_pa = np.mean([g["pass_plays"] for g in recent])
    raw_ra = np.mean([g["rush_plays"] for g in recent])
    raw_tp = np.mean([g["off_plays"] for g in recent])
    raw_pr = np.mean([g["pass_rate"] for g in recent])

    def shrink(raw, league, n_obs):
        return (n_obs * raw + _SHRINK_K * league) / (n_obs + _SHRINK_K)

    return {
        "roll_pass_att": shrink(raw_pa, _LEAGUE_MEANS["pass_att"], n),
        "roll_rush_att": shrink(raw_ra, _LEAGUE_MEANS["rush_att"], n),
        "roll_total": shrink(raw_tp, _LEAGUE_MEANS["total_plays"], n),
        "roll_pass_rate": shrink(raw_pr, _LEAGUE_MEANS["pass_rate"], n),
        "n_games": n,
    }


def _opp_defensive_features(games: list[dict], opp: str,
                             as_of_season: int, as_of_week: int) -> dict:
    """Opponent's defensive volume allowed (how many pass/rush plays
    opponents run against this defense)."""
    opp_def = [g for g in games
               if g["opp"] == opp
               and (g["season"], g["week"]) < (as_of_season, as_of_week)]
    opp_def = sorted(opp_def, key=lambda g: (g["season"], g["week"]),
                     reverse=True)[:_ROLLING_WINDOW]

    n = len(opp_def)
    if n == 0:
        return {
            "opp_pass_att_allowed": _LEAGUE_MEANS["pass_att"],
            "opp_rush_att_allowed": _LEAGUE_MEANS["rush_att"],
            "n_games": 0,
        }

    raw_pa = np.mean([g["pass_plays"] for g in opp_def])
    raw_ra = np.mean([g["rush_plays"] for g in opp_def])

    def shrink(raw, league, n_obs):
        return (n_obs * raw + _SHRINK_K * league) / (n_obs + _SHRINK_K)

    return {
        "opp_pass_att_allowed": shrink(raw_pa, _LEAGUE_MEANS["pass_att"], n),
        "opp_rush_att_allowed": shrink(raw_ra, _LEAGUE_MEANS["rush_att"], n),
        "n_games": n,
    }


def _build_feature_row(team_roll: dict, opp_def: dict,
                       spread: float, total: float,
                       is_home: bool, is_backup_qb: bool) -> np.ndarray:
    """Build feature vector for volume regression (centered)."""
    return np.array([
        1.0,
        team_roll["roll_pass_att"] - _LEAGUE_MEANS["pass_att"],
        team_roll["roll_rush_att"] - _LEAGUE_MEANS["rush_att"],
        team_roll["roll_pass_rate"] - _LEAGUE_MEANS["pass_rate"],
        opp_def["opp_pass_att_allowed"] - _LEAGUE_MEANS["pass_att"],
        opp_def["opp_rush_att_allowed"] - _LEAGUE_MEANS["rush_att"],
        spread,
        total - 44.0,
        1.0 if is_home else 0.0,
        1.0 if is_backup_qb else 0.0,
    ])


def _is_backup_qb(games: list[dict], qb_name: str,
                   as_of_season: int, as_of_week: int) -> bool:
    """Heuristic: QB with < 8 prior starts is a backup."""
    if not qb_name:
        return True
    prior = [g for g in games
             if g["qb_name"] == qb_name
             and (g["season"], g["week"]) < (as_of_season, as_of_week)]
    return len(prior) < 8


def _fit_volume_model(train_X: list[np.ndarray],
                      train_y: dict[str, list[float]]) -> dict:
    """Fit ridge-regularised OLS for pass_att, rush_att, total_plays, pass_rate."""
    if len(train_X) < 50:
        return {"fitted": False, "n_train": len(train_X)}

    X = np.array(train_X)
    XtX = X.T @ X + 0.01 * np.eye(_N_FEATURES)

    models = {}
    for target_name in ("pass_att", "rush_att", "total_plays", "pass_rate"):
        y = np.array(train_y[target_name])
        try:
            beta = np.linalg.solve(XtX, X.T @ y)
            models[target_name] = beta
        except np.linalg.LinAlgError:
            models[target_name] = None

    return {"fitted": True, "models": models, "n_train": len(train_X)}


# ══════════════════════════════════════════════════════════════════════════════
# Top-level team volume projection
# ══════════════════════════════════════════════════════════════════════════════

def project_team_volume(all_games: list[dict], team: str, opp: str,
                        spread: float, total: float, is_home: bool,
                        qb_name: str | None = None,
                        as_of_season: int = 2025,
                        as_of_week: int = 1,
                        _cached_model: dict | None = None) -> dict:
    """Project team volume for an upcoming game.

    Returns: {total_plays, pass_attempts, rush_attempts, pass_rate,
              implied_team_total, components}
    """
    team_roll = _rolling_features(all_games, team, as_of_season, as_of_week)
    opp_def = _opp_defensive_features(all_games, opp, as_of_season, as_of_week)

    if team_roll is None:
        team_roll = {k: _LEAGUE_MEANS.get(k.replace("roll_", ""), 0)
                     for k in ("roll_pass_att", "roll_rush_att",
                               "roll_total", "roll_pass_rate")}
        team_roll["n_games"] = 0

    is_backup = _is_backup_qb(all_games, qb_name, as_of_season, as_of_week)
    x = _build_feature_row(team_roll, opp_def, spread, total, is_home, is_backup)

    # Use cached model or fit fresh
    vol_model = _cached_model
    if vol_model is None:
        # Build training set from all prior data
        train_X, train_y = _build_training_set(all_games, as_of_season, as_of_week)
        vol_model = _fit_volume_model(train_X, train_y)

    if vol_model.get("fitted") and vol_model["models"].get("pass_att") is not None:
        pred_pass = float(x @ vol_model["models"]["pass_att"])
        pred_rush = float(x @ vol_model["models"]["rush_att"])
        pred_plays = float(x @ vol_model["models"]["total_plays"])
        pred_pr = float(x @ vol_model["models"]["pass_rate"])
    else:
        pred_pass = team_roll["roll_pass_att"]
        pred_rush = team_roll["roll_rush_att"]
        pred_plays = team_roll["roll_total"]
        pred_pr = team_roll["roll_pass_rate"]

    pred_pass = max(15, min(60, pred_pass))
    pred_rush = max(10, min(50, pred_rush))
    pred_plays = max(40, min(85, pred_plays))
    pred_pr = max(0.35, min(0.80, pred_pr))

    implied_team_total = (total / 2) - (spread / 2)

    return {
        "team": team,
        "opp": opp,
        "total_plays": round(pred_plays, 1),
        "pass_attempts": round(pred_pass),
        "rush_attempts": round(pred_rush),
        "pass_rate": round(pred_pr, 3),
        "implied_team_total": round(implied_team_total, 1),
        "team_rolling": team_roll,
        "opp_defense": opp_def,
        "is_backup_qb": is_backup,
        "model_n_train": vol_model.get("n_train", 0),
    }


def _build_training_set(all_games: list[dict],
                        as_of_season: int, as_of_week: int
                        ) -> tuple[list[np.ndarray], dict[str, list[float]]]:
    """Build feature matrix + targets for training."""
    train_games = [g for g in all_games
                   if (g["season"], g["week"]) < (as_of_season, as_of_week)]

    X_rows = []
    y = {"pass_att": [], "rush_att": [], "total_plays": [], "pass_rate": []}

    for g in train_games:
        team_roll = _rolling_features(all_games, g["team"],
                                       g["season"], g["week"])
        if team_roll is None or team_roll["n_games"] < 3:
            continue

        opp_def = _opp_defensive_features(all_games, g["opp"],
                                           g["season"], g["week"])
        is_backup = _is_backup_qb(all_games, g["qb_name"],
                                   g["season"], g["week"])

        row = _build_feature_row(team_roll, opp_def,
                                  g["spread"], g["total"],
                                  g["is_home"], is_backup)
        X_rows.append(row)
        y["pass_att"].append(g["pass_plays"])
        y["rush_att"].append(g["rush_plays"])
        y["total_plays"].append(g["off_plays"])
        y["pass_rate"].append(g["pass_rate"])

    return X_rows, y


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_team_volume(seasons: list[int] | None = None,
                         verbose: bool = True) -> dict:
    """Walk-forward validate team volume predictions vs actuals.

    For each (season, week), trains on all prior data, predicts team
    pass_attempts and rush_attempts, compares to actuals.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    all_games = _load_team_game_data()

    predictions = []
    prev_model_key = None
    cached_model = None

    for s in seasons:
        weeks_in_season = sorted(set(g["week"] for g in all_games
                                     if g["season"] == s))
        for w in weeks_in_season:
            if w < 2:
                continue

            # Cache the model per (season, week) — same for all teams
            model_key = (s, w)
            if model_key != prev_model_key:
                train_X, train_y = _build_training_set(all_games, s, w)
                cached_model = _fit_volume_model(train_X, train_y)
                prev_model_key = model_key

            week_games = [g for g in all_games
                          if g["season"] == s and g["week"] == w]

            for g in week_games:
                vol = project_team_volume(
                    all_games, g["team"], g["opp"],
                    g["spread"], g["total"], g["is_home"],
                    qb_name=g["qb_name"],
                    as_of_season=s, as_of_week=w,
                    _cached_model=cached_model)

                predictions.append({
                    "game_id": g["game_id"],
                    "team": g["team"],
                    "season": s,
                    "week": w,
                    "pred_pass_att": vol["pass_attempts"],
                    "pred_rush_att": vol["rush_attempts"],
                    "pred_total_plays": vol["total_plays"],
                    "pred_pass_rate": vol["pass_rate"],
                    "actual_pass": g["pass_plays"],
                    "actual_rush": g["rush_plays"],
                    "actual_plays": g["off_plays"],
                    "actual_pass_rate": g["pass_rate"],
                    "spread": g["spread"],
                    "total": g["total"],
                    "margin": g["margin"],
                    "qb_name": g["qb_name"],
                    "is_backup": vol["is_backup_qb"],
                })

    if not predictions:
        print("No predictions generated")
        return {}

    pred_pass = np.array([p["pred_pass_att"] for p in predictions])
    actual_pass = np.array([p["actual_pass"] for p in predictions])
    pred_rush = np.array([p["pred_rush_att"] for p in predictions])
    actual_rush = np.array([p["actual_rush"] for p in predictions])
    pred_plays = np.array([p["pred_total_plays"] for p in predictions])
    actual_plays = np.array([p["actual_plays"] for p in predictions])
    pred_pr = np.array([p["pred_pass_rate"] for p in predictions])
    actual_pr = np.array([p["actual_pass_rate"] for p in predictions])

    def _metrics(pred, actual, label):
        err = pred - actual
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        bias = np.mean(err)
        corr = np.corrcoef(pred, actual)[0, 1]
        return {"label": label, "mae": mae, "rmse": rmse,
                "bias": bias, "corr": corr, "n": len(pred)}

    results = {
        "pass_att": _metrics(pred_pass, actual_pass, "Pass Attempts"),
        "rush_att": _metrics(pred_rush, actual_rush, "Rush Attempts"),
        "total_plays": _metrics(pred_plays, actual_plays, "Total Plays"),
        "pass_rate": _metrics(pred_pr, actual_pr, "Pass Rate"),
        "n_predictions": len(predictions),
    }

    if verbose:
        print(f"\n{'='*70}")
        print("TEAM VOLUME MODEL — WALK-FORWARD VALIDATION")
        print(f"{'='*70}")
        print(f"Seasons: {seasons}, N = {len(predictions)} team-games")
        print(f"Features: {_FEATURE_NAMES}")
        print()

        # ── Model metrics ──
        print(f"  {'Metric':<16} {'MAE':>7} {'RMSE':>7} {'Bias':>7} "
              f"{'Corr':>7}")
        print("  " + "-" * 50)
        for key in ["pass_att", "rush_att", "total_plays", "pass_rate"]:
            m = results[key]
            fmt = ".1f" if key != "pass_rate" else ".3f"
            print(f"  {m['label']:<16} {m['mae']:>7{fmt}} {m['rmse']:>7{fmt}} "
                  f"{m['bias']:>+7{fmt}} {m['corr']:>7.3f}")

        # ── Naive baseline: league mean ──
        print(f"\n  Baseline (league mean):")
        print(f"  {'Metric':<16} {'MAE':>7} {'RMSE':>7}")
        print("  " + "-" * 35)
        for pred_arr, actual_arr, label in [
            (np.full_like(actual_pass, np.mean(actual_pass), dtype=float),
             actual_pass, "Pass Att"),
            (np.full_like(actual_rush, np.mean(actual_rush), dtype=float),
             actual_rush, "Rush Att"),
            (np.full_like(actual_plays, np.mean(actual_plays), dtype=float),
             actual_plays, "Total Plays"),
        ]:
            mae = np.mean(np.abs(pred_arr - actual_arr))
            rmse = np.sqrt(np.mean((pred_arr - actual_arr) ** 2))
            print(f"  {label:<16} {mae:>7.1f} {rmse:>7.1f}")

        # ── Regression coefficients ──
        train_X, train_y = _build_training_set(all_games, 2026, 1)
        full_model = _fit_volume_model(train_X, train_y)
        if full_model.get("fitted"):
            print(f"\n  Regression coefficients (full data, N={full_model['n_train']}):")
            print(f"  {'Feature':<22} {'pass_att':>9} {'rush_att':>9} "
                  f"{'total':>9} {'pass_rate':>9}")
            print("  " + "-" * 62)
            for i, feat in enumerate(_FEATURE_NAMES):
                vals = []
                for tgt in ("pass_att", "rush_att", "total_plays", "pass_rate"):
                    b = full_model["models"][tgt]
                    vals.append(f"{b[i]:+9.4f}" if b is not None else "     N/A")
                print(f"  {feat:<22} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

        # ── By spread bucket ──
        print(f"\n  Pass rate error by spread bucket:")
        print(f"  {'Spread':>10} {'N':>5} {'MAE':>7} {'Bias':>7}")
        print("  " + "-" * 35)
        buckets = [(-99, -7, "FAV ≤-7"), (-7, -3, "-7 to -3"),
                   (-3, 3, "±3"), (3, 7, "+3 to +7"), (7, 99, "DOG ≥+7")]
        for lo, hi, label in buckets:
            mask = [(lo < p["spread"] <= hi) for p in predictions]
            if sum(mask) < 10:
                continue
            idx = np.where(mask)[0]
            mae = np.mean(np.abs(pred_pr[idx] - actual_pr[idx]))
            bias = np.mean(pred_pr[idx] - actual_pr[idx])
            print(f"  {label:>10} {sum(mask):>5} {mae:>7.3f} {bias:>+7.3f}")

        # ── Backup QB performance ──
        backup_mask = np.array([p["is_backup"] for p in predictions])
        n_backup = int(backup_mask.sum())
        if n_backup > 10:
            s_idx = np.where(~backup_mask)[0]
            b_idx = np.where(backup_mask)[0]
            print(f"\n  Starter vs backup QB:")
            print(f"  {'QB Type':<12} {'N':>5} {'Pass MAE':>9} "
                  f"{'Rush MAE':>9} {'PR MAE':>7}")
            print("  " + "-" * 45)
            for label, idx in [("Starter", s_idx), ("Backup", b_idx)]:
                p_mae = np.mean(np.abs(pred_pass[idx] - actual_pass[idx]))
                r_mae = np.mean(np.abs(pred_rush[idx] - actual_rush[idx]))
                pr_mae = np.mean(np.abs(pred_pr[idx] - actual_pr[idx]))
                print(f"  {label:<12} {len(idx):>5} {p_mae:>9.1f} "
                      f"{r_mae:>9.1f} {pr_mae:>7.3f}")

        # ── Prediction variance check ──
        print(f"\n  Prediction spread:")
        print(f"    Pred pass att: [{pred_pass.min():.0f}, {pred_pass.max():.0f}], "
              f"std={pred_pass.std():.1f}")
        print(f"    Actual pass att: [{actual_pass.min():.0f}, {actual_pass.max():.0f}], "
              f"std={actual_pass.std():.1f}")
        print(f"    Pred rush att: [{pred_rush.min():.0f}, {pred_rush.max():.0f}], "
              f"std={pred_rush.std():.1f}")
        print(f"    Actual rush att: [{actual_rush.min():.0f}, {actual_rush.max():.0f}], "
              f"std={actual_rush.std():.1f}")

    # ── Game-clustered RMSE CI ──
    by_game = defaultdict(list)
    for p in predictions:
        by_game[p["game_id"]].append(p)

    rng = np.random.default_rng(42)
    gk_list = list(by_game.keys())

    boot_pass_rmse = []
    boot_rush_rmse = []
    for _ in range(2000):
        sampled = rng.choice(len(gk_list), size=len(gk_list), replace=True)
        b_pass_errs = []
        b_rush_errs = []
        for idx in sampled:
            for p in by_game[gk_list[idx]]:
                b_pass_errs.append((p["pred_pass_att"] - p["actual_pass"]) ** 2)
                b_rush_errs.append((p["pred_rush_att"] - p["actual_rush"]) ** 2)
        boot_pass_rmse.append(sqrt(np.mean(b_pass_errs)))
        boot_rush_rmse.append(sqrt(np.mean(b_rush_errs)))

    results["pass_att_rmse_ci"] = (
        round(np.percentile(boot_pass_rmse, 2.5), 2),
        round(np.percentile(boot_pass_rmse, 97.5), 2),
    )
    results["rush_att_rmse_ci"] = (
        round(np.percentile(boot_rush_rmse, 2.5), 2),
        round(np.percentile(boot_rush_rmse, 97.5), 2),
    )

    if verbose:
        print(f"\n  Game-clustered RMSE 95% CI:")
        print(f"    Pass att: [{results['pass_att_rmse_ci'][0]:.1f}, "
              f"{results['pass_att_rmse_ci'][1]:.1f}]")
        print(f"    Rush att: [{results['rush_att_rmse_ci'][0]:.1f}, "
              f"{results['rush_att_rmse_ci'][1]:.1f}]")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Joint share model: target + carry shares per team
# ══════════════════════════════════════════════════════════════════════════════

_SHARE_ROLLING_WINDOW = 4     # games for rolling share computation
_SHARE_SHRINK_K = 1.5         # shrinkage strength toward position mean
_POSITION_MEAN_TARGET_SHARE = {
    "WR": 0.16,   # default (overridden by rank priors below)
    "TE": 0.14,
    "RB": 0.08,
}
_POSITION_MEAN_CARRY_SHARE = {
    "RB": 0.40,   # each RB averages ~40% of team carries
}
# Rank-aware target share priors (within each position group per team)
_WR_RANK_TARGET_PRIOR = {1: 0.24, 2: 0.16, 3: 0.10}  # default to 0.05
_RB_RANK_CARRY_PRIOR = {1: 0.55, 2: 0.30}             # default to 0.10


def _build_position_map(seasons: list[int]) -> dict[str, str]:
    """Build player_name → position mapping from weekly rosters.

    PBP uses abbreviated names (T.Kelce), so we build a lookup keyed by
    (first_initial.last_name, team) → position.
    """
    import nfl_data_py as nfl_data
    roster = nfl_data.import_weekly_rosters(seasons)
    roster = roster[roster["position"].isin(["WR", "TE", "RB", "QB"])]
    roster = roster[["player_name", "position", "team"]].drop_duplicates()

    pos_map = {}
    for _, row in roster.iterrows():
        full_name = str(row["player_name"])
        pos = row["position"]
        team = row["team"]
        # Build abbreviated key: "T.Kelce"
        parts = full_name.split()
        if len(parts) >= 2:
            abbrev = parts[0][0] + "." + parts[-1]
            pos_map[(abbrev, team)] = pos
        # Also store full name
        pos_map[(full_name, team)] = pos

    return pos_map


def _load_pbp_shares(seasons: list[int]) -> tuple[dict, dict]:
    """Load PBP-based target and carry shares per player per team-week.

    Returns: (shares_dict, position_map)
      shares_dict: {(team, season, week): {
        "targets": {player_name: {targets, target_share, position}},
        "carries": {player_name: {carries, carry_share, position}},
        "team_targets": int, "team_carries": int,
      }}
      position_map: {player_name → position} (PBP abbreviated names)
    """
    import nfl_data_py as nfl_data

    pos_map = _build_position_map(seasons)
    pbp = nfl_data.import_pbp_data(seasons)
    result = {}

    def _get_pos(name, team):
        """Get position from map, with fallback."""
        p = pos_map.get((name, team))
        if p:
            return p
        # Try without team
        for (n, t), pos in pos_map.items():
            if n == name:
                return pos
        return None

    # Targets
    pass_plays = pbp[(pbp["pass_attempt"] == 1) &
                     (pbp["receiver_player_name"].notna())]
    tgt_grp = pass_plays.groupby(
        ["posteam", "season", "week", "receiver_player_name"]
    ).agg(targets=("pass_attempt", "sum")).reset_index()
    team_tgt = tgt_grp.groupby(
        ["posteam", "season", "week"]
    )["targets"].sum().reset_index(name="team_targets")
    tgt_grp = tgt_grp.merge(team_tgt, on=["posteam", "season", "week"])
    tgt_grp["target_share"] = tgt_grp["targets"] / tgt_grp["team_targets"]

    # Carries
    rushes = pbp[(pbp["rush_attempt"] == 1) &
                 (pbp["rusher_player_name"].notna())]
    carry_grp = rushes.groupby(
        ["posteam", "season", "week", "rusher_player_name"]
    ).agg(carries=("rush_attempt", "sum")).reset_index()
    team_car = carry_grp.groupby(
        ["posteam", "season", "week"]
    )["carries"].sum().reset_index(name="team_carries")
    carry_grp = carry_grp.merge(team_car, on=["posteam", "season", "week"])
    carry_grp["carry_share"] = carry_grp["carries"] / carry_grp["team_carries"]

    # Build position lookup from PBP names
    pbp_pos = {}  # (name, team) → position
    for _, row in tgt_grp.iterrows():
        name = row["receiver_player_name"]
        team = row["posteam"]
        if (name, team) not in pbp_pos:
            pos = _get_pos(name, team)
            pbp_pos[(name, team)] = pos or "WR"  # default receivers to WR

    for _, row in carry_grp.iterrows():
        name = row["rusher_player_name"]
        team = row["posteam"]
        if (name, team) not in pbp_pos:
            pos = _get_pos(name, team)
            pbp_pos[(name, team)] = pos or "RB"  # default rushers to RB

    for _, row in team_tgt.iterrows():
        key = (row["posteam"], int(row["season"]), int(row["week"]))
        result.setdefault(key, {
            "targets": {}, "carries": {},
            "team_targets": int(row["team_targets"]),
            "team_carries": 0,
        })

    for _, row in team_car.iterrows():
        key = (row["posteam"], int(row["season"]), int(row["week"]))
        result.setdefault(key, {
            "targets": {}, "carries": {},
            "team_targets": 0,
            "team_carries": int(row["team_carries"]),
        })
        result[key]["team_carries"] = int(row["team_carries"])

    for _, row in tgt_grp.iterrows():
        key = (row["posteam"], int(row["season"]), int(row["week"]))
        name = row["receiver_player_name"]
        result[key]["targets"][name] = {
            "targets": int(row["targets"]),
            "target_share": float(row["target_share"]),
            "position": pbp_pos.get((name, row["posteam"]), "WR"),
        }

    for _, row in carry_grp.iterrows():
        key = (row["posteam"], int(row["season"]), int(row["week"]))
        name = row["rusher_player_name"]
        result[key]["carries"][name] = {
            "carries": int(row["carries"]),
            "carry_share": float(row["carry_share"]),
            "position": pbp_pos.get((name, row["posteam"]), "RB"),
        }

    return result, pbp_pos


def _load_snap_shares(seasons: list[int]) -> dict:
    """Load snap count shares per player per team-week.

    Returns: {(team, season, week): {player_name: {
        offense_snaps, snap_share, position
    }}}
    """
    import nfl_data_py as nfl_data

    snaps = nfl_data.import_snap_counts(seasons)
    reg = snaps[snaps["week"] <= 18].copy()

    # Team max snaps per game
    team_max = reg.groupby(["team", "season", "week"])[
        "offense_snaps"].max().reset_index(name="team_snaps")
    reg = reg.merge(team_max, on=["team", "season", "week"])
    reg["snap_share"] = reg["offense_snaps"] / reg["team_snaps"].clip(lower=1)

    result = {}
    for _, row in reg.iterrows():
        if row["offense_snaps"] is None or row["offense_snaps"] <= 0:
            continue
        key = (row["team"], int(row["season"]), int(row["week"]))
        result.setdefault(key, {})
        result[key][row["player"]] = {
            "offense_snaps": float(row["offense_snaps"]),
            "snap_share": float(row["snap_share"]),
            "position": row["position"],
        }

    return result


_EWMA_ALPHA = 0.15            # EWMA decay for player-specific priors
_EWMA_MIN_GAMES = 4           # minimum games to use player EWMA vs rank prior


def compute_player_shares(pbp_shares: dict,
                          team: str, season: int, week: int,
                          inactive_players: list[str] | None = None,
                          ) -> dict:
    """Compute predicted target/carry shares for a team-week.

    Uses rolling PBP shares with player-specific EWMA priors.
    Scales named-player shares to (1 − estimated other-pool) and
    includes an explicit "_OTHER" bucket.

    Returns: {player_name: {
        "pred_target_share": float,
        "pred_carry_share": float,
        "n_weeks": int,
        "position": str,
        "source": "rolling" | "ewma" | "rank_prior",
    }}
    """
    # ── 1. ALL prior keys for this team (chronological, newest first) ──
    all_prior_keys = [(team, s, w) for (t, s, w) in pbp_shares
                      if t == team and (s, w) < (season, week)]
    all_prior_keys = sorted(all_prior_keys, key=lambda x: (x[1], x[2]),
                            reverse=True)

    rolling_keys = all_prior_keys[:_SHARE_ROLLING_WINDOW]   # recent 4
    history_keys = all_prior_keys[_SHARE_ROLLING_WINDOW:]   # older data

    # ── 2. Build player data from rolling window ──
    players = {}

    for pk in rolling_keys:
        week_data = pbp_shares.get(pk, {})
        for pname, tdata in week_data.get("targets", {}).items():
            players.setdefault(pname, {"target_shares": [], "carry_shares": [],
                                        "position": None,
                                        "total_targets": 0,
                                        "total_carries": 0})
            players[pname]["target_shares"].append(tdata["target_share"])
            players[pname]["total_targets"] += tdata.get("targets", 1)
            if players[pname]["position"] is None and tdata.get("position"):
                players[pname]["position"] = tdata["position"]

        for pname, cdata in week_data.get("carries", {}).items():
            players.setdefault(pname, {"target_shares": [], "carry_shares": [],
                                        "position": None,
                                        "total_targets": 0,
                                        "total_carries": 0})
            players[pname]["carry_shares"].append(cdata["carry_share"])
            players[pname]["total_carries"] += cdata.get("carries", 1)
            if players[pname]["position"] is None and cdata.get("position"):
                players[pname]["position"] = cdata["position"]

    # ── 3. Filter: appeared with targets/carries in ≥2 of 4 rolling weeks ──
    # Using game appearances (not raw target count) to exclude one-game flukes
    players = {p: d for p, d in players.items()
               if len(d["target_shares"]) >= 2 or len(d["carry_shares"]) >= 2}

    # Resolve positions
    for pname, pdata in players.items():
        pos = pdata["position"]
        if pos is None or pos == "QB":
            if pdata["total_carries"] > pdata["total_targets"]:
                pos = "RB"
            else:
                pos = "WR"
            pdata["position"] = pos

    # ── 4. Player-specific EWMA priors from history ──
    # Walk history chronologically (oldest → newest) to compute EWMA
    player_ewma = {}
    for pk in sorted(history_keys, key=lambda x: (x[1], x[2])):
        week_data = pbp_shares.get(pk, {})
        for pname, tdata in week_data.get("targets", {}).items():
            if pname not in player_ewma:
                player_ewma[pname] = {"ts": tdata["target_share"],
                                       "ts_n": 1, "cs": None, "cs_n": 0}
            else:
                e = player_ewma[pname]
                if e["ts"] is None:
                    e["ts"] = tdata["target_share"]
                else:
                    e["ts"] = (_EWMA_ALPHA * tdata["target_share"]
                               + (1 - _EWMA_ALPHA) * e["ts"])
                e["ts_n"] += 1

        for pname, cdata in week_data.get("carries", {}).items():
            if pname not in player_ewma:
                player_ewma[pname] = {"ts": None, "ts_n": 0,
                                       "cs": cdata["carry_share"], "cs_n": 1}
            else:
                e = player_ewma[pname]
                if e["cs"] is None:
                    e["cs"] = cdata["carry_share"]
                else:
                    e["cs"] = (_EWMA_ALPHA * cdata["carry_share"]
                               + (1 - _EWMA_ALPHA) * e["cs"])
                e["cs_n"] += 1

    # ── 5. Compute raw rolling means + rank for fallback priors ──
    raw_means = {}
    for pname, pdata in players.items():
        raw_means[pname] = {
            "raw_ts": np.mean(pdata["target_shares"]) if pdata["target_shares"] else 0.0,
            "raw_cs": np.mean(pdata["carry_shares"]) if pdata["carry_shares"] else 0.0,
        }

    wr_players = [(p, raw_means[p]["raw_ts"])
                  for p, d in players.items() if d["position"] == "WR"]
    wr_players.sort(key=lambda x: x[1], reverse=True)
    wr_rank = {p: rank + 1 for rank, (p, _) in enumerate(wr_players)}

    rb_players = [(p, raw_means[p]["raw_cs"])
                  for p, d in players.items() if d["position"] == "RB"]
    rb_players.sort(key=lambda x: x[1], reverse=True)
    rb_rank = {p: rank + 1 for rank, (p, _) in enumerate(rb_players)}

    # ── 6. Compute raw priors per player ──
    player_priors = {}
    for pname, pdata in players.items():
        pos = pdata["position"]
        ewma = player_ewma.get(pname)

        # Target share prior
        if ewma and ewma.get("ts") is not None and ewma["ts_n"] >= _EWMA_MIN_GAMES:
            prior_ts = ewma["ts"]
            prior_source = "ewma"
        elif pos == "WR":
            rank = wr_rank.get(pname, 99)
            prior_ts = _WR_RANK_TARGET_PRIOR.get(rank, 0.05)
            prior_source = "rank_prior"
        else:
            prior_ts = _POSITION_MEAN_TARGET_SHARE.get(pos, 0.10)
            prior_source = "rank_prior"

        # Carry share prior
        if ewma and ewma.get("cs") is not None and ewma["cs_n"] >= _EWMA_MIN_GAMES:
            prior_cs = ewma["cs"]
        elif pos == "RB":
            rank = rb_rank.get(pname, 99)
            prior_cs = _RB_RANK_CARRY_PRIOR.get(rank, 0.10)
        else:
            prior_cs = 0.0

        # Only count target prior for players with target history
        n_tgt = len(pdata["target_shares"])
        player_priors[pname] = {
            "ts": prior_ts if n_tgt > 0 else 0.0,
            "cs": prior_cs if len(pdata["carry_shares"]) > 0 else 0.0,
            "source": prior_source,
        }

    # ── 6b. Estimate other-pool from blended sources ──
    # Rolling window underestimates other pool (selection bias: pred_set
    # was chosen from it). Independent window overestimates (pred_set
    # less established in older data). Blend both for unbiased estimate.
    pred_set = set(players.keys())
    indep_keys = all_prior_keys[_SHARE_ROLLING_WINDOW:_SHARE_ROLLING_WINDOW * 2]

    def _measure_other(keys):
        ts_vals, cs_vals = [], []
        for pk in keys:
            week_data = pbp_shares.get(pk, {})
            named_ts = sum(d["target_share"]
                           for p, d in week_data.get("targets", {}).items()
                           if p in pred_set)
            named_cs = sum(d["carry_share"]
                           for p, d in week_data.get("carries", {}).items()
                           if p in pred_set)
            ts_vals.append(1.0 - named_ts)
            cs_vals.append(1.0 - named_cs)
        return (np.mean(ts_vals) if ts_vals else None,
                np.mean(cs_vals) if cs_vals else None)

    roll_ot, roll_oc = _measure_other(rolling_keys)
    indep_ot, indep_oc = _measure_other(indep_keys) if indep_keys else (None, None)

    # Blend: if both available, 50/50; otherwise use what we have
    if roll_ot is not None and indep_ot is not None:
        est_other_ts = 0.5 * roll_ot + 0.5 * indep_ot
        est_other_cs = 0.5 * roll_oc + 0.5 * indep_oc
    elif roll_ot is not None:
        est_other_ts = roll_ot
        est_other_cs = roll_oc
    else:
        est_other_ts = 0.09
        est_other_cs = 0.10

    est_other_ts = max(0.03, min(0.30, est_other_ts))
    est_other_cs = max(0.03, min(0.40, est_other_cs))

    target_ceiling = 1.0 - est_other_ts
    carry_ceiling = 1.0 - est_other_cs

    prior_ts_sum = sum(p["ts"] for p in player_priors.values())
    prior_cs_sum = sum(p["cs"] for p in player_priors.values())

    if prior_ts_sum > 0:
        ts_prior_scale = target_ceiling / prior_ts_sum
        for pp in player_priors.values():
            pp["ts"] *= ts_prior_scale

    if prior_cs_sum > 0:
        cs_prior_scale = carry_ceiling / prior_cs_sum
        for pp in player_priors.values():
            pp["cs"] *= cs_prior_scale

    # ── 7. Shrinkage with normalized priors ──
    result = {}
    for pname, pdata in players.items():
        pos = pdata["position"]
        n_tgt = len(pdata["target_shares"])
        n_car = len(pdata["carry_shares"])
        pp = player_priors[pname]

        if n_tgt > 0:
            raw_ts = raw_means[pname]["raw_ts"]
            pred_ts = ((n_tgt * raw_ts + _SHARE_SHRINK_K * pp["ts"])
                       / (n_tgt + _SHARE_SHRINK_K))
        else:
            pred_ts = 0.0

        if n_car > 0:
            raw_cs = raw_means[pname]["raw_cs"]
            pred_cs = ((n_car * raw_cs + _SHARE_SHRINK_K * pp["cs"])
                       / (n_car + _SHARE_SHRINK_K))
        else:
            pred_cs = 0.0

        result[pname] = {
            "pred_target_share": pred_ts,
            "pred_carry_share": pred_cs,
            "position": pos,
            "n_weeks": max(n_tgt, n_car),
            "source": pp["source"],
        }

    # ── 8. Final scale to (1 − estimated other-pool) ──
    total_ts = sum(r["pred_target_share"] for r in result.values())
    total_cs = sum(r["pred_carry_share"] for r in result.values()
                   if r["position"] == "RB")

    if total_ts > 0:
        ts_scale = target_ceiling / total_ts
        for r in result.values():
            r["pred_target_share"] *= ts_scale

    if total_cs > 0:
        cs_scale = carry_ceiling / total_cs
        for r in result.values():
            if r["position"] == "RB":
                r["pred_carry_share"] *= cs_scale

    # ── 8b. Participation-probability weighting ──
    # Weight each player's share by P(gets targets this week), estimated
    # from fraction of rolling weeks they appeared. Renormalize after.
    # This shifts share from sporadic players to regulars without hard caps.
    for pname, r in result.items():
        pdata = players[pname]
        n_rolling = len(rolling_keys)
        if n_rolling > 0:
            n_tgt_games = len(pdata["target_shares"])
            n_car_games = len(pdata["carry_shares"])
            p_targets = n_tgt_games / n_rolling
            p_carries = n_car_games / n_rolling
        else:
            p_targets = 0.5
            p_carries = 0.5
        r["pred_target_share"] *= p_targets
        r["pred_carry_share"] *= p_carries

    # Renormalize to ceiling after weighting
    weighted_ts = sum(r["pred_target_share"] for r in result.values())
    weighted_cs = sum(r["pred_carry_share"] for r in result.values()
                      if r["position"] == "RB")
    if weighted_ts > 0:
        for r in result.values():
            r["pred_target_share"] *= target_ceiling / weighted_ts
    if weighted_cs > 0:
        for r in result.values():
            if r["position"] == "RB":
                r["pred_carry_share"] *= carry_ceiling / weighted_cs

    # Explicit other bucket
    result["_OTHER"] = {
        "pred_target_share": est_other_ts,
        "pred_carry_share": est_other_cs,
        "position": "OTHER",
        "n_weeks": len(rolling_keys),
        "source": "other_pool",
    }

    # ── 9. Injury redistribution ──
    if inactive_players:
        inactive_set = set(inactive_players)
        freed_ts = sum(result[p]["pred_target_share"]
                       for p in inactive_set if p in result)
        freed_cs = sum(result[p]["pred_carry_share"]
                       for p in inactive_set if p in result)

        active = {p: r for p, r in result.items()
                  if p not in inactive_set and p != "_OTHER"}
        remaining_ts = sum(r["pred_target_share"] for r in active.values())
        remaining_cs = sum(r["pred_carry_share"] for r in active.values()
                          if r["position"] == "RB")

        if remaining_ts > 0 and freed_ts > 0:
            for pname, r in active.items():
                prop = r["pred_target_share"] / remaining_ts
                r["pred_target_share"] += freed_ts * prop

        if remaining_cs > 0 and freed_cs > 0:
            for pname, r in active.items():
                if r["position"] == "RB":
                    prop = r["pred_carry_share"] / remaining_cs
                    r["pred_carry_share"] += freed_cs * prop

        for p in inactive_set:
            if p in result:
                result[p]["pred_target_share"] = 0.0
                result[p]["pred_carry_share"] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward share validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_shares(seasons: list[int] | None = None,
                    verbose: bool = True) -> dict:
    """Walk-forward validate target/carry share predictions.

    For each (team, week), predicts shares from rolling prior data,
    compares to actual PBP shares.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    # Load one extra prior season for EWMA player priors
    load_seasons = [min(seasons) - 1] + list(seasons)
    print(f"Loading PBP data for share validation (seasons {load_seasons})...")
    pbp_shares, _pbp_pos = _load_pbp_shares(load_seasons)
    print(f"Loaded {len(pbp_shares)} team-weeks PBP")

    # Walk-forward: for each team-week ≥ week 5, predict shares
    predictions = []
    other_pool_records = []   # per team-week: other-pool actual share
    all_keys = sorted(pbp_shares.keys(), key=lambda x: (x[1], x[2]))

    for team, season, week in all_keys:
        if week < _SHARE_ROLLING_WINDOW + 1:
            continue

        pred = compute_player_shares(pbp_shares, team, season, week)
        actual = pbp_shares.get((team, season, week), {})
        actual_targets = actual.get("targets", {})
        actual_carries = actual.get("carries", {})

        # Predicted player set (with meaningful predictions, excluding _OTHER)
        pred_set = {p for p, d in pred.items()
                    if p != "_OTHER" and d["n_weeks"] >= 2}

        # Other-pool measurement: actual share going to non-predicted players
        named_actual_ts = sum(
            d["target_share"] for p, d in actual_targets.items()
            if p in pred_set)
        named_actual_cs = sum(
            d["carry_share"] for p, d in actual_carries.items()
            if p in pred_set)
        other_ts = 1.0 - named_actual_ts
        other_cs = 1.0 - named_actual_cs

        # Sum of predicted shares for the named set
        pred_sum_ts = sum(
            pred[p]["pred_target_share"] for p in pred_set
            if pred[p]["pred_target_share"] > 0)
        pred_sum_cs = sum(
            pred[p]["pred_carry_share"] for p in pred_set
            if pred[p]["position"] == "RB")

        other_pool_records.append({
            "team": team, "season": season, "week": week,
            "other_target_share": other_ts,
            "other_carry_share": other_cs,
            "named_actual_ts": named_actual_ts,
            "named_actual_cs": named_actual_cs,
            "pred_sum_ts": pred_sum_ts,
            "pred_sum_cs": pred_sum_cs,
            "n_predicted": len(pred_set),
        })

        for pname, pdata in pred.items():
            if pname == "_OTHER" or pdata["n_weeks"] < 2:
                continue

            # Include actual=0 for predicted players with no targets/carries
            # (ghost players). Model predicts unconditional shares; validation
            # must include 0-target weeks to avoid upward bias in actuals.
            actual_ts = actual_targets.get(pname, {}).get("target_share", 0.0)
            had_targets = pname in actual_targets

            predictions.append({
                "type": "target_share",
                "player": pname,
                "team": team,
                "season": season,
                "week": week,
                "position": pdata["position"],
                "predicted": pdata["pred_target_share"],
                "actual": actual_ts,
                "n_weeks": pdata["n_weeks"],
                "source": pdata.get("source", ""),
                "had_targets": had_targets,
            })

            if pdata["position"] == "RB":
                actual_cs = actual_carries.get(pname, {}).get(
                    "carry_share", 0.0)
                had_carries = pname in actual_carries
                predictions.append({
                    "type": "carry_share",
                    "player": pname,
                    "team": team,
                    "season": season,
                    "week": week,
                    "position": pdata["position"],
                    "predicted": pdata["pred_carry_share"],
                    "actual": actual_cs,
                    "n_weeks": pdata["n_weeks"],
                    "source": pdata.get("source", ""),
                    "had_carries": had_carries,
                })

    if not predictions:
        print("No predictions generated")
        return {}

    # ── Compute metrics ──
    def _share_metrics(preds, label):
        pred_arr = np.array([p["predicted"] for p in preds])
        act_arr = np.array([p["actual"] for p in preds])
        mae = np.mean(np.abs(pred_arr - act_arr))
        rmse = np.sqrt(np.mean((pred_arr - act_arr) ** 2))
        bias = np.mean(pred_arr - act_arr)
        corr = np.corrcoef(pred_arr, act_arr)[0, 1] if len(preds) > 2 else 0
        return {"label": label, "mae": mae, "rmse": rmse,
                "bias": bias, "corr": corr, "n": len(preds)}

    tgt_preds = [p for p in predictions if p["type"] == "target_share"]
    car_preds = [p for p in predictions if p["type"] == "carry_share"]

    results = {
        "target_share_all": _share_metrics(tgt_preds, "Target Share (all)"),
        "carry_share_all": _share_metrics(car_preds, "Carry Share (all)"),
    }

    # By position
    for pos in ["WR", "TE", "RB"]:
        pos_tgt = [p for p in tgt_preds if p["position"] == pos]
        if len(pos_tgt) > 20:
            results[f"target_share_{pos}"] = _share_metrics(
                pos_tgt, f"Target Share ({pos})")

    # Split: conditional (had targets) vs unconditional (all predicted)
    tgt_cond = [p for p in tgt_preds if p.get("had_targets", True)]
    tgt_ghost = [p for p in tgt_preds if not p.get("had_targets", True)]

    if verbose:
        print(f"\n{'='*70}")
        print("JOINT SHARE MODEL — WALK-FORWARD VALIDATION")
        print(f"{'='*70}")
        print(f"Seasons: {seasons}")
        print(f"Target predictions: {len(tgt_preds)} total "
              f"({len(tgt_cond)} had targets, {len(tgt_ghost)} ghost)")
        print(f"Carry predictions: {len(car_preds)}")
        print()

        # Unconditional (all predicted — correct for prop model)
        print(f"  UNCONDITIONAL (includes 0-target weeks):")
        print(f"  {'Metric':<22} {'MAE':>7} {'RMSE':>7} {'Bias':>7} "
              f"{'Corr':>7} {'N':>6}")
        print("  " + "-" * 60)
        for key in sorted(results.keys()):
            m = results[key]
            print(f"  {m['label']:<22} {m['mae']:>7.4f} {m['rmse']:>7.4f} "
                  f"{m['bias']:>+7.4f} {m['corr']:>7.3f} {m['n']:>6}")

        # Conditional (had targets — inflated actuals due to selection)
        cond_results = {}
        cond_results["target_share_cond"] = _share_metrics(
            tgt_cond, "Target Share (cond)")
        print(f"\n  CONDITIONAL (had-targets only — for reference):")
        m = cond_results["target_share_cond"]
        print(f"  {m['label']:<22} {m['mae']:>7.4f} {m['rmse']:>7.4f} "
              f"{m['bias']:>+7.4f} {m['corr']:>7.3f} {m['n']:>6}")

        # Ghost player diagnostics
        print(f"\n  Ghost players (predicted but 0 targets):")
        print(f"    N ghost entries: {len(tgt_ghost)}")
        if other_pool_records:
            n_tw = len(other_pool_records)
            print(f"    Ghosts per team-week: "
                  f"{len(tgt_ghost)/n_tw:.1f} / "
                  f"{len(tgt_preds)/n_tw:.1f} predicted")
        if tgt_ghost:
            ghost_pred = np.array([p["predicted"] for p in tgt_ghost])
            print(f"    Ghost mean predicted share: {ghost_pred.mean():.4f}")
            print(f"    Ghost predicted share p25/p50/p75: "
                  f"{np.percentile(ghost_pred,25):.4f} / "
                  f"{np.percentile(ghost_pred,50):.4f} / "
                  f"{np.percentile(ghost_pred,75):.4f}")
            # Ghost rate by position
            for pos in ["WR", "TE", "RB"]:
                pos_all = [p for p in tgt_preds if p["position"] == pos]
                pos_ghost = [p for p in tgt_ghost if p["position"] == pos]
                if pos_all:
                    print(f"    {pos}: {len(pos_ghost)}/{len(pos_all)} = "
                          f"{100*len(pos_ghost)/len(pos_all):.1f}% ghost rate")

        # ── Naive baseline: position-uniform share ──
        print(f"\n  Baseline (position-mean share, unconditional):")
        print(f"  {'Metric':<22} {'MAE':>7} {'Corr':>7}")
        print("  " + "-" * 40)
        for pos in ["WR", "TE", "RB"]:
            pos_preds = [p for p in tgt_preds if p["position"] == pos]
            if not pos_preds:
                continue
            pos_mean = _POSITION_MEAN_TARGET_SHARE.get(pos, 0.10)
            act = np.array([p["actual"] for p in pos_preds])
            baseline_mae = np.mean(np.abs(act - pos_mean))
            print(f"  Target Share ({pos}){'':<5} {baseline_mae:>7.4f} {'N/A':>7}")

        # ── Other-pool analysis ──
        if other_pool_records:
            otr = other_pool_records
            ot_ts = np.array([r["other_target_share"] for r in otr])
            ot_cs = np.array([r["other_carry_share"] for r in otr])
            na_ts = np.array([r["named_actual_ts"] for r in otr])
            na_cs = np.array([r["named_actual_cs"] for r in otr])
            ps_ts = np.array([r["pred_sum_ts"] for r in otr])
            ps_cs = np.array([r["pred_sum_cs"] for r in otr])
            n_pred = np.array([r["n_predicted"] for r in otr])

            print(f"\n  Other-pool analysis ({len(otr)} team-weeks):")
            print(f"    Other target share: mean={ot_ts.mean():.3f}, "
                  f"SD={ot_ts.std():.3f}, [p5={np.percentile(ot_ts,5):.3f}, "
                  f"p95={np.percentile(ot_ts,95):.3f}]")
            print(f"    Other carry share:  mean={ot_cs.mean():.3f}, "
                  f"SD={ot_cs.std():.3f}, [p5={np.percentile(ot_cs,5):.3f}, "
                  f"p95={np.percentile(ot_cs,95):.3f}]")
            print(f"    Named actual target share: mean={na_ts.mean():.3f}, "
                  f"SD={na_ts.std():.3f}")
            print(f"    Named actual carry share:  mean={na_cs.mean():.3f}, "
                  f"SD={na_cs.std():.3f}")
            print(f"    Predicted sum targets: mean={ps_ts.mean():.3f}, "
                  f"SD={ps_ts.std():.3f}")
            print(f"    Predicted sum carries: mean={ps_cs.mean():.3f}, "
                  f"SD={ps_cs.std():.3f}")
            print(f"    Gap (pred - named actual): "
                  f"targets={ps_ts.mean() - na_ts.mean():+.3f}, "
                  f"carries={ps_cs.mean() - na_cs.mean():+.3f}")
            print(f"    Players per team-week: mean={n_pred.mean():.1f}, "
                  f"min={n_pred.min()}, max={n_pred.max()}")

        # ── Prior source breakdown ──
        ewma_preds = [p for p in tgt_preds if p.get("source") == "ewma"]
        rank_preds = [p for p in tgt_preds if p.get("source") == "rank_prior"]
        print(f"\n  Prior source breakdown:")
        print(f"    EWMA priors: {len(ewma_preds)} predictions "
              f"({100*len(ewma_preds)/max(1,len(tgt_preds)):.0f}%)")
        print(f"    Rank priors: {len(rank_preds)} predictions "
              f"({100*len(rank_preds)/max(1,len(tgt_preds)):.0f}%)")
        if ewma_preds:
            em = _share_metrics(ewma_preds, "EWMA")
            rm = _share_metrics(rank_preds, "Rank") if rank_preds else None
            print(f"    EWMA: MAE={em['mae']:.4f}, bias={em['bias']:+.4f}, "
                  f"corr={em['corr']:.3f}")
            if rm:
                print(f"    Rank: MAE={rm['mae']:.4f}, bias={rm['bias']:+.4f}, "
                      f"corr={rm['corr']:.3f}")

        # ── Elite target hog bias (top-20 WR/TE by volume) ──
        # Identify players with most total predictions (proxy for high-usage)
        player_records = defaultdict(list)
        for p in tgt_preds:
            if p["position"] in ("WR", "TE"):
                player_records[p["player"]].append(p)
        # Sort by mean actual share (high-usage → high share)
        player_stats = []
        for pname, recs in player_records.items():
            if len(recs) < 8:
                continue
            mean_actual = np.mean([r["actual"] for r in recs])
            mean_pred = np.mean([r["predicted"] for r in recs])
            player_stats.append({
                "player": pname,
                "n": len(recs),
                "mean_actual": mean_actual,
                "mean_pred": mean_pred,
                "bias": mean_pred - mean_actual,
                "position": recs[0]["position"],
            })
        player_stats.sort(key=lambda x: x["mean_actual"], reverse=True)
        top20 = player_stats[:20]
        if top20:
            print(f"\n  Elite target hogs (top-20 WR/TE by actual share, ≥8 games):")
            print(f"  {'Player':<18} {'Pos':>3} {'N':>4} {'Pred':>7} "
                  f"{'Actual':>7} {'Bias':>7}")
            print("  " + "-" * 52)
            for ps in top20:
                print(f"  {ps['player']:<18} {ps['position']:>3} {ps['n']:>4} "
                      f"{ps['mean_pred']:>7.3f} {ps['mean_actual']:>7.3f} "
                      f"{ps['bias']:>+7.3f}")
            top20_bias = np.mean([ps["bias"] for ps in top20])
            print(f"  {'Top-20 mean bias:':<32} {top20_bias:>+7.3f}")

        # ── Σ predicted vs Σ actual named targets per team-week ──
        # Two comparisons:
        #   (a) Full named set: pred sum vs actual coverage (from other-pool records)
        #   (b) Matched set: pred sum vs actual sum for players in both
        print(f"\n  Named-target sum comparison (per team-week):")
        if other_pool_records:
            otr = other_pool_records
            ps_ts = np.array([r["pred_sum_ts"] for r in otr])
            na_ts_arr = np.array([r["named_actual_ts"] for r in otr])
            print(f"    (a) Full named set (incl. 0-target predicted players):")
            print(f"        Σ pred: mean={ps_ts.mean():.3f}")
            print(f"        Σ actual named: mean={na_ts_arr.mean():.3f}")
            print(f"        Bias: {(ps_ts - na_ts_arr).mean():+.3f}")

        # (b) now shows unconditional sums (includes 0-target actuals)
        tw_pred_sums = defaultdict(float)
        tw_actual_sums = defaultdict(float)
        for p in tgt_preds:
            key = (p["team"], p["season"], p["week"])
            tw_pred_sums[key] += p["predicted"]
            tw_actual_sums[key] += p["actual"]
        common_keys = set(tw_pred_sums.keys()) & set(tw_actual_sums.keys())
        if common_keys:
            pred_s = np.array([tw_pred_sums[k] for k in common_keys])
            act_s = np.array([tw_actual_sums[k] for k in common_keys])
            print(f"    (b) All predicted players (unconditional, incl 0s):")
            print(f"        Σ pred: mean={pred_s.mean():.3f}")
            print(f"        Σ actual: mean={act_s.mean():.3f}")
            print(f"        Bias: {(pred_s - act_s).mean():+.3f}")

        # ── Share-bucket calibration (unconditional) ──
        print(f"\n  Target share calibration by predicted bucket (UNCONDITIONAL):")
        print(f"  {'Bucket':<12} {'N':>6} {'Mean Pred':>10} "
              f"{'Mean Actual':>12} {'Bias':>7} {'%Ghost':>7}")
        print("  " + "-" * 60)
        buckets = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15),
                   (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
        for lo, hi in buckets:
            bucket_preds = [p for p in tgt_preds
                            if lo <= p["predicted"] < hi]
            if len(bucket_preds) < 20:
                continue
            mp = np.mean([p["predicted"] for p in bucket_preds])
            ma = np.mean([p["actual"] for p in bucket_preds])
            n_ghost = sum(1 for p in bucket_preds
                          if not p.get("had_targets", True))
            pct_ghost = 100 * n_ghost / len(bucket_preds)
            label = f"[{lo:.0%},{hi:.0%})"
            print(f"  {label:<12} {len(bucket_preds):>6} {mp:>10.4f} "
                  f"{ma:>12.4f} {mp - ma:>+7.4f} {pct_ghost:>6.1f}%")

        # ── Bucket calibration — conditional (had targets only, for reference)
        print(f"\n  Target share calibration by predicted bucket (CONDITIONAL):")
        print(f"  {'Bucket':<12} {'N':>6} {'Mean Pred':>10} "
              f"{'Mean Actual':>12} {'Bias':>7}")
        print("  " + "-" * 52)
        for lo, hi in buckets:
            bucket_cond = [p for p in tgt_cond
                           if lo <= p["predicted"] < hi]
            if len(bucket_cond) < 20:
                continue
            mp = np.mean([p["predicted"] for p in bucket_cond])
            ma = np.mean([p["actual"] for p in bucket_cond])
            label = f"[{lo:.0%},{hi:.0%})"
            print(f"  {label:<12} {len(bucket_cond):>6} {mp:>10.4f} "
                  f"{ma:>12.4f} {mp - ma:>+7.4f}")

        # ── Top-share calibration: WR1 predicted vs actual ──
        print(f"\n  WR1 share calibration (top target share per team-week):")
        wr1_by_tw = defaultdict(lambda: {"pred": 0, "actual": 0,
                                          "player": "", "team": "",
                                          "season": 0, "week": 0})
        for p in tgt_preds:
            if p["position"] != "WR":
                continue
            key = (p["team"], p["season"], p["week"])
            if p["predicted"] > wr1_by_tw[key]["pred"]:
                wr1_by_tw[key] = {"pred": p["predicted"],
                                   "actual": p["actual"],
                                   "player": p["player"],
                                   "team": p["team"],
                                   "season": p["season"],
                                   "week": p["week"]}

        wr1_list = list(wr1_by_tw.values())
        if wr1_list:
            wr1_pred = np.array([w["pred"] for w in wr1_list])
            wr1_act = np.array([w["actual"] for w in wr1_list])
            print(f"    N = {len(wr1_list)}")
            print(f"    Pred WR1 share: mean={wr1_pred.mean():.3f}, "
                  f"std={wr1_pred.std():.3f}")
            print(f"    Actual WR1 share: mean={wr1_act.mean():.3f}, "
                  f"std={wr1_act.std():.3f}")
            print(f"    Corr: {np.corrcoef(wr1_pred, wr1_act)[0,1]:.3f}")
            print(f"    MAE: {np.mean(np.abs(wr1_pred - wr1_act)):.4f}")

            # WR1 share collapse cases (actual < 0.10, predicted > 0.20)
            collapses = [w for w in wr1_list
                         if w["actual"] < 0.10 and w["pred"] > 0.20]
            if collapses:
                print(f"    WR1 share collapses (actual < 10%, pred > 20%): "
                      f"{len(collapses)}")
                for w in sorted(collapses, key=lambda x: x["actual"])[:5]:
                    print(f"      {w['player']} ({w['team']} S{w['season']} "
                          f"W{w['week']}): pred={w['pred']:.3f}, "
                          f"actual={w['actual']:.3f}")

        # ── CeeDee Lamb case study ──
        lamb_preds = [p for p in tgt_preds
                      if "Lamb" in p["player"] and p["position"] == "WR"]
        if lamb_preds:
            print(f"\n  Lamb case study (N={len(lamb_preds)}):")
            lamb_pred = np.array([p["predicted"] for p in lamb_preds])
            lamb_act = np.array([p["actual"] for p in lamb_preds])
            print(f"    Mean pred share: {lamb_pred.mean():.3f}, "
                  f"mean actual: {lamb_act.mean():.3f}")
            print(f"    Corr: {np.corrcoef(lamb_pred, lamb_act)[0,1]:.3f}"
                  if len(lamb_preds) > 2 else "")
            # Show worst misses
            lamb_err = np.abs(lamb_pred - lamb_act)
            worst = np.argsort(lamb_err)[-3:]
            print(f"    Worst misses:")
            for i in worst:
                p = lamb_preds[i]
                print(f"      S{p['season']} W{p['week']}: "
                      f"pred={p['predicted']:.3f}, actual={p['actual']:.3f}")

    # ── Game-clustered bootstrap CI on target share MAE ──
    # Cluster by (team, season, week) — within-game predictions are correlated
    from collections import defaultdict as _dd
    by_tw = _dd(list)
    for p in tgt_preds:
        by_tw[(p["team"], p["season"], p["week"])].append(p)

    rng = np.random.default_rng(42)
    tw_keys = list(by_tw.keys())
    boot_mae = []
    for _ in range(2000):
        sampled = rng.choice(len(tw_keys), size=len(tw_keys), replace=True)
        errs = []
        for idx in sampled:
            for p in by_tw[tw_keys[idx]]:
                errs.append(abs(p["predicted"] - p["actual"]))
        boot_mae.append(np.mean(errs))

    results["target_share_mae_ci"] = (
        round(np.percentile(boot_mae, 2.5), 4),
        round(np.percentile(boot_mae, 97.5), 4),
    )

    if verbose:
        print(f"\n  Game-clustered bootstrap 95% CI:")
        print(f"    Target share MAE: [{results['target_share_mae_ci'][0]:.4f}, "
              f"{results['target_share_mae_ci'][1]:.4f}]")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Market-anchored team-level scaling
# ══════════════════════════════════════════════════════════════════════════════

# Prop-settlement stats that can be scaled.
_SCALABLE_STATS = [
    "pass_yds", "completions", "pass_att",
    "rush_att", "rush_yds",
    "pass_tds", "rush_tds",
    "sacks", "interceptions",
]

# Stats that should be rounded to integers after scaling.
_INT_STATS = {"pass_att", "completions", "rush_att",
              "pass_tds", "rush_tds", "sacks", "interceptions"}


def scale_sim_draws(raw_draws: list[dict],
                    targets: dict,
                    ) -> list[dict]:
    """Scale sim draws so prop-settlement means match market targets.

    The sim produces team stats with good shape (variance, correlation,
    game-script conditioning) but biased level (+5-13% across stats).
    This function replaces the level with market/consensus values via
    multiplicative scaling, preserving within-draw correlations.

    Args:
        raw_draws: list of N dicts from _compile_play_stats (sim-internal
            format with keys: pass_att, completions, pass_yards, sacks,
            scrambles, rush_att, rush_yards, scramble_yards, kneels,
            kneel_yards, pass_tds, rush_tds, interceptions, total_plays).
        targets: dict of prop-settlement stat targets, e.g.
            ``{"pass_yds": 240, "completions": 22, "pass_att": 32}``.
            Only stats with explicit targets are scaled; others pass
            through unchanged.

    Returns:
        list of N scaled dicts in sim-internal format, suitable for
        ``nfl_player_allocation.allocate_game_stats()``.
    """
    from nfl_stat_definitions import sim_stats_to_prop

    if not raw_draws or not targets:
        return list(raw_draws)

    N = len(raw_draws)

    # ── 1. Convert each draw to prop format to compute means ──
    prop_draws = [sim_stats_to_prop(d) for d in raw_draws]

    # ── 2. Compute prop-stat means across draws ──
    prop_means = {}
    for stat in _SCALABLE_STATS:
        vals = [d.get(stat, 0) for d in prop_draws]
        prop_means[stat] = float(np.mean(vals))

    # ── 3. Compute scale factors for stats with targets ──
    factors = {}
    for stat in _SCALABLE_STATS:
        if stat in targets and prop_means.get(stat, 0) > 0:
            factors[stat] = targets[stat] / prop_means[stat]

    if not factors:
        return list(raw_draws)

    # ── 4. Apply factors back to sim-internal stats ──
    scaled = []
    for raw in raw_draws:
        d = dict(raw)  # shallow copy

        # --- pass_yards (1:1 with prop pass_yds) ---
        if "pass_yds" in factors:
            d["pass_yards"] = raw["pass_yards"] * factors["pass_yds"]

        # --- completions (1:1) ---
        if "completions" in factors:
            d["completions"] = max(0, round(
                raw["completions"] * factors["completions"]))

        # --- pass_att (prop = throws = pass_att - sacks - scrambles) ---
        if "pass_att" in factors:
            throws = raw["pass_att"] - raw["sacks"] - raw["scrambles"]
            scaled_throws = max(0, round(throws * factors["pass_att"]))
            # Reconstruct dropbacks: scaled throws + original sacks/scrambles
            d["pass_att"] = scaled_throws + raw["sacks"] + raw["scrambles"]

        # --- rush_att (prop = designed + scrambles + kneels) ---
        if "rush_att" in factors:
            prop_total = (raw["rush_att"] + raw["scrambles"]
                          + raw.get("kneels", 0))
            scaled_total = max(0, round(prop_total * factors["rush_att"]))
            # Designed rushes = scaled total - scrambles - kneels
            d["rush_att"] = max(0, scaled_total - raw["scrambles"]
                                - raw.get("kneels", 0))

        # --- rush_yds (prop = designed + scramble + kneel yards) ---
        if "rush_yds" in factors:
            prop_total = (raw["rush_yards"] + raw["scramble_yards"]
                          + raw.get("kneel_yards", 0))
            scaled_total = prop_total * factors["rush_yds"]
            # Designed yards = scaled total - scramble - kneel yards
            d["rush_yards"] = (scaled_total - raw["scramble_yards"]
                               - raw.get("kneel_yards", 0))

        # --- pass_tds (1:1) ---
        if "pass_tds" in factors:
            d["pass_tds"] = max(0, round(
                raw["pass_tds"] * factors["pass_tds"]))

        # --- rush_tds (1:1) ---
        if "rush_tds" in factors:
            d["rush_tds"] = max(0, round(
                raw["rush_tds"] * factors["rush_tds"]))

        # --- sacks (1:1) ---
        if "sacks" in factors:
            d["sacks"] = max(0, round(raw["sacks"] * factors["sacks"]))

        # --- interceptions (1:1) ---
        if "interceptions" in factors:
            d["interceptions"] = max(0, round(
                raw["interceptions"] * factors["interceptions"]))

        # --- update total_plays for consistency ---
        d["total_plays"] = (d["pass_att"] + d["rush_att"]
                            + d.get("kneels", 0))

        scaled.append(d)

    return scaled


def build_team_targets(
    consensus_qb: dict | None = None,
    consensus_rb_sum: dict | None = None,
    volume_proj: dict | None = None,
    spread: float = 0.0,
    total: float = 45.0,
) -> dict:
    """Construct prop-settlement scaling targets from available sources.

    Priority for each stat:
      1. QB/RB consensus projection (most specific)
      2. Volume model prediction
      3. Not set (stat left unscaled)

    Args:
        consensus_qb: QB's consensus projections in prop-settlement
            terms, e.g. ``{"pass_yds": 240, "completions": 22,
            "pass_att": 32, "pass_tds": 1.5, "interceptions": 0.7}``.
        consensus_rb_sum: summed RB projections for the team, e.g.
            ``{"rush_yds": 100, "rush_att": 22}``.
        volume_proj: output of ``project_team_volume()``, e.g.
            ``{"pass_attempts": 34, "rush_attempts": 26}``.
        spread: team spread (negative = favored).
        total: game total.

    Returns:
        dict of prop-settlement targets for ``scale_sim_draws()``.
    """
    targets = {}

    # --- QB consensus: pass stats ---
    if consensus_qb:
        for stat in ("pass_yds", "completions", "pass_att",
                     "pass_tds", "interceptions"):
            if stat in consensus_qb and consensus_qb[stat] is not None:
                targets[stat] = consensus_qb[stat]

    # --- Volume model: attempt counts (fills gaps) ---
    if volume_proj:
        if "pass_att" not in targets and volume_proj.get("pass_attempts"):
            targets["pass_att"] = volume_proj["pass_attempts"]
        if "rush_att" not in targets and volume_proj.get("rush_attempts"):
            targets["rush_att"] = volume_proj["rush_attempts"]

    # --- RB consensus: rush stats ---
    if consensus_rb_sum:
        for stat in ("rush_yds", "rush_att"):
            if stat in consensus_rb_sum and consensus_rb_sum[stat] is not None:
                if stat not in targets:  # don't override volume model
                    targets[stat] = consensus_rb_sum[stat]

    # --- Derived targets from consensus ---
    # If we have pass_yds and completions but not pass_att, derive:
    #   pass_att ≈ completions / league_comp_rate
    if "pass_att" not in targets and "completions" in targets:
        targets["pass_att"] = round(targets["completions"] / 0.648)

    return targets


def generate_game_draws(
    n_sims: int,
    home_profile: dict,
    away_profile: dict,
    home_qb: dict | None = None,
    away_qb: dict | None = None,
    spread: float = 0.0,
    total: float = 45.0,
    home_targets: dict | None = None,
    away_targets: dict | None = None,
    seed: int = 42,
) -> dict:
    """Run game sim and scale team draws to market targets.

    This is the entry point for the prop model: produces N market-anchored
    team stat draws ready for player allocation via
    ``nfl_player_allocation.allocate_game_n()``.

    Args:
        n_sims: number of simulation draws (recommend ≥5000 for props).
        home/away_profile: team profiles from build_team_profiles().
        home/away_qb: optional QB overrides.
        spread: home spread (negative = favored).
        total: game total.
        home/away_targets: prop-settlement stat targets from
            build_team_targets(). Pass None to skip scaling for that side.
        seed: random seed.

    Returns: {
        "home_draws": list of N scaled team stat dicts,
        "away_draws": list of N scaled team stat dicts,
        "home_scores": np.array of N scores,
        "away_scores": np.array of N scores,
        "home_scale_factors": {stat: factor} or None,
        "away_scale_factors": {stat: factor} or None,
        "n_sims": int,
    }
    """
    from nfl_game_sim import simulate_game
    from nfl_stat_definitions import sim_stats_to_prop

    rng = np.random.default_rng(seed)

    home_draws_raw = []
    away_draws_raw = []
    home_scores = []
    away_scores = []

    for _ in range(n_sims):
        result = simulate_game(rng, home_profile, away_profile,
                               home_qb, away_qb, spread, total)
        home_draws_raw.append(result["home_stats"])
        away_draws_raw.append(result["away_stats"])
        home_scores.append(result["home_score"])
        away_scores.append(result["away_score"])

    # Compute pre-scaling prop means for diagnostics
    def _prop_means(draws):
        props = [sim_stats_to_prop(d) for d in draws]
        return {stat: float(np.mean([p[stat] for p in props]))
                for stat in props[0]}

    home_pre = _prop_means(home_draws_raw)
    away_pre = _prop_means(away_draws_raw)

    # Scale
    home_draws = home_draws_raw
    away_draws = away_draws_raw
    home_factors = None
    away_factors = None

    if home_targets:
        home_draws = scale_sim_draws(home_draws_raw, home_targets)
        home_factors = {stat: home_targets[stat] / home_pre[stat]
                        for stat in home_targets
                        if stat in home_pre and home_pre[stat] > 0}

    if away_targets:
        away_draws = scale_sim_draws(away_draws_raw, away_targets)
        away_factors = {stat: away_targets[stat] / away_pre[stat]
                        for stat in away_targets
                        if stat in away_pre and away_pre[stat] > 0}

    return {
        "home_draws": home_draws,
        "away_draws": away_draws,
        "home_scores": np.array(home_scores),
        "away_scores": np.array(away_scores),
        "home_scale_factors": home_factors,
        "away_scale_factors": away_factors,
        "n_sims": n_sims,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate-volume"
    if cmd == "validate-volume":
        validate_team_volume()
    elif cmd == "validate-shares":
        validate_shares()
    elif cmd == "test-scaling":
        # Smoke test: generate draws, scale, verify means match targets
        from nfl_game_sim import build_team_profiles
        profiles = build_team_profiles()
        lg = list(profiles.values())[0]  # arbitrary team

        targets = {
            "pass_yds": 233.0,
            "completions": 21.0,
            "pass_att": 33.0,
            "rush_att": 27.0,
            "rush_yds": 117.0,
            "pass_tds": 1.5,
            "rush_tds": 0.9,
        }

        print(f"Running 2000 sims with targets: {targets}")
        result = generate_game_draws(
            2000, lg, lg, spread=0, total=45,
            home_targets=targets, seed=42)

        from nfl_stat_definitions import sim_stats_to_prop
        home_props = [sim_stats_to_prop(d) for d in result["home_draws"]]

        print(f"\n  {'Stat':<16} {'Target':>8} {'Scaled':>8} {'Error':>8}")
        print("  " + "-" * 44)
        for stat in sorted(targets):
            scaled_mean = np.mean([p[stat] for p in home_props])
            err = scaled_mean - targets[stat]
            print(f"  {stat:<16} {targets[stat]:>8.1f} "
                  f"{scaled_mean:>8.1f} {err:>+8.2f}")

        print(f"\n  Scale factors: {result['home_scale_factors']}")
        print(f"  N sims: {result['n_sims']}")
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: validate-volume, validate-shares, test-scaling")
