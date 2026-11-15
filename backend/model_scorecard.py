"""
Weekly Model Scorecard — computes and stores per-week + season-to-date
accuracy, direction, calibration, and biggest-miss metrics from
fantasy_forward_preds in historical.db.

NFL now; NBA stub (add stat types when grading accumulates).
"""
import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")

# ── Regime boundaries ────────────────────────────────────────────────────────
_REGIME_BOUNDS = [
    ("R1", "2026-11-05T22:03:48Z"),
    ("R2", "2026-11-07T10:47:26Z"),
]

_ALL = "_ALL_"

# ── DDL ──────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS weekly_model_scorecard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    stat_type TEXT NOT NULL,
    position TEXT NOT NULL,
    model_version TEXT NOT NULL,
    line_depth TEXT NOT NULL,
    n_preds INTEGER,
    mae REAL,
    median_ae REAL,
    wape REAL,
    market_mae REAL,
    direction_correct INTEGER,
    direction_n INTEGER,
    direction_pct REAL,
    direction_expected_pct REAL,
    direction_edge_pp REAL,
    brier REAL,
    market_brier REAL,
    calibration_json TEXT,
    biggest_misses_json TEXT,
    direction_lean_json TEXT,
    stale_n INTEGER,
    stale_correct INTEGER,
    stale_mae REAL,
    regime TEXT,
    computed_at TEXT,
    UNIQUE(sport, season, week, stat_type, position, model_version, line_depth)
)
"""


def _ensure_table():
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.execute(_CREATE_TABLE)
    # Migrate: add columns if they don't exist yet
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(weekly_model_scorecard)").fetchall()}
    for col, typ in [("direction_expected_pct", "REAL"),
                     ("direction_edge_pp", "REAL"),
                     ("direction_lean_json", "TEXT"),
                     ("stale_n", "INTEGER"),
                     ("stale_correct", "INTEGER"),
                     ("stale_mae", "REAL"),
                     ("cushion_json", "TEXT"),
                     ("asymmetric_mae", "REAL"),
                     ("mean_signed_error", "REAL"),
                     ("ladder_brier_json", "TEXT"),
                     ("mae_median", "REAL")]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE weekly_model_scorecard ADD COLUMN {col} {typ}")
    conn.commit()
    conn.close()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _assign_regime(created_at_str: str) -> str:
    if not created_at_str:
        return "R1"
    for label, boundary in _REGIME_BOUNDS:
        if created_at_str < boundary:
            return label
    return "R3"


def _majority_regime(regimes: pd.Series) -> str:
    if regimes.empty:
        return "R1"
    counts = regimes.value_counts()
    top = counts.index[0]
    if counts.iloc[0] / len(regimes) < 0.8 and len(counts) > 1:
        return "mixed"
    return top


def _compute_accuracy(df: pd.DataFrame) -> dict:
    """MAE, median AE, WAPE (projection vs actual), and market MAE."""
    errors = (df["consensus_mean"] - df["actual_value"]).abs()
    market_errors = (df["prop_line"] - df["actual_value"]).abs()
    actual_sum = df["actual_value"].abs().sum()
    return {
        "mae": round(errors.mean(), 3),
        "median_ae": round(errors.median(), 3),
        "wape": round(errors.sum() / actual_sum, 4) if actual_sum > 0 else None,
        "market_mae": round(market_errors.mean(), 3),
    }


def _compute_direction(df: pd.DataFrame) -> dict:
    """Direction vs market-implied baseline.

    For each prediction, the model picks a side (over if consensus > line).
    The market-implied probability of that side = devigged_implied for over,
    1-devigged_implied for under. We report actual hit rate minus expected
    (market-implied), with count and CI.
    """
    sub = df[(df["actual_value"] != df["prop_line"])
             & (df["consensus_mean"] != df["prop_line"])
             & df["devigged_implied"].notna()].copy()
    if sub.empty:
        return {"correct": 0, "n": 0, "pct": None,
                "expected_pct": None, "edge_pp": None}
    proj_over = sub["consensus_mean"] > sub["prop_line"]
    actual_over = sub["actual_value"] > sub["prop_line"]
    correct = (proj_over == actual_over).sum()
    n = len(sub)
    # Market-implied P of the side the model picked
    sub["mkt_p_picked"] = np.where(proj_over,
                                    sub["devigged_implied"],
                                    1.0 - sub["devigged_implied"])
    expected_pct = sub["mkt_p_picked"].mean()
    actual_pct = correct / n
    edge_pp = actual_pct - expected_pct

    # Per-lean (over/under) breakdown
    lean_split = {}
    for lean_label, lean_mask in [("over", proj_over), ("under", ~proj_over)]:
        ls = sub[lean_mask]
        if ls.empty:
            lean_split[lean_label] = {"n": 0, "correct": 0, "pct": None, "edge_pp": None}
            continue
        ls_correct = int((ls["consensus_mean"] > ls["prop_line"]).eq(ls["actual_value"] > ls["prop_line"]).sum())
        ls_n = len(ls)
        ls_exp = ls["mkt_p_picked"].mean()
        ls_pct = ls_correct / ls_n
        lean_split[lean_label] = {
            "n": ls_n, "correct": ls_correct,
            "pct": round(ls_pct, 4),
            "edge_pp": round(ls_pct - ls_exp, 4),
        }

    return {
        "correct": int(correct),
        "n": int(n),
        "pct": round(actual_pct, 4),
        "expected_pct": round(expected_pct, 4),
        "edge_pp": round(edge_pp, 4),
        "lean_split": lean_split,
    }


def _compute_calibration(df: pd.DataFrame, n_buckets: int = 10) -> list:
    """Bucket p_over into deciles, return predicted vs actual hit rate."""
    sub = df[df["p_over"].notna() & (df["actual_value"] != df["prop_line"])].copy()
    if sub.empty:
        return []
    sub["outcome"] = (sub["actual_value"] > sub["prop_line"]).astype(int)
    edges = np.linspace(0, 1, n_buckets + 1)
    buckets = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        if i < n_buckets - 1:
            mask = (sub["p_over"] >= lo) & (sub["p_over"] < hi)
        else:
            mask = (sub["p_over"] >= lo) & (sub["p_over"] <= hi)
        bk = sub[mask]
        if len(bk) == 0:
            continue
        buckets.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "lo": round(lo, 2),
            "hi": round(hi, 2),
            "n": int(len(bk)),
            "predicted_pct": round(bk["p_over"].mean(), 4),
            "actual_pct": round(bk["outcome"].mean(), 4),
        })
    return buckets


def _compute_brier(df: pd.DataFrame) -> dict:
    """Brier score: model p_over vs outcome, and market devigged_implied."""
    sub = df[
        df["p_over"].notna()
        & df["devigged_implied"].notna()
        & (df["actual_value"] != df["prop_line"])
    ].copy()
    if sub.empty:
        return {"brier": None, "market_brier": None}
    outcome = (sub["actual_value"] > sub["prop_line"]).astype(float)
    brier = ((sub["p_over"] - outcome) ** 2).mean()
    # devigged_implied is P(over) from the market
    market_brier = ((sub["devigged_implied"] - outcome) ** 2).mean()
    return {"brier": round(brier, 6), "market_brier": round(market_brier, 6)}


def _compute_biggest_misses(df: pd.DataFrame, top_n: int = 10) -> list:
    """Top N projection errors by |consensus_mean - actual_value|, deduplicated by player+stat."""
    sub = df.copy()
    sub["abs_error"] = (sub["consensus_mean"] - sub["actual_value"]).abs()
    # Dedup: same player+stat has identical consensus_mean and actual across line depths
    deduped = sub.sort_values("abs_error", ascending=False).drop_duplicates(
        subset=["player_name", "stat_type"], keep="first"
    )
    top = deduped.nlargest(top_n, "abs_error")
    misses = []
    for _, r in top.iterrows():
        misses.append({
            "player": r["player_name"],
            "stat": r["stat_type"],
            "actual": round(r["actual_value"], 1),
            "projection": round(r["consensus_mean"], 1),
            "line": round(r["prop_line"], 1),
            "error": round(r["abs_error"], 1),
        })
    return misses


_STALE_PCT_THRESHOLD = 0.40   # |projection - line| > 40% of line
_STALE_ABS_THRESHOLD = 10.0   # AND |projection - line| > 10 units (filters tiny-line props)
_STALE_MIN_LINE = 5.0         # Only evaluate props with line >= 5


def _compute_stale_input(df: pd.DataFrame, segment_line_depth: str = "main") -> dict:
    """Flag and report accuracy of predictions where projection diverges significantly from line.

    Only meaningful for main lines — alt lines are structurally far from projection.
    Requires BOTH: |proj - line| > 40% of line AND |proj - line| > 10 units,
    with a minimum line of 5 to exclude tiny-line props (e.g. 0.5 QB rush yds).
    """
    # Only compute for main-line segments or _ALL_ (using main subset)
    if segment_line_depth not in ("main", _ALL):
        return {"stale_n": 0, "stale_correct": None, "stale_mae": None}
    if segment_line_depth == _ALL:
        sub = df[(df["prop_line"] >= _STALE_MIN_LINE) & (df["line_depth"] == "main")].copy()
    else:
        sub = df[df["prop_line"] >= _STALE_MIN_LINE].copy()
    if sub.empty:
        return {"stale_n": 0, "stale_correct": None, "stale_mae": None}
    sub["abs_gap"] = (sub["consensus_mean"] - sub["prop_line"]).abs()
    sub["pct_gap"] = sub["abs_gap"] / sub["prop_line"]
    stale = sub[(sub["pct_gap"] > _STALE_PCT_THRESHOLD) & (sub["abs_gap"] > _STALE_ABS_THRESHOLD)]
    if stale.empty:
        return {"stale_n": 0, "stale_correct": None, "stale_mae": None}
    # Direction accuracy of stale predictions
    stale_dir = stale[(stale["actual_value"] != stale["prop_line"])
                       & (stale["consensus_mean"] != stale["prop_line"])]
    if not stale_dir.empty:
        proj_over = stale_dir["consensus_mean"] > stale_dir["prop_line"]
        actual_over = stale_dir["actual_value"] > stale_dir["prop_line"]
        stale_correct = int((proj_over == actual_over).sum())
    else:
        stale_correct = None
    stale_mae = round((stale["consensus_mean"] - stale["actual_value"]).abs().mean(), 3)
    return {
        "stale_n": int(len(stale)),
        "stale_correct": stale_correct,
        "stale_mae": stale_mae,
    }


def _compute_cushion(df: pd.DataFrame, segment_line_depth: str = "main") -> dict:
    """Side-aware accuracy: cushion = lean × (actual − line).

    Only meaningful for main lines — alt lines are structurally far from projection.
    Positive cushion = model picked the right side of the line.
    """
    if segment_line_depth not in ("main", _ALL):
        return {}
    if segment_line_depth == _ALL:
        sub = df[(df["actual_value"] != df["prop_line"])
                 & (df["consensus_mean"] != df["prop_line"])
                 & (df["line_depth"] == "main")].copy()
    else:
        sub = df[(df["actual_value"] != df["prop_line"])
                 & (df["consensus_mean"] != df["prop_line"])].copy()
    if sub.empty:
        return {}
    sub["lean"] = np.sign(sub["consensus_mean"] - sub["prop_line"])
    sub["cushion"] = sub["lean"] * (sub["actual_value"] - sub["prop_line"])
    sub["abs_error"] = (sub["consensus_mean"] - sub["actual_value"]).abs()

    right = sub[sub["cushion"] > 0]
    wrong = sub[sub["cushion"] <= 0]
    n_total = len(sub)
    n_right = len(right)
    n_wrong = len(wrong)

    result = {
        "n": n_total,
        "n_right": n_right,
        "n_wrong": n_wrong,
        "right_side_rate": round(n_right / n_total, 4) if n_total else None,
        "mean_cushion_right": round(float(right["cushion"].mean()), 2) if not right.empty else None,
        "mean_deficit_wrong": round(float(wrong["cushion"].mean()), 2) if not wrong.empty else None,
        "mae_right": round(float(right["abs_error"].mean()), 3) if not right.empty else None,
        "mae_wrong": round(float(wrong["abs_error"].mean()), 3) if not wrong.empty else None,
    }

    # Histogram: 10 bins of cushion distribution
    vals = sub["cushion"].values
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    edges = np.linspace(lo, hi, 11)
    hist_bins = []
    for i in range(10):
        if i < 9:
            mask = (vals >= edges[i]) & (vals < edges[i + 1])
        else:
            mask = (vals >= edges[i]) & (vals <= edges[i + 1])
        hist_bins.append({
            "lo": round(float(edges[i]), 1),
            "hi": round(float(edges[i + 1]), 1),
            "n": int(mask.sum()),
        })
    result["histogram"] = hist_bins
    return result


def _compute_asymmetric_mae(df: pd.DataFrame, segment_line_depth: str = "main") -> float | None:
    """Asymmetric MAE: right-side misses weighted 0.5, wrong-side weighted 1.0.

    'Right side' = model leaned over and actual went over (or under/under).
    Display only — not for model tuning.
    """
    if segment_line_depth not in ("main", _ALL):
        return None
    if segment_line_depth == _ALL:
        sub = df[(df["actual_value"] != df["prop_line"])
                 & (df["consensus_mean"] != df["prop_line"])
                 & (df["line_depth"] == "main")].copy()
    else:
        sub = df[(df["actual_value"] != df["prop_line"])
                 & (df["consensus_mean"] != df["prop_line"])].copy()
    if sub.empty:
        return None
    lean = np.sign(sub["consensus_mean"] - sub["prop_line"])
    cushion = lean * (sub["actual_value"] - sub["prop_line"])
    abs_error = (sub["consensus_mean"] - sub["actual_value"]).abs()
    weight = np.where(cushion > 0, 0.5, 1.0)
    return round(float((abs_error * weight).sum() / weight.sum()), 3)


def _compute_mean_signed_error(df: pd.DataFrame) -> float | None:
    """Bias = mean(consensus_mean − actual_value). Positive = model overshoots."""
    if df.empty:
        return None
    return round(float((df["consensus_mean"] - df["actual_value"]).mean()), 3)


def _compute_mae_median(df: pd.DataFrame) -> float | None:
    """MAE using the distributional median (ppf 0.5), not source-consensus median.

    distributional_median = shifted gamma ppf(0.5) for yardage, NegBin ppf(0.5)
    for counts. Falls back to consensus_median (source median) if distributional
    column is not populated.
    Returns None if no median values are available.
    """
    # Prefer distributional median (model's predictive distribution ppf 0.5)
    col = None
    if "distributional_median" in df.columns and df["distributional_median"].notna().any():
        col = "distributional_median"
    elif "consensus_median" in df.columns and df["consensus_median"].notna().any():
        col = "consensus_median"
    if col is None:
        return None
    sub = df[df[col].notna()]
    if sub.empty:
        return None
    return round(float((sub[col] - sub["actual_value"]).abs().mean()), 3)


def _compute_ladder_score(df: pd.DataFrame, segment_line_depth: str = "main") -> list:
    """Ladder score: model vs market Brier by rung distance from main line.

    Only computed for _ALL_ line_depth segments (needs main + alts in same df).
    Self-joins on (player_name, stat_type, method) to find each row's main-line
    sibling and compute rung_distance = prop_line - main_prop_line.

    ROI: for v1.2 rows where |edge_vs_market| >= 5pp, compute mean ROI at
    actual prices. Game-clustered bootstrap CI using player_name as cluster
    proxy. Gate on N >= 10.
    """
    import numpy as np

    if segment_line_depth != _ALL:
        return []
    required = {"player_name", "stat_type", "prop_line", "p_over",
                "devigged_implied", "actual_value", "line_depth"}
    if not required.issubset(df.columns):
        return []

    sub = df[df["p_over"].notna()
             & df["devigged_implied"].notna()
             & (df["actual_value"] != df["prop_line"])].copy()
    if sub.empty:
        return []

    # Get main-line prop_line for each (player_name, stat_type)
    join_cols = ["player_name", "stat_type"]
    if "method" in sub.columns:
        join_cols.append("method")
    main_lines = sub[sub["line_depth"] == "main"][join_cols + ["prop_line"]].copy()
    main_lines = main_lines.rename(columns={"prop_line": "main_prop_line"})
    main_lines = main_lines.drop_duplicates(subset=join_cols, keep="first")

    merged = sub.merge(main_lines, on=join_cols, how="inner")
    if merged.empty:
        return []

    merged["rung_distance"] = merged["prop_line"] - merged["main_prop_line"]
    merged["outcome"] = (merged["actual_value"] > merged["prop_line"]).astype(float)
    merged["model_sq"] = (merged["p_over"] - merged["outcome"]) ** 2
    merged["market_sq"] = (merged["devigged_implied"] - merged["outcome"]) ** 2

    # Compute per-row ROI at actual prices for edge-qualified bets
    # Direction: if p_over > devigged_implied → over, else under
    _has_prices = (merged.get("prop_over_price") is not None
                   and "prop_over_price" in merged.columns)
    if _has_prices:
        merged["bet_direction"] = np.where(
            merged["p_over"] > merged["devigged_implied"], "over", "under")
        merged["bet_price"] = np.where(
            merged["bet_direction"] == "over",
            merged["prop_over_price"],
            merged["prop_under_price"])
        merged["bet_won"] = np.where(
            merged["bet_direction"] == "over",
            merged["actual_value"] > merged["prop_line"],
            merged["actual_value"] < merged["prop_line"])
        merged["roi"] = np.where(
            merged["bet_won"],
            merged["bet_price"] - 1.0,
            -1.0)
        # Null out where prices are missing
        merged.loc[merged["bet_price"].isna() | (merged["bet_price"] <= 1), "roi"] = np.nan

    # Group by rung_distance
    grouped = merged.groupby("rung_distance").agg(
        n=("outcome", "size"),
        model_brier=("model_sq", "mean"),
        market_brier=("market_sq", "mean"),
    ).reset_index()

    grouped = grouped[grouped["n"] >= 20]
    if grouped.empty:
        return []

    # Pre-compute ROI stats per rung (edge >= 5pp, game-clustered CI)
    _EDGE_THRESHOLDS = [0.05]  # 5pp

    def _cluster_bootstrap_ci(rung_df, n_boot=1000, alpha=0.05):
        """Game-clustered bootstrap CI for mean ROI.
        Clusters by game_id (same game = correlated outcomes).
        """
        edge_col = rung_df.get("edge_vs_market")
        if edge_col is None or "roi" not in rung_df.columns:
            return None
        qual = rung_df[
            rung_df["edge_vs_market"].notna()
            & (rung_df["edge_vs_market"].abs() >= 0.05)
            & rung_df["roi"].notna()
        ]
        if len(qual) < 10:
            return None

        # Cluster by game_id; fall back to player_name if game_id missing
        cluster_col = "game_id" if "game_id" in qual.columns and qual["game_id"].notna().any() else "player_name"
        clusters = qual.groupby(cluster_col)["roi"].apply(list).values
        n_clusters = len(clusters)
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_clusters, size=n_clusters)
            sampled_rois = []
            for i in idx:
                sampled_rois.extend(clusters[i])
            if sampled_rois:
                boot_means.append(np.mean(sampled_rois))
        if not boot_means:
            return None
        lo = float(np.percentile(boot_means, 100 * alpha / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return {
            "mean": round(float(qual["roi"].mean()), 4),
            "n": int(len(qual)),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }

    result = []
    for _, row in grouped.sort_values("rung_distance").iterrows():
        rd = float(row["rung_distance"])
        entry = {
            "rung_distance": round(rd, 1),
            "n": int(row["n"]),
            "model_brier": round(float(row["model_brier"]), 6),
            "market_brier": round(float(row["market_brier"]), 6),
            "model_better": bool(row["model_brier"] < row["market_brier"]),
            "roi": None,
        }
        if _has_prices:
            rung_df = merged[merged["rung_distance"] == rd]
            roi_stats = _cluster_bootstrap_ci(rung_df)
            if roi_stats:
                entry["roi"] = roi_stats
        result.append(entry)
    return result


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (round(max(0, centre - margin), 4), round(min(1, centre + margin), 4))


# ── Core compute ─────────────────────────────────────────────────────────────

def _compute_segment(df: pd.DataFrame, line_depth: str = "main") -> dict:
    """Compute all metrics for one segment slice."""
    acc = _compute_accuracy(df)
    dirn = _compute_direction(df)
    cal = _compute_calibration(df)
    brier = _compute_brier(df)
    misses = _compute_biggest_misses(df)
    stale = _compute_stale_input(df, segment_line_depth=line_depth)
    cushion = _compute_cushion(df, segment_line_depth=line_depth)
    asym_mae = _compute_asymmetric_mae(df, segment_line_depth=line_depth)
    mse = _compute_mean_signed_error(df)
    mae_med = _compute_mae_median(df)
    ladder = _compute_ladder_score(df, segment_line_depth=line_depth)
    regime = _majority_regime(df["_regime"])
    return {
        "n_preds": len(df),
        "mae": acc["mae"],
        "median_ae": acc["median_ae"],
        "wape": acc["wape"],
        "market_mae": acc["market_mae"],
        "direction_correct": dirn["correct"],
        "direction_n": dirn["n"],
        "direction_pct": dirn["pct"],
        "direction_expected_pct": dirn["expected_pct"],
        "direction_edge_pp": dirn["edge_pp"],
        "direction_lean_json": json.dumps(dirn.get("lean_split", {})),
        "brier": brier["brier"],
        "market_brier": brier["market_brier"],
        "calibration_json": json.dumps(cal),
        "biggest_misses_json": json.dumps(misses),
        "stale_n": stale["stale_n"],
        "stale_correct": stale["stale_correct"],
        "stale_mae": stale["stale_mae"],
        "cushion_json": json.dumps(cushion) if cushion else None,
        "asymmetric_mae": asym_mae,
        "mean_signed_error": mse,
        "mae_median": mae_med,
        "ladder_brier_json": json.dumps(ladder) if ladder else None,
        "regime": regime,
    }


def _iter_segments(df: pd.DataFrame):
    """Yield (stat_type, position, model_version, line_depth, sub_df) for all combos + _ALL_ rollups."""
    stat_types = list(df["stat_type"].unique()) + [_ALL]
    positions = list(df["position"].unique()) + [_ALL]
    model_versions = list(df["model_version"].unique()) + [_ALL]
    line_depths = list(df["line_depth"].unique()) + [_ALL]

    for st in stat_types:
        for pos in positions:
            for mv in model_versions:
                for ld in line_depths:
                    sub = df
                    if st != _ALL:
                        sub = sub[sub["stat_type"] == st]
                    if pos != _ALL:
                        sub = sub[sub["position"] == pos]
                    if mv != _ALL:
                        sub = sub[sub["model_version"] == mv]
                    if ld != _ALL:
                        sub = sub[sub["line_depth"] == ld]
                    if len(sub) == 0:
                        continue
                    yield st, pos, mv, ld, sub


def compute_weekly_scorecard(sport: str, season: int, week: int, force: bool = False) -> int:
    """
    Compute and store scorecard rows for one sport/season/week.
    Returns number of segment rows written.
    """
    _ensure_table()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))

    if not force:
        existing = conn.execute(
            "SELECT COUNT(*) FROM weekly_model_scorecard WHERE sport=? AND season=? AND week=?",
            (sport, season, week)
        ).fetchone()[0]
        if existing > 0:
            conn.close()
            logger.info("[Scorecard] %s S%d W%d already computed (%d rows), skipping", sport, season, week, existing)
            return 0

    df = pd.read_sql_query(
        """SELECT player_name, stat_type, position, model_version, line_depth,
                  consensus_mean, consensus_median, distributional_median,
                  prop_line, prop_over_price, prop_under_price,
                  p_over, devigged_implied, edge_vs_market,
                  actual_value, created_at, game_id
           FROM fantasy_forward_preds
           WHERE season = ? AND week = ? AND actual_value IS NOT NULL AND data_error IS NULL""",
        conn, params=(season, week)
    )
    if df.empty:
        conn.close()
        logger.warning("[Scorecard] No graded data for %s S%d W%d", sport, season, week)
        return 0

    df["_regime"] = df["created_at"].apply(_assign_regime)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = 0

    for st, pos, mv, ld, sub in _iter_segments(df):
        metrics = _compute_segment(sub, line_depth=ld)
        conn.execute("""
            INSERT OR REPLACE INTO weekly_model_scorecard
                (sport, season, week, stat_type, position, model_version, line_depth,
                 n_preds, mae, median_ae, wape, market_mae,
                 direction_correct, direction_n, direction_pct,
                 direction_expected_pct, direction_edge_pp, direction_lean_json,
                 brier, market_brier, calibration_json, biggest_misses_json,
                 stale_n, stale_correct, stale_mae,
                 cushion_json, asymmetric_mae, mean_signed_error, mae_median,
                 ladder_brier_json,
                 regime, computed_at)
            VALUES (?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?, ?,?,?, ?,?,?,?, ?, ?,?)
        """, (
            sport, season, week, st, pos, mv, ld,
            metrics["n_preds"], metrics["mae"], metrics["median_ae"],
            metrics["wape"], metrics["market_mae"],
            metrics["direction_correct"], metrics["direction_n"], metrics["direction_pct"],
            metrics["direction_expected_pct"], metrics["direction_edge_pp"],
            metrics["direction_lean_json"],
            metrics["brier"], metrics["market_brier"],
            metrics["calibration_json"], metrics["biggest_misses_json"],
            metrics["stale_n"], metrics["stale_correct"], metrics["stale_mae"],
            metrics["cushion_json"], metrics["asymmetric_mae"],
            metrics["mean_signed_error"], metrics["mae_median"],
            metrics["ladder_brier_json"],
            metrics["regime"], now,
        ))
        written += 1

    conn.commit()
    conn.close()
    logger.info("[Scorecard] %s S%d W%d: %d segment rows written", sport, season, week, written)
    return written


def compute_season_to_date(sport: str, season: int, through_week: int) -> int:
    """
    Compute aggregated season-to-date scorecard (stored as week=0).
    """
    _ensure_table()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))

    df = pd.read_sql_query(
        """SELECT player_name, stat_type, position, model_version, line_depth,
                  consensus_mean, consensus_median, distributional_median,
                  prop_line, prop_over_price, prop_under_price,
                  p_over, devigged_implied, edge_vs_market,
                  actual_value, created_at, game_id
           FROM fantasy_forward_preds
           WHERE season = ? AND week <= ? AND week > 0
                 AND actual_value IS NOT NULL AND data_error IS NULL""",
        conn, params=(season, through_week)
    )
    if df.empty:
        conn.close()
        return 0

    # Clear old STD rows
    conn.execute(
        "DELETE FROM weekly_model_scorecard WHERE sport=? AND season=? AND week=0",
        (sport, season)
    )

    df["_regime"] = df["created_at"].apply(_assign_regime)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = 0

    for st, pos, mv, ld, sub in _iter_segments(df):
        metrics = _compute_segment(sub, line_depth=ld)
        conn.execute("""
            INSERT OR REPLACE INTO weekly_model_scorecard
                (sport, season, week, stat_type, position, model_version, line_depth,
                 n_preds, mae, median_ae, wape, market_mae,
                 direction_correct, direction_n, direction_pct,
                 direction_expected_pct, direction_edge_pp, direction_lean_json,
                 brier, market_brier, calibration_json, biggest_misses_json,
                 stale_n, stale_correct, stale_mae,
                 cushion_json, asymmetric_mae, mean_signed_error, mae_median,
                 ladder_brier_json,
                 regime, computed_at)
            VALUES (?,?,0,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?, ?,?,?, ?,?,?,?, ?, ?,?)
        """, (
            sport, season, st, pos, mv, ld,
            metrics["n_preds"], metrics["mae"], metrics["median_ae"],
            metrics["wape"], metrics["market_mae"],
            metrics["direction_correct"], metrics["direction_n"], metrics["direction_pct"],
            metrics["direction_expected_pct"], metrics["direction_edge_pp"],
            metrics["direction_lean_json"],
            metrics["brier"], metrics["market_brier"],
            metrics["calibration_json"], metrics["biggest_misses_json"],
            metrics["stale_n"], metrics["stale_correct"], metrics["stale_mae"],
            metrics["cushion_json"], metrics["asymmetric_mae"],
            metrics["mean_signed_error"], metrics["mae_median"],
            metrics["ladder_brier_json"],
            metrics["regime"], now,
        ))
        written += 1

    conn.commit()
    conn.close()
    logger.info("[Scorecard] %s S%d STD (through W%d): %d rows", sport, season, through_week, written)
    return written


def backfill_scorecards(sport: str, season: int) -> dict:
    """
    Backfill all graded weeks.  Returns {week: segments_written}.
    """
    _ensure_table()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    weeks = conn.execute(
        """SELECT DISTINCT week FROM fantasy_forward_preds
           WHERE season = ? AND actual_value IS NOT NULL AND data_error IS NULL
           ORDER BY week""",
        (season,)
    ).fetchall()
    conn.close()

    results = {}
    max_week = 0
    for (w,) in weeks:
        n = compute_weekly_scorecard(sport, season, w, force=True)
        results[w] = n
        if w > max_week:
            max_week = w
        print(f"  W{w}: {n} segments")

    if max_week > 0:
        n_std = compute_season_to_date(sport, season, max_week)
        results["STD"] = n_std
        print(f"  STD (through W{max_week}): {n_std} segments")

    return results


# ── Query ────────────────────────────────────────────────────────────────────

def get_scorecard(sport: str, season: int, week: int = None,
                  stat_type: str = None, position: str = None,
                  model_version: str = None, line_depth: str = None) -> list:
    """
    Fetch scorecard rows with optional filters.  Returns list of dicts.
    Adds Wilson 95% CI for direction_pct.
    """
    _ensure_table()
    conn = sqlite3.connect(os.path.abspath(_HIST_DB))
    conn.row_factory = sqlite3.Row

    clauses = ["sport = ?"]
    params = [sport]

    clauses.append("season = ?")
    params.append(season)

    if week is not None:
        clauses.append("week = ?")
        params.append(week)
    if stat_type:
        clauses.append("stat_type = ?")
        params.append(stat_type)
    if position:
        clauses.append("position = ?")
        params.append(position)
    if model_version:
        clauses.append("model_version = ?")
        params.append(model_version)
    if line_depth:
        clauses.append("line_depth = ?")
        params.append(line_depth)

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM weekly_model_scorecard WHERE {where} ORDER BY week, stat_type, position",
        params
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        # Parse JSON fields
        d["calibration"] = json.loads(d.pop("calibration_json")) if d.get("calibration_json") else []
        d["biggest_misses"] = json.loads(d.pop("biggest_misses_json")) if d.get("biggest_misses_json") else []
        d["direction_lean"] = json.loads(d.pop("direction_lean_json")) if d.get("direction_lean_json") else {}
        d["cushion"] = json.loads(d.pop("cushion_json")) if d.get("cushion_json") else {}
        d["ladder_brier"] = json.loads(d.pop("ladder_brier_json")) if d.get("ladder_brier_json") else []
        # Wilson CI for raw hit rate
        if d.get("direction_n") and d["direction_n"] > 0:
            lo, hi = _wilson_ci(d["direction_correct"], d["direction_n"])
            d["direction_ci_lo"] = lo
            d["direction_ci_hi"] = hi
            # Approximate CI for edge_pp: SE ≈ sqrt(p(1-p)/n) for both actual and expected
            p = d["direction_correct"] / d["direction_n"]
            se = math.sqrt(p * (1 - p) / d["direction_n"])
            edge = d.get("direction_edge_pp") or 0
            d["edge_ci_lo"] = round(edge - 1.96 * se, 4)
            d["edge_ci_hi"] = round(edge + 1.96 * se, 4)
        else:
            d["direction_ci_lo"] = None
            d["direction_ci_hi"] = None
            d["edge_ci_lo"] = None
            d["edge_ci_hi"] = None
        results.append(d)
    return results


# ── Email summary ────────────────────────────────────────────────────────────

def format_email_summary(sport: str, season: int, week: int) -> str:
    """
    Compact markdown summary for the morning email.
    Covers this week + season-to-date headline.
    """
    # Use main lines for accuracy (fair model-vs-market), _ALL_ for direction/calibration
    rows_main = get_scorecard(sport, season, week=week, position=_ALL,
                              model_version=_ALL, line_depth="main")
    rows_all = get_scorecard(sport, season, week=week, position=_ALL,
                             model_version=_ALL, line_depth=_ALL)
    std_rows = get_scorecard(sport, season, week=0, stat_type=_ALL,
                             position=_ALL, model_version=_ALL, line_depth="main")

    if not rows_main and not rows_all:
        return f"**{sport} Scorecard W{week}**: No data.\n"

    lines = [f"## {sport} Model Scorecard — Week {week}\n"]

    # Projection accuracy by stat (main lines only — fair model-vs-market)
    lines.append("### Projection Accuracy — Main Lines (MAE)")
    lines.append("| Stat | N | Model | Market | Delta | Bias |")
    lines.append("|------|---|-------|--------|-------|------|")
    for r in sorted(rows_main, key=lambda x: x["stat_type"]):
        st = r["stat_type"]
        if st == _ALL:
            continue
        delta = r["mae"] - r["market_mae"] if r["mae"] is not None and r["market_mae"] is not None else None
        indicator = ""
        if delta is not None:
            indicator = f" {'▲' if delta < 0 else '▼'}" if delta != 0 else ""
        delta_str = f"{delta:+.1f}{indicator}" if delta is not None else "—"
        bias = r.get("mean_signed_error")
        bias_str = f"{bias:+.1f}" if bias is not None else "—"
        lines.append(f"| {st} | {r['n_preds']} | {r['mae']:.1f} | {r['market_mae']:.1f} | {delta_str} | {bias_str} |")

    all_main = next((r for r in rows_main if r["stat_type"] == _ALL), None)
    if all_main:
        delta = all_main["mae"] - all_main["market_mae"]
        indicator = " ▲" if delta < 0 else " ▼" if delta > 0 else ""
        bias = all_main.get("mean_signed_error")
        bias_str = f"{bias:+.1f}" if bias is not None else "—"
        lines.append(f"| **ALL** | {all_main['n_preds']} | {all_main['mae']:.1f} | {all_main['market_mae']:.1f} | {delta:+.1f}{indicator} | {bias_str} |")

    # Direction vs market-implied (all line depths)
    all_row = next((r for r in rows_all if r["stat_type"] == _ALL), None)
    lines.append("\n### Direction vs Market-Implied")
    lines.append("| Stat | N | Hit% | Mkt Expected | Edge | 95% CI |")
    lines.append("|------|---|------|-------------|------|--------|")
    for r in sorted(rows_all, key=lambda x: x["stat_type"]):
        if r["direction_n"] == 0:
            continue
        st = r["stat_type"]
        pct = r["direction_pct"] * 100
        exp = (r.get("direction_expected_pct") or 0) * 100
        edge = (r.get("direction_edge_pp") or 0) * 100
        elo = (r.get("edge_ci_lo") or 0) * 100
        ehi = (r.get("edge_ci_hi") or 0) * 100
        label = f"**{st}**" if st == _ALL else st
        lines.append(f"| {label} | {r['direction_n']} | {pct:.1f}% | {exp:.1f}% | {edge:+.1f}pp | [{elo:+.1f}, {ehi:+.1f}] |")

    # Brier (all line depths) — overall + per-model_version
    lines.append("\n### Calibration (Brier)")
    if all_row and all_row["brier"] is not None:
        diff = all_row["brier"] - all_row["market_brier"] if all_row["market_brier"] else 0
        better = "better" if diff < 0 else "worse"
        lines.append(f"Model Brier: **{all_row['brier']:.4f}** vs Market: {all_row['market_brier']:.4f} ({better} by {abs(diff):.4f})")

    # Per-model_version Brier breakdown
    mv_rows = get_scorecard(sport, season, week=week, stat_type=_ALL,
                            position=_ALL, line_depth=_ALL)
    mv_brier = [r for r in mv_rows if r["model_version"] != _ALL
                and r["stat_type"] == _ALL and r["position"] == _ALL
                and r["line_depth"] == _ALL and r.get("brier") is not None]
    if len(mv_brier) > 1:
        mv_brier.sort(key=lambda r: r["brier"])
        parts = []
        for r in mv_brier:
            d = r["brier"] - (r["market_brier"] or r["brier"])
            tag = "✓" if d < 0 else ""
            parts.append(f"{r['model_version']}: {r['brier']:.4f} (N={r['n_preds']}){tag}")
        lines.append("By model: " + " · ".join(parts))

    # Top 5 misses (all line depths)
    if all_row and all_row.get("biggest_misses"):
        lines.append("\n### Biggest Misses")
        lines.append("| Player | Stat | Actual | Proj | Line | Error |")
        lines.append("|--------|------|--------|------|------|-------|")
        for m in all_row["biggest_misses"][:5]:
            lines.append(f"| {m['player']} | {m['stat']} | {m['actual']} | {m['projection']} | {m['line']} | {m['error']} |")

    # Stale-input summary
    if all_row and all_row.get("stale_n") and all_row["stale_n"] > 0:
        sn = all_row["stale_n"]
        sm = all_row.get("stale_mae")
        lines.append(f"\n### Stale-Input Flag (|proj - line| > 40% of line)")
        lines.append(f"{sn} predictions flagged ({sn/all_row['n_preds']*100:.0f}% of total). "
                      f"Flagged MAE: {sm:.1f}" if sm else f"{sn} predictions flagged")

    # Season-to-date
    if std_rows:
        std = std_rows[0]
        edge = (std.get("direction_edge_pp") or 0) * 100
        lines.append(f"\n### Season-to-Date")
        lines.append(f"N={std['n_preds']} | MAE {std['mae']:.1f} (mkt {std['market_mae']:.1f}) "
                      f"| Edge {edge:+.1f}pp | Brier {std['brier']:.4f} (mkt {std['market_brier']:.4f})")

    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: model_scorecard.py backfill|compute|email [args]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "backfill":
        sport = sys.argv[2] if len(sys.argv) > 2 else "NFL"
        season = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
        print(f"Backfilling {sport} S{season} scorecard...")
        results = backfill_scorecards(sport, season)
        total = sum(v for v in results.values())
        print(f"Done. {total} total segments across {len(results)} periods.")

    elif cmd == "compute":
        sport = sys.argv[2] if len(sys.argv) > 2 else "NFL"
        season = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
        week = int(sys.argv[4]) if len(sys.argv) > 4 else None
        if week is None:
            print("Usage: model_scorecard.py compute <sport> <season> <week>")
            sys.exit(1)
        n = compute_weekly_scorecard(sport, season, week, force=True)
        n_std = compute_season_to_date(sport, season, week)
        print(f"W{week}: {n} segments, STD: {n_std} segments")

    elif cmd == "email":
        sport = sys.argv[2] if len(sys.argv) > 2 else "NFL"
        season = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
        week = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        print(format_email_summary(sport, season, week))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
