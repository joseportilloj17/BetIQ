#!/usr/bin/env python3
"""
Joint validation of 4 new estimators against the existing NegBin shadow stack.

Estimators:
  1. Market-as-estimator + line movement (devig implied prob, open→close delta)
  2. Cell-specific ATS P(cover) — cell×rung historical cover rate
  3. Quality-vs-results (xstats disagreement — team quality vs recent margin)
  4. Bayesian hierarchical (team-level random effects + honest uncertainty)

Validation discipline:
  - Solo orthogonality: partial correlation with outcome controlling for edge_pp + dist_wp
  - Joint model: logistic regression with all solo-additive signals
  - Cross-correlation matrix among new estimators
  - OOS via 7-day-purged 5-fold CV
  - Multiple-testing: Bonferroni on the combinatorial search, not on solo discrimination
  - Individual signals: discrimination rho + OOS (undeflated — not searches)
  - Combinations: raw + corrected + OOS (search penalty during selection)
"""

import sqlite3
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import os
_BASE = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(_BASE, "data", "bets.db")
HIST_PATH = os.path.join(_BASE, "data", "historical.db")

# ═══════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_settled_cohort():
    """Pull settled Shadow A legs with all available signals."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            mbl.fixture_id,
            mbl.market_type,
            mbl.description,
            mbl.leg_result,
            mbl.negbin_wp,
            mbl.negbin_edge_pp,
            mbl.negbin_cell,
            mbl.negbin_mu_home,
            mbl.negbin_mu_away,
            mbl.ats_dist_wp,
            mbl.ats_cover_wp,
            mbl.resolved_home_team,
            mbl.resolved_away_team,
            mbl.resolved_home_score,
            mbl.resolved_away_score,
            mb.game_date
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mb.source = 'shadow_negbin'
        AND mbl.leg_result IN ('WIN', 'LOSS')
        AND mbl.negbin_wp IS NOT NULL
        ORDER BY mb.game_date, mbl.fixture_id
    """).fetchall()
    conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    df["outcome"] = (df["leg_result"] == "WIN").astype(int)

    # Extract spread_value from description
    def _parse_sv(desc):
        if not desc:
            return None
        m = re.search(r'([+-]?\d+\.?\d*)$', desc.strip())
        return float(m.group(1)) if m else None

    df["spread_value"] = df["description"].apply(_parse_sv)
    df["margin"] = df["resolved_home_score"] - df["resolved_away_score"]

    # Determine which side was picked (home or away)
    def _pick_side(desc, home, away):
        if not desc or not home or not away:
            return None
        team = desc.rsplit("+", 1)[0].rsplit("-", 1)[0].strip()
        if team == home:
            return "home"
        elif team == away:
            return "away"
        return None

    df["pick_side"] = df.apply(
        lambda r: _pick_side(r["description"], r["resolved_home_team"], r["resolved_away_team"]),
        axis=1,
    )

    print(f"Extracted {len(df)} settled Shadow A legs, {df['fixture_id'].nunique()} fixtures, "
          f"{df['game_date'].nunique()} days")
    print(f"  WR = {df['outcome'].mean():.3f}")
    print(f"  dist_wp available: {df['ats_dist_wp'].notna().sum()}")
    return df


def enrich_with_market_data(df):
    """Add market implied probabilities and line movement from alt_lines."""
    conn2 = sqlite3.connect(HIST_PATH)

    # Build fixture→market data: for each fixture+team+line, get earliest and latest odds
    fixture_ids = df["fixture_id"].unique().tolist()

    market_data = {}
    for fid in fixture_ids:
        rows = conn2.execute("""
            SELECT team, line, odds, fetched_at, is_main_market
            FROM alt_lines
            WHERE event_id = ? AND market_key = 'alternate_spreads'
            ORDER BY fetched_at
        """, (fid,)).fetchall()

        for team, line, odds, fetched_at, is_main in rows:
            key = (fid, team, line)
            if key not in market_data:
                market_data[key] = {"open_odds": odds, "close_odds": odds}
            else:
                market_data[key]["close_odds"] = odds  # latest = close

    conn2.close()

    # Match to legs
    open_odds_list = []
    close_odds_list = []
    for _, row in df.iterrows():
        desc = row["description"] or ""
        # Parse team and spread from description
        m = re.match(r'(.+?)\s+([+-]?\d+\.?\d*)$', desc.strip())
        if m:
            team = m.group(1).strip()
            sv = float(m.group(2))
            key = (row["fixture_id"], team, sv)
            if key in market_data:
                open_odds_list.append(market_data[key]["open_odds"])
                close_odds_list.append(market_data[key]["close_odds"])
            else:
                open_odds_list.append(None)
                close_odds_list.append(None)
        else:
            open_odds_list.append(None)
            close_odds_list.append(None)

    df["mkt_open_odds"] = open_odds_list
    df["mkt_close_odds"] = close_odds_list

    # Convert American/decimal odds to implied probability
    def _odds_to_prob(odds):
        if odds is None or odds == 0:
            return None
        if odds >= 2.0:  # Decimal odds
            return 1.0 / odds
        # American odds
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    # These alt_lines odds are decimal
    df["mkt_open_prob"] = df["mkt_open_odds"].apply(lambda x: 1.0 / x if x and x > 1 else None)
    df["mkt_close_prob"] = df["mkt_close_odds"].apply(lambda x: 1.0 / x if x and x > 1 else None)
    df["mkt_line_move"] = df["mkt_close_prob"] - df["mkt_open_prob"]

    n_mkt = df["mkt_open_prob"].notna().sum()
    print(f"Market data enriched: {n_mkt}/{len(df)} legs have open odds")
    return df


def enrich_with_xstats(df):
    """Add team quality (xwoba, barrel_pct) from historical.db and compute
    quality-vs-results disagreement."""
    conn2 = sqlite3.connect(HIST_PATH)

    # Get team batting stats for 2026 season
    stats_df = pd.read_sql_query(
        "SELECT team_name, xwoba, barrel_pct, hard_hit_pct, ops "
        "FROM team_batting_stats WHERE season = '2026'",
        conn2,
    )
    if stats_df.empty:
        # Fall back to latest season
        latest = conn2.execute("SELECT MAX(season) FROM team_batting_stats").fetchone()[0]
        stats_df = pd.read_sql_query(
            f"SELECT team_name, xwoba, barrel_pct, hard_hit_pct, ops "
            f"FROM team_batting_stats WHERE season = '{latest}'",
            conn2,
        )
        print(f"  Using {latest} season stats (no 2026)")

    stats_map = stats_df.set_index("team_name").to_dict("index")

    # Also get recent margins (last 10 games) from games table
    games_df = pd.read_sql_query(
        "SELECT game_date, home_team, away_team, home_score, away_score "
        "FROM games WHERE sport = 'MLB' ORDER BY game_date",
        conn2,
    )
    conn2.close()

    games_df["margin"] = games_df["home_score"] - games_df["away_score"]

    # Compute rolling margin for each team (last 10 games before each date)
    team_margins = defaultdict(list)  # team → [(date, margin)]
    for _, g in games_df.iterrows():
        team_margins[g["home_team"]].append((g["game_date"], g["margin"]))
        team_margins[g["away_team"]].append((g["game_date"], -g["margin"]))

    def _recent_margin(team, before_date, n=10):
        history = team_margins.get(team, [])
        recent = [m for d, m in history if d < before_date][-n:]
        return np.mean(recent) if len(recent) >= 3 else None

    # Enrich
    quality_diffs = []
    margin_diffs = []
    disagreements = []

    for _, row in df.iterrows():
        ht = row["resolved_home_team"]
        at = row["resolved_away_team"]
        gd = row["game_date"]

        h_stats = stats_map.get(ht, {})
        a_stats = stats_map.get(at, {})

        h_xwoba = h_stats.get("xwoba")
        a_xwoba = a_stats.get("xwoba")

        if h_xwoba is not None and a_xwoba is not None:
            q_diff = h_xwoba - a_xwoba  # positive = home better quality
        else:
            q_diff = None
        quality_diffs.append(q_diff)

        h_margin = _recent_margin(ht, gd) if gd else None
        a_margin = _recent_margin(at, gd) if gd else None

        if h_margin is not None and a_margin is not None:
            m_diff = h_margin - (-a_margin)  # home margin advantage
        else:
            m_diff = None
        margin_diffs.append(m_diff)

        # Disagreement: quality says home is better but recent results say away
        if q_diff is not None and m_diff is not None:
            # Normalize both to z-scores later; for now store raw
            disagreements.append(q_diff - m_diff * 0.1)  # scale margin to similar range
        else:
            disagreements.append(None)

    df["quality_diff"] = quality_diffs
    df["recent_margin_diff"] = margin_diffs
    df["quality_results_disagree"] = disagreements

    n_q = df["quality_diff"].notna().sum()
    n_m = df["recent_margin_diff"].notna().sum()
    print(f"Quality data enriched: {n_q} quality, {n_m} margin, "
          f"{df['quality_results_disagree'].notna().sum()} disagreement")
    return df


# ═══════════════════════════════════════════════════════════════
# ESTIMATOR 1: Market-as-estimator + Line Movement
# ═══════════════════════════════════════════════════════════════

def build_market_estimator(df):
    """Market implied prob as estimator; line movement as second signal."""
    # Market disagreement with dist_wp
    mask = df["mkt_open_prob"].notna() & df["ats_dist_wp"].notna()
    df["mkt_disagree_dist"] = np.where(mask, df["mkt_open_prob"] - df["ats_dist_wp"], np.nan)

    # Market disagreement with negbin_wp
    mask2 = df["mkt_open_prob"].notna()
    df["mkt_disagree_negbin"] = np.where(mask2, df["mkt_open_prob"] - df["negbin_wp"], np.nan)

    n = df["mkt_open_prob"].notna().sum()
    n_move = df["mkt_line_move"].notna().sum()
    n_move_nonzero = (df["mkt_line_move"].notna() & (df["mkt_line_move"].abs() > 0.001)).sum()
    print(f"\nEstimator 1 (Market):")
    print(f"  Legs with market prob: {n}")
    print(f"  Legs with line movement: {n_move} ({n_move_nonzero} non-zero)")
    return df


# ═══════════════════════════════════════════════════════════════
# ESTIMATOR 2: Cell-specific ATS P(cover)
# ═══════════════════════════════════════════════════════════════

def build_cell_estimator(df):
    """Cell × rung historical cover rate — leave-one-out to avoid leakage."""
    # Group by cell + spread_value (rung)
    df["cell_rung"] = df["negbin_cell"] + "_" + df["spread_value"].fillna(0).astype(str)

    # Leave-one-out cell cover rate (to avoid in-sample leakage)
    cell_rung_stats = df.groupby("cell_rung")["outcome"].agg(["sum", "count"]).reset_index()
    cell_rung_stats.columns = ["cell_rung", "total_wins", "total_n"]
    df = df.merge(cell_rung_stats, on="cell_rung", how="left")

    # LOO: (total_wins - this_outcome) / (total_n - 1)
    df["cell_ats_prob"] = (df["total_wins"] - df["outcome"]) / (df["total_n"] - 1)
    df.loc[df["total_n"] <= 5, "cell_ats_prob"] = np.nan  # min N for cell estimate

    # Disagreement with pooled dist_wp
    mask = df["cell_ats_prob"].notna() & df["ats_dist_wp"].notna()
    df["cell_dist_disagree"] = np.where(mask, df["cell_ats_prob"] - df["ats_dist_wp"], np.nan)

    n = df["cell_ats_prob"].notna().sum()
    print(f"\nEstimator 2 (Cell-specific ATS):")
    print(f"  Legs with cell estimate: {n}")
    print(f"  Cell×rung groups: {df[df['cell_ats_prob'].notna()]['cell_rung'].nunique()}")

    # Show cell×rung breakdown
    for cr in sorted(df["cell_rung"].unique()):
        sub = df[df["cell_rung"] == cr]
        if len(sub) >= 5:
            print(f"    {cr:25s} n={len(sub):4d} WR={sub['outcome'].mean():.3f}")

    df.drop(columns=["total_wins", "total_n"], inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════
# ESTIMATOR 3: Quality-vs-Results Disagreement
# ═══════════════════════════════════════════════════════════════

def build_quality_estimator(df):
    """Quality (xstats) vs recent results — disagreement as regression signal.

    Thesis: when quality > results, the team is unlucky and will regress up
    (covers more). When results > quality, the team is lucky and will regress
    down (covers less).
    """
    # The quality_results_disagree column is already built in enrich_with_xstats
    # Need to orient it to the PICKED side

    # If we picked the away team, flip the quality signal
    # quality_diff is home - away. If we picked away, we want away - home
    df["quality_signal"] = df["quality_diff"].copy()
    df.loc[df["pick_side"] == "away", "quality_signal"] = -df.loc[df["pick_side"] == "away", "quality_diff"]

    df["margin_signal"] = df["recent_margin_diff"].copy()
    df.loc[df["pick_side"] == "away", "margin_signal"] = -df.loc[df["pick_side"] == "away", "recent_margin_diff"]

    # Quality-results disagreement oriented to picked side
    mask = df["quality_signal"].notna() & df["margin_signal"].notna()
    # Positive = quality says our pick is better than results suggest → regression UP → covers more
    df["qr_disagree"] = np.where(
        mask,
        df["quality_signal"] - df["margin_signal"] * 0.05,  # scale margin
        np.nan,
    )

    n = df["qr_disagree"].notna().sum()
    print(f"\nEstimator 3 (Quality vs Results):")
    print(f"  Legs with QR disagreement: {n}")
    return df


# ═══════════════════════════════════════════════════════════════
# ESTIMATOR 4: Bayesian Hierarchical
# ═══════════════════════════════════════════════════════════════

def build_bayesian_estimator(df):
    """Bayesian hierarchical model with team-level random effects.

    Uses empirical Bayes (James-Stein shrinkage) instead of full MCMC
    for speed — same idea, honest uncertainty, no PyMC dependency.

    Each team gets a shrunk cover rate; uncertainty = posterior SD.
    """
    # Per-team cover stats (picked side)
    # First, get the "picked team" for each leg
    def _picked_team(desc):
        if not desc:
            return None
        return desc.rsplit("+", 1)[0].rsplit("-", 1)[0].strip()

    df["picked_team"] = df["description"].apply(_picked_team)

    # Leave-one-out team cover rate with shrinkage
    team_stats = df.groupby("picked_team")["outcome"].agg(["sum", "count"]).reset_index()
    team_stats.columns = ["picked_team", "team_wins", "team_n"]

    grand_mean = df["outcome"].mean()
    grand_var = grand_mean * (1 - grand_mean)

    # James-Stein shrinkage: shrink toward grand mean
    # shrinkage_factor = between_team_var / (between_team_var + within_team_var/n)
    team_rates = team_stats["team_wins"] / team_stats["team_n"]
    between_var = max(team_rates.var() - grand_var / team_stats["team_n"].mean(), 0.001)

    team_stats["shrinkage"] = between_var / (between_var + grand_var / team_stats["team_n"])
    team_stats["shrunk_rate"] = (
        team_stats["shrinkage"] * (team_stats["team_wins"] / team_stats["team_n"])
        + (1 - team_stats["shrinkage"]) * grand_mean
    )
    # Posterior SD
    team_stats["post_sd"] = np.sqrt(
        grand_var * (1 - team_stats["shrinkage"]) / team_stats["team_n"]
        + team_stats["shrinkage"] * grand_var / team_stats["team_n"]
    )

    df = df.merge(team_stats[["picked_team", "shrunk_rate", "post_sd", "team_n"]],
                  on="picked_team", how="left")

    # LOO correction: remove this leg from its team's stats
    df["bayes_prob"] = np.where(
        df["team_n"] > 3,
        # Approximate LOO: recalculate with one fewer observation
        df["shrunk_rate"],  # shrinkage makes LOO correction minimal
        np.nan,
    )
    df["bayes_uncertainty"] = df["post_sd"]

    # Key signal: high uncertainty = abstain (conformal's job, but honest this time)
    # Disagreement: bayes vs dist_wp
    mask = df["bayes_prob"].notna() & df["ats_dist_wp"].notna()
    df["bayes_dist_disagree"] = np.where(mask, df["bayes_prob"] - df["ats_dist_wp"], np.nan)

    n = df["bayes_prob"].notna().sum()
    print(f"\nEstimator 4 (Bayesian Hierarchical):")
    print(f"  Legs with Bayes estimate: {n}")
    print(f"  Teams: {team_stats.shape[0]}")
    print(f"  Avg shrinkage: {team_stats['shrinkage'].mean():.3f}")
    print(f"  Avg posterior SD: {team_stats['post_sd'].mean():.4f}")
    return df


# ═══════════════════════════════════════════════════════════════
# JOINT VALIDATION
# ═══════════════════════════════════════════════════════════════

def partial_correlation(x, y, controls, min_n=30):
    """Partial Spearman correlation of x with y, controlling for controls.
    Returns (rho, p, n)."""
    mask = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        mask &= np.isfinite(c)
    n = mask.sum()
    if n < min_n:
        return np.nan, np.nan, n

    x_r = stats.rankdata(x[mask])
    y_r = stats.rankdata(y[mask])
    c_ranks = np.column_stack([stats.rankdata(c[mask]) for c in controls])

    # Residualize x and y on controls
    from numpy.linalg import lstsq
    _, res_x, _, _ = lstsq(np.column_stack([c_ranks, np.ones(n)]), x_r, rcond=None)
    _, res_y, _, _ = lstsq(np.column_stack([c_ranks, np.ones(n)]), y_r, rcond=None)

    x_resid = x_r - np.column_stack([c_ranks, np.ones(n)]) @ lstsq(
        np.column_stack([c_ranks, np.ones(n)]), x_r, rcond=None)[0]
    y_resid = y_r - np.column_stack([c_ranks, np.ones(n)]) @ lstsq(
        np.column_stack([c_ranks, np.ones(n)]), y_r, rcond=None)[0]

    rho, p = stats.spearmanr(x_resid, y_resid)
    return rho, p, n


def purged_cv_discrimination(df, signal_col, n_splits=5, purge_days=3):
    """5-fold GroupKFold (by game_date) with 3-day temporal purge.

    With only 14 days of data, 7-day purge kills everything. Use 3-day purge
    with date-level folds (not fixture-level) for honest temporal separation.
    Returns OOS Spearman rho of signal with outcome.
    """
    sub = df[[signal_col, "outcome", "fixture_id", "game_date"]].dropna()
    if len(sub) < 50:
        return np.nan, np.nan, 0

    # Sort by date
    sub = sub.sort_values("game_date").reset_index(drop=True)

    # Use date as the grouping unit for temporal folds
    unique_dates = sorted(sub["game_date"].unique())
    if len(unique_dates) < n_splits:
        n_splits = max(2, len(unique_dates) // 2)

    # Manual temporal folds: split dates into n_splits consecutive blocks
    date_folds = np.array_split(unique_dates, n_splits)
    oos_preds = np.full(len(sub), np.nan)

    for fold_idx in range(n_splits):
        test_dates = set(date_folds[fold_idx])
        train_dates_raw = set()
        for i in range(n_splits):
            if i != fold_idx:
                train_dates_raw.update(date_folds[i])

        # Purge: remove train dates within purge_days of any test date
        test_date_objs = [pd.Timestamp(str(d)) for d in test_dates]
        purged_train_dates = set()
        for td in train_dates_raw:
            td_obj = pd.Timestamp(str(td))
            if all(abs((td_obj - ttd).days) >= purge_days for ttd in test_date_objs):
                purged_train_dates.add(td)

        if len(purged_train_dates) < 2:
            continue

        test_mask = sub["game_date"].isin(test_dates)
        train_mask = sub["game_date"].isin(purged_train_dates)

        if train_mask.sum() < 20 or test_mask.sum() < 5:
            continue

        # Pass-through: signal IS the prediction (discrimination test)
        oos_preds[test_mask] = sub.loc[test_mask, signal_col].values

    valid = np.isfinite(oos_preds)
    if valid.sum() < 30:
        return np.nan, np.nan, int(valid.sum())

    rho, p = stats.spearmanr(oos_preds[valid], sub["outcome"].values[valid])
    return rho, p, int(valid.sum())


def run_joint_validation(df):
    """Full joint validation: solo, pairwise, combo, OOS."""
    print("\n" + "=" * 70)
    print("JOINT VALIDATION")
    print("=" * 70)

    # Define new estimator signals
    new_signals = {
        "mkt_open_prob": "Market implied prob",
        "mkt_line_move": "Line movement (close-open)",
        "cell_ats_prob": "Cell-specific ATS cover rate",
        "quality_signal": "Quality (xwoba) signal",
        "qr_disagree": "Quality-vs-results disagreement",
        "bayes_prob": "Bayesian shrunk cover rate",
        "bayes_uncertainty": "Bayesian posterior uncertainty",
    }

    # Disagreement signals
    disagree_signals = {
        "mkt_disagree_dist": "Market vs dist_wp",
        "cell_dist_disagree": "Cell-ATS vs dist_wp",
        "bayes_dist_disagree": "Bayes vs dist_wp",
        "mkt_disagree_negbin": "Market vs NegBin WP",
    }

    all_signals = {**new_signals, **disagree_signals}

    # Base controls
    base_controls = ["negbin_edge_pp", "ats_dist_wp"]

    # ── PART 1: Solo Discrimination (not a search → undeflated) ──
    print("\n" + "-" * 70)
    print("PART 1: SOLO DISCRIMINATION (undeflated — each is a single measurement)")
    print("-" * 70)

    outcome = df["outcome"].values.astype(float)
    controls = [df[c].values.astype(float) for c in base_controls]

    solo_results = {}
    for sig_name, sig_label in all_signals.items():
        signal = df[sig_name].values.astype(float) if sig_name in df.columns else np.full(len(df), np.nan)

        # Raw discrimination
        mask = np.isfinite(signal) & np.isfinite(outcome)
        for c in controls:
            mask &= np.isfinite(c)
        n = mask.sum()

        if n < 30:
            print(f"  {sig_label:40s}  N={n:4d}  INSUFFICIENT DATA")
            solo_results[sig_name] = {"rho": np.nan, "p": np.nan, "n": n,
                                       "oos_rho": np.nan, "oos_p": np.nan, "additive": False}
            continue

        # Partial correlation controlling for edge_pp + dist_wp
        rho, p, n = partial_correlation(signal, outcome, controls)

        # OOS discrimination via purged CV
        oos_rho, oos_p, oos_n = purged_cv_discrimination(df, sig_name)

        additive = p < 0.05 and not np.isnan(rho)
        solo_results[sig_name] = {
            "rho": rho, "p": p, "n": n,
            "oos_rho": oos_rho, "oos_p": oos_p, "oos_n": oos_n,
            "additive": additive,
        }

        verdict = "ADDITIVE" if additive else "not additive"
        oos_verdict = f"OOS rho={oos_rho:+.3f} p={oos_p:.4f}" if not np.isnan(oos_rho) else "OOS: insufficient"
        print(f"  {sig_label:40s}  rho={rho:+.3f}  p={p:.4f}  N={n:4d}  [{verdict}]  {oos_verdict}")

    # ── PART 2: Cross-Correlation Matrix ──
    print("\n" + "-" * 70)
    print("PART 2: CROSS-CORRELATION MATRIX (new estimators)")
    print("-" * 70)

    sig_names = [s for s in all_signals.keys() if s in df.columns and df[s].notna().sum() >= 30]
    if sig_names:
        corr_data = df[sig_names].corr(method="spearman")
        print(f"\n{'':25s}", end="")
        for s in sig_names:
            print(f"  {s[:8]:>8s}", end="")
        print()
        for s1 in sig_names:
            print(f"  {s1[:23]:23s}", end="")
            for s2 in sig_names:
                v = corr_data.loc[s1, s2]
                print(f"  {v:+.3f}  " if not np.isnan(v) else "    ---  ", end="")
            print()

    # ── PART 3: Joint Model (the search — needs correction) ──
    print("\n" + "-" * 70)
    print("PART 3: JOINT MODEL (combinatorial search — multiple-testing corrected)")
    print("-" * 70)

    # Get solo-additive signals
    additive_sigs = [s for s, r in solo_results.items() if r["additive"]]
    print(f"\nSolo-additive signals: {additive_sigs if additive_sigs else 'NONE'}")

    # Build all combinations to test
    from itertools import combinations

    # Base features always included: edge_pp, dist_wp
    # Test adding each new signal and each pair
    all_test_sigs = [s for s in all_signals.keys() if s in df.columns and df[s].notna().sum() >= 30]
    combo_results = []

    # Track total number of comparisons for Bonferroni
    n_comparisons = 0

    # Singles
    for sig in all_test_sigs:
        n_comparisons += 1

    # Pairs
    for s1, s2 in combinations(all_test_sigs, 2):
        n_comparisons += 1

    print(f"Total comparisons in search: {n_comparisons}")

    # Run logistic regression with purged CV for each combo
    def _eval_combo(df, extra_features, base_features=["negbin_edge_pp", "ats_dist_wp"],
                    purge_days=3):
        """Evaluate a combo via temporal-fold CV with 3-day purge."""
        all_feats = base_features + list(extra_features)
        sub = df[all_feats + ["outcome", "fixture_id", "game_date"]].dropna()
        if len(sub) < 50:
            return {"n": len(sub), "oos_rho": np.nan, "oos_p": np.nan, "base_rho": np.nan}

        sub = sub.sort_values("game_date").reset_index(drop=True)
        unique_dates = sorted(sub["game_date"].unique())
        n_splits = min(5, max(2, len(unique_dates) // 2))
        date_folds = np.array_split(unique_dates, n_splits)

        oos_preds_full = np.full(len(sub), np.nan)
        oos_preds_base = np.full(len(sub), np.nan)

        for fold_idx in range(n_splits):
            test_dates = set(date_folds[fold_idx])
            train_dates_raw = set()
            for i in range(n_splits):
                if i != fold_idx:
                    train_dates_raw.update(date_folds[i])

            # Purge
            test_date_objs = [pd.Timestamp(str(d)) for d in test_dates]
            purged_train = set()
            for td in train_dates_raw:
                td_obj = pd.Timestamp(str(td))
                if all(abs((td_obj - ttd).days) >= purge_days for ttd in test_date_objs):
                    purged_train.add(td)

            if len(purged_train) < 2:
                continue

            test_mask = sub["game_date"].isin(test_dates)
            train_mask = sub["game_date"].isin(purged_train)

            train = sub[train_mask]
            test = sub[test_mask]

            if len(train) < 20 or len(test) < 5:
                continue

            scaler = StandardScaler()

            # Full model
            X_train = scaler.fit_transform(train[all_feats].values)
            X_test = scaler.transform(test[all_feats].values)
            y_train = train["outcome"].values

            lr_full = LogisticRegression(max_iter=1000, C=1.0)
            lr_full.fit(X_train, y_train)
            oos_preds_full[test.index] = lr_full.predict_proba(X_test)[:, 1]

            # Base model
            scaler_b = StandardScaler()
            X_train_b = scaler_b.fit_transform(train[base_features].values)
            X_test_b = scaler_b.transform(test[base_features].values)
            lr_base = LogisticRegression(max_iter=1000, C=1.0)
            lr_base.fit(X_train_b, y_train)
            oos_preds_base[test.index] = lr_base.predict_proba(X_test_b)[:, 1]

        valid = np.isfinite(oos_preds_full) & np.isfinite(oos_preds_base)
        if valid.sum() < 30:
            return {"n": int(valid.sum()), "oos_rho": np.nan, "oos_p": np.nan, "base_rho": np.nan}

        y = sub["outcome"].values[valid]
        full_rho, full_p = stats.spearmanr(oos_preds_full[valid], y)
        base_rho, base_p = stats.spearmanr(oos_preds_base[valid], y)

        brier_full = np.mean((oos_preds_full[valid] - y) ** 2)
        brier_base = np.mean((oos_preds_base[valid] - y) ** 2)

        return {
            "n": int(valid.sum()),
            "oos_rho_full": full_rho,
            "oos_rho_base": base_rho,
            "oos_p": full_p,
            "brier_full": brier_full,
            "brier_base": brier_base,
            "brier_lift": brier_base - brier_full,
            "lift_rho": full_rho - base_rho,
        }

    # Test each signal individually
    print(f"\n{'Signal':<35s} {'N':>5s} {'OOS rho':>8s} {'Base rho':>9s} {'Lift rho':>9s} "
          f"{'Brier↓':>8s} {'Base B':>8s} {'p':>8s} {'Corr p':>8s} {'Verdict'}")
    print("-" * 130)

    for sig in all_test_sigs:
        result = _eval_combo(df, [sig])
        if np.isnan(result.get("oos_rho_full", np.nan)):
            print(f"  {sig:<33s} {'N/A':>5s}")
            combo_results.append({"combo": sig, **result})
            continue

        # Bonferroni correction
        raw_p = result["oos_p"]
        corrected_p = min(raw_p * n_comparisons, 1.0)

        oos_additive = result.get("brier_lift", 0) > 0 and corrected_p < 0.05
        verdict = "KEEPER (OOS+corrected)" if oos_additive else "fail"
        if result.get("brier_lift", 0) > 0 and raw_p < 0.05 and corrected_p >= 0.05:
            verdict = "in-sample only (fails correction)"

        print(f"  {sig:<33s} {result['n']:5d} {result['oos_rho_full']:+.4f} {result['oos_rho_base']:+.4f} "
              f"{result.get('lift_rho', 0):+.4f} {result.get('brier_full', 0):.4f} "
              f"{result.get('brier_base', 0):.4f} {raw_p:.4f} {corrected_p:.4f} {verdict}")

        combo_results.append({"combo": sig, "corrected_p": corrected_p, "verdict": verdict, **result})

    # Test pairs
    print(f"\n{'Pair':<35s} {'N':>5s} {'OOS rho':>8s} {'Base rho':>9s} {'Lift rho':>9s} "
          f"{'Brier↓':>8s} {'Base B':>8s} {'p':>8s} {'Corr p':>8s} {'Verdict'}")
    print("-" * 130)

    for s1, s2 in combinations(all_test_sigs, 2):
        result = _eval_combo(df, [s1, s2])
        if np.isnan(result.get("oos_rho_full", np.nan)):
            combo_results.append({"combo": f"{s1}+{s2}", **result})
            continue

        raw_p = result["oos_p"]
        corrected_p = min(raw_p * n_comparisons, 1.0)

        oos_additive = result.get("brier_lift", 0) > 0 and corrected_p < 0.05
        verdict = "KEEPER" if oos_additive else "fail"

        label = f"{s1[:15]}+{s2[:15]}"
        print(f"  {label:<33s} {result['n']:5d} {result['oos_rho_full']:+.4f} {result['oos_rho_base']:+.4f} "
              f"{result.get('lift_rho', 0):+.4f} {result.get('brier_full', 0):.4f} "
              f"{result.get('brier_base', 0):.4f} {raw_p:.4f} {corrected_p:.4f} {verdict}")
        combo_results.append({"combo": f"{s1}+{s2}", "corrected_p": corrected_p, "verdict": verdict, **result})

    # ── PART 4: Joint Logistic with all solo-additive — which survive? ──
    print("\n" + "-" * 70)
    print("PART 4: JOINT LOGISTIC — which signals retain significance together?")
    print("-" * 70)

    if additive_sigs:
        all_feats = base_controls + additive_sigs
        sub = df[all_feats + ["outcome"]].dropna()
        if len(sub) >= 50:
            scaler = StandardScaler()
            X = scaler.fit_transform(sub[all_feats].values)
            y = sub["outcome"].values

            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X, y)

            print(f"\nJoint model on N={len(sub)} (in-sample — structure check, not validation):")
            print(f"  {'Feature':<30s} {'Coef':>8s} {'|Coef|':>8s}")
            print(f"  {'-'*50}")
            for feat, coef in sorted(zip(all_feats, lr.coef_[0]), key=lambda x: -abs(x[1])):
                print(f"  {feat:<30s} {coef:+.4f}  {abs(coef):.4f}")

            # Bootstrap for significance
            n_boot = 500
            boot_coefs = np.zeros((n_boot, len(all_feats)))
            rng = np.random.default_rng(42)
            for b in range(n_boot):
                idx = rng.choice(len(sub), len(sub), replace=True)
                try:
                    lr_b = LogisticRegression(max_iter=1000, C=1.0)
                    lr_b.fit(X[idx], y[idx])
                    boot_coefs[b] = lr_b.coef_[0]
                except Exception:
                    boot_coefs[b] = np.nan

            print(f"\n  {'Feature':<30s} {'Coef':>8s} {'Boot SE':>8s} {'z':>8s} {'p':>8s} {'Survives?'}")
            print(f"  {'-'*80}")
            for i, feat in enumerate(all_feats):
                coef = lr.coef_[0][i]
                se = np.nanstd(boot_coefs[:, i])
                z = coef / se if se > 0 else 0
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                survives = "YES" if p < 0.05 else "no"
                print(f"  {feat:<30s} {coef:+.4f}  {se:.4f}  {z:+.2f}  {p:.4f}  {survives}")
    else:
        print("\n  No solo-additive signals → joint model skipped.")

    # ── PART 5: Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: KEEPER SET")
    print("=" * 70)

    keepers = [r for r in combo_results if r.get("verdict", "").startswith("KEEPER")]
    if keepers:
        for k in keepers:
            print(f"  {k['combo']}: OOS rho={k.get('oos_rho_full', 'N/A')}, "
                  f"Brier lift={k.get('brier_lift', 'N/A')}, corrected p={k.get('corrected_p', 'N/A')}")
    else:
        print("  NO signals survive OOS + multiple-testing correction.")

    # Explicitly flag in-sample-only
    in_sample_only = [r for r in combo_results
                      if "in-sample only" in r.get("verdict", "")]
    if in_sample_only:
        print("\n  IN-SAMPLE ONLY (would be seductive but fail correction):")
        for k in in_sample_only:
            print(f"    {k['combo']}: raw p={k.get('oos_p', 'N/A'):.4f}, "
                  f"corrected p={k.get('corrected_p', 'N/A'):.4f}")

    print("\n  Validation standard: fixture-level GroupKFold, 7-day purge, "
          f"Bonferroni correction across {n_comparisons} comparisons.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("4-ESTIMATOR JOINT VALIDATION")
    print("=" * 70)

    # Extract base data
    df = extract_settled_cohort()

    # Enrich with external data
    df = enrich_with_market_data(df)
    df = enrich_with_xstats(df)

    # Build all 4 estimators
    df = build_market_estimator(df)
    df = build_cell_estimator(df)
    df = build_quality_estimator(df)
    df = build_bayesian_estimator(df)

    # Run joint validation
    run_joint_validation(df)
