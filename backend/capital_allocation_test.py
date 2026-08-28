#!/usr/bin/env python3
"""
Capital Allocation Test
=======================
Re-tests anomaly scores and expected-profit ranking as CAPITAL-ALLOCATION
signals under the correct metric (bankroll growth rate, not total flat PnL).

Three test blocks:
  1. Anomaly score as discrimination input (partial correlation, Brier, quintile)
  2. Expected-profit ranking under bankroll constraint (daily sim, Kelly variants)
  3. Unifying comparison table
"""

import sqlite3, warnings, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
# statsmodels avoided due to numpy binary compat issues; manual OLS used instead

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
DB = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"
SIGNAL_COLS = ["ats_dist_wp", "bayes_team_wp", "negbin_edge_pp",
               "mkt_disagree_dist", "ridge_composite"]
CELL_MAP = {"fav_home": 0, "fav_away": 1, "dog_home": 2, "dog_away": 3}
START_BANKROLL = 1000.0
MAX_DAILY_ALLOC = 0.50        # max 50% of bankroll per day
MAX_LEG_ALLOC_KELLY = 0.10    # cap per leg (Kelly)
ANNUALIZE = np.sqrt(252)

# ── Data loading ────────────────────────────────────────────────────────
def load_data():
    con = sqlite3.connect(DB)
    q = """
    SELECT l.ats_dist_wp, l.bayes_team_wp, l.negbin_edge_pp,
           l.mkt_disagree_dist, l.ridge_composite, l.negbin_cell,
           l.leg_result, l.description, b.game_date,
           b.odds AS parlay_odds, b.id AS bet_id, b.legs AS n_legs,
           l.win_prob, l.predicted_edge_pp
    FROM mock_bet_legs l
    JOIN mock_bets b ON l.mock_bet_id = b.id
    WHERE b.source='shadow_negbin' AND b.cohort='clean'
      AND b.status IN ('SETTLED_WIN','SETTLED_LOSS')
      AND l.leg_result IN ('WIN','LOSS')
    ORDER BY b.game_date
    """
    df = pd.read_sql(q, con)
    con.close()
    df["y"] = (df["leg_result"] == "WIN").astype(int)
    df["cell_num"] = df["negbin_cell"].map(CELL_MAP).fillna(-1).astype(int)
    # Per-leg market odds derived from model prob and edge:
    #   edge_pp = (model_prob - market_implied_prob) * 100
    #   market_implied_prob = model_prob - edge_pp/100
    #   market_odds = 1 / market_implied_prob
    # win_prob is the model's probability; predicted_edge_pp is the edge
    wp = df["win_prob"].fillna(df["ats_dist_wp"]).fillna(0.8)
    edge = df["negbin_edge_pp"].fillna(df["predicted_edge_pp"]).fillna(0)
    mkt_implied = (wp - edge / 100).clip(0.05, 0.99)
    df["leg_odds"] = 1.0 / mkt_implied
    # Also store model win prob for Kelly
    df["model_wp"] = wp
    return df

# ── Temporal purged CV (5-fold, 3-day purge, date-level blocks) ─────────
def make_cv_folds(df, n_folds=5, purge_days=3):
    dates = sorted(df["game_date"].unique())
    n = len(dates)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        test_dates = set(dates[start:end])
        # purge: remove dates within purge_days of any test date
        from datetime import datetime, timedelta
        test_dt = [datetime.strptime(d, "%Y-%m-%d") for d in test_dates]
        purge_dates = set()
        for td in test_dt:
            for delta in range(-purge_days, purge_days + 1):
                purge_dates.add((td + timedelta(days=delta)).strftime("%Y-%m-%d"))
        train_dates = set(dates) - purge_dates
        if len(train_dates) == 0:
            train_dates = set(dates) - test_dates  # fallback
        train_idx = df[df["game_date"].isin(train_dates)].index
        test_idx = df[df["game_date"].isin(test_dates)].index
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds

# ── Impute + scale helper ──────────────────────────────────────────────
def impute_scale(train_X, test_X):
    means = train_X.mean()
    train_X = train_X.fillna(means)
    test_X = test_X.fillna(means)
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    test_Xs = scaler.transform(test_X)
    return train_Xs, test_Xs, means, scaler

# ── Partial Spearman via OLS residuals ─────────────────────────────────
def _ols_residuals(y_vec, X_mat):
    """Manual OLS residuals: y - X @ (X'X)^-1 X'y."""
    X = np.column_stack([np.ones(len(y_vec)), X_mat])
    try:
        beta = np.linalg.lstsq(X, y_vec, rcond=None)[0]
        return y_vec - X @ beta
    except Exception:
        return y_vec

def partial_spearman(x, y, covariates):
    """Regress out covariates from x and y, then Spearman on residuals."""
    try:
        cov = np.asarray(covariates, dtype=float)
        res_x = _ols_residuals(np.asarray(x, dtype=float), cov)
        res_y = _ols_residuals(np.asarray(y, dtype=float), cov)
        rho, p = stats.spearmanr(res_x, res_y)
    except Exception:
        rho, p = np.nan, np.nan
    return rho, p

# ── Brier score ─────────────────────────────────────────────────────────
def brier(y_true, y_prob):
    return np.mean((y_true - y_prob) ** 2)

# ── Bankroll sim helpers ────────────────────────────────────────────────
def sim_bankroll(daily_legs, strategy_fn, start=START_BANKROLL):
    """Run daily bankroll simulation. strategy_fn(day_df, bankroll) -> allocations array."""
    bankroll = start
    daily_returns = []
    bankroll_trace = [start]
    for date, day_df in daily_legs:
        if len(day_df) == 0:
            daily_returns.append(0.0)
            bankroll_trace.append(bankroll)
            continue
        allocs = strategy_fn(day_df, bankroll)
        # leg pnl: (odds-1) if WIN else -1
        pnls = np.where(day_df["y"].values == 1,
                        day_df["leg_odds"].values - 1, -1.0)
        day_pnl = np.sum(allocs * pnls)
        day_ret = day_pnl / bankroll if bankroll > 0 else 0
        daily_returns.append(day_ret)
        bankroll = bankroll * (1 + day_ret)
        bankroll = max(bankroll, 0.01)  # floor
        bankroll_trace.append(bankroll)
    return np.array(daily_returns), bankroll, bankroll_trace

def compute_metrics(daily_returns, final_bankroll):
    dr = np.array(daily_returns)
    if len(dr) == 0 or np.std(dr) == 0:
        return {"final_bankroll": final_bankroll, "sharpe_daily": 0,
                "sharpe_ann": 0, "log_growth_ann": 0, "max_dd": 0}
    sharpe_d = np.mean(dr) / np.std(dr) if np.std(dr) > 0 else 0
    log_growth = np.mean(np.log1p(dr)) * 252
    # max drawdown from cumulative returns
    cum = np.cumprod(1 + dr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()
    return {"final_bankroll": round(final_bankroll, 2),
            "sharpe_daily": round(sharpe_d, 4),
            "sharpe_ann": round(sharpe_d * ANNUALIZE, 4),
            "log_growth_ann": round(log_growth, 4),
            "max_dd": round(max_dd, 4)}

# ── Strategy functions ──────────────────────────────────────────────────
def strat_flat(day_df, bankroll):
    n = len(day_df)
    per_leg = (bankroll * MAX_DAILY_ALLOC) / n
    return np.full(n, per_leg)

def strat_prob_ranked(day_df, bankroll):
    wp = day_df["ats_dist_wp"].fillna(0.5).values.clip(0.01, 0.99)
    weights = wp / wp.sum()
    return weights * bankroll * MAX_DAILY_ALLOC

def strat_eprofit_ranked(day_df, bankroll):
    wp = day_df["ats_dist_wp"].fillna(0.5).values.clip(0.01, 0.99)
    odds = day_df["leg_odds"].values
    ep = wp * (odds - 1) - (1 - wp)
    ep_pos = np.maximum(ep, 0)
    if ep_pos.sum() == 0:
        return np.zeros(len(day_df))
    weights = ep_pos / ep_pos.sum()
    return weights * bankroll * MAX_DAILY_ALLOC

def make_kelly_strat(mode="all", anomaly_scores=None, ep_filter=False):
    """Factory for Kelly-based strategies."""
    def strat(day_df, bankroll):
        wp = day_df["model_wp"].fillna(0.5).values.clip(0.01, 0.99)
        odds = day_df["leg_odds"].values
        kelly = (wp * odds - 1) / (odds - 1)
        kelly = np.maximum(kelly, 0) / 4  # quarter-Kelly
        kelly = np.minimum(kelly, MAX_LEG_ALLOC_KELLY)
        if ep_filter:
            ep = wp * (odds - 1) - (1 - wp)
            # top 50% by E-profit
            if len(ep) > 1:
                threshold = np.median(ep)
                kelly = np.where(ep >= threshold, kelly, 0)
            # also zero out negative E-profit
            kelly = np.where(ep > 0, kelly, 0)
        if anomaly_scores is not None:
            idx = day_df.index
            anom = np.array([anomaly_scores.get(i, 0.5) for i in idx])
            kelly = kelly * anom
        # cap total at MAX_DAILY_ALLOC of bankroll
        total = kelly.sum()
        if total > MAX_DAILY_ALLOC:
            kelly = kelly * (MAX_DAILY_ALLOC / total)
        return kelly * bankroll
    return strat

# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("CAPITAL ALLOCATION TEST")
    print("Anomaly scores + E-profit ranking as bankroll-growth signals")
    print("=" * 80)

    df = load_data()
    print(f"\nLoaded {len(df)} legs | {df['y'].sum()} W / {(1-df['y']).sum():.0f} L "
          f"| WR {df['y'].mean():.1%} | {df['game_date'].nunique()} dates")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    folds = make_cv_folds(df)
    print(f"CV folds: {len(folds)} (purge=3 days)")
    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i}: train={len(tr)}, test={len(te)}")

    # ── TEST 1: ANOMALY SCORE AS DISCRIMINATION INPUT ───────────────────
    print("\n" + "=" * 80)
    print("TEST 1: ANOMALY SCORE AS DISCRIMINATION INPUT")
    print("=" * 80)

    # Storage for anomaly scores (for later bankroll sim)
    all_anomaly_scores = {}

    # 1A: Partial correlation
    print("\n── 1A: Partial Correlation ──")
    fold_results_1a = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        train = df.loc[tr_idx]
        test = df.loc[te_idx]
        # Only fit on WINNING training legs
        win_train = train[train["y"] == 1]
        train_X = win_train[SIGNAL_COLS]
        test_X = test[SIGNAL_COLS]
        train_Xs, test_Xs, means, scaler = impute_scale(train_X, test_X)
        # Fit IsolationForest
        iforest = IsolationForest(contamination=0.1, n_estimators=200,
                                  random_state=42)
        iforest.fit(train_Xs)
        anom_score = iforest.decision_function(test_Xs)  # higher = more normal
        # Store for later use
        for idx, sc in zip(te_idx, anom_score):
            all_anomaly_scores[idx] = sc
        y_test = test["y"].values
        # Raw Spearman
        raw_rho, raw_p = stats.spearmanr(anom_score, y_test)
        # Partial Spearman
        covs = test[SIGNAL_COLS + ["cell_num"]].fillna(means.reindex(
            SIGNAL_COLS + ["cell_num"], fill_value=0))
        covs = covs.fillna(0).values
        part_rho, part_p = partial_spearman(anom_score, y_test, covs)
        fold_results_1a.append({
            "fold": fi, "raw_rho": raw_rho, "raw_p": raw_p,
            "partial_rho": part_rho, "partial_p": part_p, "n": len(test)
        })

    df_1a = pd.DataFrame(fold_results_1a)
    print(f"\n{'Fold':>4} {'Raw ρ':>8} {'Raw p':>8} {'Part ρ':>8} {'Part p':>10} {'N':>5}")
    print("-" * 47)
    for _, r in df_1a.iterrows():
        print(f"{int(r['fold']):>4} {r['raw_rho']:>8.4f} {r['raw_p']:>8.4f} "
              f"{r['partial_rho']:>8.4f} {r['partial_p']:>10.4f} {int(r['n']):>5}")
    pooled_raw = df_1a["raw_rho"].mean()
    pooled_part = df_1a["partial_rho"].mean()
    pooled_p = df_1a["partial_p"].mean()
    print(f"\nPooled:  raw ρ = {pooled_raw:.4f}  |  partial ρ = {pooled_part:.4f}  "
          f"|  avg p = {pooled_p:.4f}")
    sig = "YES" if pooled_p < 0.05 else "NO"
    print(f"Anomaly score adds discrimination beyond existing signals? {sig}")

    # 1B: Brier improvement
    print("\n── 1B: Brier Improvement ──")
    fold_results_1b = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        train = df.loc[tr_idx]
        test = df.loc[te_idx]
        feat_cols = SIGNAL_COLS + ["cell_num"]
        # Impute
        tr_means = train[feat_cols].mean()
        X_tr = train[feat_cols].fillna(tr_means).values
        X_te = test[feat_cols].fillna(tr_means).values
        y_tr = train["y"].values
        y_te = test["y"].values
        # Baseline logistic
        lr_base = LogisticRegression(max_iter=1000, random_state=42)
        lr_base.fit(X_tr, y_tr)
        p_base = lr_base.predict_proba(X_te)[:, 1]
        brier_base = brier(y_te, p_base)
        # Augmented with anomaly score
        anom_te = np.array([all_anomaly_scores.get(i, 0) for i in te_idx])
        # Need to train anomaly on training set and score training set too
        win_train = train[train["y"] == 1]
        tr_sig = win_train[SIGNAL_COLS]
        tr_sig_all = train[SIGNAL_COLS]
        te_sig = test[SIGNAL_COLS]
        tr_means_sig = tr_sig.mean()
        tr_sig = tr_sig.fillna(tr_means_sig)
        tr_sig_all = tr_sig_all.fillna(tr_means_sig)
        te_sig = te_sig.fillna(tr_means_sig)
        scaler = StandardScaler()
        scaler.fit(tr_sig)
        tr_sig_s = scaler.transform(tr_sig_all)
        te_sig_s = scaler.transform(te_sig)
        iforest = IsolationForest(contamination=0.1, n_estimators=200, random_state=42)
        iforest.fit(scaler.transform(tr_sig))
        anom_tr = iforest.decision_function(tr_sig_s)
        anom_te = iforest.decision_function(te_sig_s)
        X_tr_aug = np.column_stack([X_tr, anom_tr])
        X_te_aug = np.column_stack([X_te, anom_te])
        lr_aug = LogisticRegression(max_iter=1000, random_state=42)
        lr_aug.fit(X_tr_aug, y_tr)
        p_aug = lr_aug.predict_proba(X_te_aug)[:, 1]
        brier_aug = brier(y_te, p_aug)
        fold_results_1b.append({
            "fold": fi, "brier_base": brier_base, "brier_aug": brier_aug,
            "delta": brier_aug - brier_base, "n": len(test)
        })

    df_1b = pd.DataFrame(fold_results_1b)
    print(f"\n{'Fold':>4} {'Base Brier':>11} {'Aug Brier':>11} {'Δ Brier':>9} {'N':>5}")
    print("-" * 44)
    for _, r in df_1b.iterrows():
        print(f"{int(r['fold']):>4} {r['brier_base']:>11.5f} {r['brier_aug']:>11.5f} "
              f"{r['delta']:>9.5f} {int(r['n']):>5}")
    avg_delta = df_1b["delta"].mean()
    t_stat, t_p = stats.ttest_1samp(df_1b["delta"].values, 0)
    print(f"\nAvg Δ Brier: {avg_delta:.5f} (negative = improvement)")
    print(f"Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")

    # 1C: Quintile analysis
    print("\n── 1C: Quintile Analysis (anomaly within dist_wp quartile) ──")
    # Collect all test predictions with anomaly scores
    all_test = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        train = df.loc[tr_idx]
        test = df.loc[te_idx].copy()
        win_train = train[train["y"] == 1]
        tr_sig = win_train[SIGNAL_COLS]
        te_sig = test[SIGNAL_COLS]
        tr_means_sig = tr_sig.mean()
        tr_sig = tr_sig.fillna(tr_means_sig)
        te_sig = te_sig.fillna(tr_means_sig)
        scaler = StandardScaler()
        scaler.fit(tr_sig)
        iforest = IsolationForest(contamination=0.1, n_estimators=200, random_state=42)
        iforest.fit(scaler.transform(tr_sig))
        test["anomaly_score"] = iforest.decision_function(scaler.transform(te_sig))
        test["fold"] = fi
        all_test.append(test)
    all_test_df = pd.concat(all_test)

    # Within each dist_wp quartile, anomaly quintile WR
    all_test_df["dwp_q"] = pd.qcut(all_test_df["ats_dist_wp"].fillna(0.5),
                                    4, labels=False, duplicates="drop")
    print(f"\n{'dist_wp Q':>9} {'Anom Q':>7} {'WR':>7} {'N':>5}")
    print("-" * 32)
    for dq in sorted(all_test_df["dwp_q"].unique()):
        sub = all_test_df[all_test_df["dwp_q"] == dq].copy()
        sub["anom_q"] = pd.qcut(sub["anomaly_score"], 5, labels=False,
                                duplicates="drop")
        for aq in sorted(sub["anom_q"].unique()):
            ss = sub[sub["anom_q"] == aq]
            wr = ss["y"].mean()
            print(f"{dq:>9} {aq:>7} {wr:>7.1%} {len(ss):>5}")

    # ── TEST 2: EXPECTED-PROFIT RANKING UNDER BANKROLL CONSTRAINT ───────
    print("\n" + "=" * 80)
    print("TEST 2: EXPECTED-PROFIT RANKING UNDER BANKROLL CONSTRAINT")
    print("=" * 80)

    # Normalize anomaly scores to [0,1] per fold
    anom_norm = {}
    for fi, (tr_idx, te_idx) in enumerate(folds):
        scores = [all_anomaly_scores[i] for i in te_idx]
        mn, mx = min(scores), max(scores)
        rng = mx - mn if mx > mn else 1
        for i, idx in enumerate(te_idx):
            anom_norm[idx] = (scores[i] - mn) / rng

    # 2A: Daily bankroll simulation
    print("\n── 2A: Daily Bankroll Simulation ──")
    # Run on all test folds concatenated (each fold's test dates are disjoint)
    all_daily = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        test = df.loc[te_idx]
        for date, grp in test.groupby("game_date"):
            all_daily.append((date, grp))
    all_daily.sort(key=lambda x: x[0])

    strategies_2a = {
        "Flat": strat_flat,
        "Prob-ranked": strat_prob_ranked,
        "E-profit-ranked": strat_eprofit_ranked,
    }
    print(f"\n{'Strategy':<20} {'Final $':>9} {'Sharpe(d)':>10} {'Sharpe(a)':>10} "
          f"{'LogGrowth':>10} {'MaxDD':>8}")
    print("-" * 70)
    for name, strat in strategies_2a.items():
        dr, fb, _ = sim_bankroll(all_daily, strat)
        m = compute_metrics(dr, fb)
        print(f"{name:<20} {m['final_bankroll']:>9.2f} {m['sharpe_daily']:>10.4f} "
              f"{m['sharpe_ann']:>10.4f} {m['log_growth_ann']:>10.4f} {m['max_dd']:>8.4f}")

    # 2B: Kelly-fractional sizing
    print("\n── 2B: Kelly-Fractional Sizing ──")
    strategies_2b = {
        "QKelly all": make_kelly_strat("all"),
        "QKelly top50% EP": make_kelly_strat("all", ep_filter=True),
        "QKelly anom-wt": make_kelly_strat("all", anomaly_scores=anom_norm),
    }
    print(f"\n{'Strategy':<20} {'Final $':>9} {'Sharpe(d)':>10} {'Sharpe(a)':>10} "
          f"{'LogGrowth':>10} {'MaxDD':>8}")
    print("-" * 70)
    for name, strat in strategies_2b.items():
        dr, fb, _ = sim_bankroll(all_daily, strat)
        m = compute_metrics(dr, fb)
        print(f"{name:<20} {m['final_bankroll']:>9.2f} {m['sharpe_daily']:>10.4f} "
              f"{m['sharpe_ann']:>10.4f} {m['log_growth_ann']:>10.4f} {m['max_dd']:>8.4f}")

    # 2C: Combined — anomaly-weighted Kelly on E-profit-ranked legs
    print("\n── 2C: Combined Strategy ──")
    strat_combined = make_kelly_strat("all", anomaly_scores=anom_norm, ep_filter=True)
    strategies_2c = {
        "Flat Kelly": make_kelly_strat("all"),
        "EP + Anom Kelly": strat_combined,
    }
    print(f"\n{'Strategy':<20} {'Final $':>9} {'Sharpe(d)':>10} {'Sharpe(a)':>10} "
          f"{'LogGrowth':>10} {'MaxDD':>8}")
    print("-" * 70)
    for name, strat in strategies_2c.items():
        dr, fb, _ = sim_bankroll(all_daily, strat)
        m = compute_metrics(dr, fb)
        print(f"{name:<20} {m['final_bankroll']:>9.2f} {m['sharpe_daily']:>10.4f} "
              f"{m['sharpe_ann']:>10.4f} {m['log_growth_ann']:>10.4f} {m['max_dd']:>8.4f}")

    # ── TEST 3: THE UNIFYING TEST ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST 3: UNIFYING COMPARISON")
    print("=" * 80)

    all_strats = {
        "1. Flat (equal)": strat_flat,
        "2. dist_wp-prop": strat_prob_ranked,
        "3. QKelly (all)": make_kelly_strat("all"),
        "4. EP+Anom Kelly": make_kelly_strat("all", anomaly_scores=anom_norm,
                                              ep_filter=True),
    }

    results = {}
    daily_rets = {}
    print(f"\n{'Strategy':<20} {'Final $':>9} {'Sharpe(d)':>10} {'Sharpe(a)':>10} "
          f"{'LogGrowth':>10} {'MaxDD':>8}")
    print("-" * 70)
    for name, strat in all_strats.items():
        dr, fb, _ = sim_bankroll(all_daily, strat)
        m = compute_metrics(dr, fb)
        results[name] = m
        daily_rets[name] = dr
        print(f"{name:<20} {m['final_bankroll']:>9.2f} {m['sharpe_daily']:>10.4f} "
              f"{m['sharpe_ann']:>10.4f} {m['log_growth_ann']:>10.4f} {m['max_dd']:>8.4f}")

    # Paired t-tests vs flat
    print("\nPaired t-tests (daily returns vs Flat):")
    flat_dr = daily_rets["1. Flat (equal)"]
    for name, dr in daily_rets.items():
        if name == "1. Flat (equal)":
            continue
        t, p = stats.ttest_rel(dr, flat_dr)
        print(f"  {name}: t={t:.3f}, p={p:.4f}")

    # ── SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n1. Anomaly partial ρ = {pooled_part:.4f} (p={pooled_p:.4f})")
    if pooled_p < 0.05:
        print("   -> Anomaly score IS additive beyond existing signals")
    else:
        print("   -> Anomaly score is NOT additive beyond existing signals")
    print(f"\n2. Brier Δ from anomaly: {avg_delta:.5f} (p={t_p:.4f})")
    if avg_delta < 0 and t_p < 0.05:
        print("   -> Anomaly IMPROVES probability calibration")
    else:
        print("   -> No significant Brier improvement")
    print(f"\n3. Best bankroll-growth strategy:")
    best = max(results.items(), key=lambda x: x[1]["log_growth_ann"])
    print(f"   {best[0]}: log-growth={best[1]['log_growth_ann']:.4f}, "
          f"Sharpe(ann)={best[1]['sharpe_ann']:.4f}, "
          f"final=${best[1]['final_bankroll']:.2f}")

if __name__ == "__main__":
    main()
