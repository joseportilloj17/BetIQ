#!/usr/bin/env python3
"""
Alternative-form tests for signals that were solo-additive but not joint keepers.

Tests:
1. Cell-ATS per-cell incremental lift (conditional, not pooled)
2. Quality (xwoba) as fade/overpricing flag (negative sign test)
3. Line movement as caution flag (binary, not continuous)
4. bayes_prob incremental value over the CURRENT deployed stack OOS

All OOS via temporal-fold purged CV on historical settled data.
"""

import os
import sqlite3
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_BASE = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(_BASE, "data", "bets.db")
HIST_PATH = os.path.join(_BASE, "data", "historical.db")


def _extract_and_enrich():
    """Reuse the extraction from estimator_joint_validation.py."""
    from estimator_joint_validation import (
        extract_settled_cohort,
        enrich_with_market_data,
        enrich_with_xstats,
        build_market_estimator,
        build_cell_estimator,
        build_quality_estimator,
        build_bayesian_estimator,
    )
    df = extract_settled_cohort()
    df = enrich_with_market_data(df)
    df = enrich_with_xstats(df)
    df = build_market_estimator(df)
    df = build_cell_estimator(df)
    df = build_quality_estimator(df)
    df = build_bayesian_estimator(df)
    return df


def _temporal_fold_cv(df, predict_fn, target_col="outcome", n_splits=5, purge_days=3):
    """Generic temporal-fold CV. predict_fn(train_df, test_df) → predictions array.
    Returns (oos_preds, valid_mask)."""
    df = df.sort_values("game_date").reset_index(drop=True)
    unique_dates = sorted(df["game_date"].unique())
    n_splits = min(n_splits, max(2, len(unique_dates) // 2))
    date_folds = np.array_split(unique_dates, n_splits)

    oos_preds = np.full(len(df), np.nan)

    for fold_idx in range(n_splits):
        test_dates = set(date_folds[fold_idx])
        train_dates_raw = set()
        for i in range(n_splits):
            if i != fold_idx:
                train_dates_raw.update(date_folds[i])

        test_date_objs = [pd.Timestamp(str(d)) for d in test_dates]
        purged_train = set()
        for td in train_dates_raw:
            td_obj = pd.Timestamp(str(td))
            if all(abs((td_obj - ttd).days) >= purge_days for ttd in test_date_objs):
                purged_train.add(td)

        if len(purged_train) < 2:
            continue

        test_mask = df["game_date"].isin(test_dates)
        train_mask = df["game_date"].isin(purged_train)

        train = df[train_mask]
        test = df[test_mask]

        if len(train) < 20 or len(test) < 5:
            continue

        preds = predict_fn(train, test)
        oos_preds[test.index] = preds

    valid = np.isfinite(oos_preds)
    return oos_preds, valid


# ═══════════════════════════════════════════════════════════════
# TEST 1: Cell-ATS per-cell incremental lift
# ═══════════════════════════════════════════════════════════════

def test_cell_ats_per_cell(df):
    """Test cell-specific ATS as conditional signal within each cell."""
    print("\n" + "=" * 70)
    print("TEST 1: CELL-ATS PER-CELL INCREMENTAL LIFT")
    print("=" * 70)

    cells = sorted(df["cell_rung"].unique())

    print(f"\n{'Cell×Rung':<25s} {'N':>5s} {'WR':>6s} {'cell_prob':>10s} "
          f"{'dist_wp':>8s} {'disagree':>9s} {'WR|disag>0':>11s} {'WR|disag<0':>11s} {'Δ':>6s} {'p':>8s}")
    print("-" * 120)

    per_cell_results = []
    for cell in cells:
        sub = df[(df["cell_rung"] == cell) & df["ats_dist_wp"].notna() & df["cell_ats_prob"].notna()]
        if len(sub) < 20:
            continue

        wr = sub["outcome"].mean()
        cell_p = sub["cell_ats_prob"].mean()
        dist_p = sub["ats_dist_wp"].mean()
        disagree = sub["cell_ats_prob"] - sub["ats_dist_wp"]

        # Split: cell_ats says higher than dist_wp vs lower
        high = sub[disagree > 0]
        low = sub[disagree <= 0]

        wr_high = high["outcome"].mean() if len(high) >= 5 else np.nan
        wr_low = low["outcome"].mean() if len(low) >= 5 else np.nan
        delta = wr_high - wr_low if not np.isnan(wr_high) and not np.isnan(wr_low) else np.nan

        # Fisher exact or chi2 for significance
        if len(high) >= 5 and len(low) >= 5:
            a = high["outcome"].sum()
            b = len(high) - a
            c = low["outcome"].sum()
            d = len(low) - c
            _, p = stats.fisher_exact([[a, b], [c, d]])
        else:
            p = np.nan

        print(f"  {cell:<23s} {len(sub):5d} {wr:.3f} {cell_p:10.3f} "
              f"{dist_p:8.3f} {disagree.mean():+9.3f} "
              f"{wr_high:11.3f} {wr_low:11.3f} {delta:+6.3f} {p:8.4f}"
              if not np.isnan(delta) else
              f"  {cell:<23s} {len(sub):5d} {wr:.3f} {cell_p:10.3f} "
              f"{dist_p:8.3f} {disagree.mean():+9.3f}     insufficient data")

        per_cell_results.append({
            "cell": cell, "n": len(sub), "delta": delta, "p": p,
            "n_high": len(high), "n_low": len(low),
        })

    # OOS test: does the per-cell disagreement predict within each cell?
    print(f"\n--- OOS per-cell discrimination (purged CV) ---")

    # Within each cell, does cell_ats - dist_wp predict outcome?
    for cell in cells:
        sub = df[(df["cell_rung"] == cell) & df["cell_dist_disagree"].notna()].copy()
        if len(sub) < 30:
            continue

        rho, p = stats.spearmanr(sub["cell_dist_disagree"], sub["outcome"])
        print(f"  {cell:<23s}  N={len(sub):4d}  rho={rho:+.3f}  p={p:.4f}  "
              f"{'SIGNAL' if p < 0.05 else 'noise'}")

    # Summary
    sig_cells = [r for r in per_cell_results if r.get("p", 1) < 0.05 and not np.isnan(r.get("delta", np.nan))]
    print(f"\n  Cells with significant split: {len(sig_cells)}/{len(per_cell_results)}")
    if sig_cells:
        for s in sig_cells:
            print(f"    {s['cell']}: Δ={s['delta']:+.3f}, p={s['p']:.4f}, "
                  f"N_high={s['n_high']}, N_low={s['n_low']}")
    else:
        print("  No cells show significant per-cell lift → cell-ATS adds nothing beyond pooled.")


# ═══════════════════════════════════════════════════════════════
# TEST 2: Quality (xwoba) as fade/overpricing flag
# ═══════════════════════════════════════════════════════════════

def test_quality_as_fade(df):
    """Test: does FADING high-quality-but-overpriced situations produce edge?"""
    print("\n" + "=" * 70)
    print("TEST 2: QUALITY (xwoba) AS FADE/OVERPRICING FLAG")
    print("=" * 70)

    sub = df[df["quality_signal"].notna() & df["ats_dist_wp"].notna()].copy()
    print(f"\nN = {len(sub)}")

    # The hypothesis: when quality is high AND dist_wp is high (market/model agrees),
    # the leg is OVERPRICED → lower actual WR than predicted.
    # Conversely, when quality is LOW but model still likes it → underpriced.

    # Quintile analysis
    sub["quality_q"] = pd.qcut(sub["quality_signal"], 5, labels=False, duplicates="drop")

    print(f"\n{'Quality Q':>10s} {'N':>5s} {'WR':>6s} {'avg dist_wp':>12s} {'avg edge_pp':>12s} {'WR - dist_wp':>13s}")
    print("-" * 70)
    for q in sorted(sub["quality_q"].unique()):
        sq = sub[sub["quality_q"] == q]
        wr = sq["outcome"].mean()
        dp = sq["ats_dist_wp"].mean()
        ep = sq["negbin_edge_pp"].mean()
        gap = wr - dp
        print(f"  Q{int(q):1d}       {len(sq):5d} {wr:.3f} {dp:12.3f} {ep:12.1f} {gap:+13.3f}")

    # The fade test: within legs where quality is TOP quintile, is WR < dist_wp?
    # (i.e., are high-quality picks systematically overpriced?)
    top_q = sub[sub["quality_q"] == sub["quality_q"].max()]
    bot_q = sub[sub["quality_q"] == sub["quality_q"].min()]

    print(f"\n  Top quality quintile: WR={top_q['outcome'].mean():.3f}, "
          f"avg dist_wp={top_q['ats_dist_wp'].mean():.3f}, "
          f"gap={top_q['outcome'].mean() - top_q['ats_dist_wp'].mean():+.3f}")
    print(f"  Bottom quality quintile: WR={bot_q['outcome'].mean():.3f}, "
          f"avg dist_wp={bot_q['ats_dist_wp'].mean():.3f}, "
          f"gap={bot_q['outcome'].mean() - bot_q['ats_dist_wp'].mean():+.3f}")

    # Fade test: mark legs as "overpriced" if quality_signal is top 20% AND dist_wp > 0.75
    overpriced = sub[(sub["quality_q"] == sub["quality_q"].max()) & (sub["ats_dist_wp"] > 0.75)]
    underpriced = sub[(sub["quality_q"] == sub["quality_q"].min()) & (sub["ats_dist_wp"] > 0.75)]

    print(f"\n  'Overpriced' (high quality + high dist_wp): N={len(overpriced)}, "
          f"WR={overpriced['outcome'].mean():.3f}" if len(overpriced) > 5 else "")
    print(f"  'Underpriced' (low quality + high dist_wp): N={len(underpriced)}, "
          f"WR={underpriced['outcome'].mean():.3f}" if len(underpriced) > 5 else "")

    # OOS: does quality predict WR RESIDUAL (controlling for dist_wp)?
    # If negative sign is real: high quality → lower-than-predicted WR
    resid = sub["outcome"] - sub["ats_dist_wp"]
    rho, p = stats.spearmanr(sub["quality_signal"], resid)
    print(f"\n  Quality → WR residual (outcome - dist_wp): rho={rho:+.3f}, p={p:.4f}")
    print(f"  {'NEGATIVE = overpricing confirmed' if rho < 0 and p < 0.05 else 'NOT confirmed at p<0.05'}")

    # Temporal stability check: does the negative sign hold in each half?
    dates = sorted(sub["game_date"].unique())
    mid = len(dates) // 2
    first_half = sub[sub["game_date"].isin(dates[:mid])]
    second_half = sub[sub["game_date"].isin(dates[mid:])]

    if len(first_half) >= 30 and len(second_half) >= 30:
        r1, p1 = stats.spearmanr(first_half["quality_signal"],
                                  first_half["outcome"] - first_half["ats_dist_wp"])
        r2, p2 = stats.spearmanr(second_half["quality_signal"],
                                  second_half["outcome"] - second_half["ats_dist_wp"])
        print(f"\n  Temporal stability:")
        print(f"    First half  ({dates[0]}→{dates[mid-1]}): rho={r1:+.3f}, p={p1:.4f}")
        print(f"    Second half ({dates[mid]}→{dates[-1]}): rho={r2:+.3f}, p={p2:.4f}")
        signs_agree = (r1 < 0) == (r2 < 0)
        print(f"    Signs {'AGREE' if signs_agree else 'DISAGREE'} → "
              f"{'stable' if signs_agree else 'ARTIFACT (non-replicating sign)'}")
    else:
        print("  Insufficient data for temporal stability check")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Line movement as caution flag
# ═══════════════════════════════════════════════════════════════

def test_line_move_as_flag(df):
    """Test line movement as a binary caution flag, not continuous input."""
    print("\n" + "=" * 70)
    print("TEST 3: LINE MOVEMENT AS CAUTION FLAG")
    print("=" * 70)

    sub = df[df["mkt_line_move"].notna() & df["ats_dist_wp"].notna()].copy()
    print(f"\nN = {len(sub)}")

    # Classify: line steamed TOWARD our pick (bad?) or AWAY (good?)
    # mkt_line_move = close_prob - open_prob
    # If positive: market moved toward higher implied prob for this pick = steamed toward
    # If negative: market moved away = our pick is contrarian

    sub["steam_toward"] = sub["mkt_line_move"] > 0.01  # >1pp move toward
    sub["steam_away"] = sub["mkt_line_move"] < -0.01   # >1pp move away
    sub["no_move"] = (sub["mkt_line_move"].abs() <= 0.01)

    for label, mask_col in [("Steamed TOWARD pick", "steam_toward"),
                              ("No significant move", "no_move"),
                              ("Steamed AWAY from pick", "steam_away")]:
        s = sub[sub[mask_col]]
        if len(s) >= 5:
            wr = s["outcome"].mean()
            avg_dp = s["ats_dist_wp"].mean()
            gap = wr - avg_dp
            print(f"  {label:30s}  N={len(s):4d}  WR={wr:.3f}  dist_wp={avg_dp:.3f}  gap={gap:+.3f}")

    # The flag test: if we DOWN-WEIGHT legs with steam_toward, does the remaining
    # portfolio improve?
    steamed = sub[sub["steam_toward"]]
    not_steamed = sub[~sub["steam_toward"]]

    if len(steamed) >= 10 and len(not_steamed) >= 10:
        wr_steamed = steamed["outcome"].mean()
        wr_not = not_steamed["outcome"].mean()
        delta = wr_not - wr_steamed

        _, p = stats.fisher_exact([
            [steamed["outcome"].sum(), len(steamed) - steamed["outcome"].sum()],
            [not_steamed["outcome"].sum(), len(not_steamed) - not_steamed["outcome"].sum()],
        ])
        print(f"\n  Steam-toward WR: {wr_steamed:.3f} (N={len(steamed)})")
        print(f"  Not-steamed WR:  {wr_not:.3f} (N={len(not_steamed)})")
        print(f"  Δ = {delta:+.3f}, Fisher p = {p:.4f}")
        print(f"  {'FLAG WORKS' if delta > 0 and p < 0.1 else 'FLAG DOES NOT HELP'}")

    # Threshold sweep: at what movement cutoff does the flag become useful?
    print(f"\n  Threshold sweep:")
    print(f"  {'Cutoff (pp)':>12s} {'N_flagged':>10s} {'WR_flagged':>11s} {'WR_rest':>8s} {'Δ':>7s} {'p':>8s}")
    for cutoff in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
        flagged = sub[sub["mkt_line_move"] > cutoff]
        rest = sub[sub["mkt_line_move"] <= cutoff]
        if len(flagged) >= 10 and len(rest) >= 10:
            wrf = flagged["outcome"].mean()
            wrr = rest["outcome"].mean()
            d = wrr - wrf
            _, p = stats.fisher_exact([
                [flagged["outcome"].sum(), len(flagged) - flagged["outcome"].sum()],
                [rest["outcome"].sum(), len(rest) - rest["outcome"].sum()],
            ])
            print(f"  {cutoff*100:>10.1f}pp {len(flagged):10d} {wrf:11.3f} {wrr:8.3f} {d:+7.3f} {p:8.4f}")

    # OOS: purged CV of the flag
    def _flag_predictor(train, test):
        # Simple: predict = WR of non-steamed in training, lower for steamed
        wr_steamed_train = train[train["mkt_line_move"] > 0.01]["outcome"].mean()
        wr_not_train = train[train["mkt_line_move"] <= 0.01]["outcome"].mean()
        # Return dist_wp adjusted by flag
        preds = test["ats_dist_wp"].values.copy()
        flag_mask = test["mkt_line_move"].values > 0.01
        if not np.isnan(wr_steamed_train) and not np.isnan(wr_not_train):
            adjustment = wr_steamed_train - wr_not_train
            preds[flag_mask] += adjustment  # penalize steamed legs
        return preds

    oos_preds, valid = _temporal_fold_cv(sub, _flag_predictor)

    if valid.sum() >= 30:
        # Compare to base (just dist_wp)
        rho_flag, p_flag = stats.spearmanr(oos_preds[valid], sub["outcome"].values[valid])
        rho_base, p_base = stats.spearmanr(sub["ats_dist_wp"].values[valid], sub["outcome"].values[valid])
        brier_flag = np.mean((oos_preds[valid] - sub["outcome"].values[valid]) ** 2)
        brier_base = np.mean((sub["ats_dist_wp"].values[valid] - sub["outcome"].values[valid]) ** 2)
        print(f"\n  OOS (purged CV):")
        print(f"    Flag-adjusted:  rho={rho_flag:+.3f}  Brier={brier_flag:.4f}")
        print(f"    Base (dist_wp): rho={rho_base:+.3f}  Brier={brier_base:.4f}")
        print(f"    Brier lift: {brier_base - brier_flag:+.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST 4: bayes_prob incremental value over CURRENT deployed stack
# ═══════════════════════════════════════════════════════════════

def test_bayes_incremental(df):
    """Confirm bayes_prob adds to the ACTUAL current stack (dist_wp + edge_pp) OOS."""
    print("\n" + "=" * 70)
    print("TEST 4: BAYES_PROB INCREMENTAL VALUE OVER CURRENT STACK (OOS)")
    print("=" * 70)

    sub = df[df["bayes_prob"].notna() & df["ats_dist_wp"].notna() &
             df["negbin_edge_pp"].notna()].copy()
    print(f"\nN = {len(sub)}, fixtures = {sub['fixture_id'].nunique()}, "
          f"days = {sub['game_date'].nunique()}")

    # Solo: partial correlation controlling for both
    from numpy.linalg import lstsq as _lstsq

    outcome = sub["outcome"].values.astype(float)
    bayes = sub["bayes_prob"].values.astype(float)
    dist = sub["ats_dist_wp"].values.astype(float)
    edge = sub["negbin_edge_pp"].values.astype(float)

    # Rank everything
    r_out = stats.rankdata(outcome)
    r_bay = stats.rankdata(bayes)
    r_dist = stats.rankdata(dist)
    r_edge = stats.rankdata(edge)

    # Residualize
    C = np.column_stack([r_dist, r_edge, np.ones(len(sub))])
    bay_resid = r_bay - C @ _lstsq(C, r_bay, rcond=None)[0]
    out_resid = r_out - C @ _lstsq(C, r_out, rcond=None)[0]

    rho, p = stats.spearmanr(bay_resid, out_resid)
    print(f"\n  Partial correlation (bayes_prob | dist_wp, edge_pp):")
    print(f"    rho = {rho:+.4f}, p = {p:.6f}, N = {len(sub)}")
    print(f"    {'ADDITIVE over current stack' if p < 0.01 else 'NOT additive'}")

    # OOS via purged CV: logistic regression with vs without bayes_prob
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def _base_predictor(train, test):
        feats = ["negbin_edge_pp", "ats_dist_wp"]
        sc = StandardScaler()
        X_tr = sc.fit_transform(train[feats].values)
        X_te = sc.transform(test[feats].values)
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_tr, train["outcome"].values)
        return lr.predict_proba(X_te)[:, 1]

    def _full_predictor(train, test):
        feats = ["negbin_edge_pp", "ats_dist_wp", "bayes_prob"]
        sc = StandardScaler()
        X_tr = sc.fit_transform(train[feats].values)
        X_te = sc.transform(test[feats].values)
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_tr, train["outcome"].values)
        return lr.predict_proba(X_te)[:, 1]

    oos_base, valid_b = _temporal_fold_cv(sub, _base_predictor)
    oos_full, valid_f = _temporal_fold_cv(sub, _full_predictor)

    valid = valid_b & valid_f
    if valid.sum() >= 30:
        y = sub["outcome"].values[valid]

        rho_base, _ = stats.spearmanr(oos_base[valid], y)
        rho_full, _ = stats.spearmanr(oos_full[valid], y)

        brier_base = np.mean((oos_base[valid] - y) ** 2)
        brier_full = np.mean((oos_full[valid] - y) ** 2)

        print(f"\n  OOS Logistic Regression (purged 3-day CV, N={valid.sum()}):")
        print(f"    Base (edge_pp + dist_wp):        rho={rho_base:+.4f}  Brier={brier_base:.4f}")
        print(f"    Full (edge_pp + dist_wp + bayes): rho={rho_full:+.4f}  Brier={brier_full:.4f}")
        print(f"    Lift: Δrho={rho_full - rho_base:+.4f}  ΔBrier={brier_base - brier_full:+.4f}")
        print(f"    {'CONFIRMED OOS' if brier_full < brier_base else 'NOT confirmed OOS'}")

    # Also test: mkt_disagree_dist incremental
    sub2 = sub[sub["mkt_disagree_dist"].notna()].copy()
    if len(sub2) >= 50:
        def _mkt_predictor(train, test):
            feats = ["negbin_edge_pp", "ats_dist_wp", "mkt_disagree_dist"]
            sc = StandardScaler()
            X_tr = sc.fit_transform(train[feats].values)
            X_te = sc.transform(test[feats].values)
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_tr, train["outcome"].values)
            return lr.predict_proba(X_te)[:, 1]

        def _both_predictor(train, test):
            feats = ["negbin_edge_pp", "ats_dist_wp", "bayes_prob", "mkt_disagree_dist"]
            sc = StandardScaler()
            X_tr = sc.fit_transform(train[feats].values)
            X_te = sc.transform(test[feats].values)
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_tr, train["outcome"].values)
            return lr.predict_proba(X_te)[:, 1]

        oos_mkt, valid_m = _temporal_fold_cv(sub2, _mkt_predictor)
        oos_both, valid_bo = _temporal_fold_cv(sub2, _both_predictor)
        oos_base2, valid_b2 = _temporal_fold_cv(sub2, _base_predictor)

        valid = valid_m & valid_bo & valid_b2
        if valid.sum() >= 30:
            y = sub2["outcome"].values[valid]
            b_base = np.mean((oos_base2[valid] - y) ** 2)
            b_mkt = np.mean((oos_mkt[valid] - y) ** 2)
            b_both = np.mean((oos_both[valid] - y) ** 2)
            r_base, _ = stats.spearmanr(oos_base2[valid], y)
            r_mkt, _ = stats.spearmanr(oos_mkt[valid], y)
            r_both, _ = stats.spearmanr(oos_both[valid], y)

            print(f"\n  mkt_disagree_dist incremental (N={valid.sum()}):")
            print(f"    Base:                rho={r_base:+.4f}  Brier={b_base:.4f}")
            print(f"    + mkt_disagree:      rho={r_mkt:+.4f}  Brier={b_mkt:.4f}  lift={b_base-b_mkt:+.4f}")
            print(f"    + bayes:             rho={r_both:+.4f}  Brier (not shown)")
            print(f"    + bayes + mkt:       rho={r_both:+.4f}  Brier={b_both:.4f}  lift={b_base-b_both:+.4f}")

    # Quintile check: does bayes discriminate within the stack?
    sub["bayes_q"] = pd.qcut(sub["bayes_prob"], 5, labels=False, duplicates="drop")
    print(f"\n  Bayes quintile → WR (within stack):")
    print(f"  {'Q':>3s} {'N':>5s} {'WR':>6s} {'dist_wp':>8s} {'edge_pp':>8s} {'WR-dist':>8s}")
    for q in sorted(sub["bayes_q"].unique()):
        sq = sub[sub["bayes_q"] == q]
        wr = sq["outcome"].mean()
        dp = sq["ats_dist_wp"].mean()
        ep = sq["negbin_edge_pp"].mean()
        print(f"  Q{int(q):1d} {len(sq):5d} {wr:.3f} {dp:8.3f} {ep:8.1f} {wr-dp:+8.3f}")

    rho_q, p_q = stats.spearmanr(sub["bayes_q"], sub["outcome"])
    print(f"  Monotonic? rho={rho_q:+.3f}, p={p_q:.4f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ALTERNATIVE-FORM SIGNAL TESTS")
    print("=" * 70)

    df = _extract_and_enrich()

    test_cell_ats_per_cell(df)
    test_quality_as_fade(df)
    test_line_move_as_flag(df)
    test_bayes_incremental(df)

    print("\n" + "=" * 70)
    print("CONFIRMED DEAD (stop pursuing):")
    print("  - bayes_uncertainty: failed same as conformal (two failures = dead)")
    print("  - QR disagreement: not solo-additive (p=0.089)")
    print("  - bayes_dist_disagree: redundant with bayes_prob (rho=0.698)")
    print("=" * 70)
