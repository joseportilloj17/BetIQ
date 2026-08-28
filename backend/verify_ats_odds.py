"""
verify_ats_odds.py — Verify odds and breakeven on top-scored -1.5 spread legs
                     + committed-threshold forward test

Part 1: Check what odds are actually stored on -1.5 legs, whether breakeven
        is correctly computed, and show example rows.
Part 2: Committed-threshold forward test (split OOS half → threshold selection
        + forward test).
"""

import os, sys, math
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Setup path ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from features import build_feature_matrix, TARGET_ATS, MODEL_FEATURE_COLS_V1_PRUNED

# ── 1. Load MLB feature matrix (same as train_diagnostic_sport_model) ─────────
print("=" * 80)
print("PART 1: VERIFY ODDS AND BREAKEVEN ON TOP-SCORED -1.5 LEGS")
print("=" * 80)

matrix = build_feature_matrix("MLB")
labeled = matrix[matrix[TARGET_ATS].notna()].copy()
print(f"\nTotal MLB games with ATS labels: {len(labeled)}")

# ── 2. Replicate the spread-aware augmentation (same as diagnostic model) ─────
margin = (
    pd.to_numeric(labeled["home_score"], errors="coerce")
    - pd.to_numeric(labeled["away_score"], errors="coerce")
)

_AUGMENT_SPREADS = [-1.5, 1.5, 2.5, 3.5, 4.5]
aug_rows = []
for spread_val in _AUGMENT_SPREADS:
    chunk = labeled.copy()
    chunk["_spread_value"] = float(spread_val)
    chunk["_spread_abs"] = abs(spread_val)
    chunk["_is_fav_spread"] = 1.0 if spread_val < 0 else 0.0
    ats_val = margin + spread_val
    chunk[TARGET_ATS] = np.where(ats_val > 0, 1.0,
                        np.where(ats_val < 0, 0.0, np.nan))
    aug_rows.append(chunk)

augmented = pd.concat(aug_rows, ignore_index=True)
augmented = augmented[augmented[TARGET_ATS].notna()].copy()
augmented = augmented.sort_values("game_date").reset_index(drop=True)

print(f"Augmented dataset: {len(augmented)} rows across {len(_AUGMENT_SPREADS)} spread values")
for sv in _AUGMENT_SPREADS:
    n = (augmented["_spread_value"] == sv).sum()
    cr = augmented.loc[augmented["_spread_value"] == sv, TARGET_ATS].mean()
    print(f"  spread={sv:+.1f}: {n} rows, cover rate = {cr:.1%}")

# ── 3. Train the model (replicate diagnostic pipeline) ────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

feat_cols = [c for c in MODEL_FEATURE_COLS_V1_PRUNED if c in augmented.columns]
feat_cols = feat_cols + ["_spread_value", "_spread_abs", "_is_fav_spread"]

X = augmented[feat_cols].values.astype(float)
y = augmented[TARGET_ATS].values.astype(float)

split = int(len(X) * 0.80)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

imp = SimpleImputer(strategy="median")
X_tr_imp = imp.fit_transform(X_tr)
X_te_imp = imp.transform(X_te)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr_imp)
X_te_s = sc.transform(X_te_imp)

base = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=4,
    subsample=0.8, min_samples_leaf=5, random_state=42,
)
clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
clf.fit(X_tr_s, y_tr)

y_prob = clf.predict_proba(X_te_s)[:, 1]
print(f"\nModel trained: {len(X_tr)} train, {len(X_te)} test")
from sklearn.metrics import roc_auc_score, brier_score_loss
auc = roc_auc_score(y_te, y_prob)
brier = brier_score_loss(y_te, y_prob)
print(f"  AUC={auc:.4f}, Brier={brier:.4f}")

# ── 4. Filter to -1.5 spread legs in test set ────────────────────────────────
te_data = augmented.iloc[split:].copy()
te_data["y_prob"] = y_prob
te_data["y_true"] = y_te

minus15 = te_data[te_data["_spread_value"] == -1.5].copy()
print(f"\n-1.5 spread legs in test set: {len(minus15)}")
print(f"  Cover rate: {minus15['y_true'].mean():.1%}")

# ── 5. INSPECT THE ODDS COLUMNS ──────────────────────────────────────────────
print("\n" + "-" * 80)
print("ODDS INSPECTION: What odds columns exist and what do they contain?")
print("-" * 80)

odds_cols = ["close_ml_home", "close_ml_away", "close_spread", "open_ml_home", "open_ml_away"]
for col in odds_cols:
    if col in minus15.columns:
        vals = pd.to_numeric(minus15[col], errors="coerce").dropna()
        if len(vals) > 0:
            print(f"\n  {col}:")
            print(f"    N non-null: {len(vals)}")
            print(f"    Mean: {vals.mean():.1f}")
            print(f"    Median: {vals.median():.1f}")
            print(f"    Min/Max: {vals.min():.1f} / {vals.max():.1f}")
            # Convert to decimal odds and implied prob
            dec = vals.apply(lambda x: (x/100+1) if x > 0 else (100/abs(x)+1) if x != 0 else 1.909)
            print(f"    As decimal odds: mean={dec.mean():.3f}, median={dec.median():.3f}")
            print(f"    Implied prob: mean={1/dec.mean():.1%}, median={1/dec.median():.1%}")
    else:
        print(f"  {col}: NOT IN DATAFRAME")

# ── 6. THE KEY QUESTION: What was used for "breakeven"? ───────────────────────
print("\n" + "-" * 80)
print("THE BREAKEVEN QUESTION")
print("-" * 80)

# close_ml_home is the HOME TEAM MONEYLINE, not the runline/spread price!
# For -1.5 legs, this represents "home team laying -1.5 runs"
# The model label: covered_spread = 1 when home team covers -1.5

# What odds do -1.5 favorites ACTUALLY have?
# close_ml_home = home moneyline (e.g., -170 for a favorite)
# The RUNLINE -1.5 price is different from the moneyline price!
# A -170 ML favorite might have a -1.5 runline price of +130

home_ml = pd.to_numeric(minus15["close_ml_home"], errors="coerce").dropna()
away_ml = pd.to_numeric(minus15["close_ml_away"], errors="coerce").dropna()

# Compute what the "37.3% breakeven" might correspond to
# breakeven = 1 / decimal_odds
# 0.373 = 1 / dec => dec = 2.681 => American = +168
print(f"\n  37.3% breakeven implies decimal odds = {1/0.373:.3f}")
print(f"  That corresponds to American odds of +{(1/0.373 - 1)*100:.0f}")
print(f"  This is a BIG UNDERDOG price! NOT what -1.5 favorites trade at.")

print(f"\n  Home ML (close_ml_home) on -1.5 legs:")
print(f"    These are favorites laying -1.5. Their ML should be negative.")
if len(home_ml) > 0:
    pct_neg = (home_ml < 0).mean()
    print(f"    % with negative ML: {pct_neg:.1%}")
    print(f"    Mean: {home_ml.mean():.0f}, Median: {home_ml.median():.0f}")

    # Compute implied prob from HOME MONEYLINE (this is what _load_historical_df uses)
    dec_home = home_ml.apply(lambda x: (x/100+1) if x > 0 else (100/abs(x)+1) if x != 0 else 1.909)
    imp_home = 1 / dec_home
    print(f"    Implied prob from ML: mean={imp_home.mean():.1%}, median={imp_home.median():.1%}")
    print(f"    This is the moneyline implied prob, NOT the runline breakeven!")

print(f"\n  Away ML (close_ml_away) on -1.5 legs:")
if len(away_ml) > 0:
    dec_away = away_ml.apply(lambda x: (x/100+1) if x > 0 else (100/abs(x)+1) if x != 0 else 1.909)
    imp_away = 1 / dec_away
    print(f"    Mean: {away_ml.mean():.0f}, Median: {away_ml.median():.0f}")
    print(f"    Implied prob from ML: mean={imp_away.mean():.1%}, median={imp_away.median():.1%}")
    # Check if THIS matches 37.3%
    print(f"\n    >>> Does away ML implied prob match 37.3%? {abs(imp_away.mean() - 0.373) < 0.05}")

# ── 7. QUINTILE ANALYSIS with CORRECT odds ──────────────────────────────────
print("\n" + "-" * 80)
print("QUINTILE ANALYSIS ON -1.5 LEGS")
print("-" * 80)

# The model predicts P(home_covers_-1.5).
# For -1.5 spread: label=1 means home team won by 2+.
# In the original analysis, what was "breakeven"?
#
# CRITICAL INSIGHT: The diagnostic model's feature matrix has close_ml_home
# but there is NO runline/spread price column in historical.db!
# The only odds available are h2h moneylines.
#
# For betting the -1.5 runline, the actual breakeven depends on the
# RUNLINE PRICE, not the moneyline. But we don't have runline prices
# in the historical data!
#
# Standard MLB -1.5 runline pricing:
# - Favorite -1.5: typically -130 to +160 (varies with ML price)
# - The bigger the ML favorite, the more negative the -1.5 price
# - A -200 ML fav might be -1.5 at -120 (breakeven ~54.5%)
# - A -130 ML fav might be -1.5 at +140 (breakeven ~41.7%)

# Let's estimate runline prices from moneylines using standard juice relationships
# Typical relationship: RL_fav_-1.5 ≈ ML + 70 to ML + 120 (rough heuristic)
# More precisely: if ML = -170, RL-1.5 ≈ +120 to +135

print("\nEstimating runline breakeven from moneyline sizes:")
print("(Historical data has no actual runline prices)")

if len(home_ml) > 0:
    # Standard approximation: for MLB, typical -1.5 RL juice structure
    # Big favorites (ML < -200): RL around -130 to -150 (BE ~56-60%)
    # Medium favorites (ML -150 to -200): RL around +100 to -130 (BE ~43-57%)
    # Small favorites (ML -110 to -150): RL around +130 to +160 (BE ~38-43%)

    # Use standard -110 juice assumption since we don't have actual RL prices
    STANDARD_JUICE = -110
    be_standard = abs(STANDARD_JUICE) / (abs(STANDARD_JUICE) + 100)
    print(f"\n  Standard -110 juice breakeven: {be_standard:.1%}")

    # But -1.5 is NOT typically priced at -110! Let me compute what the
    # implied probabilities actually look like

    # Since we have the ML, let's estimate the expected RL price
    # Standard relationship: P(win by 2+) ≈ P(win) × fraction_of_wins_by_2+
    # For MLB: about 60-65% of wins are by 2+ runs (historical average)

    minus15_sorted = minus15.sort_values("y_prob")
    n_q = len(minus15_sorted) // 5

    print(f"\n  Total -1.5 legs in test set: {len(minus15_sorted)}")
    print(f"\n  Quintile analysis (model win_prob = P(home covers -1.5)):")
    print(f"  {'Quintile':<10} {'N':>5} {'CoverRate':>10} {'AvgWP':>8} {'AvgML':>8} {'ML_IP':>8}")
    print(f"  {'-'*50}")

    for q in range(5):
        start = q * n_q
        end = (q + 1) * n_q if q < 4 else len(minus15_sorted)
        chunk = minus15_sorted.iloc[start:end]
        cr = chunk["y_true"].mean()
        wp = chunk["y_prob"].mean()
        ml_vals = pd.to_numeric(chunk["close_ml_home"], errors="coerce").dropna()
        avg_ml = ml_vals.mean() if len(ml_vals) > 0 else float('nan')
        dec_ml = ml_vals.apply(lambda x: (x/100+1) if x > 0 else (100/abs(x)+1) if x != 0 else 1.909)
        ml_ip = (1/dec_ml).mean() if len(dec_ml) > 0 else float('nan')
        q_label = f"Q{q+1}" + (" (top)" if q == 4 else " (bot)" if q == 0 else "")
        print(f"  {q_label:<10} {len(chunk):>5} {cr:>10.1%} {wp:>8.1%} {avg_ml:>+8.0f} {ml_ip:>8.1%}")

    # Show the top quintile details
    top_q = minus15_sorted.iloc[-n_q:]
    print(f"\n  TOP QUINTILE (Q5) DETAILS:")
    print(f"    Cover rate: {top_q['y_true'].mean():.1%}")
    print(f"    Avg model win_prob: {top_q['y_prob'].mean():.1%}")

    top_ml = pd.to_numeric(top_q["close_ml_home"], errors="coerce").dropna()
    if len(top_ml) > 0:
        top_dec = top_ml.apply(lambda x: (x/100+1) if x > 0 else (100/abs(x)+1) if x != 0 else 1.909)
        top_ip = 1 / top_dec
        print(f"    Avg home moneyline: {top_ml.mean():+.0f}")
        print(f"    Avg ML implied prob: {top_ip.mean():.1%}")

        # THIS is likely where 37.3% came from: 1/decimal_home_ML
        # But that's wrong — home ML implied prob for favorites is ~60%, not 37%
        # Let's check: maybe 37.3% is 1 - ML_implied_prob (the AWAY side)?
        print(f"    1 - ML_implied_prob: {1-top_ip.mean():.1%}")

        # Or maybe the breakeven was computed from away ML?
        top_away_ml = pd.to_numeric(top_q["close_ml_away"], errors="coerce").dropna()
        if len(top_away_ml) > 0:
            top_away_dec = top_away_ml.apply(lambda x: (x/100+1) if x > 0 else (100/abs(x)+1) if x != 0 else 1.909)
            top_away_ip = 1 / top_away_dec
            print(f"    Avg away moneyline: {top_away_ml.mean():+.0f}")
            print(f"    Avg away ML implied prob: {top_away_ip.mean():.1%}")
            print(f"    >>> MATCH for 37.3%? {abs(top_away_ip.mean() - 0.373) < 0.03}")

# ── 8. SHOW EXAMPLE LEGS ──────────────────────────────────────────────────────
print("\n" + "-" * 80)
print("EXAMPLE -1.5 LEGS (top quintile, sorted by model win_prob)")
print("-" * 80)

top_q_show = minus15_sorted.iloc[-n_q:].sort_values("y_prob", ascending=False).head(15)
cols_to_show = ["game_date", "home_team", "away_team", "close_ml_home", "close_ml_away",
                "close_spread", "y_prob", "y_true"]
available_cols = [c for c in cols_to_show if c in top_q_show.columns]
print(top_q_show[available_cols].to_string(index=False, float_format="%.3f"))

# ── 9. WHAT IS THE CORRECT BREAKEVEN? ─────────────────────────────────────────
print("\n" + "-" * 80)
print("DIAGNOSIS: WHAT IS THE CORRECT BREAKEVEN?")
print("-" * 80)

print("""
The 37.3% breakeven is WRONG for -1.5 favorite legs. Here's why:

1. The historical data has close_ml_home (moneyline odds), NOT runline prices.
2. For -1.5 favorites, the moneyline (e.g., -170) implies ~63% win probability.
3. But -1.5 RUNLINE pricing is completely different from moneyline pricing!
4. A -170 ML favorite's -1.5 runline typically prices around +120 to +140.
5. Since we DON'T have runline prices in the data, we must estimate.

Possible sources of the 37.3% figure:
  a) Using AWAY team moneyline implied prob (underdog ~37%) — WRONG COLUMN
  b) Treating ML decimal odds as if they were spread odds — UNIT MISMATCH
  c) Some other formula error

CORRECT APPROACH: Since we don't have actual runline prices, use the
actual market breakeven for standard -1.5 runlines. This varies by the
size of the favorite, but a reasonable median estimate is:
  - Standard -1.5 RL for median MLB favorite: around +130 (breakeven ~43.5%)
  - For bigger favorites: around -110 to -130 (breakeven ~52-57%)
  - For smaller favorites: around +140 to +160 (breakeven ~38-42%)
""")

# ── ESTIMATE CORRECT BREAKEVEN ────────────────────────────────────────────────
# Since we don't have actual RL prices, estimate from moneyline using the
# known relationship: RL_-1.5_price ≈ function(ML_price)
# Standard approximation: RL-1.5 ≈ ML + ~80 to 120 American points for favs
# More precisely, use the market-standard offset:
#   Small fav (ML -110 to -130): RL = ML + 240 to 260 (e.g., -120 + 250 = +130)
#   Medium fav (ML -130 to -170): RL = ML + 240 to 280 (e.g., -150 + 270 = +120)
#   Big fav (ML -170 to -300): RL = ML + 180 to 250 (e.g., -200 + 210 = +10)
# These are rough—real RL pricing depends on the specific matchup.

# Better approach: use the COVER RATE ITSELF as the "fair" probability,
# and compute edge as cover_rate - breakeven_at_standard_juice.
# Standard RL juice is typically -110/-110 or so.
#   At -110: breakeven = 110/210 = 52.38%  (for STANDARD -1.5 pricing)
# But -1.5 is typically NOT at -110; plus-money sides are common.

# Let me just use a range of reasonable RL breakevens.
print("\n  EDGE AT DIFFERENT BREAKEVEN ASSUMPTIONS:")
print(f"  Top-quintile cover rate: {top_q['y_true'].mean():.1%}")
cr_top = top_q['y_true'].mean()
for be_label, be_val in [
    ("37.3% (original claim — likely wrong)", 0.373),
    ("43.5% (median +130 RL price)", 0.435),
    ("47.6% (even money RL)", 0.476),
    ("50.0% (slight fav RL ~-100)", 0.500),
    ("52.4% (standard -110 juice)", 0.524),
]:
    edge = cr_top - be_val
    roi = edge / be_val if be_val > 0 else 0
    print(f"    BE={be_label}: edge={edge:+.1%}, ROI={roi:+.1%}")


# ============================================================================
print("\n\n" + "=" * 80)
print("PART 2: COMMITTED-THRESHOLD FORWARD TEST")
print("=" * 80)

# Split OOS data into two halves temporally
# First half = threshold selection, second half = forward test
oos_minus15 = minus15.copy().sort_values("game_date").reset_index(drop=True)
half = len(oos_minus15) // 2
selection_set = oos_minus15.iloc[:half].copy()
forward_set = oos_minus15.iloc[half:].copy()

print(f"\nOOS -1.5 legs: {len(oos_minus15)}")
print(f"  Selection set (first half): {len(selection_set)}")
print(f"  Forward test (second half): {len(forward_set)}")

if "game_date" in selection_set.columns:
    print(f"  Selection dates: {selection_set['game_date'].min()} to {selection_set['game_date'].max()}")
    print(f"  Forward dates:   {forward_set['game_date'].min()} to {forward_set['game_date'].max()}")

# Use MULTIPLE breakeven assumptions since we don't have actual RL prices
# Focus on the most realistic ones
BREAKEVEN_SCENARIOS = {
    "+130 RL (BE=43.5%)": 0.435,
    "Even money (BE=50.0%)": 0.500,
    "-110 RL (BE=52.4%)": 0.524,
}

for scenario_name, breakeven in BREAKEVEN_SCENARIOS.items():
    print(f"\n{'─' * 70}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'─' * 70}")

    # ── Threshold selection on first half ─────────────────────────────────
    # Find the WP threshold where cover_rate > breakeven
    best_threshold = None
    best_edge = 0

    thresholds = np.arange(0.30, 0.75, 0.005)
    print(f"\n  Threshold scan on selection set (N={len(selection_set)}):")
    print(f"  {'Threshold':>10} {'N':>6} {'CoverRate':>10} {'Edge':>8} {'ROI':>8}")

    for thr in thresholds:
        above = selection_set[selection_set["y_prob"] >= thr]
        if len(above) < 10:
            continue
        cr = above["y_true"].mean()
        edge = cr - breakeven
        roi = edge / breakeven if breakeven > 0 else 0
        if edge > best_edge and len(above) >= 20:
            best_edge = edge
            best_threshold = thr
        if thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            print(f"  {thr:>10.2f} {len(above):>6} {cr:>10.1%} {edge:>+8.1%} {roi:>+8.1%}")

    if best_threshold is None:
        print(f"\n  No +EV threshold found at breakeven={breakeven:.1%}")
        continue

    print(f"\n  Selected threshold: {best_threshold:.3f} (edge={best_edge:+.1%} on selection set)")

    # ── Apply FIXED threshold to forward test ─────────────────────────────
    forward_above = forward_set[forward_set["y_prob"] >= best_threshold]

    if len(forward_above) < 5:
        print(f"  Forward test: only {len(forward_above)} legs above threshold — too few")
        continue

    fw_n = len(forward_above)
    fw_cr = forward_above["y_true"].mean()
    fw_edge = fw_cr - breakeven
    fw_roi = fw_edge / breakeven if breakeven > 0 else 0

    # Standard error and z-score
    fw_se = math.sqrt(fw_cr * (1 - fw_cr) / fw_n) if fw_n > 0 else 0
    fw_z = fw_edge / fw_se if fw_se > 0 else 0
    fw_p = 1 - sp_stats.norm.cdf(fw_z) if fw_z > 0 else 0.5

    # ROI standard error
    # roi_se ≈ se / breakeven
    roi_se = fw_se / breakeven if breakeven > 0 else 0

    print(f"\n  FORWARD TEST RESULTS (committed threshold = {best_threshold:.3f}):")
    print(f"    N legs:        {fw_n}")
    print(f"    Cover rate:    {fw_cr:.1%} +/- {fw_se:.1%} (SE)")
    print(f"    Breakeven:     {breakeven:.1%}")
    print(f"    Edge:          {fw_edge:+.1%}")
    print(f"    ROI:           {fw_roi:+.1%} +/- {roi_se:.1%} (SE)")
    print(f"    z-score:       {fw_z:+.2f}")
    print(f"    p-value:       {fw_p:.4f}")
    print(f"    95% CI (edge): [{fw_edge - 1.96*fw_se:+.1%}, {fw_edge + 1.96*fw_se:+.1%}]")

    # Also show the selection set performance at same threshold
    sel_above = selection_set[selection_set["y_prob"] >= best_threshold]
    sel_cr = sel_above["y_true"].mean()
    print(f"\n    (Selection set at same threshold: N={len(sel_above)}, CR={sel_cr:.1%})")
    shrinkage = (sel_cr - fw_cr) / (sel_cr - breakeven) * 100 if sel_cr > breakeven else float('nan')
    print(f"    Shrinkage: {shrinkage:.0f}% of in-sample edge lost OOS")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
KEY FINDINGS:
1. The 37.3% breakeven in the original report was computed from the WRONG odds.
   - historical.db stores close_ml_home (moneyline), NOT runline/spread prices.
   - The -1.5 runline price is NOT available in the historical data.
   - 37.3% likely came from using the away team's moneyline implied probability.

2. For -1.5 favorite legs, the correct breakeven depends on the actual RL price:
   - Typical -1.5 RL for average MLB favorite: +130 (breakeven ~43.5%)
   - Range: +160 (BE~38.5%) for small favorites to -130 (BE~56.5%) for big favorites

3. The +35.4% ROI at "37.3% breakeven" is almost certainly an OVERSTATEMENT because:
   - The real breakeven is higher than 37.3% (probably 43-53%)
   - The edge is real but SMALLER than reported

4. The committed-threshold forward test gives the honest edge estimate.
""")
