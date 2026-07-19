"""
cv_fullsample.py — 5-fold temporal CV for MLB ATS margin model discrimination.

ISOLATION: Read-only DB access, outputs to /tmp/cv_results/ and /tmp/cv_models/.
Does NOT compute PnL/Sharpe (full-season features corrupt absolute values).
ONLY discrimination metrics: AUC, Spearman rho, Brier, quintile cover rates.
"""
import sys, os, pickle, sqlite3
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss

from features import build_feature_matrix, MODEL_FEATURE_COLS_V1_PRUNED

# ── Setup output dirs ─────────────────────────────────────────────────────────
os.makedirs("/tmp/cv_results", exist_ok=True)
os.makedirs("/tmp/cv_models", exist_ok=True)

# ── Step 1: Load data (read-only) ────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
conn = sqlite3.connect(f"file:{os.path.abspath(DB_PATH)}?mode=ro", uri=True)

print("=" * 70)
print("MLB ATS Full-Sample 5-Fold Temporal CV — Discrimination Only")
print("=" * 70)

df = build_feature_matrix("MLB", conn=conn)
conn.close()

print(f"\nTotal games loaded: {len(df)}")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Ensure numeric scores
df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
df["margin"] = df["home_score"] - df["away_score"]

# Drop rows without scores (unsettled games)
df = df.dropna(subset=["margin"]).copy()
print(f"Games with valid scores: {len(df)}")

# Features
FEATURES = MODEL_FEATURE_COLS_V1_PRUNED
print(f"Using {len(FEATURES)} features: {FEATURES[:5]}...")

# ── Step 2: 5-fold temporal split with purge + embargo ────────────────────────
df = df.sort_values("game_date").reset_index(drop=True)
n = len(df)
fold_size = n // 5

folds = []
for k in range(5):
    start = k * fold_size
    end = (k + 1) * fold_size if k < 4 else n
    folds.append(df.iloc[start:end].index.tolist())

print(f"\n{'='*70}")
print("FOLD CONSTRUCTION")
print(f"{'='*70}")

fold_data = []  # (X_train, y_train, X_test, y_test, test_df)

for k in range(5):
    test_idx = folds[k]
    test_df = df.loc[test_idx].copy()
    test_dates = test_df["game_date"]
    test_min_date = test_dates.min()
    test_max_date = test_dates.max()

    # All other folds as candidate training
    train_idx = []
    for j in range(5):
        if j != k:
            train_idx.extend(folds[j])
    train_df = df.loc[train_idx].copy()

    # PURGE 1: Same-fixture games within 3 days of any test game
    test_matchups = set(
        zip(test_df["home_team"], test_df["away_team"])
    ) | set(
        zip(test_df["away_team"], test_df["home_team"])
    )

    purge_fixture_mask = pd.Series(False, index=train_df.index)
    for _, trow in test_df.iterrows():
        t_date = trow["game_date"]
        matchup_mask = (
            ((train_df["home_team"] == trow["home_team"]) & (train_df["away_team"] == trow["away_team"])) |
            ((train_df["home_team"] == trow["away_team"]) & (train_df["away_team"] == trow["home_team"]))
        )
        date_mask = (train_df["game_date"] - t_date).abs() <= pd.Timedelta(days=3)
        purge_fixture_mask |= (matchup_mask & date_mask)

    # PURGE 2: Temporal embargo — any game within 1 day of test fold boundaries
    embargo_mask = (
        ((train_df["game_date"] - test_min_date).abs() <= pd.Timedelta(days=1)) |
        ((train_df["game_date"] - test_max_date).abs() <= pd.Timedelta(days=1))
    )

    purge_mask = purge_fixture_mask | embargo_mask
    n_purged_fixture = purge_fixture_mask.sum()
    n_purged_embargo = embargo_mask.sum()
    n_purged_total = purge_mask.sum()

    train_df = train_df[~purge_mask].copy()

    # Verify no overlap
    train_game_ids = set(train_df["game_id"])
    test_game_ids = set(test_df["game_id"])
    assert len(train_game_ids & test_game_ids) == 0, f"Fold {k}: overlap detected!"

    print(f"\nFold {k}:")
    print(f"  Test:  {test_min_date.date()} → {test_max_date.date()} ({len(test_df)} games)")
    print(f"  Train: {train_df['game_date'].min().date()} → {train_df['game_date'].max().date()} ({len(train_df)} games)")
    print(f"  Purged: {n_purged_total} (fixture: {n_purged_fixture}, embargo: {n_purged_embargo})")
    print(f"  Date gap: train_max={train_df[train_df['game_date'] < test_min_date]['game_date'].max().date() if len(train_df[train_df['game_date'] < test_min_date]) else 'N/A'}")

    X_train = train_df[FEATURES].values
    y_train = train_df["margin"].values
    X_test = test_df[FEATURES].values
    y_test = test_df["margin"].values

    fold_data.append((X_train, y_train, X_test, y_test, test_df))

# ── Step 3: Retrain per fold ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("MODEL TRAINING")
print(f"{'='*70}")

all_predictions = []

for k, (X_train, y_train, X_test, y_test, test_df) in enumerate(fold_data):
    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Train
    model = HistGradientBoostingRegressor(
        random_state=42,
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
    )
    model.fit(X_train_imp, y_train)

    # Predict
    mu_pred = model.predict(X_test_imp)

    # Residual sigma from train
    train_pred = model.predict(X_train_imp)
    sigma_fold = np.std(y_train - train_pred)

    print(f"\nFold {k}: sigma_fold={sigma_fold:.3f}, train_RMSE={np.sqrt(np.mean((y_train - train_pred)**2)):.3f}")

    # Save model
    with open(f"/tmp/cv_models/fold_{k}.pkl", "wb") as f:
        pickle.dump({"model": model, "imputer": imputer, "sigma": sigma_fold}, f)

    # Store predictions
    pred_df = test_df[["game_id", "game_date", "home_team", "away_team", "home_score", "away_score", "margin"]].copy()
    pred_df["mu_pred"] = mu_pred
    pred_df["sigma_fold"] = sigma_fold
    pred_df["fold"] = k
    all_predictions.append(pred_df)

predictions = pd.concat(all_predictions, ignore_index=True)

# ── Step 4: Compute discrimination metrics ───────────────────────────────────
print(f"\n{'='*70}")
print("DISCRIMINATION METRICS")
print(f"{'='*70}")

SPREADS = [1.5, 2.5, 3.5, 4.5]

# Build game-spread pairs
rows = []
for _, row in predictions.iterrows():
    for S in SPREADS:
        # P(away covers +S) = P(home_margin <= S)
        dist_wp = norm.cdf((S - row["mu_pred"]) / row["sigma_fold"])
        # Actual: away covers +S if home_margin <= S (away loses by less than S or wins)
        actual_cover = 1 if row["margin"] <= S else 0
        rows.append({
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "spread": S,
            "mu_pred": row["mu_pred"],
            "sigma_fold": row["sigma_fold"],
            "dist_wp": dist_wp,
            "actual_cover": actual_cover,
            "fold": row["fold"],
        })

results_df = pd.DataFrame(rows)

# Overall metrics
print(f"\nTotal game-spread observations: {len(results_df)}")
print(f"Overall cover rate: {results_df['actual_cover'].mean():.4f}")

# Spearman rho
rho, rho_p = spearmanr(results_df["dist_wp"], results_df["actual_cover"])
print(f"\nSpearman rho (all spreads): {rho:.4f}  (p={rho_p:.2e})")

# AUC
auc = roc_auc_score(results_df["actual_cover"], results_df["dist_wp"])
print(f"AUC (all spreads):          {auc:.4f}")

# Brier score
brier = brier_score_loss(results_df["actual_cover"], results_df["dist_wp"])
print(f"Brier score (all spreads):  {brier:.4f}")

# Quintile analysis
results_df["quintile"] = pd.qcut(results_df["dist_wp"], 5, labels=False, duplicates="drop")
print(f"\nQuintile Analysis (dist_wp → cover rate):")
print(f"{'Quintile':<10} {'N':<8} {'Mean dist_wp':<15} {'Cover Rate':<12} {'Lift'}")
print("-" * 55)
overall_cr = results_df["actual_cover"].mean()
for q in sorted(results_df["quintile"].unique()):
    subset = results_df[results_df["quintile"] == q]
    cr = subset["actual_cover"].mean()
    dwp = subset["dist_wp"].mean()
    lift = cr / overall_cr
    print(f"{q:<10} {len(subset):<8} {dwp:<15.4f} {cr:<12.4f} {lift:.3f}")

# Per-spread metrics
print(f"\nPer-Spread Metrics:")
print(f"{'Spread':<10} {'N':<8} {'Cover%':<10} {'AUC':<10} {'Rho':<10} {'Brier':<10}")
print("-" * 58)
for S in SPREADS:
    sub = results_df[results_df["spread"] == S]
    s_auc = roc_auc_score(sub["actual_cover"], sub["dist_wp"])
    s_rho, _ = spearmanr(sub["dist_wp"], sub["actual_cover"])
    s_brier = brier_score_loss(sub["actual_cover"], sub["dist_wp"])
    print(f"+{S:<9} {len(sub):<8} {sub['actual_cover'].mean():<10.4f} {s_auc:<10.4f} {s_rho:<10.4f} {s_brier:<10.4f}")

# Monotonicity check
print(f"\nMonotonicity check (quintile cover rates should increase):")
quintile_rates = results_df.groupby("quintile")["actual_cover"].mean().values
monotone = all(quintile_rates[i] <= quintile_rates[i+1] for i in range(len(quintile_rates)-1))
print(f"  Rates: {[f'{r:.4f}' for r in quintile_rates]}")
print(f"  Monotonic: {'YES' if monotone else 'NO'}")

# ── Step 5: Fold integrity validation ────────────────────────────────────────
print(f"\n{'='*70}")
print("FOLD INTEGRITY VALIDATION")
print(f"{'='*70}")
for k in range(5):
    fold_preds = predictions[predictions["fold"] == k]
    other_preds = predictions[predictions["fold"] != k]
    overlap = set(fold_preds["game_id"]) & set(other_preds["game_id"])
    print(f"Fold {k}: {len(fold_preds)} test games, 0 overlap with other folds: {'PASS' if len(overlap) == 0 else 'FAIL'}")

# ── Step 6: Save outputs ─────────────────────────────────────────────────────
results_df.to_csv("/tmp/cv_results/pass_a_predictions.csv", index=False)

summary_lines = [
    "MLB ATS Full-Sample 5-Fold Temporal CV — Discrimination Results",
    "=" * 60,
    f"Total games: {len(predictions)}",
    f"Total game-spread pairs: {len(results_df)}",
    f"Date range: {predictions['game_date'].min()} to {predictions['game_date'].max()}",
    f"Features: {len(FEATURES)} (V1_PRUNED)",
    "",
    "OVERALL DISCRIMINATION:",
    f"  Spearman rho: {rho:.4f} (p={rho_p:.2e})",
    f"  AUC:          {auc:.4f}",
    f"  Brier:        {brier:.4f}",
    "",
    "PER-SPREAD:",
]
for S in SPREADS:
    sub = results_df[results_df["spread"] == S]
    s_auc = roc_auc_score(sub["actual_cover"], sub["dist_wp"])
    s_rho, _ = spearmanr(sub["dist_wp"], sub["actual_cover"])
    s_brier = brier_score_loss(sub["actual_cover"], sub["dist_wp"])
    summary_lines.append(f"  +{S}: AUC={s_auc:.4f}, rho={s_rho:.4f}, Brier={s_brier:.4f}")

summary_lines.append("")
summary_lines.append("QUINTILE COVER RATES:")
for q in sorted(results_df["quintile"].unique()):
    subset = results_df[results_df["quintile"] == q]
    summary_lines.append(f"  Q{q}: {subset['actual_cover'].mean():.4f} (n={len(subset)})")

summary_lines.append(f"\nMonotonic: {'YES' if monotone else 'NO'}")

with open("/tmp/cv_results/pass_a_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print(f"\n{'='*70}")
print("DONE — Outputs saved:")
print(f"  /tmp/cv_results/pass_a_predictions.csv ({len(results_df)} rows)")
print(f"  /tmp/cv_results/pass_a_summary.txt")
print(f"  /tmp/cv_models/fold_0..4.pkl")
print(f"{'='*70}")
