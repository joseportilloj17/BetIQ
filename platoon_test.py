"""
platoon_test.py — Test whether platoon/handedness features improve MLB ATS model.

Platoon is CATEGORICALLY DIFFERENT from prior tests (K-means, residuals, raw Pi,
component interactions) — those were all NEUTRAL because GBC already captured the
signals implicitly. Platoon/handedness is a completely MISSING dimension: the model
has ZERO handedness information.

Test design:
  - 10-seed multi-split (GBC is stochastic)
  - Temporal splits: train < 2026-05-26, early test = 2026-05-26 to 2026-06-15,
    late test = 2026-07-01 to 2026-08-05
  - 7 model variants: baseline, +diff, +full, +gap, +SP hand, +all, +best subset
  - Inversion test: when aggregate and platoon disagree, who is right?
  - Feature importance analysis: subsumption check
  - GBC(n_estimators=500, max_depth=4, lr=0.03, subsample=0.8, min_samples_leaf=5)
  - CalibratedClassifierCV(isotonic, cv=3)
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))

from features import build_feature_matrix, MODEL_FEATURE_COLS_V1_PRUNED, TARGET_ATS

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

N_SEEDS = 10
TRAIN_CUTOFF = "2026-05-26"
EARLY_START = "2026-05-26"
EARLY_END = "2026-06-15"
LATE_START = "2026-07-01"
LATE_END = "2026-08-05"

GBC_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    min_samples_leaf=5,
)

BASELINE_FEATURES = list(MODEL_FEATURE_COLS_V1_PRUNED)

# Platoon feature groups
PLATOON_DIFF = ["platoon_ops_diff"]
PLATOON_FULL = ["h_ops_vs_sp_hand", "a_ops_vs_sp_hand", "platoon_ops_diff"]
PLATOON_GAP = ["h_platoon_gap", "a_platoon_gap", "platoon_gap_diff"]
PLATOON_SP_HAND = ["h_sp_is_lhp", "a_sp_is_lhp"]
PLATOON_ALL = [
    "h_sp_is_lhp", "a_sp_is_lhp",
    "h_ops_vs_sp_hand", "a_ops_vs_sp_hand", "platoon_ops_diff",
    "h_platoon_gap", "a_platoon_gap", "platoon_gap_diff",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ci(n, w, z=1.96):
    """Wilson score interval for win rate."""
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (round((centre - margin) * 100, 1), round((centre + margin) * 100, 1))


def train_calibrated_gbc(X_train, y_train, seed):
    """Train GBC + isotonic calibration. Returns calibrated model."""
    gbc = GradientBoostingClassifier(
        **GBC_PARAMS,
        random_state=seed,
    )
    cal = CalibratedClassifierCV(gbc, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


def evaluate_model(model, X_test, y_test):
    """Return dict of AUC, Brier, accuracy, WR at threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = np.nan
    brier = brier_score_loss(y_test, proba)
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))

    # WR at >= 52% threshold (typical selection threshold)
    mask_52 = proba >= 0.52
    if mask_52.sum() > 0:
        wr_52 = y_test[mask_52].mean()
        n_52 = mask_52.sum()
    else:
        wr_52 = np.nan
        n_52 = 0

    return {
        "auc": auc,
        "brier": brier,
        "accuracy": acc,
        "wr_52": wr_52,
        "n_52": n_52,
        "proba": proba,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND JOIN DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load feature matrix and join platoon features."""
    print("=" * 110)
    print("  PLATOON / HANDEDNESS MISMATCH TEST — MLB ATS Model")
    print("=" * 110)

    print("\n  Loading feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"  Feature matrix: {len(df)} games")

    print("  Loading platoon features...")
    platoon = pd.read_csv(os.path.join(_HERE, "data", "platoon_features.csv"))
    print(f"  Platoon CSV: {len(platoon)} games")

    # Join on game_id
    platoon_cols = [
        "game_id",
        "h_sp_is_lhp", "a_sp_is_lhp",
        "h_ops_vs_sp_hand", "a_ops_vs_sp_hand", "platoon_ops_diff",
        "h_platoon_gap", "a_platoon_gap", "platoon_gap_diff",
    ]
    df = df.merge(platoon[platoon_cols], on="game_id", how="left")

    # Report fill rates
    print("\n  Platoon feature fill rates:")
    for col in platoon_cols[1:]:
        fill = df[col].notna().mean()
        print(f"    {col:25s}: {fill * 100:.1f}%")

    # Prepare target
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df[TARGET_ATS].notna()].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"\n  Games with ATS labels: {len(df)}")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PREPARE SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_splits(df):
    """Create train / early-test / late-test splits."""
    train = df[df["game_date"] < TRAIN_CUTOFF].copy()
    early = df[(df["game_date"] >= EARLY_START) & (df["game_date"] <= EARLY_END)].copy()
    late = df[(df["game_date"] >= LATE_START) & (df["game_date"] <= LATE_END)].copy()

    print(f"\n  Train: {len(train)} games ({train['game_date'].min().date()} to {train['game_date'].max().date()})")
    print(f"  Early test: {len(early)} games ({EARLY_START} to {EARLY_END})")
    print(f"  Late test:  {len(late)} games ({LATE_START} to {LATE_END})")

    return train, early, late


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: MULTI-SEED MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_comparison(df, train, early, late):
    """Train all model variants across 10 seeds, report results."""

    model_specs = {
        "A: Baseline (21)": BASELINE_FEATURES,
        "B: +Platoon diff": BASELINE_FEATURES + PLATOON_DIFF,
        "C: +Platoon full": BASELINE_FEATURES + PLATOON_FULL,
        "D: +Platoon gap": BASELINE_FEATURES + PLATOON_GAP,
        "E: +SP handedness": BASELINE_FEATURES + PLATOON_SP_HAND,
        "F: +All platoon": BASELINE_FEATURES + PLATOON_ALL,
    }

    # Check available features
    for name, feats in model_specs.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"  WARNING: {name} missing columns: {missing}")

    y_train = train[TARGET_ATS].values
    y_early = early[TARGET_ATS].values
    y_late = late[TARGET_ATS].values

    results = {}  # model_name -> {metric -> list_of_values_per_seed}

    # Store baseline and platoon-full probas for inversion test
    baseline_early_probas = []
    baseline_late_probas = []
    platoon_full_early_probas = []
    platoon_full_late_probas = []

    # Store feature importances from last seed for analysis
    feature_importances = {}

    print("\n" + "=" * 110)
    print("  STEP 3: 10-SEED MODEL COMPARISON")
    print("=" * 110)

    t0 = time.time()
    for model_name, feats in model_specs.items():
        print(f"\n  Training: {model_name} ({len(feats)} features)...")
        results[model_name] = {
            "early_auc": [], "early_brier": [], "early_acc": [], "early_wr52": [],
            "late_auc": [], "late_brier": [], "late_acc": [], "late_wr52": [],
            "n_feat": len(feats),
        }

        for seed in range(N_SEEDS):
            # Impute NaNs
            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            X_tr = train[feats].values.astype(float)
            X_tr = imp.fit_transform(X_tr)
            X_tr = scaler.fit_transform(X_tr)

            X_early = scaler.transform(imp.transform(early[feats].values.astype(float)))
            X_late = scaler.transform(imp.transform(late[feats].values.astype(float)))

            model = train_calibrated_gbc(X_tr, y_train, seed=seed * 7 + 42)

            # Evaluate on early
            early_res = evaluate_model(model, X_early, y_early)
            results[model_name]["early_auc"].append(early_res["auc"])
            results[model_name]["early_brier"].append(early_res["brier"])
            results[model_name]["early_acc"].append(early_res["accuracy"])
            results[model_name]["early_wr52"].append(early_res["wr_52"])

            # Evaluate on late
            late_res = evaluate_model(model, X_late, y_late)
            results[model_name]["late_auc"].append(late_res["auc"])
            results[model_name]["late_brier"].append(late_res["brier"])
            results[model_name]["late_acc"].append(late_res["accuracy"])
            results[model_name]["late_wr52"].append(late_res["wr_52"])

            # Store probas for inversion test (all seeds)
            if model_name == "A: Baseline (21)":
                baseline_early_probas.append(early_res["proba"])
                baseline_late_probas.append(late_res["proba"])
            elif model_name == "C: +Platoon full":
                platoon_full_early_probas.append(early_res["proba"])
                platoon_full_late_probas.append(late_res["proba"])

            # Feature importances from last seed
            if seed == N_SEEDS - 1:
                # Extract raw GBC importances from calibrated model
                # CalibratedClassifierCV wraps estimators
                try:
                    raw_importances = np.zeros(len(feats))
                    for cal_est in model.calibrated_classifiers_:
                        raw_importances += cal_est.estimator.feature_importances_
                    raw_importances /= len(model.calibrated_classifiers_)
                    feature_importances[model_name] = dict(zip(feats, raw_importances))
                except Exception as e:
                    print(f"    Could not extract importances: {e}")

    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.1f}s")

    return results, baseline_early_probas, baseline_late_probas, \
           platoon_full_early_probas, platoon_full_late_probas, feature_importances


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: PRINT RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results):
    """Pretty-print the multi-seed results comparison."""

    print("\n" + "=" * 130)
    print("  RESULTS TABLE: 10-Seed Mean +/- StdDev")
    print("=" * 130)

    # Get baseline means for delta computation
    bl = results["A: Baseline (21)"]
    bl_early_auc = np.nanmean(bl["early_auc"])
    bl_late_auc = np.nanmean(bl["late_auc"])
    bl_early_brier = np.nanmean(bl["early_brier"])
    bl_late_brier = np.nanmean(bl["late_brier"])

    header = (
        f"{'Model':>22s}  {'Nf':>3}  "
        f"{'Early AUC':>10}  {'dAUC':>7}  "
        f"{'Late AUC':>10}  {'dAUC':>7}  "
        f"{'Early Brier':>12}  {'dBrier':>8}  "
        f"{'Late Brier':>12}  {'dBrier':>8}  "
        f"{'E_WR52':>7}  {'L_WR52':>7}"
    )
    print(header)
    print("-" * 130)

    for model_name, r in results.items():
        ea = np.nanmean(r["early_auc"])
        la = np.nanmean(r["late_auc"])
        eb = np.nanmean(r["early_brier"])
        lb = np.nanmean(r["late_brier"])
        ea_sd = np.nanstd(r["early_auc"])
        la_sd = np.nanstd(r["late_auc"])
        eb_sd = np.nanstd(r["early_brier"])
        lb_sd = np.nanstd(r["late_brier"])

        d_ea = ea - bl_early_auc
        d_la = la - bl_late_auc
        d_eb = eb - bl_early_brier
        d_lb = lb - bl_late_brier

        ewr = np.nanmean(r["early_wr52"])
        lwr = np.nanmean(r["late_wr52"])

        line = (
            f"{model_name:>22s}  {r['n_feat']:>3d}  "
            f"{ea:.4f}+{ea_sd:.3f}  {d_ea:>+.4f}  "
            f"{la:.4f}+{la_sd:.3f}  {d_la:>+.4f}  "
            f"{eb:.6f}+{eb_sd:.4f}  {d_eb:>+.6f}  "
            f"{lb:.6f}+{lb_sd:.4f}  {d_lb:>+.6f}  "
            f"{ewr * 100:>5.1f}%  {lwr * 100:>5.1f}%"
        )
        print(line)

    # Statistical significance: paired t-test of Brier scores (baseline vs each)
    print("\n  Paired t-test (late Brier) vs baseline:")
    from scipy.stats import ttest_rel
    bl_late = results["A: Baseline (21)"]["late_brier"]
    for model_name, r in results.items():
        if model_name == "A: Baseline (21)":
            continue
        t_stat, p_val = ttest_rel(bl_late, r["late_brier"])
        sig = "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"    {model_name:>22s}:  t={t_stat:>+.3f}  p={p_val:.4f}  "
              f"{'SIGNIFICANT' if p_val < 0.05 else 'not significant'} {sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: INVERSION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_inversion_test(early, late, baseline_early_probas, baseline_late_probas,
                       platoon_full_early_probas, platoon_full_late_probas):
    """
    Identify games where baseline and platoon disagree.
    In those games, does platoon-favored team actually cover?
    """
    print("\n" + "=" * 110)
    print("  STEP 4: INVERSION TEST — Aggregate vs Platoon Signal Disagreement")
    print("=" * 110)

    # Average probas across seeds for more stable estimate
    bl_early_avg = np.mean(baseline_early_probas, axis=0)
    bl_late_avg = np.mean(baseline_late_probas, axis=0)
    pf_early_avg = np.mean(platoon_full_early_probas, axis=0)
    pf_late_avg = np.mean(platoon_full_late_probas, axis=0)

    for period_name, bl_proba, pf_proba, test_df in [
        ("Early", bl_early_avg, pf_early_avg, early),
        ("Late", bl_late_avg, pf_late_avg, late),
    ]:
        print(f"\n  --- {period_name} Test Period ---")

        y_true = test_df[TARGET_ATS].values
        platoon_diff = test_df["platoon_ops_diff"].values

        for threshold in [0.50, 0.52, 0.55, 0.58]:
            # Baseline says home covers (P > threshold)
            # but platoon says away has advantage (platoon_ops_diff < 0)
            inv_home = (bl_proba >= threshold) & (platoon_diff < 0) & np.isfinite(platoon_diff)
            # Baseline says away covers (P < 1 - threshold)
            # but platoon says home has advantage (platoon_ops_diff > 0)
            inv_away = (bl_proba <= (1 - threshold)) & (platoon_diff > 0) & np.isfinite(platoon_diff)

            inversions = inv_home | inv_away
            n_inv = inversions.sum()

            if n_inv < 5:
                print(f"    Threshold {threshold:.0%}: only {n_inv} inversions, skipping")
                continue

            # In inversion games: does the aggregate or platoon win?
            # "Aggregate is right" = the team the baseline picked DID cover
            agg_right = (
                (inv_home & (y_true == 1)) |  # baseline said home covers, home DID cover
                (inv_away & (y_true == 0))     # baseline said away covers, away DID cover
            )
            agg_wr = agg_right[inversions].mean()

            # "Platoon is right" = opposite of aggregate
            platoon_wr = 1 - agg_wr

            ci_lo, ci_hi = wilson_ci(n_inv, int(agg_right[inversions].sum()))
            print(f"    Threshold {threshold:.0%}: {n_inv} inversions | "
                  f"Aggregate WR: {agg_wr * 100:.1f}% [{ci_lo:.1f}-{ci_hi:.1f}] | "
                  f"Platoon WR: {platoon_wr * 100:.1f}%")

    # Additional: raw platoon signal discrimination
    print("\n  --- Raw Platoon Signal Discrimination ---")
    for period_name, test_df in [("Early", early), ("Late", late)]:
        mask = test_df["platoon_ops_diff"].notna()
        if mask.sum() < 10:
            continue
        sub = test_df[mask]
        p_diff = sub["platoon_ops_diff"].values
        y_true = sub[TARGET_ATS].values

        # Platoon says home team has advantage (positive diff)
        pos = p_diff > 0
        if pos.sum() > 0 and (~pos).sum() > 0:
            wr_pos = y_true[pos].mean()
            wr_neg = y_true[~pos].mean()
            from scipy.stats import spearmanr, pointbiserialr
            rho, p_rho = spearmanr(p_diff, y_true)
            rpb, p_rpb = pointbiserialr(p_diff, y_true)
            print(f"    {period_name}: platoon_ops_diff > 0 WR={wr_pos * 100:.1f}% (N={pos.sum()}) | "
                  f"<= 0 WR={wr_neg * 100:.1f}% (N={(~pos).sum()}) | "
                  f"rho={rho:+.4f} (p={p_rho:.4f}) | r_pb={rpb:+.4f} (p={p_rpb:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_feature_importances(feature_importances):
    """Analyze how platoon features rank and whether they subsume existing features."""
    print("\n" + "=" * 110)
    print("  STEP 6: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 110)

    # Compare baseline vs best platoon model importances
    if "A: Baseline (21)" not in feature_importances:
        print("  No baseline importances available")
        return

    bl_imp = feature_importances["A: Baseline (21)"]

    for model_name, imp in feature_importances.items():
        if model_name == "A: Baseline (21)":
            continue

        print(f"\n  --- {model_name} ---")
        # Sort by importance
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)

        # Print top 10 + all platoon features
        platoon_feats = set(PLATOON_ALL)
        print(f"  {'Rank':>4} {'Feature':>30} {'Importance':>12} {'Baseline':>12} {'Delta':>10} {'Note':>10}")
        print("  " + "-" * 85)

        for rank, (feat, val) in enumerate(sorted_imp, 1):
            bl_val = bl_imp.get(feat, np.nan)
            delta = val - bl_val if not np.isnan(bl_val) else np.nan
            is_platoon = "*PLAT*" if feat in platoon_feats else ""
            if rank <= 15 or feat in platoon_feats:
                delta_str = f"{delta:+.6f}" if not np.isnan(delta) else "NEW"
                bl_str = f"{bl_val:.6f}" if not np.isnan(bl_val) else "---"
                print(f"  {rank:>4} {feat:>30} {val:>12.6f} {bl_str:>12} {delta_str:>10} {is_platoon:>10}")

        # Subsumption check: did any baseline feature lose > 20% importance?
        print(f"\n  Subsumption check (features whose importance changed > 20%):")
        for feat in BASELINE_FEATURES:
            if feat in bl_imp and feat in imp:
                bl_v = bl_imp[feat]
                new_v = imp[feat]
                if bl_v > 0.005:  # only check non-trivial features
                    pct_change = (new_v - bl_v) / bl_v * 100
                    if abs(pct_change) > 20:
                        direction = "GAINED" if pct_change > 0 else "LOST"
                        print(f"    {feat:>30}: {bl_v:.6f} -> {new_v:.6f} ({pct_change:+.1f}%) {direction}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: BEST SUBSET (MODEL G)
# ═══════════════════════════════════════════════════════════════════════════════

def run_best_subset(results, df, train, early, late, feature_importances):
    """
    If any platoon group helped, combine the helpers into Model G.
    """
    print("\n" + "=" * 110)
    print("  STEP 7: BEST SUBSET ANALYSIS (Model G)")
    print("=" * 110)

    bl_late_brier = np.nanmean(results["A: Baseline (21)"]["late_brier"])

    # Identify which groups helped (lower late Brier = better)
    helpers = []
    group_map = {
        "B: +Platoon diff": PLATOON_DIFF,
        "C: +Platoon full": PLATOON_FULL,
        "D: +Platoon gap": PLATOON_GAP,
        "E: +SP handedness": PLATOON_SP_HAND,
    }

    for name, group in group_map.items():
        if name in results:
            mean_late_brier = np.nanmean(results[name]["late_brier"])
            delta = mean_late_brier - bl_late_brier
            helped = delta < 0
            print(f"    {name:>22s}: late Brier delta = {delta:+.6f} {'HELPED' if helped else 'HURT/NEUTRAL'}")
            if helped:
                helpers.extend(group)

    if not helpers:
        print("\n  No platoon group improved late Brier. Model G = skip.")
        return results

    helpers = list(dict.fromkeys(helpers))  # deduplicate preserving order
    print(f"\n  Best subset features: {helpers}")

    feats_g = BASELINE_FEATURES + helpers
    print(f"  Model G: {len(feats_g)} features total")

    # Train Model G
    y_train = train[TARGET_ATS].values
    y_early = early[TARGET_ATS].values
    y_late = late[TARGET_ATS].values

    results["G: +Best subset"] = {
        "early_auc": [], "early_brier": [], "early_acc": [], "early_wr52": [],
        "late_auc": [], "late_brier": [], "late_acc": [], "late_wr52": [],
        "n_feat": len(feats_g),
    }

    for seed in range(N_SEEDS):
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr = imp.fit_transform(train[feats_g].values.astype(float))
        X_tr = scaler.fit_transform(X_tr)
        X_early = scaler.transform(imp.transform(early[feats_g].values.astype(float)))
        X_late = scaler.transform(imp.transform(late[feats_g].values.astype(float)))

        model = train_calibrated_gbc(X_tr, y_train, seed=seed * 7 + 42)

        early_res = evaluate_model(model, X_early, y_early)
        late_res = evaluate_model(model, X_late, y_late)

        for src_key, dst_key in [("auc","auc"), ("brier","brier"), ("accuracy","acc"), ("wr_52","wr52")]:
            results["G: +Best subset"][f"early_{dst_key}"].append(early_res[src_key])
            results["G: +Best subset"][f"late_{dst_key}"].append(late_res[src_key])

        # Feature importances from last seed
        if seed == N_SEEDS - 1:
            try:
                raw_importances = np.zeros(len(feats_g))
                for cal_est in model.calibrated_classifiers_:
                    raw_importances += cal_est.estimator.feature_importances_
                raw_importances /= len(model.calibrated_classifiers_)
                feature_importances["G: +Best subset"] = dict(zip(feats_g, raw_importances))
            except Exception:
                pass

    # Print Model G result
    r = results["G: +Best subset"]
    ea = np.nanmean(r["early_auc"])
    la = np.nanmean(r["late_auc"])
    eb = np.nanmean(r["early_brier"])
    lb = np.nanmean(r["late_brier"])
    d_la = la - np.nanmean(results["A: Baseline (21)"]["late_auc"])
    d_lb = lb - bl_late_brier
    ewr = np.nanmean(r["early_wr52"])
    lwr = np.nanmean(r["late_wr52"])

    print(f"\n  Model G result:")
    print(f"    Early AUC: {ea:.4f}  Late AUC: {la:.4f} (dAUC={d_la:+.4f})")
    print(f"    Early Brier: {eb:.6f}  Late Brier: {lb:.6f} (dBrier={d_lb:+.6f})")
    print(f"    Early WR@52%: {ewr * 100:.1f}%  Late WR@52%: {lwr * 100:.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: EARLY vs LATE STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_early_vs_late(results):
    """Report whether platoon holds late vs early — structural stability check."""
    print("\n" + "=" * 110)
    print("  STEP 5: EARLY vs LATE STABILITY ANALYSIS")
    print("=" * 110)
    print("  (Handedness is structural — a lefty is a lefty all year)")
    print("  (If platoon helps EARLY but not LATE, it may be overfit to a short window)")
    print("  (If platoon helps LATE but not EARLY, it adds info where baseline decays)")

    bl = results["A: Baseline (21)"]
    bl_early_auc = np.nanmean(bl["early_auc"])
    bl_late_auc = np.nanmean(bl["late_auc"])
    bl_early_brier = np.nanmean(bl["early_brier"])
    bl_late_brier = np.nanmean(bl["late_brier"])

    print(f"\n  {'Model':>22s}  {'E_dAUC':>8}  {'L_dAUC':>8}  {'E_dBrier':>10}  {'L_dBrier':>10}  {'Holds Late?':>12}")
    print("  " + "-" * 80)

    for model_name, r in results.items():
        if model_name == "A: Baseline (21)":
            continue
        ea = np.nanmean(r["early_auc"]) - bl_early_auc
        la = np.nanmean(r["late_auc"]) - bl_late_auc
        eb = np.nanmean(r["early_brier"]) - bl_early_brier
        lb = np.nanmean(r["late_brier"]) - bl_late_brier

        # "Holds late" = helps or neutral late (within noise floor)
        noise_floor = 0.001  # approximate noise floor for Brier
        if lb < -noise_floor:
            holds = "YES (helps)"
        elif abs(lb) <= noise_floor:
            holds = "NEUTRAL"
        else:
            holds = "NO (hurts)"

        print(f"  {model_name:>22s}  {ea:>+.4f}  {la:>+.4f}  {eb:>+.6f}  {lb:>+.6f}  {holds:>12s}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Step 1: Load
    df = load_data()

    # Step 2: Splits
    train, early, late = prepare_splits(df)

    # Step 3: Multi-seed comparison
    results, bl_early_p, bl_late_p, pf_early_p, pf_late_p, feat_imp = \
        run_model_comparison(df, train, early, late)

    # Step 4: Results table
    print_results_table(results)

    # Step 5: Inversion test
    if pf_early_p and pf_late_p:
        run_inversion_test(early, late, bl_early_p, bl_late_p, pf_early_p, pf_late_p)
    else:
        print("\n  Inversion test skipped (no platoon full probas)")

    # Step 6: Early vs late stability
    analyze_early_vs_late(results)

    # Step 7: Feature importance analysis
    analyze_feature_importances(feat_imp)

    # Step 8: Best subset (Model G)
    results = run_best_subset(results, df, train, early, late, feat_imp)

    # Final updated table with Model G
    if "G: +Best subset" in results:
        print("\n" + "=" * 130)
        print("  FINAL RESULTS TABLE (including Model G)")
        print("=" * 130)
        print_results_table(results)

    # ═══════════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  VERDICT")
    print("=" * 110)

    bl_late_brier = np.nanmean(results["A: Baseline (21)"]["late_brier"])
    bl_late_auc = np.nanmean(results["A: Baseline (21)"]["late_auc"])

    best_model = None
    best_delta = 0.0
    for name, r in results.items():
        if name == "A: Baseline (21)":
            continue
        lb = np.nanmean(r["late_brier"])
        d = lb - bl_late_brier
        if d < best_delta:
            best_delta = d
            best_model = name

    if best_model and best_delta < -0.001:
        print(f"\n  WINNER: {best_model}")
        print(f"  Late Brier improvement: {best_delta:+.6f}")
        la = np.nanmean(results[best_model]["late_auc"])
        print(f"  Late AUC: {la:.4f} (baseline: {bl_late_auc:.4f}, delta: {la - bl_late_auc:+.4f})")
        print("  Platoon/handedness IS a genuine new dimension.")
    elif best_model and best_delta < 0:
        print(f"\n  MARGINAL: {best_model}")
        print(f"  Late Brier improvement: {best_delta:+.6f} (within noise floor ~0.001)")
        print("  Platoon may help marginally but does not clear the noise floor.")
        print("  Verdict: NEUTRAL — not enough signal to deploy.")
    else:
        print("\n  NEUTRAL: No platoon variant improves late Brier.")
        print("  The GBC cannot extract new ATS discrimination from handedness data.")
        print("  Platoon may matter for totals/NegBin (run scoring) but not for ATS (spread covering).")
        print("  This matches the pattern: ATS is saturated at 21 features.")


if __name__ == "__main__":
    main()
