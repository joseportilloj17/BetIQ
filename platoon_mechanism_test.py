"""
platoon_mechanism_test.py — Mechanism confirmation for platoon gap + GAP-feature search.

Part 1: Mechanism tests for platoon_gap_diff
  A. Effect concentration: does platoon gap help MORE in extreme matchups?
  B. Directional check: does it help on BOTH sides (home-favored & away-favored)?
  C. By starter handedness: does it help more vs LHP (larger platoon effects)?

Part 2: GAP-from-aggregate feature search
  - Home/road OPS gap: team's home OPS - overall OPS
  - Recent form gap: 10-game rolling OPS - season OPS
  - Bullpen ERA gap: team bullpen ERA - starter ERA (from game-level pitching data)
  Check orthogonality (r < 0.3 with all V1_PRUNED features), then test additive value.

Protocol: 10 seeds, temporal split, same GBC as platoon_test.py
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd
import sqlite3
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel, spearmanr

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))

from features import build_feature_matrix, MODEL_FEATURE_COLS_V1_PRUNED, TARGET_ATS

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

N_SEEDS = 10
TRAIN_CUTOFF = "2026-05-26"
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
PLATOON_GAP = ["h_platoon_gap", "a_platoon_gap", "platoon_gap_diff"]

DB_PATH = os.path.join(_HERE, "data", "historical.db")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ci(n, w, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round((centre - margin) * 100, 1), round((centre + margin) * 100, 1))


def train_calibrated_gbc(X_train, y_train, seed):
    gbc = GradientBoostingClassifier(**GBC_PARAMS, random_state=seed)
    cal = CalibratedClassifierCV(gbc, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


def evaluate_model(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = np.nan
    brier = brier_score_loss(y_test, proba)
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    return {"auc": auc, "brier": brier, "accuracy": acc, "proba": proba}


def prep_Xy(df, feats):
    """Impute + scale a feature matrix. Returns (X, imp, scaler)."""
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imp.fit_transform(df[feats].values.astype(float))
    X = scaler.fit_transform(X)
    return X, imp, scaler


def transform_Xy(df, feats, imp, scaler):
    return scaler.transform(imp.transform(df[feats].values.astype(float)))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 120)
    print("  PLATOON MECHANISM TEST + GAP FEATURE SEARCH")
    print("=" * 120)

    print("\n  Loading feature matrix...")
    df = build_feature_matrix("MLB")
    print(f"  Feature matrix: {len(df)} games")

    # Join platoon features
    platoon = pd.read_csv(os.path.join(_HERE, "data", "platoon_features.csv"))
    platoon_cols = [
        "game_id",
        "h_sp_is_lhp", "a_sp_is_lhp",
        "h_ops_vs_sp_hand", "a_ops_vs_sp_hand", "platoon_ops_diff",
        "h_platoon_gap", "a_platoon_gap", "platoon_gap_diff",
    ]
    df = df.merge(platoon[platoon_cols], on="game_id", how="left")

    # Prepare
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df[TARGET_ATS].notna()].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Games with ATS labels: {len(df)}")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: MECHANISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def part1_mechanism_tests(df):
    print("\n" + "=" * 120)
    print("  PART 1: PLATOON GAP MECHANISM TESTS")
    print("=" * 120)

    train = df[df["game_date"] < TRAIN_CUTOFF].copy()
    late = df[(df["game_date"] >= LATE_START) & (df["game_date"] <= LATE_END)].copy()

    print(f"\n  Train: {len(train)} games")
    print(f"  Late test: {len(late)} games")

    feats_baseline = BASELINE_FEATURES
    feats_d = BASELINE_FEATURES + PLATOON_GAP

    y_train = train[TARGET_ATS].values
    y_late = late[TARGET_ATS].values

    # ── Train models across seeds, store per-game probas ──────────────────
    baseline_probas = []
    model_d_probas = []

    t0 = time.time()
    for seed in range(N_SEEDS):
        # Baseline
        X_tr_bl, imp_bl, sc_bl = prep_Xy(train, feats_baseline)
        X_late_bl = transform_Xy(late, feats_baseline, imp_bl, sc_bl)
        model_bl = train_calibrated_gbc(X_tr_bl, y_train, seed * 7 + 42)
        res_bl = evaluate_model(model_bl, X_late_bl, y_late)
        baseline_probas.append(res_bl["proba"])

        # Model D (+gap)
        X_tr_d, imp_d, sc_d = prep_Xy(train, feats_d)
        X_late_d = transform_Xy(late, feats_d, imp_d, sc_d)
        model_d = train_calibrated_gbc(X_tr_d, y_train, seed * 7 + 42)
        res_d = evaluate_model(model_d, X_late_d, y_late)
        model_d_probas.append(res_d["proba"])

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s ({N_SEEDS} seeds x 2 models)")

    # Average probas across seeds for stability
    bl_avg = np.mean(baseline_probas, axis=0)
    d_avg = np.mean(model_d_probas, axis=0)

    # ──────────────────────────────────────────────────────────────────────
    # TEST A: EFFECT CONCENTRATION BY |platoon_gap_diff| MAGNITUDE
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 120)
    print("  TEST A: EFFECT CONCENTRATION — Does platoon gap help MORE in extreme matchups?")
    print("-" * 120)

    pgd = late["platoon_gap_diff"].values
    pgd_abs = np.abs(pgd)
    valid = np.isfinite(pgd_abs)

    # Define bins
    bins = [
        ("Small |gap| < 0.010",   pgd_abs < 0.010),
        ("Medium 0.010-0.025",    (pgd_abs >= 0.010) & (pgd_abs < 0.025)),
        ("Large |gap| > 0.025",   pgd_abs >= 0.025),
    ]

    print(f"\n  {'Bin':>25s}  {'N':>5}  {'BL_AUC':>8}  {'D_AUC':>8}  {'dAUC':>8}  "
          f"{'BL_Brier':>10}  {'D_Brier':>10}  {'dBrier':>10}  {'BL_WR':>7}  {'D_WR':>7}")
    print("  " + "-" * 110)

    for bin_name, mask in bins:
        mask = mask & valid
        n = mask.sum()
        if n < 20:
            print(f"  {bin_name:>25s}  {n:>5d}  (too few games)")
            continue

        y_bin = y_late[mask]
        bl_bin = bl_avg[mask]
        d_bin = d_avg[mask]

        try:
            bl_auc = roc_auc_score(y_bin, bl_bin)
        except ValueError:
            bl_auc = np.nan
        try:
            d_auc = roc_auc_score(y_bin, d_bin)
        except ValueError:
            d_auc = np.nan

        bl_brier = brier_score_loss(y_bin, bl_bin)
        d_brier = brier_score_loss(y_bin, d_bin)

        bl_wr = y_bin[bl_bin >= 0.5].mean() if (bl_bin >= 0.5).sum() > 0 else np.nan
        d_wr = y_bin[d_bin >= 0.5].mean() if (d_bin >= 0.5).sum() > 0 else np.nan

        d_auc_v = d_auc - bl_auc if not (np.isnan(d_auc) or np.isnan(bl_auc)) else np.nan
        d_brier_v = d_brier - bl_brier

        print(f"  {bin_name:>25s}  {n:>5d}  {bl_auc:>8.4f}  {d_auc:>8.4f}  "
              f"{d_auc_v:>+8.4f}  {bl_brier:>10.6f}  {d_brier:>10.6f}  "
              f"{d_brier_v:>+10.6f}  {bl_wr * 100:>5.1f}%  {d_wr * 100:>5.1f}%")

    # Continuous correlation: does |improvement| correlate with |platoon_gap_diff|?
    improvement = np.abs(d_avg - 0.5) - np.abs(bl_avg - 0.5)  # positive = D more decisive
    mask_valid = valid
    if mask_valid.sum() > 20:
        rho, p_val = spearmanr(pgd_abs[mask_valid], improvement[mask_valid])
        print(f"\n  Continuous: rank correlation of |platoon_gap_diff| vs model-D decisiveness improvement")
        print(f"    rho={rho:+.4f}, p={p_val:.4f} {'SIGNIFICANT' if p_val < 0.05 else 'not significant'}")

    # ──────────────────────────────────────────────────────────────────────
    # TEST B: DIRECTIONAL CHECK — Does it help on BOTH sides?
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 120)
    print("  TEST B: DIRECTIONAL CHECK — Does platoon gap help on BOTH sides?")
    print("-" * 120)

    sides = [
        ("Platoon favors HOME (gap_diff > 0)", pgd > 0),
        ("Platoon favors AWAY (gap_diff < 0)", pgd < 0),
    ]

    print(f"\n  {'Side':>45s}  {'N':>5}  {'BL_AUC':>8}  {'D_AUC':>8}  {'dAUC':>8}  "
          f"{'BL_Brier':>10}  {'D_Brier':>10}  {'dBrier':>10}")
    print("  " + "-" * 105)

    for side_name, mask in sides:
        mask = mask & valid
        n = mask.sum()
        if n < 20:
            print(f"  {side_name:>45s}  {n:>5d}  (too few)")
            continue

        y_bin = y_late[mask]
        bl_bin = bl_avg[mask]
        d_bin = d_avg[mask]

        try:
            bl_auc = roc_auc_score(y_bin, bl_bin)
            d_auc = roc_auc_score(y_bin, d_bin)
            d_auc_v = d_auc - bl_auc
        except ValueError:
            bl_auc = d_auc = d_auc_v = np.nan

        bl_brier = brier_score_loss(y_bin, bl_bin)
        d_brier = brier_score_loss(y_bin, d_bin)

        print(f"  {side_name:>45s}  {n:>5d}  {bl_auc:>8.4f}  {d_auc:>8.4f}  "
              f"{d_auc_v:>+8.4f}  {bl_brier:>10.6f}  {d_brier:>10.6f}  "
              f"{d_brier - bl_brier:>+10.6f}")

    # Also check: does Model D correctly predict direction more often?
    for side_name, mask in sides:
        mask = mask & valid
        n = mask.sum()
        if n < 20:
            continue

        y_bin = y_late[mask]
        bl_correct = ((bl_avg[mask] >= 0.5) == y_bin).mean()
        d_correct = ((d_avg[mask] >= 0.5) == y_bin).mean()
        print(f"    Accuracy — {side_name}: BL={bl_correct*100:.1f}%  D={d_correct*100:.1f}%  "
              f"delta={((d_correct - bl_correct)*100):+.1f}pp")

    # ──────────────────────────────────────────────────────────────────────
    # TEST C: BY STARTER HANDEDNESS — Does it help more vs LHP?
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 120)
    print("  TEST C: BY STARTER HANDEDNESS — Does platoon gap help more vs LHP?")
    print("-" * 120)
    print("  (Baseball: platoon effects are historically LARGEST vs LHP starters)")

    h_lhp = late["h_sp_is_lhp"].values if "h_sp_is_lhp" in late.columns else np.full(len(late), np.nan)
    a_lhp = late["a_sp_is_lhp"].values if "a_sp_is_lhp" in late.columns else np.full(len(late), np.nan)

    # "At least one LHP starter" vs "both RHP"
    any_lhp = (h_lhp == 1) | (a_lhp == 1)
    both_rhp = (h_lhp == 0) & (a_lhp == 0)
    any_lhp_valid = any_lhp & np.isfinite(h_lhp) & np.isfinite(a_lhp)
    both_rhp_valid = both_rhp & np.isfinite(h_lhp) & np.isfinite(a_lhp)

    hand_bins = [
        ("At least 1 LHP starter", any_lhp_valid),
        ("Both RHP starters",      both_rhp_valid),
    ]

    print(f"\n  {'Handedness':>30s}  {'N':>5}  {'BL_AUC':>8}  {'D_AUC':>8}  {'dAUC':>8}  "
          f"{'BL_Brier':>10}  {'D_Brier':>10}  {'dBrier':>10}")
    print("  " + "-" * 100)

    for hand_name, mask in hand_bins:
        n = mask.sum()
        if n < 20:
            print(f"  {hand_name:>30s}  {n:>5d}  (too few)")
            continue

        y_bin = y_late[mask]
        bl_bin = bl_avg[mask]
        d_bin = d_avg[mask]

        try:
            bl_auc = roc_auc_score(y_bin, bl_bin)
            d_auc = roc_auc_score(y_bin, d_bin)
            d_auc_v = d_auc - bl_auc
        except ValueError:
            bl_auc = d_auc = d_auc_v = np.nan

        bl_brier = brier_score_loss(y_bin, bl_bin)
        d_brier = brier_score_loss(y_bin, d_bin)

        print(f"  {hand_name:>30s}  {n:>5d}  {bl_auc:>8.4f}  {d_auc:>8.4f}  "
              f"{d_auc_v:>+8.4f}  {bl_brier:>10.6f}  {d_brier:>10.6f}  "
              f"{d_brier - bl_brier:>+10.6f}")

    # More granular: games where BOTH starters are LHP (most extreme platoon case)
    both_lhp = (h_lhp == 1) & (a_lhp == 1) & np.isfinite(h_lhp) & np.isfinite(a_lhp)
    n_both_lhp = both_lhp.sum()
    print(f"\n  Both LHP starters: {n_both_lhp} games in late test")
    if n_both_lhp >= 10:
        y_bin = y_late[both_lhp]
        bl_brier = brier_score_loss(y_bin, bl_avg[both_lhp])
        d_brier = brier_score_loss(y_bin, d_avg[both_lhp])
        print(f"    BL Brier: {bl_brier:.6f}  D Brier: {d_brier:.6f}  delta: {d_brier - bl_brier:+.6f}")

    # ──────────────────────────────────────────────────────────────────────
    # MECHANISM VERDICT
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 120)
    print("  PART 1 MECHANISM VERDICT")
    print("-" * 120)

    print("""
  Mechanism prediction: platoon gap = how much a team's batting DEVIATES from
  overall vs a specific pitcher handedness. If the mechanism is real:
    A. It should help MORE in extreme |platoon_gap_diff| games
    B. It should help on BOTH sides (home-favored and away-favored platoon)
    C. It should help MORE when a starter is LHP (larger platoon effects)

  If it helps UNIFORMLY regardless of matchup magnitude → suspicious (noise fit)
  If it helps on ONE side only → asymmetric / overfitting
  If it helps MORE vs RHP → not the platoon mechanism, something else
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: GAP FEATURE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def build_gap_features(df):
    """
    Build candidate GAP features from existing data.
    Returns df with new columns and a dict of gap feature groups.
    """
    print("\n" + "=" * 120)
    print("  PART 2: GAP FEATURE SEARCH — Building GAP-from-aggregate features")
    print("=" * 120)

    conn = sqlite3.connect(DB_PATH)
    gap_groups = {}

    # ── GAP 1: Home/Road OPS gap ──────────────────────────────────────────
    # team's home OPS - overall OPS (from team_game_batting)
    # This measures how much a team benefits from playing at home.
    print("\n  --- GAP 1: Home/Road OPS Gap ---")
    try:
        tgb = pd.read_sql_query(
            "SELECT team_name, season, game_date, is_home, ops FROM team_game_batting "
            "ORDER BY team_name, game_date",
            conn,
        )
        tgb["game_date"] = pd.to_datetime(tgb["game_date"])

        # For each team-season, compute cumulative home OPS and overall OPS
        # using only PRIOR games (shift to avoid lookahead)
        home_gap_records = []
        for (team, season), grp in tgb.groupby(["team_name", "season"]):
            grp = grp.sort_values("game_date").copy()
            # Cumulative OPS: expanding mean of all prior games
            grp["cum_ops_all"] = grp["ops"].shift(1).expanding(min_periods=5).mean()
            # Cumulative OPS: home games only
            home_mask = grp["is_home"] == 1
            grp["ops_home_only"] = grp["ops"].where(home_mask, np.nan)
            grp["cum_ops_home"] = grp["ops_home_only"].shift(1).expanding(min_periods=3).mean()
            # Cumulative OPS: road games only
            grp["ops_road_only"] = grp["ops"].where(~home_mask, np.nan)
            grp["cum_ops_road"] = grp["ops_road_only"].shift(1).expanding(min_periods=3).mean()

            # Home-road gap = home OPS - overall OPS
            grp["home_ops_gap"] = grp["cum_ops_home"] - grp["cum_ops_all"]
            # Road gap similarly
            grp["road_ops_gap"] = grp["cum_ops_road"] - grp["cum_ops_all"]

            home_gap_records.append(grp[["team_name", "season", "game_date",
                                         "home_ops_gap", "road_ops_gap"]].copy())

        if home_gap_records:
            hg_df = pd.concat(home_gap_records, ignore_index=True)
            # For each game: home team gets their home_ops_gap, away team gets their road_ops_gap
            # Need to match via team name and game date (closest prior entry)
            # Merge directly on home_team/game_date
            hg_df["game_date_str"] = hg_df["game_date"].dt.strftime("%Y-%m-%d")
            df["game_date_str"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

            # Home team's home gap
            home_merge = hg_df[["team_name", "game_date_str", "home_ops_gap"]].drop_duplicates(
                subset=["team_name", "game_date_str"], keep="last"
            )
            df = df.merge(
                home_merge.rename(columns={"team_name": "home_team", "home_ops_gap": "h_home_ops_gap"}),
                on=["home_team", "game_date_str"], how="left"
            )

            # Away team's road gap
            road_merge = hg_df[["team_name", "game_date_str", "road_ops_gap"]].drop_duplicates(
                subset=["team_name", "game_date_str"], keep="last"
            )
            df = df.merge(
                road_merge.rename(columns={"team_name": "away_team", "road_ops_gap": "a_road_ops_gap"}),
                on=["away_team", "game_date_str"], how="left"
            )

            # Differential: home team's home advantage - away team's road disadvantage
            # Positive = home team benefits more from venue than away team suffers
            df["venue_gap_diff"] = df["h_home_ops_gap"].fillna(0) - df["a_road_ops_gap"].fillna(0)

            fill_h = df["h_home_ops_gap"].notna().mean()
            fill_a = df["a_road_ops_gap"].notna().mean()
            fill_d = df["venue_gap_diff"].notna().mean()
            print(f"    h_home_ops_gap fill: {fill_h*100:.1f}%")
            print(f"    a_road_ops_gap fill: {fill_a*100:.1f}%")
            print(f"    venue_gap_diff fill: {fill_d*100:.1f}%")
            print(f"    h_home_ops_gap mean: {df['h_home_ops_gap'].mean():.4f}, std: {df['h_home_ops_gap'].std():.4f}")
            print(f"    a_road_ops_gap mean: {df['a_road_ops_gap'].mean():.4f}, std: {df['a_road_ops_gap'].std():.4f}")

            gap_groups["venue_gap"] = ["h_home_ops_gap", "a_road_ops_gap", "venue_gap_diff"]
        else:
            print("    No team_game_batting data found")
    except Exception as e:
        print(f"    ERROR building home/road gap: {e}")

    # ── GAP 2: Recent Form Gap ────────────────────────────────────────────
    # 10-game rolling OPS - season OPS
    # If already have h_ops (season) and team_game_batting per game,
    # compute rolling 10-game OPS and subtract season OPS
    print("\n  --- GAP 2: Recent Form OPS Gap ---")
    try:
        # Compute rolling 10-game OPS per team using team_game_batting
        form_records = []
        for (team, season), grp in tgb.groupby(["team_name", "season"]):
            grp = grp.sort_values("game_date").copy()
            grp["rolling_ops_10"] = grp["ops"].shift(1).rolling(10, min_periods=5).mean()
            grp["season_ops"] = grp["ops"].shift(1).expanding(min_periods=5).mean()
            grp["form_ops_gap"] = grp["rolling_ops_10"] - grp["season_ops"]
            form_records.append(grp[["team_name", "season", "game_date", "form_ops_gap"]].copy())

        if form_records:
            form_df = pd.concat(form_records, ignore_index=True)
            form_df["game_date_str"] = form_df["game_date"].dt.strftime("%Y-%m-%d")

            # Home team's form gap
            h_form = form_df[["team_name", "game_date_str", "form_ops_gap"]].drop_duplicates(
                subset=["team_name", "game_date_str"], keep="last"
            )
            df = df.merge(
                h_form.rename(columns={"team_name": "home_team", "form_ops_gap": "h_form_ops_gap"}),
                on=["home_team", "game_date_str"], how="left"
            )

            # Away team's form gap
            a_form = form_df[["team_name", "game_date_str", "form_ops_gap"]].drop_duplicates(
                subset=["team_name", "game_date_str"], keep="last"
            )
            df = df.merge(
                a_form.rename(columns={"team_name": "away_team", "form_ops_gap": "a_form_ops_gap"}),
                on=["away_team", "game_date_str"], how="left"
            )

            df["form_gap_diff"] = df["h_form_ops_gap"].fillna(0) - df["a_form_ops_gap"].fillna(0)

            fill_h = df["h_form_ops_gap"].notna().mean()
            fill_a = df["a_form_ops_gap"].notna().mean()
            print(f"    h_form_ops_gap fill: {fill_h*100:.1f}%")
            print(f"    a_form_ops_gap fill: {fill_a*100:.1f}%")
            print(f"    h_form_ops_gap mean: {df['h_form_ops_gap'].mean():.4f}, std: {df['h_form_ops_gap'].std():.4f}")
            print(f"    a_form_ops_gap mean: {df['a_form_ops_gap'].mean():.4f}, std: {df['a_form_ops_gap'].std():.4f}")

            gap_groups["form_gap"] = ["h_form_ops_gap", "a_form_ops_gap", "form_gap_diff"]
        else:
            print("    No form gap data buildable")
    except Exception as e:
        print(f"    ERROR building form gap: {e}")

    # ── GAP 3: Bullpen ERA Gap ────────────────────────────────────────────
    # Team bullpen ERA vs team overall ERA
    # From team_pitching_games + pitcher_game_logs (SP vs relief)
    print("\n  --- GAP 3: Bullpen ERA Gap ---")
    try:
        # team_pitching_games has team-level totals per game
        # pitcher_game_logs has individual pitcher appearances
        # Approximate: team total ER / team total IP = team ERA per game
        # SP ER / SP IP = starter ERA; remainder = bullpen ERA
        # But we don't have clean SP vs bullpen split in team_pitching_games.
        #
        # Alternative: use pitcher_game_logs to separate SP starts from relief.
        # SP = first pitcher to appear (listed in games table as h_sp_name/a_sp_name via platoon CSV).
        #
        # Simpler approach: compute rolling team ERA from team_pitching_games
        # and compare to SP rolling ERA (already in features as h_sp_era_5).
        # bullpen_gap = team_era_rolling - sp_era_rolling
        # High value = bullpen is worse than starters

        tpg = pd.read_sql_query(
            "SELECT game_date, team_name, innings_pitched, earned_runs "
            "FROM team_pitching_games ORDER BY team_name, game_date",
            conn,
        )
        tpg["game_date"] = pd.to_datetime(tpg["game_date"])

        bp_records = []
        for team, grp in tpg.groupby("team_name"):
            grp = grp.sort_values("game_date").copy()
            # Rolling 10-game team ERA = (sum ER / sum IP) * 9
            er_roll = grp["earned_runs"].shift(1).rolling(10, min_periods=5).sum()
            ip_roll = grp["innings_pitched"].shift(1).rolling(10, min_periods=5).sum()
            grp["team_era_10g"] = (er_roll / ip_roll * 9).where(ip_roll > 0)

            # Season cumulative team ERA
            er_cum = grp["earned_runs"].shift(1).expanding(min_periods=5).sum()
            ip_cum = grp["innings_pitched"].shift(1).expanding(min_periods=5).sum()
            grp["team_era_season"] = (er_cum / ip_cum * 9).where(ip_cum > 0)

            bp_records.append(grp[["team_name", "game_date", "team_era_10g", "team_era_season"]].copy())

        if bp_records:
            bp_df = pd.concat(bp_records, ignore_index=True)
            bp_df["game_date_str"] = bp_df["game_date"].dt.strftime("%Y-%m-%d")

            # Home team
            h_bp = bp_df[["team_name", "game_date_str", "team_era_10g", "team_era_season"]].drop_duplicates(
                subset=["team_name", "game_date_str"], keep="last"
            )
            df = df.merge(
                h_bp.rename(columns={
                    "team_name": "home_team",
                    "team_era_10g": "h_team_era_10g",
                    "team_era_season": "h_team_era_season",
                }),
                on=["home_team", "game_date_str"], how="left"
            )

            # Away team
            a_bp = bp_df[["team_name", "game_date_str", "team_era_10g", "team_era_season"]].drop_duplicates(
                subset=["team_name", "game_date_str"], keep="last"
            )
            df = df.merge(
                a_bp.rename(columns={
                    "team_name": "away_team",
                    "team_era_10g": "a_team_era_10g",
                    "team_era_season": "a_team_era_season",
                }),
                on=["away_team", "game_date_str"], how="left"
            )

            # Bullpen gap = team rolling ERA - SP rolling ERA
            # If team ERA >> SP ERA → bullpen is bad
            # h_sp_era_5 already in features (home SP rolling 5-game ERA)
            df["h_bp_era_gap"] = df["h_team_era_10g"] - df.get("h_sp_era_5", pd.Series(np.nan, index=df.index))
            df["a_bp_era_gap"] = df["a_team_era_10g"] - df.get("a_sp_era_5", pd.Series(np.nan, index=df.index))
            df["bp_era_gap_diff"] = df["a_bp_era_gap"].fillna(0) - df["h_bp_era_gap"].fillna(0)

            fill_h = df["h_bp_era_gap"].notna().mean()
            fill_a = df["a_bp_era_gap"].notna().mean()
            print(f"    h_bp_era_gap fill: {fill_h*100:.1f}%")
            print(f"    a_bp_era_gap fill: {fill_a*100:.1f}%")
            print(f"    h_bp_era_gap mean: {df['h_bp_era_gap'].mean():.3f}, std: {df['h_bp_era_gap'].std():.3f}")
            print(f"    a_bp_era_gap mean: {df['a_bp_era_gap'].mean():.3f}, std: {df['a_bp_era_gap'].std():.3f}")
            print(f"    (positive = bullpen worse than starters)")

            gap_groups["bp_gap"] = ["h_bp_era_gap", "a_bp_era_gap", "bp_era_gap_diff"]
        else:
            print("    No team_pitching_games data found")
    except Exception as e:
        print(f"    ERROR building bullpen gap: {e}")
        import traceback
        traceback.print_exc()

    conn.close()

    # Clean up temp column
    if "game_date_str" in df.columns:
        df = df.drop(columns=["game_date_str"])

    return df, gap_groups


def check_orthogonality(df, gap_groups):
    """Check correlation of each gap feature with existing V1_PRUNED features."""
    print("\n" + "-" * 120)
    print("  ORTHOGONALITY CHECK: Correlation with existing V1_PRUNED features")
    print("-" * 120)
    print(f"  Threshold: |r| > 0.30 = NOT orthogonal, skip")

    orthogonal_groups = {}

    for group_name, gap_feats in gap_groups.items():
        print(f"\n  --- {group_name} ---")
        all_orthogonal = True

        for gap_feat in gap_feats:
            if gap_feat not in df.columns:
                print(f"    {gap_feat}: NOT IN DATA, skip")
                all_orthogonal = False
                continue

            max_r = 0.0
            max_feat = ""
            gap_vals = df[gap_feat].values.astype(float)
            valid_gap = np.isfinite(gap_vals)

            for base_feat in BASELINE_FEATURES:
                if base_feat not in df.columns:
                    continue
                base_vals = df[base_feat].values.astype(float)
                valid = valid_gap & np.isfinite(base_vals)
                if valid.sum() < 50:
                    continue
                r = np.corrcoef(gap_vals[valid], base_vals[valid])[0, 1]
                if abs(r) > abs(max_r):
                    max_r = r
                    max_feat = base_feat

            status = "ORTHOGONAL" if abs(max_r) < 0.30 else "NOT ORTHOGONAL"
            if abs(max_r) >= 0.30:
                all_orthogonal = False
            print(f"    {gap_feat:25s}: max |r| = {abs(max_r):.3f} with {max_feat:25s}  [{status}]")

        if all_orthogonal:
            orthogonal_groups[group_name] = gap_feats
            print(f"    >> {group_name} PASSES orthogonality check")
        else:
            # Still include if at least the diff feature is orthogonal
            diff_feat = [f for f in gap_feats if "diff" in f]
            if diff_feat:
                df_feat = diff_feat[0]
                if df_feat in df.columns:
                    gap_vals = df[df_feat].values.astype(float)
                    valid_gap = np.isfinite(gap_vals)
                    max_r_diff = 0.0
                    for base_feat in BASELINE_FEATURES:
                        if base_feat not in df.columns:
                            continue
                        base_vals = df[base_feat].values.astype(float)
                        valid = valid_gap & np.isfinite(base_vals)
                        if valid.sum() < 50:
                            continue
                        r = np.corrcoef(gap_vals[valid], base_vals[valid])[0, 1]
                        if abs(r) > abs(max_r_diff):
                            max_r_diff = r
                    if abs(max_r_diff) < 0.30:
                        orthogonal_groups[group_name + "_diff_only"] = diff_feat
                        print(f"    >> diff feature alone passes: including {diff_feat}")
                    else:
                        print(f"    >> ENTIRE GROUP FAILS orthogonality — SKIP from model test")
            else:
                print(f"    >> ENTIRE GROUP FAILS orthogonality — SKIP from model test")

    return orthogonal_groups


def test_gap_features(df, gap_groups, orthogonal_groups):
    """Test each orthogonal gap group's additive value on top of platoon gap."""
    print("\n" + "=" * 120)
    print("  GAP FEATURE MODEL TEST — 10-seed evaluation")
    print("=" * 120)

    train = df[df["game_date"] < TRAIN_CUTOFF].copy()
    late = df[(df["game_date"] >= LATE_START) & (df["game_date"] <= LATE_END)].copy()

    y_train = train[TARGET_ATS].values
    y_late = late[TARGET_ATS].values

    # Model D baseline (with platoon gap)
    feats_d = BASELINE_FEATURES + PLATOON_GAP

    # Collect results
    results = {}

    # Models to test
    model_specs = {"D: Base+PlatoonGap": feats_d}

    for group_name, gap_feats in orthogonal_groups.items():
        model_name = f"+{group_name}"
        model_specs[model_name] = feats_d + gap_feats

    # Also test each gap ON ITS OWN (no platoon gap) against baseline
    for group_name, gap_feats in gap_groups.items():
        # Only if features exist in df
        available = [f for f in gap_feats if f in df.columns]
        if available:
            model_specs[f"BL+{group_name}"] = BASELINE_FEATURES + available

    t0 = time.time()
    for model_name, feats in model_specs.items():
        # Verify all features exist
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"  SKIP {model_name}: missing {missing}")
            continue

        results[model_name] = {"late_auc": [], "late_brier": [], "late_acc": [], "n_feat": len(feats)}

        for seed in range(N_SEEDS):
            X_tr, imp, sc = prep_Xy(train, feats)
            X_late = transform_Xy(late, feats, imp, sc)
            model = train_calibrated_gbc(X_tr, y_train, seed * 7 + 42)
            res = evaluate_model(model, X_late, y_late)
            results[model_name]["late_auc"].append(res["auc"])
            results[model_name]["late_brier"].append(res["brier"])
            results[model_name]["late_acc"].append(res["accuracy"])

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")

    # Print results
    print(f"\n  {'Model':>30s}  {'Nf':>3}  {'Late AUC':>10}  {'dAUC':>8}  "
          f"{'Late Brier':>12}  {'dBrier':>10}  {'Late Acc':>10}")
    print("  " + "-" * 100)

    # Use Model D as the reference for deltas
    d_auc = np.nanmean(results.get("D: Base+PlatoonGap", {}).get("late_auc", [np.nan]))
    d_brier = np.nanmean(results.get("D: Base+PlatoonGap", {}).get("late_brier", [np.nan]))

    for model_name, r in results.items():
        la = np.nanmean(r["late_auc"])
        lb = np.nanmean(r["late_brier"])
        la_sd = np.nanstd(r["late_auc"])
        lb_sd = np.nanstd(r["late_brier"])
        lacc = np.nanmean(r["late_acc"])

        d_la = la - d_auc
        d_lb = lb - d_brier

        print(f"  {model_name:>30s}  {r['n_feat']:>3d}  "
              f"{la:.4f}+{la_sd:.3f}  {d_la:>+.4f}  "
              f"{lb:.6f}+{lb_sd:.4f}  {d_lb:>+.6f}  "
              f"{lacc*100:.1f}%")

    # Paired t-tests vs Model D
    print(f"\n  Paired t-test (late Brier) vs Model D:")
    if "D: Base+PlatoonGap" in results:
        d_briers = results["D: Base+PlatoonGap"]["late_brier"]
        for model_name, r in results.items():
            if model_name == "D: Base+PlatoonGap":
                continue
            if len(r["late_brier"]) == len(d_briers):
                t_stat, p_val = ttest_rel(d_briers, r["late_brier"])
                sig = "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                improved = "IMPROVED" if np.nanmean(r["late_brier"]) < np.nanmean(d_briers) else "worse"
                print(f"    {model_name:>30s}: t={t_stat:>+.3f}  p={p_val:.4f}  {improved} {sig}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    df = load_data()

    # Part 1: Mechanism tests
    part1_mechanism_tests(df)

    # Part 2: Build gap features
    df, gap_groups = build_gap_features(df)

    # Orthogonality check
    orthogonal_groups = check_orthogonality(df, gap_groups)

    # Test gap features
    if orthogonal_groups or gap_groups:
        results = test_gap_features(df, gap_groups, orthogonal_groups)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  FINAL VERDICT")
    print("=" * 120)
    print("""
  Part 1 (Mechanism):
    Check output above for A/B/C results.
    Strong mechanism = concentrated + bidirectional + LHP-dominant
    Weak mechanism  = uniform + one-sided + no handedness pattern

  Part 2 (Gap Search):
    Any gap feature that:
      1. Passes orthogonality (r < 0.30 with all V1_PRUNED)
      2. Improves late Brier beyond noise floor (~0.001)
      3. Passes paired t-test (p < 0.05)
    is a candidate for deployment.

    Any gap feature that fails ALL three → close that axis.
""")


if __name__ == "__main__":
    main()
