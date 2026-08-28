"""
fip_vs_era_test.py — Test whether FIP features improve NegBin mu prediction

FIP = (13×HR + 3×BB - 2×K) / IP + constant
ERA absorbs defense, sequencing luck, inherited runners.
FIP strips those out → better predictor of FUTURE pitcher performance.

Test design:
  - Compute rolling FIP (3-start, 5-start) from pitcher_game_logs (2022-2025)
  - Train NegBin mu models: ERA-only, FIP-only, ERA+FIP, FIP-swap (replace ERA with FIP)
  - Temporal split within 2022-2025 (2026 excluded — no HR data)
  - OOS metrics: MAE, NegBin LL, and spread-cover Brier
  - Multi-split stability check (5 temporal folds)
"""
from __future__ import annotations
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
DATA_DIR = os.path.join(_HERE, "..", "data")
SUBMODEL_DIR = os.path.join(DATA_DIR, "submodels")
HIST_DB = os.path.join(DATA_DIR, "historical.db")

R_GLOBAL = 3.26
FIP_CONSTANT = 3.10  # MLB league-average constant (cancels in differentials)
N_SIMS = 10_000


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COMPUTE ROLLING FIP FROM GAME LOGS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rolling_fip(conn, n):
    """
    Compute rolling FIP over last n starts (shift(1) → zero lookahead).
    FIP = (13×HR + 3×BB - 2×K) / IP + 3.10

    Returns DataFrame: player_id, game_date, rolling_fip_{n}
    """
    import sqlite3

    df = pd.read_sql_query(
        "SELECT player_id, game_date, ip, walks, strikeouts, home_runs "
        "FROM pitcher_game_logs ORDER BY player_id, game_date",
        conn,
    )

    col = f"rolling_fip_{n}"

    # Filter to rows with HR data (2022-2025)
    df = df[df["home_runs"].notna()].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["player_id", "game_date"]).copy()

    # Per-game FIP components
    df["fip_numerator"] = 13 * df["home_runs"] + 3 * df["walks"] - 2 * df["strikeouts"]

    # shift(1) within each pitcher → zero lookahead
    grp = df.groupby("player_id")
    ip_shifted = grp["ip"].shift(1)
    fn_shifted = grp["fip_numerator"].shift(1)

    ip_roll = ip_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
    fn_roll = fn_shifted.groupby(df["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)

    df[col] = (fn_roll / ip_roll + FIP_CONSTANT).where(ip_roll > 0)

    return df[["player_id", "game_date", col]].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD FEATURE MATRIX WITH FIP
# ═══════════════════════════════════════════════════════════════════════════════

def build_matrix_with_fip():
    """Load feature matrix, join rolling FIP, return enriched DataFrame."""
    import sqlite3
    from features import build_feature_matrix

    print("=" * 110)
    print("  FIP vs ERA: OOS DISCRIMINATION TEST")
    print("=" * 110)

    conn = sqlite3.connect(HIST_DB)

    # Compute rolling FIP
    print("\n  Computing rolling FIP from pitcher_game_logs...")
    fip3 = compute_rolling_fip(conn, 3)
    fip5 = compute_rolling_fip(conn, 5)
    print(f"  Rolling FIP-3: {fip3['rolling_fip_3'].notna().sum()} values")
    print(f"  Rolling FIP-5: {fip5['rolling_fip_5'].notna().sum()} values")

    # Also compute rolling ERA from same source for consistency
    # (ensures we're comparing on exact same game set)
    df_logs = pd.read_sql_query(
        "SELECT player_id, pitcher_name, game_date, ip, er, walks, strikeouts, home_runs "
        "FROM pitcher_game_logs ORDER BY player_id, game_date",
        conn,
    )
    df_logs = df_logs[df_logs["home_runs"].notna()].copy()
    df_logs["game_date"] = pd.to_datetime(df_logs["game_date"], errors="coerce")
    df_logs = df_logs.sort_values(["player_id", "game_date"])

    # Per-game ERA from game logs
    grp = df_logs.groupby("player_id")
    for n_starts in [3, 5]:
        ip_s = grp["ip"].shift(1)
        er_s = grp["er"].shift(1)
        ip_r = ip_s.groupby(df_logs["player_id"]).rolling(n_starts, min_periods=2).sum().droplevel(0)
        er_r = er_s.groupby(df_logs["player_id"]).rolling(n_starts, min_periods=2).sum().droplevel(0)
        df_logs[f"rolling_era_gl_{n_starts}"] = (er_r / ip_r * 9).where(ip_r > 0)
        # K/9 for the same window
        k_s = grp["strikeouts"].shift(1)
        k_r = k_s.groupby(df_logs["player_id"]).rolling(n_starts, min_periods=2).sum().droplevel(0)
        df_logs[f"rolling_k9_gl_{n_starts}"] = (k_r / ip_r * 9).where(ip_r > 0)

    conn.close()

    # Load feature matrix
    print("  Loading feature matrix...")
    df = build_feature_matrix("MLB")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Feature matrix: {len(df)} games")

    # Identify starting pitchers for each game
    # The feature matrix has h_sp_era, a_sp_era etc but no pitcher identity column
    # We need to join via the pitcher_game_logs → game linking
    # Unfortunately, the feature matrix doesn't have pitcher player_id
    # Let's check if there's a way to link

    # The pitcher_stats are joined by pitcher_name in the feature builder
    # Let me check how the feature matrix links to pitchers

    # Actually, the simplest approach: build FIP features directly using the
    # same pipeline as features.py, then merge on (game_date, home_team, away_team)
    # But we need pitcher identity. Let me take a different approach:
    # use the game-level FIP stats that are already in the feature matrix pattern.

    # Alternative approach: compute per-game FIP per team (starting pitcher)
    # from pitcher_game_logs, and join to the feature matrix via the game_id
    # or (game_date, team) key.

    # Let's check what's in pitcher_game_logs vs games table
    conn2 = sqlite3.connect(HIST_DB)

    # pitcher_game_logs has game_pk which should match games.game_id
    # But games.game_id format is 'MLB_663178' while game_pk is an integer

    # Actually let me just check the linking
    c = conn2.cursor()
    c.execute("SELECT game_pk FROM pitcher_game_logs LIMIT 5")
    print(f"\n  Sample game_pk values: {[r[0] for r in c.fetchall()]}")
    c.execute("SELECT game_id FROM games WHERE sport='MLB' LIMIT 5")
    print(f"  Sample game_id values: {[r[0] for r in c.fetchall()]}")

    # They probably link via game_pk = int(game_id.split('_')[1])
    # Let's verify
    c.execute("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team,
               p.pitcher_name, p.ip, p.game_pk
        FROM games g
        JOIN pitcher_game_logs p ON p.game_pk = CAST(REPLACE(g.game_id, 'MLB_', '') AS INTEGER)
        WHERE g.sport='MLB'
        LIMIT 5
    """)
    rows = c.fetchall()
    print(f"\n  Game-pitcher join test:")
    for r in rows:
        print(f"    {r}")

    # Now: for each game, identify the starting pitcher (highest IP per team per game)
    # We need to know which team each pitcher plays for — pitcher_game_logs doesn't have team
    # But games table has home_team and away_team, and we can figure it out from
    # the context (multiple pitchers per game, but the starter has most IP)

    # Actually, let me check team_pitching_games
    c.execute("SELECT sql FROM sqlite_master WHERE name='team_pitching_games'")
    schema = c.fetchone()
    if schema:
        print(f"\n  team_pitching_games schema: {schema[0][:200]}")
    else:
        print("\n  team_pitching_games table not found")

    conn2.close()

    return df, df_logs, fip3, fip5


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DIRECT APPROACH: Per-game pitcher FIP as features
# ═══════════════════════════════════════════════════════════════════════════════

def build_fip_features_direct():
    """
    Alternative: compute FIP directly from each pitcher's game log entries,
    then aggregate per-game as starter FIP features.

    Since we can't easily link pitcher to team per game, use a simpler approach:
    compute pitcher-level rolling FIP, then match to games by checking which
    pitcher appears in a game's pitcher_game_logs with the most IP (= starter).
    """
    import sqlite3

    conn = sqlite3.connect(HIST_DB)

    print("\n  Building per-game starter FIP features...")

    # Get all pitcher game logs with HR data
    logs = pd.read_sql_query("""
        SELECT p.player_id, p.pitcher_name, p.game_date, p.game_pk,
               p.ip, p.er, p.walks, p.strikeouts, p.home_runs
        FROM pitcher_game_logs p
        WHERE p.home_runs IS NOT NULL
        ORDER BY p.player_id, p.game_date
    """, conn)
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")

    # Get games
    games = pd.read_sql_query("""
        SELECT game_id, game_date, home_team, away_team, home_score, away_score
        FROM games WHERE sport='MLB' AND status='Final'
    """, conn)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["game_pk"] = games["game_id"].str.replace("MLB_", "").astype(int)

    conn.close()

    # For each game, find the starter per side (pitcher with max IP)
    # First, join logs to games to know which team the pitcher was on
    # We can figure this out: for a given game_pk, there are ~2 starters
    # (one per team). The two pitchers with highest IP are the starters.

    merged = logs.merge(games[["game_pk", "home_team", "away_team"]],
                        on="game_pk", how="inner")

    # For each game, take top-2 IP pitchers as starters
    merged_sorted = merged.sort_values(["game_pk", "ip"], ascending=[True, False])
    starters = merged_sorted.groupby("game_pk").head(2).copy()

    # Assign side: first (highest IP) could be either home or away
    # We need to determine side. Unfortunately game_logs don't have team.
    # Use heuristic: if there are exactly 2 starters per game, one is home, one is away.
    # We'll compute FIP for each pitcher and assign to the game, then for the
    # regression test, just use the raw per-game FIP values as features.

    # SIMPLER: compute rolling FIP per pitcher, then for each game,
    # the feature is the starter's rolling FIP. Pair games to starters
    # via game_pk + max IP.

    # Compute rolling FIP per pitcher
    logs_sorted = logs.sort_values(["player_id", "game_date"]).copy()
    logs_sorted["fip_num"] = 13 * logs_sorted["home_runs"] + 3 * logs_sorted["walks"] - 2 * logs_sorted["strikeouts"]

    grp = logs_sorted.groupby("player_id")
    for n in [3, 5]:
        ip_s = grp["ip"].shift(1)
        fn_s = grp["fip_num"].shift(1)
        er_s = grp["er"].shift(1)

        ip_r = ip_s.groupby(logs_sorted["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        fn_r = fn_s.groupby(logs_sorted["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        er_r = er_s.groupby(logs_sorted["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)

        logs_sorted[f"rfip_{n}"] = (fn_r / ip_r + FIP_CONSTANT).where(ip_r > 0)
        logs_sorted[f"rera_{n}"] = (er_r / ip_r * 9).where(ip_r > 0)

    # Now: for each game, find the starters (top 2 IP) and get their rolling stats
    starter_stats = logs_sorted.merge(
        games[["game_pk", "home_team", "away_team"]],
        on="game_pk", how="inner"
    )

    # Sort by game_pk and IP to identify starters (top 2)
    starter_stats = starter_stats.sort_values(["game_pk", "ip"], ascending=[True, False])

    # For each game, the top-IP pitcher on each side
    game_starter_pairs = []
    for gpk, grp in starter_stats.groupby("game_pk"):
        top2 = grp.head(2)
        if len(top2) < 2:
            continue
        # Can't determine home/away from pitcher_game_logs alone
        # Use a ranking: first = higher IP, second = lower IP
        # We'll just average them or use both as unordered features
        p1 = top2.iloc[0]
        p2 = top2.iloc[1]

        game_starter_pairs.append({
            "game_pk": gpk,
            "game_date": p1["game_date"],
            # Use the two starters' stats as features (unordered for now)
            "starter1_rfip_3": p1.get("rfip_3"),
            "starter2_rfip_3": p2.get("rfip_3"),
            "starter1_rfip_5": p1.get("rfip_5"),
            "starter2_rfip_5": p2.get("rfip_5"),
            "starter1_rera_3": p1.get("rera_3"),
            "starter2_rera_3": p2.get("rera_3"),
            "starter1_rera_5": p1.get("rera_5"),
            "starter2_rera_5": p2.get("rera_5"),
            "starter1_ip": p1["ip"],
            "starter2_ip": p2["ip"],
            "starter1_name": p1["pitcher_name"],
            "starter2_name": p2["pitcher_name"],
        })

    sp_df = pd.DataFrame(game_starter_pairs)
    print(f"  Starter pairs identified: {len(sp_df)} games")
    print(f"  FIP-3 coverage: {sp_df['starter1_rfip_3'].notna().sum()} / {len(sp_df)} "
          f"({sp_df['starter1_rfip_3'].notna().mean()*100:.1f}%)")
    print(f"  FIP-5 coverage: {sp_df['starter1_rfip_5'].notna().sum()} / {len(sp_df)} "
          f"({sp_df['starter1_rfip_5'].notna().mean()*100:.1f}%)")

    # Compute differential features (higher FIP = worse pitcher)
    # Since we don't know home/away, use: max-min (spread) and mean
    sp_df["fip3_spread"] = (sp_df["starter1_rfip_3"] - sp_df["starter2_rfip_3"]).abs()
    sp_df["fip5_spread"] = (sp_df["starter1_rfip_5"] - sp_df["starter2_rfip_5"]).abs()
    sp_df["era3_spread"] = (sp_df["starter1_rera_3"] - sp_df["starter2_rera_3"]).abs()
    sp_df["era5_spread"] = (sp_df["starter1_rera_5"] - sp_df["starter2_rera_5"]).abs()

    # Better: use directional diff (starter1 - starter2, where starter1 = more IP)
    sp_df["fip3_diff"] = sp_df["starter1_rfip_3"] - sp_df["starter2_rfip_3"]
    sp_df["fip5_diff"] = sp_df["starter1_rfip_5"] - sp_df["starter2_rfip_5"]
    sp_df["era3_diff_gl"] = sp_df["starter1_rera_3"] - sp_df["starter2_rera_3"]
    sp_df["era5_diff_gl"] = sp_df["starter1_rera_5"] - sp_df["starter2_rera_5"]

    return sp_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DIRECT CORRELATION TEST (fast, no model training needed)
# ═══════════════════════════════════════════════════════════════════════════════

def correlation_test():
    """
    Test: does rolling FIP predict actual runs better than rolling ERA?
    Correlation of rolling stats with next-game actual runs.
    No model training — just raw correlation.
    """
    import sqlite3

    conn = sqlite3.connect(HIST_DB)

    print("\n" + "=" * 110)
    print("  DIRECT CORRELATION: Rolling FIP vs ERA → Next-Game Runs Allowed")
    print("=" * 110)

    # Get all pitcher game logs with HR
    logs = pd.read_sql_query("""
        SELECT player_id, pitcher_name, game_date, game_pk, ip, er,
               walks, strikeouts, home_runs
        FROM pitcher_game_logs
        WHERE home_runs IS NOT NULL
        ORDER BY player_id, game_date
    """, conn)
    conn.close()

    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    logs = logs.sort_values(["player_id", "game_date"]).copy()

    # Per-game stats
    logs["fip_num"] = 13 * logs["home_runs"] + 3 * logs["walks"] - 2 * logs["strikeouts"]
    logs["era_game"] = np.where(logs["ip"] > 0, logs["er"] / logs["ip"] * 9, np.nan)

    # Compute rolling stats (shifted — zero lookahead)
    grp = logs.groupby("player_id")
    for n in [3, 5, 10]:
        ip_s = grp["ip"].shift(1)
        fn_s = grp["fip_num"].shift(1)
        er_s = grp["er"].shift(1)
        k_s = grp["strikeouts"].shift(1)

        ip_r = ip_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        fn_r = fn_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        er_r = er_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        k_r = k_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)

        logs[f"rfip_{n}"] = (fn_r / ip_r + FIP_CONSTANT).where(ip_r > 0)
        logs[f"rera_{n}"] = (er_r / ip_r * 9).where(ip_r > 0)
        logs[f"rk9_{n}"] = (k_r / ip_r * 9).where(ip_r > 0)

    # Target: actual ERA in THIS game (next game from prediction perspective)
    logs["target_era"] = logs["era_game"]
    logs["target_er"] = logs["er"]

    # Filter to starters (IP >= 3) and valid rolling stats
    starters = logs[logs["ip"] >= 3].copy()
    print(f"\n  Starter appearances (IP≥3): {len(starters)}")

    # Temporal split: use last 20% as OOS
    starters = starters.sort_values("game_date").reset_index(drop=True)
    split = int(len(starters) * 0.80)
    oos = starters.iloc[split:].copy()
    print(f"  OOS starters: {len(oos)} ({str(oos['game_date'].iloc[0])[:10]} → "
          f"{str(oos['game_date'].iloc[-1])[:10]})")

    print(f"\n  Correlation with NEXT-GAME ER (OOS only):")
    print(f"  {'Feature':<20} {'ρ (Pearson)':>14} {'p-value':>12} {'N':>6}")
    print(f"  {'─'*20} {'─'*14} {'─'*12} {'─'*6}")

    features = ["rera_3", "rfip_3", "rera_5", "rfip_5", "rera_10", "rfip_10"]
    results = {}
    for feat in features:
        mask = oos[feat].notna() & oos["target_er"].notna()
        x = oos.loc[mask, feat].values
        y = oos.loc[mask, "target_er"].values
        if len(x) < 30:
            continue
        r, p = sp_stats.pearsonr(x, y)
        results[feat] = (r, p, len(x))
        print(f"  {feat:<20} {r:>+14.4f} {p:>12.4e} {len(x):>6}")

    # Compare FIP vs ERA at each window
    print(f"\n  FIP vs ERA lift (ρ difference):")
    for n in [3, 5, 10]:
        era_r = results.get(f"rera_{n}", (0,))[0]
        fip_r = results.get(f"rfip_{n}", (0,))[0]
        lift = fip_r - era_r
        print(f"    {n}-start window: ERA ρ={era_r:+.4f}, FIP ρ={fip_r:+.4f}, "
              f"Δρ={lift:+.4f} ({'FIP better' if lift > 0 else 'ERA better'})")

    # Also check: ERA-FIP gap (luck component) correlation with next-game ER
    oos["era_fip_gap_3"] = oos["rera_3"] - oos["rfip_3"]
    oos["era_fip_gap_5"] = oos["rera_5"] - oos["rfip_5"]

    print(f"\n  ERA-FIP gap (luck signal) → next-game ER:")
    for feat in ["era_fip_gap_3", "era_fip_gap_5"]:
        mask = oos[feat].notna() & oos["target_er"].notna()
        x = oos.loc[mask, feat].values
        y = oos.loc[mask, "target_er"].values
        if len(x) < 30:
            continue
        r, p = sp_stats.pearsonr(x, y)
        direction = "lucky ERA → MORE ER next" if r > 0 else "lucky ERA → LESS ER next (regression)"
        print(f"    {feat:<20} ρ={r:>+.4f}  p={p:.4e}  ({direction})")

    return logs, starters, oos


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MU MODEL TEST: ERA vs FIP features
# ═══════════════════════════════════════════════════════════════════════════════

def mu_model_comparison(logs_df):
    """
    Train NegBin-style mu models with different pitcher features.
    Compare OOS discrimination.
    """
    import sqlite3
    from features import build_feature_matrix

    print("\n" + "=" * 110)
    print("  MU MODEL COMPARISON: ERA vs FIP features")
    print("=" * 110)

    # Load feature matrix
    df = build_feature_matrix("MLB")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # Restrict to games with HR data (through 2025)
    df = df[df["game_date_dt"] < "2026-01-01"].copy()
    print(f"  Games through 2025: {len(df)}")

    # Load NegBin model to get feature list
    with open(os.path.join(SUBMODEL_DIR, "mlb_negbin_model.pkl"), "rb") as f:
        art = pickle.load(f)
    base_features = [c for c in art["feature_cols"] if c in df.columns]

    # Identify ERA features and their FIP equivalents
    era_features = [
        "h_sp_era", "a_sp_era", "sp_era_diff",
        "h_sp_era_3", "a_sp_era_3", "sp_era3_diff",
        "h_sp_era_5", "a_sp_era_5", "sp_era5_diff",
    ]
    era_in_model = [f for f in era_features if f in base_features]

    # We need to create FIP equivalents from the pitcher game logs
    # Join rolling FIP to the feature matrix via game linkage

    # Build starter FIP features per game
    conn = sqlite3.connect(HIST_DB)

    # Get rolling FIP per pitcher
    logs = pd.read_sql_query("""
        SELECT player_id, pitcher_name, game_date, game_pk,
               ip, er, walks, strikeouts, home_runs
        FROM pitcher_game_logs
        WHERE home_runs IS NOT NULL
        ORDER BY player_id, game_date
    """, conn)
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    logs = logs.sort_values(["player_id", "game_date"]).copy()
    logs["fip_num"] = 13 * logs["home_runs"] + 3 * logs["walks"] - 2 * logs["strikeouts"]

    grp = logs.groupby("player_id")
    for n in [3, 5]:
        ip_s = grp["ip"].shift(1)
        fn_s = grp["fip_num"].shift(1)
        ip_r = ip_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        fn_r = fn_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        logs[f"rfip_{n}"] = (fn_r / ip_r + FIP_CONSTANT).where(ip_r > 0)

    # Season-level FIP per pitcher (use cumulative up to game date)
    logs["cum_ip"] = grp["ip"].cumshift_and_sum = None  # need custom approach
    # Simpler: use expanding window (shift(1)) for season-level FIP
    # Group by player_id and season
    logs["season"] = logs["game_date"].dt.year
    season_grp = logs.groupby(["player_id", "season"])
    ip_s_seas = season_grp["ip"].shift(1)
    fn_s_seas = season_grp["fip_num"].shift(1)
    ip_exp = ip_s_seas.groupby([logs["player_id"], logs["season"]]).expanding(min_periods=2).sum().droplevel([0,1])
    fn_exp = fn_s_seas.groupby([logs["player_id"], logs["season"]]).expanding(min_periods=2).sum().droplevel([0,1])
    logs["season_fip"] = (fn_exp / ip_exp + FIP_CONSTANT).where(ip_exp > 0)

    # Now link starters to games
    games = pd.read_sql_query("""
        SELECT game_id, game_date, home_team, away_team
        FROM games WHERE sport='MLB' AND status='Final'
    """, conn)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["game_pk"] = games["game_id"].str.replace("MLB_", "").astype(int)
    conn.close()

    # Merge logs with games to identify which pitcher is home/away
    # Strategy: for each game, the two pitchers with highest IP are starters.
    # The one batting for the home team is the home starter.
    # Since we don't have team directly, use game context:
    # In MLB, the home starter typically pitches more IP in wins.
    # But simpler: look at which starters' names appear in team context.

    # Actually, the cleanest approach: for each game_pk, get the two pitchers
    # with highest IP. Then determine home/away by checking who pitched
    # the first/top-half innings (we don't have this).

    # Pragmatic approach: match by pitcher_name to the feature matrix's
    # sp_era source. The feature matrix already resolved which pitcher
    # is home vs away. We can look at the correlation pattern.

    # SIMPLEST VIABLE: just compute game-level FIP differential
    # (better pitcher FIP - worse pitcher FIP) and test if it correlates
    # with margin. If the game-level FIP diff correlates with margin
    # better than game-level ERA diff, FIP is the better feature.

    # For each game_pk, get the two starters
    starter_data = logs.merge(games[["game_pk"]], on="game_pk", how="inner")
    starter_data = starter_data.sort_values(["game_pk", "ip"], ascending=[True, False])

    game_features = []
    for gpk, gdf in starter_data.groupby("game_pk"):
        top2 = gdf.head(2)
        if len(top2) < 2:
            continue
        p1 = top2.iloc[0]  # higher IP = usually home team starter in wins
        p2 = top2.iloc[1]

        game_features.append({
            "game_pk": gpk,
            "game_date": p1["game_date"],
            # FIP differential (p1 - p2)
            "gl_fip3_diff": p1.get("rfip_3", np.nan) - p2.get("rfip_3", np.nan) if pd.notna(p1.get("rfip_3")) and pd.notna(p2.get("rfip_3")) else np.nan,
            "gl_fip5_diff": p1.get("rfip_5", np.nan) - p2.get("rfip_5", np.nan) if pd.notna(p1.get("rfip_5")) and pd.notna(p2.get("rfip_5")) else np.nan,
            "gl_sfip_diff": p1.get("season_fip", np.nan) - p2.get("season_fip", np.nan) if pd.notna(p1.get("season_fip")) and pd.notna(p2.get("season_fip")) else np.nan,
            # Individual FIP values
            "gl_fip3_p1": p1.get("rfip_3"),
            "gl_fip3_p2": p2.get("rfip_3"),
            "gl_fip5_p1": p1.get("rfip_5"),
            "gl_fip5_p2": p2.get("rfip_5"),
            "gl_sfip_p1": p1.get("season_fip"),
            "gl_sfip_p2": p2.get("season_fip"),
        })

    gf_df = pd.DataFrame(game_features)
    print(f"  Games with starter FIP data: {len(gf_df)}")

    # Merge with feature matrix
    df["game_pk"] = df["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")
    df = df.merge(gf_df, on="game_pk", how="left", suffixes=("", "_gf"))

    fip_fill = df["gl_fip3_diff"].notna().sum()
    print(f"  FIP joined to feature matrix: {fip_fill} / {len(df)} ({fip_fill/len(df)*100:.1f}%)")

    # Now run the actual test
    # Temporal split: 80/20
    split = int(len(df) * 0.80)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    print(f"\n  Train: {len(train)} games")
    print(f"  Test:  {len(test)} games ({str(test['game_date'].iloc[0])[:10]} → "
          f"{str(test['game_date'].iloc[-1])[:10]})")

    y_home_tr = train["home_score"].astype(float).values
    y_away_tr = train["away_score"].astype(float).values
    y_home_te = test["home_score"].astype(float).values
    y_away_te = test["away_score"].astype(float).values

    gbm_params = dict(
        loss="poisson", max_iter=300, max_depth=4,
        learning_rate=0.04, min_samples_leaf=10,
        max_features=0.8, random_state=42, early_stopping=False,
    )

    # Define feature sets
    fip_features = ["gl_fip3_p1", "gl_fip3_p2", "gl_fip5_p1", "gl_fip5_p2",
                     "gl_sfip_p1", "gl_sfip_p2", "gl_fip3_diff", "gl_fip5_diff",
                     "gl_sfip_diff"]
    non_era = [f for f in base_features if f not in era_in_model]

    configs = {
        "baseline (all ERA)": base_features,
        "no-ERA (drop ERA)": non_era,
        "ERA + FIP (both)": base_features + [f for f in fip_features if f in df.columns],
        "FIP-swap (replace ERA)": non_era + [f for f in fip_features if f in df.columns],
    }

    print(f"\n  ERA features in baseline: {era_in_model}")
    print(f"  FIP features available: {[f for f in fip_features if f in df.columns]}")

    print(f"\n  {'Config':<30} {'MAE_h':>8} {'MAE_a':>8} {'RMSE_m':>8} "
          f"{'LL_h':>10} {'LL_a':>10} {'N_feat':>7}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*7}")

    r = R_GLOBAL
    for config_name, feat_list in configs.items():
        avail = [f for f in feat_list if f in df.columns]

        X_tr = train[avail].values.astype(float)
        X_te = test[avail].values.astype(float)

        gbm_h = HistGradientBoostingRegressor(**gbm_params)
        gbm_h.fit(X_tr, y_home_tr)
        gbm_a = HistGradientBoostingRegressor(**gbm_params)
        gbm_a.fit(X_tr, y_away_tr)

        mu_h = np.clip(gbm_h.predict(X_te), 0.5, 15)
        mu_a = np.clip(gbm_a.predict(X_te), 0.5, 15)

        mae_h = np.mean(np.abs(mu_h - y_home_te))
        mae_a = np.mean(np.abs(mu_a - y_away_te))
        margin_pred = mu_h - mu_a
        margin_actual = y_home_te - y_away_te
        rmse_m = np.sqrt(np.mean((margin_pred - margin_actual) ** 2))

        # NegBin log-likelihood
        def _ll(y, mu_arr):
            total = 0
            for yi, mi in zip(y, mu_arr):
                mi = max(mi, 0.1)
                p = r / (r + mi)
                total += sp_stats.nbinom.logpmf(int(yi), r, p)
            return total

        ll_h = _ll(y_home_te, mu_h)
        ll_a = _ll(y_away_te, mu_a)

        print(f"  {config_name:<30} {mae_h:>8.4f} {mae_a:>8.4f} {rmse_m:>8.4f} "
              f"{ll_h:>10.1f} {ll_a:>10.1f} {len(avail):>7}")

    # Spread cover Brier comparison
    print(f"\n  Spread Cover Brier (OOS, +1.5 rung):")
    from scipy.stats import norm

    sigma = 4.217
    for config_name, feat_list in configs.items():
        avail = [f for f in feat_list if f in df.columns]
        X_tr = train[avail].values.astype(float)
        X_te = test[avail].values.astype(float)

        gbm_h = HistGradientBoostingRegressor(**gbm_params)
        gbm_h.fit(X_tr, y_home_tr)
        gbm_a = HistGradientBoostingRegressor(**gbm_params)
        gbm_a.fit(X_tr, y_away_tr)

        mu_h = np.clip(gbm_h.predict(X_te), 0.5, 15)
        mu_a = np.clip(gbm_a.predict(X_te), 0.5, 15)
        pred_margin = mu_h - mu_a
        actual_margin = y_home_te - y_away_te

        brier_sum = 0
        n = 0
        for rung in [1.5, 2.5, 3.5]:
            for i in range(len(pred_margin)):
                actual_cover = 1.0 if actual_margin[i] > -rung else (
                    0.0 if actual_margin[i] < -rung else np.nan)
                if np.isnan(actual_cover):
                    continue
                p_cover = float(1.0 - norm.cdf((-rung - pred_margin[i]) / sigma))
                brier_sum += (p_cover - actual_cover) ** 2
                n += 1

        brier = brier_sum / n if n > 0 else np.nan
        print(f"    {config_name:<30} Brier={brier:.6f}  (N={n})")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MULTI-SPLIT STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def multi_split_stability(logs_df):
    """5-fold temporal CV: does FIP beat ERA in majority of folds?"""
    import sqlite3
    from features import build_feature_matrix

    print("\n" + "=" * 110)
    print("  MULTI-SPLIT STABILITY: FIP vs ERA across 5 temporal folds")
    print("=" * 110)

    df = build_feature_matrix("MLB")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["home_score"] < 50].copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date_dt"] < "2026-01-01"].copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    # Build FIP features (reuse from mu_model_comparison)
    conn = sqlite3.connect(HIST_DB)
    logs = pd.read_sql_query("""
        SELECT player_id, pitcher_name, game_date, game_pk,
               ip, er, walks, strikeouts, home_runs
        FROM pitcher_game_logs WHERE home_runs IS NOT NULL
        ORDER BY player_id, game_date
    """, conn)
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    logs = logs.sort_values(["player_id", "game_date"]).copy()
    logs["fip_num"] = 13 * logs["home_runs"] + 3 * logs["walks"] - 2 * logs["strikeouts"]

    grp = logs.groupby("player_id")
    for n in [3, 5]:
        ip_s = grp["ip"].shift(1)
        fn_s = grp["fip_num"].shift(1)
        ip_r = ip_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        fn_r = fn_s.groupby(logs["player_id"]).rolling(n, min_periods=2).sum().droplevel(0)
        logs[f"rfip_{n}"] = (fn_r / ip_r + FIP_CONSTANT).where(ip_r > 0)

    logs["season"] = logs["game_date"].dt.year
    season_grp = logs.groupby(["player_id", "season"])
    ip_s_seas = season_grp["ip"].shift(1)
    fn_s_seas = season_grp["fip_num"].shift(1)
    ip_exp = ip_s_seas.groupby([logs["player_id"], logs["season"]]).expanding(min_periods=2).sum().droplevel([0,1])
    fn_exp = fn_s_seas.groupby([logs["player_id"], logs["season"]]).expanding(min_periods=2).sum().droplevel([0,1])
    logs["season_fip"] = (fn_exp / ip_exp + FIP_CONSTANT).where(ip_exp > 0)

    games = pd.read_sql_query("""
        SELECT game_id, game_date, home_team, away_team
        FROM games WHERE sport='MLB' AND status='Final'
    """, conn)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["game_pk"] = games["game_id"].str.replace("MLB_", "").astype(int)
    conn.close()

    starter_data = logs.merge(games[["game_pk"]], on="game_pk", how="inner")
    starter_data = starter_data.sort_values(["game_pk", "ip"], ascending=[True, False])

    game_features = []
    for gpk, gdf in starter_data.groupby("game_pk"):
        top2 = gdf.head(2)
        if len(top2) < 2:
            continue
        p1, p2 = top2.iloc[0], top2.iloc[1]
        game_features.append({
            "game_pk": gpk,
            "gl_fip3_p1": p1.get("rfip_3"),
            "gl_fip3_p2": p2.get("rfip_3"),
            "gl_fip5_p1": p1.get("rfip_5"),
            "gl_fip5_p2": p2.get("rfip_5"),
            "gl_sfip_p1": p1.get("season_fip"),
            "gl_sfip_p2": p2.get("season_fip"),
            "gl_fip3_diff": (p1.get("rfip_3", np.nan) - p2.get("rfip_3", np.nan)) if pd.notna(p1.get("rfip_3")) and pd.notna(p2.get("rfip_3")) else np.nan,
            "gl_fip5_diff": (p1.get("rfip_5", np.nan) - p2.get("rfip_5", np.nan)) if pd.notna(p1.get("rfip_5")) and pd.notna(p2.get("rfip_5")) else np.nan,
            "gl_sfip_diff": (p1.get("season_fip", np.nan) - p2.get("season_fip", np.nan)) if pd.notna(p1.get("season_fip")) and pd.notna(p2.get("season_fip")) else np.nan,
        })

    gf_df = pd.DataFrame(game_features)
    df["game_pk"] = df["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")
    df = df.merge(gf_df, on="game_pk", how="left")

    with open(os.path.join(SUBMODEL_DIR, "mlb_negbin_model.pkl"), "rb") as f:
        art = pickle.load(f)
    base_features = [c for c in art["feature_cols"] if c in df.columns]
    era_features = ["h_sp_era", "a_sp_era", "sp_era_diff",
                     "h_sp_era_3", "a_sp_era_3", "sp_era3_diff",
                     "h_sp_era_5", "a_sp_era_5", "sp_era5_diff"]
    era_in_model = [f for f in era_features if f in base_features]
    non_era = [f for f in base_features if f not in era_in_model]
    fip_features = ["gl_fip3_p1", "gl_fip3_p2", "gl_fip5_p1", "gl_fip5_p2",
                     "gl_sfip_p1", "gl_sfip_p2", "gl_fip3_diff", "gl_fip5_diff",
                     "gl_sfip_diff"]

    configs = {
        "baseline": base_features,
        "ERA+FIP": base_features + [f for f in fip_features if f in df.columns],
        "FIP-swap": non_era + [f for f in fip_features if f in df.columns],
    }

    gbm_params = dict(
        loss="poisson", max_iter=300, max_depth=4,
        learning_rate=0.04, min_samples_leaf=10,
        max_features=0.8, random_state=42, early_stopping=False,
    )

    # 5-fold temporal CV
    tscv = TimeSeriesSplit(n_splits=5)

    print(f"\n  {'Config':<20} ", end="")
    for i in range(5):
        print(f"{'Fold '+str(i+1):>10} ", end="")
    print(f"{'Mean':>10} {'Wins':>6}")
    print(f"  {'─'*20} " + "─" * 10 + " " + ("─" * 10 + " ") * 4 + "─" * 10 + " " + "─" * 6)

    fold_results = {name: [] for name in configs}

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(df)):
        tr = df.iloc[train_idx]
        te = df.iloc[test_idx]
        y_h_tr = tr["home_score"].astype(float).values
        y_a_tr = tr["away_score"].astype(float).values
        y_h_te = te["home_score"].astype(float).values
        y_a_te = te["away_score"].astype(float).values

        for name, feat_list in configs.items():
            avail = [f for f in feat_list if f in df.columns]
            X_tr = tr[avail].values.astype(float)
            X_te = te[avail].values.astype(float)

            gbm_h = HistGradientBoostingRegressor(**gbm_params)
            gbm_h.fit(X_tr, y_h_tr)
            gbm_a = HistGradientBoostingRegressor(**gbm_params)
            gbm_a.fit(X_tr, y_a_tr)

            mu_h = np.clip(gbm_h.predict(X_te), 0.5, 15)
            mu_a = np.clip(gbm_a.predict(X_te), 0.5, 15)
            mae = (np.mean(np.abs(mu_h - y_h_te)) + np.mean(np.abs(mu_a - y_a_te))) / 2
            fold_results[name].append(mae)

    for name, maes in fold_results.items():
        wins = sum(1 for i in range(5) if maes[i] == min(fold_results[n][i] for n in configs))
        print(f"  {name:<20} ", end="")
        for m in maes:
            print(f"{m:>10.4f} ", end="")
        print(f"{np.mean(maes):>10.4f} {wins:>6}/5")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logs, starters, oos = correlation_test()
    mu_model_comparison(logs)
    multi_split_stability(logs)
