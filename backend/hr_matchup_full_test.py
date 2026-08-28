"""
FULL 4-Way HR Matchup Test — Methodological Completeness
=========================================================
Tests whether batter+pitcher HR interaction on SPREADS works with:
  - ALL available PIT features (not just simple means)
  - Derived features: blow-up-freq, volatility, ISO, concentration, form-delta
  - Multiple interaction constructions
  - Full gates: Bonferroni, split-half, monthly fingerprint, adequate N

DATA AVAILABILITY NOTE:
  - Game-level Statcast (barrel%, exit velo, launch angle) NOT in database
  - Season-level barrel_pct/hard_hit_pct exist but = LOOKAHEAD for individual games
  - We use PIT outcome-based proxies: ISO, SLG, HR/AB, TB/AB (capture similar info)
"""
import sqlite3, json, numpy as np, pandas as pd
from scipy import stats
from numpy.linalg import lstsq

DB = '/Users/joseportillo/Downloads/BetIQ/data/historical.db'

# ── Shrinkage helper ─────────────────────────────────────────────────
def shrink(raw, n, league_mean, prior_weight):
    w = n / (n + prior_weight)
    return w * raw + (1 - w) * league_mean


# ── 1. PIT Pitching Features ────────────────────────────────────────
def build_pit_pitching():
    """
    From pitcher_game_logs, compute PIT:
      sp_hr9       — cumulative HR/9 (shrunk)
      sp_blowup    — % of prior starts with 2+ HR allowed
      sp_hr9_vol   — std of per-start HR/9
      sp_k9        — cumulative K/9 (shrunk)
      sp_whip      — cumulative WHIP (shrunk)
    All use shift-1 (only prior starts), min 3 prior starts.
    """
    db = sqlite3.connect(DB)
    logs = pd.read_sql("""
        SELECT player_id, game_date, ip, home_runs, strikeouts, walks, hits
        FROM pitcher_game_logs
        WHERE ip > 0
        ORDER BY player_id, game_date
    """, db)
    db.close()

    # League means for shrinkage
    total_ip = logs['ip'].sum()
    lg_hr9 = (logs['home_runs'].sum() / total_ip) * 9
    lg_k9  = (logs['strikeouts'].sum() / total_ip) * 9
    lg_whip = (logs['walks'].sum() + logs['hits'].sum()) / total_ip
    lg_blowup = (logs['home_runs'] >= 2).mean()

    PRIOR_N = 8
    MIN_STARTS = 3

    results = []
    for pid, grp in logs.groupby('player_id'):
        grp = grp.sort_values('game_date').reset_index(drop=True)
        cum_ip, cum_hr, cum_k, cum_bb, cum_h = 0.0, 0, 0, 0, 0
        n_prior = 0
        prior_hr9_list = []
        prior_blowup_count = 0

        for _, row in grp.iterrows():
            if n_prior >= MIN_STARTS and cum_ip > 0:
                # HR/9
                raw_hr9 = (cum_hr / cum_ip) * 9
                pit_hr9 = shrink(raw_hr9, n_prior, lg_hr9, PRIOR_N)

                # Blow-up freq
                raw_blowup = prior_blowup_count / n_prior
                pit_blowup = shrink(raw_blowup, n_prior, lg_blowup, PRIOR_N)

                # HR/9 volatility (std of per-start HR/9)
                if len(prior_hr9_list) >= 3:
                    hr9_vol = np.std(prior_hr9_list, ddof=1)
                else:
                    hr9_vol = np.nan

                # K/9
                raw_k9 = (cum_k / cum_ip) * 9
                pit_k9 = shrink(raw_k9, n_prior, lg_k9, PRIOR_N)

                # WHIP
                raw_whip = (cum_bb + cum_h) / cum_ip
                pit_whip = shrink(raw_whip, n_prior, lg_whip, PRIOR_N)

                results.append({
                    'player_id': pid,
                    'game_date': row['game_date'],
                    'sp_hr9': pit_hr9,
                    'sp_blowup': pit_blowup,
                    'sp_hr9_vol': hr9_vol,
                    'sp_k9': pit_k9,
                    'sp_whip': pit_whip,
                    'sp_n_prior': n_prior,
                })

            # Update cumulatives
            start_hr9 = (row['home_runs'] / row['ip']) * 9 if row['ip'] > 0 else 0
            prior_hr9_list.append(start_hr9)
            if row['home_runs'] >= 2:
                prior_blowup_count += 1
            cum_ip += row['ip']
            cum_hr += row['home_runs']
            cum_k  += row['strikeouts']
            cum_bb += row['walks']
            cum_h  += row['hits']
            n_prior += 1

    sp = pd.DataFrame(results)
    print(f"[Pitching] {len(sp)} pitcher-starts, {sp['player_id'].nunique()} pitchers")
    print(f"  sp_hr9:    mean={sp['sp_hr9'].mean():.3f}, std={sp['sp_hr9'].std():.3f}")
    print(f"  sp_blowup: mean={sp['sp_blowup'].mean():.3f}, std={sp['sp_blowup'].std():.3f}")
    print(f"  sp_hr9_vol: mean={sp['sp_hr9_vol'].dropna().mean():.3f}, std={sp['sp_hr9_vol'].dropna().std():.3f}")
    print(f"  sp_k9:     mean={sp['sp_k9'].mean():.3f}, std={sp['sp_k9'].std():.3f}")
    print(f"  sp_whip:   mean={sp['sp_whip'].mean():.3f}, std={sp['sp_whip'].std():.3f}")
    return sp


# ── 2. PIT Team Batting Features ────────────────────────────────────
def build_pit_batting():
    """
    From team_game_batting, compute PIT:
      bat_hr_pg    — HR per game (shrunk)
      bat_hr_ab    — HR per AB (shrunk) — more granular
      bat_iso      — isolated power = (TB - H) / AB (shrunk)
      bat_slg      — slugging = TB / AB (shrunk)
      bat_k_rate   — K / PA (shrunk)
      bat_form_delta — rolling-10 HR/game minus season HR/game (form)
    """
    db = sqlite3.connect(DB)
    bat = pd.read_sql("""
        SELECT team_name as team, game_date, at_bats, hits, home_runs,
               total_bases, strikeouts, walks, hit_by_pitch, sac_flies
        FROM team_game_batting
        ORDER BY team_name, game_date
    """, db)
    db.close()

    # League means
    lg_hr_pg = bat['home_runs'].mean()
    total_ab = bat['at_bats'].sum()
    lg_hr_ab = bat['home_runs'].sum() / total_ab if total_ab > 0 else 0
    lg_iso = (bat['total_bases'].sum() - bat['hits'].sum()) / total_ab if total_ab > 0 else 0
    lg_slg = bat['total_bases'].sum() / total_ab if total_ab > 0 else 0
    pa = bat['at_bats'] + bat['walks'] + bat['hit_by_pitch'] + bat['sac_flies']
    lg_k_rate = bat['strikeouts'].sum() / pa.sum() if pa.sum() > 0 else 0

    PRIOR_GAMES = 15
    MIN_GAMES = 5
    FORM_WINDOW = 10

    results = []
    for team, grp in bat.groupby('team'):
        grp = grp.sort_values('game_date').reset_index(drop=True)
        cum_ab, cum_h, cum_hr, cum_tb, cum_k, cum_pa = 0, 0, 0, 0, 0, 0
        n_prior = 0
        recent_hr = []  # last FORM_WINDOW games

        for _, row in grp.iterrows():
            if n_prior >= MIN_GAMES and cum_ab > 0:
                # HR per game
                raw_hr_pg = cum_hr / n_prior
                pit_hr_pg = shrink(raw_hr_pg, n_prior, lg_hr_pg, PRIOR_GAMES)

                # HR per AB
                raw_hr_ab = cum_hr / cum_ab
                pit_hr_ab = shrink(raw_hr_ab, n_prior, lg_hr_ab, PRIOR_GAMES)

                # ISO = (TB - H) / AB
                raw_iso = (cum_tb - cum_h) / cum_ab
                pit_iso = shrink(raw_iso, n_prior, lg_iso, PRIOR_GAMES)

                # SLG = TB / AB
                raw_slg = cum_tb / cum_ab
                pit_slg = shrink(raw_slg, n_prior, lg_slg, PRIOR_GAMES)

                # K rate
                raw_k = cum_k / cum_pa if cum_pa > 0 else lg_k_rate
                pit_k = shrink(raw_k, n_prior, lg_k_rate, PRIOR_GAMES)

                # Form-delta: rolling-10 HR/game vs cumulative HR/game
                if len(recent_hr) >= FORM_WINDOW:
                    recent_hr_pg = np.mean(recent_hr[-FORM_WINDOW:])
                    form_delta = recent_hr_pg - raw_hr_pg
                else:
                    form_delta = 0.0

                results.append({
                    'team': team,
                    'game_date': row['game_date'],
                    'bat_hr_pg': pit_hr_pg,
                    'bat_hr_ab': pit_hr_ab,
                    'bat_iso': pit_iso,
                    'bat_slg': pit_slg,
                    'bat_k_rate': pit_k,
                    'bat_form_delta': form_delta,
                    'bat_n_prior': n_prior,
                })

            # Update
            game_pa = row['at_bats'] + row['walks'] + row['hit_by_pitch'] + row['sac_flies']
            cum_ab += row['at_bats']
            cum_h  += row['hits']
            cum_hr += row['home_runs']
            cum_tb += row['total_bases']
            cum_k  += row['strikeouts']
            cum_pa += game_pa
            recent_hr.append(row['home_runs'])
            n_prior += 1

    bt = pd.DataFrame(results)
    print(f"\n[Batting] {len(bt)} team-dates, {bt['team'].nunique()} teams")
    print(f"  bat_hr_pg:     mean={bt['bat_hr_pg'].mean():.3f}")
    print(f"  bat_hr_ab:     mean={bt['bat_hr_ab'].mean():.4f}")
    print(f"  bat_iso:       mean={bt['bat_iso'].mean():.4f}")
    print(f"  bat_slg:       mean={bt['bat_slg'].mean():.4f}")
    print(f"  bat_k_rate:    mean={bt['bat_k_rate'].mean():.4f}")
    print(f"  bat_form_delta: mean={bt['bat_form_delta'].mean():.4f}, std={bt['bat_form_delta'].std():.4f}")
    return bt


# ── 3. HR Concentration (Herfindahl from batter_game_stats) ────────
def build_hr_concentration():
    """
    From batter_game_stats, compute per team-date:
      hr_hhi — Herfindahl index of HR among batters (high = concentrated in stars)
    Point-in-time: uses only prior games' HR distribution.
    """
    db = sqlite3.connect(DB)
    # Check if table exists
    cur = db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='batter_game_stats'")
    if not cur.fetchone():
        db.close()
        print("\n[Concentration] batter_game_stats table NOT FOUND — skipping")
        return None

    bgs = pd.read_sql("""
        SELECT bgs.game_pk, bgs.player_id, bgs.team_name, bgs.hr, tgb.game_date
        FROM batter_game_stats bgs
        JOIN team_game_batting tgb
          ON bgs.game_pk = tgb.game_pk AND bgs.team_name = tgb.team_name
        ORDER BY bgs.team_name, tgb.game_date
    """, db)
    db.close()

    if len(bgs) == 0:
        print("\n[Concentration] No data after join — skipping")
        return None

    MIN_GAMES = 10

    results = []
    for team, grp in bgs.groupby('team_name'):
        dates = sorted(grp['game_date'].unique())
        # Track cumulative HR per player
        player_hr = {}
        total_hr = 0
        n_dates = 0

        for d in dates:
            if n_dates >= MIN_GAMES and total_hr > 0:
                # Compute HHI from cumulative HR shares
                shares = np.array([h / total_hr for h in player_hr.values() if h > 0])
                hhi = np.sum(shares ** 2) if len(shares) > 0 else 1.0
                results.append({
                    'team': team,
                    'game_date': d,
                    'hr_hhi': hhi,
                })

            # Update with this date's HR
            day_data = grp[grp['game_date'] == d]
            for _, row in day_data.iterrows():
                pid = row['player_id']
                player_hr[pid] = player_hr.get(pid, 0) + row['hr']
                total_hr += row['hr']
            n_dates += 1

    if results:
        conc = pd.DataFrame(results)
        print(f"\n[Concentration] {len(conc)} team-dates")
        print(f"  hr_hhi: mean={conc['hr_hhi'].mean():.4f}, std={conc['hr_hhi'].std():.4f}")
        return conc
    print("\n[Concentration] No results computed")
    return None


# ── 4. Game Spine ────────────────────────────────────────────────────
def build_game_spine():
    db = sqlite3.connect(DB)
    lines = pd.read_sql("""
        SELECT game_id, game_date, home_team, away_team, close_spread, close_total
        FROM betting_lines
        WHERE sport = 'MLB' AND close_spread IS NOT NULL
    """, db)
    home = pd.read_sql("""
        SELECT game_id, team as home_team, score as home_score,
               opp_score as away_score, stats_json
        FROM team_stats
        WHERE sport = 'MLB' AND is_home = 1
    """, db)
    db.close()

    games = lines.merge(home, on=['game_id', 'home_team'], how='inner')
    games = games[(games['home_score'] > 0) | (games['away_score'] > 0)].copy()
    games['actual_margin'] = games['home_score'] - games['away_score']
    games['spread_residual'] = games['actual_margin'] + games['close_spread']
    games['home_covered'] = (games['spread_residual'] > 0).astype(int)

    def get_sp(js, key):
        try:
            d = json.loads(js) if isinstance(js, str) else {}
            return d.get(key)
        except:
            return None

    games['home_sp_name'] = games['stats_json'].apply(lambda x: get_sp(x, 'home_probable_pitcher'))
    games['away_sp_name'] = games['stats_json'].apply(lambda x: get_sp(x, 'away_probable_pitcher'))
    print(f"\n[Spine] {len(games)} settled MLB games")
    return games


# ── 5. Link everything ──────────────────────────────────────────────
def link_features(games, sp, bt, conc):
    db = sqlite3.connect(DB)
    name_map = pd.read_sql("""
        SELECT DISTINCT pitcher_name, player_id
        FROM pitcher_stats WHERE player_id IS NOT NULL
    """, db)
    db.close()

    name_to_id = dict(zip(name_map['pitcher_name'], name_map['player_id']))
    games['home_sp_id'] = games['home_sp_name'].map(name_to_id)
    games['away_sp_id'] = games['away_sp_name'].map(name_to_id)

    # Join SP features (home SP)
    sp_home = sp.rename(columns={c: f'home_{c}' if c not in ['player_id', 'game_date'] else c
                                  for c in sp.columns})
    sp_home = sp_home.rename(columns={'player_id': 'home_sp_id'})
    games = games.merge(sp_home, on=['home_sp_id', 'game_date'], how='left')

    # Join SP features (away SP)
    sp_away = sp.rename(columns={c: f'away_{c}' if c not in ['player_id', 'game_date'] else c
                                  for c in sp.columns})
    sp_away = sp_away.rename(columns={'player_id': 'away_sp_id'})
    games = games.merge(sp_away, on=['away_sp_id', 'game_date'], how='left')

    # Join batting (home)
    bt_home = bt.rename(columns={c: f'home_{c}' if c not in ['team', 'game_date'] else c
                                  for c in bt.columns})
    bt_home = bt_home.rename(columns={'team': 'home_team'})
    games = games.merge(bt_home, on=['home_team', 'game_date'], how='left')

    # Join batting (away)
    bt_away = bt.rename(columns={c: f'away_{c}' if c not in ['team', 'game_date'] else c
                                  for c in bt.columns})
    bt_away = bt_away.rename(columns={'team': 'away_team'})
    games = games.merge(bt_away, on=['away_team', 'game_date'], how='left')

    # Join concentration
    if conc is not None:
        conc_home = conc.rename(columns={'team': 'home_team', 'hr_hhi': 'home_hr_hhi'})
        conc_away = conc.rename(columns={'team': 'away_team', 'hr_hhi': 'away_hr_hhi'})
        games = games.merge(conc_home, on=['home_team', 'game_date'], how='left')
        games = games.merge(conc_away, on=['away_team', 'game_date'], how='left')

    # Count coverage
    sp_cols = [c for c in games.columns if c.startswith('home_sp_') and c != 'home_sp_id' and c != 'home_sp_name' and c != 'home_sp_n_prior']
    bat_cols = [c for c in games.columns if c.startswith('home_bat_') and c != 'home_bat_n_prior']
    core_mask = (games['home_sp_hr9'].notna() & games['away_sp_hr9'].notna() &
                 games['home_bat_hr_pg'].notna() & games['away_bat_hr_pg'].notna())
    print(f"\n[Link] Games with core features: {core_mask.sum()}")
    return games, core_mask


# ── 6. Build ALL interaction features ────────────────────────────────
def build_interactions(df):
    """
    Build every interaction feature we can from available PIT data.
    """
    features = {}

    # ── A. Simple diffs (LINEAR, not interaction) ────────────────────
    features['sp_hr9_diff'] = df['home_sp_hr9'] - df['away_sp_hr9']
    features['bat_hr_pg_diff'] = df['home_bat_hr_pg'] - df['away_bat_hr_pg']
    features['bat_iso_diff'] = df['home_bat_iso'] - df['away_bat_iso']
    features['bat_slg_diff'] = df['home_bat_slg'] - df['away_bat_slg']
    features['sp_k9_diff'] = df['home_sp_k9'] - df['away_sp_k9']
    features['sp_whip_diff'] = df['home_sp_whip'] - df['away_sp_whip']
    features['bat_form_delta_diff'] = df['home_bat_form_delta'] - df['away_bat_form_delta']

    # ── B. Classic 4-way (HR × HR) ──────────────────────────────────
    features['hr_exploit_net'] = (
        df['home_bat_hr_pg'] * df['away_sp_hr9'] -
        df['away_bat_hr_pg'] * df['home_sp_hr9']
    )

    # ── C. ISO × SP HR/9 (power quality × pitcher vulnerability) ───
    features['iso_exploit_net'] = (
        df['home_bat_iso'] * df['away_sp_hr9'] -
        df['away_bat_iso'] * df['home_sp_hr9']
    )

    # ── D. SLG × SP HR/9 ───────────────────────────────────────────
    features['slg_exploit_net'] = (
        df['home_bat_slg'] * df['away_sp_hr9'] -
        df['away_bat_slg'] * df['home_sp_hr9']
    )

    # ── E. HR/AB × SP HR/9 (more granular batting) ─────────────────
    features['hr_ab_exploit_net'] = (
        df['home_bat_hr_ab'] * df['away_sp_hr9'] -
        df['away_bat_hr_ab'] * df['home_sp_hr9']
    )

    # ── F. Bat power × SP blow-up frequency ────────────────────────
    if 'home_sp_blowup' in df.columns and df['home_sp_blowup'].notna().any():
        features['blowup_exploit_net'] = (
            df['home_bat_hr_pg'] * df['away_sp_blowup'] -
            df['away_bat_hr_pg'] * df['home_sp_blowup']
        )

    # ── G. Bat power × SP HR/9 volatility ──────────────────────────
    if 'home_sp_hr9_vol' in df.columns and df['home_sp_hr9_vol'].notna().any():
        features['vol_exploit_net'] = (
            df['home_bat_hr_pg'] * df['away_sp_hr9_vol'] -
            df['away_bat_hr_pg'] * df['home_sp_hr9_vol']
        )

    # ── H. ISO × WHIP (quality vs overall pitcher weakness) ────────
    features['iso_whip_exploit_net'] = (
        df['home_bat_iso'] * df['away_sp_whip'] -
        df['away_bat_iso'] * df['home_sp_whip']
    )

    # ── I. SLG × K/9 (contact quality vs strikeout ability) ────────
    # INVERTED: high K/9 = good pitcher = less exploitable
    features['slg_vs_k9_net'] = (
        df['home_bat_slg'] / (df['away_sp_k9'] + 1) -
        df['away_bat_slg'] / (df['home_sp_k9'] + 1)
    )

    # ── J. Form-delta × SP HR/9 (hot-streak exploit) ──────────────
    features['form_exploit_net'] = (
        df['home_bat_form_delta'] * df['away_sp_hr9'] -
        df['away_bat_form_delta'] * df['home_sp_hr9']
    )

    # ── K. HR concentration (HHI) ─────────────────────────────────
    if 'home_hr_hhi' in df.columns and df['home_hr_hhi'].notna().any():
        features['hr_hhi_diff'] = df['home_hr_hhi'] - df['away_hr_hhi']
        # Concentrated × SP HR/9 interaction
        features['conc_exploit_net'] = (
            df['home_hr_hhi'] * df['home_bat_hr_pg'] * df['away_sp_hr9'] -
            df['away_hr_hhi'] * df['away_bat_hr_pg'] * df['home_sp_hr9']
        )

    return features


# ── 7. Full Gate Test ────────────────────────────────────────────────
def test_signal(df, signal_values, name, n_total_tests, be=0.5238):
    """Test one signal through all gates. Returns dict of results."""
    valid = signal_values.notna()
    sig = signal_values[valid].values
    res = df.loc[valid, 'spread_residual'].values
    cov = df.loc[valid, 'home_covered'].values
    dates = df.loc[valid, 'game_date'].values
    N = len(sig)

    if N < 100:
        return {'name': name, 'N': N, 'verdict': 'SKIP (N<100)'}

    # Correlation
    r, p = stats.pearsonr(sig, res)
    p_bonf = min(p * n_total_tests, 1.0)

    # Split-half (chronological)
    idx = np.argsort(dates)
    mid = N // 2
    r1, _ = stats.pearsonr(sig[idx[:mid]], res[idx[:mid]])
    r2, _ = stats.pearsonr(sig[idx[mid:]], res[idx[mid:]])
    split_stable = np.sign(r1) == np.sign(r2) and r1 != 0 and r2 != 0

    # Effect: top vs bottom quartile
    q75, q25 = np.percentile(sig, 75), np.percentile(sig, 25)
    top_mask = sig >= q75
    bot_mask = sig <= q25
    top_wr = cov[top_mask].mean() if top_mask.sum() > 0 else np.nan
    bot_wr = cov[bot_mask].mean() if bot_mask.sum() > 0 else np.nan

    return {
        'name': name,
        'N': N,
        'r': r,
        'p_raw': p,
        'p_bonf': p_bonf,
        'r_h1': r1,
        'r_h2': r2,
        'split_stable': split_stable,
        'top_q_wr': top_wr,
        'bot_q_wr': bot_wr,
        'top_q_n': top_mask.sum(),
        'bot_q_n': bot_mask.sum(),
        'edge_pp': (top_wr - be) * 100 if not np.isnan(top_wr) else np.nan,
        'verdict': '',
    }


def monthly_fingerprint(df, signal_values, name):
    """Monthly r for a signal — check for flat=lookahead or gradient=real."""
    valid = signal_values.notna()
    months = pd.to_datetime(df.loc[valid, 'game_date']).dt.month
    sig = signal_values[valid]
    res = df.loc[valid, 'spread_residual']
    print(f"\n  Monthly fingerprint: {name}")
    for m in sorted(months.unique()):
        mask = months == m
        if mask.sum() < 50:
            continue
        r, p = stats.pearsonr(sig[mask], res[mask])
        print(f"    Month {m:2d}: r={r:+.4f}, p={p:.3f}, N={mask.sum()}")


# ── 8. Interaction Residual Test ─────────────────────────────────────
def interaction_residual_test(df, interaction_col, linear_cols, name):
    """
    After removing linear components via OLS, is there an interaction residual?
    """
    valid = df[interaction_col].notna()
    for c in linear_cols:
        valid &= df[c].notna()
    dv = df[valid].copy()

    X = np.column_stack([dv[c].values for c in linear_cols] + [np.ones(len(dv))])
    y = dv[interaction_col].values
    beta, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ beta

    r, p = stats.pearsonr(resid, dv['spread_residual'].values)
    print(f"\n  Interaction residual ({name}):")
    print(f"    After removing {linear_cols}: r={r:+.4f}, p={p:.4f}")
    print(f"    (Dead zero = interaction adds nothing beyond additive components)")
    return r, p


# ── 9. Best-of stacking test ────────────────────────────────────────
def stacking_test(df, feature_dict, n_best=5):
    """
    Take the top-N features by |r| and test a simple equal-weight stack.
    This checks if combining features extracts more than any single one.
    """
    # Get correlations
    cors = []
    for name, vals in feature_dict.items():
        valid = vals.notna()
        if valid.sum() < 100:
            continue
        r, _ = stats.pearsonr(vals[valid], df.loc[valid, 'spread_residual'])
        cors.append((name, abs(r), r))
    cors.sort(key=lambda x: x[1], reverse=True)

    if len(cors) < 2:
        print("\n  [Stack] Too few valid features for stacking")
        return

    print(f"\n  Top-{min(n_best, len(cors))} features by |r|:")
    for name, ar, r in cors[:n_best]:
        print(f"    {name}: r={r:+.4f}")

    # Stack top features (z-score normalize, equal weight)
    top_names = [c[0] for c in cors[:n_best]]
    stack_parts = []
    for name in top_names:
        v = feature_dict[name].copy()
        v_z = (v - v.mean()) / v.std()
        stack_parts.append(v_z)

    stack = pd.concat(stack_parts, axis=1)
    # Require all non-null
    valid = stack.notna().all(axis=1)
    if valid.sum() < 100:
        print("  [Stack] Too few valid rows after alignment")
        return

    composite = stack[valid].mean(axis=1)
    r, p = stats.pearsonr(composite, df.loc[valid, 'spread_residual'])
    p_bonf = min(p * (len(cors) + 1), 1.0)

    # Split-half
    dates = df.loc[valid, 'game_date'].values
    idx = np.argsort(dates)
    mid = len(idx) // 2
    r1, _ = stats.pearsonr(composite.values[idx[:mid]],
                           df.loc[valid, 'spread_residual'].values[idx[:mid]])
    r2, _ = stats.pearsonr(composite.values[idx[mid:]],
                           df.loc[valid, 'spread_residual'].values[idx[mid:]])

    print(f"\n  Equal-weight stack of top-{len(top_names)}: r={r:+.4f}, p_bonf={p_bonf:.4f}")
    print(f"  Split-half: r1={r1:+.4f}, r2={r2:+.4f} {'STABLE' if np.sign(r1)==np.sign(r2) else 'FLIPS'}")


# ── MAIN ─────────────────────────────────────────────────────────────
def run():
    print("=" * 70)
    print("FULL 4-WAY HR MATCHUP: ALL PIT Features + Derived + Interactions")
    print("=" * 70)
    print("\nDATA NOTE: Game-level Statcast (barrel%, exit velo, launch angle)")
    print("NOT in database. Using PIT outcome proxies: ISO, SLG, HR/AB, TB/AB.")
    print("Season barrel_pct/hard_hit_pct exist but would be LOOKAHEAD — not used.\n")

    # ── Build features ───────────────────────────────────────────────
    sp = build_pit_pitching()
    bt = build_pit_batting()
    conc = build_hr_concentration()
    games = build_game_spine()
    df, core_mask = link_features(games, sp, bt, conc)

    # Filter to games with core features
    df = df[core_mask].copy()
    print(f"\n[Analysis set] {len(df)} games with all core features")

    # ── Build interactions ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FEATURE CONSTRUCTION")
    print("=" * 70)

    features = build_interactions(df)
    print(f"\n  {len(features)} features constructed:")
    for name in sorted(features.keys()):
        v = features[name]
        n_valid = v.notna().sum()
        print(f"    {name}: N_valid={n_valid}")

    # ── Individual signal tests ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("INDIVIDUAL SIGNAL TESTS (vs close residual, Bonferroni-corrected)")
    print("=" * 70)

    n_tests = len(features)
    results = []
    for name in sorted(features.keys()):
        r = test_signal(df, features[name], name, n_tests)
        results.append(r)

    # Print results table
    print(f"\n  {'Feature':<30} {'r':>7} {'p_bonf':>8} {'r_h1':>7} {'r_h2':>7} {'Split':>6} {'TopQ%':>6} {'Edge':>7} {'N':>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")
    for r in sorted(results, key=lambda x: abs(x.get('r', 0)), reverse=True):
        if r.get('verdict') == 'SKIP (N<100)':
            print(f"  {r['name']:<30} {'SKIP':>7} {'N<100':>8}")
            continue
        split = 'Y' if r['split_stable'] else 'N'
        top_pct = f"{r['top_q_wr']*100:.1f}" if not np.isnan(r['top_q_wr']) else 'N/A'
        edge = f"{r['edge_pp']:+.1f}" if not np.isnan(r['edge_pp']) else 'N/A'
        print(f"  {r['name']:<30} {r['r']:+.4f} {r['p_bonf']:.4f} {r['r_h1']:+.4f} {r['r_h2']:+.4f} {split:>6} {top_pct:>6} {edge:>7} {r['N']:>6}")

    # ── Survivors? ───────────────────────────────────────────────────
    survivors = [r for r in results if r.get('p_bonf', 1) < 0.05 and r.get('split_stable', False)]
    print(f"\n  Survivors (p_bonf < 0.05 AND split-half stable): {len(survivors)}")
    for s in survivors:
        print(f"    {s['name']}: r={s['r']:+.4f}, p_bonf={s['p_bonf']:.4f}, "
              f"edge={s['edge_pp']:+.1f}pp")

    # Add features to df for interaction residual tests
    for name, vals in features.items():
        df[name] = vals.values if hasattr(vals, 'values') else vals

    # ── Interaction residual tests ───────────────────────────────────
    print(f"\n{'='*70}")
    print("INTERACTION RESIDUAL TESTS (beyond linear components)")
    print("=" * 70)

    # Classic: HR exploit beyond SP HR/9 diff + batting power diff
    interaction_residual_test(
        df, 'hr_exploit_net',
        ['sp_hr9_diff', 'bat_hr_pg_diff'],
        'HR × HR beyond linear HR'
    )

    # ISO: ISO exploit beyond SP HR/9 diff + batting ISO diff
    if 'iso_exploit_net' in features:
        interaction_residual_test(
            df, 'iso_exploit_net',
            ['sp_hr9_diff', 'bat_iso_diff'],
            'ISO × HR/9 beyond linear'
        )

    # SLG: SLG exploit beyond SP HR/9 diff + batting SLG diff
    if 'slg_exploit_net' in features:
        interaction_residual_test(
            df, 'slg_exploit_net',
            ['sp_hr9_diff', 'bat_slg_diff'],
            'SLG × HR/9 beyond linear'
        )

    # Blow-up: blowup exploit beyond SP blowup diff + batting power diff
    if 'blowup_exploit_net' in features:
        df['sp_blowup_diff'] = df['home_sp_blowup'] - df['away_sp_blowup']
        interaction_residual_test(
            df, 'blowup_exploit_net',
            ['sp_blowup_diff', 'bat_hr_pg_diff'],
            'HR × Blow-up beyond linear'
        )

    # ── Monthly fingerprint for top signals ──────────────────────────
    print(f"\n{'='*70}")
    print("MONTHLY FINGERPRINT (top signals)")
    print("=" * 70)

    # Always show top 3 by |r|
    top_by_r = sorted(results, key=lambda x: abs(x.get('r', 0)), reverse=True)[:3]
    for r in top_by_r:
        if r['name'] in features:
            monthly_fingerprint(df, features[r['name']], r['name'])

    # ── Stacking test ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STACKING TEST (top-5 by |r|, equal-weight z-score)")
    print("=" * 70)

    feat_series = {name: pd.Series(vals.values, index=df.index)
                   for name, vals in features.items()
                   if isinstance(vals, pd.Series)}
    stacking_test(df, feat_series)

    # ── Close absorption ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CLOSE ABSORPTION (how much does the spread already price?)")
    print("=" * 70)

    for name in ['hr_exploit_net', 'iso_exploit_net', 'slg_exploit_net']:
        if name in features:
            v = features[name]
            valid = v.notna()
            if valid.sum() < 100:
                continue
            r_close, _ = stats.pearsonr(v[valid], df.loc[valid, 'close_spread'])
            print(f"  {name} vs close_spread: r={r_close:+.4f} "
                  f"({'HIGH' if abs(r_close) > 0.3 else 'LOW'} absorption)")

    # ── FINAL VERDICT ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print("=" * 70)

    n_survivors = len(survivors)
    print(f"\n  Total features tested: {len(features)}")
    print(f"  Bonferroni threshold: p < {0.05/len(features):.5f} (raw)")
    print(f"  Survivors (p_bonf<0.05 + split-stable): {n_survivors}")

    if n_survivors == 0:
        print(f"""
  ALL DEAD. Neither simple HR rates, ISO, SLG, blow-up frequency,
  HR/9 volatility, form-delta, K-rate, WHIP, HR concentration,
  nor ANY of their cross-interactions predict spread residuals
  vs the closing line.

  Data limitations:
  - Game-level Statcast (barrel%, exit velo, launch angle) NOT available
  - Used outcome-based PIT proxies (ISO, SLG, HR/AB) instead
  - Season barrel_pct/hard_hit_pct exist but excluded (would be lookahead)

  The closing line absorbs public batting and pitching statistics
  AND their interactions. The "use public data better" thesis for
  HR matchups on spreads is DEAD across all available constructions.
""")
    else:
        print(f"\n  {n_survivors} survivor(s) — inspect edge size and interaction residual")
        print(f"  before building shadow arm. Remember: inverted signal = NOT exploitable.")
        for s in survivors:
            direction = 'CORRECT' if s['r'] > 0 else 'INVERTED'
            print(f"    {s['name']}: {direction} direction, edge={s['edge_pp']:+.1f}pp")


if __name__ == '__main__':
    run()
