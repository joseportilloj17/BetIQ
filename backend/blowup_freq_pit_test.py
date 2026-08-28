"""
Blow-Up Frequency / Dispersion Point-in-Time Test
===================================================
Tests whether pitcher blow-up frequency (% of starts with 2+ HR)
and HR/9 dispersion create a real point-in-time totals signal
that mean HR/9 (which died PIT) does not.
"""
import sqlite3, json, numpy as np, pandas as pd
from scipy import stats
from collections import defaultdict

DB = '/Users/joseportillo/Downloads/BetIQ/data/historical.db'

def build_pit_pitcher_features():
    """
    For each pitcher start, compute POINT-IN-TIME features
    using ONLY prior starts (shift-1 pattern):
    - mean_hr9: cumulative mean HR/9 from prior starts
    - blowup_freq: % of prior starts with 2+ HR allowed
    - hr9_std: std of per-start HR/9 across prior starts
    - hr9_cv: coefficient of variation (std/mean)
    All with Bayesian shrinkage toward league average by sample size.
    """
    db = sqlite3.connect(DB)

    # Get pitcher game logs
    logs = pd.read_sql("""
        SELECT player_id, game_date, ip, home_runs
        FROM pitcher_game_logs
        WHERE ip > 0
        ORDER BY player_id, game_date
    """, db)
    db.close()

    print(f"Pitcher game logs: {len(logs)} starts, {logs['player_id'].nunique()} pitchers")

    # Compute per-start HR/9
    logs['hr9_game'] = logs['home_runs'] / logs['ip'] * 9
    logs['is_blowup'] = (logs['home_runs'] >= 2).astype(int)

    # League-wide priors for shrinkage
    league_mean_hr9 = logs['hr9_game'].mean()
    league_blowup_rate = logs['is_blowup'].mean()
    league_hr9_std = logs.groupby('player_id')['hr9_game'].std().mean()
    PRIOR_STARTS = 8  # shrinkage weight (like 8 prior starts at league avg)

    print(f"League priors: mean_hr9={league_mean_hr9:.3f}, blowup_rate={league_blowup_rate:.3f}, hr9_std={league_hr9_std:.3f}")

    # For each start, compute features from ALL PRIOR starts only
    results = []
    for pid, group in logs.groupby('player_id'):
        group = group.sort_values('game_date').reset_index(drop=True)

        cum_ip = 0.0
        cum_hr = 0
        prior_hr9_values = []
        cum_blowups = 0
        n_prior = 0

        for idx, row in group.iterrows():
            if n_prior >= 3:  # minimum 3 prior starts
                # Raw features from prior starts
                raw_mean_hr9 = cum_hr / cum_ip * 9 if cum_ip > 0 else league_mean_hr9
                raw_blowup_freq = cum_blowups / n_prior
                raw_hr9_std = np.std(prior_hr9_values) if len(prior_hr9_values) > 1 else league_hr9_std
                raw_hr9_cv = raw_hr9_std / raw_mean_hr9 if raw_mean_hr9 > 0 else 1.0

                # Shrinkage: blend with league average by sample size
                shrink_weight = n_prior / (n_prior + PRIOR_STARTS)
                shrunk_mean_hr9 = shrink_weight * raw_mean_hr9 + (1 - shrink_weight) * league_mean_hr9
                shrunk_blowup = shrink_weight * raw_blowup_freq + (1 - shrink_weight) * league_blowup_rate
                shrunk_std = shrink_weight * raw_hr9_std + (1 - shrink_weight) * league_hr9_std
                shrunk_cv = shrunk_std / shrunk_mean_hr9 if shrunk_mean_hr9 > 0 else 1.0

                results.append({
                    'player_id': pid,
                    'game_date': row['game_date'],
                    'n_prior_starts': n_prior,
                    'pit_mean_hr9': shrunk_mean_hr9,
                    'pit_blowup_freq': shrunk_blowup,
                    'pit_hr9_std': shrunk_std,
                    'pit_hr9_cv': shrunk_cv,
                    'raw_mean_hr9': raw_mean_hr9,
                    'raw_blowup_freq': raw_blowup_freq,
                })

            # Update cumulative stats (AFTER recording, so this start is excluded)
            cum_ip += row['ip']
            cum_hr += row['home_runs']
            prior_hr9_values.append(row['hr9_game'])
            cum_blowups += row['is_blowup']
            n_prior += 1

    pit_df = pd.DataFrame(results)
    print(f"PIT features computed: {len(pit_df)} pitcher-starts with 3+ prior starts")
    return pit_df


def build_game_spine():
    """Build MLB game spine with close_total and actual total runs."""
    db = sqlite3.connect(DB)

    lines = pd.read_sql("""
        SELECT game_id, game_date, home_team, away_team, close_total
        FROM betting_lines
        WHERE sport = 'MLB' AND close_total IS NOT NULL
    """, db)

    home = pd.read_sql("""
        SELECT game_id, team as home_team, score as home_score,
               opp_score as away_score, stats_json
        FROM team_stats
        WHERE sport = 'MLB' AND is_home = 1
    """, db)
    db.close()

    games = lines.merge(home, on=['game_id', 'home_team'], how='inner')

    # Filter out unsettled (0-0) games
    games = games[(games['home_score'] > 0) | (games['away_score'] > 0)].copy()

    games['actual_total'] = games['home_score'] + games['away_score']
    games['total_residual'] = games['actual_total'] - games['close_total']
    games['over_hit'] = (games['actual_total'] > games['close_total']).astype(int)

    # Extract SP names
    def get_sp(js, key):
        try:
            d = json.loads(js) if isinstance(js, str) else {}
            return d.get(key)
        except:
            return None

    games['home_sp_name'] = games['stats_json'].apply(lambda x: get_sp(x, 'home_probable_pitcher'))
    games['away_sp_name'] = games['stats_json'].apply(lambda x: get_sp(x, 'away_probable_pitcher'))

    print(f"Game spine: {len(games)} settled MLB games")
    return games


def link_pitchers_to_games(games, pit_df):
    """Link PIT pitcher features to games via pitcher name → player_id mapping."""
    db = sqlite3.connect(DB)
    name_map = pd.read_sql("""
        SELECT DISTINCT pitcher_name, player_id
        FROM pitcher_stats
        WHERE player_id IS NOT NULL
    """, db)
    db.close()

    name_to_id = dict(zip(name_map['pitcher_name'], name_map['player_id']))

    games['home_sp_id'] = games['home_sp_name'].map(name_to_id)
    games['away_sp_id'] = games['away_sp_name'].map(name_to_id)

    matched_home = games['home_sp_id'].notna().sum()
    matched_away = games['away_sp_id'].notna().sum()
    print(f"SP name→id matching: home={matched_home}/{len(games)}, away={matched_away}/{len(games)}")

    # Join PIT features for home SP
    home_pit = pit_df.rename(columns={
        c: f'home_{c}' for c in pit_df.columns if c not in ['player_id', 'game_date']
    })
    home_pit = home_pit.rename(columns={'player_id': 'home_sp_id'})

    away_pit = pit_df.rename(columns={
        c: f'away_{c}' for c in pit_df.columns if c not in ['player_id', 'game_date']
    })
    away_pit = away_pit.rename(columns={'player_id': 'away_sp_id'})

    # Merge on (sp_id, game_date)
    merged = games.merge(home_pit, on=['home_sp_id', 'game_date'], how='left')
    merged = merged.merge(away_pit, on=['away_sp_id', 'game_date'], how='left')

    both_matched = (merged['home_pit_mean_hr9'].notna() & merged['away_pit_mean_hr9'].notna()).sum()
    print(f"Both SPs have PIT features: {both_matched}/{len(merged)} games")

    return merged


def test_signal(games, signal_col, signal_name, n_tests):
    """Test a binary or continuous signal against Over WR and close residual."""
    mask = games[signal_col].notna()
    df = games[mask].copy()

    if df[signal_col].nunique() <= 2:
        # Binary signal
        pos = df[df[signal_col] == 1]
        neg = df[df[signal_col] == 0]

        if len(pos) < 30:
            return {'signal': signal_name, 'N_pos': len(pos), 'status': 'INSUFFICIENT'}

        wr_pos = pos['over_hit'].mean()
        wr_neg = neg['over_hit'].mean()
        be = 0.5238  # -110 breakeven
        edge_pp = (wr_pos - be) * 100

        # Excess runs vs close
        resid_pos = pos['total_residual'].mean()
        resid_neg = neg['total_residual'].mean()
        t_stat, p_val = stats.ttest_ind(pos['total_residual'], neg['total_residual'])

        # Split-half (chronological)
        pos_sorted = pos.sort_values('game_date')
        mid = len(pos_sorted) // 2
        h1_wr = pos_sorted.iloc[:mid]['over_hit'].mean()
        h2_wr = pos_sorted.iloc[mid:]['over_hit'].mean()
        h1_edge = (h1_wr - be) * 100
        h2_edge = (h2_wr - be) * 100

        # Monthly fingerprint
        df_pos = pos.copy()
        df_pos['month'] = pd.to_datetime(df_pos['game_date']).dt.month
        monthly = df_pos.groupby('month')['over_hit'].agg(['mean', 'count'])
        monthly_str = ', '.join([f"M{m}: {r['mean']*100:.1f}% (N={int(r['count'])})"
                                  for m, r in monthly.iterrows()])

        # Bonferroni
        p_bonf = min(p_val * n_tests, 1.0)

        # Conditional on close_total (does edge persist within close bins?)
        conditional_edges = []
        for q_lo, q_hi in [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]:
            lo = pos['close_total'].quantile(q_lo)
            hi = pos['close_total'].quantile(q_hi)
            bin_mask = (pos['close_total'] >= lo) & (pos['close_total'] <= hi)
            if bin_mask.sum() >= 20:
                bin_wr = pos[bin_mask]['over_hit'].mean()
                conditional_edges.append((bin_wr - be) * 100)

        return {
            'signal': signal_name,
            'N_pos': len(pos), 'N_neg': len(neg),
            'wr_pos': wr_pos, 'wr_neg': wr_neg,
            'edge_pp': edge_pp,
            'excess_runs': resid_pos - resid_neg,
            't_stat': t_stat, 'p_raw': p_val, 'p_bonf': p_bonf,
            'split_h1': h1_edge, 'split_h2': h2_edge,
            'split_stable': np.sign(h1_edge) == np.sign(h2_edge) and h1_edge > 0 and h2_edge > 0,
            'monthly': monthly_str,
            'conditional_edges': conditional_edges,
        }
    else:
        # Continuous signal — use top quartile vs bottom quartile
        q75 = df[signal_col].quantile(0.75)
        q25 = df[signal_col].quantile(0.25)
        top = df[df[signal_col] >= q75]
        bot = df[df[signal_col] <= q25]

        wr_top = top['over_hit'].mean()
        wr_bot = bot['over_hit'].mean()
        be = 0.5238
        edge_pp = (wr_top - be) * 100

        resid_top = top['total_residual'].mean()
        resid_bot = bot['total_residual'].mean()

        # Correlation with residual
        r_val, p_val = stats.pearsonr(df[signal_col].astype(float), df['total_residual'].astype(float))
        p_bonf = min(p_val * n_tests, 1.0)

        # Split-half
        df_sorted = df.sort_values('game_date')
        mid = len(df_sorted) // 2
        r1, _ = stats.pearsonr(df_sorted.iloc[:mid][signal_col].astype(float),
                                df_sorted.iloc[:mid]['total_residual'].astype(float))
        r2, _ = stats.pearsonr(df_sorted.iloc[mid:][signal_col].astype(float),
                                df_sorted.iloc[mid:]['total_residual'].astype(float))

        return {
            'signal': signal_name,
            'N_top': len(top), 'N_bot': len(bot),
            'wr_top': wr_top, 'wr_bot': wr_bot,
            'edge_pp': edge_pp,
            'excess_runs_top_vs_bot': resid_top - resid_bot,
            'r_vs_residual': r_val, 'p_raw': p_val, 'p_bonf': p_bonf,
            'split_r1': r1, 'split_r2': r2,
            'split_stable': np.sign(r1) == np.sign(r2),
        }


def run():
    print("=" * 70)
    print("BLOW-UP FREQUENCY / DISPERSION POINT-IN-TIME TEST")
    print("=" * 70)

    # ── Build features ────────────────────────────────────────────────
    print("\n[1] Computing PIT pitcher features...")
    pit_df = build_pit_pitcher_features()

    print("\n[2] Building game spine...")
    games = build_game_spine()

    print("\n[3] Linking pitchers to games...")
    games = link_pitchers_to_games(games, pit_df)

    # Filter to games where both SPs have PIT features
    both_mask = games['home_pit_mean_hr9'].notna() & games['away_pit_mean_hr9'].notna()
    df = games[both_mask].copy()
    print(f"\nGames with both SPs having PIT features: {len(df)}")

    # ── Orthogonality check ───────────────────────────────────────────
    print("\n[4] ORTHOGONALITY CHECK")
    print("Are dispersion features INDEPENDENT of mean, or just 'bad pitcher = volatile'?")

    # Use home SP features for orthogonality (symmetric)
    feat_cols = ['home_pit_mean_hr9', 'home_pit_blowup_freq', 'home_pit_hr9_std', 'home_pit_hr9_cv']
    feat_labels = ['mean_hr9', 'blowup_freq', 'hr9_std', 'hr9_cv']

    print(f"\n  Pairwise correlations (home SP features, N={len(df)}):")
    for i in range(len(feat_cols)):
        for j in range(i+1, len(feat_cols)):
            mask = df[feat_cols[i]].notna() & df[feat_cols[j]].notna()
            r, p = stats.pearsonr(df.loc[mask, feat_cols[i]], df.loc[mask, feat_cols[j]])
            redundant = "REDUNDANT" if abs(r) > 0.70 else ("correlated" if abs(r) > 0.50 else "INDEPENDENT")
            print(f"    {feat_labels[i]} × {feat_labels[j]}: r={r:+.3f} ({redundant})")

    # ── Build interaction signals ─────────────────────────────────────
    print("\n[5] Building interaction signals...")

    # Signal 1: both SPs high blow-up frequency (above median)
    med_blowup = pd.concat([df['home_pit_blowup_freq'], df['away_pit_blowup_freq']]).median()
    df['ix_both_blowup'] = ((df['home_pit_blowup_freq'] > med_blowup) &
                             (df['away_pit_blowup_freq'] > med_blowup)).astype(int)

    # Signal 2: both SPs high mean HR/9 (above median) — PIT version
    med_hr9 = pd.concat([df['home_pit_mean_hr9'], df['away_pit_mean_hr9']]).median()
    df['ix_both_hr9_pit'] = ((df['home_pit_mean_hr9'] > med_hr9) &
                              (df['away_pit_mean_hr9'] > med_hr9)).astype(int)

    # Signal 3: both SPs high dispersion (std above median)
    med_std = pd.concat([df['home_pit_hr9_std'], df['away_pit_hr9_std']]).median()
    df['ix_both_volatile'] = ((df['home_pit_hr9_std'] > med_std) &
                               (df['away_pit_hr9_std'] > med_std)).astype(int)

    # Signal 4: combined blow-up environment (sum of both SPs' blowup freq)
    df['combined_blowup'] = df['home_pit_blowup_freq'] + df['away_pit_blowup_freq']

    # Signal 5: combined mean HR/9 PIT (sum)
    df['combined_hr9_pit'] = df['home_pit_mean_hr9'] + df['away_pit_mean_hr9']

    # Signal 6: combined volatility (sum of std)
    df['combined_vol'] = df['home_pit_hr9_std'] + df['away_pit_hr9_std']

    # Signal 7: blow-up freq MINUS mean HR/9 (the orthogonal part — is freq independent of mean?)
    # Residualize: regress blowup on mean, take residual
    from numpy.linalg import lstsq
    for prefix in ['home', 'away']:
        x = df[f'{prefix}_pit_mean_hr9'].values.reshape(-1, 1)
        x_aug = np.column_stack([x, np.ones(len(x))])
        y = df[f'{prefix}_pit_blowup_freq'].values
        beta, _, _, _ = lstsq(x_aug, y, rcond=None)
        df[f'{prefix}_blowup_resid'] = y - x_aug @ beta

    med_resid = pd.concat([df['home_blowup_resid'], df['away_blowup_resid']]).median()
    df['ix_both_blowup_resid'] = ((df['home_blowup_resid'] > med_resid) &
                                   (df['away_blowup_resid'] > med_resid)).astype(int)

    print(f"  ix_both_blowup: {df['ix_both_blowup'].sum()} games fire (of {len(df)})")
    print(f"  ix_both_hr9_pit: {df['ix_both_hr9_pit'].sum()} games fire")
    print(f"  ix_both_volatile: {df['ix_both_volatile'].sum()} games fire")
    print(f"  ix_both_blowup_resid: {df['ix_both_blowup_resid'].sum()} games fire")

    # ── Test each signal ──────────────────────────────────────────────
    n_tests = 7  # Bonferroni denominator

    signals_to_test = [
        ('ix_both_blowup', 'Both SPs high blow-up freq (binary)'),
        ('ix_both_hr9_pit', 'Both SPs high mean HR/9 PIT (binary)'),
        ('ix_both_volatile', 'Both SPs high HR/9 std (binary)'),
        ('ix_both_blowup_resid', 'Both SPs high blow-up RESIDUAL (binary)'),
        ('combined_blowup', 'Combined blow-up freq (continuous)'),
        ('combined_hr9_pit', 'Combined mean HR/9 PIT (continuous)'),
        ('combined_vol', 'Combined HR/9 volatility (continuous)'),
    ]

    print(f"\n[6] TESTING {n_tests} SIGNALS vs CLOSING TOTAL (Bonferroni k={n_tests})")
    print("=" * 70)

    for col, name in signals_to_test:
        result = test_signal(df, col, name, n_tests)

        print(f"\n  {result['signal']}")
        print(f"  " + "-" * 50)

        if 'status' in result and result['status'] == 'INSUFFICIENT':
            print(f"    INSUFFICIENT DATA: N_pos={result.get('N_pos', 'N/A')}")
            continue

        if 'wr_pos' in result:
            # Binary signal
            print(f"    N: fire={result['N_pos']}, no-fire={result['N_neg']}")
            print(f"    Over WR when fire: {result['wr_pos']*100:.1f}% vs pool {result['wr_neg']*100:.1f}%")
            print(f"    Edge vs BE (-110): {result['edge_pp']:+.1f}pp")
            print(f"    Excess runs (fire-nofire): {result['excess_runs']:+.3f}")
            print(f"    t-stat: {result['t_stat']:.3f}, p_raw: {result['p_raw']:.4f}, p_bonf: {result['p_bonf']:.4f}")
            print(f"    Split-half edge: H1={result['split_h1']:+.1f}pp, H2={result['split_h2']:+.1f}pp {'STABLE' if result['split_stable'] else 'UNSTABLE'}")
            print(f"    Monthly: {result['monthly']}")
            if result.get('conditional_edges'):
                print(f"    Conditional on close_total (within bins): {[f'{e:+.1f}pp' for e in result['conditional_edges']]}")
        else:
            # Continuous signal
            print(f"    N: top_Q={result['N_top']}, bot_Q={result['N_bot']}")
            print(f"    Top Q Over WR: {result['wr_top']*100:.1f}% vs Bot Q {result['wr_bot']*100:.1f}%")
            print(f"    Edge vs BE (-110): {result['edge_pp']:+.1f}pp")
            print(f"    Excess runs (top-bot): {result['excess_runs_top_vs_bot']:+.3f}")
            print(f"    r vs total_residual: {result['r_vs_residual']:+.4f}, p_raw: {result['p_raw']:.4f}, p_bonf: {result['p_bonf']:.4f}")
            print(f"    Split-half r: H1={result['split_r1']:+.4f}, H2={result['split_r2']:+.4f} {'STABLE' if result['split_stable'] else 'UNSTABLE'}")

    # ── MONTHLY FINGERPRINT for key signals ───────────────────────────
    print(f"\n{'='*70}")
    print("MONTHLY FINGERPRINT TEST (flat = lookahead, gradient = real)")
    print("=" * 70)

    df['month'] = pd.to_datetime(df['game_date']).dt.month
    be = 0.5238

    for col, name in [('ix_both_blowup', 'Both SPs blow-up'), ('ix_both_hr9_pit', 'Both SPs HR/9 PIT')]:
        print(f"\n  {name}:")
        fires = df[df[col] == 1]
        for m in sorted(fires['month'].unique()):
            m_data = fires[fires['month'] == m]
            wr = m_data['over_hit'].mean()
            edge = (wr - be) * 100
            print(f"    Month {m}: WR={wr*100:.1f}%, edge={edge:+.1f}pp, N={len(m_data)}")

    # ── VERDICT ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    # Check if any signal passes ALL gates
    survivors = []
    for col, name in signals_to_test:
        result = test_signal(df, col, name, n_tests)
        if 'status' in result:
            continue

        passes_bonf = result.get('p_bonf', 1.0) < 0.05
        passes_edge = result.get('edge_pp', 0) > 0
        passes_split = result.get('split_stable', False)

        if passes_bonf and passes_edge and passes_split:
            survivors.append((col, name, result))

    if survivors:
        print(f"\n  {len(survivors)} SURVIVOR(S):")
        for col, name, r in survivors:
            print(f"    {name}: edge={r.get('edge_pp', 0):+.1f}pp, p_bonf={r.get('p_bonf', 1):.4f}, "
                  f"split=({r.get('split_h1', r.get('split_r1', 0)):+.1f}/{r.get('split_h2', r.get('split_r2', 0)):+.1f})")
        print(f"\n  → Blow-up frequency / dispersion ADDS signal beyond mean HR/9.")
        print(f"  → Proceed to shadow arm with validated feature set.")
    else:
        print(f"""
  ZERO SURVIVORS. Blow-up frequency and dispersion features do NOT
  create a real point-in-time signal. Like mean HR/9, these statistics
  don't predict totals residuals vs the closing line.

  The totals style-mismatch hypothesis is FULLY DEAD:
  - Mean HR/9: died point-in-time (lookahead artifact)
  - Blow-up frequency: dies point-in-time
  - HR/9 dispersion/std: dies point-in-time
  - Residualized blow-up (orthogonal to mean): dies point-in-time

  No shadow_totals_style arm should be built.
""")


if __name__ == '__main__':
    run()
