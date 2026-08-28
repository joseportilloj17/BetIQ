"""
Killed-Signal Stacking Test: vs-Close Residual Analysis
========================================================
For each previously-killed signal, test whether it predicts the
RESIDUAL (actual outcome minus close-line prediction).
Signals that only predict vs naive (already in the close) are dropped.
"""

import sqlite3, json, numpy as np, pandas as pd
from scipy import stats
from collections import defaultdict

DB = '/Users/joseportillo/Downloads/BetIQ/data/historical.db'


def load_game_spine():
    """Build MLB game spine with scores + closing lines."""
    db = sqlite3.connect(DB)

    # games table has game_id, game_date, home_team, away_team, home_score, away_score
    games_df = pd.read_sql("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team,
               g.home_score, g.away_score, g.season,
               b.close_spread, b.close_total,
               b.open_spread, b.open_total
        FROM games g
        JOIN betting_lines b ON g.game_id = b.game_id
        WHERE g.sport = 'MLB'
          AND g.status = 'Final'
          AND b.close_spread IS NOT NULL
          AND b.close_total IS NOT NULL
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
    """, db)

    # Extract SP names from team_stats stats_json (home side only)
    sp_df = pd.read_sql("""
        SELECT game_id, stats_json
        FROM team_stats
        WHERE sport = 'MLB' AND is_home = 1
    """, db)

    def get_field(json_str, key):
        try:
            d = json.loads(json_str) if isinstance(json_str, str) else {}
            return d.get(key, None)
        except:
            return None

    sp_df['home_sp'] = sp_df['stats_json'].apply(lambda x: get_field(x, 'home_probable_pitcher'))
    sp_df['away_sp'] = sp_df['stats_json'].apply(lambda x: get_field(x, 'away_probable_pitcher'))
    sp_df = sp_df[['game_id', 'home_sp', 'away_sp']]

    games = games_df.merge(sp_df, on='game_id', how='left')

    # Compute residuals
    # close_spread convention from this DB: negative = home favorite (home -1.5 etc.)
    # Market prediction: home margin = -close_spread
    # Actual margin: home_score - away_score
    games['actual_margin'] = games['home_score'] - games['away_score']
    games['spread_residual'] = games['actual_margin'] + games['close_spread']
    # spread_residual > 0 means home outperformed vs market close

    games['actual_total'] = games['home_score'] + games['away_score']
    games['total_residual'] = games['actual_total'] - games['close_total']
    # total_residual > 0 means game went over vs close

    # Season as int for joins
    games['season_int'] = games['season'].astype(int)

    print(f"Game spine: {len(games)} MLB games with close lines + scores")
    print(f"  Date range: {games['game_date'].min()} to {games['game_date'].max()}")
    print(f"  Spread residual: mean={games['spread_residual'].mean():.3f}, std={games['spread_residual'].std():.3f}")
    print(f"  Total residual:  mean={games['total_residual'].mean():.3f}, std={games['total_residual'].std():.3f}")
    print(f"  SP name coverage: home={games['home_sp'].notna().sum()}, away={games['away_sp'].notna().sum()}")

    db.close()
    return games


def attach_weather(games):
    """Join game_weather data."""
    db = sqlite3.connect(DB)
    weather = pd.read_sql("""
        SELECT game_date, home_team, temperature_f, wind_speed_mph,
               wind_direction_deg, is_dome
        FROM game_weather
    """, db)
    db.close()

    merged = games.merge(weather, on=['game_date', 'home_team'], how='left')
    n_weather = merged['temperature_f'].notna().sum()
    print(f"  Weather matched: {n_weather}/{len(games)} games")
    return merged


def attach_umpire(games):
    """Join game_umpires data."""
    db = sqlite3.connect(DB)
    umps = pd.read_sql("""
        SELECT game_date, home_team,
               ump_career_rpg_above_avg, ump_season_rpg_above_avg,
               ump_career_games
        FROM game_umpires
    """, db)
    db.close()

    merged = games.merge(umps, on=['game_date', 'home_team'], how='left')
    n_ump = merged['ump_career_rpg_above_avg'].notna().sum()
    print(f"  Umpire matched: {n_ump}/{len(games)} games")
    return merged


def attach_batting_statcast(games):
    """Join season-level batting Statcast (barrel%, hard_hit%, xwoba)."""
    db = sqlite3.connect(DB)
    batting = pd.read_sql("""
        SELECT team_name as team, CAST(season AS INTEGER) as season,
               barrel_pct, hard_hit_pct, xwoba, babip
        FROM team_batting_stats
    """, db)
    db.close()

    # Join for home and away
    home_bat = batting.rename(columns={c: f'home_{c}' for c in batting.columns if c not in ['team', 'season']})
    home_bat = home_bat.rename(columns={'team': 'home_team'})
    away_bat = batting.rename(columns={c: f'away_{c}' for c in batting.columns if c not in ['team', 'season']})
    away_bat = away_bat.rename(columns={'team': 'away_team'})

    merged = games.merge(home_bat, left_on=['home_team', 'season_int'], right_on=['home_team', 'season'], how='left')
    # season col from batting overlaps — drop the extra one
    if 'season_y' in merged.columns:
        merged = merged.drop(columns=[c for c in merged.columns if c in ['season_x', 'season_y']])
    merged = merged.merge(away_bat, left_on=['away_team', 'season_int'], right_on=['away_team', 'season'], how='left')
    if 'season_y' in merged.columns:
        merged = merged.drop(columns=[c for c in merged.columns if c in ['season_x', 'season_y']])

    # Compute differentials
    for stat in ['barrel_pct', 'hard_hit_pct', 'xwoba', 'babip']:
        merged[f'{stat}_diff'] = merged[f'home_{stat}'] - merged[f'away_{stat}']

    n_bat = merged['home_barrel_pct'].notna().sum()
    print(f"  Batting Statcast matched: {n_bat}/{len(games)} games")
    return merged


def attach_pitcher_stats(games):
    """Join season-level SP stats (FIP, HR/9, BB/9). fip=99 is a sentinel for missing."""
    db = sqlite3.connect(DB)
    pitchers = pd.read_sql("""
        SELECT pitcher_name, CAST(season AS INTEGER) as season,
               fip, hr9, bb9, k9, era, whip, ip, gs
        FROM pitcher_stats
        WHERE gs >= 5
    """, db)
    db.close()

    # fip=99 is a sentinel value — null it out
    pitchers.loc[pitchers['fip'] == 99.0, 'fip'] = np.nan

    # Join on SP name + season
    home_sp = pitchers.rename(columns={c: f'home_sp_{c}' for c in pitchers.columns if c not in ['pitcher_name', 'season']})
    home_sp = home_sp.rename(columns={'pitcher_name': 'home_sp'})
    away_sp = pitchers.rename(columns={c: f'away_sp_{c}' for c in pitchers.columns if c not in ['pitcher_name', 'season']})
    away_sp = away_sp.rename(columns={'pitcher_name': 'away_sp'})

    merged = games.merge(home_sp, left_on=['home_sp', 'season_int'], right_on=['home_sp', 'season'], how='left')
    if 'season_y' in merged.columns:
        merged = merged.drop(columns=[c for c in merged.columns if c in ['season_x', 'season_y']])
    merged = merged.merge(away_sp, left_on=['away_sp', 'season_int'], right_on=['away_sp', 'season'], how='left')
    if 'season_y' in merged.columns:
        merged = merged.drop(columns=[c for c in merged.columns if c in ['season_x', 'season_y']])

    # Compute differentials (lower FIP = better home pitcher → positive diff is bad for home)
    for stat in ['fip', 'hr9', 'bb9', 'era']:
        home_col = f'home_sp_{stat}'
        away_col = f'away_sp_{stat}'
        if home_col in merged.columns and away_col in merged.columns:
            merged[f'sp_{stat}_diff'] = merged[home_col] - merged[away_col]
        else:
            merged[f'sp_{stat}_diff'] = np.nan

    n_sp = merged['home_sp_fip'].notna().sum() if 'home_sp_fip' in merged.columns else 0
    print(f"  SP stats matched (non-sentinel FIP): {n_sp}/{len(games)} games")
    return merged


def attach_schedule_signals(games):
    """Compute schedule-derived signals: back-to-back, getaway, rest days."""
    games = games.sort_values('game_date').reset_index(drop=True)

    dates_by_team = defaultdict(set)
    for _, row in games.iterrows():
        dates_by_team[row['home_team']].add(row['game_date'])
        dates_by_team[row['away_team']].add(row['game_date'])

    def is_btb(team, game_date):
        from datetime import datetime, timedelta
        dt = datetime.strptime(game_date, '%Y-%m-%d')
        yesterday = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
        return 1 if yesterday in dates_by_team[team] else 0

    def plays_tomorrow(team, game_date):
        from datetime import datetime, timedelta
        dt = datetime.strptime(game_date, '%Y-%m-%d')
        tomorrow = (dt + timedelta(days=1)).strftime('%Y-%m-%d')
        return 1 if tomorrow in dates_by_team[team] else 0

    def rest_days(team, game_date):
        from datetime import datetime, timedelta
        dt = datetime.strptime(game_date, '%Y-%m-%d')
        team_dates = sorted(dates_by_team[team])
        idx = None
        for i, d in enumerate(team_dates):
            if d == game_date:
                idx = i
                break
        if idx is None or idx == 0:
            return 1  # default
        prev = datetime.strptime(team_dates[idx - 1], '%Y-%m-%d')
        return (dt - prev).days

    games['home_btb'] = games.apply(lambda r: is_btb(r['home_team'], r['game_date']), axis=1)
    games['away_btb'] = games.apply(lambda r: is_btb(r['away_team'], r['game_date']), axis=1)
    games['btb_diff'] = games['home_btb'] - games['away_btb']

    games['home_getaway'] = games.apply(
        lambda r: 1 if is_btb(r['home_team'], r['game_date']) and plays_tomorrow(r['home_team'], r['game_date']) else 0,
        axis=1)
    games['away_getaway'] = games.apply(
        lambda r: 1 if is_btb(r['away_team'], r['game_date']) and plays_tomorrow(r['away_team'], r['game_date']) else 0,
        axis=1)
    games['getaway_diff'] = games['home_getaway'] - games['away_getaway']

    games['home_rest'] = games.apply(lambda r: rest_days(r['home_team'], r['game_date']), axis=1)
    games['away_rest'] = games.apply(lambda r: rest_days(r['away_team'], r['game_date']), axis=1)
    games['rest_diff'] = games['home_rest'] - games['away_rest']

    print(f"  Schedule signals computed for {len(games)} games")
    btb_dist = games['btb_diff'].value_counts().to_dict()
    print(f"    B2B diff distribution: {btb_dist}")
    print(f"    Rest diff: mean={games['rest_diff'].mean():.3f}, std={games['rest_diff'].std():.3f}")

    return games


def attach_line_movement(games):
    """Compute line movement: close_spread - open_spread."""
    # open_spread/open_total were already joined in load_game_spine via betting_lines
    games = games.copy()
    games['spread_movement'] = games['close_spread'] - games['open_spread']
    # positive = home became more favored at close vs open
    games['total_movement'] = games['close_total'] - games['open_total']

    n_move = games['spread_movement'].notna().sum()
    print(f"  Line movement available: {n_move}/{len(games)} games")
    return games


def test_vs_close_residual(games, signal_col, residual_col, signal_name, n_signals):
    """
    Test if a signal predicts the CLOSE RESIDUAL.
    Returns dict with correlation, p-value, and Bonferroni adjustment.
    """
    mask = games[signal_col].notna() & games[residual_col].notna()
    N = mask.sum()
    if N < 50:
        return {'signal': signal_name, 'N': N, 'status': 'INSUFFICIENT_DATA'}

    x = games.loc[mask, signal_col].values.astype(float)
    y = games.loc[mask, residual_col].values.astype(float)

    if np.std(x) < 1e-10:
        return {'signal': signal_name, 'N': N, 'status': 'NO_VARIANCE'}

    r, p = stats.pearsonr(x, y)

    # Effect size: runs of residual per 1 SD of signal
    slope = r * np.std(y) / np.std(x)

    # Split-half stability (temporal: sort by index, which is already date-sorted)
    mid = N // 2
    r1, _ = stats.pearsonr(x[:mid], y[:mid])
    r2, _ = stats.pearsonr(x[mid:], y[mid:])

    p_bonf = min(p * n_signals, 1.0)

    return {
        'signal': signal_name,
        'residual': residual_col,
        'N': N,
        'r': r,
        'p_raw': p,
        'p_bonferroni': p_bonf,
        'slope_runs_per_sd': slope,
        'r_first_half': r1,
        'r_second_half': r2,
        'split_half_same_sign': np.sign(r1) == np.sign(r2),
        'verdict': 'VS_CLOSE_RESIDUAL' if p_bonf < 0.05 and np.sign(r1) == np.sign(r2) else 'ALREADY_PRICED'
    }


def run_analysis():
    print("=" * 70)
    print("KILLED-SIGNAL VS-CLOSE RESIDUAL ANALYSIS")
    print("=" * 70)

    # ── Step 1: Build game spine ──────────────────────────────────────────
    print("\n[1] Building game spine...")
    games = load_game_spine()

    # ── Step 2: Attach all killed signals ─────────────────────────────────
    print("\n[2] Attaching killed signals...")
    games = attach_weather(games)
    games = attach_umpire(games)
    games = attach_batting_statcast(games)
    games = attach_pitcher_stats(games)
    games = attach_schedule_signals(games)
    games = attach_line_movement(games)

    # ── Step 3: Define all killed signals to test ─────────────────────────
    # Each entry: (column_name, residual_to_test, human_name, kill_reason)
    signals = [
        # Weather (killed: noise in NegBin, market prices dome/outdoor)
        ('temperature_f',        'total_residual',  'Temperature → Totals',        'Noise in NegBin'),
        ('temperature_f',        'spread_residual',  'Temperature → Spread',        'Noise in NegBin'),
        ('wind_speed_mph',       'total_residual',  'Wind Speed → Totals',         'Noise in NegBin'),
        ('wind_speed_mph',       'spread_residual', 'Wind Speed → Spread',         'Noise in NegBin'),
        ('is_dome',              'total_residual',  'Dome → Totals',               'Noise in NegBin'),

        # Umpire (killed: public game-state, already priced)
        ('ump_career_rpg_above_avg', 'total_residual',  'Ump Career RPG → Totals',  'Public, already priced'),
        ('ump_career_rpg_above_avg', 'spread_residual', 'Ump Career RPG → Spread',  'Public, already priced'),
        ('ump_season_rpg_above_avg', 'total_residual',  'Ump Season RPG → Totals',  'Public, already priced'),

        # Batting Statcast diffs (killed: redundant with ops_diff, r>0.47)
        ('barrel_pct_diff',      'spread_residual', 'Barrel% Diff → Spread',       'Redundant w/ ops_diff'),
        ('barrel_pct_diff',      'total_residual',  'Barrel% Diff → Totals',       'Redundant w/ ops_diff'),
        ('hard_hit_pct_diff',    'spread_residual', 'Hard-Hit% Diff → Spread',     'Redundant w/ ops_diff'),
        ('hard_hit_pct_diff',    'total_residual',  'Hard-Hit% Diff → Totals',     'Redundant w/ ops_diff'),
        ('xwoba_diff',           'spread_residual', 'xwOBA Diff → Spread',         'Redundant w/ ops_diff'),
        ('xwoba_diff',           'total_residual',  'xwOBA Diff → Totals',         'Redundant w/ ops_diff'),
        ('babip_diff',           'spread_residual', 'BABIP Diff → Spread',         'Process stat'),
        ('babip_diff',           'total_residual',  'BABIP Diff → Totals',         'Process stat'),

        # SP stats diffs (killed: FIP margin-inert, K/9 dead weight)
        ('sp_fip_diff',          'spread_residual', 'SP FIP Diff → Spread',        'Margin-inert'),
        ('sp_fip_diff',          'total_residual',  'SP FIP Diff → Totals',        'Margin-inert'),
        ('sp_hr9_diff',          'total_residual',  'SP HR/9 Diff → Totals',       'Part of FIP family'),
        ('sp_bb9_diff',          'spread_residual', 'SP BB/9 Diff → Spread',       'Part of FIP family'),
        ('sp_era_diff',          'spread_residual', 'SP ERA Diff → Spread',        'In model but test residual'),
        ('sp_era_diff',          'total_residual',  'SP ERA Diff → Totals',        'In model but test residual'),

        # Schedule (killed: already priced into spread)
        ('btb_diff',             'spread_residual', 'B2B Diff → Spread',           'Already priced'),
        ('btb_diff',             'total_residual',  'B2B Diff → Totals',           'Already priced'),
        ('rest_diff',            'spread_residual', 'Rest Days Diff → Spread',     'Already priced'),
        ('rest_diff',            'total_residual',  'Rest Days Diff → Totals',     'Already priced'),
        ('getaway_diff',         'spread_residual', 'Getaway Diff → Spread',       '-0.22 runs, already priced'),
        ('getaway_diff',         'total_residual',  'Getaway Diff → Totals',       '-0.22 runs'),

        # Line movement (killed: RLM = 0.500 exactly)
        ('spread_movement',      'spread_residual', 'Spread Movement → Spread',    'RLM = 0.500'),
        ('total_movement',       'total_residual',  'Total Movement → Totals',     'Not previously tested vs close'),
    ]

    n_signals = len(signals)

    # ── Step 4: Test each signal vs close residual ────────────────────────
    print(f"\n[3] Testing {n_signals} killed signals vs CLOSE RESIDUAL (Bonferroni k={n_signals})")
    print("-" * 70)

    results = []
    for col, resid, name, reason in signals:
        if col not in games.columns:
            print(f"  SKIP: {name} — column '{col}' not available")
            continue
        result = test_vs_close_residual(games, col, resid, name, n_signals)
        result['kill_reason'] = reason
        results.append(result)

    # ── Step 5: Report ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"VS-CLOSE RESIDUAL RESULTS")
    print(f"{'=' * 70}")

    valid = [r for r in results if 'r' in r]
    valid.sort(key=lambda x: abs(x['r']), reverse=True)

    survivors = []

    print(f"\n{'Signal':<35} {'N':>6} {'r':>7} {'p_raw':>10} {'p_bonf':>10} {'runs/SD':>8} {'Split':>6} {'Verdict'}")
    print("-" * 105)
    for r in valid:
        split_ok = 'Y' if r.get('split_half_same_sign') else 'N'
        print(f"{r['signal']:<35} {r['N']:>6} {r['r']:>+7.4f} {r['p_raw']:>10.4f} {r['p_bonferroni']:>10.4f} "
              f"{r['slope_runs_per_sd']:>+8.3f} {split_ok:>6} {r['verdict']}")
        if r['verdict'] == 'VS_CLOSE_RESIDUAL':
            survivors.append(r)

    missing = [r for r in results if 'r' not in r]
    if missing:
        print(f"\n  Skipped ({len(missing)}): {[r['signal'] for r in missing]}")

    print(f"\n{'=' * 70}")
    print(f"VS-CLOSE FILTER SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Signals tested: {len(valid)}")
    print(f"  Survived vs-close filter (Bonferroni p<0.05 + split-half same sign): {len(survivors)}")
    print(f"  Killed (already priced by close): {len(valid) - len(survivors)}")

    if survivors:
        print(f"\n  SURVIVORS:")
        for s in survivors:
            print(f"    {s['signal']}: r={s['r']:+.4f}, p_bonf={s['p_bonferroni']:.4f}, "
                  f"slope={s['slope_runs_per_sd']:+.3f} runs/SD, "
                  f"split=({s['r_first_half']:+.3f}/{s['r_second_half']:+.3f})")
    else:
        print(f"\n  *** NO SURVIVORS — every killed signal is already priced by the close ***")
        print(f"  This is the honest null: stacking priced signals gains nothing.")
        print(f"  The close incorporates all of: weather, umpire, Statcast, SP stats,")
        print(f"  schedule, and line movement. There is no hidden residual to stack.")

    # ── Step 6: If survivors exist, check mechanism independence + stack ──
    stack_signals = []  # define early so final verdict can reference it

    if len(survivors) >= 2:
        print(f"\n{'=' * 70}")
        print(f"MECHANISM INDEPENDENCE + STACKING")
        print(f"{'=' * 70}")

        mechanisms = defaultdict(list)
        for s in survivors:
            name = s['signal']
            if 'Temp' in name or 'Wind' in name or 'Dome' in name:
                mechanisms['weather'].append(s)
            elif 'Ump' in name:
                mechanisms['umpire'].append(s)
            elif 'Barrel' in name or 'Hard-Hit' in name or 'xwOBA' in name or 'BABIP' in name:
                mechanisms['batting_statcast'].append(s)
            elif 'SP' in name or 'FIP' in name or 'ERA' in name:
                mechanisms['sp_quality'].append(s)
            elif 'B2B' in name or 'Rest' in name or 'Getaway' in name:
                mechanisms['schedule'].append(s)
            elif 'Movement' in name:
                mechanisms['line_movement'].append(s)
            else:
                mechanisms['other'].append(s)

        print(f"\n  Survivors by mechanism:")
        for mech, sigs in mechanisms.items():
            print(f"    {mech}: {[s['signal'] for s in sigs]}")

        for mech, sigs in mechanisms.items():
            best = max(sigs, key=lambda x: abs(x['r']))
            stack_signals.append((best, mech))
            print(f"\n  Selected from {mech}: {best['signal']} (r={best['r']:+.4f})")

        if len(stack_signals) >= 2:
            print(f"\n  Orthogonality matrix (pairwise r):")
            name_to_col = {name: col for col, _, name, _ in signals}

            for i, (si, mi) in enumerate(stack_signals):
                for j, (sj, mj) in enumerate(stack_signals):
                    if j <= i:
                        continue
                    ci = name_to_col.get(si['signal'])
                    cj = name_to_col.get(sj['signal'])
                    if ci and cj:
                        mask = games[ci].notna() & games[cj].notna()
                        if mask.sum() > 50:
                            r_ij, _ = stats.pearsonr(
                                games.loc[mask, ci].astype(float),
                                games.loc[mask, cj].astype(float)
                            )
                            print(f"    {si['signal']} × {sj['signal']}: r={r_ij:+.4f}")

            print(f"\n  OOS STACK TEST (train on first half, test on second half):")

            resid_col = survivors[0]['residual']
            name_to_col = {name: col for col, _, name, _ in signals}
            feature_cols = [name_to_col[s[0]['signal']] for s in stack_signals if name_to_col.get(s[0]['signal'])]
            mask = games[feature_cols + [resid_col]].notna().all(axis=1)
            df = games.loc[mask].copy()

            if len(df) < 200:
                print(f"    Insufficient data for OOS test: {len(df)} games")
            else:
                df = df.sort_values('game_date').reset_index(drop=True)
                n = len(df)
                mid = n // 2
                train = df.iloc[:mid]
                test = df.iloc[mid:]

                X_train = train[feature_cols].astype(float)
                y_train = train[resid_col].astype(float)
                X_test = test[feature_cols].astype(float)
                y_test = test[resid_col].astype(float)

                means = X_train.mean()
                stds = X_train.std().replace(0, 1)
                X_train_z = (X_train - means) / stds
                X_test_z = (X_test - means) / stds

                from numpy.linalg import lstsq
                X_tr = np.column_stack([X_train_z.values, np.ones(len(X_train_z))])
                X_te = np.column_stack([X_test_z.values, np.ones(len(X_test_z))])

                beta, _, _, _ = lstsq(X_tr, y_train.values, rcond=None)
                pred_train = X_tr @ beta
                pred_test = X_te @ beta

                r_train, p_train = stats.pearsonr(pred_train, y_train.values)
                r_test, p_test = stats.pearsonr(pred_test, y_test.values)

                print(f"    Train: r={r_train:+.4f}, p={p_train:.4f} (N={len(train)})")
                print(f"    Test:  r={r_test:+.4f}, p={p_test:.4f} (N={len(test)})")
                print(f"    Weights: {dict(zip(feature_cols, beta[:-1].round(4)))}")
                print(f"    Intercept: {beta[-1]:.4f}")

                threshold = 0.05 / len(stack_signals)
                if p_test < threshold and np.sign(r_train) == np.sign(r_test):
                    print(f"\n    *** STACK SURVIVES OOS — genuine compounding residual ***")
                    q75 = np.percentile(pred_test, 75)
                    q25 = np.percentile(pred_test, 25)
                    top_resid = test.loc[pred_test >= q75, resid_col].mean()
                    bot_resid = test.loc[pred_test <= q25, resid_col].mean()
                    print(f"    Top quartile avg residual: {top_resid:+.3f} runs")
                    print(f"    Bottom quartile avg residual: {bot_resid:+.3f} runs")
                    print(f"    Spread: {top_resid - bot_resid:.3f} runs")
                else:
                    print(f"\n    Stack DOES NOT survive OOS — no compounding edge")
                    print(f"    (r_test={r_test:+.4f}, p_test={p_test:.4f}, need p<{threshold:.4f})")

    elif len(survivors) == 1:
        print(f"\n  Only 1 survivor — cannot test stacking (need >= 2 from different mechanisms)")

    # ── Final Verdict ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FINAL VERDICT")
    print(f"{'=' * 70}")

    if not survivors:
        print("""
  HONEST NULL: Zero killed signals have a genuine vs-close residual.

  Every signal that was killed individually was killed BECAUSE the closing
  line already prices it. Stacking signals that are individually priced
  re-derives the close — it gains nothing.

  The closing line is efficient across ALL tested dimensions:
    - Weather (temperature, wind, dome)
    - Umpire tendencies
    - Batting Statcast (barrel%, hard-hit%, xwOBA, BABIP)
    - SP quality metrics (FIP, HR/9, BB/9, ERA)
    - Schedule context (back-to-back, rest days, getaway)
    - Line movement (open->close spread/total movement)

  The user's hypothesis (orthogonal killed signals can compound) is
  FALSIFIED: the signals are not killed because they are individually
  sub-vig — they are killed because the market already prices them.
  Stacking priced signals gains nothing.
""")
    elif len(survivors) == 1:
        s = survivors[0]
        print(f"""
  ONE SURVIVOR: {s['signal']}
    r = {s['r']:+.4f}, Bonferroni p = {s['p_bonferroni']:.4f}
    Effect: {s['slope_runs_per_sd']:+.3f} runs per SD
    Split-half: {s['r_first_half']:+.3f} / {s['r_second_half']:+.3f}

  Cannot stack a single signal. Worth investigating as a standalone
  filter if effect size is large enough to clear vig.
""")
    else:
        n_mechs = len(set(s[1] for s in stack_signals))
        print(f"""
  {len(survivors)} signals survived vs-close filter from {n_mechs} mechanisms.
  See OOS stack test above for whether they compound.
""")


if __name__ == '__main__':
    run_analysis()
