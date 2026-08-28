"""
4-Way HR Matchup Test: Batter+Pitcher Interaction on Spreads
=============================================================
Tests whether the NET batting-power-vs-opposing-pitching-HR
mismatch predicts who covers the spread, point-in-time.
"""
import sqlite3, json, numpy as np, pandas as pd
from scipy import stats
from collections import defaultdict

DB = '/Users/joseportillo/Downloads/BetIQ/data/historical.db'

def discover_schema():
    """Check available columns in key tables."""
    db = sqlite3.connect(DB)
    cur = db.cursor()
    for t in ['team_game_batting', 'pitcher_game_logs']:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [(r[1], r[2]) for r in cur.fetchall()]
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        print(f"{t} ({n} rows): {cols}")
        cur.execute(f"SELECT * FROM {t} LIMIT 2")
        for row in cur.fetchall():
            print(f"  sample: {row}")
    db.close()


def build_pit_sp_hr9():
    """
    Point-in-time SP HR/9: for each start, cumulative HR/9 from ALL prior starts only.
    Uses shift-1 pattern. Minimum 3 prior starts.
    Shrunk toward league mean by sample size.
    """
    db = sqlite3.connect(DB)
    logs = pd.read_sql("""
        SELECT player_id, game_date, ip, home_runs
        FROM pitcher_game_logs
        WHERE ip > 0
        ORDER BY player_id, game_date
    """, db)
    db.close()

    league_hr9 = (logs['home_runs'].sum() / logs['ip'].sum()) * 9
    PRIOR_N = 8  # shrinkage weight

    results = []
    for pid, grp in logs.groupby('player_id'):
        grp = grp.sort_values('game_date').reset_index(drop=True)
        cum_ip = 0.0
        cum_hr = 0
        n_prior = 0
        for _, row in grp.iterrows():
            if n_prior >= 3:
                raw_hr9 = (cum_hr / cum_ip * 9) if cum_ip > 0 else league_hr9
                shrink = n_prior / (n_prior + PRIOR_N)
                pit_hr9 = shrink * raw_hr9 + (1 - shrink) * league_hr9
                results.append({
                    'player_id': pid,
                    'game_date': row['game_date'],
                    'sp_pit_hr9': pit_hr9,
                    'n_prior': n_prior,
                })
            cum_ip += row['ip']
            cum_hr += row['home_runs']
            n_prior += 1

    sp_df = pd.DataFrame(results)
    print(f"PIT SP HR/9: {len(sp_df)} pitcher-starts with 3+ prior starts, "
          f"{sp_df['player_id'].nunique()} pitchers")
    return sp_df


def build_pit_team_power():
    """
    Point-in-time team batting power: cumulative HR/game or HR/AB from prior games.
    Check what's available in team_game_batting.
    """
    db = sqlite3.connect(DB)

    # Check schema
    cur = db.cursor()
    cur.execute("PRAGMA table_info(team_game_batting)")
    cols = [r[1] for r in cur.fetchall()]
    print(f"team_game_batting columns: {cols}")

    # Try to get HR data — column might be 'home_runs', 'hr', or in a different format
    hr_col = None
    for candidate in ['home_runs', 'hr', 'HR', 'homeRuns']:
        if candidate in cols:
            hr_col = candidate
            break

    ab_col = None
    for candidate in ['at_bats', 'ab', 'AB', 'atBats']:
        if candidate in cols:
            ab_col = candidate
            break

    h_col = None
    for candidate in ['hits', 'h', 'H']:
        if candidate in cols:
            h_col = candidate
            break

    slg_col = None
    for candidate in ['slg', 'SLG', 'slugging']:
        if candidate in cols:
            slg_col = candidate
            break

    iso_col = None
    for candidate in ['iso', 'ISO']:
        if candidate in cols:
            iso_col = candidate
            break

    print(f"  Found: hr={hr_col}, ab={ab_col}, h={h_col}, slg={slg_col}, iso={iso_col}")

    # Determine best power proxy
    # Priority: ISO (isolated power = SLG - AVG), then HR/AB, then HR/game

    # First check if we have the key column for game_date and team
    team_col = None
    for c in ['team', 'team_name', 'team_abbr']:
        if c in cols:
            team_col = c
            break

    date_col = None
    for c in ['game_date', 'date']:
        if c in cols:
            date_col = c
            break

    is_home_col = None
    for c in ['is_home', 'home']:
        if c in cols:
            is_home_col = c
            break

    print(f"  Keys: team={team_col}, date={date_col}, is_home={is_home_col}")

    if hr_col is None:
        print("  WARNING: No HR column found in team_game_batting!")
        # Fallback to season stats
        bat = pd.read_sql("SELECT team_name as team, season, hr, slg, barrel_pct, hard_hit_pct FROM team_batting_stats", db)
        db.close()
        print(f"  FALLING BACK to season batting stats (LOOKAHEAD for individual games)")
        print(f"  Will flag this in results")
        return bat, 'season_fallback'

    # Build per-game batting with HR
    query = f"""
        SELECT {date_col} as game_date, {team_col} as team,
               {hr_col} as hr
    """
    if ab_col:
        query += f", {ab_col} as ab"
    if slg_col:
        query += f", {slg_col} as slg"
    if h_col:
        query += f", {h_col} as h"

    query += f" FROM team_game_batting ORDER BY {team_col}, {date_col}"

    bat = pd.read_sql(query, db)
    db.close()

    print(f"Per-game batting: {len(bat)} team-games")

    # Compute cumulative PIT power stats
    league_hr_per_game = bat['hr'].mean()
    PRIOR_GAMES = 15  # shrinkage weight for team batting

    results = []
    for team, grp in bat.groupby('team'):
        grp = grp.sort_values('game_date').reset_index(drop=True)
        cum_hr = 0
        n_prior = 0

        for _, row in grp.iterrows():
            if n_prior >= 5:  # min 5 prior games
                raw_hr_pg = cum_hr / n_prior
                shrink = n_prior / (n_prior + PRIOR_GAMES)
                pit_hr_pg = shrink * raw_hr_pg + (1 - shrink) * league_hr_per_game
                results.append({
                    'team': team,
                    'game_date': row['game_date'],
                    'bat_pit_hr_pg': pit_hr_pg,
                    'n_prior_games': n_prior,
                })
            cum_hr += row['hr']
            n_prior += 1

    pit_bat = pd.DataFrame(results)
    print(f"PIT team batting power: {len(pit_bat)} team-dates, {pit_bat['team'].nunique()} teams")
    return pit_bat, 'pit'


def build_game_spine():
    """MLB games with close_spread and actual scores."""
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
    # spread_residual > 0 = home outperformed close
    games['home_covered'] = (games['spread_residual'] > 0).astype(int)

    # Extract SP names
    def get_sp(js, key):
        try:
            d = json.loads(js) if isinstance(js, str) else {}
            return d.get(key)
        except:
            return None

    games['home_sp_name'] = games['stats_json'].apply(lambda x: get_sp(x, 'home_probable_pitcher'))
    games['away_sp_name'] = games['stats_json'].apply(lambda x: get_sp(x, 'away_probable_pitcher'))

    print(f"Game spine: {len(games)} settled MLB games with close_spread")
    return games


def run():
    print("=" * 70)
    print("4-WAY HR MATCHUP TEST: Batter+Pitcher Interaction on SPREADS")
    print("=" * 70)

    # Schema discovery
    print("\n[0] Schema discovery...")
    discover_schema()

    # ── Build features ────────────────────────────────────────────────
    print("\n[1] Building PIT SP HR/9...")
    sp_df = build_pit_sp_hr9()

    print("\n[2] Building PIT team batting power...")
    bat_result = build_pit_team_power()
    bat_df, bat_type = bat_result

    print("\n[3] Building game spine...")
    games = build_game_spine()

    # ── Link SP features ──────────────────────────────────────────────
    print("\n[4] Linking features to games...")
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

    # Join SP PIT HR/9
    home_sp = sp_df.rename(columns={'player_id': 'home_sp_id', 'sp_pit_hr9': 'home_sp_hr9'})
    away_sp = sp_df.rename(columns={'player_id': 'away_sp_id', 'sp_pit_hr9': 'away_sp_hr9'})
    games = games.merge(home_sp[['home_sp_id', 'game_date', 'home_sp_hr9']],
                         on=['home_sp_id', 'game_date'], how='left')
    games = games.merge(away_sp[['away_sp_id', 'game_date', 'away_sp_hr9']],
                         on=['away_sp_id', 'game_date'], how='left')

    # Join batting power
    if bat_type == 'pit':
        home_bat = bat_df.rename(columns={'team': 'home_team', 'bat_pit_hr_pg': 'home_bat_power'})
        away_bat = bat_df.rename(columns={'team': 'away_team', 'bat_pit_hr_pg': 'away_bat_power'})
        games = games.merge(home_bat[['home_team', 'game_date', 'home_bat_power']],
                             on=['home_team', 'game_date'], how='left')
        games = games.merge(away_bat[['away_team', 'game_date', 'away_bat_power']],
                             on=['away_team', 'game_date'], how='left')
    else:
        # Season fallback (flag as lookahead)
        games['season'] = games['game_date'].str[:4].astype(int)
        home_bat = bat_df.rename(columns={'team': 'home_team', 'season': 'season', 'hr': 'home_bat_hr_season'})
        away_bat = bat_df.rename(columns={'team': 'away_team', 'season': 'season', 'hr': 'away_bat_hr_season'})
        games = games.merge(home_bat[['home_team', 'season', 'home_bat_hr_season']],
                             on=['home_team', 'season'], how='left')
        games = games.merge(away_bat[['away_team', 'season', 'away_bat_hr_season']],
                             on=['away_team', 'season'], how='left')
        # Normalize season HR to per-game
        games['home_bat_power'] = games['home_bat_hr_season'] / 162  # approx
        games['away_bat_power'] = games['away_bat_hr_season'] / 162
        print("  WARNING: Using season batting (lookahead). Will note in results.")

    # ── Compute 4-way interaction ─────────────────────────────────────
    print("\n[5] Computing 4-way HR mismatch...")

    # Filter to games with all features
    mask = (games['home_sp_hr9'].notna() & games['away_sp_hr9'].notna() &
            games['home_bat_power'].notna() & games['away_bat_power'].notna())
    df = games[mask].copy()
    print(f"  Games with all 4 features: {len(df)}")

    # The 4-way interaction:
    # Home side: home_bat_power × away_sp_hr9 (home batters exploit away SP's homer-proneness)
    # Away side: away_bat_power × home_sp_hr9 (away batters exploit home SP's homer-proneness)
    # Net mismatch: home_side - away_side (positive = home has bigger HR exploitation advantage)

    df['home_hr_exploit'] = df['home_bat_power'] * df['away_sp_hr9']
    df['away_hr_exploit'] = df['away_bat_power'] * df['home_sp_hr9']
    df['hr_mismatch_net'] = df['home_hr_exploit'] - df['away_hr_exploit']

    # Also compute simpler versions for comparison
    df['sp_hr9_diff'] = df['home_sp_hr9'] - df['away_sp_hr9']  # pitcher-only (the dead version)
    df['bat_power_diff'] = df['home_bat_power'] - df['away_bat_power']  # batter-only

    print(f"  hr_mismatch_net: mean={df['hr_mismatch_net'].mean():.4f}, std={df['hr_mismatch_net'].std():.4f}")

    # ── Test 1: Continuous correlation with spread residual ───────────
    print(f"\n{'='*70}")
    print("[6] VS-CLOSE RESIDUAL TESTS")
    print("=" * 70)

    n_tests = 5  # Bonferroni

    for col, name in [
        ('hr_mismatch_net', '4-Way HR Mismatch (batter×pitcher netted)'),
        ('sp_hr9_diff', 'SP HR/9 Diff (pitcher-only, control)'),
        ('bat_power_diff', 'Batting Power Diff (batter-only, control)'),
        ('home_hr_exploit', 'Home HR Exploit (home_bat × away_sp)'),
        ('away_hr_exploit', 'Away HR Exploit (away_bat × home_sp)'),
    ]:
        r, p = stats.pearsonr(df[col], df['spread_residual'])
        p_bonf = min(p * n_tests, 1.0)

        # Split-half
        df_s = df.sort_values('game_date')
        mid = len(df_s) // 2
        r1, _ = stats.pearsonr(df_s.iloc[:mid][col], df_s.iloc[:mid]['spread_residual'])
        r2, _ = stats.pearsonr(df_s.iloc[mid:][col], df_s.iloc[mid:]['spread_residual'])

        # Effect size: top vs bottom quartile cover rate
        q75 = df[col].quantile(0.75)
        q25 = df[col].quantile(0.25)
        top = df[df[col] >= q75]
        bot = df[df[col] <= q25]
        top_cover = top['home_covered'].mean()
        bot_cover = bot['home_covered'].mean()

        be = 0.5238

        print(f"\n  {name}")
        print(f"    r vs spread_residual: {r:+.4f}, p_raw={p:.4f}, p_bonf={p_bonf:.4f}")
        print(f"    Split-half: r1={r1:+.4f}, r2={r2:+.4f} {'STABLE' if np.sign(r1)==np.sign(r2) else 'FLIPS'}")
        print(f"    Top Q home cover: {top_cover*100:.1f}% (N={len(top)}), Bot Q: {bot_cover*100:.1f}% (N={len(bot)})")
        print(f"    Spread: {(top_cover-bot_cover)*100:+.1f}pp, Top edge vs BE: {(top_cover-be)*100:+.1f}pp")

    # ── Test 2: Extreme mismatch + close spread ──────────────────────
    print(f"\n{'='*70}")
    print("[7] EXTREME MISMATCH + EVEN-RATED GAMES")
    print("=" * 70)

    # Even-rated: |close_spread| <= 1.5 (tight games)
    even = df[df['close_spread'].abs() <= 1.5].copy()
    print(f"  Even-rated games (|spread| <= 1.5): {len(even)}")

    if len(even) > 100:
        # Extreme mismatch: top/bottom 25% of hr_mismatch_net
        q75 = even['hr_mismatch_net'].quantile(0.75)
        q25 = even['hr_mismatch_net'].quantile(0.25)

        # Home-favored mismatch (top quartile — home has bigger HR exploit advantage)
        home_adv = even[even['hr_mismatch_net'] >= q75]
        # Away-favored mismatch (bottom quartile)
        away_adv = even[even['hr_mismatch_net'] <= q25]

        home_cover = home_adv['home_covered'].mean()
        away_cover = 1 - away_adv['home_covered'].mean()  # away covers

        be = 0.5238
        print(f"  Home HR advantage (top Q, N={len(home_adv)}): home covers {home_cover*100:.1f}%, edge={home_cover*100-52.38:+.1f}pp")
        print(f"  Away HR advantage (bot Q, N={len(away_adv)}): away covers {away_cover*100:.1f}%, edge={away_cover*100-52.38:+.1f}pp")

        # Mismatch-side cover rate (take the side with the HR advantage)
        mismatch_covers = []
        for _, row in even.iterrows():
            if row['hr_mismatch_net'] >= q75:
                mismatch_covers.append(row['home_covered'])
            elif row['hr_mismatch_net'] <= q25:
                mismatch_covers.append(1 - row['home_covered'])

        if mismatch_covers:
            mismatch_wr = np.mean(mismatch_covers)
            n_mismatch = len(mismatch_covers)
            edge = (mismatch_wr - be) * 100

            # Split-half
            mid = n_mismatch // 2
            h1 = np.mean(mismatch_covers[:mid])
            h2 = np.mean(mismatch_covers[mid:])

            print(f"\n  Combined mismatch-side (N={n_mismatch}):")
            print(f"    WR: {mismatch_wr*100:.1f}%, edge vs BE: {edge:+.1f}pp")
            print(f"    Split-half: H1={h1*100:.1f}%, H2={h2*100:.1f}% "
                  f"{'STABLE' if np.sign(h1-be)==np.sign(h2-be) else 'UNSTABLE'}")

    # ── Test 3: Monthly fingerprint ───────────────────────────────────
    print(f"\n{'='*70}")
    print("[8] MONTHLY FINGERPRINT (flat = lookahead, gradient = real)")
    print("=" * 70)

    df['month'] = pd.to_datetime(df['game_date']).dt.month

    for col, name in [('hr_mismatch_net', '4-Way Mismatch'), ('sp_hr9_diff', 'SP HR/9 Diff (control)')]:
        print(f"\n  {name}:")
        for m in sorted(df['month'].unique()):
            m_data = df[df['month'] == m]
            if len(m_data) < 50:
                continue
            r, p = stats.pearsonr(m_data[col], m_data['spread_residual'])
            print(f"    Month {m}: r={r:+.4f}, p={p:.3f}, N={len(m_data)}")

    # ── Test 4: Correlation with close_spread ─────────────────────────
    print(f"\n{'='*70}")
    print("[9] IS THE MISMATCH ALREADY IN THE SPREAD?")
    print("=" * 70)

    r_close, p_close = stats.pearsonr(df['hr_mismatch_net'], df['close_spread'])
    print(f"  hr_mismatch_net vs close_spread: r={r_close:+.4f}, p={p_close:.4f}")
    print(f"  (If high |r|, the spread already prices the HR mismatch → no residual)")

    # Partial correlation: mismatch → residual AFTER removing close_spread's influence
    # residual already has close removed, so this is redundant but confirming

    # ── VERDICT ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    r_main, p_main = stats.pearsonr(df['hr_mismatch_net'], df['spread_residual'])
    p_bonf_main = min(p_main * n_tests, 1.0)

    # Check if the batting side adds anything beyond pitcher-only
    r_sp, _ = stats.pearsonr(df['sp_hr9_diff'], df['spread_residual'])
    r_bat, _ = stats.pearsonr(df['bat_power_diff'], df['spread_residual'])

    # Partial correlation: mismatch after removing sp_hr9_diff and bat_power_diff
    from numpy.linalg import lstsq
    X = np.column_stack([df['sp_hr9_diff'].values, df['bat_power_diff'].values, np.ones(len(df))])
    beta, _, _, _ = lstsq(X, df['hr_mismatch_net'].values, rcond=None)
    mismatch_resid = df['hr_mismatch_net'].values - X @ beta
    r_partial, p_partial = stats.pearsonr(mismatch_resid, df['spread_residual'].values)

    print(f"\n  4-Way Mismatch vs spread_residual: r={r_main:+.4f}, p_bonf={p_bonf_main:.4f}")
    print(f"  SP HR/9 Diff (pitcher-only): r={r_sp:+.4f}")
    print(f"  Batting Power Diff (batter-only): r={r_bat:+.4f}")
    print(f"  Interaction residual (mismatch beyond linear components): r={r_partial:+.4f}, p={p_partial:.4f}")

    if bat_type == 'season_fallback':
        print(f"\n  *** WARNING: Batting power used SEASON stats (lookahead) ***")
        print(f"  If the 4-way signal shows promise, must retest with PIT batting")

    if p_bonf_main < 0.05:
        print(f"\n  SIGNAL DETECTED — requires further validation:")
        print(f"  - Split-half stability check")
        print(f"  - Monthly flatness fingerprint")
        print(f"  - Edge vs breakeven at actionable confidence")
    else:
        print(f"""
  PARITY: The 4-way HR mismatch does NOT predict spread residuals
  vs the closing line (r={r_main:+.4f}, p_bonf={p_bonf_main:.4f}).

  Component breakdown:
  - SP HR/9 (pitcher-only): r={r_sp:+.4f} — noise
  - Batting power (batter-only): r={r_bat:+.4f} — noise
  - Interaction (beyond components): r={r_partial:+.4f} — noise

  The closing spread already prices both sides of the HR matchup.
  Adding batter-side netting to pitcher HR/9 does not create a
  residual the close misses. This is the same public-data wall:
  HR/9, team HR rates, and their interaction are all public
  information the market incorporates by game time.

  4-way HR matchup on spreads: DEAD.
""")


if __name__ == '__main__':
    run()
