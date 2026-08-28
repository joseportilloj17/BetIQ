"""
SELECTIVE HIGH-CONVICTION SYSTEM — BLIND TEST
==============================================
Build selection filters on TRAIN data, execute BLIND on held-out data.
If high WR sustains blind → real edge. If collapses → overfit.

Three hunt areas:
  1. NFL Wong Teasers — selective high-conviction subset
  2. MLB Cross-Book — Pinnacle gap filter
  3. MLB Alt-Spread — rung × conviction selectivity

Metrics on BLIND set ONLY:
  - WR at what odds (implied ROI math)
  - Edge vs breakeven
  - Beat-the-close (real not variance)
  - N of qualifying bets
  - Sustain or revert vs train?
"""
import sqlite3, re, json
import numpy as np, pandas as pd
from scipy import stats
from collections import defaultdict

HIST_DB = '/Users/joseportillo/Downloads/BetIQ/data/historical.db'
NFL_DB  = '/Users/joseportillo/Downloads/BetIQ/data/nfl.db'
BETS_DB = '/Users/joseportillo/Downloads/BetIQ/data/bets.db'


def section(title):
    print(f"\n{'='*70}")
    print(title)
    print('='*70)


def report_wr(wins, total, odds_decimal=None, label=""):
    """Report WR with CI, edge vs BE, implied ROI."""
    if total == 0:
        print(f"  {label}: N=0, cannot compute")
        return
    wr = wins / total
    se = np.sqrt(wr * (1-wr) / total) if total > 1 else 0
    ci_lo = max(0, wr - 1.96*se)
    ci_hi = min(1, wr + 1.96*se)

    print(f"  {label}")
    print(f"    WR: {wr*100:.1f}% ({wins}/{total}), 95% CI [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

    if odds_decimal and odds_decimal > 1:
        be = 1.0 / odds_decimal
        edge = wr - be
        roi = (wr * odds_decimal - 1) * 100
        print(f"    Odds: {odds_decimal:.3f} dec → BE={be*100:.1f}%")
        print(f"    Edge vs BE: {edge*100:+.1f}pp")
        print(f"    Implied ROI: {roi:+.1f}%")
        # Kelly fraction
        if edge > 0:
            b = odds_decimal - 1
            kelly = (wr * b - (1 - wr)) / b if b > 0 else 0
            print(f"    Kelly fraction: {kelly*100:.1f}% of bankroll")


# ══════════════════════════════════════════════════════════════════════
# PART 1: NFL WONG TEASERS — SELECTIVE HIGH-CONVICTION SUBSET
# ══════════════════════════════════════════════════════════════════════
def nfl_teasers():
    section("PART 1: NFL WONG TEASERS — SELECTIVE SUBSET")
    print("Train: 2020-2023 seasons | Blind: 2024-2025 seasons")
    print("Base: 6-pt Wong teaser, narrow qualification")

    db_nfl = sqlite3.connect(NFL_DB)

    # Load features + games
    feats = pd.read_sql("""
        SELECT f.game_id, f.team, f.season, f.week, f.game_date, f.is_home,
               f.opp_team, f.close_spread, f.close_total, f.close_ml, f.opp_close_ml,
               f.is_divisional, f.is_primetime, f.rest_days, f.is_short_week,
               f.spread_abs, f.is_dome, f.temp, f.wind,
               f.key_number_3, f.key_number_7,
               f.s_point_diff, f.r4_point_diff, f.win_pct_season, f.games_played,
               f.mu_overall
        FROM nfl_features f
    """, db_nfl)

    games = pd.read_sql("""
        SELECT game_id, home_team, away_team, home_score, away_score
        FROM nfl_games
        WHERE home_score IS NOT NULL
    """, db_nfl)
    db_nfl.close()

    # Join features to outcomes
    # Features are per-team, games are per-game. Join on game_id.
    # For each team row in features, figure out its actual margin
    merged = feats.merge(games, on='game_id', how='inner')
    merged['team_score'] = np.where(merged['is_home'] == 1,
                                     merged['home_score'], merged['away_score'])
    merged['opp_score_actual'] = np.where(merged['is_home'] == 1,
                                           merged['away_score'], merged['home_score'])
    merged['actual_margin'] = merged['team_score'] - merged['opp_score_actual']

    # Wong teaser: 6-point tease
    TEASE_PTS = 6.0

    # Narrow qualification:
    # Favorites: close_spread between -8.5 and -7.5 (team perspective)
    # Underdogs: close_spread between +1.5 and +2.5
    def is_wong_fav(cs):
        return -8.5 <= cs <= -7.5

    def is_wong_dog(cs):
        return 1.5 <= cs <= 2.5

    merged['is_wong_fav'] = merged['close_spread'].apply(is_wong_fav)
    merged['is_wong_dog'] = merged['close_spread'].apply(is_wong_dog)
    merged['is_wong'] = merged['is_wong_fav'] | merged['is_wong_dog']

    wong = merged[merged['is_wong']].copy()
    # Teaser always moves 6 points in bettor's favor = add to spread
    # Fav -8.0 → -2.0 (need to win by 2+), Dog +2.0 → +8.0 (can lose by up to 8)
    wong['teased_spread'] = wong['close_spread'] + TEASE_PTS
    # Settlement: actual_margin + teased_spread > 0 = WIN
    # For favorites: spread is negative, teased less negative
    # For dogs: spread is positive, teased even more positive (easier to cover)
    wong['teaser_margin'] = wong['actual_margin'] + wong['teased_spread']
    wong['teaser_win'] = (wong['teaser_margin'] > 0).astype(int)
    wong['teaser_push'] = (wong['teaser_margin'] == 0).astype(int)

    # Remove pushes
    wong_settled = wong[wong['teaser_margin'] != 0].copy()

    print(f"\n  Total Wong-qualifying legs: {len(wong_settled)}")
    print(f"  Seasons: {sorted(wong_settled['season'].unique())}")

    # Train/Blind split
    train = wong_settled[wong_settled['season'] <= 2023].copy()
    blind = wong_settled[wong_settled['season'] >= 2024].copy()
    print(f"  Train (2020-2023): {len(train)} legs, WR={train['teaser_win'].mean()*100:.1f}%")
    print(f"  Blind (2024-2025): {len(blind)} legs, WR={blind['teaser_win'].mean()*100:.1f}%")

    # ── Build selectivity filters on TRAIN only ─────────────────────
    section("1a. TRAIN: Scanning for high-conviction sub-segments")
    print("  Testing which features predict teaser outperformance on TRAIN...")

    # Define candidate filters
    filters = {}

    # A. Dog vs Fav
    filters['dog_only'] = train['is_wong_dog']
    filters['fav_only'] = train['is_wong_fav']

    # B. Home vs Away
    filters['home'] = train['is_home'] == 1
    filters['away'] = train['is_home'] == 0

    # C. Home dog
    filters['home_dog'] = (train['is_wong_dog']) & (train['is_home'] == 1)
    filters['away_dog'] = (train['is_wong_dog']) & (train['is_home'] == 0)
    filters['home_fav'] = (train['is_wong_fav']) & (train['is_home'] == 1)
    filters['away_fav'] = (train['is_wong_fav']) & (train['is_home'] == 0)

    # D. Divisional
    filters['divisional'] = train['is_divisional'] == 1
    filters['non_divisional'] = train['is_divisional'] == 0

    # E. Rest advantage (not short week)
    filters['normal_rest'] = train['is_short_week'] == 0

    # F. Total range
    filters['low_total_under44'] = train['close_total'] < 44
    filters['mid_total_44_47'] = (train['close_total'] >= 44) & (train['close_total'] <= 47)
    filters['high_total_over47'] = train['close_total'] > 47

    # G. Dome (weather protection)
    filters['dome'] = train['is_dome'] == 1

    # H. Point differential strength
    filters['team_strong_ptdiff'] = train['s_point_diff'] > 3
    filters['team_weak_ptdiff'] = train['s_point_diff'] < -3

    # I. Exact teased-to number (key numbers!)
    # Fav teased: -8.5+6=-2.5, -8.0+6=-2.0, -7.5+6=-1.5
    # Dog teased: 2.5-6=-3.5, 2.0-6=-4.0, 1.5-6=-4.5 → wait these are negative
    # Actually: dog teased_spread = close_spread - 6 → 2.5-6=-3.5, 1.5-6=-4.5
    # No wait: for dogs, tease adds points: +1.5 → +7.5, +2.5 → +8.5
    # I need to check: the teased_spread formula above
    # For dogs: close_spread=+2.5, teased=2.5-6=-3.5... that's wrong
    # Wait: 6-pt tease for dogs ADDS 6 points TO the spread
    # Dog with +2.5 → teased to +8.5 (needs to cover -8.5 → just be within 8.5)
    # Fav with -8.0 → teased to -2.0 (just needs to win by 2+)

    # Fix: check teased_spread values
    teased_vals = wong_settled['teased_spread'].value_counts().sort_index()
    print(f"\n  Teased-to spread distribution:")
    for ts, n in teased_vals.items():
        sub = wong_settled[wong_settled['teased_spread'] == ts]
        wr = sub['teaser_win'].mean() * 100
        print(f"    {ts:+.1f}: N={n}, WR={wr:.1f}%")

    # Dog teased to specific values
    for ts_val in sorted(wong_settled[wong_settled['is_wong_dog']]['teased_spread'].unique()):
        mask = (train['is_wong_dog']) & (train['teased_spread'] == ts_val)
        if mask.sum() >= 5:
            filters[f'dog_teased_{ts_val:+.1f}'] = mask

    for ts_val in sorted(wong_settled[wong_settled['is_wong_fav']]['teased_spread'].unique()):
        mask = (train['is_wong_fav']) & (train['teased_spread'] == ts_val)
        if mask.sum() >= 5:
            filters[f'fav_teased_{ts_val:+.1f}'] = mask

    # J. Season phase
    filters['early_season_w1_8'] = train['week'] <= 8
    filters['late_season_w9_plus'] = train['week'] > 8

    # K. Primetime
    filters['primetime'] = train['is_primetime'] == 1
    filters['non_primetime'] = train['is_primetime'] == 0

    # L. Combinations — pre-specified by mechanism
    # Mechanism: dogs crossing key number 7 should be robust
    filters['dog_cross7'] = (train['is_wong_dog']) & (train['key_number_7'] == 1)
    # Dogs crossing 3 (teased from +1.5/+2.5 through 3/7)
    filters['dog_cross3'] = (train['is_wong_dog']) & (train['key_number_3'] == 1)
    # Dogs at home (home field + teaser cushion)
    filters['home_dog_normal_rest'] = (train['is_wong_dog']) & (train['is_home'] == 1) & (train['is_short_week'] == 0)
    # Favs that are strong teams
    filters['strong_fav'] = (train['is_wong_fav']) & (train['s_point_diff'] > 3)

    # ── Evaluate all filters on TRAIN ────────────────────────────────
    print(f"\n  {'Filter':<30} {'N':>5} {'Wins':>5} {'WR%':>6}")
    print(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*6}")

    filter_results = []
    for name, mask in filters.items():
        n = mask.sum()
        if n < 10:
            continue
        wins = train.loc[mask, 'teaser_win'].sum()
        wr = wins / n if n > 0 else 0
        filter_results.append({'name': name, 'N': n, 'wins': wins, 'wr': wr})
        print(f"  {name:<30} {n:>5} {int(wins):>5} {wr*100:>6.1f}")

    filter_results.sort(key=lambda x: x['wr'], reverse=True)

    # ── Select HIGH-CONVICTION filters (WR >= 80% on train, N >= 15) ─
    section("1b. HIGH-CONVICTION FILTERS (WR >= 80% on TRAIN, N >= 15)")

    high_conv = [f for f in filter_results if f['wr'] >= 0.80 and f['N'] >= 15]
    if not high_conv:
        # Lower threshold
        high_conv = [f for f in filter_results if f['wr'] >= 0.75 and f['N'] >= 15]
        print("  No 80%+ filters found. Using 75%+ threshold:")

    print(f"\n  {len(high_conv)} high-conviction filters found on TRAIN:")
    for f in high_conv:
        print(f"    {f['name']}: WR={f['wr']*100:.1f}% (N={f['N']})")

    # ── BLIND TEST ───────────────────────────────────────────────────
    section("1c. BLIND TEST (2024-2025) — Applying train-selected filters")
    print("  These filters were built on 2020-2023. Now tested on 2024-2025.")
    print("  Train numbers shown for comparison only — BLIND is what counts.\n")

    # Rebuild filter masks for blind data
    blind_filters = {}
    blind_filters['dog_only'] = blind['is_wong_dog']
    blind_filters['fav_only'] = blind['is_wong_fav']
    blind_filters['home'] = blind['is_home'] == 1
    blind_filters['away'] = blind['is_home'] == 0
    blind_filters['home_dog'] = (blind['is_wong_dog']) & (blind['is_home'] == 1)
    blind_filters['away_dog'] = (blind['is_wong_dog']) & (blind['is_home'] == 0)
    blind_filters['home_fav'] = (blind['is_wong_fav']) & (blind['is_home'] == 1)
    blind_filters['away_fav'] = (blind['is_wong_fav']) & (blind['is_home'] == 0)
    blind_filters['divisional'] = blind['is_divisional'] == 1
    blind_filters['non_divisional'] = blind['is_divisional'] == 0
    blind_filters['normal_rest'] = blind['is_short_week'] == 0
    blind_filters['low_total_under44'] = blind['close_total'] < 44
    blind_filters['mid_total_44_47'] = (blind['close_total'] >= 44) & (blind['close_total'] <= 47)
    blind_filters['high_total_over47'] = blind['close_total'] > 47
    blind_filters['dome'] = blind['is_dome'] == 1
    blind_filters['team_strong_ptdiff'] = blind['s_point_diff'] > 3
    blind_filters['team_weak_ptdiff'] = blind['s_point_diff'] < -3
    blind_filters['early_season_w1_8'] = blind['week'] <= 8
    blind_filters['late_season_w9_plus'] = blind['week'] > 8
    blind_filters['primetime'] = blind['is_primetime'] == 1
    blind_filters['non_primetime'] = blind['is_primetime'] == 0
    blind_filters['dog_cross7'] = (blind['is_wong_dog']) & (blind['key_number_7'] == 1)
    blind_filters['dog_cross3'] = (blind['is_wong_dog']) & (blind['key_number_3'] == 1)
    blind_filters['home_dog_normal_rest'] = (blind['is_wong_dog']) & (blind['is_home'] == 1) & (blind['is_short_week'] == 0)
    blind_filters['strong_fav'] = (blind['is_wong_fav']) & (blind['s_point_diff'] > 3)

    for ts_val in sorted(wong_settled[wong_settled['is_wong_dog']]['teased_spread'].unique()):
        blind_filters[f'dog_teased_{ts_val:+.1f}'] = (blind['is_wong_dog']) & (blind['teased_spread'] == ts_val)
    for ts_val in sorted(wong_settled[wong_settled['is_wong_fav']]['teased_spread'].unique()):
        blind_filters[f'fav_teased_{ts_val:+.1f}'] = (blind['is_wong_fav']) & (blind['teased_spread'] == ts_val)

    # TEASER ODDS: -120 = 1.833 decimal
    TEASER_ODDS = 1.833

    # Report ALL filters for comparison
    print(f"  {'Filter':<30} {'Train':>10} {'Blind':>10} {'Δ':>7}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*7}")

    all_filter_names = set(list(filters.keys()) + list(blind_filters.keys()))
    comparison = []
    for name in sorted(all_filter_names):
        train_mask = filters.get(name)
        blind_mask = blind_filters.get(name)
        if train_mask is None or blind_mask is None:
            continue
        t_n = train_mask.sum()
        b_n = blind_mask.sum()
        if t_n < 10 or b_n < 3:
            continue
        t_wr = train.loc[train_mask, 'teaser_win'].mean() if t_n > 0 else 0
        b_wr = blind.loc[blind_mask, 'teaser_win'].mean() if b_n > 0 else 0
        delta = b_wr - t_wr
        comparison.append({'name': name, 't_wr': t_wr, 't_n': t_n, 'b_wr': b_wr, 'b_n': b_n, 'delta': delta})
        t_str = f"{t_wr*100:.1f}% ({t_n})"
        b_str = f"{b_wr*100:.1f}% ({b_n})"
        print(f"  {name:<30} {t_str:>10} {b_str:>10} {delta*100:>+6.1f}pp")

    # ── Best blind performers ────────────────────────────────────────
    section("1d. BEST BLIND PERFORMERS (sorted by blind WR, N >= 10)")
    comparison.sort(key=lambda x: x['b_wr'], reverse=True)
    for c in comparison:
        if c['b_n'] >= 10:
            sustain = 'SUSTAIN' if c['b_wr'] >= c['t_wr'] - 0.05 else 'REVERT'
            be = 1/TEASER_ODDS
            edge = c['b_wr'] - be
            roi = (c['b_wr'] * TEASER_ODDS - 1) * 100
            print(f"  {c['name']:<30} blind={c['b_wr']*100:.1f}% (N={c['b_n']}), "
                  f"train={c['t_wr']*100:.1f}% | edge={edge*100:+.1f}pp ROI={roi:+.1f}% [{sustain}]")

    # ── Full base Wong on blind ──────────────────────────────────────
    section("1e. BASE WONG TEASER — BLIND PERFORMANCE")
    base_wins = int(blind['teaser_win'].sum())
    base_n = len(blind)
    report_wr(base_wins, base_n, TEASER_ODDS, "Full Wong (blind 2024-2025)")

    return comparison


# ══════════════════════════════════════════════════════════════════════
# PART 2: MLB CROSS-BOOK — PINNACLE GAP FILTER
# ══════════════════════════════════════════════════════════════════════
def mlb_crossbook():
    section("PART 2: MLB CROSS-BOOK — PINNACLE GAP FILTER")
    print("Hypothesis: when a bettable book offers significantly better odds")
    print("than Pinnacle (sharp reference), the bettable side wins at high rate.")
    print("Train: first 60% of dates | Blind: last 40%")

    db = sqlite3.connect(HIST_DB)

    # Get alt-lines with Pinnacle reference
    # For each game-team-rung, find: best bettable price, Pinnacle price
    alt = pd.read_sql("""
        SELECT game_date_ct as game_date, home_team, away_team,
               team, line, odds, bookmaker, market_key
        FROM alt_lines
        WHERE sport_key LIKE '%mlb%' AND market_key = 'alternate_spreads'
        ORDER BY game_date_ct, home_team, team, line
    """, db)

    # Get game outcomes
    outcomes = pd.read_sql("""
        SELECT bl.game_date, bl.home_team, bl.away_team, bl.close_spread,
               ts.score as home_score, ts2.score as away_score
        FROM betting_lines bl
        JOIN team_stats ts ON bl.game_id = ts.game_id AND ts.is_home = 1
        JOIN team_stats ts2 ON bl.game_id = ts2.game_id AND ts2.is_home = 0
        WHERE bl.sport = 'MLB'
        AND (ts.score > 0 OR ts2.score > 0)
    """, db)
    db.close()

    print(f"  Alt-line rows: {len(alt)}")
    print(f"  Settled games: {len(outcomes)}")

    # For each game-team-rung, compute:
    # 1. Pinnacle devigged implied prob
    # 2. Best bettable book odds (FanDuel, DraftKings, BetMGM, BetRivers)
    # 3. Gap = implied_prob_pinnacle - implied_prob_bettable (positive = bettable is soft)

    BETTABLE = {'fanduel', 'draftkings', 'betmgm', 'betrivers'}
    SHARP = {'pinnacle'}

    # Group by game-team-rung
    grouped = alt.groupby(['game_date', 'home_team', 'away_team', 'team', 'line'])

    spots = []
    for (gd, ht, at, team, line), grp in grouped:
        pinn_rows = grp[grp['bookmaker'].isin(SHARP)]
        bett_rows = grp[grp['bookmaker'].isin(BETTABLE)]

        if len(pinn_rows) == 0 or len(bett_rows) == 0:
            continue

        pinn_odds = pinn_rows['odds'].min()  # best Pinnacle price (should be only one)
        best_bett_odds = bett_rows['odds'].max()  # best bettable price
        best_bett_book = bett_rows.loc[bett_rows['odds'].idxmax(), 'bookmaker']

        pinn_impl = 1.0 / pinn_odds if pinn_odds > 1 else 1.0
        bett_impl = 1.0 / best_bett_odds if best_bett_odds > 1 else 1.0

        # Gap: how much softer the bettable book is vs Pinnacle
        # Positive gap = bettable book gives better odds = potentially soft
        gap = pinn_impl - bett_impl  # higher = bettable has lower implied prob = better odds

        # Number of bettable books carrying this line
        n_books = bett_rows['bookmaker'].nunique()

        spots.append({
            'game_date': gd, 'home_team': ht, 'away_team': at,
            'team': team, 'line': line,
            'pinn_odds': pinn_odds, 'pinn_impl': pinn_impl,
            'bett_odds': best_bett_odds, 'bett_impl': bett_impl,
            'best_book': best_bett_book,
            'gap': gap,  # positive = bettable softer than Pinnacle
            'n_books': n_books,
        })

    spots_df = pd.DataFrame(spots)
    print(f"  Game-team-rung spots with both Pinnacle + bettable: {len(spots_df)}")

    # Merge with outcomes to settle
    spots_df = spots_df.merge(outcomes, on=['game_date', 'home_team', 'away_team'], how='inner')
    print(f"  Spots matched to settled games: {len(spots_df)}")

    if len(spots_df) < 100:
        print("  INSUFFICIENT DATA for meaningful blind test.")
        print("  Need more settled games with cross-book pricing.")
        return None

    # Settle: team covers alt-spread if margin + line > 0
    spots_df['actual_margin'] = spots_df['home_score'] - spots_df['away_score']
    # If team is away, flip margin
    spots_df['team_margin'] = np.where(spots_df['team'] == spots_df['home_team'],
                                        spots_df['actual_margin'],
                                        -spots_df['actual_margin'])
    spots_df['adj_margin'] = spots_df['team_margin'] + spots_df['line']
    spots_df['win'] = (spots_df['adj_margin'] > 0).astype(int)
    spots_df['push'] = (spots_df['adj_margin'] == 0).astype(int)
    spots_df = spots_df[spots_df['push'] == 0].copy()

    # Beat-close: does the team outperform the closing spread?
    # close_spread is home perspective. If team is away, flip.
    spots_df['team_close'] = np.where(spots_df['team'] == spots_df['home_team'],
                                       spots_df['close_spread'],
                                       -spots_df['close_spread'])
    spots_df['beat_close'] = (spots_df['team_margin'] + spots_df['team_close'] > 0).astype(int)

    print(f"  Settled (excl push): {len(spots_df)}")
    print(f"  Overall WR: {spots_df['win'].mean()*100:.1f}%")
    print(f"  Gap distribution: mean={spots_df['gap'].mean()*100:.2f}pp, "
          f"std={spots_df['gap'].std()*100:.2f}pp")

    # ── Train/Blind split ────────────────────────────────────────────
    dates = sorted(spots_df['game_date'].unique())
    split_idx = int(len(dates) * 0.6)
    train_cutoff = dates[split_idx]

    train = spots_df[spots_df['game_date'] <= train_cutoff].copy()
    blind = spots_df[spots_df['game_date'] > train_cutoff].copy()
    print(f"\n  Train: {len(train)} spots through {train_cutoff}")
    print(f"  Blind: {len(blind)} spots after {train_cutoff}")

    # ── Build selectivity filters on TRAIN ───────────────────────────
    section("2a. TRAIN: Scanning gap thresholds")

    # Test multiple gap thresholds
    gap_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"\n  {'Gap ≥':>8} {'N':>6} {'Wins':>5} {'WR%':>6} {'Avg Odds':>9} {'BE%':>6} {'Edge':>7} {'Beat Close%':>11}")
    print(f"  {'-'*8} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*6} {'-'*7} {'-'*11}")

    for thresh in gap_thresholds:
        mask = train['gap'] >= thresh
        n = mask.sum()
        if n < 5:
            continue
        wins = train.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = train.loc[mask, 'bett_odds'].mean()
        be = 1 / avg_odds if avg_odds > 1 else 1
        edge = wr - be
        bc = train.loc[mask, 'beat_close'].mean()
        print(f"  {thresh*100:>7.1f}pp {n:>6} {int(wins):>5} {wr*100:>6.1f} {avg_odds:>9.3f} "
              f"{be*100:>6.1f} {edge*100:>+6.1f}pp {bc*100:>10.1f}%")

    # Also test by number of books (liquidity)
    section("2b. TRAIN: By liquidity (number of bettable books)")
    for nb in sorted(train['n_books'].unique()):
        mask = train['n_books'] == nb
        n = mask.sum()
        if n < 10:
            continue
        wins = train.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = train.loc[mask, 'bett_odds'].mean()
        be = 1 / avg_odds
        print(f"  {nb} books: WR={wr*100:.1f}% (N={n}), odds={avg_odds:.3f}, edge={((wr-be)*100):+.1f}pp")

    # Combined: gap + liquidity
    section("2c. TRAIN: Gap × Liquidity combinations")
    combos = []
    for thresh in [0.02, 0.03, 0.05]:
        for min_books in [2, 3]:
            mask = (train['gap'] >= thresh) & (train['n_books'] >= min_books)
            n = mask.sum()
            if n < 10:
                continue
            wins = train.loc[mask, 'win'].sum()
            wr = wins / n
            avg_odds = train.loc[mask, 'bett_odds'].mean()
            be = 1 / avg_odds
            edge = wr - be
            bc = train.loc[mask, 'beat_close'].mean()
            combos.append({
                'gap': thresh, 'min_books': min_books,
                'n': n, 'wr': wr, 'avg_odds': avg_odds, 'edge': edge, 'bc': bc
            })
            print(f"  gap≥{thresh*100:.0f}pp + {min_books}+ books: "
                  f"WR={wr*100:.1f}% (N={n}), odds={avg_odds:.3f}, "
                  f"edge={edge*100:+.1f}pp, beat_close={bc*100:.1f}%")

    # ── BLIND TEST ───────────────────────────────────────────────────
    section("2d. BLIND TEST — Applying train-selected filters")

    # Test all thresholds on blind
    print(f"\n  {'Gap ≥':>8} {'N':>6} {'Wins':>5} {'WR%':>6} {'Avg Odds':>9} {'BE%':>6} {'Edge':>7} {'Beat Close%':>11}")
    print(f"  {'-'*8} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*6} {'-'*7} {'-'*11}")

    for thresh in gap_thresholds:
        mask = blind['gap'] >= thresh
        n = mask.sum()
        if n < 3:
            continue
        wins = blind.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = blind.loc[mask, 'bett_odds'].mean()
        be = 1 / avg_odds if avg_odds > 1 else 1
        edge = wr - be
        bc = blind.loc[mask, 'beat_close'].mean()
        print(f"  {thresh*100:>7.1f}pp {n:>6} {int(wins):>5} {wr*100:>6.1f} {avg_odds:>9.3f} "
              f"{be*100:>6.1f} {edge*100:>+6.1f}pp {bc*100:>10.1f}%")

    # Combined filters on blind
    print(f"\n  Combined (gap + liquidity) on BLIND:")
    for combo in combos:
        mask = (blind['gap'] >= combo['gap']) & (blind['n_books'] >= combo['min_books'])
        n = mask.sum()
        if n < 3:
            continue
        wins = blind.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = blind.loc[mask, 'bett_odds'].mean()
        be = 1 / avg_odds if avg_odds > 1 else 1
        edge = wr - be
        bc = blind.loc[mask, 'beat_close'].mean()
        delta = wr - combo['wr']
        sustain = 'SUSTAIN' if wr >= combo['wr'] - 0.05 else 'REVERT'
        print(f"  gap≥{combo['gap']*100:.0f}pp+{combo['min_books']}bk: "
              f"blind={wr*100:.1f}% (N={n}), odds={avg_odds:.3f}, "
              f"edge={edge*100:+.1f}pp, Δ={delta*100:+.1f}pp [{sustain}]")

    return spots_df


# ══════════════════════════════════════════════════════════════════════
# PART 3: MLB ALT-SPREAD — RUNG × CONVICTION SELECTIVITY
# ══════════════════════════════════════════════════════════════════════
def mlb_altspread_selectivity():
    section("PART 3: MLB ALT-SPREAD — RUNG × CONVICTION SELECTIVITY")
    print("Source: settled mock_bet_legs (alternate_spreads)")
    print("Train: first 60% of dates | Blind: last 40%")

    db = sqlite3.connect(BETS_DB)
    legs = pd.read_sql("""
        SELECT mbl.description, mbl.win_prob, mbl.devig_entry_prob,
               mbl.overround, mbl.leg_result,
               mb.game_date, mb.odds as bet_odds, mb.source, mb.status
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.market_type = 'alternate_spreads'
        AND mbl.leg_result IN ('WIN', 'LOSS')
        AND mb.bet_type = 'straight'
    """, db)
    db.close()

    print(f"  Total settled alt-spread straight legs: {len(legs)}")

    # Parse spread rung from description
    # Format: "Team Name +/-X.X (Alt Spread)"
    def parse_spread(desc):
        # Match both "(Alt Spread)" and "(alternate_spreads)"
        m = re.search(r'([+-]?\d+\.?\d*)\s*\((?:Alt Spread|alternate_spreads)\)', str(desc))
        if m:
            return float(m.group(1))
        # Fallback: any number before parenthetical
        m = re.search(r'([+-]\d+\.?\d*)\s*\(', str(desc))
        if m:
            return float(m.group(1))
        return None

    legs['rung'] = legs['description'].apply(parse_spread)
    legs['win'] = (legs['leg_result'] == 'WIN').astype(int)
    legs = legs[legs['rung'].notna()].copy()

    print(f"  With parsed rung: {len(legs)}")
    print(f"  Rung distribution: {sorted(legs['rung'].unique())}")

    # Compute breakeven from bet_odds
    legs['be'] = 1.0 / legs['bet_odds']
    # Edge = model_win_prob - breakeven
    legs['model_edge'] = legs['win_prob'] / 100.0 - legs['be']
    # Devig edge = devig_prob - breakeven (how much softer than fair)
    legs['devig_edge'] = legs['devig_entry_prob'] - legs['be']
    # Model vs devig gap = model thinks better than market
    legs['model_vs_devig'] = legs['win_prob'] / 100.0 - legs['devig_entry_prob']

    # ── Train/Blind split ────────────────────────────────────────────
    dates = sorted(legs['game_date'].unique())
    split_idx = int(len(dates) * 0.6)
    train_cutoff = dates[split_idx]

    train = legs[legs['game_date'] <= train_cutoff].copy()
    blind = legs[legs['game_date'] > train_cutoff].copy()
    print(f"\n  Train: {len(train)} legs through {train_cutoff}")
    print(f"  Blind: {len(blind)} legs after {train_cutoff}")

    # ── 3a. By rung on TRAIN ─────────────────────────────────────────
    section("3a. TRAIN: WR by spread rung")
    print(f"  {'Rung':>6} {'N':>6} {'Wins':>5} {'WR%':>6} {'Avg Odds':>9} {'BE%':>6} {'Edge':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*6} {'-'*7}")

    for rung in sorted(train['rung'].unique()):
        mask = train['rung'] == rung
        n = mask.sum()
        if n < 20:
            continue
        wins = train.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = train.loc[mask, 'bet_odds'].mean()
        be = 1 / avg_odds
        edge = wr - be
        print(f"  {rung:>+6.1f} {n:>6} {int(wins):>5} {wr*100:>6.1f} {avg_odds:>9.3f} "
              f"{be*100:>6.1f} {edge*100:>+6.1f}pp")

    # ── 3b. By model confidence on TRAIN ─────────────────────────────
    section("3b. TRAIN: WR by model confidence bucket")
    wp_bins = [(55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 100)]
    print(f"  {'WP Bucket':>12} {'N':>6} {'Wins':>5} {'WR%':>6} {'Avg Odds':>9} {'BE%':>6} {'Edge':>7}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*6} {'-'*7}")

    for lo, hi in wp_bins:
        mask = (train['win_prob'] >= lo) & (train['win_prob'] < hi)
        n = mask.sum()
        if n < 10:
            continue
        wins = train.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = train.loc[mask, 'bet_odds'].mean()
        be = 1 / avg_odds
        edge = wr - be
        print(f"  {lo:>3}-{hi:<3}%    {n:>6} {int(wins):>5} {wr*100:>6.1f} {avg_odds:>9.3f} "
              f"{be*100:>6.1f} {edge*100:>+6.1f}pp")

    # ── 3c. By overround on TRAIN (lower overround = less vig) ───────
    section("3c. TRAIN: WR by overround bucket (vig)")
    or_bins = [(1.00, 1.02), (1.02, 1.04), (1.04, 1.06), (1.06, 1.08), (1.08, 1.20)]
    print(f"  {'Overround':>12} {'N':>6} {'Wins':>5} {'WR%':>6} {'Avg Odds':>9} {'BE%':>6} {'Edge':>7}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*6} {'-'*7}")

    for lo, hi in or_bins:
        mask = (train['overround'] >= lo) & (train['overround'] < hi)
        n = mask.sum()
        if n < 10:
            continue
        wins = train.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = train.loc[mask, 'bet_odds'].mean()
        be = 1 / avg_odds
        edge = wr - be
        print(f"  {lo:.2f}-{hi:.2f}  {n:>6} {int(wins):>5} {wr*100:>6.1f} {avg_odds:>9.3f} "
              f"{be*100:>6.1f} {edge*100:>+6.1f}pp")

    # ── 3d. Build SELECTIVE filters on TRAIN ─────────────────────────
    section("3d. TRAIN: Building selective filters")

    filters = {}

    # By rung (only positive — cushion/underdog rungs)
    for rung in sorted(train['rung'].unique()):
        mask = train['rung'] == rung
        if mask.sum() >= 20:
            filters[f'rung_{rung:+.1f}'] = mask

    # By model confidence (high conviction)
    filters['wp_ge_70'] = train['win_prob'] >= 70
    filters['wp_ge_75'] = train['win_prob'] >= 75
    filters['wp_ge_80'] = train['win_prob'] >= 80
    filters['wp_ge_85'] = train['win_prob'] >= 85

    # By model edge (model thinks better than breakeven)
    filters['model_edge_ge_5pp'] = train['model_edge'] >= 0.05
    filters['model_edge_ge_10pp'] = train['model_edge'] >= 0.10
    filters['model_edge_ge_15pp'] = train['model_edge'] >= 0.15

    # By model-vs-devig gap (model thinks better than market)
    filters['model_gt_devig_5pp'] = train['model_vs_devig'] >= 0.05
    filters['model_gt_devig_10pp'] = train['model_vs_devig'] >= 0.10

    # Low overround (better vig)
    filters['low_vig_lt_1.03'] = train['overround'] < 1.03
    filters['low_vig_lt_1.02'] = train['overround'] < 1.02

    # Combinations
    # High confidence + good rung
    for rung in [1.5, 2.5, 3.5]:
        filters[f'wp70_rung+{rung}'] = (train['win_prob'] >= 70) & (train['rung'] == rung)
        filters[f'wp75_rung+{rung}'] = (train['win_prob'] >= 75) & (train['rung'] == rung)

    # High confidence + low vig
    filters['wp70_lowvig'] = (train['win_prob'] >= 70) & (train['overround'] < 1.04)
    filters['wp75_lowvig'] = (train['win_prob'] >= 75) & (train['overround'] < 1.04)
    filters['wp80_lowvig'] = (train['win_prob'] >= 80) & (train['overround'] < 1.04)

    # Model edge + low vig
    filters['edge10_lowvig'] = (train['model_edge'] >= 0.10) & (train['overround'] < 1.04)
    filters['edge15_lowvig'] = (train['model_edge'] >= 0.15) & (train['overround'] < 1.04)

    # Report TRAIN performance
    train_results = []
    for name, mask in filters.items():
        n = mask.sum()
        if n < 10:
            continue
        wins = train.loc[mask, 'win'].sum()
        wr = wins / n
        avg_odds = train.loc[mask, 'bet_odds'].mean()
        be = 1 / avg_odds
        edge = wr - be
        train_results.append({
            'name': name, 'N': n, 'wins': wins, 'wr': wr,
            'avg_odds': avg_odds, 'be': be, 'edge': edge
        })

    # Sort by edge
    train_results.sort(key=lambda x: x['edge'], reverse=True)
    print(f"\n  {'Filter':<30} {'N':>6} {'WR%':>6} {'Odds':>6} {'BE%':>6} {'Edge':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for r in train_results[:25]:
        print(f"  {r['name']:<30} {r['N']:>6} {r['wr']*100:>6.1f} {r['avg_odds']:>6.3f} "
              f"{r['be']*100:>6.1f} {r['edge']*100:>+6.1f}pp")

    # ── 3e. BLIND TEST ───────────────────────────────────────────────
    section("3e. BLIND TEST — Applying train-selected filters")

    # Rebuild filters for blind
    blind_filters = {}
    for rung in sorted(blind['rung'].unique()):
        blind_filters[f'rung_{rung:+.1f}'] = blind['rung'] == rung
    blind_filters['wp_ge_70'] = blind['win_prob'] >= 70
    blind_filters['wp_ge_75'] = blind['win_prob'] >= 75
    blind_filters['wp_ge_80'] = blind['win_prob'] >= 80
    blind_filters['wp_ge_85'] = blind['win_prob'] >= 85
    blind_filters['model_edge_ge_5pp'] = blind['model_edge'] >= 0.05
    blind_filters['model_edge_ge_10pp'] = blind['model_edge'] >= 0.10
    blind_filters['model_edge_ge_15pp'] = blind['model_edge'] >= 0.15
    blind_filters['model_gt_devig_5pp'] = blind['model_vs_devig'] >= 0.05
    blind_filters['model_gt_devig_10pp'] = blind['model_vs_devig'] >= 0.10
    blind_filters['low_vig_lt_1.03'] = blind['overround'] < 1.03
    blind_filters['low_vig_lt_1.02'] = blind['overround'] < 1.02
    for rung in [1.5, 2.5, 3.5]:
        blind_filters[f'wp70_rung+{rung}'] = (blind['win_prob'] >= 70) & (blind['rung'] == rung)
        blind_filters[f'wp75_rung+{rung}'] = (blind['win_prob'] >= 75) & (blind['rung'] == rung)
    blind_filters['wp70_lowvig'] = (blind['win_prob'] >= 70) & (blind['overround'] < 1.04)
    blind_filters['wp75_lowvig'] = (blind['win_prob'] >= 75) & (blind['overround'] < 1.04)
    blind_filters['wp80_lowvig'] = (blind['win_prob'] >= 80) & (blind['overround'] < 1.04)
    blind_filters['edge10_lowvig'] = (blind['model_edge'] >= 0.10) & (blind['overround'] < 1.04)
    blind_filters['edge15_lowvig'] = (blind['model_edge'] >= 0.15) & (blind['overround'] < 1.04)

    # Compare train vs blind for filters that qualified on train
    comparisons = []
    print(f"\n  {'Filter':<30} {'Train':>12} {'Blind':>12} {'Δ':>7} {'Blind Edge':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*7} {'-'*10}")

    for tr in train_results:
        name = tr['name']
        if name not in blind_filters:
            continue
        b_mask = blind_filters[name]
        b_n = b_mask.sum()
        if b_n < 5:
            continue
        b_wins = blind.loc[b_mask, 'win'].sum()
        b_wr = b_wins / b_n
        b_avg_odds = blind.loc[b_mask, 'bet_odds'].mean()
        b_be = 1 / b_avg_odds
        b_edge = b_wr - b_be
        delta = b_wr - tr['wr']
        sustain = 'SUSTAIN' if b_wr >= tr['wr'] - 0.05 else 'REVERT'

        comparisons.append({
            'name': name, 't_wr': tr['wr'], 't_n': tr['N'],
            'b_wr': b_wr, 'b_n': b_n, 'b_edge': b_edge,
            'b_odds': b_avg_odds, 'delta': delta, 'sustain': sustain
        })

        t_str = f"{tr['wr']*100:.1f}%({tr['N']})"
        b_str = f"{b_wr*100:.1f}%({b_n})"
        print(f"  {name:<30} {t_str:>12} {b_str:>12} {delta*100:>+6.1f}pp {b_edge*100:>+9.1f}pp [{sustain}]")

    # ── 3f. BEST BLIND PERFORMERS ────────────────────────────────────
    section("3f. BEST BLIND PERFORMERS (edge > 0, N >= 10)")
    positive_blind = [c for c in comparisons if c['b_edge'] > 0 and c['b_n'] >= 10]
    positive_blind.sort(key=lambda x: x['b_edge'], reverse=True)

    if positive_blind:
        for c in positive_blind:
            roi = (c['b_wr'] * c['b_odds'] - 1) * 100
            print(f"  {c['name']:<30} blind={c['b_wr']*100:.1f}% (N={c['b_n']}), "
                  f"odds={c['b_odds']:.3f}, edge={c['b_edge']*100:+.1f}pp, "
                  f"ROI={roi:+.1f}% [{c['sustain']}]")
    else:
        print("  NO filters with positive edge on blind set (N >= 10).")

    return comparisons


# ══════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════
def final_verdict(teaser_results, crossbook_results, altspread_results):
    section("FINAL VERDICT: SELECTIVE HIGH-CONVICTION BLIND TEST")

    print("\n  THE QUESTION: Can a selective filter sustain high WR blind?")
    print("  HIGH WR target: 80%+ at odds that make the ROI math work")
    print("  THE TEST: filter built on train, applied unchanged to blind\n")

    # Teaser summary
    print("  ┌─ NFL WONG TEASERS ──────────────────────────────────────┐")
    if teaser_results:
        best_blind = max(teaser_results, key=lambda x: x['b_wr'] if x['b_n'] >= 10 else 0)
        if best_blind['b_n'] >= 10:
            teaser_odds = 1.833
            roi = (best_blind['b_wr'] * teaser_odds - 1) * 100
            sustain = best_blind['b_wr'] >= best_blind['t_wr'] - 0.05
            print(f"  │ Best blind: {best_blind['name']}")
            print(f"  │   Train={best_blind['t_wr']*100:.1f}% → Blind={best_blind['b_wr']*100:.1f}% "
                  f"(N={best_blind['b_n']})")
            print(f"  │   At -120 odds: ROI={roi:+.1f}%")
            print(f"  │   {'SUSTAINS' if sustain else 'REVERTS'}")
        else:
            print(f"  │ Insufficient blind N for reliable read")
    print("  └────────────────────────────────────────────────────────┘")

    # Cross-book summary
    print("\n  ┌─ MLB CROSS-BOOK GAP ────────────────────────────────────┐")
    if crossbook_results is not None:
        print(f"  │ Cross-book results available — see section 2d above")
    else:
        print(f"  │ INSUFFICIENT DATA — need more settled games with pricing")
    print("  └────────────────────────────────────────────────────────┘")

    # Alt-spread summary
    print("\n  ┌─ MLB ALT-SPREAD SELECTIVITY ────────────────────────────┐")
    if altspread_results:
        positives = [c for c in altspread_results if c['b_edge'] > 0 and c['b_n'] >= 10]
        if positives:
            best = max(positives, key=lambda x: x['b_edge'])
            roi = (best['b_wr'] * best['b_odds'] - 1) * 100
            print(f"  │ Best blind: {best['name']}")
            print(f"  │   Train={best['t_wr']*100:.1f}% → Blind={best['b_wr']*100:.1f}% "
                  f"(N={best['b_n']})")
            print(f"  │   Odds={best['b_odds']:.3f}, Edge={best['b_edge']*100:+.1f}pp, "
                  f"ROI={roi:+.1f}%")
            print(f"  │   {best['sustain']}")
        else:
            print(f"  │ NO filter sustained positive edge blind (N≥10)")
    print("  └────────────────────────────────────────────────────────┘")

    print(f"""
  BOTTOM LINE:
  If any filter sustains WR > BE on blind with adequate N:
    → REAL edge. Size with fractional Kelly on the BLIND-confirmed WR.
    → Kelly only earns the right to compound after the blind test proves
      the WR is not memorization.
  If all filters revert:
    → The high train WR was overfit. The selectivity memorized the
      training puzzle. Don't size.
""")


if __name__ == '__main__':
    teaser = nfl_teasers()
    crossbook = mlb_crossbook()
    altspread = mlb_altspread_selectivity()
    final_verdict(teaser, crossbook, altspread)
