#!/usr/bin/env python3
"""
Cross-Estimator Disagreement & Bust Signal Analysis
Investigates whether signals OUTSIDE NegBin's own WP can predict which
high-confidence fixtures bust.
"""

import sqlite3
import statistics
import math
from collections import defaultdict

DB_BETS = "../data/bets.db"
DB_HIST = "../data/historical.db"

def ml_to_prob(ml):
    """Convert American ML to implied probability (0-1)."""
    if ml is None:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    elif ml > 0:
        return 100 / (ml + 100)
    else:
        return 0.5

def mann_whitney_u(x, y):
    """Simple Mann-Whitney U test. Returns U statistic and approximate p-value."""
    if len(x) < 2 or len(y) < 2:
        return None, None
    nx, ny = len(x), len(y)
    # Rank all values
    combined = [(v, 'x') for v in x] + [(v, 'y') for v in y]
    combined.sort(key=lambda t: t[0])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed
        for k in range(i, j):
            if combined[k][1] not in ranks:
                ranks[combined[k][1]] = []
        for k in range(i, j):
            ranks.setdefault(f"{combined[k][1]}_{k}", [])
        i = j

    # Simpler: direct U computation
    u = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                u += 1
            elif xi == yi:
                u += 0.5

    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12)
    if sigma == 0:
        return u, 1.0
    z = (u - mu) / sigma
    # Approximate two-tailed p-value from z
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u, p

def cohens_d(x, y):
    """Effect size."""
    if len(x) < 2 or len(y) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.stdev(x), statistics.stdev(y)
    pooled = math.sqrt((sx**2 + sy**2) / 2)
    if pooled == 0:
        return 0
    return (mx - my) / pooled

# ============================================================
# LOAD DATA
# ============================================================
bconn = sqlite3.connect(DB_BETS)
bconn.row_factory = sqlite3.Row
bc = bconn.cursor()

hconn = sqlite3.connect(DB_HIST)
hconn.row_factory = sqlite3.Row
hc = hconn.cursor()

# Get all settled NegBin legs
bc.execute("""
SELECT l.fixture_id, l.negbin_wp, l.negbin_mu_home, l.negbin_mu_away,
       l.negbin_cell, l.negbin_edge_pp, l.leg_result, l.market_type,
       l.is_alt_line, l.resolved_home_team, l.resolved_away_team,
       l.resolved_home_score, l.resolved_away_score, l.resolved_margin,
       l.accuracy_delta, l.description,
       b.game_date, b.source, b.id as bet_id, b.status, b.odds, b.actual_profit, b.legs as n_legs
FROM mock_bet_legs l
JOIN mock_bets b ON l.mock_bet_id = b.id
WHERE l.negbin_wp > 0 AND l.leg_result IN ('WIN','LOSS')
  AND b.cohort = 'clean'
  AND b.source LIKE 'shadow_negbin%'
ORDER BY b.game_date, l.fixture_id
""")
all_legs = [dict(r) for r in bc.fetchall()]
print(f"Total settled NegBin legs: {len(all_legs)}")

# Build market implied prob lookup from historical.db betting_lines
# Join by home_team + game_date
hc.execute("""
SELECT home_team, away_team, game_date, close_ml_home, close_ml_away,
       close_spread, close_total
FROM betting_lines
WHERE game_date >= '2026-06-28' AND close_ml_home IS NOT NULL
""")
market_data = {}
for r in hc.fetchall():
    key = (r['home_team'], r['game_date'])
    market_data[key] = {
        'close_ml_home': r['close_ml_home'],
        'close_ml_away': r['close_ml_away'],
        'close_spread': r['close_spread'],
        'close_total': r['close_total'],
        'market_home_prob': ml_to_prob(r['close_ml_home']),
        'market_away_prob': ml_to_prob(r['close_ml_away']),
    }

# ============================================================
# Build fixture-level aggregates
# ============================================================
fixtures = defaultdict(lambda: {
    'legs': [], 'wins': 0, 'losses': 0,
    'negbin_wps': [], 'mu_homes': [], 'mu_aways': [],
    'edge_pps': [], 'cells': set(), 'market_types': set(),
    'game_date': None, 'home_team': None, 'away_team': None,
    'home_score': None, 'away_score': None, 'bet_ids': set(),
})

for leg in all_legs:
    fid = leg['fixture_id']
    f = fixtures[fid]
    f['legs'].append(leg)
    if leg['leg_result'] == 'WIN':
        f['wins'] += 1
    else:
        f['losses'] += 1
    f['negbin_wps'].append(leg['negbin_wp'])
    f['mu_homes'].append(leg['negbin_mu_home'])
    f['mu_aways'].append(leg['negbin_mu_away'])
    f['edge_pps'].append(leg['negbin_edge_pp'])
    f['cells'].add(leg['negbin_cell'])
    f['market_types'].add(leg['market_type'])
    f['game_date'] = leg['game_date']
    f['home_team'] = leg['resolved_home_team']
    f['away_team'] = leg['resolved_away_team']
    f['home_score'] = leg['resolved_home_score']
    f['away_score'] = leg['resolved_away_score']
    f['bet_ids'].add(leg['bet_id'])

# Enrich with market data
for fid, f in fixtures.items():
    key = (f['home_team'], f['game_date'])
    md = market_data.get(key)
    f['market'] = md

    # Compute aggregate metrics
    f['avg_negbin_wp'] = statistics.mean(f['negbin_wps'])
    f['max_negbin_wp'] = max(f['negbin_wps'])
    f['min_negbin_wp'] = min(f['negbin_wps'])
    f['avg_mu_home'] = statistics.mean(f['mu_homes'])
    f['avg_mu_away'] = statistics.mean(f['mu_aways'])
    f['total_mu'] = f['avg_mu_home'] + f['avg_mu_away']
    f['mu_diff'] = abs(f['avg_mu_home'] - f['avg_mu_away'])
    f['avg_edge_pp'] = statistics.mean(f['edge_pps'])
    f['n_legs'] = len(f['legs'])
    f['leg_win_rate'] = f['wins'] / (f['wins'] + f['losses'])
    f['is_busted'] = f['leg_win_rate'] < 0.5
    f['reuse_count'] = len(f['legs'])

    # Market disagreement
    if md and md['market_home_prob']:
        # NegBin wp is for the specific side of the bet, not always "home"
        # We'll compute market-implied ML prob for the favorite side
        f['market_fav_prob'] = max(md['market_home_prob'], md['market_away_prob'])
        f['negbin_market_gap'] = abs(f['avg_negbin_wp'] - f['market_fav_prob'])
    else:
        f['market_fav_prob'] = None
        f['negbin_market_gap'] = None

print(f"Total unique fixtures: {len(fixtures)}")
print(f"Fixtures with market data: {sum(1 for f in fixtures.values() if f['market'])}")

# ============================================================
# ANALYSIS 1: Cross-Estimator Disagreement
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 1: CROSS-ESTIMATOR DISAGREEMENT (NegBin vs Market)")
print("="*80)

# Since ATS model columns are all equal to negbin_wp on these legs,
# the only independent estimator available is the market ML line.
# We compare NegBin's confidence vs market-implied probability.

HIGH_CONF_THRESHOLD = 0.80

high_conf = {fid: f for fid, f in fixtures.items() if f['avg_negbin_wp'] >= HIGH_CONF_THRESHOLD}
hc_with_market = {fid: f for fid, f in high_conf.items() if f['negbin_market_gap'] is not None}

busted_hc = {fid: f for fid, f in hc_with_market.items() if f['is_busted']}
winning_hc = {fid: f for fid, f in hc_with_market.items() if not f['is_busted']}

print(f"\nHigh-confidence fixtures (NegBin WP >= {HIGH_CONF_THRESHOLD*100:.0f}%): {len(high_conf)}")
print(f"  With market data: {len(hc_with_market)}")
print(f"  BUSTED (leg WR < 50%): {len(busted_hc)}")
print(f"  WINNING (leg WR >= 50%): {len(winning_hc)}")

if busted_hc and winning_hc:
    bust_gaps = [f['negbin_market_gap'] for f in busted_hc.values()]
    win_gaps = [f['negbin_market_gap'] for f in winning_hc.values()]

    print(f"\n--- NegBin-Market Disagreement Gap ---")
    print(f"  BUSTED  mean: {statistics.mean(bust_gaps)*100:.1f}pp  median: {statistics.median(bust_gaps)*100:.1f}pp  (n={len(bust_gaps)})")
    print(f"  WINNING mean: {statistics.mean(win_gaps)*100:.1f}pp  median: {statistics.median(win_gaps)*100:.1f}pp  (n={len(win_gaps)})")

    u, p = mann_whitney_u(bust_gaps, win_gaps)
    d = cohens_d(bust_gaps, win_gaps)
    print(f"  Mann-Whitney p = {p:.4f}" if p else "  Mann-Whitney: insufficient data")
    print(f"  Cohen's d = {d:.3f}" if d else "  Cohen's d: N/A")

    sep = statistics.mean(bust_gaps) - statistics.mean(win_gaps)
    print(f"  Separation: {sep*100:+.1f}pp (positive = busted fixtures had MORE disagreement)")

    # Also check: NegBin WP vs market on SPECIFIC side
    # NegBin may overstate confidence relative to market
    bust_negbin = [f['avg_negbin_wp'] for f in busted_hc.values()]
    win_negbin = [f['avg_negbin_wp'] for f in winning_hc.values()]
    bust_market = [f['market_fav_prob'] for f in busted_hc.values()]
    win_market = [f['market_fav_prob'] for f in winning_hc.values()]

    print(f"\n--- NegBin WP Level ---")
    print(f"  BUSTED  mean NegBin WP: {statistics.mean(bust_negbin)*100:.1f}%  market fav: {statistics.mean(bust_market)*100:.1f}%")
    print(f"  WINNING mean NegBin WP: {statistics.mean(win_negbin)*100:.1f}%  market fav: {statistics.mean(win_market)*100:.1f}%")

    # NegBin overconfidence = negbin_wp - market_fav_prob
    bust_overconf = [f['avg_negbin_wp'] - f['market_fav_prob'] for f in busted_hc.values()]
    win_overconf = [f['avg_negbin_wp'] - f['market_fav_prob'] for f in winning_hc.values()]
    print(f"\n--- NegBin Overconfidence (NegBin WP - Market Prob) ---")
    print(f"  BUSTED  mean: {statistics.mean(bust_overconf)*100:+.1f}pp  median: {statistics.median(bust_overconf)*100:+.1f}pp")
    print(f"  WINNING mean: {statistics.mean(win_overconf)*100:+.1f}pp  median: {statistics.median(win_overconf)*100:+.1f}pp")
    u2, p2 = mann_whitney_u(bust_overconf, win_overconf)
    d2 = cohens_d(bust_overconf, win_overconf)
    print(f"  Mann-Whitney p = {p2:.4f}" if p2 else "  insufficient data")
    print(f"  Cohen's d = {d2:.3f}" if d2 else "")

# Detail table: bad-day busted fixtures
print(f"\n--- Busted High-Confidence Fixtures Detail ---")
print(f"{'Date':<12} {'Home':<22} {'Away':<22} {'NegBin':<8} {'Mkt':<8} {'Gap':<8} {'LegWR':<8} {'N':<5} {'Score'}")
bad_days = ['2026-06-30', '2026-07-01', '2026-07-03']
for fid, f in sorted(busted_hc.items(), key=lambda x: (x[1]['game_date'], -x[1]['negbin_market_gap'])):
    mkt = f'{f["market_fav_prob"]*100:.0f}%' if f['market_fav_prob'] else 'N/A'
    gap = f'{f["negbin_market_gap"]*100:.0f}pp' if f['negbin_market_gap'] else 'N/A'
    score = f'{f["home_score"]:.0f}-{f["away_score"]:.0f}' if f['home_score'] and f['away_score'] else '?'
    print(f'{f["game_date"]:<12} {(f["home_team"] or "?"):<22} {(f["away_team"] or "?"):<22} '
          f'{f["avg_negbin_wp"]*100:.0f}%     {mkt:<8} {gap:<8} {f["leg_win_rate"]*100:.0f}%     {f["n_legs"]:<5} {score}')

# Also show ALL fixtures on those bad days
print(f"\n--- ALL Fixtures on Bad Days (sorted by outcome) ---")
print(f"{'Date':<12} {'Home':<22} {'Away':<22} {'NegBin':<8} {'Mkt':<8} {'Gap':<8} {'LegWR':<8} {'N':<5} {'Status'}")
for fid, f in sorted(fixtures.items(), key=lambda x: (x[1]['game_date'], x[1]['leg_win_rate'])):
    if f['game_date'] not in bad_days:
        continue
    mkt = f'{f["market_fav_prob"]*100:.0f}%' if f.get('market_fav_prob') else 'N/A'
    gap = f'{f["negbin_market_gap"]*100:.0f}pp' if f.get('negbin_market_gap') else 'N/A'
    tag = 'BUST' if f['is_busted'] else 'OK'
    print(f'{f["game_date"]:<12} {(f["home_team"] or "?"):<22} {(f["away_team"] or "?"):<22} '
          f'{f["avg_negbin_wp"]*100:.0f}%     {mkt:<8} {gap:<8} {f["leg_win_rate"]*100:.0f}%     {f["n_legs"]:<5} {tag}')


# ============================================================
# ANALYSIS 2: Variance-Weighted Fixture Exposure Cap
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 2: VARIANCE-WEIGHTED FIXTURE EXPOSURE CAP")
print("="*80)

# Build bet-level data with fixture variance info
bc2 = bconn.cursor()
bc2.execute("""
SELECT b.id as bet_id, b.game_date, b.odds, b.status, b.actual_profit, b.legs as n_legs,
       b.source
FROM mock_bets b
WHERE b.cohort = 'clean' AND b.source LIKE 'shadow_negbin%'
  AND b.status IN ('SETTLED_WIN','SETTLED_LOSS','WON','LOST')
ORDER BY b.game_date
""")
all_bets = [dict(r) for r in bc2.fetchall()]

# For each bet, get its fixture variance contributions
bet_fixtures = defaultdict(list)
for leg in all_legs:
    fid = leg['fixture_id']
    f = fixtures[fid]
    bet_fixtures[leg['bet_id']].append({
        'fixture_id': fid,
        'total_mu': f['total_mu'],
        'mu_diff': f['mu_diff'],
        'var_proxy': f['total_mu'],  # higher total = higher variance
    })

def simulate_cap(bets, bet_fixtures_map, fixtures_data, cap_type, cap_value):
    """Simulate a fixture exposure cap.
    cap_type: 'count' or 'risk'
    For 'count': cap_value = max reuses per fixture per day
    For 'risk': cap_value = max risk_contribution per fixture per day
    """
    daily_fixture_usage = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'risk': 0}))
    accepted = []
    rejected = 0

    for bet in bets:
        bid = bet['bet_id']
        if bid not in bet_fixtures_map:
            continue

        gd = bet['game_date']
        fx_list = bet_fixtures_map[bid]

        # Check if this bet would exceed any fixture cap
        would_exceed = False
        for fx in fx_list:
            fid = fx['fixture_id']
            usage = daily_fixture_usage[gd][fid]
            if cap_type == 'count':
                if usage['count'] + 1 > cap_value:
                    would_exceed = True
                    break
            elif cap_type == 'risk':
                added_risk = fx['var_proxy']
                if usage['risk'] + added_risk > cap_value:
                    would_exceed = True
                    break

        if would_exceed:
            rejected += 1
            continue

        # Accept
        for fx in fx_list:
            fid = fx['fixture_id']
            daily_fixture_usage[gd][fid]['count'] += 1
            daily_fixture_usage[gd][fid]['risk'] += fx['var_proxy']

        is_win = bet['status'] in ('SETTLED_WIN', 'WON')
        # Shadow bets: actual_profit is always 0. Compute from odds.
        # $10 unit size.
        if is_win:
            profit = (bet['odds'] - 1) * 10
        else:
            profit = -10
        accepted.append({
            'bet_id': bid,
            'game_date': gd,
            'profit': profit,
            'is_win': is_win,
        })

    if len(accepted) < 5:
        return None

    # Compute metrics
    total_pnl = sum(a['profit'] for a in accepted)
    n_parlays = len(accepted)
    n_wins = sum(1 for a in accepted if a['is_win'])
    roi = total_pnl / (n_parlays * 10) if n_parlays > 0 else 0

    # Daily PnL for Sharpe and drawdown
    daily_pnl = defaultdict(float)
    for a in accepted:
        daily_pnl[a['game_date']] += a['profit']

    daily_vals = list(daily_pnl.values())
    if len(daily_vals) > 1:
        sharpe = statistics.mean(daily_vals) / statistics.stdev(daily_vals) if statistics.stdev(daily_vals) > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cumsum = []
    running = 0
    for d in sorted(daily_pnl.keys()):
        running += daily_pnl[d]
        cumsum.append(running)
    peak = cumsum[0]
    max_dd = 0
    for v in cumsum:
        peak = max(peak, v)
        dd = peak - v
        max_dd = max(max_dd, dd)

    return {
        'n_parlays': n_parlays,
        'n_wins': n_wins,
        'win_rate': n_wins / n_parlays if n_parlays > 0 else 0,
        'roi': roi,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_pnl': total_pnl,
        'rejected': rejected,
    }

# Run simulations
print(f"\nTotal bets in pool: {len(all_bets)}")
print(f"Bets with fixture data: {sum(1 for b in all_bets if b['bet_id'] in bet_fixtures)}")

# No cap baseline
baseline = simulate_cap(all_bets, bet_fixtures, fixtures, 'count', 9999)
if baseline:
    print(f"\n{'Cap Type':<25} {'Parlays':<10} {'WinR':<8} {'ROI':<10} {'MaxDD':<10} {'Sharpe':<8} {'PnL':<12} {'Rejected'}")
    print("-" * 100)
    print(f"{'No cap':<25} {baseline['n_parlays']:<10} {baseline['win_rate']*100:.0f}%     {baseline['roi']*100:+.1f}%    ${baseline['max_dd']:.0f}      {baseline['sharpe']:.3f}    ${baseline['total_pnl']:.0f}        0")

# Count caps
for cap in [2, 3, 5, 8, 10, 15]:
    r = simulate_cap(all_bets, bet_fixtures, fixtures, 'count', cap)
    if r:
        print(f"{'Count cap=' + str(cap):<25} {r['n_parlays']:<10} {r['win_rate']*100:.0f}%     {r['roi']*100:+.1f}%    ${r['max_dd']:.0f}      {r['sharpe']:.3f}    ${r['total_pnl']:.0f}        {r['rejected']}")

# Risk caps (total_mu-weighted)
# Typical total_mu is around 8-10 for MLB
avg_total_mu = statistics.mean([f['total_mu'] for f in fixtures.values()])
print(f"\nAvg fixture total_mu: {avg_total_mu:.1f}")
for risk_cap in [20, 40, 60, 80, 100, 150, 200]:
    r = simulate_cap(all_bets, bet_fixtures, fixtures, 'risk', risk_cap)
    if r:
        print(f"{'Risk cap=' + str(risk_cap):<25} {r['n_parlays']:<10} {r['win_rate']*100:.0f}%     {r['roi']*100:+.1f}%    ${r['max_dd']:.0f}      {r['sharpe']:.3f}    ${r['total_pnl']:.0f}        {r['rejected']}")

# Compare: at matched parlays, which is better?
print(f"\n--- Comparison at Matched Volume (deterministic order) ---")
# Find count cap that gives ~similar parlays to risk cap=60
for cap_type, cap_vals in [('count', [2,3,5,8,10]), ('risk', [20,40,60,80,100])]:
    for cv in cap_vals:
        r = simulate_cap(all_bets, bet_fixtures, fixtures, cap_type, cv)
        if r and 200 <= r['n_parlays'] <= 600:
            eff = r['total_pnl'] / r['max_dd'] if r['max_dd'] > 0 else 0
            print(f"  {cap_type}={cv}: {r['n_parlays']} parlays, ROI={r['roi']*100:+.1f}%, PnL/DD={eff:.2f}")


# Randomized cap simulation (average over N trials to remove ordering bias)
import random
random.seed(42)
N_TRIALS = 200

print(f"\n--- Randomized Cap Simulation ({N_TRIALS} trials, averaged) ---")
print(f"{'Cap Type':<25} {'Parlays':<10} {'WinR':<8} {'ROI':<12} {'MaxDD':<10} {'Sharpe':<8} {'PnL'}")
print("-" * 90)

for cap_type, cap_vals in [('count', [3, 5, 8, 10, 15, 20]), ('risk', [30, 50, 80, 100, 150])]:
    for cv in cap_vals:
        trial_results = []
        for trial in range(N_TRIALS):
            shuffled = all_bets.copy()
            random.shuffle(shuffled)
            r = simulate_cap(shuffled, bet_fixtures, fixtures, cap_type, cv)
            if r:
                trial_results.append(r)
        if trial_results:
            avg_parlays = statistics.mean([r['n_parlays'] for r in trial_results])
            avg_wr = statistics.mean([r['win_rate'] for r in trial_results])
            avg_roi = statistics.mean([r['roi'] for r in trial_results])
            avg_dd = statistics.mean([r['max_dd'] for r in trial_results])
            avg_sharpe = statistics.mean([r['sharpe'] for r in trial_results])
            avg_pnl = statistics.mean([r['total_pnl'] for r in trial_results])
            print(f"{cap_type + '=' + str(cv):<25} {avg_parlays:<10.0f} {avg_wr*100:.0f}%     {avg_roi*100:+.1f}%      ${avg_dd:.0f}      {avg_sharpe:.3f}    ${avg_pnl:.0f}")

# Per-day analysis: which days benefit most from capping?
print(f"\n--- Per-Day Impact of Count Cap=5 (randomized, 200 trials) ---")
from collections import defaultdict as dd2
day_pnl_nocap = defaultdict(float)
day_pnl_capped = defaultdict(list)

for bet in all_bets:
    is_win = bet['status'] in ('SETTLED_WIN', 'WON')
    profit = (bet['odds'] - 1) * 10 if is_win else -10
    day_pnl_nocap[bet['game_date']] += profit

for trial in range(N_TRIALS):
    shuffled = all_bets.copy()
    random.shuffle(shuffled)
    daily_fixture_usage = defaultdict(lambda: defaultdict(int))
    day_profit = defaultdict(float)
    for bet in shuffled:
        bid = bet['bet_id']
        if bid not in bet_fixtures:
            continue
        gd = bet['game_date']
        would_exceed = False
        for fx in bet_fixtures[bid]:
            if daily_fixture_usage[gd][fx['fixture_id']] + 1 > 5:
                would_exceed = True
                break
        if would_exceed:
            continue
        for fx in bet_fixtures[bid]:
            daily_fixture_usage[gd][fx['fixture_id']] += 1
        is_win = bet['status'] in ('SETTLED_WIN', 'WON')
        profit = (bet['odds'] - 1) * 10 if is_win else -10
        day_profit[gd] += profit
    for gd, p in day_profit.items():
        day_pnl_capped[gd].append(p)

print(f"{'Date':<14} {'NoCap PnL':<14} {'Capped PnL (avg)':<18} {'Delta':<12} {'Better?'}")
for gd in sorted(day_pnl_nocap.keys()):
    nc = day_pnl_nocap[gd]
    cap_avg = statistics.mean(day_pnl_capped[gd]) if gd in day_pnl_capped else 0
    delta = cap_avg - nc
    better = 'YES' if cap_avg > nc else 'no'
    print(f"{gd:<14} ${nc:+.0f}         ${cap_avg:+.0f}             ${delta:+.0f}         {better}")

# ============================================================
# ANALYSIS 3: Feature-Space Outlier Check
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 3: FEATURE-SPACE OUTLIER CHECK")
print("="*80)

# Compute distribution stats across ALL fixtures
all_total_mu = [f['total_mu'] for f in fixtures.values()]
all_mu_diff = [f['mu_diff'] for f in fixtures.values()]
all_avg_wp = [f['avg_negbin_wp'] for f in fixtures.values()]
all_edge = [f['avg_edge_pp'] for f in fixtures.values()]
all_mu_home = [f['avg_mu_home'] for f in fixtures.values()]
all_mu_away = [f['avg_mu_away'] for f in fixtures.values()]

features = {
    'total_mu': all_total_mu,
    'mu_diff': all_mu_diff,
    'avg_negbin_wp': all_avg_wp,
    'avg_edge_pp': all_edge,
    'mu_home': all_mu_home,
    'mu_away': all_mu_away,
}

stats = {}
for feat, vals in features.items():
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 1
    stats[feat] = (m, s)
    print(f"  {feat}: mean={m:.3f}, std={s:.3f}")

# Separate busted vs winning (ALL fixtures, not just high-conf)
busted_all = {fid: f for fid, f in fixtures.items() if f['is_busted']}
winning_all = {fid: f for fid, f in fixtures.items() if not f['is_busted']}

print(f"\nAll fixtures: {len(fixtures)} total, {len(busted_all)} busted, {len(winning_all)} winning")

# Z-scores for each fixture
print(f"\n--- Feature Z-Scores: Busted vs Winning ---")
print(f"{'Feature':<18} {'Bust mean-z':<14} {'Win mean-z':<14} {'Separation':<14} {'p-value'}")
print("-" * 72)

for feat in features:
    m, s = stats[feat]
    if s == 0:
        continue

    bust_z = []
    win_z = []

    for fid, f in fixtures.items():
        if feat == 'total_mu':
            val = f['total_mu']
        elif feat == 'mu_diff':
            val = f['mu_diff']
        elif feat == 'avg_negbin_wp':
            val = f['avg_negbin_wp']
        elif feat == 'avg_edge_pp':
            val = f['avg_edge_pp']
        elif feat == 'mu_home':
            val = f['avg_mu_home']
        elif feat == 'mu_away':
            val = f['avg_mu_away']
        else:
            continue

        z = abs(val - m) / s
        if f['is_busted']:
            bust_z.append(z)
        else:
            win_z.append(z)

    if bust_z and win_z:
        _, p = mann_whitney_u(bust_z, win_z)
        d = cohens_d(bust_z, win_z)
        print(f"{feat:<18} {statistics.mean(bust_z):.3f}         {statistics.mean(win_z):.3f}         "
              f"{statistics.mean(bust_z)-statistics.mean(win_z):+.3f}         "
              f"{p:.4f}" if p else f"{feat:<18} N/A")

# High-confidence only
print(f"\n--- Feature Z-Scores: HIGH-CONF Busted vs Winning ---")
print(f"{'Feature':<18} {'Bust mean-z':<14} {'Win mean-z':<14} {'Separation':<14} {'p-value'}")
print("-" * 72)

for feat in features:
    m, s = stats[feat]
    if s == 0:
        continue

    bust_z = []
    win_z = []

    for fid, f in high_conf.items():
        if feat == 'total_mu':
            val = f['total_mu']
        elif feat == 'mu_diff':
            val = f['mu_diff']
        elif feat == 'avg_negbin_wp':
            val = f['avg_negbin_wp']
        elif feat == 'avg_edge_pp':
            val = f['avg_edge_pp']
        elif feat == 'mu_home':
            val = f['avg_mu_home']
        elif feat == 'mu_away':
            val = f['avg_mu_away']
        else:
            continue

        z = abs(val - m) / s
        if f['is_busted']:
            bust_z.append(z)
        else:
            win_z.append(z)

    if bust_z and win_z:
        _, p = mann_whitney_u(bust_z, win_z)
        print(f"{feat:<18} {statistics.mean(bust_z):.3f}         {statistics.mean(win_z):.3f}         "
              f"{statistics.mean(bust_z)-statistics.mean(win_z):+.3f}         "
              f"{p:.4f}" if p else f"{feat:<18} N/A")

# Blowout proxy analysis
print(f"\n--- Blowout Proxy (mu_diff = closeness of game) ---")
# Low mu_diff = close game = more variance
# High mu_diff = blowout expected = less variance but more confident picks
bust_mudiff = [f['mu_diff'] for f in busted_all.values()]
win_mudiff = [f['mu_diff'] for f in winning_all.values()]
print(f"  BUSTED  mean mu_diff: {statistics.mean(bust_mudiff):.2f}  median: {statistics.median(bust_mudiff):.2f}")
print(f"  WINNING mean mu_diff: {statistics.mean(win_mudiff):.2f}  median: {statistics.median(win_mudiff):.2f}")
_, p = mann_whitney_u(bust_mudiff, win_mudiff)
print(f"  Mann-Whitney p = {p:.4f}" if p else "  insufficient data")

bust_totalmu = [f['total_mu'] for f in busted_all.values()]
win_totalmu = [f['total_mu'] for f in winning_all.values()]
print(f"\n  BUSTED  mean total_mu: {statistics.mean(bust_totalmu):.2f}  (higher = more runs = more variance)")
print(f"  WINNING mean total_mu: {statistics.mean(win_totalmu):.2f}")
_, p = mann_whitney_u(bust_totalmu, win_totalmu)
print(f"  Mann-Whitney p = {p:.4f}" if p else "  insufficient data")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("SUMMARY: KEY FINDINGS")
print("="*80)

print("""
Q1: Does NegBin-Market disagreement predict busts?
    - Compare the mean gap values above. If busted fixtures show significantly
      higher disagreement, this is an actionable signal.

Q2: Does variance-weighted capping beat count capping?
    - Compare ROI and Sharpe at matched parlay counts. If risk-capping
      produces better risk-adjusted returns, it should replace count caps.

Q3: Are busted fixtures feature-space outliers?
    - If busted fixtures have higher z-scores across features, a simple
      outlier filter could improve selection.
""")

# Close connections
bconn.close()
hconn.close()
