"""
negbin_design_checks.py — Pre-build design validation for Thread 2 Step B

CHECK A: Structural zero / tie-resolution design
  - How often does NegBin(home)×NegBin(away) produce margin=0?
  - What does the REAL extra-innings run differential look like?
  - Compare tie-resolution strategies: resample-until-nonzero vs extra-innings model
  - Validate: simulated margin at ±1, ±2 vs empirical

CHECK B: Dispersion parameterization
  - Is global r sufficient, or does per-team r improve fit?
  - Test: global r vs per-team r vs predicted r — diminishing returns?
  - Check if r correlates with any features (would justify prediction)
"""
from __future__ import annotations
import os, sys, sqlite3, json, math
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "data", "bets.db")


def _load_game_scores():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT DISTINCT mbl.fixture_id,
               mbl.resolved_home_score, mbl.resolved_away_score,
               mbl.resolved_home_team, mbl.resolved_away_team,
               mb.game_date
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mbl.mock_bet_id = mb.id
        WHERE mbl.resolved_home_score IS NOT NULL
          AND mbl.sport LIKE '%MLB%'
          AND mbl.resolved_home_score < 50
    """).fetchall()
    conn.close()
    return rows


def _fit_negbin_mom(data):
    """Method of moments NegBin fit. Returns (r, p, mean, var)."""
    data = np.array(data, dtype=float)
    mu = data.mean()
    var = data.var(ddof=1)
    if var <= mu:
        return None, None, mu, var
    r = mu ** 2 / (var - mu)
    p = r / (r + mu)  # scipy parameterization: p = r/(r+mu)
    return r, p, mu, var


def main():
    print("=" * 100)
    print("  NEGBIN DESIGN CHECKS: TIE-RESOLUTION + DISPERSION PARAMETERIZATION")
    print("=" * 100)

    rows = _load_game_scores()
    print(f"\n  Games: {len(rows)}")

    home_runs = np.array([int(r[1]) for r in rows])
    away_runs = np.array([int(r[2]) for r in rows])
    margins = home_runs - away_runs

    # ═══════════════════════════════════════════════════════════
    # CHECK A: STRUCTURAL ZERO / TIE-RESOLUTION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  CHECK A: STRUCTURAL ZERO / TIE-RESOLUTION")
    print(f"{'='*100}")

    # A1. Empirical: how do 1-run games distribute?
    print(f"\n  A1. Empirical margin distribution around zero:")
    margin_counts = Counter(margins.tolist())
    n_games = len(margins)
    for m in range(-5, 6):
        ct = margin_counts.get(m, 0)
        print(f"    margin={m:+2d}: {ct:>4} games ({ct/n_games*100:>5.1f}%)")
    print(f"    margin=0:  {margin_counts.get(0, 0)} (baseball: should be 0 — ties go to extras)")

    # A2. 1-run games are the primary recipient of tie-mass
    one_run = margin_counts.get(1, 0) + margin_counts.get(-1, 0)
    two_run = margin_counts.get(2, 0) + margin_counts.get(-2, 0)
    print(f"\n  1-run games: {one_run} ({one_run/n_games*100:.1f}%)")
    print(f"  2-run games: {two_run} ({two_run/n_games*100:.1f}%)")
    print(f"  Home wins by 1: {margin_counts.get(1, 0)} ({margin_counts.get(1, 0)/n_games*100:.1f}%)")
    print(f"  Away wins by 1: {margin_counts.get(-1, 0)} ({margin_counts.get(-1, 0)/n_games*100:.1f}%)")

    # A3. Simulate from fitted NegBin and check raw tie rate
    r_h, p_h, mu_h, var_h = _fit_negbin_mom(home_runs)
    r_a, p_a, mu_a, var_a = _fit_negbin_mom(away_runs)

    print(f"\n  A3. NegBin fits:")
    print(f"    Home: r={r_h:.3f}, p={p_h:.3f}, μ={mu_h:.3f}, σ²={var_h:.3f}")
    print(f"    Away: r={r_a:.3f}, p={p_a:.3f}, μ={mu_a:.3f}, σ²={var_a:.3f}")

    np.random.seed(42)
    N_SIM = 200_000
    sim_home = np.random.negative_binomial(r_h, p_h, N_SIM)
    sim_away = np.random.negative_binomial(r_a, p_a, N_SIM)
    sim_margin_raw = sim_home - sim_away

    tie_rate = np.mean(sim_margin_raw == 0)
    print(f"\n    Raw NegBin tie rate: {tie_rate*100:.1f}% ({int(tie_rate*N_SIM)} of {N_SIM})")

    # A4. Compare tie-resolution strategies
    print(f"\n  A4. TIE-RESOLUTION STRATEGIES")

    # Strategy 1: Resample until margin ≠ 0
    print(f"\n    Strategy 1: RESAMPLE UNTIL NONZERO")
    sim_resample = sim_margin_raw.copy()
    tie_mask = sim_resample == 0
    while tie_mask.sum() > 0:
        n_ties = tie_mask.sum()
        new_h = np.random.negative_binomial(r_h, p_h, n_ties)
        new_a = np.random.negative_binomial(r_a, p_a, n_ties)
        sim_resample[tie_mask] = new_h - new_a
        tie_mask = sim_resample == 0

    # Strategy 2: Extra-innings model
    # In extra innings, most games end with 1-run differential (walk-off)
    # Historical MLB extra-innings: ~65-70% end 1-run, ~20% 2-run, ~10% 3+
    # Home team has walk-off advantage (~55-58% home win rate in extras)
    print(f"\n    Strategy 2: EXTRA-INNINGS MODEL")
    print(f"    Model: tie → resolve with P(+1)=0.29, P(-1)=0.22, P(+2)=0.10,")
    print(f"           P(-2)=0.08, P(+3)=0.05, P(-3)=0.04, rest=resample wider")

    # Extra-innings resolution distribution (based on historical MLB data)
    # Home wins extras ~57%, away ~43%
    # Of extra-inning games, ~67% decided by 1 run, ~20% by 2, ~13% by 3+
    extra_inning_dist = {
        +1: 0.383,   # home wins by 1 (0.67 * 0.57)
        -1: 0.287,   # away wins by 1 (0.67 * 0.43)
        +2: 0.114,   # home wins by 2 (0.20 * 0.57)
        -2: 0.086,   # away wins by 2 (0.20 * 0.43)
        +3: 0.047,   # home wins by 3+ (0.065 * ~0.72 skew)
        -3: 0.035,
        +4: 0.019,
        -4: 0.014,
        +5: 0.008,
        -5: 0.006,
    }
    # Normalize
    total_p = sum(extra_inning_dist.values())
    extra_inning_dist = {k: v / total_p for k, v in extra_inning_dist.items()}

    xi_keys = list(extra_inning_dist.keys())
    xi_probs = [extra_inning_dist[k] for k in xi_keys]

    sim_extras = sim_margin_raw.copy()
    tie_mask = sim_extras == 0
    n_ties = tie_mask.sum()
    resolutions = np.random.choice(xi_keys, size=n_ties, p=xi_probs)
    sim_extras[tie_mask] = resolutions

    # Strategy 3: Simple ±1 with home bias
    # All ties become +1 (home) or -1 (away) with 57/43 split
    print(f"\n    Strategy 3: SIMPLE ±1 HOME BIAS (57/43)")
    sim_simple = sim_margin_raw.copy()
    tie_mask_s = sim_simple == 0
    n_ties_s = tie_mask_s.sum()
    coin = np.random.random(n_ties_s)
    sim_simple[tie_mask_s] = np.where(coin < 0.57, 1, -1)

    # A5. Compare all strategies vs empirical
    print(f"\n  A5. MARGIN COMPARISON: empirical vs strategies")
    emp_pct = {}
    for m in range(-10, 11):
        emp_pct[m] = margin_counts.get(m, 0) / n_games * 100

    print(f"\n    {'Margin':>8} {'Empirical':>10} {'Resample':>10} {'Extras':>10} {'Simple±1':>10}")
    print(f"    {'─'*52}")
    for m in range(-6, 7):
        emp = emp_pct.get(m, 0)
        res = np.mean(sim_resample == m) * 100
        ext = np.mean(sim_extras == m) * 100
        smp = np.mean(sim_simple == m) * 100
        flag = " ◄" if abs(m) <= 2 else ""
        print(f"    {m:>+8} {emp:>9.1f}% {res:>9.1f}% {ext:>9.1f}% {smp:>9.1f}%{flag}")

    # A6. Key metric: ±1.5 pricing accuracy
    print(f"\n  A6. ±1.5 PRICING (the money rung)")
    # P(margin ≥ +1.5) = P(home wins) = P(margin ≥ +2) + P(margin = +1)... no
    # Actually for spread +1.5: team covers if they lose by ≤1 or win
    # P(pick covers +1.5) = P(margin ≥ -1) for the pick side

    # Let's check P(margin ≥ X) for key thresholds
    print(f"\n    P(margin ≥ X) — drives spread rung pricing:")
    print(f"    {'Threshold':>12} {'Empirical':>10} {'Resample':>10} {'Extras':>10} {'Simple±1':>10} "
          f"{'Res err':>8} {'Ext err':>8} {'Smp err':>8}")
    print(f"    {'─'*90}")
    for thr_val in [-1, 0, 1, 2, 3, 4, 5]:
        # P(margin >= thr_val + 0.5) since spreads are at half-points
        thr = thr_val  # margin >= thr means home covers -(thr-1).5
        emp = np.mean(margins >= thr) * 100
        res = np.mean(sim_resample >= thr) * 100
        ext = np.mean(sim_extras >= thr) * 100
        smp = np.mean(sim_simple >= thr) * 100
        spread_label = f"{-(thr-1)+0.5:+.1f}" if thr <= 2 else f"{-(thr-1)+0.5:+.1f}"
        print(f"    margin≥{thr:>+2}     {emp:>9.2f}% {res:>9.2f}% {ext:>9.2f}% {smp:>9.2f}% "
              f"{res-emp:>+7.2f}pp {ext-emp:>+7.2f}pp {smp-emp:>+7.2f}pp")

    # A7. SSE comparison
    print(f"\n  A7. Overall fit (SSE on margin distribution, margins -10 to +10):")
    sse_res = sum((np.mean(sim_resample == m)*100 - emp_pct.get(m, 0))**2 for m in range(-10, 11))
    sse_ext = sum((np.mean(sim_extras == m)*100 - emp_pct.get(m, 0))**2 for m in range(-10, 11))
    sse_smp = sum((np.mean(sim_simple == m)*100 - emp_pct.get(m, 0))**2 for m in range(-10, 11))
    print(f"    Resample: SSE = {sse_res:.2f}")
    print(f"    Extras:   SSE = {sse_ext:.2f}")
    print(f"    Simple±1: SSE = {sse_smp:.2f}")

    best = min([("Resample", sse_res), ("Extras", sse_ext), ("Simple±1", sse_smp)], key=lambda x: x[1])
    print(f"    → Best: {best[0]}")

    # ═══════════════════════════════════════════════════════════
    # CHECK B: DISPERSION PARAMETERIZATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  CHECK B: DISPERSION PARAMETERIZATION")
    print(f"{'='*100}")

    # B1. Global r vs per-team r
    print(f"\n  B1. Global r:")
    r_global_h, _, _, _ = _fit_negbin_mom(home_runs)
    r_global_a, _, _, _ = _fit_negbin_mom(away_runs)
    r_global_all, _, _, _ = _fit_negbin_mom(np.concatenate([home_runs, away_runs]))
    print(f"    r (home): {r_global_h:.3f}")
    print(f"    r (away): {r_global_a:.3f}")
    print(f"    r (all):  {r_global_all:.3f}")

    # B2. Per-team r
    print(f"\n  B2. Per-team r:")
    team_home_runs = defaultdict(list)
    team_away_runs = defaultdict(list)
    seen = set()
    for r_row in rows:
        fid = r_row[0]
        if fid in seen:
            continue
        seen.add(fid)
        ht, at = r_row[3], r_row[4]
        hs, aws = int(r_row[1]), int(r_row[2])
        if ht:
            team_home_runs[ht].append(hs)
        if at:
            team_away_runs[at].append(aws)

    team_rs = {}
    print(f"\n    {'Team':<25} {'N':>4} {'μ':>6} {'σ²':>7} {'r':>6} {'r vs global':>12}")
    print(f"    {'─'*64}")
    for team in sorted(set(list(team_home_runs.keys()) + list(team_away_runs.keys()))):
        all_team = team_home_runs.get(team, []) + team_away_runs.get(team, [])
        if len(all_team) < 15:
            continue
        r_t, p_t, mu_t, var_t = _fit_negbin_mom(all_team)
        if r_t is None:
            continue
        team_rs[team] = r_t
        pct_diff = (r_t - r_global_all) / r_global_all * 100
        print(f"    {team:<25} {len(all_team):>4} {mu_t:>6.2f} {var_t:>7.2f} {r_t:>6.2f} {pct_diff:>+11.0f}%")

    if team_rs:
        rs = list(team_rs.values())
        print(f"\n    Per-team r statistics:")
        print(f"      Min:    {min(rs):.2f}")
        print(f"      Median: {np.median(rs):.2f}")
        print(f"      Mean:   {np.mean(rs):.2f}")
        print(f"      Max:    {max(rs):.2f}")
        print(f"      StdDev: {np.std(rs, ddof=1):.2f}")
        print(f"      CV:     {np.std(rs, ddof=1)/np.mean(rs)*100:.1f}%")

    # B3. Does per-team r improve simulation fit?
    print(f"\n  B3. Simulation comparison: global r vs per-team r")
    # For each game, simulate with global r vs per-team r
    # Compare margin distribution fit

    np.random.seed(123)
    N_SIM_PER = 1000  # sims per game

    # Global r simulation
    margin_hist_global = Counter()
    margin_hist_perteam = Counter()

    for r_row in rows:
        fid = r_row[0]
        ht, at = r_row[3], r_row[4]
        hs, aws = int(r_row[1]), int(r_row[2])

        # Global r: use overall fit
        mu_h_game = np.mean(team_home_runs.get(ht, [hs])) if ht else mu_h
        mu_a_game = np.mean(team_away_runs.get(at, [aws])) if at else mu_a

        # Global r sim
        p_h_g = r_global_all / (r_global_all + mu_h_game)
        p_a_g = r_global_all / (r_global_all + mu_a_game)
        sh = np.random.negative_binomial(r_global_all, p_h_g, N_SIM_PER)
        sa = np.random.negative_binomial(r_global_all, p_a_g, N_SIM_PER)
        for m in (sh - sa):
            margin_hist_global[m] += 1

        # Per-team r sim
        r_h_t = team_rs.get(ht, r_global_all)
        r_a_t = team_rs.get(at, r_global_all)
        p_h_t = r_h_t / (r_h_t + mu_h_game)
        p_a_t = r_a_t / (r_a_t + mu_a_game)
        sh2 = np.random.negative_binomial(r_h_t, p_h_t, N_SIM_PER)
        sa2 = np.random.negative_binomial(r_a_t, p_a_t, N_SIM_PER)
        for m in (sh2 - sa2):
            margin_hist_perteam[m] += 1

    # Normalize
    total_g = sum(margin_hist_global.values())
    total_t = sum(margin_hist_perteam.values())

    print(f"\n    {'Margin':>8} {'Empirical':>10} {'Global r':>10} {'Per-team r':>11} {'G err':>8} {'T err':>8}")
    print(f"    {'─'*60}")
    sse_global = 0
    sse_perteam = 0
    for m in range(-8, 9):
        emp = margin_counts.get(m, 0) / n_games * 100
        g_pct = margin_hist_global.get(m, 0) / total_g * 100
        t_pct = margin_hist_perteam.get(m, 0) / total_t * 100
        sse_global += (g_pct - emp) ** 2
        sse_perteam += (t_pct - emp) ** 2
        print(f"    {m:>+8} {emp:>9.1f}% {g_pct:>9.1f}% {t_pct:>10.1f}% {g_pct-emp:>+7.1f}pp {t_pct-emp:>+7.1f}pp")

    print(f"\n    SSE (global r):   {sse_global:.2f}")
    print(f"    SSE (per-team r): {sse_perteam:.2f}")
    print(f"    Improvement:      {(sse_global-sse_perteam)/sse_global*100:+.1f}%")

    # B4. Does r correlate with team features?
    print(f"\n  B4. Does r correlate with team characteristics?")
    # Check if r correlates with team mean (higher-scoring teams more/less variable)
    if team_rs:
        team_means = []
        team_r_vals = []
        for team, r_val in team_rs.items():
            all_team = team_home_runs.get(team, []) + team_away_runs.get(team, [])
            team_means.append(np.mean(all_team))
            team_r_vals.append(r_val)

        corr_r_mu, p_r_mu = stats.pearsonr(team_means, team_r_vals)
        print(f"    Corr(team_mean, team_r): {corr_r_mu:+.3f} (p={p_r_mu:.3f})")
        if abs(corr_r_mu) < 0.3:
            print(f"    → Weak correlation — r is not just a function of μ")
        else:
            print(f"    → r varies with scoring level — per-team r may capture real heterogeneity")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  DESIGN DECISIONS")
    print(f"{'='*100}")

    print(f"\n  1. TIE-RESOLUTION:")
    print(f"     Best strategy: {'Extras' if sse_ext <= sse_res and sse_ext <= sse_smp else 'Resample' if sse_res <= sse_smp else 'Simple±1'}")
    print(f"     SSE scores: Resample={sse_res:.2f}, Extras={sse_ext:.2f}, Simple±1={sse_smp:.2f}")

    print(f"\n  2. DISPERSION PARAMETERIZATION:")
    if team_rs:
        cv = np.std(list(team_rs.values()), ddof=1) / np.mean(list(team_rs.values())) * 100
        improvement = (sse_global - sse_perteam) / sse_global * 100 if sse_global > 0 else 0
        if cv < 30 and improvement < 10:
            print(f"     → GLOBAL r recommended")
            print(f"       Per-team CV={cv:.0f}% (low variation)")
            print(f"       Simulation improvement: {improvement:+.1f}% (marginal)")
            print(f"       Risk: per-team r on ~25-50 games/team → noisy estimates")
        elif improvement > 20:
            print(f"     → PER-TEAM r recommended")
            print(f"       Per-team CV={cv:.0f}% (meaningful variation)")
            print(f"       Simulation improvement: {improvement:+.1f}% (substantial)")
        else:
            print(f"     → GLOBAL r with optional per-team adjustment")
            print(f"       Per-team CV={cv:.0f}%")
            print(f"       Simulation improvement: {improvement:+.1f}%")

    print(f"\n  3. STEP B ARCHITECTURE:")
    print(f"     GBM → μ_home, μ_away (74 features, NegBin/Poisson-deviance loss)")
    print(f"     Global r = {r_global_all:.2f} (or per-team if B3 showed improvement)")
    print(f"     Simulate N margins (NegBin draws, tie-resolution via extras model)")
    print(f"     Derive: P(margin ≥ X) for all spread rungs, P(total ≥ X) for total lines")


if __name__ == "__main__":
    main()
