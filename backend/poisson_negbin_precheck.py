"""
poisson_negbin_precheck.py — Thread 2 Step A

Empirical pre-check: decide the distribution model architecture
BEFORE building. Goal: simplest model the data supports.

Check 1: Poisson sufficient, or overdispersed? (dispersion ratio + tail fit)
Check 2: Margin/Skellam fit (drives spread pricing)
Check 3: Home/away run correlation (independent assumption check)
Check 4: Context stability (is overdispersion uniform or concentrated?)

Analysis only — no model build.
"""
from __future__ import annotations
import os, sys, sqlite3, json, math
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from scipy.special import gammaln

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "data", "bets.db")


def _load_game_scores():
    """Load unique game scores from settled mock_bet_legs."""
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


def _negbin_loglik(data, r, p):
    return sum(stats.nbinom.logpmf(int(x), r, p) for x in data)


def _fit_negbin_mle(data):
    """Fit NegBin via method of moments + grid search."""
    data = np.array(data, dtype=float)
    mu = data.mean()
    var = data.var(ddof=1)
    if var <= mu:
        return None, None, None
    r_mom = mu ** 2 / (var - mu)
    p_mom = mu / var
    best_ll = -1e18
    best_r, best_p = r_mom, p_mom
    for r_mult in np.linspace(0.5, 2.0, 31):
        r_try = r_mom * r_mult
        if r_try <= 0:
            continue
        p_try = r_try / (r_try + mu)
        if not (0 < p_try < 1):
            continue
        ll = _negbin_loglik(data, r_try, p_try)
        if ll > best_ll:
            best_ll = ll
            best_r, best_p = r_try, p_try
    return best_r, best_p, best_ll


def _skellam_pmf(k, mu1, mu2):
    """PMF of Skellam distribution (difference of two independent Poissons)."""
    # P(X=k) = e^-(mu1+mu2) * (mu1/mu2)^(k/2) * I_|k|(2*sqrt(mu1*mu2))
    from scipy.special import iv as bessel_iv
    t = 2 * math.sqrt(mu1 * mu2)
    return math.exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2.0) * float(bessel_iv(abs(k), t))


def main():
    print("=" * 100)
    print("  THREAD 2 STEP A: POISSON vs NEGBIN EMPIRICAL PRE-CHECK")
    print("=" * 100)

    rows = _load_game_scores()
    print(f"\n  Unique games: {len(rows)}")
    dates = [r[5] for r in rows if r[5]]
    print(f"  Date range: {min(dates)} to {max(dates)}")

    home_runs = np.array([int(r[1]) for r in rows])
    away_runs = np.array([int(r[2]) for r in rows])
    all_runs = np.concatenate([home_runs, away_runs])
    margins = home_runs - away_runs
    totals = home_runs + away_runs

    # ═══════════════════════════════════════════════════════════
    # CHECK 1: Poisson sufficient, or overdispersed?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  CHECK 1: OVERDISPERSION TEST")
    print(f"{'='*100}")

    print(f"\n  A. Dispersion ratios (var/mean — Poisson expects 1.0):")
    for label, data in [("Home runs", home_runs), ("Away runs", away_runs),
                         ("All team-games", all_runs)]:
        mu = data.mean()
        var = data.var(ddof=1)
        disp = var / mu
        n = len(data)
        # Cameron-Trivedi test: under H0 (Poisson), (x-μ)²-x has E=0
        ct_stat = np.mean((data - mu)**2 - data)
        ct_se = np.std((data - mu)**2 - data, ddof=1) / np.sqrt(n)
        ct_z = ct_stat / ct_se if ct_se > 0 else 0
        ct_p = 2 * stats.norm.sf(abs(ct_z))
        print(f"\n    {label} (n={n}):")
        print(f"      Mean={mu:.3f}  Var={var:.3f}  Disp={disp:.3f}")
        if disp > 1.0:
            print(f"      → Overdispersed by {(disp-1)*100:.1f}%")
        print(f"      Cameron-Trivedi: z={ct_z:.2f}, p={ct_p:.2e}")

    # B. TAIL fit: does Poisson under-predict high-scoring games?
    print(f"\n  B. Tail fit (team-game level):")
    mu_all = all_runs.mean()
    nb_r, nb_p, nb_ll = _fit_negbin_mle(all_runs)
    n = len(all_runs)

    print(f"     NegBin fit: r={nb_r:.3f}, p={nb_p:.3f}" if nb_r else "     NegBin: no overdispersion")

    print(f"\n    {'Threshold':>12} {'Obs freq':>10} {'Poisson':>10} {'NegBin':>10} {'Pois err':>10} {'NB err':>10}")
    print(f"    {'─'*62}")
    for thr in [0, 5, 7, 8, 10, 12, 15]:
        obs_pct = np.mean(all_runs >= thr) * 100
        pois_pct = (1 - stats.poisson.cdf(thr - 1, mu_all)) * 100 if thr > 0 else 100.0
        nb_pct = (1 - stats.nbinom.cdf(thr - 1, nb_r, nb_p)) * 100 if nb_r and thr > 0 else (100.0 if thr == 0 else pois_pct)
        print(f"    runs≥{thr:>2}     {obs_pct:>9.2f}% {pois_pct:>9.2f}% {nb_pct:>9.2f}% "
              f"{obs_pct-pois_pct:>+9.2f}pp {obs_pct-nb_pct:>+9.2f}pp")

    # C. Frequency table
    print(f"\n  C. Observed vs predicted frequencies:")
    counts = Counter(all_runs.tolist())

    print(f"    {'Runs':>6} {'Obs':>6} {'Obs%':>7} {'Pois%':>7} {'NB%':>7} {'P err':>7} {'NB err':>7}")
    print(f"    {'─'*55}")
    for k in range(min(max(counts.keys()) + 1, 16)):
        obs = counts.get(k, 0)
        obs_pct = obs / n * 100
        pois_pct = stats.poisson.pmf(k, mu_all) * 100
        nb_pct = stats.nbinom.pmf(k, nb_r, nb_p) * 100 if nb_r else pois_pct
        print(f"    {k:>6} {obs:>6} {obs_pct:>6.1f}% {pois_pct:>6.1f}% {nb_pct:>6.1f}% "
              f"{obs_pct-pois_pct:>+6.1f}pp {obs_pct-nb_pct:>+6.1f}pp")
    obs_over = int(np.sum(all_runs >= 16))
    print(f"    {'≥16':>6} {obs_over:>6} {obs_over/n*100:>6.1f}% "
          f"{(1-stats.poisson.cdf(15, mu_all))*100:>6.1f}% "
          f"{(1-stats.nbinom.cdf(15, nb_r, nb_p))*100:>6.1f}%")

    # D. AIC/BIC
    pois_ll = sum(stats.poisson.logpmf(int(x), mu_all) for x in all_runs)
    aic_pois = -2 * pois_ll + 2
    aic_nb = -2 * nb_ll + 4
    bic_pois = -2 * pois_ll + math.log(n)
    bic_nb = -2 * nb_ll + 2 * math.log(n)
    lr_stat = 2 * (nb_ll - pois_ll)
    lr_p = stats.chi2.sf(lr_stat, 1)

    print(f"\n  D. Model selection:")
    print(f"    Poisson LL: {pois_ll:.1f}   NegBin LL: {nb_ll:.1f}")
    print(f"    AIC: Poisson={aic_pois:.0f}  NegBin={aic_nb:.0f}  Δ={aic_pois-aic_nb:+.0f}")
    print(f"    BIC: Poisson={bic_pois:.0f}  NegBin={bic_nb:.0f}  Δ={bic_pois-bic_nb:+.0f}")
    print(f"    LR test: χ²={lr_stat:.2f}, p={lr_p:.2e}")

    # ═══════════════════════════════════════════════════════════
    # CHECK 2: Margin / Skellam fit
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  CHECK 2: MARGIN DISTRIBUTION / SKELLAM FIT")
    print(f"{'='*100}")

    mu_h = home_runs.mean()
    mu_a = away_runs.mean()
    print(f"\n  μ_home={mu_h:.3f}  μ_away={mu_a:.3f}")

    # Under Skellam (diff of two independent Poissons):
    # E[margin] = μ_h - μ_a, Var[margin] = μ_h + μ_a
    skel_mean = mu_h - mu_a
    skel_var = mu_h + mu_a
    obs_mean = margins.mean()
    obs_var = margins.var(ddof=1)
    print(f"  Margin mean: obs={obs_mean:+.3f}  Skellam={skel_mean:+.3f}")
    print(f"  Margin var:  obs={obs_var:.3f}  Skellam={skel_var:.3f}  excess={obs_var-skel_var:+.3f} ({(obs_var/skel_var-1)*100:+.1f}%)")

    # Empirical vs Skellam frequency
    print(f"\n  Margin frequency table:")
    margin_counts = Counter(margins.tolist())
    print(f"    {'Margin':>8} {'Obs':>6} {'Obs%':>7} {'Skellam%':>9} {'Err':>8}")
    print(f"    {'─'*42}")
    for k in range(-10, 11):
        obs = margin_counts.get(k, 0)
        obs_pct = obs / len(margins) * 100
        skel_pct = _skellam_pmf(k, mu_h, mu_a) * 100
        print(f"    {k:>+8} {obs:>6} {obs_pct:>6.1f}% {skel_pct:>8.1f}%  {obs_pct-skel_pct:>+7.1f}pp")

    # Deep-cushion tail check (what matters for alt-spreads)
    print(f"\n  Deep-rung tail check (drives +3.5/+4.5/+5.5 pricing):")
    print(f"    {'Condition':>20} {'Obs':>10} {'Skellam':>10} {'Error':>10}")
    print(f"    {'─'*52}")
    for thr in [3, 4, 5, 6, 7, 8]:
        # P(|margin| >= thr) — either direction
        obs_pct_abs = np.mean(np.abs(margins) >= thr) * 100
        skel_pct_abs = sum(_skellam_pmf(k, mu_h, mu_a) for k in range(-30, -thr + 1)) * 100
        skel_pct_abs += sum(_skellam_pmf(k, mu_h, mu_a) for k in range(thr, 31)) * 100
        print(f"    |margin|≥{thr}        {obs_pct_abs:>9.2f}% {skel_pct_abs:>9.2f}%  {obs_pct_abs-skel_pct_abs:>+9.2f}pp")

    # Directional: P(margin >= thr) for home covering
    print(f"\n    P(margin ≥ X), directional (home cover):")
    for thr in [2, 3, 4, 5, 6]:
        obs_pct = np.mean(margins >= thr) * 100
        skel_pct = sum(_skellam_pmf(k, mu_h, mu_a) for k in range(thr, 31)) * 100
        print(f"    margin≥{thr:>+2}:  obs={obs_pct:>6.2f}%  Skellam={skel_pct:>6.2f}%  err={obs_pct-skel_pct:>+6.2f}pp")

    # ═══════════════════════════════════════════════════════════
    # CHECK 3: Home/away run correlation
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  CHECK 3: HOME/AWAY RUN CORRELATION")
    print(f"{'='*100}")

    corr_pearson, p_pearson = stats.pearsonr(home_runs, away_runs)
    corr_spearman, p_spearman = stats.spearmanr(home_runs, away_runs)

    print(f"\n  Pearson r:  {corr_pearson:+.4f}  (p={p_pearson:.4f})")
    print(f"  Spearman ρ: {corr_spearman:+.4f}  (p={p_spearman:.4f})")

    # Shared-environment check: in high-total games, are BOTH teams scoring more?
    print(f"\n  Shared-environment evidence:")
    median_total = np.median(totals)
    high_mask = totals >= median_total + 2
    low_mask = totals <= median_total - 2
    if high_mask.sum() > 20 and low_mask.sum() > 20:
        print(f"    High-total games (total≥{int(median_total+2)}, n={high_mask.sum()}):")
        print(f"      Home: {home_runs[high_mask].mean():.2f}  Away: {away_runs[high_mask].mean():.2f}")
        print(f"    Low-total games (total≤{int(median_total-2)}, n={low_mask.sum()}):")
        print(f"      Home: {home_runs[low_mask].mean():.2f}  Away: {away_runs[low_mask].mean():.2f}")

    # If runs were independent, cov(home, away) = 0
    # Excess covariance = measure of shared environment
    cov_ha = np.cov(home_runs, away_runs)[0, 1]
    print(f"\n  Cov(home, away): {cov_ha:.3f}")
    print(f"  If independent: 0.000")
    if abs(corr_pearson) < 0.05:
        print(f"  → Correlation is negligible — independent assumption holds")
    elif abs(corr_pearson) < 0.15:
        print(f"  → Weak correlation — independent assumption is approximate")
    else:
        print(f"  → Meaningful correlation — shared-environment term may help")

    # Does the correlation inflate margin variance?
    # Var(H-A) = Var(H) + Var(A) - 2*Cov(H,A)
    # If cov > 0, margin variance is REDUCED (both go up/down together → smaller diff)
    # If cov < 0, margin variance is INFLATED
    var_h = home_runs.var(ddof=1)
    var_a = away_runs.var(ddof=1)
    expected_var_margin_ind = var_h + var_a
    expected_var_margin_corr = var_h + var_a - 2 * cov_ha
    print(f"\n  Var(margin) decomposition:")
    print(f"    Var(H) + Var(A) = {expected_var_margin_ind:.3f}")
    print(f"    - 2*Cov(H,A)   = {-2*cov_ha:+.3f}")
    print(f"    Expected:         {expected_var_margin_corr:.3f}")
    print(f"    Observed:         {obs_var:.3f}")

    # Correlation impact on totals
    # Var(H+A) = Var(H) + Var(A) + 2*Cov(H,A)
    expected_var_total_ind = var_h + var_a
    expected_var_total_corr = var_h + var_a + 2 * cov_ha
    obs_var_total = totals.var(ddof=1)
    print(f"\n  Var(total) decomposition:")
    print(f"    Var(H) + Var(A) = {expected_var_total_ind:.3f}")
    print(f"    + 2*Cov(H,A)   = {+2*cov_ha:+.3f}")
    print(f"    Expected:         {expected_var_total_corr:.3f}")
    print(f"    Observed:         {obs_var_total:.3f}")

    # ═══════════════════════════════════════════════════════════
    # CHECK 4: Context stability
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  CHECK 4: CONTEXT STABILITY")
    print(f"{'='*100}")

    # A. Monthly stability
    print(f"\n  A. Monthly dispersion:")
    monthly = defaultdict(list)
    for r in rows:
        month = r[5][:7] if r[5] else None
        if month:
            monthly[month].extend([int(r[1]), int(r[2])])

    print(f"    {'Month':>10} {'N':>6} {'Mean':>7} {'Var':>7} {'Disp':>7}")
    print(f"    {'─'*42}")
    for month in sorted(monthly.keys()):
        data = np.array(monthly[month])
        mu = data.mean()
        var = data.var(ddof=1)
        disp = var / mu
        print(f"    {month:>10} {len(data):>6} {mu:>7.3f} {var:>7.3f} {disp:>7.3f}")

    # B. Per-team dispersion distribution
    print(f"\n  B. Per-team dispersion (combined home+away):")
    team_runs = defaultdict(list)
    seen = set()
    for r in rows:
        fid = r[0]
        if fid in seen:
            continue
        seen.add(fid)
        ht, at = r[3], r[4]
        if ht:
            team_runs[ht].append(int(r[1]))
        if at:
            team_runs[at].append(int(r[2]))

    disps = []
    for team in sorted(team_runs.keys()):
        data = np.array(team_runs[team])
        if len(data) < 15:
            continue
        mu = data.mean()
        var = data.var(ddof=1)
        disps.append(var / mu)

    disps = np.array(disps)
    print(f"    Teams with ≥15 games: {len(disps)}")
    print(f"    Dispersion ratio — min={disps.min():.2f}  median={np.median(disps):.2f}  "
          f"mean={disps.mean():.2f}  max={disps.max():.2f}")
    print(f"    Teams with disp > 1.0: {np.sum(disps > 1.0)}/{len(disps)} ({np.mean(disps > 1.0)*100:.0f}%)")
    print(f"    Teams with disp > 1.5: {np.sum(disps > 1.5)}/{len(disps)} ({np.mean(disps > 1.5)*100:.0f}%)")

    # C. Is overdispersion from a few outlier parks/teams, or broad?
    # Compare: remove top-3 most overdispersed teams, recompute overall
    sorted_teams = sorted(team_runs.items(), key=lambda x: np.var(x[1], ddof=1)/max(np.mean(x[1]),0.01), reverse=True)
    top3_teams = [t[0] for t in sorted_teams[:3]]
    print(f"\n  C. Sensitivity: remove 3 most overdispersed teams ({', '.join(top3_teams)}):")
    remaining = [x for r in rows for x in [int(r[1]), int(r[2])]
                 if r[3] not in top3_teams and r[4] not in top3_teams]
    remaining = np.array(remaining)
    mu_rem = remaining.mean()
    var_rem = remaining.var(ddof=1)
    print(f"    Before: n={len(all_runs)}, disp={all_runs.var(ddof=1)/all_runs.mean():.3f}")
    print(f"    After:  n={len(remaining)}, disp={var_rem/mu_rem:.3f}")
    if var_rem / mu_rem > 1.05:
        print(f"    → Overdispersion is BROAD, not driven by outlier teams")
    else:
        print(f"    → Overdispersion was concentrated in removed teams")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  VERDICT: ARCHITECTURE RECOMMENDATION")
    print(f"{'='*100}")

    overall_disp = all_runs.var(ddof=1) / all_runs.mean()
    margin_excess = obs_var / skel_var

    print(f"\n  CHECK 1 — Overdispersion:")
    print(f"    Dispersion ratio: {overall_disp:.3f} ({'YES' if overall_disp > 1.05 else 'NO'}: {'overdispersed' if overall_disp > 1.05 else 'Poisson OK'})")

    print(f"\n  CHECK 2 — Margin/Skellam:")
    print(f"    Margin excess variance: {(margin_excess-1)*100:+.1f}%")
    skellam_ok = abs(margin_excess - 1) < 0.10

    print(f"\n  CHECK 3 — Correlation:")
    print(f"    Pearson r: {corr_pearson:+.4f} ({'negligible' if abs(corr_pearson) < 0.05 else 'weak' if abs(corr_pearson) < 0.15 else 'meaningful'})")

    print(f"\n  CHECK 4 — Context stability:")
    broad = var_rem / mu_rem > 1.05
    print(f"    Overdispersion is {'broad' if broad else 'concentrated'} across teams")

    print(f"\n  RECOMMENDED ARCHITECTURE:")
    needs_negbin = overall_disp > 1.05
    needs_corr = abs(corr_pearson) >= 0.10
    if needs_negbin and needs_corr:
        print(f"    → NegBin with shared-environment correlation term")
        print(f"    → Dispersion demands NegBin; correlation demands bivariate/copula")
    elif needs_negbin:
        print(f"    → NegBin (independent home/away)")
        print(f"    → Overdispersion demands NegBin; correlation too weak to model")
    elif needs_corr:
        print(f"    → Poisson with shared-environment term")
        print(f"    → No overdispersion; but correlation needs modeling")
    else:
        print(f"    → Independent Poisson (simplest)")
        print(f"    → No overdispersion, no meaningful correlation")

    if needs_negbin:
        print(f"\n  WHAT NEGBIN GAINS:")
        print(f"    → Better tail probability estimates (Poisson under-predicts high-scoring games)")
        print(f"    → More accurate alt-spread and alt-total pricing on deep rungs")
        print(f"    → {(overall_disp-1)*100:.0f}% excess variance captured by r parameter")

    print(f"\n  WHAT THE DATA DOES NOT DEMAND:")
    if not needs_corr:
        print(f"    → Bivariate/copula model (correlation is negligible)")
    if not needs_negbin:
        print(f"    → NegBin complexity (Poisson fits adequately)")
    print(f"    → Zero-inflation (MLB rarely has 0 runs relative to Poisson prediction)")


if __name__ == "__main__":
    main()
