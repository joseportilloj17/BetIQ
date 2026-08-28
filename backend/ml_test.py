"""
MONEYLINE TEST: Does our win-signal monetize on ML where high WR pays?

De-duplicates to game-team level (656 pairs), gets ML odds from fixtures
bookmakers JSON, determines team-won-outright, and tests WR vs ML breakeven
by signal bands. Split-half + multi-fold validation.
"""

import sqlite3, json, re, math
import numpy as np
from collections import defaultdict

DB = "/Users/joseportillo/Downloads/BetIQ/data/bets.db"


def get_ml_dataset():
    """Build game-team level dataset with ML odds and win outcome."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # 1. Get all scored legs joined with bet_clv
    rows = conn.execute("""
        SELECT c.fixture_id, c.team, c.spread, c.leg_result,
               l.ats_dist_wp, l.negbin_wp, l.bayes_team_wp, l.ridge_composite,
               l.negbin_edge_pp,
               l.resolved_home_score, l.resolved_away_score,
               l.resolved_home_team, l.resolved_away_team
        FROM bet_clv c
        JOIN mock_bet_legs l ON l.id = c.leg_id
        WHERE l.resolved_home_score IS NOT NULL
          AND l.resolved_away_score IS NOT NULL
    """).fetchall()

    # 2. De-duplicate to game-team level, keeping features from +1.5 rung
    game_teams = {}  # (fixture_id, team) -> record
    for r in rows:
        key = (r["fixture_id"], r["team"])
        spread = r["spread"]
        if key not in game_teams:
            game_teams[key] = {
                "fixture_id": r["fixture_id"],
                "team": r["team"],
                "home_score": r["resolved_home_score"],
                "away_score": r["resolved_away_score"],
                "home_team": r["resolved_home_team"],
                "away_team": r["resolved_away_team"],
                "bayes_team_wp": r["bayes_team_wp"],
                # Will be filled from +1.5 rung
                "dist_wp_15": None,
                "ridge_15": None,
                "negbin_wp_15": None,
                "edge_pp_15": None,
                "best_spread": spread,
            }
        # Prefer +1.5 rung features (closest to ML)
        if spread == 1.5 or game_teams[key]["dist_wp_15"] is None:
            if spread <= (game_teams[key].get("best_spread") or 999):
                game_teams[key]["dist_wp_15"] = r["ats_dist_wp"]
                game_teams[key]["ridge_15"] = r["ridge_composite"]
                game_teams[key]["negbin_wp_15"] = r["negbin_wp"]
                game_teams[key]["edge_pp_15"] = r["negbin_edge_pp"]
                game_teams[key]["best_spread"] = spread

    # 3. Get ML odds from fixtures bookmakers JSON
    fixture_ids = list(set(gt[0] for gt in game_teams))
    ml_odds = {}  # fixture_id -> {team: median_decimal_odds}

    for fid in fixture_ids:
        fix = conn.execute(
            "SELECT bookmakers FROM fixtures WHERE id = ?", (fid,)
        ).fetchone()
        if not fix or not fix["bookmakers"]:
            continue
        try:
            books = json.loads(fix["bookmakers"])
        except (json.JSONDecodeError, TypeError):
            continue

        team_prices = defaultdict(list)
        for book in books:
            for mkt in book.get("markets", []):
                if mkt["key"] == "h2h":
                    for out in mkt.get("outcomes", []):
                        team_prices[out["name"]].append(out["price"])

        ml_odds[fid] = {
            team: float(np.median(prices))
            for team, prices in team_prices.items()
            if prices
        }

    conn.close()

    # 4. Build final dataset
    dataset = []
    no_ml_odds = 0
    no_features = 0
    for key, gt in game_teams.items():
        fid, team = key
        if fid not in ml_odds or team not in ml_odds[fid]:
            no_ml_odds += 1
            continue
        if gt["bayes_team_wp"] is None and gt["dist_wp_15"] is None:
            no_features += 1
            continue

        # Determine if team WON outright
        is_home = (team == gt["home_team"])
        if is_home:
            team_won = gt["home_score"] > gt["away_score"]
        else:
            team_won = gt["away_score"] > gt["home_score"]

        ml_dec = ml_odds[fid][team]
        ml_be = 1.0 / ml_dec  # breakeven implied probability

        dataset.append({
            "fixture_id": fid,
            "team": team,
            "team_won": team_won,
            "ml_decimal": ml_dec,
            "ml_be": ml_be,
            "bayes_wp": gt["bayes_team_wp"],
            "dist_wp_15": gt["dist_wp_15"],
            "ridge_15": gt["ridge_15"],
            "negbin_wp_15": gt["negbin_wp_15"],
            "edge_pp_15": gt["edge_pp_15"],
        })

    print(f"Dataset: {len(dataset)} game-team pairs")
    print(f"  Skipped (no ML odds): {no_ml_odds}")
    print(f"  Skipped (no features): {no_features}")
    return dataset


def band_analysis(dataset, feature_name, bands, label=""):
    """Compute WR vs ML BE for each band of a feature."""
    print(f"\n{'='*70}")
    print(f"  {label or feature_name} — ML WR vs Breakeven by band")
    print(f"{'='*70}")
    print(f"{'Band':>18}  {'N':>5}  {'WR%':>6}  {'ML_BE%':>7}  {'Edge_pp':>8}  {'Z':>6}  {'PnL/unit':>9}")
    print("-" * 70)

    results = []
    for lo, hi, name in bands:
        subset = [d for d in dataset
                  if d.get(feature_name) is not None
                  and lo <= d[feature_name] < hi]
        if not subset:
            continue
        n = len(subset)
        wins = sum(1 for d in subset if d["team_won"])
        wr = wins / n
        avg_be = np.mean([d["ml_be"] for d in subset])
        avg_ml = np.mean([d["ml_decimal"] for d in subset])
        edge = wr - avg_be
        # PnL per unit: win pays (ml_dec - 1), loss pays -1
        pnl_per = sum(
            (d["ml_decimal"] - 1) if d["team_won"] else -1.0
            for d in subset
        ) / n
        # Z-score: (WR - BE) / sqrt(BE * (1-BE) / N)
        se = math.sqrt(avg_be * (1 - avg_be) / n) if n > 0 else 1
        z = edge / se if se > 0 else 0

        print(f"{name:>18}  {n:>5}  {wr*100:>5.1f}%  {avg_be*100:>6.1f}%  "
              f"{edge*100:>+7.1f}pp  {z:>+5.2f}  ${pnl_per:>+8.3f}")
        results.append({
            "band": name, "n": n, "wr": wr, "be": avg_be,
            "edge": edge, "z": z, "pnl_per": pnl_per, "feature": feature_name
        })

    # Also do correlation
    vals = [(d[feature_name], 1 if d["team_won"] else 0) for d in dataset
            if d.get(feature_name) is not None]
    if vals:
        xs, ys = zip(*vals)
        r = np.corrcoef(xs, ys)[0, 1]
        pnl_vals = [(d[feature_name],
                     (d["ml_decimal"] - 1) if d["team_won"] else -1.0)
                    for d in dataset if d.get(feature_name) is not None]
        xs2, ys2 = zip(*pnl_vals)
        r_pnl = np.corrcoef(xs2, ys2)[0, 1]
        print(f"\n  r({feature_name}, team_won) = {r:+.4f}  "
              f"r({feature_name}, ML_PnL) = {r_pnl:+.4f}  N={len(vals)}")

    return results


def split_half_test(dataset, feature_name, lo, hi, n_seeds=5):
    """Chronological + random split-half stability test."""
    subset = [d for d in dataset
              if d.get(feature_name) is not None
              and lo <= d[feature_name] < hi]
    n = len(subset)
    if n < 20:
        print(f"    Split-half: N={n} too small, skipping")
        return

    # Chronological split (order by fixture_id as proxy for time)
    subset_sorted = sorted(subset, key=lambda d: d["fixture_id"])
    mid = n // 2
    h1 = subset_sorted[:mid]
    h2 = subset_sorted[mid:]

    def edge_of(s):
        if not s:
            return 0, 0, 0
        wins = sum(1 for d in s if d["team_won"])
        wr_ = wins / len(s)
        be_ = np.mean([d["ml_be"] for d in s])
        return wr_ - be_, wr_, be_

    e1, wr1, be1 = edge_of(h1)
    e2, wr2, be2 = edge_of(h2)
    print(f"    Chrono split: H1 edge={e1*100:+.1f}pp (WR={wr1*100:.1f}% BE={be1*100:.1f}% N={len(h1)})  "
          f"H2 edge={e2*100:+.1f}pp (WR={wr2*100:.1f}% BE={be2*100:.1f}% N={len(h2)})")

    # Random splits
    rng = np.random.default_rng(42)
    pos_count = 0
    for seed in range(n_seeds):
        rng2 = np.random.default_rng(seed)
        idx = rng2.permutation(n)
        s1 = [subset[i] for i in idx[:mid]]
        s2 = [subset[i] for i in idx[mid:]]
        e_s1, _, _ = edge_of(s1)
        e_s2, _, _ = edge_of(s2)
        if e_s1 > 0 and e_s2 > 0:
            pos_count += 1
    print(f"    Random splits: {pos_count}/{n_seeds} seeds with BOTH halves positive")


def multi_fold_test(dataset, feature_name, lo, hi, k=5, n_seeds=5):
    """Multi-fold validation with noise floor."""
    subset = [d for d in dataset
              if d.get(feature_name) is not None
              and lo <= d[feature_name] < hi]
    n = len(subset)
    if n < 30:
        se_edge = math.sqrt(0.5 * 0.5 / n) if n > 0 else 999
        games_for_25se = int(math.ceil((2.5 / (0.05)) ** 2 * 0.25))  # rough
        print(f"    Multi-fold: N={n} too small. SE(edge)≈{se_edge*100:.1f}pp. "
              f"Need ~{games_for_25se} games for 2.5 SE at 5pp edge.")
        return

    subset_sorted = sorted(subset, key=lambda d: d["fixture_id"])
    noise_floor = math.sqrt(1.0 / (k * n))  # ~0.000846 at k=10, N=140

    all_fold_edges = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        fold_edges = []
        fold_size = n // k
        for f in range(k):
            start = f * fold_size
            end = start + fold_size if f < k - 1 else n
            fold = [subset_sorted[idx[i]] for i in range(start, end)]
            wins = sum(1 for d in fold if d["team_won"])
            wr_ = wins / len(fold) if fold else 0
            be_ = np.mean([d["ml_be"] for d in fold]) if fold else 0
            fold_edges.append(wr_ - be_)
        all_fold_edges.append(fold_edges)

    # Average across seeds
    avg_edges = np.mean(all_fold_edges, axis=0)
    folds_positive = sum(1 for e in avg_edges if e > noise_floor)
    overall_edge = np.mean(avg_edges)
    overall_se = np.std(avg_edges) / math.sqrt(k)

    print(f"    {k}-fold (avg {n_seeds} seeds): {folds_positive}/{k} folds above noise floor ({noise_floor*100:.3f}pp)")
    print(f"    Mean fold edge: {overall_edge*100:+.2f}pp  SE: {overall_se*100:.2f}pp  "
          f"Z: {overall_edge/overall_se:+.2f}" if overall_se > 0 else "")
    print(f"    Per-fold edges: {['%+.1f' % (e*100) for e in avg_edges]}")


def bet_type_sweep(dataset):
    """Sweep bet types: what if we bet ML instead of spreads?"""
    print(f"\n{'='*70}")
    print(f"  BET-TYPE COMPARISON: Spread Cover vs ML Win")
    print(f"{'='*70}")

    # Get spread results for same game-teams
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # For each game-team, get spread cover results by rung
    rows = conn.execute("""
        SELECT c.fixture_id, c.team, c.spread, c.leg_result, c.entry_decimal
        FROM bet_clv c
    """).fetchall()
    conn.close()

    # Spread PnL by game-team (use +1.5 rung)
    spread_results = {}
    for r in rows:
        key = (r["fixture_id"], r["team"])
        spread = r["spread"]
        if key not in spread_results or spread < spread_results[key]["spread"]:
            spread_results[key] = {
                "spread": spread,
                "won": r["leg_result"] == "WIN",
                "entry_dec": r["entry_decimal"]
            }

    # Compare spread +1.5 vs ML for same game-teams
    both = 0
    spread_only = 0
    ml_only = 0
    neither = 0
    spread_pnl_total = 0
    ml_pnl_total = 0
    n_compare = 0

    for d in dataset:
        key = (d["fixture_id"], d["team"])
        if key not in spread_results:
            continue
        sr = spread_results[key]
        n_compare += 1

        spread_won = sr["won"]
        ml_won = d["team_won"]

        if spread_won and ml_won:
            both += 1
        elif spread_won and not ml_won:
            spread_only += 1
        elif not spread_won and ml_won:
            ml_only += 1
        else:
            neither += 1

        # PnL
        spread_pnl_total += (sr["entry_dec"] - 1) if spread_won else -1.0
        ml_pnl_total += (d["ml_decimal"] - 1) if ml_won else -1.0

    print(f"\n  Matched game-teams: {n_compare}")
    print(f"  Both win:       {both:>5} ({both/n_compare*100:.1f}%)")
    print(f"  Spread only:    {spread_only:>5} ({spread_only/n_compare*100:.1f}%)")
    print(f"  ML only:        {ml_only:>5} ({ml_only/n_compare*100:.1f}%)")
    print(f"  Neither:        {neither:>5} ({neither/n_compare*100:.1f}%)")
    print(f"\n  Spread +1.5 PnL/unit: ${spread_pnl_total/n_compare:+.4f}")
    print(f"  ML PnL/unit:          ${ml_pnl_total/n_compare:+.4f}")


def sample_size_calc(dataset, feature_name, lo, hi, target_se=2.5):
    """How many games needed for target SE from zero?"""
    subset = [d for d in dataset
              if d.get(feature_name) is not None
              and lo <= d[feature_name] < hi]
    n = len(subset)
    if n < 5:
        return

    wins = sum(1 for d in subset if d["team_won"])
    wr = wins / n
    avg_be = np.mean([d["ml_be"] for d in subset])
    edge = wr - avg_be
    if edge <= 0:
        print(f"    Sample-size calc: edge={edge*100:+.1f}pp (negative), N/A")
        return

    # SE of edge = sqrt(p*(1-p)/n)
    # For Z = edge / SE = target_se:
    # n_needed = p*(1-p) * (target_se / edge)^2
    p = wr
    n_needed = p * (1 - p) * (target_se / edge) ** 2
    print(f"    Sample-size: edge={edge*100:+.1f}pp, current SE={math.sqrt(p*(1-p)/n)*100:.1f}pp, "
          f"current Z={edge/math.sqrt(p*(1-p)/n):+.2f}")
    print(f"    Need N≈{int(math.ceil(n_needed))} for Z={target_se:.1f} (have {n})")


def main():
    dataset = get_ml_dataset()

    # Basic stats
    wins = sum(1 for d in dataset if d["team_won"])
    print(f"  Overall ML WR: {wins}/{len(dataset)} = {wins/len(dataset)*100:.1f}%")
    avg_be = np.mean([d["ml_be"] for d in dataset])
    print(f"  Average ML BE: {avg_be*100:.1f}%")
    avg_ml = np.mean([d["ml_decimal"] for d in dataset])
    print(f"  Average ML odds: {avg_ml:.3f}")
    edge = wins / len(dataset) - avg_be
    se = math.sqrt(avg_be * (1 - avg_be) / len(dataset))
    print(f"  Overall ML edge: {edge*100:+.1f}pp  Z={edge/se:+.2f}")

    # Home/Away split
    home_picks = [d for d in dataset if d["team"] == "home"]  # won't match, use below
    print(f"\n  Underdog context: all our picks are +rung (underdog side)")
    favs = [d for d in dataset if d["ml_be"] > 0.5]
    dogs = [d for d in dataset if d["ml_be"] <= 0.5]
    print(f"  ML favorites (BE>50%): N={len(favs)}, "
          f"WR={sum(1 for d in favs if d['team_won'])/len(favs)*100:.1f}% vs BE={np.mean([d['ml_be'] for d in favs])*100:.1f}%"
          if favs else "  No favorites")
    print(f"  ML underdogs (BE<=50%): N={len(dogs)}, "
          f"WR={sum(1 for d in dogs if d['team_won'])/len(dogs)*100:.1f}% vs BE={np.mean([d['ml_be'] for d in dogs])*100:.1f}%"
          if dogs else "  No underdogs")

    # ── FEATURE SWEEP ──────────────────────────────────────────
    # bayes_team_wp (rung-independent win probability)
    bayes_bands = [
        (0.0, 0.40, "<40%"),
        (0.40, 0.50, "40-50%"),
        (0.50, 0.60, "50-60%"),
        (0.60, 0.70, "60-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 0.90, "80-90%"),
        (0.90, 1.01, "90%+"),
    ]
    results_bayes = band_analysis(dataset, "bayes_wp", bayes_bands,
                                  "bayes_team_wp (team win prob)")

    # dist_wp at +1.5 rung (closest to ML)
    dist_bands = [
        (0.0, 0.50, "<50%"),
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 0.90, "80-90%"),
        (0.90, 1.01, "90%+"),
    ]
    results_dist = band_analysis(dataset, "dist_wp_15", dist_bands,
                                 "dist_wp at +1.5 rung")

    # ridge_composite at +1.5
    ridge_bands = [
        (0.0, 0.50, "<50%"),
        (0.50, 0.60, "50-60%"),
        (0.60, 0.70, "60-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 0.90, "80-90%"),
        (0.90, 1.01, "90%+"),
    ]
    results_ridge = band_analysis(dataset, "ridge_15", ridge_bands,
                                  "ridge_composite at +1.5 rung")

    # edge_pp at +1.5
    epp_bands = [
        (-50, 0, "epp<0"),
        (0, 3, "0-3"),
        (3, 6, "3-6"),
        (6, 9, "6-9"),
        (9, 12, "9-12"),
        (12, 50, "12+"),
    ]
    results_epp = band_analysis(dataset, "edge_pp_15", epp_bands,
                                "edge_pp at +1.5 rung")

    # ── SPLIT-HALF & MULTI-FOLD for interesting bands ──────────
    print(f"\n{'='*70}")
    print(f"  SPLIT-HALF & MULTI-FOLD VALIDATION")
    print(f"{'='*70}")

    # Test all bayes_wp bands with positive edge
    for res in results_bayes:
        if res["edge"] > 0 and res["n"] >= 15:
            band = res["band"]
            lo, hi = next((b[0], b[1]) for b in bayes_bands if b[2] == band)
            print(f"\n  bayes_wp {band} (N={res['n']}, edge={res['edge']*100:+.1f}pp):")
            split_half_test(dataset, "bayes_wp", lo, hi)
            multi_fold_test(dataset, "bayes_wp", lo, hi, k=5)
            sample_size_calc(dataset, "bayes_wp", lo, hi)

    # Test all dist_wp bands with positive edge
    for res in results_dist:
        if res["edge"] > 0 and res["n"] >= 15:
            band = res["band"]
            lo, hi = next((b[0], b[1]) for b in dist_bands if b[2] == band)
            print(f"\n  dist_wp_15 {band} (N={res['n']}, edge={res['edge']*100:+.1f}pp):")
            split_half_test(dataset, "dist_wp_15", lo, hi)
            multi_fold_test(dataset, "dist_wp_15", lo, hi, k=5)
            sample_size_calc(dataset, "dist_wp_15", lo, hi)

    # Test ridge bands with positive edge
    for res in results_ridge:
        if res["edge"] > 0 and res["n"] >= 15:
            band = res["band"]
            lo, hi = next((b[0], b[1]) for b in ridge_bands if b[2] == band)
            print(f"\n  ridge_15 {band} (N={res['n']}, edge={res['edge']*100:+.1f}pp):")
            split_half_test(dataset, "ridge_15", lo, hi)
            multi_fold_test(dataset, "ridge_15", lo, hi, k=5)
            sample_size_calc(dataset, "ridge_15", lo, hi)

    # ── BET TYPE COMPARISON ────────────────────────────────────
    bet_type_sweep(dataset)

    # ── FEATURE CORRELATION MATRIX ─────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FEATURE CORRELATIONS WITH ML OUTCOME")
    print(f"{'='*70}")
    features = ["bayes_wp", "dist_wp_15", "ridge_15", "negbin_wp_15", "edge_pp_15"]
    for feat in features:
        valid = [(d[feat], 1 if d["team_won"] else 0,
                  (d["ml_decimal"] - 1) if d["team_won"] else -1.0)
                 for d in dataset if d.get(feat) is not None]
        if not valid:
            continue
        xs, won, pnl = zip(*valid)
        r_won = np.corrcoef(xs, won)[0, 1]
        r_pnl = np.corrcoef(xs, pnl)[0, 1]
        print(f"  {feat:>15}: r(won)={r_won:+.4f}  r(ML_pnl)={r_pnl:+.4f}  N={len(valid)}")


if __name__ == "__main__":
    main()
