"""
tie_resolution_percell.py — Validate tie-resolution strategies across
the Home/Away × Fav/Dog matrix, not just aggregate.

For each quadrant (FavHome, FavAway, DogHome, DogAway), compare
simulated vs empirical cover rates at ±1.5, ±2.5, ±3.5 rungs
under three tie-resolution strategies.

KEY QUESTION: does the 60/40 home-biased tie rule misprice away-side
rungs (Fav:Away, Dog:Away)?
"""
from __future__ import annotations
import os, sys, sqlite3, json, math
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "data", "bets.db")


def _load_games_with_fav():
    """Load games with scores + moneyline-based fav classification."""
    conn = sqlite3.connect(DB_PATH)

    # Get unique games with scores
    games = conn.execute("""
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

    # Get moneyline prices for fav/dog classification
    fixtures = conn.execute("""
        SELECT id, home_team, away_team, bookmakers
        FROM fixtures
        WHERE sport_key = 'baseball_mlb' AND bookmakers IS NOT NULL
    """).fetchall()
    conn.close()

    # Build fav map: fixture_id -> home_is_fav
    fav_map = {}
    for fid, home, away, bk_json in fixtures:
        if not bk_json:
            continue
        try:
            bks = json.loads(bk_json)
        except:
            continue
        home_price = None
        for pref in ["fanduel", "draftkings", "pinnacle", None]:
            for bk in bks:
                if pref is not None and bk.get("key") != pref:
                    continue
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    for oc in mkt.get("outcomes", []):
                        if oc.get("name", "").lower().strip() == home.lower().strip():
                            home_price = float(oc.get("price", 0))
                            break
                    if home_price:
                        break
                if home_price:
                    break
            if home_price:
                break
        if home_price and home_price > 0:
            fav_map[fid] = home_price < 2.0  # True = home is fav

    # Merge
    result = []
    for fid, hs, aws, ht, at, gd in games:
        home_is_fav = fav_map.get(fid)
        if home_is_fav is None:
            continue
        result.append({
            "fid": fid, "hs": int(hs), "as": int(aws),
            "ht": ht, "at": at, "gd": gd,
            "home_is_fav": home_is_fav,
        })
    return result


def _fit_negbin_mom(data):
    data = np.array(data, dtype=float)
    mu = data.mean()
    var = data.var(ddof=1)
    if var <= mu:
        return mu ** 2 / 0.01, mu / (mu + 0.01), mu, var  # degenerate
    r = mu ** 2 / (var - mu)
    p = r / (r + mu)
    return r, p, mu, var


def main():
    print("=" * 100)
    print("  TIE-RESOLUTION: PER-CELL VALIDATION (Home/Away × Fav/Dog)")
    print("=" * 100)

    games = _load_games_with_fav()
    print(f"\n  Games with fav classification: {len(games)}")

    # Split into quadrants
    # "Perspective" is the PICK side. For spread pricing, we care about
    # each team's cover probability. So we need to check from BOTH sides.
    # But first, let's understand the structure.

    # Quadrants based on the TEAM whose cover we're pricing:
    # FavHome: home team is favorite, pricing home side
    # FavAway: away team is favorite, pricing away side
    # DogHome: home team is dog, pricing home side
    # DogAway: away team is dog, pricing away side

    # Note: FavHome and DogAway are the SAME games (just opposite sides)
    # FavAway and DogHome are the SAME games (opposite sides)

    fav_home_games = [g for g in games if g["home_is_fav"]]
    fav_away_games = [g for g in games if not g["home_is_fav"]]

    print(f"  Home is fav: {len(fav_home_games)} games")
    print(f"  Away is fav: {len(fav_away_games)} games")

    # Empirical margin distributions per quadrant
    # margin = home - away (positive = home wins)
    print(f"\n  Empirical margin distributions:")
    for label, gset in [("Fav@Home", fav_home_games), ("Fav@Away", fav_away_games)]:
        margins = [g["hs"] - g["as"] for g in gset]
        print(f"    {label}: n={len(gset)}, mean_margin={np.mean(margins):+.2f}")

    # Fit NegBin per quadrant
    # We need per-quadrant μ because fav teams score more
    all_home = np.array([g["hs"] for g in games])
    all_away = np.array([g["as"] for g in games])
    r_global, p_global, mu_global, _ = _fit_negbin_mom(np.concatenate([all_home, all_away]))

    # Per-quadrant means
    fh_home_runs = [g["hs"] for g in fav_home_games]
    fh_away_runs = [g["as"] for g in fav_home_games]
    fa_home_runs = [g["hs"] for g in fav_away_games]
    fa_away_runs = [g["as"] for g in fav_away_games]

    print(f"\n  Per-quadrant scoring:")
    print(f"    Fav@Home: home(fav)={np.mean(fh_home_runs):.2f}, away(dog)={np.mean(fh_away_runs):.2f}")
    print(f"    Fav@Away: home(dog)={np.mean(fa_home_runs):.2f}, away(fav)={np.mean(fa_away_runs):.2f}")

    # ═══════════════════════════════════════════════════════════
    # SIMULATE per quadrant under each tie-resolution strategy
    # ═══════════════════════════════════════════════════════════
    np.random.seed(42)
    N_SIM = 50_000  # sims per game-type

    strategies = {
        "±1 60/40 home": {"type": "simple", "home_p": 0.60},
        "±1 57/43 home": {"type": "simple", "home_p": 0.57},
        "extras model":  {"type": "extras"},
    }

    # Extra-innings resolution distribution
    xi_dist = {
        +1: 0.383, -1: 0.287,
        +2: 0.114, -2: 0.086,
        +3: 0.047, -3: 0.035,
        +4: 0.019, -4: 0.014,
        +5: 0.008, -5: 0.006,
    }
    xi_total = sum(xi_dist.values())
    xi_keys = list(xi_dist.keys())
    xi_probs = [xi_dist[k] / xi_total for k in xi_keys]

    def resolve_ties(margins_raw, strategy):
        margins = margins_raw.copy()
        tie_mask = margins == 0
        n_ties = tie_mask.sum()
        if n_ties == 0:
            return margins

        s = strategies[strategy]
        if s["type"] == "simple":
            coin = np.random.random(n_ties)
            margins[tie_mask] = np.where(coin < s["home_p"], 1, -1)
        elif s["type"] == "extras":
            margins[tie_mask] = np.random.choice(xi_keys, size=n_ties, p=xi_probs)
        return margins

    # For each quadrant, simulate and compute cover rates
    quadrants = [
        ("Fav@Home", fav_home_games, fh_home_runs, fh_away_runs),
        ("Fav@Away", fav_away_games, fa_home_runs, fa_away_runs),
    ]

    # The four pick-side cells:
    # FavHome_pick_home (fav is home, pricing fav's cover)
    # FavHome_pick_away (fav is home, pricing dog's cover)
    # FavAway_pick_home (fav is away, pricing dog's cover)
    # FavAway_pick_away (fav is away, pricing fav's cover)

    results = {}  # {(quadrant, strategy): {rung: simulated_cover_pct}}
    empirical = {}  # {quadrant: {rung: empirical_cover_pct}}

    # Rungs to check: from PICK's perspective
    # For pick=home: cover +1.5 means margin >= -1 (home loses by ≤1 or wins)
    # For pick=away: cover +1.5 means margin <= +1 (away loses by ≤1 or wins)
    # More precisely: spread +X.5 means pick covers if pick_margin >= -(X+0.5)
    # For home pick: home_margin >= -(X+0.5), i.e. margin >= -X for integer X
    # For away pick: away_margin >= -(X+0.5), i.e. -margin >= -X, i.e. margin <= X

    pick_sides = ["fav", "dog"]

    print(f"\n{'='*100}")
    print(f"  PER-CELL COVER RATES: EMPIRICAL vs SIMULATED")
    print(f"{'='*100}")

    for quad_label, gset, h_runs, a_runs in quadrants:
        mu_h = np.mean(h_runs)
        mu_a = np.mean(a_runs)
        p_h = r_global / (r_global + mu_h)
        p_a = r_global / (r_global + mu_a)

        emp_margins = np.array([g["hs"] - g["as"] for g in gset])
        n_emp = len(emp_margins)

        # Simulate
        sim_h = np.random.negative_binomial(r_global, p_h, N_SIM)
        sim_a = np.random.negative_binomial(r_global, p_a, N_SIM)
        sim_margin_raw = sim_h - sim_a

        # Determine which side is fav, which is dog
        if quad_label == "Fav@Home":
            fav_side = "home"
            dog_side = "away"
        else:
            fav_side = "away"
            dog_side = "home"

        for pick_label, pick_is_home in [("Fav", fav_side == "home"), ("Dog", dog_side == "home")]:
            cell = f"{pick_label}@{'Home' if pick_is_home else 'Away'}"

            print(f"\n  ── {cell} (n={n_emp}) ──")
            print(f"     {'Rung':>8} {'Empirical':>10}", end="")
            for strat in strategies:
                print(f" {strat:>14}", end="")
            print(f"", end="")
            for strat in strategies:
                print(f" {'err':>8}", end="")
            print()
            print(f"     {'─'*90}")

            for rung in [1.5, 2.5, 3.5, 4.5]:
                # Empirical cover rate for this pick at this rung
                # Pick covers +rung if pick wins outright or loses by < rung
                if pick_is_home:
                    # home pick: covers +rung if margin > -rung, i.e. margin >= -(rung-0.5) = -(rung) + 0.5
                    emp_cover = np.mean(emp_margins > -rung) * 100
                else:
                    # away pick: covers +rung if -margin > -rung, i.e. margin < rung
                    emp_cover = np.mean(emp_margins < rung) * 100

                print(f"     +{rung:.1f}   {emp_cover:>9.2f}%", end="")

                errors = []
                for strat in strategies:
                    sim_resolved = resolve_ties(sim_margin_raw.copy(), strat)
                    if pick_is_home:
                        sim_cover = np.mean(sim_resolved > -rung) * 100
                    else:
                        sim_cover = np.mean(sim_resolved < rung) * 100
                    errors.append(sim_cover - emp_cover)
                    print(f" {sim_cover:>13.2f}%", end="")

                for e in errors:
                    print(f" {e:>+7.2f}pp", end="")
                print()

            # Also check the MINUS side (-1.5 = pick is laying)
            print(f"     {'─'*90}")
            for rung in [1.5, 2.5, 3.5]:
                if pick_is_home:
                    emp_cover = np.mean(emp_margins > rung) * 100
                else:
                    emp_cover = np.mean(emp_margins < -rung) * 100

                print(f"     -{rung:.1f}   {emp_cover:>9.2f}%", end="")

                errors = []
                for strat in strategies:
                    sim_resolved = resolve_ties(sim_margin_raw.copy(), strat)
                    if pick_is_home:
                        sim_cover = np.mean(sim_resolved > rung) * 100
                    else:
                        sim_cover = np.mean(sim_resolved < -rung) * 100
                    errors.append(sim_cover - emp_cover)
                    print(f" {sim_cover:>13.2f}%", end="")

                for e in errors:
                    print(f" {e:>+7.2f}pp", end="")
                print()

    # ═══════════════════════════════════════════════════════════
    # SUMMARY: per-cell SSE across strategies
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  SUMMARY: PER-CELL SSE (lower = better)")
    print(f"{'='*100}")

    # Recompute with consistent RNG per strategy
    cell_sse = {strat: {} for strat in strategies}

    for quad_label, gset, h_runs, a_runs in quadrants:
        mu_h = np.mean(h_runs)
        mu_a = np.mean(a_runs)
        p_h = r_global / (r_global + mu_h)
        p_a = r_global / (r_global + mu_a)
        emp_margins = np.array([g["hs"] - g["as"] for g in gset])

        if quad_label == "Fav@Home":
            fav_side_home = True
        else:
            fav_side_home = False

        for strat in strategies:
            np.random.seed(42)
            sim_h = np.random.negative_binomial(r_global, p_h, N_SIM)
            sim_a = np.random.negative_binomial(r_global, p_a, N_SIM)
            sim_resolved = resolve_ties(sim_h - sim_a, strat)

            for pick_label, pick_is_home in [("Fav", fav_side_home), ("Dog", not fav_side_home)]:
                cell = f"{pick_label}@{'Home' if pick_is_home else 'Away'}"
                sse = 0
                for rung in [1.5, 2.5, 3.5, 4.5, -1.5, -2.5, -3.5]:
                    if rung > 0:
                        if pick_is_home:
                            emp = np.mean(emp_margins > -rung) * 100
                            sim = np.mean(sim_resolved > -rung) * 100
                        else:
                            emp = np.mean(emp_margins < rung) * 100
                            sim = np.mean(sim_resolved < rung) * 100
                    else:
                        if pick_is_home:
                            emp = np.mean(emp_margins > -rung) * 100
                            sim = np.mean(sim_resolved > -rung) * 100
                        else:
                            emp = np.mean(emp_margins < rung) * 100
                            sim = np.mean(sim_resolved < rung) * 100
                    sse += (sim - emp) ** 2
                cell_sse[strat][cell] = sse

    print(f"\n    {'Cell':<15}", end="")
    for strat in strategies:
        print(f" {strat:>14}", end="")
    print(f" {'Best':>14}")
    print(f"    {'─'*72}")

    all_cells = sorted(set(c for s in cell_sse.values() for c in s))
    total_sse = {strat: 0 for strat in strategies}
    for cell in all_cells:
        print(f"    {cell:<15}", end="")
        cell_vals = {}
        for strat in strategies:
            v = cell_sse[strat].get(cell, 0)
            cell_vals[strat] = v
            total_sse[strat] += v
            print(f" {v:>13.2f}", end="")
        best = min(cell_vals, key=cell_vals.get)
        print(f" {best:>14}")

    print(f"    {'─'*72}")
    print(f"    {'TOTAL':<15}", end="")
    for strat in strategies:
        print(f" {total_sse[strat]:>13.2f}", end="")
    best_total = min(total_sse, key=total_sse.get)
    print(f" {best_total:>14}")

    # ═══════════════════════════════════════════════════════════
    # FOCUS: Away-side ±1.5 specifically
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  FOCUS: AWAY-SIDE ±1.5 (where home bias could hurt)")
    print(f"{'='*100}")

    for quad_label, gset, h_runs, a_runs in quadrants:
        mu_h = np.mean(h_runs)
        mu_a = np.mean(a_runs)
        p_h = r_global / (r_global + mu_h)
        p_a = r_global / (r_global + mu_a)
        emp_margins = np.array([g["hs"] - g["as"] for g in gset])

        fav_side_home = (quad_label == "Fav@Home")
        pick_is_home = not fav_side_home if quad_label == "Fav@Home" else fav_side_home
        # Dog@Away for Fav@Home games, Fav@Away for Fav@Away games

        # Away pick cells
        for pick_label, pick_is_home in [("Dog", not fav_side_home), ("Fav", fav_side_home)]:
            if pick_is_home:
                continue  # skip home picks, we want away

            cell = f"{pick_label}@Away"
            print(f"\n  {cell} (n={len(gset)}):")

            for rung in [1.5, -1.5]:
                if rung > 0:
                    emp = np.mean(emp_margins < rung) * 100
                else:
                    emp = np.mean(emp_margins < rung) * 100

                print(f"    {'+' if rung > 0 else ''}{rung:.1f}: emp={emp:.2f}%", end="")
                for strat in strategies:
                    np.random.seed(42)
                    sim_h = np.random.negative_binomial(r_global, p_h, N_SIM)
                    sim_a = np.random.negative_binomial(r_global, p_a, N_SIM)
                    sim_resolved = resolve_ties(sim_h - sim_a, strat)
                    if rung > 0:
                        sim = np.mean(sim_resolved < rung) * 100
                    else:
                        sim = np.mean(sim_resolved < rung) * 100
                    print(f"  {strat}={sim:.2f}% ({sim-emp:+.2f}pp)", end="")
                print()

    # ═══════════════════════════════════════════════════════════
    # EMPIRICAL extra-innings calibration
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  EMPIRICAL 1-RUN GAME HOME BIAS BY QUADRANT")
    print(f"{'='*100}")

    for quad_label, gset, _, _ in quadrants:
        emp_margins = np.array([g["hs"] - g["as"] for g in gset])
        one_run = np.sum(np.abs(emp_margins) == 1)
        home_win_1 = np.sum(emp_margins == 1)
        away_win_1 = np.sum(emp_margins == -1)
        if one_run > 0:
            print(f"  {quad_label}: 1-run games={one_run}, home wins={home_win_1} ({home_win_1/one_run*100:.1f}%), "
                  f"away wins={away_win_1} ({away_win_1/one_run*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  VERDICT")
    print(f"{'='*100}")

    print(f"\n  Per-cell SSE totals:")
    for strat in sorted(total_sse, key=total_sse.get):
        print(f"    {strat:<20} {total_sse[strat]:.2f}")

    print(f"\n  Best overall: {best_total}")
    print(f"\n  Key question: does 60/40 home bias hurt away cells?")

    # Check away-cell performance specifically
    away_cells = [c for c in all_cells if "Away" in c]
    for strat in strategies:
        away_sse = sum(cell_sse[strat].get(c, 0) for c in away_cells)
        home_cells_list = [c for c in all_cells if "Home" in c]
        home_sse = sum(cell_sse[strat].get(c, 0) for c in home_cells_list)
        print(f"    {strat:<20} Home SSE={home_sse:.2f}  Away SSE={away_sse:.2f}")


if __name__ == "__main__":
    main()
