"""
Diagnose sim scoring chain: drive outcomes, RZ rates, points/yard vs NFL actuals.
One-shot diagnostic — not production code.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
import numpy as np
from nfl_game_sim import (
    simulate_game, _LG_PACE, _LG_PASS_RATE, _LG_NO_HUDDLE,
    _LG_COMP_RATE, _LG_YPA_COMP, _LG_YPA_COMP_SD, _LG_SACK_RATE,
    _LG_SCRAMBLE_RATE, _LG_INT_RATE, _LG_YPC, _LG_YPC_SD,
    _LG_FUMBLE_RATE, _sim_drive, _kickoff_yl, _quarter_from_remaining,
    _classify_situation, _GAME_SECONDS, _CHANGE_OF_POSSESSION_SECS,
)

# League-average profile
LG = {
    "pace": _LG_PACE, "pass_rate": _LG_PASS_RATE,
    "no_huddle": _LG_NO_HUDDLE,
    "comp_rate": _LG_COMP_RATE, "ypa_comp": _LG_YPA_COMP,
    "ypa_comp_sd": _LG_YPA_COMP_SD, "sack_rate": _LG_SACK_RATE,
    "scramble_rate": _LG_SCRAMBLE_RATE, "int_rate": _LG_INT_RATE,
    "ypc": _LG_YPC, "ypc_sd": _LG_YPC_SD,
    "fumble_rate": _LG_FUMBLE_RATE, "n_games": 100,
}


def sim_with_drive_tracking(rng, profile, n_sims=2000):
    """Run sims and track drive-level outcomes."""
    all_games = []

    for _ in range(n_sims):
        res = simulate_game(rng, profile, profile)

        # Reconstruct drive outcomes from plays for home team.
        # We need to re-simulate with drive tracking. Instead, let's
        # parse the play list to extract drives.
        # Each drive ends with: TD, FG, punt, turnover, turnover_on_downs,
        # end_of_half. We can infer from play sequences.

        # Actually, simulate_game doesn't expose drive outcomes directly.
        # Let's modify approach: run _sim_drive directly in a loop.
        all_games.append(res)

    return all_games


def sim_game_with_drives(rng, profile):
    """Simulate a full game, returning per-drive outcomes."""
    home_score = 0
    away_score = 0
    game_secs = _GAME_SECONDS
    home_receives_first = rng.random() < 0.5
    home_has_ball = home_receives_first
    current_yl = _kickoff_yl(rng)
    is_second_half = False

    home_drives = []
    away_drives = []

    while game_secs > 0:
        profile_use = profile  # both teams use same profile
        score_diff = (home_score - away_score) if home_has_ball else (away_score - home_score)
        qtr = _quarter_from_remaining(game_secs)

        if game_secs <= 1800 and not is_second_half:
            is_second_half = True
            home_has_ball = not home_receives_first
            current_yl = _kickoff_yl(rng)
            game_secs = 1800
            continue

        drive = _sim_drive(rng, profile_use, None,
                           start_yl=current_yl, game_secs=game_secs,
                           score_diff=score_diff, qtr=qtr)

        # Count plays (exclude kneels)
        n_plays = sum(1 for p in drive["plays"]
                      if p.get("play_result") != "kneel")
        total_yards = sum(p["yards"] for p in drive["plays"]
                          if p.get("play_result") != "kneel")
        pass_yards = sum(p["yards"] for p in drive["plays"]
                         if p.get("completion"))
        rush_yards = sum(p["yards"] for p in drive["plays"]
                         if p.get("play_result") == "rush")
        scramble_yards = sum(p["yards"] for p in drive["plays"]
                             if p.get("scramble"))

        # Was this a red zone trip?
        # RZ trip = drive reached opponent's 20 or closer (yl <= 20)
        min_yl = current_yl
        yl_tracker = current_yl
        for p in drive["plays"]:
            if p.get("play_result") != "kneel":
                yl_tracker = max(1, yl_tracker - p["yards"])
                min_yl = min(min_yl, yl_tracker)
        reached_rz = min_yl <= 20

        drive_rec = {
            "outcome": drive["outcome"],
            "start_yl": current_yl,
            "n_plays": n_plays,
            "total_yards": total_yards,
            "pass_yards": pass_yards,
            "rush_yards": rush_yards,
            "scramble_yards": scramble_yards,
            "reached_rz": reached_rz,
            "min_yl": min_yl,
        }

        if home_has_ball:
            home_drives.append(drive_rec)
        else:
            away_drives.append(drive_rec)

        game_secs -= drive["seconds_used"]

        if drive["outcome"] == "td":
            points = 6
            if rng.random() < 0.05:
                points += 2 if rng.random() < 0.48 else 0
            else:
                points += 1 if rng.random() < 0.94 else 0
            if home_has_ball:
                home_score += points
            else:
                away_score += points
        elif drive["outcome"] == "fg":
            if home_has_ball:
                home_score += 3
            else:
                away_score += 3

        if drive["outcome"] in ("td", "fg", "fg_miss"):
            current_yl = _kickoff_yl(rng)
            game_secs -= _CHANGE_OF_POSSESSION_SECS
        elif drive["outcome"] in ("punt", "turnover", "turnover_on_downs"):
            current_yl = drive["end_yl"]
            game_secs -= _CHANGE_OF_POSSESSION_SECS * 0.5
        elif drive["outcome"] == "end_of_half":
            if is_second_half:
                break
            game_secs = 1800.01
            continue

        home_has_ball = not home_has_ball

    return {
        "home_score": home_score, "away_score": away_score,
        "home_drives": home_drives, "away_drives": away_drives,
    }


def main():
    N = 2000
    rng = np.random.default_rng(42)
    print(f"Running {N} sims with league-average profiles...")

    # Accumulate per-team stats
    scores = []
    drives_total = []
    outcome_counts = defaultdict(list)
    rz_trips = []
    rz_tds = []
    plays_total = []
    total_yards_all = []
    pass_yards_all = []
    rush_yards_all = []
    scramble_yards_all = []

    for i in range(N):
        res = sim_game_with_drives(rng, LG)
        # Use home team (symmetric, both LG-average)
        drives = res["home_drives"]
        scores.append(res["home_score"])
        drives_total.append(len(drives))

        # Count outcomes
        oc = defaultdict(int)
        rz_t = 0
        rz_td = 0
        n_plays = 0
        tot_yds = 0
        p_yds = 0
        r_yds = 0
        sc_yds = 0

        for d in drives:
            oc[d["outcome"]] += 1
            n_plays += d["n_plays"]
            tot_yds += d["total_yards"]
            p_yds += d["pass_yards"]
            r_yds += d["rush_yards"]
            sc_yds += d["scramble_yards"]
            if d["reached_rz"]:
                rz_t += 1
                if d["outcome"] == "td":
                    rz_td += 1

        for outcome_type in ["td", "fg", "fg_miss", "punt", "turnover",
                             "turnover_on_downs", "end_of_half", "kneel"]:
            outcome_counts[outcome_type].append(oc.get(outcome_type, 0))

        rz_trips.append(rz_t)
        rz_tds.append(rz_td)
        plays_total.append(n_plays)
        total_yards_all.append(tot_yds)
        pass_yards_all.append(p_yds)
        rush_yards_all.append(r_yds)
        scramble_yards_all.append(sc_yds)

    # Report
    scores = np.array(scores, dtype=float)
    drives_total = np.array(drives_total, dtype=float)
    rz_trips = np.array(rz_trips, dtype=float)
    rz_tds = np.array(rz_tds, dtype=float)
    plays_total = np.array(plays_total, dtype=float)
    total_yards_all = np.array(total_yards_all, dtype=float)
    pass_yards_all = np.array(pass_yards_all, dtype=float)
    rush_yards_all = np.array(rush_yards_all, dtype=float)
    scramble_yards_all = np.array(scramble_yards_all, dtype=float)

    print(f"\n{'='*70}")
    print("SIM SCORING CHAIN DIAGNOSTICS (per team-game, {N} sims)")
    print(f"{'='*70}")

    # NFL actuals from 2023-2024 PBP analysis
    actuals = {
        "Drives":        (10.50, drives_total.mean()),
        "Plays":         (62.0,  plays_total.mean()),
        "Total yards":   (336.6, total_yards_all.mean()),
        "Pass yds(gross)":(233.1, pass_yards_all.mean()),
        "Rush yds":      (103.5, rush_yards_all.mean()),
        "Scramble yds":  (18.6,  scramble_yards_all.mean()),
        "Rush+scram yds":(122.1, rush_yards_all.mean() + scramble_yards_all.mean()),
        "Score":         (22.5,  scores.mean()),
        "RZ trips":      (3.30,  rz_trips.mean()),
        "RZ TDs":        (1.90,  rz_tds.mean()),
    }

    print(f"\n  {'Metric':<20} {'NFL':>8} {'Sim':>8} {'Diff':>8} {'Diff%':>8}")
    print("  " + "-" * 55)
    for label, (nfl_val, sim_val) in actuals.items():
        diff = sim_val - nfl_val
        pct = (diff / nfl_val * 100) if nfl_val != 0 else 0
        print(f"  {label:<20} {nfl_val:>8.1f} {sim_val:>8.1f} "
              f"{diff:>+8.1f} {pct:>+7.1f}%")

    # Derived ratios
    print(f"\n  Derived ratios:")
    nfl_pts_per_yd = 22.5 / 336.6
    sim_pts_per_yd = scores.mean() / total_yards_all.mean()
    print(f"    Points/yard:      NFL={nfl_pts_per_yd:.4f}  "
          f"Sim={sim_pts_per_yd:.4f}  (ratio={sim_pts_per_yd/nfl_pts_per_yd:.2f})")

    nfl_pts_per_drive = 22.5 / 10.5
    sim_pts_per_drive = scores.mean() / drives_total.mean()
    print(f"    Points/drive:     NFL={nfl_pts_per_drive:.2f}    "
          f"Sim={sim_pts_per_drive:.2f}    (ratio={sim_pts_per_drive/nfl_pts_per_drive:.2f})")

    nfl_yds_per_drive = 336.6 / 10.5
    sim_yds_per_drive = total_yards_all.mean() / drives_total.mean()
    print(f"    Yards/drive:      NFL={nfl_yds_per_drive:.1f}   "
          f"Sim={sim_yds_per_drive:.1f}   (ratio={sim_yds_per_drive/nfl_yds_per_drive:.2f})")

    nfl_rz_td_rate = 1.9 / 3.3
    sim_rz_td_rate = rz_tds.mean() / rz_trips.mean() if rz_trips.mean() > 0 else 0
    print(f"    RZ TD rate:       NFL={nfl_rz_td_rate:.3f}  "
          f"Sim={sim_rz_td_rate:.3f}  (ratio={sim_rz_td_rate/nfl_rz_td_rate:.2f})")

    nfl_rz_trip_rate = 3.3 / 10.5
    sim_rz_trip_rate = rz_trips.mean() / drives_total.mean()
    print(f"    RZ trip/drive:    NFL={nfl_rz_trip_rate:.3f}  "
          f"Sim={sim_rz_trip_rate:.3f}  (ratio={sim_rz_trip_rate/nfl_rz_trip_rate:.2f})")

    # Drive outcome distribution
    print(f"\n  Drive outcomes (per team-game):")
    nfl_outcomes = {
        "td": 2.38, "fg": 1.70, "fg_miss": 0.32,
        "punt": 3.90, "turnover": 1.11,
        "turnover_on_downs": 0.61, "end_of_half": 0.81,
    }
    print(f"    {'Outcome':<22} {'NFL':>6} {'Sim':>6} {'Diff':>7}")
    print("    " + "-" * 45)
    for oc_name in ["td", "fg", "fg_miss", "punt", "turnover",
                    "turnover_on_downs", "end_of_half", "kneel"]:
        sim_val = np.array(outcome_counts[oc_name]).mean()
        nfl_val = nfl_outcomes.get(oc_name, 0)
        diff = sim_val - nfl_val
        print(f"    {oc_name:<22} {nfl_val:>6.2f} {sim_val:>6.2f} {diff:>+7.2f}")

    # Score decomposition
    # Points = TDs × ~6.95 (avg PAT value) + FGs × 3
    tds = np.array(outcome_counts["td"]).mean()
    fgs = np.array(outcome_counts["fg"]).mean()
    td_pts = tds * 6.95  # avg: 6 + 0.95*1 = 6.95
    fg_pts = fgs * 3
    print(f"\n  Score decomposition:")
    print(f"    TD pts:  {tds:.2f} TDs × 6.95 = {td_pts:.1f}")
    print(f"    FG pts:  {fgs:.2f} FGs × 3    = {fg_pts:.1f}")
    print(f"    Total:   {td_pts + fg_pts:.1f} (actual sim: {scores.mean():.1f})")

    nfl_td_pts = 2.38 * 6.95
    nfl_fg_pts = 1.70 * 3
    print(f"    NFL TD:  2.38 TDs × 6.95 = {nfl_td_pts:.1f}")
    print(f"    NFL FG:  1.70 FGs × 3    = {nfl_fg_pts:.1f}")
    print(f"    NFL tot: {nfl_td_pts + nfl_fg_pts:.1f} (actual: 22.5)")

    # Starting field position
    start_yls = []
    for i in range(2000):
        res = sim_game_with_drives(np.random.default_rng(1000+i), LG)
        for d in res["home_drives"]:
            start_yls.append(d["start_yl"])
    start_yls = np.array(start_yls)
    print(f"\n  Starting field position:")
    print(f"    Mean: {start_yls.mean():.1f}  (NFL ~73)")
    print(f"    SD:   {start_yls.std():.1f}  (NFL ~8)")
    print(f"    p10:  {np.percentile(start_yls, 10):.0f}  "
          f"p50: {np.percentile(start_yls, 50):.0f}  "
          f"p90: {np.percentile(start_yls, 90):.0f}")

    # Plays per drive by outcome
    print(f"\n  Plays/drive by outcome:")
    # Re-run to collect
    ppd_by_outcome = defaultdict(list)
    ypd_by_outcome = defaultdict(list)
    for i in range(N):
        res = sim_game_with_drives(np.random.default_rng(5000+i), LG)
        for d in res["home_drives"]:
            ppd_by_outcome[d["outcome"]].append(d["n_plays"])
            ypd_by_outcome[d["outcome"]].append(d["total_yards"])
    for oc in ["td", "fg", "punt", "turnover", "turnover_on_downs"]:
        if ppd_by_outcome[oc]:
            pp = np.mean(ppd_by_outcome[oc])
            yy = np.mean(ypd_by_outcome[oc])
            print(f"    {oc:<22} plays={pp:.1f}  yards={yy:.1f}")


if __name__ == "__main__":
    main()
