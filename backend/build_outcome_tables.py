"""
Extract empirical play outcome rates from PBP, bucketed by
down × distance × field-position zone.

Produces a JSON lookup table for nfl_game_sim.py to replace hand-tuned
RZ compression and 3rd-down adjustments.

Also extracts 4th-down go-for-it rates from actual NFL decisions.
"""
import json
import os
import sys

import numpy as np
import nfl_data_py as nfl

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "data", "sim_outcome_tables.json")

# ── Bucket definitions ────────────────────────────────────────────────
DIST_BUCKETS = {
    "short": (1, 3),    # 1-3 yards
    "med":   (4, 6),    # 4-6 yards
    "long":  (7, 10),   # 7-10 yards
    "xlong": (11, 99),  # 11+ yards
}

YL_ZONES = {
    "own20":  (80, 99),   # own 1-20 (yl 80-99)
    "mid":    (41, 79),   # own 21 - opp 40
    "opp40":  (21, 40),   # opp 21-40
    "rz":     (1, 20),    # red zone (opp 1-20)
}


def _dist_bucket(distance):
    for name, (lo, hi) in DIST_BUCKETS.items():
        if lo <= distance <= hi:
            return name
    return "xlong"


def _yl_zone(yardline_100):
    for name, (lo, hi) in YL_ZONES.items():
        if lo <= yardline_100 <= hi:
            return name
    return "mid"


def build_tables(seasons=None):
    if seasons is None:
        seasons = [2023, 2024, 2025]

    print(f"Loading PBP for {seasons}...")
    pbp = nfl.import_pbp_data(seasons)

    # Filter to real plays (pass/run), exclude 2pt conversions and spikes
    plays = pbp[
        pbp["play_type"].isin(["pass", "run"]) &
        (pbp["two_point_attempt"] != 1) &
        (pbp["aborted_play"] != 1)
    ].copy()

    print(f"  {len(plays)} plays loaded")

    # Key columns
    plays["is_dropback"] = ((plays["pass_attempt"] == 1) |
                            (plays["qb_scramble"] == 1) |
                            (plays["sack"] == 1)).astype(int)
    plays["is_sack"] = (plays["sack"] == 1).astype(int)
    plays["is_scramble"] = (plays["qb_scramble"] == 1).astype(int)
    plays["is_completion"] = (
        (plays["pass_attempt"] == 1) &
        (plays["complete_pass"] == 1) &
        (plays["sack"] != 1)
    ).astype(int)
    plays["is_interception"] = (plays["interception"] == 1).astype(int)
    # Designed rushes only (exclude scrambles — scrambles are handled
    # separately in the sim via sack/scramble branch of dropbacks).
    plays["is_rush"] = (
        (plays["rush_attempt"] == 1) & (plays["qb_scramble"] != 1)
    ).astype(int)
    plays["is_td"] = (plays["touchdown"] == 1).astype(int)
    plays["is_fumble_lost"] = (plays["fumble_lost"] == 1).astype(int)

    # Use yardline_100 directly (already in PBP)
    plays["yl"] = plays["yardline_100"].fillna(50).astype(int).clip(1, 99)

    # Buckets
    plays["dist_bucket"] = plays["ydstogo"].fillna(10).astype(int).apply(_dist_bucket)
    plays["yl_zone"] = plays["yl"].apply(_yl_zone)

    # ── Build outcome tables ──────────────────────────────────────────
    print("Building outcome tables...")
    outcome_tables = {}

    for down in [1, 2, 3, 4]:
        for db in DIST_BUCKETS:
            for yz in YL_ZONES:
                mask = (
                    (plays["down"] == down) &
                    (plays["dist_bucket"] == db) &
                    (plays["yl_zone"] == yz)
                )
                subset = plays[mask]
                n = len(subset)
                if n < 30:
                    continue

                dropbacks = subset[subset["is_dropback"] == 1]
                pass_atts = subset[subset["pass_attempt"] == 1]
                non_sack_pass = pass_atts[pass_atts["sack"] != 1]
                completions = subset[subset["is_completion"] == 1]
                rushes = subset[subset["is_rush"] == 1]
                scrambles = subset[subset["is_scramble"] == 1]

                n_db = len(dropbacks)
                n_ns = len(non_sack_pass)

                # Rates (per dropback for sack/scramble, per non-sack att for comp/INT)
                comp_rate = float(non_sack_pass["complete_pass"].mean()) if n_ns > 10 else None
                int_rate = float(non_sack_pass["interception"].mean()) if n_ns > 10 else None
                sack_rate = float(dropbacks["is_sack"].mean()) if n_db > 10 else None
                scramble_rate = float(dropbacks["is_scramble"].mean()) if n_db > 10 else None

                # Y/C: yards per completion (gross)
                comp_yards = completions["yards_gained"].dropna()
                ypc_comp = float(comp_yards.mean()) if len(comp_yards) > 10 else None
                ypc_comp_sd = float(comp_yards.std()) if len(comp_yards) > 10 else None

                # YPC: yards per rush (designed carries only, not scrambles)
                rush_yards = rushes["yards_gained"].dropna()
                ypc_rush = float(rush_yards.mean()) if len(rush_yards) > 10 else None
                ypc_rush_sd = float(rush_yards.std()) if len(rush_yards) > 10 else None

                # Scramble yards
                scram_yards = scrambles["yards_gained"].dropna()
                ypc_scram = float(scram_yards.mean()) if len(scram_yards) > 5 else None

                # TD rate per play (useful for calibration)
                td_rate = float(subset["is_td"].mean()) if n > 20 else None

                # Fumble rate
                fumble_rate = float(subset["is_fumble_lost"].mean()) if n > 30 else None

                # First down conversion rate
                first_down_rate = None
                if "first_down" in subset.columns:
                    first_down_rate = float(subset["first_down"].fillna(0).mean())

                # Pass rate (how often the NFL actually passes in this situation)
                pass_rate = float(subset["is_dropback"].mean()) if n > 20 else None

                key = f"{down}|{db}|{yz}"
                outcome_tables[key] = {
                    "n": n,
                    "comp_rate": round(comp_rate, 4) if comp_rate is not None else None,
                    "int_rate": round(int_rate, 4) if int_rate is not None else None,
                    "sack_rate": round(sack_rate, 4) if sack_rate is not None else None,
                    "scramble_rate": round(scramble_rate, 4) if scramble_rate is not None else None,
                    "ypc_comp": round(ypc_comp, 2) if ypc_comp is not None else None,
                    "ypc_comp_sd": round(ypc_comp_sd, 2) if ypc_comp_sd is not None else None,
                    "ypc_rush": round(ypc_rush, 2) if ypc_rush is not None else None,
                    "ypc_rush_sd": round(ypc_rush_sd, 2) if ypc_rush_sd is not None else None,
                    "ypc_scram": round(ypc_scram, 2) if ypc_scram is not None else None,
                    "td_rate": round(td_rate, 4) if td_rate is not None else None,
                    "fumble_rate": round(fumble_rate, 4) if fumble_rate is not None else None,
                    "first_down_rate": round(first_down_rate, 4) if first_down_rate is not None else None,
                    "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
                }

    # ── 4th-down decisions ────────────────────────────────────────────
    print("Extracting 4th-down decisions...")
    fourth_down = pbp[
        (pbp["down"] == 4) &
        pbp["play_type"].isin(["pass", "run", "punt", "field_goal"])
    ].copy()

    fourth_down["went_for_it"] = fourth_down["play_type"].isin(["pass", "run"]).astype(int)
    fourth_down["punted"] = (fourth_down["play_type"] == "punt").astype(int)
    fourth_down["kicked_fg"] = (fourth_down["play_type"] == "field_goal").astype(int)
    fourth_down["yl"] = fourth_down["yardline_100"].fillna(50).astype(int).clip(1, 99)
    fourth_down["dist_bucket"] = fourth_down["ydstogo"].fillna(10).astype(int).apply(_dist_bucket)
    fourth_down["yl_zone"] = fourth_down["yl"].apply(_yl_zone)

    fourth_down_rates = {}
    for db in DIST_BUCKETS:
        for yz in YL_ZONES:
            mask = (
                (fourth_down["dist_bucket"] == db) &
                (fourth_down["yl_zone"] == yz)
            )
            subset = fourth_down[mask]
            n = len(subset)
            if n < 10:
                continue

            key = f"{db}|{yz}"
            fourth_down_rates[key] = {
                "n": n,
                "go_for_it": round(float(subset["went_for_it"].mean()), 4),
                "punt": round(float(subset["punted"].mean()), 4),
                "fg": round(float(subset["kicked_fg"].mean()), 4),
            }

    # ── Summary stats ──────────────────────────────────────────────────
    print(f"\n  Outcome table: {len(outcome_tables)} buckets")
    print(f"  4th-down decisions: {len(fourth_down_rates)} buckets")

    # Print key comparisons
    print("\n  Comp rate by down (all zones, med distance):")
    for down in [1, 2, 3, 4]:
        for yz in ["mid", "rz"]:
            key = f"{down}|med|{yz}"
            if key in outcome_tables:
                t = outcome_tables[key]
                cr = f"{t['comp_rate']:.3f}" if t['comp_rate'] is not None else "  N/A"
                yc = f"{t['ypc_comp']:.1f}" if t['ypc_comp'] is not None else " N/A"
                sr = f"{t['sack_rate']:.3f}" if t['sack_rate'] is not None else "  N/A"
                ir = f"{t['int_rate']:.4f}" if t['int_rate'] is not None else "   N/A"
                yr = f"{t['ypc_rush']:.1f}" if t['ypc_rush'] is not None else " N/A"
                print(f"    D{down} med {yz:>5}: comp={cr}  "
                      f"Y/C={yc}  sack={sr}  INT={ir}  YPC={yr}  "
                      f"n={t['n']}")

    print("\n  RZ outcomes by down (short distance):")
    for down in [1, 2, 3, 4]:
        key = f"{down}|short|rz"
        if key in outcome_tables:
            t = outcome_tables[key]
            cr = f"{t['comp_rate']:.3f}" if t['comp_rate'] is not None else "  N/A"
            yc = f"{t['ypc_comp']:.1f}" if t['ypc_comp'] is not None else " N/A"
            yr = f"{t['ypc_rush']:.1f}" if t['ypc_rush'] is not None else " N/A"
            td = f"{t['td_rate']:.3f}" if t['td_rate'] is not None else "  N/A"
            print(f"    D{down} short rz: comp={cr}  "
                  f"Y/C={yc}  YPC={yr}  TD={td}  n={t['n']}")

    print("\n  4th-down go-for-it rates:")
    for db in ["short", "med", "long", "xlong"]:
        for yz in ["rz", "opp40", "mid", "own20"]:
            key = f"{db}|{yz}"
            if key in fourth_down_rates:
                r = fourth_down_rates[key]
                print(f"    {db:>5} {yz:>5}: go={r['go_for_it']:.2f}  "
                      f"punt={r['punt']:.2f}  fg={r['fg']:.2f}  n={r['n']}")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "seasons": seasons,
        "dist_buckets": {k: list(v) for k, v in DIST_BUCKETS.items()},
        "yl_zones": {k: list(v) for k, v in YL_ZONES.items()},
        "outcome_tables": outcome_tables,
        "fourth_down_rates": fourth_down_rates,
    }

    with open(_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {_OUT}")

    return output


if __name__ == "__main__":
    build_tables()
