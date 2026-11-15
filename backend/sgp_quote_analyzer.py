"""
sgp_quote_analyzer.py — Analyze FanDuel SGP bet-slip quotes to measure
per-archetype adjustment factors and compare against true correlations.

NO BETS PLACED. Reads quoted SGP prices from data/sgp_quotes.csv.

Usage:
    python sgp_quote_analyzer.py              # analyze all quotes
    python sgp_quote_analyzer.py --fetch      # fetch live FanDuel SGP prices (ToS-gray)

CSV format (data/sgp_quotes.csv):
    timestamp,sport,game,archetype,player_a,stat_a,line_a,price_a,
    player_b,stat_b,line_b,price_b,sgp_quoted_odds,book,notes
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# True archetype correlations from pooled 2023-2024 NFL game data.
# IMPORTANT: true k varies with marginal probability (line depth).
# These are MEDIAN-split (p50) estimates. For accurate comparison against
# book SGP quotes, use the matched-probability k from the curve below.
TRUE_K = {
    "QB_passyds_passtds":   1.452,   # N=1052, p50 threshold (geo_ip ≈ 0.40)
    "WR_recyds_receptions": 1.728,   # N=3362, p50 threshold (geo_ip ≈ 0.42)
    "QB_WR_stack":          1.315,   # N=980  (QB pass_yds <-> WR1 rec_yds)
    "QB_WR2_stack":         1.280,   # N=845
    "QB_TE_stack":          1.237,   # N=849
    "RB_QB_sameteam":       0.895,   # N=933  (negative correlation)
    "QB_passyds_passints":  1.000,   # N=1052 (independent)
}

# True k by marginal probability level (geo mean of leg implied probs).
# Use this for matched-probability comparison against book quotes.
# Format: {(archetype, geo_ip): (k, ci_lo, ci_hi, n)}
TRUE_K_BY_PROB = {
    # WR rec_yds ↔ receptions (N=3362 game-level obs, 266 player-seasons)
    ("WR_recyds_receptions", 0.88): (1.077, 1.070, 1.084),
    ("WR_recyds_receptions", 0.81): (1.125, 1.117, 1.133),
    ("WR_recyds_receptions", 0.77): (1.168, 1.158, 1.179),
    ("WR_recyds_receptions", 0.65): (1.298, 1.283, 1.313),
    ("WR_recyds_receptions", 0.52): (1.502, 1.478, 1.528),
    ("WR_recyds_receptions", 0.42): (1.728, 1.688, 1.767),
    ("WR_recyds_receptions", 0.35): (1.970, 1.908, 2.028),
    ("WR_recyds_receptions", 0.22): (2.760, 2.634, 2.881),
    # QB pass_yds ↔ pass_tds (N=1052 game-level obs, 79 player-seasons)
    ("QB_passyds_passtds", 0.84): (1.034, 1.022, 1.045),
    ("QB_passyds_passtds", 0.79): (1.056, 1.038, 1.074),
    ("QB_passyds_passtds", 0.73): (1.084, 1.063, 1.107),
    ("QB_passyds_passtds", 0.62): (1.150, 1.116, 1.186),
    ("QB_passyds_passtds", 0.54): (1.222, 1.180, 1.266),
    ("QB_passyds_passtds", 0.40): (1.452, 1.372, 1.532),
    ("QB_passyds_passtds", 0.31): (1.617, 1.483, 1.747),
    ("QB_passyds_passtds", 0.20): (1.774, 1.526, 2.022),
}

# Breakeven adjustment: book must adjust WEAKER than 1/k for edge to exist.
# If book adj > breakeven, there's a gap.  If book adj <= breakeven, edge = 0.
BREAKEVEN_ADJ = {k: round(1.0 / v, 4) for k, v in TRUE_K.items()}

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sgp_quotes.csv"


def load_quotes(csv_path: str | Path = CSV_PATH) -> list[dict]:
    """Load SGP quotes from CSV, filtering out example/template rows."""
    quotes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip example rows
            sgp = row.get("sgp_quoted_odds", "").strip()
            if not sgp or sgp == "FILL":
                continue
            try:
                price_a = float(row["price_a"])
                price_b = float(row["price_b"])
                sgp_odds = float(sgp)
            except (ValueError, KeyError):
                continue

            naive = price_a * price_b
            adj = sgp_odds / naive if naive > 0 else None

            quotes.append({
                "timestamp": row.get("timestamp", ""),
                "sport": row.get("sport", ""),
                "game": row.get("game", ""),
                "archetype": row.get("archetype", ""),
                "player_a": row.get("player_a", ""),
                "stat_a": row.get("stat_a", ""),
                "price_a": price_a,
                "player_b": row.get("player_b", ""),
                "stat_b": row.get("stat_b", ""),
                "price_b": price_b,
                "sgp_quoted": sgp_odds,
                "naive_product": round(naive, 4),
                "adj_factor": round(adj, 4) if adj else None,
                "book": row.get("book", ""),
                "notes": row.get("notes", ""),
            })
    return quotes


def analyze(quotes: list[dict] | None = None) -> dict:
    """Compute per-archetype adjustment factor stats and edge verdicts.

    Returns a dict with per-archetype stats and the key comparison
    against breakeven thresholds.
    """
    if quotes is None:
        quotes = load_quotes()

    if not quotes:
        return {
            "status": "no_data",
            "message": (
                "No SGP quotes found. Record quotes in data/sgp_quotes.csv:\n"
                "  1. Open FanDuel bet slip for an NFL game\n"
                "  2. Add two legs (e.g. WR rec_yds + receptions for same player)\n"
                "  3. Record individual leg prices and the quoted SGP combo price\n"
                "  4. archetype = QB_passyds_passtds | WR_recyds_receptions | "
                "QB_WR_stack | RB_QB_sameteam"
            ),
        }

    # Group by archetype
    by_arch: dict[str, list[float]] = defaultdict(list)
    by_arch_all: list[float] = []
    for q in quotes:
        arch = q["archetype"]
        if q["adj_factor"] is not None:
            by_arch[arch].append(q["adj_factor"])
            by_arch_all.append(q["adj_factor"])

    result = {
        "total_quotes": len(quotes),
        "archetypes": {},
        "verdicts": {},
    }

    # Overall stats
    if by_arch_all:
        result["overall"] = {
            "n": len(by_arch_all),
            "median_adj": round(float(np.median(by_arch_all)), 4),
            "mean_adj": round(float(np.mean(by_arch_all)), 4),
            "std_adj": round(float(np.std(by_arch_all)), 4),
        }

    # Per-archetype
    for arch, factors in sorted(by_arch.items()):
        n = len(factors)
        med = float(np.median(factors))
        mn = float(np.mean(factors))
        true_k = TRUE_K.get(arch, 1.0)
        be_adj = BREAKEVEN_ADJ.get(arch, 1.0)

        entry = {
            "n": n,
            "median_adj": round(med, 4),
            "mean_adj": round(mn, 4),
            "min_adj": round(min(factors), 4),
            "max_adj": round(max(factors), 4),
            "true_k": true_k,
            "breakeven_adj": be_adj,
            "sufficient": n >= 10,
        }

        # Verdict
        if n < 10:
            entry["verdict"] = "INSUFFICIENT — need 10+ quotes"
        elif med > be_adj * 1.05:
            # Book adjustment is materially weaker than breakeven
            gap = med - be_adj
            entry["verdict"] = f"EDGE EXISTS — book adj {med:.3f} > breakeven {be_adj:.3f} (+{gap:.3f})"
            entry["edge_gap"] = round(gap, 4)
        elif med < be_adj * 0.95:
            # Book over-adjusts (tighter than correlation warrants)
            entry["verdict"] = f"NO EDGE — book adj {med:.3f} < breakeven {be_adj:.3f} (copula captures it)"
        else:
            entry["verdict"] = f"MARGINAL — book adj {med:.3f} ≈ breakeven {be_adj:.3f}"

        result["archetypes"][arch] = entry

    # Summary verdicts for the two target pairs
    for target, label in [
        ("WR_recyds_receptions", "WR rec_yds + receptions"),
        ("QB_passyds_passtds", "QB pass_yds + pass_tds"),
    ]:
        if target in result["archetypes"]:
            a = result["archetypes"][target]
            result["verdicts"][label] = a["verdict"]

    # Negative-pair sanity check
    neg_arch = "RB_QB_sameteam"
    if neg_arch in result["archetypes"]:
        a = result["archetypes"][neg_arch]
        if a["n"] >= 3:
            if a["median_adj"] < 1.0:
                result["verdicts"]["copula_responsive"] = (
                    f"YES — RB+QB gets haircut ({a['median_adj']:.3f}), "
                    f"book copula recognizes negative correlation"
                )
            elif a["median_adj"] > 1.05:
                result["verdicts"]["copula_responsive"] = (
                    f"YES (opposite) — RB+QB gets BOOST ({a['median_adj']:.3f}), "
                    f"book recognizes negative correlation and pays more"
                )
            else:
                result["verdicts"]["copula_responsive"] = (
                    f"UNCLEAR — RB+QB adj ≈ 1.0 ({a['median_adj']:.3f}), "
                    f"copula may not detect this pair"
                )

    return result


def print_report(result: dict | None = None):
    """Pretty-print the SGP quote analysis."""
    if result is None:
        result = analyze()

    if result.get("status") == "no_data":
        print(result["message"])
        return

    print("=" * 70)
    print("SGP BET-SLIP QUOTE ANALYSIS")
    print("=" * 70)

    if "overall" in result:
        o = result["overall"]
        print(f"\nTotal quotes: {result['total_quotes']}")
        print(f"Overall: N={o['n']}, median adj={o['median_adj']:.4f}, "
              f"mean={o['mean_adj']:.4f}, std={o['std_adj']:.4f}")

    print(f"\n{'Archetype':<25} {'N':>4} {'Median':>8} {'BE Adj':>8} {'True k':>8}  Verdict")
    print("-" * 90)
    for arch, a in sorted(result.get("archetypes", {}).items()):
        print(f"  {arch:<23} {a['n']:>4} {a['median_adj']:>8.4f} "
              f"{a['breakeven_adj']:>8.4f} {a['true_k']:>8.3f}  {a['verdict']}")

    if result.get("verdicts"):
        print(f"\n{'KEY VERDICTS':}")
        print("-" * 50)
        for label, v in result["verdicts"].items():
            print(f"  {label}: {v}")

    # Collection guide
    archs_needed = []
    for arch in ["WR_recyds_receptions", "QB_passyds_passtds", "RB_QB_sameteam"]:
        n = result.get("archetypes", {}).get(arch, {}).get("n", 0)
        if n < 10:
            archs_needed.append(f"  {arch}: {n}/10 quotes")
    if archs_needed:
        print(f"\nDATA NEEDED:")
        for a in archs_needed:
            print(a)
        print("\nRecord in data/sgp_quotes.csv (no bets needed — just read the slip).")


if __name__ == "__main__":
    print_report()
