#!/usr/bin/env python3
"""
Four-Cell Cross-Book Split — Disambiguate the side-split finding
================================================================
Prior analysis found "+1.5 has edge, -1.5 reverses" — but pooled
by RUNG, not by TEAM-ROLE. This could be hiding the real axis.

THE FOUR BETS:
  Cell 1: FAV -1.5  (favorite wins by 2+) — aggressive favorite lay
  Cell 2: FAV +1.5  (favorite wins or loses by 1) — ultra-safe chalk
  Cell 3: DOG +1.5  (dog loses by ≤1 or wins) — classic take-the-points
  Cell 4: DOG -1.5  (dog wins by 2+) — longshot underdog

If the edge is a RUNG effect: cells 2+3 (all +1.5) beat cells 1+4 (all -1.5)
If the edge is a TEAM-ROLE effect: cells 3+4 (all dog) beat cells 1+2 (all fav)
If it's CELL-SPECIFIC: only cell 3 (dog +1.5) carries it

Data: alt_lines (Pinnacle + soft books), outcomes from games table.
"""

import sqlite3
import statistics
from collections import defaultdict

DB_PATH = "data/historical.db"


def _implied(odds):
    return 1.0 / odds if odds > 1.0 else 1.0


def _devig(odds_a, odds_b):
    ia, ib = _implied(odds_a), _implied(odds_b)
    t = ia + ib
    return (ia / t, ib / t) if t > 0 else (0.5, 0.5)


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print("=" * 70)
    print("FOUR-CELL CROSS-BOOK SPLIT")
    print("Is the edge a RUNG effect (+1.5 vs -1.5) or")
    print("a TEAM-ROLE effect (underdog vs favorite)?")
    print("=" * 70)

    # ── Load outcomes ──
    outcomes = {}
    for r in db.execute("""
        SELECT home_team, away_team, game_date, home_score, away_score
        FROM games WHERE sport='MLB' AND home_score IS NOT NULL
    """).fetchall():
        outcomes[(r["home_team"], r["away_team"], r["game_date"])] = (
            r["home_score"], r["away_score"])

    # ── Load alt_lines: latest snapshot per (game, book, team, line) ──
    alt_rows = db.execute("""
        SELECT fetched_at, bookmaker, team, odds, line,
               home_team, away_team, game_date_ct
        FROM alt_lines
        WHERE sport_key='baseball_mlb' AND market_key='alternate_spreads'
          AND line IN (-1.5, 1.5)
          AND bookmaker IN ('pinnacle', 'draftkings', 'fanduel', 'betmgm')
        ORDER BY fetched_at DESC
    """).fetchall()

    # Keep latest per (game, book, team, line)
    alt_latest = {}
    for r in alt_rows:
        key = (r["home_team"], r["away_team"], r["game_date_ct"],
               r["bookmaker"], r["team"], r["line"])
        if key not in alt_latest:
            alt_latest[key] = r["odds"]

    # Organize: {(home, away, date): {book: {(team, line): odds}}}
    games_data = defaultdict(lambda: defaultdict(dict))
    for (home, away, gdate, book, team, line), odds in alt_latest.items():
        games_data[(home, away, gdate)][book][(team, line)] = odds

    print(f"\n  Games with alt_lines: {len(games_data)}")
    print(f"  Games with outcomes: {len(outcomes)}")

    # ── For each game, build all 4 cells ──
    # Determine favorite: team with LOWER -1.5 odds (= higher implied prob)
    # Use Pinnacle if available, else consensus

    cells = {
        "fav_neg15": [],  # Cell 1: Favorite -1.5
        "fav_pos15": [],  # Cell 2: Favorite +1.5
        "dog_pos15": [],  # Cell 3: Dog +1.5
        "dog_neg15": [],  # Cell 4: Dog -1.5
    }

    matched = 0
    skipped_no_outcome = 0
    skipped_no_pair = 0

    for (home, away, gdate), books in games_data.items():
        outcome = outcomes.get((home, away, gdate))
        if not outcome:
            skipped_no_outcome += 1
            continue

        home_score, away_score = outcome
        home_won = home_score > away_score
        margin = home_score - away_score  # positive = home won

        # Determine favorite using Pinnacle or consensus
        # Favorite = team whose -1.5 has lower odds (higher probability)
        ref_book = "pinnacle" if "pinnacle" in books else None
        if not ref_book:
            # Use consensus
            for b in ["draftkings", "fanduel", "betmgm"]:
                if b in books:
                    ref_book = b
                    break
        if not ref_book:
            continue

        home_neg15_odds = books.get(ref_book, {}).get((home, -1.5))
        away_neg15_odds = books.get(ref_book, {}).get((away, -1.5))

        if not home_neg15_odds or not away_neg15_odds:
            skipped_no_pair += 1
            continue

        # Favorite = lower odds on -1.5 (higher implied prob of covering)
        if home_neg15_odds <= away_neg15_odds:
            fav, dog = home, away
            fav_margin = margin  # positive = fav won
        else:
            fav, dog = away, home
            fav_margin = -margin  # flip so positive = fav won

        # Outcomes for each cell
        fav_covers_neg15 = fav_margin >= 2    # fav wins by 2+
        fav_covers_pos15 = fav_margin >= -1   # fav wins or loses by ≤1
        dog_covers_pos15 = fav_margin <= 1    # dog wins or loses by ≤1 (same as NOT fav_covers_neg15... wait)
        dog_covers_neg15 = fav_margin <= -2   # dog wins by 2+

        # Sanity: fav_covers_neg15 complement is dog_covers_pos15
        # But NOT exactly! fav_margin >= 2 complement is fav_margin <= 1.
        # fav_margin <= 1 means dog +1.5 covers? Let's check:
        # If fav wins by 1 (margin=1): fav -1.5 NO, dog +1.5 YES (dog lost by 1, within 1.5)
        # If fav wins by 2 (margin=2): fav -1.5 YES, dog +1.5 NO (dog lost by 2, > 1.5)
        # If dog wins by 1 (margin=-1): fav -1.5 NO, dog +1.5 YES
        # So dog_covers_pos15 = fav_margin < 2 → fav_margin <= 1

        # Actually: "covers +1.5" means the team's result + 1.5 > opponent
        # Dog score + 1.5 > Fav score → Dog score - Fav score > -1.5
        # → fav_margin < 1.5 → fav_margin <= 1 (integer scores)
        dog_covers_pos15 = fav_margin <= 1

        # Similarly: fav_covers_pos15 = fav score + 1.5 > dog score
        # → fav_margin > -1.5 → fav_margin >= -1 (integer)
        fav_covers_pos15 = fav_margin >= -1

        # For each soft book, compute de-vigged fair probs and compare to Pinnacle
        for soft_book in ["draftkings", "fanduel", "betmgm"]:
            if soft_book not in books:
                continue

            sb = books[soft_book]
            pin = books.get("pinnacle", {})

            # We need BOTH de-vig pairs for all 4 cells:
            # Pair A: fav -1.5 vs dog +1.5
            # Pair B: dog -1.5 vs fav +1.5

            # Pair A: fav -1.5 / dog +1.5
            sb_fav_neg15 = sb.get((fav, -1.5))
            sb_dog_pos15 = sb.get((dog, 1.5))
            pin_fav_neg15 = pin.get((fav, -1.5))
            pin_dog_pos15 = pin.get((dog, 1.5))

            if sb_fav_neg15 and sb_dog_pos15 and pin_fav_neg15 and pin_dog_pos15:
                if sb_fav_neg15 > 1 and sb_dog_pos15 > 1 and pin_fav_neg15 > 1 and pin_dog_pos15 > 1:
                    sb_fair_fav_neg15, sb_fair_dog_pos15 = _devig(sb_fav_neg15, sb_dog_pos15)
                    pin_fair_fav_neg15, pin_fair_dog_pos15 = _devig(pin_fav_neg15, pin_dog_pos15)

                    diff_fav_neg15 = (sb_fair_fav_neg15 - pin_fair_fav_neg15) * 100
                    diff_dog_pos15 = (sb_fair_dog_pos15 - pin_fair_dog_pos15) * 100

                    # Overround
                    sb_vig_a = (_implied(sb_fav_neg15) + _implied(sb_dog_pos15) - 1) * 100
                    pin_vig_a = (_implied(pin_fav_neg15) + _implied(pin_dog_pos15) - 1) * 100

                    cells["fav_neg15"].append({
                        "book": soft_book,
                        "soft_fair": sb_fair_fav_neg15,
                        "pin_fair": pin_fair_fav_neg15,
                        "diff_pp": diff_fav_neg15,
                        "won": fav_covers_neg15,
                        "soft_odds": sb_fav_neg15,
                        "vig": sb_vig_a,
                    })
                    cells["dog_pos15"].append({
                        "book": soft_book,
                        "soft_fair": sb_fair_dog_pos15,
                        "pin_fair": pin_fair_dog_pos15,
                        "diff_pp": diff_dog_pos15,
                        "won": dog_covers_pos15,
                        "soft_odds": sb_dog_pos15,
                        "vig": sb_vig_a,
                    })

            # Pair B: dog -1.5 / fav +1.5
            sb_dog_neg15 = sb.get((dog, -1.5))
            sb_fav_pos15 = sb.get((fav, 1.5))
            pin_dog_neg15 = pin.get((dog, -1.5))
            pin_fav_pos15 = pin.get((fav, 1.5))

            if sb_dog_neg15 and sb_fav_pos15 and pin_dog_neg15 and pin_fav_pos15:
                if sb_dog_neg15 > 1 and sb_fav_pos15 > 1 and pin_dog_neg15 > 1 and pin_fav_pos15 > 1:
                    sb_fair_dog_neg15, sb_fair_fav_pos15 = _devig(sb_dog_neg15, sb_fav_pos15)
                    pin_fair_dog_neg15, pin_fair_fav_pos15 = _devig(pin_dog_neg15, pin_fav_pos15)

                    diff_dog_neg15 = (sb_fair_dog_neg15 - pin_fair_dog_neg15) * 100
                    diff_fav_pos15 = (sb_fair_fav_pos15 - pin_fair_fav_pos15) * 100

                    sb_vig_b = (_implied(sb_dog_neg15) + _implied(sb_fav_pos15) - 1) * 100
                    pin_vig_b = (_implied(pin_dog_neg15) + _implied(pin_fav_pos15) - 1) * 100

                    cells["dog_neg15"].append({
                        "book": soft_book,
                        "soft_fair": sb_fair_dog_neg15,
                        "pin_fair": pin_fair_dog_neg15,
                        "diff_pp": diff_dog_neg15,
                        "won": dog_covers_neg15,
                        "soft_odds": sb_dog_neg15,
                        "vig": sb_vig_b,
                    })
                    cells["fav_pos15"].append({
                        "book": soft_book,
                        "soft_fair": sb_fair_fav_pos15,
                        "pin_fair": pin_fair_fav_pos15,
                        "diff_pp": diff_fav_pos15,
                        "won": fav_covers_pos15,
                        "soft_odds": sb_fav_pos15,
                        "vig": sb_vig_b,
                    })

        matched += 1

    print(f"  Matched games: {matched}")
    print(f"  Skipped (no outcome): {skipped_no_outcome}")
    print(f"  Skipped (no pair): {skipped_no_pair}")

    # ═══════════════════════════════════════════════════════════════════
    # REPORT: ALL 4 CELLS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("THE 4 CELLS — baseline stats")
    print("=" * 70)

    labels = {
        "fav_neg15": "Cell 1: FAV -1.5 (fav wins by 2+)",
        "fav_pos15": "Cell 2: FAV +1.5 (fav wins or loses by 1)",
        "dog_pos15": "Cell 3: DOG +1.5 (dog loses by ≤1 or wins)",
        "dog_neg15": "Cell 4: DOG -1.5 (dog wins by 2+)",
    }

    for cell_name in ["fav_neg15", "fav_pos15", "dog_pos15", "dog_neg15"]:
        data = cells[cell_name]
        if not data:
            print(f"\n  {labels[cell_name]}: NO DATA")
            continue

        n = len(data)
        wr = sum(d["won"] for d in data) / n * 100
        avg_fair = statistics.mean([d["soft_fair"] for d in data]) * 100
        avg_pin = statistics.mean([d["pin_fair"] for d in data]) * 100
        avg_diff = statistics.mean([d["diff_pp"] for d in data])
        avg_vig = statistics.mean([d["vig"] for d in data])

        print(f"\n  {labels[cell_name]}:")
        print(f"    N={n}, WR={wr:.1f}%, Avg soft fair={avg_fair:.1f}%, "
              f"Avg pin fair={avg_pin:.1f}%")
        print(f"    Avg soft-pin disagreement: {avg_diff:+.2f}pp")
        print(f"    Avg vig: {avg_vig:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # KEY TEST: When soft disagrees with Pinnacle (>1pp), which cells win?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("WHEN SOFT DISAGREES WITH PINNACLE (>1pp)")
    print("Bet the soft book's side — does it win?")
    print("=" * 70)

    for cell_name in ["fav_neg15", "fav_pos15", "dog_pos15", "dog_neg15"]:
        data = cells[cell_name]
        label = labels[cell_name]

        # Bets where soft thinks prob is HIGHER than Pinnacle (soft overprices)
        over = [d for d in data if d["diff_pp"] > 1.0]  # soft > pin by 1pp
        under = [d for d in data if d["diff_pp"] < -1.0]  # soft < pin by 1pp

        print(f"\n  {label}:")

        if over:
            wr_over = sum(d["won"] for d in over) / len(over) * 100
            exp_over = statistics.mean([d["soft_fair"] for d in over]) * 100
            pin_exp = statistics.mean([d["pin_fair"] for d in over]) * 100
            avg_diff = statistics.mean([d["diff_pp"] for d in over])
            print(f"    SOFT OVERPRICES this bet (soft > pin by avg {avg_diff:.1f}pp):")
            print(f"      N={len(over)}, WR={wr_over:.1f}%, "
                  f"soft exp={exp_over:.1f}%, pin exp={pin_exp:.1f}%")
            print(f"      Edge vs pin: {wr_over - pin_exp:+.1f}pp, "
                  f"vs soft: {wr_over - exp_over:+.1f}pp")
        else:
            print(f"    SOFT OVERPRICES: N=0")

        if under:
            wr_under = sum(d["won"] for d in under) / len(under) * 100
            exp_under = statistics.mean([d["soft_fair"] for d in under]) * 100
            pin_exp = statistics.mean([d["pin_fair"] for d in under]) * 100
            avg_diff = statistics.mean([d["diff_pp"] for d in under])
            print(f"    SOFT UNDERPRICES this bet (soft < pin by avg {abs(avg_diff):.1f}pp):")
            print(f"      N={len(under)}, WR={wr_under:.1f}%, "
                  f"soft exp={exp_under:.1f}%, pin exp={pin_exp:.1f}%")
            print(f"      Edge vs pin: {wr_under - pin_exp:+.1f}pp, "
                  f"vs soft: {wr_under - exp_under:+.1f}pp")
        else:
            print(f"    SOFT UNDERPRICES: N=0")

    # ═══════════════════════════════════════════════════════════════════
    # AXIS TEST: RUNG vs TEAM-ROLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("AXIS TEST: RUNG effect vs TEAM-ROLE effect")
    print("=" * 70)

    # Pool by rung
    plus15 = cells["fav_pos15"] + cells["dog_pos15"]
    neg15 = cells["fav_neg15"] + cells["dog_neg15"]
    # Pool by team-role
    fav = cells["fav_neg15"] + cells["fav_pos15"]
    dog = cells["dog_neg15"] + cells["dog_pos15"]

    for label, data in [("ALL +1.5 (rung)", plus15),
                         ("ALL -1.5 (rung)", neg15),
                         ("ALL FAV (role)", fav),
                         ("ALL DOG (role)", dog)]:
        if not data:
            continue
        wr = sum(d["won"] for d in data) / len(data) * 100
        avg_diff = statistics.mean([d["diff_pp"] for d in data])

        # Edge when soft disagrees > 1pp (bet soft's side)
        disagree = [d for d in data if abs(d["diff_pp"]) > 1.0]
        if disagree:
            # Soft overprices: bet it → does it win?
            over = [d for d in disagree if d["diff_pp"] > 1.0]
            under = [d for d in disagree if d["diff_pp"] < -1.0]

            over_edge = None
            if over:
                over_wr = sum(d["won"] for d in over) / len(over) * 100
                over_exp = statistics.mean([d["pin_fair"] for d in over]) * 100
                over_edge = over_wr - over_exp

            under_edge = None
            if under:
                under_wr = sum(d["won"] for d in under) / len(under) * 100
                under_exp = statistics.mean([d["pin_fair"] for d in under]) * 100
                under_edge = under_wr - under_exp

            print(f"\n  {label}: N={len(data)}, base WR={wr:.1f}%, avg diff={avg_diff:+.2f}pp")
            if over_edge is not None:
                print(f"    Soft overprices → bet it: {over_edge:+.1f}pp edge vs pin (N={len(over)})")
            if under_edge is not None:
                print(f"    Soft underprices → fade it: {-under_edge:+.1f}pp edge fading (N={len(under)})")

    # ═══════════════════════════════════════════════════════════════════
    # CALIBRATION CHECK: who is actually better calibrated?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CALIBRATION: BRIER BY CELL")
    print("(who predicts each cell's outcome better?)")
    print("=" * 70)

    for cell_name in ["fav_neg15", "fav_pos15", "dog_pos15", "dog_neg15"]:
        data = cells[cell_name]
        if not data:
            continue

        soft_brier = statistics.mean(
            [(d["soft_fair"] - (1 if d["won"] else 0)) ** 2 for d in data])
        pin_brier = statistics.mean(
            [(d["pin_fair"] - (1 if d["won"] else 0)) ** 2 for d in data])

        label = labels[cell_name]
        better = "SOFT" if soft_brier < pin_brier else "PINNACLE"
        print(f"\n  {label}:")
        print(f"    Soft Brier: {soft_brier:.6f}, Pin Brier: {pin_brier:.6f}")
        print(f"    Gap: {soft_brier - pin_brier:+.6f} → {better} is sharper")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICT: RUNG or TEAM-ROLE?")
    print("=" * 70)

    # Compute per-cell edge when soft disagrees
    cell_edges = {}
    for cell_name in ["fav_neg15", "fav_pos15", "dog_pos15", "dog_neg15"]:
        data = cells[cell_name]
        over = [d for d in data if d["diff_pp"] > 1.0]
        if len(over) >= 20:
            wr = sum(d["won"] for d in over) / len(over) * 100
            exp = statistics.mean([d["pin_fair"] for d in over]) * 100
            cell_edges[cell_name] = (wr - exp, len(over))
        else:
            cell_edges[cell_name] = (None, len(over))

    print(f"""
  PER-CELL EDGE (soft overprices → bet it, edge vs Pinnacle):
    Cell 1 FAV -1.5:  {f'{cell_edges["fav_neg15"][0]:+.1f}pp (N={cell_edges["fav_neg15"][1]})' if cell_edges["fav_neg15"][0] is not None else f'N={cell_edges["fav_neg15"][1]} (too few)'}
    Cell 2 FAV +1.5:  {f'{cell_edges["fav_pos15"][0]:+.1f}pp (N={cell_edges["fav_pos15"][1]})' if cell_edges["fav_pos15"][0] is not None else f'N={cell_edges["fav_pos15"][1]} (too few)'}
    Cell 3 DOG +1.5:  {f'{cell_edges["dog_pos15"][0]:+.1f}pp (N={cell_edges["dog_pos15"][1]})' if cell_edges["dog_pos15"][0] is not None else f'N={cell_edges["dog_pos15"][1]} (too few)'}
    Cell 4 DOG -1.5:  {f'{cell_edges["dog_neg15"][0]:+.1f}pp (N={cell_edges["dog_neg15"][1]})' if cell_edges["dog_neg15"][0] is not None else f'N={cell_edges["dog_neg15"][1]} (too few)'}
""")

    # Determine axis
    pos15_edges = [cell_edges[c][0] for c in ["fav_pos15", "dog_pos15"]
                   if cell_edges[c][0] is not None]
    neg15_edges = [cell_edges[c][0] for c in ["fav_neg15", "dog_neg15"]
                   if cell_edges[c][0] is not None]
    fav_edges = [cell_edges[c][0] for c in ["fav_neg15", "fav_pos15"]
                 if cell_edges[c][0] is not None]
    dog_edges = [cell_edges[c][0] for c in ["dog_neg15", "dog_pos15"]
                 if cell_edges[c][0] is not None]

    avg_pos15 = statistics.mean(pos15_edges) if pos15_edges else None
    avg_neg15 = statistics.mean(neg15_edges) if neg15_edges else None
    avg_fav = statistics.mean(fav_edges) if fav_edges else None
    avg_dog = statistics.mean(dog_edges) if dog_edges else None

    print(f"  BY RUNG:       +1.5 avg={f'{avg_pos15:+.1f}pp' if avg_pos15 else '?'}, "
          f"-1.5 avg={f'{avg_neg15:+.1f}pp' if avg_neg15 else '?'}")
    print(f"  BY TEAM-ROLE:  Dog avg={f'{avg_dog:+.1f}pp' if avg_dog else '?'}, "
          f"Fav avg={f'{avg_fav:+.1f}pp' if avg_fav else '?'}")

    rung_gap = abs((avg_pos15 or 0) - (avg_neg15 or 0))
    role_gap = abs((avg_dog or 0) - (avg_fav or 0))

    print(f"\n  Rung gap: {rung_gap:.1f}pp")
    print(f"  Role gap: {role_gap:.1f}pp")

    if rung_gap > role_gap + 1:
        print(f"\n  → RUNG EFFECT dominates: +1.5 vs -1.5 is the primary axis")
    elif role_gap > rung_gap + 1:
        print(f"\n  → TEAM-ROLE EFFECT dominates: dog vs fav is the primary axis")
    else:
        print(f"\n  → NEITHER dominates — both axes contribute or signal is noise")

    db.close()


if __name__ == "__main__":
    main()
