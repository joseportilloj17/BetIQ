"""
Signal Conflict Tracker — backfill and analyze same-game opposing picks.

NOT wired into live pick generation. This is a read-only analysis tool that
populates signal_conflict_outcomes and reports signal reliability when
different signals (cover_rate, pi, scout, model_wp) disagree.
"""

import re
import sqlite3
from collections import defaultdict
from datetime import datetime


DB_PATH = "data/bets.db"

_TEAM_RE = re.compile(r'^(.+?)\s+[+\-\(]')


def _extract_team(description: str) -> str | None:
    """Extract team name from leg description like 'KC Royals +1.5 (Alt Spread)'."""
    m = _TEAM_RE.match(description or "")
    if m:
        return m.group(1).strip()
    # h2h: "Team Name (h2h)" or "Team Name (Moneyline)"
    for suffix in (" (h2h)", " (Moneyline)"):
        if (description or "").endswith(suffix):
            return description[: -len(suffix)].strip()
    return None


def create_table(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_conflict_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id TEXT,
            game_date TEXT,
            pick_a TEXT,
            pick_b TEXT,
            spread_a REAL,
            spread_b REAL,
            cover_rate_a REAL,
            cover_rate_b REAL,
            cover_rate_favored TEXT,
            pi_diff_a REAL,
            pi_diff_b REAL,
            pi_favored TEXT,
            scout_a TEXT,
            scout_b TEXT,
            scout_favored TEXT,
            model_wp_a REAL,
            model_wp_b REAL,
            model_wp_favored TEXT,
            result_a TEXT,
            result_b TEXT,
            outcome_type TEXT,
            winner TEXT,
            cover_rate_correct INTEGER,
            pi_correct INTEGER,
            scout_correct INTEGER,
            model_wp_correct INTEGER,
            settled_at DATETIME
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_sco_fixture ON signal_conflict_outcomes(fixture_id)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_sco_outcome ON signal_conflict_outcomes(outcome_type)"
    )
    con.commit()


_SCOUT_RANK = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}


def _scout_favored(sa: str | None, sb: str | None) -> str | None:
    ra = _SCOUT_RANK.get(sa)
    rb = _SCOUT_RANK.get(sb)
    if ra is None or rb is None:
        return None
    if ra > rb:
        return "a"
    if rb > ra:
        return "b"
    return "tie"


def _build_cover_rate_lookup(con: sqlite3.Connection) -> dict[str, float]:
    """Build 14d rolling cover rate per description from settled production legs."""
    rows = con.execute("""
        SELECT mbl.description,
               AVG(CASE WHEN mbl.leg_result='WIN' THEN 1.0 ELSE 0.0 END) as cr
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mb.source IN ('prospective','prospective_pm','top_picks_page')
        AND mbl.leg_result IN ('WIN','LOSS')
        AND COALESCE(mbl.excluded_from_metrics, 0) = 0
        GROUP BY mbl.description
    """).fetchall()
    return {desc: cr for desc, cr in rows}


def backfill(con: sqlite3.Connection) -> dict:
    """Find all same-game opposing-side pairs and populate signal_conflict_outcomes."""

    # Clear existing data for clean backfill
    con.execute("DELETE FROM signal_conflict_outcomes")

    cover_rates = _build_cover_rate_lookup(con)

    # Pull all settled legs with fixture_id
    rows = con.execute("""
        SELECT mbl.fixture_id, mbl.description, mbl.leg_result,
               mbl.win_prob, mbl.spread_value, mbl.scout_grade,
               mbl.pi_diff, mbl.home_team_home_pi, mbl.away_team_away_pi,
               mb.game_date, mbl.id,
               mbl.resolved_home_team, mbl.resolved_away_team
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mbl.leg_result IN ('WIN','LOSS')
        AND COALESCE(mbl.excluded_from_metrics, 0) = 0
        AND mbl.fixture_id IS NOT NULL AND mbl.fixture_id != ''
    """).fetchall()

    # Group by fixture_id
    by_fixture: dict[str, list] = defaultdict(list)
    for r in rows:
        by_fixture[r[0]].append(r)

    stats = {"total_fixtures": 0, "pairs": 0, "skipped_same_team": 0}
    now = datetime.utcnow().isoformat()

    for fid, legs in by_fixture.items():
        # Deduplicate: for each (team, spread_value), keep the leg with highest win_prob
        # This avoids duplicate rows from exploration + prospective on same pick
        best_by_key: dict[tuple, dict] = {}
        for r in legs:
            team = _extract_team(r[1])
            if not team:
                continue
            sv = r[4]
            key = (team, sv)
            home_team = r[11] or ""
            away_team = r[12] or ""
            # Determine if this pick is for the home or away team
            is_home_pick = team in home_team or home_team in team
            # Pi for this pick's team: home_team_home_pi if home pick, away_team_away_pi if away
            team_pi = r[7] if is_home_pick else r[8]  # home_pi or away_pi
            entry = {
                "fixture_id": r[0], "description": r[1], "result": r[2],
                "win_prob": r[3], "spread_value": r[4], "scout_grade": r[5],
                "pi_diff": r[6], "home_pi": r[7], "away_pi": r[8],
                "game_date": r[9], "leg_id": r[10], "team": team,
                "home_team": home_team, "away_team": away_team,
                "is_home_pick": is_home_pick, "team_pi": team_pi,
            }
            if key not in best_by_key or (entry["win_prob"] or 0) > (best_by_key[key]["win_prob"] or 0):
                best_by_key[key] = entry

        deduped = list(best_by_key.values())

        # Group by team
        by_team: dict[str, list[dict]] = defaultdict(list)
        for e in deduped:
            by_team[e["team"]].append(e)

        teams = list(by_team.keys())
        if len(teams) < 2:
            continue

        stats["total_fixtures"] += 1

        # Generate cross-team pairs
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a, team_b = teams[i], teams[j]
                legs_a = by_team[team_a]
                legs_b = by_team[team_b]

                # For each pair of (team_a leg, team_b leg), create a conflict row
                # To avoid combinatorial explosion, pair by closest spread magnitude
                # Use best leg per team (highest model WP) for the primary comparison
                best_a = max(legs_a, key=lambda x: x["win_prob"] or 0)
                best_b = max(legs_b, key=lambda x: x["win_prob"] or 0)

                a, b = best_a, best_b
                cr_a = cover_rates.get(a["description"])
                cr_b = cover_rates.get(b["description"])

                # Determine which signal favors which side
                # Cover rate
                cr_favored = None
                if cr_a is not None and cr_b is not None:
                    if cr_a > cr_b + 0.02:
                        cr_favored = "a"
                    elif cr_b > cr_a + 0.02:
                        cr_favored = "b"
                    else:
                        cr_favored = "tie"

                # Pi: compare team-specific Pi ratings
                # team_pi is the Pi for the specific team being picked
                pi_a_team = a.get("team_pi")
                pi_b_team = b.get("team_pi")
                pi_favored = None
                if pi_a_team is not None and pi_b_team is not None:
                    if pi_a_team > pi_b_team + 0.1:
                        pi_favored = "a"
                    elif pi_b_team > pi_a_team + 0.1:
                        pi_favored = "b"
                    else:
                        pi_favored = "tie"

                # Scout
                sc_favored = _scout_favored(a.get("scout_grade"), b.get("scout_grade"))

                # Model WP
                wp_a = a.get("win_prob") or 0
                wp_b = b.get("win_prob") or 0
                wp_favored = None
                if wp_a > 0 and wp_b > 0:
                    if wp_a > wp_b + 1.0:
                        wp_favored = "a"
                    elif wp_b > wp_a + 1.0:
                        wp_favored = "b"
                    else:
                        wp_favored = "tie"

                # Outcome
                res_a = a["result"]
                res_b = b["result"]
                if res_a == "WIN" and res_b == "LOSS":
                    outcome_type = "one_win_one_loss"
                    winner = "a"
                elif res_b == "WIN" and res_a == "LOSS":
                    outcome_type = "one_win_one_loss"
                    winner = "b"
                elif res_a == "WIN" and res_b == "WIN":
                    outcome_type = "both_win"
                    winner = "both"
                else:
                    outcome_type = "both_loss"
                    winner = "neither"

                # Correctness (only for one_win_one_loss)
                cr_correct = None
                pi_correct = None
                sc_correct = None
                wp_correct = None

                if outcome_type == "one_win_one_loss":
                    if cr_favored and cr_favored not in ("tie", None):
                        cr_correct = 1 if cr_favored == winner else 0
                    if pi_favored and pi_favored not in ("tie", None):
                        pi_correct = 1 if pi_favored == winner else 0
                    if sc_favored and sc_favored not in ("tie", None):
                        sc_correct = 1 if sc_favored == winner else 0
                    if wp_favored and wp_favored not in ("tie", None):
                        wp_correct = 1 if wp_favored == winner else 0

                con.execute("""
                    INSERT INTO signal_conflict_outcomes (
                        fixture_id, game_date, pick_a, pick_b,
                        spread_a, spread_b, cover_rate_a, cover_rate_b, cover_rate_favored,
                        pi_diff_a, pi_diff_b, pi_favored,
                        scout_a, scout_b, scout_favored,
                        model_wp_a, model_wp_b, model_wp_favored,
                        result_a, result_b, outcome_type, winner,
                        cover_rate_correct, pi_correct, scout_correct, model_wp_correct,
                        settled_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    fid, a["game_date"], a["team"], b["team"],
                    a["spread_value"], b["spread_value"],
                    cr_a, cr_b, cr_favored,
                    pi_a_team, pi_b_team, pi_favored,
                    a.get("scout_grade"), b.get("scout_grade"), sc_favored,
                    wp_a, wp_b, wp_favored,
                    res_a, res_b, outcome_type, winner,
                    cr_correct, pi_correct, sc_correct, wp_correct,
                    now,
                ))
                stats["pairs"] += 1

    con.commit()
    return stats


def run_assessment(con: sqlite3.Connection) -> str:
    """Run full signal reliability assessment and return formatted report."""
    lines = []

    # a) Totals by outcome_type
    rows = con.execute("""
        SELECT outcome_type, COUNT(*) FROM signal_conflict_outcomes
        GROUP BY outcome_type ORDER BY COUNT(*) DESC
    """).fetchall()
    total = sum(r[1] for r in rows)
    lines.append(f"=== (a) Same-game pairs: {total} total ===")
    for ot, cnt in rows:
        lines.append(f"  {ot:<20} {cnt:>5} ({100*cnt/total:.1f}%)")

    # b) Signal reliability on one_win_one_loss
    lines.append("\n=== (b) Signal reliability (one_win_one_loss only) ===")
    for signal in ["cover_rate", "pi", "scout", "model_wp"]:
        col = f"{signal}_correct"
        row = con.execute(f"""
            SELECT COUNT(*) as conflicts,
                   SUM(CASE WHEN {col}=1 THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN {col}=0 THEN 1 ELSE 0 END) as wrong
            FROM signal_conflict_outcomes
            WHERE outcome_type='one_win_one_loss' AND {col} IS NOT NULL
        """).fetchone()
        conflicts, correct, wrong = row
        acc = 100 * correct / conflicts if conflicts > 0 else 0
        lines.append(f"  {signal:<14} conflicts={conflicts:>4}  correct={correct:>4}  accuracy={acc:.1f}%")

    # c) Signal pair matchups — when two signals disagree, which wins?
    lines.append("\n=== (c) Signal pair matchups (when they disagree) ===")
    signal_pairs = [
        ("cover_rate", "pi"), ("cover_rate", "scout"), ("cover_rate", "model_wp"),
        ("pi", "scout"), ("pi", "model_wp"), ("scout", "model_wp"),
    ]
    for s1, s2 in signal_pairs:
        fav1 = f"{s1}_favored"
        fav2 = f"{s2}_favored"
        cor1 = f"{s1}_correct"
        cor2 = f"{s2}_correct"
        # Find cases where s1 and s2 disagree (favor different sides) and outcome is decisive
        row = con.execute(f"""
            SELECT
                COUNT(*) as n,
                SUM(CASE WHEN {cor1}=1 THEN 1 ELSE 0 END) as s1_wins,
                SUM(CASE WHEN {cor2}=1 THEN 1 ELSE 0 END) as s2_wins
            FROM signal_conflict_outcomes
            WHERE outcome_type='one_win_one_loss'
            AND {fav1} IS NOT NULL AND {fav2} IS NOT NULL
            AND {fav1} NOT IN ('tie') AND {fav2} NOT IN ('tie')
            AND {fav1} != {fav2}
            AND {cor1} IS NOT NULL AND {cor2} IS NOT NULL
        """).fetchone()
        n, s1w, s2w = row
        if n > 0:
            lines.append(f"  {s1} vs {s2}: {n} disagreements → {s1} wins {s1w} ({100*s1w/n:.0f}%), {s2} wins {s2w} ({100*s2w/n:.0f}%)")
        else:
            lines.append(f"  {s1} vs {s2}: 0 disagreements found")

    # d) Game predictability
    lines.append("\n=== (d) Game predictability ===")
    for ot in ["both_win", "both_loss", "one_win_one_loss"]:
        row = con.execute("""
            SELECT COUNT(*) FROM signal_conflict_outcomes WHERE outcome_type=?
        """, (ot,)).fetchone()
        pct = 100 * row[0] / total if total > 0 else 0
        lines.append(f"  {ot:<20} {row[0]:>5} ({pct:.1f}%)")

    # Characteristics of both_win games
    bw = con.execute("""
        SELECT AVG(model_wp_a), AVG(model_wp_b), AVG(spread_a), AVG(spread_b),
               AVG(cover_rate_a), AVG(cover_rate_b)
        FROM signal_conflict_outcomes WHERE outcome_type='both_win'
    """).fetchone()
    if bw[0]:
        lines.append(f"\n  both_win characteristics:")
        lines.append(f"    avg model WP: A={bw[0]:.1f}% B={bw[1]:.1f}%")
        lines.append(f"    avg spread:   A={bw[2]:+.1f}  B={bw[3]:+.1f}")
        cr_a_str = f"{bw[4]:.3f}" if bw[4] else "N/A"
        cr_b_str = f"{bw[5]:.3f}" if bw[5] else "N/A"
        lines.append(f"    avg cover_rate: A={cr_a_str} B={cr_b_str}")

    bl = con.execute("""
        SELECT AVG(model_wp_a), AVG(model_wp_b), AVG(spread_a), AVG(spread_b),
               AVG(cover_rate_a), AVG(cover_rate_b)
        FROM signal_conflict_outcomes WHERE outcome_type='both_loss'
    """).fetchone()
    if bl[0]:
        lines.append(f"\n  both_loss characteristics:")
        lines.append(f"    avg model WP: A={bl[0]:.1f}% B={bl[1]:.1f}%")
        lines.append(f"    avg spread:   A={bl[2]:+.1f}  B={bl[3]:+.1f}")
        cr_a_str = f"{bl[4]:.3f}" if bl[4] else "N/A"
        cr_b_str = f"{bl[5]:.3f}" if bl[5] else "N/A"
        lines.append(f"    avg cover_rate: A={cr_a_str} B={cr_b_str}")

    # e) KC vs CIN case study (Jun 4)
    lines.append("\n=== (e) Case study: KC vs CIN (Jun 4) ===")
    kc_cin = con.execute("""
        SELECT pick_a, pick_b, spread_a, spread_b,
               cover_rate_a, cover_rate_b, cover_rate_favored,
               pi_diff_a, pi_diff_b, pi_favored,
               scout_a, scout_b, scout_favored,
               model_wp_a, model_wp_b, model_wp_favored,
               result_a, result_b, outcome_type, winner,
               cover_rate_correct, pi_correct, scout_correct, model_wp_correct
        FROM signal_conflict_outcomes
        WHERE game_date = '2026-06-04'
        AND ((pick_a LIKE '%Royals%' OR pick_a LIKE '%Kansas%')
             OR (pick_b LIKE '%Royals%' OR pick_b LIKE '%Kansas%'))
        AND ((pick_a LIKE '%Reds%' OR pick_a LIKE '%Cincinnati%')
             OR (pick_b LIKE '%Reds%' OR pick_b LIKE '%Cincinnati%'))
    """).fetchall()
    if kc_cin:
        for r in kc_cin:
            lines.append(f"  {r[0]} ({r[2]:+.1f}) vs {r[1]} ({r[3]:+.1f})")
            lines.append(f"    Results: {r[0]}={r[16]}, {r[1]}={r[17]} → {r[18]} (winner={r[19]})")
            lines.append(f"    Cover rate: favored={r[6]} correct={r[20]}")
            lines.append(f"    Pi:         favored={r[9]} correct={r[21]}")
            lines.append(f"    Scout:      {r[10]} vs {r[11]} favored={r[12]} correct={r[22]}")
            lines.append(f"    Model WP:   {r[13]:.1f}% vs {r[14]:.1f}% favored={r[15]} correct={r[23]}")
    else:
        lines.append("  No KC vs CIN pairs found for Jun 4")

    # f) Data sufficiency
    owol = con.execute(
        "SELECT COUNT(*) FROM signal_conflict_outcomes WHERE outcome_type='one_win_one_loss'"
    ).fetchone()[0]
    lines.append(f"\n=== (f) Data sufficiency ===")
    lines.append(f"  one_win_one_loss pairs: {owol}")
    lines.append(f"  Threshold for reliability: 50")
    lines.append(f"  Status: {'SUFFICIENT' if owol >= 50 else 'INSUFFICIENT — need more data'}")

    # Per-signal coverage
    for signal in ["cover_rate", "pi", "scout", "model_wp"]:
        col = f"{signal}_correct"
        cnt = con.execute(f"""
            SELECT COUNT(*) FROM signal_conflict_outcomes
            WHERE outcome_type='one_win_one_loss' AND {col} IS NOT NULL
        """).fetchone()[0]
        lines.append(f"    {signal}: {cnt} pairs with signal data")

    return "\n".join(lines)


if __name__ == "__main__":
    con = sqlite3.connect(DB_PATH)
    print("Creating table...")
    create_table(con)
    print("Running backfill...")
    stats = backfill(con)
    print(f"Backfill complete: {stats}")
    print()
    print(run_assessment(con))
    con.close()
