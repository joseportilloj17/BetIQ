#!/usr/bin/env python3
"""
Full-ladder counterfactual analysis on settled MLB spread bets.

For every settled MLB game with a known margin, compute whether
+1.5, +2.5, +3.5, +4.5, +5.5 would have covered, even for rungs
we didn't actually bet at. This amplifies sample from ~130 per rung
to ~500+ per rung.
"""

import sqlite3
import re
from collections import defaultdict

RUNGS = [1.5, 2.5, 3.5, 4.5, 5.5]

# ── 1. Load settled MLB spread legs ──────────────────────────────────────────
bets_conn = sqlite3.connect("data/bets.db")
bets_conn.row_factory = sqlite3.Row
bc = bets_conn.cursor()

bc.execute("""
    SELECT l.id, l.fixture_id, l.description, l.market_type, l.leg_result,
           l.resolved_home_team, l.resolved_away_team,
           l.resolved_home_score, l.resolved_away_score,
           l.spread_value, l.home_starter_era, l.away_starter_era,
           b.game_date
    FROM mock_bet_legs l
    JOIN mock_bets b ON l.mock_bet_id = b.id
    WHERE l.sport IN ('MLB','baseball_mlb')
      AND l.market_type IN ('spreads','alternate_spreads')
      AND l.leg_result IS NOT NULL
      AND COALESCE(l.excluded_from_metrics,0) = 0
      AND l.resolved_home_score IS NOT NULL
""")
legs = bc.fetchall()
print(f"Loaded {len(legs)} settled MLB spread legs")

# ── 2. Deduplicate to unique games (by fixture_id) ──────────────────────────
# For each fixture we need: home_team, away_team, home_score, away_score,
# game_date, pitcher ERAs. Take the first leg's data per fixture.
games_map = {}  # fixture_id -> dict
for leg in legs:
    fid = leg["fixture_id"]
    if fid not in games_map:
        games_map[fid] = {
            "fixture_id": fid,
            "home_team": leg["resolved_home_team"],
            "away_team": leg["resolved_away_team"],
            "home_score": leg["resolved_home_score"],
            "away_score": leg["resolved_away_score"],
            "game_date": leg["game_date"],
            "home_starter_era": leg["home_starter_era"],
            "away_starter_era": leg["away_starter_era"],
        }
    else:
        # Fill in pitcher ERA if missing from earlier leg
        if games_map[fid]["home_starter_era"] is None and leg["home_starter_era"] is not None:
            games_map[fid]["home_starter_era"] = leg["home_starter_era"]
        if games_map[fid]["away_starter_era"] is None and leg["away_starter_era"] is not None:
            games_map[fid]["away_starter_era"] = leg["away_starter_era"]

print(f"Unique games (fixtures): {len(games_map)}")

# Also collect which teams were actually picked per fixture from the legs
# We'll parse the description to get pick_team for each leg
def parse_pick_team(desc):
    """Parse team name from description like 'Miami Marlins +1.5 (Alt Spread)'"""
    m = re.match(r'^(.+?)\s+[+-][\d.]+', desc)
    if m:
        return m.group(1).strip()
    # Try without +/- for h2h style
    m = re.match(r'^(.+?)\s+\(', desc)
    if m:
        return m.group(1).strip()
    return desc.strip()

# Collect unique pick teams per fixture
fixture_pick_teams = defaultdict(set)
for leg in legs:
    team = parse_pick_team(leg["description"])
    fixture_pick_teams[leg["fixture_id"]].add(team)

# ── 3. Load historical games for form calculation ────────────────────────────
hist_conn = sqlite3.connect("data/historical.db")
hist_conn.row_factory = sqlite3.Row
hc = hist_conn.cursor()

# Load all MLB games sorted by date for efficient form lookups
hc.execute("""
    SELECT game_id, game_date, home_team, away_team, home_score, away_score
    FROM games
    WHERE sport='MLB' AND home_score IS NOT NULL
    ORDER BY game_date
""")
all_mlb_games = hc.fetchall()
print(f"Loaded {len(all_mlb_games)} historical MLB games for form calc")

# Build team game history index: team -> [(game_date, won_bool)]
team_history = defaultdict(list)
for g in all_mlb_games:
    hteam = g["home_team"]
    ateam = g["away_team"]
    gd = g["game_date"]
    hs, aws = g["home_score"], g["away_score"]
    home_won = hs > aws
    team_history[hteam].append((gd, home_won))
    team_history[ateam].append((gd, not home_won))

def get_last5_wr(team, before_date):
    """Get win rate of last 5 games before a given date."""
    hist = team_history.get(team, [])
    # Filter games before this date
    prior = [h for h in hist if h[0] < before_date]
    if len(prior) == 0:
        return None
    last5 = prior[-5:]
    wins = sum(1 for _, w in last5 if w)
    return wins / len(last5)

def form_bucket(wr):
    if wr is None:
        return "unknown"
    if wr < 0.35:
        return "cold"
    elif wr <= 0.65:
        return "neutral"
    else:
        return "hot"

def era_bucket(era):
    if era is None:
        return "unknown"
    if era < 3.0:
        return "elite"
    elif era <= 4.5:
        return "average"
    else:
        return "poor"

# ── 4. For each unique game, compute counterfactual covers for BOTH teams ────
# Each game yields TWO rows: one for the home team perspective, one for away.
# For each team, margin = team_score - opponent_score.
# Then for each rung, would_cover = 1 if (margin + rung) > 0.

results = []  # list of dicts

for fid, gdata in games_map.items():
    hs = gdata["home_score"]
    aws = gdata["away_score"]
    gd = gdata["game_date"]
    home_team = gdata["home_team"]
    away_team = gdata["away_team"]

    if hs is None or aws is None:
        continue

    # Home team perspective
    home_margin = hs - aws  # positive = home won
    away_margin = aws - hs

    # Form for each team
    home_wr = get_last5_wr(home_team, gd) if gd else None
    away_wr = get_last5_wr(away_team, gd) if gd else None

    # Pitcher ERA: pick team's starter ERA
    home_era = gdata["home_starter_era"]
    away_era = gdata["away_starter_era"]

    # For counterfactual, we simulate BOTH teams getting the spread
    for team, margin, team_wr, opp_starter_era in [
        (home_team, home_margin, home_wr, away_era),   # home team getting + spread, opposing starter matters
        (away_team, away_margin, away_wr, home_era),   # away team getting + spread
    ]:
        fb = form_bucket(team_wr)
        # The pick team is getting the + spread, so the relevant pitcher is their own starter
        # Actually, for a pick team getting +1.5, it's about whether they cover.
        # The pick team's own starter ERA matters for their defense.
        # Let's use the pick team's own starter ERA for the pitcher bucket.
        if team == home_team:
            pick_era = home_era
        else:
            pick_era = away_era
        pb = era_bucket(pick_era)

        for rung in RUNGS:
            # Would the pick team cover +rung?
            # Cover if team_score + rung > opponent_score
            # i.e., margin + rung > 0
            would_cover = 1 if (margin + rung) > 0 else 0
            # Push case: margin + rung == 0 -> no cover (push = loss for our purposes)

            results.append({
                "fixture_id": fid,
                "team": team,
                "margin": margin,
                "rung": rung,
                "would_cover": would_cover,
                "form_bucket": fb,
                "pitcher_bucket": pb,
                "team_wr": team_wr,
                "pitcher_era": pick_era,
            })

print(f"Total counterfactual rows: {len(results)}")
print()

# ── 5. Aggregate and print tables ───────────────────────────────────────────

# Helper
def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for r in rows:
                w = max(w, len(str(r[i])))
            col_widths.append(w + 2)

    header_str = "".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_str)
    print("-" * len(header_str))
    for r in rows:
        print("".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))))

# ── Table A: Overall cover rate by rung ──────────────────────────────────────
print("=" * 70)
print("TABLE A: Overall Cover Rate by Rung (Counterfactual, Both Teams)")
print("=" * 70)
rung_stats = defaultdict(lambda: {"n": 0, "covers": 0})
for r in results:
    rung_stats[r["rung"]]["n"] += 1
    rung_stats[r["rung"]]["covers"] += r["would_cover"]

rows_a = []
for rung in RUNGS:
    s = rung_stats[rung]
    rate = s["covers"] / s["n"] if s["n"] > 0 else 0
    rows_a.append((f"+{rung}", s["n"], f"{rate:.3f}", s["covers"]))

print_table(["Rung", "n_games", "cover_rate", "would_cover_count"], rows_a)
print()

# ── Table B: Cover rate by rung x form_bucket ────────────────────────────────
print("=" * 70)
print("TABLE B: Cover Rate by Rung x Form Bucket")
print("=" * 70)
form_rung_stats = defaultdict(lambda: {"n": 0, "covers": 0})
for r in results:
    key = (r["form_bucket"], r["rung"])
    form_rung_stats[key]["n"] += 1
    form_rung_stats[key]["covers"] += r["would_cover"]

rows_b = []
for fb in ["cold", "neutral", "hot", "unknown"]:
    for rung in RUNGS:
        key = (fb, rung)
        s = form_rung_stats[key]
        if s["n"] == 0:
            continue
        rate = s["covers"] / s["n"] if s["n"] > 0 else 0
        rows_b.append((fb, f"+{rung}", s["n"], f"{rate:.3f}"))

print_table(["form_bucket", "rung", "n", "cover_rate"], rows_b)
print()

# ── Table C: Cover rate by rung x pitcher_era_bucket ─────────────────────────
print("=" * 70)
print("TABLE C: Cover Rate by Rung x Pitcher ERA Bucket")
print("=" * 70)
pitch_rung_stats = defaultdict(lambda: {"n": 0, "covers": 0})
for r in results:
    key = (r["pitcher_bucket"], r["rung"])
    pitch_rung_stats[key]["n"] += 1
    pitch_rung_stats[key]["covers"] += r["would_cover"]

rows_c = []
for pb in ["elite", "average", "poor", "unknown"]:
    for rung in RUNGS:
        key = (pb, rung)
        s = pitch_rung_stats[key]
        if s["n"] == 0:
            continue
        rate = s["covers"] / s["n"] if s["n"] > 0 else 0
        rows_c.append((pb, f"+{rung}", s["n"], f"{rate:.3f}"))

print_table(["pitcher_bucket", "rung", "n", "cover_rate"], rows_c)
print()

# ── Table D: Cover rate by rung x form x pitcher (n>=20 only) ────────────────
print("=" * 70)
print("TABLE D: Cover Rate by Form x Pitcher x Rung (n >= 20 only)")
print("=" * 70)
combo_stats = defaultdict(lambda: {"n": 0, "covers": 0})
for r in results:
    key = (r["form_bucket"], r["pitcher_bucket"], r["rung"])
    combo_stats[key]["n"] += 1
    combo_stats[key]["covers"] += r["would_cover"]

rows_d = []
for fb in ["cold", "neutral", "hot", "unknown"]:
    for pb in ["elite", "average", "poor", "unknown"]:
        for rung in RUNGS:
            key = (fb, pb, rung)
            s = combo_stats[key]
            if s["n"] < 20:
                continue
            rate = s["covers"] / s["n"] if s["n"] > 0 else 0
            rows_d.append((fb, pb, f"+{rung}", s["n"], f"{rate:.3f}"))

if rows_d:
    print_table(["form", "pitcher", "rung", "n", "cover_rate"], rows_d)
else:
    print("(No cells with n >= 20)")
print()

# ── Table E: Isotonic monotonicity check ─────────────────────────────────────
print("=" * 70)
print("TABLE E: Isotonic Monotonicity Check")
print("=" * 70)
print("Verifying cover rate is monotonically increasing in rung for each slice.")
print()

def check_monotonicity(label, stats_dict, key_fn):
    """Check if cover rate increases with rung for a given condition."""
    violations = []
    # Get unique condition keys (excluding rung)
    conditions = set()
    for k in stats_dict:
        cond = key_fn(k)
        conditions.add(cond)

    for cond in sorted(conditions):
        rates = []
        for rung in RUNGS:
            full_key = cond + (rung,) if isinstance(cond, tuple) else (cond, rung)
            s = stats_dict[full_key]
            if s["n"] > 0:
                rates.append((rung, s["covers"] / s["n"], s["n"]))
            else:
                rates.append((rung, None, 0))

        # Check monotonicity
        prev_rate = -1
        is_mono = True
        for rung, rate, n in rates:
            if rate is not None:
                if rate < prev_rate - 0.001:  # small tolerance for floating point
                    is_mono = False
                    violations.append((cond, rung, prev_rate, rate))
                prev_rate = rate

        status = "PASS" if is_mono else "FAIL"
        rates_str = " -> ".join(f"+{r}: {rt:.3f}" for r, rt, _ in rates if rt is not None)
        cond_str = str(cond) if isinstance(cond, str) else " / ".join(str(c) for c in cond)
        print(f"  {status}  {label} = {cond_str}: {rates_str}")

    return violations

# Overall
print("Overall:")
all_violations = []
overall_rates = []
for rung in RUNGS:
    s = rung_stats[rung]
    rate = s["covers"] / s["n"] if s["n"] > 0 else 0
    overall_rates.append((rung, rate))

prev = -1
is_mono = True
for rung, rate in overall_rates:
    if rate < prev - 0.001:
        is_mono = False
    prev = rate
status = "PASS" if is_mono else "FAIL"
rates_str = " -> ".join(f"+{r}: {rt:.3f}" for r, rt in overall_rates)
print(f"  {status}  Overall: {rates_str}")
print()

# By form
print("By Form Bucket:")
v = check_monotonicity("form", form_rung_stats, lambda k: k[0])
all_violations.extend(v)
print()

# By pitcher
print("By Pitcher ERA Bucket:")
v = check_monotonicity("pitcher", pitch_rung_stats, lambda k: k[0])
all_violations.extend(v)
print()

# By form x pitcher
print("By Form x Pitcher:")
v = check_monotonicity("form x pitcher", combo_stats, lambda k: (k[0], k[1]))
all_violations.extend(v)
print()

if all_violations:
    print(f"WARNING: {len(all_violations)} monotonicity violations found!")
    for v in all_violations:
        print(f"  Condition: {v[0]}, Rung: +{v[1]}, prev_rate={v[2]:.3f}, curr_rate={v[3]:.3f}")
else:
    print("All slices pass monotonicity check (as expected by construction).")

print()

# ── Bonus: Summary stats ────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY")
print("=" * 70)
margins = [r["margin"] for r in results if r["rung"] == 1.5]  # one per team-game
print(f"Total team-game observations: {len(margins)}")
print(f"Unique fixtures: {len(games_map)}")
print(f"Mean margin (from pick team perspective): {sum(margins)/len(margins):.2f}")
print(f"Median margin: {sorted(margins)[len(margins)//2]:.1f}")
blowouts = sum(1 for m in margins if abs(m) >= 6)
print(f"Blowouts (|margin| >= 6): {blowouts} ({100*blowouts/len(margins):.1f}%)")

bets_conn.close()
hist_conn.close()
