#!/usr/bin/env python3
"""S5-2 day-1 smoke test — run the morning after the first nba_forward_preds capture.

For one finished game, verifies the full chain:
  1. scout rows in nba_forward_preds (snapshot_type='scout')
  2. pre_tip rows in nba_forward_preds (snapshot_type='pre_tip')
  3. matching player_props rows for same game_date + players
  4. graded actual_hit (after grade_nba_forward_preds runs)
  5. model-vs-devigged-market join (sim_p_over vs devigged implied prob)

Usage:
    conda run -n betiq python s5_2_smoke_test.py [YYYY-MM-DD]
    (defaults to yesterday ET)
"""
from __future__ import annotations
import os, sys, sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_HIST = os.path.join(_DATA, "historical.db")
_BETS = os.path.join(_DATA, "bets.db")

# Resolve target date
if len(sys.argv) > 1:
    target_date = sys.argv[1]
else:
    target_date = (datetime.now(_ET) - timedelta(days=1)).strftime("%Y-%m-%d")

print(f"=== S5-2 Smoke Test for {target_date} ===\n")

hcon = sqlite3.connect(_HIST)
hcon.row_factory = sqlite3.Row
bcon = sqlite3.connect(_BETS)
bcon.row_factory = sqlite3.Row

# ── 0. Table existence ────────────────────────────────────────────────────────
tbl = hcon.execute(
    "SELECT name FROM sqlite_master WHERE name='nba_forward_preds'"
).fetchone()
if not tbl:
    print("FAIL: nba_forward_preds table does not exist in historical.db")
    sys.exit(1)
print("OK: nba_forward_preds table exists\n")

# ── 1. Scout rows ────────────────────────────────────────────────────────────
scout = hcon.execute("""
    SELECT COUNT(*) cnt, COUNT(DISTINCT player_name) players,
           COUNT(DISTINCT stat_type) stats,
           COUNT(DISTINCT game_id) games
    FROM nba_forward_preds
    WHERE game_date = ? AND snapshot_type = 'scout'
""", (target_date,)).fetchone()
print(f"1. Scout rows:  {scout['cnt']} rows, {scout['players']} players, "
      f"{scout['stats']} stat types, {scout['games']} games")
if scout['cnt'] == 0:
    print("   FAIL: No scout rows — did run_daily_scout() fire?")

# ── 2. Pre-tip rows ──────────────────────────────────────────────────────────
pretip = hcon.execute("""
    SELECT COUNT(*) cnt, COUNT(DISTINCT player_name) players,
           COUNT(DISTINCT game_id) games
    FROM nba_forward_preds
    WHERE game_date = ? AND snapshot_type = 'pre_tip'
""", (target_date,)).fetchone()
print(f"2. Pre-tip rows: {pretip['cnt']} rows, {pretip['players']} players, "
      f"{pretip['games']} games")
if pretip['cnt'] == 0:
    print("   WARNING: No pre-tip rows — games may not have had 30-60 min window coverage")

# ── 3. Matching player_props ─────────────────────────────────────────────────
# Pick one game for detailed inspection
sample_game = hcon.execute("""
    SELECT game_id, home_team, away_team
    FROM nba_forward_preds
    WHERE game_date = ? AND snapshot_type = 'scout'
    LIMIT 1
""", (target_date,)).fetchone()

if sample_game:
    gid = sample_game['game_id']
    print(f"\n--- Sample game: {sample_game['away_team']} @ {sample_game['home_team']} "
          f"(game_id={gid}) ---\n")

    # Get forward preds for this game
    preds = hcon.execute("""
        SELECT player_name, stat_type, line, side,
               sim_p_over, sim_p_under, sim_median,
               blend_hit_prob, snapshot_type,
               actual_value, actual_hit, hours_to_tip
        FROM nba_forward_preds
        WHERE game_date = ? AND game_id = ?
        ORDER BY snapshot_type, player_name, stat_type
    """, (target_date, gid)).fetchall()

    scout_rows = [r for r in preds if r['snapshot_type'] == 'scout']
    pretip_rows = [r for r in preds if r['snapshot_type'] == 'pre_tip']
    print(f"   Scout preds: {len(scout_rows)}, Pre-tip preds: {len(pretip_rows)}")

    # Show first 5 scout rows
    print(f"\n   Scout sample (first 5):")
    print(f"   {'Player':25s} {'Stat':18s} {'Line':>5s} {'Side':5s} "
          f"{'simP':>6s} {'median':>6s} {'actual':>6s} {'hit':>3s}")
    for r in scout_rows[:5]:
        actual_str = f"{r['actual_value']:.1f}" if r['actual_value'] is not None else "---"
        hit_str = str(r['actual_hit']) if r['actual_hit'] is not None else "---"
        p = r['sim_p_over'] if r['side'] == 'over' else r['sim_p_under']
        print(f"   {r['player_name']:25s} {r['stat_type']:18s} {r['line']:5.1f} "
              f"{r['side']:5s} {p:6.3f} {r['sim_median'] or 0:6.1f} "
              f"{actual_str:>6s} {hit_str:>3s}")

    # ── 3b. Matching player_props from historical.db ──────────────────────────
    pp_count = hcon.execute("""
        SELECT COUNT(*) FROM player_props
        WHERE game_date = ? AND sport_key = 'basketball_nba'
    """, (target_date,)).fetchone()[0]
    print(f"\n3. Player_props for {target_date}: {pp_count} rows")

    # Match one player across both
    if scout_rows:
        sample_player = scout_rows[0]['player_name']
        pp_match = hcon.execute("""
            SELECT market_key, point, COUNT(*) cnt, MIN(fetched_at) first, MAX(fetched_at) last
            FROM player_props
            WHERE game_date = ? AND player_name = ? AND sport_key = 'basketball_nba'
            GROUP BY market_key, point
            ORDER BY market_key
        """, (target_date, sample_player)).fetchall()
        print(f"   Player_props for '{sample_player}': {sum(r['cnt'] for r in pp_match)} rows")
        for r in pp_match:
            print(f"     {r['market_key']:25s} line={r['point']}  "
                  f"({r['cnt']} books, {r['first'][:16]}→{r['last'][:16]})")

    # ── 4. Grading status ─────────────────────────────────────────────────────
    graded = hcon.execute("""
        SELECT COUNT(*) total,
               SUM(CASE WHEN actual_hit IS NOT NULL THEN 1 ELSE 0 END) graded,
               SUM(CASE WHEN actual_hit = 1 THEN 1 ELSE 0 END) hits
        FROM nba_forward_preds WHERE game_date = ?
    """, (target_date,)).fetchone()
    print(f"\n4. Grading: {graded['graded']}/{graded['total']} graded"
          f" ({graded['hits'] or 0} hits)")
    if graded['graded'] == 0 and graded['total'] > 0:
        # Check if game logs exist
        gl = hcon.execute("""
            SELECT COUNT(DISTINCT player_name) FROM nba_player_game_logs
            WHERE game_date = ?
        """, (target_date,)).fetchone()[0]
        if gl == 0:
            print("   INFO: No game_logs yet — nightly_ats hasn't run or game not played")
        else:
            print(f"   FAIL: {gl} players in game_logs but 0 graded — name match issue?")

    # ── 5. Model vs devigged market ───────────────────────────────────────────
    print(f"\n5. Model vs Devigged Market (scout snapshot, sample game):")
    # Get devigged lines from player_props for sample player
    if scout_rows and pp_match:
        sample_stat = scout_rows[0]['stat_type']
        # Map stat_type to player_props market_key
        _stat_to_mkt = {
            "player_points": "player_points", "player_rebounds": "player_rebounds",
            "player_assists": "player_assists", "player_threes": "player_threes",
            "player_steals": "player_steals", "player_blocks": "player_blocks",
            "player_pra": "player_points_rebounds_assists",
        }
        mkt_key = _stat_to_mkt.get(sample_stat, sample_stat)

        market_lines = hcon.execute("""
            SELECT point, price_over, price_under, bookmaker
            FROM player_props
            WHERE game_date = ? AND player_name = ?
              AND market_key = ? AND sport_key = 'basketball_nba'
            LIMIT 10
        """, (target_date, sample_player, mkt_key)).fetchall()

        if market_lines:
            print(f"   Player: {sample_player}, Stat: {sample_stat}")
            print(f"   {'Book':15s} {'Line':>5s} {'mktOver':>8s} {'mktUnder':>8s} "
                  f"{'devig_O':>8s} {'simP_O':>8s} {'diff':>7s}")
            for ml in market_lines[:5]:
                over_odds = ml['price_over'] or 0
                under_odds = ml['price_under'] or 0
                # American odds → implied prob
                def _imp(american):
                    if american is None or american == 0:
                        return 0.5
                    if american > 0:
                        return 100 / (american + 100)
                    return abs(american) / (abs(american) + 100)
                imp_o = _imp(over_odds)
                imp_u = _imp(under_odds)
                # Devig (power method)
                total = imp_o + imp_u
                dv_o = imp_o / total if total > 0 else 0.5

                # Find matching sim prediction
                match = [r for r in scout_rows
                         if r['player_name'] == sample_player
                         and r['stat_type'] == sample_stat
                         and abs((r['line'] or 0) - (ml['point'] or 0)) < 0.01
                         and r['side'] == 'over']
                sim_o = match[0]['sim_p_over'] if match else None
                diff = f"{(sim_o - dv_o):+.3f}" if sim_o is not None else "---"
                sim_str = f"{sim_o:.4f}" if sim_o is not None else "---"

                print(f"   {ml['bookmaker']:15s} {ml['point'] or 0:5.1f} "
                      f"{over_odds:>+8.0f} {under_odds:>+8.0f} "
                      f"{dv_o:8.4f} {sim_str:>8s} {diff:>7s}")
        else:
            print(f"   No matching player_props for {sample_player}/{mkt_key}")
    else:
        print("   SKIP: No scout rows or player_props to compare")

else:
    print("\nFAIL: No forward preds found for this date")
    graded = None

# ── 6. Model ↔ player_props join match rate ──────────────────────────────────
print(f"\n6. Model ↔ Player Props (Odds API) Name Match Rate:")
# Get distinct player names from forward preds (scout snapshot)
fp_players = hcon.execute("""
    SELECT DISTINCT player_name FROM nba_forward_preds
    WHERE game_date = ? AND snapshot_type = 'scout'
""", (target_date,)).fetchall()
fp_names = {r['player_name'] for r in fp_players}

# Get distinct player names from player_props for same date
pp_players = hcon.execute("""
    SELECT DISTINCT player_name FROM player_props
    WHERE game_date = ? AND sport_key = 'basketball_nba'
""", (target_date,)).fetchall()
pp_names = {r['player_name'] for r in pp_players}

if fp_names and pp_names:
    matched = fp_names & pp_names
    fp_only = fp_names - pp_names
    pp_only = pp_names - fp_names
    print(f"   Forward preds players: {len(fp_names)}")
    print(f"   Player props players:  {len(pp_names)}")
    print(f"   Exact match:           {len(matched)} ({100*len(matched)/len(fp_names):.0f}%)")
    if fp_only:
        print(f"   In model but NOT in props ({len(fp_only)}):")
        for n in sorted(fp_only):
            print(f"     - {n}")
    if pp_only and len(pp_only) <= 20:
        print(f"   In props but NOT in model ({len(pp_only)}):")
        for n in sorted(pp_only):
            print(f"     - {n}")
    elif pp_only:
        print(f"   In props but NOT in model: {len(pp_only)} players (too many to list)")
else:
    print("   SKIP: No forward pred or player prop names to compare")

# ── 7. Alias table status ────────────────────────────────────────────────────
print(f"\n7. Alias Table:")
alias_tbl = hcon.execute(
    "SELECT name FROM sqlite_master WHERE name='nba_player_alias'"
).fetchone()
if alias_tbl:
    alias_rows = hcon.execute("SELECT * FROM nba_player_alias").fetchall()
    print(f"   {len(alias_rows)} alias entries")
    for a in alias_rows:
        print(f"     {a['alias_name']:25s} → {a['canonical_name']:25s}  "
              f"(bdl_id={a['bdl_player_id']}, src={a['source']})")
else:
    print("   WARNING: nba_player_alias table does not exist")

# ── 8. bdl_player_id fill rate ───────────────────────────────────────────────
print(f"\n8. bdl_player_id Fill Rate:")
bdl_fill = hcon.execute("""
    SELECT COUNT(*) total,
           SUM(CASE WHEN bdl_player_id IS NOT NULL THEN 1 ELSE 0 END) filled
    FROM nba_forward_preds WHERE game_date = ?
""", (target_date,)).fetchone()
if bdl_fill and bdl_fill['total'] > 0:
    pct = 100 * (bdl_fill['filled'] or 0) / bdl_fill['total']
    print(f"   {bdl_fill['filled']}/{bdl_fill['total']} rows have bdl_player_id ({pct:.0f}%)")
    if pct < 50:
        print("   NOTE: Low fill expected on day 1 (game_logs may not exist yet at scout time)")
else:
    print("   No rows to check")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Chain status for {target_date} ===")
chain = []
chain.append(("scout_write", scout['cnt'] > 0))
chain.append(("pre_tip_write", pretip['cnt'] > 0))
chain.append(("player_props", hcon.execute(
    "SELECT COUNT(*) FROM player_props WHERE game_date=? AND sport_key='basketball_nba'",
    (target_date,)).fetchone()[0] > 0))
chain.append(("game_logs", hcon.execute(
    "SELECT COUNT(*) FROM nba_player_game_logs WHERE game_date=?",
    (target_date,)).fetchone()[0] > 0))
chain.append(("grading", (graded['graded'] or 0) > 0 if graded else False))

for name, ok in chain:
    print(f"  {'OK' if ok else 'MISSING':>7s}  {name}")

hcon.close()
bcon.close()

any_fail = any(not ok for name, ok in chain if name != "pre_tip_write")
sys.exit(1 if any_fail else 0)
