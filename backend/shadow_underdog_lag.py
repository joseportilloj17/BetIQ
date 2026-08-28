"""
Shadow Underdog Lag — Forward Paper Test for Cross-Book Lag Edge
================================================================
Detects when a soft book's UNDERDOG line (dog +1.5, dog -1.5) or FAV +1.5
is STALE vs the fast-moving book consensus. When detected, logs a $0 paper
bet at the stale soft price.

Validation: does the stale line beat the CLOSING LINE (CLV-positive)?
The close is the sharp benchmark we trust most — no Kalshi/Pinnacle needed.

Architecture:
  - Runs AFTER alt_lines_afternoon (2 PM CT) — needs AM + PM snapshots
    to detect intra-day staleness.
  - Compares each soft book against consensus (median across all books
    for that game/spread) at the LATEST snapshot.
  - If a soft book's price hasn't moved from the AM snapshot while
    consensus has shifted >= 1pp, that line is stale.
  - Logs paper bets as source='shadow_underdog_lag' in mock_bets.
  - Settlement + CLV capture handled by existing ESPN pipeline +
    alt_lines_clv (5 PM CT near-close snapshot).
  - Independent source: NOT deduped against other arms.

Cells tracked:
  - dog_neg15: underdog covers -1.5 (PRIMARY test, highest volume, 1.0pp threshold)
  - dog_pos15: underdog covers +1.5 (secondary, tighter market, 0.75pp threshold)
  - fav_pos15: favorite covers +1.5 (NEGATIVE CONTROL — 2x2 found no edge here;
    if CLV-positive alongside dog cells → detection is capturing noise, not lag)

Per-cell thresholds: +1.5 side has structurally tighter pricing (base rate ~65%,
median lag 0.46pp vs 0.62pp for -1.5). Using 0.75pp for +1.5 cells captures real
staleness; 1.0pp for -1.5 cells filters noise on the wider-spread side.
"""

import os
import sys
import uuid
import sqlite3
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

_CT = ZoneInfo("America/Chicago")
_HIST_DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
_BETS_DB = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")

SOFT_BOOKS = ["draftkings", "fanduel", "betmgm"]
ALL_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm"]

# Per-cell stale thresholds (in probability points).
# The +1.5 side has structurally tighter pricing (higher base rate ~65%)
# with median lags ~0.46pp vs ~0.62pp for -1.5. Using 0.75pp for +1.5
# cells captures real staleness while keeping the -1.5 cells at 1.0pp.
CELL_THRESHOLD_PP = {
    "dog_pos15": 0.75,   # tighter market, lower threshold for volume
    "dog_neg15": 1.0,    # primary test cell, standard threshold
    "fav_pos15": 0.75,   # negative control (no expected edge), match dog_pos15 threshold
    "fav_neg15": 1.0,    # not tracked (excluded from cells below)
}
# Default for any cell not in the dict
DEFAULT_THRESHOLD_PP = 1.0


def _implied(odds):
    """Decimal odds -> implied probability."""
    return 1.0 / odds if odds > 1.0 else 1.0


def _devig(odds_a, odds_b):
    """Multiplicative de-vig. Returns (fair_prob_a, fair_prob_b)."""
    ia, ib = _implied(odds_a), _implied(odds_b)
    t = ia + ib
    return (ia / t, ib / t) if t > 0 else (0.5, 0.5)


def _determine_favorite(game_odds):
    """Determine favorite from consensus -1.5 odds.

    game_odds: dict of {bookmaker: {(team, line): odds}}
    Returns (fav_team, dog_team) or (None, None).
    """
    # Collect -1.5 implied probs per team across all books
    team_probs = defaultdict(list)
    for book, book_odds in game_odds.items():
        for (team, line), odds in book_odds.items():
            if line == -1.5 and odds > 1.0:
                team_probs[team].append(_implied(odds))

    if len(team_probs) < 2:
        return None, None

    # Favorite = higher avg implied prob on -1.5
    teams = list(team_probs.keys())
    avg_probs = {t: sum(ps) / len(ps) for t, ps in team_probs.items()}

    sorted_teams = sorted(teams, key=lambda t: avg_probs.get(t, 0), reverse=True)
    return sorted_teams[0], sorted_teams[1]


def generate_underdog_lag_shadow(db=None, today_str=None):
    """Detect stale underdog lines and log paper bets.

    Compares AM snapshot vs latest (PM) snapshot for today's games.
    When a soft book's underdog line hasn't moved but consensus has,
    logs a paper bet at the stale price.

    Returns list of detected stale-line opportunities.
    """
    if today_str is None:
        today_str = datetime.now(_CT).strftime("%Y-%m-%d")

    print(f"\n[underdog-lag] Generating shadow for {today_str}")

    hist_con = sqlite3.connect(_HIST_DB)
    hist_con.row_factory = sqlite3.Row

    # ── Load today's alt_lines snapshots ──
    # Get all snapshots for today's games (by game_date_ct)
    rows = hist_con.execute("""
        SELECT fetched_at, event_id, bookmaker, team, odds, line,
               home_team, away_team, game_date_ct
        FROM alt_lines
        WHERE sport_key = 'baseball_mlb'
          AND market_key = 'alternate_spreads'
          AND line IN (-1.5, 1.5)
          AND bookmaker IN ('pinnacle', 'draftkings', 'fanduel', 'betmgm')
          AND game_date_ct = ?
        ORDER BY event_id, fetched_at
    """, (today_str,)).fetchall()

    if not rows:
        print("[underdog-lag] No alt_lines data for today")
        hist_con.close()
        return []

    # Organize: {event_id: {fetched_at: {bookmaker: {(team, line): odds}}}}
    events = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    event_meta = {}

    for r in rows:
        events[r["event_id"]][r["fetched_at"]][r["bookmaker"]][(r["team"], r["line"])] = r["odds"]
        event_meta[r["event_id"]] = {
            "home": r["home_team"],
            "away": r["away_team"],
            "game_date": r["game_date_ct"],
        }

    print(f"[underdog-lag] {len(events)} games with alt_lines data today")

    # Need at least 2 snapshots to detect staleness
    multi_snap = {eid: snaps for eid, snaps in events.items()
                  if len(snaps) >= 2}
    print(f"[underdog-lag] {len(multi_snap)} games with 2+ snapshots")

    if not multi_snap:
        print("[underdog-lag] Need 2+ snapshots per game — run after PM alt_lines fetch")
        hist_con.close()
        return []

    # ── Also need fixture_id mapping for settlement ──
    # The event_id IS the fixture_id (OddsAPI event ID)

    # ── Detect stale lines ──
    opportunities = []

    for event_id, snapshots in multi_snap.items():
        times = sorted(snapshots.keys())
        meta = event_meta[event_id]
        home, away = meta["home"], meta["away"]

        # AM = first snapshot, latest = most recent
        am_time = times[0]
        latest_time = times[-1]
        am_snap = snapshots[am_time]
        latest_snap = snapshots[latest_time]

        # Determine favorite from latest consensus
        fav, dog = _determine_favorite(latest_snap)
        if not fav or not dog:
            continue

        # Build consensus (median) de-vigged probs at AM and latest
        # For each complementary pair: team -1.5 vs opponent +1.5
        for spread_team in [home, away]:
            other_team = away if spread_team == home else home

            # Is this team the fav or dog?
            if spread_team == fav:
                team_role = "fav"
            else:
                team_role = "dog"

            # De-vig pair: spread_team -1.5 vs other_team +1.5
            # Compute consensus fair prob at AM and latest

            def _consensus_fair(snap):
                """Median de-vigged fair prob for spread_team -1.5 across books."""
                fairs = []
                for book in ALL_BOOKS:
                    if book not in snap:
                        continue
                    neg_odds = snap[book].get((spread_team, -1.5))
                    pos_odds = snap[book].get((other_team, 1.5))
                    if neg_odds and pos_odds and neg_odds > 1 and pos_odds > 1:
                        fair_neg, _ = _devig(neg_odds, pos_odds)
                        fairs.append(fair_neg)
                if fairs:
                    fairs.sort()
                    return fairs[len(fairs) // 2]  # median
                return None

            consensus_am = _consensus_fair(am_snap)
            consensus_latest = _consensus_fair(latest_snap)
            if consensus_am is None or consensus_latest is None:
                continue

            consensus_shift_pp = (consensus_latest - consensus_am) * 100

            # Now check each soft book for staleness
            for book in SOFT_BOOKS:
                if book not in am_snap or book not in latest_snap:
                    continue

                # Soft book's -1.5 odds at AM and latest
                soft_neg_am = am_snap[book].get((spread_team, -1.5))
                soft_pos_am = am_snap[book].get((other_team, 1.5))
                soft_neg_latest = latest_snap[book].get((spread_team, -1.5))
                soft_pos_latest = latest_snap[book].get((other_team, 1.5))

                if not all([soft_neg_am, soft_pos_am, soft_neg_latest, soft_pos_latest]):
                    continue
                if any(o <= 1 for o in [soft_neg_am, soft_pos_am, soft_neg_latest, soft_pos_latest]):
                    continue

                # Soft book's de-vigged fair prob at AM and latest
                soft_fair_neg_am, soft_fair_pos_am = _devig(soft_neg_am, soft_pos_am)
                soft_fair_neg_latest, soft_fair_pos_latest = _devig(soft_neg_latest, soft_pos_latest)

                soft_shift_neg_pp = (soft_fair_neg_latest - soft_fair_neg_am) * 100

                # Staleness = consensus moved but soft book didn't
                # Method: soft's current price disagrees with consensus by > threshold
                # AND consensus moved from AM (the shift is real, not just noise)

                # Check the -1.5 side: spread_team -1.5
                lag_neg_pp = (soft_fair_neg_latest - consensus_latest) * 100

                # Check the +1.5 side: other_team +1.5
                lag_pos_pp = (soft_fair_pos_latest - (1 - consensus_latest)) * 100

                # The cells we care about (per the 2x2 finding + user addendum):
                #   dog_pos15: dog covers +1.5 — if spread_team is fav, other is dog
                #   dog_neg15: dog covers -1.5 — if spread_team is dog
                #   fav_pos15: fav covers +1.5 — if spread_team is dog, other is fav

                # For each eligible cell, check if soft is stale
                # "Stale" = soft overprices vs consensus (soft fair > consensus fair)
                # by > threshold. This means soft gives WORSE odds for this side,
                # which means the OPPOSITE side is being offered at better-than-fair.

                # Actually, let's think about this from the bettor's perspective:
                # If soft OVERPRICES the -1.5 side (soft_fair_neg > consensus_neg),
                # that means soft's -1.5 odds are too LOW (too much implied prob).
                # The +1.5 side is then UNDERPRICED → bet +1.5 at soft.
                #
                # If soft UNDERPRICES the -1.5 side (soft_fair_neg < consensus_neg),
                # soft gives higher -1.5 odds → bet -1.5 at soft.

                # Compute vig for this pair
                vig_pct = (_implied(soft_neg_latest) + _implied(soft_pos_latest) - 1) * 100

                # Check: soft overprices -1.5 → bet +1.5 at soft
                other_role = "fav" if other_team == fav else "dog"
                cell_pos = f"{other_role}_pos15"
                thresh_pos = CELL_THRESHOLD_PP.get(cell_pos, DEFAULT_THRESHOLD_PP)

                if lag_neg_pp > thresh_pos:
                    # The +1.5 side (other_team) is stale-cheap at soft
                    cell = cell_pos

                    # Only track cells we care about: dog_pos15, fav_pos15
                    if cell in ("dog_pos15", "fav_pos15"):
                        opportunities.append({
                            "event_id": event_id,
                            "home": home, "away": away,
                            "game_date": today_str,
                            "book": book,
                            "cell": cell,
                            "bet_team": other_team,
                            "bet_side": "+1.5",
                            "bet_odds": soft_pos_latest,  # realistic actionable price
                            "soft_fair": soft_fair_pos_latest,
                            "consensus_fair": 1 - consensus_latest,
                            "lag_pp": lag_neg_pp,  # how stale
                            "consensus_shift_pp": consensus_shift_pp,
                            "vig_pct": vig_pct,
                            "am_time": am_time,
                            "detect_time": latest_time,
                            "fav": fav, "dog": dog,
                        })

                # Check: soft underprices -1.5 → bet -1.5 at soft
                cell_neg = f"{team_role}_neg15"
                thresh_neg = CELL_THRESHOLD_PP.get(cell_neg, DEFAULT_THRESHOLD_PP)

                if lag_neg_pp < -thresh_neg:
                    # The -1.5 side (spread_team) is stale-cheap at soft
                    cell = cell_neg

                    # Only track dog_neg15
                    if cell == "dog_neg15":
                        opportunities.append({
                            "event_id": event_id,
                            "home": home, "away": away,
                            "game_date": today_str,
                            "book": book,
                            "cell": cell,
                            "bet_team": spread_team,
                            "bet_side": "-1.5",
                            "bet_odds": soft_neg_latest,  # realistic actionable price
                            "soft_fair": soft_fair_neg_latest,
                            "consensus_fair": consensus_latest,
                            "lag_pp": abs(lag_neg_pp),
                            "consensus_shift_pp": consensus_shift_pp,
                            "vig_pct": vig_pct,
                            "am_time": am_time,
                            "detect_time": latest_time,
                            "fav": fav, "dog": dog,
                        })

    hist_con.close()

    # Dedup: one opportunity per (event, book, cell)
    seen = set()
    deduped = []
    for opp in opportunities:
        key = (opp["event_id"], opp["book"], opp["cell"])
        if key not in seen:
            seen.add(key)
            deduped.append(opp)
    opportunities = deduped

    print(f"[underdog-lag] Detected {len(opportunities)} stale-line opportunities")

    # Breakdown by cell
    cell_counts = defaultdict(int)
    for opp in opportunities:
        cell_counts[opp["cell"]] += 1
    for cell, count in sorted(cell_counts.items()):
        print(f"  {cell}: {count}")

    # Store as mock_bets
    if db is not None and opportunities:
        stored = _store_lag_bets(db, opportunities, today_str)
        print(f"[underdog-lag] Stored {stored} paper bets (source=shadow_underdog_lag)")

    return opportunities


def _store_lag_bets(db, opportunities, today_str):
    """Persist stale-line opportunities as 1-leg mock_bets."""
    from database import MockBet, MockBetLeg
    from sqlalchemy import text as _text

    run_id = str(uuid.uuid4())
    stored = 0

    for opp in opportunities:
        mock_id = str(uuid.uuid4())

        # Build description matching the fixture format
        bet_team = opp["bet_team"]
        bet_side = opp["bet_side"]
        sign = "+" if float(bet_side) > 0 else ""
        desc = f"{bet_team} {sign}{bet_side} (Alt Spread)"

        bet_info = (f"{desc} | lag={opp['lag_pp']:.1f}pp "
                    f"book={opp['book']} cell={opp['cell']}")

        # Dedup: skip if same leg already stored today
        existing = db.execute(_text(
            "SELECT COUNT(*) FROM mock_bets WHERE game_date=:gd "
            "AND source='shadow_underdog_lag' AND bet_info=:bi"
        ), {"gd": today_str, "bi": bet_info}).scalar()
        if existing > 0:
            continue

        # The "win prob" here is the consensus fair probability
        # (not a model prediction — this arm tests market structure, not model)
        consensus_wp = opp["consensus_fair"] * 100

        mock = MockBet(
            id=mock_id,
            generation_run_id=run_id,
            game_date=today_str,
            sport="MLB",
            bet_type="straight",
            odds=opp["bet_odds"],
            amount=0.0,  # paper only
            legs=1,
            bet_info=bet_info,
            predicted_win_prob=consensus_wp,
            model_used="crossbook_lag_v1",
            status="PENDING",
            source="shadow_underdog_lag",
            cohort="clean",
            notes=(f"cell={opp['cell']} lag={opp['lag_pp']:.1f}pp "
                   f"shift={opp['consensus_shift_pp']:.1f}pp "
                   f"vig={opp['vig_pct']:.1f}% "
                   f"fav={opp['fav']} dog={opp['dog']}"),
        )
        db.add(mock)

        # Determine fixture_id and market_type
        mock_leg = MockBetLeg(
            mock_bet_id=mock_id,
            leg_index=0,
            description=desc,
            market_type="alternate_spreads",
            sport="MLB",
            win_prob=consensus_wp,
            ev=0.0,  # not model-driven
            fixture_id=opp["event_id"],
            is_alt_line=True,
            open_odds=_dec_to_american(opp["bet_odds"]),
        )
        db.add(mock_leg)
        stored += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[underdog-lag] Commit error: {exc}")
        return 0

    return stored


def _dec_to_american(dec_odds):
    """Convert decimal odds to American format."""
    if dec_odds >= 2.0:
        return int(round((dec_odds - 1) * 100))
    elif dec_odds > 1.0:
        return int(round(-100 / (dec_odds - 1)))
    return 100


if __name__ == "__main__":
    from database import SessionLocal, init_db
    init_db()
    db_session = SessionLocal()
    try:
        today = None
        if len(sys.argv) > 1:
            today = sys.argv[1]  # allow override: python shadow_underdog_lag.py 2026-08-10
        opps = generate_underdog_lag_shadow(db_session, today)
        if opps:
            print(f"\n[underdog-lag] Summary: {len(opps)} opportunities")
            for o in opps[:10]:
                print(f"  {o['cell']:12s} {o['book']:12s} {o['bet_team']:25s} "
                      f"{o['bet_side']:5s} odds={o['bet_odds']:.3f} "
                      f"lag={o['lag_pp']:.1f}pp")
    finally:
        db_session.close()
