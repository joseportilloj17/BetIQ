#!/usr/bin/env python3
"""
Quick backfill: fetch batter_game_stats + game_lineups for Jun 18 - Aug 13 2026.
Uses MLB Schedule API to find game_pks, then fetches boxscores directly.
Skips games already in batter_game_stats.
"""
import os, sqlite3, time, requests, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backfill")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "historical.db")
START_DATE = "2026-06-18"
END_DATE = "2026-08-13"


def get_game_pks_from_schedule(start, end):
    """Get all MLB game_pks from schedule API for date range."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}&gameType=R"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    pks = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                pks.append(game["gamePk"])
    return sorted(set(pks))


def fetch_boxscore(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
        r = requests.get(url, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def parse_boxscore(game_pk, data):
    lineup_rows, batter_rows = [], []
    teams = data.get("teams", {})
    for side, is_home in [("home", 1), ("away", 0)]:
        team_data = teams.get(side, {})
        team_name = team_data.get("team", {}).get("name", "Unknown")
        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})
        for pid in batting_order:
            pdata = players.get(f"ID{pid}", {})
            person = pdata.get("person", {})
            name = person.get("fullName", "")
            bo = pdata.get("battingOrder", 0)
            pos = pdata.get("position", {}).get("abbreviation", "")
            batting = pdata.get("stats", {}).get("batting", {})
            ab = batting.get("atBats", 0)
            hits = batting.get("hits", 0)
            doubles = batting.get("doubles", 0)
            triples = batting.get("triples", 0)
            hr = batting.get("homeRuns", 0)
            bb = batting.get("baseOnBalls", 0)
            hbp = batting.get("hitByPitch", 0)
            sf = batting.get("sacFlies", 0)
            k = batting.get("strikeOuts", 0)
            singles = hits - doubles - triples - hr
            tb = singles + 2 * doubles + 3 * triples + 4 * hr
            lineup_rows.append(dict(game_pk=game_pk, team_name=team_name, is_home=is_home,
                                    player_id=pid, batting_order=bo, position=pos))
            batter_rows.append(dict(game_pk=game_pk, player_id=pid, player_name=name,
                                    team_name=team_name, is_home=is_home,
                                    ab=ab, hits=hits, doubles=doubles, triples=triples,
                                    hr=hr, bb=bb, hbp=hbp, sf=sf, k=k, tb=tb))
    return lineup_rows, batter_rows


def main():
    conn = sqlite3.connect(DB_PATH)
    # Ensure tables exist
    conn.execute("""CREATE TABLE IF NOT EXISTS game_lineups (
        game_pk INTEGER NOT NULL, team_name TEXT NOT NULL, is_home INTEGER NOT NULL,
        player_id INTEGER NOT NULL, batting_order INTEGER NOT NULL, position TEXT,
        PRIMARY KEY (game_pk, player_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS batter_game_stats (
        game_pk INTEGER NOT NULL, player_id INTEGER NOT NULL, player_name TEXT,
        team_name TEXT NOT NULL, is_home INTEGER NOT NULL,
        ab INTEGER DEFAULT 0, hits INTEGER DEFAULT 0, doubles INTEGER DEFAULT 0,
        triples INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, bb INTEGER DEFAULT 0,
        hbp INTEGER DEFAULT 0, sf INTEGER DEFAULT 0, k INTEGER DEFAULT 0,
        tb INTEGER DEFAULT 0, PRIMARY KEY (game_pk, player_id))""")
    conn.commit()

    # Get existing game_pks
    existing = {r[0] for r in conn.execute("SELECT DISTINCT game_pk FROM batter_game_stats").fetchall()}

    # Get new game_pks from schedule
    log.info(f"Fetching MLB schedule {START_DATE} to {END_DATE}...")
    all_pks = get_game_pks_from_schedule(START_DATE, END_DATE)
    new_pks = [pk for pk in all_pks if pk not in existing]
    log.info(f"Schedule: {len(all_pks)} final games, {len(new_pks)} new")

    # Fetch boxscores
    done = 0
    for pk in new_pks:
        data = fetch_boxscore(pk)
        if data:
            lineup_rows, batter_rows = parse_boxscore(pk, data)
            for lr in lineup_rows:
                conn.execute("INSERT OR IGNORE INTO game_lineups VALUES (?,?,?,?,?,?)",
                             (lr["game_pk"], lr["team_name"], lr["is_home"],
                              lr["player_id"], lr["batting_order"], lr["position"]))
            for br in batter_rows:
                conn.execute("""INSERT OR IGNORE INTO batter_game_stats
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (br["game_pk"], br["player_id"], br["player_name"],
                     br["team_name"], br["is_home"], br["ab"], br["hits"],
                     br["doubles"], br["triples"], br["hr"], br["bb"],
                     br["hbp"], br["sf"], br["k"], br["tb"]))
            conn.commit()
            done += 1
            if done % 50 == 0:
                log.info(f"  {done}/{len(new_pks)} games fetched")
        time.sleep(0.12)

    # Also ensure games table has these game_pks with dates
    # (needed for joins later)
    log.info("Syncing games table...")
    schedule = requests.get(
        f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={START_DATE}&endDate={END_DATE}&gameType=R",
        timeout=30).json()
    for date_entry in schedule.get("dates", []):
        game_date = date_entry.get("date", "")
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            gpk = game["gamePk"]
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            home_score = game.get("teams", {}).get("home", {}).get("score")
            away_score = game.get("teams", {}).get("away", {}).get("score")
            game_id = f"MLB_{gpk}"
            conn.execute("""INSERT OR IGNORE INTO games
                (game_id, sport, season, game_date, home_team, away_team, home_score, away_score, status, source)
                VALUES (?, 'MLB', '2026', ?, ?, ?, ?, ?, 'Final', 'mlb_api')""",
                (game_id, game_date, home, away, home_score, away_score))
    conn.commit()

    # Final counts
    total = conn.execute("SELECT COUNT(DISTINCT game_pk) FROM batter_game_stats WHERE game_pk IN (SELECT DISTINCT game_pk FROM batter_game_stats)").fetchone()[0]
    new_max = conn.execute("""SELECT MAX(g.game_date) FROM batter_game_stats b
        JOIN games g ON g.game_id = 'MLB_' || CAST(b.game_pk AS TEXT)""").fetchone()[0]
    log.info(f"Done! {done} new games fetched. Total game_pks: {total}. Max date: {new_max}")
    conn.close()


if __name__ == "__main__":
    main()
