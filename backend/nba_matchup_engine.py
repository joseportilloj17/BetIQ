"""
nba_matchup_engine.py — NFL-level matchup modeling for NBA projections.

Builds per-team offensive / defensive / transition profiles and per-player
profiles from the current NBA season.  Auto-detects season from today's date.
When the season is young (<20 games per team), blends prior-season data
with weight shifting toward current as games accrue.

Detects mid-season trades and re-weights projections accordingly.

Data sources:
  - nba_api: TeamGameLog, PlayerGameLog, LeagueDashTeamStats (opponent),
             TeamEstimatedMetrics
  - historical.db: team_stats for fallback / caching

Usage:
    from nba_matchup_engine import NBAMatchupEngine
    engine = NBAMatchupEngine()  # auto-detects current season
    engine.load()                # pulls + caches everything
    proj = engine.project_player("Jayson Tatum", opp="MIL")
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from statistics import mean, stdev
from typing import Optional

import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger("NBAMatchup")

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_DB = os.path.join(_DATA_DIR, "nba_matchup_cache.db")
_HIST_DB = os.path.join(_DATA_DIR, "historical.db")

# ── NBA team ID ↔ abbreviation ───────────────────────────────────────────────
# fmt: off
_TEAM_ID_MAP = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA",
    1610612741: "CHI", 1610612739: "CLE", 1610612742: "DAL", 1610612743: "DEN",
    1610612765: "DET", 1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612740: "NOP", 1610612752: "NYK",
    1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR",
    1610612762: "UTA", 1610612764: "WAS",
}
_ABBR_TO_ID = {v: k for k, v in _TEAM_ID_MAP.items()}
# fmt: on

# ESPN → nba_api abbreviation bridge
_ESPN_TO_NBA_ABBR = {
    "WSH": "WAS", "SA": "SAS", "NO": "NOP", "GS": "GSW",
    "NY": "NYK", "UTAH": "UTA", "PHO": "PHX", "BK": "BKN",
}

def normalize_abbr(abbr: str) -> str:
    """Normalize team abbreviation to nba_api format."""
    return _ESPN_TO_NBA_ABBR.get(abbr, abbr)

# ── League-average baselines (updated each load) ─────────────────────────────
_LG_AVG: dict = {}

# Stat keys from TeamGameLog rows
_TEAM_STAT_COLS = [
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
]

_PLAYER_STAT_COLS = [
    "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
]

# Trade deadline (approximate, updated per season)
_TRADE_DEADLINE = "2027-02-06"


def _current_nba_season() -> str:
    """Return the current NBA season string (e.g. '2026-27').

    NBA season runs Oct → Jun.  If today is Jul-Sep, it's off-season
    (return the previous season).  Oct onward = new season.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month
    if month >= 10:
        # Oct-Dec: new season started this calendar year
        return f"{year}-{str(year + 1)[-2:]}"
    elif month <= 6:
        # Jan-Jun: still the season that started last October
        return f"{year - 1}-{str(year)[-2:]}"
    else:
        # Jul-Sep: off-season, use previous season for data
        return f"{year - 1}-{str(year)[-2:]}"


def _prior_season(season: str) -> str:
    """Return the prior season string. '2026-27' → '2025-26'."""
    start_year = int(season[:4])
    return f"{start_year - 1}-{str(start_year)[-2:]}"

# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TeamProfile:
    """Per-team offensive + defensive + transition profile."""
    team: str
    games_played: int = 0

    # Offense (per-game averages)
    off_ppg: float = 0.0
    off_fg_pct: float = 0.0
    off_fg3m: float = 0.0
    off_fg3a: float = 0.0
    off_fg3_pct: float = 0.0
    off_ftm: float = 0.0
    off_fta: float = 0.0
    off_ft_pct: float = 0.0
    off_reb: float = 0.0
    off_ast: float = 0.0
    off_tov: float = 0.0
    off_oreb: float = 0.0

    # Defense (opponent per-game averages — what opponents shoot against us)
    def_ppg: float = 0.0
    def_fg_pct: float = 0.0
    def_fg3m: float = 0.0
    def_fg3a: float = 0.0
    def_fg3_pct: float = 0.0
    def_ftm: float = 0.0
    def_fta: float = 0.0
    def_ast: float = 0.0
    def_tov: float = 0.0   # forced turnovers
    def_stl: float = 0.0
    def_blk: float = 0.0

    # Transition / pace
    pace: float = 0.0       # possessions per 48 min (from estimated metrics)
    off_rating: float = 0.0  # points per 100 possessions
    def_rating: float = 0.0
    net_rating: float = 0.0

    # Schedule context
    avg_rest_days: float = 0.0
    b2b_record: str = ""  # e.g. "12-8"

    # B4: Synergy play-type profile (from prior or current season)
    # {play_type: {"off_ppp": float, "off_poss_pct": float,
    #              "def_ppp": float, "def_poss_pct": float}}
    synergy: dict = field(default_factory=dict)


@dataclass
class PlayerProfile:
    """Per-player profile with role detection + trade awareness."""
    player_id: int = 0
    player_name: str = ""
    team: str = ""
    position: str = ""

    # Per-game averages (post-trade if traded)
    ppg: float = 0.0
    rpg: float = 0.0
    apg: float = 0.0
    spg: float = 0.0
    bpg: float = 0.0
    tpg: float = 0.0   # threes per game
    mpg: float = 0.0
    fpg: float = 0.0   # FGA per game (usage proxy)
    ftm_pg: float = 0.0
    fta_pg: float = 0.0
    tov_pg: float = 0.0

    # Consistency (std dev of key stats)
    pts_std: float = 0.0
    reb_std: float = 0.0
    ast_std: float = 0.0
    thr_std: float = 0.0

    # Role indicators
    games_played: int = 0
    is_starter: bool = False   # avg mpg >= 28
    usage_rate: float = 0.0    # approx (FGA + 0.44*FTA + TOV) / minutes proxy

    # Recent form (last 5 games)
    recent_ppg: float = 0.0
    recent_rpg: float = 0.0
    recent_apg: float = 0.0
    recent_tpg: float = 0.0

    # Home / away splits
    home_ppg: float = 0.0
    away_ppg: float = 0.0

    # Trade info
    was_traded: bool = False
    pre_trade_team: str = ""
    trade_date: str = ""
    post_trade_games: int = 0

    # B2B performance
    b2b_ppg: float = 0.0       # performance on 2nd night of B2B
    b2b_mpg: float = 0.0

    # Per-minute production rates (B2)
    pts_per_min: float = 0.0
    reb_per_min: float = 0.0
    ast_per_min: float = 0.0
    stl_per_min: float = 0.0
    blk_per_min: float = 0.0
    fg3m_per_min: float = 0.0
    ftm_per_min: float = 0.0
    tov_per_min: float = 0.0

    # Rate variability (std of per-minute rates, game-to-game)
    pts_rate_std: float = 0.0
    reb_rate_std: float = 0.0
    ast_rate_std: float = 0.0
    fg3m_rate_std: float = 0.0

    # Combined PRA rate (B3: captures within-player covariance)
    pra_per_min: float = 0.0
    pra_rate_std: float = 0.0


@dataclass
class MatchupProjection:
    """Full game + player projection."""
    home: str = ""
    away: str = ""
    projected_total: float = 0.0
    projected_spread: float = 0.0   # negative = home favored
    home_pts: float = 0.0
    away_pts: float = 0.0
    pace_factor: float = 1.0        # game pace relative to league avg
    # Per-player projections
    players: dict = field(default_factory=dict)  # {name: PlayerProjection}


@dataclass
class PlayerProjection:
    """Single-player projected stat line for a specific game."""
    player_name: str = ""
    team: str = ""
    opponent: str = ""
    minutes: float = 0.0
    points: float = 0.0
    rebounds: float = 0.0
    assists: float = 0.0
    steals: float = 0.0
    blocks: float = 0.0
    threes: float = 0.0
    pra: float = 0.0
    # Standard deviations for prop evaluation
    pts_std: float = 0.0
    reb_std: float = 0.0
    ast_std: float = 0.0
    thr_std: float = 0.0
    # Adjustment factors applied
    pace_adj: float = 1.0
    def_adj: float = 1.0
    form_adj: float = 1.0
    rest_adj: float = 1.0

    # B2: matchup-adjusted per-minute rates (stat = minutes × adj_rate)
    adj_pts_rate: float = 0.0
    adj_reb_rate: float = 0.0
    adj_ast_rate: float = 0.0
    adj_stl_rate: float = 0.0
    adj_blk_rate: float = 0.0
    adj_fg3m_rate: float = 0.0

    # B3: PRA std via combined rate (captures covariance through shared minutes)
    pra_std: float = 0.0
    # B3: reconciliation scale factors (1.0 = no adjustment)
    recon_pts_scale: float = 1.0
    recon_reb_scale: float = 1.0
    recon_ast_scale: float = 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class NBAMatchupEngine:
    """
    Builds team + player profiles from nba_api game logs, detects trades,
    and produces matchup-specific projections.
    """

    # Minimum current-season games per team before we trust current data alone.
    # Below this threshold, prior-season data is blended in.
    _MIN_GAMES_FULL_WEIGHT = 20

    def __init__(self, season: str | None = None,
                 window_pct: float = 1.0,
                 cache: bool = True):
        # Auto-detect season when not specified
        self.season = season or _current_nba_season()
        self.prior_season = _prior_season(self.season)
        self.window_pct = window_pct
        self.cache = cache

        self.team_logs: dict[str, pd.DataFrame] = {}        # abbr → game log
        self.player_logs: dict[int, pd.DataFrame] = {}       # player_id → game log
        self.team_profiles: dict[str, TeamProfile] = {}
        self.player_profiles: dict[str, PlayerProfile] = {}  # name → profile
        self.trades: dict[int, dict] = {}                     # player_id → trade info
        self.opp_stats: pd.DataFrame = pd.DataFrame()
        self.est_metrics: pd.DataFrame = pd.DataFrame()

        # Prior-season data for blending
        self._prior_team_logs: dict[str, pd.DataFrame] = {}
        self._prior_player_logs: dict[int, pd.DataFrame] = {}
        self._prior_opp_stats: pd.DataFrame = pd.DataFrame()
        self._prior_est_metrics: pd.DataFrame = pd.DataFrame()
        self._blend_weight: float = 1.0  # 1.0 = all current, 0.0 = all prior

        # Minutes model (built from game logs, optional)
        self.minutes_model = None  # NBAMinutesModel or None

        # B4: Synergy play-type data (populated externally for ablation)
        # {team_abbr: {play_type: {"off_ppp":..., "def_ppp":..., ...}}}
        self.synergy_data: dict[str, dict] = {}

        # B4: Position-defense data (populated externally for ablation)
        # {team_abbr: {"G": factor, "F": factor, "C": factor}}
        self.position_defense: dict[str, dict[str, float]] = {}

        self._loaded = False

    # ── Data loading ─────────────────────────────────────────────────────────

    def load(self):
        """Load engine data: DB-first for game logs, API for aggregates.

        Game logs (player + team) are stored in historical.db and refreshed
        via bulk LeagueGameLog (2 API calls vs 200-350 per-player calls).
        Aggregate stats (opponent, estimated metrics, rosters) still come
        from nba_api because they're summary endpoints.

        When the current season has fewer than _MIN_GAMES_FULL_WEIGHT games
        per team, also loads prior-season data and blends it in.
        """
        if self._loaded:
            return
        t_start = time.time()
        logger.info("[NBA] Loading matchup engine for %s (window %.0f%%)",
                     self.season, self.window_pct * 100)

        # Try cache first
        if self.cache and self._load_from_cache():
            self._loaded = True
            return

        # Sync game logs to historical.db (bulk fetch, ≤2 API calls)
        self._sync_game_log_db(self.season)

        # Load game logs from DB (0 API calls)
        self._fetch_team_game_logs()
        self._fetch_player_game_logs()

        # Aggregate stats (2 API calls)
        self._fetch_opponent_stats()
        self._fetch_estimated_metrics()

        self._detect_trades()
        self._fetch_roster_teams()

        # Check if we need prior-season blend
        avg_gp = self._avg_team_games_played()
        if avg_gp < self._MIN_GAMES_FULL_WEIGHT:
            self._blend_weight = min(avg_gp / self._MIN_GAMES_FULL_WEIGHT, 1.0)
            logger.info("[NBA] Early season (%d avg GP) — loading prior season %s "
                        "(blend weight: %.2f current / %.2f prior)",
                        avg_gp, self.prior_season,
                        self._blend_weight, 1.0 - self._blend_weight)
            # Sync prior-season logs (completed season → 0 API calls if already in DB)
            self._sync_game_log_db(self.prior_season)
            self._fetch_prior_season()
        else:
            self._blend_weight = 1.0

        self._build_team_profiles()
        self._build_player_profiles()
        if self._blend_weight < 1.0:
            self._blend_with_prior()
        self._apply_roster_adjustment()
        self._compute_league_averages()

        # Build minutes model from raw game logs (not available from cache)
        if self.player_logs:
            try:
                from nba_minutes_model import NBAMinutesModel
                mm = NBAMinutesModel()
                mm.build(self.player_logs, self.team_logs, self.player_profiles,
                         prior_player_logs=self._prior_player_logs or None)
                self.minutes_model = mm
                logger.info("[NBA] Minutes model built: %d players",
                             len(mm._player_stats))
            except Exception as e:
                logger.warning("[NBA] Minutes model build failed: %s", e)

        if self.cache:
            self._save_to_cache()

        self._loaded = True
        elapsed = time.time() - t_start
        logger.info("[NBA] Engine loaded: %d teams, %d players, %d trades, "
                     "blend=%.2f (season %s) — %.1fs total",
                     len(self.team_profiles), len(self.player_profiles),
                     len(self.trades), self._blend_weight, self.season,
                     elapsed)

    def _avg_team_games_played(self) -> float:
        """Average games played across loaded team logs."""
        if not self.team_logs:
            return 0
        return mean(len(df) for df in self.team_logs.values())

    # ── Game-log DB layer (historical.db) ──────────────────────────────────

    @staticmethod
    def _ensure_game_log_tables():
        """Create nba_player/team_game_logs tables in historical.db if missing."""
        conn = sqlite3.connect(_HIST_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nba_player_game_logs (
                season TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                player_name TEXT,
                team_id INTEGER,
                team_abbr TEXT,
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                matchup TEXT,
                wl TEXT,
                min REAL, fgm REAL, fga REAL, fg_pct REAL,
                fg3m REAL, fg3a REAL, fg3_pct REAL,
                ftm REAL, fta REAL, ft_pct REAL,
                oreb REAL, dreb REAL, reb REAL,
                ast REAL, stl REAL, blk REAL,
                tov REAL, pf REAL, pts REAL,
                plus_minus REAL,
                PRIMARY KEY (season, player_id, game_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nba_team_game_logs (
                season TEXT NOT NULL,
                team_id INTEGER NOT NULL,
                team_abbr TEXT,
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                matchup TEXT,
                wl TEXT,
                min REAL, fgm REAL, fga REAL, fg_pct REAL,
                fg3m REAL, fg3a REAL, fg3_pct REAL,
                ftm REAL, fta REAL, ft_pct REAL,
                oreb REAL, dreb REAL, reb REAL,
                ast REAL, stl REAL, blk REAL,
                tov REAL, pf REAL, pts REAL,
                plus_minus REAL,
                PRIMARY KEY (season, team_id, game_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nba_pgl_season_date "
                     "ON nba_player_game_logs(season, game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nba_tgl_season_date "
                     "ON nba_team_game_logs(season, game_date)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nba_daily_cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                cached_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _sync_game_log_db(self, season: str, force: bool = False):
        """Bulk-fetch game logs via LeagueGameLog and store in historical.db.

        Incremental: skips if DB data is fresh enough.
        For completed (non-current) seasons, skips if any data exists.
        """
        from nba_api.stats.endpoints import leaguegamelog
        self._ensure_game_log_tables()

        from datetime import timezone

        conn = sqlite3.connect(_HIST_DB)
        count = conn.execute(
            "SELECT COUNT(*) FROM nba_player_game_logs WHERE season = ?",
            (season,),
        ).fetchone()[0]
        sync_row = conn.execute(
            "SELECT cached_at FROM nba_daily_cache WHERE key = ?",
            (f"game_log_sync_{season}",),
        ).fetchone()
        conn.close()

        is_current = (season == self.season)

        if count > 0 and not force:
            if not is_current:
                logger.info("[NBA] DB has %d player-log rows for %s "
                            "(completed season) — skip fetch", count, season)
                return
            # Current season: re-sync at most once per 12 h
            if sync_row and sync_row[0]:
                last_sync = datetime.fromisoformat(sync_row[0])
                age_s = (datetime.now(timezone.utc) - last_sync).total_seconds()
                if age_s < 43200:  # 12 h
                    logger.info("[NBA] Game-log sync fresh for %s "
                                "(%.1fh ago, %d rows) — skip fetch",
                                season, age_s / 3600, count)
                    return

        # ── Bulk fetch: 1 call for players, 1 for teams ──
        logger.info("[NBA] Bulk-fetching game logs for %s via LeagueGameLog…",
                    season)
        t0 = time.time()

        try:
            plgl = leaguegamelog.LeagueGameLog(
                season=season, player_or_team_abbreviation="P"
            )
            pdf = plgl.get_data_frames()[0]
            logger.info("[NBA] Fetched %d player rows for %s (%.1fs)",
                        len(pdf), season, time.time() - t0)
        except Exception as e:
            logger.warning("[NBA] Bulk player LeagueGameLog failed for %s: %s",
                           season, e)
            return

        time.sleep(1.0)

        try:
            tlgl = leaguegamelog.LeagueGameLog(
                season=season, player_or_team_abbreviation="T"
            )
            tdf = tlgl.get_data_frames()[0]
            logger.info("[NBA] Fetched %d team rows for %s (%.1fs)",
                        len(tdf), season, time.time() - t0)
        except Exception as e:
            logger.warning("[NBA] Bulk team LeagueGameLog failed for %s: %s",
                           season, e)
            tdf = pd.DataFrame()

        # ── Store to DB ──
        _stat_cols = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                      "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
                      "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]

        conn = sqlite3.connect(_HIST_DB)
        if len(pdf) > 0:
            p_rows = []
            for _, r in pdf.iterrows():
                p_rows.append((
                    season, int(r["PLAYER_ID"]), r["PLAYER_NAME"],
                    int(r["TEAM_ID"]), r["TEAM_ABBREVIATION"],
                    r["GAME_ID"], r["GAME_DATE"], r["MATCHUP"], r["WL"],
                    float(r["MIN"]) if pd.notna(r["MIN"]) else 0.0,
                    *(float(r[c]) if pd.notna(r[c]) else 0.0 for c in _stat_cols),
                ))
            conn.executemany(
                "INSERT OR REPLACE INTO nba_player_game_logs "
                "(season,player_id,player_name,team_id,team_abbr,"
                "game_id,game_date,matchup,wl,min,"
                "fgm,fga,fg_pct,fg3m,fg3a,fg3_pct,"
                "ftm,fta,ft_pct,oreb,dreb,reb,"
                "ast,stl,blk,tov,pf,pts,plus_minus) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                p_rows,
            )

        if len(tdf) > 0:
            t_rows = []
            for _, r in tdf.iterrows():
                t_rows.append((
                    season, int(r["TEAM_ID"]), r["TEAM_ABBREVIATION"],
                    r["GAME_ID"], r["GAME_DATE"], r["MATCHUP"], r["WL"],
                    float(r["MIN"]) if pd.notna(r["MIN"]) else 0.0,
                    *(float(r[c]) if pd.notna(r[c]) else 0.0 for c in _stat_cols),
                ))
            conn.executemany(
                "INSERT OR REPLACE INTO nba_team_game_logs "
                "(season,team_id,team_abbr,game_id,game_date,matchup,wl,min,"
                "fgm,fga,fg_pct,fg3m,fg3a,fg3_pct,"
                "ftm,fta,ft_pct,oreb,dreb,reb,"
                "ast,stl,blk,tov,pf,pts,plus_minus) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                t_rows,
            )

        # Stamp sync time
        conn.execute(
            "INSERT OR REPLACE INTO nba_daily_cache (key, data, cached_at) "
            "VALUES (?, ?, ?)",
            (f"game_log_sync_{season}", str(len(pdf)),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        logger.info("[NBA] Stored game logs for %s → %d player, %d team rows "
                    "(%.1fs total)", season, len(pdf), len(tdf),
                    time.time() - t0)

    @staticmethod
    def _daily_cache_get(key: str, max_age_s: float = 43200) -> str | None:
        """Return cached data if fresh, else None."""
        from datetime import timezone
        try:
            conn = sqlite3.connect(_HIST_DB)
            row = conn.execute(
                "SELECT data, cached_at FROM nba_daily_cache WHERE key = ?",
                (key,),
            ).fetchone()
            conn.close()
            if not row or not row[1]:
                return None
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(row[1])).total_seconds()
            return row[0] if age < max_age_s else None
        except Exception:
            return None

    @staticmethod
    def _daily_cache_put(key: str, data: str):
        """Store data in the daily cache with current UTC timestamp."""
        from datetime import timezone
        conn = sqlite3.connect(_HIST_DB)
        conn.execute(
            "INSERT OR REPLACE INTO nba_daily_cache (key, data, cached_at) "
            "VALUES (?, ?, ?)",
            (key, data, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _load_player_logs_from_db(season: str) -> dict[int, pd.DataFrame]:
        """Read player game logs from historical.db → {player_id: DataFrame}."""
        conn = sqlite3.connect(_HIST_DB)
        df = pd.read_sql_query(
            "SELECT * FROM nba_player_game_logs WHERE season = ? "
            "ORDER BY game_date", conn, params=(season,),
        )
        conn.close()
        if len(df) == 0:
            return {}
        df.columns = [c.upper() for c in df.columns]
        df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"])
        result: dict[int, pd.DataFrame] = {}
        for pid, grp in df.groupby("PLAYER_ID"):
            result[int(pid)] = grp.reset_index(drop=True)
        return result

    @staticmethod
    def _load_team_logs_from_db(season: str) -> dict[str, pd.DataFrame]:
        """Read team game logs from historical.db → {team_abbr: DataFrame}."""
        conn = sqlite3.connect(_HIST_DB)
        df = pd.read_sql_query(
            "SELECT * FROM nba_team_game_logs WHERE season = ? "
            "ORDER BY game_date", conn, params=(season,),
        )
        conn.close()
        if len(df) == 0:
            return {}
        df.columns = [c.upper() for c in df.columns]
        df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"])
        result: dict[str, pd.DataFrame] = {}
        for abbr, grp in df.groupby("TEAM_ABBR"):
            result[str(abbr)] = grp.reset_index(drop=True)
        return result

    def _fetch_prior_season(self):
        """Load prior-season data for early-season blending.

        Game logs come from historical.db (already synced by load()).
        Aggregate stats (opp + estimated metrics) are still fetched via API
        because they're summary endpoints that change shape each season.
        """
        from nba_api.stats.endpoints import (
            leaguedashteamstats, teamestimatedmetrics,
        )
        logger.info("[NBA] Loading prior season %s for blending…", self.prior_season)

        # ── Team game logs from DB — apply 50% cutoff + tank downweight ──
        prior_team_all = self._load_team_logs_from_db(self.prior_season)
        tank_teams = 0
        for abbr, df in prior_team_all.items():
            cutoff = max(int(len(df) * 0.5), 0)
            half = df.iloc[cutoff:].copy()

            half["_game_weight"] = 1.0
            if "WL" in half.columns:
                wins = (half["WL"] == "W").sum()
                total = len(half)
                win_pct = wins / max(total, 1)
                if win_pct < 0.40 and total >= 20:
                    pts_mean = half["PTS"].mean()
                    pts_std = half["PTS"].std()
                    last_15_idx = half.index[-15:]
                    low_pts = half["PTS"] < (pts_mean - pts_std)
                    tank_mask = half.index.isin(last_15_idx) & low_pts
                    half.loc[tank_mask, "_game_weight"] = 0.3
                    n_down = tank_mask.sum()
                    if n_down > 0:
                        tank_teams += 1
                        logger.debug("[NBA] %s: downweighted %d tank/rest games",
                                     abbr, n_down)

            self._prior_team_logs[abbr] = half
            if abbr not in self.team_logs or len(self.team_logs[abbr]) == 0:
                self.team_logs[abbr] = half
        logger.info("[NBA] Prior season: %d team logs from DB "
                    "(%d teams had tank/rest downweights)",
                    len(self._prior_team_logs), tank_teams)

        # ── Opponent stats (API — 1 call) — also fill current if empty ──
        try:
            opp = leaguedashteamstats.LeagueDashTeamStats(
                season=self.prior_season, measure_type_detailed_defense="Opponent"
            )
            self._prior_opp_stats = opp.get_data_frames()[0]
            self._prior_opp_stats["ABBR"] = self._prior_opp_stats["TEAM_ID"].map(_TEAM_ID_MAP)
            if len(self.opp_stats) == 0:
                self.opp_stats = self._prior_opp_stats.copy()
            time.sleep(0.5)
        except Exception:
            pass

        # ── Estimated metrics (API — 1 call) — also fill current if empty ──
        try:
            em = teamestimatedmetrics.TeamEstimatedMetrics(season=self.prior_season)
            self._prior_est_metrics = em.get_data_frames()[0]
            self._prior_est_metrics["ABBR"] = self._prior_est_metrics["TEAM_ID"].map(_TEAM_ID_MAP)
            if len(self.est_metrics) == 0:
                self.est_metrics = self._prior_est_metrics.copy()
            time.sleep(0.5)
        except Exception:
            pass

        # ── Player logs from DB — apply 50% cutoff + qualifying filter ──
        prior_player_all = self._load_player_logs_from_db(self.prior_season)
        use_prior_as_primary = len(self.player_logs) == 0

        if use_prior_as_primary:
            logger.info("[NBA] Current season has 0 players — using prior-season "
                        "roster as primary")

        fetched = 0
        for pid, df in prior_player_all.items():
            if len(df) < 10:
                continue
            # Use last 50% of prior season
            cutoff = max(int(len(df) * 0.5), 0)
            prior_df = df.iloc[cutoff:].copy()
            # Qualifying: >= 20 GP in full season, >= 15 MPG
            if len(df) < 20 or df["MIN"].astype(float).mean() < 15.0:
                continue
            self._prior_player_logs[pid] = prior_df
            if pid not in self.player_logs:
                self.player_logs[pid] = prior_df
            fetched += 1

        logger.info("[NBA] Prior season: %d player logs from DB (primary=%s)",
                    fetched, use_prior_as_primary)

    def _fetch_team_game_logs(self):
        """Load team game logs from historical.db (populated by _sync_game_log_db)."""
        logger.info("[NBA] Loading team game logs from DB…")
        self.team_logs = self._load_team_logs_from_db(self.season)
        logger.info("[NBA] Loaded game logs for %d teams from DB", len(self.team_logs))

    def _fetch_opponent_stats(self):
        """Fetch opponent (defensive) stats — cached for 12 h."""
        cache_key = f"opp_stats_{self.season}"
        cached = self._daily_cache_get(cache_key)
        if cached:
            from io import StringIO
            self.opp_stats = pd.read_json(StringIO(cached), dtype={"TEAM_ID": int})
            self.opp_stats["ABBR"] = self.opp_stats["TEAM_ID"].map(_TEAM_ID_MAP)
            logger.info("[NBA] Opponent stats loaded from daily cache")
            return

        from nba_api.stats.endpoints import leaguedashteamstats
        logger.info("[NBA] Fetching opponent stats (API)…")
        try:
            opp = leaguedashteamstats.LeagueDashTeamStats(
                season=self.season,
                measure_type_detailed_defense="Opponent"
            )
            self.opp_stats = opp.get_data_frames()[0]
            self.opp_stats["ABBR"] = self.opp_stats["TEAM_ID"].map(_TEAM_ID_MAP)
            self._daily_cache_put(cache_key, self.opp_stats.to_json())
            time.sleep(0.5)
        except Exception as e:
            logger.warning("[NBA] Failed to fetch opponent stats: %s", e)

    def _fetch_estimated_metrics(self):
        """Fetch pace / offensive / defensive ratings — cached for 12 h."""
        cache_key = f"est_metrics_{self.season}"
        cached = self._daily_cache_get(cache_key)
        if cached:
            from io import StringIO
            self.est_metrics = pd.read_json(StringIO(cached), dtype={"TEAM_ID": int})
            self.est_metrics["ABBR"] = self.est_metrics["TEAM_ID"].map(_TEAM_ID_MAP)
            logger.info("[NBA] Estimated metrics loaded from daily cache")
            return

        from nba_api.stats.endpoints import teamestimatedmetrics
        logger.info("[NBA] Fetching estimated metrics (API)…")
        try:
            em = teamestimatedmetrics.TeamEstimatedMetrics(season=self.season)
            self.est_metrics = em.get_data_frames()[0]
            self.est_metrics["ABBR"] = self.est_metrics["TEAM_ID"].map(_TEAM_ID_MAP)
            self._daily_cache_put(cache_key, self.est_metrics.to_json())
            time.sleep(0.5)
        except Exception as e:
            logger.warning("[NBA] Failed to fetch estimated metrics: %s", e)

    def _fetch_player_game_logs(self):
        """Load player game logs from historical.db with qualifying filter.

        Applies the same GP / MIN thresholds as the old per-player fetch but
        reads all rows in one shot from the local DB.
        """
        logger.info("[NBA] Loading player game logs from DB…")
        all_logs = self._load_player_logs_from_db(self.season)

        avg_gp = self._avg_team_games_played()
        min_gp = max(3, min(10, int(avg_gp * 0.5)))

        for pid, df in all_logs.items():
            if len(df) >= min_gp and df["MIN"].astype(float).mean() >= 10.0:
                self.player_logs[pid] = df

        logger.info("[NBA] Loaded %d qualifying player logs from DB "
                    "(of %d total, min_gp=%d)",
                    len(self.player_logs), len(all_logs), min_gp)

    def _detect_trades(self):
        """Detect players who changed teams mid-season from game log matchup strings."""
        logger.info("[NBA] Detecting trades...")

        for pid, df in self.player_logs.items():
            if len(df) < 2:
                continue

            # Extract team from MATCHUP: "BOS vs. NYK" or "BOS @ NYK"
            teams = []
            for _, row in df.iterrows():
                mu = str(row.get("MATCHUP", ""))
                # Team is the first 3 chars of the matchup
                team_abbr = mu[:3].strip()
                teams.append((row.get("GAME_DATE", ""), team_abbr))

            # Check if team changed
            unique_teams = list(dict.fromkeys(t[1] for t in teams if t[1]))
            if len(unique_teams) > 1:
                # Find the transition point
                current_team = unique_teams[-1]
                prev_team = unique_teams[-2] if len(unique_teams) >= 2 else ""

                # Find approximate trade date
                trade_date = ""
                for date_str, team in teams:
                    if team == current_team and trade_date == "":
                        trade_date = date_str

                self.trades[pid] = {
                    "from_team": prev_team,
                    "to_team": current_team,
                    "trade_date": trade_date,
                    "player_name": df["PLAYER_NAME"].iloc[0] if "PLAYER_NAME" in df.columns else str(pid),
                }

        logger.info("[NBA] Detected %d trades", len(self.trades))

    def _fetch_roster_teams(self):
        """Fetch rosters from CommonTeamRoster — cached for 12 h.

        Populates self._roster_teams: {player_id: team_abbr}.
        """
        cache_key = f"roster_teams_{self.season}"
        cached = self._daily_cache_get(cache_key)
        if cached:
            self._roster_teams = {int(k): v
                                  for k, v in json.loads(cached).items()}
            logger.info("[NBA] Roster teams loaded from daily cache: %d players",
                        len(self._roster_teams))
            return

        from nba_api.stats.endpoints import commonteamroster
        logger.info("[NBA] Fetching CommonTeamRoster for %d teams (API)…",
                    len(_TEAM_ID_MAP))

        self._roster_teams = {}
        fetched = 0

        for team_id, team_abbr in _TEAM_ID_MAP.items():
            try:
                time.sleep(0.6)
                r = commonteamroster.CommonTeamRoster(
                    team_id=team_id, season=self.season
                )
                df = r.get_data_frames()[0]
                for _, row in df.iterrows():
                    pid = int(row["PLAYER_ID"])
                    self._roster_teams[pid] = team_abbr
                fetched += 1
            except Exception as e:
                logger.warning("[NBA] CommonTeamRoster failed for %s: %s",
                               team_abbr, e)

        self._daily_cache_put(cache_key, json.dumps(self._roster_teams))
        logger.info("[NBA] Roster teams: %d players across %d teams",
                    len(self._roster_teams), fetched)

    # ── Injury status ───────────────────────────────────────────────────────

    _SLEEPER_NBA_URL = "https://api.sleeper.app/v1/players/nba"
    _ESPN_INJURIES_URL = ("https://site.api.espn.com/apis/site/v2/sports/"
                          "basketball/nba/injuries")
    # ESPN team ID → NBA abbreviation (ESPN uses different IDs than nba_api)
    _ESPN_TEAM_ID_TO_ABBR = {
        "1": "ATL", "2": "BOS", "17": "BKN", "30": "CHA",
        "4": "CHI", "5": "CLE", "6": "DAL", "7": "DEN",
        "8": "DET", "9": "GSW", "10": "HOU", "11": "IND",
        "12": "LAC", "13": "LAL", "29": "MEM", "14": "MIA",
        "15": "MIL", "16": "MIN", "3": "NOP", "18": "NYK",
        "25": "OKC", "19": "ORL", "20": "PHI", "21": "PHX",
        "22": "POR", "23": "SAC", "24": "SAS", "28": "TOR",
        "26": "UTA", "27": "WAS",
    }

    # NBA full team name → abbreviation (for official injury report PDF parsing)
    _NBA_TEAM_ABBRS = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }

    # Statuses that mean "expected to play" — overrides N≥7 absence filter
    _STATUS_AVAILABLE = {"Active", "Healthy", None}
    # Sleeper DTD = Day-To-Day (may or may not play)
    # GTD = Game-Time Decision
    _STATUS_EXCLUDE = {"Out", "IR"}
    _STATUS_UNCERTAIN = {"DTD", "GTD", "Doubtful", "Questionable", "Day-To-Day"}

    def fetch_injury_report(self) -> dict[str, dict]:
        """Fetch NBA injury data. Source priority:

        1. Official NBA injury report PDF (via nbainjuries) — authoritative,
           released ~5PM ET on game days. Contains status + reason + matchup.
        2. Sleeper — updates between report releases, real-time.
        3. ESPN — secondary, for players not in Sleeper.

        Conflicts are logged. NBA PDF wins over Sleeper/ESPN when available;
        Sleeper/ESPN add players not on the official report (e.g. late
        scratches announced after the PDF was published).

        Returns {player_name: {"status": str, "team": str, "source": str,
                               "body_part": str, "sleeper_id": str,
                               "espn_conflict": bool}}.
        """
        import requests

        injuries: dict[str, dict] = {}

        # ── Primary: Official NBA Injury Report PDF ──
        nba_pdf_count = 0
        try:
            from nbainjuries import injury as nba_inj
            from datetime import datetime as _dt, timezone as _tz

            # Use 5PM ET (22:00 UTC) — standard report release time
            now_utc = _dt.now(_tz.utc)
            # Try today's 5PM report first, then yesterday's
            for offset_days in [0, 1]:
                try:
                    report_dt = now_utc.replace(
                        hour=17, minute=0, second=0, microsecond=0,
                    )
                    if offset_days:
                        report_dt = report_dt - __import__("datetime").timedelta(days=1)
                    df = nba_inj.get_reportdata(report_dt, return_df=True)
                    if df is not None and len(df) > 0:
                        # NBA PDF status mapping
                        _NBA_STATUS_MAP = {
                            "Out": "Out",
                            "Doubtful": "Doubtful",
                            "Questionable": "Questionable",
                            "Probable": "Probable",
                            "Available": "Active",
                        }
                        for _, row in df.iterrows():
                            name = str(row.get("Player Name", "")).strip()
                            team_full = str(row.get("Team", ""))
                            raw_status = str(row.get("Current Status", ""))
                            reason = str(row.get("Reason", ""))
                            if not name or not raw_status:
                                continue
                            status = _NBA_STATUS_MAP.get(raw_status, raw_status)
                            # Map full team name to abbreviation
                            team = self._NBA_TEAM_ABBRS.get(team_full, "")
                            injuries[name] = {
                                "status": status,
                                "team": team,
                                "source": "nba_pdf",
                                "body_part": reason,
                                "sleeper_id": "",
                                "espn_conflict": False,
                            }
                            nba_pdf_count += 1
                        logger.info("[NBA] Official PDF: %d players (report %s)",
                                    nba_pdf_count, report_dt.strftime("%Y-%m-%d %H:%M"))
                        break  # Got a valid report
                except Exception as e:
                    logger.debug("[NBA] PDF fetch for offset %d failed: %s",
                                 offset_days, str(e)[:80])
                    continue
        except ImportError:
            logger.debug("[NBA] nbainjuries not installed, skipping PDF source")
        except Exception as e:
            logger.warning("[NBA] NBA PDF injury fetch failed: %s", e)

        # ── Secondary: Sleeper ──
        sleeper_added = 0
        sleeper_upgraded = 0
        try:
            r = requests.get(self._SLEEPER_NBA_URL, timeout=20)
            r.raise_for_status()
            data = r.json()
            for sid, p in data.items():
                inj_status = p.get("injury_status")
                if not inj_status:
                    continue
                name = p.get("full_name", "")
                team = p.get("team", "")
                if not name or not team:
                    continue
                if name in injuries:
                    # NBA PDF already has this player — log conflicts
                    pdf_status = injuries[name]["status"]
                    pdf_out = pdf_status in self._STATUS_EXCLUDE
                    sleeper_out = inj_status in self._STATUS_EXCLUDE
                    if pdf_out != sleeper_out:
                        logger.warning(
                            "[NBA] SOURCE CONFLICT: %s — nba_pdf=%s, sleeper=%s "
                            "(Sleeper may be more recent)",
                            name, pdf_status, inj_status,
                        )
                        injuries[name]["espn_conflict"] = True
                    # Preserve PDF as source (authoritative)
                    injuries[name]["sleeper_id"] = sid
                    if not injuries[name]["team"]:
                        injuries[name]["team"] = team
                else:
                    # Not on PDF — Sleeper has later info (post-PDF scratch, etc.)
                    injuries[name] = {
                        "status": inj_status,
                        "team": team,
                        "source": "sleeper",
                        "body_part": p.get("injury_body_part", ""),
                        "sleeper_id": sid,
                        "espn_conflict": False,
                    }
                    sleeper_added += 1
            logger.info("[NBA] Sleeper: %d new players (beyond PDF), %d total",
                        sleeper_added, len([v for v in data.values() if v.get("injury_status")]))
        except Exception as e:
            logger.warning("[NBA] Sleeper NBA injury fetch failed: %s", e)

        # ── Tertiary: ESPN ──
        espn_injuries: dict[str, str] = {}  # name → status
        try:
            r = requests.get(self._ESPN_INJURIES_URL, timeout=10)
            r.raise_for_status()
            data = r.json()
            for team_entry in data.get("injuries", []):
                team_id = str(team_entry.get("id", ""))
                team_abbr = self._ESPN_TEAM_ID_TO_ABBR.get(team_id, "")
                for inj in team_entry.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    name = athlete.get("displayName", "")
                    status = inj.get("status", "")
                    if name and status:
                        espn_injuries[name] = status
            logger.info("[NBA] ESPN injuries: %d players", len(espn_injuries))
        except Exception as e:
            logger.warning("[NBA] ESPN NBA injury fetch failed: %s", e)

        # ── Conflict detection (ESPN vs existing) ──
        _STATUS_MAP = {"Day-To-Day": "DTD"}
        for name, espn_status in espn_injuries.items():
            espn_norm = _STATUS_MAP.get(espn_status, espn_status)
            if name in injuries:
                existing_status = injuries[name]["status"]
                existing_out = existing_status in self._STATUS_EXCLUDE
                espn_out = espn_norm in {"Out", "IR"}
                if existing_out != espn_out:
                    logger.warning("[NBA] ESPN CONFLICT: %s — %s=%s, espn=%s",
                                   name, injuries[name]["source"],
                                   existing_status, espn_norm)
                    injuries[name]["espn_conflict"] = True
            else:
                # ESPN has injury not in PDF or Sleeper — add as tertiary
                injuries[name] = {
                    "status": espn_norm,
                    "team": "",  # ESPN doesn't include abbreviation
                    "source": "espn",
                    "body_part": "",
                    "sleeper_id": "",
                    "espn_conflict": False,
                }

        logger.info("[NBA] Injury report total: %d players (pdf=%d, sleeper+=%d, espn+=%d)",
                    len(injuries), nba_pdf_count, sleeper_added,
                    len(injuries) - nba_pdf_count - sleeper_added)
        self._injury_report = injuries
        return injuries

    def get_excluded_players(self, team: str | None = None) -> set[str]:
        """Return set of player names to exclude from projections.

        Excludes players with status in _STATUS_EXCLUDE (Out, IR).
        Optionally filters to a specific team.
        """
        if not hasattr(self, "_injury_report") or not self._injury_report:
            return set()

        excluded = set()
        for name, info in self._injury_report.items():
            if info["status"] in self._STATUS_EXCLUDE:
                if team is None or info["team"] == team:
                    excluded.add(name)
        return excluded

    def get_available_overrides(self) -> set[str]:
        """Return player names that are healthy/active per injury report.

        These override the N≥7 consecutive-absence filter — a player returning
        from absence who is listed as active should NOT be excluded.
        """
        if not hasattr(self, "_injury_report"):
            return set()

        # Players with injury report entries that are NOT out/uncertain
        available = set()
        for name, info in self._injury_report.items():
            if info["status"] not in self._STATUS_EXCLUDE and \
               info["status"] not in self._STATUS_UNCERTAIN:
                available.add(name)

        # Also: any profiled player NOT in the injury report is implicitly available
        # (only the injury report tracks absences)
        return available

    def log_injury_snapshot(self, game_date: str | None = None):
        """Log current injury report snapshot for forward P(plays|status) measurement.

        Writes to `nba_injury_forward_log` in historical.db:
            snapshot_utc, game_date, player_name, player_id, team, status,
            source, body_part, role, mpg

        Outcome columns (played, actual_minutes) are backfilled after the game
        settles via `settle_injury_log()`.
        """
        if not hasattr(self, "_injury_report") or not self._injury_report:
            return
        import sqlite3
        from datetime import datetime, timezone
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        if not os.path.exists(db_path):
            return
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        gd = game_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = []
        for name, info in self._injury_report.items():
            pid = 0
            role = ""
            mpg = 0.0
            profile = self.player_profiles.get(name) if self.player_profiles else None
            if profile:
                pid = profile.player_id or 0
                mpg = profile.mpg
                if mpg >= 28:
                    role = "starter"
                elif mpg >= 15:
                    role = "rotation"
                else:
                    role = "spot"
            rows.append((
                now_utc, gd, name, pid, info.get("team", ""),
                info.get("status", ""), info.get("source", ""),
                info.get("body_part", ""), role, mpg,
            ))
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nba_injury_forward_log (
                    snapshot_utc   TEXT,
                    game_date      TEXT,
                    player_name    TEXT,
                    player_id      INTEGER,
                    team           TEXT,
                    status         TEXT,
                    source         TEXT,
                    body_part      TEXT,
                    role           TEXT,
                    mpg            REAL,
                    played         INTEGER,       -- backfilled: 1/0/NULL
                    actual_minutes REAL,          -- backfilled
                    PRIMARY KEY (snapshot_utc, player_name)
                )
            """)
            conn.executemany("""
                INSERT OR IGNORE INTO nba_injury_forward_log
                (snapshot_utc, game_date, player_name, player_id, team,
                 status, source, body_part, role, mpg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()
            logger.info("[NBA] Injury snapshot logged: %d players for %s", len(rows), gd)
        except Exception as e:
            logger.warning("[NBA] Injury log failed: %s", e)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strip Unicode accents and normalize whitespace for name matching.

        NBA injury PDFs strip diacritics (e.g. "Dennis Schroder") while
        nba_api game logs preserve them ("Dennis Schröder").  Normalising
        both sides to NFKD-then-strip-combining gives a common key.

        Also handles:
        - "Last, First" format from PDF → "First Last"
        - Suffix-first: "Jr., Ernest Udeh" → "Ernest Udeh Jr."
        """
        import re
        import unicodedata
        if not name or name == "nan":
            return ""
        # NFKD decomposes accented chars; drop combining marks
        nfkd = unicodedata.normalize("NFKD", name)
        ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
        # Fix suffix-first from PDF: "Jr., Ernest Udeh" → "Ernest Udeh Jr."
        m = re.match(r"^(Jr\.|Sr\.|III|II|IV),\s*(.+)$", ascii_name)
        if m:
            ascii_name = f"{m.group(2)} {m.group(1)}"
        # Fix "Last, First" format from PDF: "Adams, Steven" → "Steven Adams"
        elif "," in ascii_name:
            parts = [p.strip() for p in ascii_name.split(",", 1)]
            if len(parts) == 2 and parts[1]:
                ascii_name = f"{parts[1]} {parts[0]}"
        return ascii_name.strip()

    @staticmethod
    def settle_injury_log(game_date: str | None = None):
        """Backfill played/actual_minutes in nba_injury_forward_log by
        joining against nba_player_game_logs.

        For each injury-log row on the given game_date, check if a matching
        box-score row exists. If so: played=1, actual_minutes from MIN.
        If not and game logs exist for that date: played=0, actual_minutes=0.
        Skips dates with no game-log data (games not yet played / not synced).
        """
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
        conn = sqlite3.connect(os.path.abspath(db_path))

        if game_date:
            unsettled = conn.execute("""
                SELECT DISTINCT game_date FROM nba_injury_forward_log
                WHERE game_date = ? AND played IS NULL
            """, (game_date,)).fetchall()
        else:
            unsettled = conn.execute("""
                SELECT DISTINCT game_date FROM nba_injury_forward_log
                WHERE played IS NULL
                ORDER BY game_date
            """).fetchall()

        _norm = NBAMatchupEngine._normalize_name
        total_settled = 0
        skipped_dates = []
        for (gd,) in unsettled:
            # Guard: skip dates where game logs haven't been synced yet
            has_logs = conn.execute("""
                SELECT 1 FROM nba_player_game_logs
                WHERE GAME_DATE LIKE ? LIMIT 1
            """, (f"%{gd}%",)).fetchone()
            if not has_logs:
                skipped_dates.append(gd)
                continue

            # Get all players on the injury log for this date
            log_rows = conn.execute("""
                SELECT player_name, team FROM nba_injury_forward_log
                WHERE game_date = ? AND played IS NULL
            """, (gd,)).fetchall()
            if not log_rows:
                continue

            # Get box-score entries for this date — normalise names
            box = conn.execute("""
                SELECT PLAYER_NAME, MIN
                FROM nba_player_game_logs
                WHERE GAME_DATE LIKE ?
            """, (f"%{gd}%",)).fetchall()
            played_set = set()
            minutes_map: dict[str, float] = {}
            for name, mins in box:
                nk = _norm(name)
                played_set.add(nk)
                try:
                    if isinstance(mins, str) and ":" in mins:
                        parts = mins.split(":")
                        minutes_map[nk] = int(parts[0]) + int(parts[1]) / 60
                    elif mins:
                        minutes_map[nk] = float(mins)
                except (ValueError, TypeError):
                    minutes_map[nk] = 0.0

            for pname, team in log_rows:
                norm_pname = _norm(pname)
                if not norm_pname:
                    continue  # skip 'nan' placeholder rows
                did_play = 1 if norm_pname in played_set else 0
                actual_min = minutes_map.get(norm_pname, 0.0)
                conn.execute("""
                    UPDATE nba_injury_forward_log
                    SET played = ?, actual_minutes = ?
                    WHERE game_date = ? AND player_name = ?
                """, (did_play, actual_min, gd, pname))
                total_settled += 1

            conn.commit()

        conn.close()
        if skipped_dates:
            logger.info("[NBA] Skipped settlement for %d date(s) with no game logs: %s",
                        len(skipped_dates), ", ".join(skipped_dates))
        if total_settled:
            logger.info("[NBA] Settled %d injury-log rows", total_settled)
        return total_settled

    # ── Profile building ─────────────────────────────────────────────────────

    def _get_window_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the last window_pct of games from a sorted game log."""
        n = len(df)
        if n == 0:
            return df
        cutoff = max(int(n * (1.0 - self.window_pct)), 0)
        return df.iloc[cutoff:]

    @staticmethod
    def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
        """Weighted mean, falling back to simple mean if weights are all zero."""
        w_sum = weights.sum()
        if w_sum <= 0:
            val = series.mean()
        else:
            val = (series * weights).sum() / w_sum
        return float(val) if val == val else 0.0

    def _build_team_profiles(self):
        """Build offensive + defensive + transition profiles for each team.

        Uses weighted means when _game_weight column is present (tank/rest
        downweight from prior-season fetch).
        """
        for abbr, df in self.team_logs.items():
            window = self._get_window_games(df)
            if len(window) == 0:
                continue

            gp = len(window)
            tp = TeamProfile(team=abbr, games_played=gp)

            # Use weighted mean when weights are available (tank/rest downweight)
            has_weights = "_game_weight" in window.columns
            if has_weights:
                w = window["_game_weight"]
                _wm = lambda col: self._weighted_mean(window[col], w)
            else:
                _wm = lambda col: float(window[col].mean()) if window[col].mean() == window[col].mean() else 0.0

            tp.off_ppg = _wm("PTS")
            tp.off_fg_pct = _wm("FG_PCT")
            tp.off_fg3m = _wm("FG3M")
            tp.off_fg3a = _wm("FG3A")
            tp.off_fg3_pct = _wm("FG3_PCT") if window["FG3A"].mean() > 0 else 0.0
            tp.off_ftm = _wm("FTM")
            tp.off_fta = _wm("FTA")
            tp.off_ft_pct = _wm("FT_PCT")
            tp.off_reb = _wm("REB")
            tp.off_ast = _wm("AST")
            tp.off_tov = _wm("TOV")
            tp.off_oreb = _wm("OREB")

            # Defense (from opponent stats endpoint — full season)
            opp_row = self.opp_stats[self.opp_stats["ABBR"] == abbr]
            if len(opp_row) > 0:
                r = opp_row.iloc[0]
                opp_gp = float(r.get("GP", gp) or gp)
                tp.def_ppg = float(r.get("OPP_PTS", 0)) / max(opp_gp, 1)
                tp.def_fg_pct = float(r.get("OPP_FG_PCT", 0))
                tp.def_fg3m = float(r.get("OPP_FG3M", 0)) / max(opp_gp, 1)
                tp.def_fg3a = float(r.get("OPP_FG3A", 0)) / max(opp_gp, 1)
                tp.def_fg3_pct = float(r.get("OPP_FG3_PCT", 0))
                tp.def_ftm = float(r.get("OPP_FTM", 0)) / max(opp_gp, 1)
                tp.def_fta = float(r.get("OPP_FTA", 0)) / max(opp_gp, 1)
                tp.def_ast = float(r.get("OPP_AST", 0)) / max(opp_gp, 1)
                tp.def_tov = float(r.get("OPP_TOV", 0)) / max(opp_gp, 1)
                tp.def_stl = float(r.get("OPP_STL", 0)) / max(opp_gp, 1) if "OPP_STL" in r.index else 0.0
                tp.def_blk = float(r.get("OPP_BLK", 0)) / max(opp_gp, 1) if "OPP_BLK" in r.index else 0.0

            # Transition / ratings (from estimated metrics)
            em_row = self.est_metrics[self.est_metrics["ABBR"] == abbr]
            if len(em_row) > 0:
                r = em_row.iloc[0]
                tp.pace = float(r.get("E_PACE", 100.0))
                tp.off_rating = float(r.get("E_OFF_RATING", 110.0))
                tp.def_rating = float(r.get("E_DEF_RATING", 110.0))
                tp.net_rating = float(r.get("E_NET_RATING", 0.0))

            # Schedule context from game log
            if len(window) >= 2:
                dates = window["GAME_DATE_DT"].values
                rest_days = []
                for i in range(1, len(dates)):
                    delta = (dates[i] - dates[i - 1]) / pd.Timedelta("1D")
                    rest_days.append(delta - 1)  # days between games minus 1
                tp.avg_rest_days = mean(rest_days) if rest_days else 1.0

            self.team_profiles[abbr] = tp

    def _build_player_profiles(self):
        """Build per-player profiles with trade awareness."""
        for pid, df in self.player_logs.items():
            if len(df) == 0:
                continue

            name = df["PLAYER_NAME"].iloc[0] if "PLAYER_NAME" in df.columns else str(pid)
            trade = self.trades.get(pid)

            # If traded, only use post-trade games
            if trade:
                trade_date_str = trade["trade_date"]
                try:
                    # Filter to games with the new team
                    new_team = trade["to_team"]
                    mask = df["MATCHUP"].str[:3] == new_team
                    post_trade = df[mask]
                    if len(post_trade) >= 5:
                        df_use = post_trade
                    else:
                        df_use = self._get_window_games(df)
                except Exception:
                    df_use = self._get_window_games(df)
            else:
                df_use = self._get_window_games(df)

            min_use = max(3, min(5, int(self._avg_team_games_played() * 0.3)))
            if len(df_use) < min_use:
                continue

            pp = PlayerProfile(player_id=pid, player_name=name)

            # Current team: (1) trade detection, (2) roster lookup, (3) most recent game
            if trade:
                pp.team = trade["to_team"]
            elif hasattr(self, "_roster_teams") and pid in self._roster_teams:
                pp.team = self._roster_teams[pid]
            else:
                last_matchup = str(df.iloc[-1].get("MATCHUP", ""))
                pp.team = last_matchup[:3].strip()

            # Per-game averages (cast to native float from pandas)
            _f = lambda s: float(s) if s == s else 0.0
            pp.ppg = _f(df_use["PTS"].mean())
            pp.rpg = _f(df_use["REB"].mean())
            pp.apg = _f(df_use["AST"].mean())
            pp.spg = _f(df_use["STL"].mean())
            pp.bpg = _f(df_use["BLK"].mean())
            pp.tpg = _f(df_use["FG3M"].mean())
            pp.mpg = _f(df_use["MIN"].mean())
            pp.fpg = _f(df_use["FGA"].mean())
            pp.ftm_pg = _f(df_use["FTM"].mean())
            pp.fta_pg = _f(df_use["FTA"].mean())
            pp.tov_pg = _f(df_use["TOV"].mean())
            pp.games_played = len(df_use)

            # Consistency (std dev)
            if len(df_use) >= 3:
                pp.pts_std = _f(df_use["PTS"].std())
                pp.reb_std = _f(df_use["REB"].std())
                pp.ast_std = _f(df_use["AST"].std())
                pp.thr_std = _f(df_use["FG3M"].std())

            # Per-minute rates (B2) — from games with >= 10 minutes played
            meaningful = df_use[df_use["MIN"].astype(float) >= 10.0]
            if len(meaningful) >= 3:
                _mv = meaningful["MIN"].astype(float)
                pp.pts_per_min = _f((meaningful["PTS"].astype(float) / _mv).mean())
                pp.reb_per_min = _f((meaningful["REB"].astype(float) / _mv).mean())
                pp.ast_per_min = _f((meaningful["AST"].astype(float) / _mv).mean())
                pp.stl_per_min = _f((meaningful["STL"].astype(float) / _mv).mean())
                pp.blk_per_min = _f((meaningful["BLK"].astype(float) / _mv).mean())
                pp.fg3m_per_min = _f((meaningful["FG3M"].astype(float) / _mv).mean())
                pp.ftm_per_min = _f((meaningful["FTM"].astype(float) / _mv).mean())
                pp.tov_per_min = _f((meaningful["TOV"].astype(float) / _mv).mean())
                # Combined PRA rate (B3: captures covariance through shared minutes)
                _pra_per_game = (meaningful["PTS"].astype(float) +
                                 meaningful["REB"].astype(float) +
                                 meaningful["AST"].astype(float))
                pp.pra_per_min = _f((_pra_per_game / _mv).mean())
                if len(meaningful) >= 5:
                    pp.pts_rate_std = _f((meaningful["PTS"].astype(float) / _mv).std())
                    pp.reb_rate_std = _f((meaningful["REB"].astype(float) / _mv).std())
                    pp.ast_rate_std = _f((meaningful["AST"].astype(float) / _mv).std())
                    pp.fg3m_rate_std = _f((meaningful["FG3M"].astype(float) / _mv).std())
                    pp.pra_rate_std = _f((_pra_per_game / _mv).std())
            elif pp.mpg > 0:
                # Fallback: derive from per-game averages
                pp.pts_per_min = pp.ppg / pp.mpg
                pp.reb_per_min = pp.rpg / pp.mpg
                pp.ast_per_min = pp.apg / pp.mpg
                pp.stl_per_min = pp.spg / pp.mpg
                pp.blk_per_min = pp.bpg / pp.mpg
                pp.fg3m_per_min = pp.tpg / pp.mpg
                pp.ftm_per_min = pp.ftm_pg / pp.mpg
                pp.tov_per_min = pp.tov_pg / pp.mpg
                pp.pra_per_min = (pp.ppg + pp.rpg + pp.apg) / pp.mpg

            # Role detection
            pp.is_starter = bool(pp.mpg >= 28.0)
            if pp.mpg > 0:
                pp.usage_rate = float((pp.fpg + 0.44 * pp.fta_pg + pp.tov_pg) / (pp.mpg / 5.0))

            # Recent form (last 5 games)
            recent = df.tail(5)
            if len(recent) > 0:
                pp.recent_ppg = _f(recent["PTS"].mean())
                pp.recent_rpg = _f(recent["REB"].mean())
                pp.recent_apg = _f(recent["AST"].mean())
                pp.recent_tpg = _f(recent["FG3M"].mean())

            # Home / away splits
            home_games = df_use[~df_use["MATCHUP"].str.contains("@")]
            away_games = df_use[df_use["MATCHUP"].str.contains("@")]
            pp.home_ppg = _f(home_games["PTS"].mean()) if len(home_games) > 0 else pp.ppg
            pp.away_ppg = _f(away_games["PTS"].mean()) if len(away_games) > 0 else pp.ppg

            # Trade info
            if trade:
                pp.was_traded = True
                pp.pre_trade_team = trade["from_team"]
                pp.trade_date = trade["trade_date"]
                pp.post_trade_games = len(df_use)

            # B2B performance
            dates = df["GAME_DATE_DT"].values
            b2b_indices = []
            for i in range(1, len(dates)):
                delta = (dates[i] - dates[i - 1]) / pd.Timedelta("1D")
                if delta <= 1.0:
                    b2b_indices.append(i)
            if b2b_indices:
                b2b_df = df.iloc[b2b_indices]
                pp.b2b_ppg = _f(b2b_df["PTS"].mean())
                pp.b2b_mpg = _f(b2b_df["MIN"].mean())

            self.player_profiles[name] = pp

    def _blend_with_prior(self):
        """Blend current-season profiles with prior-season data.

        Called only when _blend_weight < 1.0 (early season).
        For each numeric stat: blended = w * current + (1-w) * prior.
        """
        w = self._blend_weight
        pw = 1.0 - w

        # Build prior-season team profiles using _prior data (with tank/rest weights)
        prior_team_profiles: dict[str, TeamProfile] = {}
        for abbr, df in self._prior_team_logs.items():
            if len(df) == 0:
                continue
            gp = len(df)
            tp = TeamProfile(team=abbr, games_played=gp)
            has_w = "_game_weight" in df.columns
            if has_w:
                wt = df["_game_weight"]
                _wm = lambda col: self._weighted_mean(df[col], wt)
            else:
                _wm = lambda col: float(df[col].mean()) if df[col].mean() == df[col].mean() else 0.0
            tp.off_ppg = _wm("PTS")
            tp.off_fg_pct = _wm("FG_PCT")
            tp.off_fg3m = _wm("FG3M")
            tp.off_fg3a = _wm("FG3A")
            tp.off_fg3_pct = _wm("FG3_PCT") if df["FG3A"].mean() > 0 else 0.0
            tp.off_ftm = _wm("FTM")
            tp.off_fta = _wm("FTA")
            tp.off_ft_pct = _wm("FT_PCT")
            tp.off_reb = _wm("REB")
            tp.off_ast = _wm("AST")
            tp.off_tov = _wm("TOV")
            tp.off_oreb = _wm("OREB")

            # Prior opp stats
            opp_row = self._prior_opp_stats[self._prior_opp_stats["ABBR"] == abbr] \
                if len(self._prior_opp_stats) > 0 else pd.DataFrame()
            if len(opp_row) > 0:
                r = opp_row.iloc[0]
                opp_gp = float(r.get("GP", gp) or gp)
                tp.def_ppg = float(r.get("OPP_PTS", 0)) / max(opp_gp, 1)
                tp.def_fg_pct = float(r.get("OPP_FG_PCT", 0))
                tp.def_fg3m = float(r.get("OPP_FG3M", 0)) / max(opp_gp, 1)
                tp.def_fg3a = float(r.get("OPP_FG3A", 0)) / max(opp_gp, 1)
                tp.def_fg3_pct = float(r.get("OPP_FG3_PCT", 0))
                tp.def_ftm = float(r.get("OPP_FTM", 0)) / max(opp_gp, 1)
                tp.def_fta = float(r.get("OPP_FTA", 0)) / max(opp_gp, 1)
                tp.def_ast = float(r.get("OPP_AST", 0)) / max(opp_gp, 1)
                tp.def_tov = float(r.get("OPP_TOV", 0)) / max(opp_gp, 1)

            # Prior estimated metrics
            em_row = self._prior_est_metrics[self._prior_est_metrics["ABBR"] == abbr] \
                if len(self._prior_est_metrics) > 0 else pd.DataFrame()
            if len(em_row) > 0:
                r = em_row.iloc[0]
                tp.pace = float(r.get("E_PACE", 100.0))
                tp.off_rating = float(r.get("E_OFF_RATING", 110.0))
                tp.def_rating = float(r.get("E_DEF_RATING", 110.0))
                tp.net_rating = float(r.get("E_NET_RATING", 0.0))

            prior_team_profiles[abbr] = tp

        # Blend team profiles
        _team_blend_fields = [
            "off_ppg", "off_fg_pct", "off_fg3m", "off_fg3a", "off_fg3_pct",
            "off_reb", "off_ftm", "off_fta", "off_ft_pct", "off_ast", "off_tov", "off_oreb",
            "def_ppg", "def_fg_pct", "def_fg3m", "def_fg3a", "def_fg3_pct",
            "def_ftm", "def_fta", "def_ast", "def_tov",
            "pace", "off_rating", "def_rating", "net_rating",
        ]
        blended_count = 0
        for abbr, cur in self.team_profiles.items():
            prior = prior_team_profiles.get(abbr)
            if not prior:
                continue
            for fld in _team_blend_fields:
                cur_val = getattr(cur, fld, 0.0)
                pri_val = getattr(prior, fld, 0.0)
                if cur_val > 0 and pri_val > 0:
                    setattr(cur, fld, w * cur_val + pw * pri_val)
            blended_count += 1

        # Blend player profiles — same approach
        _player_blend_fields = [
            "ppg", "rpg", "apg", "spg", "bpg", "tpg", "mpg",
            "fpg", "ftm_pg", "fta_pg", "tov_pg",
        ]
        blended_players = 0
        for name, cur_pp in self.player_profiles.items():
            pid = cur_pp.player_id
            prior_df = self._prior_player_logs.get(pid)
            if prior_df is None or len(prior_df) < 5:
                continue
            # Build prior averages inline
            for fld, col in [
                ("ppg", "PTS"), ("rpg", "REB"), ("apg", "AST"),
                ("spg", "STL"), ("bpg", "BLK"), ("tpg", "FG3M"),
                ("mpg", "MIN"), ("fpg", "FGA"), ("ftm_pg", "FTM"),
                ("fta_pg", "FTA"), ("tov_pg", "TOV"),
            ]:
                cur_val = getattr(cur_pp, fld, 0.0)
                pri_val = float(prior_df[col].mean()) if col in prior_df.columns else 0.0
                if pri_val == pri_val and pri_val > 0 and cur_val > 0:  # nan guard
                    setattr(cur_pp, fld, w * cur_val + pw * pri_val)

            # Blend per-minute rates (B2)
            if "MIN" in prior_df.columns:
                prior_m = prior_df[prior_df["MIN"].astype(float) >= 10.0]
                if len(prior_m) >= 3:
                    _pm = prior_m["MIN"].astype(float)
                    for fld, col in [
                        ("pts_per_min", "PTS"), ("reb_per_min", "REB"),
                        ("ast_per_min", "AST"), ("stl_per_min", "STL"),
                        ("blk_per_min", "BLK"), ("fg3m_per_min", "FG3M"),
                        ("ftm_per_min", "FTM"), ("tov_per_min", "TOV"),
                    ]:
                        cur_val = getattr(cur_pp, fld, 0.0)
                        pri_val = float((prior_m[col].astype(float) / _pm).mean())
                        if pri_val == pri_val and pri_val > 0 and cur_val > 0:
                            setattr(cur_pp, fld, w * cur_val + pw * pri_val)
                    # B3: blend PRA rate
                    _pra_prior = (prior_m["PTS"].astype(float) +
                                  prior_m["REB"].astype(float) +
                                  prior_m["AST"].astype(float)) / _pm
                    pri_pra = float(_pra_prior.mean())
                    if pri_pra == pri_pra and pri_pra > 0 and cur_pp.pra_per_min > 0:
                        cur_pp.pra_per_min = w * cur_pp.pra_per_min + pw * pri_pra

            blended_players += 1

        logger.info("[NBA] Blended %d teams, %d players (w=%.2f current)",
                     blended_count, blended_players, w)

    def _apply_roster_adjustment(self):
        """Adjust team profiles using minutes-weighted player contributions.

        Computes team offensive strength bottom-up from player per-minute rates
        × their minutes share of the 240 total player-minutes per game.  Only
        applied when we have reliable per-player data (skipped when all data is
        from the same prior-season source, since team logs already reflect that
        roster).
        """
        if not self.player_profiles or not self.team_profiles:
            return

        # When blend_weight ≈ 0 (all prior-season data), team logs and player
        # logs are from the same season — no roster mismatch to correct.
        # The adjustment becomes useful once current-season games start
        # revealing the actual roster's performance.
        if self._blend_weight < 0.15:
            logger.info("[NBA] Roster adjustment skipped (blend_weight=%.2f, "
                         "prior-season data self-consistent)", self._blend_weight)
            return

        adjusted = 0
        for abbr, tp in self.team_profiles.items():
            if tp.off_ppg <= 0 or tp.games_played < 5:
                continue
            team_players = [
                p for p in self.player_profiles.values() if p.team == abbr
            ]
            if not team_players:
                continue

            # Minutes-normalized PPG: Σ(ppg × mpg) / 240 estimates team PPG
            # from known roster (5 players × 48 min = 240 player-min per game)
            total_player_min = sum(p.mpg for p in team_players)
            if total_player_min <= 0:
                continue
            roster_ppg = sum(p.ppg * p.mpg for p in team_players) / 240.0
            # Continuity: fraction of 240 minutes accounted for
            continuity = min(total_player_min / 240.0, 1.0)

            if continuity < 0.70:
                # Too many missing minutes — don't adjust
                continue

            # Blend roster-implied PPG with game-log PPG
            # Weight by continuity: high continuity → trust roster build-up more
            adj_ppg = continuity * roster_ppg + (1 - continuity) * tp.off_ppg
            ratio = adj_ppg / tp.off_ppg if tp.off_ppg > 0 else 1.0
            ratio = max(0.90, min(1.10, ratio))

            if abs(ratio - 1.0) > 0.02:
                tp.off_ppg *= ratio
                tp.off_fg3m *= ratio
                tp.off_ftm *= ratio
                tp.off_ast *= ratio
                adjusted += 1
                logger.debug("[NBA] %s roster adj: %.3f (roster=%.1f, logs=%.1f, "
                             "cont=%.2f)", abbr, ratio, roster_ppg,
                             tp.off_ppg / ratio, continuity)

        if adjusted:
            logger.info("[NBA] Applied roster continuity adjustment to %d teams",
                         adjusted)

    def _compute_league_averages(self):
        """Compute league-wide averages for normalization."""
        global _LG_AVG
        if not self.team_profiles:
            return

        profiles = list(self.team_profiles.values())
        _LG_AVG["ppg"] = mean(tp.off_ppg for tp in profiles)
        _LG_AVG["fg_pct"] = mean(tp.off_fg_pct for tp in profiles)
        _LG_AVG["fg3m"] = mean(tp.off_fg3m for tp in profiles)
        _LG_AVG["fg3_pct"] = mean(tp.off_fg3_pct for tp in profiles)
        _LG_AVG["ast"] = mean(tp.off_ast for tp in profiles)
        _LG_AVG["tov"] = mean(tp.off_tov for tp in profiles)
        _LG_AVG["pace"] = mean(tp.pace for tp in profiles if tp.pace > 0)
        _LG_AVG["off_rating"] = mean(tp.off_rating for tp in profiles if tp.off_rating > 0)
        _LG_AVG["def_rating"] = mean(tp.def_rating for tp in profiles if tp.def_rating > 0)
        _LG_AVG["def_ppg"] = mean(tp.def_ppg for tp in profiles if tp.def_ppg > 0)

    # ── Matchup adjustments ──────────────────────────────────────────────────

    def _expected_possessions(self, home: str, away: str,
                              market_total: float = 0.0) -> float:
        """Expected game possessions.

        When market_total is provided:
            poss = total × 100 / (off_rating_home + off_rating_away)
        Fallback (no market total):
            Oliver formula: (home_pace × away_pace) / league_pace
        """
        lg_pace = _LG_AVG.get("pace", 100.0)

        if market_total > 0:
            h_ort = self.team_profiles.get(home, TeamProfile(team=home)).off_rating
            a_ort = self.team_profiles.get(away, TeamProfile(team=away)).off_rating
            h_ort = h_ort if h_ort > 0 else _LG_AVG.get("off_rating", 112.0)
            a_ort = a_ort if a_ort > 0 else _LG_AVG.get("off_rating", 112.0)
            # total = poss × (home_ort + away_ort) / 100
            sum_ort = h_ort + a_ort
            if sum_ort > 0:
                return market_total * 100.0 / sum_ort
        # Fallback: Oliver formula
        h_pace = self.team_profiles.get(home, TeamProfile(team=home)).pace or lg_pace
        a_pace = self.team_profiles.get(away, TeamProfile(team=away)).pace or lg_pace
        return (h_pace * a_pace) / lg_pace if lg_pace > 0 else 100.0

    def _player_pace_adj(self, team: str, expected_poss: float) -> float:
        """Per-minute rate multiplier: expected game pace vs player's team pace.

        A player's per-minute rate was measured at their team's typical pace.
        When the expected game has more/fewer possessions, scale the rate
        proportionally.  Capped at ±15% to avoid extreme adjustments.
        """
        team_pace = self.team_profiles.get(team, TeamProfile(team=team)).pace
        lg_pace = _LG_AVG.get("pace", 100.0)
        team_pace = team_pace if team_pace > 0 else lg_pace
        if team_pace <= 0:
            return 1.0
        adj = expected_poss / team_pace
        return max(0.85, min(adj, 1.15))

    def _defensive_factor(self, opp: str, stat: str = "points") -> float:
        """How much does this opponent's defense suppress/inflate a stat?

        Returns multiplier: >1.0 = opponent allows more than league avg (good
        matchup for offense), <1.0 = tough defense.
        """
        opp_profile = self.team_profiles.get(opp)
        if not opp_profile:
            return 1.0

        lg_def_ppg = _LG_AVG.get("def_ppg", 112.0)

        if stat in ("points", "pra"):
            opp_allowed = opp_profile.def_ppg
            return opp_allowed / lg_def_ppg if lg_def_ppg > 0 else 1.0
        elif stat == "threes":
            lg_3m = _LG_AVG.get("fg3m", 12.0)
            opp_3m = opp_profile.def_fg3m
            return opp_3m / lg_3m if lg_3m > 0 else 1.0
        elif stat == "assists":
            lg_ast = _LG_AVG.get("ast", 25.0)
            opp_ast = opp_profile.def_ast
            return opp_ast / lg_ast if lg_ast > 0 else 1.0
        elif stat == "rebounds":
            # Defensive rebounds = opponent's missed shots
            # More turnovers / lower FG% = more rebound opportunities
            opp_fg = opp_profile.def_fg_pct
            lg_fg = _LG_AVG.get("fg_pct", 0.47)
            if lg_fg > 0 and opp_fg > 0:
                # Lower opp FG% → more misses → more rebound opportunities
                miss_rate_opp = 1.0 - opp_fg
                miss_rate_lg = 1.0 - lg_fg
                return miss_rate_opp / miss_rate_lg if miss_rate_lg > 0 else 1.0
            return 1.0
        elif stat in ("steals", "blocks"):
            # Steals/blocks depend more on the PLAYER's team defense
            # than the opponent, so minimal adjustment
            return 1.0

        return 1.0

    def _synergy_factor(self, team: str, opp: str) -> float:
        """B4: Play-type matchup factor from Synergy data.

        Compares the offensive team's play-type mix against the defensive
        team's efficiency at defending each play type, relative to league avg.

        Returns multiplier: >1.0 = favorable offensive matchup (team's
        preferred play types face weaker-than-average defense), <1.0 = tough.

        If synergy_data is not populated, returns 1.0 (no-op).
        """
        team_syn = self.synergy_data.get(team)
        opp_syn = self.synergy_data.get(opp)
        if not team_syn or not opp_syn:
            return 1.0

        # Compute league-average defensive PPP per play type
        # from all teams in synergy_data
        all_teams = self.synergy_data.values()
        lg_def_ppp: dict[str, float] = {}
        for pt in team_syn:
            def_ppps = [t.get(pt, {}).get("def_ppp", 0)
                        for t in all_teams if t.get(pt, {}).get("def_ppp", 0) > 0]
            lg_def_ppp[pt] = sum(def_ppps) / len(def_ppps) if def_ppps else 1.0

        # Weighted average: team's play-type mix × opponent's relative defensive efficiency
        weighted_sum = 0.0
        total_weight = 0.0
        for pt, pt_data in team_syn.items():
            off_poss_pct = pt_data.get("off_poss_pct", 0)
            if off_poss_pct <= 0:
                continue

            opp_def_ppp = opp_syn.get(pt, {}).get("def_ppp", 0)
            lg_ppp = lg_def_ppp.get(pt, 1.0)
            if opp_def_ppp <= 0 or lg_ppp <= 0:
                continue

            # ratio > 1 = opponent allows more than avg for this play type
            ratio = opp_def_ppp / lg_ppp
            weighted_sum += off_poss_pct * ratio
            total_weight += off_poss_pct

        if total_weight <= 0:
            return 1.0

        raw = weighted_sum / total_weight
        # Regress toward 1.0: 40% signal, 60% regression
        factor = 0.60 + 0.40 * raw
        return max(0.90, min(1.10, factor))

    def _matchup_factor(self, player: "PlayerProfile", opp: str) -> float:
        """B4: Position-specific defensive matchup factor.

        Uses per-team position defense data (from MatchupsRollup) to compute
        how well the opponent defends this player's position.

        Returns multiplier: >1.0 = opponent weak at guarding this position,
        <1.0 = opponent strong.  If position_defense is not populated,
        returns 1.0 (no-op).
        """
        opp_def = self.position_defense.get(opp)
        if not opp_def:
            return 1.0

        # Classify player position from stats
        rpg = player.rpg if player.rpg > 0 else 0.0
        apg = player.apg if player.apg > 0 else 0.0
        bpg = player.bpg if player.bpg > 0 else 0.0
        fg3a = player.tpg / max(player.mpg, 1.0) * 36.0 if player.mpg > 0 else 0.0

        if rpg >= 8 and bpg >= 1.0:
            pos = "C"
        elif rpg >= 6 and apg < 5:
            pos = "F"
        elif apg >= 5 or fg3a >= 5:
            pos = "G"
        elif rpg >= 5:
            pos = "F"
        else:
            pos = "G"

        raw = opp_def.get(pos, 1.0)
        # Regress toward 1.0: 50% signal, 50% regression
        factor = 0.50 + 0.50 * raw
        return max(0.85, min(1.15, factor))

    def _rest_factor(self, team: str, rest_days: int = 1,
                     is_b2b: bool = False) -> float:
        """Adjustment for rest advantage / disadvantage.

        Based on NBA research: B2B teams score ~2.5 fewer points.
        """
        if is_b2b:
            return 0.975  # ~2.5% scoring reduction on B2B
        elif rest_days >= 3:
            return 1.01   # small boost for extended rest
        return 1.0

    def _form_factor(self, player: PlayerProfile) -> float:
        """Recent form adjustment — last 5 vs season baseline."""
        if player.ppg <= 0 or player.recent_ppg <= 0:
            return 1.0

        ratio = player.recent_ppg / player.ppg
        # Regress toward 1.0 (don't overweight streaks)
        # 30% weight on recent form, 70% regression to mean
        adj = 0.70 + 0.30 * ratio
        return max(0.85, min(1.15, adj))

    def _home_away_factor(self, player: PlayerProfile, is_home: bool) -> float:
        """Home/away scoring split adjustment."""
        if player.ppg <= 0:
            return 1.0
        if is_home:
            return player.home_ppg / player.ppg if player.ppg > 0 else 1.0
        else:
            return player.away_ppg / player.ppg if player.ppg > 0 else 1.0

    # ── B3: Minutes + team-level reconciliation ───────────────────────────

    def _enforce_team_minutes(
        self,
        proj: MatchupProjection,
        home: str,
        away: str,
        target: float = 240.0,
    ) -> None:
        """Scale each team's player minutes proportionally to sum to target.

        5 players × 48 min = 240.  Every team MUST sum to 240 (or target)
        for counting-stat reconciliation to be meaningful.  After scaling
        minutes, recompute counting stats from stored adj_rates.
        """
        for team_abbr in (home, away):
            team_players = [
                (n, p) for n, p in proj.players.items() if p.team == team_abbr
            ]
            if len(team_players) < 3:
                continue

            total_min = sum(p.minutes for _, p in team_players)
            if total_min <= 0:
                continue

            scale = target / total_min
            # Apply proportional scaling (no cap — this is a hard physical constraint)
            for _, p in team_players:
                new_min = round(p.minutes * scale, 1)
                # Floor: starters ≥ 20, rotation ≥ 8, anyone ≥ 0
                profile = self.player_profiles.get(p.player_name)
                if profile and profile.is_starter:
                    new_min = max(new_min, 20.0)
                elif profile and profile.mpg >= 15:
                    new_min = max(new_min, 8.0)
                # Ceiling: no one plays > 44 min
                new_min = min(new_min, 44.0)
                p.minutes = new_min
                # Recompute counting stats from stored rates
                if p.adj_pts_rate > 0:
                    p.points = round(new_min * p.adj_pts_rate, 1)
                    p.rebounds = round(new_min * p.adj_reb_rate, 1)
                    p.assists = round(new_min * p.adj_ast_rate, 1)
                    p.steals = round(new_min * p.adj_stl_rate, 1)
                    p.blocks = round(new_min * p.adj_blk_rate, 1)
                    p.threes = round(new_min * p.adj_fg3m_rate, 1)
                    p.pra = round(p.points + p.rebounds + p.assists, 1)

            # Second pass: if floor/ceiling constraints broke the sum,
            # distribute residual to unconstrained players
            new_total = sum(p.minutes for _, p in team_players)
            residual = target - new_total
            if abs(residual) > 0.5:
                # Find players not at floor or ceiling
                flexible = [
                    (n, p) for n, p in team_players
                    if 8.5 < p.minutes < 43.5
                ]
                if flexible:
                    flex_total = sum(p.minutes for _, p in flexible)
                    if flex_total > 0:
                        for _, p in flexible:
                            share = p.minutes / flex_total
                            adj = round(p.minutes + residual * share, 1)
                            adj = max(adj, 0.0)
                            adj = min(adj, 44.0)
                            p.minutes = adj
                            if p.adj_pts_rate > 0:
                                p.points = round(adj * p.adj_pts_rate, 1)
                                p.rebounds = round(adj * p.adj_reb_rate, 1)
                                p.assists = round(adj * p.adj_ast_rate, 1)
                                p.steals = round(adj * p.adj_stl_rate, 1)
                                p.blocks = round(adj * p.adj_blk_rate, 1)
                                p.threes = round(adj * p.adj_fg3m_rate, 1)
                                p.pra = round(p.points + p.rebounds + p.assists, 1)

    @staticmethod
    def _derive_scoring_breakdown(team_pts: float, tp: TeamProfile) -> dict:
        """Derive FGM, FGA, FG missed, FT missed from projected team points.

        Uses team shooting profile (FG%, 3PT%, FT%) to decompose projected
        points into shot attempts and makes.  Returns dict with all components.
        """
        if tp.off_ppg <= 0 or team_pts <= 0:
            return {}
        # Scale factor: how much this game's scoring differs from season avg
        scale = team_pts / tp.off_ppg

        fg3m = tp.off_fg3m * scale
        ftm = tp.off_ftm * scale
        fta = tp.off_fta * scale
        # 2-point FGM: (PTS - 3*FG3M - FTM) / 2
        fg2m = max(0, (team_pts - 3 * fg3m - ftm) / 2.0)
        fgm = fg2m + fg3m

        # FGA from FGM / FG%
        fg_pct = tp.off_fg_pct if tp.off_fg_pct > 0 else 0.46
        fga = fgm / fg_pct if fg_pct > 0 else fgm * 2.2
        fg_missed = fga - fgm
        ft_missed = fta - ftm

        # Assist rate (assists per FGM)
        ast_rate = (tp.off_ast / (tp.off_ppg - tp.off_ftm) * 2.0
                    ) if tp.off_ppg > tp.off_ftm else 0.60
        # Simplified: assists / fgm from season data
        season_fgm = (tp.off_ppg - 3 * tp.off_fg3m - tp.off_ftm) / 2.0 + tp.off_fg3m
        ast_rate = tp.off_ast / season_fgm if season_fgm > 0 else 0.60

        return {
            "fgm": fgm, "fga": fga, "fg_missed": fg_missed,
            "ftm": ftm, "fta": fta, "ft_missed": ft_missed,
            "fg3m": fg3m, "ast_rate": ast_rate, "scale": scale,
        }

    # ── OTHER bucket: expected share of team scoring outside the named pool ──
    # Empirical from R1+R2 (9,740 team-games, 4 seasons).
    # Key: n_named (players in named pool with actuals), value: OTHER_%pts.
    _OTHER_PTS_SHARE = {
        0: 1.000, 1: 0.898, 2: 0.677, 3: 0.578, 4: 0.442,
        5: 0.322, 6: 0.221, 7: 0.147, 8: 0.096, 9: 0.062, 10: 0.047,
    }

    @classmethod
    def _named_pts_share(cls, n_named: int) -> float:
        """Fraction of team points attributable to named pool, given pool size."""
        n = max(0, min(n_named, 10))
        return 1.0 - cls._OTHER_PTS_SHARE.get(n, 0.047)

    def _reconcile_team_stats(
        self,
        proj: MatchupProjection,
        home: str,
        away: str,
        cap: float = 0.25,
        strength: float = 1.0,
    ) -> None:
        """Scale player projections so team sums match projected totals.

        Points: reconcile to proj.home_pts × named_share (not full team total).
            named_share is estimated from the number of projected players,
            based on R1+R2 empirical OTHER bucket data (9,740 team-games).
            This prevents the named pool from being inflated to account for
            bench scoring that leaks to un-evaluated players.
        Rebounds: derived from market-anchored game state — opponent's
            missed FGs/FTs create rebound opportunities + own offensive rebounds.
        Assists: derived from team FGM × assist rate.  Σ assists ≤ team FGM.

        cap: maximum |scale − 1| allowed (0.25 = ±25%).  Set to e.g. 99.0
            to effectively uncap.
        strength: 0.0–1.0 partial pull.  1.0 = full reconciliation,
            0.5 = move halfway toward target, 0.0 = no change.

        Modifies PlayerProjection objects in proj.players in-place.
        Stores scale factors on each projection for transparency.
        """
        h_prof = self.team_profiles.get(home, TeamProfile(team=home))
        a_prof = self.team_profiles.get(away, TeamProfile(team=away))

        # Derive scoring breakdown for both teams (needed for cross-team rebounds)
        h_breakdown = self._derive_scoring_breakdown(proj.home_pts, h_prof)
        a_breakdown = self._derive_scoring_breakdown(proj.away_pts, a_prof)

        for team_abbr, team_target_pts in [
            (home, proj.home_pts),
            (away, proj.away_pts),
        ]:
            team_players = [
                (n, p) for n, p in proj.players.items() if p.team == team_abbr
            ]
            if len(team_players) < 3:
                continue

            is_home = (team_abbr == home)
            team_prof = h_prof if is_home else a_prof
            opp_prof = a_prof if is_home else h_prof
            team_bd = h_breakdown if is_home else a_breakdown
            opp_bd = a_breakdown if is_home else h_breakdown

            # ── Points: reconcile to team total × named_share ──
            # named_share accounts for the OTHER bucket — scoring by
            # players outside the named pool (bench, two-way, etc.).
            # When pool_fix=False (legacy), named_share=1.0 (full team total).
            n_named = len(team_players)
            if self._pool_fix:
                named_share = self._named_pts_share(n_named)
            else:
                named_share = 1.0
            pts_target = team_target_pts * named_share
            sum_pts = sum(p.points for _, p in team_players)
            if sum_pts > 0 and pts_target > 0:
                raw_pts_scale = pts_target / sum_pts
                # Cap and partial-pull
                pts_scale = max(1.0 - cap, min(raw_pts_scale, 1.0 + cap))
                if strength < 1.0:
                    pts_scale = 1.0 + strength * (pts_scale - 1.0)
                if pts_scale != raw_pts_scale:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[PTS_CAP_BIND] %s: raw_scale=%.3f capped=%.3f "
                        "target=%.1f (%.0f%% of %.1f) Σpts=%.1f n_named=%d",
                        team_abbr, raw_pts_scale, pts_scale,
                        pts_target, named_share * 100, team_target_pts,
                        sum_pts, n_named,
                    )
                for _, p in team_players:
                    p.points = round(p.points * pts_scale, 1)
                    p.adj_pts_rate *= pts_scale
                    p.recon_pts_scale = round(pts_scale, 4)

            # ── Rebounds: market-derived from missed shots ──
            # Total player-credited rebound opportunities for this team:
            #   DREB ≈ opp missed FGs × live_ball_rate × (1 - OPP OREB%)
            #        + opp missed FTs × 0.85
            #   OREB ≈ own missed FGs × live_ball_rate × team OREB rate
            #
            # _LIVE_BALL_RATE = 0.78 removes dead-ball misses (~22%), which
            # are the same population as "team rebounds" (not credited to
            # individual players, ~3.5/game/team).  No additional deduction
            # needed — mean reb_scale = 0.976 without one.
            #
            # DREB uses the OPPONENT's OREB% (not the team's own) because
            # the opponent's offensive rebounding rate determines what
            # fraction of their misses they recover vs. what's left for us.
            _LIVE_BALL_RATE = 0.78
            if team_bd and opp_bd:
                # Opponent's missed shots → defensive rebound chances.
                # opp OREB% determines what fraction the opponent recovers;
                # the rest are available as defensive rebounds for this team.
                opp_oreb_rate = (opp_prof.off_oreb / max(opp_bd["fg_missed"] * _LIVE_BALL_RATE, 1)
                                 ) if opp_bd["fg_missed"] > 0 else 0.25
                opp_oreb_rate = min(opp_oreb_rate, 0.40)  # cap
                def_reb_chances = (opp_bd["fg_missed"] * _LIVE_BALL_RATE * (1 - opp_oreb_rate)
                                   + opp_bd["ft_missed"] * 0.85)

                # Own missed shots → offensive rebound chances
                team_oreb_rate = (team_prof.off_oreb / max(team_bd["fg_missed"] * _LIVE_BALL_RATE, 1)
                                  ) if team_bd["fg_missed"] > 0 else 0.25
                team_oreb_rate = min(team_oreb_rate, 0.40)
                off_reb = team_bd["fg_missed"] * _LIVE_BALL_RATE * team_oreb_rate

                team_reb_target = def_reb_chances + off_reb
                sum_reb = sum(p.rebounds for _, p in team_players)
                if sum_reb > 0 and team_reb_target > 0:
                    reb_scale = team_reb_target / sum_reb
                    reb_cap = min(cap, 0.15)  # REB cap at most ±15%
                    reb_scale = max(1.0 - reb_cap, min(reb_scale, 1.0 + reb_cap))
                    if strength < 1.0:
                        reb_scale = 1.0 + strength * (reb_scale - 1.0)
                    if abs(reb_scale - 1.0) > 0.02:
                        for _, p in team_players:
                            p.rebounds = round(p.rebounds * reb_scale, 1)
                            p.adj_reb_rate *= reb_scale
                            p.recon_reb_scale = round(reb_scale, 4)

            # ── Assists: from team FGM × assist rate, capped at FGM ──
            if team_bd and team_bd.get("fgm", 0) > 0:
                team_fgm = team_bd["fgm"]
                ast_rate = team_bd["ast_rate"]
                team_ast_target = team_fgm * ast_rate
                # Hard constraint: assists ≤ FGM
                team_ast_target = min(team_ast_target, team_fgm)

                sum_ast = sum(p.assists for _, p in team_players)
                if sum_ast > 0 and team_ast_target > 0:
                    ast_scale = team_ast_target / sum_ast
                    ast_cap = min(cap, 0.15)  # AST cap at most ±15%
                    ast_scale = max(1.0 - ast_cap, min(ast_scale, 1.0 + ast_cap))
                    if strength < 1.0:
                        ast_scale = 1.0 + strength * (ast_scale - 1.0)
                    if abs(ast_scale - 1.0) > 0.02:
                        for _, p in team_players:
                            p.assists = round(p.assists * ast_scale, 1)
                            p.adj_ast_rate *= ast_scale
                            p.recon_ast_scale = round(ast_scale, 4)

                # Enforce hard cap: Σ assists ≤ team FGM
                sum_ast_post = sum(p.assists for _, p in team_players)
                if sum_ast_post > team_fgm:
                    cap_scale = team_fgm / sum_ast_post
                    for _, p in team_players:
                        p.assists = round(p.assists * cap_scale, 1)
                        p.adj_ast_rate *= cap_scale

            # Recompute PRA after reconciliation
            for _, p in team_players:
                p.pra = round(p.points + p.rebounds + p.assists, 1)

    # ── Projection ───────────────────────────────────────────────────────────

    def project_game(self, home: str, away: str,
                     home_rest: int = 1, away_rest: int = 1,
                     home_b2b: bool = False, away_b2b: bool = False,
                     market_total: float = 0.0,
                     market_spread: float = 0.0,
                     factor_config: dict | None = None,
                     exclude_players: set | None = None,
                     soft_allocation: bool = False,
                     ) -> MatchupProjection:
        """Project a full game: team totals + player lines.

        exclude_players: optional set of player names to exclude from
            game-day rotation (e.g. known inactives from BoxScoreSummaryV2).
            When provided, these players are filtered out BEFORE roster
            selection, so their minutes are naturally redistributed to
            the remaining active players.

        soft_allocation: when True, removes the hard top-10/cum-270 roster cap
            and includes ALL available players with MPG >= 3.  The minutes model
            and _enforce_team_minutes() handle the 240-minute constraint.
            Deep bench players get small minutes naturally.  (R11 pool fix.)

        factor_config: optional dict controlling which adjustments are active.
            Keys and defaults (when omitted):
              defense: True   — opponent defensive factor (B4: KEEP)
              blowout: True   — blowout script minutes adjustment (B4: KEEP)
              reconcile: True  — B3 team-total reconciliation (R13: 50% PROMOTED)
              reconcile_pool_fix: True — adjust reconcile target by OTHER bucket share
              reconcile_strength: 0.5 — partial pull (R13: 50% beats full and none for PTS/PRA)
              reconcile_cap: 0.25 — max |scale - 1| for pts reconciliation
              form: False     — recent form adjustment (B4: KILL)
              synergy: False  — Synergy play-type factor (B4: KILL)
              matchup: False  — position matchup factor (B4: KILL)
            Used by B4 ablation to selectively enable/disable factors.

        When market_total is provided, game totals are derived from it
        (anchored to market) and expected possessions are computed from
        total / (off_rating_home + off_rating_away).  Per-minute rates
        are then scaled by expected_poss / player_team_pace.

        When market_total is 0, falls back to Pythagorean blend + Oliver
        pace formula (original behavior, without the old double-count).
        """
        if not self._loaded:
            self.load()

        home = normalize_abbr(home)
        away = normalize_abbr(away)
        proj = MatchupProjection(home=home, away=away)

        h_prof = self.team_profiles.get(home, TeamProfile(team=home))
        a_prof = self.team_profiles.get(away, TeamProfile(team=away))

        # Expected game possessions (from market total or Oliver formula)
        expected_poss = self._expected_possessions(home, away, market_total)
        lg_pace = _LG_AVG.get("pace", 100.0)
        proj.pace_factor = expected_poss / lg_pace if lg_pace > 0 else 1.0

        # ── Game totals ──
        if market_total > 0:
            # Anchor to market total; split by spread + HCA
            total = market_total
            spread = market_spread  # negative = home favored
            h_raw = (total / 2.0) - (spread / 2.0)
            a_raw = (total / 2.0) + (spread / 2.0)
        else:
            # Pythagorean blend (no separate pace multiplier — already in off_ppg)
            lg_ppg = _LG_AVG.get("ppg", 112.0)
            lg_def = _LG_AVG.get("def_ppg", 112.0)

            h_off = h_prof.off_ppg if h_prof.off_ppg > 0 else lg_ppg
            a_def = a_prof.def_ppg if a_prof.def_ppg > 0 else lg_def
            h_raw = sqrt(h_off * a_def) if h_off > 0 and a_def > 0 else lg_ppg

            a_off = a_prof.off_ppg if a_prof.off_ppg > 0 else lg_ppg
            h_def = h_prof.def_ppg if h_prof.def_ppg > 0 else lg_def
            a_raw = sqrt(a_off * h_def) if a_off > 0 and h_def > 0 else lg_ppg

            # Home court advantage (only when not using market)
            h_raw += 1.5
            a_raw -= 1.5

        # Rest adjustments (only for model-derived totals; market total
        # already reflects this via the line itself)
        if market_total <= 0:
            h_raw *= self._rest_factor(home, home_rest, home_b2b)
            a_raw *= self._rest_factor(away, away_rest, away_b2b)

        proj.home_pts = round(h_raw, 1)
        proj.away_pts = round(a_raw, 1)
        proj.projected_total = round(h_raw + a_raw, 1)
        proj.projected_spread = round(a_raw - h_raw, 1)  # negative = home favored

        # Player projections
        proj.players = {}
        # Collect minutes projections per team for redistribution
        team_min_projs: dict[str, dict[str, object]] = {home: {}, away: {}}

        # Select game-day rotation per team.
        # soft_allocation=False (legacy): top players by MPG, up to 10 or
        #   until cumulative MPG reaches 270. Avoids inflating from deep rosters.
        # soft_allocation=True (R11): include ALL available players with
        #   MPG >= 3; let the minutes model allocate 240 across them.
        _exclude = exclude_players or set()
        game_roster: dict[str, list[str]] = {}
        if soft_allocation:
            _MIN_MPG_SOFT = 3.0
            for team_abbr in (home, away):
                candidates = sorted(
                    [(n, p) for n, p in self.player_profiles.items()
                     if p.team == team_abbr and p.mpg >= _MIN_MPG_SOFT
                     and n not in _exclude],
                    key=lambda x: x[1].mpg, reverse=True,
                )
                game_roster[team_abbr] = [n for n, _ in candidates]
        else:
            _MAX_ROTATION = 10
            _MAX_CUMULATIVE_MPG = 270.0
            for team_abbr in (home, away):
                candidates = sorted(
                    [(n, p) for n, p in self.player_profiles.items()
                     if p.team == team_abbr and p.mpg >= 10.0
                     and n not in _exclude],
                    key=lambda x: x[1].mpg, reverse=True,
                )
                selected = []
                cum_mpg = 0.0
                for n, p in candidates:
                    if len(selected) >= _MAX_ROTATION:
                        break
                    if cum_mpg >= _MAX_CUMULATIVE_MPG and len(selected) >= 8:
                        break
                    selected.append(n)
                    cum_mpg += p.mpg
                game_roster[team_abbr] = selected

        for name, pp in self.player_profiles.items():
            if pp.team not in (home, away):
                continue
            if name not in game_roster.get(pp.team, []):
                continue

            is_home = (pp.team == home)
            opp = away if is_home else home
            # Pass game spread for blowout script (from player's team perspective)
            team_spread = proj.projected_spread if is_home else -proj.projected_spread

            poss_adj = self._player_pace_adj(pp.team, expected_poss)
            player_proj = self._project_single_player(
                pp, opp, is_home, poss_adj,
                is_b2b=(home_b2b if is_home else away_b2b),
                market_spread=team_spread,
                factor_config=factor_config,
            )
            proj.players[name] = player_proj

            # Stash minutes projection for redistribution
            if self.minutes_model and pp.player_id:
                mp = self.minutes_model.get_player_stats(pp.player_id)
                if mp:
                    from nba_minutes_model import MinutesProjection
                    team_min_projs[pp.team][name] = MinutesProjection(
                        projected_minutes=player_proj.minutes,
                        role=mp.get("role", "rotation"),
                    )

        # Redistribute team minutes → 240 (starters fixed, bench scales)
        if self.minutes_model:
            from nba_minutes_model import NBAMinutesModel
            for team_abbr in (home, away):
                team_mp = team_min_projs.get(team_abbr, {})
                if len(team_mp) < 5:
                    continue
                NBAMinutesModel.redistribute_team_minutes(team_mp)
                for name, mp in team_mp.items():
                    if name in proj.players:
                        p = proj.players[name]
                        new_min = mp.projected_minutes
                        if abs(new_min - p.minutes) > 0.05:
                            p.minutes = round(new_min, 1)
                            # Recompute counting stats from stored rates (B2)
                            if p.adj_pts_rate > 0:
                                p.points = round(new_min * p.adj_pts_rate, 1)
                                p.rebounds = round(new_min * p.adj_reb_rate, 1)
                                p.assists = round(new_min * p.adj_ast_rate, 1)
                                p.steals = round(new_min * p.adj_stl_rate, 1)
                                p.blocks = round(new_min * p.adj_blk_rate, 1)
                                p.threes = round(new_min * p.adj_fg3m_rate, 1)
                                p.pra = round(p.points + p.rebounds + p.assists, 1)

        # B3: Enforce 240-minute constraint for ALL teams (always runs).
        # After the minutes-model redistribution (if available), proportionally
        # scale remaining players so each team sums to exactly 240.
        self._enforce_team_minutes(proj, home, away, target=240.0)

        # B3: Reconcile player sums → team totals
        # R13: 50% reconciliation PROMOTED. Tested on corrected pool (R11:
        # OTHER=2.1%, soft allocation). PTS ΔCRPS=-0.026***, PRA=-0.021***,
        # REB neutral. 50% beats both full (100%) and none (0%).
        _fc = factor_config or {}
        if _fc.get("reconcile", True):
            self._pool_fix = _fc.get("reconcile_pool_fix", True)
            self._reconcile_team_stats(
                proj, home, away,
                cap=_fc.get("reconcile_cap", 0.25),
                strength=_fc.get("reconcile_strength", 0.5),
            )

        return proj

    def get_absence_impacts_for_game(
        self, proj, home: str, away: str,
    ) -> dict[str, dict[str, float]]:
        """Extract measured absence impacts for the sim.

        Returns {absent_player_name: {beneficiary_name: delta_min}}.
        Only includes players in the projection with measured pairwise data.
        Uses the minutes model's shrunk with/without deltas.
        """
        if not self.minutes_model:
            return {}
        # Build name→pid and pid→name maps for projected players
        name_to_pid: dict[str, int] = {}
        pid_to_name: dict[int, str] = {}
        for name, pp in proj.players.items():
            profile = self.player_profiles.get(name)
            if profile and profile.player_id:
                name_to_pid[name] = profile.player_id
                pid_to_name[profile.player_id] = name

        impacts: dict[str, dict[str, float]] = {}
        for team in (home, away):
            # Get all (team, absent_pid) keys from the minutes model
            team_pids = {pid for pid, n in pid_to_name.items()
                         if proj.players.get(n, None) and
                         proj.players[n].team == team}
            for absent_pid in team_pids:
                pair_impacts = self.minutes_model.get_absence_impact(team, absent_pid)
                if not pair_impacts:
                    continue
                absent_name = pid_to_name.get(absent_pid)
                if not absent_name:
                    continue
                beneficiary_deltas = {}
                for ben_pid, info in pair_impacts.items():
                    ben_name = pid_to_name.get(ben_pid)
                    if ben_name and info["delta_min"] != 0:
                        beneficiary_deltas[ben_name] = info["delta_min"]
                if beneficiary_deltas:
                    impacts[absent_name] = beneficiary_deltas
        return impacts

    def _project_single_player(
        self,
        player: PlayerProfile,
        opp: str,
        is_home: bool,
        pace_factor: float,
        is_b2b: bool = False,
        market_spread: float = 0.0,
        factor_config: dict | None = None,
    ) -> PlayerProjection:
        """Project a single player's stat line for a specific matchup.

        B2 rate-based projection: stat = projected_minutes × per_min_rate × matchup_adj.
        Matchup adjustments (pace, defense, form) modify the per-minute rate,
        not the counting stat.  Minutes come from B1 (minutes model).

        factor_config: optional dict to disable specific adjustments for ablation.
        """
        _fc = factor_config or {}
        pp = PlayerProjection(
            player_name=player.player_name,
            team=player.team,
            opponent=opp,
        )

        base_min = player.mpg
        min_proj = None

        # ── Minutes projection (B1) ──
        # factor_config.blowout=False → zero spread to disable blowout script
        _ms = market_spread if _fc.get("blowout", True) else 0.0
        if self.minutes_model and player.player_id:
            min_proj = self.minutes_model.project(
                player.player_id, opp, is_b2b=is_b2b,
                market_spread=_ms,
                disable_foul_adj=not _fc.get("foul_adj", True),
                disable_absence_adj=not _fc.get("absence_adj", True),
            )
            if min_proj:
                pp.minutes = round(min_proj.projected_minutes, 1)
            else:
                pp.minutes = round(base_min * (0.94 if is_b2b else 1.0), 1)
        elif is_b2b:
            if player.b2b_mpg > 0 and player.mpg > 0:
                pp.minutes = round(base_min * player.b2b_mpg / player.mpg, 1)
            else:
                pp.minutes = round(base_min * 0.94, 1)
        else:
            pp.minutes = round(base_min, 1)

        # ── Base per-minute rates (B2) ──
        _safe = lambda rate, fallback_pg: rate if rate > 0 else (fallback_pg / max(base_min, 1.0))
        pts_rate = _safe(player.pts_per_min, player.ppg)
        reb_rate = _safe(player.reb_per_min, player.rpg)
        ast_rate = _safe(player.ast_per_min, player.apg)
        stl_rate = _safe(player.stl_per_min, player.spg)
        blk_rate = _safe(player.blk_per_min, player.bpg)
        fg3m_rate = _safe(player.fg3m_per_min, player.tpg)

        # Absence rate deltas killed by ablation (+14.9% worse MAE,
        # game-clustered CI [+0.60, +0.71]).  Minutes redistribution (B1)
        # is the only surviving absence mechanism.

        # ── Matchup adjustments (applied to rates) ──

        # 1. Pace — more possessions per minute
        pace_adj = pace_factor
        pp.pace_adj = pace_adj

        # 2. Defense — opponent allows more/fewer per possession
        if _fc.get("defense", True):
            pts_def = self._defensive_factor(opp, "points")
            reb_def = self._defensive_factor(opp, "rebounds")
            ast_def = self._defensive_factor(opp, "assists")
            thr_def = self._defensive_factor(opp, "threes")
        else:
            pts_def = reb_def = ast_def = thr_def = 1.0
        pp.def_adj = pts_def

        # 3. Form (recent efficiency vs season baseline)
        # B4 verdict: DISABLED by default. Hurts starters (+0.0133 PTS,
        # +0.0312 PRA) — over-weights recent noise for stable-role players.
        if _fc.get("form", False):
            form = self._form_factor(player)
        else:
            form = 1.0
        pp.form_adj = form

        # 4. Rest — small efficiency loss on B2B (minutes handled by B1)
        rest = 0.985 if is_b2b else 1.0
        pp.rest_adj = rest

        # 5. Home/away
        ha = self._home_away_factor(player, is_home)

        # 6. Synergy play-type matchup (B4)
        if _fc.get("synergy", False):
            syn = self._synergy_factor(player.team, opp)
        else:
            syn = 1.0

        # 7. Position-specific defender matchup (B4)
        if _fc.get("matchup", False):
            mu = self._matchup_factor(player, opp)
        else:
            mu = 1.0

        # ── Adjusted per-minute rates (stored for redistribution) ──
        pp.adj_pts_rate = pts_rate * pace_adj * pts_def * form * rest * ha * syn * mu
        pp.adj_reb_rate = reb_rate * pace_adj * reb_def * form * rest * syn * mu
        pp.adj_ast_rate = ast_rate * pace_adj * ast_def * form * rest * syn * mu
        pp.adj_stl_rate = stl_rate * form * rest
        pp.adj_blk_rate = blk_rate * form * rest
        pp.adj_fg3m_rate = fg3m_rate * pace_adj * thr_def * form * rest * syn * mu

        # ── Counting stats = minutes × adjusted rate ──
        pp.points = round(pp.minutes * pp.adj_pts_rate, 1)
        pp.rebounds = round(pp.minutes * pp.adj_reb_rate, 1)
        pp.assists = round(pp.minutes * pp.adj_ast_rate, 1)
        pp.steals = round(pp.minutes * pp.adj_stl_rate, 1)
        pp.blocks = round(pp.minutes * pp.adj_blk_rate, 1)
        pp.threes = round(pp.minutes * pp.adj_fg3m_rate, 1)
        pp.pra = round(pp.points + pp.rebounds + pp.assists, 1)

        # ── Standard deviations ──
        # Variance decomposition: Var(stat) ≈ min² × rate_var + rate² × min_var
        min_std = base_min * 0.15  # default
        if self.minutes_model and player.player_id:
            ms = self.minutes_model.get_player_stats(player.player_id)
            if ms:
                min_std = ms.get("std_min", base_min * 0.15)

        def _std(rate_std_player: float, adj_rate: float, fallback_frac: float) -> float:
            rate_std = rate_std_player if rate_std_player > 0 else adj_rate * fallback_frac
            return sqrt(pp.minutes ** 2 * rate_std ** 2 + adj_rate ** 2 * min_std ** 2)

        pp.pts_std = _std(player.pts_rate_std, pp.adj_pts_rate, 0.25)
        pp.reb_std = _std(player.reb_rate_std, pp.adj_reb_rate, 0.30)
        pp.ast_std = _std(player.ast_rate_std, pp.adj_ast_rate, 0.35)
        pp.thr_std = _std(player.fg3m_rate_std, pp.adj_fg3m_rate, 0.45)

        # B3: PRA std via combined rate (captures pts/reb/ast covariance)
        adj_pra_rate = pp.adj_pts_rate + pp.adj_reb_rate + pp.adj_ast_rate
        pp.pra_std = _std(player.pra_rate_std, adj_pra_rate, 0.20)

        return pp

    # ── Public lookup helpers ────────────────────────────────────────────────

    def project_player(self, name: str, opp: str,
                       is_home: bool = True,
                       is_b2b: bool = False,
                       market_total: float = 0.0) -> Optional[PlayerProjection]:
        """Quick single-player projection lookup."""
        if not self._loaded:
            self.load()
        opp = normalize_abbr(opp)
        pp = self.player_profiles.get(name)
        if not pp:
            return None
        home = pp.team if is_home else opp
        away = opp if is_home else pp.team
        expected_poss = self._expected_possessions(home, away, market_total)
        poss_adj = self._player_pace_adj(pp.team, expected_poss)
        return self._project_single_player(pp, opp, is_home, poss_adj, is_b2b)

    def get_team_profile(self, abbr: str) -> Optional[TeamProfile]:
        if not self._loaded:
            self.load()
        return self.team_profiles.get(normalize_abbr(abbr))

    def get_player_profile(self, name: str) -> Optional[PlayerProfile]:
        if not self._loaded:
            self.load()
        return self.player_profiles.get(name)

    def get_traded_players(self) -> list[dict]:
        """Return list of detected trades."""
        return list(self.trades.values())

    def get_team_players(self, abbr: str) -> list[PlayerProfile]:
        """All players for a team, sorted by minutes."""
        if not self._loaded:
            self.load()
        return sorted(
            [p for p in self.player_profiles.values() if p.team == abbr],
            key=lambda p: p.mpg, reverse=True,
        )

    # ── Caching ──────────────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        return os.path.abspath(_CACHE_DB)

    @staticmethod
    def _to_native(obj):
        """Convert numpy types to native Python for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _json_safe_dict(self, d: dict) -> dict:
        return {k: self._to_native(v) for k, v in d.items()}

    def _save_to_cache(self):
        """Persist profiles to SQLite cache for fast reload."""
        path = self._cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.execute("DROP TABLE IF EXISTS team_profiles")
            conn.execute("DROP TABLE IF EXISTS player_profiles")
            conn.execute("DROP TABLE IF EXISTS trades")
            conn.execute("DROP TABLE IF EXISTS roster_teams")
            conn.execute("DROP TABLE IF EXISTS meta")

            conn.execute("""
                CREATE TABLE team_profiles (
                    team TEXT PRIMARY KEY, data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE player_profiles (
                    name TEXT PRIMARY KEY, data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE trades (
                    player_id INTEGER PRIMARY KEY, data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE roster_teams (
                    player_id INTEGER PRIMARY KEY, team TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY, value TEXT
                )
            """)

            for abbr, tp in self.team_profiles.items():
                conn.execute("INSERT INTO team_profiles VALUES (?, ?)",
                             (abbr, json.dumps(self._json_safe_dict(tp.__dict__))))
            for name, pp in self.player_profiles.items():
                conn.execute("INSERT INTO player_profiles VALUES (?, ?)",
                             (name, json.dumps(self._json_safe_dict(pp.__dict__))))
            for pid, td in self.trades.items():
                conn.execute("INSERT INTO trades VALUES (?, ?)",
                             (pid, json.dumps(td)))

            if hasattr(self, "_roster_teams") and self._roster_teams:
                for pid, team in self._roster_teams.items():
                    conn.execute("INSERT INTO roster_teams VALUES (?, ?)",
                                 (pid, team))

            conn.execute("INSERT INTO meta VALUES (?, ?)",
                         ("season", self.season))
            conn.execute("INSERT INTO meta VALUES (?, ?)",
                         ("window_pct", str(self.window_pct)))
            conn.execute("INSERT INTO meta VALUES (?, ?)",
                         ("cached_at", datetime.utcnow().isoformat()))
            conn.execute("INSERT INTO meta VALUES (?, ?)",
                         ("lg_avg", json.dumps(_LG_AVG)))
            conn.commit()
            logger.info("[NBA] Cache saved to %s", path)
        finally:
            conn.close()

    def _load_from_cache(self) -> bool:
        """Load profiles from cache if fresh (< 24 hours)."""
        global _LG_AVG
        path = self._cache_path()
        if not os.path.exists(path):
            return False

        try:
            conn = sqlite3.connect(path)
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())

            if meta.get("season") != self.season:
                conn.close()
                return False
            if float(meta.get("window_pct", 0)) != self.window_pct:
                conn.close()
                return False

            # Check freshness (24 hours)
            cached_at = meta.get("cached_at", "")
            if cached_at:
                age = (datetime.utcnow() - datetime.fromisoformat(cached_at)).total_seconds()
                if age > 86400:  # 24 hours
                    conn.close()
                    return False

            # Load team profiles
            for row in conn.execute("SELECT team, data FROM team_profiles").fetchall():
                d = json.loads(row[1])
                tp = TeamProfile(**{k: v for k, v in d.items()
                                    if k in TeamProfile.__dataclass_fields__})
                self.team_profiles[row[0]] = tp

            # Load player profiles
            for row in conn.execute("SELECT name, data FROM player_profiles").fetchall():
                d = json.loads(row[1])
                pp = PlayerProfile(**{k: v for k, v in d.items()
                                      if k in PlayerProfile.__dataclass_fields__})
                self.player_profiles[row[0]] = pp

            # Load trades
            for row in conn.execute("SELECT player_id, data FROM trades").fetchall():
                self.trades[int(row[0])] = json.loads(row[1])

            # Load roster teams (may not exist in older caches)
            try:
                self._roster_teams = {}
                for row in conn.execute("SELECT player_id, team FROM roster_teams").fetchall():
                    self._roster_teams[int(row[0])] = row[1]
                if self._roster_teams:
                    logger.info("[NBA] Loaded %d roster team assignments from cache",
                                len(self._roster_teams))
            except Exception:
                pass  # Table doesn't exist in older caches

            # Load league averages
            lg_str = meta.get("lg_avg", "{}")
            _LG_AVG.update(json.loads(lg_str))

            conn.close()
            logger.info("[NBA] Loaded from cache: %d teams, %d players",
                         len(self.team_profiles), len(self.player_profiles))
            return True

        except Exception as e:
            logger.warning("[NBA] Cache load failed: %s", e)
            return False

    def report_roster_changes(self) -> list[dict]:
        """Compare roster-based team assignments vs game-log-based.

        Returns list of players whose official roster team differs from
        the team that would be inferred from game logs alone (MATCHUP string
        or _detect_trades). Useful for catching offseason moves.
        """
        if not hasattr(self, "_roster_teams") or not self._roster_teams:
            return []

        changes = []
        for name, pp in self.player_profiles.items():
            pid = pp.player_id
            if not pid or pid not in self._roster_teams:
                continue

            roster_team = self._roster_teams[pid]
            if roster_team != pp.team:
                changes.append({
                    "player_name": name,
                    "player_id": pid,
                    "engine_team": pp.team,
                    "roster_team": roster_team,
                    "mpg": pp.mpg,
                    "traded": pid in self.trades,
                })

        changes.sort(key=lambda x: x["mpg"], reverse=True)
        if changes:
            logger.info("[NBA] Roster changes: %d players have different "
                        "roster team vs engine assignment", len(changes))
            for c in changes[:10]:
                logger.info("  %s: engine=%s → roster=%s (%.1f mpg, traded=%s)",
                            c["player_name"], c["engine_team"],
                            c["roster_team"], c["mpg"], c["traded"])
        return changes

    # ── Summary / diagnostics ────────────────────────────────────────────────

    def summary(self) -> dict:
        """Quick diagnostic summary of the engine state."""
        if not self._loaded:
            self.load()

        top_offenses = sorted(self.team_profiles.values(),
                              key=lambda t: t.off_ppg, reverse=True)[:5]
        top_defenses = sorted(self.team_profiles.values(),
                              key=lambda t: t.def_ppg)[:5]
        traded = self.get_traded_players()

        return {
            "season": self.season,
            "window_pct": self.window_pct,
            "teams": len(self.team_profiles),
            "players": len(self.player_profiles),
            "trades_detected": len(traded),
            "traded_players": [t["player_name"] for t in traded[:10]],
            "top_offenses": [(t.team, round(t.off_ppg, 1)) for t in top_offenses],
            "top_defenses": [(t.team, round(t.def_ppg, 1)) for t in top_defenses],
            "league_avg": _LG_AVG,
        }

    def validate_rate_projection(self, min_history: int = 10) -> dict:
        """Walk-forward ablation: rate-based (B2) vs counting-stat baseline.

        For each player-game (test set), computes both:
          - Baseline: PPG × (projected_min / avg_min)  (old approach)
          - B2: projected_min × pts_per_min             (rate approach)
        Compares MAE per stat per role.

        Requires raw player_logs (not available from cache load).
        """
        if not self.player_logs:
            return {"error": "No raw player_logs — load without cache"}

        _STATS = [("PTS", "ppg"), ("REB", "rpg"), ("AST", "apg"), ("FG3M", "tpg")]

        results = {
            stat_col: {"baseline": [], "rate": [], "role": []}
            for stat_col, _ in _STATS
        }

        for pid, df in self.player_logs.items():
            if len(df) < min_history + 3:
                continue

            mins = df["MIN"].astype(float).values
            name = df["PLAYER_NAME"].iloc[0] if "PLAYER_NAME" in df.columns else str(pid)

            for stat_col, profile_field in _STATS:
                stat_vals = df[stat_col].astype(float).values

                for i in range(min_history, len(df)):
                    train_mins = mins[:i]
                    train_stats = stat_vals[:i]
                    test_min = mins[i]
                    test_stat = stat_vals[i]

                    if test_min < 5:
                        continue  # skip garbage-time appearances

                    avg_min = float(mean(train_mins))
                    avg_stat = float(mean(train_stats))

                    # Role from training window
                    if avg_min >= 28:
                        role = "starter"
                    elif avg_min >= 15:
                        role = "rotation"
                    else:
                        role = "spot"

                    # ── Baseline (counting-stat): PPG × (test_min / avg_min) ──
                    # Approximates old approach: PPG scaled by minutes ratio
                    if avg_min > 0:
                        baseline_pred = avg_stat * (test_min / avg_min)
                    else:
                        baseline_pred = avg_stat

                    # ── B2 (rate-based): test_min × per_min_rate ──
                    # Compute per-minute rate from training games with >= 10 min
                    meaningful_mask = train_mins >= 10.0
                    if meaningful_mask.sum() >= 3:
                        rate_vals = train_stats[meaningful_mask] / train_mins[meaningful_mask]
                        rate = float(mean(rate_vals))
                    elif avg_min > 0:
                        rate = avg_stat / avg_min
                    else:
                        rate = 0.0
                    rate_pred = test_min * rate

                    results[stat_col]["baseline"].append(abs(test_stat - baseline_pred))
                    results[stat_col]["rate"].append(abs(test_stat - rate_pred))
                    results[stat_col]["role"].append(role)

        # Aggregate
        summary = {}
        for stat_col, _ in _STATS:
            r = results[stat_col]
            n = len(r["baseline"])
            if n == 0:
                continue

            baseline_maes = r["baseline"]
            rate_maes = r["rate"]
            roles = r["role"]

            overall_baseline = round(mean(baseline_maes), 3)
            overall_rate = round(mean(rate_maes), 3)
            delta = round(overall_rate - overall_baseline, 3)

            # By role
            by_role = {}
            for role in ("starter", "rotation", "spot"):
                idx = [j for j, rv in enumerate(roles) if rv == role]
                if len(idx) < 20:
                    continue
                bl = mean(baseline_maes[j] for j in idx)
                rt = mean(rate_maes[j] for j in idx)
                by_role[role] = {
                    "baseline_mae": round(bl, 3),
                    "rate_mae": round(rt, 3),
                    "delta": round(rt - bl, 3),
                    "pct_change": round((rt - bl) / bl * 100, 1) if bl > 0 else 0,
                    "n": len(idx),
                }

            summary[stat_col] = {
                "baseline_mae": overall_baseline,
                "rate_mae": overall_rate,
                "delta": delta,
                "pct_change": round(delta / overall_baseline * 100, 1) if overall_baseline > 0 else 0,
                "n": n,
                "by_role": by_role,
            }

        return summary
