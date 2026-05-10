"""
database.py — SQLAlchemy models + DB init
Run once on startup; safe to re-run (create_all is idempotent).
"""
from __future__ import annotations

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Boolean, Text, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
DATABASE_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    # Keep connections open for the process lifetime — avoids repeated file open/close
    pool_pre_ping=True,
)

# WAL mode: safer for concurrent readers/writers, checkpointed on close.
# synchronous=FULL: every commit is flushed to the OS before returning.
from sqlalchemy import event as _sa_event

@_sa_event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Tables ────────────────────────────────────────────────────────────────────

class Bet(Base):
    """Every bet — imported from Pikkit CSV or placed directly in the app."""
    __tablename__ = "bets"

    id            = Column(String, primary_key=True)          # Pikkit bet_id or generated UUID
    source        = Column(String, default="pikkit")          # "pikkit" | "app"
    sportsbook    = Column(String, default="FanDuel")
    bet_type      = Column(String)                            # "parlay" | "straight"
    status        = Column(String)                            # SETTLED_WIN / SETTLED_LOSS / PLACED / MOCK
    odds          = Column(Float)
    closing_line  = Column(Float, nullable=True)
    ev            = Column(Float, nullable=True)
    amount        = Column(Float)
    profit        = Column(Float, nullable=True)
    legs          = Column(Integer)
    sports        = Column(String, nullable=True)
    leagues       = Column(String, nullable=True)
    bet_info      = Column(Text, nullable=True)               # pipe-delimited leg descriptions
    tags          = Column(String, nullable=True)
    is_mock              = Column(Boolean, default=False)     # paper-trade flag
    time_placed          = Column(DateTime, nullable=True)
    time_settled         = Column(DateTime, nullable=True)
    created_at           = Column(DateTime, default=datetime.utcnow)
    # Phase 6A — promo tracking
    promo_type           = Column(String,  nullable=True, default="none")
    promo_boosted_odds   = Column(Integer, nullable=True)
    promo_ev_lift        = Column(Float,   nullable=True)
    promo_was_free_bet   = Column(Integer, nullable=True, default=0)
    # Phase 6B — cash-out tracking
    cash_out_offers_log       = Column(Text,    nullable=True)
    cashed_out                = Column(Integer, nullable=True, default=0)
    cash_out_amount           = Column(Float,   nullable=True)
    cash_out_timestamp        = Column(String,  nullable=True)
    cash_out_vs_final_decision = Column(String, nullable=True)
    cash_out_target           = Column(Float,   nullable=True)
    cash_out_target_pct       = Column(Float,   nullable=True)
    cash_out_target_rationale = Column(Text,    nullable=True)


class BetLeg(Base):
    """Individual legs parsed out of bet_info for feature engineering."""
    __tablename__ = "bet_legs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    bet_id        = Column(String)
    leg_index     = Column(Integer)
    description   = Column(Text)
    market_type   = Column(String, nullable=True)   # Moneyline / Spread / Total / Player prop
    team          = Column(String, nullable=True)
    opponent      = Column(String, nullable=True)
    sport         = Column(String, nullable=True)
    league        = Column(String, nullable=True)
    # Per-leg outcome (null = unknown; populated for manual/FanDuel imports only)
    # Pikkit CSV does not export per-leg results — only parlay-level status is available.
    leg_result        = Column(String, nullable=True)   # WIN / LOSS / PUSH / null
    odds_str          = Column(String, nullable=True)   # American odds string e.g. "-229", "+168" (null for Pikkit)
    # Retrospective resolver fields (populated by leg_resolver.py)
    accuracy_delta    = Column(Float,  nullable=True)   # margin of resolution (positive = won with cushion)
    resolution_source = Column(String, nullable=True)   # historical_db / pitcher_logs / inferred_parlay_win / unresolvable
    actual_value      = Column(Float,  nullable=True)   # actual score/stat used for resolution
    # Leg Quality Score (populated by leg_quality.py at bet time or retrospectively)
    lqs               = Column(Float,  nullable=True)   # 0–100 quality score
    lqs_grade         = Column(String, nullable=True)   # A / B / C / D
    # Scheduled start time for this leg's game (ISO datetime string, UTC)
    # Populated at placement time; used by live_monitor to anchor game_date lookup.
    game_commence_time = Column(String, nullable=True)
    # Soccer market sub-type: double_chance | btts | corners | player_prop | None
    subtype            = Column(String, nullable=True)


class Fixture(Base):
    """Upcoming games fetched from TheOddsAPI."""
    __tablename__ = "fixtures"

    id            = Column(String, primary_key=True)   # OddsAPI game id
    sport_key     = Column(String)
    sport_title   = Column(String)
    home_team     = Column(String)
    away_team     = Column(String)
    commence_time = Column(DateTime)
    bookmakers    = Column(JSON, nullable=True)         # raw odds payload
    fetched_at    = Column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    """Metadata for each ML training run."""
    __tablename__ = "model_runs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    run_at        = Column(DateTime, default=datetime.utcnow)
    algorithm     = Column(String)
    n_train       = Column(Integer)
    n_test        = Column(Integer)
    accuracy      = Column(Float)
    roc_auc       = Column(Float)
    feature_names = Column(JSON)
    notes         = Column(Text, nullable=True)


class Prediction(Base):
    """Model predictions for real or mock bets."""
    __tablename__ = "predictions"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    bet_id        = Column(String, nullable=True)     # NULL for hypothetical bets
    model_run_id  = Column(Integer, nullable=True)
    win_prob      = Column(Float)
    expected_value= Column(Float)
    predicted_at  = Column(DateTime, default=datetime.utcnow)
    features_used = Column(JSON, nullable=True)


class SettleLog(Base):
    """Audit trail for every auto-settle attempt."""
    __tablename__ = "settle_log"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    bet_id        = Column(String)
    sport_key     = Column(String, nullable=True)
    result        = Column(String)           # WIN / LOSS / PUSH / SKIP / ERROR
    profit        = Column(Float, nullable=True)
    method        = Column(String)           # "auto" | "manual"
    raw_score     = Column(JSON, nullable=True)   # raw OddsAPI score payload
    notes         = Column(Text, nullable=True)
    settled_at    = Column(DateTime, default=datetime.utcnow)
    retrained     = Column(Boolean, default=False)


class SchedulerRun(Base):
    """Log of each scheduler cycle (settle + retrain)."""
    __tablename__ = "scheduler_runs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_at         = Column(DateTime, default=datetime.utcnow)
    bets_checked   = Column(Integer, default=0)
    bets_settled   = Column(Integer, default=0)
    retrained      = Column(Boolean, default=False)
    new_accuracy   = Column(Float, nullable=True)
    errors         = Column(Text, nullable=True)
    duration_secs  = Column(Float, nullable=True)


def _migrate_promo_columns():
    """Add Phase 6A promo columns to bets table if they don't exist."""
    migrations = [
        "ALTER TABLE bets ADD COLUMN promo_type TEXT DEFAULT 'none'",
        "ALTER TABLE bets ADD COLUMN promo_boosted_odds INTEGER DEFAULT NULL",
        "ALTER TABLE bets ADD COLUMN promo_ev_lift REAL DEFAULT NULL",
        "ALTER TABLE bets ADD COLUMN promo_was_free_bet INTEGER DEFAULT 0",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_cash_out_columns():
    """Add Phase 6B cash-out tracking columns to bets table if they don't exist."""
    migrations = [
        # JSON array of {timestamp, offer_amount, recommendation, legs_status}
        "ALTER TABLE bets ADD COLUMN cash_out_offers_log TEXT DEFAULT NULL",
        # 1 = user accepted a cash out offer; 0 = held (default)
        "ALTER TABLE bets ADD COLUMN cashed_out INTEGER DEFAULT 0",
        # Dollar amount received when cashing out
        "ALTER TABLE bets ADD COLUMN cash_out_amount REAL DEFAULT NULL",
        # ISO timestamp of cash-out acceptance
        "ALTER TABLE bets ADD COLUMN cash_out_timestamp TEXT DEFAULT NULL",
        # Retrospective verdict (set by auto_settle after game outcomes resolve):
        # "would_have_won"  → bet would have won if held  (cashed unnecessarily)
        # "would_have_lost" → bet would have lost if held (cashed wisely)
        # "cashed_wisely"   → alias for would_have_lost
        # "cashed_unnecessarily" → alias for would_have_won
        "ALTER TABLE bets ADD COLUMN cash_out_vs_final_decision TEXT DEFAULT NULL",
        # Placement-time cash-out target fields
        "ALTER TABLE bets ADD COLUMN cash_out_target REAL DEFAULT NULL",
        "ALTER TABLE bets ADD COLUMN cash_out_target_pct REAL DEFAULT NULL",
        "ALTER TABLE bets ADD COLUMN cash_out_target_rationale TEXT DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_bet_leg_result():
    """Add leg_result, odds_str, and resolver columns to bet_legs if they don't exist."""
    migrations = [
        "ALTER TABLE bet_legs ADD COLUMN leg_result TEXT DEFAULT NULL",
        "ALTER TABLE bet_legs ADD COLUMN odds_str TEXT DEFAULT NULL",
        "ALTER TABLE bet_legs ADD COLUMN accuracy_delta REAL DEFAULT NULL",
        "ALTER TABLE bet_legs ADD COLUMN resolution_source TEXT DEFAULT NULL",
        "ALTER TABLE bet_legs ADD COLUMN actual_value REAL DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_mock_bet_columns():
    """Add source and weight columns to mock_bets if they don't exist."""
    migrations = [
        "ALTER TABLE mock_bets ADD COLUMN source TEXT DEFAULT 'prospective'",
        "ALTER TABLE mock_bets ADD COLUMN weight REAL DEFAULT 1.0",
        "ALTER TABLE mock_bets ADD COLUMN game_date TEXT DEFAULT NULL",  # for retroactive bets
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


class MockBet(Base):
    """Auto-generated paper bets for model validation (System 3)."""
    __tablename__ = "mock_bets"

    id                  = Column(String, primary_key=True)
    generation_run_id   = Column(String, nullable=True)    # batch ID (date + run index)
    generated_at        = Column(DateTime, default=datetime.utcnow)
    sport               = Column(String, nullable=True)
    bet_type            = Column(String, default="parlay")
    odds                = Column(Float)
    amount              = Column(Float, default=10.0)      # paper stake
    legs                = Column(Integer, default=1)
    bet_info            = Column(Text, nullable=True)
    status              = Column(String, default="PENDING") # PENDING / SETTLED_WIN / SETTLED_LOSS / EXPIRED
    predicted_win_prob  = Column(Float, nullable=True)
    predicted_ev        = Column(Float, nullable=True)
    confidence          = Column(String, nullable=True)    # HIGH / MEDIUM / LOW
    model_confidence_avg = Column(Float, nullable=True)   # avg per-leg win prob (0-1)
    model_used          = Column(String, nullable=True)
    model_auc           = Column(Float, nullable=True)
    settled_at          = Column(DateTime, nullable=True)
    actual_profit       = Column(Float, nullable=True)
    notes               = Column(Text, nullable=True)
    # System 3F — retroactive backfill fields
    source              = Column(String, nullable=True, default="prospective")  # prospective | retroactive_mock
    weight              = Column(Float, nullable=True, default=1.0)             # training signal weight
    game_date           = Column(String, nullable=True)                         # YYYY-MM-DD for retroactive bets
    # LQS tracking — average Leg Quality Score across legs at generation time
    avg_lqs             = Column(Float, nullable=True)
    # Predicted odds at generation time (American format, e.g. +350)
    predicted_odds      = Column(Integer, nullable=True)
    # Boost simulation tracking (Phase 7A)
    promo_type          = Column(String,  nullable=True)  # PROFIT_BOOST / BONUS_BET / NO_SWEAT / None
    promo_boost_pct     = Column(Float,   nullable=True)  # 0.25 / 0.30 / 0.50 / None
    boost_strategy      = Column(String,  nullable=True)  # eligibility route/reason string
    promo_ev_lift       = Column(Float,   nullable=True)  # EV gain in $ from the boost
    promo_boosted_odds  = Column(Float,   nullable=True)  # post-boost decimal odds (PROFIT_BOOST only)
    # User curation
    user_excluded           = Column(Boolean,  nullable=True, default=False)
    user_excluded_reason    = Column(String,   nullable=True)
    user_excluded_at        = Column(DateTime, nullable=True)
    user_excluded_thesis_id = Column(Integer,  nullable=True)
    # Multi-leg curation state
    has_excluded_legs                   = Column(Boolean,  nullable=True, default=False)
    exclusion_mode_summary              = Column(String,   nullable=True)
    recalculated_odds_decimal           = Column(Float,    nullable=True)
    recalculated_combined_odds_american = Column(Integer,  nullable=True)
    recalculated_actual_profit          = Column(Float,    nullable=True)
    counterfactual_message              = Column(Text,     nullable=True)


class MockBetLeg(Base):
    """Individual legs of a MockBet (System 3)."""
    __tablename__ = "mock_bet_legs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    mock_bet_id  = Column(String)
    leg_index    = Column(Integer)
    description  = Column(Text)
    market_type  = Column(String, nullable=True)
    sport        = Column(String, nullable=True)
    win_prob     = Column(Float, nullable=True)
    ev           = Column(Float, nullable=True)
    grade        = Column(String, nullable=True)
    model_used   = Column(String, nullable=True)
    fixture_id   = Column(String, nullable=True)    # OddsAPI fixture id for settlement lookup
    is_alt_line  = Column(Boolean, nullable=True, default=False)  # True if alt spread/total

    # Predicted signals at generation time — written alongside the leg
    predicted_win_prob       = Column(Float,   nullable=True)   # model win prob for this leg (0–100)
    predicted_edge_pp        = Column(Float,   nullable=True)   # edge in percentage points vs implied

    # CLV (Closing Line Value) tracking
    open_odds                = Column(Integer, nullable=True)   # American odds at bet generation time
    close_odds               = Column(Integer, nullable=True)   # American odds at game start (latest snapshot)
    clv_cents                = Column(Integer, nullable=True)   # close_odds - open_odds (positive = beat closing line)
    clv_available            = Column(Integer, nullable=True)   # 1 if close_odds found, 0 if not

    # ALE (Alternative Line Evaluation) metadata — written at pick generation time
    ale_considered           = Column(Boolean, nullable=True, default=False)  # True if ALE ran for this fixture
    ale_naive_pick           = Column(Text,    nullable=True)   # description of what default sort would have chosen
    ale_switched             = Column(Boolean, nullable=True, default=False)  # True if ALE changed the selection
    ale_los_improvement      = Column(Float,   nullable=True)   # LOS gain from switching (0 if not switched)

    # Settlement audit — written when the leg is resolved
    leg_result               = Column(String,  nullable=True)   # WIN / LOSS / PUSH
    resolved_home_team       = Column(String,  nullable=True)
    resolved_away_team       = Column(String,  nullable=True)
    resolved_home_score      = Column(Float,   nullable=True)
    resolved_away_score      = Column(Float,   nullable=True)
    resolved_margin          = Column(Float,   nullable=True)   # team_score - opp_score
    resolved_adjusted_margin = Column(Float,   nullable=True)   # margin + line (spread); diff from line (total)
    accuracy_delta           = Column(Float,   nullable=True)   # same as adjusted_margin

    # Line quality — written at settlement time (two-dimensional evaluation)
    main_market_line         = Column(Float,   nullable=True)   # standard market line (e.g. 8.5 total, -1.5 spread)
    main_market_result       = Column(String,  nullable=True)   # OVER|UNDER|COVERED|NOT_COVERED
    direction_correct        = Column(Integer, nullable=True)   # 1 if pick direction matched main market result
    optimal_line             = Column(Float,   nullable=True)   # best line that would have won
    line_delta               = Column(Float,   nullable=True)   # our_line - optimal_line (+ve=too aggressive)
    ab_alt_line              = Column(Float,   nullable=True)   # one step closer to main market
    ab_alt_result            = Column(String,  nullable=True)   # WIN|LOSS for the A/B line
    ab_alt_odds              = Column(Float,   nullable=True)   # decimal odds for A/B line
    ab_alt_ev                = Column(Float,   nullable=True)   # EV of A/B pick

    # Qualification tier — assigned at generation time based on market grade + win_prob
    # Tier 1: CUSHION + wp>=65% (highest confidence)
    # Tier 2: CUSHION + wp>=55%
    # Tier 3: CLOSE   + wp>=65% (model overrides market grade)
    # Tier 4: MIXED   + wp>=70% (very high model confidence only)
    qualification_tier       = Column(String,   nullable=True)

    # User curation — leg-level exclusion
    user_excluded            = Column(Boolean,  nullable=True, default=False)
    user_excluded_reason     = Column(String,   nullable=True)
    user_excluded_at         = Column(DateTime, nullable=True)
    user_excluded_thesis_id  = Column(Integer,  nullable=True)
    exclusion_mode           = Column(String,   nullable=True)  # null_bet | recalculate | counterfactual


class UserThesis(Base):
    """User-created handicapping theses that filter/exclude picks automatically."""
    __tablename__ = "user_theses"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    thesis_type          = Column(String,  nullable=False)   # matchup|team_state|player_state|systematic
    title                = Column(String,  nullable=False)
    description          = Column(Text,    nullable=True)
    sport                = Column(String,  nullable=True)
    team                 = Column(String,  nullable=True)
    opponent             = Column(String,  nullable=True)
    player               = Column(String,  nullable=True)
    market_filters       = Column(Text,    nullable=True)    # JSON: {block:[...], alt_spreads_min_line:25}
    active               = Column(Boolean, nullable=False, default=True)
    created_at           = Column(DateTime, default=datetime.utcnow)
    expires_at           = Column(DateTime, nullable=True)
    expire_after_games   = Column(Integer,  nullable=True)
    games_filtered_count = Column(Integer,  nullable=False, default=0)
    total_excluded_bets     = Column(Integer,  nullable=False, default=0)
    excluded_pnl_avoided    = Column(Float,    nullable=False, default=0.0)
    excluded_pnl_missed     = Column(Float,    nullable=False, default=0.0)
    net_value               = Column(Float,    nullable=False, default=0.0)
    # Per-mode accountability (added with exclusion_mode feature)
    total_excluded_legs     = Column(Integer,  nullable=False, default=0)
    total_recalculated_bets = Column(Integer,  nullable=False, default=0)
    pnl_avoided_null        = Column(Float,    nullable=False, default=0.0)
    pnl_avoided_recalc      = Column(Float,    nullable=False, default=0.0)
    pnl_missed_null         = Column(Float,    nullable=False, default=0.0)
    pnl_missed_recalc       = Column(Float,    nullable=False, default=0.0)
    reviewed_at             = Column(DateTime, nullable=True)
    next_review_at          = Column(DateTime, nullable=True)


class MarketRegimeLog(Base):
    """
    Daily market regime classification — written by scheduler at 8:30 AM CT.
    Read-only from picks logic perspective; used by signal_analysis.py for
    regime performance correlation.
    """
    __tablename__ = "market_regime_log"

    date                       = Column(String,  primary_key=True)  # YYYY-MM-DD
    regime                     = Column(String,  nullable=True)      # sharp/efficient/low_signal/mixed/sparse
    legs_evaluated             = Column(Integer, nullable=True)
    legs_positive_ev           = Column(Integer, nullable=True)
    pos_ev_pct                 = Column(Float,   nullable=True)
    hq_legs                    = Column(Integer, nullable=True)
    weighted_model_confidence  = Column(Float,   nullable=True)
    avg_implied_prob           = Column(Float,   nullable=True)
    model_breakdown            = Column(Text,    nullable=True)      # JSON
    suggested_weights          = Column(Text,    nullable=True)      # JSON
    actual_weights             = Column(Text,    nullable=True)      # JSON (always 50/30/20 now)
    mock_bets_generated        = Column(Integer, nullable=True)
    mock_win_rate              = Column(Float,   nullable=True)      # populated after settlement
    mock_pnl                   = Column(Float,   nullable=True)      # populated after settlement
    clv_avg                    = Column(Float,   nullable=True)      # populated after close odds available
    note                       = Column(Text,    nullable=True)
    created_at                 = Column(String,  nullable=True)


class SoccerResult(Base):
    """
    Cached API-Football v3 match results.
    One row per completed fixture.  Populated by soccer_data.store_soccer_results().
    """
    __tablename__ = "soccer_results"

    fixture_id    = Column(Integer, primary_key=True)   # API-Football fixture id
    date          = Column(String,  nullable=False)      # YYYY-MM-DD
    league_id     = Column(Integer, nullable=True)
    league_name   = Column(String,  nullable=True)
    home_team     = Column(String,  nullable=False)
    away_team     = Column(String,  nullable=False)
    home_goals    = Column(Integer, nullable=True)
    away_goals    = Column(Integer, nullable=True)
    home_corners  = Column(Integer, nullable=True)
    away_corners  = Column(Integer, nullable=True)
    fetched_at    = Column(DateTime, default=datetime.utcnow)


class LegQualityProfile(Base):
    """
    Aggregated accuracy-delta statistics per (market_type, sport) profile.
    Rebuilt by leg_quality.update_quality_profiles() after every resolve-legs run.
    """
    __tablename__ = "leg_quality_profiles"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    market_type       = Column(String,  nullable=False)
    sport             = Column(String,  nullable=True)
    team_or_player    = Column(String,  nullable=True)    # NULL = aggregate profile
    mean_delta        = Column(Float,   nullable=True)
    std_delta         = Column(Float,   nullable=True)
    p25_delta         = Column(Float,   nullable=True)
    p75_delta         = Column(Float,   nullable=True)
    win_rate          = Column(Float,   nullable=True)    # all resolved legs (incl. parlay-win infer)
    unbiased_win_rate = Column(Float,   nullable=True)    # excl. inferred_parlay_win legs
    close_loss_rate   = Column(Float,   nullable=True)    # losses with delta > -1.0
    bad_loss_rate     = Column(Float,   nullable=True)    # losses with delta < -3.0
    consistency_score = Column(Float,   nullable=True)    # 1 - std / max(|mean|,1)
    sample_size       = Column(Integer, nullable=True)
    unbiased_sample   = Column(Integer, nullable=True)    # legs excl. inferred_parlay_win
    last_updated      = Column(String,  nullable=True)    # ISO timestamp string


class LqsCalibrationLog(Base):
    """
    Weekly self-calibration log. Written after each FanDuel sync.
    Records LQS correlation, component weight drift, and whether re-tuning is recommended.
    """
    __tablename__ = "lqs_calibration_log"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_date         = Column(String,  nullable=False)      # ISO date
    n_unbiased       = Column(Integer, nullable=True)       # legs used (excl. parlay-win infer)
    current_weights  = Column(Text,    nullable=True)       # JSON: {"A":0.40,"B":0.30,...}
    optimal_weights  = Column(Text,    nullable=True)       # JSON: best-fit from data
    correlation      = Column(Float,   nullable=True)       # Pearson LQS–win corr
    drift_detected   = Column(Boolean, nullable=True)       # any weight diff > 5pp
    note             = Column(Text,    nullable=True)       # human-readable summary


class LegHistoricalResolution(Base):
    """
    Cache of individually-resolved leg outcomes from historical.db game scores.

    Used as input to Component A scoring — replaces inferred_parlay_win attribution.
    Each row is one leg resolved to WIN/LOSS against an actual game score.
    The resolution_source distinguishes legs resolved from mock_bet settlement vs.
    historical.db backfill.

    Queried by compute_component_a() for recency-weighted cover rate.
    """
    __tablename__ = "leg_historical_resolution"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    bet_leg_id        = Column(String,  nullable=True)   # mock_bet_legs.id or bet_legs.id
    game_date         = Column(String,  nullable=False)  # YYYY-MM-DD
    sport             = Column(String,  nullable=False)  # MLB / NHL / NBA / NFL / Soccer
    team              = Column(String,  nullable=False)  # canonical team name
    market_type       = Column(String,  nullable=False)  # Moneyline / Spread / Total
    line              = Column(Float,   nullable=True)   # numeric line (spread or total)
    result            = Column(String,  nullable=True)   # WIN / LOSS / PUSH
    margin            = Column(Float,   nullable=True)   # covered/missed by (accuracy_delta)
    resolution_source = Column(String,  nullable=False, default="historical_db")
    resolved_at       = Column(String,  nullable=True)   # ISO timestamp


class UserPick(Base):
    """
    User-submitted picks for A/B test comparison against the model.
    One row per parlay or straight bet the user claims they would/did place.
    """
    __tablename__ = "user_picks"

    id              = Column(String,  primary_key=True)      # UUID
    submitted_at    = Column(DateTime, default=datetime.utcnow)
    game_date       = Column(String,  nullable=False)        # YYYY-MM-DD CT (date of games)
    bet_type        = Column(String,  default="parlay")      # parlay | straight
    stake           = Column(Float,   default=10.0)          # paper stake ($)
    legs            = Column(Integer, default=1)
    combined_odds   = Column(Float,   nullable=True)         # decimal odds (computed from legs)
    notes           = Column(Text,    nullable=True)
    reasoning       = Column(Text,    nullable=True)         # separate reasoning field (editable post-submit)
    # Promo / boost (editable while PENDING)
    promo_type          = Column(String,  nullable=True)     # PROFIT_BOOST | BONUS_BET | NO_SWEAT | None
    promo_boost_pct     = Column(Float,   nullable=True)     # 0.25 | 0.30 | 0.50
    promo_boosted_odds  = Column(Float,   nullable=True)     # post-boost decimal odds
    potential_profit    = Column(Float,   nullable=True)     # stake * (promo_boosted_odds - 1)
    # Settlement
    status          = Column(String,  default="PENDING")     # PENDING | SETTLED_WIN | SETTLED_LOSS | VOID | CASHED_OUT
    actual_profit   = Column(Float,   nullable=True)         # profit after settlement
    settled_at      = Column(DateTime, nullable=True)
    # v2 LLM feature extraction
    llm_features_extracted      = Column(Integer, nullable=True, default=0)   # 0=not run, 1=done
    reasoning_complexity_score  = Column(Float,   nullable=True)              # 0.0–1.0 from LLM
    # Screenshot ingestion fields (Phase 9)
    bet_placed_at       = Column(DateTime, nullable=True)   # when placed at FanDuel (from screenshot)
    fanduel_bet_id      = Column(String,   nullable=True)   # FanDuel bet ID (optional, not unique — null for non-FD picks)
    added_retroactively = Column(Boolean,  nullable=True, default=False)
    ingestion_source    = Column(String,   nullable=True, default="manual")  # manual | screenshot_pregame | screenshot_retroactive | text_paste
    cashed_out_value    = Column(Float,    nullable=True)   # actual cash-out amount if CASHED_OUT
    potential_payout    = Column(Float,    nullable=True)   # total payout (stake + profit) from screenshot


class UserPickLeg(Base):
    """Individual legs of a UserPick."""
    __tablename__ = "user_pick_legs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_pick_id    = Column(String,  nullable=False)        # FK → user_picks.id
    leg_index       = Column(Integer, nullable=False)
    description     = Column(Text,    nullable=False)        # human-readable (e.g. "Lakers -4.5")
    sport           = Column(String,  nullable=True)
    market_type     = Column(String,  nullable=True)         # Moneyline | Spread | Total | Prop
    team            = Column(String,  nullable=True)
    odds_american   = Column(Integer, nullable=True)         # e.g. -110, +185
    odds_decimal    = Column(Float,   nullable=True)         # computed from odds_american
    # Settlement
    leg_result      = Column(String,  nullable=True)         # WIN | LOSS | PUSH
    # Optional link to a model MockBet for overlap analysis
    mock_bet_id     = Column(String,  nullable=True)
    # Screenshot ingestion fields (Phase 9)
    is_part_of_sgp  = Column(Boolean, nullable=True, default=False)
    price_source    = Column(String,  nullable=True, default="manual")  # manual | screenshot_visible | alt_lines_lookup | unresolved
    price_confidence = Column(Float,  nullable=True)   # 0.0–1.0
    # Phase 10 — per-leg failure analysis
    point                  = Column(Float,  nullable=True)   # spread/total/prop line (e.g. -1.5, 209.5, 20)
    player                 = Column(String, nullable=True)   # player name for props (separate from team)
    actual_outcome_value   = Column(Float,  nullable=True)   # actual stat/score/margin
    expected_outcome_value = Column(Float,  nullable=True)   # threshold needed to win
    miss_margin            = Column(Float,  nullable=True)   # actual_outcome_value - expected_outcome_value
    outcome_source         = Column(String, nullable=True)   # auto_settlement | user_provided | unknown


class UserPickSignal(Base):
    """
    Extracted signals from a UserPick or its legs.
    Each row encodes one interpretable pattern unit (team, market type, boost, etc.)
    that can be tracked across many picks to identify EDGE_POSITIVE / EDGE_NEGATIVE patterns.

    feature_source distinguishes v1 keyword extractor from v2 LLM extractor:
      'keyword' — regex/keyword matching (user_signal_learning.py)
      'llm'     — Claude Haiku structured extraction (llm_signal_extractor.py)
    """
    __tablename__ = "user_pick_signals"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    user_pick_id       = Column(String,  nullable=False)        # FK → user_picks.id
    user_pick_leg_id   = Column(Integer, nullable=True)         # FK → user_pick_legs.id (NULL = pick-level signal)
    signal_type        = Column(String,  nullable=False)        # team_preference | pitcher_command_aware | causal_reasoner | …
    signal_key         = Column(String,  nullable=True)         # e.g. "team:Phillies" | "pitcher_factors:command_issues"
    signal_value       = Column(Float,   nullable=True)         # numeric signal value (optional)
    reasoning_excerpt  = Column(Text,    nullable=True)         # key excerpt from pick notes relevant to this signal
    outcome            = Column(String,  nullable=True)         # WON | LOST | PENDING
    # v2 LLM extractor fields
    feature_source     = Column(String,  nullable=True, default="keyword")   # 'keyword' | 'llm'
    feature_complexity = Column(Float,   nullable=True)                       # LLM complexity score 0.0–1.0
    extracted_at       = Column(DateTime, default=datetime.utcnow)


class UserSignalPerformance(Base):
    """
    Aggregated performance statistics per (signal_type, signal_key).
    Updated after each pick settles. Drives bidirectional model learning:
    EDGE_POSITIVE patterns are adopted, EDGE_NEGATIVE patterns are avoided.
    """
    __tablename__ = "user_signal_performance"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    signal_type        = Column(String,  nullable=False)
    signal_key         = Column(String,  nullable=False)
    total_uses         = Column(Integer, default=0)
    wins               = Column(Integer, default=0)
    losses             = Column(Integer, default=0)
    wr_pct             = Column(Float,   nullable=True)
    # Pattern classification
    # 'EDGE_POSITIVE'     (>= 60% WR, n>=10) → adopt in model
    # 'EDGE_NEGATIVE'     (<= 40% WR, n>=10) → avoid in model
    # 'NEUTRAL'           (40-60% WR or n<10) → no action yet
    # 'INSUFFICIENT_DATA' (n<5)               → keep watching
    pattern_class      = Column(String,  nullable=True, default="INSUFFICIENT_DATA")
    # 0.5 (strong avoid) → 1.0 (neutral) → 1.5 (strong adopt)
    confidence_weight  = Column(Float,   nullable=True, default=1.0)
    last_updated       = Column(DateTime, nullable=True)


class RegimeAbLog(Base):
    """
    Daily A/B comparison: standard composite weights vs regime-suggested weights.

    Standard picks are what actually run in production (and get mock_bets created).
    Regime picks are shadow-run only — never served to the UI, never bet.

    Both sets are settled after the fact so win_rate columns get filled in
    by the settlement attribution step.
    """
    __tablename__ = "regime_ab_log"

    date              = Column(String,  primary_key=True)  # YYYY-MM-DD CT
    regime            = Column(String,  nullable=True)     # sharp / efficient / low_signal / mixed

    # Standard weights (what ran in production that day)
    standard_weights  = Column(Text,    nullable=True)     # JSON {win_prob, ev, lqs}
    standard_pick_ids = Column(Text,    nullable=True)     # JSON array of mock_bet ids
    standard_win_rate = Column(Float,   nullable=True)     # filled after settlement
    standard_pnl      = Column(Float,   nullable=True)
    standard_n        = Column(Integer, nullable=True)

    # Regime weights (shadow — never served)
    regime_weights    = Column(Text,    nullable=True)     # JSON suggested weights
    regime_pick_ids   = Column(Text,    nullable=True)     # JSON array (shadow pick ids/sigs)
    regime_win_rate   = Column(Float,   nullable=True)     # filled after settlement
    regime_pnl        = Column(Float,   nullable=True)
    regime_n          = Column(Integer, nullable=True)

    # Overlap between the two pick sets
    overlap_n         = Column(Integer, nullable=True)
    only_standard_n   = Column(Integer, nullable=True)
    only_regime_n     = Column(Integer, nullable=True)

    created_at        = Column(String,  nullable=True)


def _migrate_game_commence_time():
    """Add game_commence_time column to bet_legs if it doesn't exist."""
    with engine.connect() as conn:
        try:
            conn.execute(__import__("sqlalchemy").text(
                "ALTER TABLE bet_legs ADD COLUMN game_commence_time TEXT DEFAULT NULL"
            ))
            conn.commit()
        except Exception:
            pass  # column already exists


def _migrate_mock_bet_leg_resolution():
    """Add settlement audit columns to mock_bet_legs."""
    migrations = [
        "ALTER TABLE mock_bet_legs ADD COLUMN leg_result TEXT DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN resolved_home_team TEXT DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN resolved_away_team TEXT DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN resolved_home_score REAL DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN resolved_away_score REAL DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN resolved_margin REAL DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN resolved_adjusted_margin REAL DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN accuracy_delta REAL DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_leg_quality_columns():
    """Add lqs, lqs_grade to bet_legs; avg_lqs to mock_bets; is_alt_line to mock_bet_legs."""
    migrations = [
        "ALTER TABLE bet_legs      ADD COLUMN lqs         REAL    DEFAULT NULL",
        "ALTER TABLE bet_legs      ADD COLUMN lqs_grade   TEXT    DEFAULT NULL",
        "ALTER TABLE mock_bets     ADD COLUMN avg_lqs     REAL    DEFAULT NULL",
        "ALTER TABLE mock_bet_legs      ADD COLUMN is_alt_line       BOOLEAN DEFAULT FALSE",
        "ALTER TABLE leg_quality_profiles ADD COLUMN unbiased_win_rate REAL DEFAULT NULL",
        "ALTER TABLE leg_quality_profiles ADD COLUMN unbiased_sample   INTEGER DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_bet_leg_subtype():
    """Add subtype column to bet_legs for soccer market sub-type tracking."""
    with engine.connect() as conn:
        try:
            conn.execute(__import__("sqlalchemy").text(
                "ALTER TABLE bet_legs ADD COLUMN subtype TEXT DEFAULT NULL"
            ))
            conn.commit()
        except Exception:
            pass  # column already exists


def _migrate_clv_columns():
    """Add CLV tracking columns to mock_bet_legs."""
    migrations = [
        "ALTER TABLE mock_bet_legs ADD COLUMN open_odds   INTEGER DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN close_odds  INTEGER DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN clv_cents   INTEGER DEFAULT NULL",
        "ALTER TABLE mock_bet_legs ADD COLUMN clv_available INTEGER DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass


def _migrate_boost_columns():
    """Add promo_type and promo_boost_pct to mock_bets for boost simulation (Phase 7A)."""
    migrations = [
        "ALTER TABLE mock_bets ADD COLUMN promo_type     TEXT  DEFAULT NULL",
        "ALTER TABLE mock_bets ADD COLUMN promo_boost_pct REAL DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_user_curation_columns():
    """Add user curation columns to mock_bets if they don't exist."""
    migrations = [
        "ALTER TABLE mock_bets ADD COLUMN user_excluded BOOLEAN DEFAULT 0",
        "ALTER TABLE mock_bets ADD COLUMN user_excluded_reason TEXT",
        "ALTER TABLE mock_bets ADD COLUMN user_excluded_at DATETIME",
        "ALTER TABLE mock_bets ADD COLUMN user_excluded_thesis_id INTEGER",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_exclusion_mode_columns():
    """Add exclusion mode columns: leg-level curation, recalculated odds, counterfactual."""
    migrations = [
        # mock_bet_legs — leg-level exclusion fields
        "ALTER TABLE mock_bet_legs ADD COLUMN user_excluded BOOLEAN DEFAULT 0",
        "ALTER TABLE mock_bet_legs ADD COLUMN user_excluded_reason TEXT",
        "ALTER TABLE mock_bet_legs ADD COLUMN user_excluded_at DATETIME",
        "ALTER TABLE mock_bet_legs ADD COLUMN user_excluded_thesis_id INTEGER",
        "ALTER TABLE mock_bet_legs ADD COLUMN exclusion_mode TEXT",
        # mock_bets — multi-leg curation summary fields
        "ALTER TABLE mock_bets ADD COLUMN has_excluded_legs BOOLEAN DEFAULT 0",
        "ALTER TABLE mock_bets ADD COLUMN exclusion_mode_summary TEXT",
        "ALTER TABLE mock_bets ADD COLUMN recalculated_odds_decimal REAL",
        "ALTER TABLE mock_bets ADD COLUMN recalculated_combined_odds_american INTEGER",
        "ALTER TABLE mock_bets ADD COLUMN recalculated_actual_profit REAL",
        "ALTER TABLE mock_bets ADD COLUMN counterfactual_message TEXT",
        # user_theses — per-mode accountability
        "ALTER TABLE user_theses ADD COLUMN total_excluded_legs INTEGER DEFAULT 0",
        "ALTER TABLE user_theses ADD COLUMN total_recalculated_bets INTEGER DEFAULT 0",
        "ALTER TABLE user_theses ADD COLUMN pnl_avoided_null REAL DEFAULT 0.0",
        "ALTER TABLE user_theses ADD COLUMN pnl_avoided_recalc REAL DEFAULT 0.0",
        "ALTER TABLE user_theses ADD COLUMN pnl_missed_null REAL DEFAULT 0.0",
        "ALTER TABLE user_theses ADD COLUMN pnl_missed_recalc REAL DEFAULT 0.0",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_user_picks_tables():
    """Create user_picks, user_pick_legs, user_pick_signals, user_signal_performance tables."""
    # Base.metadata.create_all is idempotent — safe to call on existing DBs.
    Base.metadata.create_all(bind=engine)


def _migrate_llm_signal_columns():
    """Add v2 LLM extractor columns to user_pick_signals and user_picks."""
    migrations = [
        # user_pick_signals — v2 extractor source tag + complexity
        "ALTER TABLE user_pick_signals ADD COLUMN feature_source TEXT DEFAULT 'keyword'",
        "ALTER TABLE user_pick_signals ADD COLUMN feature_complexity REAL DEFAULT NULL",
        # user_picks — LLM extraction status + reasoning quality
        "ALTER TABLE user_picks ADD COLUMN llm_features_extracted INTEGER DEFAULT 0",
        "ALTER TABLE user_picks ADD COLUMN reasoning_complexity_score REAL DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_user_picks_editable_columns():
    """Add promo/reasoning columns that support post-submit editing (Phase 8B)."""
    migrations = [
        "ALTER TABLE user_picks ADD COLUMN reasoning TEXT DEFAULT NULL",
        "ALTER TABLE user_picks ADD COLUMN promo_type TEXT DEFAULT NULL",
        "ALTER TABLE user_picks ADD COLUMN promo_boost_pct REAL DEFAULT NULL",
        "ALTER TABLE user_picks ADD COLUMN promo_boosted_odds REAL DEFAULT NULL",
        "ALTER TABLE user_picks ADD COLUMN potential_profit REAL DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_user_picks_ingest_columns():
    """Add screenshot ingestion columns to user_picks and user_pick_legs (Phase 9)."""
    migrations = [
        # user_picks
        "ALTER TABLE user_picks ADD COLUMN bet_placed_at DATETIME DEFAULT NULL",
        "ALTER TABLE user_picks ADD COLUMN fanduel_bet_id TEXT DEFAULT NULL",
        "ALTER TABLE user_picks ADD COLUMN added_retroactively BOOLEAN DEFAULT 0",
        "ALTER TABLE user_picks ADD COLUMN ingestion_source TEXT DEFAULT 'manual'",
        "ALTER TABLE user_picks ADD COLUMN cashed_out_value REAL DEFAULT NULL",
        "ALTER TABLE user_picks ADD COLUMN potential_payout REAL DEFAULT NULL",
        # user_pick_legs
        "ALTER TABLE user_pick_legs ADD COLUMN is_part_of_sgp BOOLEAN DEFAULT 0",
        "ALTER TABLE user_pick_legs ADD COLUMN price_source TEXT DEFAULT 'manual'",
        "ALTER TABLE user_pick_legs ADD COLUMN price_confidence REAL DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def _migrate_user_pick_leg_outcome_columns():
    """Add per-leg failure analysis columns to user_pick_legs (Phase 10)."""
    migrations = [
        "ALTER TABLE user_pick_legs ADD COLUMN point REAL DEFAULT NULL",
        "ALTER TABLE user_pick_legs ADD COLUMN player TEXT DEFAULT NULL",
        "ALTER TABLE user_pick_legs ADD COLUMN actual_outcome_value REAL DEFAULT NULL",
        "ALTER TABLE user_pick_legs ADD COLUMN expected_outcome_value REAL DEFAULT NULL",
        "ALTER TABLE user_pick_legs ADD COLUMN miss_margin REAL DEFAULT NULL",
        "ALTER TABLE user_pick_legs ADD COLUMN outcome_source TEXT DEFAULT NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
            except Exception:
                pass   # column already exists


def init_db():
    Base.metadata.create_all(bind=engine)
    # Initialize migration safety infrastructure first
    try:
        import safe_migrate as _sm
        _sm.initialize(engine)
    except Exception as _sme:
        print(f"[DB] safe_migrate init warning (non-fatal): {_sme}")
    _migrate_promo_columns()
    _migrate_cash_out_columns()
    _migrate_mock_bet_columns()
    _migrate_bet_leg_result()
    _migrate_leg_quality_columns()
    _migrate_mock_bet_leg_resolution()
    _migrate_game_commence_time()
    _migrate_bet_leg_subtype()
    _migrate_clv_columns()
    _migrate_boost_columns()
    _migrate_user_curation_columns()
    _migrate_exclusion_mode_columns()
    _migrate_user_picks_tables()
    _migrate_llm_signal_columns()
    _migrate_user_picks_editable_columns()
    _migrate_user_picks_ingest_columns()
    _migrate_user_pick_leg_outcome_columns()
    print(f"[DB] Initialized at {DB_PATH}")
    # New tables are created by Base.metadata.create_all above (LegHistoricalResolution,
    # RegimeAbLog).  No explicit migration needed — they're new, not additions to old tables.
