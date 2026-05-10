# BetIQ

**https://github.com/joseportilloj17/BetIQ**

Personal sports betting analytics platform. Tracks real bets, runs an ML model, generates daily picks, and compares your handicapping against the model over time.

---

## Features

**Bet tracking**
- Import Pikkit CSV exports or log bets manually
- FanDuel screenshot parser (auto-extracts legs, odds, stake)
- Quick-entry form with live parlay preview
- Per-bet cash-out tracking and retrospective verdict

**ML model**
- Gradient boosted model trained on your personal bet history
- SHAP feature importance, drift detection, weekly auto-retrain
- Per-leg quality scoring (LQS A/B/C/D) across Moneyline / Spread / Total / Props

**Daily picks engine**
- Pulls live odds from TheOddsAPI (MLB, NBA, NHL, NFL, Soccer)
- ALE (Alternative Line Evaluation) — compares your line vs standard market
- CLV (Closing Line Value) tracking — did you beat the closing line?
- Parlay builder with EV optimizer and Kelly sizing
- Promo/boost simulation (profit boost, bonus bet, no-sweat)
- Best-line shopping across books

**A/B test — you vs the model**
- Submit your own picks before games start
- Side-by-side win rate, ROI, and CLV vs model picks over time
- Per-leg failure analysis: miss margin, outcome capture, unresolved leg tracker
- Signal learning: detects EDGE_POSITIVE / EDGE_NEGATIVE patterns in your reasoning

**Scheduler (background, no cron needed)**
- Fixture refresh: 7:15 AM CT
- Mock bet generation: 7:30 AM and 3:00 PM CT
- Auto-settlement: every :00 and :30 CT
- Calibration drift check: 10:30 AM CT daily
- Weekly FanDuel sync, signal analysis, personal edge profile refresh

**Guardrails**
- Schema migration safety: dated DB backups before every migration, `migration_history` audit table
- Calibration drift detection: per-grade and per-sport actual vs expected win rate (10pp = alert, 20pp = critical)
- `GET /api/health/daily-summary` — combined scheduler health, drift, migration audit, DB stats

---

## Project structure

```
BetIQ/
├── START.sh / START.bat      # One-click launch (Mac/Windows)
├── requirements.txt
├── backend/
│   ├── main.py               # FastAPI server (~6,600 lines, 60+ endpoints)
│   ├── database.py           # SQLAlchemy models + migration runner
│   ├── scheduler.py          # Background job scheduler (threading)
│   ├── recommender.py        # Today's Picks generator
│   ├── parlay_builder.py     # EV optimizer + leg combinator
│   ├── ml_model.py           # GBM model + SHAP + drift detection
│   ├── auto_settle.py        # Settlement engine + retrain trigger
│   ├── analytics.py          # Stats engine
│   ├── odds_api.py           # TheOddsAPI integration
│   ├── creator_tier.py       # Props, best lines, CLV fetch
│   ├── fanduel_parser.py     # Screenshot OCR → structured legs
│   ├── leg_quality.py        # LQS scoring engine
│   ├── calibration_tracker.py # Drift detection
│   ├── safe_migrate.py       # Migration safety wrapper + DB backups
│   ├── kelly.py              # Kelly sizing + bankroll tracker
│   ├── attribution.py        # Performance attribution
│   ├── live_monitor.py       # In-game cash-out monitor
│   └── analysis/
│       ├── regime_classifier.py
│       └── signal_analysis.py
└── frontend/
    └── index.html            # Full dashboard (single file, no build step)
```

---

## Setup

### Mac / Linux
```bash
bash START.sh
```

### Windows
Double-click `START.bat`.

Both scripts create a `.venv`, install dependencies, and start the backend on `http://localhost:8000`. Then open `frontend/index.html` in Chrome.

### Manual
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## First run

1. **Model tab** → Train Now
2. **Fixtures tab** → Fetch Latest Odds
3. **Today's Picks** → Generate Picks

The scheduler starts automatically in the background and handles settlement, retrain, and fixture refresh on its own schedule.

---

## Configuration

API keys are loaded from environment variables. Create a `.env` file in the project root (never committed):

```
ODDS_API_KEY=your_theoddsapi_key
ANTHROPIC_API_KEY=your_anthropic_key       # for LLM signal extraction
API_FOOTBALL_KEY=your_apifootball_key      # for soccer results
WEATHER_API_KEY=your_weatherapi_key        # optional
```

Or set them in your shell before running `START.sh`.

---

## Health check

```
GET http://localhost:8000/api/health/daily-summary
```

Returns scheduler status, calibration drift by grade/sport, migration history, and DB row counts.

```
GET http://localhost:8000/api/scheduler/health
```

Per-job watchdog status (fixtures, pitchers, picks, mocks, settlement).

---

## API docs

```
http://localhost:8000/docs
```

Interactive Swagger UI for all endpoints.

---

## Troubleshooting

**Port 8000 in use**
```bash
cd backend && uvicorn main:app --port 8001
```
Update `const API = 'http://localhost:8001'` at the top of `frontend/index.html`.

**No picks available** — fetch odds first (Fixtures tab), then train the model (Model tab).

**"Not enough data"** — seed the DB from a Pikkit CSV export:
```bash
cd backend && python etl.py --csv ../data/transactions.csv --reset
```

**LibreSSL warning on Mac** — harmless, ignore it.
