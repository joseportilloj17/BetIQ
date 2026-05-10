# BetIQ — Sports Betting Analytics

## What's in this folder

```
BetIQ/
├── START.sh              ← Mac/Linux: double-click or run in terminal
├── START.bat             ← Windows: double-click to run
├── requirements.txt      ← Python dependencies
├── backend/
│   ├── main.py           ← FastAPI server (51 endpoints)
│   ├── database.py       ← SQLite schema
│   ├── etl.py            ← Pikkit CSV importer
│   ├── analytics.py      ← Stats engine
│   ├── ml_model.py       ← GBM model v2 + SHAP + drift detection
│   ├── odds_api.py       ← TheOddsAPI integration
│   ├── parlay_builder.py ← EV optimizer + leg scorer
│   ├── auto_settle.py    ← Auto-settlement engine
│   ├── scheduler.py      ← Background polling (every 30 min)
│   ├── recommender.py    ← Today's Picks generator
│   ├── attribution.py    ← Performance attribution
│   └── kelly.py          ← Kelly sizing + bankroll + paper trading
├── frontend/
│   └── index.html        ← Full dashboard (no build step needed)
└── data/
    └── transactions.csv  ← Your Pikkit bet history (seed data)
```

---

## How to run

### Mac / Linux
```bash
bash START.sh
```
Or just double-click `START.sh` in Finder.

### Windows
Double-click `START.bat`.

Both scripts:
1. Create a `.venv` virtual environment (first run only, ~1 min)
2. Install all dependencies
3. Start the backend on `http://localhost:8000`

Then open `frontend/index.html` in Chrome or Firefox.

---

## First time in the app

Do these three things in order:

1. **Model tab** → click **Train Now**
2. **Fixtures tab** → click **Fetch Latest Odds**
3. **⭐ Today's Picks** → click **Generate Picks**

Everything flows from there. The scheduler runs automatically in the background every 30 minutes to settle bets and retrain the model.

---

## API docs

With the backend running, visit:
```
http://localhost:8000/docs
```
Interactive docs for all 51 endpoints.

---

## Import new Pikkit data

Bet History tab → click **Import CSV** → select your Pikkit export.
Or via API: `POST /api/bets/import-csv`

---

## TheOddsAPI key

Pre-configured in `backend/odds_api.py` (Creator tier — 20,000 requests/month):
```

```
To use your own key, set the environment variable:
```bash
export ODDS_API_KEY=your_key_here
```

---

## Troubleshooting

**"No module named X"** — run `pip install -r requirements.txt` from the BetIQ folder

**"Port 8000 already in use"** — change the port:
```bash
cd backend && uvicorn main:app --port 8001
```
Then edit `const API = 'http://localhost:8001'` at the top of `frontend/index.html`

**LibreSSL warning on Mac** — harmless, ignore it

**"No picks available"** — fetch odds first (Fixtures tab), then train the model (Model tab)

**Model tab says "not enough data"** — the DB needs to be seeded. Stop the server, run:
```bash
cd backend
python etl.py --csv ../data/transactions.csv --reset
```
Then restart.
