"""
main.py — FastAPI backend for the sports betting analytics app.

Start with:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /api/stats/summary
    GET  /api/stats/by-legs
    GET  /api/stats/by-sport
    GET  /api/stats/by-market
    GET  /api/stats/monthly-pnl
    GET  /api/stats/risk-profile

    POST /api/bets/import-csv
    GET  /api/bets
    POST /api/bets/place          — real or mock bet
    PUT  /api/bets/{id}/settle    — settle a mock bet

    POST /api/model/train
    GET  /api/model/predict
    GET  /api/model/backtest

    GET  /api/fixtures
    GET  /api/fixtures/recommend
    POST /api/fixtures/refresh
"""
from __future__ import annotations

import os
import uuid
import json
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import init_db, get_db, Bet, BetLeg, Fixture, MockBet, UserThesis, UserPick, UserPickLeg, UserPickSignal, MockBetLeg
import user_signal_learning as usl
import llm_signal_extractor as lse
import fanduel_parser as fp
from etl import import_csv
from fanduel_importer import detect_csv_format, import_sportsbook_scout_csv
import analytics as ana
import ml_model as ml
import odds_api as oapi
import parlay_builder as pb
import recommender as rec
import cash_out_engine as co_engine
import attribution as attr
import kelly as kly
import auto_settle as asettler
import scheduler as sched
import live_monitor as lm

# ─── Today's picks cache ──────────────────────────────────────────────────────
# In-memory cache keyed by CT date string ("YYYY-MM-DD").
# Populated by the 7 AM scheduler job and lazily on first POST /api/recommend/today.
# Persisted to disk so server restarts don't blank the picks page.

from zoneinfo import ZoneInfo as _ZoneInfo
_CACHE_TZ   = _ZoneInfo("America/Chicago")
_picks_cache: dict = {}   # {"2026-04-24": {result_dict}}

_DATA_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
_CACHE_FILE = os.path.join(_DATA_DIR, "picks_cache.json")

def _extract_screenshot_timestamp(image_path: str):
    """
    Extract EXIF DateTimeOriginal/Digitized from screenshot.
    iPhone EXIF is in local time. Returns datetime or None.
    PNG screenshots from iOS often lack EXIF — handled gracefully.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        # JPEG has _getexif(); PNG typically does not
        exif_data = getattr(img, '_getexif', lambda: None)()
        if not exif_data:
            return None
        for tag_id in [36867, 36868, 306]:  # DateTimeOriginal, DateTimeDigitized, DateTime
            if tag_id in exif_data:
                dt_str = exif_data[tag_id]
                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        return None
    except Exception:
        return None


def _cache_key() -> str:
    from datetime import datetime as _dt
    return _dt.now(_CACHE_TZ).strftime("%Y-%m-%d")

def _get_cached_picks() -> dict | None:
    return _picks_cache.get(_cache_key())

def _set_cached_picks(result: dict) -> None:
    key = _cache_key()
    _picks_cache.clear()          # evict any previous date
    _picks_cache[key] = result
    # Persist to disk so the cache survives server restarts
    try:
        payload = {"date": key, "picks": result, "cached_at": datetime.utcnow().isoformat()}
        with open(_CACHE_FILE, "w") as _f:
            json.dump(payload, _f, default=str)
    except Exception as _ce:
        print(f"[cache] disk write failed: {_ce}")

def _load_disk_cache() -> None:
    """Load picks cache from disk if today's date matches. Safe to call multiple times."""
    today_str = _cache_key()          # "YYYY-MM-DD" in CT timezone
    print(f"[cache] _load_disk_cache() — today_ct={today_str}")
    try:
        print(f"[cache] looking for {_CACHE_FILE}")
        if not os.path.exists(_CACHE_FILE):
            print("[cache] file not found — nothing to load")
            return
        with open(_CACHE_FILE) as _f:
            payload = json.load(_f)
        stored_date = str(payload.get("date", ""))[:10]   # normalise to YYYY-MM-DD
        print(f"[cache] stored_date={stored_date!r}  today_str={today_str!r}  match={stored_date == today_str}")
        if stored_date == today_str and payload.get("picks"):
            _picks_cache.clear()
            _picks_cache[today_str] = payload["picks"]
            tier_b = len(payload["picks"].get("tier_b", []))
            tier_c = len(payload["picks"].get("tier_c", []))
            tier_d = len(payload["picks"].get("tier_d", []))
            print(f"[cache] ✓ restored from disk — tier_b={tier_b} tier_c={tier_c} tier_d={tier_d} "
                  f"(cached_at={payload.get('cached_at','')})")
        else:
            print(f"[cache] stale or empty — skipping (stored={stored_date})")
    except Exception as _le:
        print(f"[cache] disk load failed: {_le}")

# Load at import time so the cache is available immediately when the module loads,
# regardless of whether the FastAPI startup event fires before the first request.
_load_disk_cache()

def invalidate_picks_cache() -> None:
    _picks_cache.clear()
    try:
        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)
    except Exception:
        pass

_regen_in_progress: bool = False

# Only one mock-bets generation may run at a time.
# Endpoint fires a background thread and returns immediately; clients poll
# /api/mock-bets/status for completion.
import threading as _threading
_mock_gen_lock    = _threading.Lock()
_mock_gen_in_progress: bool   = False
_mock_gen_result:  dict | None = None
_mock_gen_error:   str  | None = None

def _background_regen(req_dict: dict) -> None:
    """Run pick generation in a background thread with its own DB session."""
    global _regen_in_progress
    try:
        from database import SessionLocal as _SL
        _db = _SL()
        try:
            _req = RecommendRequest(**req_dict)
            result = _build_todays_picks(_req, _db)
            _set_cached_picks(result)
        finally:
            _db.close()
    except Exception as _e:
        import traceback
        traceback.print_exc()
    finally:
        _regen_in_progress = False

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Betting Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────────────────────
import os as _os
_FRONTEND = _os.path.join(_os.path.dirname(__file__), "..", "frontend")
_FRONTEND = _os.path.abspath(_FRONTEND)

@app.get("/admin", response_class=HTMLResponse)
def serve_admin():
    """Mobile-friendly admin panel for common operations."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BetIQ Admin</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117; color: #e2e8f0;
    min-height: 100vh; padding: 24px 16px;
  }
  h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 6px; color: #f8fafc; }
  .subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 20px; }

  /* ── Section headers ── */
  .section-title {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #64748b;
    margin: 24px 0 10px;
  }

  /* ── Action buttons ── */
  .btn {
    display: block; width: 100%; padding: 18px 20px;
    margin-bottom: 14px; border: none; border-radius: 12px;
    font-size: 1rem; font-weight: 600; cursor: pointer;
    text-align: left; transition: opacity .15s, transform .1s;
    position: relative;
  }
  .btn:active { transform: scale(0.98); opacity: 0.85; }
  .btn-label { display: block; font-size: 1rem; }
  .btn-sub   { display: block; font-size: 0.75rem; font-weight: 400; opacity: 0.7; margin-top: 3px; }
  .btn-settle   { background: #1e40af; color: #fff; }
  .btn-picks    { background: #065f46; color: #fff; }
  .btn-cal      { background: #6b21a8; color: #fff; }
  .btn-injuries { background: #92400e; color: #fff; }
  .btn-validate { background: #1e3a5f; color: #fff; }
  .btn-upload   { background: #0e7490; color: #fff; }
  .btn .spinner {
    display: none; position: absolute; right: 20px; top: 50%;
    transform: translateY(-50%);
    width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: translateY(-50%) rotate(360deg); } }
  .btn.loading .spinner { display: block; }
  .btn.loading { opacity: 0.7; pointer-events: none; }

  /* ── Result box ── */
  .result-box {
    margin-top: 8px; padding: 16px; border-radius: 10px;
    background: #1e2433; border: 1px solid #2d3748;
    font-size: 0.82rem; line-height: 1.6;
    white-space: pre-wrap; word-break: break-word;
    display: none;
  }
  .result-box.success { border-color: #22c55e; }
  .result-box.error   { border-color: #ef4444; }
  .result-label { font-size: 0.7rem; color: #64748b; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .05em; }

  /* ── Status cards ── */
  .status-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; margin-bottom: 4px;
  }
  .stat-card {
    background: #1e2433; border: 1px solid #2d3748;
    border-radius: 10px; padding: 14px 12px;
  }
  .stat-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
  .stat-value { font-size: 1.1rem; font-weight: 700; color: #f8fafc; }
  .stat-value.loading-val { color: #475569; font-size: 0.85rem; }

  /* ── Screenshot upload ── */
  .upload-zone {
    background: #1e2433; border: 2px dashed #334155;
    border-radius: 12px; padding: 20px 16px;
    text-align: center; margin-bottom: 14px;
  }
  .upload-zone label {
    display: block; cursor: pointer;
    font-size: 0.9rem; color: #94a3b8; margin-bottom: 10px;
  }
  .upload-zone input[type=file] { display: none; }
  #file-name { font-size: 0.8rem; color: #22c55e; margin-top: 6px; min-height: 18px; }

  .ts-block {
    background: #1e2433; border: 1px solid #2d3748;
    border-radius: 10px; padding: 14px 12px; margin-top: 10px; display: none;
  }
  .ts-block .ts-display { font-size: 0.85rem; color: #94a3b8; margin-bottom: 10px; }
  .ts-block label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; display: block; margin-bottom: 4px; }
  .ts-block input[type=datetime-local] {
    width: 100%; background: #0f1117; border: 1px solid #334155;
    border-radius: 8px; color: #e2e8f0; padding: 10px 12px;
    font-size: 0.9rem;
  }
</style>
</head>
<body>
<h1>BetIQ Admin</h1>
<p class="subtitle">Tap a button to run an operation.</p>

<!-- ── Status section ── -->
<div class="section-title">Server Status</div>
<div class="status-grid">
  <div class="stat-card">
    <div class="stat-label">Total Settled</div>
    <div class="stat-value loading-val" id="stat-settled">…</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Win Rate</div>
    <div class="stat-value loading-val" id="stat-wr">…</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Today's Picks</div>
    <div class="stat-value loading-val" id="stat-picks">…</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Pending Bets</div>
    <div class="stat-value loading-val" id="stat-pending">…</div>
  </div>
</div>
<div class="stat-card" style="margin-bottom:4px">
  <div class="stat-label">Server Time</div>
  <div class="stat-value loading-val" id="stat-uptime">…</div>
</div>

<!-- ── Action buttons ── -->
<div class="section-title">Actions</div>

<button class="btn btn-settle" onclick="run(this, 'POST', '/api/mock-bets/settle', 'Run Settlement', 'result')">
  <span class="btn-label">Run Settlement</span>
  <span class="btn-sub">Settle all pending mock bets</span>
  <span class="spinner"></span>
</button>

<button class="btn btn-picks" onclick="run(this, 'POST', '/api/recommend/today', 'Generate PM Picks', 'result')">
  <span class="btn-label">Generate PM Picks</span>
  <span class="btn-sub">Refresh today&rsquo;s recommendations</span>
  <span class="spinner"></span>
</button>

<button class="btn btn-cal" onclick="run(this, 'POST', '/api/scout/calibration/run', 'Refresh Calibration', 'result')">
  <span class="btn-label">Refresh Calibration</span>
  <span class="btn-sub">Run scout calibration check</span>
  <span class="spinner"></span>
</button>

<button class="btn btn-injuries" onclick="run(this, 'POST', '/api/health/refresh-injuries', 'Refresh Injuries', 'result')">
  <span class="btn-label">Refresh Injuries</span>
  <span class="btn-sub">Pull latest injury reports</span>
  <span class="spinner"></span>
</button>

<button class="btn btn-validate" onclick="run(this, 'POST', '/api/validation/run', 'Run Validations', 'result')">
  <span class="btn-label">Run Validations</span>
  <span class="btn-sub">Model + signal validation suite</span>
  <span class="spinner"></span>
</button>

<div class="result-box" id="result">
  <div class="result-label" id="result-label"></div>
  <div id="result-body"></div>
</div>

<!-- ── Screenshot upload ── -->
<div class="section-title">Upload Bet Slip Screenshot</div>
<div class="upload-zone">
  <label for="slip-file">
    Tap to choose a screenshot (camera or photo roll)
  </label>
  <input type="file" id="slip-file" accept="image/*" capture="environment"
         onchange="onFileChosen(this)">
  <div id="file-name"></div>
</div>

<button class="btn btn-upload" id="upload-btn" onclick="uploadScreenshot(this)" disabled style="opacity:0.4">
  <span class="btn-label">Parse Screenshot</span>
  <span class="btn-sub">Upload to /api/fanduel/parse-screenshot</span>
  <span class="spinner"></span>
</button>

<div class="ts-block" id="ts-block">
  <div class="ts-display" id="ts-display"></div>
  <label for="ts-override">Override timestamp</label>
  <input type="datetime-local" id="ts-override">
</div>

<div class="result-box" id="upload-result">
  <div class="result-label" id="upload-result-label"></div>
  <div id="upload-result-body"></div>
</div>

<script>
// ── Status loader ──────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const [perfRes, picksRes, healthRes] = await Promise.allSettled([
      fetch('/api/mock-bets/performance'),
      fetch('/api/recommend/today'),
      fetch('/api/health'),
    ]);

    if (perfRes.status === 'fulfilled' && perfRes.value.ok) {
      const d = await perfRes.value.json();
      const s = document.getElementById('stat-settled');
      const w = document.getElementById('stat-wr');
      const p = document.getElementById('stat-pending');
      s.textContent = d.total_settled ?? '—';
      s.className = 'stat-value';
      const wr = d.win_rate != null ? (d.win_rate * 100).toFixed(1) + '%' : '—';
      w.textContent = wr;
      w.className = 'stat-value';
      p.textContent = d.total_pending ?? '—';
      p.className = 'stat-value';
    }

    if (picksRes.status === 'fulfilled' && picksRes.value.ok) {
      const d = await picksRes.value.json();
      const el = document.getElementById('stat-picks');
      const picks = d.picks ?? d.recommendations ?? d.bets ?? [];
      const cnt = Array.isArray(picks) ? picks.length : (d.count ?? '—');
      el.textContent = cnt;
      el.className = 'stat-value';
    }

    if (healthRes.status === 'fulfilled' && healthRes.value.ok) {
      const d = await healthRes.value.json();
      const el = document.getElementById('stat-uptime');
      el.textContent = d.time ?? d.uptime ?? d.status ?? '—';
      el.className = 'stat-value';
    }
  } catch(e) { /* non-fatal */ }
}
loadStatus();

// ── Generic action runner ──────────────────────────────────────────────────
async function run(btn, method, url, label, resultId) {
  btn.classList.add('loading');
  const el = document.getElementById(resultId);
  const lb = document.getElementById(resultId + '-label');
  const rb = document.getElementById(resultId + '-body');
  el.style.display = 'block';
  el.className = 'result-box';
  lb.textContent = label + ' — running…';
  rb.textContent = '';
  try {
    const res = await fetch(url, { method, headers: { 'Content-Type': 'application/json' } });
    const text = await res.text();
    let pretty = text;
    try { pretty = JSON.stringify(JSON.parse(text), null, 2); } catch {}
    lb.textContent = label + (res.ok ? ' — OK ' + res.status : ' — ERROR ' + res.status);
    rb.textContent = pretty;
    el.className = 'result-box ' + (res.ok ? 'success' : 'error');
  } catch (e) {
    lb.textContent = label + ' — FAILED';
    rb.textContent = e.toString();
    el.className = 'result-box error';
  } finally {
    btn.classList.remove('loading');
  }
}

// ── Screenshot upload ──────────────────────────────────────────────────────
function onFileChosen(input) {
  const fn = document.getElementById('file-name');
  const btn = document.getElementById('upload-btn');
  if (input.files && input.files[0]) {
    fn.textContent = input.files[0].name;
    btn.disabled = false;
    btn.style.opacity = '1';
  } else {
    fn.textContent = '';
    btn.disabled = true;
    btn.style.opacity = '0.4';
  }
  // Hide previous results when new file chosen
  document.getElementById('ts-block').style.display = 'none';
  document.getElementById('upload-result').style.display = 'none';
}

async function uploadScreenshot(btn) {
  const fileInput = document.getElementById('slip-file');
  if (!fileInput.files || !fileInput.files[0]) return;

  btn.classList.add('loading');
  const el = document.getElementById('upload-result');
  const lb = document.getElementById('upload-result-label');
  const rb = document.getElementById('upload-result-body');
  el.style.display = 'block';
  el.className = 'result-box';
  lb.textContent = 'Parse Screenshot — uploading…';
  rb.textContent = '';
  document.getElementById('ts-block').style.display = 'none';

  try {
    const form = new FormData();
    form.append('image', fileInput.files[0]);

    const res = await fetch('/api/fanduel/parse-screenshot', { method: 'POST', body: form });
    const text = await res.text();
    let data = null;
    let pretty = text;
    try { data = JSON.parse(text); pretty = JSON.stringify(data, null, 2); } catch {}

    lb.textContent = 'Parse Screenshot — ' + (res.ok ? 'OK ' + res.status : 'ERROR ' + res.status);
    rb.textContent = pretty;
    el.className = 'result-box ' + (res.ok ? 'success' : 'error');

    // Show timestamp block if response has timestamp info
    if (data && data.timestamp_display) {
      const tsBlock = document.getElementById('ts-block');
      document.getElementById('ts-display').textContent = data.timestamp_display;
      const tsInput = document.getElementById('ts-override');
      tsInput.value = data.timestamp_iso || '';
      tsBlock.style.display = 'block';
    }
  } catch (e) {
    lb.textContent = 'Parse Screenshot — FAILED';
    rb.textContent = e.toString();
    el.className = 'result-box error';
  } finally {
    btn.classList.remove('loading');
  }
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the BetIQ dashboard."""
    from fastapi.responses import Response
    import time as _time
    html_path = _os.path.join(_FRONTEND, "index.html")
    with open(html_path, "r") as f:
        html = f.read()
    # Inject build stamp so browser always loads fresh JS and user can verify version
    mtime = int(_os.path.getmtime(html_path))
    build_stamp = f'<meta name="betiq-build" content="{mtime}">'
    html = html.replace("</head>", f"{build_stamp}\n</head>", 1)
    import hashlib as _hashlib
    etag = '"' + _hashlib.md5(str(mtime).encode()).hexdigest()[:12] + '"'
    return Response(
        content=html,
        media_type="text/html",
        headers={
            "Content-Security-Policy": "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "ETag": etag,
        }
    )

# Mount static files (CSS/JS/assets if any)
if _os.path.exists(_FRONTEND):
    app.mount("/frontend", StaticFiles(directory=_FRONTEND), name="frontend")

@app.on_event("startup")
def startup():
    init_db()
    # Ensure personal_edge_profile table exists and is populated.
    # This is a fast no-op if already populated; runs ~1s if empty on first boot.
    try:
        import personal_edge_profile as _pep
        _pep.ensure_populated()
    except Exception as _pep_err:
        print(f"[Startup] personal_edge_profile init error (non-fatal): {_pep_err}")
    _load_disk_cache()   # restore today's picks if server restarted

    # ── Sub-model boot health check ──────────────────────────────────────────
    # Attempt to load every expected sub-model.  Log CRITICAL if any fails so
    # a broken pickle is never silently swallowed (broke 2026-06-09..06-13).
    _submodel_status = {}
    _sm_checks = {
        "mlb_ats":    ("MLB",    ml._load_ats_submodel),
        "mlb_totals": ("MLB",    ml._load_totals_submodel),
    }
    for _sm_name, (_sm_sport, _sm_loader) in _sm_checks.items():
        try:
            _sm_result = _sm_loader(_sm_sport)
            if _sm_result is None:
                _submodel_status[_sm_name] = "MISSING"
                print(f"CRITICAL [Startup] Sub-model '{_sm_name}' returned None "
                      f"-- all {_sm_sport} legs will fall back to combined_v1")
            else:
                _sm_clf, _sm_sc, _sm_imp, _sm_cols = _sm_result
                _submodel_status[_sm_name] = f"OK ({len(_sm_cols)} features, {type(_sm_clf).__name__})"
                print(f"[Startup] Sub-model '{_sm_name}' loaded: "
                      f"{type(_sm_clf).__name__}, {len(_sm_cols)} features")
        except Exception as _sm_err:
            _submodel_status[_sm_name] = f"ERROR: {_sm_err}"
            print(f"CRITICAL [Startup] Sub-model '{_sm_name}' failed to load: {_sm_err}")
    # Also check global model
    try:
        _g_clf, _g_sc, _g_imp, _g_cols = ml.load_model()
        _submodel_status["global"] = f"OK ({len(_g_cols)} features)"
        print(f"[Startup] Global model loaded: {len(_g_cols)} features")
    except Exception as _g_err:
        _submodel_status["global"] = f"ERROR: {_g_err}"
        print(f"CRITICAL [Startup] Global model failed to load: {_g_err}")
    app.state.submodel_status = _submodel_status

    # Auto-import seed CSV if DB is empty
    seed = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
    if os.path.exists(seed):
        db = next(get_db())
        count = db.query(Bet).count()
        if count == 0:
            print("[Startup] Seeding DB from transactions.csv…")
            import_csv(seed)
    # Start background auto-settle scheduler (every 30 min by default)
    sched.start(interval_mins=30, auto_retrain=True, daemon=True)
    print("[Startup] Auto-settle scheduler started (30 min interval)")

    # System 3F: Auto-backfill retroactive mocks if table is sparse
    import threading as _threading
    def _maybe_backfill():
        import time; time.sleep(10)   # let server finish startup
        try:
            import mock_bets as _mb
            from database import SessionLocal as _SL
            _db = _SL()
            try:
                n = _db.query(__import__("database").MockBet).count()
            finally:
                _db.close()
            if n < 100:
                print(f"[Startup] mock_bets has {n} rows — starting retroactive backfill")
                _mb.start_backfill_job(lookback_days=180, n_per_day=30)
            else:
                print(f"[Startup] mock_bets has {n} rows — skipping auto-backfill")
        except Exception as exc:
            print(f"[Startup] Auto-backfill check failed: {exc}")
    _threading.Thread(target=_maybe_backfill, daemon=True, name="StartupBackfill").start()

    # Startup watchdog: runs 60 s after boot to recover any missed morning jobs.
    # Catches the case where the server restarted mid-morning (e.g. 9 AM restart
    # after fixture_refresh ran at 7:45 AM) and the scheduler loop hasn't ticked
    # yet.  _run_watchdog() is safe to call here — it uses its own sqlite3
    # connections and only acts during active hours (7 AM – 11 PM CT).
    def _run_watchdog_startup():
        import time as _time
        _time.sleep(60)
        try:
            sched._run_watchdog()
            print("[Watchdog] Startup check complete")
        except Exception as _e:
            print(f"[Watchdog] Startup check failed: {_e}")

    _threading.Thread(target=_run_watchdog_startup, daemon=True, name="StartupWatchdog").start()

    # Scheduler thread watchdog: every 5 minutes, check if the scheduler loop
    # is still alive.  If it hasn't run a cycle in > 35 minutes, auto-restart.
    # This guards against silent thread crashes (e.g. uncaught exception in
    # SettlementCron self-heal, unexpected disk I/O error bubbling out).
    def _scheduler_health_watchdog():
        import time as _time
        _time.sleep(120)   # let startup settle before first check
        while True:
            _time.sleep(300)   # check every 5 minutes
            try:
                last = sched._scheduler_state.get("last_run")
                running = sched._scheduler_state.get("running", False)
                if not running:
                    continue  # scheduler intentionally stopped
                if last is None:
                    continue  # scheduler just started, no cycle yet
                from datetime import datetime as _dt, timezone as _tz
                elapsed = (_dt.now(_tz.utc) - _dt.fromisoformat(last)).total_seconds()
                interval = sched._scheduler_state.get("interval_mins", 30) * 60
                # Overdue threshold: 1.5x the interval (e.g. 45 min for 30-min interval)
                threshold = interval * 1.5
                if elapsed > threshold:
                    print(f"[SchedulerWatchdog] ⚠️  Scheduler overdue "
                          f"({elapsed/60:.0f}m since last cycle, threshold {threshold/60:.0f}m) — restarting")
                    sched.stop()
                    _time.sleep(2)
                    sched.start(interval_mins=30, auto_retrain=True, daemon=True)
                    print("[SchedulerWatchdog] ✅ Scheduler restarted")
            except Exception as _we:
                print(f"[SchedulerWatchdog] check error: {_we}")

    _threading.Thread(target=_scheduler_health_watchdog, daemon=True, name="SchedulerWatchdog").start()

    # Scout schema: add scouted_prop_id + scout_grade to mock_bet_legs and user_pick_legs
    try:
        import safe_migrate as _sm
        from database import engine as _engine
        _sm.initialize(_engine)
        _sm.safe_add_column(_engine, "mock_bet_legs",  "scouted_prop_id",    "INTEGER")
        _sm.safe_add_column(_engine, "mock_bet_legs",  "scout_grade",        "TEXT")
        _sm.safe_add_column(_engine, "mock_bet_legs",  "scout_hit_prob",     "REAL")
        _sm.safe_add_column(_engine, "mock_bet_legs",  "model_n_features",   "INTEGER")
        _sm.safe_add_column(_engine, "user_pick_legs", "scouted_prop_id",    "INTEGER")
        _sm.safe_add_column(_engine, "user_pick_legs", "scout_grade",        "TEXT")
        _sm.safe_add_column(_engine, "user_pick_legs", "scout_hit_prob",     "REAL")
        _sm.safe_add_column(_engine, "user_picks",     "placed_mock_bet_id", "VARCHAR")
        print("[Startup] Scout schema migrations applied")
    except Exception as _sc_err:
        print(f"[Startup] Scout schema migration error (non-fatal): {_sc_err}")

@app.on_event("shutdown")
def shutdown():
    sched.stop()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class PlaceBetRequest(BaseModel):
    bet_type:   str = "parlay"          # parlay | straight
    is_mock:    bool = False
    odds:       float
    amount:     float
    legs:       int
    sports:     Optional[str] = None
    leagues:    Optional[str] = None
    bet_info:   Optional[str] = None    # pipe-delimited leg descriptions
    sportsbook: str = "FanDuel"
    notes:      Optional[str] = None

class SettleBetRequest(BaseModel):
    result: str   # "WIN" | "LOSS"
    profit: Optional[float] = None

class TrainRequest(BaseModel):
    algorithm: str = "gradient_boost"   # gradient_boost | random_forest | logistic_regression

class PredictRequest(BaseModel):
    legs:         int   = 4
    odds:         float = 5.0
    sports:       str   = "Basketball"
    leagues:      str   = "NBA"
    bet_info:     Optional[str] = None
    stake:        float = 10.0
    is_parlay:    bool  = True
    hour_placed:  Optional[int] = None
    day_of_week:  Optional[int] = None


# ─── Stats routes ─────────────────────────────────────────────────────────────

@app.get("/api/stats/summary")
def stats_summary(db: Session = Depends(get_db)):
    return ana.get_summary_stats(db)

@app.get("/api/stats/by-legs")
def stats_by_legs(db: Session = Depends(get_db)):
    return ana.get_stats_by_legs(db)

@app.get("/api/stats/by-sport")
def stats_by_sport(db: Session = Depends(get_db)):
    return ana.get_stats_by_sport(db)

@app.get("/api/stats/by-market")
def stats_by_market(db: Session = Depends(get_db)):
    return ana.get_stats_by_market(db)

@app.get("/api/stats/monthly-pnl")
def stats_monthly_pnl(db: Session = Depends(get_db)):
    return ana.get_monthly_pnl(db)

@app.get("/api/stats/risk-profile")
def stats_risk_profile(db: Session = Depends(get_db)):
    return ana.get_risk_profile(db)


# ─── Bet routes ───────────────────────────────────────────────────────────────

@app.post("/api/bets/import-csv")
async def import_bets_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a bet CSV export.  Format is auto-detected:

      pikkit           — Pikkit export (one row per parlay)
      fanduel          — FanDuel export (legs_summary column)
      sportsbook_scout — Sportsbook Scout (one row per leg, Leg Result column)

    Sportsbook Scout imports apply leg_result to matching Pikkit bets
    (crosswalk by date + amount + legs) before creating new records.
    """
    import pandas as pd
    from io import BytesIO

    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    fmt = detect_csv_format(df)

    if fmt == "sportsbook_scout":
        result = import_sportsbook_scout_csv(df, db)
        return {
            "status":  "ok",
            "format":  "sportsbook_scout",
            "message": (
                f"Sportsbook Scout import complete: "
                f"{result['inserted']} new bets, "
                f"{result['pikkit_crosswalked']} Pikkit bets enriched with leg results, "
                f"{result['leg_results_written']} leg outcomes written"
            ),
            **result,
        }

    if fmt == "fanduel":
        from fanduel_importer import import_fanduel_csv
        tmp = f"/tmp/fd_upload_{uuid.uuid4()}.csv"
        with open(tmp, "wb") as f:
            f.write(content)
        try:
            summary = import_fanduel_csv(tmp, db)
        finally:
            os.remove(tmp)
        return {"status": "ok", "format": "fanduel", **summary}

    if fmt == "pikkit":
        tmp = f"/tmp/pikkit_upload_{uuid.uuid4()}.csv"
        with open(tmp, "wb") as f:
            f.write(content)
        try:
            import_csv(tmp)
        finally:
            os.remove(tmp)
        return {"status": "ok", "format": "pikkit", "message": "Pikkit CSV imported successfully"}

    return {
        "status":  "error",
        "format":  "unknown",
        "message": (
            "Could not detect CSV format. "
            "Expected columns for Pikkit (bet_id + bet_info), "
            "FanDuel (legs_summary or american_odds), or "
            "Sportsbook Scout (External Bet ID + Primary + Leg Result)."
        ),
    }

@app.get("/api/bets")
def list_bets(
    limit:   int = Query(50, le=500),
    offset:  int = 0,
    status:  Optional[str] = None,
    is_mock: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Bet).order_by(Bet.time_placed.desc())
    if status:  q = q.filter(Bet.status == status)
    if is_mock is not None: q = q.filter(Bet.is_mock == is_mock)
    total = q.count()
    bets  = q.offset(offset).limit(limit).all()
    return {
        "total": total,
        "bets": [_bet_to_dict(b) for b in bets]
    }

@app.post("/api/bets/place")
def place_bet(req: PlaceBetRequest, db: Session = Depends(get_db)):
    """
    Record a bet placed through the app (real or mock paper trade).
    Real bets go into the learning loop just like Pikkit imports.
    """
    import math
    bet_id = str(uuid.uuid4())
    now    = datetime.utcnow()

    # Run model prediction
    feature_dict = {
        "legs":         req.legs,
        "odds":         req.odds,
        "log_odds":     math.log(req.odds) if req.odds > 1 else 0,
        "implied_prob": 1 / req.odds if req.odds > 1 else 0.5,
        "stake":        req.amount,
        "is_parlay":    1 if req.bet_type == "parlay" else 0,
        "sport_id":     0,
        "league_id":    0,
        "hour_placed":  now.hour,
        "day_of_week":  now.weekday(),
        "ml_pct":       0.5, "spread_pct": 0.2, "total_pct": 0.2, "prop_pct": 0.1,
        "multi_sport":  0, "n_sports": 1, "has_ev": 0, "ev_value": 0.0,
        "closing_line_diff": 0.0,
    }
    prediction = ml.predict_bet(feature_dict)

    # Compute placement-time cash-out target
    n_legs_val = req.legs if isinstance(req.legs, int) else 1
    target_info = lm.compute_cashout_target(
        amount  = req.amount,
        odds    = req.odds,
        n_legs  = n_legs_val,
        avg_lqs = 65.0,   # no LQS available at manual placement
    )

    bet = Bet(
        id                      = bet_id,
        source                  = "app",
        sportsbook              = req.sportsbook,
        bet_type                = req.bet_type,
        status                  = "PLACED",
        odds                    = req.odds,
        amount                  = req.amount,
        legs                    = req.legs,
        sports                  = req.sports,
        leagues                 = req.leagues,
        bet_info                = req.bet_info,
        is_mock                 = req.is_mock,
        time_placed             = now,
        cash_out_target         = target_info["target_amount"],
        cash_out_target_pct     = target_info["target_pct"],
        cash_out_target_rationale = target_info["rationale"],
    )
    db.add(bet)
    db.commit()
    db.refresh(bet)

    # ── Store bet_legs with game_commence_time ────────────────────────────────
    # Parse each pipe-delimited leg from bet_info, look up the fixture
    # commence_time, and write a BetLeg row so the live monitor has an
    # authoritative game-start anchor without re-querying fixtures each cycle.
    legs_created = 0
    if req.bet_info:
        leg_descs = [d.strip() for d in req.bet_info.split("|") if d.strip()]
        sport_for_legs = (req.sports or "").split("|")[0].strip()
        for idx, leg_desc in enumerate(leg_descs):
            try:
                has_mu = " @ " in leg_desc or " v " in leg_desc or " vs " in leg_desc
                import leg_resolver as _lr
                p = _lr.parse_leg_details(leg_desc) if has_mu else lm._parse_compact_leg(leg_desc)
                team_can = _lr.normalize_team(p.get("selected_team_or_player") or "")
                gct = lm._lookup_commence_time(team_can, sport_for_legs, now, db)
                bl = BetLeg(
                    bet_id            = bet_id,
                    leg_index         = idx,
                    description       = leg_desc,
                    market_type       = p.get("market_type"),
                    team              = team_can,
                    sport             = sport_for_legs,
                    league            = sport_for_legs,
                    odds_str          = str(p.get("odds")) if p.get("odds") else None,
                    game_commence_time = gct.isoformat() if gct else None,
                )
                db.add(bl)
                legs_created += 1
            except Exception:
                pass
        if legs_created:
            db.commit()

    return {
        "bet_id":            bet_id,
        "status":            "PLACED",
        "is_mock":           req.is_mock,
        "prediction":        prediction,
        "cash_out_target":   target_info,
        "legs_stored":       legs_created,
        "message":           "Mock bet recorded — settle when result is known." if req.is_mock
                             else "Bet recorded. Settle when result is known via PUT /api/bets/{id}/settle"
    }

@app.put("/api/bets/{bet_id}/settle")
def settle_bet(bet_id: str, req: SettleBetRequest, db: Session = Depends(get_db)):
    """
    Settle an app-placed or mock bet with the real outcome.
    This feeds the result into the learning loop.
    """
    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if not bet:
        raise HTTPException(404, "Bet not found")
    # Allow re-settling to correct mistakes

    bet.status       = f"SETTLED_{req.result}"
    bet.time_settled = datetime.utcnow()
    # Auto-calculate profit
    if req.profit is not None:
        bet.profit = req.profit
    elif req.result == "WIN":
        bet.profit = round((bet.odds - 1) * bet.amount, 2)
    elif req.result == "LOSS":
        bet.profit = -bet.amount
    elif req.result == "PUSH":
        bet.profit = 0
    db.commit()
    return {
        "bet_id":  bet_id,
        "status":  bet.status,
        "profit":  bet.profit,
        "message": "Bet settled and added to training data. Retrain model to include this result."
    }

def _bet_to_dict(b: Bet) -> dict:
    return {
        "id":                         b.id,
        "source":                     b.source,
        "bet_type":                   b.bet_type,
        "status":                     b.status,
        "odds":                       b.odds,
        "amount":                     b.amount,
        "profit":                     b.profit,
        "legs":                       b.legs,
        "sports":                     b.sports,
        "leagues":                    b.leagues,
        "bet_info":                   b.bet_info,
        "is_mock":                    b.is_mock,
        "time_placed":                b.time_placed.isoformat() if b.time_placed else None,
        "time_settled":               b.time_settled.isoformat() if b.time_settled else None,
        "promo_type":                 b.promo_type or "none",
        "promo_boosted_odds":         b.promo_boosted_odds,
        "promo_ev_lift":              b.promo_ev_lift,
        "promo_was_free_bet":         b.promo_was_free_bet or 0,
        # Phase 6B — cash-out tracking
        "cashed_out":                 getattr(b, "cashed_out", 0) or 0,
        "cash_out_amount":            getattr(b, "cash_out_amount", None),
        "cash_out_timestamp":         getattr(b, "cash_out_timestamp", None),
        "cash_out_vs_final_decision": getattr(b, "cash_out_vs_final_decision", None),
        "cash_out_offers_log":        getattr(b, "cash_out_offers_log", None),
    }


# ─── Model routes ─────────────────────────────────────────────────────────────

@app.post("/api/model/train")
def train_model(req: TrainRequest, db: Session = Depends(get_db)):
    """Train / retrain the ML model on all settled bets (including app-placed ones)."""
    result = ml.train(db, algorithm=req.algorithm)
    return result

@app.post("/api/model/predict")
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """Score a hypothetical bet without placing it."""
    import math
    features = {
        "legs":         req.legs,
        "odds":         req.odds,
        "log_odds":     math.log(req.odds) if req.odds > 1 else 0,
        "implied_prob": 1 / req.odds if req.odds > 1 else 0.5,
        "stake":        req.stake,
        "is_parlay":    1 if req.is_parlay else 0,
        "sport_id":     ml.SPORT_MAP.get(req.sports, 6),
        "league_id":    ml.LEAGUE_MAP.get(req.leagues, 10),
        "hour_placed":  req.hour_placed or datetime.utcnow().hour,
        "day_of_week":  req.day_of_week or datetime.utcnow().weekday(),
        "ml_pct": 0.5, "spread_pct": 0.2, "total_pct": 0.2, "prop_pct": 0.1,
        "multi_sport": 0, "n_sports": 1, "has_ev": 0, "ev_value": 0.0,
        "closing_line_diff": 0.0,
    }
    return ml.predict_bet(features)

@app.get("/api/model/backtest")
def backtest(db: Session = Depends(get_db)):
    """Run retroactive simulation using historical settled bets."""
    return ml.backtest(db)

@app.get("/api/model/drift")
def model_drift(window: int = 30, db: Session = Depends(get_db)):
    """Check if model accuracy has drifted vs training baseline."""
    return ml.check_drift(db, window=window)

@app.post("/api/model/explain")
def explain_bet(req: PredictRequest, db: Session = Depends(get_db)):
    """Full prediction + SHAP-style feature attribution for a hypothetical bet."""
    import math as _math
    now = datetime.utcnow()
    features = {
        "legs": req.legs, "odds": req.odds,
        "log_odds": _math.log(req.odds) if req.odds > 1 else 0,
        "implied_prob": 1/req.odds if req.odds > 1 else 0.5,
        "stake": req.stake, "is_parlay": 1 if req.is_parlay else 0,
        "sport_id": ml.SPORT_MAP.get(req.sports, 6),
        "sport_label": req.sports,
        "league_id": ml.LEAGUE_MAP.get(req.leagues, 10),
        "hour_placed": req.hour_placed or now.hour,
        "day_of_week": req.day_of_week or now.weekday(),
        "ml_pct": 0.5, "spread_pct": 0.2, "total_pct": 0.2, "prop_pct": 0.1,
        "multi_sport": 0, "n_sports": 1, "has_ev": 0, "ev_value": 0.0,
        "closing_line_diff": 0.0,
        "odds_bracket": 1 if req.odds < 5 else (2 if req.odds < 15 else 3),
        "leg_diversity": 0.25, "clv": 0.0, "clv_available": 0,
        "multi_league": 0, "n_leagues": 1, "prop_density": 0.0,
        "stake_ratio": req.stake / 6.8,
        "month_sin": _math.sin(2*_math.pi*now.month/12),
        "month_cos": _math.cos(2*_math.pi*now.month/12),
        "hour_sin": _math.sin(2*_math.pi*now.hour/24),
        "hour_cos": _math.cos(2*_math.pi*now.hour/24),
    }
    return ml.predict_bet(features, use_submodel=True)

@app.get("/api/model/submodels")
def list_submodels():
    """List available per-sport ATS sub-models."""
    import glob
    files = glob.glob(os.path.join(ml.SUBMODEL_DIR, "*_ats_clf.pkl"))
    result = []
    for f in files:
        sport = os.path.basename(f).replace("_ats_clf.pkl", "").upper()
        result.append({"sport": sport, "file": os.path.basename(f)})
    return {"submodels": result}


class GameATSRequest(BaseModel):
    sport:        str             # "NHL" or "MLB"
    home_team:    str
    away_team:    str
    game_date:    str | None   = None   # YYYY-MM-DD; defaults to today
    spread_value: float | None = None   # e.g. -1.5, +2.5; defaults to standard run line


@app.post("/api/model/predict-game")
def predict_game(req: GameATSRequest):
    """
    Predict the probability that *home_team* covers the spread using the
    saved diagnostic ATS sub-model for the given sport.

    Never calls train().  Never touches the production model.

    Errors:
      404 — no saved sub-model for this sport yet (train it via Run Diagnostic)
      422 — sub-model exists but no matching game found in historical.db
    """
    import json as _json, os as _os

    sport = req.sport.upper()

    # ── 1. Guard: sub-model must already be saved ─────────────────────────────
    artifacts = ml._load_ats_submodel(sport)
    if artifacts is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No saved sub-model for {sport}. "
                "Run the Diagnostic model with 'Save model if AUC ≥ 0.55' checked first."
            ),
        )

    # ── 2. Load persisted metadata (AUC, feature count, saved_at) ────────────
    meta_path = _os.path.join(ml.SUBMODEL_DIR, f"{sport.lower()}_ats_meta.json")
    meta: dict = {}
    if _os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = _json.load(f)

    # ── 3. Score the game (reads historical.db feature matrix, no training) ──
    prob = ml.predict_game_ats(
        sport        = sport,
        home_team    = req.home_team,
        away_team    = req.away_team,
        game_date    = req.game_date,
        spread_value = req.spread_value,
    )
    if prob is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Sub-model for {sport} is loaded but no matching game was found in "
                "historical.db for the given teams"
                + (f" on {req.game_date}" if req.game_date else "")
                + ". Check team name spelling."
            ),
        )

    # ── 4. Build response ─────────────────────────────────────────────────────
    ev = prob * 1.909 - (1 - prob)   # assumes standard -110 (1.909 dec)
    confidence = (
        "HIGH"   if prob >= 0.60 or prob <= 0.40 else
        "MEDIUM" if prob >= 0.55 or prob <= 0.45 else
        "LOW"
    )
    return {
        "sport":         sport,
        "home_team":     req.home_team,
        "away_team":     req.away_team,
        "game_date":     req.game_date,
        "cover_prob":    round(prob * 100, 2),
        "ev_at_110":     round(ev * 100, 2),
        "confidence":    confidence,
        "recommendation": (
            "COVER"    if prob >= 0.55 else
            "NO COVER" if prob <= 0.45 else
            "LEAN"
        ),
        "model_used":    f"{sport} ATS sub-model",
        "model_auc":     meta.get("auc"),
        "model_saved_at": meta.get("saved_at"),
        "n_features":    meta.get("n_features"),
    }

@app.get("/api/model/stats")
def model_stats(db: Session = Depends(get_db)):
    """Validated model metrics for the dashboard widget."""
    return ml.get_model_stats(db)


@app.get("/api/health/clv-summary")
def health_clv_summary(days: int = Query(30, ge=7, le=90), db: Session = Depends(get_db)):
    """
    Closing Line Value summary for the last `days` days.

    CLV sign convention (American odds):
      clv_cents = close_odds − open_odds
      clv_cents < 0  →  open_odds > close_odds  →  POSITIVE CLV (beat the close)
      clv_cents > 0  →  open_odds < close_odds  →  NEGATIVE CLV (line moved against us)

    Closing line proxy = latest alt_lines snapshot before game start,
    populated from the existing morning + afternoon odds fetch
    (zero additional credit cost).
    """
    return mb_module.get_clv_summary(db, days=days)


@app.post("/api/admin/backfill-clv-opening")
def admin_backfill_clv_opening(db: Session = Depends(get_db)):
    """
    One-time backfill of open_odds for all legs that have fixture_id
    but no open_odds stored.  Safe to call repeatedly (idempotent).
    Uses alt_lines (afternoon fetch data) as opening proxy.
    """
    return mb_module.backfill_clv_opening(db)


@app.get("/api/health/model-status")
def health_model_status(db: Session = Depends(get_db)):
    """
    Current model version, training metadata, live 7-day holdout performance,
    next scheduled retrain time, and version history (last 10 retrains).
    """
    import weekly_retrain as _wr
    return _wr.get_model_status(db)


@app.post("/api/model/rollback")
def model_rollback(
    target_version: int = None,
    db: Session = Depends(get_db),
):
    """
    Roll back to a previous model version.

    ?target_version=N rolls back to model_vN.pkl.
    Omit the parameter to roll back to the most recent prior version.

    Restores model.pkl / scaler.pkl / imputer.pkl from the versioned backup
    and marks the restored version as 'active' in model_versions.
    """
    import weekly_retrain as _wr
    return _wr.rollback_model(db, target_version_n=target_version)


@app.post("/api/model/approve-candidate")
def approve_candidate(db: Session = Depends(get_db)):
    """
    Approve the held candidate model and deploy it as active.
    Weekly retrain now produces candidates on hold — this endpoint
    is the human gate that triggers deployment.
    """
    import weekly_retrain as _wr
    return _wr.approve_candidate(db)


@app.post("/api/model/reject-candidate")
def reject_candidate(db: Session = Depends(get_db)):
    """
    Reject the held candidate model. Current model stays active.
    """
    import weekly_retrain as _wr
    return _wr.reject_candidate(db)


@app.get("/api/model/progress")
def model_progress(db: Session = Depends(get_db)):
    """
    Return model performance snapshots from model_performance_log.

    Response shape:
      {
        "current_week": { tier1_bets, tier1_wr, tier1_pnl,
                          tier2_bets, tier2_wr, tier2_pnl,
                          tier3_bets, tier3_wr, tier3_pnl,
                          gates_passed, ... },
        "history": [ ...same shape, newest first, up to 52 rows... ]
      }

    Returns {"current_week": null, "history": []} before first snapshot.
    Written by signal_analysis.py — this endpoint is read-only.
    """
    import json as _json
    from sqlalchemy import text as _mp_text

    # Check if table exists
    tbl = db.execute(_mp_text("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='model_performance_log'
    """)).fetchone()
    if not tbl:
        return {"current_week": None, "history": []}

    raw_result = db.execute(_mp_text("""
        SELECT * FROM model_performance_log
        ORDER BY week_ending DESC
        LIMIT 52
    """)).fetchall()
    raw_rows = []
    for r in raw_result:
        row = dict(r._mapping)
        if row.get("sport_breakdown"):
            try:
                row["sport_breakdown"] = _json.loads(row["sport_breakdown"])
            except Exception:
                row["sport_breakdown"] = {}
        raw_rows.append(row)

    def _shape(r: dict) -> dict:
        return {
            "week_ending":       r.get("week_ending"),
            "computed_at":       r.get("computed_at"),
            # Tier 1 — PM batch + top picks page
            "tier1_bets":        r.get("t1_n"),
            "tier1_wins":        r.get("t1_wins"),
            "tier1_wr":          r.get("t1_win_rate"),
            "tier1_pnl":         r.get("t1_pnl"),
            # Tier 2 — morning prospective picks
            "tier2_bets":        r.get("t2_n"),
            "tier2_wins":        r.get("t2_wins"),
            "tier2_wr":          r.get("t2_win_rate"),
            "tier2_pnl":         r.get("t2_pnl"),
            # Tier 3 — forced_generation + retroactive_mock
            "tier3_bets":        r.get("t3_n"),
            "tier3_wins":        r.get("t3_wins"),
            "tier3_wr":          r.get("t3_win_rate"),
            "tier3_pnl":         r.get("t3_pnl"),
            # Overall (T1+T2 only)
            "total_n":           r.get("total_n"),
            "overall_wr":        r.get("overall_win_rate"),
            "overall_pnl":       r.get("overall_pnl"),
            # Sport breakdown
            "sport_breakdown":   r.get("sport_breakdown", {}),
            # Signal correlations
            "corr_lqs_won":      r.get("corr_lqs_won"),
            "corr_wp_won":       r.get("corr_wp_won"),
            "corr_ev_won":       r.get("corr_ev_won"),
            "n_corr":            r.get("n_corr"),
            # Trust gates: gates_passed = number of checks that pass (out of 5 defined)
            "gates_passed":      r.get("trust_gate_passed"),
            "trust_gate_reason": r.get("trust_gate_reason"),
            "lqs_monotone":      r.get("lqs_monotone"),
        }

    shaped = [_shape(r) for r in raw_rows]
    return {
        "current_week": shaped[0] if shaped else None,
        "history":      shaped,
    }


@app.get("/api/analysis/regime")
def analysis_regime(db: Session = Depends(get_db)):
    """
    Today's market regime classification + last 14 days history.

    Regime is written by the scheduler at 8:30 AM CT (regime_classification job).
    This endpoint is read-only — it queries market_regime_log directly.
    weights_applied is always False (Phase 2 feature).
    """
    import json as _json
    import sqlite3 as _sql3
    from database import DB_PATH

    con = _sql3.connect(DB_PATH)
    con.row_factory = _sql3.Row
    cur = con.cursor()

    # Ensure table exists (idempotent)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_regime_log (
            date                      TEXT PRIMARY KEY,
            regime                    TEXT,
            legs_evaluated            INTEGER,
            legs_positive_ev          INTEGER,
            pos_ev_pct                REAL,
            hq_legs                   INTEGER,
            weighted_model_confidence REAL,
            avg_implied_prob          REAL,
            model_breakdown           TEXT,
            suggested_weights         TEXT,
            actual_weights            TEXT,
            mock_bets_generated       INTEGER,
            mock_win_rate             REAL,
            mock_pnl                  REAL,
            clv_avg                   REAL,
            note                      TEXT,
            created_at                TEXT
        )
    """)
    con.commit()

    today = str(__import__("datetime").date.today())

    # Today's row
    cur.execute("SELECT * FROM market_regime_log WHERE date = ?", (today,))
    today_row = cur.fetchone()

    def _parse_row(r) -> dict:
        d = dict(r)
        for field in ("model_breakdown", "suggested_weights", "actual_weights"):
            raw = d.get(field)
            if raw:
                try:
                    d[field] = _json.loads(raw)
                except Exception:
                    pass
        return d

    today_data: dict | None = None
    if today_row:
        today_data = _parse_row(today_row)
        today_data["weights_applied"] = False
    else:
        today_data = {
            "date":    today,
            "regime":  "pending",
            "note":    "Regime classification runs at 8:30 AM CT — check back then",
            "weights_applied": False,
        }

    # History: last 14 days (excluding today)
    cur.execute("""
        SELECT date, regime, mock_win_rate, mock_pnl, weighted_model_confidence,
               legs_evaluated, legs_positive_ev, pos_ev_pct, suggested_weights,
               clv_avg, note
        FROM market_regime_log
        WHERE date < ?
        ORDER BY date DESC
        LIMIT 14
    """, (today,))
    history = []
    for row in cur.fetchall():
        h = _parse_row(row)
        h["weights_applied"] = False
        history.append(h)

    con.close()
    return {
        "today":    today_data,
        "history":  history,
        "weights_applied": False,
        "note":     "Informational only — suggested weights are NOT applied to picks",
    }


@app.get("/api/analysis/regime-ab")
def analysis_regime_ab(db: Session = Depends(get_db)):
    """
    Regime A/B test results: standard composite weights vs regime-suggested weights.

    flip_ready = True when:
      - ≥ 20 days logged
      - ≥ 2 regimes have settled win_rate data
      - regime_wr > standard_wr + 5pp on the same days
      - advantage holds across 3+ consecutive weeks
    """
    import json as _json
    import sqlite3 as _sq
    from database import DB_PATH

    con = _sq.connect(DB_PATH)
    con.row_factory = _sq.Row

    # Ensure table exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS regime_ab_log (
            date TEXT PRIMARY KEY, regime TEXT,
            standard_weights TEXT, standard_pick_ids TEXT,
            standard_win_rate REAL, standard_pnl REAL, standard_n INTEGER,
            regime_weights TEXT, regime_pick_ids TEXT,
            regime_win_rate REAL, regime_pnl REAL, regime_n INTEGER,
            overlap_n INTEGER, only_standard_n INTEGER, only_regime_n INTEGER,
            created_at TEXT
        )
    """)
    con.commit()

    rows = con.execute(
        "SELECT * FROM regime_ab_log ORDER BY date DESC LIMIT 60"
    ).fetchall()
    con.close()

    days_logged    = len(rows)
    regimes_seen   = sorted({r["regime"] for r in rows if r["regime"]})

    # Aggregate by regime
    by_regime: dict = {}
    std_wrs = []
    reg_wrs = []

    for r in rows:
        rn = r["regime"] or "unknown"
        rdata = by_regime.setdefault(rn, {
            "days": 0, "std_wr_sum": 0, "std_n": 0,
            "reg_wr_sum": 0, "reg_n": 0,
            "regime_weights_tested": None,
        })
        rdata["days"] += 1
        if r["standard_win_rate"] is not None:
            rdata["std_wr_sum"] += r["standard_win_rate"]
            rdata["std_n"]      += 1
            std_wrs.append(r["standard_win_rate"])
        if r["regime_win_rate"] is not None:
            rdata["reg_wr_sum"] += r["regime_win_rate"]
            rdata["reg_n"]      += 1
            reg_wrs.append(r["regime_win_rate"])
        if r["regime_weights"] and not rdata["regime_weights_tested"]:
            try:
                rdata["regime_weights_tested"] = _json.loads(r["regime_weights"])
            except Exception:
                pass

    # Build summary per regime
    by_regime_out: dict = {}
    for rn, rdata in by_regime.items():
        std_wr = (rdata["std_wr_sum"] / rdata["std_n"]) if rdata["std_n"] else None
        reg_wr = (rdata["reg_wr_sum"] / rdata["reg_n"]) if rdata["reg_n"] else None
        if std_wr is None or reg_wr is None:
            verdict = "insufficient_data"
        elif reg_wr > std_wr + 0.05:
            verdict = "regime_better"
        elif reg_wr < std_wr - 0.05:
            verdict = "standard_better"
        else:
            verdict = "no_difference"

        by_regime_out[rn] = {
            "days":                   rdata["days"],
            "standard_wr":            round(std_wr, 4) if std_wr is not None else None,
            "regime_wr":              round(reg_wr, 4) if reg_wr is not None else None,
            "regime_weights_tested":  rdata["regime_weights_tested"],
            "verdict":                verdict,
        }

    std_overall = round(sum(std_wrs) / len(std_wrs), 4) if std_wrs else None
    reg_overall = round(sum(reg_wrs) / len(reg_wrs), 4) if reg_wrs else None

    # Flip conditions
    regimes_with_data = sum(
        1 for rdata in by_regime_out.values()
        if rdata["regime_wr"] is not None
    )
    flip_ready = (
        days_logged >= 20
        and regimes_with_data >= 2
        and reg_overall is not None
        and std_overall is not None
        and reg_overall > std_overall + 0.05
    )

    return {
        "days_logged":         days_logged,
        "regimes_seen":        regimes_seen,
        "standard_overall_wr": std_overall,
        "regime_overall_wr":   reg_overall,
        "by_regime":           by_regime_out,
        "flip_ready":          flip_ready,
        "flip_conditions": {
            "min_days":       f"20 needed, {days_logged} logged",
            "min_regimes":    f"2 needed with outcomes, {regimes_with_data} have data",
            "min_advantage":  "regime_wr > standard_wr + 5pp",
            "consistency":    "advantage must hold 3+ consecutive weeks",
        },
    }


class DiagnosticModelRequest(BaseModel):
    sport: str = "NBA"          # "NBA", "NFL", "MLB", …
    save:  bool = False         # persist if AUC >= 0.55


@app.post("/api/model/train-diagnostic-sport")
def train_diagnostic_sport(req: DiagnosticModelRequest):
    """
    DIAGNOSTIC path — train a per-sport model using rolling team stats only.
    Does NOT touch the production model.pkl or affect live recommendations.
    Returns AUC, calibration, top features, and a WORTH BUILDING / WAIT / FEATURE WORK decision.
    """
    return ml.train_diagnostic_sport_model(req.sport.upper(), save=req.save)


# ─── Cash-out routes (Phase 6B) ───────────────────────────────────────────────

class LegStatusItem(BaseModel):
    leg_id:          str
    status:          str            # "won"|"lost"|"pending"|"in_game"|"at_risk"
    updated_prob:    float          # current win probability (0–1)
    at_risk_flag:    bool  = False
    at_risk_reason:  str   = ""
    original_prob:   Optional[float] = None   # prob at placement; optional

class CashOutCheckRequest(BaseModel):
    current_offer_amount:   float
    remaining_legs_status:  List[LegStatusItem]
    user_doubt_signal:      bool = False
    concern_text:           str  = ""    # free-text from "What's the concern?" field

class MarkCashedOutRequest(BaseModel):
    cash_out_amount: float
    timestamp:       Optional[str] = None   # ISO string; defaults to utcnow
    user_action:     str = "followed"       # "followed"|"cashed_against_suggestion"

class LogHoldDecisionRequest(BaseModel):
    user_action: str   # "overrode_red"|"followed"


@app.post("/api/bets/{bet_id}/cash-out-check")
def cash_out_check(
    bet_id: str,
    req:    CashOutCheckRequest,
    db:     Session = Depends(get_db),
):
    """
    Evaluate a sportsbook cash-out offer against current leg probabilities.
    Appends this check to cash_out_offers_log; does NOT settle the bet.
    """
    from sqlalchemy import text as _text

    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if bet is None:
        raise HTTPException(status_code=404, detail="Bet not found")

    bet_dict = {
        "original_stake": bet.amount,
        "original_odds":  bet.odds,
        "legs":           bet.legs,
    }
    legs = [l.dict() for l in req.remaining_legs_status]
    result = co_engine.evaluate_cash_out(
        bet_dict, req.current_offer_amount, legs, req.user_doubt_signal
    )

    # Append to cash_out_offers_log (JSON array)
    existing_raw = getattr(bet, "cash_out_offers_log", None)
    try:
        log_list = json.loads(existing_raw) if existing_raw else []
    except Exception:
        log_list = []

    log_entry = {
        "timestamp":        datetime.utcnow().isoformat(),
        "offer_amount":     req.current_offer_amount,
        "recommendation":   result["recommendation"],
        "ev_hold":          result["ev_hold"],
        "ev_cash_out":      result["ev_cash_out"],
        "vig_pct":          result["cash_out_vig_pct"],
        "combined_prob":    result["combined_remaining_prob"],
        "legs_status":      legs,
        "user_doubt":       req.user_doubt_signal,
        "concern_text":     req.concern_text.strip() if req.concern_text else "",
        # user_action filled in later by mark-cashed-out or log-hold-decision
        "user_action":      None,
    }
    log_list.append(log_entry)

    try:
        db.execute(
            _text("UPDATE bets SET cash_out_offers_log = :v WHERE id = :id"),
            {"v": json.dumps(log_list), "id": bet_id},
        )
        db.commit()
    except Exception:
        pass   # column may be missing on first run before migration; non-fatal

    return {"bet_id": bet_id, **result}


@app.post("/api/bets/{bet_id}/mark-cashed-out")
def mark_cashed_out(
    bet_id: str,
    req:    MarkCashedOutRequest,
    db:     Session = Depends(get_db),
):
    """
    Record that the user accepted the cash-out offer.
    Does NOT change bet status to SETTLED — auto-settle still runs so we can
    compute the retrospective cash_out_vs_final_decision verdict later.
    """
    from sqlalchemy import text as _text

    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if bet is None:
        raise HTTPException(status_code=404, detail="Bet not found")

    ts = req.timestamp or datetime.utcnow().isoformat()

    # Backfill user_action on the most recent log entry
    existing_raw = getattr(bet, "cash_out_offers_log", None)
    try:
        log_list = json.loads(existing_raw) if existing_raw else []
        if log_list:
            log_list[-1]["user_action"] = req.user_action
        updated_log = json.dumps(log_list) if log_list else existing_raw
    except Exception:
        updated_log = existing_raw

    try:
        db.execute(
            _text("""
                UPDATE bets
                SET cashed_out              = 1,
                    cash_out_amount         = :amount,
                    cash_out_timestamp      = :ts,
                    cash_out_offers_log     = :log
                WHERE id = :id
            """),
            {"amount": req.cash_out_amount, "ts": ts,
             "log": updated_log, "id": bet_id},
        )
        db.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "bet_id":           bet_id,
        "cashed_out":       True,
        "cash_out_amount":  req.cash_out_amount,
        "cash_out_timestamp": ts,
        "message": (
            "Cash out recorded. Bet remains PLACED so auto-settle can "
            "determine the retrospective verdict after game outcomes resolve."
        ),
    }


@app.post("/api/bets/{bet_id}/log-hold-decision")
def log_hold_decision(
    bet_id: str,
    req:    LogHoldDecisionRequest,
    db:     Session = Depends(get_db),
):
    """
    Record the user's hold decision on the most recent cash-out evaluation.
    Called when user clicks 'Hold Anyway' (overrode_red) or 'Acknowledge' (followed).
    Updates user_action on the last entry in cash_out_offers_log.
    """
    from sqlalchemy import text as _text

    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if bet is None:
        raise HTTPException(status_code=404, detail="Bet not found")

    existing_raw = getattr(bet, "cash_out_offers_log", None)
    try:
        log_list = json.loads(existing_raw) if existing_raw else []
    except Exception:
        log_list = []

    if not log_list:
        return {"bet_id": bet_id, "updated": False, "reason": "no log entries"}

    log_list[-1]["user_action"] = req.user_action

    try:
        db.execute(
            _text("UPDATE bets SET cash_out_offers_log = :v WHERE id = :id"),
            {"v": json.dumps(log_list), "id": bet_id},
        )
        db.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"bet_id": bet_id, "updated": True, "user_action": req.user_action}


from sqlalchemy import text as _sqla_text

# ─── Live Bet Monitor Endpoints ───────────────────────────────────────────────

def _build_live_response(db: Session, auto_settle: bool = True) -> list[dict]:
    """Resolve all PLACED bets and attach cashout advice."""
    resolved = lm.resolve_placed_bets(db, auto_settle=auto_settle)
    output: list[dict] = []
    for item in resolved:
        bet_outcome = item.get("bet_outcome")
        # Fetch current Bet row (status may have changed if auto-settled)
        bet_row = db.query(Bet).filter(Bet.id == item["bet_id"]).first()
        if bet_row is None:
            continue
        cashout = None
        if bet_outcome == "IN_PROGRESS" and not getattr(bet_row, "cashed_out", False):
            cashout = lm.evaluate_cashout(bet_row, item["resolved_legs"], db)
            lm.log_cashout_recommendation(bet_row, cashout, db)
        output.append({**item, "cashout": cashout})
    return output


@app.get("/api/bets/live")
def get_live_bets(db: Session = Depends(get_db)):
    """
    Return all PLACED bets with live resolution and cash-out advice.
    Auto-settles bets whose outcome is now clear.
    """
    return _build_live_response(db, auto_settle=True)


@app.post("/api/bets/live/refresh")
def refresh_live_bets(db: Session = Depends(get_db)):
    """Force re-resolve all PLACED bets (manual trigger)."""
    return _build_live_response(db, auto_settle=True)


class LiveCashOutRequest(BaseModel):
    amount: float


@app.patch("/api/bets/{bet_id}/cashout")
def live_mark_cashed_out(
    bet_id: str,
    req:    LiveCashOutRequest,
    db:     Session = Depends(get_db),
):
    """
    Record that the user cashed out a live bet manually at the given amount.
    Updates cashed_out=True, cash_out_amount, status stays PLACED until
    auto-settle retroactively determines the cash_out_vs_final_decision verdict.
    """
    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if bet is None:
        raise HTTPException(status_code=404, detail="Bet not found")

    ts = datetime.utcnow().isoformat()
    profit = round(req.amount - (bet.amount or 0), 2)

    # Compare cash-out amount against stored placement-time target
    stored_target = getattr(bet, "cash_out_target", None)
    if stored_target and stored_target > 0:
        ratio = req.amount / stored_target
        if ratio >= 1.0:
            target_verdict = "Met target"
        elif ratio >= 0.90:
            target_verdict = "Within 10% of target"
        elif ratio >= 0.75:
            target_verdict = "Below but acceptable"
        else:
            target_verdict = "Significantly below target"
    else:
        target_verdict = None

    try:
        db.execute(
            _sqla_text("""
                UPDATE bets
                SET cashed_out=1, cash_out_amount=:amt,
                    cash_out_timestamp=:ts, status='SETTLED_WIN', profit=:profit
                WHERE id=:id
            """),
            {"amt": req.amount, "ts": ts, "profit": profit, "id": bet_id},
        )
        db.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    resp: dict = {
        "bet_id":             bet_id,
        "cashed_out":         True,
        "cash_out_amount":    req.amount,
        "cash_out_timestamp": ts,
        "profit":             profit,
        "message":            "Cash out recorded. Retrospective verdict will be set after games finish.",
    }
    if stored_target is not None:
        resp["cash_out_target"]  = stored_target
        resp["target_verdict"]   = target_verdict
    return resp


@app.post("/api/bets/backfill-cashout-targets")
def backfill_cashout_targets(db: Session = Depends(get_db)):
    """
    Retroactively compute and store cash_out_target for existing PLACED bets
    that don't yet have a target set.
    """
    bets = db.query(Bet).filter(
        Bet.is_mock.is_(False),
        Bet.status == "PLACED",
        Bet.cash_out_target.is_(None),  # type: ignore[attr-defined]
    ).all()

    updated = 0
    for bet in bets:
        try:
            n = int(bet.legs or 1) if str(bet.legs or "").isdigit() else 1
            target = lm.compute_cashout_target(
                amount  = float(bet.amount or 10),
                odds    = float(bet.odds or 1),
                n_legs  = n,
                avg_lqs = 65.0,
            )
            db.execute(
                _sqla_text("""
                    UPDATE bets
                    SET cash_out_target=:t, cash_out_target_pct=:p,
                        cash_out_target_rationale=:r
                    WHERE id=:id
                """),
                {"t": target["target_amount"], "p": target["target_pct"],
                 "r": target["rationale"], "id": bet.id},
            )
            updated += 1
        except Exception:
            pass

    db.commit()
    return {"backfilled": updated, "message": f"Set cash-out targets for {updated} existing bets."}


@app.get("/api/analytics/cash-out-performance")
def cash_out_performance(
    days: int = Query(90, ge=1, le=730),
    db:   Session = Depends(get_db),
):
    """
    Aggregate cash-out decision performance for the last *days* days.
    Only shown in the UI after 5+ cash-outs are logged.
    """
    from sqlalchemy import text as _text
    from datetime import timedelta

    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    try:
        rows = db.execute(_text("""
            SELECT
                cashed_out,
                cash_out_amount,
                amount,
                odds,
                cash_out_vs_final_decision,
                cash_out_offers_log,
                cash_out_timestamp
            FROM bets
            WHERE cashed_out = 1
              AND (cash_out_timestamp IS NULL OR cash_out_timestamp >= :since)
        """), {"since": since}).fetchall()
    except Exception:
        return {"error": "cash_out columns not yet migrated — restart the server"}

    total         = len(rows)
    wisely        = 0
    unnecessarily = 0
    saved_amounts: list[float]   = []
    forgone_amounts: list[float] = []

    # Segmented by what the recommendation was at time of cash-out
    red_cashed_wisely        = 0
    red_cashed_unnecessarily = 0
    yellow_cashed_wisely     = 0
    yellow_cashed_unnecessarily = 0

    for row in rows:
        cash_amount = row[1] or 0.0
        orig_stake  = row[2] or 0.0
        orig_odds   = row[3] or 1.0
        verdict     = row[4]
        log_raw     = row[5]
        full_payout = orig_stake * orig_odds

        if verdict == "cashed_wisely":
            wisely += 1
            saved_amounts.append(round(cash_amount, 2))
        elif verdict == "cashed_unnecessarily":
            unnecessarily += 1
            forgone_amounts.append(round(full_payout - cash_amount, 2))

        # Segment by banner colour at time of cash-out decision
        if verdict is not None:
            try:
                log_list = json.loads(log_raw) if log_raw else []
                # Find last entry where user_action indicates they cashed
                cashed_entry = next(
                    (e for e in reversed(log_list)
                     if e.get("user_action") in ("followed", "cashed_against_suggestion", None)),
                    log_list[-1] if log_list else None,
                )
                last_rec = cashed_entry.get("recommendation", "") if cashed_entry else ""
                if last_rec == "CASH_OUT_RECOMMENDED":
                    if verdict == "cashed_wisely":        red_cashed_wisely += 1
                    else:                                  red_cashed_unnecessarily += 1
                elif last_rec == "CASH_OUT_OK":
                    if verdict == "cashed_wisely":        yellow_cashed_wisely += 1
                    else:                                  yellow_cashed_unnecessarily += 1
            except Exception:
                pass

    # ── Held-through-red stats: settled non-cashed bets with annotated log entries ──
    # The backfill in auto_settle._backfill_cash_out_decisions annotates
    # final_outcome on CASH_OUT_RECOMMENDED entries when bets settle.
    try:
        held_rows = db.execute(_text("""
            SELECT status, cash_out_offers_log
            FROM bets
            WHERE cashed_out = 0
              AND status IN ('SETTLED_WIN', 'SETTLED_LOSS')
              AND cash_out_offers_log IS NOT NULL
              AND (time_settled IS NULL OR time_settled >= :since)
        """), {"since": since}).fetchall()
    except Exception:
        held_rows = []

    held_red_won  = 0
    held_red_lost = 0
    held_red_override_won  = 0   # explicitly overrode red, then won
    held_red_override_lost = 0   # explicitly overrode red, then lost

    for row in held_rows:
        status  = row[0]
        log_raw = row[1]
        try:
            log_list = json.loads(log_raw) if log_raw else []
            for entry in log_list:
                if entry.get("recommendation") != "CASH_OUT_RECOMMENDED":
                    continue
                action = entry.get("user_action")
                if action not in ("overrode_red", "followed", None):
                    continue   # user cashed; skip
                # This was a hold on a red banner
                outcome = "won" if status == "SETTLED_WIN" else "lost"
                if outcome == "won":
                    held_red_won += 1
                else:
                    held_red_lost += 1
                if action == "overrode_red":
                    if outcome == "won":  held_red_override_won  += 1
                    else:                 held_red_override_lost += 1
        except Exception:
            pass

    held_red_total = held_red_won + held_red_lost

    # ── Summaries ──────────────────────────────────────────────────────────────
    avg_saved   = round(sum(saved_amounts)   / len(saved_amounts),   2) if saved_amounts   else None
    avg_forgone = round(sum(forgone_amounts) / len(forgone_amounts), 2) if forgone_amounts else None
    net_impact  = round(sum(saved_amounts) - sum(forgone_amounts), 2) \
                  if (saved_amounts or forgone_amounts) else None

    # Primary rec accuracy: when CASH_OUT_RECOMMENDED was shown and user cashed → wise %
    red_cashed_total = red_cashed_wisely + red_cashed_unnecessarily
    red_cashed_acc   = round(red_cashed_wisely / red_cashed_total * 100, 1) \
                       if red_cashed_total >= 3 else None

    # Held-through-red loss rate (the key validation metric)
    held_red_loss_rate = round(held_red_lost / held_red_total * 100, 1) \
                         if held_red_total >= 3 else None

    return {
        "days":                       days,
        # ── Cash-out totals ──────────────────────────────────────────────────
        "total_cash_outs":            total,
        "verdict_available":          wisely + unnecessarily,
        "cashed_wisely_count":        wisely,
        "cashed_unnecessarily_count": unnecessarily,
        "cash_out_accuracy_pct":      round(wisely / (wisely + unnecessarily) * 100, 1)
                                      if (wisely + unnecessarily) > 0 else None,
        "avg_amount_saved":           avg_saved,
        "avg_amount_left_on_table":   avg_forgone,
        "total_net_impact":           net_impact,
        # ── Red banner cash-out accuracy ─────────────────────────────────────
        "red_cashed_wisely":          red_cashed_wisely,
        "red_cashed_unnecessarily":   red_cashed_unnecessarily,
        "red_recommendation_accuracy_pct": red_cashed_acc,
        # ── Yellow zone cash-out outcomes ────────────────────────────────────
        "yellow_cashed_wisely":       yellow_cashed_wisely,
        "yellow_cashed_unnecessarily":yellow_cashed_unnecessarily,
        # ── Held-through-red outcomes (primary validation signal) ────────────
        "held_red_total":             held_red_total,
        "held_red_won":               held_red_won,
        "held_red_lost":              held_red_lost,
        "held_red_loss_rate_pct":     held_red_loss_rate,
        "held_red_override_won":      held_red_override_won,
        "held_red_override_lost":     held_red_override_lost,
        # ── Widget visibility ────────────────────────────────────────────────
        "show_widget":                total >= 5 or held_red_total >= 3,
    }


# ─── Cashout live monitor (Phase 6C) ────────────────────────────────────────

@app.get("/api/cashout/live")
def cashout_live_dashboard():
    """
    Live cashout dashboard for open LIVE bets.
    Returns per-bet fair value, spread position, and recommendation band.
    No FanDuel integration — this is a fair-value estimator only.
    Compare BetIQ's fair value to the offer in the FD app.
    """
    try:
        dashboard = co_engine.get_cashout_dashboard()
        return {
            "open_bets":   len(dashboard),
            "bets":        dashboard,
            "poll_status": sched.get_state().get("last_cashout_poll_result"),
            "last_poll":   sched.get_state().get("last_cashout_poll"),
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"error": str(exc), "open_bets": 0, "bets": []}


@app.post("/api/cashout/poll")
def cashout_poll_now():
    """Force an immediate live-odds poll for cashout monitoring."""
    try:
        result = co_engine.poll_live_odds()
        return result
    except Exception as exc:
        return {"error": str(exc)}


# ─── Fixtures / odds routes ───────────────────────────────────────────────────

@app.get("/api/fixtures")
def list_fixtures(
    sport: Optional[str] = None,
    limit: int = 50,
    include_past: bool = False,
    db: Session = Depends(get_db)
):
    from datetime import datetime, timezone as _tz
    q = db.query(Fixture).order_by(Fixture.commence_time)
    if sport:
        q = q.filter(Fixture.sport_key.contains(sport))
    if not include_past:
        now = datetime.now(_tz.utc).replace(tzinfo=None)
        q = q.filter(Fixture.commence_time >= now)
    fixtures = q.limit(limit).all()
    return [{
        "id":           f.id,
        "sport":        f.sport_title,
        "sport_key":    f.sport_key,
        "home_team":    f.home_team,
        "away_team":    f.away_team,
        "commence_time":f.commence_time.isoformat() if f.commence_time else None,
        "bookmakers":   f.bookmakers or [],
    } for f in fixtures]

@app.get("/api/fixtures/recommend")
def fixture_recommendations(
    min_ev: float = -0.1,
    db: Session = Depends(get_db)
):
    """Return upcoming fixtures with model EV scores, sorted best-first."""
    results = oapi.fixtures_with_ev(db)
    return [r for r in results if r["ev"] >= min_ev]

@app.post("/api/fixtures/refresh")
def refresh_fixtures(
    sports: Optional[List[str]] = None,
    include_pitcher_fetch: bool = False,
    db: Session = Depends(get_db)
):
    """
    Pull latest odds from TheOddsAPI and upsert into fixtures table.

    Also optionally fetches today's MLB probable pitchers from the free MLB
    Stats API (include_pitcher_fetch=true) and invalidates the ML model cache
    so the next pick generation uses fresh data.

    Returns: { new, updated, sports_fetched, sports, pitcher_fetch? }
    """
    result = oapi.fetch_all_fixtures(db, sport_keys=sports)

    # New fixture odds → cached picks are stale; force regeneration on next call.
    invalidate_picks_cache()

    if include_pitcher_fetch:
        try:
            from historical_etl import upsert_mlb_today_pitchers
            pitcher_result = upsert_mlb_today_pitchers()
            result["pitcher_fetch"] = pitcher_result
        except Exception as exc:
            result["pitcher_fetch"] = {"error": str(exc)}

    # After fresh odds arrive, re-run exploration bets idempotently.
    # This catches the common case where alt_lines for a sport (e.g. MLB) arrive
    # after the morning generation run, so exploration legs that had no odds get
    # a second chance once the data is populated.
    try:
        expl = mb_module.generate_exploration_bets(db)
        if expl.get("created", 0) > 0:
            result["exploration_topped_up"] = expl
    except Exception as _expl_err:
        result["exploration_topped_up"] = {"error": str(_expl_err)}

    return result


@app.post("/api/mlb/pitcher-fetch")
def mlb_pitcher_fetch(db: Session = Depends(get_db)):
    """
    Fetch today's MLB probable pitchers from the free MLB Stats API and upsert
    into historical.db. Invalidates the ML model feature-matrix cache so the
    next pick generation builds pitcher context into the feature matrix.

    Safe to call multiple times — upserts only, no duplicates.
    Returns: { date, games_found, games_written, stats_written }
    """
    try:
        from historical_etl import upsert_mlb_today_pitchers
        return upsert_mlb_today_pitchers()
    except Exception as exc:
        return {"error": str(exc)}


# ─── Parlay builder routes (Phase 2A) ────────────────────────────────────────

class OptimizeRequest(BaseModel):
    parlay_size:    int          = 4
    n_results:      int          = 5
    stake:          float        = 10.0
    min_leg_grade:  str          = "C"       # A/B/C/D/F
    markets:        Optional[List[str]] = None   # ["h2h","spreads","totals"]
    sport_filter:   Optional[List[str]] = None   # ["NBA","MLB"]
    max_same_game:  int          = 1
    min_odds:       float        = 1.0       # min decimal odds per leg

class CustomParlayRequest(BaseModel):
    leg_ids: List[str]    # list of leg_id strings from /api/parlay/legs
    stake:   float = 10.0

@app.get("/api/parlay/legs")
def parlay_legs(
    markets:      Optional[str] = None,   # comma-separated: h2h,spreads,totals,alternate_spreads,alternate_totals
    sport_filter: Optional[str] = None,   # comma-separated: NBA,MLB
    db: Session = Depends(get_db)
):
    """
    Return all scoreable legs from upcoming fixtures, each with
    model win_prob, EV, edge, and grade.
    """
    mkt_list    = markets.split(",")      if markets      else ["h2h", "spreads", "totals"]
    sport_list  = sport_filter.split(",") if sport_filter else None
    raw = pb.get_available_legs(db, markets=mkt_list)
    if sport_list:
        raw = [l for l in raw if l["sport"] in sport_list]
    scored = [pb.score_leg(l) for l in raw]
    scored.sort(key=lambda x: -x["ev"])
    return {"legs": scored, "total": len(scored)}

@app.post("/api/parlay/optimize")
def optimize_parlay(req: OptimizeRequest, db: Session = Depends(get_db)):
    """
    Automatically find the highest-EV parlays of the requested size
    from all available legs.
    """
    return pb.optimize_parlays(
        db,
        n_results    = req.n_results,
        parlay_size  = req.parlay_size,
        stake        = req.stake,
        min_leg_grade= req.min_leg_grade,
        markets      = req.markets,
        sport_filter = req.sport_filter,
        max_same_game= req.max_same_game,
        min_odds     = req.min_odds,
    )

@app.post("/api/parlay/custom")
def custom_parlay(req: CustomParlayRequest, db: Session = Depends(get_db)):
    """
    Score a user-assembled parlay from a list of leg_ids.
    Returns combined odds, win prob, EV, and correlation warnings.
    """
    all_legs = pb.get_available_legs(db)
    leg_map  = {l["leg_id"]: l for l in all_legs}

    selected = []
    missing  = []
    for lid in req.leg_ids:
        if lid in leg_map:
            selected.append(pb.score_leg(leg_map[lid]))
        else:
            missing.append(lid)

    if not selected:
        raise HTTPException(400, "No valid leg_ids found. Refresh fixtures first.")

    result = pb.build_parlay(selected, stake=req.stake)
    if missing:
        result["warnings"] = result.get("warnings", []) + [
            f"Leg IDs not found (fixture may have started): {missing}"
        ]
    return result

@app.post("/api/parlay/place")
def place_parlay(
    req: CustomParlayRequest,
    is_mock: bool = True,
    sportsbook: str = "FanDuel",
    db: Session = Depends(get_db)
):
    """
    Score, validate, then record a parlay as a bet (real or mock).
    Returns the full parlay analysis + bet_id for later settlement.
    """
    all_legs = pb.get_available_legs(db)
    leg_map  = {l["leg_id"]: l for l in all_legs}
    selected = [pb.score_leg(leg_map[lid]) for lid in req.leg_ids if lid in leg_map]

    if not selected:
        raise HTTPException(400, "No valid legs found.")

    parlay   = pb.build_parlay(selected, stake=req.stake)
    bet_info = " | ".join(f"{l['description']} ({l['game']})" for l in selected)
    sports   = " | ".join({l["sport"] for l in selected})
    leagues  = " | ".join({l["sport"] for l in selected})

    # Compute avg LQS across selected legs (use scored leg lqs field)
    lqs_vals = [l.get("lqs") or l.get("avg_lqs") or 65 for l in selected]
    avg_lqs_val = sum(lqs_vals) / len(lqs_vals) if lqs_vals else 65.0

    target_info = lm.compute_cashout_target(
        amount  = req.stake,
        odds    = parlay["combined_odds"],
        n_legs  = len(selected),
        avg_lqs = avg_lqs_val,
    )

    bet_id = str(uuid.uuid4())
    bet = Bet(
        id                        = bet_id,
        source                    = "app",
        sportsbook                = sportsbook,
        bet_type                  = "parlay",
        status                    = "PLACED",
        odds                      = parlay["combined_odds"],
        amount                    = req.stake,
        legs                      = len(selected),
        sports                    = sports,
        leagues                   = leagues,
        bet_info                  = bet_info,
        is_mock                   = is_mock,
        time_placed               = datetime.utcnow(),
        cash_out_target           = target_info["target_amount"],
        cash_out_target_pct       = target_info["target_pct"],
        cash_out_target_rationale = target_info["rationale"],
    )
    db.add(bet)
    db.commit()

    return {
        "bet_id":          bet_id,
        "is_mock":         is_mock,
        "parlay":          parlay,
        "cash_out_target": target_info,
        "message":         f"{'Mock' if is_mock else 'Real'} parlay recorded. "
                           f"Settle via PUT /api/bets/{bet_id}/settle when results are in."
    }


# ─── Auto-settle routes (Phase 2B) ───────────────────────────────────────────

class SchedulerConfigRequest(BaseModel):
    interval_mins: int  = 30
    auto_retrain:  bool = True

@app.post("/api/settle/run")
def manual_settle(
    days_back:    int  = 3,
    auto_retrain: bool = True,
    db: Session = Depends(get_db)
):
    """
    Trigger a single auto-settle cycle immediately.
    Checks all PLACED bets against OddsAPI scores and settles what it can.
    """
    result = asettler.run_auto_settle(db, days_back=days_back, auto_retrain=auto_retrain)
    return result

@app.get("/api/settle/log")
def settle_log(limit: int = 50, db: Session = Depends(get_db)):
    """Full audit log of all auto-settle attempts."""
    return asettler.get_settle_log(db, limit=limit)

@app.get("/api/settle/scheduler")
def scheduler_status():
    """Current state of the background scheduler."""
    return sched.get_state()


def _scout_health(state: dict, today_ct_str: str, db) -> dict:
    """Compute scout pipeline health for the health endpoint."""
    ran_today = (state.get("last_scout_run") or "")[:10] == today_ct_str
    last_result = state.get("last_scout_run_result") or {}
    props_today = 0
    cal_level   = "unknown"
    try:
        from sqlalchemy import text
        row = db.execute(text(
            "SELECT COUNT(*) FROM scouted_props WHERE scout_date = :d"
        ), {"d": today_ct_str}).fetchone()
        props_today = row[0] if row else 0
    except Exception:
        pass
    try:
        import scout.calibration as _sc
        from database import engine as _engine
        summary = _sc.latest_scout_calibration_summary(_engine)
        cal_level = (summary or {}).get("alert_level", "no_data")
    except Exception:
        pass
    return {
        "ran_today":       ran_today,
        "props_today":     props_today,
        "by_sport":        last_result.get("by_sport", {}),
        "errors":          last_result.get("errors", []),
        "calibration":     cal_level,
        "last_run_date":   state.get("last_scout_run"),
    }


@app.get("/api/scheduler/health")
def scheduler_health(db: Session = Depends(get_db)):
    """
    Watchdog health check — returns verification state for all 5 critical jobs.
    overall: 'healthy' | 'degraded' | 'critical'
    """
    from zoneinfo import ZoneInfo
    from datetime import datetime as _dt
    _CT = ZoneInfo("America/Chicago")
    now_ct = _dt.now(_CT)
    ts_str = now_ct.strftime("%Y-%m-%d %H:%M CT")

    state = sched.get_state()

    # Run live verification checks (read-only, cheap queries)
    fx_ok,  fx_det  = sched._verify_fixtures_today()
    pit_ok, pit_det = sched._verify_pitcher_today()
    pk_ok,  pk_det  = sched._verify_picks_published()
    mb_ok,  mb_det  = sched._verify_mock_generated()
    st_ok,  st_det  = sched._verify_settle_recent()

    # Parse fixture count
    def _parse_int(s: str, key: str = "") -> int:
        try: return int(s.split("_")[0])
        except: return 0

    # Settlement minutes-ago
    settle_mins: int | None = None
    last_ct = state.get("settle_last_ran_ct")
    if last_ct:
        try:
            elapsed = (now_ct - last_ct).total_seconds()
            settle_mins = int(elapsed / 60)
        except Exception:
            pass

    today_ct_str = now_ct.strftime("%Y-%m-%d")

    # PM job status (store_ts=False → stored as today_ct date string)
    def _pm_status(state_key: str) -> dict:
        ran_today = state.get(state_key) == today_ct_str
        result    = state.get(state_key + "_result") or {}
        return {
            "ran_today":    ran_today,
            "events_found": result.get("events_found") or result.get("generated", 0),
            "rows_or_bets": result.get("rows_inserted") or result.get("generated", 0),
            "skipped":      result.get("skipped_reason"),
            "error":        result.get("error"),
        }

    checks = {
        "fixture_refresh": {
            "verified":          fx_ok,
            "fixtures_today":    _parse_int(fx_det),
            "detail":            fx_det,
            "fixture_age_hours": sched.fixture_staleness_hours(),
            "fixtures_stale":    (lambda age: age is not None and age > 8.0)(sched.fixture_staleness_hours()),
        },
        "pitcher_data": {
            "verified":        pit_ok,
            "teams_with_starters": _parse_int(pit_det),
            "detail":          pit_det,
        },
        "picks_published": {
            "verified":    pk_ok,
            "picks_count": _parse_int(pk_det),
            "detail":      pk_det,
        },
        "mock_generated": {
            "verified":   mb_ok,
            "bets_today": _parse_int(mb_det),
            "detail":     mb_det,
        },
        "settlement": {
            "last_ran_ct":       last_ct.strftime("%Y-%m-%d %H:%M CT") if last_ct else None,
            "minutes_ago":       settle_mins,
            "overdue":           not st_ok,
            "detail":            st_det,
            "next_scheduled_ct": sched._next_settle_marks_str()[0],
        },
        # PM jobs — informational (not in critical_checks)
        "alt_lines_morning":   _pm_status("last_alt_lines_fetch_morning"),
        "alt_lines_afternoon": _pm_status("last_alt_lines_fetch_afternoon"),
        "mock_generate_pm":    _pm_status("last_mock_generate_pm"),
        # Scout pipeline — informational
        "scout": _scout_health(state, today_ct_str, db),
    }

    # Derive overall status
    from auto_settle import RETRAIN_STATE as _RS
    retrain_status = _RS.get("status", "idle")

    critical_checks = [fx_ok, pk_ok, mb_ok]
    degraded_checks = [pit_ok, st_ok]
    if not all(critical_checks):
        overall = "critical"
    elif not all(degraded_checks) or retrain_status == "failed":
        overall = "degraded"
    else:
        overall = "healthy"

    # Elapsed hours for running retrain
    retrain_elapsed = None
    if retrain_status == "running" and _RS.get("started_ct"):
        try:
            from datetime import datetime as _dt2, timezone as _tz2
            _started = _dt2.fromisoformat(_RS["started_ct"]).astimezone(_tz2.utc)
            retrain_elapsed = round((_dt2.now(_tz2.utc) - _started).total_seconds() / 3600, 2)
        except Exception:
            pass

    # ── API credit snapshot ────────────────────────────────────────────────────
    _credit_data: dict = {}
    try:
        from creator_tier import get_credit_status as _gcs
        _cs = _gcs()
        _remaining   = _cs.get("credits_remaining") or 0
        _burn        = _cs.get("burn_rate_daily", 0.0)
        _days_left   = _cs.get("days_until_reset", 1)
        _proj_eoc    = round(_remaining - (_burn * _days_left))
        if _proj_eoc > 5_000:
            _credit_status = "healthy"
        elif _proj_eoc >= 1_000:
            _credit_status = "caution"
        else:
            _credit_status = "critical"
        _credit_data = {
            "remaining":               _remaining,
            "used_today":              _cs.get("credits_used_today", 0),
            "daily_burn_7d_avg":       _burn,
            "days_remaining_in_cycle": _days_left,
            "projected_eoc_remaining": _proj_eoc,
            "status":                  _credit_status,
            "budget_status":           _cs.get("status", "unknown"),
        }
    except Exception as _ce:
        _credit_data = {"error": str(_ce)}

    return {
        "timestamp_ct": ts_str,
        "checks":        checks,
        "overall":       overall,
        "api_credits":   _credit_data,
        "watchdog_flags": {
            "fixture_verified":   state.get("watchdog_fixture_verified"),
            "pitcher_verified":   state.get("watchdog_pitcher_verified"),
            "picks_verified":     state.get("watchdog_picks_verified"),
            "mocks_verified":     state.get("watchdog_mocks_verified"),
            "settlement_current": settle_mins is not None and settle_mins < 35,
        },
        "settlement_cron": {
            "running":        state.get("settle_cron_running", False),
            "cron_alive":     any(
                t.name == "SettlementCron" and t.is_alive()
                for t in __import__("threading").enumerate()
            ),
            "last_trigger_ct":   state.get("settle_cron_last_trigger_ct"),
            "last_trigger_type": state.get("settle_cron_last_trigger_type"),
            "next_scheduled_ct": sched._next_settle_marks_str()[0],
            "watchdog_next_ct":  sched._next_settle_marks_str()[1],
            "schedule":          ":00 and :30 CT (primary), :05 and :35 CT (watchdog)",
        },
        "retrain": {
            "status":        retrain_status,
            "pid":           _RS.get("pid"),
            "sport":         _RS.get("sport"),
            "started_ct":    _RS.get("started_ct"),
            "completed_ct":  _RS.get("completed_ct"),
            "elapsed_hours": retrain_elapsed,
            "log_path":      _RS.get("log_path"),
            "error":         _RS.get("error"),
        },
    }


@app.get("/api/health/daily-summary")
def health_daily_summary(db: Session = Depends(get_db)):
    """
    Combined daily health snapshot:
      • Scheduler job statuses (from existing health endpoint)
      • Calibration drift across grades and sports
      • Schema migration audit (last N migrations + backup info)
      • DB stats (row counts for key tables)
      • Recent errors (last 5 settle_log errors)
    """
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dt2
    import safe_migrate as _sm
    import calibration_tracker as _ct
    from database import engine

    _CT = _ZI("America/Chicago")
    now_ct = _dt2.now(_CT)

    # ── Scheduler overview ────────────────────────────────────────────────────
    state = sched.get_state()
    fx_ok,  _ = sched._verify_fixtures_today()
    pk_ok,  _ = sched._verify_picks_published()
    mb_ok,  _ = sched._verify_mock_generated()
    st_ok,  _ = sched._verify_settle_recent()
    fixture_age_h = sched.fixture_staleness_hours()

    _STALE_HOURS = 8.0   # fixtures older than this are flagged
    fixtures_stale = fixture_age_h is not None and fixture_age_h > _STALE_HOURS

    sched_status = (
        "critical" if not all([fx_ok, pk_ok, mb_ok])
        else "degraded" if (not st_ok or fixtures_stale)
        else "healthy"
    )

    # ── Calibration drift ─────────────────────────────────────────────────────
    try:
        drift = _ct.latest_drift_summary(engine)
        drift["last_scheduler_run"] = state.get("last_calibration_check")
        drift["last_scheduler_result"] = state.get("last_calibration_check_result")
    except Exception as _de:
        drift = {"error": str(_de), "alert_level": "ok", "last_check": None}

    # ── Migration health ──────────────────────────────────────────────────────
    try:
        migrations = _sm.migration_history(engine)
        backup_info = _sm.last_backup_info()
        migration_health = {
            "total_migrations": len(migrations),
            "last_5": migrations[:5],
            "backup": backup_info,
        }
    except Exception as _me:
        migration_health = {"error": str(_me)}

    # ── DB stats ──────────────────────────────────────────────────────────────
    def _count(table: str) -> int:
        try:
            return db.execute(__import__("sqlalchemy").text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
        except Exception:
            return -1

    db_stats = {
        "mock_bets_total":     _count("mock_bets"),
        "mock_bets_settled":   db.execute(__import__("sqlalchemy").text(
            "SELECT COUNT(*) FROM mock_bets WHERE status IN ('SETTLED_WIN','SETTLED_LOSS')"
        )).scalar() or 0,
        "user_picks_total":    _count("user_picks"),
        "user_picks_pending":  db.execute(__import__("sqlalchemy").text(
            "SELECT COUNT(*) FROM user_picks WHERE status = 'PENDING'"
        )).scalar() or 0,
        "fixtures_today":      db.execute(__import__("sqlalchemy").text(
            "SELECT COUNT(*) FROM fixtures WHERE game_date_ct = :today"
        ), {"today": datetime.now(_CACHE_TZ).strftime("%Y-%m-%d")}).scalar() or 0,
        "unresolved_legs":     db.execute(__import__("sqlalchemy").text(
            """SELECT COUNT(*) FROM user_pick_legs upl
               JOIN user_picks up ON up.id = upl.user_pick_id
               WHERE up.status IN ('SETTLED_WIN','SETTLED_LOSS')
                 AND upl.actual_outcome_value IS NULL
                 AND upl.expected_outcome_value IS NOT NULL
                 AND upl.outcome_source != 'auto_settlement'"""
        )).scalar() or 0,
    }

    # ── Recent settle errors ──────────────────────────────────────────────────
    try:
        err_rows = db.execute(__import__("sqlalchemy").text("""
            SELECT bet_id, result, notes, settled_at
            FROM settle_log
            WHERE result = 'ERROR'
            ORDER BY settled_at DESC
            LIMIT 5
        """)).fetchall()
        recent_errors = [
            {"bet_id": r[0], "result": r[1], "notes": r[2], "settled_at": str(r[3])}
            for r in err_rows
        ]
    except Exception as _ee:
        recent_errors = [{"error": str(_ee)}]

    # ── Boost framework tracking ──────────────────────────────────────────────
    import mock_bets as _mb
    _sqla_text = __import__("sqlalchemy").text
    _STRAT_B_DRIFT_WR   = 0.60   # WR below this triggers tighten
    _STRAT_B_MIN_SAMPLE = 30     # minimum settled picks before evaluating

    try:
        # Today's picks (game_date = today CT) — new-format strategy labels only
        _today_str = now_ct.strftime("%Y-%m-%d")
        _new_strats = "('amplify_positive_ev','rescue_high_wr_chalk','no_boost')"
        _today_rows = db.execute(_sqla_text(f"""
            SELECT
                boost_strategy,
                COUNT(*) as n,
                AVG(CASE WHEN boost_strategy IN ('amplify_positive_ev','rescue_high_wr_chalk')
                         THEN base_ev_per_10 END)     as avg_base_ev,
                AVG(CASE WHEN boost_strategy IN ('amplify_positive_ev','rescue_high_wr_chalk')
                         THEN boosted_ev_per_10 END)  as avg_boosted_ev
            FROM mock_bets
            WHERE game_date = :today
              AND source NOT IN ('forced_generation','exploration','retroactive_mock')
              AND boost_strategy IN {_new_strats}
            GROUP BY boost_strategy
        """), {"today": _today_str}).fetchall()

        _td: dict[str, dict] = {}
        for _r in _today_rows:
            _td[_r[0] or "no_boost"] = {
                "n": _r[1], "avg_base_ev": _r[2], "avg_boosted_ev": _r[3]
            }

        _td_total  = sum(v["n"] for v in _td.values())
        _td_a      = _td.get("amplify_positive_ev", {})
        _td_b      = _td.get("rescue_high_wr_chalk", {})
        _td_none   = _td.get("no_boost", {})
        _td_boosted = (_td_a.get("n", 0) or 0) + (_td_b.get("n", 0) or 0)

        today_boost = {
            "total_picks":   _td_total,
            "boosted_picks": _td_boosted,
            "boost_rate":    round(_td_boosted / _td_total, 3) if _td_total else 0.0,
            "strategy_a":    _td_a.get("n", 0) or 0,
            "strategy_b":    _td_b.get("n", 0) or 0,
            "no_boost":      _td_none.get("n", 0) or 0,
            "avg_base_ev_a":    round(_td_a.get("avg_base_ev") or 0, 3),
            "avg_boosted_ev_a": round(_td_a.get("avg_boosted_ev") or 0, 3),
            "avg_base_ev_b":    round(_td_b.get("avg_base_ev") or 0, 3),
            "avg_boosted_ev_b": round(_td_b.get("avg_boosted_ev") or 0, 3),
        }

        # 30-day settled performance by strategy — new-format labels only
        _30d_rows = db.execute(_sqla_text(f"""
            SELECT
                boost_strategy,
                COUNT(*) as picks,
                SUM(CASE WHEN status = 'SETTLED_WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status IN ('SETTLED_WIN','SETTLED_LOSS') THEN 1 ELSE 0 END) as settled,
                SUM(COALESCE(actual_profit, 0)) as total_profit,
                SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_stake
            FROM mock_bets
            WHERE generated_at >= date('now', '-30 days')
              AND status IN ('SETTLED_WIN','SETTLED_LOSS')
              AND source NOT IN ('forced_generation','exploration','retroactive_mock')
              AND boost_strategy IN {_new_strats}
            GROUP BY boost_strategy
        """)).fetchall()

        def _row_metrics(rows, key: str) -> dict:
            for _r in rows:
                if (_r[0] or "no_boost") == key:
                    _settled = _r[3] or 0
                    _wins    = _r[2] or 0
                    _stake   = _r[5] or 0
                    _profit  = _r[4] or 0
                    return {
                        "picks":  _r[1] or 0,
                        "wr":     round(_wins / _settled, 3) if _settled else None,
                        "roi":    round(_profit / _stake, 3) if _stake else None,
                        "wins":   _wins,
                        "settled": _settled,
                    }
            return {"picks": 0, "wr": None, "roi": None}

        _l30_a    = _row_metrics(_30d_rows, "amplify_positive_ev")
        _l30_b    = _row_metrics(_30d_rows, "rescue_high_wr_chalk")
        _l30_none = _row_metrics(_30d_rows, "no_boost")

        last_30 = {
            "strategy_a_picks":   _l30_a["picks"],
            "strategy_a_wr":      _l30_a["wr"],
            "strategy_a_roi":     _l30_a["roi"],
            "strategy_b_picks":   _l30_b["picks"],
            "strategy_b_wr":      _l30_b["wr"],
            "strategy_b_roi":     _l30_b["roi"],
            "no_boost_picks":     _l30_none["picks"],
            "no_boost_wr":        _l30_none["wr"],
            "no_boost_roi":       _l30_none["roi"],
        }

        # ── Calibration status + auto-tighten ────────────────────────────────
        _b_settled = _l30_b.get("settled") or _l30_b["picks"]
        _b_wr      = _l30_b["wr"]
        _cfg       = _mb._BOOST_STRATEGY_CONFIG

        if _b_settled < _STRAT_B_MIN_SAMPLE:
            _calib_status = "strategy_b_validating"
        elif _b_wr is not None and _b_wr < _STRAT_B_DRIFT_WR:
            _calib_status = "strategy_b_drift"
            # Auto-tighten if not already tightened to 70%
            if not _cfg.get("tightened") or _cfg.get("strat_b_min_wp", 65.0) < 70.0:
                _mb.tighten_strat_b_threshold(
                    new_wp=70.0,
                    reason=(
                        f"30d Strategy B WR={_b_wr:.1%} < 60% on {_b_settled} picks "
                        f"— tightened from {_cfg.get('strat_b_min_wp', 65.0):.0f}% to 70%"
                    ),
                )
        else:
            _calib_status = "well_calibrated"
            # If previously tightened but WR recovered, log (manual reset required)
            if _cfg.get("tightened"):
                _calib_status = "well_calibrated_tightened"  # still tight, but WR recovered

        boost_framework: dict = {
            "today":              today_boost,
            "last_30_days":       last_30,
            "calibration_status": _calib_status,
            "strat_b_min_wp_active": _cfg.get("strat_b_min_wp", 65.0),
            "strat_b_tightened":     _cfg.get("tightened", False),
            "strat_b_tightened_at":  _cfg.get("tightened_at"),
            "strat_b_tighten_reason":_cfg.get("tighten_reason"),
        }
    except Exception as _bfe:
        boost_framework = {"error": str(_bfe)}

    # ── Exploration status (tiered diagnostic system) ─────────────────────────
    try:
        import mock_bets as _mb_exp
        _sqla_text2 = __import__("sqlalchemy").text

        # Today's counts by tier
        _exp_today = db.execute(_sqla_text2("""
            SELECT exploration_tier, COUNT(*) as n
            FROM mock_bets
            WHERE source = 'exploration'
              AND game_date = :today
              AND exploration_tier IS NOT NULL
            GROUP BY exploration_tier
        """), {"today": now_ct.strftime("%Y-%m-%d")}).fetchall()
        _exp_today_dict = {r[0]: r[1] for r in _exp_today}

        # 30-day counts by tier
        _exp_30d = db.execute(_sqla_text2("""
            SELECT exploration_tier, COUNT(*) as n,
              SUM(CASE WHEN status IN ('SETTLED_WIN','SETTLED_LOSS') THEN 1 ELSE 0 END) as settled
            FROM mock_bets
            WHERE source = 'exploration'
              AND generated_at >= date('now', '-30 days')
              AND exploration_tier IS NOT NULL
            GROUP BY exploration_tier
        """)).fetchall()
        _exp_30d_dict = {r[0]: {"n": r[1], "settled": r[2]} for r in _exp_30d}

        # Per-market verdicts from calibration function (last 30 days)
        _calib = _mb_exp.get_exploration_calibration(days=30)

        # Expansion markets needing action
        _expansion_markets = [
            {"market": f"{e['sport']} {e['market_type']}", "n": e["n_legs"],
             "actual_wr": e["actual_wr"], "verdict": e["verdict"]}
            for e in _calib.get("expansion", [])
        ]
        _boundary_probed = [
            f"{e['sport']} {e['market_type']}"
            for e in _calib.get("boundary_testing", [])
        ]

        exploration_status = {
            "validation_tier": {
                "legs_today":   _exp_today_dict.get("validation", 0),
                "legs_30d":     (_exp_30d_dict.get("validation") or {}).get("n", 0),
                "settled_30d":  (_exp_30d_dict.get("validation") or {}).get("settled", 0),
                "calibration_markets": [
                    {"market": f"{e['sport']} {e['market_type']}", "n": e["n_legs"],
                     "actual_wr": e["actual_wr"], "avg_model_wp": e["avg_model_wp"],
                     "verdict": e["verdict"]}
                    for e in _calib.get("validation", [])
                ],
            },
            "expansion_tier": {
                "legs_today":              _exp_today_dict.get("expansion", 0),
                "legs_30d":                (_exp_30d_dict.get("expansion") or {}).get("n", 0),
                "markets_under_test":      _expansion_markets,
                "markets_ready_to_promote": [m for m in _expansion_markets if m["verdict"] == "promote"],
                "markets_to_demote":       [m for m in _expansion_markets if m["verdict"] == "demote"],
            },
            "boundary_testing_tier": {
                "legs_today":              _exp_today_dict.get("boundary_testing", 0),
                "legs_30d":                (_exp_30d_dict.get("boundary_testing") or {}).get("n", 0),
                "broken_zones_probed":     _boundary_probed,
                "zones_showing_recovery":  [
                    f"{e['sport']} {e['market_type']}"
                    for e in _calib.get("boundary_testing", [])
                    if e["verdict"] in ("consider_expand", "slight_improvement")
                ],
            },
        }
    except Exception as _expe:
        exploration_status = {"error": str(_expe)}

    # ── Brier score summary ───────────────────────────────────────────────────
    try:
        _bs7  = mb_module.get_brier_scores(db, days=7)
        _bs30 = mb_module.get_brier_scores(db, days=30)
        _leg_today  = _bs7["overall"].get("per_leg_brier")
        _leg_30d    = _bs30["overall"].get("per_leg_brier")
        _par_today  = _bs7["overall"].get("parlay_brier")
        _trend_7d   = _bs7["overall"].get("trend_7d")

        def _brier_tier(b):
            if b is None:
                return "unknown"
            if b <= 0.15:
                return "elite"
            if b <= 0.18:
                return "excellent"
            if b <= 0.22:
                return "good"
            if b <= 0.25:
                return "mediocre"
            return "poor"

        def _trend_label(t):
            if t is None:
                return "stable"
            if t < -0.01:
                return "improving"
            if t > 0.01:
                return "degrading"
            return "stable"

        _brier_summary = {
            "per_leg_today":   _leg_today,
            "per_leg_7d_avg":  _leg_today,
            "parlay_today":    _par_today,
            "trend":           _trend_label(_trend_7d),
            "target":          0.22,
            "elite_target":    0.15,
            "current_tier":    _brier_tier(_leg_today),
        }
    except Exception as _bse:
        _brier_summary = {"error": str(_bse)}

    # ── Overall status ────────────────────────────────────────────────────────
    drift_level = drift.get("alert_level", "ok")
    _boost_drift = boost_framework.get("calibration_status") == "strategy_b_drift"
    if sched_status == "critical" or drift_level == "critical":
        overall = "critical"
    elif sched_status == "degraded" or drift_level == "alert" or _boost_drift:
        overall = "degraded"
    else:
        overall = "healthy"

    # ── Market promotion/demotion candidates ─────────────────────────────────
    try:
        pending_market_actions = mb_module.get_pending_market_actions(db)
    except Exception as _pma_e:
        pending_market_actions = {"error": str(_pma_e)}

    # ── Retraining readiness ──────────────────────────────────────────────────
    try:
        retraining_status = mb_module.check_retraining_readiness(db)
    except Exception as _rt_e:
        retraining_status = {"error": str(_rt_e)}

    # ── Model version status ──────────────────────────────────────────────────
    try:
        import weekly_retrain as _wrt
        model_status = _wrt.get_model_status(db)
    except Exception as _ms_e:
        model_status = {"error": str(_ms_e)}

    # ── CLV summary ───────────────────────────────────────────────────────────
    try:
        clv_summary = mb_module.get_clv_summary(db, days=30)
    except Exception as _clv_e:
        clv_summary = {"error": str(_clv_e)}

    # ── Validation accuracy (latest persisted snapshot) ───────────────────────
    try:
        import validation_tracker as _valtrk
        validation_accuracy = _valtrk.get_latest_validation_summary()
    except Exception as _va_e:
        validation_accuracy = {"error": str(_va_e)}

    # ── Scout trajectory (per-grade WR trend vs targets) ─────────────────────
    try:
        scout_trajectory = _valtrk.get_scout_trajectory()
    except Exception as _st_e:
        scout_trajectory = {"error": str(_st_e)}

    # ── Tier summary + activation gate ─────────────────────────────────────
    try:
        _tier_summary = mb_module.get_tier_summary(db)
    except Exception as _ts_e:
        _tier_summary = {"error": str(_ts_e)}
    try:
        _activation_gate = mb_module.check_activation_readiness(db)
    except Exception as _ag_e:
        _activation_gate = {"error": str(_ag_e)}

    # ── Model routing distribution (7-day) ────────────────────────────────
    try:
        _mr_text = __import__("sqlalchemy").text
        _mr_today = now_ct.strftime("%Y-%m-%d")
        _mr_rows = db.execute(_mr_text("""
            SELECT mbl.model_used, COUNT(*) as legs,
                   ROUND(AVG(CASE WHEN mbl.leg_result='WIN' THEN 100.0
                                  WHEN mbl.leg_result='LOSS' THEN 0.0
                                  ELSE NULL END), 1) as win_rate
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.game_date >= DATE(:today, '-7 days')
              AND mb.source NOT IN ('exploration','retroactive_mock')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
            GROUP BY mbl.model_used
            ORDER BY legs DESC
        """), {"today": _mr_today}).fetchall()

        _mr_dist = [
            {"model": r[0] or "(null)", "legs": r[1], "win_rate": r[2]}
            for r in _mr_rows
        ]
        _mr_models = {r[0] for r in _mr_rows if r[0]}
        _mr_alert = None
        if "mlb_ats_v1" not in _mr_models and any(
            r[0] == "combined_v1" for r in _mr_rows
        ):
            _mr_alert = "mlb_ats_v1 has 0 legs in 7d -- possible silent fallback to combined_v1"
        if "mlb_totals_v1" not in _mr_models and any(
            r[0] == "combined_v1" for r in _mr_rows
        ):
            _t_msg = "mlb_totals_v1 has 0 legs in 7d -- possible silent fallback to combined_v1"
            _mr_alert = f"{_mr_alert}; {_t_msg}" if _mr_alert else _t_msg

        model_routing = {
            "distribution_7d": _mr_dist,
            "submodel_boot_status": getattr(app.state, "submodel_status", {}),
            "alert": _mr_alert,
        }
    except Exception as _mr_e:
        model_routing = {"error": str(_mr_e)}

    return {
        "timestamp_ct":      now_ct.strftime("%Y-%m-%d %H:%M CT"),
        "overall":           overall,
        "scheduler": {
            "status":              sched_status,
            "fixtures_ok":         fx_ok,
            "picks_ok":            pk_ok,
            "mocks_ok":            mb_ok,
            "settlement_ok":       st_ok,
            "fixture_age_hours":   fixture_age_h,
            "fixtures_stale":      fixtures_stale,
            "fixture_last_result": state.get("last_fixture_refresh_result"),
        },
        "calibration":       drift,
        "migration_health":  migration_health,
        "db_stats":          db_stats,
        "recent_settle_errors": recent_errors,
        "boost_framework":   boost_framework,
        "exploration_status": exploration_status,
        "brier_scores":      _brier_summary,
        "pending_market_actions": pending_market_actions,
        "retraining_status": retraining_status,
        "model_status": model_status,
        "clv_summary": clv_summary,
        "validation_accuracy": validation_accuracy,
        "scout_trajectory": scout_trajectory,
        "tier_summary": _tier_summary,
        "activation_gate": _activation_gate,
        "model_routing": model_routing,
        "chain_status": {
            "morning":   state.get("last_morning_chain_result"),
            "afternoon": state.get("last_afternoon_chain_result"),
        },
    }


@app.get("/api/health/brier-scores")
def health_brier_scores(days: int = Query(30, ge=7, le=365), db: Session = Depends(get_db)):
    """
    Full Brier score tracking: per-leg and parlay Brier, Murphy decomposition
    by sport/market, by-grade breakdown, win-prob calibration buckets, and
    7-day / 30-day trend deltas.
    """
    return mb_module.get_brier_scores(db, days=days)


@app.get("/api/health/brier-decomposition")
def health_brier_decomposition(
    days: int = Query(30, ge=7, le=365),
    windows: str = Query(None, description="Comma-separated window sizes, e.g. '7,14,30'"),
    db: Session = Depends(get_db),
):
    """
    Full Murphy Brier decomposition per (sport, market_type) cell.

    Returns cells sorted by Reliability (calibration leverage), parlay decomposition,
    and 90-100% WP bucket investigation. Use to identify highest-priority
    calibration vs feature-engineering interventions.

    Pass ?windows=7,14,30 to get rolling Brier windows with trajectory label instead.
    """
    if windows:
        try:
            _win_list = [int(w.strip()) for w in windows.split(",") if w.strip().isdigit()]
        except Exception:
            _win_list = [7, 14, 30]
        return mb_module.get_brier_windows(db, windows=_win_list)
    return mb_module.get_brier_decomposition(db, days=days)


@app.get("/api/health/activation-gate")
def health_activation_gate(db: Session = Depends(get_db)):
    """Activation readiness for real-money betting — 11 criteria check."""
    return mb_module.check_activation_readiness(db)


@app.get("/api/health/parlay-calibration")
def health_parlay_calibration():
    """
    Return current parlay_calibration table contents + trigger a rebuild
    from all settled mock_bets.  Use to see and refresh the parlay WP haircuts
    applied by get_calibrated_parlay_wp().
    """
    import os as _os
    from win_prob_calibration import (
        build_parlay_calibration_table as _build,
        _PARLAY_CAL_CREATE,
    )
    import sqlite3 as _sq3

    _db = _os.path.join(_os.path.dirname(__file__), "..", "data", "bets.db")
    build_result = _build(_db)

    rows = []
    try:
        con = _sq3.connect(_db, timeout=15)
        raw = con.execute("""
            SELECT wp_bucket, bucket_lo, bucket_hi, observed_pred_wp,
                   observed_actual_wr, haircut_pp, sample_size, confidence, last_updated
            FROM parlay_calibration ORDER BY bucket_lo
        """).fetchall()
        con.close()
        rows = [
            {
                "wp_bucket":         r[0],
                "avg_predicted_wp":  round(r[3], 2) if r[3] else None,
                "actual_win_rate":   round(r[4], 2) if r[4] else None,
                "haircut_pp":        round(r[5], 2) if r[5] else None,
                "n":                 r[6],
                "confidence":        r[7],
                "last_updated":      r[8],
            }
            for r in raw
        ]
    except Exception as exc:
        return {"error": str(exc), "build_result": build_result}

    return {"buckets": rows, "build_result": build_result}


@app.get("/api/health/player-projections")
def health_player_projections(
    game_date: str = Query(None, description="YYYY-MM-DD; defaults to today CT"),
    sport: str     = Query(None, description="Filter by sport: MLB, NBA, NHL"),
    db: Session    = Depends(get_db),
):
    """
    Player projection status for a game date.

    Returns stored projections from the player_projections table.
    If no projections exist for the date, triggers a build run.
    """
    from zoneinfo import ZoneInfo
    _CT = ZoneInfo("America/Chicago")
    today = game_date or datetime.now(_CT).strftime("%Y-%m-%d")

    rows = db.execute(
        __import__("sqlalchemy").text("""
            SELECT player_name, sport, team, opponent,
                   stat_type, projection, confidence, matchup_adjustment,
                   venue_adjustment, projected_at
            FROM player_projections
            WHERE game_date = :gd
              AND (:sport IS NULL OR sport = :sport)
            ORDER BY sport, confidence DESC, projection DESC
            LIMIT 500
        """),
        {"gd": today, "sport": sport},
    ).fetchall()

    if not rows:
        # Trigger a build if empty
        try:
            import player_projections as _pp
            build_result = _pp.run_daily_projections(db, game_date=today)
        except Exception as exc:
            build_result = {"error": str(exc)}

        rows = db.execute(
            __import__("sqlalchemy").text("""
                SELECT player_name, sport, team, opponent,
                       stat_type, projection, confidence, matchup_adjustment,
                       venue_adjustment, projected_at
                FROM player_projections
                WHERE game_date = :gd
                  AND (:sport IS NULL OR sport = :sport)
                ORDER BY sport, confidence DESC, projection DESC
                LIMIT 500
            """),
            {"gd": today, "sport": sport},
        ).fetchall()
    else:
        build_result = None

    projections = [
        {
            "player_name":       r[0],
            "sport":             r[1],
            "team":              r[2],
            "opponent":          r[3],
            "stat_type":         r[4],
            "projection":        round(r[5], 3) if r[5] is not None else None,
            "confidence":        round(r[6], 3) if r[6] is not None else None,
            "matchup_adjustment": round(r[7], 3) if r[7] is not None else None,
            "venue_adjustment":  round(r[8], 3) if r[8] is not None else None,
            "projected_at":      r[9],
        }
        for r in rows
    ]

    # Group by sport for summary
    from collections import defaultdict
    by_sport: dict = defaultdict(int)
    for p in projections:
        by_sport[p["sport"]] += 1

    return {
        "game_date":   today,
        "count":       len(projections),
        "by_sport":    dict(by_sport),
        "projections": projections,
        "build_result": build_result,
    }


@app.get("/api/health/per-market-auc")
def health_per_market_auc(db: Session = Depends(get_db)):
    """
    Per-sport-market performance diagnostic (Phase A formalized as endpoint).

    Computes:
      • Per sport+market: leg WR, avg model WP, avg EV, leg count
      • Calibration buckets: actual WR vs predicted WP in 10pp bands
      • AUC proxy: leg WR relative to WP (high WR + close to WP = well calibrated)
      • Market eligibility recommendations based on data

    Recomputed fresh on each call (no caching — typically called once/day from health UI).
    Only clean prospective sources (prospective, prospective_pm) are included.
    Exploration bets are excluded (they are training data, not performance evaluation).
    """
    from sqlalchemy import text as _sql
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dt2

    _CT   = _ZI("America/Chicago")
    as_of = _dt2.now(_CT).strftime("%Y-%m-%d %H:%M CT")

    _CLEAN_SOURCES = "('prospective','prospective_pm')"
    _MIN_LEGS = 10
    # Current-era floor: exclude legs from primitive models (April debugging)
    _ERA_FLOOR = "2026-05-26"

    # ── Per sport-market performance ──────────────────────────────────────────
    market_rows = db.execute(_sql(f"""
        SELECT
            mbl.sport,
            mbl.market_type,
            COUNT(*) as legs,
            SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END) /
                NULLIF(COUNT(*), 0), 1) as leg_wr_pct,
            ROUND(AVG(mbl.win_prob), 1) as avg_model_wp,
            ROUND(AVG(mbl.ev), 3) as avg_ev
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mb.source IN {_CLEAN_SOURCES}
          AND mbl.leg_result IN ('WIN','LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mb.game_date >= :era_floor
        GROUP BY mbl.sport, mbl.market_type
        ORDER BY COUNT(*) DESC, leg_wr_pct DESC
    """), {"era_floor": _ERA_FLOOR}).fetchall()

    by_sport_market = []
    high_auc_markets = []
    low_auc_markets  = []

    for r in market_rows:
        sport, mkt, n, wins, wr_pct, avg_wp, avg_ev = r
        if not n:
            continue

        # AUC proxy: how close is actual WR to predicted WP?
        # Simple metric: |actual_wr - avg_wp| < 8pp → well calibrated
        # We also use WR relative to 50% as signal strength
        auc_proxy = None
        calib_delta = None
        if wr_pct is not None and avg_wp is not None:
            calib_delta = round(wr_pct - avg_wp, 1)
            # AUC proxy: a WR above 55% in a market where model assigns avg WP 60-80%
            # and the delta is within ±10pp suggests the model has genuine signal
            if n >= _MIN_LEGS:
                if wr_pct >= 65.0 and abs(calib_delta) <= 12.0:
                    auc_proxy = "high_signal"
                    high_auc_markets.append(f"{sport} {mkt}")
                elif wr_pct < 45.0 or (calib_delta is not None and calib_delta < -20):
                    auc_proxy = "no_signal"
                    low_auc_markets.append(f"{sport} {mkt}")
                elif wr_pct >= 55.0:
                    auc_proxy = "moderate_signal"
                else:
                    auc_proxy = "weak_signal"
                    low_auc_markets.append(f"{sport} {mkt}")

        by_sport_market.append({
            "sport":            sport,
            "market_type":      mkt,
            "n_legs":           n,
            "leg_wr_pct":       wr_pct,
            "avg_model_wp":     avg_wp,
            "calibration_delta":calib_delta,
            "avg_ev":           avg_ev,
            "signal":           auc_proxy,
            "low_n":            n < _MIN_LEGS,
        })

    # ── Calibration buckets ───────────────────────────────────────────────────
    calib_rows = db.execute(_sql(f"""
        SELECT
            mbl.sport,
            CASE
                WHEN mbl.win_prob >= 75 THEN '75%+'
                WHEN mbl.win_prob >= 65 THEN '65-75%'
                WHEN mbl.win_prob >= 55 THEN '55-65%'
                WHEN mbl.win_prob >= 45 THEN '45-55%'
                ELSE '<45%'
            END as predicted_range,
            COUNT(*) as legs,
            ROUND(100.0 * SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END) /
                NULLIF(COUNT(*), 0), 1) as actual_wr_pct,
            ROUND(AVG(mbl.win_prob), 1) as avg_predicted_wp,
            ROUND(
                100.0 * SUM(CASE WHEN mbl.leg_result='WIN' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
                - AVG(mbl.win_prob)
            , 1) as calibration_delta_pp
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mb.source IN {_CLEAN_SOURCES}
          AND mbl.leg_result IN ('WIN','LOSS')
          AND COALESCE(mbl.excluded_from_metrics, 0) = 0
          AND mbl.win_prob IS NOT NULL
          AND mb.game_date >= :era_floor
        GROUP BY mbl.sport, predicted_range
        ORDER BY mbl.sport, AVG(mbl.win_prob) DESC
    """), {"era_floor": _ERA_FLOOR}).fetchall()

    calibration = [
        {
            "sport":             r[0],
            "predicted_range":   r[1],
            "n_legs":            r[2],
            "actual_wr_pct":     r[3],
            "avg_predicted_wp":  r[4],
            "calibration_delta": r[5],
            "verdict": (
                "overconfident"   if (r[5] or 0) < -8 and (r[2] or 0) >= _MIN_LEGS else
                "underconfident"  if (r[5] or 0) > 8  and (r[2] or 0) >= _MIN_LEGS else
                "calibrated"      if (r[2] or 0) >= _MIN_LEGS else
                "low_n"
            ),
        }
        for r in calib_rows
    ]

    # ── Injury data coverage ──────────────────────────────────────────────────
    try:
        inj_count = db.execute(_sql(
            "SELECT COUNT(*) FROM injury_reports WHERE game_date = date('now')"
        )).scalar() or 0
        inj_sports = db.execute(_sql(
            "SELECT DISTINCT sport FROM injury_reports WHERE game_date = date('now')"
        )).fetchall()
        injury_coverage = {
            "today_records": inj_count,
            "sports_covered": [r[0] for r in inj_sports],
        }
    except Exception:
        injury_coverage = {"today_records": 0, "sports_covered": []}

    # ── Line movement coverage ────────────────────────────────────────────────
    try:
        lm_count = db.execute(_sql(
            "SELECT COUNT(*) FROM line_movement_log WHERE date(fetched_at) = date('now')"
        )).scalar() or 0
        line_movement_coverage = {"today_snapshots": lm_count}
    except Exception:
        line_movement_coverage = {"today_snapshots": 0}

    return {
        "as_of":             as_of,
        "source_filter":     "prospective + prospective_pm only (exploration excluded)",
        "min_legs_for_auc":  _MIN_LEGS,
        "by_sport_market":   by_sport_market,
        "calibration":       calibration,
        "interpretation": {
            "high_signal_markets": list(set(high_auc_markets)),
            "low_signal_markets":  list(set(low_auc_markets)),
            "note": (
                "Markets with signal >= 65% WR and |calib_delta| <= 12pp are "
                "eligible for composite scoring uplift. Markets with WR < 45% "
                "or calib_delta < -20pp are candidates for ban or min_wp floor raise."
            ),
        },
        "injury_coverage":        injury_coverage,
        "line_movement_coverage": line_movement_coverage,
    }


@app.post("/api/health/refresh-injuries")
def refresh_injuries_endpoint(
    game_date: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Manually trigger injury data refresh for today's games.

    Called by the cron scheduler at 7 AM, 1 PM, 4 PM CT.
    Also callable manually to force a mid-day update.
    """
    try:
        import injury_fetcher as _if
        result = _if.refresh_injuries(db, game_date=game_date)
        return {"status": "ok", **result}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.get("/api/exploration/calibration")
def exploration_calibration(days: int = 30):
    """
    Per-tier per-market performance for the exploration diagnostic system.

    Returns verdict per market:
      validation  → calibrated | overconfident_zone | watch | insufficient_data
      expansion   → promote | demote | monitor | insufficient_data
      boundary_testing → consider_expand | slight_improvement | still_broken | insufficient_data

    Promotion rule (expansion → prospective):
      actual_wr >= expected_wr - 5pp AND n >= 30 → 'promote'

    Demotion rule (expansion → boundary_testing):
      actual_wr < expected_wr - 15pp AND n >= 30 → 'demote'
    """
    import mock_bets as _mb
    return _mb.get_exploration_calibration(days=days)


@app.post("/api/exploration/promote-market")
def promote_market(body: dict, db: Session = Depends(get_db)):
    """
    Manually promote a market from exploration to prospective generation.

    Writes a 'promote' override to market_eligibility_overrides. The recommender
    picks this up on the next request — no restart required.

    Request body:
      {
        "sport":       "NBA",
        "market_type": "player_points",
        "min_wp":      62.0,       // optional — inferred from exploration calibration if omitted
        "priority":    1.1,        // optional — defaults to 1.1 for newly promoted markets
        "reason":      "30 legs, 67% WR, injury feed reliable",  // optional
        "applied_by":  "analyst"   // optional
      }

    Returns the effective eligibility entry that will be applied.
    """
    from database import MarketEligibilityOverride
    from recommender import _normalize_market_key
    from datetime import datetime, timezone

    sport       = (body.get("sport") or "").strip().upper()
    market_type = _normalize_market_key(body.get("market_type") or "")
    reason      = body.get("reason") or ""
    applied_by  = body.get("applied_by") or ""

    if not sport or not market_type:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="sport and market_type are required")

    # Infer min_wp from exploration calibration if not supplied
    min_wp = body.get("min_wp")
    if min_wp is None:
        try:
            import mock_bets as _mb
            calib = _mb.get_exploration_calibration(days=60)
            for tier_data in calib.get("tiers", {}).values():
                for entry in tier_data.get("markets", []):
                    if entry.get("sport") == sport and entry.get("market_type") == market_type:
                        actual_wr = entry.get("actual_wr")
                        if actual_wr is not None:
                            # min_wp = actual_wr - 5pp, floored at 55%
                            min_wp = max(55.0, round(actual_wr - 5.0, 1))
        except Exception:
            pass
    if min_wp is None:
        min_wp = 60.0  # safe fallback

    priority = body.get("priority") or 1.1  # slight boost for newly promoted markets

    row = MarketEligibilityOverride(
        sport       = sport,
        market_type = market_type,
        action      = "promote",
        min_wp      = float(min_wp),
        priority    = float(priority),
        banned      = False,
        reason      = reason or f"Manually promoted via API",
        applied_at  = datetime.now(timezone.utc),
        applied_by  = applied_by or "api",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    effective = {"min_wp": row.min_wp, "priority": row.priority, "banned": False}
    print(f"[market-promote] ({sport}, {market_type}) min_wp={row.min_wp} priority={row.priority} by={applied_by!r}")
    return {
        "status":        "promoted",
        "sport":         sport,
        "market_type":   market_type,
        "effective":     effective,
        "override_id":   row.id,
        "note":          "Active on next recommendation request — no restart required.",
    }


@app.post("/api/exploration/demote-market")
def demote_market(body: dict, db: Session = Depends(get_db)):
    """
    Manually demote (ban) a market from prospective generation.

    Writes a 'demote' override to market_eligibility_overrides. The recommender
    picks this up on the next request — no restart required.

    Request body:
      {
        "sport":       "MLB",
        "market_type": "spreads",
        "reason":      "9.1% WR, confirmed broken — keep in boundary_testing only",
        "applied_by":  "analyst"
      }

    A demotion sets banned=True. The market continues to appear in
    boundary_testing exploration so recovery can be detected.
    """
    from database import MarketEligibilityOverride
    from recommender import _normalize_market_key
    from datetime import datetime, timezone

    sport       = (body.get("sport") or "").strip().upper()
    market_type = _normalize_market_key(body.get("market_type") or "")
    reason      = body.get("reason") or ""
    applied_by  = body.get("applied_by") or ""

    if not sport or not market_type:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="sport and market_type are required")

    row = MarketEligibilityOverride(
        sport       = sport,
        market_type = market_type,
        action      = "demote",
        min_wp      = 999.0,   # effectively unreachable
        priority    = 0.0,
        banned      = True,
        reason      = reason or "Manually demoted via API",
        applied_at  = datetime.now(timezone.utc),
        applied_by  = applied_by or "api",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    print(f"[market-demote] ({sport}, {market_type}) banned=True by={applied_by!r} reason={reason!r}")
    return {
        "status":      "demoted",
        "sport":       sport,
        "market_type": market_type,
        "banned":      True,
        "override_id": row.id,
        "note":        "Banned from prospective generation. Still tracked in boundary_testing exploration.",
    }


@app.get("/api/exploration/market-overrides")
def list_market_overrides(db: Session = Depends(get_db)):
    """
    List all active market eligibility overrides, most recent first.

    Shows the history of promote/demote decisions. The latest override
    per (sport, market_type) is what the recommender actually uses.
    """
    from database import MarketEligibilityOverride
    rows = (
        db.query(MarketEligibilityOverride)
        .order_by(MarketEligibilityOverride.applied_at.desc())
        .all()
    )
    return {
        "overrides": [
            {
                "id":          r.id,
                "sport":       r.sport,
                "market_type": r.market_type,
                "action":      r.action,
                "min_wp":      r.min_wp,
                "priority":    r.priority,
                "banned":      r.banned,
                "reason":      r.reason,
                "applied_at":  r.applied_at.isoformat() if r.applied_at else None,
                "applied_by":  r.applied_by,
            }
            for r in rows
        ],
        "total": len(rows),
    }


@app.post("/api/exploration/run-tiered")
def run_tiered_exploration(
    daily_target: int = 40,
    game_date: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Manually trigger tiered exploration bet generation.

    Called by the daily scheduler after morning prospective generation.
    Can also be triggered manually for testing.
    """
    import mock_bets as _mb
    result = _mb.generate_tiered_exploration_bets(db, game_date=game_date, daily_target=daily_target)
    return {"status": "ok", **result}


@app.get("/api/retrain-log")
def get_retrain_log():
    """Serve the most recent retrain subprocess log as plain text."""
    from auto_settle import RETRAIN_STATE as _RS
    from fastapi.responses import PlainTextResponse
    log_path = _RS.get("log_path")
    if not log_path or not _os.path.exists(log_path):
        # Try to find most recent log in data/
        import glob as _glob
        logs = sorted(_glob.glob(_os.path.join(_os.path.dirname(__file__), "..", "data", "retrain_*.log")))
        if not logs:
            return PlainTextResponse("No retrain log found.", status_code=404)
        log_path = logs[-1]
    with open(log_path) as f:
        content = f.read()
    return PlainTextResponse(content or "(empty log)", media_type="text/plain")


@app.post("/api/scheduler/run-pm")
def scheduler_run_pm(force: bool = False):
    """
    Manually trigger the 3 PM CT afternoon mock-bet generation.

    By default uses require_change=True (only generates for fixtures with
    material changes since morning).  Pass ?force=true to bypass the
    material-change filter and generate for all evening fixtures.

    Returns {generated, skipped_dup, run_id, forced}.
    """
    from datetime import datetime as _dt, timezone as _tz
    import mock_bets as _mb
    from database import SessionLocal as _SL, init_db as _idb

    ct = sched._now_ct()
    if ct.hour < 15:
        return {"error": "PM run is only available after 3:00 PM CT", "hour_ct": ct.hour}

    _idb()
    db = _SL()
    try:
        result = _mb.generate_mock_bets(
            db,
            n_picks        = 10,
            source         = "prospective_pm",
            afternoon_only = True,
            require_change = not force,
        )
    finally:
        db.close()

    result["forced"] = force
    # Update PM state so guard knows it ran
    sched._scheduler_state["last_mock_generate_pm"]        = ct.strftime("%Y-%m-%d")
    sched._scheduler_state["last_mock_generate_pm_result"] = result
    return result


@app.get("/api/scheduler/jobs")
def scheduler_jobs():
    """
    List all scheduled daily jobs with their next scheduled run time (UTC).

    Each entry: {job_id, name, schedule_utc, last_run_date, next_run_utc, status}
    """
    from datetime import datetime, timezone, timedelta
    from zoneinfo import ZoneInfo
    _CT = ZoneInfo("America/Chicago")

    now_utc = datetime.now(timezone.utc)
    now_ct  = datetime.now(_CT)
    today_ct = now_ct.strftime("%Y-%m-%d")
    state   = sched.get_state()

    def _next_run_utc(hour_ct: int, last_run_date: str | None, minute_ct: int = 0) -> str:
        """Return ISO UTC timestamp of next run for a job scheduled at hour_ct:minute_ct CT."""
        # Build today's run time in CT
        candidate_ct = now_ct.replace(hour=hour_ct, minute=minute_ct, second=0, microsecond=0)
        if last_run_date == today_ct or candidate_ct <= now_ct:
            candidate_ct += timedelta(days=1)
        return candidate_ct.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    jobs = [
        {
            "job_id":        "fixture_refresh",
            "name":          "Fixture odds refresh (MLB / NBA / NHL)",
            "schedule_ct":   "07:45 CT",
            "last_run_date": state.get("last_fixture_refresh"),
            "next_run_utc":  _next_run_utc(7, state.get("last_fixture_refresh"), 45),
            "status":        "ran_today" if state.get("last_fixture_refresh") == today_ct else "pending",
        },
        {
            "job_id":        "mlb_pitcher_fetch",
            "name":          "MLB probable pitcher fetch (MLB Stats API)",
            "schedule_ct":   "07:45 CT",
            "last_run_date": state.get("last_mlb_pitcher_fetch"),
            "next_run_utc":  _next_run_utc(7, state.get("last_mlb_pitcher_fetch"), 45),
            "status":        "ran_today" if state.get("last_mlb_pitcher_fetch") == today_ct else "pending",
        },
        {
            "job_id":        "player_projections",
            "name":          "Player projections (MLB/NBA/NHL — free APIs)",
            "schedule_ct":   "08:00 CT",
            "last_run_date": state.get("last_player_projections"),
            "next_run_utc":  _next_run_utc(8, state.get("last_player_projections")),
            "status":        "ran_today" if state.get("last_player_projections") == today_ct else "pending",
        },
        {
            "job_id":        "weekly_retrain",
            "name":          "Weekly model retraining (Sunday 3 AM CT)",
            "schedule_ct":   "Sunday 03:00 CT",
            "last_run_date": state.get("last_weekly_retrain"),
            "last_result":   state.get("last_weekly_retrain_result"),
            "status":        "ran_this_week" if state.get("last_weekly_retrain") == today_ct else "pending",
        },
        {
            "job_id":        "mock_generate",
            "name":          "Mock bet generation (System 3)",
            "schedule_ct":   "08:00 CT",
            "last_run_date": state.get("last_mock_generate"),
            "next_run_utc":  _next_run_utc(8, state.get("last_mock_generate")),
            "status":        "ran_today" if state.get("last_mock_generate") == today_ct else "pending",
        },
        {
            "job_id":        "mock_settle",
            "name":          "Mock bet settlement (System 3)",
            "schedule_ct":   "09:00 CT",
            "last_run_date": state.get("last_mock_settle"),
            "next_run_utc":  _next_run_utc(9, state.get("last_mock_settle")),
            "status":        "ran_today" if state.get("last_mock_settle") == today_ct else "pending",
        },
        {
            "job_id":        "soccer_results",
            "name":          "Soccer results fetch (API-Football)",
            "schedule_ct":   "09:00 CT",
            "last_run_date": state.get("last_soccer_fetch"),
            "next_run_utc":  _next_run_utc(9, state.get("last_soccer_fetch")),
            "status":        "ran_today" if state.get("last_soccer_fetch") == today_ct else "pending",
        },
        {
            "job_id":        "mlb_etl",
            "name":          "MLB feature-matrix ETL refresh",
            "schedule_ct":   "10:00 CT",
            "last_run_date": state.get("last_mlb_etl_update"),
            "next_run_utc":  _next_run_utc(10, state.get("last_mlb_etl_update")),
            "status":        "ran_today" if state.get("last_mlb_etl_update") == today_ct else "pending",
        },
        {
            "job_id":        "nhl_etl",
            "name":          "NHL feature-matrix ETL refresh",
            "schedule_ct":   "10:00 CT",
            "last_run_date": state.get("last_nhl_etl_update"),
            "next_run_utc":  _next_run_utc(10, state.get("last_nhl_etl_update")),
            "status":        "ran_today" if state.get("last_nhl_etl_update") == today_ct else "pending",
        },
        {
            "job_id":        "mlb_scores",
            "name":          "MLB nightly scores ingest (MLB Stats API)",
            "schedule_ct":   "02:00 CT",
            "last_run_date": state.get("last_mlb_update"),
            "next_run_utc":  _next_run_utc(2, state.get("last_mlb_update")),
            "status":        "ran_today" if state.get("last_mlb_update") == today_ct else "pending",
        },
        {
            "job_id":        "nhl_scores",
            "name":          "NHL nightly scores ingest (NHL Web API)",
            "schedule_ct":   "02:00 CT",
            "last_run_date": state.get("last_nhl_update"),
            "next_run_utc":  _next_run_utc(2, state.get("last_nhl_update")),
            "status":        "ran_today" if state.get("last_nhl_update") == today_ct else "pending",
        },
        {
            "job_id":        "regime_classification",
            "name":          "regime_classification",
            "schedule_ct":   "08:30 CT",
            "last_run_date": state.get("last_regime_classify"),
            "next_run_utc":  _next_run_utc(8, state.get("last_regime_classify"), 30),
            "status":        "ran_today" if state.get("last_regime_classify") == today_ct else "pending",
        },
        {
            "job_id":        "mock_generate_pm",
            "name":          "Afternoon mock bet top-up / evening games (3 PM CT)",
            "schedule_ct":   "15:00 CT",
            "last_run_date": state.get("last_mock_generate_pm"),
            "next_run_utc":  _next_run_utc(15, state.get("last_mock_generate_pm")),
            "status":        "ran_today" if state.get("last_mock_generate_pm") == today_ct else "pending",
        },
        {
            "job_id":        "alt_lines_morning",
            "name":          "Alt lines batch fetch — morning (MLB/NHL/NBA/6 soccer, 10 sports)",
            "schedule_ct":   "07:45 CT",
            "last_run_date": state.get("last_alt_lines_fetch_morning"),
            "next_run_utc":  _next_run_utc(7, state.get("last_alt_lines_fetch_morning"), 45),
            "status":        "ran_today" if state.get("last_alt_lines_fetch_morning") == today_ct else "pending",
        },
        {
            "job_id":        "alt_lines_afternoon",
            "name":          "Alt lines batch fetch — afternoon (pre-PM run, 2:30 PM CT)",
            "schedule_ct":   "14:30 CT",
            "last_run_date": state.get("last_alt_lines_fetch_afternoon"),
            "next_run_utc":  _next_run_utc(14, state.get("last_alt_lines_fetch_afternoon"), 30),
            "status":        "ran_today" if state.get("last_alt_lines_fetch_afternoon") == today_ct else "pending",
        },
    ]
    return {
        "scheduler_running": state.get("running", False),
        "interval_mins":     state.get("interval_mins", 30),
        "jobs":              jobs,
    }

@app.post("/api/settle/scheduler/start")
def start_scheduler(req: SchedulerConfigRequest):
    """Start (or restart) the background scheduler with new settings."""
    sched.stop()
    import time; time.sleep(1)
    sched.start(interval_mins=req.interval_mins, auto_retrain=req.auto_retrain, daemon=True)
    return {"status": "started", "interval_mins": req.interval_mins, "auto_retrain": req.auto_retrain}

@app.post("/api/settle/scheduler/stop")
def stop_scheduler():
    """Stop the background scheduler."""
    sched.stop()
    return {"status": "stopped"}

@app.post("/api/odds/nightly-update")
def trigger_nightly_odds():
    """Manually trigger the NBA nightly closing-odds update (fetches yesterday's ESPN odds)."""
    result = sched.run_nightly_odds_now()
    return {"status": "ok", "result": result}

@app.post("/api/model/etl/nhl")
def trigger_nhl_etl():
    """
    Force a daily NHL feature-matrix ETL refresh.
    Fetches current-season schedule + team stats, then invalidates the prediction cache.
    Runs in the foreground — takes ~30s.
    """
    result = sched.run_daily_nhl_etl_now()
    return {"status": result.get("status"), "result": result}

@app.post("/api/model/etl/mlb")
def trigger_mlb_etl():
    """
    Force a daily MLB feature-matrix ETL refresh (quick mode — skips pitcher game logs).
    Fetches current-season schedule + team/batting stats, then invalidates the prediction cache.
    Runs in the foreground — takes ~2-3 min.
    """
    result = sched.run_daily_mlb_etl_now()
    return {"status": result.get("status"), "result": result}

@app.get("/api/settle/history")
def scheduler_history(limit: int = 20, db: Session = Depends(get_db)):
    """History of all scheduler run summaries."""
    return asettler.get_scheduler_history(db, limit=limit)




# ─── Attribution routes (Phase 4A) ───────────────────────────────────────────

@app.get("/api/attribution/full")
def attribution_full(include_mock: bool = False, db: Session = Depends(get_db)):
    """Full multi-dimension attribution report."""
    return attr.full_attribution(db, include_mock=include_mock)

@app.get("/api/attribution/league")
def attribution_league(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.by_league(db, include_mock=include_mock)

@app.get("/api/attribution/sport")
def attribution_sport(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.by_sport(db, include_mock=include_mock)

@app.get("/api/attribution/market")
def attribution_market(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.by_market_type(db, include_mock=include_mock)

@app.get("/api/attribution/odds-bracket")
def attribution_odds(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.by_odds_bracket(db, include_mock=include_mock)

@app.get("/api/attribution/legs")
def attribution_legs(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.by_legs(db, include_mock=include_mock)

@app.get("/api/attribution/timing")
def attribution_timing(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.by_hour(db, include_mock=include_mock)

@app.get("/api/attribution/streaks")
def attribution_streaks(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.streak_analysis(db, include_mock=include_mock)

@app.get("/api/attribution/leaks")
def attribution_leaks(include_mock: bool = False, db: Session = Depends(get_db)):
    return attr.find_leaks_and_edges(db, include_mock=include_mock)


# ─── Kelly / bankroll routes (Phase 4B) ──────────────────────────────────────

class KellyRequest(BaseModel):
    win_prob:     float          # 0.0-1.0
    decimal_odds: float
    bankroll:     Optional[float] = None   # uses stored bankroll if None

class KellyParlayRequest(BaseModel):
    legs:     list               # [{win_prob, odds, description}]
    bankroll: Optional[float] = None

class BankrollSetRequest(BaseModel):
    amount:          float
    kelly_fraction:  float = 0.25

@app.post("/api/kelly/single")
def kelly_single(req: KellyRequest, db: Session = Depends(get_db)):
    """Kelly stake recommendation for a single bet."""
    bankroll = req.bankroll or kly.load_bankroll()["current_bankroll"]
    return kly.kelly_stake(req.win_prob, req.decimal_odds, bankroll)

@app.post("/api/kelly/parlay")
def kelly_parlay(req: KellyParlayRequest, db: Session = Depends(get_db)):
    """Kelly stake for a parlay (combined + individual legs)."""
    bankroll = req.bankroll or kly.load_bankroll()["current_bankroll"]
    return kly.kelly_parlay(req.legs, bankroll)

@app.get("/api/bankroll")
def get_bankroll(db: Session = Depends(get_db)):
    """Current bankroll stats and sizing guidelines."""
    return kly.bankroll_stats(db)

@app.post("/api/bankroll/set")
def set_bankroll(req: BankrollSetRequest):
    """Set starting bankroll and Kelly fraction."""
    return kly.set_bankroll(req.amount, req.kelly_fraction)

@app.get("/api/bankroll/paper")
def paper_ledger(db: Session = Depends(get_db)):
    """Virtual P&L ledger for mock bets (paper trading)."""
    return kly.paper_ledger(db)

@app.get("/api/bankroll/timing")
def timing_advice(db: Session = Depends(get_db)):
    """Best hours/days to bet based on historical data + CLV signal."""
    return kly.timing_advice(db)

# ─── Recommender routes (Phase 3) ────────────────────────────────────────────

class RecommendRequest(BaseModel):
    n_picks:      int        = 5
    stake:        float      = 10.0
    max_legs:     int        = 5
    min_legs:     int        = 2
    min_odds:     float      = 2.0
    sort_by:      str        = "win_prob"
    sport_filter: Optional[list] = None   # e.g. ["NBA"] or None for all sports
    refresh:      bool       = False  # True → bypass cache and regenerate

class ModifyParlayRequest(BaseModel):
    current_legs:   list          # scored leg dicts from a pick
    add_leg_ids:    list[str] = []
    remove_leg_ids: list[str] = []
    stake:          float = 10.0

class ScoreParlayRequest(BaseModel):
    leg_ids: list[str]
    stake:   float = 10.0

def _build_todays_picks(req: RecommendRequest, db: Session) -> dict:
    """Run full generation + LQS enrichment. Called on cache miss or refresh."""
    resp = rec.generate_todays_picks(
        db,
        n_picks      = req.n_picks,
        stake        = req.stake,
        max_legs     = req.max_legs,
        min_legs     = req.min_legs,
        min_odds     = req.min_odds,
        sort_by      = req.sort_by,
        sport_filter = req.sport_filter if req.sport_filter else None,
    )
    # Enrich each pick's legs with LQS (best-effort; silently skip on error)
    # Include tiered fallback lists (tier_b/c/d) in addition to standard pick lists.
    # Use a dict to deduplicate — same pick object may appear in both "picks" and a tier list.
    _all_pick_lists_raw = (
        (resp.get("picks") or []) +
        (resp.get("anchor") or []) +
        (resp.get("core") or []) +
        (resp.get("mixed") or []) +
        (resp.get("section_a") or []) +
        (resp.get("power_picks") or []) +
        (resp.get("tier_b") or []) +
        (resp.get("tier_c") or []) +
        (resp.get("tier_d") or [])
    )
    _seen_pick_ids: set = set()
    _all_pick_lists = []
    for _p in _all_pick_lists_raw:
        _pid = id(_p)
        if _pid not in _seen_pick_ids:
            _seen_pick_ids.add(_pid)
            _all_pick_lists.append(_p)
    for pick in _all_pick_lists:
        lqs_scores = []
        for leg in (pick.get("legs") or []):
            try:
                # Parse home/away from "Away @ Home" game label
                _game_str  = leg.get("game", "")
                _pick_team = leg.get("pick", "")
                _home_team = _game_str.split(" @ ")[1].strip() if " @ " in _game_str else ""
                _away_team = _game_str.split(" @ ")[0].strip() if " @ " in _game_str else ""
                _is_home: Optional[bool] = (
                    True  if _home_team and _pick_team == _home_team else
                    False if _away_team and _pick_team == _away_team else
                    None
                )
                _opponent = _away_team if _is_home else (_home_team if _is_home is False else "")

                candidate = {
                    "market_type":      leg.get("market_type") or leg.get("market"),
                    "sport":            leg.get("sport"),
                    "team_or_player":   _pick_team,
                    "model_confidence": leg.get("win_prob"),
                    "model_used":       pick.get("model_used") or leg.get("model_used"),
                    "edge_pp":          leg.get("edge"),
                    "line":             leg.get("point"),
                    "opponent":         _opponent,
                    "is_home":          _is_home,
                }
                q = lq.compute_leg_quality_score(candidate, db)
                leg["lqs"]              = q["lqs"]
                leg["lqs_grade"]        = q["lqs_grade"]
                leg["lqs_warnings"]     = q["warnings"]
                leg["matchup_context"]  = q.get("matchup_context")
                lqs_scores.append(q["lqs"])
                # Flag total/spread legs with a close call pattern
                cc = q.get("close_call_history")
                if cc and cc.get("close_call_count", 0) >= 2:
                    leg["alt_pivot_hint"] = True
                    leg["close_call_count"] = cc["close_call_count"]
            except Exception:
                pass
        if lqs_scores:
            pick["avg_lqs"]       = round(sum(lqs_scores) / len(lqs_scores), 1)
            pick["min_lqs"]       = min(lqs_scores)
            pick["weakest_leg_idx"] = lqs_scores.index(pick["min_lqs"])

    # ── Boost eligibility annotation ─────────────────────────────────────────
    # Attach boost EV and eligibility info to every Section A pick for display.
    from mock_bets import _check_boost_eligible, _boost_ev, _is_sgp, _is_single_sport
    _STAKE = 10.0
    for pick in _all_pick_lists:
        _odds  = float(pick.get("combined_odds") or pick.get("odds") or 1.0)
        _legs  = len(pick.get("legs") or [])
        _sgp   = _is_sgp(pick)
        _ss    = _is_single_sport(pick)
        _wp    = float(pick.get("combined_win_prob") or pick.get("win_prob") or 50.0)
        # Stamp flags onto pick for frontend display
        pick["is_sgp"]           = _sgp
        pick["is_single_sport"]  = _ss

        tier_info: dict = {}
        for _t in (0.25, 0.30, 0.50):
            _ok, _reason = _check_boost_eligible(_t, _legs, _odds, _sgp, _ss)
            _ev = _boost_ev(_STAKE, _odds, _wp, _t) if _ok else None
            tier_info[f"+{int(_t*100)}%"] = {
                "eligible": _ok,
                "reason":   _reason,
                "ev":       _ev,
            }
        pick["boost_eligibility"] = tier_info

        # Best available boost EV lift for quick display
        _best_lift = 0.0
        _best_tier = None
        for _tier_key, _tier_val in tier_info.items():
            if _tier_val["eligible"] and _tier_val.get("ev"):
                _lift = _tier_val["ev"].get("ev_lift", 0.0)
                if _lift > _best_lift:
                    _best_lift = _lift
                    _best_tier = _tier_key
        if _best_tier:
            pick["best_boost_tier"] = _best_tier
            pick["best_boost_ev_lift"] = round(_best_lift, 2)

    # Re-sort all pick lists by avg_lqs descending so highest-quality picks show first
    for key in ("picks", "anchor", "core", "mixed", "power_picks", "tier_b", "tier_c", "tier_d"):
        lst = resp.get(key)
        if lst:
            lst.sort(key=lambda x: x.get("avg_lqs") or 0, reverse=True)

    return resp


@app.get("/api/recommend/today")
def todays_picks_cached():
    """
    Read-only: return today's picks from cache.
    Returns 204 (no content) if the cache has not been populated yet today.
    Use POST /api/recommend/today to generate or refresh picks.
    """
    cached = _get_cached_picks()
    if cached is None:
        from fastapi.responses import Response
        return Response(status_code=204)
    return cached


@app.post("/api/recommend/today")
def todays_picks(req: RecommendRequest, db: Session = Depends(get_db)):
    """
    Generate today's top parlay picks using historical pattern matching
    combined with live fixture EV scores. Each leg is enriched with LQS.

    Cache behaviour:
    - First call of the day triggers full generation (~30-45 s) and caches result.
    - Subsequent calls return the cached result instantly.
    - Pass {"refresh": true} to bypass the cache and regenerate in the background.
      Returns {"status": "regenerating", "eta_seconds": 30} immediately; poll
      GET /api/recommend/today every 3 s until it returns 200.
    - Cache is also invalidated on every fixture odds refresh.
    """
    global _regen_in_progress

    if req.refresh:
        # If a regen is already running, just report that.
        if _regen_in_progress:
            return {"status": "regenerating", "eta_seconds": 30}
        invalidate_picks_cache()
        _regen_in_progress = True
        import threading
        t = threading.Thread(
            target=_background_regen,
            args=(req.dict(exclude={"refresh"}),),
            daemon=True,
        )
        t.start()
        return {"status": "regenerating", "eta_seconds": 30}

    cached = _get_cached_picks()
    if cached is not None:
        return cached

    # First-time population (no refresh flag) — run synchronously so the caller
    # gets the result back directly without needing to poll.
    resp = _build_todays_picks(req, db)
    if not req.sport_filter:
        _set_cached_picks(resp)
    return resp

@app.post("/api/recommend/score")
def score_custom(req: ScoreParlayRequest, db: Session = Depends(get_db)):
    """
    Score any set of legs as a parlay — returns EV, quality rating,
    historical pattern match, and payout table.
    """
    from parlay_builder import get_available_legs, score_leg
    all_legs = get_available_legs(db)
    leg_map  = {l["leg_id"]: l for l in all_legs}
    selected = [score_leg(leg_map[lid]) for lid in req.leg_ids if lid in leg_map]
    if not selected:
        raise HTTPException(400, "No valid legs found.")
    return rec.score_parlay(selected, stake=req.stake, db=db)

@app.post("/api/recommend/modify")
def modify_pick(req: ModifyParlayRequest, db: Session = Depends(get_db)):
    """
    Add/remove legs from a recommended pick and recompute
    quality rating, EV, payout table instantly.
    """
    return rec.modify_parlay(
        current_legs   = req.current_legs,
        add_leg_ids    = req.add_leg_ids,
        remove_leg_ids = req.remove_leg_ids,
        stake          = req.stake,
        db             = db,
    )

@app.get("/api/recommend/patterns")
def historical_patterns(db: Session = Depends(get_db)):
    """Your top historical betting patterns by ROI — drives the recommender."""
    from recommender import _get_patterns, _top_patterns
    patterns = _get_patterns(db)
    return {"patterns": _top_patterns(patterns)}


@app.get("/api/stats/leg-win-rates")
def leg_win_rates(db: Session = Depends(get_db)):
    """
    Per-leg win rate analysis from historical bets.
    Uses geometric mean method: bet_win_rate ^ (1/avg_legs).
    Returns overall, by sport, and by legs bucket.
    """
    return ana.get_leg_win_rates(db)


# ─── Promo routes (Phase 6A) ─────────────────────────────────────────────────
import promo_engine as promo

class PromoScoreRequest(BaseModel):
    win_prob:      float
    american_odds: int
    promo_type:    str   = "none"
    stake:         float = 10.0

class PromoPatchRequest(BaseModel):
    promo_type: str

@app.post("/api/promo/score")
def promo_score(req: PromoScoreRequest):
    """Score a bet with promo applied — returns EV lift, Kelly, and strategy rec."""
    if req.promo_type not in promo.PROMO_TYPES:
        raise HTTPException(400, f"Unknown promo_type: {req.promo_type}")
    return promo.score_with_promo(
        win_prob      = req.win_prob,
        american_odds = req.american_odds,
        promo_type    = req.promo_type,
        stake         = req.stake,
    )

@app.patch("/api/bets/{bet_id}/promo")
def patch_bet_promo(bet_id: str, req: PromoPatchRequest, db: Session = Depends(get_db)):
    """
    Update the promo on a bet. Recalculates and stores boosted_odds, ev_lift,
    and was_free_bet so history analytics are accurate.
    """
    from database import Bet as BetModel
    bet = db.query(BetModel).filter(BetModel.id == bet_id).first()
    if not bet:
        raise HTTPException(404, "Bet not found")
    if req.promo_type not in promo.PROMO_TYPES:
        raise HTTPException(400, f"Unknown promo_type: {req.promo_type}")

    ptype = req.promo_type
    pinfo = promo.PROMO_TYPES[ptype]

    # Derive win_prob from stored decimal odds (implied) — best we have retroactively
    dec_odds  = float(bet.odds) if bet.odds else 1.91
    win_prob  = 1.0 / dec_odds   # implied probability as fallback

    # American odds from decimal
    am_odds = promo.decimal_to_american(dec_odds)
    scored  = promo.score_with_promo(win_prob, am_odds, ptype, float(bet.amount or 10))

    bet.promo_type         = ptype
    bet.promo_boosted_odds = scored["boosted_odds_american"]
    bet.promo_ev_lift      = scored["ev_lift"]
    bet.promo_was_free_bet = 1 if pinfo["is_free"] else 0
    db.commit()
    db.refresh(bet)

    return {
        "id":                  bet.id,
        "promo_type":          bet.promo_type,
        "promo_boosted_odds":  bet.promo_boosted_odds,
        "promo_ev_lift":       bet.promo_ev_lift,
        "promo_was_free_bet":  bet.promo_was_free_bet,
        "promo_label":         pinfo["label"],
        "score":               scored,
    }

@app.get("/api/promo/performance")
def promo_performance(db: Session = Depends(get_db)):
    """Promo ROI breakdown across all settled bets."""
    return ana.get_promo_performance(db)


# ─── Bet slip image parser (Vision API) ──────────────────────────────────────
import slip_parser as sp

class ConfirmSlipRequest(BaseModel):
    parsed:  dict          # the enriched parsed slip from /api/bets/parse-slip
    stake:   float = 10.0
    is_mock: bool  = True  # default mock until user confirms real



class AnalyzeTextRequest(BaseModel):
    legs:                 list          # [{team, market, point, american_odds}]
    combined_american_odds: Optional[int] = None
    n_legs:               Optional[int] = None
    bet_type:             str           = "parlay"
    sportsbook:           str           = "FanDuel"
    stake:                float         = 10.0

# ─── LQS slip enrichment helpers ────────────────────────────────────────────

# Keyword sets for sport inference (used by manual bet slip analysis)
_SPORT_KEYWORDS: dict[str, set] = {
    "MLB": {"pirates","braves","yankees","red sox","mets","dodgers","giants","cubs",
            "cardinals","padres","brewers","phillies","marlins","reds","rockies","rays",
            "rangers","orioles","twins","tigers","angels","astros","athletics","royals",
            "white sox","guardians","mariners","nationals","blue jays","diamondbacks"},
    "NBA": {"lakers","celtics","warriors","heat","bulls","knicks","nets","bucks",
            "nuggets","suns","clippers","raptors","76ers","sixers","hawks","cavaliers",
            "cavs","mavericks","mavs","pacers","thunder","rockets","spurs","magic",
            "hornets","pistons","wizards","kings","pelicans","grizzlies","trail blazers","jazz"},
    "NHL": {"penguins","bruins","lightning","avalanche","oilers","maple leafs","leafs",
            "canucks","islanders","devils","flyers","capitals","caps","blackhawks",
            "red wings","stars","blues","ducks","sharks","panthers","hurricanes",
            "predators","jets","golden knights","kraken","sabres","flames","wild",
            "senators","canadiens","blue jackets","mammoth"},
    "NFL": {"patriots","cowboys","packers","steelers","chiefs","49ers","ravens","bills",
            "chargers","raiders","broncos","rams","eagles","bengals","seahawks",
            "buccaneers","bucs","saints","bears","colts","dolphins","texans","titans",
            "jaguars","commanders","vikings","browns","lions","falcons"},
}

def _infer_sport_from_team(name: str) -> str:
    """Guess sport from a team/description string via keyword matching."""
    n = (name or "").lower()
    for sport, kws in _SPORT_KEYWORDS.items():
        if any(kw in n for kw in kws):
            return sport
    return ""


def _enrich_legs_with_lqs(legs: list, db: Session) -> dict:
    """
    Enrich each slip leg with LQS, close call history, and alt-pivot suggestion.
    Returns {legs: enriched_list, lqs_summary: dict}.

    Designed for manual bet slip entries where model_confidence is unknown —
    component B defaults to neutral (52% implied confidence).
    """
    scored: list  = []
    lqs_scores:   list  = []
    cc_total:     int   = 0
    pivot_total:  int   = 0

    for leg in legs:
        el = dict(leg)

        team_desc  = leg.get("team", "")
        market_raw = leg.get("market", "Moneyline")
        desc_lower = team_desc.lower()
        direction  = "over" if "over" in desc_lower else ("under" if "under" in desc_lower else "")
        sport      = _infer_sport_from_team(team_desc)
        am         = int(leg.get("american_odds") or -110)
        point      = leg.get("point")

        candidate = {
            "market_type":    market_raw,
            "sport":          sport,
            "team_or_player": team_desc,
            "direction":      direction,
            "odds":           am,
            "line":           point,
            # No model confidence for manual entries — neutral default applied in scorer
        }

        try:
            q = lq.compute_leg_quality_score(candidate, db)
            el.update({
                "lqs":               q["lqs"],
                "lqs_grade":         q["lqs_grade"],
                "lqs_warnings":      q["warnings"],
                "recommendation":    q["recommendation"],
                "component_scores":  q["component_scores"],
                "accuracy_profile":  q["accuracy_profile"],
                "close_call_history": q.get("close_call_history"),
            })
            lqs_scores.append(q["lqs"])
            cc = q.get("close_call_history")
            if cc and cc.get("close_call_count", 0) >= 2:
                cc_total += 1
        except Exception:
            pass

        # ── Alt pivot (Total / Spread only) ──────────────────────────────────
        mt_canon = lq._canonical_market(market_raw)
        el["alt_pivot"] = None
        if mt_canon in ("Total", "Spread", "Alt Spread") and point is not None:
            try:
                dir_use = direction or "over"
                if mt_canon == "Total":
                    alt_line = float(point) + 1.0 if dir_use == "under" else float(point) - 1.0
                    alt_am   = max(-350, am - 30)
                else:
                    alt_line = float(point) - 1.5
                    alt_am   = max(-350, am - 20)

                main_l = {"market_type": market_raw, "direction": dir_use,
                          "line": point, "odds": am, "sport": sport}
                alt_l  = {"market_type": market_raw, "direction": dir_use,
                          "line": alt_line, "odds": alt_am, "sport": sport}
                pivot  = lq.evaluate_alt_line_pivot(main_l, alt_l, 52.0, db)
                pivot["available"]    = True
                pivot["is_estimated"] = True
                el["alt_pivot"] = pivot
                if pivot.get("pivot_recommended"):
                    pivot_total += 1
            except Exception:
                pass

        scored.append(el)

    # ── Slip-level summary ────────────────────────────────────────────────────
    summary: dict = {}
    if lqs_scores:
        avg_lqs = round(sum(lqs_scores) / len(lqs_scores), 1)
        min_lqs = min(lqs_scores)
        weakest = lqs_scores.index(min_lqs)
        summary = {
            "avg_lqs":               avg_lqs,
            "min_lqs":               min_lqs,
            "weakest_leg_idx":       weakest,
            "weakest_leg_desc":      scored[weakest].get("team", "") if scored else "",
            "close_call_legs":       cc_total,
            "pivot_recommended_legs": pivot_total,
            "parlay_quality_grade":  (
                "A" if avg_lqs >= 80 and min_lqs >= 65 else
                "B" if avg_lqs >= 65 and min_lqs >= 50 else
                "C" if avg_lqs >= 50 else "D"
            ),
        }

    return {"legs": scored, "lqs_summary": summary}


@app.post("/api/bets/analyze-text")
def analyze_text(req: AnalyzeTextRequest, db: Session = Depends(get_db)):
    """
    Analyze a manually entered or pasted bet without needing the vision API.
    Takes structured legs and returns full EV analysis + BetIQ recommendation
    enriched with per-leg LQS, close call history, and alt-pivot suggestions.
    """
    n = req.n_legs or len(req.legs)
    # Add decimal_odds and implied_prob to each leg
    for leg in req.legs:
        am = leg.get("american_odds", -110)
        dec = (1 + am/100) if am > 0 else (1 + 100/abs(am))
        leg["decimal_odds"] = round(dec, 4)
        leg["implied_prob"] = round(1/dec*100, 1) if dec > 1 else 0

    parsed = {
        "sportsbook":            req.sportsbook,
        "bet_type":              req.bet_type,
        "n_legs":                n,
        "combined_american_odds":req.combined_american_odds,
        "legs":                  req.legs,
    }
    enriched = sp._enrich(parsed)

    # ── LQS + pivot enrichment ────────────────────────────────────────────────
    try:
        lqs_result = _enrich_legs_with_lqs(enriched.get("legs", []), db)
        enriched["legs"]        = lqs_result["legs"]
        enriched["lqs_summary"] = lqs_result["lqs_summary"]
    except Exception as _e:
        enriched["lqs_error"] = str(_e)

    return enriched


# ─── System 1: FanDuel Weekly Sync ───────────────────────────────────────────

@app.post("/api/sync/fanduel")
def sync_fanduel(
    imports_dir: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Trigger the FanDuel weekly CSV sync immediately.

    Looks for the most recent fanduel_*.csv in data/imports/ (then data/),
    finds newly settled bets not yet in the DB, imports them, and
    backfills cash-out decisions.

    After import, automatically runs the LQS self-calibration loop:
      backfill_lqs → update_quality_profiles → tune_weights → drift check → log.

    Returns: {status, file, new_bets, settled, skipped, errors, synced_at,
              calibration: {...}}
    """
    import fanduel_importer as fi
    result = fi.run_weekly_sync(imports_dir)
    # Also update the scheduler state so the widget shows "just now"
    sched._scheduler_state["last_fanduel_sync"]   = datetime.utcnow().strftime("%Y-%m-%d")
    sched._scheduler_state["last_fanduel_result"] = result

    # Self-calibration loop after every sync (best-effort — never block sync result)
    try:
        cal = lq.run_self_calibration(db)
        result["calibration"] = {
            "n_unbiased":    cal.get("n_unbiased"),
            "correlation":   cal.get("correlation"),
            "drift_detected": cal.get("drift_detected"),
            "drift_note":    cal.get("drift_note"),
            "profiles":      cal.get("profiles", {}).get("profiles_updated"),
            "backfill":      cal.get("backfill", {}).get("scored"),
        }
    except Exception as _ce:
        result["calibration"] = {"error": str(_ce)}

    return result


@app.post("/api/legs/calibrate")
def trigger_calibration(db: Session = Depends(get_db)):
    """
    Manually trigger the LQS self-calibration loop.
    Equivalent to what runs automatically after each FanDuel sync.

    Steps:
      1. backfill_lqs_on_bet_legs()
      2. update_quality_profiles()
      3. tune_lqs_weights() — unbiased only
      4. Compare weights, detect drift
      5. Write to lqs_calibration_log

    Returns full calibration summary.
    """
    return lq.run_self_calibration(db)


@app.get("/api/sync/fanduel/status")
def fanduel_sync_status():
    """Last FanDuel sync result and timestamp from the scheduler."""
    return {
        "last_sync":   sched._scheduler_state.get("last_fanduel_sync"),
        "last_result": sched._scheduler_state.get("last_fanduel_result"),
    }


# ─── FanDuel API Import (JSON — richer than CSV) ──────────────────────────────

import fanduel_api_importer as fd_api

class FanDuelImportRequest(BaseModel):
    token: str
    since_date: str = "2024-01-04"
    region: str = "IL"


@app.post("/api/fanduel/import-bets")
def fanduel_import_bets(req: FanDuelImportRequest):
    """
    Fetch all FanDuel settled bets via the sportsbook API, parse and upsert
    into fanduel_bets + fanduel_bet_legs tables, return analysis.

    Body:
      token       — x-authentication JWT (grab from browser console interceptor)
      since_date  — stop fetching before this date (default: 2024-01-04)
      region      — sportsbook region code (default: IL)

    Returns full analysis including boost performance, SGM split, player props.
    """
    try:
        result = fd_api.import_from_token(
            token      = req.token,
            since_date = req.since_date,
            region     = req.region,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fanduel/import-file")
async def fanduel_import_file(file: UploadFile = File(...)):
    """
    Upload a saved fanduel_bets_raw.json file and import it.
    Accepts: single API response page OR list of pages.
    """
    import sqlite3 as _sq
    content = await file.read()
    tmp = f"/tmp/fd_api_{uuid.uuid4()}.json"
    try:
        with open(tmp, "wb") as f:
            f.write(content)
        conn = _sq.connect(fd_api._BETS_DB)
        fd_api.create_tables(conn)
        result = fd_api.import_from_file(tmp, conn)
        analysis = fd_api.run_analysis(conn)
        conn.close()
        return {"upsert": result, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    """
    Combined real + simulated performance summary for the Dashboard tab.
    Returns real FanDuel bets stats, simulated mock bets by source,
    and a validation signal comparing exploration vs prospective win rates.
    """
    import sqlite3 as _sq
    from sqlalchemy import text as _sat

    # ── Real bets (FanDuel) ────────────────────────────────────────────────
    real: dict = {}
    try:
        conn = _sq.connect(fd_api._BETS_DB)
        conn.row_factory = _sq.Row
        fd_api.create_tables(conn)
        _fa = fd_api.run_analysis(conn)
        _s  = _fa.get("summary", {})
        total_r   = _s.get("total", 0) or 0
        wins_r    = _s.get("wins", 0) or 0
        settled_r = ((_s.get("wins") or 0) + (_s.get("losses") or 0) + (_s.get("cashed_out") or 0))
        wr_r      = round(wins_r / settled_r * 100, 1) if settled_r else None
        real = {
            "total":       total_r,
            "settled":     settled_r,
            "wins":        wins_r,
            "win_rate":    wr_r,
            "pnl":         _s.get("total_pnl"),
            "total_staked":_s.get("total_staked"),
            "since":       _s.get("earliest_bet"),
            "latest":      _s.get("latest_bet"),
            "by_boost":    _fa.get("boost_perf", []),
            "by_legs_boost": _fa.get("by_legs_boost", []),
            "sgm_split":   _fa.get("sgm_split", []),
        }
        conn.close()
    except Exception as _re:
        real = {"error": str(_re)}

    # ── Simulated bets (mock_bets by source) ──────────────────────────────
    _SIM_SOURCES = ("prospective", "prospective_pm", "top_picks_page",
                    "forced_generation", "retroactive_mock", "prospective_legacy")
    sim: dict = {}
    try:
        _rows = db.execute(_sat("""
            SELECT source,
                   COUNT(*) AS n,
                   SUM(CASE WHEN status='SETTLED_WIN' THEN 1 ELSE 0 END) AS wins,
                   ROUND(SUM(COALESCE(actual_profit,0)), 2) AS pnl,
                   ROUND(SUM(COALESCE(amount,0)), 2) AS staked,
                   MIN(date(generated_at)) AS since
            FROM mock_bets
            WHERE status IN ('SETTLED_WIN','SETTLED_LOSS')
            GROUP BY source
            ORDER BY n DESC
        """)).fetchall()
        by_source = {}
        total_sim = wins_sim = 0
        pnl_sim = staked_sim = 0.0
        for r in _rows:
            n = r[1] or 0; w = r[2] or 0
            wr = round(w / n * 100, 1) if n else None
            by_source[r[0]] = {
                "n": n, "wins": w, "win_rate": wr,
                "sim_pnl": r[3], "staked": r[4], "since": r[5],
            }
            if r[0] in _SIM_SOURCES:
                total_sim += n; wins_sim += w
                pnl_sim   += (r[3] or 0); staked_sim += (r[4] or 0)
        sim = {
            "total":    total_sim,
            "wins":     wins_sim,
            "win_rate": round(wins_sim / total_sim * 100, 1) if total_sim else None,
            "sim_pnl":  round(pnl_sim, 2),
            "by_source": by_source,
        }
    except Exception as _se:
        sim = {"error": str(_se)}

    # ── Validation signal: exploration vs prospective ──────────────────────
    val: dict = {}
    try:
        _exp = sim.get("by_source", {}).get("exploration", {})
        _pro = sim.get("by_source", {}).get("prospective", {})
        exp_wr = _exp.get("win_rate")
        pro_wr = _pro.get("win_rate")
        gap    = round(exp_wr - pro_wr, 1) if (exp_wr is not None and pro_wr is not None) else None
        n_exp  = _exp.get("n", 0)
        n_pro  = _pro.get("n", 0)
        if gap is not None and gap >= 20 and n_exp >= 50:
            interp = "Exploration system finding significantly higher WR — ALE calibration data ready."
        elif gap is not None and gap >= 10:
            interp = "Exploration ahead of prospective — building calibration signal."
        elif gap is not None:
            interp = "Exploration and prospective tracking closely — continue accumulating data."
        else:
            interp = "Insufficient data for comparison."
        val = {
            "exploration_wr": exp_wr, "exploration_n": n_exp,
            "prospective_wr": pro_wr, "prospective_n": n_pro,
            "gap": gap, "interpretation": interp,
            "calibration_ready": (n_exp >= 200),
        }
    except Exception as _ve:
        val = {"error": str(_ve)}

    return {"real_bets": real, "simulated_bets": sim, "validation_signal": val}


@app.get("/api/bet-history")
def bet_history(
    source: str  = Query("real", description="real | simulated"),
    limit:  int  = Query(50, ge=1, le=200),
    offset: int  = Query(0, ge=0),
    status: str  = Query(None, description="WON | LOST | OPEN (real) or SETTLED_WIN | SETTLED_LOSS (simulated)"),
    db: Session  = Depends(get_db),
):
    """
    Unified bet history — real FanDuel bets or simulated mock bets.

    source=real      → fanduel_bets + fanduel_bet_legs (most recent first)
    source=simulated → mock_bets + mock_bet_legs (most recent first)
    """
    import sqlite3 as _sq
    from sqlalchemy import text as _sat

    if source == "real":
        try:
            conn = _sq.connect(fd_api._BETS_DB)
            conn.row_factory = _sq.Row
            where = "WHERE 1=1"
            params: list = []
            if status:
                where += " AND b.result = ?"
                params.append(status)
            rows = conn.execute(f"""
                SELECT b.bet_receipt_id, b.bet_type, b.legs, b.stake, b.result,
                       b.pnl, b.placed_date, b.settled_date,
                       b.reward_type, b.boost_pct, b.is_sgm, b.odds_final
                FROM fanduel_bets b
                {where}
                ORDER BY b.placed_date DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset]).fetchall()

            result = []
            for r in rows:
                rid = r["bet_receipt_id"]
                legs_rows = conn.execute("""
                    SELECT leg_number, result, selection, market_type,
                           competition, price, is_player_selection, handicap, over_under
                    FROM fanduel_bet_legs WHERE bet_receipt_id = ?
                    ORDER BY leg_number
                """, (rid,)).fetchall()
                result.append({
                    "bet_id":       rid,
                    "type":         r["bet_type"],
                    "n_legs":       r["legs"],
                    "stake":        r["stake"],
                    "result":       r["result"],
                    "pnl":          r["pnl"],
                    "placed_date":  r["placed_date"],
                    "settled_date": r["settled_date"],
                    "boost_type":   r["reward_type"],
                    "boost_pct":    r["boost_pct"],
                    "is_sgm":       bool(r["is_sgm"]),
                    "odds":         r["odds_final"],
                    "legs": [{
                        "leg_num":    lg["leg_number"],
                        "result":     lg["result"],
                        "selection":  lg["selection"],
                        "market":     lg["market_type"],
                        "competition":lg["competition"],
                        "price":      lg["price"],
                        "is_prop":    bool(lg["is_player_selection"]),
                        "handicap":   lg["handicap"],
                        "over_under": lg["over_under"],
                    } for lg in legs_rows],
                })
            conn.close()
            return {"source": "real", "total": len(result), "bets": result}
        except Exception as _e:
            raise HTTPException(status_code=500, detail=str(_e))

    else:  # simulated
        from sqlalchemy import text as _sat2
        try:
            where_clauses = ["mb.status IN ('SETTLED_WIN','SETTLED_LOSS','PENDING')"]
            if status:
                where_clauses = [f"mb.status = '{status}'"]
            where_sql = " AND ".join(where_clauses)
            mock_rows = db.execute(_sat2(f"""
                SELECT mb.id, mb.source, mb.legs, mb.amount, mb.status,
                       mb.actual_profit, mb.generated_at, mb.game_date,
                       mb.odds, mb.predicted_win_prob, mb.avg_lqs, mb.sport,
                       mb.confidence, mb.promo_type, mb.promo_boost_pct,
                       mb.user_excluded, mb.user_excluded_reason,
                       mb.has_excluded_legs, mb.exclusion_mode_summary,
                       mb.recalculated_odds_decimal, mb.recalculated_combined_odds_american,
                       mb.recalculated_actual_profit, mb.counterfactual_message
                FROM mock_bets mb
                WHERE {where_sql}
                ORDER BY mb.generated_at DESC
                LIMIT :lim OFFSET :off
            """), {"lim": limit, "off": offset}).fetchall()

            result = []
            for r in mock_rows:
                bid = r[0]
                leg_rows = db.execute(_sat2("""
                    SELECT id, leg_index, description, market_type, sport,
                           win_prob, grade, model_used, is_alt_line,
                           open_odds, user_excluded, exclusion_mode
                    FROM mock_bet_legs WHERE mock_bet_id = :bid ORDER BY leg_index
                """), {"bid": bid}).fetchall()
                result.append({
                    "bet_id":       bid,
                    "source":       r[1],
                    "n_legs":       r[2],
                    "stake":        r[3],
                    "status":       r[4],
                    "sim_pnl":      r[5],
                    "generated_at": str(r[6]) if r[6] else None,
                    "game_date":    r[7],
                    "odds":         r[8],
                    "win_prob":     r[9],
                    "avg_lqs":      r[10],
                    "sport":        r[11],
                    "confidence":   r[12],
                    "boost_type":   r[13],
                    "boost_pct":    r[14],
                    "user_excluded":         bool(r[15]),
                    "user_excluded_reason":  r[16],
                    "has_excluded_legs":     bool(r[17]),
                    "exclusion_mode_summary": r[18],
                    "recalc_odds_decimal":   r[19],
                    "recalc_odds_american":  r[20],
                    "recalc_pnl":            r[21],
                    "counterfactual_message": r[22],
                    "legs": [{
                        "leg_id":      lg[0],
                        "leg_index":   lg[1],
                        "description": lg[2],
                        "market":      lg[3],
                        "sport":       lg[4],
                        "win_prob":    lg[5],
                        "grade":       lg[6],
                        "model":       lg[7],
                        "is_alt_line": bool(lg[8]),
                        "open_odds":   lg[9],
                        "user_excluded":  bool(lg[10]),
                        "exclusion_mode": lg[11],
                    } for lg in leg_rows],
                })
            return {"source": "simulated", "total": len(result), "bets": result}
        except Exception as _e:
            raise HTTPException(status_code=500, detail=str(_e))


# ── User Curation: Exclusions ─────────────────────────────────────────────────

class ExcludeBetRequest(BaseModel):
    type:       str  = "one_off"   # one_off | use_thesis
    reason:     Optional[str] = None
    thesis_id:  Optional[int] = None


@app.post("/api/mock-bets/{bet_id}/exclude")
def exclude_mock_bet(bet_id: str, req: ExcludeBetRequest, db: Session = Depends(get_db)):
    """
    Exclude a mock bet from performance metrics.
    type=one_off:    marks this bet only (no thesis linkage).
    type=use_thesis: marks this bet AND links to an existing thesis.
    """
    from sqlalchemy import text as _sat
    from datetime import datetime as _dt

    bet = db.get(MockBet, bet_id)
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")

    bet.user_excluded           = True
    bet.user_excluded_reason    = req.reason
    bet.user_excluded_at        = _dt.utcnow()
    bet.user_excluded_thesis_id = req.thesis_id

    # If linking to a thesis, increment its excluded count and update P&L accountability
    if req.thesis_id:
        thesis = db.get(UserThesis, req.thesis_id)
        if thesis:
            thesis.total_excluded_bets = (thesis.total_excluded_bets or 0) + 1
            # Accountability: check if bet is already settled
            if bet.status == "SETTLED_LOSS":
                thesis.excluded_pnl_avoided = round(
                    (thesis.excluded_pnl_avoided or 0) + abs(bet.actual_profit or bet.amount or 0), 2
                )
            elif bet.status == "SETTLED_WIN":
                thesis.excluded_pnl_missed = round(
                    (thesis.excluded_pnl_missed or 0) + abs(bet.actual_profit or 0), 2
                )
            thesis.net_value = round(
                (thesis.excluded_pnl_avoided or 0) - (thesis.excluded_pnl_missed or 0), 2
            )

    db.commit()
    return {"excluded": True, "bet_id": bet_id, "thesis_id": req.thesis_id}


@app.post("/api/mock-bets/{bet_id}/unexclude")
def unexclude_mock_bet(bet_id: str, db: Session = Depends(get_db)):
    """Remove user exclusion from a bet."""
    bet = db.get(MockBet, bet_id)
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    bet.user_excluded           = False
    bet.user_excluded_reason    = None
    bet.user_excluded_at        = None
    bet.user_excluded_thesis_id = None
    db.commit()
    return {"excluded": False, "bet_id": bet_id}


# ── User Curation: Leg-level Exclusions ───────────────────────────────────────

class ExcludeLegRequest(BaseModel):
    type:           str  = "one_off"      # one_off | use_thesis
    exclusion_mode: str  = "recalculate"  # null_bet | recalculate | counterfactual
    reason:         Optional[str] = None
    thesis_id:      Optional[int] = None


def _american_to_decimal(american_odds) -> Optional[float]:
    """Convert American odds integer to decimal odds. Returns None if invalid."""
    try:
        a = int(american_odds)
        if a >= 100:
            return round(a / 100 + 1, 6)
        elif a < 0:
            return round(100 / abs(a) + 1, 6)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def _decimal_to_american(decimal_odds) -> Optional[int]:
    """Convert decimal odds to nearest American odds integer."""
    try:
        d = float(decimal_odds)
        if d <= 1.0:
            return None
        if d >= 2.0:
            return int(round((d - 1) * 100))
        else:
            return int(round(-100 / (d - 1)))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def _recompute_recalculated_odds(bet: MockBet, db: Session) -> None:
    """
    Recompute recalculated_odds_decimal and _american for a bet by dividing out
    all excluded legs' decimal odds from the combined bet odds.

    Uses leg.open_odds (American at generation time) as the per-leg odds source.
    Falls back to bet.odds unchanged if any leg's open_odds is missing.
    """
    from database import MockBetLeg as _MBL
    all_legs = db.query(_MBL).filter(_MBL.mock_bet_id == bet.id).all()
    excluded = [l for l in all_legs if l.user_excluded and l.exclusion_mode == "recalculate"]
    if not excluded:
        bet.recalculated_odds_decimal = None
        bet.recalculated_combined_odds_american = None
        return

    combined = float(bet.odds or 1.0)
    for leg in excluded:
        dec = _american_to_decimal(leg.open_odds)
        if dec is None or dec <= 1.0:
            # Can't divide — bail out without storing
            return
        combined = combined / dec

    combined = max(1.001, round(combined, 4))
    bet.recalculated_odds_decimal = combined
    bet.recalculated_combined_odds_american = _decimal_to_american(combined)


@app.post("/api/mock-bet-legs/{leg_id}/exclude")
def exclude_mock_bet_leg(leg_id: int, req: ExcludeLegRequest, db: Session = Depends(get_db)):
    """
    Exclude a single leg from a mock bet with one of three modes:
      null_bet       — flag the parent bet as fully excluded (Mode A)
      recalculate    — keep bet active but recompute odds/P&L without this leg (Mode B)
      counterfactual — exclude bet from stats and generate a what-if message at settlement (Mode C)
    """
    from database import MockBetLeg as _MBL
    from datetime import datetime as _dt

    leg = db.get(_MBL, leg_id)
    if not leg:
        raise HTTPException(status_code=404, detail="Leg not found")

    valid_modes = {"null_bet", "recalculate", "counterfactual"}
    if req.exclusion_mode not in valid_modes:
        raise HTTPException(status_code=422, detail=f"exclusion_mode must be one of {valid_modes}")

    # Stamp the leg
    leg.user_excluded           = True
    leg.user_excluded_reason    = req.reason
    leg.user_excluded_at        = _dt.utcnow()
    leg.user_excluded_thesis_id = req.thesis_id
    leg.exclusion_mode          = req.exclusion_mode

    # Update the parent bet
    bet = db.get(MockBet, leg.mock_bet_id)
    if not bet:
        db.commit()
        return {"excluded": True, "leg_id": leg_id}

    bet.has_excluded_legs = True

    if req.exclusion_mode in ("null_bet", "counterfactual"):
        # Modes A & C: remove the whole bet from stats
        bet.user_excluded           = True
        bet.user_excluded_reason    = req.reason or f"leg excluded ({req.exclusion_mode})"
        bet.user_excluded_at        = _dt.utcnow()
        bet.user_excluded_thesis_id = req.thesis_id
    else:
        # Mode B: bet stays active; recompute odds
        _recompute_recalculated_odds(bet, db)

    # Update exclusion_mode_summary (most restrictive mode wins)
    db.flush()   # ensure leg changes are visible to the query below
    from database import MockBetLeg as _MBL2
    all_excl = db.query(_MBL2).filter(
        _MBL2.mock_bet_id == bet.id,
        _MBL2.user_excluded == True
    ).all()
    modes = {l.exclusion_mode for l in all_excl if l.exclusion_mode}
    if "null_bet" in modes:
        bet.exclusion_mode_summary = "null_bet"
    elif "counterfactual" in modes:
        bet.exclusion_mode_summary = "counterfactual"
    elif "recalculate" in modes:
        bet.exclusion_mode_summary = "recalculate"

    # Thesis accountability
    if req.thesis_id:
        thesis = db.get(UserThesis, req.thesis_id)
        if thesis:
            thesis.total_excluded_legs = (thesis.total_excluded_legs or 0) + 1
            if req.exclusion_mode == "recalculate":
                thesis.total_recalculated_bets = (thesis.total_recalculated_bets or 0) + 1
            else:
                thesis.total_excluded_bets = (thesis.total_excluded_bets or 0) + 1
            # Settled P&L accountability (if bet already settled)
            _update_thesis_pnl_for_leg(thesis, bet, leg, req.exclusion_mode)
            thesis.net_value = round(
                (thesis.excluded_pnl_avoided or 0) - (thesis.excluded_pnl_missed or 0), 2
            )

    db.commit()
    return {
        "excluded":       True,
        "leg_id":         leg_id,
        "bet_id":         leg.mock_bet_id,
        "exclusion_mode": req.exclusion_mode,
        "recalc_odds_decimal":  bet.recalculated_odds_decimal,
        "recalc_odds_american": bet.recalculated_combined_odds_american,
    }


@app.post("/api/mock-bet-legs/{leg_id}/unexclude")
def unexclude_mock_bet_leg(leg_id: int, db: Session = Depends(get_db)):
    """Remove leg-level exclusion and recompute parent bet state."""
    from database import MockBetLeg as _MBL
    leg = db.get(_MBL, leg_id)
    if not leg:
        raise HTTPException(status_code=404, detail="Leg not found")

    leg.user_excluded           = False
    leg.user_excluded_reason    = None
    leg.user_excluded_at        = None
    leg.user_excluded_thesis_id = None
    leg.exclusion_mode          = None

    bet = db.get(MockBet, leg.mock_bet_id)
    if bet:
        # Re-check remaining excluded legs
        from database import MockBetLeg as _MBL2
        remaining = db.query(_MBL2).filter(
            _MBL2.mock_bet_id == bet.id,
            _MBL2.user_excluded == True
        ).all()
        if not remaining:
            # No more excluded legs — clear all curation state
            bet.has_excluded_legs               = False
            bet.exclusion_mode_summary          = None
            bet.user_excluded                   = False
            bet.user_excluded_reason            = None
            bet.user_excluded_at                = None
            bet.recalculated_odds_decimal       = None
            bet.recalculated_combined_odds_american = None
        else:
            modes = {l.exclusion_mode for l in remaining if l.exclusion_mode}
            if "null_bet" in modes:
                bet.exclusion_mode_summary = "null_bet"
                bet.user_excluded = True
            elif "counterfactual" in modes:
                bet.exclusion_mode_summary = "counterfactual"
                bet.user_excluded = True
            else:
                bet.exclusion_mode_summary = "recalculate"
                bet.user_excluded = False
                _recompute_recalculated_odds(bet, db)

    db.commit()
    return {"excluded": False, "leg_id": leg_id}


def _update_thesis_pnl_for_leg(thesis, bet, leg, mode: str) -> None:
    """Update thesis P&L accountability when a leg exclusion is applied to an already-settled bet."""
    if bet.status not in ("SETTLED_WIN", "SETTLED_LOSS"):
        return  # Will be computed at settlement time instead
    stake = bet.amount or 0.0
    if mode == "recalculate":
        # Mode B: compare original outcome vs what recalculated bet outcome would be
        recalc_dec = bet.recalculated_odds_decimal
        if recalc_dec and recalc_dec > 1.0:
            orig_profit = bet.actual_profit or 0
            if bet.status == "SETTLED_WIN":
                recalc_profit = round((recalc_dec - 1) * stake, 2)
                diff = recalc_profit - orig_profit  # positive = recalc is better
                if diff > 0:
                    thesis.pnl_avoided_recalc = round((thesis.pnl_avoided_recalc or 0) + diff, 2)
                    thesis.excluded_pnl_avoided = round((thesis.excluded_pnl_avoided or 0) + diff, 2)
                else:
                    thesis.pnl_missed_recalc = round((thesis.pnl_missed_recalc or 0) + abs(diff), 2)
                    thesis.excluded_pnl_missed = round((thesis.excluded_pnl_missed or 0) + abs(diff), 2)
            else:  # SETTLED_LOSS
                # Recalc might win if excluded leg was the losing leg
                if leg.leg_result == "LOSS":
                    recalc_profit = round((recalc_dec - 1) * stake, 2)
                    total_saved = abs(orig_profit) + recalc_profit
                    thesis.pnl_avoided_recalc = round((thesis.pnl_avoided_recalc or 0) + total_saved, 2)
                    thesis.excluded_pnl_avoided = round((thesis.excluded_pnl_avoided or 0) + total_saved, 2)
                else:
                    thesis.pnl_avoided_recalc = round((thesis.pnl_avoided_recalc or 0) + abs(orig_profit), 2)
                    thesis.excluded_pnl_avoided = round((thesis.excluded_pnl_avoided or 0) + abs(orig_profit), 2)
    else:
        # Mode A / C: whole bet excluded from stats
        if bet.status == "SETTLED_LOSS":
            avoided = abs(bet.actual_profit or stake)
            thesis.pnl_avoided_null = round((thesis.pnl_avoided_null or 0) + avoided, 2)
            thesis.excluded_pnl_avoided = round((thesis.excluded_pnl_avoided or 0) + avoided, 2)
        elif bet.status == "SETTLED_WIN":
            missed = abs(bet.actual_profit or 0)
            thesis.pnl_missed_null = round((thesis.pnl_missed_null or 0) + missed, 2)
            thesis.excluded_pnl_missed = round((thesis.excluded_pnl_missed or 0) + missed, 2)


# ── User Curation: Theses ─────────────────────────────────────────────────────

class ThesisCreateRequest(BaseModel):
    thesis_type:          str  = "matchup"
    title:                str
    description:          Optional[str] = None
    sport:                Optional[str] = None
    team:                 Optional[str] = None
    opponent:             Optional[str] = None
    player:               Optional[str] = None
    market_filters:       Optional[dict] = None   # {block:[...], alt_spreads_min_line:25}
    expires_at:           Optional[str] = None    # ISO date string
    expire_after_games:   Optional[int] = None


@app.post("/api/theses")
def create_thesis(req: ThesisCreateRequest, db: Session = Depends(get_db)):
    """Create a new user handicapping thesis."""
    import json as _json
    from datetime import datetime as _dt

    expires_dt = None
    if req.expires_at:
        try:
            expires_dt = _dt.fromisoformat(req.expires_at)
        except Exception:
            pass

    thesis = UserThesis(
        thesis_type        = req.thesis_type,
        title              = req.title,
        description        = req.description,
        sport              = req.sport,
        team               = req.team,
        opponent           = req.opponent,
        player             = req.player,
        market_filters     = _json.dumps(req.market_filters) if req.market_filters else None,
        active             = True,
        expires_at         = expires_dt,
        expire_after_games = req.expire_after_games,
        next_review_at     = expires_dt,
    )
    db.add(thesis)
    db.commit()
    db.refresh(thesis)
    return _thesis_to_dict(thesis)


@app.get("/api/theses")
def list_theses(active: Optional[bool] = None, db: Session = Depends(get_db)):
    """List theses, optionally filtered by active status."""
    from datetime import datetime as _dt
    q = db.query(UserThesis)
    if active is not None:
        q = q.filter(UserThesis.active == active)

    # Auto-expire theses whose expires_at has passed
    now = _dt.utcnow()
    rows = q.order_by(UserThesis.created_at.desc()).all()
    for t in rows:
        if t.active and t.expires_at and t.expires_at < now:
            t.active = False
    db.commit()

    return [_thesis_to_dict(t) for t in rows]


@app.get("/api/theses/{thesis_id}")
def get_thesis(thesis_id: int, db: Session = Depends(get_db)):
    t = db.get(UserThesis, thesis_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thesis not found")
    return _thesis_to_dict(t)


@app.patch("/api/theses/{thesis_id}/disable")
def disable_thesis(thesis_id: int, db: Session = Depends(get_db)):
    """Soft-deactivate a thesis. Future picks no longer filtered; past history retained."""
    t = db.get(UserThesis, thesis_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thesis not found")
    t.active = False
    db.commit()
    return {"disabled": True, "id": thesis_id}


@app.patch("/api/theses/{thesis_id}/renew")
def renew_thesis(thesis_id: int, req: dict = Body(default={}), db: Session = Depends(get_db)):
    """Extend or reactivate a thesis. Optionally update expires_at."""
    from datetime import datetime as _dt
    t = db.get(UserThesis, thesis_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thesis not found")
    t.active      = True
    t.reviewed_at = _dt.utcnow()
    if req.get("expires_at"):
        try:
            t.expires_at    = _dt.fromisoformat(req["expires_at"])
            t.next_review_at = t.expires_at
        except Exception:
            pass
    if req.get("expire_after_games") is not None:
        t.expire_after_games = req["expire_after_games"]
    db.commit()
    return _thesis_to_dict(t)


def _thesis_to_dict(t: "UserThesis") -> dict:
    import json as _json
    from datetime import datetime as _dt

    # Compute verdict
    avoided = t.excluded_pnl_avoided or 0
    missed  = t.excluded_pnl_missed  or 0
    net     = t.net_value or 0
    if t.total_excluded_bets and t.total_excluded_bets > 0:
        if net >= 5:
            verdict = "Adding value"
        elif net <= -5:
            verdict = "Costing money"
        else:
            verdict = "Even"
    else:
        verdict = "No data yet"

    # Games remaining if expire_after_games set
    games_info = None
    if t.expire_after_games:
        played   = t.games_filtered_count or 0
        remaining = max(0, t.expire_after_games - played)
        games_info = f"after {t.expire_after_games} games ({played} played, {remaining} remaining)"

    mf = None
    if t.market_filters:
        try:
            mf = _json.loads(t.market_filters)
        except Exception:
            mf = t.market_filters

    return {
        "id":              t.id,
        "thesis_type":     t.thesis_type,
        "title":           t.title,
        "description":     t.description,
        "sport":           t.sport,
        "team":            t.team,
        "opponent":        t.opponent,
        "player":          t.player,
        "market_filters":  mf,
        "active":          t.active,
        "created":         t.created_at.date().isoformat() if t.created_at else None,
        "expires":         t.expires_at.isoformat() if t.expires_at else games_info,
        "games_filtered":  t.games_filtered_count or 0,
        "bets_excluded":   t.total_excluded_bets  or 0,
        "legs_excluded":   t.total_excluded_legs   or 0,
        "bets_recalculated": t.total_recalculated_bets or 0,
        "pnl_avoided":     round(avoided, 2),
        "pnl_missed":      round(missed,  2),
        "net_value":       round(net,     2),
        "verdict":         verdict,
        "mode_breakdown": {
            "null_bet": {
                "bets":        t.total_excluded_bets or 0,
                "pnl_avoided": round(t.pnl_avoided_null or 0, 2),
                "pnl_missed":  round(t.pnl_missed_null  or 0, 2),
            },
            "recalculate": {
                "bets":        t.total_recalculated_bets or 0,
                "pnl_avoided": round(t.pnl_avoided_recalc or 0, 2),
                "pnl_missed":  round(t.pnl_missed_recalc  or 0, 2),
            },
        },
        "reviewed_at":     t.reviewed_at.isoformat() if t.reviewed_at else None,
        "next_review_at":  t.next_review_at.isoformat() if t.next_review_at else None,
    }


@app.get("/api/curation/summary")
def curation_summary(db: Session = Depends(get_db)):
    """Dashboard curation card: active theses, weekly excluded bets and P&L accountability."""
    from datetime import datetime as _dt, timedelta as _td
    from sqlalchemy import text as _sat

    week_ago = (_dt.utcnow() - _td(days=7)).isoformat()

    # Active theses
    active_theses = db.query(UserThesis).filter(UserThesis.active == True).all()

    # Weekly exclusion stats from mock_bets
    try:
        _row = db.execute(_sat("""
            SELECT
              COUNT(*) as total_excluded,
              SUM(CASE WHEN status='SETTLED_LOSS' THEN ABS(COALESCE(actual_profit,amount,0)) ELSE 0 END) as pnl_avoided,
              SUM(CASE WHEN status='SETTLED_WIN'  THEN ABS(COALESCE(actual_profit,0)) ELSE 0 END) as pnl_missed
            FROM mock_bets
            WHERE user_excluded=1 AND user_excluded_at >= :wa
        """), {"wa": week_ago}).fetchone()
        weekly = {
            "bets_excluded":  _row[0] or 0,
            "pnl_avoided":    round(_row[1] or 0, 2),
            "pnl_missed":     round(_row[2] or 0, 2),
            "net_value":      round((_row[1] or 0) - (_row[2] or 0), 2),
        }
    except Exception:
        weekly = {"bets_excluded": 0, "pnl_avoided": 0, "pnl_missed": 0, "net_value": 0}

    # All-time stats
    try:
        _all = db.execute(_sat("""
            SELECT
              COUNT(*) as total_excluded,
              SUM(CASE WHEN status='SETTLED_LOSS' THEN ABS(COALESCE(actual_profit,amount,0)) ELSE 0 END) as pnl_avoided,
              SUM(CASE WHEN status='SETTLED_WIN'  THEN ABS(COALESCE(actual_profit,0)) ELSE 0 END) as pnl_missed
            FROM mock_bets WHERE user_excluded=1
        """)).fetchone()
        alltime = {
            "bets_excluded": _all[0] or 0,
            "pnl_avoided":   round(_all[1] or 0, 2),
            "pnl_missed":    round(_all[2] or 0, 2),
            "net_value":     round((_all[1] or 0) - (_all[2] or 0), 2),
        }
    except Exception:
        alltime = {"bets_excluded": 0, "pnl_avoided": 0, "pnl_missed": 0, "net_value": 0}

    # Per-bet-value metric
    n = weekly["bets_excluded"]
    per_bet = round(weekly["net_value"] / n, 2) if n else None

    # System suggestion
    suggestion = None
    if per_bet is not None and per_bet < -5 and n >= 10:
        suggestion = "Your overrides have cost more than they've saved recently. Consider trusting the model more on team-level vetoes."

    theses_summary = [_thesis_to_dict(t) for t in active_theses]

    return {
        "active_theses":   theses_summary,
        "weekly":          weekly,
        "alltime":         alltime,
        "per_bet_value":   per_bet,
        "suggestion":      suggestion,
    }


@app.get("/api/fanduel/analysis")
def fanduel_analysis():
    """
    Return the five analysis queries on the fanduel_bets / fanduel_bet_legs tables
    without re-fetching — uses whatever data is already imported.
    """
    import sqlite3 as _sq
    try:
        conn = _sq.connect(fd_api._BETS_DB)
        conn.row_factory = _sq.Row
        fd_api.create_tables(conn)
        result = fd_api.run_analysis(conn)
        conn.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── System 2: Leg Attribution Analysis ──────────────────────────────────────

import leg_attribution as la

@app.get("/api/attribution/legs-60d")
def legs_attribution_60d(
    days: int = Query(60, ge=7, le=365),
    db: Session = Depends(get_db),
):
    """
    Leg-level win/loss attribution for the last N days.

    Parses bet_legs rows and bet_info strings to surface which market types,
    sports, and odds ranges perform best and worst.

    Returns: {by_market, by_sport, by_odds, top_losing, summary}
    """
    return la.analyze_legs_60d(db, days=days)


@app.get("/api/attribution/leg/{bet_id}/{leg_index}")
def leg_detail(
    bet_id:    str,
    leg_index: int,
    db: Session = Depends(get_db),
):
    """
    Detail + historical win rate context for a specific leg of a bet.
    """
    result = la.get_leg_detail(db, bet_id, leg_index)
    if result is None:
        raise HTTPException(status_code=404, detail="Bet or leg not found")
    return result


@app.post("/api/attribution/backfill-legs")
def backfill_legs(db: Session = Depends(get_db)):
    """
    Enrich bet_legs table from stored bet_info strings.

    - Creates missing bet_legs for any bet that has bet_info but no leg rows.
    - Enriches existing legs with odds_str (FanDuel format), team, and fixed market_type.
    - Infers leg_result for straight bets (legs=1) from the parlay outcome (100% reliable).
    - Parlay leg_result stays NULL — can't determine without FanDuel per-leg grading.
    """
    from fanduel_importer import backfill_bet_legs_from_bet_info
    result = backfill_bet_legs_from_bet_info(db)

    # Report current state after backfill
    from sqlalchemy import text as sqla_text
    totals = db.execute(sqla_text("""
        SELECT
            COUNT(*) as total_legs,
            SUM(CASE WHEN odds_str IS NOT NULL AND odds_str != '' THEN 1 ELSE 0 END) as legs_with_odds,
            SUM(CASE WHEN leg_result IS NOT NULL THEN 1 ELSE 0 END) as legs_with_result,
            SUM(CASE WHEN team IS NOT NULL THEN 1 ELSE 0 END) as legs_with_team
        FROM bet_legs
        WHERE bet_id NOT IN (SELECT id FROM bets WHERE is_mock=1)
    """)).fetchone()

    sport_breakdown = db.execute(sqla_text("""
        SELECT sport, COUNT(*) as cnt
        FROM bet_legs
        WHERE bet_id NOT IN (SELECT id FROM bets WHERE is_mock=1)
        GROUP BY sport ORDER BY cnt DESC LIMIT 15
    """)).fetchall()

    market_breakdown = db.execute(sqla_text("""
        SELECT market_type, COUNT(*) as cnt
        FROM bet_legs
        WHERE bet_id NOT IN (SELECT id FROM bets WHERE is_mock=1)
        GROUP BY market_type ORDER BY cnt DESC
    """)).fetchall()

    return {
        **result,
        "current_totals": {
            "total_legs":       totals[0],
            "legs_with_odds":   totals[1],
            "legs_with_result": totals[2],
            "legs_with_team":   totals[3],
            "odds_coverage_pct": round(totals[1] / max(totals[0], 1) * 100, 1),
            "result_coverage_pct": round(totals[2] / max(totals[0], 1) * 100, 1),
        },
        "by_sport":  [{"sport": r[0], "legs": r[1]} for r in sport_breakdown],
        "by_market": [{"market_type": r[0], "legs": r[1]} for r in market_breakdown],
    }


@app.post("/api/attribution/resolve-legs")
def resolve_legs(
    overwrite: bool = Query(False, description="Re-resolve legs that already have a result"),
    db: Session = Depends(get_db),
):
    """
    Retrospective leg outcome resolver.

    Cross-references bet_legs against historical.db game scores to populate:
      - leg_result        (WIN / LOSS / PUSH)
      - accuracy_delta    (margin: positive = cushion, negative = miss)
      - resolution_source (historical_db / pitcher_logs / inferred_parlay_win)
      - actual_value      (actual score or stat used)

    Resolution hierarchy:
      1. historical_db game scores (Moneyline, Spread, Total)
      2. pitcher_game_logs (MLB strikeout props)
      3. inferred_parlay_win (all legs of SETTLED_WIN parlay)

    Returns a summary dict with counts by resolution method and sport.
    """
    import leg_resolver as lr
    hist_db_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
    if not os.path.exists(hist_db_path):
        raise HTTPException(
            status_code=503,
            detail=f"historical.db not found at {hist_db_path}. Run MLB/NBA ingest first."
        )
    result = lr.resolve_all_legs(db, hist_db_path=hist_db_path, overwrite=overwrite)
    # Score any newly resolved legs and rebuild quality profiles
    try:
        import leg_quality as lq
        backfill_result = lq.backfill_lqs_on_bet_legs(db)
        profile_result  = lq.update_quality_profiles(db)
        result["lqs_scored"]             = backfill_result["scored"]
        result["quality_profiles_updated"] = profile_result["profiles_updated"]
    except Exception as _e:
        result["quality_profiles_error"] = str(_e)
    return result


@app.post("/api/attribution/backfill-soccer")
def backfill_soccer(
    date: Optional[str] = Query(None, description="YYYY-MM-DD — omit to auto-detect from legs"),
    db: Session = Depends(get_db),
):
    """
    Fetch API-Football v3 match results for all unresolved soccer leg dates
    and resolve them.

    Date detection order per leg:
      1. leg.game_commence_time[:10]  (set at placement time)
      2. parent bet.time_placed[:10]  (fallback when game_commence_time is NULL)
      3. explicit ?date= query param  (always included)

    One API call per unique date.  Stores results in soccer_results table then
    re-runs soccer resolution on all unresolved bet_legs.

    Note: API-Football free plan only allows yesterday / today / tomorrow.
    Legs with game dates outside that window will return 0 fixtures.

    Returns: dates_attempted, fixtures_stored, legs_resolved, legs_skipped,
             unresolvable, by_sport, plan_note
    """
    from soccer_data import fetch_soccer_results, store_soccer_results
    import leg_resolver as lr
    from datetime import date as _date
    from sqlalchemy.orm import joinedload

    # ── Build date set from unresolved soccer legs ────────────────────────────
    # We need the parent bet for time_placed fallback, so load legs + bets together
    from sqlalchemy import text as _text

    rows = db.execute(_text("""
        SELECT bl.id, bl.game_commence_time, b.time_placed
        FROM bet_legs bl
        JOIN bets b ON b.id = bl.bet_id
        WHERE bl.sport = 'Soccer'
          AND bl.leg_result IS NULL
    """)).fetchall()

    dates_to_fetch: set[str] = set()

    # Add explicit override date if provided
    if date:
        dates_to_fetch.add(date)

    for leg_id, game_commence_time, time_placed in rows:
        d: Optional[str] = None
        # Prefer game_commence_time
        if game_commence_time:
            try:
                d = str(game_commence_time)[:10]
            except Exception:
                pass
        # Fall back to parent bet's time_placed
        if not d and time_placed:
            try:
                d = str(time_placed)[:10]
            except Exception:
                pass
        if d:
            # Also queue ±1 day so resolver's fallback window has cached data
            from datetime import timedelta as _td, date as _ddate
            try:
                base = _ddate.fromisoformat(d)
                dates_to_fetch.add(d)
                dates_to_fetch.add((base + _td(days=1)).isoformat())
                dates_to_fetch.add((base - _td(days=1)).isoformat())
            except ValueError:
                dates_to_fetch.add(d)

    # ── Skip dates already fully cached ──────────────────────────────────────
    from database import SoccerResult as _SR
    from sqlalchemy import text as _t2
    import time as _time

    already_cached: set[str] = {
        r[0] for r in db.execute(_t2("SELECT DISTINCT date FROM soccer_results")).fetchall()
    }
    uncached_dates = sorted(d for d in dates_to_fetch if d not in already_cached)
    cached_dates   = sorted(d for d in dates_to_fetch if d in already_cached)

    print(
        f"[backfill-soccer] {len(rows)} unresolved soccer legs → "
        f"{len(dates_to_fetch)} unique dates "
        f"({len(cached_dates)} already cached, {len(uncached_dates)} to fetch)"
    )

    # ── Fetch + store (rate-limited: 7 s between new fdorg requests) ──────────
    fetch_log: list[dict] = []
    total_fixtures = 0

    for i, d in enumerate(uncached_dates):
        results = fetch_soccer_results(d)
        n = store_soccer_results(results, db)
        total_fixtures += n
        fetch_log.append({"date": d, "fixtures_stored": n, "source": "new"})
        # Respect fdorg 10 req/min rate limit — sleep between every call
        if i < len(uncached_dates) - 1:
            _time.sleep(7)

    # Cached dates: upsert only (no API call needed)
    for d in cached_dates:
        fetch_log.append({"date": d, "fixtures_stored": "cached", "source": "cached"})

    print(f"[backfill-soccer] Stored {total_fixtures} new fixtures ({len(cached_dates)} dates skipped — already cached)")

    # ── Re-run leg resolution ─────────────────────────────────────────────────
    hist_db_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
    resolve_result = lr.resolve_all_legs(
        db,
        hist_db_path=hist_db_path if os.path.exists(hist_db_path) else None,
        overwrite=False,
    )

    print(f"[backfill-soccer] Resolution: {resolve_result}")

    return {
        "dates_attempted":  sorted(dates_to_fetch),
        "fetch_log":        fetch_log,
        "fixtures_stored":  total_fixtures,
        "legs_resolved":    resolve_result.get("resolved", 0),
        "legs_skipped":     resolve_result.get("skipped_no_sport", 0),
        "unresolvable":     resolve_result.get("unresolvable", 0),
        "by_sport":         resolve_result.get("by_sport", {}),
        "plan_note": (
            "API-Football free plan allows yesterday/today/tomorrow only. "
            "Legs with older game dates will not resolve via this endpoint."
        ),
    }


# ─── Leg Quality Score ────────────────────────────────────────────────────────

import leg_quality as lq

class LegCandidateRequest(BaseModel):
    market_type:      str
    sport:            Optional[str] = None
    team_or_player:   Optional[str] = None
    odds:             Optional[int] = None
    model_confidence: Optional[float] = None
    model_used:       Optional[str]  = None
    edge_pp:          Optional[float] = None
    line:             Optional[float] = None


@app.post("/api/legs/quality-score")
def leg_quality_score(
    req: LegCandidateRequest,
    db: Session = Depends(get_db),
):
    """
    Score a single candidate leg 0–100 using historical accuracy + model confidence.

    Returns: lqs, lqs_grade, recommendation (ADD/CONSIDER/AVOID),
             component_scores, accuracy_profile, warnings.
    """
    return lq.compute_leg_quality_score(req.dict(), db)


@app.post("/api/legs/quality-score-batch")
def leg_quality_score_batch(
    legs: List[LegCandidateRequest],
    db:   Session = Depends(get_db),
):
    """
    Score a list of candidate legs in one call.
    Returns: list of {lqs, lqs_grade, recommendation, component_scores, warnings}
    """
    return [lq.compute_leg_quality_score(leg.dict(), db) for leg in legs]


@app.get("/api/legs/deviation-profiles")
def leg_deviation_profiles(db: Session = Depends(get_db)):
    """
    All leg_quality_profiles rows sorted by sample_size descending.
    Shows which leg types are most/least predictable.
    """
    from database import LegQualityProfile as LQP
    rows = db.query(LQP).order_by(LQP.sample_size.desc()).all()
    return [
        {
            "id":               r.id,
            "market_type":      r.market_type,
            "sport":            r.sport,
            "team_or_player":   r.team_or_player,
            "mean_delta":       r.mean_delta,
            "std_delta":        r.std_delta,
            "p25_delta":        r.p25_delta,
            "p75_delta":        r.p75_delta,
            "win_rate":         r.win_rate,
            "close_loss_rate":  r.close_loss_rate,
            "bad_loss_rate":    r.bad_loss_rate,
            "consistency_score": r.consistency_score,
            "sample_size":      r.sample_size,
            "last_updated":     r.last_updated,
        }
        for r in rows
    ]


@app.post("/api/legs/update-profiles")
def update_leg_profiles(db: Session = Depends(get_db)):
    """
    Rebuild leg_quality_profiles from all resolved bet_legs.
    Call after /api/attribution/resolve-legs or weekly FanDuel sync.
    """
    return lq.update_quality_profiles(db)


@app.post("/api/legs/tune-weights")
def tune_lqs_weights(db: Session = Depends(get_db)):
    """
    Correlate stored LQS against actual leg outcomes to validate predictive value.
    Reports: lqs_win_corr, mean LQS when won vs lost, grade-level win rates.
    Requires 100+ resolved legs with LQS stored.
    """
    return lq.tune_lqs_weights(db)


@app.get("/api/legs/quality-attribution")
def leg_quality_attribution(db: Session = Depends(get_db)):
    """
    LQS distribution (grade A/B/C/D), close calls (delta > -1.0 losses),
    and bad misses (delta < -3.0 losses) for the attribution UI.
    """
    return lq.get_lqs_attribution(db)


@app.post("/api/legs/backfill-lqs")
def backfill_lqs(db: Session = Depends(get_db)):
    """
    Retroactively populate lqs / lqs_grade on existing bet_legs that have no score.
    Then rebuilds leg_quality_profiles so the attribution page reflects the new data.

    Returns: {scored, skipped, errors, bets_updated, profiles_updated, profiles_skipped}
    """
    backfill_result = lq.backfill_lqs_on_bet_legs(db)
    profile_result  = lq.update_quality_profiles(db)
    return {
        **backfill_result,
        "profiles_updated": profile_result["profiles_updated"],
        "profiles_skipped": profile_result["profiles_skipped"],
    }


# ─── Close Call Reduction / Alt-Line Pivot ────────────────────────────────────

class PivotEvalRequest(BaseModel):
    main_market_type: str
    main_direction:   str
    main_line:        float
    main_odds:        int
    main_prob:        float
    alt_line:         float
    alt_odds:         int
    sport:            Optional[str] = None


class CloseCallCheckRequest(BaseModel):
    market_type: str
    sport:       Optional[str]  = None
    direction:   Optional[str]  = None
    line:        Optional[float] = None


class PivotSuggestionRequest(BaseModel):
    market_type:  str
    market_key:   Optional[str]   = None  # raw key: "totals", "spreads"
    direction:    Optional[str]   = None  # "over" / "under" / team name
    line:         float
    odds:         float                   # decimal (≥1.1) OR American (<0)
    sport:        Optional[str]   = None
    sport_key:    Optional[str]   = None  # "baseball_mlb" etc for alt line lookup
    fixture_id:   Optional[str]   = None
    win_prob:     Optional[float] = None  # model probability 0-100
    leg_id:       Optional[str]   = None


@app.post("/api/legs/evaluate-pivot")
def evaluate_pivot_endpoint(req: PivotEvalRequest, db: Session = Depends(get_db)):
    """
    Compare a specific main line vs alt line on cushion-adjusted EV.
    Uses the model probability supplied by caller.

    Returns: main_ev, alt_ev, ev_improvement, prob_jump_pp,
             pivot_recommended, pivot_reason, cushion_units, odds_penalty.
    """
    main_leg = {
        "market_type": req.main_market_type,
        "direction":   req.main_direction,
        "line":        req.main_line,
        "odds":        req.main_odds,
        "sport":       req.sport,
    }
    alt_leg = {
        "market_type": req.main_market_type,
        "direction":   req.main_direction,
        "line":        req.alt_line,
        "odds":        req.alt_odds,
        "sport":       req.sport,
    }
    return lq.evaluate_alt_line_pivot(main_leg, alt_leg, req.main_prob, db)


@app.post("/api/legs/check-close-calls")
def check_close_calls_endpoint(req: CloseCallCheckRequest, db: Session = Depends(get_db)):
    """
    Return close call history for a leg profile: how often have similar legs
    just missed (accuracy_delta between -1.0 and 0)?
    """
    return lq.check_close_call_history(
        req.market_type, req.sport or "", req.direction or "", req.line or 0.0, db
    )


@app.post("/api/legs/pivot-suggestion")
def pivot_suggestion_endpoint(req: PivotSuggestionRequest, db: Session = Depends(get_db)):
    """
    All-in-one pivot analysis for a parlay builder leg.

    1. Tries to find an actual adjacent alt line in the alt_lines DB for the event.
    2. Falls back to a synthesised adjacent line if none stored.
    3. Runs evaluate_alt_line_pivot() and check_close_call_history().

    Returns pivot analysis + close call history + is_estimated flag.
    """
    import creator_tier as ct

    mt_canonical = lq._canonical_market(req.market_type)
    if mt_canonical not in ("Total", "Spread", "Alt Spread"):
        return {"pivot_available": False, "reason": "Not a pivotable market type (Total/Spread only)"}

    # ── Main probability ────────────────────────────────────────────────────
    main_prob_pct = req.win_prob or 52.0
    main_prob = main_prob_pct / 100.0

    # ── Convert main odds to American ───────────────────────────────────────
    raw = float(req.odds)
    if 1.01 <= raw <= 30.0:                      # decimal
        main_am = lq._dec_to_american(raw)
    else:
        main_am = int(raw)

    direction = (req.direction or "over").lower()

    # ── Look up actual alt line from historical.db ──────────────────────────
    alt_am:       Optional[int]   = None
    alt_line_val: Optional[float] = None
    is_estimated  = True

    if req.fixture_id:
        try:
            lines = ct.get_scored_alt_lines(
                req.fixture_id, main_prob, req.line, min_label="ALL"
            )
            # Find closest adjacent line for the right direction
            best_shift = None
            for row in lines:
                row_dir  = (row.get("over_under") or "").lower()
                row_line = row.get("line")
                if row_line is None:
                    continue
                if row_dir and row_dir != direction:
                    continue   # wrong direction
                shift = abs(float(row_line) - float(req.line))
                if shift < 0.1:
                    continue   # same line
                if shift > 3.0:
                    continue   # too far
                if best_shift is None or shift < best_shift:
                    best_shift    = shift
                    alt_line_val  = float(row_line)
                    row_dec       = float(row.get("odds", 1.909))
                    alt_am        = lq._dec_to_american(row_dec)
                    is_estimated  = False
        except Exception:
            pass

    # ── Synthesise adjacent line if nothing found ───────────────────────────
    if alt_am is None:
        if mt_canonical == "Total":
            if direction == "under":
                alt_line_val = req.line + 1.0   # higher line = easier Under
            else:
                alt_line_val = req.line - 1.0   # lower line = easier Over
            alt_am = main_am - 30               # typical ~30-pt juice increase
        else:   # Spread / Alt Spread
            alt_line_val = req.line - 1.5       # shift 1.5 pts easier
            alt_am = main_am - 20

    # Guard: never worse than -350
    alt_am = max(-350, alt_am)

    main_leg = {
        "market_type": req.market_type,
        "direction":   direction,
        "line":        req.line,
        "odds":        main_am,
        "sport":       req.sport or "",
    }
    alt_leg = {
        "market_type": req.market_type,
        "direction":   direction,
        "line":        alt_line_val,
        "odds":        alt_am,
        "sport":       req.sport or "",
    }

    pivot  = lq.evaluate_alt_line_pivot(main_leg, alt_leg, main_prob_pct, db)
    cc     = lq.check_close_call_history(
        req.market_type, req.sport or "", direction, req.line, db
    )

    return {
        "pivot_available":    True,
        "is_estimated":       is_estimated,
        "close_call_history": cc,
        **pivot,
    }


# ─── System 3: Mock Bet Training Loop ────────────────────────────────────────

import mock_bets as mb_module

class MockBetGenerateRequest(BaseModel):
    stake:    float = 10.0
    n_picks:  int   = 5
    max_legs: int   = 4
    source:   str   = "prospective"


def _run_mock_gen_background(req_stake: float, req_n_picks: int, req_max_legs: int, req_source: str) -> None:
    """Background worker: runs full mock-bet generation, stores result in module globals."""
    global _mock_gen_in_progress, _mock_gen_result, _mock_gen_error
    from database import SessionLocal as _SL
    _bg_db = _SL()
    try:
        _cached = _get_cached_picks()

        if req_source == "top_picks_page":
            # Seed top_picks_page bets directly from the picks cache so the
            # simulation tracks exactly what the user sees on the recommendations
            # page — bypasses the prospective-source dedup that blocks regeneration.
            # Stakes are 25% Kelly on $500 bankroll per pick.
            _cache_picks = _cached.get("picks", []) if _cached else []
            result = mb_module.seed_top_picks_mock_bets(
                _cache_picks,
                _bg_db,
                kelly_fraction = 0.25,
                bankroll       = 500.0,
            )
            expl = mb_module.generate_exploration_bets(_bg_db, run_id=result.get("run_id"))
            result["exploration"] = expl
            try:
                tiered = mb_module.generate_tiered_exploration_bets(_bg_db, run_id=result.get("run_id"))
                result["exploration_tiered"] = tiered
            except Exception as _te:
                result["exploration_tiered"] = {"error": str(_te)}
        else:
            _tier_b = _cached.get("tier_b", []) if _cached else []
            result  = mb_module.generate_mock_bets(
                _bg_db,
                stake        = req_stake,
                n_picks      = req_n_picks,
                max_legs     = req_max_legs,
                source       = req_source,
                tier_b_picks = _tier_b,
            )
            expl = mb_module.generate_exploration_bets(_bg_db, run_id=result.get("run_id"))
            result["exploration"] = expl
            try:
                tiered = mb_module.generate_tiered_exploration_bets(_bg_db, run_id=result.get("run_id"))
                result["exploration_tiered"] = tiered
            except Exception as _te:
                result["exploration_tiered"] = {"error": str(_te)}

        _mock_gen_result = result
        _mock_gen_error  = None
        print(f"[mock-gen] background complete — generated={result.get('generated')} run_id={result.get('run_id')}")

        # Backfill scout grades on legs created during this generation
        try:
            from placement import backfill_scout_grades_for_today
            _bf = backfill_scout_grades_for_today()
            if _bf:
                result["scout_backfill_updated"] = _bf
                print(f"[mock-gen] scout backfill: updated {_bf} legs")
                # Rescore composites now that scout grades are attached
                from placement import rescore_composites_for_today
                _rc = rescore_composites_for_today()
                result["composites_rescored"] = _rc.get("updated", 0)
        except Exception as _bf_exc:
            print(f"[mock-gen] scout backfill error: {_bf_exc}")
    except Exception as _e:
        import traceback as _tb
        _mock_gen_error  = str(_e)
        _mock_gen_result = None
        print(f"[mock-gen] background ERROR: {_e}\n{_tb.format_exc()}")
    finally:
        _bg_db.close()
        _mock_gen_in_progress = False
        _mock_gen_lock.release()


@app.post("/api/mock-bets/generate")
def mock_bets_generate(
    req: MockBetGenerateRequest,
):
    """
    Start mock-bet generation in a background thread and return immediately.
    Generation takes 2-4 minutes; poll GET /api/mock-bets/status for completion.
    Returns 'started' if kicked off, 'generating' if already in progress.
    """
    global _mock_gen_in_progress, _mock_gen_result, _mock_gen_error
    if not _mock_gen_lock.acquire(blocking=False):
        return {"status": "generating", "message": "Mock bet generation already in progress. Poll /api/mock-bets/status for completion."}

    _mock_gen_in_progress = True
    _mock_gen_result      = None
    _mock_gen_error       = None

    _t = _threading.Thread(
        target  = _run_mock_gen_background,
        args    = (req.stake, req.n_picks, req.max_legs, req.source),
        daemon  = True,
        name    = "mock-gen-bg",
    )
    _t.start()
    return {"status": "started", "message": "Generation started in background. Poll /api/mock-bets/status (~2-4 min)."}


@app.get("/api/mock-bets/top-picks-count")
def mock_bets_top_picks_count(db: Session = Depends(get_db)):
    """Return how many top_picks_page bets exist for today (CT)."""
    from datetime import datetime, timezone
    import pytz
    _ct = pytz.timezone("US/Central")
    today = datetime.now(_ct).strftime("%Y-%m-%d")
    from sqlalchemy import text
    row = db.execute(text(
        "SELECT COUNT(*) FROM mock_bets "
        "WHERE source = 'top_picks_page' AND game_date = :d"
    ), {"d": today}).fetchone()
    return {"count": row[0] if row else 0, "date": today}


@app.get("/api/mock-bets/status")
def mock_bets_gen_status():
    """
    Poll mock-bet generation progress.
    Returns: {status: 'idle'|'generating'|'complete'|'error', result, error}
    """
    if _mock_gen_in_progress:
        return {"status": "generating", "result": None, "error": None}
    if _mock_gen_error:
        return {"status": "error",      "result": None, "error": _mock_gen_error}
    if _mock_gen_result is not None:
        return {"status": "complete",   "result": _mock_gen_result, "error": None}
    return {"status": "idle", "result": None, "error": None}


@app.post("/api/mock-bets/settle")
def mock_bets_settle(db: Session = Depends(get_db)):
    """
    Settle all pending mock bets whose game results are available.
    Bets older than 7 days with no result are marked EXPIRED.
    Returns: {settled, expired, skipped}
    """
    result = mb_module.settle_mock_bets(db)
    # Record CT timestamp so watchdog + health check see a recent settlement
    from zoneinfo import ZoneInfo as _ZoneInfo
    from datetime import datetime as _dt
    sched._scheduler_state["settle_last_ran_ct"] = _dt.now(_ZoneInfo("America/Chicago"))
    return result


@app.post("/api/mock-bets/{bet_id}/manual-settle")
def mock_bet_manual_settle(
    bet_id: str,
    home_score: int = Query(..., description="Final home team score"),
    away_score: int = Query(..., description="Final away team score"),
    db: Session = Depends(get_db),
):
    """
    Manually settle a single PENDING bet by injecting the home/away final score.
    Used for end-of-season playoff games that ESPN/OddsAPI no longer expose.
    """
    return mb_module.manual_settle_bet(db, bet_id, home_score, away_score)


@app.post("/api/attribution/backfill-individual-resolution")
def backfill_individual_resolution(db: Session = Depends(get_db)):
    """
    Backfill leg_historical_resolution from historical.db for all resolvable legs.

    Processes two sources in batches of 100:
      1. Settled mock_bet_legs (WIN/LOSS) from simulation bets
      2. Legacy bet_legs with resolution_source = 'inferred_parlay_win'

    After backfill, run signal_analysis.py to verify corr(component_a, won) improves.
    Returns: {attempted, resolved, not_found, by_sport, by_market, cached_already}
    """
    from leg_resolver import resolve_leg_from_game, parse_leg_details, infer_sport
    from database import LegHistoricalResolution
    from sqlalchemy import text as _sqla_text
    from datetime import datetime as _dt

    _BATCH = 100
    attempted = resolved = not_found = cached_already = 0
    by_sport:  dict = {}
    by_market: dict = {}

    # ── Source 1: settled mock_bet_legs ──────────────────────────────────────
    offset = 0
    while True:
        batch = db.execute(_sqla_text("""
            SELECT bl.id, bl.description, bl.sport, bl.market_type,
                   mb.game_date
            FROM   mock_bet_legs bl
            JOIN   mock_bets mb ON mb.id = bl.mock_bet_id
            WHERE  bl.leg_result IN ('WIN', 'LOSS')
              AND  COALESCE(bl.excluded_from_metrics, 0) = 0
              AND  mb.game_date IS NOT NULL
              AND  mb.source IN (
                       'prospective', 'prospective_pm', 'top_picks_page',
                       'forced_generation', 'retroactive_mock'
                   )
            ORDER BY mb.game_date
            LIMIT :lim OFFSET :off
        """), {"lim": _BATCH, "off": offset}).fetchall()

        if not batch:
            break
        offset += _BATCH

        for row in batch:
            leg_id, desc, sport, mtype, game_date = row
            if not desc or not game_date:
                continue
            attempted += 1

            # Skip if already in cache
            existing = db.execute(_sqla_text(
                "SELECT id FROM leg_historical_resolution WHERE bet_leg_id = :lid LIMIT 1"
            ), {"lid": str(leg_id)}).fetchone()
            if existing:
                cached_already += 1
                continue

            r = resolve_leg_from_game(leg_id, desc, sport or "", game_date, db,
                                      cache=True)
            if r["game_found"] and r["result"] in ("WIN", "LOSS", "PUSH"):
                resolved += 1
                parsed = parse_leg_details(desc)
                hist_sp = infer_sport(sport or "", sport or "") or sport or "?"
                mt      = parsed.get("market_type") or mtype or "Other"
                by_sport[hist_sp]  = by_sport.get(hist_sp, 0) + 1
                by_market[mt]      = by_market.get(mt, 0) + 1
            else:
                not_found += 1

    # ── Source 2: legacy bet_legs with inferred_parlay_win ───────────────────
    # These legs have team/sport/market_type columns directly.
    # Use bets.settled_at as a date hint (game was usually the day before).
    offset = 0
    while True:
        batch = db.execute(_sqla_text("""
            SELECT bl.id, bl.description, bl.team, bl.sport, bl.market_type,
                   bl.game_commence_time, b.time_settled
            FROM   bet_legs bl
            JOIN   bets b ON b.id = bl.bet_id
            WHERE  bl.resolution_source = 'inferred_parlay_win'
              AND  b.time_settled IS NOT NULL
              AND  b.time_settled >= '2023-01-01'
            ORDER BY b.time_settled
            LIMIT :lim OFFSET :off
        """), {"lim": _BATCH, "off": offset}).fetchall()

        if not batch:
            break
        offset += _BATCH

        for row in batch:
            leg_id2, desc2, team, sport2, mtype2, commence_time, time_settled = row
            if not (desc2 or team):
                continue
            # Derive game_date from commence_time or time_settled - 1 day
            if commence_time and len(commence_time) >= 10:
                game_date2 = commence_time[:10]
            elif time_settled and len(str(time_settled)) >= 10:
                try:
                    from datetime import date as _d, timedelta as _td
                    game_date2 = (
                        _d.fromisoformat(str(time_settled)[:10]) - _td(days=1)
                    ).isoformat()
                except Exception:
                    continue
            else:
                continue

            attempted += 1

            existing = db.execute(_sqla_text(
                "SELECT id FROM leg_historical_resolution WHERE bet_leg_id = :lid LIMIT 1"
            ), {"lid": str(leg_id2)}).fetchone()
            if existing:
                cached_already += 1
                continue

            # Build a synthetic description if we have team/market but no text
            raw_desc = desc2 or f"{team} {mtype2 or 'Moneyline'}"
            r = resolve_leg_from_game(leg_id2, raw_desc, sport2 or "", game_date2, db,
                                      cache=True)
            if r["game_found"] and r["result"] in ("WIN", "LOSS", "PUSH"):
                resolved += 1
                hist_sp = infer_sport(sport2 or "", sport2 or "") or sport2 or "?"
                mt2     = mtype2 or "Other"
                by_sport[hist_sp]  = by_sport.get(hist_sp, 0) + 1
                by_market[mt2]     = by_market.get(mt2, 0) + 1
            else:
                not_found += 1

    return {
        "attempted":       attempted,
        "resolved":        resolved,
        "not_found":       not_found,
        "cached_already":  cached_already,
        "by_sport":        by_sport,
        "by_market":       by_market,
        "note": (
            "Run python3 backend/analysis/signal_analysis.py after backfill "
            "to check corr(component_a, won) improvement."
        ),
    }


class RetroactiveBetScore(BaseModel):
    home_team:  str
    away_team:  str
    home_score: int
    away_score: int

class ForcedLegSpec(BaseModel):
    fixture_id: str
    market:     str = "spreads"
    pick:       str           # outcome name (e.g. "Pittsburgh Pirates")
    point:      float | None = None

class RetroactiveBetRequest(BaseModel):
    fixture_ids:         list[str]
    known_scores:        dict[str, RetroactiveBetScore]  # fixture_id → scores
    stake:               float = 10.0
    weight:              float = 0.25
    source:              str   = "retroactive_mock"
    retroactive_reason:  str   = ""
    note:                str   = ""
    game_date:           str   = None
    forced_legs_spec:    list[ForcedLegSpec] | None = None

@app.post("/api/mock-bets/retroactive")
def mock_bets_retroactive(req: RetroactiveBetRequest, db: Session = Depends(get_db)):
    """
    Generate mock bets for already-completed games and settle immediately.

    Bypasses the future-game-only filter — use for games that were missed
    during morning generation (fixture staleness, model bugs, etc.).
    Each bet is tagged with source, weight, retroactive_reason, and note.

    Returns: {generated, settled, wins, losses, bets (with per-leg outcomes)}
    """
    known_scores = {
        fid: sc.dict() for fid, sc in req.known_scores.items()
    }
    forced = [s.dict() for s in req.forced_legs_spec] if req.forced_legs_spec else None
    return mb_module.generate_retroactive_bets(
        db,
        fixture_ids         = req.fixture_ids,
        known_scores        = known_scores,
        stake               = req.stake,
        weight              = req.weight,
        source              = req.source,
        retroactive_reason  = req.retroactive_reason,
        note                = req.note,
        game_date           = req.game_date,
        forced_legs_spec    = forced,
    )


@app.get("/api/mock-bets/performance")
def mock_bets_performance(
    days: int = Query(30, ge=1, le=365),
    bet_category: Optional[str] = Query(None, description="'team' | 'player_prop' | omit for all"),
    game_date: Optional[str] = Query(None, description="'YYYY-MM-DD' or 'today' — filter by game_date"),
    source: str = Query("simulation", description="'simulation' (default), specific source, or 'all'"),
    db: Session = Depends(get_db),
):
    """
    Aggregate win rate, P&L, and trust metrics for settled mock bets.

    Returns trust_level (VALIDATED / BUILDING / UNDERPERFORMING / INSUFFICIENT_DATA),
    daily trend, breakdowns by confidence / sport / model.
    bet_category=team shows only team bets; bet_category=player_prop shows only prop bets.
    game_date=today (or YYYY-MM-DD) filters to a single day instead of rolling window.
    source=simulation (default) filters to prospective/top_picks_page/etc. only.
    """
    return mb_module.get_mock_performance(db, days=days, bet_category=bet_category,
                                          game_date=game_date, source=source)


@app.get("/api/mock-bets/leg-accuracy")
def mock_bets_leg_accuracy(
    days: int = Query(90, ge=1, le=365),
    source: str = Query("simulation", description="'simulation', 'exploration', or 'prospective'"),
    game_date: str = Query(None, description="Exact date filter: 'today', 'yesterday', or YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    """Per-leg win-rate breakdown by sport × market_type for the selected date window."""
    return mb_module.get_leg_accuracy(db, days=days, source=source, game_date=game_date)


@app.get("/api/mock-bets/line-quality")
def mock_bets_line_quality(
    since: str = Query(None, description="ISO date floor, e.g. 2026-04-29 (default: SYSTEM_LAUNCH_DATE)"),
    db: Session = Depends(get_db),
):
    """
    Two-dimensional line quality metrics for settled mock bet legs.

    since: floor date for generated_at (default: SYSTEM_LAUNCH_DATE = 2026-04-29,
           the day CUSHION/AVOID margin grades + personal_edge_profile went live).
           Only simulation sources (prospective, prospective_pm, etc.) are included.

    Returns:
        since             — the floor date used
        dir_accuracy_pct  — % of legs where direction matched main market result
        avg_line_delta    — avg (our_line - optimal_line); +ve = too aggressive
        ab_recovery_pct   — of losses, % that would have won one step closer to main
        real_loss_pct     — % of legs wrong direction entirely (no line fix helps)
        breakdown         — per sport/market table
        interpretation    — plain-English summary
        calibration_ready — True once ≥20 legs have line_delta data
    """
    return mb_module.get_line_quality_summary(db, since=since)


@app.get("/api/mock-bets/clv-summary")
def mock_bets_clv_summary(
    days: int = Query(30, ge=1, le=365, description="Look-back window in days (default: 30)"),
    db: Session = Depends(get_db),
):
    """
    True Closing Line Value (CLV) summary for settled mock bet legs.

    CLV sign convention:
        clv_cents = close_odds - open_odds  (American format)
        clv_cents < 0  →  positive CLV — we got BETTER odds than closing line
        clv_cents > 0  →  negative CLV — line moved against us before game start

    Returns:
        beat_close_pct   — % of legs that beat the closing line (target > 50%)
        avg_clv_cents    — average clv_cents; negative = positive edge
        win_rate_positive_clv — WR on legs that beat the close
        win_rate_negative_clv — WR on legs where line moved against us
        breakdown        — per-market breakdown
        interpretation   — plain-English summary
    """
    return mb_module.get_clv_summary(db, days=days)


@app.get("/api/mock-bets/training-signal")
def mock_bets_training_signal(
    min_settled: int = Query(50, ge=10),
    db: Session = Depends(get_db),
):
    """
    Check if enough mock bets are settled to generate a training signal.
    Returns: {rows, ready, message}
    """
    result = mb_module.generate_mock_training_signal(db, min_settled=min_settled)
    # Don't return the full data array in the API — just the metadata
    result.pop("data", None)
    return result


@app.get("/api/mock-bets/top-picks-calibration")
def top_picks_calibration(
    days: int = Query(30, ge=7, le=365),
    db: Session = Depends(get_db),
):
    """
    Calibration and validation signal for top_picks_page simulation bets.

    Answers:
      1. Parlay-level calibration — does predicted_win_prob predict outcomes?
      2. Leg-level accuracy — does per-leg win_prob predict leg wins?
      3. Min-leg-confidence vs parlay outcome correlation.
      4. Kelly ROI by stake tier.

    Returns bucketed calibration curves, Brier scores, and plain-English
    interpretation. Data populates as top_picks_page bets settle over time.
    """
    return mb_module.get_top_picks_calibration(db, days=days)


@app.get("/api/recommend/live-picks")
def live_picks(
    game_date: str = Query(None, description="YYYY-MM-DD, defaults to today CT"),
    db: Session = Depends(get_db),
):
    """
    Live bet recommendation endpoint.

    Applies strict multi-signal filter to today's production picks and returns
    the slate confidence, qualifying picks (max 5), and red flags.
    """
    return mb_module.apply_live_recommendation(db, game_date=game_date)


@app.get("/api/recommend/live-picks/backtest")
def live_picks_backtest(
    days: int = Query(10, ge=3, le=60),
    db: Session = Depends(get_db),
):
    """
    Backtest the live filter over the last N days.

    Shows per-day comparison: live-qualifying WR/P&L vs full-slate WR/P&L.
    """
    return mb_module.backtest_live_filter(db, days=days)


@app.get("/api/mock-bets/pi-concentration")
def pi_concentration_analysis(db: Session = Depends(get_db)):
    """
    Analysis of Pi-gated concentration flags.

    Shows how many legs would have been filtered by Pi thresholds
    and what their actual outcomes were.  Used to decide whether
    to wire Pi filtering into live generation.
    """
    from sqlalchemy import text as _t
    rows = db.execute(_t("""
        SELECT would_have_filtered,
               COUNT(*) as legs,
               SUM(CASE WHEN leg_result='WIN' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN leg_result='LOSS' THEN 1 ELSE 0 END) as losses,
               ROUND(100.0 * SUM(CASE WHEN leg_result='WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as wr,
               ROUND(AVG(pi_diff), 2) as avg_pi,
               ROUND(AVG(n_bets_containing), 1) as avg_bets
        FROM pi_concentration_flags
        GROUP BY would_have_filtered
        ORDER BY would_have_filtered
    """)).fetchall()

    total = db.execute(_t("SELECT COUNT(*) FROM pi_concentration_flags")).scalar() or 0

    return {
        "total_tracked": total,
        "buckets": [
            {
                "filter": r[0],
                "legs": r[1],
                "wins": r[2],
                "losses": r[3],
                "win_rate": r[4],
                "avg_pi_diff": r[5],
                "avg_bets_containing": r[6],
            }
            for r in rows
        ],
        "recommendation": (
            "Accumulate 50+ flagged legs before deciding to wire live."
            if sum(r[1] for r in rows if r[0] != "none") < 50
            else "Sufficient data — review win rates to decide."
        ),
    }


class BackfillRequest(BaseModel):
    lookback_days: int = 180
    n_per_day:     int = 30


@app.post("/api/mock-bets/backfill")
def mock_bets_backfill(req: BackfillRequest):
    """
    Start a retroactive mock bet backfill as a background job.

    Generates and immediately settles mock parlays for every date in
    [today - lookback_days, yesterday] using games from historical.db
    and the current sub-model scores.

    Returns: {job_id, status, message}

    Poll GET /api/mock-bets/backfill/{job_id} for progress.
    Uses lookahead bias (current model features include post-game data).
    Retroactive bets are stored with weight=0.25 and source='retroactive_mock'.
    """
    # Guard: only one backfill at a time
    active = [j for j in mb_module.list_backfill_jobs()
              if j.get("status") in ("queued", "running")]
    if active:
        return {
            "job_id":  active[0]["job_id"],
            "status":  "already_running",
            "message": f"Backfill already in progress ({active[0]['job_id']}). Poll that job.",
        }

    job_id = mb_module.start_backfill_job(
        lookback_days = req.lookback_days,
        n_per_day     = req.n_per_day,
    )
    return {
        "job_id":  job_id,
        "status":  "queued",
        "message": (
            f"Retroactive backfill started: {req.lookback_days} days × "
            f"up to {req.n_per_day} parlays/day. "
            "Poll GET /api/mock-bets/backfill/{job_id} for progress."
        ),
    }


@app.get("/api/mock-bets/backfill/{job_id}")
def mock_bets_backfill_status(job_id: str):
    """Poll a running or completed backfill job by ID."""
    status = mb_module.get_backfill_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    pct = round(status.get("processed", 0) / max(status.get("total", 1), 1) * 100, 1)
    return {"job_id": job_id, "progress_pct": pct, **status}


@app.get("/api/mock-bets/backfill")
def mock_bets_backfill_list():
    """List all backfill jobs (queued, running, completed, error)."""
    jobs = mb_module.list_backfill_jobs()
    for j in jobs:
        j["progress_pct"] = round(
            j.get("processed", 0) / max(j.get("total", 1), 1) * 100, 1
        )
    return {"jobs": jobs}


@app.get("/api/mock-bets")
def list_mock_bets(
    limit:  int  = Query(50, le=500),
    offset: int  = 0,
    status: Optional[str] = None,
    sort:   str  = Query("desc"),
    bet_category: Optional[str] = Query(None, description="'team' | 'player_prop' | omit for all"),
    bet_tier: Optional[str] = Query(None, description="'T1' | 'T2' | 'T3' | omit for all"),
    source: Optional[str] = Query(None, description="Source filter: 'model' (prospective+pm+picks), 'exploration', 'simulation', 'retro', or exact source name. Omit for all."),
    db: Session = Depends(get_db),
):
    """List mock bets with optional status/tier/source filter and sort order (asc|desc, default desc)."""
    from database import MockBet
    from database import MockBetLeg as _MBL
    from sqlalchemy import text as _sqlt
    q = db.query(MockBet)
    if status:
        q = q.filter(MockBet.status == status)
    if bet_tier:
        q = q.filter(MockBet.bet_tier == bet_tier)
    # Source filter — group aliases map to multiple source values
    _SOURCE_GROUPS = {
        "model":       ["prospective", "prospective_pm", "top_picks_page", "prospective_low_lqs",
                        "prospective_legacy", "forced_generation"],
        "exploration": ["exploration"],
        "simulation":  ["simulation"],
        "retro":       ["retroactive_mock"],
        "shadow":      ["shadow_negbin", "shadow_negbin_fh", "shadow_negbin_excluded",
                        "shadow_negbin_recal", "shadow_negbin_recal_fh"],
        "ats_ungated": ["ats_ungated"],
    }
    if source:
        src_list = _SOURCE_GROUPS.get(source)
        if src_list:
            q = q.filter(MockBet.source.in_(src_list))
        else:
            q = q.filter(MockBet.source == source)
    if bet_category:
        # Fetch matching IDs via raw SQL, then apply a WHERE EXISTS subquery
        # instead of a giant IN(...) to avoid SQLite page errors.
        matched_ids = [
            row[0] for row in db.execute(
                _sqlt("""
                    SELECT mock_bet_id FROM mock_bet_legs
                    GROUP BY mock_bet_id
                    HAVING SUM(CASE WHEN COALESCE(bet_category,'team')=:cat THEN 1 ELSE 0 END)=COUNT(*)
                """),
                {"cat": bet_category},
            )
        ]
        if not matched_ids:
            q = q.filter(MockBet.id == None)  # empty result
        else:
            # Batch into chunks of 500 to keep IN clauses SQLite-safe
            from sqlalchemy import or_
            chunks = [matched_ids[i:i+500] for i in range(0, len(matched_ids), 500)]
            q = q.filter(or_(*[MockBet.id.in_(chunk) for chunk in chunks]))
    if sort == "asc":
        q = q.order_by(MockBet.generated_at.asc())
    else:
        q = q.order_by(MockBet.generated_at.desc())
    total = q.count()
    rows  = q.offset(offset).limit(limit).all()

    # Fetch leg-level ALE fields in one query for all returned bets
    bet_ids = [m.id for m in rows]
    leg_rows = db.query(_MBL).filter(_MBL.mock_bet_id.in_(bet_ids)).all() if bet_ids else []
    legs_by_bet: dict = {}
    for leg in leg_rows:
        legs_by_bet.setdefault(leg.mock_bet_id, []).append({
            "leg_index":      leg.leg_index,
            "description":    leg.description,
            "market_type":    leg.market_type,
            "sport":          leg.sport,
            "win_prob":       leg.win_prob,
            "open_odds":      leg.open_odds,
            "ale_considered": bool(leg.ale_considered),
            "ale_switched":   bool(leg.ale_switched),
            "ale_naive_pick": leg.ale_naive_pick,
        })

    return {
        "total": total,
        "mock_bets": [
            {
                "id":                 m.id,
                "generated_at":       m.generated_at.isoformat() if m.generated_at else None,
                "game_date":          m.game_date,
                "sport":              m.sport,
                "bet_type":           m.bet_type,
                "odds":               m.odds,
                "amount":             m.amount,
                "legs":               m.legs,
                "bet_info":           m.bet_info,
                "status":             m.status,
                "predicted_win_prob": m.predicted_win_prob,
                "predicted_ev":       m.predicted_ev,
                "confidence":         m.confidence,
                "model_used":         m.model_used,
                "model_auc":          m.model_auc,
                "settled_at":         m.settled_at.isoformat() if m.settled_at else None,
                "actual_profit":      m.actual_profit,
                "avg_lqs":            m.avg_lqs,
                "source":             m.source,
                "weight":             m.weight,
                "notes":              m.notes,
                "promo_type":         m.promo_type,
                "promo_boost_pct":    m.promo_boost_pct,
                "boost_strategy":     m.boost_strategy,
                "promo_ev_lift":      m.promo_ev_lift,
                "promo_boosted_odds": m.promo_boosted_odds,
                "bet_tier":           m.bet_tier,
                "composite_score":    m.composite_score,
                "leg_details":        sorted(legs_by_bet.get(m.id, []), key=lambda x: x["leg_index"] or 0),
            }
            for m in rows
        ],
    }

# ─── Creator Tier Endpoints ───────────────────────────────────────────────────

import creator_tier as ct
import weather as wx_module

class ClvBackfillRequest(BaseModel):
    sport:     str = "NHL"    # NHL | MLB
    days_back: int = 180
    dry_run:   bool = False

@app.post("/api/odds/clv-backfill")
def clv_backfill(req: ClvBackfillRequest):
    """
    Fill missing close_spread / close_ml_home / close_ml_away in historical.db
    for NHL/MLB games using TheOddsAPI historical odds (costs ~20 credits/date).
    Run once; safe to re-run (only fills NULL rows).
    """
    result = ct.backfill_clv(
        sport     = req.sport,
        days_back = req.days_back,
        dry_run   = req.dry_run,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/credits/status")
def credits_status():
    """
    Current API credit budget: remaining, burn rate, projected month-end total.
    Read-only — no API call needed, reads from credits_log.
    """
    return ct.get_credit_status()


@app.post("/api/odds/imminent-snapshot")
def imminent_snapshot():
    """
    Fetch line snapshots for games starting within 6 hours (targeted, credit-efficient).
    The scheduler runs this automatically every 45 min, 8 AM–11 PM local time.
    """
    return ct.fetch_imminent_games_odds()


@app.post("/api/odds/line-snapshot")
def line_snapshot(sport_key: Optional[str] = None):
    """
    Full line snapshot for all (or a specific) sport.
    Use sparingly — higher credit cost than imminent snapshot.
    Subject to budget guard.
    """
    return ct.capture_line_snapshot(sport_key)


@app.get("/api/odds/line-movement/{event_id}")
def line_movement(
    event_id:   str,
    market_key: str = Query("spreads"),
):
    """
    All line snapshots for an event + movement analysis.
    Returns opening line, current line, total move, and steam/reverse alert.
    """
    return ct.detect_line_movement(event_id, market_key)


@app.get("/api/odds/best-lines/{sport_key}")
def best_lines(
    sport_key: str,
    market:    str = Query("h2h"),
):
    """
    Cross-bookmaker line shopping for a sport.
    Flags outcomes where FanDuel is ≥5% worse than market best.
    """
    return {"sport_key": sport_key, "market": market,
            "lines": ct.get_best_lines(sport_key, market)}


@app.get("/api/odds/line-shopping-summary")
def line_shopping_summary():
    """
    Across all active sports: how many FanDuel lines are worse than market best?
    Returns {sport_key: {total_outcomes, alerts, worst_gap_pct, worst_game}}.
    """
    return ct.get_line_shopping_summary()


@app.post("/api/props/fetch")
def props_fetch(sport_key: str = "baseball_mlb"):
    """
    Fetch player prop odds for a sport right now and store to historical.db.
    The scheduler runs this automatically at 9 AM UTC daily.
    """
    result = ct.fetch_player_props(sport_key)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/props/{event_id}")
def props_for_event(
    event_id:   str,
    market_key: Optional[str] = None,
):
    """All stored player prop lines for an event (optionally filtered by market)."""
    return {"event_id": event_id, "props": ct.get_props_for_event(event_id, market_key)}


@app.get("/api/props/{event_id}/best")
def best_props(event_id: str, market_key: str = Query("pitcher_strikeouts")):
    """
    Best available price per player/over-under for a specific prop market,
    with FanDuel comparison and gap %.
    """
    return {"event_id": event_id, "market_key": market_key,
            "props": ct.get_best_props(event_id, market_key)}


@app.get("/api/weather")
def game_weather(
    home_team:    str,
    commence_time: str,
    sport:        str = "mlb",
):
    """
    Weather forecast for an outdoor game (MLB / NFL).
    Returns temperature, wind, precip chance and an alert flag.
    Domed/indoor stadiums return is_dome=True with no weather data.
    """
    return wx_module.get_game_weather(home_team, commence_time, sport)


@app.get("/api/weather/fixtures/{sport_key}")
def weather_for_fixtures(sport_key: str, db: Session = Depends(get_db)):
    """
    Enrich all upcoming fixtures for a sport key with weather forecasts.
    Only meaningful for outdoor sports (baseball_mlb, americanfootball_nfl, etc).
    """
    from odds_api import fetch_odds
    sport_label = "nfl" if "football" in sport_key else "mlb"
    events = fetch_odds(sport_key) or []
    fixtures = [
        {
            "home_team":     e.get("home_team", ""),
            "away_team":     e.get("away_team", ""),
            "commence_time": e.get("commence_time", ""),
            "event_id":      e.get("id", ""),
        }
        for e in events
    ]
    return {"sport_key": sport_key, "fixtures": wx_module.enrich_fixtures_with_weather(fixtures, sport_label)}


@app.get("/api/bets/slip-status")
def slip_status():
    """Check if Anthropic API key is configured for slip scanning."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "ready": bool(key),
        "message": "Slip scanner ready." if key else "Set ANTHROPIC_API_KEY to enable slip scanning.",
        "setup_instructions": "In terminal: export ANTHROPIC_API_KEY=your-key  then  bash START.sh"
    }

@app.post("/api/bets/parse-slip")
async def parse_slip(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a bet slip screenshot. Claude vision extracts all legs,
    returns structured JSON with EV analysis + LQS quality enrichment.
    """
    image_bytes = await file.read()
    mime_type   = file.content_type or "image/png"
    result      = sp.parse_slip_image(image_bytes, mime_type)
    # Enrich with LQS + pivot suggestions (best-effort)
    if result and not result.get("error") and result.get("legs"):
        try:
            lqs_result = _enrich_legs_with_lqs(result["legs"], db)
            result["legs"]        = lqs_result["legs"]
            result["lqs_summary"] = lqs_result["lqs_summary"]
        except Exception:
            pass
    return result

@app.post("/api/bets/confirm-slip")
def confirm_slip(req: ConfirmSlipRequest, db: Session = Depends(get_db)):
    """
    After reviewing the parsed slip, confirm to save it as a placed bet.
    Flows into model training exactly like any other bet.
    """
    return sp.save_parsed_slip(req.parsed, req.stake, req.is_mock, db)


class EditBetRequest(BaseModel):
    amount: Optional[float] = None
    status: Optional[str]   = None

@app.patch("/api/bets/{bet_id}/edit")
def edit_bet(bet_id: str, req: EditBetRequest, db: Session = Depends(get_db)):
    """Edit stake and/or status of any bet."""
    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if not bet:
        raise HTTPException(404, "Bet not found")
    if req.amount is not None:
        bet.amount = req.amount
    if req.status is not None:
        bet.status = req.status
    # Recalculate profit whenever amount or status changes on a settled bet
    final_status = bet.status or ""
    if final_status == "SETTLED_WIN":
        bet.profit = round((bet.odds - 1) * bet.amount, 2)
        if not bet.time_settled:
            bet.time_settled = datetime.utcnow()
    elif final_status == "SETTLED_LOSS":
        bet.profit = round(-bet.amount, 2)
        if not bet.time_settled:
            bet.time_settled = datetime.utcnow()
    elif final_status == "SETTLED_PUSH":
        bet.profit = 0
        if not bet.time_settled:
            bet.time_settled = datetime.utcnow()
    db.commit()
    return {"bet_id": bet_id, "amount": bet.amount, "status": bet.status, "profit": bet.profit}

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "scheduler": sched.get_state()["running"]}


# ── Alt Lines ─────────────────────────────────────────────────────────────────

class AltLineFetchRequest(BaseModel):
    sport_key: str
    event_id:  str
    bookmakers: list[str] = []

class AltLineScoreRequest(BaseModel):
    sport_key:  str
    event_id:   str
    model_prob: float = 0.52
    main_line:  Optional[float] = None

@app.post("/api/alt-lines/fetch")
def fetch_alt_lines_endpoint(req: AltLineFetchRequest):
    """Fetch and store alternate lines for one event from TheOddsAPI."""
    return ct.fetch_alt_lines(
        sport_key  = req.sport_key,
        event_id   = req.event_id,
        bookmakers = req.bookmakers or None,
    )


@app.post("/api/alt-lines/fetch-today")
def fetch_todays_alt_lines(db: Session = Depends(get_db)):
    """
    Batch-fetch alt lines for all upcoming fixtures with configured alt markets.

    Eligible sport keys: baseball_mlb, basketball_nba, icehockey_nhl, americanfootball_nfl.
    Skips fixtures that already have an alt-line snapshot fetched today.
    Credits are checked per-fixture by fetch_alt_lines().

    Returns: {fetched, skipped_existing, skipped_no_market, errors, total_rows, fixtures}
    """
    from datetime import timezone as _tz
    now        = datetime.now(_tz.utc)
    today_str  = now.strftime("%Y-%m-%dT00:00:00Z")

    # Fixtures with alt market configs
    eligible = db.query(Fixture).filter(
        Fixture.commence_time > now
    ).all()

    fetched            = 0
    skipped_existing   = 0
    skipped_no_market  = 0
    errors             = 0
    total_rows         = 0
    results            = []

    for fix in eligible:
        if fix.sport_key not in ct._ALT_MARKETS:
            skipped_no_market += 1
            continue

        # Skip if already fetched today (check historical.db via creator_tier's conn)
        try:
            _conn = ct._hist_conn()
            existing_today = _conn.execute(
                "SELECT COUNT(*) FROM alt_lines WHERE event_id=? AND fetched_at >= ?",
                (fix.id, today_str)
            ).fetchone()[0]
            _conn.close()
            if existing_today > 0:
                skipped_existing += 1
                continue
        except Exception:
            pass  # if we can't check, just re-fetch

        result = ct.fetch_alt_lines(
            sport_key = fix.sport_key,
            event_id  = fix.id,
        )
        if result.get("error") or result.get("skipped_reason"):
            errors += 1
        else:
            fetched   += 1
            rows       = result.get("rows_inserted", 0)
            total_rows += rows
        results.append({
            "event_id":  fix.id,
            "sport_key": fix.sport_key,
            "game":      f"{fix.away_team} @ {fix.home_team}",
            **result,
        })

    return {
        "fetched":           fetched,
        "skipped_existing":  skipped_existing,
        "skipped_no_market": skipped_no_market,
        "errors":            errors,
        "total_rows":        total_rows,
        "fixtures":          results,
    }

@app.get("/api/alt-lines/{event_id}")
def get_alt_lines(
    event_id:   str,
    model_prob: float = 0.52,
    main_line:  Optional[float] = None,
    filter:     str = "VALUE",
):
    """Return scored alt lines for an event from the most recent stored fetch."""
    lines = ct.get_scored_alt_lines(
        event_id   = event_id,
        model_prob = model_prob,
        main_line  = main_line,
        min_label  = filter.upper(),
    )
    return {"event_id": event_id, "lines": lines, "count": len(lines)}

@app.get("/api/alt-lines")
def get_todays_alt_lines(
    sport_key: str = "",   # e.g. baseball_mlb
    sport:     str = "",   # alias for sport_key — either param works
    market:    str = "",   # e.g. alternate_spreads, alternate_totals
    filter:    str = "ALL",
):
    """
    All stored alt lines for today sorted by EV descending.
    No odds-range filtering applied — raw rows from the alt_lines table.

    Query params:
      sport_key / sport — filter by sport (either param works)
      market            — filter by market_key (e.g. alternate_spreads)
      filter            — ALL | VALUE | ANCHOR  (value_label gate)
    """
    _sk = sport_key or sport   # accept either spelling
    lines = ct.get_todays_alt_lines(
        sport_key    = _sk,
        market_key   = market,
        filter_label = filter.upper(),
    )
    return {"lines": lines, "count": len(lines)}


# ─── Personal Edge Profile ────────────────────────────────────────────────────

import personal_edge_profile as pep


@app.get("/api/personal-edge-profile")
def get_personal_edge_profiles():
    """
    Return all rows from personal_edge_profile — Jose's unbiased win rates
    by sport × market_type × line_bucket, with margin grades and parlay depth limits.

    Fields per row:
      sport, market_type, line_bucket, sample_size,
      unbiased_wr (personal_wr), mean_delta, std_delta, edge_ratio,
      close_call_rate, narrow_loss_rate, margin_grade, max_parlay_legs,
      avg_odds, avg_ev, data_sources, last_updated
    """
    import sqlite3 as _sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "bets.db")
    try:
        conn = _sqlite3.connect(db_path)
        conn.row_factory = _sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM personal_edge_profile ORDER BY margin_grade, sport, market_type"
        ).fetchall()
        conn.close()
        result = [dict(r) for r in rows]
        for r in result:
            r['unbiased_wr'] = r.get('personal_wr')
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"personal_edge_profile error: {e}")


@app.post("/api/personal-edge-profile/refresh")
def refresh_personal_edge_profiles():
    """
    Rebuild personal_edge_profile from resolved bet_legs.
    Equivalent to what runs automatically on server startup and weekly Sunday scheduler.
    """
    try:
        result = pep.refresh_personal_edge_profiles()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh error: {e}")


# ─── A/B Test — User Picks (Phase 8) ─────────────────────────────────────────
# Compare the user's hand-picked parlays against the model's auto-generated
# mock bets.  Shows BOTH absolute P&L and rate metrics (WR%, ROI%) so the user
# can verify their claim of beating the model on total P&L at lower volume.
# ─────────────────────────────────────────────────────────────────────────────

class UserPickLegRequest(BaseModel):
    # Primary fields
    description:   Optional[str]  = None   # "Yankees ML" — or derived from bet_info
    sport:         Optional[str]  = None
    market_type:   Optional[str]  = None
    team:          Optional[str]  = None
    odds_american: Optional[int]  = None
    # Alternate field names (FanDuel / OddsAPI style)
    bet_info:      Optional[str]  = None   # alias for description
    price:         Optional[int]  = None   # alias for odds_american
    home_team:     Optional[str]  = None   # used to derive team when team is absent
    away_team:     Optional[str]  = None
    point:         Optional[float]= None   # spread/total line value

    def resolved_description(self) -> str:
        return self.description or self.bet_info or (
            f"{self.home_team} vs {self.away_team}" if self.home_team else "Unknown"
        )

    def resolved_odds_american(self) -> Optional[int]:
        return self.odds_american or self.price

    def resolved_team(self) -> Optional[str]:
        """Primary team: explicit team field, else home_team."""
        return self.team or self.home_team

class UserPickRequest(BaseModel):
    game_date:       Optional[str]   = None   # YYYY-MM-DD; defaults to today CT
    bet_type:        str              = "parlay"
    stake:           float            = 10.0
    notes:           Optional[str]   = None
    reasoning:       Optional[str]   = None   # alias for notes
    legs:            List[UserPickLegRequest] = []
    # Promo metadata (stored in notes, not a separate DB column yet)
    promo_type:      Optional[str]   = None   # PROFIT_BOOST | BONUS_BET | etc.
    promo_boost_pct: Optional[float] = None   # 0.25 | 0.30 | 0.50


def _american_to_decimal(american: int) -> float:
    """Convert American odds integer to decimal."""
    try:
        n = float(american)
        if n > 0:  return round(1 + n / 100, 4)
        if n < 0:  return round(1 + 100 / abs(n), 4)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return 1.0


def _combined_decimal_odds(legs: list) -> float:
    """Multiply decimal odds across all legs."""
    combined = 1.0
    for leg in legs:
        dec = leg.odds_decimal or (
            _american_to_decimal(leg.odds_american) if leg.odds_american else 1.0
        )
        combined *= dec
    return round(combined, 4)


@app.post("/api/user-picks")
def submit_user_pick(req: UserPickRequest, db: Session = Depends(get_db)):
    """Submit a user pick for A/B comparison. Returns the created pick with its ID."""
    if not req.legs:
        raise HTTPException(status_code=400, detail="At least one leg is required.")

    # Resolve game_date — default to today in CT
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dt2
    _game_date = req.game_date or _dt2.now(_ZI("America/Chicago")).strftime("%Y-%m-%d")

    # Merge notes + reasoning; append promo metadata
    _notes_parts = []
    if req.notes:      _notes_parts.append(req.notes)
    if req.reasoning and req.reasoning != req.notes:
        _notes_parts.append(req.reasoning)
    if req.promo_type:
        boost_str = f"+{int(req.promo_boost_pct*100)}%" if req.promo_boost_pct else ""
        _notes_parts.append(f"[{req.promo_type} {boost_str}]".strip())
    _notes = " | ".join(_notes_parts) if _notes_parts else None

    pick_id = str(uuid.uuid4())

    # Build leg ORM objects using resolved field helpers
    leg_objs = []
    for i, leg_req in enumerate(req.legs):
        amer  = leg_req.resolved_odds_american()
        dec   = _american_to_decimal(amer) if amer is not None else None
        desc  = leg_req.resolved_description()
        # Append point spread/total to description if present and not already there
        if leg_req.point is not None and str(leg_req.point) not in desc:
            desc = f"{desc} ({'+' if leg_req.point > 0 else ''}{leg_req.point})"
        mkt_lo = (leg_req.market_type or '').lower()
        _exp_val = 0.5 if mkt_lo in ('h2h', 'moneyline', 'ml') else leg_req.point
        leg_objs.append(UserPickLeg(
            user_pick_id           = pick_id,
            leg_index              = i,
            description            = desc,
            sport                  = leg_req.sport,
            market_type            = leg_req.market_type,
            team                   = leg_req.resolved_team(),
            odds_american          = amer,
            odds_decimal           = dec,
            point                  = leg_req.point,
            expected_outcome_value = _exp_val,
        ))

    # Compute combined decimal odds
    legs_with_odds = [l for l in leg_objs if l.odds_decimal]
    combined = None
    if legs_with_odds:
        combined = 1.0
        for l in legs_with_odds:
            combined *= l.odds_decimal
        combined = round(combined, 4)

    # Compute boosted odds / potential profit if a PROFIT_BOOST is applied
    _promo_type     = req.promo_type     or None
    _promo_boost    = req.promo_boost_pct or None
    _boosted_odds   = None
    _potential_prof = None
    if combined and _promo_type == "PROFIT_BOOST" and _promo_boost:
        _boosted_odds   = round(combined * (1 + _promo_boost), 4)
        _potential_prof = round((_boosted_odds - 1) * req.stake, 2)
    elif combined:
        _potential_prof = round((combined - 1) * req.stake, 2)

    # Reasoning stored separately from notes
    _reasoning = req.reasoning if req.reasoning and req.reasoning != req.notes else None

    pick = UserPick(
        id                  = pick_id,
        game_date           = _game_date,
        bet_type            = req.bet_type if len(req.legs) > 1 else "straight",
        stake               = req.stake,
        legs                = len(req.legs),
        combined_odds       = combined,
        notes               = _notes,
        reasoning           = _reasoning,
        promo_type          = _promo_type,
        promo_boost_pct     = _promo_boost,
        promo_boosted_odds  = _boosted_odds,
        potential_profit    = _potential_prof,
        status              = "PENDING",
    )
    db.add(pick)
    for leg_obj in leg_objs:
        db.add(leg_obj)
    db.commit()
    db.refresh(pick)

    # Enrich user pick legs with scout grade from scouted_props (best-effort)
    try:
        import placement as _plc
        from sqlalchemy import text as _stext
        _fresh = db.query(UserPickLeg).filter(UserPickLeg.user_pick_id == pick.id).all()
        for _uleg in _fresh:
            _leg_dict = {
                "fixture_id":  _uleg.sport,       # UserPickLeg has no fixture_id; use sport fallback
                "game_id":     None,
                "market_type": _uleg.market_type,
                "description": _uleg.description,
                "player_name": _uleg.team,
            }
            _enriched = _plc.attach_scout_grade_to_leg(_leg_dict, db)
            if _enriched.get("scout_grade"):
                db.execute(_stext(
                    "UPDATE user_pick_legs SET scout_grade=:g, scout_hit_prob=:hp, "
                    "scouted_prop_id=:pid WHERE id=:lid"
                ), {
                    "g":   _enriched["scout_grade"],
                    "hp":  _enriched.get("scout_hit_prob"),
                    "pid": _enriched.get("scouted_prop_id"),
                    "lid": _uleg.id,
                })
        db.commit()
    except Exception as _scout_err:
        print(f"[user-pick] scout enrich non-fatal: {_scout_err}")

    # Extract learning signals now that legs have IDs
    try:
        fresh_legs = db.query(UserPickLeg).filter(UserPickLeg.user_pick_id == pick.id).all()
        # v1: keyword extractor (always runs)
        usl.extract_signals(pick, fresh_legs, db)
        db.commit()
    except Exception as _se:
        print(f"[signal-extract v1] non-fatal: {_se}")

    # v2: LLM extractor (runs when reasoning is present and API key is set)
    try:
        fresh_legs = db.query(UserPickLeg).filter(UserPickLeg.user_pick_id == pick.id).all()
        lse.extract_llm_signals(pick, fresh_legs, db)
        db.commit()
    except Exception as _le:
        print(f"[signal-extract v2 LLM] non-fatal: {_le}")

    # v3: comparison-vs-model signals (overlap_with_model / user_unique_pick)
    try:
        fresh_legs = db.query(UserPickLeg).filter(UserPickLeg.user_pick_id == pick.id).all()
        usl.extract_comparison_signals(pick, fresh_legs, db)
        db.commit()
    except Exception as _ce:
        print(f"[signal-extract comparison] non-fatal: {_ce}")

    return {
        "id":            pick.id,
        "game_date":     pick.game_date,
        "bet_type":      pick.bet_type,
        "stake":         pick.stake,
        "legs":          pick.legs,
        "combined_odds": pick.combined_odds,
        "status":        pick.status,
        "submitted_at":  pick.submitted_at.isoformat() if pick.submitted_at else None,
        "notes":         pick.notes,
        "legs_detail":   [
            {
                "leg_index":    l.leg_index,
                "description":  l.description,
                "sport":        l.sport,
                "market_type":  l.market_type,
                "team":         l.team,
                "odds_american":l.odds_american,
                "odds_decimal": l.odds_decimal,
            }
            for l in leg_objs
        ],
    }


class PlaceFromMockRequest(BaseModel):
    mock_bet_id: str
    actual_odds_american: Optional[List[Optional[int]]] = None  # per-leg overrides
    actual_stake: float = 10.0
    book: str = "FanDuel"


@app.post("/api/user-picks/place-from-mock")
def place_from_mock(req: PlaceFromMockRequest, db: Session = Depends(get_db)):
    """Create a user_pick from an existing mock bet — records actual placed odds/stake."""
    from database import MockBet, MockBetLeg as _MBL
    from zoneinfo import ZoneInfo as _ZI
    from sqlalchemy import text as _sqlt

    mock = db.query(MockBet).filter(MockBet.id == req.mock_bet_id).first()
    if not mock:
        raise HTTPException(status_code=404, detail="Mock bet not found")

    mock_legs = (
        db.query(_MBL)
        .filter(_MBL.mock_bet_id == req.mock_bet_id)
        .order_by(_MBL.leg_index)
        .all()
    )
    if not mock_legs:
        raise HTTPException(status_code=400, detail="Mock bet has no legs")

    _CT_TZ = _ZI("America/Chicago")
    _now = datetime.now(_CT_TZ)
    _game_date = mock.game_date or _now.strftime("%Y-%m-%d")
    pick_id = str(uuid.uuid4())

    # Build legs — use actual_odds overrides if provided, else fall back to model odds
    overrides = req.actual_odds_american or []
    leg_objs = []
    for i, ml in enumerate(mock_legs):
        amer = overrides[i] if i < len(overrides) and overrides[i] is not None else ml.open_odds
        dec = _american_to_decimal(amer) if amer else None
        leg_objs.append(UserPickLeg(
            user_pick_id  = pick_id,
            leg_index     = i,
            description   = ml.description,
            sport         = ml.sport,
            market_type   = ml.market_type,
            team          = ml.resolved_home_team,
            odds_american = amer,
            odds_decimal  = dec,
            point         = None,
        ))

    combined = 1.0
    for lo in leg_objs:
        if lo.odds_decimal:
            combined *= lo.odds_decimal
    combined = round(combined, 4)
    potential = round(req.actual_stake * (combined - 1), 2) if combined > 1 else 0

    pick = UserPick(
        id              = pick_id,
        submitted_at    = _now,
        game_date       = _game_date,
        bet_type        = mock.bet_type or "parlay",
        stake           = req.actual_stake,
        legs            = len(leg_objs),
        combined_odds   = combined,
        notes           = f"Placed from model T{mock.bet_tier or '?'} pick | book={req.book} | mock={req.mock_bet_id[:8]}",
        status          = "PENDING",
        potential_profit = potential,
        ingestion_source = "user_placed_from_model",
        bet_placed_at   = _now,
    )
    db.add(pick)
    for lo in leg_objs:
        db.add(lo)
    db.flush()

    # Store placed_mock_bet_id via raw SQL (column may not be on ORM yet)
    try:
        db.execute(_sqlt(
            "UPDATE user_picks SET placed_mock_bet_id = :mid WHERE id = :pid"
        ), {"mid": req.mock_bet_id, "pid": pick_id})
    except Exception:
        pass  # column may not exist yet on first run before restart

    db.commit()

    return {
        "pick_id": pick_id,
        "mock_bet_id": req.mock_bet_id,
        "stake": req.actual_stake,
        "combined_odds": combined,
        "potential_profit": potential,
        "book": req.book,
        "legs": len(leg_objs),
    }


@app.get("/api/user-picks")
def list_user_picks(
    date: Optional[str] = None,
    db:   Session       = Depends(get_db),
):
    """
    List user picks. Optional ?date=YYYY-MM-DD filter.
    Returns picks newest-first with legs attached.
    """
    q = db.query(UserPick)
    if date:
        q = q.filter(UserPick.game_date == date)
    picks = q.order_by(UserPick.submitted_at.desc()).all()

    result = []
    for pick in picks:
        legs = db.query(UserPickLeg).filter(
            UserPickLeg.user_pick_id == pick.id
        ).order_by(UserPickLeg.leg_index).all()
        result.append({
            "id":            pick.id,
            "game_date":     pick.game_date,
            "bet_type":      pick.bet_type,
            "stake":         pick.stake,
            "legs":          pick.legs,
            "combined_odds": pick.combined_odds,
            "status":        pick.status,
            "actual_profit": pick.actual_profit,
            "submitted_at":  pick.submitted_at.isoformat() if pick.submitted_at else None,
            "settled_at":    pick.settled_at.isoformat()   if pick.settled_at   else None,
            "notes":         pick.notes,
            "legs_detail":   [
                {
                    "id":                    l.id,
                    "leg_index":             l.leg_index,
                    "description":           l.description,
                    "sport":                 l.sport,
                    "market_type":           l.market_type,
                    "team":                  l.team,
                    "player":                l.player,
                    "odds_american":         l.odds_american,
                    "odds_decimal":          l.odds_decimal,
                    "leg_result":            l.leg_result,
                    "point":                 l.point,
                    "actual_outcome_value":  l.actual_outcome_value,
                    "expected_outcome_value":l.expected_outcome_value,
                    "miss_margin":           l.miss_margin,
                    "outcome_source":        l.outcome_source,
                }
                for l in legs
            ],
        })
    return result


@app.get("/api/user-picks/{pick_id}")
def get_user_pick(pick_id: str, db: Session = Depends(get_db)):
    """Get a single user pick by ID with full leg detail."""
    pick = db.query(UserPick).filter(UserPick.id == pick_id).first()
    if not pick:
        raise HTTPException(status_code=404, detail="Pick not found.")
    legs = db.query(UserPickLeg).filter(
        UserPickLeg.user_pick_id == pick_id
    ).order_by(UserPickLeg.leg_index).all()
    return {
        "id":            pick.id,
        "game_date":     pick.game_date,
        "bet_type":      pick.bet_type,
        "stake":         pick.stake,
        "legs":          pick.legs,
        "combined_odds": pick.combined_odds,
        "status":        pick.status,
        "actual_profit": pick.actual_profit,
        "submitted_at":  pick.submitted_at.isoformat() if pick.submitted_at else None,
        "settled_at":    pick.settled_at.isoformat()   if pick.settled_at   else None,
        "notes":         pick.notes,
        "legs_detail":   [
            {
                "id":                    l.id,
                "leg_index":             l.leg_index,
                "description":           l.description,
                "sport":                 l.sport,
                "market_type":           l.market_type,
                "team":                  l.team,
                "player":                l.player,
                "odds_american":         l.odds_american,
                "odds_decimal":          l.odds_decimal,
                "leg_result":            l.leg_result,
                "point":                 l.point,
                "actual_outcome_value":  l.actual_outcome_value,
                "expected_outcome_value":l.expected_outcome_value,
                "miss_margin":           l.miss_margin,
                "outcome_source":        l.outcome_source,
            }
            for l in legs
        ],
    }


@app.delete("/api/user-picks/{pick_id}")
def delete_user_pick(pick_id: str, db: Session = Depends(get_db)):
    """Delete a user pick and all its legs."""
    pick = db.query(UserPick).filter(UserPick.id == pick_id).first()
    if not pick:
        raise HTTPException(status_code=404, detail="Pick not found.")
    db.query(UserPickLeg).filter(UserPickLeg.user_pick_id == pick_id).delete()
    db.delete(pick)
    db.commit()
    return {"deleted": pick_id}


class UserPickEditRequest(BaseModel):
    promo_type:      Optional[str]   = None     # None = "not provided"; set clear_boost=true to explicitly clear
    promo_boost_pct: Optional[float] = None
    reasoning:       Optional[str]   = None
    clear_boost:     bool            = False    # pass true to explicitly clear boost (set promo_type=null)


@app.patch("/api/user-picks/{pick_id}")
def edit_user_pick(pick_id: str, req: UserPickEditRequest, db: Session = Depends(get_db)):
    """
    Edit a user pick post-submission.

    PENDING picks: boost AND reasoning can be changed.
    SETTLED picks: ONLY reasoning can be changed.
      Changing boost on a settled pick corrupts historical P&L — rejected with 409.

    If reasoning changes: signals are re-extracted (keyword + LLM).
    If boost changes: promo_boosted_odds and potential_profit are recomputed.
    """
    pick = db.query(UserPick).filter(UserPick.id == pick_id).first()
    if not pick:
        raise HTTPException(status_code=404, detail="Pick not found.")

    is_settled = pick.status in ("SETTLED_WIN", "SETTLED_LOSS", "VOID")

    # ── Boost edit guard ─────────────────────────────────────────────────────
    boost_in_payload = (req.promo_type is not None) or (req.promo_boost_pct is not None) or req.clear_boost
    if boost_in_payload and is_settled:
        raise HTTPException(
            status_code=409,
            detail="Cannot change boost on a settled pick — would corrupt historical P&L."
        )

    reasoning_changed = False
    boost_changed     = False

    # ── Apply reasoning ──────────────────────────────────────────────────────
    if req.reasoning is not None and req.reasoning != (pick.reasoning or ""):
        pick.reasoning = req.reasoning or None
        pick.llm_features_extracted = 0   # flag for re-extraction
        reasoning_changed = True
        # Also sync into notes (keep legacy field consistent)
        _base_notes = (pick.notes or "").split("[PROFIT_BOOST")[0].split("[BONUS_BET")[0].split("[NO_SWEAT")[0].strip()
        _promo_suffix = _build_promo_note(pick.promo_type, pick.promo_boost_pct)
        parts = [p for p in [pick.reasoning, _base_notes, _promo_suffix] if p]
        pick.notes = " | ".join(parts) if parts else None

    # ── Apply boost ──────────────────────────────────────────────────────────
    if boost_in_payload and not is_settled:
        old_promo = pick.promo_type
        if req.clear_boost:
            pick.promo_type      = None
            pick.promo_boost_pct = None
        else:
            pick.promo_type      = req.promo_type
            pick.promo_boost_pct = req.promo_boost_pct
        boost_changed = (old_promo != pick.promo_type)

        # Recompute boosted odds and potential profit
        if pick.combined_odds and pick.promo_type == "PROFIT_BOOST" and pick.promo_boost_pct:
            pick.promo_boosted_odds = round(pick.combined_odds * (1 + pick.promo_boost_pct), 4)
            pick.potential_profit   = round((pick.promo_boosted_odds - 1) * (pick.stake or 10.0), 2)
        else:
            pick.promo_boosted_odds = None
            pick.potential_profit   = (
                round((pick.combined_odds - 1) * (pick.stake or 10.0), 2) if pick.combined_odds else None
            )

        # Re-sync notes promo suffix
        _base = (pick.notes or "")
        for _tok in ("[PROFIT_BOOST", "[BONUS_BET", "[NO_SWEAT"):
            if _tok in _base:
                _base = _base[:_base.index(_tok)].strip()
        _sfx = _build_promo_note(pick.promo_type, pick.promo_boost_pct)
        parts = [p for p in [pick.reasoning or _base, _sfx] if p]
        pick.notes = " | ".join(parts) if parts else None

    db.commit()
    db.refresh(pick)

    # ── Re-extract signals if reasoning changed ──────────────────────────────
    if reasoning_changed:
        try:
            # Delete old signals for this pick first
            db.query(UserPickSignal).filter(UserPickSignal.user_pick_id == pick.id).delete()
            db.commit()
            fresh_legs = db.query(UserPickLeg).filter(UserPickLeg.user_pick_id == pick.id).all()
            usl.extract_signals(pick, fresh_legs, db)
            db.commit()
            lse.extract_llm_signals(pick, fresh_legs, db)
            db.commit()
        except Exception as _se:
            print(f"[signal-re-extract] non-fatal: {_se}")

    return {
        "id":                 pick.id,
        "status":             pick.status,
        "reasoning":          pick.reasoning,
        "promo_type":         pick.promo_type,
        "promo_boost_pct":    pick.promo_boost_pct,
        "promo_boosted_odds": pick.promo_boosted_odds,
        "potential_profit":   pick.potential_profit,
        "notes":              pick.notes,
        "reasoning_changed":  reasoning_changed,
        "boost_changed":      boost_changed,
    }


def _build_promo_note(promo_type: Optional[str], promo_boost_pct: Optional[float]) -> Optional[str]:
    """Build the promo suffix string stored in notes, e.g. '[PROFIT_BOOST +50%]'."""
    if not promo_type:
        return None
    boost_str = f" +{int(promo_boost_pct * 100)}%" if promo_boost_pct else ""
    return f"[{promo_type}{boost_str}]"


@app.post("/api/mock-bets/exploration-generate")
def exploration_generate(db: Session = Depends(get_db)):
    """
    Manually trigger exploration bet generation from today's Section A picks.
    Uses only already-fetched odds in the DB — does NOT consume OddsAPI credits.
    Safe to call multiple times (idempotent — deduplicates on bet_info).
    """
    try:
        result = mb_module.generate_exploration_bets(db)
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/mock-bets/exploration-tiered")
def exploration_tiered(
    game_date: Optional[str] = None,
    daily_target: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Manually trigger tiered exploration bet generation (player props + new markets).
    Safe to call multiple times — idempotent via bet_info dedup.
    """
    try:
        result = mb_module.generate_tiered_exploration_bets(
            db, game_date=game_date, daily_target=daily_target
        )
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/mock-bets/daily-sims")
def daily_sims(
    game_date: Optional[str] = None,
    bets_per_game: int = 3,
    db: Session = Depends(get_db),
):
    """
    Generate full-slate simulation bets for every game today.
    Covers all NBA/MLB/NHL games regardless of whether they made the main picks cut.
    Safe to call multiple times — idempotent via bet_info dedup.
    """
    try:
        result = mb_module.generate_daily_sims(db, game_date=game_date, bets_per_game=bets_per_game)
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Screenshot / slip-text parser endpoints ───────────────────────────────────

@app.post("/api/user-picks/parse")
async def parse_user_picks(
    images:    Optional[List[UploadFile]] = File(default=None),
    slip_text: Optional[str]             = Form(default=None),
    game_date: Optional[str]             = Form(default=None),
):
    """
    Parse one or more FanDuel bet slip screenshots OR pasted slip text.

    Accepts multipart/form-data with:
      images[]  — one or more image files (PNG/JPG)
      slip_text — pasted bet slip text (alternative to images)
      game_date — YYYY-MM-DD of the games (optional; defaults to today CT)

    Returns parsed picks ready for user review before submission.
    No picks are saved to the DB — call /api/user-picks/submit-parsed to commit.
    No OddsAPI credits consumed.
    """
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dt

    gdate = game_date or _dt.now(_ZI("America/Chicago")).strftime("%Y-%m-%d")

    parsed_picks: list = []

    # Vision parsing — one pick per image
    if images:
        for img in images:
            raw_bytes = await img.read()
            mime      = img.content_type or "image/png"
            result    = fp.parse_screenshot(raw_bytes, mime_type=mime, game_date=gdate)
            if "error" in result:
                parsed_picks.append({"error": result["error"], "source": img.filename})
            else:
                result["_source_filename"] = img.filename
                result["_input_mode"]      = "screenshot"
                parsed_picks.append(result)

    # Text parsing — single pick from pasted text
    elif slip_text:
        result = fp.parse_slip_text(slip_text, game_date=gdate)
        if "error" in result:
            parsed_picks.append({"error": result["error"], "source": "pasted_text"})
        else:
            result["_input_mode"] = "text_paste"
            parsed_picks.append(result)

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'images' (multipart file upload) or 'slip_text' (form field)."
        )

    ready      = sum(1 for p in parsed_picks if "error" not in p and not p.get("warnings"))
    needs_rev  = sum(1 for p in parsed_picks if "error" not in p and p.get("warnings"))
    errors     = sum(1 for p in parsed_picks if "error" in p)

    return {
        "parsed_picks": parsed_picks,
        "parse_summary": {
            "total_parsed":      len(parsed_picks),
            "ready_to_submit":   ready,
            "needs_review":      needs_rev,
            "parse_errors":      errors,
            "duplicates_detected": 0,  # checked at submit time
        },
    }


class ParsedLegSubmit(BaseModel):
    sport:           Optional[str]   = None
    market_type:     Optional[str]   = None
    team:            Optional[str]   = None
    player:          Optional[str]   = None
    point:           Optional[float] = None
    over_under:      Optional[str]   = None
    bet_info:        Optional[str]   = None
    home_team:       Optional[str]   = None
    away_team:       Optional[str]   = None
    matched_price:   Optional[int]   = None
    price_source:    Optional[str]   = "manual"
    price_confidence: Optional[float] = None
    leg_status:      Optional[str]   = "PENDING"
    is_part_of_sgp:  Optional[bool]  = False


class ParsedPickSubmit(BaseModel):
    n_legs:                  Optional[int]   = None
    combined_odds_american:  Optional[int]   = None
    combined_odds_decimal:   Optional[float] = None
    stake:                   Optional[float] = 10.0
    potential_payout:        Optional[float] = None
    potential_profit:        Optional[float] = None
    cashout_value:           Optional[float] = None
    bet_status:              Optional[str]   = "PENDING"
    fanduel_bet_id:          Optional[str]   = None
    bet_placed_at:           Optional[str]   = None
    boost:                   Optional[dict]  = None
    game_date:               Optional[str]   = None
    parsed_legs:             List[ParsedLegSubmit] = []
    reasoning:               Optional[str]   = None


class SubmitParsedRequest(BaseModel):
    picks:            List[ParsedPickSubmit]
    user_attestation: bool = False
    submission_mode:  str  = "pre_game"   # pre_game | retroactive


@app.post("/api/user-picks/submit-parsed")
def submit_parsed_picks(req: SubmitParsedRequest, db: Session = Depends(get_db)):
    """
    Submit parsed picks (from screenshot or text) to the A/B test layer.

    Runs:
      • Validation (dedup by fanduel_bet_id, mode checks)
      • Creates UserPick + UserPickLeg records
      • For retroactive SETTLED picks: immediately sets status + actual_profit
      • For CASHED_OUT picks: sets status = CASHED_OUT + actual_profit from cashout_value
      • Runs signal extraction (v1 keyword + v2 LLM + v3 comparison)
    """
    if not req.user_attestation:
        raise HTTPException(status_code=400, detail="user_attestation must be true.")

    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dt2

    created_ids: list[str] = []
    validation_results: list[dict] = []

    for parsed in req.picks:
        # ── Validate ──────────────────────────────────────────────────────────
        parsed_dict = parsed.dict()
        # Remap parsed_legs to the format validate_pick_submission expects
        parsed_dict["parsed_legs"] = [l.dict() for l in parsed.parsed_legs]

        val = fp.validate_pick_submission(parsed_dict, req.submission_mode, db)
        validation_results.append(val)
        if not val["passed"]:
            continue  # skip blocked picks, report blockers in response

        # ── Map bet_status → user_picks.status ───────────────────────────────
        bs   = (parsed.bet_status or "PENDING").upper()
        _status_map = {
            "PENDING":      "PENDING",
            "PARTIAL":      "PENDING",     # treat partial as pending at pick level
            "SETTLED_WIN":  "SETTLED_WIN",
            "SETTLED_LOSS": "SETTLED_LOSS",
            "CASHED_OUT":   "CASHED_OUT",
        }
        pick_status = _status_map.get(bs, "PENDING")

        # ── Game date ─────────────────────────────────────────────────────────
        gdate = parsed.game_date or _dt2.now(_ZI("America/Chicago")).strftime("%Y-%m-%d")

        # ── Boost / odds ──────────────────────────────────────────────────────
        boost_info  = parsed.boost or {}
        promo_type  = boost_info.get("type")  if boost_info else None
        promo_pct   = boost_info.get("pct")   if boost_info else None
        promo_boost_am = boost_info.get("boosted_odds_american") if boost_info else None

        combined_dec = parsed.combined_odds_decimal
        if combined_dec is None and parsed.combined_odds_american:
            combined_dec = fp._american_to_decimal(parsed.combined_odds_american)

        boosted_dec = None
        if promo_boost_am:
            boosted_dec = fp._american_to_decimal(promo_boost_am)

        # ── Compute actual_profit for settled / cashed-out picks ───────────────
        stake        = parsed.stake or 10.0
        actual_profit = None
        settled_at    = None
        if pick_status == "SETTLED_WIN":
            if promo_type == "PROFIT_BOOST" and promo_pct and combined_dec:
                base_profit   = round((combined_dec - 1) * stake, 2)
                actual_profit = round(base_profit * (1 + promo_pct), 2)
            elif promo_type == "BONUS_BET":
                actual_profit = round((combined_dec - 1) * stake, 2) if combined_dec else None
            elif combined_dec:
                actual_profit = round((combined_dec - 1) * stake, 2)
            settled_at = _dt2.utcnow()
        elif pick_status == "SETTLED_LOSS":
            actual_profit = 0.0 if promo_type in ("BONUS_BET", "NO_SWEAT") else -stake
            settled_at    = _dt2.utcnow()
        elif pick_status == "CASHED_OUT":
            co_val        = parsed.cashout_value or 0.0
            actual_profit = round(co_val - stake, 2)
            settled_at    = _dt2.utcnow()

        # ── ingestion_source ──────────────────────────────────────────────────
        is_retro = (req.submission_mode == "retroactive")
        if is_retro:
            ing_src = "screenshot_retroactive"
        else:
            ing_src = "screenshot_pregame"
        # Override for text paste if indicated (parsed from _input_mode field, not modelled here)

        # ── Promo note ────────────────────────────────────────────────────────
        _notes_parts = []
        if parsed.reasoning:
            _notes_parts.append(parsed.reasoning)
        if promo_type:
            boost_str = f" +{int(promo_pct*100)}%" if promo_pct else ""
            _notes_parts.append(f"[{promo_type}{boost_str}]")
        _notes = " | ".join(_notes_parts) if _notes_parts else None

        # ── potential_profit ──────────────────────────────────────────────────
        if parsed.potential_profit is not None:
            _pot_profit = parsed.potential_profit
        elif boosted_dec and promo_type == "PROFIT_BOOST":
            _pot_profit = round((boosted_dec - 1) * stake, 2)
        elif combined_dec:
            _pot_profit = round((combined_dec - 1) * stake, 2)
        else:
            _pot_profit = None

        # ── bet_placed_at ─────────────────────────────────────────────────────
        _bet_placed_at = None
        if parsed.bet_placed_at:
            try:
                _bet_placed_at = _dt2.fromisoformat(parsed.bet_placed_at.replace("Z", "+00:00"))
            except Exception:
                pass

        pick_id = str(uuid.uuid4())

        pick = UserPick(
            id                  = pick_id,
            game_date           = gdate,
            bet_type            = "parlay" if len(parsed.parsed_legs) > 1 else "straight",
            stake               = stake,
            legs                = len(parsed.parsed_legs),
            combined_odds       = combined_dec,
            notes               = _notes,
            reasoning           = parsed.reasoning or None,
            promo_type          = promo_type,
            promo_boost_pct     = promo_pct,
            promo_boosted_odds  = boosted_dec,
            potential_profit    = _pot_profit,
            potential_payout    = parsed.potential_payout,
            status              = pick_status,
            actual_profit       = actual_profit,
            settled_at          = settled_at,
            # ingestion fields
            fanduel_bet_id      = parsed.fanduel_bet_id or None,
            bet_placed_at       = _bet_placed_at,
            added_retroactively = is_retro,
            ingestion_source    = ing_src,
            cashed_out_value    = parsed.cashout_value if pick_status == "CASHED_OUT" else None,
        )
        db.add(pick)

        # ── Legs ──────────────────────────────────────────────────────────────
        leg_objs = []
        for i, pl in enumerate(parsed.parsed_legs):
            am  = pl.matched_price
            dec = fp._american_to_decimal(am) if am is not None else None
            desc = pl.bet_info or pl.team or f"Leg {i+1}"

            # Pre-populate leg_result from screenshot for settled picks
            lr = None
            if pl.leg_status in ("WON", "WIN"):
                lr = "WIN"
            elif pl.leg_status in ("LOST", "LOSS"):
                lr = "LOSS"
            elif pl.leg_status == "PUSH":
                lr = "PUSH"

            _mkt_lo2 = (pl.market_type or '').lower()
            _exp2 = 0.5 if _mkt_lo2 in ('h2h', 'moneyline', 'ml') else pl.point
            leg_obj = UserPickLeg(
                user_pick_id           = pick_id,
                leg_index              = i,
                description            = desc,
                sport                  = pl.sport,
                market_type            = pl.market_type,
                team                   = pl.team,
                player                 = pl.player,
                odds_american          = am,
                odds_decimal           = dec,
                leg_result             = lr,
                is_part_of_sgp         = pl.is_part_of_sgp or False,
                price_source           = pl.price_source or "manual",
                price_confidence       = pl.price_confidence,
                point                  = pl.point,
                expected_outcome_value = _exp2,
            )
            db.add(leg_obj)
            leg_objs.append(leg_obj)

        db.commit()
        db.refresh(pick)

        # ── Signal extraction ─────────────────────────────────────────────────
        fresh_legs = db.query(UserPickLeg).filter(
            UserPickLeg.user_pick_id == pick.id
        ).all()

        try:
            usl.extract_signals(pick, fresh_legs, db)
            db.commit()
        except Exception as _se:
            print(f"[submit-parsed signal-v1] non-fatal: {_se}")

        try:
            lse.extract_llm_signals(pick, fresh_legs, db)
            db.commit()
        except Exception as _le:
            print(f"[submit-parsed signal-v2] non-fatal: {_le}")

        try:
            usl.extract_comparison_signals(pick, fresh_legs, db)
            db.commit()
        except Exception as _ce:
            print(f"[submit-parsed signal-v3] non-fatal: {_ce}")

        # ── For pre-settled picks: propagate leg signals immediately ───────────
        if pick_status in ("SETTLED_WIN", "SETTLED_LOSS"):
            try:
                _propagate_leg_signals(pick, db)
            except Exception as _pe:
                print(f"[submit-parsed propagate] non-fatal: {_pe}")

        created_ids.append(pick_id)

    blocked = [
        {"blockers": v["blockers"]}
        for v in validation_results if not v["passed"]
    ]

    return {
        "created":        len(created_ids),
        "created_ids":    created_ids,
        "blocked":        len(blocked),
        "blocked_detail": blocked,
        "warnings_total": sum(len(v["warnings"]) for v in validation_results),
    }


@app.get("/api/ab-test/comparison")
def ab_test_comparison(
    days:             Optional[int] = None,
    since:            Optional[str] = None,
    ingestion_filter: Optional[str] = None,  # pregame_only | include_cashouts | all
    include_cashouts: Optional[bool] = True,
    db:               Session       = Depends(get_db),
):
    """
    Head-to-head comparison of user picks vs model mock bets.

    Both ABSOLUTE and RATE metrics are returned with equal prominence.
    The user's hypothesis: they can beat the model on TOTAL P&L even at lower volume.

    Query params:
      ?days=N                  — last N days
      ?since=YYYY-MM-DD        — from that date forward
      ?ingestion_filter=pregame_only — exclude added_retroactively=1 picks
      ?include_cashouts=false  — exclude CASHED_OUT picks from metrics

    Response includes:
      user / model   — full stats objects
      head_to_head   — absolute winner, rate winner, verdict
      by_sport       — per-sport breakdown
      daily_series   — day-by-day cumulative P&L for chart
      overlap_analysis — shared vs user-only vs model-only picks
      ingestion_summary — breakdown of pre-game vs retroactive picks
    """
    from datetime import datetime as _dt, timedelta as _td
    import math as _math

    # ── Date filter ──────────────────────────────────────────────────────────
    date_floor: Optional[str] = None
    if since:
        date_floor = since
    elif days:
        _ct_now = _dt.now(__import__("zoneinfo").ZoneInfo("America/Chicago"))
        date_floor = (_ct_now - _td(days=days)).strftime("%Y-%m-%d")

    # ── Fetch ALL user picks (for ingestion_summary) ──────────────────────────
    uq_all = db.query(UserPick)
    if date_floor:
        uq_all = uq_all.filter(UserPick.game_date >= date_floor)
    user_picks_raw = uq_all.all()

    # ── Ingestion summary (always computed from full set) ─────────────────────
    n_pregame    = sum(1 for p in user_picks_raw if not p.added_retroactively)
    n_retro      = sum(1 for p in user_picks_raw if p.added_retroactively)
    n_cashout    = sum(1 for p in user_picks_raw if p.status == "CASHED_OUT")
    ingestion_summary = {
        "total":            len(user_picks_raw),
        "pregame":          n_pregame,
        "retroactive":      n_retro,
        "cashed_out":       n_cashout,
    }

    # ── Apply ingestion_filter ────────────────────────────────────────────────
    user_picks_all = list(user_picks_raw)
    if ingestion_filter == "pregame_only":
        user_picks_all = [p for p in user_picks_all if not p.added_retroactively]
    if not include_cashouts:
        user_picks_all = [p for p in user_picks_all if p.status != "CASHED_OUT"]
    # CASHED_OUT picks count toward submission count but not win/loss stats

    settled_user = [p for p in user_picks_all if p.status in ("SETTLED_WIN", "SETTLED_LOSS")]

    def _user_stats(picks):
        n = len(picks)
        if not n:
            return {"n_bets_submitted": 0, "n_settled": 0, "wins": 0, "losses": 0,
                    "wr_pct": None, "total_staked": 0.0, "total_pnl": 0.0,
                    "roi_pct": None, "avg_pnl_per_bet": None}
        wins   = sum(1 for p in picks if p.status == "SETTLED_WIN")
        losses = sum(1 for p in picks if p.status == "SETTLED_LOSS")
        staked = sum(p.stake or 0.0 for p in picks)
        pnl    = sum(p.actual_profit or 0.0 for p in picks)
        n_s    = wins + losses
        return {
            "n_bets_submitted": len(picks),
            "n_settled":        n_s,
            "wins":             wins,
            "losses":           losses,
            "wr_pct":           round(wins / n_s * 100, 2) if n_s else None,
            "total_staked":     round(staked, 2),
            "total_pnl":        round(pnl, 2),
            "roi_pct":          round(pnl / staked * 100, 2) if staked else None,
            "avg_pnl_per_bet":  round(pnl / n_s, 2) if n_s else None,
        }

    user_stats = _user_stats(settled_user)

    # ── Fetch model mock bets ────────────────────────────────────────────────
    mq = db.query(MockBet).filter(
        MockBet.source.notin_(["exploration", "forced_generation"]),
        MockBet.status.in_(["SETTLED_WIN", "SETTLED_LOSS"]),
    )
    if date_floor:
        mq = mq.filter(MockBet.game_date >= date_floor)
    model_bets = mq.all()

    def _model_stats(bets):
        n = len(bets)
        if not n:
            return {"n_bets_submitted": n, "n_settled": n, "wins": 0, "losses": 0,
                    "wr_pct": None, "total_staked": 0.0, "total_pnl": 0.0,
                    "roi_pct": None, "avg_pnl_per_bet": None}
        wins   = sum(1 for b in bets if b.status == "SETTLED_WIN")
        losses = sum(1 for b in bets if b.status == "SETTLED_LOSS")
        staked = sum((b.amount or 10.0) for b in bets)
        pnl    = sum((b.actual_profit or 0.0) for b in bets)
        return {
            "n_bets_submitted": n,
            "n_settled":        n,
            "wins":             wins,
            "losses":           losses,
            "wr_pct":           round(wins / n * 100, 2) if n else None,
            "total_staked":     round(staked, 2),
            "total_pnl":        round(pnl, 2),
            "roi_pct":          round(pnl / staked * 100, 2) if staked else None,
            "avg_pnl_per_bet":  round(pnl / n, 2) if n else None,
        }

    model_stats = _model_stats(model_bets)

    # ── Head-to-head ─────────────────────────────────────────────────────────
    u_pnl  = user_stats["total_pnl"]
    m_pnl  = model_stats["total_pnl"]
    u_roi  = user_stats["roi_pct"]
    m_roi  = model_stats["roi_pct"]
    u_wr   = user_stats["wr_pct"]
    m_wr   = model_stats["wr_pct"]

    pnl_winner = "user" if u_pnl > m_pnl else ("model" if m_pnl > u_pnl else "tie")
    roi_winner = (
        "user"  if (u_roi is not None and m_roi is not None and u_roi > m_roi) else
        "model" if (u_roi is not None and m_roi is not None and m_roi > u_roi) else
        "insufficient_data"
    )
    wr_winner = (
        "user"  if (u_wr is not None and m_wr is not None and u_wr > m_wr) else
        "model" if (u_wr is not None and m_wr is not None and m_wr > u_wr) else
        "insufficient_data"
    )

    scores = {"user": 0, "model": 0}
    for w in (pnl_winner, roi_winner, wr_winner):
        if w in scores:
            scores[w] += 1

    if scores["user"] == 3:
        verdict = "user_sweep"
    elif scores["model"] == 3:
        verdict = "model_sweep"
    elif scores["user"] > scores["model"]:
        verdict = "user_leads"
    elif scores["model"] > scores["user"]:
        verdict = "model_leads"
    else:
        verdict = "split"

    pnl_adv  = round(u_pnl - m_pnl, 2)
    roi_adv  = round((u_roi or 0.0) - (m_roi or 0.0), 2)
    wr_adv   = round((u_wr  or 0.0) - (m_wr  or 0.0), 2)

    user_wins_pnl_at_lower_volume = (
        pnl_winner == "user"
        and user_stats["n_bets_submitted"] < model_stats["n_bets_submitted"]
    )

    verdict_parts = []
    if pnl_winner == "user":
        verdict_parts.append(f"User leads by ${abs(pnl_adv):.2f} total P&L")
    elif pnl_winner == "model":
        verdict_parts.append(f"Model leads by ${abs(pnl_adv):.2f} total P&L")
    if roi_winner == "user":
        verdict_parts.append(f"User leads by {abs(roi_adv):.1f} ROI points")
    elif roi_winner == "model":
        verdict_parts.append(f"Model leads by {abs(roi_adv):.1f} ROI points")
    if wr_winner == "user":
        verdict_parts.append(f"User leads by {abs(wr_adv):.1f} WR points")
    elif wr_winner == "model":
        verdict_parts.append(f"Model leads by {abs(wr_adv):.1f} WR points")
    verdict_summary = " | ".join(verdict_parts) if verdict_parts else "Insufficient data"

    head_to_head = {
        "absolute": {
            "user_total_pnl":              u_pnl,
            "model_total_pnl":             m_pnl,
            "user_pnl_advantage":          pnl_adv,
            "winner_by_total_pnl":         pnl_winner,
            "user_wins_pnl_at_lower_volume": user_wins_pnl_at_lower_volume,
        },
        "rate": {
            "user_roi_pct":                u_roi,
            "model_roi_pct":               m_roi,
            "roi_advantage_pct_pts":       roi_adv,
            "winner_by_roi":               roi_winner,
        },
        "win_rate": {
            "user_wr_pct":                 u_wr,
            "model_wr_pct":                m_wr,
            "wr_advantage_pct_pts":        wr_adv,
            "winner_by_wr":                wr_winner,
        },
        "verdict":         verdict,
        "verdict_summary": verdict_summary,
        "scores":          scores,
    }

    # ── Per-sport breakdown ──────────────────────────────────────────────────
    user_legs_all = db.query(UserPickLeg).filter(
        UserPickLeg.user_pick_id.in_([p.id for p in settled_user])
    ).all() if settled_user else []

    user_sport_map: dict[str, list] = {}
    for p in settled_user:
        legs = [l for l in user_legs_all if l.user_pick_id == p.id]
        sports = list({l.sport for l in legs if l.sport}) or ["Unknown"]
        for sp in sports:
            user_sport_map.setdefault(sp, []).append(p)

    model_sport_map: dict[str, list] = {}
    for b in model_bets:
        sp = (b.sport or "Unknown").strip()
        model_sport_map.setdefault(sp, []).append(b)

    all_sports = sorted(set(list(user_sport_map.keys()) + list(model_sport_map.keys())))
    by_sport = []
    for sp in all_sports:
        u_bets = user_sport_map.get(sp, [])
        m_bets = model_sport_map.get(sp, [])
        u_wins = sum(1 for p in u_bets if p.status == "SETTLED_WIN")
        m_wins = sum(1 for b in m_bets if b.status == "SETTLED_WIN")
        u_sp_pnl = round(sum(p.actual_profit or 0.0 for p in u_bets), 2)
        m_sp_pnl = round(sum(b.actual_profit or 0.0 for b in m_bets), 2)
        by_sport.append({
            "sport":          sp,
            "user_n":         len(u_bets),
            "user_wins":      u_wins,
            "user_wr_pct":    round(u_wins / len(u_bets) * 100, 1) if u_bets else None,
            "user_pnl":       u_sp_pnl,
            "model_n":        len(m_bets),
            "model_wins":     m_wins,
            "model_wr_pct":   round(m_wins / len(m_bets) * 100, 1) if m_bets else None,
            "model_pnl":      m_sp_pnl,
        })

    # ── Daily series for cumulative P&L chart ───────────────────────────────
    user_by_date: dict[str, float] = {}
    for p in settled_user:
        d = p.game_date or (p.settled_at.strftime("%Y-%m-%d") if p.settled_at else "unknown")
        user_by_date[d] = round(user_by_date.get(d, 0.0) + (p.actual_profit or 0.0), 2)

    model_by_date: dict[str, float] = {}
    for b in model_bets:
        d = b.game_date or (b.settled_at.strftime("%Y-%m-%d") if b.settled_at else "unknown")
        model_by_date[d] = round(model_by_date.get(d, 0.0) + (b.actual_profit or 0.0), 2)

    all_dates = sorted(set(list(user_by_date.keys()) + list(model_by_date.keys())))
    daily_series = []
    u_cum = m_cum = 0.0
    for d in all_dates:
        u_cum = round(u_cum + user_by_date.get(d, 0.0), 2)
        m_cum = round(m_cum + model_by_date.get(d, 0.0), 2)
        daily_series.append({
            "date":              d,
            "user_daily_pnl":    user_by_date.get(d, 0.0),
            "model_daily_pnl":   model_by_date.get(d, 0.0),
            "user_cumulative":   u_cum,
            "model_cumulative":  m_cum,
        })

    # ── Overlap analysis ────────────────────────────────────────────────────
    # A pick "overlaps" if the user submitted a pick on the same game_date and sport
    # as a model mock bet.  Exact leg matching is not feasible without fixture IDs.
    user_date_sports  = {(p.game_date, l.sport) for p in settled_user
                          for l in user_legs_all if l.user_pick_id == p.id and l.sport}
    model_date_sports = {(b.game_date, b.sport or "") for b in model_bets if b.sport}
    shared = user_date_sports & model_date_sports
    user_only  = user_date_sports  - model_date_sports
    model_only = model_date_sports - user_date_sports

    overlap_analysis = {
        "shared_date_sport_combos":    len(shared),
        "user_only_date_sport_combos": len(user_only),
        "model_only_date_sport_combos":len(model_only),
        "note": (
            "Overlap is approximated by (game_date, sport) pairs. "
            "Exact leg-level matching requires fixture IDs on user pick legs."
        ),
    }

    return {
        "period": {
            "since":          date_floor,
            "days_requested": days,
        },
        "user":               user_stats,
        "model":              model_stats,
        "head_to_head":       head_to_head,
        "by_sport":           by_sport,
        "daily_series":       daily_series,
        "overlap_analysis":   overlap_analysis,
        "ingestion_summary":  ingestion_summary,
        "filters_applied": {
            "ingestion_filter": ingestion_filter or "all",
            "include_cashouts": include_cashouts,
        },
    }


@app.get("/api/ab-test/learning-curve")
def ab_test_learning_curve(
    days:  Optional[int] = None,
    since: Optional[str] = None,
    db:    Session       = Depends(get_db),
):
    """
    Bidirectional learning report — what patterns the model has adopted (user edges)
    and what it now avoids (user weaknesses).

    Returns:
      user_signals_ingested, patterns_classified (by class),
      top_adopt_patterns, top_avoid_patterns,
      model_improvement_attribution
    """
    from datetime import datetime as _dt, timedelta as _td
    date_floor: Optional[str] = None
    if since:
        date_floor = since
    elif days:
        _ct_now = _dt.now(__import__("zoneinfo").ZoneInfo("America/Chicago"))
        date_floor = (_ct_now - _td(days=days)).strftime("%Y-%m-%d")

    report = usl.learning_curve_report(db, date_floor=date_floor)
    # Augment with feature-source comparison (keyword vs LLM)
    try:
        report["feature_source_comparison"] = lse.feature_source_comparison(db)
    except Exception as _fse:
        report["feature_source_comparison"] = {"error": str(_fse)}
    return report


@app.post("/api/user-picks/backfill-llm-features")
def backfill_llm_features(
    limit: int = 100,
    db:    Session = Depends(get_db),
):
    """
    Run LLM signal extraction on all user_picks where it hasn't run yet
    (llm_features_extracted=0) and notes are present.

    Processes at most `limit` picks per call to avoid long blocking requests.
    Call repeatedly until picks_remaining=0.
    """
    return lse.backfill_llm_features(db, limit=limit)


def _compute_settle_profit(pick: UserPick, result_upper: str, explicit_profit: Optional[float]) -> float:
    """
    Compute actual_profit for a user pick settlement.

    Boost rules (FanDuel semantics — boost applied to NET PROFIT only, not total decimal):
      PROFIT_BOOST:  base_profit * (1 + promo_boost_pct)
      BONUS_BET WIN: base_profit only (stake not returned — free bet)
      BONUS_BET LOSS: 0.0 (free bet, nothing lost)
      NO_SWEAT LOSS:  0.0 (first-loss insurance, treated as site credit returned)
      NO_SWEAT WIN:   normal profit (insurance not triggered)
    """
    stake = pick.stake or 10.0

    if result_upper == "WIN":
        if explicit_profit is not None:
            return explicit_profit
        base_profit = round((pick.combined_odds - 1) * stake, 2) if pick.combined_odds else 0.0

        if pick.promo_type == "PROFIT_BOOST" and pick.promo_boost_pct:
            # Boost is on profit only: profit × (1 + boost_pct)
            return round(base_profit * (1 + pick.promo_boost_pct), 2)
        elif pick.promo_type == "BONUS_BET":
            # Free bet wins return profit only, stake is not returned
            return base_profit
        else:
            return base_profit

    else:  # LOSS
        if explicit_profit is not None:
            return explicit_profit
        if pick.promo_type == "BONUS_BET":
            return 0.0   # free bet loss costs nothing
        if pick.promo_type == "NO_SWEAT":
            return 0.0   # first-loss insurance returns stake as site credit
        return -stake


class ProvideOutcomeRequest(BaseModel):
    actual_outcome_value: float
    notes: Optional[str] = None


@app.post("/api/user-pick-legs/{leg_id}/provide-outcome")
def provide_leg_outcome(
    leg_id: int,
    req:    ProvideOutcomeRequest,
    db:     Session = Depends(get_db),
):
    """
    User-provided actual outcome value for a settled leg.

    For player props/spreads/totals the system can't auto-resolve,
    user enters the actual stat/score so miss_margin can be computed.

    Examples:
      SGA 20+ pts: actual_outcome_value=18 → miss_margin=-2.0
      Lakers/Thunder Over 209.5: actual_outcome_value=205 → miss_margin=-4.5
      Yankees +2.5: actual_outcome_value=-4 → miss_margin=-6.5
    """
    leg = db.query(UserPickLeg).filter(UserPickLeg.id == leg_id).first()
    if not leg:
        raise HTTPException(status_code=404, detail="Leg not found.")

    leg.actual_outcome_value = req.actual_outcome_value
    leg.outcome_source       = "user_provided"
    if leg.expected_outcome_value is not None:
        leg.miss_margin = round(req.actual_outcome_value - leg.expected_outcome_value, 4)

    db.commit()
    db.refresh(leg)
    return {
        "leg_id":                leg.id,
        "description":           leg.description,
        "actual_outcome_value":  leg.actual_outcome_value,
        "expected_outcome_value":leg.expected_outcome_value,
        "miss_margin":           leg.miss_margin,
        "outcome_source":        leg.outcome_source,
    }


@app.get("/api/ab-test/unresolved-legs")
def get_unresolved_legs(db: Session = Depends(get_db)):
    """
    Return settled legs missing actual_outcome_value.
    Excludes h2h legs (auto-resolved at settlement).
    Used to show the 'N legs need outcome' badge in the UI.
    """
    _H2H = {'h2h', 'moneyline', 'ml', ''}
    settled_picks = db.query(UserPick).filter(
        UserPick.status.in_(["SETTLED_WIN", "SETTLED_LOSS"])
    ).all()
    if not settled_picks:
        return {"count": 0, "legs": []}

    pick_map = {p.id: p for p in settled_picks}
    all_legs = db.query(UserPickLeg).filter(
        UserPickLeg.user_pick_id.in_(list(pick_map.keys()))
    ).all()

    unresolved = [
        l for l in all_legs
        if (l.market_type or '').lower() not in _H2H
        and l.actual_outcome_value is None
        and l.expected_outcome_value is not None   # only prompt when we know the threshold
    ]

    return {
        "count": len(unresolved),
        "legs": [
            {
                "leg_id":                l.id,
                "pick_id":               l.user_pick_id,
                "game_date":             pick_map[l.user_pick_id].game_date,
                "description":           l.description,
                "market_type":           l.market_type,
                "team":                  l.team,
                "player":                l.player,
                "leg_result":            l.leg_result,
                "expected_outcome_value":l.expected_outcome_value,
                "outcome_source":        l.outcome_source,
            }
            for l in unresolved
        ],
    }


@app.put("/api/user-picks/{pick_id}/settle")
def settle_user_pick(
    pick_id: str,
    result:  str            = Query(..., description="WIN or LOSS"),
    profit:  Optional[float]= Query(None, description="Actual profit (optional — computed from odds+boost if omitted)"),
    force:   bool           = Query(False, description="Re-settle an already-settled pick (corrects actual_profit)"),
    db:      Session        = Depends(get_db),
):
    """
    Manually settle a user pick.

    result: WIN | LOSS
    profit: explicit dollar P&L (overrides auto-calculation when supplied)
    force:  true → re-settle an already-settled pick (useful for boost corrections)

    Auto-calculation applies boost logic:
      PROFIT_BOOST → profit × (1 + boost_pct)
      BONUS_BET loss → $0 (free bet, nothing lost)
      NO_SWEAT loss  → $0 (insurance)
    """
    pick = db.query(UserPick).filter(UserPick.id == pick_id).first()
    if not pick:
        raise HTTPException(status_code=404, detail="Pick not found.")
    if pick.status not in ("PENDING",) and not force:
        raise HTTPException(status_code=409, detail=f"Pick already settled: {pick.status}. Use ?force=true to re-settle.")

    r = result.upper()
    if r not in ("WIN", "LOSS"):
        raise HTTPException(status_code=400, detail="result must be WIN or LOSS")

    pick.status       = "SETTLED_WIN" if r == "WIN" else "SETTLED_LOSS"
    pick.actual_profit = _compute_settle_profit(pick, r, profit)
    pick.settled_at   = datetime.utcnow()

    db.commit()
    db.refresh(pick)

    # Propagate outcome to per-leg signals (all legs treated as same outcome as pick)
    _propagate_leg_signals(pick, db)

    # Update learning signal performance aggregates
    try:
        usl.settle_pick_signals(pick, db)
    except Exception as _le:
        print(f"[signal-settle] non-fatal: {_le}")

    return {
        "id":              pick.id,
        "status":          pick.status,
        "actual_profit":   pick.actual_profit,
        "settled_at":      pick.settled_at.isoformat(),
        "promo_type":      pick.promo_type,
        "promo_boost_pct": pick.promo_boost_pct,
        "boost_applied":   pick.promo_type == "PROFIT_BOOST" and bool(pick.promo_boost_pct),
    }


def _propagate_leg_signals(pick: UserPick, db: Session) -> None:
    """
    After pick-level settlement, populate:
      1. user_pick_legs.leg_result = WIN | LOSS  (all legs match pick outcome for parlays)
      2. user_pick_signals.outcome  = WON | LOST for signals tied to specific legs
         and for pick-level signals (user_pick_leg_id IS NULL)
    """
    outcome     = "WON"  if pick.status == "SETTLED_WIN"  else "LOST"
    leg_result  = "WIN"  if pick.status == "SETTLED_WIN"  else "LOSS"

    # ── 1. Leg results + outcome capture ─────────────────────────────────────
    legs = db.query(UserPickLeg).filter(UserPickLeg.user_pick_id == pick.id).all()
    _H2H_MKTS = {'h2h', 'moneyline', 'ml', ''}
    for leg in legs:
        leg.leg_result = leg_result
        # Skip if already user-provided
        if leg.outcome_source == 'user_provided':
            continue
        mkt = (leg.market_type or '').lower()
        if mkt in _H2H_MKTS:
            # Binary: 1 = won, 0 = lost
            leg.actual_outcome_value   = 1.0 if leg_result == 'WIN' else 0.0
            leg.expected_outcome_value = leg.expected_outcome_value if leg.expected_outcome_value is not None else 0.5
            leg.miss_margin            = round(leg.actual_outcome_value - leg.expected_outcome_value, 4)
            leg.outcome_source         = 'auto_settlement'
        else:
            # spread / total / player prop — expected is known (stored point), actual needs user input
            if leg.expected_outcome_value is None and leg.point is not None:
                leg.expected_outcome_value = leg.point
            if leg.actual_outcome_value is None:
                leg.outcome_source = 'unknown'
    db.flush()

    # ── 2. Signal outcomes ────────────────────────────────────────────────────
    signals = db.query(UserPickSignal).filter(
        UserPickSignal.user_pick_id == pick.id,
        UserPickSignal.outcome == "PENDING",
    ).all()
    for sig in signals:
        sig.outcome = outcome
    db.flush()
    db.commit()


# ─── Scout routes (Phase 5 — scouting + placement layer) ─────────────────────

@app.get("/api/scout/props")
def get_scout_props(
    date:    Optional[str] = None,
    sport:   Optional[str] = None,
    grade:   Optional[str] = None,
    market:  Optional[str] = None,
    search:  Optional[str] = None,
    limit:   int = 200,
    offset:  int = 0,
    db: Session = Depends(get_db),
):
    """
    Return scouted props for a given date (defaults to today).
    Filterable by sport, quality_grade, market_type, and free-text search
    (matches player_name, team, home_team, or away_team).
    """
    import json as _json
    from sqlalchemy import text
    from datetime import datetime, timezone

    scout_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    filters = ["scout_date = :date"]
    params: dict = {"date": scout_date}
    if sport:
        filters.append("sport = :sport")
        params["sport"] = sport.upper()
    if grade:
        filters.append("quality_grade = :grade")
        params["grade"] = grade.upper()
    if market:
        filters.append("market_type = :market")
        params["market"] = market
    if search:
        filters.append(
            "(player_name LIKE :q OR team LIKE :q OR home_team LIKE :q OR away_team LIKE :q)"
        )
        params["q"] = f"%{search}%"

    where = " AND ".join(filters)
    rows = db.execute(text(f"""
        SELECT id, scout_date, sport, game_id, home_team, away_team,
               commence_time, market_type, player_name, player_id, team, side,
               threshold, projected_value, projected_low_95, projected_high_95,
               projected_std_dev, hit_probability, quality_grade,
               confidence_factors, risk_factors,
               actual_outcome_value, actual_hit, scout_accuracy,
               data_source, projection_version, created_at
        FROM   scouted_props
        WHERE  {where}
        ORDER  BY hit_probability DESC
        LIMIT  :limit OFFSET :offset
    """), {**params, "limit": limit, "offset": offset}).fetchall()

    cols = [
        "id", "scout_date", "sport", "game_id", "home_team", "away_team",
        "commence_time", "market_type", "player_name", "player_id", "team", "side",
        "threshold", "projected_value", "projected_low_95", "projected_high_95",
        "projected_std_dev", "hit_probability", "quality_grade",
        "confidence_factors", "risk_factors",
        "actual_outcome_value", "actual_hit", "scout_accuracy",
        "data_source", "projection_version", "created_at",
    ]
    props = []
    for row in rows:
        d = dict(zip(cols, row))
        for jf in ("confidence_factors", "risk_factors"):
            try:
                d[jf] = _json.loads(d[jf]) if d[jf] else []
            except Exception:
                d[jf] = []
        props.append(d)

    # Count total matching for pagination info
    total_count_row = db.execute(text(f"""
        SELECT COUNT(*) FROM scouted_props WHERE {where}
    """), params).fetchone()
    total_count = total_count_row[0] if total_count_row else len(props)

    return {
        "scout_date":  scout_date,
        "total":       total_count,
        "returned":    len(props),
        "offset":      offset,
        "props":       props,
    }


@app.get("/api/scout/compare")
def scout_market_compare(
    date:  Optional[str] = None,
    sport: Optional[str] = None,
    grade: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Compare today's scouted props against current market odds from fixtures.
    For each prop, finds the closest market line and computes:
      - market_line: the bookmaker's current line
      - market_implied_prob: implied probability from best available odds
      - edge_pp: our projected hit_probability - market_implied_prob (pp)
    Only player props are matched (team market matching is todo).
    """
    import json as _json
    from sqlalchemy import text
    from datetime import datetime, timezone

    scout_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Fetch scouted props
    filters = ["sp.scout_date = :d", "sp.actual_hit IS NULL"]
    params: dict = {"d": scout_date}
    if sport:
        filters.append("sp.sport = :sport")
        params["sport"] = sport.upper()
    if grade:
        filters.append("sp.quality_grade = :grade")
        params["grade"] = grade.upper()

    props = db.execute(text(f"""
        SELECT sp.id, sp.sport, sp.home_team, sp.away_team, sp.market_type,
               sp.player_name, sp.side, sp.threshold, sp.hit_probability,
               sp.quality_grade
        FROM   scouted_props sp
        WHERE  {" AND ".join(filters)}
        ORDER  BY sp.hit_probability DESC
        LIMIT  100
    """), params).fetchall()

    prop_cols = ["id", "sport", "home_team", "away_team", "market_type",
                 "player_name", "side", "threshold", "hit_probability", "quality_grade"]

    # Fetch fixtures with odds for today's games
    fixtures = db.execute(text("""
        SELECT id, home_team, away_team, bookmakers FROM fixtures
        WHERE commence_time >= datetime('now', '-6 hours')
        AND   commence_time <= datetime('now', '+30 hours')
    """)).fetchall()

    # Build a lookup: (home_team_lower, away_team_lower) -> bookmakers
    def _name_key(s: str) -> str:
        return (s or "").lower().strip()

    fixture_map: dict = {}
    for fx_id, home, away, bm_json in fixtures:
        bm = bm_json if isinstance(bm_json, list) else _json.loads(bm_json or "[]")
        fixture_map[(_name_key(home), _name_key(away))] = bm

    def _implied_prob(american_odds: int) -> float:
        """Convert American odds to no-vig implied probability."""
        if american_odds > 0:
            return 100.0 / (american_odds + 100.0)
        return abs(american_odds) / (abs(american_odds) + 100.0)

    def _find_market_line(bookmakers: list, market_key: str, team_or_player: str,
                          side: str, threshold: float) -> tuple | None:
        """
        Search bookmaker markets for a matching line.
        Returns (best_odds, implied_prob) or None.
        """
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key", "") != market_key:
                    continue
                for outcome in mkt.get("outcomes", []):
                    o_name  = (outcome.get("name") or "").lower()
                    o_point = outcome.get("point")
                    o_price = outcome.get("price")
                    # Match by team/player name fragment
                    if team_or_player and team_or_player.lower() not in o_name:
                        continue
                    # Match threshold if relevant
                    if threshold is not None and o_point is not None:
                        if abs(float(o_point) - float(threshold)) > 1.0:
                            continue
                    if o_price is not None:
                        return float(o_price), _implied_prob(int(o_price))
        return None

    # Market key mapping from our market_type to OddsAPI key
    _MKT_MAP = {
        "player_points":   "player_points",
        "player_rebounds": "player_rebounds",
        "player_assists":  "player_assists",
        "player_pra":      "player_points_rebounds_assists",
        "player_threes":   "player_threes",
        "player_steals":   "player_steals",
        "player_blocks":   "player_blocks",
        "player_hits":     "batter_hits",
        "player_strikeouts": "pitcher_strikeouts",
        "player_total_bases": "batter_total_bases",
        "player_shots":    "player_shots_on_goal",
        "player_goals":    "player_goal_scorer",
        "h2h":             "h2h",
        "spread":          "spreads",
        "totals":          "totals",
    }

    comparisons = []
    for row in props:
        d = dict(zip(prop_cols, row))
        bm = fixture_map.get((_name_key(d["home_team"]), _name_key(d["away_team"])), [])

        market_key = _MKT_MAP.get(d["market_type"], d["market_type"])
        team_or_player = d.get("player_name") or ""
        match = _find_market_line(bm, market_key, team_or_player, d["side"], d["threshold"])

        d["market_odds"]         = match[0] if match else None
        d["market_implied_prob"] = round(match[1], 4) if match else None
        d["edge_pp"] = (
            round((d["hit_probability"] - match[1]) * 100, 1)
            if match else None
        )
        d["has_market_line"] = match is not None
        comparisons.append(d)

    # Sort by edge desc (nulls last)
    comparisons.sort(key=lambda x: (x["edge_pp"] is None, -(x["edge_pp"] or 0)))

    edges = [c for c in comparisons if c["edge_pp"] is not None]

    return {
        "scout_date":    scout_date,
        "total":         len(comparisons),
        "matched":       len(edges),
        "avg_edge_pp":   round(sum(c["edge_pp"] for c in edges) / len(edges), 1) if edges else None,
        "comparisons":   comparisons,
    }


@app.get("/api/scout/settled")
def get_scout_settled(
    days:   int = 7,
    sport:  Optional[str] = None,
    grade:  Optional[str] = None,
    limit:  int = 100,
    db: Session = Depends(get_db),
):
    """
    Return recently settled scouted props (actual_hit IS NOT NULL).
    Shows projection performance: projected vs actual, hit/miss.
    """
    import json as _json
    from sqlalchemy import text
    from datetime import datetime, timezone, timedelta

    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    filters = ["actual_hit IS NOT NULL", "scout_date >= :since"]
    params: dict = {"since": since}
    if sport:
        filters.append("sport = :sport")
        params["sport"] = sport.upper()
    if grade:
        filters.append("quality_grade = :grade")
        params["grade"] = grade.upper()

    where = " AND ".join(filters)

    rows = db.execute(text(f"""
        SELECT id, scout_date, sport, home_team, away_team,
               market_type, player_name, team, side, threshold,
               projected_value, hit_probability, quality_grade,
               actual_hit, actual_outcome_value
        FROM   scouted_props
        WHERE  {where}
        ORDER  BY scout_date DESC, hit_probability DESC
        LIMIT  :limit
    """), {**params, "limit": limit}).fetchall()

    cols = [
        "id", "scout_date", "sport", "home_team", "away_team",
        "market_type", "player_name", "team", "side", "threshold",
        "projected_value", "hit_probability", "quality_grade",
        "actual_hit", "actual_outcome_value",
    ]

    props = [dict(zip(cols, r)) for r in rows]
    hits  = sum(1 for p in props if p.get("actual_hit") == 1)

    return {
        "days":      days,
        "since":     since,
        "total":     len(props),
        "hit_count": hits,
        "hit_rate":  round(hits / len(props), 4) if props else None,
        "props":     props,
    }


@app.get("/api/scout/summary")
def get_scout_summary(
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Grade/sport breakdown for scouted props on a date."""
    from sqlalchemy import text
    from datetime import datetime, timezone

    scout_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows = db.execute(text("""
        SELECT sport, quality_grade, COUNT(*) as n,
               AVG(hit_probability) as avg_prob
        FROM   scouted_props
        WHERE  scout_date = :d
        GROUP  BY sport, quality_grade
        ORDER  BY sport, quality_grade
    """), {"d": scout_date}).fetchall()

    breakdown = []
    for r in rows:
        breakdown.append({
            "sport":        r[0],
            "grade":        r[1],
            "count":        r[2],
            "avg_hit_prob": round(float(r[3]) * 100, 1) if r[3] else None,
        })

    total_row = db.execute(text(
        "SELECT COUNT(*) FROM scouted_props WHERE scout_date = :d"
    ), {"d": scout_date}).fetchone()

    last_scout = None
    try:
        import scheduler as sched
        last_scout = sched.get_state().get("last_scout_run")
    except Exception:
        pass

    return {
        "scout_date":  scout_date,
        "total_props": total_row[0] if total_row else 0,
        "breakdown":   breakdown,
        "last_scout_run": last_scout,
    }


@app.post("/api/scout/run")
def trigger_scout_run(db: Session = Depends(get_db)):
    """
    Manually trigger the full scout pipeline.
    Runs synchronously — may take 60-120 seconds.
    Returns summary with per-sport counts and grade breakdown.
    """
    from database import engine as _engine
    from sqlalchemy import text
    import scout.runner as _runner

    result = _runner.run_daily_scout(_engine, db)
    scout_date = result.get("scout_date")

    # Augment with grade breakdown
    if scout_date:
        try:
            grade_rows = db.execute(text("""
                SELECT quality_grade, COUNT(*) FROM scouted_props
                WHERE scout_date = :d GROUP BY quality_grade
            """), {"d": scout_date}).fetchall()
            result["by_grade"] = {r[0]: r[1] for r in grade_rows}
        except Exception:
            pass

    # Backfill scout grades on legs created before scout data existed
    try:
        from placement import backfill_scout_grades_for_today
        _bf = backfill_scout_grades_for_today()
        if _bf:
            result["scout_backfill_updated"] = _bf
            # Rescore composites now that scout grades are attached
            from placement import rescore_composites_for_today
            _rc = rescore_composites_for_today()
            result["composites_rescored"] = _rc.get("updated", 0)
    except Exception:
        pass

    # Update scheduler state
    try:
        import scheduler as _sched
        _sched.get_state()["last_scout_run"]        = scout_date
        _sched.get_state()["last_scout_run_result"] = result
    except Exception:
        pass

    return result


@app.get("/api/scout/calibration")
def get_scout_calibration(db: Session = Depends(get_db)):
    """Latest scout calibration drift summary."""
    from database import engine as _engine
    import scout.calibration as sc
    summary = sc.latest_scout_calibration_summary(_engine)
    return summary or {"message": "No calibration data yet"}


@app.post("/api/scout/calibration/run")
def run_scout_calibration(db: Session = Depends(get_db)):
    """Run scout calibration check now."""
    from database import engine as _engine
    import scout.calibration as sc
    sc.initialize(_engine)
    return sc.run_scout_calibration(_engine, db)


@app.get("/api/scout/accuracy")
def get_scout_accuracy(
    days: int = 30,
    sport: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Historical accuracy of scouted props — hit rate by grade, sport, and market.
    Only includes settled props (actual_hit IS NOT NULL).
    """
    from sqlalchemy import text
    from datetime import datetime, timezone, timedelta

    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    sport_filter = ""
    params: dict = {"since": since}
    if sport:
        sport_filter = "AND sport = :sport"
        params["sport"] = sport.upper()

    # By grade
    by_grade = db.execute(text(f"""
        SELECT quality_grade,
               COUNT(*) AS n,
               SUM(actual_hit) AS hits,
               AVG(CAST(actual_hit AS REAL)) AS hit_rate,
               AVG(hit_probability) AS expected_rate
        FROM   scouted_props
        WHERE  actual_hit IS NOT NULL
        AND    scout_date >= :since
        {sport_filter}
        GROUP  BY quality_grade
        ORDER  BY quality_grade
    """), params).fetchall()

    # By sport
    by_sport = db.execute(text(f"""
        SELECT sport,
               COUNT(*) AS n,
               SUM(actual_hit) AS hits,
               AVG(CAST(actual_hit AS REAL)) AS hit_rate,
               AVG(hit_probability) AS expected_rate
        FROM   scouted_props
        WHERE  actual_hit IS NOT NULL
        AND    scout_date >= :since
        {sport_filter}
        GROUP  BY sport
    """), params).fetchall()

    # By market
    by_market = db.execute(text(f"""
        SELECT market_type,
               COUNT(*) AS n,
               AVG(CAST(actual_hit AS REAL)) AS hit_rate,
               AVG(hit_probability) AS expected_rate
        FROM   scouted_props
        WHERE  actual_hit IS NOT NULL
        AND    scout_date >= :since
        {sport_filter}
        GROUP  BY market_type
        ORDER  BY n DESC
        LIMIT  15
    """), params).fetchall()

    # Total settled
    total_row = db.execute(text(f"""
        SELECT COUNT(*), SUM(actual_hit)
        FROM   scouted_props
        WHERE  actual_hit IS NOT NULL
        AND    scout_date >= :since
        {sport_filter}
    """), params).fetchone()

    def _fmt_rows(rows, cols):
        return [dict(zip(cols, r)) for r in rows]

    total_n    = total_row[0] if total_row else 0
    total_hits = total_row[1] if total_row else 0

    # Daily trend (last 30 days of settled props)
    by_day = db.execute(text(f"""
        SELECT scout_date,
               COUNT(*) AS n,
               SUM(actual_hit) AS hits,
               AVG(CAST(actual_hit AS REAL)) AS hit_rate
        FROM   scouted_props
        WHERE  actual_hit IS NOT NULL
        AND    scout_date >= :since
        {sport_filter}
        GROUP  BY scout_date
        ORDER  BY scout_date
    """), params).fetchall()

    return {
        "days":        days,
        "since":       since,
        "total_settled": total_n,
        "overall_hit_rate": round(total_hits / total_n, 4) if total_n else None,
        "by_grade":  _fmt_rows(by_grade,  ["grade", "n", "hits", "hit_rate", "expected_rate"]),
        "by_sport":  _fmt_rows(by_sport,  ["sport", "n", "hits", "hit_rate", "expected_rate"]),
        "by_market": _fmt_rows(by_market, ["market_type", "n", "hit_rate", "expected_rate"]),
        "by_day":    _fmt_rows(by_day,    ["date", "n", "hits", "hit_rate"]),
    }


# ─── Placement routes ─────────────────────────────────────────────────────────

import placement as _placement

@app.get("/api/scout/export")
def export_scout_props(
    date:   Optional[str] = None,
    sport:  Optional[str] = None,
    grade:  Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Export today's (or a given date's) scouted props as a CSV download.
    Filterable by sport and quality_grade.
    """
    import csv, io
    from fastapi.responses import StreamingResponse
    from sqlalchemy import text
    from datetime import datetime, timezone

    scout_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filters = ["scout_date = :date"]
    params: dict = {"date": scout_date}
    if sport:
        filters.append("sport = :sport")
        params["sport"] = sport.upper()
    if grade:
        filters.append("quality_grade = :grade")
        params["grade"] = grade.upper()

    rows = db.execute(text(f"""
        SELECT scout_date, sport, home_team, away_team, commence_time,
               market_type, player_name, team, side, threshold,
               projected_value, projected_low_95, projected_high_95,
               projected_std_dev, hit_probability, quality_grade,
               data_source, actual_hit, actual_outcome_value
        FROM   scouted_props
        WHERE  {" AND ".join(filters)}
        ORDER  BY quality_grade, hit_probability DESC
        LIMIT  500
    """), params).fetchall()

    cols = [
        "scout_date", "sport", "home_team", "away_team", "commence_time",
        "market_type", "player_name", "team", "side", "threshold",
        "projected_value", "projected_low_95", "projected_high_95",
        "projected_std_dev", "hit_probability", "quality_grade",
        "data_source", "actual_hit", "actual_outcome_value",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(zip(cols, row)))
    buf.seek(0)

    fname = f"betiq_scout_{scout_date}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


class PlacementSizeRequest(BaseModel):
    hit_probability: float
    quality_grade:   str
    decimal_odds:    float
    bankroll:        Optional[float] = None

@app.get("/api/placement/recommendations")
def placement_recommendations(
    sport:     Optional[str] = None,
    min_grade: str = "B",
    limit:     int = 30,
    db: Session = Depends(get_db),
):
    """
    Grade-weighted placement recommendations for today's scouted props.
    Includes Kelly sizing, action (PLAY/SKIP), and multiplier.
    """
    sport_filter = [sport.upper()] if sport else None
    return _placement.get_todays_recommendations(
        db,
        sport_filter = sport_filter,
        min_grade    = min_grade,
        limit        = limit,
    )


@app.post("/api/placement/size")
def placement_size(req: PlacementSizeRequest, db: Session = Depends(get_db)):
    """Kelly sizing for a single scouted prop given grade + odds."""
    return _placement.size_single_bet(
        hit_probability = req.hit_probability,
        quality_grade   = req.quality_grade,
        decimal_odds    = req.decimal_odds,
        bankroll        = req.bankroll,
    )


@app.get("/api/placement/parlay-suggestions")
def placement_parlay_suggestions(
    max_legs: int = 3,
    db: Session = Depends(get_db),
):
    """Top A-grade props as a suggested parlay, with combined hit probability."""
    return _placement.get_parlay_suggestions(db, max_legs=max_legs)


# ─── Validation Tracking ──────────────────────────────────────────────────────

@app.get("/api/validation/summary")
def validation_summary():
    """Latest persisted snapshot from all four validation layers (no re-run)."""
    import validation_tracker as _vt
    return _vt.get_latest_validation_summary()


@app.post("/api/validation/run")
def validation_run(window_days: int = Query(30, ge=7, le=90)):
    """Manually trigger all four validation layers and return fresh results."""
    import validation_tracker as _vt
    return _vt.run_all_validations(window_days=window_days)


@app.get("/api/validation/scout")
def validation_scout(window_days: int = Query(30, ge=7, le=90)):
    """Scout prediction accuracy (per-grade WR and Brier score)."""
    import validation_tracker as _vt
    return _vt.validate_scout_predictions(window_days=window_days)


@app.get("/api/validation/projections")
def validation_projections(game_date: Optional[str] = Query(None)):
    """
    Player projection accuracy vs ESPN box scores.
    game_date format: YYYY-MM-DD. Defaults to yesterday + today.
    """
    import validation_tracker as _vt
    return _vt.validate_player_projections(game_date=game_date)


@app.get("/api/validation/placement")
def validation_placement(
    since: Optional[str] = Query(None, description="Start date YYYY-MM-DD (default: 2026-05-16, post-calibration-fix)"),
    window_days: Optional[int] = Query(None, ge=1, le=365, description="Alternative: last N days"),
    bet_category: Optional[str] = Query(None, description="Filter: 'team' | 'player_prop' | omit for all"),
):
    """
    ROI by source (placement rec_type proxy).
    Default cutoff is 2026-05-16 (post win_prob scale + _ALT_MARKETS calibration fix).
    Pass since=2026-01-01 to see pre-fix data for comparison.
    bet_category=team shows only team bets; bet_category=player_prop shows only prop bets.
    """
    import validation_tracker as _vt
    return _vt.validate_placement(since=since, window_days=window_days, bet_category=bet_category)


@app.get("/api/validation/signal-agreement")
def validation_signal_agreement(
    window_days: int = Query(30, ge=7, le=90),
    bet_category: Optional[str] = Query(None, description="Filter: 'team' | 'player_prop' | omit for all"),
):
    """Win rate by number of aligned prediction signals (0–3). Filter by bet_category for split view."""
    import validation_tracker as _vt
    return _vt.compute_signal_agreement(window_days=window_days, bet_category=bet_category)


@app.get("/api/validation/scout-trajectory")
def validation_scout_trajectory(lookback_runs: int = Query(10, ge=2, le=50)):
    """Per-grade WR and Brier trend across the last N validation runs, vs targets."""
    import validation_tracker as _vt
    return _vt.get_scout_trajectory(lookback_runs=lookback_runs)


@app.get("/api/validation/grade-drivers")
def validation_grade_drivers(
    grade: str = Query("B", description="Scout grade to analyse: A, B, C, or D"),
    since: Optional[str] = Query(None, description="Start date YYYY-MM-DD (default: 2026-05-10)"),
):
    """
    Per sport/market WR breakdown for a scout grade. Flags cells that
    outperform or underperform the grade average by ≥10 points.
    """
    import validation_tracker as _vt
    return _vt.analyze_grade_drivers(grade=grade, since=since)


@app.get("/api/validation/scout-multipliers")
def validation_scout_multipliers(db: Session = Depends(get_db)):
    """
    Current dynamic scout multipliers used in composite scoring, plus the
    empirical data they're derived from.
    """
    import placement as _plc
    dynamic = _plc.get_dynamic_scout_multipliers(db)
    return {
        "dynamic": dynamic,
        "static_fallback": _plc._SCOUT_GRADE_MULTIPLIERS,
        "using_dynamic": dynamic != _plc._SCOUT_GRADE_MULTIPLIERS,
        "cache_ttl_seconds": _plc._DYNAMIC_SCOUT_MULT_TTL,
        "min_n_for_trust": _plc._SCOUT_GRADE_MULT_MIN_N,
    }


@app.post("/api/fanduel/parse-screenshot")
async def fanduel_parse_screenshot(
    image: UploadFile = File(...),
    game_date: Optional[str] = Form(default=None),
):
    """
    Parse a single FanDuel bet slip screenshot.
    Accepts multipart/form-data with:
      image     — image file (PNG/JPG)
      game_date — YYYY-MM-DD of the games (optional; defaults to today CT)

    Returns parsed pick dict plus EXIF timestamp info:
      bet_date, bet_time, timestamp_source ('exif'|'current'), timestamp_display
    """
    from zoneinfo import ZoneInfo as _ZI
    _ct_tz = _ZI("America/Chicago")
    now_ct = datetime.now(_ct_tz)

    gdate = game_date or now_ct.strftime("%Y-%m-%d")

    raw_bytes = await image.read()
    mime = image.content_type or "image/png"

    # Save to temp file for EXIF extraction
    tmp_path = f"/tmp/fd_screenshot_{uuid.uuid4()}.img"
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)

        # EXIF timestamp extraction
        exif_dt = _extract_screenshot_timestamp(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if exif_dt is not None:
        timestamp_source = "exif"
        bet_dt = exif_dt
        timestamp_display = bet_dt.strftime("%-m/%-d/%Y") + " at " + bet_dt.strftime("%-I:%M %p") + " CT (from EXIF)"
    else:
        timestamp_source = "current"
        bet_dt = now_ct.replace(tzinfo=None)
        timestamp_display = "No EXIF timestamp — using current time: " + now_ct.strftime("%-m/%-d/%Y") + " at " + now_ct.strftime("%-I:%M %p") + " CT"

    bet_date = bet_dt.strftime("%Y-%m-%d")
    bet_time = bet_dt.strftime("%H:%M:%S")
    # ISO string for datetime-local input (no timezone suffix)
    timestamp_iso = bet_dt.strftime("%Y-%m-%dT%H:%M")

    # ── Parse: free OCR path first, Claude vision as optional upgrade ────────────
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

    # Always run the free OCR parser — no key needed
    result = fp.parse_screenshot_free(raw_bytes, game_date=gdate)

    # If OCR found no legs AND the Anthropic key is available, upgrade to Claude
    if not result.get("parsed_legs") and has_api_key:
        claude_result = fp.parse_screenshot(raw_bytes, mime_type=mime, game_date=gdate)
        if "error" not in claude_result:
            claude_result["parse_method"] = "anthropic_vision"
            result = claude_result
        # else: stick with the empty OCR result

    # If OCR found no legs AND no API key, surface a clear warning (not an error)
    if not result.get("parsed_legs") and not has_api_key:
        result["warnings"] = result.get("warnings", []) + [
            "OCR could not extract bet details from this image. "
            "Review the raw_text field and enter legs manually. "
            "Set ANTHROPIC_API_KEY for automatic vision parsing."
        ]

    result["bet_date"] = bet_date
    result["bet_time"] = bet_time
    result["timestamp_source"] = timestamp_source
    result["timestamp_display"] = timestamp_display
    result["timestamp_iso"] = timestamp_iso

    return result


# ─── NegBin Shadow Tracker ────────────────────────────────────────────────────

_NEGBIN_ODDS_BANDS = [
    ("1.3-1.7x", 1.3, 1.7),
    ("1.7-2.2x", 1.7, 2.2),
    ("2.2-2.8x", 2.2, 2.8),
    ("2.8-3.5x", 2.8, 3.5),
    ("3.5-5.0x", 3.5, 5.0),
]

_NEGBIN_CELL_ADJ = {
    "fav_away": 1.10,
    "dog_home": 0.70,
    "fav_home": 1.00,
    "dog_away": 1.00,
}

_NEGBIN_CELL_LABELS = {
    "fav_away": "Fav @ Away",
    "dog_home": "Dog @ Home",
    "fav_home": "Fav @ Home",
    "dog_away": "Dog @ Away",
}

_NEGBIN_STAKE = 10.0


def _negbin_db_connect():
    import sqlite3 as _sq3
    from database import DB_PATH
    con = _sq3.connect(DB_PATH, timeout=15)
    con.row_factory = _sq3.Row
    return con


def _negbin_is_won(leg_results_str: str | None) -> bool | None:
    """Return True if all legs are WIN, False if any leg is LOSS, None if pending."""
    if not leg_results_str:
        return None
    results = [r.strip() for r in leg_results_str.split(",") if r.strip()]
    if not results:
        return None
    if any(r == "LOSS" for r in results):
        return False
    if all(r == "WIN" for r in results):
        return True
    return None  # pending / mixed


def _negbin_parlay_status(leg_results_str: str | None) -> str:
    won = _negbin_is_won(leg_results_str)
    if won is True:
        return "WON"
    if won is False:
        return "LOST"
    return "PENDING"


def _negbin_pnl(won: bool | None, odds: float) -> float | None:
    if won is True:
        return round((_NEGBIN_STAKE * odds) - _NEGBIN_STAKE, 4)
    if won is False:
        return round(-_NEGBIN_STAKE, 4)
    return None


def _negbin_band_for_odds(odds: float) -> str | None:
    for label, lo, hi in _NEGBIN_ODDS_BANDS:
        if lo <= odds < hi:
            return label
    return None


def _negbin_quality(eff_n: float) -> str:
    if eff_n >= 30:
        return "RELIABLE"
    if eff_n >= 10:
        return "DIRECTIONAL"
    return "TOO THIN"


def _negbin_cohort_filter(cohort: str) -> str:
    if cohort == "clean":
        return "AND mb.cohort = 'clean'"
    if cohort == "prospective_legacy":
        return "AND mb.cohort = 'prospective_legacy'"
    return ""  # "all" — no filter


def _negbin_view_filter(view: str) -> str:
    """Filter for measurement (all) vs live-realistic (only live_realistic=1) view."""
    if view == "live_realistic":
        return "AND mb.live_realistic = 1"
    return ""  # "measurement" — no filter, show all


_VALID_SHADOW_SOURCES = {
    "shadow_negbin", "shadow_negbin_recal",
    "shadow_negbin_edgegate", "shadow_negbin_fullstack",
    "shadow_negbin_composite", "shadow_negbin_atscap",
    "shadow_negbin_distwp_gate", "shadow_negbin_bothgates",
}


def _negbin_source_filter(source: str) -> str:
    """Filter by shadow source for A/B comparison or specific arm detail."""
    if source == "shadow_a":
        return "AND mb.source = 'shadow_negbin'"
    if source == "shadow_b":
        return "AND mb.source = 'shadow_negbin_recal'"
    if source == "all":
        return "AND mb.source IN ('shadow_negbin', 'shadow_negbin_recal')"
    # Direct source tag — validate to prevent injection
    if source in _VALID_SHADOW_SOURCES:
        return f"AND mb.source = '{source}'"
    # Fallback: A only
    return "AND mb.source = 'shadow_negbin'"


@app.get("/api/negbin/tracker")
def negbin_tracker(
    days: int = Query(30, ge=1, le=365),
    cohort: str = Query("clean"),
    view: str = Query("measurement"),
    source: str = Query("shadow_a"),
):
    """
    Comprehensive NegBin shadow pipeline tracker.
    Returns band-for-band stats, 2x2 cell performance, parlay structure,
    daily stats, per-parlay drill-down, and overall status.

    source: 'shadow_a' (uncalibrated control), 'shadow_b' (recalibrated),
            'all' (both combined).
    """
    import sqlite3 as _sq3
    from collections import defaultdict as _dd

    cohort_filter = _negbin_cohort_filter(cohort)
    view_filter = _negbin_view_filter(view)
    source_filter = _negbin_source_filter(source)

    con = _negbin_db_connect()
    cur = con.cursor()

    # ── 1. Fetch all NegBin shadow bets in the window ────────────────────────
    nb_sql = f"""
        SELECT mb.id, mb.odds, mb.negbin_odds_band, mb.status, mb.game_date,
               mb.legs, mb.negbin_leg_count, mb.predicted_win_prob,
               mb.composite_score, mb.negbin_cell_adj, mb.cohort,
               mb.live_realistic,
               GROUP_CONCAT(mbl.leg_result, ',')     AS leg_results,
               GROUP_CONCAT(mbl.fixture_id, ',')     AS fixture_ids,
               GROUP_CONCAT(mbl.description, '||')   AS leg_descs,
               GROUP_CONCAT(mbl.negbin_cell, ',')    AS leg_cells,
               GROUP_CONCAT(mbl.negbin_wp, ',')      AS leg_negbin_wps,
               GROUP_CONCAT(mbl.negbin_mu_home, ',') AS leg_mu_homes,
               GROUP_CONCAT(mbl.negbin_mu_away, ',') AS leg_mu_aways,
               GROUP_CONCAT(mbl.negbin_edge_pp, ',') AS leg_edge_pps,
               GROUP_CONCAT(mbl.predicted_win_prob, ',') AS leg_pred_wps,
               GROUP_CONCAT(mbl.market_type, ',')    AS leg_market_types,
               GROUP_CONCAT(mbl.sport, ',')          AS leg_sports
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.game_date >= date('now', '-' || ? || ' days')
          {source_filter}
          {cohort_filter}
          {view_filter}
        GROUP BY mb.id
        ORDER BY mb.game_date DESC, mb.id
    """
    cur.execute(nb_sql, (days,))
    nb_rows = [dict(r) for r in cur.fetchall()]

    # ── 2. Fetch ATS comparison bets (same window, MLB only) ─────────────────
    ats_sql = """
        SELECT mb.id, mb.odds, mb.negbin_odds_band, mb.status, mb.game_date,
               mb.legs,
               GROUP_CONCAT(mbl.leg_result, ',') AS leg_results
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.source IN ('prospective', 'prospective_pm')
          AND mb.sport = 'MLB'
          AND mb.game_date >= date('now', '-' || ? || ' days')
        GROUP BY mb.id
    """
    cur.execute(ats_sql, (days,))
    ats_rows = [dict(r) for r in cur.fetchall()]

    # ── 3. Fetch full leg details for per-parlay list ─────────────────────────
    leg_detail_sql = f"""
        SELECT mbl.mock_bet_id, mbl.description, mbl.negbin_wp, mbl.negbin_mu_home,
               mbl.negbin_mu_away, mbl.negbin_cell, mbl.negbin_edge_pp,
               mbl.predicted_win_prob, mbl.leg_result, mbl.market_type, mbl.sport,
               mbl.fixture_id
        FROM mock_bet_legs mbl
        JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
        WHERE mb.game_date >= date('now', '-' || ? || ' days')
          {source_filter}
          {cohort_filter}
          {view_filter}
        ORDER BY mbl.id
    """
    cur.execute(leg_detail_sql, (days,))
    all_leg_details = [dict(r) for r in cur.fetchall()]
    con.close()

    # Index leg details by parlay id
    legs_by_parlay: dict = _dd(list)
    for leg in all_leg_details:
        legs_by_parlay[leg["mock_bet_id"]].append(leg)

    # ── 4. Band-for-band analysis ─────────────────────────────────────────────
    # Group NegBin bets by band
    nb_by_band: dict = _dd(list)
    for row in nb_rows:
        band = row.get("negbin_odds_band") or _negbin_band_for_odds(row.get("odds") or 0.0)
        if band:
            nb_by_band[band].append(row)

    # Group ATS bets by band (use stored band if available, else derive from odds)
    ats_by_band: dict = _dd(list)
    for row in ats_rows:
        band = row.get("negbin_odds_band") or _negbin_band_for_odds(row.get("odds") or 0.0)
        if band:
            ats_by_band[band].append(row)

    def _band_stats(rows: list) -> dict:
        """Compute win/pnl stats for a list of parlay rows."""
        n = len(rows)
        wins = 0
        total_pnl = 0.0
        total_odds = 0.0
        for r in rows:
            won = _negbin_is_won(r.get("leg_results"))
            pnl = _negbin_pnl(won, r.get("odds") or 1.0)
            if pnl is not None:
                total_pnl += pnl
            if won is True:
                wins += 1
            total_odds += r.get("odds") or 0.0
        wr = round(wins / n * 100, 2) if n else None
        roi = round(total_pnl / (n * _NEGBIN_STAKE) * 100, 2) if n else None
        avg_odds = round(total_odds / n, 4) if n else None
        breakeven = round(1 / avg_odds * 100, 2) if avg_odds else None
        edge = round(wr - breakeven, 2) if (wr is not None and breakeven is not None) else None
        return {
            "n": n, "wins": wins, "wr": wr, "pnl": round(total_pnl, 4),
            "roi": roi, "avg_odds": avg_odds, "breakeven": breakeven, "edge": edge,
        }

    def _leg_reuse(rows: list, game_date: str | None = None) -> tuple:
        """Return (raw_n, avg_reuse, eff_n) for a list of parlay rows."""
        # Track leg appearances using date:description as key
        leg_appearances: dict = _dd(int)
        for r in rows:
            gd = r.get("game_date", "")
            descs = (r.get("leg_descs") or "").split("||")
            for d in descs:
                d = d.strip()
                if d:
                    leg_appearances[f"{gd}:{d}"] += 1
        raw_n = len(rows)
        distinct = len(leg_appearances)
        total_appearances = sum(leg_appearances.values())
        avg_reuse = round(total_appearances / distinct, 4) if distinct else 1.0
        eff_n = round(raw_n / avg_reuse, 2) if avg_reuse else float(raw_n)
        return raw_n, avg_reuse, eff_n

    def _by_legs_split(rows: list) -> list:
        """Split parlay rows by leg count and compute stats per count."""
        by_legs: dict = _dd(list)
        for r in rows:
            lc = r.get("negbin_leg_count") or r.get("legs") or 0
            by_legs[int(lc)].append(r)
        result = []
        for lc in sorted(by_legs.keys()):
            rs = by_legs[lc]
            s = _band_stats(rs)
            result.append({
                "legs": lc,
                "n": s["n"], "wins": s["wins"], "wr": s["wr"],
                "edge": s["edge"],
                "comparable": s["n"] >= 5,
            })
        return result

    bands_out = []
    nb_total_n = nb_total_wins = 0
    nb_total_pnl = 0.0
    ats_total_n = ats_total_wins = 0
    ats_total_pnl = 0.0

    for band_label, lo, hi in _NEGBIN_ODDS_BANDS:
        nb_rows_band = nb_by_band.get(band_label, [])
        ats_rows_band = ats_by_band.get(band_label, [])

        nb_s = _band_stats(nb_rows_band)
        ats_s = _band_stats(ats_rows_band)

        raw_n, avg_reuse, eff_n = _leg_reuse(nb_rows_band)
        quality = _negbin_quality(eff_n)

        nb_vs_ats_edge = None
        if nb_s["edge"] is not None and ats_s["edge"] is not None:
            nb_vs_ats_edge = round(nb_s["edge"] - ats_s["edge"], 2)

        # Underfill diagnostic
        underfill_reason = None
        if nb_s["n"] < 10 and nb_s["n"] > 0:
            underfill_reason = "LEG_CONSUMPTION"
        elif nb_s["n"] == 0:
            underfill_reason = "NO_CONSTRUCTABLE"

        nb_total_n += nb_s["n"]
        nb_total_wins += nb_s["wins"]
        nb_total_pnl += nb_s["pnl"]
        ats_total_n += ats_s["n"]
        ats_total_wins += ats_s["wins"]
        ats_total_pnl += ats_s["pnl"]

        bands_out.append({
            "band": band_label, "lo": lo, "hi": hi,
            # NegBin shadow
            "nb_n": nb_s["n"],
            "nb_wins": nb_s["wins"],
            "nb_wr": nb_s["wr"],
            "nb_pnl": nb_s["pnl"],
            "nb_roi": nb_s["roi"],
            "nb_avg_odds": nb_s["avg_odds"],
            "nb_breakeven": nb_s["breakeven"],
            "nb_edge": nb_s["edge"],
            # Effective N
            "nb_raw_n": raw_n,
            "nb_avg_reuse": avg_reuse,
            "nb_eff_n": eff_n,
            "nb_quality": quality,
            # ATS baseline
            "ats_n": ats_s["n"],
            "ats_wins": ats_s["wins"],
            "ats_wr": ats_s["wr"],
            "ats_pnl": ats_s["pnl"],
            "ats_roi": ats_s["roi"],
            "ats_edge": ats_s["edge"],
            # Delta
            "nb_vs_ats_edge": nb_vs_ats_edge,
            # Leg count split
            "by_legs": _by_legs_split(nb_rows_band),
            # Underfill diagnostic
            "underfill_reason": underfill_reason,
        })

    # Band totals
    nb_total_wr = round(nb_total_wins / nb_total_n * 100, 2) if nb_total_n else None
    nb_total_roi = round(nb_total_pnl / (nb_total_n * _NEGBIN_STAKE) * 100, 2) if nb_total_n else None
    ats_total_wr = round(ats_total_wins / ats_total_n * 100, 2) if ats_total_n else None
    ats_total_roi = round(ats_total_pnl / (ats_total_n * _NEGBIN_STAKE) * 100, 2) if ats_total_n else None

    # ── 5. 2×2 Cell performance ───────────────────────────────────────────────
    cell_stats: dict = _dd(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "odds_sum": 0.0})
    for r in nb_rows:
        won = _negbin_is_won(r.get("leg_results"))
        pnl = _negbin_pnl(won, r.get("odds") or 1.0)
        # Determine dominant cell from leg cells
        cells_str = r.get("leg_cells") or ""
        leg_cells = [c.strip() for c in cells_str.split(",") if c.strip()]
        if not leg_cells:
            continue
        # Use most common cell as the parlay's cell
        from collections import Counter as _Counter
        cell = _Counter(leg_cells).most_common(1)[0][0]
        cell_stats[cell]["n"] += 1
        if won is True:
            cell_stats[cell]["wins"] += 1
        if pnl is not None:
            cell_stats[cell]["pnl"] += pnl
        cell_stats[cell]["odds_sum"] += r.get("odds") or 0.0

    cells_out = []
    for cell_key in ["fav_away", "dog_home", "fav_home", "dog_away"]:
        cs = cell_stats[cell_key]
        n = cs["n"]
        wins = cs["wins"]
        pnl = round(cs["pnl"], 4)
        wr = round(wins / n * 100, 2) if n else None
        roi = round(pnl / (n * _NEGBIN_STAKE) * 100, 2) if n else None
        avg_odds = round(cs["odds_sum"] / n, 4) if n else None
        breakeven = round(1 / avg_odds * 100, 2) if avg_odds else None
        edge = round(wr - breakeven, 2) if (wr is not None and breakeven is not None) else None
        adj = _NEGBIN_CELL_ADJ.get(cell_key, 1.0)
        adj_pct = int(round((adj - 1.0) * 100))
        adj_label = f"+{adj_pct}%" if adj_pct >= 0 else f"{adj_pct}%"
        cells_out.append({
            "cell": cell_key,
            "label": _NEGBIN_CELL_LABELS.get(cell_key, cell_key),
            "n": n, "wins": wins, "wr": wr, "roi": roi, "edge": edge,
            "cell_adj": adj,
            "adj_label": adj_label,
        })

    # ── 6. Parlay structure ───────────────────────────────────────────────────
    leg_count_dist: dict = _dd(int)
    for r in nb_rows:
        lc = int(r.get("negbin_leg_count") or r.get("legs") or 0)
        leg_count_dist[lc] += 1

    total_parlays = len(nb_rows)
    leg_distribution = []
    for lc in sorted(leg_count_dist.keys()):
        cnt = leg_count_dist[lc]
        pct = round(cnt / total_parlays * 100, 1) if total_parlays else 0.0
        leg_distribution.append({
            "legs": lc,
            "n": cnt,
            "pct": pct,
            "validated": cnt >= 5,
        })

    # Reuse histogram: count how many distinct legs appear N times
    all_leg_app: dict = _dd(int)
    for r in nb_rows:
        gd = r.get("game_date", "")
        descs = (r.get("leg_descs") or "").split("||")
        for d in descs:
            d = d.strip()
            if d:
                all_leg_app[f"{gd}:{d}"] += 1
    reuse_histogram: dict = _dd(int)
    for key, cnt in all_leg_app.items():
        reuse_histogram[cnt] += 1
    # Sort by appearance count
    reuse_histogram_out = {str(k): v for k, v in sorted(reuse_histogram.items())}

    # Parlays at max reuse cap (appear in 3+ parlays)
    at_cap = sum(1 for cnt in all_leg_app.values() if cnt >= 3)
    total_distinct_legs = len(all_leg_app)

    # Underfill by band
    underfill_by_band = []
    for band_obj in bands_out:
        if band_obj["nb_n"] < 10:
            underfill_by_band.append({
                "band": band_obj["band"],
                "target": 10,
                "filled": band_obj["nb_n"],
                "reason": band_obj["underfill_reason"],
                "available_combos": None,  # runtime computation not available here
            })

    structure_out = {
        "leg_distribution": leg_distribution,
        "reuse_histogram": reuse_histogram_out,
        "at_cap": at_cap,
        "total_distinct_legs": total_distinct_legs,
        "underfill_by_band": underfill_by_band,
    }

    # ── 7. Daily stats ────────────────────────────────────────────────────────
    daily_map: dict = _dd(lambda: {"generated": 0, "settled": 0, "pending": 0,
                                    "pnl": 0.0, "wins": 0, "n_settled": 0,
                                    "covered": 0, "total": 0})
    for r in nb_rows:
        gd = r.get("game_date", "unknown")
        won = _negbin_is_won(r.get("leg_results"))
        pnl = _negbin_pnl(won, r.get("odds") or 1.0)
        daily_map[gd]["generated"] += 1
        daily_map[gd]["total"] += 1
        if won is not None:
            daily_map[gd]["settled"] += 1
            daily_map[gd]["n_settled"] += 1
            if pnl is not None:
                daily_map[gd]["pnl"] += pnl
            if won is True:
                daily_map[gd]["wins"] += 1
        else:
            daily_map[gd]["pending"] += 1
        # Coverage: does the parlay have all fixture_ids populated?
        fids = r.get("fixture_ids") or ""
        fid_list = [f for f in fids.split(",") if f.strip()]
        leg_cnt = int(r.get("negbin_leg_count") or r.get("legs") or 1)
        if len(fid_list) >= leg_cnt:
            daily_map[gd]["covered"] += 1

    daily_out = []
    for gd in sorted(daily_map.keys(), reverse=True):
        dm = daily_map[gd]
        ns = dm["n_settled"]
        wr = round(dm["wins"] / ns * 100, 2) if ns else None
        cov = round(dm["covered"] / dm["total"] * 100, 1) if dm["total"] else 0.0
        daily_out.append({
            "date": gd,
            "generated": dm["generated"],
            "settled": dm["settled"],
            "pending": dm["pending"],
            "pnl": round(dm["pnl"], 4),
            "wr": wr,
            "coverage_pct": cov,
        })

    # ── 8. Per-parlay list ────────────────────────────────────────────────────
    parlays_out = []
    for r in nb_rows:
        won = _negbin_is_won(r.get("leg_results"))
        pnl = _negbin_pnl(won, r.get("odds") or 1.0)
        status = _negbin_parlay_status(r.get("leg_results"))

        # Build leg list from detailed query
        leg_details = legs_by_parlay.get(r["id"], [])
        legs_out = []
        for leg in leg_details:
            legs_out.append({
                "description": leg.get("description"),
                "negbin_wp": leg.get("negbin_wp"),
                "negbin_mu_home": leg.get("negbin_mu_home"),
                "negbin_mu_away": leg.get("negbin_mu_away"),
                "negbin_cell": leg.get("negbin_cell"),
                "negbin_edge_pp": leg.get("negbin_edge_pp"),
                "predicted_wp": leg.get("predicted_win_prob"),
                "leg_result": leg.get("leg_result"),
                "market_type": leg.get("market_type"),
                "sport": leg.get("sport"),
                "fixture_id": leg.get("fixture_id"),
            })

        parlays_out.append({
            "id": r["id"],
            "game_date": r.get("game_date"),
            "odds": r.get("odds"),
            "band": r.get("negbin_odds_band") or _negbin_band_for_odds(r.get("odds") or 0.0),
            "status": status,
            "cohort": r.get("cohort"),
            "win_prob": r.get("predicted_win_prob"),
            "composite": r.get("composite_score"),
            "leg_count": int(r.get("negbin_leg_count") or r.get("legs") or 0),
            "cell_adj": r.get("negbin_cell_adj"),
            "pnl": pnl,
            "legs": legs_out,
        })

    # ── 9. Overall status ─────────────────────────────────────────────────────
    settled_count = sum(1 for r in nb_rows if _negbin_is_won(r.get("leg_results")) is not None)
    pending_count = len(nb_rows) - settled_count
    status_label = "EVALUABLE" if settled_count >= 50 else "COLLECTING"

    # First clean date
    clean_rows = [r for r in nb_rows if r.get("cohort") == "clean"]
    first_clean_date = None
    days_since_start = None
    if clean_rows:
        dates = [r.get("game_date") for r in clean_rows if r.get("game_date")]
        if dates:
            first_clean_date = min(dates)
            try:
                from datetime import date as _date, timedelta as _td
                fd = _date.fromisoformat(first_clean_date)
                days_since_start = (_date.today() - fd).days
            except Exception:
                pass

    # ── Fav@Home carve-out stats ─────────────────────────────────────────────
    fh_out = None
    try:
        con2 = _negbin_db_connect()
        cur2 = con2.cursor()
        fh_sql = f"""
            SELECT mb.id, mb.odds, mb.status, mb.negbin_odds_band, mb.game_date,
                   GROUP_CONCAT(mbl.leg_result, ',') AS leg_results,
                   GROUP_CONCAT(mbl.negbin_cell, ',') AS leg_cells,
                   GROUP_CONCAT(mbl.description, '||') AS leg_descs
            FROM mock_bets mb
            JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
            WHERE mb.source IN ('shadow_negbin_fh'{", 'shadow_negbin_recal_fh'" if source != "shadow_a" else ""})
              AND mb.game_date >= date('now', '-' || ? || ' days')
            GROUP BY mb.id
        """
        cur2.execute(fh_sql, (days,))
        fh_rows = [dict(r) for r in cur2.fetchall()]
        con2.close()

        if fh_rows:
            fh_n = len(fh_rows)
            fh_wins = sum(1 for r in fh_rows if _negbin_is_won(r.get("leg_results")) is True)
            fh_settled = sum(1 for r in fh_rows if _negbin_is_won(r.get("leg_results")) is not None)
            fh_pending = fh_n - fh_settled
            fh_pnl = sum(_negbin_pnl(_negbin_is_won(r.get("leg_results")), r.get("odds") or 1.0) or 0
                         for r in fh_rows)
            fh_wr = round(fh_wins / fh_settled * 100, 2) if fh_settled else None
            fh_roi = round(fh_pnl / (fh_settled * _NEGBIN_STAKE) * 100, 2) if fh_settled else None
            fh_out = {
                "n": fh_n, "settled": fh_settled, "pending": fh_pending,
                "wins": fh_wins, "wr": fh_wr, "pnl": round(fh_pnl, 4), "roi": fh_roi,
            }
    except Exception:
        pass

    return {
        "filters": {"days": days, "cohort": cohort, "view": view, "source": source},
        "bands": bands_out,
        "nb_total": {
            "n": nb_total_n, "wins": nb_total_wins,
            "wr": nb_total_wr,
            "pnl": round(nb_total_pnl, 4),
            "roi": nb_total_roi,
        },
        "ats_total": {
            "n": ats_total_n, "wins": ats_total_wins,
            "wr": ats_total_wr,
            "pnl": round(ats_total_pnl, 4),
            "roi": ats_total_roi,
        },
        "cells": cells_out,
        "structure": structure_out,
        "daily": daily_out,
        "parlays": parlays_out,
        "fav_home_carveout": fh_out,
        "status": status_label,
        "settled_count": settled_count,
        "pending_count": pending_count,
        "first_clean_date": first_clean_date,
        "days_since_start": days_since_start,
    }


@app.get("/api/negbin/parlay/{parlay_id}")
def negbin_parlay_detail(parlay_id: str):
    """
    Full details for a single NegBin shadow parlay.
    Returns the same shape as a parlays[] item in /api/negbin/tracker.
    """
    import sqlite3 as _sq3

    con = _negbin_db_connect()
    cur = con.cursor()

    # Fetch the parlay header
    cur.execute("""
        SELECT mb.id, mb.odds, mb.negbin_odds_band, mb.status, mb.game_date,
               mb.legs, mb.negbin_leg_count, mb.predicted_win_prob,
               mb.composite_score, mb.negbin_cell_adj, mb.cohort,
               GROUP_CONCAT(mbl.leg_result, ',') AS leg_results
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.id = ?
          AND mb.source IN ('shadow_negbin', 'shadow_negbin_recal')
        GROUP BY mb.id
    """, (parlay_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise HTTPException(status_code=404, detail=f"NegBin parlay '{parlay_id}' not found")

    r = dict(row)

    # Fetch full leg details
    cur.execute("""
        SELECT description, negbin_wp, negbin_mu_home, negbin_mu_away,
               negbin_cell, negbin_edge_pp, predicted_win_prob,
               leg_result, market_type, sport, fixture_id
        FROM mock_bet_legs
        WHERE mock_bet_id = ?
        ORDER BY id
    """, (parlay_id,))
    leg_rows = [dict(lg) for lg in cur.fetchall()]
    con.close()

    won = _negbin_is_won(r.get("leg_results"))
    pnl = _negbin_pnl(won, r.get("odds") or 1.0)
    status = _negbin_parlay_status(r.get("leg_results"))

    legs_out = [
        {
            "description": lg.get("description"),
            "negbin_wp": lg.get("negbin_wp"),
            "negbin_mu_home": lg.get("negbin_mu_home"),
            "negbin_mu_away": lg.get("negbin_mu_away"),
            "negbin_cell": lg.get("negbin_cell"),
            "negbin_edge_pp": lg.get("negbin_edge_pp"),
            "predicted_wp": lg.get("predicted_win_prob"),
            "leg_result": lg.get("leg_result"),
            "market_type": lg.get("market_type"),
            "sport": lg.get("sport"),
            "fixture_id": lg.get("fixture_id"),
        }
        for lg in leg_rows
    ]

    return {
        "id": r["id"],
        "game_date": r.get("game_date"),
        "odds": r.get("odds"),
        "band": r.get("negbin_odds_band") or _negbin_band_for_odds(r.get("odds") or 0.0),
        "status": status,
        "cohort": r.get("cohort"),
        "win_prob": r.get("predicted_win_prob"),
        "composite": r.get("composite_score"),
        "leg_count": int(r.get("negbin_leg_count") or r.get("legs") or 0),
        "cell_adj": r.get("negbin_cell_adj"),
        "pnl": pnl,
        "legs": legs_out,
    }


@app.get("/api/negbin/arms_comparison")
def negbin_arms_comparison(
    days: int = Query(30, ge=1, le=365),
):
    """
    Comparative view of all active shadow counterfactual arms.
    Returns per-arm summary metrics + daily cumulative PnL series.
    """
    from collections import defaultdict as _dd

    _ARMS = [
        ("shadow_negbin", "A (control)"),
        ("shadow_negbin_recal", "B (recal)"),
        ("shadow_negbin_edgegate", "Edge Gate"),
        ("shadow_negbin_fullstack", "Full-Stack"),
        ("shadow_negbin_composite", "Composite"),
        ("shadow_negbin_atscap", "ATS Cap"),
        ("shadow_negbin_distwp_gate", "dist_wp Gate"),
        ("shadow_negbin_bothgates", "Both Gates"),
    ]

    con = _negbin_db_connect()
    cur = con.cursor()

    arms_out = []
    daily_series = {}  # source → [{date, cum_pnl}, ...]

    for source_tag, label in _ARMS:
        # Fetch settled + pending parlays for this arm
        cur.execute("""
            SELECT mb.id, mb.odds, mb.status, mb.game_date, mb.amount, mb.weight,
                   GROUP_CONCAT(mbl.leg_result, ',') AS leg_results,
                   COUNT(mbl.id) AS n_legs
            FROM mock_bets mb
            JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
            WHERE mb.source = ?
              AND mb.cohort = 'clean'
              AND mb.game_date >= date('now', '-' || ? || ' days')
            GROUP BY mb.id
            ORDER BY mb.game_date
        """, (source_tag, days))
        rows = [dict(r) for r in cur.fetchall()]

        n_parlays = len(rows)
        n_settled = 0
        n_wins = 0
        total_pnl = 0.0
        total_legs = 0
        settled_dates = set()
        pnl_by_date = _dd(float)

        for r in rows:
            total_legs += r.get("n_legs", 0)
            won = _negbin_is_won(r.get("leg_results"))
            if won is None:
                continue  # pending
            n_settled += 1
            settled_dates.add(r["game_date"])
            odds = r.get("odds") or 1.0
            # Always use flat $10 stake for consistent comparison across arms
            if won:
                n_wins += 1
                pnl = _NEGBIN_STAKE * (odds - 1)
            else:
                pnl = -_NEGBIN_STAKE
            total_pnl += pnl
            pnl_by_date[r["game_date"]] += pnl

        cover_rate = round(n_wins / n_settled * 100, 2) if n_settled else None
        # ROI = return per dollar staked (comparable across arms and with backtests)
        roi = round(total_pnl / (n_settled * _NEGBIN_STAKE) * 100, 2) if n_settled else None
        settled_days = len(settled_dates)

        # Build daily cumulative series
        sorted_dates = sorted(pnl_by_date.keys())
        cum = 0.0
        series = []
        for d in sorted_dates:
            cum += pnl_by_date[d]
            series.append({"date": d, "cum_pnl": round(cum, 2), "day_pnl": round(pnl_by_date[d], 2)})
        daily_series[source_tag] = series

        arms_out.append({
            "source": source_tag,
            "label": label,
            "n_parlays": n_parlays,
            "n_settled": n_settled,
            "n_legs": total_legs,
            "cover_rate": cover_rate,
            "roi": roi,
            "cum_pnl": round(total_pnl, 2),
            "settled_days": settled_days,
            "status": "accumulating" if n_settled < 10 else "active",
        })

    con.close()

    return {
        "arms": arms_out,
        "daily_series": daily_series,
        "days": days,
    }


@app.get("/api/ats-ungated/tracker")
def ats_ungated_tracker(
    days: int = Query(30, ge=1, le=365),
    cohort: str = Query("clean"),
):
    """
    ATS ungated shadow tracker.
    Returns per-rung stats, score-bucket breakdown, rung×bucket cross-tab,
    calibration bins, and full leg list for the ats_ungated shadow source.
    """
    from collections import defaultdict as _dd
    from datetime import date as _date

    _STAKE = 10.0

    con = _negbin_db_connect()
    cur = con.cursor()

    sql = """
        SELECT mb.id, mb.odds, mb.status, mb.game_date, mb.predicted_win_prob,
               mb.notes, mb.bet_info,
               mbl.description, mbl.win_prob, mbl.leg_result, mbl.market_type,
               mbl.fixture_id, mbl.is_alt_line
        FROM mock_bets mb
        JOIN mock_bet_legs mbl ON mbl.mock_bet_id = mb.id
        WHERE mb.source = 'ats_ungated'
          AND mb.cohort = ?
          AND mb.game_date >= date('now', '-' || ? || ' days')
        ORDER BY mb.game_date DESC, mb.id
    """
    cur.execute(sql, (cohort, days))
    raw_rows = [dict(r) for r in cur.fetchall()]
    con.close()

    # ── Parse each row ───────────────────────────────────────────────────────
    legs = []
    for row in raw_rows:
        # Parse spread/gate from notes
        notes = row.get("notes") or ""
        spread_val = None
        gate_val = None
        for part in notes.split():
            if part.startswith("spread="):
                try:
                    spread_val = float(part.split("=", 1)[1])
                except ValueError:
                    pass
            elif part.startswith("gate="):
                gate_val = part.split("=", 1)[1]

        side = "fav" if (spread_val is not None and spread_val < 0) else "dog"
        rung = str(spread_val) if spread_val is not None else "unknown"

        # Win probability (prefer leg-level, fall back to bet-level)
        wp_raw = row.get("win_prob") or row.get("predicted_win_prob")
        try:
            wp = float(wp_raw) if wp_raw is not None else None
        except (ValueError, TypeError):
            wp = None

        # Decimal odds and breakeven
        odds_raw = row.get("odds")
        try:
            odds = float(odds_raw) if odds_raw is not None else None
        except (ValueError, TypeError):
            odds = None
        valid_odds = odds is not None and odds > 1.0
        breakeven = round(1.0 / odds, 6) if valid_odds else None

        # Status / result
        leg_result = row.get("leg_result")
        if leg_result in ("W", "WIN"):
            status = "WON"
        elif leg_result in ("L", "LOSS"):
            status = "LOST"
        else:
            status = "PENDING"

        legs.append({
            "id": str(row["id"]),
            "date": row.get("game_date") or "",
            "description": row.get("description") or "",
            "rung": rung,
            "side": side,
            "gate": gate_val or "",
            "wp": wp,
            "odds": odds,
            "breakeven": breakeven,
            "no_odds": not valid_odds,
            "status": status,
            "result": leg_result,
            "spread_val": spread_val,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(legs)
    settled_legs = [lg for lg in legs if lg["status"] != "PENDING"]
    pending_legs = [lg for lg in legs if lg["status"] == "PENDING"]
    won_legs = [lg for lg in legs if lg["status"] == "WON"]
    lost_legs = [lg for lg in legs if lg["status"] == "LOST"]
    settled = len(settled_legs)
    won = len(won_legs)
    lost = len(lost_legs)

    def _leg_pnl(lg: dict) -> float:
        if lg["status"] == "WON":
            odds_v = lg["odds"] if lg["odds"] and lg["odds"] > 1.0 else 2.0
            return (odds_v - 1.0) * _STAKE
        if lg["status"] == "LOST":
            return -_STAKE
        return 0.0

    total_pnl = sum(_leg_pnl(lg) for lg in settled_legs)
    roi = round(total_pnl / (settled * _STAKE), 6) if settled else None

    dates = [lg["date"] for lg in legs if lg["date"]]
    first_date = min(dates) if dates else None
    days_active = None
    if first_date:
        try:
            fd = _date.fromisoformat(first_date)
            days_active = (_date.today() - fd).days
        except Exception:
            pass

    summary = {
        "total": total,
        "settled": settled,
        "pending": len(pending_legs),
        "won": won,
        "lost": lost,
        "first_date": first_date,
        "days_active": days_active,
        "pnl": round(total_pnl, 4),
        "roi": round(roi, 6) if roi is not None else None,
    }

    # ── Helper: compute stats for a group of legs ────────────────────────────
    def _group_stats(group: list) -> dict:
        n = len(group)
        grp_settled = [lg for lg in group if lg["status"] != "PENDING"]
        grp_won = [lg for lg in group if lg["status"] == "WON"]
        grp_lost = [lg for lg in group if lg["status"] == "LOST"]
        s = len(grp_settled)
        w = len(grp_won)
        pnl = sum(_leg_pnl(lg) for lg in grp_settled)
        wps = [lg["wp"] for lg in group if lg["wp"] is not None]
        odds_list = [lg["odds"] for lg in group if lg["odds"] and lg["odds"] > 1.0]
        be_list = [lg["breakeven"] for lg in group if lg["breakeven"] is not None]
        no_odds_count = sum(1 for lg in group if lg["no_odds"])
        avg_wp = round(sum(wps) / len(wps), 6) if wps else None
        avg_odds = round(sum(odds_list) / len(odds_list), 6) if odds_list else None
        avg_be = round(sum(be_list) / len(be_list), 6) if be_list else None
        cover_rate = round(w / s, 6) if s > 0 else None
        roi_v = round(pnl / (s * _STAKE), 6) if s > 0 else None
        return {
            "n": n, "settled": s, "won": w, "lost": len(grp_lost),
            "avg_wp": avg_wp, "avg_odds": avg_odds, "avg_breakeven": avg_be,
            "cover_rate": cover_rate, "pnl": round(pnl, 4), "roi": roi_v,
            "no_odds_count": no_odds_count,
        }

    # ── By rung ──────────────────────────────────────────────────────────────
    rung_map: dict = _dd(list)
    for lg in legs:
        rung_map[(lg["rung"], lg["side"])].append(lg)

    by_rung = []
    for (rung_key, side_key) in sorted(rung_map.keys(), key=lambda x: (float(x[0]) if x[0] != "unknown" else 9999)):
        group = rung_map[(rung_key, side_key)]
        stats = _group_stats(group)
        by_rung.append({
            "rung": rung_key,
            "side": side_key,
            **{k: v for k, v in stats.items() if k != "no_odds_count"},
        })

    # ── Score buckets (quintiles by win_prob) ────────────────────────────────
    _BUCKET_LABELS = ["Q1 (top)", "Q2", "Q3", "Q4", "Q5 (bottom)"]
    scored_legs = sorted([lg for lg in legs if lg["wp"] is not None], key=lambda x: -x["wp"])
    n_scored = len(scored_legs)
    bucket_assignments: dict = {}
    if n_scored > 0:
        bucket_size = n_scored / 5
        for i, lg in enumerate(scored_legs):
            bucket_idx = min(int(i / bucket_size), 4)
            bucket_assignments[lg["id"]] = bucket_idx

    bucket_map: dict = _dd(list)
    for lg in legs:
        idx = bucket_assignments.get(lg["id"], 4)
        bucket_map[idx].append(lg)

    by_score_bucket = []
    for idx in range(5):
        group = bucket_map[idx]
        if not group:
            continue
        stats = _group_stats(group)
        wps_in_bucket = [lg["wp"] for lg in group if lg["wp"] is not None]
        wp_min = round(min(wps_in_bucket), 1) if wps_in_bucket else None  # already 0-100
        wp_max = round(max(wps_in_bucket), 1) if wps_in_bucket else None
        wp_range = f"{wp_min}-{wp_max}" if (wp_min is not None and wp_max is not None) else "n/a"
        by_score_bucket.append({
            "bucket": _BUCKET_LABELS[idx],
            "wp_range": wp_range,
            **stats,
        })

    # ── By rung × score bucket ────────────────────────────────────────────────
    cross_map: dict = _dd(list)
    for lg in legs:
        idx = bucket_assignments.get(lg["id"], 4)
        cross_map[(lg["rung"], lg["side"], _BUCKET_LABELS[idx])].append(lg)

    by_rung_score = []
    for (rung_key, side_key, bucket_label) in sorted(
        cross_map.keys(),
        key=lambda x: (float(x[0]) if x[0] != "unknown" else 9999, x[2]),
    ):
        group = cross_map[(rung_key, side_key, bucket_label)]
        stats = _group_stats(group)
        by_rung_score.append({
            "rung": rung_key,
            "side": side_key,
            "bucket": bucket_label,
            **{k: v for k, v in stats.items() if k != "no_odds_count"},
        })

    # ── Calibration bins (5pp bins of predicted wp) ──────────────────────────
    cal_map: dict = _dd(list)
    for lg in legs:
        if lg["wp"] is None:
            continue
        wp_pct = lg["wp"]  # already 0-100 scale
        bin_lo = int(wp_pct // 5) * 5
        bin_hi = bin_lo + 5
        bin_label = f"{bin_lo}-{bin_hi}"
        cal_map[bin_label].append(lg)

    calibration = []
    for bin_label in sorted(cal_map.keys(), key=lambda x: int(x.split("-")[0])):
        group = cal_map[bin_label]
        n_bin = len(group)
        settled_bin = [lg for lg in group if lg["status"] != "PENDING"]
        s_bin = len(settled_bin)
        wps = [lg["wp"] for lg in group if lg["wp"] is not None]
        avg_pred = round(sum(wps) / len(wps), 4) if wps else None  # already 0-100
        won_bin = sum(1 for lg in settled_bin if lg["status"] == "WON")
        realized = round(won_bin / s_bin, 6) if s_bin > 0 else None
        gap = round((realized * 100) - avg_pred, 2) if (realized is not None and avg_pred is not None) else None  # gap in pp
        calibration.append({
            "predicted_bin": bin_label,
            "n": n_bin,
            "settled": s_bin,
            "avg_predicted": avg_pred,
            "realized": realized,
            "gap": gap,
        })

    # ── Leg list (strip internal fields) ─────────────────────────────────────
    legs_out = [
        {
            "id": lg["id"],
            "date": lg["date"],
            "description": lg["description"],
            "rung": lg["rung"],
            "side": lg["side"],
            "gate": lg["gate"],
            "wp": lg["wp"],
            "odds": lg["odds"],
            "breakeven": lg["breakeven"],
            "status": lg["status"],
            "result": lg["result"],
        }
        for lg in legs
    ]

    return {
        "summary": summary,
        "by_rung": by_rung,
        "by_score_bucket": by_score_bucket,
        "by_rung_score": by_rung_score,
        "calibration": calibration,
        "legs": legs_out,
    }
