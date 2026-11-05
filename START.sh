#!/bin/bash
# BetIQ startup script
# Runs the server in the background (nohup + disown) so it survives
# terminal close and Mac sleep.  PID is saved to .betiq.pid for STOP.sh.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.betiq.pid"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/betiq_server.log"

echo "================================"
echo "  BetIQ — Sports Betting AI"
echo "================================"
echo "Working directory: $SCRIPT_DIR"
echo ""

# ── Kill any existing server process ────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Also kill any stray uvicorn on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

# ── Activate betiq conda env (SQLite 3.54.1, Python 3.12) ─────────────────
CONDA_BASE="/Users/joseportillo/anaconda3"
BETIQ_PYTHON="$CONDA_BASE/envs/betiq/bin/python"

if [ ! -x "$BETIQ_PYTHON" ]; then
    echo "ERROR: betiq conda env not found at $BETIQ_PYTHON"
    echo "Run: conda create -n betiq python=3.12 -c conda-forge && conda run -n betiq pip install -r requirements.txt"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate betiq
echo "Python: $(python --version)"
echo "SQLite: $(python -c 'import sqlite3; print(sqlite3.sqlite_version)')"
echo ""

# ── Verify backend ────────────────────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/backend" ]; then
    echo "ERROR: backend/ folder not found"
    exit 1
fi

# ── Launch server in background (nohup + disown) ─────────────────────────────
echo "Starting BetIQ on http://localhost:8000"
echo "Log: $LOG_FILE"
echo ""
echo "  → Dashboard: http://localhost:8000"
echo "  → API docs:  http://localhost:8000/docs"
echo "  → Stop:      bash STOP.sh  (or kill $(cat $PID_FILE 2>/dev/null))"
echo ""

cd "$SCRIPT_DIR/backend"

# ── Prevent Mac idle sleep while server is running ───────────────────────────
CAFFEINATE_PID_FILE="$SCRIPT_DIR/.caffeinate.pid"
if [ -f "$CAFFEINATE_PID_FILE" ]; then
    kill "$(cat "$CAFFEINATE_PID_FILE")" 2>/dev/null || true
    rm -f "$CAFFEINATE_PID_FILE"
fi
caffeinate -i &
CAFFEINATE_PID=$!
echo $CAFFEINATE_PID > "$CAFFEINATE_PID_FILE"
echo "Caffeinate started (PID $CAFFEINATE_PID) — Mac sleep prevented."

# Rotate log (keep last 5000 lines)
if [ -f "$LOG_FILE" ]; then
    tail -5000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
fi

nohup "$BETIQ_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000 \
    >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
disown $SERVER_PID

echo "Server started (PID $SERVER_PID)"
echo ""

# ── Wait for server to be ready (up to 30s) ──────────────────────────────────
echo -n "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo " ready in ${i}s"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Open browser
(open "http://localhost:8000" 2>/dev/null || true) &

echo "Server is running in the background."
echo "It will stay alive after this terminal closes."
echo ""
echo "  tail -f $LOG_FILE    # follow logs"
echo "  bash STOP.sh          # stop server"
echo "  curl http://localhost:8000/api/scheduler/health  # check health"
