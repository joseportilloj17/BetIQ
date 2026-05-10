#!/bin/bash
# BetIQ startup script
# Runs the server in the background (nohup + disown) so it survives
# terminal close and Mac sleep.  PID is saved to .betiq.pid for STOP.sh.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.betiq.pid"
LOG_FILE="/tmp/betiq_server.log"

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

# ── Check Python ─────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install from https://python.org"
    exit 1
fi
echo "Python: $(python3 --version)"

# ── Venv: only rebuild when missing or requirements changed ──────────────────
VENV_STAMP="$SCRIPT_DIR/.venv/.requirements_stamp"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

need_install=0

if [ ! -d "$SCRIPT_DIR/.venv" ] || [ ! -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    need_install=1
elif [ ! -f "$VENV_STAMP" ] || ! diff -q "$REQ_FILE" "$VENV_STAMP" > /dev/null 2>&1; then
    echo "requirements.txt changed — reinstalling dependencies..."
    need_install=1
else
    echo "Virtual environment up to date — skipping install."
fi

source "$SCRIPT_DIR/.venv/bin/activate"

if [ $need_install -eq 1 ]; then
    echo "Upgrading pip..."
    pip install --upgrade pip -q
    echo "Installing dependencies..."
    pip install -r "$REQ_FILE" -q
    cp "$REQ_FILE" "$VENV_STAMP"
    echo "Dependencies installed."
fi
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

nohup "$SCRIPT_DIR/.venv/bin/python" -m uvicorn main:app --port 8000 \
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
