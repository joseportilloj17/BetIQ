#!/bin/bash
# Stop the BetIQ server

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/.betiq.pid"
CAFFEINATE_PID_FILE="$SCRIPT_DIR/.caffeinate.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping BetIQ server (PID $PID)..."
        kill "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null
        fi
        echo "Stopped."
    else
        echo "Server (PID $PID) was not running."
    fi
    rm -f "$PID_FILE"
else
    echo "No PID file found — trying to kill any uvicorn on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "Killed." || echo "Nothing found on port 8000."
fi

# ── Stop caffeinate ───────────────────────────────────────────────────────────
if [ -f "$CAFFEINATE_PID_FILE" ]; then
    kill "$(cat "$CAFFEINATE_PID_FILE")" 2>/dev/null && echo "Caffeinate stopped." || true
    rm -f "$CAFFEINATE_PID_FILE"
fi
