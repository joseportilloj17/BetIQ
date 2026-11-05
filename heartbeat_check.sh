#!/bin/bash
# BetIQ External Heartbeat — runs via cron OUTSIDE the server process.
# Checks: (1) scheduler heartbeat file is fresh, (2) today's shadow bets exist,
# (3) neg15 arm fired, (4) bayes fill rate.
# Install: crontab -e → "0 14 * * * /Users/joseportillo/Downloads/BetIQ/heartbeat_check.sh"
#   (14:00 UTC = 9:00 AM CT)

# Read-only wrapper: -readonly opens with SQLITE_OPEN_READONLY (no checkpoint,
# no WAL write).  query_only=1 as second layer in case -readonly is ever lost.
BETS_DB="/Users/joseportillo/Downloads/BetIQ/data/bets.db"
sql_ro() { sqlite3 -readonly "$BETS_DB" "PRAGMA query_only=1; $1"; }
HEARTBEAT_FILE="/Users/joseportillo/Downloads/BetIQ/.betiq_heartbeat"
LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/betiq_heartbeat.log"
TODAY=$(date +%Y-%m-%d)
MAX_STALE_HOURS=3  # alert if heartbeat older than this

echo "[$(date)] ── Heartbeat check ──" >> "$LOG"

# ── 1. Scheduler heartbeat file staleness ──────────────────────────────────
if [ -f "$HEARTBEAT_FILE" ]; then
    HB_EPOCH=$(cat "$HEARTBEAT_FILE" | head -1)
    NOW_EPOCH=$(date +%s)
    if [ -n "$HB_EPOCH" ] && [ "$HB_EPOCH" -gt 0 ] 2>/dev/null; then
        AGE_SECS=$((NOW_EPOCH - HB_EPOCH))
        AGE_HOURS=$((AGE_SECS / 3600))
        AGE_MINS=$((AGE_SECS / 60))
        echo "[$(date)] Heartbeat age: ${AGE_MINS}m" >> "$LOG"
        if [ "$AGE_SECS" -gt $((MAX_STALE_HOURS * 3600)) ]; then
            MSG="ALERT: Scheduler heartbeat STALE (${AGE_HOURS}h old)"
            echo "[$(date)] $MSG" >> "$LOG"
            osascript -e "display notification \"$MSG\" with title \"BetIQ STALE\" sound name \"Sosumi\"" 2>/dev/null
        else
            echo "[$(date)] OK: heartbeat fresh (${AGE_MINS}m)" >> "$LOG"
        fi
    else
        echo "[$(date)] WARNING: Could not parse heartbeat epoch: $HB_EPOCH" >> "$LOG"
    fi
else
    MSG="ALERT: Heartbeat file MISSING — scheduler may never have started"
    echo "[$(date)] $MSG" >> "$LOG"
    osascript -e "display notification \"$MSG\" with title \"BetIQ CRITICAL\" sound name \"Sosumi\"" 2>/dev/null
fi

# ── 2. Server process alive check ─────────────────────────────────────────
if ! lsof -ti:8000 > /dev/null 2>&1; then
    MSG="SERVER PROCESS NOT RUNNING on port 8000 — launchd should restart"
    echo "[$(date)] $MSG" >> "$LOG"
    osascript -e "display notification \"$MSG\" with title \"BetIQ CRITICAL\" sound name \"Sosumi\"" 2>/dev/null
fi

# ── 3. Shadow dispatch existence check ────────────────────────────────────
COUNT=$(sql_ro "SELECT COUNT(*) FROM mock_bets WHERE source='shadow_negbin' AND game_date='$TODAY';" 2>/dev/null)
echo "[$(date)] shadow_negbin legs for $TODAY = ${COUNT:-ERROR}" >> "$LOG"

if [ -z "$COUNT" ] || [ "$COUNT" -eq 0 ]; then
    MSG="ALERT: NegBin shadow dispatch MISSING for $TODAY"
    echo "[$(date)] $MSG" >> "$LOG"
    osascript -e "display notification \"$MSG\" with title \"BetIQ ALERT\" sound name \"Basso\"" 2>/dev/null
else
    echo "[$(date)] OK: $COUNT shadow_negbin bets found" >> "$LOG"
fi

# ── 4. Neg-1.5 arm check ─────────────────────────────────────────────────
N15_COUNT=$(sql_ro "SELECT COUNT(*) FROM mock_bets WHERE source='shadow_negbin_neg15' AND game_date='$TODAY';" 2>/dev/null)
echo "[$(date)] shadow_negbin_neg15 bets for $TODAY = ${N15_COUNT:-ERROR}" >> "$LOG"

if [ -z "$N15_COUNT" ] || [ "$N15_COUNT" -eq 0 ]; then
    # Only alert if shadow_negbin already ran (neg15 depends on Shadow A legs)
    if [ "$COUNT" -gt 0 ]; then
        MSG="WARNING: Neg-1.5 arm produced 0 bets (may be legitimate if no fixtures in 85-90% band)"
        echo "[$(date)] $MSG" >> "$LOG"
        # Softer alert — 0 bets is possible on sparse days
        osascript -e "display notification \"$MSG\" with title \"BetIQ neg15\" sound name \"Ping\"" 2>/dev/null
    fi
else
    echo "[$(date)] OK: $N15_COUNT neg-1.5 bets found" >> "$LOG"
fi

# ── 5. Bayes fill rate (cascade failure detection) ────────────────────────
if [ "$COUNT" -gt 0 ]; then
    BAYES_FILL=$(sql_ro "SELECT CAST(SUM(CASE WHEN bayes_team_wp IS NOT NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) FROM mock_bet_legs WHERE mock_bet_id IN (SELECT id FROM mock_bets WHERE source='shadow_negbin' AND game_date='$TODAY');" 2>/dev/null)
    echo "[$(date)] Bayes fill rate: $BAYES_FILL" >> "$LOG"

    if [ "$BAYES_FILL" = "0.0" ]; then
        osascript -e 'display notification "BetIQ: Bayes signals ALL NULL today — pipeline cascade failure" with title "BetIQ WARNING" sound name "Ping"' 2>/dev/null
        echo "[$(date)] WARNING: Bayes fill rate is 0% — cascade failure" >> "$LOG"
    fi
fi

# ── 6. Disk space check ──────────────────────────────────────────────────
FREE_KB=$(df -k / | tail -1 | awk '{print $4}')
FREE_GB=$((FREE_KB / 1048576))
echo "[$(date)] Free disk space: ${FREE_GB} GB" >> "$LOG"

if [ "$FREE_GB" -lt 5 ]; then
    MSG="CRITICAL: Only ${FREE_GB} GB free on / — database writes at risk"
    echo "[$(date)] $MSG" >> "$LOG"
    osascript -e "display notification \"$MSG\" with title \"BetIQ CRITICAL\" sound name \"Sosumi\"" 2>/dev/null
elif [ "$FREE_GB" -lt 10 ]; then
    MSG="WARNING: Only ${FREE_GB} GB free on / — approaching low disk space"
    echo "[$(date)] $MSG" >> "$LOG"
    osascript -e "display notification \"$MSG\" with title \"BetIQ WARNING\" sound name \"Ping\"" 2>/dev/null
else
    echo "[$(date)] OK: ${FREE_GB} GB free disk space" >> "$LOG"
fi
