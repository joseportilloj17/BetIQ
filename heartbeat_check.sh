#!/bin/bash
# BetIQ External Heartbeat — runs via cron OUTSIDE the server process.
# Checks if today's NegBin shadow dispatch completed. Alerts if missing by 9 AM CT.
# Install: crontab -e → "0 14 * * * /Users/joseportillo/Downloads/BetIQ/heartbeat_check.sh"
#   (14:00 UTC = 9:00 AM CT)

BETS_DB="/Users/joseportillo/Downloads/BetIQ/data/bets.db"
LOG="/tmp/betiq_heartbeat.log"
TODAY=$(date +%Y-%m-%d)

# Check if shadow_negbin dispatch exists for today
COUNT=$(sqlite3 "$BETS_DB" "SELECT COUNT(*) FROM mock_bets WHERE source='shadow_negbin' AND game_date='$TODAY';" 2>/dev/null)

echo "[$(date)] Heartbeat check: shadow_negbin legs for $TODAY = ${COUNT:-ERROR}" >> "$LOG"

if [ -z "$COUNT" ] || [ "$COUNT" -eq 0 ]; then
    # Alert: no dispatch today
    MSG="ALERT: BetIQ NegBin shadow dispatch MISSING for $TODAY. Check server status."
    echo "[$(date)] $MSG" >> "$LOG"

    # macOS notification
    osascript -e "display notification \"$MSG\" with title \"BetIQ ALERT\" sound name \"Basso\"" 2>/dev/null

    # Also check if server process is alive
    if ! lsof -ti:8000 > /dev/null 2>&1; then
        echo "[$(date)] SERVER PROCESS NOT RUNNING on port 8000!" >> "$LOG"
        osascript -e 'display notification "BetIQ server is DOWN — launchd should restart" with title "BetIQ CRITICAL" sound name "Sosumi"' 2>/dev/null
    fi
else
    echo "[$(date)] OK: $COUNT shadow_negbin bets found for $TODAY" >> "$LOG"
fi

# Check weather refresh landed today (prevent silent month-long dropout)
HIST_DB="/Users/joseportillo/Downloads/BetIQ/data/historical.db"
WX_COUNT=$(sqlite3 "$HIST_DB" "SELECT COUNT(*) FROM game_weather WHERE game_date='$TODAY' AND wind_speed_forecast_mph IS NOT NULL;" 2>/dev/null)
echo "[$(date)] Weather forecast rows for $TODAY: ${WX_COUNT:-ERROR}" >> "$LOG"
if [ -z "$WX_COUNT" ] || [ "$WX_COUNT" -eq 0 ]; then
    osascript -e "display notification \"BetIQ: Weather refresh MISSING for $TODAY — forecast wind not available\" with title \"BetIQ Weather FAIL\" sound name \"Basso\"" 2>/dev/null
    echo "[$(date)] ALERT: Weather refresh missing for $TODAY" >> "$LOG"
fi

# Check bayes fill rate (catch the 7/18-style cascade failure)
if [ "$COUNT" -gt 0 ]; then
    BAYES_FILL=$(sqlite3 "$BETS_DB" "SELECT CAST(SUM(CASE WHEN bayes_team_wp IS NOT NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) FROM mock_bet_legs WHERE mock_bet_id IN (SELECT id FROM mock_bets WHERE source='shadow_negbin' AND game_date='$TODAY');" 2>/dev/null)
    echo "[$(date)] Bayes fill rate: $BAYES_FILL" >> "$LOG"

    # Alert if bayes is completely missing (0.0)
    if [ "$BAYES_FILL" = "0.0" ]; then
        osascript -e 'display notification "BetIQ: Bayes signals ALL NULL today — pipeline cascade failure" with title "BetIQ WARNING" sound name "Ping"' 2>/dev/null
        echo "[$(date)] WARNING: Bayes fill rate is 0% — cascade failure" >> "$LOG"
    fi
fi
