# BetIQ — Standing Rules

## Persistence
- **Never write persistent artifacts to `/tmp`.**  Temporary analysis is fine, but any data that must survive a reboot (backfills, forward logs, calibration tables) goes into `data/` or `historical.db`.

## Failure Handling
- **No silent fallbacks or silently skipped jobs.**  Every code path that catches an exception and continues must log a warning at minimum.  Scheduler jobs that produce zero output on a day they were expected to run must alert (print + set a state flag the health-check reads).
- **Failures must alert.**  If a scheduled write (settlement, snapshot, backfill) completes with 0 rows on a day games were played, that is a failure, not a no-op.

## Reporting
- **Every status report starts with the OPEN_ITEMS diff.**  Before any new analysis, list all outstanding items with their current status (DONE / OPEN / PARTIAL) so nothing is silently dropped.

## Data Safety
- **Off-machine backups only copy the daily checkpoint**, never the live DB.  The live `bets.db` must never be read by rsync/rclone/cp while the server is running.
- **`INSERT OR REPLACE` for bulk syncs** (game logs, etc.) so stat corrections are captured on every refresh.
- **`sqlite3.backup()` API for all DB copies.**  Never `shutil.copy*` on a WAL-mode database.

## SQLite WAL-Reset Bug (P0 until upgraded)
- **Minimum safe SQLite: 3.51.3+**.  `database.py` hard-fails on startup if below 3.51.3 (versions 3.45.x–3.50.6 and 3.51.0–3.51.2 are still vulnerable).  Current env: `conda activate betiq` → SQLite 3.54.1 (Python 3.12).
- **Single-writer discipline**: no ad-hoc scripts may write to `bets.db` or `historical.db` while the scheduler is running.  All writes go through the API server process.  Checkpoints stay PASSIVE.
- **No `UPDATE … ORDER BY … LIMIT`**: SQLite on this system does not support it.  Use a subquery: `UPDATE t SET … WHERE id = (SELECT id FROM t WHERE … ORDER BY … LIMIT 1)`.

## MLB Season Gate
- MLB scheduled jobs (`/scores` polling, alt-line fetches, settlement) are gated on `_is_mlb_season()`.  Outside the active season window (roughly late-March through early-November), these jobs must not run or consume API credits.

## Conda Environment
- **Never run `conda install` or `conda update` in the base env.**  The base env's dependency solver can deadlock and fill the disk.  All packages go into the `betiq` env: `conda run -n betiq pip install <pkg>` or `conda install -n betiq -c conda-forge <pkg>`.
- **The server runs in the `betiq` conda env** (Python 3.12, SQLite 3.54.1).  All launchers (`server_foreground.sh`, `START.sh`) activate this env.

## Background Processes
- **Stop background shells when their task finishes.**  Never leave a `run_in_background` command, monitor, or subagent running after its purpose is complete.
- **List any still-running background tasks at the end of each session** so the user can kill stragglers.

## Secrets & Credentials
- **Never hardcode API keys, passwords, or tokens in source files.**  All secrets live in `backend/.env` (gitignored).  Code reads them via `os.environ.get()` or `from config import`.
- **Never commit data files** (`*.db`, `*.sqlite`, `*.csv`, `*.json` data exports) to git.  The `data/` directory is gitignored.
- **`backend/.env.example`** has variable names with placeholder values — keep it updated when adding new keys.
- **Gitleaks pre-commit hook** blocks commits containing secrets.  If it fires, move the secret to `.env`.  False positives go in `.gitleaks.toml` allowlist.
- **GitHub secret scanning + push protection** are enabled on the repo.
