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

## NFL Prereg Freeze (until W14 verdict, ~2026-12-05)
- **Do NOT change P(over) logic in `find_prop_gaps()`.**  P(over) is frozen as lognormal for yardage stats.  The prereg (FW-2) reads stored `p_over` values; changing the distribution mid-test invalidates the forward read.  The `distributional_median` column uses shifted gamma ppf(0.5) — that is a separate metric and may be updated.
- **Do NOT change `p_over_line()` function behavior** or the variance inputs to it for yardage stats.  Changes to NegBin (counts) P(over) are allowed if justified.
- **After W14 verdict**: evaluate switching P(over) to shifted gamma (`p_over_line_gamma_shifted` from `nfl_interim_dispersion.py`) for yardage.  This is logged in OPEN_ITEMS.md entry 2026-11-08a.

## Forward-Prediction Storage
- **Forward-prediction tables store every market-posted line.**  For each player × stat, store sim probabilities at every line the market posted (main + all alt rungs), with book, prices, and devig method.  Selection and suppression logic runs downstream only, never at storage time.

## UI Metric Labels
- **Every metric shown in the UI must display**: sport, time period, filters applied, N (sample size), and a confidence interval where applicable.  No naked headline numbers (e.g. "60.5% WR") without context.
- **Run the frontend smoke test** (`frontend/smoke_test.py`) before reporting any frontend change as done.  The test verifies every nav tab's page div exists, its loader function exists, and there are no unclosed JS blocks.

## Production Verification
- **"Verified in production" means exercised through the scheduled production code path with rows written**, not through a test harness, manual script, or import check.  A function that parses correctly and passes unit tests is not verified until the scheduler has called it and the expected rows appear in the database.
- **After deploying new pipeline code**, confirm rows in the target table from the next scheduled run before marking the feature as live.  Check the morning after, not the same session.
- **A fix to a scheduled path isn't done until the server is restarted and the running commit matches.**  Code on disk doesn't help if the running process is still using the old version.  After any deploy, verify via `/api/health` that `git_commit_running == git_commit_disk`.

## Null-Result Logging
- **Record null results of tested hypotheses in OPEN_ITEMS** under "Tested Hypotheses — Null Results" with the date, method summary, and a pre-specified re-test condition.  Do not re-run a null hypothesis ad hoc before its re-test condition is met.  This prevents wasting cycles on already-closed questions and ensures re-tests use fresh data with a pre-committed design.

## Session Discipline
- **Commit and push at the end of every session.**  Uncommitted changes on disk make the version hash meaningless.  The health email flags a dirty working tree, but the fix is to not leave one.
- **Pre-registration freezes always reference a tagged, pushed commit.**  Tag format: `s5-4-freeze-YYYY-MM-DD`.  The tag must exist on `origin` so anyone can `git checkout` the exact frozen code.  Never freeze to an unpushed or uncommitted state.

## Secrets & Credentials
- **Never hardcode API keys, passwords, or tokens in source files.**  All secrets live in `backend/.env` (gitignored).  Code reads them via `os.environ.get()` or `from config import`.
- **Never commit data files** (`*.db`, `*.sqlite`, `*.csv`, `*.json` data exports) to git.  The `data/` directory is gitignored.
- **`backend/.env.example`** has variable names with placeholder values — keep it updated when adding new keys.
- **Gitleaks pre-commit hook** blocks commits containing secrets.  If it fires, move the secret to `.env`.  False positives go in `.gitleaks.toml` allowlist.
- **GitHub secret scanning + push protection** are enabled on the repo.
