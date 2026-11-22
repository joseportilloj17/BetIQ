# BetIQ — Open Items

Last updated: 2026-11-22
**Last reconciled**: 2026-11-07 (re-check weekly)

## B1: Minutes Model — ALL COMPLETE

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Spread-aware path + two-sided blowout | DONE | `_p_blowout()` (minutes_model.py:1164), `project()` line 690-707: both-sided excess probability vs `_BASELINE_BLOWOUT_PROB=0.264` |
| 2 | Measured blowout cuts with 240-sum | DONE | `_BLOWOUT_STARTER_CUT=-2.3`, `_BLOWOUT_ROTATION_CUT=+4.0`, `_BLOWOUT_SPOT_CUT=+7.0` (measured). `redistribute_team_minutes()` enforces Σ=240 |
| 3 | Starter short-game left tail | DONE | Two-component mixture in `MinutesProjection.cdf()`: `(1-0.035)×N(μ,σ) + 0.035×N(14.8, 4.7)` |
| 4 | Rotation recency blend | DONE | `_BLEND_MODEL_WEIGHT = {"starter": 0.6, "rotation": 0.3, "spot": 0.2}`, `_recency_weighted_mean()` with half-life=20 |
| 5 | Team-changer verification | DONE | `_build_player_profiles()` filters to post-trade games via MATCHUP string |
| 6 | Minutes-cap CLI | DONE | `set_restriction()`, `remove_restriction()`, `list_restrictions()`, `load_restrictions()` in minutes_model.py:1004+ with JSON persistence |

## B2: Rate × Minutes — ALL COMPLETE

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Minutes-deviation subset test (≥4 min from norm, all rotation+) | DONE | Parity confirmed: rate vs baseline at parity in both high-dev and normal subsets (15,644 games, 350 players). Positive bias +0.37–0.50 in high-dev = usage absorption (expected). |
| 2 | Joint absence-rate model (ridge, shrinkage, timestamp gate, ablation) | DONE → KILLED | Ridge (243 models, 2071 pairs, α=1.0) deconfounded correctly but walk-forward ablation: +14.9% worse MAE (N=12,243, CI [+0.60, +0.71]). Median |β|=0.049 on base σ≈0.20 — signal too weak. Code removed from production, kept for research. |
| 3 | Pace from market total (remove engine pace multiplier) | DONE | `_expected_possessions()` + `_player_pace_adj()` replace `_pace_factor()`. Team-relative (not league-relative). Old double-count removed. |

## B3: Reconciliation — COMPLETE (this session)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | PRA combined rate (covariance-aware std) | DONE | `pra_per_min` + `pra_rate_std` on PlayerProfile. Delta method PRA std +4% to +32% over independent sum. |
| 2 | 240-minute enforcement | DONE | `_enforce_team_minutes()` always runs. Mean gap: 0.00 (was -55). |
| 3 | Rotation selection (top-10 by MPG) | DONE | Replaced flat 15 MPG cutoff with top-10 / cum-MPG-270 selection. Roster: 8-10/team. |
| 4 | Points reconciliation to team total | DONE | Proportional scaling, cap ±25%. Mean scale 0.971 (structural: healthy rotation > injury-avg). |
| 5 | Rebounds from market-derived missed shots | DONE | Opponent missed FGs × 0.78 (live-ball) × (1 - opp OREB%) + missed FTs × 0.85 + own OREB. Mean reb_scale=0.976. |
| 6 | Assists from team FGM × assist rate | DONE | Hard cap: Σ assists ≤ team FGM. 0 violations across 30 teams. |
| 7 | scout_game() wired to project_game() | DONE | `_engine_game_projection()` runs once per game, feeds reconciled projections. Falls back to per-player. |

## B3 Open Follow-ups

| # | Item | Status | Notes |
|---|------|--------|-------|
| B3-F1 | Market data passthrough | DONE | `runner.py` calls `_fetch_market_lines()` before `scout_game()`. Populates `game.extra['market_total']`, `game.extra['market_spread']`, `game.extra['market_book']`. Fail-loud `[NO MARKET ANCHOR]` log + risk_factor tag when missing. FanDuel preferred, DraftKings fallback. |
| B3-F2 | pts_scale distribution | DONE | With game-specific market totals: mean pts_scale=0.996 (was 0.971 hardcoded). Outliers roster-driven: CHA 0.795 (deep rotation), MIA 1.156 (thin), BOS 1.090 (8 players). ±25% cap handles all. |
| B3-F3 | Rebound dead-ball rate calibration | DONE | DREB confirmed using opponent's OREB% (correct). `_LIVE_BALL_RATE=0.78` implicitly accounts for team rebounds (~3.5/game, same dead-ball population). No additive deduction needed — mean reb_scale=0.978 across 15 games with market data. Comments clarified. |

## B4: Ablation — COMPLETE

Walk-forward factor ablation (2023-24 → 2025-26, 2022-23 training start):
- Base = current production (B1–B3)
- Factors added one at a time: team defense → play-type/Synergy → matchups → form
- Also ablate hand-set rules vs measured alternatives
- CRPS + MAE with game-clustered bootstrap CIs; PIT checks
- Per-stat (PTS, REB, AST, 3PM, PRA, STL, BLK) × role
- Keep factor only if CRPS improves with CI excluding zero

### B4 Implementation Status

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Ablation harness (`nba_ablation.py`) | DONE | `build_engine_asof()`, `run_ablation()`, CRPS/MAE/PIT scoring, game-clustered bootstrap, paired comparison, CLI (pull/run/report/compare), incremental save + resume |
| 2 | `factor_config` in engine | DONE | `project_game(factor_config=...)` gates defense, form, synergy, matchup, reconcile, blowout, foul_adj, absence_adj |
| 3 | Data pull cache | DONE | All 4 seasons: 2022-23 (415), 2023-24 (421), 2024-25 (435), 2025-26 (435) player logs + team logs + opp stats + synergy + matchup rollup |
| 4 | Smoke-test validation | DONE | 30-game test: CRPS/MAE/PIT all sensible. Defense tiny effect, form small, reconcile clearly helps, blowout negligible. |
| 5 | Synergy play-type factor | DONE | `_synergy_factor()` in engine: weighted opp_def_ppp by team off_poss_pct, 40% signal, ±10% cap. Gated by `factor_config.synergy` |
| 6 | Defender matchup factor | DONE | `_matchup_factor()` in engine: G/F/C position defense from MatchupsRollup, 50% signal, ±15% cap. Gated by `factor_config.matchup` |
| 7 | Hand-set rule ablation | DONE | 12 configs: base chain (5) + full_no_{reconcile,blowout,synergy,matchup,foul_adj,absence_adj} (7). Foul/absence gated via minutes model kwargs |
| 8 | Full-season run | DONE | 3,875,410 predictions, 3,677 games, 3 seasons, 11 configs. ~50 min runtime with preload optimization. |

### B4 Results — Factor Verdicts

**Additive chain ΔCRPS vs base ("all" role, negative = better, `*` = CI excludes zero):**

| Config | PTS | REB | AST | FG3M | PRA |
|--------|-----|-----|-----|------|-----|
| base+defense | -0.0004* | 0.0000 | -0.0020* | -0.0014* | -0.0017* |
| base+def+synergy | -0.0003 | +0.0001 | -0.0021* | -0.0012* | -0.0020* |
| base+def+syn+matchup | +0.0002 | +0.0008* | -0.0020* | -0.0010* | -0.0002 |
| full | -0.0004 | +0.0049* | -0.0021* | -0.0001 | +0.0019 |

**Ablation ΔCRPS (full vs full_no_X, positive = factor helps):**

| Factor | PTS | REB | AST | PRA | Verdict |
|--------|-----|-----|-----|-----|---------|
| reconcile | -0.0248 | -0.0201 | -0.0081 | -0.0754 | **KILL — dramatically hurts all stats** |
| blowout | +0.0052 | +0.0017 | -0.0004 | +0.0087 | **KEEP — significant PTS/REB/PRA benefit** |
| synergy | -0.0001 | 0.0000 | +0.0002 | -0.0001 | KILL — negligible |
| matchup | +0.0002 | +0.0003 | +0.0004 | +0.0016 | KILL — negligible |
| foul_adj | +0.0003 | +0.0004 | 0.0000 | +0.0008 | KILL — negligible |
| absence_adj | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD — already removed in B2 |

**Factor decisions:**

| Factor | Decision | Key Evidence |
|--------|----------|-------------|
| defense | **ENABLE** | AST -0.0020*, FG3M -0.0014*, PRA -0.0017*. Only additive factor that consistently helps all roles. |
| blowout | **ENABLE** | Removing hurts PTS (+0.0052), REB (+0.0017), PRA (+0.0087). Two-sided blowout with minute swaps provides real value. |
| reconcile | **KILL** | B4 + R4 both show reconciliation hurts. R9 invalidated (format bug — no inactives excluded). KILL stands on R4 (4 seasons, ΔCRPS +0.021 to +0.084, all significantly worse). R13 = final recheck after pool fix. |
| synergy | DISABLE | Near-zero incremental. 40% signal cap may be too conservative, or defense already captures what Synergy adds. |
| matchup | DISABLE | Near-zero to slightly harmful. G/F/C position-level too coarse for useful signal. |
| form | DISABLE | Hurts starters (+0.0133 PTS, +0.0312 PRA) while helping rotation. Over-weights recent noise for stable-role players. Net unreliable. |
| foul_adj | DISABLE | Negligible effect. |
| absence_adj | **KILLED (R13)** | Confirmed dead on R11-corrected pool. ΔCRPS = 0.000 for every stat × role. Kill stands on both thin and corrected pools. |

**Confirmed production config: defense + blowout + 50% reconciliation.** Synergy, matchup, form, foul_adj, absence_adj all disabled. Reconciliation PROMOTED to 50% pull (R13: PTS -0.026, PRA -0.021, REB neutral). Prior kill was on thin pool (14.6% OTHER); corrected pool (2.1% OTHER) shows clear benefit.

**Starter/rotation split:** The `full` config shows dramatic divergence: starters hurt, rotation helped. Driven by form factor over-weighting recent noise for stable-role players. `base+defense` shows no such split — both roles improve equally.

### B4 Addendum — Reconciliation: Thin-Pool Artifact

The B4 ablation showed reconciliation hurting all stats (+0.0850 PRA CRPS). Before accepting, four checks were run:

**Check 1 — Historical line quality:** Verified correct (spread sign, formula, MAE all healthy).

**Check 2 — Player pool mismatch (ROOT CAUSE):** Engine projects 10-12 players/team, but `if actuals is None: continue` drops projected-but-DNP players, leaving only 6.8 evaluated per team. Reconciliation distributes volume across the full pool, but evaluation only scores the intersection → systematically penalizes reconciled players. 79% of team-games have <8 evaluated players ("thin pool").

**Check 3 — Deep/thin split confirms the artifact:**

| Subset | N games | PTS | REB | PRA |
|--------|---------|-----|-----|-----|
| Deep (≥9 eval, 21%) | 813 | **-0.0195*** | +0.0150* | +0.0006 |
| Thin (<8 eval, 79%) | 3,129 | +0.0408* | +0.0237* | +0.1058* |

Reconcile HELPS on deep-pool games. Aggregate harm is driven by thin games where the model leaks volume to un-evaluated bench players.

**Check 4 — Verdict: UNDECIDED.** Thin-pool artifact confirmed real. R4 pool-fixed re-test showed all variants still hurt — but engine has ~8% systematic under-projection (PTS scale 0.921). R4 results are confounded by this bias; must fix bias first (R6–R8), then re-run (R9).

### B4-R: Reconciliation Re-Test (pool-fixed)

| # | Item | Status | Notes |
|---|------|--------|-------|
| R1 | Production pool check | DONE | 9,740 team-games. Engine projects 9.9/team, 7.1 with actuals. Min coverage 79.8% (median 82.9%). |
| R2 | OTHER bucket | DONE | OTHER = 48.5 min / 19.1 pts mean (16.7% of team scoring). Ground truth from team box score logs. Growing: 12.2% (2022-23) → 21.4% (2025-26). |
| R3 | Historical pool from BoxScoreSummaryV2 | IN PROGRESS | Pulling inactives for 4,910 games. 2022-23: 1230/1230, 2023-24: 1230/1230, 2024-25: 821/1225 (67%). |
| R4 | Re-run ablation with pool fix | DONE | 1.93M predictions, 4 configs, 4 seasons. All variants hurt — but engine has ~8% under-projection bug (PTS scale 0.921). R4 results invalid until bias fixed. |
| R5 | Decide reconciliation | **KILL** | R4 confirms reconciliation hurts (4 seasons, all configs significantly worse). R9 invalidated (format bug). KILL stands on R4 evidence. R13 = final recheck after pool fix. |
| R6 | Bias decomposition | **DONE** | 48,430 player-games, 3 seasons. See R6 Results below. |
| R7 | Target audit | **DONE** | 3,677 games. Market-implied totals unbiased (bias=-0.23, MAE=8.95, r=0.486). No mapping errors. See R7 Results below. |
| R8 | Fix pace baseline | **SKIP** | Pace_factor=1.004 across 48K player-games — NOT systematically low. DET-BOS 0.94 was one slow-paced game, not a systematic bias. No fix needed. |
| R9 | Re-run reconciliation | **INVALIDATED** | Format bug: `_load_inactives_lookup` stored 10-digit game_ids ("0022300001"), player_logs use 8-digit ("22300001"). Zero inactives were actually excluded. R9 ≡ R4 on 2 seasons. "Baseline improvement" (3.805→3.748) was sample composition, not inactives fix. Bug fixed in nba_ablation.py. R13 = first valid test. |
| R10 | Pool mismatch audit | **DONE** | 2,460 team-games (2023-24). Format bug found & fixed. See R10 Results. |
| R11 | Pool construction fix | **DONE** | Trade detection + soft allocation (MPG>=3) + absence filter (N>=7). OTHER: 14.6% → **2.1%**. Pool: 9.6 → 11.0. See R11 Results. |
| R11-B | Bias decomp on R11 pool | **DONE** | PTS bias -1.37 → -0.44 (68% reduction). Rate bias GONE (selection artifact). Remaining bias = minutes only. Phantom: 20.2 min/tg (8.4%). See R11 Bias Decomposition. |
| R12 | Rate conditional on minutes | **SKIP** | Rate bias resolved by R11 pool fix (all roles ±1.3%). Remaining -0.44 PTS bias = minutes only. Clean subset (excl OT + phantom starters): bias -0.18 — availability artifact, not model deficiency. |
| PROD-1 | CommonTeamRoster cache | **DONE** | 521 players across 30 teams. Tier-2 team assignment in `_build_player_profiles()`. Cached in engine DB. |
| PROD-2 | NBA injury report | **DONE** | Sleeper (primary) + ESPN (secondary). 57 Out/IR, 34 DTD/GTD, 8 conflicts. Wired to `scout_game()` exclusions. |
| PROD-3 | Minutes diagnostic | **DONE** | Clean subset (excl OT + phantom starters): min bias -0.45 (from -0.95), PTS bias -0.18. Starters +0.17 min. Residual = availability artifact. |

### R6 Results — Bias Decomposition (48,430 player-games, 3 seasons)

Walk-forward decomposition of the ~9% PTS under-projection into components, using production config (defense + blowout only). Script: `/tmp/r6_bias_decomp.py`, data: `/tmp/r6_decomp.csv`.

**1. Pace: NOT the problem (R8 = SKIP)**

| Metric | Value |
|--------|-------|
| Mean game_pace_factor (expected_poss / lg_pace) | **1.004** |
| Mean player pace_adj (expected_poss / team_pace) | 1.004 |
| Mean def_adj | 1.000 |

All three factors centered at 1.0. The DET-BOS sample (0.94) was one slow-paced game. No possession baseline fix needed.

**2. Minutes: PRIMARY bias source**

| Role | N | Pred Min | Actual Min | Bias | Ratio |
|------|---|----------|------------|------|-------|
| all | 48,430 | 25.3 | 27.8 | -2.50 | 0.960 |
| starter | 23,448 | 31.6 | 32.1 | **-0.55** | **1.024** |
| rotation | 24,272 | 19.7 | 24.1 | **-4.37** | **0.897** |
| spot | 710 | 12.5 | 15.7 | -3.17 | 0.982 |

Starters nearly unbiased. Rotation minutes under-predicted by 18% — caused by phantom DNP players absorbing minutes budget in model (engine projects 10/team, only ~7 play).

**3. Rates: secondary issue (rotation/spot only)**

| Role | Adj Rate | Actual Rate | Δ% |
|------|----------|-------------|-----|
| starter | 0.5628 | 0.5620 | **+0.1%** |
| rotation | 0.4099 | 0.4325 | -5.2% |
| spot | 0.3670 | 0.4356 | -15.7% |

Starter rates unbiased. Rotation/spot under-predicted — likely selection effect (players who play score better than profile average).

**4. PTS bias — the bottom line**

| Role | N | Pred PTS | Actual PTS | Bias | Δ% |
|------|---|----------|------------|------|----|
| all | 48,430 | 12.9 | 14.2 | -1.37 | -9.2% |
| starter | 23,448 | 17.9 | 18.3 | **-0.34** | **-1.9% ✓** |
| rotation | 24,272 | 8.2 | 10.5 | -2.34 | -21.9% |
| spot | 710 | 4.6 | 6.5 | -1.91 | -29.2% |

**Starters: -1.9% — at ±2% target.** Rotation/spot bias driven by minutes under-allocation (phantom DNP redistribution).

**5. Stable across seasons (no drift)**

| Season | N | PTS_scale ratio | Min_ratio | Rate_ratio |
|--------|---|-----------------|-----------|------------|
| 2023-24 | 16,785 | 0.908 | 0.969 | 0.974 |
| 2024-25 | 16,114 | 0.906 | 0.957 | 0.975 |
| 2025-26 | 15,531 | 0.907 | 0.953 | 0.977 |

(PTS_scale ratio = mean(pred_pts) / mean(actual_pts), corrected from per-player ratio)

**Diagnosis**: The 9% PTS under-projection decomposes as:
- **~4% from minutes** — rotation players get too few minutes because phantom DNPs take share of 240-min budget
- **~2.5% from rates** — rotation/spot per-min rates under-predicted (selection bias: players who play > profile average)
- **~2.5% from interaction** — compounding of minutes × rate biases

**Fix path**: R3 inactives → remove phantom DNPs from pool → reallocate minutes → should close ~4% minutes gap + may improve rates as usage adjusts. R8 (pace fix) NOT needed. R11 DONE: inactives + absence filter + soft allocation → OTHER 14.6% → 2.1%. R13 = reconciliation recheck on fixed pool.

### R7 Results — Target Audit (3,677 games, 3 seasons)

Script: `/tmp/r7_target_audit.py`, data: `/tmp/r7_market_audit.csv`, `/tmp/r7_merged_audit.csv`.

**Market-implied team totals vs actual team pts:**

| Metric | Value |
|--------|-------|
| Mean market-implied total | 113.08 |
| Mean actual team pts | 113.31 |
| Bias (market - actual) | **-0.23** |
| MAE | 8.95 |
| Correlation | 0.486 |

Market is well-calibrated. No spread sign error, no formula error.

**Model team totals:** Exactly match market anchor (by design: `_compute_team_totals()` sets team_total = market-implied). Delta = 0.00.

**Named-player sum / actual team pts:** 1.053 — confirms phantom DNP over-projection (+5.3% of team scoring allocated to players who don't play).

**OTHER share lookup table:** Engine's `_OTHER_PTS_SHARE` exactly matches R2 empirical data for N=5 through N=10. No calibration fix needed.

### R10 Results — Pool Mismatch Audit (2,460 team-games, 2023-24)

Per team-game audit classifying why projected players don't have actuals and why actual players weren't projected. Script: `/tmp/r10_pool_audit.py`.

**Critical bug found and fixed**: `_load_inactives_lookup()` in `nba_ablation.py` stored game_ids in 10-digit format ("0022300001") but lookups used 8-digit format ("22300001"). Fix: `str(int(gid))` normalisation. This also **invalidated R9** (zero inactives were excluded).

**Overview:**

| Metric | Value |
|--------|-------|
| Avg projected/team | 9.6 |
| Avg matched to actuals | 6.9 (72%) |
| Avg actually played | 9.8 |
| Avg projected-no-actuals | 2.7 |
| Avg played-not-projected | 2.8 |

**(a) Projected but no actuals (2.7/team-game, 6,579 total):**

| Category | Count | Pct | Notes |
|----------|-------|-----|-------|
| inactive_not_in_box | 3,661 | 55.6% | Confirmed inactives (BoxScoreSummaryV2). 0 actual PTS. |
| absent_unknown | 2,899 | 44.1% | Not in box score AND not in inactives DB. G-League assignments, personal leave, pre-game scratches, inactives data gaps. 0 actual PTS. ~1.2/team-game. |
| dnp_cd_0min | 10 | 0.2% | In box score with 0 minutes, not on inactives list. |
| played_other_team | 9 | 0.1% | Recently traded — engine has old team. 66 total PTS. |

**(b) Played but not projected (2.8/team-game, 6,990 total):**

| Category | Count | Pct | Total PTS | Avg PTS | Notes |
|----------|-------|-----|-----------|---------|-------|
| beyond_roster_cap | 5,200 | 74.4% | 31,498 | 6.1 | Right team, MPG ≥ 10, but cut by top-10/cum-270. **PRIMARY driver of OTHER.** |
| below_mpg_threshold | 1,124 | 16.1% | 4,865 | 4.3 | In engine but MPG < 10. |
| too_few_games | 363 | 5.2% | 1,732 | 4.8 | Not in engine profiles (< 3 games before cutoff). Rookies, new signings. |
| wrong_team_in_engine | 303 | 4.3% | 2,395 | 7.9 | Traded players, engine has old team. **Highest avg PTS — fixable.** |

**Corrected OTHER share: 14.6%** (vs 16.7% from R2). Avg 16.5 PTS/team-game from played-not-projected players.

**Key findings:**
1. **OTHER is structural, not data quality.** 74.4% of played-not-projected are qualified players cut by the top-10/cum-270 roster cap. Fixing trades (4.3%) and new players (5.2%) only reduces OTHER from ~14.6% to ~13.2%.
2. **To reach OTHER = 3-6%, must expand roster cap** from 10 to ~13. This captures the 5,200 `beyond_roster_cap` players (6.1 PTS avg) but adds projection noise for low-minute players.
3. **Projected-no-actuals are 56% confirmed inactives** (with game-day data) and 44% unavoidable uncertainty (G-League, rest, etc.). With game-day inactives: 1.5/team-game predictable, 1.2/team-game not.
4. **wrong_team_in_engine** (303 cases, 7.9 PTS avg) is the highest-ROI roster fix — ~1.0 PTS/team-game from traded players on wrong team.

**R11 scope (revised):** Fix wrong-team assignment + consider roster cap expansion. Note: R10 audit uses slightly tighter roster selection (min 5 / cum_mpg+p > 270) vs engine (min 8 / cum_mpg >= 270), giving 9.6 avg vs R1's 9.9. Categories are representative.

### R11 Results — Pool Construction Fix (2,460 team-games, 2023-24)

Three fixes applied: (1) trade detection via `_detect_trades()` in `build_engine_asof()` + team assignment from `trade["to_team"]`, (2) soft allocation (MPG >= 3, no hard top-10/cum-270 cap), (3) consecutive-absence filter (N >= 7 games absent → exclude). Script: `/tmp/r11_verify.py`.

**Overview (R10 → R11):**

| Metric | R10 | R11 | Δ |
|--------|-----|-----|---|
| Avg pool size/team | 9.6 | **11.0** | +1.4 |
| Avg matched to actuals | 6.9 (72%) | **9.4 (85%)** | +2.5 |
| Avg actually played | 9.8 | 9.8 | — |
| Avg projected-no-actuals | 2.7 | **1.6** | -1.1 |
| Avg played-not-projected | 2.1 | **0.4** | -1.7 |
| **OTHER PTS share** | **14.6%** | **2.1%** | **-12.5pp** |

**Exclusion stats:**
- Inactives excluded: 2.3/team-game (confirmed via BoxScoreSummaryV2)
- Absence-filtered: 2.6/team-game (7+ consecutive games absent)
- Trades detected: ~48/date (cumulative 7,722 across 160 dates)

**(a) Projected but no actuals (1.6/team-game, 3,840 total):**

| Category | Count | Pct | Notes |
|----------|-------|-----|-------|
| absent_unknown | 3,781 | 98.5% | Pre-game scratches, G-League call-downs, rest. Only knowable with game-day info. |
| dnp_cd_0min | 48 | 1.2% | In box score with 0 minutes. |
| played_other_team | 11 | 0.3% | Recently traded. |

The inactives categories (55.6% in R10) are gone — now excluded pre-roster-selection.

**(b) Played but not projected (0.4/team-game, 880 total):**

| Category | Count | Pct | Total PTS | Avg PTS |
|----------|-------|-----|-----------|---------|
| too_few_games | 363 | 41.2% | 1,732 | 4.8 | New players, <3 games before cutoff. |
| wrong_team_in_engine | 303 | 34.4% | 2,395 | 7.9 | Offseason/recent trades with no post-trade games in log. |
| excluded_but_played | 199 | 22.6% | 1,510 | 7.6 | Absence filter false positives (3.1% FPR). |
| below_mpg_threshold | 15 | 1.7% | 60 | 4.0 | MPG < 3. |

**OTHER share: 2.1%** (target was 3-6%). Mean 2.3 PTS/team-game from unmodeled players. Median 0.0%, P75 0.0%. The `beyond_roster_cap` category (74.4% in R10, 5,200 cases) is **eliminated** by soft allocation.

**Wrong-team cases (303) unchanged from R10:** These are offseason trades where the player has no post-trade games before cutoff — `_detect_trades()` can't detect team changes with only one team in the log. Requires transaction data (not available in ablation). In production, injury watcher + live rosters handle this.

**Absence filter performance:** 6,396 total exclusions, 199 false positives (3.1% FPR, 96.9% TPR). The 199 false positives are players who returned after 7+ game absence. Contributes ~0.6 PTS/team-game to OTHER.

**Code changes:**
- `nba_matchup_engine.py`: `_build_player_profiles()` uses `trade["to_team"]` for team assignment; `project_game(soft_allocation=True)` includes all MPG >= 3 players
- `nba_ablation.py`: `build_engine_asof()` calls `_detect_trades()` before profiles; `_precompute_absence_data()` + `_find_absent_players()` for N >= 7 consecutive absence filter; `run_ablation(soft_allocation=True)` wires all fixes

### R11 Bias Decomposition (R11-fixed pool, 2023-24)

Bias decomposition re-run on R11 pool (trade detection + soft allocation + absence filter). Script: `/tmp/r11_bias_decomp.py`. Comparison to R6 baseline (pre-fix).

**1. PTS Bias by Role (R6 → R11):**

| Role | N | Pred PTS | Actual PTS | R11 Bias | R6 Bias | Δ |
|------|---|----------|------------|----------|---------|---|
| all | ~40K | 12.7 | 13.1 | **-0.44** | -1.37 | **68% reduction** |
| starter | ~19K | 17.0 | 17.1 | **-0.09** | -0.34 | Within ±2% ✓ |
| rotation | ~18K | 9.3 | 9.9 | -0.63 | -2.34 | 73% reduction |
| spot | ~3K | 5.2 | 5.9 | -0.67 | -1.91 | 65% reduction |

**2. Minutes by Role (R6 → R11):**

| Role | Pred Min | Actual Min | R11 Bias | R6 Bias | Ratio |
|------|----------|------------|----------|---------|-------|
| all | 23.9 | 24.9 | **-0.95** | -2.50 | 0.962 |
| starter | 30.8 | 31.1 | **-0.32** | -0.55 | 0.990 |
| rotation | 18.7 | 19.9 | -1.19 | -4.37 | 0.940 |
| spot | 10.2 | 11.8 | -1.59 | -3.17 | 0.866 |

**3. Rates by Role — SELECTION BIAS SOLVED:**

| Role | Pred Rate | Actual Rate | R11 Δ% | R6 Δ% |
|------|-----------|-------------|--------|-------|
| starter | 0.5530 | 0.5520 | **+0.2%** | +0.1% |
| rotation | 0.4948 | 0.4982 | **-0.7%** | -5.2% |
| spot | 0.4799 | 0.4860 | **-1.3%** | -15.7% |

All roles within ±1.3%. The R6 rotation (-5.2%) and spot (-15.7%) rate biases were entirely a pool-composition artifact: the old pool cut qualified players who then played, making "actuals" a positively selected subset.

**4. Bias Decomposition (PTS_bias ≈ min_component + rate_component + interaction):**

| Role | Total Bias | Min Component | Rate Component | Interaction |
|------|-----------|---------------|----------------|-------------|
| all | -0.44 | **-0.43** | +0.08 | -0.00 |
| starter | -0.09 | -0.18 | +0.09 | +0.00 |
| rotation | -0.63 | **-0.59** | -0.07 | +0.01 |
| spot | -0.67 | **-0.77** | -0.07 | +0.02 |

**Key finding: remaining bias is virtually ALL minutes.** Rate component ≈ 0 for all roles. R12's original scope ("rate conditional on minutes") is no longer needed — the rate bias was a selection artifact, now fixed.

**5. Phantom Minutes (projected to players who didn't play):**

| Metric | Value |
|--------|-------|
| Total phantom projections | 3,792 |
| Team-games with phantoms | 1,649 / 2,460 (67%) |
| Mean phantom min / ALL team-games | **20.2** (8.4% of 240) |
| Mean phantom min / affected team-games | 28.4 |
| P90 phantom min / affected team-games | 57.7 |

By role:
| Role | N phantom | Mean pred min | Mean pred PTS |
|------|-----------|---------------|---------------|
| starter | 323 | **30.3** | 16.1 |
| rotation | 1,314 | 16.2 | 7.8 |
| spot | 2,155 | 8.6 | 4.0 |

**20.2 min > user's 5-8 min target.** Dominated by unexpected starter/rotation absences (not soft-allocation floor issue — spot phantoms average only 8.6 min). These are `absent_unknown` cases: pre-game scratches, G-League assignments, rest days that only game-day inactives data can resolve. Tightening the MPG floor would miss the starters (30.3 min avg). Mitigation path: game-day inactives API in production.

**6. Wrong-Team Roster Fix:**

Backtest team assignment now uses three-tier fallback:
1. `_detect_trades()` → `trade["to_team"]` (mid-season trades with post-trade games)
2. `_roster_teams` lookup from `player_roster` table → restricted to `_prior_only_pids` (players with no current-season games before cutoff date)
3. Most recent MATCHUP string → `last_matchup[:3]` (all other players)

Restriction to `_prior_only_pids` prevents lookahead: `player_roster` is sourced from `LeagueDashPlayerStats` which is end-of-season data. For mid-season traded players WITH current-season games, the MATCHUP string already captures the correct (pre-trade) team for pre-trade dates and `_detect_trades()` handles post-trade dates.

Production: `CommonTeamRoster` endpoint needed for real-time team assignment (not yet cached). `injury_watcher.py` is NFL-only — no NBA team-change detection exists. Add to production checklist.

### R11 Production Notes

**Absence filter override (item 4):** In production, an injury-report status of **available** or **probable** must override the N≥7 consecutive-absence rule. This catches returning players — the 199 false positives (3.1% FPR, 7.6 PTS avg) in the backtest are exactly this case. Requires injury report integration for NBA (not yet built; `injury_watcher.py` is NFL-only).

**too_few_games / rookie priors (item 5):** 363 cases (41.2% of played-not-projected in R11), 4.8 PTS avg. These are rookies and new signings with <3 games before the cutoff date — the engine has no profile for them. Future work: rookie priors from draft slot / prior league / preseason data. Kept on OPEN_ITEMS for B5+ scope.

**CommonTeamRoster (production):** Not yet cached. Needed for real-time team assignment (covers offseason trades that `_detect_trades()` can't see from game logs alone). Currently 303 wrong-team cases in backtest from this gap.

### R12 — SKIP (Rate Bias Resolved)

R12 was originally scoped as "rate conditional on minutes" to address the 2.5% rate component from R6 decomposition. The R11 bias decomposition shows **rate bias is gone** (all roles within ±1.3%) — it was entirely a selection artifact from the thin pool. User decision: SKIP.

### PROD-3 — Minutes Diagnostic (Phantom Starters + OT Exclusion)

Script: `/tmp/r11_minutes_diagnostic.py`. 23,208 player-game records (2023-24).

**Key finding: clean subset min bias halved (-0.95 → -0.45), PTS bias near zero (-0.18).**

| Filter | N | Min Bias (all) | Min Bias (starter) | PTS Bias (all) |
|--------|---|----------------|--------------------|-|
| ALL (no filter) | 23,208 | -0.96 | -0.34 | -0.44 |
| Excl OT only | 22,142 | -0.78 | **-0.02** | -0.37 |
| Excl phantom-starter only | 20,882 | -0.63 | -0.15 | -0.26 |
| **Excl BOTH (clean)** | **19,914** | **-0.45** | **+0.17** | **-0.18** |

Isolated contributors:
- **Phantom-starter TGs** (non-OT): -3.75 min bias, rotation gets -4.58 absorbing missing starter load
- **OT TGs** (no phantom starter): -4.21 min bias, starters play 38.5 actual vs 32.3 projected

**Verdict: residual bias is availability artifact + OT modeling gap, not a minutes model deficiency.** The 20.2 min/team-game phantom allocation is dominated by unexpected absences. Game-day injury exclusion (PROD-2) handles this in production. OT is handled by Sim Step 2 (S2-3: overtime draw).

### PROD-1 — CommonTeamRoster Cache

`_fetch_roster_teams()` added to `NBAMatchupEngine.load()` (nba_matchup_engine.py). Fetches `CommonTeamRoster` from nba_api for all 30 teams. 521 players mapped to `{player_id: team_abbr}`.

- Tier-2 team assignment: `_roster_teams[pid]` used in `_build_player_profiles()` when no trade detected and player on roster
- Cached in engine SQLite DB (`roster_teams` table), refreshed with 24h cache TTL
- `report_roster_changes()` logs any mismatches at scout startup
- At 3 GP into 2026-27: 0 mismatches (roster correctly authoritative for all 338 profiled players on current rosters)

### PROD-2 — NBA Injury Report

`fetch_injury_report()` + `get_excluded_players()` added to `NBAMatchupEngine` (nba_matchup_engine.py).

**Sources:**
1. **Sleeper** (primary): `api.sleeper.app/v1/players/nba` — free, no auth, has Out/IR/DTD/GTD statuses, team abbreviations, body part, notes
2. **ESPN** (secondary): `site.api.espn.com/.../nba/injuries` — free, has Out/Day-To-Day, no team abbreviation. Logs conflicts vs Sleeper.

**Current report (2026-10-27):**
- 91 total injuries (57 Out/IR, 34 DTD/GTD, 8 conflicts)
- 28 profiled players with Out/IR status → excluded from projections
- Exclusions passed to `project_game()` and B5 simulation via `scout_game()`

**Absence filter override:** Players NOT in the injury report are implicitly available — overrides the N≥7 consecutive-absence filter for returning players. `get_available_overrides()` returns the set.

**Code changes:**
- `nba_matchup_engine.py`: `fetch_injury_report()`, `get_excluded_players()`, `get_available_overrides()`, ESPN team ID map
- `scout/nba_player_props.py`: `_get_engine()` reports roster + injury stats; `scout_game()` fetches injuries and passes exclusions to `_engine_game_projection()` and `_engine_game_simulation()`

### R9 Results — INVALIDATED (format bug)

**Bug**: `_load_inactives_lookup()` loaded game_ids in 10-digit format ("0022300001") but `run_ablation()` looked up 8-digit format ("22300001"). Zero inactives were ever excluded. R9 ≡ R4 on the 2-season subset.

**"Baseline improvement" was sample composition**: R9 CRPS 3.748 vs R4 CRPS 3.805 reflects running on 2 seasons vs 4, not inactives exclusion. The numbers below are R4 on a 2-season subset, not a valid inactives test.

**Fix applied**: `nba_ablation.py:1208` now normalises game_ids with `str(int(gid))`. R13 will be the first valid inactives-fixed reconciliation test.

~~Walk-forward ablation on 2022-23 and 2023-24. Numbers below are identical to R4 on the same subset:~~

**CRPS by config × stat (lower = better):**

| Config | PTS | REB | AST | PRA |
|--------|-----|-----|-----|-----|
| r4_norecon (baseline) | **3.748** | **1.459** | **1.058** | **4.855** |
| r4_recon_nofix (legacy) | 3.766 | 1.478 | 1.066 | 4.917 |
| r4_recon_poolfix (full) | 3.814 | 1.478 | 1.066 | 4.984 |
| r4_recon_poolfix_50 (half) | 3.763 | 1.464 | 1.059 | 4.895 |

No-recon best across ALL stats.

**ΔCRPS vs no-reconcile baseline (positive = variant WORSE, `*` = CI excludes zero):**

| Config | PTS | REB | AST | PRA |
|--------|-----|-----|-----|-----|
| legacy (no pool fix) | +0.018* | +0.019* | +0.008* | +0.062* |
| pool fix (full) | **+0.065*** | +0.019* | +0.008* | **+0.129*** |
| pool fix (50%) | +0.015* | +0.004* | +0.001* | +0.040* |

ALL variants significantly worse across ALL stats. Pool fix remains worst.

**PTS bias by role × config (pred - actual):**

| Config | All | Starter | Rotation | Spot |
|--------|-----|---------|----------|------|
| no reconcile | **-0.96** | **-0.30** | **-1.55** | **-1.16** |
| legacy reconcile | -1.28 | -0.77 | -1.75 | -1.27 |
| pool fix (full) | -1.86 | -1.61 | -2.11 | -1.48 |
| pool fix (50%) | -1.41 | -0.96 | -1.83 | -1.32 |

No-recon has lowest bias magnitude in ALL roles. Reconciliation pushes projections DOWN when model already under-projects.

### R1+R2 Empirical Results (9,740 team-games, 4 seasons)

**Named pool vs OTHER:**

| Metric | Mean | P10 | P25 | P50 | P75 | P90 |
|--------|------|-----|-----|-----|-----|-----|
| Named players with actuals | 7.1 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 |
| Named actual min | 191.5 | 142 | 173 | 199 | 219 | 231 |
| Named pts / team pts | 0.830 | 0.627 | 0.766 | 0.871 | 0.942 | 0.983 |
| OTHER minutes | 48.5 | 9.0 | 21.0 | 41.0 | 67.0 | 98.0 |
| OTHER points | 19.1 | 2.0 | 7.0 | 15.0 | 27.0 | 42.0 |

**OTHER by named pool size (N=number of named players with actuals):**

| N_named | Count | OTHER_min | OTHER_pts | OTHER_%pts |
|---------|-------|-----------|-----------|------------|
| 5 | 731 | 91.5 | 36.0 | 32.2% |
| 6 | 1,605 | 65.4 | 24.8 | 22.1% |
| 7 | 2,487 | 43.5 | 16.8 | 14.7% |
| 8 | 2,600 | 27.8 | 11.1 | 9.6% |
| 9 | 1,421 | 17.4 | 7.2 | 6.2% |
| 10 | 255 | 12.5 | 5.6 | 4.7% |

**OTHER growing over time (engine pool shrinking relative to active rosters):**

| Season | N_named | OTHER_min | OTHER_%pts |
|--------|---------|-----------|------------|
| 2022-23 | 7.8 | 33.9 | 12.2% |
| 2023-24 | 7.1 | 45.7 | 15.7% |
| 2024-25 | 6.8 | 52.1 | 18.4% |
| 2025-26 | 6.6 | 62.1 | 21.4% |

**Implication for reconciliation:** Engine currently reconciles named-pool points to full `market_total`. Pool fix reconciles to `market_total × (1 - OTHER_share)`. Tested in R4 — did not help (see below).

### R4 Results — Pool-Fixed Reconciliation Ablation (1,929,732 preds, 4,863 games)

**CRPS by config × stat (lower = better):**

| Config | PTS | REB | AST | PRA |
|--------|-----|-----|-----|-----|
| r4_norecon (baseline) | 3.805 | 1.465 | 1.074 | 4.905 |
| r4_recon_nofix (legacy) | 3.832 | 1.484 | 1.083 | 4.981 |
| r4_recon_poolfix (full) | 3.889 | 1.484 | 1.083 | 5.059 |
| r4_recon_poolfix_50 (half) | 3.826 | 1.469 | 1.075 | 4.954 |

**ΔCRPS vs no-reconcile baseline (positive = variant WORSE, `*` = CI excludes zero):**

| Config | PTS | REB | AST | PRA |
|--------|-----|-----|-----|-----|
| legacy (no pool fix) | +0.027* | +0.019* | +0.009* | +0.076* |
| pool fix (full) | **+0.084*** | +0.019* | +0.009* | **+0.154*** |
| pool fix (50%) | +0.021* | +0.005* | +0.002* | +0.049* |

**Pool fix made PTS WORSE**, not better (+0.084 vs +0.027 for legacy). The fix reduced the reconciliation target (to 95.3% of market total for 10 named players), but this over-deflated projections for players who actually played, increasing negative bias.

**PTS bias (pred - actual):**

| Config | All | Starter | Rotation | Spot |
|--------|-----|---------|----------|------|
| no reconcile | -1.06 | -0.27 | -1.79 | -1.05 |
| legacy reconcile | -1.48 | -0.88 | -2.06 | -1.14 |
| pool fix (full) | **-2.06** | **-1.71** | **-2.42** | -1.36 |
| pool fix (50%) | -1.56 | -0.99 | -2.11 | -1.20 |

**Root cause**: The engine already under-projects (PTS scale 0.921 without reconcile). Reconciliation pulls predictions further DOWN toward a team total that is split among more players than actually play. The pool fix compounds this by reducing the target even further.

**R5 Decision: KILL RECONCILIATION (R4 validated; R13 = final recheck after pool fix).**

Two independent tests confirm reconciliation hurts:
1. **B4 ablation**: ΔCRPS +0.025 PTS, +0.075 PRA (full model vs full_no_reconcile)
2. **R4 pool-fixed**: ΔCRPS +0.027 to +0.084 PTS (all variants worse, CIs exclude zero)

~~R9 was intended to close the last escape hatch but was invalidated by a format bug (zero inactives excluded).~~ R13 will re-run after pool construction fixes (R11) to verify KILL with correct inactives exclusion.

**Mechanism**: Reconciliation scales named-player projections DOWN to match a team total that is intentionally lower than their sum (because unnamed bench players will also score). R10 confirms OTHER = 14.6% is structural (74% from roster-cap overflow, not data quality). The OTHER share lookup handles this implicitly — reconciliation double-corrects.

**Bugs found and fixed during build:**
- `pull_team_opp_stats()` used `per_mode_detailed="PerGame"` but engine expects Totals (divides by GP). Fixed pull + added heuristic detection in `build_engine_asof()`.
- `_compute_league_averages()` was never called in ablation path, leaving `_LG_AVG` empty → all defensive factors ≈ 0. Fixed.
- Cached player game logs lack `PLAYER_NAME` column. Fixed with roster lookup in `build_engine_asof()`.
- Cached DataFrames lack `GAME_DATE_DT`. Fixed with `_prepare_game_log()` helper.

### Historical NBA Line Inventory (for B4)

**IMPORTANT: Line source mismatch between backtest and production.**

| Source | Table | Seasons | Games | Spread | Total | Line Type |
|--------|-------|---------|-------|--------|-------|-----------|
| ESPN (historical.db) | `betting_lines` | 2022-23 | 1,230 | 1,230 | 1,229 | **ESPN near-close** (scraped post-game, ~closing line but not bookmaker-specific) |
| ESPN (historical.db) | `betting_lines` | 2023-24 | 1,230 | 1,230 | 1,230 | ESPN near-close |
| ESPN (historical.db) | `betting_lines` | 2024-25 | 1,225 | 1,224 | 1,225 | ESPN near-close, Oct 22 – Apr 13 |
| ESPN (historical.db) | `betting_lines` | 2025-26 | 1,225 | 1,224 | 1,224 | ESPN near-close, Oct 21 – Apr 12 |
| TheOddsAPI (bets.db) | `fixtures` | 2026-27 | 71 | 71 | 71 | **Live FanDuel pre-game** (fetched hours before tip, book-specific devig) |

**Backtest uses ESPN near-close lines** (close_spread, close_total): not true bookmaker-specific closing lines, but close approximation scraped from ESPN post-game. No open-spread data (open_spread = NULL).

**Production uses live FanDuel pre-game lines**: fetched from TheOddsAPI, Shin-devigged, book-specific (FanDuel > DraftKings > first available). Lines are pre-game, not closing.

**Implication**: B4 backtest results apply to a near-close anchor. Production anchors to pre-game lines, which can differ by 0.5–2 points from close. Factor ablation conclusions (which factors help/hurt) transfer because they are relative, but absolute calibration numbers may shift.

## B5: Joint Draws (SGP + Cross-Player Correlation)

Depends on B4-R (reconciliation re-test with pool fix). Execution order below.

### Step 1: Reconciliation verdict — COMPLETE (R13 DONE)

R1–R7 + R10 + R11 + R11-B + R12(SKIP) + PROD-1/2/3 + **R13** done. R11 fixed pool: OTHER 14.6% → 2.1%.

**R13 results** (2023-24, N=23,212 player-games, 1,230 games, game-clustered CIs):

Q1 — Reconciliation: ΔCRPS vs `def_blow_norecon` (negative = improvement):

| Variant | PTS all | REB all | AST all | PRA all |
|---------|---------|---------|---------|---------|
| base+defense (100%) | -0.024*** | +0.012*** | -0.006*** | -0.006 |
| recon_nocap (100%) | -0.024*** | +0.012*** | -0.006*** | -0.006 |
| **recon_50pct (50%)** | **-0.026***  | **+0.001** | **-0.005***  | **-0.021***  |

50% partial reconciliation is the clear winner:
- PTS: -0.026 CRPS (starters -0.048), significant → scoring benefits from team-total pull
- REB: +0.001, NOT significant → no REB harm (unlike 100% which hurts REB +0.012***)
- AST: -0.005, significant → assists also improve
- PRA: -0.021 (starters -0.046), significant → composite benefit
- FG3M/STL/BLK: ΔCRPS=0.000 (not reconciled, unaffected)

Q2 — Absence rate model: ΔCRPS = 0.000 everywhere. **CONFIRMED DEAD on corrected pool.**

**Decision: reconcile=True with 50% pull in production.** This reverses the earlier R4/R5/R9 kill (which used the thin 14.6% OTHER pool). The 50% pull captures PTS/AST/PRA improvement while neutralizing the REB degradation that full reconciliation causes. `absence_adj` KILLED (confirmed on both pools).

### Step 2: Sim shared state per draw (`nba_game_sim.py`)

| # | Item | Status | Notes |
|---|------|--------|-------|
| S2-1 | Final margin draw | **DONE** | `margin ~ N(-spread, 15.8)`. σ=15.8 measured from 4,668 regulation games (4 seasons). Kurtosis +0.19 (Normal OK). |
| S2-2 | Team points draw | **DONE** | `game_total ~ N(market_total, 19.4)`, per-team pts = (total±margin)/2. Dampened per-team scoring multiplier (~5.5% CV) applied to PTS/AST/FG3M rates. NOT reconciliation — soft correlation only. Corr(home,away)≈+0.20. |
| S2-3 | Overtime | **DONE** | P(OT)=5.12% (252/4,920 games). Blowout draws can't be OT. Mean-subtracted OT minute shifts: starters +4, rotation +1, spot 0. Team minutes target = 245 for OT draws. |
| S2-4 | 240-min constraint + blowout swaps | **DONE** | Σ=240 (245 for OT) enforcement with ±30% cap. Per-draw mean-subtracted blowout shifts: starters -2.3, rotation +4.0, spot +7.0. P(blowout)=32.6% at |margin|≥15. |
| S2-5 | Probabilistic availability | **DONE** | Void rules (own prop conditional on playing), absence-impact redistribution (measured with/without), forward log, SGP void-leg repricing. See below. |

### S2-5: Probabilistic Availability — DONE (with void rules + absence redistribution)

**Measured base rates** from R11 pool (2023-24, N=26,959, before injury report integration). **UPPER BOUNDS** for "no designation" players — include absences an injury report would have flagged:

| Role | N | P(plays) | P(unexp. absence) | Note |
|------|---|----------|--------------------|------|
| starter | 8,623 | ≤0.963 | ≤3.7% | Upper bound |
| rotation | 11,963 | ≤0.891 | ≤10.9% | Upper bound |
| spot | 6,373 | ≤0.658 | ≤34.2% | Upper bound |

Reason breakdown for non-plays (N=3,800): absent_unknown 98.6%, dnp_cd 1.3%, wrong_team 0.1%.

**Injury status rates** — HAND-SET from literature, NOT measured. Replace with `nbainjuries` historical backfill or forward log:

| Status | P(plays) | Provenance |
|--------|----------|------------|
| Out | 0.00 | Definitional |
| IR | 0.00 | Definitional |
| Doubtful | 0.20 | HAND-SET — literature |
| Questionable | 0.65 | HAND-SET — literature |
| GTD | 0.50 | HAND-SET — Sleeper label |
| DTD | 0.70 | HAND-SET — Sleeper label |
| Probable | 0.95 | HAND-SET — literature |
| Active | 1.00 | Definitional |

**Historical backfill available**: `nbainjuries` package (installed, v1.1.1) parses official NBA injury report PDFs from the CDN. Confirmed accessible 2020-21 through 2026-27 (~170 game-days/season × 80-150 rows per PDF). Can measure P(plays|status) properly by joining vs `BoxScoreSummaryV2` inactives. **Not yet done** — queued.

**Void rules** (`SimulationResult`):
- `p_over()`/`p_under()`/`marginal_stats()` condition on draws where player is available. A player's own prop is voided if they don't play → only count draws where they play. Verified: healthy star P(over low line) = 0.987, NOT capped at 0.963.
- `p_joint()`/`p_parlay()` condition on ALL referenced players playing (standard SGP pricing).
- `p_parlay_void(legs, book="fanduel", leg_odds=None)` is book-aware:
  - **FanDuel**: voided legs removed, surviving legs repriced as shorter parlay. With `leg_odds`, estimates sub-parlay payout via measured correlation multiplier k × Π(surviving leg odds). Returns `ev_void_scenarios` with probability-weighted EV across void configurations.
  - **DraftKings / other books**: any voided leg → entire parlay refunded (no repricing). `p_void_adjusted = p_full`.

**Absence redistribution** (per draw):
- When player absent in a draw, teammates get measured delta_min from minutes model's with/without patterns (shrunk, capped at +8 min).
- Built as (P×P) impact matrix, applied as vectorized matmul: `delta = absent_float @ impact_matrix.T`.
- Proportional 240 rescaling handles residual gap after measured redistribution.
- Falls back to proportional-only when absence impacts unavailable (backtest without minutes model).

**Production source order** (`fetch_injury_report()`):
1. **Official NBA injury report PDF** (via `nbainjuries`) — authoritative, released ~5PM ET game days. Tries today then yesterday.
2. **Sleeper** — adds players not on PDF (late scratches, post-PDF updates). Conflicts logged with `SOURCE CONFLICT` warning.
3. **ESPN** — tertiary, adds remaining players not in PDF or Sleeper.
Source field tracks provenance: `nba_pdf`, `sleeper`, `espn`. PDF wins when present.

**Forward log**: `nba_injury_forward_log` table in historical.db. Logged on each `fetch_injury_report()` call. Columns: snapshot_utc, game_date, player_name, player_id, team, status, source, body_part, role, mpg, played (backfill), actual_minutes (backfill). Re-estimate P(plays|status) in-season when N≥30 per status bucket.

### Step 3: Count stats (Poisson/NegBin)

| # | Item | Status | Notes |
|---|------|--------|-------|
| S3-1 | Dispersion fit by stat × role | **DONE** | STL/BLK near-Poisson (OD 1.02-1.05), REB/AST/FG3M mild (1.12-1.46), PTS over-dispersed (2.79). Rate-based Var explains 80-100% — extracted rate_std already absorbs Poisson shot noise. |
| S3-2 | Draw counts given minutes × rate | **DONE** (implementation) | Gamma-Poisson draws deployed 2026-10-30 for STL/BLK/FG3M in `GameSimulator.run()` (lines 1079-1129). Gamma rate draw + Poisson count draw = NegBin marginal. **Verification pending**: ablation PIT can't validate this (uses Normal-model predictions, not sim draws). Need forward data from `nba_forward_preds` (0 rows as of 2026-11-08; collection started 2026-11-06). Run sim-based PIT when N≥100 graded forward games (~mid-Dec 2026). |

### Step 4: Validation vs reality (4 seasons of box scores)

| # | Item | Status | Notes |
|---|------|--------|-------|
| S4-1 | Marginal calibration | **DONE** | PIT on 162,484 predictions (def_blow_recon_50pct, 2023-24). PTS/REB/AST starters well-calibrated (KS 0.04–0.07). STL/BLK badly miscalibrated (KS 0.19–0.23) — Normal approximation wrong for discrete low-count stats; NegBin or Poisson CDF needed. Empirical same-team residual correlations near zero: PTS r=-0.016±0.010, REB r=+0.006±0.008, AST r=-0.023±0.009. These are calibration TARGETS for the sim. |
| S4-2 | Joint calibration by combo type | **DONE** | Empirical k factors measured from 162,484 predictions across 1,230 games. Key findings: same-player PTS+REB k=1.20 (shared minutes + engagement), teammate PTS k=1.00 (independent), opposing PTS k=1.11 [1.00,1.22] (wide CI). **Sim changes**: (1) per-player engagement draw (eng_cv=0.15, PTS=1.0, REB=0.8, AST=0.0) matches PTS+REB k. (2) Scoring damp 0.50→0.30. (3) AST removed from team scoring mult. **Residual gaps**: teammate 3-leg PTS sim 1.05 vs emp 0.99 (shared pace slightly over-correlates); opposing PTS sim 1.01 vs emp 1.11 (no mechanistic fix, CI wide). |

### Step 5: B5 proper — calibration & comparison vs market

| # | Item | Status | Notes |
|---|------|--------|-------|
| S5-1 | Two-stage calibration | **DONE** | PIT re-verified post engagement draw. Holdout validation (H1→H2, H2→H1) showed all Normal corrections are one-way or dead → REMOVED. Gamma-Poisson for discrete stats is the only surviving structural fix. See S5-1 results below. |
| S5-2 | Comparison vs devigged market | **ACTIVE — first full capture 2026-11-15** | `_persist_nba_forward_preds()` rewritten: stores sim P(over/under) at EVERY market-posted line (main + all alt rungs), with best_price, best_book, devig_implied, devig_method, n_books. Queries `player_props` for all lines per game via `_get_market_lines_for_game()`. Two-sided lines → multiplicative devig; one-sided alts → borrowed_margin (from same player/stat/book main-line overround, fallback to stat/book median, then global median ~1.05). `_pick_best_threshold` replaced by EV-based selection (`_ev_upgrade_props`, odds band [1.667, 3.50], shadow-only). **First full capture: 2026-11-15** — 4,602 rows (9 games, 117 players), main-line join rate 100% (620/620 two-sided devig), alt lines 3,953 (borrowed_margin), fallback 29 (no_market=1). `no_market` flag excludes fallback rows from S5-2 and pick analyses. 3,771 pre-EV NBA mock bets tagged `pre_ev_threshold`. Heavy-fav share 10% (was ~100% under old logic). **Bug fixed 2026-11-15**: `_persist_nba_forward_preds` ran `CREATE TABLE IF NOT EXISTS` but not `ALTER TABLE ADD COLUMN` migration loop — `no_market` column missing caused silent INSERT failure; added migration loop to match `_ensure_nba_forward_preds_table()`. **Prior data**: Nov 8-9 rows (430) all `no_market=1` (old storage, no market lines). **Effective day 1: 2026-11-15**. First Brier read at N≥100 games (~late-Dec 2026). |
| S5-3 | Forward weeks separate | **ACTIVE** | 2026-27 NBA season underway. First full market-linked capture 2026-11-15: 4,602 rows (9 games, 620 main + 3,953 alt + 29 fallback). Prior rows (Nov 8-9, 430 rows) are no_market=1 fallback only. Pre-tip snapshots scheduled for game windows. First Brier read at N≥100 games (~late-Dec 2026). |
| S5-4 | Pre-register rules | **DONE** | See locked configuration below. |

#### S5-4: Pre-registered sim configuration (RE-FROZEN 2026-11-15, tag `s5-4-freeze-2026-11-15`, commit 47389d4)

**Amendment log**: Original freeze 2026-10-30. Re-frozen 2026-11-15 before any day-1 data analyzed. Changes from original: P(plays) rates updated from hand-set to measured (178 game dates); timing-conditional tables added; role-based base rates added; supplementary blowout/OT/early-exit constants documented. No sim logic or engine parameters changed — only injury P(plays) inputs.

**Forward test day 1**: ~~2026-10-30~~ → ~~2026-11-06~~ → ~~2026-11-09~~ → ~~2026-11-08~~ → ~~2026-11-10~~ → **2026-11-15** (Oct 30–Nov 5: server downtime; Nov 6-7: `_fetch_market_lines()` + `datetime` bugs; Nov 8-9: threshold storage bug — `_pick_best_threshold` selected lowest rungs, 89% P(over)>0.80, 0% joinable to market; Nov 10-14: `no_market` column missing caused silent INSERT failure in `_persist_nba_forward_preds`; fixed 2026-11-15 by adding ALTER TABLE migration loop alongside CREATE TABLE IF NOT EXISTS).

**Engine (nba_matchup_engine.py):** unchanged from original freeze
- `reconcile: True`, `reconcile_strength: 0.5`, `reconcile_cap: 0.25`
- `defense: True`, `blowout: True`, `form: False`, `synergy: False`, `matchup: False`
- `reconcile_pool_fix: True` (OTHER bucket adjustment)

**Sim (nba_game_sim.py):**
- `_POSS_CV = 0.04` (pace variability)
- `_MARGIN_SIGMA = 15.8` (margin std, 4,668 games)
- `_GAME_TOTAL_STD = 19.4` (derived: σ²_game = 4 × 12.5² − 15.8² = 375.4 → σ = 19.4)
- `_TEAM_TOTAL_STD = 12.5` (team-level scoring std, 9,336 team-games)
- `_TEAM_SCORING_DAMP = 0.30` (scoring efficiency dampening)
- `_ENGAGEMENT_CV = 0.15` (per-player good-game/bad-game)
- `_ENGAGEMENT_LOADINGS = {PTS: 1.0, REB: 0.8, AST: 0.0, STL: 0.0, BLK: 0.0, FG3M: 1.0, PRA: 0.8}`
- Team scoring mult applies to PTS, FG3M only (not AST)
- `_BLOWOUT_THRESHOLD = 15.0`, `_BASELINE_BLOWOUT_PROB = 0.326`
- `_BLOWOUT_STARTER_CUT = 2.3` min, `_BLOWOUT_ROTATION_GAIN = 4.0` min, `_BLOWOUT_SPOT_GAIN = 7.0` min
- `_OT_PROB = 0.0512`
- `_OT_MINUTES_STARTER = 4.0`, `_OT_MINUTES_ROTATION = 1.0`, `_OT_MINUTES_SPOT = 0.0`
- In-game early exit: `_EARLY_EXIT_PROB = {starter: 0.035, rotation: 0.020, spot: 0.010}`
- `_EARLY_EXIT_MEAN = 14.8` min, `_EARLY_EXIT_STD = 4.7` min (minutes at exit)
- **P(plays) rates**: MEASURED from 178 game dates (nba_injury_forward_log + nba_player_game_logs), replacing original hand-set values:

  | Status | Original (hand-set) | Running (measured) | Source |
  |--------|--------------------|--------------------|--------|
  | Out | 0.00 | 0.00 | 7/2,591 = 0.3% (timing artifacts) |
  | IR | 0.00 | 0.00 | definitional |
  | Doubtful | 0.20 | **0.02** | 9/411 (pooled starter+rotation) |
  | Questionable | 0.65 | **0.55** | 1,220/2,198 (pooled starter+rotation) |
  | Probable | 0.95 | **0.92** | 763/826 (pooled starter+rotation) |
  | GTD | — | 0.50 | Sleeper-specific label, hand-set |
  | DTD | — | 0.70 | Sleeper-specific label, hand-set |
  | Active | — | 1.00 | confirmed healthy |
  | None/unlisted | — | 1.00 | defers to role base rate |

- **Timing-conditional P(plays)** (used when `hours_to_tip` available):

  | Status | Day-before (>12h) | Game-day (≤12h) |
  |--------|-------------------|-----------------|
  | Out | 0.000 | 0.000 |
  | Doubtful | 0.057 (9/159) | 0.000 (0/252) |
  | Questionable | 0.565 (534/945) | 0.547 (686/1,253) |
  | Probable | 0.893 (274/307) | 0.942 (489/519) |

- **Role base rates** (players NOT on injury report): starter 0.998, rotation 0.998, spot 0.658 (captures rest days, late scratches)
- **Calibration corrections**: NONE — all Normal-distribution corrections (bias + σ_scale) failed holdout validation (fit on H1, test on H2 and vice versa). PTS/PRA corrections were one-way only (helped one half, hurt the other). Only structural fix survived: Gamma-Poisson for discrete stats.
- **Discrete distributions (Gamma-Poisson compound = exact NegBin):**
  - STL, BLK, FG3M (`_DISCRETE_STAT_INDICES = {3, 4, 5}`): Gamma(α,β) rate draw → Poisson(λ = min × rate) — always
  - AST (`_AST_INDEX = 2`): Gamma-Poisson for rotation/spot, Normal for starters
  - Non-negative by construction (no floor at 0), integer output, NegBin-like overdispersion from game-to-game rate uncertainty
  - Gamma params: α = (μ/σ)², β = σ²/μ (matching mean and variance of per-minute rate)

**Storage & selection (scout/runner.py, scout/nba_player_props.py):**
- Forward preds stored at EVERY market-posted line (main + alt rungs) via `_get_market_lines_for_game()`
- Two-sided lines: multiplicative devig; one-sided alts: borrowed_margin (same player/stat/book main overround)
- EV-based pick selection: `_MIN_ODDS = 1.667`, `_MAX_ODDS = 3.50`, shadow-only (`EV_SHADOW` risk factor)
- Fallback rows: `no_market = 1`, excluded from S5-2 primary analysis

**Decision rules (pre-registered):**
- Report P(over) conditioned on playing (void rule)
- SGP k from sim draws, not independent product
- Edge = |sim P(over) - devigged market P(over)| (requires market odds from prop capture)
- Threshold for bet: edge ≥ 5pp AND sim P(over) ∈ [0.30, 0.70] (uncertainty zone)
- Stat coverage: PTS, REB, AST, PRA, FG3M, STL, BLK (all stats; discrete stats use Gamma-Poisson)
- Player coverage: starters + rotation only (spot excluded)

**Holdout calibration validation (S5-1):** unchanged from original

| Stat × Role | H1→H2 ΔKS | H2→H1 ΔKS | Verdict |
|-------------|-----------|-----------|---------|
| PTS starter | +0.044 (hurts) | -0.030 (helps) | ONE-WAY → removed |
| PTS rotation | +0.063 (hurts) | -0.041 (helps) | ONE-WAY → removed |
| PRA starter | +0.007 (neutral) | -0.025 (helps) | ONE-WAY → removed |
| PRA rotation | +0.010 (neutral) | -0.047 (helps) | ONE-WAY → removed |
| REB/AST/FG3M all | +0.01 to +0.10 | +0.00 to +0.03 | DEAD → removed |
| STL/BLK | -0.04 to -0.06 | -0.02 to -0.07 | SURVIVE (but root cause = discrete mismatch, fixed by Gamma-Poisson) |

**Changes after this point require new evidence (forward data) and explicit justification.**

### Empirical correlation targets

**B4 data (3,677 games, detrended per-game):**

| Stat | Same-team residual r | Cross-team residual r |
|------|---------------------|--------------------- |
| PTS  | -0.010              | +0.020               |
| REB  | +0.009              | +0.016               |
| AST  | -0.003              | —                    |
| PRA  | +0.030              | —                    |

**S4-1 player-pair cross-game method (no mechanical artifact, R13 data):**

| Stat | Same-team residual r | SE    |
|------|---------------------|-------|
| PTS  | -0.016              | 0.010 |
| REB  | +0.006              | 0.008 |
| AST  | -0.023              | 0.009 |

Cross-team PTS r (+0.020) > same-team (-0.010) — shared game pace dominates minute competition.
S4-1 per-pair method preferred (avoids N-player detrending artifact). Key finding: all near zero → sim should NOT inject strong same-team correlation beyond what shared minutes/pace already produce.

### S4-2: Empirical k factors (SGP combo calibration)

| Combo | Type | Emp k | 95% CI | Sim k | Match? |
|-------|------|-------|--------|-------|--------|
| PTS+REB (same player) | within-player | 1.200 | [1.175, 1.225] | 1.200 | YES |
| PTS+AST (same player) | within-player | 1.079 | [1.054, 1.105] | 1.112 | ~3% over |
| REB+AST (same player) | within-player | 1.059 | [1.035, 1.084] | 1.097 | ~4% over |
| PTS+REB+AST (same player) | within-player | 1.366 | [1.324, 1.410] | 1.386 | YES |
| PTS_under+REB_over | anti-correlated | 0.762 | [0.738, 0.783] | 0.800 | ~4% |
| Teammate PTS | teammate | 0.998 | [0.960, 1.034] | 1.014 | YES |
| Teammate REB | teammate | 0.971 | [0.931, 1.012] | 1.013 | within CI |
| Teammate PTS+REB | mixed | 1.011 | [0.982, 1.040] | 1.010 | YES |
| 3-leg teammate PTS | 3-leg | 0.986 | [0.890, 1.080] | 1.050 | within CI |
| Opposing star PTS | cross-team | 1.108 | [1.003, 1.216] | 1.013 | GAP (no fix) |

**Sim changes deployed**: engagement draw (eng_cv=0.15, stat loadings), scoring damp 0.30, AST removed from scoring mult.
**Accepted gaps**: opposing PTS (no mechanistic fix, CI wide); 3-leg teammate (shared pace).

### S4-2 Addendum: Holdout k factor stability check

Split R13 data at 2024-01-18: H1 = Oct-Jan (607 games, 80K preds), H2 = Jan-Apr (623 games, 82K preds).

| Combo | Full k | H1 k | H2 k | |H1-H2| | Stable? |
|-------|--------|------|------|---------|---------|
| PTS+REB (same player) | 1.200 | 1.208 | 1.192 | 0.017 | YES |
| PTS+AST (same player) | 1.079 | 1.067 | 1.091 | 0.024 | YES |
| REB+AST (same player) | 1.059 | 1.050 | 1.068 | 0.019 | YES |
| PTS+REB+AST (same player) | 1.366 | 1.358 | 1.375 | 0.017 | YES |
| PTSu+REBo (anti-corr) | 0.762 | 0.747 | 0.777 | 0.030 | ~ok |
| Teammate PTS | 0.998 | 1.032 | 0.962 | 0.070 | UNSTABLE |
| Teammate REB | 0.971 | 0.936 | 1.010 | 0.074 | UNSTABLE |
| Teammate PTS+REB mixed | 1.011 | 1.005 | 1.017 | 0.012 | YES |
| Teammate PTS+AST mixed | 0.998 | 0.991 | 1.006 | 0.015 | YES |
| 3-leg teammate PTS | 0.986 | 1.071 | 0.892 | 0.179 | UNSTABLE |
| Opposing PTS | 1.108 | 1.075 | 1.142 | 0.067 | UNSTABLE |

**Verdict**: Same-player k factors (engagement-driven, the only tuned parameters) are STABLE across halves (|H1-H2| ≤ 0.030). Engagement parameters (eng_cv=0.15, loadings) are NOT overfit. Cross-player k factors are UNSTABLE but the sim treats them as ~1.0 (no tuned parameter) — instability is sampling noise on N=600 games, not a model problem.

### S5-1: Two-stage PIT calibration (post engagement draw)

PIT re-run on 162,484 predictions (def_blow_recon_50pct) after engagement draw + scoring damp changes.

| Stat | Role | N | Bias | σ_scale | KS_raw | KS_fix | Verdict |
|------|------|---|------|---------|--------|--------|---------|
| PTS | starter | 8,302 | +0.530 | 1.226 | 0.043 | 0.033 | GOOD |
| PTS | rotation | 10,679 | +0.792 | 1.375 | 0.067 | 0.067 | OK (discrete helps) |
| REB | starter | 8,302 | +0.197 | 1.116 | 0.037 | 0.041 | ~neutral |
| REB | rotation | 10,679 | +0.294 | 1.150 | 0.048 | 0.060 | WORSE (skip) |
| AST | starter | 8,302 | +0.110 | 1.161 | 0.039 | 0.046 | ~neutral |
| AST | rotation | 10,679 | +0.140 | 1.388 | 0.050 | 0.104 | WORSE (need discrete) |
| PRA | starter | 8,302 | +0.837 | 1.149 | 0.046 | 0.019 | EXCELLENT |
| PRA | rotation | 10,679 | +1.226 | 1.227 | 0.071 | 0.038 | GOOD |
| FG3M | starter | 8,302 | +0.011 | 1.201 | 0.089 | 0.090 | POOR (need discrete) |
| FG3M | rotation | 10,679 | +0.065 | 1.214 | 0.116 | 0.127 | POOR (need discrete) |
| STL | starter | 8,302 | +0.005 | 1.628 | 0.234 | 0.171 | POOR (need discrete) |
| STL | rotation | 10,679 | +0.035 | 1.613 | 0.204 | 0.160 | POOR (need discrete) |
| BLK | starter | 8,302 | +0.005 | 1.401 | 0.187 | 0.167 | POOR (need discrete) |
| BLK | rotation | 10,679 | +0.028 | 1.327 | 0.219 | 0.215 | POOR (need discrete) |

**Calibration corrections applied in sim (after holdout validation):**
1. **Normal-distribution corrections (PTS, PRA, REB, AST)**: ALL REMOVED. Holdout test (fit H1, evaluate H2 and vice versa) showed every Normal correction is one-way (helps one half, hurts the other) or dead. In-sample KS improvement was overfit to the sample half.
2. **Discrete-distribution stats (STL, BLK, FG3M, AST rotation)**: Gamma(α,β) rate draw → Poisson(λ=min×rate) = exact NegBin marginal. Non-negative by construction, integer output. This is the only calibration fix that survives holdout — it addresses the structural mismatch (Normal CDF on integer counts), not a fitted parameter.
3. **Gamma parameters**: α = (μ/σ)², β = σ²/μ — derived from the player's per-minute rate mean and std (already in the model). No new fitted parameters.

### S5-1 Addendum: Prop odds capture pipeline

NBA player prop projections are generated daily by `scout.runner.run_daily_scout()` at 9:00 AM CT via `scheduler._run_daily_scout()`. The pipeline:
1. Fetches today's NBA games from the Odds API (~70 credits/day for spreads/totals)
2. Runs `nba_player_props.scout_game()` using `NBAMatchupEngine.project_game()` — no additional API credits
3. Persists ~50-200 player prop projections per game day to `scouted_props` table

**Credit cost**: ~70 credits/day for game lines (spreads/totals). Player prop projections are model-generated, not fetched from the API. Total budget: 100K credits/month (Creator tier). NBA prop pipeline is negligible cost.

**Forward validation**: Props stored with `projection_version` field. Outcomes backfilled via ESPN settlement. "Uncalibrated" badge remains until S5-2 comparison vs devigged market passes (requires N≥100 settled games, ~Dec 2026).

### Refinement implementation status audit (2026-10-30)

| # | Refinement | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Sub-parlay void pricing | **DEFINED** (not wired) | `p_parlay_void()` fully coded in nba_game_sim.py:339-462 but zero callers in production. `get_game_sim()` in nba_player_props.py defined but never called. |
| 2 | Pre-game vs in-game separation | **IMPLEMENTED** | Both mechanisms active in `run()`: pre-game availability (Step 2b, voided) + in-game early exit (Step 3, bet has action). |
| 3 | nbainjuries historical backfill | **PARTIAL** | /tmp/nba_injury_backfill.db exists (99,526 rows, 5 seasons). Measured P(plays) rates confounded by bench players — see diagnosis below. |
| 4 | Official report primary source | **IMPLEMENTED** | `fetch_injury_report()`: NBA PDF (nbainjuries package) > Sleeper > ESPN. Conflict logging active. |
| 5 | P(plays) rates | **HAND-SET** | Backfill rates exist but not usable as-is (bench player confound). Update rule pre-registered. |

#### P(plays) backfill diagnosis

Raw rates from nbainjuries historical backfill:

| Status | N | Played | Raw Rate | Expected |
|--------|---|--------|----------|----------|
| Out | 68,374 | 2,102 | 0.031 | ~0.00 |
| Doubtful | 1,876 | 61 | 0.033 | ~0.20 |
| Questionable | 13,315 | 3,087 | 0.232 | ~0.65 |
| Probable | 4,569 | 2,047 | 0.448 | ~0.95 |
| Available | 4,920 | 2,635 | 0.536 | ~1.00 |

**Root cause**: Denominators include end-of-bench players (15th man) who are healthy ("Available") but never play (DNP-Coach's Decision). 1,737 of 4,221 "Available" non-G-League entries have 0 minutes = bench warmers. For "Questionable", 8,822/11,849 (74.5%) have 0 minutes. Many are bench players who wouldn't play even if healthy.

**Fix needed**: Role-stratify by average minutes when healthy. A "Questionable" starter has ~65-75% P(plays); a "Questionable" 15th man has ~5% P(plays). The model only projects starters + rotation (≥15 MPG), so the relevant rate is conditional on being in the projection pool.

#### Pre-registered P(plays) update rule

HAND-SET rates remain in production until ALL of:
1. Role-stratified rates computed from backfill (status × role, N≥50 per cell)
2. Rates validated against a held-out season (fit on 2022-24, test on 2024-25)
3. Forward log from 2026-27 season confirms consistency (≥30 games)

When criteria are met, replace HAND-SET rates in `_P_PLAYS_BY_STATUS` and `_P_PLAYS_BASE_BY_ROLE` with measured values. Log the date and evidence. No interim partial updates.

### NBA player prop odds capture (added 2026-10-30)

Four snapshot types via `creator_tier.py`:
- **7:00 AM CT**: NBA-only morning snapshot (`fetch_player_props`, hours_ahead=16). ~67 credits/8-game slate.
- **11:00 AM CT**: All sports (`fetch_player_props`, hours_ahead=12). ~200 credits/day.
- **4:00 PM CT**: All sports (same). ~200 credits/day.
- **Pre-tip** (NEW): Per-game snapshot 30-60 min before tip (`fetch_pretip_nba_props`). Runs every scheduler cycle 5-11 PM CT. Catches each game once. ~9 credits/game.

Markets captured (NBA): player_points, player_rebounds, player_assists, player_threes, player_blocks, player_steals, player_points_rebounds_assists, plus alternate ladders (points_alternate, rebounds_alternate, assists_alternate). 10 markets total.

Books: FanDuel, DraftKings, BetMGM, Caesars, Pinnacle, BetRivers, PointsBet, Fanatics, ESPN BET. (9 books)

Stored in `historical.db:player_props`. UNIQUE constraint includes `point` (fixed 2026-10-31 — alternate lines at different thresholds were being overwritten).

Devig: Two-sided (over+under pair → proportional vig removal) when both sides available; one-sided raw implied when only one side. `get_nba_prop_odds_for_game()` provides devigged lookup by team names.

**Credit cost** (actual from API headers, Oct 29-31 avg):
- Daily: ~475 credits/day (201 API calls)
- Monthly projection: ~14,250 credits/month = 14.3% of 100K budget
- Per-call breakdown: event_list ~2 credits, per_event_odds ~3 credits
- Pre-tip window expanded to 10 AM – 11 PM CT (covers weekend/holiday morning tips)

**S5-2 comparison rule (PRE-REGISTERED)**: Model vs market uses the **last snapshot before tip** (pre-tip when available, 4 PM fallback). Earlier snapshots are opening-line reference only.

**Bug fix (2026-10-31)**: Props were broken Oct 28-31 — `_migrate_game_date_ct()` crashed on corrupt bets.db. Fixed: bets.db query wrapped in try/except. Schema fix: added `point` to player_props unique constraint (alternate lines were colliding). NFL alt-prop data Sep 20 – Oct 31 has irrecoverable loss from the same UNIQUE constraint bug (shared `player_props` table).

**Pipeline hardening (2026-10-31)**:
- WAL `wal_checkpoint(TRUNCATE)` REMOVED (root cause of recurring page-1 corruption). PASSIVE only.
- Failed prop slots no longer mark done — will retry on next cycle.
- `safe_migrate.py` backup switched from `shutil.copy2` to `sqlite3.backup()` API.
- Integrity check added to daily backup cycle (backup + live DB).
- Daily prop health check: rows per game/market/book with alerts on zero.
- First day of complete NBA capture recorded as forward test effective day 1.

**bets.db corruption status (2026-10-31)**:
- Current bets.db: page-1 header destroyed, completely unrecoverable.
- Root cause: `PRAGMA wal_checkpoint(TRUNCATE)` interrupted mid-write.
- Contributing: 16+ writers bypass `DB_WRITE_LOCK` (threading-only, no cross-process protection).
- Best recoverable: `/private/tmp/betiq_backups/2026-10-31_03-12_bets.db` → `.recover` → passes integrity_check, 228MB, all Oct 23-30 data except some scouted_props.
- Recovery file saved: `data/bets_safe_20261031/bets_recovered_from_oct31.db`
- All daily checkpoints and safe_migrate copies carry pre-existing B-tree corruption (pages 53527/53819). Only `.recover` output is clean.
- Awaiting user decision on restore + merge + Pikkit re-import.

## Infrastructure (not urgent)

| # | Item | Status | Notes |
|---|------|--------|-------|
| INFRA-1 | Archive pre-season/closed-season odds data | NOT STARTED | Move rows from player_props/alt_lines where sport is off-season into data/props_archive.db (same schema, indexed by sport/date). Trigger: 10 GB free-disk warning (~6 months at peak growth, ~250 MB/week). Runway at 17.8 GB free: ~14 months without archive. |
| INFRA-2 | Persist nightly maintenance results | DONE | maintenance_log table in bets.db (run_date, result_json, has_errors). 30-day retention. Written at end of _run_db_maintenance(). |
| INFRA-3 | Store NBA sim P(over) at scout time | DONE | nba_forward_preds table in historical.db. Written at scout (9 AM) and pre-tip (30-60 min before tip). Grading wired (runs after nightly_ats) with 3-tier resolution: bdl_player_id → alias table → normalized name. `nba_player_alias` table (7 seed aliases) resolves Odds API ↔ BDL name mismatches. `bdl_player_id` column populated at persist time. Smoke test (`s5_2_smoke_test.py`) reports: chain status, match rates (model ↔ player_props), unmatched names, alias table, bdl fill rate. Oct 30–Nov 5 reconstruction LOST (see S5-2). Forward capture starts Nov 6. |
| INFRA-4 | Repo security cleanup | **DONE** | Hardcoded keys replaced with `os.environ.get()` + `backend/.env` (gitignored). Git history scrubbed with `git filter-repo --replace-text`. Gitleaks pre-commit hook active. GitHub secret scanning + push protection enabled. Credit tripwire deployed: daily 3 AM check compares provider-side usage vs logged calls; alerts (macOS + email) on >5% unexplained gap. API endpoint: `/api/credits/tripwire`. BallDontLie has no usage headers — tripwire covers TheOddsAPI only. Historical gap analysis: 30.7% all-time gap explained by backfill scripts that bypass `_log_credits()`. |

## NFL Items

| # | Item | Status | Notes |
|---|------|--------|-------|
| NFL-1 | Violation report + architecture (Phase 0) | **DONE** | `nfl_reconciliation_diagnostics.py` built. Team-level and player-level diagnostics, sim scoring chain validation. |
| NFL-2 | Coverage data audit (Phase 1) | **CLOSED** | Coverage gaps addressed by NFL-6/7/9 pipeline fixes (2026-11-07). Main-line coverage now 100% across all 5 stats. |
| NFL-3 | Team volume reconciliation (Phase 2) | **DONE** | `nfl_volume_reconciliation.py` + `nfl_player_allocation.py` built. Player share profiles feed `nfl_game_sim.py`. |
| NFL-4 | Game sim + market anchoring (Phase 3) | **DONE (archived)** | `nfl_game_sim.py` built (drive-level sim with pace, backup QB, OT, penalties). Market anchoring via spread/total. Head-to-head vs v1.2 consensus showed v1.2 dominates — sim archived, not deployed to production. |
| NFL-5 | Bet-time logging (Phase 4) | **CLOSED** | `model_version` column deployed on `fantasy_forward_preds` (v0, v0_capped, v1.2_shifted_gamma). Regime tagging (R1/R2/R3) tracks pipeline versions. No separate A/B needed — v1.2 is sole production model. |
| NFL-6 | Lab coverage fix: show all posted props | **DONE** | 2026-11-07. Added "no projection" badge, alias table (7 seeds), calibration-unverified badge, per-game×market QA, API endpoint `/api/lab/props`. Before: 160 players shown (predictions-only). After: all players with posted prop or prediction shown. |
| NFL-7 | Pipeline root cause: alt-line dedup + suppression fix | **DONE** | 2026-11-07. Two bugs: (1) alt-line at same point as main overwrites via UNIQUE key; (2) main-line entries suppressed by bet-selection filters. Fixed both. Main-line coverage now 100% across all 5 stats: pass_yds 28/28, pass_tds 28/28, rush_yds 79/79, rec_yds 156/156, receptions 151/151. |
| NFL-8 | Name normalization: apostrophe strip + alias cleanup | **DONE** | 2026-11-07. `_normalize_player_name` now strips apostrophes (fixes Tre' Harris → Tre Harris). Removed 2 bad aliases (Cam Sample→DE, Tre Harris→wrong person). Added Matt Hibner alias. Takes effect on next pipeline run. |
| NFL-9 | Suppression → flag column (store all, filter at pick layer) | **DONE** | 2026-11-07. Added `suppression_rule TEXT` column. Three rules stored as flags: `alt_over`, `deep_alt_under_rec_pass`, `rb_rush_over_low_conv`. All predictions now stored (none skipped). Grading/calibration use ALL rows (no suppression filter). Suppression applied only at pick-generation layer (`run_weekly_pipeline` display filter). Impact: ~48% of alt-line rows were silently dropped in weeks 5-9 (~16,600 rows). |
| NFL-10 | Alias validation: both-side resolution + cleanup | **DONE** | 2026-11-07. Bug: `_resolve_nfl_name` only applied to Odds API names, not consensus — Chig Okonkwo, Mike Evans, Gabe Davis all had broken matches (alias mapped to canonical on one side only). Fix: `build_consensus` now uses `_resolve_nfl_name` instead of `_normalize_player_name`, so both sides resolve to the same key. Removed "Ken Walker"→"Kenneth Walker" alias (risky: could merge WR/FA with RB/KC; suffix normalization handles "III" already). 6 aliases validated against Sleeper roster: all team+position match. |
| NFL-11 | Add rush_attempts + receptions_alternate markets | **DONE** | 2026-11-07. Verified on Odds API: `player_rush_attempts` (2-6 outcomes/book, DraftKings/BetMGM/BetOnline/Bovada/Fanatics) and `player_receptions_alternate` (53-86 outcomes/book, DraftKings/Caesars/FanDuel/BetOnline/BetRivers/Fanatics). Added to `_PROP_MARKETS`, `STAT_TO_ALT_PROP`, and `_MARKET_TO_STAT`. Rush attempts shows with "no projection" badge (no Sleeper/ESPN source). Receptions alternate uses existing receptions consensus. |

## NFL Prereg Coverage Log

| Date | Change | rec_yds rows (before) | rec_yds rows (after) | Notes |
|------|--------|-----------------------|----------------------|-------|
| 2026-11-07a | Lab coverage fix: show prop-only players with "no projection" badge; add `nfl_player_alias` (7 seeds); calibration-unverified on receptions/pass_tds; per-game×market coverage QA alert (<95%) | 944 forward preds (224 players) | — | Display-only change, no prediction rows affected |
| 2026-11-07b | Pipeline root cause fix: alt-line dedup + suppression fix + normalization (apostrophe strip) | 944 (224 players) | 831 rows (156 players) main-line only; 4,975 total (202 players all depths) | Pool EXPANDED: main-line rec_yds from ~125 to 156 players (100% of consensus). Before: `find_prop_gaps` filtered by edge; `store_forward_predictions` suppressed main-line entries; alt-line overwrote main via UNIQUE key. After: all consensus players stored, main never suppressed, alt deduped. Global: pass_yds 100%, pass_tds 100%, rush_yds 100%, rec_yds 97.5%, receptions 95.6%. Remaining gaps (4 rec_yds, 7 receptions) are genuine no-projection players. |
| 2026-11-07c | **DATA-REGIME CHANGE**: suppression → flag column | — | All predictions now stored with `suppression_rule` flag | Weeks 2-4: pre-suppression (complete). Weeks 5-9: suppressed rows were DROPPED (never stored). Estimated ~16,600 alt-line rows lost across weeks 5-9 (~48% of full alt pool). Three rules: `alt_over` (all alt OVER, -18 to -26% structural ROI), `deep_alt_under_rec_pass` (deep alt UNDER rec/pass, -3.4% ROI), `rb_rush_over_low_conv` (RB rush OVER non-high). From W10 forward, all rows stored with flag; pick-generation filters at display/bet layer. **Prereg impact**: rec_yds OVER near_alt rows (~13-30/week) were suppressed by `alt_over` rule; these include eligible prereg candidates (edge>=5pp, odds>-150). Prereg results before W10 are computed on the stored (filtered) pool only. |
| 2026-11-07d | **PREREG REGIME R3**: full-storage + alias fix | — | — | **R1** (created_at < 2026-11-05T22:03:48Z, W7–W8): v1.2 deployed, sparse alt rungs. Bugs: (a) suppression dropped near_alt/deep_alt OVER (~48% of alt pool, active since 2026-10-02); (b) consensus-side alias bug (Chig Okonkwo/Mike Evans/Gabe Davis; 68 rows flagged `data_error='alias_mismatch'`). Last R1 row: 2026-11-04T21:27:45Z. **R2** (2026-11-05T22:03:48Z <= created_at < 2026-11-07T10:47:26Z, W9 pre-fix): same bugs as R1; bias-corrected projections deployed; 3,171 rows; last R2 row: 2026-11-06T16:26:12Z. **R3** (created_at >= 2026-11-07T10:47:26Z, W9 post-fix + W10+): 1,804 W9 rows + all future; all rows stored with `suppression_rule` flag, alias resolved both sides, `data_error` flags applied. R3 is the first clean regime — primary evaluation pool. Grading/calibration use ALL rows (no suppression filter); suppression applies only at pick-generation layer. No rows exist in either overnight gap (R1→R2 or R2→R3). |
| 2026-11-07e | **Shadow-pick suppression deployed** | — | — | Pick-generation filter in `run_weekly_pipeline` now excludes `suppression_rule IS NOT NULL`. Shadow picks were **unfiltered** from 2026-11-07T10:47Z (R3 start) to 2026-11-07 (this fix). Prior to R3, suppression was applied at storage time (rows dropped, never stored). Effect: no shadow picks were generated from suppressed rows in any regime — R1/R2 dropped them, R3 pre-fix stored but never generated picks from them (the display filter already required abs_gap>=1.0 + edge>=5pp which filtered most). This note records the regime boundary for shadow-pick tracking. |
| 2026-11-07f | **Post-kickoff contamination flagged** | — | — | 3,045 rows flagged `data_error='post_kickoff'` (created_at > kickoff_utc). Root cause: `store_forward_predictions()` used INSERT OR REPLACE without kickoff freeze — Sunday refreshes overwrote pre-kickoff rows after games started. Per-game freeze now enforced (skip rows where `now > kickoff_utc`). Snapshot-based grading restores pre-kickoff consensus when needed. **Prereg R1 impact**: 13/154 R1 picks (W7–W8, rec_yds OVER, v1.2, edge≥5pp) were post-kickoff. Clean R1: N=141, W=93, L=48, WR=66.0%, ROI=+24.7% (vs contaminated: N=154, WR=64.9%, ROI=+22.7%). All analysis queries now filter `data_error IS NULL`. Clean settled pool: 12,931/15,341 (84.3%). |
| 2026-11-08a | **P(over) distribution mislabel identified + frozen** | — | — | `find_prop_gaps()` has used **lognormal** for yardage P(over) since file creation (commit a91b992, 2026-10-02). All stored `p_over` values for rows tagged `model_version='v1.2_shifted_gamma'` were computed with lognormal, NOT shifted gamma. The shifted gamma CDF (`p_over_line_gamma_shifted`) was only used in `nfl_calibration_diagnosis.py`'s `_rescore_v1(use_v12=True)` path — never in the production storage pipeline. Difference: mean 3.9pp, max 10.4pp (rec_yds WR worst). `distributional_median` now correctly uses shifted gamma ppf(0.5); P(over) is frozen as lognormal until after W14 prereg verdict. Any analysis reading stored `p_over` directly gets lognormal; analyses using `_rescore_v1(use_v12=True)` get shifted gamma. This two-path inconsistency is known but frozen — the prereg reads stored values and must not change mid-test. |
| 2026-11-08b | **Shadow P(over) v1.2 gamma + distribution label deployed** | — | — | Two new columns on `fantasy_forward_preds`: (1) `p_over_v12_gamma` — shifted gamma CDF P(over) computed alongside frozen lognormal `p_over`. Populated from W10 forward for yardage stats (pass_yds, rush_yds, rec_yds); NULL for counts. Uses same `p_over_line_gamma_shifted(line, mean, cv, base_rate)` as analysis-tool `_rescore_v1`. (2) `p_over_distribution` — labels which CDF produced `p_over`: 'lognormal' for yardage, 'negbin' for counts. Backfilled on all 28,612 existing rows. Reports should use `p_over_distribution` instead of `model_version` to identify the P(over) distribution. Post-W14 head-to-head: compare `p_over` (lognormal) vs `p_over_v12_gamma` (shifted gamma) on same predictions — Brier, calibration, ROI by odds band, prereg rule under each. |
| 2026-11-08c | **Head-to-head lognormal vs gamma (W2-W8)** | — | — | Backfilled `p_over_v12_gamma` on 6,432 graded yardage rows. **All yardage (main+alt, N=6,432)**: gamma Brier 0.1674 vs lognormal 0.2093 (Δ=-0.042, paired t=-17.6). Calibration error: gamma 0.047 vs lognormal 0.172. **Main-line only (N=2,698)**: gamma Brier 0.250 vs lognormal 0.336 (Δ=-0.086); game-clustered 95% CI: [-0.105, -0.067]. Main-line prereg ROI: gamma +13.5% @≥3pp edge (N=956, WR=59.9%) vs lognormal -0.4% (N=1,879, WR=52.6%); game-clustered 95% CI on ROI: [+4.1%, +22.1%]. Direction disagreements (N=637): gamma right 358 vs lognormal 279 (56.2%). Gamma wins every stat, every week, every position. **Caveat**: v1.2 base rates fitted W2-W5 (in-sample); W5-W8 OOS improvements larger (good sign). **Brier populations**: Δ=-0.042 = all yardage (main+alt); Δ=-0.086 = main-line only. Use main-line as primary (matches prereg population). |
| 2026-11-08d | **FW-2 prereg amendment + FW-2b parallel prereg** | — | — | (1) FW-2 amended with O'Brien-Fleming interim looks at W10/W12/W14 (alpha spending 0.0005/0.014/0.045, two-sided). Filed at `data/prereg_fw2_amendment_20261108.md` before any W10 graded. (2) FW-2b registered: parallel prereg using `p_over_v12_gamma` (shifted-gamma) instead of `p_over` (lognormal), same design + OBF schedule. Filed at `data/prereg_fw2b_gamma_20261108.md`. Both run simultaneously on identical predictions. Production P(over) stays frozen. |
| 2026-11-08e | **Deferred items triage** | — | — | Reviewed all BLOCKED/DEFERRED/PARKED items. (1) FW-5 (suppression re-validation) + FW-6 (R3 re-runs): **truly blocked** by missing data (need R3 W10+W11 settled, earliest ~2026-11-21). (2) S3-2 (NBA NegBin for STL/BLK): **unblocked** — trigger condition (PIT discrete mismatch KS=0.19-0.23) already met; promoted to OPEN. (3) PFF subscription: correctly parked (no new evidence). (4) CV Phase 2: correctly closed for CB matchup. (5) P(over) switch: was "after W14"; now has interim looks — first possible verdict at W10 (~2026-11-14). (6) CV/base-rate refit: can run as shadow analysis now; deploy after prereg verdict. |
| 2026-11-08f | **Decision layer → gamma P(over) for yardage** | — | — | **REGIME R4** (2026-11-08): Lab display and `_recalc_p_over()` (injury redistribution) now use shifted-gamma P(over) (`p_over_v12_gamma`) for yardage stats. Counts (receptions, pass_tds) still use NegBin. **Stored columns preserved**: `edge_vs_market` = lognormal, `dist_direction` = lognormal (frozen for FW-2 prereg). **New columns**: `edge_gamma`, `dist_direction_gamma` store gamma-derived values separately. `_suppression_rule()` uses lognormal `p_over` (matches FW-2). Shadow picks/mock bets from W10+ use gamma-derived edges via `edge_gamma`. Lab shows both values: gamma primary, lognormal greyed with "ln" label. Game-clustered 95% CI on **main-line** (N=2,698) gamma ROI @≥3pp: +13.5% [+4.1%, +22.1%] overall; OOS (W6-W8): +15.8% [+3.0%, +27.3%]. Main-line Brier Δ (gamma-ln): -0.086 [-0.105, -0.067] overall; OOS: -0.106 [-0.132, -0.079]. |

| 2026-11-08g | **B3-F1 market anchoring never ran in NBA production** | — | — | `_fetch_market_lines()` was added to `runner.py` (commit 535d8a4, Nov 5) but never defined in `nba_team_markets.py`. Impact: `AttributeError` at runner.py:854 killed the ENTIRE per-game try block — not just market anchoring, but `nba_props.scout_game()`, `nba_teams.scout_game()`, and `_persist_nba_forward_preds()` all skipped. **Confirmed: 0 NBA scouted_props from Nov 6-8** (last working: Oct 24). This means: (1) NBA player prop scouting = dead since Nov 6; (2) NBA team market scouting = dead; (3) nba_forward_preds = 0 rows (S5-2 broken from start); (4) market anchoring features (blowout minutes, pace from total, team-total reconciliation) = never exercised in production. **No fallback behavior** — the exception was caught silently by the per-game `except Exception` and `continue`d. NBA Lab outputs during Nov 6-8 were unaffected (Lab re-runs engine on demand, not from scouted_props). Fixed 2026-11-08: `_fetch_market_lines()` implemented, queries `fixtures.bookmakers` JSON. Production verification pending (first rows expected Nov 9). |
| 2026-11-08h | **Silent-failure audit + pipeline health infrastructure** | — | — | Audit of all broad `try/except` in scheduled paths found 20 silent swallows (5 HIGH, 12 MEDIUM, 3 LOW). **ALL 20 FIXED.** HIGH: scout/runner.py per-game exceptions (print-only → logger.error + traceback), fantasy_projections.py gamma calc (except:pass → logger.warning + counter). MEDIUM: scheduler.py `_dispatch()` error handler, 4 nightly job handlers, auto_settle, LiveMonitor, 3 restart-detection bare-except blocks, scorecard error — all now log with `traceback.format_exc()`. LOW: 3 fantasy_projections.py query fallbacks → logger.warning. Infrastructure: (1) `pipeline_health.py` module — 12 tracked pipelines (nfl_fantasy_pipeline, nfl_grading, nba_scout, nba_pretip, nba_grading, nfl_scout, mock_bet_generate, mock_bet_settle, injury_watcher, model_scorecard, nightly_maintenance, db_backup) with per-run tracking (rows written, errors, status OK/DEGRADED/DOWN/STALE), zero-row alerting, health summary. Cadences tuned so STALE = genuinely overdue. (2) **Daily health email at 8 AM CT** — sends per-pipeline status; subject flags issue count; macOS alert on DOWN. (3) Health block appended to Tuesday scorecard email. (4) `record_run()` wired into ALL 12 pipelines (success + error paths). (5) Gamma calc failure tracking: `find_prop_gaps` counts attempts/failures, logs summary; `nfl_fantasy_pipeline` records DEGRADED if >1% of yardage predictions lack `p_over_v12_gamma`. (6) `RotatingFileHandler` in main.py: `logs/betiq_app.log`, 10 MB × 5 backups, all `logger.*` calls now persist to disk with rotation. (7) `CLAUDE.md` rule: "verified in production" = rows from scheduled path. |
| 2026-11-08i | **NBA Lab market anchoring confirmed** | — | — | Lab on-demand path calls `_get_market_lines()` from `historical.db:betting_lines` (ESPN close lines), passes `market_total`/`market_spread` to `project_game()`. All three anchoring features active: (1) blowout minutes via `market_spread` → minutes model, (2) expected possessions from `market_total` → pace factor, (3) game totals derived from market. Scout path uses `_fetch_market_lines()` from `bets.db:fixtures.bookmakers` (OddsAPI). Both paths apply identical engine anchoring. Lab outputs during Nov 6-8 B3-F1 outage were **unaffected** — Lab re-runs engine on demand with its own market source, independent of scouted_props. |
| 2026-11-08j | **S5-2 smoke test: 0 rows — server restart required** | — | — | `s5_2_smoke_test.py` run for 2026-11-08 and 2026-11-09: both show 0 scout rows, 0 pre-tip rows. `player_props` OK (128K rows Nov 8, 9K rows Nov 9), `fixtures` OK (8 games Nov 8, 5 games Nov 9). Root cause: server PID 81526 still running pre-fix code — `_fetch_market_lines` AttributeError continues to kill NBA scout path silently. Additional running-server errors: `fetch_pretip_nba_props` ImportError (pre-tip broken), `_iw_tr` unbound in injury_watch_poll, `database is locked` on props fetch. **All code fixes deployed to disk; server restart will activate them.** Day 1 will be first game-day after restart. Alias table OK (7 entries). |

## Analyses Needing Re-Run on Complete Data (post-W10)

These analyses were computed on the filtered pool (weeks 5-9 missing ~48% of alt-line rows, plus 68 alias-bug rows now flagged `data_error='alias_mismatch'`). All analytical queries now include `AND data_error IS NULL`. Suppression filter is NOT applied to analyses — only at shadow-pick/mock-bet layer.

**Re-run date**: after W11 settles (~2026-11-21), providing 2+ full weeks of clean R3 data (W10–W11). Report flagged vs unflagged segments separately.

| Analysis | File | Query location | Impact | Re-run needed |
|----------|------|----------------|--------|---------------|
| v1.2 Brier score / calibration | nfl_calibration_diagnosis.py | line ~138 | Alt-line calibration unknown — only UNDER alts were graded | Yes — W10+ adds OVER alts |
| Prereg forward test (rec_yds OVER) | nfl_calibration_diagnosis.py | line ~2377 | Near-alt OVER rec_yds rows suppressed — eligible prereg pool truncated | Yes — R3 pool is first clean set |
| Prereg Brier assessment | nfl_calibration_diagnosis.py | line ~2789 | Same pool truncation | Yes |
| Performance report (ROI by segment) | fantasy_projections.py | line ~1699 | Alt-line ROI segments incomplete | Yes — report by suppression_rule |
| Bias prior (rolling segment bias) | fantasy_projections.py | line ~791 | Bias computed on filtered subset; likely negligible (main-only) | Low priority |
| Dud mixture fit | nfl_dud_mixture.py | line ~45 | Only yardage UNDER alts included; dud probability for OVER unknown | Yes |
| CV / base-rate model fits | nfl_interim_dispersion.py | lines ~636, ~898 | Same filtered-pool issue | Yes |
| Injury baseline edge | injury_watcher.py | line ~583 | Minor — avg abs edge across all preds, but pool was truncated | Low priority |

## NFL-12: Re-validate suppression rules on R3 data

**Status**: NOT STARTED — blocked until W11 settles (~2026-11-21)

After 2+ weeks of clean R3 data (W10–W11), re-validate each suppression rule on forward data.
The rules were derived from R1/R2 data which had suppression + alias bugs; they may not hold on clean data.

| Rule | Rationale (R1/R2 evidence) | Re-validation test |
|------|---------------------------|-------------------|
| `alt_over` | -18 to -26% structural ROI from alt OVER vig | R3 alt OVER hit rate vs BE + ROI (N>=50) |
| `deep_alt_under_rec_pass` | -3.4 to -3.7% ROI, vig eats edge | R3 deep-alt UNDER rec/pass hit rate vs BE (N>=30) |
| `rb_rush_over_low_conv` | RB rush OVER non-high has poor calibration | R3 RB rush OVER by conviction: hit rate vs BE (N>=30) |

**Decision protocol**: If any rule's R3 forward ROI crosses zero (CI includes positive), promote to OPEN for review. If ROI is clearly negative (CI entirely below zero), confirm KEEP. If N is insufficient, extend to W13.

## NFL-13: Matchup Context Module

**Status**: CLOSED as far as public data allows — 2026-11-07. Volume DEAD across 3 seasons (2023-2025, 1,502 WR1-games). Yardage suppression null with PFR label (UNINFORMATIVE — label r=−0.08) AND null with consensus-elite label (stable, scout-based — 71-86% retention). Forward test (human-reported shadows vs market line) continues as the only remaining path.

`backend/nfl_matchup_context.py` — Reusable per-game matchup module. Display only, no model changes.

| Section | Status | Data source | Notes |
|---------|--------|-------------|-------|
| 0. Storage tables | **DONE** | historical.db | `nfl_matchup_team_context`, `nfl_matchup_player_context`, `nfl_cb_wr_evidence`, `nfl_candidate_features`, `nfl_cb_side_tendency`, `nfl_wr_side_tendency`, `nfl_shadow_log` (new — reported shadows from free sources) |
| 1. Team verified stats | **CORRECTED** | nflverse PBP + FTN | RZ TD% now per-trip (not per-play). Blitz rate shown two ways with N denominator. Pressure source labeled. League averages on every stat. N + 95% CI on all rates/means. LOW-CONFIDENCE flag when N<100. |
| 2. QB vs blitz | **CORRECTED** | PBP + FTN | N and CI on EPA/drop, comp%, sack rate. LOW-CONFIDENCE flag on small splits. |
| 3. Separation | **CORRECTED** | NGS receiving | YAC-allowed proxy REPLACED with NGS receiver-week opponent adjustment. opp_effect = avg(receiver_sep_vs_opp - baseline) by position. CIs on both receiver baselines and opp effects. |
| 4. CB-on-WR proxy | **VALIDATED DEAD** | PBP credited defenders vs BDB tracking | BDB 2023 (14,079 plays): 38.4% top-1 accuracy vs 37.5% random baseline. Man coverage 36.3% (below chance). HHI confirms non-random assignment but proxy can't exploit it with ~2.7 CBs/play. Labeled INFORMATIONAL ONLY. shadow_cb_flag and primary_cb_prob candidate features marked DEAD. |
| 5. Script/volume | **CORRECTED** | PBP | N and CI on neutral pass rate. N on PROE. LOW-CONFIDENCE flag. |
| 6. Lab output | **DONE** | historical.db | `matchup_context` key in `build_lab_props()` return. |
| 7. Hypotheses | **ALL DEAD** | nfl_candidate_features | 8 candidate features ALL DEAD. shadow_cb_flag + primary_cb_prob (proxy dead). Remaining 6 tested (2026-11-07): 1/12 historical, 0/24 forward match predicted direction significantly. Market absorbs all. |
| 8. CV scope | **CORRECTED** | — | Clarified: BDB is for training/validating CV model, not an in-season substitute for PBP proxy. |
| As-of computation | **CORRECTED** | — | `through_week = week - 1` always (pre-kickoff data only). Retrospective flag changes labels only, not data window. |

**Follow-ups:**
- ~~Re-run DET-MIN retrospectives with corrected module~~ — **DONE** (2026-11-07). 2026 W8 + 2025 W9 re-run with through_week=week-1. Leaky rows dropped from historical.db.
- ~~Run module for current week's games~~ — **DONE** (2026-11-07). 2026 W9 DET @ MIA full Lab panel displayed.
- ~~Validate CB proxy accuracy on BDB tracking data~~ — **DONE** (2026-11-07). BDB 2023 (14,079 plays, 18 weeks): top-1 38.4% vs 37.5% random. **DEAD** — no predictive value. shadow_cb_flag and primary_cb_prob features killed.
- Evaluate cost/benefit of PFF matchup data subscription — **PARKED** (2026-11-07). PFF Pro ($199.99/yr) researched but deferred: target-volume test shows zero effect → no reason to pay for matchup data until multi-season BDB results (if ever) reverse the verdict. See Decision Rule below.

### BDB 2023 Ceiling + Outcome Tests (2026-11-07)

**Ceiling test** (true tracking-data matchups, weeks 1..w-1 → predict week w):
- Cross-game alignment-based: 32.2% (below random 40.1%)
- Within-game WR-specific: 32.2% (lift +2.3pp)
- Within-game zone-based: 38.9% (best performer)
- Route-share MAE: 0.970 (random 0.625)
- **Interpretation**: With ~2.7 CBs/play, the ceiling for pre-game CB-on-WR prediction is near random. Even perfect tracking data from prior games doesn't help. Zone-based assignment within a game is the only consistent signal.

**Outcome test** (does facing top CB hurt WR production?):
- vs Top CB: 8.2 yds/target, 69.7% comp, +0.233 EPA (N=3,562)
- vs Other: 8.2 yds/target, 68.4% comp, +0.208 EPA (N=7,950)
- Game-clustered bootstrap: yards diff +0.00 [-0.40, +0.38], EPA diff +0.02 [-0.04, +0.09]
- Shadow effect (N=201): yards -0.21 [-1.83, +1.48], EPA +0.05 [-0.19, +0.25]
- **Verdict**: Zero detectable outcome effect. Facing the opponent's top CB does not suppress WR production at the per-target level.

### Decisive Round (2026-11-07)

**1. Power analysis (WR1s, N=515 WR1-games, 270 unique games):**

| Metric | Effect (top CB − other) | 95% CI | MDE (80% power) |
|--------|------------------------|--------|-----------------|
| Target share | +0.75 pp | [−0.76, +2.28] | 2.18 pp |
| Targets/game | −0.24 | [−0.73, +0.20] | 0.66 |
| Rec yards/game | −2.50 | [−8.86, +3.85] | 9.31 |

**−2pp WR1 target-share effect: RULED OUT** (CI lower bound −0.76 pp, above −2.0).

**2a. Elite CB dose-response (29 elite CBs, top quartile EPA/target allowed, min 50 snaps):**

| Metric | Elite CB effect | 95% CI | Poor CB effect | 95% CI |
|--------|----------------|--------|----------------|--------|
| Target share | −0.09 pp | [−1.86, +1.65] | −1.41 pp | [−3.17, +0.36] |
| Targets/game | −0.25 | [−0.80, +0.28] | −0.43 | [−0.95, +0.08] |
| Rec yards/game | **−10.99** | **[−17.51, −4.55]** | +6.73 | [−1.56, +14.88] |

Per-target breakdown (WR1 targeted plays, **same-season** elite definition): elite CB suppresses yds/target by −2.05 [−2.69, −1.41] and EPA/target by −0.376 [−0.489, −0.256]. BUT: target share is unchanged, and CB matchup prediction is at chance (38.4%), so the outcome effect is **not actionable**.

**⚠ CIRCULARITY WARNING (2026-11-07)**: Same-season EPA definition is mechanical — defining "elite" by EPA allowed then finding they allow less EPA is tautological. See "Circularity Fix" section below.

Dose-response: slope = −1.87 pp per 100% elite exposure, r = −0.047, p = 0.29 — not significant.

**2b. Bracket/safety help (Lamb mechanism, 2+ defenders within 5 yds):**
- 41.0% of WR1 routes are bracketed (N=11,632 WR1 plays)
- **Play-level**: −6.10 pp targeting rate when bracketed (real mechanism)
- **Game-level**: −0.11 pp [−1.69, +1.44] — **fully compensated** (QBs find WR1 on non-bracketed plays)
- MDE at 80% power: 2.24 pp

**3. Redistribution test (WR1 → WR2/TE):**

| When WR1 faces... | WR2 target share effect | 95% CI | TE1 target share effect | 95% CI |
|-------------------|------------------------|--------|------------------------|--------|
| Top CB | −0.15 pp | [−1.64, +1.35] | −0.26 pp | [−1.73, +1.26] |
| Elite CB | +0.46 pp | [−1.52, +2.38] | +1.19 pp | [−0.75, +3.08] |
| Heavy bracket | +0.68 pp | [−0.87, +2.24] | +0.10 pp | [−1.35, +1.49] |

No redistribution detected at any level.

**VERDICT — PARK NFL-13**: Volume (target share, targets/game) null at every level tested; −2pp ruled out. Outcome (yards/target) suppressed by elite CBs but not actionable (can't predict matchup). Bracket mechanism real at play level but fully compensated at game level. No redistribution. CB matchup features **DEAD** for props. All 8 candidate features **DEAD** (2 proxy, 6 tested — 0/24 forward tests match predicted direction significantly).

### Candidate Feature Test Results (2026-11-07)

**6 remaining features tested against pre-registered directions** (display-only; no model changes).

**Historical (2023–2025)**: as-of features (data before each game), outcome = player stat vs season-to-date baseline, player FE + opponent pass EPA adjustment, game-clustered SEs. 2,200–3,400 obs per feature.

| Feature | Stat | Pred | β | 95% CI | p | Match |
|---------|------|------|---|--------|---|-------|
| opp_blitz_rate | pass_yds | − | **+82.0** | [+20.1, +143.8] | 0.010 | **WRONG*** |
| opp_blitz_rate | receptions | + | **+0.87** | [+0.18, +1.56] | 0.014 | **YES*** |
| qb_blitz×opp_blitz | pass_yds | − | −19.2 | [−87.0, +48.7] | 0.578 | null |
| tgt_shift×opp_blitz | receptions | + | −0.84 | [−5.17, +3.49] | 0.695 | wrong, null |
| tgt_shift×opp_blitz | rec_yds | + | −24.6 | [−85.6, +36.5] | 0.423 | wrong, null |
| ngs_sep_baseline | rec_yds | + | −1.64 | [−5.63, +2.35] | 0.404 | wrong, null |
| ngs_sep_baseline | receptions | + | −0.16 | [−0.40, +0.07] | 0.175 | wrong, null |
| neutral_pass_rate | pass_yds | + | **−194.4** | [−335.4, −53.5] | 0.006 | **WRONG*** |
| neutral_pass_rate | rec_yds | + | **−40.0** | [−55.7, −24.3] | <0.001 | **WRONG*** |
| neutral_pass_rate | receptions | + | **−2.9** | [−4.3, −1.6] | <0.001 | **WRONG*** |
| opp_proe_allowed | rec_yds | + | −6.1 | [−27.3, +15.1] | 0.573 | wrong, null |
| opp_proe_allowed | receptions | + | −0.25 | [−1.78, +1.28] | 0.749 | wrong, null |

**Historical verdict**: 1/12 matches predicted direction significantly (opp_blitz→receptions). 4/12 significant in WRONG direction (neutral_pass_rate all 3 stats + opp_blitz→pass_yds). 7/12 null. Neutral pass rate's wrong-direction effect likely selection bias: QBs with high neutral pass rates face better pass defenses → their stat baselines already reflect this.

**Forward (2026 W2–W9)**: outcome = actual − devigged market line / actual − v1.2 projection. 857–3,288 obs per feature. HC1 robust SEs (game-clustered SEs failed for some features due to mixed types).

| Feature | Stat | Pred | β (vs mkt) | p | β (vs v1.2) | p | Match |
|---------|------|------|-----------|---|------------|---|-------|
| opp_blitz_rate | pass_yds | − | +84.4 | 0.068 | +76.7 | 0.132 | no |
| opp_blitz_rate | receptions | + | −0.22 | 0.793 | +0.48 | 0.544 | null |
| qb_blitz×opp_blitz | pass_yds | − | −35.4 | 0.487 | −20.9 | 0.678 | right, null |
| tgt_shift×opp_blitz | receptions | + | −6.4 | 0.185 | −8.8 | 0.063 | wrong, null |
| tgt_shift×opp_blitz | rec_yds | + | −69.8 | 0.341 | −90.5 | 0.216 | wrong, null |
| ngs_sep_baseline | rec_yds | + | −2.3 | 0.243 | −1.2 | 0.529 | wrong, null |
| ngs_sep_baseline | receptions | + | −0.08 | 0.619 | −0.08 | 0.618 | wrong, null |
| neutral_pass_rate | pass_yds | + | −10.3 | 0.770 | +11.5 | 0.737 | null |
| neutral_pass_rate | rec_yds | + | −13.1 | 0.149 | −6.5 | 0.483 | wrong, null |
| neutral_pass_rate | receptions | + | −1.3 | 0.189 | −1.4 | 0.169 | wrong, null |
| opp_proe_allowed | rec_yds | + | +17.5 | 0.493 | +13.4 | 0.629 | right, null |
| opp_proe_allowed | receptions | + | −0.9 | 0.708 | −2.0 | 0.391 | null |

**Forward verdict**: 0/24 tests significant in predicted direction. All null. The sole historical survivor (opp_blitz→receptions) does not replicate forward (β = −0.22, p = 0.793 vs market).

**OVERALL: ALL 6 CANDIDATE FEATURES DEAD.** No feature shows a consistent, correctly-signed, significant effect across both historical and forward tests. The market already incorporates whatever signal these features carry.

### Multi-Season Pooling Results (2026-11-07)

**Data**: BDB 2023 (2023 season, all-22 tracking, 515 WR1-games) + BDB 2027 (2024/2025 seasons, player_play.csv tabular path, 987 WR1-games). Total: 1,502 WR1-game observations.

**BDB 2027 tracking limitation**: Only 1-2 focus players per play (competition entrant evaluation), NOT all-22. Extracted routes/closest-CB from player_play.csv using per-play primary-CB assignment (highest season-snap CB on each play's defense). Target column only filled for 14% of pass plays (competition-curated subset). Elite/poor CB tiers have only 4 CBs each in 2024/2025 (vs 29 in 2023).

**Test 1 — Top-CB WR1 target share:**

| Season | N | n_high | Effect (pp) | 95% CI |
|--------|---|--------|-------------|--------|
| 2023 | 515 | 125 | +0.75 | [−0.76, +2.28] |
| 2024 | 483 | 201 | +0.33 | [−1.50, +2.10] |
| 2025 | 504 | 213 | +0.42 | [−1.04, +1.88] |

All 3 seasons independently null. Volume suppression CONFIRMED DEAD.

**Test 1 — Rec yds/game (supplementary):**

| Season | Effect | 95% CI |
|--------|--------|--------|
| 2023 | −2.50 | [−8.86, +3.85] |
| 2024 | **−5.00** | **[−9.37, −0.76]** |
| 2025 | −1.82 | [−7.31, +3.52] |

2024 yardage suppression is significant and directionally consistent with 2023's elite-CB finding (−10.99). ~~Partial out-of-sample support for the yardage hypothesis.~~ **RETRACTED** — see Circularity Fix below. This finding used same-season top-CB definition; it does not survive prior-season definition.

### Circularity Fix (2026-11-07)

The elite-CB tests defined "elite" from same-season EPA allowed, making the EPA effect **mechanical**. Re-ran all 3 seasons with elite defined from **prior** season's PFR passer-rating-allowed (2022→2023, 2023→2024, 2024→2025), same min-target filter (≥25), bottom 25th percentile.

**Prior-season elite-CB test (PFR passer_rating_allowed → next season BDB data):**

| Season | N | n_elite | yds/tgt | 95% CI | EPA/tgt | 95% CI | rec_yds/g | 95% CI |
|--------|---|---------|---------|--------|---------|--------|-----------|--------|
| 2022→2023 | 515 | 41 | +1.12 | [−0.55, +2.68] | +0.045 | [−0.176, +0.268] | +9.65 | [−3.93, +22.50] |
| 2023→2024 | 485 | 13 | +2.09 | [−0.23, +4.28] | −0.429 | [−1.041, +0.237] | +0.13 | [−7.80, +7.47] |
| 2024→2025 | 520 | 28 | +2.55 | [+1.17, +3.90] | −0.243 | [−0.490, −0.007] | +29.39 | [+16.03, +41.54] |
| POOLED | 1,520 | 82 | +2.72 | [+1.61, +3.73] | −0.103 | [−0.272, +0.065] | +23.10 | [+14.23, +31.53] |

**Verdict (raw prior-season)**:
- **EPA "replication" was circular.** With prior-season definition, pooled EPA is null (−0.103, CI spans zero). The 2023 EPA effect flips to +0.045.
- **Yardage effects reverse sign.** All 3 seasons show WR1s doing BETTER (not worse) against prior-season elite CBs. Pooled yds/tgt is +2.72 (significant, wrong direction). Pooled rec_yds/game is +23.10 (significant, wrong direction).
- **Why effects reverse**: Prior-season elite CBs face the best WR1s (selection confound).

### Within-WR Fixed Effects (2026-11-07)

Controls for WR quality confounding: WR fixed effects (each WR vs his own other games) + opponent pass-defense EPA (leave-one-out). Game-clustered SEs.

| Season | N | n_elite | Rec yds/g β [95% CI] | Yds/tgt β [95% CI] | EPA/tgt β [95% CI] |
|--------|---|---------|----------------------|---------------------|---------------------|
| 2023 | 515 | 41 | +4.6 [−11.7, +20.9] | +0.09 [−1.9, +2.1] | −0.09 [−0.4, +0.2] |
| 2024 | 276 | 10 | +1.0 [−9.6, +11.6] | −1.37 [−4.5, +1.8] | −0.30 [−1.2, +0.6] |
| 2025 | 366 | 28 | **+17.8** [+4.7, +30.8] | −0.79 [−2.7, +1.1] | −0.24 [−0.6, +0.1] |
| POOLED | 1,157 | 79 | +8.8 [−0.8, +18.5] | −0.51 [−1.8, +0.8] | −0.15 [−0.4, +0.1] |

- **Yds/target**: null in every season and pooled. No within-WR suppression.
- **EPA/target**: null in every season and pooled. Consistently negative direction but never significant.
- **Rec yds/game**: null in 2023/2024. 2025 anomaly (+17.8, p=0.01) persists with WR FE (28 obs, likely small-sample or remaining game-script confound). Pooled borderline null (p=0.07, wrong direction).
- **WR FE absorbed the raw reversal** in 2023 (+9.65 → +4.6) and pooled (+23.1 → +8.8), confirming WR quality was the dominant confound.

**Verdict (after WR FE)**: No evidence of elite-CB yardage suppression at the within-WR level. The per-target measures (yds/target, EPA/target) are the cleanest tests and are definitively null.

### Elite Label Stability (2026-11-07)

PFR passer_rating_allowed year-to-year correlation (CBs qualifying both years, ≥25 targets):

| Pair | N (both years) | r (unweighted) | r (target-weighted) | Elite retention |
|------|---------------|----------------|---------------------|-----------------|
| 2021→2022 | 63 | −0.036 | −0.017 | 17% (3/18) |
| 2022→2023 | 53 | −0.179 | −0.216 | 19% (3/16) |
| 2023→2024 | 78 | −0.026 | −0.035 | 39% (9/23) |

**Average r = −0.08 (NO stability).** The prior-season "elite" label is essentially noise. 17-39% of elite CBs retain elite status the next year. This means:
- The null prior-season results are expected regardless of whether a real effect exists — the label is too noisy to detect it.
- The same-season test was significant because it was tautological, not because the effect is real.
- We cannot distinguish "no effect" from "effect exists but label is too noisy" with this methodology.

Scripts: `data/bdb/prior_season_elite_cb_test.py`, `data/bdb/within_wr_elite_cb_test.py`.

### Consensus-Elite Label Test (2026-11-07)

The PFR statistical label (passer_rating_allowed) has r=−0.08 year-to-year — essentially noise. Historical tests with that label are **UNINFORMATIVE** (null results expected regardless of whether an effect exists). To resolve: built a stable, scout-based "consensus-elite" label.

**Consensus-elite criteria (union):**
- All-Pro or Pro Bowl CB in the prior 2 seasons
- Top-10 CB contract AAV at time of test season
- First-round draft pick ≤ 4 years into career

**Label properties:**

| Test season | N CBs | Awards | Contract | Draft | Year-to-year retention |
|-------------|-------|--------|----------|-------|----------------------|
| 2023 | 36 | 15 | 10 | 22 | — |
| 2024 | 38 | 15 | 10 | 24 | 86% (31/36 from 2023) |
| 2025 | 30 | 15 | 10 | 20 | 71% (27/38 from 2024) |

Retention 71-86% vs PFR's 17-39% — a **much more stable label**. If there is a real elite-CB suppression effect, this label should detect it.

**Within-WR FE regression with consensus-elite label:**

| Season | N | n_elite | Rec yds/g β | p | Yds/tgt β | p | EPA/tgt β | p |
|--------|---|---------|-------------|---|-----------|---|-----------|---|
| 2023 | 515 | 81 | −2.8 | 0.584 | −0.28 | 0.642 | −0.01 | 0.902 |
| 2024 | 276 | 40 | +6.0 | 0.278 | −1.53 | 0.335 | −0.60* | 0.008 |
| 2025 | 366 | 22 | +8.1 | 0.249 | +0.93 | 0.680 | +0.18 | 0.504 |
| POOLED | 1,157 | 143 | +2.7 | 0.425 | +0.04 | 0.951 | −0.05 | 0.518 |

**Verdict**: Even with a stable label, **all primary measures null**. Rec yds/game null in every season and pooled. Yds/target pooled = +0.04 (effectively zero, p=0.951). EPA/target pooled null (p=0.518). The 2024 EPA anomaly (p=0.008) is 1 of 12 tests (expected ~0.6 false positives at α=0.05) and does not replicate in 2023 or 2025.

**Suppression bound**: Pooled rec yds/game 95% CI = [**−4.0**, +9.5]. This **rules out suppression ≥ 4 yds/game** — less than half the retracted same-season estimate (−10.99). The forward test MDE at N=30 is ~19 yds, far above this bound; expect INCONCLUSIVE even if the true effect is at the upper bound of what public data allows (−4 yds).

**Naive (no WR FE) comparison**: Pooled rec_yds naive is +9.2* (p=0.013) — WR quality confounding confirmed. WR FE absorbs this entirely (+2.7, null).

**Test 4 — Redistribution:**
- WR2: null in all seasons and pooled
- TE1: −1.85 pp in 2025 [−3.24, −0.49], pooled −2.05 pp [−2.99, −1.11] — NEGATIVE (opposite of redistribution hypothesis; elite defenses may suppress passing broadly)

**Pipeline**: `data/bdb/multi_season_pipeline.py` — dual-path (tracking for 2023, player_play.csv for BDB 2027).

### Shadow Log (2026-11-07)
- `nfl_shadow_log` table added to historical.db for manually-entered projected shadows
- CLI: `--add-shadow DEF CB WR OFF SOURCE DATE --week W` and `--show-shadows --week W`
- Post-game confirmation: `--confirm-shadow DEF CB WR --week W` / `--deny-shadow DEF CB WR --week W`
- Source accuracy tracking: `--source-accuracy --season YYYY` (reports confirmed/not-confirmed/pending by source)
- Lab panel Section 4b displays reported shadows labeled "REPORTED, NOT MODELED"
- User enters from PFF/RotoBaller/ESPN free previews

### Weekly Shadow Checklist
| Day | Action |
|-----|--------|
| **Thursday** | Enter projected shadows from free articles (PFF, RotoBaller, ESPN) using `--add-shadow` |
| **Tuesday** | Confirm/deny shadows from completed games using `--confirm-shadow` / `--deny-shadow` |

### Decision Rule (NFL-13) — CLOSED as far as public data allows

Historical tests are **UNINFORMATIVE** with the PFR statistical label (r=−0.08, noise). With a stable consensus-elite label (71-86% retention), **all measures remain null** (pooled yds/tgt +0.04 p=0.951, EPA/tgt −0.05 p=0.518, rec yds/g +2.7 p=0.425). Two independent labels, same answer: no detectable suppression.

- **PFF Pro**: **NOT NEEDED.** Both labels null. No signal to justify subscription.
- **CV Phase 2**: **CLOSED for CB-matchup question.** Available for other uses (formation classification, alignment detection, route recognition) if scoped separately.
- **Shadow log**: **ACTIVE** — forward test (human-reported shadows vs market line) continues. This is the only remaining path: if real elite-CB suppression exists but requires play-by-play shadow knowledge (not statistical classification), the market-comparison design will detect it. Prior expectation: null. MDE at N=30 ≈ 19 yds. Expect inconclusive. Weekly checklist required.

### Pre-Registered Forward Test: Elite-CB Shadow → WR1 Yardage (Amended 2026-11-07)

**Status**: ACTIVE from W10 (2026-11-07). Full prereg at `data/bdb/prereg_elite_cb_shadow_test_2025.md`.

**Amendment (2026-11-07, before any W10 game)**: Season corrected from 2025 to 2026. Elite CB list rebuilt from 2025+2026 PFR data (37 CBs, PR ≤ 78.9). WR1 list rebuilt from 2026 PBP using targets/game rate (not total targets) to handle injured WR1s. MDE and playoffs added. Circularity + within-WR FE + label stability findings noted — prior expectation is now null.

**Rationale (updated)**: BDB 2023 decisive round found elite CBs suppress WR1 rec yards by −10.99/game using **same-season** elite definition. However, re-running with **prior-season** PFR passer-rating-allowed definition kills all effects (2023: +9.65, null; 2024: +0.13, null; 2025: +29.39, wrong direction). EPA "replication" was circular. The forward test proceeds as a final check: if there IS a real suppression effect that requires shadow-level information (who covers whom), the market comparison will detect it independently of the circular BDB finding.

**Design**:
- **Elite CB list**: 37 CBs frozen from 2025 full-season + 2026 W1-W8 PFR Advanced Defense Stats (via nflverse `import_weekly_pfr('def')`). Criteria: `depth_chart_position == 'CB'`, ≥25 coverage targets, passer_rating_allowed ≤ 78.9 (bottom 25th percentile of 148 qualified CBs). Stored in `data/bdb/elite_cb_list_2026.csv` with timestamp 2026-11-07. **No changes to this list during the test.**
- **WR1 list**: Receiver with highest **targets per game played** (min 3 games) per team through current week (from 2026 nflverse PBP). Rate-based definition handles injured WR1s returning (e.g., PHI DeVonta Smith at 10.25/g replaces Lemon at 5.62/g). Updated weekly; stored in `data/bdb/wr1_per_team_2026.csv`.
- **Trigger**: A `nfl_shadow_log` entry created before kickoff, where `cb_name` matches a frozen elite CB and `wr_name` is the opponent's WR1 (current leader in targets for the opposing team).
- **Measures** (all per-observation, game-clustered CIs):
  1. WR1 actual rec yds − devigged market rec_yds line (primary)
  2. WR1 actual rec yds − v1.2 projection (does v1.2 already account?)
  3. WR1 targets vs season average
  4. WR1 yds/target vs season average
- **Stored in**: `nfl_elite_cb_shadow_results` table
- **Evaluation**: `--elite-shadow-test` CLI command; also callable via `evaluate_elite_cb_shadow_test()`
- **Power**: SD of WR1 rec yds/game = 43.2 (from 2023 data). Estimated SD of (actual − market line) ≈ 36.7 (0.85 × raw SD).
  - MDE at N=30: **18.8 rec yds** (80% power, α=0.05)
  - MDE at N=40: 16.3 rec yds
  - MDE at N=50: 14.6 rec yds
  - Historical (now-retracted) same-season effect was ~11 yds/game. **Expect INCONCLUSIVE** unless the real effect (if any) is much larger than the circular estimate suggested.
- **Decision criteria** (report at end of **playoffs**, W19+; was W18):
  - If CI for diff-vs-market excludes zero (WR1 underperforms): elite-CB shadow has predictive value beyond the market → consider as prop adjustment
  - If CI spans zero but diff-vs-v1.2 < 0: market already prices it but v1.2 doesn't → v1.2 gap to close
  - If both CIs span zero: elite-CB shadow not actionable even with perfect shadow info
- **Observation window**: W10 through end of playoffs (W19+). Regular season W10-W18 + any playoff games with elite-CB shadow triggers.
- **Minimum N for verdict**: 30 observations (estimated ~4-6 per week × 9 regular + ~2-4 playoff weeks = 36-60+ by end of playoffs)
- **Display-only**: No changes to the frozen v1.2 model. Shadow log entries tagged with elite-CB status in Lab panel.

**Weekly workflow**:
1. Thursday: enter shadows via `--add-shadow`. Check elite CB list for matchups.
2. Sunday/Monday: record WR1 actual stats (rec yds, targets, yds/target).
3. Tuesday: confirm/deny shadows using `--confirm-shadow` / `--deny-shadow`.
4. Periodic: run `--elite-shadow-test` to check accumulating results.

### CV Phase 2 Groundwork (2026-11-07) — CLOSED for CB matchup; available for other uses
- Design doc: `backend/docs/cv_phase2_design.md` (491 lines)
- BDB labels: ~7.7M player-frame positions, ~14K matchup labels, ~14K formation/route labels
- Approach: Roboflow sports + YOLOv11 + BoT-SORT for detection/tracking; mkoshkina jersey-number-pipeline for OCR
- Video sources: HDMI capture of broadcast (best legal option); NFL+/YouTube prohibit automated access; synthetic (Unreal) for pre-training
- **CB-matchup use case CLOSED**: consensus-elite + PFR label both null; no CV-derived matchup feature to build toward
- **Remaining uses if scoped separately**: formation classification (shotgun/under-center/etc.), alignment detection (WR splits, box count), route recognition (classification from tracking). These are independent of the CB-matchup question.

## NFL Model Scorecard — DONE (2026-11-07, corrections applied)

Weekly scorecard computing projection accuracy, direction, calibration, and biggest misses from `fantasy_forward_preds`. Backfilled W2, W4-W8 (W3 no data, W9 ungraded).

| # | Item | Status | Notes |
|---|------|--------|-------|
| SC-1 | `model_scorecard.py` — compute + store | **DONE** | `weekly_model_scorecard` table in historical.db. Per-segment metrics (stat×pos×model_version×line_depth + `_ALL_` rollups). Accuracy (MAE, median AE, WAPE), direction vs market-implied (edge_pp + CI), calibration (10 p_over buckets), Brier (model vs market, per-model_version), biggest misses (top 10, deduplicated), stale-input flag, direction lean split (over/under). Regime tagging (R1/R2/R3). STD stored as week=0. |
| SC-2 | API endpoints | **DONE** | `GET /api/scorecard/{sport}` with filter params (season, week, stat_type, position, model_version, line_depth). `POST /api/scorecard/{sport}/compute` for manual compute/backfill. Returns edge CI, lean split, stale metrics. |
| SC-3 | Frontend Lab tab | **DONE** | "Scorecard" nav item. Dropdowns: week, stat, position, model version, line depth. 4 cards: Accuracy (color-coded delta), Direction vs Market-Implied (edge + CI + lean sub-rows), Calibration (bucket table + per-model Brier breakdown), Biggest Misses (stale-input amber banner). MAE trend chart. |
| SC-4 | Scheduler integration | **DONE** | Tuesday 1 PM CT scorecard compute + email. Pre-kickoff consensus refresh (Thu 5 PM, Sun 9:30 AM, Mon 5 PM CT) re-pulls Sleeper/ESPN without API cost. |
| SC-5 | Backfill W2-W9 | **DONE** | 1,288 segment rows across W2, W4-W8 + STD. W3 skipped (no data). W9 skipped (ungraded). |
| SC-6 | Email summary | **DONE** | Main-line accuracy, direction vs market-implied (edge + CI), per-model Brier, top 5 misses, stale-input summary, STD headline. |

**Season-to-Date (main lines):** N=7,420 | MAE 17.8 (mkt 17.7) | Edge +3.9pp vs market-implied | Brier 0.2908 (mkt 0.2459).

**Per-model Brier (W8):** v1.2_shifted_gamma 0.1890 (✓ beats market) · v0_capped 0.2153 · v0 0.2163. v1.2 eliminated 90-100% tail overconfidence (4 preds vs 818 for v0).

**Rush yds direction (STD, main):** OVER leans 42.6% hit / -7.4pp edge (N=782) — systematic over-projection. UNDER 49.7% / -0.3pp (N=372). RB OVER worst at -8.3pp (N=525).

**Stale-input (W8):** 42 flagged (1.0%), MAE 23.5, direction 27/42. Threshold: |proj - line| > 40% AND > 10 units, line ≥ 5, main lines only.

**Pending:** NBA stub ready (add stat types when 2026-27 grading accumulates, ~Dec 2026).

## Forward Watch — Active Trackers

All items collecting forward data. Each has a decision date and minimum N.

| # | Item | Status | Decision date | Notes |
|---|------|--------|---------------|-------|
| FW-1 | NBA S5-2 market comparison | **BLOCKED — threshold mismatch** | Late-Dec 2026 (N≥100 games) | Day 1 = 2026-11-08 (212 rows, 7 games). 193/212 graded. **Model↔market join: 0/193 (0%)**. Unjoin causes: line not posted for player (138, 71%), alt market one-sided/over-only (45, 23%), player not in props (10, 5%). Root cause: `_pick_best_threshold()` maximizes |P-0.5|, always selecting the lowest rung (pts 10.5, reb 3.5, pra 14.5) — far below market-posted main lines for starters. **Action**: must persist predictions at market-posted lines. Also fixed: `player_props.game_date` UTC→CT backfill (704K rows); `creator_tier.py` uses `_utc_to_ct_date()` going forward. Oct 30–Nov 7 LOST. |
| FW-2 | NFL prereg rec_yds OVER (lognormal) | **COLLECTING** | W14 (~2026-12-05) with interim looks W10/W12 | Three regimes: R1 (W2-W8, suppression bug), R2 (W9 pre-fix), R3 (W10+, clean). Verdict on R3 data only. Coverage 156/156 rec_yds players on main line. **R1 corrected** (2026-11-07f): 13/154 picks were post-kickoff contaminated; clean R1: N=141, WR=66.0%, ROI=+24.7%. **Shadow gamma** (2026-11-08b): `p_over_v12_gamma` stored from W10; post-W14 head-to-head lognormal vs gamma on same predictions. **Amendment 2026-11-08** (pre-W10): Added O'Brien-Fleming interim analysis at W10/W12/W14 (alpha spending 0.0005/0.014/0.045 for two-sided; see `data/prereg_fw2_amendment_20261108.md`). **W2-W8 head-to-head** (2026-11-08): all yardage (main+alt, N=6,432) gamma Brier 0.167 vs lognormal 0.209 (Δ=-0.042, t=-17.6); main-line only (N=2,698) Δ=-0.086; calibration error 0.047 vs 0.172; main-line prereg ROI +13.5% vs -0.4%. All in-sample for base rates (W2-W5 training); W5-W8 OOS improvements *larger*. |
| FW-2b | NFL prereg rec_yds OVER (shifted-gamma) | **COLLECTING** | W14 (~2026-12-05) with interim looks W10/W12 | **Parallel prereg** registered 2026-11-08 (pre-W10). Uses `p_over_v12_gamma` (shifted-gamma CDF) instead of `p_over` (lognormal). Same design: rec_yds OVER, edge ≥ 5pp vs devigged implied, main-line, method=distributional. Same O'Brien-Fleming schedule. Runs simultaneously with FW-2 on identical predictions — only the P(over) and therefore edge calculation differs. See `data/prereg_fw2b_gamma_20261108.md`. |
| FW-2c | NFL low-line OVER (shifted-gamma) | **COLLECTING — BET** | W14 (~2026-12-05) with OBF interim W10/W12/W14 | Registered 2026-11-15 pre-W10 kickoff. See definition below. **OVER only** — UNDER ROI -28.6% (N=28). Live bets placed from W10. **W10 result**: 81.5% WR (22/27), +45.3% ROI. Season (W2-W10): N=202, 71.3% WR, +26.0% ROI [+14.2%, +37.1%] — significant. |
| FW-2d | NFL QB rush_yds (shifted-gamma) | **COLLECTING — BET** | W14 (~2026-12-05) with OBF interim W10/W12/W14 | Registered 2026-11-15 pre-W10. See definition below. Both sides; UNDER stronger (+32.2%) but OVER not significant (+16.0%). **Amended to BET** per user decision (2026-11-15). Live bets placed from W10 (combined with FW-2c in 5u cap). **W10 result**: 60.0% WR (12/20), +14.4% ROI. Season (W2-W10): N=105, 64.8% WR, +22.4% ROI [+4.4%, +40.4%] — significant. |
| FW-2e | NFL edge >15pp non-low non-QB-rush (shifted-gamma) | **COLLECTING — TRACK ONLY** | W14 (~2026-12-05) with OBF interim W10/W12/W14 | Registered 2026-11-15 pre-W10. See definition below. Diverse composition (rec_yds 62, receptions 52, rush_yds 51). Track only — high edge might be model artifact despite +15.9% ROI. **W10 result**: 45.5% WR (15/33), -15.5% ROI. Season (W2-W10): N=192, 58.3% WR, +7.0% ROI [-5.6%, +19.7%] — NOT significant. Track-only validated. |
| FW-3 | SGP validation (personal bets) | **PARTIAL** | Pending Pikkit export | 602 bets in bets.db (2024-01-04 → 2026-04-23). 100/589 fully priceable (17%). Overcharge c rises 1.05→1.18 by leg count. Need Pikkit export ~2026-04-24 → today to complete. |
| FW-4 | Injury edge-at-news collection | **COLLECTING** | N≥50 events (~Jan 2027) | `injury_watcher.py` polls Sleeper every 15 min on game days. 1,417 snapshots in `injury_edge_snapshots`. Tracks teammate prop edge at first post-news fetch vs baseline. |
| FW-5 | NFL-12 suppression re-validation | **BLOCKED** | After W11 (~2026-11-21) | Re-validate `alt_over`, `deep_alt_under_rec_pass`, `rb_rush_over_low_conv` on R3 data. Need N≥30-50 per rule. |
| FW-6 | R3 re-runs (8 analyses) | **BLOCKED** | ~2026-11-21 (W10+W11 settled) | Brier, prereg, performance report, bias prior, dud mixture, CV fits, injury baseline. See "Analyses Needing Re-Run" section. |
| FW-7 | NFL-13 elite-CB shadow forward test | **COLLECTING** | End of playoffs (W19+, ~Feb 2027) | Market-comparison design. 37 frozen elite CBs. MDE ~19 yds at N=30. Expect INCONCLUSIVE. Weekly shadow log entries. |
| FW-8 | NBA S5-3 forward weeks | **ACTIVE** | Late-Dec 2026 (N≥100 games) | 2026-27 season underway. Day 1 = 2026-11-08 (212 rows: 97 scout + 115 pre-tip, 7 games). Nov 8 graded: 193/212 (6 players unmatched: Sabonis, Hardy, Markkanen, Powell, Prince, Missi). Two code bugs fixed (datetime shadowing + missing _fetch_market_lines). Four infrastructure fixes deployed 2026-11-09: git hash capture, health status display, db_backup retry, per-game tip freeze. Market anchors missing (fixtures empty) — model-only fallback. |
| FW-9 | Lognormal vs gamma weekly update | **ACTIVE** | OBF interim W10/W12/W14 | `backend/h2h_pover_analysis.py` — re-run after each week grades. Compare `p_over` (lognormal) vs `p_over_v12_gamma` (shifted-gamma) Brier on R3 data. Apply OBF boundaries (W10: \|z\|>3.471, W12: \|z\|>2.454, W14: \|z\|>2.004). Baseline W2-W8: **all yardage (main+alt, N=6,432) Δ=-0.042 (t=-17.6)**; **main-line only (N=2,698) Δ=-0.086**. Use main-line as primary metric (matches prereg population). |

#### FW-2c: Pre-registered low-line OVER rule (registered 2026-11-15T16:20Z, commit 8a172b3)

**Population**: `fantasy_forward_preds` WHERE `season=2026 AND method='distributional' AND line_depth='main' AND data_error IS NULL`. Stale main-line duplicates excluded (see 2026-11-15 data-quality note below).

**Qualifying criteria** (all must hold):
1. **Low line**: `(stat_type IN ('pass_yds','rush_yds','rec_yds') AND prop_line < 10)` OR `(stat_type = 'receptions' AND prop_line <= 1.5)`
2. **Not QB rush**: exclude `position = 'QB' AND stat_type = 'rush_yds'` (that's FW-2d)
3. **OVER side only**: gamma P(over) > 0.5 for yardage, lognormal P(over) > 0.5 for counts → side = OVER
4. **Edge ≥ 3pp**: `(model_prob - mkt_prob) × 100 ≥ 3.0` where `model_prob = p_over_v12_gamma` (yardage) or `p_over` (counts), `mkt_prob = devigged_implied`
5. **Price band ≤ +100**: FanDuel decimal price ≤ 2.0
6. **Positive EV**: `model_prob × (fd_price - 1) - (1 - model_prob) > 0`
7. **Not stale**: `NOT (prop_line >= 5 AND ABS(consensus_mean - prop_line) > 0.40 * prop_line AND ABS(consensus_mean - prop_line) > 10)`
8. **Not past kickoff**: `kickoff_utc > now`
9. **Injury**: Out/IR/Doubtful excluded; Questionable listed separately, re-checked after inactives

**Staking**: 1u for yardage stats, 0.5u for count stats (lower model confidence). Max 1 bet per player, 2 per game, 5u total per slate.

**Decision**: same OBF schedule as FW-2 (interim W10: |z|>3.471, W12: |z|>2.454, W14: |z|>2.004). Primary metric: ROI with game-clustered bootstrap CI excluding zero.

**Backtest (in-sample, deduped)**: OVER N=111, 50 games, WR=81.1%, ROI=+45.6% [+32.1%, +58.0%]. UNDER excluded (N=28, ROI=-28.6%). Both early (W2-W5) and late (W6-W9) periods significant for OVER.

#### FW-2d: Pre-registered QB rush_yds rule (registered 2026-11-15T16:20Z, commit 8a172b3)

**Population**: same as FW-2c.

**Qualifying criteria**:
1. `position = 'QB' AND stat_type = 'rush_yds'`
2. Edge ≥ 3pp (gamma P(over) for rush_yds)
3. Price band ≤ +100 (FanDuel decimal ≤ 2.0)
4. Positive EV, not stale, not past kickoff, injury filtered (same as FW-2c)
5. **Both OVER and UNDER** — UNDER stronger (+32.2% ROI) but sample small

**Status**: TRACK ONLY. Live bets from W10 (user's decision to include).

**Staking**: 1u per pick. Max 1 per player, 2 per game (within 5u total with FW-2c).

**Backtest (in-sample, deduped)**: ALL N=80, 57 games, WR=65.0%, ROI=+22.7% [+1.3%, +42.6%]. UNDER N=33, +32.2% [+4.7%, +59.5%]. OVER N=47, +16.0% [-12.3%, +43.2%] (not significant). Early (W2-W5) strong +44.8%, late (W6-W9) +6.3% (not significant) — possible regime drift.

#### FW-2e: Pre-registered edge >15pp rule (registered 2026-11-15T16:20Z, commit 8a172b3)

**Population**: same as FW-2c.

**Qualifying criteria**:
1. Edge > 15pp (same gamma/lognormal P(over) and devigged_implied as FW-2c)
2. **Not low-line** (not qualifying for FW-2c) and **not QB rush** (not qualifying for FW-2d)
3. Price band ≤ +100 (FanDuel decimal ≤ 2.0)
4. Positive EV, not stale, not past kickoff, injury filtered

**Status**: TRACK ONLY (no bets). Diverse composition: rec_yds 62, receptions 52, rush_yds 51, pass_tds 10, pass_yds 5. High edge may be model artifact — tracking to distinguish.

**Backtest (in-sample, deduped)**: N=153, 62 games, WR=61.4%, ROI=+15.9% [+0.2%, +31.5%] (barely significant).

---

### 2026-11-15: Stale main-line duplicate data-quality fix

**Bug**: when market lines moved between projection refreshes (e.g. Metchie rec_yds 13.5→7.5), the UNIQUE constraint `(season, week, player_name, stat_type, method, prop_line, line_depth)` included `prop_line`, so old rows at the previous line kept `line_depth='main'` alongside the new row at the current line. Both were treated as main-line predictions.

**Impact**: 3,874 stale duplicate rows across all 2026 weeks (44% of graded main-line data). All calibration figures, scorecard computations, and h2h analyses prior to this fix used contaminated data with double-counted predictions. FW-2 R1 was partially affected (alt-line rows also duplicated).

**Fix (3 parts)**:
1. `fantasy_projections.py`: `store_forward_predictions()` now runs `UPDATE SET line_depth='stale_main'` on existing main rows with a different `prop_line` before inserting the new row.
2. `lab_props.py`: dedup logic changed from "closest to median consensus mean" to "latest `created_at`" — ensures frontend shows the current market line.
3. DB cleanup: 3,874 rows demoted from `main` to `stale_main`. Zero main-line duplicates remain.

**Corrected calibration (W2-W8, deduped, edge ≥ 3pp, price ≤ +100)**:

| Category | N (old→new) | ROI | 95% CI (10k boot) | Sig? |
|---|---|---|---|---|
| (a) QB rush | 154→84 | +24.8% | [+4.4%, +43.5%] | YES (was no) |
| (b) Low lines | 259→163 | +27.4% | [+11.8%, +41.7%] | YES |
| (c) Edge >15pp | 374→153 | +15.9% | [+0.2%, +31.5%] | YES (was no) |
| (d) Rest 3-15pp | 1736→709 | +4.6% | [-3.3%, +12.5%] | no |

**Action**: all forward analyses (h2h, FW-2 R1, scorecard, ladder) must be re-run on deduped data. Prior figures are superseded.

**Re-run completed 2026-11-15 10:30 CT** (all on deduped DB):
- **H2H gamma vs lognormal**: gamma still dominates (Brier 0.1674 vs 0.2093, Δ=-0.042, t=-17.6). Main-line only (N=1030): gamma Brier 0.2473 vs LN 0.3289. Calibration MAE: gamma 0.047 vs LN 0.172. All weeks gamma-better. Unchanged conclusion.
- **FW-2 R1 (rec_yds OVER prereg)**: N=140 graded (W7-W8, 26 games). HR=66.4% [57.1%, 76.1%], ROI=+25.6% [+8.1%, +43.7%]. All R1 regime (suppression+alias bugs). R2/R3 not yet graded. 6 more weeks needed.
- **Scorecard backfill**: 1,626 segments across 7 periods. STD main-line: MAE=11.94, Direction=57.5% (vs 51.4% market), Brier(model)=0.2699, Brier(mkt)=0.2433, Bias=-0.92, Right-side rate=57.4%, N=3,916.
- **Ladder**: model beats market at select rungs (QB rung -3, +1, +11, +12; RB rung +0, +17, +29, +31). No systematic ladder advantage — model-better rungs are scattered, not concentrated at near-main.

**W9+W10 grading + scorecard update completed 2026-11-15 23:35 CT**:
- W9 graded: 10,275 total (1,352 main). W10 graded: 6,960 total (550 main, 13/14 games — LAC@BAL MNF excluded).
- Scorecard computed for W9, W10, STD refreshed through W10.
- STD (W2-W10, main): N=5,775, MAE=11.67, Dir=56.4% (mkt 51.4%), Brier=0.277 (mkt 0.245), Bias=-0.50.
- **Updated season calibration (W2-W10, edge≥3pp, price≤+100)**:

| Category | N | Games | WR | ROI | 95% CI | Sig? |
|---|---|---|---|---|---|---|
| (a) QB rush | 105 | 74 | 64.8% | +22.4% | [+4.4%, +40.4%] | YES |
| (b) Low lines | 202 | 75 | 71.3% | +26.0% | [+14.2%, +37.1%] | YES |
| (c) Edge >15pp | 192 | 84 | 58.3% | +7.0% | [-5.6%, +19.7%] | no (was borderline) |
| (d) Rest 3-15pp | 913 | 102 | 56.5% | +4.5% | [-1.9%, +11.0%] | no |

- W10 FW-2c (low-line OVER): 81.5% WR (22/27), +45.3% ROI. W10 FW-2d (QB rush): 60.0% WR (12/20), +14.4% ROI.
- W10 live bets (FW-2c+2d combined): 3/5 settled, +1.64u. 1 pending (Love rush_yds — Sleeper data gap).
- (c) edge >15pp dropped from borderline significant to non-significant — validates track-only decision.
- Jordan Love rush_yds actual missing from Sleeper — data gap to investigate.

### 2026-11-22: W11 Sunday capture incident + ad-hoc pipeline run

**What happened**: W11 forward preds only covered TNF (567 rows, stored Thu Nov 19 10:05 CT — before Sunday props were first fetched at 2026-11-20T00:03Z). Fri/Sat daily runs and Sun 9 AM `pregame_sun` refresh never fired: the morning chain runs synchronously in `_loop()` (scheduler.py:4998) ahead of the NFL block (scheduler.py:5188), and chains were taking hours (fixture_refresh 6414s, afternoon chain 66968s, scout "TIMED OUT after 5455s" vs 300s ceiling — `_run_with_ceiling` abandons the thread but it keeps running → orphan threads, server ~180% CPU). Morning chain re-started 3× on Nov 22 (08:14, 09:43, 09:48).
**Disk full**: 537 MB free. Cause: `safe_migrate` writes a full bets.db copy (~300 MB) on every DB init with no retention — 54 copies / 13 GB in `data/backups/safe_migrate/`. User-approved cleanup: kept newest 3, freed ~12 GB.
**Fix applied (user-approved exception to single-writer rule)**: historical.db backed up via `sqlite3.backup()` to `checkpoints/pre_w11_adhoc_20261122/`, then one-off `run_weekly_pipeline(2026, 11, fetch_props=False)` at 2026-11-22T16:00Z → 5,295 forward preds stored (162 players; 7 early + 3 late + SNF games, all pre-kickoff). 13 `fantasy_consensus` inserts failed `database is locked` (forward preds for those players were stored). MNF props not yet posted.

| # | Item | Status | Notes |
|---|------|--------|-------|
| OPS-1 | `safe_migrate` backup retention | **OPEN** | Add retention (keep N=3) + skip when free disk < 2× DB size. P0 — will refill disk in ~3 weeks. |
| OPS-2 | Chains block `_loop()` / orphaned timed-out threads | **OPEN** | Move NFL fantasy block before chains or run chains in own thread; ceiling must not leak threads; investigate morning-chain re-trigger loop. |
| OPS-3 | Restart server | **OPEN** | Running d4e0c2c vs disk f9321cd (`version_match=false`); restart after W11 Sunday slate to clear orphan threads. |
| OPS-4 | ESPN NFL injuries endpoint 400 | **OPEN** | Every fetch Nov 20–22 returned 400; NFL injury records = 0. Sleeper is the only injury source this weekend. |
| OPS-5 | NHL ATS model load `No module named '_loss'` | **OPEN** | sklearn pickle/version mismatch in betiq env; NHL sub-model unavailable, repeated every mock-gen. |
| OPS-6 | W11 grading date | **OPEN** | W11 settles after MNF (Nov 23); Tuesday grader Nov 24. FW-5/FW-6 "~2026-11-21" unblock estimate was early → now ~2026-11-24. |

**FW-2 evaluator**: `backend/analysis/nfl_fw2_rules.py` (read-only). `history` mode per-week table; `slate` mode upcoming picks. **Pricing note**: prereg says FanDuel price; the logged W10 numbers above were computed with the row's stored (latest-any-book) price — `--price stored` reproduces FW-2d W10 exactly (N=20, 12W, +14.4%). FanDuel-priced W10: FW-2c 18/23 +37.2% (14 picks lacked an FD price at the same line), FW-2d 11/18 +15.2%, FW-2e 5/12 −24.2%.
**W11 TNF (FD-priced)**: FW-2c 1-0 (+1 void: Deion Burks no stat line), FW-2d none, FW-2e 1-2. Season FW-2c N=129 79.8% WR vs 56.9% BE, ROI +40.8% [+27.1, +52.7] (game-clustered).

## Tested Hypotheses — Null Results

Record null results here so they are not re-tested ad hoc. Each entry has a re-test condition; do not re-run before that condition is met.

| Date | Hypothesis | Result | Re-test condition |
|------|-----------|--------|-------------------|
| 2026-11-15 | Team persistence in model direction accuracy | **NULL**. Split-half r ≈ 0 (odd/even: r=-0.052; W2-W6 vs W7-W10: r=+0.095). Top-8 OOS Δ ROI -0.8% [-14.0%, +11.5%]. Null after category × stat × price-band mix control (adjusted r=+0.101 early/late, +0.013 odd/even). Stability features (HR volatility, target concentration CV, rush volume std) all null. Decision: no team filters. | After W18 with full season (same split-half + OOS method, pre-specified). |

## Cleanup Reminders

| Item | Action | Deadline |
|------|--------|----------|
| `data/git_mirror_backup_20261105_1817` | Delete (pre-scrub backup; no longer needed after git-filter-repo confirmed clean) | 2026-11-19 |
| INFRA-1: Archive odds data | Move off-season sport rows to `props_archive.db` | No urgency — 17.8 GB free, ~14-month runway |
