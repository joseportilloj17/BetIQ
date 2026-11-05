# NBA Prop Model — Open Items

Last updated: 2026-11-05

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
| S3-2 | Draw counts given minutes × rate | **DEFERRED** | Adding NegBin requires rate_std decomposition (true_rate vs shot_noise) to avoid double-counting. For SGP k factors, marginal bias cancels in ratio. Revisit after S4 marginal calibration if PIT shows systematic discrete mismatch. |

### Step 4: Validation vs reality (4 seasons of box scores)

| # | Item | Status | Notes |
|---|------|--------|-------|
| S4-1 | Marginal calibration | **DONE** | PIT on 162,484 predictions (def_blow_recon_50pct, 2023-24). PTS/REB/AST starters well-calibrated (KS 0.04–0.07). STL/BLK badly miscalibrated (KS 0.19–0.23) — Normal approximation wrong for discrete low-count stats; NegBin or Poisson CDF needed. Empirical same-team residual correlations near zero: PTS r=-0.016±0.010, REB r=+0.006±0.008, AST r=-0.023±0.009. These are calibration TARGETS for the sim. |
| S4-2 | Joint calibration by combo type | **DONE** | Empirical k factors measured from 162,484 predictions across 1,230 games. Key findings: same-player PTS+REB k=1.20 (shared minutes + engagement), teammate PTS k=1.00 (independent), opposing PTS k=1.11 [1.00,1.22] (wide CI). **Sim changes**: (1) per-player engagement draw (eng_cv=0.15, PTS=1.0, REB=0.8, AST=0.0) matches PTS+REB k. (2) Scoring damp 0.50→0.30. (3) AST removed from team scoring mult. **Residual gaps**: teammate 3-leg PTS sim 1.05 vs emp 0.99 (shared pace slightly over-correlates); opposing PTS sim 1.01 vs emp 1.11 (no mechanistic fix, CI wide). |

### Step 5: B5 proper — calibration & comparison vs market

| # | Item | Status | Notes |
|---|------|--------|-------|
| S5-1 | Two-stage calibration | **DONE** | PIT re-verified post engagement draw. Holdout validation (H1→H2, H2→H1) showed all Normal corrections are one-way or dead → REMOVED. Gamma-Poisson for discrete stats is the only surviving structural fix. See S5-1 results below. |
| S5-2 | Comparison vs devigged market | **COLLECTING** | `nba_forward_preds` table in historical.db stores sim P(over/under), median, config at scout + pre-tip. Grading wired from `nba_player_game_logs`. **Effective day 1: 2026-11-06** (first scout+forward_preds run). Oct 30–Nov 5 LOST: bets.db corruption→server down Oct 31-Nov 2; recovery missed NBA scout Oct 30, Nov 3-5; code deployed after Nov 5 scout. N≥100 games target: ~Dec 2026. |
| S5-3 | Forward weeks separate | **BLOCKED** | 2026-27 NBA season. Target: first read at N=100 games (~Dec 2026). |
| S5-4 | Pre-register rules | **DONE** | See locked configuration below. |

#### S5-4: Pre-registered sim configuration (FROZEN 2026-10-30)

**Forward test day 1**: ~~2026-10-30~~ → **2026-11-06** (Oct 30–Nov 5 lost to bets.db corruption + server downtime + code deploy timing; see S5-2 status).

**Engine (nba_matchup_engine.py):**
- `reconcile: True`, `reconcile_strength: 0.5`, `reconcile_cap: 0.25`
- `defense: True`, `blowout: True`, `form: False`, `synergy: False`, `matchup: False`
- `reconcile_pool_fix: True` (OTHER bucket adjustment)

**Sim (nba_game_sim.py):**
- `_POSS_CV = 0.04` (pace variability)
- `_MARGIN_SIGMA = 15.8` (margin std, 4,668 games)
- `_GAME_TOTAL_STD = 19.4` (derived)
- `_TEAM_SCORING_DAMP = 0.30` (scoring efficiency dampening)
- `_ENGAGEMENT_CV = 0.15` (per-player good-game/bad-game)
- `_ENGAGEMENT_LOADINGS = {PTS: 1.0, REB: 0.8, AST: 0.0, STL: 0.0, BLK: 0.0, FG3M: 1.0, PRA: 0.8}`
- Team scoring mult applies to PTS, FG3M only (not AST)
- `_BLOWOUT_THRESHOLD = 15.0`, `_BASELINE_BLOWOUT_PROB = 0.326`
- `_OT_PROB = 0.0512`
- P(plays) rates: HAND-SET (Out=0, Doubtful=0.20, Questionable=0.65, Probable=0.95). Update rule pre-registered separately.
- In-game early exit: starter 3.5%, rotation 2%, spot 1%
- **Calibration corrections**: NONE — all Normal-distribution corrections (bias + σ_scale) failed holdout validation (fit on H1, test on H2 and vice versa). PTS/PRA corrections were one-way only (helped one half, hurt the other). Only structural fix survived: Gamma-Poisson for discrete stats.
- **Discrete distributions (Gamma-Poisson compound = exact NegBin):**
  - STL, BLK, FG3M: Gamma(α,β) rate draw → Poisson(λ = min × rate) — always
  - AST: Gamma-Poisson for rotation/spot, Normal for starters
  - Non-negative by construction (no floor at 0), integer output, NegBin-like overdispersion from game-to-game rate uncertainty
  - Gamma params: α = (μ/σ)², β = σ²/μ (matching mean and variance of per-minute rate)

**Decision rules (pre-registered):**
- Report P(over) conditioned on playing (void rule)
- SGP k from sim draws, not independent product
- Edge = |sim P(over) - devigged market P(over)| (requires market odds from prop capture)
- Threshold for bet: edge ≥ 5pp AND sim P(over) ∈ [0.30, 0.70] (uncertainty zone)
- Stat coverage: PTS, REB, AST, PRA, FG3M, STL, BLK (all stats; discrete stats use Gamma-Poisson)
- Player coverage: starters + rotation only (spot excluded)

**Holdout calibration validation (S5-1):**

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

## NFL Items (from existing plan)

| # | Item | Status | Notes |
|---|------|--------|-------|
| NFL-1 | Violation report + architecture (Phase 0) | NOT STARTED | `nfl_reconciliation_diagnostics.py` |
| NFL-2 | Coverage data audit (Phase 1) | NOT STARTED | `nfl_coverage_layer.py` |
| NFL-3 | Team volume reconciliation (Phase 2) | NOT STARTED | `nfl_volume_reconciliation.py` |
| NFL-4 | Coverage proxy (Phase 3) | BLOCKED by NFL-2 | Conditional on B1 results |
| NFL-5 | Bet-time logging (Phase 4) | NOT STARTED | model_version column + A/B |
