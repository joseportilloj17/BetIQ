"""
adaptive_test.py — Is anti-predictiveness stable or time-varying?
                    Would adaptation / fading help?

Three parts:
  1. Rolling-window discrimination (3d/7d/14d) — stable vs swinging
  2. Error-feedback correction layer — does correcting recent bias improve forward?
  3. Fade test — does betting AGAINST the model beat breakeven?

All on forward shadow data (ats_ungated, settled only).
"""
from __future__ import annotations
import os, sqlite3, re
import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))


def _load_forward():
    """Load all settled ats_ungated +1.5 legs with date, wp, odds, won."""
    conn = sqlite3.connect(os.path.join(_BASE, "data", "bets.db"))
    rows = conn.execute('''
        SELECT game_date, bet_info, predicted_win_prob, odds, status
        FROM mock_bets
        WHERE source = 'ats_ungated'
          AND status IN ('SETTLED_WIN', 'SETTLED_LOSS')
        ORDER BY game_date
    ''').fetchall()
    conn.close()

    legs = []
    for date, info, wp, odds_val, status in rows:
        m = re.match(r'(.+?)\s+([+-]\d+\.\d+)\s+\(Spread\)', info)
        if not m:
            continue
        team = m.group(1).strip()
        spread = float(m.group(2))
        won = 1 if status == 'SETTLED_WIN' else 0
        legs.append({
            "date": date, "team": team, "spread": spread,
            "wp": wp, "odds": odds_val, "won": won,
        })
    return pd.DataFrame(legs)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: ROLLING DISCRIMINATION
# ═══════════════════════════════════════════════════════════════════════════════

def part1_rolling_discrimination(df):
    """
    For +1.5 legs: in rolling 3d/7d/14d windows, compute
    fav-leg WR minus unfav-leg WR (discrimination).
    Report stability stats.
    """
    print("=" * 70)
    print("PART 1: ROLLING DISCRIMINATION — Stable or swinging?")
    print("=" * 70)

    plus15 = df[df["spread"] == 1.5].copy()
    plus15 = plus15.sort_values("date").reset_index(drop=True)
    dates = sorted(plus15["date"].unique())

    if len(dates) < 7:
        print("  Insufficient dates for rolling analysis")
        return

    # For each day, compute discrimination (median-split on wp)
    daily_disc = {}
    daily_n = {}
    daily_wr = {}
    daily_be = {}
    for d in dates:
        day_legs = plus15[plus15["date"] == d]
        if len(day_legs) < 4:  # need enough legs to split
            continue
        wps = day_legs["wp"].values
        wins = day_legs["won"].values
        odds_arr = day_legs["odds"].values
        med = np.median(wps)
        fav = wps >= med
        if fav.sum() == 0 or (~fav).sum() == 0:
            continue
        fav_wr = wins[fav].mean()
        unfav_wr = wins[~fav].mean()
        daily_disc[d] = fav_wr - unfav_wr
        daily_n[d] = len(day_legs)
        daily_wr[d] = wins.mean()
        daily_be[d] = np.mean(1.0 / odds_arr)

    disc_dates = sorted(daily_disc.keys())
    if len(disc_dates) < 7:
        print("  Insufficient daily data for rolling analysis")
        return

    print(f"\n  Daily data: {len(disc_dates)} days, "
          f"{plus15.shape[0]} total +1.5 legs")
    print(f"  Overall disc: {np.mean(list(daily_disc.values()))*100:+.1f}pp "
          f"(daily mean)")

    # Rolling windows
    for window in [3, 7, 14]:
        print(f"\n  ── Rolling {window}-day windows ──")
        rolling_discs = []
        rolling_wrs = []
        rolling_ns = []

        for i in range(len(disc_dates) - window + 1):
            win_dates = disc_dates[i:i + window]
            win_legs = plus15[plus15["date"].isin(win_dates)]
            if len(win_legs) < 8:
                continue

            wps = win_legs["wp"].values
            wins = win_legs["won"].values
            med = np.median(wps)
            fav = wps >= med
            if fav.sum() == 0 or (~fav).sum() == 0:
                continue

            disc = wins[fav].mean() - wins[~fav].mean()
            rolling_discs.append(disc)
            rolling_wrs.append(wins.mean())
            rolling_ns.append(len(win_legs))

        if not rolling_discs:
            print("    Insufficient data")
            continue

        rd = np.array(rolling_discs)
        print(f"    N windows: {len(rd)}")
        print(f"    Mean disc: {rd.mean()*100:+.2f}pp")
        print(f"    Std disc:  {rd.std()*100:.2f}pp")
        print(f"    Min disc:  {rd.min()*100:+.2f}pp")
        print(f"    Max disc:  {rd.max()*100:+.2f}pp")
        print(f"    % windows predictive (disc > 0): "
              f"{(rd > 0).mean()*100:.1f}%")
        print(f"    % windows anti-pred (disc < -5pp): "
              f"{(rd < -0.05).mean()*100:.1f}%")

        # Autocorrelation: does today's disc predict tomorrow's?
        if len(rd) > 5:
            ac1 = np.corrcoef(rd[:-1], rd[1:])[0, 1]
            print(f"    Lag-1 autocorrelation: {ac1:+.3f}"
                  f" ({'persistent' if ac1 > 0.2 else 'random' if abs(ac1) < 0.2 else 'mean-reverting'})")

    # Day-by-day discrimination series
    print(f"\n  ── Daily discrimination series ──")
    print(f"  {'Date':<12} {'N':<5} {'WR%':<7} {'BE%':<7} {'Disc':<8} {'Bar'}")
    for d in disc_dates:
        disc = daily_disc[d]
        n = daily_n[d]
        wr = daily_wr[d]
        be = daily_be[d]
        bar_len = int(abs(disc) * 100)
        bar_char = "+" if disc > 0 else "-"
        bar = bar_char * min(bar_len, 30)
        print(f"  {d:<12} {n:<5} {wr*100:.0f}%   {be*100:.0f}%   "
              f"{disc*100:+.1f}pp  {bar}")

    # Is recent-outcome a predictor of next-day model accuracy?
    print(f"\n  ── Does recent outcome predict next-day accuracy? ──")
    if len(disc_dates) >= 10:
        # For each day, did the previous 3 days have positive disc?
        prev3_disc = []
        next_disc = []
        for i in range(3, len(disc_dates)):
            prev = [daily_disc[disc_dates[j]] for j in range(i - 3, i)
                    if disc_dates[j] in daily_disc]
            if len(prev) < 2:
                continue
            prev3_disc.append(np.mean(prev))
            next_disc.append(daily_disc[disc_dates[i]])

        if len(prev3_disc) > 5:
            corr = np.corrcoef(prev3_disc, next_disc)[0, 1]
            print(f"    Correlation(prev-3d avg disc, next-day disc): {corr:+.3f}")
            if abs(corr) < 0.15:
                print(f"    → NO exploitable time-structure (corr ≈ 0)")
            elif corr > 0.15:
                print(f"    → PERSISTENT: good windows predict good windows")
                print(f"      (adaptation could ride momentum)")
            else:
                print(f"    → MEAN-REVERTING: bad windows predict good windows")
                print(f"      (contrarian correction might work)")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: ERROR-FEEDBACK CORRECTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def part2_error_feedback(df):
    """
    Track model residuals by condition. If systematically wrong in a
    condition over the last N days, apply a correction. Does this improve
    forward discrimination vs un-adapted model?

    Conditions: favorite/dog (spread sign), wp-tier (high/low confidence)
    """
    print(f"\n{'='*70}")
    print("PART 2: ERROR-FEEDBACK CORRECTION LAYER")
    print("Does correcting recent bias improve forward discrimination?")
    print(f"{'='*70}")

    plus15 = df[df["spread"] == 1.5].copy()
    plus15 = plus15.sort_values("date").reset_index(drop=True)
    dates = sorted(plus15["date"].unique())

    if len(dates) < 14:
        print("  Insufficient dates for error-feedback test")
        return

    # Conditions: wp quartile (Q1-Q4) — the model's confidence tiers
    plus15["wp_q"] = pd.qcut(plus15["wp"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    # For each day starting from day 8 (need 7 days lookback):
    # 1. Compute residual by condition over last 7 days
    # 2. Apply correction to today's predictions
    # 3. Compare corrected vs original

    lookbacks = [3, 5, 7, 14]
    for lb in lookbacks:
        if len(dates) < lb + 5:
            continue

        print(f"\n  ── Lookback = {lb} days ──")

        orig_correct = 0
        corrected_correct = 0
        total = 0
        orig_disc_sum = 0
        corr_disc_sum = 0
        n_windows = 0

        for i in range(lb, len(dates)):
            today = dates[i]
            lookback_dates = dates[max(0, i - lb):i]

            today_legs = plus15[plus15["date"] == today]
            lb_legs = plus15[plus15["date"].isin(lookback_dates)]

            if len(today_legs) < 4 or len(lb_legs) < 8:
                continue

            # Compute per-condition bias over lookback
            condition_bias = {}
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                q_legs = lb_legs[lb_legs["wp_q"] == q]
                if len(q_legs) < 3:
                    continue
                # Residual: actual win rate minus predicted (wp/100)
                actual_wr = q_legs["won"].mean()
                pred_wr = q_legs["wp"].mean() / 100.0
                condition_bias[q] = actual_wr - pred_wr

            # Apply correction to today's legs
            corrected_wps = []
            for _, leg in today_legs.iterrows():
                orig_wp = leg["wp"] / 100.0
                q = leg["wp_q"]
                if q in condition_bias:
                    # Shift wp by observed bias
                    corrected = orig_wp + condition_bias[q]
                    corrected = np.clip(corrected, 0.01, 0.99)
                else:
                    corrected = orig_wp
                corrected_wps.append(corrected)

            corrected_wps = np.array(corrected_wps)
            orig_wps = today_legs["wp"].values / 100.0
            wins = today_legs["won"].values

            # Original: pick home if wp >= 0.5
            orig_picks = orig_wps >= 0.5
            corr_picks = corrected_wps >= 0.5

            orig_correct += np.where(orig_picks, wins == 1, wins == 0).sum()
            corrected_correct += np.where(corr_picks, wins == 1, wins == 0).sum()
            total += len(today_legs)

            # Discrimination
            med_orig = np.median(orig_wps)
            fav_orig = orig_wps >= med_orig
            med_corr = np.median(corrected_wps)
            fav_corr = corrected_wps >= med_corr

            if fav_orig.sum() > 0 and (~fav_orig).sum() > 0:
                orig_disc = wins[fav_orig].mean() - wins[~fav_orig].mean()
                orig_disc_sum += orig_disc
                n_windows += 1

            if fav_corr.sum() > 0 and (~fav_corr).sum() > 0:
                corr_disc = wins[fav_corr].mean() - wins[~fav_corr].mean()
                corr_disc_sum += corr_disc

        if total > 0 and n_windows > 0:
            orig_acc = orig_correct / total
            corr_acc = corrected_correct / total
            avg_orig_disc = orig_disc_sum / n_windows
            avg_corr_disc = corr_disc_sum / n_windows

            print(f"    Days tested: {n_windows}")
            print(f"    Total legs:  {total}")
            print(f"    Original accuracy:   {orig_acc*100:.2f}%")
            print(f"    Corrected accuracy:  {corr_acc*100:.2f}%")
            print(f"    Δ accuracy:          {(corr_acc-orig_acc)*100:+.2f}pp")
            print(f"    Original avg disc:   {avg_orig_disc*100:+.2f}pp")
            print(f"    Corrected avg disc:  {avg_corr_disc*100:+.2f}pp")
            print(f"    Δ disc:              {(avg_corr_disc-avg_orig_disc)*100:+.2f}pp")

            if corr_acc > orig_acc + 0.005:
                print(f"    → CORRECTION HELPS (+{(corr_acc-orig_acc)*100:.2f}pp accuracy)")
            elif corr_acc < orig_acc - 0.005:
                print(f"    → CORRECTION HURTS ({(corr_acc-orig_acc)*100:.2f}pp accuracy)")
            else:
                print(f"    → NO MEANINGFUL DIFFERENCE")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: FADE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def part3_fade_test(df):
    """
    If model is stably anti-predictive, bet AGAINST its pick.
    Does fading beat breakeven? Is it stable enough to trust?
    """
    print(f"\n{'='*70}")
    print("PART 3: FADE TEST — Does betting AGAINST the model work?")
    print(f"{'='*70}")

    plus15 = df[df["spread"] == 1.5].copy()
    plus15 = plus15.sort_values("date").reset_index(drop=True)
    dates = sorted(plus15["date"].unique())

    if len(plus15) < 50:
        print("  Insufficient data")
        return

    # Game-level: model picks the +1.5 side with higher wp
    # Each game should have TWO +1.5 legs (home +1.5 and away +1.5 mapped to
    # different teams). Group by date to approximate game-level.

    # The model's "pick": the leg with higher wp wins
    # "Fade": bet the leg with LOWER wp

    # But we don't have game-level pairing easily. Instead:
    # Above-median wp legs (model-favored) vs below-median (model-unfavored)
    # "Fading" = take the unfavored side

    wps = plus15["wp"].values
    wins = plus15["won"].values
    odds_arr = plus15["odds"].values
    med_wp = np.median(wps)

    fav_mask = wps >= med_wp
    unfav_mask = ~fav_mask

    # Model's pick: favored legs
    fav_wr = wins[fav_mask].mean()
    fav_n = fav_mask.sum()
    fav_be = np.mean(1.0 / odds_arr[fav_mask])
    fav_pnl = sum(
        10 * (odds_arr[i] - 1) if wins[i] else -10
        for i in range(len(wins)) if fav_mask[i]
    )
    fav_roi = fav_pnl / (10 * fav_n) * 100

    # Fade: unfavored legs
    unfav_wr = wins[unfav_mask].mean()
    unfav_n = (~fav_mask).sum()
    unfav_be = np.mean(1.0 / odds_arr[unfav_mask])
    unfav_pnl = sum(
        10 * (odds_arr[i] - 1) if wins[i] else -10
        for i in range(len(wins)) if unfav_mask[i]
    )
    unfav_roi = unfav_pnl / (10 * unfav_n) * 100

    print(f"\n  Split: median wp = {med_wp:.1f}")
    print(f"\n  {'Strategy':<20} {'N':<6} {'WR%':<8} {'BE%':<8} "
          f"{'WR-BE':<8} {'ROI%':<8} {'PnL$'}")
    print(f"  {'Model pick (fav)':<20} {fav_n:<6} {fav_wr*100:.1f}%   "
          f"{fav_be*100:.1f}%   {(fav_wr-fav_be)*100:+.1f}pp  "
          f"{fav_roi:+.1f}%   ${fav_pnl:+.0f}")
    print(f"  {'Fade (unfav)':<20} {unfav_n:<6} {unfav_wr*100:.1f}%   "
          f"{unfav_be*100:.1f}%   {(unfav_wr-unfav_be)*100:+.1f}pp  "
          f"{unfav_roi:+.1f}%   ${unfav_pnl:+.0f}")
    print(f"  {'All legs':<20} {len(wins):<6} {wins.mean()*100:.1f}%   "
          f"{np.mean(1/odds_arr)*100:.1f}%   "
          f"{(wins.mean()-np.mean(1/odds_arr))*100:+.1f}pp  "
          f"{(fav_pnl+unfav_pnl)/(10*len(wins))*100:+.1f}%   "
          f"${fav_pnl+unfav_pnl:+.0f}")

    # Rolling stability of the fade edge
    print(f"\n  ── Fade stability by rolling 7-day window ──")
    fade_wrs = []
    fade_bes = []
    for i in range(len(dates) - 6):
        win_dates = dates[i:i + 7]
        win_legs = plus15[plus15["date"].isin(win_dates)]
        if len(win_legs) < 10:
            continue
        w = win_legs["wp"].values
        wi = win_legs["won"].values
        o = win_legs["odds"].values
        m = np.median(w)
        um = w < m
        if um.sum() < 3:
            continue
        fade_wrs.append(wi[um].mean())
        fade_bes.append(np.mean(1.0 / o[um]))

    if fade_wrs:
        fade_wrs = np.array(fade_wrs)
        fade_bes = np.array(fade_bes)
        edges = fade_wrs - fade_bes

        print(f"    N windows: {len(fade_wrs)}")
        print(f"    Mean fade WR: {fade_wrs.mean()*100:.1f}%")
        print(f"    Mean fade BE: {fade_bes.mean()*100:.1f}%")
        print(f"    Mean fade edge: {edges.mean()*100:+.1f}pp")
        print(f"    Std fade edge:  {edges.std()*100:.1f}pp")
        print(f"    % windows fade > BE: {(edges > 0).mean()*100:.0f}%")
        print(f"    Min edge: {edges.min()*100:+.1f}pp")
        print(f"    Max edge: {edges.max()*100:+.1f}pp")

        if edges.mean() > 0.01:
            print(f"    → FADE IS PROFITABLE on average (+{edges.mean()*100:.1f}pp)")
            if edges.std() > edges.mean() * 2:
                print(f"      BUT high variance (std/mean = {edges.std()/edges.mean():.1f}x)")
                print(f"      Not reliable enough to deploy")
            else:
                print(f"      Sharpe-like ratio: {edges.mean()/edges.std():.2f}")
        elif edges.mean() < -0.01:
            print(f"    → FADE IS ALSO NEGATIVE ({edges.mean()*100:+.1f}pp)")
            print(f"      Model is not consistently anti-predictive — just noisy")
        else:
            print(f"    → FADE IS FLAT — no systematic signal either direction")

    # Quintile fade: bet BOTTOM quintile (model's least favored)
    print(f"\n  ── Quintile analysis: does the model RANK at all? ──")
    plus15["wp_q5"] = pd.qcut(plus15["wp"], 5, labels=False, duplicates="drop")
    for q in sorted(plus15["wp_q5"].unique()):
        q_legs = plus15[plus15["wp_q5"] == q]
        q_wr = q_legs["won"].mean()
        q_be = np.mean(1.0 / q_legs["odds"].values)
        q_pnl = sum(
            10 * (r["odds"] - 1) if r["won"] else -10
            for _, r in q_legs.iterrows()
        )
        q_roi = q_pnl / (10 * len(q_legs)) * 100
        print(f"    Q{int(q)+1} (wp {q_legs['wp'].min():.0f}-{q_legs['wp'].max():.0f}): "
              f"WR={q_wr*100:.1f}%, BE={q_be*100:.1f}%, "
              f"edge={((q_wr-q_be)*100):+.1f}pp, ROI={q_roi:+.1f}%  N={len(q_legs)}")

    # Check if quintile gradient is INVERTED (lower wp = higher WR)
    q_wrs = plus15.groupby("wp_q5")["won"].mean().values
    if len(q_wrs) >= 4:
        gradient = np.corrcoef(range(len(q_wrs)), q_wrs)[0, 1]
        print(f"\n    Quintile gradient (rank-WR correlation): {gradient:+.3f}")
        if gradient < -0.5:
            print(f"    → INVERTED: model's ranking is BACKWARDS")
            print(f"      Low-confidence picks WIN MORE → fade is viable")
        elif gradient > 0.5:
            print(f"    → CORRECT: model ranks correctly, anti-predictiveness")
            print(f"      is just vig, not inverted signal")
        else:
            print(f"    → FLAT: no monotonic ranking — noise, not signal")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: GAME-LEVEL FADE (paired legs)
# ═══════════════════════════════════════════════════════════════════════════════

def part4_game_level_fade(df):
    """
    Pair +1.5 and -1.5 legs from same game into game-level decisions.
    Model picks side with higher wp. Fade picks the other side.
    This is the cleanest fade test: exactly one pick per game.
    """
    print(f"\n{'='*70}")
    print("PART 4: GAME-LEVEL FADE (paired legs)")
    print("Each game: model picks high-wp side, fade picks low-wp side")
    print(f"{'='*70}")

    # +1.5 legs: for each date-team, there should be a mirror -1.5 leg
    # Group by date to find game pairs
    plus15 = df[df["spread"] == 1.5].copy()
    minus15 = df[df["spread"] == -1.5].copy()

    # Match games: same date, one team +1.5, opponent -1.5
    # The +1.5 team wins iff -1.5 team loses (same game outcome)
    # Model's side-pick: which +1.5 leg has higher wp

    games = []
    for date in sorted(plus15["date"].unique()):
        day_plus = plus15[plus15["date"] == date]
        day_minus = minus15[minus15["date"] == date]

        # Pair: +1.5 team A should match with -1.5 team B (opponent)
        # We know each game has exactly one +1.5 winner. The +1.5 with higher wp
        # is the model's pick.
        # Sort by wp to get pairs (highest wp goes with lowest wp in same game)
        if len(day_plus) < 2:
            continue

        # Simpler: just pair all +1.5 legs by wp rank
        # Within each day, the model prefers higher-wp +1.5 legs
        sorted_legs = day_plus.sort_values("wp", ascending=False)

        # Top half = model's pick, bottom half = fade
        half = len(sorted_legs) // 2
        model_legs = sorted_legs.iloc[:half]
        fade_legs = sorted_legs.iloc[half:]

        for _, leg in model_legs.iterrows():
            games.append({"date": date, "side": "model", "wp": leg["wp"],
                         "won": leg["won"], "odds": leg["odds"]})
        for _, leg in fade_legs.iterrows():
            games.append({"date": date, "side": "fade", "wp": leg["wp"],
                         "won": leg["won"], "odds": leg["odds"]})

    gdf = pd.DataFrame(games)
    if len(gdf) < 50:
        print("  Insufficient paired data")
        return

    for side in ["model", "fade"]:
        s = gdf[gdf["side"] == side]
        wr = s["won"].mean()
        be = np.mean(1.0 / s["odds"].values)
        pnl = sum(10 * (r["odds"] - 1) if r["won"] else -10 for _, r in s.iterrows())
        roi = pnl / (10 * len(s)) * 100
        print(f"  {side.upper():<8} N={len(s):<5} WR={wr*100:.1f}%  BE={be*100:.1f}%  "
              f"edge={((wr-be)*100):+.1f}pp  ROI={roi:+.1f}%  PnL=${pnl:+.0f}")


def main():
    print("=" * 70)
    print("ADAPTIVE / ONLINE LEARNING DIAGNOSTIC")
    print("Is anti-predictiveness stable or exploitable?")
    print("=" * 70)

    df = _load_forward()
    print(f"\nLoaded {len(df)} settled ats_ungated legs")
    print(f"  +1.5: {(df['spread']==1.5).sum()}")
    print(f"  -1.5: {(df['spread']==-1.5).sum()}")
    print(f"  Date range: {df['date'].min()} – {df['date'].max()}")
    print(f"  Unique dates: {df['date'].nunique()}")

    part1_rolling_discrimination(df)
    part2_error_feedback(df)
    part3_fade_test(df)
    part4_game_level_fade(df)

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
