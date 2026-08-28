"""
side_selection_diagnostic.py — The decisive test:
  Can the model predict GAME WINNERS (moneyline) even though it can't
  beat the SPREAD?

If yes → efficient market (spread absorbs all available info)
If no  → bad model (room for improvement with better modeling)

Also: does the market's own line (closing ML implied prob) predict
better than our model? If so, by how much — that gap is the
"market knows more than us" measure.
"""
from __future__ import annotations
import os, sys, sqlite3, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_BASE, "backend"))


def _implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds is None or np.isnan(american_odds):
        return np.nan
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def _load_data():
    """Load feature matrix + game outcomes + closing moneylines."""
    from features import (build_feature_matrix, MODEL_FEATURE_COLS_V1_PRUNED,
                          TARGET_ATS, TARGET_OU)

    fm = build_feature_matrix("MLB")
    fm["game_pk"] = fm["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")
    fm["year"] = pd.to_datetime(fm["game_date"]).dt.year

    # Get actual scores for moneyline outcome
    conn = sqlite3.connect(os.path.join(_BASE, "data", "historical.db"))
    games = pd.read_sql_query("""
        SELECT game_id, home_score, away_score
        FROM games
        WHERE sport = 'MLB'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """, conn)
    games["game_pk"] = games["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # Get closing moneylines (market implied probabilities)
    ml = pd.read_sql_query("""
        SELECT game_id, close_ml_home, close_ml_away
        FROM betting_lines
        WHERE sport = 'MLB'
          AND close_ml_home IS NOT NULL
          AND bookmaker = 'consensus'
    """, conn)
    if len(ml) == 0:
        # Try any bookmaker
        ml = pd.read_sql_query("""
            SELECT game_id, close_ml_home, close_ml_away,
                   ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY bookmaker) as rn
            FROM betting_lines
            WHERE sport = 'MLB'
              AND close_ml_home IS NOT NULL
        """, conn)
        ml = ml[ml["rn"] == 1].drop(columns=["rn"])

    ml["game_pk"] = ml["game_id"].str.replace("MLB_", "").astype(int, errors="ignore")
    ml["market_home_prob"] = ml["close_ml_home"].apply(_implied_prob)
    ml["market_away_prob"] = ml["close_ml_away"].apply(_implied_prob)
    # Devig: normalize so probs sum to 1
    total = ml["market_home_prob"] + ml["market_away_prob"]
    ml["market_home_prob_dv"] = ml["market_home_prob"] / total

    conn.close()

    merged = fm.merge(games[["game_pk", "home_win"]], on="game_pk", how="inner")
    merged = merged.merge(ml[["game_pk", "market_home_prob_dv", "close_ml_home", "close_ml_away"]],
                          on="game_pk", how="left")
    merged = merged.sort_values("game_date").reset_index(drop=True)

    return merged, MODEL_FEATURE_COLS_V1_PRUNED, TARGET_ATS


def _train_gbc(X_train, y_train, X_test, seed=42):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=seed)
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]


def part1_moneyline_vs_spread(merged, feat_cols, target_ats):
    """
    Train model on two targets using same features:
      A) home_win (moneyline outcome — did home team WIN?)
      B) covered_spread (ATS outcome — did home team COVER -1.5?)

    Compare AUC, quintile gradient, discrimination.
    Walk-forward: train on years < test_year, test on test_year.
    """
    print("=" * 70)
    print("PART 1: MODEL SKILL — Moneyline (game winner) vs Spread (ATS)")
    print("Same features, same GBC, different targets")
    print("=" * 70)

    available = [c for c in feat_cols if c in merged.columns]

    targets = {
        "Moneyline (home win)": "home_win",
        "Spread (covered -1.5)": target_ats,
    }

    for test_year in [2024, 2025, 2026]:
        test_mask = merged["year"] == test_year
        train_mask = merged["year"] < test_year

        test_df = merged[test_mask].copy()
        train_df = merged[train_mask].copy()

        if len(test_df) < 100 or len(train_df) < 500:
            continue

        print(f"\n  ── Test year: {test_year} ──")
        print(f"  Train: {len(train_df)} games, Test: {len(test_df)} games")

        for target_name, target_col in targets.items():
            t_train = train_df.dropna(subset=[target_col])
            t_test = test_df.dropna(subset=[target_col])

            if len(t_test) < 50:
                print(f"    {target_name}: insufficient test data")
                continue

            X_tr = t_train[available].fillna(t_train[available].median())
            y_tr = t_train[target_col].values.astype(int)
            X_te = t_test[available].fillna(t_test[available].median())
            y_te = t_test[target_col].values.astype(int)

            base_rate = y_te.mean()
            naive_wr = max(base_rate, 1 - base_rate)

            # Average across 3 seeds
            aucs = []
            wrs = []
            all_probs = []
            for seed in range(3):
                probs = _train_gbc(X_tr, y_tr, X_te, seed=seed)
                aucs.append(roc_auc_score(y_te, probs))
                picks = (probs >= 0.5).astype(int)
                wrs.append((picks == y_te).mean())
                all_probs.append(probs)

            avg_probs = np.mean(all_probs, axis=0)
            auc = np.mean(aucs)
            wr = np.mean(wrs)

            # Quintile analysis
            q_labels = pd.qcut(avg_probs, 5, labels=False, duplicates="drop")
            q_wrs = []
            for q in sorted(set(q_labels)):
                mask = q_labels == q
                q_wrs.append(y_te[mask].mean())

            gradient = np.corrcoef(range(len(q_wrs)), q_wrs)[0, 1] if len(q_wrs) >= 3 else 0

            print(f"\n    {target_name}:")
            print(f"      AUC:        {auc:.4f}  ({'skill' if auc > 0.55 else 'weak' if auc > 0.52 else 'no skill'})")
            print(f"      WR:         {wr*100:.1f}%  (base={base_rate*100:.1f}%, naive={naive_wr*100:.1f}%)")
            print(f"      WR-naive:   {(wr-naive_wr)*100:+.1f}pp")
            print(f"      Gradient:   {gradient:+.3f}  ({'monotonic' if abs(gradient) > 0.8 else 'some' if abs(gradient) > 0.5 else 'flat'})")
            print(f"      Quintiles:  ", end="")
            for i, qw in enumerate(q_wrs):
                print(f"Q{i+1}={qw*100:.0f}%  ", end="")
            print()

            # Market comparison (only for moneyline target)
            if target_col == "home_win" and "market_home_prob_dv" in t_test.columns:
                mkt = t_test["market_home_prob_dv"].values
                valid_mkt = ~np.isnan(mkt)
                if valid_mkt.sum() > 50:
                    mkt_auc = roc_auc_score(y_te[valid_mkt], mkt[valid_mkt])
                    mkt_picks = (mkt[valid_mkt] >= 0.5).astype(int)
                    mkt_wr = (mkt_picks == y_te[valid_mkt]).mean()
                    print(f"      Market AUC: {mkt_auc:.4f}  (closing ML implied)")
                    print(f"      Market WR:  {mkt_wr*100:.1f}%")
                    print(f"      Model-Mkt:  {(auc-mkt_auc):+.4f} AUC gap")
                    if auc < mkt_auc - 0.02:
                        print(f"      → Market BEATS model — market has more info")
                    elif auc > mkt_auc + 0.02:
                        print(f"      → Model BEATS market — model has unpriced info!")
                    else:
                        print(f"      → COMPARABLE — model ≈ market")


def part2_spread_residual_analysis(merged, feat_cols, target_ats):
    """
    Key question: does the model learn ANYTHING about game direction
    that's independent of what the spread already prices?

    Test: after conditioning on the closing spread, does the model
    add information? (Partial correlation / residual prediction)
    """
    print(f"\n{'='*70}")
    print("PART 2: DOES THE MODEL ADD INFO BEYOND THE SPREAD?")
    print("After controlling for close_spread, does model_wp predict outcomes?")
    print(f"{'='*70}")

    available = [c for c in feat_cols if c in merged.columns]

    for test_year in [2025, 2026]:
        test_mask = merged["year"] == test_year
        train_mask = merged["year"] < test_year

        test_df = merged[test_mask].dropna(subset=[target_ats, "home_win"]).copy()
        train_df = merged[train_mask].dropna(subset=[target_ats, "home_win"]).copy()

        if len(test_df) < 100 or len(train_df) < 500:
            continue

        # Train model on home_win target
        X_tr = train_df[available].fillna(train_df[available].median())
        y_tr = train_df["home_win"].values.astype(int)
        X_te = test_df[available].fillna(test_df[available].median())
        y_te_ml = test_df["home_win"].values.astype(int)
        y_te_ats = test_df[target_ats].values.astype(int)

        probs_ml = _train_gbc(X_tr, y_tr, X_te)

        # Market prob
        mkt_prob = test_df["market_home_prob_dv"].values

        print(f"\n  ── Test year: {test_year} (N={len(test_df)}) ──")

        # Bin by market-implied probability (what the spread/ML already prices)
        valid = ~np.isnan(mkt_prob)
        if valid.sum() < 50:
            print("    Insufficient market data")
            continue

        test_v = test_df[valid].copy()
        probs_v = probs_ml[valid]
        mkt_v = mkt_prob[valid]
        y_ml_v = y_te_ml[valid]
        y_ats_v = y_te_ats[valid]

        # Within each market-tier, does the model discriminate?
        test_v["mkt_q"] = pd.qcut(mkt_v, 4, labels=["Underdog", "Slight dog", "Slight fav", "Favorite"])
        test_v["model_wp"] = probs_v

        print(f"\n    Within market tiers — does model add info?")
        print(f"    {'Market tier':<15} {'N':<6} {'Model AUC':<11} {'Mkt AUC':<9} {'Model adds?'}")

        for tier in ["Underdog", "Slight dog", "Slight fav", "Favorite"]:
            tier_mask = test_v["mkt_q"] == tier
            tier_y = y_ml_v[tier_mask.values]
            tier_model = probs_v[tier_mask.values]
            tier_mkt = mkt_v[tier_mask.values]

            if len(tier_y) < 20 or tier_y.std() == 0:
                continue

            try:
                m_auc = roc_auc_score(tier_y, tier_model)
                mk_auc = roc_auc_score(tier_y, tier_mkt)
                adds = "YES" if m_auc > mk_auc + 0.02 else ("maybe" if m_auc > mk_auc else "no")
                print(f"    {tier:<15} {len(tier_y):<6} {m_auc:.4f}     {mk_auc:.4f}   {adds}")
            except ValueError:
                print(f"    {tier:<15} {len(tier_y):<6} (insufficient variance)")

        # Overall: model residuals after removing market
        # Regress model_wp on market_wp, take residual, check if residual predicts
        from numpy.polynomial import polynomial as P
        coefs = P.polyfit(mkt_v, probs_v, deg=1)
        model_resid = probs_v - P.polyval(mkt_v, coefs)

        try:
            resid_auc = roc_auc_score(y_ml_v, model_resid + 0.5)
            print(f"\n    Model residual (after removing market) AUC: {resid_auc:.4f}")
            if resid_auc > 0.52:
                print(f"    → Model has info BEYOND market (+{(resid_auc-0.5)*100:.1f}pp)")
            else:
                print(f"    → Model adds NOTHING beyond market (≈ 0.50)")
        except ValueError:
            print(f"\n    Could not compute residual AUC")


def part3_forward_moneyline_check(merged, feat_cols):
    """
    For the forward shadow period: the model's wp is a spread-cover prediction.
    Does higher model_wp correlate with the team actually WINNING the game
    (not just covering)?

    This uses ats_ungated legs matched to game outcomes.
    """
    print(f"\n{'='*70}")
    print("PART 3: FORWARD — Does model wp predict GAME WINNERS?")
    print("Using ats_ungated shadow legs + actual game outcomes")
    print(f"{'='*70}")

    import re

    conn_bets = sqlite3.connect(os.path.join(_BASE, "data", "bets.db"))
    rows = conn_bets.execute('''
        SELECT game_date, bet_info, predicted_win_prob, odds, status
        FROM mock_bets
        WHERE source = 'ats_ungated'
          AND status IN ('SETTLED_WIN', 'SETTLED_LOSS')
    ''').fetchall()
    conn_bets.close()

    legs = []
    for date, info, wp, odds_val, status in rows:
        m = re.match(r'(.+?)\s+([+-]\d+\.\d+)\s+\(Spread\)', info)
        if not m:
            continue
        team = m.group(1).strip()
        spread = float(m.group(2))
        won = 1 if status == 'SETTLED_WIN' else 0
        legs.append({"date": date, "team": team, "spread": spread,
                     "wp": wp, "odds": odds_val, "won_ats": won})

    legs_df = pd.DataFrame(legs)

    # Get game outcomes for the forward period
    conn_hist = sqlite3.connect(os.path.join(_BASE, "data", "historical.db"))
    games = pd.read_sql_query("""
        SELECT game_id, game_date, home_team, away_team, home_score, away_score
        FROM games
        WHERE sport = 'MLB'
          AND home_score IS NOT NULL
          AND game_date >= '2026-07-06'
    """, conn_hist)
    conn_hist.close()

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # Match legs to games: +1.5 teams should be matchable
    # For +1.5 legs, the team is the home team (home +1.5) or away team (away +1.5)
    # We need to figure out if the leg's team is home or away
    # Then check if that team won the game outright

    plus15 = legs_df[legs_df["spread"] == 1.5].copy()

    # Match: for each +1.5 leg, find the game where team is home or away
    matched = []
    for _, leg in plus15.iterrows():
        team = leg["team"]
        date = leg["date"]
        # Check if team is home
        g_home = games[(games["game_date"] == date) & (games["home_team"] == team)]
        if len(g_home) > 0:
            g = g_home.iloc[0]
            team_won_game = int(g["home_score"] > g["away_score"])
            matched.append({**leg.to_dict(), "team_won_game": team_won_game,
                           "is_home": True, "margin": g["home_score"] - g["away_score"]})
            continue

        # Check if team is away
        g_away = games[(games["game_date"] == date) & (games["away_team"] == team)]
        if len(g_away) > 0:
            g = g_away.iloc[0]
            team_won_game = int(g["away_score"] > g["home_score"])
            matched.append({**leg.to_dict(), "team_won_game": team_won_game,
                           "is_home": False, "margin": g["away_score"] - g["home_score"]})

    mdf = pd.DataFrame(matched)
    if len(mdf) < 50:
        print(f"  Only {len(mdf)} legs matched to game outcomes — insufficient")
        return

    print(f"\n  Matched {len(mdf)} +1.5 legs to game outcomes")
    print(f"  {mdf['is_home'].sum()} home legs, {(~mdf['is_home']).sum()} away legs")

    # Model wp predicts spread cover. Does it also predict game winner?
    try:
        ats_auc = roc_auc_score(mdf["won_ats"], mdf["wp"])
        ml_auc = roc_auc_score(mdf["team_won_game"], mdf["wp"])
    except ValueError:
        print("  Could not compute AUC (constant target)")
        return

    print(f"\n  Model wp AUC for spread cover: {ats_auc:.4f}")
    print(f"  Model wp AUC for game winner:  {ml_auc:.4f}")

    # Quintile analysis for game winner
    mdf["wp_q"] = pd.qcut(mdf["wp"], 5, labels=False, duplicates="drop")
    print(f"\n  Quintile analysis — model wp vs game outcomes:")
    print(f"  {'Q':<4} {'WP range':<15} {'N':<5} {'Spread WR':<11} {'Game Win%':<11} {'Margin'}")
    for q in sorted(mdf["wp_q"].unique()):
        qm = mdf[mdf["wp_q"] == q]
        print(f"  Q{int(q)+1}  wp {qm['wp'].min():.0f}-{qm['wp'].max():.0f}     "
              f"{len(qm):<5} {qm['won_ats'].mean()*100:.1f}%      "
              f"{qm['team_won_game'].mean()*100:.1f}%      "
              f"{qm['margin'].mean():+.1f}")

    # Gradient for game winners
    q_game_wrs = mdf.groupby("wp_q")["team_won_game"].mean().values
    gradient = np.corrcoef(range(len(q_game_wrs)), q_game_wrs)[0, 1] if len(q_game_wrs) >= 3 else 0
    print(f"\n  Game-winner quintile gradient: {gradient:+.3f}")

    # Discrimination: above-median wp vs below-median wp
    med = mdf["wp"].median()
    fav = mdf["wp"] >= med
    fav_game_wr = mdf.loc[fav, "team_won_game"].mean()
    unfav_game_wr = mdf.loc[~fav, "team_won_game"].mean()
    fav_ats_wr = mdf.loc[fav, "won_ats"].mean()
    unfav_ats_wr = mdf.loc[~fav, "won_ats"].mean()

    print(f"\n  Discrimination (median split):")
    print(f"    Spread cover:  fav={fav_ats_wr*100:.1f}%, unfav={unfav_ats_wr*100:.1f}%, "
          f"disc={((fav_ats_wr-unfav_ats_wr)*100):+.1f}pp")
    print(f"    Game winner:   fav={fav_game_wr*100:.1f}%, unfav={unfav_game_wr*100:.1f}%, "
          f"disc={((fav_game_wr-unfav_game_wr)*100):+.1f}pp")

    if ml_auc > 0.55 and ats_auc < 0.52:
        print(f"\n  → MODEL PREDICTS WINNERS ({ml_auc:.3f}) BUT NOT SPREAD ({ats_auc:.3f})")
        print(f"    = EFFICIENT MARKET: spread absorbs game-direction skill")
        print(f"    = Better modeling won't help. Need UNPRICED information.")
    elif ml_auc > 0.55 and ats_auc > 0.52:
        print(f"\n  → MODEL PREDICTS BOTH — surprising, check stability")
    elif ml_auc < 0.52:
        print(f"\n  → MODEL CAN'T EVEN PREDICT WINNERS ({ml_auc:.3f})")
        print(f"    = BAD MODEL: room for improvement with better modeling")
    else:
        print(f"\n  → WEAK on both — borderline case")


def part4_market_efficiency_test(merged, feat_cols, target_ats):
    """
    Direct market efficiency test:
    1. How much info does the closing line contain about game outcomes?
    2. After the closing line, what info remains in our features?
    3. Is there ANY residual signal for a model to exploit?
    """
    print(f"\n{'='*70}")
    print("PART 4: MARKET EFFICIENCY — How much info is in the close?")
    print(f"{'='*70}")

    valid = merged.dropna(subset=["market_home_prob_dv", "home_win"]).copy()
    valid = valid[valid["year"] >= 2024]

    if len(valid) < 200:
        print("  Insufficient data with closing MLs")
        return

    # Market calibration: how well does closing ML predict outcomes?
    mkt = valid["market_home_prob_dv"].values
    y = valid["home_win"].values.astype(int)

    mkt_auc = roc_auc_score(y, mkt)
    mkt_picks = (mkt >= 0.5).astype(int)
    mkt_wr = (mkt_picks == y).mean()

    print(f"\n  Market closing line performance (N={len(valid)}):")
    print(f"    AUC: {mkt_auc:.4f}")
    print(f"    WR:  {mkt_wr*100:.1f}%")
    print(f"    Home base rate: {y.mean()*100:.1f}%")

    # Calibration by decile
    print(f"\n  Market calibration (implied prob vs actual):")
    deciles = pd.qcut(mkt, 10, labels=False, duplicates="drop")
    print(f"  {'Decile':<8} {'Implied':<10} {'Actual':<10} {'N':<5} {'Gap'}")
    for d in sorted(set(deciles)):
        mask = deciles == d
        implied = mkt[mask].mean()
        actual = y[mask].mean()
        n = mask.sum()
        print(f"  D{d+1:<6} {implied*100:.1f}%     {actual*100:.1f}%     {n:<5} "
              f"{(actual-implied)*100:+.1f}pp")

    # Feature importance for moneyline: which features predict WINNERS?
    available = [c for c in feat_cols if c in valid.columns]
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X = valid[available].fillna(valid[available].median())
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    imp = SimpleImputer(strategy="median")
    X_i = imp.fit_transform(X_s)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    clf.fit(X_i, y)
    importances = clf.feature_importances_

    print(f"\n  Feature importance for GAME WINNER prediction:")
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(min(10, len(available))):
        idx = sorted_idx[i]
        print(f"    {i+1}. {available[idx]:<25} {importances[idx]:.4f}")

    # Now: how correlated are these features with closing ML?
    print(f"\n  Feature correlation with closing ML (|r| > 0.1 only):")
    for i in range(min(10, len(available))):
        idx = sorted_idx[i]
        feat_vals = X[:, idx] if isinstance(X, np.ndarray) else X.iloc[:, idx].values
        valid_mask = ~np.isnan(feat_vals) & ~np.isnan(mkt)
        if valid_mask.sum() < 50:
            continue
        corr = np.corrcoef(feat_vals[valid_mask], mkt[valid_mask])[0, 1]
        if abs(corr) > 0.1:
            print(f"    {available[idx]:<25} r={corr:+.3f}")


def main():
    print("=" * 70)
    print("SIDE-SELECTION DIAGNOSTIC")
    print("Can the model predict GAME WINNERS (moneyline)?")
    print("Or does the spread already price everything?")
    print("=" * 70)

    merged, feat_cols, target_ats = _load_data()
    print(f"\nLoaded {len(merged)} games")
    print(f"  With home_win: {merged['home_win'].notna().sum()}")
    print(f"  With market ML: {merged['market_home_prob_dv'].notna().sum()}")
    print(f"  Years: {sorted(merged['year'].unique())}")

    part1_moneyline_vs_spread(merged, feat_cols, target_ats)
    part2_spread_residual_analysis(merged, feat_cols, target_ats)
    part3_forward_moneyline_check(merged, feat_cols)
    part4_market_efficiency_test(merged, feat_cols, target_ats)

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
