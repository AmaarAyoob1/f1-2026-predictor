#!/usr/bin/env python3
"""
F1 2026 Championship Prediction Pipeline
==========================================
End-to-end ML pipeline that predicts the 2026 F1 World Drivers'
and Constructors' Championship winners.

Pipeline stages:
1. Load & validate historical data (2014-2025)
2. Build Elo rating system
3. Engineer features (rolling stats, team strength, engine metrics)
4. Train models (GBM position predictor, logistic probability, Bayesian ensemble)
5. Run Monte Carlo season simulation (10,000 iterations)
6. Evaluate models (Brier score, calibration, bookmaker comparison)
7. Generate 2026 predictions

Author: Ayoob — Claremont Graduate University
Target: Quant Analyst / ML Engineering portfolio project
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data.historical_data import (
    get_driver_df, get_constructor_df,
    GRID_2026, PRESEASON_TESTING_2026,
    BOOKMAKER_ODDS_DRIVERS_2026, BOOKMAKER_ODDS_CONSTRUCTORS_2026
)
from features.elo_ratings import build_elo_ratings
from features.feature_engineering import (
    build_feature_matrix, build_2026_features
)
from models.train_models import (
    PositionPredictor, ChampionshipProbPredictor,
    BayesianEnsemble, MonteCarloSimulator,
    predict_constructors
)
from evaluation.metrics import (
    compute_brier_score, compute_brier_skill_score,
    evaluate_position_model, compare_with_bookmakers,
    generate_evaluation_report
)


def print_header(text, char="═"):
    width = 65
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_section(text):
    print(f"\n── {text} {'─' * (60 - len(text))}")


def main():
    print_header("F1 2026 CHAMPIONSHIP PREDICTION PIPELINE")
    print("  Models: GBM · Logistic · Bayesian Ensemble · Monte Carlo")
    print("  Data: 2014-2025 hybrid era · Pre-season testing · Bookmaker odds")
    print(f"  Drivers on 2026 grid: {len(GRID_2026)}")
    
    # ═══════════════════════════════════════════════════
    # STAGE 1: LOAD DATA
    # ═══════════════════════════════════════════════════
    print_section("Stage 1: Loading Historical Data")
    
    driver_df = get_driver_df()
    constructor_df = get_constructor_df()
    
    years = sorted(driver_df["year"].unique())
    print(f"  Seasons loaded: {years[0]}-{years[-1]} ({len(years)} seasons)")
    print(f"  Driver-season records: {len(driver_df)}")
    print(f"  Constructor-season records: {len(constructor_df)}")
    print(f"  Unique drivers: {driver_df['driver'].nunique()}")
    
    # ═══════════════════════════════════════════════════
    # STAGE 2: BUILD ELO RATINGS
    # ═══════════════════════════════════════════════════
    print_section("Stage 2: Building Elo Rating System")
    
    elo = build_elo_ratings(driver_df)
    elo_df = elo.get_ratings_df()
    
    print("\n  Current Elo Ratings (Top 10):")
    print("  " + "─" * 45)
    for _, row in elo_df.head(10).iterrows():
        bar = "█" * int(row["elo_rating"] / 50)
        print(f"  {row['driver']:<22} {row['elo_rating']:>7.1f}  {bar}")
    
    # ═══════════════════════════════════════════════════
    # STAGE 3: FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════
    print_section("Stage 3: Feature Engineering")
    
    X_train, y_train, train_data = build_feature_matrix(driver_df, constructor_df, elo)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Feature list: {list(X_train.columns)}")
    
    # Build 2026 features
    features_2026 = build_2026_features(
        driver_df, constructor_df, elo,
        GRID_2026, PRESEASON_TESTING_2026,
        BOOKMAKER_ODDS_DRIVERS_2026
    )
    print(f"\n  2026 prediction features built for {len(features_2026)} drivers")
    
    # ═══════════════════════════════════════════════════
    # STAGE 4: TRAIN MODELS
    # ═══════════════════════════════════════════════════
    print_section("Stage 4: Training Models")
    
    # Model 1: GBM Position Predictor
    print("\n  [1/4] Gradient Boosting Position Predictor...")
    pos_model = PositionPredictor(n_estimators=200, max_depth=4, learning_rate=0.05)
    pos_model.fit(X_train, y_train)
    metrics = pos_model.get_metrics()
    print(f"        CV RMSE: {metrics['cv_rmse_mean']:.2f} ± {metrics['cv_rmse_std']:.2f}")
    
    # Model 2: Logistic Championship Probability
    print("\n  [2/4] Logistic Regression Probability Model...")
    prob_model = ChampionshipProbPredictor(C=0.5)
    prob_model.fit(X_train, y_train)
    print("        Champion & Top-3 classifiers fitted")
    
    # Model 3: Bayesian Ensemble
    print("\n  [3/4] Bayesian Ensemble...")
    bayesian = BayesianEnsemble(ml_weight=0.4, testing_weight=0.3, bookmaker_weight=0.3)
    print("        Initialized (ML=0.4, Testing=0.3, Bookmaker=0.3)")
    
    # Model 4: Monte Carlo Simulator
    print("\n  [4/4] Monte Carlo Season Simulator (10,000 iterations)...")
    mc = MonteCarloSimulator(n_simulations=10000, random_state=42)
    print("        Ready")
    
    # ═══════════════════════════════════════════════════
    # STAGE 5: GENERATE 2026 PREDICTIONS
    # ═══════════════════════════════════════════════════
    print_section("Stage 5: 2026 Predictions")
    
    # GBM position predictions
    feature_cols = pos_model.FEATURE_COLS
    X_2026 = features_2026[feature_cols].fillna(0)
    gbm_positions = pos_model.predict(X_2026)
    features_2026["gbm_predicted_position"] = gbm_positions
    
    # Logistic probability predictions
    prob_preds = prob_model.predict_proba(features_2026)
    features_2026["p_champion_logistic"] = prob_preds["p_champion"]
    features_2026["p_top3_logistic"] = prob_preds["p_top3"]
    
    # Monte Carlo simulation
    # Compute driver strengths from composite features
    driver_strengths = {}
    driver_teams = {}
    team_reliability = {}
    
    for _, row in features_2026.iterrows():
        # ── Composite strength for 2026 ──
        # KEY INSIGHT: 2026 is a massive regulation change (new PU, active aero,
        # 50/50 ICE/electric split). Historical analysis of 2014/2017/2022 reg
        # resets shows that engine advantage and team adaptation matter MORE
        # than historical driver Elo in year 1 of new regs.
        #
        # We apply a "regulation discount" that reduces the weight of backward-
        # looking features and amplifies forward-looking signals (testing, engine).
        
        REG_DISCOUNT = 0.40  # reduce historical weight by 60% in reg-change years
        # Justification: In 2014 (last comparable PU reg reset), the pre-reg
        # champion (Vettel/Red Bull) dropped from P1 to P5. Engine advantage
        # dominated. We apply aggressive discounting to historical features.
        
        # Backward-looking components (discounted for reg change)
        historical = (
            row["elo_at_season"] / 20 +            # driver Elo
            (22 - row["rolling_avg_position"]) * 1.5 +  # recent championship positions
            row["rolling_win_rate"] * 25             # win rate
        ) * REG_DISCOUNT
        
        # Forward-looking components (amplified for reg change)
        # Engine maturity was THE differentiator in 2014's reg reset
        forward = (
            row.get("preseason_expert_rating", 50) * 0.65 +  # testing form (strongest 2026 signal)
            row["engine_maturity_score"] * 0.55 +             # engine PU quality
            row.get("preseason_reliability", 70) * 0.12 +     # reliability from testing
            row["reg_change_score"] * 0.18                    # how well team adapts to regs
        )
        
        # Pure driver talent component (partially reg-independent)
        # Great drivers adapt faster — but the gap compresses in year 1
        driver_talent = (
            row["rolling_podium_rate"] * 15 +
            min(row["career_seasons"], 10) * 0.5  # experience helps adaptation
        )
        
        strength = historical + forward + driver_talent
        driver_strengths[row["driver"]] = strength
        driver_teams[row["driver"]] = row["team"]
    
    for team, data in PRESEASON_TESTING_2026.items():
        team_reliability[team] = data["reliability_score"] / 100
    
    mc_results = mc.simulate_season(driver_strengths, team_reliability, driver_teams)
    
    # Bayesian ensemble
    ml_probs = features_2026["p_champion_logistic"].values
    testing_scores = features_2026["preseason_expert_rating"].values
    bookmaker_probs = features_2026["bookmaker_prob"].values
    
    bayesian_results = bayesian.compute_posterior(
        ml_probs, testing_scores, bookmaker_probs,
        features_2026["driver"].tolist()
    )
    
    # ═══════════════════════════════════════════════════
    # RESULTS: WORLD DRIVERS' CHAMPIONSHIP
    # ═══════════════════════════════════════════════════
    print_header("🏆 2026 WORLD DRIVERS' CHAMPIONSHIP PREDICTIONS")
    
    # Combine all model outputs
    final_wdc = mc_results[["driver", "team", "p_champion", "p_top3", "mean_points", "mean_wins"]].copy()
    final_wdc = final_wdc.merge(
        bayesian_results[["driver", "posterior_prob"]].rename(columns={"posterior_prob": "bayesian_prob"}),
        on="driver", how="left"
    )
    final_wdc = final_wdc.merge(
        features_2026[["driver", "elo_at_season", "gbm_predicted_position"]],
        on="driver", how="left"
    )
    
    # Ensemble score: weighted average of MC and Bayesian
    final_wdc["ensemble_prob"] = (
        0.5 * final_wdc["p_champion"] +
        0.5 * final_wdc["bayesian_prob"]
    )
    final_wdc = final_wdc.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)
    
    print(f"\n  {'Pos':<4} {'Driver':<22} {'Team':<14} {'Elo':>6} {'MC Win%':>8} {'Bayes%':>8} {'Ensemble':>9} {'Avg Pts':>8}")
    print("  " + "─" * 85)
    
    for i, row in final_wdc.iterrows():
        pos = i + 1
        marker = " ◄" if pos <= 3 else ""
        print(f"  {pos:<4} {row['driver']:<22} {row['team']:<14} "
              f"{row['elo_at_season']:>6.0f} {row['p_champion']*100:>7.1f}% "
              f"{row['bayesian_prob']*100:>7.1f}% {row['ensemble_prob']*100:>8.1f}% "
              f"{row['mean_points']:>7.0f}{marker}")
    
    # ═══════════════════════════════════════════════════
    # RESULTS: WORLD CONSTRUCTORS' CHAMPIONSHIP
    # ═══════════════════════════════════════════════════
    print_header("🏗  2026 WORLD CONSTRUCTORS' CHAMPIONSHIP PREDICTIONS")
    
    constructor_results = predict_constructors(mc_results)
    
    print(f"\n  {'Pos':<4} {'Constructor':<16} {'Drivers':<35} {'Total Pts':>10} {'Best Driver Win%':>18}")
    print("  " + "─" * 85)
    
    for i, row in constructor_results.iterrows():
        pos = i + 1
        drivers_str = " & ".join([d.split()[-1] for d in row["drivers"]])
        marker = " ◄" if pos <= 3 else ""
        print(f"  {pos:<4} {row['team']:<16} {drivers_str:<35} "
              f"{row['total_mean_points']:>9.0f} {row['best_driver_p_champion']*100:>16.1f}%{marker}")
    
    # ═══════════════════════════════════════════════════
    # RESULTS: MODEL VS BOOKMAKERS
    # ═══════════════════════════════════════════════════
    print_header("📊 MODEL vs BOOKMAKER COMPARISON")
    
    model_probs_dict = dict(zip(final_wdc["driver"], final_wdc["ensemble_prob"]))
    comparison = compare_with_bookmakers(model_probs_dict, BOOKMAKER_ODDS_DRIVERS_2026)
    
    edges = comparison[comparison["direction"] != "NEUTRAL"].head(10)
    print(f"\n  {'Driver':<22} {'Model':>8} {'Books':>8} {'Edge':>8} {'Signal':<12}")
    print("  " + "─" * 60)
    for _, row in edges.iterrows():
        edge_color = "↑" if row["edge"] > 0 else "↓"
        print(f"  {row['driver']:<22} {row['model_prob']*100:>7.1f}% {row['bookmaker_prob']*100:>7.1f}% "
              f"{row['edge']*100:>+7.1f}% {edge_color} {row['direction']}")
    
    # ═══════════════════════════════════════════════════
    # STAGE 6: MODEL EVALUATION
    # ═══════════════════════════════════════════════════
    print_section("Stage 6: Model Evaluation")
    
    # Evaluate on training data (in-sample — note: proper evaluation would use holdout)
    eval_metrics = evaluate_position_model(y_train.values, pos_model.predict(X_train))
    print(f"\n  Position Model (In-Sample):")
    print(f"    RMSE:             {eval_metrics['rmse']}")
    print(f"    MAE:              {eval_metrics['mae']}")
    print(f"    Top-3 Accuracy:   {eval_metrics['top3_accuracy']}")
    print(f"    Champion Correct: {eval_metrics['champion_correct']}")
    
    # Leave-one-season-out evaluation
    print(f"\n  Leave-One-Season-Out Cross-Validation:")
    loo_results = []
    for test_year in range(2020, 2026):
        train_mask = train_data["year"] < test_year
        test_mask = train_data["year"] == test_year
        
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        
        X_tr = X_train[train_mask.values]
        y_tr = y_train[train_mask.values]
        X_te = X_train[test_mask.values]
        y_te = y_train[test_mask.values]
        
        temp_model = PositionPredictor(n_estimators=150, max_depth=3, learning_rate=0.05)
        temp_model.fit(X_tr, y_tr)
        y_pred = temp_model.predict(X_te)
        
        rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
        loo_results.append({"year": test_year, "rmse": rmse, "n_drivers": len(y_te)})
        print(f"    {test_year}: RMSE = {rmse:.2f} ({len(y_te)} drivers)")
    
    avg_loo_rmse = np.mean([r["rmse"] for r in loo_results])
    print(f"    Average LOO RMSE: {avg_loo_rmse:.2f}")
    
    # Feature importance
    print("\n  Top 10 Feature Importances:")
    for _, row in pos_model.feature_importances.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")
    
    # ═══════════════════════════════════════════════════
    # GENERATE VISUALIZATION
    # ═══════════════════════════════════════════════════
    print_section("Stage 7: Generating Visualizations")
    
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    report_path = generate_evaluation_report(
        pos_model, prob_model,
        X_train, y_train,
        mc_results,
        BOOKMAKER_ODDS_DRIVERS_2026,
        output_path="visualizations/evaluation_dashboard.png"
    )
    print(f"  Evaluation dashboard saved: {report_path}")
    
    # Save predictions to CSV
    final_wdc.to_csv("visualizations/wdc_predictions_2026.csv", index=False)
    constructor_results.to_csv("visualizations/wcc_predictions_2026.csv", index=False)
    elo_df.to_csv("visualizations/elo_ratings.csv", index=False)
    comparison.to_csv("visualizations/model_vs_bookmakers.csv", index=False)
    print("  Prediction CSVs saved to visualizations/")
    
    # ═══════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════
    print_header("FINAL PREDICTIONS SUMMARY", "★")
    
    wdc_winner = final_wdc.iloc[0]
    wcc_winner = constructor_results.iloc[0]
    
    print(f"""
  🏆 Predicted 2026 World Drivers' Champion:
     {wdc_winner['driver']} ({wdc_winner['team']})
     Win probability: {wdc_winner['ensemble_prob']*100:.1f}%
     Elo rating: {wdc_winner['elo_at_season']:.0f}
     Expected season points: {wdc_winner['mean_points']:.0f}

  🏗  Predicted 2026 World Constructors' Champion:
     {wcc_winner['team']}
     Drivers: {' & '.join(wcc_winner['drivers'])}
     Expected season points: {wcc_winner['total_mean_points']:.0f}

  📊 Model Confidence:
     GBM Position RMSE: {metrics['cv_rmse_mean']:.2f}
     Leave-One-Out RMSE: {avg_loo_rmse:.2f}
     Monte Carlo sims: 10,000
     Bayesian ensemble: ML(40%) + Testing(30%) + Bookmaker(30%)
""")
    
    print_header("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
