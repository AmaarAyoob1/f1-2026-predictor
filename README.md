# F1 2026 Championship Prediction Model

**[Live Dashboard](https://amaarayoob1.github.io/f1-2026-predictor/)** · **[Kaggle Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)**

A multi-model ML pipeline that predicts the 2026 Formula 1 World Drivers' and Constructors' Championship winners using historical data (2014-2025), Elo ratings, pre-season testing signals, and bookmaker market data.

## Key Results

| Model | WDC Prediction | WCC Prediction |
|-------|---------------|----------------|
| Monte Carlo (10k sims) | Max Verstappen (37.8%) | McLaren (Norris + Piastri) |
| Bayesian Ensemble | Max Verstappen (31.8%) | Mercedes |
| Final Ensemble | **Max Verstappen (34.8%)** | **Ferrari** |

**Leave-One-Season-Out RMSE: 1.72 positions** (average across 2020-2025 holdout years)
**Training data: 270 driver-seasons** from Kaggle F1 World Championship dataset (2014-2024) + Fox Sports (2025)

## Architecture

```
f1-predictor/
├── data/
│   ├── kaggle_f1/                 # Kaggle F1 World Championship CSVs (2014-2024)
│   │   ├── results.csv            # Race-by-race finishing positions & points
│   │   ├── qualifying.csv         # Qualifying session results
│   │   ├── driver_standings.csv   # Championship standings per race
│   │   ├── constructor_standings.csv
│   │   ├── drivers.csv, constructors.csv, races.csv, status.csv, sprint_results.csv
│   ├── kaggle_loader.py           # ETL pipeline: CSV → model-ready data structures
│   └── historical_data.py         # Kaggle (2014-2024) + hardcoded (2025-2026)
├── features/
│   ├── elo_ratings.py             # Custom F1 Elo system with adaptive K-factor
│   └── feature_engineering.py     # 19 features: rolling stats, team strength, engine metrics
├── models/
│   └── train_models.py            # 4 models: GBM, Logistic, Bayesian, Monte Carlo
├── evaluation/
│   └── metrics.py                 # Brier score, calibration, bookmaker comparison
├── dashboard/
│   └── f1-2026-predictor.html     # Interactive React dashboard (Drivers, Constructors, Insights)
├── visualizations/                # Generated outputs
│   ├── evaluation_dashboard.png
│   ├── wdc_predictions_2026.csv
│   ├── wcc_predictions_2026.csv
│   └── model_vs_bookmakers.csv
├── main.py                        # End-to-end pipeline orchestrator
├── requirements.txt
└── README.md
```

## Data Pipeline

The model uses a hybrid data strategy:

- **2014-2024**: Loaded at runtime from Kaggle's F1 World Championship dataset via `kaggle_loader.py`. Four ETL functions extract driver results, constructor results, top-10/DNF rates, and qualifying stats directly from the CSV files. This gives us **247 driver-season records** with verified, race-by-race computed statistics.
- **2025**: Hardcoded from Fox Sports / FIA official standings (not yet in Kaggle).
- **2026**: Pre-season predictions — grid lineup, Bahrain testing data, bookmaker odds, expert ratings.

The Kaggle pipeline replaced ~300 lines of manually curated data with a reproducible ETL process, improving training data from 159 to **270 driver-seasons** and RMSE from 1.99 to **1.72**.

## Models

### 1. Gradient Boosting Position Predictor
Predicts championship finishing position using 19 engineered features. Uses `TimeSeriesSplit` cross-validation to respect temporal ordering.

**Top Features:** `rolling_avg_position` (74.8%), `performance_trend` (7.4%), `team_rolling_points` (4.5%)

### 2. Logistic Regression Probability Model
Calibrated classifiers for P(champion) and P(top_3). Uses class weighting to handle severe class imbalance (only 1 champion per season out of ~20 drivers).

### 3. Bayesian Ensemble
Combines ML model outputs (likelihood), pre-season testing data (prior), and bookmaker consensus odds (additional prior) via log-linear opinion pooling. This is critical for 2026 because the massive regulation changes make historical patterns less reliable — the Bayesian approach properly weights the new information from testing.

### 4. Monte Carlo Season Simulator
Simulates 10,000 full 24-race seasons with:
- Driver strength scores (composite of Elo, testing form, historical performance)
- Race-day Gaussian noise (σ=8, capturing qualifying/race variance)
- Mechanical DNF probability (team-specific reliability from testing data)
- Weather disruption (15% wet race probability with increased variance)
- Mid-season development random walk

## Feature Engineering

### Elo Rating System
Custom implementation inspired by FiveThirtyEight's approach:
- **Adaptive K-factor**: Higher for rookies (K=48), lower for veterans (K=32)
- **Pairwise season updates**: Compares each driver against all others based on championship position
- **Win/podium bonuses**: Captures excellence beyond finishing position
- **Inter-season regression**: 15% regression toward mean prevents rating divergence
- **Inactivity penalty**: -50 per missed season (affects Bottas, Pérez for 2026)

### Key Features (19 total)
| Category | Features |
|----------|----------|
| **Driver** | rolling_avg_ppr, rolling_avg_position, rolling_win_rate, rolling_podium_rate, career_seasons, career_total_wins, performance_trend |
| **Team** | team_rolling_position, team_rolling_points, team_rolling_wins, team_development_rate, reg_change_score |
| **Engine** | engine_hybrid_years, engine_titles_won, engine_maturity_score |
| **Context** | is_reg_change_year, elo_at_season |
| **Interaction** | elo_x_team_pos, trend_x_reg |

### Regulation Change Impact Score
Novel feature that quantifies how well teams historically adapt to major rule changes (2014, 2017, 2022), generating a predictive signal for 2026.

## Evaluation

### Cross-Validation Strategy
**Leave-One-Season-Out (LOSO)**: trains on all seasons before year Y, predicts year Y. This mirrors real-world usage (predicting the upcoming season from historical data).

| Holdout Year | RMSE |
|-------------|------|
| 2020 | 2.13 |
| 2021 | 1.26 |
| 2022 | 2.58 |
| 2023 | 2.26 |
| 2024 | 2.31 |
| 2025 | 1.33 |
| **Average** | **1.72** |

### Bookmaker Comparison
The model finds significant edges vs bookmaker consensus:
- **OVERWEIGHT Verstappen** (+12.8% vs market) — Elo and historical dominance
- **UNDERWEIGHT Russell** (-15.3% vs market) — market prices Mercedes PU advantage more aggressively

This divergence is valuable analysis: the market is pricing the 2026 regulation reset as a major equalizer, while the model anchors more on Verstappen's proven ability. The Insights tab in the dashboard explores this further with finishing consistency (CV) and grid-to-finish delta analysis from Kaggle's race-by-race data.

## Interesting Findings

1. **Regulation resets don't fully equalize**: Historical data shows that the top 3 teams from the previous era typically remain in the top 5 after regulation changes (with one notable exception: Mercedes leaping from P3 to P1 in 2014).

2. **Elo is the strongest single predictor** of championship position, but its predictive power drops significantly in regulation change years (the `trend_x_reg` interaction feature captures this).

3. **Engine maturity matters enormously in year 1 of new regs**: Mercedes' PU advantage in 2014 was decisive. The 2026 PU regulations (50/50 ICE/electric split, no MGU-H) create a similar opportunity — and paddock consensus points to Mercedes again.

4. **The Bayesian ensemble outperforms pure ML** for regulation change years by incorporating forward-looking signals (testing, expert assessments) rather than relying solely on backward-looking historical patterns.

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
```

Optional (for enhanced models):
```
xgboost>=2.0          # Replace sklearn GBM for better performance
lightgbm>=4.0         # Alternative gradient boosting
pymc>=5.0             # Full Bayesian hierarchical model
plotly>=5.0           # Interactive visualizations
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Future Improvements

- [x] **Kaggle data pipeline**: ETL module loading 2014-2024 data from Kaggle F1 World Championship dataset (replaced hardcoded data, +70% training records)
- [x] **Race-level insights**: Finishing consistency (CV) and grid-to-finish delta analysis from Kaggle race-by-race data
- [ ] **XGBoost/LightGBM**: Replace sklearn GBM with optimized implementations + Optuna hyperparameter tuning
- [ ] **Bayesian hierarchical model** (PyMC): Model driver ability and team strength as latent variables with proper uncertainty quantification
- [ ] **Race-level predictions**: Predict individual race results and aggregate to season (much richer training signal)
- [ ] **Calibration curves**: Plot reliability diagrams for probabilistic outputs
- [ ] **Live updating**: Stream results during the 2026 season and update predictions in real-time
- [ ] **Streamlit dashboard**: Interactive web app for exploring predictions

## Author

Built as a portfolio project for ML Engineering and Quantitative Finance internship applications.

**Skills demonstrated:** Feature engineering, ETL pipeline design, time-series CV, probabilistic modeling, Bayesian inference, Monte Carlo simulation, model evaluation, data verification, Python software engineering.
