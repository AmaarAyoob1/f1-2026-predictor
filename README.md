# 🏎️ F1 2026 Championship Prediction Model

A multi-model machine learning pipeline that predicts the 2026 Formula 1 World Drivers' and Constructors' Championships using Gradient Boosting, Bayesian Ensemble, Elo Ratings, and Monte Carlo simulation.

**LOO Cross-Validation RMSE: 1.98 positions** · **10,000 simulated seasons** · **19 engineered features** · **12 seasons of training data (2014–2025)**

---

## 📊 Key Predictions

| # | Driver | Team | Win Probability | Expected Points |
|---|--------|------|:-:|:-:|
| 1 | Max Verstappen | Red Bull | **30.0%** | 334 |
| 2 | George Russell | Mercedes | **21.5%** | 338 |
| 3 | Lando Norris | McLaren | **17.0%** | 305 |
| 4 | Oscar Piastri | McLaren | **9.7%** | 236 |
| 5 | Charles Leclerc | Ferrari | **9.1%** | 318 |
| 6 | Lewis Hamilton | Ferrari | **8.2%** | 319 |

| # | Constructor | Title Probability | Expected Points |
|---|-------------|:-:|:-:|
| 1 | Ferrari | **32.5%** | 637 |
| 2 | Mercedes | **28.0%** | 601 |
| 3 | McLaren | **22.0%** | 541 |
| 4 | Red Bull | **15.5%** | 408 |

## 🧠 Model Architecture

```
Historical Data (2014-2025)
        │
        ├──→ Elo Rating System (custom F1-adapted)
        │         │
        ├──→ Feature Engineering (19 features)
        │         │
        │    ┌────┴────────────────┐
        │    │                     │
        │    ▼                     ▼
        │  GBM Position      Logistic Regression
        │  Predictor          (P(champion), P(top3))
        │    │                     │
        │    └──────┬──────────────┘
        │           │
        │           ▼
        │    Bayesian Ensemble ◄── Pre-season Testing
        │    (Log-linear pooling)◄── Bookmaker Odds
        │           │
        │           ▼
        └──→ Monte Carlo Simulator (10,000 seasons)
                    │
                    ▼
            Championship Probabilities
```

### What Makes 2026 Special

The 2026 season introduces the most significant regulation changes since 2014: new power unit rules (50/50 ICE/electric split), active aerodynamics, and an 11th team (Cadillac). This creates a unique modeling challenge — historical performance data becomes less predictive when the cars fundamentally change.

**Two critical innovations in this model:**

1. **Regulation Discount (Section 7):** Historical features are reduced by 60% in regulation-change years, based on the 2014 and 2022 precedents where dominant teams fell dramatically.

2. **Works Team Bonus (Section 8):** Factory teams (who build their own engine) receive a +15 engine maturity bonus over customer teams. In 2014, works Mercedes won 16/19 races while customer Williams won 0 despite using the same engine. This single feature moved George Russell from 4.4% to 21.5% championship probability.

## 🏗️ Project Structure

```
f1-predictor/
├── data/
│   ├── historical_data.py         # 2014-2025 seasons, 2026 grid, testing, odds
│   └── qualifying_data.py         # Qualifying stats (explored, excluded — see docs)
├── features/
│   ├── elo_ratings.py             # Custom F1 Elo system with adaptive K-factors
│   └── feature_engineering.py     # 19 features + works team bonus
├── models/
│   └── train_models.py            # GBM, Logistic, Bayesian, Monte Carlo
├── evaluation/
│   └── metrics.py                 # Brier score, calibration, LOO-CV
├── visualizations/                # Generated CSVs and plots
├── docs/                          # Technical documentation (.docx)
├── dashboard/                     # Interactive React dashboard (.html)
├── main.py                        # End-to-end pipeline orchestrator
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/f1-2026-predictor.git
cd f1-2026-predictor

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

The pipeline runs end-to-end in ~30 seconds and outputs:
- Championship probability tables (WDC + WCC)
- Model vs bookmaker comparison
- LOO cross-validation results
- Feature importance rankings
- Evaluation dashboard (PNG)
- Prediction CSVs

## 📈 Model Evaluation

**Leave-One-Season-Out Cross-Validation:**

| Holdout Year | RMSE | Context |
|:---:|:---:|---|
| 2020 | 2.13 | COVID-shortened season |
| 2021 | 1.26 | Verstappen vs Hamilton title fight |
| 2022 | 2.58 | Regulation change year |
| 2023 | 2.26 | Verstappen dominance |
| 2024 | 2.31 | McLaren resurgence |
| 2025 | 1.33 | Norris WDC |
| **Average** | **1.98** | |

**Top 5 Feature Importances (GBM):**

| Feature | Importance | What It Captures |
|---|:-:|---|
| `rolling_avg_ppr` | 0.4412 | Recent points-per-race consistency |
| `rolling_avg_position` | 0.3176 | Championship finishing trend |
| `performance_trend` | 0.0941 | Improving vs declining trajectory |
| `elo_x_team_pos` | 0.0580 | Elite driver × top team interaction |
| `rolling_podium_rate` | 0.0229 | Sustained excellence signal |

## 🔬 Feature Ablation: Qualifying Data

We investigated adding qualifying performance features (`avg_quali_pos`, `q3_rate`, `front_row_rate`) and found they **worsened** the model:

| Metric | Without Qualifying | With Qualifying |
|---|:-:|:-:|
| LOO RMSE | **1.98** | 2.08 |
| Verstappen edge vs books | +8.0% | +14.0% |
| Russell edge vs books | -6.6% | -8.3% |

**Root cause:** Multicollinearity with existing features (qualifying correlates r>0.85 with points-per-race) and ~70% car-dependency making qualifying unreliable in regulation-change years. See the technical documentation for the full ablation study.

## 📄 Documentation

A comprehensive 14-section [technical document](docs/F1_2026_Prediction_Model_Technical_Documentation.docx) covers the complete theory, math, and code — including Elo derivations, Bayesian log-linear pooling equations, Monte Carlo race simulation mechanics, regulation discount justification, works team bonus evidence, and the qualifying ablation study.

## ⚡ Interactive Dashboard

An [interactive React dashboard](dashboard/f1-2026-predictor.html) lets you explore all predictions — toggle between Ensemble/Monte Carlo/Bayesian, compare model vs bookmaker odds, and tap any driver for detailed breakdowns. Open directly in any browser, no build tools needed.

## 🛠️ Tech Stack

- **Python 3.12** — Core pipeline
- **scikit-learn** — GBM, Logistic Regression, TimeSeriesSplit, StandardScaler
- **NumPy / SciPy** — Monte Carlo simulation, Bayesian computation
- **Pandas** — Data engineering
- **Matplotlib** — Evaluation visualizations
- **React 18** — Interactive dashboard (via CDN, no build step)

## 📝 License

MIT

## 👤 Author

**Ayoob** — MS Statistics & Machine Learning · MS Financial Engineering · Claremont Graduate University

Built as a portfolio project demonstrating applied ML, sports analytics, Bayesian methods, and Monte Carlo simulation for quantitative analysis roles.
