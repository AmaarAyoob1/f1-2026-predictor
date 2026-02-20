# 🏎️ F1 2026 Championship Prediction Model

A multi-model machine learning pipeline that predicts the 2026 Formula 1 World Drivers' and Constructors' Championships using ensemble methods combining Gradient Boosting, Bayesian probability estimation, and Monte Carlo season simulation.

## 🏆 Predictions

| Pos | Driver | Team | Win Prob | Avg Points |
|-----|--------|------|----------|------------|
| 1 | Max Verstappen | Red Bull | 32.8% | 320 |
| 2 | Lando Norris | McLaren | 21.6% | 288 |
| 3 | George Russell | Mercedes | 14.4% | 297 |
| 4 | Charles Leclerc | Ferrari | 11.6% | 293 |
| 5 | Oscar Piastri | McLaren | 11.1% | 228 |
| 6 | Lewis Hamilton | Ferrari | 4.3% | 255 |

**Constructors:** Ferrari (548 pts) > McLaren (516) > Mercedes (482) > Red Bull (395)

## 🧠 Model Architecture

### Three-Model Ensemble
1. **Gradient Boosting Regressor** — Predicts championship position from 19 features (LOO RMSE: 1.99)
2. **Bayesian Probability Model** — Calibrates bookmaker odds with historical Elo priors
3. **Monte Carlo Simulator** — 10,000 season simulations with realistic race chaos

### Composite Driver Strength Formula (8 Components)
- **Historical performance** (40% regulation discount for 2026's radical rule change)
- **Car/Engine quality** (preseason testing, engine maturity, works team bonus)
- **Podium rate** (sustained excellence metric)
- **F1 Experience** (actual career years with √diminishing returns)
- **Regulation veteran bonus** (2.5 pts per reg change survived: 2014/2017/2022)
- **Age-prime curve** (peak at 25-31, gradual decline after 35, sharp drop after 40)
- **Top-10 consistency bonus** (points-scoring reliability)
- **DNF rate penalty** (reliability and crash-proneness)

### Monte Carlo Realism Features
- Strength compression (30-100 scale, ensures backmarkers can score in chaos)
- Per-season form variance (σ=5, allows teammate flips)
- Safety car modeling (40% probability, ±6 strategy luck)
- Wet race compression (15% probability, reduces top-team advantage 30%)
- First-lap incidents (20% probability, 12% driver involvement)
- Mid-season development convergence (factor 0.10)

## 📊 Custom Elo Rating System
Tracks driver skill evolution across 2014-2025 (K=32). Current ratings:
- Verstappen: 1747 | Hamilton: 1653 | Norris: 1593 | Leclerc: 1577 | Piastri: 1573

## 📁 Project Structure

```
f1-predictor/
├── main.py                          # Pipeline entry point + strength formula + MC simulator
├── data/
│   ├── historical_data.py           # 159 driver-seasons (2014-2025) + top-10/DNF rates
│   └── qualifying_data.py           # Qualifying performance data
├── features/
│   ├── elo_ratings.py               # Custom Elo rating system
│   └── feature_engineering.py       # Rolling features + 2026 feature builder
├── models/
│   └── train_models.py              # GBM + Bayesian + Monte Carlo
├── dashboard/
│   └── f1-2026-predictor.html       # Interactive React dashboard
└── docs/
    └── F1_2026_Technical_Doc.docx   # Full technical documentation
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/AmaarAyoob1/f1-2026-predictor.git
cd f1-2026-predictor

# Install dependencies
pip install pandas numpy scikit-learn matplotlib

# Run predictions
python main.py

# View dashboard
open dashboard/f1-2026-predictor.html
```

## 🔧 Technical Details

- **Training data:** 159 driver-season records (2014-2025 hybrid era)
- **Validation:** Leave-One-Out CV, RMSE = 1.99
- **2025 data:** Verified against official FIA standings (Fox Sports, Wikipedia)
- **MC simulations:** 10,000 complete 24-race seasons
- **Ensemble:** 50% MC win probability + 50% Bayesian win probability

## 📈 Key Design Decisions

**Why 2014+ data only?** The 2014 turbo-hybrid regulation change is the closest analog to 2026's power unit revolution. Pre-2014 data (V8 era) has limited predictive value for post-regulation performance.

**Why 40% regulation discount?** Historical analysis shows that in 2014's comparable PU reset, the pre-regulation champion (Vettel/Red Bull) dropped from P1 to P5. Engine advantage dominated year-1 outcomes.

**Why age-prime curve?** F1 drivers peak at 25-31 (reaction time, G-force tolerance, neck stamina). Schumacher's 2010-12 return (age 41-43) showed ~0.3s/lap decline. Alonso at 44 is the oldest driver since the 2000s.

## 🏁 Future Work
- **Live update pipeline:** Bayesian blending of pre-season estimates with actual 2026 race results
- **Teammate head-to-head:** Transitive comparison networks for relative driver ability
- **Circuit-type clustering:** Simulating different race types (street, high-downforce, power-sensitive)

## 📄 License
MIT License — see LICENSE file
