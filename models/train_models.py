"""
F1 Prediction Models
====================
Multiple modeling approaches for championship prediction:

1. GradientBoosting Regressor — predicts championship position
2. Logistic Regression — predicts P(top_3) and P(champion)
3. Bayesian Ensemble — combines model outputs with prior beliefs
4. Monte Carlo Simulation — race-by-race stochastic simulation

Model selection rationale:
- GBM captures nonlinear interactions (team × driver × engine)
- Logistic Regression provides calibrated probabilities
- Bayesian approach incorporates pre-season testing as priors
- Monte Carlo adds realistic race-day variance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, log_loss, brier_score_loss
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════
# MODEL 1: GRADIENT BOOSTING POSITION PREDICTOR
# ═══════════════════════════════════════════════════

class PositionPredictor:
    """
    Gradient Boosting Regressor to predict championship finishing position.
    Uses TimeSeriesSplit to respect temporal ordering.
    """
    
    FEATURE_COLS = [
        "rolling_avg_ppr", "rolling_avg_position", "rolling_win_rate",
        "rolling_podium_rate", "career_seasons", "career_total_wins",
        "performance_trend", "team_rolling_position", "team_rolling_points",
        "team_rolling_wins", "team_development_rate", "reg_change_score",
        "elo_at_season", "engine_hybrid_years", "engine_titles_won",
        "engine_maturity_score", "is_reg_change_year",
        "elo_x_team_pos", "trend_x_reg",
    ]
    
    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.cv_scores = None
        self.feature_importances = None
    
    def fit(self, X, y):
        """Fit the model with time-series cross-validation."""
        X_scaled = self.scaler.fit_transform(X[self.FEATURE_COLS])
        
        # Time-series CV (respects temporal ordering)
        tscv = TimeSeriesSplit(n_splits=3)
        self.cv_scores = -cross_val_score(
            self.model, X_scaled, y,
            cv=tscv, scoring="neg_mean_squared_error"
        )
        
        # Fit on full data
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Feature importances
        self.feature_importances = pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return self
    
    def predict(self, X):
        """Predict championship positions."""
        X_input = X[self.FEATURE_COLS] if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_input)
        return self.model.predict(X_scaled)
    
    def get_metrics(self):
        """Return cross-validation metrics."""
        return {
            "cv_rmse_mean": np.sqrt(np.mean(self.cv_scores)),
            "cv_rmse_std": np.sqrt(np.std(self.cv_scores)),
        }


# ═══════════════════════════════════════════════════
# MODEL 2: CHAMPIONSHIP PROBABILITY (LOGISTIC)
# ═══════════════════════════════════════════════════

class ChampionshipProbPredictor:
    """
    Logistic Regression to predict:
    - P(champion) — probability of winning the title
    - P(top_3)   — probability of finishing in top 3
    
    Outputs calibrated probabilities suitable for Brier score evaluation.
    """
    
    FEATURE_COLS = [
        "rolling_avg_ppr", "rolling_avg_position", "rolling_win_rate",
        "rolling_podium_rate", "career_seasons", "career_total_wins",
        "performance_trend", "team_rolling_position", "team_rolling_points",
        "team_rolling_wins", "team_development_rate", "reg_change_score",
        "elo_at_season", "engine_hybrid_years", "engine_titles_won",
        "engine_maturity_score", "is_reg_change_year",
        "elo_x_team_pos", "trend_x_reg",
    ]
    
    def __init__(self, C=1.0):
        self.champion_model = LogisticRegression(
            C=C, max_iter=1000, random_state=42, class_weight="balanced"
        )
        self.top3_model = LogisticRegression(
            C=C, max_iter=1000, random_state=42, class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y_position):
        """
        Fit both champion and top-3 models.
        y_position should be championship finishing position (1, 2, 3, ...).
        """
        X_scaled = self.scaler.fit_transform(X[self.FEATURE_COLS])
        
        y_champion = (y_position == 1).astype(int)
        y_top3 = (y_position <= 3).astype(int)
        
        self.champion_model.fit(X_scaled, y_champion)
        self.top3_model.fit(X_scaled, y_top3)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities for championship and top-3.
        
        Returns DataFrame with columns:
        - p_champion: probability of winning title
        - p_top3: probability of top-3 finish
        """
        X_input = X[self.FEATURE_COLS] if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_input)
        
        p_champ = self.champion_model.predict_proba(X_scaled)[:, 1]
        p_top3 = self.top3_model.predict_proba(X_scaled)[:, 1]
        
        return pd.DataFrame({
            "p_champion": p_champ,
            "p_top3": p_top3
        })


# ═══════════════════════════════════════════════════
# MODEL 3: BAYESIAN ENSEMBLE
# ═══════════════════════════════════════════════════

class BayesianEnsemble:
    """
    Bayesian ensemble that combines:
    - ML model outputs (likelihood)
    - Pre-season testing data (prior)
    - Bookmaker odds (additional prior)
    
    Uses Bayesian updating to produce posterior championship probabilities.
    This is especially valuable for 2026 where historical patterns
    are disrupted by regulation changes.
    """
    
    def __init__(self, ml_weight=0.4, testing_weight=0.3, bookmaker_weight=0.3):
        self.ml_weight = ml_weight
        self.testing_weight = testing_weight
        self.bookmaker_weight = bookmaker_weight
    
    def compute_posterior(
        self,
        ml_probs: np.ndarray,
        testing_scores: np.ndarray,
        bookmaker_probs: np.ndarray,
        drivers: list
    ) -> pd.DataFrame:
        """
        Compute posterior championship probabilities via weighted Bayesian update.
        
        Parameters
        ----------
        ml_probs : array-like
            P(champion) from ML model for each driver
        testing_scores : array-like
            Pre-season testing expert ratings (0-100) per driver's team
        bookmaker_probs : array-like
            Bookmaker implied probabilities per driver
            
        Returns
        -------
        DataFrame with posterior probabilities, ranked by likelihood
        """
        # Normalize testing scores to probabilities
        testing_probs = testing_scores / testing_scores.sum()
        
        # Ensure bookmaker probs sum to 1
        bookmaker_probs = np.array(bookmaker_probs)
        bookmaker_probs = bookmaker_probs / bookmaker_probs.sum()
        
        # Ensure ML probs are valid
        ml_probs = np.array(ml_probs)
        ml_probs = np.maximum(ml_probs, 1e-6)
        ml_probs = ml_probs / ml_probs.sum()
        
        # Bayesian update: posterior ∝ prior × likelihood
        # Using log-linear pooling (standard for combining probability forecasts)
        log_posterior = (
            self.ml_weight * np.log(ml_probs + 1e-10) +
            self.testing_weight * np.log(testing_probs + 1e-10) +
            self.bookmaker_weight * np.log(bookmaker_probs + 1e-10)
        )
        
        # Softmax to normalize
        posterior = np.exp(log_posterior - np.max(log_posterior))
        posterior = posterior / posterior.sum()
        
        results = pd.DataFrame({
            "driver": drivers,
            "ml_prob": ml_probs,
            "testing_prob": testing_probs,
            "bookmaker_prob": bookmaker_probs,
            "posterior_prob": posterior
        }).sort_values("posterior_prob", ascending=False).reset_index(drop=True)
        
        return results


# ═══════════════════════════════════════════════════
# MODEL 4: MONTE CARLO RACE SIMULATION
# ═══════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Full-season Monte Carlo simulation.
    
    Simulates each race individually with:
    - Normalized driver strength (compressed to 0-100 scale)
    - Per-season driver form variance (allows teammate flips)
    - Race-day variance (mechanical failures, weather, incidents)
    - Safety car / red flag chaos (occasionally shuffles entire field)
    - Team development trajectory (mid-season upgrades)
    - First-lap incident risk
    
    Produces championship probability distributions.
    """
    
    POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    N_RACES_2026 = 24
    
    def __init__(self, n_simulations=10000, random_state=42):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_state)
    
    def _normalize_strengths(self, driver_strengths: dict) -> dict:
        """
        Compress raw strengths to 0-100 scale.
        
        This is critical: the raw strength gap (~80 points) is too wide
        relative to per-race noise, meaning lower-tier drivers can literally
        never score points. In real F1, even backmarkers occasionally score
        via safety cars, rain, or DNFs ahead. Compression to 0-100 with a
        floor of 30 ensures a realistic spread.
        """
        vals = np.array(list(driver_strengths.values()))
        mn, mx = vals.min(), vals.max()
        
        # Normalize to 30-100 range (floor of 30 ensures even the weakest
        # driver has *some* chance of scoring in a chaotic race)
        normalized = {}
        for driver, raw in driver_strengths.items():
            normalized[driver] = 30 + 70 * (raw - mn) / (mx - mn)
        return normalized
    
    def simulate_season(
        self,
        driver_strengths: dict,
        team_reliability: dict,
        driver_teams: dict,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation of a full F1 season.
        
        Parameters
        ----------
        driver_strengths : dict
            driver_name -> base strength score (raw, will be normalized)
        team_reliability : dict
            team_name -> reliability probability (0-1)
        driver_teams : dict
            driver_name -> team_name
        
        Returns
        -------
        DataFrame with simulation results per driver
        """
        drivers = list(driver_strengths.keys())
        n_drivers = len(drivers)
        
        # Normalize to 0-100 scale for realistic race simulation
        norm_strengths = self._normalize_strengths(driver_strengths)
        
        # Group teammates for form-variance modeling
        team_drivers = {}
        for d, t in driver_teams.items():
            team_drivers.setdefault(t, []).append(d)
        
        # Track championship points across simulations
        all_points = np.zeros((self.n_simulations, n_drivers))
        all_wins = np.zeros((self.n_simulations, n_drivers))
        
        for sim in range(self.n_simulations):
            season_points = np.zeros(n_drivers)
            season_wins = np.zeros(n_drivers)
            
            # ── Per-season driver form offset ──
            # Each driver gets a season-long form offset.
            # This represents: confidence, car setup affinity, personal life,
            # adaptation speed to new regs. Crucially, this allows teammates
            # to flip — e.g., in some seasons Piastri adapts faster than Norris.
            # σ=5 means ~68% of seasons the offset is within ±5 points,
            # which is enough to flip closely-matched teammates (~30% of the time
            # for a 6-point gap).
            season_form = {
                drivers[i]: self.rng.normal(0, 5)
                for i in range(n_drivers)
            }
            
            for race in range(self.N_RACES_2026):
                race_performances = np.zeros(n_drivers)
                season_progress = race / self.N_RACES_2026
                
                # ── Determine race conditions ──
                is_wet = self.rng.random() < 0.15          # ~15% wet races
                has_safety_car = self.rng.random() < 0.40   # ~40% of races have SC
                has_first_lap_chaos = self.rng.random() < 0.20  # T1 incidents
                
                for i, driver in enumerate(drivers):
                    base = norm_strengths[driver]
                    team = driver_teams[driver]
                    
                    # Apply season form
                    effective_strength = base + season_form[driver]
                    
                    # ── Race-day noise ──
                    # σ=15 for regulation-change year (higher unpredictability
                    # than normal season σ~10, because teams are still learning
                    # the cars, setups are experimental, drivers adapting)
                    noise = self.rng.normal(0, 15)
                    
                    # ── Mechanical failure (DNF) ──
                    base_reliability = team_reliability.get(team, 0.85)
                    # Reliability improves ~30% over the season as PU matures
                    reliability = base_reliability + (1 - base_reliability) * 0.3 * season_progress
                    if self.rng.random() > reliability:
                        race_performances[i] = -100  # DNF
                        continue
                    
                    # ── First-lap incident ──
                    # Any driver can get caught in T1 chaos regardless of pace
                    if has_first_lap_chaos and self.rng.random() < 0.12:
                        # ~12% chance per driver of being involved when chaos occurs
                        if self.rng.random() < 0.4:
                            race_performances[i] = -100  # DNF from lap 1
                            continue
                        else:
                            noise -= 15  # Damage, dropped positions, pit for repairs
                    
                    # ── Weather effect ──
                    # Wet races are great equalizers — midfield can shine
                    weather_effect = 0
                    if is_wet:
                        # Compress field: reduce the advantage of top teams
                        # and add large variance (rain driving skill varies)
                        weather_effect = self.rng.normal(0, 10)
                        effective_strength = (effective_strength * 0.7 + 
                                            30 * 0.3)  # pull toward midfield
                    
                    # ── Safety car shuffle ──
                    # Safety cars compress the field and reward strategy luck
                    sc_effect = 0
                    if has_safety_car:
                        sc_effect = self.rng.normal(0, 6)  # strategy luck/unluck
                    
                    # ── Mid-season development ──
                    # Weaker teams converge as they copy concepts from frontrunners
                    mean_strength = np.mean(list(norm_strengths.values()))
                    development = self.rng.normal(0, 1.5) * season_progress
                    convergence = (mean_strength - base) * 0.10 * season_progress
                    
                    race_performances[i] = (effective_strength + noise + 
                                           weather_effect + sc_effect + 
                                           development + convergence)
                
                # Convert performances to positions and points
                race_order = np.argsort(-race_performances)  # highest perf = P1
                
                for pos, driver_idx in enumerate(race_order):
                    if race_performances[driver_idx] <= -50:  # DNF
                        continue
                    if pos < len(self.POINTS_SYSTEM):
                        season_points[driver_idx] += self.POINTS_SYSTEM[pos]
                    if pos == 0:
                        season_wins[driver_idx] += 1
            
            all_points[sim] = season_points
            all_wins[sim] = season_wins
        
        # Compute statistics
        results = []
        for i, driver in enumerate(drivers):
            pts_dist = all_points[:, i]
            wins_dist = all_wins[:, i]
            
            # Championship wins: count simulations where this driver had most points
            champion_count = 0
            top3_count = 0
            for sim in range(self.n_simulations):
                sim_rankings = np.argsort(-all_points[sim])
                driver_pos = np.where(sim_rankings == i)[0][0]
                if driver_pos == 0:
                    champion_count += 1
                if driver_pos < 3:
                    top3_count += 1
            
            results.append({
                "driver": driver,
                "team": driver_teams[driver],
                "mean_points": pts_dist.mean(),
                "std_points": pts_dist.std(),
                "median_points": np.median(pts_dist),
                "p5_points": np.percentile(pts_dist, 5),
                "p95_points": np.percentile(pts_dist, 95),
                "mean_wins": wins_dist.mean(),
                "p_champion": champion_count / self.n_simulations,
                "p_top3": top3_count / self.n_simulations,
            })
        
        return pd.DataFrame(results).sort_values("p_champion", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════
# CONSTRUCTOR CHAMPIONSHIP PREDICTOR
# ═══════════════════════════════════════════════════

def predict_constructors(driver_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate driver predictions into constructor championship predictions.
    
    Parameters
    ----------
    driver_results : pd.DataFrame
        Output from MonteCarloSimulator with per-driver stats
    
    Returns
    -------
    DataFrame with constructor championship predictions
    """
    constructor_results = driver_results.groupby("team").agg({
        "mean_points": "sum",
        "std_points": lambda x: np.sqrt((x**2).sum()),  # propagate uncertainty
        "mean_wins": "sum",
        "p_champion": "max",  # team's best driver's title chance
        "p_top3": "max",
        "driver": list,
    }).reset_index()
    
    constructor_results.columns = [
        "team", "total_mean_points", "total_std_points",
        "total_mean_wins", "best_driver_p_champion", "best_driver_p_top3",
        "drivers"
    ]
    
    return constructor_results.sort_values("total_mean_points", ascending=False).reset_index(drop=True)
