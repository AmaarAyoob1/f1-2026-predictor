"""
F1 Elo Rating System
====================
Computes dynamic Elo ratings for drivers based on head-to-head 
season results. Elo ratings capture relative driver strength and 
naturally decay with inactivity.

Methodology inspired by FiveThirtyEight's Elo approach adapted
for F1's unique championship structure.
"""

import pandas as pd
import numpy as np


class F1EloSystem:
    """
    Elo rating system for Formula 1 drivers.
    
    Each driver starts at a base rating (1500). After each season,
    ratings are updated based on:
    - Championship position relative to expected position
    - Points share within team (teammate comparison)
    - Win and podium bonuses
    
    K-factor adapts based on career length (higher K for new drivers).
    Ratings regress toward the mean between seasons.
    """

    BASE_RATING = 1500
    K_FACTOR_NEW = 48      # first 2 seasons
    K_FACTOR_ESTABLISHED = 32  # 3+ seasons
    REGRESSION_FACTOR = 0.15   # regress 15% toward mean between seasons
    INACTIVITY_PENALTY = 50    # per season missed

    def __init__(self):
        self.ratings = {}       # driver -> current rating
        self.history = {}       # driver -> [(year, rating)]
        self.seasons_active = {}  # driver -> number of seasons

    def get_k_factor(self, driver):
        """Adaptive K-factor: higher for newcomers, lower for veterans."""
        seasons = self.seasons_active.get(driver, 0)
        if seasons <= 2:
            return self.K_FACTOR_NEW
        elif seasons <= 5:
            return 40
        else:
            return self.K_FACTOR_ESTABLISHED

    def expected_score(self, rating_a, rating_b):
        """Standard Elo expected score."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def process_season(self, year, season_data):
        """
        Update Elo ratings based on a season's results.
        
        Parameters
        ----------
        year : int
        season_data : list of dicts with keys:
            driver, team, position, points, wins, podiums, races
        """
        # Filter to drivers who actually raced
        active = [d for d in season_data if d.get("races", 0) > 0 and d.get("position") is not None]
        
        if not active:
            return

        # Initialize new drivers
        for d in active:
            name = d["driver"]
            if name not in self.ratings:
                self.ratings[name] = self.BASE_RATING
                self.history[name] = []
                self.seasons_active[name] = 0

        # Sort by championship position
        active_sorted = sorted(active, key=lambda x: x["position"])
        n = len(active_sorted)

        # Pairwise Elo updates
        for i, driver_a in enumerate(active_sorted):
            name_a = driver_a["driver"]
            k_a = self.get_k_factor(name_a)
            total_update = 0.0
            comparisons = 0

            for j, driver_b in enumerate(active_sorted):
                if i == j:
                    continue
                name_b = driver_b["driver"]
                
                # Actual score: 1 if beat, 0 if lost, 0.5 if same position
                if driver_a["position"] < driver_b["position"]:
                    actual = 1.0
                elif driver_a["position"] > driver_b["position"]:
                    actual = 0.0
                else:
                    actual = 0.5

                expected = self.expected_score(
                    self.ratings[name_a], self.ratings[name_b]
                )
                total_update += actual - expected
                comparisons += 1

            if comparisons > 0:
                # Scale update — don't over-penalize in large fields
                avg_update = total_update / comparisons
                elo_change = k_a * avg_update

                # Bonus for wins/podiums (captures excellence beyond position)
                win_bonus = driver_a.get("wins", 0) * 3
                podium_bonus = driver_a.get("podiums", 0) * 1.2
                
                # Points-per-race bonus (rewards consistency)
                ppr = driver_a.get("points", 0) / max(driver_a.get("races", 1), 1)
                ppr_bonus = min(ppr / 25 * 8, 10)  # capped at 10

                self.ratings[name_a] += elo_change + win_bonus + podium_bonus + ppr_bonus

            self.seasons_active[name_a] = self.seasons_active.get(name_a, 0) + 1
            self.history[name_a].append((year, self.ratings[name_a]))

        # Apply regression toward mean between seasons
        mean_rating = np.mean(list(self.ratings.values()))
        for name in self.ratings:
            self.ratings[name] += self.REGRESSION_FACTOR * (mean_rating - self.ratings[name])

    def apply_inactivity_penalty(self, active_drivers):
        """Penalize drivers who missed a season (e.g., Bottas, Perez in 2025)."""
        for driver, rating in self.ratings.items():
            if driver not in active_drivers:
                self.ratings[driver] = max(
                    self.BASE_RATING - 200,
                    rating - self.INACTIVITY_PENALTY
                )

    def get_current_ratings(self):
        """Return dict of driver -> current Elo rating."""
        return dict(self.ratings)

    def get_rating(self, driver):
        """Get a single driver's rating (returns base if unknown)."""
        return self.ratings.get(driver, self.BASE_RATING)

    def get_ratings_df(self):
        """Return ratings as a sorted DataFrame."""
        data = [
            {"driver": d, "elo_rating": round(r, 1), "seasons": self.seasons_active.get(d, 0)}
            for d, r in self.ratings.items()
        ]
        return pd.DataFrame(data).sort_values("elo_rating", ascending=False).reset_index(drop=True)

    def get_history_df(self):
        """Return full Elo history as DataFrame for plotting."""
        rows = []
        for driver, hist in self.history.items():
            for year, rating in hist:
                rows.append({"driver": driver, "year": year, "elo_rating": round(rating, 1)})
        return pd.DataFrame(rows)


def build_elo_ratings(driver_df):
    """
    Build Elo ratings from historical driver season data.
    
    Parameters
    ----------
    driver_df : pd.DataFrame
        From historical_data.get_driver_df()
    
    Returns
    -------
    F1EloSystem
        Populated Elo system with ratings through 2025.
    """
    elo = F1EloSystem()

    for year in sorted(driver_df["year"].unique()):
        year_data = driver_df[driver_df["year"] == year].to_dict("records")
        elo.process_season(year, year_data)

        # Track who was active for inactivity penalties
        active = {d["driver"] for d in year_data if d.get("races", 0) > 0}
        elo.apply_inactivity_penalty(active)

    return elo
