"""
Feature Engineering for F1 2026 Prediction
===========================================
Generates ML-ready features from historical data, Elo ratings,
pre-season testing, and regulation change analysis.

Feature categories:
1. Driver features   — Elo, rolling performance, career trajectory
2. Team features     — historical strength, development rate, budget
3. Engine features   — manufacturer competitiveness, PU maturity
4. Regulation change — how teams historically respond to reg resets
5. External signals  — pre-season testing, bookmaker odds
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════
# REGULATION CHANGE IMPACT ANALYSIS
# ═══════════════════════════════════════════════════

# Years with major regulation changes and which teams gained/lost
REGULATION_YEARS = {
    2009: "Aero overhaul (double diffuser era)",
    2014: "Turbo-hybrid V6 introduction",
    2017: "Wider cars, more downforce",
    2022: "Ground effect / porpoising era",
    2026: "New PU regs, active aero, 50/50 ICE/electric split",
}

# Team position changes across regulation resets (pre-reg → post-reg)
REG_CHANGE_HISTORY = {
    # team: [(pre_year_pos, post_year_pos, year_of_change)]
    "Mercedes": [(3, 1, 2014), (1, 1, 2017), (1, 3, 2022)],
    "Ferrari": [(2, 4, 2014), (2, 2, 2017), (2, 2, 2022)],
    "Red Bull": [(1, 2, 2014), (3, 3, 2017), (2, 1, 2022)],
    "McLaren": [(5, 5, 2014), (9, 9, 2017), (4, 5, 2022)],
    "Williams": [(9, 3, 2014), (5, 5, 2017), (10, 10, 2022)],
    "Aston Martin": [(6, 7, 2022)],  # limited history under current name
}


def compute_reg_change_score(team_name: str) -> float:
    """
    Compute a team's historical ability to adapt to regulation changes.
    
    Returns a score 0-100 based on average position improvement/decline
    across regulation resets.
    """
    history = REG_CHANGE_HISTORY.get(team_name, [])
    if not history:
        return 50.0  # neutral for teams with no history

    improvements = []
    for pre_pos, post_pos, _ in history:
        # Positive = improved, negative = declined
        change = pre_pos - post_pos  # lower position number = better
        # Normalize: +5 change → very good, -5 → very bad
        improvements.append(change)

    avg_improvement = np.mean(improvements)
    # Map to 0-100 scale: -5 → 25, 0 → 50, +5 → 75
    score = 50 + avg_improvement * 5
    return np.clip(score, 10, 95)


# ═══════════════════════════════════════════════════
# DRIVER FEATURE ENGINEERING
# ═══════════════════════════════════════════════════

def compute_driver_rolling_features(driver_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Compute rolling features for each driver over a window of seasons.
    
    Features generated:
    - rolling_avg_points_per_race: consistency metric
    - rolling_avg_position: trajectory (improving or declining)
    - rolling_win_rate: peak performance metric
    - rolling_podium_rate: sustained excellence
    - career_seasons: experience factor
    - peak_elo: historical ceiling
    - recent_trend: slope of performance (positive = improving)
    """
    features = []
    
    for driver in driver_df["driver"].unique():
        d_data = driver_df[driver_df["driver"] == driver].sort_values("year")
        active = d_data[d_data["races"] > 0].copy()
        
        if active.empty:
            continue
        
        for idx, row in active.iterrows():
            year = row["year"]
            # Get window of previous seasons (inclusive of current)
            window_data = active[
                (active["year"] <= year) & (active["year"] > year - window)
            ]
            
            feat = {
                "year": year,
                "driver": driver,
                "team": row["team"],
                "engine": row["engine"],
                "position": row["position"],
                "points": row["points"],
                "wins": row["wins"],
                "races": row["races"],
                # Rolling features
                "rolling_avg_ppr": window_data["points_per_race"].mean(),
                "rolling_avg_position": window_data["position"].mean(),
                "rolling_win_rate": window_data["win_rate"].mean(),
                "rolling_podium_rate": window_data["podium_rate"].mean(),
                # Qualifying features (rolling)
                "rolling_avg_quali_pos": window_data["avg_quali_pos"].mean() if "avg_quali_pos" in window_data.columns else 15.0,
                "rolling_q3_rate": window_data["q3_rate"].mean() if "q3_rate" in window_data.columns else 0.0,
                "rolling_front_row_rate": window_data["front_row_rate"].mean() if "front_row_rate" in window_data.columns else 0.0,
                # Consistency & reliability features (rolling)
                "rolling_top_10_rate": window_data["top_10_rate"].mean() if "top_10_rate" in window_data.columns else 0.50,
                "rolling_dnf_rate": window_data["dnf_rate"].mean() if "dnf_rate" in window_data.columns else 0.08,
                # Career features
                "career_seasons": len(active[active["year"] <= year]),
                "career_total_wins": active[active["year"] <= year]["wins"].sum(),
                "career_total_points": active[active["year"] <= year]["points"].sum(),
            }
            
            # Trend: slope of points_per_race over window
            if len(window_data) >= 2:
                x = np.arange(len(window_data))
                y = window_data["points_per_race"].values
                if len(x) > 1 and np.std(y) > 0:
                    slope = np.polyfit(x, y, 1)[0]
                    feat["performance_trend"] = slope
                else:
                    feat["performance_trend"] = 0.0
            else:
                feat["performance_trend"] = 0.0
            
            features.append(feat)
    
    return pd.DataFrame(features)


# ═══════════════════════════════════════════════════
# TEAM FEATURE ENGINEERING
# ═══════════════════════════════════════════════════

def compute_team_features(constructor_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Compute team-level features from constructor championship data.
    
    Features:
    - team_rolling_position: avg constructors position over window
    - team_rolling_points: avg points over window
    - team_win_rate: recent win frequency
    - team_development_rate: year-over-year improvement
    - team_engine_consistency: stability of engine partnership
    """
    features = []
    
    for team in constructor_df["team"].unique():
        t_data = constructor_df[constructor_df["team"] == team].sort_values("year")
        
        for _, row in t_data.iterrows():
            year = row["year"]
            window_data = t_data[
                (t_data["year"] <= year) & (t_data["year"] > year - window)
            ]
            
            feat = {
                "year": year,
                "team": team,
                "engine": row["engine"],
                "constructor_position": row["position"],
                "constructor_points": row["points"],
                # Rolling features
                "team_rolling_position": window_data["position"].mean(),
                "team_rolling_points": window_data["points"].mean(),
                "team_rolling_wins": window_data["wins"].mean(),
                "team_seasons_in_data": len(t_data[t_data["year"] <= year]),
            }
            
            # Development rate: improvement in position year-over-year
            if len(window_data) >= 2:
                positions = window_data["position"].values
                feat["team_development_rate"] = positions[0] - positions[-1]  # positive = improving
            else:
                feat["team_development_rate"] = 0.0
            
            # Regulation change adaptation score
            feat["reg_change_score"] = compute_reg_change_score(team)
            
            features.append(feat)
    
    return pd.DataFrame(features)


# ═══════════════════════════════════════════════════
# ENGINE MANUFACTURER FEATURES
# ═══════════════════════════════════════════════════

ENGINE_MATURITY = {
    # engine_family: (years_in_hybrid_era, titles_won, 2026_generation)
    "Mercedes": {"hybrid_years": 12, "titles_won": 8, "gen": "evolved", "maturity_score": 95},
    "Ferrari": {"hybrid_years": 12, "titles_won": 0, "gen": "evolved", "maturity_score": 82},
    "Honda RBPT": {"hybrid_years": 7, "titles_won": 4, "gen": "evolved", "maturity_score": 80},
    "Ford (RBPT)": {"hybrid_years": 0, "titles_won": 0, "gen": "new", "maturity_score": 65},
    "Honda": {"hybrid_years": 7, "titles_won": 4, "gen": "new_for_aston", "maturity_score": 70},
    "Audi": {"hybrid_years": 0, "titles_won": 0, "gen": "new", "maturity_score": 40},
    "Renault": {"hybrid_years": 12, "titles_won": 0, "gen": "discontinued", "maturity_score": 50},
    "Ferrari (customer)": {"hybrid_years": 12, "titles_won": 0, "gen": "customer", "maturity_score": 70},
}

# Works team bonus: historically the factory team has deep PU integration
# advantages over customer teams, especially in year 1 of new regs.
# In 2014, Mercedes works team won 16/19 races while customer Williams
# won 0. The works team gets the PU earlier, has tighter chassis-PU
# integration, runs full-spec software, and co-develops cooling/packaging.
WORKS_TEAM_MAP = {
    "Mercedes": "Mercedes",      # Mercedes is own works team
    "Ferrari": "Ferrari",         # Ferrari is own works team  
    "Red Bull": "Ford (RBPT)",    # Red Bull co-developed their PU with Ford
    "Audi": "Audi",              # Audi is own works team (ex-Sauber)
    "Racing Bulls": None,         # customer of Red Bull/Ford PU
    "McLaren": None,              # customer of Mercedes PU
    "Williams": None,             # customer of Mercedes PU
    "Alpine": None,               # customer of Mercedes PU (switched from Renault)
    "Aston Martin": None,         # customer of Honda PU
    "Haas": None,                 # customer of Ferrari PU
    "Cadillac": None,             # customer of Ferrari PU
}

WORKS_TEAM_BONUS = 15  # points added to engine maturity for works teams


def get_engine_features(engine: str) -> Dict:
    """Get engine manufacturer features for a given PU."""
    # Normalize engine name to closest match
    for key in ENGINE_MATURITY:
        if key.lower() in engine.lower() or engine.lower() in key.lower():
            return ENGINE_MATURITY[key]
    return {"hybrid_years": 0, "titles_won": 0, "gen": "unknown", "maturity_score": 50}


# ═══════════════════════════════════════════════════
# FULL FEATURE MATRIX BUILDER
# ═══════════════════════════════════════════════════

def build_feature_matrix(
    driver_df: pd.DataFrame,
    constructor_df: pd.DataFrame,
    elo_system,
    target_col: str = "position"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the complete feature matrix for model training.
    
    Merges driver features, team features, engine features, and Elo ratings
    into a single ML-ready DataFrame.
    
    Parameters
    ----------
    driver_df : pd.DataFrame
        From historical_data.get_driver_df()
    constructor_df : pd.DataFrame
        From historical_data.get_constructor_df()
    elo_system : F1EloSystem
        Populated Elo rating system
    target_col : str
        Target variable ('position' for classification)
    
    Returns
    -------
    X : pd.DataFrame - feature matrix
    y : pd.Series - target variable
    """
    # Compute rolling features
    driver_features = compute_driver_rolling_features(driver_df)
    team_features = compute_team_features(constructor_df)
    
    # Merge driver + team features
    merged = driver_features.merge(
        team_features[["year", "team", "team_rolling_position", "team_rolling_points",
                        "team_rolling_wins", "team_development_rate", "reg_change_score"]],
        on=["year", "team"],
        how="left"
    )
    
    # Add Elo ratings from history
    elo_history = elo_system.get_history_df()
    merged = merged.merge(
        elo_history.rename(columns={"elo_rating": "elo_at_season"}),
        on=["year", "driver"],
        how="left"
    )
    
    # Add engine features
    for eng_feat in ["hybrid_years", "titles_won", "maturity_score"]:
        merged[f"engine_{eng_feat}"] = merged["engine"].apply(
            lambda e: get_engine_features(str(e)).get(eng_feat, 0) if pd.notna(e) else 0
        )
    
    # Add regulation change indicator
    merged["is_reg_change_year"] = merged["year"].apply(lambda y: 1 if y in REGULATION_YEARS else 0)
    
    # Interaction features
    merged["elo_x_team_pos"] = merged["elo_at_season"] * (1 / (merged["team_rolling_position"] + 1))
    merged["trend_x_reg"] = merged["performance_trend"] * merged["is_reg_change_year"]
    
    # Filter to rows with valid targets
    valid = merged[merged[target_col].notna() & (merged["races"] > 0)].copy()
    
    # Select feature columns
    feature_cols = [
        "rolling_avg_ppr", "rolling_avg_position", "rolling_win_rate",
        "rolling_podium_rate", "career_seasons", "career_total_wins",
        "performance_trend", "team_rolling_position", "team_rolling_points",
        "team_rolling_wins", "team_development_rate", "reg_change_score",
        "elo_at_season", "engine_hybrid_years", "engine_titles_won",
        "engine_maturity_score", "is_reg_change_year",
        "elo_x_team_pos", "trend_x_reg",
    ]
    
    X = valid[feature_cols].fillna(0)
    y = valid[target_col]
    
    return X, y, valid


def build_2026_features(
    driver_df: pd.DataFrame,
    constructor_df: pd.DataFrame,
    elo_system,
    grid_2026: list,
    preseason_data: dict,
    bookmaker_odds: dict
) -> pd.DataFrame:
    """
    Build feature vectors for 2026 predictions.
    
    Combines latest historical features with 2026-specific signals
    (pre-season testing, bookmaker odds, new team/engine info).
    """
    features = []
    
    for entry in grid_2026:
        driver = entry["driver"]
        team = entry["team"]
        engine = entry["engine"]
        
        # Get latest driver data (from 2025 or most recent season)
        d_hist = driver_df[driver_df["driver"] == driver].sort_values("year")
        active_hist = d_hist[d_hist["races"] > 0]
        
        # Rolling features from last 3 seasons
        recent = active_hist.tail(3)
        
        feat = {
            "driver": driver,
            "team": team,
            "engine": engine,
            # Driver rolling features
            "rolling_avg_ppr": recent["points_per_race"].mean() if len(recent) > 0 else 0,
            "rolling_avg_position": recent["position"].mean() if len(recent) > 0 else 15,
            "rolling_win_rate": recent["win_rate"].mean() if len(recent) > 0 else 0,
            "rolling_podium_rate": recent["podium_rate"].mean() if len(recent) > 0 else 0,
            # Qualifying features
            "rolling_avg_quali_pos": recent["avg_quali_pos"].mean() if len(recent) > 0 and "avg_quali_pos" in recent.columns else 15.0,
            "rolling_q3_rate": recent["q3_rate"].mean() if len(recent) > 0 and "q3_rate" in recent.columns else 0.0,
            "rolling_front_row_rate": recent["front_row_rate"].mean() if len(recent) > 0 and "front_row_rate" in recent.columns else 0.0,
            "rolling_top_10_rate": recent["top_10_rate"].mean() if len(recent) > 0 and "top_10_rate" in recent.columns else 0.50,
            "rolling_dnf_rate": recent["dnf_rate"].mean() if len(recent) > 0 and "dnf_rate" in recent.columns else 0.08,
            "career_seasons": len(active_hist),
            "career_total_wins": active_hist["wins"].sum() if len(active_hist) > 0 else 0,
        }
        
        # Performance trend
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = recent["points_per_race"].values
            feat["performance_trend"] = np.polyfit(x, y, 1)[0] if np.std(y) > 0 else 0
        else:
            feat["performance_trend"] = 0
        
        # Team features from 2025
        t_hist = constructor_df[constructor_df["team"] == team].sort_values("year")
        # Handle team name mapping for new teams
        team_mapped = team
        if team == "Audi":
            t_hist = constructor_df[constructor_df["team"].isin(["Sauber", "Audi"])].sort_values("year")
        elif team == "Cadillac":
            t_hist = pd.DataFrame()  # brand new team
        
        recent_team = t_hist.tail(3)
        feat["team_rolling_position"] = recent_team["position"].mean() if len(recent_team) > 0 else 10
        feat["team_rolling_points"] = recent_team["points"].mean() if len(recent_team) > 0 else 30
        feat["team_rolling_wins"] = recent_team["wins"].mean() if len(recent_team) > 0 else 0
        
        if len(recent_team) >= 2:
            positions = recent_team["position"].values
            feat["team_development_rate"] = positions[0] - positions[-1]
        else:
            feat["team_development_rate"] = 0
        
        feat["reg_change_score"] = compute_reg_change_score(team)
        
        # Elo rating
        feat["elo_at_season"] = elo_system.get_rating(driver)
        
        # Engine features for 2026
        eng_feats = get_engine_features(engine)
        feat["engine_hybrid_years"] = eng_feats["hybrid_years"]
        feat["engine_titles_won"] = eng_feats["titles_won"]
        
        # Apply works team bonus: factory teams get deeper PU integration
        # In 2014, Mercedes works team won 16/19 races while customer Williams
        # won 0 despite the same PU. The works advantage includes:
        # - Earlier PU access during development
        # - Tighter chassis-PU packaging co-development  
        # - Full-spec software and calibration maps
        # - Dedicated PU engineers embedded in the team
        is_works_team = WORKS_TEAM_MAP.get(team) is not None
        base_maturity = eng_feats["maturity_score"]
        feat["engine_maturity_score"] = base_maturity + (WORKS_TEAM_BONUS if is_works_team else 0)
        feat["is_works_team"] = 1 if is_works_team else 0
        
        # Regulation change year = 1 (2026 is a major reg change)
        feat["is_reg_change_year"] = 1
        
        # Interaction features
        feat["elo_x_team_pos"] = feat["elo_at_season"] * (1 / (feat["team_rolling_position"] + 1))
        feat["trend_x_reg"] = feat["performance_trend"] * feat["is_reg_change_year"]
        
        # === 2026-SPECIFIC FEATURES ===
        
        # Pre-season testing data
        test_data = preseason_data.get(team, {})
        feat["preseason_rank"] = test_data.get("rank", 11)
        feat["preseason_reliability"] = test_data.get("reliability_score", 50)
        feat["preseason_expert_rating"] = test_data.get("expert_rating", 40)
        feat["preseason_laps"] = test_data.get("laps", 150)
        
        # Bookmaker implied probability
        feat["bookmaker_prob"] = bookmaker_odds.get(driver, 0.005)
        
        features.append(feat)
    
    return pd.DataFrame(features)
