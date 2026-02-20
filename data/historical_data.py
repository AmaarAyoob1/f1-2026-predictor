"""
F1 Historical Race Data (2014-2025)
===================================
Hybrid data pipeline:
  - 2014-2024: Loaded from Kaggle F1 World Championship dataset via kaggle_loader.py
  - 2025: Hardcoded from Fox Sports / FIA official standings (not yet in Kaggle)
  - 2026: Pre-season predictions (grid, testing, bookmaker odds)

Data source (2014-2024): https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data.kaggle_loader import (
    load_driver_season_results,
    load_constructor_season_results,
    load_top10_dnf_data,
    load_qualifying_data,
)


# ═════════════════════════════════════════════════
# 2014-2024: LOADED FROM KAGGLE
# ═════════════════════════════════════════════════

_KAGGLE_DIR = Path(__file__).parent / "kaggle_f1"

_kaggle_driver_results = load_driver_season_results(_KAGGLE_DIR)
_kaggle_constructor_results = load_constructor_season_results(_KAGGLE_DIR)
_kaggle_top10_dnf = load_top10_dnf_data(_KAGGLE_DIR)
_kaggle_qualifying = load_qualifying_data(_KAGGLE_DIR)


# ═════════════════════════════════════════════════
# 2025: HARDCODED (not yet in Kaggle dataset)
# Source: Fox Sports / FIA official standings
# ═════════════════════════════════════════════════

_DRIVER_SEASON_2025 = [
    # 2025 — Norris WDC (423 pts, +2 over Verstappen), McLaren WCC
    {"year": 2025, "driver": "Lando Norris", "team": "McLaren", "engine": "Mercedes", "position": 1, "points": 423, "wins": 7, "podiums": 18, "poles": 7, "races": 24},
    {"year": 2025, "driver": "Max Verstappen", "team": "Red Bull", "engine": "Honda RBPT", "position": 2, "points": 421, "wins": 8, "podiums": 15, "poles": 8, "races": 24},
    {"year": 2025, "driver": "Oscar Piastri", "team": "McLaren", "engine": "Mercedes", "position": 3, "points": 410, "wins": 7, "podiums": 16, "poles": 6, "races": 24},
    {"year": 2025, "driver": "George Russell", "team": "Mercedes", "engine": "Mercedes", "position": 4, "points": 319, "wins": 2, "podiums": 9, "poles": 3, "races": 24},
    {"year": 2025, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 5, "points": 242, "wins": 0, "podiums": 7, "poles": 1, "races": 24},
    {"year": 2025, "driver": "Lewis Hamilton", "team": "Ferrari", "engine": "Ferrari", "position": 6, "points": 156, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Kimi Antonelli", "team": "Mercedes", "engine": "Mercedes", "position": 7, "points": 150, "wins": 0, "podiums": 1, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Alex Albon", "team": "Williams", "engine": "Mercedes", "position": 8, "points": 73, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Carlos Sainz", "team": "Williams", "engine": "Mercedes", "position": 9, "points": 64, "wins": 0, "podiums": 2, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Fernando Alonso", "team": "Aston Martin", "engine": "Mercedes", "position": 10, "points": 56, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Nico Hulkenberg", "team": "Sauber", "engine": "Ferrari", "position": 11, "points": 51, "wins": 0, "podiums": 1, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Isack Hadjar", "team": "Racing Bulls", "engine": "Honda RBPT", "position": 12, "points": 51, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Oliver Bearman", "team": "Haas", "engine": "Ferrari", "position": 13, "points": 41, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Liam Lawson", "team": "Racing Bulls", "engine": "Honda RBPT", "position": 14, "points": 38, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2025, "driver": "Esteban Ocon", "team": "Haas", "engine": "Ferrari", "position": 15, "points": 38, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Lance Stroll", "team": "Aston Martin", "engine": "Mercedes", "position": 16, "points": 33, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Yuki Tsunoda", "team": "Red Bull", "engine": "Honda RBPT", "position": 17, "points": 33, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2025, "driver": "Pierre Gasly", "team": "Alpine", "engine": "Renault", "position": 18, "points": 22, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Gabriel Bortoleto", "team": "Sauber", "engine": "Ferrari", "position": 19, "points": 19, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Franco Colapinto", "team": "Alpine", "engine": "Renault", "position": 20, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 18},
    {"year": 2025, "driver": "Jack Doohan", "team": "Alpine", "engine": "Renault", "position": 21, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 6},
    {"year": 2025, "driver": "Sergio Perez", "team": None, "engine": None, "position": None, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 0},
    {"year": 2025, "driver": "Valtteri Bottas", "team": None, "engine": None, "position": None, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 0},
]

_CONSTRUCTOR_SEASON_2025 = [
    {"year": 2025, "team": "McLaren", "engine": "Mercedes", "position": 1, "points": 833, "wins": 14},
    {"year": 2025, "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 469, "wins": 2},
    {"year": 2025, "team": "Red Bull", "engine": "Honda RBPT", "position": 3, "points": 454, "wins": 8},
    {"year": 2025, "team": "Ferrari", "engine": "Ferrari", "position": 4, "points": 398, "wins": 0},
    {"year": 2025, "team": "Williams", "engine": "Mercedes", "position": 5, "points": 137, "wins": 0},
    {"year": 2025, "team": "Racing Bulls", "engine": "Honda RBPT", "position": 6, "points": 92, "wins": 0},
    {"year": 2025, "team": "Aston Martin", "engine": "Mercedes", "position": 7, "points": 89, "wins": 0},
    {"year": 2025, "team": "Haas", "engine": "Ferrari", "position": 8, "points": 79, "wins": 0},
    {"year": 2025, "team": "Sauber", "engine": "Ferrari", "position": 9, "points": 70, "wins": 0},
    {"year": 2025, "team": "Alpine", "engine": "Renault", "position": 10, "points": 22, "wins": 0},
]

_TOP10_DNF_2025 = {
    (2025, "Lando Norris"): (0.83, 0.08), (2025, "Max Verstappen"): (0.96, 0.04),
    (2025, "Oscar Piastri"): (0.92, 0.04), (2025, "George Russell"): (0.96, 0.00),
    (2025, "Charles Leclerc"): (0.79, 0.08), (2025, "Lewis Hamilton"): (0.75, 0.08),
    (2025, "Kimi Antonelli"): (0.58, 0.17), (2025, "Alex Albon"): (0.46, 0.17),
    (2025, "Carlos Sainz"): (0.42, 0.17), (2025, "Fernando Alonso"): (0.38, 0.21),
    (2025, "Nico Hulkenberg"): (0.33, 0.12), (2025, "Isack Hadjar"): (0.42, 0.12),
    (2025, "Oliver Bearman"): (0.38, 0.12), (2025, "Liam Lawson"): (0.29, 0.21),
    (2025, "Esteban Ocon"): (0.38, 0.04), (2025, "Lance Stroll"): (0.25, 0.08),
    (2025, "Yuki Tsunoda"): (0.29, 0.04), (2025, "Pierre Gasly"): (0.21, 0.08),
    (2025, "Gabriel Bortoleto"): (0.21, 0.21), (2025, "Franco Colapinto"): (0.17, 0.11),
    (2025, "Jack Doohan"): (0.17, 0.17), (2025, "Sergio Perez"): (0.00, 0.00),
    (2025, "Valtteri Bottas"): (0.00, 0.00),
}

_QUALIFYING_2025 = {
    (2025, "Lando Norris"):       {"avg_quali_pos": 3.0, "q3_rate": 0.96, "front_row_rate": 0.42},
    (2025, "Max Verstappen"):     {"avg_quali_pos": 2.5, "q3_rate": 1.00, "front_row_rate": 0.58},
    (2025, "Oscar Piastri"):      {"avg_quali_pos": 3.5, "q3_rate": 0.96, "front_row_rate": 0.33},
    (2025, "George Russell"):     {"avg_quali_pos": 3.0, "q3_rate": 0.96, "front_row_rate": 0.38},
    (2025, "Charles Leclerc"):    {"avg_quali_pos": 5.0, "q3_rate": 0.88, "front_row_rate": 0.08},
    (2025, "Lewis Hamilton"):     {"avg_quali_pos": 12.0, "q3_rate": 0.33, "front_row_rate": 0.00},
    (2025, "Kimi Antonelli"):     {"avg_quali_pos": 6.5, "q3_rate": 0.79, "front_row_rate": 0.08},
    (2025, "Isack Hadjar"):       {"avg_quali_pos": 9.5, "q3_rate": 0.46, "front_row_rate": 0.00},
    (2025, "Carlos Sainz"):       {"avg_quali_pos": 9.0, "q3_rate": 0.50, "front_row_rate": 0.00},
    (2025, "Fernando Alonso"):    {"avg_quali_pos": 10.0, "q3_rate": 0.38, "front_row_rate": 0.00},
    (2025, "Nico Hulkenberg"):    {"avg_quali_pos": 9.5, "q3_rate": 0.42, "front_row_rate": 0.00},
    (2025, "Oliver Bearman"):     {"avg_quali_pos": 11.0, "q3_rate": 0.29, "front_row_rate": 0.00},
    (2025, "Liam Lawson"):        {"avg_quali_pos": 12.0, "q3_rate": 0.18, "front_row_rate": 0.00},
    (2025, "Esteban Ocon"):       {"avg_quali_pos": 11.5, "q3_rate": 0.25, "front_row_rate": 0.00},
    (2025, "Lance Stroll"):       {"avg_quali_pos": 12.5, "q3_rate": 0.17, "front_row_rate": 0.00},
    (2025, "Yuki Tsunoda"):       {"avg_quali_pos": 10.5, "q3_rate": 0.32, "front_row_rate": 0.00},
    (2025, "Pierre Gasly"):       {"avg_quali_pos": 12.0, "q3_rate": 0.21, "front_row_rate": 0.00},
    (2025, "Gabriel Bortoleto"):  {"avg_quali_pos": 11.5, "q3_rate": 0.21, "front_row_rate": 0.00},
    (2025, "Franco Colapinto"):   {"avg_quali_pos": 15.0, "q3_rate": 0.06, "front_row_rate": 0.00},
    (2025, "Jack Doohan"):        {"avg_quali_pos": 16.0, "q3_rate": 0.00, "front_row_rate": 0.00},
    (2025, "Alex Albon"):         {"avg_quali_pos": 13.0, "q3_rate": 0.13, "front_row_rate": 0.00},
    (2025, "Sergio Perez"):       {"avg_quali_pos": 20.0, "q3_rate": 0.00, "front_row_rate": 0.00},
    (2025, "Valtteri Bottas"):    {"avg_quali_pos": 20.0, "q3_rate": 0.00, "front_row_rate": 0.00},
}


# ═════════════════════════════════════════════════
# COMBINED DATASETS (2014-2025)
# ═════════════════════════════════════════════════

DRIVER_SEASON_RESULTS = _kaggle_driver_results + _DRIVER_SEASON_2025
CONSTRUCTOR_SEASON_RESULTS = _kaggle_constructor_results + _CONSTRUCTOR_SEASON_2025
TOP10_DNF_DATA = {**_kaggle_top10_dnf, **_TOP10_DNF_2025}
QUALIFYING_DATA = {**_kaggle_qualifying, **_QUALIFYING_2025}


# ═════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════

def get_driver_df():
    """Return driver season results as a DataFrame with qualifying data."""
    df = pd.DataFrame(DRIVER_SEASON_RESULTS)
    df["points_per_race"] = df.apply(
        lambda r: r["points"] / r["races"] if r["races"] > 0 else 0, axis=1
    )
    df["win_rate"] = df.apply(
        lambda r: r["wins"] / r["races"] if r["races"] > 0 else 0, axis=1
    )
    df["podium_rate"] = df.apply(
        lambda r: r["podiums"] / r["races"] if r["races"] > 0 else 0, axis=1
    )

    # Merge qualifying data
    df["avg_quali_pos"] = df.apply(
        lambda r: QUALIFYING_DATA.get((r["year"], r["driver"]), {}).get("avg_quali_pos", 15.0), axis=1
    )
    df["q3_rate"] = df.apply(
        lambda r: QUALIFYING_DATA.get((r["year"], r["driver"]), {}).get("q3_rate", 0.0), axis=1
    )
    df["front_row_rate"] = df.apply(
        lambda r: QUALIFYING_DATA.get((r["year"], r["driver"]), {}).get("front_row_rate", 0.0), axis=1
    )

    # Merge top-10 finish rate and DNF rate
    df["top_10_rate"] = df.apply(
        lambda r: TOP10_DNF_DATA.get((r["year"], r["driver"]), (0.30, 0.08))[0], axis=1
    )
    df["dnf_rate"] = df.apply(
        lambda r: TOP10_DNF_DATA.get((r["year"], r["driver"]), (0.30, 0.08))[1], axis=1
    )
    return df


def get_constructor_df():
    """Return constructor season results as a DataFrame."""
    df = pd.DataFrame(CONSTRUCTOR_SEASON_RESULTS)
    return df


# ═════════════════════════════════════════════════
# 2026 GRID & PRE-SEASON DATA
# ═════════════════════════════════════════════════

GRID_2026 = [
    {"driver": "Lando Norris", "team": "McLaren", "engine": "Mercedes"},
    {"driver": "Oscar Piastri", "team": "McLaren", "engine": "Mercedes"},
    {"driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari"},
    {"driver": "Lewis Hamilton", "team": "Ferrari", "engine": "Ferrari"},
    {"driver": "Max Verstappen", "team": "Red Bull", "engine": "Ford (RBPT)"},
    {"driver": "Isack Hadjar", "team": "Red Bull", "engine": "Ford (RBPT)"},
    {"driver": "George Russell", "team": "Mercedes", "engine": "Mercedes"},
    {"driver": "Kimi Antonelli", "team": "Mercedes", "engine": "Mercedes"},
    {"driver": "Fernando Alonso", "team": "Aston Martin", "engine": "Honda"},
    {"driver": "Lance Stroll", "team": "Aston Martin", "engine": "Honda"},
    {"driver": "Alex Albon", "team": "Williams", "engine": "Mercedes"},
    {"driver": "Carlos Sainz", "team": "Williams", "engine": "Mercedes"},
    {"driver": "Pierre Gasly", "team": "Alpine", "engine": "Mercedes"},
    {"driver": "Franco Colapinto", "team": "Alpine", "engine": "Mercedes"},
    {"driver": "Esteban Ocon", "team": "Haas", "engine": "Ferrari"},
    {"driver": "Oliver Bearman", "team": "Haas", "engine": "Ferrari"},
    {"driver": "Nico Hulkenberg", "team": "Audi", "engine": "Audi"},
    {"driver": "Gabriel Bortoleto", "team": "Audi", "engine": "Audi"},
    {"driver": "Liam Lawson", "team": "Racing Bulls", "engine": "Ford (RBPT)"},
    {"driver": "Arvid Lindblad", "team": "Racing Bulls", "engine": "Ford (RBPT)"},
    {"driver": "Sergio Perez", "team": "Cadillac", "engine": "Ferrari (customer)"},
    {"driver": "Valtteri Bottas", "team": "Cadillac", "engine": "Ferrari (customer)"},
]

PRESEASON_TESTING_2026 = {
    "Mercedes": {"rank": 1, "fastest_time": 93.669, "laps": 350, "reliability_score": 85, "expert_rating": 95},
    "Ferrari": {"rank": 2, "fastest_time": 94.1, "laps": 420, "reliability_score": 95, "expert_rating": 92},
    "McLaren": {"rank": 4, "fastest_time": 94.5, "laps": 422, "reliability_score": 98, "expert_rating": 82},
    "Red Bull": {"rank": 3, "fastest_time": 94.3, "laps": 380, "reliability_score": 90, "expert_rating": 85},
    "Racing Bulls": {"rank": 5, "fastest_time": 95.2, "laps": 340, "reliability_score": 88, "expert_rating": 68},
    "Haas": {"rank": 6, "fastest_time": 95.4, "laps": 310, "reliability_score": 85, "expert_rating": 62},
    "Williams": {"rank": 7, "fastest_time": 95.6, "laps": 422, "reliability_score": 92, "expert_rating": 58},
    "Alpine": {"rank": 8, "fastest_time": 95.8, "laps": 300, "reliability_score": 80, "expert_rating": 55},
    "Audi": {"rank": 9, "fastest_time": 96.0, "laps": 280, "reliability_score": 70, "expert_rating": 48},
    "Aston Martin": {"rank": 10, "fastest_time": 97.5, "laps": 250, "reliability_score": 60, "expert_rating": 42},
    "Cadillac": {"rank": 11, "fastest_time": 98.0, "laps": 200, "reliability_score": 55, "expert_rating": 30},
}

BOOKMAKER_ODDS_DRIVERS_2026 = {
    "George Russell": 0.28,
    "Max Verstappen": 0.22,
    "Lando Norris": 0.16,
    "Oscar Piastri": 0.10,
    "Charles Leclerc": 0.09,
    "Lewis Hamilton": 0.05,
    "Kimi Antonelli": 0.04,
    "Carlos Sainz": 0.02,
    "Alex Albon": 0.01,
    "Fernando Alonso": 0.01,
    "Isack Hadjar": 0.01,
    "Others": 0.01,
}

BOOKMAKER_ODDS_CONSTRUCTORS_2026 = {
    "Mercedes": 0.35,
    "Ferrari": 0.20,
    "Red Bull": 0.18,
    "McLaren": 0.15,
    "Williams": 0.03,
    "Racing Bulls": 0.02,
    "Haas": 0.02,
    "Aston Martin": 0.02,
    "Alpine": 0.01,
    "Audi": 0.01,
    "Cadillac": 0.01,
}
