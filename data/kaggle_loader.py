"""
Kaggle F1 Data Loader
=====================
ETL pipeline that loads and transforms the Kaggle "Formula 1 World Championship
(1950-2025)" dataset into the data structures expected by the prediction model.

Replaces hardcoded DRIVER_SEASON_RESULTS, CONSTRUCTOR_SEASON_RESULTS,
TOP10_DNF_DATA, and QUALIFYING_DATA for seasons 2014-2024.

Data source: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────

KAGGLE_DIR = Path(__file__).parent.parent / "data" / "kaggle_f1"
MIN_YEAR = 2014
MAX_YEAR = 2024  # 2025 stays hardcoded (not in Kaggle)

# Driver name normalization: Kaggle → model convention
DRIVER_NAME_MAP = {
    "Alexander Albon": "Alex Albon",
    "Kimi Räikkönen": "Kimi Raikkonen",
    "Nico Hülkenberg": "Nico Hulkenberg",
    "Sergio Pérez": "Sergio Perez",
}

# Kaggle constructor name → model team name
TEAM_NAME_MAP = {
    "Alpine F1 Team": "Alpine",
    "Haas F1 Team": "Haas",
    "RB F1 Team": "Racing Bulls",
    "Lotus F1": "Lotus",
    "Manor Marussia": "Manor",
}

# Engine supplier mapping: (year, kaggle_team_name) → engine
# Kaggle doesn't track engine suppliers, so we maintain this lookup.
# This is the ONE piece of domain knowledge that can't come from the dataset.
ENGINE_MAP = {
    # 2014
    (2014, "Mercedes"): "Mercedes", (2014, "Red Bull"): "Renault",
    (2014, "Williams"): "Mercedes", (2014, "Ferrari"): "Ferrari",
    (2014, "McLaren"): "Mercedes", (2014, "Force India"): "Mercedes",
    (2014, "Toro Rosso"): "Renault", (2014, "Sauber"): "Ferrari",
    (2014, "Lotus"): "Renault", (2014, "Manor"): "Ferrari",
    (2014, "Caterham"): "Renault",
    # 2015
    (2015, "Mercedes"): "Mercedes", (2015, "Ferrari"): "Ferrari",
    (2015, "Williams"): "Mercedes", (2015, "Red Bull"): "Renault",
    (2015, "Force India"): "Mercedes", (2015, "Lotus"): "Mercedes",
    (2015, "Toro Rosso"): "Renault", (2015, "Sauber"): "Ferrari",
    (2015, "McLaren"): "Honda", (2015, "Manor"): "Ferrari",
    # 2016
    (2016, "Mercedes"): "Mercedes", (2016, "Red Bull"): "TAG Heuer",
    (2016, "Ferrari"): "Ferrari", (2016, "Williams"): "Mercedes",
    (2016, "Force India"): "Mercedes", (2016, "McLaren"): "Honda",
    (2016, "Toro Rosso"): "Ferrari", (2016, "Haas"): "Ferrari",
    (2016, "Renault"): "Renault", (2016, "Sauber"): "Ferrari",
    (2016, "Manor"): "Mercedes",
    # 2017
    (2017, "Mercedes"): "Mercedes", (2017, "Ferrari"): "Ferrari",
    (2017, "Red Bull"): "TAG Heuer", (2017, "Force India"): "Mercedes",
    (2017, "Williams"): "Mercedes", (2017, "McLaren"): "Honda",
    (2017, "Toro Rosso"): "Renault", (2017, "Renault"): "Renault",
    (2017, "Haas"): "Ferrari", (2017, "Sauber"): "Ferrari",
    # 2018
    (2018, "Mercedes"): "Mercedes", (2018, "Ferrari"): "Ferrari",
    (2018, "Red Bull"): "TAG Heuer", (2018, "Renault"): "Renault",
    (2018, "Haas"): "Ferrari", (2018, "McLaren"): "Renault",
    (2018, "Force India"): "Mercedes", (2018, "Sauber"): "Ferrari",
    (2018, "Toro Rosso"): "Honda", (2018, "Williams"): "Mercedes",
    # 2019
    (2019, "Mercedes"): "Mercedes", (2019, "Ferrari"): "Ferrari",
    (2019, "Red Bull"): "Honda", (2019, "McLaren"): "Renault",
    (2019, "Renault"): "Renault", (2019, "Toro Rosso"): "Honda",
    (2019, "Racing Point"): "Mercedes", (2019, "Alfa Romeo"): "Ferrari",
    (2019, "Haas"): "Ferrari", (2019, "Williams"): "Mercedes",
    # 2020
    (2020, "Mercedes"): "Mercedes", (2020, "Red Bull"): "Honda",
    (2020, "McLaren"): "Renault", (2020, "Racing Point"): "Mercedes",
    (2020, "Renault"): "Renault", (2020, "Ferrari"): "Ferrari",
    (2020, "AlphaTauri"): "Honda", (2020, "Alfa Romeo"): "Ferrari",
    (2020, "Haas"): "Ferrari", (2020, "Williams"): "Mercedes",
    # 2021
    (2021, "Mercedes"): "Mercedes", (2021, "Red Bull"): "Honda",
    (2021, "Ferrari"): "Ferrari", (2021, "McLaren"): "Mercedes",
    (2021, "Alpine"): "Renault", (2021, "AlphaTauri"): "Honda",
    (2021, "Aston Martin"): "Mercedes", (2021, "Williams"): "Mercedes",
    (2021, "Alfa Romeo"): "Ferrari", (2021, "Haas"): "Ferrari",
    # 2022
    (2022, "Red Bull"): "RBPT", (2022, "Ferrari"): "Ferrari",
    (2022, "Mercedes"): "Mercedes", (2022, "Alpine"): "Renault",
    (2022, "McLaren"): "Mercedes", (2022, "Alfa Romeo"): "Ferrari",
    (2022, "Aston Martin"): "Mercedes", (2022, "Haas"): "Ferrari",
    (2022, "AlphaTauri"): "RBPT", (2022, "Williams"): "Mercedes",
    # 2023
    (2023, "Red Bull"): "Honda RBPT", (2023, "Mercedes"): "Mercedes",
    (2023, "Ferrari"): "Ferrari", (2023, "McLaren"): "Mercedes",
    (2023, "Aston Martin"): "Mercedes", (2023, "Alpine"): "Renault",
    (2023, "Williams"): "Mercedes", (2023, "AlphaTauri"): "Honda RBPT",
    (2023, "Alfa Romeo"): "Ferrari", (2023, "Haas"): "Ferrari",
    # 2024
    (2024, "Red Bull"): "Honda RBPT", (2024, "McLaren"): "Mercedes",
    (2024, "Ferrari"): "Ferrari", (2024, "Mercedes"): "Mercedes",
    (2024, "Aston Martin"): "Mercedes", (2024, "Racing Bulls"): "Honda RBPT",
    (2024, "Haas"): "Ferrari", (2024, "Alpine"): "Renault",
    (2024, "Williams"): "Mercedes", (2024, "Sauber"): "Ferrari",
}

# Status IDs that count as "finished" (not a DNF)
# 1 = Finished, 11-19 = +1 to +9 Laps, plus various "+N Laps" IDs
_FINISHED_STATUS_IDS = (
    {1} |
    set(range(11, 20)) |
    {45, 50, 53, 55, 58, 88, 111, 112, 113, 114, 115,
     116, 117, 118, 119, 120, 122, 123, 124, 125, 127, 128, 133, 134}
)

# Status IDs that are NOT DNFs but also not "finished" (DQ, withdrew, etc.)
_NON_DNF_NON_FINISH_IDS = {2, 54, 62, 77, 81, 96, 97}


# ─────────────────────────────────────────────────
# CORE LOADING FUNCTIONS
# ─────────────────────────────────────────────────

def _load_raw_tables(kaggle_dir: Path = KAGGLE_DIR) -> dict:
    """Load all raw CSV files into DataFrames."""
    tables = {}
    for name in ["results", "races", "drivers", "constructors",
                  "driver_standings", "constructor_standings",
                  "qualifying", "sprint_results", "status"]:
        path = kaggle_dir / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
        else:
            print(f"  Warning: {path} not found")
    return tables


def _normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply driver and team name normalization."""
    if "driver_name" in df.columns:
        df["driver_name"] = df["driver_name"].replace(DRIVER_NAME_MAP)
    if "team" in df.columns:
        df["team"] = df["team"].replace(TEAM_NAME_MAP)
    return df


def _get_engine(year: int, team: str) -> str:
    """Look up engine supplier for a team-year combination."""
    return ENGINE_MAP.get((year, team), "Unknown")


# ─────────────────────────────────────────────────
# DATA EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────

def load_driver_season_results(kaggle_dir: Path = KAGGLE_DIR) -> list[dict]:
    """
    Build DRIVER_SEASON_RESULTS from Kaggle data (2014-2024).

    Returns list of dicts with keys:
        year, driver, team, engine, position, points, wins, podiums, poles, races
    """
    tables = _load_raw_tables(kaggle_dir)
    results = tables["results"]
    races = tables["races"]
    drivers = tables["drivers"]
    constructors = tables["constructors"]
    standings = tables["driver_standings"]
    qualifying = tables["qualifying"]

    # Merge race-level data
    race_df = results.merge(races[["raceId", "year"]], on="raceId")
    race_df = race_df.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
    race_df = race_df.merge(constructors[["constructorId", "name"]], on="constructorId")
    race_df["driver_name"] = race_df["forename"] + " " + race_df["surname"]
    race_df = race_df.rename(columns={"name": "team"})
    race_df = _normalize_names(race_df)
    race_df = race_df[(race_df["year"] >= MIN_YEAR) & (race_df["year"] <= MAX_YEAR)]

    # Qualifying data for poles
    qual_df = qualifying.merge(races[["raceId", "year"]], on="raceId")
    qual_df = qual_df.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
    qual_df["driver_name"] = qual_df["forename"] + " " + qual_df["surname"]
    qual_df = _normalize_names(qual_df)
    qual_df = qual_df[(qual_df["year"] >= MIN_YEAR) & (qual_df["year"] <= MAX_YEAR)]

    poles = qual_df[qual_df["position"] == 1].groupby(
        ["year", "driver_name"]
    ).size().reset_index(name="poles")

    # Championship standings (final race per year) for position + points
    last_races = races.groupby("year")["raceId"].max().reset_index()
    final_standings = standings.merge(last_races, on="raceId")
    final_standings = final_standings.merge(
        drivers[["driverId", "forename", "surname"]], on="driverId"
    )
    final_standings["driver_name"] = final_standings["forename"] + " " + final_standings["surname"]
    final_standings = _normalize_names(final_standings)
    final_standings = final_standings[
        (final_standings["year"] >= MIN_YEAR) & (final_standings["year"] <= MAX_YEAR)
    ]

    # Aggregate race-level stats
    race_stats = race_df.groupby(["year", "driver_name", "team"]).agg(
        races=("raceId", "count"),
        wins=("positionOrder", lambda x: (x == 1).sum()),
        podiums=("positionOrder", lambda x: (x <= 3).sum()),
    ).reset_index()

    # Merge championship standings
    output = race_stats.merge(
        final_standings[["year", "driver_name", "position", "points"]],
        on=["year", "driver_name"],
        how="left"
    )

    # Merge poles
    output = output.merge(poles, on=["year", "driver_name"], how="left")
    output["poles"] = output["poles"].fillna(0).astype(int)

    # Resolve drivers with multiple teams in a season (e.g., Sainz mid-season swap)
    # Keep the team they raced most races with
    output = output.sort_values("races", ascending=False).drop_duplicates(
        subset=["year", "driver_name"], keep="first"
    )

    # Add engine
    output["engine"] = output.apply(
        lambda r: _get_engine(int(r["year"]), r["team"]), axis=1
    )

    # Build output list
    records = []
    for _, row in output.iterrows():
        records.append({
            "year": int(row["year"]),
            "driver": row["driver_name"],
            "team": row["team"],
            "engine": row["engine"],
            "position": int(row["position"]) if pd.notna(row["position"]) else None,
            "points": float(row["points"]) if pd.notna(row["points"]) else 0,
            "wins": int(row["wins"]),
            "podiums": int(row["podiums"]),
            "poles": int(row["poles"]),
            "races": int(row["races"]),
        })

    return sorted(records, key=lambda r: (r["year"], r["position"] or 99))


def load_constructor_season_results(kaggle_dir: Path = KAGGLE_DIR) -> list[dict]:
    """
    Build CONSTRUCTOR_SEASON_RESULTS from Kaggle data (2014-2024).

    Returns list of dicts with keys:
        year, team, engine, position, points, wins
    """
    tables = _load_raw_tables(kaggle_dir)
    races = tables["races"]
    constructors = tables["constructors"]
    con_standings = tables["constructor_standings"]

    # Final standings per year
    last_races = races.groupby("year")["raceId"].max().reset_index()
    final = con_standings.merge(last_races, on="raceId")
    final = final.merge(constructors[["constructorId", "name"]], on="constructorId")
    final = final.rename(columns={"name": "team"})
    final = _normalize_names(final)
    final = final[(final["year"] >= MIN_YEAR) & (final["year"] <= MAX_YEAR)]

    records = []
    for _, row in final.iterrows():
        team = row["team"]
        year = int(row["year"])
        records.append({
            "year": year,
            "team": team,
            "engine": _get_engine(year, team),
            "position": int(row["position"]),
            "points": float(row["points"]),
            "wins": int(row["wins"]),
        })

    return sorted(records, key=lambda r: (r["year"], r["position"]))


def load_top10_dnf_data(kaggle_dir: Path = KAGGLE_DIR) -> dict:
    """
    Build TOP10_DNF_DATA from Kaggle race-level results (2014-2024).

    Returns dict: (year, driver_name) → (top10_rate, dnf_rate)
    """
    tables = _load_raw_tables(kaggle_dir)
    results = tables["results"]
    races = tables["races"]
    drivers = tables["drivers"]

    df = results.merge(races[["raceId", "year"]], on="raceId")
    df = df.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
    df["driver_name"] = df["forename"] + " " + df["surname"]
    df = _normalize_names(df)
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    df["top10"] = df["positionOrder"] <= 10
    df["is_dnf"] = (
        ~df["statusId"].isin(_FINISHED_STATUS_IDS) &
        ~df["statusId"].isin(_NON_DNF_NON_FINISH_IDS)
    )

    grouped = df.groupby(["year", "driver_name"]).agg(
        races=("raceId", "count"),
        top10_count=("top10", "sum"),
        dnf_count=("is_dnf", "sum"),
    ).reset_index()

    grouped["top10_rate"] = (grouped["top10_count"] / grouped["races"]).round(2)
    grouped["dnf_rate"] = (grouped["dnf_count"] / grouped["races"]).round(2)

    return {
        (int(row["year"]), row["driver_name"]): (row["top10_rate"], row["dnf_rate"])
        for _, row in grouped.iterrows()
    }


def load_qualifying_data(kaggle_dir: Path = KAGGLE_DIR) -> dict:
    """
    Build QUALIFYING_DATA from Kaggle qualifying results (2014-2024).

    Returns dict: (year, driver_name) → {avg_quali_pos, q3_rate, front_row_rate}
    """
    tables = _load_raw_tables(kaggle_dir)
    qualifying = tables["qualifying"]
    races = tables["races"]
    drivers = tables["drivers"]

    df = qualifying.merge(races[["raceId", "year"]], on="raceId")
    df = df.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
    df["driver_name"] = df["forename"] + " " + df["surname"]
    df = _normalize_names(df)
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    # Q3 appearance: qualified in position 1-10 (top 10)
    df["in_q3"] = df["position"] <= 10
    df["front_row"] = df["position"] <= 2

    grouped = df.groupby(["year", "driver_name"]).agg(
        avg_quali_pos=("position", "mean"),
        sessions=("raceId", "count"),
        q3_count=("in_q3", "sum"),
        front_row_count=("front_row", "sum"),
    ).reset_index()

    grouped["q3_rate"] = (grouped["q3_count"] / grouped["sessions"]).round(2)
    grouped["front_row_rate"] = (grouped["front_row_count"] / grouped["sessions"]).round(2)
    grouped["avg_quali_pos"] = grouped["avg_quali_pos"].round(1)

    return {
        (int(row["year"]), row["driver_name"]): {
            "avg_quali_pos": row["avg_quali_pos"],
            "q3_rate": row["q3_rate"],
            "front_row_rate": row["front_row_rate"],
        }
        for _, row in grouped.iterrows()
    }


# ─────────────────────────────────────────────────
# CONVENIENCE: LOAD EVERYTHING
# ─────────────────────────────────────────────────

def load_all(kaggle_dir: Path = KAGGLE_DIR) -> dict:
    """
    Load all datasets from Kaggle CSVs.

    Returns dict with keys:
        driver_results, constructor_results, top10_dnf, qualifying
    """
    return {
        "driver_results": load_driver_season_results(kaggle_dir),
        "constructor_results": load_constructor_season_results(kaggle_dir),
        "top10_dnf": load_top10_dnf_data(kaggle_dir),
        "qualifying": load_qualifying_data(kaggle_dir),
    }


# ─────────────────────────────────────────────────
# CLI: Preview loaded data
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Kaggle F1 data...")
    data = load_all()

    dr = data["driver_results"]
    print(f"\n✓ DRIVER_SEASON_RESULTS: {len(dr)} records")
    from collections import Counter
    year_counts = Counter(r["year"] for r in dr)
    for yr in sorted(year_counts):
        print(f"    {yr}: {year_counts[yr]} drivers")

    cr = data["constructor_results"]
    print(f"\n✓ CONSTRUCTOR_SEASON_RESULTS: {len(cr)} records")

    t10 = data["top10_dnf"]
    print(f"\n✓ TOP10_DNF_DATA: {len(t10)} driver-season records")

    q = data["qualifying"]
    print(f"\n✓ QUALIFYING_DATA: {len(q)} driver-season records")

    # Spot check
    print("\n── Spot Check: Verstappen 2024 ──")
    ver = [r for r in dr if r["driver"] == "Max Verstappen" and r["year"] == 2024][0]
    print(f"    {ver}")
    ver_t10 = t10.get((2024, "Max Verstappen"))
    print(f"    Top-10: {ver_t10[0]:.0%}, DNF: {ver_t10[1]:.0%}")
    ver_q = q.get((2024, "Max Verstappen"))
    print(f"    Avg Quali: P{ver_q['avg_quali_pos']}, Q3: {ver_q['q3_rate']:.0%}")

    print("\n── Spot Check: Hamilton 2014 ──")
    ham = [r for r in dr if r["driver"] == "Lewis Hamilton" and r["year"] == 2014][0]
    print(f"    {ham}")
    ham_t10 = t10.get((2014, "Lewis Hamilton"))
    print(f"    Top-10: {ham_t10[0]:.0%}, DNF: {ham_t10[1]:.0%}")
