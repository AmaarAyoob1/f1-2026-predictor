"""
F1 Historical Race Data (2014-2025)
===================================
Curated dataset covering the hybrid era, which is most relevant for
predicting 2026 performance under new hybrid PU regulations.

Data sources: formula1.com, Wikipedia F1 season articles, StatsF1
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────
# SEASON-LEVEL RESULTS (Drivers' Championship)
# ─────────────────────────────────────────────────

DRIVER_SEASON_RESULTS = [
    # 2014 — First turbo-hybrid era
    {"year": 2014, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 384, "wins": 11, "podiums": 16, "poles": 7, "races": 19},
    {"year": 2014, "driver": "Nico Rosberg", "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 317, "wins": 5, "podiums": 15, "poles": 11, "races": 19},
    {"year": 2014, "driver": "Daniel Ricciardo", "team": "Red Bull", "engine": "Renault", "position": 3, "points": 238, "wins": 3, "podiums": 8, "poles": 1, "races": 19},
    {"year": 2014, "driver": "Valtteri Bottas", "team": "Williams", "engine": "Mercedes", "position": 4, "points": 186, "wins": 0, "podiums": 6, "poles": 0, "races": 19},
    {"year": 2014, "driver": "Sebastian Vettel", "team": "Red Bull", "engine": "Renault", "position": 5, "points": 167, "wins": 0, "podiums": 4, "poles": 0, "races": 19},
    {"year": 2014, "driver": "Fernando Alonso", "team": "Ferrari", "engine": "Ferrari", "position": 6, "points": 161, "wins": 0, "podiums": 2, "poles": 0, "races": 19},
    {"year": 2014, "driver": "Felipe Massa", "team": "Williams", "engine": "Mercedes", "position": 7, "points": 134, "wins": 0, "podiums": 5, "poles": 1, "races": 19},
    {"year": 2014, "driver": "Jenson Button", "team": "McLaren", "engine": "Mercedes", "position": 8, "points": 126, "wins": 0, "podiums": 1, "poles": 0, "races": 19},
    {"year": 2014, "driver": "Nico Hulkenberg", "team": "Force India", "engine": "Mercedes", "position": 9, "points": 96, "wins": 0, "podiums": 0, "poles": 0, "races": 19},
    {"year": 2014, "driver": "Kevin Magnussen", "team": "McLaren", "engine": "Mercedes", "position": 11, "points": 55, "wins": 0, "podiums": 1, "poles": 0, "races": 19},

    # 2015
    {"year": 2015, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 381, "wins": 10, "podiums": 17, "poles": 11, "races": 19},
    {"year": 2015, "driver": "Nico Rosberg", "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 322, "wins": 6, "podiums": 15, "poles": 7, "races": 19},
    {"year": 2015, "driver": "Sebastian Vettel", "team": "Ferrari", "engine": "Ferrari", "position": 3, "points": 278, "wins": 3, "podiums": 13, "poles": 1, "races": 19},
    {"year": 2015, "driver": "Kimi Raikkonen", "team": "Ferrari", "engine": "Ferrari", "position": 4, "points": 150, "wins": 0, "podiums": 3, "poles": 0, "races": 19},
    {"year": 2015, "driver": "Valtteri Bottas", "team": "Williams", "engine": "Mercedes", "position": 5, "points": 136, "wins": 0, "podiums": 2, "poles": 0, "races": 19},
    {"year": 2015, "driver": "Felipe Massa", "team": "Williams", "engine": "Mercedes", "position": 6, "points": 121, "wins": 0, "podiums": 2, "poles": 0, "races": 19},
    {"year": 2015, "driver": "Daniel Ricciardo", "team": "Red Bull", "engine": "Renault", "position": 8, "points": 92, "wins": 0, "podiums": 2, "poles": 0, "races": 19},
    {"year": 2015, "driver": "Max Verstappen", "team": "Toro Rosso", "engine": "Renault", "position": 12, "points": 49, "wins": 0, "podiums": 0, "poles": 0, "races": 19},
    {"year": 2015, "driver": "Carlos Sainz", "team": "Toro Rosso", "engine": "Renault", "position": 15, "points": 18, "wins": 0, "podiums": 0, "poles": 0, "races": 19},

    # 2016
    {"year": 2016, "driver": "Nico Rosberg", "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 385, "wins": 9, "podiums": 16, "poles": 8, "races": 21},
    {"year": 2016, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 380, "wins": 10, "podiums": 17, "poles": 12, "races": 21},
    {"year": 2016, "driver": "Daniel Ricciardo", "team": "Red Bull", "engine": "TAG Heuer", "position": 3, "points": 256, "wins": 1, "podiums": 8, "poles": 1, "races": 21},
    {"year": 2016, "driver": "Max Verstappen", "team": "Red Bull", "engine": "TAG Heuer", "position": 5, "points": 204, "wins": 1, "podiums": 7, "poles": 0, "races": 21},
    {"year": 2016, "driver": "Sebastian Vettel", "team": "Ferrari", "engine": "Ferrari", "position": 4, "points": 212, "wins": 0, "podiums": 7, "poles": 1, "races": 21},
    {"year": 2016, "driver": "Kimi Raikkonen", "team": "Ferrari", "engine": "Ferrari", "position": 6, "points": 186, "wins": 0, "podiums": 4, "poles": 0, "races": 21},
    {"year": 2016, "driver": "Valtteri Bottas", "team": "Williams", "engine": "Mercedes", "position": 8, "points": 85, "wins": 0, "podiums": 1, "poles": 0, "races": 21},
    {"year": 2016, "driver": "Carlos Sainz", "team": "Toro Rosso", "engine": "Ferrari", "position": 12, "points": 46, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2016, "driver": "Fernando Alonso", "team": "McLaren", "engine": "Honda", "position": 10, "points": 54, "wins": 0, "podiums": 0, "poles": 0, "races": 21},

    # 2017 — Wider cars, more downforce
    {"year": 2017, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 363, "wins": 9, "podiums": 13, "poles": 11, "races": 20},
    {"year": 2017, "driver": "Sebastian Vettel", "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 317, "wins": 5, "podiums": 13, "poles": 4, "races": 20},
    {"year": 2017, "driver": "Valtteri Bottas", "team": "Mercedes", "engine": "Mercedes", "position": 3, "points": 305, "wins": 3, "podiums": 13, "poles": 4, "races": 20},
    {"year": 2017, "driver": "Daniel Ricciardo", "team": "Red Bull", "engine": "TAG Heuer", "position": 5, "points": 200, "wins": 1, "podiums": 9, "poles": 0, "races": 20},
    {"year": 2017, "driver": "Max Verstappen", "team": "Red Bull", "engine": "TAG Heuer", "position": 6, "points": 168, "wins": 2, "podiums": 4, "poles": 0, "races": 20},
    {"year": 2017, "driver": "Kimi Raikkonen", "team": "Ferrari", "engine": "Ferrari", "position": 4, "points": 205, "wins": 0, "podiums": 7, "poles": 1, "races": 20},
    {"year": 2017, "driver": "Carlos Sainz", "team": "Toro Rosso", "engine": "Renault", "position": 9, "points": 54, "wins": 0, "podiums": 0, "poles": 0, "races": 20},
    {"year": 2017, "driver": "Esteban Ocon", "team": "Force India", "engine": "Mercedes", "position": 8, "points": 87, "wins": 0, "podiums": 0, "poles": 0, "races": 20},
    {"year": 2017, "driver": "Fernando Alonso", "team": "McLaren", "engine": "Honda", "position": 15, "points": 17, "wins": 0, "podiums": 0, "poles": 0, "races": 20},
    {"year": 2017, "driver": "Lance Stroll", "team": "Williams", "engine": "Mercedes", "position": 12, "points": 40, "wins": 0, "podiums": 1, "poles": 0, "races": 20},

    # 2018
    {"year": 2018, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 408, "wins": 11, "podiums": 17, "poles": 11, "races": 21},
    {"year": 2018, "driver": "Sebastian Vettel", "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 320, "wins": 5, "podiums": 12, "poles": 5, "races": 21},
    {"year": 2018, "driver": "Kimi Raikkonen", "team": "Ferrari", "engine": "Ferrari", "position": 3, "points": 251, "wins": 1, "podiums": 12, "poles": 1, "races": 21},
    {"year": 2018, "driver": "Max Verstappen", "team": "Red Bull", "engine": "TAG Heuer", "position": 4, "points": 249, "wins": 2, "podiums": 11, "poles": 2, "races": 21},
    {"year": 2018, "driver": "Valtteri Bottas", "team": "Mercedes", "engine": "Mercedes", "position": 5, "points": 247, "wins": 0, "podiums": 8, "poles": 4, "races": 21},
    {"year": 2018, "driver": "Daniel Ricciardo", "team": "Red Bull", "engine": "TAG Heuer", "position": 6, "points": 170, "wins": 2, "podiums": 6, "poles": 2, "races": 21},
    {"year": 2018, "driver": "Carlos Sainz", "team": "Renault", "engine": "Renault", "position": 10, "points": 53, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2018, "driver": "Charles Leclerc", "team": "Sauber", "engine": "Ferrari", "position": 13, "points": 39, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2018, "driver": "Pierre Gasly", "team": "Toro Rosso", "engine": "Honda", "position": 15, "points": 29, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2018, "driver": "Fernando Alonso", "team": "McLaren", "engine": "Renault", "position": 11, "points": 50, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2018, "driver": "Esteban Ocon", "team": "Force India", "engine": "Mercedes", "position": 12, "points": 49, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2018, "driver": "Lance Stroll", "team": "Williams", "engine": "Mercedes", "position": 18, "points": 6, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2018, "driver": "Lando Norris", "team": None, "engine": None, "position": None, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 0},

    # 2019
    {"year": 2019, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 413, "wins": 11, "podiums": 17, "poles": 5, "races": 21},
    {"year": 2019, "driver": "Valtteri Bottas", "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 326, "wins": 4, "podiums": 15, "poles": 5, "races": 21},
    {"year": 2019, "driver": "Max Verstappen", "team": "Red Bull", "engine": "Honda", "position": 3, "points": 278, "wins": 3, "podiums": 9, "poles": 2, "races": 21},
    {"year": 2019, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 4, "points": 264, "wins": 2, "podiums": 10, "poles": 7, "races": 21},
    {"year": 2019, "driver": "Sebastian Vettel", "team": "Ferrari", "engine": "Ferrari", "position": 5, "points": 240, "wins": 1, "podiums": 9, "poles": 2, "races": 21},
    {"year": 2019, "driver": "Carlos Sainz", "team": "McLaren", "engine": "Renault", "position": 6, "points": 96, "wins": 0, "podiums": 1, "poles": 0, "races": 21},
    {"year": 2019, "driver": "Pierre Gasly", "team": "Red Bull", "engine": "Honda", "position": 7, "points": 95, "wins": 0, "podiums": 2, "poles": 0, "races": 21},
    {"year": 2019, "driver": "Alex Albon", "team": "Red Bull", "engine": "Honda", "position": 8, "points": 92, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2019, "driver": "Lando Norris", "team": "McLaren", "engine": "Renault", "position": 11, "points": 49, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2019, "driver": "George Russell", "team": "Williams", "engine": "Mercedes", "position": 20, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2019, "driver": "Lance Stroll", "team": "Racing Point", "engine": "Mercedes", "position": 15, "points": 21, "wins": 0, "podiums": 0, "poles": 0, "races": 21},
    {"year": 2019, "driver": "Nico Hulkenberg", "team": "Renault", "engine": "Renault", "position": 14, "points": 37, "wins": 0, "podiums": 0, "poles": 0, "races": 21},

    # 2020 (17 races, COVID season)
    {"year": 2020, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 347, "wins": 11, "podiums": 14, "poles": 10, "races": 17},
    {"year": 2020, "driver": "Valtteri Bottas", "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 223, "wins": 2, "podiums": 11, "poles": 5, "races": 17},
    {"year": 2020, "driver": "Max Verstappen", "team": "Red Bull", "engine": "Honda", "position": 3, "points": 214, "wins": 2, "podiums": 11, "poles": 1, "races": 17},
    {"year": 2020, "driver": "Sergio Perez", "team": "Racing Point", "engine": "Mercedes", "position": 4, "points": 125, "wins": 1, "podiums": 3, "poles": 0, "races": 17},
    {"year": 2020, "driver": "Daniel Ricciardo", "team": "Renault", "engine": "Renault", "position": 5, "points": 119, "wins": 0, "podiums": 2, "poles": 0, "races": 17},
    {"year": 2020, "driver": "Carlos Sainz", "team": "McLaren", "engine": "Renault", "position": 6, "points": 105, "wins": 0, "podiums": 2, "poles": 0, "races": 17},
    {"year": 2020, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 8, "points": 98, "wins": 0, "podiums": 2, "poles": 2, "races": 17},
    {"year": 2020, "driver": "Lando Norris", "team": "McLaren", "engine": "Renault", "position": 9, "points": 97, "wins": 0, "podiums": 1, "poles": 0, "races": 17},
    {"year": 2020, "driver": "Alex Albon", "team": "Red Bull", "engine": "Honda", "position": 7, "points": 105, "wins": 0, "podiums": 2, "poles": 0, "races": 17},
    {"year": 2020, "driver": "Pierre Gasly", "team": "AlphaTauri", "engine": "Honda", "position": 10, "points": 75, "wins": 1, "podiums": 1, "poles": 0, "races": 17},
    {"year": 2020, "driver": "Lance Stroll", "team": "Racing Point", "engine": "Mercedes", "position": 11, "points": 75, "wins": 0, "podiums": 2, "poles": 1, "races": 17},
    {"year": 2020, "driver": "Esteban Ocon", "team": "Renault", "engine": "Renault", "position": 12, "points": 62, "wins": 0, "podiums": 1, "poles": 0, "races": 17},
    {"year": 2020, "driver": "George Russell", "team": "Williams", "engine": "Mercedes", "position": 18, "points": 3, "wins": 0, "podiums": 0, "poles": 0, "races": 17},
    {"year": 2020, "driver": "Sebastian Vettel", "team": "Ferrari", "engine": "Ferrari", "position": 13, "points": 33, "wins": 0, "podiums": 1, "poles": 0, "races": 17},

    # 2021 — epic Verstappen vs Hamilton
    {"year": 2021, "driver": "Max Verstappen", "team": "Red Bull", "engine": "Honda", "position": 1, "points": 395.5, "wins": 10, "podiums": 18, "poles": 10, "races": 22},
    {"year": 2021, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 387.5, "wins": 8, "podiums": 17, "poles": 5, "races": 22},
    {"year": 2021, "driver": "Valtteri Bottas", "team": "Mercedes", "engine": "Mercedes", "position": 3, "points": 226, "wins": 1, "podiums": 11, "poles": 5, "races": 22},
    {"year": 2021, "driver": "Sergio Perez", "team": "Red Bull", "engine": "Honda", "position": 4, "points": 190, "wins": 1, "podiums": 5, "poles": 0, "races": 22},
    {"year": 2021, "driver": "Carlos Sainz", "team": "Ferrari", "engine": "Ferrari", "position": 5, "points": 164.5, "wins": 0, "podiums": 4, "poles": 0, "races": 22},
    {"year": 2021, "driver": "Lando Norris", "team": "McLaren", "engine": "Mercedes", "position": 6, "points": 160, "wins": 0, "podiums": 4, "poles": 1, "races": 22},
    {"year": 2021, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 7, "points": 159, "wins": 0, "podiums": 4, "poles": 2, "races": 22},
    {"year": 2021, "driver": "Daniel Ricciardo", "team": "McLaren", "engine": "Mercedes", "position": 8, "points": 115, "wins": 1, "podiums": 1, "poles": 0, "races": 22},
    {"year": 2021, "driver": "Pierre Gasly", "team": "AlphaTauri", "engine": "Honda", "position": 9, "points": 110, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2021, "driver": "Fernando Alonso", "team": "Alpine", "engine": "Renault", "position": 10, "points": 81, "wins": 0, "podiums": 1, "poles": 0, "races": 22},
    {"year": 2021, "driver": "Esteban Ocon", "team": "Alpine", "engine": "Renault", "position": 11, "points": 74, "wins": 1, "podiums": 1, "poles": 0, "races": 22},
    {"year": 2021, "driver": "George Russell", "team": "Williams", "engine": "Mercedes", "position": 15, "points": 16, "wins": 0, "podiums": 1, "poles": 0, "races": 22},
    {"year": 2021, "driver": "Lance Stroll", "team": "Aston Martin", "engine": "Mercedes", "position": 13, "points": 34, "wins": 0, "podiums": 0, "poles": 0, "races": 22},

    # 2022 — Ground effect era
    {"year": 2022, "driver": "Max Verstappen", "team": "Red Bull", "engine": "RBPT", "position": 1, "points": 454, "wins": 15, "podiums": 17, "poles": 7, "races": 22},
    {"year": 2022, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 308, "wins": 3, "podiums": 11, "poles": 9, "races": 22},
    {"year": 2022, "driver": "Sergio Perez", "team": "Red Bull", "engine": "RBPT", "position": 3, "points": 305, "wins": 2, "podiums": 11, "poles": 1, "races": 22},
    {"year": 2022, "driver": "Carlos Sainz", "team": "Ferrari", "engine": "Ferrari", "position": 5, "points": 246, "wins": 1, "podiums": 9, "poles": 3, "races": 22},
    {"year": 2022, "driver": "George Russell", "team": "Mercedes", "engine": "Mercedes", "position": 4, "points": 275, "wins": 1, "podiums": 8, "poles": 1, "races": 22},
    {"year": 2022, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 6, "points": 240, "wins": 0, "podiums": 9, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Lando Norris", "team": "McLaren", "engine": "Mercedes", "position": 7, "points": 122, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Esteban Ocon", "team": "Alpine", "engine": "Renault", "position": 8, "points": 92, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Fernando Alonso", "team": "Alpine", "engine": "Renault", "position": 9, "points": 81, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Valtteri Bottas", "team": "Alfa Romeo", "engine": "Ferrari", "position": 10, "points": 49, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Alex Albon", "team": "Williams", "engine": "Mercedes", "position": 19, "points": 4, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Oscar Piastri", "team": None, "engine": None, "position": None, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 0},
    {"year": 2022, "driver": "Pierre Gasly", "team": "AlphaTauri", "engine": "RBPT", "position": 14, "points": 23, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Lance Stroll", "team": "Aston Martin", "engine": "Mercedes", "position": 15, "points": 18, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2022, "driver": "Nico Hulkenberg", "team": None, "engine": None, "position": None, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 0},

    # 2023
    {"year": 2023, "driver": "Max Verstappen", "team": "Red Bull", "engine": "Honda RBPT", "position": 1, "points": 575, "wins": 19, "podiums": 21, "poles": 12, "races": 22},
    {"year": 2023, "driver": "Sergio Perez", "team": "Red Bull", "engine": "Honda RBPT", "position": 2, "points": 285, "wins": 2, "podiums": 8, "poles": 2, "races": 22},
    {"year": 2023, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 3, "points": 234, "wins": 0, "podiums": 8, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Fernando Alonso", "team": "Aston Martin", "engine": "Mercedes", "position": 4, "points": 206, "wins": 0, "podiums": 8, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 5, "points": 206, "wins": 0, "podiums": 5, "poles": 3, "races": 22},
    {"year": 2023, "driver": "Lando Norris", "team": "McLaren", "engine": "Mercedes", "position": 6, "points": 205, "wins": 0, "podiums": 7, "poles": 1, "races": 22},
    {"year": 2023, "driver": "Carlos Sainz", "team": "Ferrari", "engine": "Ferrari", "position": 7, "points": 200, "wins": 1, "podiums": 5, "poles": 3, "races": 22},
    {"year": 2023, "driver": "George Russell", "team": "Mercedes", "engine": "Mercedes", "position": 8, "points": 175, "wins": 0, "podiums": 2, "poles": 1, "races": 22},
    {"year": 2023, "driver": "Oscar Piastri", "team": "McLaren", "engine": "Mercedes", "position": 9, "points": 97, "wins": 1, "podiums": 2, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Pierre Gasly", "team": "Alpine", "engine": "Renault", "position": 11, "points": 62, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Alex Albon", "team": "Williams", "engine": "Mercedes", "position": 12, "points": 27, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Esteban Ocon", "team": "Alpine", "engine": "Renault", "position": 12, "points": 58, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Lance Stroll", "team": "Aston Martin", "engine": "Mercedes", "position": 10, "points": 74, "wins": 0, "podiums": 1, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Nico Hulkenberg", "team": "Haas", "engine": "Ferrari", "position": 14, "points": 9, "wins": 0, "podiums": 0, "poles": 0, "races": 22},
    {"year": 2023, "driver": "Valtteri Bottas", "team": "Alfa Romeo", "engine": "Ferrari", "position": 16, "points": 10, "wins": 0, "podiums": 0, "poles": 0, "races": 22},

    # 2024
    {"year": 2024, "driver": "Max Verstappen", "team": "Red Bull", "engine": "Honda RBPT", "position": 1, "points": 437, "wins": 9, "podiums": 14, "poles": 10, "races": 24},
    {"year": 2024, "driver": "Lando Norris", "team": "McLaren", "engine": "Mercedes", "position": 2, "points": 374, "wins": 4, "podiums": 15, "poles": 8, "races": 24},
    {"year": 2024, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 3, "points": 356, "wins": 3, "podiums": 14, "poles": 4, "races": 24},
    {"year": 2024, "driver": "Oscar Piastri", "team": "McLaren", "engine": "Mercedes", "position": 4, "points": 292, "wins": 2, "podiums": 9, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Carlos Sainz", "team": "Ferrari", "engine": "Ferrari", "position": 5, "points": 290, "wins": 2, "podiums": 10, "poles": 2, "races": 24},
    {"year": 2024, "driver": "George Russell", "team": "Mercedes", "engine": "Mercedes", "position": 6, "points": 245, "wins": 2, "podiums": 7, "poles": 2, "races": 24},
    {"year": 2024, "driver": "Lewis Hamilton", "team": "Mercedes", "engine": "Mercedes", "position": 7, "points": 223, "wins": 2, "podiums": 5, "poles": 1, "races": 24},
    {"year": 2024, "driver": "Sergio Perez", "team": "Red Bull", "engine": "Honda RBPT", "position": 8, "points": 152, "wins": 0, "podiums": 4, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Fernando Alonso", "team": "Aston Martin", "engine": "Mercedes", "position": 9, "points": 70, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Pierre Gasly", "team": "Alpine", "engine": "Renault", "position": 10, "points": 42, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Nico Hulkenberg", "team": "Haas", "engine": "Ferrari", "position": 11, "points": 41, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Alex Albon", "team": "Williams", "engine": "Mercedes", "position": 14, "points": 12, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Esteban Ocon", "team": "Alpine", "engine": "Renault", "position": 14, "points": 23, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Lance Stroll", "team": "Aston Martin", "engine": "Mercedes", "position": 13, "points": 24, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Valtteri Bottas", "team": "Sauber", "engine": "Ferrari", "position": 22, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2024, "driver": "Oliver Bearman", "team": "Haas", "engine": "Ferrari", "position": None, "points": 7, "wins": 0, "podiums": 0, "poles": 0, "races": 3},

    # 2025 — Norris WDC, McLaren WCC
    {"year": 2025, "driver": "Lando Norris", "team": "McLaren", "engine": "Mercedes", "position": 1, "points": 423, "wins": 7, "podiums": 18, "poles": 6, "races": 24},
    {"year": 2025, "driver": "Max Verstappen", "team": "Red Bull", "engine": "Honda RBPT", "position": 2, "points": 421, "wins": 8, "podiums": 16, "poles": 8, "races": 24},
    {"year": 2025, "driver": "Oscar Piastri", "team": "McLaren", "engine": "Mercedes", "position": 3, "points": 410, "wins": 7, "podiums": 17, "poles": 5, "races": 24},
    {"year": 2025, "driver": "George Russell", "team": "Mercedes", "engine": "Mercedes", "position": 4, "points": 319, "wins": 2, "podiums": 10, "poles": 3, "races": 24},
    {"year": 2025, "driver": "Charles Leclerc", "team": "Ferrari", "engine": "Ferrari", "position": 5, "points": 242, "wins": 0, "podiums": 6, "poles": 2, "races": 24},
    {"year": 2025, "driver": "Lewis Hamilton", "team": "Ferrari", "engine": "Ferrari", "position": 6, "points": 156, "wins": 0, "podiums": 3, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Kimi Antonelli", "team": "Mercedes", "engine": "Mercedes", "position": 7, "points": 150, "wins": 0, "podiums": 2, "poles": 1, "races": 24},
    {"year": 2025, "driver": "Alex Albon", "team": "Williams", "engine": "Mercedes", "position": 8, "points": 73, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Carlos Sainz", "team": "Williams", "engine": "Mercedes", "position": 9, "points": 64, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Fernando Alonso", "team": "Aston Martin", "engine": "Mercedes", "position": 10, "points": 56, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Isack Hadjar", "team": "Racing Bulls", "engine": "Honda RBPT", "position": 11, "points": 51, "wins": 0, "podiums": 1, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Nico Hulkenberg", "team": "Sauber", "engine": "Ferrari", "position": 12, "points": 49, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Oliver Bearman", "team": "Haas", "engine": "Ferrari", "position": 13, "points": 42, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Liam Lawson", "team": "Racing Bulls", "engine": "Honda RBPT", "position": 14, "points": 38, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Esteban Ocon", "team": "Haas", "engine": "Ferrari", "position": 15, "points": 38, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Lance Stroll", "team": "Aston Martin", "engine": "Mercedes", "position": 16, "points": 34, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Pierre Gasly", "team": "Alpine", "engine": "Renault", "position": 18, "points": 22, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Gabriel Bortoleto", "team": "Sauber", "engine": "Ferrari", "position": 19, "points": 19, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Franco Colapinto", "team": "Alpine", "engine": "Renault", "position": 20, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 24},
    {"year": 2025, "driver": "Sergio Perez", "team": None, "engine": None, "position": None, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 0},
    {"year": 2025, "driver": "Valtteri Bottas", "team": None, "engine": None, "position": None, "points": 0, "wins": 0, "podiums": 0, "poles": 0, "races": 0},
]

# ─────────────────────────────────────────────────
# CONSTRUCTOR SEASON RESULTS
# ─────────────────────────────────────────────────

CONSTRUCTOR_SEASON_RESULTS = [
    # 2014
    {"year": 2014, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 701, "wins": 16},
    {"year": 2014, "team": "Red Bull", "engine": "Renault", "position": 2, "points": 405, "wins": 3},
    {"year": 2014, "team": "Williams", "engine": "Mercedes", "position": 3, "points": 320, "wins": 0},
    {"year": 2014, "team": "Ferrari", "engine": "Ferrari", "position": 4, "points": 216, "wins": 0},
    {"year": 2014, "team": "McLaren", "engine": "Mercedes", "position": 5, "points": 181, "wins": 0},
    # 2015
    {"year": 2015, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 703, "wins": 16},
    {"year": 2015, "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 428, "wins": 3},
    {"year": 2015, "team": "Williams", "engine": "Mercedes", "position": 3, "points": 257, "wins": 0},
    {"year": 2015, "team": "Red Bull", "engine": "Renault", "position": 4, "points": 187, "wins": 0},
    {"year": 2015, "team": "McLaren", "engine": "Honda", "position": 9, "points": 27, "wins": 0},
    # 2016
    {"year": 2016, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 765, "wins": 19},
    {"year": 2016, "team": "Red Bull", "engine": "TAG Heuer", "position": 2, "points": 468, "wins": 2},
    {"year": 2016, "team": "Ferrari", "engine": "Ferrari", "position": 3, "points": 398, "wins": 0},
    {"year": 2016, "team": "Williams", "engine": "Mercedes", "position": 5, "points": 138, "wins": 0},
    {"year": 2016, "team": "McLaren", "engine": "Honda", "position": 6, "points": 76, "wins": 0},
    # 2017
    {"year": 2017, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 668, "wins": 12},
    {"year": 2017, "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 522, "wins": 5},
    {"year": 2017, "team": "Red Bull", "engine": "TAG Heuer", "position": 3, "points": 368, "wins": 3},
    {"year": 2017, "team": "Williams", "engine": "Mercedes", "position": 5, "points": 83, "wins": 0},
    {"year": 2017, "team": "McLaren", "engine": "Honda", "position": 9, "points": 30, "wins": 0},
    # 2018
    {"year": 2018, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 655, "wins": 11},
    {"year": 2018, "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 571, "wins": 6},
    {"year": 2018, "team": "Red Bull", "engine": "TAG Heuer", "position": 3, "points": 419, "wins": 4},
    {"year": 2018, "team": "McLaren", "engine": "Renault", "position": 6, "points": 62, "wins": 0},
    {"year": 2018, "team": "Williams", "engine": "Mercedes", "position": 10, "points": 7, "wins": 0},
    # 2019
    {"year": 2019, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 739, "wins": 15},
    {"year": 2019, "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 504, "wins": 3},
    {"year": 2019, "team": "Red Bull", "engine": "Honda", "position": 3, "points": 417, "wins": 3},
    {"year": 2019, "team": "McLaren", "engine": "Renault", "position": 4, "points": 145, "wins": 0},
    {"year": 2019, "team": "Williams", "engine": "Mercedes", "position": 10, "points": 1, "wins": 0},
    # 2020
    {"year": 2020, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 573, "wins": 13},
    {"year": 2020, "team": "Red Bull", "engine": "Honda", "position": 2, "points": 319, "wins": 2},
    {"year": 2020, "team": "McLaren", "engine": "Renault", "position": 3, "points": 202, "wins": 0},
    {"year": 2020, "team": "Ferrari", "engine": "Ferrari", "position": 6, "points": 131, "wins": 0},
    {"year": 2020, "team": "Williams", "engine": "Mercedes", "position": 10, "points": 0, "wins": 0},
    # 2021
    {"year": 2021, "team": "Mercedes", "engine": "Mercedes", "position": 1, "points": 613.5, "wins": 9},
    {"year": 2021, "team": "Red Bull", "engine": "Honda", "position": 2, "points": 585.5, "wins": 11},
    {"year": 2021, "team": "Ferrari", "engine": "Ferrari", "position": 3, "points": 323.5, "wins": 0},
    {"year": 2021, "team": "McLaren", "engine": "Mercedes", "position": 4, "points": 275, "wins": 1},
    {"year": 2021, "team": "Aston Martin", "engine": "Mercedes", "position": 7, "points": 77, "wins": 0},
    {"year": 2021, "team": "Williams", "engine": "Mercedes", "position": 8, "points": 23, "wins": 0},
    # 2022
    {"year": 2022, "team": "Red Bull", "engine": "RBPT", "position": 1, "points": 759, "wins": 17},
    {"year": 2022, "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 554, "wins": 4},
    {"year": 2022, "team": "Mercedes", "engine": "Mercedes", "position": 3, "points": 515, "wins": 1},
    {"year": 2022, "team": "McLaren", "engine": "Mercedes", "position": 5, "points": 159, "wins": 0},
    {"year": 2022, "team": "Aston Martin", "engine": "Mercedes", "position": 7, "points": 55, "wins": 0},
    {"year": 2022, "team": "Williams", "engine": "Mercedes", "position": 10, "points": 8, "wins": 0},
    # 2023
    {"year": 2023, "team": "Red Bull", "engine": "Honda RBPT", "position": 1, "points": 860, "wins": 21},
    {"year": 2023, "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 409, "wins": 0},
    {"year": 2023, "team": "Ferrari", "engine": "Ferrari", "position": 3, "points": 406, "wins": 1},
    {"year": 2023, "team": "McLaren", "engine": "Mercedes", "position": 4, "points": 302, "wins": 1},
    {"year": 2023, "team": "Aston Martin", "engine": "Mercedes", "position": 5, "points": 280, "wins": 0},
    {"year": 2023, "team": "Williams", "engine": "Mercedes", "position": 7, "points": 28, "wins": 0},
    # 2024
    {"year": 2024, "team": "McLaren", "engine": "Mercedes", "position": 1, "points": 666, "wins": 6},
    {"year": 2024, "team": "Ferrari", "engine": "Ferrari", "position": 2, "points": 652, "wins": 5},
    {"year": 2024, "team": "Red Bull", "engine": "Honda RBPT", "position": 3, "points": 589, "wins": 9},
    {"year": 2024, "team": "Mercedes", "engine": "Mercedes", "position": 4, "points": 468, "wins": 4},
    {"year": 2024, "team": "Aston Martin", "engine": "Mercedes", "position": 5, "points": 94, "wins": 0},
    {"year": 2024, "team": "Williams", "engine": "Mercedes", "position": 9, "points": 17, "wins": 0},
    # 2025
    {"year": 2025, "team": "McLaren", "engine": "Mercedes", "position": 1, "points": 833, "wins": 14},
    {"year": 2025, "team": "Mercedes", "engine": "Mercedes", "position": 2, "points": 469, "wins": 2},
    {"year": 2025, "team": "Red Bull", "engine": "Honda RBPT", "position": 3, "points": 451, "wins": 8},
    {"year": 2025, "team": "Ferrari", "engine": "Ferrari", "position": 4, "points": 398, "wins": 0},
    {"year": 2025, "team": "Williams", "engine": "Mercedes", "position": 5, "points": 137, "wins": 0},
    {"year": 2025, "team": "Racing Bulls", "engine": "Honda RBPT", "position": 6, "points": 92, "wins": 0},
    {"year": 2025, "team": "Aston Martin", "engine": "Mercedes", "position": 7, "points": 90, "wins": 0},
    {"year": 2025, "team": "Haas", "engine": "Ferrari", "position": 8, "points": 80, "wins": 0},
    {"year": 2025, "team": "Sauber", "engine": "Ferrari", "position": 9, "points": 68, "wins": 0},
    {"year": 2025, "team": "Alpine", "engine": "Renault", "position": 10, "points": 22, "wins": 0},
]


def get_driver_df():
    """Return driver season results as a DataFrame with qualifying data."""
    from data.qualifying_data import QUALIFYING_DATA
    
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
    return df


def get_constructor_df():
    """Return constructor season results as a DataFrame."""
    df = pd.DataFrame(CONSTRUCTOR_SEASON_RESULTS)
    return df


# ─────────────────────────────────────────────────
# 2026 GRID & PRE-SEASON DATA
# ─────────────────────────────────────────────────

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

# Pre-season testing rankings (from Bahrain Test 1, Feb 2026)
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

# Bookmaker odds (converted to implied probabilities, from late Jan 2026)
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
