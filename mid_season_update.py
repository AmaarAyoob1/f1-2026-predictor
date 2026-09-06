"""
F1 2026 Mid-Season Update — After Round 12 (Dutch GP, Zandvoort)

Season state: 12 of 23 rounds complete. 11 GPs + 1 sprint remaining (283 pts available).

Bayesian blending approach:
1. Lock in actual points from Rounds 1-12 (official standings)
2. Derive per-driver performance metrics from real finishing positions
3. Blend pre-season strength with actual performance (alpha decays as season progresses)
4. Recompute team reliability from real DNF/DNS counts
5. Monte Carlo the remaining 11 rounds
6. Final projection = actual points + simulated remaining points

Two blend weights produce two model views:
  Monte Carlo (alpha=0.20) still gives pre-season pedigree some weight
  Bayesian    (alpha=0.05) trusts season-to-date data almost entirely
  Ensemble    averages the two
"""

import numpy as np
import pandas as pd
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.train_models import MonteCarloSimulator

ROUNDS_COMPLETE = 12
ROUNDS_TOTAL = 23
ROUNDS_REMAINING = 11

RACE_ORDER = ["AUS", "CHN", "JPN", "MIA", "CAN", "MCO",
              "BAR", "AUT", "GBR", "BEL", "HUN", "NLD"]

# Official points after Round 12 + per-race finishing positions.
# "D" = DNF, "N" = DNS, "-" = did not participate.
SEASON_TO_DATE = {
    "Kimi Antonelli":    {"team": "Mercedes",     "pts": 242, "pos": [2, 1, 1, 1, 1, 1, 16, 3, 15, 1, 3, 2]},
    "George Russell":    {"team": "Mercedes",     "pts": 183, "pos": [1, 2, 4, 4, "D", 12, 2, 1, 2, "D", 7, 3]},
    "Lewis Hamilton":    {"team": "Ferrari",      "pts": 183, "pos": [4, 3, 6, 6, 2, 2, 1, 5, 3, 4, 5, 4]},
    "Lando Norris":      {"team": "McLaren",      "pts": 159, "pos": [5, "N", 5, 2, "D", "D", 3, 7, 4, 7, 1, 1]},
    "Charles Leclerc":   {"team": "Ferrari",      "pts": 155, "pos": [3, 4, 3, 8, 4, "D", 15, 8, 1, 2, 4, 5]},
    "Max Verstappen":    {"team": "Red Bull",     "pts": 112, "pos": [6, "D", 8, 5, 3, "D", 4, 2, 20, 3, 2, "D"]},
    "Oscar Piastri":     {"team": "McLaren",      "pts": 106, "pos": ["N", "N", 2, 3, 11, 4, 5, 4, 11, 5, "D", 6]},
    "Isack Hadjar":      {"team": "Red Bull",     "pts": 71,  "pos": ["D", 8, 12, "D", 5, 3, 6, 6, 5, 6, 6, "-"]},
    "Liam Lawson":       {"team": "Racing Bulls", "pts": 51,  "pos": [13, 7, 9, "D", 7, 5, 8, 9, 6, 12, 8, 7]},
    "Pierre Gasly":      {"team": "Alpine",       "pts": 35,  "pos": [10, 6, 7, "D", 8, 7, 7, 13, 10, 11, 12, 10]},
    "Arvid Lindblad":    {"team": "Racing Bulls", "pts": 25,  "pos": [8, 12, 14, 14, "N", 6, 9, 10, 7, 9, 10, 12]},
    "Franco Colapinto":  {"team": "Alpine",       "pts": 19,  "pos": [14, 10, 16, 7, 6, 14, 10, 15, 9, 10, 15, 14]},
    "Oliver Bearman":    {"team": "Haas",         "pts": 18,  "pos": [7, 5, "D", 11, 10, "D", 17, 14, 12, 14, 19, "D"]},
    "Gabriel Bortoleto": {"team": "Audi",         "pts": 10,  "pos": [9, "N", 13, 12, 13, 11, 11, 11, 8, 8, 11, 13]},
    "Nico Hulkenberg":   {"team": "Audi",         "pts": 6,   "pos": ["N", 11, 11, "D", 12, 13, "D", 12, "D", 13, 9, 8]},
    "Carlos Sainz":      {"team": "Williams",     "pts": 6,   "pos": [15, 9, 15, 9, 9, 16, 12, "D", 17, 16, 18, 16]},
    "Alex Albon":        {"team": "Williams",     "pts": 5,   "pos": [12, "N", 20, 10, "D", 8, "D", 17, "D", 15, 17, 17]},
    "Esteban Ocon":      {"team": "Haas",         "pts": 3,   "pos": [11, 14, 10, 13, 14, 9, 13, 16, 13, 17, 16, "D"]},
    "Fernando Alonso":   {"team": "Aston Martin", "pts": 3,   "pos": ["D", "D", 18, 15, "D", 10, "D", 18, 18, 19, 14, 9]},
    "Yuki Tsunoda":      {"team": "Racing Bulls", "pts": 0,   "pos": ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", 11]},
    "Lance Stroll":      {"team": "Aston Martin", "pts": 0,   "pos": ["D", "D", "D", 17, 15, "D", "D", "D", 19, "D", 13, "D"]},
    "Valtteri Bottas":   {"team": "Cadillac",     "pts": 0,   "pos": ["D", 13, 19, 18, 16, "D", "D", "D", 16, 18, "D", "D"]},
}

DRIVER_TEAMS = {d: v["team"] for d, v in SEASON_TO_DATE.items()}
ALL_DRIVERS = list(SEASON_TO_DATE.keys())

PRESEASON_STRENGTHS = {
    "Max Verstappen": 195.2, "Lando Norris": 176.8, "George Russell": 172.4,
    "Charles Leclerc": 170.1, "Oscar Piastri": 163.5, "Lewis Hamilton": 168.7,
    "Carlos Sainz": 148.2, "Kimi Antonelli": 141.4, "Isack Hadjar": 130.5,
    "Alex Albon": 128.7, "Pierre Gasly": 133.1, "Esteban Ocon": 130.2,
    "Oliver Bearman": 131.8, "Franco Colapinto": 125.6,
    "Liam Lawson": 128.9, "Arvid Lindblad": 127.3, "Lance Stroll": 121.5,
    "Fernando Alonso": 123.8, "Valtteri Bottas": 119.2,
    "Nico Hulkenberg": 117.5, "Gabriel Bortoleto": 118.9,
    "Yuki Tsunoda": 128.0,
}

PRESEASON_RELIABILITY = {
    "Mercedes": 0.92, "Ferrari": 0.90, "Red Bull": 0.85,
    "McLaren": 0.88, "Williams": 0.80, "Haas": 0.82,
    "Alpine": 0.80, "Racing Bulls": 0.82, "Cadillac": 0.75,
    "Aston Martin": 0.75, "Audi": 0.78,
}

PRESEASON_PROBS = {
    "Max Verstappen": 34.8, "Lando Norris": 23.0, "George Russell": 12.7,
    "Oscar Piastri": 11.8, "Charles Leclerc": 9.9, "Lewis Hamilton": 4.1,
    "Carlos Sainz": 0.9, "Kimi Antonelli": 0.8,
}


def compute_actual_metrics():
    rows = []
    for driver, d in SEASON_TO_DATE.items():
        positions = [p for p in d["pos"] if isinstance(p, int)]
        dnfs = sum(1 for p in d["pos"] if p == "D")
        dns = sum(1 for p in d["pos"] if p == "N")
        entered = max(len(positions) + dnfs + dns, 1)

        avg_finish = float(np.mean(positions)) if positions else 22.0
        wins = sum(1 for p in positions if p == 1)
        podiums = sum(1 for p in positions if p <= 3)
        top10 = sum(1 for p in positions if p <= 10)

        rows.append({
            "driver": driver, "team": d["team"], "actual_points": d["pts"],
            "avg_finish": round(avg_finish, 1),
            "pts_per_round": round(d["pts"] / entered, 1),
            "wins": wins, "podiums": podiums,
            "top10_rate": round(top10 / entered, 2),
            "reliability": round(len(positions) / entered, 2),
            "dnfs": dnfs, "dns": dns, "entered": entered,
        })
    return pd.DataFrame(rows).sort_values("actual_points", ascending=False).reset_index(drop=True)


def compute_actual_strength(metrics_df):
    strengths = {}
    for _, r in metrics_df.iterrows():
        strengths[r["driver"]] = (
            100
            + r["pts_per_round"] * 3.2
            + max(0, (22 - r["avg_finish"]) * 2.6)
            + r["wins"] * 2.5
            + r["podiums"] * 1.2
        )
    return strengths


def compute_actual_reliability(metrics_df):
    entered, clean = {}, {}
    for _, r in metrics_df.iterrows():
        t = r["team"]
        entered[t] = entered.get(t, 0) + r["entered"]
        clean[t] = clean.get(t, 0) + (r["entered"] - r["dnfs"] - r["dns"])
    return {t: clean[t] / entered[t] for t in entered if entered[t] > 0}


def blend(preseason, actual, alpha):
    return {k: alpha * preseason[k] + (1 - alpha) * actual.get(k, preseason[k]) for k in preseason}


# Strength estimates are themselves uncertain. Treating them as known exactly
# is what made the previous version return 100% title probabilities. We resample
# each driver's strength per batch from N(estimate, STRENGTH_SIGMA); sigma shrinks
# as more of the season is observed.
STRENGTH_SIGMA = 6.0 * np.sqrt(ROUNDS_REMAINING / ROUNDS_TOTAL)
N_PARAM_DRAWS = 40
SIMS_PER_DRAW = 400


def run_projection(metrics, actual_strengths, actual_reliability, alpha_s, alpha_r, seed=42):
    bl_s = blend(PRESEASON_STRENGTHS, actual_strengths, alpha_s)
    bl_r = blend(PRESEASON_RELIABILITY, actual_reliability, alpha_r)

    raw = list(bl_s.values())
    lo, hi = min(raw), max(raw)
    norm = {d: 30 + (s - lo) / (hi - lo) * 70 for d, s in bl_s.items()}

    actual_pts = dict(zip(metrics["driver"], metrics["actual_points"]))
    rng = np.random.default_rng(seed)

    point_batches, driver_order = [], None

    for draw in range(N_PARAM_DRAWS):
        # Sample a plausible "true" strength vector for this draw
        jittered = {
            d: float(np.clip(s + rng.normal(0, STRENGTH_SIGMA), 5, 110))
            for d, s in norm.items()
        }
        mc = MonteCarloSimulator(n_simulations=SIMS_PER_DRAW, random_state=seed + draw)
        mc.N_RACES_2026 = ROUNDS_REMAINING
        mc.simulate_season(jittered, bl_r, DRIVER_TEAMS)
        if driver_order is None:
            driver_order = mc.last_drivers
        point_batches.append(mc.last_all_points)

    all_points = np.vstack(point_batches)                    # (n_sims_total, n_drivers)
    locked = np.array([actual_pts.get(d, 0) for d in driver_order])
    totals = all_points + locked

    winners = np.argmax(totals, axis=1)
    counts = np.bincount(winners, minlength=len(driver_order))
    n_total = all_points.shape[0]
    probs = {driver_order[i]: counts[i] / n_total * 100 for i in range(len(driver_order))}

    res = pd.DataFrame({
        "driver": driver_order,
        "team": [DRIVER_TEAMS[d] for d in driver_order],
        "actual_points": locked,
        "mean_remaining": all_points.mean(axis=0),
        "std_remaining": all_points.std(axis=0),
        "projected_total": totals.mean(axis=0),
        "p_top3": (np.argsort(np.argsort(-totals, axis=1), axis=1) < 3).mean(axis=0) * 100,
    })
    res = res.sort_values("projected_total", ascending=False).reset_index(drop=True)
    return res, probs


def main():
    print("=" * 78)
    print(f"  F1 2026 MID-SEASON UPDATE — After Round {ROUNDS_COMPLETE} of {ROUNDS_TOTAL} (Dutch GP)")
    print("=" * 78)

    metrics = compute_actual_metrics()
    actual_pts = dict(zip(metrics["driver"], metrics["actual_points"]))

    print(f"\nACTUAL STANDINGS AFTER {ROUNDS_COMPLETE} ROUNDS")
    print(f"  {'Driver':<20} {'Team':<14} {'Pts':>5} {'AvgFin':>7} {'W':>3} {'Pod':>4} {'Rel':>6}")
    print("  " + "-" * 66)
    for _, r in metrics.iterrows():
        print(f"  {r['driver']:<20} {r['team']:<14} {r['actual_points']:>5} "
              f"{r['avg_finish']:>7} {r['wins']:>3} {r['podiums']:>4} {r['reliability']:>6}")

    actual_strengths = compute_actual_strength(metrics)
    actual_reliability = compute_actual_reliability(metrics)

    print(f"\nTEAM RELIABILITY - pre-season vs actual")
    print(f"  {'Team':<16} {'Pre':>7} {'Actual':>8} {'Delta':>8}")
    print("  " + "-" * 42)
    for t in sorted(actual_reliability, key=lambda x: actual_reliability[x], reverse=True):
        pre = PRESEASON_RELIABILITY.get(t, 0.80)
        act = actual_reliability[t]
        print(f"  {t:<16} {pre:>7.2f} {act:>8.2f} {act-pre:>+8.2f}")

    mc_res, mc_probs = run_projection(metrics, actual_strengths, actual_reliability, 0.20, 0.25, seed=42)
    bay_res, bay_probs = run_projection(metrics, actual_strengths, actual_reliability, 0.05, 0.10, seed=101)
    ens_probs = {d: (mc_probs.get(d, 0) + bay_probs.get(d, 0)) / 2 for d in ALL_DRIVERS}

    mc_tot = dict(zip(mc_res["driver"], mc_res["projected_total"]))
    bay_tot = dict(zip(bay_res["driver"], bay_res["projected_total"]))
    order = sorted(ALL_DRIVERS, key=lambda d: (mc_tot.get(d, 0) + bay_tot.get(d, 0)) / 2, reverse=True)

    print("\n" + "=" * 78)
    print(f"  UPDATED WDC PROJECTION - {ROUNDS_REMAINING} rounds remaining (283 pts available)")
    print("=" * 78)
    print(f"\n  {'Pos':<4} {'Driver':<20} {'Now':>5} {'Proj':>6} {'MC%':>7} {'Bayes%':>8} {'Ens%':>7}")
    print("  " + "-" * 62)
    for i, d in enumerate(order):
        proj = (mc_tot.get(d, 0) + bay_tot.get(d, 0)) / 2
        print(f"  {i+1:<4} {d:<20} {actual_pts.get(d,0):>5} {proj:>6.0f} "
              f"{mc_probs.get(d,0):>6.1f}% {bay_probs.get(d,0):>7.1f}% {ens_probs.get(d,0):>6.1f}%")

    print("\n" + "=" * 78)
    print("  UPDATED WCC PROJECTION")
    print("=" * 78)
    team_now, team_proj = {}, {}
    for d in ALL_DRIVERS:
        t = DRIVER_TEAMS[d]
        team_now[t] = team_now.get(t, 0) + actual_pts.get(d, 0)
        team_proj[t] = team_proj.get(t, 0) + (mc_tot.get(d, 0) + bay_tot.get(d, 0)) / 2
    print(f"\n  {'Pos':<4} {'Team':<16} {'Now':>6} {'Projected':>11}")
    print("  " + "-" * 42)
    for i, (t, p) in enumerate(sorted(team_proj.items(), key=lambda x: x[1], reverse=True)):
        print(f"  {i+1:<4} {t:<16} {team_now.get(t,0):>6} {p:>11.0f}")

    print("\n" + "=" * 78)
    print("  PRE-SEASON MODEL SCORECARD")
    print("=" * 78)
    actual_rank = {r["driver"]: i + 1 for i, (_, r) in enumerate(metrics.iterrows())}
    preseason_rank = {d: i + 1 for i, d in enumerate(PRESEASON_PROBS)}
    print(f"\n  {'Driver':<20} {'Pre %':>7} {'PreRank':>8} {'NowRank':>8} {'Error':>7}")
    print("  " + "-" * 54)
    errors = []
    for d, p in PRESEASON_PROBS.items():
        pr, ar = preseason_rank[d], actual_rank.get(d, 22)
        errors.append(abs(pr - ar))
        print(f"  {d:<20} {p:>6.1f}% {pr:>8} {ar:>8} {pr-ar:>+7}")
    print(f"\n  Mean absolute rank error (top 8): {np.mean(errors):.2f} positions")

    out = []
    for d in order:
        out.append({
            "driver": d, "team": DRIVER_TEAMS[d],
            "actual_pts": int(actual_pts.get(d, 0)),
            "projected_total": int((mc_tot.get(d, 0) + bay_tot.get(d, 0)) / 2),
            "mc_win": round(mc_probs.get(d, 0), 1),
            "bayes_win": round(bay_probs.get(d, 0), 1),
            "ensemble_win": round(ens_probs.get(d, 0), 1),
        })
    os.makedirs("visualizations", exist_ok=True)
    with open("visualizations/updated_predictions_r12.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved -> visualizations/updated_predictions_r12.json")
    print("=" * 78)
    return out, metrics, team_now, team_proj


if __name__ == "__main__":
    main()
