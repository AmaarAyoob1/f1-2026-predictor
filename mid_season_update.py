"""
F1 2026 Mid-Season Update — After Round 3 (Japan)

Bayesian blending approach:
1. Lock in actual results from Races 1-3 (Australia, China Sprint, China, Japan)
2. Compute actual driver performance metrics from real race data
3. Blend pre-season strength with actual performance
4. Update team reliability from real DNF/DNS data  
5. Re-run Monte Carlo for remaining 21 races
6. Final prediction = actual points + simulated remaining points
"""

import numpy as np
import pandas as pd
import json, sys, os

sys.path.insert(0, os.path.dirname(__file__))

from models.train_models import MonteCarloSimulator

# ═══════════════════════════════════════════════════════════
# ACTUAL 2026 RACE RESULTS — Rounds 1-3 + China Sprint
# ═══════════════════════════════════════════════════════════

# Points: 25-18-15-12-10-8-6-4-2-1 (race), 8-7-6-5-4-3-2-1 (sprint top 8)
# DNS = Did Not Start, DNF = Did Not Finish, NC = Not Classified

ACTUAL_RESULTS = {
    "australia": {
        "round": 1, "type": "race",
        "results": [
            ("George Russell", 1, 25),
            ("Kimi Antonelli", 2, 18),
            ("Charles Leclerc", 3, 15),
            ("Lewis Hamilton", 4, 12),
            ("Lando Norris", 5, 10),
            ("Max Verstappen", 6, 8),
            ("Oliver Bearman", 7, 6),
            ("Arvid Lindblad", 8, 4),
            ("Gabriel Bortoleto", 9, 2),
            ("Pierre Gasly", 10, 1),
            ("Carlos Sainz", 11, 0),
            ("Liam Lawson", 12, 0),
            ("Franco Colapinto", 13, 0),
            ("Esteban Ocon", 14, 0),
            ("Sergio Perez", 15, 0),
            ("Lance Stroll", 16, 0),
            ("Fernando Alonso", 17, 0),
            ("Alex Albon", 18, 0),
        ],
        "dnf": ["Isack Hadjar", "Valtteri Bottas"],
        "dns": ["Oscar Piastri", "Nico Hulkenberg"],
    },
    "china_sprint": {
        "round": 2, "type": "sprint",
        "results": [
            ("George Russell", 1, 8),
            ("Charles Leclerc", 2, 7),
            ("Lewis Hamilton", 3, 6),
            ("Kimi Antonelli", 4, 5),
            ("Lando Norris", 5, 4),
            ("Max Verstappen", 6, 3),
            ("Oliver Bearman", 7, 2),  # estimated based on standings
            ("Liam Lawson", 8, 1),
        ],
        "dnf": [],
        "dns": [],
    },
    "china": {
        "round": 2, "type": "race",
        "results": [
            ("Kimi Antonelli", 1, 25),
            ("George Russell", 2, 18),
            ("Lewis Hamilton", 3, 15),
            ("Charles Leclerc", 4, 12),
            ("Oliver Bearman", 5, 10),
            ("Pierre Gasly", 6, 8),
            ("Liam Lawson", 7, 6),
            ("Isack Hadjar", 8, 4),
            ("Carlos Sainz", 9, 2),
            ("Franco Colapinto", 10, 1),
            ("Nico Hulkenberg", 11, 0),
            ("Arvid Lindblad", 12, 0),
            ("Valtteri Bottas", 13, 0),
            ("Esteban Ocon", 14, 0),
            ("Sergio Perez", 15, 0),
            ("Fernando Alonso", 16, 0),
        ],
        "dnf": ["Max Verstappen", "Lance Stroll"],
        "dns": ["Lando Norris", "Oscar Piastri", "Alex Albon", "Gabriel Bortoleto"],
    },
    "japan": {
        "round": 3, "type": "race",
        "results": [
            ("Kimi Antonelli", 1, 25),
            ("Oscar Piastri", 2, 18),
            ("Charles Leclerc", 3, 15),
            ("George Russell", 4, 12),
            ("Lando Norris", 5, 10),
            ("Lewis Hamilton", 6, 8),
            ("Pierre Gasly", 7, 6),
            ("Max Verstappen", 8, 4),
            ("Liam Lawson", 9, 2),
            ("Esteban Ocon", 10, 1),
            ("Nico Hulkenberg", 11, 0),
            ("Isack Hadjar", 12, 0),
            ("Gabriel Bortoleto", 13, 0),
            ("Arvid Lindblad", 14, 0),
            ("Carlos Sainz", 15, 0),
            ("Franco Colapinto", 16, 0),
            ("Sergio Perez", 17, 0),
            ("Fernando Alonso", 18, 0),
            ("Valtteri Bottas", 19, 0),
            ("Alex Albon", 20, 0),
        ],
        "dnf": ["Oliver Bearman", "Lance Stroll"],
        "dns": [],
    },
}

# ═══════════════════════════════════════════════════════════
# COMPUTE ACTUAL PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════

ALL_DRIVERS = [
    "Max Verstappen", "Lando Norris", "George Russell", "Oscar Piastri",
    "Charles Leclerc", "Lewis Hamilton", "Carlos Sainz", "Kimi Antonelli",
    "Isack Hadjar", "Alex Albon", "Pierre Gasly", "Esteban Ocon",
    "Oliver Bearman", "Sergio Perez", "Franco Colapinto", "Liam Lawson",
    "Arvid Lindblad", "Lance Stroll", "Fernando Alonso", "Valtteri Bottas",
    "Nico Hulkenberg", "Gabriel Bortoleto",
]

DRIVER_TEAMS = {
    "Max Verstappen": "Red Bull", "Isack Hadjar": "Red Bull",
    "Lando Norris": "McLaren", "Oscar Piastri": "McLaren",
    "George Russell": "Mercedes", "Kimi Antonelli": "Mercedes",
    "Charles Leclerc": "Ferrari", "Lewis Hamilton": "Ferrari",
    "Carlos Sainz": "Williams", "Alex Albon": "Williams",
    "Pierre Gasly": "Alpine", "Franco Colapinto": "Alpine",
    "Esteban Ocon": "Haas", "Oliver Bearman": "Haas",
    "Sergio Perez": "Cadillac", "Valtteri Bottas": "Cadillac",
    "Liam Lawson": "Racing Bulls", "Arvid Lindblad": "Racing Bulls",
    "Lance Stroll": "Aston Martin", "Fernando Alonso": "Aston Martin",
    "Nico Hulkenberg": "Audi", "Gabriel Bortoleto": "Audi",
}


def compute_actual_metrics():
    """Compute per-driver performance metrics from actual race results."""
    
    driver_stats = {d: {
        "points": 0, "races_entered": 0, "races_finished": 0,
        "finishes": [], "wins": 0, "podiums": 0, "top10": 0,
        "dnfs": 0, "dns": 0,
    } for d in ALL_DRIVERS}
    
    for event_name, event in ACTUAL_RESULTS.items():
        for driver, pos, pts in event["results"]:
            s = driver_stats[driver]
            s["points"] += pts
            s["races_entered"] += 1
            s["races_finished"] += 1
            s["finishes"].append(pos)
            if pos == 1: s["wins"] += 1
            if pos <= 3: s["podiums"] += 1
            if pos <= 10: s["top10"] += 1
        
        for driver in event.get("dnf", []):
            s = driver_stats[driver]
            s["races_entered"] += 1
            s["dnfs"] += 1
            s["finishes"].append(22)  # DNF treated as last
        
        for driver in event.get("dns", []):
            s = driver_stats[driver]
            s["dns"] += 1
            # DNS counts against reliability but not finish position
    
    # Compute derived metrics
    results = []
    for driver in ALL_DRIVERS:
        s = driver_stats[driver]
        n_events = s["races_entered"] + s["dns"]  # total events driver should have raced
        avg_finish = np.mean(s["finishes"]) if s["finishes"] else 22
        pts_per_race = s["points"] / max(n_events, 1)
        reliability = s["races_finished"] / max(n_events, 1)
        
        results.append({
            "driver": driver,
            "team": DRIVER_TEAMS[driver],
            "actual_points": s["points"],
            "avg_finish": round(avg_finish, 1),
            "pts_per_race": round(pts_per_race, 1),
            "wins": s["wins"],
            "podiums": s["podiums"],
            "top10_rate": round(s["top10"] / max(s["races_entered"], 1), 2),
            "reliability": round(reliability, 2),
            "dnfs": s["dnfs"],
            "dns": s["dns"],
            "events": n_events,
        })
    
    return pd.DataFrame(results).sort_values("actual_points", ascending=False)


# ═══════════════════════════════════════════════════════════
# PRE-SEASON STRENGTH SCORES (from original model)
# ═══════════════════════════════════════════════════════════

# These are the composite strength scores from the pre-season model
# (historical + forward-looking + driver talent components)
# Extracted from running main.py
PRESEASON_STRENGTHS = {
    "Max Verstappen": 195.2, "Lando Norris": 176.8, "George Russell": 172.4,
    "Charles Leclerc": 170.1, "Oscar Piastri": 163.5, "Lewis Hamilton": 168.7,
    "Carlos Sainz": 148.2, "Kimi Antonelli": 141.4, "Isack Hadjar": 130.5,
    "Alex Albon": 128.7, "Pierre Gasly": 133.1, "Esteban Ocon": 130.2,
    "Oliver Bearman": 131.8, "Sergio Perez": 126.4, "Franco Colapinto": 125.6,
    "Liam Lawson": 128.9, "Arvid Lindblad": 127.3, "Lance Stroll": 121.5,
    "Fernando Alonso": 123.8, "Valtteri Bottas": 119.2,
    "Nico Hulkenberg": 117.5, "Gabriel Bortoleto": 118.9,
}

PRESEASON_RELIABILITY = {
    "Mercedes": 0.92, "Ferrari": 0.90, "Red Bull": 0.85,
    "McLaren": 0.88, "Williams": 0.80, "Haas": 0.82,
    "Alpine": 0.80, "Racing Bulls": 0.82, "Cadillac": 0.75,
    "Aston Martin": 0.75, "Audi": 0.78,
}


# ═══════════════════════════════════════════════════════════
# BAYESIAN BLENDING: PRE-SEASON + ACTUAL PERFORMANCE
# ═══════════════════════════════════════════════════════════

def compute_actual_strength(metrics_df):
    """
    Convert actual race metrics into a strength score on the same scale
    as the pre-season composite strength (~100-200 range).
    
    Uses: points per race, average finish, win rate, and reliability.
    """
    actual_strengths = {}
    
    for _, row in metrics_df.iterrows():
        driver = row["driver"]
        
        # Points-per-race scaled to strength (25 pts/race max → ~200 strength)
        pts_component = row["pts_per_race"] * 7  # 25 * 7 = 175 for a perfect scorer
        
        # Finish position component (P1 avg → high bonus, P20 → low)
        position_component = max(0, (22 - row["avg_finish"]) * 3)
        
        # Win bonus
        win_component = row["wins"] * 5
        
        # Base floor (even a DNS driver has some inherent ability)
        base = 100
        
        actual_strengths[driver] = base + pts_component + position_component + win_component
    
    return actual_strengths


def compute_actual_reliability(metrics_df):
    """Compute team reliability from actual DNF/DNS data."""
    team_events = {}
    team_clean = {}
    
    for _, row in metrics_df.iterrows():
        team = row["team"]
        events = row["events"]
        clean = events - row["dnfs"] - row["dns"]
        
        team_events[team] = team_events.get(team, 0) + events
        team_clean[team] = team_clean.get(team, 0) + clean
    
    actual_reliability = {}
    for team in team_events:
        if team_events[team] > 0:
            actual_reliability[team] = team_clean[team] / team_events[team]
        else:
            actual_reliability[team] = 0.80
    
    return actual_reliability


def blend_strengths(preseason, actual, alpha=0.50):
    """
    Blend pre-season and actual strengths.
    
    alpha = weight on pre-season (0.5 = equal blend after 3 races)
    As more races happen, alpha should decrease:
      3 races: alpha = 0.50
      6 races: alpha = 0.30
      12 races: alpha = 0.15
      18+ races: alpha = 0.05
    """
    blended = {}
    for driver in preseason:
        pre = preseason[driver]
        act = actual.get(driver, pre)  # fallback to preseason if no actual data
        blended[driver] = alpha * pre + (1 - alpha) * act
    return blended


def blend_reliability(preseason, actual, alpha=0.40):
    """Blend pre-season and actual reliability. Less trust in preseason for reliability."""
    blended = {}
    for team in preseason:
        pre = preseason[team]
        act = actual.get(team, pre)
        blended[team] = alpha * pre + (1 - alpha) * act
    return blended


# ═══════════════════════════════════════════════════════════
# MAIN UPDATE PIPELINE
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  F1 2026 MID-SEASON UPDATE — After Round 3 (Japan)")
    print("=" * 70)
    
    # Step 1: Compute actual metrics
    metrics = compute_actual_metrics()
    print("\n📊 ACTUAL STANDINGS AFTER 3 RACES:")
    print(f"  {'Driver':<22} {'Pts':>5} {'AvgFin':>7} {'Pts/R':>6} {'W':>3} {'Rel':>5}")
    print("  " + "─" * 55)
    for _, r in metrics.head(15).iterrows():
        print(f"  {r['driver']:<22} {r['actual_points']:>5} {r['avg_finish']:>7} "
              f"{r['pts_per_race']:>6} {r['wins']:>3} {r['reliability']:>5}")
    
    # Step 2: Compute actual strength scores
    actual_strengths = compute_actual_strength(metrics)
    
    # Step 3: Blend pre-season + actual
    ALPHA = 0.50  # 50% preseason, 50% actual after 3 races
    blended_strengths = blend_strengths(PRESEASON_STRENGTHS, actual_strengths, alpha=ALPHA)
    
    actual_reliability = compute_actual_reliability(metrics)
    blended_reliability = blend_reliability(PRESEASON_RELIABILITY, actual_reliability, alpha=0.40)
    
    print(f"\n🔀 BLENDED STRENGTHS (α={ALPHA} preseason / {1-ALPHA} actual):")
    print(f"  {'Driver':<22} {'PreSeason':>10} {'Actual':>10} {'Blended':>10} {'Δ':>8}")
    print("  " + "─" * 62)
    sorted_drivers = sorted(blended_strengths.keys(), key=lambda d: blended_strengths[d], reverse=True)
    for d in sorted_drivers:
        pre = PRESEASON_STRENGTHS.get(d, 0)
        act = actual_strengths.get(d, 0)
        bl = blended_strengths[d]
        delta = bl - pre
        print(f"  {d:<22} {pre:>10.1f} {act:>10.1f} {bl:>10.1f} {delta:>+8.1f}")
    
    print(f"\n🔧 BLENDED TEAM RELIABILITY:")
    for team in sorted(blended_reliability, key=lambda t: blended_reliability[t], reverse=True):
        pre = PRESEASON_RELIABILITY.get(team, 0)
        act = actual_reliability.get(team, 0)
        bl = blended_reliability[team]
        print(f"  {team:<16} Pre: {pre:.2f}  Actual: {act:.2f}  Blended: {bl:.2f}")
    
    # Step 4: Normalize strengths to 30-100 scale for MC simulator
    raw_strengths = list(blended_strengths.values())
    min_s, max_s = min(raw_strengths), max(raw_strengths)
    normalized = {}
    for d, s in blended_strengths.items():
        normalized[d] = 30 + (s - min_s) / (max_s - min_s) * 70
    
    # Step 5: Run MC simulation for REMAINING 21 races
    print(f"\n🎲 Running Monte Carlo simulation for 21 remaining races...")
    
    mc = MonteCarloSimulator(n_simulations=10000, random_state=42)
    # Override the number of races
    mc.N_RACES_2026 = 21  # remaining races
    
    mc_results = mc.simulate_season(normalized, blended_reliability, DRIVER_TEAMS)
    
    # Step 6: Add actual points to simulated remaining points
    actual_pts = dict(zip(metrics["driver"], metrics["actual_points"]))
    mc_results["actual_points"] = mc_results["driver"].map(actual_pts).fillna(0)
    mc_results["total_mean_points"] = mc_results["actual_points"] + mc_results["mean_points"]
    
    # Re-rank by total projected points
    mc_results = mc_results.sort_values("total_mean_points", ascending=False).reset_index(drop=True)
    
    # Step 7: Output updated predictions
    print("\n" + "=" * 70)
    print("  🏆 UPDATED 2026 WDC PREDICTIONS (Post-Japan)")
    print("=" * 70)
    print(f"\n  {'Pos':<4} {'Driver':<22} {'Team':<14} {'Actual':>7} {'Proj Rem':>9} {'Total':>7} {'Win%':>7}")
    print("  " + "─" * 75)
    
    for i, row in mc_results.iterrows():
        pos = i + 1
        win_pct = row["p_champion"] * 100
        marker = " ◄" if pos <= 3 else ""
        print(f"  {pos:<4} {row['driver']:<22} {row['team']:<14} "
              f"{row['actual_points']:>7.0f} {row['mean_points']:>9.0f} "
              f"{row['total_mean_points']:>7.0f} {win_pct:>6.1f}%{marker}")
    
    # Step 8: Constructors
    print("\n" + "=" * 70)
    print("  🏗  UPDATED 2026 WCC PREDICTIONS (Post-Japan)")
    print("=" * 70)
    
    team_totals = {}
    for _, row in mc_results.iterrows():
        team = row["team"]
        if team not in team_totals:
            team_totals[team] = {"total": 0, "actual": 0, "drivers": []}
        team_totals[team]["total"] += row["total_mean_points"]
        team_totals[team]["actual"] += row["actual_points"]
        team_totals[team]["drivers"].append(row["driver"].split()[-1])
    
    sorted_teams = sorted(team_totals.items(), key=lambda x: x[1]["total"], reverse=True)
    print(f"\n  {'Pos':<4} {'Team':<16} {'Drivers':<28} {'Actual':>7} {'Projected':>10}")
    print("  " + "─" * 68)
    for i, (team, data) in enumerate(sorted_teams):
        drivers_str = " & ".join(data["drivers"])
        print(f"  {i+1:<4} {team:<16} {drivers_str:<28} {data['actual']:>7.0f} {data['total']:>10.0f}")
    
    # Step 9: Generate comparison table
    print("\n" + "=" * 70)
    print("  📊 PRE-SEASON vs UPDATED COMPARISON")
    print("=" * 70)
    
    preseason_order = ["Max Verstappen", "Lando Norris", "George Russell",
                       "Oscar Piastri", "Charles Leclerc", "Lewis Hamilton",
                       "Carlos Sainz", "Kimi Antonelli"]
    
    print(f"\n  {'Driver':<22} {'Pre-Season':>10} {'Updated':>10} {'Actual Pos':>11}")
    print("  " + "─" * 55)
    
    actual_ranking = {row["driver"]: i+1 for i, (_, row) in enumerate(metrics.iterrows())}
    updated_ranking = {row["driver"]: i+1 for i, (_, row) in enumerate(mc_results.iterrows())}
    
    for d in preseason_order:
        pre_rank = preseason_order.index(d) + 1
        upd_rank = updated_ranking.get(d, "?")
        act_rank = actual_ranking.get(d, "?")
        print(f"  {d:<22} {'P'+str(pre_rank):>10} {'P'+str(upd_rank):>10} {'P'+str(act_rank):>11}")
    
    # Step 10: Save results for dashboard update
    dashboard_data = []
    for i, row in mc_results.iterrows():
        dashboard_data.append({
            "driver": row["driver"],
            "team": row["team"],
            "actual_pts": int(row["actual_points"]),
            "projected_total": int(row["total_mean_points"]),
            "projected_remaining": int(row["mean_points"]),
            "win_pct": round(row["p_champion"] * 100, 1),
            "top3_pct": round(row["p_top3"] * 100, 1),
        })
    
    with open("visualizations/updated_predictions_r3.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"\n  Updated predictions saved to visualizations/updated_predictions_r3.json")
    
    print("\n" + "=" * 70)
    print("  UPDATE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
