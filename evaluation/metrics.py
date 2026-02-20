"""
Model Evaluation & Calibration
================================
Comprehensive evaluation suite for F1 prediction models:

- Brier Score: measures calibration of probabilistic predictions
- Log-Loss: penalizes confident wrong predictions
- RMSE: for position prediction accuracy
- Calibration Curves: visual reliability assessment
- Bookmaker Comparison: does our model find edge vs market?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Brier Score: measures accuracy of probabilistic predictions.
    
    Range: [0, 1], lower is better.
    0 = perfect prediction, 1 = worst possible.
    A climatological baseline (predicting base rate) gives ~0.04 for champion prediction.
    """
    return brier_score_loss(y_true, y_pred_proba)


def compute_brier_skill_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Brier Skill Score: improvement over climatological baseline.
    
    BSS = 1 - BS_model / BS_baseline
    BSS > 0 means model beats naive baseline.
    BSS = 1 means perfect prediction.
    """
    bs_model = brier_score_loss(y_true, y_pred_proba)
    base_rate = y_true.mean()
    bs_baseline = brier_score_loss(y_true, np.full_like(y_pred_proba, base_rate))
    
    if bs_baseline == 0:
        return 0.0
    return 1 - bs_model / bs_baseline


def evaluate_position_model(y_true, y_pred):
    """Evaluate position prediction model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Top-3 accuracy
    true_top3 = set(np.where(y_true <= 3)[0])
    pred_top3 = set(np.argsort(y_pred)[:3])
    top3_accuracy = len(true_top3 & pred_top3) / max(len(true_top3), 1)
    
    # Champion accuracy
    true_champ = np.argmin(y_true)
    pred_champ = np.argmin(y_pred)
    champion_correct = int(true_champ == pred_champ)
    
    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "top3_accuracy": round(top3_accuracy, 3),
        "champion_correct": champion_correct,
    }


def compare_with_bookmakers(
    model_probs: dict,
    bookmaker_probs: dict,
    actual_champion: str = None
) -> pd.DataFrame:
    """
    Compare model predictions against bookmaker odds.
    
    Identifies where the model disagrees with the market —
    potential 'edges' that could indicate value.
    """
    rows = []
    for driver in model_probs:
        model_p = model_probs.get(driver, 0)
        book_p = bookmaker_probs.get(driver, 0)
        
        edge = model_p - book_p
        direction = "OVERWEIGHT" if edge > 0.02 else "UNDERWEIGHT" if edge < -0.02 else "NEUTRAL"
        
        row = {
            "driver": driver,
            "model_prob": round(model_p, 4),
            "bookmaker_prob": round(book_p, 4),
            "edge": round(edge, 4),
            "direction": direction,
        }
        
        if actual_champion:
            row["actual_winner"] = "✓" if driver == actual_champion else ""
        
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values("model_prob", ascending=False).reset_index(drop=True)


def generate_evaluation_report(
    position_model,
    prob_model,
    X_test, y_test,
    mc_results,
    bookmaker_odds,
    output_path="evaluation/evaluation_report.png"
):
    """
    Generate comprehensive visual evaluation report.
    
    Creates a multi-panel figure with:
    1. Feature importance (GBM)
    2. Predicted vs Actual positions
    3. Model vs Bookmaker comparison
    4. Monte Carlo championship probability distribution
    """
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    
    colors = {
        "primary": "#e53e3e",
        "secondary": "#27F4D2",
        "accent": "#FF8000",
        "bg": "#0f1117",
        "text": "#e2e8f0",
        "muted": "#64748b",
        "grid": "#1e293b",
    }
    
    # ── Panel 1: Feature Importance ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(colors["bg"])
    
    if position_model.feature_importances is not None:
        fi = position_model.feature_importances.head(12)
        bars = ax1.barh(
            range(len(fi)), fi["importance"].values,
            color=colors["primary"], alpha=0.85, height=0.6
        )
        ax1.set_yticks(range(len(fi)))
        ax1.set_yticklabels(fi["feature"].values, fontsize=7, color=colors["text"])
        ax1.invert_yaxis()
        ax1.set_xlabel("Importance", fontsize=9, color=colors["muted"])
        ax1.set_title("Feature Importance (GBM)", fontsize=11, color=colors["text"], fontweight="bold", pad=10)
        ax1.tick_params(colors=colors["muted"], labelsize=7)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_color(colors["grid"])
        ax1.spines["left"].set_color(colors["grid"])
    
    # ── Panel 2: Position Predictions Scatter ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(colors["bg"])
    
    if position_model.is_fitted and X_test is not None:
        y_pred = position_model.predict(X_test)
        ax2.scatter(y_test, y_pred, alpha=0.5, s=25, color=colors["secondary"], edgecolor="none")
        ax2.plot([0, 22], [0, 22], "--", color=colors["muted"], alpha=0.5, linewidth=1)
        ax2.set_xlabel("Actual Position", fontsize=9, color=colors["muted"])
        ax2.set_ylabel("Predicted Position", fontsize=9, color=colors["muted"])
        ax2.set_title("Predicted vs Actual Championship Position", fontsize=11, color=colors["text"], fontweight="bold", pad=10)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax2.text(0.05, 0.92, f"RMSE: {rmse:.2f}", transform=ax2.transAxes,
                fontsize=9, color=colors["primary"], fontweight="bold")
    
    ax2.tick_params(colors=colors["muted"], labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color(colors["grid"])
    ax2.spines["left"].set_color(colors["grid"])
    
    # ── Panel 3: CV Metrics ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(colors["bg"])
    
    metrics = position_model.get_metrics()
    metric_names = ["CV RMSE\n(Position)", "CV RMSE\nStd Dev"]
    metric_vals = [metrics["cv_rmse_mean"], metrics["cv_rmse_std"]]
    
    bars = ax3.bar(
        range(len(metric_names)), metric_vals,
        color=[colors["primary"], colors["accent"]],
        alpha=0.85, width=0.5
    )
    for bar, val in zip(bars, metric_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=10, color=colors["text"], fontweight="bold")
    
    ax3.set_xticks(range(len(metric_names)))
    ax3.set_xticklabels(metric_names, fontsize=9, color=colors["text"])
    ax3.set_title("Cross-Validation Metrics", fontsize=11, color=colors["text"], fontweight="bold", pad=10)
    ax3.tick_params(colors=colors["muted"], labelsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_color(colors["grid"])
    ax3.spines["left"].set_color(colors["grid"])
    
    # ── Panel 4: Monte Carlo Championship Probabilities ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor(colors["bg"])
    
    if mc_results is not None:
        top_drivers = mc_results.head(10)
        team_colors = {
            "Mercedes": "#27F4D2", "Ferrari": "#E8002D", "Red Bull": "#3671C6",
            "McLaren": "#FF8000", "Aston Martin": "#229971", "Williams": "#64C4FF",
            "Racing Bulls": "#6692FF", "Haas": "#B6BABD", "Alpine": "#FF87BC",
            "Audi": "#00E701", "Cadillac": "#FFD700",
        }
        
        bar_colors = [team_colors.get(row["team"], colors["muted"]) for _, row in top_drivers.iterrows()]
        
        bars = ax4.barh(
            range(len(top_drivers)), top_drivers["p_champion"].values * 100,
            color=bar_colors, alpha=0.85, height=0.6
        )
        
        for i, (_, row) in enumerate(top_drivers.iterrows()):
            pct = row["p_champion"] * 100
            label = f'{pct:.1f}%'
            ax4.text(pct + 0.5, i, label, va="center", fontsize=8,
                    color=colors["text"], fontweight="bold")
        
        ax4.set_yticks(range(len(top_drivers)))
        ax4.set_yticklabels(
            [f"{row['driver']} ({row['team']})" for _, row in top_drivers.iterrows()],
            fontsize=8, color=colors["text"]
        )
        ax4.invert_yaxis()
        ax4.set_xlabel("Championship Win Probability (%)", fontsize=9, color=colors["muted"])
        ax4.set_title("Monte Carlo: WDC Probability (10,000 sims)", fontsize=11,
                      color=colors["text"], fontweight="bold", pad=10)
    
    ax4.tick_params(colors=colors["muted"], labelsize=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_color(colors["grid"])
    ax4.spines["left"].set_color(colors["grid"])
    
    # ── Panel 5: Model vs Bookmaker ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(colors["bg"])
    
    if mc_results is not None and bookmaker_odds:
        top8 = mc_results.head(8)
        model_probs = top8["p_champion"].values * 100
        book_probs = [bookmaker_odds.get(row["driver"], 0.005) * 100 for _, row in top8.iterrows()]
        
        x = np.arange(len(top8))
        width = 0.35
        ax5.barh(x - width/2, model_probs, width, label="Our Model", color=colors["primary"], alpha=0.85)
        ax5.barh(x + width/2, book_probs, width, label="Bookmakers", color=colors["secondary"], alpha=0.85)
        
        ax5.set_yticks(x)
        short_names = [row["driver"].split()[-1] for _, row in top8.iterrows()]
        ax5.set_yticklabels(short_names, fontsize=8, color=colors["text"])
        ax5.invert_yaxis()
        ax5.legend(fontsize=8, loc="lower right", framealpha=0.3,
                  facecolor=colors["bg"], edgecolor=colors["grid"],
                  labelcolor=colors["text"])
        ax5.set_xlabel("Win Probability (%)", fontsize=9, color=colors["muted"])
        ax5.set_title("Model vs Bookmaker Odds", fontsize=11,
                      color=colors["text"], fontweight="bold", pad=10)
    
    ax5.tick_params(colors=colors["muted"], labelsize=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["bottom"].set_color(colors["grid"])
    ax5.spines["left"].set_color(colors["grid"])
    
    # Title
    fig.suptitle(
        "F1 2026 Championship Prediction — Model Evaluation Dashboard",
        fontsize=15, color=colors["text"], fontweight="bold", y=0.98
    )
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=colors["bg"], edgecolor="none")
    plt.close()
    
    return output_path
