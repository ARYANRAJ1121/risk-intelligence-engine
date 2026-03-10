"""
evaluate.py — Model Evaluation & Visualization Module
======================================================
WHY THIS EXISTS:
    Credit risk models are judged by specific metrics that go beyond
    simple accuracy. AUC-ROC, KS statistic, and Gini coefficient are
    the industry standard. This module computes all of them and generates
    publication-ready plots for the dashboard and PDF report.

WHAT IT PRODUCES:
    - metrics JSON file in artifacts/metrics/
    - ROC curve plot (all models overlaid)
    - Precision-Recall curve
    - KS statistic plot
    - Confusion matrix for the best model
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("models.evaluate")


# ============================================================
#  METRICS COMPUTATION
# ============================================================

def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Kolmogorov-Smirnov statistic.

    KS = maximum separation between the CDF of default probabilities
    for actual defaults vs. actual non-defaults.

    In credit risk:
        - KS > 0.20 = acceptable
        - KS > 0.30 = good
        - KS > 0.40 = excellent

    Args:
        y_true: Actual labels (0/1)
        y_prob: Predicted probabilities of default

    Returns:
        float: KS statistic
    """
    from scipy.stats import ks_2samp

    default_probs = y_prob[y_true == 1]
    non_default_probs = y_prob[y_true == 0]

    ks_stat, _ = ks_2samp(default_probs, non_default_probs)
    return ks_stat


def evaluate_single_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a single model on the test set.

    Returns a dict with all credit risk metrics.
    """
    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Core metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc_roc - 1
    ks = compute_ks_statistic(y_test.values, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
    }

    logger.info("  %s — AUC: %.4f | Gini: %.4f | KS: %.4f | F1: %.4f",
                model_name, auc_roc, gini, ks, f1)

    return metrics


def evaluate_all_models(results: dict, data: dict) -> dict:
    """
    Evaluate all trained models and generate comparison plots.

    Args:
        results: dict from train_all_models()
        data: dict from prepare_data()

    Returns:
        dict of {model_name: metrics_dict}
    """
    logger.info("Evaluating models on test set (%d rows)...", len(data["X_test"]))

    all_metrics = {}
    for name, result in results.items():
        metrics = evaluate_single_model(
            result["model"], data["X_test"], data["y_test"], name
        )
        metrics["train_time_sec"] = round(result["train_time"], 2)
        all_metrics[name] = metrics

    # Print comparison table
    logger.info("")
    logger.info("%-25s  %8s  %8s  %8s  %8s  %8s",
                "MODEL", "AUC-ROC", "GINI", "KS", "PR-AUC", "F1")
    logger.info("-" * 80)
    for name, m in all_metrics.items():
        logger.info("%-25s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
                     name, m["auc_roc"], m["gini"], m["ks_statistic"],
                     m["pr_auc"], m["f1_score"])

    return all_metrics


# ============================================================
#  VISUALIZATION
# ============================================================

def plot_roc_curves(results: dict, data: dict, dataset: str) -> str:
    """
    Plot ROC curves for all models on one chart.

    Returns the path to the saved plot.
    """
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for i, (name, result) in enumerate(results.items()):
        y_prob = result["model"].predict_proba(data["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(data["y_test"], y_prob)
        auc = roc_auc_score(data["y_test"], y_prob)

        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — PD Model Comparison ({dataset.upper()})", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    path = plots_dir / f"{dataset}_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve plot to %s", path)
    return str(path)


def plot_precision_recall(results: dict, data: dict, dataset: str) -> str:
    """
    Plot Precision-Recall curves for all models.

    PR curves are more informative than ROC for imbalanced datasets
    because they focus on the minority class (defaults).
    """
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for i, (name, result) in enumerate(results.items()):
        y_prob = result["model"].predict_proba(data["X_test"])[:, 1]
        precision, recall, _ = precision_recall_curve(data["y_test"], y_prob)
        ap = average_precision_score(data["y_test"], y_prob)

        ax.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve ({dataset.upper()})", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)

    path = plots_dir / f"{dataset}_pr_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve plot to %s", path)
    return str(path)


def plot_ks_curve(model, data: dict, dataset: str, model_name: str) -> str:
    """
    Plot the KS curve for the best model.

    Shows the CDFs of predicted probabilities for defaults vs non-defaults,
    with the KS statistic marked as the maximum gap.
    """
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(data["X_test"])[:, 1]
    y_true = data["y_test"].values

    # Compute CDFs
    thresholds = np.linspace(0, 1, 500)
    default_probs = y_prob[y_true == 1]
    non_default_probs = y_prob[y_true == 0]

    cdf_default = [np.mean(default_probs <= t) for t in thresholds]
    cdf_non_default = [np.mean(non_default_probs <= t) for t in thresholds]

    cdf_default = np.array(cdf_default)
    cdf_non_default = np.array(cdf_non_default)

    ks_values = np.abs(cdf_default - cdf_non_default)
    ks_max_idx = np.argmax(ks_values)
    ks_stat = ks_values[ks_max_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(thresholds, cdf_default, color="#FF5722", lw=2, label="Default CDF")
    ax.plot(thresholds, cdf_non_default, color="#2196F3", lw=2, label="Non-Default CDF")

    # Mark KS point
    ax.vlines(thresholds[ks_max_idx], cdf_non_default[ks_max_idx],
              cdf_default[ks_max_idx], color="#4CAF50", lw=3,
              linestyle="--", label=f"KS = {ks_stat:.4f}")

    ax.set_xlabel("Predicted Probability of Default", fontsize=12)
    ax.set_ylabel("Cumulative Proportion", fontsize=12)
    ax.set_title(f"KS Statistic — {model_name} ({dataset.upper()})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    path = plots_dir / f"{dataset}_ks_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved KS curve to %s", path)
    return str(path)


def plot_confusion_matrix(model, data: dict, dataset: str, model_name: str) -> str:
    """
    Plot the confusion matrix for the best model.
    """
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(data["X_test"])
    cm = confusion_matrix(data["y_test"], y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Default", "Default"], fontsize=12)
    ax.set_yticklabels(["Non-Default", "Default"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name} ({dataset.upper()})", fontsize=14)

    fig.colorbar(im, ax=ax, shrink=0.8)

    path = plots_dir / f"{dataset}_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)
    return str(path)


def plot_pd_distribution(model, data: dict, dataset: str, model_name: str) -> str:
    """
    Plot the distribution of predicted PD scores.

    This is a key visualization for bank risk committees — shows
    the spread of default probabilities across the portfolio.
    """
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(data["X_test"])[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_prob, bins=50, color="#2196F3", alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(np.mean(y_prob), color="#FF5722", lw=2, linestyle="--",
               label=f"Mean PD = {np.mean(y_prob):.4f}")
    ax.axvline(np.median(y_prob), color="#4CAF50", lw=2, linestyle="--",
               label=f"Median PD = {np.median(y_prob):.4f}")

    ax.set_xlabel("Predicted Probability of Default", fontsize=12)
    ax.set_ylabel("Number of Loans", fontsize=12)
    ax.set_title(f"PD Score Distribution — {model_name} ({dataset.upper()})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    path = plots_dir / f"{dataset}_pd_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PD distribution plot to %s", path)
    return str(path)


# ============================================================
#  SAVE / EXPORT
# ============================================================

def save_metrics(metrics: dict, dataset: str) -> str:
    """Save all metrics to a JSON file."""
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    path = metrics_dir / f"{dataset}_model_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved metrics to %s", path)
    return str(path)


def generate_all_plots(results: dict, data: dict, dataset: str,
                       best_model, best_model_name: str) -> list[str]:
    """
    Generate all evaluation plots.

    Returns list of paths to saved plots.
    """
    logger.info("Generating evaluation plots...")
    paths = []

    paths.append(plot_roc_curves(results, data, dataset))
    paths.append(plot_precision_recall(results, data, dataset))
    paths.append(plot_ks_curve(best_model, data, dataset, best_model_name))
    paths.append(plot_confusion_matrix(best_model, data, dataset, best_model_name))
    paths.append(plot_pd_distribution(best_model, data, dataset, best_model_name))

    logger.info("Generated %d plots in artifacts/plots/", len(paths))
    return paths
