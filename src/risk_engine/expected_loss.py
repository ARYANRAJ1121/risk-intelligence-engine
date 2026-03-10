"""
expected_loss.py — Basel III Expected Loss Calculator
=====================================================
WHY THIS EXISTS:
    The Basel III framework requires banks to hold capital reserves
    proportional to the Expected Loss (EL) of their loan portfolio.

    EL = PD x LGD x EAD

    This module takes the PD predictions from our model and computes
    EL at the loan level, segment level, and portfolio level.

WHAT IT PRODUCES:
    - Loan-level EL table (CSV)
    - Portfolio EL summary (JSON)
    - EL by risk grade segment (CSV)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("risk_engine.el")

# Basel III standard assumptions for unsecured consumer credit
DEFAULT_LGD = 0.45   # Loss Given Default = 45%
DEFAULT_EAD_COL = None  # If no EAD column, use loan_amnt or fixed amount


def compute_portfolio_el(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    dataset: str,
    lgd: float = DEFAULT_LGD,
    fixed_ead: float = 15000.0,
) -> dict:
    """
    Compute Expected Loss across the portfolio.

    For each loan:
        EL_i = PD_i x LGD x EAD_i

    Portfolio EL = sum of all loan-level EL values.

    Args:
        model: Trained PD model
        X: Feature matrix (test set or full portfolio)
        feature_names: Feature column names
        dataset: 'gmsc' or 'lc'
        lgd: Loss Given Default (Basel standard = 0.45)
        fixed_ead: Fixed EAD if no loan_amnt column available

    Returns:
        dict with portfolio-level summary
    """
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing Basel III Expected Loss...")
    logger.info("  Assumptions: LGD = %.0f%%, EAD = fixed $%.0f per loan", lgd * 100, fixed_ead)

    # ---- Predict PD ---- #
    pd_scores = model.predict_proba(X)[:, 1]

    # ---- Build loan-level EL table ---- #
    el_df = pd.DataFrame({
        "pd_predicted": pd_scores,
        "lgd": lgd,
        "ead": fixed_ead,
    })
    el_df["expected_loss"] = el_df["pd_predicted"] * el_df["lgd"] * el_df["ead"]

    # ---- Assign risk grades based on PD ---- #
    el_df["risk_grade"] = pd.cut(
        el_df["pd_predicted"],
        bins=[0, 0.05, 0.10, 0.20, 0.50, 1.0],
        labels=["A (Very Low)", "B (Low)", "C (Medium)", "D (High)", "E (Very High)"],
        include_lowest=True,
    )

    # ---- Portfolio summary ---- #
    total_el = el_df["expected_loss"].sum()
    total_ead = el_df["ead"].sum()
    avg_pd = el_df["pd_predicted"].mean()
    median_pd = el_df["pd_predicted"].median()

    summary = {
        "total_loans": len(el_df),
        "total_ead": round(total_ead, 2),
        "total_expected_loss": round(total_el, 2),
        "el_as_pct_of_ead": round((total_el / total_ead) * 100, 4),
        "average_pd": round(avg_pd, 4),
        "median_pd": round(median_pd, 4),
        "lgd_assumption": lgd,
        "fixed_ead_assumption": fixed_ead,
    }

    logger.info("  Portfolio: %d loans", summary["total_loans"])
    logger.info("  Total EAD: $%s", f"{summary['total_ead']:,.2f}")
    logger.info("  Total Expected Loss: $%s", f"{summary['total_expected_loss']:,.2f}")
    logger.info("  EL as %% of EAD: %.2f%%", summary["el_as_pct_of_ead"])
    logger.info("  Average PD: %.4f | Median PD: %.4f", avg_pd, median_pd)

    # ---- Segment-level breakdown ---- #
    segment_summary = el_df.groupby("risk_grade", observed=True).agg(
        loan_count=("pd_predicted", "count"),
        avg_pd=("pd_predicted", "mean"),
        total_ead=("ead", "sum"),
        total_el=("expected_loss", "sum"),
    ).reset_index()
    segment_summary["pct_of_portfolio"] = (
        segment_summary["loan_count"] / len(el_df) * 100
    ).round(2)
    segment_summary["el_pct_of_ead"] = (
        segment_summary["total_el"] / segment_summary["total_ead"] * 100
    ).round(4)

    logger.info("")
    logger.info("  Risk Grade Breakdown:")
    logger.info("  %-20s  %8s  %8s  %12s  %12s",
                "GRADE", "LOANS", "AVG PD", "TOTAL EAD", "TOTAL EL")
    logger.info("  " + "-" * 70)
    for _, row in segment_summary.iterrows():
        logger.info("  %-20s  %8d  %8.4f  $%11s  $%11s",
                     row["risk_grade"], row["loan_count"], row["avg_pd"],
                     f"{row['total_ead']:,.0f}", f"{row['total_el']:,.0f}")

    # ---- Save outputs ---- #
    # Portfolio summary JSON
    summary_path = metrics_dir / f"{dataset}_el_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved EL summary to %s", summary_path)

    # Segment breakdown CSV
    segment_path = metrics_dir / f"{dataset}_el_by_segment.csv"
    segment_summary.to_csv(segment_path, index=False)
    logger.info("Saved segment breakdown to %s", segment_path)

    # ---- Generate EL visualization ---- #
    _plot_el_by_segment(segment_summary, dataset, plots_dir)
    _plot_el_distribution(el_df, dataset, plots_dir)

    return summary


def _plot_el_by_segment(segment_df: pd.DataFrame, dataset: str, plots_dir: Path):
    """Bar chart of Expected Loss by risk grade."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#B71C1C"]

    # Left: loan count by grade
    ax1.bar(segment_df["risk_grade"], segment_df["loan_count"],
            color=colors[:len(segment_df)], edgecolor="black", lw=0.5)
    ax1.set_xlabel("Risk Grade", fontsize=12)
    ax1.set_ylabel("Number of Loans", fontsize=12)
    ax1.set_title("Loan Distribution by Risk Grade", fontsize=14)
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(alpha=0.3, axis="y")

    # Right: total EL by grade
    ax2.bar(segment_df["risk_grade"], segment_df["total_el"],
            color=colors[:len(segment_df)], edgecolor="black", lw=0.5)
    ax2.set_xlabel("Risk Grade", fontsize=12)
    ax2.set_ylabel("Total Expected Loss ($)", fontsize=12)
    ax2.set_title("Expected Loss by Risk Grade", fontsize=14)
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Basel III Risk Segmentation ({dataset.upper()})",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = plots_dir / f"{dataset}_el_by_segment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved EL by segment plot to %s", path)


def _plot_el_distribution(el_df: pd.DataFrame, dataset: str, plots_dir: Path):
    """Histogram of loan-level Expected Loss values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(el_df["expected_loss"], bins=50, color="#2196F3",
            alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(el_df["expected_loss"].mean(), color="#FF5722", lw=2,
               linestyle="--", label=f"Mean EL = ${el_df['expected_loss'].mean():,.0f}")
    ax.axvline(el_df["expected_loss"].median(), color="#4CAF50", lw=2,
               linestyle="--", label=f"Median EL = ${el_df['expected_loss'].median():,.0f}")

    ax.set_xlabel("Expected Loss per Loan ($)", fontsize=12)
    ax.set_ylabel("Number of Loans", fontsize=12)
    ax.set_title(f"Expected Loss Distribution ({dataset.upper()})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    path = plots_dir / f"{dataset}_el_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved EL distribution plot to %s", path)
