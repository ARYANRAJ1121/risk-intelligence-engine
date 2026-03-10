"""
shap_explainer.py — SHAP Model Explainability Module
=====================================================
WHY THIS EXISTS:
    Regulators (Fed SR 11-7, GDPR Art. 22, RBI guidelines) require
    that credit scoring models be explainable. SHAP assigns each
    feature a contribution score for every individual prediction.

WHAT IT PRODUCES:
    - Beeswarm plot (global feature importance)
    - Waterfall plot (single-loan explanation)
    - Feature importance table (CSV + JSON)
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
logger = logging.getLogger("explainability.shap")


def run_shap_analysis(
    model,
    X_test: pd.DataFrame,
    feature_names: list[str],
    dataset: str,
    model_name: str,
    n_samples: int = 500,
) -> dict:
    """
    Run SHAP analysis on the best model.

    Args:
        model: Trained sklearn/XGBoost/LightGBM model
        X_test: Test feature matrix
        feature_names: List of feature column names
        dataset: 'gmsc' or 'lc'
        model_name: Name of the model (e.g., 'XGBoost')
        n_samples: Number of samples for SHAP (more = slower but more accurate)

    Returns:
        dict with feature importance rankings
    """
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for speed
    if len(X_test) > n_samples:
        X_sample = X_test.sample(n=n_samples, random_state=42)
    else:
        X_sample = X_test

    logger.info("Running SHAP on %d samples with %s...", len(X_sample), model_name)

    try:
        import shap

        # Choose the right explainer based on model type
        model_type = type(model).__name__
        if model_type in ("XGBClassifier", "LGBMClassifier"):
            explainer = shap.TreeExplainer(model)
        elif model_type == "RandomForestClassifier":
            explainer = shap.TreeExplainer(model)
        else:
            # Logistic Regression — use LinearExplainer or KernelExplainer
            explainer = shap.LinearExplainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take class 1 (default)
        # Or it might be a 3D numpy array (n_samples, n_features, n_classes)
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        # ---- Beeswarm Plot ---- #
        logger.info("Generating SHAP beeswarm plot...")
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            show=False, plot_size=(12, 8),
        )
        path_beeswarm = plots_dir / f"{dataset}_shap_beeswarm.png"
        plt.savefig(path_beeswarm, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved beeswarm plot to %s", path_beeswarm)

        # ---- Waterfall Plot (single loan) ---- #
        logger.info("Generating SHAP waterfall plot (loan index 0)...")
        try:
            # Create Explanation object for waterfall
            if hasattr(explainer, "expected_value"):
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            else:
                base_value = 0

            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=base_value,
                data=X_sample.iloc[0].values,
                feature_names=feature_names,
            )

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.waterfall(explanation, show=False)
            path_waterfall = plots_dir / f"{dataset}_shap_waterfall.png"
            plt.savefig(path_waterfall, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("Saved waterfall plot to %s", path_waterfall)
        except Exception as e:
            logger.warning("Waterfall plot failed (non-critical): %s", e)

        # ---- Feature Importance Table ---- #
        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": importance,
        }).sort_values("mean_abs_shap", ascending=False)

        # Save as CSV
        csv_path = metrics_dir / f"{dataset}_shap_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        logger.info("Saved feature importance to %s", csv_path)

        # Save as JSON
        importance_dict = {
            row["feature"]: round(row["mean_abs_shap"], 6)
            for _, row in importance_df.iterrows()
        }
        json_path = metrics_dir / f"{dataset}_shap_importance.json"
        with open(json_path, "w") as f:
            json.dump(importance_dict, f, indent=2)

        # ---- Bar Plot (top 15 features) ---- #
        logger.info("Generating SHAP bar chart (top 15)...")
        top_n = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            top_n["feature"][::-1],
            top_n["mean_abs_shap"][::-1],
            color="#2196F3", edgecolor="black", lw=0.5,
        )
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title(f"Top 15 Features — {model_name} ({dataset.upper()})", fontsize=14)
        ax.grid(alpha=0.3, axis="x")

        path_bar = plots_dir / f"{dataset}_shap_bar.png"
        fig.savefig(path_bar, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP bar chart to %s", path_bar)

        return importance_dict

    except ImportError:
        logger.warning("shap not installed — generating sklearn feature importance instead")
        return _fallback_importance(model, feature_names, dataset, metrics_dir, plots_dir)


def _fallback_importance(model, feature_names, dataset, metrics_dir, plots_dir) -> dict:
    """
    Fallback: use sklearn's built-in feature importance when SHAP isn't available.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        logger.warning("Model has no feature_importances_ or coef_")
        return {}

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    # Save
    csv_path = metrics_dir / f"{dataset}_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # Plot
    top_n = importance_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_n["feature"][::-1], top_n["importance"][::-1],
            color="#2196F3", edgecolor="black", lw=0.5)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top 15 Features (sklearn) — {dataset.upper()}", fontsize=14)
    ax.grid(alpha=0.3, axis="x")

    path = plots_dir / f"{dataset}_feature_importance_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fallback feature importance to %s", path)

    return {row["feature"]: round(row["importance"], 6) for _, row in importance_df.iterrows()}
