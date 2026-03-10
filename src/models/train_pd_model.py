"""
train_pd_model.py — Probability of Default Model Training Pipeline
===================================================================
WHY THIS EXISTS:
    This is the core ML engine. It trains 4 models on the feature-
    engineered data from Layer 1, handles class imbalance with SMOTE,
    and exports the best model for downstream use in Basel III EL
    calculation, stress testing, and the Streamlit dashboard.

WHAT IT PRODUCES:
    - Trained models saved to models/
    - Training metrics saved to artifacts/metrics/
    - The best model is selected by AUC-ROC on the test set

HOW TO RUN:
    python src/models/train_pd_model.py --dataset gmsc
    python src/models/train_pd_model.py --dataset lc
"""

import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Suppress convergence warnings during grid search
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Project paths ---- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.config import Config

logger = logging.getLogger("models.train")

# ============================================================
#  CONSTANTS
# ============================================================

# GMSC target column
GMSC_TARGET = "SeriousDlqin2yrs"

# LendingClub target column
LC_TARGET = "is_default"

# Columns to DROP before training (identifiers, leaky, or non-numeric)
DROP_COLS = [
    "loan_id", "row_id", "Unnamed: 0",
    # Leaky or redundant
    "issue_d", "issue_year", "issue_quarter",
    # Bucket columns (categorical strings, we keep the numeric originals)
    "RevolvingUtilizationOfUnsecuredLines_bucket",
    "revol_util_bucket", "dti_bucket",
    # Already encoded
    "grade", "emp_length", "home_ownership", "purpose",
]

# Random seed for reproducibility
RANDOM_STATE = 42


# ============================================================
#  DATA LOADING & PREPARATION
# ============================================================

def load_feature_data(dataset: str) -> tuple[pd.DataFrame, str]:
    """
    Load the feature-engineered parquet from Layer 1's feature store.

    Args:
        dataset: 'gmsc' or 'lc'

    Returns:
        (DataFrame, target_column_name)
    """
    if dataset == "gmsc":
        path = Config.FEATURE_STORE_DIR / "gmsc_features.parquet"
        target = GMSC_TARGET
    else:
        path = Config.FEATURE_STORE_DIR / "lc_features.parquet"
        target = LC_TARGET

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            f"Run Layer 1 first: python src/ingestion/run_pipeline.py --dataset {dataset}"
        )

    df = pd.read_parquet(path)
    logger.info("Loaded feature data: %s (%d rows x %d cols)",
                path.name, len(df), len(df.columns))
    return df, target


def prepare_data(df: pd.DataFrame, target: str) -> dict:
    """
    Prepare data for model training.

    Steps:
        1. Separate features (X) and target (y)
        2. Drop non-numeric and identifier columns
        3. Handle remaining missing values
        4. Stratified train/validation/test split (70/15/15)
        5. Scale features with StandardScaler
        6. Apply SMOTE to training set only

    Args:
        df: Feature-engineered DataFrame from Layer 1
        target: Name of the target column

    Returns:
        dict with X_train, X_val, X_test, y_train, y_val, y_test,
             scaler, feature_names, and class distribution info
    """
    logger.info("Preparing data for training...")

    # ---- Separate target ---- #
    y = df[target].astype(int)
    X = df.drop(columns=[target], errors="ignore")

    # ---- Drop non-feature columns ---- #
    cols_to_drop = [c for c in DROP_COLS if c in X.columns]
    X = X.drop(columns=cols_to_drop, errors="ignore")

    # ---- Drop non-numeric columns (categorical strings left over) ---- #
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.info("Dropping non-numeric columns: %s", non_numeric)
        X = X.drop(columns=non_numeric)

    # ---- Handle remaining NaN/inf ---- #
    X = X.replace([np.inf, -np.inf], np.nan)
    null_counts = X.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        logger.info("Filling %d columns with nulls using median",
                     len(cols_with_nulls))
        X = X.fillna(X.median())

    feature_names = X.columns.tolist()
    logger.info("Feature matrix: %d rows x %d features", len(X), len(feature_names))
    logger.info("Target distribution: %.1f%% default, %.1f%% non-default",
                y.mean() * 100, (1 - y.mean()) * 100)

    # ---- Stratified split: 70% train, 15% val, 15% test ---- #
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765,  # 0.1765 of 0.85 ≈ 0.15 of total
        stratify=y_temp, random_state=RANDOM_STATE
    )

    logger.info("Split sizes — Train: %d | Val: %d | Test: %d",
                len(X_train), len(X_val), len(X_test))

    # ---- Scale features ---- #
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_names, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names, index=X_test.index
    )

    logger.info("Features scaled with StandardScaler")

    # ---- SMOTE on training set only ---- #
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_scaled, y_train
        )
        logger.info("SMOTE applied — Train: %d -> %d rows (balanced)",
                     len(X_train), len(X_train_resampled))
    except ImportError:
        logger.warning("imbalanced-learn not installed — using class_weight instead")
        X_train_resampled = X_train_scaled
        y_train_resampled = y_train

    return {
        "X_train": X_train_resampled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_resampled,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "default_rate": y.mean(),
    }


# ============================================================
#  MODEL DEFINITIONS
# ============================================================

def get_models() -> dict:
    """
    Define the 4 candidate models.

    Returns a dict of {name: model_instance}.
    Each model uses class_weight='balanced' as an additional safeguard
    against class imbalance (on top of SMOTE).
    """
    models = {}

    # 1. Logistic Regression — interpretable baseline
    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
        n_jobs=-1,
    )

    # 2. Random Forest — ensemble baseline
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 3. XGBoost — expected best performer
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # SMOTE already balances
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost")

    # 4. LightGBM — fast alternative
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm not installed — skipping LightGBM")

    return models


# ============================================================
#  TRAINING LOOP
# ============================================================

def train_all_models(data: dict) -> dict:
    """
    Train all candidate models and return results.

    Args:
        data: dict from prepare_data()

    Returns:
        dict of {model_name: {"model": fitted_model, "train_time": seconds}}
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        start = time.time()

        # XGBoost/LightGBM support early stopping with a validation set
        model_type = type(model).__name__
        if model_type == "XGBClassifier":
            model.fit(
                data["X_train"], data["y_train"],
                eval_set=[(data["X_val"], data["y_val"])],
                verbose=False,
            )
            if hasattr(model, "best_iteration"):
                logger.info("  Early stopped at iteration %d", model.best_iteration)
        elif model_type == "LGBMClassifier":
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                data["X_train"], data["y_train"],
                eval_set=[(data["X_val"], data["y_val"])],
                callbacks=[early_stopping(30, verbose=False), log_evaluation(period=0)],
            )
            if hasattr(model, "best_iteration_"):
                logger.info("  Early stopped at iteration %d", model.best_iteration_)
        else:
            model.fit(data["X_train"], data["y_train"])

        train_time = time.time() - start
        results[name] = {
            "model": model,
            "train_time": train_time,
        }
        logger.info("  %s trained in %.1f seconds", name, train_time)

    return results


# ============================================================
#  MAIN PIPELINE
# ============================================================

def run_training_pipeline(dataset: str) -> dict:
    """
    Execute the full training pipeline.

    Args:
        dataset: 'gmsc' or 'lc'

    Returns:
        dict with trained models, data splits, and metadata
    """
    logger.info("=" * 70)
    logger.info("  LAYER 2: PD Model Training Pipeline")
    logger.info("  Dataset: %s", dataset.upper())
    logger.info("=" * 70)

    start_time = time.time()

    # Step 1: Load data
    df, target = load_feature_data(dataset)

    # Step 2: Prepare data
    data = prepare_data(df, target)

    # Step 3: Train models
    results = train_all_models(data)

    # Step 4: Evaluate (imported from evaluate.py)
    from src.models.evaluate import evaluate_all_models, save_metrics, generate_all_plots
    metrics = evaluate_all_models(results, data)
    save_metrics(metrics, dataset)

    # Step 4b: Find best model early so we can pass it to plot generation
    best_name = max(metrics, key=lambda k: metrics[k]["auc_roc"])
    best_model = results[best_name]["model"]
    best_auc = metrics[best_name]["auc_roc"]

    # Step 4c: Generate all evaluation plots (ROC, PR, KS, CM, PD dist)
    generate_all_plots(results, data, dataset, best_model, best_name)

    logger.info("=" * 70)
    logger.info("  BEST MODEL: %s (AUC-ROC = %.4f)", best_name, best_auc)
    logger.info("=" * 70)

    # Step 6: Save best model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{dataset}_pd_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Saved best model to %s", model_path)

    # Save scaler too
    scaler_path = model_dir / f"{dataset}_scaler.pkl"
    joblib.dump(data["scaler"], scaler_path)
    logger.info("Saved scaler to %s", scaler_path)

    # Save feature names
    features_path = model_dir / f"{dataset}_feature_names.json"
    with open(features_path, "w") as f:
        json.dump(data["feature_names"], f, indent=2)
    logger.info("Saved feature names to %s", features_path)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  TRAINING PIPELINE COMPLETE — %.1f seconds", elapsed)
    logger.info("=" * 70)

    return {
        "results": results,
        "metrics": metrics,
        "data": data,
        "best_model_name": best_name,
        "best_model": best_model,
        "dataset": dataset,
    }


# ============================================================
#  CLI ENTRY POINT
# ============================================================

def main():
    """CLI entry point."""
    import io
    # Force UTF-8 output on Windows
    handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    handler.setFormatter(logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    parser = argparse.ArgumentParser(
        description="Train PD models on feature-engineered data"
    )
    parser.add_argument(
        "--dataset", choices=["gmsc", "lc"], default="gmsc",
        help="Which dataset to train on"
    )
    args = parser.parse_args()

    pipeline_output = run_training_pipeline(args.dataset)

    # Step 7: SHAP explainability
    logger.info("Running SHAP explainability...")
    from src.explainability.shap_explainer import run_shap_analysis
    run_shap_analysis(
        model=pipeline_output["best_model"],
        X_test=pipeline_output["data"]["X_test"],
        feature_names=pipeline_output["data"]["feature_names"],
        dataset=args.dataset,
        model_name=pipeline_output["best_model_name"],
    )

    # Step 8: Basel III Expected Loss
    logger.info("Computing Basel III Expected Loss...")
    from src.risk_engine.expected_loss import compute_portfolio_el
    compute_portfolio_el(
        model=pipeline_output["best_model"],
        X=pipeline_output["data"]["X_test"],
        feature_names=pipeline_output["data"]["feature_names"],
        dataset=args.dataset,
    )

    logger.info("=" * 70)
    logger.info("  LAYER 2 COMPLETE — All outputs saved to artifacts/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
