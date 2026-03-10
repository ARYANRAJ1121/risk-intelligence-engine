"""
train_dl.py — Deep Learning Training Pipeline (Autoencoder Anomaly Detection)
===========================================================================
This script takes the processed Layer 1 feature store, specifically the 
LendingClub dataset, and trains the Ensemble Anomaly Detector to flag 
highly unusual/suspicious credit applications.
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.config import Config
from src.models.dl_anomaly import EnsembleAnomalyDetector

logger = logging.getLogger("models.train_dl")

# Reusing same dropped columns from train_pd_model
DROP_COLS = [
    "loan_id", "row_id", "Unnamed: 0",
    "issue_d", "issue_year", "issue_quarter",
    "RevolvingUtilizationOfUnsecuredLines_bucket",
    "revol_util_bucket", "dti_bucket",
    "grade", "emp_length", "home_ownership", "purpose",
]


def setup_logging() -> None:
    import io
    handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    handler.setFormatter(logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def load_and_prepare_data(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load parquet data, split features, and handle simple typing issues."""
    target = "is_default" if dataset == "lc" else "SeriousDlqin2yrs"
    path = Config.FEATURE_STORE_DIR / f"{dataset}_features.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
        
    df = pd.read_parquet(path)
    logger.info(f"Loaded {path.name} | {len(df)} rows, {len(df.columns)} cols")
    
    y = df[target].astype(int)
    X = df.drop(columns=[target], errors="ignore")
    
    # Drop leaky/unusable columns
    cols_to_drop = [c for c in DROP_COLS if c in X.columns]
    X = X.drop(columns=cols_to_drop, errors="ignore")
    
    # Drop any remaining string categoricals
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        
    # Handle NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X, y


def test_autoencoder_anomaly(dataset: str):
    logger.info("=" * 70)
    logger.info(f"  LAYER 3: DEEP LEARNING (Anomaly Detection on {dataset.upper()})")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # 1. Load the features
    X, y = load_and_prepare_data(dataset)
    
    # 2. Extract ONLY "Healthy" loans to train the Autoencoder
    #    (If y=0, the loan was fully paid/healthy)
    X_healthy = X[y == 0].copy()
    logger.info(f"Filtered to {len(X_healthy)} completely healthy loans for training.")
    
    # Since LendingClub data is huge (1.3M rows), let's subsample for this demonstration
    # to train the PyTorch model within a few seconds instead of hours.
    num_train_samples = min(20000, len(X_healthy))
    X_train_healthy = X_healthy.sample(n=num_train_samples, random_state=42)
    
    # 3. Initialize the Ensemble Detector (assuming 1% contamination)
    detector = EnsembleAnomalyDetector(contamination=0.01)
    
    # 4. Train the Detector 
    #    The Autoencoder will squeeze the data into 8 dimensions and reconstruct it over 30 epochs
    detector.fit(X_train_healthy)
    
    # 5. Score the entire original dataset
    #    We pass a slightly larger sample for evaluation output to prevent out-of-memory errors on terminal viewing
    num_eval_samples = min(50000, len(X))
    X_eval = X.sample(n=num_eval_samples, random_state=42)
    y_eval = y.loc[X_eval.index]
    
    logger.info(f"\nScoring {len(X_eval)} loans for Anomalies & Fraud...")
    anomaly_results = detector.predict(X_eval)
    
    # Combine original features with the new anomaly scores
    results_df = X_eval.copy()
    results_df["is_default_target"] = y_eval
    results_df = pd.concat([results_df, anomaly_results], axis=1)
    
    # 6. Evaluate findings
    num_iso = anomaly_results['is_iso_anomaly'].sum()
    num_ae = anomaly_results['is_ae_anomaly'].sum()
    num_ensemble = anomaly_results['strong_anomaly_flag'].sum()
    
    logger.info("\n--- ENSEMBLE ANOMALY RESULTS ---")
    logger.info(f"Total Evaluated: {len(X_eval)}")
    logger.info(f"Caught by Isolation Forest (Rule-Based): {num_iso}")
    logger.info(f"Caught by Autoencoder (Hidden Maths): {num_ae}")
    logger.info(f"Caught by BOTH (STRONG FLAG): {num_ensemble}")
    
    # Analysis: Are anomalies more likely to default?
    strong_anomalies = results_df[results_df["strong_anomaly_flag"] == True]
    normal_loans = results_df[results_df["strong_anomaly_flag"] == False]
    
    if len(strong_anomalies) > 0 and len(normal_loans) > 0:
        anom_default_rate = strong_anomalies['is_default_target'].mean() * 100
        norm_default_rate = normal_loans['is_default_target'].mean() * 100
        
        logger.info(f"\n--- BUSINESS VALUE ---")
        logger.info(f"Default rate for Normal Loans: {norm_default_rate:.2f}%")
        logger.info(f"Default rate for Anomalous Loans: {anom_default_rate:.2f}%")
        
        if anom_default_rate > norm_default_rate:
            ratio = anom_default_rate / max(norm_default_rate, 0.001)
            logger.info(f"-> Anomalies default at a rate {ratio:.1f}x higher than normal loans!")
            logger.info("   *This proves Deep Learning finds hidden risk invisible to standard credit scoring.*")

    # 7. Distribution Plot
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    defaults = results_df[results_df["is_default_target"] == 1]
    healthy = results_df[results_df["is_default_target"] == 0]
    
    # We plot the log of the reconstruction error so we can read the scale easily
    ax.hist(np.log1p(healthy["reconstruction_error"]), bins=50, alpha=0.5, label="Healthy Loans", density=True, color="#4CAF50")
    ax.hist(np.log1p(defaults["reconstruction_error"]), bins=50, alpha=0.5, label="Defaulted Loans", density=True, color="#FF5722")
    
    # Add anomaly threshold line
    threshold = np.percentile(results_df["reconstruction_error"], 99)
    ax.axvline(np.log1p(threshold), color="red", linestyle="--", label="Top 1% Anomaly Cutoff")
    
    ax.set_title(f"Autoencoder Reconstruction Error ({dataset.upper()})", fontsize=14)
    ax.set_xlabel("Log1P(Reconstruction Error)  --> Right Side = Weird/Anomalous", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    
    plot_path = plots_dir / f"{dataset}_anomaly_reconstruction.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\nSaved Reconstruction Error plot to {plot_path}")
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"  DL ANOMALY DETECTOR COMPLETE — {elapsed:.1f} seconds")
    logger.info("=" * 70)


def test_lstm_sequence(dataset: str):
    from src.models.lstm_sequence import create_sequences, LSTMDetector
    logger.info("=" * 70)
    logger.info(f"  LAYER 3: DEEP LEARNING (LSTM Sequences on {dataset.upper()})")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # 1. Load basic features to simulate temporal data
    X_flat, y_flat = load_and_prepare_data(dataset)
    
    # We will slice a small portion to represent 3 months of historical data per loan
    # In a real environment, this data comes from Freddie Mac monthly performance logs
    sample_size = min(15000, len(X_flat))
    X_sample = X_flat.head(sample_size)
    
    logger.info(f"Converting flat data into 3D Temporal Sequences [Batch, Seq, Features]...")
    seq_length = 3
    X_3d, y_seq = create_sequences(X_sample, target_col="target_dummy", seq_length=seq_length)
    
    logger.info(f"Generated {len(X_3d)} sequences. Shape: {X_3d.shape}")
    
    # 2. Train the LSTM model
    logger.info("\nInitializing LSTM (Hidden Dim=64, 2 Layers)...")
    lstm_model = LSTMDetector(input_dim=X_3d.shape[2], epochs=20)
    
    # Fit the network
    lstm_model.fit(X_3d, y_seq)
    
    # 3. Output Predictions
    logger.info("\nPredicting Probability of Default based on Temporal Sequences...")
    pd_predictions = lstm_model.predict_proba(X_3d)
    
    logger.info(f"Sample Predictions (First 5 Sequence PDs):")
    for i in range(min(5, len(pd_predictions))):
        logger.info(f"  Sequence {i}: PD = {pd_predictions[i][0]:.4f}")
        
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"  DL LSTM SEQUENCE COMPLETE — {elapsed:.1f} seconds")
    logger.info("=" * 70)

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Test PyTorch Deep Learning Models")
    parser.add_argument("--dataset", choices=["gmsc", "lc"], default="lc", help="Dataset to test")
    parser.add_argument("--model", choices=["autoencoder", "lstm", "both"], default="both", help="Which DL model to run")
    args = parser.parse_args()
    
    if args.model in ["autoencoder", "both"]:
        test_autoencoder_anomaly(args.dataset)
        
    if args.model in ["lstm", "both"]:
        test_lstm_sequence(args.dataset)
