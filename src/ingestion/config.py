"""
config.py — Centralized Configuration for the Risk Intelligence Engine
=======================================================================
WHY THIS EXISTS:
    In a Big 4 consulting engagement, hardcoded paths, thresholds, and
    credentials scattered across scripts are a compliance violation.
    Auditors expect ONE place to see every configurable parameter.
    This file is that single source of truth.

WHAT IT CONTROLS:
    - File paths (raw data, processed output, models, reports)
    - Database connection parameters (PostgreSQL or Parquet fallback)
    - FRED API configuration
    - Feature engineering parameters (column lists, thresholds)
    - Model training hyperparameters (added in Layer 2)

HOW IT WORKS:
    Import `from src.ingestion.config import Config` anywhere in the
    codebase. All values are class-level constants — no instantiation needed.
"""

import os
from pathlib import Path


class Config:
    """
    Central configuration class for the Credit Risk Intelligence Platform.

    All paths are resolved relative to the project root directory,
    making the pipeline portable across machines and environments.
    """

    # ------------------------------------------------------------------ #
    #  PROJECT ROOT — automatically resolves from this file's location
    # ------------------------------------------------------------------ #
    _THIS_DIR = Path(__file__).resolve().parent          # src/ingestion/
    PROJECT_ROOT = _THIS_DIR.parent.parent               # risk-intelligence-engine/

    # ------------------------------------------------------------------ #
    #  DATA DIRECTORIES
    # ------------------------------------------------------------------ #
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    # ------------------------------------------------------------------ #
    #  RAW DATASET FILENAMES (drop CSVs here)
    # ------------------------------------------------------------------ #
    GMSC_TRAIN_FILE = RAW_DATA_DIR / "cs-training.csv"          # Give Me Some Credit
    GMSC_TEST_FILE = RAW_DATA_DIR / "cs-test.csv"
    LENDING_CLUB_FILE = RAW_DATA_DIR / "accepted_2007_to_2018Q4.csv"  # LendingClub
    FREDDIE_MAC_DIR = RAW_DATA_DIR / "freddie_mac"              # Freddie Mac monthly files

    # ------------------------------------------------------------------ #
    #  PROCESSED OUTPUT PATHS
    # ------------------------------------------------------------------ #
    FEATURE_STORE_DIR = PROCESSED_DATA_DIR / "feature_store"
    GMSC_FEATURES_FILE = FEATURE_STORE_DIR / "gmsc_features.parquet"
    LC_FEATURES_FILE = FEATURE_STORE_DIR / "lc_features.parquet"
    MACRO_DATA_FILE = EXTERNAL_DATA_DIR / "fred_macro_quarterly.csv"

    # ------------------------------------------------------------------ #
    #  MODEL ARTIFACTS
    # ------------------------------------------------------------------ #
    MODEL_DIR = PROJECT_ROOT / "models"
    BEST_MODEL_PATH = MODEL_DIR / "best_pd_model.pkl"
    MODEL_CARD_PATH = MODEL_DIR / "model_card.json"
    LSTM_MODEL_PATH = MODEL_DIR / "lstm_deterioration.h5"
    AUTOENCODER_PATH = MODEL_DIR / "autoencoder_anomaly.h5"

    # ------------------------------------------------------------------ #
    #  REPORTS
    # ------------------------------------------------------------------ #
    REPORTS_DIR = PROJECT_ROOT / "reports"

    # ------------------------------------------------------------------ #
    #  DATABASE CONFIGURATION
    # ------------------------------------------------------------------ #
    # Set USE_POSTGRES = True and fill in credentials to write to PostgreSQL.
    # Otherwise the pipeline writes Parquet files (zero-install fallback).
    USE_POSTGRES = False
    POSTGRES_HOST = os.getenv("PG_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("PG_PORT", "5432"))
    POSTGRES_DB = os.getenv("PG_DB", "risk_intelligence")
    POSTGRES_USER = os.getenv("PG_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("PG_PASSWORD", "")

    @classmethod
    def get_postgres_uri(cls) -> str:
        """
        Build a SQLAlchemy-compatible PostgreSQL connection string.
        Returns: 'postgresql://user:pass@host:port/dbname'
        """
        return (
            f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )

    # ------------------------------------------------------------------ #
    #  FRED API CONFIGURATION
    # ------------------------------------------------------------------ #
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    FRED_SERIES = {
        "UNRATE": "Unemployment Rate",
        "GDP": "GDP Growth",
        "FEDFUNDS": "Federal Funds Rate",
        "DRCCLACBS": "Credit Card Delinquency Rate",
    }

    # ------------------------------------------------------------------ #
    #  GIVE ME SOME CREDIT — COLUMN DEFINITIONS
    # ------------------------------------------------------------------ #
    GMSC_TARGET_COL = "SeriousDlqin2yrs"
    GMSC_ID_COL = "Unnamed: 0"  # row index in the Kaggle CSV
    GMSC_FEATURE_COLS = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]

    # Columns to log-transform (right-skewed distributions)
    GMSC_LOG_TRANSFORM_COLS = [
        "MonthlyIncome",
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
    ]

    # ------------------------------------------------------------------ #
    #  LENDINGCLUB — COLUMN DEFINITIONS
    # ------------------------------------------------------------------ #
    LC_TARGET_COL = "loan_status"
    LC_TARGET_POSITIVE_LABELS = ["Charged Off", "Default"]
    LC_TARGET_NEGATIVE_LABELS = ["Fully Paid"]
    LC_KEY_FEATURES = [
        "loan_amnt", "int_rate", "annual_inc", "dti", "grade",
        "home_ownership", "delinq_2yrs", "revol_util", "open_acc",
        "pub_rec", "revol_bal", "installment", "sub_grade", "term",
        "emp_length", "purpose", "addr_state", "issue_d",
        "verification_status",
    ]
    LC_LOG_TRANSFORM_COLS = ["annual_inc", "revol_bal", "loan_amnt"]

    # ------------------------------------------------------------------ #
    #  FEATURE ENGINEERING THRESHOLDS
    # ------------------------------------------------------------------ #
    DELINQUENCY_30DPD_THRESHOLD = 1   # >= 1 occurrence  →  flag = 1
    DELINQUENCY_60DPD_THRESHOLD = 1
    DELINQUENCY_90DPD_THRESHOLD = 1

    # ------------------------------------------------------------------ #
    #  SPARK CONFIGURATION
    # ------------------------------------------------------------------ #
    SPARK_APP_NAME = "RiskIntelligenceEngine"
    SPARK_MASTER = "local[*]"   # use all available cores
    SPARK_DRIVER_MEMORY = "4g"

    # ------------------------------------------------------------------ #
    #  LOGGING
    # ------------------------------------------------------------------ #
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
