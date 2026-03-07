import sys
import time
import logging
import argparse
from pathlib import Path

# Ensure src/ is on the Python path regardless of where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.config import Config
from src.ingestion.ingest import get_spark_session, ingest_gmsc, ingest_lending_club
from src.features.feature_engineer import engineer_gmsc_features, engineer_lc_features
from src.features.macro_enrichment import fetch_fred_data, join_macro_to_loans
from src.ingestion.db_writer import write_feature_store


def setup_logging() -> None:
    """
    Configure structured logging for the pipeline.

    All pipeline stages log to both console and a timestamped log file
    in the project's reports/ directory. This creates an audit trail
    that can be attached to engagement deliverables.
    """
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_gmsc_pipeline() -> None:
    """
    Execute the full data pipeline for the Give Me Some Credit dataset.

    This is the PROTOTYPING pipeline — run this first to validate the
    entire flow before scaling to LendingClub. Uses pandas internally
    (150K rows fits comfortably in memory) but demonstrates the same
    pipeline stages as production.

    Pipeline stages:
        1. PySpark ingestion + validation → Spark DataFrame
        2. Convert to pandas (small dataset — pandas is faster for feature eng)
        3. Feature engineering (flags, buckets, log-transforms)
        4. Macro enrichment (FRED quarterly join)
        5. Write to feature store (Parquet)
    """
    logger = logging.getLogger("pipeline.gmsc")
    logger.info("=" * 70)
    logger.info("  PIPELINE: Give Me Some Credit (Prototyping)")
    logger.info("=" * 70)

    start_time = time.time()

    # ---- Step 1: PySpark Ingestion ---- #
    logger.info("STAGE 1/5: Ingesting raw data via PySpark...")
    spark = get_spark_session()

    try:
        spark_df = ingest_gmsc(spark)
    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.error("Please download the Give Me Some Credit dataset from:")
        logger.error("  https://www.kaggle.com/competitions/GiveMeSomeCredit")
        logger.error("Place 'cs-training.csv' in: %s", Config.RAW_DATA_DIR)
        spark.stop()
        return

    # ---- Step 2: Convert to pandas ---- #
    logger.info("STAGE 2/5: Converting to pandas for feature engineering...")
    pdf = spark_df.toPandas()
    logger.info("Converted to pandas DataFrame: %d rows × %d cols",
                len(pdf), len(pdf.columns))

    # Rename loan_id if present
    if "loan_id" not in pdf.columns and "row_id" in pdf.columns:
        pdf = pdf.rename(columns={"row_id": "loan_id"})

    # ---- Step 3: Feature Engineering ---- #
    logger.info("STAGE 3/5: Engineering features...")
    pdf = engineer_gmsc_features(pdf)

    # ---- Step 4: Macro Enrichment ---- #
    logger.info("STAGE 4/5: Enriching with macroeconomic data...")
    macro_df = fetch_fred_data(save_path=str(Config.MACRO_DATA_FILE))
    pdf = join_macro_to_loans(pdf, macro_df)

    # ---- Step 5: Write to Feature Store ---- #
    logger.info("STAGE 5/5: Writing to feature store...")
    write_feature_store(pdf, dataset_name="gmsc")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  GMSC PIPELINE COMPLETE — %.1f seconds", elapsed)
    logger.info("  Output: %s", Config.FEATURE_STORE_DIR)
    logger.info("  Rows: %d | Columns: %d", len(pdf), len(pdf.columns))
    logger.info("=" * 70)

    spark.stop()


def run_lc_pipeline() -> None:
    """
    Execute the full data pipeline for the LendingClub dataset.

    This is the PRODUCTION pipeline — 2.2M rows. PySpark handles the
    heavy lifting for ingestion and validation. Feature engineering
    converts to pandas after initial filtering reduces row count.

    Pipeline stages:
        1. PySpark ingestion + validation + target encoding
        2. Convert to pandas (after filtering to labeled rows only)
        3. Feature engineering (date parsing, grade encoding, flags, transforms)
        4. Macro enrichment (FRED quarterly join on origination date)
        5. Write to feature store (Parquet)
    """
    logger = logging.getLogger("pipeline.lc")
    logger.info("=" * 70)
    logger.info("  PIPELINE: LendingClub (Production Scale)")
    logger.info("=" * 70)

    start_time = time.time()

    # ---- Step 1: PySpark Ingestion ---- #
    logger.info("STAGE 1/5: Ingesting raw data via PySpark...")
    spark = get_spark_session()

    try:
        spark_df = ingest_lending_club(spark)
    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.error("Please download the LendingClub dataset from:")
        logger.error("  https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        logger.error("Place 'accepted_2007_to_2018Q4.csv' in: %s", Config.RAW_DATA_DIR)
        spark.stop()
        return

    # ---- Step 2: Convert to pandas ---- #
    logger.info("STAGE 2/5: Converting to pandas for feature engineering...")
    pdf = spark_df.toPandas()
    logger.info("Converted to pandas DataFrame: %d rows × %d cols",
                len(pdf), len(pdf.columns))

    # ---- Step 3: Feature Engineering ---- #
    logger.info("STAGE 3/5: Engineering features...")
    pdf = engineer_lc_features(pdf)

    # ---- Step 4: Macro Enrichment ---- #
    logger.info("STAGE 4/5: Enriching with macroeconomic data...")
    macro_df = fetch_fred_data(save_path=str(Config.MACRO_DATA_FILE))
    pdf = join_macro_to_loans(pdf, macro_df)

    # ---- Step 5: Write to Feature Store ---- #
    logger.info("STAGE 5/5: Writing to feature store...")
    write_feature_store(pdf, dataset_name="lc")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  LENDINGCLUB PIPELINE COMPLETE — %.1f seconds", elapsed)
    logger.info("  Output: %s", Config.FEATURE_STORE_DIR)
    logger.info("  Rows: %d | Columns: %d", len(pdf), len(pdf.columns))
    logger.info("=" * 70)

    spark.stop()


def main():
    """
    CLI entry point for the data engineering pipeline.

    Usage:
        python src/ingestion/run_pipeline.py --dataset gmsc    # prototyping
        python src/ingestion/run_pipeline.py --dataset lc      # production
        python src/ingestion/run_pipeline.py --dataset all     # both
    """
    parser = argparse.ArgumentParser(
        description="Risk Intelligence Engine — Data Engineering Pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["gmsc", "lc", "all"],
        default="gmsc",
        help="Which dataset to process: 'gmsc' (prototyping), 'lc' (production), 'all'"
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("pipeline")
    logger.info("Risk Intelligence Engine — Data Pipeline v1.0")
    logger.info("Dataset: %s", args.dataset)

    if args.dataset in ("gmsc", "all"):
        run_gmsc_pipeline()

    if args.dataset in ("lc", "all"):
        run_lc_pipeline()

    logger.info("All pipelines complete.")


if __name__ == "__main__":
    main()
