"""
ingest.py — PySpark Data Ingestion & Validation Module
=======================================================
WHY THIS EXISTS:
    Raw data from Kaggle, Freddie Mac, and FRED arrives in inconsistent
    formats — mixed types, nulls in critical columns, duplicate rows.
    This module is the FIRST gate in the pipeline. It loads CSV files
    into PySpark DataFrames with enforced schemas, validates data quality,
    handles nulls with documented strategies, and removes duplicates.

    In a Big 4 engagement, this is the 'Data Quality Assessment' phase
    that precedes any modeling work. The output is a clean, validated
    DataFrame that downstream modules can trust.

WHAT IT RETURNS:
    Clean PySpark DataFrames ready for feature engineering.

HOW IT WORKS:
    1. Initialize a local SparkSession
    2. Read CSV with enforced schema (rejects type mismatches)
    3. Drop rows where the TARGET column is null (can't train on unknowns)
    4. Impute numeric nulls with column medians
    5. Remove exact-duplicate rows
    6. Log row counts at each stage for auditability
"""

import logging
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from src.ingestion.config import Config
from src.ingestion.schema import get_gmsc_schema, get_lending_club_schema

logger = logging.getLogger(__name__)


def get_spark_session() -> SparkSession:
    """
    Initialize and return a PySpark SparkSession in local mode.

    WHY local mode: We demonstrate the same PySpark API that runs on
    Databricks / EMR in production, but without requiring a cluster.
    The code is portable — change SPARK_MASTER to a cluster URL and
    it scales horizontally with zero code changes.

    Returns:
        SparkSession: configured Spark session.
    """
    spark = (
        SparkSession.builder
        .appName(Config.SPARK_APP_NAME)
        .master(Config.SPARK_MASTER)
        .config("spark.driver.memory", Config.SPARK_DRIVER_MEMORY)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    # Reduce Spark's verbose logging noise
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession initialized — master=%s, memory=%s",
                Config.SPARK_MASTER, Config.SPARK_DRIVER_MEMORY)
    return spark


def load_csv_with_schema(
    spark: SparkSession,
    file_path: str,
    schema: Optional[StructType] = None,
    header: bool = True,
    infer_schema: bool = False,
) -> DataFrame:
    """
    Load a CSV file into a PySpark DataFrame with optional schema enforcement.

    Args:
        spark: Active SparkSession.
        file_path: Absolute path to the CSV file.
        schema: PySpark StructType to enforce. If None, Spark will infer.
        header: Whether the CSV has a header row.
        infer_schema: If True and schema is None, Spark infers column types.

    Returns:
        DataFrame: Raw PySpark DataFrame with schema applied.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    from pathlib import Path
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    reader = spark.read.option("header", str(header).lower())

    if schema is not None:
        reader = reader.schema(schema)
    elif infer_schema:
        reader = reader.option("inferSchema", "true")

    df = reader.csv(str(file_path))
    row_count = df.count()
    col_count = len(df.columns)
    logger.info("Loaded %s → %d rows × %d columns", file_path, row_count, col_count)
    return df


def validate_and_clean(
    df: DataFrame,
    target_col: str,
    id_col: Optional[str] = None,
    dataset_name: str = "dataset",
) -> DataFrame:
    """
    Apply data quality checks and cleaning rules to a raw DataFrame.

    Steps (in order):
        1. Drop rows where the target column is null
           (rationale: can't train a supervised model without labels)
        2. Remove exact-duplicate rows
        3. Impute remaining numeric nulls with column medians
        4. Log row counts after each step

    Args:
        df: Raw PySpark DataFrame.
        target_col: Name of the target/label column.
        id_col: Optional ID column to exclude from dedup logic.
        dataset_name: Human-readable name for logging.

    Returns:
        DataFrame: Cleaned DataFrame ready for feature engineering.
    """
    initial_count = df.count()
    logger.info("[%s] Starting validation — %d rows", dataset_name, initial_count)

    # Step 1: Drop rows with null target
    df = df.filter(F.col(target_col).isNotNull())
    after_target_filter = df.count()
    dropped_target = initial_count - after_target_filter
    logger.info("[%s] Dropped %d rows with null target '%s' → %d rows remain",
                dataset_name, dropped_target, target_col, after_target_filter)

    # Step 2: Remove duplicates
    if id_col and id_col in df.columns:
        # If there's an ID column, drop it before dedup to avoid false uniqueness
        dedup_cols = [c for c in df.columns if c != id_col]
        df = df.dropDuplicates(dedup_cols)
    else:
        df = df.dropDuplicates()
    after_dedup = df.count()
    dropped_dupes = after_target_filter - after_dedup
    logger.info("[%s] Removed %d duplicate rows → %d rows remain",
                dataset_name, dropped_dupes, after_dedup)

    # Step 3: Impute numeric nulls with column medians
    numeric_cols = [
        field.name for field in df.schema.fields
        if str(field.dataType) in ("IntegerType()", "DoubleType()", "FloatType()", "LongType()")
        and field.name != target_col
    ]

    if numeric_cols:
        # Compute medians in one pass using approxQuantile
        medians = {}
        for col_name in numeric_cols:
            null_count = df.filter(F.col(col_name).isNull()).count()
            if null_count > 0:
                median_val = df.approxQuantile(col_name, [0.5], 0.01)
                if median_val:
                    medians[col_name] = median_val[0]
                    logger.info("[%s] Imputing %d nulls in '%s' with median=%.4f",
                                dataset_name, null_count, col_name, median_val[0])

        if medians:
            df = df.fillna(medians)

    final_count = df.count()
    logger.info("[%s] Validation complete — %d → %d rows (%.1f%% retained)",
                dataset_name, initial_count, final_count,
                100 * final_count / max(initial_count, 1))
    return df


def ingest_gmsc(spark: SparkSession) -> DataFrame:
    """
    Load and validate the Give Me Some Credit (GMSC) dataset.

    This is the PROTOTYPING dataset (150K rows). We use it to build
    and test the entire pipeline before scaling to LendingClub.

    Args:
        spark: Active SparkSession.

    Returns:
        DataFrame: Cleaned GMSC DataFrame with enforced schema.
    """
    logger.info("=" * 60)
    logger.info("INGESTING: Give Me Some Credit (Prototyping Dataset)")
    logger.info("=" * 60)

    schema = get_gmsc_schema()
    df = load_csv_with_schema(
        spark,
        str(Config.GMSC_TRAIN_FILE),
        schema=schema,
        header=True,
    )

    # Rename the unnamed index column
    if "row_id" in df.columns:
        df = df.withColumnRenamed("row_id", "loan_id")

    df = validate_and_clean(
        df,
        target_col=Config.GMSC_TARGET_COL,
        id_col="loan_id",
        dataset_name="GMSC",
    )

    return df


def ingest_lending_club(spark: SparkSession) -> DataFrame:
    """
    Load and validate the LendingClub dataset.

    This is the PRODUCTION dataset (2.2M rows). We use it after the
    pipeline is validated on GMSC.

    Special handling:
        - int_rate and revol_util arrive as strings with '%' suffix
        - issue_d arrives as "Mon-YYYY" string
        - loan_status is filtered to only Fully Paid / Charged Off
          (other statuses like Current are excluded — can't label them)

    Args:
        spark: Active SparkSession.

    Returns:
        DataFrame: Cleaned LendingClub DataFrame with binary target.
    """
    logger.info("=" * 60)
    logger.info("INGESTING: LendingClub (Production Dataset)")
    logger.info("=" * 60)

    # Read with inferred schema first (LC has 150+ columns, we'll select key ones)
    df = load_csv_with_schema(
        spark,
        str(Config.LENDING_CLUB_FILE),
        schema=None,
        header=True,
        infer_schema=True,
    )

    # Select only the columns we need
    available_cols = [c for c in Config.LC_KEY_FEATURES if c in df.columns]
    missing_cols = [c for c in Config.LC_KEY_FEATURES if c not in df.columns]
    if missing_cols:
        logger.warning("Missing expected columns in LendingClub data: %s", missing_cols)

    # Always include loan_status for target encoding
    if Config.LC_TARGET_COL not in available_cols:
        available_cols.append(Config.LC_TARGET_COL)
    df = df.select(available_cols)

    # Filter to labeled rows only (Fully Paid or Charged Off)
    valid_labels = Config.LC_TARGET_POSITIVE_LABELS + Config.LC_TARGET_NEGATIVE_LABELS
    df = df.filter(F.col(Config.LC_TARGET_COL).isin(valid_labels))
    logger.info("Filtered to labeled loans (Fully Paid / Charged Off): %d rows", df.count())

    # Encode target: Charged Off / Default = 1, Fully Paid = 0
    df = df.withColumn(
        "is_default",
        F.when(F.col(Config.LC_TARGET_COL).isin(Config.LC_TARGET_POSITIVE_LABELS), 1)
        .otherwise(0)
    )

    # Clean percentage columns: "13.56%" → 13.56
    if "int_rate" in df.columns:
        df = df.withColumn(
            "int_rate",
            F.regexp_replace(F.col("int_rate"), "%", "").cast("double")
        )
    if "revol_util" in df.columns:
        df = df.withColumn(
            "revol_util",
            F.regexp_replace(F.col("revol_util"), "%", "").cast("double")
        )

    df = validate_and_clean(
        df,
        target_col="is_default",
        dataset_name="LendingClub",
    )

    return df
