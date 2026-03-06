"""
db_writer.py — Star Schema Database Writer
============================================
WHY THIS EXISTS:
    The output of the data pipeline needs to land somewhere queryable.
    In production, that's a PostgreSQL (or Snowflake/BigQuery) data
    warehouse organized as a star schema. BI teams, risk analysts, and
    regulators query these tables — they never touch raw CSVs.

    This module supports TWO modes:
    1. PostgreSQL mode  → writes fact + dimension tables via SQLAlchemy
    2. Parquet fallback  → writes the same logical tables as .parquet files
                          (zero-install, runs on any machine)

WHAT IT WRITES:
    Star schema tables:
    - fact_loans:      loan-level metrics (amount, rate, PD, EL, default flag)
    - dim_borrower:    borrower attributes (income, DTI, delinquency)
    - dim_loan_terms:  loan characteristics (grade, term, purpose)
    - dim_geography:   geographic data (state)
    - dim_time:        temporal data (issue date, year, quarter, vintage)
    - dim_macro:       macroeconomic context at origination

HOW IT WORKS:
    If Config.USE_POSTGRES is True, creates tables via SQLAlchemy.
    Otherwise, writes Parquet files to data/processed/feature_store/.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.config import Config

logger = logging.getLogger(__name__)


def write_feature_store(
    df: pd.DataFrame,
    dataset_name: str = "gmsc",
    output_format: Optional[str] = None,
) -> None:
    """
    Write the feature-engineered DataFrame to persistent storage.

    This is the main entry point. It decides between PostgreSQL and
    Parquet based on Config.USE_POSTGRES (or the output_format override).

    Args:
        df: Feature-engineered pandas DataFrame.
        dataset_name: 'gmsc' or 'lc' — determines output filenames.
        output_format: Override format ('postgres' or 'parquet'). If None,
                       uses Config.USE_POSTGRES to decide.
    """
    use_pg = (output_format == "postgres") if output_format else Config.USE_POSTGRES

    if use_pg:
        _write_to_postgres(df, dataset_name)
    else:
        _write_to_parquet(df, dataset_name)


def _write_to_parquet(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Write the complete feature set as a single Parquet file,
    plus split it into star-schema dimension tables.

    Parquet format preserves types, compresses well, and is readable
    by PySpark, pandas, and every BI tool.

    Args:
        df: Feature-engineered DataFrame.
        dataset_name: 'gmsc' or 'lc'.
    """
    output_dir = Config.FEATURE_STORE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Write full feature set ---- #
    full_path = output_dir / f"{dataset_name}_features.parquet"
    df.to_parquet(str(full_path), index=False, engine="pyarrow")
    logger.info("Wrote full feature set → %s (%d rows × %d cols)",
                full_path, len(df), len(df.columns))

    # ---- Write star schema tables ---- #
    _write_star_schema_parquet(df, dataset_name, output_dir)


def _write_star_schema_parquet(
    df: pd.DataFrame, dataset_name: str, output_dir: Path
) -> None:
    """
    Split the flat feature DataFrame into star-schema dimension tables
    and write each as a separate Parquet file.

    This mirrors the PostgreSQL star schema but in file-based format.

    Args:
        df: Full feature DataFrame.
        dataset_name: 'gmsc' or 'lc'.
        output_dir: Directory to write Parquet files to.
    """
    # Define column mappings for each dimension
    # (only include columns that actually exist in the DataFrame)
    def _select_existing(columns):
        return [c for c in columns if c in df.columns]

    # -- Fact table: core loan metrics -- #
    fact_cols = _select_existing([
        "loan_id", "loan_amnt", "funded_amnt", "int_rate", "installment",
        "is_default", "SeriousDlqin2yrs",
    ])
    if fact_cols:
        fact_df = df[fact_cols].copy()
        fact_df.to_parquet(str(output_dir / f"{dataset_name}_fact_loans.parquet"),
                           index=False)
        logger.info("  → fact_loans: %d rows × %d cols", len(fact_df), len(fact_cols))

    # -- Dim: Borrower -- #
    borrower_cols = _select_existing([
        "loan_id", "annual_inc", "dti", "delinq_2yrs", "revol_util",
        "open_acc", "pub_rec", "age", "MonthlyIncome", "DebtRatio",
        "NumberOfDependents", "RevolvingUtilizationOfUnsecuredLines",
        "emp_length_years",
    ])
    if borrower_cols:
        borrower_df = df[borrower_cols].copy()
        borrower_df.to_parquet(str(output_dir / f"{dataset_name}_dim_borrower.parquet"),
                               index=False)
        logger.info("  → dim_borrower: %d rows × %d cols",
                     len(borrower_df), len(borrower_cols))

    # -- Dim: Loan Terms -- #
    terms_cols = _select_existing([
        "loan_id", "grade", "sub_grade", "term", "purpose",
        "verification_status", "grade_numeric",
    ])
    if terms_cols:
        terms_df = df[terms_cols].copy()
        terms_df.to_parquet(str(output_dir / f"{dataset_name}_dim_loan_terms.parquet"),
                            index=False)
        logger.info("  → dim_loan_terms: %d rows × %d cols",
                     len(terms_df), len(terms_cols))

    # -- Dim: Geography -- #
    geo_cols = _select_existing(["loan_id", "addr_state"])
    if geo_cols:
        geo_df = df[geo_cols].copy()
        geo_df.to_parquet(str(output_dir / f"{dataset_name}_dim_geography.parquet"),
                           index=False)
        logger.info("  → dim_geography: %d rows × %d cols", len(geo_df), len(geo_cols))

    # -- Dim: Time -- #
    time_cols = _select_existing([
        "loan_id", "issue_date", "issue_year", "issue_quarter",
        "loan_vintage_months",
    ])
    if time_cols:
        time_df = df[time_cols].copy()
        time_df.to_parquet(str(output_dir / f"{dataset_name}_dim_time.parquet"),
                            index=False)
        logger.info("  → dim_time: %d rows × %d cols", len(time_df), len(time_cols))

    # -- Dim: Macro -- #
    macro_cols = _select_existing([
        "loan_id", "unemployment_rate", "gdp", "fed_funds_rate",
        "credit_card_delinq_rate",
    ])
    if macro_cols:
        macro_df = df[macro_cols].copy()
        macro_df.to_parquet(str(output_dir / f"{dataset_name}_dim_macro.parquet"),
                             index=False)
        logger.info("  → dim_macro: %d rows × %d cols", len(macro_df), len(macro_cols))


def _write_to_postgres(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Write the feature set to PostgreSQL as star-schema tables.

    Uses SQLAlchemy with 'replace' mode (truncate + reload).
    Tables are prefixed with dataset_name for namespace separation.

    Requires:
        - PostgreSQL running and accessible
        - Config.POSTGRES_* credentials set
        - psycopg2-binary installed

    Args:
        df: Feature-engineered DataFrame.
        dataset_name: 'gmsc' or 'lc'.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError:
        logger.error("sqlalchemy not installed — install with: pip install sqlalchemy")
        logger.info("Falling back to Parquet output")
        _write_to_parquet(df, dataset_name)
        return

    try:
        engine = create_engine(Config.get_postgres_uri())
        logger.info("Connected to PostgreSQL: %s", Config.POSTGRES_DB)

        # Write full feature table
        table_name = f"{dataset_name}_features"
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info("Wrote table '%s' → %d rows", table_name, len(df))

        # Also write star schema tables
        _write_star_schema_postgres(df, dataset_name, engine)

        engine.dispose()
        logger.info("PostgreSQL write complete for dataset '%s'", dataset_name)

    except Exception as e:
        logger.error("PostgreSQL write failed: %s", e)
        logger.info("Falling back to Parquet output")
        _write_to_parquet(df, dataset_name)


def _write_star_schema_postgres(df, dataset_name, engine):
    """
    Write star schema dimension tables to PostgreSQL.

    Args:
        df: Full feature DataFrame.
        dataset_name: Table name prefix.
        engine: SQLAlchemy engine.
    """
    def _select_existing(columns):
        return [c for c in columns if c in df.columns]

    table_mappings = {
        "fact_loans": ["loan_id", "loan_amnt", "int_rate", "installment",
                       "is_default", "SeriousDlqin2yrs"],
        "dim_borrower": ["loan_id", "annual_inc", "dti", "delinq_2yrs",
                         "revol_util", "open_acc", "pub_rec", "age",
                         "MonthlyIncome", "DebtRatio"],
        "dim_loan_terms": ["loan_id", "grade", "sub_grade", "term", "purpose"],
        "dim_geography": ["loan_id", "addr_state"],
        "dim_time": ["loan_id", "issue_year", "issue_quarter", "loan_vintage_months"],
        "dim_macro": ["loan_id", "unemployment_rate", "gdp", "fed_funds_rate",
                      "credit_card_delinq_rate"],
    }

    for table_suffix, columns in table_mappings.items():
        existing_cols = _select_existing(columns)
        if existing_cols:
            table_name = f"{dataset_name}_{table_suffix}"
            subset = df[existing_cols].copy()
            subset.to_sql(table_name, engine, if_exists="replace", index=False)
            logger.info("  → %s: %d rows × %d cols",
                        table_name, len(subset), len(existing_cols))
