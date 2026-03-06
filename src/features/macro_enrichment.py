"""
macro_enrichment.py — FRED Macroeconomic Data Integration
==========================================================
WHY THIS EXISTS:
    A borrower's probability of default doesn't exist in a vacuum.
    A borrower with a 35% DTI in a 3.5% unemployment economy is
    fundamentally different from the same borrower in a 10% unemployment
    recession. Macroeconomic features capture this SYSTEMIC risk.

    Regulators (Fed, EBA, RBI) REQUIRE that stress testing models
    incorporate macroeconomic variables. Without this join, Layer 4
    (Stress Testing) is impossible.

WHAT IT DOES:
    1. Pulls quarterly macro data from FRED (or loads from cached CSV)
    2. Joins macro variables to loan data on origination quarter
    3. Each loan now carries the economic conditions at the time it was issued

HOW IT WORKS:
    - If a FRED API key is available, pulls fresh data via fredapi
    - Otherwise, falls back to a cached CSV in data/external/
    - Joins on (year, quarter) key derived from loan origination date
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.config import Config

logger = logging.getLogger(__name__)


def fetch_fred_data(
    api_key: Optional[str] = None,
    start_date: str = "2005-01-01",
    end_date: str = "2024-12-31",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch macroeconomic time series from FRED and aggregate to quarterly.

    Series pulled:
        - UNRATE:    US Unemployment Rate (monthly → quarterly average)
        - GDP:       US Gross Domestic Product (already quarterly)
        - FEDFUNDS:  Federal Funds Rate (monthly → quarterly average)
        - DRCCLACBS: Credit Card Delinquency Rate (quarterly)

    Args:
        api_key: FRED API key. If None, uses Config.FRED_API_KEY or cached file.
        start_date: Start date for data pull (YYYY-MM-DD).
        end_date: End date for data pull (YYYY-MM-DD).
        save_path: If provided, save the result as CSV for reproducibility.

    Returns:
        pd.DataFrame: Quarterly macro data with columns
            [year, quarter, unemployment_rate, gdp, fed_funds_rate, credit_card_delinq_rate]
    """
    key = api_key or Config.FRED_API_KEY

    # ---- Try FRED API first ---- #
    if key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=key)
            logger.info("Connected to FRED API — pulling macro series")

            series_data = {}
            for series_id, description in Config.FRED_SERIES.items():
                try:
                    data = fred.get_series(series_id, start_date, end_date)
                    series_data[series_id] = data
                    logger.info("  Pulled %s (%s): %d observations",
                                series_id, description, len(data))
                except Exception as e:
                    logger.warning("  Failed to pull %s: %s", series_id, e)

            if series_data:
                macro_df = _aggregate_to_quarterly(series_data)
                if save_path:
                    macro_df.to_csv(save_path, index=False)
                    logger.info("Saved FRED data to %s", save_path)
                return macro_df

        except ImportError:
            logger.warning("fredapi not installed — falling back to cached CSV")
        except Exception as e:
            logger.warning("FRED API error: %s — falling back to cached CSV", e)

    # ---- Fallback: load cached CSV ---- #
    return _load_cached_macro_data()


def _aggregate_to_quarterly(series_data: dict) -> pd.DataFrame:
    """
    Aggregate FRED monthly series to quarterly averages.

    Args:
        series_data: Dict mapping series ID to pandas Series.

    Returns:
        pd.DataFrame: Quarterly aggregated macro data.
    """
    combined = pd.DataFrame(series_data)
    combined.index = pd.to_datetime(combined.index)

    # Resample to quarterly, taking the mean
    quarterly = combined.resample("QE").mean()

    quarterly["year"] = quarterly.index.year
    quarterly["quarter"] = quarterly.index.quarter

    # Rename columns to human-readable names
    rename_map = {
        "UNRATE": "unemployment_rate",
        "GDP": "gdp",
        "FEDFUNDS": "fed_funds_rate",
        "DRCCLACBS": "credit_card_delinq_rate",
    }
    quarterly = quarterly.rename(columns=rename_map)

    quarterly = quarterly.reset_index(drop=True)
    logger.info("Aggregated to quarterly: %d quarters, columns=%s",
                len(quarterly), list(quarterly.columns))
    return quarterly


def _load_cached_macro_data() -> pd.DataFrame:
    """
    Load macro data from the cached CSV file.

    If the cached file doesn't exist, generate a synthetic macro dataset
    covering 2005–2024 based on historical US economic patterns.
    This ensures the pipeline can run end-to-end without a FRED API key.

    Returns:
        pd.DataFrame: Quarterly macro data.
    """
    cache_path = Config.MACRO_DATA_FILE

    if Path(cache_path).exists():
        df = pd.read_csv(cache_path)
        logger.info("Loaded cached macro data from %s — %d rows", cache_path, len(df))
        return df

    # Generate synthetic but realistic macro data for pipeline testing
    logger.info("No cached macro data found — generating synthetic macro data")
    return _generate_synthetic_macro_data(str(cache_path))


def _generate_synthetic_macro_data(save_path: str) -> pd.DataFrame:
    """
    Generate synthetic quarterly macro data based on real US economic patterns.

    This is NOT random data — it follows the actual macro trajectory:
        - 2007–2009: Great Recession (unemployment spikes, GDP drops)
        - 2010–2019: Recovery and expansion
        - 2020: COVID shock
        - 2021–2024: Recovery with elevated rates

    WHY synthetic: Allows the full pipeline to run without a FRED API key
    while still producing realistic stress test results in Layer 4.

    Args:
        save_path: Path to save the generated CSV.

    Returns:
        pd.DataFrame: Synthetic quarterly macro data.
    """
    import numpy as np

    quarters = []
    for year in range(2005, 2025):
        for q in range(1, 5):
            quarters.append({"year": year, "quarter": q})

    df = pd.DataFrame(quarters)

    # Unemployment Rate trajectory
    n = len(df)
    unemployment = np.full(n, 5.0)
    for i, row in df.iterrows():
        y, q = row["year"], row["quarter"]
        if 2007 <= y <= 2009:
            unemployment[i] = 5.0 + (y - 2007) * 2.5 + q * 0.3
        elif 2010 <= y <= 2019:
            unemployment[i] = max(3.5, 10.0 - (y - 2009) * 0.65)
        elif y == 2020:
            unemployment[i] = 3.5 + q * 2.5 if q <= 2 else 13.0 - q * 1.5
        elif y >= 2021:
            unemployment[i] = max(3.5, 6.5 - (y - 2020) * 0.8)
    unemployment = np.clip(unemployment, 3.0, 14.5)
    df["unemployment_rate"] = np.round(unemployment, 1)

    # GDP Growth (annualized quarterly)
    gdp = np.full(n, 2.5)
    for i, row in df.iterrows():
        y = row["year"]
        if y == 2008:
            gdp[i] = -2.0
        elif y == 2009:
            gdp[i] = -4.0
        elif 2010 <= y <= 2019:
            gdp[i] = 2.0 + np.random.uniform(-0.5, 0.5)
        elif y == 2020:
            gdp[i] = -5.0 if row["quarter"] <= 2 else 3.0
        elif y >= 2021:
            gdp[i] = 3.0 + np.random.uniform(-0.5, 0.5)
    df["gdp"] = np.round(gdp, 2)

    # Federal Funds Rate
    ff = np.full(n, 2.0)
    for i, row in df.iterrows():
        y = row["year"]
        if y <= 2007:
            ff[i] = 4.5
        elif 2008 <= y <= 2015:
            ff[i] = 0.25
        elif 2016 <= y <= 2018:
            ff[i] = 0.25 + (y - 2015) * 0.75
        elif y == 2019:
            ff[i] = 2.25
        elif y == 2020:
            ff[i] = 0.25
        elif y >= 2021:
            ff[i] = min(5.5, 0.25 + (y - 2020) * 1.5)
    df["fed_funds_rate"] = np.round(ff, 2)

    # Credit Card Delinquency Rate
    df["credit_card_delinq_rate"] = np.round(
        df["unemployment_rate"] * 0.4 + np.random.uniform(-0.3, 0.3, n), 2
    )

    # Save for reproducibility
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info("Generated synthetic macro data — %d quarters, saved to %s",
                len(df), save_path)
    return df


def join_macro_to_loans(
    loan_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join macroeconomic features to loan data on origination quarter.

    Each loan gets the unemployment rate, GDP, fed funds rate, and
    credit card delinquency rate from the quarter it was originated.

    For GMSC data (no origination date): assigns a default quarter
    based on the dataset's known timeframe (2005–2012).

    Args:
        loan_df: Loan DataFrame with 'issue_year' and 'issue_quarter' columns.
        macro_df: Quarterly macro DataFrame. If None, loads from cache/FRED.

    Returns:
        pd.DataFrame: Loan DataFrame enriched with macro columns.
    """
    if macro_df is None:
        macro_df = fetch_fred_data()

    # Check if loan data has time dimensions for joining
    if "issue_year" in loan_df.columns and "issue_quarter" in loan_df.columns:
        # Standard join on (year, quarter)
        loan_df = loan_df.merge(
            macro_df,
            left_on=["issue_year", "issue_quarter"],
            right_on=["year", "quarter"],
            how="left",
        )
        # Drop redundant join keys from macro side
        loan_df = loan_df.drop(columns=["year", "quarter"], errors="ignore")
        matched = loan_df["unemployment_rate"].notna().sum()
        total = len(loan_df)
        logger.info("Macro join complete: %d/%d loans matched (%.1f%%)",
                     matched, total, 100 * matched / max(total, 1))
    else:
        # GMSC dataset: no origination date — use dataset midpoint (2009 Q4)
        logger.warning("No issue_year/issue_quarter columns — assigning default "
                        "macro context (2009 Q4) for GMSC data")
        default_macro = macro_df[
            (macro_df["year"] == 2009) & (macro_df["quarter"] == 4)
        ]
        if not default_macro.empty:
            for col in ["unemployment_rate", "gdp", "fed_funds_rate",
                         "credit_card_delinq_rate"]:
                if col in default_macro.columns:
                    loan_df[col] = default_macro[col].values[0]

    return loan_df
