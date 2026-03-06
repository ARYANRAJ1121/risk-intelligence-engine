"""
feature_engineer.py — Credit Risk Feature Engineering Module
==============================================================
WHY THIS EXISTS:
    Raw loan data contains fields like 'MonthlyIncome' and 'DebtRatio',
    but not the derived risk signals that a PD model needs. Feature
    engineering transforms raw attributes into meaningful predictors:

    - Delinquency flags (30/60/90 DPD) → standard Basel III risk buckets
    - Log-transformed skewed columns → improves model convergence
    - Credit utilization → a key driver of default probability
    - Loan vintage → older loans have different risk profiles

    In a consulting engagement, the feature engineering logic is documented
    in a 'Feature Specification' deliverable that gets signed off by the
    client's model validation team. Every feature must be explainable.

WHAT IT RETURNS:
    A PySpark (or pandas) DataFrame with the original columns PLUS
    all derived features, ready for model training.

HOW IT WORKS:
    Each feature is computed by a dedicated function with docstring
    explaining its business meaning. The main `engineer_features()`
    function orchestrates them in sequence.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ================================================================== #
#  GMSC FEATURE ENGINEERING
# ================================================================== #

def engineer_gmsc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the Give Me Some Credit dataset.

    This function transforms raw GMSC columns into model-ready features.
    Designed for the prototyping phase (150K rows → pandas is fine here).

    Steps:
        1. Cap extreme outliers in delinquency count columns
        2. Create binary delinquency flags (30/60/90 DPD)
        3. Create credit utilization bucketed feature
        4. Log-transform skewed numeric columns
        5. Handle remaining edge cases (negative ages, etc.)

    Args:
        df: pandas DataFrame with raw GMSC columns.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    logger.info("Starting GMSC feature engineering — %d rows × %d cols",
                len(df), len(df.columns))
    df = df.copy()

    # ---- Step 1: Cap extreme outliers ---- #
    df = _cap_delinquency_outliers(df)

    # ---- Step 2: Binary delinquency flags ---- #
    df = _create_delinquency_flags_gmsc(df)

    # ---- Step 3: Credit utilization buckets ---- #
    df = _create_utilization_buckets(df, col="RevolvingUtilizationOfUnsecuredLines")

    # ---- Step 4: Log-transform skewed columns ---- #
    skewed_cols = ["MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
    df = _log_transform(df, columns=skewed_cols)

    # ---- Step 5: Age and edge case cleanup ---- #
    df = _clean_age(df)

    # ---- Step 6: Total delinquency score ---- #
    df = _create_total_delinquency_score_gmsc(df)

    logger.info("GMSC feature engineering complete — %d rows × %d cols",
                len(df), len(df.columns))
    return df


def engineer_lc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the LendingClub dataset.

    Steps:
        1. Parse issue_d into datetime and compute loan vintage
        2. Clean and encode emp_length into numeric
        3. Create binary delinquency flags
        4. Create DTI and utilization buckets
        5. Log-transform skewed columns
        6. Extract origination quarter for FRED macro join

    Args:
        df: pandas DataFrame with raw LendingClub columns.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    logger.info("Starting LendingClub feature engineering — %d rows × %d cols",
                len(df), len(df.columns))
    df = df.copy()

    # ---- Parse issue_d for loan vintage and time dimension ---- #
    df = _parse_issue_date(df)

    # ---- Encode emp_length to numeric ---- #
    df = _encode_emp_length(df)

    # ---- Binary delinquency flags ---- #
    df = _create_delinquency_flags_lc(df)

    # ---- DTI buckets ---- #
    df = _create_dti_buckets(df)

    # ---- Credit utilization buckets ---- #
    if "revol_util" in df.columns:
        df = _create_utilization_buckets(df, col="revol_util")

    # ---- Log-transform skewed columns ---- #
    log_cols = ["annual_inc", "revol_bal", "loan_amnt"]
    existing_log_cols = [c for c in log_cols if c in df.columns]
    df = _log_transform(df, columns=existing_log_cols)

    # ---- Grade encoding (ordinal) ---- #
    df = _encode_grade(df)

    logger.info("LendingClub feature engineering complete — %d rows × %d cols",
                len(df), len(df.columns))
    return df


# ================================================================== #
#  PRIVATE HELPER FUNCTIONS
# ================================================================== #

def _cap_delinquency_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap delinquency count columns at 96 and 98 → replace with median.

    WHY: In the GMSC dataset, values of 96 and 98 in delinquency columns
    are known data artifacts (likely encoding 'unknown' or 'not applicable').
    Treating them as real counts would inflate delinquency features and
    introduce noise into the PD model.

    Args:
        df: DataFrame with GMSC delinquency columns.

    Returns:
        pd.DataFrame: DataFrame with outliers replaced by column medians.
    """
    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    for col in delinq_cols:
        if col in df.columns:
            # Cast to float64 to avoid FutureWarning when assigning median
            df[col] = df[col].astype("float64")
            mask = df[col].isin([96, 98])
            replacement = df.loc[~mask, col].median()
            replaced_count = mask.sum()
            if replaced_count > 0:
                df.loc[mask, col] = replacement
                logger.info("Capped %d outlier values in '%s' → median=%.1f",
                            replaced_count, col, replacement)
    return df


def _create_delinquency_flags_gmsc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary delinquency flags from GMSC count columns.

    Business meaning:
        - flag_30dpd = 1 if borrower was 30+ days past due at least once
        - flag_60dpd = 1 if borrower was 60+ days past due at least once
        - flag_90dpd = 1 if borrower was 90+ days past due at least once

    These flags are standard Basel III risk bucketing — regulators use
    them to classify accounts into 'performing' vs 'non-performing'.

    Args:
        df: DataFrame with GMSC delinquency count columns.

    Returns:
        pd.DataFrame: DataFrame with three new binary flag columns.
    """
    flag_map = {
        "flag_30dpd": "NumberOfTime30-59DaysPastDueNotWorse",
        "flag_60dpd": "NumberOfTime60-89DaysPastDueNotWorse",
        "flag_90dpd": "NumberOfTimes90DaysLate",
    }
    for flag_name, source_col in flag_map.items():
        if source_col in df.columns:
            df[flag_name] = (df[source_col] >= 1).astype(int)
            positive_pct = df[flag_name].mean() * 100
            logger.info("Created '%s' — %.1f%% positive", flag_name, positive_pct)
    return df


def _create_delinquency_flags_lc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary delinquency flag from LendingClub's delinq_2yrs column.

    delinq_2yrs = number of 30+ DPD incidences in past 2 years.

    Args:
        df: DataFrame with LendingClub columns.

    Returns:
        pd.DataFrame: DataFrame with new flag_delinq column.
    """
    if "delinq_2yrs" in df.columns:
        df["flag_delinq"] = (df["delinq_2yrs"] >= 1).astype(int)
        logger.info("Created 'flag_delinq' — %.1f%% positive",
                     df["flag_delinq"].mean() * 100)
    return df


def _create_utilization_buckets(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Bucket credit utilization into risk categories.

    Business meaning:
        - Low (0–30%): healthy credit usage
        - Medium (30–60%): moderate risk
        - High (60–90%): elevated risk
        - Very High (90%+): near-maxed accounts — strongest default predictor

    Args:
        df: DataFrame with a utilization column (0–100 scale or 0–1 scale).
        col: Name of the utilization column.

    Returns:
        pd.DataFrame: DataFrame with new '{col}_bucket' column.
    """
    if col not in df.columns:
        return df

    series = df[col].copy()
    # Normalize to 0–100 scale if needed
    if series.median() < 2:  # likely 0–1 scale
        series = series * 100

    bucket_col = f"{col}_bucket"
    df[bucket_col] = pd.cut(
        series,
        bins=[-np.inf, 30, 60, 90, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )
    logger.info("Created utilization buckets in '%s'", bucket_col)
    return df


def _create_dti_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket Debt-to-Income ratio into risk categories.

    Business meaning:
        - Low DTI (<20%): conservative borrower
        - Medium (20–35%): typical range
        - High (35–50%): stressed
        - Very High (>50%): severe debt burden — highest default risk

    Args:
        df: DataFrame with 'dti' column.

    Returns:
        pd.DataFrame: DataFrame with new 'dti_bucket' column.
    """
    if "dti" not in df.columns:
        return df

    df["dti_bucket"] = pd.cut(
        df["dti"],
        bins=[-np.inf, 20, 35, 50, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )
    logger.info("Created DTI buckets — distribution:\n%s",
                df["dti_bucket"].value_counts().to_string())
    return df


def _log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply log1p transformation to skewed numeric columns.

    WHY: Columns like MonthlyIncome and RevolvingUtilization are heavily
    right-skewed. Tree-based models (XGBoost) are somewhat robust to
    skewness, but log-transforming improves logistic regression performance
    and makes SHAP plots more interpretable.

    Uses np.log1p (log(1 + x)) to handle zeros safely.

    Args:
        df: DataFrame with numeric columns.
        columns: List of column names to log-transform.

    Returns:
        pd.DataFrame: DataFrame with new 'log_{col}' columns added.
    """
    for col in columns:
        if col in df.columns:
            new_col = f"log_{col}"
            # Clip negative values to 0 before log
            df[new_col] = np.log1p(df[col].clip(lower=0))
            logger.info("Created '%s' (log1p of '%s')", new_col, col)
    return df


def _clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the age column: remove impossible values (0 or negative).

    WHY: A small number of rows in GMSC have age=0, which is clearly
    a data quality issue. We cap minimum age at 18 (legal lending age).

    Args:
        df: DataFrame with 'age' column.

    Returns:
        pd.DataFrame: DataFrame with cleaned age values.
    """
    if "age" in df.columns:
        invalid_mask = df["age"] < 18
        if invalid_mask.sum() > 0:
            logger.info("Capping %d rows with age < 18 to 18", invalid_mask.sum())
            df.loc[invalid_mask, "age"] = 18
    return df


def _create_total_delinquency_score_gmsc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite delinquency score summing all past-due events.

    Business meaning: aggregates all delinquency severity levels into
    one 'total risk' signal. Higher score = more troubled borrower.

    Args:
        df: DataFrame with GMSC delinquency count columns.

    Returns:
        pd.DataFrame: DataFrame with new 'total_delinquency_score' column.
    """
    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    available = [c for c in delinq_cols if c in df.columns]
    if available:
        df["total_delinquency_score"] = df[available].sum(axis=1)
        logger.info("Created 'total_delinquency_score' — mean=%.2f, max=%.0f",
                     df["total_delinquency_score"].mean(),
                     df["total_delinquency_score"].max())
    return df


def _parse_issue_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse LendingClub's issue_d column ('Dec-2011') into datetime components.

    Derived columns:
        - issue_date: datetime object
        - issue_year: integer
        - issue_quarter: integer (1–4)
        - loan_vintage_months: months since origination (from most recent date)

    WHY loan_vintage matters: newer loans haven't had time to default,
    creating survivorship bias. Vintage normalization is standard practice
    in credit risk analytics.

    Args:
        df: DataFrame with 'issue_d' column.

    Returns:
        pd.DataFrame: DataFrame with parsed date components.
    """
    if "issue_d" not in df.columns:
        return df

    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = df["issue_date"].dt.year
    df["issue_quarter"] = df["issue_date"].dt.quarter

    # Loan vintage in months (from most recent issue date in dataset)
    max_date = df["issue_date"].max()
    if pd.notna(max_date):
        df["loan_vintage_months"] = (
            (max_date.year - df["issue_date"].dt.year) * 12
            + (max_date.month - df["issue_date"].dt.month)
        )
    logger.info("Parsed issue_d → year range: %s–%s",
                df["issue_year"].min(), df["issue_year"].max())
    return df


def _encode_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode LendingClub's emp_length from string to numeric years.

    Mapping:
        '< 1 year' → 0, '1 year' → 1, '2 years' → 2, ..., '10+ years' → 10
        null / 'n/a' → -1

    WHY: Employment length is a key feature in credit scoring.
    Longer employment → more stable income → lower default risk.

    Args:
        df: DataFrame with 'emp_length' column.

    Returns:
        pd.DataFrame: DataFrame with 'emp_length_years' numeric column.
    """
    if "emp_length" not in df.columns:
        return df

    mapping = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10,
    }
    df["emp_length_years"] = df["emp_length"].map(mapping).fillna(-1).astype(int)
    logger.info("Encoded emp_length → emp_length_years (mean=%.1f)",
                df.loc[df["emp_length_years"] >= 0, "emp_length_years"].mean())
    return df


def _encode_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode LendingClub's letter grade into ordinal numeric.

    Mapping: A=1, B=2, C=3, D=4, E=5, F=6, G=7
    Higher number = higher risk (consistent with Basel convention).

    Args:
        df: DataFrame with 'grade' column.

    Returns:
        pd.DataFrame: DataFrame with 'grade_numeric' column.
    """
    if "grade" not in df.columns:
        return df

    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_numeric"] = df["grade"].map(grade_map)
    logger.info("Encoded grade → grade_numeric")
    return df
