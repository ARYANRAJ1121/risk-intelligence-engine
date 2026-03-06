"""
schema.py — Centralized Schema Definitions for Raw Datasets
============================================================
WHY THIS EXISTS:
    In production credit risk pipelines, schema drift is a silent killer.
    A column name changes in a vendor feed, a float becomes a string,
    and suddenly your PD model is training on garbage. This module
    defines the EXPECTED schema for each dataset. The ingestion layer
    validates incoming data against these schemas BEFORE any processing.

    In a consulting engagement, the schema file is part of the
    methodology documentation — auditors review it to understand
    what data the model expects.

WHAT IT RETURNS:
    PySpark StructType objects and column-name lists for each dataset.
"""

from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, FloatType, DoubleType, StringType, LongType
)


def get_gmsc_schema() -> StructType:
    """
    Return the expected PySpark schema for the Give Me Some Credit dataset.

    Dataset: Kaggle 'GiveMeSomeCredit' (cs-training.csv / cs-test.csv)
    Rows: ~150K
    Target: SeriousDlqin2yrs (binary: 1 = serious delinquency within 2 years)

    Returns:
        StructType: PySpark schema with enforced types for every column.
    """
    return StructType([
        StructField("row_id", IntegerType(), True),
        StructField("SeriousDlqin2yrs", IntegerType(), True),
        StructField("RevolvingUtilizationOfUnsecuredLines", DoubleType(), True),
        StructField("age", IntegerType(), True),
        StructField("NumberOfTime30-59DaysPastDueNotWorse", IntegerType(), True),
        StructField("DebtRatio", DoubleType(), True),
        StructField("MonthlyIncome", DoubleType(), True),
        StructField("NumberOfOpenCreditLinesAndLoans", IntegerType(), True),
        StructField("NumberOfTimes90DaysLate", IntegerType(), True),
        StructField("NumberRealEstateLoansOrLines", IntegerType(), True),
        StructField("NumberOfTime60-89DaysPastDueNotWorse", IntegerType(), True),
        StructField("NumberOfDependents", DoubleType(), True),
    ])


def get_gmsc_column_names() -> list:
    """
    Return ordered column names for the GMSC dataset.
    Useful for CSV ingestion where header mapping is needed.
    """
    return [field.name for field in get_gmsc_schema().fields]


def get_lending_club_schema() -> StructType:
    """
    Return the expected PySpark schema for the LendingClub dataset.

    Dataset: Kaggle 'lending-club' (accepted_2007_to_2018Q4.csv)
    Rows: ~2.2M
    Target: loan_status → encoded as binary (Fully Paid = 0, Charged Off = 1)

    Note: LendingClub has 150+ columns. We define the schema only for
    the key columns we actually use in the model + features. Additional
    columns are read as StringType and selectively cast during feature
    engineering.

    Returns:
        StructType: PySpark schema for key LendingClub columns.
    """
    return StructType([
        StructField("id", LongType(), True),
        StructField("loan_amnt", DoubleType(), True),
        StructField("funded_amnt", DoubleType(), True),
        StructField("term", StringType(), True),
        StructField("int_rate", StringType(), True),       # comes as "13.56%" string
        StructField("installment", DoubleType(), True),
        StructField("grade", StringType(), True),
        StructField("sub_grade", StringType(), True),
        StructField("emp_length", StringType(), True),
        StructField("home_ownership", StringType(), True),
        StructField("annual_inc", DoubleType(), True),
        StructField("verification_status", StringType(), True),
        StructField("issue_d", StringType(), True),         # date as string "Dec-2011"
        StructField("loan_status", StringType(), True),
        StructField("purpose", StringType(), True),
        StructField("addr_state", StringType(), True),
        StructField("dti", DoubleType(), True),
        StructField("delinq_2yrs", DoubleType(), True),
        StructField("open_acc", DoubleType(), True),
        StructField("pub_rec", DoubleType(), True),
        StructField("revol_bal", DoubleType(), True),
        StructField("revol_util", StringType(), True),      # comes as "83.7%" string
        StructField("total_acc", DoubleType(), True),
    ])


def get_freddie_mac_schema() -> StructType:
    """
    Return the expected PySpark schema for Freddie Mac monthly performance data.

    Dataset: Freddie Mac Single-Family Loan-Level Dataset
    Granularity: one row per loan per month
    Key columns for LSTM: Monthly Rpt Prd, Current Loan Delinquency Status,
                          Current UPB, Loan Age

    Returns:
        StructType: PySpark schema for key Freddie Mac columns.
    """
    return StructType([
        StructField("loan_seq_num", StringType(), True),
        StructField("monthly_rpt_prd", StringType(), True),  # YYYYMM or MM/DD/YYYY
        StructField("current_upb", DoubleType(), True),
        StructField("current_loan_delinq_status", StringType(), True),  # 0,1,2,...,R
        StructField("loan_age", IntegerType(), True),
        StructField("remaining_months_to_maturity", IntegerType(), True),
        StructField("zero_balance_code", StringType(), True),
        StructField("zero_balance_eff_date", StringType(), True),
    ])
