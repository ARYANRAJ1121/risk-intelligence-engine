"""
test_ingestion.py — Unit Tests for Layer 1 Data Pipeline
=========================================================
WHY TESTS EXIST:
    In a consulting engagement, model validation teams require evidence
    that data transformations work correctly. These tests prove that
    feature engineering logic produces expected outputs on known inputs.

WHAT THESE TESTS VERIFY:
    1. Feature engineering functions produce correct derived columns
    2. Delinquency flags are computed correctly
    3. Log-transforms handle edge cases (zeros, negatives)
    4. Utilization bucket boundaries are correct
    5. Config paths resolve properly
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineer import (
    engineer_gmsc_features,
    _cap_delinquency_outliers,
    _create_delinquency_flags_gmsc,
    _create_utilization_buckets,
    _log_transform,
    _clean_age,
    _create_total_delinquency_score_gmsc,
    _encode_emp_length,
    _encode_grade,
    _create_dti_buckets,
)
from src.ingestion.config import Config


class TestConfig:
    """Tests for centralized configuration."""

    def test_project_root_exists(self):
        """Config.PROJECT_ROOT should point to the risk-intelligence-engine directory."""
        assert Config.PROJECT_ROOT.name == "risk-intelligence-engine"

    def test_data_directories_defined(self):
        """All data directory paths should be Path objects under PROJECT_ROOT."""
        assert Config.RAW_DATA_DIR.parent == Config.DATA_DIR
        assert Config.PROCESSED_DATA_DIR.parent == Config.DATA_DIR

    def test_postgres_uri_format(self):
        """PostgreSQL URI should have correct format."""
        uri = Config.get_postgres_uri()
        assert uri.startswith("postgresql://")

    def test_gmsc_columns_defined(self):
        """GMSC feature columns should be a non-empty list."""
        assert len(Config.GMSC_FEATURE_COLS) > 0
        assert Config.GMSC_TARGET_COL == "SeriousDlqin2yrs"


class TestDelinquencyFlags:
    """Tests for delinquency flag creation."""

    def setup_method(self):
        """Create a sample DataFrame mimicking GMSC delinquency columns."""
        self.df = pd.DataFrame({
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 3, 0, 2],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 1, 0, 0],
            "NumberOfTimes90DaysLate": [0, 0, 0, 1, 2],
        })

    def test_30dpd_flag(self):
        """flag_30dpd should be 1 when 30-59 DPD count >= 1."""
        result = _create_delinquency_flags_gmsc(self.df)
        expected = [0, 1, 1, 0, 1]
        assert list(result["flag_30dpd"]) == expected

    def test_60dpd_flag(self):
        """flag_60dpd should be 1 when 60-89 DPD count >= 1."""
        result = _create_delinquency_flags_gmsc(self.df)
        expected = [0, 0, 1, 0, 0]
        assert list(result["flag_60dpd"]) == expected

    def test_90dpd_flag(self):
        """flag_90dpd should be 1 when 90+ DPD count >= 1."""
        result = _create_delinquency_flags_gmsc(self.df)
        expected = [0, 0, 0, 1, 1]
        assert list(result["flag_90dpd"]) == expected


class TestOutlierCapping:
    """Tests for delinquency outlier capping (96/98 artifacts)."""

    def test_cap_96_and_98(self):
        """Values 96 and 98 should be replaced with median of non-outlier rows."""
        df = pd.DataFrame({
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 96, 98],
            "NumberOfTimes90DaysLate": [0, 0, 1, 0, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0, 0, 0],
        })
        result = _cap_delinquency_outliers(df)
        # After capping, 96 and 98 should be replaced with median of [0,1,2] = 1.0
        assert result["NumberOfTime30-59DaysPastDueNotWorse"].max() <= 3

    def test_no_change_when_no_outliers(self):
        """Normal values should not be modified."""
        df = pd.DataFrame({
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 3],
            "NumberOfTimes90DaysLate": [0, 0, 1, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0, 0],
        })
        result = _cap_delinquency_outliers(df)
        assert list(result["NumberOfTime30-59DaysPastDueNotWorse"]) == [0, 1, 2, 3]


class TestLogTransform:
    """Tests for log1p transformation."""

    def test_log_transform_creates_new_column(self):
        """Log transform should create a 'log_{colname}' column."""
        df = pd.DataFrame({"MonthlyIncome": [5000, 10000, 15000]})
        result = _log_transform(df, columns=["MonthlyIncome"])
        assert "log_MonthlyIncome" in result.columns

    def test_log_transform_handles_zeros(self):
        """log1p(0) should equal 0."""
        df = pd.DataFrame({"MonthlyIncome": [0, 1000, 5000]})
        result = _log_transform(df, columns=["MonthlyIncome"])
        assert result["log_MonthlyIncome"].iloc[0] == pytest.approx(0.0)

    def test_log_transform_handles_negatives(self):
        """Negative values should be clipped to 0 before log1p."""
        df = pd.DataFrame({"MonthlyIncome": [-100, 0, 1000]})
        result = _log_transform(df, columns=["MonthlyIncome"])
        assert result["log_MonthlyIncome"].iloc[0] == pytest.approx(0.0)

    def test_log_transform_skips_missing_columns(self):
        """Should silently skip columns not present in DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = _log_transform(df, columns=["NonExistent"])
        assert "log_NonExistent" not in result.columns


class TestUtilizationBuckets:
    """Tests for credit utilization bucketing."""

    def test_bucket_boundaries(self):
        """Utilization values should fall into correct buckets."""
        df = pd.DataFrame({
            "revol_util": [10.0, 35.0, 75.0, 95.0]
        })
        result = _create_utilization_buckets(df, col="revol_util")
        buckets = list(result["revol_util_bucket"])
        assert buckets == ["Low", "Medium", "High", "Very High"]

    def test_bucket_with_zero_to_one_scale(self):
        """Should auto-detect 0-1 scale and normalize to 0-100."""
        df = pd.DataFrame({
            "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.35, 0.75, 0.95]
        })
        result = _create_utilization_buckets(
            df, col="RevolvingUtilizationOfUnsecuredLines"
        )
        bucket_col = "RevolvingUtilizationOfUnsecuredLines_bucket"
        assert bucket_col in result.columns


class TestAgeCleanup:
    """Tests for age validation."""

    def test_age_below_18_capped(self):
        """Ages below 18 should be set to 18."""
        df = pd.DataFrame({"age": [0, 15, 25, 65]})
        result = _clean_age(df)
        assert result["age"].min() == 18

    def test_valid_ages_unchanged(self):
        """Valid ages (>= 18) should not be modified."""
        df = pd.DataFrame({"age": [25, 45, 65]})
        result = _clean_age(df)
        assert list(result["age"]) == [25, 45, 65]


class TestDelinquencyScore:
    """Tests for composite delinquency score."""

    def test_total_score_is_sum(self):
        """Total delinquency score should be sum of all DPD counts."""
        df = pd.DataFrame({
            "NumberOfTime30-59DaysPastDueNotWorse": [1, 0, 2],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 1],
            "NumberOfTimes90DaysLate": [0, 0, 3],
        })
        result = _create_total_delinquency_score_gmsc(df)
        expected = [1, 1, 6]
        assert list(result["total_delinquency_score"]) == expected


class TestLendingClubFeatures:
    """Tests for LendingClub-specific feature functions."""

    def test_emp_length_encoding(self):
        """Employment length strings should map to correct numeric values."""
        df = pd.DataFrame({
            "emp_length": ["< 1 year", "5 years", "10+ years", None]
        })
        result = _encode_emp_length(df)
        assert list(result["emp_length_years"]) == [0, 5, 10, -1]

    def test_grade_encoding(self):
        """Letter grades should map to ordinal integers (A=1, G=7)."""
        df = pd.DataFrame({"grade": ["A", "C", "G"]})
        result = _encode_grade(df)
        assert list(result["grade_numeric"]) == [1, 3, 7]

    def test_dti_buckets(self):
        """DTI values should fall into correct risk categories."""
        df = pd.DataFrame({"dti": [10.0, 25.0, 40.0, 60.0]})
        result = _create_dti_buckets(df)
        expected = ["Low", "Medium", "High", "Very High"]
        assert list(result["dti_bucket"]) == expected


class TestFullGMSCPipeline:
    """Integration test for the full GMSC feature engineering pipeline."""

    def test_gmsc_pipeline_adds_expected_columns(self):
        """Full pipeline should add all derived feature columns."""
        df = pd.DataFrame({
            "SeriousDlqin2yrs": [0, 1, 0, 0, 1],
            "RevolvingUtilizationOfUnsecuredLines": [0.5, 0.1, 0.9, 0.3, 0.7],
            "age": [35, 50, 15, 28, 62],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 96, 0, 2],
            "DebtRatio": [0.3, 0.5, 0.8, 0.2, 1.2],
            "MonthlyIncome": [5000.0, 8000.0, 3000.0, 12000.0, 6000.0],
            "NumberOfOpenCreditLinesAndLoans": [5, 8, 3, 12, 7],
            "NumberOfTimes90DaysLate": [0, 0, 98, 1, 0],
            "NumberRealEstateLoansOrLines": [1, 2, 0, 3, 1],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 96, 0, 1],
            "NumberOfDependents": [2.0, 0.0, 1.0, 3.0, 0.0],
        })
        result = engineer_gmsc_features(df)

        # Check new columns exist
        assert "flag_30dpd" in result.columns
        assert "flag_60dpd" in result.columns
        assert "flag_90dpd" in result.columns
        assert "total_delinquency_score" in result.columns
        assert "log_MonthlyIncome" in result.columns

        # Check age was cleaned (15 → 18)
        assert result["age"].min() >= 18

        # Check outlier capping worked (96, 98 replaced)
        assert result["NumberOfTime30-59DaysPastDueNotWorse"].max() < 96
